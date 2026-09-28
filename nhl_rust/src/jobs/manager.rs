//! Manager Game sweeper (M5): periodic background job that advances league
//! state — season-start transition (setup/drafting → running), finalization of
//! manager games once the underlying NHL game is FINAL, and best-effort mode-1
//! roster seeding. Runs on an interval (env `MANAGER_SWEEP_INTERVAL_SECONDS`,
//! default 300); toggle `MANAGER_SWEEPER=1`.

use std::sync::atomic::{AtomicBool, Ordering};

use chrono::Utc;
use serde_json::{json, Value};

use crate::state::AppState;
use crate::supabase::write;
use crate::util::parse::{safe_int, str_value};

pub fn start(state: AppState) {
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async move {
        let interval = env_u64("MANAGER_SWEEP_INTERVAL_SECONDS", 300).max(30);
        loop {
            run_once(&state).await;
            tokio::time::sleep(std::time::Duration::from_secs(interval)).await;
        }
    });
}

fn env_u64(name: &str, default: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(default)
}

fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(default)
}

async fn run_once(state: &AppState) {
    let Some(sb) = state.sb.as_ref() else {
        return;
    };
    let leagues = sb
        .read("manager_leagues", "id,mode,status", None, None, None, 0)
        .await
        .unwrap_or_default();
    for league_row in leagues {
        let id = str_value(league_row.get("id"));
        let mode = str_value(league_row.get("mode"));
        let status = str_value(league_row.get("status"));
        if id.is_empty() {
            continue;
        }
        match status.as_str() {
            crate::manager::STATUS_SETUP | crate::manager::STATUS_DRAFTING => {
                // Season-start: once the league's first game date is reached,
                // flip to running; seed mode-1 rosters once.
                if season_started(sb, &id).await {
                    if mode == crate::manager::MODE_TRUE_ROSTERS {
                        let _ = crate::manager::standings::seed_true_rosters(state, &id).await;
                    }
                    let _ = write::update_rows(
                        sb,
                        "manager_leagues",
                        &[("id", id.as_str())],
                        &json!({"status": "running"}),
                    )
                    .await;
                }
            }
            crate::manager::STATUS_RUNNING => {
                finalize_recent_games(state, sb, &id).await;
            }
            _ => {}
        }
    }
}

/// True once any manager game in the league has a date <= today.
async fn season_started(sb: &crate::supabase::read::SbClient, league_id: &str) -> bool {
    let today = Utc::now().date_naive().to_string();
    let rows = sb
        .read(
            "manager_games",
            "date",
            Some(&crate::manager::league::eq_filter("league_id", league_id)),
            None,
            Some("date.asc"),
            1,
        )
        .await
        .unwrap_or_default();
    rows.first()
        .map(|r| str_value(r.get("date")) <= today)
        .unwrap_or(false)
}

/// Try to record + finalize recently-played games. Bounded per sweep.
async fn finalize_recent_games(state: &AppState, sb: &crate::supabase::read::SbClient, league_id: &str) {
    let today = Utc::now().date_naive().to_string();
    let batch = env_i64("MANAGER_SWEEP_BATCH", 50).max(1) as usize;
    let rows = sb
        .read(
            "manager_games",
            "nhl_game_id,status,date",
            Some(&crate::manager::league::eq_filter("league_id", league_id)),
            None,
            Some("date.asc,nhl_game_id.asc"),
            batch,
        )
        .await
        .unwrap_or_default();
    for g in rows {
        if str_value(g.get("status")) == "final" {
            continue;
        }
        if str_value(g.get("date")) > today {
            continue;
        }
        let Some(nhl_game_id) = safe_int(g.get("nhl_game_id")) else {
            continue;
        };
        // Record stat lines; only finalize once the NHL game is FINAL.
        let recorded = crate::manager::games::record_game_stats(state, nhl_game_id).await;
        match recorded {
            Ok((game_state, _)) => {
                let gs = game_state.to_uppercase();
                if gs.contains("FINAL") || gs == "OFF" {
                    let _ = crate::manager::games::finalize_game(state, league_id, nhl_game_id).await;
                }
            }
            Err(_) => {
                // NHL feed unreachable / not ready — retry next sweep.
            }
        }
    }
}

