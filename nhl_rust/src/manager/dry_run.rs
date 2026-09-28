//! Live end-to-end dry run of the Manager game engine (created for manual
//! verification against live Supabase + the NHL feed). Runs as the server
//! binary under `MANAGER_DRY_RUN=1` and exits (no HTTP listening).
//!
//! It drives the in-process functions directly (bypassing HTTP auth) to
//! exercise: league creation (real 2026/27 schedule snapshot), a full
//! 26-round snake draft with CPU autopick, a human-to-human trade, a mode-1
//! lineup set, and clean-up. Prints progress; any unexpected state is an error.

use serde_json::json;

use crate::error::ApiError;
use crate::manager::{draft, games, league, trades};
use crate::manager::MODE_DRAFT;
use crate::state::AppState;
use crate::util::parse::str_value;

const COMMISH: &str = "dry-run-commish";
const PARTNER: &str = "dry-run-partner";
const MY_TEAM: &str = "ANA";

fn log(msg: &str) {
    println!("[dry-run] {msg}");
}

/// Delete a league (cascades teams/games/rosters/offers). Used to clean up
/// test leagues when a dry run aborts early.
pub async fn delete_league(state: &AppState, league_id: &str) -> Result<(), ApiError> {
    let sb = state
        .sb
        .as_ref()
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let ok = crate::supabase::write::delete_rows(
        sb,
        "manager_leagues",
        &[("id", league_id)],
    )
    .await;
    if ok {
        log(&format!("deleted league {league_id}"));
        Ok(())
    } else {
        Err(ApiError::Internal("league delete failed".into()))
    }
}

/// Build the 12 F / 6 D / 2 G lineup for a team from its roster pool.
fn build_lineup(roster: &serde_json::Value) -> Result<Vec<(String, i64)>, ApiError> {
    let mut fwd: Vec<i64> = Vec::new();
    let mut def: Vec<i64> = Vec::new();
    let mut gol: Vec<i64> = Vec::new();
    for rec in roster
        .get("forwards")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        if let Some(pid) = rec.get("player_id").and_then(|v| v.as_i64()) {
            fwd.push(pid);
        }
    }
    for rec in roster
        .get("defense")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        if let Some(pid) = rec.get("player_id").and_then(|v| v.as_i64()) {
            def.push(pid);
        }
    }
    for rec in roster
        .get("goalies")
        .and_then(|v| v.as_array())
        .into_iter()
        .flatten()
    {
        if let Some(pid) = rec.get("player_id").and_then(|v| v.as_i64()) {
            gol.push(pid);
        }
    }
    if fwd.len() < 12 || def.len() < 6 || gol.len() < 2 {
        return Err(ApiError::Internal("insufficient roster for lineup".into()));
    }
    let mut out = Vec::new();
    for (i, pid) in fwd.iter().take(12).enumerate() {
        out.push((format!("F{}", i + 1), *pid));
    }
    for (i, pid) in def.iter().take(6).enumerate() {
        out.push((format!("D{}", i + 1), *pid));
    }
    for (i, pid) in gol.iter().take(2).enumerate() {
        out.push((format!("G{}", i + 1), *pid));
    }
    Ok(out)
}

pub async fn run(state: &AppState) -> Result<(), ApiError> {
    let sb = state
        .sb
        .as_ref()
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let mut league_id = String::new();
    // Always clean up the test league (even on error) so no rows linger.
    let result = run_inner(state, &mut league_id).await;
    if !league_id.is_empty() {
        let _ = crate::supabase::write::delete_rows(
            sb,
            "manager_leagues",
            &[("id", league_id.as_str())],
        )
        .await;
        log("cleaned up test league");
    }
    result
}

async fn run_inner(state: &AppState, league_id_out: &mut String) -> Result<(), ApiError> {
    // 1) Create a draft league (snapshots the real 2026/27 schedule).
    let league = league::create_league(
        state,
        "Dry Run League".to_string(),
        MODE_DRAFT.to_string(),
        crate::manager::DEFAULT_SEASON,
        COMMISH.to_string(),
        MY_TEAM.to_string(),
    )
    .await?;
    *league_id_out = str_value(league.get("id"));
    log(&format!(
        "league created id={} games={}",
        *league_id_out,
        league.get("games_total").and_then(|v| v.as_i64()).unwrap_or(0)
    ));

    // 2) Start the draft, then auto-fill MY_TEAM's picks until complete.
    let mut st = draft::start_draft(state, &*league_id_out, COMMISH).await?;
    let mut guard = 0;
    loop {
        guard += 1;
        if guard > 2000 {
            return Err(ApiError::Internal("draft loop guard tripped".into()));
        }
        let complete = st
            .get("complete")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);
        let status = str_value(st.get("status"));
        if complete || status != "drafting" {
            break;
        }
        let picker = str_value(st.get("current_picker"));
        if picker == MY_TEAM {
            let pid = draft::best_pick_for(state, &*league_id_out, MY_TEAM)
                .await
                .ok_or_else(|| ApiError::Internal("no pick available for MY_TEAM".into()))?;
            st = draft::submit_pick(state, &*league_id_out, COMMISH, pid).await?;
        } else {
            st = draft::draft_state(state, &*league_id_out, Some(COMMISH)).await?;
        }
    }
    let complete = st.get("complete").and_then(|v| v.as_bool()).unwrap_or(false);
    if !complete {
        return Err(ApiError::Internal("draft did not complete".into()));
    }
    log(&format!("draft complete (round marker {})", st.get("draft_round").and_then(|v| v.as_i64()).unwrap_or(0)));

    // Verify roster counts (minimums met, 26 total).
    let pre_roster = games::get_team_roster(state, &*league_id_out, MY_TEAM).await?;
    let counts = pre_roster.get("counts").cloned().unwrap_or(json!({}));
    log(&format!("MY_TEAM roster counts: {counts}"));
    let (f, d, g) = (
        counts.get("forwards").and_then(|v| v.as_i64()).unwrap_or(0),
        counts.get("defense").and_then(|v| v.as_i64()).unwrap_or(0),
        counts.get("goalies").and_then(|v| v.as_i64()).unwrap_or(0),
    );
    if f < 12 || d < 6 || g < 2 {
        return Err(ApiError::Internal(format!("draft produced invalid counts F{f} D{d} G{g}")));
    }

    // 3) Human-to-human trade: give BOS to a second "user", then ANA ↔ BOS.
    let bos = "BOS";
    let _ = crate::supabase::write::update_rows(
        state.sb.as_ref().unwrap(),
        "manager_league_teams",
        &[("league_id", league_id_out.as_str()), ("team_abbrev", bos)],
        &json!({"user_id": PARTNER}),
    )
    .await;
    let ana_fwds: Vec<i64> = pre_roster
        .get("forwards")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.get("player_id").and_then(|v| v.as_i64())).collect())
        .unwrap_or_default();
    let bos_roster = games::get_team_roster(state, &*league_id_out, bos).await?;
    let bos_fwds: Vec<i64> = bos_roster
        .get("forwards")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.get("player_id").and_then(|v| v.as_i64())).collect())
        .unwrap_or_default();
    if ana_fwds.is_empty() || bos_fwds.is_empty() {
        return Err(ApiError::Internal("no forwards for trade".into()));
    }
    let offer = trades::create_offer(
        state,
        &*league_id_out,
        MY_TEAM,
        bos,
        &[ana_fwds[0]],
        &[bos_fwds[0]],
        COMMISH,
    )
    .await?;
    log(&format!(
        "offer created {} -> {} (status {})",
        str_value(offer.get("from_team")),
        str_value(offer.get("to_team")),
        str_value(offer.get("status"))
    ));
    let offer_id = str_value(offer.get("id"));
    let accepted = trades::respond_offer(state, &*league_id_out, &offer_id, "accept", PARTNER).await?;
    log(&format!("trade accepted (status {})", str_value(accepted.get("status"))));

    // 4) Set MY_TEAM's round-1 lineup (re-fetch AFTER the trade, so only
    // currently-owned players are fielded; should not be locked pre-season).
    let post_roster = games::get_team_roster(state, &*league_id_out, MY_TEAM).await?;
    let lineup = build_lineup(&post_roster)?;
    let mut slots = std::collections::BTreeMap::new();
    for (slot, pid) in &lineup {
        slots.insert(slot.clone(), *pid);
    }
    let lu = games::set_lineup(state, &*league_id_out, MY_TEAM, 1, &slots).await?;
    log(&format!(
        "lineup set (locked={})",
        lu.get("locked").and_then(|v| v.as_bool()).unwrap_or(false)
    ));

    log("DRY RUN OK");
    Ok(())
}
