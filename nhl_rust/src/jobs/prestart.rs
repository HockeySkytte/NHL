//! Prestart logger thread (M5) — port of `_start_prestart_logger_thread_once`:
//! periodically scans today's ET schedule and appends a CSV snapshot row for
//! games within the prestart window (or grace period after puck drop).

use std::path::PathBuf;

use chrono::{TimeZone, Utc};
use serde_json::{json, Value};

use crate::config::Config;
use crate::state::AppState;

const CSV_FIELDS: [&str; 12] = [
    "TimestampUTC", "DateET", "GameID", "StartTimeET",
    "Away", "Home", "WinAway", "WinHome", "OddsAway", "OddsHome", "BetAway", "BetHome",
];

pub fn prestart_csv_path(cfg: &Config) -> PathBuf {
    if let Some(dir) = &cfg.prestart_dir {
        return dir.join(&cfg.prestart_csv);
    }
    let base = std::env::temp_dir();
    base.join(&cfg.prestart_csv)
}

fn to_decimal_odds(american: Option<f64>) -> Option<f64> {
    match american {
        None => None,
        Some(a) => {
            if a > 0.0 {
                Some(1.0 + a / 100.0)
            } else if a < 0.0 {
                Some(1.0 + 100.0 / a.abs())
            } else {
                None
            }
        }
    }
}

fn bet_fraction_kelly03(prob: Option<f64>, american: Option<f64>) -> Option<f64> {
    let p = prob?;
    if !(0.0..=1.0).contains(&p) {
        return None;
    }
    let dec = to_decimal_odds(american)?;
    if dec <= 1.0 {
        return None;
    }
    let b = dec - 1.0;
    let q = 1.0 - p;
    let f = (b * p - q) / b;
    let f_scaled = 0.3 * f;
    Some(if f_scaled > 0.0 { f_scaled } else { 0.0 })
}

fn round3(v: Option<f64>) -> Option<Value> {
    v.map(|x| Value::from((x * 100.0).round() / 100.0))
}

fn parse_utc(iso: &str) -> Option<chrono::DateTime<Utc>> {
    let raw = iso.trim();
    if raw.is_empty() {
        return None;
    }
    let normalized = raw.replace('Z', "+00:00");
    chrono::DateTime::parse_from_rfc3339(&normalized)
        .ok()
        .map(|d| d.with_timezone(&Utc))
}

async fn append_row(state: &AppState, row: &Value) {
    let path = prestart_csv_path(&state.cfg);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let exists = path.exists();
    let line = build_csv_line(row);
    let mut out = String::new();
    if !exists {
        out.push_str(&CSV_FIELDS.join(","));
        out.push('\n');
    }
    out.push_str(&line);
    out.push('\n');
    use std::io::Write;
    let mut f = match std::fs::OpenOptions::new().create(true).append(true).open(&path) {
        Ok(f) => f,
        Err(_) => return,
    };
    let _ = f.write_all(out.as_bytes());
}

fn build_csv_line(row: &Value) -> String {
    CSV_FIELDS
        .iter()
        .map(|f| {
            let v = row.get(*f).unwrap_or(&Value::Null);
            match v {
                Value::Null => String::new(),
                Value::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
                other => other.to_string(),
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn as_str(v: Option<&Value>) -> String {
    crate::web::auth_state::as_str_of(v)
}

fn as_f64(v: &Value) -> Option<f64> {
    v.as_f64()
}

/// Fetch the ET schedule for a date (port of `_build_games_for_date` core).
async fn fetch_games_for_date(state: &AppState, date_et: &str) -> Vec<Value> {
    let url = format!("https://api-web.nhle.com/v1/schedule/{date_et}");
    let Ok(resp) = state.http.get(&url).timeout(std::time::Duration::from_secs(20)).send().await else {
        return Vec::new();
    };
    if resp.status() != reqwest::StatusCode::OK {
        return Vec::new();
    }
    let Ok(js) = resp.json::<Value>().await else {
        return Vec::new();
    };
    let mut out: Vec<Value> = Vec::new();
    if let Some(weeks) = js.get("gameWeek").and_then(Value::as_array) {
        for wk in weeks {
            if wk.get("date").and_then(Value::as_str).unwrap_or("").chars().take(10).collect::<String>() != date_et {
                continue;
            }
            if let Some(games) = wk.get("games").and_then(Value::as_array) {
                for g in games {
                    let home = g.get("homeTeam").cloned().unwrap_or(Value::Null);
                    let away = g.get("awayTeam").cloned().unwrap_or(Value::Null);
                    out.push(json!({
                        "id": g.get("id").cloned().unwrap_or(Value::Null),
                        "startTimeUTC": g.get("startTimeUTC").cloned().unwrap_or(Value::Null),
                        "homeAbbrev": as_str(home.get("abbrev")).to_uppercase(),
                        "awayAbbrev": as_str(away.get("abbrev")).to_uppercase(),
                    }));
                }
            }
        }
    }
    out
}

/// Start the prestart logger task (idempotent per process via the flag).
pub fn start(state: AppState) {
    use std::sync::atomic::{AtomicBool, Ordering};
    static STARTED: AtomicBool = AtomicBool::new(false);
    if STARTED.swap(true, Ordering::SeqCst) {
        return;
    }
    tokio::spawn(async move {
        let cfg = state.cfg.clone();
        loop {
            run_once(&state, &cfg).await;
            tokio::time::sleep(std::time::Duration::from_secs(120)).await;
        }
    });
}

async fn run_once(state: &AppState, cfg: &Config) {
    let window_secs = env_i64("PRESTART_WINDOW_SECONDS", 3600).max(30);
    let grace_secs = env_i64("PRESTART_GRACE_SECONDS", 300).max(0);
    let now_utc = Utc::now();
    let now_et = chrono_tz::America::New_York.from_utc_datetime(&now_utc.naive_utc());
    let date_et = now_et.date_naive().to_string();
    let games = fetch_games_for_date(state, &date_et).await;
    for g in &games {
        let gid = g.get("id").and_then(Value::as_i64);
        let Some(gid) = gid else { continue };
        let Some(st_raw) = g.get("startTimeUTC").and_then(Value::as_str) else { continue };
        let Some(start_utc) = parse_utc(st_raw) else { continue };
        let delta_before = (start_utc - now_utc).num_seconds();
        let delta_after = (now_utc - start_utc).num_seconds();
        if !((0..=window_secs).contains(&delta_before) || (0..=grace_secs).contains(&delta_after)) {
            continue;
        }
        // Load odds snapshot for this game (best-effort).
        let snapshot = crate::data::odds::load_snapshot_rows(&state.caches, state.sb.as_ref(), gid)
            .await
            .into_iter()
            .next();
        let odds_away = snapshot
            .as_ref()
            .and_then(|r| r.get("oddsAway"))
            .and_then(Value::as_f64);
        let odds_home = snapshot
            .as_ref()
            .and_then(|r| r.get("oddsHome"))
            .and_then(Value::as_f64);
        let win_away = snapshot
            .as_ref()
            .and_then(|r| r.get("winAwayPct"))
            .and_then(Value::as_f64)
            .map(|v| v / 100.0);
        let win_home = snapshot
            .as_ref()
            .and_then(|r| r.get("winHomePct"))
            .and_then(Value::as_f64)
            .map(|v| v / 100.0);
        let bet_away = bet_fraction_kelly03(win_away, odds_away);
        let bet_home = bet_fraction_kelly03(win_home, odds_home);
        let ts = now_utc.format("%Y-%m-%dT%H:%M:%S%.fZ").to_string();
        let row = json!({
            "TimestampUTC": ts,
            "DateET": date_et,
            "GameID": gid,
            "StartTimeET": st_raw,
            "Away": as_str(g.get("awayAbbrev")),
            "Home": as_str(g.get("homeAbbrev")),
            "WinAway": round3(win_away),
            "WinHome": round3(win_home),
            "OddsAway": odds_away,
            "OddsHome": odds_home,
            "BetAway": round3(bet_away),
            "BetHome": round3(bet_home),
        });
        append_row(state, &row).await;
    }
}

fn env_i64(name: &str, default: i64) -> i64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(default)
}
