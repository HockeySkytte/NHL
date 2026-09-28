//! Manager Game games: per-game stat-line recording, lineups (+ round lock),
//! and match finalization (W/L only, no OT).

use std::collections::{BTreeMap, HashSet};

use serde_json::{json, Value};

use crate::error::ApiError;
use crate::manager::{league, scoring};
use crate::manager::scoring::StatLine;
use crate::state::AppState;
use crate::supabase::read::SbClient;
use crate::supabase::write;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

/// Lineup shape helper: which position a slot label implies, and the slot list.
const F_SLOTS: &[&str] = &[
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12",
];
const D_SLOTS: &[&str] = &["D1", "D2", "D3", "D4", "D5", "D6"];
const G_SLOTS: &[&str] = &["G1", "G2"];

fn slot_position(slot: &str) -> Option<&'static str> {
    if F_SLOTS.contains(&slot) {
        Some("F")
    } else if D_SLOTS.contains(&slot) {
        Some("D")
    } else if G_SLOTS.contains(&slot) {
        Some("G")
    } else {
        None
    }
}

/// Build a pid → position map from the current rosters pool.
async fn pid_positions(state: &AppState) -> BTreeMap<i64, String> {
    let pool = crate::manager::draft::draft_pool(state).await;
    pool.into_iter()
        .filter_map(|(pid, info)| {
            let pos = str_value(info.get("position"))
                .chars()
                .next()
                .map(|c| c.to_string())
                .unwrap_or_default();
            if pos.is_empty() {
                None
            } else {
                Some((pid, pos))
            }
        })
        .collect()
}

/// Player ids on the team's roster (empty when the pool isn't seeded yet).
async fn load_roster_pids(sb: &SbClient, league_id: &str, team: &str) -> HashSet<i64> {
    let rows = sb
        .read(
            "manager_rosters",
            "player_id",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("team_abbrev".to_string(), format!("eq.{team}")),
            ])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    rows.into_iter()
        .filter_map(|r| safe_int(r.get("player_id")))
        .collect()
}

/// A team's roster pool (for the lineup selector): players with name, position,
/// and NHL team, plus F/D/G counts and the required minimums.
pub async fn get_team_roster(
    state: &AppState,
    league_id: &str,
    team: &str,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let team = team.trim().to_uppercase();
    let pool = crate::manager::draft::draft_pool(state).await;
    let rows = sb
        .read(
            "manager_rosters",
            "player_id,position",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("team_abbrev".to_string(), format!("eq.{team}")),
            ])),
            None,
            Some("position.asc,player_id.asc"),
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;

    let mut forwards: Vec<Value> = Vec::new();
    let mut defense: Vec<Value> = Vec::new();
    let mut goalies: Vec<Value> = Vec::new();
    for r in &rows {
        let pid = safe_int(r.get("player_id")).unwrap_or(0);
        let pos = str_value(r.get("position"));
        let info = pool.get(&pid);
        let player = json!({
            "player_id": pid,
            "name": info.map(|i| str_value(i.get("name"))).unwrap_or_default(),
            "position": pos,
            "team": info.map(|i| str_value(i.get("team"))).unwrap_or_default(),
        });
        match pos.as_str() {
            "F" => forwards.push(player),
            "D" => defense.push(player),
            "G" => goalies.push(player),
            _ => {}
        }
    }
    Ok(json!({
        "league_id": league_id,
        "team_abbrev": team,
        "forwards": forwards,
        "defense": defense,
        "goalies": goalies,
        "counts": {
            "forwards": forwards.len(),
            "defense": defense.len(),
            "goalies": goalies.len(),
        },
        "minimums": {
            "forwards": crate::manager::LINEUP_FORWARDS,
            "defense": crate::manager::LINEUP_DEFENSE,
            "goalies": crate::manager::LINEUP_GOALIES,
        },
    }))
}

/// Record per-player stat lines for a finalized (or any) game into
/// `manager_game_stats`, computed from the live play-by-play pipeline.
/// Returns `(game_state, rows)`; caller decides whether to finalize.
pub async fn record_game_stats(
    state: &AppState,
    nhl_game_id: i64,
) -> Result<(String, Vec<Value>), ApiError> {
    let sb = league::sb_required(state)?;
    let (plays, game_state) = crate::routes::pbp::build_plays(state, nhl_game_id, "xG_F", false)
        .await
        .map_err(|_| ApiError::Internal("pbp build failed".into()))?;
    let stats = scoring::stat_lines_from_plays(&plays);

    let mut rows = Vec::with_capacity(stats.len());
    for (pid, line) in &stats {
        rows.push(scoring::stat_line_to_value(*pid, line));
    }
    // Storage rows (DB columns).
    let db_rows: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "nhl_game_id": nhl_game_id,
                "player_id": safe_int(r.get("player_id")).unwrap_or(0),
                "goals": safe_int(r.get("goals")).unwrap_or(0),
                "a1": safe_int(r.get("a1")).unwrap_or(0),
                "a2": safe_int(r.get("a2")).unwrap_or(0),
                "sog": safe_int(r.get("sog")).unwrap_or(0),
                "pent": safe_int(r.get("pent")).unwrap_or(0),
                "pend": safe_int(r.get("pend")).unwrap_or(0),
                "xgf_5v5": parse_locale_float(r.get("xgf_5v5")).unwrap_or(0.0),
                "xga_5v5": parse_locale_float(r.get("xga_5v5")).unwrap_or(0.0),
                "gsax": parse_locale_float(r.get("gsax")).unwrap_or(0.0),
                "fantasy_points": parse_locale_float(r.get("fantasy_points")).unwrap_or(0.0),
            })
        })
        .collect();
    if !db_rows.is_empty() {
        write::upsert_rows(sb, "manager_game_stats", &db_rows, "nhl_game_id,player_id")
            .await
            .ok_or_else(|| ApiError::Internal("stat storage failed".into()))?;
    }
    Ok((game_state, rows))
}

/// True when a team's lineup for `round` is locked: the team's NHL game for
/// that round has started (status != pending).
pub async fn lineup_locked(
    sb: &SbClient,
    league_id: &str,
    team: &str,
    round: i64,
) -> bool {
    let game = sb
        .read(
            "manager_games",
            "status",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("or".to_string(), format!(
                    "(and(home_abbrev.eq.{team},home_round.eq.{round}),and(away_abbrev.eq.{team},away_round.eq.{round}))"
                )),
            ])),
            None,
            None,
            1,
        )
        .await
        .unwrap_or_default();
    let status = game.first().map(|r| str_value(r.get("status"))).unwrap_or_default();
    matches!(status.as_str(), "live" | "final" | "postponed")
}

/// Set a team's full 20-slot lineup for a round (mode 1). Validates slot
/// position vs player position, enforces the round lock, and replaces the
/// previous lineup for that (team, round).
pub async fn set_lineup(
    state: &AppState,
    league_id: &str,
    team: &str,
    round: i64,
    slots: &BTreeMap<String, i64>,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let team = team.trim().to_uppercase();
    if round < 1 {
        return Err(ApiError::BadRequest(json!({"error": "invalid_round"})));
    }

    if lineup_locked(sb, league_id, &team, round).await {
        return Err(ApiError::BadRequest(json!({"error": "lineup_locked"})));
    }

    // Every slot must be present and position-correct.
    let pos_map = pid_positions(state).await;
    // Only field players you own (from your roster). If the team has no roster
    // rows yet (mode-1 pool not seeded until M5), skip this constraint.
    let owned: HashSet<i64> = load_roster_pids(sb, league_id, &team).await;
    let mut linelist: Vec<Value> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();
    for slot in crate::manager::lineup_slots() {
        let pid = *slots
            .get(&slot)
            .ok_or_else(|| ApiError::BadRequest(json!({"error": "incomplete_lineup"})))?;
        if pid <= 0 || !seen.insert(pid) {
            return Err(ApiError::BadRequest(json!({"error": "invalid_lineup"})));
        }
        if !owned.is_empty() && !owned.contains(&pid) {
            return Err(ApiError::BadRequest(json!({"error": "not_owned"})));
        }
        let slot_pos = slot_position(&slot).unwrap_or("");
        let ppos = pos_map.get(&pid).map(|s| s.as_str()).unwrap_or("");
        if ppos != slot_pos {
            return Err(ApiError::BadRequest(json!({"error": "position_mismatch"})));
        }
        linelist.push(json!({
            "league_id": league_id,
            "team_abbrev": team,
            "round": round,
            "slot": slot,
            "player_id": pid,
        }));
    }

    // Replace the previous lineup for (team, round).
    let _ = write::delete_rows(
        sb,
        "manager_lineups",
        &[
            ("league_id", league_id),
            ("team_abbrev", team.as_str()),
            ("round", &round.to_string()),
        ],
    )
    .await;
    write::upsert_rows(sb, "manager_lineups", &linelist, "league_id,team_abbrev,round,slot")
        .await
        .ok_or_else(|| ApiError::Internal("lineup save failed".into()))?;

    get_lineup(sb, league_id, &team, round).await
}

/// Get a team's active lineup for a round.
pub async fn get_lineup(
    sb: &SbClient,
    league_id: &str,
    team: &str,
    round: i64,
) -> Result<Value, ApiError> {
    let rows = sb
        .read(
            "manager_lineups",
            "slot,player_id",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("team_abbrev".to_string(), format!("eq.{team}")),
                ("round".to_string(), format!("eq.{round}")),
            ])),
            None,
            Some("slot.asc"),
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let locked = lineup_locked(sb, league_id, team, round).await;
    // Resolve names from the pool for a friendly response.
    let mut by_slot: BTreeMap<String, Value> = BTreeMap::new();
    let mut pids: Vec<i64> = Vec::new();
    for r in &rows {
        let slot = str_value(r.get("slot"));
        let pid = safe_int(r.get("player_id")).unwrap_or(0);
        let key = slot.clone();
        by_slot.insert(key, json!({"slot": slot, "player_id": pid, "name": ""}));
        if pid > 0 {
            pids.push(pid);
        }
    }
    Ok(json!({
        "league_id": league_id,
        "team_abbrev": team,
        "round": round,
        "locked": locked,
        "slots": by_slot,
        "pids": pids,
    }))
}

async fn load_stats(sb: &SbClient, nhl_game_id: i64) -> BTreeMap<i64, StatLine> {
    let rows = sb
        .read(
            "manager_game_stats",
            "player_id,goals,a1,a2,sog,pent,pend,xgf_5v5,xga_5v5,gsax",
            Some(&BTreeMap::from([(
                "nhl_game_id".to_string(),
                format!("eq.{nhl_game_id}"),
            )])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    rows.into_iter()
        .filter_map(|r| {
            let pid = safe_int(r.get("player_id"))?;
            let line = StatLine {
                goals: safe_int(r.get("goals")).unwrap_or(0),
                a1: safe_int(r.get("a1")).unwrap_or(0),
                a2: safe_int(r.get("a2")).unwrap_or(0),
                sog: safe_int(r.get("sog")).unwrap_or(0),
                pent: safe_int(r.get("pent")).unwrap_or(0),
                pend: safe_int(r.get("pend")).unwrap_or(0),
                xgf_5v5: parse_locale_float(r.get("xgf_5v5")).unwrap_or(0.0),
                xga_5v5: parse_locale_float(r.get("xga_5v5")).unwrap_or(0.0),
                gsax: parse_locale_float(r.get("gsax")).unwrap_or(0.0),
            };
            if pid > 0 {
                Some((pid, line))
            } else {
                None
            }
        })
        .collect()
}

/// Finalize a manager game: compute both lineups' round scores from the
/// recorded stat lines, decide a winner (W/L only), and update the game.
/// Returns an error if either lineup isn't set yet.
pub async fn finalize_game(
    state: &AppState,
    league_id: &str,
    nhl_game_id: i64,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let filters = BTreeMap::from([
        ("league_id".to_string(), format!("eq.{league_id}")),
        ("nhl_game_id".to_string(), format!("eq.{nhl_game_id}")),
    ]);
    let rows = sb
        .read("manager_games", "*", Some(&filters), None, None, 1)
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let game = rows
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("game not found".into()))?;
    if str_value(game.get("status")) == "final" {
        return Ok(game);
    }

    let home = str_value(game.get("home_abbrev"));
    let away = str_value(game.get("away_abbrev"));
    let home_round = safe_int(game.get("home_round")).unwrap_or(0);
    let away_round = safe_int(game.get("away_round")).unwrap_or(0);

    let stats = load_stats(sb, nhl_game_id).await;
    let home_lineup = get_lineup(sb, league_id, &home, home_round).await?;
    let away_lineup = get_lineup(sb, league_id, &away, away_round).await?;
    let lineup_slots = |lineup: &Value| -> Vec<(String, i64)> {
        lineup
            .get("slots")
            .and_then(Value::as_object)
            .map(|m| {
                m.iter()
                    .filter_map(|(slot, v)| v.get("player_id").and_then(|x| x.as_i64()).map(|pid| (slot.clone(), pid)))
                    .collect()
            })
            .unwrap_or_default()
    };
    let home_slots = lineup_slots(&home_lineup);
    let away_slots = lineup_slots(&away_lineup);
    if home_slots.len() < crate::manager::LINEUP_SLOTS || away_slots.len() < crate::manager::LINEUP_SLOTS {
        return Err(ApiError::BadRequest(json!({"error": "lineup_missing"})));
    }

    let home_score = scoring::team_round_score(&home_slots, &stats);
    let away_score = scoring::team_round_score(&away_slots, &stats);
    let winner = scoring::decide_winner(home_score, away_score, &home_slots, &away_slots, &stats);

    let patch = json!({
        "status": "final",
        "home_score": round2(home_score),
        "away_score": round2(away_score),
        "finalized_at": chrono::Utc::now(),
    });
    write::update_rows(sb, "manager_games", &[("league_id", league_id), ("nhl_game_id", &nhl_game_id.to_string())], &patch)
        .await;

    Ok(json!({
        "league_id": league_id,
        "nhl_game_id": nhl_game_id,
        "home_abbrev": home,
        "away_abbrev": away,
        "home_round": home_round,
        "away_round": away_round,
        "home_score": round2(home_score),
        "away_score": round2(away_score),
        "winner": if winner == "home" { home } else { away },
        "status": "final",
    }))
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slot_positions_are_correct() {
        assert_eq!(slot_position("F1"), Some("F"));
        assert_eq!(slot_position("F12"), Some("F"));
        assert_eq!(slot_position("D6"), Some("D"));
        assert_eq!(slot_position("G2"), Some("G"));
        assert_eq!(slot_position("X9"), None);
    }
}
