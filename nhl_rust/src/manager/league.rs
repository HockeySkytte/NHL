//! League state: creation, join codes, franchise slots (CPU = unclaimed),
//! membership, and league detail/summary payloads.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{json, Value};

use crate::data::projections;
use crate::error::ApiError;
use crate::manager::{self, schedule};
use crate::state::AppState;
use crate::supabase::write;
use crate::util::parse::str_value;

/// Join-code alphabet without ambiguous characters (no I/L/O/0/1).
const JOIN_CODE_ALPHABET: &[u8] = b"ABCDEFGHJKMNPQRSTUVWXYZ23456789";

/// Random 8-character join code. Modulo bias over bytes is acceptable for a
/// non-secret lobby code (uniqueness is enforced by the DB unique index).
pub fn join_code() -> String {
    let mut buf = [0u8; 8];
    let _ = getrandom::getrandom(&mut buf);
    let n = JOIN_CODE_ALPHABET.len() as u8;
    buf.iter()
        .map(|b| JOIN_CODE_ALPHABET[(b % n) as usize] as char)
        .collect()
}

/// Fisher-Yates shuffle of 1..N (draft slots), seeded from OS randomness.
pub fn shuffled_slots(n: usize) -> Vec<i64> {
    let mut slots: Vec<i64> = (1..=n as i64).collect();
    let mut buf = [0u8; 8];
    for i in (1..slots.len()).rev() {
        let _ = getrandom::getrandom(&mut buf);
        let r = u64::from_le_bytes(buf);
        slots.swap(i, (r as usize) % (i + 1));
    }
    slots
}

pub(crate) fn sb_required(state: &AppState) -> Result<&crate::supabase::read::SbClient, ApiError> {
    state
        .sb
        .as_ref()
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))
}

pub(crate) fn eq_filter(col: &str, val: &str) -> BTreeMap<String, String> {
    BTreeMap::from([(col.to_string(), format!("eq.{val}"))])
}

/// Create a league: league row, 32 franchise slots (creator claims
/// `franchise`), and the schedule snapshot with per-team round indices.
pub async fn create_league(
    state: &AppState,
    name: String,
    mode: String,
    season: i64,
    commissioner_user_id: String,
    franchise: String,
) -> Result<Value, ApiError> {
    let sb = sb_required(state)?;

    let name = name.trim().to_string();
    if name.is_empty() || name.chars().count() > 80 {
        return Err(ApiError::BadRequest(json!({"error": "invalid_name"})));
    }
    if mode != manager::MODE_TRUE_ROSTERS && mode != manager::MODE_DRAFT {
        return Err(ApiError::BadRequest(json!({"error": "invalid_mode"})));
    }
    let season = if season == 0 { manager::DEFAULT_SEASON } else { season };
    let franchise = franchise.trim().to_uppercase();

    let teams = projections::active_team_abbrevs(state);
    if teams.is_empty() {
        return Err(ApiError::Internal("no active teams".into()));
    }
    if !teams.contains(&franchise) {
        return Err(ApiError::BadRequest(json!({"error": "invalid_franchise"})));
    }

    // Unique join code (retry on collision; the DB unique index is the
    // final guard).
    let mut code = join_code();
    for _ in 0..5 {
        let existing = sb
            .read(
                "manager_leagues",
                "id",
                Some(&eq_filter("join_code", &code)),
                None,
                None,
                1,
            )
            .await;
        if existing.as_ref().map(|r| r.is_empty()).unwrap_or(false) {
            break;
        }
        code = join_code();
    }

    let league_payload = json!({
        "name": name,
        "season": season,
        "mode": mode,
        "status": manager::STATUS_SETUP,
        "commissioner_user_id": commissioner_user_id,
        "join_code": code,
        "cpu_trades": true,
        "draft_round": 0,
        "draft_pick": 0,
        "draft_paused": false,
    });
    let created = write::upsert_rows(
        sb,
        "manager_leagues",
        std::slice::from_ref(&league_payload),
        "join_code",
    )
    .await
    .ok_or_else(|| ApiError::Internal("league create failed".into()))?;
    let league_row = created
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Internal("league create returned no row".into()))?;
    let league_id = str_value(league_row.get("id"));
    if league_id.is_empty() {
        return Err(ApiError::Internal("league create returned no id".into()));
    }

    // Franchise slots: 32 rows, CPU by default, creator claims one.
    let slots = shuffled_slots(teams.len());
    let team_rows: Vec<Value> = teams
        .iter()
        .zip(slots.iter())
        .map(|(t, slot)| {
            json!({
                "league_id": league_id,
                "team_abbrev": t,
                "user_id": if *t == franchise { json!(commissioner_user_id) } else { Value::Null },
                "team_name": "",
                "draft_slot": slot,
            })
        })
        .collect();
    write::upsert_rows(
        sb,
        "manager_league_teams",
        &team_rows,
        "league_id,team_abbrev",
    )
    .await
    .ok_or_else(|| ApiError::Internal("league team rows failed".into()))?;

    // Schedule snapshot.
    let mut schedule_rows = schedule::build_league_schedule_rows(state, season).await?;
    for r in &mut schedule_rows {
        r["league_id"] = json!(league_id);
    }
    write::upsert_rows(sb, "manager_games", &schedule_rows, "league_id,nhl_game_id")
        .await
        .ok_or_else(|| ApiError::Internal("league schedule insert failed".into()))?;

    league_detail(state, &league_id).await
}

/// List leagues the user belongs to (as commissioner or as a member).
pub async fn list_leagues_for_user(state: &AppState, user_id: &str) -> Result<Vec<Value>, ApiError> {
    let sb = sb_required(state)?;
    let cols = "id,name,season,mode,status,join_code,commissioner_user_id,created_at";

    let as_commissioner = sb
        .read(
            "manager_leagues",
            cols,
            Some(&eq_filter("commissioner_user_id", user_id)),
            None,
            Some("created_at.desc"),
            0,
        )
        .await
        .unwrap_or_default();

    let memberships = sb
        .read(
            "manager_league_teams",
            "league_id,team_abbrev,team_name",
            Some(&eq_filter("user_id", user_id)),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    let member_ids: BTreeSet<String> = memberships
        .iter()
        .filter_map(|r| {
            let id = str_value(r.get("league_id"));
            if id.is_empty() {
                None
            } else {
                Some(id)
            }
        })
        .collect();
    // The caller's franchise per league (for lobby cards: logo + "my team").
    let mut my_teams: BTreeMap<String, (String, String)> = BTreeMap::new();
    for r in &memberships {
        let league_id = str_value(r.get("league_id"));
        let abbrev = str_value(r.get("team_abbrev"));
        if !league_id.is_empty() && !abbrev.is_empty() {
            let team_name = str_value(r.get("team_name"));
            my_teams.insert(league_id, (abbrev, team_name));
        }
    }

    let mut out: Vec<Value> = as_commissioner;
    if !member_ids.is_empty() {
        let in_filter = member_ids.iter().cloned().collect::<Vec<_>>().join(",");
        let extra = sb
            .read(
                "manager_leagues",
                cols,
                Some(&BTreeMap::from([(
                    "id".to_string(),
                    format!("in.({in_filter})"),
                )])),
                None,
                Some("created_at.desc"),
                0,
            )
            .await
            .unwrap_or_default();
        let seen: BTreeSet<String> = out
            .iter()
            .filter_map(|r| {
                let id = str_value(r.get("id"));
                if id.is_empty() {
                    None
                } else {
                    Some(id)
                }
            })
            .collect();
        out.extend(
            extra
                .into_iter()
                .filter(|r| !seen.contains(&str_value(r.get("id")))),
        );
    }
    for row in out.iter_mut() {
        let league_id = str_value(row.get("id"));
        if let Some((abbrev, team_name)) = my_teams.get(&league_id) {
            row["my_team"] = json!(abbrev);
            row["my_team_name"] = json!(team_name);
        }
    }

    Ok(out)
}

/// Full league detail: league row, teams, schedule summary.
pub async fn league_detail(state: &AppState, league_id: &str) -> Result<Value, ApiError> {
    let sb = sb_required(state)?;
    let rows = sb
        .read(
            "manager_leagues",
            "*",
            Some(&eq_filter("id", league_id)),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let league_row = rows
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("league not found".into()))?;

    let teams = sb
        .read(
            "manager_league_teams",
            "*",
            Some(&eq_filter("league_id", league_id)),
            None,
            Some("draft_slot.asc"),
            0,
        )
        .await
        .unwrap_or_default();

    // Game count by status (M1: totals only; per-game data via /schedule).
    let game_rows = sb
        .read(
            "manager_games",
            "status",
            Some(&eq_filter("league_id", league_id)),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    let mut total = 0i64;
    let mut final_count = 0i64;
    for g in &game_rows {
        total += 1;
        if str_value(g.get("status")) == "final" {
            final_count += 1;
        }
    }

    Ok(json!({
        "id": league_row.get("id"),
        "name": league_row.get("name"),
        "season": league_row.get("season"),
        "mode": league_row.get("mode"),
        "status": league_row.get("status"),
        "join_code": league_row.get("join_code"),
        "commissioner_user_id": league_row.get("commissioner_user_id"),
        "cpu_trades": league_row.get("cpu_trades"),
        "draft_round": league_row.get("draft_round"),
        "draft_pick": league_row.get("draft_pick"),
        "draft_paused": league_row.get("draft_paused"),
        "created_at": league_row.get("created_at"),
        "teams": teams,
        "games_total": total,
        "games_final": final_count,
    }))
}

/// The user's team row in a league (or `None` when not a member).
pub async fn membership(
    state: &AppState,
    league_id: &str,
    user_id: &str,
) -> Option<Value> {
    let sb = state.sb.as_ref()?;
    let rows = sb
        .read(
            "manager_league_teams",
            "*",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("user_id".to_string(), format!("eq.{user_id}")),
            ])),
            None,
            None,
            1,
        )
        .await?;
    rows.into_iter().next()
}

/// Join a league by join code, claiming an unclaimed franchise.
pub async fn join_league(
    state: &AppState,
    join_code_in: &str,
    franchise: &str,
    user_id: &str,
) -> Result<Value, ApiError> {
    let sb = sb_required(state)?;
    let code = join_code_in.trim().to_uppercase();
    if code.is_empty() {
        return Err(ApiError::BadRequest(json!({"error": "invalid_code"})));
    }

    let rows = sb
        .read(
            "manager_leagues",
            "*",
            Some(&eq_filter("join_code", &code)),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let league_row = rows
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("league not found".into()))?;
    let league_id = str_value(league_row.get("id"));
    let status = str_value(league_row.get("status"));
    if status != manager::STATUS_SETUP && status != manager::STATUS_DRAFTING {
        return Err(ApiError::BadRequest(json!({"error": "league_locked"})));
    }

    let franchise = franchise.trim().to_uppercase();
    let teams = projections::active_team_abbrevs(state);
    if !teams.contains(&franchise) {
        return Err(ApiError::BadRequest(json!({"error": "invalid_franchise"})));
    }

    let team_rows = sb
        .read(
            "manager_league_teams",
            "team_abbrev,user_id",
            Some(&eq_filter("league_id", &league_id)),
            None,
            None,
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;

    // One team per user per league.
    if team_rows
        .iter()
        .any(|r| str_value(r.get("user_id")) == user_id)
    {
        return Err(ApiError::BadRequest(json!({"error": "already_member"})));
    }

    let mut target_found = false;
    for r in &team_rows {
        if str_value(r.get("team_abbrev")) == franchise {
            target_found = true;
            if r.get("user_id").map(|v| !v.is_null()).unwrap_or(false) {
                return Err(ApiError::BadRequest(json!({"error": "franchise_taken"})));
            }
        }
    }
    if !target_found {
        return Err(ApiError::BadRequest(json!({"error": "invalid_franchise"})));
    }

    let ok = write::update_rows(
        sb,
        "manager_league_teams",
        &[("league_id", league_id.as_str()), ("team_abbrev", franchise.as_str())],
        &json!({"user_id": user_id}),
    )
    .await;
    if !ok {
        return Err(ApiError::Internal("join update failed".into()));
    }

    league_detail(state, &league_id).await
}

/// True when the user is the league commissioner.
pub async fn is_commissioner(state: &AppState, league_id: &str, user_id: &str) -> bool {
    let Some(sb) = state.sb.as_ref() else {
        return false;
    };
    let rows = sb
        .read(
            "manager_leagues",
            "id",
            Some(&BTreeMap::from([
                ("id".to_string(), format!("eq.{league_id}")),
                ("commissioner_user_id".to_string(), format!("eq.{user_id}")),
            ])),
            None,
            None,
            1,
        )
        .await;
    matches!(rows, Some(r) if !r.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn join_codes_are_eight_unambiguous_chars() {
        for _ in 0..50 {
            let code = join_code();
            assert_eq!(code.len(), 8);
            assert!(code
                .bytes()
                .all(|b| JOIN_CODE_ALPHABET.contains(&b)));
        }
    }

    #[test]
    fn shuffled_slots_is_a_permutation() {
        for n in [2usize, 8, 32] {
            let slots = shuffled_slots(n);
            let mut sorted = slots.clone();
            sorted.sort();
            assert_eq!(sorted, (1..=n as i64).collect::<Vec<_>>());
        }
    }
}
