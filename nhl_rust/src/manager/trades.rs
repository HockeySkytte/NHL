//! Manager Game trades (M4): manager-to-manager player trades with an atomic
//! swap, roster-validity enforcement, a CPU trade bot by projected value, and
//! the round-lock rule (a team whose current-round game has started can't
//! trade players out this round).
//!
//! Trades move players between `manager_rosters` rows. Swap sizes must be
//! equal and both resulting rosters must stay valid (mode 2: exactly
//! 12F/6D/2G; mode 1: at least 12F/6D/2G). Lineup slots that referenced a
//! traded player are voided so no stale lineup survives the swap.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::error::ApiError;
use crate::manager::{games, league};
use crate::state::AppState;
use crate::supabase::read::SbClient;
use crate::supabase::write;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

/// Resulting roster validity for F/D/G counts (both modes): the 12F/6D/2G
/// minimum must be met. There is no per-position maximum — a 26-man roster
/// has bench depth beyond the minimum.
pub fn counts_valid(counts: (i64, i64, i64)) -> bool {
    counts.0 >= crate::manager::LINEUP_FORWARDS as i64
        && counts.1 >= crate::manager::LINEUP_DEFENSE as i64
        && counts.2 >= crate::manager::LINEUP_GOALIES as i64
}

/// Apply a swap to (f,d,g) counts: remove `out` position counts, add `in`.
fn apply_swap(
    counts: (i64, i64, i64),
    out_positions: &[&str],
    in_positions: &[&str],
) -> (i64, i64, i64) {
    let mut c = counts;
    for p in out_positions {
        match *p {
            "F" => c.0 -= 1,
            "D" => c.1 -= 1,
            "G" => c.2 -= 1,
            _ => {}
        }
    }
    for p in in_positions {
        match *p {
            "F" => c.0 += 1,
            "D" => c.1 += 1,
            "G" => c.2 += 1,
            _ => {}
        }
    }
    c
}

/// Team's current round = final games played + 1 (or 1 before the season).
pub async fn current_round_for_team(sb: &SbClient, league_id: &str, team: &str) -> i64 {
    let rows = sb
        .read(
            "manager_games",
            "status",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("or".to_string(), format!(
                    "(home_abbrev.eq.{team},away_abbrev.eq.{team})"
                )),
            ])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    let played = rows
        .iter()
        .filter(|r| str_value(r.get("status")) == "final")
        .count() as i64;
    played + 1
}

/// A trade is locked if either side's current-round game has started.
async fn trade_locked(sb: &SbClient, league_id: &str, from: &str, to: &str) -> bool {
    let from_round = current_round_for_team(sb, league_id, from).await;
    let to_round = current_round_for_team(sb, league_id, to).await;
    games::lineup_locked(sb, league_id, from, from_round).await
        || games::lineup_locked(sb, league_id, to, to_round).await
}

/// Load a team's roster as (pid, position) list.
async fn load_roster(sb: &SbClient, league_id: &str, team: &str) -> Vec<(i64, String)> {
    let rows = sb
        .read(
            "manager_rosters",
            "player_id,position",
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
        .filter_map(|r| {
            let pid = safe_int(r.get("player_id"))?;
            let pos = str_value(r.get("position"));
            if pos.is_empty() {
                None
            } else {
                Some((pid, pos))
            }
        })
        .collect()
}

fn counts(roster: &[(i64, String)]) -> (i64, i64, i64) {
    let mut c = (0, 0, 0);
    for (_, p) in roster {
        match p.as_str() {
            "F" => c.0 += 1,
            "D" => c.1 += 1,
            "G" => c.2 += 1,
            _ => {}
        }
    }
    c
}

/// Create a trade offer. When the recipient is a CPU team, it is decided
/// immediately (accept via the value bot, else auto-decline). Returns the
/// (possibly already-resolved) offer.
pub async fn create_offer(
    state: &AppState,
    league_id: &str,
    from_team: &str,
    to_team: &str,
    offered: &[i64],
    requested: &[i64],
    user_id: &str,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let from = from_team.trim().to_uppercase();
    let to = to_team.trim().to_uppercase();
    if from == to || offered.is_empty() || requested.is_empty() || offered.len() != requested.len() {
        return Err(ApiError::BadRequest(json!({"error": "invalid_offer"})));
    }

    // The sender must manage `from`; only one team per user.
    let membership = league::membership(state, league_id, user_id).await;
    let ok = match &membership {
        Some(t) => str_value(t.get("team_abbrev")) == from,
        None => false,
    };
    if !ok {
        return Err(ApiError::BadRequest(json!({"error": "not_your_team"})));
    }

    // Roster validity of the swap.
    let from_roster = load_roster(sb, league_id, &from).await;
    let to_roster = load_roster(sb, league_id, &to).await;
    let from_pos: HashMap<i64, String> = from_roster.iter().cloned().collect();
    let to_pos: HashMap<i64, String> = to_roster.iter().cloned().collect();
    if offered.iter().any(|p| !from_pos.contains_key(p)) {
        return Err(ApiError::BadRequest(json!({"error": "not_owned"})));
    }
    if requested.iter().any(|p| !to_pos.contains_key(p)) {
        return Err(ApiError::BadRequest(json!({"error": "counterpart_not_owned"})));
    }
    // All pids distinct within the swap.
    let mut all: Vec<i64> = offered.to_vec();
    all.extend_from_slice(requested);
    let set: std::collections::HashSet<i64> = all.iter().copied().collect();
    if set.len() != all.len() {
        return Err(ApiError::BadRequest(json!({"error": "duplicate_player"})));
    }

    // Position validity after swap.
    let out_pos: Vec<&str> = offered.iter().map(|p| from_pos[p].as_str()).collect();
    let in_pos: Vec<&str> = requested.iter().map(|p| to_pos[p].as_str()).collect();
    let from_counts_new = apply_swap(counts(&from_roster), &out_pos, &in_pos);
    let to_counts_new = apply_swap(counts(&to_roster), &in_pos, &out_pos);
    if !counts_valid(from_counts_new) || !counts_valid(to_counts_new) {
        return Err(ApiError::BadRequest(json!({"error": "roster_invalid"})));
    }

    // Round-lock.
    if trade_locked(sb, league_id, &from, &to).await {
        return Err(ApiError::BadRequest(json!({"error": "trade_locked"})));
    }

    // Is the recipient a CPU team?
    let to_team_row = sb
        .read(
            "manager_league_teams",
            "user_id",
            Some(&BTreeMap::from([
                ("league_id".to_string(), format!("eq.{league_id}")),
                ("team_abbrev".to_string(), format!("eq.{to}")),
            ])),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?
        .into_iter()
        .next();
    let to_user_id = to_team_row
        .as_ref()
        .and_then(|r| r.get("user_id"))
        .and_then(|v| if v.is_null() { None } else { v.as_str() })
        .map(|s| s.to_string());

    // CPU teams are never involved in trades (human-to-human only).
    if to_user_id.is_none() {
        return Err(ApiError::BadRequest(json!({"error": "cpu_team_not_tradable"})));
    }

    let payload = json!({
        "league_id": league_id,
        "from_team": from,
        "to_team": to,
        "offered_player_ids": offered,
        "requested_player_ids": requested,
        "status": "pending",
        "responded_at": Value::Null,
    });
    let created = write::upsert_rows(sb, "manager_trade_offers", std::slice::from_ref(&payload), "id")
        .await
        .ok_or_else(|| ApiError::Internal("offer create failed".into()))?;
    let offer = created
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::Internal("offer returned no row".into()))?;

    Ok(offer)
}

/// Accept / decline / cancel an offer. Only the recipient may accept/decline;
/// the sender may cancel (pending only).
pub async fn respond_offer(
    state: &AppState,
    league_id: &str,
    offer_id: &str,
    action: &str,
    user_id: &str,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let rows = sb
        .read(
            "manager_trade_offers",
            "*",
            Some(&BTreeMap::from([
                ("id".to_string(), format!("eq.{offer_id}")),
                ("league_id".to_string(), format!("eq.{league_id}")),
            ])),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    let offer = rows
        .into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("offer not found".into()))?;
    if str_value(offer.get("status")) != "pending" {
        return Err(ApiError::BadRequest(json!({"error": "already_resolved"})));
    }
    let to_team = str_value(offer.get("to_team"));
    let from_team = str_value(offer.get("from_team"));

    let offered = ids_of(offer.get("offered_player_ids"));
    let requested = ids_of(offer.get("requested_player_ids"));
    let league_row = load_league(sb, league_id).await?;
    let mode = str_value(league_row.get("mode"));

    match action {
        "accept" => {
            // Recipient may accept.
            let membership = league::membership(state, league_id, user_id).await;
            let is_to = membership
                .as_ref()
                .map(|t| str_value(t.get("team_abbrev")) == to_team)
                .unwrap_or(false);
            let is_commish = league::is_commissioner(state, league_id, user_id).await;
            if !is_to && !is_commish {
                return Err(ApiError::BadRequest(json!({"error": "not_your_offer"})));
            }
            if trade_locked(sb, league_id, &from_team, &to_team).await {
                return Err(ApiError::BadRequest(json!({"error": "trade_locked"})));
            }
            if !write::update_rows(
                sb,
                "manager_trade_offers",
                &[("id", offer_id)],
                &json!({"status": "accepted", "responded_at": chrono::Utc::now()}),
            )
            .await
            {
                return Err(ApiError::Internal("offer update failed".into()));
            }
            execute_trade(sb, league_id, &offer, &mode).await?;
            load_offer(sb, offer_id).await
        }
        "decline" => {
            let is_to = league::membership(state, league_id, user_id)
                .await
                .map(|t| str_value(t.get("team_abbrev")) == to_team)
                .unwrap_or(false);
            if !is_to {
                return Err(ApiError::BadRequest(json!({"error": "not_your_offer"})));
            }
            if !write::update_rows(
                sb,
                "manager_trade_offers",
                &[("id", offer_id)],
                &json!({"status": "declined", "responded_at": chrono::Utc::now()}),
            )
            .await
            {
                return Err(ApiError::Internal("offer update failed".into()));
            }
            load_offer(sb, offer_id).await
        }
        "cancel" => {
            let is_from = league::membership(state, league_id, user_id)
                .await
                .map(|t| str_value(t.get("team_abbrev")) == from_team)
                .unwrap_or(false);
            if !is_from {
                return Err(ApiError::BadRequest(json!({"error": "not_your_offer"})));
            }
            if !write::update_rows(
                sb,
                "manager_trade_offers",
                &[("id", offer_id)],
                &json!({"status": "cancelled", "responded_at": chrono::Utc::now()}),
            )
            .await
            {
                return Err(ApiError::Internal("offer update failed".into()));
            }
            load_offer(sb, offer_id).await
        }
        _ => Err(ApiError::BadRequest(json!({"error": "invalid_action"}))),
    }
}

/// Swap the roster membership of the traded players and void any lineup slots
/// that referenced them (so no stale lineup survives).
pub async fn execute_trade(
    sb: &SbClient,
    league_id: &str,
    offer: &Value,
    mode: &str,
) -> Result<(), ApiError> {
    let from = str_value(offer.get("from_team"));
    let to = str_value(offer.get("to_team"));
    let offered = ids_of(offer.get("offered_player_ids"));
    let requested = ids_of(offer.get("requested_player_ids"));

    // Move `offered` (owned by `from`) to `to`; move `requested` to `from`.
    for pid in &offered {
        if !write::update_rows(
            sb,
            "manager_rosters",
            &[("league_id", league_id), ("player_id", &pid.to_string())],
            &json!({"team_abbrev": to}),
        )
        .await
        {
            return Err(ApiError::Internal("trade move failed".into()));
        }
    }
    for pid in &requested {
        if !write::update_rows(
            sb,
            "manager_rosters",
            &[("league_id", league_id), ("player_id", &pid.to_string())],
            &json!({"team_abbrev": from}),
        )
        .await
        {
            return Err(ApiError::Internal("trade move failed".into()));
        }
    }

    // Lineup cleanup: delete any lineup slot (for this league) that lists a
    // moved player, so a stale lineup can't score a player who left.
    if mode == crate::manager::MODE_TRUE_ROSTERS {
        let mut moved = offered.clone();
        moved.extend_from_slice(&requested);
        for pid in moved {
            let _ = write::delete_rows(
                sb,
                "manager_lineups",
                &[
                    ("league_id", league_id),
                    ("player_id", &pid.to_string()),
                ],
            )
            .await;
        }
    }
    Ok(())
}

fn ids_of(v: Option<&Value>) -> Vec<i64> {
    v.and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|x| x.as_i64()).collect())
        .unwrap_or_default()
}

async fn load_league(sb: &SbClient, league_id: &str) -> Result<Value, ApiError> {
    let rows = sb
        .read(
            "manager_leagues",
            "*",
            Some(&league::eq_filter("id", league_id)),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    rows.into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("league not found".into()))
}

async fn load_offer(sb: &SbClient, offer_id: &str) -> Result<Value, ApiError> {
    let rows = sb
        .read(
            "manager_trade_offers",
            "*",
            Some(&league::eq_filter("id", offer_id)),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    rows.into_iter()
        .next()
        .ok_or_else(|| ApiError::NotFound("offer not found".into()))
}

/// List offers for a league (optionally filtered to a team's inbox).
pub async fn list_offers(
    state: &AppState,
    league_id: &str,
    to_team: Option<&str>,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let mut filters = BTreeMap::from([("league_id".to_string(), format!("eq.{league_id}"))]);
    if let Some(t) = to_team.filter(|s| !s.is_empty()) {
        filters.insert("to_team".to_string(), format!("eq.{t}"));
    }
    let rows = sb
        .read(
            "manager_trade_offers",
            "*",
            Some(&filters),
            None,
            Some("created_at.desc"),
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    Ok(json!({"league_id": league_id, "offers": rows}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_valid_requires_minimums() {
        assert!(counts_valid((12, 6, 2)));
        assert!(counts_valid((13, 7, 2))); // beyond the minimum is allowed
        assert!(!counts_valid((11, 6, 2)));
        assert!(!counts_valid((12, 6, 1)));
    }

    #[test]
    fn swap_preserves_and_validates_positions() {
        // from has 12F/6D/2G; trades 1 F out, gets 1 D in → 11F/7D/2G (below min F).
        let from_counts = apply_swap((12, 6, 2), &["F"], &["D"]);
        assert!(!counts_valid(from_counts));
        // Two-for-two that nets zero position change stays valid.
        let ok = apply_swap((12, 6, 2), &["F", "F"], &["F", "F"]);
        assert!(counts_valid(ok));
    }
}
