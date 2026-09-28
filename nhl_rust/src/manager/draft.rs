//! Draft engine (M2): 20-round snake draft with CPU autopick.
//!
//! Draft picks are `manager_rosters` rows with `acquired_via='draft'`; pick
//! order comes from `picked_at`. Snake order: odd rounds ascend by
//! `draft_slot`, even rounds descend (pick 1 in round 1 picks 32nd in
//! round 2). CPU teams pick the best available player by projected value
//! (existing GM-mode projections) subject to the 12F/6D/2G feasibility rule.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::{Arc, OnceLock};

use serde_json::{json, Value};

use crate::data::{projections, rosters};
use crate::error::ApiError;
use crate::manager::{self, league};
use crate::state::AppState;
use crate::supabase::read::SbClient;
use crate::supabase::write;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

/// Draft length (spec): 26 rounds × 32 teams, with a 12F/6D/2G **minimum**.
pub const DRAFT_ROUNDS: i64 = crate::manager::DRAFT_TOTAL as i64;

/// Fallback values for players without a projection, by position (keeps
/// unknown players below every projected player, F > D > G ordering).
const FALLBACK_VALUE_F: f64 = -0.0246;
const FALLBACK_VALUE_D: f64 = -0.0318;
const FALLBACK_VALUE_G: f64 = -0.12;

/// In-process per-league pick locks: the pick transaction (verify turn →
/// insert → advance → CPU autopick) must be serialized per league.
static LEAGUE_DRAFT_LOCKS: OnceLock<
    tokio::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>,
> = OnceLock::new();

async fn lock_for(league_id: &str) -> Arc<tokio::sync::Mutex<()>> {
    let map =
        LEAGUE_DRAFT_LOCKS.get_or_init(|| tokio::sync::Mutex::new(HashMap::new()));
    let mut m = map.lock().await;
    m.entry(league_id.to_string())
        .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(())))
        .clone()
}

/// Snake order for a round: draft-slot order ascending on odd rounds,
/// reversed on even rounds.
pub fn pick_order(slots_sorted: &[String], round: i64) -> Vec<String> {
    let mut v = slots_sorted.to_vec();
    if round % 2 == 0 {
        v.reverse();
    }
    v
}

/// The team picking at position `pick` (1-based) of `round` in snake order.
pub fn current_picker(slots_sorted: &[String], round: i64, pick: i64) -> Option<String> {
    if round < 1 || pick < 1 || pick as usize > slots_sorted.len() {
        return None;
    }
    pick_order(slots_sorted, round).get(pick as usize - 1).cloned()
}

/// Whether adding one player of `pos` keeps the 12F/6D/2G minimum reachable:
/// the team must have enough picks left to fill any positional deficit after
/// this pick. There is **no per-position cap** (a 26-man roster has bench
/// depth; e.g. extra defenses beyond 6 are allowed).
pub fn feasible_pick(counts: (i64, i64, i64), pos: &str, total_picked: i64) -> bool {
    let (f, d, g) = counts;
    let (f2, d2, g2) = match pos {
        "F" => (f + 1, d, g),
        "D" => (f, d + 1, g),
        "G" => (f, d, g + 1),
        _ => return false,
    };
    let min_f = manager::LINEUP_FORWARDS as i64;
    let min_d = manager::LINEUP_DEFENSE as i64;
    let min_g = manager::LINEUP_GOALIES as i64;
    let picks_left = DRAFT_ROUNDS - total_picked - 1;
    let need = (min_f - f2).max(0) + (min_d - d2).max(0) + (min_g - g2).max(0);
    picks_left >= 0 && picks_left >= need
}

/// CPU pick value: projected value when available, else the positional
/// fallback (unknowns always rank below known quantities).
pub fn cpu_pick_value(proj_row: Option<&Value>, pos: &str) -> f64 {
    if let Some(r) = proj_row {
        if let Some(v) = parse_locale_float(r.get("projected_value")) {
            return v;
        }
    }
    match pos {
        "D" => FALLBACK_VALUE_D,
        "G" => FALLBACK_VALUE_G,
        _ => FALLBACK_VALUE_F,
    }
}

/// Best available feasible player for a CPU team. Ties break on player id.
pub fn cpu_choose(
    pool: &HashMap<i64, Value>,
    proj: &HashMap<i64, Value>,
    taken: &HashSet<i64>,
    counts: (i64, i64, i64),
    total_picked: i64,
) -> Option<i64> {
    let mut best: Vec<(f64, i64)> = Vec::new();
    for (&pid, info) in pool {
        if taken.contains(&pid) {
            continue;
        }
        let pos = str_value(info.get("position"))
            .chars()
            .next()
            .map(|c| c.to_string())
            .unwrap_or_default();
        if !feasible_pick(counts, &pos, total_picked) {
            continue;
        }
        let v = cpu_pick_value(proj.get(&pid), &pos);
        best.push((v, pid));
    }
    best.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    best.first().map(|t| t.1)
}

/// Best available feasible pick for a specific team (used by the dry-run /
/// any "auto-pick for a human" helper). Reads live roster state.
pub async fn best_pick_for(state: &AppState, league_id: &str, team: &str) -> Option<i64> {
    let sb = league::sb_required(state).ok()?;
    let rosters = read_rosters(sb, league_id).await.ok()?;
    let taken: HashSet<i64> = rosters
        .iter()
        .filter_map(|r| safe_int(r.get("player_id")))
        .collect();
    let mut counts = (0i64, 0i64, 0i64);
    for r in &rosters {
        if str_value(r.get("team_abbrev")) == team {
            match str_value(r.get("position")).as_str() {
                "F" => counts.0 += 1,
                "D" => counts.1 += 1,
                "G" => counts.2 += 1,
                _ => {}
            }
        }
    }
    let total = counts.0 + counts.1 + counts.2;
    let pool = draft_pool(state).await;
    let proj = crate::data::projections::load_gm_mode_projections_cached(state).await;
    cpu_choose(&pool, &proj, &taken, counts, total)
}

/// The draft pool: player id → bios record (all current rosters).
pub async fn draft_pool(state: &AppState) -> HashMap<i64, Value> {
    let value = rosters::all_rosters(&state.caches, &state.http).await;
    let mut out = HashMap::new();
    if let Some(obj) = value.as_object() {
        for (k, v) in obj {
            if let Ok(pid) = k.parse::<i64>() {
                if pid > 0 {
                    out.insert(pid, v.clone());
                }
            }
        }
    }
    out
}

fn pos_of(info: &Value) -> String {
    str_value(info.get("position"))
        .chars()
        .next()
        .map(|c| c.to_string())
        .unwrap_or_default()
}

async fn read_league_row(sb: &SbClient, league_id: &str) -> Result<Value, ApiError> {
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

async fn read_teams_sorted(sb: &SbClient, league_id: &str) -> Result<Vec<Value>, ApiError> {
    let rows = sb
        .read(
            "manager_league_teams",
            "*",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            Some("draft_slot.asc"),
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    Ok(rows)
}

async fn read_rosters(sb: &SbClient, league_id: &str) -> Result<Vec<Value>, ApiError> {
    let rows = sb
        .read(
            "manager_rosters",
            "team_abbrev,player_id,position,picked_at",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            Some("picked_at.asc"),
            0,
        )
        .await
        .ok_or_else(|| ApiError::Internal("supabase unavailable".into()))?;
    Ok(rows)
}

/// Full draft state for the UI: order, counts, current picker, taken map,
/// recent picks.
pub async fn draft_state(
    state: &AppState,
    league_id: &str,
    viewer_user_id: Option<&str>,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let league_row = read_league_row(sb, league_id).await?;
    let mode = str_value(league_row.get("mode"));
    let status = str_value(league_row.get("status"));
    let draft_round = safe_int(league_row.get("draft_round")).unwrap_or(0);
    let draft_pick = safe_int(league_row.get("draft_pick")).unwrap_or(0);

    let teams = read_teams_sorted(sb, league_id).await?;
    let rosters = read_rosters(sb, league_id).await?;

    let pool = draft_pool(state).await;

    let mut counts: BTreeMap<String, (i64, i64, i64)> = BTreeMap::new();
    let mut taken: BTreeMap<i64, String> = BTreeMap::new();
    let mut recent: Vec<Value> = Vec::new();
    for r in &rosters {
        let team = str_value(r.get("team_abbrev"));
        let pid = safe_int(r.get("player_id")).unwrap_or(0);
        let pos = str_value(r.get("position"));
        let e = counts.entry(team.clone()).or_insert((0, 0, 0));
        match pos.as_str() {
            "F" => e.0 += 1,
            "D" => e.1 += 1,
            "G" => e.2 += 1,
            _ => {}
        }
        if pid > 0 {
            taken.insert(pid, team.clone());
            if recent.len() < 16 {
                let name = pool
                    .get(&pid)
                    .map(|i| str_value(i.get("name")))
                    .unwrap_or_default();
                recent.push(json!({
                    "team_abbrev": team,
                    "player_id": pid,
                    "name": name,
                    "position": pos,
                }));
            }
        }
    }
    recent.reverse();

    let slots: Vec<String> = teams
        .iter()
        .map(|t| str_value(t.get("team_abbrev")))
        .collect();
    let complete = draft_round > DRAFT_ROUNDS
        || status == manager::STATUS_RUNNING
        || status == manager::STATUS_FINISHED;
    let picker = if status == manager::STATUS_DRAFTING && !complete {
        current_picker(&slots, draft_round, draft_pick)
    } else {
        None
    };

    let mut order: Vec<Value> = Vec::new();
    for t in &teams {
        let abbrev = str_value(t.get("team_abbrev"));
        let (f, d, g) = counts.get(&abbrev).copied().unwrap_or((0, 0, 0));
        order.push(json!({
            "team_abbrev": abbrev,
            "user_id": t.get("user_id"),
            "team_name": t.get("team_name"),
            "draft_slot": t.get("draft_slot"),
            "roster_count": f + d + g,
            "f": f, "d": d, "g": g,
            "is_cpu": t.get("user_id").map(|v| v.is_null()).unwrap_or(true),
        }));
    }

    let my_team = viewer_user_id
        .and_then(|uid| {
            teams
                .iter()
                .find(|t| str_value(t.get("user_id")) == uid)
                .map(|t| str_value(t.get("team_abbrev")))
        })
        .filter(|s| !s.is_empty());

    Ok(json!({
        "league_id": league_id,
        "mode": mode,
        "status": status,
        "draft_round": draft_round,
        "draft_pick": draft_pick,
        "draft_paused": league_row.get("draft_paused"),
        "complete": complete,
        "order": order,
        "current_picker": picker,
        "my_team": my_team,
        "recent_picks": recent,
        "taken": taken,
        "rounds_total": DRAFT_ROUNDS,
    }))
}

/// Advance (round, pick) after a pick; when the draft completes, round is
/// set past DRAFT_ROUNDS (sentinel) and pick resets to 0.
fn advance(round: i64, pick: i64, n_teams: usize) -> (i64, i64) {
    let mut r = round;
    let mut p = pick + 1;
    if p as usize > n_teams {
        r += 1;
        p = 1;
    }
    (r, p)
}

/// CPU autopick: keep picking for CPU teams until the next human's turn or
/// draft completion. Callers must hold the league lock.
async fn cpu_autopick_loop(state: &AppState, sb: &SbClient, league_id: &str) -> Result<(), ApiError> {
    let pool = draft_pool(state).await;
    let proj = projections::load_gm_mode_projections_cached(state).await;

    loop {
        let league_row = read_league_row(sb, league_id).await?;
        if str_value(league_row.get("status")) != manager::STATUS_DRAFTING {
            return Ok(());
        }
        let round = safe_int(league_row.get("draft_round")).unwrap_or(0);
        let pick = safe_int(league_row.get("draft_pick")).unwrap_or(0);
        if round < 1 || round > DRAFT_ROUNDS {
            return Ok(());
        }
        let teams = read_teams_sorted(sb, league_id).await?;
        let slots: Vec<String> = teams
            .iter()
            .map(|t| str_value(t.get("team_abbrev")))
            .collect();
        let Some(picker) = current_picker(&slots, round, pick) else {
            return Ok(());
        };
        let picker_row = teams
            .iter()
            .find(|t| str_value(t.get("team_abbrev")) == picker);
        let is_cpu = picker_row
            .and_then(|t| t.get("user_id"))
            .map(|v| v.is_null())
            .unwrap_or(true);
        if !is_cpu {
            return Ok(()); // human's turn — stop here
        }

        let rosters = read_rosters(sb, league_id).await?;
        let taken: HashSet<i64> = rosters
            .iter()
            .filter_map(|r| safe_int(r.get("player_id")))
            .collect();
        let mut counts = (0i64, 0i64, 0i64);
        for r in &rosters {
            if str_value(r.get("team_abbrev")) == picker {
                match str_value(r.get("position")).as_str() {
                    "F" => counts.0 += 1,
                    "D" => counts.1 += 1,
                    "G" => counts.2 += 1,
                    _ => {}
                }
            }
        }
        let total_picked = counts.0 + counts.1 + counts.2;
        let Some(pid) = cpu_choose(&pool, &proj, &taken, counts, total_picked) else {
            return Ok(());
        };
        let pos = pos_of(pool.get(&pid).unwrap_or(&Value::Null));

        let row = json!({
            "league_id": league_id,
            "team_abbrev": picker,
            "player_id": pid,
            "position": pos,
            "acquired_via": "draft",
        });
        if write::upsert_rows(sb, "manager_rosters", std::slice::from_ref(&row), "league_id,team_abbrev,player_id")
            .await
            .is_none()
        {
            return Err(ApiError::Internal("draft insert failed".into()));
        }

        let (nr, np) = advance(round, pick, slots.len());
        let patch = if nr > DRAFT_ROUNDS {
            json!({"draft_round": nr, "draft_pick": 0})
        } else {
            json!({"draft_round": nr, "draft_pick": np})
        };
        if !write::update_rows(sb, "manager_leagues", &[("id", league_id)], &patch).await {
            return Err(ApiError::Internal("draft advance failed".into()));
        }
    }
}

/// Commissioner starts the draft (mode=draft, status=setup): clears any
/// previous draft picks, resets counters, and lets CPU teams pick until the
/// first human's turn.
pub async fn start_draft(
    state: &AppState,
    league_id: &str,
    user_id: &str,
) -> Result<Value, ApiError> {
    if !league::is_commissioner(state, league_id, user_id).await {
        return Err(ApiError::BadRequest(json!({"error": "not_commissioner"})));
    }
    let lk = lock_for(league_id).await;
    let _guard = lk.lock().await;
    let sb = league::sb_required(state)?;

    let league_row = read_league_row(sb, league_id).await?;
    if str_value(league_row.get("mode")) != manager::MODE_DRAFT {
        return Err(ApiError::BadRequest(json!({"error": "invalid_mode"})));
    }
    match str_value(league_row.get("status")).as_str() {
        manager::STATUS_DRAFTING => {
            return Err(ApiError::BadRequest(json!({"error": "draft_started"})));
        }
        manager::STATUS_RUNNING | manager::STATUS_FINISHED => {
            return Err(ApiError::BadRequest(json!({"error": "season_started"})));
        }
        _ => {}
    }

    // Reset previous draft state (re-draft before the season).
    let _ = write::delete_rows(
        sb,
        "manager_rosters",
        &[("league_id", league_id), ("acquired_via", "draft")],
    )
    .await;
    if !write::update_rows(
        sb,
        "manager_leagues",
        &[("id", league_id)],
        &json!({"status": "drafting", "draft_round": 1, "draft_pick": 1}),
    )
    .await
    {
        return Err(ApiError::Internal("draft start failed".into()));
    }

    cpu_autopick_loop(state, sb, league_id).await?;
    draft_state(state, league_id, Some(user_id)).await
}

/// Submit a human pick. Validates turn order, availability, and positional
/// feasibility, then advances and lets CPU teams pick until the next human.
pub async fn submit_pick(
    state: &AppState,
    league_id: &str,
    user_id: &str,
    player_id: i64,
) -> Result<Value, ApiError> {
    if player_id <= 0 {
        return Err(ApiError::BadRequest(json!({"error": "invalid_player"})));
    }
    let lk = lock_for(league_id).await;
    let _guard = lk.lock().await;
    let sb = league::sb_required(state)?;

    let league_row = read_league_row(sb, league_id).await?;
    if str_value(league_row.get("mode")) != manager::MODE_DRAFT {
        return Err(ApiError::BadRequest(json!({"error": "invalid_mode"})));
    }
    if str_value(league_row.get("status")) != manager::STATUS_DRAFTING {
        return Err(ApiError::BadRequest(json!({"error": "draft_not_active"})));
    }
    let round = safe_int(league_row.get("draft_round")).unwrap_or(0);
    let pick = safe_int(league_row.get("draft_pick")).unwrap_or(0);
    if round < 1 || round > DRAFT_ROUNDS {
        return Err(ApiError::BadRequest(json!({"error": "draft_complete"})));
    }

    let teams = read_teams_sorted(sb, league_id).await?;
    let slots: Vec<String> = teams
        .iter()
        .map(|t| str_value(t.get("team_abbrev")))
        .collect();
    let picker = current_picker(&slots, round, pick)
        .ok_or_else(|| ApiError::BadRequest(json!({"error": "draft_not_active"})))?;
    let my_team = teams
        .iter()
        .find(|t| str_value(t.get("user_id")) == user_id)
        .map(|t| str_value(t.get("team_abbrev")))
        .filter(|s| !s.is_empty())
        .ok_or_else(|| ApiError::BadRequest(json!({"error": "not_member"})))?;
    if my_team != picker {
        return Err(ApiError::BadRequest(json!({"error": "not_your_turn"})));
    }

    let rosters = read_rosters(sb, league_id).await?;
    let taken: HashSet<i64> = rosters
        .iter()
        .filter_map(|r| safe_int(r.get("player_id")))
        .collect();
    if taken.contains(&player_id) {
        return Err(ApiError::BadRequest(json!({"error": "player_taken"})));
    }

    let pool = draft_pool(state).await;
    let Some(info) = pool.get(&player_id) else {
        return Err(ApiError::BadRequest(json!({"error": "invalid_player"})));
    };
    let pos = pos_of(info);
    if pos.is_empty() || !matches!(pos.as_str(), "F" | "D" | "G") {
        return Err(ApiError::BadRequest(json!({"error": "invalid_player"})));
    }

    let mut counts = (0i64, 0i64, 0i64);
    for r in &rosters {
        if str_value(r.get("team_abbrev")) == my_team {
            match str_value(r.get("position")).as_str() {
                "F" => counts.0 += 1,
                "D" => counts.1 += 1,
                "G" => counts.2 += 1,
                _ => {}
            }
        }
    }
    let total_picked = counts.0 + counts.1 + counts.2;
    if !feasible_pick(counts, &pos, total_picked) {
        return Err(ApiError::BadRequest(json!({"error": "invalid_pick_position"})));
    }

    let row = json!({
        "league_id": league_id,
        "team_abbrev": my_team,
        "player_id": player_id,
        "position": pos,
        "acquired_via": "draft",
    });
    if write::upsert_rows(
        sb,
        "manager_rosters",
        std::slice::from_ref(&row),
        "league_id,team_abbrev,player_id",
    )
    .await
    .is_none()
    {
        return Err(ApiError::Internal("draft insert failed".into()));
    }

    let (nr, np) = advance(round, pick, slots.len());
    let patch = if nr > DRAFT_ROUNDS {
        json!({"draft_round": nr, "draft_pick": 0})
    } else {
        json!({"draft_round": nr, "draft_pick": np})
    };
    if !write::update_rows(sb, "manager_leagues", &[("id", league_id)], &patch).await {
        return Err(ApiError::Internal("draft advance failed".into()));
    }

    cpu_autopick_loop(state, sb, league_id).await?;
    draft_state(state, league_id, Some(user_id)).await
}

/// Available (undrafted) players with projection values, for the draft room.
pub async fn available_players(
    state: &AppState,
    league_id: &str,
    search: &str,
    pos_filter: &str,
    team_filter: &str,
    limit: usize,
    offset: usize,
) -> Result<Value, ApiError> {
    let sb = league::sb_required(state)?;
    let rosters = read_rosters(sb, league_id).await?;
    let taken: HashSet<i64> = rosters
        .iter()
        .filter_map(|r| safe_int(r.get("player_id")))
        .collect();

    let pool = draft_pool(state).await;
    let proj = projections::load_gm_mode_projections_cached(state).await;

    let q = search.trim().to_lowercase();
    let pf = pos_filter.trim().to_uppercase();
    let tf = team_filter.trim().to_uppercase();

    let mut rows: Vec<(f64, i64, Value)> = Vec::new();
    for (&pid, info) in &pool {
        if taken.contains(&pid) {
            continue;
        }
        let pos = pos_of(info);
        let name = str_value(info.get("name"));
        let team = str_value(info.get("team"));
        if !pf.is_empty() && pos != pf {
            continue;
        }
        if !tf.is_empty() && team.to_uppercase() != tf {
            continue;
        }
        if !q.is_empty() && !name.to_lowercase().contains(&q) {
            continue;
        }
        let value = cpu_pick_value(proj.get(&pid), &pos);
        rows.push((
            value,
            pid,
            json!({
                "player_id": pid,
                "name": name,
                "position": pos,
                "team": team,
                "projected_value": proj
                    .get(&pid)
                    .and_then(|r| parse_locale_float(r.get("projected_value"))),
            }),
        ));
    }
    rows.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let total = rows.len();
    let page: Vec<Value> = rows
        .into_iter()
        .skip(offset)
        .take(limit.max(1).min(200))
        .map(|(_, _, v)| v)
        .collect();
    Ok(json!({"league_id": league_id, "players": page, "total": total}))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snake_order_reverses_even_rounds() {
        let slots: Vec<String> = vec!["A", "B", "C", "D"]
            .into_iter()
            .map(String::from)
            .collect();
        assert_eq!(
            pick_order(&slots, 1),
            vec!["A", "B", "C", "D"]
        );
        assert_eq!(
            pick_order(&slots, 2),
            vec!["D", "C", "B", "A"]
        );
        assert_eq!(
            pick_order(&slots, 3),
            vec!["A", "B", "C", "D"]
        );
        // Pick 1 in round 1 drafts 32nd in round 2 (spec).
        let slots32: Vec<String> = (1..=32).map(|i| format!("T{i:02}")).collect();
        assert_eq!(current_picker(&slots32, 1, 1).as_deref(), Some("T01"));
        assert_eq!(current_picker(&slots32, 2, 32).as_deref(), Some("T01"));
        assert_eq!(current_picker(&slots32, 2, 1).as_deref(), Some("T32"));
    }

    #[test]
    fn feasibility_requires_reaching_minimums() {
        // Early picks are always feasible (26 rounds, 12/6/2 minimum).
        assert!(feasible_pick((0, 0, 0), "F", 0));
        assert!(feasible_pick((0, 0, 0), "G", 0));
        // No per-position cap: drafting beyond the 12/6/2 minimum is allowed.
        assert!(feasible_pick((12, 6, 2), "F", 20));
        // Always keep enough picks to reach a deficit.
        assert!(feasible_pick((11, 6, 2), "F", 19));
        // Last pick must fill the remaining deficit (0 picks to spare).
        assert!(!feasible_pick((16, 8, 1), "F", 25)); // leaves 0 G with 0 picks
        assert!(feasible_pick((16, 8, 1), "G", 25)); // closes the G deficit
        // Unknown position is invalid.
        assert!(!feasible_pick((0, 0, 0), "X", 0));
    }

    #[test]
    fn cpu_values_rank_known_above_unknown() {
        let known = json!({"projected_value": 1.5});
        assert_eq!(cpu_pick_value(Some(&known), "F"), 1.5);
        assert!(cpu_pick_value(None, "F") > cpu_pick_value(None, "D"));
        assert!(cpu_pick_value(None, "D") > cpu_pick_value(None, "G"));
    }

    #[test]
    fn cpu_choose_picks_best_feasible() {
        let mut pool = HashMap::new();
        pool.insert(1, json!({"position": "F", "name": "P1"}));
        pool.insert(2, json!({"position": "G", "name": "P2"}));
        pool.insert(3, json!({"position": "D", "name": "P3"}));
        let proj: HashMap<i64, Value> = HashMap::from([
            (1, json!({"projected_value": 2.0})),
            (2, json!({"projected_value": 5.0})),
            (3, json!({"projected_value": 3.0})),
        ]);
        let taken: HashSet<i64> = HashSet::new();
        // Best by value, feasible.
        assert_eq!(cpu_choose(&pool, &proj, &taken, (0, 0, 0), 0), Some(2));
        // Taken players are skipped.
        let taken2: HashSet<i64> = HashSet::from([2]);
        assert_eq!(cpu_choose(&pool, &proj, &taken2, (0, 0, 0), 0), Some(3));
        // No per-position cap: a third goalie is allowed (bench depth).
        assert_eq!(cpu_choose(&pool, &proj, &taken, (0, 0, 2), 2), Some(2));
        // Late-pick deficit narrows to the required position.
        assert_eq!(cpu_choose(&pool, &proj, &taken, (16, 8, 0), 24), Some(2));
    }

    #[test]
    fn advance_wraps_rounds() {
        assert_eq!(advance(1, 1, 32), (1, 2));
        assert_eq!(advance(1, 32, 32), (2, 1));
        assert_eq!(advance(20, 32, 32), (21, 1));
    }
}
