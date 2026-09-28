//! League schedule: the real NHL schedule turned into manager-game rows with
//! per-team round indices ("home game 4 vs away game 3" is a normal state).
//!
//! Rounds are per franchise: round *k* of a team = its *k*-th regular-season
//! game of the season (sorted by date). The round window for round *k* is the
//! calendar span from the previous franchise game (exclusive) through game *k*
//! (inclusive) — the window in which lineup players earn points.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::data::projections;
use crate::error::ApiError;
use crate::state::AppState;
use crate::util::parse::{safe_int, str_value};

/// Assign round indices 1..N to a team's games (already sorted by date).
pub fn assign_rounds(games: &[Value]) -> Vec<Value> {
    games
        .iter()
        .enumerate()
        .map(|(i, g)| {
            let mut g2 = g.clone();
            g2["round"] = json!(i as i64 + 1);
            g2
        })
        .collect()
}

/// Round window for round `round` of a team: `(start_exclusive, end_inclusive)`
/// dates. Round 1's window has no start bound (opens at league creation).
/// Returns `None` for invalid/out-of-range rounds or missing dates.
pub fn round_window(games: &[Value], round: usize) -> Option<(Option<String>, String)> {
    if round == 0 || round > games.len() {
        return None;
    }
    let end = str_value(games.get(round - 1)?.get("date"));
    if end.is_empty() {
        return None;
    }
    let start = if round >= 2 {
        let prev = str_value(games.get(round - 2)?.get("date"));
        if prev.is_empty() {
            None
        } else {
            Some(prev)
        }
    } else {
        None
    };
    Some((start, end))
}

/// Pure core: build manager-game rows from per-team regular-season game lists
/// (each list sorted by date). Games are deduped by NHL game id; each row
/// carries each side's own round index.
pub fn league_rows_from_team_games(team_games: &BTreeMap<String, Vec<Value>>) -> Vec<Value> {
    // Round index per (team, date).
    let mut rounds: BTreeMap<String, BTreeMap<String, i64>> = BTreeMap::new();
    for (team, games) in team_games {
        let mut m = BTreeMap::new();
        for (i, g) in games.iter().enumerate() {
            let date = str_value(g.get("date"));
            if !date.is_empty() {
                m.insert(date, i as i64 + 1);
            }
        }
        rounds.insert(team.clone(), m);
    }

    // Dedup games by id, keep one copy each.
    let mut by_id: BTreeMap<i64, Value> = BTreeMap::new();
    for games in team_games.values() {
        for g in games {
            if let Some(id) = safe_int(g.get("id")) {
                by_id.entry(id).or_insert_with(|| g.clone());
            }
        }
    }
    let mut all: Vec<Value> = by_id.into_values().collect();
    all.sort_by(|a, b| {
        str_value(a.get("date"))
            .cmp(&str_value(b.get("date")))
            .then_with(|| safe_int(a.get("id")).unwrap_or(0).cmp(&safe_int(b.get("id")).unwrap_or(0)))
    });

    let mut rows: Vec<Value> = Vec::new();
    for g in all {
        let home = str_value(g.get("home"));
        let away = str_value(g.get("away"));
        let date = str_value(g.get("date"));
        let id = safe_int(g.get("id")).unwrap_or(0);
        let home_round = rounds
            .get(&home)
            .and_then(|m| m.get(&date))
            .copied()
            .unwrap_or(0);
        let away_round = rounds
            .get(&away)
            .and_then(|m| m.get(&date))
            .copied()
            .unwrap_or(0);
        if id == 0 || home_round == 0 || away_round == 0 || home.is_empty() || away.is_empty() {
            continue;
        }
        rows.push(json!({
            "nhl_game_id": id,
            "date": date,
            "home_abbrev": home,
            "away_abbrev": away,
            "home_round": home_round,
            "away_round": away_round,
            "status": "pending",
        }));
    }
    rows
}

/// Fetch every active franchise's regular-season schedule for `season` and
/// build the league's manager-game rows (league_id is added by the caller).
pub async fn build_league_schedule_rows(
    state: &AppState,
    season: i64,
) -> Result<Vec<Value>, ApiError> {
    let teams = projections::active_team_abbrevs(state);
    if teams.is_empty() {
        return Err(ApiError::Internal("no active teams".into()));
    }

    let mut team_games: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for team in &teams {
        let games = projections::fetch_club_schedule_games(state, team, season).await;
        let mut reg: Vec<Value> = games
            .into_iter()
            .filter(|g| safe_int(g.get("gameType")).unwrap_or(0) == 2)
            .collect();
        reg.sort_by(|a, b| {
            str_value(a.get("date"))
                .cmp(&str_value(b.get("date")))
                .then_with(|| {
                    safe_int(a.get("id"))
                        .unwrap_or(0)
                        .cmp(&safe_int(b.get("id")).unwrap_or(0))
                })
        });
        if reg.is_empty() {
            return Err(ApiError::Internal(format!(
                "no {season} schedule for {team}"
            )));
        }
        team_games.insert(team.clone(), reg);
    }

    Ok(league_rows_from_team_games(&team_games))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn game(id: i64, date: &str, home: &str, away: &str) -> Value {
        json!({"id": id, "gameType": 2, "date": date, "home": home, "away": away})
    }

    #[test]
    fn assign_rounds_indexes_from_one() {
        let games = vec![
            game(1, "2026-10-07", "ANA", "BOS"),
            game(2, "2026-10-09", "ANA", "CAR"),
        ];
        let out = assign_rounds(&games);
        assert_eq!(safe_int(out[0].get("round")), Some(1));
        assert_eq!(safe_int(out[1].get("round")), Some(2));
    }

    #[test]
    fn round_window_opens_unbounded_for_round_one() {
        let games = vec![
            game(1, "2026-10-07", "ANA", "BOS"),
            game(2, "2026-10-09", "ANA", "CAR"),
            game(3, "2026-10-10", "ANA", "DAL"),
        ];
        assert_eq!(
            round_window(&games, 1),
            Some((None, "2026-10-07".to_string()))
        );
        assert_eq!(
            round_window(&games, 2),
            Some((Some("2026-10-07".to_string()), "2026-10-09".to_string()))
        );
        assert_eq!(
            round_window(&games, 3),
            Some((Some("2026-10-09".to_string()), "2026-10-10".to_string()))
        );
        assert_eq!(round_window(&games, 0), None);
        assert_eq!(round_window(&games, 4), None);
    }

    #[test]
    fn league_rows_carry_per_side_round_indices() {
        // ANA is on its 4th game while BOS is on its 3rd (the spec's
        // "game 4 vs game 3" case), and ANA plays CAR twice.
        let mut team_games: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        team_games.insert(
            "ANA".to_string(),
            vec![
                game(10, "2026-10-07", "ANA", "BOS"),  // ANA 1, BOS 1
                game(11, "2026-10-09", "ANA", "CAR"),  // ANA 2, CAR 1
                game(12, "2026-10-11", "ANA", "DAL"),  // ANA 3, DAL 1
                game(13, "2026-10-13", "BOS", "ANA"),  // ANA 4, BOS 3
            ],
        );
        team_games.insert(
            "BOS".to_string(),
            vec![
                game(10, "2026-10-07", "ANA", "BOS"),
                game(20, "2026-10-09", "BOS", "NYR"),
                game(13, "2026-10-13", "BOS", "ANA"),
            ],
        );
        team_games.insert(
            "CAR".to_string(),
            vec![game(11, "2026-10-09", "ANA", "CAR")],
        );
        team_games.insert(
            "DAL".to_string(),
            vec![game(12, "2026-10-11", "ANA", "DAL")],
        );
        team_games.insert("NYR".to_string(), vec![game(20, "2026-10-09", "BOS", "NYR")]);

        let rows = league_rows_from_team_games(&team_games);
        assert_eq!(rows.len(), 5, "one row per distinct NHL game");

        let by_id: BTreeMap<i64, &Value> = rows
            .iter()
            .map(|r| (safe_int(r.get("nhl_game_id")).unwrap(), r))
            .collect();
        assert_eq!(by_id[&13]["home_abbrev"], "BOS");
        assert_eq!(by_id[&13]["away_abbrev"], "ANA");
        // BOS hosts its 3rd game; ANA visits for its 4th.
        assert_eq!(safe_int(by_id[&13].get("home_round")), Some(3));
        assert_eq!(safe_int(by_id[&13].get("away_round")), Some(4));
        assert_eq!(by_id[&10]["date"], "2026-10-07");
    }
}
