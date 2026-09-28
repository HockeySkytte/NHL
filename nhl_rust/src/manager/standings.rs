//! Manager Game standings, league feed, and mode-1 roster seeding.

use std::collections::BTreeMap;

use serde_json::{json, Value};

use crate::manager::league;
use crate::state::AppState;
use crate::supabase::read::SbClient;
use crate::util::parse::{safe_int, str_value};

/// Compute W/L standings from `manager_games` rows (final games only).
/// W/L only — no OT, no ties (the winner is already resolved per game).
pub fn standings_from_games(games: &[Value]) -> Vec<Value> {
    let mut w: BTreeMap<String, i64> = BTreeMap::new();
    let mut l: BTreeMap<String, i64> = BTreeMap::new();
    let mut gf: BTreeMap<String, i64> = BTreeMap::new();
    let mut ga: BTreeMap<String, i64> = BTreeMap::new();
    let mut gp: BTreeMap<String, i64> = BTreeMap::new();

    for g in games {
        if str_value(g.get("status")) != "final" {
            continue;
        }
        let home = str_value(g.get("home_abbrev"));
        let away = str_value(g.get("away_abbrev"));
        if home.is_empty() || away.is_empty() {
            continue;
        }
        let hs = parse_locale_float_val(g.get("home_score")).unwrap_or(0.0);
        let as_ = parse_locale_float_val(g.get("away_score")).unwrap_or(0.0);

        *gp.entry(home.clone()).or_default() += 1;
        *gp.entry(away.clone()).or_default() += 1;
        *gf.entry(home.clone()).or_default() += hs.round() as i64;
        *ga.entry(home.clone()).or_default() += as_.round() as i64;
        *gf.entry(away.clone()).or_default() += as_.round() as i64;
        *ga.entry(away.clone()).or_default() += hs.round() as i64;

        let (winner, loser) = if hs > as_ { (home, away) } else { (away, home) };
        *w.entry(winner).or_default() += 1;
        *l.entry(loser).or_default() += 1;
    }

    let mut rows: Vec<Value> = Vec::new();
    let mut all_teams: Vec<String> = w
        .keys()
        .chain(l.keys())
        .chain(gf.keys())
        .map(|k| k.clone())
        .collect();
    all_teams.sort();
    for t in all_teams {
        let wins = *w.get(&t).unwrap_or(&0);
        let losses = *l.get(&t).unwrap_or(&0);
        let pct = if wins + losses > 0 {
            wins as f64 / (wins + losses) as f64
        } else {
            0.0
        };
        rows.push(json!({
            "team_abbrev": t,
            "wins": wins,
            "losses": losses,
            "games_played": *gp.get(&t).unwrap_or(&0),
            "win_pct": (pct * 1000.0).round() / 1000.0,
            "goals_for": *gf.get(&t).unwrap_or(&0),
            "goals_against": *ga.get(&t).unwrap_or(&0),
            "goal_diff": gf.get(&t).unwrap_or(&0) - ga.get(&t).unwrap_or(&0),
        }));
    }
    // Sort by win% desc, then wins, then goal diff, then team.
    rows.sort_by(|a, b| {
        let wp = |r: &Value| parse_locale_float_val(r.get("win_pct")).unwrap_or(0.0);
        wp(b)
            .partial_cmp(&wp(a))
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| safe_int(b.get("wins")).unwrap_or(0).cmp(&safe_int(a.get("wins")).unwrap_or(0)))
            .then_with(|| safe_int(b.get("goal_diff")).unwrap_or(0).cmp(&safe_int(a.get("goal_diff")).unwrap_or(0)))
            .then_with(|| str_value(a.get("team_abbrev")).cmp(&str_value(b.get("team_abbrev"))))
    });
    rows
}

fn parse_locale_float_val(v: Option<&Value>) -> Option<f64> {
    crate::util::parse::parse_locale_float(v)
}

/// League standings: read the league's games and compute W/L.
pub async fn league_standings(state: &AppState, league_id: &str) -> Result<Value, ApiErrorT> {
    let sb = league::sb_required(state)?;
    let rows = sb
        .read(
            "manager_games",
            "home_abbrev,away_abbrev,home_score,away_score,status",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            Some("date.asc,nhl_game_id.asc"),
            0,
        )
        .await
        .ok_or_else(|| ApiErrorT::Internal("supabase unavailable".into()))?;
    Ok(json!({
        "league_id": league_id,
        "standings": standings_from_games(&rows),
    }))
}

/// League feed: recent trades and finalized results.
pub async fn league_feed(state: &AppState, league_id: &str, limit: usize) -> Result<Value, ApiErrorT> {
    let sb = league::sb_required(state)?;
    let trades = sb
        .read(
            "manager_trade_offers",
            "created_at,from_team,to_team,offered_player_ids,requested_player_ids,status,responded_at",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            Some("created_at.desc"),
            limit,
        )
        .await
        .unwrap_or_default();
    let games = sb
        .read(
            "manager_games",
            "date,home_abbrev,away_abbrev,home_score,away_score,status,nhl_game_id",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            Some("date.desc,nhl_game_id.desc"),
            limit,
        )
        .await
        .unwrap_or_default();

    let mut feed: Vec<Value> = Vec::new();
    for t in &trades {
        if str_value(t.get("status")) == "pending" {
            continue;
        }
        feed.push(json!({
            "kind": "trade",
            "at": t.get("created_at"),
            "from_team": t.get("from_team"),
            "to_team": t.get("to_team"),
            "offered_player_ids": t.get("offered_player_ids"),
            "requested_player_ids": t.get("requested_player_ids"),
            "status": t.get("status"),
        }));
    }
    for g in &games {
        if str_value(g.get("status")) != "final" {
            continue;
        }
        feed.push(json!({
            "kind": "result",
            "at": g.get("date"),
            "nhl_game_id": g.get("nhl_game_id"),
            "home_abbrev": g.get("home_abbrev"),
            "away_abbrev": g.get("away_abbrev"),
            "home_score": g.get("home_score"),
            "away_score": g.get("away_score"),
        }));
    }
    Ok(json!({"league_id": league_id, "feed": feed}))
}

/// Mode-1 roster seeding: populate `manager_rosters` with each franchise's
/// current NHL roster players (acquired_via = 'initial'). Position F/D/G from
/// the roster pool. Idempotent per player (upsert). Returns the count seeded.
pub async fn seed_true_rosters(state: &AppState, league_id: &str) -> Result<Value, ApiErrorT> {
    let sb = league::sb_required(state)?;
    let league_row = league_meta(sb, league_id).await?;
    if str_value(league_row.get("mode")) != crate::manager::MODE_TRUE_ROSTERS {
        return Err(ApiErrorT::BadRequest(json!({"error": "invalid_mode"})));
    }
    let teams = sb
        .read(
            "manager_league_teams",
            "team_abbrev",
            Some(&league::eq_filter("league_id", league_id)),
            None,
            None,
            0,
        )
        .await
        .ok_or_else(|| ApiErrorT::Internal("supabase unavailable".into()))?;

    let pool = crate::manager::draft::draft_pool(state).await;
    let mut rows: Vec<Value> = Vec::new();
    for t in &teams {
        let team = str_value(t.get("team_abbrev"));
        for (pid, info) in &pool {
            if str_value(info.get("team")) != team {
                continue;
            }
            let pos = str_value(info.get("position"))
                .chars()
                .next()
                .map(|c| c.to_string())
                .unwrap_or_default();
            if !matches!(pos.as_str(), "F" | "D" | "G") {
                continue;
            }
            rows.push(json!({
                "league_id": league_id,
                "team_abbrev": team,
                "player_id": pid,
                "position": pos,
                "acquired_via": "initial",
            }));
        }
    }
    if !rows.is_empty() {
        crate::supabase::write::upsert_rows(
            sb,
            "manager_rosters",
            &rows,
            "league_id,team_abbrev,player_id",
        )
        .await
        .ok_or_else(|| ApiErrorT::Internal("roster seed failed".into()))?;
    }
    Ok(json!({"league_id": league_id, "seeded": rows.len()}))
}

async fn league_meta(sb: &SbClient, league_id: &str) -> Result<Value, ApiErrorT> {
    let rows = sb
        .read(
            "manager_leagues",
            "id,mode,status",
            Some(&league::eq_filter("id", league_id)),
            None,
            None,
            1,
        )
        .await
        .ok_or_else(|| ApiErrorT::Internal("supabase unavailable".into()))?;
    rows.into_iter()
        .next()
        .ok_or_else(|| ApiErrorT::NotFound("league not found".into()))
}

/// Alias to keep the module self-contained.
type ApiErrorT = crate::error::ApiError;

#[cfg(test)]
mod tests {
    use super::*;

    fn final_game(home: &str, away: &str, hs: f64, as_: f64) -> Value {
        json!({
            "status": "final",
            "home_abbrev": home,
            "away_abbrev": away,
            "home_score": hs,
            "away_score": as_,
        })
    }

    #[test]
    fn standings_are_win_loss_and_sorted() {
        let games = vec![
            final_game("ANA", "BOS", 3.0, 2.0),
            final_game("ANA", "CAR", 2.0, 1.0),
            final_game("BOS", "CAR", 4.0, 2.0),
        ];
        let rows = standings_from_games(&games);
        let by = |t: &str| rows.iter().find(|r| str_value(r.get("team_abbrev")) == t).cloned();
        let ana = by("ANA").unwrap();
        assert_eq!(safe_int(ana.get("wins")), Some(2));
        assert_eq!(safe_int(ana.get("losses")), Some(0));
        let car = by("CAR").unwrap();
        assert_eq!(safe_int(car.get("wins")), Some(0));
        assert_eq!(safe_int(car.get("losses")), Some(2));
        // Sorted by win% desc: ANA (1.0) first.
        assert_eq!(str_value(rows[0].get("team_abbrev")), "ANA");
    }

    #[test]
    fn pending_games_are_excluded() {
        let games = vec![
            json!({"status": "pending", "home_abbrev": "ANA", "away_abbrev": "BOS"}),
            final_game("ANA", "CAR", 1.0, 1.0),
        ];
        let rows = standings_from_games(&games);
        let ana = rows.iter().find(|r| str_value(r.get("team_abbrev")) == "ANA").unwrap();
        assert_eq!(safe_int(ana.get("games_played")), Some(1));
    }
}
