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
    #[derive(Default)]
    struct TeamLine {
        wins: i64,
        losses: i64,
        gp: i64,
        gf: i64,
        ga: i64,
    }
    let mut stats: BTreeMap<String, TeamLine> = BTreeMap::new();

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
        let h_goals = hs.round() as i64;
        let a_goals = as_.round() as i64;
        let home_won = hs > as_;

        let he = stats.entry(home).or_default();
        he.gp += 1;
        he.gf += h_goals;
        he.ga += a_goals;
        if home_won { he.wins += 1; } else { he.losses += 1; }

        let ae = stats.entry(away).or_default();
        ae.gp += 1;
        ae.gf += a_goals;
        ae.ga += h_goals;
        if home_won { ae.losses += 1; } else { ae.wins += 1; }
    }

    let mut rows: Vec<Value> = Vec::with_capacity(stats.len());
    for (t, s) in stats {
        let pct = if s.wins + s.losses > 0 {
            s.wins as f64 / (s.wins + s.losses) as f64
        } else {
            0.0
        };
        rows.push(json!({
            "team_abbrev": t,
            "wins": s.wins,
            "losses": s.losses,
            "games_played": s.gp,
            "win_pct": (pct * 1000.0).round() / 1000.0,
            "goals_for": s.gf,
            "goals_against": s.ga,
            "goal_diff": s.gf - s.ga,
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
            .then_with(|| {
                let ta = a.get("team_abbrev").and_then(Value::as_str).unwrap_or("");
                let tb = b.get("team_abbrev").and_then(Value::as_str).unwrap_or("");
                ta.cmp(tb)
            })
    });
    rows
}

fn parse_locale_float_val(v: Option<&Value>) -> Option<f64> {
    crate::util::parse::parse_locale_float(v)
}

/// League standings: read only finalized games (filtered in SQL) and compute W/L.
pub async fn league_standings(state: &AppState, league_id: &str) -> Result<Value, ApiErrorT> {
    let sb = league::sb_required(state)?;
    let filters = BTreeMap::from([
        ("league_id".to_string(), format!("eq.{league_id}")),
        ("status".to_string(), "eq.final".to_string()),
    ]);
    let rows = sb
        .read(
            "manager_games",
            "home_abbrev,away_abbrev,home_score,away_score,status",
            Some(&filters),
            None,
            None,
            0,
        )
        .await
        .ok_or_else(|| ApiErrorT::Internal("supabase unavailable".into()))?;
    Ok(json!({
        "league_id": league_id,
        "standings": standings_from_games(&rows),
    }))
}

/// League feed: recent trades and finalized results (filtered in SQL + parallelized).
pub async fn league_feed(state: &AppState, league_id: &str, limit: usize) -> Result<Value, ApiErrorT> {
    let sb = league::sb_required(state)?;
    let trade_filters = BTreeMap::from([
        ("league_id".to_string(), format!("eq.{league_id}")),
        ("status".to_string(), "neq.pending".to_string()),
    ]);
    let game_filters = BTreeMap::from([
        ("league_id".to_string(), format!("eq.{league_id}")),
        ("status".to_string(), "eq.final".to_string()),
    ]);
    let (trades_opt, games_opt) = tokio::join!(
        sb.read(
            "manager_trade_offers",
            "created_at,from_team,to_team,offered_player_ids,requested_player_ids,status,responded_at",
            Some(&trade_filters),
            None,
            Some("created_at.desc"),
            limit,
        ),
        sb.read(
            "manager_games",
            "date,home_abbrev,away_abbrev,home_score,away_score,status,nhl_game_id",
            Some(&game_filters),
            None,
            Some("date.desc,nhl_game_id.desc"),
            limit,
        ),
    );
    let trades = trades_opt.unwrap_or_default();
    let games = games_opt.unwrap_or_default();

    let mut feed: Vec<Value> = Vec::with_capacity(trades.len() + games.len());
    for t in &trades {
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
    let f_lid = league::eq_filter("league_id", league_id);
    let (league_res, teams_opt, pool) = tokio::join!(
        league_meta(sb, league_id),
        sb.read(
            "manager_league_teams",
            "team_abbrev",
            Some(&f_lid),
            None,
            None,
            0,
        ),
        crate::manager::draft::draft_pool(state),
    );
    let league_row = league_res?;
    if str_value(league_row.get("mode")) != crate::manager::MODE_TRUE_ROSTERS {
        return Err(ApiErrorT::BadRequest(json!({"error": "invalid_mode"})));
    }
    let teams = teams_opt.ok_or_else(|| ApiErrorT::Internal("supabase unavailable".into()))?;
    let team_set: std::collections::HashSet<String> = teams
        .iter()
        .map(|t| str_value(t.get("team_abbrev")))
        .filter(|s| !s.is_empty())
        .collect();

    let mut rows: Vec<Value> = Vec::new();
    for (pid, info) in &pool {
        let team = str_value(info.get("team"));
        if !team_set.contains(&team) {
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
    if !rows.is_empty() {
        if !crate::supabase::write::upsert_rows_minimal(
            sb,
            "manager_rosters",
            &rows,
            "league_id,team_abbrev,player_id",
        )
        .await
        {
            return Err(ApiErrorT::Internal("roster seed failed".into()));
        }
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
        assert_eq!(rows.len(), 3, "standings must not contain duplicate teams");
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
