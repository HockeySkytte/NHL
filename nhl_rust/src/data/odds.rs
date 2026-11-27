//! Odds snapshots — port of `_load_odds_snapshot_rows` + `_build_odds_snapshot_payloads`
//! (plus the `_snapshot_*` field helpers). Supabase `odds_snapshots` is read-only here.

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Map, Value};

use crate::nhl::client::{get_json, API_WEB};
use crate::state::Caches;
use crate::supabase::read::SbClient;
use crate::util::parse::{ci_get, parse_locale_float, safe_int, str_value};

/// `_load_odds_snapshot_rows(game_id=...)` with cache.
pub async fn load_snapshot_rows(
    caches: &Caches,
    sb: Option<&SbClient>,
    game_id: i64,
) -> Vec<Value> {
    let cache_key = format!("game:{game_id}");
    if let Some(v) = caches.odds_snapshot_rows.get(&cache_key) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }
    let out = if let Some(sb) = sb {
        let mut rows = sb
            .read(
                "odds_snapshots",
                "*",
                Some(&crate::supabase::read::filters(&[("game_id", &format!("eq.{game_id}"))])),
                None,
                None,
                0,
            )
            .await;
        if rows.is_none() {
            rows = sb
                .read(
                    "odds_snapshots",
                    "*",
                    Some(&crate::supabase::read::filters(&[("gameid", &format!("eq.{game_id}"))])),
                    None,
                    None,
                    0,
                )
                .await;
        }
        rows.unwrap_or_default()
    } else {
        Vec::new()
    };
    caches
        .odds_snapshot_rows
        .insert(cache_key, Value::Array(out.clone()));
    out
}

fn snapshot_pick(row: &Map<String, Value>, keys: &[&str]) -> Option<Value> {
    for key in keys {
        if let Some(v) = ci_get(row, key) {
            if !v.is_null() && str_value(Some(v)) != "" {
                return Some(v.clone());
            }
        }
    }
    None
}

fn snapshot_percent(v: Option<Value>) -> Option<f64> {
    let val = parse_locale_float(v.as_ref())?;
    let pct = if (0.0..=1.0).contains(&val) { val * 100.0 } else { val };
    Some(pct)
}

fn snapshot_timestamp(row: &Map<String, Value>) -> String {
    str_value(
        snapshot_pick(
            row,
            &["timestamp", "timestamp_utc", "fetched_at_utc", "snapshot_at", "created_at", "updated_at", "TimestampUTC"],
        )
        .as_ref(),
    )
}

fn snapshot_team(row: &Map<String, Value>, side: Option<&str>) -> String {
    let raw = match side {
        Some("away") => snapshot_pick(row, &["away_team", "away", "away_abbrev", "awayTeam", "Away"]),
        Some("home") => snapshot_pick(row, &["home_team", "home", "home_abbrev", "homeTeam", "Home"]),
        _ => snapshot_pick(row, &["team", "team_abbrev", "teamAbbrev", "abbrev", "Team"]),
    };
    str_value(raw.as_ref()).to_uppercase()
}

fn snapshot_side_ml(row: &Map<String, Value>, side: Option<&str>) -> Option<f64> {
    let raw = match side {
        Some("away") => snapshot_pick(row, &["odds_away", "away_odds", "away_ml", "ml_away", "oddsAway", "OddsAway", "awayPrice"]),
        Some("home") => snapshot_pick(row, &["odds_home", "home_odds", "home_ml", "ml_home", "oddsHome", "OddsHome", "homePrice"]),
        _ => snapshot_pick(row, &["money_line_2_way", "ml", "odds", "american_odds", "price", "value"]),
    };
    parse_locale_float(raw.as_ref())
}

fn snapshot_side_pct(row: &Map<String, Value>, side: &str, kind: &str) -> Option<f64> {
    let raw = if kind == "win" {
        let away = side == "away";
        snapshot_pick(
            row,
            &[
                &format!("{side}_win_pct"),
                &format!("win_{side}_pct"),
                &format!("{side}WinPct"),
                &format!("{side}_win_prob"),
                &format!("win_{side}_prob"),
                if away { "winAwayPct" } else { "winHomePct" },
                if away { "WinAway" } else { "WinHome" },
            ],
        )
    } else {
        let away = side == "away";
        snapshot_pick(
            row,
            &[
                &format!("{side}_bet_pct"),
                &format!("bet_{side}_pct"),
                &format!("{side}BetPct"),
                if away { "betAwayPct" } else { "betHomePct" },
                if away { "BetAway" } else { "BetHome" },
            ],
        )
    };
    snapshot_percent(raw)
}

/// `_build_odds_snapshot_payloads(rows, away, home)`.
pub fn build_payloads(
    rows: &[Value],
    away_abbrev: &str,
    home_abbrev: &str,
) -> (Value, BTreeMap<String, Vec<Value>>) {
    let mut groups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    for (idx, row) in rows.iter().enumerate() {
        let obj = row.as_object().cloned().unwrap_or_default();
        let ts = snapshot_timestamp(&obj);
        let ts = if ts.is_empty() {
            format!("row-{idx:08}")
        } else {
            ts
        };
        groups.entry(ts).or_default().push(row.clone());
    }

    let mut latest_payload = Value::Null;
    let mut points_by_team: BTreeMap<String, Vec<Value>> = BTreeMap::new();

    for (ts, group) in &groups {
        let mut by_team: Map<String, Value> = Map::new();
        let mut payload = json!({
            "timestamp": ts,
            "awayTeam": away_abbrev.to_string(),
            "homeTeam": home_abbrev.to_string(),
            "oddsAway": null,
            "oddsHome": null,
            "winAwayPct": null,
            "winHomePct": null,
            "betAwayPct": null,
            "betHomePct": null,
        });
        let payload_obj = payload.as_object_mut().expect("payload object");

        for row in group {
            let Some(obj) = row.as_object() else { continue };
            let away_team = snapshot_team(obj, Some("away"));
            let home_team = snapshot_team(obj, Some("home"));
            if !away_team.is_empty() {
                payload_obj.insert("awayTeam".into(), Value::String(away_team.clone()));
            }
            if !home_team.is_empty() {
                payload_obj.insert("homeTeam".into(), Value::String(home_team.clone()));
            }
            if let Some(v) = snapshot_side_ml(obj, Some("away")) {
                payload_obj.insert("oddsAway".into(), json!(v));
            }
            if let Some(v) = snapshot_side_ml(obj, Some("home")) {
                payload_obj.insert("oddsHome".into(), json!(v));
            }
            if let Some(v) = snapshot_side_pct(obj, "away", "win") {
                payload_obj.insert("winAwayPct".into(), json!(v));
            }
            if let Some(v) = snapshot_side_pct(obj, "home", "win") {
                payload_obj.insert("winHomePct".into(), json!(v));
            }
            if let Some(v) = snapshot_side_pct(obj, "away", "bet") {
                payload_obj.insert("betAwayPct".into(), json!(v));
            }
            if let Some(v) = snapshot_side_pct(obj, "home", "bet") {
                payload_obj.insert("betHomePct".into(), json!(v));
            }

            let team_row = snapshot_team(obj, None);
            if !team_row.is_empty() {
                let entry = by_team
                    .entry(team_row.clone())
                    .or_insert_with(|| json!({"ml": null, "winPct": null, "betPct": null}));
                let e = entry.as_object_mut().expect("team rec object");
                if let Some(v) = snapshot_side_ml(obj, None) {
                    e.insert("ml".into(), json!(v));
                }
                if let Some(v) = snapshot_percent(snapshot_pick(obj, &["win_pct", "win_prop", "win_probability", "implied_win_pct"])) {
                    e.insert("winPct".into(), json!(v));
                }
                if let Some(v) = snapshot_percent(snapshot_pick(obj, &["bet_pct", "bet_prop", "kelly_pct"])) {
                    e.insert("betPct".into(), json!(v));
                }
            }
        }

        // Uppercase payload teams, falling back to the boxscore abbrevs.
        let away_team = str_value(payload_obj.get("awayTeam")).to_uppercase();
        let home_team = str_value(payload_obj.get("homeTeam")).to_uppercase();
        payload_obj.insert("awayTeam".into(), Value::String(away_team.clone()));
        payload_obj.insert("homeTeam".into(), Value::String(home_team.clone()));

        if payload_obj.get("oddsAway").and_then(Value::as_f64).is_none() && !away_team.is_empty() {
            if let Some(rec) = by_team.get(&away_team) {
                if let Some(ml) = rec.get("ml").and_then(Value::as_f64) {
                    payload_obj.insert("oddsAway".into(), json!(ml));
                }
                if payload_obj.get("winAwayPct").and_then(Value::as_f64).is_none() {
                    if let Some(v) = rec.get("winPct").and_then(Value::as_f64) {
                        payload_obj.insert("winAwayPct".into(), json!(v));
                    }
                }
                if payload_obj.get("betAwayPct").and_then(Value::as_f64).is_none() {
                    if let Some(v) = rec.get("betPct").and_then(Value::as_f64) {
                        payload_obj.insert("betAwayPct".into(), json!(v));
                    }
                }
            }
        }
        if payload_obj.get("oddsHome").and_then(Value::as_f64).is_none() && !home_team.is_empty() {
            if let Some(rec) = by_team.get(&home_team) {
                if let Some(ml) = rec.get("ml").and_then(Value::as_f64) {
                    payload_obj.insert("oddsHome".into(), json!(ml));
                }
                if payload_obj.get("winHomePct").and_then(Value::as_f64).is_none() {
                    if let Some(v) = rec.get("winPct").and_then(Value::as_f64) {
                        payload_obj.insert("winHomePct".into(), json!(v));
                    }
                }
                if payload_obj.get("betHomePct").and_then(Value::as_f64).is_none() {
                    if let Some(v) = rec.get("betPct").and_then(Value::as_f64) {
                        payload_obj.insert("betHomePct".into(), json!(v));
                    }
                }
            }
        }

        let mut added: std::collections::HashSet<String> = Default::default();
        if !away_team.is_empty() {
            if let Some(ml) = payload_obj.get("oddsAway").and_then(Value::as_f64) {
                let win_prop = parse_locale_float(payload_obj.get("winAwayPct"));
                points_by_team.entry(away_team.clone()).or_default().push(json!({
                    "t": ts,
                    "ml": ml,
                    "winProp": win_prop.map(|v| v / 100.0),
                }));
                added.insert(away_team.clone());
            }
        }
        if !home_team.is_empty() {
            if let Some(ml) = payload_obj.get("oddsHome").and_then(Value::as_f64) {
                let win_prop = parse_locale_float(payload_obj.get("winHomePct"));
                points_by_team.entry(home_team.clone()).or_default().push(json!({
                    "t": ts,
                    "ml": ml,
                    "winProp": win_prop.map(|v| v / 100.0),
                }));
                added.insert(home_team.clone());
            }
        }
        for (team_key, rec) in &by_team {
            if added.contains(team_key) {
                continue;
            }
            let Some(ml) = rec.get("ml").and_then(Value::as_f64) else { continue };
            let win_pct = parse_locale_float(rec.get("winPct"));
            points_by_team.entry(team_key.clone()).or_default().push(json!({
                "t": ts,
                "ml": ml,
                "winProp": win_pct.map(|v| v / 100.0),
            }));
        }

        payload_obj.insert("_by_team".into(), Value::Object(by_team));
        latest_payload = payload;
    }

    (latest_payload, points_by_team)
}

/// Fetches boxscore abbrevs, consulting the box cache first (port of
/// `_load_game_team_abbrevs`).
pub async fn game_team_abbrevs(
    caches: &Caches,
    http: &reqwest::Client,
    game_id: i64,
) -> (String, String) {
    let data = match caches.box_cache.get(&game_id) {
        Some(v) => v,
        None => {
            let url = format!("{}/v1/gamecenter/{game_id}/boxscore", crate::nhl::client::API_WEB);
            match crate::nhl::client::get_json(http, &url, 20).await {
                Ok(v) => {
                    caches.box_cache.insert(game_id, v.clone());
                    v
                }
                Err(_) => Value::Null,
            }
        }
    };
    let away = data
        .get("awayTeam")
        .and_then(|t| t.get("abbrev").or_else(|| t.get("abbreviation")))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_uppercase();
    let home = data
        .get("homeTeam")
        .and_then(|t| t.get("abbrev").or_else(|| t.get("abbreviation")))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_uppercase();
    (away, home)
}

/// `_safe_int` wrapper for snapshot game ids.
fn snapshot_game_id(row: &Map<String, Value>) -> Option<i64> {
    safe_int(
        snapshot_pick(row, &["game_id", "gameId", "GameID", "gameid"]).as_ref(),
    )
}

/// Port of `_load_latest_odds_snapshot_map(game_team_map)`: fetches all
/// `odds_snapshots` rows for the requested games in a single batched query
/// (`game_id=in.(...)`) and returns the latest structured payload per game id.
pub async fn load_latest_snapshot_map(
    sb: Option<&SbClient>,
    game_team_map: &HashMap<i64, (String, String)>,
) -> HashMap<i64, Value> {
    let mut out = HashMap::new();
    let Some(sb) = sb else { return out };
    if game_team_map.is_empty() {
        return out;
    }
    let mut ids: Vec<i64> = game_team_map.keys().copied().filter(|&id| id > 0).collect();
    ids.sort_unstable();
    ids.dedup();
    if ids.is_empty() {
        return out;
    }
    let id_list = ids
        .iter()
        .map(|id| id.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let in_expr = format!("in.({id_list})");
    let rows = match sb
        .read(
            "odds_snapshots",
            "*",
            Some(&crate::supabase::read::filters(&[("game_id", &in_expr)])),
            None,
            None,
            0,
        )
        .await
    {
        Some(r) => r,
        None => sb
            .read(
                "odds_snapshots",
                "*",
                Some(&crate::supabase::read::filters(&[("gameid", &in_expr)])),
                None,
                None,
                0,
            )
            .await
            .unwrap_or_default(),
    };
    let mut by_game: HashMap<i64, Vec<Value>> = HashMap::new();
    for row in rows {
        if let Some(gid) = row.as_object().and_then(snapshot_game_id) {
            by_game.entry(gid).or_default().push(row);
        }
    }
    for (gid, (away_abbrev, home_abbrev)) in game_team_map {
        let Some(game_rows) = by_game.get(gid) else { continue };
        let (latest, _) = build_payloads(game_rows, away_abbrev, home_abbrev);
        if latest.is_object() {
            out.insert(*gid, latest);
        }
    }
    out
}

/// Fetches live bookmaker moneyline odds mapped by NHL `game_id` → `{"away", "home"}`.
///
/// Queries `/v1/score/{date_str}` (which carries DraftKings `providerId: 9` and
/// FanDuel `providerId: 7` American moneylines all day, including before noon ET
/// and for tomorrow's games) and `/v1/partner-game/US/now` concurrently, letting
/// partner-game `MONEY_LINE_2_WAY` overlay score odds when present for a game.
pub async fn fetch_partner_odds_map(
    http: &reqwest::Client,
    date_str: &str,
) -> HashMap<i64, Value> {
    let score_url = format!("{API_WEB}/v1/score/{date_str}");
    let partner_url = format!("{API_WEB}/v1/partner-game/US/now");
    let (score_res, partner_res) = tokio::join!(
        get_json(http, &score_url, 15),
        get_json(http, &partner_url, 15),
    );
    let mut out = score_res
        .ok()
        .map(|js| parse_score_odds_json(&js))
        .unwrap_or_default();
    if let Ok(js) = partner_res {
        for (gid, val) in parse_partner_odds_json(&js) {
            out.insert(gid, val);
        }
    }
    out
}

/// Preferred US American-odds provider IDs in `/v1/score/{date}`:
/// 9 = DraftKings (US), 7 = FanDuel (CA, American format).
const SCORE_AMERICAN_PROVIDERS: [i64; 2] = [9, 7];

fn extract_score_team_ml(team_obj: Option<&Value>) -> Option<Value> {
    let list = team_obj?.get("odds")?.as_array()?;
    for want_provider in SCORE_AMERICAN_PROVIDERS {
        for entry in list {
            if safe_int(entry.get("providerId")) == Some(want_provider) {
                if let Some(val) = entry.get("value") {
                    let s = str_value(Some(val));
                    if !s.is_empty() {
                        return Some(val.clone());
                    }
                }
            }
        }
    }
    None
}

/// Extracts `{ game_id -> {"away": <ml>, "home": <ml>} }` from `/v1/score/{date}`.
pub fn parse_score_odds_json(js: &Value) -> HashMap<i64, Value> {
    let mut out = HashMap::new();
    let Some(games) = js.get("games").and_then(Value::as_array) else {
        return out;
    };
    for g in games {
        let Some(gid) = safe_int(g.get("id").or_else(|| g.get("gameId"))) else {
            continue;
        };
        let away = extract_score_team_ml(g.get("awayTeam"));
        let home = extract_score_team_ml(g.get("homeTeam"));
        if away.is_some() || home.is_some() {
            out.insert(
                gid,
                json!({
                    "away": away.unwrap_or(Value::Null),
                    "home": home.unwrap_or(Value::Null),
                }),
            );
        }
    }
    out
}

fn extract_partner_team_ml(team_obj: Option<&Value>) -> Option<Value> {
    let list = team_obj?.get("odds")?.as_array()?;
    for want_desc in ["MONEY_LINE_2_WAY", "MONEY_LINE_2_WAY_TNB"] {
        for entry in list {
            let desc = str_value(entry.get("description")).to_uppercase();
            if desc == want_desc {
                if let Some(val) = entry.get("value") {
                    let s = str_value(Some(val));
                    if !s.is_empty() {
                        return Some(val.clone());
                    }
                }
            }
        }
    }
    None
}

/// Extracts `{ game_id -> {"away": <ml>, "home": <ml>} }` from `/v1/partner-game/US/*`.
pub fn parse_partner_odds_json(js: &Value) -> HashMap<i64, Value> {
    let mut out = HashMap::new();
    let Some(games) = js.get("games").and_then(Value::as_array) else {
        return out;
    };
    for g in games {
        let Some(gid) = safe_int(
            g.get("gameId")
                .or_else(|| g.get("id"))
                .or_else(|| g.get("eventId")),
        ) else {
            continue;
        };
        let away = extract_partner_team_ml(g.get("awayTeam"));
        let home = extract_partner_team_ml(g.get("homeTeam"));
        if away.is_some() || home.is_some() {
            out.insert(
                gid,
                json!({
                    "away": away.unwrap_or(Value::Null),
                    "home": home.unwrap_or(Value::Null),
                }),
            );
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_score_odds_prefers_draftkings_over_fanduel_and_ignores_decimal_providers() {
        let payload = json!({
            "games": [
                {
                    "id": 2026020377,
                    "awayTeam": {
                        "abbrev": "TOR",
                        "odds": [
                            {"providerId": 3, "value": "2.30"},
                            {"providerId": 7, "value": "+126"},
                            {"providerId": 9, "value": "+130"}
                        ]
                    },
                    "homeTeam": {
                        "abbrev": "BOS",
                        "odds": [
                            {"providerId": 3, "value": "1.64"},
                            {"providerId": 7, "value": "-152"},
                            {"providerId": 9, "value": "-155"}
                        ]
                    }
                },
                {
                    "id": 2026020378,
                    "awayTeam": {
                        "abbrev": "MTL",
                        "odds": [
                            {"providerId": 3, "value": "1.95"},
                            {"providerId": 7, "value": "-105"}
                        ]
                    },
                    "homeTeam": {
                        "abbrev": "OTT",
                        "odds": [
                            {"providerId": 3, "value": "1.87"},
                            {"providerId": 7, "value": "-115"}
                        ]
                    }
                },
                {
                    "id": 2026020379,
                    "awayTeam": { "abbrev": "DET", "odds": [{"providerId": 3, "value": "2.10"}] },
                    "homeTeam": { "abbrev": "BUF", "odds": [{"providerId": 3, "value": "1.75"}] }
                }
            ]
        });
        let map = parse_score_odds_json(&payload);
        assert_eq!(map.get(&2026020377), Some(&json!({"away": "+130", "home": "-155"})));
        assert_eq!(map.get(&2026020378), Some(&json!({"away": "-105", "home": "-115"})));
        assert!(!map.contains_key(&2026020379));
    }

    #[test]
    fn parse_partner_odds_extracts_money_line_2_way() {
        let payload = json!({
            "games": [
                {
                    "gameId": 2026020377,
                    "awayTeam": {
                        "odds": [
                            {"description": "PUCK_LINE", "value": "+1.5"},
                            {"description": "MONEY_LINE_2_WAY", "value": "+135"}
                        ]
                    },
                    "homeTeam": {
                        "odds": [
                            {"description": "MONEY_LINE_2_WAY_TNB", "value": "-160"}
                        ]
                    }
                }
            ]
        });
        let map = parse_partner_odds_json(&payload);
        assert_eq!(map.get(&2026020377), Some(&json!({"away": "+135", "home": "-160"})));
    }
}
