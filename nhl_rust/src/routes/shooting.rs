//! M2b shooting / goaltending routes. Ports of `api_skaters_shooting` and
//! `api_goalies_goaltending` from `app/routes.py`. Supabase-first (`pbp` table);
//! the NHL-API team-season rebuild fallback is deferred to M3 (PBP routes).

use std::collections::{BTreeMap, HashMap};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::data::players as players_data;
use crate::state::AppState;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/skaters/shooting", get(api_skaters_shooting))
        .route("/api/goalies/goaltending", get(api_goalies_goaltending))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn q<'a>(params: &'a HashMap<String, String>, key: &str, default: &'a str) -> String {
    params
        .get(key)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| default.to_string())
}

fn i_of(v: &Value, key: &str) -> i64 {
    safe_int(v.get(key)).unwrap_or(0)
}

fn f_of(v: &Value, key: &str) -> f64 {
    parse_locale_float(v.get(key)).unwrap_or(0.0)
}

fn s_of(v: &Value, key: &str) -> String {
    str_value(v.get(key))
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn season_ids_from_str(s: &str) -> Vec<i64> {
    let mut out: Vec<i64> = s
        .split(',')
        .filter_map(|x| x.trim().parse::<i64>().ok())
        .filter(|x| *x > 0)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn xg_col_for(xg_model: &str) -> &'static str {
    match xg_model {
        "xG_S" => "xg_s",
        "xG_F2" => "xg_f2",
        _ => "xg_f",
    }
}

/// Applies the strength-state filter. `invert` flips the strength semantics
/// (used for goaltending where `strength_state` is from the shooter's side).
fn apply_strength(events: Vec<Value>, strength: &str, invert: bool) -> Vec<Value> {
    let strength = strength.trim();
    if strength.is_empty() || strength.eq_ignore_ascii_case("all") {
        return events;
    }
    match strength {
        "5v5" => events
            .into_iter()
            .filter(|e| s_of(e, "strength_state") == "5v5")
            .collect(),
        "PP" => {
            if invert {
                // Goalie's team on PP → shooter is shorthanded.
                events
                    .into_iter()
                    .filter(|e| matches!(s_of(e, "strength_state").as_str(), "4v5" | "3v5" | "3v4"))
                    .collect()
            } else {
                events
                    .into_iter()
                    .filter(|e| matches!(s_of(e, "strength_state").as_str(), "5v4" | "5v3" | "4v3"))
                    .collect()
            }
        }
        "SH" => {
            if invert {
                // Goalie's team on PK → shooter has the power play.
                events
                    .into_iter()
                    .filter(|e| matches!(s_of(e, "strength_state").as_str(), "5v4" | "5v3" | "4v3"))
                    .collect()
            } else {
                events
                    .into_iter()
                    .filter(|e| matches!(s_of(e, "strength_state").as_str(), "4v5" | "3v5" | "3v4"))
                    .collect()
            }
        }
        other => events
            .into_iter()
            .filter(|e| s_of(e, "strength_state") == other)
            .collect(),
    }
}

/// `GET /api/skaters/shooting` — port of `api_skaters_shooting`.
async fn api_skaters_shooting(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    if team.is_empty() || season.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "team and season required" })))
            .into_response();
    }
    let season_ids = season_ids_from_str(&season);

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();

    let player_id = q(&params, "player", "");

    let roster_ids_raw = q(&params, "roster", "");
    let roster_ids: Option<HashSetI64> = if !roster_ids_raw.is_empty() && player_id.is_empty() {
        let set: HashSetI64 = roster_ids_raw
            .split(',')
            .filter(|x| !x.trim().is_empty())
            .filter_map(|x| x.trim().parse::<i64>().ok())
            .collect();
        if set.is_empty() { None } else { Some(set) }
    } else {
        None
    };

    let cache_key = json!({
        "team": team,
        "seasons": season_ids,
        "ss": ss,
        "strength": strength,
        "xg_model": xg_model,
        "player": player_id,
        "roster": roster_ids.iter().flat_map(|s| s.iter().copied()).collect::<Vec<i64>>(),
    })
    .to_string();
    if let Some(v) = state.caches.skaters_shooting.get(&cache_key) {
        return json_no_store(v);
    }

    let Some(sb) = state.sb.as_ref() else {
        let payload = empty_shooting_payload();
        state.caches.skaters_shooting.insert(cache_key, payload.clone());
        return json_no_store(payload);
    };

    let mut all_events: Vec<Value> = Vec::new();
    let mut sb_failed = false;
    let cols = format!(
        "event_index,game_id,x,y,box_id,shot,goal,corsi,fenwick,{xg_col},goalie_id,shot_type,player1_id,position,shoots,event_team,opponent,score_state,shot_distance,strength_state,event,highlight_url,period"
    );
    for season_id in &season_ids {
        let mut db_filters = vec![
            ("event_team", format!("eq.{team}")),
            ("season", format!("eq.{season_id}")),
            ("corsi", "eq.1".to_string()),
        ];
        if !player_id.is_empty() {
            db_filters.push(("player1_id", format!("eq.{player_id}")));
        }
        if !ss.is_empty() && ss != "all" {
            db_filters.push(("season_state", format!("eq.{ss}")));
        }
        let fmap: BTreeMap<String, String> =
            db_filters.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        match sb
            .read("pbp", &cols, Some(&fmap), None, Some("event_index"), 0)
            .await
        {
            Some(rows) => {
                all_events.extend(rows.into_iter().filter(|e| i_of(e, "period") != 5));
            }
            None => {
                sb_failed = true;
                all_events.clear();
                break;
            }
        }
    }
    let _ = sb_failed; // fallback deferred to M3.

    if !player_id.is_empty() {
        if let Ok(pid) = player_id.parse::<i64>() {
            all_events.retain(|e| i_of(e, "player1_id") == pid);
        }
    }
    all_events.retain(|e| {
        s_of(e, "event_team").to_uppercase() == team
            && i_of(e, "corsi") == 1
            && i_of(e, "period") != 5
    });
    if let Some(roster) = &roster_ids {
        all_events.retain(|e| roster.contains(&i_of(e, "player1_id")));
    }
    all_events = apply_strength(all_events, &strength, false);
    all_events.sort_by(|a, b| {
        i_of(a, "game_id")
            .cmp(&i_of(b, "game_id"))
            .then_with(|| i_of(a, "event_index").cmp(&i_of(b, "event_index")))
    });

    let mut goalie_agg: BTreeMap<i64, GoalieAgg> = BTreeMap::new();
    let mut zone_agg: BTreeMap<String, ZoneAgg> = BTreeMap::new();
    let mut events_out: Vec<Value> = Vec::new();
    let mut total_shots: i64 = 0;
    let mut total_xg: f64 = 0.0;
    let mut total_goals: i64 = 0;

    for e in &all_events {
        let xg_val = f_of(e, &xg_col);
        let is_goal = i_of(e, "goal") == 1;
        let is_shot = i_of(e, "shot") == 1 || is_goal;
        let is_blocked = i_of(e, "corsi") == 1 && i_of(e, "fenwick") != 1 && !is_shot;
        let outcome = if is_goal {
            "Goal"
        } else if is_shot {
            "On Net"
        } else if is_blocked {
            "Block"
        } else {
            "Miss"
        };
        let gid = safe_int(e.get("goalie_id"));
        let box_id = s_of(e, "box_id");
        let score_state_out = safe_int(e.get("score_state")).filter(|v| v.is_negative() || *v >= 0);

        if !is_blocked {
            total_shots += 1;
            total_xg += xg_val;
            if is_goal {
                total_goals += 1;
            }
        }

        if let Some(gid) = gid {
            if !is_blocked {
                let g = goalie_agg.entry(gid).or_insert_with(|| GoalieAgg {
                    goalie_id: gid,
                    shots: 0,
                    xg: 0.0,
                    goals: 0,
                });
                g.shots += 1;
                g.xg += xg_val;
                if is_goal {
                    g.goals += 1;
                }
            }
        }

        if !box_id.is_empty() && !is_blocked {
            let z = zone_agg.entry(box_id.clone()).or_insert_with(|| ZoneAgg {
                shots: 0,
                xg: 0.0,
                goals: 0,
            });
            z.shots += 1;
            z.xg += xg_val;
            if is_goal {
                z.goals += 1;
            }
        }

        events_out.push(json!({
            "eventIndex": safe_int(e.get("event_index")),
            "gameId": safe_int(e.get("game_id")),
            "x": f_of(e, "x"),
            "y": f_of(e, "y"),
            "boxId": box_id,
            "goal": if is_goal { 1 } else { 0 },
            "shot": if is_shot { 1 } else { 0 },
            "fenwick": i_of(e, "fenwick"),
            "corsi": i_of(e, "corsi"),
            "isBlocked": if is_blocked { 1 } else { 0 },
            "outcome": outcome,
            "xG": r4(xg_val),
            "goalieId": gid,
            "goalieName": s_of(e, "goalie"),
            "shotType": s_of(e, "shot_type"),
            "playerId": safe_int(e.get("player1_id")),
            "playerName": s_of(e, "player1"),
            "position": s_of(e, "position").to_uppercase(),
            "shoots": s_of(e, "shoots").to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_default(),
            "eventTeam": s_of(e, "event_team").to_uppercase(),
            "opponent": s_of(e, "opponent").to_uppercase(),
            "scoreState": score_state_out,
            "shotDistance": if e.get("shot_distance").is_some() && !e.get("shot_distance").unwrap().is_null() { Some(f_of(e, "shot_distance")) } else { None },
            "highlightUrl": if is_goal { s_of(e, "highlight_url") } else { String::new() },
        }));
    }

    let sh_pct = if total_shots > 0 { total_goals as f64 / total_shots as f64 * 100.0 } else { 0.0 };
    let x_sh_pct = if total_shots > 0 { total_xg / total_shots as f64 * 100.0 } else { 0.0 };
    let d_sh_pct = sh_pct - x_sh_pct;
    let gax = total_goals as f64 - total_xg;

    let mut goalies_list: Vec<Value> = goalie_agg
        .values()
        .map(|g| {
            json!({
                "goalieId": g.goalie_id,
                "shots": g.shots,
                "xG": r2(g.xg),
                "goals": g.goals,
            })
        })
        .collect();
    goalies_list.sort_by(|a, b| {
        let a_net = i_of(a, "goals") as f64 - f_of(a, "xG");
        let b_net = i_of(b, "goals") as f64 - f_of(b, "xG");
        b_net.partial_cmp(&a_net).unwrap_or(std::cmp::Ordering::Equal)
    });

    let player_names = players_data::load_player_names_for_seasons(&state.caches, state.sb.as_ref(), &season_ids).await;
    for ev in &mut events_out {
        if let Some(pid) = safe_int(ev.get("playerId")) {
            let name = player_names
                .get(&pid)
                .cloned()
                .unwrap_or_else(|| {
                    let cur = str_value(ev.get("playerName"));
                    if cur.is_empty() { pid.to_string() } else { cur }
                });
            ev["playerName"] = json!(name);
        }
        if let Some(gid) = safe_int(ev.get("goalieId")) {
            let name = player_names
                .get(&gid)
                .cloned()
                .unwrap_or_else(|| {
                    let cur = str_value(ev.get("goalieName"));
                    if cur.is_empty() { gid.to_string() } else { cur }
                });
            ev["goalieName"] = json!(name);
        }
    }
    for g in &mut goalies_list {
        if let Some(gid) = safe_int(g.get("goalieId")) {
            let name = player_names
                .get(&gid)
                .cloned()
                .unwrap_or_else(|| gid.to_string());
            g["name"] = json!(name);
        }
        g["gax"] = json!(r2(i_of(g, "goals") as f64 - f_of(g, "xG")));
    }

    let mut zones = serde_json::Map::new();
    for (k, z) in &zone_agg {
        zones.insert(
            k.clone(),
            json!({"shots": z.shots, "xG": r2(z.xg), "goals": z.goals}),
        );
    }

    let payload = json!({
        "kpis": {
            "shots": total_shots,
            "xG": r2(total_xg),
            "goals": total_goals,
            "gax": r2(gax),
            "shPct": r1(sh_pct),
            "xShPct": r1(x_sh_pct),
            "dShPct": r1(d_sh_pct),
        },
        "goalies": goalies_list,
        "events": events_out,
        "zones": Value::Object(zones),
    });
    state.caches.skaters_shooting.insert(cache_key, payload.clone());
    json_no_store(payload)
}

/// `GET /api/goalies/goaltending` — port of `api_goalies_goaltending`.
async fn api_goalies_goaltending(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    if team.is_empty() || season.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({ "error": "team and season required" })))
            .into_response();
    }
    let season_ids = season_ids_from_str(&season);

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();

    let goalie_id = q(&params, "player", "");

    let roster_ids_raw = q(&params, "roster", "");
    let roster_ids: Option<HashSetI64> = if !roster_ids_raw.is_empty() && goalie_id.is_empty() {
        let set: HashSetI64 = roster_ids_raw
            .split(',')
            .filter(|x| !x.trim().is_empty())
            .filter_map(|x| x.trim().parse::<i64>().ok())
            .collect();
        if set.is_empty() { None } else { Some(set) }
    } else {
        None
    };

    let cache_key = json!({
        "team": team,
        "seasons": season_ids,
        "ss": ss,
        "strength": strength,
        "xg_model": xg_model,
        "player": goalie_id,
        "roster": roster_ids.iter().flat_map(|s| s.iter().copied()).collect::<Vec<i64>>(),
    })
    .to_string();
    if let Some(v) = state.caches.goalies_goaltending.get(&cache_key) {
        return json_no_store(v);
    }

    let Some(sb) = state.sb.as_ref() else {
        let payload = empty_goaltending_payload();
        state.caches.goalies_goaltending.insert(cache_key, payload.clone());
        return json_no_store(payload);
    };

    let mut all_events: Vec<Value> = Vec::new();
    let cols = format!(
        "event_index,game_id,x,y,box_id,shot,goal,corsi,fenwick,{xg_col},goalie_id,shot_type,player1_id,position,shoots,event_team,opponent,score_state,shot_distance,strength_state,highlight_url,period"
    );
    for season_id in &season_ids {
        let mut db_filters = vec![
            ("opponent", format!("eq.{team}")),
            ("season", format!("eq.{season_id}")),
            ("corsi", "eq.1".to_string()),
        ];
        if !goalie_id.is_empty() {
            db_filters.push(("goalie_id", format!("eq.{goalie_id}")));
        }
        if !ss.is_empty() && ss != "all" {
            db_filters.push(("season_state", format!("eq.{ss}")));
        }
        let fmap: BTreeMap<String, String> =
            db_filters.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        match sb
            .read("pbp", &cols, Some(&fmap), None, Some("event_index"), 0)
            .await
        {
            Some(rows) => {
                all_events.extend(rows.into_iter().filter(|e| i_of(e, "period") != 5));
            }
            None => {
                all_events.clear();
                break;
            }
        }
    }

    if !goalie_id.is_empty() {
        if let Ok(pid) = goalie_id.parse::<i64>() {
            all_events.retain(|e| i_of(e, "goalie_id") == pid);
        }
    }
    all_events.retain(|e| {
        s_of(e, "opponent").to_uppercase() == team
            && i_of(e, "corsi") == 1
            && i_of(e, "period") != 5
    });
    if let Some(roster) = &roster_ids {
        all_events.retain(|e| roster.contains(&i_of(e, "goalie_id")));
    }
    // NOTE: strength_state is from the SHOOTER's perspective — invert for goalies.
    all_events = apply_strength(all_events, &strength, true);
    all_events.sort_by(|a, b| {
        i_of(a, "game_id")
            .cmp(&i_of(b, "game_id"))
            .then_with(|| i_of(a, "event_index").cmp(&i_of(b, "event_index")))
    });

    let mut shooter_agg: BTreeMap<i64, ShooterAgg> = BTreeMap::new();
    let mut zone_agg: BTreeMap<String, ZoneAgg> = BTreeMap::new();
    let mut events_out: Vec<Value> = Vec::new();
    let mut total_sa: i64 = 0;
    let mut total_xga: f64 = 0.0;
    let mut total_ga: i64 = 0;

    for e in &all_events {
        let xg_val = f_of(e, &xg_col);
        let is_goal = i_of(e, "goal") == 1;
        let is_shot = i_of(e, "shot") == 1 || is_goal;
        let is_blocked = i_of(e, "corsi") == 1 && i_of(e, "fenwick") != 1 && !is_shot;
        let outcome = if is_goal {
            "Goal"
        } else if is_shot {
            "On Net"
        } else if is_blocked {
            "Block"
        } else {
            "Miss"
        };
        let pid = safe_int(e.get("player1_id"));
        let box_id = s_of(e, "box_id");
        let score_state_out = safe_int(e.get("score_state")).filter(|v| v.is_negative() || *v >= 0);

        if !is_blocked {
            total_sa += 1;
            total_xga += xg_val;
            if is_goal {
                total_ga += 1;
            }
        }

        if let Some(pid) = pid {
            if !is_blocked {
                let s = shooter_agg.entry(pid).or_insert_with(|| ShooterAgg {
                    player_id: pid,
                    sa: 0,
                    xga: 0.0,
                    ga: 0,
                });
                s.sa += 1;
                s.xga += xg_val;
                if is_goal {
                    s.ga += 1;
                }
            }
        }

        if !box_id.is_empty() && !is_blocked {
            let z = zone_agg.entry(box_id.clone()).or_insert_with(|| ZoneAgg {
                shots: 0,
                xg: 0.0,
                goals: 0,
            });
            z.shots += 1;
            z.xg += xg_val;
            if is_goal {
                z.goals += 1;
            }
        }

        events_out.push(json!({
            "eventIndex": safe_int(e.get("event_index")),
            "gameId": safe_int(e.get("game_id")),
            "x": f_of(e, "x"),
            "y": f_of(e, "y"),
            "boxId": box_id,
            "goal": if is_goal { 1 } else { 0 },
            "shot": if is_shot { 1 } else { 0 },
            "fenwick": i_of(e, "fenwick"),
            "corsi": i_of(e, "corsi"),
            "isBlocked": if is_blocked { 1 } else { 0 },
            "outcome": outcome,
            "xG": r4(xg_val),
            "playerId": pid,
            "playerName": s_of(e, "player1"),
            "goalieId": safe_int(e.get("goalie_id")),
            "goalieName": s_of(e, "goalie"),
            "shotType": s_of(e, "shot_type"),
            "position": s_of(e, "position").to_uppercase(),
            "shoots": s_of(e, "shoots").to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_default(),
            "eventTeam": s_of(e, "event_team").to_uppercase(),
            "opponent": s_of(e, "opponent").to_uppercase(),
            "scoreState": score_state_out,
            "shotDistance": if e.get("shot_distance").is_some() && !e.get("shot_distance").unwrap().is_null() { Some(f_of(e, "shot_distance")) } else { None },
            "highlightUrl": if is_goal { s_of(e, "highlight_url") } else { String::new() },
        }));
    }

    let sv_pct = if total_sa > 0 { (total_sa - total_ga) as f64 / total_sa as f64 * 100.0 } else { 0.0 };
    let x_sv_pct = if total_sa > 0 { (total_sa as f64 - total_xga) / total_sa as f64 * 100.0 } else { 0.0 };
    let d_sv_pct = sv_pct - x_sv_pct;
    let gsax = total_xga - total_ga as f64;

    let mut shooters_list: Vec<Value> = shooter_agg
        .values()
        .map(|s| {
            json!({
                "playerId": s.player_id,
                "sa": s.sa,
                "xGA": r2(s.xga),
                "ga": s.ga,
            })
        })
        .collect();
    shooters_list.sort_by(|a, b| {
        let a_net = f_of(a, "xGA") - i_of(a, "ga") as f64;
        let b_net = f_of(b, "xGA") - i_of(b, "ga") as f64;
        b_net.partial_cmp(&a_net).unwrap_or(std::cmp::Ordering::Equal)
    });

    let player_names = players_data::load_player_names_for_seasons(&state.caches, state.sb.as_ref(), &season_ids).await;
    for ev in &mut events_out {
        if let Some(pid) = safe_int(ev.get("playerId")) {
            let name = player_names
                .get(&pid)
                .cloned()
                .unwrap_or_else(|| {
                    let cur = str_value(ev.get("playerName"));
                    if cur.is_empty() { pid.to_string() } else { cur }
                });
            ev["playerName"] = json!(name);
        }
        if let Some(gid) = safe_int(ev.get("goalieId")) {
            let name = player_names
                .get(&gid)
                .cloned()
                .unwrap_or_else(|| {
                    let cur = str_value(ev.get("goalieName"));
                    if cur.is_empty() { gid.to_string() } else { cur }
                });
            ev["goalieName"] = json!(name);
        }
    }
    for s in &mut shooters_list {
        if let Some(pid) = safe_int(s.get("playerId")) {
            let name = player_names
                .get(&pid)
                .cloned()
                .unwrap_or_else(|| pid.to_string());
            s["name"] = json!(name);
        }
        s["gsax"] = json!(r2(f_of(s, "xGA") - i_of(s, "ga") as f64));
    }

    let mut zones = serde_json::Map::new();
    for (k, z) in &zone_agg {
        zones.insert(
            k.clone(),
            json!({"sa": z.shots, "xGA": r2(z.xg), "ga": z.goals}),
        );
    }

    let payload = json!({
        "kpis": {
            "sa": total_sa,
            "xGA": r2(total_xga),
            "ga": total_ga,
            "gsax": r2(gsax),
            "svPct": r1(sv_pct),
            "xSvPct": r1(x_sv_pct),
            "dSvPct": r1(d_sv_pct),
        },
        "shooters": shooters_list,
        "events": events_out,
        "zones": Value::Object(zones),
    });
    state.caches.goalies_goaltending.insert(cache_key, payload.clone());
    json_no_store(payload)
}

type HashSetI64 = std::collections::HashSet<i64>;

struct GoalieAgg {
    goalie_id: i64,
    shots: i64,
    xg: f64,
    goals: i64,
}

struct ShooterAgg {
    player_id: i64,
    sa: i64,
    xga: f64,
    ga: i64,
}

struct ZoneAgg {
    shots: i64,
    xg: f64,
    goals: i64,
}

fn empty_shooting_payload() -> Value {
    json!({
        "kpis": {"shots": 0, "xG": 0, "goals": 0, "gax": 0, "shPct": 0, "xShPct": 0, "dShPct": 0},
        "goalies": [],
        "events": [],
        "zones": {},
    })
}

fn empty_goaltending_payload() -> Value {
    json!({
        "kpis": {"sa": 0, "xGA": 0, "ga": 0, "gsax": 0, "svPct": 0, "xSvPct": 0, "dSvPct": 0},
        "shooters": [],
        "events": [],
        "zones": {},
    })
}
