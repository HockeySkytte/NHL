//! M4 projections + GM Mode routes — ports of the projections family from
//! `app/routes.py` (V2 player projections, current projections, team season
//! points, custom lineups, games, and full-season simulation).

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::extract::{Path, Query, State};
use axum::http::header;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::data::lineups;
use crate::data::projections as p;
use crate::data::rapm;
use crate::data::rosters;
use crate::nhl::client::{get_json, API_WEB};
use crate::state::AppState;
use crate::util::dates::current_season_id;
use crate::util::mt19937::Mt19937;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/player-projections/v2", get(api_player_projections_v2))
        .route("/api/player-current-projections", get(api_player_current_projections))
        .route("/api/player-current-projections/public", get(api_player_current_projections_public))
        .route("/api/skaters/current-projections", get(api_skaters_current_projections))
        .route("/api/teams/current-projections", get(api_teams_current_projections))
        .route("/api/skaters/player-projection-trend/{player_id}", get(api_skaters_player_projection_trend))
        .route("/api/projections/team-season-points", get(api_projections_team_season_points))
        .route("/api/projections/team-season-points-custom", post(api_projections_team_season_points_custom))
        .route("/api/projections/all-teams-custom", post(api_projections_all_teams_custom))
        .route("/api/projections/custom-lineups-cache", get(api_projections_custom_lineups_cache_get).post(api_projections_custom_lineups_cache_post))
        .route("/api/projections/games", get(api_projections_games))
        .route("/api/projections/simulate-season", post(api_projections_simulate_season))
        .route("/api/projections/simulate-season-batch", post(api_projections_simulate_season_batch))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn json_err(status: StatusCode, v: Value) -> Response {
    (status, Json(v)).into_response()
}

fn q<'a>(params: &'a HashMap<String, String>, key: &str, default: &'a str) -> String {
    params
        .get(key)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| default.to_string())
}

fn season_param(params: &HashMap<String, String>, key: &str, default: i64) -> i64 {
    q(params, key, "")
        .parse::<i64>()
        .ok()
        .filter(|s| *s > 0)
        .unwrap_or(default)
}

/// `_projection_cache_actor_key()` — anonymous cookie key until M5 auth.
/// Returns (actor_key, Option<set_cookie>). A fresh cookie is generated when absent.
fn actor_key(cookies: Option<&str>) -> (String, Option<String>) {
    if let Some(c) = cookies {
        for pair in c.split(';') {
            let mut it = pair.trim().splitn(2, '=');
            if let (Some(k), Some(v)) = (it.next(), it.next()) {
                if k.trim() == "gm_projection_cache_key" && !v.trim().is_empty() {
                    return (format!("anon:{}", v.trim()), None);
                }
            }
        }
    }
    let mut rng = Mt19937::new(std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_nanos() & 0xffffffff) as u32)
        .unwrap_or(1));
    let hex: String = (0..16).map(|_| format!("{:02x}", rng.randrange(256))).collect();
    let set_cookie = format!("gm_projection_cache_key={hex}; Path=/; HttpOnly; SameSite=Lax");
    (format!("anon:{hex}"), Some(set_cookie))
}

fn attach_cookie(mut resp: Response, set_cookie: Option<String>) -> Response {
    if let Some(c) = set_cookie {
        let h = axum::http::HeaderValue::from_str(&c)
            .unwrap_or_else(|_| axum::http::HeaderValue::from_static(""));
        resp.headers_mut().insert(header::SET_COOKIE, h);
    }
    resp
}

fn get_cookie_header(headers: &axum::http::HeaderMap) -> Option<&str> {
    headers.get(header::COOKIE).and_then(|v| v.to_str().ok())
}

// ── /api/player-projections/v2 ──
async fn api_player_projections_v2(State(state): State<AppState>) -> Response {
    let season = current_season_id(None);
    let result = p::build_v2_player_projections(&state, Some(season)).await;
    json_no_store(json!({
        "players": result,
        "meta": {
            "coefficients": p::EVPP_COEF.iter().map(|(k, v)| (k.to_string(), json!(v))).collect::<serde_json::Map<_, _>>(),
            "count": result.len(),
        },
    }))
}

// ── /api/player-current-projections ──
async fn api_player_current_projections(State(state): State<AppState>) -> Response {
    let data = p::load_gm_mode_projections_cached(&state).await;
    let roster_map = rosters::all_rosters(&state.caches, &state.http).await;
    let mut out: serde_json::Map<String, Value> = serde_json::Map::new();
    for (k, mut row) in data {
        let pid = safe_int(row.get("player_id")).or_else(|| safe_int(row.get("playerId"))).unwrap_or(k);
        let info = roster_map.get(&pid.to_string()).cloned().unwrap_or(Value::Null);
        if info.is_object() {
            if str_value(row.get("name").or_else(|| row.get("player"))).is_empty() {
                row["name"] = json!(str_value(info.get("name")));
            }
            if str_value(row.get("team").or_else(|| row.get("Team"))).is_empty() {
                row["team"] = json!(str_value(info.get("team")).to_uppercase());
            }
            if str_value(row.get("position").or_else(|| row.get("Position"))).is_empty() {
                row["position"] = json!(str_value(info.get("position")).to_uppercase());
            }
        }
        out.insert(k.to_string(), row);
    }
    json_no_store(Value::Object(out))
}

// ── /api/player-current-projections/public ──
async fn api_player_current_projections_public(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let season = season_param(&params, "season", current_season_id(None));
    let model_key = q(&params, "model_key", "preseason_updating");
    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(json!({"rows": []}));
    };
    let mut db_filters = vec![("model_key", format!("eq.{model_key}"))];
    if season > 0 {
        db_filters.push(("season", format!("eq.{season}")));
    }
    let fmap: BTreeMap<String, String> = db_filters.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let rows = sb
        .read(
            "player_current_projections",
            "player_id,player,raw_projected_value,projected_value,generated_at",
            Some(&fmap),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    let mut out: Vec<Value> = Vec::new();
    for r in &rows {
        let pid = safe_int(r.get("player_id")).unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        out.push(json!({
            "player_id": pid,
            "player": str_value(r.get("player")),
            "raw_projected_value": parse_locale_float(r.get("raw_projected_value")),
            "projected_value": parse_locale_float(r.get("projected_value")),
            "generated_at": r.get("generated_at"),
        }));
    }
    out.sort_by(|a, b| {
        let av = parse_locale_float(a.get("projected_value")).unwrap_or(0.0);
        let bv = parse_locale_float(b.get("projected_value")).unwrap_or(0.0);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
    });
    json_no_store(json!({"rows": out}))
}

// ── /api/skaters/current-projections ──
async fn api_skaters_current_projections(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = season_param(&params, "season", current_season_id(None));

    let proj_map = p::load_current_player_projections_cached(&state).await;
    let roster_map = rosters::all_rosters_for_season(&state.caches, &state.http, season).await;
    let ctx_rows = rapm::load_context_rows(&state.caches, state.sb.as_ref(), &state.cfg).await;

    let mut ctx_by_pid: HashMap<i64, Value> = HashMap::new();
    for row in &ctx_rows {
        let row_season = safe_int(row.get("Season")).unwrap_or(0);
        if season > 0 && row_season != 0 && row_season != season {
            continue;
        }
        if str_value(row.get("StrengthState")).trim() != "5v5" {
            continue;
        }
        let pid = safe_int(row.get("PlayerID")).unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        ctx_by_pid.insert(pid, json!({
            "QoT": parse_locale_float(row.get("QoT_blend_xG67_G33")),
            "QoC": parse_locale_float(row.get("QoC_blend_xG67_G33")),
            "ZS": parse_locale_float(row.get("ZS_Difficulty")),
        }));
    }

    let mut players: Vec<Value> = Vec::new();
    for (k, raw) in &proj_map {
        let pid = safe_int(raw.get("player_id")).or_else(|| safe_int(raw.get("playerId"))).unwrap_or(*k);
        if pid <= 0 {
            continue;
        }
        let mut pos = str_value(raw.get("position")).to_uppercase();
        let roster_info = roster_map.get(&pid.to_string()).cloned().unwrap_or(Value::Null);
        if !str_value(roster_info.get("position")).is_empty() {
            pos = str_value(roster_info.get("position")).to_uppercase();
        }
        if pos.starts_with('G') {
            continue;
        }
        let team_abbrev = str_value(roster_info.get("team")).to_uppercase();
        if !team.is_empty() && team_abbrev != team {
            continue;
        }
        let projected_value = parse_locale_float(raw.get("projected_value"));
        let gp = safe_int(raw.get("games_in_window"))
            .or_else(|| safe_int(raw.get("window_games")))
            .or_else(|| safe_int(raw.get("gp")))
            .unwrap_or(0);
        let pname = str_value(raw.get("player"));
        let pname = if pname.is_empty() {
            str_value(roster_info.get("name"))
        } else {
            pname
        };
        players.push(json!({
            "playerId": pid,
            "name": pname,
            "team": team_abbrev,
            "position": pos,
            "gp": gp,
            "projectedValue": projected_value,
            "projection": projected_value,
            "contextData": ctx_by_pid.get(&pid).cloned().unwrap_or_else(|| json!({"QoT": Value::Null, "QoC": Value::Null, "ZS": Value::Null})),
        }));
    }
    players.sort_by(|a, b| {
        let av = parse_locale_float(a.get("projectedValue")).unwrap_or(f64::NEG_INFINITY);
        let bv = parse_locale_float(b.get("projectedValue")).unwrap_or(f64::NEG_INFINITY);
        bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| str_value(a.get("name")).cmp(&str_value(b.get("name"))).reverse())
    });
    json_no_store(json!({"players": players, "season": season}))
}

// ── /api/teams/current-projections ──
async fn api_teams_current_projections(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let season = season_param(&params, "season", current_season_id(None));
    let proj_map = p::load_current_player_projections_cached(&state).await;
    let skaters = rosters::skater_bios(&state.caches, &state.http, season).await;
    let goalies = rosters::goalie_bios(&state.caches, &state.http, season).await;
    let mut roster_map = skaters.as_object().cloned().unwrap_or_default();
    if let Some(g) = goalies.as_object() {
        for (k, v) in g {
            roster_map.insert(k.clone(), v.clone());
        }
    }
    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;

    let mut lineup_pids_by_team: HashMap<String, HashSet<i64>> = HashMap::new();
    if let Some(obj) = lineups_all.as_object() {
        for (team_abbrev, node) in obj {
            let team_key = team_abbrev.to_uppercase();
            let mut ids = HashSet::new();
            for bucket in ["forwards", "defense", "goalies"] {
                if let Some(arr) = node.get(bucket).and_then(|a| a.as_array()) {
                    for player in arr {
                        if let Some(pid) = safe_int(player.get("playerId")) {
                            if pid > 0 {
                                ids.insert(pid);
                            }
                        }
                    }
                }
            }
            lineup_pids_by_team.insert(team_key, ids);
        }
    }

    let mut players: Vec<Value> = Vec::new();
    for (k, raw) in &proj_map {
        let pid = safe_int(raw.get("player_id")).or_else(|| safe_int(raw.get("playerId"))).unwrap_or(*k);
        if pid <= 0 {
            continue;
        }
        let info = roster_map.get(&pid.to_string()).cloned().unwrap_or(Value::Null);
        let team_abbrev = str_value(info.get("team")).to_uppercase();
        if team_abbrev.is_empty() {
            continue;
        }
        let mut pos = str_value(raw.get("position")).to_uppercase();
        if pos.is_empty() {
            pos = str_value(info.get("position")).to_uppercase();
        }
        let pos = if pos.starts_with('L') || pos.starts_with('R') || pos.starts_with('C') {
            "F".to_string()
        } else if pos.starts_with('D') {
            "D".to_string()
        } else if pos.starts_with('G') {
            "G".to_string()
        } else {
            let f = str_value(info.get("position")).to_uppercase();
            if f.is_empty() { "F".to_string() } else { f }
        };
        let projection = parse_locale_float(raw.get("projected_value"));
        let age = age_from_birthdate(info.get("birthDate"));
        let name = str_value(info.get("name"));
        let name = if name.is_empty() { str_value(raw.get("player")) } else { name };
        players.push(json!({
            "playerId": pid,
            "name": name,
            "team": team_abbrev,
            "position": pos,
            "projection": projection,
            "age": age,
            "inCurrentLineup": lineup_pids_by_team.get(&team_abbrev).map(|s| s.contains(&pid)).unwrap_or(false),
        }));
    }
    players.sort_by(|a, b| {
        str_value(a.get("team")).cmp(&str_value(b.get("team")))
            .then_with(|| str_value(a.get("position")).cmp(&str_value(b.get("position"))))
            .then_with(|| {
                let av = parse_locale_float(a.get("projection")).unwrap_or(-9999.0);
                let bv = parse_locale_float(b.get("projection")).unwrap_or(-9999.0);
                bv.partial_cmp(&av).unwrap_or(std::cmp::Ordering::Equal)
            })
            .then_with(|| str_value(a.get("name")).cmp(&str_value(b.get("name"))))
    });
    json_no_store(json!({"players": players, "season": season}))
}

fn age_from_birthdate(birth_date: Option<&Value>) -> Option<f64> {
    let raw = str_value(birth_date);
    if raw.len() < 10 {
        return None;
    }
    let born = chrono::NaiveDate::parse_from_str(&raw[..10], "%Y-%m-%d").ok()?;
    let today = chrono::Utc::now().date_naive();
    let days = (today - born).num_days();
    if days <= 0 {
        return None;
    }
    Some(days as f64 / 365.2425)
}

// ── /api/skaters/player-projection-trend/<player_id> ──
async fn api_skaters_player_projection_trend(
    State(state): State<AppState>,
    Path(player_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    if player_id <= 0 {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "invalid_player_id"}));
    }
    let season = season_param(&params, "season", current_season_id(None));
    if season <= 0 {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "invalid_season"}));
    }
    let Some(sb) = state.sb.as_ref() else {
        return json_err(StatusCode::NOT_FOUND, json!({"error": "not_found"}));
    };
    let rows = sb
        .read(
            "nhl_player_metrics",
            "*",
            Some(&filters_static(&[
                ("season", format!("eq.{season}")),
                ("nhl_api_player_id", format!("eq.{player_id}")),
            ])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    if rows.is_empty() {
        return json_err(StatusCode::NOT_FOUND, json!({"error": "not_found"}));
    }

    let want_ss = q(&params, "strengthState", "");
    let rows = if !want_ss.is_empty() && want_ss.to_lowercase() != "all" {
        let allowed: Option<HashSet<&str>> = match want_ss.to_lowercase().as_str() {
            "5v5" => Some(["5v5"].iter().copied().collect()),
            "pp" => Some(["5v4", "4v3", "5v3"].iter().copied().collect()),
            "sh" => Some(["4v5", "3v4", "3v5"].iter().copied().collect()),
            "other" => Some(["4v4", "3v3"].iter().copied().collect()),
            _ => None,
        };
        match allowed {
            Some(allowed) => rows.into_iter().filter(|r| {
                let ss = str_value(r.get("strengthstate"));
                allowed.contains(ss.as_str())
            }).collect(),
            None => rows,
        }
    } else {
        rows
    };

    let want_stage = q(&params, "seasonState", "").to_lowercase();
    let rows = if !want_stage.is_empty() && want_stage != "all" {
        if want_stage == "regular" {
            rows.into_iter().filter(|r| {
                let s = str_value(r.get("seasonstage")).to_lowercase();
                s.is_empty() || s == "regular" || s == "reg"
            }).collect()
        } else {
            rows.into_iter().filter(|r| {
                let s = str_value(r.get("seasonstage")).to_lowercase();
                s == "playoffs" || s == "playoff" || s == "po"
            }).collect()
        }
    } else {
        rows
    };
    if rows.is_empty() {
        return json_err(StatusCode::NOT_FOUND, json!({"error": "not_found"}));
    }

    // Group by gameid.
    let mut games: BTreeMap<i64, Value> = BTreeMap::new();
    for r in &rows {
        let gid = safe_int(r.get("gameid")).unwrap_or(0);
        if gid <= 0 {
            continue;
        }
        let entry = games.entry(gid).or_insert_with(|| json!({
            "gameid": gid,
            "prior_games": safe_int(r.get("prior_games")).unwrap_or(0),
            "team": str_value(r.get("team")),
            "position": str_value(r.get("position")),
            "player_name": str_value(r.get("nhl_player_name")),
            "rows": Vec::<Value>::new(),
        }));
        if let Some(arr) = entry.get_mut("rows").and_then(|a| a.as_array_mut()) {
            arr.push(r.clone());
        }
    }
    if games.is_empty() {
        return json_err(StatusCode::NOT_FOUND, json!({"error": "not_found"}));
    }

    let mut points: Vec<Value> = Vec::new();
    let mut player_name = String::new();
    let mut position = String::new();
    let mut team_abbrev = String::new();

    for (idx, (gid, g)) in games.iter().enumerate() {
        let pg = safe_int(g.get("prior_games")).unwrap_or(0);
        let gp = (pg + 1).min(41);
        let game_weight = gp as f64 / 41.0;

        let rows = g.get("rows").and_then(|a| a.as_array()).cloned().unwrap_or_default();
        let mut roll = [0.0f64; 7];
        let mut perf = [0.0f64; 7];
        for r in &rows {
            let rc = component_contribs(r, false);
            let pc = component_contribs(r, true);
            for i in 0..7 {
                roll[i] += rc[i];
                perf[i] += pc[i];
            }
        }
        let comp_keys = ["evo", "evd", "pp", "sh", "gax", "gsax", "rookie"];
        let mut components = serde_json::Map::new();
        let mut perf_components = serde_json::Map::new();
        for (i, k) in comp_keys.iter().enumerate() {
            components.insert(k.to_string(), json!(round6(roll[i] * game_weight)));
            perf_components.insert(k.to_string(), json!(round6(perf[i])));
        }
        let projection = round6(roll.iter().sum::<f64>() * game_weight);
        let performance = round6(perf.iter().sum::<f64>());

        if player_name.is_empty() {
            player_name = str_value(g.get("player_name"));
        }
        if position.is_empty() {
            position = str_value(g.get("position")).to_uppercase();
        }
        if team_abbrev.is_empty() {
            team_abbrev = str_value(g.get("team")).to_uppercase();
        }

        points.push(json!({
            "gameNumber": idx + 1,
            "gameId": gid,
            "projection": projection,
            "performance": performance,
            "components": Value::Object(components),
            "perf_components": Value::Object(perf_components),
            "priorGames": pg,
            "gp": gp,
        }));
    }

    json_no_store(json!({
        "playerId": player_id,
        "season": season,
        "name": player_name,
        "position": position,
        "team": team_abbrev,
        "points": points,
        "source": "nhl_player_metrics",
    }))
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

/// Component contributions (evo, evd, pp, sh, gax, gsax, rookie). Index order
/// matches `comp_keys` above.
fn component_contribs(r: &Value, use_gs: bool) -> [f64; 7] {
    let ss = str_value(r.get("strengthstate"));
    let pos = str_value(r.get("position"));
    let prefix = if use_gs { "gs_" } else { "" };
    let f = |key: &str| parse_locale_float(r.get(&format!("{prefix}{key}"))).unwrap_or(0.0);

    let mut out = [0.0f64; 7];
    let in_ev = p::EV_SS.contains(&ss.as_str());
    let in_pp = p::PP_SS.contains(&ss.as_str());
    let in_sh = p::SH_SS.contains(&ss.as_str());
    if in_ev {
        out[0] = (f("faceoffs") + f("passes") + f("carries")) * p::coef("poss_value_ev");
        out[1] = (f("defensive") + f("dump_ins_outs")) * p::coef("poss_value_ev") + f("xga") * p::coef("xga_ev");
    }
    if in_pp {
        out[2] = f("faceoffs") * p::coef("poss_value_st") + f("xgf") * p::coef("xgf_pp");
    }
    if in_sh {
        out[3] = (f("faceoffs") + f("defensive") + f("dump_ins_outs")) * p::coef("poss_value_st")
            + f("off_the_puck") * p::coef("off_the_puck_sh")
            + f("xga") * p::coef("xga_sh");
    }
    out[4] = f("gax") * p::coef("gax");
    out[5] = f("gsax") * p::coef("gsax");
    if !use_gs && ss == "5v5" {
        let rv = if pos == "D" {
            parse_locale_float(r.get("rookie_d").or_else(|| r.get("rookie"))).unwrap_or(0.0)
        } else if pos == "G" {
            parse_locale_float(r.get("rookie_g").or_else(|| r.get("rookie"))).unwrap_or(0.0)
        } else {
            parse_locale_float(r.get("rookie_f").or_else(|| r.get("rookie"))).unwrap_or(0.0)
        };
        if rv > 0.0 {
            out[6] = rv * match pos.as_str() {
                "D" => p::coef("rookie_d"),
                "G" => p::coef("rookie_g"),
                _ => p::coef("rookie_f"),
            };
        }
    }
    out
}

fn filters_static(pairs: &[(&str, String)]) -> BTreeMap<String, String> {
    pairs.iter().map(|(k, v)| (k.to_string(), v.clone())).collect()
}

// ── /api/projections/team-season-points ──
async fn api_projections_team_season_points(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = season_param(&params, "season", 20252026);
    if team.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "team_required"}));
    }
    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_v2_player_projections_cached(&state).await;
    let custom_lineups = custom_lineups_cache_get(&state, &season).await;
    let team_proj_map = p::team_proj_map_for_season(&lineups_all, &proj_map, &custom_lineups, &state);
    let (points, games) = p::projected_points_for_team(&state, &team, season, &team_proj_map, None, None, None).await;
    json_no_store(json!({
        "team": team,
        "season": season,
        "games": games,
        "projectedPoints": (points * 1000.0).round() / 1000.0,
        "model": "v2_g_diff_evppsh",
    }))
}

// ── /api/projections/team-season-points-custom ──
async fn api_projections_team_season_points_custom(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let team = str_value(body.get("team")).to_uppercase();
    if team.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "team_required"}));
    }
    let season = safe_int(body.get("season")).unwrap_or(20252026);
    let lineup_raw = body.get("lineup").cloned().unwrap_or(Value::Null);
    if !lineup_raw.is_array() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "lineup_must_be_array"}));
    }

    let (key, set_cookie) = actor_key(get_cookie_header(&headers));
    let mut custom_lineups = custom_lineups_cache_get_for(&state, &season, &key).await;
    custom_lineups.insert(team.clone(), p::normalize_custom_lineup_entries(Some(&lineup_raw)));
    custom_lineups_cache_set(&state, &season, &key, &custom_lineups).await;

    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_v2_player_projections_cached(&state).await;
    let team_proj_map = p::team_proj_map_for_season(&lineups_all, &proj_map, &custom_lineups, &state);
    let injuries = p::normalize_injuries(body.get("injuries"));
    let lineup_entries = custom_lineups.get(&team).cloned().unwrap_or_default();
    let (points, games) = p::projected_points_for_team(
        &state, &team, season, &team_proj_map,
        Some(&lineup_entries), Some(&proj_map),
        if injuries.is_empty() { None } else { Some(&injuries) },
    ).await;
    let resp = json_no_store(json!({
        "team": team,
        "season": season,
        "games": games,
        "projectedPoints": (points * 1000.0).round() / 1000.0,
        "model": "v2_g_diff_evppsh",
    }));
    attach_cookie(resp, set_cookie)
}

// ── /api/projections/all-teams-custom ──
async fn api_projections_all_teams_custom(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let team = str_value(body.get("team")).to_uppercase();
    let season = safe_int(body.get("season")).unwrap_or(20252026);
    let lineup_raw = body.get("lineup").cloned().unwrap_or(Value::Null);
    if !lineup_raw.is_null() && !lineup_raw.is_array() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "lineup_must_be_array"}));
    }
    let (key, set_cookie) = actor_key(get_cookie_header(&headers));
    let mut custom_lineups = custom_lineups_cache_get_for(&state, &season, &key).await;

    if body.get("lineupsByTeam").is_some() && body.get("lineupsByTeam").unwrap().is_object() {
        custom_lineups = p::normalize_custom_lineups_by_team(body.get("lineupsByTeam"));
    }
    if !team.is_empty() && lineup_raw.is_array() {
        custom_lineups.insert(team.clone(), p::normalize_custom_lineup_entries(Some(&lineup_raw)));
    }
    custom_lineups_cache_set(&state, &season, &key, &custom_lineups).await;

    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_v2_player_projections_cached(&state).await;
    let team_proj_map = p::team_proj_map_for_season(&lineups_all, &proj_map, &custom_lineups, &state);

    let all_team_abbrevs = p::all_team_abbrevs(&state);
    if all_team_abbrevs.is_empty() {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "no_teams_found"}));
    }
    let mut teams: Vec<Value> = Vec::new();
    for team_i in &all_team_abbrevs {
        let (points_i, games_i) = p::projected_points_for_team(&state, team_i, season, &team_proj_map, None, None, None).await;
        teams.push(json!({
            "team": team_i,
            "projectedPoints": (points_i * 1000.0).round() / 1000.0,
            "games": games_i,
        }));
    }
    teams.sort_by(|a, b| str_value(a.get("team")).cmp(&str_value(b.get("team"))));
    let resp = json_no_store(json!({
        "customTeam": team,
        "season": season,
        "model": "v2_g_diff_evppsh",
        "cachedCustomTeams": custom_lineups.keys().cloned().collect::<Vec<_>>(),
        "teams": teams,
    }));
    attach_cookie(resp, set_cookie)
}

// ── /api/projections/custom-lineups-cache (GET/POST) ──
async fn api_projections_custom_lineups_cache_get(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let season = season_param(&params, "season", 20252026);
    let (key, set_cookie) = actor_key(get_cookie_header(&headers));
    let custom = custom_lineups_cache_get_for(&state, &season, &key).await;
    let resp = json_no_store(json!({
        "season": season,
        "teams": custom.keys().cloned().collect::<Vec<_>>(),
        "lineupsByTeam": custom,
    }));
    attach_cookie(resp, set_cookie)
}

async fn api_projections_custom_lineups_cache_post(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let season = safe_int(body.get("season")).unwrap_or(20252026);
    let (key, set_cookie) = actor_key(get_cookie_header(&headers));

    if let Some(lbt) = body.get("lineupsByTeam") {
        if lbt.is_object() {
            let custom = p::normalize_custom_lineups_by_team(Some(lbt));
            custom_lineups_cache_set(&state, &season, &key, &custom).await;
            return attach_cookie(json_no_store(json!({
                "ok": true,
                "season": season,
                "mode": "replace",
                "teams": custom.keys().cloned().collect::<Vec<_>>(),
                "teamCount": custom.len(),
            })), set_cookie);
        }
    }
    let team = str_value(body.get("team")).to_uppercase();
    let lineup_raw = body.get("lineup").cloned().unwrap_or(Value::Null);
    if team.is_empty() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "team_required"}));
    }
    if !lineup_raw.is_array() {
        return json_err(StatusCode::BAD_REQUEST, json!({"error": "lineup_must_be_array"}));
    }
    let mut custom = custom_lineups_cache_get_for(&state, &season, &key).await;
    custom.insert(team.clone(), p::normalize_custom_lineup_entries(Some(&lineup_raw)));
    custom_lineups_cache_set(&state, &season, &key, &custom).await;
    let lineup_size = custom.get(&team).map(|v| v.len()).unwrap_or(0);
    attach_cookie(json_no_store(json!({
        "ok": true,
        "season": season,
        "mode": "upsert",
        "team": team,
        "lineupSize": lineup_size,
        "teams": custom.keys().cloned().collect::<Vec<_>>(),
    })), set_cookie)
}

async fn custom_lineups_cache_get(state: &AppState, season: &i64) -> BTreeMap<String, Vec<Value>> {
    custom_lineups_cache_get_for(state, season, "anon:local").await
}

async fn custom_lineups_cache_get_for(
    state: &AppState,
    season: &i64,
    key: &str,
) -> BTreeMap<String, Vec<Value>> {
    let ck = (key.to_string(), *season);
    if let Some(v) = state.caches.custom_lineups.get(&ck) {
        return p::normalize_custom_lineups_by_team(Some(&v));
    }
    BTreeMap::new()
}

async fn custom_lineups_cache_set(
    state: &AppState,
    season: &i64,
    key: &str,
    lineups_by_team: &BTreeMap<String, Vec<Value>>,
) {
    let safe = p::normalize_custom_lineups_by_team(Some(&json!(lineups_by_team)));
    state
        .caches
        .custom_lineups
        .insert((key.to_string(), *season), json!(safe));
}

// ── /api/projections/games ──
async fn api_projections_games(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    use chrono::TimeZone;
    let which = q(&params, "which", "today").to_lowercase();
    let now_et = chrono_tz::America::New_York
        .from_utc_datetime(&chrono::Utc::now().naive_utc());
    let date_et = match which.as_str() {
        "yesterday" => now_et.date_naive() - chrono::Duration::days(1),
        "tomorrow" => now_et.date_naive() + chrono::Duration::days(1),
        _ => now_et.date_naive(),
    };
    let date_str = date_et.format("%Y-%m-%d").to_string();

    let url = format!("{API_WEB}/v1/schedule/{date_str}");
    let js = match get_json(&state.http, &url, 20).await {
        Ok(v) => v,
        Err(_) => return json_err(StatusCode::BAD_GATEWAY, json!({"games": [], "date": date_str, "error": "fetch_failed"})),
    };

    let mut logo_by_abbrev: HashMap<String, String> = HashMap::new();
    for tr in state.teams.iter() {
        logo_by_abbrev.insert(str_value(tr.get("Team")).to_uppercase(), str_value(tr.get("Logo")));
    }

    let to_et = |iso_utc: &str| -> Option<String> {
        let s = iso_utc.replace('Z', "+00:00");
        let dt = chrono::DateTime::parse_from_rfc3339(&s).ok()?;
        Some(dt.with_timezone(&chrono_tz::America::New_York).to_rfc3339())
    };

    let mut out: Vec<Value> = Vec::new();
    if let Some(game_week) = js.get("gameWeek").and_then(|v| v.as_array()) {
        for wk in game_week {
            if str_value(wk.get("date")).get(..10) != Some(date_str.as_str()) {
                continue;
            }
            for g in wk.get("games").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                let home = g.get("homeTeam").cloned().unwrap_or(Value::Null);
                let away = g.get("awayTeam").cloned().unwrap_or(Value::Null);
                let ha = str_value(home.get("abbrev")).to_uppercase();
                let aa = str_value(away.get("abbrev")).to_uppercase();
                out.push(game_payload(&g, &ha, &aa, &logo_by_abbrev, &to_et));
            }
        }
    }
    if out.is_empty() {
        if let Some(games) = js.get("games").and_then(|v| v.as_array()) {
            for g in games {
                let st = str_value(g.get("startTimeUTC").or_else(|| g.get("gameDate")));
                if st.len() < 10 || st.replace('Z', "")[..10] != date_str {
                    continue;
                }
                let home = g.get("homeTeam").cloned().unwrap_or(Value::Null);
                let away = g.get("awayTeam").cloned().unwrap_or(Value::Null);
                let ha = str_value(home.get("abbrev")).to_uppercase();
                let aa = str_value(away.get("abbrev")).to_uppercase();
                out.push(game_payload(&g, &ha, &aa, &logo_by_abbrev, &to_et));
            }
        }
    }

    // B2B via previous date.
    let prev_date = (date_et - chrono::Duration::days(1)).format("%Y-%m-%d").to_string();
    let mut prev_set: HashSet<String> = HashSet::new();
    let prev_url = format!("{API_WEB}/v1/schedule/{prev_date}");
    if let Ok(js2) = get_json(&state.http, &prev_url, 20).await {
        for wk in js2.get("gameWeek").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
            if str_value(wk.get("date")).get(..10) != Some(prev_date.as_str()) {
                continue;
            }
            for g in wk.get("games").and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                if let Some(a) = g.get("homeTeam").and_then(|t| t.get("abbrev")).and_then(Value::as_str) {
                    prev_set.insert(a.to_uppercase());
                }
                if let Some(a) = g.get("awayTeam").and_then(|t| t.get("abbrev")).and_then(Value::as_str) {
                    prev_set.insert(a.to_uppercase());
                }
            }
        }
    }

    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_current_player_projections_cached(&state).await;

    let situation_for = |aa: &str, ha: &str| -> (String, f64, bool, bool) {
        let a_b2b = prev_set.contains(&aa.to_uppercase());
        let h_b2b = prev_set.contains(&ha.to_uppercase());
        let (key, val) = match (a_b2b, h_b2b) {
            (true, true) => ("Away-B2B-B2B", -0.126602018),
            (true, false) => ("Away-B2B-Rested", -0.400515738),
            (false, true) => ("Away-Rested-B2B", 0.174538991),
            (false, false) => ("Away-Rested-Rested", -0.153396566),
        };
        (key.to_string(), val, a_b2b, h_b2b)
    };

    for g in &mut out {
        let aa = str_value(g.get("awayTeam").and_then(|t| t.get("abbrev")));
        let ha = str_value(g.get("homeTeam").and_then(|t| t.get("abbrev")));
        let proj_away = p::team_proj_from_lineup(&aa, &lineups_all, &proj_map);
        let proj_home = p::team_proj_from_lineup(&ha, &lineups_all, &proj_map);
        let dproj = proj_away - proj_home;
        let (key, sval, a_b2b, h_b2b) = situation_for(&aa, &ha);
        let win_away = 1.0 / (1.0 + (-(dproj) - sval).exp());
        let win_home = 1.0 - win_away;
        g["b2bAway"] = json!(a_b2b);
        g["b2bHome"] = json!(h_b2b);
        g["projections"] = json!({
            "projAway": round6(proj_away),
            "projHome": round6(proj_home),
            "dProj": round6(dproj),
            "situationKey": key,
            "situationValue": (sval * 1e9).round() / 1e9,
            "winProbAway": round6(win_away),
            "winProbHome": round6(win_home),
        });
    }

    json_no_store(json!({"date": date_str, "timezone": "ET", "games": out}))
}

fn game_payload(
    g: &Value,
    ha: &str,
    aa: &str,
    logo_by_abbrev: &HashMap<String, String>,
    to_et: &dyn Fn(&str) -> Option<String>,
) -> Value {
    let home = g.get("homeTeam").cloned().unwrap_or(Value::Null);
    let away = g.get("awayTeam").cloned().unwrap_or(Value::Null);
    let st = str_value(g.get("startTimeUTC"));
    let start_et = to_et(&st);
    json!({
        "id": g.get("id"),
        "season": g.get("season"),
        "gameType": g.get("gameType"),
        "startTimeUTC": st,
        "startTimeET": start_et,
        "gameState": g.get("gameState").or_else(|| g.get("gameStatus")).cloned(),
        "venue": g.get("venue"),
        "homeTeam": {"abbrev": ha, "score": home.get("score"), "logo": logo_by_abbrev.get(ha).cloned().unwrap_or_default()},
        "awayTeam": {"abbrev": aa, "score": away.get("score"), "logo": logo_by_abbrev.get(aa).cloned().unwrap_or_default()},
        "periodDescriptor": g.get("periodDescriptor"),
    })
}

// ── /api/projections/simulate-season ──
async fn api_projections_simulate_season(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let season = safe_int(body.get("season")).unwrap_or(20252026);
    let seed = safe_int(body.get("seed"));
    let mut rng = match seed {
        Some(s) => Mt19937::new(s as u32),
        None => Mt19937::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| (d.as_nanos() & 0xffffffff) as u32)
                .unwrap_or(1),
        ),
    };

    let (key, set_cookie) = actor_key(get_cookie_header(&headers));
    let mut custom_lineups = custom_lineups_cache_get_for(&state, &season, &key).await;
    if body.get("lineupsByTeam").map(|v| v.is_object()).unwrap_or(false) {
        custom_lineups = p::normalize_custom_lineups_by_team(body.get("lineupsByTeam"));
    }
    let team = str_value(body.get("team")).to_uppercase();
    let lineup_raw = body.get("lineup").cloned().unwrap_or(Value::Null);
    if !team.is_empty() && lineup_raw.is_array() {
        custom_lineups.insert(team.clone(), p::normalize_custom_lineup_entries(Some(&lineup_raw)));
    }
    custom_lineups_cache_set(&state, &season, &key, &custom_lineups).await;

    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_v2_player_projections_cached(&state).await;
    let team_proj_map = p::team_proj_map_for_season(&lineups_all, &proj_map, &custom_lineups, &state);

    let teams = p::active_team_abbrevs(&state);
    if teams.len() < 2 {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "not_enough_teams"}));
    }

    let (team_games, b2b_sets, schedule, team_rosters) =
        build_sim_inputs(&state, &teams, &lineups_all, &proj_map, &custom_lineups, season).await;

    let mut injuries_by_team: HashMap<String, Vec<Value>> = HashMap::new();
    if !team.is_empty() {
        injuries_by_team.insert(team.clone(), p::normalize_injuries(body.get("injuries")));
    }

    let sim_result = p::run_single_sim(
        &state,
        &schedule,
        &team_proj_map,
        &b2b_sets,
        &team_rosters,
        &teams,
        &proj_map,
        season,
        &mut rng,
        if injuries_by_team.is_empty() { None } else { Some(&injuries_by_team) },
        Some(&custom_lineups),
    )
    .await;

    let resp = json_no_store(json!({
        "season": season,
        "standings": sim_result.get("standings"),
        "playoffSeeds": sim_result.get("playoffSeeds"),
        "playoffs": sim_result.get("playoffs"),
        "playerStats": sim_result.get("playerStats"),
        "simulatedGames": schedule.len(),
        "model": "v2_g_diff_evppsh",
    }));
    attach_cookie(resp, set_cookie)
}

async fn build_sim_inputs(
    state: &AppState,
    teams: &[String],
    lineups_all: &Value,
    proj_map: &HashMap<i64, Value>,
    custom_lineups: &BTreeMap<String, Vec<Value>>,
    season: i64,
) -> (
    HashMap<String, Vec<Value>>,
    HashMap<String, HashSet<String>>,
    Vec<Value>,
    HashMap<String, Vec<Value>>,
) {
    let mut team_games: HashMap<String, Vec<Value>> = HashMap::new();
    for t in teams {
        team_games.insert(t.clone(), p::fetch_club_schedule_games(state, t, season).await);
    }
    let b2b_sets = p::b2b_date_sets(&team_games);

    let mut by_id: HashMap<i64, Value> = HashMap::new();
    for (t, games) in &team_games {
        for g in games {
            if safe_int(g.get("gameType")).unwrap_or(0) != 2 {
                continue;
            }
            if let Some(gid) = safe_int(g.get("id")) {
                by_id.entry(gid).or_insert_with(|| g.clone());
            }
        }
    }
    let mut schedule: Vec<Value> = by_id.into_values().collect();
    schedule.sort_by(|a, b| {
        str_value(a.get("date")).cmp(&str_value(b.get("date")))
            .then_with(|| str_value(a.get("id")).cmp(&str_value(b.get("id"))))
    });

    let mut team_rosters: HashMap<String, Vec<Value>> = HashMap::new();
    for t in teams {
        team_rosters.insert(t.clone(), p::build_team_roster_rates(t, lineups_all, proj_map, custom_lineups.get(t).map(|v| v.as_slice())));
    }
    (team_games, b2b_sets, schedule, team_rosters)
}

// ── /api/projections/simulate-season-batch ──
async fn api_projections_simulate_season_batch(
    State(state): State<AppState>,
    headers: axum::http::HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let season = safe_int(body.get("season")).unwrap_or(20252026);
    let num_sims = safe_int(body.get("numSims")).unwrap_or(100).clamp(1, 10000);
    let base_seed = safe_int(body.get("seed")).unwrap_or_else(|| {
        let mut rng = Mt19937::new(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| (d.as_nanos() & 0xffffffff) as u32)
                .unwrap_or(1),
        );
        rng.randrange(i32::MAX as usize) as i64
    });

    let (key, set_cookie) = actor_key(get_cookie_header(&headers));
    let mut custom_lineups: BTreeMap<String, Vec<Value>> = BTreeMap::new();
    if body.get("lineupsByTeam").map(|v| v.is_object()).unwrap_or(false) {
        custom_lineups = p::normalize_custom_lineups_by_team(body.get("lineupsByTeam"));
    }
    let team_ab = str_value(body.get("team")).to_uppercase();
    let lineup_raw = body.get("lineup").cloned().unwrap_or(Value::Null);
    if !team_ab.is_empty() && lineup_raw.is_array() {
        custom_lineups.insert(team_ab.clone(), p::normalize_custom_lineup_entries(Some(&lineup_raw)));
    }

    let lineups_all = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    let proj_map = p::load_v2_player_projections_cached(&state).await;
    let team_proj_map = p::team_proj_map_for_season(&lineups_all, &proj_map, &custom_lineups, &state);

    let teams = p::active_team_abbrevs(&state);
    if teams.len() < 2 {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "not_enough_teams"}));
    }

    let (_, b2b_sets, schedule, team_rosters) =
        build_sim_inputs(&state, &teams, &lineups_all, &proj_map, &custom_lineups, season).await;
    if schedule.is_empty() {
        return json_err(StatusCode::INTERNAL_SERVER_ERROR, json!({"error": "no_schedule_found"}));
    }

    let mut injuries_by_team: HashMap<String, Vec<Value>> = HashMap::new();
    if !team_ab.is_empty() {
        injuries_by_team.insert(team_ab.clone(), p::normalize_injuries(body.get("injuries")));
    }

    // Team aggregation accumulators.
    let mut team_agg: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for t in &teams {
        team_agg.insert(t.clone(), vec![0.0; 13]);
    }
    let mut player_agg: BTreeMap<i64, (String, String, String, i64, f64, f64, f64, f64)> = BTreeMap::new();

    for i in 0..num_sims {
        let mut rng = Mt19937::new((base_seed as u32).wrapping_add(i as u32));
        let sim = p::run_single_sim(
            &state,
            &schedule,
            &team_proj_map,
            &b2b_sets,
            &team_rosters,
            &teams,
            &proj_map,
            season,
            &mut rng,
            if injuries_by_team.is_empty() { None } else { Some(&injuries_by_team) },
            Some(&custom_lineups),
        )
        .await;

        for row in sim.get("standings").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
            let t = str_value(row.get("team"));
            let Some(a) = team_agg.get_mut(&t) else { continue };
            a[0] += safe_int(row.get("gp")).unwrap_or(0) as f64;
            a[1] += safe_int(row.get("wins")).unwrap_or(0) as f64;
            a[2] += safe_int(row.get("losses")).unwrap_or(0) as f64;
            a[3] += safe_int(row.get("otLosses")).unwrap_or(0) as f64;
            a[4] += safe_int(row.get("points")).unwrap_or(0) as f64;
            a[5] += safe_int(row.get("goalsFor")).unwrap_or(0) as f64;
            a[6] += safe_int(row.get("goalsAgainst")).unwrap_or(0) as f64;
            a[7] += safe_int(row.get("goalDifferential")).unwrap_or(0) as f64;
        }

        let po = sim.get("playoffs").cloned().unwrap_or(Value::Null);
        for t in &teams {
            let mut made = false;
            let mut r2 = false;
            let mut r3 = false;
            let mut final_ = false;
            let mut champion = false;
            for conf in ["East", "West"] {
                for s in po.get("round1").and_then(|r| r.get(conf)).and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                    if str_value(s.get("winner")) == *t || str_value(s.get("loser")) == *t {
                        made = true;
                    }
                }
                for s in po.get("round2").and_then(|r| r.get(conf)).and_then(|v| v.as_array()).cloned().unwrap_or_default() {
                    if str_value(s.get("winner")) == *t || str_value(s.get("loser")) == *t {
                        r2 = true;
                    }
                }
                let s = po.get("conferenceFinals").and_then(|r| r.get(conf)).cloned();
                if let Some(s) = s {
                    if str_value(s.get("winner")) == *t || str_value(s.get("loser")) == *t {
                        r3 = true;
                    }
                }
            }
            let sf = po.get("stanleyFinal").cloned();
            if let Some(sf) = sf {
                if str_value(sf.get("winner")) == *t || str_value(sf.get("loser")) == *t {
                    final_ = true;
                    if str_value(sf.get("winner")) == *t {
                        champion = true;
                    }
                }
            }
            let a = team_agg.get_mut(t).unwrap();
            if made { a[8] += 1.0; }
            if r2 { a[9] += 1.0; }
            if r3 { a[10] += 1.0; }
            if final_ { a[11] += 1.0; }
            if champion { a[12] += 1.0; }
        }

        for pstat in sim.get("playerStats").and_then(|s| s.as_array()).cloned().unwrap_or_default() {
            let pid = safe_int(pstat.get("pid")).unwrap_or(0);
            if pid <= 0 {
                continue;
            }
            let entry = player_agg.entry(pid).or_insert_with(|| (
                str_value(pstat.get("name")),
                str_value(pstat.get("team")),
                str_value(pstat.get("position")),
                safe_int(pstat.get("gp")).unwrap_or(82),
                0.0, 0.0, 0.0, 0.0,
            ));
            entry.4 += safe_int(pstat.get("goals")).unwrap_or(0) as f64;
            entry.5 += safe_int(pstat.get("a1")).unwrap_or(0) as f64;
            entry.6 += safe_int(pstat.get("a2")).unwrap_or(0) as f64;
            entry.7 += safe_int(pstat.get("points")).unwrap_or(0) as f64;
        }
    }

    let n = num_sims as f64;
    let r1 = |x: f64| (x * 10.0).round() / 10.0;
    let mut team_list: Vec<Value> = Vec::new();
    for t in &teams {
        let a = &team_agg[t];
        team_list.push(json!({
            "team": t,
            "conference": p::team_conf(t),
            "avgGp": r1(a[0] / n),
            "avgWins": r1(a[1] / n),
            "avgLosses": r1(a[2] / n),
            "avgOtLosses": r1(a[3] / n),
            "avgPoints": r1(a[4] / n),
            "avgGoalsFor": r1(a[5] / n),
            "avgGoalsAgainst": r1(a[6] / n),
            "avgGoalDifferential": r1(a[7] / n),
            "sumPlayoffs": a[8] as i64,
            "sumSecondRound": a[9] as i64,
            "sumThirdRound": a[10] as i64,
            "sumFinal": a[11] as i64,
            "sumChampion": a[12] as i64,
        }));
    }
    team_list.sort_by(|a, b| {
        parse_locale_float(a.get("avgPoints")).unwrap_or(0.0)
            .partial_cmp(&parse_locale_float(b.get("avgPoints")).unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then_with(|| {
                parse_locale_float(a.get("avgWins")).unwrap_or(0.0)
                    .partial_cmp(&parse_locale_float(b.get("avgWins")).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            })
            .then_with(|| str_value(a.get("team")).cmp(&str_value(b.get("team"))))
    });

    let mut player_list: Vec<Value> = Vec::new();
    for (pid, (name, team, position, gp, goals, a1, a2, pts)) in &player_agg {
        player_list.push(json!({
            "pid": pid, "name": name, "team": team, "position": position, "gp": gp,
            "avgGoals": r1(goals / n),
            "avgA1": r1(a1 / n),
            "avgA2": r1(a2 / n),
            "avgPoints": r1(pts / n),
        }));
    }
    player_list.sort_by(|a, b| {
        parse_locale_float(a.get("avgPoints")).unwrap_or(0.0)
            .partial_cmp(&parse_locale_float(b.get("avgPoints")).unwrap_or(0.0))
            .unwrap_or(std::cmp::Ordering::Equal)
            .reverse()
            .then_with(|| {
                parse_locale_float(a.get("avgGoals")).unwrap_or(0.0)
                    .partial_cmp(&parse_locale_float(b.get("avgGoals")).unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .reverse()
            })
            .then_with(|| str_value(a.get("name")).cmp(&str_value(b.get("name"))))
    });

    let resp = json_no_store(json!({
        "numSims": num_sims,
        "season": season,
        "teamAgg": team_list,
        "playerAgg": player_list,
        "model": "v2_g_diff_evppsh",
    }));
    attach_cookie(resp, set_cookie)
}
