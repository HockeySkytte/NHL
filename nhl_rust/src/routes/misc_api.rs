//! M1 API routes: proxies and simple JSON endpoints (§5.2 of PORT_PLAN.md).
//! Ports of the corresponding Flask handlers in `app/routes.py`.

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::extract::{Path, Query, State};
use axum::http::{header, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::data::{lineups, odds, player_projections, rosters, teamseasonstats};
use crate::nhl::client::{get_bytes, get_json, API_STATS, API_WEB};
use crate::nhl::stats_rest::summary_rows;
use crate::state::AppState;
use crate::util::dates::current_season_id;
use crate::util::parse::{
    ci_get, flag_param, parse_locale_float, parse_season_ids, primary_season_id, safe_int,
    str_value,
};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/standings/{season}", get(api_standings))
        .route("/api/live-games", get(api_live_games))
        .route("/api/seasons/{team_code}", get(api_seasons))
        .route("/api/schedule/{team_code}/{season}", get(api_schedule_alias))
        .route("/api/team/{team_code}/{season}/schedule", get(api_team_schedule))
        .route("/api/roster/{team_code}/current", get(api_roster_current))
        .route("/api/player/{player_id}/landing", get(api_player_landing))
        .route("/api/game/{game_id}/boxscore", get(api_game_boxscore))
        .route("/api/game/{game_id}/right-rail", get(api_game_right_rail))
        .route("/api/team-logo/{team_abbrev}", get(api_team_logo_svg))
        .route("/api/player-headshot/{player_id}", get(api_player_headshot_png))
        .route("/api/diag/models", get(api_diag_models))
        .route("/api/player-projections/{player_id}", get(api_player_projections))
        .route("/api/player-projections/sheets", get(api_player_projections_sheets))
        .route("/api/player-projections/league", get(api_player_projections_league))
        .route("/api/lineups/all", get(api_lineups_all))
        .route("/api/odds/history/{game_id}", get(api_odds_history))
        .route("/api/skaters/players", get(api_skaters_players))
        .route("/api/goalies/players", get(api_goalies_players))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [(header::CACHE_CONTROL, "no-store")], Json(v)).into_response()
}

fn json_err(status: StatusCode, v: Value) -> Response {
    (status, Json(v)).into_response()
}

fn normalize_standings(data: Value) -> Value {
    if let Some(standings) = data.get("standings").cloned() {
        return json!({ "standings": standings.as_array().cloned().unwrap_or_default() });
    }
    if data.is_array() {
        return json!({ "standings": data });
    }
    json!({ "standings": [] })
}

/// `GET /api/standings/<season>` — port of `api_standings`.
async fn api_standings(State(state): State<AppState>, Path(season): Path<i64>) -> Response {
    let current = current_season_id(None);
    let mut urls: Vec<String> = Vec::new();
    if season == current {
        urls.push(format!("{API_WEB}/v1/standings/now"));
    }
    if let Some(last_date) = state.last_dates.get(&season) {
        urls.push(format!("{API_WEB}/v1/standings/{last_date}"));
        urls.push(format!("{API_WEB}/v1/standings/{last_date}?gameType=2"));
    }
    urls.extend([
        format!("{API_WEB}/v1/standings/{season}"),
        format!("{API_WEB}/v1/standings/{season}?gameType=2"),
        format!("{API_WEB}/v1/standings-season/{season}"),
        format!("{API_WEB}/v1/standings-season/{season}?gameType=2"),
        format!("{API_WEB}/v1/standings?season={season}"),
        format!("{API_WEB}/v1/standings?season={season}&gameType=2"),
    ]);

    let mut last_status: Option<u16> = None;
    for url in &urls {
        match get_json(&state.http, url, 25).await {
            Ok(data) => return json_no_store(normalize_standings(data)),
            Err(e) => {
                last_status = extract_status(&e).or(last_status);
                continue;
            }
        }
    }
    if season == current {
        if let Ok(data) = get_json(&state.http, &format!("{API_WEB}/v1/standings/now"), 25).await {
            return json_no_store(normalize_standings(data));
        }
    }

    // Final fallback: stats REST standings by season.
    let stats_url = format!(
        "{API_STATS}/team/standings?isAggregate=false&reportType=basic&isGame=true&reportName=teamstandings&cayenneExp=seasonId={season}%20and%20gameTypeId=2"
    );
    if let Ok(js) = get_json(&state.http, &stats_url, 30).await {
        let rows = js.get("data").and_then(Value::as_array).cloned().unwrap_or_default();
        let mut logo_by_abbrev: HashMap<String, String> = HashMap::new();
        for tr in state.teams.iter() {
            let ab = str_value(tr.get("Team")).to_uppercase();
            if !ab.is_empty() {
                logo_by_abbrev.insert(ab, str_value(tr.get("Logo")));
            }
        }
        let mut out: Vec<Value> = Vec::new();
        for rrow in &rows {
            let Some(obj) = rrow.as_object() else { continue };
            let ab = ci_get(obj, "teamAbbrev")
                .or_else(|| ci_get(obj, "teamAbbrevDefault"))
                .map(|v| str_value(Some(v)).to_uppercase())
                .unwrap_or_default();
            let gp = safe_int(ci_get(obj, "gamesPlayed").or_else(|| ci_get(obj, "gp"))).unwrap_or(0);
            let pts = safe_int(ci_get(obj, "points").or_else(|| ci_get(obj, "pts"))).unwrap_or(0);
            let w = safe_int(ci_get(obj, "wins").or_else(|| ci_get(obj, "w"))).unwrap_or(0);
            let l = safe_int(ci_get(obj, "losses").or_else(|| ci_get(obj, "l"))).unwrap_or(0);
            let otl = safe_int(
                ci_get(obj, "otLosses")
                    .or_else(|| ci_get(obj, "otl"))
                    .or_else(|| ci_get(obj, "overtimeLosses")),
            )
            .unwrap_or(0);
            let ties = safe_int(ci_get(obj, "ties")).unwrap_or(0);
            let gf = safe_int(ci_get(obj, "goalsFor").or_else(|| ci_get(obj, "gf"))).unwrap_or(0);
            let ga = safe_int(ci_get(obj, "goalsAgainst").or_else(|| ci_get(obj, "ga"))).unwrap_or(0);
            let diff = gf - ga;
            let mut ppct = parse_locale_float(
                ci_get(obj, "pointsPercentage").or_else(|| ci_get(obj, "pointPctg")),
            );
            if ppct.is_none() {
                ppct = if gp > 0 {
                    Some(pts as f64 / (2.0 * gp as f64))
                } else {
                    Some(0.0)
                };
            }
            let l10w = safe_int(ci_get(obj, "lastTenWins").or_else(|| ci_get(obj, "l10Wins"))).unwrap_or(0);
            let l10l = safe_int(ci_get(obj, "lastTenLosses").or_else(|| ci_get(obj, "l10Losses"))).unwrap_or(0);
            let l10o = safe_int(ci_get(obj, "lastTenOtLosses").or_else(|| ci_get(obj, "l10OtLosses"))).unwrap_or(0);
            let streak_code = str_value(ci_get(obj, "streakCode").or_else(|| ci_get(obj, "streakType")));
            let streak_num = safe_int(ci_get(obj, "streakNumber").or_else(|| ci_get(obj, "streakCount"))).unwrap_or(0);
            let div_name = str_value(ci_get(obj, "divisionName").or_else(|| ci_get(obj, "divisionAbbrev")));
            let conf_name = str_value(ci_get(obj, "conferenceName").or_else(|| ci_get(obj, "conferenceAbbrev")));
            out.push(json!({
                "teamAbbrev": ab,
                "divisionName": div_name,
                "conferenceName": conf_name,
                "gamesPlayed": gp,
                "points": pts,
                "wins": w,
                "losses": l,
                "ties": ties,
                "otLosses": otl,
                "goalFor": gf,
                "goalAgainst": ga,
                "goalDifferential": diff,
                "pointPctg": ppct,
                "l10Wins": l10w,
                "l10Losses": l10l,
                "l10OtLosses": l10o,
                "streakCode": if streak_code.is_empty() { "" } else { &streak_code[..1] },
                "streakCount": streak_num,
                "teamLogo": logo_by_abbrev.get(&ab).cloned().unwrap_or_default(),
            }));
        }
        return json_no_store(json!({ "standings": out }));
    }
    json_err(
        StatusCode::BAD_GATEWAY,
        json!({ "error": "Upstream error", "status": last_status }),
    )
}

/// `GET /api/live-games` — port of `api_live_games`.
async fn api_live_games(State(state): State<AppState>) -> Response {
    let url = format!("{API_WEB}/v1/schedule/now");
    let data = match get_json(&state.http, &url, 20).await {
        Ok(data) => data,
        Err(_) => {
            return json_err(
                StatusCode::BAD_GATEWAY,
                json!({ "games": [], "error": "Fetch failed" }),
            )
        }
    };
    let live_states = ["LIVE", "INPROGRESS", "CRIT", "OT", "SHOOTOUT"];
    let mut out: Vec<Value> = Vec::new();
    if let Some(weeks) = data.get("gameWeek").and_then(Value::as_array) {
        for wk in weeks {
            if let Some(games) = wk.get("games").and_then(Value::as_array) {
                for g in games {
                    let st = str_value(g.get("gameState")).to_uppercase();
                    if !live_states.contains(&st.as_str()) {
                        continue;
                    }
                    out.push(json!({
                        "id": g.get("id"),
                        "season": g.get("season"),
                        "gameType": g.get("gameType"),
                        "startTimeUTC": g.get("startTimeUTC"),
                        "gameState": g.get("gameState"),
                        "venue": g.get("venue"),
                        "awayTeam": g.get("awayTeam"),
                        "homeTeam": g.get("homeTeam"),
                        "periodDescriptor": g.get("periodDescriptor"),
                    }));
                }
            }
        }
    }
    json_no_store(json!({ "games": out }))
}

/// `GET /api/seasons/<team_code>` — port of `api_seasons`.
async fn api_seasons(State(state): State<AppState>, Path(team_code): Path<String>) -> Response {
    let team = team_code.trim().to_uppercase();
    if team.is_empty() {
        return json_no_store(json!([]));
    }
    if let Some(cached) = state.caches.team_seasons.get(&team) {
        return json_no_store(cached);
    }

    let mut season_map: BTreeMap<i64, HashSet<String>> = BTreeMap::new();
    for row in teamseasonstats::iter_rows(&state.caches, state.sb.as_ref(), &state.cfg).await {
        let row_team = str_value(row.get("Team").or_else(|| row.get("team"))).to_uppercase();
        if row_team != team {
            continue;
        }
        let season_i = safe_int(row.get("Season").or_else(|| row.get("season")));
        let game_type = season_state_to_game_type(
            str_value(row.get("SeasonState").or_else(|| row.get("season_state"))),
        );
        if let (Some(season_i), Some(game_type)) = (season_i, game_type) {
            season_map.entry(season_i).or_default().insert(game_type);
        }
    }

    // Supplement with NHL API to discover seasons not yet in our data.
    let url = format!("{API_WEB}/v1/club-stats-season/{team}");
    if let Ok(data) = get_json(&state.http, &url, 20).await {
        if let Some(list) = data.as_array() {
            for it in list {
                let s_int = safe_int(it.get("season"));
                let gtypes = it.get("gameTypes").and_then(Value::as_array);
                if let (Some(s_int), Some(gtypes)) = (s_int, gtypes) {
                    let gts: HashSet<String> = gtypes
                        .iter()
                        .filter_map(|gt| gt.as_str().map(|s| s.to_string()))
                        .collect();
                    season_map
                        .entry(s_int)
                        .or_default()
                        .extend(gts);
                }
            }
        }
    }

    let mut out: Vec<Value> = season_map
        .iter()
        .map(|(season_i, game_types)| {
            let mut sorted: Vec<String> = game_types.iter().cloned().collect();
            sorted.sort();
            json!({ "season": season_i, "gameTypes": sorted })
        })
        .collect();
    out.sort_by(|a, b| b["season"].as_i64().cmp(&a["season"].as_i64()));

    state.caches.team_seasons.insert(team, json!(out));
    json_no_store(json!(out))
}

fn season_state_to_game_type(v: String) -> Option<String> {
    let raw = v.trim().to_lowercase();
    match raw.as_str() {
        "2" | "reg" | "regular" | "regularseason" | "regular_season" => Some("2".to_string()),
        "3" | "po" | "playoffs" | "playoff" => Some("3".to_string()),
        _ => None,
    }
}

fn normalize_game_date(game_date: Option<&str>) -> String {
    let Some(d) = game_date else { return String::new() };
    let d = d.trim();
    if d.is_empty() {
        return String::new();
    }
    if let Some(rest) = d.strip_suffix('Z') {
        // `datetime.fromisoformat(d.replace('Z', '+00:00')).isoformat()`
        if rest.contains('T') {
            return format!("{rest}+00:00");
        }
        return rest.to_string();
    }
    d.to_string()
}

/// `GET /api/team/<team_code>/<season>/schedule` — port of `api_team_schedule`.
async fn api_team_schedule(
    State(state): State<AppState>,
    Path((team_code, season)): Path<(String, i64)>,
) -> Response {
    let team = team_code.trim().to_uppercase();
    let url = format!("{API_WEB}/v1/club-schedule-season/{team}/{season}");
    let data = match get_json(&state.http, &url, 20).await {
        Ok(data) => data,
        Err(_) => return json_err(StatusCode::BAD_GATEWAY, json!({ "error": "Failed to fetch schedule" })),
    };
    let mut games_out: Vec<Value> = Vec::new();
    if let Some(games) = data.get("games").and_then(Value::as_array) {
        for g in games {
            let Some(obj) = g.as_object() else { continue };
            let game_date = str_value(ci_get(obj, "gameDate").or_else(|| ci_get(obj, "startTimeUTC")));
            let date = normalize_game_date(Some(&game_date));
            let home = ci_get(obj, "homeTeam")
                .and_then(|t| t.get("abbrev"))
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let away = ci_get(obj, "awayTeam")
                .and_then(|t| t.get("abbrev"))
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            let home_score = ci_get(obj, "homeTeam").and_then(|t| t.get("score")).cloned();
            let away_score = ci_get(obj, "awayTeam").and_then(|t| t.get("score")).cloned();
            let opp = if home.as_deref() == Some(team.as_str()) { away.clone() } else { home.clone() };
            let is_home = home.as_deref() == Some(team.as_str());
            let status = str_value(ci_get(obj, "gameState").or_else(|| ci_get(obj, "gameStatus")));
            let last_period_type = ci_get(obj, "gameOutcome")
                .and_then(|o| o.get("lastPeriodType"))
                .or_else(|| {
                    ci_get(obj, "periodDescriptor").and_then(|p| p.get("periodType"))
                })
                .and_then(Value::as_str)
                .map(|s| s.to_string());
            games_out.push(json!({
                "date": date,
                "home": home,
                "away": away,
                "opponent": opp,
                "is_home": is_home,
                "status": status,
                "gameType": ci_get(obj, "gameType").or_else(|| ci_get(obj, "gameTypeId")).cloned(),
                "home_score": home_score,
                "away_score": away_score,
                "lastPeriodType": last_period_type,
                "id": ci_get(obj, "id").or_else(|| ci_get(obj, "gamePk")).cloned(),
            }));
        }
    }
    json_no_store(json!(games_out))
}

/// `GET /api/schedule/<team_code>/<season>` — alias.
async fn api_schedule_alias(
    state: State<AppState>,
    path: Path<(String, i64)>,
) -> Response {
    api_team_schedule(state, path).await
}

/// `GET /api/roster/<team_code>/current` — port of `api_roster_current`.
async fn api_roster_current(State(state): State<AppState>, Path(team_code): Path<String>) -> Response {
    let team = team_code.trim().to_uppercase();
    if team.is_empty() {
        return json_no_store(json!({ "forwards": [], "defensemen": [], "goalies": [] }));
    }
    let url = format!("{API_WEB}/v1/roster/{team}/current");
    let data = match get_json(&state.http, &url, 20).await {
        Ok(data) => data,
        Err(_) => {
            return json_err(
                StatusCode::BAD_GATEWAY,
                json!({ "forwards": [], "defensemen": [], "goalies": [], "error": "fetch_failed" }),
            )
        }
    };
    json_no_store(json!({
        "forwards": data.get("forwards").cloned().unwrap_or_else(|| json!([])),
        "defensemen": data.get("defensemen").cloned().unwrap_or_else(|| json!([])),
        "goalies": data.get("goalies").cloned().unwrap_or_else(|| json!([])),
    }))
}

async fn build_bios_fallback_payload(
    state: &AppState,
    pid: i64,
    season_value: Option<&str>,
    team_value: Option<&str>,
) -> Option<Value> {
    let season_i = season_value.and_then(|s| s.trim().parse::<i64>().ok()).unwrap_or(0);
    let season_i = if season_i > 0 { season_i } else { current_season_id(None) };
    let team_q = str_value(team_value.map(Value::from).as_ref()).to_uppercase();

    let mut candidates: Vec<Value> = Vec::new();
    let sk = rosters::skater_bios(&state.caches, &state.http, season_i).await;
    let gk = rosters::goalie_bios(&state.caches, &state.http, season_i).await;
    for info in [&sk, &gk] {
        if let Some(v) = info.get(&pid.to_string()) {
            if v.is_object() && !v.as_object().map(|o| o.is_empty()).unwrap_or(true) {
                candidates.push(v.clone());
            }
        }
    }

    if !team_q.is_empty() {
        candidates.sort_by_key(|info| {
            if str_value(info.get("team")).to_uppercase() == team_q {
                0
            } else {
                1
            }
        });
    }

    let info = candidates.into_iter().next()?;
    let name = str_value(info.get("name"));
    if name.is_empty() {
        return None;
    }
    let parts: Vec<&str> = name.split_whitespace().collect();
    let first_name = parts.first().map(|s| s.to_string()).unwrap_or_default();
    let last_name = if parts.len() > 1 {
        parts[1..].join(" ")
    } else {
        String::new()
    };
    let mut payload = json!({
        "playerId": pid,
        "firstName": { "default": first_name },
        "lastName": { "default": last_name },
        "partial": true,
        "fallbackSource": "bios",
    });
    let birth_date = str_value(info.get("birthDate"));
    if !birth_date.is_empty() {
        payload["birthDate"] = json!(birth_date);
    }
    let position_code = str_value(info.get("positionCode").or_else(|| info.get("position"))).to_uppercase();
    if !position_code.is_empty() {
        payload["position"] = json!(position_code);
    }
    let shoots = str_value(info.get("shoots")).to_uppercase();
    if !shoots.is_empty() {
        payload["shootsCatches"] = json!(shoots);
    }
    let team_name = str_value(info.get("team")).to_uppercase();
    if !team_name.is_empty() {
        payload["currentTeamAbbrev"] = json!(team_name);
    }
    Some(payload)
}

/// `GET /api/player/<player_id>/landing` — port of `api_player_landing`.
async fn api_player_landing(
    State(state): State<AppState>,
    Path(player_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let pid = player_id;
    if pid <= 0 {
        return json_err(StatusCode::BAD_REQUEST, json!({ "error": "invalid_player_id" }));
    }
    let season_q = params.get("season").map(String::as_str).unwrap_or("");
    let team_q = params.get("team").map(String::as_str).unwrap_or("").to_uppercase();

    // Serve fresh cache.
    if let Some(cached) = state.caches.player_landing.get(&pid) {
        return json_no_store(cached);
    }

    // Stale-cache / bios fallbacks.
    async fn fallback(state: &AppState, pid: i64, season_q: &str, team_q: &str) -> Option<Response> {
        if let Some(stale) = state.caches.player_landing.get(&pid) {
            let mut payload = stale.clone();
            if let Some(obj) = payload.as_object_mut() {
                obj.insert("stale".into(), json!(true));
                if !obj.contains_key("fallbackSource") {
                    obj.insert("fallbackSource".into(), json!("stale-cache"));
                }
            }
            return Some(json_no_store(payload));
        }
        let bios_payload =
            build_bios_fallback_payload(state, pid, Some(season_q), Some(team_q)).await;
        if bios_payload.is_some() {
            return Some(json_no_store(bios_payload.unwrap()));
        }
        None
    }

    let url = format!("{API_WEB}/v1/player/{pid}/landing");
    let data = match get_json(&state.http, &url, 8).await {
        Ok(data) => data,
        Err(e) => {
            if let Some(fb) = fallback(&state, pid, season_q, &team_q).await {
                return fb;
            }
            return json_err(StatusCode::BAD_GATEWAY, json!({ "error": upstream_error_code(&e) }));
        }
    };
    if !data.is_object() {
        if let Some(fb) = fallback(&state, pid, season_q, &team_q).await {
            return fb;
        }
        return json_err(StatusCode::BAD_GATEWAY, json!({ "error": "invalid_upstream" }));
    }
    state.caches.player_landing.insert(pid, data.clone());
    json_no_store(data)
}

fn upstream_error_code(e: &str) -> String {
    if e.contains("fetch_failed") {
        "fetch_failed".to_string()
    } else if e.contains("upstream_status") {
        "upstream_error".to_string()
    } else {
        "invalid_upstream".to_string()
    }
}

fn extract_status(e: &str) -> Option<u16> {
    e.strip_prefix("upstream_status:")
        .and_then(|s| s.trim().parse().ok())
}

/// `GET /api/game/<game_id>/boxscore` — port of `api_game_boxscore`.
async fn api_game_boxscore(
    State(state): State<AppState>,
    Path(game_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let force = flag_param(&params_to_map(&params), "force", "0");
    if !force {
        if let Some(cached) = state.caches.box_cache.get(&game_id) {
            return json_no_store(cached);
        }
    }
    let url = format!("{API_WEB}/v1/gamecenter/{game_id}/boxscore");
    let mut data = match get_json(&state.http, &url, 20).await {
        Ok(data) => data,
        Err(_) => return json_err(StatusCode::BAD_GATEWAY, json!({ "error": "Fetch failed" })),
    };
    // Rename id → gameId for consistency.
    if data.get("id").is_some() && data.get("gameId").is_none() {
        if let Some(id) = data.get("id").cloned() {
            data.as_object_mut()
                .expect("boxscore object")
                .insert("gameId".into(), id);
        }
    }
    state.caches.box_cache.insert(game_id, data.clone());
    if force {
        json_no_store(data)
    } else {
        Json(data).into_response()
    }
}

/// `GET /api/game/<game_id>/right-rail` — port of `api_game_right_rail`.
async fn api_game_right_rail(
    State(state): State<AppState>,
    Path(game_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let force = flag_param(&params_to_map(&params), "force", "0");
    let url = format!("{API_WEB}/v1/gamecenter/{game_id}/right-rail");
    let data = match get_json(&state.http, &url, 20).await {
        Ok(data) => data,
        Err(_) => return json_err(StatusCode::BAD_GATEWAY, json!({ "error": "Upstream error" })),
    };
    if force {
        json_no_store(data)
    } else {
        Json(data).into_response()
    }
}

fn params_to_map(params: &HashMap<String, String>) -> serde_json::Map<String, Value> {
    let mut out = serde_json::Map::new();
    for (k, v) in params {
        out.insert(k.clone(), Value::String(v.clone()));
    }
    out
}

fn host_of(url: &str) -> Option<String> {
    let rest = url.split_once("://")?.1;
    let host = rest.split(['/', '?', '#']).next()?;
    Some(host.to_lowercase())
}

fn is_allowed_assets_url(url: &str) -> bool {
    let Some((scheme, _)) = url.split_once("://") else {
        return false;
    };
    if scheme != "http" && scheme != "https" {
        return false;
    }
    host_of(url).map(|h| h == "assets.nhle.com").unwrap_or(false)
}

fn svg_response(data: Vec<u8>) -> Response {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "image/svg+xml"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        data,
    )
        .into_response()
}

fn png_response(data: Vec<u8>) -> Response {
    (
        StatusCode::OK,
        [
            (header::CONTENT_TYPE, "image/png"),
            (header::CACHE_CONTROL, "public, max-age=86400"),
        ],
        data,
    )
        .into_response()
}

fn not_found() -> Response {
    (StatusCode::NOT_FOUND, "").into_response()
}

/// `GET /api/team-logo/<abbrev>.svg` — port of `api_team_logo_svg`.
async fn api_team_logo_svg(State(state): State<AppState>, Path(raw): Path<String>) -> Response {
    let trimmed = raw.trim();
    let a = trimmed
        .strip_suffix(".svg")
        .or_else(|| trimmed.strip_suffix(".SVG"))
        .unwrap_or(trimmed)
        .to_uppercase();
    if a.len() < 2 || a.len() > 4 || !a.chars().all(|c| c.is_ascii_alphabetic()) {
        return not_found();
    }
    if let Some(cached) = state.caches.team_logo.get(&a) {
        return svg_response(cached);
    }
    let mut src = crate::nhl::images::team_logo_source_url(&state.teams, &a)
        .unwrap_or_else(|| format!("https://assets.nhle.com/logos/nhl/svg/{a}_light.svg"));
    if !is_allowed_assets_url(&src) {
        src = format!("https://assets.nhle.com/logos/nhl/svg/{a}_light.svg");
    }
    match get_bytes(&state.http, &src, 10).await {
        Ok(raw) => {
            let data = match String::from_utf8(raw) {
                Ok(txt) => crate::nhl::images::normalize_svg_dimensions(&txt).into_bytes(),
                Err(bytes) => bytes.into_bytes(),
            };
            state.caches.team_logo.insert(a, data.clone());
            svg_response(data)
        }
        Err(e) => {
            tracing::warn!("team logo fetch failed for {a}: {e}");
            not_found()
        }
    }
}

/// `GET /api/player-headshot/<player_id>.png` — port of `api_player_headshot_png`.
async fn api_player_headshot_png(
    State(state): State<AppState>,
    Path(raw): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let trimmed = raw.trim();
    let raw_id = trimmed
        .strip_suffix(".png")
        .or_else(|| trimmed.strip_suffix(".PNG"))
        .unwrap_or(trimmed);
    let Ok(pid) = raw_id.parse::<i64>() else {
        return not_found();
    };
    if pid <= 0 {
        return not_found();
    }
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let team_abbrev = params.get("team").map(|s| s.to_uppercase()).unwrap_or_default();
    let cache_key = format!("{pid}|{season}|{team_abbrev}");
    if let Some(cached) = state.caches.player_headshot.get(&cache_key) {
        return png_response(cached);
    }
    for src in crate::nhl::images::player_headshot_source_urls(pid, season, &team_abbrev) {
        if !is_allowed_assets_url(&src) {
            continue;
        }
        match get_bytes(&state.http, &src, 10).await {
            Ok(data) => {
                if data.is_empty() {
                    continue;
                }
                state.caches.player_headshot.insert(cache_key.clone(), data.clone());
                return png_response(data);
            }
            Err(e) => {
                tracing::warn!("headshot fetch failed for {src}: {e}");
                continue;
            }
        }
    }
    not_found()
}

/// `GET /api/diag/models` — port of `api_diag_models`.
async fn api_diag_models(State(state): State<AppState>) -> Response {
    let model_dir = state.cfg.model_dir.clone();
    let mut files: Vec<String> = Vec::new();
    if let Ok(rd) = std::fs::read_dir(&model_dir) {
        for entry in rd.flatten() {
            let name = entry.file_name().to_string_lossy().to_string();
            if name.ends_with(".pkl") {
                files.push(name);
            }
        }
    }
    files.sort();
    json_no_store(json!({
        "python": format!("rust/{}", env!("CARGO_PKG_VERSION")),
        "versions": {
            "numpy": null,
            "pandas": null,
            "sklearn": null,
            "xgboost": null,
            "joblib": null,
        },
        "model_dir": model_dir.display().to_string(),
        "model_count": files.len(),
        "models": files,
    }))
}

/// `GET /api/player-projections/<player_id>` — port of `api_player_projections`.
async fn api_player_projections(
    State(state): State<AppState>,
    Path(player_id): Path<i64>,
) -> Response {
    let pid = player_id;
    if pid <= 0 {
        return json_err(StatusCode::BAD_REQUEST, json!({ "error": "invalid_player_id" }));
    }
    let map = player_projections::load_map(&state.caches, state.sb.as_ref(), &state.cfg).await;
    match map.get(&pid) {
        Some(row) => json_no_store(json!({ "playerId": pid, "row": row })),
        None => json_err(StatusCode::NOT_FOUND, json!({ "error": "not_found" })),
    }
}

/// `GET /api/player-projections/sheets` — port of `api_player_projections_sheets`.
async fn api_player_projections_sheets(State(state): State<AppState>) -> Response {
    let map = player_projections::load_map(&state.caches, state.sb.as_ref(), &state.cfg).await;
    let mut out = serde_json::Map::new();
    for (k, v) in map {
        out.insert(k.to_string(), v);
    }
    json_no_store(Value::Object(out))
}

/// `GET /api/player-projections/league` — port of `api_player_projections_league`.
async fn api_player_projections_league(
    State(state): State<AppState>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let team = params.get("team").map(|s| s.trim().to_uppercase()).unwrap_or_default();
    let include_goalies = flag_param(&params_to_map(&params), "include_goalies", "0");

    let proj_map = player_projections::load_map(&state.caches, state.sb.as_ref(), &state.cfg).await;
    let roster_map = rosters::all_rosters(&state.caches, &state.http).await;

    let mut out: Vec<Value> = Vec::new();
    for (pid, raw) in &proj_map {
        let row = player_projections::parse_proj_row(raw);
        if row.get("playerId").is_none() {
            continue;
        }
        let pos = str_value(row.get("position")).to_uppercase();
        if !include_goalies && pos.starts_with('G') {
            continue;
        }
        let info = roster_map.get(&pid.to_string()).cloned().unwrap_or_else(|| json!({}));
        let t = str_value(info.get("team")).to_uppercase();
        if !team.is_empty() {
            if !t.is_empty() && t != team {
                continue;
            }
            if t.is_empty() {
                continue;
            }
        }
        let pos_final = if matches!(pos.as_str(), "F" | "D" | "G") {
            pos.clone()
        } else {
            str_value(info.get("position"))
        };
        out.push(json!({
            "playerId": row.get("playerId"),
            "position": row.get("position"),
            "gp": row.get("gp"),
            "Age": row.get("Age"),
            "Rookie": row.get("Rookie"),
            "EVO": row.get("EVO"),
            "EVD": row.get("EVD"),
            "PP": row.get("PP"),
            "SH": row.get("SH"),
            "GSAx": row.get("GSAx"),
            "total": row.get("total"),
            "name": str_value(info.get("name")),
            "team": t,
            "position_final": pos_final,
        }));
    }
    out.sort_by(|a, b| {
        let ta = a.get("total").and_then(Value::as_f64).unwrap_or(0.0);
        let tb = b.get("total").and_then(Value::as_f64).unwrap_or(0.0);
        tb.partial_cmp(&ta).unwrap_or(std::cmp::Ordering::Equal)
    });
    json_no_store(json!({ "players": out }))
}

/// `GET /api/lineups/all` — port of `api_lineups_all`.
async fn api_lineups_all(State(state): State<AppState>) -> Response {
    let data = lineups::load_all(&state.caches, state.sb.as_ref(), &state.cfg.static_dir).await;
    json_no_store(data)
}

/// `GET /api/odds/history/<game_id>` — port of `api_odds_history`.
async fn api_odds_history(State(state): State<AppState>, Path(game_id): Path<i64>) -> Response {
    let mut color_by_team: HashMap<String, String> = HashMap::new();
    for row in state.teams.iter() {
        let t = str_value(row.get("Team")).to_uppercase();
        let c = str_value(row.get("Color"));
        if !t.is_empty() && !c.is_empty() {
            color_by_team.insert(t, c);
        }
    }
    let (away_abbrev, home_abbrev) =
        odds::game_team_abbrevs(&state.caches, &state.http, game_id).await;
    let snapshot_rows = odds::load_snapshot_rows(&state.caches, state.sb.as_ref(), game_id).await;
    let (_latest, points_by_team) =
        odds::build_payloads(&snapshot_rows, &away_abbrev, &home_abbrev);

    if !points_by_team.is_empty() {
        let mut teams_out: Vec<Value> = Vec::new();
        for (team_s, points) in &points_by_team {
            teams_out.push(json!({
                "abbrev": team_s,
                "color": color_by_team.get(team_s).cloned().unwrap_or_else(|| "".to_string()),
                "points": points,
            }));
        }
        return json_no_store(json!({ "gameId": game_id, "teams": teams_out }));
    }
    json_no_store(json!({ "gameId": game_id, "teams": [] }))
}

fn cayenne_exp(season_id: i64, season_state: &str, team: Option<&str>) -> String {
    let ss = match season_state {
        "playoffs" => "gameTypeId=3".to_string(),
        "all" => "(gameTypeId=2 or gameTypeId=3)".to_string(),
        _ => "gameTypeId=2".to_string(),
    };
    match team {
        Some(t) => format!("seasonId={season_id} and {ss} and teamAbbrev=\"{t}\""),
        None => format!("seasonId={season_id} and {ss}"),
    }
}

async fn summary_players(
    state: &AppState,
    kind: &str,
    season_ids: &[i64],
    is_league: bool,
    team: &str,
    season_state: &str,
    is_goalie: bool,
) -> Vec<Value> {
    let mut by_pid: BTreeMap<i64, Value> = BTreeMap::new();
    for season_id in season_ids {
        let cay = cayenne_exp(*season_id, season_state, if is_league { None } else { Some(team) });
        let rows = match summary_rows(&state.http, kind, &cay).await {
            Ok(rows) => rows,
            Err(_) => continue,
        };
        for row in rows {
            let pid = if is_goalie {
                safe_int(
                    row.get("playerId")
                        .or_else(|| row.get("goalieId"))
                        .or_else(|| row.get("id")),
                )
            } else {
                safe_int(row.get("playerId"))
            };
            let Some(pid) = pid else { continue };
            if pid <= 0 {
                continue;
            }
            if !is_goalie {
                let pos = str_value(row.get("positionCode")).to_uppercase();
                if pos.starts_with('G') {
                    continue;
                }
            }
            let name = if is_goalie {
                str_value(
                    row.get("goalieFullName")
                        .or_else(|| row.get("playerFullName"))
                        .or_else(|| row.get("skaterFullName")),
                )
            } else {
                str_value(row.get("skaterFullName"))
            };
            let name = if name.is_empty() { pid.to_string() } else { name };
            let mut team_abbrev = str_value(
                row.get("teamAbbrev")
                    .or_else(|| row.get("teamAbbrevs"))
                    .or_else(|| row.get("currentTeamAbbrev")),
            )
            .to_uppercase();
            if let Some((first, _)) = team_abbrev.split_once('/') {
                team_abbrev = first.trim().to_uppercase();
            }
            let mut rec = serde_json::Map::new();
            rec.insert("playerId".into(), json!(pid));
            rec.insert("name".into(), json!(name));
            rec.insert(
                "pos".into(),
                json!(if is_goalie { "G".to_string() } else { str_value(row.get("positionCode")).to_uppercase() }),
            );
            if !team_abbrev.is_empty() {
                rec.insert("team".into(), json!(team_abbrev));
            }
            by_pid.insert(pid, Value::Object(rec));
        }
    }
    by_pid.into_values().collect()
}

/// `GET /api/skaters/players` — port of `api_skaters_players`.
async fn api_skaters_players(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    players_endpoint(state, false, params).await
}

/// `GET /api/goalies/players` — port of `api_goalies_players`.
async fn api_goalies_players(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    players_endpoint(state, true, params).await
}

async fn players_endpoint(state: AppState, is_goalie: bool, params: HashMap<String, String>) -> Response {
    let scope = params
        .get("scope")
        .or_else(|| params.get("playerScope"))
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_else(|| "team".to_string());
    let is_league = matches!(scope.as_str(), "league" | "all" | "full")
        || params
            .get("league")
            .map(|v| matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes"))
            .unwrap_or(false);

    let team = params.get("team").map(|s| s.trim().to_uppercase()).unwrap_or_default();
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let season_state = params
        .get("seasonState")
        .or_else(|| params.get("season_state"))
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_else(|| "regular".to_string());
    if !is_league && team.is_empty() {
        return json_no_store(json!({ "players": [] }));
    }

    let current = current_season_id(None);
    let season_ids = parse_season_ids(Some(season), current);
    let season_i = primary_season_id(&season_ids, current);
    let season_state = if matches!(season_state.as_str(), "regular" | "playoffs" | "all") {
        season_state
    } else {
        "regular".to_string()
    };

    let cache_key = (
        season_ids.clone(),
        if is_league { "__LEAGUE__".to_string() } else { team.clone() },
        season_state.clone(),
    );
    let cache = if is_goalie {
        &state.caches.goalies_players
    } else {
        &state.caches.skaters_players
    };
    if let Some(cached) = cache.get(&cache_key) {
        return json_no_store(cached);
    }

    let mut players =
        summary_players(&state, if is_goalie { "goalie" } else { "skater" }, &season_ids, is_league, &team, &season_state, is_goalie).await;

    // De-dupe by playerId.
    let mut seen: HashSet<i64> = HashSet::new();
    let mut uniq: Vec<Value> = Vec::new();
    for p in players.drain(..) {
        let pid = p.get("playerId").and_then(|v| safe_int(Some(v))).unwrap_or(0);
        if pid <= 0 || !seen.insert(pid) {
            continue;
        }
        uniq.push(p);
    }
    players = uniq;

    // Skater bios fallback (team scope).
    if players.is_empty() && !is_goalie && !is_league {
        let bios = rosters::skater_bios(&state.caches, &state.http, season_i).await;
        if let Some(map) = bios.as_object() {
            for (pid_s, info) in map {
                let pid = pid_s.parse::<i64>().unwrap_or(0);
                if pid <= 0 {
                    continue;
                }
                let t = str_value(info.get("team")).to_uppercase();
                if t != team {
                    continue;
                }
                let name = str_value(info.get("name"));
                let name = if name.is_empty() { pid.to_string() } else { name };
                let pos_code = str_value(info.get("positionCode").or_else(|| info.get("position"))).to_uppercase();
                players.push(json!({ "playerId": pid, "name": name, "pos": pos_code }));
            }
        }
    }

    // Historical roster fallback (older seasons, team scope).
    if players.is_empty() && !is_league && season_i != 0 && current != 0 && season_i != current {
        let url = format!("{API_WEB}/v1/roster/{}/{}", team.to_lowercase(), season_i);
        if let Ok(data) = get_json(&state.http, &url, 20).await {
            let buckets: [&str; 3] = if is_goalie {
                ["goalies", "", ""]
            } else {
                ["forwards", "defensemen", ""]
            };
            for bucket in buckets {
                if bucket.is_empty() {
                    continue;
                }
                if let Some(list) = data.get(bucket).and_then(Value::as_array) {
                    for p in list {
                        let pid = safe_int(p.get("id").or_else(|| p.get("playerId")));
                        let Some(pid) = pid else { continue };
                        if pid <= 0 {
                            continue;
                        }
                        let first = match p.get("firstName") {
                            Some(Value::Object(_)) => str_value(p.get("firstName").and_then(|v| v.get("default"))),
                            _ => str_value(p.get("firstName")),
                        };
                        let last = match p.get("lastName") {
                            Some(Value::Object(_)) => str_value(p.get("lastName").and_then(|v| v.get("default"))),
                            _ => str_value(p.get("lastName")),
                        };
                        let name = format!("{} {}", first.trim(), last.trim()).trim().to_string();
                        let name = if name.is_empty() { pid.to_string() } else { name };
                        let pos = str_value(p.get("positionCode").or_else(|| p.get("position"))).to_uppercase();
                        players.push(json!({ "playerId": pid, "name": name, "pos": if is_goalie { "G".to_string() } else { pos }, "team": team }));
                    }
                }
            }
        }
    }

    cache.insert(cache_key, json!(players));
    json_no_store(json!({ "players": players }))
}
