//! Manager Game routes (M1): the `/manager` page plus league CRUD, join,
//! schedule, and rounds APIs.
//!
//! Access model (confirmed with the user): free for all logged-in users —
//! the premium predicate in `web/auth_state.rs` only matches `/projections`
//! and `/api/projections/`, so no gating change is needed.

use std::collections::{BTreeMap, HashMap};

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::manager::{self, league};
use crate::state::AppState;
use crate::util::parse::{safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/manager", get(manager_page))
        .route("/manager/{league_id}", get(manager_league_page))
        .route("/manager/{league_id}/lineup", get(manager_lineup_page))
        .route("/manager/{league_id}/trades", get(manager_trades_page))
        .route(
            "/api/manager/leagues",
            get(api_my_leagues).post(api_create_league),
        )
        .route("/api/manager/leagues/join", post(api_join_league))
        .route("/api/manager/leagues/{league_id}", get(api_league_detail))
        .route(
            "/api/manager/leagues/{league_id}/schedule",
            get(api_league_schedule),
        )
        .route(
            "/api/manager/leagues/{league_id}/rounds",
            get(api_league_rounds),
        )
        .route(
            "/api/manager/leagues/{league_id}/draft",
            get(api_draft_state),
        )
        .route(
            "/api/manager/leagues/{league_id}/draft/start",
            post(api_draft_start),
        )
        .route(
            "/api/manager/leagues/{league_id}/draft/players",
            get(api_draft_players),
        )
        .route(
            "/api/manager/leagues/{league_id}/draft/pick",
            post(api_draft_pick),
        )
        .route(
            "/api/manager/leagues/{league_id}/lineup",
            get(api_get_lineup).post(api_set_lineup),
        )
        .route(
            "/api/manager/leagues/{league_id}/roster",
            get(api_get_roster),
        )
        .route(
            "/api/manager/games/{nhl_game_id}/stats",
            post(api_game_stats),
        )
        .route(
            "/api/manager/leagues/{league_id}/finalize/{nhl_game_id}",
            post(api_finalize_game),
        )
        .route(
            "/api/manager/leagues/{league_id}/trades/offers",
            get(api_list_offers).post(api_create_offer),
        )
        .route(
            "/api/manager/trades/{offer_id}/respond",
            post(api_respond_offer),
        )
        .route(
            "/api/manager/leagues/{league_id}/standings",
            get(api_standings),
        )
        .route(
            "/api/manager/leagues/{league_id}/feed",
            get(api_feed),
        )
        .route(
            "/api/manager/leagues/{league_id}/sweep",
            post(api_sweep),
        )
        .route(
            "/api/manager/leagues/{league_id}/seed-rosters",
            post(api_seed_rosters),
        )
}

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::HOST).and_then(|v| v.to_str().ok())
}

/// The acting user id from the session; falls back to a dev id when Supabase
/// auth is not configured (mirrors how the premium gate skips without auth).
fn user_id_of(cfg: &crate::config::Config, headers: &HeaderMap) -> Option<String> {
    if let Some(auth) = crate::routes::auth::auth_user_from_headers(cfg, headers) {
        let id = auth
            .get("user_id")
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .filter(|s| !s.is_empty());
        if id.is_some() {
            return id;
        }
    }
    if !crate::supabase::read::auth_is_configured() {
        return Some("local-dev".to_string());
    }
    None
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": "auth_required", "loginUrl": "/login?next=/manager"})),
    )
        .into_response()
}

/// CSRF check for write endpoints (skipped when auth is not configured).
fn csrf_ok(state: &AppState, headers: &HeaderMap) -> bool {
    if !crate::supabase::read::auth_is_configured() {
        return true;
    }
    let session = crate::routes::auth::session_from_headers(&state.cfg, headers);
    let provided = headers
        .get(header::HeaderName::from_static("x-csrf-token"))
        .and_then(|v| v.to_str().ok());
    crate::routes::auth::csrf_validate(&session, provided)
}

fn bad_csrf() -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({"error": "invalid_csrf"})),
    )
        .into_response()
}

/// `GET /manager` — Manager Game hub page.
async fn manager_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra: BTreeMap<&'static str, Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Manager"));
    extra.insert(
        "meta_title",
        json!("Manager Game · Hockey-Statistics"),
    );
    extra.insert(
        "meta_description",
        json!("Create or join an NHL Manager league: true rosters or a 26-round snake draft, trades, and round-based scoring on the real 2026/27 schedule."),
    );
    extra.insert("manager_season_default", json!(manager::DEFAULT_SEASON));
    extra.insert("manager_modes", json!([
        {"value": manager::MODE_TRUE_ROSTERS, "label": "True rosters"},
        {"value": manager::MODE_DRAFT, "label": "Draft (26 rounds, snake)"},
    ]));
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/manager",
        "manager_home.html",
        &mut session,
        extra,
    )
}

/// `GET /api/manager/leagues` — the caller's leagues.
async fn api_my_leagues(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    match league::list_leagues_for_user(&state, &user_id).await {
        Ok(leagues) => crate::routes::auth::json_no_store(json!({"leagues": leagues})),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues` — create a league.
async fn api_create_league(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let name = str_value(body.get("name"));
    let mode = str_value(body.get("mode"));
    let season = safe_int(body.get("season")).unwrap_or(manager::DEFAULT_SEASON);
    let franchise = str_value(body.get("franchise")).to_uppercase();
    match league::create_league(&state, name, mode, season, user_id, franchise).await {
        Ok(detail) => crate::routes::auth::json_no_store(detail),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/join` — join via join code + franchise.
async fn api_join_league(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let code = str_value(body.get("join_code"));
    let franchise = str_value(body.get("franchise"));
    match league::join_league(&state, &code, &franchise, &user_id).await {
        Ok(detail) => crate::routes::auth::json_no_store(detail),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}` — league detail.
async fn api_league_detail(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
) -> Response {
    match league::league_detail(&state, &league_id).await {
        Ok(detail) => crate::routes::auth::json_no_store(detail),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/schedule?team=&round=&status=&from=&to=`
async fn api_league_schedule(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let Some(sb) = state.sb.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response();
    };

    let team = params.get("team").map(|s| s.trim().to_uppercase()).filter(|s| !s.is_empty());
    let round = params
        .get("round")
        .and_then(|s| s.trim().parse::<i64>().ok())
        .filter(|r| *r > 0);

    let mut filters: BTreeMap<String, String> =
        BTreeMap::from([("league_id".to_string(), format!("eq.{league_id}"))]);
    match (team.as_deref(), round) {
        (Some(t), Some(r)) => {
            filters.insert(
                "or".to_string(),
                format!(
                    "(and(home_abbrev.eq.{t},home_round.eq.{r}),and(away_abbrev.eq.{t},away_round.eq.{r}))"
                ),
            );
        }
        (Some(t), None) => {
            filters.insert(
                "or".to_string(),
                format!("(home_abbrev.eq.{t},away_abbrev.eq.{t})"),
            );
        }
        (None, Some(r)) => {
            filters.insert(
                "or".to_string(),
                format!("(home_round.eq.{r},away_round.eq.{r})"),
            );
        }
        (None, None) => {}
    }
    if let Some(status) = params.get("status").map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
        filters.insert("status".to_string(), format!("eq.{status}"));
    }
    let from = params.get("from").map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let to = params.get("to").map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    match (from, to) {
        (Some(f), Some(t)) => {
            filters.insert("date".to_string(), format!("and=(date.gte.{f},date.lte.{t})"));
        }
        (Some(f), None) => {
            filters.insert("date".to_string(), format!("gte.{f}"));
        }
        (None, Some(t)) => {
            filters.insert("date".to_string(), format!("lte.{t}"));
        }
        (None, None) => {}
    }

    let limit = params
        .get("limit")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(250);

    let cols = "nhl_game_id,date,home_abbrev,away_abbrev,home_round,away_round,status,home_score,away_score,finalized_at";
    let rows = sb
        .read(
            "manager_games",
            cols,
            Some(&filters),
            None,
            Some("date.asc,nhl_game_id.asc"),
            limit,
        )
        .await;
    match rows {
        Some(games) => crate::routes::auth::json_no_store(json!({
            "league_id": league_id,
            "games": games,
            "count": games.len(),
        })),
        None => (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "storage_unavailable"})),
        )
            .into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/rounds` — per-team round state: games
/// played, next round, and the next round's scoring window.
async fn api_league_rounds(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
) -> Response {
    let Some(sb) = state.sb.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response();
    };
    let filters: BTreeMap<String, String> =
        BTreeMap::from([("league_id".to_string(), format!("eq.{league_id}"))]);
    let rows = sb
        .read(
            "manager_games",
            "nhl_game_id,date,home_abbrev,away_abbrev,home_round,away_round,status",
            Some(&filters),
            None,
            Some("date.asc,nhl_game_id.asc"),
            0,
        )
        .await;
    let Some(games) = rows else {
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(json!({"error": "storage_unavailable"})),
        )
            .into_response();
    };

    // Per team: ordered list of (round, date, final).
    let mut by_team: BTreeMap<String, Vec<(i64, String, bool)>> = BTreeMap::new();
    for g in &games {
        let home = str_value(g.get("home_abbrev"));
        let away = str_value(g.get("away_abbrev"));
        let date = str_value(g.get("date"));
        let hr = safe_int(g.get("home_round")).unwrap_or(0);
        let ar = safe_int(g.get("away_round")).unwrap_or(0);
        let final_game = str_value(g.get("status")) == "final";
        if !home.is_empty() && hr > 0 {
            by_team
                .entry(home)
                .or_default()
                .push((hr, date.clone(), final_game));
        }
        if !away.is_empty() && ar > 0 {
            by_team.entry(away).or_default().push((ar, date, final_game));
        }
    }

    let mut out: Vec<Value> = Vec::new();
    for (team, mut rounds) in by_team {
        rounds.sort_by_key(|r| r.0);
        let played = rounds.iter().filter(|r| r.2).count() as i64;
        let next_round = played + 1;
        let next = rounds.get(played as usize);
        let prev_date = if played > 0 {
            rounds.get(played as usize - 1).map(|r| r.1.clone())
        } else {
            None
        };
        out.push(json!({
            "team_abbrev": team,
            "played": played,
            "total": rounds.len(),
            "next_round": next_round,
            "next_date": next.map(|r| r.1.clone()).unwrap_or_default(),
            "window_start_exclusive": prev_date,
        }));
    }
    out.sort_by(|a, b| str_value(a.get("team_abbrev")).cmp(&str_value(b.get("team_abbrev"))));
    crate::routes::auth::json_no_store(json!({"league_id": league_id, "teams": out}))
}

/// `GET /manager/{league_id}` — league dashboard (draft room for now).
async fn manager_league_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Result<Response, ApiError> {
    let mut extra: BTreeMap<&'static str, Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Manager"));
    extra.insert("meta_title", json!("League · Manager Game · Hockey-Statistics"));
    extra.insert("meta_description", json!("League dashboard: draft room, schedule, and scoring."));
    extra.insert("manager_league_id", json!(league_id));
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/manager",
        "manager_league.html",
        &mut session,
        extra,
    )
}

/// `GET /manager/{league_id}/lineup` — the lineup selector page.
async fn manager_lineup_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Result<Response, ApiError> {
    let mut extra: BTreeMap<&'static str, Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Manager"));
    extra.insert("meta_title", json!("Lineup · Manager Game · Hockey-Statistics"));
    extra.insert("meta_description", json!("Set your active 12 F / 6 D / 2 G lineup for the round."));
    extra.insert("manager_league_id", json!(league_id));
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/manager",
        "manager_lineup.html",
        &mut session,
        extra,
    )
}

/// `GET /manager/{league_id}/trades` — the trade center page.
async fn manager_trades_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Result<Response, ApiError> {
    let mut extra: BTreeMap<&'static str, Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Manager"));
    extra.insert("meta_title", json!("Trades · Manager Game · Hockey-Statistics"));
    extra.insert("meta_description", json!("Offer and manage player trades in your Manager league."));
    extra.insert("manager_league_id", json!(league_id));
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/manager",
        "manager_trades.html",
        &mut session,
        extra,
    )
}

/// `GET /api/manager/leagues/{id}/draft`
async fn api_draft_state(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Response {
    let viewer = user_id_of(&state.cfg, &headers);
    match crate::manager::draft::draft_state(&state, &league_id, viewer.as_deref()).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/draft/start` (commissioner)
async fn api_draft_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    match crate::manager::draft::start_draft(&state, &league_id, &user_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/draft/players?search=&pos=&team=&limit=&offset=`
async fn api_draft_players(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let search = params.get("search").cloned().unwrap_or_default();
    let pos = params.get("pos").cloned().unwrap_or_default();
    let team = params.get("team").cloned().unwrap_or_default();
    let limit = params
        .get("limit")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(50);
    let offset = params
        .get("offset")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(0);
    match crate::manager::draft::available_players(
        &state, &league_id, &search, &pos, &team, limit, offset,
    )
    .await
    {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/draft/pick` — body: {"player_id": int}
async fn api_draft_pick(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let player_id = safe_int(body.get("player_id")).unwrap_or(0);
    match crate::manager::draft::submit_pick(&state, &league_id, &user_id, player_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/lineup?team=&round=` — read a lineup.
async fn api_get_lineup(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let Some(sb) = state.sb.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response();
    };
    let team = params.get("team").cloned().unwrap_or_default();
    let round = params
        .get("round")
        .and_then(|s| s.trim().parse::<i64>().ok())
        .unwrap_or(0);
    match crate::manager::games::get_lineup(sb, &league_id, &team, round).await {
        Ok(mut v) => {
            // get_lineup returns empty names; fill them from the draft pool so
            // the lineup selector can render player names directly.
            let needs_names = v
                .get("slots")
                .and_then(|s| s.as_object())
                .map(|slots| {
                    slots.values().any(|s| {
                        let empty = s
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map(|n| n.is_empty())
                            .unwrap_or(true);
                        empty && s.get("player_id").and_then(|p| p.as_i64()).unwrap_or(0) > 0
                    })
                })
                .unwrap_or(false);
            if needs_names {
                let pool = crate::manager::draft::draft_pool(&state).await;
                if let Some(slots) = v.get_mut("slots").and_then(|s| s.as_object_mut()) {
                    for s in slots.values_mut() {
                        let empty = s
                            .get("name")
                            .and_then(|n| n.as_str())
                            .map(|n| n.is_empty())
                            .unwrap_or(true);
                        let pid = s.get("player_id").and_then(|p| p.as_i64()).unwrap_or(0);
                        if empty && pid > 0 {
                            if let Some(name) = pool
                                .get(&pid)
                                .and_then(|info| info.get("name"))
                                .and_then(|n| n.as_str())
                            {
                                s["name"] = json!(name);
                            }
                        }
                    }
                }
            }
            crate::routes::auth::json_no_store(v)
        }
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/roster?team=` — a team's roster pool.
async fn api_get_roster(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let team = params.get("team").cloned().unwrap_or_default();
    match crate::manager::games::get_team_roster(&state, &league_id, &team).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/lineup` — set mode-1 lineup.
/// Body: {"team": "...", "round": N, "slots": {"F1": pid, ..., "G2": pid}}
async fn api_set_lineup(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let team = str_value(body.get("team"));
    let round = safe_int(body.get("round")).unwrap_or(0);

    // Only the team's manager (or commissioner) may set its lineup.
    let is_member = match crate::manager::league::membership(&state, &league_id, &user_id).await {
        Some(t) => str_value(t.get("team_abbrev")) == team,
        None => false,
    };
    let is_commish = crate::manager::league::is_commissioner(&state, &league_id, &user_id).await;
    if !is_member && !is_commish {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "forbidden"}))).into_response();
    }

    let mut slots: BTreeMap<String, i64> = BTreeMap::new();
    if let Some(obj) = body.get("slots").and_then(Value::as_object) {
        for (k, v) in obj {
            if let Some(pid) = v.as_i64() {
                slots.insert(k.clone(), pid);
            }
        }
    }
    match crate::manager::games::set_lineup(&state, &league_id, &team, round, &slots).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/games/{nhl_game_id}/stats` — (re)record a game's stat
/// lines from live play-by-play. Returns game_state + rows.
async fn api_game_stats(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(nhl_game_id): Path<i64>,
) -> Response {
    if user_id_of(&state.cfg, &headers).is_none() {
        return unauthorized();
    }
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    match crate::manager::games::record_game_stats(&state, nhl_game_id).await {
        Ok((game_state, rows)) => crate::routes::auth::json_no_store(json!({
            "nhl_game_id": nhl_game_id,
            "game_state": game_state,
            "rows": rows,
            "count": rows.len(),
        })),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/finalize/{nhl_game_id}` — finalize a game.
async fn api_finalize_game(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((league_id, nhl_game_id)): Path<(String, i64)>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let is_commish = crate::manager::league::is_commissioner(&state, &league_id, &user_id).await;
    if !is_commish {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "forbidden"}))).into_response();
    }
    match crate::manager::games::finalize_game(&state, &league_id, nhl_game_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/trades/offers?to_team=` — list offers.
async fn api_list_offers(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let to_team = params.get("to_team").map(|s| s.as_str());
    match crate::manager::trades::list_offers(&state, &league_id, to_team).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/trades/offers` — create an offer.
/// Body: {"from_team": "...", "to_team": "...", "offered_player_ids": [...],
///        "requested_player_ids": [...]}
async fn api_create_offer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let from = str_value(body.get("from_team"));
    let to = str_value(body.get("to_team"));
    let offered: Vec<i64> = body
        .get("offered_player_ids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default();
    let requested: Vec<i64> = body
        .get("requested_player_ids")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|v| v.as_i64()).collect())
        .unwrap_or_default();
    match crate::manager::trades::create_offer(
        &state,
        &league_id,
        &from,
        &to,
        &offered,
        &requested,
        &user_id,
    )
    .await
    {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/trades/{offer_id}/respond` — accept/decline/cancel.
/// Body: {"action": "accept" | "decline" | "cancel", "league_id": "..."}
async fn api_respond_offer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(offer_id): Path<String>,
    Json(body): Json<Value>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let league_id = str_value(body.get("league_id"));
    let action = str_value(body.get("action"));
    match crate::manager::trades::respond_offer(&state, &league_id, &offer_id, &action, &user_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/standings` — W/L standings.
async fn api_standings(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
) -> Response {
    match crate::manager::standings::league_standings(&state, &league_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `GET /api/manager/leagues/{id}/feed` — recent trades + finalized results.
async fn api_feed(
    State(state): State<AppState>,
    Path(league_id): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let limit = params
        .get("limit")
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(30);
    match crate::manager::standings::league_feed(&state, &league_id, limit).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}

/// `POST /api/manager/leagues/{id}/sweep` — commissioner-triggered run of the
/// season-start + finalization logic (for immediate updates / testing).
async fn api_sweep(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let is_commish = crate::manager::league::is_commissioner(&state, &league_id, &user_id).await;
    if !is_commish {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "forbidden"}))).into_response();
    }
    // Best-effort single-league sweep mirroring the background job.
    let status = {
        let Some(sb) = state.sb.as_ref() else {
            return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response();
        };
        sb.read("manager_leagues", "status", Some(&BTreeMap::from([("id".to_string(), format!("eq.{league_id}"))])), None, None, 1)
            .await
            .unwrap_or_default()
            .first()
            .map(|r| str_value(r.get("status")))
            .unwrap_or_default()
    };
    match status.as_str() {
        crate::manager::STATUS_SETUP | crate::manager::STATUS_DRAFTING => {
            if let Err(e) = crate::manager::standings::league_standings(&state, &league_id).await {
                return e.into_response();
            }
        }
        _ => {}
    }
    crate::routes::auth::json_no_store(json!({"league_id": league_id, "sweep": "ok", "status": status}))
}

/// `POST /api/manager/leagues/{id}/seed-rosters` — commissioner seeds mode-1
/// rosters from the franchise current rosters.
async fn api_seed_rosters(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(league_id): Path<String>,
) -> Response {
    let Some(user_id) = user_id_of(&state.cfg, &headers) else {
        return unauthorized();
    };
    if !csrf_ok(&state, &headers) {
        return bad_csrf();
    }
    let is_commish = crate::manager::league::is_commissioner(&state, &league_id, &user_id).await;
    if !is_commish {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "forbidden"}))).into_response();
    }
    match crate::manager::standings::seed_true_rosters(&state, &league_id).await {
        Ok(v) => crate::routes::auth::json_no_store(v),
        Err(e) => e.into_response(),
    }
}
