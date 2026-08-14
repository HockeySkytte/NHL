//! Page routes available in M0: `/`, `/about(+/<slug>)`, `/robots.txt`,
//! `/sitemap.xml`, `/favicon.png`. The remaining 13 pages arrive in M1.

use std::collections::BTreeMap;

use axum::body::Body;
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::Router;
use serde_json::{json, Value};

use crate::data::about;
use crate::error::ApiError;
use crate::state::AppState;
use crate::web::templates;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/", get(index_page))
        .route("/schedule", get(schedule_page))
        .route("/live", get(live_games_page))
        .route("/standings", get(standings_page))
        .route("/projections", get(game_projections_page))
        .route("/community", get(community_page))
        .route("/donation", get(donation_page))
        .route("/skaters", get(skaters_page))
        .route("/goalies", get(goalies_page))
        .route("/line-tool", get(line_tool_page))
        .route("/gm-mode", get(gm_mode_page))
        .route("/teams", get(teams_page))
        .route("/card-builder", get(card_builder_page))
        .route("/about", get(about_page))
        .route("/about/{slug}", get(about_page_slug))
        .route("/odds/{game_id}", get(odds_page))
        .route("/game/{game_id}", get(game_page))
        .route("/robots.txt", get(robots_txt))
        .route("/sitemap.xml", get(sitemap_xml))
        .route("/favicon.png", get(favicon_png))
}

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::HOST).and_then(|v| v.to_str().ok())
}

/// Flask `redirect(...)` is a 302 Found; axum's `Redirect` doesn't expose 302.
fn redirect_found(to: &str) -> Response {
    Response::builder()
        .status(StatusCode::FOUND)
        .header(header::LOCATION, to)
        .body(Body::empty())
        .expect("static redirect response")
}

/// `GET /` — landing page (`index_page`).
async fn index_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, ApiError> {
    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Home"));
    extra.insert("meta_title", json!("NHL Analytics · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Advanced NHL hockey analytics powered by Hockey-Statistics. Explore xG stats, skater and goalie performance, live scores, standings, and game projections."),
    );
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/",
        "home.html",
        &mut session,
        extra,
    )
}

/// `GET /about` — redirect to the first headline section (`about_page`).
async fn about_page(State(state): State<AppState>) -> Response {
    match state.about.first_slug() {
        Some(slug) => redirect_found(&format!("/about/{slug}")),
        None => redirect_found("/"),
    }
}

/// `GET /about/<section_slug>` — one headline section per route (`about_page_slug`).
async fn about_page_slug(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(raw_slug): Path<String>,
) -> Result<Response, ApiError> {
    let slug = raw_slug.trim().to_lowercase();
    if slug.is_empty() || !state.about.valid_slug(&slug) {
        return Ok(redirect_found("/about"));
    }

    let title = state.about.section_title(&slug).unwrap_or("About");
    let raw_text = state.about.section_text(&slug);
    let text = about::strip_leading_heading(raw_text, title);
    let segments = about::text_segments(&text, &state.cfg.static_dir);
    let is_glossary = state.about.is_glossary_slug(&slug);

    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!("About"));
    extra.insert("show_filters", json!(true));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(false));
    extra.insert(
        "glossary_sections",
        Value::Array(state.about.glossary_sections.clone()),
    );
    extra.insert(
        "about_headlines",
        Value::Array(state.about.nav_items(&slug)),
    );
    extra.insert("about_section_slug", json!(slug));
    extra.insert("about_section_title", json!(title));
    extra.insert("about_segments", Value::Array(segments));
    extra.insert("about_is_glossary", json!(is_glossary));
    extra.insert(
        "meta_title",
        json!(format!("{title} · About · Hockey-Statistics")),
    );
    extra.insert(
        "meta_description",
        json!("Learn about the metrics and models behind Hockey-Statistics NHL analytics — xG, RAPM, GSAx, Corsi, Fenwick, and more."),
    );

    let path = format!("/about/{slug}");
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        &path,
        "about.html",
        &mut session,
        extra,
    )
}

/// `GET /robots.txt` — byte-for-byte port of `robots_txt`.
async fn robots_txt(State(state): State<AppState>, headers: HeaderMap) -> Response {
    let base = templates::url_root(&state.cfg, host_str(&headers))
        .trim_end_matches('/')
        .to_string();
    let content = format!(
        "User-agent: *\nAllow: /\nDisallow: /admin/\nDisallow: /api/\nDisallow: /account\nDisallow: /login\nDisallow: /projections\nDisallow: /signup\nDisallow: /logout\nDisallow: /stripe/\n\nSitemap: {base}/sitemap.xml\n"
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "text/plain; charset=utf-8")
        .body(Body::from(content))
        .expect("static robots response")
}

/// `GET /sitemap.xml` — port of `sitemap_xml` (same static page list).
async fn sitemap_xml(State(state): State<AppState>, headers: HeaderMap) -> Response {
    const STATIC_PAGES: [(&str, &str, &str); 10] = [
        ("/", "weekly", "1.0"),
        ("/schedule", "daily", "0.9"),
        ("/live", "always", "0.9"),
        ("/standings", "daily", "0.8"),
        ("/projections", "daily", "0.9"),
        ("/skaters", "weekly", "0.8"),
        ("/goalies", "weekly", "0.8"),
        ("/line-tool", "weekly", "0.8"),
        ("/teams", "weekly", "0.8"),
        ("/about", "monthly", "0.5"),
    ];
    let base = templates::url_root(&state.cfg, host_str(&headers))
        .trim_end_matches('/')
        .to_string();
    let mut urls = String::new();
    for (path, changefreq, priority) in STATIC_PAGES {
        urls.push_str(&format!(
            "<url><loc>{base}{path}</loc><changefreq>{changefreq}</changefreq><priority>{priority}</priority></url>"
        ));
    }
    let xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">{urls}</urlset>"
    );
    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/xml")
        .body(Body::from(xml))
        .expect("static sitemap response")
}

/// `GET /favicon.png` — serves `static/Logo.png` (Flask `favicon_png`).
async fn favicon_png(State(state): State<AppState>) -> Result<Response, ApiError> {
    let path = state.cfg.static_dir.join("Logo.png");
    let bytes = tokio::fs::read(&path).await.map_err(|e| {
        ApiError::NotFound(format!("favicon missing at {}: {e}", path.display()))
    })?;
    Ok(Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "image/png")
        .body(Body::from(bytes))
        .expect("static favicon response"))
}

async fn render_page(
    state: &AppState,
    headers: &HeaderMap,
    template: &str,
    path: &str,
    extra: BTreeMap<&'static str, serde_json::Value>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, headers);
    crate::routes::auth::render_with_session(state, host_str(headers), path, template, &mut session, extra)
}

/// `GET /schedule` (`schedule_page`).
async fn schedule_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Schedule"));
    extra.insert("show_season_state", json!(true));
    extra.insert("meta_title", json!("NHL Schedule & Game Results · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Browse the full NHL schedule with game results, scores, and play-by-play stats for every game."),
    );
    render_page(&state, &headers, "index.html", "/schedule", extra).await
}

/// `GET /live` (`live_games_page`).
async fn live_games_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Live Games"));
    extra.insert("show_season_state", json!(false));
    extra.insert("meta_title", json!("NHL Live Games · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Live NHL game scores, real-time expected goals (xG), shot counts, and in-game analytics — updated continuously."),
    );
    render_page(&state, &headers, "live.html", "/live", extra).await
}

/// `GET /standings` (`standings_page`).
async fn standings_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    // Seasons from Last_date.csv, newest first (matches Flask `standings_page`).
    let season_objs: Vec<Value> = state
        .last_dates
        .keys()
        .rev()
        .map(|s| json!({ "season": s }))
        .collect();
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Standings"));
    extra.insert("show_season_state", json!(false));
    extra.insert("seasons", Value::Array(season_objs));
    extra.insert("meta_title", json!("NHL Standings · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("NHL standings with points, win percentage, streaks, and goal differential for every team and season."),
    );
    render_page(&state, &headers, "standings.html", "/standings", extra).await
}

/// `GET /projections` (`game_projections_page`).
async fn game_projections_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Game Projections"));
    extra.insert("show_season_state", json!(false));
    extra.insert("meta_title", json!("NHL Game Projections · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Model-based NHL game projections with win probabilities, projected goals, and matchup analytics for today's games."),
    );
    render_page(&state, &headers, "projections.html", "/projections", extra).await
}

/// `GET /community` (`community_page`).
async fn community_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Community"));
    extra.insert("show_filters", json!(true));
    extra.insert("meta_title", json!("NHL Community · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Read the latest and top hockey community posts from Hockey Skytte Community."),
    );
    render_page(&state, &headers, "community.html", "/community", extra).await
}

/// `GET /donation` (`donation_page`).
async fn donation_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Donation"));
    extra.insert("show_filters", json!(false));
    render_page(&state, &headers, "donation.html", "/donation", extra).await
}

/// `GET /skaters` (`skaters_page`).
async fn skaters_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Skaters"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(false));
    extra.insert("meta_title", json!("NHL Skater Stats & xG · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("In-depth NHL skater statistics including goals, assists, xG, Corsi, Fenwick, and zone entry data. Filter by team, season, and strength state."),
    );
    render_page(&state, &headers, "skaters.html", "/skaters", extra).await
}

/// `GET /goalies` (`goalies_page`).
async fn goalies_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Goalies"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(false));
    extra.insert("meta_title", json!("NHL Goalie Stats & GSAx · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Advanced NHL goalie statistics: GSAx, xGA, save percentages by shot type, and zone-level shot-against heat maps. Compare goalies across seasons."),
    );
    render_page(&state, &headers, "goalies.html", "/goalies", extra).await
}

/// `GET /line-tool` (`line_tool_page`).
async fn line_tool_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Line Tool"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(false));
    extra.insert("meta_title", json!("NHL Line Combinations & On-Ice Stats · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("Explore NHL forward line combinations and defense pairings with on-ice xG, Corsi, and zone heat maps. Supports WOWY analysis and multi-season views."),
    );
    render_page(&state, &headers, "line_tool.html", "/line-tool", extra).await
}

/// `GET /gm-mode` (`gm_mode_page`).
async fn gm_mode_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("GM Mode"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(false));
    extra.insert("show_season_slicer", json!(false));
    extra.insert("meta_title", json!("GM Mode · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("GM Mode for viewing Daily Faceoff lines, defensive pairings, and goalie depth chart by team."),
    );
    render_page(&state, &headers, "gm_mode.html", "/gm-mode", extra).await
}

/// `GET /teams` (`teams_page`).
async fn teams_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Teams"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(true));
    extra.insert("meta_title", json!("NHL Team Stats · Hockey-Statistics"));
    extra.insert(
        "meta_description",
        json!("NHL team analytics: xGF%, Corsi, Fenwick, zone starts, on-ice shooting percentage, and goalie performance. Compare all 32 teams across seasons."),
    );
    render_page(&state, &headers, "teams.html", "/teams", extra).await
}

/// `GET /card-builder` (`card_builder_page`).
async fn card_builder_page(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Card Builder"));
    extra.insert("show_season_state", json!(false));
    extra.insert("show_include_historic", json!(true));
    render_page(&state, &headers, "card_builder.html", "/card-builder", extra).await
}

/// `GET /odds/<game_id>` (`odds_page`).
async fn odds_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(game_id): Path<i64>,
) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Game Projections"));
    extra.insert("show_season_state", json!(false));
    extra.insert("game_id", json!(game_id));
    render_page(&state, &headers, "odds.html", &format!("/odds/{game_id}"), extra).await
}

/// `GET /game/<game_id>` (`game_page`).
async fn game_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(game_id): Path<i64>,
) -> Result<Response, ApiError> {
    let mut extra = BTreeMap::new();
    extra.insert("active_tab", json!("Schedule"));
    extra.insert("show_season_state", json!(false));
    extra.insert("game_id", json!(game_id));
    render_page(&state, &headers, "game.html", &format!("/game/{game_id}"), extra).await
}
