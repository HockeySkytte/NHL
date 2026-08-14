pub mod account;
pub mod admin;
pub mod analytics;
pub mod auth;
pub mod community;
pub mod line_tool;
pub mod misc_api;
pub mod pages;
pub mod pbp;
pub mod projections_api;
pub mod rapm_api;
pub mod shooting;
pub mod stripe;

use axum::extract::{Request, State};
use axum::http::StatusCode;
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::Router;
use tower_http::compression::predicate::{DefaultPredicate, SizeAbove};
use tower_http::compression::{CompressionLayer, CompressionLevel, Predicate};
use tower_http::services::ServeDir;
use tower_http::trace::TraceLayer;

use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    let static_dir = state.cfg.static_dir.clone();
    let http_compress = state.cfg.http_compress;
    let compress_min_size = state.cfg.compress_min_size;
    let compress_level = state.cfg.compress_level;

    let router = Router::new()
        .merge(pages::router())
        .merge(auth::router())
        .merge(account::router())
        .merge(admin::router())
        .merge(stripe::router())
        .merge(community::router())
        .merge(misc_api::router())
        .merge(analytics::router())
        .merge(rapm_api::router())
        .merge(line_tool::router())
        .merge(shooting::router())
        .merge(pbp::router())
        .merge(projections_api::router())
        .nest_service("/static", ServeDir::new(static_dir))
        .fallback(fallback_404);

    // Premium gating must run while the router is still `Router<AppState>`
    // (the middleware extracts `State<AppState>`).
    let router = router.layer(middleware::from_fn_with_state(
        state.clone(),
        enforce_premium_gating,
    ));
    let router = router.layer(middleware::from_fn(log_request_duration));
    let router = router.with_state(state);
    let router = router.layer(TraceLayer::new_for_http());

    // Flask-Compress equivalent: gzip/deflate/br above COMPRESS_MIN_SIZE,
    // toggled by HTTP_COMPRESS and tuned by COMPRESS_LEVEL.
    if http_compress {
        let min_size_u16 = compress_min_size.min(u64::from(u16::MAX)) as u16;
        let predicate = DefaultPredicate::new().and(SizeAbove::new(min_size_u16));
        let compression = CompressionLayer::new()
            .br(true)
            .gzip(true)
            .deflate(true)
            .quality(CompressionLevel::Precise(compress_level))
            .compress_when(predicate);
        router.layer(compression)
    } else {
        router
    }
}

/// Port of `enforce_auth_for_premium_routes` + `_deny_premium_access`:
/// `/projections` and `/api/projections/*` (minus the GM public allowlist)
/// require a logged-in user with `has_access`.
async fn enforce_premium_gating(
    State(state): State<AppState>,
    req: Request,
    next: Next,
) -> Response {
    use crate::web::auth_state;

    if !crate::supabase::read::auth_is_configured() {
        return next.run(req).await;
    }
    let path = req.uri().path().to_string();
    if !auth_state::auth_is_premium_path(&path) {
        return next.run(req).await;
    }
    let auth_user = crate::routes::auth::auth_user_from_headers(&state.cfg, req.headers());
    let has_access = auth_user
        .as_ref()
        .map(|u| auth_state::as_bool_of(u.get("has_access")))
        .unwrap_or(false);
    if auth_user.is_some() && has_access {
        return next.run(req).await;
    }

    let method = req.method().as_str();
    let ua = req
        .headers()
        .get(axum::http::header::USER_AGENT)
        .and_then(|v| v.to_str().ok());
    let is_api = path.starts_with("/api/");
    let next_url = auth_state::safe_next_url(&path).unwrap_or_else(|| "/projections".to_string());

    if auth_user.is_none() {
        if auth_state::is_crawler_request(method, ua) {
            return crate::routes::auth::minimal_bot_response(if is_api { 401 } else { 404 });
        }
        if is_api {
            return (
                StatusCode::UNAUTHORIZED,
                axum::Json(serde_json::json!({
                    "error": "auth_required",
                    "loginUrl": format!("/login?next={next_url}"),
                })),
            )
                .into_response();
        }
        return crate::routes::auth::redirect_found(&format!("/login?next={next_url}"));
    }
    if is_api {
        return (
            StatusCode::FORBIDDEN,
            axum::Json(serde_json::json!({
                "error": "trial_expired",
                "accountUrl": "/account",
            })),
        )
            .into_response();
    }
    crate::routes::auth::redirect_found("/account")
}

async fn fallback_404() -> (StatusCode, &'static str) {
    (StatusCode::NOT_FOUND, "Not Found")
}

/// Logs requests that exceed `SLOW_REQUEST_MS` (default 2000 ms) at WARN with
/// method/path/status/duration — a lightweight latency monitor for the load
/// test and general perf observability.
async fn log_request_duration(req: Request, next: Next) -> Response {
    let slow_ms: u128 = std::env::var("SLOW_REQUEST_MS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(2000);
    let method = req.method().clone();
    let uri = req.uri().clone();
    let start = std::time::Instant::now();
    let resp = next.run(req).await;
    let elapsed = start.elapsed();
    let status = resp.status();
    let ms = elapsed.as_millis();
    if ms > slow_ms {
        tracing::warn!(
            "slow request: {method} {uri} -> {status} in {ms}ms"
        );
    }
    resp
}
