//! Full auth (M5): Supabase GoTrue login/signup/logout backed by the signed
//! session cookie (auth_user record + CSRF token + flash messages), with a
//! minimal admin fallback (env `EMAIL`/`PASSWORD`) so the admin pipeline keeps
//! working even when the Supabase auth user is absent.

use std::collections::{BTreeMap, HashMap};

use axum::body::Body;
use axum::extract::{Form, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::config::Config;
use crate::error::ApiError;
use crate::state::AppState;
use crate::web::auth_state;
use crate::web::session;
use crate::web::session::SessionData;
use crate::web::templates;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/login", axum::routing::get(login_page).post(login_post))
        .route("/signup", axum::routing::get(signup_page).post(signup_post))
        .route("/logout", post(logout_page))
}

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::HOST).and_then(|v| v.to_str().ok())
}

pub fn redirect_found(to: &str) -> Response {
    Response::builder()
        .status(StatusCode::FOUND)
        .header(header::LOCATION, to)
        .body(Body::empty())
        .expect("static redirect")
}

pub fn redirect_see_other(to: &str) -> Response {
    Response::builder()
        .status(StatusCode::SEE_OTHER)
        .header(header::LOCATION, to)
        .body(Body::empty())
        .expect("static redirect")
}

fn sanitize_next(next: &str) -> String {
    auth_state::safe_next_url(next).unwrap_or_else(|| "/projections".to_string())
}

/// Decode the session cookie (if any) from the request headers.
pub fn session_from_headers(cfg: &Config, headers: &HeaderMap) -> SessionData {
    let Some(cookie) = headers.get(header::COOKIE).and_then(|v| v.to_str().ok()) else {
        return SessionData::default();
    };
    let Some(value) = session::session_cookie_value(cookie) else {
        return SessionData::default();
    };
    SessionData::from_cookie(&cfg.secret_key, &value).unwrap_or_default()
}

/// Full `auth_user` state (port of `_current_auth_user`) or `None`.
pub fn auth_user_from_headers(cfg: &Config, headers: &HeaderMap) -> Option<Value> {
    let session = session_from_headers(cfg, headers);
    session
        .auth_user
        .map(|r| auth_state::auth_state_from_record(&r, cfg.auth_trial_days))
}

/// True when the request carries a valid admin session.
pub fn is_admin(cfg: &Config, headers: &HeaderMap) -> bool {
    auth_user_from_headers(cfg, headers)
        .and_then(|u| u.get("is_admin").and_then(|v| v.as_bool()))
        .unwrap_or(false)
}

/// Get-or-create the CSRF token for a session. Returns (token, changed).
pub fn csrf_token_for(session: &mut SessionData) -> (String, bool) {
    if !session.csrf_token.is_empty() {
        return (session.csrf_token.clone(), false);
    }
    let token = rand_support::token_urlsafe(32);
    session.csrf_token = token.clone();
    (token, true)
}

/// Port of `_csrf_validate`: compare form/header token against the session.
pub fn csrf_validate(session: &SessionData, provided: Option<&str>) -> bool {
    let expected = session.csrf_token.trim();
    let provided = provided.unwrap_or("").trim();
    if expected.is_empty() || provided.is_empty() {
        return false;
    }
    ct_eq_bytes(expected.as_bytes(), provided.as_bytes())
}

fn ct_eq_bytes(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Add a flash message to the session (port of Flask `flash()`).
pub fn flash(session: &mut SessionData, category: &str, message: &str) {
    session.flashes.push(json!([category, message]));
}

pub fn session_cookie_header_value(cfg: &Config, session: &SessionData) -> String {
    let value = session.encode(&cfg.secret_key);
    format!("session={value}; Path=/; HttpOnly; SameSite=Lax; Max-Age=2592000")
}

pub fn clear_cookie() -> &'static str {
    "session=; Path=/; HttpOnly; SameSite=Lax; Max-Age=0"
}

fn attach_session(mut resp: Response, cfg: &Config, session: &SessionData) -> Response {
    if let Ok(v) = HeaderValue::from_str(&session_cookie_header_value(cfg, session)) {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
    resp
}

/// Attach the session cookie to an existing response (used by Stripe redirects).
pub fn attach_session_header(resp: &mut Response, cfg: &Config, session: &SessionData) {
    if let Ok(v) = HeaderValue::from_str(&session_cookie_header_value(cfg, session)) {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
}

/// Render a template with the session-aware context and persist the session
/// cookie on the response — but only when the session actually carries state
/// (auth_user, CSRF token, or flashes). Anonymous, stateless page views do not
/// emit a `Set-Cookie` (mirrors Flask only writing the session cookie when the
/// session changes, and keeps the cookie jar clean for testing).
pub fn render_with_session(
    state: &AppState,
    host: Option<&str>,
    path: &str,
    template: &str,
    session: &mut SessionData,
    extra: BTreeMap<&'static str, serde_json::Value>,
) -> Result<Response, ApiError> {
    // Create the CSRF token lazily (like Flask's `_csrf_token()` in the
    // context processor) so form POSTs always have a token to validate.
    let _ = csrf_token_for(session);
    let ctx = templates::base_context_with_session(
        &state.cfg,
        host,
        path,
        &state.teams,
        session,
        extra,
    );
    let html = Html(state.templates.render(template, ctx)?).into_response();
    // Flash messages are consumed on display (like Flask's get_flashed_messages
    // removes them from the session), so "Logged in." etc. don't reappear on
    // later page loads — combined with the front-end auto-dismiss after ~5s.
    session.flashes.clear();
    if session.auth_user.is_some() || !session.csrf_token.is_empty() {
        Ok(attach_session(html, &state.cfg, session))
    } else {
        Ok(html)
    }
}

/// JSON helper used by admin endpoints.
pub fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn form_get(form: &HashMap<String, String>, key: &str) -> String {
    form.get(key).cloned().unwrap_or_default()
}

fn is_premium_redirect_target(path: &str) -> bool {
    auth_state::auth_is_premium_path(path)
}

/// `GET /login` — renders `login.html` (redirects when already logged in).
async fn login_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let ua = headers.get(header::USER_AGENT).and_then(|v| v.to_str().ok());
    if auth_state::is_crawler_request("GET", ua) {
        return Ok(minimal_bot_response(404));
    }
    let raw_next = params.get("next").map(|s| s.as_str()).unwrap_or("");
    if !raw_next.is_empty() && auth_state::safe_next_url(raw_next).is_none() {
        return Ok(minimal_bot_response(400));
    }
    let next = sanitize_next(raw_next);
    let mut session = session_from_headers(&state.cfg, &headers);
    if session.auth_user.is_some() {
        return Ok(redirect_found(&next));
    }
    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!(""));
    extra.insert("show_filters", json!(false));
    extra.insert("next_url", json!(next));
    render_with_session(&state, host_str(&headers), "/login", "login.html", &mut session, extra)
}

/// `POST /login` — validates credentials (Supabase GoTrue, admin fallback).
async fn login_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = session_from_headers(&state.cfg, &headers);
    let next_raw = form
        .get("next")
        .cloned()
        .or_else(|| params.get("next").cloned())
        .unwrap_or_else(|| "/projections".to_string());
    let next = sanitize_next(&next_raw);

    let render_error = |state: &AppState,
                        headers: &HeaderMap,
                        session: &mut SessionData,
                        next: &str| {
        let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
        extra.insert("active_tab", json!(""));
        extra.insert("show_filters", json!(false));
        extra.insert("next_url", json!(next));
        render_with_session(state, host_str(headers), "/login", "login.html", session, extra)
    };

    if !crate::supabase::read::auth_is_configured() {
        let mut s2 = session.clone();
        flash(&mut s2, "error", "Auth is not configured in this environment yet.");
        return render_error(&state, &headers, &mut s2, &next);
    }
    let email = form_get(&form, "email").trim().to_lowercase();
    let password = form_get(&form, "password");
    if email.is_empty() || password.is_empty() {
        let mut s2 = session.clone();
        flash(&mut s2, "error", "Enter both email and password.");
        return render_error(&state, &headers, &mut s2, &next);
    }

    // 1) Supabase GoTrue sign-in.
    let mut supabase_ok = false;
    let mut has_access = false;
    if let Some(sb) = state.sb.as_ref() {
        match sb
            .sign_in_with_password(
                state.cfg.supabase_anon_key.as_deref(),
                &email,
                &password,
            )
            .await
        {
            Some(resp) => {
                let user = resp.get("user").cloned().unwrap_or(Value::Null);
                if user.is_object() && !auth_state::as_str_of(user.get("id")).is_empty() {
                    let record = sync_user_account_from_supabase_user(&state, &user, None).await;
                    let full = auth_state::auth_state_from_record(&record, state.cfg.auth_trial_days);
                    has_access = auth_state::as_bool_of(full.get("has_access"));
                    session.auth_user = Some(auth_state::auth_session_payload(&full));
                    supabase_ok = true;
                }
            }
            None => {}
        }
    }

    // 2) Admin fallback (env EMAIL/PASSWORD).
    if !supabase_ok
        && !state.cfg.admin_email.is_empty()
        && email == state.cfg.admin_email.trim().to_lowercase()
        && password == state.cfg.admin_password
    {
        let record = admin_record(&email);
        let full = auth_state::auth_state_from_record(&record, state.cfg.auth_trial_days);
        has_access = auth_state::as_bool_of(full.get("has_access"));
        session.auth_user = Some(auth_state::auth_session_payload(&full));
        supabase_ok = true;
    }

    if !supabase_ok {
        let mut s2 = session.clone();
        flash(&mut s2, "error", "Invalid email or password.");
        return render_error(&state, &headers, &mut s2, &next);
    }

    flash(&mut session, "success", "Logged in.");
    if !has_access && is_premium_redirect_target(&next) {
        return Ok(redirect_with_session("/account", &state, &session));
    }
    Ok(redirect_with_session(&next, &state, &session))
}

pub fn redirect_with_session(to: &str, state: &AppState, session: &SessionData) -> Response {
    attach_session(redirect_found(to), &state.cfg, session)
}

fn admin_record(email: &str) -> Value {
    json!({
        "user_id": "admin",
        "email": email,
        "username": "admin",
        "display_name": "Admin",
        "created_at": "",
        "trial_started_at": "",
        "trial_expires_at": "",
        "subscription_status": "active",
        "subscription_plan": "free",
        "billing_interval": "",
        "is_admin": true,
        "subscription_source": "",
        "stripe_customer_id": "",
        "stripe_subscription_id": "",
        "stripe_price_id": "",
        "stripe_current_period_end": "",
    })
}

/// Port of `_sync_user_account_from_supabase_user` (async, uses Supabase).
pub async fn sync_user_account_from_supabase_user(
    state: &AppState,
    user: &Value,
    overrides: Option<&Value>,
) -> Value {
    let trial_days = state.cfg.auth_trial_days;
    let base = auth_state::auth_record_from_supabase_user(user, None, trial_days);
    let auth_user_id = auth_state::as_str_of(base.get("user_id"));
    let Some(sb) = state.sb.as_ref() else {
        return base;
    };
    if auth_user_id.is_empty() {
        return base;
    }
    let existing = sb.get_user_account(&auth_user_id).await;
    let now = chrono::Utc::now();
    let ov = overrides.cloned().unwrap_or(Value::Null);
    let trial_started = auth_state::parse_iso_datetime(ov.get("trial_started_at"))
        .or_else(|| {
            existing
                .as_ref()
                .and_then(|e| auth_state::parse_iso_datetime(e.get("trial_started_at")))
        })
        .or_else(|| auth_state::parse_iso_datetime(base.get("trial_started_at")))
        .unwrap_or(now);
    let trial_expires = auth_state::parse_iso_datetime(ov.get("trial_expires_at"))
        .or_else(|| {
            existing
                .as_ref()
                .and_then(|e| auth_state::parse_iso_datetime(e.get("trial_expires_at")))
        })
        .or_else(|| auth_state::parse_iso_datetime(base.get("trial_expires_at")))
        .unwrap_or_else(|| trial_started + chrono::Duration::days(i64::from(trial_days)));

    let ex = |k: &str| -> String {
        existing
            .as_ref()
            .map(|e| auth_state::as_str_of(e.get(k)))
            .unwrap_or_default()
    };
    let ov_str = |k: &str| -> String { auth_state::as_str_of(ov.get(k)) };

    let email = {
        let v = ov_str("email");
        let v = if !v.is_empty() { v } else { ex("email") };
        let v = if !v.is_empty() { v } else { auth_state::as_str_of(base.get("email")) };
        v.to_lowercase()
    };
    let username = {
        let v = ov_str("username");
        let v = if !v.is_empty() { v } else { ex("username") };
        if !v.is_empty() {
            v
        } else {
            auth_state::auth_username_candidate(&base, existing.as_ref())
        }
    };
    let display = {
        let v = ov_str("display_name");
        let v = if !v.is_empty() { v } else { ex("display_name") };
        let v = if !v.is_empty() { v } else { auth_state::as_str_of(base.get("display_name")) };
        if v.is_empty() { "Account".to_string() } else { v }
    };
    let is_admin = if ov.get("is_admin").is_some() {
        auth_state::as_bool_of(ov.get("is_admin"))
    } else {
        existing
            .as_ref()
            .map(|e| auth_state::as_bool_of(e.get("is_admin")))
            .unwrap_or(false)
    };
    let status = {
        let v = ov_str("subscription_status");
        let v = if !v.is_empty() { v } else { ex("subscription_status") };
        let v = if !v.is_empty() { v } else { "trialing".to_string() };
        v.to_lowercase()
    };
    let plan = {
        let v = ov_str("subscription_plan");
        let v = if !v.is_empty() { v } else { ex("subscription_plan") };
        let v = if !v.is_empty() { v } else { "trial".to_string() };
        v
    };
    let interval = {
        let v = ov_str("billing_interval");
        let v = if !v.is_empty() { v } else { ex("billing_interval") };
        v.to_lowercase()
    };

    let mut payload = serde_json::Map::new();
    payload.insert("auth_user_id".into(), auth_user_id.clone().into());
    payload.insert("email".into(), email.into());
    payload.insert("username".into(), username.into());
    payload.insert("display_name".into(), display.into());
    payload.insert("is_admin".into(), is_admin.into());
    payload.insert("subscription_status".into(), status.into());
    payload.insert("subscription_plan".into(), plan.into());
    payload.insert("billing_interval".into(), interval.into());
    payload.insert(
        "trial_started_at".into(),
        auth_state::isoformat_utc(Some(trial_started)).into(),
    );
    payload.insert(
        "trial_expires_at".into(),
        auth_state::isoformat_utc(Some(trial_expires)).into(),
    );
    let started = auth_state::parse_iso_datetime(ov.get("subscription_started_at"))
        .or_else(|| existing.as_ref().and_then(|e| auth_state::parse_iso_datetime(e.get("subscription_started_at"))));
    payload.insert(
        "subscription_started_at".into(),
        auth_state::isoformat_utc(started).into(),
    );
    let ends = auth_state::parse_iso_datetime(ov.get("subscription_ends_at"))
        .or_else(|| existing.as_ref().and_then(|e| auth_state::parse_iso_datetime(e.get("subscription_ends_at"))));
    payload.insert(
        "subscription_ends_at".into(),
        auth_state::isoformat_utc(ends).into(),
    );
    payload.insert("updated_at".into(), auth_state::isoformat_utc(Some(now)).into());

    let saved = sb.upsert_user_account(Value::Object(payload)).await;
    let saved = saved.as_ref().or(existing.as_ref());
    auth_state::auth_record_from_supabase_user(user, saved, trial_days)
}

pub fn minimal_bot_response(status_code: u16) -> Response {
    let mut resp = Response::builder()
        .status(StatusCode::from_u16(status_code).unwrap_or(StatusCode::NOT_FOUND))
        .body(Body::empty())
        .expect("bot response");
    resp.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("no-store, max-age=0"),
    );
    resp.headers_mut().insert(
        header::HeaderName::from_static("x-robots-tag"),
        HeaderValue::from_static("noindex, nofollow, noarchive"),
    );
    resp
}

/// `GET /signup` — renders `signup.html`.
async fn signup_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let ua = headers.get(header::USER_AGENT).and_then(|v| v.to_str().ok());
    if auth_state::is_crawler_request("GET", ua) {
        return Ok(minimal_bot_response(404));
    }
    let raw_next = params.get("next").map(|s| s.as_str()).unwrap_or("");
    if !raw_next.is_empty() && auth_state::safe_next_url(raw_next).is_none() {
        return Ok(minimal_bot_response(400));
    }
    let next = sanitize_next(raw_next);
    let mut session = session_from_headers(&state.cfg, &headers);
    if session.auth_user.is_some() {
        return Ok(redirect_found(&next));
    }
    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!(""));
    extra.insert("show_filters", json!(false));
    extra.insert("next_url", json!(next));
    render_with_session(&state, host_str(&headers), "/signup", "signup.html", &mut session, extra)
}

/// `POST /signup` — creates a Supabase user + `user_accounts` row, logs in.
async fn signup_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = session_from_headers(&state.cfg, &headers);
    let next_raw = form
        .get("next")
        .cloned()
        .or_else(|| params.get("next").cloned())
        .unwrap_or_else(|| "/projections".to_string());
    let next = sanitize_next(&next_raw);

    let render_error = |state: &AppState,
                        headers: &HeaderMap,
                        session: &mut SessionData,
                        next: &str| {
        let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
        extra.insert("active_tab", json!(""));
        extra.insert("show_filters", json!(false));
        extra.insert("next_url", json!(next));
        render_with_session(state, host_str(headers), "/signup", "signup.html", session, extra)
    };

    if !crate::supabase::read::auth_is_configured() {
        let mut s2 = session.clone();
        flash(&mut s2, "error", "Auth is not configured in this environment yet.");
        return render_error(&state, &headers, &mut s2, &next);
    }
    let name = form_get(&form, "name").trim().to_string();
    let email = form_get(&form, "email").trim().to_lowercase();
    let password = form_get(&form, "password");
    let confirm_password = form_get(&form, "confirm_password");

    let mut s2 = session.clone();
    if name.is_empty() {
        flash(&mut s2, "error", "Enter your name.");
        return render_error(&state, &headers, &mut s2, &next);
    }
    if !auth_state::valid_email(&email) {
        flash(&mut s2, "error", "Enter a valid email address.");
        return render_error(&state, &headers, &mut s2, &next);
    }
    if password.len() < 8 {
        flash(&mut s2, "error", "Password must be at least 8 characters.");
        return render_error(&state, &headers, &mut s2, &next);
    }
    if password != confirm_password {
        flash(&mut s2, "error", "Passwords do not match.");
        return render_error(&state, &headers, &mut s2, &next);
    }

    let Some(sb) = state.sb.as_ref() else {
        let mut s3 = session.clone();
        flash(&mut s3, "error", "Unable to create your account right now.");
        return render_error(&state, &headers, &mut s3, &next);
    };
    let now = chrono::Utc::now();
    let trial_expires = now + chrono::Duration::days(i64::from(state.cfg.auth_trial_days));
    let metadata = json!({
        "display_name": name,
        "trial_started_at": auth_state::isoformat_utc(Some(now)),
        "trial_expires_at": auth_state::isoformat_utc(Some(trial_expires)),
        "trial_days": state.cfg.auth_trial_days,
        "subscription_status": "trialing",
        "subscription_plan": "trial",
    });
    if sb.admin_create_user(&email, &password, &metadata).await.is_none() {
        let mut s3 = session.clone();
        flash(&mut s3, "error", "Unable to create your account right now.");
        return render_error(&state, &headers, &mut s3, &next);
    }
    // Automatic login.
    let sign_in = sb
        .sign_in_with_password(state.cfg.supabase_anon_key.as_deref(), &email, &password)
        .await;
    let user = sign_in.and_then(|r| r.get("user").cloned());
    let Some(user) = user.filter(|u| u.is_object()) else {
        let mut s3 = session.clone();
        flash(&mut s3, "error", "Account created, but automatic login failed. Please log in.");
        return render_error(&state, &headers, &mut s3, &next);
    };
    let overrides = json!({
        "display_name": name,
        "subscription_status": "trialing",
        "subscription_plan": "trial",
        "trial_started_at": auth_state::isoformat_utc(Some(now)),
        "trial_expires_at": auth_state::isoformat_utc(Some(trial_expires)),
    });
    let record = sync_user_account_from_supabase_user(&state, &user, Some(&overrides)).await;
    let full = auth_state::auth_state_from_record(&record, state.cfg.auth_trial_days);
    session.auth_user = Some(auth_state::auth_session_payload(&full));
    flash(
        &mut session,
        "success",
        &format!("Account created. Your {}-day trial starts now.", state.cfg.auth_trial_days),
    );
    Ok(redirect_with_session(&next, &state, &session))
}

/// `POST /logout` — clears the session cookie (Flask redirects to /login).
async fn logout_page() -> Response {
    let mut resp = redirect_found("/login");
    if let Ok(v) = HeaderValue::from_str(clear_cookie()) {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
    resp
}

/// `secrets.token_urlsafe` equivalent.
mod rand_support {
    use base64::engine::general_purpose::URL_SAFE_NO_PAD;
    use base64::Engine;

    pub fn token_urlsafe(nbytes: usize) -> String {
        let mut bytes = vec![0u8; nbytes];
        fill_random(&mut bytes);
        URL_SAFE_NO_PAD.encode(&bytes)
    }

    fn fill_random(bytes: &mut [u8]) {
        use getrandom::getrandom;
        let _ = getrandom(bytes);
    }
}
