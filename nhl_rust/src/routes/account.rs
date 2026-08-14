//! Account page + account actions (M5): `/account`, `/account/*` POSTs,
//! `/donate` POST, and the auth-gated card-builder layouts API.

use std::collections::{BTreeMap, HashMap};

use axum::body::Body;
use axum::extract::{Form, Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::error::ApiError;
use crate::state::AppState;
use crate::web::auth_state;
use crate::web::session::SessionData;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/account", get(account_page))
        .route("/account/plan", post(account_plan_update))
        .route("/account/billing", post(account_billing_portal))
        .route("/account/donate", post(account_donate))
        .route("/account/unsubscribe", post(account_unsubscribe))
        .route("/account/profile", post(account_profile_update))
        .route("/account/password", post(account_password_update))
        .route("/account/delete", post(account_delete))
        .route("/donate", post(public_donate))
        .route("/api/card-builder/layouts", get(api_card_builder_layouts_get).post(api_card_builder_layouts_post))
        .route("/api/card-builder/layouts/{layout_id}", delete(api_card_builder_layouts_delete))
}

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::HOST).and_then(|v| v.to_str().ok())
}

fn form_get(form: &HashMap<String, String>, key: &str) -> String {
    form.get(key).cloned().unwrap_or_default()
}

/// Guard: requires a logged-in user. Returns the auth record or a redirect.
fn account_guard(session: &SessionData, path: &str) -> Result<Value, Response> {
    if let Some(raw) = session.auth_user.clone() {
        return Ok(raw);
    }
    Err(crate::routes::auth::redirect_found(&format!("/login?next={path}")))
}

/// Guard: requires an admin (page redirect semantics).
fn admin_page_guard(state: &AppState, session: &SessionData, path: &str) -> Result<Value, Response> {
    let raw = account_guard(session, path)?;
    let full = auth_state::auth_state_from_record(&raw, state.cfg.auth_trial_days);
    if auth_state::as_bool_of(full.get("is_admin")) {
        return Ok(raw);
    }
    Err(crate::routes::auth::redirect_found("/account"))
}

/// Port of `_refresh_current_auth_user`: re-fetch user_accounts + merge + update
/// the session. Returns the full state.
pub async fn refresh_current_auth_user(
    state: &AppState,
    session: &mut SessionData,
) -> Option<Value> {
    let raw = session.auth_user.clone()?;
    let account = if let Some(sb) = state.sb.as_ref() {
        let uid = auth_state::as_str_of(raw.get("user_id"));
        if !uid.is_empty() {
            sb.get_user_account(&uid).await
        } else {
            None
        }
    } else {
        None
    };
    let merged = auth_state::merge_auth_user_account(&raw, account.as_ref());
    let full = auth_state::auth_state_from_record(&merged, state.cfg.auth_trial_days);
    session.auth_user = Some(auth_state::auth_session_payload(&full));
    Some(full)
}

/// Port of `_persist_auth_user_updates`: upsert user_accounts, optionally
/// update the Supabase auth user metadata/password, and refresh the session.
pub async fn persist_auth_user_updates(
    state: &AppState,
    session: &mut SessionData,
    auth_user: &Value,
    updates: &Value,
    auth_metadata: Option<&Value>,
    auth_password: Option<&str>,
) -> Value {
    let payload = auth_state::build_account_payload(auth_user, Some(updates));
    let saved = if let Some(sb) = state.sb.as_ref() {
        sb.upsert_user_account(payload.clone()).await
    } else {
        None
    };
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    if !user_id.is_empty() {
        if let Some(sb) = state.sb.as_ref() {
            let mut attrs = serde_json::Map::new();
            if let Some(meta) = auth_metadata {
                attrs.insert("user_metadata".to_string(), meta.clone());
            }
            if let Some(pw) = auth_password {
                attrs.insert("password".to_string(), Value::String(pw.to_string()));
            }
            if !attrs.is_empty() {
                let _ = sb.admin_update_user(&user_id, &Value::Object(attrs)).await;
            }
        }
    }
    let record = auth_state::merge_auth_user_account(
        auth_user,
        saved.as_ref().or(Some(&payload)),
    );
    let full = auth_state::auth_state_from_record(&record, state.cfg.auth_trial_days);
    session.auth_user = Some(auth_state::auth_session_payload(&full));
    full
}

/// Port of `_stripe_billing_state` — config-only (no Stripe API calls).
pub fn billing_state(cfg: &crate::config::Config, auth_user: &Value) -> Value {
    let has_customer = !auth_state::as_str_of(auth_user.get("stripe_customer_id")).is_empty();
    let missing = stripe_missing_config(cfg, None);
    let checkout_enabled = missing.is_empty();
    let portal_enabled = !cfg.stripe_secret_key.as_deref().unwrap_or("").is_empty() && has_customer;
    let stripe_source = auth_state::as_str_of(auth_user.get("subscription_source"));
    let has_stripe_sub = !auth_state::as_str_of(auth_user.get("stripe_subscription_id")).is_empty();
    let managed_by_stripe = stripe_source == "stripe" || has_stripe_sub;
    let any_configured = cfg.stripe_secret_key.is_some()
        || cfg.stripe_webhook_secret.is_some()
        || cfg.stripe_price_monthly_id.is_some()
        || cfg.stripe_price_yearly_id.is_some();
    json!({
        "checkout_enabled": checkout_enabled,
        "portal_enabled": portal_enabled,
        "has_customer": has_customer,
        "managed_by_stripe": managed_by_stripe,
        "partial_config": any_configured && !checkout_enabled,
        "missing_config": missing,
    })
}

pub fn stripe_missing_config(cfg: &crate::config::Config, plan_key: Option<&str>) -> Vec<String> {
    let mut missing: Vec<String> = Vec::new();
    if cfg.stripe_secret_key.as_deref().unwrap_or("").is_empty() {
        missing.push("STRIPE_SECRET_KEY".to_string());
    }
    let plans: Vec<&str> = match plan_key {
        Some(p) => vec![p],
        None => vec!["monthly", "yearly"],
    };
    for key in plans {
        let env_name = match key {
            "monthly" => "STRIPE_PRICE_MONTHLY_ID",
            "yearly" => "STRIPE_PRICE_YEARLY_ID",
            _ => "",
        };
        if !env_name.is_empty() {
            let v = match key {
                "monthly" => &cfg.stripe_price_monthly_id,
                _ => &cfg.stripe_price_yearly_id,
            };
            if v.as_deref().unwrap_or("").is_empty() {
                missing.push(env_name.to_string());
            }
        }
    }
    let mut seen = std::collections::HashSet::new();
    missing.retain(|m| seen.insert(m.clone()));
    missing
}

/// `GET /account` — renders `account.html`.
async fn account_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let ua = headers.get(header::USER_AGENT).and_then(|v| v.to_str().ok());
    let raw = match session.auth_user.clone() {
        Some(raw) => raw,
        None => {
            if auth_state::is_crawler_request("GET", ua) {
                return Ok(crate::routes::auth::minimal_bot_response(404));
            }
            return Ok(crate::routes::auth::redirect_found("/login?next=/account"));
        }
    };
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(raw);

    let billing_status = params.get("billing").cloned().unwrap_or_default().trim().to_lowercase();
    let billing_banner: Option<Value> = match billing_status.as_str() {
        "success" => Some(json!({
            "category": if auth_state::as_bool_of(full.get("has_subscription")) { "success" } else { "info" },
            "title": "Stripe checkout complete",
            "detail": "Your billing update has been sent to the app. If the plan label has not updated yet, refresh again in a few seconds while the webhook finishes syncing.",
        })),
        "canceled" => Some(json!({
            "category": "info",
            "title": "Stripe checkout canceled",
            "detail": "No billing changes were applied.",
        })),
        "donation_success" => Some(json!({
            "category": "success",
            "title": "Thank you for your donation",
            "detail": "Your support helps keep the app running and improving.",
        })),
        "donation_canceled" => Some(json!({
            "category": "info",
            "title": "Donation checkout canceled",
            "detail": "No donation was processed.",
        })),
        _ => None,
    };

    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!("Account"));
    extra.insert("show_filters", json!(false));
    extra.insert("plan_options", Value::Array(crate::web::templates::auth_plan_options()));
    extra.insert("auth_user", full.clone());
    extra.insert("billing", billing_state(&state.cfg, &full));
    if let Some(banner) = billing_banner {
        extra.insert("billing_banner", banner);
    }
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/account",
        "account.html",
        &mut session,
        extra,
    )
}

fn redirect_to_account(state: &AppState, session: &SessionData, flash_msg: (&str, &str)) -> Response {
    let mut s = session.clone();
    crate::routes::auth::flash(&mut s, flash_msg.0, flash_msg.1);
    crate::routes::auth::redirect_with_session("/account", state, &s)
}

/// `POST /account/plan` — switch plan (Stripe checkout or direct update).
async fn account_plan_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user.clone());
    let plan_key = form_get(&form, "plan").trim().to_lowercase();
    if plan_key != "monthly" && plan_key != "yearly" {
        return Ok(redirect_to_account(&state, &session, ("error", "Choose a valid plan.")));
    }
    let is_free_active = auth_state::as_str_of(full.get("subscription_plan")) == "free"
        && auth_state::as_str_lower(full.get("subscription_status")) == "active";
    if is_free_active {
        return Ok(redirect_to_account(&state, &session, ("info", "This account already has free access. No Stripe checkout is needed.")));
    }
    // If Stripe is configured, go through checkout (M5c); otherwise persist directly.
    let any_stripe = state.cfg.stripe_secret_key.is_some()
        || state.cfg.stripe_price_monthly_id.is_some()
        || state.cfg.stripe_price_yearly_id.is_some();
    if any_stripe {
        return Ok(crate::routes::stripe::create_checkout_redirect(&state, &mut session, &full, &plan_key).await);
    }
    let updates = auth_state::subscription_update_for_plan(&plan_key, &full);
    persist_auth_user_updates(&state, &mut session, &full, &updates, None, None).await;
    let label = if plan_key == "monthly" { "Pro Monthly" } else { "Pro Yearly" };
    Ok(redirect_to_account(&state, &session, ("success", &format!("Plan updated to {label}."))))
}

/// `POST /account/billing` — open the Stripe billing portal.
async fn account_billing_portal(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user);
    Ok(crate::routes::stripe::create_billing_portal_redirect(&state, &mut session, &full).await)
}

/// `POST /account/donate` — logged-in donation checkout.
async fn account_donate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user);
    let amount = form_get(&form, "donation_amount");
    Ok(crate::routes::stripe::create_donation_checkout_redirect(
        &state,
        &mut session,
        Some(&full),
        &amount,
        "",
    ).await)
}

/// `POST /donate` — public (guest) donation checkout.
async fn public_donate(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let guest_email = form_get(&form, "guest_email").trim().to_lowercase();
    let amount = form_get(&form, "donation_amount");
    let auth_user = session.auth_user.clone().map(|r| {
        auth_state::auth_state_from_record(&r, state.cfg.auth_trial_days)
    });
    Ok(crate::routes::stripe::create_donation_checkout_redirect(
        &state,
        &mut session,
        auth_user.as_ref(),
        &amount,
        &guest_email,
    ).await)
}

/// `POST /account/unsubscribe` — cancel the paid plan.
async fn account_unsubscribe(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user);
    let has_customer = !auth_state::as_str_of(full.get("stripe_customer_id")).is_empty();
    if state.cfg.stripe_secret_key.is_some() && has_customer {
        return Ok(crate::routes::stripe::create_billing_portal_redirect(&state, &mut session, &full).await);
    }
    if form_get(&form, "confirm_unsubscribe") != "1" {
        return Ok(redirect_to_account(&state, &session, ("error", "Confirm the unsubscribe action to continue.")));
    }
    let updates = auth_state::subscription_update_for_plan("unsubscribe", &full);
    persist_auth_user_updates(&state, &mut session, &full, &updates, None, None).await;
    Ok(redirect_to_account(&state, &session, ("success", "Subscription canceled. Projections will stay locked until you reactivate a plan.")))
}

/// `POST /account/profile` — update the username.
async fn account_profile_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user.clone());
    let username = auth_state::normalize_username(&form_get(&form, "username"));
    if !auth_state::valid_username(&username) {
        return Ok(redirect_to_account(&state, &session, ("error", "Username must be 3-32 characters and use letters, numbers, dots, dashes, or underscores.")));
    }
    let user_id = auth_state::as_str_of(full.get("user_id"));
    if let Some(sb) = state.sb.as_ref() {
        if let Some(rows) = sb.list_user_accounts().await {
            let current_username = auth_state::as_str_of(full.get("username"));
            if current_username == username {
                return Ok(redirect_to_account(&state, &session, ("info", "Username unchanged.")));
            }
            let taken = rows.iter().any(|r| {
                let uid = auth_state::as_str_of(r.get("auth_user_id"));
                if !user_id.is_empty() && uid == user_id {
                    return false;
                }
                auth_state::normalize_username(&auth_state::as_str_of(r.get("username"))) == username
            });
            if taken {
                return Ok(redirect_to_account(&state, &session, ("error", "That username is already in use. Choose another one.")));
            }
        }
    }
    let metadata = json!({
        "display_name": if !auth_state::as_str_of(full.get("display_name")).is_empty() { auth_state::as_str_of(full.get("display_name")) } else { username.clone() },
        "username": username.clone(),
        "is_admin": auth_state::as_bool_of(full.get("is_admin")),
    });
    persist_auth_user_updates(&state, &mut session, &full, &json!({"username": username}), Some(&metadata), None).await;
    Ok(redirect_to_account(&state, &session, ("success", "Username updated.")))
}

/// `POST /account/password` — change password via Supabase admin.
async fn account_password_update(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let full = refresh_current_auth_user(&state, &mut session).await.unwrap_or(auth_user);
    let password = form_get(&form, "password");
    let confirm = form_get(&form, "confirm_password");
    if password.len() < 8 {
        return Ok(redirect_to_account(&state, &session, ("error", "Password must be at least 8 characters.")));
    }
    if password != confirm {
        return Ok(redirect_to_account(&state, &session, ("error", "Passwords do not match.")));
    }
    persist_auth_user_updates(&state, &mut session, &full, &Value::Null, None, Some(&password)).await;
    Ok(redirect_to_account(&state, &session, ("success", "Password updated. Use the new password the next time you log in.")))
}

/// `POST /account/delete` — delete the account.
async fn account_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let auth_user = match account_guard(&session, "/account") {
        Ok(u) => u,
        Err(r) => return Ok(r),
    };
    if !csrf_valid(&session, &form) {
        return Ok(plain_text_400("Invalid CSRF token"));
    }
    let confirmation = form_get(&form, "confirmation").trim().to_uppercase();
    if confirmation != "DELETE" {
        return Ok(redirect_to_account(&state, &session, ("error", "Type DELETE to confirm account deletion.")));
    }
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    if !user_id.is_empty() {
        if let Some(sb) = state.sb.as_ref() {
            let _ = sb.admin_delete_user(&user_id).await;
        }
    }
    // Clear session and redirect to login.
    let mut resp = crate::routes::auth::redirect_found("/login");
    if let Ok(v) = axum::http::HeaderValue::from_str(crate::routes::auth::clear_cookie()) {
        resp.headers_mut().insert(header::SET_COOKIE, v);
    }
    Ok(resp)
}

fn csrf_valid(session: &SessionData, form: &HashMap<String, String>) -> bool {
    let provided = form.get("csrf_token").map(|s| s.as_str());
    crate::routes::auth::csrf_validate(session, provided)
}

fn plain_text_400(msg: &str) -> Response {
    (StatusCode::BAD_REQUEST, msg.to_string()).into_response()
}

// ── Card builder layouts API ──────────────────────────────────────

fn normalize_layout_id(value: &str) -> Result<String, String> {
    let raw = value.trim();
    if raw.is_empty() {
        return Err("invalid_layout_id".to_string());
    }
    let parsed = uuid_lite::parse(raw);
    parsed.ok_or_else(|| "invalid_layout_id".to_string())
}

fn new_uuid() -> String {
    uuid_lite::v4()
}

fn normalize_layout_name(value: &str, card_type: &str) -> String {
    let raw: String = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if raw.is_empty() {
        format!("{} card", card_type)
    } else {
        raw.chars().take(80).collect()
    }
}

fn normalize_card_type(value: &str) -> String {
    let raw = value.trim().to_lowercase();
    match raw.as_str() {
        "skater" | "goalie" | "team" | "gm_mode" => raw,
        _ => "skater".to_string(),
    }
}

/// `GET /api/card-builder/layouts`
async fn api_card_builder_layouts_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    let session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let Some(auth_user) = session.auth_user else {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "auth_required", "loginUrl": "/login"}))).into_response();
    };
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    let Some(sb) = state.sb.as_ref() else {
        return crate::routes::auth::json_no_store(json!({"layouts": [], "storageAvailable": false}));
    };
    if user_id.is_empty() {
        return crate::routes::auth::json_no_store(json!({"layouts": [], "storageAvailable": false}));
    }
    let layouts = sb.list_card_builder_layouts(&user_id).await.unwrap_or_default();
    let out: Vec<Value> = layouts.iter().map(|r| card_builder_layout_response(r)).collect();
    crate::routes::auth::json_no_store(json!({"layouts": out, "storageAvailable": true}))
}

/// `POST /api/card-builder/layouts`
async fn api_card_builder_layouts_post(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let Some(auth_user) = session.auth_user.clone() else {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "auth_required", "loginUrl": "/login"}))).into_response();
    };
    let form_csrf = headers
        .get(header::HeaderName::from_static("x-csrf-token"))
        .and_then(|v| v.to_str().ok());
    if !crate::routes::auth::csrf_validate(&session, form_csrf) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_csrf"}))).into_response();
    }
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_payload"}))).into_response(),
    };
    let Some(sb) = state.sb.as_ref() else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response();
    };
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    let layout_id = match normalize_layout_id(&auth_state::as_str_of(parsed.get("id")).to_string().trim()) {
        Ok(id) => id,
        Err(_) => {
            // Empty/missing -> generate
            if auth_state::as_str_of(parsed.get("id")).is_empty() {
                new_uuid()
            } else {
                return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_layout_id"}))).into_response();
            }
        }
    };
    let card_type = normalize_card_type(&auth_state::as_str_of(parsed.get("cardType")));
    let name = normalize_layout_name(&auth_state::as_str_of(parsed.get("name")), &card_type);
    let config = parsed.get("config").cloned().unwrap_or(Value::Object(Default::default()));
    let payload = json!({
        "id": layout_id,
        "auth_user_id": user_id,
        "name": name,
        "card_type": card_type,
        "config_json": config,
        "updated_at": auth_state::isoformat_utc(Some(chrono::Utc::now())),
    });
    match sb.upsert_card_builder_layout(payload).await {
        Some(saved) => {
            let _ = &mut session;
            crate::routes::auth::json_no_store(json!({"ok": true, "layout": card_builder_layout_response(&saved)}))
        }
        None => (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "storage_unavailable"}))).into_response(),
    }
}

/// `DELETE /api/card-builder/layouts/{layout_id}`
async fn api_card_builder_layouts_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(layout_id): Path<String>,
) -> Response {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let Some(auth_user) = session.auth_user.clone() else {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "auth_required", "loginUrl": "/login"}))).into_response();
    };
    let form_csrf = headers
        .get(header::HeaderName::from_static("x-csrf-token"))
        .and_then(|v| v.to_str().ok());
    if !crate::routes::auth::csrf_validate(&session, form_csrf) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_csrf"}))).into_response();
    }
    let Ok(layout_id_norm) = normalize_layout_id(&layout_id) else {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_layout_id"}))).into_response();
    };
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    if let Some(sb) = state.sb.as_ref() {
        let _ = sb.delete_card_builder_layout(&user_id, &layout_id_norm).await;
    }
    let _ = &mut session;
    crate::routes::auth::json_no_store(json!({"ok": true, "id": layout_id_norm}))
}

fn card_builder_layout_response(row: &Value) -> Value {
    let cfg = row.get("config_json").cloned().unwrap_or(Value::Object(Default::default()));
    let cfg = if cfg.is_object() { cfg } else { Value::Object(Default::default()) };
    let card_type_raw = row.get("card_type").and_then(Value::as_str);
    let card_type = normalize_card_type(card_type_raw.unwrap_or("skater"));
    json!({
        "id": auth_state::as_str_of(row.get("id")),
        "name": normalize_layout_name(&auth_state::as_str_of(row.get("name")), &card_type),
        "cardType": card_type,
        "createdAt": auth_state::isoformat_utc(auth_state::parse_iso_datetime(row.get("created_at"))),
        "updatedAt": auth_state::isoformat_utc(auth_state::parse_iso_datetime(row.get("updated_at"))),
        "config": cfg,
    })
}

// ── tiny UUID helpers (v4, dashed) ────────────────────────────────
mod uuid_lite {
    use crate::routes::account::rand_bytes;

    pub fn v4() -> String {
        let mut b = rand_bytes(16);
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        let h: Vec<String> = b.iter().map(|x| format!("{:02x}", x)).collect();
        format!(
            "{}{}{}{}-{}{}-{}{}-{}{}-{}{}{}{}{}{}",
            h[0], h[1], h[2], h[3],
            h[4], h[5],
            h[6], h[7],
            h[8], h[9],
            h[10], h[11], h[12], h[13], h[14], h[15]
        )
    }

    pub fn parse(raw: &str) -> Option<String> {
        let clean: String = raw.chars().filter(|c| *c != '-').collect();
        if clean.len() != 32 || !clean.chars().all(|c| c.is_ascii_hexdigit()) {
            return None;
        }
        Some(raw.to_string())
    }
}

pub fn rand_bytes(n: usize) -> Vec<u8> {
    let mut out = vec![0u8; n];
    let _ = getrandom::getrandom(&mut out);
    out
}
