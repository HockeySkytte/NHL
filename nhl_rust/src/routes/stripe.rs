//! Stripe billing (M5): checkout/portal/donation session creation + webhook
//! sync. Raw HTTP to the Stripe API (no SDK), consistent with the raw
//! PostgREST/GoTrue approach elsewhere.

use std::collections::{BTreeMap, HashMap};

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use hmac::{Hmac, Mac};
use serde_json::{json, Value};
use sha2::Sha256;

use crate::config::Config;
use crate::state::AppState;
use crate::web::auth_state;
use crate::web::session::SessionData;

type HmacSha256 = Hmac<Sha256>;

pub fn router() -> Router<AppState> {
    Router::new().route("/stripe/webhook", post(stripe_webhook))
}

const STRIPE_API: &str = "https://api.stripe.com/v1";

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(header::HOST).and_then(|v| v.to_str().ok())
}

fn stripe_portal_enabled(cfg: &Config) -> bool {
    !cfg.stripe_secret_key.as_deref().unwrap_or("").is_empty()
}

fn app_base_url(cfg: &Config, host: Option<&str>) -> String {
    crate::web::templates::url_root(cfg, host)
        .trim_end_matches('/')
        .to_string()
}

fn stripe_price_id(cfg: &Config, plan_key: &str) -> String {
    match plan_key {
        "monthly" => cfg.stripe_price_monthly_id.clone().unwrap_or_default(),
        "yearly" => cfg.stripe_price_yearly_id.clone().unwrap_or_default(),
        _ => String::new(),
    }
}

fn stripe_client(cfg: &Config) -> Option<reqwest::Client> {
    if cfg.stripe_secret_key.as_deref().unwrap_or("").is_empty() {
        return None;
    }
    Some(reqwest::Client::new())
}

fn flash_and_redirect(state: &AppState, session: &SessionData, to: &str, msg: (&str, &str)) -> Response {
    let mut s = session.clone();
    crate::routes::auth::flash(&mut s, msg.0, msg.1);
    crate::routes::auth::redirect_with_session(to, state, &s)
}

/// `POST /account/plan` path when Stripe is configured — create a checkout
/// session and redirect (303).
pub async fn create_checkout_redirect(
    state: &AppState,
    session: &mut SessionData,
    auth_user: &Value,
    plan_key: &str,
) -> Response {
    let cfg = &state.cfg;
    let missing = crate::routes::account::stripe_missing_config(cfg, Some(plan_key));
    if !missing.is_empty() {
        return flash_and_redirect(
            state,
            session,
            "/account",
            ("error", &format!("Stripe billing is not fully configured yet. Missing: {}.", missing.join(", "))),
        );
    }
    let Some(client) = stripe_client(cfg) else {
        return flash_and_redirect(state, session, "/account", ("error", "Stripe billing is not fully configured yet."));
    };
    let customer_id = auth_state::as_str_of(auth_user.get("stripe_customer_id"));
    let email = auth_state::as_str_of(auth_user.get("email")).to_lowercase();
    let user_id = auth_state::as_str_of(auth_user.get("user_id"));
    let base = app_base_url(cfg, None);
    let mut form: Vec<(String, String)> = vec![
        ("mode".into(), "subscription".into()),
        ("line_items[0][price]".into(), stripe_price_id(cfg, plan_key)),
        ("line_items[0][quantity]".into(), "1".into()),
        ("success_url".into(), format!("{base}/account?billing=success")),
        ("cancel_url".into(), format!("{base}/account?billing=canceled")),
        ("client_reference_id".into(), user_id.clone()),
        ("metadata[auth_user_id]".into(), user_id.clone()),
        ("metadata[plan_key]".into(), plan_key.to_string()),
        ("subscription_data[metadata][auth_user_id]".into(), user_id.clone()),
        ("subscription_data[metadata][plan_key]".into(), plan_key.to_string()),
        ("allow_promotion_codes".into(), "true".into()),
    ];
    if customer_id.is_empty() {
        form.push(("customer_email".into(), email));
    } else {
        form.push(("customer".into(), customer_id));
    }
    let resp = client
        .post(format!("{STRIPE_API}/checkout/sessions"))
        .bearer_auth(cfg.stripe_secret_key.as_deref().unwrap_or(""))
        .form(&form)
        .send()
        .await;
    match resp {
        Ok(r) if r.status().is_success() => {
            let body: Value = r.json().await.unwrap_or(Value::Null);
            let url = body.get("url").and_then(Value::as_str).unwrap_or("").to_string();
            if url.is_empty() {
                return flash_and_redirect(state, session, "/account", ("error", "Could not start Stripe checkout right now. Please try again in a moment."));
            }
            // 303 redirect to the hosted checkout.
            let mut resp = crate::routes::auth::redirect_see_other(&url);
            crate::routes::auth::attach_session_header(&mut resp, cfg, session);
            resp
        }
        Ok(r) => {
            let err = r.text().await.unwrap_or_default();
            let msg = checkout_error_message(&err);
            flash_and_redirect(state, session, "/account", ("error", &msg))
        }
        Err(_) => flash_and_redirect(state, session, "/account", ("error", "Could not start Stripe checkout right now. Please try again in a moment.")),
    }
}

/// `POST /account/billing` — create a billing portal session and redirect.
pub async fn create_billing_portal_redirect(
    state: &AppState,
    session: &mut SessionData,
    auth_user: &Value,
) -> Response {
    let cfg = &state.cfg;
    if !stripe_portal_enabled(cfg) {
        return flash_and_redirect(state, session, "/account", ("error", "Stripe billing portal is not configured yet."));
    }
    let customer_id = auth_state::as_str_of(auth_user.get("stripe_customer_id"));
    if customer_id.is_empty() {
        return flash_and_redirect(state, session, "/account", ("error", "No Stripe billing profile exists yet for this account. Start checkout first."));
    }
    let Some(client) = stripe_client(cfg) else {
        return flash_and_redirect(state, session, "/account", ("error", "Stripe billing portal is not configured yet."));
    };
    let base = app_base_url(cfg, None);
    let form: Vec<(String, String)> = vec![
        ("customer".into(), customer_id),
        ("return_url".into(), format!("{base}/account")),
    ];
    match client
        .post(format!("{STRIPE_API}/billing_portal/sessions"))
        .bearer_auth(cfg.stripe_secret_key.as_deref().unwrap_or(""))
        .form(&form)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            let body: Value = r.json().await.unwrap_or(Value::Null);
            let url = body.get("url").and_then(Value::as_str).unwrap_or("").to_string();
            if url.is_empty() {
                return flash_and_redirect(state, session, "/account", ("error", "Could not open Stripe billing right now. Please try again in a moment."));
            }
            let mut resp = crate::routes::auth::redirect_see_other(&url);
            crate::routes::auth::attach_session_header(&mut resp, cfg, session);
            resp
        }
        _ => flash_and_redirect(state, session, "/account", ("error", "Could not open Stripe billing right now. Please try again in a moment.")),
    }
}

/// `POST /account/donate` + `POST /donate` — one-time payment checkout.
pub async fn create_donation_checkout_redirect(
    state: &AppState,
    session: &mut SessionData,
    auth_user: Option<&Value>,
    amount_raw: &str,
    guest_email: &str,
) -> Response {
    let cfg = &state.cfg;
    let error_target = if auth_user.is_some() { "/account" } else { "/donation" };
    if !stripe_portal_enabled(cfg) {
        return flash_and_redirect(state, session, error_target, ("error", "Stripe is not configured yet for donations."));
    }
    let amount: f64 = match amount_raw.trim().parse() {
        Ok(v) => v,
        Err(_) => return flash_and_redirect(state, session, error_target, ("error", "Enter a valid donation amount.")),
    };
    let amount_cents = (amount * 100.0).round() as i64;
    if amount_cents < 100 {
        return flash_and_redirect(state, session, error_target, ("error", "Minimum donation is $1.00."));
    }
    if amount_cents > 500000 {
        return flash_and_redirect(state, session, error_target, ("error", "Maximum donation is $5,000.00 per checkout."));
    }
    let Some(client) = stripe_client(cfg) else {
        return flash_and_redirect(state, session, error_target, ("error", "Stripe is not configured yet for donations."));
    };
    let auth = auth_user.cloned().unwrap_or(Value::Null);
    let customer_id = auth_state::as_str_of(auth.get("stripe_customer_id"));
    let resolved_email = {
        let e = auth_state::as_str_of(auth.get("email")).to_lowercase();
        if !e.is_empty() { e } else { guest_email.trim().to_lowercase() }
    };
    let user_id = auth_state::as_str_of(auth.get("user_id"));
    let base = app_base_url(cfg, None);
    let success_base = if auth_user.is_some() { format!("{base}/account") } else { format!("{base}/donation") };
    let mut form: Vec<(String, String)> = vec![
        ("mode".into(), "payment".into()),
        ("line_items[0][price_data][currency]".into(), "usd".into()),
        ("line_items[0][price_data][product_data][name]".into(), "NHL Analytics Donation".into()),
        ("line_items[0][price_data][unit_amount]".into(), amount_cents.to_string()),
        ("line_items[0][quantity]".into(), "1".into()),
        ("success_url".into(), format!("{success_base}?billing=donation_success")),
        ("cancel_url".into(), format!("{success_base}?billing=donation_canceled")),
        ("client_reference_id".into(), user_id.clone()),
        ("metadata[auth_user_id]".into(), user_id.clone()),
        ("metadata[kind]".into(), "donation".into()),
        ("metadata[amount_cents]".into(), amount_cents.to_string()),
    ];
    if customer_id.is_empty() {
        if !resolved_email.is_empty() {
            form.push(("customer_email".into(), resolved_email));
        }
    } else {
        form.push(("customer".into(), customer_id));
    }
    match client
        .post(format!("{STRIPE_API}/checkout/sessions"))
        .bearer_auth(cfg.stripe_secret_key.as_deref().unwrap_or(""))
        .form(&form)
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            let body: Value = r.json().await.unwrap_or(Value::Null);
            let url = body.get("url").and_then(Value::as_str).unwrap_or("").to_string();
            if url.is_empty() {
                return flash_and_redirect(state, session, error_target, ("error", "Could not start donation checkout right now. Please try again in a moment."));
            }
            let mut resp = crate::routes::auth::redirect_see_other(&url);
            crate::routes::auth::attach_session_header(&mut resp, cfg, session);
            resp
        }
        _ => flash_and_redirect(state, session, error_target, ("error", "Could not start donation checkout right now. Please try again in a moment.")),
    }
}

fn checkout_error_message(raw: &str) -> String {
    let lowered = raw.to_lowercase();
    if lowered.contains("a similar object exists in test mode, but a live mode key was used") {
        return "Stripe is using a live secret key with test-mode Price IDs. Update STRIPE_PRICE_MONTHLY_ID and STRIPE_PRICE_YEARLY_ID to live prices.".to_string();
    }
    if lowered.contains("a similar object exists in live mode, but a test mode key was used") {
        return "Stripe is using a test secret key with live-mode Price IDs. Use a matching key/price mode pair.".to_string();
    }
    if lowered.contains("no such price") {
        return "The configured Stripe Price ID was not found. Verify STRIPE_PRICE_MONTHLY_ID and STRIPE_PRICE_YEARLY_ID in Render.".to_string();
    }
    "Could not start Stripe checkout right now. Please try again in a moment.".to_string()
}

// ── Webhook ───────────────────────────────────────────────────────

fn ct_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn verify_webhook_signature(secret: &str, payload: &[u8], signature_header: &str) -> bool {
    // Stripe-Signature: t=<timestamp>,v1=<hex>[,v1=<hex>...]
    let mut timestamp = String::new();
    let mut expected: Option<String> = None;
    for part in signature_header.split(',') {
        if let Some(v) = part.strip_prefix("t=") {
            timestamp = v.to_string();
        } else if let Some(v) = part.strip_prefix("v1=") {
            expected = Some(v.to_string());
        }
    }
    let (Some(expected), false) = (expected, timestamp.is_empty()) else {
        return false;
    };
    let mut mac = match HmacSha256::new_from_slice(secret.as_bytes()) {
        Ok(m) => m,
        Err(_) => return false,
    };
    mac.update(timestamp.as_bytes());
    mac.update(b".");
    mac.update(payload);
    let computed = hex_encode(&mac.finalize().into_bytes());
    ct_eq(computed.as_bytes(), expected.as_bytes())
}

/// `POST /stripe/webhook` — verify + sync subscription events.
async fn stripe_webhook(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> Response {
    let cfg = &state.cfg;
    let Some(secret) = cfg.stripe_webhook_secret.as_deref().filter(|s| !s.is_empty()) else {
        return (StatusCode::SERVICE_UNAVAILABLE, Json(json!({"error": "stripe_webhook_not_configured"}))).into_response();
    };
    let signature = headers
        .get(header::HeaderName::from_static("stripe-signature"))
        .and_then(|v| v.to_str().ok())
        .unwrap_or("");
    if !verify_webhook_signature(secret, &body, signature) {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_signature"}))).into_response();
    }
    let parsed: Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(_) => return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_payload"}))).into_response(),
    };
    let event_type = parsed.get("type").and_then(Value::as_str).unwrap_or("").to_string();
    let obj = parsed
        .get("data")
        .and_then(|d| d.get("object"))
        .cloned()
        .unwrap_or(Value::Null);
    if !obj.is_object() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_payload"}))).into_response();
    }
    match event_type.as_str() {
        "checkout.session.completed" => {
            sync_checkout_session(&state, &obj).await;
        }
        "customer.subscription.created" | "customer.subscription.updated" | "customer.subscription.deleted" => {
            sync_subscription(&state, &obj, None, None).await;
        }
        _ => {}
    }
    Json(json!({"received": true})).into_response()
}

/// Port of `_stripe_status_to_account_status`.
fn stripe_status_to_account_status(status: &str) -> String {
    let raw = status.trim().to_lowercase();
    match raw.as_str() {
        "active" | "trialing" | "past_due" | "canceled" => raw,
        "unpaid" => "past_due".to_string(),
        "incomplete_expired" => "expired".to_string(),
        _ => "inactive".to_string(),
    }
}

/// Port of `_stripe_price_from_subscription` + `_stripe_interval_from_subscription`.
fn stripe_interval_from_subscription(subscription: &Value) -> Option<String> {
    let items = subscription
        .get("items")
        .and_then(|i| i.get("data"))
        .and_then(Value::as_array);
    let price = items
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("price"));
    let interval = price
        .and_then(|p| p.get("recurring"))
        .and_then(|r| r.get("interval"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_lowercase();
    match interval.as_str() {
        "month" => Some("monthly".to_string()),
        "year" => Some("yearly".to_string()),
        _ => None,
    }
}

fn stripe_datetime(value: Option<&Value>) -> Option<chrono::DateTime<chrono::Utc>> {
    value
        .and_then(Value::as_i64)
        .or_else(|| value.and_then(Value::as_str).and_then(|s| s.parse::<i64>().ok()))
        .and_then(|ts| chrono::DateTime::from_timestamp(ts, 0))
}

/// Port of `_stripe_updates_from_subscription`.
fn stripe_updates_from_subscription(subscription: &Value, customer_id: Option<&str>) -> Value {
    let items = subscription
        .get("items")
        .and_then(|i| i.get("data"))
        .and_then(Value::as_array);
    let price = items
        .and_then(|arr| arr.first())
        .and_then(|item| item.get("price"));
    let price_id = price.and_then(|p| p.get("id")).and_then(Value::as_str).unwrap_or("").to_string();
    let billing_interval = stripe_interval_from_subscription(subscription).unwrap_or_default();
    let status = stripe_status_to_account_status(
        subscription.get("status").and_then(Value::as_str).unwrap_or(""),
    );
    let started_at = stripe_datetime(subscription.get("start_date"));
    let current_period_end = stripe_datetime(subscription.get("current_period_end"));
    let ended_at = stripe_datetime(subscription.get("canceled_at"))
        .or_else(|| stripe_datetime(subscription.get("ended_at")));
    let cancel_at_period_end = subscription.get("cancel_at_period_end").and_then(Value::as_bool).unwrap_or(false);
    let subscription_ends = ended_at.or_else(|| {
        if cancel_at_period_end {
            current_period_end
        } else {
            None
        }
    });
    let plan_value = if !price_id.is_empty() || !billing_interval.is_empty() {
        "pro"
    } else if status == "canceled" || status == "expired" {
        "canceled"
    } else {
        "inactive"
    };
    let stripe_customer = customer_id
        .map(|c| c.to_string())
        .or_else(|| subscription.get("customer").and_then(Value::as_str).map(|s| s.to_string()))
        .unwrap_or_default();
    json!({
        "subscription_status": status,
        "subscription_plan": plan_value,
        "billing_interval": billing_interval,
        "subscription_started_at": auth_state::isoformat_utc(started_at),
        "subscription_ends_at": auth_state::isoformat_utc(subscription_ends),
        "subscription_source": "stripe",
        "stripe_customer_id": if stripe_customer.is_empty() { Value::Null } else { Value::String(stripe_customer) },
        "stripe_subscription_id": subscription.get("id").and_then(Value::as_str).unwrap_or("").to_string(),
        "stripe_price_id": if price_id.is_empty() { Value::Null } else { Value::String(price_id) },
        "stripe_current_period_end": auth_state::isoformat_utc(current_period_end),
    })
}

fn as_str(v: Option<&Value>) -> String {
    crate::web::auth_state::as_str_of(v)
}

/// Port of `_sync_stripe_subscription`.
pub async fn sync_subscription(
    state: &AppState,
    subscription: &Value,
    auth_user_id: Option<&str>,
    customer_id: Option<&str>,
) -> Option<Value> {
    let Some(sb) = state.sb.as_ref() else {
        return None;
    };
    let metadata = subscription.get("metadata").cloned().unwrap_or(Value::Null);
    let mut resolved = auth_user_id
        .map(|s| s.to_string())
        .or_else(|| metadata.get("auth_user_id").and_then(Value::as_str).map(|s| s.to_string()));
    if resolved.as_deref().map(|s| s.is_empty()).unwrap_or(true) {
        let sub_id = as_str(subscription.get("id"));
        let cust = customer_id
            .map(|c| c.to_string())
            .or_else(|| subscription.get("customer").and_then(Value::as_str).map(|s| s.to_string()));
        // scan user_accounts for matching subscription/customer id
        if let Some(rows) = sb.list_user_accounts().await {
            for row in rows {
                let row_sub = as_str(row.get("stripe_subscription_id"));
                let row_cust = as_str(row.get("stripe_customer_id"));
                let cust_matches = cust
                    .as_deref()
                    .map(|c| !c.is_empty() && row_cust == c)
                    .unwrap_or(false);
                if (!sub_id.is_empty() && row_sub == sub_id) || cust_matches {
                    resolved = Some(as_str(row.get("auth_user_id")));
                    break;
                }
            }
        }
    }
    let user_id = resolved.filter(|s| !s.is_empty())?;
    let updates = stripe_updates_from_subscription(subscription, customer_id);
    let _ = sb.upsert_user_account(updates).await;
    Some(user_id.into())
}

/// Port of `_sync_stripe_checkout_session`.
pub async fn sync_checkout_session(state: &AppState, checkout: &Value) -> Option<Value> {
    let metadata = checkout.get("metadata").cloned().unwrap_or(Value::Null);
    let auth_user_id = metadata
        .get("auth_user_id")
        .and_then(Value::as_str)
        .or_else(|| checkout.get("client_reference_id").and_then(Value::as_str))
        .map(|s| s.to_string());
    let customer_id = checkout.get("customer").and_then(Value::as_str).map(|s| s.to_string());
    let subscription_id = checkout.get("subscription").and_then(Value::as_str).map(|s| s.to_string());
    if let Some(ref sub_id) = subscription_id {
        if let Some(cfg_client) = stripe_client(&state.cfg) {
            if let Ok(r) = cfg_client
                .get(format!("{STRIPE_API}/subscriptions/{sub_id}"))
                .bearer_auth(state.cfg.stripe_secret_key.as_deref().unwrap_or(""))
                .send()
                .await
            {
                if r.status().is_success() {
                    if let Ok(sub) = r.json::<Value>().await {
                        return sync_subscription(&state, &sub, auth_user_id.as_deref(), customer_id.as_deref()).await;
                    }
                }
            }
        }
    }
    if let Some(user_id) = auth_user_id.filter(|s| !s.is_empty()) {
        if let Some(sb) = state.sb.as_ref() {
            let payload = json!({
                "subscription_source": "stripe",
                "stripe_customer_id": opt_str(customer_id.as_deref()),
                "stripe_subscription_id": opt_str(subscription_id.as_deref()),
            });
            let _ = sb.upsert_user_account(payload).await;
        }
        return Some(user_id.into());
    }
    None
}

fn opt_str(v: Option<&str>) -> Value {
    match v {
        Some(s) if !s.is_empty() => Value::String(s.to_string()),
        _ => Value::Null,
    }
}
