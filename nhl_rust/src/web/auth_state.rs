//! Pure auth state machine — port of the Flask auth helpers in
//! `app/routes.py` (lines ~137–1375). No HTTP; all functions are deterministic
//! transforms of the Supabase user / `user_accounts` records so the Rust app
//! reproduces the exact trial/access/plan semantics.

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde_json::{json, Value};

/// Default trial length (days), overridable via `AUTH_TRIAL_DAYS`.
pub const AUTH_TRIAL_DAYS_DEFAULT: u32 = 14;

// ── datetime helpers ──────────────────────────────────────────────

/// Port of `_parse_iso_datetime`: parse an ISO-8601 string to UTC.
pub fn parse_iso_datetime(value: Option<&Value>) -> Option<DateTime<Utc>> {
    if let Some(dt) = value.and_then(Value::as_str) {
        let raw = dt.trim();
        if raw.is_empty() {
            return None;
        }
        return DateTime::parse_from_rfc3339(raw)
            .map(|d| d.with_timezone(&Utc))
            .ok();
    }
    None
}

/// Port of `_isoformat_utc`: `...Z` with Python-style microseconds (omitted
/// when zero).
pub fn isoformat_utc(value: Option<DateTime<Utc>>) -> String {
    let Some(dt) = value else {
        return String::new();
    };
    let base = dt.format("%Y-%m-%dT%H:%M:%S").to_string();
    let nanos = dt.nanosecond();
    if nanos == 0 {
        format!("{base}Z")
    } else {
        format!("{base}.{:06}Z", nanos / 1000)
    }
}

fn now_utc() -> DateTime<Utc> {
    Utc::now()
}

fn parse_or(value: &Value, fallback: DateTime<Utc>) -> DateTime<Utc> {
    parse_iso_datetime(Some(value)).unwrap_or(fallback)
}

pub fn as_str_of(value: Option<&Value>) -> String {
    value.and_then(Value::as_str).unwrap_or("").trim().to_string()
}

pub fn as_str_lower(value: Option<&Value>) -> String {
    as_str_of(value).to_lowercase()
}

pub fn as_bool_of(value: Option<&Value>) -> bool {
    value.and_then(Value::as_bool).unwrap_or(false)
}

/// Python `datetime + timedelta(days=n)` on a UTC datetime.
fn add_days(dt: DateTime<Utc>, days: i64) -> DateTime<Utc> {
    dt + chrono::Duration::days(days)
}

// ── username / email validation (regex parity with Python) ───────

pub fn normalize_username(value: &str) -> String {
    // re.sub(r'[^a-z0-9._-]+', '-', lower).strip('-')
    let lower = value.trim().to_lowercase();
    let re = match regex::Regex::new(r"[^a-z0-9._-]+") {
        Ok(re) => re,
        Err(_) => return lower,
    };
    re.replace_all(&lower, "-").trim_matches('-').to_string()
}

pub fn valid_username(value: &str) -> bool {
    let username = normalize_username(value);
    // r'[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?'
    let re = match regex::Regex::new(r"^[a-z0-9](?:[a-z0-9._-]{1,30}[a-z0-9])?$") {
        Ok(re) => re,
        Err(_) => return false,
    };
    re.is_match(&username)
}

pub fn valid_email(value: &str) -> bool {
    // r'^[^@\s]+@[^@\s]+\.[^@\s]+$'
    let re = match regex::Regex::new(r"^[^@\s]+@[^@\s]+\.[^@\s]+$") {
        Ok(re) => re,
        Err(_) => return false,
    };
    re.is_match(&value.trim().to_lowercase())
}

/// Port of `_auth_username_candidate`.
pub fn auth_username_candidate(record: &Value, existing_record: Option<&Value>) -> String {
    let existing_username = existing_record
        .map(|r| as_str_of(r.get("username")))
        .unwrap_or_default();
    if !existing_username.is_empty() {
        return existing_username;
    }
    let email = as_str_of(record.get("email"));
    let email_local = email.split('@').next().unwrap_or("").trim().to_lowercase();
    let cleaned_email: String = email_local
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '.' || *c == '_' || *c == '-')
        .collect();
    if !cleaned_email.is_empty() {
        return cleaned_email.chars().take(64).collect();
    }
    let name = as_str_of(record.get("display_name")).to_lowercase();
    // re.sub(r'[^a-z0-9._-]+', '-', name).strip('-')
    let re = match regex::Regex::new(r"[^a-z0-9._-]+") {
        Ok(re) => re,
        Err(_) => return String::new(),
    };
    let cleaned_name = re.replace_all(&name, "-").trim_matches('-').to_string();
    if !cleaned_name.is_empty() {
        return cleaned_name.chars().take(64).collect();
    }
    String::new()
}

// ── record building / merging ─────────────────────────────────────

/// Port of `_merge_auth_user_account` — the account record wins.
pub fn merge_auth_user_account(base_record: &Value, account_record: Option<&Value>) -> Value {
    let Some(account) = account_record else {
        return base_record.clone();
    };
    let merged = base_record.as_object().cloned().unwrap_or_default();
    let mut out = serde_json::Map::new();
    let get = |keys: &[&str]| -> Value {
        for k in keys {
            if let Some(v) = account.get(*k) {
                if !v.is_null() && as_str_of(Some(v)).is_empty() == false {
                    return v.clone();
                }
            }
            if let Some(v) = merged.get(*k) {
                if !v.is_null() {
                    return v.clone();
                }
            }
        }
        Value::Null
    };
    out.insert("user_id".into(), as_str_of(Some(&get(&["auth_user_id", "user_id"]))).into());
    out.insert("email".into(), as_str_of(Some(&get(&["email"]))).to_lowercase().into());
    out.insert("username".into(), as_str_of(Some(&get(&["username"]))).into());
    let display = account.get("display_name");
    let display = if !as_str_of(Some(&display.unwrap_or(&Value::Null))).is_empty() {
        display.unwrap().clone()
    } else {
        let uname = as_str_of(account.get("username"));
        if !uname.is_empty() {
            Value::String(uname)
        } else {
            get(&["display_name"])
        }
    };
    let display = as_str_of(Some(&display));
    out.insert("display_name".into(), if display.is_empty() { "Account".to_string() } else { display }.into());
    out.insert(
        "trial_started_at".into(),
        isoformat_utc(
            parse_iso_datetime(account.get("trial_started_at"))
                .or_else(|| parse_iso_datetime(merged.get("trial_started_at"))),
        )
        .into(),
    );
    out.insert(
        "trial_expires_at".into(),
        isoformat_utc(
            parse_iso_datetime(account.get("trial_expires_at"))
                .or_else(|| parse_iso_datetime(merged.get("trial_expires_at"))),
        )
        .into(),
    );
    out.insert(
        "subscription_status".into(),
        as_str_lower(Some(&get(&["subscription_status"]))).into(),
    );
    out.insert("subscription_plan".into(), as_str_of(Some(&get(&["subscription_plan"]))).into());
    out.insert(
        "billing_interval".into(),
        as_str_lower(Some(&get(&["billing_interval"]))).into(),
    );
    let is_admin = as_bool_of(account.get("is_admin")) || as_bool_of(merged.get("is_admin"));
    out.insert("is_admin".into(), is_admin.into());
    out.insert(
        "subscription_source".into(),
        as_str_of(Some(&get(&["subscription_source"]))).into(),
    );
    out.insert(
        "stripe_customer_id".into(),
        as_str_of(Some(&get(&["stripe_customer_id"]))).into(),
    );
    out.insert(
        "stripe_subscription_id".into(),
        as_str_of(Some(&get(&["stripe_subscription_id"]))).into(),
    );
    out.insert("stripe_price_id".into(), as_str_of(Some(&get(&["stripe_price_id"]))).into());
    out.insert(
        "stripe_current_period_end".into(),
        isoformat_utc(
            parse_iso_datetime(account.get("stripe_current_period_end"))
                .or_else(|| parse_iso_datetime(merged.get("stripe_current_period_end"))),
        )
        .into(),
    );
    Value::Object(out)
}

/// Port of `_auth_record_from_supabase_user(user, account_record=None)`.
pub fn auth_record_from_supabase_user(
    user: &Value,
    account_record: Option<&Value>,
    trial_days: u32,
) -> Value {
    let user_meta = user.get("user_metadata").cloned().unwrap_or(Value::Object(Default::default()));
    let app_meta = user.get("app_metadata").cloned().unwrap_or(Value::Object(Default::default()));
    let created_at = parse_iso_datetime(user.get("created_at")).unwrap_or_else(now_utc);
    let trial_started = parse_iso_datetime(user_meta.get("trial_started_at")).unwrap_or(created_at);
    let trial_expires = parse_iso_datetime(user_meta.get("trial_expires_at"))
        .unwrap_or_else(|| add_days(trial_started, i64::from(trial_days)));
    let email = as_str_of(user.get("email"));
    let display = as_str_of(Some(&first_nonempty(&[
        user_meta.get("display_name"),
        user_meta.get("name"),
        user.get("email"),
    ])));
    let base_record = json!({
        "user_id": as_str_of(user.get("id")),
        "email": email,
        "username": as_str_of(user_meta.get("username")),
        "display_name": if display.is_empty() { "Account".to_string() } else { display },
        "created_at": isoformat_utc(Some(created_at)),
        "trial_started_at": isoformat_utc(Some(trial_started)),
        "trial_expires_at": isoformat_utc(Some(trial_expires)),
        "subscription_status": as_str_lower(Some(&first_nonempty(&[
            app_meta.get("subscription_status"),
            user_meta.get("subscription_status"),
        ]))),
        "subscription_plan": as_str_of(Some(&first_nonempty(&[
            app_meta.get("subscription_plan"),
            user_meta.get("subscription_plan"),
        ]))),
        "billing_interval": as_str_lower(Some(&first_nonempty(&[
            app_meta.get("billing_interval"),
            user_meta.get("billing_interval"),
        ]))),
        "is_admin": as_bool_of(Some(&first_nonempty(&[
            app_meta.get("is_admin"),
            user_meta.get("is_admin"),
        ]))),
    });
    merge_auth_user_account(&base_record, account_record)
}

fn first_nonempty(values: &[Option<&Value>]) -> Value {
    for v in values {
        if let Some(v) = v {
            let v: &Value = v;
            if !v.is_null() && as_str_of(Some(v)).is_empty() == false {
                return v.clone();
            }
        }
    }
    Value::Null
}

/// Port of `_auth_state_from_record` — computes the full auth-user state dict.
pub fn auth_state_from_record(record: &Value, trial_days: u32) -> Value {
    let created_at = parse_iso_datetime(record.get("created_at")).unwrap_or_else(now_utc);
    let trial_started = parse_iso_datetime(record.get("trial_started_at")).unwrap_or(created_at);
    let trial_expires = parse_iso_datetime(record.get("trial_expires_at"))
        .unwrap_or_else(|| add_days(trial_started, i64::from(trial_days)));
    let subscription_status = as_str_lower(record.get("subscription_status"));
    let now = now_utc();
    let trial_active = trial_expires > now;
    let has_subscription = subscription_status == "active" || subscription_status == "paid";
    let is_admin = as_bool_of(record.get("is_admin"));
    let has_access = is_admin || has_subscription || trial_active;
    let remaining_seconds = (trial_expires - now).num_milliseconds().max(0) as f64 / 1000.0;
    let remaining_days = if remaining_seconds > 0.0 {
        (remaining_seconds / 86400.0).ceil() as i64
    } else {
        0
    };
    let billing_interval = as_str_lower(record.get("billing_interval"));
    let subscription_plan = as_str_of(record.get("subscription_plan"));

    let (access_label, plan_label) = if is_admin {
        ("Admin access".to_string(), "Admin".to_string())
    } else if subscription_plan == "free" && has_subscription {
        ("Free access".to_string(), "Free access".to_string())
    } else if has_subscription {
        let access = "Subscription active".to_string();
        let plan = if billing_interval == "yearly" {
            "Pro yearly".to_string()
        } else if billing_interval == "monthly" {
            "Pro monthly".to_string()
        } else {
            subscription_plan.clone()
        };
        (access, plan)
    } else if trial_active {
        let label = if remaining_days == 1 {
            "1 day left in trial".to_string()
        } else {
            format!("{remaining_days} days left in trial")
        };
        (label, "14-day free trial".to_string())
    } else {
        (
            "Trial ended".to_string(),
            if subscription_plan.is_empty() {
                "No active plan".to_string()
            } else {
                subscription_plan.clone()
            },
        )
    };

    let mut out = record.as_object().cloned().unwrap_or_default();
    out.insert("created_at".into(), isoformat_utc(Some(created_at)).into());
    out.insert("trial_started_at".into(), isoformat_utc(Some(trial_started)).into());
    out.insert("trial_expires_at".into(), isoformat_utc(Some(trial_expires)).into());
    out.insert("subscription_status".into(), subscription_status.into());
    out.insert("subscription_plan".into(), subscription_plan.into());
    out.insert("billing_interval".into(), billing_interval.into());
    out.insert("trial_active".into(), trial_active.into());
    out.insert("has_access".into(), has_access.into());
    out.insert("has_subscription".into(), has_subscription.into());
    out.insert("trial_days_remaining".into(), remaining_days.into());
    out.insert("access_label".into(), access_label.into());
    out.insert("plan_label".into(), plan_label.into());
    out.insert("is_authenticated".into(), true.into());
    out.insert("subscription_source".into(), as_str_of(record.get("subscription_source")).into());
    out.insert("stripe_customer_id".into(), as_str_of(record.get("stripe_customer_id")).into());
    out.insert("stripe_subscription_id".into(), as_str_of(record.get("stripe_subscription_id")).into());
    out.insert("stripe_price_id".into(), as_str_of(record.get("stripe_price_id")).into());
    out.insert(
        "stripe_current_period_end".into(),
        isoformat_utc(parse_iso_datetime(record.get("stripe_current_period_end"))).into(),
    );
    Value::Object(out)
}

/// Port of `_set_auth_session`'s stored payload — the compact subset kept in
/// the session cookie (Flask stores this under `session['auth_user']`).
pub fn auth_session_payload(state: &Value) -> Value {
    let keys = [
        "user_id",
        "email",
        "username",
        "display_name",
        "created_at",
        "trial_started_at",
        "trial_expires_at",
        "subscription_status",
        "subscription_plan",
        "billing_interval",
        "is_admin",
        "subscription_source",
        "stripe_customer_id",
        "stripe_subscription_id",
        "stripe_price_id",
        "stripe_current_period_end",
    ];
    let mut out = serde_json::Map::new();
    for k in keys {
        out.insert(k.to_string(), state.get(k).cloned().unwrap_or(Value::Null));
    }
    Value::Object(out)
}

/// Port of `_build_account_payload(auth_user, updates=None)`.
pub fn build_account_payload(auth_user: &Value, updates: Option<&Value>) -> Value {
    let mut out = serde_json::Map::new();
    out.insert("auth_user_id".into(), as_str_of(auth_user.get("user_id")).into());
    out.insert("email".into(), as_str_lower(auth_user.get("email")).into());
    let username = as_str_of(auth_user.get("username"));
    out.insert(
        "username".into(),
        if username.is_empty() {
            auth_username_candidate(auth_user, None)
        } else {
            username
        }
        .into(),
    );
    let display = as_str_of(auth_user.get("display_name"));
    let display = if display.is_empty() {
        let uname = as_str_of(auth_user.get("username"));
        if !uname.is_empty() {
            uname
        } else {
            as_str_of(auth_user.get("email"))
        }
    } else {
        display
    };
    out.insert("display_name".into(), if display.is_empty() { "Account".to_string() } else { display }.into());
    out.insert("is_admin".into(), as_bool_of(auth_user.get("is_admin")).into());
    let status = as_str_lower(auth_user.get("subscription_status"));
    out.insert("subscription_status".into(), if status.is_empty() { "inactive".to_string() } else { status }.into());
    let plan = as_str_of(auth_user.get("subscription_plan"));
    out.insert("subscription_plan".into(), if plan.is_empty() { "inactive".to_string() } else { plan }.into());
    let interval = as_str_lower(auth_user.get("billing_interval"));
    out.insert(
        "billing_interval".into(),
        if interval.is_empty() { Value::Null } else { Value::String(interval) },
    );
    out.insert("trial_started_at".into(), opt_str(auth_user.get("trial_started_at")));
    out.insert("trial_expires_at".into(), opt_str(auth_user.get("trial_expires_at")));
    out.insert("subscription_started_at".into(), opt_str(auth_user.get("subscription_started_at")));
    out.insert("subscription_ends_at".into(), opt_str(auth_user.get("subscription_ends_at")));
    out.insert("subscription_source".into(), opt_str(auth_user.get("subscription_source")));
    out.insert("stripe_customer_id".into(), opt_str(auth_user.get("stripe_customer_id")));
    out.insert("stripe_subscription_id".into(), opt_str(auth_user.get("stripe_subscription_id")));
    out.insert("stripe_price_id".into(), opt_str(auth_user.get("stripe_price_id")));
    out.insert("stripe_current_period_end".into(), opt_str(auth_user.get("stripe_current_period_end")));
    out.insert("updated_at".into(), isoformat_utc(Some(now_utc())).into());
    if let Some(updates) = updates {
        if let Value::Object(u) = updates {
            for (k, v) in u {
                out.insert(k.clone(), v.clone());
            }
        }
    }
    Value::Object(out)
}

fn opt_str(v: Option<&Value>) -> Value {
    let s = as_str_of(v);
    if s.is_empty() {
        Value::Null
    } else {
        Value::String(s)
    }
}

/// Port of `_subscription_update_for_plan(plan_key, current_auth_user)`.
pub fn subscription_update_for_plan(plan_key: &str, _current_auth_user: &Value) -> Value {
    let now = isoformat_utc(Some(now_utc()));
    match plan_key {
        "monthly" => json!({
            "subscription_status": "active",
            "subscription_plan": "pro",
            "billing_interval": "monthly",
            "subscription_started_at": now,
            "subscription_ends_at": Value::Null,
        }),
        "yearly" => json!({
            "subscription_status": "active",
            "subscription_plan": "pro",
            "billing_interval": "yearly",
            "subscription_started_at": now,
            "subscription_ends_at": Value::Null,
        }),
        "free" => json!({
            "subscription_status": "active",
            "subscription_plan": "free",
            "billing_interval": Value::Null,
            "subscription_started_at": now,
            "subscription_ends_at": Value::Null,
        }),
        _ => json!({
            "subscription_status": "canceled",
            "subscription_plan": "canceled",
            "billing_interval": Value::Null,
            "subscription_ends_at": now,
        }),
    }
}

/// Port of `_auth_error_message`.
pub fn auth_error_message(exc: &str, fallback: &str) -> String {
    let raw = exc.trim();
    if raw.is_empty() {
        return fallback.to_string();
    }
    let lowered = raw.to_lowercase();
    if lowered.contains("already registered") {
        return "That email is already registered. Try logging in instead.".to_string();
    }
    if lowered.contains("invalid login credentials") {
        return "Invalid email or password.".to_string();
    }
    if lowered.contains("password") && lowered.contains("weak") {
        return "Choose a stronger password.".to_string();
    }
    raw.to_string()
}

// ── next-URL / crawler helpers ────────────────────────────────────

/// Port of `_safe_next_url` — rejects non-local, encoded, or query-bearing
/// redirect targets (prevents open-redirect / redirect loops).
pub fn safe_next_url(value: &str) -> Option<String> {
    let raw = value.trim();
    if raw.is_empty() || !raw.starts_with('/') || raw.starts_with("//") {
        return None;
    }
    if raw.contains('%') {
        return None;
    }
    if raw.contains('?') {
        return None;
    }
    let clean = raw.trim_end_matches('/');
    if clean == "/login" || clean == "/signup" {
        return None;
    }
    if raw.contains("://") {
        return None;
    }
    Some(raw.to_string())
}

pub fn auth_redirect_target(default: &str, next: Option<&str>) -> String {
    next.and_then(safe_next_url)
        .unwrap_or_else(|| default.to_string())
}

/// Port of `_auth_login_target` — current path (without query) is the target.
pub fn auth_login_target(path: &str, _full_path: &str) -> String {
    safe_next_url(path).unwrap_or_else(|| "/projections".to_string())
}

/// Port of `_is_crawler_request`.
pub fn is_crawler_request(method: &str, user_agent: Option<&str>) -> bool {
    let method_upper = method.to_uppercase();
    if method_upper != "GET" && method_upper != "HEAD" {
        return false;
    }
    let Some(ua) = user_agent else {
        return false;
    };
    let ua = ua.trim().to_lowercase();
    if ua.is_empty() {
        return false;
    }
    const TOKENS: [&str; 12] = [
        "meta-externalagent",
        "facebookexternalhit",
        "slackbot",
        "discordbot",
        "linkedinbot",
        "twitterbot",
        "whatsapp",
        "telegrambot",
        "skypeuripreview",
        "crawler",
        "spider",
        "bot",
    ];
    TOKENS.iter().any(|t| ua.contains(t))
}

/// Port of `_auth_is_premium_path` — which paths require premium access.
pub fn auth_is_premium_path(path: &str) -> bool {
    if path.is_empty() {
        return false;
    }
    const GM_PUBLIC: [&str; 5] = [
        "/api/projections/team-season-points-custom",
        "/api/projections/all-teams-custom",
        "/api/projections/simulate-season",
        "/api/projections/simulate-season-batch",
        "/api/projections/custom-lineups-cache",
    ];
    let trimmed = path.trim_end_matches('/');
    if GM_PUBLIC.contains(&trimmed) {
        return false;
    }
    const PAGE_PREFIXES: [&str; 1] = ["/projections"];
    const API_PREFIXES: [&str; 1] = ["/api/projections/"];
    for prefix in PAGE_PREFIXES.iter().chain(API_PREFIXES.iter()) {
        if path == *prefix || path.starts_with(prefix) {
            return true;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn username_validation_matches_python() {
        assert!(valid_username("jane.doe"));
        assert!(valid_username("a"));
        assert!(!valid_username("ab")); // needs 3+ total
        assert!(valid_username("abc"));
        assert!(valid_username("has space")); // normalizes to has-space (Python parity)
        assert!(valid_username("-lead")); // normalizes to lead (Python parity)
        assert!(valid_username("valid-username_1.x"));
        assert_eq!(normalize_username(" John  Doe! "), "john-doe");
    }

    #[test]
    fn email_validation() {
        assert!(valid_email("a@b.co"));
        assert!(!valid_email("nope"));
        assert!(!valid_email("a b@c.d"));
    }

    #[test]
    fn safe_next_rejects_bad_targets() {
        assert_eq!(safe_next_url("/gm-mode"), Some("/gm-mode".to_string()));
        assert_eq!(safe_next_url("//evil.com"), None);
        assert_eq!(safe_next_url("http://evil.com"), None);
        assert_eq!(safe_next_url("/login?next=/x"), None);
        assert_eq!(safe_next_url("/login"), None);
        assert_eq!(safe_next_url("relative"), None);
        assert_eq!(safe_next_url("/a%2Fb"), None);
    }

    #[test]
    fn crawler_detection() {
        assert!(is_crawler_request("GET", Some("Googlebot/2.1")));
        assert!(is_crawler_request("GET", Some("Mozilla facebookexternalhit/1.1")));
        assert!(!is_crawler_request("POST", Some("Googlebot")));
        assert!(!is_crawler_request("GET", Some("Mozilla/5.0 Chrome")));
        assert!(!is_crawler_request("GET", None));
    }

    #[test]
    fn premium_path_matches_flask() {
        assert!(auth_is_premium_path("/projections"));
        assert!(auth_is_premium_path("/api/projections/games"));
        assert!(auth_is_premium_path("/api/projections/team-season-points"));
        assert!(!auth_is_premium_path("/api/projections/team-season-points-custom"));
        assert!(!auth_is_premium_path("/api/projections/simulate-season"));
        assert!(!auth_is_premium_path("/api/projections/custom-lineups-cache"));
        assert!(!auth_is_premium_path("/gm-mode"));
    }

    #[test]
    fn trial_state_computation() {
        let now = Utc::now();
        let expires = now + chrono::Duration::days(10);
        let record = json!({
            "user_id": "u1",
            "email": "a@b.c",
            "subscription_status": "trialing",
            "subscription_plan": "trial",
            "trial_started_at": isoformat_utc(Some(now - chrono::Duration::days(4))),
            "trial_expires_at": isoformat_utc(Some(expires)),
        });
        let state = auth_state_from_record(&record, 14);
        assert_eq!(state["has_access"], true);
        assert_eq!(state["trial_active"], true);
        assert_eq!(state["trial_days_remaining"], 10);
        assert_eq!(state["access_label"], "10 days left in trial");
        assert_eq!(state["plan_label"], "14-day free trial");
    }

    #[test]
    fn isoformat_matches_python_shape() {
        let dt = DateTime::parse_from_rfc3339("2026-08-14T01:00:51.604432100+00:00")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(isoformat_utc(Some(dt)), "2026-08-14T01:00:51.604432Z");
        let dt2 = DateTime::parse_from_rfc3339("2026-08-14T01:00:51Z")
            .unwrap()
            .with_timezone(&Utc);
        assert_eq!(isoformat_utc(Some(dt2)), "2026-08-14T01:00:51Z");
    }
}
