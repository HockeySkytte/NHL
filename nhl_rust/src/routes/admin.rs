//! Admin routes: run the lineup scrape + GP estimation + Supabase sync as a
//! background job (admin-only), poll job status, and full user management.

use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use axum::extract::{Form, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::routes::auth::{is_admin, json_no_store};
use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/admin/run-lineups-gp", post(run_lineups_gp))
        .route("/api/admin/job/{job_id}", axum::routing::get(job_status))
        .route("/admin/users", axum::routing::get(user_management_page))
        .route("/admin/users/sync", post(user_management_sync))
        .route("/admin/users/create", post(user_management_create))
        .route("/admin/users/{user_id}/free", post(user_management_free))
        .route("/admin/users/{user_id}/cancel-free", post(user_management_cancel_free))
        .route("/admin/users/{user_id}/password", post(user_management_password))
        .route("/admin/users/{user_id}/delete", post(user_management_delete))
        .route("/admin/update", axum::routing::get(admin_update_page))
        .route("/admin/prestart-snapshots", axum::routing::get(admin_prestart_snapshots))
}

fn repo_root() -> PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf()
}

fn python_exe(repo: &PathBuf) -> String {
    let candidates = [
        repo.join(".venv").join("Scripts").join("python.exe"),
        repo.join(".venv").join("bin").join("python"),
    ];
    for c in &candidates {
        if c.exists() {
            return c.to_string_lossy().to_string();
        }
    }
    "python".to_string()
}

/// Spawn a python script and stream its combined output; returns Ok on exit 0.
async fn run_script(state: &AppState, job_id: &str, script: &str, extra: &[&str]) -> Result<(), String> {
    let repo = repo_root();
    let py = python_exe(&repo);
    let script_path = repo.join("scripts").join(script);
    let mut cmd = tokio::process::Command::new(&py);
    cmd.arg(script_path);
    cmd.args(extra);
    cmd.current_dir(&repo);
    let output = cmd
        .output()
        .await
        .map_err(|e| format!("failed to run {script}: {e}"))?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    if !output.status.success() {
        let tail = stdout.lines().rev().take(6).collect::<Vec<_>>().join("\n");
        let err_tail = stderr.lines().rev().take(6).collect::<Vec<_>>().join("\n");
        return Err(format!("{script} exited {:?}\n{tail}\n{err_tail}", output.status.code()));
    }
    // Record a short tail in the job log.
    let tail = stdout.lines().rev().take(6).collect::<Vec<_>>().join("\n");
    if let Ok(mut jobs) = state.jobs.lock() {
        if let Some(job) = jobs.get_mut(job_id) {
            job["log"] = json!(format!("{script}: {tail}"));
        }
    }
    Ok(())
}

/// `POST /api/admin/run-lineups-gp` — scrape lineups, estimate GP, sync to Supabase.
async fn run_lineups_gp(State(state): State<AppState>, headers: HeaderMap) -> Response {
    if !is_admin(&state.cfg, &headers) {
        return (StatusCode::UNAUTHORIZED, Json(json!({"error": "auth_required", "loginUrl": "/login?next=/gm-mode"})))
            .into_response();
    }
    let job_id = format!(
        "lgp-{}-{}",
        chrono::Utc::now().timestamp(),
        std::process::id()
    );
    {
        let mut jobs = state.jobs.lock().unwrap();
        jobs.insert(
            job_id.clone(),
            json!({"status": "running", "updated_at": chrono::Utc::now().to_rfc3339(), "log": ""}),
        );
    }

    let job_state = state.clone();
    let job_id_resp = job_id.clone();
    tokio::spawn(async move {
        let steps: [(&str, &[&str]); 3] = [
            ("lineups.py", &["--all", "--season", "20262027", "--quiet"]),
            ("estimate_gp.py", &[]),
            ("sync_lineups_to_supabase.py", &["--season", "20262027"]),
        ];
        let mut result: Result<(), String> = Ok(());
        for (script, extra) in steps {
            match run_script(&job_state, &job_id, script, extra).await {
                Ok(()) => {}
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        // Invalidate the lineups cache so the app picks up the new data.
        job_state.caches.lineups_all.invalidate(&());
        let status = match result {
            Ok(()) => "done",
            Err(e) => {
                if let Ok(mut jobs) = job_state.jobs.lock() {
                    if let Some(job) = jobs.get_mut(&job_id) {
                        job["error"] = json!(e);
                    }
                }
                "error"
            }
        };
        if let Ok(mut jobs) = job_state.jobs.lock() {
            if let Some(job) = jobs.get_mut(&job_id) {
                job["status"] = json!(status);
                job["updated_at"] = json!(chrono::Utc::now().to_rfc3339());
            }
        }
    });

    json_no_store(json!({"ok": true, "jobId": job_id_resp}))
}

/// `GET /api/admin/job/<job_id>` — poll status.
async fn job_status(State(state): State<AppState>, Path(job_id): Path<String>) -> Response {
    let jobs = state.jobs.lock().unwrap();
    match jobs.get(&job_id) {
        Some(job) => json_no_store(job.clone()),
        None => (StatusCode::NOT_FOUND, Json(json!({"error": "not_found"}))).into_response(),
    }
}

// ── User management (port of /admin/users routes) ────────────────

fn host_str(headers: &HeaderMap) -> Option<&str> {
    headers.get(axum::http::header::HOST).and_then(|v| v.to_str().ok())
}

fn form_get(form: &HashMap<String, String>, key: &str) -> String {
    form.get(key).cloned().unwrap_or_default()
}

fn filter_values(query: &HashMap<String, String>, form: Option<&HashMap<String, String>>) -> (String, String, String) {
    let get = |qk: &str, fk: &str| -> String {
        form.and_then(|f| f.get(fk).cloned())
            .or_else(|| query.get(qk).cloned())
            .unwrap_or_default()
    };
    let q = get("q", "filter_q").trim().to_string();
    let mut access = get("access", "filter_access").trim().to_lowercase();
    if access.is_empty() {
        access = "all".to_string();
    }
    let mut role = get("role", "filter_role").trim().to_lowercase();
    if role.is_empty() {
        role = "all".to_string();
    }
    if !["all", "trial", "free", "pro", "inactive"].contains(&access.as_str()) {
        access = "all".to_string();
    }
    if !["all", "admin", "member"].contains(&role.as_str()) {
        role = "all".to_string();
    }
    (q, access, role)
}

fn user_management_redirect(query: &HashMap<String, String>, form: Option<&HashMap<String, String>>) -> Response {
    let (q, access, role) = filter_values(query, form);
    let mut parts: Vec<String> = Vec::new();
    if !q.is_empty() {
        parts.push(format!("q={}", q));
    }
    if access != "all" {
        parts.push(format!("access={}", access));
    }
    if role != "all" {
        parts.push(format!("role={}", role));
    }
    let qs = if parts.is_empty() {
        String::new()
    } else {
        format!("?{}", parts.join("&"))
    };
    crate::routes::auth::redirect_found(&format!("/admin/users{qs}"))
}

fn admin_guard(state: &AppState, session: &crate::web::session::SessionData, path: &str) -> Result<(), Response> {
    let Some(raw) = session.auth_user.clone() else {
        return Err(crate::routes::auth::redirect_found(&format!("/login?next={path}")));
    };
    let full = crate::web::auth_state::auth_state_from_record(&raw, state.cfg.auth_trial_days);
    if crate::web::auth_state::as_bool_of(full.get("is_admin")) {
        Ok(())
    } else {
        let mut s = session.clone();
        crate::routes::auth::flash(&mut s, "error", "Admin access required.");
        Err(crate::routes::auth::redirect_with_session("/account", state, &s))
    }
}

fn csrf_valid(session: &crate::web::session::SessionData, form: &HashMap<String, String>) -> bool {
    crate::routes::auth::csrf_validate(session, form.get("csrf_token").map(|s| s.as_str()))
}

/// Ensure a `user_accounts` row exists for a Supabase auth user (port of
/// `_ensure_user_account_row`).
async fn ensure_user_account_row(state: &AppState, auth_user: &Value, existing: Option<&Value>) -> Option<Value> {
    let user_id = crate::web::auth_state::as_str_of(auth_user.get("id"));
    let sb = state.sb.as_ref()?;
    if user_id.is_empty() {
        return existing.cloned();
    }
    let auth_like = crate::web::auth_state::auth_record_from_supabase_user(auth_user, existing, state.cfg.auth_trial_days);
    let email = crate::web::auth_state::as_str_of(auth_like.get("email")).to_lowercase();
    if email.is_empty() {
        return existing.cloned();
    }
    let mut username = crate::web::auth_state::normalize_username(&crate::web::auth_state::as_str_of(auth_like.get("username")));
    if !crate::web::auth_state::valid_username(&username) {
        username = String::new();
    }
    let now = crate::web::auth_state::isoformat_utc(Some(chrono::Utc::now()));
    let payload = json!({
        "auth_user_id": user_id,
        "email": email,
        "username": if username.is_empty() { Value::Null } else { Value::String(username) },
        "display_name": crate::web::auth_state::as_str_of(auth_like.get("display_name")),
        "is_admin": crate::web::auth_state::as_bool_of(auth_like.get("is_admin")),
        "subscription_status": crate::web::auth_state::as_str_of(auth_like.get("subscription_status")),
        "subscription_plan": crate::web::auth_state::as_str_of(auth_like.get("subscription_plan")),
        "billing_interval": opt_str(auth_like.get("billing_interval")),
        "trial_started_at": opt_str(auth_like.get("trial_started_at")),
        "trial_expires_at": opt_str(auth_like.get("trial_expires_at")),
        "subscription_started_at": opt_str(auth_like.get("subscription_started_at")),
        "subscription_ends_at": opt_str(auth_like.get("subscription_ends_at")),
        "subscription_source": opt_str(auth_like.get("subscription_source")),
        "stripe_customer_id": opt_str(auth_like.get("stripe_customer_id")),
        "stripe_subscription_id": opt_str(auth_like.get("stripe_subscription_id")),
        "stripe_price_id": opt_str(auth_like.get("stripe_price_id")),
        "stripe_current_period_end": opt_str(auth_like.get("stripe_current_period_end")),
        "updated_at": now,
    });
    let saved = sb.upsert_user_account(payload).await;
    if let Some(saved) = saved {
        if !crate::web::auth_state::as_str_of(saved.get("auth_user_id")).is_empty() {
            return Some(saved);
        }
    }
    // fall back to a fresh fetch
    sb.get_user_account(&user_id).await
}

fn opt_str(v: Option<&Value>) -> Value {
    let s = crate::web::auth_state::as_str_of(v);
    if s.is_empty() {
        Value::Null
    } else {
        Value::String(s)
    }
}

/// Port of `_user_management_rows` — auth users joined with account rows.
async fn user_management_rows(
    state: &AppState,
    query: &str,
    access_filter: &str,
    role_filter: &str,
) -> Vec<Value> {
    let Some(sb) = state.sb.as_ref() else {
        return Vec::new();
    };
    let auth_users = sb.admin_list_users(1, 1000).await.unwrap_or_default();
    let accounts = sb.list_user_accounts().await.unwrap_or_default();
    let mut account_by_id: HashMap<String, Value> = HashMap::new();
    for row in &accounts {
        let uid = crate::web::auth_state::as_str_of(row.get("auth_user_id"));
        if !uid.is_empty() {
            account_by_id.insert(uid, row.clone());
        }
    }
    // Backfill missing rows.
    for au in &auth_users {
        let uid = crate::web::auth_state::as_str_of(au.get("id"));
        if uid.is_empty() || account_by_id.contains_key(&uid) {
            continue;
        }
        if let Some(created) = ensure_user_account_row(state, au, None).await {
            account_by_id.insert(uid, created);
        }
    }
    let q = query.trim().to_lowercase();
    let mut rows: Vec<Value> = Vec::new();
    for au in &auth_users {
        let uid = crate::web::auth_state::as_str_of(au.get("id"));
        if uid.is_empty() {
            continue;
        }
        let account = account_by_id.get(&uid);
        let record = crate::web::auth_state::auth_record_from_supabase_user(au, account, state.cfg.auth_trial_days);
        let mut state_row = crate::web::auth_state::auth_state_from_record(&record, state.cfg.auth_trial_days);
        let obj = state_row.as_object_mut().unwrap();
        obj.insert(
            "last_sign_in_at".into(),
            crate::web::auth_state::isoformat_utc(crate::web::auth_state::parse_iso_datetime(au.get("last_sign_in_at"))).into(),
        );
        obj.insert(
            "created_at".into(),
            crate::web::auth_state::isoformat_utc(crate::web::auth_state::parse_iso_datetime(au.get("created_at"))).into(),
        );
        let email_confirmed = au.get("email_confirmed_at").and_then(Value::as_str).is_some()
            || au.get("confirmed_at").and_then(Value::as_str).is_some();
        obj.insert("email_confirmed".into(), email_confirmed.into());

        let is_admin = crate::web::auth_state::as_bool_of(state_row.get("is_admin"));
        let plan = crate::web::auth_state::as_str_of(state_row.get("subscription_plan"));
        let interval = crate::web::auth_state::as_str_of(state_row.get("billing_interval"));
        let has_access = crate::web::auth_state::as_bool_of(state_row.get("has_access"));
        if role_filter == "admin" && !is_admin {
            continue;
        }
        if role_filter == "member" && is_admin {
            continue;
        }
        if access_filter == "trial" && plan != "trial" {
            continue;
        }
        if access_filter == "free" && plan != "free" {
            continue;
        }
        if access_filter == "pro" && interval != "monthly" && interval != "yearly" {
            continue;
        }
        if access_filter == "inactive" && has_access {
            continue;
        }
        if !q.is_empty() {
            let haystack = format!(
                "{} {} {} {} {}",
                crate::web::auth_state::as_str_of(state_row.get("email")),
                crate::web::auth_state::as_str_of(state_row.get("username")),
                crate::web::auth_state::as_str_of(state_row.get("display_name")),
                crate::web::auth_state::as_str_of(state_row.get("plan_label")),
                crate::web::auth_state::as_str_of(state_row.get("access_label")),
            )
            .to_lowercase();
            if !haystack.contains(&q) {
                continue;
            }
        }
        rows.push(state_row);
    }
    rows.sort_by(|a, b| {
        let a_admin = if crate::web::auth_state::as_bool_of(a.get("is_admin")) { 0 } else { 1 };
        let b_admin = if crate::web::auth_state::as_bool_of(b.get("is_admin")) { 0 } else { 1 };
        let a_email = crate::web::auth_state::as_str_of(a.get("email")).to_lowercase();
        let b_email = crate::web::auth_state::as_str_of(b.get("email")).to_lowercase();
        (a_admin, a_email).cmp(&(b_admin, b_email))
    });
    rows
}

/// `GET /admin/users` — user management page.
async fn user_management_page(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    let (q, access, role) = filter_values(&query, None);
    let users = user_management_rows(&state, &q, &access, &role).await;
    let counts = json!({
        "total": users.len(),
        "admins": users.iter().filter(|u| crate::web::auth_state::as_bool_of(u.get("is_admin"))).count(),
        "trial": users.iter().filter(|u| crate::web::auth_state::as_str_of(u.get("subscription_plan")) == "trial").count(),
        "free": users.iter().filter(|u| crate::web::auth_state::as_str_of(u.get("subscription_plan")) == "free").count(),
        "pro": users.iter().filter(|u| ["monthly", "yearly"].contains(&crate::web::auth_state::as_str_of(u.get("billing_interval")).as_str())).count(),
    });
    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!("User Management"));
    extra.insert("show_filters", json!(false));
    extra.insert("users", Value::Array(users));
    extra.insert("plan_options", Value::Array(crate::web::templates::auth_plan_options()));
    extra.insert("filters", json!({"q": q, "access": access, "role": role}));
    extra.insert("counts", counts);
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/admin/users",
        "user_management.html",
        &mut session,
        extra,
    )
}

/// `POST /admin/users/sync` — backfill missing user_accounts rows.
async fn user_management_sync(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    let Some(sb) = state.sb.as_ref() else {
        return Ok(user_management_redirect(&query, Some(&form)));
    };
    let auth_users = sb.admin_list_users(1, 1000).await.unwrap_or_default();
    let accounts = sb.list_user_accounts().await.unwrap_or_default();
    let mut account_by_id: HashMap<String, Value> = HashMap::new();
    for row in &accounts {
        account_by_id.insert(crate::web::auth_state::as_str_of(row.get("auth_user_id")), row.clone());
    }
    let mut scanned = 0;
    let mut inserted_or_updated = 0;
    let mut skipped = 0;
    let mut failed = 0;
    for au in &auth_users {
        let uid = crate::web::auth_state::as_str_of(au.get("id"));
        if uid.is_empty() {
            continue;
        }
        scanned += 1;
        let existing = account_by_id.get(&uid);
        if existing.is_some() {
            skipped += 1;
            continue;
        }
        match ensure_user_account_row(&state, au, existing).await {
            Some(created) if !crate::web::auth_state::as_str_of(created.get("auth_user_id")).is_empty() => {
                inserted_or_updated += 1;
                account_by_id.insert(uid, created);
            }
            _ => failed += 1,
        }
    }
    let msg = if failed > 0 {
        format!(
            "Sync completed with failures. scanned={scanned}, saved={inserted_or_updated}, skipped={skipped}, failed={failed}"
        )
    } else {
        format!("Sync complete. scanned={scanned}, saved={inserted_or_updated}, skipped={skipped}")
    };
    crate::routes::auth::flash(&mut session, if failed > 0 { "error" } else { "success" }, &msg);
    let mut resp = user_management_redirect(&query, Some(&form));
    crate::routes::auth::attach_session_header(&mut resp, &state.cfg, &session);
    Ok(resp)
}

/// `POST /admin/users/create` — create a user.
async fn user_management_create(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    let Some(sb) = state.sb.as_ref() else {
        return Ok(user_management_redirect(&query, Some(&form)));
    };
    let username = crate::web::auth_state::normalize_username(&form_get(&form, "username"));
    let email = form_get(&form, "email").trim().to_lowercase();
    let password = form_get(&form, "password");
    let confirm_password = form_get(&form, "confirm_password");
    let access = {
        let a = form_get(&form, "access").trim().to_lowercase();
        if a.is_empty() { "trial".to_string() } else { a }
    };
    let is_admin = ["1", "true", "on", "yes"].contains(&form_get(&form, "is_admin").trim().to_lowercase().as_str());

    let fail = |state: &AppState, session: &crate::web::session::SessionData, query: &HashMap<String, String>, form: &HashMap<String, String>, msg: &str| -> Result<Response, crate::error::ApiError> {
        let mut s = session.clone();
        crate::routes::auth::flash(&mut s, "error", msg);
        let mut resp = user_management_redirect(query, Some(form));
        crate::routes::auth::attach_session_header(&mut resp, &state.cfg, &s);
        Ok(resp)
    };

    if !crate::web::auth_state::valid_username(&username) {
        return fail(&state, &session, &query, &form, "Username must be 3-32 characters and use letters, numbers, dots, dashes, or underscores.");
    }
    if !crate::web::auth_state::valid_email(&email) {
        return fail(&state, &session, &query, &form, "Enter a valid email address.");
    }
    if let Some(accounts) = sb.list_user_accounts().await {
        let username_taken = accounts.iter().any(|r| crate::web::auth_state::normalize_username(&crate::web::auth_state::as_str_of(r.get("username"))) == username);
        let email_taken = accounts.iter().any(|r| crate::web::auth_state::as_str_of(r.get("email")).to_lowercase() == email);
        if username_taken {
            return fail(&state, &session, &query, &form, "That username is already in use.");
        }
        if email_taken {
            return fail(&state, &session, &query, &form, "That email address already has an account.");
        }
    }
    if let Some(auth_users) = sb.admin_list_users(1, 1000).await {
        if auth_users.iter().any(|u| crate::web::auth_state::as_str_of(u.get("email")).to_lowercase() == email) {
            return fail(&state, &session, &query, &form, "That email address already has an account.");
        }
    }
    if password.len() < 8 {
        return fail(&state, &session, &query, &form, "Password must be at least 8 characters.");
    }
    if password != confirm_password {
        return fail(&state, &session, &query, &form, "Passwords do not match.");
    }
    let now = chrono::Utc::now();
    let trial_expires = now + chrono::Duration::days(i64::from(state.cfg.auth_trial_days));
    let metadata = json!({
        "display_name": username,
        "username": username,
        "is_admin": is_admin,
        "trial_started_at": crate::web::auth_state::isoformat_utc(Some(now)),
        "trial_expires_at": crate::web::auth_state::isoformat_utc(Some(trial_expires)),
        "trial_days": state.cfg.auth_trial_days,
    });
    let created = sb.admin_create_user(&email, &password, &metadata).await;
    let Some(created_user) = created.and_then(|c| c.get("user").cloned()) else {
        return fail(&state, &session, &query, &form, "Unable to create user.");
    };
    let created_id = crate::web::auth_state::as_str_of(created_user.get("id"));
    if created_id.is_empty() {
        return fail(&state, &session, &query, &form, "Unable to create user.");
    }
    let auth_like = crate::web::auth_state::auth_record_from_supabase_user(&created_user, None, state.cfg.auth_trial_days);
    let mut updates = json!({
        "username": username,
        "display_name": username,
        "is_admin": is_admin,
        "trial_started_at": crate::web::auth_state::isoformat_utc(Some(now)),
        "trial_expires_at": crate::web::auth_state::isoformat_utc(Some(trial_expires)),
    });
    if access == "free" {
        let plan = crate::web::auth_state::subscription_update_for_plan("free", &auth_like);
        merge_updates(&mut updates, &plan);
    } else if access == "monthly" || access == "yearly" {
        let plan = crate::web::auth_state::subscription_update_for_plan(&access, &auth_like);
        merge_updates(&mut updates, &plan);
    } else {
        updates["subscription_status"] = json!("trialing");
        updates["subscription_plan"] = json!("trial");
        updates["billing_interval"] = Value::Null;
        updates["subscription_started_at"] = Value::Null;
        updates["subscription_ends_at"] = Value::Null;
    }
    let payload = crate::web::auth_state::build_account_payload(&auth_like, Some(&updates));
    let _ = sb.upsert_user_account(payload).await;
    crate::routes::auth::flash(
        &mut session,
        "success",
        &format!("User created for {email} with {} access.", if is_admin { "admin" } else { "member" }),
    );
    let mut resp = user_management_redirect(&query, Some(&form));
    crate::routes::auth::attach_session_header(&mut resp, &state.cfg, &session);
    Ok(resp)
}

fn merge_updates(target: &mut Value, source: &Value) {
    if let (Value::Object(t), Value::Object(s)) = (target, source) {
        for (k, v) in s {
            t.insert(k.clone(), v.clone());
        }
    }
}

async fn get_user_record(state: &AppState, user_id: &str) -> Option<Value> {
    let sb = state.sb.as_ref()?;
    let auth_row = sb.admin_get_user(user_id).await?;
    let account = sb.get_user_account(user_id).await;
    Some(crate::web::auth_state::auth_record_from_supabase_user(&auth_row, account.as_ref(), state.cfg.auth_trial_days))
}

async fn persist_for_user(state: &AppState, user_id: &str, updates: &Value) -> Option<Value> {
    let sb = state.sb.as_ref()?;
    let auth_row = sb.admin_get_user(user_id).await?;
    let account = sb.get_user_account(user_id).await;
    let auth_like = crate::web::auth_state::auth_record_from_supabase_user(&auth_row, account.as_ref(), state.cfg.auth_trial_days);
    let payload = crate::web::auth_state::build_account_payload(&auth_like, Some(updates));
    sb.upsert_user_account(payload).await
}

/// `POST /admin/users/{user_id}/free`
async fn user_management_free(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(user_id): Path<String>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    if form_get(&form, "confirm_free") != "1" {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Confirm the free-access change to continue."));
    }
    if let Some(auth_like) = get_user_record(&state, &user_id).await {
        let plan = crate::web::auth_state::subscription_update_for_plan("free", &auth_like);
        let _ = persist_for_user(&state, &user_id, &plan).await;
        let email = crate::web::auth_state::as_str_of(auth_like.get("email"));
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "success", &format!("{email} now has free access.")));
    }
    Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "User not found."))
}

/// `POST /admin/users/{user_id}/cancel-free`
async fn user_management_cancel_free(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(user_id): Path<String>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    if form_get(&form, "confirm_cancel_free") != "1" {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Confirm the cancel-free action to continue."));
    }
    if let Some(auth_like) = get_user_record(&state, &user_id).await {
        if crate::web::auth_state::as_str_of(auth_like.get("subscription_plan")) != "free" {
            return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "info", "This user is not currently on free access."));
        }
        let mut updates = crate::web::auth_state::subscription_update_for_plan("unsubscribe", &auth_like);
        updates["trial_expires_at"] = json!(crate::web::auth_state::isoformat_utc(Some(chrono::Utc::now())));
        let _ = persist_for_user(&state, &user_id, &updates).await;
        let email = crate::web::auth_state::as_str_of(auth_like.get("email"));
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "success", &format!("{email} free access has been canceled.")));
    }
    Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "User not found."))
}

/// `POST /admin/users/{user_id}/password`
async fn user_management_password(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(user_id): Path<String>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    let password = form_get(&form, "password");
    let confirm = form_get(&form, "confirm_password");
    if password.len() < 8 {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Password must be at least 8 characters."));
    }
    if password != confirm {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Passwords do not match."));
    }
    if let Some(sb) = state.sb.as_ref() {
        let _ = sb.admin_update_user(&user_id, &json!({"password": password})).await;
    }
    Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "success", "Password reset completed."))
}

/// `POST /admin/users/{user_id}/delete`
async fn user_management_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
    Path(user_id): Path<String>,
    Form(form): Form<HashMap<String, String>>,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/users") {
        return Ok(resp);
    }
    if !csrf_valid(&session, &form) {
        return Ok((StatusCode::BAD_REQUEST, "Invalid CSRF token").into_response());
    }
    if form_get(&form, "confirm_delete") != "1" {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Confirm the delete action to continue."));
    }
    let current_user_id = session
        .auth_user
        .as_ref()
        .map(|u| crate::web::auth_state::as_str_of(u.get("user_id")))
        .unwrap_or_default();
    if current_user_id == user_id {
        return Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "error", "Delete your own account from Account instead."));
    }
    if let Some(sb) = state.sb.as_ref() {
        let _ = sb.admin_delete_user(&user_id).await;
    }
    Ok(redirect_with_flash(&state, &mut session, &query, Some(&form), "success", "User deleted."))
}

fn redirect_with_flash(
    state: &AppState,
    session: &mut crate::web::session::SessionData,
    query: &HashMap<String, String>,
    form: Option<&HashMap<String, String>>,
    category: &str,
    msg: &str,
) -> Response {
    crate::routes::auth::flash(session, category, msg);
    let mut resp = user_management_redirect(query, form);
    crate::routes::auth::attach_session_header(&mut resp, &state.cfg, session);
    resp
}

/// `GET /admin/update` — port of the legacy update page (admin-only).
async fn admin_update_page(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, crate::error::ApiError> {
    let mut session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    if let Err(resp) = admin_guard(&state, &session, "/admin/update") {
        return Ok(resp);
    }
    let mut extra: BTreeMap<&'static str, serde_json::Value> = BTreeMap::new();
    extra.insert("active_tab", json!(""));
    extra.insert("show_filters", json!(false));
    crate::routes::auth::render_with_session(
        &state,
        host_str(&headers),
        "/admin/update",
        "update.html",
        &mut session,
        extra,
    )
}

/// `GET /admin/prestart-snapshots?mode=preview|download&limit=N&gameId=G`
/// — port of `admin_prestart_snapshots` (admin API guard).
async fn admin_prestart_snapshots(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<HashMap<String, String>>,
) -> Response {
    let session = crate::routes::auth::session_from_headers(&state.cfg, &headers);
    let Some(raw) = session.auth_user else {
        return (
            StatusCode::UNAUTHORIZED,
            Json(json!({"error": "auth_required", "loginUrl": "/login?next=/admin/prestart-snapshots"})),
        )
            .into_response();
    };
    let full = crate::web::auth_state::auth_state_from_record(&raw, state.cfg.auth_trial_days);
    if !crate::web::auth_state::as_bool_of(full.get("is_admin")) {
        return (StatusCode::FORBIDDEN, Json(json!({"error": "admin_required"}))).into_response();
    }
    let mode = query.get("mode").map(|s| s.trim().to_lowercase()).unwrap_or_else(|| "preview".to_string());
    let path = crate::jobs::prestart::prestart_csv_path(&state.cfg);

    if mode == "download" {
        if !path.exists() {
            return (StatusCode::NOT_FOUND, Json(json!({"error": "file_not_found", "path": path.display().to_string()}))).into_response();
        }
        match tokio::fs::read(&path).await {
            Ok(bytes) => {
                let filename = path.file_name().map(|f| f.to_string_lossy().to_string()).unwrap_or_default();
                let mut resp = axum::response::Response::new(axum::body::Body::from(bytes));
                resp.headers_mut().insert(
                    axum::http::header::CONTENT_TYPE,
                    axum::http::HeaderValue::from_static("text/csv; charset=utf-8"),
                );
                resp.headers_mut().insert(
                    axum::http::header::CONTENT_DISPOSITION,
                    axum::http::HeaderValue::from_str(&format!("attachment; filename=\"{filename}\"")).unwrap_or_else(|_| axum::http::HeaderValue::from_static("attachment")),
                );
                resp.headers_mut().insert(
                    axum::http::header::CACHE_CONTROL,
                    axum::http::HeaderValue::from_static("no-store"),
                );
                resp
            }
            Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"error": "download_failed"}))).into_response(),
        }
    } else {
        let limit = query
            .get("limit")
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(100);
        let game_id_filter = query.get("gameId").and_then(|s| s.parse::<i64>().ok());
        if !path.exists() {
            return crate::routes::auth::json_no_store(json!({"exists": false, "path": path.display().to_string(), "rows": [], "total": 0}));
        }
        let mut rows: Vec<Value> = Vec::new();
        match std::fs::read_to_string(&path) {
            Ok(content) => {
                let mut rdr = csv::Reader::from_reader(content.as_bytes());
                for record in rdr.records().flatten() {
                    let mut row = serde_json::Map::new();
                    for (i, field) in record.iter().enumerate() {
                        if let Some(key) = CSV_FIELDS_DEF.get(i) {
                            row.insert(key.to_string(), Value::String(field.to_string()));
                        }
                    }
                    if let Some(gid_filter) = game_id_filter {
                        let gid = row.get("GameID").and_then(Value::as_str).and_then(|s| s.parse::<i64>().ok());
                        if gid != Some(gid_filter) {
                            continue;
                        }
                    }
                    rows.push(Value::Object(row));
                }
            }
            Err(_) => {
                return (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({"exists": true, "path": path.display().to_string(), "rows": [], "total": 0, "error": "read_failed"}))).into_response();
            }
        }
        let total = rows.len();
        if limit > 0 && total > limit {
            let start = total - limit;
            rows = rows[start..].to_vec();
        }
        crate::routes::auth::json_no_store(json!({
            "exists": true,
            "path": path.display().to_string(),
            "total": total,
            "limit": limit,
            "rows": rows,
        }))
    }
}

const CSV_FIELDS_DEF: [&str; 12] = [
    "TimestampUTC", "DateET", "GameID", "StartTimeET",
    "Away", "Home", "WinAway", "WinHome", "OddsAway", "OddsHome", "BetAway", "BetHome",
];
