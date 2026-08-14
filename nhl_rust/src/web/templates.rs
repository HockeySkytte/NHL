//! minijinja template environment + context helpers.
//!
//! The 23 Jinja templates are shared with the Flask app (`app/templates`) and
//! are nearly compatible as-is. The two Flask-specific callables used by the
//! templates are provided here:
//! - `url_for(...)` — route-name → path mapping (blueprint names included),
//! - `get_flashed_messages(...)` — returns an empty list until the session
//!   flash store lands in M5.

use std::collections::BTreeMap;
use std::sync::Arc;

use minijinja::value::{Kwargs, Object, ObjectRepr, Rest, Value};
use minijinja::{Environment, Error, ErrorKind, State};

use crate::config::Config;
use crate::error::ApiError;
use crate::web::auth_state;
use crate::web::session::SessionData;

pub const SOCIAL_DEFAULT_TITLE: &str = "NHL Analytics · Hockey-Statistics";
pub const SOCIAL_DEFAULT_DESCRIPTION: &str = "Advanced NHL analytics: live games, standings, skater and goalie stats, xG heat maps, line combinations, and game projections — all powered by Hockey-Statistics.";

/// `_AUTH_PLAN_OPTIONS` — the two Pro plans shown on the account page.
pub fn auth_plan_options() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({
            "key": "monthly",
            "label": "Pro Monthly",
            "price_label": "$5/month",
            "detail": "Full access to Projections with monthly billing.",
        }),
        serde_json::json!({
            "key": "yearly",
            "label": "Pro Yearly",
            "price_label": "$40/year",
            "detail": "Same access with the lowest annual price.",
        }),
    ]
}

#[derive(Clone)]
pub struct TemplateEnv {
    env: Environment<'static>,
}

impl TemplateEnv {
    pub fn new(cfg: &Config) -> Result<Self, ApiError> {
        let mut env = Environment::new();
        env.set_loader(minijinja::path_loader(&cfg.template_dir));
        env.add_function("url_for", url_for_fn);
        env.add_function("get_flashed_messages", get_flashed_messages);
        Ok(Self { env })
    }

    pub fn render(&self, name: &str, ctx: Value) -> Result<String, ApiError> {
        let template = self.env.get_template(name).map_err(|e| {
            tracing::error!("template load failed for {name}: {e:#}");
            ApiError::Template(format!("template {name}: {e}"))
        })?;
        template.render(ctx).map_err(|e| {
            tracing::error!("template render failed for {name}: {e:#}");
            ApiError::Template(format!("render {name}: {e}"))
        })
    }
}

/// `get_flashed_messages(with_categories=true)` — reads the `_flashes` context
/// variable (set by `base_context_with_session`) as a list of `[category, msg]`.
fn get_flashed_messages(
    state: &State<'_, '_>,
    _args: Rest<Value>,
    _kwargs: Kwargs,
) -> Result<Value, Error> {
    let flashes = state
        .lookup("_flashes")
        .unwrap_or_else(|| Value::from(Vec::<Value>::new()));
    Ok(flashes)
}
/// Flask-compatible `request.url_root`: scheme + host + "/".
/// Priority: `APP_BASE_URL`, request `Host` header, loopback fallback.
pub fn url_root(cfg: &Config, host: Option<&str>) -> String {
    if let Some(base) = cfg.app_base_url.as_deref().filter(|b| !b.is_empty()) {
        return format!("{}/", base.trim_end_matches('/'));
    }
    if let Some(h) = host.filter(|h| !h.is_empty()) {
        return format!("http://{h}/");
    }
    format!("http://127.0.0.1:{}/", cfg.port)
}

/// Builds the base template context (port of `inject_auth_state()` + the
/// page-route context values) with a decoded session. `extra` holds
/// page-specific keys.
///
/// Returns a native minijinja `Value` (not JSON) because `request.url_root`
/// must be an object exposing `.rstrip('/')` — see `StrippableString`.
pub fn base_context_with_session(
    cfg: &Config,
    host: Option<&str>,
    path: &str,
    teams: &[serde_json::Value],
    session: &SessionData,
    extra: BTreeMap<&'static str, serde_json::Value>,
) -> Value {
    let root = url_root(cfg, host);
    let mut map: BTreeMap<&'static str, Value> = BTreeMap::new();
    map.insert("teams", Value::from_serialize(teams));
    let auth_enabled = crate::supabase::read::auth_is_configured();
    map.insert("auth_enabled", Value::from(auth_enabled));
    let auth_user = session
        .auth_user
        .as_ref()
        .map(|r| auth_state::auth_state_from_record(r, cfg.auth_trial_days));
    map.insert(
        "auth_user",
        match auth_user {
            Some(v) => Value::from_serialize(&v),
            None => Value::from(()),
        },
    );
    map.insert(
        "auth_plan_options",
        Value::from_serialize(&auth_plan_options()),
    );
    map.insert(
        "auth_login_target",
        Value::from(
            auth_state::safe_next_url(path).unwrap_or_else(|| "/projections".to_string()),
        ),
    );
    map.insert("csrf_token", Value::from(session.csrf_token.clone()));
    map.insert("social_default_title", Value::from(SOCIAL_DEFAULT_TITLE));
    map.insert(
        "social_default_description",
        Value::from(SOCIAL_DEFAULT_DESCRIPTION),
    );
    map.insert(
        "social_default_image",
        Value::from(format!("{root}static/social-preview.png")),
    );
    map.insert(
        "social_default_url",
        Value::from(format!("{}{}", root.trim_end_matches('/'), path)),
    );
    map.insert(
        "request",
        Value::from_object(RequestObj {
            url_root: root,
        }),
    );
    map.insert("_flashes", Value::from_serialize(&session.flashes));
    for (key, value) in extra {
        map.insert(key, Value::from_serialize(&value));
    }
    Value::from(map)
}

/// Session-less `base_context` (legacy callers): empty session, empty CSRF.
pub fn base_context(
    cfg: &Config,
    host: Option<&str>,
    path: &str,
    teams: &[serde_json::Value],
    auth_user: Option<serde_json::Value>,
    extra: BTreeMap<&'static str, serde_json::Value>,
) -> Value {
    let mut session = SessionData::default();
    if let Some(u) = auth_user {
        session.auth_user = Some(u);
    }
    base_context_with_session(cfg, host, path, teams, &session, extra)
}

/// `request` object: attribute `url_root` returns a `StrippableString` so the
/// templates can call `request.url_root.rstrip('/')` like Flask/Jinja allow.
#[derive(Debug)]
struct RequestObj {
    url_root: String,
}

impl Object for RequestObj {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Map
    }

    fn get_value(self: &Arc<Self>, key: &Value) -> Option<Value> {
        if key.as_str() == Some("url_root") {
            Some(Value::from_object(StrippableString {
                inner: self.url_root.clone(),
            }))
        } else {
            None
        }
    }
}

/// String-like object with an `rstrip(chars)` method (used by base.html).
#[derive(Debug)]
struct StrippableString {
    inner: String,
}

impl Object for StrippableString {
    fn repr(self: &Arc<Self>) -> ObjectRepr {
        ObjectRepr::Plain
    }

    fn call_method(
        self: &Arc<Self>,
        _state: &State<'_, '_>,
        name: &str,
        args: &[Value],
    ) -> Result<Value, Error> {
        if name == "rstrip" {
            let chars: Vec<char> = args
                .first()
                .and_then(Value::as_str)
                .map(|s| s.chars().collect())
                .unwrap_or_default();
            let trimmed = self
                .inner
                .trim_end_matches(|c: char| chars.contains(&c))
                .to_string();
            // Safe string: URL roots must not be HTML-escaped (minijinja escapes
            // `/` -> `&#x2f;`, which Jinja2 does not, and which breaks URLs that
            // end up inside <script> blocks).
            Ok(Value::from_safe_string(trimmed))
        } else {
            Err(Error::new(
                ErrorKind::UnknownMethod,
                format!("string has no method named {name}"),
            ))
        }
    }
}

/// `url_for(name, **kwargs)` for the routes the templates reference.
///
/// Returns **safe** strings: minijinja's HTML escaper escapes `/` -> `&#x2f;`,
/// which Jinja2 does not. url_for output is used both in HTML attributes (where
/// the browser decodes entities, so escaping is harmless) AND inside `<script>`
/// blocks (where entities are NOT decoded, so `/` escaping breaks URLs). Marking
/// it safe reproduces Flask's observable output everywhere.
fn url_for_fn(
    _state: &minijinja::State,
    args: Rest<Value>,
    kwargs: Kwargs,
) -> Result<Value, minijinja::Error> {
    let name = args.0.first().and_then(Value::as_str).unwrap_or("");
    if name == "static" {
        let filename = kwargs.get::<String>("filename").unwrap_or_default();
        return Ok(Value::from_safe_string(format!(
            "/static/{}",
            filename.trim_start_matches('/')
        )));
    }
    let path = match name {
        "main.login_page" => {
            let next = kwargs.get::<String>("next").unwrap_or_default();
            if next.is_empty() {
                "/login".to_string()
            } else {
                format!("/login?next={next}")
            }
        }
        "main.signup_page" => {
            let next = kwargs.get::<String>("next").unwrap_or_default();
            if next.is_empty() {
                "/signup".to_string()
            } else {
                format!("/signup?next={next}")
            }
        }
        "main.about_page_slug" => {
            let slug = kwargs.get::<String>("section_slug").unwrap_or_default();
            format!("/about/{slug}")
        }
        "main.account_page" => "/account".to_string(),
        "main.account_plan_update_page" => "/account/plan".to_string(),
        "main.account_billing_portal_page" => "/account/billing".to_string(),
        "main.account_donate_page" => "/account/donate".to_string(),
        "main.account_unsubscribe_page" => "/account/unsubscribe".to_string(),
        "main.account_profile_update_page" => "/account/profile".to_string(),
        "main.account_password_update_page" => "/account/password".to_string(),
        "main.account_delete_page" => "/account/delete".to_string(),
        "main.donation_page" => "/donation".to_string(),
        "main.game_projections_page" => "/projections".to_string(),
        "main.user_management_page" => "/admin/users".to_string(),
        "main.user_management_sync_page" => "/admin/users/sync".to_string(),
        "main.user_management_create_page" => "/admin/users/create".to_string(),
        "main.user_management_password_page" => {
            let uid = kwargs.get::<String>("user_id").unwrap_or_default();
            format!("/admin/users/{uid}/password")
        }
        "main.user_management_free_page" => {
            let uid = kwargs.get::<String>("user_id").unwrap_or_default();
            format!("/admin/users/{uid}/free")
        }
        "main.user_management_cancel_free_page" => {
            let uid = kwargs.get::<String>("user_id").unwrap_or_default();
            format!("/admin/users/{uid}/cancel-free")
        }
        "main.user_management_delete_page" => {
            let uid = kwargs.get::<String>("user_id").unwrap_or_default();
            format!("/admin/users/{uid}/delete")
        }
        "main.logout_page" => "/logout".to_string(),
        "main.about_page" => "/about".to_string(),
        _ => "/".to_string(),
    };
    Ok(Value::from_safe_string(path))
}
