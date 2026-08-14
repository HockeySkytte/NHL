//! Runtime configuration loaded from environment variables.
//!
//! Env var names and defaults mirror the Flask app (`app/__init__.py` and
//! `app/routes.py`) so the Rust service is a drop-in replacement. See
//! `PORT_PLAN.md` §7 for the full parity table.

use std::path::PathBuf;

#[derive(Debug, Clone)]
pub struct Config {
    /// `FLASK_SECRET_KEY` | `SECRET_KEY` (dev fallback kept for parity).
    pub secret_key: String,
    /// `APP_BASE_URL` — canonical base URL used for absolute URLs / `request.url_root`.
    pub app_base_url: Option<String>,
    /// `HTTP_COMPRESS` (default `1`; enabled only when exactly "1", like Flask).
    pub http_compress: bool,
    /// `COMPRESS_MIN_SIZE` (default 512, clamped to >= 256 like Flask).
    pub compress_min_size: u64,
    /// `COMPRESS_LEVEL` (default 6, clamped 1..=9 like Flask).
    pub compress_level: i32,
    /// `XG_PRELOAD` (default 1).
    pub xg_preload: bool,
    /// `PRELOAD_GM_CACHES` (default 1).
    pub preload_gm_caches: bool,
    /// `PRESTART_LOGGER` (default 1).
    pub prestart_logger: bool,
    /// `XG_CACHE_DIR` — on-disk cache override.
    pub xg_cache_dir: Option<PathBuf>,
    /// `TEMPLATE_DIR` — Jinja template folder (default `../app/templates`).
    pub template_dir: PathBuf,
    /// `STATIC_DIR` — static assets folder (default `../app/static`).
    pub static_dir: PathBuf,
    /// `TEAMS_CSV_PATH` — fallback Teams.csv location (default `../Teams.csv`).
    pub teams_csv_path: PathBuf,
    /// `LAST_DATES_CSV_PATH` — Last_date.csv location (default `../Last_date.csv`).
    pub last_dates_csv_path: PathBuf,
    /// `MODEL_DIR` — directory of the pickled xG model files (default `../Model`).
    pub model_dir: PathBuf,
    /// `ABOUT_DATA_JSON` — about-page content extracted from routes.py (default `data/about.json`).
    pub about_data_json: PathBuf,
    /// `PORT` (Render) — default 5000 for Flask dev parity.
    pub port: u16,
    /// `EMAIL` — admin login email (used for the minimal admin session fallback).
    pub admin_email: String,
    /// `PASSWORD` — admin login password.
    pub admin_password: String,
    /// `SUPABASE_URL` — project URL for PostgREST + GoTrue.
    pub supabase_url: Option<String>,
    /// `SUPABASE_SERVICE_KEY` — service-role key (admin reads/writes + GoTrue admin).
    pub supabase_service_key: Option<String>,
    /// `SUPABASE_ANON_KEY` | `SUPABASE_PUBLISHABLE_KEY` — client key for password sign-in.
    pub supabase_anon_key: Option<String>,
    /// `AUTH_TRIAL_DAYS` (default 14) — trial length for new signups.
    pub auth_trial_days: u32,
    /// `STRIPE_SECRET_KEY` — Stripe API key (billing).
    pub stripe_secret_key: Option<String>,
    /// `STRIPE_WEBHOOK_SECRET` — webhook signing secret.
    pub stripe_webhook_secret: Option<String>,
    /// `STRIPE_PRICE_MONTHLY_ID` / `STRIPE_PRICE_YEARLY_ID`.
    pub stripe_price_monthly_id: Option<String>,
    pub stripe_price_yearly_id: Option<String>,
    /// `COMMUNITY_SITE_URL` — community feed fallback base URL.
    pub community_site_url: Option<String>,
    /// `COMMUNITY_NHL_HUB_ID` — hub filter for Supabase community posts.
    pub community_nhl_hub_id: Option<String>,
    /// `COMMUNITY_FEED_CACHE_TTL_SECONDS` (default 300).
    pub community_feed_cache_ttl_seconds: u64,
    /// `PRESTART_CSV` / `PRESTART_DIR` — prestart snapshot output.
    pub prestart_csv: String,
    pub prestart_dir: Option<std::path::PathBuf>,
}

fn env_nonempty(name: &str) -> Option<String> {
    std::env::var(name)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

/// Mirrors Python `os.getenv(name, '1') == '1'`: enabled only when the value
/// is exactly `"1"`, defaulting to `default` when unset.
fn env_is_one(name: &str, default: bool) -> bool {
    match std::env::var(name) {
        Ok(v) => v.trim() == "1",
        Err(_) => default,
    }
}

impl Config {
    pub fn from_env() -> Self {
        let secret_key = env_nonempty("FLASK_SECRET_KEY")
            .or_else(|| env_nonempty("SECRET_KEY"))
            .unwrap_or_else(|| "nhl-dev-secret-change-me".to_string());

        let compress_min_size = env_nonempty("COMPRESS_MIN_SIZE")
            .and_then(|v| v.parse::<u64>().ok())
            .map(|v| v.max(256))
            .unwrap_or(512);
        let compress_level = env_nonempty("COMPRESS_LEVEL")
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| v.clamp(1, 9))
            .unwrap_or(6);
        let port = env_nonempty("PORT")
            .and_then(|v| v.parse::<u16>().ok())
            .unwrap_or(5000);

        // Defaults are relative to the nhl_rust/ working directory, which in
        // dev points back at the shared Flask tree (single source of truth).
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));

        Self {
            secret_key,
            app_base_url: env_nonempty("APP_BASE_URL"),
            http_compress: env_is_one("HTTP_COMPRESS", true),
            compress_min_size,
            compress_level,
            xg_preload: env_is_one("XG_PRELOAD", true),
            preload_gm_caches: env_is_one("PRELOAD_GM_CACHES", true),
            prestart_logger: env_is_one("PRESTART_LOGGER", true),
            xg_cache_dir: env_nonempty("XG_CACHE_DIR").map(PathBuf::from),
            template_dir: env_nonempty("TEMPLATE_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("../app/templates")),
            static_dir: env_nonempty("STATIC_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("../app/static")),
            teams_csv_path: env_nonempty("TEAMS_CSV_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("../Teams.csv")),
            last_dates_csv_path: env_nonempty("LAST_DATES_CSV_PATH")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("../Last_date.csv")),
            model_dir: env_nonempty("MODEL_DIR")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("../Model")),
            about_data_json: env_nonempty("ABOUT_DATA_JSON")
                .map(PathBuf::from)
                .unwrap_or_else(|| cwd.join("data/about.json")),
            port,
            admin_email: env_nonempty("EMAIL").unwrap_or_default(),
            admin_password: env_nonempty("PASSWORD").unwrap_or_default(),
            supabase_url: env_nonempty("SUPABASE_URL"),
            supabase_service_key: env_nonempty("SUPABASE_SERVICE_KEY"),
            supabase_anon_key: env_nonempty("SUPABASE_ANON_KEY")
                .or_else(|| env_nonempty("SUPABASE_PUBLISHABLE_KEY")),
            auth_trial_days: env_nonempty("AUTH_TRIAL_DAYS")
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(14),
            stripe_secret_key: env_nonempty("STRIPE_SECRET_KEY"),
            stripe_webhook_secret: env_nonempty("STRIPE_WEBHOOK_SECRET"),
            stripe_price_monthly_id: env_nonempty("STRIPE_PRICE_MONTHLY_ID"),
            stripe_price_yearly_id: env_nonempty("STRIPE_PRICE_YEARLY_ID"),
            community_site_url: env_nonempty("COMMUNITY_SITE_URL"),
            community_nhl_hub_id: env_nonempty("COMMUNITY_NHL_HUB_ID"),
            community_feed_cache_ttl_seconds: env_nonempty("COMMUNITY_FEED_CACHE_TTL_SECONDS")
                .and_then(|v| v.parse::<u64>().ok())
                .map(|v| v.max(30))
                .unwrap_or(300),
            prestart_csv: env_nonempty("PRESTART_CSV")
                .unwrap_or_else(|| "prestart_snapshots.csv".to_string()),
            prestart_dir: env_nonempty("PRESTART_DIR")
                .or_else(|| env_nonempty("XG_CACHE_DIR"))
                .map(std::path::PathBuf::from),
        }
    }
}
