#![recursion_limit = "256"]

mod cache;
mod config;
mod data;
mod disk_cache;
mod error;
mod jobs;
mod manager;
mod models;
mod nhl;
mod routes;
mod state;
mod supabase;
mod util;
mod web;

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    dotenvy::dotenv().ok();
    // Local dev: also load the repo-root .env (Supabase creds etc.) when the
    // server is run from nhl_rust/.
    let _ = dotenvy::from_filename("../.env");
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("nhl_rust=info,tower_http=info")),
        )
        .init();

    let cfg = config::Config::from_env();
    let state = match state::AppState::build(cfg).await {
        Ok(s) => s,
        Err(e) => {
            tracing::error!("startup failed: {e}");
            std::process::exit(1);
        }
    };

    // Feature toggles are honored at startup (parity with create_app()).
    if state.cfg.xg_preload {
        tracing::info!("XG_PRELOAD=1 (xG model preload lands in M3)");
    }
    if state.cfg.preload_gm_caches {
        tracing::info!("PRELOAD_GM_CACHES=1 (GM cache preload lands in M4)");
    }
    if std::env::var("MANAGER_DRY_RUN").map(|v| v == "1").unwrap_or(false) {
        tracing::info!("MANAGER_DRY_RUN=1 (running the Manager dry run, then exiting)");
        match manager::dry_run::run(&state).await {
            Ok(()) => std::process::exit(0),
            Err(e) => {
                eprintln!("[dry-run] FAILED: {e}");
                std::process::exit(1);
            }
        }
    }
    if let Ok(id) = std::env::var("MANAGER_DELETE_LEAGUE") {
        let id = id.trim();
        if !id.is_empty() {
            tracing::info!("MANAGER_DELETE_LEAGUE={id} (deleting, then exiting)");
            match manager::dry_run::delete_league(&state, id).await {
                Ok(()) => std::process::exit(0),
                Err(e) => {
                    eprintln!("[dry-run] delete FAILED: {e}");
                    std::process::exit(1);
                }
            }
        }
    }
    if state.cfg.prestart_logger {
        tracing::info!("PRESTART_LOGGER=1 (prestart logger starting)");
        jobs::prestart::start(state.clone());
    }
    if std::env::var("MANAGER_SWEEPER").map(|v| v == "1").unwrap_or(false) {
        tracing::info!("MANAGER_SWEEPER=1 (manager sweeper starting)");
        jobs::manager::start(state.clone());
    }

    let port = state.cfg.port;
    let app = routes::build_router(state);
    let addr = format!("0.0.0.0:{port}");
    let listener = match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => listener,
        Err(e) => {
            tracing::error!("failed to bind {addr}: {e}");
            std::process::exit(1);
        }
    };
    tracing::info!("nhl-rust listening on http://{addr}");
    if let Err(e) = axum::serve(listener, app).await {
        tracing::error!("server error: {e}");
    }
}
