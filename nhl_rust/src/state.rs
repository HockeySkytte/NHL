//! Shared application state (Axum `State`) plus the module-level cache
//! registry mirroring `app/routes.py` (see PORT_PLAN.md §7 for TTL parity).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::cache::{env_max, env_ttl, SingleSlot, TtlCache};
use crate::config::Config;
use crate::data::about::AboutData;
use crate::data::teams;
use crate::error::ApiError;
use crate::supabase::read::SbClient;
use crate::web::templates::TemplateEnv;

/// All in-process TTL caches. TTL/max env vars match the Flask names.
pub struct Caches {
    pub box_cache: TtlCache<i64, Value>,
    pub skaters_players: TtlCache<(Vec<i64>, String, String), Value>,
    pub goalies_players: TtlCache<(Vec<i64>, String, String), Value>,
    pub team_seasons: TtlCache<String, Value>,
    pub player_landing: TtlCache<i64, Value>,
    pub team_logo: TtlCache<String, Vec<u8>>,
    pub player_headshot: TtlCache<String, Vec<u8>>,
    pub lineups_all: TtlCache<(), Value>,
    pub player_projections: TtlCache<(), Value>,
    pub odds_snapshot_rows: TtlCache<String, Value>,
    pub skater_bios: TtlCache<i64, Value>,
    pub goalie_bios: TtlCache<i64, Value>,
    pub all_rosters: TtlCache<(), Value>,
    pub teamseasonstats: TtlCache<(), Value>,
    pub club_schedule: TtlCache<(String, i64), Value>,
    pub seasonstats_agg: TtlCache<String, Value>,
    pub goalies_agg: TtlCache<String, Value>,
    pub career_matrix: TtlCache<String, Value>,
    pub rapm_static: TtlCache<(), Value>,
    pub context_static: TtlCache<(), Value>,
    pub card_metrics_defs: TtlCache<String, Value>,
    pub team_stats_rest: TtlCache<String, Value>,
    pub edge_api: TtlCache<String, Value>,
    pub skaters_scatter: TtlCache<String, Value>,
    pub goalies_scatter: TtlCache<String, Value>,
    pub goalie_team_by_season: TtlCache<(i64, String), Value>,
    pub lt_shifts: SingleSlot<String, Value>,
    pub lt_pbp: SingleSlot<String, Value>,
    pub lt_data: SingleSlot<String, Value>,
    pub lt_base: SingleSlot<String, Value>,
    pub skaters_shooting: TtlCache<String, Value>,
    pub goalies_goaltending: TtlCache<String, Value>,
    pub player_names: TtlCache<i64, Value>,
    pub pbp: TtlCache<(i64, String), Value>,
    pub shifts: TtlCache<i64, Value>,
    pub model: TtlCache<String, std::sync::Arc<crate::models::xg::XgModel>>,
    pub box_id_map: TtlCache<(), Value>,
    pub all_rosters_by_season: TtlCache<i64, Value>,
    pub v2_build: TtlCache<String, Value>,
    pub current_player_projections: TtlCache<(), Value>,
    pub gm_projections: TtlCache<(), Value>,
    pub custom_lineups: TtlCache<(String, i64), Value>,
    pub playoff_bracket: TtlCache<i64, Value>,
    pub community_feed: TtlCache<String, Value>,
}

impl Caches {
    pub fn new() -> Self {
        let days30 = 30 * 24 * 3600u64;        Self {
            box_cache: TtlCache::new(
                env_ttl("BOX_CACHE_TTL_SECONDS", 600),
                env_max("BOX_CACHE_MAX_ITEMS", 64),
            ),
            skaters_players: TtlCache::new(
                env_ttl("SKATERS_PLAYERS_CACHE_TTL_SECONDS", 21600),
                env_max("SKATERS_PLAYERS_CACHE_MAX_ITEMS", 12),
            ),
            goalies_players: TtlCache::new(
                env_ttl("GOALIES_PLAYERS_CACHE_TTL_SECONDS", 21600),
                env_max("GOALIES_PLAYERS_CACHE_MAX_ITEMS", 12),
            ),
            team_seasons: TtlCache::new(env_ttl("SEASONS_CACHE_TTL_SECONDS", 3600), 64),
            player_landing: TtlCache::new(
                env_ttl("PLAYER_LANDING_CACHE_TTL_SECONDS", 3600),
                env_max("PLAYER_LANDING_CACHE_MAX_ITEMS", 512),
            ),
            team_logo: TtlCache::new(
                Duration::from_secs(days30),
                env_max("TEAM_LOGO_PROXY_CACHE_MAX_ITEMS", 64),
            ),
            player_headshot: TtlCache::new(
                Duration::from_secs(days30),
                env_max("PLAYER_HEADSHOT_PROXY_CACHE_MAX_ITEMS", 256),
            ),
            lineups_all: TtlCache::new_weighted(
                env_ttl("LINEUPS_SHEET_CACHE_TTL_SECONDS", 300),
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            player_projections: TtlCache::new_weighted(
                env_ttl("PLAYER_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            odds_snapshot_rows: TtlCache::new(
                env_ttl("ODDS_SNAPSHOTS_CACHE_TTL_SECONDS", 60),
                64,
            ),
            skater_bios: TtlCache::new(
                env_ttl("SKATER_BIOS_CACHE_TTL_SECONDS", 21600),
                env_max("SKATER_BIOS_CACHE_MAX_ITEMS", 4),
            ),
            goalie_bios: TtlCache::new(
                env_ttl("SKATER_BIOS_CACHE_TTL_SECONDS", 21600),
                env_max("SKATER_BIOS_CACHE_MAX_ITEMS", 4),
            ),
            all_rosters: TtlCache::new_weighted(
                env_ttl("ALL_ROSTERS_CACHE_TTL_SECONDS", 21600),
                16 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            teamseasonstats: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_CACHE_TTL_SECONDS", 1800),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            club_schedule: TtlCache::new(
                env_ttl("CLUB_SCHEDULE_CACHE_TTL_SECONDS", 21600),
                64,
            ),
            seasonstats_agg: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                32 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalies_agg: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                32 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            career_matrix: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                32 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            rapm_static: TtlCache::new_weighted(
                env_ttl("RAPM_STATIC_CACHE_TTL_SECONDS", 600),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            context_static: TtlCache::new_weighted(
                env_ttl("CONTEXT_STATIC_CACHE_TTL_SECONDS", 600),
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            card_metrics_defs: TtlCache::new(
                env_ttl("CARD_METRICS_DEF_CACHE_TTL_SECONDS", 600),
                8,
            ),
            team_stats_rest: TtlCache::new_weighted(
                env_ttl("TEAM_STATS_REST_CACHE_TTL_SECONDS", 3600),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            edge_api: TtlCache::new_weighted(
                env_ttl("EDGE_API_CACHE_TTL_SECONDS", 3600),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            skaters_scatter: TtlCache::new_weighted(
                env_ttl("SKATERS_SCATTER_CACHE_TTL_SECONDS", 300),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalies_scatter: TtlCache::new_weighted(
                env_ttl("GOALIES_SCATTER_CACHE_TTL_SECONDS", 180),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalie_team_by_season: TtlCache::new(
                env_ttl("GOALIES_TEAM_BY_SEASON_CACHE_TTL_SECONDS", 7 * 24 * 3600),
                128,
            ),
            // The line-tool datasets are huge (a full team season of shifts +
            // PBP can exceed 100MB working set each). Keep ONLY the most recent
            // recent team with a deterministic single slot (moka's lazy
            // capacity eviction lets multiple giants coexist, so it can't be
            // used here) — browsing team-to-team can't accumulate memory.
            lt_shifts: SingleSlot::new(),
            lt_pbp: SingleSlot::new(),
            lt_data: SingleSlot::new(),
            lt_base: SingleSlot::new(),
            skaters_shooting: TtlCache::new_weighted(
                env_ttl("SKATERS_SHOOTING_CACHE_TTL_SECONDS", 180),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalies_goaltending: TtlCache::new_weighted(
                env_ttl("GOALIES_GOALTENDING_CACHE_TTL_SECONDS", 180),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            player_names: TtlCache::new(std::time::Duration::from_secs(21600), 8),
            pbp: TtlCache::new_weighted(
                env_ttl("PBP_CACHE_TTL_SECONDS", 600),
                32 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            shifts: TtlCache::new_weighted(
                env_ttl("SHIFTS_CACHE_TTL_SECONDS", 600),
                32 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            model: TtlCache::new(std::time::Duration::from_secs(7 * 24 * 3600), 24),
            box_id_map: TtlCache::new(std::time::Duration::from_secs(7 * 24 * 3600), 2),
            all_rosters_by_season: TtlCache::new_weighted(
                env_ttl("ALL_ROSTERS_BY_SEASON_CACHE_TTL_SECONDS", 21600),
                16 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            v2_build: TtlCache::new_weighted(
                env_ttl("V2_PROJECTIONS_BUILD_CACHE_TTL_SECONDS", 300),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            current_player_projections: TtlCache::new_weighted(
                env_ttl("PLAYER_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            gm_projections: TtlCache::new_weighted(
                env_ttl("GM_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            custom_lineups: TtlCache::new_weighted(
                env_ttl("CUSTOM_LINEUPS_CACHE_TTL_SECONDS", 43200),
                24 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            playoff_bracket: TtlCache::new_weighted(
                env_ttl("PLAYOFF_BRACKET_CACHE_TTL_SECONDS", 300),
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            community_feed: TtlCache::new(
                env_ttl("COMMUNITY_FEED_CACHE_TTL_SECONDS", 300),
                8,
            ),
        }
    }

    pub fn pbp_ttl_secs(&self) -> u64 {
        env_ttl("PBP_CACHE_TTL_SECONDS", 600).as_secs()
    }

    pub fn shifts_ttl_secs(&self) -> u64 {
        env_ttl("SHIFTS_CACHE_TTL_SECONDS", 600).as_secs()
    }
}

#[derive(Clone)]
pub struct AppState {
    pub cfg: Arc<Config>,
    pub http: reqwest::Client,
    pub templates: TemplateEnv,
    pub sb: Option<SbClient>,
    pub caches: Arc<Caches>,
    pub teams: Arc<Vec<Value>>,
    pub about: Arc<AboutData>,
    /// Season → last date (Supabase `last_dates` table), loaded once.
    pub last_dates: Arc<BTreeMap<i64, String>>,
    /// Admin background jobs (e.g. lineup + GP refresh) keyed by job id.
    pub jobs: Arc<std::sync::Mutex<HashMap<String, serde_json::Value>>>,
}

impl AppState {
    pub async fn build(cfg: Config) -> Result<Self, ApiError> {
        let http = reqwest::Client::builder()
            .user_agent("hockey-statistics-nhl-rust/0.1")
            .build()
            .map_err(|e| ApiError::Internal(format!("failed to build HTTP client: {e}")))?;

        let sb = SbClient::from_env(http.clone());
        let templates = TemplateEnv::new(&cfg)?;
        let teams = Arc::new(teams::load(sb.as_ref(), &cfg).await);
        let last_dates = Arc::new(crate::data::last_dates::load(sb.as_ref(), &cfg).await);
        let about = Arc::new(AboutData::load(&cfg.about_data_json).unwrap_or_else(|e| {
            tracing::warn!("about data unavailable ({e}); serving empty about page");
            AboutData::empty()
        }));

        Ok(Self {
            cfg: Arc::new(cfg),
            http,
            templates,
            sb,
            caches: Arc::new(Caches::new()),
            teams,
            about,
            last_dates,
            jobs: Arc::new(std::sync::Mutex::new(HashMap::new())),
        })
    }
}
