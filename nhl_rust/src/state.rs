//! Shared application state (Axum `State`) plus the module-level cache
//! registry mirroring `app/routes.py` (see PORT_PLAN.md §7 for TTL parity).

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use std::time::Duration;

use serde_json::Value;

use crate::cache::{env_max, env_ttl, InFlight, SingleSlot, TtlCache};
use crate::config::Config;
use crate::data::about::AboutData;
use crate::data::rapm::RapmRows;
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
    pub team_logo: TtlCache<String, Arc<Vec<u8>>>,
    pub player_headshot: TtlCache<String, Arc<Vec<u8>>>,
    pub lineups_all: TtlCache<(), Value>,
    pub player_projections: TtlCache<(), Value>,
    pub odds_snapshot_rows: TtlCache<String, Value>,
    pub skater_bios: TtlCache<i64, Value>,
    pub goalie_bios: TtlCache<i64, Value>,
    pub all_rosters: TtlCache<(), Value>,
    pub teamseasonstats: TtlCache<(), Value>,
    pub team_seasonstats_filtered: TtlCache<String, Arc<Vec<Value>>>,
    pub club_schedule: TtlCache<(String, i64), Value>,
    pub seasonstats_agg: TtlCache<String, Arc<(Value, Value)>>,
    pub goalies_agg: TtlCache<String, Arc<(Value, Value)>>,
    pub career_matrix: TtlCache<String, Arc<(Value, Value)>>,
    pub goalies_league_baseline: TtlCache<String, Arc<HashMap<i64, (f64, f64)>>>,
    pub rapm_static: TtlCache<i64, std::sync::Arc<RapmRows>>,
    pub context_static: TtlCache<i64, std::sync::Arc<Vec<Value>>>,
    pub card_metrics_defs: TtlCache<String, Value>,
    pub team_stats_rest: TtlCache<String, Value>,
    pub edge_api: TtlCache<String, Value>,
    pub skaters_scatter: TtlCache<String, Value>,
    pub goalies_scatter: TtlCache<String, Value>,
    pub goalie_team_by_season: TtlCache<(i64, String), Value>,
    pub skater_team_by_season: TtlCache<i64, Value>,
    pub lt_shifts: SingleSlot<String, Arc<Vec<Value>>>,
    pub lt_pbp: SingleSlot<String, Arc<Vec<Value>>>,
    pub lt_data: SingleSlot<String, Arc<Value>>,
    pub lt_base: SingleSlot<String, Arc<Value>>,
    pub skaters_shooting: TtlCache<String, Value>,
    pub goalies_goaltending: TtlCache<String, Value>,
    pub player_names: TtlCache<i64, Arc<BTreeMap<i64, String>>>,
    pub pbp: TtlCache<(i64, String), Value>,
    pub shifts: TtlCache<i64, Value>,
    pub model: TtlCache<String, std::sync::Arc<crate::models::xg::XgModel>>,
    pub box_id_map: TtlCache<(), Arc<crate::data::boxids::BoxIdMap>>,
    pub all_rosters_by_season: TtlCache<i64, Value>,
    pub v2_build: TtlCache<String, Value>,
    pub latest_v2_season: TtlCache<(), i64>,
    pub player_current_proj_public: TtlCache<String, Arc<Value>>,
    pub current_player_projections: TtlCache<(), Value>,
    pub gm_projections: TtlCache<(), Value>,
    pub custom_lineups: TtlCache<(String, i64), Value>,
    pub playoff_bracket: TtlCache<i64, Value>,
    pub community_feed: TtlCache<String, Value>,
    /// Pre-parsed TTLs (avoids `std::env::var` per PBP/shifts request).
    pub pbp_ttl_secs_val: u64,
    pub shifts_ttl_secs_val: u64,
    /// Singleflight coalescing for the expensive shared loads (season-stats
    /// aggregates, RAPM/context) so concurrent users can't multiply memory.
    pub inflight: InFlight,
    /// Serializes the heavy line-tool loads (a full team-season of shifts+BPP
    /// is ~100MB+ working set each). Without a cap, a few concurrent
    /// `/api/line-tool/*` requests (a user toggling teams/seasons) load many
    /// giant datasets simultaneously and blow past the memory limit.
    pub lt_gate: Arc<tokio::sync::Semaphore>,
}

impl Caches {
    pub fn new() -> Self {
        let days30 = 30 * 24 * 3600u64;
        let pbp_ttl = env_ttl("PBP_CACHE_TTL_SECONDS", 600);
        let shifts_ttl = env_ttl("SHIFTS_CACHE_TTL_SECONDS", 600);
        let lt_ttl = env_ttl("LINE_TOOL_CACHE_TTL_SECONDS", 1800);
        Self {
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
                6 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            player_projections: TtlCache::new_weighted(
                env_ttl("PLAYER_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                6 * 1024 * 1024,
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
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            teamseasonstats: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_CACHE_TTL_SECONDS", 1800),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            team_seasonstats_filtered: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_CACHE_TTL_SECONDS", 1800),
                12 * 1024 * 1024,
                |v: &Arc<Vec<Value>>| crate::cache::json_rows_weight(v),
            ),
            club_schedule: TtlCache::new(
                env_ttl("CLUB_SCHEDULE_CACHE_TTL_SECONDS", 21600),
                64,
            ),
            seasonstats_agg: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                16 * 1024 * 1024,
                |v: &Arc<(Value, Value)>| {
                    crate::cache::json_value_weight(&v.0)
                        .saturating_add(crate::cache::json_value_weight(&v.1))
                },
            ),
            goalies_agg: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                16 * 1024 * 1024,
                |v: &Arc<(Value, Value)>| {
                    crate::cache::json_value_weight(&v.0)
                        .saturating_add(crate::cache::json_value_weight(&v.1))
                },
            ),
            career_matrix: TtlCache::new_weighted(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                16 * 1024 * 1024,
                |v: &Arc<(Value, Value)>| {
                    crate::cache::json_value_weight(&v.0)
                        .saturating_add(crate::cache::json_value_weight(&v.1))
                },
            ),
            goalies_league_baseline: TtlCache::new(
                env_ttl("SEASONSTATS_AGG_CACHE_TTL_SECONDS", 1800),
                32,
            ),
            // RAPM/context rows are cached PER SEASON (the current season is
            // ~1.4k rows, not the full 21k) and shared via Arc so a cache hit
            // is a pointer clone. Total budget covers ~all seasons combined;
            // warm caches only grow when career-scope endpoints are used.
            rapm_static: TtlCache::new_weighted(
                env_ttl("RAPM_STATIC_CACHE_TTL_SECONDS", 600),
                24 * 1024 * 1024,
                |v: &std::sync::Arc<RapmRows>| {
                    crate::cache::json_rows_weight(&v.totals)
                        .saturating_add(crate::cache::json_rows_weight(&v.rates))
                },
            ),
            context_static: TtlCache::new_weighted(
                env_ttl("CONTEXT_STATIC_CACHE_TTL_SECONDS", 600),
                8 * 1024 * 1024,
                |v: &std::sync::Arc<Vec<Value>>| crate::cache::json_rows_weight(v),
            ),
            card_metrics_defs: TtlCache::new(
                env_ttl("CARD_METRICS_DEF_CACHE_TTL_SECONDS", 600),
                8,
            ),
            team_stats_rest: TtlCache::new_weighted(
                env_ttl("TEAM_STATS_REST_CACHE_TTL_SECONDS", 3600),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            edge_api: TtlCache::new_weighted(
                env_ttl("EDGE_API_CACHE_TTL_SECONDS", 3600),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            skaters_scatter: TtlCache::new_weighted(
                env_ttl("SKATERS_SCATTER_CACHE_TTL_SECONDS", 300),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalies_scatter: TtlCache::new_weighted(
                env_ttl("GOALIES_SCATTER_CACHE_TTL_SECONDS", 180),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalie_team_by_season: TtlCache::new(
                env_ttl("GOALIES_TEAM_BY_SEASON_CACHE_TTL_SECONDS", 7 * 24 * 3600),
                128,
            ),
            skater_team_by_season: TtlCache::new(
                env_ttl("SKATERS_TEAM_BY_SEASON_CACHE_TTL_SECONDS", 7 * 24 * 3600),
                128,
            ),
            // The line-tool datasets are huge (a full team season of shifts +
            // PBP can exceed 100MB working set each). Keep ONLY the most recent
            // entry with a deterministic single slot wrapped in Arc + TTL so
            // hits are an O(1) ref-count bump rather than a 100MB clone under a mutex.
            lt_shifts: SingleSlot::new_with_ttl(lt_ttl),
            lt_pbp: SingleSlot::new_with_ttl(lt_ttl),
            lt_data: SingleSlot::new_with_ttl(lt_ttl),
            lt_base: SingleSlot::new_with_ttl(lt_ttl),
            skaters_shooting: TtlCache::new_weighted(
                env_ttl("SKATERS_SHOOTING_CACHE_TTL_SECONDS", 180),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            goalies_goaltending: TtlCache::new_weighted(
                env_ttl("GOALIES_GOALTENDING_CACHE_TTL_SECONDS", 180),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            player_names: TtlCache::new(Duration::from_secs(21600), 32),
            pbp: TtlCache::new_weighted(
                pbp_ttl,
                16 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            shifts: TtlCache::new_weighted(
                shifts_ttl,
                16 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            model: TtlCache::new(Duration::from_secs(7 * 24 * 3600), 24),
            box_id_map: TtlCache::new(Duration::from_secs(7 * 24 * 3600), 2),
            all_rosters_by_season: TtlCache::new_weighted(
                env_ttl("ALL_ROSTERS_BY_SEASON_CACHE_TTL_SECONDS", 21600),
                8 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            v2_build: TtlCache::new_weighted(
                env_ttl("V2_PROJECTIONS_BUILD_CACHE_TTL_SECONDS", 300),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            latest_v2_season: TtlCache::new(
                env_ttl("V2_PROJECTIONS_BUILD_CACHE_TTL_SECONDS", 600),
                2,
            ),
            player_current_proj_public: TtlCache::new_weighted(
                env_ttl("PLAYER_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                8 * 1024 * 1024,
                |v: &Arc<Value>| crate::cache::json_value_weight(v),
            ),
            current_player_projections: TtlCache::new_weighted(
                env_ttl("PLAYER_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            gm_projections: TtlCache::new_weighted(
                env_ttl("GM_PROJECTIONS_CACHE_TTL_SECONDS", 300),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            custom_lineups: TtlCache::new_weighted(
                env_ttl("CUSTOM_LINEUPS_CACHE_TTL_SECONDS", 43200),
                12 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            playoff_bracket: TtlCache::new_weighted(
                env_ttl("PLAYOFF_BRACKET_CACHE_TTL_SECONDS", 300),
                6 * 1024 * 1024,
                |v: &Value| crate::cache::json_value_weight(v),
            ),
            community_feed: TtlCache::new(
                env_ttl("COMMUNITY_FEED_CACHE_TTL_SECONDS", 300),
                8,
            ),
            pbp_ttl_secs_val: pbp_ttl.as_secs(),
            shifts_ttl_secs_val: shifts_ttl.as_secs(),
            inflight: InFlight::new(),
            lt_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        }
    }

    pub fn pbp_ttl_secs(&self) -> u64 {
        self.pbp_ttl_secs_val
    }

    pub fn shifts_ttl_secs(&self) -> u64 {
        self.shifts_ttl_secs_val
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
            .connect_timeout(Duration::from_secs(10))
            .timeout(Duration::from_secs(30))
            .pool_idle_timeout(Duration::from_secs(90))
            .tcp_keepalive(Duration::from_secs(60))
            .build()
            .map_err(|e| ApiError::Internal(format!("failed to build HTTP client: {e}")))?;

        let sb = SbClient::from_env(http.clone());
        let templates = TemplateEnv::new(&cfg)?;
        let (teams_vec, last_dates_map) = tokio::join!(
            teams::load(sb.as_ref(), &cfg),
            crate::data::last_dates::load(sb.as_ref(), &cfg),
        );
        let teams = Arc::new(teams_vec);
        let last_dates = Arc::new(last_dates_map);
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
