//! Public, user-friendly NHL Contracts REST API (`/api/v1/contracts/*`).
//!
//! - Open access (no API key required) with CORS enabled for browser clients.
//! - Lightweight per-IP fixed-window rate limiter (`CONTRACTS_API_RATE_LIMIT_PER_MIN`,
//!   default 60 requests/minute) with standard `X-RateLimit-*` headers.
//! - Accepts flexible season inputs (`20252026`, `2025-26`, `2025/26`, `2025`,
//!   or comma-separated lists like `2024-25,2025-26`).
//! - Primary data source: NHLPA compensation data, supplemented by official
//!   NHL/club announcements and public cap-penalty reporting.

use std::collections::{BTreeSet, HashMap};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use moka::sync::Cache;
use serde_json::{json, Value};

use crate::data::contracts::{
    load_team_contracts, summarize_player_contract,
};
use crate::state::AppState;
use crate::supabase::read::{filters, SbClient};
use crate::util::dates::current_season_id;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

const DEFAULT_RATE_LIMIT_PER_MIN: u32 = 60;
const RATE_WINDOW_SECS: u64 = 60;
const DEFAULT_LIMIT: usize = 50;
const MAX_LIMIT: usize = 200;

const CONTRACTS_PUBLIC_COLS: &str = "contract_id,player_id,player_name,position,birth_date,draft_year,signing_team,teams,start_year,end_year,start_season,end_season,term_years,signed_date,contract_status,contract_status_note,expiry_status,expiry_status_note,aav_usd,total_value_usd,two_way,is_active_2026_27,is_future_extension,source_url,nhlpa_url,updated_at";

const CONTRACT_YEARS_PUBLIC_COLS: &str = "contract_id,player_id,player_name,position,season_year,season_id,season,team,aav_usd,total_salary_usd,base_salary_usd,signing_bonus_usd,minor_league_salary_usd,two_way,is_projected";

const TEAM_YEARS_PUBLIC_COLS: &str = "team,season_year,season_id,season,is_projected,contracts_count,one_way_contracts_count,two_way_contracts_count,total_aav_usd,one_way_aav_usd,total_salary_usd,total_base_salary_usd,total_signing_bonus_usd,buyout_penalty_usd,retained_salary_penalty_usd,bonus_overage_penalty_usd,total_cap_penalties_usd,combined_aav_plus_penalties_usd,combined_one_way_plus_penalties_usd,cap_ceiling_usd,cap_floor_usd,projected_cap_space_usd,penalties_detail,updated_at";

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/v1/contracts", get(api_contracts_index).options(cors_preflight))
        .route(
            "/api/v1/contracts/players",
            get(api_contracts_players).options(cors_preflight),
        )
        .route(
            "/api/v1/contracts/players/{player_id}",
            get(api_contracts_player_detail).options(cors_preflight),
        )
        .route(
            "/api/v1/contracts/teams",
            get(api_contracts_teams).options(cors_preflight),
        )
        .route(
            "/api/v1/contracts/teams/{team}",
            get(api_contracts_team_detail).options(cors_preflight),
        )
        .route(
            "/api/v1/contracts/expiring",
            get(api_contracts_expiring).options(cors_preflight),
        )
        .route(
            "/api/v1/contracts/penalties",
            get(api_contracts_penalties).options(cors_preflight),
        )
}

// ── Per-IP Fixed-Window Rate Limiter ────────────────────────────────────────

struct RateWindow {
    started_at: Instant,
    reset_epoch: u64,
    count: u32,
}

#[derive(Clone, Copy, Debug)]
struct RateDecision {
    allowed: bool,
    limit: u32,
    remaining: u32,
    reset_epoch: u64,
    retry_after_secs: u64,
}

struct IpRateLimiter {
    limit: u32,
    window: Duration,
    buckets: Cache<String, Arc<Mutex<RateWindow>>>,
}

impl IpRateLimiter {
    fn new(limit: u32, window_secs: u64) -> Self {
        Self {
            limit,
            window: Duration::from_secs(window_secs.max(1)),
            buckets: Cache::builder()
                .max_capacity(10_000)
                .time_to_idle(Duration::from_secs((window_secs * 2).max(120)))
                .build(),
        }
    }

    fn check(&self, key: &str) -> RateDecision {
        let now_epoch = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        if self.limit == 0 {
            return RateDecision {
                allowed: true,
                limit: 0,
                remaining: 0,
                reset_epoch: now_epoch,
                retry_after_secs: 0,
            };
        }

        let now = Instant::now();
        let entry = self.buckets.get_with(key.to_string(), || {
            Arc::new(Mutex::new(RateWindow {
                started_at: now,
                reset_epoch: now_epoch + self.window.as_secs(),
                count: 0,
            }))
        });

        let mut guard = entry.lock().unwrap_or_else(|p| p.into_inner());
        let elapsed = now.saturating_duration_since(guard.started_at);
        if elapsed >= self.window {
            guard.started_at = now;
            guard.reset_epoch = now_epoch + self.window.as_secs();
            guard.count = 0;
        }

        if guard.count < self.limit {
            guard.count += 1;
            RateDecision {
                allowed: true,
                limit: self.limit,
                remaining: self.limit.saturating_sub(guard.count),
                reset_epoch: guard.reset_epoch,
                retry_after_secs: 0,
            }
        } else {
            let retry_after = self
                .window
                .saturating_sub(elapsed)
                .as_secs()
                .max(1);
            RateDecision {
                allowed: false,
                limit: self.limit,
                remaining: 0,
                reset_epoch: guard.reset_epoch,
                retry_after_secs: retry_after,
            }
        }
    }
}

pub fn rate_limit_per_min() -> u32 {
    static LIMIT: OnceLock<u32> = OnceLock::new();
    *LIMIT.get_or_init(|| {
        std::env::var("CONTRACTS_API_RATE_LIMIT_PER_MIN")
            .ok()
            .and_then(|v| v.trim().parse::<u32>().ok())
            .unwrap_or(DEFAULT_RATE_LIMIT_PER_MIN)
    })
}

fn global_limiter() -> &'static IpRateLimiter {
    static LIMITER: OnceLock<IpRateLimiter> = OnceLock::new();
    LIMITER.get_or_init(|| IpRateLimiter::new(rate_limit_per_min(), RATE_WINDOW_SECS))
}

fn client_ip_key(headers: &HeaderMap) -> String {
    if let Some(xff) = headers.get("x-forwarded-for").and_then(|v| v.to_str().ok()) {
        if let Some(first) = xff.split(',').next().map(str::trim).filter(|s| !s.is_empty()) {
            return first.to_string();
        }
    }
    if let Some(xri) = headers.get("x-real-ip").and_then(|v| v.to_str().ok()) {
        let trimmed = xri.trim();
        if !trimmed.is_empty() {
            return trimmed.to_string();
        }
    }
    "unknown".to_string()
}

fn apply_public_headers(resp: &mut Response, decision: Option<RateDecision>, cache_secs: u64) {
    let h = resp.headers_mut();
    h.insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, HeaderValue::from_static("*"));
    h.insert(
        header::ACCESS_CONTROL_ALLOW_METHODS,
        HeaderValue::from_static("GET, OPTIONS"),
    );
    h.insert(
        header::ACCESS_CONTROL_ALLOW_HEADERS,
        HeaderValue::from_static("Content-Type, Accept"),
    );
    if cache_secs > 0 {
        if let Ok(v) = HeaderValue::from_str(&format!("public, max-age={cache_secs}")) {
            h.insert(header::CACHE_CONTROL, v);
        }
    } else {
        h.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    }
    if let Some(d) = decision {
        if d.limit > 0 {
            if let Ok(v) = HeaderValue::from_str(&d.limit.to_string()) {
                h.insert("x-ratelimit-limit", v);
            }
            if let Ok(v) = HeaderValue::from_str(&d.remaining.to_string()) {
                h.insert("x-ratelimit-remaining", v);
            }
            if let Ok(v) = HeaderValue::from_str(&d.reset_epoch.to_string()) {
                h.insert("x-ratelimit-reset", v);
            }
            if !d.allowed {
                if let Ok(v) = HeaderValue::from_str(&d.retry_after_secs.to_string()) {
                    h.insert(header::RETRY_AFTER, v);
                }
            }
        }
    }
}

fn public_json(status: StatusCode, body: Value, decision: RateDecision, cache_secs: u64) -> Response {
    let mut resp = (status, Json(body)).into_response();
    apply_public_headers(&mut resp, Some(decision), cache_secs);
    resp
}

fn rate_limited_response(decision: RateDecision) -> Response {
    public_json(
        StatusCode::TOO_MANY_REQUESTS,
        json!({
            "error": "rate_limit_exceeded",
            "message": format!(
                "Rate limit exceeded ({} requests per minute). Please retry after {} seconds.",
                decision.limit, decision.retry_after_secs
            ),
            "limit": decision.limit,
            "retryAfterSeconds": decision.retry_after_secs,
        }),
        decision,
        0,
    )
}

async fn cors_preflight() -> Response {
    let mut resp = StatusCode::NO_CONTENT.into_response();
    apply_public_headers(&mut resp, None, 86400);
    resp
}

fn check_rate_limit(headers: &HeaderMap) -> Result<RateDecision, Response> {
    let key = client_ip_key(headers);
    let decision = global_limiter().check(&key);
    if !decision.allowed {
        return Err(rate_limited_response(decision));
    }
    Ok(decision)
}

// ── Flexible Parameter Parsing ──────────────────────────────────────────────

/// Parses a single season token in any common NHL format into an 8-digit
/// `season_id` (e.g., `20252026`, `2025-26`, `2025/26`, `2025-2026`, `2025`).
pub fn parse_flexible_season(token: &str) -> Option<i64> {
    let s = token.trim();
    if s.is_empty() {
        return None;
    }
    if s.len() == 8 && s.bytes().all(|b| b.is_ascii_digit()) {
        let start: i64 = s[0..4].parse().ok()?;
        let end: i64 = s[4..8].parse().ok()?;
        if (1917..=2099).contains(&start) && end == start + 1 {
            return Some(start * 10000 + end);
        }
        return None;
    }
    if s.len() == 4 && s.bytes().all(|b| b.is_ascii_digit()) {
        let start: i64 = s.parse().ok()?;
        if (1917..=2099).contains(&start) {
            return Some(start * 10000 + (start + 1));
        }
        return None;
    }
    let sep = if s.contains('-') {
        '-'
    } else if s.contains('/') {
        '/'
    } else {
        return None;
    };
    let (left, right) = s.split_once(sep)?;
    if left.len() != 4 || !left.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    let start: i64 = left.parse().ok()?;
    if !(1917..=2099).contains(&start) {
        return None;
    }
    let expected_end = start + 1;
    if right.len() == 4 && right.bytes().all(|b| b.is_ascii_digit()) {
        let end: i64 = right.parse().ok()?;
        if end == expected_end {
            return Some(start * 10000 + end);
        }
    } else if right.len() == 2 && right.bytes().all(|b| b.is_ascii_digit()) {
        let end2: i64 = right.parse().ok()?;
        if end2 == expected_end % 100 {
            return Some(start * 10000 + expected_end);
        }
    }
    None
}

/// Parses comma-separated flexible season values into sorted, deduplicated 8-digit season IDs.
pub fn parse_flexible_season_list(raw: &str) -> Vec<i64> {
    let mut out: Vec<i64> = raw
        .split(',')
        .filter_map(parse_flexible_season)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn parse_bool_param(params: &HashMap<String, String>, key: &str) -> Option<bool> {
    let raw = params.get(key)?.trim().to_lowercase();
    match raw.as_str() {
        "true" | "1" | "yes" => Some(true),
        "false" | "0" | "no" => Some(false),
        _ => None,
    }
}

fn parse_usize_param(params: &HashMap<String, String>, key: &str, default: usize, max: usize) -> usize {
    params
        .get(key)
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map(|v| v.min(max))
        .unwrap_or(default)
}

fn parse_f64_param(params: &HashMap<String, String>, key: &str) -> Option<f64> {
    params
        .get(key)
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|v| v.is_finite())
}

fn sanitize_team_code(raw: &str) -> Option<String> {
    let s: String = raw
        .trim()
        .to_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .collect();
    if (2..=4).contains(&s.len()) {
        Some(s)
    } else {
        None
    }
}

fn sanitize_search_query(raw: &str) -> Option<String> {
    let cleaned: String = raw
        .trim()
        .chars()
        .filter(|c| c.is_alphanumeric() || matches!(c, ' ' | '-' | '\'' | '.'))
        .take(64)
        .collect();
    let trimmed = cleaned.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

fn opt_f64(v: Option<&Value>) -> Option<f64> {
    let val = v?;
    if val.is_null() {
        return None;
    }
    parse_locale_float(Some(val))
}

fn opt_bool(v: Option<&Value>) -> bool {
    match v {
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_i64().map(|x| x != 0).unwrap_or(false),
        _ => false,
    }
}

fn opt_str(v: Option<&Value>) -> Option<String> {
    let s = str_value(v);
    if s.is_empty() { None } else { Some(s) }
}

fn format_contract_row(r: &Value) -> Value {
    let end_year = safe_int(r.get("end_year"));
    json!({
        "contractId": opt_str(r.get("contract_id")),
        "playerId": safe_int(r.get("player_id")),
        "playerName": opt_str(r.get("player_name")),
        "position": opt_str(r.get("position")),
        "birthDate": opt_str(r.get("birth_date")),
        "draftYear": safe_int(r.get("draft_year")),
        "signingTeam": opt_str(r.get("signing_team")),
        "teams": r.get("teams").cloned().filter(|v| v.is_array()).unwrap_or_else(|| json!([])),
        "startSeason": opt_str(r.get("start_season")),
        "endSeason": opt_str(r.get("end_season")),
        "startYear": safe_int(r.get("start_year")),
        "endYear": end_year,
        "expiryYear": end_year.map(|y| y + 1),
        "termYears": safe_int(r.get("term_years")).filter(|&t| t > 0),
        "signedDate": opt_str(r.get("signed_date")),
        "contractStatus": opt_str(r.get("contract_status")),
        "contractStatusNote": opt_str(r.get("contract_status_note")),
        "expiryStatus": opt_str(r.get("expiry_status")),
        "expiryStatusNote": opt_str(r.get("expiry_status_note")),
        "aavUsd": opt_f64(r.get("aav_usd")),
        "totalValueUsd": opt_f64(r.get("total_value_usd")),
        "twoWay": opt_bool(r.get("two_way")),
        "isActiveCurrent": opt_bool(r.get("is_active_2026_27")),
        "isFutureExtension": opt_bool(r.get("is_future_extension")),
        "sourceUrl": opt_str(r.get("source_url")),
        "nhlpaUrl": opt_str(r.get("nhlpa_url")),
    })
}

fn format_contract_year_row(r: &Value, meta: Option<&Value>) -> Value {
    let end_year = meta.and_then(|m| safe_int(m.get("end_year")));
    json!({
        "contractId": opt_str(r.get("contract_id")),
        "playerId": safe_int(r.get("player_id")),
        "playerName": opt_str(r.get("player_name")),
        "position": opt_str(r.get("position")),
        "seasonId": safe_int(r.get("season_id")),
        "season": opt_str(r.get("season")),
        "seasonYear": safe_int(r.get("season_year")),
        "team": opt_str(r.get("team")),
        "aavUsd": opt_f64(r.get("aav_usd")),
        "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
        "baseSalaryUsd": opt_f64(r.get("base_salary_usd")),
        "signingBonusUsd": opt_f64(r.get("signing_bonus_usd")),
        "minorLeagueSalaryUsd": opt_f64(r.get("minor_league_salary_usd")),
        "twoWay": opt_bool(r.get("two_way")),
        "isProjected": opt_bool(r.get("is_projected")),
        "termYears": meta.and_then(|m| safe_int(m.get("term_years"))).filter(|&t| t > 0),
        "startSeason": meta.and_then(|m| opt_str(m.get("start_season"))),
        "endSeason": meta.and_then(|m| opt_str(m.get("end_season"))),
        "expiryYear": end_year.map(|y| y + 1),
        "contractStatus": meta.and_then(|m| opt_str(m.get("contract_status"))),
        "expiryStatus": meta.and_then(|m| opt_str(m.get("expiry_status"))),
    })
}

fn format_team_cap_row(r: &Value) -> Value {
    json!({
        "team": opt_str(r.get("team")),
        "seasonId": safe_int(r.get("season_id")),
        "season": opt_str(r.get("season")),
        "seasonYear": safe_int(r.get("season_year")),
        "isProjected": opt_bool(r.get("is_projected")),
        "contractsCount": safe_int(r.get("contracts_count")),
        "oneWayContractsCount": safe_int(r.get("one_way_contracts_count")),
        "twoWayContractsCount": safe_int(r.get("two_way_contracts_count")),
        "totalAavUsd": opt_f64(r.get("total_aav_usd")),
        "oneWayAavUsd": opt_f64(r.get("one_way_aav_usd")),
        "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
        "totalBaseSalaryUsd": opt_f64(r.get("total_base_salary_usd")),
        "totalSigningBonusUsd": opt_f64(r.get("total_signing_bonus_usd")),
        "buyoutPenaltyUsd": opt_f64(r.get("buyout_penalty_usd")),
        "retainedSalaryPenaltyUsd": opt_f64(r.get("retained_salary_penalty_usd")),
        "bonusOveragePenaltyUsd": opt_f64(r.get("bonus_overage_penalty_usd")),
        "totalCapPenaltiesUsd": opt_f64(r.get("total_cap_penalties_usd")),
        "combinedAavPlusPenaltiesUsd": opt_f64(r.get("combined_aav_plus_penalties_usd")),
        "combinedOneWayPlusPenaltiesUsd": opt_f64(r.get("combined_one_way_plus_penalties_usd")),
        "capCeilingUsd": opt_f64(r.get("cap_ceiling_usd")),
        "capFloorUsd": opt_f64(r.get("cap_floor_usd")),
        "projectedCapSpaceUsd": opt_f64(r.get("projected_cap_space_usd")),
        "penaltiesDetail": r.get("penalties_detail").cloned().filter(|v| v.is_array()),
    })
}

async fn fetch_contracts_map(sb: &SbClient, contract_ids: &[String]) -> HashMap<String, Value> {
    let unique: Vec<&str> = contract_ids
        .iter()
        .map(String::as_str)
        .filter(|s| !s.is_empty())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let mut out = HashMap::with_capacity(unique.len());
    for chunk in unique.chunks(60) {
        let in_list = chunk.join(",");
        let rows = sb
            .read(
                "contracts",
                CONTRACTS_PUBLIC_COLS,
                Some(&filters(&[("contract_id", &format!("in.({in_list})"))])),
                None,
                None,
                chunk.len(),
            )
            .await
            .unwrap_or_default();
        for row in rows {
            let cid = str_value(row.get("contract_id"));
            if !cid.is_empty() {
                out.insert(cid, row);
            }
        }
    }
    out
}

fn require_sb<'a>(state: &'a AppState, decision: RateDecision) -> Result<&'a SbClient, Response> {
    state.sb.as_ref().ok_or_else(|| {
        public_json(
            StatusCode::SERVICE_UNAVAILABLE,
            json!({
                "error": "service_unavailable",
                "message": "Contracts database is not configured.",
            }),
            decision,
            0,
        )
    })
}

// ── Route Handlers ──────────────────────────────────────────────────────────

/// `GET /api/v1/contracts` — API discovery & metadata.
async fn api_contracts_index(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let host = headers.get(header::HOST).and_then(|v| v.to_str().ok());
    let base = crate::web::templates::url_root(&state.cfg, host)
        .trim_end_matches('/')
        .to_string();
    let current = current_season_id(None);

    public_json(
        StatusCode::OK,
        json!({
            "name": "Hockey-Statistics NHL Contracts API",
            "version": "v1",
            "docsUrl": format!("{base}/about/contracts-api"),
            "currentSeasonId": current,
            "coverage": {
                "fromSeason": "2010-11",
                "toSeason": "2034-35 (includes projected future contract years)",
            },
            "rateLimit": {
                "requestsPerMinutePerIp": rate_limit_per_min(),
                "headers": ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"],
            },
            "sources": {
                "primary": "NHLPA (https://www.nhlpa.com/compensation-by-team)",
                "supplemental": [
                    "Official NHL.com & club contract announcements",
                    "Public hockey reporting (ProHockeyRumors, Sportsnet, TSN, Daily Faceoff) for cap penalties and recent signings"
                ],
            },
            "endpoints": [
                {
                    "method": "GET",
                    "path": "/api/v1/contracts",
                    "description": "API metadata, data sources, rate limits, and endpoint directory."
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/players",
                    "description": "Search and filter player contracts by name, team, season, position, status, two-way flag, or AAV range.",
                    "example": format!("{base}/api/v1/contracts/players?team=ANA&season=2025-26&limit=25")
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/players/{player_id}",
                    "description": "Full contract history for a player by NHL player ID, or season-specific / multi-season averaged summary when ?season is provided.",
                    "example": format!("{base}/api/v1/contracts/players/8478402?season=2025-26")
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/teams",
                    "description": "League-wide team salary cap overview for a season.",
                    "example": format!("{base}/api/v1/contracts/teams?season=2025-26")
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/teams/{team}",
                    "description": "Single team cap summary, full roster contracts, cap penalties, and multi-year cap outlook.",
                    "example": format!("{base}/api/v1/contracts/teams/ANA?season=2025-26")
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/expiring",
                    "description": "Upcoming expiring contracts (pending UFAs and RFAs) filterable by expiry year, status, team, position, and AAV.",
                    "example": format!("{base}/api/v1/contracts/expiring?year=2026&expiry_status=UFA&limit=25")
                },
                {
                    "method": "GET",
                    "path": "/api/v1/contracts/penalties",
                    "description": "Team dead cap penalties (buyouts, retained salary, bonus overages) for a season.",
                    "example": format!("{base}/api/v1/contracts/penalties?season=2026-27")
                }
            ]
        }),
        decision,
        300,
    )
}

/// `GET /api/v1/contracts/players` — search & filter player contracts.
async fn api_contracts_players(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let sb = match require_sb(&state, decision) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let q_raw = params.get("q").or_else(|| params.get("search")).map(String::as_str);
    let q_clean = q_raw.and_then(sanitize_search_query);
    let team = params.get("team").and_then(|t| sanitize_team_code(t));
    let season_raw = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = parse_flexible_season_list(season_raw);
    if !season_raw.trim().is_empty() && season_ids.is_empty() {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({
                "error": "invalid_season",
                "message": "Invalid season format. Use 20252026, 2025-26, or comma-separated values.",
            }),
            decision,
            0,
        );
    }

    let pos = params
        .get("position")
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let status = params
        .get("status")
        .or_else(|| params.get("contract_status"))
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let expiry_status = params
        .get("expiry_status")
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let two_way = parse_bool_param(&params, "two_way");
    let active = parse_bool_param(&params, "active");
    let min_aav = parse_f64_param(&params, "min_aav");
    let max_aav = parse_f64_param(&params, "max_aav");
    let sort = params
        .get("sort")
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_else(|| "aav_desc".to_string());
    let limit = parse_usize_param(&params, "limit", DEFAULT_LIMIT, MAX_LIMIT).max(1);
    let offset = parse_usize_param(&params, "offset", 0, 50_000);

    let cache_key = format!(
        "players|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{:?}|{}|{}|{}",
        q_clean, team, season_ids, pos, status, expiry_status, two_way, active, min_aav, max_aav, sort, limit, offset
    );
    if let Some(cached) = state.caches.contracts_api.get(&cache_key) {
        return public_json(StatusCode::OK, (*cached).clone(), decision, 300);
    }

    // Season-scoped search queries `contract_years` and joins `contracts` metadata.
    let payload: Value = if !season_ids.is_empty() {
        let mut f_pairs: Vec<(String, String)> = Vec::new();
        let sid_in = season_ids
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        f_pairs.push(("season_id".into(), format!("in.({sid_in})")));
        if let Some(ref t) = team {
            f_pairs.push(("team".into(), format!("eq.{t}")));
        }
        if let Some(ref qv) = q_clean {
            f_pairs.push(("player_name".into(), format!("ilike.*{qv}*")));
        }
        if let Some(ref p) = pos {
            if p == "F" {
                f_pairs.push(("position".into(), "in.(C,L,R,LW,RW)".into()));
            } else {
                f_pairs.push(("position".into(), format!("eq.{p}")));
            }
        }
        if let Some(tw) = two_way {
            f_pairs.push(("two_way".into(), format!("eq.{tw}")));
        }
        if let (Some(lo), Some(hi)) = (min_aav, max_aav) {
            f_pairs.push(("and".into(), format!("(aav_usd.gte.{},aav_usd.lte.{})", lo.round() as i64, hi.round() as i64)));
        } else if let Some(lo) = min_aav {
            f_pairs.push(("aav_usd".into(), format!("gte.{}", lo.round() as i64)));
        } else if let Some(hi) = max_aav {
            f_pairs.push(("aav_usd".into(), format!("lte.{}", hi.round() as i64)));
        }

        let order = match sort.as_str() {
            "aav_asc" => "aav_usd.asc.nullslast,player_name.asc",
            "name_asc" => "player_name.asc,season_id.desc",
            _ => "aav_usd.desc.nullslast,player_name.asc",
        };
        let f_map: std::collections::BTreeMap<String, String> = f_pairs.into_iter().collect();
        // If filtering by contract-table-only fields (status / expiry_status), fetch a wider slice before filtering.
        let need_post_filter = status.is_some() || expiry_status.is_some();
        let fetch_limit = if need_post_filter { 0 } else { offset + limit };

        let rows = sb
            .read(
                "contract_years",
                CONTRACT_YEARS_PUBLIC_COLS,
                Some(&f_map),
                None,
                Some(order),
                fetch_limit,
            )
            .await
            .unwrap_or_default();

        let cids: Vec<String> = rows
            .iter()
            .map(|r| str_value(r.get("contract_id")))
            .filter(|s| !s.is_empty())
            .collect();
        let meta_map = fetch_contracts_map(sb, &cids).await;

        let mut items: Vec<Value> = rows
            .iter()
            .filter_map(|r| {
                let cid = str_value(r.get("contract_id"));
                let meta = meta_map.get(&cid);
                if let Some(ref st) = status {
                    let actual = meta
                        .map(|m| str_value(m.get("contract_status")).to_uppercase())
                        .unwrap_or_default();
                    if &actual != st {
                        return None;
                    }
                }
                if let Some(ref es) = expiry_status {
                    let actual = meta
                        .map(|m| str_value(m.get("expiry_status")).to_uppercase())
                        .unwrap_or_default();
                    if &actual != es {
                        return None;
                    }
                }
                Some(format_contract_year_row(r, meta))
            })
            .collect();

        let page: Vec<Value> = items.drain(..).skip(offset).take(limit).collect();
        json!({
            "scope": "season_years",
            "seasons": season_ids,
            "count": page.len(),
            "limit": limit,
            "offset": offset,
            "data": page,
        })
    } else {
        // Contract-level search against `contracts`.
        let mut f_pairs: Vec<(String, String)> = Vec::new();
        if let Some(ref qv) = q_clean {
            f_pairs.push(("player_name".into(), format!("ilike.*{qv}*")));
        }
        if let Some(ref t) = team {
            f_pairs.push(("signing_team".into(), format!("eq.{t}")));
        }
        if let Some(ref p) = pos {
            if p == "F" {
                f_pairs.push(("position".into(), "in.(C,L,R,LW,RW)".into()));
            } else {
                f_pairs.push(("position".into(), format!("eq.{p}")));
            }
        }
        if let Some(ref st) = status {
            f_pairs.push(("contract_status".into(), format!("eq.{st}")));
        }
        if let Some(ref es) = expiry_status {
            f_pairs.push(("expiry_status".into(), format!("eq.{es}")));
        }
        if let Some(tw) = two_way {
            f_pairs.push(("two_way".into(), format!("eq.{tw}")));
        }
        if let Some(act) = active {
            f_pairs.push(("is_active_2026_27".into(), format!("eq.{act}")));
        }
        if let (Some(lo), Some(hi)) = (min_aav, max_aav) {
            f_pairs.push(("and".into(), format!("(aav_usd.gte.{},aav_usd.lte.{})", lo.round() as i64, hi.round() as i64)));
        } else if let Some(lo) = min_aav {
            f_pairs.push(("aav_usd".into(), format!("gte.{}", lo.round() as i64)));
        } else if let Some(hi) = max_aav {
            f_pairs.push(("aav_usd".into(), format!("lte.{}", hi.round() as i64)));
        }

        let order = match sort.as_str() {
            "aav_asc" => "aav_usd.asc.nullslast,player_name.asc",
            "name_asc" => "player_name.asc,start_year.desc",
            "end_year_asc" => "end_year.asc.nullslast,aav_usd.desc.nullslast",
            "end_year_desc" => "end_year.desc.nullslast,aav_usd.desc.nullslast",
            _ => "aav_usd.desc.nullslast,start_year.desc",
        };
        let f_map: std::collections::BTreeMap<String, String> = f_pairs.into_iter().collect();
        let rows = sb
            .read(
                "contracts",
                CONTRACTS_PUBLIC_COLS,
                if f_map.is_empty() { None } else { Some(&f_map) },
                None,
                Some(order),
                offset + limit,
            )
            .await
            .unwrap_or_default();

        let page: Vec<Value> = rows
            .iter()
            .skip(offset)
            .take(limit)
            .map(format_contract_row)
            .collect();
        json!({
            "scope": "contracts",
            "count": page.len(),
            "limit": limit,
            "offset": offset,
            "data": page,
        })
    };

    let arc = Arc::new(payload);
    state.caches.contracts_api.insert(cache_key, arc.clone());
    public_json(StatusCode::OK, (*arc).clone(), decision, 300)
}

/// `GET /api/v1/contracts/players/{player_id}` — single player contract history
/// or season-filtered / multi-season averaged summary.
async fn api_contracts_player_detail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(player_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    if player_id <= 0 {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({ "error": "invalid_player_id", "message": "player_id must be a positive integer." }),
            decision,
            0,
        );
    }
    let sb = match require_sb(&state, decision) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let season_raw = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = parse_flexible_season_list(season_raw);
    if !season_raw.trim().is_empty() && season_ids.is_empty() {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({
                "error": "invalid_season",
                "message": "Invalid season format. Use 20252026, 2025-26, or comma-separated values.",
            }),
            decision,
            0,
        );
    }

    let cache_key = format!("player|{player_id}|{season_ids:?}");
    if let Some(cached) = state.caches.contracts_api.get(&cache_key) {
        let status = if cached.get("error").is_some() {
            StatusCode::NOT_FOUND
        } else {
            StatusCode::OK
        };
        return public_json(status, (*cached).clone(), decision, 300);
    }

    let pid_eq = format!("eq.{player_id}");
    let f_contracts = filters(&[("player_id", &pid_eq)]);
    let mut f_years = filters(&[("player_id", &pid_eq)]);
    if !season_ids.is_empty() {
        let sid_in = season_ids
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        f_years.insert("season_id".into(), format!("in.({sid_in})"));
    }

    let (contracts_opt, years_opt) = tokio::join!(
        sb.read(
            "contracts",
            CONTRACTS_PUBLIC_COLS,
            Some(&f_contracts),
            None,
            Some("start_year.desc"),
            50,
        ),
        sb.read(
            "contract_years",
            CONTRACT_YEARS_PUBLIC_COLS,
            Some(&f_years),
            None,
            Some("season_id.asc"),
            100,
        ),
    );
    let contract_rows = contracts_opt.unwrap_or_default();
    let year_rows = years_opt.unwrap_or_default();

    if contract_rows.is_empty() && year_rows.is_empty() {
        let not_found = json!({
            "error": "not_found",
            "message": format!("No contract data found for player_id {player_id}."),
        });
        return public_json(StatusCode::NOT_FOUND, not_found, decision, 60);
    }

    let meta_map: HashMap<String, Value> = contract_rows
        .iter()
        .filter_map(|r| {
            let cid = str_value(r.get("contract_id"));
            if cid.is_empty() { None } else { Some((cid, r.clone())) }
        })
        .collect();

    let player_name = contract_rows
        .first()
        .or_else(|| year_rows.first())
        .and_then(|r| opt_str(r.get("player_name")));
    let position = contract_rows
        .first()
        .or_else(|| year_rows.first())
        .and_then(|r| opt_str(r.get("position")));

    let summary = if !season_ids.is_empty() {
        summarize_player_contract(&season_ids, &year_rows, &meta_map)
    } else {
        None
    };

    let matched_cids: BTreeSet<String> = year_rows
        .iter()
        .map(|r| str_value(r.get("contract_id")))
        .filter(|s| !s.is_empty())
        .collect();

    let filtered_contracts: Vec<Value> = if season_ids.is_empty() {
        contract_rows.iter().map(format_contract_row).collect()
    } else {
        contract_rows
            .iter()
            .filter(|r| matched_cids.contains(&str_value(r.get("contract_id"))))
            .map(format_contract_row)
            .collect()
    };

    let formatted_years: Vec<Value> = year_rows
        .iter()
        .map(|r| {
            let cid = str_value(r.get("contract_id"));
            format_contract_year_row(r, meta_map.get(&cid))
        })
        .collect();

    let mut payload = json!({
        "playerId": player_id,
        "playerName": player_name,
        "position": position,
        "contracts": filtered_contracts,
        "years": formatted_years,
    });
    if !season_ids.is_empty() {
        payload["seasonsRequested"] = json!(season_ids);
        payload["summary"] = summary.unwrap_or(Value::Null);
    }

    let arc = Arc::new(payload);
    state.caches.contracts_api.insert(cache_key, arc.clone());
    public_json(StatusCode::OK, (*arc).clone(), decision, 300)
}

/// `GET /api/v1/contracts/teams` — league-wide team salary cap overview for a season.
async fn api_contracts_teams(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let sb = match require_sb(&state, decision) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let season_raw = params.get("season").map(String::as_str).unwrap_or("");
    let season_id = if season_raw.trim().is_empty() {
        current_season_id(None)
    } else if let Some(sid) = parse_flexible_season(season_raw) {
        sid
    } else {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({
                "error": "invalid_season",
                "message": "Invalid season format. Use 20252026 or 2025-26.",
            }),
            decision,
            0,
        );
    };

    let sort = params
        .get("sort")
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_else(|| "cap_space_desc".to_string());

    let cache_key = format!("teams|{season_id}|{sort}");
    if let Some(cached) = state.caches.contracts_api.get(&cache_key) {
        return public_json(StatusCode::OK, (*cached).clone(), decision, 300);
    }

    let order = match sort.as_str() {
        "cap_space_asc" => "projected_cap_space_usd.asc.nullslast,team.asc",
        "cap_hit_desc" => "combined_one_way_plus_penalties_usd.desc.nullslast,team.asc",
        "cap_hit_asc" => "combined_one_way_plus_penalties_usd.asc.nullslast,team.asc",
        "total_aav_desc" => "total_aav_usd.desc.nullslast,team.asc",
        "team_asc" => "team.asc",
        _ => "projected_cap_space_usd.desc.nullslast,team.asc",
    };

    let rows = sb
        .read(
            "contract_teams_years",
            TEAM_YEARS_PUBLIC_COLS,
            Some(&filters(&[("season_id", &format!("eq.{season_id}"))])),
            None,
            Some(order),
            64,
        )
        .await
        .unwrap_or_default();

    let formatted: Vec<Value> = rows.iter().map(format_team_cap_row).collect();
    let cap_ceiling = formatted.first().and_then(|r| r.get("capCeilingUsd")).cloned();
    let cap_floor = formatted.first().and_then(|r| r.get("capFloorUsd")).cloned();

    let payload = Arc::new(json!({
        "seasonId": season_id,
        "season": formatted.first().and_then(|r| r.get("season")).cloned().unwrap_or(Value::Null),
        "capCeilingUsd": cap_ceiling.unwrap_or(Value::Null),
        "capFloorUsd": cap_floor.unwrap_or(Value::Null),
        "count": formatted.len(),
        "data": formatted,
    }));
    state.caches.contracts_api.insert(cache_key, payload.clone());
    public_json(StatusCode::OK, (*payload).clone(), decision, 300)
}

/// `GET /api/v1/contracts/teams/{team}` — single team cap summary, roster & outlook.
async fn api_contracts_team_detail(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(raw_team): Path<String>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let Some(team) = sanitize_team_code(&raw_team) else {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({ "error": "invalid_team", "message": "Team must be a valid NHL abbreviation (e.g., ANA, TOR, EDM)." }),
            decision,
            0,
        );
    };

    let season_raw = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = if season_raw.trim().is_empty() {
        vec![current_season_id(None)]
    } else {
        let parsed = parse_flexible_season_list(season_raw);
        if parsed.is_empty() {
            return public_json(
                StatusCode::BAD_REQUEST,
                json!({
                    "error": "invalid_season",
                    "message": "Invalid season format. Use 20252026, 2025-26, or comma-separated values.",
                }),
                decision,
                0,
            );
        }
        parsed
    };

    let pos_filter = params
        .get("position")
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let two_way_filter = parse_bool_param(&params, "two_way");

    let raw_data = load_team_contracts(&state.caches, state.sb.as_ref(), &team, &season_ids).await;
    let mut payload = (*raw_data).clone();

    if pos_filter.is_some() || two_way_filter.is_some() {
        if let Some(arr) = payload.get_mut("roster").and_then(Value::as_array_mut) {
            arr.retain(|r| {
                if let Some(ref pf) = pos_filter {
                    let pos = str_value(r.get("position")).to_uppercase();
                    let grp = str_value(r.get("posGroup")).to_uppercase();
                    if pf == "F" || pf == "D" || pf == "G" {
                        if &grp != pf && &pos != pf {
                            return false;
                        }
                    } else if &pos != pf {
                        return false;
                    }
                }
                if let Some(tw) = two_way_filter {
                    if opt_bool(r.get("twoWay")) != tw {
                        return false;
                    }
                }
                true
            });
        }
    }

    public_json(StatusCode::OK, payload, decision, 300)
}

/// `GET /api/v1/contracts/expiring` — upcoming expiring contracts (UFAs / RFAs).
async fn api_contracts_expiring(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let sb = match require_sb(&state, decision) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    // `year` is the calendar expiry year (e.g., 2026 => contracts ending in 2025-26 => end_year = 2025).
    let default_expiry_year = (current_season_id(None) % 10000) as i64;
    let expiry_year = params
        .get("year")
        .and_then(|v| v.trim().parse::<i64>().ok())
        .unwrap_or(default_expiry_year);
    if !(2011..=2040).contains(&expiry_year) {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({ "error": "invalid_year", "message": "year must be a 4-digit calendar year between 2011 and 2040 (e.g., 2026)." }),
            decision,
            0,
        );
    }
    let end_year = expiry_year - 1;

    let expiry_status = params
        .get("expiry_status")
        .or_else(|| params.get("status"))
        .map(|s| s.trim().to_uppercase())
        .filter(|s| matches!(s.as_str(), "UFA" | "RFA"));
    let team = params.get("team").and_then(|t| sanitize_team_code(t));
    let pos = params
        .get("position")
        .map(|s| s.trim().to_uppercase())
        .filter(|s| !s.is_empty());
    let min_aav = parse_f64_param(&params, "min_aav");
    let limit = parse_usize_param(&params, "limit", DEFAULT_LIMIT, MAX_LIMIT).max(1);
    let offset = parse_usize_param(&params, "offset", 0, 50_000);

    let cache_key = format!(
        "expiring|{expiry_year}|{expiry_status:?}|{team:?}|{pos:?}|{min_aav:?}|{limit}|{offset}"
    );
    if let Some(cached) = state.caches.contracts_api.get(&cache_key) {
        return public_json(StatusCode::OK, (*cached).clone(), decision, 300);
    }

    let mut f_map = filters(&[("end_year", &format!("eq.{end_year}"))]);
    if let Some(ref es) = expiry_status {
        f_map.insert("expiry_status".into(), format!("eq.{es}"));
    }
    if let Some(ref t) = team {
        f_map.insert("signing_team".into(), format!("eq.{t}"));
    }
    if let Some(ref p) = pos {
        if p == "F" {
            f_map.insert("position".into(), "in.(C,L,R,LW,RW)".into());
        } else {
            f_map.insert("position".into(), format!("eq.{p}"));
        }
    }
    if let Some(lo) = min_aav {
        f_map.insert("aav_usd".into(), format!("gte.{}", lo.round() as i64));
    }

    let rows = sb
        .read(
            "contracts",
            CONTRACTS_PUBLIC_COLS,
            Some(&f_map),
            None,
            Some("aav_usd.desc.nullslast,player_name.asc"),
            offset + limit,
        )
        .await
        .unwrap_or_default();

    let page: Vec<Value> = rows
        .iter()
        .skip(offset)
        .take(limit)
        .map(format_contract_row)
        .collect();

    let payload = Arc::new(json!({
        "expiryYear": expiry_year,
        "endSeason": format!("{}-{:02}", end_year, expiry_year % 100),
        "count": page.len(),
        "limit": limit,
        "offset": offset,
        "data": page,
    }));
    state.caches.contracts_api.insert(cache_key, payload.clone());
    public_json(StatusCode::OK, (*payload).clone(), decision, 300)
}

/// `GET /api/v1/contracts/penalties` — dead cap penalties (buyouts, retained salary, overages).
async fn api_contracts_penalties(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let decision = match check_rate_limit(&headers) {
        Ok(d) => d,
        Err(resp) => return resp,
    };
    let sb = match require_sb(&state, decision) {
        Ok(s) => s,
        Err(resp) => return resp,
    };

    let season_raw = params.get("season").map(String::as_str).unwrap_or("");
    let season_id = if season_raw.trim().is_empty() {
        current_season_id(None)
    } else if let Some(sid) = parse_flexible_season(season_raw) {
        sid
    } else {
        return public_json(
            StatusCode::BAD_REQUEST,
            json!({
                "error": "invalid_season",
                "message": "Invalid season format. Use 20252026 or 2025-26.",
            }),
            decision,
            0,
        );
    };
    let team = params.get("team").and_then(|t| sanitize_team_code(t));

    let cache_key = format!("penalties|{season_id}|{team:?}");
    if let Some(cached) = state.caches.contracts_api.get(&cache_key) {
        return public_json(StatusCode::OK, (*cached).clone(), decision, 300);
    }

    let mut f_map = filters(&[
        ("season_id", &format!("eq.{season_id}")),
        ("total_cap_penalties_usd", "gt.0"),
    ]);
    if let Some(ref t) = team {
        f_map.insert("team".into(), format!("eq.{t}"));
    }

    let rows = sb
        .read(
            "contract_teams_years",
            "team,season_id,season,buyout_penalty_usd,retained_salary_penalty_usd,bonus_overage_penalty_usd,total_cap_penalties_usd,penalties_detail",
            Some(&f_map),
            None,
            Some("total_cap_penalties_usd.desc,team.asc"),
            64,
        )
        .await
        .unwrap_or_default();

    let formatted: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "team": opt_str(r.get("team")),
                "seasonId": safe_int(r.get("season_id")),
                "season": opt_str(r.get("season")),
                "buyoutPenaltyUsd": opt_f64(r.get("buyout_penalty_usd")),
                "retainedSalaryPenaltyUsd": opt_f64(r.get("retained_salary_penalty_usd")),
                "bonusOveragePenaltyUsd": opt_f64(r.get("bonus_overage_penalty_usd")),
                "totalCapPenaltiesUsd": opt_f64(r.get("total_cap_penalties_usd")),
                "penaltiesDetail": r.get("penalties_detail").cloned().filter(|v| v.is_array()).unwrap_or_else(|| json!([])),
            })
        })
        .collect();

    let payload = Arc::new(json!({
        "seasonId": season_id,
        "count": formatted.len(),
        "data": formatted,
    }));
    state.caches.contracts_api.insert(cache_key, payload.clone());
    public_json(StatusCode::OK, (*payload).clone(), decision, 300)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flexible_season_parsing_handles_all_formats() {
        assert_eq!(parse_flexible_season("20252026"), Some(20252026));
        assert_eq!(parse_flexible_season("2025-26"), Some(20252026));
        assert_eq!(parse_flexible_season("2025/26"), Some(20252026));
        assert_eq!(parse_flexible_season("2025-2026"), Some(20252026));
        assert_eq!(parse_flexible_season("2025"), Some(20252026));
        assert_eq!(parse_flexible_season("1999-00"), Some(19992000));
        assert_eq!(parse_flexible_season("2025-27"), None);
        assert_eq!(parse_flexible_season("invalid"), None);

        assert_eq!(
            parse_flexible_season_list("2025-26, 20242025, 2025/26"),
            vec![20242025, 20252026]
        );
    }

    #[test]
    fn rate_limiter_enforces_per_ip_quota() {
        let lim = IpRateLimiter::new(2, 60);
        let d1 = lim.check("1.2.3.4");
        assert!(d1.allowed);
        assert_eq!(d1.remaining, 1);

        let d2 = lim.check("1.2.3.4");
        assert!(d2.allowed);
        assert_eq!(d2.remaining, 0);

        let d3 = lim.check("1.2.3.4");
        assert!(!d3.allowed);
        assert!(d3.retry_after_secs >= 1);

        // Separate IP gets its own quota.
        let other = lim.check("5.6.7.8");
        assert!(other.allowed);
        assert_eq!(other.remaining, 1);
    }
}
