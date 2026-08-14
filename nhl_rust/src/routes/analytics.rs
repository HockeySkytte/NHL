//! M2 analytics routes: card definitions, skater/goalie/team cards, tables,
//! scatters, goalie career series, and skater Edge. Ports of the corresponding
//! Flask handlers in `app/routes.py`.

use std::collections::{BTreeMap, HashMap, HashSet};

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use crate::data::{card_defs, rapm, seasonstats};
use crate::nhl::client::API_WEB;
use crate::state::AppState;
use crate::util::parse::{parse_locale_float, parse_season_ids, primary_season_id, safe_int, str_value};
use crate::util::stats as stats;

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/skaters/card/defs", get(api_skaters_card_defs))
        .route("/api/goalies/card/defs", get(api_goalies_card_defs))
        .route("/api/teams/card/defs", get(api_teams_card_defs))
        .route("/api/skaters/card", get(api_skaters_card))
        .route("/api/goalies/card", get(api_goalies_card))
        .route("/api/teams/card", get(api_teams_card))
        .route("/api/skaters/table", get(api_skaters_table_get).post(api_skaters_table_post))
        .route("/api/goalies/table", get(api_goalies_table_get).post(api_goalies_table_post))
        .route("/api/teams/table", get(api_teams_table_get).post(api_teams_table_post))
        .route("/api/skaters/scatter", get(api_skaters_scatter))
        .route("/api/goalies/scatter", get(api_goalies_scatter))
        .route("/api/teams/scatter", get(api_teams_scatter))
        .route("/api/goalies/series", get(api_goalies_series))
        .route("/api/skaters/edge", get(api_skaters_edge))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn num(v: Option<&Value>) -> f64 {
    parse_locale_float(v).unwrap_or(0.0)
}

fn q<'a>(params: &'a HashMap<String, String>, key: &str, default: &'a str) -> String {
    params
        .get(key)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| default.to_string())
}

pub(crate) fn param_i64(params: &HashMap<String, String>, keys: &[&str]) -> i64 {
    keys.iter()
        .find_map(|k| params.get(*k).and_then(|s| s.trim().parse::<i64>().ok()))
        .unwrap_or(0)
}

fn param_f64(params: &HashMap<String, String>, keys: &[&str]) -> f64 {
    keys.iter()
        .find_map(|k| params.get(*k).and_then(|s| s.trim().parse::<f64>().ok()))
        .unwrap_or(0.0)
}

fn flag_from_params(params: &HashMap<String, String>, key: &str, fallback: &str) -> bool {
    let raw = params
        .get(key)
        .map(|s| s.trim().to_lowercase())
        .unwrap_or_default();
    if raw.is_empty() {
        return matches!(fallback.to_lowercase().as_str(), "1" | "true" | "yes" | "y" | "force");
    }
    matches!(raw.as_str(), "1" | "true" | "yes" | "y" | "force")
}

#[derive(Clone)]
struct CardParams {
    season_ids: Vec<i64>,
    season: i64,
    season_state: String,
    strength_state: String,
    xg_model: String,
    rates: String,
    scope: String,
    min_gp: i64,
    min_toi: f64,
    metric_ids: Vec<String>,
}

fn parse_card_params(params: &HashMap<String, String>) -> CardParams {
    let mut season_state = q(params, "seasonState", "regular").to_lowercase();
    if !matches!(season_state.as_str(), "regular" | "playoffs" | "all") {
        season_state = "regular".to_string();
    }
    let mut strength_state = q(params, "strengthState", "5v5");
    if !matches!(strength_state.as_str(), "5v5" | "PP" | "SH" | "Other" | "all") {
        strength_state = "5v5".to_string();
    }
    let mut xg_model = q(params, "xgModel", "xG_F");
    if !matches!(xg_model.as_str(), "xG_S" | "xG_F" | "xG_F2") {
        xg_model = "xG_F".to_string();
    }
    let mut rates = q(params, "rates", "Totals");
    if !matches!(rates.as_str(), "Totals" | "Per60" | "PerGame") {
        rates = "Totals".to_string();
    }
    let mut scope = q(params, "scope", "season").to_lowercase();
    if !matches!(scope.as_str(), "season" | "career") {
        scope = "season".to_string();
    }
    let season_ids = parse_season_ids(
        params.get("season").map(String::as_str),
        current_season_fallback(),
    );
    let season = primary_season_id(&season_ids, current_season_fallback());
    let min_gp = param_i64(&params, &["minGP", "minGp", "min_gp"]).max(0);
    let min_toi = param_f64(&params, &["minTOI", "minToi", "min_toi"]).max(0.0);
    let metric_ids_raw = q(params, "metricIds", "");
    let metric_ids_raw = if metric_ids_raw.is_empty() {
        q(params, "metrics", "")
    } else {
        metric_ids_raw
    };
    let metric_ids: Vec<String> = metric_ids_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    CardParams {
        season_ids,
        season,
        season_state,
        strength_state,
        xg_model,
        rates,
        scope,
        min_gp,
        min_toi,
        metric_ids,
    }
}

fn current_season_fallback() -> i64 {
    crate::util::dates::current_season_id(None)
}

// ── Card definitions ─────────────────────────────────────────────

async fn api_skaters_card_defs(State(state): State<AppState>) -> Response {
    let defs = card_defs::load_defs(&state.caches, &state.cfg, "skaters").await;
    json_no_store(defs)
}
async fn api_goalies_card_defs(State(state): State<AppState>) -> Response {
    let defs = card_defs::load_defs(&state.caches, &state.cfg, "goalies").await;
    json_no_store(defs)
}
async fn api_teams_card_defs(State(state): State<AppState>) -> Response {
    let defs = card_defs::load_defs(&state.caches, &state.cfg, "teams").await;
    json_no_store(defs)
}

// ── Skater metric engine (shared by card/table/scatter) ──────────

fn attempts(v: &Map<String, Value>, xg_model: &str) -> f64 {
    if xg_model == "xG_S" {
        num(v.get("iShots"))
    } else {
        num(v.get("iFenwick"))
    }
}

fn ixg(v: &Map<String, Value>, xg_model: &str) -> f64 {
    match xg_model {
        "xG_F" => num(v.get("ixG_F")),
        "xG_F2" => num(v.get("ixG_F2")),
        _ => num(v.get("ixG_S")),
    }
}

fn xgf(v: &Map<String, Value>, xg_model: &str) -> f64 {
    match xg_model {
        "xG_F" => num(v.get("xGF_F")),
        "xG_F2" => num(v.get("xGF_F2")),
        _ => num(v.get("xGF_S")),
    }
}

fn xga(v: &Map<String, Value>, xg_model: &str) -> f64 {
    match xg_model {
        "xG_F" => num(v.get("xGA_F")),
        "xG_F2" => num(v.get("xGA_F2")),
        _ => num(v.get("xGA_S")),
    }
}

fn norm_rates_totals(v: Option<&Value>) -> String {
    let s = str_value(v).to_lowercase();
    if s.starts_with("tot") {
        "Totals".to_string()
    } else if s.starts_with("rate") {
        "Rates".to_string()
    } else {
        str_value(v)
    }
}

/// Picks the RAPM row for the requested strength + rates for a player/season.
fn pick_rapm_row(rows: &[Value], pid: i64, season: i64, want_strength: &str, want_rates: &str) -> Option<Value> {
    let mut candidates: Vec<&Value> = Vec::new();
    for r in rows {
        let p = safe_int(r.get("PlayerID")).unwrap_or(0);
        let s = safe_int(r.get("Season")).unwrap_or(0);
        if p != pid || (season != 0 && s != season) {
            continue;
        }
        candidates.push(r);
    }
    for r in &candidates {
        if str_value(r.get("StrengthState")) == want_strength
            && norm_rates_totals(r.get("Rates_Totals")) == want_rates
        {
            return Some((*r).clone());
        }
    }
    for r in &candidates {
        if norm_rates_totals(r.get("Rates_Totals")) == want_rates {
            return Some((*r).clone());
        }
    }
    candidates.first().map(|r| (*r).clone())
}

fn pick_ctx_row(rows: &[Value], pid: i64, season: i64, want_strength: &str) -> Option<Value> {
    let mut candidates: Vec<&Value> = Vec::new();
    for r in rows {
        let p = safe_int(r.get("PlayerID")).unwrap_or(0);
        let s = safe_int(r.get("Season")).unwrap_or(0);
        if p != pid || (season != 0 && s != season) {
            continue;
        }
        candidates.push(r);
    }
    for r in &candidates {
        if str_value(r.get("StrengthState")) == want_strength {
            return Some((*r).clone());
        }
    }
    for r in &candidates {
        if str_value(r.get("StrengthState")) == "5v5" {
            return Some((*r).clone());
        }
    }
    candidates.first().map(|r| (*r).clone())
}

/// Pre-builds an index of RAPM row indices keyed by (PlayerID, Season) so
/// bulk table requests (940+ players) don't do an O(rows) scan per player.
fn build_rapm_index(rows: &[Value]) -> HashMap<(i64, i64), Vec<usize>> {
    let mut idx: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        let p = safe_int(r.get("PlayerID")).unwrap_or(0);
        let s = safe_int(r.get("Season")).unwrap_or(0);
        idx.entry((p, s)).or_default().push(i);
    }
    idx
}

/// Indexed variant of `pick_rapm_row` — O(few) instead of a full-table scan.
fn pick_rapm_row_indexed(
    rows: &[Value],
    idx: &HashMap<(i64, i64), Vec<usize>>,
    pid: i64,
    season: i64,
    want_strength: &str,
    want_rates: &str,
) -> Option<Value> {
    let indices = idx.get(&(pid, season))?;
    for &i in indices {
        let r = &rows[i];
        if str_value(r.get("StrengthState")) == want_strength
            && norm_rates_totals(r.get("Rates_Totals")) == want_rates
        {
            return Some((*r).clone());
        }
    }
    for &i in indices {
        let r = &rows[i];
        if norm_rates_totals(r.get("Rates_Totals")) == want_rates {
            return Some((*r).clone());
        }
    }
    indices.first().map(|&i| rows[i].clone())
}

fn build_ctx_index(rows: &[Value]) -> HashMap<(i64, i64), Vec<usize>> {
    let mut idx: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (i, r) in rows.iter().enumerate() {
        let p = safe_int(r.get("PlayerID")).unwrap_or(0);
        let s = safe_int(r.get("Season")).unwrap_or(0);
        idx.entry((p, s)).or_default().push(i);
    }
    idx
}

/// Indexed variant of `pick_ctx_row` — O(few) instead of a full-table scan.
fn pick_ctx_row_indexed(
    rows: &[Value],
    idx: &HashMap<(i64, i64), Vec<usize>>,
    pid: i64,
    season: i64,
    want_strength: &str,
) -> Option<Value> {
    let indices = idx.get(&(pid, season))?;
    for &i in indices {
        let r = &rows[i];
        if str_value(r.get("StrengthState")) == want_strength {
            return Some((*r).clone());
        }
    }
    for &i in indices {
        let r = &rows[i];
        if str_value(r.get("StrengthState")) == "5v5" {
            return Some((*r).clone());
        }
    }
    indices.first().map(|&i| rows[i].clone())
}

#[allow(clippy::too_many_arguments)]
async fn skater_metric(
    state: &AppState,
    metric_id: &str,
    v: &Map<String, Value>,
    pid: i64,
    requested_pid: i64,
    params: &CardParams,
    rapm_row: Option<&Value>,
    ctx_row: Option<&Value>,
    def_map: &Map<String, Value>,
    special_pct: &mut BTreeMap<String, Option<f64>>,
) -> Option<f64> {
    let gp = num(v.get("GP"));
    let toi = num(v.get("TOI"));
    let igoals = num(v.get("iGoals"));
    let a1 = num(v.get("Assists1"));
    let a2 = num(v.get("Assists2"));
    let pts = igoals + a1 + a2;
    let att = attempts(v, &params.xg_model);
    let ixg_v = ixg(v, &params.xg_model);
    let cf = num(v.get("CF"));
    let ca = num(v.get("CA"));
    let ff = num(v.get("FF"));
    let fa = num(v.get("FA"));
    let sf = num(v.get("SF"));
    let sa = num(v.get("SA"));
    let gf = num(v.get("GF"));
    let ga = num(v.get("GA"));
    let xgf_v = xgf(v, &params.xg_model);
    let xga_v = xga(v, &params.xg_model);
    let pim_taken = num(v.get("PIM_taken"));
    let pim_drawn = num(v.get("PIM_drawn"));
    let pim_for = num(v.get("PIM_for"));
    let pim_against = num(v.get("PIM_against"));
    let hits = num(v.get("Hits"));
    let takeaways = num(v.get("Takeaways"));
    let giveaways = num(v.get("Giveaways"));

    let (category, metric) = match metric_id.split_once('|') {
        Some((c, m)) => (Some(c), m),
        None => (None, metric_id),
    };

    // Edge metrics.
    if category == Some("Edge") {
        if pid != requested_pid {
            return None;
        }
        if params.season < 20212022 {
            special_pct.insert(metric_id.to_string(), None);
            return None;
        }
        let mdef = def_map.get(metric_id).cloned().unwrap_or_else(|| json!({}));
        let link = str_value(mdef.get("link"));
        if link.is_empty() {
            special_pct.insert(metric_id.to_string(), None);
            return None;
        }
        let game_type = stats::edge_game_type(&params.season_state);
        let url = stats::edge_format_url(&link, pid, params.season, game_type);
        let Some(url) = url else {
            special_pct.insert(metric_id.to_string(), None);
            return None;
        };
        let payload = stats::edge_get_cached_json(&state.caches, &state.http, &url).await;
        let Some(payload) = payload else {
            special_pct.insert(metric_id.to_string(), None);
            return None;
        };
        let strength_code = if str_value(mdef.get("strengthCode")).to_lowercase() == "strengthcode" {
            stats::edge_strength_code(&params.strength_state)
        } else {
            None
        };
        if metric == "distanceTotal or distancePer60" {
            let (total_val, total_pct) = stats::edge_extract_value_and_pct(&payload, "distanceTotal", strength_code);
            let (per60_val, per60_pct) = stats::edge_extract_value_and_pct(&payload, "distancePer60", strength_code);
            if params.rates == "Per60" {
                special_pct.insert(metric_id.to_string(), per60_pct);
                return per60_val;
            }
            if params.rates == "PerGame" {
                special_pct.insert(metric_id.to_string(), total_pct);
                if total_val.is_none() || gp <= 0.0 {
                    return None;
                }
                return Some(total_val.unwrap() / gp);
            }
            special_pct.insert(metric_id.to_string(), total_pct);
            return total_val;
        }
        let (val_e, pct_e) = stats::edge_extract_value_and_pct(&payload, metric, strength_code);
        let mut val_e = val_e;
        if pct_e.is_some() {
            special_pct.insert(metric_id.to_string(), pct_e);
        } else {
            special_pct.insert(metric_id.to_string(), None);
        }
        if metric.to_lowercase().ends_with("pctg") {
            if let Some(fv) = val_e {
                if (0.0..=1.5).contains(&fv) {
                    val_e = Some(100.0 * fv);
                }
            }
        }
        return val_e;
    }

    // RAPM / context metrics (requested player only).
    if pid == requested_pid {
        if metric.starts_with("RAPM ") {
            let rapm_row = rapm_row?;
            let base = metric.trim_start_matches("RAPM").trim();
            let (col, zcol): (Option<&str>, Option<&str>) = match base {
                "CF" | "CA" | "GF" | "GA" | "xGF" | "xGA" => (Some(base), Some(metric_static_zcol(base))),
                "C+/-" => (Some("C_plusminus"), Some("C_plusminus_zscore")),
                "G+/-" => (Some("G_plusminus"), Some("G_plusminus_zscore")),
                "xG+/-" => (Some("xG_plusminus"), Some("xG_plusminus_zscore")),
                _ => (None, None),
            };
            let Some(col) = col else { return None };
            let val = parse_locale_float(rapm_row.get(col));
            let z = zcol.and_then(|z| parse_locale_float(rapm_row.get(z)));
            let mut pct = stats::z_to_pct(z);
            if pct.is_some() && stats::lower_is_better(metric_id) {
                pct = pct.map(|p| 100.0 - p);
            }
            special_pct.insert(metric_id.to_string(), pct);
            return val;
        }
        if category == Some("Context") && matches!(metric, "QoT" | "QoC" | "ZS") {
            let ctx_row = ctx_row?;
            let col = match metric {
                "QoT" => "QoT_blend_xG67_G33",
                "QoC" => "QoC_blend_xG67_G33",
                _ => "ZS_Difficulty",
            };
            let val = parse_locale_float(ctx_row.get(col));
            let pct = val.and_then(|v| stats::z_to_pct(Some(v)));
            special_pct.insert(metric_id.to_string(), pct);
            return val;
        }
    }

    let rate = |vv: f64| stats::rate_from(gp, toi, Some(vv), &params.rates);

    if metric == "GP" {
        return Some(gp);
    }
    if metric == "TOI" {
        return Some(toi);
    }
    if metric == "iGoals" {
        return rate(igoals);
    }
    if metric == "Assists1" {
        return rate(a1);
    }
    if metric == "Assists2" {
        return rate(a2);
    }
    if metric == "Points" {
        return rate(pts);
    }
    if matches!(metric, "iShots" | "iFenwick" | "iShots or iFenwick") {
        let vv = if params.xg_model == "xG_S" {
            num(v.get("iShots"))
        } else {
            num(v.get("iFenwick"))
        };
        return rate(vv);
    }
    if matches!(metric, "ixG" | "Individual xG") {
        return rate(ixg_v);
    }
    if category == Some("Shooting") && matches!(metric, "Sh% or FSh%" | "Sh%") {
        return stats::pct(Some(igoals), Some(att));
    }
    if category == Some("Shooting") && matches!(metric, "xSh% or xFS%" | "xSh% or xFSh%" | "xSh%") {
        return stats::pct(Some(ixg_v), Some(att));
    }
    if category == Some("Shooting") && matches!(metric, "dSh% or dFSh%") {
        let sh = stats::pct(Some(igoals), Some(att));
        let xsh = stats::pct(Some(ixg_v), Some(att));
        return match (sh, xsh) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };
    }
    if category == Some("Shooting") && metric == "GAx" {
        return rate(igoals - ixg_v);
    }

    let on_ice = |key: &str| -> Option<f64> {
        let vv = match key {
            "CF" => cf,
            "CA" => ca,
            "FF" => ff,
            "FA" => fa,
            "SF" => sf,
            "SA" => sa,
            "GF" => gf,
            "GA" => ga,
            "xGF" => xgf_v,
            "xGA" => xga_v,
            _ => 0.0,
        };
        rate(vv)
    };
    for key in ["CF", "CA", "FF", "FA", "SF", "SA", "GF", "GA", "xGF", "xGA"] {
        if metric == key {
            return on_ice(key);
        }
    }
    let pct_pair = |a: f64, b: f64| stats::pct(Some(a), Some(b));
    match metric {
        "CF%" => return pct_pair(cf, cf + ca),
        "FF%" => return pct_pair(ff, ff + fa),
        "SF%" => return pct_pair(sf, sf + sa),
        "GF%" => return pct_pair(gf, gf + ga),
        "xGF%" => return pct_pair(xgf_v, xgf_v + xga_v),
        "C+/-" => return rate(cf - ca),
        "F+/-" => return rate(ff - fa),
        "S+/-" => return rate(sf - sa),
        "G+/-" => return rate(gf - ga),
        "xG+/-" => return rate(xgf_v - xga_v),
        _ => {}
    }
    if category == Some("Context") && metric == "Sh%" {
        return stats::pct(Some(gf), Some(sf));
    }
    if category == Some("Context") && metric == "Sv%" {
        if sa <= 0.0 {
            return Some(if ga <= 0.0 { 100.0 } else { 0.0 });
        }
        return Some(100.0 * (1.0 - (ga / sa)));
    }
    if category == Some("Context") && metric == "PDO" {
        let sh_oi = stats::pct(Some(gf), Some(sf));
        let sv_oi = if sa <= 0.0 && ga <= 0.0 {
            Some(100.0)
        } else if sa <= 0.0 {
            Some(0.0)
        } else {
            Some(100.0 * (1.0 - (ga / sa)))
        };
        return match (sh_oi, sv_oi) {
            (Some(a), Some(b)) => Some(a + b),
            _ => None,
        };
    }
    if category == Some("Context") && metric == "GAx" {
        return rate(gf - xgf_v);
    }
    if category == Some("Context") && metric == "GSAx" {
        return rate(xga_v - ga);
    }
    if category == Some("Penalties") && metric == "PIM_taken" {
        return rate(pim_taken);
    }
    if category == Some("Penalties") && metric == "PIM_drawn" {
        return rate(pim_drawn);
    }
    if category == Some("Penalties") && metric == "PIM+/-" {
        return rate(pim_drawn - pim_taken);
    }
    if category == Some("Penalties") && metric == "PIM_For" {
        return rate(pim_for);
    }
    if category == Some("Penalties") && metric == "PIM_Against" {
        return rate(pim_against);
    }
    if category == Some("Penalties") && metric == "oiPIM+/-" {
        return rate(pim_for - pim_against);
    }
    if category == Some("Other") && metric == "Hits" {
        return rate(hits);
    }
    if category == Some("Other") && metric == "Takeaways" {
        return rate(takeaways);
    }
    if category == Some("Other") && metric == "Giveaways" {
        return rate(giveaways);
    }
    if v.contains_key(metric) {
        return rate(num(v.get(metric)));
    }
    None
}

fn metric_static_zcol(base: &str) -> &'static str {
    match base {
        "CF" => "CF_zscore",
        "CA" => "CA_zscore",
        "GF" => "GF_zscore",
        "GA" => "GA_zscore",
        "xGF" => "xGF_zscore",
        "xGA" => "xGA_zscore",
        _ => "",
    }
}

// ── Skaters card ─────────────────────────────────────────────────

async fn api_skaters_card(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let p = parse_card_params(&params);
    let pid = param_i64(&params, &["playerId", "player_id"]);
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "missing_playerId"}))).into_response();
    }

    let (agg, pos_group) = seasonstats::build_skater_agg(
        &state.caches,
        state.sb.as_ref(),
        &state.cfg,
        &p.scope,
        p.season,
        &p.season_ids,
        &p.season_state,
        &p.strength_state,
    )
    .await;
    let (mut agg_map, mut pos_map) = (agg.as_object().cloned().unwrap_or_default(), pos_group.as_object().cloned().unwrap_or_default());

    if p.min_gp > 0 || p.min_toi > 0.0 {
        let eligible: HashSet<String> = agg_map
            .iter()
            .filter(|(_, d)| {
                num(d.get("GP")) >= p.min_gp as f64 && num(d.get("TOI")) >= p.min_toi
            })
            .map(|(k, _)| k.clone())
            .collect();
        agg_map.retain(|k, _| eligible.contains(k));
        pos_map.retain(|k, _| eligible.contains(k));
    }

    let defs = card_defs::load_defs(&state.caches, &state.cfg, "skaters").await;
    let mut def_map: Map<String, Value> = Map::new();
    if let Some(metrics) = defs.get("metrics").and_then(Value::as_array) {
        for m in metrics {
            if let Some(id) = m.get("id").and_then(Value::as_str) {
                def_map.insert(id.to_string(), m.clone());
            }
        }
    }

    let metric_ids: Vec<String> = if p.metric_ids.is_empty() {
        vec![
            "Ice Time|GP".to_string(),
            "Ice Time|TOI".to_string(),
            "Production|iGoals".to_string(),
            "Production|Assists1".to_string(),
            "Production|Assists2".to_string(),
            "Production|Points".to_string(),
            "Shooting|ixG".to_string(),
            "Shooting|Sh% or FSh%".to_string(),
            "Shooting|xSh% or xFS%".to_string(),
            "Shooting|dSh% or dFSh%".to_string(),
        ]
    } else {
        p.metric_ids.clone()
    };

    let want_strength = if matches!(p.strength_state.as_str(), "5v5" | "PP" | "SH") {
        p.strength_state.clone()
    } else {
        "5v5".to_string()
    };
    let want_rapm_rates = if p.rates == "Totals" { "Totals" } else { "Rates" };

    let needs_rapm = metric_ids.iter().any(|m| m.contains("|RAPM "));
    let needs_ctx = metric_ids.iter().any(|m| matches!(m.as_str(), "Context|QoT" | "Context|QoC" | "Context|ZS"));

    let rapm_row = if needs_rapm {
        let rows = rapm::load_rapm_rows(&state.caches, state.sb.as_ref(), &state.cfg).await;
        pick_rapm_row(&rows, pid, p.season, &want_strength, want_rapm_rates)
    } else {
        None
    };
    let ctx_row = if needs_ctx {
        let rows = rapm::load_context_rows(&state.caches, state.sb.as_ref(), &state.cfg).await;
        pick_ctx_row(&rows, pid, p.season, &want_strength)
    } else {
        None
    };

    let mut special_pct: BTreeMap<String, Option<f64>> = BTreeMap::new();
    let mut derived: BTreeMap<i64, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for (pid_s, v) in &agg_map {
        let Some(vobj) = v.as_object() else { continue };
        let pid_i = pid_s.parse::<i64>().unwrap_or(0);
        let mut per_player: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for mid in &metric_ids {
            per_player.insert(
                mid.clone(),
                skater_metric(&state, mid, vobj, pid_i, pid, &p, rapm_row.as_ref(), ctx_row.as_ref(), &def_map, &mut special_pct).await,
            );
        }
        derived.insert(pid_i, per_player);
    }

    // Percentile pools.
    let mut dist_all: HashMap<String, Vec<f64>> = metric_ids.iter().map(|m| (m.clone(), Vec::new())).collect();
    let mut dist_by_pos: HashMap<(String, String), Vec<f64>> = HashMap::new();
    let edge_ids: HashSet<String> = metric_ids.iter().filter(|m| m.starts_with("Edge|")).cloned().collect();
    for (pid_i, m) in &derived {
        let g = pos_map
            .get(&pid_i.to_string())
            .and_then(Value::as_str)
            .map(|s| s.to_string())
            .unwrap_or_else(|| "F".to_string());
        let g = if g == "D" { "D".to_string() } else { "F".to_string() };
        for mid in &metric_ids {
            if edge_ids.contains(mid) {
                continue;
            }
            if let Some(fv) = m.get(mid).and_then(|v| *v) {
                if fv.is_finite() {
                    dist_all.get_mut(mid).unwrap().push(fv);
                    dist_by_pos.entry((g.clone(), mid.clone())).or_default().push(fv);
                }
            }
        }
    }
    for arr in dist_all.values_mut() {
        arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }
    for arr in dist_by_pos.values_mut() {
        arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mine_opt = derived.get(&pid).cloned();
    let mut seasonstats_missing = false;
    let mine = match mine_opt {
        Some(m) => m,
        None => {
            if p.season == 20252026 {
                seasonstats_missing = true;
                let empty_v = seasonstats_empty_row();
                let mut m: BTreeMap<String, Option<f64>> = BTreeMap::new();
                for mid in &metric_ids {
                    m.insert(
                        mid.clone(),
                        skater_metric(&state, mid, &empty_v, pid, pid, &p, rapm_row.as_ref(), ctx_row.as_ref(), &def_map, &mut special_pct).await,
                    );
                }
                m
            } else {
                return (
                    StatusCode::NOT_FOUND,
                    Json(json!({"error": "not_found", "playerId": pid, "source": "supabase"})),
                )
                    .into_response();
            }
        }
    };

    let my_group = pos_map
        .get(&pid.to_string())
        .and_then(Value::as_str)
        .map(|s| s.to_string())
        .unwrap_or_else(|| "F".to_string());
    let my_group = if my_group == "D" { "D".to_string() } else { "F".to_string() };

    let mut out_metrics: Map<String, Value> = Map::new();
    for mid in &metric_ids {
        let val = mine.get(mid).copied().flatten();
        let mut pct = special_pct.get(mid).copied().flatten();
        if pct.is_none() {
            let pool = dist_by_pos.get(&(my_group.clone(), mid.clone())).cloned()
                .filter(|v| !v.is_empty())
                .or_else(|| dist_all.get(mid).cloned());
            let mut p = stats::percentile_sorted(pool.as_deref().unwrap_or(&[]), val);
            if p.is_some() && stats::lower_is_better(mid) {
                p = p.map(|x| 100.0 - x);
            }
            pct = p;
        }
        out_metrics.insert(mid.clone(), json!({"value": val, "pct": pct}));
    }

    let label_attempts = if p.xg_model == "xG_S" { "iShots" } else { "iFenwick" };
    let (label_sh, label_xsh, label_dsh) = if p.xg_model == "xG_S" {
        ("Sh%", "xSh%", "dSh%")
    } else {
        ("FSh%", "xFSh%", "dFSh%")
    };

    json_no_store(json!({
        "playerId": pid,
        "season": p.season,
        "positionGroup": my_group,
        "scope": p.scope,
        "seasonState": p.season_state,
        "strengthState": p.strength_state,
        "xgModel": p.xg_model,
        "rates": p.rates,
        "minGP": p.min_gp,
        "minTOI": p.min_toi,
        "source": "supabase",
        "seasonStatsMissing": seasonstats_missing,
        "labels": {"Attempts": label_attempts, "Sh": label_sh, "xSh": label_xsh, "dSh": label_dsh},
        "metrics": Value::Object(out_metrics),
    }))
}

fn seasonstats_empty_row() -> Map<String, Value> {
    let mut m = Map::new();
    for k in [
        "GP", "TOI", "iGoals", "Assists1", "Assists2", "iShots", "iFenwick",
        "ixG_S", "ixG_F", "ixG_F2", "CA", "CF", "FA", "FF", "SA", "SF", "GA", "GF",
        "xGA_S", "xGF_S", "xGA_F", "xGF_F", "xGA_F2", "xGF_F2",
        "PIM_taken", "PIM_drawn", "PIM_for", "PIM_against", "Hits", "Takeaways", "Giveaways",
    ] {
        m.insert(k.to_string(), json!(0.0));
    }
    m
}

// ── Goalie card ──────────────────────────────────────────────────

async fn api_goalies_card(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let p = parse_card_params(&params);
    let pid = param_i64(&params, &["playerId", "player_id"]);
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "missing_playerId"}))).into_response();
    }

    let (agg, _pos) = seasonstats::build_goalie_agg(
        &state.caches,
        state.sb.as_ref(),
        &state.cfg,
        &p.scope,
        p.season,
        &p.season_ids,
        &p.season_state,
        &p.strength_state,
    )
    .await;
    let mut agg_map = agg.as_object().cloned().unwrap_or_default();

    if p.min_gp > 0 || p.min_toi > 0.0 {
        let eligible: HashSet<String> = agg_map
            .iter()
            .filter(|(_, d)| num(d.get("GP")) >= p.min_gp as f64 && num(d.get("TOI")) >= p.min_toi)
            .map(|(k, _)| k.clone())
            .collect();
        agg_map.retain(|k, _| eligible.contains(k));
    }

    // League-average save percentage from the pool.
    let (league_sa, league_ga): (f64, f64) = agg_map.values().fold((0.0, 0.0), |(sa, ga), d| {
        (sa + num(d.get("SA")), ga + num(d.get("GA")))
    });
    let league_sv_pct = if league_sa > 0.0 {
        100.0 * (1.0 - league_ga / league_sa)
    } else {
        100.0
    };

    let defs = card_defs::load_defs(&state.caches, &state.cfg, "goalies").await;
    let metric_ids: Vec<String> = if p.metric_ids.is_empty() {
        defs.get("metrics")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        p.metric_ids.clone()
    };

    let mut derived: BTreeMap<i64, BTreeMap<String, Option<f64>>> = BTreeMap::new();
    for (pid_s, v) in &agg_map {
        let Some(vobj) = v.as_object() else { continue };
        let pid_i = pid_s.parse::<i64>().unwrap_or(0);
        let mut per_player: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for mid in &metric_ids {
            per_player.insert(mid.clone(), goalie_metric(mid, vobj, &p, league_sv_pct));
        }
        derived.insert(pid_i, per_player);
    }

    let mine = derived.get(&pid).cloned();
    let Some(mine) = mine else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not_found"}))).into_response();
    };

    let mut dist_all: HashMap<String, Vec<f64>> = metric_ids.iter().map(|m| (m.clone(), Vec::new())).collect();
    for m in derived.values() {
        for (mid, val) in m {
            if let Some(fv) = val {
                if fv.is_finite() {
                    dist_all.get_mut(mid).unwrap().push(*fv);
                }
            }
        }
    }
    for arr in dist_all.values_mut() {
        arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut out_metrics: Map<String, Value> = Map::new();
    for mid in &metric_ids {
        let val = mine.get(mid).copied().flatten();
        let mut pct = stats::percentile_sorted(dist_all.get(mid).map(|v| v.as_slice()).unwrap_or(&[]), val);
        if pct.is_some() && stats::lower_is_better(mid) {
            pct = pct.map(|x| 100.0 - x);
        }
        out_metrics.insert(mid.clone(), json!({"value": val, "pct": pct}));
    }

    json_no_store(json!({
        "playerId": pid,
        "season": p.season,
        "scope": p.scope,
        "seasonState": p.season_state,
        "strengthState": p.strength_state,
        "xgModel": p.xg_model,
        "rates": p.rates,
        "minGP": p.min_gp,
        "minTOI": p.min_toi,
        "source": "supabase",
        "labels": {"Attempts": "SA", "Sv": "Sv%", "xSv": "xSv%", "dSv": "dSv%"},
        "metrics": Value::Object(out_metrics),
    }))
}

fn goalie_metric(metric_id: &str, v: &Map<String, Value>, p: &CardParams, league_sv_pct: f64) -> Option<f64> {
    let gp = num(v.get("GP"));
    let toi = num(v.get("TOI"));
    let fa = num(v.get("FA"));
    let sa = num(v.get("SA"));
    let ga = num(v.get("GA"));
    let xga_col = match p.xg_model.as_str() {
        "xG_S" => "xGA_S",
        "xG_F2" => "xGA_F2",
        _ => "xGA_F",
    };
    let xga = num(v.get(xga_col));
    let (_, metric) = metric_id.split_once('|').unwrap_or(("", metric_id));
    let rate = |vv: f64| stats::rate_from(gp, toi, Some(vv), &p.rates);

    // Flask `_sv_frac`: 1 - ga/att; att <= 0 -> 1.0 if ga <= 0 else 0.0.
    let sv_frac = |goals: f64, att: f64| -> f64 {
        if att <= 0.0 {
            if goals <= 0.0 { 1.0 } else { 0.0 }
        } else {
            1.0 - goals / att
        }
    };
    // Goalie IDs are "Sv% or FSv%": the denominator is SA for shots (xG_S)
    // and FA (Fenwick against) for Fenwick models (xG_F / xG_F2).
    let save_denom = if p.xg_model == "xG_S" { sa } else { fa };

    let sv_pct = if sa > 0.0 { Some(100.0 * (1.0 - ga / sa)) } else { None };
    let xsv_pct = if sa > 0.0 { Some(100.0 * (1.0 - xga / sa)) } else { None };

    match metric {
        "FA" => rate(fa),
        "SA" => rate(sa),
        "GA" => rate(ga),
        "xGA" | "xGA_S" | "xGA_F" | "xGA_F2" => rate(xga),
        "TOI" => Some(toi),
        "Sv% or FSv%" => Some(100.0 * sv_frac(ga, save_denom)),
        "xSv% or xFSv%" => Some(100.0 * sv_frac(xga, save_denom)),
        "dSv% or dFSv%" => Some(100.0 * (sv_frac(ga, save_denom) - sv_frac(xga, save_denom))),
        "Sv%" => sv_pct,
        "xSv%" => xsv_pct,
        "dSv%" => match (sv_pct, xsv_pct) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        },
        "GSAA" => {
            let player_sv = sv_pct.unwrap_or(100.0);
            Some((player_sv - league_sv_pct) / 100.0 * sa)
        }
        "GSAx" => Some(xga - ga),
        _ => None,
    }
}

// ── Team metric engine + teams card/table/scatter ────────────────

fn team_metric(metric_id: &str, b: &Map<String, Value>, rates: &str) -> Option<f64> {
    let gp = num(b.get("GP"));
    let toi = num(b.get("TOI"));
    let cf = num(b.get("CF"));
    let ca = num(b.get("CA"));
    let ff = num(b.get("FF"));
    let fa = num(b.get("FA"));
    let sf = num(b.get("SF"));
    let sa = num(b.get("SA"));
    let gf = num(b.get("GF"));
    let ga = num(b.get("GA"));
    let xgf = num(b.get("xGF"));
    let xga = num(b.get("xGA"));
    let (_, metric) = metric_id.split_once('|').unwrap_or(("", metric_id));
    let rate = |vv: f64| stats::rate_from(gp, toi, Some(vv), rates);
    let pct_pair = |a: f64, c: f64| stats::pct(Some(a), Some(c));

    match metric {
        "GP" => Some(gp),
        "TOI" => Some(toi),
        "CF" => rate(cf),
        "CA" => rate(ca),
        "FF" => rate(ff),
        "FA" => rate(fa),
        "SF" => rate(sf),
        "SA" => rate(sa),
        "GF" => rate(gf),
        "GA" => rate(ga),
        "xGF" => rate(xgf),
        "xGA" => rate(xga),
        "CF%" => pct_pair(cf, cf + ca),
        "FF%" => pct_pair(ff, ff + fa),
        "SF%" => pct_pair(sf, sf + sa),
        "GF%" => pct_pair(gf, gf + ga),
        "xGF%" => pct_pair(xgf, xgf + xga),
        "C+/-" => rate(cf - ca),
        "F+/-" => rate(ff - fa),
        "S+/-" => rate(sf - sa),
        "G+/-" => rate(gf - ga),
        "xG+/-" => rate(xgf - xga),
        "Sh%" => pct_pair(gf, sf),
        "Sv%" => {
            if sa > 0.0 {
                Some(100.0 * (1.0 - ga / sa))
            } else {
                None
            }
        }
        "PDO" => {
            let sh = pct_pair(gf, sf);
            let sv = if sa > 0.0 { Some(100.0 * (1.0 - ga / sa)) } else { None };
            match (sh, sv) {
                (Some(a), Some(b)) => Some(a + b),
                _ => None,
            }
        }
        "GAx" => rate(gf - xgf),
        "GSAx" => rate(xga - ga),
        _ => None,
    }
}

async fn api_teams_card(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let scope = q(&params, "scope", "season");
    let season_state = q(&params, "seasonState", "regular");
    let strength_state = q(&params, "strengthState", "5v5");
    let rates = q(&params, "rates", "Totals");
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = parse_season_ids(Some(season), current_season_fallback());
    let season_int = primary_season_id(&season_ids, current_season_fallback());
    let team = q(&params, "team", "");
    let team = if team.is_empty() {
        q(&params, "teamAbbrev", "")
    } else {
        team
    };
    let team = if team.is_empty() {
        q(&params, "team_abbrev", "")
    } else {
        team
    };
    let team = team.to_uppercase();
    if team.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "team_required"}))).into_response();
    }
    let metric_ids_raw = q(&params, "metricIds", "");

    let base = seasonstats::build_team_base_stats(
        &state.caches,
        &state.http,
        state.sb.as_ref(),
        &state.cfg,
        &state.teams,
        &scope,
        season_int,
        &season_ids,
        &season_state,
        &strength_state,
    )
    .await;
    let Some(team_base) = base.get(&team).cloned() else {
        return (StatusCode::NOT_FOUND, Json(json!({"error": "not_found", "team": team}))).into_response();
    };
    let team_obj = team_base.as_object().cloned().unwrap_or_default();

    let defs = card_defs::load_defs(&state.caches, &state.cfg, "teams").await;
    let metric_ids: Vec<String> = if metric_ids_raw.is_empty() {
        defs.get("metrics")
            .and_then(Value::as_array)
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| m.get("id").and_then(Value::as_str).map(|s| s.to_string()))
                    .collect()
            })
            .unwrap_or_default()
    } else {
        metric_ids_raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    };

    // Percentile pools across all teams.
    let mut dist_all: HashMap<String, Vec<f64>> = metric_ids.iter().map(|m| (m.clone(), Vec::new())).collect();
    for (_, b) in base.as_object().unwrap_or(&Map::new()) {
        let Some(bobj) = b.as_object() else { continue };
        for mid in &metric_ids {
            if mid.starts_with("Edge|") || mid.starts_with("Projection|") {
                continue;
            }
            if let Some(fv) = team_metric(mid, bobj, &rates) {
                if fv.is_finite() {
                    dist_all.get_mut(mid).unwrap().push(fv);
                }
            }
        }
    }
    for arr in dist_all.values_mut() {
        arr.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }

    let mut out_metrics: Map<String, Value> = Map::new();
    for mid in &metric_ids {
        // Edge and Projection metrics are not yet wired for teams (M2 follow-up).
        if mid.starts_with("Edge|") || mid.starts_with("Projection|") {
            out_metrics.insert(mid.clone(), json!({"value": null, "pct": null}));
            continue;
        }
        let val = team_metric(mid, &team_obj, &rates);
        let mut pct = stats::percentile_sorted(dist_all.get(mid).map(|v| v.as_slice()).unwrap_or(&[]), val);
        if pct.is_some() && stats::lower_is_better(mid) {
            pct = pct.map(|x| 100.0 - x);
        }
        out_metrics.insert(mid.clone(), json!({"value": val, "pct": pct}));
    }

    json_no_store(json!({
        "team": team,
        "teamId": num(team_obj.get("teamId")) as i64,
        "season": team_obj.get("season"),
        "seasons": season_ids,
        "scope": scope,
        "seasonState": season_state,
        "strengthState": strength_state,
        "rates": rates,
        "metrics": Value::Object(out_metrics),
    }))
}

fn active_team_filter(include_historic: bool) -> Box<dyn Fn(&Value) -> bool + Send + Sync> {
    if include_historic {
        Box::new(|_| true)
    } else {
        Box::new(|row| {
            row.get("Active")
                .and_then(Value::as_str)
                .map(|s| s == "1")
                .unwrap_or(false)
        })
    }
}

/// Converts a JSON POST body into the same string-params map the GET query
/// uses (arrays joined with commas, scalars stringified).
fn params_from_json(value: Value) -> HashMap<String, String> {
    let mut out = HashMap::new();
    if let Value::Object(map) = value {
        for (k, v) in map {
            let s = match v {
                Value::String(s) => s,
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Array(arr) => arr
                    .iter()
                    .map(|x| match x {
                        Value::String(s) => s.clone(),
                        Value::Number(n) => n.to_string(),
                        other => other.to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(","),
                Value::Null => String::new(),
                other => other.to_string(),
            };
            out.insert(k, s);
        }
    }
    out
}

// ── Teams table ─────────────────────────────────────────────────

async fn api_teams_table_get(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    teams_table_endpoint(state, &params).await
}

async fn api_teams_table_post(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    teams_table_endpoint(state, &params_from_json(body)).await
}

async fn teams_table_endpoint(state: AppState, params: &HashMap<String, String>) -> Response {
    let scope = q(params, "scope", "season");
    let season_state = q(params, "seasonState", "regular");
    let strength_state = q(params, "strengthState", "5v5");
    let rates = q(params, "rates", "Totals");
    let include_historic = flag_from_params(params, "includeHistoric", "0");
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = parse_season_ids(Some(season), current_season_fallback());
    let season_int = primary_season_id(&season_ids, current_season_fallback());
    let metric_ids_raw = q(params, "metricIds", "");
    let metric_ids: Vec<String> = if metric_ids_raw.is_empty() {
        Vec::new()
    } else {
        metric_ids_raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()
    };
    if metric_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "metricIds_required"}))).into_response();
    }

    let base = seasonstats::build_team_base_stats(
        &state.caches,
        &state.http,
        state.sb.as_ref(),
        &state.cfg,
        &state.teams,
        &scope,
        season_int,
        &season_ids,
        &season_state,
        &strength_state,
    )
    .await;

    let active = active_team_filter(include_historic);
    let mut rows: Vec<Value> = Vec::new();
    for (team_abbrev, b) in base.as_object().unwrap_or(&Map::new()) {
        // Only include teams from the Teams table (respecting Active).
        let Some(team_row) = state.teams.iter().find(|t| {
            str_value(t.get("Team")).to_uppercase() == *team_abbrev
        }) else {
            continue;
        };
        if !active(team_row) {
            continue;
        }
        let Some(bobj) = b.as_object() else { continue };
        let mut rec = Map::new();
        rec.insert("team".into(), json!(team_abbrev));
        rec.insert("teamId".into(), bobj.get("teamId").cloned().unwrap_or(Value::Null));
        for mid in &metric_ids {
            if mid.starts_with("Edge|") {
                rec.insert(mid.clone(), Value::Null);
                continue;
            }
            rec.insert(mid.clone(), team_metric(mid, bobj, &rates).map(|v| json!(v)).unwrap_or(Value::Null));
        }
        rows.push(Value::Object(rec));
    }
    rows.sort_by(|a, b| {
        str_value(a.get("team")).cmp(&str_value(b.get("team")))
    });

    json_no_store(json!({
        "season": season_int,
        "seasons": season_ids,
        "scope": scope,
        "seasonState": season_state,
        "strengthState": strength_state,
        "rates": rates,
        "metricIds": metric_ids,
        "rows": rows,
    }))
}

async fn api_teams_scatter(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let scope = q(&params, "scope", "season");
    let season_state = q(&params, "seasonState", "regular");
    let strength_state = q(&params, "strengthState", "5v5");
    let rates = q(&params, "rates", "Totals");
    let include_historic = flag_from_params(&params, "includeHistoric", "0");
    let x_mid = q(&params, "xMetricId", "");
    let y_mid = q(&params, "yMetricId", "");
    if x_mid.is_empty() || y_mid.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "metric_ids_required"}))).into_response();
    }
    if x_mid.starts_with("Edge|") || y_mid.starts_with("Edge|") {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "edge_not_supported"}))).into_response();
    }
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let season_ids = parse_season_ids(Some(season), current_season_fallback());
    let season_int = primary_season_id(&season_ids, current_season_fallback());

    let base = seasonstats::build_team_base_stats(
        &state.caches,
        &state.http,
        state.sb.as_ref(),
        &state.cfg,
        &state.teams,
        &scope,
        season_int,
        &season_ids,
        &season_state,
        &strength_state,
    )
    .await;
    let active = active_team_filter(include_historic);
    let mut points: Vec<Value> = Vec::new();
    for (team_abbrev, b) in base.as_object().unwrap_or(&Map::new()) {
        let Some(team_row) = state.teams.iter().find(|t| str_value(t.get("Team")).to_uppercase() == *team_abbrev) else {
            continue;
        };
        if !active(team_row) {
            continue;
        }
        let Some(bobj) = b.as_object() else { continue };
        let x = team_metric(&x_mid, bobj, &rates);
        let y = team_metric(&y_mid, bobj, &rates);
        let (Some(x), Some(y)) = (x, y) else { continue };
        if !x.is_finite() || !y.is_finite() {
            continue;
        }
        points.push(json!({
            "team": team_abbrev,
            "name": str_value(team_row.get("Name")),
            "x": x,
            "y": y,
        }));
    }
    json_no_store(json!({
        "season": season_int,
        "seasons": season_ids,
        "scope": scope,
        "seasonState": season_state,
        "rates": rates,
        "xMetricId": x_mid,
        "yMetricId": y_mid,
        "points": points,
    }))
}

// ── Skaters / goalies tables ─────────────────────────────────────

async fn api_skaters_table_get(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    skaters_table_endpoint(state, &params).await
}

async fn api_skaters_table_post(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    skaters_table_endpoint(state, &params_from_json(body)).await
}

async fn skaters_table_endpoint(state: AppState, params: &HashMap<String, String>) -> Response {
    let p = parse_card_params(params);
    let player_ids_raw = q(params, "playerIds", "");
    let player_ids: Vec<i64> = player_ids_raw
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if player_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "playerIds_required"}))).into_response();
    }
    let metric_ids_raw = q(params, "metricIds", "");
    let metric_ids: Vec<String> = metric_ids_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if metric_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "metricIds_required"}))).into_response();
    }

    let (agg, _pos_group) = seasonstats::build_skater_agg(
        &state.caches,
        state.sb.as_ref(),
        &state.cfg,
        &p.scope,
        p.season,
        &p.season_ids,
        &p.season_state,
        &p.strength_state,
    )
    .await;
    let agg_map = agg.as_object().cloned().unwrap_or_default();
    let defs = card_defs::load_defs(&state.caches, &state.cfg, "skaters").await;
    let mut def_map: Map<String, Value> = Map::new();
    if let Some(metrics) = defs.get("metrics").and_then(Value::as_array) {
        for m in metrics {
            if let Some(id) = m.get("id").and_then(Value::as_str) {
                def_map.insert(id.to_string(), m.clone());
            }
        }
    }
    let want_strength = if matches!(p.strength_state.as_str(), "5v5" | "PP" | "SH") {
        p.strength_state.clone()
    } else {
        "5v5".to_string()
    };
    let want_rapm_rates = if p.rates == "Totals" { "Totals" } else { "Rates" };
    let needs_rapm = metric_ids.iter().any(|m| m.contains("|RAPM "));
    let needs_ctx = metric_ids.iter().any(|m| matches!(m.as_str(), "Context|QoT" | "Context|QoC" | "Context|ZS"));
    let rapm_all = if needs_rapm {
        Some(rapm::load_rapm_rows(&state.caches, state.sb.as_ref(), &state.cfg).await)
    } else {
        None
    };
    let ctx_all = if needs_ctx {
        Some(rapm::load_context_rows(&state.caches, state.sb.as_ref(), &state.cfg).await)
    } else {
        None
    };

    // Pre-index RAPM/context rows by (PlayerID, Season) once so the per-player
    // pick below is O(few) instead of a full-table scan per player (~40M ops
    // for 940 players x 42k RAPM rows).
    let rapm_idx = rapm_all.as_ref().map(|rows| build_rapm_index(rows));
    let ctx_idx = ctx_all.as_ref().map(|rows| build_ctx_index(rows));
    let mut special_pct: BTreeMap<String, Option<f64>> = BTreeMap::new();
    let mut players_out: Vec<Value> = Vec::new();
    for pid in &player_ids {
        let Some(v) = agg_map.get(&pid.to_string()) else { continue };
        let Some(vobj) = v.as_object() else { continue };
        if num(vobj.get("GP")) < p.min_gp as f64 || num(vobj.get("TOI")) < p.min_toi {
            continue;
        }
        let rapm_row = match (&rapm_all, &rapm_idx) {
            (Some(rows), Some(idx)) => {
                pick_rapm_row_indexed(rows, idx, *pid, p.season, &want_strength, want_rapm_rates)
            }
            (Some(rows), None) => pick_rapm_row(rows, *pid, p.season, &want_strength, want_rapm_rates),
            _ => None,
        };
        let ctx_row = match (&ctx_all, &ctx_idx) {
            (Some(rows), Some(idx)) => pick_ctx_row_indexed(rows, idx, *pid, p.season, &want_strength),
            (Some(rows), None) => pick_ctx_row(rows, *pid, p.season, &want_strength),
            _ => None,
        };
        let mut metrics_out: Map<String, Value> = Map::new();
        for mid in &metric_ids {
            if mid.starts_with("Edge|") {
                metrics_out.insert(mid.clone(), Value::Null);
                continue;
            }
            let val = skater_metric(&state, mid, vobj, *pid, *pid, &p, rapm_row.as_ref(), ctx_row.as_ref(), &def_map, &mut special_pct).await;
            metrics_out.insert(mid.clone(), val.map(|v| json!(v)).unwrap_or(Value::Null));
        }
        players_out.push(json!({"playerId": pid, "metrics": Value::Object(metrics_out)}));
    }

    let label_attempts = if p.xg_model == "xG_S" { "iShots" } else { "iFenwick" };
    let (label_sh, label_xsh, label_dsh) = if p.xg_model == "xG_S" {
        ("Sh%", "xSh%", "dSh%")
    } else {
        ("FSh%", "xFSh%", "dFSh%")
    };
    json_no_store(json!({
        "season": p.season,
        "scope": p.scope,
        "seasonState": p.season_state,
        "strengthState": p.strength_state,
        "xgModel": p.xg_model,
        "rates": p.rates,
        "minGP": p.min_gp,
        "minTOI": p.min_toi,
        "playerIds": player_ids,
        "metricIds": metric_ids,
        "labels": {"Attempts": label_attempts, "Sh": label_sh, "xSh": label_xsh, "dSh": label_dsh},
        "players": players_out,
    }))
}

async fn api_goalies_table_get(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    goalies_table_endpoint(state, &params).await
}

async fn api_goalies_table_post(
    State(state): State<AppState>,
    Json(body): Json<Value>,
) -> Response {
    goalies_table_endpoint(state, &params_from_json(body)).await
}

async fn goalies_table_endpoint(state: AppState, params: &HashMap<String, String>) -> Response {
    let p = parse_card_params(params);
    let player_ids_raw = q(params, "playerIds", "");
    let player_ids: Vec<i64> = player_ids_raw
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    if player_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "playerIds_required"}))).into_response();
    }
    let metric_ids_raw = q(params, "metricIds", "");
    let metric_ids: Vec<String> = metric_ids_raw
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if metric_ids.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "metricIds_required"}))).into_response();
    }

    let (agg, _) = seasonstats::build_goalie_agg(
        &state.caches,
        state.sb.as_ref(),
        &state.cfg,
        &p.scope,
        p.season,
        &p.season_ids,
        &p.season_state,
        &p.strength_state,
    )
    .await;
    let agg_map = agg.as_object().cloned().unwrap_or_default();
    let (league_sa, league_ga): (f64, f64) = agg_map.values().fold((0.0, 0.0), |(sa, ga), d| {
        (sa + num(d.get("SA")), ga + num(d.get("GA")))
    });
    let league_sv_pct = if league_sa > 0.0 {
        100.0 * (1.0 - league_ga / league_sa)
    } else {
        100.0
    };

    let mut players_out: Vec<Value> = Vec::new();
    for pid in &player_ids {
        let Some(v) = agg_map.get(&pid.to_string()) else { continue };
        let Some(vobj) = v.as_object() else { continue };
        if num(vobj.get("GP")) < p.min_gp as f64 || num(vobj.get("TOI")) < p.min_toi {
            continue;
        }
        let mut metrics_out: Map<String, Value> = Map::new();
        for mid in &metric_ids {
            metrics_out.insert(mid.clone(), goalie_metric(mid, vobj, &p, league_sv_pct).map(|v| json!(v)).unwrap_or(Value::Null));
        }
        players_out.push(json!({"playerId": pid, "metrics": Value::Object(metrics_out)}));
    }

    json_no_store(json!({
        "season": p.season,
        "scope": p.scope,
        "seasonState": p.season_state,
        "strengthState": p.strength_state,
        "xgModel": p.xg_model,
        "rates": p.rates,
        "minGP": p.min_gp,
        "minTOI": p.min_toi,
        "playerIds": player_ids,
        "metricIds": metric_ids,
        "players": players_out,
    }))
}

// ── Scatters ─────────────────────────────────────────────────────

async fn api_skaters_scatter(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    scatter_endpoint(state, params, false).await
}

async fn api_goalies_scatter(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    scatter_endpoint(state, params, true).await
}

async fn scatter_endpoint(state: AppState, params: Query<HashMap<String, String>>, is_goalie: bool) -> Response {
    let p = parse_card_params(&params);
    let x_mid = q(&params, "xMetricId", "");
    let y_mid = q(&params, "yMetricId", "");
    if x_mid.is_empty() || y_mid.is_empty() {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "metric_ids_required"}))).into_response();
    }
    if x_mid.starts_with("Edge|") || y_mid.starts_with("Edge|") {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "edge_not_supported"}))).into_response();
    }
    let key = json!([p.season, p.season_ids, p.season_state, p.strength_state, p.xg_model, p.rates, p.scope, p.min_gp, p.min_toi, x_mid, y_mid]).to_string();
    let cache = if is_goalie {
        &state.caches.goalies_scatter
    } else {
        &state.caches.skaters_scatter
    };
    if let Some(cached) = cache.get(&key) {
        return json_no_store(cached);
    }

    let (agg, _pos_group) = if is_goalie {
        seasonstats::build_goalie_agg(
            &state.caches,
            state.sb.as_ref(),
            &state.cfg,
            &p.scope,
            p.season,
            &p.season_ids,
            &p.season_state,
            &p.strength_state,
        )
        .await
    } else {
        seasonstats::build_skater_agg(
            &state.caches,
            state.sb.as_ref(),
            &state.cfg,
            &p.scope,
            p.season,
            &p.season_ids,
            &p.season_state,
            &p.strength_state,
        )
        .await
    };
    let agg_map = agg.as_object().cloned().unwrap_or_default();

    // Names/teams from current rosters (best-effort).
    let roster_map = crate::data::rosters::all_rosters(&state.caches, &state.http).await;

    let mut points: Vec<Value> = Vec::new();
    if is_goalie {
        let (league_sa, league_ga): (f64, f64) = agg_map.values().fold((0.0, 0.0), |(sa, ga), d| {
            (sa + num(d.get("SA")), ga + num(d.get("GA")))
        });
        let league_sv_pct = if league_sa > 0.0 { 100.0 * (1.0 - league_ga / league_sa) } else { 100.0 };
        for (pid_s, v) in &agg_map {
            let Some(vobj) = v.as_object() else { continue };
            let pid = pid_s.parse::<i64>().unwrap_or(0);
            if num(vobj.get("GP")) < p.min_gp as f64 || num(vobj.get("TOI")) < p.min_toi {
                continue;
            }
            let x = goalie_metric(&x_mid, vobj, &p, league_sv_pct);
            let y = goalie_metric(&y_mid, vobj, &p, league_sv_pct);
            let (Some(x), Some(y)) = (x, y) else { continue };
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let info = roster_map.get(&pid.to_string()).cloned().unwrap_or_else(|| json!({}));
            points.push(json!({
                "playerId": pid,
                "name": str_value(info.get("name")),
                "team": str_value(info.get("team")),
                "x": x,
                "y": y,
                "gp": num(vobj.get("GP")),
                "toi": num(vobj.get("TOI")),
            }));
        }
    } else {
        let defs = card_defs::load_defs(&state.caches, &state.cfg, "skaters").await;
        let mut def_map: Map<String, Value> = Map::new();
        if let Some(metrics) = defs.get("metrics").and_then(Value::as_array) {
            for m in metrics {
                if let Some(id) = m.get("id").and_then(Value::as_str) {
                    def_map.insert(id.to_string(), m.clone());
                }
            }
        }
        let mut special_pct: BTreeMap<String, Option<f64>> = BTreeMap::new();
        for (pid_s, v) in &agg_map {
            let Some(vobj) = v.as_object() else { continue };
            let pid = pid_s.parse::<i64>().unwrap_or(0);
            if num(vobj.get("GP")) < p.min_gp as f64 || num(vobj.get("TOI")) < p.min_toi {
                continue;
            }
            let x = skater_metric(&state, &x_mid, vobj, pid, pid, &p, None, None, &def_map, &mut special_pct).await;
            let y = skater_metric(&state, &y_mid, vobj, pid, pid, &p, None, None, &def_map, &mut special_pct).await;
            let (Some(x), Some(y)) = (x, y) else { continue };
            if !x.is_finite() || !y.is_finite() {
                continue;
            }
            let info = roster_map.get(&pid.to_string()).cloned().unwrap_or_else(|| json!({}));
            points.push(json!({
                "playerId": pid,
                "name": str_value(info.get("name")),
                "team": str_value(info.get("team")),
                "x": x,
                "y": y,
                "gp": num(vobj.get("GP")),
                "toi": num(vobj.get("TOI")),
            }));
        }
    }

    let payload = json!({
        "season": p.season,
        "scope": p.scope,
        "seasonState": p.season_state,
        "strengthState": p.strength_state,
        "xgModel": p.xg_model,
        "rates": p.rates,
        "minGP": p.min_gp,
        "minTOI": p.min_toi,
        "xMetricId": x_mid,
        "yMetricId": y_mid,
        "points": points,
    });
    cache.insert(key, payload.clone());
    json_no_store(payload)
}

// ── Goalies series ───────────────────────────────────────────────

async fn api_goalies_series(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let pid = param_i64(&params, &["playerId"]);
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "playerId_required"}))).into_response();
    }
    let season_state = q(&params, "seasonState", "regular");
    let strength_state = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xga_col = match xg_model.as_str() {
        "xG_S" => "xGA_S",
        "xG_F2" => "xGA_F2",
        _ => "xGA_F",
    };

    let (by_pid_season, league_sa_ga) = seasonstats::build_goalies_career_matrix(
        &state.caches,
        state.sb.as_ref(),
        &state.cfg,
        &season_state,
        &strength_state,
        pid,
    )
    .await;

    let team_map = goalie_team_by_season(&state, pid).await;
    let mut seasons_out: Vec<Value> = Vec::new();
    let Some(seasons_map) = by_pid_season.get(&pid.to_string()).and_then(Value::as_object) else {
        return json_no_store(json!({"playerId": pid, "seasonState": season_state, "strengthState": strength_state, "xgModel": xg_model, "seasons": []}));
    };
    let mut seasons_sorted: Vec<i64> = seasons_map.keys().filter_map(|k| k.parse().ok()).collect();
    seasons_sorted.sort_unstable();
    for s in seasons_sorted {
        let Some(d) = seasons_map.get(&s.to_string()).and_then(Value::as_object) else { continue };
        let sa = num(d.get("SA"));
        let ga = num(d.get("GA"));
        let xga = num(d.get(xga_col));
        let league = league_sa_ga.get(&s.to_string()).and_then(Value::as_array).cloned().unwrap_or_default();
        let league_sa = league.first().and_then(Value::as_f64).unwrap_or(0.0);
        let league_ga = league.get(1).and_then(Value::as_f64).unwrap_or(0.0);
        let league_sv = if league_sa > 0.0 { 1.0 - league_ga / league_sa } else { 0.0 };
        let player_sv = if sa > 0.0 { 1.0 - ga / sa } else { 0.0 };
        let gsaa = (player_sv - league_sv) * sa;
        let gsax = if s < 20102011 { 0.0 } else { xga - ga };
        let team = team_map.get(&s.to_string()).cloned().unwrap_or_default();
        seasons_out.push(json!({"season": s, "team": team, "GSAA": gsaa, "GSAx": gsax}));
    }
    json_no_store(json!({
        "playerId": pid,
        "seasonState": season_state,
        "strengthState": strength_state,
        "xgModel": xg_model,
        "seasons": seasons_out,
    }))
}

/// Goalie primary team per season from the NHL `goalie/summary` endpoint.
async fn goalie_team_by_season(state: &AppState, pid: i64) -> BTreeMap<String, String> {
    let cache_key = (pid, "all".to_string());
    if let Some(v) = state.caches.goalie_team_by_season.get(&cache_key) {
        let mut out = BTreeMap::new();
        if let Some(obj) = v.as_object() {
            for (k, val) in obj {
                out.insert(k.clone(), str_value(Some(val)));
            }
        }
        return out;
    }
    let mut out: BTreeMap<String, (i64, f64, String)> = BTreeMap::new();
    let cay = format!("playerId={pid}");
    let rows = crate::nhl::stats_rest::summary_rows(&state.http, "goalie", &cay).await.unwrap_or_default();
    for r in rows {
        let season = str_value(r.get("seasonId"));
        let team = str_value(r.get("teamAbbrev")).to_uppercase();
        let gp = safe_int(r.get("gamesPlayed")).unwrap_or(0);
        let toi_sec = num(r.get("timeOnIce"));
        let weight = gp * 100_000 + toi_sec as i64;
        let entry = out.entry(season.clone()).or_insert((0, 0.0, String::new()));
        if weight >= entry.0 {
            *entry = (weight, toi_sec, team);
        }
    }
    let mut result: BTreeMap<String, String> = BTreeMap::new();
    for (s, (_, _, t)) in out {
        result.insert(s, t);
    }
    state
        .caches
        .goalie_team_by_season
        .insert(cache_key, json!(result));
    result
}

// ── Skaters Edge ─────────────────────────────────────────────────

async fn api_skaters_edge(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let pid = param_i64(&params, &["playerId"]);
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "playerId_required"}))).into_response();
    }
    let season = params.get("season").map(String::as_str).unwrap_or("");
    let season_state = q(&params, "seasonState", "regular");
    let season_ids = parse_season_ids(Some(season), current_season_fallback());
    let seasons_eligible: Vec<i64> = season_ids
        .iter()
        .copied()
        .filter(|s| *s >= 20212022)
        .collect();
    let game_type = stats::edge_game_type(&season_state);

    let base_urls = [
        "skater-shot-speed-detail",
        "skater-skating-speed-detail",
        "skater-zone-time",
        "skater-skating-distance-detail",
    ];
    let mut payloads: Vec<(String, Value)> = Vec::new();
    for endpoint in base_urls {
        let mut merged = json!({});
        for s in &seasons_eligible {
            let url = format!("{API_WEB}/v1/edge/{endpoint}/{pid}/{s}/{game_type}");
            if let Some(p) = stats::edge_get_cached_json(&state.caches, &state.http, &url).await {
                merged = merge_edge_payload(merged, p);
            }
        }
        payloads.push((endpoint.to_string(), merged));
    }

    let mut out = json!({
        "playerId": pid,
        "season": primary_season_id(&season_ids, current_season_fallback()),
        "seasons": seasons_eligible,
        "seasonState": season_state,
        "gameType": game_type,
        "available": !seasons_eligible.is_empty(),
        "shotSpeed": json!({}),
        "skatingSpeed": json!({}),
        "zoneTime": json!({}),
        "skatingDistance": json!({}),
    });
    if seasons_eligible.is_empty() {
        return json_no_store(out);
    }

    let find = |name: &str| -> Option<&Value> {
        payloads.iter().find(|(n, _)| n == name).map(|(_, v)| v)
    };

    // Shot speed.
    if let Some(p) = find("skater-shot-speed-detail") {
        let mut block = Map::new();
        for key in ["topShotSpeed", "avgShotSpeed", "shotAttempts70to80", "shotAttempts80to90", "shotAttempts90to100", "shotAttemptsOver100"] {
            let (v, pct, _) = stats::edge_extract_value_pct_avg(p, key, None);
            block.insert(key.to_string(), json!({"value": v, "pct": pct, "avg": null}));
        }
        out["shotSpeed"] = Value::Object(block);
    }
    // Skating speed.
    if let Some(p) = find("skater-skating-speed-detail") {
        let mut block = Map::new();
        for key in ["maxSkatingSpeed", "bursts18to20", "bursts20to22", "burstsOver22"] {
            let (v, pct, _) = stats::edge_extract_value_pct_avg(p, key, None);
            block.insert(key.to_string(), json!({"value": v, "pct": pct, "avg": null}));
        }
        out["skatingSpeed"] = Value::Object(block);
    }
    // Zone time (per strength code).
    if let Some(p) = find("skater-zone-time") {
        let mut block = Map::new();
        for code in ["all", "es", "pp", "pk"] {
            let mut sub = Map::new();
            for key in ["offensiveZonePctg", "neutralZonePctg", "defensiveZonePctg"] {
                let (v, pct, avg) = stats::edge_extract_value_pct_avg(p, key, Some(code));
                sub.insert(key.to_string(), json!({"value": v, "pct": pct, "avg": avg}));
            }
            block.insert(code.to_string(), Value::Object(sub));
        }
        out["zoneTime"] = Value::Object(block);
    }
    // Skating distance (per strength code).
    if let Some(p) = find("skater-skating-distance-detail") {
        let mut block = Map::new();
        for code in ["all", "es", "pp", "pk"] {
            let mut sub = Map::new();
            for key in ["distanceTotal", "distancePer60"] {
                let (v, pct, avg) = stats::edge_extract_value_pct_avg(p, key, Some(code));
                sub.insert(key.to_string(), json!({"value": v, "pct": pct, "avg": avg}));
            }
            block.insert(code.to_string(), Value::Object(sub));
        }
        out["skatingDistance"] = Value::Object(block);
    }
    json_no_store(out)
}

fn merge_edge_payload(a: Value, b: Value) -> Value {
    // Combine dict-of-dicts by extending keys (Python merges by metric across
    // seasons via max/sum/avg semantics handled downstream; here we just merge
    // object maps so the last season's per-key extraction remains available).
    match (a, b) {
        (Value::Object(mut a), Value::Object(b)) => {
            for (k, v) in b {
                a.entry(k).or_insert(v);
            }
            Value::Object(a)
        }
        (_, b) => b,
    }
}
