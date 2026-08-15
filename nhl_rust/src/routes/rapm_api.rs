//! RAPM / context API routes — ports of `/api/rapm/player`,
//! `/api/context/player`, `/api/rapm/scale`, `/api/rapm/career`.

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::data::rapm;
use crate::routes::analytics::param_i64;
use crate::state::AppState;
use crate::util::parse::{parse_locale_float, parse_season_ids, primary_season_id, safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/rapm/player/{player_id}", get(api_rapm_player))
        .route("/api/context/player/{player_id}", get(api_context_player))
        .route("/api/rapm/scale", get(api_rapm_scale))
        .route("/api/rapm/career", get(api_rapm_career))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn q<'a>(params: &'a HashMap<String, String>, key: &str, default: &'a str) -> String {
    params
        .get(key)
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| default.to_string())
}

fn num(v: Option<&Value>) -> f64 {
    parse_locale_float(v).unwrap_or(0.0)
}

fn current_season() -> i64 {
    crate::util::dates::current_season_id(None)
}

/// `_load_rapm_player_rows_static` equivalent: rows filtered to a player.
fn rows_for_player<'a>(rows: &'a [Value], pid: i64) -> Vec<&'a Value> {
    rows.iter()
        .filter(|r| safe_int(r.get("PlayerID")).unwrap_or(0) == pid)
        .collect()
}

const RAPM_PROJECT_COLS: [&str; 28] = [
    "PlayerID", "Season", "StrengthState", "Rates_Totals",
    "CF", "CA", "GF", "GA", "xGF", "xGA",
    "C_plusminus", "G_plusminus", "xG_plusminus",
    "CF_zscore", "CA_zscore", "GF_zscore", "GA_zscore", "xGF_zscore", "xGA_zscore",
    "C_plusminus_zscore", "G_plusminus_zscore", "xG_plusminus_zscore",
    "PP_CF", "PP_GF", "PP_xGF", "SH_CA", "SH_GA", "SH_xGA",
];

fn project_row(row: &Value, cols: &[&str]) -> Value {
    let mut out = serde_json::Map::new();
    if let Some(obj) = row.as_object() {
        for c in cols {
            if let Some(v) = obj.get(*c) {
                out.insert(c.to_string(), v.clone());
            }
        }
    }
    Value::Object(out)
}

fn strength_order(st: &str) -> u8 {
    match st {
        "5v5" => 0,
        "PP" => 1,
        "SH" => 2,
        _ => 3,
    }
}

async fn api_rapm_player(
    State(state): State<AppState>,
    Path(player_id): Path<i64>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let pid = player_id;
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "invalid_player_id"}))).into_response();
    }
    let season_param = q(&params, "season", "");
    let season = season_param.parse::<i64>().ok();

    let season_list: Vec<i64> = match season {
        Some(s) => vec![s],
        None => state.last_dates.keys().copied().collect(),
    };
    let data = rapm::load_rapm_seasons(&state.caches, state.sb.as_ref(), &season_list).await;
    let mut player_rows: Vec<&Value> = Vec::new();
    for d in &data {
        player_rows.extend(rows_for_player(&d.totals, pid));
        player_rows.extend(rows_for_player(&d.rates, pid));
    }
    let mut out: Vec<Value> = player_rows
        .into_iter()
        .filter(|r| {
            if let Some(s) = season {
                safe_int(r.get("Season")).unwrap_or(0) == s
            } else {
                true
            }
        })
        .map(|r| project_row(r, &RAPM_PROJECT_COLS))
        .collect();
    out.sort_by(|a, b| {
        let sa = safe_int(a.get("Season")).unwrap_or(0);
        let sb = safe_int(b.get("Season")).unwrap_or(0);
        let oa = strength_order(&str_value(a.get("StrengthState")));
        let ob = strength_order(&str_value(b.get("StrengthState")));
        (sa, oa).cmp(&(sb, ob))
    });
    json_no_store(json!({"playerId": pid, "rows": out, "source": "supabase"}))
}

const CONTEXT_PROJECT_COLS: [&str; 7] = [
    "PlayerID", "Season", "StrengthState", "Minutes",
    "QoT_blend_xG67_G33", "QoC_blend_xG67_G33", "ZS_Difficulty",
];

async fn api_context_player(
    State(state): State<AppState>,
    Path(player_id): Path<i64>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let pid = player_id;
    let season_param = q(&params, "season", "");
    let season = season_param.parse::<i64>().ok();

    let season_list: Vec<i64> = match season {
        Some(s) => vec![s],
        None => state.last_dates.keys().copied().collect(),
    };
    let ctx_rows = rapm::load_context_seasons(&state.caches, state.sb.as_ref(), &season_list).await;
    let mut player_rows: Vec<&Value> = Vec::new();
    for rows in &ctx_rows {
        player_rows.extend(rows_for_player(rows, pid));
    }
    let mut out: Vec<Value> = player_rows
        .into_iter()
        .filter(|r| {
            if let Some(s) = season {
                safe_int(r.get("Season")).unwrap_or(0) == s
            } else {
                true
            }
        })
        .map(|r| project_row(r, &CONTEXT_PROJECT_COLS))
        .collect();
    out.sort_by(|a, b| {
        let sa = safe_int(a.get("Season")).unwrap_or(0);
        let sb = safe_int(b.get("Season")).unwrap_or(0);
        let oa = strength_order(&str_value(a.get("StrengthState")));
        let ob = strength_order(&str_value(b.get("StrengthState")));
        (sa, oa).cmp(&(sb, ob))
    });
    json_no_store(json!({"playerId": pid, "rows": out, "source": "supabase"}))
}

// ── RAPM scale / career ──────────────────────────────────────────

fn metric_cols(
    metric: &str,
) -> (&'static str, &'static str, &'static str, &'static str, &'static str, &'static str) {
    match metric {
        "xg" => ("xGF", "xGA", "PP_xGF", "SH_xGA", "xG+/-", "xG_plusminus"),
        "goals" => ("GF", "GA", "PP_GF", "SH_GA", "G+/-", "G_plusminus"),
        _ => ("CF", "CA", "PP_CF", "SH_CA", "C+/-", "C_plusminus"),
    }
}

fn rows_kind<'a>(rows: impl Iterator<Item = &'a Value>, rates: &str) -> Vec<&'a Value> {
    let want_totals = rates == "Totals";
    rows.filter(|r| {
        let rt = str_value(r.get("Rates_Totals")).to_lowercase();
        if want_totals {
            rt.starts_with("tot")
        } else {
            rt.starts_with("rate")
        }
    })
    .collect()
}

fn minutes_map(ctx_rows: &[Value]) -> HashMap<(String, String, String), f64> {
    let mut out = HashMap::new();
    for r in ctx_rows {
        let key = (
            str_value(r.get("PlayerID")),
            str_value(r.get("Season")),
            str_value(r.get("StrengthState")),
        );
        if let Some(m) = parse_locale_float(r.get("Minutes")) {
            if m > 0.0 {
                out.insert(key, m);
            }
        }
    }
    out
}

fn minutes_map_all(arcs: &[Arc<Vec<Value>>]) -> HashMap<(String, String, String), f64> {
    let mut out = HashMap::new();
    for rows in arcs {
        out.extend(minutes_map(rows));
    }
    out
}

/// `_load_rapm_static_csv`-style index for a single season: strength+pid → row.
fn index_season<'a>(
    rows: &'a [Value],
    season: i64,
    rates: &str,
) -> (Vec<i64>, HashMap<(String, i64), &'a Value>) {
    let mut by_key: HashMap<(String, i64), &Value> = HashMap::new();
    let mut pids: Vec<i64> = Vec::new();
    let mut seen: std::collections::HashSet<i64> = Default::default();
    let want_totals = rates == "Totals";
    for r in rows {
        let rt = str_value(r.get("Rates_Totals")).to_lowercase();
        if want_totals {
            if !rt.starts_with("tot") {
                continue;
            }
        } else if !rt.starts_with("rate") {
            continue;
        }
        if safe_int(r.get("Season")).unwrap_or(0) != season {
            continue;
        }
        let pid = safe_int(r.get("PlayerID")).unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        let strength = str_value(r.get("StrengthState"));
        by_key.entry((strength, pid)).or_insert(r);
        if seen.insert(pid) {
            pids.push(pid);
        }
    }
    (pids, by_key)
}

fn eligible(minutes: &HashMap<(String, String, String), f64>, pid: i64, season: i64, strength: &str) -> bool {
    let key = (pid.to_string(), season.to_string(), strength.to_string());
    match minutes.get(&key) {
        Some(m) => {
            let min = match strength {
                "PP" | "SH" => 40.0,
                _ => 100.0,
            };
            *m >= min
        }
        None => false,
    }
}

fn percentile_sorted_asc(sorted: &[f64], v: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let idx = sorted.partition_point(|&x| x <= v);
    Some(100.0 * (idx as f64 / sorted.len() as f64))
}

async fn api_rapm_scale(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let season_param = q(&params, "season", "");
    let season_ids = parse_season_ids(Some(&season_param), current_season());
    let primary = primary_season_id(&season_ids, current_season());
    let rates = q(&params, "rates", "Rates");
    let metric = q(&params, "metric", "corsi");
    let player_id = param_i64(&params, &["playerId"]);
    let player_id = if player_id == 0 { None } else { Some(player_id) };

    let rapm_all = rapm::load_rapm_data(&state.caches, state.sb.as_ref(), primary).await;
    let ctx_all = rapm::load_context_rows(&state.caches, state.sb.as_ref(), primary).await;
    let (col5_off, col5_def, col_pp, col_sh, _, _) = metric_cols(&metric);
    let minutes = minutes_map(&ctx_all);

    let rows_slice: &[Value] = if rates == "Totals" { &rapm_all.totals } else { &rapm_all.rates };
    let (seen_pids, by_key) = index_season(rows_slice, primary, &rates);

    // Values per player (season scope: use primary season).
    let mut off5: BTreeMap<i64, f64> = BTreeMap::new();
    let mut def5: BTreeMap<i64, f64> = BTreeMap::new();
    let mut pp_off: BTreeMap<i64, f64> = BTreeMap::new();
    let mut sh_def: BTreeMap<i64, f64> = BTreeMap::new();
    let mut eligible_map: BTreeMap<i64, bool> = BTreeMap::new();

    for pid in seen_pids {
        let el5 = eligible(&minutes, pid, primary, "5v5");
        let el_pp = eligible(&minutes, pid, primary, "PP");
        let el_sh = eligible(&minutes, pid, primary, "SH");
        eligible_map.insert(pid, el5 || el_pp || el_sh);
        if let Some(r) = by_key.get(&("5v5".to_string(), pid)) {
            off5.insert(pid, num(r.get(col5_off)));
            def5.insert(pid, num(r.get(col5_def)));
        }
        if let Some(r) = by_key.get(&("PP".to_string(), pid)) {
            pp_off.insert(pid, num(r.get(col_pp)));
        }
        if let Some(r) = by_key.get(&("SH".to_string(), pid)) {
            sh_def.insert(pid, num(r.get(col_sh)));
        }
    }

    // Distributions (def components sign-flipped so "good = positive").
    let dist_off5: Vec<f64> = off5.values().copied().filter(|v| v.is_finite()).collect();
    let dist_def5: Vec<f64> = def5.values().map(|v| -v).filter(|v| v.is_finite()).collect();
    let dist_pp: Vec<f64> = pp_off.values().copied().filter(|v| v.is_finite()).collect();
    let dist_sh: Vec<f64> = sh_def.values().map(|v| -v).filter(|v| v.is_finite()).collect();
    let mut sorted_off5 = dist_off5.clone();
    let mut sorted_def5 = dist_def5.clone();
    let mut sorted_pp = dist_pp.clone();
    let mut sorted_sh = dist_sh.clone();
    sorted_off5.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_def5.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_pp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    sorted_sh.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let scale5 = scale_of(&dist_off5, &dist_def5);
    let scale_pp = scale_of(&dist_pp, &[]);
    let scale_sh = scale_of(&dist_sh, &[]);

    let mut payload = json!({
        "season": primary,
        "rates": rates,
        "metric": metric,
        "source": "supabase",
        "contextSource": "supabase",
        "thresholds": {"fivev5": 100, "pp": 40, "sh": 40},
        "fivev5": {"min": scale5.0, "max": scale5.1},
        "pp": {"min": scale_pp.0, "max": scale_pp.1},
        "sh": {"min": scale_sh.0, "max": scale_sh.1},
    });

    if let Some(pid) = player_id {
        let minutes_5v5 = minutes.get(&(pid.to_string(), primary.to_string(), "5v5".to_string())).copied().unwrap_or(0.0);
        let minutes_pp = minutes.get(&(pid.to_string(), primary.to_string(), "PP".to_string())).copied().unwrap_or(0.0);
        let minutes_sh = minutes.get(&(pid.to_string(), primary.to_string(), "SH".to_string())).copied().unwrap_or(0.0);
        let is_eligible = *eligible_map.get(&pid).unwrap_or(&false);
        let o5 = off5.get(&pid).copied();
        let d5 = def5.get(&pid).copied();
        let pp = pp_off.get(&pid).copied();
        let sh = sh_def.get(&pid).copied();
        let diff = match (o5, d5) {
            (Some(o), Some(d)) => Some(o - d),
            _ => None,
        };
        let pct = |dist: &[f64], v: Option<f64>| v.and_then(|x| percentile_sorted_asc(dist, x));
        payload["playerId"] = json!(pid);
        payload["player"] = json!({
            "minutes": json!({"fivev5": minutes_5v5, "pp": minutes_pp, "sh": minutes_sh}),
            "eligible": is_eligible,
            "percentiles": {
                "5v5_off": pct(&sorted_off5, o5),
                "5v5_def": pct(&sorted_def5, d5.map(|v| -v)),
                "5v5_diff": pct(&sorted_off5, diff),
                "pp_off": pct(&sorted_pp, pp),
                "sh_def": pct(&sorted_sh, sh.map(|v| -v)),
            },
        });
    }
    json_no_store(payload)
}

fn scale_of(off: &[f64], def: &[f64]) -> (f64, f64) {
    let mut all = Vec::new();
    all.extend(off.iter().copied());
    all.extend(def.iter().copied());
    if all.is_empty() {
        return (0.0, 0.0);
    }
    let min = all.iter().copied().fold(f64::INFINITY, f64::min);
    let max = all.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let pad = (max.abs().max(min.abs())) * 0.1;
    (-max.abs().max(min.abs()) - pad, max.abs().max(min.abs()) + pad)
}

async fn api_rapm_career(
    State(state): State<AppState>,
    params: Query<HashMap<String, String>>,
) -> Response {
    let pid = param_i64(&params, &["playerId"]);
    if pid <= 0 {
        return (StatusCode::BAD_REQUEST, Json(json!({"error": "playerId_required"}))).into_response();
    }
    let rates = q(&params, "rates", "Rates");
    let metric = q(&params, "metric", "corsi");
    let strength = q(&params, "strength", "All");

    let season_list: Vec<i64> = state.last_dates.keys().copied().collect();
    let rapm_all = rapm::load_rapm_seasons(&state.caches, state.sb.as_ref(), &season_list).await;
    let ctx_arcs = rapm::load_context_seasons(&state.caches, state.sb.as_ref(), &season_list).await;
    let mut rows: Vec<&Value> = Vec::new();
    for d in &rapm_all {
        let slice: &[Value] = if rates == "Totals" { &d.totals } else { &d.rates };
        rows.extend(rows_kind(slice.iter(), &rates));
    }
    let (col5_off, col5_def, col_pp, col_sh, _, _) = metric_cols(&metric);
    let minutes = minutes_map_all(&ctx_arcs);

    // Group by season.
    let mut seasons: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
    for r in &rows {
        let s = safe_int(r.get("Season")).unwrap_or(0);
        if s > 0 {
            seasons.entry(s).or_default().push(*r);
        }
    }
    let mut seasons_sorted: Vec<i64> = seasons.keys().copied().collect();
    seasons_sorted.sort_unstable();

    let mut points: Vec<Value> = Vec::new();
    let mut overall_min = f64::INFINITY;
    let mut overall_max = f64::NEG_INFINITY;
    for s in &seasons_sorted {
        let season_rows = &seasons[s];
        let mut off5: BTreeMap<i64, f64> = BTreeMap::new();
        let mut def5: BTreeMap<i64, f64> = BTreeMap::new();
        let mut pp_off: BTreeMap<i64, f64> = BTreeMap::new();
        let mut sh_def: BTreeMap<i64, f64> = BTreeMap::new();
        for r in season_rows {
            let p = safe_int(r.get("PlayerID")).unwrap_or(0);
            let st = str_value(r.get("StrengthState"));
            match st.as_str() {
                "5v5" => {
                    off5.insert(p, num(r.get(col5_off)));
                    def5.insert(p, num(r.get(col5_def)));
                }
                "PP" => {
                    pp_off.insert(p, num(r.get(col_pp)));
                }
                "SH" => {
                    sh_def.insert(p, num(r.get(col_sh)));
                }
                _ => {}
            }
        }
        let dist_off5: Vec<f64> = off5.values().copied().filter(|v| v.is_finite()).collect();
        let dist_def5: Vec<f64> = def5.values().map(|v| -v).filter(|v| v.is_finite()).collect();
        let dist_pp: Vec<f64> = pp_off.values().copied().filter(|v| v.is_finite()).collect();
        let dist_sh: Vec<f64> = sh_def.values().map(|v| -v).filter(|v| v.is_finite()).collect();
        let mut so5 = dist_off5.clone();
        let mut sd5 = dist_def5.clone();
        let mut spp = dist_pp.clone();
        let mut ssh = dist_sh.clone();
        so5.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        sd5.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        spp.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ssh.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let o5 = off5.get(&pid).copied();
        let d5 = def5.get(&pid).copied();
        let pp = pp_off.get(&pid).copied();
        let sh = sh_def.get(&pid).copied();
        let diff = match (o5, d5) {
            (Some(o), Some(d)) => Some(o - d),
            _ => None,
        };
        let total = match (o5, d5, pp, sh) {
            (Some(o), Some(d), Some(p), Some(sh)) => Some(o - d + p + (-sh)),
            (Some(o), Some(d), _, _) => Some(o - d),
            _ => None,
        };
        let min_5 = minutes.get(&(pid.to_string(), s.to_string(), "5v5".to_string())).copied().unwrap_or(0.0);
        let min_pp = minutes.get(&(pid.to_string(), s.to_string(), "PP".to_string())).copied().unwrap_or(0.0);
        let min_sh = minutes.get(&(pid.to_string(), s.to_string(), "SH".to_string())).copied().unwrap_or(0.0);
        let el5 = eligible(&minutes, pid, *s, "5v5");
        let el_pp = eligible(&minutes, pid, *s, "PP");
        let el_sh = eligible(&minutes, pid, *s, "SH");

        let pct = |dist: &[f64], v: Option<f64>| v.and_then(|x| percentile_sorted_asc(dist, x));
        let z = |dist: &[f64], v: Option<f64>| -> Option<f64> {
            if dist.is_empty() {
                return None;
            }
            let mean = dist.iter().sum::<f64>() / dist.len() as f64;
            let var = dist.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / dist.len() as f64;
            let std = var.sqrt();
            if std <= 0.0 {
                return None;
            }
            v.map(|x| (x - mean) / std)
        };

        for v in [o5, d5.map(|x| -x), pp, sh.map(|x| -x)] {
            if let Some(x) = v {
                overall_min = overall_min.min(x);
                overall_max = overall_max.max(x);
            }
        }

        let point = json!({
            "Season": s,
            "minutes": json!({"fivev5": min_5, "pp": min_pp, "sh": min_sh}),
            "eligible": el5 || el_pp || el_sh,
            "5v5_off": o5, "5v5_off_z": z(&so5, o5), "5v5_off_pct": pct(&so5, o5),
            "5v5_def": d5, "5v5_def_z": z(&sd5, d5.map(|v| -v)), "5v5_def_pct": pct(&sd5, d5.map(|v| -v)),
            "5v5_diff": diff, "5v5_diff_z": z(&so5, diff), "5v5_diff_pct": pct(&so5, diff),
            "5v5_total": total, "5v5_total_z": z(&so5, total), "5v5_total_pct": pct(&so5, total),
            "pp_off": pp, "pp_off_z": z(&spp, pp), "pp_off_pct": pct(&spp, pp),
            "sh_def": sh, "sh_def_z": z(&ssh, sh.map(|v| -v)), "sh_def_pct": pct(&ssh, sh.map(|v| -v)),
            "all_total": total, "all_total_z": z(&so5, total), "all_total_pct": pct(&so5, total),
        });
        points.push(point);
    }
    if overall_min.is_finite() && overall_max.is_finite() {
        let pad = overall_max.abs().max(overall_min.abs()) * 0.1;
        overall_min = -overall_max.abs().max(overall_min.abs()) - pad;
        overall_max = overall_max.abs().max(overall_min.abs()) + pad;
    } else {
        overall_min = 0.0;
        overall_max = 0.0;
    }

    json_no_store(json!({
        "playerId": pid,
        "rates": rates,
        "metric": metric,
        "strength": strength,
        "thresholds": {"fivev5": 100, "pp": 40, "sh": 40},
        "seasons": seasons_sorted,
        "points": points,
        "scale": {"min": overall_min, "max": overall_max},
    }))
}
