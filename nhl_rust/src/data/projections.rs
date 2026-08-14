//! M4 projections + GM Mode math — ports of the V2 g_diff_evppsh builder,
//! V2 win/expected-points, team projected points, custom lineups, club
//! schedules, and the full-season simulation engine from `app/routes.py`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::state::{AppState, Caches};
use crate::supabase::read::{filters, SbClient};
use crate::util::dates::current_season_id;
use crate::util::mt19937::Mt19937;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

// ── Coefficients & constants (verbatim from Flask) ──
pub const EVPP_COEF: [(&str, f64); 11] = [
    ("poss_value_ev", 1.315740),
    ("xga_ev", -0.191099),
    ("poss_value_st", 1.952908),
    ("xgf_pp", 0.127149),
    ("off_the_puck_sh", 0.272929),
    ("xga_sh", -0.284162),
    ("gax", 0.232042),
    ("gsax", 0.529976),
    ("rookie_f", -0.018529),
    ("rookie_d", -0.026125),
    ("rookie_g", -0.419905),
];

pub fn coef(name: &str) -> f64 {
    EVPP_COEF
        .iter()
        .find(|(k, _)| *k == name)
        .map(|(_, v)| *v)
        .unwrap_or(0.0)
}

pub const EV_SS: [&str; 3] = ["5v5", "4v4", "3v3"];
pub const PP_SS: [&str; 3] = ["5v4", "5v3", "4v3"];
pub const SH_SS: [&str; 3] = ["4v5", "3v4", "3v5"];

pub fn v2_lg_avg(season: i64) -> f64 {
    match season {
        20192020 => 2.9505,
        20202021 => 2.8834,
        20212022 => 3.1096,
        20222023 => 3.1422,
        20232024 => 3.0679,
        20242025 => 3.0168,
        20252026 => 3.0724,
        _ => 3.0,
    }
}

pub fn v2_situation(home_b2b: bool, away_b2b: bool) -> f64 {
    match (home_b2b, away_b2b) {
        (false, false) => 0.181937,
        (false, true) => 0.539297,
        (true, false) => -0.196136,
        (true, true) => 0.247761,
    }
}

pub const V2_CONSERVATIVE_WEIGHT: f64 = 0.7;

pub const ROOKIE_FALLBACK: [(&str, f64); 3] = [
    ("D", -0.031768511),
    ("F", -0.024601581),
    ("G", -0.12),
];

pub fn rookie_fallback(pos: &str) -> f64 {
    ROOKIE_FALLBACK
        .iter()
        .find(|(k, _)| *k == pos)
        .map(|(_, v)| *v)
        .unwrap_or(ROOKIE_FALLBACK[1].1)
}

pub const TEAM_CONF: [(&str, &str); 32] = [
    ("BOS", "East"), ("NJD", "East"), ("NYI", "East"), ("NYR", "East"),
    ("PHI", "East"), ("PIT", "East"), ("BUF", "East"), ("MTL", "East"),
    ("OTT", "East"), ("TOR", "East"), ("TBL", "East"), ("FLA", "East"),
    ("CAR", "East"), ("WSH", "East"), ("DET", "East"), ("CBJ", "East"),
    ("CHI", "West"), ("NSH", "West"), ("STL", "West"), ("WPG", "West"),
    ("DAL", "West"), ("COL", "West"), ("MIN", "West"), ("CGY", "West"),
    ("EDM", "West"), ("VAN", "West"), ("ANA", "West"), ("LAK", "West"),
    ("SJS", "West"), ("SEA", "West"), ("VGK", "West"), ("UTA", "West"),
];

pub fn team_conf(abbrev: &str) -> &'static str {
    TEAM_CONF
        .iter()
        .find(|(k, _)| *k == abbrev)
        .map(|(_, v)| *v)
        .unwrap_or("East")
}

const SIM_FLOOR_F_IG: f64 = 0.030;
const SIM_FLOOR_F_A: f64 = 0.050;
const SIM_FLOOR_D_IG: f64 = 0.012;
const SIM_FLOOR_D_A: f64 = 0.025;
const SIM_ASSIST_TWO: f64 = 0.65;
const SIM_ASSIST_ONE: f64 = 0.20;

// ── V2 math ──

pub fn v2_norm_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm::erf(x / std::f64::consts::SQRT_2))
}

pub fn v2_win_probability(home_proj: f64, away_proj: f64, home_b2b: bool, away_b2b: bool, season: i64) -> f64 {
    let sit = v2_situation(home_b2b, away_b2b);
    let mu = V2_CONSERVATIVE_WEIGHT * (home_proj - away_proj + sit);
    let lg = v2_lg_avg(season);
    let gf_home = (lg + mu / 2.0).max(0.5);
    let gf_away = (lg - mu / 2.0).max(0.5);
    let sigma = (gf_home + gf_away).sqrt();
    v2_norm_cdf(mu / sigma)
}

const MAX_K: usize = 30;

fn log_fact() -> Vec<f64> {
    let mut lf = vec![0.0; MAX_K + 1];
    for k in 1..=MAX_K {
        lf[k] = lf[k - 1] + (k as f64).ln();
    }
    lf
}

/// `_v2_expected_points_exact`: exact Poisson + OT expected points.
pub fn v2_expected_points_exact(gf_home: f64, gf_away: f64, p_home_norm: f64) -> (f64, f64) {
    let h_lam = gf_home.max(0.01);
    let a_lam = gf_away.max(0.01);
    let lf = log_fact();
    let mut h_pmf = [0.0f64; MAX_K + 1];
    let mut a_pmf = [0.0f64; MAX_K + 1];
    for k in 0..=MAX_K {
        h_pmf[k] = ((k as f64) * h_lam.ln() - h_lam - lf[k]).exp();
        a_pmf[k] = ((k as f64) * a_lam.ln() - a_lam - lf[k]).exp();
    }
    let mut e_home = 0.0;
    let mut e_away = 0.0;
    for i in 0..=MAX_K {
        let pi = h_pmf[i];
        if pi == 0.0 {
            continue;
        }
        for j in 0..=MAX_K {
            let pj = a_pmf[j];
            if pj == 0.0 {
                continue;
            }
            let prob = pi * pj;
            if i > j + 1 {
                e_home += 2.0 * prob;
            } else if i == j + 1 {
                e_home += 2.0 * prob;
                e_away += 0.3 * prob;
            } else if i == j {
                e_home += (2.0 * p_home_norm + 1.0 * (1.0 - p_home_norm)) * prob;
                e_away += (2.0 * (1.0 - p_home_norm) + 1.0 * p_home_norm) * prob;
            } else if i + 1 == j {
                e_away += 2.0 * prob;
                e_home += 0.3 * prob;
            } else {
                e_away += 2.0 * prob;
            }
        }
    }
    (e_home, e_away)
}

fn round6(x: f64) -> f64 {
    (x * 1e6).round() / 1e6
}

// ── V2 builder ──

/// `_compute_projection_value(row)`.
pub fn compute_projection_value(r: &Value) -> f64 {
    let ss = str_value(r.get("strengthstate"));
    let pos = str_value(r.get("position"));
    let mut v = 0.0;
    let f = |key: &str| parse_locale_float(r.get(key)).unwrap_or(0.0);

    let in_ev = EV_SS.contains(&ss.as_str());
    let in_pp = PP_SS.contains(&ss.as_str());
    let in_sh = SH_SS.contains(&ss.as_str());

    if in_ev {
        v += f("faceoffs") * coef("poss_value_ev");
    } else if in_pp || in_sh {
        v += f("faceoffs") * coef("poss_value_st");
    }
    if in_ev {
        v += f("defensive") * coef("poss_value_ev");
    } else if in_sh {
        v += f("defensive") * coef("poss_value_st");
    }
    if in_ev {
        v += f("dump_ins_outs") * coef("poss_value_ev");
    } else if in_sh {
        v += f("dump_ins_outs") * coef("poss_value_st");
    }
    if in_ev {
        v += f("passes") * coef("poss_value_ev");
    }
    if in_ev {
        v += f("carries") * coef("poss_value_ev");
    }
    if in_ev {
        v += f("xga") * coef("xga_ev");
    } else if in_sh {
        v += f("xga") * coef("xga_sh");
    }
    if in_pp {
        v += f("xgf") * coef("xgf_pp");
    }
    if in_sh {
        v += f("off_the_puck") * coef("off_the_puck_sh");
    }
    v += f("gax") * coef("gax");
    v += f("gsax") * coef("gsax");
    if ss == "5v5" {
        let rv = f("rookie");
        if rv > 0.0 {
            v += rv * match pos.as_str() {
                "F" => coef("rookie_f"),
                "D" => coef("rookie_d"),
                "G" => coef("rookie_g"),
                _ => coef("rookie_f"),
            };
        }
    }
    v
}

/// `_build_v2_player_projections(season)`: cached at the builder level.
pub async fn build_v2_player_projections(
    state: &AppState,
    season: Option<i64>,
) -> Vec<Value> {
    let cache_key = season.map(|s| s.to_string()).unwrap_or_else(|| "ALL".to_string());
    if let Some(v) = state.caches.v2_build.get(&cache_key) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }
    let Some(sb) = state.sb.as_ref() else { return Vec::new() };

    let mut db_filters: Vec<(&str, String)> = Vec::new();
    if let Some(s) = season {
        db_filters.push(("season", format!("eq.{s}")));
    }
    let fmap: BTreeMap<String, String> = db_filters
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    let rows = sb
        .read(
            "nhl_current_playerprojections",
            "*",
            if fmap.is_empty() { None } else { Some(&fmap) },
            None,
            Some("playerid,strengthstate"),
            0,
        )
        .await
        .unwrap_or_default();
    if rows.is_empty() {
        return Vec::new();
    }

    let ss_groups: [(&str, &[&str]); 4] = [
        ("5v5", &["5v5"]),
        ("PP", &["5v4", "4v3", "5v3"]),
        ("SH", &["4v5", "3v4", "3v5"]),
        ("Other", &["4v4", "3v3"]),
    ];

    #[derive(Default)]
    struct Acc {
        player_id: i64,
        name: String,
        position: String,
        team: String,
        gp: i64,
        proj_by_ss: HashMap<String, f64>,
        evo: f64,
        evd: f64,
        pp_raw: f64,
        sh_raw: f64,
        gax: f64,
        gsax: f64,
        rookie: f64,
        ig: f64,
        a1: f64,
        a2: f64,
    }

    let mut players: BTreeMap<i64, Acc> = BTreeMap::new();
    for r in &rows {
        let pid = safe_int(r.get("nhl_api_player_id"))
            .or_else(|| safe_int(r.get("playerid")))
            .unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        let ss = str_value(r.get("strengthstate"));
        let acc = players.entry(pid).or_insert_with(|| Acc {
            player_id: pid,
            name: str_value(r.get("nhl_player_name")),
            position: str_value(r.get("position")),
            team: str_value(r.get("team")),
            gp: safe_int(r.get("gp")).unwrap_or(0),
            ..Default::default()
        });
        if !ss.is_empty() {
            acc.proj_by_ss.insert(ss.clone(), compute_projection_value(r));
        }
        acc.ig += parse_locale_float(r.get("ig")).unwrap_or(0.0);
        acc.a1 += parse_locale_float(r.get("a1")).unwrap_or(0.0);
        acc.a2 += parse_locale_float(r.get("a2")).unwrap_or(0.0);
        let f = parse_locale_float(r.get("faceoffs")).unwrap_or(0.0);
        let pa = parse_locale_float(r.get("passes")).unwrap_or(0.0);
        let ca = parse_locale_float(r.get("carries")).unwrap_or(0.0);
        let de = parse_locale_float(r.get("defensive")).unwrap_or(0.0);
        let di = parse_locale_float(r.get("dump_ins_outs")).unwrap_or(0.0);
        let ot = parse_locale_float(r.get("off_the_puck")).unwrap_or(0.0);
        let xg = parse_locale_float(r.get("xga")).unwrap_or(0.0);
        let xf = parse_locale_float(r.get("xgf")).unwrap_or(0.0);
        let ga = parse_locale_float(r.get("gax")).unwrap_or(0.0);
        let gs = parse_locale_float(r.get("gsax")).unwrap_or(0.0);
        if EV_SS.contains(&ss.as_str()) {
            acc.evo += (f + pa + ca) * coef("poss_value_ev");
            acc.evd += (de + di) * coef("poss_value_ev") + xg * coef("xga_ev");
        }
        if PP_SS.contains(&ss.as_str()) {
            acc.pp_raw += f * coef("poss_value_st") + xf * coef("xgf_pp");
        }
        if SH_SS.contains(&ss.as_str()) {
            acc.sh_raw += (f + de + di) * coef("poss_value_st") + ot * coef("off_the_puck_sh") + xg * coef("xga_sh");
        }
        acc.gax += ga * coef("gax");
        acc.gsax += gs * coef("gsax");
        if ss == "5v5" {
            let pos = str_value(r.get("position"));
            let rv = parse_locale_float(r.get("rookie")).unwrap_or(0.0);
            if rv > 0.0 {
                acc.rookie += rv * match pos.as_str() {
                    "D" => coef("rookie_d"),
                    "G" => coef("rookie_g"),
                    _ => coef("rookie_f"),
                };
            }
        }
    }

    // Context data.
    let ctx_rows = crate::data::rapm::load_context_rows(&state.caches, state.sb.as_ref(), &state.cfg).await;
    let season_filter = season.unwrap_or_else(|| current_season_id(None));
    let mut ctx_by_pid: HashMap<i64, (Option<f64>, Option<f64>, Option<f64>)> = HashMap::new();
    for row in &ctx_rows {
        if safe_int(row.get("Season")).unwrap_or(0) != season_filter {
            continue;
        }
        if str_value(row.get("StrengthState")).trim() != "5v5" {
            continue;
        }
        let pid = safe_int(row.get("PlayerID"))
            .or_else(|| safe_int(row.get("player_id")))
            .unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        ctx_by_pid.insert(
            pid,
            (
                parse_locale_float(row.get("QoT_blend_xG67_G33")),
                parse_locale_float(row.get("QoC_blend_xG67_G33")),
                parse_locale_float(row.get("ZS_Difficulty")),
            ),
        );
    }

    let mut result: Vec<Value> = Vec::new();
    for (pid, mut p) in players {
        let mut projections = serde_json::Map::new();
        for (group_name, ss_list) in &ss_groups {
            let mut sum = 0.0;
            for ss in *ss_list {
                sum += p.proj_by_ss.get(*ss).copied().unwrap_or(0.0);
            }
            projections.insert(group_name.to_string(), json!(round6(sum)));
        }
        let ctx = ctx_by_pid.get(&pid).copied().unwrap_or((None, None, None));
        let fmt_ctx = |v: Option<f64>| match v {
            None => Value::Null,
            Some(x) => json!(round6(x)),
        };
        result.push(json!({
            "player_id": pid,
            "name": p.name,
            "position": p.position,
            "team": p.team,
            "gp": p.gp,
            "projections": Value::Object(projections),
            "evo": round6(p.evo),
            "evd": round6(p.evd),
            "pp_raw": round6(p.pp_raw),
            "sh_raw": round6(p.sh_raw),
            "gax": round6(p.gax),
            "gsax": round6(p.gsax),
            "rookie": round6(p.rookie),
            "ig": round6(p.ig),
            "a1": round6(p.a1),
            "a2": round6(p.a2),
            "qot": fmt_ctx(ctx.0),
            "qoc": fmt_ctx(ctx.1),
            "zs": fmt_ctx(ctx.2),
        }));
    }
    result.sort_by_key(|p| safe_int(p.get("player_id")).unwrap_or(0));
    state.caches.v2_build.insert(cache_key, Value::Array(result.clone()));
    result
}

/// `_load_v2_player_projections_cached()`: keyed by nhl_api_player_id, current season.
pub async fn load_v2_player_projections_cached(state: &AppState) -> HashMap<i64, Value> {
    let season = current_season_id(None);
    let players = build_v2_player_projections(state, Some(season)).await;
    let mut out: HashMap<i64, Value> = HashMap::new();
    for p in players {
        let pid = safe_int(p.get("player_id")).unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        let gp = safe_int(p.get("gp")).unwrap_or(0);
        let gp_weight = if gp > 0 && gp < 41 { gp as f64 / 41.0 } else { 1.0 };
        let f = |key: &str| parse_locale_float(p.get(key)).unwrap_or(0.0);
        let evo = f("evo") * gp_weight;
        let evd = f("evd") * gp_weight;
        let pp = f("pp_raw") * gp_weight;
        let sh = f("sh_raw") * gp_weight;
        let gax = f("gax") * gp_weight;
        let gsax = f("gsax") * gp_weight;
        let rookie = f("rookie"); // not GP-weighted
        let proj_total = evo + evd + pp + sh + gax + gsax + rookie;
        let raw_total = f("evo") + f("evd") + f("pp_raw") + f("sh_raw") + f("gax") + f("gsax") + f("rookie");
        out.insert(pid, json!({
            "player_id": pid,
            "player": str_value(p.get("name")),
            "position": str_value(p.get("position")),
            "team": str_value(p.get("team")),
            "projected_value": round6(proj_total),
            "raw_projected_value": round6(raw_total),
            "games_in_window": gp,
            "ig": round6(f("ig") * gp_weight),
            "a1": round6(f("a1") * gp_weight),
            "a2": round6(f("a2") * gp_weight),
        }));
    }
    out
}

/// `_load_gm_mode_projections_cached()`: all seasons (no season filter).
pub async fn load_gm_mode_projections_cached(state: &AppState) -> HashMap<i64, Value> {
    if let Some(v) = state.caches.gm_projections.get(&()) {
        return value_to_pid_map(&v);
    }
    let players = build_v2_player_projections(state, None).await;
    let mut out: HashMap<i64, Value> = HashMap::new();
    for p in players {
        let pid = safe_int(p.get("player_id")).unwrap_or(0);
        if pid <= 0 {
            continue;
        }
        let gp = safe_int(p.get("gp")).unwrap_or(0);
        let gp_weight = if gp > 0 && gp < 41 { gp as f64 / 41.0 } else { 1.0 };
        let f = |key: &str| parse_locale_float(p.get(key)).unwrap_or(0.0);
        let proj_total = f("evo") * gp_weight + f("evd") * gp_weight + f("pp_raw") * gp_weight
            + f("sh_raw") * gp_weight + f("gax") * gp_weight + f("gsax") * gp_weight + f("rookie");
        let raw_total = f("evo") + f("evd") + f("pp_raw") + f("sh_raw") + f("gax") + f("gsax") + f("rookie");
        out.insert(pid, json!({
            "player_id": pid,
            "player": str_value(p.get("name")),
            "position": str_value(p.get("position")),
            "team": str_value(p.get("team")),
            "projected_value": round6(proj_total),
            "raw_projected_value": round6(raw_total),
            "games_in_window": gp,
            "ig": round6(f("ig") * gp_weight),
            "a1": round6(f("a1") * gp_weight),
            "a2": round6(f("a2") * gp_weight),
        }));
    }
    let v = json!(out.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<serde_json::Map<_, _>>());
    state.caches.gm_projections.insert((), v);
    out
}

fn value_to_pid_map(v: &Value) -> HashMap<i64, Value> {
    let mut out = HashMap::new();
    if let Some(obj) = v.as_object() {
        for (k, val) in obj {
            if let Ok(pid) = k.parse::<i64>() {
                out.insert(pid, val.clone());
            }
        }
    }
    out
}

/// `_load_current_player_projections_cached()`: `player_current_projections`
/// for the current season + `preseason_updating` model.
pub async fn load_current_player_projections_cached(state: &AppState) -> HashMap<i64, Value> {
    if let Some(v) = state.caches.current_player_projections.get(&()) {
        return value_to_pid_map(&v);
    }
    let Some(sb) = state.sb.as_ref() else { return HashMap::new() };
    let season = current_season_id(None);
    let rows = sb
        .read(
            "player_current_projections",
            "season,player_id,position,raw_projected_value,projected_value,games_in_window,rookie_factor,source_player_id,source_game_id,model_key",
            Some(&filters(&[
                ("season", &format!("eq.{season}")),
                ("model_key", "eq.preseason_updating"),
            ])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    let mut out: HashMap<i64, Value> = HashMap::new();
    for r in &rows {
        if let Some(pid) = safe_int(r.get("player_id")) {
            if pid > 0 {
                out.insert(pid, r.clone());
            }
        }
    }
    let v = json!(out.iter().map(|(k, v)| (k.to_string(), v.clone())).collect::<serde_json::Map<_, _>>());
    state.caches.current_player_projections.insert((), v);
    out
}

/// `_proj_value_for_player(row)`.
pub fn proj_value_for_player(row: Option<&Value>) -> f64 {
    let row = match row {
        Some(r) => r,
        None => return 0.0,
    };
    if let Some(raw) = parse_locale_float(
        row.get("raw_projected_value")
            .or_else(|| row.get("rawProjectedValue"))
            .or_else(|| row.get("RawProjectedValue")),
    ) {
        return raw;
    }
    let f = |k: &str| -> f64 {
        parse_locale_float(row.get(k)).unwrap_or_else(|| {
            row.as_object()
                .and_then(|o| {
                    o.iter()
                        .find(|(kk, _)| kk.to_lowercase() == k.to_lowercase())
                        .and_then(|(_, vv)| parse_locale_float(Some(vv)))
                })
                .unwrap_or(0.0)
        })
    };
    f("Age") + f("Rookie") + f("EVO") + f("EVD") + f("PP") + f("SH") + f("GSAx")
}

/// `_team_proj_from_lineup(team, lineups_all, proj_map)`.
pub fn team_proj_from_lineup(team: &str, lineups_all: &Value, proj_map: &HashMap<i64, Value>) -> f64 {
    let t = team.to_uppercase();
    let li = lineups_all.get(&t).cloned().unwrap_or(Value::Null);
    let mut total = 0.0;
    for sec in ["forwards", "defense", "goalies"] {
        let arr = li.get(sec).and_then(|a| a.as_array()).cloned().unwrap_or_default();
        for it in &arr {
            let unit_val = str_value(it.get("unit")).to_uppercase();
            if unit_val == "EXT" {
                continue;
            }
            let pid = safe_int(it.get("playerId"));
            let pos = str_value(it.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_default();
            if pos == "G" {
                if !unit_val.is_empty() && unit_val != "G1" {
                    continue;
                }
            }
            if let Some(pid) = pid {
                total += proj_value_for_player(proj_map.get(&pid));
            } else {
                total += rookie_fallback(if pos.is_empty() { "F" } else { &pos });
            }
        }
    }
    total
}

/// `_team_proj_from_custom_lineup_entries(entries, proj_map)`.
pub fn team_proj_from_custom_lineup_entries(entries: &[Value], proj_map: &HashMap<i64, Value>) -> f64 {
    let mut total = 0.0;
    for entry in entries {
        let pid = safe_int(entry.get("pid")).unwrap_or(0);
        let pos = str_value(entry.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_else(|| "F".to_string());
        let games = normalize_games(entry.get("games"));
        let weight = games as f64 / 82.0;
        if pid > 0 {
            total += proj_value_for_player(proj_map.get(&pid)) * weight;
        } else {
            total += rookie_fallback(&pos) * weight;
        }
    }
    total
}

fn normalize_games(v: Option<&Value>) -> i64 {
    let games = safe_int(v).unwrap_or(84);
    games.clamp(0, 84)
}

/// `_normalize_custom_lineup_entries(raw_lineup)`.
pub fn normalize_custom_lineup_entries(raw: Option<&Value>) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(arr) = raw.and_then(|v| v.as_array()) else { return out };
    for entry in arr {
        let pid = safe_int(entry.get("pid")).unwrap_or(0);
        let mut pos = str_value(entry.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_else(|| "F".to_string());
        if !matches!(pos.as_str(), "F" | "D" | "G") {
            pos = "F".to_string();
        }
        if pid <= 0 {
            continue;
        }
        let games = normalize_games(entry.get("games"));
        let scratch = entry.get("scratch").and_then(|s| s.as_bool()).unwrap_or(false);
        out.push(json!({"pid": pid, "pos": pos, "games": games, "scratch": scratch}));
    }
    out
}

/// `_normalize_custom_lineups_by_team(raw_map)`.
pub fn normalize_custom_lineups_by_team(raw: Option<&Value>) -> BTreeMap<String, Vec<Value>> {
    let mut out = BTreeMap::new();
    let Some(obj) = raw.and_then(|v| v.as_object()) else { return out };
    for (team_raw, lineup_raw) in obj {
        let team = team_raw.to_uppercase();
        if team.is_empty() {
            continue;
        }
        out.insert(team, normalize_custom_lineup_entries(Some(lineup_raw)));
    }
    out
}

/// `_fetch_club_schedule_games(team, season)` — cached (6h).
pub async fn fetch_club_schedule_games(state: &AppState, team: &str, season: i64) -> Vec<Value> {
    let team = team.to_uppercase();
    let cache_key = (team.clone(), season);
    if let Some(v) = state.caches.club_schedule.get(&cache_key) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }

    // Try the real schedule for the requested season first (e.g. 20262027 is
    // now published by the NHL API). Only if it comes back empty do we fall
    // back to shifting the previous published season forward (the old
    // pre-publication behavior).
    let mut out = fetch_schedule_inner(state, &team, season).await;
    let mut shift_years = 0i64;
    if out.is_empty() && season >= 20262027 {
        shift_years = (season / 10000) - 2025;
        out = fetch_schedule_inner(state, &team, 20252026).await;
        if shift_years > 0 {
            let mut shifted = Vec::new();
            for g in &out {
                let mut g2 = g.clone();
                let d = str_value(g.get("date"));
                if d.len() >= 10 {
                    if let (Ok(y), Ok(m), Ok(dd)) = (
                        d[..4].parse::<i64>(),
                        d[5..7].parse::<i64>(),
                        d[8..10].parse::<i64>(),
                    ) {
                        g2["date"] = json!(format!("{:04}-{:02}-{:02}", y + shift_years, m, dd));
                    }
                }
                g2["id"] = json!(format!("{}_shift{shift_years}", str_value(g.get("id"))));
                shifted.push(g2);
            }
            out = shifted;
        }
    }
    state.caches.club_schedule.insert(cache_key, Value::Array(out.clone()));
    out
}

async fn fetch_schedule_inner(state: &AppState, team: &str, fetch_season: i64) -> Vec<Value> {
    let url = format!("{}/v1/club-schedule-season/{team}/{fetch_season}", crate::nhl::client::API_WEB);
    let mut out: Vec<Value> = Vec::new();
    if let Ok(js) = crate::nhl::client::get_json(&state.http, &url, 25).await {
        let games = js.get("games").and_then(|g| g.as_array()).cloned().unwrap_or_default();
        for g in games {
            let away_abbrev = g.get("awayTeam").and_then(|t| t.get("abbrev")).and_then(Value::as_str).map(|s| s.to_uppercase()).unwrap_or_default();
            let home_abbrev = g.get("homeTeam").and_then(|t| t.get("abbrev")).and_then(Value::as_str).map(|s| s.to_uppercase()).unwrap_or_default();
            let game_type = safe_int(g.get("gameType")).unwrap_or(0);
            let date_raw = str_value(g.get("gameDate").or_else(|| g.get("startTimeUTC")));
            let date_iso = if date_raw.len() >= 10 { date_raw[..10].to_string() } else { String::new() };
            if away_abbrev.is_empty() || home_abbrev.is_empty() || date_iso.is_empty() {
                continue;
            }
            out.push(json!({
                "id": g.get("id").or_else(|| g.get("gamePk")).or_else(|| g.get("gameId")).cloned(),
                "gameType": game_type,
                "status": str_value(g.get("gameState").or_else(|| g.get("gameStatus"))),
                "date": date_iso,
                "away": away_abbrev,
                "home": home_abbrev,
            }));
        }
    }
    out.sort_by(|a, b| {
        str_value(a.get("date")).cmp(&str_value(b.get("date")))
            .then_with(|| safe_int(a.get("id")).unwrap_or(0).cmp(&safe_int(b.get("id")).unwrap_or(0)))
    });
    out
}

/// `_is_team_b2b_on_date(games, date_iso)`.
pub fn is_team_b2b_on_date(games: &[Value], date_iso: &str) -> bool {
    if games.is_empty() || date_iso.is_empty() {
        return false;
    }
    let Ok(d) = chrono::NaiveDate::parse_from_str(&date_iso[..date_iso.len().min(10)], "%Y-%m-%d") else {
        return false;
    };
    let prev = (d - chrono::Duration::days(1)).format("%Y-%m-%d").to_string();
    games.iter().any(|g| str_value(g.get("date")) == prev)
}

/// `_projected_points_for_team(team, season, team_proj_map, ...)`.
pub async fn projected_points_for_team(
    state: &AppState,
    team: &str,
    season: i64,
    team_proj_map: &HashMap<String, f64>,
    lineup_entries: Option<&[Value]>,
    proj_map: Option<&HashMap<i64, Value>>,
    injuries: Option<&[Value]>,
) -> (f64, i64) {
    let team_games_all = fetch_club_schedule_games(state, team, season).await;
    let team_games: Vec<Value> = team_games_all
        .iter()
        .filter(|g| safe_int(g.get("gameType")).unwrap_or(0) == 2)
        .cloned()
        .collect();
    if team_games.is_empty() {
        return (0.0, 0);
    }

    let mut projected_points = 0.0;
    let mut usable_games = 0i64;

    for g in &team_games {
        let away = str_value(g.get("away")).to_uppercase();
        let home = str_value(g.get("home")).to_uppercase();
        let date_iso = str_value(g.get("date"));
        if away.is_empty() || home.is_empty() || date_iso.is_empty() {
            continue;
        }
        if away != team && home != team {
            continue;
        }
        let opp = if away == team { home.clone() } else { away.clone() };
        let mut team_proj = team_proj_map.get(team).copied().unwrap_or(0.0);
        let opp_proj = team_proj_map.get(&opp).copied().unwrap_or(0.0);

        if let (Some(injuries), Some(lineup_entries), Some(proj_map)) = (injuries, lineup_entries, proj_map) {
            team_proj = team_proj_with_injuries(team_proj, lineup_entries, proj_map, injuries, &date_iso);
        }

        let team_is_b2b = is_team_b2b_on_date(&team_games, &date_iso);
        let opp_games = fetch_club_schedule_games(state, &opp, season).await;
        let opp_is_b2b = is_team_b2b_on_date(&opp_games, &date_iso);
        let (home_b2b, away_b2b) = if away == team {
            (opp_is_b2b, team_is_b2b)
        } else {
            (team_is_b2b, opp_is_b2b)
        };

        let home_proj = if home == team { team_proj } else { opp_proj };
        let away_proj = if away == team { team_proj } else { opp_proj };
        let sit = v2_situation(home_b2b, away_b2b);
        let mu = V2_CONSERVATIVE_WEIGHT * (home_proj - away_proj + sit);
        let lg = v2_lg_avg(season);
        let gf_home = (lg + mu / 2.0).max(0.5);
        let gf_away = (lg - mu / 2.0).max(0.5);
        let sigma = (gf_home + gf_away).sqrt();
        let p_home_win = v2_norm_cdf(mu / sigma);
        let (home_pts, away_pts) = v2_expected_points_exact(gf_home, gf_away, p_home_win);
        let game_points = if home == team { home_pts } else { away_pts };
        projected_points += game_points.max(0.0);
        usable_games += 1;
    }
    (projected_points, usable_games)
}

/// `_team_proj_with_injuries(base_team_proj, ...)`.
pub fn team_proj_with_injuries(
    base_team_proj: f64,
    lineup_entries: &[Value],
    proj_map: &HashMap<i64, Value>,
    injuries: &[Value],
    date_iso: &str,
) -> f64 {
    if injuries.is_empty() || date_iso.is_empty() || lineup_entries.is_empty() {
        return base_team_proj;
    }
    let mut total_delta = 0.0;
    for entry in lineup_entries {
        let pid = safe_int(entry.get("pid")).unwrap_or(0);
        let pos = str_value(entry.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_else(|| "F".to_string());
        let games = normalize_games(entry.get("games"));
        let weight = games as f64 / 82.0;
        if pid <= 0 {
            continue;
        }
        for inj in injuries {
            if safe_int(inj.get("injured_pid")).unwrap_or(0) != pid {
                continue;
            }
            let s = str_value(inj.get("start_date"));
            let e = str_value(inj.get("end_date"));
            if !s.is_empty() && date_iso < s.as_str() {
                continue;
            }
            if !e.is_empty() && date_iso > e.as_str() {
                continue;
            }
            let repl_pid = safe_int(inj.get("replacement_pid")).unwrap_or(0);
            if repl_pid <= 0 {
                continue;
            }
            let orig_val = proj_value_for_player(proj_map.get(&pid));
            let repl_val = proj_value_for_player(proj_map.get(&repl_pid));
            total_delta += (repl_val - orig_val) * weight;
            break;
        }
    }
    base_team_proj + total_delta
}

/// `_normalize_injuries(raw)`.
pub fn normalize_injuries(raw: Option<&Value>) -> Vec<Value> {
    let mut out = Vec::new();
    let Some(arr) = raw.and_then(|v| v.as_array()) else { return out };
    for entry in arr {
        let injured = safe_int(entry.get("injuredPid").or_else(|| entry.get("injured_pid"))).unwrap_or(0);
        let replacement = safe_int(entry.get("replacementPid").or_else(|| entry.get("replacement_pid"))).unwrap_or(0);
        if injured <= 0 || replacement <= 0 {
            continue;
        }
        let start = str_value(entry.get("startDate").or_else(|| entry.get("start_date")));
        let end = str_value(entry.get("endDate").or_else(|| entry.get("end_date")));
        out.push(json!({
            "injured_pid": injured,
            "replacement_pid": replacement,
            "start_date": start.chars().take(10).collect::<String>(),
            "end_date": end.chars().take(10).collect::<String>(),
        }));
    }
    out
}

/// Team abbrevs from `TEAM_ROWS`.
pub fn all_team_abbrevs(state: &AppState) -> Vec<String> {
    let mut set: HashSet<String> = HashSet::new();
    for r in state.teams.iter() {
        let ab = str_value(r.get("Team")).to_uppercase();
        if !ab.is_empty() {
            set.insert(ab);
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

/// `_active_team_abbrevs()`: only `Active == '1'`.
pub fn active_team_abbrevs(state: &AppState) -> Vec<String> {
    let mut set: HashSet<String> = HashSet::new();
    for r in state.teams.iter() {
        let ab = str_value(r.get("Team")).to_uppercase();
        if !ab.is_empty() && str_value(r.get("Active")) == "1" {
            set.insert(ab);
        }
    }
    let mut v: Vec<String> = set.into_iter().collect();
    v.sort();
    v
}

// ── Simulation engine ──

/// `_b2b_date_sets(team_games)`.
pub fn b2b_date_sets(team_games: &HashMap<String, Vec<Value>>) -> HashMap<String, HashSet<String>> {
    let mut out = HashMap::new();
    for (team, games) in team_games {
        let reg: Vec<&Value> = games.iter().filter(|g| safe_int(g.get("gameType")).unwrap_or(0) == 2).collect();
        let dates: HashSet<String> = reg.iter().filter_map(|g| {
            let d = str_value(g.get("date"));
            if d.is_empty() { None } else { Some(d[..d.len().min(10)].to_string()) }
        }).collect();
        let mut b2b = HashSet::new();
        for g in &reg {
            let d = str_value(g.get("date"));
            if d.len() < 10 {
                continue;
            }
            if let Ok(dd) = chrono::NaiveDate::parse_from_str(&d[..10], "%Y-%m-%d") {
                let prev = (dd - chrono::Duration::days(1)).format("%Y-%m-%d").to_string();
                if dates.contains(&prev) {
                    b2b.insert(d[..10].to_string());
                }
            }
        }
        out.insert(team.clone(), b2b);
    }
    out
}

/// `_poisson_draw(lam, rng)` — Knuth's algorithm.
pub fn poisson_draw(lam: f64, rng: &mut Mt19937) -> i64 {
    if lam <= 0.0 {
        return 0;
    }
    let l = (-lam).exp();
    let mut k = 0i64;
    let mut p = 1.0;
    loop {
        k += 1;
        p *= rng.next_f64();
        if p <= l {
            return k - 1;
        }
    }
}

/// `_weighted_choice(weights, rng)`.
pub fn weighted_choice(weights: &[f64], rng: &mut Mt19937) -> usize {
    let total: f64 = weights.iter().sum();
    if total <= 0.0 {
        return rng.randrange(weights.len());
    }
    let r = rng.next_f64() * total;
    let mut cumulative = 0.0;
    for (i, w) in weights.iter().enumerate() {
        cumulative += w;
        if r <= cumulative {
            return i;
        }
    }
    weights.len() - 1
}

/// `_simulate_goal_scorers(roster, num_goals, rng)`.
pub fn simulate_goal_scorers(roster: &[Value], num_goals: i64, rng: &mut Mt19937) -> Vec<Value> {
    let mut out = Vec::new();
    if roster.is_empty() || num_goals <= 0 {
        return out;
    }
    let scorer_weights: Vec<f64> = roster.iter().map(|p| parse_locale_float(p.get("ig")).unwrap_or(0.0)).collect();
    let a1_base: Vec<f64> = roster.iter().map(|p| parse_locale_float(p.get("a1")).unwrap_or(0.0)).collect();
    let a2_base: Vec<f64> = roster.iter().map(|p| parse_locale_float(p.get("a2")).unwrap_or(0.0)).collect();
    let n = roster.len();
    for _ in 0..num_goals {
        let si = weighted_choice(&scorer_weights, rng);
        let scorer_pid = safe_int(roster[si].get("pid")).unwrap_or(0);
        let r = rng.next_f64();
        let num_assists = if r < SIM_ASSIST_TWO {
            2
        } else if r < SIM_ASSIST_TWO + SIM_ASSIST_ONE {
            1
        } else {
            0
        };
        let mut a1_pid: Option<i64> = None;
        let mut a2_pid: Option<i64> = None;
        if num_assists >= 1 {
            let w1: Vec<f64> = (0..n).map(|j| if j != si { a1_base[j] } else { 0.0 }).collect();
            a1_pid = safe_int(roster[weighted_choice(&w1, rng)].get("pid"));
        }
        if num_assists >= 2 {
            let w2: Vec<f64> = (0..n)
                .map(|j| {
                    let pidj = safe_int(roster[j].get("pid")).unwrap_or(0);
                    if j != si && Some(pidj) != a1_pid {
                        a2_base[j]
                    } else {
                        0.0
                    }
                })
                .collect();
            a2_pid = safe_int(roster[weighted_choice(&w2, rng)].get("pid"));
        }
        out.push(json!({"scorer": scorer_pid, "a1": a1_pid, "a2": a2_pid}));
    }
    out
}

/// `_build_team_roster_rates(team, lineups_all, proj_map, custom_lineup)`.
pub fn build_team_roster_rates(
    team: &str,
    lineups_all: &Value,
    proj_map: &HashMap<i64, Value>,
    custom_lineup: Option<&[Value]>,
) -> Vec<Value> {
    let t = team.to_uppercase();
    let mut roster: Vec<Value> = Vec::new();
    let mut seen: HashSet<i64> = HashSet::new();

    let mut add_player = |pid: i64, pos: &str, games_est: i64, roster: &mut Vec<Value>, seen: &mut HashSet<i64>| {
        if pid <= 0 || seen.contains(&pid) {
            return;
        }
        seen.insert(pid);
        let row = proj_map.get(&pid);
        let mut p = pos.to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_default();
        if !matches!(p.as_str(), "F" | "D") {
            if let Some(row) = row {
                p = str_value(row.get("position")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_else(|| "F".to_string());
            }
            if !matches!(p.as_str(), "F" | "D") {
                p = "F".to_string();
            }
        }
        let (ig_floor, a_floor) = if p == "D" {
            (SIM_FLOOR_D_IG, SIM_FLOOR_D_A)
        } else {
            (SIM_FLOOR_F_IG, SIM_FLOOR_F_A)
        };
        let (ig, a1, a2) = match row {
            Some(row) => (
                parse_locale_float(row.get("ig")).unwrap_or(0.0),
                parse_locale_float(row.get("a1")).unwrap_or(0.0),
                parse_locale_float(row.get("a2")).unwrap_or(0.0),
            ),
            None => (0.0, 0.0, 0.0),
        };
        let ig = ig.max(ig_floor);
        let a1 = a1.max(a_floor);
        let a2 = a2.max(a_floor);
        let gp_weight = (games_est as f64 / 82.0).clamp(0.0, 1.0);
        let ig = ig * gp_weight;
        let a1 = a1 * gp_weight;
        let a2 = a2 * gp_weight;
        if ig <= 0.0 && a1 <= 0.0 && a2 <= 0.0 {
            return;
        }
        roster.push(json!({"pid": pid, "pos": p, "ig": ig, "a1": a1, "a2": a2}));
    };

    if let Some(custom) = custom_lineup {
        for entry in custom {
            let pid = safe_int(entry.get("pid")).unwrap_or(0);
            let pos = str_value(entry.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_else(|| "F".to_string());
            if pid <= 0 || pos == "G" {
                continue;
            }
            let games_est = normalize_games(entry.get("games"));
            add_player(pid, &pos, games_est, &mut roster, &mut seen);
        }
    } else {
        let li = lineups_all.get(&t).cloned().unwrap_or(Value::Null);
        for sec in ["forwards", "defense"] {
            let arr = li.get(sec).and_then(|a| a.as_array()).cloned().unwrap_or_default();
            for it in &arr {
                if str_value(it.get("unit")).to_uppercase() == "EXT" {
                    continue;
                }
                let pid = safe_int(it.get("playerId")).unwrap_or(0);
                let pos = str_value(it.get("pos")).to_uppercase().chars().next().map(|c| c.to_string()).unwrap_or_default();
                if pid > 0 && pos != "G" {
                    add_player(pid, &pos, 82, &mut roster, &mut seen);
                }
            }
        }
    }
    roster
}

/// `_standings_from_results(teams, results)`.
pub fn standings_from_results(teams: &[String], results: &[Value]) -> Vec<Value> {
    let mut pts: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut wins: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut losses: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut otl: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut gf: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut ga: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();
    let mut gp: HashMap<&str, i64> = teams.iter().map(|t| (t.as_str(), 0)).collect();

    for g in results {
        let h = str_value(g.get("home"));
        let a = str_value(g.get("away"));
        if !pts.contains_key(h.as_str()) || !pts.contains_key(a.as_str()) {
            continue;
        }
        *gp.get_mut(h.as_str()).unwrap() += 1;
        *gp.get_mut(a.as_str()).unwrap() += 1;
        *pts.get_mut(h.as_str()).unwrap() += safe_int(g.get("homePoints")).unwrap_or(0);
        *pts.get_mut(a.as_str()).unwrap() += safe_int(g.get("awayPoints")).unwrap_or(0);
        let w = str_value(g.get("winner"));
        let l = str_value(g.get("loser"));
        *wins.get_mut(w.as_str()).unwrap_or(&mut 0) += 1;
        if g.get("ot").and_then(|o| o.as_bool()).unwrap_or(false) {
            *otl.get_mut(l.as_str()).unwrap_or(&mut 0) += 1;
        } else {
            *losses.get_mut(l.as_str()).unwrap_or(&mut 0) += 1;
        }
        let mut gh = safe_int(g.get("homeGoals")).unwrap_or(0);
        let mut gav = safe_int(g.get("awayGoals")).unwrap_or(0);
        if gh == 0 && gav == 0 {
            if w == h {
                gh = 3;
                gav = 2;
            } else {
                gh = 2;
                gav = 3;
            }
        }
        *gf.get_mut(h.as_str()).unwrap() += gh;
        *ga.get_mut(h.as_str()).unwrap() += gav;
        *gf.get_mut(a.as_str()).unwrap() += gav;
        *ga.get_mut(a.as_str()).unwrap() += gh;
    }

    let mut rows: Vec<Value> = Vec::new();
    for t in teams {
        rows.push(json!({
            "team": t,
            "conference": team_conf(t),
            "gp": gp.get(t.as_str()).copied().unwrap_or(0),
            "wins": wins.get(t.as_str()).copied().unwrap_or(0),
            "losses": losses.get(t.as_str()).copied().unwrap_or(0),
            "otLosses": otl.get(t.as_str()).copied().unwrap_or(0),
            "points": pts.get(t.as_str()).copied().unwrap_or(0),
            "goalsFor": gf.get(t.as_str()).copied().unwrap_or(0),
            "goalsAgainst": ga.get(t.as_str()).copied().unwrap_or(0),
            "goalDifferential": gf.get(t.as_str()).copied().unwrap_or(0) - ga.get(t.as_str()).copied().unwrap_or(0),
            "regulationWins": wins.get(t.as_str()).copied().unwrap_or(0),
        }));
    }
    rows.sort_by(|a, b| {
        safe_int(a.get("points")).unwrap_or(0).cmp(&safe_int(b.get("points")).unwrap_or(0)).reverse()
            .then_with(|| safe_int(a.get("wins")).unwrap_or(0).cmp(&safe_int(b.get("wins")).unwrap_or(0)).reverse())
            .then_with(|| safe_int(a.get("goalsFor")).unwrap_or(0).cmp(&safe_int(b.get("goalsFor")).unwrap_or(0)).reverse())
            .then_with(|| str_value(a.get("team")).cmp(&str_value(b.get("team"))))
    });
    rows
}

/// `_seed_playoffs(standings)`.
pub fn seed_playoffs(standings: &[Value]) -> (Vec<String>, Vec<String>) {
    let east: Vec<String> = standings.iter().filter(|r| str_value(r.get("conference")) == "East").map(|r| str_value(r.get("team"))).take(8).collect();
    let west: Vec<String> = standings.iter().filter(|r| str_value(r.get("conference")) == "West").map(|r| str_value(r.get("team"))).take(8).collect();
    (east, west)
}

/// `_simulate_series(top, bottom, team_proj_map, season, rng)`.
pub fn simulate_series(top: &str, bottom: &str, team_proj_map: &HashMap<String, f64>, season: i64, rng: &mut Mt19937) -> Value {
    let mut top_wins = 0;
    let mut bottom_wins = 0;
    let mut games_played = 0;
    let home_pattern = [top, top, bottom, bottom, top, bottom, top];
    while top_wins < 4 && bottom_wins < 4 {
        let home = home_pattern[games_played];
        let away = if home == top { bottom } else { top };
        let p_home = v2_win_probability(
            team_proj_map.get(home).copied().unwrap_or(0.0),
            team_proj_map.get(away).copied().unwrap_or(0.0),
            false, false, season,
        );
        let winner = if rng.next_f64() < p_home { home } else { away };
        if winner == top {
            top_wins += 1;
        } else {
            bottom_wins += 1;
        }
        games_played += 1;
    }
    let winner = if top_wins == 4 { top } else { bottom };
    let loser = if winner == top { bottom } else { top };
    json!({
        "top": top, "bottom": bottom, "winner": winner, "loser": loser,
        "topWins": top_wins, "bottomWins": bottom_wins, "games": games_played,
    })
}

/// `_simulate_playoffs(seeds, team_proj_map, season, rng)`.
pub fn simulate_playoffs(
    seeds: (Vec<String>, Vec<String>),
    team_proj_map: &HashMap<String, f64>,
    season: i64,
    rng: &mut Mt19937,
) -> Value {
    let (east, west) = seeds;
    let mut conf_round = |team_list: &[String]| -> Vec<Value> {
        vec![
            simulate_series(&team_list[0], &team_list[7], team_proj_map, season, rng),
            simulate_series(&team_list[1], &team_list[6], team_proj_map, season, rng),
            simulate_series(&team_list[2], &team_list[5], team_proj_map, season, rng),
            simulate_series(&team_list[3], &team_list[4], team_proj_map, season, rng),
        ]
    };
    let east_r1 = conf_round(&east);
    let west_r1 = conf_round(&west);
    let reseeds = |round_results: &[Value], original: &[String]| -> Vec<String> {
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, t) in original.iter().enumerate() {
            idx.insert(t.as_str(), i);
        }
        let mut remain: Vec<String> = round_results.iter().map(|s| str_value(s.get("winner"))).collect();
        remain.sort_by_key(|t| idx.get(t.as_str()).copied().unwrap_or(99));
        remain
    };

    let east_remain = reseeds(&east_r1, &east);
    let west_remain = reseeds(&west_r1, &west);

    let east_r2 = vec![
        simulate_series(&east_remain[0], &east_remain[3], team_proj_map, season, rng),
        simulate_series(&east_remain[1], &east_remain[2], team_proj_map, season, rng),
    ];
    let west_r2 = vec![
        simulate_series(&west_remain[0], &west_remain[3], team_proj_map, season, rng),
        simulate_series(&west_remain[1], &west_remain[2], team_proj_map, season, rng),
    ];

    let east_final_seeds = reseeds(&east_r2, &east_remain);
    let west_final_seeds = reseeds(&west_r2, &west_remain);
    let east_conf_final = simulate_series(&east_final_seeds[0], &east_final_seeds[1], team_proj_map, season, rng);
    let west_conf_final = simulate_series(&west_final_seeds[0], &west_final_seeds[1], team_proj_map, season, rng);

    let stanley = simulate_series(
        &str_value(east_conf_final.get("winner")),
        &str_value(west_conf_final.get("winner")),
        team_proj_map, season, rng,
    );

    json!({
        "round1": {"East": east_r1, "West": west_r1},
        "round2": {"East": east_r2, "West": west_r2},
        "conferenceFinals": {"East": east_conf_final, "West": west_conf_final},
        "stanleyFinal": stanley,
        "champion": stanley.get("winner"),
    })
}

/// `_run_single_sim(...)`.
#[allow(clippy::too_many_arguments)]
pub async fn run_single_sim(
    state: &AppState,
    schedule: &[Value],
    team_proj_map: &HashMap<String, f64>,
    b2b_sets: &HashMap<String, HashSet<String>>,
    team_rosters: &HashMap<String, Vec<Value>>,
    teams: &[String],
    proj_map: &HashMap<i64, Value>,
    season: i64,
    rng: &mut Mt19937,
    injuries_by_team: Option<&HashMap<String, Vec<Value>>>,
    custom_lineups: Option<&BTreeMap<String, Vec<Value>>>,
) -> Value {
    let lg = v2_lg_avg(season);
    let sqrt2 = std::f64::consts::SQRT_2;
    let mut results: Vec<Value> = Vec::new();
    for g in schedule {
        let home = str_value(g.get("home")).to_uppercase();
        let away = str_value(g.get("away")).to_uppercase();
        let date_iso = str_value(g.get("date"));
        if home.is_empty() || away.is_empty() || !team_proj_map.contains_key(&home) || !team_proj_map.contains_key(&away) {
            continue;
        }
        let mut home_proj = team_proj_map.get(&home).copied().unwrap_or(0.0);
        let mut away_proj = team_proj_map.get(&away).copied().unwrap_or(0.0);

        if let (Some(inj), Some(cl)) = (injuries_by_team, custom_lineups) {
            if let Some(entries) = cl.get(&home) {
                if let Some(injs) = inj.get(&home) {
                    home_proj = team_proj_with_injuries(home_proj, entries, proj_map, injs, &date_iso);
                }
            }
            if let Some(entries) = cl.get(&away) {
                if let Some(injs) = inj.get(&away) {
                    away_proj = team_proj_with_injuries(away_proj, entries, proj_map, injs, &date_iso);
                }
            }
        }
        let hb = if b2b_sets.get(&home).map(|s| s.contains(&date_iso)).unwrap_or(false) { 1 } else { 0 };
        let ab = if b2b_sets.get(&away).map(|s| s.contains(&date_iso)).unwrap_or(false) { 1 } else { 0 };
        let sit = v2_situation(hb == 1, ab == 1);
        let mu = V2_CONSERVATIVE_WEIGHT * (home_proj - away_proj + sit);
        let gf_home = (lg + mu / 2.0).max(0.5);
        let gf_away = (lg - mu / 2.0).max(0.5);
        let sigma = (gf_home + gf_away).sqrt();
        let p_home = 0.5 * (1.0 + libm::erf(mu / (sigma * sqrt2)));
        let gh = poisson_draw(gf_home, rng);
        let ga = poisson_draw(gf_away, rng);
        let (winner, loser, mut ot) = if gh > ga {
            (home.clone(), away.clone(), false)
        } else if ga > gh {
            (away.clone(), home.clone(), false)
        } else {
            if rng.next_f64() < p_home {
                (home.clone(), away.clone(), true)
            } else {
                (away.clone(), home.clone(), true)
            }
        };
        if !ot && (gh - ga).abs() == 1 && rng.next_f64() < 0.30 {
            ot = true;
        }
        let home_scorers = simulate_goal_scorers(team_rosters.get(&home).map(|r| r.as_slice()).unwrap_or(&[]), gh, rng);
        let away_scorers = simulate_goal_scorers(team_rosters.get(&away).map(|r| r.as_slice()).unwrap_or(&[]), ga, rng);
        results.push(json!({
            "home": home, "away": away, "winner": winner, "loser": loser,
            "ot": ot,
            "homeGoals": gh, "awayGoals": ga,
            "homePoints": if winner == home { 2 } else if ot { 1 } else { 0 },
            "awayPoints": if winner == away { 2 } else if ot { 1 } else { 0 },
            "homeScorers": home_scorers, "awayScorers": away_scorers,
        }));
    }

    // Player stats.
    let mut gp_by_pid: HashMap<i64, i64> = HashMap::new();
    if let Some(cl) = custom_lineups {
        for (_t, entries) in cl {
            for e in entries {
                if let Some(pid) = safe_int(e.get("pid")) {
                    if pid > 0 {
                        let g = safe_int(e.get("games")).unwrap_or(0).clamp(0, 84);
                        let cur = gp_by_pid.entry(pid).or_insert(0);
                        *cur = (*cur).max(g);
                    }
                }
            }
        }
    }
    let mut player_stats: BTreeMap<i64, Value> = BTreeMap::new();
    for g in &results {
        for (scorers_list, team_abbr) in [
            (g.get("homeScorers").and_then(|v| v.as_array()).cloned().unwrap_or_default(), str_value(g.get("home"))),
            (g.get("awayScorers").and_then(|v| v.as_array()).cloned().unwrap_or_default(), str_value(g.get("away"))),
        ] {
            for sg in &scorers_list {
                for (role, pid) in [
                    ("scorer", safe_int(sg.get("scorer"))),
                    ("a1", safe_int(sg.get("a1"))),
                    ("a2", safe_int(sg.get("a2"))),
                ] {
                    let Some(pid) = pid else { continue };
                    let entry = player_stats.entry(pid).or_insert_with(|| {
                        let row = proj_map.get(&pid).cloned().unwrap_or(Value::Null);
                        let row_team = str_value(row.get("team"));
                        let team = if row_team.is_empty() {
                            team_abbr.clone()
                        } else {
                            row_team
                        };
                        json!({
                            "pid": pid,
                            "name": str_value(row.get("player")),
                            "team": team,
                            "position": str_value(row.get("position")),
                            "gp": gp_by_pid.get(&pid).copied().unwrap_or(82),
                            "goals": 0, "a1": 0, "a2": 0, "points": 0,
                        })
                    });
                    match role {
                        "scorer" => {
                            entry["goals"] = json!(safe_int(entry.get("goals")).unwrap_or(0) + 1);
                            entry["points"] = json!(safe_int(entry.get("points")).unwrap_or(0) + 1);
                        }
                        "a1" => {
                            entry["a1"] = json!(safe_int(entry.get("a1")).unwrap_or(0) + 1);
                            entry["points"] = json!(safe_int(entry.get("points")).unwrap_or(0) + 1);
                        }
                        "a2" => {
                            entry["a2"] = json!(safe_int(entry.get("a2")).unwrap_or(0) + 1);
                            entry["points"] = json!(safe_int(entry.get("points")).unwrap_or(0) + 1);
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let standings = standings_from_results(teams, &results);
    let (mut east, mut west) = seed_playoffs(&standings);
    if east.len() < 8 || west.len() < 8 {
        let top16: Vec<String> = standings.iter().take(16).map(|r| str_value(r.get("team"))).collect();
        east = top16[..top16.len().min(8)].to_vec();
        west = if top16.len() > 8 { top16[8..].to_vec() } else { Vec::new() };
    }
    let playoffs = simulate_playoffs((east.clone(), west.clone()), team_proj_map, season, rng);

    json!({
        "standings": standings,
        "playoffSeeds": {"East": east, "West": west},
        "playoffs": playoffs,
        "playerStats": player_stats.into_values().collect::<Vec<_>>(),
    })
}

/// `_team_proj_map_for_season(...)`.
pub fn team_proj_map_for_season(
    lineups_all: &Value,
    proj_map: &HashMap<i64, Value>,
    custom_lineups_by_team: &BTreeMap<String, Vec<Value>>,
    state: &AppState,
) -> HashMap<String, f64> {
    let mut out = HashMap::new();
    for team in all_team_abbrevs(state) {
        if let Some(custom) = custom_lineups_by_team.get(&team) {
            out.insert(team.clone(), team_proj_from_custom_lineup_entries(custom, proj_map));
        } else {
            out.insert(team.clone(), team_proj_from_lineup(&team, lineups_all, proj_map));
        }
    }
    out
}
