//! M2b Line Tool routes. Ports of `api_line_tool_{players,data,wowy,versus,lines}`
//! plus their helpers (`_get_lt_shifts`, `_get_lt_pbp`, `_get_lt_base_context`,
//! `_compute_line_tool_team_combos`, ...) from `app/routes.py`.

use std::collections::{HashMap, HashSet};
use std::env;

use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::data::players as players_data;
use crate::state::{AppState, Caches};
use crate::supabase::read::{filters, SbClient};
use crate::util::parse::{parse_locale_float, safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/line-tool/players", get(api_line_tool_players))
        .route("/api/line-tool/data", get(api_line_tool_data))
        .route("/api/line-tool/wowy", get(api_line_tool_wowy))
        .route("/api/line-tool/versus", get(api_line_tool_versus))
        .route("/api/line-tool/lines", get(api_line_tool_lines))
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

fn i_of(v: &Value, key: &str) -> i64 {
    safe_int(v.get(key)).unwrap_or(0)
}

fn f_of(v: &Value, key: &str) -> f64 {
    parse_locale_float(v.get(key)).unwrap_or(0.0)
}

fn s_of(v: &Value, key: &str) -> String {
    str_value(v.get(key))
}

fn r1(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

fn r4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

/// `_parse_request_season_ids` (without a default).
fn season_ids_from_str(s: &str) -> Vec<i64> {
    let mut out: Vec<i64> = s
        .split(',')
        .filter_map(|x| x.trim().parse::<i64>().ok())
        .filter(|x| *x > 0)
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

/// `_filter_shifts_season_state`: game_id digits 5-6 (02=reg, 03=playoff).
fn filter_shifts_season_state(rows: Vec<Value>, ss: &str) -> Vec<Value> {
    let allowed: HashSet<&str> = if ss.is_empty() || ss == "all" {
        ["02", "03"].iter().copied().collect()
    } else {
        match ss {
            "regular" => ["02"].iter().copied().collect(),
            "playoffs" => ["03"].iter().copied().collect(),
            _ => return rows,
        }
    };
    rows.into_iter()
        .filter(|s| {
            let gid = s_of(s, "game_id");
            gid.len() >= 6 && allowed.contains(&gid[4..6])
        })
        .collect()
}

/// `_apply_lt_strength_filter`.
fn apply_lt_strength_filter(rows: Vec<Value>, strength: &str) -> Vec<Value> {
    let strength = strength.trim();
    if strength.is_empty() || strength.eq_ignore_ascii_case("all") {
        return rows;
    }
    let strength_sets: HashMap<&str, &[&str]> = HashMap::from([
        ("5v5", &["5v5"][..]),
        ("PP", &["5v4", "5v3", "4v3"][..]),
        ("SH", &["4v5", "3v5", "3v4"][..]),
    ]);
    if let Some(allowed) = strength_sets.get(strength) {
        return rows
            .into_iter()
            .filter(|s| allowed.contains(&s_of(s, "strength_state").as_str()))
            .collect();
    }
    if strength == "Other" {
        let all_special: HashSet<&str> =
            ["5v5", "5v4", "5v3", "4v3", "4v5", "3v5", "3v4"].iter().copied().collect();
        return rows
            .into_iter()
            .filter(|s| !all_special.contains(s_of(s, "strength_state").as_str()))
            .collect();
    }
    rows
}

/// `_get_lt_shifts(team, season)` — cached (1800s, max 40).
async fn get_lt_shifts(caches: &Caches, sb: &SbClient, team: &str, season: i64) -> Vec<Value> {
    let key = format!("{team}|{season}");
    if let Some(v) = caches.lt_shifts.get(&key) {
        if let Value::Array(r) = v {
            return r;
        }
    }
    let rows = sb
        .read(
            "shifts",
            "shift_index,game_id,player_id,duration,strength_state",
            Some(&filters(&[
                ("team", &format!("eq.{team}")),
                ("season", &format!("eq.{season}")),
            ])),
            None,
            Some("shift_index"),
            0,
        )
        .await
        .unwrap_or_default();
    if !rows.is_empty() {
        caches.lt_shifts.insert(key, Value::Array(rows.clone()));
    }
    rows
}

/// `_get_lt_pbp(season, game_ids, xg_col, extra_cols)` — cached (1800s, max 10).
async fn get_lt_pbp(
    caches: &Caches,
    sb: &SbClient,
    season: i64,
    game_ids: &[i64],
    xg_col: &str,
    extra_cols: &str,
) -> Vec<Value> {
    let mut sorted: Vec<i64> = game_ids.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let gid_key = sorted
        .iter()
        .map(|g| g.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let key = format!("{season}|{gid_key}|{xg_col}|{extra_cols}");
    if let Some(v) = caches.lt_pbp.get(&key) {
        if let Value::Array(r) = v {
            return r;
        }
    }
    let base_cols = if extra_cols.is_empty() {
        format!(
            "game_id,shift_index,event_team,opponent,shot,goal,fenwick,corsi,{xg_col},strength_state,period,season_state"
        )
    } else {
        format!(
            "game_id,shift_index,event_team,opponent,shot,goal,fenwick,corsi,{xg_col},strength_state,period,season_state,{extra_cols}"
        )
    };
    let batch_size = env::var("LINE_TOOL_PBP_BATCH_SIZE")
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
        .unwrap_or(50)
        .clamp(10, 100);
    let mut all: Vec<Value> = Vec::new();
    for chunk in sorted.chunks(batch_size) {
        let gid_filter = chunk
            .iter()
            .map(|g| g.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let rows = sb
            .read(
                "pbp",
                &base_cols,
                Some(&filters(&[
                    ("season", &format!("eq.{season}")),
                    ("game_id", &format!("in.({gid_filter})")),
                    ("corsi", "eq.1"),
                ])),
                None,
                None,
                0,
            )
            .await
            .unwrap_or_default();
        all.extend(rows);
    }
    if !all.is_empty() {
        caches.lt_pbp.insert(key, Value::Array(all.clone()));
    }
    all
}

/// `_get_lt_shifts_parallel`: per-season cache check, single `in.(seasons)` fetch.
async fn get_lt_shifts_parallel(
    caches: &Caches,
    sb: &SbClient,
    team: &str,
    season_ids: &[i64],
) -> Vec<Value> {
    let mut sorted_sids: Vec<i64> = season_ids.to_vec();
    sorted_sids.sort_unstable();
    sorted_sids.dedup();
    if sorted_sids.is_empty() {
        return Vec::new();
    }
    if sorted_sids.len() == 1 {
        return get_lt_shifts(caches, sb, team, sorted_sids[0]).await;
    }

    let mut all_rows: Vec<Value> = Vec::new();
    let mut missing: Vec<i64> = Vec::new();
    for sid in &sorted_sids {
        let key = format!("{team}|{sid}");
        if let Some(v) = caches.lt_shifts.get(&key) {
            if let Value::Array(r) = v {
                all_rows.extend(r);
                continue;
            }
        }
        missing.push(*sid);
    }
    if missing.is_empty() {
        return all_rows;
    }

    let season_list = missing
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let new_rows = sb
        .read(
            "shifts",
            "shift_index,game_id,player_id,duration,strength_state,season",
            Some(&filters(&[
                ("team", &format!("eq.{team}")),
                ("season", &format!("in.({season_list})")),
            ])),
            None,
            Some("shift_index"),
            0,
        )
        .await
        .unwrap_or_default();

    let mut by_sid: HashMap<i64, Vec<Value>> = HashMap::new();
    for r in &new_rows {
        let sid_val = safe_int(r.get("season"));
        if let Some(sid) = sid_val {
            if missing.contains(&sid) {
                // Strip the season column before caching.
                let mut stripped = r.clone();
                if let Value::Object(m) = &mut stripped {
                    m.remove("season");
                }
                by_sid.entry(sid).or_default().push(stripped);
            }
        }
    }

    for sid in &missing {
        let season_rows = by_sid.get(sid).cloned().unwrap_or_default();
        if !season_rows.is_empty() {
            let cache_key = format!("{team}|{sid}");
            caches.lt_shifts.insert(cache_key, Value::Array(season_rows.clone()));
        }
        all_rows.extend(season_rows);
    }
    all_rows
}

/// `_get_lt_pbp_parallel`.
async fn get_lt_pbp_parallel(
    caches: &Caches,
    sb: &SbClient,
    season_ids: &[i64],
    game_list: &[i64],
    xg_col: &str,
    extra_cols: &str,
) -> Vec<Value> {
    let mut all: Vec<Value> = Vec::new();
    let mut sids: Vec<i64> = season_ids.to_vec();
    sids.sort_unstable();
    sids.dedup();
    for sid in sids {
        let sid_prefix = sid.to_string();
        let gids: Vec<i64> = game_list
            .iter()
            .copied()
            .filter(|g| {
                let gs = g.to_string();
                gs.len() >= 8 && gs[..4] == sid_prefix[..4]
            })
            .collect();
        if !gids.is_empty() {
            all.extend(get_lt_pbp(caches, sb, sid, &gids, xg_col, extra_cols).await);
        }
    }
    all
}

/// `_get_lt_base_context(team, season_ids, player_ids, ss, strength, xg_col)`.
#[allow(clippy::too_many_arguments)]
async fn get_lt_base_context(
    caches: &Caches,
    sb: &SbClient,
    team: &str,
    season_ids: &[i64],
    player_ids: &[String],
    ss: &str,
    strength: &str,
    xg_col: &str,
) -> Value {
    let cache_key = json!({
        "base-context": "",
        "team": team,
        "seasons": season_ids,
        "players": player_ids,
        "ss": ss,
        "strength": strength,
        "xg_col": xg_col,
    })
    .to_string();
    if let Some(v) = caches.lt_base.get(&cache_key) {
        return v;
    }

    let shift_rows = get_lt_shifts_parallel(caches, sb, team, season_ids).await;
    if shift_rows.is_empty() {
        let result = json!({"shiftRows": [], "baseShifts": [], "allPbp": []});
        caches.lt_base.insert(cache_key, result.clone());
        return result;
    }

    let shift_rows = filter_shifts_season_state(shift_rows, ss);
    let shift_rows = apply_lt_strength_filter(shift_rows, strength);

    let base_shifts: Vec<Value> = if player_ids.is_empty() {
        shift_rows.clone()
    } else {
        let player_id_set: HashSet<String> = player_ids.iter().cloned().collect();
        shift_rows
            .iter()
            .filter(|s| {
                let pids_str = s_of(s, "player_id");
                let pids: HashSet<&str> = pids_str.split_whitespace().collect();
                player_id_set.iter().all(|p| pids.contains(p.as_str()))
            })
            .cloned()
            .collect()
    };

    let mut game_list: Vec<i64> = base_shifts
        .iter()
        .map(|s| i_of(s, "game_id"))
        .filter(|g| *g > 0)
        .collect();
    game_list.sort_unstable();
    game_list.dedup();

    let all_pbp = if game_list.is_empty() {
        Vec::new()
    } else {
        get_lt_pbp_parallel(caches, sb, season_ids, &game_list, xg_col, "x,y,box_id,highlight_url").await
    };

    let result = json!({
        "shiftRows": shift_rows,
        "baseShifts": base_shifts,
        "allPbp": all_pbp,
    });
    caches.lt_base.insert(cache_key, result.clone());
    result
}

/// `_lt_game_ids_for_opponent(team, vs_team, pbp_rows)`.
fn lt_game_ids_for_opponent(team: &str, vs_team: &str, pbp_rows: &[Value]) -> HashSet<i64> {
    let mut game_ids: HashSet<i64> = HashSet::new();
    let team_abbr = team.trim().to_uppercase();
    let opponent_abbr = vs_team.trim().to_uppercase();
    if team_abbr.is_empty() || opponent_abbr.is_empty() {
        return game_ids;
    }
    for row in pbp_rows {
        let gid = safe_int(row.get("game_id"));
        let Some(gid) = gid else { continue };
        let event_team = s_of(row, "event_team").to_uppercase();
        let opponent = s_of(row, "opponent").to_uppercase();
        if (event_team == team_abbr && opponent == opponent_abbr)
            || (event_team == opponent_abbr && opponent == team_abbr)
        {
            game_ids.insert(gid);
        }
    }
    game_ids
}

/// `_accumulate_line_tool_combo(acc, row)`.
fn accumulate_combo(acc: &mut Value, row: &Value) {
    acc["gp"] = json!(i_of(acc, "gp") + i_of(row, "gp"));
    acc["toi"] = json!(f_of(acc, "toi") + f_of(row, "toi"));
    for key in ["cf", "ca", "ff", "fa", "sf", "sa", "gf", "ga"] {
        acc[key] = json!(i_of(acc, key) + i_of(row, key));
    }
    acc["xgf"] = json!(f_of(acc, "xgf") + f_of(row, "xgf"));
    acc["xga"] = json!(f_of(acc, "xga") + f_of(row, "xga"));
}

/// `_finalize_line_tool_combo(team, players, acc)`.
fn finalize_combo(team: &str, players: &[String], acc: &Value) -> Value {
    let cf = i_of(acc, "cf");
    let ca = i_of(acc, "ca");
    let ff = i_of(acc, "ff");
    let fa = i_of(acc, "fa");
    let sf = i_of(acc, "sf");
    let sa = i_of(acc, "sa");
    let gf = i_of(acc, "gf");
    let ga = i_of(acc, "ga");
    let xgf_v = r2(f_of(acc, "xgf"));
    let xga_v = r2(f_of(acc, "xga"));
    let sh_pct = r1(100.0 * gf as f64 / (sf.max(1)) as f64);
    let sv_pct = r1(100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64));
    json!({
        "players": players,
        "team": team,
        "gp": i_of(acc, "gp"),
        "toi": r1(f_of(acc, "toi")),
        "cf": cf, "ca": ca,
        "cfPct": r1(100.0 * cf as f64 / (cf + ca).max(1) as f64),
        "ff": ff, "fa": fa,
        "ffPct": r1(100.0 * ff as f64 / (ff + fa).max(1) as f64),
        "sf": sf, "sa": sa,
        "sfPct": r1(100.0 * sf as f64 / (sf + sa).max(1) as f64),
        "gf": gf, "ga": ga,
        "gfPct": r1(100.0 * gf as f64 / (gf + ga).max(1) as f64),
        "xgf": xgf_v, "xga": xga_v,
        "xgfPct": r1(100.0 * xgf_v / (xgf_v + xga_v).max(0.001)),
        "shPct": sh_pct,
        "svPct": sv_pct,
        "pdo": r1(sh_pct + sv_pct),
    })
}

/// `_compute_line_tool_team_combos(team, season, ss, strength, xg_col, line_type, pid_info)`.
#[allow(clippy::too_many_arguments)]
async fn compute_line_tool_team_combos(
    caches: &Caches,
    sb: &SbClient,
    team: &str,
    season: i64,
    ss: &str,
    strength: &str,
    xg_col: &str,
    line_type: &str,
    pid_info: &HashMap<String, Value>,
) -> Vec<Value> {
    let (target_positions, combo_size): (HashSet<&str>, usize) = if line_type == "def" {
        (["D"].iter().copied().collect(), 2)
    } else {
        (["C", "L", "R"].iter().copied().collect(), 3)
    };

    let mut t_rows = get_lt_shifts(caches, sb, team, season).await;
    if t_rows.is_empty() {
        return Vec::new();
    }
    t_rows = filter_shifts_season_state(t_rows, ss);
    t_rows = apply_lt_strength_filter(t_rows, strength);

    // group by (team, sorted line pids)
    let mut combo_groups: HashMap<(String, Vec<String>), (i64, HashSet<i64>, HashSet<(i64, i64)>)> =
        HashMap::new();
    for s in &t_rows {
        let pids_str = s_of(s, "player_id");
        let pids_on_ice: Vec<&str> = pids_str.split_whitespace().collect();
        let mut line_pids: Vec<String> = pids_on_ice
            .iter()
            .filter(|pid| {
                pid_info
                    .get(**pid)
                    .and_then(|info| info.get("position"))
                    .and_then(|p| p.as_str())
                    .map(|p| target_positions.contains(p))
                    .unwrap_or(false)
            })
            .map(|p| p.to_string())
            .collect();
        if line_pids.len() != combo_size {
            continue;
        }
        line_pids.sort();
        let key = (team.to_string(), line_pids);
        let gid = i_of(s, "game_id");
        let si = i_of(s, "shift_index");
        let dur = i_of(s, "duration").max(0);
        let grp = combo_groups.entry(key).or_insert((0, HashSet::new(), HashSet::new()));
        grp.0 += dur;
        grp.1.insert(gid);
        grp.2.insert((gid, si));
    }

    if combo_groups.is_empty() {
        return Vec::new();
    }

    let mut combo_onice: HashMap<(String, Vec<String>), OnIce> = HashMap::new();
    for key in combo_groups.keys() {
        combo_onice.insert(key.clone(), OnIce::default());
    }

    let mut all_game_ids: HashSet<i64> = HashSet::new();
    for (_, (_, game_ids, _)) in &combo_groups {
        all_game_ids.extend(game_ids.iter().copied());
    }
    let mut all_game_ids: Vec<i64> = all_game_ids.into_iter().collect();
    all_game_ids.sort_unstable();
    let all_pbp = get_lt_pbp(caches, sb, season, &all_game_ids, xg_col, "").await;

    let mut sk_to_combos: HashMap<(i64, i64), Vec<(String, Vec<String>)>> = HashMap::new();
    for (key, (_, _, shift_keys)) in &combo_groups {
        for sk in shift_keys {
            sk_to_combos.entry(*sk).or_default().push(key.clone());
        }
    }

    for e in &all_pbp {
        if i_of(e, "period") == 5 {
            continue;
        }
        let Some(si) = safe_int(e.get("shift_index")) else { continue };
        let gid = i_of(e, "game_id");
        let ckeys = sk_to_combos.get(&(gid, si));
        let Some(ckeys) = ckeys else { continue };
        if !ss.is_empty() && ss != "all" && s_of(e, "season_state").to_lowercase() != ss {
            continue;
        }

        let et = s_of(e, "event_team").to_uppercase();
        let opp = s_of(e, "opponent").to_uppercase();
        let is_shot = i_of(e, "shot") == 1;
        let is_goal = i_of(e, "goal") == 1;
        let is_fenwick = i_of(e, "fenwick") == 1;
        let xg_val = f_of(e, xg_col);

        for ckey in ckeys {
            let cteam = &ckey.0;
            let oi = combo_onice.get_mut(ckey).unwrap();
            if &et == cteam {
                oi.cf += 1;
                if is_fenwick {
                    oi.ff += 1;
                }
                if is_shot {
                    oi.sf += 1;
                }
                if is_goal {
                    oi.gf += 1;
                }
                oi.xgf += xg_val;
            } else if &opp == cteam {
                oi.ca += 1;
                if is_fenwick {
                    oi.fa += 1;
                }
                if is_shot {
                    oi.sa += 1;
                }
                if is_goal {
                    oi.ga += 1;
                }
                oi.xga += xg_val;
            }
        }
    }

    let mut combos: Vec<Value> = Vec::new();
    for (key, (duration, game_ids, _)) in &combo_groups {
        let toi_min = *duration as f64 / 60.0;
        if toi_min < 0.1 {
            continue;
        }
        let pids = &key.1;
        let oi = &combo_onice[key];
        let cf = oi.cf;
        let ca = oi.ca;
        let ff = oi.ff;
        let fa = oi.fa;
        let sf = oi.sf;
        let sa = oi.sa;
        let gf = oi.gf;
        let ga = oi.ga;
        let xgf_v = r2(oi.xgf);
        let xga_v = r2(oi.xga);
        combos.push(json!({
            "players": pids,
            "team": team,
            "gp": game_ids.len(),
            "toi": r1(toi_min),
            "cf": cf, "ca": ca,
            "cfPct": r1(100.0 * cf as f64 / (cf + ca).max(1) as f64),
            "ff": ff, "fa": fa,
            "ffPct": r1(100.0 * ff as f64 / (ff + fa).max(1) as f64),
            "sf": sf, "sa": sa,
            "sfPct": r1(100.0 * sf as f64 / (sf + sa).max(1) as f64),
            "gf": gf, "ga": ga,
            "gfPct": r1(100.0 * gf as f64 / (gf + ga).max(1) as f64),
            "xgf": xgf_v, "xga": xga_v,
            "xgfPct": r1(100.0 * xgf_v / (xgf_v + xga_v).max(0.001)),
            "shPct": r1(100.0 * gf as f64 / (sf.max(1)) as f64),
            "svPct": r1(100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64)),
            "pdo": r1(100.0 * gf as f64 / (sf.max(1)) as f64 + 100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64)),
        }));
    }
    combos
}

#[derive(Default, Clone)]
struct OnIce {
    cf: i64,
    ca: i64,
    ff: i64,
    fa: i64,
    sf: i64,
    sa: i64,
    gf: i64,
    ga: i64,
    xgf: f64,
    xga: f64,
}

/// `_empty_line_tool_response()`.
fn empty_line_tool_response() -> Value {
    json!({
        "gp": 0, "toi": 0,
        "cf": 0, "ca": 0, "cfPct": 0,
        "ff": 0, "fa": 0, "ffPct": 0,
        "sf": 0, "sa": 0, "sfPct": 0,
        "gf": 0, "ga": 0, "gfPct": 0,
        "xgf": 0, "xga": 0, "xgfPct": 0,
        "shPct": 0, "svPct": 0, "pdo": 0,
        "ozZones": {}, "dzZones": {},
        "ozDetail": {}, "dzDetail": {},
    })
}

fn xg_col_for(xg_model: &str) -> &'static str {
    match xg_model {
        "xG_S" => "xg_s",
        "xG_F2" => "xg_f2",
        _ => "xg_f",
    }
}

/// `GET /api/line-tool/players` — port of `api_line_tool_players`.
async fn api_line_tool_players(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    if team.is_empty() || season.is_empty() {
        return json_no_store(json!({"players": []}));
    }
    let season_ids = season_ids_from_str(&season);
    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(json!({"players": []}));
    };

    // Step 1: player IDs who played for this team via game_data.
    let mut team_pids: HashSet<i64> = HashSet::new();
    for season_id in &season_ids {
        let rows = sb
            .read(
                "game_data",
                "player_id",
                Some(&filters(&[
                    ("season", &format!("eq.{season_id}")),
                    ("team", &format!("eq.{team}")),
                ])),
                None,
                None,
                0,
            )
            .await
            .unwrap_or_default();
        for r in &rows {
            if let Some(pid) = safe_int(r.get("player_id")) {
                team_pids.insert(pid);
            }
        }
    }
    if team_pids.is_empty() {
        return json_no_store(json!({"players": []}));
    }

    // Step 2: targeted players lookup.
    let pid_list: Vec<i64> = {
        let mut v: Vec<i64> = team_pids.iter().copied().collect();
        v.sort_unstable();
        v
    };
    let mut pid_info = players_data::load_player_info_targeted(sb, &pid_list).await;

    // Fallback: season-based lookup.
    if pid_info.is_empty() {
        pid_info = players_data::load_player_info_for_seasons(&state.caches, Some(sb), &season_ids).await;
    }

    let mut players_map: Vec<Value> = Vec::new();
    for pid in &team_pids {
        let info = pid_info.get(&pid.to_string()).cloned().unwrap_or_else(|| json!({}));
        players_map.push(json!({
            "id": pid,
            "name": str_value(info.get("name")),
            "position": str_value(info.get("position")),
        }));
    }

    let pos_rank = |p: &Value| -> i64 {
        match str_value(p.get("position")).as_str() {
            "C" | "L" | "R" => 0,
            "D" => 1,
            _ => 2,
        }
    };
    players_map.sort_by(|a, b| {
        pos_rank(a)
            .cmp(&pos_rank(b))
            .then_with(|| str_value(a.get("name")).cmp(&str_value(b.get("name"))))
    });
    json_no_store(json!({"players": players_map}))
}

/// `GET /api/line-tool/data` — port of `api_line_tool_data`.
async fn api_line_tool_data(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    let players_raw = q(&params, "players", "");
    if team.is_empty() || season.is_empty() {
        return json_no_store(json!({"error": "team and season required"}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST);
    }
    let season_ids = season_ids_from_str(&season);

    let player_ids: Result<Vec<String>, ()> = players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .map(|x| x.trim().parse::<i64>().map(|v| v.to_string()).map_err(|_| ()))
        .collect();
    let player_ids = match player_ids {
        Ok(ids) if ids.len() <= 5 => ids,
        _ => return json_no_store(json!({"error": "invalid player IDs"}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST),
    };

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();

    let vs_team_raw = q(&params, "vs_team", "").to_uppercase();
    let vs_team: Option<String> = if vs_team_raw.is_empty() { None } else { Some(vs_team_raw) };
    let vs_players_raw = q(&params, "vs_players", "");
    let mut vs_player_ids: Vec<String> = vs_players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| x.trim().parse::<i64>().ok().map(|v| v.to_string()))
        .collect();
    vs_player_ids.truncate(3);

    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(empty_line_tool_response());
    };

    let lt_data_cache_key = json!({
        "data": "",
        "team": team,
        "seasons": season_ids,
        "players": player_ids,
        "ss": ss,
        "strength": strength,
        "xg_model": xg_model,
        "vs_team": vs_team.clone().unwrap_or_default(),
        "vs_players": vs_player_ids,
    })
    .to_string();
    if let Some(v) = state.caches.lt_data.get(&lt_data_cache_key) {
        return json_no_store(v);
    }

    let base_ctx = get_lt_base_context(
        &state.caches,
        sb,
        &team,
        &season_ids,
        &player_ids,
        &ss,
        &strength,
        &xg_col,
    )
    .await;

    let shift_rows = base_ctx.get("shiftRows").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if shift_rows.is_empty() {
        return json_no_store(empty_line_tool_response());
    }
    let mut common_shifts = base_ctx.get("baseShifts").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if common_shifts.is_empty() {
        return json_no_store(empty_line_tool_response());
    }
    let all_pbp = base_ctx.get("allPbp").and_then(|v| v.as_array()).cloned().unwrap_or_default();

    // ── vs filter ──
    if let Some(vs_team) = &vs_team {
        if !vs_player_ids.is_empty() {
            let mut opp_shift_rows = get_lt_shifts_parallel(&state.caches, sb, vs_team, &season_ids).await;
            opp_shift_rows = filter_shifts_season_state(opp_shift_rows, &ss);
            let vs_id_set: HashSet<&str> = vs_player_ids.iter().map(|s| s.as_str()).collect();
            let mut opp_valid_keys: HashSet<(i64, i64)> = HashSet::new();
            for s in &opp_shift_rows {
                let pids_str = s_of(s, "player_id");
                let pids: HashSet<&str> = pids_str.split_whitespace().collect();
                if vs_id_set.iter().all(|p| pids.contains(p)) {
                    opp_valid_keys.insert((i_of(s, "game_id"), i_of(s, "shift_index")));
                }
            }
            common_shifts.retain(|s| opp_valid_keys.contains(&(i_of(s, "game_id"), i_of(s, "shift_index"))));
        } else {
            let opponent_game_ids = lt_game_ids_for_opponent(&team, vs_team, &all_pbp);
            common_shifts.retain(|s| opponent_game_ids.contains(&i_of(s, "game_id")));
        }
        if common_shifts.is_empty() {
            return json_no_store(empty_line_tool_response());
        }
    }

    // ── 4. GP/TOI from shifts ──
    let mut game_ids: HashSet<i64> = HashSet::new();
    let mut shift_keys: HashSet<(i64, i64)> = HashSet::new();
    let mut total_duration: i64 = 0;
    for s in &common_shifts {
        let gid = i_of(s, "game_id");
        let si = i_of(s, "shift_index");
        let dur = i_of(s, "duration").max(0);
        game_ids.insert(gid);
        shift_keys.insert((gid, si));
        total_duration += dur;
    }
    let gp = game_ids.len();
    let toi_min = total_duration as f64 / 60.0;

    // ── 4b. Team-level shift keys + TOI ──
    let mut team_shift_keys: HashSet<(i64, i64)> = HashSet::new();
    let mut team_toi_sec: i64 = 0;
    for s in &shift_rows {
        let gid = i_of(s, "game_id");
        if !game_ids.contains(&gid) {
            continue;
        }
        let si = i_of(s, "shift_index");
        team_shift_keys.insert((gid, si));
        team_toi_sec += i_of(s, "duration").max(0);
    }
    let team_toi_min = team_toi_sec as f64 / 60.0;

    // ── 6. Filter PBP to matching shift indexes ──
    let mut events_for: Vec<Value> = Vec::new();
    let mut events_against: Vec<Value> = Vec::new();
    let mut team_oz_detail: HashMap<String, ZoneDetail> = HashMap::new();
    let mut team_dz_detail: HashMap<String, ZoneDetail> = HashMap::new();
    let mut league_zone_detail: HashMap<String, ZoneDetail> = HashMap::new();

    for e in &all_pbp {
        if i_of(e, "period") == 5 {
            continue;
        }
        let Some(si) = safe_int(e.get("shift_index")) else { continue };
        let gid = i_of(e, "game_id");
        if !ss.is_empty() && ss != "all" && s_of(e, "season_state").to_lowercase() != ss {
            continue;
        }
        let et = s_of(e, "event_team").to_uppercase();
        let opp = s_of(e, "opponent").to_uppercase();
        let key = (gid, si);

        if shift_keys.contains(&key) {
            if et == team {
                events_for.push(e.clone());
            } else if opp == team {
                events_against.push(e.clone());
            }
        }

        if team_shift_keys.contains(&key) {
            let bid = s_of(e, "box_id");
            if bid.starts_with('O') {
                let is_fen = i_of(e, "fenwick") == 1;
                let is_sh = i_of(e, "shot") == 1;
                let is_gl = i_of(e, "goal") == 1;
                let xgv = f_of(e, &xg_col);
                let ld = league_zone_detail.entry(bid.clone()).or_default();
                ld.count += 1;
                if is_fen {
                    ld.fenwick += 1;
                }
                if is_sh {
                    ld.shots += 1;
                }
                if is_gl {
                    ld.goals += 1;
                }
                ld.xg += xgv;
                if et == team {
                    let td = team_oz_detail.entry(bid.clone()).or_default();
                    td.count += 1;
                    if is_fen {
                        td.fenwick += 1;
                    }
                    if is_sh {
                        td.shots += 1;
                    }
                    if is_gl {
                        td.goals += 1;
                    }
                    td.xg += xgv;
                } else if opp == team {
                    let dz_bid = format!("D{}", &bid[1..]);
                    let td = team_dz_detail.entry(dz_bid).or_default();
                    td.count += 1;
                    if is_fen {
                        td.fenwick += 1;
                    }
                    if is_sh {
                        td.shots += 1;
                    }
                    if is_gl {
                        td.goals += 1;
                    }
                    td.xg += xgv;
                }
            }
        }
    }

    // ── 7. KPIs ──
    let kpi_for = sum_kpis(&events_for, &xg_col);
    let kpi_against = sum_kpis(&events_against, &xg_col);

    let cf = kpi_for.corsi;
    let ca = kpi_against.corsi;
    let ff = kpi_for.fenwick;
    let fa = kpi_against.fenwick;
    let sf = kpi_for.shots;
    let sa = kpi_against.shots;
    let gf = kpi_for.goals;
    let ga = kpi_against.goals;
    let xgf = kpi_for.xg;
    let xga = kpi_against.xg;

    let cf_pct = r1(100.0 * cf as f64 / (cf + ca).max(1) as f64);
    let ff_pct = r1(100.0 * ff as f64 / (ff + fa).max(1) as f64);
    let sf_pct = r1(100.0 * sf as f64 / (sf + sa).max(1) as f64);
    let gf_pct = r1(100.0 * gf as f64 / (gf + ga).max(1) as f64);
    let xgf_pct = r1(100.0 * xgf / (xgf + xga).max(0.001));
    let sh_pct = r1(100.0 * gf as f64 / (sf.max(1)) as f64);
    let sv_pct = r1(100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64));
    let pdo = r1(sh_pct + sv_pct);

    // ── 8. Zone heat map data ──
    let mut oz_counts: HashMap<String, i64> = HashMap::new();
    let mut oz_detail: HashMap<String, ZoneDetail> = HashMap::new();
    for ev in &events_for {
        let bid = s_of(ev, "box_id");
        if !bid.starts_with('O') {
            continue;
        }
        *oz_counts.entry(bid.clone()).or_insert(0) += 1;
        let d = oz_detail.entry(bid.clone()).or_default();
        d.count += 1;
        if i_of(ev, "fenwick") == 1 {
            d.fenwick += 1;
        }
        if i_of(ev, "shot") == 1 {
            d.shots += 1;
        }
        if i_of(ev, "goal") == 1 {
            d.goals += 1;
        }
        d.xg += f_of(ev, &xg_col);
    }

    let mut dz_counts: HashMap<String, i64> = HashMap::new();
    let mut dz_detail: HashMap<String, ZoneDetail> = HashMap::new();
    for ev in &events_against {
        let bid = s_of(ev, "box_id");
        if !bid.starts_with('O') {
            continue;
        }
        let dz_bid = format!("D{}", &bid[1..]);
        *dz_counts.entry(dz_bid.clone()).or_insert(0) += 1;
        let d = dz_detail.entry(dz_bid.clone()).or_default();
        d.count += 1;
        if i_of(ev, "fenwick") == 1 {
            d.fenwick += 1;
        }
        if i_of(ev, "shot") == 1 {
            d.shots += 1;
        }
        if i_of(ev, "goal") == 1 {
            d.goals += 1;
        }
        d.xg += f_of(ev, &xg_col);
    }

    let oz_detail_json = zone_detail_json(&oz_detail);
    let dz_detail_json = zone_detail_json(&dz_detail);
    let team_oz_json = zone_detail_json(&team_oz_detail);
    let team_dz_json = zone_detail_json(&team_dz_detail);
    let league_zone_json = zone_detail_json(&league_zone_detail);

    // ── 9. Goal events with highlight URLs ──
    let mut goal_highlights: Vec<Value> = Vec::new();
    for ev in &events_for {
        if i_of(ev, "goal") == 1 {
            let hl = s_of(ev, "highlight_url");
            goal_highlights.push(json!({
                "highlightUrl": if hl.is_empty() { Value::Null } else { json!(hl) },
                "x": f_of(ev, "x"),
                "y": f_of(ev, "y"),
                "xG": r4(f_of(ev, &xg_col)),
                "boxId": s_of(ev, "box_id"),
                "direction": "for",
            }));
        }
    }
    for ev in &events_against {
        if i_of(ev, "goal") == 1 {
            let hl = s_of(ev, "highlight_url");
            goal_highlights.push(json!({
                "highlightUrl": if hl.is_empty() { Value::Null } else { json!(hl) },
                "x": f_of(ev, "x"),
                "y": f_of(ev, "y"),
                "xG": r4(f_of(ev, &xg_col)),
                "boxId": s_of(ev, "box_id"),
                "direction": "against",
            }));
        }
    }

    let result = json!({
        "gp": gp,
        "toi": r1(toi_min),
        "cf": cf, "ca": ca, "cfPct": cf_pct,
        "ff": ff, "fa": fa, "ffPct": ff_pct,
        "sf": sf, "sa": sa, "sfPct": sf_pct,
        "gf": gf, "ga": ga, "gfPct": gf_pct,
        "xgf": xgf, "xga": xga, "xgfPct": xgf_pct,
        "shPct": sh_pct, "svPct": sv_pct, "pdo": pdo,
        "ozZones": oz_counts,
        "dzZones": dz_counts,
        "ozDetail": oz_detail_json,
        "dzDetail": dz_detail_json,
        "teamToi": r1(team_toi_min),
        "teamOzDetail": team_oz_json,
        "teamDzDetail": team_dz_json,
        "leagueToi": r1(team_toi_min * 2.0),
        "leagueZoneDetail": league_zone_json,
        "goalHighlights": goal_highlights,
    });
    state.caches.lt_data.insert(lt_data_cache_key, result.clone());
    json_no_store(result)
}

#[derive(Default, Clone)]
struct ZoneDetail {
    count: i64,
    fenwick: i64,
    shots: i64,
    goals: i64,
    xg: f64,
}

fn zone_detail_json(detail: &HashMap<String, ZoneDetail>) -> Value {
    let mut out = serde_json::Map::new();
    for (k, d) in detail {
        out.insert(
            k.clone(),
            json!({
                "count": d.count,
                "fenwick": d.fenwick,
                "shots": d.shots,
                "goals": d.goals,
                "xg": r2(d.xg),
            }),
        );
    }
    Value::Object(out)
}

struct Kpis {
    corsi: i64,
    fenwick: i64,
    shots: i64,
    xg: f64,
    goals: i64,
}

fn sum_kpis(evts: &[Value], xg_col: &str) -> Kpis {
    let mut out = Kpis {
        corsi: 0,
        fenwick: 0,
        shots: 0,
        xg: 0.0,
        goals: 0,
    };
    for ev in evts {
        out.corsi += 1;
        if i_of(ev, "fenwick") == 1 {
            out.fenwick += 1;
        }
        if i_of(ev, "shot") == 1 {
            out.shots += 1;
        }
        if i_of(ev, "goal") == 1 {
            out.goals += 1;
        }
        out.xg += f_of(ev, xg_col);
    }
    out.xg = r2(out.xg);
    out
}

trait WithStatus {
    fn with_status(self, status: StatusCode) -> Response;
}

impl WithStatus for Response {
    fn with_status(mut self, status: StatusCode) -> Response {
        *self.status_mut() = status;
        self
    }
}

/// `GET /api/line-tool/wowy` — port of `api_line_tool_wowy`.
async fn api_line_tool_wowy(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    let players_raw = q(&params, "players", "");
    if team.is_empty() || season.is_empty() || players_raw.is_empty() {
        return json_no_store(json!({"combos": [], "players": []}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST);
    }
    let season_ids = season_ids_from_str(&season);

    let player_ids: Vec<String> = players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| x.trim().parse::<i64>().ok().map(|v| v.to_string()))
        .collect();
    if player_ids.is_empty() || player_ids.len() > 5 {
        return json_no_store(json!({"combos": [], "players": []}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST);
    }

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();

    let vs_team_raw = q(&params, "vs_team", "").to_uppercase();
    let vs_team: Option<String> = if vs_team_raw.is_empty() { None } else { Some(vs_team_raw) };
    let vs_players_raw = q(&params, "vs_players", "");
    let mut vs_player_ids: Vec<String> = vs_players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| x.trim().parse::<i64>().ok().map(|v| v.to_string()))
        .collect();
    vs_player_ids.truncate(3);

    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(json!({"combos": [], "players": []}));
    };

    // ── 1. shifts ──
    let mut shift_rows = get_lt_shifts_parallel(&state.caches, sb, &team, &season_ids).await;
    if shift_rows.is_empty() {
        return json_no_store(json!({"combos": [], "players": []}));
    }
    shift_rows = filter_shifts_season_state(shift_rows, &ss);
    shift_rows = apply_lt_strength_filter(shift_rows, &strength);

    // ── 2b. vs filter ──
    if let Some(vs_team) = &vs_team {
        let mut opp_shift_rows = get_lt_shifts_parallel(&state.caches, sb, vs_team, &season_ids).await;
        opp_shift_rows = filter_shifts_season_state(opp_shift_rows, &ss);
        let opp_valid_keys: HashSet<(i64, i64)> = if !vs_player_ids.is_empty() {
            let vs_id_set: HashSet<&str> = vs_player_ids.iter().map(|s| s.as_str()).collect();
            let mut keys = HashSet::new();
            for s in &opp_shift_rows {
                let pids_str = s_of(s, "player_id");
                let pids: HashSet<&str> = pids_str.split_whitespace().collect();
                if vs_id_set.iter().all(|p| pids.contains(p)) {
                    keys.insert((i_of(s, "game_id"), i_of(s, "shift_index")));
                }
            }
            keys
        } else {
            opp_shift_rows
                .iter()
                .map(|s| (i_of(s, "game_id"), i_of(s, "shift_index")))
                .collect()
        };
        shift_rows.retain(|s| opp_valid_keys.contains(&(i_of(s, "game_id"), i_of(s, "shift_index"))));
        if shift_rows.is_empty() {
            return json_no_store(json!({"combos": [], "players": []}));
        }
    }

    // ── 3. group shifts by player mask ──
    // mask = Vec<bool>, one per player_id in order.
    let mut mask_groups: HashMap<Vec<bool>, (i64, HashSet<i64>, HashSet<(i64, i64)>)> = HashMap::new();
    for s in &shift_rows {
        let pids_str = s_of(s, "player_id");
        let pids_on_ice: HashSet<&str> = pids_str.split_whitespace().collect();
        let mask: Vec<bool> = player_ids.iter().map(|pid| pids_on_ice.contains(pid.as_str())).collect();
        let gid = i_of(s, "game_id");
        let si = i_of(s, "shift_index");
        let dur = i_of(s, "duration").max(0);
        let grp = mask_groups.entry(mask).or_insert((0, HashSet::new(), HashSet::new()));
        grp.0 += dur;
        grp.1.insert(gid);
        grp.2.insert((gid, si));
    }

    // ── 4. PBP ──
    let mut all_game_ids: HashSet<i64> = HashSet::new();
    for (_, (_, game_ids, _)) in &mask_groups {
        all_game_ids.extend(game_ids.iter().copied());
    }
    let mut game_list: Vec<i64> = all_game_ids.into_iter().collect();
    game_list.sort_unstable();
    let all_pbp = get_lt_pbp_parallel(
        &state.caches,
        sb,
        &season_ids,
        &game_list,
        &xg_col,
        "player1_id,player2_id,player3_id,goalie_id",
    )
    .await;

    let mut sk_to_mask: HashMap<(i64, i64), Vec<bool>> = HashMap::new();
    for (mask, (_, _, shift_keys)) in &mask_groups {
        for sk in shift_keys {
            sk_to_mask.insert(*sk, mask.clone());
        }
    }

    let mut mask_onice: HashMap<Vec<bool>, OnIce> = HashMap::new();
    let mut mask_indiv: HashMap<Vec<bool>, HashMap<String, IndivAcc>> = HashMap::new();
    for mask in mask_groups.keys() {
        mask_onice.insert(mask.clone(), OnIce::default());
        mask_indiv.insert(
            mask.clone(),
            player_ids
                .iter()
                .map(|pid| (pid.clone(), IndivAcc::default()))
                .collect(),
        );
    }

    for e in &all_pbp {
        if i_of(e, "period") == 5 {
            continue;
        }
        let Some(si) = safe_int(e.get("shift_index")) else { continue };
        let gid = i_of(e, "game_id");
        let Some(mask) = sk_to_mask.get(&(gid, si)).cloned() else { continue };
        if !ss.is_empty() && ss != "all" && s_of(e, "season_state").to_lowercase() != ss {
            continue;
        }

        let et = s_of(e, "event_team").to_uppercase();
        let opp = s_of(e, "opponent").to_uppercase();
        let is_for = et == team;
        let is_against = opp == team;
        let is_shot = i_of(e, "shot") == 1;
        let is_goal = i_of(e, "goal") == 1;
        let xg_val = f_of(e, &xg_col);
        let is_fenwick = i_of(e, "fenwick") == 1;

        let oi = mask_onice.get_mut(&mask).unwrap();
        if is_for {
            oi.cf += 1;
            if is_fenwick {
                oi.ff += 1;
            }
            if is_shot {
                oi.sf += 1;
            }
            if is_goal {
                oi.gf += 1;
            }
            oi.xgf += xg_val;
        } else if is_against {
            oi.ca += 1;
            if is_fenwick {
                oi.fa += 1;
            }
            if is_shot {
                oi.sa += 1;
            }
            if is_goal {
                oi.ga += 1;
            }
            oi.xga += xg_val;
        }

        let p1 = s_of(e, "player1_id");
        let p2 = s_of(e, "player2_id");
        let p3 = s_of(e, "player3_id");
        let goalie = s_of(e, "goalie_id");

        for pid in &player_ids {
            let iv = mask_indiv.get_mut(&mask).unwrap().get_mut(pid).unwrap();
            if is_for {
                if p1 == *pid {
                    if is_shot {
                        iv.shots += 1;
                    }
                    iv.ixg += xg_val;
                    if is_goal {
                        iv.goals += 1;
                    }
                }
                if is_goal && p2 == *pid {
                    iv.a1 += 1;
                }
                if is_goal && p3 == *pid {
                    iv.a2 += 1;
                }
            } else if is_against {
                if goalie == *pid {
                    iv.fa += 1;
                    if is_shot {
                        iv.sa += 1;
                    }
                    if is_goal {
                        iv.ga += 1;
                    }
                    iv.xga += xg_val;
                }
            }
        }
    }

    // ── 6. build combos ──
    let mut combos: Vec<Value> = Vec::new();
    for (mask, (duration, game_ids, _)) in &mask_groups {
        let toi_min = *duration as f64 / 60.0;
        if toi_min < 0.1 {
            continue;
        }
        let oi = &mask_onice[mask];
        let cf = oi.cf;
        let ca = oi.ca;
        let ff = oi.ff;
        let fa = oi.fa;
        let sf = oi.sf;
        let sa = oi.sa;
        let gf = oi.gf;
        let ga = oi.ga;
        let xgf_v = r2(oi.xgf);
        let xga_v = r2(oi.xga);

        let mut individual = serde_json::Map::new();
        for pid in &player_ids {
            let iv = &mask_indiv[mask][pid];
            let sh_pct = r1(100.0 * iv.goals as f64 / (iv.shots.max(1)) as f64);
            let sv_pct = if iv.sa > 0 { r1(100.0 * (1.0 - iv.ga as f64 / iv.sa as f64)) } else { 0.0 };
            let xsv_pct = if iv.sa > 0 { r1(100.0 * (1.0 - iv.xga / iv.sa as f64)) } else { 0.0 };
            let dsv_pct = if iv.sa > 0 { r1(sv_pct - xsv_pct) } else { 0.0 };
            individual.insert(
                pid.clone(),
                json!({
                    "goals": iv.goals,
                    "a1": iv.a1,
                    "a2": iv.a2,
                    "points": iv.goals + iv.a1 + iv.a2,
                    "shots": iv.shots,
                    "ixg": r2(iv.ixg),
                    "gax": r2(iv.goals as f64 - iv.ixg),
                    "shPct": sh_pct,
                    "fa": iv.fa,
                    "ga": iv.ga,
                    "sa": iv.sa,
                    "xga": r2(iv.xga),
                    "gsax": r2(iv.xga - iv.ga as f64),
                    "svPct": sv_pct,
                    "xsvPct": xsv_pct,
                    "dsvPct": dsv_pct,
                }),
            );
        }

        combos.push(json!({
            "mask": mask,
            "gp": game_ids.len(),
            "toi": r1(toi_min),
            "cf": cf, "ca": ca,
            "cfPct": r1(100.0 * cf as f64 / (cf + ca).max(1) as f64),
            "ff": ff, "fa": fa,
            "ffPct": r1(100.0 * ff as f64 / (ff + fa).max(1) as f64),
            "sf": sf, "sa": sa,
            "sfPct": r1(100.0 * sf as f64 / (sf + sa).max(1) as f64),
            "gf": gf, "ga": ga,
            "gfPct": r1(100.0 * gf as f64 / (gf + ga).max(1) as f64),
            "xgf": xgf_v, "xga": xga_v,
            "xgfPct": r1(100.0 * xgf_v / (xgf_v + xga_v).max(0.001)),
            "shPct": r1(100.0 * gf as f64 / (sf.max(1)) as f64),
            "svPct": r1(100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64)),
            "pdo": r1(100.0 * gf as f64 / (sf.max(1)) as f64 + 100.0 * (1.0 - ga as f64 / (sa.max(1)) as f64)),
            "individual": Value::Object(individual),
        }));
    }

    combos.sort_by(|a, b| f_of(a, "toi").partial_cmp(&f_of(b, "toi")).unwrap_or(std::cmp::Ordering::Equal).reverse());

    let pid_ints: Vec<i64> = player_ids
        .iter()
        .filter_map(|p| p.parse::<i64>().ok())
        .collect();
    let pid_info = players_data::load_player_info_targeted(sb, &pid_ints).await;

    let mut player_info: Vec<Value> = Vec::new();
    for pid in &player_ids {
        let info = pid_info.get(pid).cloned().unwrap_or_else(|| json!({}));
        let name = str_value(info.get("name"));
        player_info.push(json!({
            "id": pid,
            "name": if name.is_empty() { format!("#{pid}") } else { name },
            "position": str_value(info.get("position")),
        }));
    }

    json_no_store(json!({"combos": combos, "players": player_info}))
}

#[derive(Default)]
struct IndivAcc {
    goals: i64,
    a1: i64,
    a2: i64,
    shots: i64,
    ixg: f64,
    fa: i64,
    ga: i64,
    sa: i64,
    xga: f64,
}

/// `GET /api/line-tool/versus` — port of `api_line_tool_versus`.
async fn api_line_tool_versus(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    let vs_team = q(&params, "vs_team", "").to_uppercase();
    let players_raw = q(&params, "players", "");
    let vs_players_raw = q(&params, "vs_players", "");
    if team.is_empty() || season.is_empty() || vs_team.is_empty() {
        return json_no_store(json!({"rows": [], "vsTeam": vs_team, "vsPlayers": []}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST);
    }
    let season_ids = season_ids_from_str(&season);

    let mut player_ids: Vec<String> = players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| x.trim().parse::<i64>().ok().map(|v| v.to_string()))
        .collect();
    player_ids.truncate(5);

    let mut vs_player_ids: Vec<String> = vs_players_raw
        .split(',')
        .filter(|x| !x.trim().is_empty())
        .filter_map(|x| x.trim().parse::<i64>().ok().map(|v| v.to_string()))
        .collect();
    vs_player_ids.truncate(3);

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();

    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(json!({"rows": [], "vsTeam": vs_team, "vsPlayers": []}));
    };

    let versus_cache_key = json!({
        "versus": "",
        "team": team,
        "seasons": season_ids,
        "players": player_ids,
        "vs_team": vs_team,
        "vs_players": vs_player_ids,
        "ss": ss,
        "strength": strength,
        "xg_model": xg_model,
    })
    .to_string();
    if let Some(v) = state.caches.lt_data.get(&versus_cache_key) {
        return json_no_store(v);
    }

    let base_ctx = get_lt_base_context(
        &state.caches,
        sb,
        &team,
        &season_ids,
        &player_ids,
        &ss,
        &strength,
        &xg_col,
    )
    .await;

    let shift_rows = base_ctx.get("shiftRows").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if shift_rows.is_empty() {
        return json_no_store(json!({"rows": [], "vsTeam": vs_team, "vsPlayers": []}));
    }
    let base_shifts = base_ctx.get("baseShifts").and_then(|v| v.as_array()).cloned().unwrap_or_default();
    if base_shifts.is_empty() {
        return json_no_store(json!({"rows": [], "vsTeam": vs_team, "vsPlayers": []}));
    }
    let all_pbp = base_ctx.get("allPbp").and_then(|v| v.as_array()).cloned().unwrap_or_default();

    let mut opp_key_to_pids: HashMap<(i64, i64), HashSet<String>> = HashMap::new();
    let opponent_game_ids = lt_game_ids_for_opponent(&team, &vs_team, &all_pbp);
    if !vs_player_ids.is_empty() {
        let mut opp_shift_rows = get_lt_shifts_parallel(&state.caches, sb, &vs_team, &season_ids).await;
        opp_shift_rows = filter_shifts_season_state(opp_shift_rows, &ss);
        for s in &opp_shift_rows {
            let gid = i_of(s, "game_id");
            let si = i_of(s, "shift_index");
            let pset: HashSet<String> = s_of(s, "player_id").split_whitespace().map(|p| p.to_string()).collect();
            opp_key_to_pids
                .entry((gid, si))
                .or_insert_with(HashSet::new)
                .extend(pset);
        }
    }

    // 3) Partition shifts into groups.
    type GKey = (String, Option<Vec<bool>>);
    let mut group_rows: HashMap<GKey, (i64, HashSet<i64>, HashSet<(i64, i64)>)> = HashMap::new();
    group_rows.insert(("not_vs".to_string(), None), (0, HashSet::new(), HashSet::new()));
    if vs_player_ids.is_empty() {
        group_rows.insert(("vs".to_string(), Some(Vec::new())), (0, HashSet::new(), HashSet::new()));
    } else {
        let combo_count = 1usize << vs_player_ids.len();
        for m in 0..combo_count {
            let mask: Vec<bool> = (0..vs_player_ids.len()).map(|i| ((m >> i) & 1) == 1).collect();
            group_rows.insert(("vs".to_string(), Some(mask)), (0, HashSet::new(), HashSet::new()));
        }
    }

    for s in &base_shifts {
        let gid = i_of(s, "game_id");
        let si = i_of(s, "shift_index");
        let dur = i_of(s, "duration").max(0);
        let key = (gid, si);
        let gkey: GKey = if vs_player_ids.is_empty() {
            if opponent_game_ids.contains(&gid) {
                ("vs".to_string(), Some(Vec::new()))
            } else {
                ("not_vs".to_string(), None)
            }
        } else {
            match opp_key_to_pids.get(&key) {
                None => ("not_vs".to_string(), None),
                Some(opp_pids) => {
                    let mask: Vec<bool> = vs_player_ids.iter().map(|pid| opp_pids.contains(pid)).collect();
                    ("vs".to_string(), Some(mask))
                }
            }
        };
        let grp = group_rows.get_mut(&gkey).unwrap();
        grp.0 += dur;
        grp.1.insert(gid);
        grp.2.insert(key);
    }

    let mut sk_to_group: HashMap<(i64, i64), GKey> = HashMap::new();
    for (gkey, (_, _, shift_keys)) in &group_rows {
        for sk in shift_keys {
            sk_to_group.insert(*sk, gkey.clone());
        }
    }

    let mut group_stats: HashMap<GKey, OnIce> = HashMap::new();
    for gkey in group_rows.keys() {
        group_stats.insert(gkey.clone(), OnIce::default());
    }

    for e in &all_pbp {
        if i_of(e, "period") == 5 {
            continue;
        }
        let Some(si) = safe_int(e.get("shift_index")) else { continue };
        let gid = i_of(e, "game_id");
        let Some(gkey) = sk_to_group.get(&(gid, si)).cloned() else { continue };
        if !ss.is_empty() && ss != "all" && s_of(e, "season_state").to_lowercase() != ss {
            continue;
        }

        let et = s_of(e, "event_team").to_uppercase();
        let opp = s_of(e, "opponent").to_uppercase();
        let is_for = et == team;
        let is_against = opp == team;
        if !is_for && !is_against {
            continue;
        }
        let is_fenwick = i_of(e, "fenwick") == 1;
        let is_shot = i_of(e, "shot") == 1;
        let is_goal = i_of(e, "goal") == 1;
        let xg_val = f_of(e, &xg_col);

        let st = group_stats.get_mut(&gkey).unwrap();
        if is_for {
            st.cf += 1;
            if is_fenwick {
                st.ff += 1;
            }
            if is_shot {
                st.sf += 1;
            }
            if is_goal {
                st.gf += 1;
            }
            st.xgf += xg_val;
        } else if is_against {
            st.ca += 1;
            if is_fenwick {
                st.fa += 1;
            }
            if is_shot {
                st.sa += 1;
            }
            if is_goal {
                st.ga += 1;
            }
            st.xga += xg_val;
        }
    }

    // Targeted lookup for opponent player labels.
    let vs_pid_ints: Vec<i64> = vs_player_ids
        .iter()
        .filter_map(|p| p.parse::<i64>().ok())
        .collect();
    let pid_info = players_data::load_player_info_targeted(sb, &vs_pid_ints).await;
    let mut vs_player_info: Vec<Value> = Vec::new();
    for pid in &vs_player_ids {
        let info = pid_info.get(pid).cloned().unwrap_or_else(|| json!({}));
        let name = str_value(info.get("name"));
        vs_player_info.push(json!({
            "id": pid,
            "name": if name.is_empty() { format!("#{pid}") } else { name },
            "position": str_value(info.get("position")),
        }));
    }

    fn label_for_row(vs_team: &str, vs_player_info: &[Value], context: &str, mask: &[bool]) -> String {
        if context == "not_vs" {
            return format!("w/o {vs_team}");
        }
        if vs_player_info.is_empty() {
            return format!("vs {vs_team}");
        }
        let mut parts = vec![format!("vs {vs_team}")];
        for (i, p) in vs_player_info.iter().enumerate() {
            let name = str_value(p.get("name"));
            let ln = name.split(' ').last().filter(|s| !s.is_empty()).map(|s| s.to_string()).unwrap_or_else(|| {
                format!("#{}", str_value(p.get("id")))
            });
            let on = i < mask.len() && mask[i];
            parts.push(if on { format!("vs {ln}") } else { format!("w/o {ln}") });
        }
        parts.join(" · ")
    }

    let mut rows: Vec<Value> = Vec::new();
    for (gkey, (duration, game_ids, _)) in &group_rows {
        let context = &gkey.0;
        let mask = gkey.1.clone().unwrap_or_default();
        let st = &group_stats[gkey];
        let toi_min = *duration as f64 / 60.0;
        let cf = st.cf;
        let ca = st.ca;
        let gf = st.gf;
        let ga = st.ga;
        let xgf_v = r2(st.xgf);
        let xga_v = r2(st.xga);
        let sh_pct = r1(100.0 * gf as f64 / (st.sf.max(1)) as f64);
        let sv_pct = r1(100.0 * (1.0 - ga as f64 / (st.sa.max(1)) as f64));
        rows.push(json!({
            "context": if context == "vs" { "vs" } else { "not_vs" },
            "label": label_for_row(&vs_team, &vs_player_info, context, &mask),
            "mask": mask,
            "gp": game_ids.len(),
            "toi": r1(toi_min),
            "cf": cf, "ca": ca,
            "cfPct": r1(100.0 * cf as f64 / (cf + ca).max(1) as f64),
            "gf": gf, "ga": ga,
            "gfPct": r1(100.0 * gf as f64 / (gf + ga).max(1) as f64),
            "xgf": xgf_v, "xga": xga_v,
            "xgfPct": r1(100.0 * xgf_v / (xgf_v + xga_v).max(0.001)),
            "pdo": r1(sh_pct + sv_pct),
        }));
    }

    // Deterministic sort: vs rows first (most specific mask first), then not_vs.
    rows.sort_by(|a, b| {
        let a_ctx = if str_value(a.get("context")) == "vs" { 0 } else { 1 };
        let b_ctx = if str_value(b.get("context")) == "vs" { 0 } else { 1 };
        let a_on = a.get("mask").and_then(|m| m.as_array()).map(|m| m.iter().filter(|v| v.as_bool().unwrap_or(false)).count()).unwrap_or(0);
        let b_on = b.get("mask").and_then(|m| m.as_array()).map(|m| m.iter().filter(|v| v.as_bool().unwrap_or(false)).count()).unwrap_or(0);
        a_ctx.cmp(&b_ctx)
            .then_with(|| b_on.cmp(&a_on))
            .then_with(|| {
                f_of(b, "toi")
                    .partial_cmp(&f_of(a, "toi"))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let result = json!({"rows": rows, "vsTeam": vs_team, "vsPlayers": vs_player_info});
    state.caches.lt_data.insert(versus_cache_key, result.clone());
    json_no_store(result)
}

/// `GET /api/line-tool/lines` — port of `api_line_tool_lines`.
async fn api_line_tool_lines(State(state): State<AppState>, Query(params): Query<HashMap<String, String>>) -> Response {
    let team = q(&params, "team", "").to_uppercase();
    let season = q(&params, "season", "");
    if team.is_empty() || season.is_empty() {
        return json_no_store(json!({"combos": [], "players": {}}))
            .into_response()
            .with_status(StatusCode::BAD_REQUEST);
    }
    let season_ids = season_ids_from_str(&season);

    let ss = q(&params, "seasonState", "regular").to_lowercase();
    let strength = q(&params, "strengthState", "5v5");
    let xg_model = q(&params, "xgModel", "xG_F");
    let xg_col = xg_col_for(&xg_model).to_string();
    let line_type = q(&params, "type", "fwd").to_lowercase();
    let scope = q(&params, "scope", "team").to_lowercase();

    let lt_lines_cache_key = json!({
        "lines": "",
        "team": team,
        "seasons": season_ids,
        "ss": ss,
        "strength": strength,
        "xg_col": xg_col,
        "line_type": line_type,
        "scope": scope,
    })
    .to_string();
    if let Some(v) = state.caches.lt_data.get(&lt_lines_cache_key) {
        return json_no_store(v);
    }

    let Some(sb) = state.sb.as_ref() else {
        return json_no_store(json!({"combos": [], "players": {}}));
    };

    let result = compute_api_line_tool_lines(&state.caches, sb, &team, &season_ids, &ss, &strength, &xg_col, &line_type, &scope).await;
    state.caches.lt_data.insert(lt_lines_cache_key, result.clone());
    json_no_store(result)
}

/// `_compute_api_line_tool_lines`.
#[allow(clippy::too_many_arguments)]
async fn compute_api_line_tool_lines(
    caches: &Caches,
    sb: &SbClient,
    team: &str,
    season_ids: &[i64],
    ss: &str,
    strength: &str,
    xg_col: &str,
    line_type: &str,
    scope: &str,
) -> Value {
    let team_pids = players_data::get_team_pids_for_seasons(sb, team, season_ids).await;
    let team_pids_vec: Vec<i64> = team_pids.iter().copied().collect();
    let mut pid_info = if team_pids_vec.is_empty() {
        HashMap::new()
    } else {
        players_data::load_player_info_targeted(sb, &team_pids_vec).await
    };
    if pid_info.is_empty() {
        pid_info = players_data::load_player_info_for_seasons(caches, Some(sb), season_ids).await;
    }

    if scope == "league" {
        let tbl = if line_type == "fwd" { "forward_lines" } else { "defense_pairings" };
        let xg_f_col = match xg_col {
            "xg_s" => "xgf_s",
            "xg_f2" => "xgf_f2",
            _ => "xgf",
        };
        let xg_a_col = match xg_col {
            "xg_s" => "xga_s",
            "xg_f2" => "xga_f2",
            _ => "xga",
        };
        let stage_map: HashMap<&str, &str> = HashMap::from([("regular", "2"), ("playoffs", "3")]);

        let mut combo_acc: HashMap<(String, Vec<String>), Value> = HashMap::new();
        let mut live_keys: HashSet<(String, Vec<String>)> = HashSet::new();

        for season_id in season_ids {
            let mut db_filters: Vec<(&str, String)> =
                vec![("season", format!("eq.{season_id}"))];
            if let Some(stage) = stage_map.get(ss) {
                db_filters.push(("season_stage", format!("eq.{stage}")));
            }
            let fmap: std::collections::BTreeMap<String, String> = db_filters
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            let rows = sb.read(tbl, "*", Some(&fmap), None, None, 0).await.unwrap_or_default();

            let team_live_combos = compute_line_tool_team_combos(
                caches, sb, team, *season_id, ss, strength, xg_col, line_type, &pid_info,
            )
            .await;
            for combo in &team_live_combos {
                let pids: Vec<String> = combo
                    .get("players")
                    .and_then(|p| p.as_array())
                    .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                    .unwrap_or_default();
                let combo_key = (str_value(combo.get("team")), pids);
                live_keys.insert(combo_key.clone());
                let acc = combo_acc
                    .entry(combo_key)
                    .or_insert_with(|| json!({"gp": 0, "toi": 0.0, "cf": 0, "ca": 0, "ff": 0, "fa": 0, "sf": 0, "sa": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0}));
                accumulate_combo(acc, combo);
            }

            for r in &rows {
                let toi = f_of(r, "toi");
                if toi < 0.1 {
                    continue;
                }
                let pids: Vec<String> = s_of(r, "player_ids").split_whitespace().map(|s| s.to_string()).collect();
                let combo_key = (str_value(r.get("team")), pids);
                if live_keys.contains(&combo_key) {
                    continue;
                }
                let acc = combo_acc
                    .entry(combo_key)
                    .or_insert_with(|| json!({"gp": 0, "toi": 0.0, "cf": 0, "ca": 0, "ff": 0, "fa": 0, "sf": 0, "sa": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0}));
                accumulate_combo(
                    acc,
                    &json!({
                        "gp": i_of(r, "gp"),
                        "toi": toi,
                        "cf": i_of(r, "cf"),
                        "ca": i_of(r, "ca"),
                        "ff": i_of(r, "ff"),
                        "fa": i_of(r, "fa"),
                        "sf": i_of(r, "sf"),
                        "sa": i_of(r, "sa"),
                        "gf": i_of(r, "gf"),
                        "ga": i_of(r, "ga"),
                        "xgf": f_of(r, xg_f_col),
                        "xga": f_of(r, xg_a_col),
                    }),
                );
            }
        }

        let mut combos: Vec<Value> = Vec::new();
        for ((team_abbr, pids), acc) in &combo_acc {
            if f_of(acc, "toi") >= 0.1 {
                combos.push(finalize_combo(team_abbr, pids, acc));
            }
        }

        // Resolve any missing player names.
        let mut missing_pids: Vec<i64> = Vec::new();
        for combo in &combos {
            if let Some(players) = combo.get("players").and_then(|p| p.as_array()) {
                for p in players {
                    if let Some(pid) = p.as_str() {
                        if !pid_info.contains_key(pid) {
                            if let Ok(pid_i) = pid.parse::<i64>() {
                                missing_pids.push(pid_i);
                            }
                        }
                    }
                }
            }
        }
        if !missing_pids.is_empty() {
            let extra = players_data::load_player_info_targeted(sb, &missing_pids).await;
            pid_info.extend(extra);
        }

        let mut players_out = serde_json::Map::new();
        for combo in &combos {
            if let Some(players) = combo.get("players").and_then(|p| p.as_array()) {
                for p in players {
                    let Some(pid) = p.as_str() else { continue };
                    if !players_out.contains_key(pid) {
                        let info = pid_info.get(pid).cloned().unwrap_or_else(|| json!({}));
                        let name = str_value(info.get("name"));
                        let name = if name.is_empty() { format!("#{pid}") } else { name };
                        players_out.insert(
                            pid.to_string(),
                            json!({
                                "name": name,
                                "position": str_value(info.get("position")),
                            }),
                        );
                    }
                }
            }
        }
        combos.sort_by(|a, b| {
            f_of(b, "toi")
                .partial_cmp(&f_of(a, "toi"))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return json!({"combos": combos, "players": players_out});
    }

    // Team scope: live combos only.
    let mut combo_acc2: HashMap<(String, Vec<String>), Value> = HashMap::new();
    for season_id in season_ids {
        let combos = compute_line_tool_team_combos(
            caches, sb, team, *season_id, ss, strength, xg_col, line_type, &pid_info,
        )
        .await;
        for combo in &combos {
            let pids: Vec<String> = combo
                .get("players")
                .and_then(|p| p.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str().map(|s| s.to_string())).collect())
                .unwrap_or_default();
            let combo_key = (str_value(combo.get("team")), pids);
            let acc = combo_acc2
                .entry(combo_key)
                .or_insert_with(|| json!({"gp": 0, "toi": 0.0, "cf": 0, "ca": 0, "ff": 0, "fa": 0, "sf": 0, "sa": 0, "gf": 0, "ga": 0, "xgf": 0.0, "xga": 0.0}));
            accumulate_combo(acc, combo);
        }
    }
    let mut combos: Vec<Value> = Vec::new();
    for ((team_abbr, pids), acc) in &combo_acc2 {
        if f_of(acc, "toi") >= 0.1 {
            combos.push(finalize_combo(team_abbr, pids, acc));
        }
    }
    json!({"combos": combos, "players": pid_info})
}
