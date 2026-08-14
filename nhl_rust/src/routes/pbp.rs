//! M3 game core routes — ports of `api_game_pbp` and `api_game_shifts` from
//! `app/routes.py` (PBP pipeline §6.1 + shifts HTML §6.2 + xG §6.3).

use std::collections::{BTreeMap, HashMap};

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Map, Value};

use crate::data::boxids;
use crate::data::rosters;
use crate::disk_cache;
use crate::models::xg;
use crate::nhl::client::{get_json, get_json_with_ua, API_WEB};
use crate::nhl::shifts_html;
use crate::state::AppState;
use crate::util::parse::{parse_locale_float, safe_int, str_value};

pub fn router() -> Router<AppState> {
    Router::new()
        .route("/api/game/{game_id}/play-by-play", get(api_game_pbp))
        .route("/api/game/{game_id}/shifts", get(api_game_shifts))
}

fn json_no_store(v: Value) -> Response {
    (StatusCode::OK, [("Cache-Control", "no-store")], Json(v)).into_response()
}

fn json_err(status: StatusCode, v: Value) -> Response {
    (status, Json(v)).into_response()
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

fn is_truthy(v: &str) -> bool {
    matches!(v.trim().to_lowercase().as_str(), "1" | "true" | "yes" | "y" | "force" | "lite")
}

fn normalize_xg_model_name(xg_model: Option<&str>) -> Option<String> {
    let raw = xg_model.unwrap_or("").trim().to_uppercase().replace('-', "_");
    match raw.as_str() {
        "XG_F" => Some("xG_F".to_string()),
        "XG_S" => Some("xG_S".to_string()),
        "XG_F2" => Some("xG_F2".to_string()),
        _ => None,
    }
}

fn live_ttl(gstate: &str) -> bool {
    matches!(
        gstate,
        "LIVE" | "SCHEDULED" | "PREVIEW" | "INPROGRESS"
    )
}

// ---------------------------------------------------------------------------
// Shifts route
// ---------------------------------------------------------------------------

/// Internal shifts loader shared by the route and the PBP shift-join.
/// Returns the full `{gameId, seasonDir, suffix, source, shifts, gameState}` object.
async fn load_shifts_for_game(
    state: &AppState,
    game_id: i64,
    force: bool,
    no_cache: bool,
    no_disk: bool,
) -> Result<Value, Response> {
    let gid = game_id.to_string();
    if gid.len() < 10 {
        return Err(json_err(StatusCode::BAD_REQUEST, json!({"error": "Invalid gameId"})));
    }
    let start_year: i64 = gid[..4].parse().unwrap_or(0);
    if start_year == 0 {
        return Err(json_err(StatusCode::BAD_REQUEST, json!({"error": "Invalid gameId"})));
    }
    let season_dir = format!("{}{}", start_year, start_year + 1);
    let suffix = &gid[4..];
    let urls = json!({
        "away": format!("https://www.nhl.com/scores/htmlreports/{season_dir}/TV{suffix}.HTM"),
        "home": format!("https://www.nhl.com/scores/htmlreports/{season_dir}/TH{suffix}.HTM"),
    });

    let std_ttl = state.caches.shifts_ttl_secs();
    let disk_base = disk_cache::disk_cache_base(state.cfg.xg_cache_dir.as_deref());
    let disk_path = disk_cache::shifts_path(&disk_base, game_id);

    if !force && !no_disk {
        if let Some(js) = disk_cache::read_json_if_fresh(&disk_path, 5.0, std_ttl as f64) {
            let payload: Map<String, Value> = js
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter(|(k, _)| !k.starts_with('_'))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default();
            return Ok(Value::Object(payload));
        }
    }
    if !force && !no_cache {
        if let Some(payload) = state.caches.shifts.get(&game_id) {
            return Ok(payload);
        }
    }

    // Fetch HTML reports (browser UA) and boxscore.
    let away_url = urls["away"].as_str().unwrap_or("").to_string();
    let home_url = urls["home"].as_str().unwrap_or("").to_string();
    let away_html = fetch_shift_html(&state.http, &away_url).await;
    let home_html = fetch_shift_html(&state.http, &home_url).await;

    let box_url = format!("{API_WEB}/v1/gamecenter/{game_id}/boxscore");
    let boxscore = match get_json(&state.http, &box_url, 20).await {
        Ok(b) => b,
        Err(_) => return Err(json_err(StatusCode::BAD_GATEWAY, json!({"error": "Failed to fetch boxscore"}))),
    };

    let shifts = shifts_html::compute_shifts_from_html(game_id, &away_html, &home_html, &boxscore);

    let mut out = json!({
        "gameId": game_id,
        "seasonDir": season_dir,
        "suffix": suffix,
        "source": urls,
        "shifts": shifts,
    });
    let game_state = str_value(
        boxscore.get("gameState")
            .or_else(|| boxscore.get("gameStatus")),
    )
    .to_uppercase();
    if !game_state.is_empty() {
        out["gameState"] = json!(game_state);
    }

    if !no_cache {
        state.caches.shifts.insert(game_id, out.clone());
    }
    if !no_disk {
        let mut js = out.clone();
        if let Value::Object(o) = &mut js {
            o.insert("_cachedAt".to_string(), json!(now_epoch()));
        }
        disk_cache::write_json(&disk_path, &js).ok();
    }
    Ok(out)
}

fn now_epoch() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

async fn fetch_shift_html(http: &reqwest::Client, url: &str) -> String {
    let resp = match http
        .get(url)
        .timeout(std::time::Duration::from_secs(25))
        .header("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36")
        .header("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return String::new(),
    };
    if !resp.status().is_success() {
        return String::new();
    }
    let text = match resp.text().await {
        Ok(t) => t,
        Err(_) => return String::new(),
    };
    if text.len() > 500 {
        text
    } else {
        String::new()
    }
}

/// `GET /api/game/<game_id>/shifts`.
async fn api_game_shifts(
    State(state): State<AppState>,
    Path(game_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let force = params.get("force").map(|s| is_truthy(s)).unwrap_or(false);
    let no_cache = params
        .get("nocache")
        .map(|s| is_truthy(s))
        .unwrap_or(false)
        || params
            .get("cache")
            .map(|s| matches!(s.trim().to_lowercase().as_str(), "0" | "false"))
            .unwrap_or(false);
    let no_disk = params
        .get("nodisk")
        .map(|s| is_truthy(s))
        .unwrap_or(false)
        || params
            .get("disk")
            .map(|s| matches!(s.trim().to_lowercase().as_str(), "0" | "false"))
            .unwrap_or(false);
    match load_shifts_for_game(&state, game_id, force, no_cache, no_disk).await {
        Ok(payload) => {
            if force {
                return json_no_store(payload);
            }
            (StatusCode::OK, Json(payload)).into_response()
        }
        Err(r) => r,
    }
}

// ---------------------------------------------------------------------------
// PBP route
// ---------------------------------------------------------------------------

fn parse_time_to_seconds(t: &str) -> Option<i64> {
    let (mm, ss) = t.trim().split_once(':')?;
    Some(mm.parse::<i64>().ok()? * 60 + ss.parse::<i64>().ok()?)
}

fn strength_from_situation(code: &str, event_owner: Option<i64>, away_id: Option<i64>, home_id: Option<i64>) -> String {
    let s = code;
    if s.is_empty() {
        return String::new();
    }
    let away_empty = s.starts_with('0');
    let home_empty = s.ends_with('0');
    if away_empty || home_empty {
        if event_owner == away_id {
            return if away_empty { "ENF".to_string() } else { "ENA".to_string() };
        }
        if event_owner == home_id {
            return if home_empty { "ENF".to_string() } else { "ENA".to_string() };
        }
        return String::new();
    }
    if s.len() == 4 && s.chars().all(|c| c.is_ascii_digit()) {
        let away_skaters = s.chars().nth(1).and_then(|c| c.to_digit(10)).unwrap_or(0) as i64;
        let home_skaters = s.chars().nth(2).and_then(|c| c.to_digit(10)).unwrap_or(0) as i64;
        return match event_owner {
            None => format!("{away_skaters}v{home_skaters}"),
            Some(o) if Some(o) == home_id => format!("{home_skaters}v{away_skaters}"),
            Some(o) if Some(o) == away_id => format!("{away_skaters}v{home_skaters}"),
            _ => format!("{away_skaters}v{home_skaters}"),
        };
    }
    s.to_string()
}

fn remap_strength_state(s: &str) -> String {
    let s2 = s.to_uppercase();
    match s2.as_str() {
        "5V4" | "ENF" => "PP1".to_string(),
        "5V3" | "4V3" => "PP2".to_string(),
        "4V5" | "3V5" | "3V4" => "SH".to_string(),
        _ => s.to_string(),
    }
}

fn compute_boxid2(boxid: &str, shoots: &str) -> String {
    let b = boxid.trim().to_uppercase();
    let h = if shoots.is_empty() { String::new() } else { shoots.trim().to_uppercase().chars().next().unwrap().to_string() };
    let is_r_or_null = h == "R" || h.is_empty();
    let is_l_or_null = h == "L" || h.is_empty();
    match b.as_str() {
        "O01" | "O03" => "O01".to_string(),
        "O02" => "O02".to_string(),
        "O04" => if is_r_or_null { "O04-W" } else { "O04-S" }.to_string(),
        "O05" => if is_r_or_null { "O05-W" } else { "O05-S" }.to_string(),
        "O06" => if is_r_or_null { "O06-W" } else { "O06-S" }.to_string(),
        "O07" => "O07".to_string(),
        "O08" => if is_l_or_null { "O06-W" } else { "O06-S" }.to_string(),
        "O09" => if is_l_or_null { "O05-W" } else { "O05-S" }.to_string(),
        "O10" => if is_l_or_null { "O04-W" } else { "O04-S" }.to_string(),
        "O11" => "O11".to_string(),
        "O12" => if is_r_or_null { "O12-W" } else { "O12-S" }.to_string(),
        "O13" => if is_r_or_null { "O13-W" } else { "O13-S" }.to_string(),
        "O14" => if is_r_or_null { "O14-W" } else { "O14-S" }.to_string(),
        "O15" => "O15".to_string(),
        "O16" => if is_l_or_null { "O14-W" } else { "O14-S" }.to_string(),
        "O17" => if is_l_or_null { "O13-W" } else { "O13-S" }.to_string(),
        "O18" => if is_l_or_null { "O12-W" } else { "O12-S" }.to_string(),
        "O19" => if is_r_or_null { "O19-W" } else { "O19-S" }.to_string(),
        "O20" => if is_r_or_null { "O20-W" } else { "O20-S" }.to_string(),
        "O21" => "O21".to_string(),
        "O22" => if is_l_or_null { "O20-W" } else { "O20-S" }.to_string(),
        "O23" => if is_l_or_null { "O19-W" } else { "O19-S" }.to_string(),
        "O24" => if is_r_or_null { "O24-W" } else { "O24-S" }.to_string(),
        "O25" => "O25".to_string(),
        "O26" => if is_l_or_null { "O24-W" } else { "O24-S" }.to_string(),
        _ => "D_or_N".to_string(),
    }
}

/// `GET /api/game/<game_id>/play-by-play` — port of `api_game_pbp`.
async fn api_game_pbp(
    State(state): State<AppState>,
    Path(game_id): Path<i64>,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let force = params.get("force").map(|s| is_truthy(s)).unwrap_or(false);
    let lite_mode = params
        .get("lite")
        .map(|s| is_truthy(s))
        .unwrap_or(false);
    let std_ttl = state.caches.pbp_ttl_secs();
    let xg_scope = normalize_xg_model_name(params.get("xgModel").map(|s| s.as_str()))
        .unwrap_or_else(|| "all".to_string());
    let cache_scope = if lite_mode {
        format!("{xg_scope}_lite")
    } else {
        xg_scope.clone()
    };
    let cache_key = (game_id, cache_scope.clone());
    let disk_base = disk_cache::disk_cache_base(state.cfg.xg_cache_dir.as_deref());
    let disk_path = disk_cache::pbp_path(
        &disk_base,
        game_id,
        if cache_scope == "all" { None } else { Some(&cache_scope) },
    );

    if !force {
        if let Some(js) = disk_cache::read_json_if_fresh(&disk_path, 5.0, std_ttl as f64) {
            let payload: Map<String, Value> = js
                .as_object()
                .map(|o| {
                    o.iter()
                        .filter(|(k, _)| !k.starts_with('_'))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect()
                })
                .unwrap_or_default();
            return (StatusCode::OK, Json(Value::Object(payload))).into_response();
        }
        if let Some(cached) = state.caches.pbp.get(&cache_key) {
            return (StatusCode::OK, Json(cached)).into_response();
        }
    }

    let url = format!("{API_WEB}/v1/gamecenter/{game_id}/play-by-play");
    let data = match get_json(&state.http, &url, 25).await {
        Ok(d) => d,
        Err(_) => return json_err(StatusCode::BAD_GATEWAY, json!({"error": "Fetch failed"})),
    };
    let game_state = str_value(data.get("gameState").or_else(|| data.get("gameStatus"))).to_uppercase();

    // Skater bios -> shoots map (optional).
    let mut shoots_map: HashMap<i64, String> = HashMap::new();
    if std::env::var("FETCH_BIOS").map(|v| v == "1").unwrap_or(false) {
        let gid_for_bios = data.get("id").and_then(Value::as_i64).unwrap_or(game_id);
        let bios_url = format!(
            "https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=gameId={gid_for_bios}"
        );
        if let Ok(bios_json) = get_json_with_ua(&state.http, &bios_url, 15).await {
            if let Some(rows) = bios_json.get("data").and_then(Value::as_array) {
                for row in rows {
                    let pid = row.get("playerId").and_then(Value::as_i64);
                    let sc = row
                        .get("shootsCatches")
                        .or_else(|| row.get("shoots"))
                        .or_else(|| row.get("ShootsCatches"))
                        .and_then(Value::as_str)
                        .map(|s| s.trim().to_uppercase());
                    if let (Some(pid), Some(sc)) = (pid, sc) {
                        let first = sc.chars().next().map(|c| c.to_string()).unwrap_or_default();
                        if !first.is_empty() {
                            shoots_map.insert(pid, first);
                        }
                    }
                }
            }
        }
    }

    // Landing -> goal highlight URLs.
    let mut goal_highlights: HashMap<i64, String> = HashMap::new();
    let landing_url = format!("{API_WEB}/v1/gamecenter/{game_id}/landing");
    if let Ok(land_data) = get_json(&state.http, &landing_url, 15).await {
        if let Some(scoring) = land_data.get("summary").and_then(|s| s.get("scoring")).and_then(Value::as_array) {
            for per in scoring {
                if let Some(goals) = per.get("goals").and_then(Value::as_array) {
                    for gl in goals {
                        let eid = gl.get("eventId").and_then(Value::as_i64);
                        let clip = gl.get("highlightClipSharingUrl").and_then(Value::as_str).map(|s| s.to_string());
                        if let (Some(eid), Some(clip)) = (eid, clip) {
                            if !clip.is_empty() {
                                goal_highlights.insert(eid, clip);
                            }
                        }
                    }
                }
            }
        }
    }

    let plays_raw = data.get("plays").and_then(Value::as_array).cloned().unwrap_or_default();
    let away_team = data.get("awayTeam").cloned().unwrap_or(Value::Null);
    let home_team = data.get("homeTeam").cloned().unwrap_or(Value::Null);
    let away_id = away_team.get("id").and_then(Value::as_i64);
    let home_id = home_team.get("id").and_then(Value::as_i64);
    let away_abbrev = away_team.get("abbrev").and_then(Value::as_str).map(|s| s.to_string());
    let home_abbrev = home_team.get("abbrev").and_then(Value::as_str).map(|s| s.to_string());

    let rink_venue_value: Option<String> = data
        .get("season")
        .and_then(Value::as_str)
        .zip(home_abbrev.as_ref())
        .map(|(sv, ha)| format!("{sv}-{ha}"));

    let roster: HashMap<i64, Value> = data
        .get("rosterSpots")
        .and_then(Value::as_array)
        .map(|arr| {
            arr.iter()
                .filter_map(|r| {
                    safe_int(r.get("playerId")).map(|pid| (pid, r.clone()))
                })
                .collect()
        })
        .unwrap_or_default();

    // season_player_info (used for position/shoots when bios unavailable).
    let mut season_player_info: Value = Value::Null;
    if shoots_map.is_empty() {
        let season = safe_int(data.get("season")).unwrap_or(0);
        if season > 0 {
            season_player_info =
                rosters::all_rosters_for_season(&state.caches, &state.http, season).await;
        }
    }

    fn player_name<'a>(roster: &'a HashMap<i64, Value>, pid: Option<i64>) -> Option<String> {
        let pid = pid?;
        let r = roster.get(&pid)?;
        let fn_ = match r.get("firstName") {
            Some(Value::Object(m)) => m.get("default").and_then(Value::as_str).map(|s| s.to_string()),
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        let ln = match r.get("lastName") {
            Some(Value::Object(m)) => m.get("default").and_then(Value::as_str).map(|s| s.to_string()),
            Some(Value::String(s)) => Some(s.clone()),
            _ => None,
        };
        match (fn_, ln) {
            (Some(f), Some(l)) => Some(format!("{f} {l}").trim().to_string()),
            (Some(f), None) => Some(f),
            (None, Some(l)) => Some(l),
            _ => None,
        }
    }

    // Orientation: sum of x per (period, owner).
    let mut period_team_sum_x: HashMap<(i64, i64), f64> = HashMap::new();
    let mut period_sum_all: HashMap<i64, f64> = HashMap::new();
    for pl in &plays_raw {
        let pd = pl
            .get("periodDescriptor")
            .and_then(|p| p.get("number"))
            .and_then(Value::as_i64);
        let Some(pd_key) = pd else { continue };
        let tc = pl.get("typeCode").and_then(Value::as_i64);
        if !matches!(tc, Some(505) | Some(506) | Some(507) | Some(508)) {
            continue;
        }
        let d0 = pl.get("details").cloned().unwrap_or(Value::Null);
        let x0 = d0.get("xCoord").and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|i| i as f64)));
        let Some(xx) = x0 else { continue };
        *period_sum_all.entry(pd_key).or_insert(0.0) += xx;
        let owner0 = d0.get("eventOwnerTeamId").and_then(Value::as_i64);
        if let Some(owner0) = owner0 {
            if Some(owner0) == home_id || Some(owner0) == away_id {
                *period_team_sum_x.entry((pd_key, owner0)).or_insert(0.0) += xx;
            }
        }
    }

    // Shifts join (skipped in lite mode).
    let mut slices: Vec<(i64, i64, i64)> = Vec::new(); // (start, end, shift_index)
    let mut starts: Vec<i64> = Vec::new();
    let mut slice_players: HashMap<i64, Vec<Value>> = HashMap::new();
    if !lite_mode {
        if let Ok(shifts_obj) = load_shifts_for_game(&state, game_id, false, false, false).await {
            if let Some(rows) = shifts_obj.get("shifts").and_then(Value::as_array) {
                let mut by_idx: HashMap<i64, (i64, i64)> = HashMap::new();
                for r in rows {
                    let si = safe_int(r.get("ShiftIndex"));
                    let st = safe_int(r.get("Start"));
                    let en = safe_int(r.get("End"));
                    let (Some(sii), Some(sti), Some(eni)) = (si, st, en) else { continue };
                    let entry = by_idx.entry(sii).or_insert((sti, eni));
                    entry.0 = entry.0.min(sti);
                    entry.1 = entry.1.max(eni);
                    let pl_id = r.get("PlayerID").cloned();
                    let pl_nm = r.get("Name").cloned();
                    let pl_pos = r.get("Position").and_then(Value::as_str).map(|s| s.to_uppercase()).unwrap_or_default();
                    let pl_tm = r.get("Team").cloned();
                    if pl_id.is_some() && pl_tm.is_some() {
                        slice_players.entry(sii).or_default().push(json!({
                            "PlayerID": pl_id,
                            "Name": pl_nm,
                            "Position": pl_pos,
                            "Team": pl_tm,
                        }));
                    }
                }
                let mut tmp: Vec<(i64, i64, i64)> = by_idx
                    .iter()
                    .map(|(k, (a, b))| (*a, *b, *k))
                    .collect();
                tmp.sort_by_key(|x| x.0);
                slices = tmp;
                starts = slices.iter().map(|s| s.0).collect();
            }
        }
    }

    // On-ice cache per shift index.
    let mut onice_cache: HashMap<i64, Map<String, Value>> = HashMap::new();
    if !lite_mode {
        for (si, plist) in &slice_players {
            let home_abbr = home_abbrev.clone().unwrap_or_default();
            let away_abbr = away_abbrev.clone().unwrap_or_default();
            let filter_and_sort = |team_abbr: &str, pos_code: &str| -> (Option<String>, Option<String>) {
                let mut flt: Vec<&Value> = plist
                    .iter()
                    .filter(|p| {
                        str_value(p.get("Team")) == team_abbr
                            && str_value(p.get("Position")).to_uppercase() == pos_code
                    })
                    .collect();
                flt.sort_by_key(|p| safe_int(p.get("PlayerID")).unwrap_or(0));
                let ids: Vec<String> = flt
                    .iter()
                    .filter_map(|p| {
                        safe_int(p.get("PlayerID")).map(|x| x.to_string())
                    })
                    .collect();
                let names: Vec<String> = flt
                    .iter()
                    .filter_map(|p| {
                        let n = str_value(p.get("Name"));
                        if n.is_empty() {
                            None
                        } else {
                            Some(n)
                        }
                    })
                    .collect();
                let ids = if ids.is_empty() { None } else { Some(ids.join(" ")) };
                let names = if names.is_empty() { None } else { Some(names.join(" - ")) };
                (ids, names)
            };
            let (hf_id, hf_nm) = filter_and_sort(&home_abbr, "F");
            let (hd_id, hd_nm) = filter_and_sort(&home_abbr, "D");
            let (hg_id, hg_nm) = filter_and_sort(&home_abbr, "G");
            let (af_id, af_nm) = filter_and_sort(&away_abbr, "F");
            let (ad_id, ad_nm) = filter_and_sort(&away_abbr, "D");
            let (ag_id, ag_nm) = filter_and_sort(&away_abbr, "G");
            onice_cache.insert(
                *si,
                json!({
                    "Home_Forwards_ID": hf_id, "Home_Forwards": hf_nm,
                    "Home_Defenders_ID": hd_id, "Home_Defenders": hd_nm,
                    "Home_Goalie_ID": hg_id, "Home_Goalie": hg_nm,
                    "Away_Forwards_ID": af_id, "Away_Forwards": af_nm,
                    "Away_Defenders_ID": ad_id, "Away_Defenders": ad_nm,
                    "Away_Goalie_ID": ag_id, "Away_Goalie": ag_nm,
                })
                .as_object()
                .cloned()
                .unwrap_or_default(),
            );
        }
    }

    // find_shift_index_for_event with bisect.
    let find_shift_index_for_event = |gt: i64, event_key: Option<&str>| -> Option<i64> {
        if slices.is_empty() {
            return None;
        }
        let ev = (event_key.unwrap_or("")).to_lowercase();
        let ev_norm = ev.replace('-', "_");
        let k = starts.partition_point(|&x| x < gt);
        if k < starts.len() && starts[k] == gt {
            let idx = if ev_norm == "faceoff" || ev_norm == "period_start" {
                k
            } else {
                if k > 0 { k - 1 } else { k }
            };
            return if idx < slices.len() { Some(slices[idx].2) } else { None };
        }
        let i = starts.partition_point(|&x| x <= gt).checked_sub(1)?;
        if i >= slices.len() {
            return None;
        }
        let (s0, e0, si0) = slices[i];
        if gt < e0 {
            return Some(si0);
        }
        if gt == e0 {
            if i + 1 < slices.len() && starts[i + 1] == gt {
                if ev_norm == "faceoff" || ev_norm == "period_start" {
                    return Some(slices[i + 1].2);
                }
                return Some(si0);
            }
        }
        None
    };

    // BoxID map.
    let repo_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or(std::path::Path::new("."))
        .to_path_buf();
    let box_id_map = boxids::get_boxid_map(&state.caches, state.sb.as_ref(), &repo_root).await;

    let gt_map: HashMap<&str, &str> = HashMap::from([("2", "regular"), ("3", "playoffs")]);
    let season_state = gt_map
        .get(str_value(data.get("gameType")).as_str())
        .copied()
        .unwrap_or("other")
        .to_string();
    let gid_for_idx = data.get("id").and_then(Value::as_i64).unwrap_or(game_id);

    let mut mapped: Vec<Value> = Vec::new();
    let mut running_away: i64 = 0;
    let mut running_home: i64 = 0;

    for (idx_pl, pl) in plays_raw.iter().enumerate() {
        let period = pl.get("periodDescriptor").and_then(|p| p.get("number")).and_then(Value::as_i64);
        let time_in_period = str_value(pl.get("timeInPeriod"));
        let type_code = pl.get("typeCode").and_then(Value::as_i64);
        let event_key = str_value(pl.get("typeDescKey"));
        let details = pl.get("details").cloned().unwrap_or(Value::Null);
        let situation = str_value(pl.get("situationCode"));
        let strength = strength_from_situation(&situation, details.get("eventOwnerTeamId").and_then(Value::as_i64), away_id, home_id);
        let x = details.get("xCoord").and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)));
        let y = details.get("yCoord").and_then(|v| v.as_f64().or_else(|| v.as_i64().map(|i| i as f64)));
        let mut zone = str_value(details.get("zoneCode"));
        let reason = details.get("reason").cloned();
        let secondary_reason = details.get("secondaryReason").cloned();
        let type_code2 = details.get("typeCode").and_then(Value::as_str).map(|s| s.to_string());
        let pen_dur = details.get("duration").cloned();
        let event_owner = details.get("eventOwnerTeamId").and_then(Value::as_i64);
        let event_team_abbrev = if event_owner == away_id {
            away_abbrev.clone()
        } else if event_owner == home_id {
            home_abbrev.clone()
        } else {
            None
        };
        let opponent_abbrev = if event_team_abbrev.as_ref() == away_abbrev.as_ref() {
            home_abbrev.clone()
        } else if event_team_abbrev.as_ref() == home_abbrev.as_ref() {
            away_abbrev.clone()
        } else {
            None
        };
        let goalie_id = details.get("goalieInNetId").and_then(Value::as_i64);
        let goalie_name = if goalie_id.is_some() { player_name(&roster, goalie_id) } else { None };

        // candidate ids in priority order
        let mut candidate_ids: Vec<i64> = Vec::new();
        for key in [
            "scoringPlayerId", "shootingPlayerId", "playerId", "hittingPlayerId",
            "hitteePlayerId", "assist1PlayerId", "assist2PlayerId", "blockingPlayerId",
            "losingPlayerId", "winningPlayerId", "committedByPlayerId", "drawnByPlayerId",
        ] {
            if let Some(pid) = details.get(key).and_then(Value::as_i64) {
                if !candidate_ids.contains(&pid) {
                    candidate_ids.push(pid);
                }
            }
        }
        let p1_id = candidate_ids.first().copied();
        let p2_id = candidate_ids.get(1).copied();
        let p3_id = candidate_ids.get(2).copied();
        let p1_name = player_name(&roster, p1_id);
        let p2_name = player_name(&roster, p2_id);
        let p3_name = player_name(&roster, p3_id);

        let is_goal = type_code == Some(505);
        let is_sog = type_code == Some(506) || is_goal;
        let is_miss = type_code == Some(507);
        let is_block = type_code == Some(508);
        if is_block && (zone == "O" || zone == "D") {
            zone = if zone == "D" { "O".to_string() } else { "D".to_string() };
        }

        // Normalize coords.
        let mut nx: Option<f64> = None;
        let mut ny: Option<f64> = None;
        let sign = if let Some(pd_key2) = period {
            if let Some(owner0) = event_owner {
                if Some(owner0) == home_id || Some(owner0) == away_id {
                    if let Some(&s) = period_team_sum_x.get(&(pd_key2, owner0)) {
                        if s >= 0.0 { 1.0 } else { -1.0 }
                    } else {
                        let opp = if Some(owner0) == away_id { home_id } else { away_id };
                        if let Some(opp) = opp {
                            if let Some(&s) = period_team_sum_x.get(&(pd_key2, opp)) {
                                if s >= 0.0 { -1.0 } else { 1.0 }
                            } else {
                                if period_sum_all.get(&pd_key2).copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 }
                            }
                        } else {
                            if period_sum_all.get(&pd_key2).copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 }
                        }
                    }
                } else {
                    if period_sum_all.get(&pd_key2).copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 }
                }
            } else {
                if period_sum_all.get(&pd_key2).copied().unwrap_or(0.0) >= 0.0 { 1.0 } else { -1.0 }
            }
        } else {
            1.0
        };
        if let Some(x) = x {
            nx = Some(x * sign);
        }
        if let Some(y) = y {
            ny = Some(y * sign);
        }

        // ScoreState (pre-event).
        let score_state_val = match event_owner {
            Some(o) if Some(o) == away_id => running_away - running_home,
            Some(o) if Some(o) == home_id => running_home - running_away,
            _ => running_away - running_home,
        };
        let score_state2_val = if score_state_val < -2 {
            -3
        } else if score_state_val > 2 {
            3
        } else {
            score_state_val
        };

        let corsi = if (is_goal || is_sog || is_miss || is_block) && event_team_abbrev.is_some() {
            1
        } else {
            0
        };
        let fenwick = if (is_goal || is_sog || is_miss) && event_team_abbrev.is_some() {
            1
        } else {
            0
        };
        let shot = if is_sog { 1 } else { 0 };

        // Position & shoots.
        let mut position: Option<String> = None;
        let mut shoots: Option<String> = None;
        let game_player_info = p1_id.and_then(|pid| roster.get(&pid)).cloned();
        let season_player = if let Some(pid) = p1_id {
            season_player_info.get(&pid.to_string()).cloned()
        } else {
            None
        };
        let player_info = if game_player_info.is_some() {
            game_player_info.as_ref()
        } else {
            season_player.as_ref()
        };
        if let Some(pi) = player_info {
            let pos_code = str_value(pi.get("positionCode"));
            let pos_code = if pos_code.is_empty() {
                let p2 = str_value(pi.get("position"));
                if p2.is_empty() {
                    season_player
                        .as_ref()
                        .map(|sp| str_value(sp.get("positionCode")))
                        .unwrap_or_default()
                } else {
                    p2
                }
            } else {
                pos_code
            }
            .to_uppercase();
            if !pos_code.is_empty() {
                let c = pos_code.chars().next().unwrap_or('F');
                position = Some(if matches!(c, 'C' | 'L' | 'R') { "F".to_string() } else { c.to_string() });
            }
        }
        if let Some(pid) = p1_id {
            if let Some(sc) = shoots_map.get(&pid) {
                shoots = Some(sc.clone());
            } else if let Some(sp) = &season_player {
                let shoots_raw = str_value(sp.get("shoots"));
                let shoots_raw = if shoots_raw.is_empty() {
                    let s2 = str_value(sp.get("shootsCatches"));
                    if s2.is_empty() {
                        str_value(sp.get("catches"))
                    } else {
                        s2
                    }
                } else {
                    shoots_raw
                }
                .to_uppercase();
                let first = shoots_raw.chars().next().map(|c| c.to_string()).unwrap_or_default();
                if first == "L" || first == "R" {
                    shoots = Some(first);
                }
            }
        }

        let secs_elapsed = parse_time_to_seconds(&time_in_period).unwrap_or(0);
        let game_time = if let Some(period) = period {
            (period - 1) * 20 * 60 + secs_elapsed
        } else {
            secs_elapsed
        };

        let venue_ha = if event_owner == home_id {
            "Home".to_string()
        } else if event_owner == away_id {
            "Away".to_string()
        } else {
            String::new()
        };

        // Shot geometry.
        let mut shot_distance: Option<f64> = None;
        let mut shot_angle: Option<f64> = None;
        if nx.is_some() && ny.is_some() && (is_goal || is_sog || is_miss || is_block) {
            let dx = 89.0 - nx.unwrap();
            let dy = 0.0 - ny.unwrap();
            let dist = (dx * dx + dy * dy).sqrt();
            let ang = dy.abs().atan2(if dx != 0.0 { dx } else { 1e-6 }).to_degrees();
            shot_distance = Some(round_to(dist, 2));
            shot_angle = Some(round_to(ang, 2));
        }

        let event_index_val = gid_for_idx * 10000 + (idx_pl as i64 + 1);
        let shift_index_val = find_shift_index_for_event(game_time, if event_key.is_empty() { None } else { Some(event_key.as_str()) });

        let oi = onice_cache.get(&shift_index_val.unwrap_or(-1)).cloned().unwrap_or_default();

        // Box geometry.
        let mut box_id: Option<String> = None;
        let mut box_rev: Option<String> = None;
        let mut box_size: Option<i64> = None;
        let mut xi: Option<i64> = None;
        let mut yi: Option<i64> = None;
        if nx.is_some() && ny.is_some() {
            let xf = nx.unwrap().clamp(-100.0, 100.0);
            let yf = ny.unwrap().clamp(-42.0, 42.0);
            let xii = xf.round() as i64;
            let yii = yf.round() as i64;
            xi = Some(xii);
            yi = Some(yii);
            if let Some((bid, bre, bsi)) = box_id_map.get(&(xii, yii)) {
                box_id = Some(bid.clone());
                box_rev = Some(bre.clone());
                box_size = Some(*bsi);
            }
        }

        let raw_shot_type = str_value(details.get("shotType"));
        let st_lower = raw_shot_type.to_lowercase();
        let allowed = [
            "wrist", "tip-in", "snap", "slap", "backhand", "deflected", "wrap-around", "",
        ];
        let shot_type2 = if allowed.contains(&st_lower.as_str()) {
            raw_shot_type.clone()
        } else {
            "other".to_string()
        };
        let strength2 = remap_strength_state(&strength);

        let highlight = if is_goal {
            pl.get("eventId")
                .and_then(Value::as_i64)
                .and_then(|eid| goal_highlights.get(&eid))
                .cloned()
        } else {
            None
        };

        mapped.push(json!({
            "GameID": data.get("id"),
            "Season": data.get("season"),
            "SeasonState": season_state,
            "Venue": venue_ha,
            "Period": period,
            "gameTime": game_time,
            "StrengthState": strength,
            "StrengthState2": strength2,
            "typeCode": type_code,
            "Event": event_key,
            "x": nx,
            "y": ny,
            "X": xi,
            "Y": yi,
            "Zone": zone,
            "reason": reason,
            "shotType": details.get("shotType"),
            "shotType2": shot_type2,
            "secondaryReason": secondary_reason,
            "typeCode2": type_code2,
            "PEN_duration": pen_dur,
            "EventTeam": event_team_abbrev,
            "Opponent": opponent_abbrev,
            "Goalie_ID": goalie_id,
            "Goalie": goalie_name,
            "Player1_ID": p1_id,
            "Player1": p1_name,
            "Player2_ID": p2_id,
            "Player2": p2_name,
            "Player3_ID": p3_id,
            "Player3": p3_name,
            "Corsi": corsi,
            "Fenwick": fenwick,
            "Shot": shot,
            "Goal": if is_goal { 1 } else { 0 },
            "EventIndex": event_index_val,
            "ShiftIndex": shift_index_val,
            "ScoreState": score_state_val,
            "ScoreState2": score_state2_val,
            "Home_Forwards_ID": oi.get("Home_Forwards_ID"),
            "Home_Forwards": oi.get("Home_Forwards"),
            "Home_Defenders_ID": oi.get("Home_Defenders_ID"),
            "Home_Defenders": oi.get("Home_Defenders"),
            "Home_Goalie_ID": oi.get("Home_Goalie_ID"),
            "Home_Goalie": oi.get("Home_Goalie"),
            "Away_Forwards_ID": oi.get("Away_Forwards_ID"),
            "Away_Forwards": oi.get("Away_Forwards"),
            "Away_Defenders_ID": oi.get("Away_Defenders_ID"),
            "Away_Defenders": oi.get("Away_Defenders"),
            "Away_Goalie_ID": oi.get("Away_Goalie_ID"),
            "Away_Goalie": oi.get("Away_Goalie"),
            "BoxID": box_id,
            "BoxID_rev": box_rev,
            "BoxSize": box_size,
            "BoxID2": Value::Null,
            "box_id": box_id,
            "box_rev": box_rev,
            "box_size": box_size,
            "ShotDistance": shot_distance,
            "ShotAngle": shot_angle,
            "Position": position,
            "Shoots": shoots,
            "RinkVenue": rink_venue_value,
            "HighlightUrl": highlight,
            "LastEvent": Value::Null,
            "xG_F": Value::Null,
            "xG_S": Value::Null,
            "xG_F2": Value::Null,
        }));

        // Update running score after goal.
        if is_goal {
            let ra = details.get("awayScore").and_then(Value::as_i64);
            let rh = details.get("homeScore").and_then(Value::as_i64);
            match (ra, rh) {
                (Some(ra), Some(rh)) => {
                    running_away = ra;
                    running_home = rh;
                }
                _ => {
                    if event_owner == away_id {
                        running_away += 1;
                    } else if event_owner == home_id {
                        running_home += 1;
                    }
                }
            }
        }
    }

    // Post-process BoxID2 + LastEvent.
    let mut last_event_name: Option<String> = None;
    let mut last_game_time: Option<i64> = None;
    for row in &mut mapped {
        let boxid = str_value(row.get("BoxID"));
        let shoots = str_value(row.get("Shoots"));
        row["BoxID2"] = json!(compute_boxid2(&boxid, &shoots));
        if i_of(row, "Fenwick") == 1 {
            let prev_ev = last_event_name.clone().unwrap_or_default();
            let gt = row.get("gameTime").and_then(Value::as_i64);
            let tsle = match (gt, last_game_time) {
                (Some(g), Some(l)) => Some(g - l),
                _ => None,
            };
            let last_event = if let Some(ts) = tsle {
                if ts < 4 && matches!(prev_ev.as_str(), "blocked-shot" | "shot-on-goal" | "takeaway" | "giveaway") {
                    "Rebound".to_string()
                } else if ts < 4 {
                    "Quick".to_string()
                } else {
                    "None".to_string()
                }
            } else {
                "None".to_string()
            };
            row["LastEvent"] = json!(last_event);
        } else {
            row["LastEvent"] = json!("");
        }
        let ev = str_value(row.get("Event"));
        if !ev.is_empty() {
            last_event_name = Some(ev);
        }
        if let Some(gt) = row.get("gameTime").and_then(Value::as_i64) {
            last_game_time = Some(gt);
        }
    }

    // xG computation.
    compute_xg(
        &state,
        &xg_scope,
        &mut mapped,
        |row| {
            row.get("Season")
                .and_then(Value::as_i64)
                .or_else(|| row.get("Season").and_then(Value::as_str).and_then(|s| s.parse().ok()))
        },
    );

    // Sanitize: null non-finite floats.
    for row in &mut mapped {
        if let Value::Object(o) = row {
            for v in o.values_mut() {
                if let Value::Number(n) = v {
                    if let Some(f) = n.as_f64() {
                        if !f.is_finite() {
                            *v = Value::Null;
                        }
                    }
                }
            }
        }
    }

    let out_obj = json!({
        "gameId": data.get("id"),
        "plays": mapped,
        "gameState": game_state,
    });
    state
        .caches
        .pbp
        .insert(cache_key, out_obj.clone());
    let mut js = out_obj.clone();
    if let Value::Object(o) = &mut js {
        o.insert("_cachedAt".to_string(), json!(now_epoch()));
    }
    disk_cache::write_json(&disk_path, &js).ok();
    if force {
        return json_no_store(out_obj);
    }
    (StatusCode::OK, Json(out_obj)).into_response()
}

fn round_to(x: f64, places: i32) -> f64 {
    let m = 10f64.powi(places);
    (x * m).round() / m
}

/// Port of the xG section of `api_game_pbp` (ENA + windowed model families).
fn compute_xg(
    state: &AppState,
    xg_scope: &str,
    mapped: &mut [Value],
    season_of: impl Fn(&Value) -> Option<i64>,
) {
    if std::env::var("XG_DISABLED").map(|v| v == "1").unwrap_or(false) {
        return;
    }
    let requested_family = match xg_scope {
        "xG_S" => Some("xgbs"),
        "xG_F" => Some("xgb"),
        "xG_F2" => Some("xgb2"),
        _ => None,
    };
    let model_dir = state.cfg.model_dir.join("rust");

    fn compute_empty_net_fenwick(sd: Option<f64>, sa: Option<f64>) -> Option<f64> {
        let (sd, sa) = (sd?, sa?);
        let val = 1.0 / (1.0 + (0.013609495 * sd + 0.023174225 * sa.abs() - 1.97392131).exp());
        Some(val)
    }

    // 1) ENA upfront.
    for row in mapped.iter_mut() {
        if str_value(row.get("StrengthState")) == "ENA" {
            if i_of(row, "Shot") == 1 && matches!(requested_family, None | Some("xgbs")) {
                row["xG_S"] = json!(1.0);
            }
            if i_of(row, "Fenwick") == 1 {
                let val_en = compute_empty_net_fenwick(
                    row.get("ShotDistance").and_then(Value::as_f64),
                    row.get("ShotAngle").and_then(Value::as_f64),
                );
                if let Some(v) = val_en {
                    if matches!(requested_family, None | Some("xgb")) {
                        row["xG_F"] = json!(round_to(v, 6));
                    }
                    if matches!(requested_family, None | Some("xgb2")) {
                        row["xG_F2"] = json!(round_to(v, 6));
                    }
                }
            }
        }
    }

    // 2) Family grouping.
    let mut families: Vec<(&str, usize)> = Vec::new();
    for (i, r) in mapped.iter().enumerate() {
        let shot = i_of(r, "Shot") == 1;
        let fenwick = i_of(r, "Fenwick") == 1;
        let not_ena = str_value(r.get("StrengthState")) != "ENA";
        if shot && not_ena && matches!(requested_family, None | Some("xgbs")) {
            families.push(("xgbs", i));
        }
        if fenwick && not_ena {
            if matches!(requested_family, None | Some("xgb")) {
                families.push(("xgb", i));
            }
            if matches!(requested_family, None | Some("xgb2")) {
                families.push(("xgb2", i));
            }
        }
    }
    let mut by_family: HashMap<&str, Vec<usize>> = HashMap::new();
    for (f, i) in families {
        by_family.entry(f).or_default().push(i);
    }

    for (family, idxs) in by_family {
        if idxs.is_empty() {
            continue;
        }
        // Group by season.
        let mut by_season: BTreeMap<i64, Vec<usize>> = BTreeMap::new();
        for &i in &idxs {
            if let Some(s) = season_of(&mapped[i]) {
                by_season.entry(s).or_default().push(i);
            }
        }
        for (s_cur, row_idx) in by_season {
            let names = xg::window_filenames_for_season(s_cur, family);
            let n_windows = xg::num_windows();
            let mut models: Vec<std::sync::Arc<xg::XgModel>> = Vec::new();
            for n in &names {
                if let Some(m) = xg::load_model_file(&state.caches, n, &model_dir) {
                    models.push(m);
                    if models.len() >= n_windows {
                        break;
                    }
                }
            }
            if models.is_empty() {
                continue;
            }
            // Vectorize + predict each row, average across windows.
            let mut preds_accum: Vec<Vec<f64>> = vec![Vec::new(); row_idx.len()];
            for model in &models {
                for (j, &i) in row_idx.iter().enumerate() {
                    let Some(obj) = mapped[i].as_object() else { continue };
                    let vec = xg::vectorize_row(model, obj);
                    preds_accum[j].push(model.predict(&vec));
                }
            }
            for (j, &i) in row_idx.iter().enumerate() {
                if preds_accum[j].is_empty() {
                    continue;
                }
                let avgp = preds_accum[j].iter().sum::<f64>() / preds_accum[j].len() as f64;
                match family {
                    "xgbs" => mapped[i]["xG_S"] = json!(round_to(avgp, 6)),
                    "xgb" => mapped[i]["xG_F"] = json!(round_to(avgp, 6)),
                    "xgb2" => mapped[i]["xG_F2"] = json!(round_to(avgp, 6)),
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xg_model_name_normalization() {
        assert_eq!(normalize_xg_model_name(Some("xG_F")), Some("xG_F".to_string()));
        assert_eq!(normalize_xg_model_name(Some("xg-s")), Some("xG_S".to_string()));
        assert_eq!(normalize_xg_model_name(Some("foo")), None);
        assert_eq!(normalize_xg_model_name(None), None);
    }

    #[test]
    fn strength_from_situation_cases() {
        let (away, home) = (Some(1), Some(2));
        // Numeric branch (no leading/trailing 0): s[1]=away skaters, s[2]=home skaters.
        assert_eq!(strength_from_situation("5454", Some(1), away, home), "4v5");
        assert_eq!(strength_from_situation("5454", Some(2), away, home), "5v4");
        assert_eq!(strength_from_situation("5444", Some(1), away, home), "4v4");
        assert_eq!(strength_from_situation("5555", Some(1), away, home), "5v5");
        assert_eq!(strength_from_situation("5555", None, away, home), "5v5");
        // Empty-net branch: leading 0 => away's net empty; trailing 0 => home's net empty.
        assert_eq!(strength_from_situation("0454", Some(1), away, home), "ENF");
        assert_eq!(strength_from_situation("0454", Some(2), away, home), "ENA");
        assert_eq!(strength_from_situation("4540", Some(2), away, home), "ENF");
        assert_eq!(strength_from_situation("4540", Some(1), away, home), "ENA");
        assert_eq!(strength_from_situation("0054", Some(1), away, home), "ENF");
        // Unknown owner with an empty net -> empty string.
        assert_eq!(strength_from_situation("0454", None, away, home), "");
        // Empty code.
        assert_eq!(strength_from_situation("", Some(1), away, home), "");
    }

    #[test]
    fn boxid2_mapping() {
        assert_eq!(compute_boxid2("O01", "R"), "O01");
        assert_eq!(compute_boxid2("O04", "R"), "O04-W");
        assert_eq!(compute_boxid2("O04", "L"), "O04-S");
        assert_eq!(compute_boxid2("O08", "L"), "O06-W");
        assert_eq!(compute_boxid2("O08", "R"), "O06-S");
        assert_eq!(compute_boxid2("D05", "R"), "D_or_N");
        assert_eq!(compute_boxid2("", ""), "D_or_N");
    }

    #[test]
    fn remap_strength() {
        assert_eq!(remap_strength_state("5v4"), "PP1");
        assert_eq!(remap_strength_state("ENF"), "PP1");
        assert_eq!(remap_strength_state("5v3"), "PP2");
        assert_eq!(remap_strength_state("4v5"), "SH");
        assert_eq!(remap_strength_state("5v5"), "5v5");
    }
}
