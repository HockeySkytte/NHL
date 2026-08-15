//! SeasonStats aggregation builders — ports of `_iter_seasonstats_static_rows`,
//! `_build_seasonstats_agg`, `_build_goalies_seasonstats_agg`,
//! `_build_goalies_career_season_matrix`, `_build_team_base_stats` and
//! `_team_stats_rest_get` from `app/routes.py`.

use std::collections::HashMap;

use serde_json::{json, Map, Value};

use crate::state::Caches;
use crate::config::Config;
use crate::nhl::client::{get_json_with_ua, API_STATS};
use crate::supabase::read::{filters, read_each, SbClient};
use crate::util::parse::{ci_get, parse_locale_float, safe_int, str_value};

pub fn season_stats_col_map() -> HashMap<String, String> {
    [
        ("season", "Season"),
        ("season_state", "SeasonState"),
        ("strength_state", "StrengthState"),
        ("player_id", "PlayerID"),
        ("position", "Position"),
        ("gp", "GP"),
        ("plus_minus", "plusMinus"),
        ("blocked_shots", "blockedShots"),
        ("toi", "TOI"),
        ("i_goals", "iGoals"),
        ("assists1", "Assists1"),
        ("assists2", "Assists2"),
        ("i_corsi", "iCorsi"),
        ("i_fenwick", "iFenwick"),
        ("i_shots", "iShots"),
        ("ixg_f", "ixG_F"),
        ("ixg_s", "ixG_S"),
        ("ixg_f2", "ixG_F2"),
        ("pim_taken", "PIM_taken"),
        ("pim_drawn", "PIM_drawn"),
        ("hits", "Hits"),
        ("takeaways", "Takeaways"),
        ("giveaways", "Giveaways"),
        ("so_goal", "SO_Goal"),
        ("so_attempt", "SO_Attempt"),
        ("ca", "CA"),
        ("cf", "CF"),
        ("fa", "FA"),
        ("ff", "FF"),
        ("sa", "SA"),
        ("sf", "SF"),
        ("ga", "GA"),
        ("gf", "GF"),
        ("xga_f", "xGA_F"),
        ("xgf_f", "xGF_F"),
        ("xga_s", "xGA_S"),
        ("xgf_s", "xGF_S"),
        ("xga_f2", "xGA_F2"),
        ("xgf_f2", "xGF_F2"),
        ("pim_for", "PIM_for"),
        ("pim_against", "PIM_against"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// `_iter_seasonstats_static_rows(seasons=...)` — Supabase-only, streamed:
/// rows are delivered page-by-page to `on_page` and dropped immediately, so a
/// full-table (career-scope) scan never materializes ~135k rows (hundreds of
/// MB as serde Values) in memory at once. Failed season reads are skipped
/// (same as the previous per-season loop); a failed career read yields no
/// rows.
pub(crate) async fn for_each_player_row(
    sb: Option<&SbClient>,
    seasons: &[i64],
    goalie_only: bool,
    mut on_page: impl FnMut(Vec<Value>),
) {
    let Some(sb) = sb else { return };
    let cm = season_stats_col_map();
    if !seasons.is_empty() {
        for s in seasons {
            let mut f = std::collections::BTreeMap::new();
            f.insert("season".to_string(), format!("eq.{s}"));
            if goalie_only {
                f.insert("position".to_string(), "eq.G".to_string());
            }
            let _ = read_each(sb, "season_stats", "*", Some(&f), Some(&cm), None, 0, |page| on_page(page))
                .await;
        }
        return;
    }
    // Career scope: stream all rows (optionally goalies only).
    let f = if goalie_only { Some(filters(&[("position", "eq.G")])) } else { None };
    let _ = read_each(sb, "season_stats", "*", f.as_ref(), Some(&cm), None, 0, |page| on_page(page))
        .await;
}

fn flt(v: Option<&Value>) -> f64 {
    parse_locale_float(v).unwrap_or(0.0)
}

fn i64v(v: Option<&Value>) -> i64 {
    safe_int(v).unwrap_or(0)
}

fn parse_ss(v: Option<&Value>) -> String {
    let raw = str_value(v).to_lowercase();
    match raw.as_str() {
        "2" | "reg" | "regular" | "regularseason" | "regular_season" => "regular".to_string(),
        "3" | "po" | "playoffs" | "playoff" => "playoffs".to_string(),
        other => {
            if other.is_empty() {
                "regular".to_string()
            } else {
                other.to_string()
            }
        }
    }
}

/// Aggregation state for `build_skater_agg` — rows are ingested page-by-page
/// from the Supabase stream and dropped, so peak memory is the (small) final
/// per-player aggregate rather than the full row set.
struct SkaterAggState {
    scope_career: bool,
    primary: i64,
    ss_norm: &'static str,
    st_norm: String,
    agg: Map<String, Value>,
    pos_group: Map<String, Value>,
    gp_max: HashMap<(i64, i64, String), i64>,
}

impl SkaterAggState {
    fn new(scope_career: bool, primary: i64, ss_norm: &'static str, st_norm: String) -> Self {
        Self {
            scope_career,
            primary,
            ss_norm,
            st_norm,
            agg: Map::new(),
            pos_group: Map::new(),
            gp_max: HashMap::new(),
        }
    }

    fn ingest(&mut self, obj: &Map<String, Value>) {
        let pos = str_value(ci_get(obj, "Position")).to_uppercase();
        if pos.starts_with('G') {
            return;
        }
        let season_row = i64v(ci_get(obj, "Season"));
        let season_row = if season_row == 0 {
            if self.scope_career { 20252026 } else { self.primary }
        } else {
            season_row
        };
        let ss = parse_ss(ci_get(obj, "SeasonState"));
        let st = str_value(ci_get(obj, "StrengthState"));
        let st = if st.is_empty() { "Other".to_string() } else { st };
        if self.ss_norm != "all" && ss != self.ss_norm {
            return;
        }
        if self.st_norm != "all" && st != self.st_norm {
            return;
        }
        let pid = i64v(ci_get(obj, "PlayerID"));
        if pid <= 0 {
            return;
        }
        let gp_row = i64v(ci_get(obj, "GP"));
        let k = (pid, season_row, ss.clone());
        let prev = self.gp_max.get(&k).copied();
        if prev.is_none() || gp_row > prev.unwrap_or(0) {
            self.gp_max.insert(k, gp_row);
        }
        if !self.pos_group.contains_key(&pid.to_string()) {
            self.pos_group
                .insert(pid.to_string(), json!(if pos.starts_with('D') { "D" } else { "F" }));
        }
        let d = self
            .agg
            .entry(pid.to_string())
            .or_insert_with(empty_skater_row);
        let d = d.as_object_mut().expect("agg row object");
        sum_row(d, obj);
    }

    fn finish(mut self) -> (Value, Value) {
        let mut gp_sum: HashMap<i64, i64> = HashMap::new();
        for ((pid, _, _), gp) in &self.gp_max {
            *gp_sum.entry(*pid).or_insert(0) += *gp;
        }
        for (pid_s, d) in self.agg.iter_mut() {
            if let Some(pid) = pid_s.parse::<i64>().ok() {
                d.as_object_mut()
                    .expect("agg row object")
                    .insert("GP".into(), json!(gp_sum.get(&pid).copied().unwrap_or(0)));
            }
        }
        (Value::Object(self.agg), Value::Object(self.pos_group))
    }
}

/// `_build_seasonstats_agg` (skaters). Returns `(agg, pos_group_by_pid)` as
/// JSON objects keyed by playerId string.
#[allow(clippy::too_many_arguments)]
pub async fn build_skater_agg(
    caches: &Caches,
    sb: Option<&SbClient>,
    _cfg: &Config,
    scope: &str,
    season_int: i64,
    season_ids: &[i64],
    season_state: &str,
    strength_state: &str,
) -> (Value, Value) {
    let scope_norm = normalize_scope(scope);
    let ss_norm = normalize_ss(season_state);
    let st_norm = normalize_st(strength_state);
    let primary = primary_or(season_ids, season_int);

    let key = json!([
        scope_norm,
        if scope_norm == "season" { season_ids } else { &[][..] },
        primary,
        ss_norm,
        st_norm,
    ])
    .to_string();
    let inflight_key = format!("ssagg:{key}");
    caches
        .inflight
        .run(&inflight_key, || async {
            if let Some(cached) = caches.seasonstats_agg.get(&key) {
                if let Some(arr) = cached.as_array() {
                    if arr.len() == 2 {
                        return (arr[0].clone(), arr[1].clone());
                    }
                }
            }

            let mut aggst =
                SkaterAggState::new(scope_norm == "career", primary, ss_norm, st_norm.clone());
            let seasons: Vec<i64> = if scope_norm == "career" {
                Vec::new()
            } else if season_ids.is_empty() {
                vec![primary]
            } else {
                season_ids.to_vec()
            };
            for_each_player_row(sb, &seasons, false, |page| {
                for r in page {
                    if let Some(obj) = r.as_object() {
                        aggst.ingest(obj);
                    }
                }
            })
            .await;

            let result = aggst.finish();
            caches
                .seasonstats_agg
                .insert(key, json!([result.0.clone(), result.1.clone()]));
            result
        })
        .await
}

fn empty_skater_row() -> Value {
    json!({
        "GP": 0, "TOI": 0.0, "iGoals": 0.0, "Assists1": 0.0, "Assists2": 0.0,
        "iShots": 0.0, "iFenwick": 0.0, "ixG_S": 0.0, "ixG_F": 0.0, "ixG_F2": 0.0,
        "CA": 0.0, "CF": 0.0, "FA": 0.0, "FF": 0.0, "SA": 0.0, "SF": 0.0,
        "GA": 0.0, "GF": 0.0, "xGA_S": 0.0, "xGF_S": 0.0, "xGA_F": 0.0, "xGF_F": 0.0,
        "xGA_F2": 0.0, "xGF_F2": 0.0,
        "PIM_taken": 0.0, "PIM_drawn": 0.0, "PIM_for": 0.0, "PIM_against": 0.0,
        "Hits": 0.0, "Takeaways": 0.0, "Giveaways": 0.0,
    })
}

fn sum_row(d: &mut Map<String, Value>, obj: &Map<String, Value>) {
    for key in [
        "TOI", "iGoals", "Assists1", "Assists2", "iShots", "iFenwick",
        "ixG_S", "ixG_F", "ixG_F2", "CA", "CF", "FA", "FF", "SA", "SF", "GA", "GF",
        "xGA_S", "xGF_S", "xGA_F", "xGF_F", "xGA_F2", "xGF_F2",
        "PIM_taken", "PIM_drawn", "PIM_for", "PIM_against", "Hits", "Takeaways", "Giveaways",
    ] {
        let cur = d.get(key).and_then(Value::as_f64).unwrap_or(0.0);
        d.insert(key.to_string(), json!(cur + flt(ci_get(obj, key))));
    }
}

fn normalize_scope(scope: &str) -> &'static str {
    match scope.trim().to_lowercase().as_str() {
        "career" => "career",
        _ => "season",
    }
}

fn normalize_ss(ss: &str) -> &'static str {
    match ss.trim().to_lowercase().as_str() {
        "playoffs" => "playoffs",
        "all" => "all",
        _ => "regular",
    }
}

fn normalize_st(st: &str) -> String {
    let st = st.trim();
    match st {
        "5v5" | "PP" | "SH" | "Other" | "all" => st.to_string(),
        _ => "5v5".to_string(),
    }
}

fn primary_or(season_ids: &[i64], fallback: i64) -> i64 {
    season_ids.iter().copied().max().unwrap_or(fallback).max(0)
}

/// Aggregation state for `build_goalie_agg` — streamed page-by-page.
struct GoalieAggState {
    scope_career: bool,
    primary: i64,
    ss_norm: &'static str,
    st_norm: String,
    agg: Map<String, Value>,
    pos_group: Map<String, Value>,
    gp_max: HashMap<(i64, i64, String), i64>,
}

impl GoalieAggState {
    fn new(scope_career: bool, primary: i64, ss_norm: &'static str, st_norm: String) -> Self {
        Self {
            scope_career,
            primary,
            ss_norm,
            st_norm,
            agg: Map::new(),
            pos_group: Map::new(),
            gp_max: HashMap::new(),
        }
    }

    fn ingest(&mut self, obj: &Map<String, Value>) {
        let pos = str_value(ci_get(obj, "Position")).to_uppercase();
        if !pos.starts_with('G') {
            return;
        }
        let season_row = i64v(ci_get(obj, "Season"));
        let season_row = if season_row == 0 {
            if self.scope_career { 20252026 } else { self.primary }
        } else {
            season_row
        };
        let ss = parse_ss(ci_get(obj, "SeasonState"));
        let st = str_value(ci_get(obj, "StrengthState"));
        let st = if st.is_empty() { "Other".to_string() } else { st };
        if self.ss_norm != "all" && ss != self.ss_norm {
            return;
        }
        if self.st_norm != "all" && st != self.st_norm {
            return;
        }
        let pid = i64v(ci_get(obj, "PlayerID"));
        if pid <= 0 {
            return;
        }
        let gp_row = i64v(ci_get(obj, "GP"));
        let k = (pid, season_row, ss.clone());
        let prev = self.gp_max.get(&k).copied();
        if prev.is_none() || gp_row > prev.unwrap_or(0) {
            self.gp_max.insert(k, gp_row);
        }
        self.pos_group.insert(pid.to_string(), json!("G"));
        let d = self.agg.entry(pid.to_string()).or_insert_with(|| {
            json!({"GP": 0, "TOI": 0.0, "FA": 0.0, "SA": 0.0, "GA": 0.0,
                   "xGA_S": 0.0, "xGA_F": 0.0, "xGA_F2": 0.0})
        });
        let d = d.as_object_mut().expect("goalie agg row object");
        for key in ["TOI", "FA", "SA", "GA", "xGA_S", "xGA_F", "xGA_F2"] {
            let cur = d.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            d.insert(key.to_string(), json!(cur + flt(ci_get(obj, key))));
        }
    }

    fn finish(mut self) -> (Value, Value) {
        let mut gp_sum: HashMap<i64, i64> = HashMap::new();
        for ((pid, _, _), gp) in &self.gp_max {
            *gp_sum.entry(*pid).or_insert(0) += *gp;
        }
        for (pid_s, d) in self.agg.iter_mut() {
            if let Some(pid) = pid_s.parse::<i64>().ok() {
                d.as_object_mut()
                    .expect("goalie agg row object")
                    .insert("GP".into(), json!(gp_sum.get(&pid).copied().unwrap_or(0)));
            }
        }
        (Value::Object(self.agg), Value::Object(self.pos_group))
    }
}

/// `_build_goalies_seasonstats_agg`. Returns `(agg, pos_group)` keyed by pid.
#[allow(clippy::too_many_arguments)]
pub async fn build_goalie_agg(
    caches: &Caches,
    sb: Option<&SbClient>,
    _cfg: &Config,
    scope: &str,
    season_int: i64,
    season_ids: &[i64],
    season_state: &str,
    strength_state: &str,
) -> (Value, Value) {
    let scope_norm = normalize_scope(scope);
    let ss_norm = normalize_ss(season_state);
    let st_norm = normalize_st(strength_state);
    let primary = primary_or(season_ids, season_int);

    let key = json!([
        "goalies", scope_norm,
        if scope_norm == "season" { season_ids } else { &[][..] },
        primary, ss_norm, st_norm,
    ])
    .to_string();
    let inflight_key = format!("gagg:{key}");
    caches
        .inflight
        .run(&inflight_key, || async {
            if let Some(cached) = caches.goalies_agg.get(&key) {
                if let Some(arr) = cached.as_array() {
                    if arr.len() == 2 {
                        return (arr[0].clone(), arr[1].clone());
                    }
                }
            }

            let mut aggst =
                GoalieAggState::new(scope_norm == "career", primary, ss_norm, st_norm.clone());
            let seasons: Vec<i64> = if scope_norm == "career" {
                Vec::new()
            } else if season_ids.is_empty() {
                vec![primary]
            } else {
                season_ids.to_vec()
            };
            for_each_player_row(sb, &seasons, true, |page| {
                for r in page {
                    if let Some(obj) = r.as_object() {
                        aggst.ingest(obj);
                    }
                }
            })
            .await;

            let result = aggst.finish();
            caches
                .goalies_agg
                .insert(key, json!([result.0.clone(), result.1.clone()]));
            result
        })
        .await
}

/// Aggregation state for `build_goalies_career_matrix` — streamed page-by-page
/// over all goalie rows (only the requested goalie's rows are retained).
struct CareerMatrixState {
    ss_norm: &'static str,
    st_norm: String,
    target_pid: i64,
    by_pid_season: Map<String, Value>,
    league_acc: Map<String, Value>,
}

impl CareerMatrixState {
    fn new(ss_norm: &'static str, st_norm: String, target_pid: i64) -> Self {
        Self {
            ss_norm,
            st_norm,
            target_pid,
            by_pid_season: Map::new(),
            league_acc: Map::new(),
        }
    }

    fn ingest(&mut self, obj: &Map<String, Value>) {
        let pos = str_value(ci_get(obj, "Position")).to_uppercase();
        if !pos.starts_with('G') {
            return;
        }
        let season_row = i64v(ci_get(obj, "Season"));
        let season_row = if season_row == 0 { 20252026 } else { season_row };
        let ss = parse_ss(ci_get(obj, "SeasonState"));
        let st = str_value(ci_get(obj, "StrengthState"));
        let st = if st.is_empty() { "Other".to_string() } else { st };
        if self.ss_norm != "all" && ss != self.ss_norm {
            return;
        }
        if self.st_norm != "all" && st != self.st_norm {
            return;
        }
        let pid = i64v(ci_get(obj, "PlayerID"));
        if pid <= 0 {
            return;
        }
        let la = self
            .league_acc
            .entry(season_row.to_string())
            .or_insert_with(|| json!({"SA": 0.0, "GA": 0.0}));
        let la = la.as_object_mut().expect("league acc object");
        la.insert(
            "SA".into(),
            json!(la.get("SA").and_then(Value::as_f64).unwrap_or(0.0) + flt(ci_get(obj, "SA"))),
        );
        la.insert(
            "GA".into(),
            json!(la.get("GA").and_then(Value::as_f64).unwrap_or(0.0) + flt(ci_get(obj, "GA"))),
        );
        // Only retain the requested goalie's per-season rows.
        if pid != self.target_pid {
            return;
        }
        let pmap = self
            .by_pid_season
            .entry(pid.to_string())
            .or_insert_with(|| json!({}));
        let pmap = pmap.as_object_mut().expect("pid map object");
        let d = pmap.entry(season_row.to_string()).or_insert_with(|| {
            json!({"TOI": 0.0, "FA": 0.0, "SA": 0.0, "GA": 0.0, "xGA_S": 0.0, "xGA_F": 0.0, "xGA_F2": 0.0})
        });
        let d = d.as_object_mut().expect("season row object");
        for key in ["TOI", "FA", "SA", "GA", "xGA_S", "xGA_F", "xGA_F2"] {
            let cur = d.get(key).and_then(Value::as_f64).unwrap_or(0.0);
            d.insert(key.to_string(), json!(cur + flt(ci_get(obj, key))));
        }
    }

    fn finish(self) -> (Value, Value) {
        let mut league_sa_ga: Map<String, Value> = Map::new();
        for (s, d) in &self.league_acc {
            let sa = d.get("SA").and_then(Value::as_f64).unwrap_or(0.0);
            let ga = d.get("GA").and_then(Value::as_f64).unwrap_or(0.0);
            league_sa_ga.insert(s.clone(), json!([sa, ga]));
        }
        (
            Value::Object(self.by_pid_season),
            Value::Object(league_sa_ga),
        )
    }
}

/// `_build_goalies_career_season_matrix`. Returns `(by_pid_season, league_sa_ga)`.
pub async fn build_goalies_career_matrix(
    caches: &Caches,
    sb: Option<&SbClient>,
    _cfg: &Config,
    season_state: &str,
    strength_state: &str,
    target_pid: i64,
) -> (Value, Value) {
    let ss_norm = normalize_ss(season_state);
    let st_norm = normalize_st(strength_state);
    let key = json!(["goalies_career", ss_norm, st_norm, target_pid]).to_string();
    let inflight_key = format!("cm:{key}");
    caches
        .inflight
        .run(&inflight_key, || async {
            if let Some(cached) = caches.career_matrix.get(&key) {
                if let Some(arr) = cached.as_array() {
                    if arr.len() == 2 {
                        return (arr[0].clone(), arr[1].clone());
                    }
                }
            }

            // Stream ONLY goalie rows (position = G) across all seasons from
            // Supabase; there is no CSV fallback — all data lives in Supabase.
            let mut state = CareerMatrixState::new(ss_norm, st_norm.clone(), target_pid);
            for_each_player_row(sb, &[], true, |page| {
                for r in page {
                    if let Some(obj) = r.as_object() {
                        state.ingest(obj);
                    }
                }
            })
            .await;

            let result = state.finish();
            caches
                .career_matrix
                .insert(key, json!([result.0.clone(), result.1.clone()]));
            result
        })
        .await
}

/// `_team_id_by_abbrev()`.
pub fn team_id_by_abbrev(teams: &[Value]) -> HashMap<String, i64> {
    let mut out = HashMap::new();
    for r in teams {
        let ab = str_value(r.get("Team")).to_uppercase();
        let tid = safe_int(r.get("TeamID")).unwrap_or(0);
        if !ab.is_empty() && tid > 0 {
            out.insert(ab, tid);
        }
    }
    out
}

/// `_team_stats_rest_get(url)` with a small TTL cache.
pub async fn team_stats_rest_get(caches: &Caches, http: &reqwest::Client, url: &str) -> Option<Value> {
    if let Some(v) = caches.team_stats_rest.get(&url.to_string()) {
        return Some(v);
    }
    let data = get_json_with_ua(http, url, 25).await.ok()?;
    if !data.is_object() {
        return None;
    }
    caches.team_stats_rest.insert(url.to_string(), data.clone());
    Some(data)
}

/// `_build_team_base_stats`.
#[allow(clippy::too_many_arguments)]
pub async fn build_team_base_stats(
    caches: &Caches,
    http: &reqwest::Client,
    sb: Option<&SbClient>,
    cfg: &Config,
    teams: &[Value],
    scope: &str,
    season_int: i64,
    season_ids: &[i64],
    season_state: &str,
    strength_state: &str,
) -> Value {
    let scope_norm = normalize_scope_total(scope);
    let ss_norm = normalize_ss(season_state);
    let st_norm = normalize_st(strength_state);
    let primary = primary_or(season_ids, season_int);
    let team_id_map = team_id_by_abbrev(teams);
    let abbrev_by_id: HashMap<i64, String> = team_id_map.iter().map(|(k, v)| (*v, k.clone())).collect();

    if scope_norm == "season" {
        // Derived team SeasonStats (Supabase → CSV).
        let rows = crate::data::teamseasonstats::iter_rows_filtered(caches, sb, cfg, season_ids, primary).await;
        let mut acc: Map<String, Value> = Map::new();
        let mut gp_max: HashMap<(String, String), i64> = HashMap::new();
        for r in &rows {
            let Some(obj) = r.as_object() else { continue };
            let team = str_value(ci_get(obj, "Team")).to_uppercase();
            if team.is_empty() {
                continue;
            }
            let ss_row = parse_ss(ci_get(obj, "SeasonState"));
            let st_row = str_value(ci_get(obj, "StrengthState"));
            let st_row = if st_row.is_empty() { "Other".to_string() } else { st_row };
            if ss_norm != "all" && ss_row != ss_norm {
                continue;
            }
            if st_norm != "all" && st_row != st_norm {
                continue;
            }
            let d = acc.entry(team.clone()).or_insert_with(|| {
                json!({"team": team, "teamId": team_id_map.get(&team).copied().unwrap_or(0),
                       "scope": "season", "season": primary, "seasonState": ss_norm,
                       "GP": 0, "TOI": 0.0, "CF": 0.0, "CA": 0.0, "FF": 0.0, "FA": 0.0,
                       "SF": 0.0, "SA": 0.0, "GF": 0.0, "GA": 0.0, "xGF": 0.0, "xGA": 0.0})
            });
            let gp_row = i64v(ci_get(obj, "GP"));
            let k = (team.clone(), ss_row);
            let prev = gp_max.get(&k).copied();
            if prev.is_none() || gp_row > prev.unwrap_or(0) {
                gp_max.insert(k, gp_row);
            }
            let d = d.as_object_mut().expect("team agg object");
            for key in ["TOI", "CF", "CA", "FF", "FA", "SF", "SA", "GF", "GA"] {
                let cur = d.get(key).and_then(Value::as_f64).unwrap_or(0.0);
                d.insert(key.to_string(), json!(cur + flt(ci_get(obj, key))));
            }
            // xG default = xG_F columns.
            let cur_xgf = d.get("xGF").and_then(Value::as_f64).unwrap_or(0.0);
            d.insert("xGF".into(), json!(cur_xgf + flt(ci_get(obj, "xGF_F"))));
            let cur_xga = d.get("xGA").and_then(Value::as_f64).unwrap_or(0.0);
            d.insert("xGA".into(), json!(cur_xga + flt(ci_get(obj, "xGA_F"))));
        }
        let mut gp_sum: HashMap<String, i64> = HashMap::new();
        for ((team, _), gp) in &gp_max {
            *gp_sum.entry(team.clone()).or_insert(0) += *gp;
        }
        for (team, d) in acc.iter_mut() {
            d.as_object_mut()
                .expect("team agg object")
                .insert("GP".into(), json!(gp_sum.get(team).copied().unwrap_or(0)));
        }
        if !acc.is_empty() {
            return Value::Object(acc);
        }
        // Fall through to NHL stats REST season aggregates.
    }

    let summary_url = |season_id: i64, game_type_id: i64| -> String {
        format!(
            "{API_STATS}/team/summary?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummary&cayenneExp=seasonId={season_id}%20and%20gameTypeId={game_type_id}"
        )
    };
    let shoot_url = |season_id: i64, game_type_id: i64| -> String {
        format!(
            "{API_STATS}/team/summaryshooting?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummaryshooting&cayenneExp=seasonId={season_id}%20and%20gameTypeId={game_type_id}"
        )
    };
    let summary_url_total = |team_id: i64| -> String {
        format!(
            "{API_STATS}/team/summary?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummary&cayenneExp=teamId={team_id}"
        )
    };
    let shoot_url_total = |team_id: i64| -> String {
        format!(
            "{API_STATS}/team/summaryshooting?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummaryshooting&cayenneExp=teamId={team_id}"
        )
    };

    if scope_norm == "season" {
        let gtypes: Vec<i64> = match ss_norm {
            "playoffs" => vec![3],
            "all" => vec![2, 3],
            _ => vec![2],
        };
        let mut summary_by_tid: Map<String, Value> = Map::new();
        let mut shoot_by_tid: Map<String, Value> = Map::new();
        for season_id in if season_ids.is_empty() { vec![primary] } else { season_ids.to_vec() } {
            for gt in &gtypes {
                let js = team_stats_rest_get(caches, http, &summary_url(season_id, *gt)).await.unwrap_or(Value::Null);
                let jj = team_stats_rest_get(caches, http, &shoot_url(season_id, *gt)).await.unwrap_or(Value::Null);
                for (tid, r) in rows_by_teamid(&js) {
                    let key = tid.to_string();
                    match summary_by_tid.get_mut(&key) {
                        None => {
                            summary_by_tid.insert(key, r);
                        }
                        Some(prev) => {
                            let p = prev.as_object_mut().expect("summary row object");
                            let add = |p: &mut Map<String, Value>, k: &str, v: &Value| {
                                let cur = p.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                                p.insert(k.to_string(), json!(cur + flt(Some(v))));
                            };
                            add(p, "gamesPlayed", r.get("gamesPlayed").unwrap_or(&json!(0)));
                            add(p, "goalsFor", r.get("goalsFor").unwrap_or(&json!(0)));
                            add(p, "goalsAgainst", r.get("goalsAgainst").unwrap_or(&json!(0)));
                            let sfp = flt(r.get("shotsForPerGame")) * flt(r.get("gamesPlayed"));
                            let cur = p.get("_shotsForTotal").and_then(Value::as_f64).unwrap_or(0.0);
                            p.insert("_shotsForTotal".into(), json!(cur + sfp));
                            let sap = flt(r.get("shotsAgainstPerGame")) * flt(r.get("gamesPlayed"));
                            let cur = p.get("_shotsAgainstTotal").and_then(Value::as_f64).unwrap_or(0.0);
                            p.insert("_shotsAgainstTotal".into(), json!(cur + sap));
                        }
                    }
                }
                for (tid, r) in rows_by_teamid(&jj) {
                    let key = tid.to_string();
                    match shoot_by_tid.get_mut(&key) {
                        None => {
                            shoot_by_tid.insert(key, r);
                        }
                        Some(prev) => {
                            let p = prev.as_object_mut().expect("shoot row object");
                            for k in ["gamesPlayed", "satFor", "satAgainst", "usatFor", "usatAgainst"] {
                                let cur = p.get(k).and_then(Value::as_f64).unwrap_or(0.0);
                                p.insert(k.to_string(), json!(cur + flt(r.get(k))));
                            }
                        }
                    }
                }
            }
        }
        let mut out: Map<String, Value> = Map::new();
        for (tid_s, rsum) in &summary_by_tid {
            let tid = tid_s.parse::<i64>().unwrap_or(0);
            let Some(ab) = abbrev_by_id.get(&tid) else { continue };
            let gp = flt(rsum.get("gamesPlayed"));
            let sf_total = if rsum.get("_shotsForTotal").is_some() {
                flt(rsum.get("_shotsForTotal"))
            } else {
                flt(rsum.get("shotsForPerGame")) * gp
            };
            let sa_total = if rsum.get("_shotsAgainstTotal").is_some() {
                flt(rsum.get("_shotsAgainstTotal"))
            } else {
                flt(rsum.get("shotsAgainstPerGame")) * gp
            };
            out.insert(
                ab.clone(),
                json!({"team": ab, "teamId": tid, "scope": "season", "season": primary,
                       "seasonState": ss_norm, "GP": gp, "GF": flt(rsum.get("goalsFor")),
                       "GA": flt(rsum.get("goalsAgainst")), "SF": sf_total, "SA": sa_total,
                       "xGF": null, "xGA": null}),
            );
        }
        for (tid_s, rsh) in &shoot_by_tid {
            let tid = tid_s.parse::<i64>().unwrap_or(0);
            let Some(ab) = abbrev_by_id.get(&tid) else { continue };
            if let Some(entry) = out.get_mut(ab) {
                let e = entry.as_object_mut().expect("team out object");
                e.insert("CF".into(), json!(flt(rsh.get("satFor"))));
                e.insert("CA".into(), json!(flt(rsh.get("satAgainst"))));
                e.insert("FF".into(), json!(flt(rsh.get("usatFor"))));
                e.insert("FA".into(), json!(flt(rsh.get("usatAgainst"))));
            }
        }
        return Value::Object(out);
    }

    // Total scope: per-team aggregate endpoints.
    let mut out2: Map<String, Value> = Map::new();
    for (ab, tid) in &team_id_map {
        let js = team_stats_rest_get(caches, http, &summary_url_total(*tid)).await;
        let jj = team_stats_rest_get(caches, http, &shoot_url_total(*tid)).await;
        let rs = js
            .and_then(|j| j.get("data").and_then(Value::as_array).cloned())
            .and_then(|arr| arr.into_iter().next())
            .unwrap_or(Value::Null);
        let rsh = jj
            .and_then(|j| j.get("data").and_then(Value::as_array).cloned())
            .and_then(|arr| arr.into_iter().next())
            .unwrap_or(Value::Null);
        let gp = flt(rs.get("gamesPlayed"));
        let sf_total = flt(rs.get("shotsForPerGame")) * gp;
        let sa_total = flt(rs.get("shotsAgainstPerGame")) * gp;
        out2.insert(
            ab.clone(),
            json!({"team": ab, "teamId": tid, "scope": "total", "season": null,
                   "seasonState": "all", "GP": gp, "GF": flt(rs.get("goalsFor")),
                   "GA": flt(rs.get("goalsAgainst")), "SF": sf_total, "SA": sa_total,
                   "CF": flt(rsh.get("satFor")), "CA": flt(rsh.get("satAgainst")),
                   "FF": flt(rsh.get("usatFor")), "FA": flt(rsh.get("usatAgainst")),
                   "xGF": null, "xGA": null}),
        );
    }
    Value::Object(out2)
}

fn normalize_scope_total(scope: &str) -> &'static str {
    match scope.trim().to_lowercase().as_str() {
        "total" => "total",
        _ => "season",
    }
}

fn rows_by_teamid(j: &Value) -> Vec<(i64, Value)> {
    let mut out = Vec::new();
    if let Some(rows) = j.get("data").and_then(Value::as_array) {
        for r in rows {
            let tid = safe_int(r.get("teamId")).unwrap_or(0);
            if tid > 0 {
                out.push((tid, r.clone()));
            }
        }
    }
    out
}
