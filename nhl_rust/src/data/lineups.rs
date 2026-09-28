//! Lineups loader — port of `_load_lineups_all()` + `_merge_gp_est_from_json()`.
//! Supabase `lineups` table first, `app/static/lineups_all.json` fallback.

use std::path::Path;

use chrono::{DateTime, Utc};
use serde_json::{json, Map, Value};

use crate::state::Caches;
use crate::supabase::read::SbClient;
use crate::util::parse::{ci_get, safe_int, str_value};

/// Mirrors Python `_JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS`. A static
/// `lineups_all.json` is a *snapshot*; Supabase is the live source, so a snapshot
/// must never re-introduce players Supabase has already dropped (e.g. someone
/// traded away after the snapshot was committed and shipped in the deploy image).
/// `sync_lineups_to_supabase.py` PATCHes every row and the `updated_at` trigger
/// stamps that *after* the snapshot was generated, so a snapshot from the current
/// pipeline run is legitimately a little older than the Supabase rows. This window
/// absorbs the scrape -> estimate -> sync cycle.
const JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS: i64 = 6 * 3600;

/// Parse an ISO-8601 timestamp ('Z', explicit offset, or naive) as UTC.
fn parse_iso_utc(value: Option<&Value>) -> Option<DateTime<Utc>> {
    let s = str_value(value);
    if s.is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(&s) {
        return Some(dt.with_timezone(&Utc));
    }
    // Naive timestamp (no offset) - assume UTC, matching `datetime.replace(tzinfo=utc)`.
    for fmt in ["%Y-%m-%dT%H:%M:%S%.f", "%Y-%m-%dT%H:%M:%S"] {
        if let Ok(dt) = chrono::NaiveDateTime::parse_from_str(&s, fmt) {
            return Some(dt.and_utc());
        }
    }
    None
}

/// True when the static snapshot is current enough to complete a team's pool.
/// Unknown/unparseable timestamps keep the historical behaviour, so a missing
/// timestamp can never silently empty a team's scratch pool.
fn snapshot_may_add_players(json_generated_at: Option<&Value>, supabase_generated_at: Option<&Value>) -> bool {
    match (parse_iso_utc(json_generated_at), parse_iso_utc(supabase_generated_at)) {
        (Some(j), Some(s)) => (s - j).num_seconds() <= JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS,
        _ => true,
    }
}

fn lineups_json_path(static_dir: &Path) -> std::path::PathBuf {
    static_dir.join("lineups_all.json")
}

/// Loads lineups; result is cached in `caches.lineups_all`.
pub async fn load_all(caches: &Caches, sb: Option<&SbClient>, static_dir: &Path) -> Value {
    if let Some(v) = caches.lineups_all.get(&()) {
        return v;
    }
    let out = load_all_inner(sb, static_dir).await;
    caches.lineups_all.insert((), out.clone());
    out
}

async fn load_all_inner(sb: Option<&SbClient>, static_dir: &Path) -> Value {
    // Supabase first.
    if let Some(sb) = sb {
        if let Some(rows) = sb.read("lineups", "*", None, None, None, 0).await {
            if !rows.is_empty() {
                return build_from_supabase_rows(rows, static_dir);
            }
        }
    }
    // Fallback: static JSON verbatim.
    let json_path = lineups_json_path(static_dir);
    if let Ok(raw) = std::fs::read_to_string(&json_path) {
        if let Ok(fallback) = serde_json::from_str::<Value>(&raw) {
            if fallback.is_object() && !fallback.as_object().map(|o| o.is_empty()).unwrap_or(true) {
                return fallback;
            }
        }
    }
    Value::Object(Map::new())
}

/// Normalizes Supabase rows into the same internal shape Flask builds, then
/// buckets/dedupes and merges gp_est from the static JSON.
fn build_from_supabase_rows(sb_raw: Vec<Value>, static_dir: &Path) -> Value {
    let mut rows: Vec<Map<String, Value>> = Vec::with_capacity(sb_raw.len());
    for r in sb_raw {
        let Some(obj) = r.as_object() else { continue };
        rows.push(normalize_supabase_row(obj));
    }
    // Sort by Timestamp descending (latest wins the dedupe).
    rows.sort_by(|a, b| {
        let ta = str_value(a.get("Timestamp")).to_string();
        let tb = str_value(b.get("Timestamp")).to_string();
        tb.cmp(&ta)
    });

    let mut out: Map<String, Value> = Map::new();
    let mut injuries_by_team: std::collections::HashMap<String, Vec<Value>> = Default::default();
    let mut seen: std::collections::HashSet<(String, i64)> = std::collections::HashSet::new();
    let mut latest_ts_by_team: std::collections::HashMap<String, String> = Default::default();

    for r in &rows {
        let team = str_value(r.get("Team")).to_uppercase();
        if team.is_empty() {
            continue;
        }
        let unit = str_value(r.get("Unit")).to_uppercase();
        let pos_raw = str_value(r.get("Pos")).to_uppercase();
        let pos_first = pos_raw.chars().next().map(|c| c.to_string()).unwrap_or_default();
        let name = str_value(r.get("PlayerName"));
        let Some(pid) = safe_int(r.get("playerId")) else { continue };
        let ts = str_value(r.get("Timestamp"));

        let key = (team.clone(), pid);
        if seen.contains(&key) {
            continue;
        }
        seen.insert(key);

        if !ts.is_empty() {
            let cur = latest_ts_by_team.get(&team).cloned().unwrap_or_default();
            if ts > cur {
                latest_ts_by_team.insert(team.clone(), ts);
            }
        }

        let mut rec = Map::new();
        rec.insert("name".into(), Value::String(name));
        rec.insert("playerId".into(), json!(pid));
        rec.insert("unit".into(), Value::String(unit.clone()));
        let pos = if unit.starts_with('G') { "G" } else { pos_first.as_str() };
        rec.insert("pos".into(), Value::String(pos.to_string()));
        if let Some(gp_est) = r.get("gp_est") {
            if let Some(v) = safe_int(Some(gp_est)) {
                rec.insert("gp_est".into(), json!(v));
            }
        }
        let gp_note = str_value(r.get("gp_est_note"));
        if !gp_note.is_empty() {
            rec.insert("gp_est_note".into(), Value::String(gp_note));
        }

        let bucket = if pos == "G" || unit.starts_with('G') {
            rec.insert("pos".into(), Value::String("G".to_string()));
            "goalies"
        } else if pos == "D" || unit.starts_with("LD") || unit.starts_with("RD") {
            rec.insert("pos".into(), Value::String("D".to_string()));
            "defense"
        } else {
            rec.insert("pos".into(), Value::String("F".to_string()));
            "forwards"
        };

        let is_injured = safe_int(r.get("is_injured")).unwrap_or(0) == 1;
        if is_injured {
            let replacement = safe_int(r.get("replacement_id")).unwrap_or(0);
            injuries_by_team
                .entry(team.clone())
                .or_default()
                .push(json!({
                    "injuredPid": pid,
                    "replacementPid": replacement,
                    "startDate": str_value(r.get("injury_start")),
                    "endDate": str_value(r.get("injury_end")),
                }));
        }

        let node = out.entry(team.clone()).or_insert_with(|| {
            json!({"team": team, "forwards": [], "defense": [], "goalies": [], "generated_at": null})
        });
        let node_obj = node.as_object_mut().expect("team node object");
        node_obj
            .get_mut(bucket)
            .and_then(Value::as_array_mut)
            .expect("bucket array")
            .push(Value::Object(rec));
    }

    for (team, node) in out.iter_mut() {
        let obj = node.as_object_mut().expect("node object");
        obj.insert(
            "generated_at".into(),
            latest_ts_by_team
                .get(team)
                .map(|v| Value::String(v.clone()))
                .unwrap_or(Value::Null),
        );
        if let Some(inj) = injuries_by_team.get(team) {
            if !inj.is_empty() {
                obj.insert("injuries".into(), Value::Array(inj.clone()));
            }
        }
    }

    merge_gp_est_from_json(&mut out, static_dir);
    Value::Object(out)
}

fn normalize_supabase_row(r: &Map<String, Value>) -> Map<String, Value> {
    let mut out = Map::new();
    let pick = |keys: &[&str]| -> Option<Value> {
        for k in keys {
            if let Some(v) = ci_get(r, k) {
                if !v.is_null() && str_value(Some(v)) != "" {
                    return Some(v.clone());
                }
            }
        }
        None
    };
    out.insert("Team".into(), pick(&["team", "Team"]).unwrap_or(Value::String(String::new())));
    out.insert("Unit".into(), pick(&["line_unit", "unit", "Unit"]).unwrap_or(Value::String(String::new())));
    out.insert("Pos".into(), pick(&["position", "pos", "Pos"]).unwrap_or(Value::String(String::new())));
    out.insert("PlayerName".into(), pick(&["player_name", "player", "name", "PlayerName"]).unwrap_or(Value::String(String::new())));
    out.insert("playerId".into(), pick(&["player_id", "playerId", "PlayerID"]).unwrap_or(Value::Null));
    out.insert(
        "Timestamp".into(),
        Value::String(str_value(pick(&["updated_at", "timestamp", "created_at", "Timestamp"]).as_ref())),
    );
    if let Some(v) = pick(&["estimated_gp", "gp_est"]) {
        out.insert("gp_est".into(), v);
    }
    if let Some(v) = pick(&["gp_note", "gp_est_note"]) {
        out.insert("gp_est_note".into(), v);
    }
    for key in ["starter", "is_injured", "injury_start", "injury_end", "replacement_id"] {
        if let Some(v) = r.get(key).cloned() {
            if !v.is_null() {
                out.insert(key.into(), v);
            }
        }
    }
    out
}

/// Port of `_merge_gp_est_from_json`: merges gp_est into existing players and
/// appends EXT/scratch players missing from Supabase - the append only happens
/// while the snapshot is current (see `snapshot_may_add_players`).
fn merge_gp_est_from_json(out: &mut Map<String, Value>, static_dir: &Path) {
    let json_path = lineups_json_path(static_dir);
    let raw = match std::fs::read_to_string(&json_path) {
        Ok(raw) => raw,
        Err(_) => return,
    };
    let json_data: Value = match serde_json::from_str(&raw) {
        Ok(v) => v,
        Err(_) => return,
    };
    let Some(json_map) = json_data.as_object() else { return };

    for (team_abbrev, team_node) in json_map {
        let Some(team_node) = team_node.as_object() else { continue };
        let Some(out_team) = out.get_mut(team_abbrev).and_then(Value::as_object_mut) else {
            continue;
        };
        // A stale snapshot may still supply gp_est, but must not add players.
        let snapshot_may_add = snapshot_may_add_players(
            team_node.get("generated_at"),
            out_team.get("generated_at"),
        );
        for group_key in ["forwards", "defense", "goalies"] {
            let json_players: Vec<Value> = team_node
                .get(group_key)
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let out_players = match out_team.get_mut(group_key).and_then(Value::as_array_mut) {
                Some(p) => p,
                None => continue,
            };
            // JSON lookup by pid.
            let mut json_by_pid: std::collections::HashMap<i64, &Value> = Default::default();
            for p in &json_players {
                if let Some(pid) = p.get("playerId").and_then(|v| safe_int(Some(v))) {
                    json_by_pid.insert(pid, p);
                }
            }
            let mut out_pids: std::collections::HashSet<i64> = Default::default();
            for op in out_players.iter() {
                if let Some(pid) = op.get("playerId").and_then(|v| safe_int(Some(v))) {
                    out_pids.insert(pid);
                }
            }
            // Merge gp_est into existing players.
            for op in out_players.iter_mut() {
                let pid = op.get("playerId").and_then(|v| safe_int(Some(v)));
                if let Some(pid) = pid {
                    if let Some(jp) = json_by_pid.get(&pid) {
                        if op.get("gp_est").is_none() {
                            if let Some(v) = jp.get("gp_est") {
                                op.as_object_mut().unwrap().insert("gp_est".into(), v.clone());
                            }
                        }
                        if op.get("gp_est_note").is_none() {
                            if let Some(v) = jp.get("gp_est_note") {
                                op.as_object_mut().unwrap().insert("gp_est_note".into(), v.clone());
                            }
                        }
                    }
                }
            }
            if !snapshot_may_add {
                continue;
            }
            // Append JSON players missing from Supabase.
            let default_pos = if group_key == "forwards" {
                "F"
            } else if group_key == "defense" {
                "D"
            } else {
                "G"
            };
            for jp in &json_players {
                let Some(pid) = jp.get("playerId").and_then(|v| safe_int(Some(v))) else {
                    continue;
                };
                let name = jp.get("name").and_then(Value::as_str).unwrap_or("");
                if out_pids.contains(&pid) || name.is_empty() {
                    continue;
                }
                let mut extra = Map::new();
                extra.insert("name".into(), Value::String(name.to_string()));
                extra.insert("playerId".into(), json!(pid));
                extra.insert(
                    "unit".into(),
                    Value::String(jp.get("unit").and_then(Value::as_str).unwrap_or("EXT").to_string()),
                );
                extra.insert(
                    "pos".into(),
                    Value::String(
                        jp.get("pos")
                            .and_then(Value::as_str)
                            .unwrap_or(default_pos)
                            .to_string(),
                    ),
                );
                if let Some(v) = jp.get("gp_est") {
                    extra.insert("gp_est".into(), v.clone());
                }
                if let Some(v) = jp.get("gp_est_note") {
                    extra.insert("gp_est_note".into(), v.clone());
                }
                out_players.push(Value::Object(extra));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Snapshot from the 2026-08-14 deploy vs the Supabase rows written on
    // 2026-09-28 - the stale bundle that produced duplicate starter slots.
    const STALE_JSON_GENERATED_AT: &str = "2026-08-14T01:02:26.942467+00:00";
    const SUPABASE_GENERATED_AT: &str = "2026-09-28T19:04:40.711208+00:00";
    // Same pipeline run: the snapshot is generated first, then synced to Supabase.
    const PIPELINE_JSON_GENERATED_AT: &str = "2026-09-28T19:02:26.000000+00:00";

    const CURRENT_LW_PID: i64 = 111;
    const TRADED_AWAY_PID: i64 = 999;
    const EXTRA_D_PID: i64 = 222;

    fn temp_static_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("nhl-lineups-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");
        dir
    }

    fn write_snapshot(dir: &Path, generated_at: Value) -> PathBuf {
        let snapshot = json!({
            "ANA": {
                "team": "ANA",
                "generated_at": generated_at,
                "forwards": [
                    {"name": "Current LW", "playerId": CURRENT_LW_PID, "unit": "LW1", "pos": "F",
                     "gp_est": 75, "gp_est_note": "wtd-avg last 3"},
                    {"name": "Traded Away", "playerId": TRADED_AWAY_PID, "unit": "LW1", "pos": "F",
                     "gp_est": 70, "gp_est_note": "stale-note"}
                ],
                "defense": [
                    {"name": "Extra D", "playerId": EXTRA_D_PID, "unit": "EXT", "pos": "D"}
                ],
                "goalies": []
            }
        });
        let path = dir.join("lineups_all.json");
        std::fs::write(&path, snapshot.to_string()).expect("write snapshot");
        path
    }

    fn supabase_side(generated_at: Value) -> Map<String, Value> {
        let mut out = Map::new();
        out.insert(
            "ANA".into(),
            json!({
                "team": "ANA",
                "generated_at": generated_at,
                "forwards": [{"name": "Current LW", "playerId": CURRENT_LW_PID, "unit": "LW1", "pos": "F"}],
                "defense": [],
                "goalies": []
            }),
        );
        out
    }

    fn group_pids(out: &Map<String, Value>, group: &str) -> Vec<i64> {
        out["ANA"][group]
            .as_array()
            .map(|a| a.iter().filter_map(|p| p.get("playerId").and_then(Value::as_i64)).collect())
            .unwrap_or_default()
    }

    // ── the regression ───────────────────────────────────────────────────────

    #[test]
    fn stale_snapshot_does_not_add_players() {
        let dir = temp_static_dir("stale");
        write_snapshot(&dir, json!(STALE_JSON_GENERATED_AT));
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        assert_eq!(group_pids(&out, "forwards"), vec![CURRENT_LW_PID]);
        assert!(group_pids(&out, "defense").is_empty());
    }

    #[test]
    fn stale_snapshot_does_not_duplicate_a_starter_slot() {
        let dir = temp_static_dir("stale-dup");
        write_snapshot(&dir, json!(STALE_JSON_GENERATED_AT));
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        let lw1: Vec<i64> = out["ANA"]["forwards"]
            .as_array()
            .expect("forwards array")
            .iter()
            .filter(|p| p.get("unit").and_then(Value::as_str) == Some("LW1"))
            .filter_map(|p| p.get("playerId").and_then(Value::as_i64))
            .collect();
        assert_eq!(lw1, vec![CURRENT_LW_PID]);
    }

    #[test]
    fn stale_snapshot_still_fills_missing_gp_est() {
        let dir = temp_static_dir("stale-gp");
        write_snapshot(&dir, json!(STALE_JSON_GENERATED_AT));
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        let rec = &out["ANA"]["forwards"][0];
        assert_eq!(rec.get("gp_est").and_then(Value::as_i64), Some(75));
        assert_eq!(rec.get("gp_est_note").and_then(Value::as_str), Some("wtd-avg last 3"));
    }

    // ── the behaviour that must keep working ─────────────────────────────────

    #[test]
    fn fresh_snapshot_completes_the_pool() {
        let dir = temp_static_dir("fresh");
        write_snapshot(&dir, json!("2026-10-01T12:00:00+00:00"));
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        assert_eq!(group_pids(&out, "forwards"), vec![CURRENT_LW_PID, TRADED_AWAY_PID]);
        assert_eq!(group_pids(&out, "defense"), vec![EXTRA_D_PID]);
    }

    #[test]
    fn snapshot_from_the_current_pipeline_run_still_completes_the_pool() {
        let dir = temp_static_dir("pipeline");
        write_snapshot(&dir, json!(PIPELINE_JSON_GENERATED_AT));
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        assert!(group_pids(&out, "forwards").contains(&TRADED_AWAY_PID));
        assert_eq!(group_pids(&out, "defense"), vec![EXTRA_D_PID]);
    }

    #[test]
    fn unknown_timestamps_keep_legacy_behaviour() {
        let dir = temp_static_dir("unknown");
        write_snapshot(&dir, Value::Null);
        let mut out = supabase_side(Value::Null);

        merge_gp_est_from_json(&mut out, &dir);

        assert!(group_pids(&out, "forwards").contains(&TRADED_AWAY_PID));
    }

    #[test]
    fn missing_snapshot_is_a_noop() {
        let dir = temp_static_dir("absent");
        let mut out = supabase_side(json!(SUPABASE_GENERATED_AT));

        merge_gp_est_from_json(&mut out, &dir);

        assert_eq!(group_pids(&out, "forwards"), vec![CURRENT_LW_PID]);
    }

    // ── timestamp helpers ────────────────────────────────────────────────────

    #[test]
    fn parse_iso_utc_accepts_offset_z_and_naive() {
        for s in [
            "2026-09-28T19:04:40.711208+00:00",
            "2026-09-28T19:04:40Z",
            "2026-09-28T19:04:40",
        ] {
            let dt = parse_iso_utc(Some(&json!(s))).unwrap_or_else(|| panic!("failed to parse {s}"));
            assert_eq!(dt.format("%Y-%m-%d").to_string(), "2026-09-28");
        }
        assert!(parse_iso_utc(None).is_none());
        assert!(parse_iso_utc(Some(&json!("nope"))).is_none());
    }

    #[test]
    fn tolerance_boundary() {
        let sb = json!(SUPABASE_GENERATED_AT);
        assert!(snapshot_may_add_players(Some(&sb), Some(&sb)));

        let base = DateTime::parse_from_rfc3339(SUPABASE_GENERATED_AT).expect("rfc3339");
        let inside = (base - chrono::Duration::seconds(JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS)).to_rfc3339();
        assert!(snapshot_may_add_players(Some(&json!(inside)), Some(&sb)));

        let outside =
            (base - chrono::Duration::seconds(JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS + 1)).to_rfc3339();
        assert!(!snapshot_may_add_players(Some(&json!(outside)), Some(&sb)));
    }
}
