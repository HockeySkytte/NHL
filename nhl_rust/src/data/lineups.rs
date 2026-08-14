//! Lineups loader — port of `_load_lineups_all()` + `_merge_gp_est_from_json()`.
//! Supabase `lineups` table first, `app/static/lineups_all.json` fallback.

use std::path::Path;

use serde_json::{json, Map, Value};

use crate::state::Caches;
use crate::supabase::read::SbClient;
use crate::util::parse::{ci_get, safe_int, str_value};

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
/// appends EXT/scratch players missing from Supabase.
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
