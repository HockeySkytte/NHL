//! Player projections map (port of `_load_player_projections_from_sheets` /
//! `_load_player_projections_csv` and `_parse_proj_row`).

use std::collections::{BTreeMap, HashMap};

use serde_json::{json, Value};

use crate::state::Caches;
use crate::config::Config;
use crate::supabase::read::SbClient;
use crate::util::parse::{parse_locale_float, safe_int};

fn col_map() -> HashMap<String, String> {
    [
        ("player_id", "PlayerID"),
        ("position", "Position"),
        ("game_no", "Game_No"),
        ("age", "Age"),
        ("rookie", "Rookie"),
        ("evo", "EVO"),
        ("evd", "EVD"),
        ("pp", "PP"),
        ("sh", "SH"),
        ("gsax", "GSAx"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

fn map_to_value(map: &BTreeMap<i64, Value>) -> Value {
    let mut out = serde_json::Map::new();
    for (k, v) in map {
        out.insert(k.to_string(), v.clone());
    }
    Value::Object(out)
}

fn value_to_map(v: Value) -> BTreeMap<i64, Value> {
    let mut out = BTreeMap::new();
    if let Value::Object(map) = v {
        for (k, row) in map {
            if let Ok(pid) = k.parse::<i64>() {
                out.insert(pid, row);
            }
        }
    }
    out
}

/// Supabase-first, CSV fallback. Keyed by int playerId.
pub async fn load_map(caches: &Caches, sb: Option<&SbClient>, cfg: &Config) -> BTreeMap<i64, Value> {
    if let Some(v) = caches.player_projections.get(&()) {
        return value_to_map(v);
    }
    let out = load_map_inner(sb, cfg).await;
    caches.player_projections.insert((), map_to_value(&out));
    out
}

async fn load_map_inner(sb: Option<&SbClient>, cfg: &Config) -> BTreeMap<i64, Value> {
    if let Some(sb) = sb {
        if let Some(rows) = sb.read("player_projections", "*", None, Some(&col_map()), None, 0).await {
            let mut out = BTreeMap::new();
            for row in rows {
                let pid = safe_int(
                    row.get("PlayerID")
                        .or_else(|| row.get("playerId"))
                        .or_else(|| row.get("player_id")),
                );
                if let Some(pid) = pid {
                    if pid > 0 {
                        out.insert(pid, row);
                    }
                }
            }
            return out;
        }
    }
    // CSV fallback (flexible id column, like Python).
    load_csv(&cfg.static_dir.join("player_projections.csv"))
}

fn load_csv(path: &std::path::Path) -> BTreeMap<i64, Value> {
    let mut out = BTreeMap::new();
    if !path.exists() {
        return out;
    }
    let Ok(mut reader) = csv::ReaderBuilder::new().has_headers(true).from_path(path) else {
        return out;
    };
    let headers: Vec<String> = reader
        .headers()
        .map(|h| h.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let lower_map: HashMap<String, String> = headers
        .iter()
        .map(|h| (h.to_lowercase(), h.clone()))
        .collect();
    let id_col = ["playerid", "player_id", "id"]
        .iter()
        .find_map(|c| lower_map.get(*c))
        .cloned()
        .or_else(|| {
            if headers.iter().any(|h| h == "playerId") {
                Some("playerId".to_string())
            } else {
                None
            }
        });
    for record in reader.records() {
        let Ok(record) = record else { continue };
        let row: HashMap<String, Value> = headers
            .iter()
            .enumerate()
            .map(|(i, h)| (h.clone(), Value::from(record.get(i).unwrap_or(""))))
            .collect();
        let pid_raw = id_col.as_ref().and_then(|c| row.get(c));
        let Some(pid) = pid_raw.and_then(|v| safe_int(Some(v))) else {
            continue;
        };
        out.insert(pid, serde_json::to_value(&row).unwrap_or(Value::Null));
    }
    out
}

/// `_parse_proj_row`: player_projections.csv column semantics; total excludes GSAx.
pub fn parse_proj_row(row: &Value) -> Value {
    let pid = safe_int(
        row.get("PlayerID")
            .or_else(|| row.get("playerId"))
            .or_else(|| row.get("player_id"))
            .or_else(|| row.get("id")),
    );
    let pos_raw = row
        .get("Position")
        .or_else(|| row.get("position"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .trim()
        .to_uppercase();
    let pos = pos_raw.chars().next().map(|c| c.to_string()).unwrap_or_default();
    let gp = safe_int(
        row.get("Game_No")
            .or_else(|| row.get("GP"))
            .or_else(|| row.get("games"))
            .or_else(|| row.get("gamesPlayed")),
    );
    let age = parse_locale_float(row.get("Age"));
    let rookie = parse_locale_float(row.get("Rookie"));
    let evo = parse_locale_float(row.get("EVO"));
    let evd = parse_locale_float(row.get("EVD"));
    let pp = parse_locale_float(row.get("PP"));
    let sh = parse_locale_float(row.get("SH"));
    let gsax = parse_locale_float(
        row.get("GSAx")
            .or_else(|| row.get("gsax"))
            .or_else(|| row.get("Gsax"))
            .or_else(|| row.get("GsaX")),
    );
    let total = age.unwrap_or(0.0)
        + rookie.unwrap_or(0.0)
        + evo.unwrap_or(0.0)
        + evd.unwrap_or(0.0)
        + pp.unwrap_or(0.0)
        + sh.unwrap_or(0.0);
    json!({
        "playerId": pid,
        "position": pos,
        "gp": gp,
        "Age": age,
        "Rookie": rookie,
        "EVO": evo,
        "EVD": evd,
        "PP": pp,
        "SH": sh,
        "GSAx": gsax,
        "total": total,
    })
}
