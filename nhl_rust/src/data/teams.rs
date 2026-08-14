//! Team rows — port of `_load_teams_csv()` / `TEAM_ROWS`.
//!
//! Supabase `teams` table first, `Teams.csv` fallback. Loaded once at startup
//! (the Flask module loads it once at import).

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::config::Config;
use crate::supabase::read::SbClient;

fn col_map() -> HashMap<String, String> {
    [
        ("team", "Team"),
        ("team_id", "TeamID"),
        ("name", "Name"),
        ("logo", "Logo"),
        ("color", "Color"),
        ("active", "Active"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Normalise the `Active` column to `'0'`/`'1'` strings (Supabase may return
/// booleans or numbers; the CSV has strings). Port of `_load_teams_csv`.
fn normalize_active(mut row: Value) -> Value {
    if let Some(obj) = row.as_object_mut() {
        if let Some(active) = obj.get("Active") {
            let normalized = match active {
                Value::Bool(b) => json!(if *b { "1" } else { "0" }),
                Value::Number(n) => json!(n.to_string()),
                _ => return row,
            };
            obj.insert("Active".to_string(), normalized);
        }
    }
    row
}

pub async fn load(sb: Option<&SbClient>, cfg: &Config) -> Vec<Value> {
    if let Some(sb) = sb {
        if let Some(rows) = sb.read("teams", "*", None, Some(&col_map()), None, 0).await {
            return rows.into_iter().map(normalize_active).collect();
        }
    }
    load_csv(&cfg.teams_csv_path)
}

fn load_csv(path: &std::path::Path) -> Vec<Value> {
    if !path.exists() {
        tracing::warn!("Teams.csv not found at {} — serving no teams", path.display());
        return Vec::new();
    }
    let Ok(mut reader) = csv::ReaderBuilder::new().has_headers(true).from_path(path) else {
        tracing::warn!("failed to open Teams.csv at {}", path.display());
        return Vec::new();
    };
    let headers: Vec<String> = match reader.headers() {
        Ok(h) => h
            .iter()
            .map(|s| s.trim_start_matches('\u{feff}').to_string())
            .collect(),
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for record in reader.records() {
        let Ok(record) = record else { continue };
        let mut map = serde_json::Map::new();
        for (i, value) in record.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                map.insert(header.clone(), json!(value));
            }
        }
        out.push(Value::Object(map));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_active_values() {
        let row = json!({"Team": "BOS", "Active": true});
        let row = normalize_active(row);
        assert_eq!(row["Active"], "1");

        let row = json!({"Team": "WPG", "Active": 1});
        let row = normalize_active(row);
        assert_eq!(row["Active"], "1");

        let row = json!({"Team": "CAR", "Active": "0"});
        let row = normalize_active(row);
        assert_eq!(row["Active"], "0");
    }
}
