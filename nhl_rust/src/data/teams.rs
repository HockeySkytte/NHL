//! Team rows — port of `_load_teams_csv()` / `TEAM_ROWS`.
//!
//! Supabase `teams` table ONLY (no CSV fallback). Loaded once at startup.

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

pub async fn load(sb: Option<&SbClient>, _cfg: &Config) -> Vec<Value> {
    let Some(sb) = sb else {
        return Vec::new();
    };
    sb.read("teams", "*", None, Some(&col_map()), None, 0)
        .await
        .map(|rows| rows.into_iter().map(normalize_active).collect())
        .unwrap_or_default()
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
