//! Supabase `last_dates` table → { season: last date } (port of Flask
//! `_load_last_dates`, which reads the same Supabase table — no CSV).

use std::collections::BTreeMap;

use serde_json::Value;

use crate::config::Config;
use crate::supabase::read::SbClient;

fn col_map() -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    m.insert("season".to_string(), "Season".to_string());
    m.insert("last_date".to_string(), "Last_Date".to_string());
    m
}

pub async fn load(sb: Option<&SbClient>, _cfg: &Config) -> BTreeMap<i64, String> {
    let mut out = BTreeMap::new();
    let Some(sb) = sb else {
        return out;
    };
    if let Some(rows) = sb.read("last_dates", "*", None, Some(&col_map()), None, 0).await {
        for row in rows {
            let season = row
                .get("Season")
                .and_then(Value::as_i64)
                .or_else(|| {
                    row.get("Season")
                        .and_then(Value::as_str)
                        .and_then(|s| s.trim().parse::<i64>().ok())
                });
            let date = row
                .get("Last_Date")
                .and_then(Value::as_str)
                .map(|s| s.trim().to_string())
                .unwrap_or_default();
            if let Some(s) = season {
                if !date.is_empty() {
                    out.insert(s, date);
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_last_dates_shape() {
        // Supabase `last_dates` rows (season number + last_date string) map to
        // `Season`/`Last_Date` after the col_map rename.
        let row = serde_json::json!({"Season": 20242025, "Last_Date": "2025-06-30"});
        assert_eq!(row["Season"].as_i64(), Some(20242025));
        assert_eq!(row["Last_Date"].as_str(), Some("2025-06-30"));
        let cm = col_map();
        assert_eq!(cm.get("season").map(String::as_str), Some("Season"));
        assert_eq!(cm.get("last_date").map(String::as_str), Some("Last_Date"));
    }
}
