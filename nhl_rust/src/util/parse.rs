//! Lenient value-parsing helpers ported from `app/routes.py`
//! (`_safe_int`, `_parse_locale_float`, `_safe_float`, season parsing).

use serde_json::{Map, Value};

/// `_safe_int`: accepts int/float/str, returns None for empty/invalid.
pub fn safe_int(v: Option<&Value>) -> Option<i64> {
    match v? {
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(i)
            } else if let Some(u) = n.as_u64() {
                Some(u.min(i64::MAX as u64) as i64)
            } else {
                n.as_f64().map(|f| f as i64)
            }
        }
        Value::String(s) => {
            let t = s.trim();
            t.parse::<i64>()
                .ok()
                .or_else(|| t.parse::<f64>().ok().map(|f| f as i64))
        }
        _ => None,
    }
}

/// `_parse_locale_float`: comma-decimal aware, None for empty/invalid.
pub fn parse_locale_float(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n
            .as_f64()
            .or_else(|| n.as_i64().map(|i| i as f64)),
        Value::String(s) => {
            let raw = s.trim();
            if raw.is_empty() {
                return None;
            }
            raw.replace(',', ".").parse::<f64>().ok()
        }
        _ => None,
    }
}

/// `_safe_float`: like `_parse_locale_float`.
pub fn safe_float(v: Option<&Value>) -> Option<f64> {
    parse_locale_float(v)
}

/// Stringify a value like `str(x or '')` (trimmed).
pub fn str_value(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        Some(Value::Bool(b)) => b.to_string(),
        _ => String::new(),
    }
}

/// Case-insensitive key lookup on a JSON object (Supabase columns may vary in
/// casing across tables/backfills).
pub fn ci_get<'a>(obj: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    if let Some(v) = obj.get(key) {
        return Some(v);
    }
    let lower = key.to_lowercase();
    obj.iter()
        .find(|(k, _)| k.to_lowercase() == lower)
        .map(|(_, v)| v)
}

/// Parses a `season` query param (comma-separated, like `_parse_request_season_ids`).
/// Empty/invalid falls back to the current season.
pub fn parse_season_ids(param: Option<&str>, current: i64) -> Vec<i64> {
    let raw = param.unwrap_or("").trim();
    if raw.is_empty() {
        return vec![current];
    }
    let mut out: Vec<i64> = raw
        .split(',')
        .filter_map(|s| s.trim().parse::<i64>().ok())
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        vec![current]
    } else {
        out
    }
}

/// `_primary_season_id`: the newest season id in the list.
pub fn primary_season_id(ids: &[i64], current: i64) -> i64 {
    ids.iter().copied().max().unwrap_or(current)
}

/// Flask-style truthy query param (`1|true|yes|y`).
pub fn flag_param(map: &Map<String, Value>, key: &str, fallback: &str) -> bool {
    let raw = ci_get(map, key)
        .map(|v| str_value(Some(v)).to_lowercase())
        .unwrap_or_default();
    if raw.is_empty() {
        let f = fallback.to_lowercase();
        return matches!(f.as_str(), "1" | "true" | "yes" | "y" | "force");
    }
    matches!(raw.as_str(), "1" | "true" | "yes" | "y" | "force")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn safe_int_variants() {
        assert_eq!(safe_int(Some(&json!(8471214))), Some(8471214));
        assert_eq!(safe_int(Some(&json!(8471214.0))), Some(8471214));
        assert_eq!(safe_int(Some(&json!("8471214"))), Some(8471214));
        assert_eq!(safe_int(Some(&json!("  "))), None);
        assert_eq!(safe_int(None), None);
    }

    #[test]
    fn locale_float() {
        assert_eq!(parse_locale_float(Some(&json!("1,75"))), Some(1.75));
        assert_eq!(parse_locale_float(Some(&json!("2.5"))), Some(2.5));
        assert_eq!(parse_locale_float(Some(&json!(""))), None);
        assert_eq!(parse_locale_float(Some(&json!(3))), Some(3.0));
    }

    #[test]
    fn season_ids() {
        assert_eq!(parse_season_ids(Some("20232024,20252026"), 20252026), vec![20232024, 20252026]);
        assert_eq!(parse_season_ids(None, 20252026), vec![20252026]);
        assert_eq!(parse_season_ids(Some("garbage"), 20252026), vec![20252026]);
        assert_eq!(primary_season_id(&[20232024, 20252026], 20252026), 20252026);
    }

    #[test]
    fn ci_lookup() {
        let obj = json!({"Team": "BOS", "player_id": 5});
        let map = obj.as_object().unwrap();
        assert_eq!(ci_get(map, "team").and_then(Value::as_str), Some("BOS"));
        assert_eq!(ci_get(map, "PLAYER_ID").and_then(Value::as_i64), Some(5));
    }
}
