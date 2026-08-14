//! BoxID grid map — port of `_get_boxid_map()` in `app/routes.py`.
//! Supabase `box_ids` table ONLY (no CSV fallback).
//! Map: (xi, yi) -> (BoxID, BoxID_rev, BoxSize).

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::state::Caches;
use crate::supabase::read::SbClient;

pub type BoxIdMap = HashMap<(i64, i64), (String, String, i64)>;

fn col_map() -> std::collections::HashMap<String, String> {
    let mut m = std::collections::HashMap::new();
    m.insert("x".to_string(), "x".to_string());
    m.insert("y".to_string(), "y".to_string());
    m.insert("box_id".to_string(), "BoxID".to_string());
    m.insert("box_id_rev".to_string(), "BoxID_rev".to_string());
    m.insert("box_size".to_string(), "Boxsize".to_string());
    m
}

fn parse_row(v: &Value, out: &mut BoxIdMap) {
    let x = v.get("x").and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|i| i as f64)));
    let y = v.get("y").and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|i| i as f64)));
    let (Some(x), Some(y)) = (x, y) else { return };
    let bid = v.get("BoxID").and_then(|b| b.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let bre = v.get("BoxID_rev").and_then(|b| b.as_str()).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    let bsi = v.get("Boxsize").and_then(|b| b.as_i64());
    let (Some(bid), Some(bre), Some(bsi)) = (bid, bre, bsi) else { return };
    out.insert((x as i64, y as i64), (bid, bre, bsi));
}

fn map_to_value(m: &BoxIdMap) -> Value {
    let mut rows = Vec::new();
    for ((x, y), (bid, bre, bsi)) in m {
        rows.push(json!({"x": x, "y": y, "BoxID": bid, "BoxID_rev": bre, "Boxsize": bsi}));
    }
    Value::Array(rows)
}

fn value_to_map(v: &Value) -> BoxIdMap {
    let mut m = BoxIdMap::new();
    if let Some(arr) = v.as_array() {
        for row in arr {
            parse_row(row, &mut m);
        }
    }
    m
}

/// `_get_boxid_map()`, cached for a week. Supabase `box_ids` only (no CSV).
pub async fn get_boxid_map(
    caches: &Caches,
    sb: Option<&SbClient>,
    _repo_root: &Path,
) -> BoxIdMap {
    if let Some(v) = caches.box_id_map.get(&()) {
        return value_to_map(&v);
    }
    let mut map = BoxIdMap::new();
    if let Some(sb) = sb {
        if let Some(rows) = sb
            .read("box_ids", "*", None, Some(&col_map()), None, 0)
            .await
        {
            for row in &rows {
                parse_row(row, &mut map);
            }
        }
    }
    caches.box_id_map.insert((), map_to_value(&map));
    map
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_row_from_supabase_shape() {
        let mut m = BoxIdMap::new();
        let row = json!({"x": 100, "y": -14, "BoxID": "O01", "BoxID_rev": "D01", "Boxsize": 194});
        parse_row(&row, &mut m);
        assert_eq!(m.get(&(100, -14)), Some(&("O01".to_string(), "D01".to_string(), 194)));
    }
}
