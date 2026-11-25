//! BoxID grid map — port of `_get_boxid_map()` in `app/routes.py`.
//! Supabase `box_ids` table ONLY (no CSV fallback).
//! Map: (xi, yi) -> (BoxID, BoxID_rev, BoxSize).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::state::Caches;
use crate::supabase::read::SbClient;

pub type BoxIdMap = HashMap<(i64, i64), (String, String, i64)>;

fn parse_row(v: &Value, out: &mut BoxIdMap) {
    let x = v.get("x").and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|i| i as f64)));
    let y = v.get("y").and_then(|x| x.as_f64().or_else(|| x.as_i64().map(|i| i as f64)));
    let (Some(x), Some(y)) = (x, y) else { return };
    let bid = v
        .get("box_id")
        .or_else(|| v.get("BoxID"))
        .and_then(|b| b.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let bre = v
        .get("box_id_rev")
        .or_else(|| v.get("BoxID_rev"))
        .and_then(|b| b.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let bsi = v
        .get("box_size")
        .or_else(|| v.get("Boxsize"))
        .and_then(|b| b.as_i64());
    let (Some(bid), Some(bre), Some(bsi)) = (bid, bre, bsi) else { return };
    out.insert((x as i64, y as i64), (bid, bre, bsi));
}

/// `_get_boxid_map()`, cached as `Arc<BoxIdMap>` for a week. Supabase `box_ids` only.
pub async fn get_boxid_map(
    caches: &Caches,
    sb: Option<&SbClient>,
    _repo_root: &Path,
) -> Arc<BoxIdMap> {
    if let Some(m) = caches.box_id_map.get(&()) {
        return m;
    }
    let mut map = BoxIdMap::new();
    if let Some(sb) = sb {
        if let Some(rows) = sb
            .read("box_ids", "x,y,box_id,box_id_rev,box_size", None, None, None, 0)
            .await
        {
            for row in &rows {
                parse_row(row, &mut map);
            }
        }
    }
    let arc = Arc::new(map);
    if !arc.is_empty() {
        caches.box_id_map.insert((), arc.clone());
    }
    arc
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parse_row_from_supabase_shape() {
        let mut m = BoxIdMap::new();
        let row = json!({"x": 100, "y": -14, "BoxID": "O01", "BoxID_rev": "D01", "Boxsize": 194});
        parse_row(&row, &mut m);
        assert_eq!(m.get(&(100, -14)), Some(&("O01".to_string(), "D01".to_string(), 194)));
    }
}
