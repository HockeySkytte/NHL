//! BoxID grid map — port of `_get_boxid_map()` in `app/routes.py`.
//! Supabase `box_ids` table first, then `BoxID.csv` fallback.
//! Map: (xi, yi) -> (BoxID, BoxID_rev, BoxSize).

use std::collections::HashMap;
use std::path::Path;

use serde_json::{json, Value};

use crate::state::Caches;
use crate::supabase::read::{filters, SbClient};

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

fn load_csv(path: &Path, out: &mut BoxIdMap) {
    let mut rdr = match csv::ReaderBuilder::new()
        .flexible(true)
        .has_headers(true)
        .from_path(path)
    {
        Ok(r) => r,
        Err(_) => return,
    };
    let headers: Vec<String> = rdr
        .headers()
        .map(|h| {
            h.iter()
                .map(|s| s.trim().trim_start_matches('\u{feff}').to_ascii_lowercase())
                .collect()
        })
        .unwrap_or_default();
    let col_x = headers.iter().position(|h| h == "x");
    let col_y = headers.iter().position(|h| h == "y");
    let col_bid = headers.iter().position(|h| h == "boxid");
    let col_bre = headers.iter().position(|h| h == "boxid_rev");
    let col_bsz = headers.iter().position(|h| h == "boxsize");
    for rec in rdr.records().flatten() {
        let cell = |i: Option<usize>| i.and_then(|i| rec.get(i)).unwrap_or("");
        let xi = cell(col_x).trim().parse::<f64>().ok();
        let yi = cell(col_y).trim().parse::<f64>().ok();
        let bid = cell(col_bid).trim().to_string();
        let bre = cell(col_bre).trim().to_string();
        let bsz = cell(col_bsz).trim().parse::<i64>().ok();
        let (Some(xi), Some(yi)) = (xi, yi) else { continue };
        let (Some(bsi), bidi, brei) = (bsz, bid, bre) else { continue };
        if !bidi.is_empty() && !brei.is_empty() {
            out.insert((xi as i64, yi as i64), (bidi, brei, bsi));
        }
    }
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

/// `_get_boxid_map()`, cached for a week.
pub async fn get_boxid_map(
    caches: &Caches,
    sb: Option<&SbClient>,
    repo_root: &Path,
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
    if map.is_empty() {
        let candidates = [
            repo_root.join("BoxID.csv"),
            Path::new("BoxID.csv").to_path_buf(),
        ];
        for p in candidates {
            load_csv(&p, &mut map);
            if !map.is_empty() {
                break;
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
    fn csv_parse_bom_header() {
        let mut m = BoxIdMap::new();
        // BOM-prefixed header, case-insensitive keys.
        let dir = std::env::temp_dir();
        let path = dir.join("boxid_test.csv");
        std::fs::write(&path, "\u{feff}x,y,BoxID,BoxID_rev,Boxsize\n100,-42,O01,D01,194\n5,3,O02,D02,150\n")
            .unwrap();
        load_csv(&path, &mut m);
        assert_eq!(m.get(&(100, -42)), Some(&("O01".to_string(), "D01".to_string(), 194)));
        assert_eq!(m.get(&(5, 3)), Some(&("O02".to_string(), "D02".to_string(), 150)));
        std::fs::remove_file(&path).ok();
    }
}
