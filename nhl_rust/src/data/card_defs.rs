//! Card metric definitions — port of `_load_card_metrics_defs` reading
//! `app/static/card_metrics.csv` (tab/semicolon auto-detect, BOM-tolerant).

use serde_json::{json, Value};

use crate::state::Caches;
use crate::config::Config;

pub async fn load_defs(caches: &Caches, cfg: &Config, card: &str) -> Value {
    let card_norm = card.trim().to_lowercase();
    let card_norm = if card_norm.is_empty() {
        "skaters".to_string()
    } else {
        card_norm
    };
    if let Some(v) = caches.card_metrics_defs.get(&card_norm) {
        return v;
    }
    let out = load_defs_inner(cfg, &card_norm);
    caches.card_metrics_defs.insert(card_norm, out.clone());
    out
}

fn load_defs_inner(cfg: &Config, card_norm: &str) -> Value {
    let path = cfg.static_dir.join("card_metrics.csv");
    let Ok(raw) = std::fs::read_to_string(&path) else {
        return json!({ "categories": [], "metrics": [] });
    };
    // Strip UTF-8 BOM.
    let raw = raw.trim_start_matches('\u{feff}');
    // Auto-detect delimiter.
    let first_line = raw.lines().next().unwrap_or("");
    let delim = if first_line.matches('\t').count() > first_line.matches(';').count() {
        b'\t'
    } else {
        b';'
    };
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .delimiter(delim)
        .from_reader(raw.as_bytes());
    let headers: Vec<String> = reader
        .headers()
        .map(|h| h.iter().map(|s| s.to_string()).collect())
        .unwrap_or_default();
    let hget = |row: &csv::StringRecord, key: &str| -> String {
        for (i, h) in headers.iter().enumerate() {
            if h.eq_ignore_ascii_case(key) {
                return row.get(i).unwrap_or("").trim().to_string();
            }
        }
        String::new()
    };

    let mut metrics: Vec<Value> = Vec::new();
    let mut cats: Vec<String> = Vec::new();
    let mut seen_cat: std::collections::HashSet<String> = Default::default();
    for row in reader.records() {
        let Ok(row) = row else { continue };
        let card = hget(&row, "Card");
        if !card.is_empty() && !card.eq_ignore_ascii_case(card_norm) {
            continue;
        }
        let category = hget(&row, "Category");
        let metric = hget(&row, "Metric");
        if category.is_empty() || metric.is_empty() {
            continue;
        }
        let name = hget(&row, "Name");
        let calc = hget(&row, "Calculation");
        let place = hget(&row, "Place");
        let place = if place.is_empty() { "0".to_string() } else { place };
        let default_raw = hget(&row, "Default");
        let is_default = matches!(default_raw.as_str(), "1" | "true" | "True" | "YES" | "Yes" | "yes");
        let link = hget(&row, "Link");
        let strength_code = hget(&row, "StrengthCode");
        let position_code = hget(&row, "PositionCode");
        metrics.push(json!({
            "id": format!("{category}|{metric}"),
            "category": category,
            "metric": metric,
            "name": if name.is_empty() { metric.clone() } else { name },
            "calculation": calc,
            "default": is_default,
            "place": place,
            "link": link,
            "strengthCode": strength_code,
            "positionCode": position_code,
        }));
        if !seen_cat.contains(&category) {
            seen_cat.insert(category.clone());
            cats.push(category);
        }
    }
    json!({ "categories": cats, "metrics": metrics })
}
