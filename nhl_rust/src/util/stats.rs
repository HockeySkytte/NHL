//! Shared analytics math helpers ported from `app/routes.py`
//! (normal CDF via erf, bisect percentiles, rate conversion, NHL Edge
//! extraction helpers).

use serde_json::{Map, Value};

use crate::util::parse::ci_get;

/// Normal CDF percentile from a z-score (`_z_to_pct`).
pub fn z_to_pct(z: Option<f64>) -> Option<f64> {
    let zz = z?;
    if !zz.is_finite() {
        return None;
    }
    Some(50.0 * (1.0 + libm::erf(zz / std::f64::consts::SQRT_2)))
}

/// `bisect.bisect_right(values, v)`.
pub fn bisect_right(values: &[f64], v: f64) -> usize {
    values.partition_point(|&x| x <= v)
}

/// `_percentile_sorted(values_sorted, v)`.
pub fn percentile_sorted(values_sorted: &[f64], v: Option<f64>) -> Option<f64> {
    let v = v?;
    if values_sorted.is_empty() {
        return None;
    }
    let idx = bisect_right(values_sorted, v);
    Some(100.0 * (idx as f64 / values_sorted.len() as f64))
}

/// `_rate_from(gp, toi, vv, rates)`.
pub fn rate_from(gp: f64, toi: f64, vv: Option<f64>, rates: &str) -> Option<f64> {
    if rates == "Totals" {
        return vv;
    }
    let denom = if rates == "PerGame" {
        if gp > 0.0 {
            Some(gp)
        } else {
            None
        }
    } else if toi > 0.0 {
        Some(toi / 60.0)
    } else {
        None
    };
    match (vv, denom) {
        (Some(v), Some(d)) if d > 0.0 => Some(v / d),
        _ => None,
    }
}

/// `_pct(n, d)`.
pub fn pct(n: Option<f64>, d: Option<f64>) -> Option<f64> {
    match (n, d) {
        (Some(n), Some(d)) if d > 0.0 => Some(100.0 * n / d),
        _ => None,
    }
}

/// `_lower_is_better(metric_id)`.
pub fn lower_is_better(metric_id: &str) -> bool {
    let m = metric_id.rsplit('|').next().unwrap_or(metric_id).trim();
    matches!(
        m,
        "CA" | "FA" | "SA" | "GA" | "xGA"
            | "PIM_taken"
            | "PIM_Against"
            | "Giveaways"
            | "RAPM CA"
            | "RAPM GA"
            | "RAPM xGA"
    )
}

/// `_edge_game_type`.
pub fn edge_game_type(season_state: &str) -> i64 {
    if season_state.trim().to_lowercase() == "playoffs" {
        3
    } else {
        2
    }
}

/// `_edge_strength_code`.
pub fn edge_strength_code(strength_state: &str) -> Option<&'static str> {
    match strength_state.trim() {
        "5v5" => Some("es"),
        "PP" => Some("pp"),
        "SH" => Some("pk"),
        "all" => Some("all"),
        _ => None,
    }
}

/// `_edge_format_url`.
pub fn edge_format_url(link: &str, player_id: i64, season: i64, game_type: i64) -> Option<String> {
    let mut link = link.trim().to_string();
    if link.is_empty() {
        return None;
    }
    if link.starts_with("api-web.nhle.com/") {
        link = format!("https://{link}");
    }
    if let Some(rest) = link.strip_prefix("http://") {
        link = format!("https://{rest}");
    }
    if !link.starts_with("https://") {
        return None;
    }
    let trimmed = link.trim_end_matches('/');
    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() >= 3
        && parts[parts.len() - 1].chars().all(|c| c.is_ascii_digit())
        && parts[parts.len() - 2].chars().all(|c| c.is_ascii_digit())
        && parts[parts.len() - 3].chars().all(|c| c.is_ascii_digit())
    {
        let mut new_parts = parts.clone();
        let n = new_parts.len();
        let pid_s = player_id.to_string();
        let season_s = season.to_string();
        let gt_s = game_type.to_string();
        new_parts[n - 3] = &pid_s;
        new_parts[n - 2] = &season_s;
        new_parts[n - 1] = &gt_s;
        return Some(new_parts.join("/"));
    }
    Some(link)
}

/// `_edge_pct_to_100`.
pub fn edge_pct_to_100(p: Option<&Value>) -> Option<f64> {
    let raw = parse_num(p)?;
    if !raw.is_finite() {
        return None;
    }
    if (0.0..=1.0).contains(&raw) {
        Some(100.0 * raw)
    } else if (0.0..=100.0).contains(&raw) {
        Some(raw)
    } else {
        None
    }
}

fn parse_num(v: Option<&Value>) -> Option<f64> {
    match v? {
        Value::Number(n) => n.as_f64().or_else(|| n.as_i64().map(|i| i as f64)),
        Value::String(s) => s.trim().parse::<f64>().ok(),
        _ => None,
    }
}

fn coerce_pctg(v: Option<f64>, key: &str) -> Option<f64> {
    let f = v?;
    if key.ends_with("Pctg") && (0.0..=1.5).contains(&f) {
        Some(100.0 * f)
    } else {
        Some(f)
    }
}

fn pick_val(node: &Map<String, Value>) -> Option<f64> {
    ci_get(node, "imperial")
        .or_else(|| ci_get(node, "value"))
        .or_else(|| ci_get(node, "metric"))
        .and_then(|v| parse_num(Some(v)))
}

/// `_edge_extract_value_and_pct`.
pub fn edge_extract_value_and_pct(
    payload: &Value,
    metric_key: &str,
    strength_code: Option<&str>,
) -> (Option<f64>, Option<f64>) {
    let Some(pobj) = payload.as_object() else {
        return (None, None);
    };

    // 1) Direct hit in a nested dict of details.
    for v in pobj.values() {
        if let Some(node) = v.as_object() {
            if let Some(n) = ci_get(node, metric_key) {
                if let Some(nobj) = n.as_object() {
                    let val = pick_val(nobj);
                    let pct = edge_pct_to_100(ci_get(nobj, "percentile"));
                    return (coerce_pctg(val, metric_key), pct);
                }
                if let Some(f) = parse_num(Some(n)) {
                    return (
                        coerce_pctg(Some(f), metric_key),
                        edge_pct_to_100(ci_get(node, &format!("{metric_key}Percentile"))),
                    );
                }
            }
        }
    }

    // 2) Strength-split list.
    let mut rows: Option<Vec<Value>> = None;
    for v in pobj.values() {
        if let Some(arr) = v.as_array() {
            if !arr.is_empty() {
                if let Some(first) = arr[0].as_object() {
                    if first.keys().any(|k| k.to_lowercase() == "strengthcode") {
                        rows = Some(arr.clone());
                        break;
                    }
                }
            }
        }
    }
    if let Some(rows) = rows {
        let wanted = strength_code.map(|s| s.to_lowercase());
        let mut row: Option<Value> = None;
        if let Some(w) = &wanted {
            for rr in &rows {
                if str_of(rr.get("strengthCode")).to_lowercase() == *w {
                    row = Some(rr.clone());
                    break;
                }
            }
        }
        if row.is_none() {
            for rr in &rows {
                if str_of(rr.get("strengthCode")).to_lowercase() == "all" {
                    row = Some(rr.clone());
                    break;
                }
            }
        }
        if row.is_none() {
            row = rows.first().cloned();
        }
        let Some(row) = row else { return (None, None) };
        let robj = row.as_object().cloned().unwrap_or_default();
        let val0 = ci_get(&robj, metric_key).cloned();
        let mut pct_raw = ci_get(&robj, &format!("{metric_key}Percentile")).cloned();
        if pct_raw.is_none() && metric_key.ends_with("Pctg") {
            let base = &metric_key[..metric_key.len() - 4];
            pct_raw = ci_get(&robj, &format!("{base}Percentile")).cloned();
        }
        let pct0 = edge_pct_to_100(pct_raw.as_ref());
        if let Some(vobj) = val0.as_ref().and_then(|v| v.as_object()) {
            let val = pick_val(vobj);
            let pct = edge_pct_to_100(ci_get(vobj, "percentile"));
            return (coerce_pctg(val, metric_key), pct);
        }
        let out_val = parse_num(val0.as_ref());
        return (coerce_pctg(out_val, metric_key), pct0);
    }
    (None, None)
}

/// `_edge_extract_value_pct_avg`.
pub fn edge_extract_value_pct_avg(
    payload: &Value,
    metric_key: &str,
    strength_code: Option<&str>,
) -> (Option<f64>, Option<f64>, Option<f64>) {
    let (val, pct) = edge_extract_value_and_pct(payload, metric_key, strength_code);

    let coerce_avg = |x: Option<f64>| -> Option<f64> {
        let f = x?;
        if !f.is_finite() {
            return None;
        }
        if metric_key.ends_with("Pctg") && (0.0..=1.0).contains(&f) {
            Some(100.0 * f)
        } else {
            Some(f)
        }
    };

    let Some(pobj) = payload.as_object() else {
        return (val, pct, None);
    };
    // Strength-split list rows.
    let mut rows: Option<Vec<Value>> = None;
    for v in pobj.values() {
        if let Some(arr) = v.as_array() {
            if !arr.is_empty() {
                if let Some(first) = arr[0].as_object() {
                    if first.keys().any(|k| k.to_lowercase() == "strengthcode") {
                        rows = Some(arr.clone());
                        break;
                    }
                }
            }
        }
    }
    if let Some(rows) = rows {
        let wanted = strength_code.map(|s| s.to_lowercase());
        let mut row: Option<Value> = None;
        if let Some(w) = &wanted {
            for rr in &rows {
                if str_of(rr.get("strengthCode")).to_lowercase() == *w {
                    row = Some(rr.clone());
                    break;
                }
            }
        }
        if row.is_none() {
            for rr in &rows {
                if str_of(rr.get("strengthCode")).to_lowercase() == "all" {
                    row = Some(rr.clone());
                    break;
                }
            }
        }
        if row.is_none() {
            row = rows.first().cloned();
        }
        if let Some(row) = row {
            let robj = row.as_object().cloned().unwrap_or_default();
            let base = if metric_key.ends_with("Pctg") {
                &metric_key[..metric_key.len() - 4]
            } else {
                metric_key
            };
            let avg_raw = ci_get(&robj, &format!("{metric_key}LeagueAvg"))
                .or_else(|| ci_get(&robj, &format!("{base}LeagueAvg")))
                .or_else(|| ci_get(&robj, &format!("{metric_key}LeagueAverage")))
                .or_else(|| ci_get(&robj, &format!("{base}LeagueAverage")))
                .and_then(|v| parse_num(Some(v)));
            return (val, pct, coerce_avg(avg_raw));
        }
    }
    // Nested dict-of-metrics nodes.
    for v in pobj.values() {
        if let Some(node) = v.as_object() {
            if let Some(n) = ci_get(node, metric_key) {
                if let Some(nobj) = n.as_object() {
                    let avg_raw = ci_get(nobj, "leagueAvg")
                        .or_else(|| ci_get(nobj, "leagueAverage"))
                        .or_else(|| ci_get(nobj, "nhlAvg"))
                        .or_else(|| ci_get(nobj, "nhlAverage"))
                        .and_then(|v| parse_num(Some(v)));
                    return (val, pct, coerce_avg(avg_raw));
                }
            }
        }
    }
    (val, pct, None)
}

fn str_of(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// Fetch an NHL Edge payload with a TTL cache (`_edge_get_cached_json`).
pub async fn edge_get_cached_json(
    caches: &crate::state::Caches,
    http: &reqwest::Client,
    url: &str,
) -> Option<Value> {
    if let Some(v) = caches.edge_api.get(&url.to_string()) {
        return Some(v);
    }
    let data = crate::nhl::client::get_json(http, url, 20).await.ok()?;
    if !data.is_object() {
        return None;
    }
    caches.edge_api.insert(url.to_string(), data.clone());
    Some(data)
}
