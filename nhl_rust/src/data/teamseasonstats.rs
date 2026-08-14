//! Team SeasonStats rows — port of `_iter_teamseasonstats_static_rows` for the
//! no-filter case (used by `/api/seasons/<team>`). Supabase-first, CSV fallback.

use std::collections::HashMap;

use serde_json::Value;

use crate::state::Caches;
use crate::config::Config;
use crate::supabase::read::SbClient;

fn col_map() -> HashMap<String, String> {
    [
        ("season", "Season"),
        ("season_state", "SeasonState"),
        ("strength_state", "StrengthState"),
        ("team", "Team"),
        ("gp", "GP"),
        ("toi", "TOI"),
        ("cf", "CF"),
        ("ca", "CA"),
        ("ff", "FF"),
        ("fa", "FA"),
        ("sf", "SF"),
        ("sa", "SA"),
        ("gf", "GF"),
        ("ga", "GA"),
        ("xgf_f", "xGF_F"),
        ("xga_f", "xGA_F"),
        ("xgf_s", "xGF_S"),
        ("xga_s", "xGA_S"),
        ("xgf_f2", "xGF_F2"),
        ("xga_f2", "xGA_F2"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// Returns all Team SeasonStats rows (Supabase, else CSV). Cached.
pub async fn iter_rows(caches: &Caches, sb: Option<&SbClient>, cfg: &Config) -> Vec<Value> {
    if let Some(v) = caches.teamseasonstats.get(&()) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }
    let rows = iter_rows_filtered(caches, sb, cfg, &[], 0).await;
    caches.teamseasonstats.insert((), Value::Array(rows.clone()));
    rows
}

/// `_iter_teamseasonstats_static_rows(seasons=...)`: Supabase per-season first,
/// CSV fallback for missing seasons (or all when `seasons` is empty).
pub async fn iter_rows_filtered(
    _caches: &Caches,
    sb: Option<&SbClient>,
    cfg: &Config,
    seasons: &[i64],
    _primary: i64,
) -> Vec<Value> {
    if !seasons.is_empty() {
        let mut all: Vec<Value> = Vec::new();
        let mut missing: Vec<i64> = Vec::new();
        if let Some(sb) = sb {
            for s in seasons {
                match sb
                    .read(
                        "season_stats_teams",
                        "*",
                        Some(&crate::supabase::read::filters(&[("season", &format!("eq.{s}"))])),
                        Some(&col_map()),
                        None,
                        0,
                    )
                    .await
                {
                    Some(rows) => all.extend(rows),
                    None => missing.push(*s),
                }
            }
        } else {
            missing = seasons.to_vec();
        }
        if missing.is_empty() {
            return all;
        }
        let csv = load_csv(&cfg.static_dir.join("nhl_seasonstats_teams.csv"));
        let missing_set: std::collections::HashSet<i64> = missing.into_iter().collect();
        for r in &csv {
            if let Some(s) = r
                .get("Season")
                .and_then(Value::as_str)
                .and_then(|v| v.trim().parse::<i64>().ok())
            {
                if missing_set.contains(&s) {
                    all.push(r.clone());
                }
            }
        }
        return all;
    }
    if let Some(sb) = sb {
        match sb.read("season_stats_teams", "*", None, Some(&col_map()), None, 0).await {
            Some(rows) => return rows,
            None => return load_csv(&cfg.static_dir.join("nhl_seasonstats_teams.csv")),
        }
    }
    load_csv(&cfg.static_dir.join("nhl_seasonstats_teams.csv"))
}

fn load_csv(path: &std::path::Path) -> Vec<Value> {
    let mut out = Vec::new();
    if !path.exists() {
        return out;
    }
    let Ok(mut reader) = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
    else {
        return out;
    };
    let headers: Vec<String> = reader
        .headers()
        .map(|h| h.iter().map(|s| s.trim_start_matches('\u{feff}').to_string()).collect())
        .unwrap_or_default();
    for record in reader.records() {
        let Ok(record) = record else { continue };
        let mut map = serde_json::Map::new();
        for (i, value) in record.iter().enumerate() {
            if let Some(header) = headers.get(i) {
                map.insert(header.clone(), Value::String(value.to_string()));
            }
        }
        out.push(Value::Object(map));
    }
    out
}
