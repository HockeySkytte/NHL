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

/// `_iter_teamseasonstats_static_rows(seasons=...)`: Supabase-only (the team
/// season stats live in the `season_stats_teams` table; NO CSV fallback).
pub async fn iter_rows_filtered(
    _caches: &Caches,
    sb: Option<&SbClient>,
    _cfg: &Config,
    seasons: &[i64],
    _primary: i64,
) -> Vec<Value> {
    let Some(sb) = sb else {
        return Vec::new();
    };
    if !seasons.is_empty() {
        let mut all: Vec<Value> = Vec::new();
        for s in seasons {
            if let Some(rows) = sb
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
                all.extend(rows);
            }
        }
        return all;
    }
    sb.read("season_stats_teams", "*", None, Some(&col_map()), None, 0)
        .await
        .unwrap_or_default()
}
