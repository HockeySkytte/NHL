//! Team SeasonStats rows — port of `_iter_teamseasonstats_static_rows` for the
//! no-filter case (used by `/api/seasons/<team>`). Supabase-first, CSV fallback.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::config::Config;
use crate::state::Caches;
use crate::supabase::read::SbClient;

const COLS: &str = "season,season_state,strength_state,team,gp,toi,cf,ca,ff,fa,sf,sa,gf,ga,xgf_f,xga_f,xgf_s,xga_s,xgf_f2,xga_f2";

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

/// Returns all Team SeasonStats rows (Supabase-only). Cached.
pub async fn iter_rows(caches: &Caches, sb: Option<&SbClient>, cfg: &Config) -> Vec<Value> {
    if let Some(v) = caches.teamseasonstats.get(&()) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }
    let rows = iter_rows_filtered(caches, sb, cfg, &[], 0).await;
    if !rows.is_empty() {
        caches.teamseasonstats.insert((), Value::Array(rows.clone()));
    }
    rows
}

/// `_iter_teamseasonstats_static_rows(seasons=...)`: Supabase-only, cached per
/// normalized season list and batched with `season=in.(...)`.
pub async fn iter_rows_filtered(
    caches: &Caches,
    sb: Option<&SbClient>,
    _cfg: &Config,
    seasons: &[i64],
    _primary: i64,
) -> Vec<Value> {
    let mut norm: Vec<i64> = seasons.to_vec();
    norm.sort_unstable();
    norm.dedup();
    let cache_key = if norm.is_empty() {
        "ALL".to_string()
    } else {
        norm.iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    if let Some(cached) = caches.team_seasonstats_filtered.get(&cache_key) {
        return (*cached).clone();
    }
    let Some(sb) = sb else {
        return Vec::new();
    };
    let cm = col_map();
    let fetched = if !norm.is_empty() {
        let expr = if norm.len() == 1 {
            format!("eq.{}", norm[0])
        } else {
            format!("in.({cache_key})")
        };
        sb.read(
            "season_stats_teams",
            COLS,
            Some(&crate::supabase::read::filters(&[("season", &expr)])),
            Some(&cm),
            None,
            0,
        )
        .await
    } else {
        sb.read("season_stats_teams", COLS, None, Some(&cm), None, 0)
            .await
    };
    let rows = fetched.unwrap_or_default();
    if !rows.is_empty() {
        caches
            .team_seasonstats_filtered
            .insert(cache_key, Arc::new(rows.clone()));
    }
    rows
}
