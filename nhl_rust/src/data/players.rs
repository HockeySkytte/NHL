//! Player name/position lookups — ports of `_load_player_names_db`,
//! `_load_player_names_for_seasons`, `_load_player_info_targeted`,
//! `_load_player_info_for_seasons` and `_get_team_pids_for_seasons`.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::sync::Arc;

use serde_json::{json, Value};

use crate::state::Caches;
use crate::supabase::read::{filters, SbClient};
use crate::util::parse::safe_int;

fn players_cols() -> &'static str {
    "player_id,player,position"
}

/// `_load_player_names_db(season)`: players table → `Arc<{pid: name}>`, TTL cached.
pub async fn load_player_names_db(
    caches: &Caches,
    sb: &SbClient,
    season: i64,
) -> Arc<BTreeMap<i64, String>> {
    if let Some(m) = caches.player_names.get(&season) {
        return m;
    }
    let mut out = BTreeMap::new();
    if let Some(rows) = sb
        .read(
            "players",
            "player_id,player",
            Some(&filters(&[("season", &format!("eq.{season}"))])),
            None,
            None,
            0,
        )
        .await
    {
        for r in &rows {
            if let Some(pid) = safe_int(r.get("player_id")) {
                let name = str_of(r.get("player"));
                if !name.is_empty() {
                    out.insert(pid, name);
                }
            }
        }
        let arc = Arc::new(out);
        if !arc.is_empty() {
            caches.player_names.insert(season, arc.clone());
        }
        return arc;
    }
    Arc::new(out)
}

/// `_load_player_names_for_seasons(season_ids)`: DB names (batched `in.(...)`
/// for uncached seasons), then roster fallback.
pub async fn load_player_names_for_seasons(
    caches: &Caches,
    sb: Option<&SbClient>,
    season_ids: &[i64],
) -> BTreeMap<i64, String> {
    let mut names: BTreeMap<i64, String> = BTreeMap::new();
    let mut normalized: Vec<i64> = season_ids.to_vec();
    normalized.sort_unstable();
    normalized.dedup();

    let mut missing: Vec<i64> = Vec::new();
    for &season in &normalized {
        if let Some(cached) = caches.player_names.get(&season) {
            for (pid, name) in cached.iter() {
                names.insert(*pid, name.clone());
            }
        } else {
            missing.push(season);
        }
    }

    let mut need_roster_fallback = false;
    if !missing.is_empty() {
        if let Some(sb) = sb {
            if missing.len() == 1 {
                let db_names = load_player_names_db(caches, sb, missing[0]).await;
                if db_names.is_empty() {
                    need_roster_fallback = true;
                } else {
                    for (pid, name) in db_names.iter() {
                        names.insert(*pid, name.clone());
                    }
                }
            } else {
                let season_csv = missing
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                let mut by_season: HashMap<i64, BTreeMap<i64, String>> = HashMap::new();
                if let Some(rows) = sb
                    .read(
                        "players",
                        "season,player_id,player",
                        Some(&filters(&[("season", &format!("in.({season_csv})"))])),
                        None,
                        None,
                        0,
                    )
                    .await
                {
                    for r in &rows {
                        let Some(s) = safe_int(r.get("season")) else { continue };
                        let Some(pid) = safe_int(r.get("player_id")) else { continue };
                        let name = str_of(r.get("player"));
                        if !name.is_empty() {
                            by_season.entry(s).or_default().insert(pid, name);
                        }
                    }
                }
                for &s in &missing {
                    if let Some(map) = by_season.remove(&s) {
                        for (pid, name) in &map {
                            names.insert(*pid, name.clone());
                        }
                        if !map.is_empty() {
                            caches.player_names.insert(s, Arc::new(map));
                        } else {
                            need_roster_fallback = true;
                        }
                    } else {
                        need_roster_fallback = true;
                    }
                }
            }
        }
    }

    if need_roster_fallback {
        if let Some(sb) = sb {
            let rosters = crate::data::rosters::all_rosters(caches, sb.http()).await;
            if let Some(obj) = rosters.as_object() {
                for (pid_s, info) in obj {
                    if let Ok(pid) = pid_s.parse::<i64>() {
                        let name = str_of(info.get("name"));
                        if !name.is_empty() {
                            names.entry(pid).or_insert(name);
                        }
                    }
                }
            }
        }
    }
    names
}

fn str_of(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// `_get_team_pids_for_seasons`: player IDs who played for a team via game_data
/// (single `in.(...)` query across all seasons).
pub async fn get_team_pids_for_seasons(
    sb: &SbClient,
    team: &str,
    season_ids: &[i64],
) -> HashSet<i64> {
    let mut team_pids = HashSet::new();
    if season_ids.is_empty() {
        return team_pids;
    }
    let season_expr = if season_ids.len() == 1 {
        format!("eq.{}", season_ids[0])
    } else {
        let csv = season_ids
            .iter()
            .map(|s| s.to_string())
            .collect::<Vec<_>>()
            .join(",");
        format!("in.({csv})")
    };
    let rows = sb
        .read(
            "game_data",
            "player_id",
            Some(&filters(&[
                ("season", &season_expr),
                ("team", &format!("eq.{team}")),
            ])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    for r in &rows {
        if let Some(pid) = safe_int(r.get("player_id")) {
            if pid > 0 {
                team_pids.insert(pid);
            }
        }
    }
    team_pids
}

/// `_load_player_info_targeted(player_ids)`: chunked `in.(pids)` query.
pub async fn load_player_info_targeted(
    sb: &SbClient,
    player_ids: &[i64],
) -> HashMap<String, Value> {
    let mut info: HashMap<String, Value> = HashMap::new();
    if player_ids.is_empty() {
        return info;
    }
    let mut pid_list: Vec<i64> = player_ids.iter().copied().collect();
    pid_list.sort_unstable();
    pid_list.dedup();
    for chunk in pid_list.chunks(200) {
        let pid_filter = chunk
            .iter()
            .map(|p| p.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let rows = sb
            .read(
                "players",
                players_cols(),
                Some(&filters(&[("player_id", &format!("in.({pid_filter})"))])),
                None,
                None,
                0,
            )
            .await
            .unwrap_or_default();
        for r in &rows {
            if let Some(pid) = safe_int(r.get("player_id")) {
                info.entry(pid.to_string()).or_insert_with(|| {
                    json!({
                        "name": str_of(r.get("player")),
                        "position": str_of(r.get("position")).to_uppercase(),
                    })
                });
            }
        }
    }
    info
}

/// `_load_player_info_for_seasons(season_ids)`: players table (single batched
/// query), then roster fallback. Returns {pid_str: {name, position}}.
pub async fn load_player_info_for_seasons(
    caches: &Caches,
    sb: Option<&SbClient>,
    season_ids: &[i64],
) -> HashMap<String, Value> {
    let mut info: HashMap<String, Value> = HashMap::new();
    let mut normalized: Vec<i64> = season_ids.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    if let Some(sb) = sb {
        if !normalized.is_empty() {
            let season_expr = if normalized.len() == 1 {
                format!("eq.{}", normalized[0])
            } else {
                let csv = normalized
                    .iter()
                    .map(|s| s.to_string())
                    .collect::<Vec<_>>()
                    .join(",");
                format!("in.({csv})")
            };
            let rows = sb
                .read(
                    "players",
                    players_cols(),
                    Some(&filters(&[("season", &season_expr)])),
                    None,
                    None,
                    0,
                )
                .await
                .unwrap_or_default();
            for r in &rows {
                let Some(pid) = safe_int(r.get("player_id")) else { continue };
                if pid <= 0 {
                    continue;
                }
                let key = pid.to_string();
                let rec = info
                    .entry(key)
                    .or_insert_with(|| json!({"name": "", "position": ""}));
                let name = str_of(r.get("player"));
                let pos = str_of(r.get("position")).to_uppercase();
                if !name.is_empty() && str_of(rec.get("name")).is_empty() {
                    rec.as_object_mut().unwrap().insert("name".into(), json!(name));
                }
                if !pos.is_empty() && str_of(rec.get("position")).is_empty() {
                    rec.as_object_mut().unwrap().insert("position".into(), json!(pos));
                }
            }
        }
    }
    if !info.is_empty() {
        return info;
    }
    // Roster fallback.
    let names = match sb {
        Some(sb) => load_player_names_for_seasons(caches, Some(sb), &normalized).await,
        None => BTreeMap::new(),
    };
    for (pid, name) in names {
        info.insert(pid.to_string(), json!({"name": name, "position": ""}));
    }
    info
}
