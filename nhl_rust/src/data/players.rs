//! Player name/position lookups — ports of `_load_player_names_db`,
//! `_load_player_names_for_seasons`, `_load_player_info_targeted`,
//! `_load_player_info_for_seasons` and `_get_team_pids_for_seasons`.

use std::collections::{BTreeMap, HashMap, HashSet};

use serde_json::{json, Value};

use crate::state::Caches;
use crate::supabase::read::{filters, SbClient};
use crate::util::parse::safe_int;

fn players_cols() -> &'static str {
    "player_id,player,position"
}

/// `_load_player_names_db(season)`: players table → {pid: name}, TTL cached.
pub async fn load_player_names_db(caches: &Caches, sb: &SbClient, season: i64) -> BTreeMap<i64, String> {
    if let Some(v) = caches.player_names.get(&season) {
        let mut out = BTreeMap::new();
        if let Some(obj) = v.as_object() {
            for (k, val) in obj {
                if let Ok(pid) = k.parse::<i64>() {
                    out.insert(pid, str_of(Some(val)));
                }
            }
        }
        return out;
    }
    let mut out = BTreeMap::new();
    let rows = sb
        .read(
            "players",
            players_cols(),
            Some(&filters(&[("season", &format!("eq.{season}"))])),
            None,
            None,
            0,
        )
        .await
        .unwrap_or_default();
    for r in &rows {
        if let Some(pid) = safe_int(r.get("player_id")) {
            let name = str_of(r.get("player"));
            if !name.is_empty() {
                out.insert(pid, name);
            }
        }
    }
    caches.player_names.insert(season, json!(out));
    out
}

/// `_load_player_names_for_seasons(season_ids)`: DB names, then roster fallback.
pub async fn load_player_names_for_seasons(
    caches: &Caches,
    sb: Option<&SbClient>,
    season_ids: &[i64],
) -> BTreeMap<i64, String> {
    let mut names: BTreeMap<i64, String> = BTreeMap::new();
    let mut normalized: Vec<i64> = season_ids.to_vec();
    normalized.sort_unstable();
    normalized.dedup();
    for season in &normalized {
        let db_names = match sb {
            Some(sb) => load_player_names_db(caches, sb, *season).await,
            None => BTreeMap::new(),
        };
        if !db_names.is_empty() {
            for (pid, name) in db_names {
                names.insert(pid, name);
            }
            continue;
        }
        // Roster fallback (current-season bios), mirroring Flask.
        if let Some(sb) = sb {
            let rosters = crate::data::rosters::all_rosters(caches, sb.http()).await;
            if let Some(obj) = rosters.as_object() {
                for (pid_s, info) in obj {
                    if let Ok(pid) = pid_s.parse::<i64>() {
                        let name = str_of(info.get("name"));
                        if !name.is_empty() {
                            names.insert(pid, name);
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

/// `_get_team_pids_for_seasons`: player IDs who played for a team via game_data.
pub async fn get_team_pids_for_seasons(
    sb: &SbClient,
    team: &str,
    season_ids: &[i64],
) -> HashSet<i64> {
    let mut team_pids = HashSet::new();
    for season in season_ids {
        let rows = sb
            .read(
                "game_data",
                "player_id",
                Some(&filters(&[
                    ("season", &format!("eq.{season}")),
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
    }
    team_pids
}

/// `_load_player_info_targeted(player_ids)`: one small `in.(pids)` query.
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
    let pid_filter = pid_list
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
    info
}

/// `_load_player_info_for_seasons(season_ids)`: players table per season, then
/// roster fallback. Returns {pid_str: {name, position}}.
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
        for season in &normalized {
            let rows = sb
                .read(
                    "players",
                    players_cols(),
                    Some(&filters(&[("season", &format!("eq.{season}"))])),
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
