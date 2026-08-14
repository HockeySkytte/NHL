//! Player bios/roster maps (ports of `_load_skater_bios_season_cached`,
//! `_load_goalie_bios_season_cached`, `_load_all_rosters_cached`).

use serde_json::{json, Map, Value};

use crate::state::Caches;
use crate::nhl::stats_rest::bios_rows;
use crate::util::dates::current_season_id;
use crate::util::parse::{safe_int, str_value};

fn bios_record(
    pid: i64,
    name: &str,
    team: &str,
    pos: &str,
    pos_code: &str,
    shoots: &str,
    birth_date: &str,
) -> Value {
    json!({
        "playerId": pid.to_string(),
        "name": name,
        "team": team,
        "position": pos,
        "positionCode": pos_code,
        "shoots": shoots,
        "birthDate": birth_date,
    })
}

/// `_load_skater_bios_season_cached(season)`.
pub async fn skater_bios(caches: &Caches, http: &reqwest::Client, season: i64) -> Value {
    if let Some(v) = caches.skater_bios.get(&season) {
        return v;
    }
    let mut out: Map<String, Value> = Map::new();
    if let Ok(rows) = bios_rows(http, "skater", season).await {
        for row in rows {
            let Some(pid) = safe_int(row.get("playerId")) else { continue };
            if pid <= 0 {
                continue;
            }
            let team = str_value(row.get("currentTeamAbbrev")).to_uppercase();
            let name = str_value(row.get("skaterFullName"));
            let pos_raw = str_value(row.get("positionCode")).to_uppercase();
            let pos = if pos_raw.starts_with('D') { "D" } else { "F" };
            let shoots_raw = first_nonempty(&[
                str_value(row.get("shootsCatches")),
                str_value(row.get("shoots")),
                str_value(row.get("ShootsCatches")),
            ])
            .to_uppercase();
            let shoots = if shoots_raw.starts_with('L') {
                "L"
            } else if shoots_raw.starts_with('R') {
                "R"
            } else {
                ""
            };
            let entry = bios_record(pid, &name, &team, pos, &pos_raw, shoots, &str_value(row.get("birthDate")));
            // Python prefers keeping an existing record when the new name is empty.
            let existing = out.get(&pid.to_string());
            let keep_new = match existing {
                None => true,
                Some(e) => !name.is_empty() && str_value(e.get("name")).is_empty(),
            };
            if keep_new {
                out.insert(pid.to_string(), entry);
            }
        }
    }
    let v = Value::Object(out);
    caches.skater_bios.insert(season, v.clone());
    v
}

/// `_load_goalie_bios_season_cached(season)`.
pub async fn goalie_bios(caches: &Caches, http: &reqwest::Client, season: i64) -> Value {
    if let Some(v) = caches.goalie_bios.get(&season) {
        return v;
    }
    let mut out: Map<String, Value> = Map::new();
    if let Ok(rows) = bios_rows(http, "goalie", season).await {
        for row in rows {
            let Some(pid) = safe_int(row.get("playerId")) else { continue };
            if pid <= 0 {
                continue;
            }
            let team = str_value(row.get("currentTeamAbbrev")).to_uppercase();
            let name = first_nonempty(&[
                str_value(row.get("goalieFullName")),
                str_value(row.get("playerFullName")),
                str_value(row.get("skaterFullName")),
            ]);
            let pos_raw = str_value(row.get("positionCode")).to_uppercase();
            let pos_raw = if pos_raw.is_empty() { "G".to_string() } else { pos_raw };
            let shoots_raw = first_nonempty(&[
                str_value(row.get("shootsCatches")),
                str_value(row.get("catches")),
            ])
            .to_uppercase();
            let shoots = if shoots_raw.starts_with('L') {
                "L"
            } else if shoots_raw.starts_with('R') {
                "R"
            } else {
                ""
            };
            out.insert(
                pid.to_string(),
                bios_record(pid, &name, &team, "G", &pos_raw, shoots, &str_value(row.get("birthDate"))),
            );
        }
    }
    let v = Value::Object(out);
    caches.goalie_bios.insert(season, v.clone());
    v
}

fn first_nonempty(values: &[String]) -> String {
    values
        .iter()
        .find(|v| !v.is_empty())
        .cloned()
        .unwrap_or_default()
}

/// `_load_all_rosters_cached()`: skater bios merged with goalie bios for the
/// current season.
pub async fn all_rosters(caches: &Caches, http: &reqwest::Client) -> Value {
    if let Some(v) = caches.all_rosters.get(&()) {
        return v;
    }
    let season = current_season_id(None);
    let skaters = skater_bios(caches, http, season).await;
    let goalies = goalie_bios(caches, http, season).await;
    let merged = match (skaters, goalies) {
        (Value::Object(mut a), Value::Object(b)) => {
            for (k, v) in b {
                a.insert(k, v);
            }
            Value::Object(a)
        }
        (Value::Object(a), _) => Value::Object(a),
        (_, Value::Object(b)) => Value::Object(b),
        _ => Value::Object(Map::new()),
    };
    caches.all_rosters.insert((), merged.clone());
    merged
}

/// `_load_all_rosters_for_season_cached(season)`: per-season playerId->info map.
/// Current season uses skater+goalie bios; historical seasons use a best-effort
/// per-team roster fetch from api-web.nhle.com.
pub async fn all_rosters_for_season(caches: &Caches, http: &reqwest::Client, season: i64) -> Value {
    if let Some(v) = caches.all_rosters_by_season.get(&season) {
        return v;
    }
    let current = current_season_id(None);
    let out: Value = if season == current {
        let skaters = skater_bios(caches, http, season).await;
        let goalies = goalie_bios(caches, http, season).await;
        match (skaters, goalies) {
            (Value::Object(mut a), Value::Object(b)) => {
                for (k, v) in b {
                    a.insert(k, v);
                }
                Value::Object(a)
            }
            (Value::Object(a), _) => Value::Object(a),
            (_, Value::Object(b)) => Value::Object(b),
            _ => Value::Object(Map::new()),
        }
    } else {
        // Historical: fetch each team's roster from api-web.nhle.com.
        let mut out_map: Map<String, Value> = Map::new();
        let season_str = season.to_string();
        let abbrevs: Vec<&str> = [
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
            "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
            "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
            "WPG", "WSH", "PHX", "ATL", "ARI",
        ]
        .to_vec();
        for abbr in abbrevs {
            let url = format!("https://api-web.nhle.com/v1/roster/{abbr}/{season_str}");
            let Ok(resp) = http.get(&url).send().await else { continue };
            if !resp.status().is_success() {
                continue;
            }
            let Ok(data) = resp.json::<Value>().await else { continue };
            for grp in ["forwards", "defenses", "goalies"] {
                let Some(rows) = data.get(grp).and_then(|v| v.as_array()) else { continue };
                for row in rows {
                    let Some(pid) = safe_int(row.get("id")) else { continue };
                    if pid <= 0 {
                        continue;
                    }
                    let first = str_value(row.get("firstName"));
                    let last = str_value(row.get("lastName"));
                    let name = if first.is_empty() {
                        last
                    } else if last.is_empty() {
                        first
                    } else {
                        format!("{first} {last}")
                    };
                    let pos_raw = str_value(row.get("positionCode")).to_uppercase();
                    let pos = if pos_raw.starts_with('D') {
                        "D"
                    } else if pos_raw == "G" {
                        "G"
                    } else {
                        "F"
                    };
                    let shoots = str_value(row.get("shootsCatches")).to_uppercase();
                    let entry = json!({
                        "playerId": pid.to_string(),
                        "name": name,
                        "position": pos,
                        "positionCode": pos_raw,
                        "shoots": shoots,
                    });
                    out_map.entry(pid.to_string()).or_insert(entry);
                }
            }
        }
        Value::Object(out_map)
    };
    caches.all_rosters_by_season.insert(season, out.clone());
    out
}
