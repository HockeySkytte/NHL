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
            let shoots_raw = first_nonempty_val(&[
                row.get("shootsCatches"),
                row.get("shoots"),
                row.get("ShootsCatches"),
            ])
            .to_uppercase();
            let shoots = if shoots_raw.starts_with('L') {
                "L"
            } else if shoots_raw.starts_with('R') {
                "R"
            } else {
                ""
            };
            let pid_s = pid.to_string();
            let entry = bios_record(pid, &name, &team, pos, &pos_raw, shoots, &str_value(row.get("birthDate")));
            // Python prefers keeping an existing record when the new name is empty.
            let keep_new = match out.get(&pid_s) {
                None => true,
                Some(e) => !name.is_empty() && str_value(e.get("name")).is_empty(),
            };
            if keep_new {
                out.insert(pid_s, entry);
            }
        }
    }
    let has_rows = !out.is_empty();
    let v = Value::Object(out);
    if has_rows {
        caches.skater_bios.insert(season, v.clone());
    }
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
            let name = first_nonempty_val(&[
                row.get("goalieFullName"),
                row.get("playerFullName"),
                row.get("skaterFullName"),
            ]);
            let pos_raw = str_value(row.get("positionCode")).to_uppercase();
            let pos_raw = if pos_raw.is_empty() { "G".to_string() } else { pos_raw };
            let shoots_raw = first_nonempty_val(&[
                row.get("shootsCatches"),
                row.get("catches"),
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
    let has_rows = !out.is_empty();
    let v = Value::Object(out);
    if has_rows {
        caches.goalie_bios.insert(season, v.clone());
    }
    v
}

fn first_nonempty_val(values: &[Option<&Value>]) -> String {
    for v in values {
        let s = str_value(*v);
        if !s.is_empty() {
            return s;
        }
    }
    String::new()
}

/// `_load_all_rosters_cached()`: skater bios merged with goalie bios for the
/// current season (fetched concurrently).
pub async fn all_rosters(caches: &Caches, http: &reqwest::Client) -> Value {
    if let Some(v) = caches.all_rosters.get(&()) {
        return v;
    }
    let season = current_season_id(None);
    let (skaters, goalies) = tokio::join!(
        skater_bios(caches, http, season),
        goalie_bios(caches, http, season),
    );
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
    if merged.as_object().is_some_and(|m| !m.is_empty()) {
        caches.all_rosters.insert((), merged.clone());
    }
    merged
}

/// `_load_all_rosters_for_season_cached(season)`: per-season playerId->info map.
/// Current season uses skater+goalie bios; historical seasons use bounded-parallel
/// per-team roster fetches from api-web.nhle.com.
pub async fn all_rosters_for_season(caches: &Caches, http: &reqwest::Client, season: i64) -> Value {
    if let Some(v) = caches.all_rosters_by_season.get(&season) {
        return v;
    }
    let current = current_season_id(None);
    let out: Value = if season == current {
        let (skaters, goalies) = tokio::join!(
            skater_bios(caches, http, season),
            goalie_bios(caches, http, season),
        );
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
        // Historical: fetch all teams with bounded concurrency instead of 35 serial HTTP requests.
        const ABBREVS: [&str; 35] = [
            "ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET",
            "EDM", "FLA", "LAK", "MIN", "MTL", "NJD", "NSH", "NYI", "NYR", "OTT",
            "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
            "WPG", "WSH", "PHX", "ATL", "ARI",
        ];
        const CONCURRENCY: usize = 8;
        let season_str = season.to_string();
        let sem = std::sync::Arc::new(tokio::sync::Semaphore::new(CONCURRENCY));
        let mut handles = Vec::with_capacity(ABBREVS.len());
        for abbr in ABBREVS {
            let http = http.clone();
            let sem = sem.clone();
            let url = format!("https://api-web.nhle.com/v1/roster/{abbr}/{season_str}");
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.ok()?;
                let resp = http
                    .get(&url)
                    .timeout(std::time::Duration::from_secs(15))
                    .send()
                    .await
                    .ok()?;
                if !resp.status().is_success() {
                    return None;
                }
                resp.json::<Value>().await.ok()
            }));
        }
        let mut out_map: Map<String, Value> = Map::new();
        for h in handles {
            let Ok(Some(data)) = h.await else { continue };
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
                    let pid_s = pid.to_string();
                    out_map.entry(pid_s.clone()).or_insert_with(|| {
                        json!({
                            "playerId": pid_s,
                            "name": name,
                            "position": pos,
                            "positionCode": pos_raw,
                            "shoots": shoots,
                        })
                    });
                }
            }
        }
        Value::Object(out_map)
    };
    if out.as_object().is_some_and(|m| !m.is_empty()) {
        caches.all_rosters_by_season.insert(season, out.clone());
    }
    out
}
