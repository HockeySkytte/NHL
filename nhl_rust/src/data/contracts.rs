//! Contract data loaders backed by Supabase `contracts`, `contract_years`,
//! and `contract_teams_years`.
//!
//! No silent fallbacks: if no rows exist for the requested season(s), returns
//! `None` / empty lists rather than substituting another season.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;

use serde_json::{json, Map, Value};

use crate::state::Caches;
use crate::supabase::read::{filters, SbClient};
use crate::util::parse::{parse_locale_float, safe_int, str_value};

const CONTRACT_YEARS_COLS: &str = "contract_id,player_id,player_name,position,season_id,season,team,aav_usd,total_salary_usd,base_salary_usd,signing_bonus_usd,minor_league_salary_usd,two_way,is_projected";

const CONTRACTS_META_COLS: &str = "contract_id,start_year,end_year,start_season,end_season,term_years,contract_status,expiry_status,aav_usd,total_value_usd,two_way,signed_date,is_left_censored_2010,is_right_censored_2026";

const TEAM_YEARS_COLS: &str = "team,season_year,season_id,season,is_projected,contracts_count,one_way_contracts_count,two_way_contracts_count,total_aav_usd,one_way_aav_usd,total_salary_usd,total_base_salary_usd,total_signing_bonus_usd,buyout_penalty_usd,retained_salary_penalty_usd,bonus_overage_penalty_usd,total_cap_penalties_usd,combined_aav_plus_penalties_usd,combined_one_way_plus_penalties_usd,cap_ceiling_usd,cap_floor_usd,projected_cap_space_usd,penalties_detail";

fn norm_season_ids(season_ids: &[i64]) -> Vec<i64> {
    let mut out: Vec<i64> = season_ids.iter().copied().filter(|&s| s > 0).collect();
    out.sort_unstable();
    out.dedup();
    out
}

fn opt_f64(v: Option<&Value>) -> Option<f64> {
    let val = v?;
    if val.is_null() {
        return None;
    }
    parse_locale_float(Some(val))
}

fn opt_bool(v: Option<&Value>) -> Option<bool> {
    match v {
        Some(Value::Bool(b)) => Some(*b),
        Some(Value::Number(n)) => n.as_i64().map(|x| x != 0),
        Some(Value::String(s)) => match s.trim().to_lowercase().as_str() {
            "true" | "t" | "1" | "yes" => Some(true),
            "false" | "f" | "0" | "no" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

fn opt_nonempty_str(v: Option<&Value>) -> Option<String> {
    let s = str_value(v);
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

fn avg_opt(vals: impl Iterator<Item = Option<f64>>) -> Option<f64> {
    let mut sum = 0.0;
    let mut n = 0usize;
    for v in vals.flatten() {
        if v.is_finite() {
            sum += v;
            n += 1;
        }
    }
    if n == 0 {
        None
    } else {
        Some((sum / n as f64).round())
    }
}

fn pos_group(pos: &str) -> &'static str {
    match pos.trim().to_uppercase().as_str() {
        "G" => "G",
        "D" | "LD" | "RD" => "D",
        _ => "F",
    }
}

async fn fetch_contracts_meta_by_ids(
    sb: &SbClient,
    contract_ids: &[String],
) -> HashMap<String, Value> {
    let mut unique: BTreeSet<&str> = BTreeSet::new();
    for cid in contract_ids {
        let t = cid.trim();
        if !t.is_empty() {
            unique.insert(t);
        }
    }
    if unique.is_empty() {
        return HashMap::new();
    }
    let ids: Vec<&str> = unique.into_iter().collect();
    let mut out: HashMap<String, Value> = HashMap::with_capacity(ids.len());
    // Chunk IN filters to keep PostgREST query strings comfortably short.
    for chunk in ids.chunks(60) {
        let in_list = chunk.join(",");
        let rows = sb
            .read(
                "contracts",
                CONTRACTS_META_COLS,
                Some(&filters(&[("contract_id", &format!("in.({in_list})"))])),
                None,
                None,
                chunk.len(),
            )
            .await
            .unwrap_or_default();
        for row in rows {
            let cid = str_value(row.get("contract_id"));
            if !cid.is_empty() {
                out.insert(cid, row);
            }
        }
    }
    out
}

/// Pure aggregation helper for a player's contract summary across the requested
/// seasons. Exposed for unit tests.
pub fn summarize_player_contract(
    season_ids_requested: &[i64],
    year_rows: &[Value],
    meta_by_id: &HashMap<String, Value>,
) -> Option<Value> {
    if year_rows.is_empty() || season_ids_requested.is_empty() {
        return None;
    }

    let mut rows: Vec<&Value> = year_rows.iter().collect();
    rows.sort_by_key(|r| safe_int(r.get("season_id")).unwrap_or(0));

    let is_multi = season_ids_requested.len() > 1;
    let seasons_matched = rows.len();

    if !is_multi {
        let r = rows[0];
        let cid = str_value(r.get("contract_id"));
        let meta = meta_by_id.get(&cid);
        let end_year = meta.and_then(|m| safe_int(m.get("end_year")));
        let expiry_year = end_year.map(|y| y + 1);
        let term_years = meta
            .and_then(|m| safe_int(m.get("term_years")))
            .filter(|&t| t > 0);

        return Some(json!({
            "mode": "single",
            "seasonsRequested": season_ids_requested.len(),
            "seasonsMatched": seasons_matched,
            "contractId": opt_nonempty_str(r.get("contract_id")),
            "seasonId": safe_int(r.get("season_id")),
            "season": opt_nonempty_str(r.get("season")),
            "team": opt_nonempty_str(r.get("team")),
            "aavUsd": opt_f64(r.get("aav_usd")).or_else(|| meta.and_then(|m| opt_f64(m.get("aav_usd")))),
            "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
            "baseSalaryUsd": opt_f64(r.get("base_salary_usd")),
            "signingBonusUsd": opt_f64(r.get("signing_bonus_usd")),
            "minorLeagueSalaryUsd": opt_f64(r.get("minor_league_salary_usd")),
            "twoWay": opt_bool(r.get("two_way")).or_else(|| meta.and_then(|m| opt_bool(m.get("two_way")))).unwrap_or(false),
            "isProjected": opt_bool(r.get("is_projected")).unwrap_or(false),
            "termYears": term_years,
            "startSeason": meta.and_then(|m| opt_nonempty_str(m.get("start_season"))),
            "endSeason": meta.and_then(|m| opt_nonempty_str(m.get("end_season"))),
            "expiryYear": expiry_year,
            "contractStatus": meta.and_then(|m| opt_nonempty_str(m.get("contract_status"))),
            "expiryStatus": meta.and_then(|m| opt_nonempty_str(m.get("expiry_status"))),
            "totalValueUsd": meta.and_then(|m| opt_f64(m.get("total_value_usd"))),
            "signedDate": meta.and_then(|m| opt_nonempty_str(m.get("signed_date"))),
        }));
    }

    // Multiple seasons chosen: average financial values over the matched years.
    let avg_aav = avg_opt(rows.iter().map(|r| opt_f64(r.get("aav_usd"))));
    let avg_total_salary = avg_opt(rows.iter().map(|r| opt_f64(r.get("total_salary_usd"))));
    let avg_base_salary = avg_opt(rows.iter().map(|r| opt_f64(r.get("base_salary_usd"))));
    let avg_signing_bonus = avg_opt(rows.iter().map(|r| opt_f64(r.get("signing_bonus_usd"))));

    let distinct_cids: BTreeSet<String> = rows
        .iter()
        .map(|r| str_value(r.get("contract_id")))
        .filter(|s| !s.is_empty())
        .collect();
    let shared_meta = if distinct_cids.len() == 1 {
        distinct_cids.iter().next().and_then(|cid| meta_by_id.get(cid))
    } else {
        None
    };

    let seasons_detail: Vec<Value> = rows
        .iter()
        .map(|r| {
            json!({
                "seasonId": safe_int(r.get("season_id")),
                "season": opt_nonempty_str(r.get("season")),
                "team": opt_nonempty_str(r.get("team")),
                "aavUsd": opt_f64(r.get("aav_usd")),
                "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
                "twoWay": opt_bool(r.get("two_way")).unwrap_or(false),
                "isProjected": opt_bool(r.get("is_projected")).unwrap_or(false),
            })
        })
        .collect();

    let end_year = shared_meta.and_then(|m| safe_int(m.get("end_year")));
    let expiry_year = end_year.map(|y| y + 1);
    let term_years = shared_meta
        .and_then(|m| safe_int(m.get("term_years")))
        .filter(|&t| t > 0);

    Some(json!({
        "mode": "average",
        "seasonsRequested": season_ids_requested.len(),
        "seasonsMatched": seasons_matched,
        "aavUsd": avg_aav,
        "totalSalaryUsd": avg_total_salary,
        "baseSalaryUsd": avg_base_salary,
        "signingBonusUsd": avg_signing_bonus,
        "contractsCount": distinct_cids.len(),
        "termYears": term_years,
        "startSeason": shared_meta.and_then(|m| opt_nonempty_str(m.get("start_season"))),
        "endSeason": shared_meta.and_then(|m| opt_nonempty_str(m.get("end_season"))),
        "expiryYear": expiry_year,
        "contractStatus": shared_meta.and_then(|m| opt_nonempty_str(m.get("contract_status"))),
        "expiryStatus": shared_meta.and_then(|m| opt_nonempty_str(m.get("expiry_status"))),
        "seasons": seasons_detail,
    }))
}

/// Loads a player's contract info for the selected season(s), cached in memory.
pub async fn load_player_contract_summary(
    caches: &Caches,
    sb: Option<&SbClient>,
    player_id: i64,
    season_ids: &[i64],
) -> Option<Value> {
    let sb = sb?;
    if player_id <= 0 {
        return None;
    }
    let sids = norm_season_ids(season_ids);
    if sids.is_empty() {
        return None;
    }
    let cache_key = (player_id, sids.clone());
    if let Some(cached) = caches.player_contract.get(&cache_key) {
        return if cached.is_null() { None } else { Some(cached) };
    }

    let sid_in = sids
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let year_rows = sb
        .read(
            "contract_years",
            CONTRACT_YEARS_COLS,
            Some(&filters(&[
                ("player_id", &format!("eq.{player_id}")),
                ("season_id", &format!("in.({sid_in})")),
            ])),
            None,
            Some("season_id.asc"),
            sids.len().max(1) * 2,
        )
        .await
        .unwrap_or_default();

    if year_rows.is_empty() {
        caches.player_contract.insert(cache_key, Value::Null);
        return None;
    }

    let cids: Vec<String> = year_rows
        .iter()
        .map(|r| str_value(r.get("contract_id")))
        .filter(|s| !s.is_empty())
        .collect();
    let meta = fetch_contracts_meta_by_ids(sb, &cids).await;
    let summary = summarize_player_contract(&sids, &year_rows, &meta);
    caches
        .player_contract
        .insert(cache_key, summary.clone().unwrap_or(Value::Null));
    summary
}

fn format_team_year_row(r: &Value) -> Value {
    json!({
        "team": opt_nonempty_str(r.get("team")),
        "seasonYear": safe_int(r.get("season_year")),
        "seasonId": safe_int(r.get("season_id")),
        "season": opt_nonempty_str(r.get("season")),
        "isProjected": opt_bool(r.get("is_projected")).unwrap_or(false),
        "contractsCount": safe_int(r.get("contracts_count")),
        "oneWayContractsCount": safe_int(r.get("one_way_contracts_count")),
        "twoWayContractsCount": safe_int(r.get("two_way_contracts_count")),
        "totalAavUsd": opt_f64(r.get("total_aav_usd")),
        "oneWayAavUsd": opt_f64(r.get("one_way_aav_usd")),
        "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
        "totalBaseSalaryUsd": opt_f64(r.get("total_base_salary_usd")),
        "totalSigningBonusUsd": opt_f64(r.get("total_signing_bonus_usd")),
        "buyoutPenaltyUsd": opt_f64(r.get("buyout_penalty_usd")),
        "retainedSalaryPenaltyUsd": opt_f64(r.get("retained_salary_penalty_usd")),
        "bonusOveragePenaltyUsd": opt_f64(r.get("bonus_overage_penalty_usd")),
        "totalCapPenaltiesUsd": opt_f64(r.get("total_cap_penalties_usd")),
        "combinedAavPlusPenaltiesUsd": opt_f64(r.get("combined_aav_plus_penalties_usd")),
        "combinedOneWayPlusPenaltiesUsd": opt_f64(r.get("combined_one_way_plus_penalties_usd")),
        "capCeilingUsd": opt_f64(r.get("cap_ceiling_usd")),
        "capFloorUsd": opt_f64(r.get("cap_floor_usd")),
        "projectedCapSpaceUsd": opt_f64(r.get("projected_cap_space_usd")),
        "penaltiesDetail": r.get("penalties_detail").cloned().filter(|v| v.is_array()),
    })
}

/// Pure builder for team contract summary, roster breakdown, and future cap
/// outlook. Exposed for unit tests.
pub fn build_team_contracts_payload(
    team: &str,
    season_ids_requested: &[i64],
    team_year_rows: &[Value],
    roster_year_rows: &[Value],
    meta_by_id: &HashMap<String, Value>,
    outlook_rows: &[Value],
) -> Value {
    let is_multi = season_ids_requested.len() > 1;

    let mut ty_sorted: Vec<&Value> = team_year_rows.iter().collect();
    ty_sorted.sort_by_key(|r| safe_int(r.get("season_id")).unwrap_or(0));

    let summary: Value = if ty_sorted.is_empty() {
        Value::Null
    } else if !is_multi {
        format_team_year_row(ty_sorted[0])
    } else {
        let mut s = Map::new();
        s.insert("team".into(), json!(team));
        s.insert("seasonsMatched".into(), json!(ty_sorted.len()));
        s.insert(
            "contractsCount".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("contracts_count"))))),
        );
        s.insert(
            "oneWayContractsCount".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("one_way_contracts_count"))))),
        );
        s.insert(
            "twoWayContractsCount".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("two_way_contracts_count"))))),
        );
        s.insert(
            "totalAavUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("total_aav_usd"))))),
        );
        s.insert(
            "oneWayAavUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("one_way_aav_usd"))))),
        );
        s.insert(
            "totalSalaryUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("total_salary_usd"))))),
        );
        s.insert(
            "totalBaseSalaryUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("total_base_salary_usd"))))),
        );
        s.insert(
            "totalSigningBonusUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("total_signing_bonus_usd"))))),
        );
        s.insert(
            "buyoutPenaltyUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("buyout_penalty_usd"))))),
        );
        s.insert(
            "retainedSalaryPenaltyUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("retained_salary_penalty_usd"))))),
        );
        s.insert(
            "bonusOveragePenaltyUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("bonus_overage_penalty_usd"))))),
        );
        s.insert(
            "totalCapPenaltiesUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("total_cap_penalties_usd"))))),
        );
        s.insert(
            "combinedAavPlusPenaltiesUsd".into(),
            json!(avg_opt(
                ty_sorted
                    .iter()
                    .map(|r| opt_f64(r.get("combined_aav_plus_penalties_usd")))
            )),
        );
        s.insert(
            "combinedOneWayPlusPenaltiesUsd".into(),
            json!(avg_opt(
                ty_sorted
                    .iter()
                    .map(|r| opt_f64(r.get("combined_one_way_plus_penalties_usd")))
            )),
        );
        s.insert(
            "capCeilingUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("cap_ceiling_usd"))))),
        );
        s.insert(
            "capFloorUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("cap_floor_usd"))))),
        );
        s.insert(
            "projectedCapSpaceUsd".into(),
            json!(avg_opt(ty_sorted.iter().map(|r| opt_f64(r.get("projected_cap_space_usd"))))),
        );
        s.insert(
            "seasons".into(),
            Value::Array(ty_sorted.iter().map(|r| format_team_year_row(r)).collect()),
        );
        Value::Object(s)
    };

    // Build roster entries (single-season exact rows, or multi-season per-player averages).
    let mut roster: Vec<Value> = if !is_multi {
        roster_year_rows
            .iter()
            .map(|r| {
                let cid = str_value(r.get("contract_id"));
                let meta = meta_by_id.get(&cid);
                let pos = str_value(r.get("position")).to_uppercase();
                let end_year = meta.and_then(|m| safe_int(m.get("end_year")));
                let term_years = meta
                    .and_then(|m| safe_int(m.get("term_years")))
                    .filter(|&t| t > 0);
                json!({
                    "playerId": safe_int(r.get("player_id")),
                    "playerName": opt_nonempty_str(r.get("player_name")),
                    "position": pos,
                    "posGroup": pos_group(&pos),
                    "contractId": opt_nonempty_str(r.get("contract_id")),
                    "seasonId": safe_int(r.get("season_id")),
                    "season": opt_nonempty_str(r.get("season")),
                    "yearsCount": 1,
                    "aavUsd": opt_f64(r.get("aav_usd")),
                    "totalSalaryUsd": opt_f64(r.get("total_salary_usd")),
                    "baseSalaryUsd": opt_f64(r.get("base_salary_usd")),
                    "signingBonusUsd": opt_f64(r.get("signing_bonus_usd")),
                    "minorLeagueSalaryUsd": opt_f64(r.get("minor_league_salary_usd")),
                    "twoWay": opt_bool(r.get("two_way")).unwrap_or(false),
                    "isProjected": opt_bool(r.get("is_projected")).unwrap_or(false),
                    "termYears": term_years,
                    "startSeason": meta.and_then(|m| opt_nonempty_str(m.get("start_season"))),
                    "endSeason": meta.and_then(|m| opt_nonempty_str(m.get("end_season"))),
                    "expiryYear": end_year.map(|y| y + 1),
                    "contractStatus": meta.and_then(|m| opt_nonempty_str(m.get("contract_status"))),
                    "expiryStatus": meta.and_then(|m| opt_nonempty_str(m.get("expiry_status"))),
                    "totalValueUsd": meta.and_then(|m| opt_f64(m.get("total_value_usd"))),
                    "signedDate": meta.and_then(|m| opt_nonempty_str(m.get("signed_date"))),
                })
            })
            .collect()
    } else {
        let mut by_pid: BTreeMap<i64, Vec<&Value>> = BTreeMap::new();
        for r in roster_year_rows {
            if let Some(pid) = safe_int(r.get("player_id")) {
                by_pid.entry(pid).or_default().push(r);
            }
        }
        by_pid
            .into_iter()
            .map(|(pid, mut prows)| {
                prows.sort_by_key(|r| safe_int(r.get("season_id")).unwrap_or(0));
                let latest = prows.last().copied().expect("non-empty group");
                let cid = str_value(latest.get("contract_id"));
                let meta = meta_by_id.get(&cid);
                let pos = str_value(latest.get("position")).to_uppercase();
                let end_year = meta.and_then(|m| safe_int(m.get("end_year")));
                let term_years = meta
                    .and_then(|m| safe_int(m.get("term_years")))
                    .filter(|&t| t > 0);
                json!({
                    "playerId": pid,
                    "playerName": opt_nonempty_str(latest.get("player_name")),
                    "position": pos,
                    "posGroup": pos_group(&pos),
                    "contractId": opt_nonempty_str(latest.get("contract_id")),
                    "yearsCount": prows.len(),
                    "aavUsd": avg_opt(prows.iter().map(|r| opt_f64(r.get("aav_usd")))),
                    "totalSalaryUsd": avg_opt(prows.iter().map(|r| opt_f64(r.get("total_salary_usd")))),
                    "baseSalaryUsd": avg_opt(prows.iter().map(|r| opt_f64(r.get("base_salary_usd")))),
                    "signingBonusUsd": avg_opt(prows.iter().map(|r| opt_f64(r.get("signing_bonus_usd")))),
                    "minorLeagueSalaryUsd": avg_opt(prows.iter().map(|r| opt_f64(r.get("minor_league_salary_usd")))),
                    "twoWay": opt_bool(latest.get("two_way")).unwrap_or(false),
                    "isProjected": prows.iter().all(|r| opt_bool(r.get("is_projected")).unwrap_or(false)),
                    "termYears": term_years,
                    "startSeason": meta.and_then(|m| opt_nonempty_str(m.get("start_season"))),
                    "endSeason": meta.and_then(|m| opt_nonempty_str(m.get("end_season"))),
                    "expiryYear": end_year.map(|y| y + 1),
                    "contractStatus": meta.and_then(|m| opt_nonempty_str(m.get("contract_status"))),
                    "expiryStatus": meta.and_then(|m| opt_nonempty_str(m.get("expiry_status"))),
                    "totalValueUsd": meta.and_then(|m| opt_f64(m.get("total_value_usd"))),
                    "signedDate": meta.and_then(|m| opt_nonempty_str(m.get("signed_date"))),
                })
            })
            .collect()
    };

    roster.sort_by(|a, b| {
        let aa = a.get("aavUsd").and_then(Value::as_f64).unwrap_or(-1.0);
        let bb = b.get("aavUsd").and_then(Value::as_f64).unwrap_or(-1.0);
        bb.partial_cmp(&aa)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                str_value(a.get("playerName")).cmp(&str_value(b.get("playerName")))
            })
    });

    let mut out_sorted: Vec<&Value> = outlook_rows.iter().collect();
    out_sorted.sort_by_key(|r| safe_int(r.get("season_id")).unwrap_or(0));
    let outlook: Vec<Value> = out_sorted.into_iter().map(format_team_year_row).collect();

    json!({
        "team": team,
        "mode": if is_multi { "average" } else { "single" },
        "seasonsRequested": season_ids_requested,
        "seasonsMatched": ty_sorted.len(),
        "summary": summary,
        "roster": roster,
        "outlook": outlook,
    })
}

/// Loads team cap summary, roster contracts, and multi-year cap outlook.
pub async fn load_team_contracts(
    caches: &Caches,
    sb: Option<&SbClient>,
    team: &str,
    season_ids: &[i64],
) -> Arc<Value> {
    let team_up = team.trim().to_uppercase();
    let sids = norm_season_ids(season_ids);
    if team_up.is_empty() || sids.is_empty() {
        return Arc::new(json!({
            "team": team_up,
            "mode": "single",
            "seasonsRequested": sids,
            "seasonsMatched": 0,
            "summary": Value::Null,
            "roster": [],
            "outlook": [],
        }));
    }
    let Some(sb) = sb else {
        return Arc::new(json!({
            "team": team_up,
            "mode": if sids.len() > 1 { "average" } else { "single" },
            "seasonsRequested": sids,
            "seasonsMatched": 0,
            "summary": Value::Null,
            "roster": [],
            "outlook": [],
        }));
    };

    let cache_key = (team_up.clone(), sids.clone());
    if let Some(cached) = caches.team_contracts.get(&cache_key) {
        return cached;
    }

    let sid_in = sids
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let primary_sid = *sids.last().unwrap_or(&0);

    let team_eq = format!("eq.{team_up}");
    let sid_in_val = format!("in.({sid_in})");
    let sid_gte_val = format!("gte.{primary_sid}");

    let f_team_years = filters(&[("team", &team_eq), ("season_id", &sid_in_val)]);
    let f_roster = filters(&[("team", &team_eq), ("season_id", &sid_in_val)]);
    let f_outlook = filters(&[("team", &team_eq), ("season_id", &sid_gte_val)]);

    let (team_years_opt, roster_opt, outlook_opt) = tokio::join!(
        sb.read(
            "contract_teams_years",
            TEAM_YEARS_COLS,
            Some(&f_team_years),
            None,
            Some("season_id.asc"),
            sids.len().max(1),
        ),
        sb.read(
            "contract_years",
            CONTRACT_YEARS_COLS,
            Some(&f_roster),
            None,
            Some("aav_usd.desc"),
            0,
        ),
        sb.read(
            "contract_teams_years",
            TEAM_YEARS_COLS,
            Some(&f_outlook),
            None,
            Some("season_id.asc"),
            10,
        ),
    );

    let team_year_rows = team_years_opt.unwrap_or_default();
    let roster_rows = roster_opt.unwrap_or_default();
    let outlook_rows = outlook_opt.unwrap_or_default();

    let cids: Vec<String> = roster_rows
        .iter()
        .map(|r| str_value(r.get("contract_id")))
        .filter(|s| !s.is_empty())
        .collect();
    let meta = fetch_contracts_meta_by_ids(sb, &cids).await;

    let payload = Arc::new(build_team_contracts_payload(
        &team_up,
        &sids,
        &team_year_rows,
        &roster_rows,
        &meta,
        &outlook_rows,
    ));
    caches.team_contracts.insert(cache_key, payload.clone());
    payload
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_season_player_contract_summary() {
        let year_rows = vec![json!({
            "contract_id": "8484762_2024-25",
            "player_id": 8484762,
            "season_id": 20252026,
            "season": "2025-26",
            "team": "ANA",
            "aav_usd": 956250,
            "total_salary_usd": 975000,
            "base_salary_usd": 877500,
            "signing_bonus_usd": 97500,
            "minor_league_salary_usd": 85000,
            "two_way": true,
            "is_projected": false
        })];
        let mut meta = HashMap::new();
        meta.insert(
            "8484762_2024-25".to_string(),
            json!({
                "contract_id": "8484762_2024-25",
                "start_year": 2024,
                "end_year": 2026,
                "start_season": "2024-25",
                "end_season": "2026-27",
                "term_years": 3,
                "contract_status": "ELC",
                "expiry_status": "RFA",
                "aav_usd": 956250,
                "total_value_usd": 2868750,
                "two_way": true
            }),
        );

        let out = summarize_player_contract(&[20252026], &year_rows, &meta).unwrap();
        assert_eq!(out["mode"], "single");
        assert_eq!(out["aavUsd"], 956250.0);
        assert_eq!(out["totalSalaryUsd"], 975000.0);
        assert_eq!(out["termYears"], 3);
        assert_eq!(out["contractStatus"], "ELC");
        assert_eq!(out["expiryStatus"], "RFA");
        assert_eq!(out["expiryYear"], 2027);
        assert_eq!(out["twoWay"], true);
    }

    #[test]
    fn multi_season_player_contract_averages_over_matched_years() {
        let year_rows = vec![
            json!({
                "contract_id": "8477492_2016-17",
                "player_id": 8477492,
                "season_id": 20222023,
                "season": "2022-23",
                "team": "COL",
                "aav_usd": 6300000,
                "total_salary_usd": 6300000,
                "base_salary_usd": 6300000,
                "signing_bonus_usd": 0,
                "two_way": false,
                "is_projected": false
            }),
            json!({
                "contract_id": "8477492_2023-24",
                "player_id": 8477492,
                "season_id": 20232024,
                "season": "2023-24",
                "team": "COL",
                "aav_usd": 12602500,
                "total_salary_usd": 16500000,
                "base_salary_usd": 1000000,
                "signing_bonus_usd": 15500000,
                "two_way": false,
                "is_projected": false
            }),
        ];
        let out = summarize_player_contract(
            &[20212022, 20222023, 20232024],
            &year_rows,
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(out["mode"], "average");
        assert_eq!(out["seasonsRequested"], 3);
        assert_eq!(out["seasonsMatched"], 2);
        assert_eq!(out["aavUsd"], 9451250.0);
        assert_eq!(out["totalSalaryUsd"], 11400000.0);
        assert_eq!(out["contractsCount"], 2);
        assert_eq!(out["seasons"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn empty_rows_returns_none_without_fallback() {
        assert!(summarize_player_contract(&[20082009], &[], &HashMap::new()).is_none());
    }

    #[test]
    fn team_contracts_multi_season_averages_summary_and_roster() {
        let team_rows = vec![
            json!({
                "team": "ANA",
                "season_id": 20242025,
                "season": "2024-25",
                "contracts_count": 44,
                "one_way_contracts_count": 16,
                "two_way_contracts_count": 28,
                "total_aav_usd": 80000000,
                "one_way_aav_usd": 60000000,
                "total_cap_penalties_usd": 0,
                "combined_one_way_plus_penalties_usd": 60000000,
                "cap_ceiling_usd": 88000000,
                "projected_cap_space_usd": 28000000
            }),
            json!({
                "team": "ANA",
                "season_id": 20252026,
                "season": "2025-26",
                "contracts_count": 46,
                "one_way_contracts_count": 18,
                "two_way_contracts_count": 28,
                "total_aav_usd": 90000000,
                "one_way_aav_usd": 70000000,
                "total_cap_penalties_usd": 2000000,
                "combined_one_way_plus_penalties_usd": 72000000,
                "cap_ceiling_usd": 95500000,
                "projected_cap_space_usd": 23500000
            }),
        ];
        let roster_rows = vec![
            json!({
                "contract_id": "c1",
                "player_id": 100,
                "player_name": "Star Player",
                "position": "C",
                "season_id": 20242025,
                "season": "2024-25",
                "aav_usd": 6000000,
                "total_salary_usd": 6000000,
                "two_way": false
            }),
            json!({
                "contract_id": "c2",
                "player_id": 100,
                "player_name": "Star Player",
                "position": "C",
                "season_id": 20252026,
                "season": "2025-26",
                "aav_usd": 8000000,
                "total_salary_usd": 8000000,
                "two_way": false
            }),
            json!({
                "contract_id": "c3",
                "player_id": 200,
                "player_name": "Goalie One",
                "position": "G",
                "season_id": 20252026,
                "season": "2025-26",
                "aav_usd": 4000000,
                "total_salary_usd": 4000000,
                "two_way": false
            }),
        ];

        let out = build_team_contracts_payload(
            "ANA",
            &[20242025, 20252026],
            &team_rows,
            &roster_rows,
            &HashMap::new(),
            &[],
        );
        assert_eq!(out["mode"], "average");
        assert_eq!(out["seasonsMatched"], 2);
        assert_eq!(out["summary"]["combinedOneWayPlusPenaltiesUsd"], 66000000.0);
        assert_eq!(out["summary"]["projectedCapSpaceUsd"], 25750000.0);
        let roster = out["roster"].as_array().unwrap();
        assert_eq!(roster.len(), 2);
        assert_eq!(roster[0]["playerId"], 100);
        assert_eq!(roster[0]["aavUsd"], 7000000.0);
        assert_eq!(roster[0]["yearsCount"], 2);
        assert_eq!(roster[1]["posGroup"], "G");
    }
}
