//! Manager Game score model (M3): a simple, explainable fantasy score from
//! real NHL game stats, aggregated from the play-by-play pipeline.
//!
//! Weights (MANAGER_GAME_PLAN §3):
//!   skaters: Goal +4, A1 +3, A2 +2, SOG +0.5, PEND +0.5, PENT −0.5,
//!            5v5 on-ice xG differential (xG_F) +1.0 per xG
//!   goalies: GSAx (xGA − GA) +3.0 per GSAx
//!
//! A team's round score = Σ fantasy points of the lineup players, from the
//! real NHL game(s) in the round window.

use std::collections::BTreeMap;

use serde_json::Value;

use crate::util::parse::{parse_locale_float, safe_int, str_value};

// ── Weights ────────────────────────────────────────────────────────────
pub const W_GOAL: f64 = 4.0;
pub const W_A1: f64 = 3.0;
pub const W_A2: f64 = 2.0;
pub const W_SOG: f64 = 0.5;
pub const W_PEND: f64 = 0.5;
pub const W_PENT: f64 = -0.5;
pub const W_XGDIFF: f64 = 1.0;
pub const W_GSAX: f64 = 3.0;

/// The backup goalie (lineup slot G2) counts at 50%; every other lineup slot
/// (12F/6D/G1) counts at 100%.
pub const BACKUP_GOALIE_WEIGHT: f64 = 0.5;

/// Weight for a lineup slot label (100% except the G2 backup goalie).
pub fn slot_weight(slot: &str) -> f64 {
    if slot == "G2" {
        BACKUP_GOALIE_WEIGHT
    } else {
        1.0
    }
}

/// One player's stat line from a single NHL game.
#[derive(Clone, Default, Debug, PartialEq)]
pub struct StatLine {
    pub goals: i64,
    pub a1: i64,
    pub a2: i64,
    pub sog: i64,
    pub pent: i64,
    pub pend: i64,
    pub xgf_5v5: f64,
    pub xga_5v5: f64,
    pub gsax: f64,
}

impl StatLine {
    /// Fantasy points. Non-applicable fields are 0 for both skaters and
    /// goalies, so one formula covers both: skaters carry no GSAx; goalies
    /// carry no counting stats or on-ice xG split.
    pub fn fantasy_points(&self) -> f64 {
        self.goals as f64 * W_GOAL
            + self.a1 as f64 * W_A1
            + self.a2 as f64 * W_A2
            + self.sog as f64 * W_SOG
            + self.pend as f64 * W_PEND
            + self.pent as f64 * W_PENT
            + (self.xgf_5v5 - self.xga_5v5) * W_XGDIFF
            + self.gsax * W_GSAX
    }
}

fn split_ids(s: &str) -> Vec<i64> {
    s.split_whitespace()
        .filter_map(|tok| tok.parse::<i64>().ok())
        .collect()
}

/// Aggregate per-player stat lines from the wide PBP rows produced by
/// `build_plays` (xg_scope = "xG_F", lite_mode = false).
pub fn stat_lines_from_plays(plays: &[Value]) -> BTreeMap<i64, StatLine> {
    let mut m: BTreeMap<i64, StatLine> = BTreeMap::new();
    for p in plays {
        let type_code = safe_int(p.get("typeCode")).unwrap_or(0);
        let shot_flag = safe_int(p.get("Shot")).unwrap_or(0) == 1;
        let goal_flag = safe_int(p.get("Goal")).unwrap_or(0) == 1;
        let strength = str_value(p.get("StrengthState"));
        let xg = parse_locale_float(p.get("xG_F")).unwrap_or(0.0);

        // Goals → G/A1/A2 (Player1=scorer, Player2=assist1, Player3=assist2).
        if type_code == 505 || goal_flag {
            if let Some(pid) = safe_int(p.get("Player1_ID")) {
                m.entry(pid).or_default().goals += 1;
            }
            if let Some(pid) = safe_int(p.get("Player2_ID")) {
                m.entry(pid).or_default().a1 += 1;
            }
            if let Some(pid) = safe_int(p.get("Player3_ID")) {
                m.entry(pid).or_default().a2 += 1;
            }
        }
        // SOG → shooter (Player1 = shooting/scoring player).
        if shot_flag {
            if let Some(pid) = safe_int(p.get("Player1_ID")) {
                m.entry(pid).or_default().sog += 1;
            }
        }
        // Penalties: Player1 = committedBy, Player2 = drawnBy.
        if type_code == 509 {
            if let Some(pid) = safe_int(p.get("Player1_ID")) {
                m.entry(pid).or_default().pent += 1;
            }
            if let Some(pid) = safe_int(p.get("Player2_ID")) {
                m.entry(pid).or_default().pend += 1;
            }
        }
        // 5v5 on-ice xG± (xG_F): shooter's on-ice skaters get +xg (xGF),
        // opponent's on-ice skaters get +xg (xGA).
        if strength == "5v5" && xg > 0.0 && shot_flag {
            let venue = str_value(p.get("Venue"));
            let (sf, sd, of, od) = if venue == "Away" {
                (
                    str_value(p.get("Away_Forwards_ID")),
                    str_value(p.get("Away_Defenders_ID")),
                    str_value(p.get("Home_Forwards_ID")),
                    str_value(p.get("Home_Defenders_ID")),
                )
            } else {
                (
                    str_value(p.get("Home_Forwards_ID")),
                    str_value(p.get("Home_Defenders_ID")),
                    str_value(p.get("Away_Forwards_ID")),
                    str_value(p.get("Away_Defenders_ID")),
                )
            };
            for pid in split_ids(&sf).into_iter().chain(split_ids(&sd)) {
                m.entry(pid).or_default().xgf_5v5 += xg;
            }
            for pid in split_ids(&of).into_iter().chain(split_ids(&od)) {
                m.entry(pid).or_default().xga_5v5 += xg;
            }
        }
    }

    // GSAx: goalie-in-net faces every shot/goal; gsax = ΣxG_F(shots) − GA.
    let goalie_agg = goalie_gsax_from_plays(plays);
    for (pid, (xga, ga)) in goalie_agg {
        let e = m.entry(pid).or_default();
        e.gsax = xga - ga as f64;
    }
    m
}

/// (goalie_id → (xGA, GA)) from shot/goal events against `goalieInNetId`.
fn goalie_gsax_from_plays(plays: &[Value]) -> BTreeMap<i64, (f64, i64)> {
    let mut acc: BTreeMap<i64, (f64, i64)> = BTreeMap::new();
    for p in plays {
        let Some(gid) = safe_int(p.get("Goalie_ID")) else {
            continue;
        };
        let shot_flag = safe_int(p.get("Shot")).unwrap_or(0) == 1;
        let goal_flag = safe_int(p.get("Goal")).unwrap_or(0) == 1;
        let e = acc.entry(gid).or_insert((0.0, 0));
        if shot_flag {
            e.0 += parse_locale_float(p.get("xG_F")).unwrap_or(0.0);
        }
        if goal_flag {
            e.1 += 1;
        }
    }
    acc
}

/// A team's round score: Σ (fantasy points × slot weight) of its lineup. The
/// lineup is `(slot_label, player_id)`; the G2 backup goalie weights at 50%.
pub fn team_round_score(lineup: &[(String, i64)], stats: &BTreeMap<i64, StatLine>) -> f64 {
    lineup
        .iter()
        .map(|(slot, pid)| {
            stats
                .get(pid)
                .map(|s| s.fantasy_points() * slot_weight(slot))
                .unwrap_or(0.0)
        })
        .sum()
}

/// Extract the player ids from a slot-aware lineup (for tie-breaking on raw
/// counting stats, which are unaffected by the goalie weight).
fn lineup_pids(lineup: &[(String, i64)]) -> Vec<i64> {
    lineup.iter().map(|(_, pid)| *pid).collect()
}

/// Tie-break ladder (W/L only, no OT): 1) total goals, 2) total A1,
/// 3) fewer PENT, 4) combined GSAx, 5) deterministic home win.
pub fn decide_winner(
    home_score: f64,
    away_score: f64,
    home_lineup: &[(String, i64)],
    away_lineup: &[(String, i64)],
    stats: &BTreeMap<i64, StatLine>,
) -> &'static str {
    if home_score != away_score {
        return if home_score > away_score { "home" } else { "away" };
    }
    let hpids = lineup_pids(home_lineup);
    let apids = lineup_pids(away_lineup);
    let sum = |l: &[i64], f: &dyn Fn(&StatLine) -> f64| -> f64 {
        l.iter()
            .filter_map(|pid| stats.get(pid))
            .map(|s| f(s))
            .sum()
    };
    let hg = sum(&hpids, &|s| s.goals as f64);
    let ag = sum(&apids, &|s| s.goals as f64);
    if hg != ag {
        return if hg > ag { "home" } else { "away" };
    }
    let ha = sum(&hpids, &|s| s.a1 as f64);
    let aa = sum(&apids, &|s| s.a1 as f64);
    if ha != aa {
        return if ha > aa { "home" } else { "away" };
    }
    let hp = sum(&hpids, &|s| s.pent as f64);
    let ap = sum(&apids, &|s| s.pent as f64);
    if hp != ap {
        return if hp < ap { "home" } else { "away" };
    }
    let hx = sum(&hpids, &|s| s.gsax);
    let ax = sum(&apids, &|s| s.gsax);
    if hx != ax {
        return if hx > ax { "home" } else { "away" };
    }
    "home"
}

/// Fantasy points JSON for a stat line (for storage / display).
pub fn stat_line_to_value(pid: i64, line: &StatLine) -> Value {
    serde_json::json!({
        "player_id": pid,
        "goals": line.goals,
        "a1": line.a1,
        "a2": line.a2,
        "sog": line.sog,
        "pent": line.pent,
        "pend": line.pend,
        "xgf_5v5": round4(line.xgf_5v5),
        "xga_5v5": round4(line.xga_5v5),
        "gsax": round4(line.gsax),
        "fantasy_points": round2(line.fantasy_points()),
    })
}

fn round2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}
fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

#[cfg(test)]
mod tests {
    use super::*;

    // A shot/goal row at 5v5, home on-ice 10,11,12; away on-ice 20,21,22;
    // goalie in net = 30.
    fn row(type_code: i64, p1: Option<i64>, p2: Option<i64>, p3: Option<i64>) -> Value {
        serde_json::json!({
            "typeCode": type_code,
            "Shot": if type_code == 505 || type_code == 506 { 1 } else { 0 },
            "Goal": if type_code == 505 { 1 } else { 0 },
            "Player1_ID": p1,
            "Player2_ID": p2,
            "Player3_ID": p3,
            "StrengthState": "5v5",
            "xG_F": 0.5,
            "Venue": "Home",
            "Home_Forwards_ID": "10 11",
            "Home_Defenders_ID": "12",
            "Away_Forwards_ID": "20 21",
            "Away_Defenders_ID": "22",
            "Goalie_ID": 30,
        })
    }

    #[test]
    fn fantasy_points_match_weights() {
        let line = StatLine {
            goals: 1, a1: 1, a2: 0, sog: 4, pent: 0, pend: 1,
            xgf_5v5: 1.5, xga_5v5: 1.0,
            gsax: 0.0,
        };
        // 1*4 + 1*3 + 4*0.5 + 1*0.5 + (1.5-1.0)*1
        let expected = 4.0 + 3.0 + 2.0 + 0.5 + 0.5;
        assert!((line.fantasy_points() - expected).abs() < 1e-9);
    }

    #[test]
    fn goalie_points_only_from_gsax() {
        let line = StatLine { gsax: 2.0, ..Default::default() };
        assert!((line.fantasy_points() - 6.0).abs() < 1e-9);
    }

    #[test]
    fn stat_lines_aggregate_goal_events() {
        let plays = vec![row(505, Some(10), Some(11), Some(12)), row(506, Some(10), None, None)];
        let m = stat_lines_from_plays(&plays);
        assert_eq!(m[&10].goals, 1);
        assert_eq!(m[&10].sog, 2);
        assert_eq!(m[&11].a1, 1);
        assert_eq!(m[&12].a2, 1);
    }

    #[test]
    fn stat_lines_penalty_actors() {
        let plays = vec![serde_json::json!({
            "typeCode": 509, "Shot": 0, "Goal": 0,
            "Player1_ID": 10, "Player2_ID": 11, "Player3_ID": null,
            "StrengthState": "5v5", "xG_F": 0.0, "Venue": "Home",
            "Home_Forwards_ID": "", "Home_Defenders_ID": "", "Away_Forwards_ID": "", "Away_Defenders_ID": "",
            "Goalie_ID": null,
        })];
        let m = stat_lines_from_plays(&plays);
        assert_eq!(m[&10].pent, 1);
        assert_eq!(m[&11].pend, 1);
    }

    #[test]
    fn five_on_five_on_ice_xg_split() {
        let plays = vec![row(506, Some(10), None, None)];
        let m = stat_lines_from_plays(&plays);
        for pid in [10, 11, 12] {
            assert!((m[&pid].xgf_5v5 - 0.5).abs() < 1e-9, "home P{pid} xgf");
        }
        for pid in [20, 21, 22] {
            assert!((m[&pid].xga_5v5 - 0.5).abs() < 1e-9, "away P{pid} xga");
        }
    }

    #[test]
    fn goalie_gsax_is_xga_minus_ga() {
        let goal = row(505, Some(10), None, None);
        let save = row(506, Some(11), None, None);
        let plays = vec![goal, save];
        let m = stat_lines_from_plays(&plays);
        // goalie 30: xGA = 0.5 + 0.3? No — save row xG_F is 0.5, so 0.5+0.5=1.0; GA=1 → gsax 0.0.
        assert!((m[&30].gsax - 0.0).abs() < 1e-9);
    }

    #[test]
    fn team_round_score_sums_weighted_lineup() {
        let mut stats = BTreeMap::new();
        stats.insert(1, StatLine { goals: 1, ..Default::default() });   // 4 pts
        stats.insert(2, StatLine { sog: 2, ..Default::default() });      // 1 pt
        stats.insert(3, StatLine { gsax: 2.0, ..Default::default() });   // 6 pts
        let lineup = vec![
            ("F1".to_string(), 1),
            ("F2".to_string(), 2),
            ("G2".to_string(), 3), // backup goalie → 50% = 3 pts
        ];
        let score = team_round_score(&lineup, &stats);
        // 4 + 1 + 3 = 8.
        assert!((score - 8.0).abs() < 1e-9);
    }

    #[test]
    fn backup_goalie_counts_at_half() {
        assert!((slot_weight("F1") - 1.0).abs() < 1e-9);
        assert!((slot_weight("D6") - 1.0).abs() < 1e-9);
        assert!((slot_weight("G1") - 1.0).abs() < 1e-9);
        assert!((slot_weight("G2") - BACKUP_GOALIE_WEIGHT).abs() < 1e-9);
    }

    #[test]
    fn winner_uses_score_then_tiebreak() {
        let mut stats = BTreeMap::new();
        stats.insert(1, StatLine { goals: 2, a1: 1, ..Default::default() });
        stats.insert(2, StatLine { goals: 1, a1: 2, ..Default::default() });
        let home_lineup = vec![("F1".to_string(), 1)];
        let away_lineup = vec![("F1".to_string(), 2)];
        assert_eq!(decide_winner(11.0, 10.0, &home_lineup, &away_lineup, &stats), "home");
        assert_eq!(decide_winner(10.0, 10.0, &home_lineup, &away_lineup, &stats), "home");
    }
}
