//! Manager Game — a league-based game mode built on the existing NHL data
//! stack (schedule fetch, rosters, projections, PBP/xG pipeline).
//!
//! M1 delivers leagues, franchise slots (CPU = unclaimed), the schedule
//! snapshot with per-team round indices, and the first APIs. Later milestones
//! add the draft, scoring, trades, and lineups.

pub mod draft;
pub mod dry_run;
pub mod games;
pub mod league;
pub mod schedule;
pub mod scoring;
pub mod standings;
pub mod trades;

/// Target season for new leagues: 2026/27 (the 84-game season).
pub const DEFAULT_SEASON: i64 = 20262027;

/// League lifecycle.
pub const STATUS_SETUP: &str = "setup";
pub const STATUS_DRAFTING: &str = "drafting";
pub const STATUS_RUNNING: &str = "running";
pub const STATUS_FINISHED: &str = "finished";

/// League game modes.
pub const MODE_TRUE_ROSTERS: &str = "true_rosters";
pub const MODE_DRAFT: &str = "draft";

/// Lineup shape (spec): 12 forwards, 6 defensemen, 2 goalies.
pub const LINEUP_FORWARDS: usize = 12;
pub const LINEUP_DEFENSE: usize = 6;
pub const LINEUP_GOALIES: usize = 2;

/// Active lineup slots (the 12/6/2 fielded each round).
pub const LINEUP_SLOTS: usize = LINEUP_FORWARDS + LINEUP_DEFENSE + LINEUP_GOALIES;

/// Draft roster size: 26 players, with a **minimum** of 12F/6D/2G (the extra
/// 6 are bench depth, which makes trades easier). The active lineup each round
/// is still the 12F/6D/2G from `LINEUP_SLOTS`, chosen from these 26.
pub const DRAFT_TOTAL: usize = 26;

/// Lineup slot labels F1..F12, D1..D6, G1, G2 (matches the DB check).
pub fn lineup_slots() -> Vec<String> {
    let mut out = Vec::with_capacity(LINEUP_SLOTS);
    for i in 1..=LINEUP_FORWARDS {
        out.push(format!("F{i}"));
    }
    for i in 1..=LINEUP_DEFENSE {
        out.push(format!("D{i}"));
    }
    for i in 1..=LINEUP_GOALIES {
        out.push(format!("G{i}"));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lineup_slots_match_spec_shape() {
        let slots = lineup_slots();
        assert_eq!(slots.len(), 20);
        assert_eq!(slots[0], "F1");
        assert_eq!(slots[11], "F12");
        assert_eq!(slots[12], "D1");
        assert_eq!(slots[17], "D6");
        assert_eq!(slots[18], "G1");
        assert_eq!(slots[19], "G2");
    }
}
