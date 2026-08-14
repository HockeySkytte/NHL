//! Date/time helpers ported from `app/routes.py`.
// `allow(dead_code)`: consumed from M1+ (PBP, standings, prestart logger).
#![allow(dead_code)]

use chrono::{DateTime, Datelike, TimeZone, Utc};
use chrono_tz::America::New_York;

pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Current datetime with the America/New_York offset ("ET now").
pub fn et_now() -> DateTime<chrono::FixedOffset> {
    Utc::now().with_timezone(&New_York).fixed_offset()
}

/// NHL season id from a UTC date (September boundary).
/// Port of `current_season_id()`.
pub fn current_season_id(now: Option<DateTime<Utc>>) -> i64 {
    let d = now.unwrap_or_else(utc_now);
    let y = d.year();
    let (start_y, end_y) = if d.month() >= 9 { (y, y + 1) } else { (y - 1, y) };
    (i64::from(start_y) * 10000) + i64::from(end_y)
}

/// Lenient ISO-8601-ish parser (port of `_parse_iso_datetime`).
/// Accepts RFC 3339, `YYYY-MM-DD HH:MM:SS`, `YYYY-MM-DDTHH:MM:SS`.
pub fn parse_iso_utc(s: &str) -> Option<DateTime<Utc>> {
    let s = s.trim();
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    let naive = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S")
        .or_else(|_| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S"))
        .ok()?;
    Some(Utc.from_utc_datetime(&naive))
}

/// Port of `_isoformat_utc`: `2026-08-13T12:34:56.789012Z`.
pub fn isoformat_utc(dt: DateTime<Utc>) -> String {
    format!(
        "{}.{:06}Z",
        dt.format("%Y-%m-%dT%H:%M:%S"),
        dt.timestamp_subsec_micros()
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn season_id_boundaries() {
        // September 1 starts the new season.
        let sep1 = Utc.with_ymd_and_hms(2026, 9, 1, 0, 0, 0).unwrap();
        assert_eq!(current_season_id(Some(sep1)), 20262027);
        // August 31 is still the previous season.
        let aug31 = Utc.with_ymd_and_hms(2026, 8, 31, 23, 59, 59).unwrap();
        assert_eq!(current_season_id(Some(aug31)), 20252026);
        // January falls in the season that started the prior September.
        let jan1 = Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap();
        assert_eq!(current_season_id(Some(jan1)), 20252026);
    }

    #[test]
    fn parse_and_format_iso() {
        let dt = parse_iso_utc("2026-08-13T12:34:56Z").unwrap();
        assert_eq!(dt.timestamp(), 1786624496);
        let dt = parse_iso_utc("2026-08-13 12:34:56").unwrap();
        assert_eq!(dt.year(), 2026);
        assert!(parse_iso_utc("not-a-date").is_none());
        let out = isoformat_utc(dt);
        assert!(out.ends_with('Z'));
    }
}
