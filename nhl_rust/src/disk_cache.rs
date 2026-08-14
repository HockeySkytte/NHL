//! Small on-disk cache utilities (persist across restarts).
// `allow(dead_code)`: consumers arrive in M3 (PBP/shifts caches).
#![allow(dead_code)]
//!
//! Port of `app/routes.py` `_disk_cache_base` / `_disk_cache_path_pbp` /
//! `_disk_cache_path_shifts`. The PBP/shifts caches are plain JSON with
//! `_cachedAt`/`gameState` metadata and are format-compatible with the
//! Python app (both apps can safely share the cache dir).

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::Value;

/// Game states that get the short "live" TTL (5s in Flask).
pub const LIVE_STATES: [&str; 4] = ["LIVE", "SCHEDULED", "PREVIEW", "INPROGRESS"];

/// `_disk_cache_base()`: `XG_CACHE_DIR`, else `%TEMP%/nhl_cache` or `/tmp/nhl_cache`.
pub fn disk_cache_base(xg_cache_dir: Option<&Path>) -> PathBuf {
    if let Some(d) = xg_cache_dir {
        if !d.as_os_str().is_empty() {
            return d.to_path_buf();
        }
    }
    std::env::temp_dir().join("nhl_cache")
}

/// `_disk_cache_path_pbp(game_id, xg_scope)`: scope suffix is lowercased,
/// skipped for `all`/`full`, and sanitized to `[a-z0-9_]`.
pub fn pbp_path(base: &Path, game_id: i64, xg_scope: Option<&str>) -> PathBuf {
    let scope = xg_scope.unwrap_or("").trim().to_lowercase();
    let mut suffix = String::new();
    if !scope.is_empty() && scope != "all" && scope != "full" {
        suffix.push('_');
        for c in scope.chars() {
            suffix.push(if c.is_ascii_alphanumeric() || c == '_' {
                c
            } else {
                '_'
            });
        }
    }
    base.join(format!("pbp_{game_id}{suffix}.json"))
}

/// `_disk_cache_path_shifts(game_id)`.
pub fn shifts_path(base: &Path, game_id: i64) -> PathBuf {
    base.join(format!("shifts_{game_id}.json"))
}

/// Admin job persistence dir: `<XG_CACHE_DIR or temp>/nhl_admin_jobs`.
pub fn jobs_dir(base: &Path) -> PathBuf {
    base.join("nhl_admin_jobs")
}

fn now_epoch() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Reads a JSON cache file written with `_cachedAt`/`gameState` metadata and
/// returns the payload only if it is still fresh. Live game states use
/// `live_ttl_s`, everything else uses `std_ttl_s`.
pub fn read_json_if_fresh(path: &Path, live_ttl_s: f64, std_ttl_s: f64) -> Option<Value> {
    let raw = std::fs::read_to_string(path).ok()?;
    let parsed: Value = serde_json::from_str(&raw).ok()?;
    let cached_at = parsed.get("_cachedAt").and_then(Value::as_f64).unwrap_or(0.0);
    let game_state = parsed.get("gameState").and_then(Value::as_str).unwrap_or("");
    let ttl = if LIVE_STATES.contains(&game_state) {
        live_ttl_s
    } else {
        std_ttl_s
    };
    if now_epoch() - cached_at <= ttl {
        Some(parsed)
    } else {
        None
    }
}

/// Writes JSON to the cache file, creating parent directories.
pub fn write_json(path: &Path, payload: &Value) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_vec(payload)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_temp_dir(name: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("nhl_rust_{name}_{}_{nanos}", std::process::id()))
    }

    #[test]
    fn pbp_path_scopes() {
        let base = Path::new("/tmp/cache");
        assert_eq!(
            pbp_path(base, 2024010123, None),
            PathBuf::from("/tmp/cache/pbp_2024010123.json")
        );
        assert_eq!(
            pbp_path(base, 2024010123, Some("xG_F_lite")),
            PathBuf::from("/tmp/cache/pbp_2024010123_xg_f_lite.json")
        );
        assert_eq!(
            pbp_path(base, 2024010123, Some("all")),
            PathBuf::from("/tmp/cache/pbp_2024010123.json")
        );
    }

    #[test]
    fn read_json_if_fresh_ttl_logic() {
        let dir = unique_temp_dir("disk");
        let path = dir.join("pbp_1.json");

        // Fresh, non-live entry within std TTL.
        write_json(
            &path,
            &serde_json::json!({"_cachedAt": now_epoch(), "gameState": "OFF", "plays": []}),
        )
        .unwrap();
        assert!(read_json_if_fresh(&path, 5.0, 10.0).is_some());

        // Stale non-live entry.
        write_json(
            &path,
            &serde_json::json!({"_cachedAt": now_epoch() - 100.0, "gameState": "OFF", "plays": []}),
        )
        .unwrap();
        assert!(read_json_if_fresh(&path, 5.0, 10.0).is_none());

        // Live entry: fresh under live TTL, stale under it even within std TTL.
        write_json(
            &path,
            &serde_json::json!({"_cachedAt": now_epoch() - 7.0, "gameState": "LIVE", "plays": []}),
        )
        .unwrap();
        assert!(read_json_if_fresh(&path, 5.0, 60.0).is_none());
        write_json(
            &path,
            &serde_json::json!({"_cachedAt": now_epoch() - 1.0, "gameState": "LIVE", "plays": []}),
        )
        .unwrap();
        assert!(read_json_if_fresh(&path, 5.0, 60.0).is_some());

        // Missing file.
        assert!(read_json_if_fresh(&dir.join("nope.json"), 5.0, 60.0).is_none());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
