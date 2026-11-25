//! Pure-Rust XGBoost tree scorer + model loading.
//!
//! Models are exported by `scripts/export_models_for_rust.py` into
//! `Model/rust/{fname}.json` using the canonical `Booster.save_raw` JSON, with
//! a calibrated `base_score`. The tree walk below is structurally identical to
//! XGBoost's predictor (verified against `predict_proba`, |Δ| < 2e-7).

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};

use serde::Deserialize;
use serde_json::Value;

use crate::state::Caches;
use crate::util::parse::parse_locale_float;

#[derive(Clone, Deserialize)]
pub struct Tree {
    pub split_indices: Vec<i64>,
    pub split_conditions: Vec<f64>,
    pub left: Vec<i64>,
    pub right: Vec<i64>,
    #[serde(default)]
    pub default_left: Vec<i64>,
    pub leaf: Vec<f64>,
}

impl Tree {
    fn score(&self, row: &[f64]) -> f64 {
        let mut i: usize = 0;
        loop {
            let lc = self.left[i];
            if lc == -1 {
                return self.leaf[i];
            }
            let f = self.split_indices[i];
            if f < 0 || f as usize >= row.len() {
                // Feature unavailable — treat as missing.
                i = if self.default_left.get(i).copied().unwrap_or(0) == 1 {
                    lc
                } else {
                    self.right[i]
                } as usize;
                continue;
            }
            let v = row[f as usize];
            let next = if v.is_nan() {
                if self.default_left.get(i).copied().unwrap_or(0) == 1 {
                    lc
                } else {
                    self.right[i]
                }
            } else if v < self.split_conditions[i] {
                lc
            } else {
                self.right[i]
            };
            i = next as usize;
        }
    }
}

#[derive(Clone, Deserialize)]
pub struct XgModel {
    pub features: Vec<String>,
    pub base_score: f64,
    #[serde(default)]
    #[allow(dead_code)]
    pub objective: String,
    pub trees: Vec<Tree>,
    /// Pre-indexed map: `(base_col, { suffix -> feature_idx })` built once at
    /// load time so `vectorize_row` does ~7 lookups instead of O(F) allocations.
    #[serde(skip)]
    base_index: Vec<(String, HashMap<String, usize>)>,
}

impl XgModel {
    fn build_index(&mut self) {
        let mut grouped: HashMap<String, HashMap<String, usize>> = HashMap::new();
        for (i, cname) in self.features.iter().enumerate() {
            let Some((base, suffix)) = cname.split_once('_') else {
                continue;
            };
            grouped
                .entry(base.to_string())
                .or_default()
                .insert(suffix.to_string(), i);
        }
        self.base_index = grouped.into_iter().collect();
    }

    /// Raw margin: base_score + sum of tree leaves.
    pub fn score_row(&self, row: &[f64]) -> f64 {
        let mut margin = self.base_score;
        for t in &self.trees {
            margin += t.score(row);
        }
        margin
    }

    /// binary:logistic probability.
    pub fn predict(&self, row: &[f64]) -> f64 {
        let m = self.score_row(row);
        1.0 / (1.0 + (-m).exp())
    }
}

/// `load_model_file(fname)`: bounded-cache loader for `Model/rust/{fname}.json`.
pub fn load_model_file(caches: &Caches, fname: &str, model_dir: &Path) -> Option<Arc<XgModel>> {
    let key = fname.to_string();
    if let Some(m) = caches.model.get(&key) {
        return Some(m);
    }
    let path = model_dir.join(format!("{fname}.json"));
    let data = std::fs::read(&path).ok()?;
    let mut model: XgModel = serde_json::from_slice(&data).ok()?;
    model.build_index();
    let arc = Arc::new(model);
    caches.model.insert(key, arc.clone());
    Some(arc)
}

/// The 7 base one-hot columns (`base_feature_cols` in Flask).
#[allow(dead_code)]
pub const BASE_FEATURE_COLS: [&str; 7] = [
    "Venue",
    "shotType2",
    "ScoreState2",
    "RinkVenue",
    "StrengthState2",
    "BoxID2",
    "LastEvent",
];

/// `_vectorize_row_for_model`: build the one-hot vector aligned to `features`.
/// Uses the precomputed `base_index` for ~7 lookups (zero String allocations
/// for string feature values) instead of scanning all O(F) columns.
pub fn vectorize_row(model: &XgModel, row_obj: &serde_json::Map<String, Value>) -> Vec<f64> {
    let mut vec = vec![0.0; model.features.len()];
    if !model.base_index.is_empty() {
        for (base, suffix_map) in &model.base_index {
            let idx = match row_obj.get(base.as_str()) {
                None => suffix_map.get("missing").copied(),
                Some(Value::Null) => suffix_map.get("missing").copied(),
                Some(Value::String(s)) => suffix_map.get(s.as_str()).copied(),
                Some(v) => {
                    let s = parse_locale_float(Some(v))
                        .map(|f| format!("{f}"))
                        .unwrap_or_else(|| v.to_string());
                    suffix_map.get(s.as_str()).copied()
                }
            };
            if let Some(i) = idx {
                vec[i] = 1.0;
            }
        }
        return vec;
    }
    for (i, cname) in model.features.iter().enumerate() {
        let Some((base, suffix)) = cname.split_once('_') else { continue };
        let matched = match row_obj.get(base) {
            None | Some(Value::Null) => suffix == "missing",
            Some(Value::String(s)) => s == suffix,
            Some(v) => {
                let s = parse_locale_float(Some(v))
                    .map(|f| format!("{f}"))
                    .unwrap_or_else(|| v.to_string());
                s == suffix
            }
        };
        if matched {
            vec[i] = 1.0;
        }
    }
    vec
}

fn season_prev(s: i64) -> i64 {
    let a = (s / 10000) - 1;
    let b = (s % 10000) - 1;
    a * 10000 + b
}

fn season_next(s: i64) -> i64 {
    let a = (s / 10000) + 1;
    let b = (s % 10000) + 1;
    a * 10000 + b
}

pub fn num_windows() -> usize {
    static WINDOWS: OnceLock<usize> = OnceLock::new();
    *WINDOWS.get_or_init(|| {
        std::env::var("XG_WINDOWS")
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .map(|w| w.clamp(1, 3))
            .unwrap_or(1)
    })
}

/// Newest trained `{prefix}_<start>_<end>.pkl` in `model_dir`, by window end year.
/// Cached per `(prefix, model_dir)` so `std::fs::read_dir` is not repeated per game/shot.
fn latest_model_file(prefix: &str, model_dir: &Path) -> Option<String> {
    static CACHE: OnceLock<Mutex<HashMap<(String, PathBuf), Option<String>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = (prefix.to_string(), model_dir.to_path_buf());
    if let Ok(guard) = cache.lock() {
        if let Some(hit) = guard.get(&key) {
            return hit.clone();
        }
    }
    let mut best: Option<(i64, String)> = None;
    if let Ok(rd) = std::fs::read_dir(model_dir) {
        for entry in rd.flatten() {
            let fname = entry.file_name().to_string_lossy().to_string();
            let Some(stem) = fname.strip_suffix(".json") else { continue };
            let Some(rest) = stem.strip_prefix(prefix) else { continue };
            let Some(rest) = rest.strip_prefix('_') else { continue };
            let Some(rest) = rest.strip_suffix(".pkl") else { continue };
            let mut parts = rest.split('_');
            let (Some(_start), Some(end), None) = (parts.next(), parts.next(), parts.next()) else {
                continue;
            };
            let Ok(end_year) = end.parse::<i64>() else { continue };
            if best.as_ref().map(|(y, _)| end_year > *y).unwrap_or(true) {
                best = Some((end_year, stem.to_string()));
            }
        }
    }
    let out = best.map(|(_, name)| name);
    if out.is_some() {
        if let Ok(mut guard) = cache.lock() {
            guard.insert(key, out.clone());
        }
    }
    out
}

/// `window_filenames_for_season` / the `predict_avg_for_row` name selection.
/// Returns candidate filenames in preferred order [middle, prev, next].
///
/// The trained windows slide (s-2..s, s-1..s+1, s..s+2). A season with no
/// trained window of its own — 20262027 at the time of writing — would resolve
/// to no model at all, which left every shot's xG at `null` and rendered the
/// game report's xG card as 0.00. The newest available window for the family is
/// appended so a brand-new season still scores.
pub fn window_filenames_for_season(s_cur: i64, prefix: &str, model_dir: &Path) -> Vec<String> {
    let s_prev = season_prev(s_cur);
    let s_next = season_next(s_cur);
    let s_prev2 = season_prev(s_prev);
    let s_next2 = season_next(s_next);
    let all = vec![
        format!("{prefix}_{s_prev2}_{s_cur}.pkl"),
        format!("{prefix}_{s_prev}_{s_next}.pkl"),
        format!("{prefix}_{s_cur}_{s_next2}.pkl"),
    ];
    let mut names = vec![all[1].clone(), all[0].clone(), all[2].clone()];
    if let Some(fallback) = latest_model_file(prefix, model_dir) {
        if !names.contains(&fallback) {
            names.push(fallback);
        }
    }
    names
}

/// `predict_avg_for_row`: average prediction across loaded windows.
pub fn predict_avg_for_row(
    caches: &Caches,
    model_dir: &Path,
    row_obj: &serde_json::Map<String, Value>,
    season_val: Option<i64>,
    model_prefix: &str,
) -> Option<f64> {
    let s_cur = season_val?;
    let names = window_filenames_for_season(s_cur, model_prefix, model_dir);
    let n_windows = num_windows();
    let mut preds: Vec<f64> = Vec::new();
    for n in names {
        let Some(model) = load_model_file(caches, &n, model_dir) else { continue };
        let vec = vectorize_row(&model, row_obj);
        preds.push(model.predict(&vec));
        if preds.len() >= n_windows {
            break;
        }
    }
    if preds.is_empty() {
        return None;
    }
    Some(preds.iter().sum::<f64>() / preds.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn season_window_math() {
        assert_eq!(season_prev(20242025), 20232024);
        assert_eq!(season_next(20242025), 20252026);
        // An empty model dir means no fallback candidate is appended.
        let empty = std::path::Path::new("/nonexistent-model-dir");
        let names = window_filenames_for_season(20242025, "xgb", empty);
        assert_eq!(
            names,
            vec![
                "xgb_20232024_20252026.pkl",
                "xgb_20222023_20242025.pkl",
                "xgb_20242025_20262027.pkl"
            ]
        );
        let special = window_filenames_for_season(20252026, "xgbs", empty);
        assert_eq!(special.len(), 3);
        assert_eq!(special[0], "xgbs_20242025_20262027.pkl");
    }

    /// A season whose sliding windows were never trained (20262027) must still
    /// resolve to the newest available model instead of scoring nothing.
    #[test]
    fn untrained_season_falls_back_to_newest_model() {
        let dir = std::env::temp_dir().join(format!("xg-models-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for name in [
            "xgb_20212022_20232024.pkl.json",
            "xgb_20222023_20242025.pkl.json",
            "xgb2_20222023_20242025.pkl.json",
            "xgbs_20222023_20242025.pkl.json",
        ] {
            std::fs::write(dir.join(name), b"{}").unwrap();
        }

        let names = window_filenames_for_season(20262027, "xgb", &dir);
        assert_eq!(
            names,
            vec![
                "xgb_20252026_20272028.pkl",
                "xgb_20242025_20262027.pkl",
                "xgb_20262027_20282029.pkl",
                "xgb_20222023_20242025.pkl",
            ]
        );
        // Prefix matching must not confuse xgb with xgb2.
        let names2 = window_filenames_for_season(20262027, "xgb2", &dir);
        assert_eq!(names2.last().unwrap(), "xgb2_20222023_20242025.pkl");

        std::fs::remove_dir_all(&dir).ok();
    }
}
