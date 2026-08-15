//! RAPM / context loaders — ports of `_load_rapm_static_csv`,
//! `_load_context_static_csv`, `_synthesize_rates_rows` and `_sb_read_rapm`.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::state::Caches;

use crate::supabase::read::SbClient;
use crate::util::parse::parse_locale_float;

/// RAPM rows shared across requests via `Arc` — a cache hit is a cheap
/// pointer clone, never a deep copy of ~21k serde Values. `totals` are the raw
/// Supabase rows; `rates` are the per-minute rows synthesized from totals +
/// context minutes (built once per TTL window, also shared).
pub struct RapmRows {
    pub totals: Arc<Vec<Value>>,
    pub rates: Arc<Vec<Value>>,
}

impl RapmRows {
    pub fn iter(&self) -> impl Iterator<Item = &Value> {
        self.totals.iter().chain(self.rates.iter())
    }
}

pub fn rapm_col_map() -> HashMap<String, String> {
    [
        ("player_id", "PlayerID"),
        ("season", "Season"),
        ("strength_state", "StrengthState"),
        ("rates_totals", "Rates_Totals"),
        ("cf", "CF"),
        ("ca", "CA"),
        ("gf", "GF"),
        ("ga", "GA"),
        ("xgf", "xGF"),
        ("xga", "xGA"),
        ("pen_taken", "PEN_taken"),
        ("pen_drawn", "PEN_drawn"),
        ("c_plusminus", "C_plusminus"),
        ("g_plusminus", "G_plusminus"),
        ("xg_plusminus", "xG_plusminus"),
        ("pen_plusminus", "PEN_plusminus"),
        ("alpha_cf", "Alpha_CF"),
        ("alpha_gf", "Alpha_GF"),
        ("alpha_xgf", "Alpha_xGF"),
        ("alpha_pen", "Alpha_PEN"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

pub fn context_col_map() -> HashMap<String, String> {
    [
        ("player_id", "PlayerID"),
        ("season", "Season"),
        ("strength_state", "StrengthState"),
        ("minutes", "Minutes"),
        ("qot_blend_xg67_g33", "QoT_blend_xG67_G33"),
        ("qoc_blend_xg67_g33", "QoC_blend_xG67_G33"),
        ("zs_difficulty", "ZS_Difficulty"),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v.to_string()))
    .collect()
}

/// `_sb_read_rapm`: read `rapm` for ONE season and unpack the JSONB
/// `zscore`/`pp_sh` columns back to flat rows with original column names.
///
/// The `stddev` JSONB object is intentionally dropped — no consumer reads
/// `*_stddev` fields (Flask keeps them only because `dict(r)` copies them;
/// no endpoint projects them). This shaves ~30% off the row blob.
///
/// Data is read from Supabase ONLY (the static `rapm/rapm.csv` export is
/// outdated and intentionally never used). Per-season reads keep the payload
/// at ~1.4k rows instead of the full ~21k-row table.
async fn read_rapm_supabase(sb: &SbClient, season: i64) -> Option<Vec<Value>> {
    // Targeted columns (no `*`) keep the pagination fast; the JSONB columns
    // carry the z-scores/standard deviations unpacked below.
    const COLS: &str = "player_id,season,strength_state,rates_totals,cf,ca,gf,ga,xgf,xga,pen_taken,pen_drawn,c_plusminus,g_plusminus,xg_plusminus,pen_plusminus,stddev,zscore,pp_sh";
    let mut f = std::collections::BTreeMap::new();
    f.insert("season".to_string(), format!("eq.{season}"));
    let raw = sb.read("rapm", COLS, Some(&f), None, None, 0).await?;
    let cm = rapm_col_map();
    let mut out = Vec::new();
    for row in raw {
        let Some(obj) = row.as_object() else { continue };
        let mut r = serde_json::Map::new();
        for (k, v) in obj {
            if k == "zscore" || k == "pp_sh" {
                if let Some(inner) = v.as_object() {
                    for (ik, iv) in inner {
                        r.insert(ik.clone(), iv.clone());
                    }
                }
            } else if k == "stddev" {
                // dropped: never read by any consumer
                continue;
            } else {
                let key = cm.get(k).cloned().unwrap_or_else(|| k.clone());
                r.insert(key, v.clone());
            }
        }
        out.push(Value::Object(r));
    }
    Some(out)
}

/// `_load_context_static_csv` (Supabase-only, per-season TTL cached,
/// singleflight). Returns an `Arc` so concurrent requests share one copy
/// instead of each deep-cloning the context rows.
pub async fn load_context_rows(
    caches: &Caches,
    sb: Option<&SbClient>,
    season: i64,
) -> Arc<Vec<Value>> {
    let inflight_key = format!("ctx_static:{season}");
    caches
        .inflight
        .run(&inflight_key, || async {
            if let Some(v) = caches.context_static.get(&season) {
                return v;
            }
            const COLS: &str = "player_id,season,strength_state,minutes,qot_blend_xg67_g33,qoc_blend_xg67_g33,zs_difficulty";
            let rows: Vec<Value> = if let Some(sb) = sb {
                let mut f = std::collections::BTreeMap::new();
                f.insert("season".to_string(), format!("eq.{season}"));
                sb.read("rapm_context", COLS, Some(&f), Some(&context_col_map()), None, 0)
                    .await
                    .unwrap_or_default()
            } else {
                Vec::new()
            };
            let arc = Arc::new(rows);
            caches.context_static.insert(season, arc.clone());
            arc
        })
        .await
}

/// `_load_rapm_static_csv`: Supabase-only (the static `rapm/rapm.csv` export is
/// outdated and intentionally never used). Per-season TTL cached + singleflight;
/// Rates rows are synthesized from Totals + context minutes ONCE per TTL window
/// and shared, so requests never re-clone or re-build the combined row set.
/// Reading one season (~1.4k rows) instead of the full ~21k-row table is what
/// keeps the resident working set small.
pub async fn load_rapm_data(
    caches: &Caches,
    sb: Option<&SbClient>,
    season: i64,
) -> Arc<RapmRows> {
    let inflight_key = format!("rapm_static:{season}");
    caches
        .inflight
        .run(&inflight_key, || async {
            if let Some(v) = caches.rapm_static.get(&season) {
                return v;
            }
            let raw: Vec<Value> = match sb {
                Some(sb) => read_rapm_supabase(sb, season).await.unwrap_or_default(),
                None => Vec::new(),
            };
            let totals = Arc::new(raw);
            let mut rates: Vec<Value> = Vec::new();
            if !totals.is_empty() {
                let ctx_rows = load_context_rows(caches, sb, season).await;
                if !ctx_rows.is_empty() {
                    rates = synthesize_rates_rows(&totals, &ctx_rows);
                }
            }
            let data = Arc::new(RapmRows {
                totals,
                rates: Arc::new(rates),
            });
            caches.rapm_static.insert(season, data.clone());
            data
        })
        .await
}

/// Loads RAPM rows for several seasons with bounded parallelism (used by the
/// career-scope endpoints that need per-season distributions across the
/// league). Results are returned in the same order as `seasons`.
pub async fn load_rapm_seasons(
    caches: &Arc<Caches>,
    sb: Option<&SbClient>,
    seasons: &[i64],
) -> Vec<Arc<RapmRows>> {
    const CONCURRENCY: usize = 4;
    let sb_owned = sb.cloned();
    let mut handles: std::collections::VecDeque<tokio::task::JoinHandle<Arc<RapmRows>>> =
        std::collections::VecDeque::new();
    let mut results: Vec<Arc<RapmRows>> = Vec::with_capacity(seasons.len());
    let mut next: usize = 0;
    while results.len() < seasons.len() {
        while handles.len() < CONCURRENCY && next < seasons.len() {
            let caches = caches.clone();
            let sb = sb_owned.clone();
            let s = seasons[next];
            handles.push_back(tokio::spawn(async move {
                load_rapm_data(&caches, sb.as_ref(), s).await
            }));
            next += 1;
        }
        if let Some(handle) = handles.pop_front() {
            match handle.await {
                Ok(rows) => results.push(rows),
                Err(_) => results.push(Arc::new(RapmRows {
                    totals: Arc::new(Vec::new()),
                    rates: Arc::new(Vec::new()),
                })),
            }
        }
    }
    results
}

/// Loads context rows for several seasons with bounded parallelism.
pub async fn load_context_seasons(
    caches: &Arc<Caches>,
    sb: Option<&SbClient>,
    seasons: &[i64],
) -> Vec<Arc<Vec<Value>>> {
    const CONCURRENCY: usize = 4;
    let sb_owned = sb.cloned();
    let mut handles: std::collections::VecDeque<tokio::task::JoinHandle<Arc<Vec<Value>>>> =
        std::collections::VecDeque::new();
    let mut results: Vec<Arc<Vec<Value>>> = Vec::with_capacity(seasons.len());
    let mut next: usize = 0;
    while results.len() < seasons.len() {
        while handles.len() < CONCURRENCY && next < seasons.len() {
            let caches = caches.clone();
            let sb = sb_owned.clone();
            let s = seasons[next];
            handles.push_back(tokio::spawn(async move {
                load_context_rows(&caches, sb.as_ref(), s).await
            }));
            next += 1;
        }
        if let Some(handle) = handles.pop_front() {
            match handle.await {
                Ok(rows) => results.push(rows),
                Err(_) => results.push(Arc::new(Vec::new())),
            }
        }
    }
    results
}

fn str_of(v: Option<&Value>) -> String {
    match v {
        Some(Value::String(s)) => s.trim().to_string(),
        Some(Value::Number(n)) => n.to_string(),
        _ => String::new(),
    }
}

/// `_synthesize_rates_rows`: divide value columns by (minutes/60).
pub fn synthesize_rates_rows(totals_rows: &[Value], ctx_rows: &[Value]) -> Vec<Value> {
    const VALUE_COLS: [&str; 18] = [
        "CF", "CA", "GF", "GA", "xGF", "xGA", "PEN_taken", "PEN_drawn",
        "C_plusminus", "G_plusminus", "xG_plusminus", "PEN_plusminus",
        "PP_CF", "PP_GF", "PP_xGF", "SH_CA", "SH_GA", "SH_xGA",
    ];
    let mut mins_map: HashMap<(String, String, String), f64> = HashMap::new();
    for cr in ctx_rows {
        let key = (
            str_of(cr.get("PlayerID")),
            str_of(cr.get("Season")),
            str_of(cr.get("StrengthState")),
        );
        if let Some(m) = parse_locale_float(cr.get("Minutes")) {
            if m > 0.0 {
                mins_map.insert(key, m);
            }
        }
    }
    let mut rates: Vec<Value> = Vec::new();
    for r in totals_rows {
        let rt = str_of(r.get("Rates_Totals")).to_lowercase();
        if !rt.starts_with("tot") {
            continue;
        }
        let key = (
            str_of(r.get("PlayerID")),
            str_of(r.get("Season")),
            str_of(r.get("StrengthState")),
        );
        let Some(mins) = mins_map.get(&key) else { continue };
        if *mins <= 0.0 {
            continue;
        }
        let factor = 60.0 / mins;
        let mut rate_row = r.clone();
        if let Some(obj) = rate_row.as_object_mut() {
            obj.insert("Rates_Totals".into(), json!("Rates"));
            for col in VALUE_COLS {
                if let Some(v) = parse_locale_float(obj.get(col)) {
                    obj.insert(col.to_string(), json!(v * factor));
                }
            }
        }
        rates.push(rate_row);
    }
    rates
}
