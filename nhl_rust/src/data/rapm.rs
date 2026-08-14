//! RAPM / context loaders — ports of `_load_rapm_static_csv`,
//! `_load_context_static_csv`, `_synthesize_rates_rows` and `_sb_read_rapm`.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::state::Caches;
use crate::config::Config;

use crate::supabase::read::SbClient;
use crate::util::parse::parse_locale_float;

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

/// `_sb_read_rapm`: read `rapm` and unpack the JSONB `stddev`/`zscore`/`pp_sh`
/// columns back to flat rows with original column names.
///
/// Data is read from Supabase ONLY (the static `rapm/rapm.csv` export is
/// outdated and intentionally never used).
async fn read_rapm_supabase(sb: &SbClient) -> Option<Vec<Value>> {
    // Targeted columns (no `*`) keep the full-table pagination fast; the JSONB
    // columns carry the z-scores/standard deviations unpacked below.
    const COLS: &str = "player_id,season,strength_state,rates_totals,cf,ca,gf,ga,xgf,xga,pen_taken,pen_drawn,c_plusminus,g_plusminus,xg_plusminus,pen_plusminus,stddev,zscore,pp_sh";
    let raw = sb.read("rapm", COLS, None, None, None, 0).await?;
    let cm = rapm_col_map();
    let mut out = Vec::new();
    for row in raw {
        let Some(obj) = row.as_object() else { continue };
        let mut r = serde_json::Map::new();
        for (k, v) in obj {
            if k == "stddev" || k == "zscore" || k == "pp_sh" {
                if let Some(inner) = v.as_object() {
                    for (ik, iv) in inner {
                        r.insert(ik.clone(), iv.clone());
                    }
                }
            } else {
                let key = cm.get(k).cloned().unwrap_or_else(|| k.clone());
                r.insert(key, v.clone());
            }
        }
        out.push(Value::Object(r));
    }
    Some(out)
}

/// `_load_context_static_csv` (Supabase-only, TTL cached).
pub async fn load_context_rows(caches: &Caches, sb: Option<&SbClient>, _cfg: &Config) -> Vec<Value> {
    if let Some(v) = caches.context_static.get(&()) {
        if let Value::Array(rows) = v {
            return rows;
        }
    }
    eprintln!("DBG rapm.rs: load_context_rows CACHE MISS");
    const COLS: &str = "player_id,season,strength_state,minutes,qot_blend_xg67_g33,qoc_blend_xg67_g33,zs_difficulty";
    let rows = if let Some(sb) = sb {
        sb.read("rapm_context", COLS, None, Some(&context_col_map()), None, 0)
            .await
            .unwrap_or_default()
    } else {
        Vec::new()
    };
    caches.context_static.insert((), Value::Array(rows.clone()));
    rows
}

/// `_load_rapm_static_csv`: Supabase-only (the static `rapm/rapm.csv` export is
/// outdated and intentionally never used). TTL cached; Rates rows are
/// synthesized from Totals + context minutes.
pub async fn load_rapm_rows(caches: &Caches, sb: Option<&SbClient>, _cfg: &Config) -> Vec<Value> {
    // The cache stores ONLY the raw Supabase rows (Totals). Rates rows are
    // synthesized from context minutes on demand per call and never cached:
    // the combined Totals+Rates blob (~42k rows) exceeded the cache weight cap
    // and self-evicted after every hit, forcing a full Supabase reload on
    // alternating requests.
    if let Some(v) = caches.rapm_static.get(&()) {
        if let Value::Array(rows) = v {
            return rows_with_rates(rows, caches, sb, _cfg).await;
        }
    }
    let raw: Vec<Value> = match sb {
        Some(sb) => read_rapm_supabase(sb).await.unwrap_or_default(),
        None => Vec::new(),
    };
    if raw.is_empty() {
        return raw;
    }
    let v = Value::Array(raw.clone());
    caches.rapm_static.insert((), v);
    rows_with_rates(raw, caches, sb, _cfg).await
}

/// Appends synthesized Rates rows when the cached rows are Totals-only.
async fn rows_with_rates(
    rows: Vec<Value>,
    caches: &Caches,
    sb: Option<&SbClient>,
    cfg: &Config,
) -> Vec<Value> {
    let has_rates = rows
        .iter()
        .any(|r| str_of(r.get("Rates_Totals")).to_lowercase().starts_with("rate"));
    if has_rates {
        return rows;
    }
    let ctx_rows = load_context_rows(caches, sb, cfg).await;
    if ctx_rows.is_empty() {
        return rows;
    }
    let synth = synthesize_rates_rows(&rows, &ctx_rows);
    let mut combined = rows;
    combined.extend(synth);
    combined
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
