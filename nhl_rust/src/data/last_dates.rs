//! `Last_date.csv` → { season: last date } (port of `_load_last_dates`).

use std::collections::BTreeMap;

use crate::config::Config;

pub fn load(cfg: &Config) -> BTreeMap<i64, String> {
    let mut out = BTreeMap::new();
    let path = &cfg.last_dates_csv_path;
    if !path.exists() {
        tracing::warn!("Last_date.csv not found at {} — standings season dates unavailable", path.display());
        return out;
    }
    let Ok(mut reader) = csv::ReaderBuilder::new().has_headers(true).from_path(path) else {
        return out;
    };
    for record in reader.records() {
        let Ok(record) = record else { continue };
        let season = record.get(0).unwrap_or("").trim();
        let date = record.get(1).unwrap_or("").trim();
        if let Ok(s) = season.parse::<i64>() {
            if !date.is_empty() {
                out.insert(s, date.to_string());
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Config;

    #[test]
    fn parses_last_dates() {
        let mut cfg = Config::from_env();
        // Point at the real repo CSV (default path is ../Last_date.csv relative
        // to the nhl_rust working dir).
        let path = std::path::Path::new("..").join("Last_date.csv");
        if path.exists() {
            cfg.last_dates_csv_path = path;
            let map = load(&cfg);
            assert!(!map.is_empty());
            assert!(map.contains_key(&20242025));
        }
    }
}
