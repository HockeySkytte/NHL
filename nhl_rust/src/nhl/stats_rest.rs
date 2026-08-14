//! NHL Stats REST endpoints (`api.nhle.com/stats/rest/en/...`).

use serde_json::Value;

use crate::nhl::client::{get_json_with_ua, API_STATS};

/// Fetch `{kind}/summary?limit=-1&start=0&cayenneExp=...` rows.
pub async fn summary_rows(
    http: &reqwest::Client,
    kind: &str,
    cayenne_exp: &str,
) -> Result<Vec<Value>, String> {
    let url = format!("{API_STATS}/{kind}/summary?limit=-1&start=0&cayenneExp={cayenne_exp}");
    let data = get_json_with_ua(http, &url, 25).await?;
    Ok(data
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}

/// Fetch `{kind}/bios?limit=-1&start=0&cayenneExp=seasonId={season}` rows.
pub async fn bios_rows(
    http: &reqwest::Client,
    kind: &str,
    season: i64,
) -> Result<Vec<Value>, String> {
    let url = format!("{API_STATS}/{kind}/bios?limit=-1&start=0&cayenneExp=seasonId={season}");
    let data = get_json_with_ua(http, &url, 20).await?;
    Ok(data
        .get("data")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default())
}
