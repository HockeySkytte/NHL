//! Shared NHL HTTP helpers (replace `requests.get` call sites in routes.py).

use std::time::Duration;

use serde_json::Value;

pub const API_WEB: &str = "https://api-web.nhle.com";
pub const API_STATS: &str = "https://api.nhle.com/stats/rest/en";

async fn send(
    http: &reqwest::Client,
    url: &str,
    timeout_s: u64,
    browser_ua: bool,
) -> Result<reqwest::Response, String> {
    let mut req = http.get(url).timeout(Duration::from_secs(timeout_s));
    if browser_ua {
        req = req.header("User-Agent", "Mozilla/5.0");
    }
    req.send()
        .await
        .map_err(|e| format!("fetch_failed: {e}"))
}

/// GET + JSON parse; Ok only on HTTP 200 with valid JSON.
pub async fn get_json(http: &reqwest::Client, url: &str, timeout_s: u64) -> Result<Value, String> {
    let resp = send(http, url, timeout_s, false).await?;
    let status = resp.status();
    if status != reqwest::StatusCode::OK {
        return Err(format!("upstream_status:{status}"));
    }
    resp.json::<Value>()
        .await
        .map_err(|e| format!("invalid_upstream: {e}"))
}

/// GET + JSON with a browser User-Agent (the stats REST API needs it).
pub async fn get_json_with_ua(
    http: &reqwest::Client,
    url: &str,
    timeout_s: u64,
) -> Result<Value, String> {
    let resp = send(http, url, timeout_s, true).await?;
    let status = resp.status();
    if status != reqwest::StatusCode::OK {
        return Err(format!("upstream_status:{status}"));
    }
    resp.json::<Value>()
        .await
        .map_err(|e| format!("invalid_upstream: {e}"))
}

/// GET raw bytes (image proxies). Sends a browser User-Agent because
/// `assets.nhle.com` rejects the default reqwest UA.
pub async fn get_bytes(
    http: &reqwest::Client,
    url: &str,
    timeout_s: u64,
) -> Result<Vec<u8>, String> {
    let resp = send(http, url, timeout_s, true).await?;
    let status = resp.status();
    if status != reqwest::StatusCode::OK {
        return Err(format!("upstream_status:{status}"));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| format!("invalid_upstream: {e}"))
}
