//! Generic PostgREST write helpers for app-managed tables.
//!
//! The `supabase/read.rs` client covers reads; these helpers add the write
//! side the Manager game needs (bulk upsert, patch, delete). Patterns mirror
//! `app/supabase_client.py` / the upserts in `supabase/auth.rs`:
//! service key auth, `Prefer: resolution=merge-duplicates`, PostgREST
//! `on_conflict` columns.

use serde_json::Value;

use crate::supabase::read::SbClient;

/// PostgREST bulk-insert chunk size (keeps request bodies small).
const CHUNK: usize = 500;

fn apply_auth(sb: &SbClient, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
    req.header("apikey", &sb.service_key)
        .header("Authorization", &sb.auth_header)
        .header("Content-Type", "application/json")
}

/// Upsert `rows` into `table` with `on_conflict` (comma-separated columns).
/// Batches writes in chunks. Returns the flattened `return=representation`
/// rows on success (single row → object, bulk → array), or `None` on any
/// failure.
pub async fn upsert_rows(
    sb: &SbClient,
    table: &str,
    rows: &[Value],
    on_conflict: &str,
) -> Option<Vec<Value>> {
    if rows.is_empty() {
        return Some(Vec::new());
    }
    let url = format!("{}/rest/v1/{table}?on_conflict={on_conflict}", sb.url);
    let mut out: Vec<Value> = Vec::with_capacity(rows.len());
    for chunk in rows.chunks(CHUNK) {
        let req = apply_auth(sb, sb.http.post(&url))
            .header("Prefer", "resolution=merge-duplicates,return=representation");
        let req = if chunk.len() == 1 {
            req.json(&chunk[0])
        } else {
            req.json(chunk)
        };
        let resp = req.send().await.ok()?;
        if !resp.status().is_success() {
            return None;
        }
        if let Ok(v) = resp.json::<Value>().await {
            match v {
                Value::Array(items) => out.extend(items),
                other => out.push(other),
            }
        }
    }
    Some(out)
}

/// Upsert `rows` with `Prefer: resolution=merge-duplicates,return=minimal` so
/// PostgREST does not serialize and return the inserted rows when the caller
/// only cares about success/failure.
pub async fn upsert_rows_minimal(
    sb: &SbClient,
    table: &str,
    rows: &[Value],
    on_conflict: &str,
) -> bool {
    if rows.is_empty() {
        return true;
    }
    let url = format!("{}/rest/v1/{table}?on_conflict={on_conflict}", sb.url);
    for chunk in rows.chunks(CHUNK) {
        let req = apply_auth(sb, sb.http.post(&url))
            .header("Prefer", "resolution=merge-duplicates,return=minimal");
        let req = if chunk.len() == 1 {
            req.json(&chunk[0])
        } else {
            req.json(chunk)
        };
        match req.send().await {
            Ok(resp) if resp.status().is_success() => {}
            _ => return false,
        }
    }
    true
}

/// Patch rows matching `filters` (`(column, value)` pairs → `eq.` filters)
/// with `patch`. Uses `return=minimal` since only success/failure is returned.
pub async fn update_rows(
    sb: &SbClient,
    table: &str,
    filters: &[(&str, &str)],
    patch: &Value,
) -> bool {
    let url = format!("{}/rest/v1/{table}", sb.url);
    let pairs: Vec<(String, String)> = filters
        .iter()
        .map(|(col, val)| (col.to_string(), format!("eq.{val}")))
        .collect();
    let resp = apply_auth(sb, sb.http.patch(&url).query(&pairs))
        .header("Prefer", "return=minimal")
        .json(patch)
        .send()
        .await;
    matches!(resp, Ok(r) if r.status().is_success())
}

/// Delete rows matching `filters`. Returns true on success.
pub async fn delete_rows(sb: &SbClient, table: &str, filters: &[(&str, &str)]) -> bool {
    let url = format!("{}/rest/v1/{table}", sb.url);
    let pairs: Vec<(String, String)> = filters
        .iter()
        .map(|(col, val)| (col.to_string(), format!("eq.{val}")))
        .collect();
    let resp = apply_auth(sb, sb.http.delete(&url).query(&pairs))
        .send()
        .await;
    matches!(resp, Ok(r) if r.status().is_success())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::supabase::read::SbClient;

    fn client() -> SbClient {
        SbClient::new(reqwest::Client::new(), "http://127.0.0.1:9".into(), "k".into())
    }

    #[tokio::test]
    async fn upsert_rows_short_circuits_on_empty() {
        let sb = client();
        let rows: Vec<Value> = Vec::new();
        assert!(upsert_rows(&sb, "manager_leagues", &rows, "id").await.is_some());
    }

    #[tokio::test]
    async fn upsert_rows_returns_none_on_connection_failure() {
        let sb = client();
        let rows = vec![serde_json::json!({"name": "L"})];
        assert!(upsert_rows(&sb, "manager_leagues", &rows, "id").await.is_none());
    }
}
