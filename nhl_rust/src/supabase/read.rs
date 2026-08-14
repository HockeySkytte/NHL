//! Supabase PostgREST read client — port of `_sb_read()` in `app/routes.py`.
//!
//! We talk directly to the REST API (`/rest/v1/{table}`) with the service key;
//! no SDK is needed. Semantics replicated from `supabase-py`:
//! - paginated reads of 1000 rows/page via the `Range` header,
//! - PostgREST filter expressions like `eq.20252026` / `in.(a,b,c)`,
//! - snake_case → CSV column renames via `col_map`,
//! - **`None` on any failure** (Supabase down) vs `Some(vec![])` for an empty
//!   table — callers rely on this distinction to pick CSV fallbacks.

use std::collections::{BTreeMap, HashMap};

use serde_json::Value;

pub const PAGE_SIZE: usize = 1000;

/// True when both `SUPABASE_URL` and `SUPABASE_SERVICE_KEY` are set
/// (port of `auth_is_configured()`).
pub fn auth_is_configured() -> bool {
    std::env::var("SUPABASE_URL")
        .map(|v| !v.trim().is_empty())
        .unwrap_or(false)
        && std::env::var("SUPABASE_SERVICE_KEY")
            .map(|v| !v.trim().is_empty())
            .unwrap_or(false)
}

#[derive(Clone)]
pub struct SbClient {
    pub(crate) http: reqwest::Client,
    pub(crate) url: String,
    pub(crate) service_key: String,
}

impl SbClient {
    pub fn from_env(http: reqwest::Client) -> Option<Self> {
        let url = std::env::var("SUPABASE_URL").ok()?;
        let service_key = std::env::var("SUPABASE_SERVICE_KEY").ok()?;
        if url.trim().is_empty() || service_key.trim().is_empty() {
            return None;
        }
        Some(Self::new(http, url.trim_end_matches('/').to_string(), service_key))
    }

    pub fn new(http: reqwest::Client, url: String, service_key: String) -> Self {
        Self {
            http,
            url: url.trim_end_matches('/').to_string(),
            service_key,
        }
    }

    pub fn http(&self) -> &reqwest::Client {
        &self.http
    }

    /// Port of `_sb_read(table, columns, filters, col_map, order, limit)`.
    ///
    /// - `filters`: PostgREST expressions (`{"season": "eq.20252026"}`).
    /// - `limit == 0` means no limit (like the Python default).
    #[allow(clippy::too_many_arguments)]
    pub async fn read(
        &self,
        table: &str,
        columns: &str,
        filters: Option<&BTreeMap<String, String>>,
        col_map: Option<&HashMap<String, String>>,
        order: Option<&str>,
        limit: usize,
    ) -> Option<Vec<Value>> {
        let mut all: Vec<Value> = Vec::new();
        let mut offset: usize = 0;
        loop {
            let mut query: Vec<(String, String)> = Vec::new();
            query.push(("select".to_string(), columns.to_string()));
            if let Some(filters) = filters {
                for (col, expr) in filters {
                    query.push((col.clone(), expr.clone()));
                }
            }
            if let Some(order) = order {
                query.push(("order".to_string(), order.to_string()));
            }

            let url = format!("{}/rest/v1/{table}", self.url);
            let resp = self
                .http
                .get(&url)
                .query(&query)
                .header("apikey", &self.service_key)
                .header("Authorization", format!("Bearer {}", self.service_key))
                .header("Range", format!("{offset}-{}", offset + PAGE_SIZE - 1))
                .send()
                .await
                .ok()?;
            if !resp.status().is_success() {
                return None;
            }
            let rows: Vec<Value> = resp.json().await.ok()?;
            let count = rows.len();
            for row in rows {
                let row = if let Some(col_map) = col_map {
                    rename_keys(row, col_map)
                } else {
                    row
                };
                all.push(row);
            }
            offset += count;
            if count < PAGE_SIZE {
                break;
            }
        }
        if limit > 0 && all.len() > limit {
            all.truncate(limit);
        }
        Some(all)
    }
}

/// Applies the snake_case → original column rename to a row object.
pub fn rename_keys(mut row: Value, col_map: &HashMap<String, String>) -> Value {
    if let Value::Object(map) = &mut row {
        let mut out = serde_json::Map::new();
        for (k, v) in map.iter() {
            let key = col_map.get(k).cloned().unwrap_or_else(|| k.clone());
            out.insert(key, v.clone());
        }
        *map = out;
    }
    row
}

/// Convenience for building a filter map from `&[(&str, &str)]` pairs.
#[allow(dead_code)]
pub fn filters(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn read_applies_headers_filters_and_col_map() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/v1/teams"))
            .and(query_param("select", "*"))
            .and(query_param("season", "eq.20252026"))
            .and(header("apikey", "svc"))
            .and(header("authorization", "Bearer svc"))
            .and(header("range", "0-999"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!([{"team": "BOS", "active": true}])),
            )
            .expect(1)
            .mount(&server)
            .await;

        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let cm: HashMap<String, String> = [
            ("team".to_string(), "Team".to_string()),
            ("active".to_string(), "Active".to_string()),
        ]
        .into_iter()
        .collect();
        let rows = sb
            .read(
                "teams",
                "*",
                Some(&filters(&[("season", "eq.20252026")])),
                Some(&cm),
                None,
                0,
            )
            .await
            .unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0]["Team"], "BOS");
        assert_eq!(rows[0]["Active"], true);
    }

    #[tokio::test]
    async fn read_paginates_with_range_headers() {
        let server = MockServer::start().await;
        let page: Vec<Value> = (0..PAGE_SIZE).map(|i| json!({"game_id": i})).collect();
        Mock::given(method("GET"))
            .and(path("/rest/v1/pbp"))
            .and(header("range", "0-999"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&page))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/v1/pbp"))
            .and(header("range", "1000-1999"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([{"game_id": 1000}])))
            .expect(1)
            .mount(&server)
            .await;

        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let rows = sb.read("pbp", "*", None, None, None, 0).await.unwrap();
        assert_eq!(rows.len(), PAGE_SIZE + 1);
    }

    #[tokio::test]
    async fn read_returns_none_on_http_error() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(500))
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        assert!(sb.read("teams", "*", None, None, None, 0).await.is_none());
    }

    #[tokio::test]
    async fn read_empty_table_is_some_empty() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!([])))
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        assert_eq!(sb.read("teams", "*", None, None, None, 0).await, Some(vec![]));
    }
}
