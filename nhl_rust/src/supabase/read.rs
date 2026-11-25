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
    pub(crate) auth_header: String,
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
        let auth_header = format!("Bearer {service_key}");
        Self {
            http,
            url: url.trim_end_matches('/').to_string(),
            service_key,
            auth_header,
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

        // When `limit` fits within a single page, request only `0..limit-1` and
        // skip `Prefer: count=exact` so Postgres avoids an expensive COUNT(*) scan.
        let page0_size = if limit > 0 { limit.min(PAGE_SIZE) } else { PAGE_SIZE };
        let need_count = limit == 0 || limit > PAGE_SIZE;

        let (page0, content_range) = match fetch_page(
            &self.http,
            &url,
            &query,
            &self.service_key,
            &self.auth_header,
            0,
            page0_size,
            need_count,
        )
        .await
        {
            Some(v) => v,
            None => return None,
        };
        let page0_count = page0.len();
        if page0_count < page0_size || (limit > 0 && page0_count >= limit) {
            return Some(finish_rows(page0, col_map, limit));
        }
        let Some(total) = content_range
            .and_then(|cr| cr.rsplit('/').next().and_then(|s| s.parse::<usize>().ok()))
        else {
            // No exact count available; use the sequential path.
            return read_sequential(self, &url, &query, col_map, limit, page0).await;
        };
        let effective_total = if limit > 0 { total.min(limit) } else { total };
        let pages = effective_total.div_ceil(PAGE_SIZE);
        let mut page_rows: Vec<Vec<Value>> = Vec::with_capacity(pages);
        page_rows.push(page0);
        const CHUNK: usize = 16;
        for chunk_start in (1..pages).step_by(CHUNK) {
            let chunk_end = (chunk_start + CHUNK).min(pages);
            let mut handles = Vec::with_capacity(chunk_end - chunk_start);
            for page in chunk_start..chunk_end {
                let offset = page * PAGE_SIZE;
                let page_size = (effective_total - offset).min(PAGE_SIZE);
                let http = self.http.clone();
                let url = url.clone();
                let query = query.clone();
                let service_key = self.service_key.clone();
                let auth_header = self.auth_header.clone();
                handles.push(tokio::spawn(async move {
                    fetch_page(
                        &http,
                        &url,
                        &query,
                        &service_key,
                        &auth_header,
                        offset,
                        page_size,
                        false,
                    )
                    .await
                }));
            }
            for h in handles {
                match h.await {
                    Ok(Some((rows, _))) => page_rows.push(rows),
                    _ => return None,
                }
            }
        }
        let total_rows: usize = page_rows.iter().map(Vec::len).sum();
        let mut all: Vec<Value> = Vec::with_capacity(total_rows);
        for rows in page_rows {
            all.extend(rows);
        }
        Some(finish_rows(all, col_map, limit))
    }
}

/// Fetches one page of rows; `prefer_count` adds `Prefer: count=exact` so the
/// caller can learn the total from the `Content-Range` header.
#[allow(clippy::too_many_arguments)]
async fn fetch_page(
    http: &reqwest::Client,
    url: &str,
    query: &[(String, String)],
    service_key: &str,
    auth_header: &str,
    offset: usize,
    page_size: usize,
    prefer_count: bool,
) -> Option<(Vec<Value>, Option<String>)> {
    let end = offset + page_size.max(1) - 1;
    let mut req = http
        .get(url)
        .query(query)
        .header("apikey", service_key)
        .header("Authorization", auth_header)
        .header("Range", format!("{offset}-{end}"));
    if prefer_count {
        req = req.header("Prefer", "count=exact");
    }
    let resp = req.send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let content_range = resp
        .headers()
        .get("content-range")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());
    let rows: Vec<Value> = resp.json().await.ok()?;
    Some((rows, content_range))
}

/// Sequential fallback (used when the exact count is unavailable); keeps the
/// original `_sb_read` loop semantics while honoring `limit`.
async fn read_sequential(
    sb: &SbClient,
    url: &str,
    query: &[(String, String)],
    col_map: Option<&HashMap<String, String>>,
    limit: usize,
    mut all: Vec<Value>,
) -> Option<Vec<Value>> {
    let mut offset: usize = all.len();
    loop {
        if limit > 0 && all.len() >= limit {
            break;
        }
        let page_size = if limit > 0 {
            (limit - all.len()).min(PAGE_SIZE)
        } else {
            PAGE_SIZE
        };
        let resp = sb
            .http
            .get(url)
            .query(query)
            .header("apikey", &sb.service_key)
            .header("Authorization", &sb.auth_header)
            .header("Range", format!("{offset}-{}", offset + page_size - 1))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let rows: Vec<Value> = resp.json().await.ok()?;
        let count = rows.len();
        all.extend(rows);
        offset += count;
        if count < page_size {
            break;
        }
    }
    Some(finish_rows(all, col_map, limit))
}

/// Applies the column rename map and truncates to `limit` (0 = no limit).
fn finish_rows(rows: Vec<Value>, col_map: Option<&HashMap<String, String>>, limit: usize) -> Vec<Value> {
    let mut all = map_rows(rows, col_map);
    if limit > 0 && all.len() > limit {
        all.truncate(limit);
    }
    all
}

/// Applies the snake_case → original column rename to a row object without
/// cloning values (moves values out of the original map).
pub fn rename_keys(mut row: Value, col_map: &HashMap<String, String>) -> Value {
    if let Value::Object(map) = &mut row {
        let old = std::mem::take(map);
        let mut out = serde_json::Map::with_capacity(old.len());
        for (k, v) in old {
            let key = match col_map.get(&k) {
                Some(renamed) => renamed.clone(),
                None => k,
            };
            out.insert(key, v);
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

/// Applies the column rename map to an entire page of rows.
fn map_rows(rows: Vec<Value>, col_map: Option<&HashMap<String, String>>) -> Vec<Value> {
    match col_map {
        Some(cm) => rows.into_iter().map(|row| rename_keys(row, cm)).collect(),
        None => rows,
    }
}

/// Memory-bounded read: like [`SbClient::read`], but pages are handed to
/// `on_page` (in order) and dropped immediately instead of being accumulated
/// into one giant `Vec`. This lets a full-table scan of `season_stats`
/// (~135k rows → several hundred MB as serde Values) aggregate with only a
/// handful of pages in memory at once.
///
/// Pages are fetched with bounded in-flight concurrency (`CONCURRENCY`)
/// but emitted strictly in order, so downstream aggregation stays
/// deterministic. Falls back to the sequential loop when no exact count is
/// available. `limit == 0` means no limit (same as `read`).
#[allow(clippy::too_many_arguments)]
pub async fn read_each(
    sb: &SbClient,
    table: &str,
    columns: &str,
    filters: Option<&BTreeMap<String, String>>,
    col_map: Option<&HashMap<String, String>>,
    order: Option<&str>,
    limit: usize,
    mut on_page: impl FnMut(Vec<Value>),
) -> Option<()> {
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
    let url = format!("{}/rest/v1/{table}", sb.url);

    let page0_size = if limit > 0 { limit.min(PAGE_SIZE) } else { PAGE_SIZE };
    let need_count = limit == 0 || limit > PAGE_SIZE;

    let (page0, content_range) = match fetch_page(
        &sb.http,
        &url,
        &query,
        &sb.service_key,
        &sb.auth_header,
        0,
        page0_size,
        need_count,
    )
    .await
    {
        Some(v) => v,
        None => return None,
    };
    let mut remaining = if limit > 0 { limit } else { usize::MAX };
    let page0_count = page0.len();

    // Single page (short table) or a limit that fits in the first page.
    if page0_count < page0_size || remaining <= page0_count {
        let rows = map_rows(page0, col_map);
        let n = rows.len().min(remaining);
        if n > 0 {
            on_page(rows.into_iter().take(n).collect());
        }
        return Some(());
    }

    on_page(map_rows(page0, col_map));
    remaining -= page0_count;

    let Some(total) = content_range
        .and_then(|cr| cr.rsplit('/').next().and_then(|s| s.parse::<usize>().ok()))
    else {
        // No exact count available; use the sequential path.
        return read_sequential_each(sb, &url, &query, col_map, remaining, page0_count, on_page)
            .await;
    };
    let effective_total = if limit > 0 { total.min(limit) } else { total };
    let pages = effective_total.div_ceil(PAGE_SIZE);

    // Bounded in-flight window: keeps peak transient memory to ~CONCURRENCY
    // pages instead of the whole table.
    const CONCURRENCY: usize = 8;
    type PageHandle = tokio::task::JoinHandle<Option<(Vec<Value>, Option<String>)>>;
    let mut handles: std::collections::VecDeque<(usize, PageHandle)> = std::collections::VecDeque::new();
    let mut next_page: usize = 1;
    let spawn_page = |handles: &mut std::collections::VecDeque<(usize, PageHandle)>, page: usize| {
        let offset = page * PAGE_SIZE;
        let page_size = (effective_total - offset).min(PAGE_SIZE);
        let http = sb.http.clone();
        let url = url.clone();
        let query = query.clone();
        let service_key = sb.service_key.clone();
        let auth_header = sb.auth_header.clone();
        handles.push_back((page, tokio::spawn(async move {
            fetch_page(
                &http,
                &url,
                &query,
                &service_key,
                &auth_header,
                offset,
                page_size,
                false,
            )
            .await
        })));
    };
    while next_page < pages && handles.len() < CONCURRENCY {
        spawn_page(&mut handles, next_page);
        next_page += 1;
    }
    while let Some((_, handle)) = handles.pop_front() {
        let result = match handle.await {
            Ok(r) => r,
            Err(_) => return None,
        };
        let rows = match result {
            Some((rows, _)) => rows,
            None => return None,
        };
        let count = rows.len();
        let rows = map_rows(rows, col_map);
        if remaining <= count {
            if remaining > 0 {
                on_page(rows.into_iter().take(remaining).collect());
            }
            return Some(());
        }
        on_page(rows);
        remaining -= count;
        if count < PAGE_SIZE {
            return Some(());
        }
        if next_page < pages {
            spawn_page(&mut handles, next_page);
            next_page += 1;
        }
    }
    Some(())
}

/// Sequential fallback for [`SbClient::read_each`] (used when the exact count
/// is unavailable); emits pages through `on_page` and drops them.
async fn read_sequential_each(
    sb: &SbClient,
    url: &str,
    query: &[(String, String)],
    col_map: Option<&HashMap<String, String>>,
    mut remaining: usize,
    mut offset: usize,
    mut on_page: impl FnMut(Vec<Value>),
) -> Option<()> {
    loop {
        if remaining == 0 {
            break;
        }
        let page_size = remaining.min(PAGE_SIZE);
        let resp = sb
            .http
            .get(url)
            .query(query)
            .header("apikey", &sb.service_key)
            .header("Authorization", &sb.auth_header)
            .header("Range", format!("{offset}-{}", offset + page_size - 1))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let rows: Vec<Value> = resp.json().await.ok()?;
        let count = rows.len();
        let rows = map_rows(rows, col_map);
        if remaining <= count {
            if remaining > 0 {
                on_page(rows.into_iter().take(remaining).collect());
            }
            return Some(());
        }
        on_page(rows);
        remaining -= count;
        offset += count;
        if count < page_size {
            break;
        }
    }
    Some(())
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

    #[tokio::test]
    async fn read_respects_small_limit_without_extra_pages() {
        let server = MockServer::start().await;
        Mock::given(method("GET"))
            .and(path("/rest/v1/manager_games"))
            .and(header("range", "0-49"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_json((0..50).map(|i| json!({"id": i})).collect::<Vec<_>>()),
            )
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let rows = sb
            .read("manager_games", "id", None, None, None, 50)
            .await
            .unwrap();
        assert_eq!(rows.len(), 50);
    }
}
