//! Supabase GoTrue auth + `user_accounts` PostgREST helpers.
//!
//! Ports of `app/supabase_client.py`'s auth functions (raw HTTP, no SDK):
//! - GoTrue: password sign-in, admin user create/get/list/update/delete.
//! - PostgREST: `user_accounts` read/upsert/list/delete.
//!
//! Error semantics mirror the Python side: return `None` on failure where the
//! caller distinguishes "unavailable" from "empty".

use serde_json::{json, Value};

use super::read::SbClient;

impl SbClient {
    fn anon_or_service_key(&self, anon_key: Option<&str>) -> String {
        anon_key
            .map(|k| k.trim().to_string())
            .filter(|k| !k.is_empty())
            .unwrap_or_else(|| self.service_key.clone())
    }

    /// `POST /auth/v1/token?grant_type=password` — port of
    /// `auth_sign_in_with_password`. Returns the full response dict (with
    /// `user`), or `None` on network/HTTP error.
    pub async fn sign_in_with_password(
        &self,
        anon_key: Option<&str>,
        email: &str,
        password: &str,
    ) -> Option<Value> {
        let url = format!("{}/auth/v1/token", self.url);
        let key = self.anon_or_service_key(anon_key);
        let resp = self
            .http
            .post(&url)
            .query(&[("grant_type", "password")])
            .header("apikey", &key)
            .header("Content-Type", "application/json")
            .json(&json!({"email": email, "password": password}))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().await.ok()
    }

    /// `POST /auth/v1/admin/users` — port of `auth_admin_create_user`.
    pub async fn admin_create_user(&self, email: &str, password: &str, user_metadata: &Value) -> Option<Value> {
        let url = format!("{}/auth/v1/admin/users", self.url);
        let resp = self
            .http
            .post(&url)
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .header("Content-Type", "application/json")
            .json(&json!({
                "email": email,
                "password": password,
                "email_confirm": true,
                "user_metadata": user_metadata,
            }))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let plain: Value = resp.json().await.ok()?;
        Some(plain)
    }

    /// `GET /auth/v1/admin/users/{uid}` — returns the `user` object.
    pub async fn admin_get_user(&self, uid: &str) -> Option<Value> {
        let url = format!("{}/auth/v1/admin/users/{}", self.url, uid);
        let resp = self
            .http
            .get(&url)
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let plain: Value = resp.json().await.ok()?;
        plain.get("user").cloned().or(Some(plain))
    }

    /// `GET /auth/v1/admin/users` — port of `auth_admin_list_users`.
    /// `page`/`per_page` are 1-based (GoTrue); defaults match the SDK (50).
    pub async fn admin_list_users(&self, page: usize, per_page: usize) -> Option<Vec<Value>> {
        let url = format!("{}/auth/v1/admin/users", self.url);
        let resp = self
            .http
            .get(&url)
            .query(&[
                ("page", page.to_string()),
                ("per_page", per_page.to_string()),
            ])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let plain: Value = resp.json().await.ok()?;
        match plain {
            Value::Array(users) => Some(users),
            Value::Object(map) => map
                .get("users")
                .and_then(|u| u.as_array().cloned())
                .map(|u| u)
                .or(Some(Vec::new())),
            _ => Some(Vec::new()),
        }
    }

    /// `PUT /auth/v1/admin/users/{uid}` — port of `auth_admin_update_user`.
    pub async fn admin_update_user(&self, uid: &str, attributes: &Value) -> Option<Value> {
        let url = format!("{}/auth/v1/admin/users/{}", self.url, uid);
        let resp = self
            .http
            .put(&url)
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .header("Content-Type", "application/json")
            .json(attributes)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let plain: Value = resp.json().await.ok()?;
        Some(plain)
    }

    /// `DELETE /auth/v1/admin/users/{uid}` — port of `auth_admin_delete_user`.
    pub async fn admin_delete_user(&self, uid: &str) -> bool {
        let url = format!("{}/auth/v1/admin/users/{}", self.url, uid);
        self.http
            .delete(&url)
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// `GET /rest/v1/user_accounts?auth_user_id=eq.{id}&limit=1` — port of
    /// `get_user_account`.
    pub async fn get_user_account(&self, auth_user_id: &str) -> Option<Value> {
        let url = format!("{}/rest/v1/user_accounts", self.url);
        let resp = self
            .http
            .get(&url)
            .query(&[
                ("select", "*"),
                ("auth_user_id", &format!("eq.{auth_user_id}")),
                ("limit", "1"),
            ])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let rows: Vec<Value> = resp.json().await.ok()?;
        rows.into_iter().next()
    }

    /// `POST /rest/v1/user_accounts` upsert (on_conflict=auth_user_id), then
    /// re-fetch — port of `upsert_user_account`. On a missing-column error the
    /// offending column is dropped and the upsert retried (up to 10 times),
    /// matching the Python retry loop.
    pub async fn upsert_user_account(&self, mut payload: Value) -> Option<Value> {
        let auth_user_id = payload
            .get("auth_user_id")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())?;
        for _ in 0..10 {
            let url = format!(
                "{}/rest/v1/user_accounts?on_conflict=auth_user_id",
                self.url
            );
            let resp = self
                .http
                .post(&url)
                .header("apikey", &self.service_key)
                .header("Authorization", format!("Bearer {}", self.service_key))
                .header("Content-Type", "application/json")
                .header("Prefer", "resolution=merge-duplicates,return=representation")
                .json(&payload)
                .send()
                .await
                .ok()?;
            if resp.status().is_success() {
                return self.get_user_account(&auth_user_id).await;
            }
            // Missing-column retry: extract the column name from the error.
            let err_text = resp.text().await.unwrap_or_default();
            if let Some(col) = missing_column_name(&err_text, "user_accounts") {
                if let Value::Object(map) = &mut payload {
                    if map.remove(&col).is_some() {
                        continue;
                    }
                }
            }
            return None;
        }
        None
    }

    /// `GET /rest/v1/user_accounts` (ordered by created_at desc) — port of
    /// `list_user_accounts`. Returns `Some(vec![])` for an empty table and
    /// `None` on failure (Python raises on non-missing-table errors; the
    /// callers treat both as "no rows", but keep the distinction).
    pub async fn list_user_accounts(&self) -> Option<Vec<Value>> {
        let url = format!("{}/rest/v1/user_accounts", self.url);
        let resp = self
            .http
            .get(&url)
            .query(&[("select", "*"), ("order", "created_at.desc")])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().await.ok()
    }

    /// `DELETE /rest/v1/user_accounts?auth_user_id=eq.{id}` — port of
    /// `delete_user_account`.
    pub async fn delete_user_account(&self, auth_user_id: &str) -> bool {
        let url = format!("{}/rest/v1/user_accounts", self.url);
        self.http
            .delete(&url)
            .query(&[("auth_user_id", &format!("eq.{auth_user_id}"))])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }

    /// `GET /rest/v1/card_builder_layouts?auth_user_id=eq.{id}` — port of
    /// `list_card_builder_layouts`.
    pub async fn list_card_builder_layouts(&self, auth_user_id: &str) -> Option<Vec<Value>> {
        let url = format!("{}/rest/v1/card_builder_layouts", self.url);
        let resp = self
            .http
            .get(&url)
            .query(&[
                ("select", "*"),
                ("auth_user_id", &format!("eq.{auth_user_id}")),
                ("order", "updated_at.desc"),
            ])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        resp.json().await.ok()
    }

    /// `POST /rest/v1/card_builder_layouts` upsert (on_conflict=id) — port of
    /// `upsert_card_builder_layout`. Returns the re-fetched row.
    pub async fn upsert_card_builder_layout(&self, payload: Value) -> Option<Value> {
        let layout_id = payload.get("id").and_then(Value::as_str).map(|s| s.to_string())?;
        let auth_user_id = payload.get("auth_user_id").and_then(Value::as_str).map(|s| s.to_string())?;
        let url = format!("{}/rest/v1/card_builder_layouts?on_conflict=id", self.url);
        let resp = self
            .http
            .post(&url)
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .header("Content-Type", "application/json")
            .header("Prefer", "resolution=merge-duplicates,return=representation")
            .json(&payload)
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        // Re-fetch to match Python's get_card_builder_layout.
        let url = format!("{}/rest/v1/card_builder_layouts", self.url);
        let resp = self
            .http
            .get(&url)
            .query(&[
                ("select", "*"),
                ("auth_user_id", &format!("eq.{auth_user_id}")),
                ("id", &format!("eq.{layout_id}")),
                ("limit", "1"),
            ])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .ok()?;
        if !resp.status().is_success() {
            return None;
        }
        let rows: Vec<Value> = resp.json().await.ok()?;
        rows.into_iter().next()
    }

    /// `DELETE /rest/v1/card_builder_layouts?...` — port of
    /// `delete_card_builder_layout`.
    pub async fn delete_card_builder_layout(&self, auth_user_id: &str, layout_id: &str) -> bool {
        let url = format!("{}/rest/v1/card_builder_layouts", self.url);
        self.http
            .delete(&url)
            .query(&[
                ("auth_user_id", &format!("eq.{auth_user_id}")),
                ("id", &format!("eq.{layout_id}")),
            ])
            .header("apikey", &self.service_key)
            .header("Authorization", format!("Bearer {}", self.service_key))
            .send()
            .await
            .map(|r| r.status().is_success())
            .unwrap_or(false)
    }
}

/// Extracts a missing PostgREST column name from an error body, if any.
fn missing_column_name(err_text: &str, table: &str) -> Option<String> {
    let lowered = err_text.to_lowercase();
    if !lowered.contains(table.to_lowercase().as_str()) {
        return None;
    }
    // column "subscription_source" of relation "user_accounts" does not exist
    let re_col = regex::Regex::new(r#"column\s+"([a-zA-Z0-9_]+)"\s+of\s+relation\s+"%s"\s+does\s+not\s+exist"#.replace("%s", table).as_str());
    if let Ok(re) = re_col {
        if let Some(cap) = re.captures(err_text) {
            return cap.get(1).map(|m| m.as_str().to_string());
        }
    }
    // Could not find the 'subscription_source' column of 'user_accounts' in the schema cache
    let re_cache = regex::Regex::new(
        r#"could\s+not\s+find\s+the\s+'([a-zA-Z0-9_]+)'\s+column\s+of\s+'%s'\s+in\s+the\s+schema\s+cache"#
            .replace("%s", table)
            .as_str(),
    );
    if let Ok(re) = re_cache {
        if let Some(cap) = re.captures(err_text) {
            return cap.get(1).map(|m| m.as_str().to_string());
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use wiremock::matchers::{body_json, header, method, path, query_param};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn client() -> reqwest::Client {
        reqwest::Client::new()
    }

    #[tokio::test]
    async fn sign_in_posts_to_token_endpoint() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/v1/token"))
            .and(query_param("grant_type", "password"))
            .and(header("apikey", "anon"))
            .and(body_json(json!({"email": "a@b.c", "password": "pw"})))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"user": {"id": "u1", "email": "a@b.c"}})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let out = sb
            .sign_in_with_password(Some("anon"), "a@b.c", "pw")
            .await
            .unwrap();
        assert_eq!(out["user"]["id"], "u1");
    }

    #[tokio::test]
    async fn admin_create_and_get_user() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/auth/v1/admin/users"))
            .and(header("authorization", "Bearer svc"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"user": {"id": "u1"}})),
            )
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/auth/v1/admin/users/u1"))
            .and(header("authorization", "Bearer svc"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!({"user": {"id": "u1", "email": "a@b.c"}})),
            )
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let created = sb
            .admin_create_user("a@b.c", "pw", &json!({"display_name": "A"}))
            .await
            .unwrap();
        assert_eq!(created["user"]["id"], "u1");
        let fetched = sb.admin_get_user("u1").await.unwrap();
        assert_eq!(fetched["email"], "a@b.c");
    }

    #[tokio::test]
    async fn upsert_user_account_returns_fetched_row() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/rest/v1/user_accounts"))
            .respond_with(ResponseTemplate::new(201).set_body_json(json!([{"auth_user_id": "u1"}])))
            .expect(1)
            .mount(&server)
            .await;
        Mock::given(method("GET"))
            .and(path("/rest/v1/user_accounts"))
            .and(query_param("auth_user_id", "eq.u1"))
            .respond_with(
                ResponseTemplate::new(200).set_body_json(json!([{"auth_user_id": "u1", "email": "a@b.c"}])),
            )
            .expect(1)
            .mount(&server)
            .await;
        let sb = SbClient::new(client(), server.uri(), "svc".to_string());
        let row = sb
            .upsert_user_account(json!({"auth_user_id": "u1", "email": "a@b.c"}))
            .await
            .unwrap();
        assert_eq!(row["email"], "a@b.c");
    }
}
