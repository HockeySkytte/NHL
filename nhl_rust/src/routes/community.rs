//! Community feed API (M5): `/api/community/posts` — Supabase-first with an
//! HTML scrape fallback (port of `_community_fetch_posts_supabase` +
//! `_community_fetch_posts_html`).

use std::collections::HashMap;

use axum::extract::{Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde_json::{json, Value};

use crate::state::AppState;

pub fn router() -> Router<AppState> {
    Router::new().route("/api/community/posts", get(api_community_posts))
}

fn community_site_base_url(cfg: &crate::config::Config) -> String {
    let base = cfg
        .community_site_url
        .clone()
        .unwrap_or_else(|| "https://community.hockey-statistics.com".to_string());
    base.trim_end_matches('/').to_string()
}

fn community_abs_url(cfg: &crate::config::Config, url: &str) -> String {
    let raw = url.trim();
    if raw.is_empty() {
        return String::new();
    }
    if raw.starts_with("http://") || raw.starts_with("https://") {
        return raw.to_string();
    }
    let base = community_site_base_url(cfg);
    if raw.starts_with('/') {
        format!("{base}{raw}")
    } else {
        format!("{base}/{raw}")
    }
}

fn community_strip_preview_text(text: &Value, max_chars: usize) -> String {
    let raw = text.as_str().unwrap_or("").trim().to_string();
    if raw.is_empty() {
        return String::new();
    }
    let collapsed: String = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    if collapsed.chars().count() <= max_chars {
        collapsed
    } else {
        let cut: String = collapsed.chars().take(max_chars).collect();
        format!("{cut}…")
    }
}

fn to_int(v: &Value) -> i64 {
    v.as_i64().unwrap_or_else(|| {
        v.as_str()
            .and_then(|s| s.parse::<i64>().ok())
            .unwrap_or(0)
    })
}

/// Supabase-first fetch (port of `_community_fetch_posts_supabase`).
async fn fetch_posts_supabase(state: &AppState, sort: &str, limit: usize) -> Vec<Value> {
    let Some(sb) = state.sb.as_ref() else {
        return Vec::new();
    };
    let hub_id = state.cfg.community_nhl_hub_id.clone().unwrap_or_default();
    let columns = "id,title,author_display_name,created_at,body,preview_image_url,score,comment_count,status,hub_id";
    let mut filters: Vec<(String, String)> = Vec::new();
    if !hub_id.is_empty() {
        filters.push(("hub_id".to_string(), format!("eq.{hub_id}")));
    }
    filters.push(("status".to_string(), "eq.active".to_string()));
    let order = if sort == "top" {
        Some("score.desc,comment_count.desc,created_at.desc")
    } else {
        Some("created_at.desc")
    };
    let mut rows = sb
        .read(
            "community_posts",
            columns,
            Some(&filters.into_iter().collect()),
            None,
            order,
            limit.max(5),
        )
        .await;
    if rows.as_ref().map(|r| r.is_empty()).unwrap_or(true) && !hub_id.is_empty() {
        // Fallback path without status/hub constraints.
        let mut filters2: Vec<(String, String)> = Vec::new();
        filters2.push(("hub_id".to_string(), format!("eq.{hub_id}")));
        let order2 = if sort == "top" {
            Some("score.desc,comment_count.desc,created_at.desc")
        } else {
            Some("created_at.desc")
        };
        rows = sb
            .read(
                "community_posts",
                columns,
                Some(&filters2.into_iter().collect()),
                None,
                order2,
                limit.max(5),
            )
            .await;
    }
    let Some(all_rows) = rows else {
        return Vec::new();
    };
    if all_rows.is_empty() {
        return Vec::new();
    }
    let post_ids: Vec<String> = all_rows
        .iter()
        .filter_map(|r| r.get("id").and_then(Value::as_str).map(|s| s.to_string()))
        .filter(|s| !s.is_empty())
        .collect();
    let mut media_by_post: HashMap<String, String> = HashMap::new();
    if !post_ids.is_empty() {
        let media_rows = sb
            .read(
                "community_post_media",
                "post_id,public_url,media_kind,sort_order",
                Some(&{
                    let mut m = std::collections::BTreeMap::new();
                    m.insert("post_id".to_string(), format!("in.({})", post_ids.join(",")));
                    m.insert("media_kind".to_string(), "eq.image".to_string());
                    m
                }),
                None,
                Some("sort_order"),
                0,
            )
            .await
            .unwrap_or_default();
        for m in media_rows {
            let pid = m.get("post_id").and_then(Value::as_str).unwrap_or("").to_string();
            if pid.is_empty() || media_by_post.contains_key(&pid) {
                continue;
            }
            let url = community_abs_url(&state.cfg, m.get("public_url").and_then(Value::as_str).unwrap_or(""));
            media_by_post.insert(pid, url);
        }
    }
    let base = community_site_base_url(&state.cfg);
    let mut out: Vec<Value> = Vec::new();
    for row in all_rows.iter().take(limit) {
        let pid = row.get("id").and_then(Value::as_str).unwrap_or("").to_string();
        if pid.is_empty() {
            continue;
        }
        let preview = {
            let p = community_abs_url(&state.cfg, row.get("preview_image_url").and_then(Value::as_str).unwrap_or(""));
            if p.is_empty() {
                media_by_post.get(&pid).cloned().unwrap_or_default()
            } else {
                p
            }
        };
        let date = row
            .get("created_at")
            .and_then(Value::as_str)
            .map(|s| s.chars().take(10).collect::<String>())
            .unwrap_or_default();
        let author = row
            .get("author_display_name")
            .and_then(Value::as_str)
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "Member".to_string());
        out.push(json!({
            "id": pid,
            "title": row.get("title").and_then(Value::as_str).unwrap_or("").trim(),
            "author": author,
            "date": date,
            "snippet": community_strip_preview_text(row.get("body").unwrap_or(&Value::Null), 260),
            "preview_image": preview,
            "score": to_int(row.get("score").unwrap_or(&Value::Null)),
            "comment_count": to_int(row.get("comment_count").unwrap_or(&Value::Null)),
            "post_url": format!("{base}/posts/{pid}"),
        }));
    }
    out
}

/// HTML scrape fallback (port of `_community_fetch_posts_html`).
async fn fetch_posts_html(state: &AppState, sort: &str, limit: usize) -> Vec<Value> {
    let base = community_site_base_url(&state.cfg);
    let urls: Vec<String> = if sort == "top" {
        vec![
            format!("{base}/home?sort=top"),
            format!("{base}/hubs/nhl?sort=top"),
            format!("{base}/home"),
        ]
    } else {
        vec![format!("{base}/home")]
    };
    for url in urls {
        let Ok(resp) = state.http.get(&url).timeout(std::time::Duration::from_secs(15)).send().await else {
            continue;
        };
        if resp.status() != StatusCode::OK {
            continue;
        }
        let Ok(html_text) = resp.text().await else {
            continue;
        };
        let document = scraper::Html::parse_document(&html_text);
        // Select article.post-card-compact cards (fallback: any article).
        let mut out: Vec<Value> = Vec::new();
        let compact_sel = match scraper::Selector::parse("article.post-card-compact") {
            Ok(s) => Some(s),
            Err(_) => None,
        };
        let article_sel = match scraper::Selector::parse("article") {
            Ok(s) => Some(s),
            Err(_) => None,
        };
        let mut cards: Vec<scraper::ElementRef> = Vec::new();
        if let Some(sel) = &compact_sel {
            cards.extend(document.select(sel));
        }
        if cards.is_empty() {
            if let Some(sel) = &article_sel {
                cards.extend(document.select(sel));
            }
        }
        for card in cards {
            let mut post_url = card
                .value()
                .attr("data-post-url")
                .map(|u| community_abs_url(&state.cfg, u))
                .unwrap_or_default();
            if post_url.is_empty() {
                if let Some(link) = card.select(&match scraper::Selector::parse("h2 a, h3 a") {
                    Ok(s) => s,
                    Err(_) => continue,
                }).next() {
                    if let Some(href) = link.value().attr("href") {
                        post_url = community_abs_url(&state.cfg, href);
                    }
                }
            }
            let title = card
                .select(&match scraper::Selector::parse("h2 a, h3 a, h2, h3") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .next()
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .unwrap_or_default();
            let meta: Vec<String> = card
                .select(&match scraper::Selector::parse(".post-meta-row span") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .collect();
            let snippet = card
                .select(&match scraper::Selector::parse("p.post-body-preview") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .next()
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .unwrap_or_default();
            let img_src = card
                .select(&match scraper::Selector::parse("img.post-card-thumb") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .next()
                .and_then(|el| el.value().attr("src"))
                .map(|s| community_abs_url(&state.cfg, s))
                .unwrap_or_default();
            let score_txt = card
                .select(&match scraper::Selector::parse(".reaction-button .score-label, .reaction-button.no-pointer") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .next()
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .unwrap_or_else(|| "0".to_string());
            let comments_txt = card
                .select(&match scraper::Selector::parse("a[href*=\"#comments\"]") {
                    Ok(s) => s,
                    Err(_) => continue,
                })
                .next()
                .map(|el| el.text().collect::<Vec<_>>().join(" ").trim().to_string())
                .unwrap_or_else(|| "0 comments".to_string());
            let score_val = first_int(&score_txt);
            let comment_val = first_int(&comments_txt);
            out.push(json!({
                "id": "",
                "title": title,
                "author": meta.first().cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| "Member".to_string()),
                "date": meta.get(1).cloned().unwrap_or_default(),
                "snippet": community_strip_preview_text(&Value::String(snippet), 260),
                "preview_image": img_src,
                "score": score_val,
                "comment_count": comment_val,
                "post_url": post_url,
            }));
            if out.len() >= limit {
                break;
            }
        }
        if !out.is_empty() {
            return out;
        }
    }
    Vec::new()
}

fn first_int(text: &str) -> i64 {
    let re = match regex::Regex::new(r"-?\d+") {
        Ok(re) => re,
        Err(_) => return 0,
    };
    re.find(text)
        .and_then(|m| m.as_str().parse::<i64>().ok())
        .unwrap_or(0)
}

/// `GET /api/community/posts?sort=new|top&limit=N`
async fn api_community_posts(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<HashMap<String, String>>,
) -> Response {
    let _ = headers;
    let sort = params
        .get("sort")
        .map(|s| s.trim().to_lowercase())
        .filter(|s| s == "new" || s == "top")
        .unwrap_or_else(|| "new".to_string());
    let limit = params
        .get("limit")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(5)
        .clamp(1, 20);
    let cache_key = format!("{sort}:{limit}");
    if let Some(cached) = state.caches.community_feed.get(&cache_key) {
        return (StatusCode::OK, [("Cache-Control", "no-store")], Json(json!({"sort": sort, "posts": cached, "source": "cache"}))).into_response();
    }
    let mut posts = fetch_posts_supabase(&state, &sort, limit).await;
    let mut source = "supabase";
    if posts.is_empty() {
        posts = fetch_posts_html(&state, &sort, limit).await;
        source = "html";
    }
    if !posts.is_empty() {
        state
            .caches
            .community_feed
            .insert(cache_key, Value::Array(posts.clone()));
    }
    (
        StatusCode::OK,
        [("Cache-Control", "no-store")],
        Json(json!({"sort": sort, "posts": posts, "source": source})),
    )
        .into_response()
}
