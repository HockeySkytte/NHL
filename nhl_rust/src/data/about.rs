//! About-page content and helpers — port of the `_ABOUT_*` data and
//! `_about_*` helper functions in `app/routes.py` (~lines 4948–5622).
//!
//! The prose itself is extracted from the Python module into
//! `data/about.json` (via `scripts/export_models_for_rust.py`-style one-liner;
//! see PORT_PLAN.md) so it stays in sync without copying hundreds of lines
//! of copy into Rust source.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;
use serde_json::{json, Value};

#[derive(Debug, Clone, Deserialize)]
pub struct AboutData {
    pub headlines: Vec<String>,
    pub section_text: BTreeMap<String, String>,
    #[serde(default)]
    pub glossary_sections: Vec<Value>,
}

impl AboutData {
    pub fn load(path: &Path) -> Result<Self, String> {
        let raw = std::fs::read_to_string(path)
            .map_err(|e| format!("about data {}: {e}", path.display()))?;
        serde_json::from_str(&raw).map_err(|e| format!("about data {}: {e}", path.display()))
    }

    pub fn empty() -> Self {
        Self {
            headlines: vec!["About".to_string()],
            section_text: BTreeMap::new(),
            glossary_sections: Vec::new(),
        }
    }

    pub fn first_slug(&self) -> Option<String> {
        self.headlines.first().map(|t| slug_from_title(t))
    }

    pub fn valid_slug(&self, slug: &str) -> bool {
        self.headlines.iter().any(|t| slug_from_title(t) == slug)
    }

    pub fn is_glossary_slug(&self, slug: &str) -> bool {
        slug == slug_from_title("Glossary")
    }

    pub fn section_title(&self, slug: &str) -> Option<&str> {
        self.headlines
            .iter()
            .find(|t| slug_from_title(t) == slug)
            .map(String::as_str)
    }

    pub fn section_text(&self, slug: &str) -> &str {
        self.section_text.get(slug).map(String::as_str).unwrap_or("")
    }

    /// Port of `_about_nav_items(active_slug)`.
    pub fn nav_items(&self, active_slug: &str) -> Vec<Value> {
        self.headlines
            .iter()
            .map(|title| {
                let slug = slug_from_title(title);
                json!({
                    "title": title,
                    "slug": slug,
                    "url": format!("/about/{slug}"),
                    "active": slug == active_slug,
                })
            })
            .collect()
    }
}

/// Port of `_about_slug_from_title`: `re.sub(r'[^a-z0-9]+', '-', title.lower())`
/// then strip dashes.
pub fn slug_from_title(title: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in title.trim().to_lowercase().chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c);
            dash = false;
        } else if !dash {
            out.push('-');
            dash = true;
        }
    }
    out.trim_matches('-').to_string()
}

/// Port of `_about_strip_leading_heading(text, heading)`.
pub fn strip_leading_heading(text: &str, heading: &str) -> String {
    let raw = text;
    if raw.trim().is_empty() {
        return raw.to_string();
    }
    let lines: Vec<&str> = raw.lines().collect();
    if lines.is_empty() {
        return raw.to_string();
    }
    let first = lines[0].trim().trim_end_matches(':').to_lowercase();
    let expected = heading.trim().trim_end_matches(':').to_lowercase();
    if first != expected {
        return raw.to_string();
    }
    let mut idx = 1;
    while idx < lines.len() && lines[idx].trim().is_empty() {
        idx += 1;
    }
    lines[idx..].join("\n")
}

/// Port of `_about_text_segments(text)`: splits on `(PictureN)` /
/// `(live_games_image)` tokens into text and image segments.
pub fn text_segments(text: &str, static_dir: &Path) -> Vec<Value> {
    let mut segments = Vec::new();
    let mut pos = 0usize;
    while let Some((start, end, token)) = find_image_token(text, pos) {
        if start > pos {
            segments.push(json!({"type": "text", "value": &text[pos..start]}));
        }
        let filename = if token == "live_games_image" {
            "live_games_image.png".to_string()
        } else {
            format!("{token}.png")
        };
        let exists = static_dir.join("about").join(&filename).exists();
        segments.push(json!({
            "type": "image",
            "token": token,
            "filename": filename,
            "url": format!("/static/about/{filename}"),
            "exists": exists,
        }));
        pos = end;
    }
    if pos < text.len() {
        segments.push(json!({"type": "text", "value": &text[pos..]}));
    }
    segments
}

/// Scans for `(PictureN)` or `(live_games_image)` starting at `from`.
/// Returns `(start, end, token)` where `start..end` covers the full token
/// including parentheses. All returned offsets sit on `'('` positions, which
/// are always valid UTF-8 char boundaries.
fn find_image_token(text: &str, from: usize) -> Option<(usize, usize, &str)> {
    let bytes = text.as_bytes();
    let mut i = from;
    while i < bytes.len() {
        if bytes[i] == b'(' {
            let rest = &text[i + 1..];
            if let Some(stripped) = rest.strip_prefix("Picture") {
                let digits = stripped
                    .chars()
                    .take_while(|c| c.is_ascii_digit())
                    .count();
                if digits > 0 && stripped.as_bytes().get(digits) == Some(&b')') {
                    let token_start = i + 1;
                    let token_end = token_start + "Picture".len() + digits;
                    return Some((i, token_end + 1, &text[token_start..token_end]));
                }
            } else if let Some(stripped) = rest.strip_prefix("live_games_image") {
                if stripped.as_bytes().first() == Some(&b')') {
                    let token_start = i + 1;
                    let token_end = token_start + "live_games_image".len();
                    return Some((i, token_end + 1, &text[token_start..token_end]));
                }
            }
        }
        i += 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slugs_match_python_regex() {
        assert_eq!(slug_from_title("Inside the App"), "inside-the-app");
        assert_eq!(slug_from_title("  Why This App Matters  "), "why-this-app-matters");
        assert_eq!(slug_from_title("A Practical Way to Read the App"), "a-practical-way-to-read-the-app");
        assert_eq!(slug_from_title("5v5 & PP — Notes"), "5v5-pp-notes");
    }

    #[test]
    fn strip_leading_heading_works() {
        let text = "Inside the App\n\nSome body text.\nMore.";
        assert_eq!(strip_leading_heading(text, "Inside the App"), "Some body text.\nMore.");
        let untouched = "Different heading\nbody";
        assert_eq!(strip_leading_heading(untouched, "Inside the App"), untouched);
    }

    #[test]
    fn segments_parse_images() {
        let text = "Intro text.\n(Picture1)\nAfter.\n(live_games_image)\nTail.";
        let segments = text_segments(text, Path::new("/tmp/static"));
        assert_eq!(segments.len(), 5);
        assert_eq!(segments[0]["type"], "text");
        assert_eq!(segments[1]["type"], "image");
        assert_eq!(segments[1]["filename"], "Picture1.png");
        assert_eq!(segments[1]["url"], "/static/about/Picture1.png");
        assert_eq!(segments[3]["filename"], "live_games_image.png");
        assert_eq!(segments[4]["type"], "text");
    }
}
