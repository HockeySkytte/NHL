//! Team logo / player headshot proxy helpers (ports of `_team_logo_source_url`,
//! `_normalize_svg_dimensions`, `_player_headshot_source_urls`).

use serde_json::Value;

/// `_team_logo_source_url`: prefers the `Logo` column in Teams.csv rows, else
/// the standard assets.nhle.com light SVG.
pub fn team_logo_source_url(teams: &[Value], team_abbrev: &str) -> Option<String> {
    let a = team_abbrev.trim().to_uppercase();
    if a.is_empty() {
        return None;
    }
    for row in teams {
        let team = row
            .get("Team")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_uppercase();
        if team == a {
            let logo = row.get("Logo").and_then(Value::as_str).unwrap_or("").trim();
            if !logo.is_empty() {
                return Some(logo.to_string());
            }
        }
    }
    Some(format!("https://assets.nhle.com/logos/nhl/svg/{a}_light.svg"))
}

/// `_player_headshot_source_urls`.
pub fn player_headshot_source_urls(player_id: i64, season: &str, team_abbrev: &str) -> Vec<String> {
    if player_id <= 0 {
        return Vec::new();
    }
    let mut urls = Vec::new();
    let season_s = season.trim();
    let team_s = team_abbrev.trim().to_uppercase();
    let season_ok = season_s.len() == 8 && season_s.chars().all(|c| c.is_ascii_digit());
    let team_ok = (2..=4).contains(&team_s.len()) && team_s.chars().all(|c| c.is_ascii_alphabetic());
    if season_ok && team_ok {
        urls.push(format!("https://assets.nhle.com/mugs/nhl/{season_s}/{team_s}/{player_id}.png"));
    }
    urls.push(format!("https://assets.nhle.com/mugs/nhl/latest/{player_id}.png"));
    urls
}

/// `_normalize_svg_dimensions`: inject width/height when only viewBox exists.
pub fn normalize_svg_dimensions(svg_text: &str) -> String {
    let head = svg_text.floor_char_boundary(2048.min(svg_text.len()));
    let head_slice = &svg_text[..head];
    let lower = head_slice.to_ascii_lowercase();
    if lower.contains("width=") && lower.contains("height=") {
        return svg_text.to_string();
    }
    let Some((w, h)) = extract_viewbox(head_slice) else {
        return svg_text.to_string();
    };
    // Find the first `<svg ...>` open tag anywhere in the document.
    let lower_all = svg_text.to_ascii_lowercase();
    let Some(idx) = lower_all.find("<svg") else {
        return svg_text.to_string();
    };
    let rest = &svg_text[idx + 4..];
    let Some(close) = rest.find('>') else {
        return svg_text.to_string();
    };
    let attrs = &rest[..close];
    let attrs_lower = attrs.to_ascii_lowercase();
    if attrs_lower.contains("width=") || attrs_lower.contains("height=") {
        return svg_text.to_string();
    }
    let new_tag = format!("<svg{attrs} width=\"{w}\" height=\"{h}\">");
    let mut out = String::with_capacity(svg_text.len() + 24);
    out.push_str(&svg_text[..idx]);
    out.push_str(&new_tag);
    out.push_str(&rest[close + 1..]);
    out
}

/// Extracts `(width, height)` from a `viewBox="minx miny w h"` (spaces only,
/// matching the Python regex).
fn extract_viewbox(head: &str) -> Option<(String, String)> {
    let lower = head.to_ascii_lowercase();
    let pos = lower.find("viewbox")?;
    let after = &head[pos + "viewbox".len()..];
    let eq = after.find('=')?;
    let val = after[eq + 1..].trim_start();
    let quote = val.chars().next()?;
    if quote != '"' && quote != '\'' {
        return None;
    }
    let inner = &val[quote.len_utf8()..];
    let end = inner.find(quote)?;
    let content = &inner[..end];
    let parts: Vec<&str> = content.split_whitespace().collect();
    if parts.len() != 4 {
        return None;
    }
    Some((parts[2].to_string(), parts[3].to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn svg_dimension_injection() {
        let svg = r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120"><path/></svg>"#;
        let out = normalize_svg_dimensions(svg);
        assert!(out.contains("width=\"120\" height=\"120\""));
        assert!(out.starts_with("<svg"));

        // Already has width/height → untouched.
        let with_dims = r#"<svg width="100" height="100" viewBox="0 0 120 120">"#;
        assert_eq!(normalize_svg_dimensions(with_dims), with_dims);
    }

    #[test]
    fn headshot_urls() {
        let urls = player_headshot_source_urls(8471214, "20252026", "bos");
        assert_eq!(urls.len(), 2);
        assert!(urls[0].contains("20252026/BOS/8471214.png"));
        assert!(urls[1].contains("latest/8471214.png"));
        let urls = player_headshot_source_urls(8471214, "", "");
        assert_eq!(urls.len(), 1);
    }

    #[test]
    fn logo_source_prefers_teams_map() {
        let teams = serde_json::json!([
            {"Team": "BOS", "Logo": "https://custom.example/logo.svg"},
            {"Team": "NYR", "Logo": ""}
        ]);
        let rows: Vec<Value> = serde_json::from_value(teams).unwrap();
        assert_eq!(
            team_logo_source_url(&rows, "bos"),
            Some("https://custom.example/logo.svg".to_string())
        );
        assert_eq!(
            team_logo_source_url(&rows, "nyr"),
            Some("https://assets.nhle.com/logos/nhl/svg/NYR_light.svg".to_string())
        );
        assert_eq!(team_logo_source_url(&rows, "TBL"), Some("https://assets.nhle.com/logos/nhl/svg/TBL_light.svg".to_string()));
    }
}
