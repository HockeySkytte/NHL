//! Shifts HTML parsing — port of `api_game_shifts` from `app/routes.py`
//! (the three parsing strategies, name/jersey normalization, slicing and
//! strength-state bucketing). Uses `scraper` for the BeautifulSoup strategies
//! and `regex` for the final fallback.

use std::collections::{HashMap, HashSet};

use regex::Regex;
use scraper::{ElementRef, Html, Node, Selector};
use serde_json::{json, Value};

use crate::util::parse::safe_int;
use crate::util::parse::str_value;

/// `_normalize_jersey`: keep only digits, drop leading zeros.
fn normalize_jersey(s: &str) -> String {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        String::new()
    } else {
        digits.trim_start_matches('0').to_string()
    }
}

/// `_strip_diacritics` (NFKD, drop combining marks).
fn strip_diacritics(text: &str) -> String {
    use unicode_normalization::UnicodeNormalization;
    text.nfkd().filter(|c| !is_combining_mark_char(*c)).collect()
}

fn is_combining_mark_char(c: char) -> bool {
    use unicode_normalization::char::is_combining_mark;
    is_combining_mark(c)
}

/// `_strip_parentheticals_local`: remove ` (anything)`.
fn strip_parentheticals(s: &str) -> String {
    lazy_re_sub(r"\s*\([^)]*\)", "", s).trim().to_string()
}

fn collapse_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn lazy_re_sub(pattern: &str, replacement: &str, text: &str) -> String {
    match Regex::new(pattern) {
        Ok(re) => re.replace_all(text, replacement).into_owned(),
        Err(_) => text.to_string(),
    }
}

fn lazy_re_match<'t>(pattern: &str, text: &'t str) -> Option<regex::Captures<'t>> {
    Regex::new(pattern).ok()?.captures(text)
}

/// `norm_name`.
fn norm_name(s: &str) -> String {
    let t = s.replace('\u{a0}', " ").trim().to_string();
    let t = strip_parentheticals(&t);
    let t = if let Some((a, b)) = t.split_once(',') {
        format!("{} {}", b.trim(), a.trim())
    } else {
        t
    };
    let t = t.replace('.', " ").replace('\'', "").replace('-', " ");
    let t = collapse_ws(&t);
    strip_diacritics(&t).to_lowercase()
}

/// `last_token_norm`.
fn last_token_norm(name: &str) -> String {
    let base = norm_name(name);
    if base.is_empty() {
        return String::new();
    }
    let mut toks: Vec<&str> = base.split(' ').collect();
    let suffixes = ["jr", "sr", "ii", "iii", "iv", "v"];
    while let Some(last) = toks.last() {
        let cleaned = last.trim_matches('.').to_lowercase();
        if suffixes.contains(&cleaned.as_str()) {
            toks.pop();
        } else {
            break;
        }
    }
    toks.last().map(|s| s.to_string()).unwrap_or_default()
}

/// `proper_name(last_upper, first_upper)`: unicode-aware title-casing.
fn proper_name(last_upper: &str, first_upper: &str) -> String {
    fn fix(part: &str) -> String {
        let part = part.trim().replace('\u{a0}', " ");
        let mut tokens = Vec::new();
        for tok in part.split(' ') {
            if tok.is_empty() {
                continue;
            }
            // Split on apostrophe/hyphen, title-case alpha subtokens.
            let mut out = String::new();
            let mut cur = String::new();
            for ch in tok.chars() {
                if ch == '\'' || ch == '-' {
                    if !cur.is_empty() {
                        out.push_str(&title_case(&cur));
                        cur.clear();
                    }
                    out.push(ch);
                } else {
                    cur.push(ch);
                }
            }
            if !cur.is_empty() {
                out.push_str(&title_case(&cur));
            }
            tokens.push(out);
        }
        tokens.join(" ")
    }
    let f = fix(first_upper);
    let l = fix(last_upper);
    if f.is_empty() && l.is_empty() {
        String::new()
    } else {
        format!("{f} {l}").trim().to_string()
    }
}

fn title_case(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(first) => {
            let mut out = String::new();
            out.extend(first.to_uppercase());
            out.extend(chars.flat_map(|c| c.to_lowercase()));
            out
        }
        None => String::new(),
    }
}

/// `to_seconds`: `MM:SS` (optionally `MM:SS/OVERTIME`).
fn to_seconds(ts: &str) -> Option<i64> {
    let ts = ts.trim();
    if ts.is_empty() {
        return None;
    }
    let ts = ts.split('/').next().unwrap_or(ts).trim();
    let re = Regex::new(r"^(\d{1,2}):(\d{2})$").ok()?;
    let caps = re.captures(ts)?;
    let mm: i64 = caps.get(1)?.as_str().parse().ok()?;
    let ss: i64 = caps.get(2)?.as_str().parse().ok()?;
    Some(mm * 60 + ss)
}

/// `parse_period_value`: numeric -> int, "OT" -> 4, "SO"/other -> None.
fn parse_period_value(p: Option<&str>) -> Option<i64> {
    let s = p?.trim().to_uppercase();
    if s.is_empty() {
        return None;
    }
    if s == "OT" {
        return Some(4);
    }
    if s == "SO" {
        return None;
    }
    s.parse::<i64>().ok()
}

#[derive(Clone, Debug)]
pub struct RosterEntry {
    pub player_id: Option<i64>,
    pub name: String,
    pub sweater_number: String,
    pub pos: String,
}

/// `unify_roster(team_stats)`: forwards/defense/goalies -> RosterEntry.
fn unify_roster(team_stats: &Value) -> Vec<RosterEntry> {
    let mut res = Vec::new();
    for grp in ["forwards", "defense", "goalies"] {
        if let Some(players) = team_stats.get(grp).and_then(|v| v.as_array()) {
            for p in players {
                let mut nm = str_value(p.get("name"));
                if nm.is_empty() {
                    if let Some(d) = p.get("name").and_then(|v| v.as_object()) {
                        nm = str_value(d.get("default"));
                    }
                }
                let raw_pos = str_value(p.get("position"));
                let raw_pos = if raw_pos.is_empty() {
                    str_value(p.get("positionCode"))
                } else {
                    raw_pos
                }
                .to_uppercase();
                let pos = if !raw_pos.is_empty() {
                    let code = raw_pos.chars().next().unwrap_or('F');
                    if matches!(code, 'C' | 'L' | 'R') {
                        "F".to_string()
                    } else {
                        code.to_string()
                    }
                } else {
                    match grp {
                        "forwards" => "F".to_string(),
                        "defense" => "D".to_string(),
                        _ => "G".to_string(),
                    }
                    .to_string()
                };
                let sweater = str_value(p.get("sweaterNumber"));
                let sweater = if sweater.is_empty() {
                    let s2 = str_value(p.get("sweater"));
                    if s2.is_empty() {
                        str_value(p.get("jersey"))
                    } else {
                        s2
                    }
                } else {
                    sweater
                };
                res.push(RosterEntry {
                    player_id: safe_int(p.get("playerId")),
                    name: nm,
                    sweater_number: sweater.trim().to_string(),
                    pos,
                });
            }
        }
    }
    res
}

#[derive(Default)]
pub struct Indices {
    by_num: HashMap<String, RosterEntry>,
    by_name: HashMap<String, RosterEntry>,
    by_last: HashMap<String, Vec<RosterEntry>>,
}

/// `build_indices(roster)`.
fn build_indices(roster: &[RosterEntry]) -> Indices {
    let mut idx = Indices::default();
    for p in roster {
        let num = normalize_jersey(&p.sweater_number);
        if !num.is_empty() {
            idx.by_num.insert(num, p.clone());
        }
        let nm = norm_name(&p.name);
        if !nm.is_empty() {
            idx.by_name.insert(nm.clone(), p.clone());
            idx.by_last
                .entry(last_token_norm(&nm))
                .or_default()
                .push(p.clone());
        }
    }
    idx
}

/// Text of an element joined like `get_text(' ', strip=True)`.
fn text_of(el: &ElementRef) -> String {
    collapse_ws(&el.text().collect::<Vec<_>>().join(" "))
}

fn text_nodes(html: &Html) -> Vec<String> {
    let mut out = Vec::new();
    for node in html.tree.nodes() {
        if let Node::Text(t) = node.value() {
            let s = t.text.trim().replace('\u{a0}', " ");
            if !s.is_empty() {
                out.push(s);
            }
        }
    }
    out
}

fn find_next_table<'a>(el: ElementRef<'a>) -> Option<ElementRef<'a>> {
    let mut cur = el.next_sibling();
    while let Some(node) = cur {
        if let Node::Element(e) = node.value() {
            if e.name() == "table" {
                return ElementRef::wrap(node);
            }
        }
        cur = node.next_sibling();
    }
    None
}

fn resolve_player(
    jersey: &str,
    name: &str,
    idx: &Indices,
) -> (Option<i64>, String) {
    let mut pid = None;
    let mut pos = String::new();
    if !jersey.is_empty() {
        if let Some(p) = idx.by_num.get(&normalize_jersey(jersey)) {
            pid = p.player_id;
            pos = p.pos.clone();
        }
    }
    if pid.is_none() && !name.is_empty() {
        if let Some(p) = idx.by_name.get(&norm_name(name)) {
            pid = p.player_id;
            pos = p.pos.clone();
        }
    }
    if pid.is_none() {
        let last_tok = last_token_norm(name);
        if !last_tok.is_empty() {
            if let Some(cands) = idx.by_last.get(&last_tok) {
                if cands.len() == 1 {
                    pid = cands[0].player_id;
                    pos = cands[0].pos.clone();
                } else {
                    let nj = normalize_jersey(jersey);
                    for cand in cands {
                        if normalize_jersey(&cand.sweater_number) == nj {
                            pid = cand.player_id;
                            pos = cand.pos.clone();
                            break;
                        }
                    }
                }
            }
        }
    }
    (pid, pos)
}

fn canonical_name_for(pid: Option<i64>, fallback: &str, name_by_id: &HashMap<i64, String>) -> String {
    if let Some(pid) = pid {
        if let Some(n) = name_by_id.get(&pid) {
            return n.clone();
        }
    }
    fallback.to_string()
}

#[derive(Clone, Debug)]
struct ShiftRow {
    player_id: Option<i64>,
    name: String,
    position: String,
    team: String,
    period: i64,
    start: i64,
    end: i64,
}

/// Strategy 1: content-table scan (primary).
fn content_table_strategy(
    html: &Html,
    idx: &Indices,
    name_by_id: &HashMap<i64, String>,
    team_abbrev: &str,
) -> Vec<ShiftRow> {
    let tr_sel = Selector::parse("tr").unwrap();
    let td_sel = Selector::parse("td").unwrap();
    let table_sel = Selector::parse("table").unwrap();

    let all_tr: Vec<ElementRef> = html.select(&tr_sel).collect();
    let content_tbl: Option<ElementRef> = if all_tr.len() >= 4 {
        let tr3 = &all_tr[3];
        let tds: Vec<ElementRef> = tr3.select(&td_sel).collect();
        tds.first()
            .and_then(|td| td.select(&table_sel).next())
            .or_else(|| find_content_table_heuristic(html))
    } else {
        find_content_table_heuristic(html)
    };

    let Some(tbl) = content_tbl else { return Vec::new() };

    let mut out: Vec<ShiftRow> = Vec::new();
    let mut current_name: Option<String> = None;
    let mut current_jersey = String::new();
    let mut current_pid: Option<i64> = None;
    let mut current_pos = String::new();

    for tr in tbl.select(&tr_sel) {
        let tds_all: Vec<ElementRef> = tr.select(&td_sel).collect();
        if tds_all.is_empty() {
            continue;
        }
        if tds_all.len() == 1 && tds_all[0].value().attr("colspan").is_some() {
            let txt = text_of(&tds_all[0]);
            let txt2 = strip_parentheticals(&txt);
            let Ok(re1) = Regex::new(r"^(\d{1,2})\s+([A-Z\x{c0}-\x{d6}\x{d8}-\x{de} .'-]+),\s*([A-Z\x{c0}-\x{d6}\x{d8}-\x{de} .'-]+)$") else { return Vec::new() };
            let Ok(re2) = Regex::new(r"^(\d{1,2})\s+([A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff} .'-]+)$") else { return Vec::new() };
            if let Some(caps) = re1.captures(&txt2) {
                current_jersey = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let last_u = caps.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
                let first_u = strip_parentheticals(caps.get(3).map(|m| m.as_str()).unwrap_or(""));
                current_name = Some(proper_name(&last_u, &first_u));
            } else if let Some(caps) = re2.captures(&txt2) {
                current_jersey = caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default();
                let name_plain = strip_parentheticals(caps.get(2).map(|m| m.as_str()).unwrap_or(""));
                let parts: Vec<&str> = name_plain.split_whitespace().collect();
                let name = parts
                    .iter()
                    .map(|p| title_case(p))
                    .collect::<Vec<_>>()
                    .join(" ");
                current_name = Some(name);
            } else {
                current_name = None;
                current_jersey.clear();
            }
            current_pid = None;
            current_pos = String::new();
            if !current_jersey.is_empty() || current_name.is_some() {
                let name = current_name.clone().unwrap_or_default();
                let (pid, pos) = resolve_player(&current_jersey, &name, idx);
                current_pid = pid;
                current_pos = pos;
            }
            continue;
        }

        // Data rows: skip colspan/rowspan cells.
        let tds: Vec<ElementRef> = tds_all
            .into_iter()
            .filter(|td| td.value().attr("colspan").is_none() && td.value().attr("rowspan").is_none())
            .collect();
        if tds.len() < 4 {
            continue;
        }
        let ctext: Vec<String> = tds.iter().take(6).map(text_of).collect();
        let shift_no = ctext.get(0).cloned().unwrap_or_default().trim().to_string();
        let per_txt = ctext.get(1).cloned().unwrap_or_default();
        let start_txt = ctext.get(2).cloned().unwrap_or_default();
        let end_txt = ctext.get(3).cloned().unwrap_or_default();
        let per_val = parse_period_value(Some(&per_txt));
        if !shift_no.chars().all(|c| c.is_ascii_digit()) || per_val.is_none() {
            continue;
        }
        let (Some(start_sec), Some(end_sec)) = (to_seconds(&start_txt), to_seconds(&end_txt)) else {
            continue;
        };
        if current_pid.is_none() && current_name.is_none() {
            continue;
        }
        let name_out = canonical_name_for(current_pid, current_name.as_deref().unwrap_or(""), name_by_id);
        out.push(ShiftRow {
            player_id: current_pid,
            name: name_out,
            position: current_pos.clone(),
            team: team_abbrev.to_string(),
            period: per_val.unwrap(),
            start: start_sec,
            end: end_sec,
        });
    }
    out
}

fn find_content_table_heuristic(html: &Html) -> Option<ElementRef> {
    let table_sel = Selector::parse("table").unwrap();
    let tr_sel = Selector::parse("tr").unwrap();
    let td_sel = Selector::parse("td").unwrap();
    let th_sel = Selector::parse("th").unwrap();
    for tbl in html.select(&table_sel) {
        let rows: Vec<ElementRef> = tbl.select(&tr_sel).take(15).collect();
        for tr in &rows {
            let mut texts: Vec<String> = Vec::new();
            for c in tr.select(&td_sel).chain(tr.select(&th_sel)) {
                texts.push(text_of(&c).to_lowercase());
            }
            if texts.is_empty() {
                continue;
            }
            let has_shift = texts.iter().any(|t| t.contains("shift"));
            let has_per = texts
                .iter()
                .any(|t| t == "per" || t.contains("period") || t.starts_with("per"));
            if has_shift && has_per {
                return Some(tbl);
            }
        }
        let has_colspan = tbl
            .select(&td_sel)
            .any(|td| td.value().attr("colspan").is_some());
        if has_colspan {
            let mut dense = false;
            for tr in &rows {
                let tds: Vec<ElementRef> = tr
                    .select(&td_sel)
                    .filter(|td| td.value().attr("colspan").is_none() && td.value().attr("rowspan").is_none())
                    .collect();
                if tds.len() >= 6 {
                    dense = true;
                    break;
                }
            }
            if dense {
                return Some(tbl);
            }
        }
    }
    None
}

/// Strategy 2: player-header scan.
fn header_scan_strategy(
    html: &Html,
    idx: &Indices,
    name_by_id: &HashMap<i64, String>,
    team_abbrev: &str,
) -> Vec<ShiftRow> {
    let tr_sel = Selector::parse("tr").unwrap();
    let td_sel = Selector::parse("td").unwrap();
    let th_sel = Selector::parse("th").unwrap();
    let Ok(re1) = Regex::new(r"^(\d{1,2})\s+([A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff} .'-]+),\s*([A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff} .'-]+)$") else { return Vec::new() };
    let Ok(re2) = Regex::new(r"^(\d{1,2})\s+([A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff}][A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff} .'-]+)$") else { return Vec::new() };

    let mut results: Vec<ShiftRow> = Vec::new();
    // Find matching text nodes and their parent elements.
    let mut candidates: Vec<ElementRef> = Vec::new();
    for node in html.tree.nodes() {
        if let Node::Text(t) = node.value() {
            let txt = t.text.trim().replace('\u{a0}', " ");
            if txt.is_empty() {
                continue;
            }
            let txt2 = strip_parentheticals(&txt);
            if re1.is_match(&txt2) || re2.is_match(&txt2) {
                if let Some(parent) = node.parent().and_then(|p| ElementRef::wrap(p)) {
                    candidates.push(parent);
                }
            }
        }
    }

    for node in candidates {
        let raw = text_of(&node);
        let raw2 = strip_parentheticals(&raw);
        let (jersey, disp_name, last_for_idx) = if let Some(caps) = re1.captures(&raw2) {
            (
                caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                {
                    let last_u = caps.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
                    let first_u = strip_parentheticals(caps.get(3).map(|m| m.as_str()).unwrap_or(""));
                    proper_name(&last_u, &first_u)
                },
                last_token_norm(caps.get(2).map(|m| m.as_str()).unwrap_or("")),
            )
        } else if let Some(caps) = re2.captures(&raw2) {
            let name_plain = strip_parentheticals(caps.get(2).map(|m| m.as_str()).unwrap_or(""));
            let parts: Vec<&str> = name_plain.split_whitespace().collect();
            let name = parts
                .iter()
                .map(|p| title_case(p))
                .collect::<Vec<_>>()
                .join(" ");
            let last = parts.last().map(|p| p.to_string()).unwrap_or_default();
            (
                caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                name,
                last_token_norm(&last),
            )
        } else {
            continue;
        };

        let Some(tbl) = find_next_table(node) else { continue };
        let trs: Vec<ElementRef> = tbl.select(&tr_sel).collect();
        if trs.is_empty() {
            continue;
        }
        let mut header_row_idx: Option<usize> = None;
        let mut i_shift: i64 = -1;
        let mut i_per: i64 = -1;
        let mut i_start: i64 = -1;
        let mut i_end: i64 = -1;

        for (ridx, tr) in trs.iter().take(6).enumerate() {
            let cells: Vec<String> = tr
                .select(&td_sel)
                .chain(tr.select(&th_sel))
                .map(|c| text_of(&c).to_lowercase())
                .collect();
            if cells.is_empty() {
                continue;
            }
            let idx_of = |parts: &[&str]| -> i64 {
                for (i, h) in cells.iter().enumerate() {
                    if parts.iter().all(|p| h.contains(p)) {
                        return i as i64;
                    }
                }
                -1
            };
            i_shift = idx_of(&["shift"]);
            let i_per2 = idx_of(&["per"]);
            i_per = if i_per2 >= 0 { i_per2 } else { idx_of(&["period"]) };
            i_start = idx_of(&["start"]);
            i_end = idx_of(&["end"]);
            if i_shift >= 0 && i_per >= 0 && i_start >= 0 && i_end >= 0 {
                header_row_idx = Some(ridx);
                break;
            }
        }
        let Some(hri) = header_row_idx else { continue };

        let (pid, pos_val) = resolve_player(&jersey, &disp_name, idx);
        let mut name_for_idx = disp_name.clone();
        if let Some(p) = idx.by_num.get(&normalize_jersey(&jersey)) {
            if p.player_id.is_some() {
                // resolved via jersey already; fall back to last-name resolution below
            }
        }
        let mut pid_final = pid;
        let mut pos_final = pos_val;
        if pid_final.is_none() && !last_for_idx.is_empty() {
            if let Some(cands) = idx.by_last.get(&last_for_idx) {
                if cands.len() == 1 {
                    pid_final = cands[0].player_id;
                    pos_final = cands[0].pos.clone();
                } else {
                    let nj = normalize_jersey(&jersey);
                    for cand in cands {
                        if normalize_jersey(&cand.sweater_number) == nj {
                            pid_final = cand.player_id;
                            pos_final = cand.pos.clone();
                            break;
                        }
                    }
                }
            }
        }

        for tr in trs.iter().skip(hri + 1) {
            let tds: Vec<String> = tr.select(&td_sel).map(|td| text_of(&td)).collect();
            if tds.len() <= (i_shift.max(i_per).max(i_start).max(i_end)) as usize {
                continue;
            }
            let per_val = parse_period_value(Some(tds[i_per as usize].trim()));
            let shift_no = tds[i_shift as usize].trim().to_string();
            if !shift_no.chars().all(|c| c.is_ascii_digit()) || per_val.is_none() {
                continue;
            }
            let (Some(start_sec), Some(end_sec)) = (
                to_seconds(&tds[i_start as usize]),
                to_seconds(&tds[i_end as usize]),
            ) else {
                continue;
            };
            let name_out = canonical_name_for(pid_final, &name_for_idx, name_by_id);
            if pid_final.is_none() && name_out.is_empty() {
                continue;
            }
            results.push(ShiftRow {
                player_id: pid_final,
                name: name_out,
                position: pos_final.clone(),
                team: team_abbrev.to_string(),
                period: per_val.unwrap(),
                start: start_sec,
                end: end_sec,
            });
        }
    }
    results
}

/// Strategy 3: regex fallback.
fn regex_strategy(
    html: &str,
    idx: &Indices,
    name_by_id: &HashMap<i64, String>,
    team_abbrev: &str,
) -> Vec<ShiftRow> {
    fn strip_tags(s: &str) -> String {
        let s = lazy_re_sub(r"<[^>]+>", " ", s);
        collapse_ws(&s)
    }

    let re_colspan = Regex::new(r#"<td[^>]*colspan="?\d+"?[^>]*>\s*(.*?)\s*</td>"#).unwrap();
    let re_row = Regex::new(r"(?is)<tr[^>]*>\s*(.*?)\s*</tr>").unwrap();
    let re_cell = Regex::new(r"(?is)<t[dh][^>]*>\s*(.*?)\s*</t[dh]>").unwrap();
    let re_colspan_in_row = Regex::new(r"(?i)<td[^>]*colspan=").unwrap();
    let Ok(re1) = Regex::new(r"^(\d{1,2})\s+([A-Z\x{c0}-\x{d6}\x{d8}-\x{de} .'-]+),\s*([A-Z\x{c0}-\x{d6}\x{d8}-\x{de} .'-]+)$") else { return Vec::new() };
    let Ok(re2) = Regex::new(r"^(\d{1,2})\s+([A-Za-z\x{c0}-\x{d6}\x{d8}-\x{f6}\x{f8}-\x{ff} .'-]+)$") else { return Vec::new() };

    let mut positions: Vec<(usize, usize, String)> = Vec::new();
    for m in re_colspan.captures_iter(html) {
        let start = m.get(0).map(|x| x.start()).unwrap_or(0);
        let end = m.get(0).map(|x| x.end()).unwrap_or(0);
        let group = m.get(1).map(|x| x.as_str().to_string()).unwrap_or_default();
        positions.push((start, end, group));
    }
    positions.push((html.len(), html.len(), String::new()));

    let mut out: Vec<ShiftRow> = Vec::new();
    for i in 0..positions.len().saturating_sub(1) {
        let (start, end, header_html) = &positions[i];
        let next_start = positions[i + 1].0;
        let header_text = strip_tags(header_html);
        let header_text2 = strip_parentheticals(&header_text);
        let (jersey, disp_name, last_for_idx) = if let Some(caps) = re1.captures(&header_text2) {
            (
                caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                {
                    let last_u = caps.get(2).map(|m| m.as_str().to_string()).unwrap_or_default();
                    let first_u = strip_parentheticals(caps.get(3).map(|m| m.as_str()).unwrap_or(""));
                    proper_name(&last_u, &first_u)
                },
                last_token_norm(caps.get(2).map(|m| m.as_str()).unwrap_or("")),
            )
        } else if let Some(caps) = re2.captures(&header_text2) {
            let name_plain = strip_parentheticals(caps.get(2).map(|m| m.as_str()).unwrap_or(""));
            let parts: Vec<&str> = name_plain.split_whitespace().collect();
            let name = parts
                .iter()
                .map(|p| title_case(p))
                .collect::<Vec<_>>()
                .join(" ");
            let last = parts.last().map(|p| p.to_string()).unwrap_or_default();
            (
                caps.get(1).map(|m| m.as_str().to_string()).unwrap_or_default(),
                name,
                last_token_norm(&last),
            )
        } else {
            continue;
        };

        let (pid, pos_val) = resolve_player(&jersey, &disp_name, idx);
        let section_html = &html[*end..next_start];
        for row_caps in re_row.captures_iter(section_html) {
            let row_html = row_caps.get(1).map(|m| m.as_str()).unwrap_or("");
            if re_colspan_in_row.is_match(row_html) {
                continue;
            }
            let cells_html: Vec<&str> = re_cell
                .captures_iter(row_html)
                .filter_map(|c| c.get(1).map(|m| m.as_str()))
                .collect();
            let cells: Vec<String> = cells_html.iter().map(|c| strip_tags(c)).collect();
            if cells.len() < 4 {
                continue;
            }
            let shift_no = cells[0].trim().to_string();
            let per_txt = cells[1].trim().to_string();
            let start_txt = cells[2].trim().to_string();
            let end_txt = cells[3].trim().to_string();
            let per_val = parse_period_value(Some(&per_txt));
            if !shift_no.chars().all(|c| c.is_ascii_digit()) || per_val.is_none() {
                continue;
            }
            let (Some(start_sec), Some(end_sec)) = (to_seconds(&start_txt), to_seconds(&end_txt)) else {
                continue;
            };
            let name_out = canonical_name_for(pid, &disp_name, name_by_id);
            if pid.is_none() && name_out.is_empty() {
                continue;
            }
            out.push(ShiftRow {
                player_id: pid,
                name: name_out,
                position: pos_val.clone(),
                team: team_abbrev.to_string(),
                period: per_val.unwrap(),
                start: start_sec,
                end: end_sec,
            });
        }
    }
    out
}

fn parse_shifts_from_html(
    html: &str,
    idx: &Indices,
    name_by_id: &HashMap<i64, String>,
    team_abbrev: &str,
) -> Vec<ShiftRow> {
    if html.is_empty() {
        return Vec::new();
    }
    let doc = Html::parse_document(html);
    let s1 = content_table_strategy(&doc, idx, name_by_id, team_abbrev);
    if !s1.is_empty() {
        return s1;
    }
    let s2 = header_scan_strategy(&doc, idx, name_by_id, team_abbrev);
    if !s2.is_empty() {
        return s2;
    }
    regex_strategy(html, idx, name_by_id, team_abbrev)
}

/// `_normalize_strength_state` (port of the inner function in `api_game_shifts`).
fn normalize_strength_state(my_s: i64, their_s: i64, my_g: i64, their_g: i64) -> String {
    let observed_goalies = (my_g + their_g) > 0;
    if observed_goalies && my_g == 0 && their_g >= 1 {
        return "ENF".to_string();
    }
    if observed_goalies && their_g == 0 && my_g >= 1 {
        return "ENA".to_string();
    }
    let mut ms = my_s.clamp(0, 6);
    let mut ts = their_s.clamp(0, 6);
    if ms == 6 && ts == 6 {
        return "5v5".to_string();
    }
    if (ms, ts) == (6, 5) || (ms, ts) == (5, 6) {
        return "5v5".to_string();
    }
    if (ms, ts) == (6, 4) {
        return "PP".to_string();
    }
    if (ms, ts) == (4, 6) {
        return "SH".to_string();
    }
    if ms == 6 || ts == 6 {
        return format!("{ms}v{ts}");
    }
    if ts == 4 && ms == 5 {
        return "5v4".to_string();
    }
    if ts == 3 && ms == 5 {
        return "5v3".to_string();
    }
    if ts == 3 && ms == 4 {
        return "4v3".to_string();
    }
    if ms == 4 && ts == 5 {
        return "4v5".to_string();
    }
    if ms == 3 && ts == 5 {
        return "3v5".to_string();
    }
    if ms == 3 && ts == 4 {
        return "3v4".to_string();
    }
    if ms == 2 && (3..=5).contains(&ts) {
        return format!("2v{ts}");
    }
    if ts == 2 && (3..=5).contains(&ms) {
        return format!("{ms}v2");
    }
    if ms == 4 && ts == 4 {
        return "4v4".to_string();
    }
    if ms == 3 && ts == 3 {
        return "3v3".to_string();
    }
    if ms == 5 && ts == 5 {
        return "5v5".to_string();
    }
    if ts == 0 && ms >= 1 {
        return "1v0".to_string();
    }
    if ms == 0 && ts >= 1 {
        return "0v1".to_string();
    }
    let m = ms.max(ts);
    if m <= 3 {
        return "3v3".to_string();
    }
    if m == 4 {
        return "4v4".to_string();
    }
    "5v5".to_string()
}

/// Full shifts computation for a game — port of the body of `api_game_shifts`.
/// `pages`: (away_html, home_html). Returns the `shifts` array.
pub fn compute_shifts_from_html(
    game_id: i64,
    away_html: &str,
    home_html: &str,
    boxscore: &Value,
) -> Value {
    let pbg = boxscore.get("playerByGameStats").cloned().unwrap_or(Value::Null);
    let away_team = pbg.get("awayTeam").cloned().unwrap_or(Value::Null);
    let home_team = pbg.get("homeTeam").cloned().unwrap_or(Value::Null);
    let roster_home = unify_roster(&home_team);
    let roster_away = unify_roster(&away_team);

    let mut name_by_id: HashMap<i64, String> = HashMap::new();
    for p in roster_home.iter().chain(roster_away.iter()) {
        if let Some(pid) = p.player_id {
            if !p.name.is_empty() {
                name_by_id.insert(pid, p.name.clone());
            }
        }
    }

    let idx_home = build_indices(&roster_home);
    let idx_away = build_indices(&roster_away);

    let away_abbrev = boxscore
        .get("awayTeam")
        .and_then(|t| t.get("abbrev"))
        .and_then(|a| a.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "AWY".to_string());
    let home_abbrev = boxscore
        .get("homeTeam")
        .and_then(|t| t.get("abbrev"))
        .and_then(|a| a.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| "HME".to_string());

    let mut shifts_out: Vec<ShiftRow> = Vec::new();
    shifts_out.extend(parse_shifts_from_html(away_html, &idx_away, &name_by_id, &away_abbrev));
    shifts_out.extend(parse_shifts_from_html(home_html, &idx_home, &name_by_id, &home_abbrev));

    // Build entries with game-time seconds and boundaries.
    #[derive(Clone)]
    struct Entry {
        gs: i64,
        ge: i64,
        player_id: Option<i64>,
        name: String,
        position: String,
        team: String,
    }
    let mut entries: Vec<Entry> = Vec::new();
    let mut boundaries: HashSet<i64> = HashSet::new();
    let mut max_end: i64 = 0;
    for row in shifts_out {
        let per = if row.period > 0 { row.period } else { 1 };
        let gs = (per - 1) * 1200 + row.start.max(0);
        let ge = (per - 1) * 1200 + row.end.max(0);
        if ge <= gs {
            continue;
        }
        boundaries.insert(gs);
        boundaries.insert(ge);
        if ge > max_end {
            max_end = ge;
        }
        entries.push(Entry {
            gs,
            ge,
            player_id: row.player_id,
            name: row.name.clone(),
            position: row.position.clone(),
            team: row.team.clone(),
        });
    }

    if entries.is_empty() {
        return json!([]);
    }

    let mut times: Vec<i64> = boundaries.into_iter().collect();
    times.sort_unstable();
    if times.is_empty() || times[times.len() - 1] != max_end {
        times.push(max_end);
        times.sort_unstable();
    }

    let mut split_rows: Vec<Value> = Vec::new();
    for i in 0..times.len().saturating_sub(1) {
        let s = times[i];
        let e = times[i + 1];
        if e <= s {
            continue;
        }
        let shift_index = game_id * 10000 + (i as i64 + 1);
        let active: Vec<&Entry> = entries.iter().filter(|rec| rec.gs <= s && s < rec.ge).collect();
        if active.is_empty() {
            continue;
        }

        let mut team_players: HashMap<String, (HashSet<i64>, HashSet<String>)> = HashMap::new();
        for rec in &active {
            let team = rec.team.clone();
            let pos = rec.position.to_uppercase();
            let tp = team_players.entry(team).or_insert((HashSet::new(), HashSet::new()));
            let key = rec.player_id.map(|p| p.to_string()).unwrap_or_else(|| rec.name.clone());
            if key.is_empty() {
                continue;
            }
            if pos == "G" {
                if let Some(pid) = rec.player_id {
                    tp.0.insert(pid);
                }
            } else {
                if let Some(pid) = rec.player_id {
                    tp.1.insert(pid.to_string());
                }
            }
        }

        let mut team_counts_raw: HashMap<String, (i64, i64)> = HashMap::new();
        let mut team_counts_clamped: HashMap<String, (i64, i64)> = HashMap::new();
        for (t, (g_set, s_set)) in &team_players {
            let g_raw = g_set.len() as i64;
            let s_raw = s_set.len() as i64;
            team_counts_raw.insert(t.clone(), (g_raw, s_raw));
            team_counts_clamped.insert(t.clone(), (g_raw.clamp(0, 1), s_raw.clamp(0, 6)));
        }

        for rec in &active {
            let team = rec.team.clone();
            let opp = if team == away_abbrev {
                home_abbrev.clone()
            } else if team == home_abbrev {
                away_abbrev.clone()
            } else if team.eq_ignore_ascii_case("away") {
                "Home".to_string()
            } else if team.eq_ignore_ascii_case("home") {
                "Away".to_string()
            } else {
                team_counts_raw
                    .keys()
                    .find(|t| **t != team)
                    .cloned()
                    .unwrap_or_default()
            };

            let my_raw = team_counts_raw.get(&team).copied().unwrap_or((0, 0));
            let their_raw = team_counts_raw.get(&opp).copied().unwrap_or((0, 0));
            let my = team_counts_clamped.get(&team).copied().unwrap_or((0, 0));
            let their = team_counts_clamped.get(&opp).copied().unwrap_or((0, 0));

            let my_g = my.0;
            let their_g = their.0;
            let my_s = my.1;
            let their_s = their.1;

            let both_goalies_in = (my_g >= 1 && their_g >= 1) || (my_g == 0 && their_g == 0);
            let strength_bucket = if both_goalies_in && my_s >= 5 && their_s >= 5 {
                "5v5".to_string()
            } else if both_goalies_in && (their_s == 3 || their_s == 4) && my_s > their_s {
                "PP".to_string()
            } else if both_goalies_in && (my_s == 3 || my_s == 4) && their_s > my_s {
                "SH".to_string()
            } else {
                "Other".to_string()
            };

            let strength = normalize_strength_state(my_s, their_s, my_g, their_g);
            let strength_raw = format!("{}v{}", my_raw.1, their_raw.1);
            let period_calc = 1 + s / 1200;

            split_rows.push(json!({
                "ShiftIndex": shift_index,
                "PlayerID": rec.player_id,
                "Name": rec.name,
                "Position": rec.position,
                "Team": rec.team,
                "Period": period_calc,
                "Start": s,
                "End": e,
                "Duration": e - s,
                "StrengthState": strength,
                "StrengthStateRaw": strength_raw,
                "StrengthStateBucket": strength_bucket,
                "SkatersOnIceFor": my_raw.1,
                "SkatersOnIceAgainst": their_raw.1,
                "GoaliesOnIceFor": my_raw.0,
                "GoaliesOnIceAgainst": their_raw.0,
            }));
        }
    }
    Value::Array(split_rows)
}
