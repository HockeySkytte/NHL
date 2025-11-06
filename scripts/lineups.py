"""lineups.py

DailyFaceoff lineup scraper (HTML only) + mapper to NHL playerIds via current roster.

We intentionally do NOT backfill from the roster when scraping fails – empty sections remain empty
so downstream consumers can detect scrape issues. Units (LW1/C1/RW1, LD1/RD1, G1/G2) are assigned
purely by the order we encounter player anchors in the DailyFaceoff HTML near the section headings.

Features:
    * Build Daily Faceoff lineup URL from team abbrev using Teams.csv, or accept a URL directly
    * Scrape expected lines (Forwards / Defense / Goalies) from DF HTML only (no NHL API inference)
    * Map scraped names to playerIds using current roster endpoint (strict match; no roster autofill)
    * Optional: save mapped JSON to app/static/lineup_<TEAM>.json or combined file

Usage (PowerShell):
    pwsh> & .\\.venv\\Scripts\\python.exe .\\scripts\\lineups.py --team ANA --save
    pwsh> & .\\.venv\\Scripts\\python.exe .\\scripts\\lineups.py --url https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations --season 20252026 --save
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple
import json
import re

try:
    import requests
except ImportError:
    print("Missing dependency: requests. Please install project requirements first.\n"
          "Tip (PowerShell): python -m pip install -r requirements.txt\n"
          "Or use the workspace venv: .\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt",
          file=sys.stderr)
    raise

# Optional HTML parsing for lineup scraping
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore


# Workspace root, to find Teams.csv and write static files
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Default headers for NHL API calls (avoid 403/empty responses)
REQUEST_HEADERS_JSON = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
    'Referer': 'https://www.nhl.com/',
}


def _load_teams_csv_local() -> List[Dict[str, str]]:
    paths = [
        os.path.join(REPO_ROOT, 'Teams.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Teams.csv'),
        os.path.join(os.getcwd(), 'Teams.csv'),
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                # Lazy CSV read without pandas
                import csv
                with open(p, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    return [{k: (v or '') for k, v in row.items()} for row in reader]
        except Exception:
            continue
    return []


TEAMS_ROWS_LOCAL = _load_teams_csv_local()


def _team_row_by_abbrev(team_abbrev: str) -> Optional[Dict[str, str]]:
    t = (team_abbrev or '').upper()
    for r in TEAMS_ROWS_LOCAL:
        if (r.get('Team') or '').upper() == t:
            return r
    return None


def _slugify_name(name: str) -> str:
    s = (name or '').strip().lower()
    s = s.replace('&', 'and')
    s = re.sub(r"[^a-z0-9\s\-]", '', s)
    s = re.sub(r"\s+", ' ', s).strip()
    return s.replace(' ', '-')


def build_dailyfaceoff_url_from_team(team_abbrev: str) -> str:
    """Build a Daily Faceoff lineup URL using the team name from Teams.csv.
    Example: ANA -> https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations
    """
    row = _team_row_by_abbrev(team_abbrev)
    if not row:
        raise RuntimeError(f"Unknown team abbrev: {team_abbrev}")
    name = row.get('Name') or ''
    slug = _slugify_name(name)
    return f"https://www.dailyfaceoff.com/teams/{slug}/line-combinations"


# Canonical Daily Faceoff lineup URLs provided by request (covers all active NHL teams)
TEAM_URLS: Dict[str, str] = {
    'ANA': 'https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations',
    'BOS': 'https://www.dailyfaceoff.com/teams/boston-bruins/line-combinations',
    'BUF': 'https://www.dailyfaceoff.com/teams/buffalo-sabres/line-combinations',
    'CGY': 'https://www.dailyfaceoff.com/teams/calgary-flames/line-combinations',
    'CAR': 'https://www.dailyfaceoff.com/teams/carolina-hurricanes/line-combinations',
    'CHI': 'https://www.dailyfaceoff.com/teams/chicago-blackhawks/line-combinations',
    'COL': 'https://www.dailyfaceoff.com/teams/colorado-avalanche/line-combinations',
    'CBJ': 'https://www.dailyfaceoff.com/teams/columbus-blue-jackets/line-combinations',
    'DAL': 'https://www.dailyfaceoff.com/teams/dallas-stars/line-combinations',
    'DET': 'https://www.dailyfaceoff.com/teams/detroit-red-wings/line-combinations',
    'EDM': 'https://www.dailyfaceoff.com/teams/edmonton-oilers/line-combinations',
    'FLA': 'https://www.dailyfaceoff.com/teams/florida-panthers/line-combinations',
    'LAK': 'https://www.dailyfaceoff.com/teams/los-angeles-kings/line-combinations',
    'MIN': 'https://www.dailyfaceoff.com/teams/minnesota-wild/line-combinations',
    'MTL': 'https://www.dailyfaceoff.com/teams/montreal-canadiens/line-combinations',
    'NSH': 'https://www.dailyfaceoff.com/teams/nashville-predators/line-combinations',
    'NJD': 'https://www.dailyfaceoff.com/teams/new-jersey-devils/line-combinations',
    'NYI': 'https://www.dailyfaceoff.com/teams/new-york-islanders/line-combinations',
    'NYR': 'https://www.dailyfaceoff.com/teams/new-york-rangers/line-combinations',
    'OTT': 'https://www.dailyfaceoff.com/teams/ottawa-senators/line-combinations',
    'PHI': 'https://www.dailyfaceoff.com/teams/philadelphia-flyers/line-combinations',
    'PIT': 'https://www.dailyfaceoff.com/teams/pittsburgh-penguins/line-combinations',
    'SJS': 'https://www.dailyfaceoff.com/teams/san-jose-sharks/line-combinations',
    'SEA': 'https://www.dailyfaceoff.com/teams/seattle-kraken/line-combinations',
    'VAN': 'https://www.dailyfaceoff.com/teams/vancouver-canucks/line-combinations',
    'STL': 'https://www.dailyfaceoff.com/teams/st-louis-blues/line-combinations',
    'TBL': 'https://www.dailyfaceoff.com/teams/tampa-bay-lightning/line-combinations',
    'UTA': 'https://www.dailyfaceoff.com/teams/utah-mammoth/line-combinations',
    'TOR': 'https://www.dailyfaceoff.com/teams/toronto-maple-leafs/line-combinations',
    'VGK': 'https://www.dailyfaceoff.com/teams/vegas-golden-knights/line-combinations',
    'WPG': 'https://www.dailyfaceoff.com/teams/winnipeg-jets/line-combinations',
    'WSH': 'https://www.dailyfaceoff.com/teams/washington-capitals/line-combinations',
}


def infer_team_from_dailyfaceoff_url(url: str) -> str:
    """Infer NHL team abbrev (e.g., ANA) from a DailyFaceoff team URL using Teams.csv."""
    try:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        parts = [p for p in (parsed.path or '').split('/') if p]
        # Expect ['teams', '<slug>', 'line-combinations']
        slug = parts[1] if len(parts) >= 2 else ''
        name_guess = slug.replace('-', ' ').strip()
        # Title-case words to match Teams.csv Name field
        name_norm = ' '.join(w.capitalize() for w in name_guess.split())
        # Find exact match on Name (Active teams preferred)
        cands = [r for r in TEAMS_ROWS_LOCAL if (r.get('Name') or '').lower() == name_norm.lower()]
        if not cands:
            # Fallback: partial match
            cands = [r for r in TEAMS_ROWS_LOCAL if name_norm.lower() in (r.get('Name') or '').lower()]
        if cands:
            cands.sort(key=lambda r: (r.get('Active') != '1'))
            return (cands[0].get('Team') or '').upper()
    except Exception:
        pass
    raise RuntimeError(f"Unable to infer team from URL: {url}")


def fetch_recent_roster(team_abbrev: str, season: Optional[int]) -> List[Dict]:
    """Fetch a recent roster for a team (with jersey numbers and playerIds) via last game boxscore.
    Uses club-schedule-season to locate a recent game in the season, then reads boxscore roster.
    """
    team = (team_abbrev or '').upper()
    if not team:
        return []
    if season is None:
        d = datetime.utcnow()
        season = (d.year if d.month >= 9 else d.year - 1) * 10000 + (d.year + 1 if d.month >= 9 else d.year)
    sched_url = f"https://api-web.nhle.com/v1/club-schedule-season/{team}/{season}"
    try:
        rs = requests.get(sched_url, timeout=20, headers=REQUEST_HEADERS_JSON)
        rs.raise_for_status()
        data = rs.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch schedule for roster: {e}")
    games = data.get('games') or []
    finished_states = {'OFF', 'FINAL', 'COMPLETED', 'OFFICIAL'}
    last_game_id = None
    for g in reversed(games):
        st = str(g.get('gameState') or g.get('gameStatus') or '').upper()
        gid = g.get('id') or g.get('gamePk')
        if gid and (st in finished_states or st in {'FUT','PREVIEW','SCHEDULED'}):
            last_game_id = gid
            break
    if not last_game_id and games:
        last_game_id = games[-1].get('id') or games[-1].get('gamePk')
    if not last_game_id:
        return []
    # Fetch boxscore roster
    try:
        rb = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{last_game_id}/boxscore", timeout=20, headers=REQUEST_HEADERS_JSON)
        rb.raise_for_status()
        box = rb.json()
    except Exception as e:
        raise RuntimeError(f"Failed to fetch boxscore for roster: {e}")

    def unify(team_stats: Dict) -> List[Dict]:
        res: List[Dict] = []
        for grp in ('forwards', 'defense', 'goalies'):
            for p in (team_stats.get(grp) or []):
                nm = p.get('name'); nm = nm.get('default') if isinstance(nm, dict) else nm
                pos_raw = (p.get('position') or p.get('positionCode') or '').strip().upper()
                pos = 'F' if (pos_raw[:1] in ('C','L','R')) else (pos_raw[:1] or None)
                res.append({
                    'playerId': p.get('playerId'),
                    'name': nm,
                    'sweaterNumber': str(p.get('sweaterNumber') or p.get('sweater') or p.get('jersey') or '').strip(),
                    'pos': pos,
                })
        return res

    pbg = box.get('playerByGameStats') or {}
    if (box.get('homeTeam') or {}).get('abbrev') == team:
        roster = unify(pbg.get('homeTeam') or {})
    elif (box.get('awayTeam') or {}).get('abbrev') == team:
        roster = unify(pbg.get('awayTeam') or {})
    else:
        roster = (unify(pbg.get('homeTeam') or {}) + unify(pbg.get('awayTeam') or {}))
    return roster


def fetch_current_roster(team_abbrev: str) -> List[Dict]:
    """Fetch the current roster for a team using NHL API: /v1/roster/<TEAM>/current
    Returns list of dicts: { playerId, name, sweaterNumber, pos }
    """
    team = (team_abbrev or '').upper()
    if not team:
        return []
    url = f"https://api-web.nhle.com/v1/roster/{team}/current"
    try:
        r = requests.get(url, timeout=20, headers=REQUEST_HEADERS_JSON)
        r.raise_for_status()
        js = r.json() or {}
    except Exception:
        return []

    def get_name(p: Dict) -> str:
        fn = p.get('firstName'); ln = p.get('lastName')
        if isinstance(fn, dict): fn = fn.get('default')
        if isinstance(ln, dict): ln = ln.get('default')
        parts = [str(fn or '').strip(), str(ln or '').strip()]
        return ' '.join([x for x in parts if x])

    out: List[Dict] = []
    groups = [
        ('forwards', 'F'), ('forward', 'F'),
        ('defensemen', 'D'), ('defencemen', 'D'), ('defense', 'D'), ('defence', 'D'),
        ('goalies', 'G'), ('goalie', 'G'),
        # Some payloads nest under "roster": { forwards:[], defense:[], goalies:[] }
        ('roster.forwards', 'F'), ('roster.defense', 'D'), ('roster.defence', 'D'), ('roster.goalies', 'G')
    ]
    for key, pos in groups:
        # Support dotted key paths
        cur = js
        if '.' in key:
            parts = key.split('.')
            for p in parts:
                if isinstance(cur, dict):
                    cur = cur.get(p) or {}
            arr = cur if isinstance(cur, list) else []
        else:
            arr = js.get(key) or []
        if not isinstance(arr, list):
            continue
        for p in arr:
            try:
                pid = p.get('id') or p.get('playerId') or p.get('playerID')
                num = p.get('sweaterNumber') or p.get('number') or p.get('sweater')
                nm = get_name(p)
                out.append({
                    'playerId': pid,
                    'name': nm,
                    'sweaterNumber': str(num or '').strip(),
                    'pos': pos,
                })
            except Exception:
                continue
    return out


def _team_id_from_csv(team_abbrev: str) -> Optional[int]:
    """Lookup NHL numeric team ID from Teams.csv for use with StatsAPI."""
    row = _team_row_by_abbrev(team_abbrev)
    if not row:
        return None
    try:
        tid = row.get('TeamID')
        return int(tid) if (tid is not None and tid.strip()) else None
    except Exception:
        return None


# Removed StatsAPI roster usage per requirement to rely solely on api-web current roster.


def scrape_dailyfaceoff_lineup(url: str, team_abbrev: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Scrape DailyFaceoff lineup HTML and return ONLY the ordered player lists.

    Returned structure (UNMAPPED):
      { 'forwards': [ {name,pos,unit}, ... ], 'defense': [...], 'goalies': [...] }

    Units are assigned purely by encounter order inside each section:
      Forwards: LWn,Cn,RWn for the first 4 lines (max 12 players) else EXT
      Defense: LDn,RDn for first 3 pairs (max 6 players) else EXT
      Goalies: G1 for first goalie, G2 for second, extras EXT

    Team/roster/playerId resolution is performed later by map_lineup_to_player_ids.
    This function must NOT consult NHL API data nor invent players.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Referer': 'https://www.dailyfaceoff.com/'
    }
    try:
        resp = requests.get(url, timeout=25, headers=headers)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch lineup page: {e}")

    out: Dict[str, List[Dict]] = {'forwards': [], 'defense': [], 'goalies': []}

    if BeautifulSoup is None:
        return out  # Without parser we can't scrape

    soup = BeautifulSoup(html, 'html.parser')

    def norm_text(s: str) -> str:
        return ' '.join((s or '').replace('\xa0', ' ').strip().split())

    def find_heading(tokens: List[str]) -> Optional[object]:
        toks = [t.lower() for t in tokens]
        for tag in soup.find_all(['h1','h2','h3','h4','div','span']):
            txt = norm_text(tag.get_text(' ', strip=True)).lower()
            if any(t in txt for t in toks):
                return tag
        return None

    h_f = find_heading(['forwards'])
    h_d = find_heading(['defense','defence'])
    h_g = find_heading(['goalies','goalie'])

    def _extract_player_name(card) -> Optional[str]:
        # Prefer anchor span text, fallback to image alt
        anchor_name = card.select_one('div.flex.flex-row.justify-center a span')
        if anchor_name:
            nm_txt = norm_text(anchor_name.get_text(' ', strip=True))
            if nm_txt:
                return nm_txt
        img = card.find('img')
        if img:
            alt = norm_text(img.get('alt') or '')
            if alt:
                return alt
        return None

    def parse_forwards_section() -> List[List[str]]:
        """Return list of forward lines (each line list of 1-3 names) and extras as last list if present.

        HTML pattern observed:
          span#forwards heading
          ... header row containing span#lw-forwards span#c-forwards span#rw-forwards
          subsequent rows: <div class="mb-4 flex flex-row flex-wrap justify-evenly border-b"> with 3 player cards
          final row (no border-b) contains extras.
        """
        lines: List[List[str]] = []
        extras: List[str] = []
        span_fw = soup.select_one('span#forwards')
        if not span_fw:
            return []
        # Locate header row (contains lw-forwards/c-forwards/rw-forwards spans)
        header_row = soup.select_one('span#lw-forwards')
        sibling_iter = []
        if header_row:
            header_container = header_row.find_parent('div')  # inner div for LW
            if header_container:
                header_row_container = header_container.find_parent('div')  # row containing LW/C/RW
                if header_row_container:
                    sibling_iter = list(header_row_container.find_next_siblings('div'))
        if not sibling_iter:  # fallback: iterate siblings after span#forwards parent
            candidate_parent = span_fw.parent
            if candidate_parent:
                sibling_iter = [sib for sib in candidate_parent.find_next_siblings('div')]
        # Acquire defense heading element to know when to stop
        defense_span = soup.select_one('span#defense') or soup.select_one('span#defence')
        for sib in sibling_iter:
            if defense_span and defense_span in sib.descendants:
                break
            if getattr(sib, 'name', None) != 'div':
                continue
            cls_tokens = sib.get('class') or []
            cls_join = ' '.join(cls_tokens)
            if 'flex-row' not in cls_join or 'flex-wrap' not in cls_join:
                continue
            # Player cards are children with class containing text-center
            cards = sib.find_all('div', class_=lambda c: c and (('text-center' in ' '.join(c)) if isinstance(c, list) else 'text-center' in c))
            row_names: List[str] = []
            for card in cards:
                nm = _extract_player_name(card)
                if nm and nm.lower() not in {'lw','c','rw','forwards'} and nm not in row_names:
                    row_names.append(nm)
            if not row_names:
                continue
            # Distinguish lines vs extras: lines have 'border-b' class, extras lack it OR beyond 4 lines
            is_line = 'border-b' in cls_tokens and len(lines) < 4
            if is_line:
                lines.append(row_names)
            else:
                # treat everything else as extras
                extras.extend(row_names)
        # If we have extras, append them as last list (for unit EXT assignment later)
        if extras:
            lines.append(extras)  # extras combined at end
        return lines

    def parse_defense_section() -> List[List[str]]:
        """Return list of defense pairs (each list size 1-2) and extras as last list if present.

        Pattern:
          span#defense heading ("Defensive Pairings")
          subsequent rows: <div class="mb-4 flex flex-row flex-wrap justify-evenly border-b"> (two player cards)
          final row (no border-b) -> extras.
        """
        pairs: List[List[str]] = []
        extras: List[str] = []
        span_def = soup.select_one('span#defense') or soup.select_one('span#defence')
        if not span_def:
            return []
        sibling_iter = [sib for sib in span_def.parent.find_next_siblings('div')]
        # Stop at goalies heading
        goalies_span = soup.select_one('span#goalies') or soup.select_one('span#goalie')
        for sib in sibling_iter:
            if goalies_span and goalies_span in sib.descendants:
                break
            if getattr(sib, 'name', None) != 'div':
                continue
            cls_tokens = sib.get('class') or []
            cls_join = ' '.join(cls_tokens)
            if 'flex-row' not in cls_join or 'flex-wrap' not in cls_join:
                continue
            cards = sib.find_all('div', class_=lambda c: c and (('text-center' in ' '.join(c)) if isinstance(c, list) else 'text-center' in c))
            row_names: List[str] = []
            for card in cards:
                nm = _extract_player_name(card)
                if nm and nm.lower() not in {'defensive pairings','defense','defence'} and nm not in row_names:
                    row_names.append(nm)
            if not row_names:
                continue
            is_pair = 'border-b' in cls_tokens and len(pairs) < 3
            if is_pair:
                pairs.append(row_names)
            else:
                extras.extend(row_names)
        if extras:
            pairs.append(extras)
        return pairs

    forward_lines = parse_forwards_section()  # list of lists
    defense_pairs = parse_defense_section()   # list of lists

    # Flatten with unit assignments
    # Build raw name and preliminary unit lists preserving structure
    f_raw: List[str] = []
    f_units: List[str] = []
    for line_index, line_names in enumerate(forward_lines):
        if line_index < 4:
            for i, nm in enumerate(line_names):
                pos_label = ['LW','C','RW'][i] if i < 3 else 'EXT'
                unit = f"{pos_label}{line_index+1}" if i < 3 else 'EXT'
                f_raw.append(nm); f_units.append(unit)
        else:
            for nm in line_names:
                f_raw.append(nm); f_units.append('EXT')

    d_raw: List[str] = []
    d_units: List[str] = []
    for pair_index, pair_names in enumerate(defense_pairs):
        if pair_index < 3:
            for i, nm in enumerate(pair_names):
                side = ['LD','RD'][i] if i < 2 else 'EXT'
                unit = f"{side}{pair_index+1}" if i < 2 else 'EXT'
                d_raw.append(nm); d_units.append(unit)
        else:
            for nm in pair_names:
                d_raw.append(nm); d_units.append('EXT')

    # Specialized goalie extraction: structure differs from simple anchors sequence.
    def extract_goalies_section(head) -> List[str]:
        names: List[str] = []
        if not head:
            return names
        # Use span#goalies as anchor if present
        span_goalies = soup.select_one('span#goalies') or soup.select_one('span#goalie')
        root = (span_goalies.find_parent('div') if span_goalies else None) or head
        # Iterate through subsequent sibling rows until next section heading
        stop_spans = [s for s in [soup.select_one('span#forwards'), soup.select_one('span#defense'), soup.select_one('span#defence')] if s]
        for sib in root.find_next_siblings('div'):
            # Stop when we reach another major section
            if any(sp in sib.descendants for sp in stop_spans):
                break
            if getattr(sib, 'name', None) != 'div':
                continue
            cls = ' '.join(sib.get('class') or [])
            if 'flex-row' not in cls or 'flex-wrap' not in cls:
                continue
            # Extract from all card blocks within this row
            cards = sib.find_all('div', class_=lambda c: c and (('text-center' in ' '.join(c)) if isinstance(c, list) else 'text-center' in c))
            for card in cards:
                # Try centered anchor span text first
                anchor_name = card.select_one('div.flex.flex-row.justify-center a span')
                nm = None
                if anchor_name:
                    nm_txt = norm_text(anchor_name.get_text(' ', strip=True))
                    if nm_txt:
                        nm = nm_txt
                if not nm:
                    # Fallback to jersey image alt (covers blinking image variant)
                    img = card.find('img')
                    if img:
                        alt = norm_text(img.get('alt') or '')
                        if alt:
                            nm = alt
                if nm and any(c.isalpha() for c in nm):
                    low = nm.lower()
                    if low not in {'goalies','goalie'} and nm not in names:
                        names.append(nm)
                if len(names) >= 3:
                    break
            if len(names) >= 3:
                break
        return names

    g_raw = extract_goalies_section(h_g)

    def dedup_preserve(names: List[str]) -> List[str]:
        seen = set(); outn: List[str] = []
        for nm in names:
            key = nm.lower()
            if key in seen:
                continue
            seen.add(key)
            # Normalize shouting (ALL CAPS) to title case
            words = nm.split()
            if words and all(w.isupper() for w in words):
                nm = ' '.join(w.capitalize() for w in words)
            outn.append(nm)
        return outn

    f_names = dedup_preserve(f_raw)
    d_names = dedup_preserve(d_raw)
    g_names = dedup_preserve(g_raw)

    # Build first-occurrence unit maps so units align with deduped names
    def first_unit_map(names: List[str], units: List[str]) -> Dict[str, str]:
        m: Dict[str, str] = {}
        for nm, u in zip(names, units):
            key = (nm or '').lower()
            if key and key not in m:
                m[key] = u
        return m

    f_unit_map = first_unit_map(f_raw, f_units)
    d_unit_map = first_unit_map(d_raw, d_units)

    # Assign units with special cases: 13th forward -> F5, 7th defender -> D4; rest beyond -> EXT
    for idx, nm in enumerate(f_names):
        base = f_unit_map.get(nm.lower(), 'EXT')
        unit = 'F5' if idx == 12 else ('EXT' if idx > 12 else base)
        out['forwards'].append({'name': nm, 'pos': 'F', 'unit': unit})
    for idx, nm in enumerate(d_names):
        base = d_unit_map.get(nm.lower(), 'EXT')
        unit = 'D4' if idx == 6 else ('EXT' if idx > 6 else base)
        out['defense'].append({'name': nm, 'pos': 'D', 'unit': unit})
    # Only take the first goalie as G1 (starter); ignore the rest
    if g_names:
        out['goalies'].append({'name': g_names[0], 'pos': 'G', 'unit': 'G1'})

    # Fallback: if nothing was captured from anchors near headings, try parsing Next.js __NEXT_DATA__ for structured lineup
    if not (out['forwards'] or out['defense'] or out['goalies']):
        try:
            next_tag = soup.find('script', id='__NEXT_DATA__')
            if next_tag and (next_tag.string or '').strip():
                data = json.loads(next_tag.string)

                def extract_names_from_obj(obj) -> Dict[str, List[str]]:
                    acc = {'forwards': [], 'defense': [], 'goalies': []}
                    locked = {'forwards': False, 'defense': False}

                    def full_name(d: Dict) -> Optional[str]:
                        keys_pairs = [
                            ('name', None), ('playerName', None), ('fullName', None),
                            ('firstName', 'lastName'), ('first', 'last'), ('first_name', 'last_name'),
                            ('first_name', 'surname'), ('givenName', 'familyName'), ('displayName', None)
                        ]
                        for k1, k2 in keys_pairs:
                            if k2 is None and k1 in d and isinstance(d[k1], str) and any(c.isalpha() for c in d[k1]):
                                return d[k1]
                            if k2 and (k1 in d) and (k2 in d):
                                a = d.get(k1); b = d.get(k2)
                                if isinstance(a, str) and isinstance(b, str):
                                    nm = f"{a} {b}".strip()
                                    if any(c.isalpha() for c in nm):
                                        return nm
                        return None

                    def is_player_like(d: Dict) -> bool:
                        if not isinstance(d, dict):
                            return False
                        fn = full_name(d)
                        return bool(fn)

                    def walk(o, path_keys: List[str]):
                        # Look for arrays of arrays with player-like dicts
                        if isinstance(o, dict):
                            for k, v in o.items():
                                walk(v, path_keys + [str(k).lower()])
                        elif isinstance(o, list):
                            # list of lists (forward lines/pairs)
                            if o and all(isinstance(x, list) for x in o):
                                # Determine section strictly by path tokens to avoid global noise
                                path_txt = ' '.join(path_keys)
                                section = None
                                if any(tok in path_txt for tok in ['forward','forwards','line-combinations','lines']):
                                    section = 'forwards'
                                elif any(tok in path_txt for tok in ['defense','defence','pairs']):
                                    section = 'defense'
                                # Single-pass lock-in: only fill forwards/defense once
                                if section in ('forwards','defense') and locked.get(section):
                                    return
                                if section:
                                    for sub in o:
                                        if not isinstance(sub, list):
                                            continue
                                        for ent in sub:
                                            if is_player_like(ent):
                                                nm = full_name(ent)
                                                if nm:
                                                    acc[section].append(nm)
                                if section in ('forwards','defense'):
                                    locked[section] = True
                            else:
                                # flat list of player-like dicts (e.g., goalies)
                                if o and all(isinstance(x, dict) for x in o) and any(is_player_like(x) for x in o):
                                    path_txt = ' '.join(path_keys)
                                    # Only consider arrays explicitly tied to goalies
                                    if any(tok in path_txt for tok in ['goalie','goalies','tandem','starter','net']):
                                        # Collect names but we will filter and dedupe later
                                        for ent in o:
                                            if is_player_like(ent):
                                                nm = full_name(ent)
                                                if nm:
                                                    acc['goalies'].append(nm)
                        # Other types ignored

                    # Limit search to pageProps subtree to avoid cross-site noise
                    root = obj
                    if isinstance(obj, dict):
                        root = obj.get('props', {}).get('pageProps', obj)
                    walk(root, [])
                    return acc

                names_acc = extract_names_from_obj(data)

                # Deduplicate across sections (E): remove names already assigned to forwards from defense/goalies; and from defense to goalies
                f_set = {n.lower() for n in names_acc['forwards']}
                names_acc['defense'] = [n for n in names_acc['defense'] if n.lower() not in f_set]
                used_set = f_set.union({n.lower() for n in names_acc['defense']})
                names_acc['goalies'] = [n for n in names_acc['goalies'] if n.lower() not in used_set]

                # Goalie isolation (D): if goalie list suspiciously long (>5), drop it
                if len(names_acc['goalies']) > 5:
                    names_acc['goalies'] = []

                # If we found anything, assign units using the same rules
                if names_acc['forwards'] or names_acc['defense'] or names_acc['goalies']:
                    out = {'forwards': [], 'defense': [], 'goalies': []}
                    for i, nm in enumerate(names_acc['forwards']):
                        unit = ['LW','C','RW'][i % 3] + str(i//3 + 1) if i < 12 else 'EXT'
                        out['forwards'].append({'name': nm, 'pos': 'F', 'unit': unit})
                    for i, nm in enumerate(names_acc['defense']):
                        unit = ['LD','RD'][i % 2] + str(i//2 + 1) if i < 6 else 'EXT'
                        out['defense'].append({'name': nm, 'pos': 'D', 'unit': unit})
                    if names_acc['goalies']:
                        out['goalies'].append({'name': names_acc['goalies'][0], 'pos': 'G', 'unit': 'G1'})
        except Exception:
            pass

    return out


def map_lineup_to_player_ids(lineup: Dict[str, List[Dict]], roster: List[Dict], team_abbrev: Optional[str]) -> Dict[str, object]:
    """Map scraped lineup entries to NHL playerIds using ONLY the provided current roster.

    Strict rules:
      * No roster backfill: players not scraped are not added.
      * A name is matched case-insensitively to full roster name first.
      * If full name fails, try last-name-only when it yields exactly ONE roster candidate.
      * Ambiguous or unmatched entries are dropped.
    Returned structure mirrors lineup but each item gains playerId.
    """
    def norm_name(s: str) -> str:
        return ' '.join(re.sub(r"[^a-zA-Z\s\-']", '', (s or '')).strip().split()).lower()

    # Build lookup maps
    by_full: Dict[str, Dict] = {}
    by_last: Dict[str, List[Dict]] = {}
    for p in roster:
        nm = p.get('name') or ''
        nm_norm = norm_name(nm)
        if nm_norm:
            by_full[nm_norm] = p
        last = nm_norm.split(' ')[-1]
        if last:
            by_last.setdefault(last, []).append(p)

    def resolve_pid(scraped_name: str) -> Optional[int]:
        nm_norm = norm_name(scraped_name)
        if not nm_norm:
            return None
        p = by_full.get(nm_norm)
        if p and p.get('playerId') is not None:
            return p.get('playerId')
        last = nm_norm.split(' ')[-1]
        cands = by_last.get(last, [])
        if len(cands) == 1 and cands[0].get('playerId') is not None:
            return cands[0].get('playerId')
        return None

    def map_section(items: List[Dict]) -> List[Dict]:
        out_items: List[Dict] = []
        for it in items:
            pid = resolve_pid(it.get('name') or '')
            if pid is None:
                continue  # strict: skip unmatched
            out_items.append({'name': it.get('name'), 'playerId': pid, 'unit': it.get('unit'), 'pos': it.get('pos')})
        return out_items

    starters_forwards = map_section(lineup.get('forwards', []))
    starters_defense = map_section(lineup.get('defense', []))
    starters_goalies = map_section(lineup.get('goalies', []))

    included_ids = {it['playerId'] for it in starters_forwards + starters_defense + starters_goalies if it.get('playerId') is not None}

    # Append EXT players from roster not already included
    ext_forwards: List[Dict] = []
    ext_defense: List[Dict] = []
    ext_goalies: List[Dict] = []
    for p in roster:
        pid = p.get('playerId')
        if pid is None or pid in included_ids:
            continue
        pos = (p.get('pos') or '').upper()[:1]
        name = p.get('name') or ''
        if pos == 'F':
            ext_forwards.append({'name': name, 'playerId': pid, 'unit': 'EXT', 'pos': 'F'})
        elif pos == 'D':
            ext_defense.append({'name': name, 'playerId': pid, 'unit': 'EXT', 'pos': 'D'})
        elif pos == 'G':
            # Only non-starter goalies added as EXT
            ext_goalies.append({'name': name, 'playerId': pid, 'unit': 'EXT', 'pos': 'G'})

    result: Dict[str, object] = {
        'team': (team_abbrev or '').upper() if team_abbrev else None,
        'forwards': starters_forwards + ext_forwards,
        'defense': starters_defense + ext_defense,
        'goalies': starters_goalies + ext_goalies,
        # metadata keys like generated_at added by caller
    }
    return result


def _validate_season(s: str) -> str:
    s2 = str(s).strip()
    if not re.match(r'^\d{8}$', s2):
        raise argparse.ArgumentTypeError('Season must be like 20252026')
    return s2


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description='Scrape expected lineups and map to playerIds')
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--url', help='DailyFaceoff team lineup URL (e.g., https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations)')
    g.add_argument('--team', help='NHL team abbrev (e.g., ANA)')
    g.add_argument('--all', action='store_true', help='Scrape all active NHL teams and write one combined JSON file')
    g.add_argument('--teams', help='Comma-separated list of team abbrevs to scrape (e.g., ANA,BOS,BUF)')
    ap.add_argument('--season', default='20252026', type=_validate_season, help='Season code for roster source, e.g., 20252026')
    ap.add_argument('--save', action='store_true', help='Save JSON to app/static/lineup_<TEAM>.json')
    ap.add_argument('--out', help='Custom output path for JSON (overrides --save default path)')
    ap.add_argument('--out-all', help='Output path for combined JSON when using --all/--teams (default app/static/lineups_all.json)')
    ap.add_argument('--quiet', action='store_true', help='Reduce console output in batch mode')
    args = ap.parse_args(argv)

    # Batch mode for all or selected teams
    if args.all or args.teams:
        teams_list: List[str]
        if args.all:
            teams_list = sorted(list(TEAM_URLS.keys()))
        else:
            teams_list = [t.strip().upper() for t in (args.teams or '').split(',') if t.strip()]
        out_all_path = args.out_all or os.path.join(REPO_ROOT, 'app', 'static', 'lineups_all.json')
        os.makedirs(os.path.dirname(out_all_path), exist_ok=True)

        combined: Dict[str, Dict] = {}
        season_int = int(args.season)
        for t in teams_list:
            url_t = TEAM_URLS.get(t) or build_dailyfaceoff_url_from_team(t)
            if not args.quiet:
                print(f"Scraping {t} | {url_t}")
            try:
                lineup = scrape_dailyfaceoff_lineup(url_t, team_abbrev=t)
            except Exception as e:
                print(f"[warn] scrape failed for {t}: {e}")
                lineup = {'forwards': [], 'defense': [], 'goalies': []}
            try:
                roster = fetch_current_roster(t)
                mapped = map_lineup_to_player_ids(lineup, roster, t)
                # Attach generation timestamp for downstream consumers
                try:
                    mapped['generated_at'] = datetime.now(timezone.utc).isoformat()
                except Exception:
                    pass
            except Exception as e:
                print(f"[warn] mapping failed for {t}: {e}")
                mapped = {'team': t, 'forwards': [], 'defense': [], 'goalies': []}
                try:
                    mapped['generated_at'] = datetime.now(timezone.utc).isoformat()
                except Exception:
                    pass
            combined[t] = mapped
            if not args.quiet:
                f_obj = mapped.get('forwards')
                d_obj = mapped.get('defense')
                g_obj = mapped.get('goalies')
                f_list: List[Dict] = f_obj if isinstance(f_obj, list) else []
                d_list: List[Dict] = d_obj if isinstance(d_obj, list) else []
                g_list: List[Dict] = g_obj if isinstance(g_obj, list) else []
                print(f"  -> F:{len(f_list)} D:{len(d_list)} G:{len(g_list)}")

        with open(out_all_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(combined, ensure_ascii=False))
        print(f"[file] wrote combined lineups JSON: {out_all_path}")
        return 0

    # Resolve URL and team (single-team mode)
    if args.url:
        url = args.url
        try:
            team = infer_team_from_dailyfaceoff_url(url)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2
    else:
        team = (args.team or '').upper()
        try:
            url = build_dailyfaceoff_url_from_team(team)
        except Exception as e:
            print(f"[error] {e}", file=sys.stderr)
            return 2

    print(f"Team: {team} | URL: {url}")

    # Scrape lineup
    try:
        lineup = scrape_dailyfaceoff_lineup(url, team_abbrev=team)
    except Exception as e:
        print(f"[error] lineup scrape failed: {e}", file=sys.stderr)
        return 3

    # Fetch roster and map
    try:
        season_int = int(args.season)
        # Use current roster endpoint only
        roster = fetch_current_roster(team)
        # No roster-based goalies fill; rely strictly on scraped content.
        mapped = map_lineup_to_player_ids(lineup, roster, team)
    except Exception as e:
        print(f"[error] roster/map failed: {e}", file=sys.stderr)
        return 4

    # No roster fallback; leave empty if scrape produced nothing.

    # Print compact summary
    print("\nExpected lineup (mapped to PlayerIDs):")
    print(json.dumps(mapped, ensure_ascii=False, indent=2))

    # Save if requested
    if args.save or args.out:
        out_path = args.out
        if not out_path:
            out_path = os.path.join(REPO_ROOT, 'app', 'static', f'lineup_{team}.json')
        try:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            # Attach generation timestamp
            try:
                mapped_out = dict(mapped)
                mapped_out['generated_at'] = datetime.now(timezone.utc).isoformat()
            except Exception:
                mapped_out = mapped
            with open(out_path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(mapped_out, ensure_ascii=False))
            print(f"[file] wrote lineup JSON: {out_path}")
        except Exception as e:
            print(f"[error] failed to write JSON: {e}", file=sys.stderr)
            return 5

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
