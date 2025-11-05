"""
lineups.py

Standalone lineup scraper and mapper.

Features:
- Build Daily Faceoff lineup URL from team abbrev using Teams.csv, or accept a URL directly
- Scrape expected lines (Forwards/Defense/Goalies)
- Fetch a recent roster (jerseys + playerIds) from NHL boxscore API
- Map names/jerseys to playerIds (jersey-first, then full-name, then last-name fallback)
- Optional: save mapped JSON to app/static/lineup_<TEAM>.json or custom path

Usage (PowerShell):
  pwsh> & .\.venv\Scripts\python.exe .\scripts\lineups.py --team ANA --save
  pwsh> & .\.venv\Scripts\python.exe .\scripts\lineups.py --url https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations --season 20252026 --save
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
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
        rs = requests.get(sched_url, timeout=20)
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
        rb = requests.get(f"https://api-web.nhle.com/v1/gamecenter/{last_game_id}/boxscore", timeout=20)
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
        r = requests.get(url, timeout=20)
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
        ('goalies', 'G'), ('goalie', 'G')
    ]
    for key, pos in groups:
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
        return int(row.get('TeamID')) if row.get('TeamID') else None
    except Exception:
        return None


# Removed StatsAPI roster usage per requirement to rely solely on api-web current roster.


def scrape_dailyfaceoff_lineup(url: str, team_abbrev: Optional[str] = None) -> Dict[str, List[Dict]]:
    """Scrape DailyFaceoff lineup page for a single team.
    Returns dict with keys: forwards (list of lines), defense (list of pairs), goalies (list).
    Each entry is a dict: { name, jersey (optional), pos (F/D/G), unit (e.g., L1/D1) }.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    try:
        r = requests.get(url, timeout=25, headers=headers)
        r.raise_for_status()
        html = r.text
    except Exception as e:
        raise RuntimeError(f"Failed to fetch lineup page: {e}")

    out = {'forwards': [], 'defense': [], 'goalies': []}

    def norm_text(s: str) -> str:
        return ' '.join((s or '').replace('\xa0', ' ').strip().split())

    def parse_name_and_jersey(text: str) -> Tuple[str, Optional[str]]:
        t = norm_text(text)
        # Patterns like "11 Trevor Zegras" or "#11 Trevor Zegras"
        m = re.match(r'^(?:#?\s*(\d{1,2})\s+)?(.+?)$', t)
        if m:
            num = m.group(1)
            name = m.group(2)
            return name.strip(), (num.strip() if num else None)
        return t, None

    if BeautifulSoup is not None:
        soup = BeautifulSoup(html, 'html.parser')

        def collect_players(container, unit_label_prefix: str, pos_code: str):
            items = []
            if not container:
                return items
            # Broaden to all candidate nodes; don't restrict to string-only matches
            candidates = container.find_all(['a', 'div', 'span'])
            for node in candidates:
                try:
                    txt = norm_text(node.get_text(' ', strip=True))
                except Exception:
                    txt = ''
                if not txt or len(txt) < 2:
                    continue
                # Filter out generic words/section labels
                lowered = txt.lower()
                if lowered in ('forwards', 'defense', 'defence', 'defense pairings', 'goalies', 'goalie', 'line combinations'):
                    continue
                # Prefer likely player anchors or bold name spans
                href = node.get('href') if hasattr(node, 'get') else None
                cls = ' '.join((node.get('class') or [])) if hasattr(node, 'get') else ''
                is_playerish = False
                if isinstance(href, str) and '/players/' in href:
                    is_playerish = True
                if ('font-bold' in cls) or ('uppercase' in cls) or ('text-xs' in cls) or ('xl:text-base' in cls):
                    is_playerish = True
                # Parse name and optional jersey from text
                name, j = parse_name_and_jersey(txt)
                # crude name filter
                if is_playerish and any(c.isalpha() for c in name) and len(name) <= 60:
                    items.append({'name': name, 'jersey': j, 'pos': pos_code, 'unit': unit_label_prefix})
            return items

        def find_section_by_heading(soup, heading_texts: List[str]):
            for htag in soup.find_all(['h2', 'h3', 'h4', 'h5', 'div', 'span']):
                txt = norm_text(htag.get_text(' ', strip=True)).lower()
                if any(ht in txt for ht in heading_texts):
                    cont = htag.find_next()
                    parent = htag.parent
                    return cont or parent
            return None

        # Locate containers by headings (best-effort; we'll prefer extracting within these containers)
        fwd_cont = find_section_by_heading(soup, ['forwards', 'forward lines'])
        d_cont = find_section_by_heading(soup, ['defense', 'defence', 'defensive pairings'])
        g_cont = find_section_by_heading(soup, ['goalies', 'goalie'])

        # Helper: extract players with jerseys from a specific container only
        def extract_from_container(container, sec: str):
            items: List[Dict] = []
            if not container:
                return items
            
            def find_next_player_name(start_node) -> Optional[str]:
                # Scan forward until next jersey image or reasonable limit, and pick the next plausible player name
                limit = 200
                steps = 0
                for node in start_node.next_elements:
                    steps += 1
                    if steps > limit:
                        break
                    if getattr(node, 'name', None) == 'img':
                        # Stop at next jersey image to avoid crossing into next player block
                        src = (node.get('src') or '') + ' ' + (node.get('srcset') or '')
                        if 'uploads/player/jersey' in src:
                            break
                    # Look for spans/anchors that look like player name blocks
                    if getattr(node, 'name', None) in ('span', 'a', 'div'):
                        txt = norm_text(getattr(node, 'get_text', lambda *a, **k: '')(' ', strip=True))
                        if not txt:
                            continue
                        # Heuristics: typical classes include font-bold/uppercase; name should have letters/spaces and not be generic
                        cls = ' '.join((node.get('class') or [])) if hasattr(node, 'get') else ''
                        if (('font-bold' in cls) or ('uppercase' in cls) or ('players' in (node.get('href') or ''))):
                            # Basic name sanity
                            if any(c.isalpha() for c in txt) and len(txt.split()) <= 4 and len(txt) <= 40:
                                bad = {'forwards','defense','defence','goalies','goalie','line combinations'}
                                if txt.lower() not in bad:
                                    return txt
                return None

            imgs = container.find_all('img')
            for img in imgs:
                alt = norm_text(img.get('alt') or '')
                if not alt:
                    continue
                full = decode_next_image_src(img)
                if 'uploads/player/jersey' not in full:
                    continue
                jersey = None
                team_code = None
                m = re.search(r'/([A-Z]{2,3})_([0-9]{1,2})_', full)
                if m:
                    team_code = m.group(1)
                    jersey = m.group(2)
                # Enforce team match when provided. If we cannot determine team_code, skip to avoid cross-team bleed.
                if team_abbrev:
                    if (not team_code) or (team_code.upper() != team_abbrev.upper()):
                        continue
                # Try to find the next explicit player name block; fallback to alt if not found
                name_near = find_next_player_name(img) or alt
                items.append({'name': name_near, 'jersey': jersey, 'teamCode': team_code, 'pos': ('F' if sec=='forwards' else 'D' if sec=='defense' else 'G')})
            return items

        # NEW: Prefer extracting strictly within sections to avoid sidebar/global widgets
        def decode_next_image_src(img_tag) -> str:
            src = img_tag.get('src') or ''
            srcset = img_tag.get('srcset') or ''
            cand = src or srcset
            if not cand:
                return ''
            try:
                from urllib.parse import urlparse, parse_qs, unquote
                # src may be like "/_next/image?url=https%3A%2F%2Fapi.dailyfaceoff.com%2Fuploads%2Fplayer%2Fjersey%2F...png&w=3840&q=60"
                u = urlparse(cand.split(' ')[0])
                qs = parse_qs(u.query)
                if 'url' in qs and qs['url']:
                    return unquote(qs['url'][0])
                # or cand itself may already be a full URL
                if cand.startswith('http'):
                    return cand
            except Exception:
                pass
            return cand

        # Attempt strict extraction from the identified containers
        forwards_list: List[Dict] = extract_from_container(fwd_cont, 'forwards')
        defense_list: List[Dict]  = extract_from_container(d_cont, 'defense')
        goalies_list: List[Dict]  = extract_from_container(g_cont, 'goalies')

        # NEW: Fallback to text-based names if some players have no jersey images
        def merge_text_candidates(container, sec: str, target_list: List[Dict]):
            if not container:
                return
            try:
                txt_items = collect_players(container, unit_label_prefix=sec, pos_code=('F' if sec=='forwards' else 'D' if sec=='defense' else 'G'))
            except Exception:
                txt_items = []
            existing_names = { (it.get('name') or '').strip().lower() for it in target_list }
            for ti in txt_items:
                nm = (ti.get('name') or '').strip()
                if not nm:
                    continue
                if nm.strip().lower() in existing_names:
                    continue
                # Add as name-only candidate; assign teamCode to current team to allow name-based mapping later
                target_list.append({
                    'name': nm,
                    'jersey': ti.get('jersey'),
                    'teamCode': (team_abbrev or '').upper() if team_abbrev else None,
                    'pos': ('F' if sec=='forwards' else 'D' if sec=='defense' else 'G')
                })

        merge_text_candidates(fwd_cont, 'forwards', forwards_list)
        merge_text_candidates(d_cont, 'defense', defense_list)
        merge_text_candidates(g_cont, 'goalies', goalies_list)

        # Augment: Walk the document in section order and add name-only candidates
        # This helps capture players rendered without jersey images inside the section content
        try:
            from bs4 import Tag  # type: ignore
            current_section: Optional[str] = None
            def maybe_add_name(sec: str, nm: str):
                nm2 = norm_text(nm)
                if not nm2:
                    return
                # crude filter
                bad = {'forwards','defense','defence','defense pairings','goalies','goalie','line combinations'}
                if nm2.lower() in bad:
                    return
                target = forwards_list if sec == 'forwards' else defense_list if sec == 'defense' else goalies_list
                keyset = { ((it.get('name') or '').strip().lower(), (it.get('jersey') or ''), (it.get('teamCode') or '')) for it in target }
                key = (nm2.strip().lower(), '', (team_abbrev or '').upper())
                if key in keyset:
                    return
                target.append({
                    'name': nm2,
                    'jersey': None,
                    'teamCode': (team_abbrev or '').upper() if team_abbrev else None,
                    'pos': ('F' if sec=='forwards' else 'D' if sec=='defense' else 'G')
                })

            for node in soup.descendants:
                if not isinstance(node, Tag):
                    continue
                # Track section via headings encountered in order
                if node.name in ('h1','h2','h3','h4','h5','div','span'):
                    txt = norm_text(node.get_text(' ', strip=True)).lower()
                    if 'goalies' in txt:
                        current_section = 'goalies'
                    elif ('defensive pairings' in txt) or ('defence' in txt) or ('defense' in txt):
                        current_section = 'defense'
                    elif 'forwards' in txt:
                        current_section = 'forwards'
                if current_section and node.name in ('a','span','div'):
                    # Favor player links and bold/uppercase markers
                    href = node.get('href') if hasattr(node, 'get') else None
                    cls = ' '.join((node.get('class') or [])) if hasattr(node, 'get') else ''
                    is_playerish = False
                    if isinstance(href, str) and '/players/' in href:
                        is_playerish = True
                    if ('font-bold' in cls) or ('uppercase' in cls) or ('text-xs' in cls) or ('xl:text-base' in cls):
                        is_playerish = True
                    if not is_playerish:
                        continue
                    nm_txt = norm_text(node.get_text(' ', strip=True))
                    if nm_txt:
                        maybe_add_name(current_section, nm_txt)
        except Exception:
            pass

        # Fallback: If any section is empty, perform a conservative global scan with team filter
        if not forwards_list or not defense_list or not goalies_list:
            from bs4 import Tag  # type: ignore
            current_section: Optional[str] = None
            def to_item(name: str, jersey: Optional[str], sec: str, team_code: Optional[str]) -> Dict:
                return {
                    'name': name,
                    'jersey': jersey,
                    'teamCode': team_code,
                    'pos': ('F' if sec == 'forwards' else 'D' if sec == 'defense' else 'G'),
                }
            for node in soup.descendants:
                if not isinstance(node, Tag):
                    continue
                if node.name in ('h1','h2','h3','h4','h5','div','span'):
                    txt = norm_text(node.get_text(' ', strip=True)).lower()
                    if 'goalies' in txt:
                        current_section = 'goalies'
                    elif ('defensive pairings' in txt) or ('defence' in txt) or ('defense' in txt):
                        current_section = 'defense'
                    elif 'forwards' in txt:
                        current_section = 'forwards'
                if node.name == 'img':
                    alt = norm_text(node.get('alt') or '')
                    if not alt:
                        continue
                    full = decode_next_image_src(node)
                    if 'uploads/player/jersey' not in full:
                        continue
                    jersey = None
                    team_code = None
                    m = re.search(r'/([A-Z]{2,3})_([0-9]{1,2})_', full)
                    if m:
                        team_code = m.group(1)
                        jersey = m.group(2)
                    # Enforce team match strictly; if team_code is missing or mismatched, skip.
                    if team_abbrev:
                        if (not team_code) or (team_code.upper() != team_abbrev.upper()):
                            continue
                    sec = current_section or 'forwards'
                    item = to_item(alt, jersey, sec, team_code)
                    if sec == 'forwards':
                        forwards_list.append(item)
                    elif sec == 'defense':
                        defense_list.append(item)
                    elif sec == 'goalies':
                        if len(goalies_list) < 2:
                            goalies_list.append(item)

        # Remove duplicates across sections (favor earlier sections)
        seen = set()
        def dedup(lst: List[Dict]) -> List[Dict]:
            res = []
            for it in lst:
                key = (it.get('name') or '', it.get('jersey') or '', it.get('teamCode') or '')
                if key in seen:
                    continue
                seen.add(key)
                res.append(it)
            return res

        forwards_list = dedup(forwards_list)
        defense_list = dedup(defense_list)
        goalies_list = dedup(goalies_list)

        # Assign units based on order
        # Forwards: LW1, C1, RW1, LW2, C2, RW2, LW3, C3, RW3, LW4, C4, RW4, then EXT
        f_cols = ['LW', 'C', 'RW']
        for i, it in enumerate(forwards_list):
            if i < 12:
                col = f_cols[i % 3]
                line_no = (i // 3) + 1
                it['unit'] = f"{col}{line_no}"
            else:
                it['unit'] = 'EXT'

        # Defense: LD1, RD1, LD2, RD2, LD3, RD3, then EXT
        d_cols = ['LD', 'RD']
        for i, it in enumerate(defense_list):
            if i < 6:
                side = d_cols[i % 2]
                pair_no = (i // 2) + 1
                it['unit'] = f"{side}{pair_no}"
            else:
                it['unit'] = 'EXT'

        # Goalies: G1, G2 (extras -> EXT)
        for i, it in enumerate(goalies_list):
            it['unit'] = f"G{i+1}" if i < 2 else 'EXT'

        # Replace out with classified lists (avoid duplicates)
        out['forwards'] = forwards_list
        out['defense'] = defense_list
        out['goalies'] = goalies_list
    else:
        # Fallback: regex-based extraction
        lines = [l.strip() for l in html.splitlines() if l.strip()]
        for ln in lines:
            m = re.findall(r'(?:#?\s*(\d{1,2})\s+)?([A-Z][a-z]+\s+[A-Z][a-z\-\']+)', ln)
            for num, nm in m:
                nm2 = nm.strip()
                num2 = num.strip() if num else None
                if nm2:
                    out['forwards'].append({'name': nm2, 'jersey': num2, 'pos': 'F', 'unit': 'L'})

    # Deduplicate by (name, jersey)
    def dedup(lst: List[Dict]) -> List[Dict]:
        seen = set(); res = []
        for it in lst:
            key = (it.get('name') or '', it.get('jersey') or '', it.get('teamCode') or '')
            if key in seen:
                continue
            seen.add(key)
            res.append(it)
        return res

    out['forwards'] = dedup(out['forwards'])
    out['defense'] = dedup(out['defense'])
    out['goalies'] = dedup(out['goalies'])
    return out


def map_lineup_to_player_ids(lineup: Dict[str, List[Dict]], roster: List[Dict], team_abbrev: str) -> Dict[str, List[Dict]]:
    """Map scraped lineup entries to playerIds using jersey-first, then name fallback.
    Strict mode: keep ONLY players that (a) were scraped from Daily Faceoff and
    (b) are present on the team's roster. Do NOT auto-fill extra players from
    the roster to hit counts, as that can introduce wrong players when the
    roster API has noise or off-season artifacts.

    roster: list of { playerId, name, sweaterNumber, pos }
    Returns the lineup dict with an added 'playerId' key where matched.
    Units are preserved from scrape when present, else assigned by order.
    """
    def norm(s: Optional[str]) -> str:
        if not s:
            return ''
        try:
            import unicodedata as _ud
            s2 = _ud.normalize('NFKD', s)
            s2 = ''.join([c for c in s2 if not _ud.combining(c)])
        except Exception:
            s2 = s
        s2 = s2.replace('.', ' ').replace('-', ' ').replace("'", '').strip().lower()
        s2 = ' '.join(s2.split())
        return s2

    def jersey_norm(s: Optional[str]) -> str:
        digits = ''.join(ch for ch in str(s or '') if ch.isdigit())
        return str(int(digits)) if digits.isdigit() else ''

    by_num: Dict[str, Dict] = {}
    by_name: Dict[str, Dict] = {}
    by_last: Dict[str, List[Dict]] = {}
    roster_ids: set[int] = set()
    for p in roster:
        num = jersey_norm(p.get('sweaterNumber'))
        if num:
            by_num[num] = p
        nm = norm(p.get('name'))
        if nm:
            by_name[nm] = p
            last = nm.split(' ')[-1]
            by_last.setdefault(last, []).append(p)
        try:
            pid = int(p.get('playerId')) if p.get('playerId') is not None else None
            if pid:
                roster_ids.add(pid)
        except Exception:
            pass

    def match_player(name: Optional[str], jersey: Optional[str]) -> Optional[int]:
        j = jersey_norm(jersey)
        if j and j in by_num:
            return by_num[j].get('playerId')
        nm = norm(name)
        if nm and nm in by_name:
            return by_name[nm].get('playerId')
        last = (nm.split(' ')[-1] if nm else '')
        cands = by_last.get(last, [])
        if len(cands) == 1:
            return cands[0].get('playerId')
        return None

    # External suggest/StatsAPI lookups removed; only map to players present on current roster.

    # Build helpers for roster position lookup (playerId -> F/D/G)
    roster_pos_by_pid: Dict[int, str] = {}
    roster_jersey_by_pid: Dict[int, str] = {}
    for p in roster:
        try:
            pid = int(p.get('playerId')) if p.get('playerId') is not None else None
        except Exception:
            pid = None
        if pid is None:
            continue
        # Record jersey from roster for optional backfill
        jn = jersey_norm(p.get('sweaterNumber'))
        if jn:
            roster_jersey_by_pid[pid] = jn
        pos_raw = (p.get('pos') or '').strip().upper()
        pos_final = 'F' if pos_raw.startswith(('C','L','R','F')) else ('D' if pos_raw.startswith('D') else ('G' if pos_raw.startswith('G') else ''))
        if pos_final:
            roster_pos_by_pid[pid] = pos_final

    # Step 1: Consider all scraped items together (order preserved approximately)
    def is_valid_name(name: Optional[str]) -> bool:
        n = (name or '').strip()
        if not n:
            return False
        # Drop obvious noise from DF widgets
        bad_tokens = {'games stats', 'game stats', 'stats', 'lines', 'pairs'}
        if n.lower() in bad_tokens:
            return False
        return any(c.isalpha() for c in n)

    all_items: List[Dict] = []
    for key in ('forwards', 'defense', 'goalies'):
        all_items.extend(list(lineup.get(key, [])))
    # Prefer entries with an explicit jersey so they win ahead of name-only candidates
    def _has_jersey(it: Dict) -> bool:
        j = jersey_norm(it.get('jersey'))
        return bool(j)
    try:
        all_items.sort(key=lambda it: (0 if _has_jersey(it) else 1))
    except Exception:
        pass

    # Step 2: Map to roster playerIds, enforce membership, then reclassify using roster positions
    f_items: List[Dict] = []
    d_items: List[Dict] = []
    g_items: List[Dict] = []
    seen_pid: set[int] = set()
    for it in all_items:
        # Strictly enforce team code from scrape if available; skip if missing or mismatched
        tc = (it.get('teamCode') or '').strip().upper()
        if team_abbrev and tc != team_abbrev.strip().upper():
            continue
        # Basic name sanity
        if not is_valid_name(it.get('name')):
            continue
        # Resolve strictly against current roster by jersey and name
        pid = match_player(it.get('name'), it.get('jersey'))
        if pid is None:
            continue
        try:
            pid_int = int(pid)
        except Exception:
            pid_int = None
        if (pid_int is None) or (pid_int in seen_pid):
            continue
        # Require membership on current roster; otherwise skip
        if pid_int not in roster_ids:
            continue
        pos_final = roster_pos_by_pid.get(pid_int, '')
        it2 = {
            'name': it.get('name') or '',
            'jersey': (jersey_norm(it.get('jersey')) or roster_jersey_by_pid.get(pid_int, '')),
            'playerId': pid_int,
        }
        # Classify: roster-derived pos wins; otherwise use scraped pos hint
        scraped_pos = (it.get('pos') or '').strip().upper()
        if pos_final == 'F' or (not pos_final and scraped_pos[:1] in ('F','C','L','R')):
            f_items.append(it2)
        elif pos_final == 'D' or (not pos_final and scraped_pos.startswith('D')):
            d_items.append(it2)
        elif pos_final == 'G' or (not pos_final and scraped_pos.startswith('G')):
            g_items.append(it2)
        seen_pid.add(pid_int)

    # No additional team verification needed; roster membership guarantees correctness.

    # Step 3: Assign units by order (overwrite any existing unit field)
    def assign_units(lst: List[Dict], unit_patterns: List[str]) -> List[Dict]:
        out: List[Dict] = []
        for i, it in enumerate(lst):
            it2 = dict(it)
            if unit_patterns:
                it2['unit'] = unit_patterns[i] if i < len(unit_patterns) else 'EXT'
            else:
                it2['unit'] = 'EXT'
            out.append(it2)
        return out

    f_units = [f"{c}{n}" for n in range(1,5) for c in ('LW','C','RW')]  # LW1,C1,RW1,...,LW4,C4,RW4
    d_units = [f"{c}{n}" for n in range(1,4) for c in ('LD','RD')]       # LD1,RD1,LD2,RD2,LD3,RD3
    # Per requirement: treat the backup goalie as EXT
    g_units = ['G1','EXT']

    mapped = {'team': team_abbrev, 'forwards': assign_units(f_items, f_units), 'defense': assign_units(d_items, d_units), 'goalies': assign_units(g_items, g_units)}

    return mapped


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
                # roster: use current roster endpoint only (source of truth for playerIds)
                roster = fetch_current_roster(t)
                # goalie fallback if needed
                if (not lineup.get('goalies')) and roster:
                    def jersey_norm(s: Optional[str]) -> str:
                        digits = ''.join(ch for ch in str(s or '') if ch.isdigit())
                        return str(int(digits)) if digits.isdigit() else ''
                    gl = [p for p in roster if (p.get('pos') == 'G')][:2]
                    lineup['goalies'] = [
                        {'name': (p.get('name') or ''), 'jersey': jersey_norm(p.get('sweaterNumber')), 'pos': 'G', 'unit': f'G{i+1}'}
                        for i, p in enumerate(gl)
                    ]
                mapped = map_lineup_to_player_ids(lineup, roster, t)
                # Attach generation timestamp for downstream consumers
                try:
                    mapped['generated_at'] = datetime.utcnow().isoformat() + 'Z'
                except Exception:
                    pass
            except Exception as e:
                print(f"[warn] mapping failed for {t}: {e}")
                mapped = {'team': t, 'forwards': [], 'defense': [], 'goalies': []}
                try:
                    mapped['generated_at'] = datetime.utcnow().isoformat() + 'Z'
                except Exception:
                    pass
            combined[t] = mapped
            if not args.quiet:
                fct = len(mapped.get('forwards', [])); dct = len(mapped.get('defense', [])); gct = len(mapped.get('goalies', []))
                print(f"  -> F:{fct} D:{dct} G:{gct}")

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
        # If goalies section came back empty, fallback to roster (top 2 goalies)
        if (not lineup.get('goalies')) and roster:
            def jersey_norm(s: Optional[str]) -> str:
                digits = ''.join(ch for ch in str(s or '') if ch.isdigit())
                return str(int(digits)) if digits.isdigit() else ''
            gl = [p for p in roster if (p.get('pos') == 'G')]
            gl = gl[:2]
            lineup['goalies'] = [
                {'name': (p.get('name') or ''), 'jersey': jersey_norm(p.get('sweaterNumber')), 'pos': 'G', 'unit': f'G{i+1}'}
                for i, p in enumerate(gl)
            ]
        mapped = map_lineup_to_player_ids(lineup, roster, team)
    except Exception as e:
        print(f"[error] roster/map failed: {e}", file=sys.stderr)
        return 4

    # Fallback: if the page was JS-rendered and yielded no names, populate from roster groups
    if (not mapped.get('forwards')) and (not mapped.get('defense')) and (not mapped.get('goalies')) and roster:
        def jersey_norm(s: Optional[str]) -> str:
            digits = ''.join(ch for ch in str(s or '') if ch.isdigit())
            return str(int(digits)) if digits.isdigit() else ''
        f = [
            {'name': (p.get('name') or ''), 'jersey': jersey_norm(p.get('sweaterNumber')), 'pos': 'F', 'unit': 'F', 'playerId': p.get('playerId')}
            for p in roster if (p.get('pos') == 'F')
        ]
        d = [
            {'name': (p.get('name') or ''), 'jersey': jersey_norm(p.get('sweaterNumber')), 'pos': 'D', 'unit': 'D', 'playerId': p.get('playerId')}
            for p in roster if (p.get('pos') == 'D')
        ]
        g = [
            {'name': (p.get('name') or ''), 'jersey': jersey_norm(p.get('sweaterNumber')), 'pos': 'G', 'unit': 'G', 'playerId': p.get('playerId')}
            for p in roster if (p.get('pos') == 'G')
        ]
        mapped = {'team': team, 'forwards': f, 'defense': d, 'goalies': g}

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
                mapped_out['generated_at'] = datetime.utcnow().isoformat() + 'Z'
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
