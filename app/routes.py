from __future__ import annotations

import os
import csv
import re
import math
import bisect
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import requests
import numpy as np  # for numeric handling in model inference
import joblib       # to load pickled models
from flask import Blueprint, jsonify, render_template

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore


# Blueprint
main_bp = Blueprint('main', __name__)
@main_bp.route('/')
def index_page():
    """Frontpage Schedule view."""
    return render_template('index.html', teams=TEAM_ROWS, active_tab='Schedule', show_season_state=True)


@main_bp.route('/standings')
def standings_page():
    """Standings page."""
    # Provide seasons list for the template (used to seed UI/state)
    seasons = []
    try:
        csv_path = os.path.join(os.getcwd(), 'Last_date.csv')
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                rdr = csv.DictReader(f)
                seasons = sorted({int(row['Season']) for row in rdr if row.get('Season')}, reverse=True)
    except Exception:
        seasons = []
    # Convert to list of objects for template parity
    season_objs = [{ 'season': s } for s in seasons]
    return render_template('standings.html', teams=TEAM_ROWS, seasons=season_objs, active_tab='Standings', show_season_state=False)


@main_bp.route('/api/seasons/<team_code>')
def api_seasons(team_code: str):
    """Return seasons for a given team using NHL club-stats-season endpoint.
    Shape: [{ "season": 20242025, "gameTypes": ["2", "3"] }, ...]
    """
    team = (team_code or '').upper().strip()
    if not team:
        return jsonify([])
    url = f'https://api-web.nhle.com/v1/club-stats-season/{team}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return jsonify([])
        data = r.json()
        out = []
        if isinstance(data, list):
            for it in data:
                if not isinstance(it, dict):
                    continue
                season_val = it.get('season')
                gtypes = it.get('gameTypes')
                try:
                    season_int = int(season_val) if season_val is not None else None
                except Exception:
                    season_int = None
                if season_int is None or not isinstance(gtypes, list):
                    continue
                # Normalize gameTypes to strings for UI compatibility
                gts = [str(gt) for gt in gtypes if gt is not None]
                out.append({ 'season': season_int, 'gameTypes': gts })
        # Sort descending by season
        out.sort(key=lambda x: x['season'], reverse=True)
        return jsonify(out)
    except Exception:
        return jsonify([])


@main_bp.route('/api/standings/<int:season>')
def api_standings(season: int):
    """Proxy NHL standings for a season, pass-through relevant fields."""
    def _current_season_id(now: Optional[datetime] = None) -> int:
        d = now or datetime.utcnow()
        y = d.year
        # NHL seasons typically begin around Sep/Oct; use Sep (9) as boundary
        if d.month >= 9:
            start_y = y
            end_y = y + 1
        else:
            start_y = y - 1
            end_y = y
        return start_y * 10000 + end_y

    def _normalize(data: Any) -> Any:
        if isinstance(data, dict) and 'standings' in data:
            return {'standings': data.get('standings') or []}
        if isinstance(data, list):
            return {'standings': data}
        return {'standings': []}

    # Try a series of endpoints known to serve standings across seasons
    urls: List[str] = []
    # If current season, 'now' is reliable
    if season == _current_season_id():
        urls.append('https://api-web.nhle.com/v1/standings/now')
    # If we have a last-date for this season, prefer date-based endpoints
    last_date = LAST_DATES.get(season)
    if last_date:
        urls.append(f'https://api-web.nhle.com/v1/standings/{last_date}')
        urls.append(f'https://api-web.nhle.com/v1/standings/{last_date}?gameType=2')
    # Season-coded variants
    urls.extend([
        f'https://api-web.nhle.com/v1/standings/{season}',
        f'https://api-web.nhle.com/v1/standings/{season}?gameType=2',
        f'https://api-web.nhle.com/v1/standings-season/{season}',
        f'https://api-web.nhle.com/v1/standings-season/{season}?gameType=2',
        f'https://api-web.nhle.com/v1/standings?season={season}',
        f'https://api-web.nhle.com/v1/standings?season={season}&gameType=2',
    ])

    last_status: Optional[int] = None
    try:
        for url in urls:
            try:
                r = requests.get(url, timeout=25)
            except Exception:
                continue
            last_status = r.status_code
            if r.status_code == 200:
                try:
                    data = r.json()
                except Exception:
                    continue
                return jsonify(_normalize(data))
        # If none succeeded and it's the current season, try 'now' one last time
        if season == _current_season_id():
            try:
                r2 = requests.get('https://api-web.nhle.com/v1/standings/now', timeout=25)
                if r2.status_code == 200:
                    return jsonify(_normalize(r2.json()))
            except Exception:
                pass
        # Final fallback: use stats REST standings by season (gameTypeId=2 regular season)
        try:
            stats_url = (
                'https://api.nhle.com/stats/rest/en/team/standings'
                '?isAggregate=false&reportType=basic&isGame=true&reportName=teamstandings'
                f'&cayenneExp=seasonId={season}%20and%20gameTypeId=2'
            )
            rs = requests.get(stats_url, timeout=30)
            last_status = rs.status_code
            if rs.status_code == 200:
                js = rs.json()
                rows = js.get('data') if isinstance(js, dict) else None
                out = []
                if isinstance(rows, list):
                    # Build team logo lookup
                    logo_by_abbrev = {}
                    try:
                        for tr in TEAM_ROWS:
                            ab = (tr.get('Team') or '').upper()
                            logo_by_abbrev[ab] = tr.get('Logo') or ''
                    except Exception:
                        logo_by_abbrev = {}
                    for rrow in rows:
                        try:
                            ab = (rrow.get('teamAbbrev') or rrow.get('teamAbbrevDefault') or '').upper()
                            gp = rrow.get('gamesPlayed') or rrow.get('gp') or 0
                            pts = rrow.get('points') or rrow.get('pts') or 0
                            w = rrow.get('wins') or rrow.get('w') or 0
                            l = rrow.get('losses') or rrow.get('l') or 0
                            otl = rrow.get('otLosses') or rrow.get('otl') or rrow.get('overtimeLosses') or 0
                            ties = rrow.get('ties') or 0
                            gf = rrow.get('goalsFor') or rrow.get('gf') or 0
                            ga = rrow.get('goalsAgainst') or rrow.get('ga') or 0
                            diff = (gf or 0) - (ga or 0)
                            ppct = rrow.get('pointsPercentage') or rrow.get('pointPctg')
                            if ppct is None:
                                try:
                                    ppct = (float(pts) / (2.0 * float(gp))) if gp else 0.0
                                except Exception:
                                    ppct = 0.0
                            l10w = rrow.get('lastTenWins') or rrow.get('l10Wins') or 0
                            l10l = rrow.get('lastTenLosses') or rrow.get('l10Losses') or 0
                            l10o = rrow.get('lastTenOtLosses') or rrow.get('l10OtLosses') or 0
                            # Streak
                            streak_code = rrow.get('streakCode') or rrow.get('streakType') or ''
                            streak_num = rrow.get('streakNumber') or rrow.get('streakCount') or 0
                            # Grouping
                            div_name = rrow.get('divisionName') or rrow.get('divisionAbbrev') or ''
                            conf_name = rrow.get('conferenceName') or rrow.get('conferenceAbbrev') or ''
                            out.append({
                                'teamAbbrev': ab,
                                'divisionName': div_name,
                                'conferenceName': conf_name,
                                'gamesPlayed': gp,
                                'points': pts,
                                'wins': w,
                                'losses': l,
                                'ties': ties,
                                'otLosses': otl,
                                'goalFor': gf,
                                'goalAgainst': ga,
                                'goalDifferential': diff,
                                'pointPctg': ppct,
                                'l10Wins': l10w,
                                'l10Losses': l10l,
                                'l10OtLosses': l10o,
                                'streakCode': (str(streak_code)[:1]).upper() if streak_code else '',
                                'streakCount': streak_num or 0,
                                'teamLogo': logo_by_abbrev.get(ab, ''),
                            })
                        except Exception:
                            continue
                return jsonify({'standings': out})
        except Exception:
            pass
        return jsonify({'error': 'Upstream error', 'status': last_status or 502}), 502
    except Exception:
        return jsonify({'error': 'Failed to fetch standings'}), 502


# Load Teams.csv (used for theming and lookups in templates)
def _load_teams_csv() -> List[Dict[str, str]]:
    paths = [
        os.path.join(os.path.dirname(__file__), '..', 'Teams.csv'),
        os.path.join(os.getcwd(), 'Teams.csv'),
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f)
                    return [row for row in reader]
        except Exception:
            continue
    return []


TEAM_ROWS: List[Dict[str, str]] = _load_teams_csv()

# Load Last_date.csv mapping Season -> Last_Date (YYYY-MM-DD)
def _load_last_dates() -> Dict[int, str]:
    paths = [
        os.path.join(os.getcwd(), 'Last_date.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Last_date.csv'),
    ]
    out: Dict[int, str] = {}
    for p in paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    rdr = csv.DictReader(f)
                    for row in rdr:
                        try:
                            s = row.get('Season')
                            d = row.get('Last_Date')
                            if s and d:
                                out[int(str(s).strip())] = str(d).strip()
                        except Exception:
                            continue
                break
        except Exception:
            continue
    return out

LAST_DATES: Dict[int, str] = _load_last_dates()

# Lazy-loaded cache for BoxID lookups by (x,y)
_BOXID_MAP: Optional[Dict[Tuple[int, int], Tuple[str, str, int]]] = None

def _get_boxid_map() -> Dict[Tuple[int, int], Tuple[str, str, int]]:
    global _BOXID_MAP
    if _BOXID_MAP is not None:
        # If the cached map appears degenerate (e.g., only x==0 keys due to BOM/header issues), rebuild it
        try:
            if any(k[0] != 0 for k in _BOXID_MAP.keys()):
                return _BOXID_MAP
        except Exception:
            if _BOXID_MAP:
                return _BOXID_MAP
        _BOXID_MAP = None
    # Try locating BoxID.csv next to Teams.csv or in CWD
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), '..', 'BoxID.csv'),
        os.path.join(os.getcwd(), 'BoxID.csv'),
    ]
    mapping: Dict[Tuple[int, int], Tuple[str, str, int]] = {}
    for p in candidate_paths:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8', newline='') as f:
                    rdr = csv.DictReader(f)
                    # Build a case-insensitive and BOM-tolerant field map
                    raw_fields = rdr.fieldnames or []
                    fns = [fn.strip() for fn in raw_fields]
                    def norm_field(s: str) -> str:
                        return s.lstrip('\ufeff').lower().strip()
                    lower_map = {norm_field(fn): fn for fn in fns}
                    col_x = lower_map.get('x')
                    col_y = lower_map.get('y')
                    col_boxid = lower_map.get('boxid')
                    col_boxid_rev = lower_map.get('boxid_rev')
                    # Boxsize/BoxSize
                    col_boxsize = lower_map.get('boxsize')
                    for row in rdr:
                        try:
                            # Access with original name; if missing, attempt BOM-stripped variants
                            xi_raw = row.get(col_x) if col_x else (row.get('x') or row.get('\ufeffx'))
                            yi_raw = row.get(col_y) if col_y else row.get('y')
                            bid_raw = row.get(col_boxid) if col_boxid else (row.get('BoxID') or row.get('boxid'))
                            bre_raw = row.get(col_boxid_rev) if col_boxid_rev else (row.get('BoxID_rev') or row.get('boxid_rev'))
                            bsz_raw = row.get(col_boxsize) if col_boxsize else (row.get('Boxsize') or row.get('BoxSize') or row.get('boxsize'))
                            xi = int(float(xi_raw or 0))
                            yi = int(float(yi_raw or 0))
                            bid = str(bid_raw or '').strip() or None
                            bre = str(bre_raw or '').strip() or None
                            bsi = int(str(bsz_raw).strip()) if (bsz_raw is not None and str(bsz_raw).strip().lstrip('-').isdigit()) else None
                            if bid and bre and bsi is not None:
                                mapping[(xi, yi)] = (bid, bre, bsi)
                        except Exception:
                            continue
                break
        except Exception:
            continue
    _BOXID_MAP = mapping
    return mapping


@main_bp.route('/api/team/<team_code>/<int:season>/schedule')
def api_team_schedule(team_code: str, season: int):
    """Team schedule for a given season using NHL club-schedule-season endpoint."""
    url = f"https://api-web.nhle.com/v1/club-schedule-season/{team_code.upper()}/{season}"
    try:
        r = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Failed to fetch schedule'}), 502
    if r.status_code != 200:
        return jsonify({'error': 'Failed to fetch schedule'}), 502
    data = r.json()
    games_out = []
    for g in data.get('games', []) or []:
        game_date = g.get('gameDate') or g.get('startTimeUTC')
        try:
            dt = datetime.fromisoformat(game_date.replace('Z', '+00:00')) if game_date else None
        except Exception:
            dt = None
        home = (g.get('homeTeam') or {}).get('abbrev')
        away = (g.get('awayTeam') or {}).get('abbrev')
        home_score = (g.get('homeTeam') or {}).get('score')
        away_score = (g.get('awayTeam') or {}).get('score')
        opp = away if (home and home == team_code.upper()) else home
        is_home = bool(home and home == team_code.upper())
        status = g.get('gameState') or g.get('gameStatus')
        last_period_type = (g.get('gameOutcome') or {}).get('lastPeriodType') or (g.get('periodDescriptor') or {}).get('periodType')
        games_out.append({
            'date': dt.isoformat() if dt else game_date,
            'home': home,
            'away': away,
            'opponent': opp,
            'is_home': is_home,
            'status': status,
            'gameType': g.get('gameType') or g.get('gameTypeId'),
            'home_score': home_score,
            'away_score': away_score,
            'lastPeriodType': last_period_type,
            'id': g.get('id') or g.get('gamePk'),
        })
    return jsonify(games_out)

# Alias to match frontend fetch path `/api/schedule/{team}/{season}`
@main_bp.route('/api/schedule/<team_code>/<int:season>')
def api_schedule_alias(team_code: str, season: int):
    return api_team_schedule(team_code, season)


@main_bp.route('/game/<int:game_id>')
def game_page(game_id: int):
    """Render a game detail page."""
    teams = TEAM_ROWS
    return render_template('game.html', game_id=game_id, teams=teams, active_tab='Schedule', show_season_state=False)


@main_bp.route('/api/game/<int:game_id>/boxscore')
def api_game_boxscore(game_id: int):
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore'
    try:
        resp = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    data = resp.json()
    # Pass through mostly untouched; rename id to gameId for consistency
    if 'id' in data and 'gameId' not in data:
        data['gameId'] = data['id']
    return jsonify(data)


@main_bp.route('/api/game/<int:game_id>/right-rail')
def api_game_right_rail(game_id: int):
    """Proxy NHL right-rail endpoint for a game to avoid browser CORS."""
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/right-rail'
    try:
        resp = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    try:
        data = resp.json()
    except Exception:
        # Fall back to raw text if upstream is not JSON
        return jsonify({'error': 'Invalid upstream format'}), 502
    return jsonify(data)


@main_bp.route('/api/game/<int:game_id>/play-by-play')
def api_game_pbp(game_id: int):
    """Fetch NHL play-by-play and map to requested wide schema."""
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
    try:
        resp = requests.get(url, timeout=25)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    data = resp.json()

    # Fetch skater bios for this game to retrieve shoots/catches for players
    shoots_map: Dict[int, str] = {}
    try:
        gid_for_bios = data.get('id') or game_id
        bios_url = f"https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=gameId={gid_for_bios}"
        r_bios = requests.get(bios_url, timeout=20)
        if r_bios.status_code == 200:
            bios_json = r_bios.json()
            rows = bios_json.get('data') if isinstance(bios_json, dict) else []
            if isinstance(rows, list):
                for row in rows:
                    try:
                        pid = row.get('playerId')
                        sc = row.get('shootsCatches') or row.get('shoots') or row.get('ShootsCatches')
                        if isinstance(pid, int) and sc:
                            shoots_map[pid] = str(sc).strip().upper()[:1]
                    except Exception:
                        continue
    except Exception:
        shoots_map = {}

    plays_raw = data.get('plays', []) if isinstance(data, dict) else []
    away_team = (data.get('awayTeam') or {})
    home_team = (data.get('homeTeam') or {})
    away_id = away_team.get('id')
    home_id = home_team.get('id')
    away_abbrev = away_team.get('abbrev')
    home_abbrev = home_team.get('abbrev')
    # RinkVenue per request: Season-HomeTeam
    rink_venue_value = None
    try:
        sv = data.get('season')
        if sv is not None and home_abbrev:
            rink_venue_value = f"{sv}-{home_abbrev}"
    except Exception:
        rink_venue_value = None
    roster = {r.get('playerId'): r for r in data.get('rosterSpots', [])}

    def player_name(pid: Optional[int]) -> Optional[str]:
        r = roster.get(pid)
        if not r:
            return None
        fn = (r.get('firstName') or {}).get('default') if isinstance(r.get('firstName'), dict) else r.get('firstName')
        ln = (r.get('lastName') or {}).get('default') if isinstance(r.get('lastName'), dict) else r.get('lastName')
        if fn and ln:
            return f"{fn} {ln}".strip()
        return fn or ln

    def parse_time_to_seconds(t: str) -> Optional[int]:
        try:
            mm, ss = t.split(':')
            return int(mm) * 60 + int(ss)
        except Exception:
            return None

    def strength_from_situation(code: str, event_owner_team_id: Optional[int]) -> str:
        s = code or ''
        if not s:
            return ''
        # Empty-net handling: if situation code starts with '0' => Away ENF/Home ENA; ends with '0' => Home ENF/Away ENA
        away_empty = len(s) >= 1 and s[0] == '0'
        home_empty = len(s) >= 1 and s[-1] == '0'
        if away_empty or home_empty:
            if event_owner_team_id == away_id:
                return 'ENF' if away_empty else 'ENA'
            if event_owner_team_id == home_id:
                return 'ENF' if home_empty else 'ENA'
            # Unknown owner: fall back to neutral numeric state when possible
            # If both empty or unknown, return empty string to avoid misleading tag
            return ''
        # Numeric skater counts: typical 4-digit code, use middle digits for A/H skaters
        if len(s) == 4 and s.isdigit():
            away_skaters = int(s[1])
            home_skaters = int(s[2])
            if event_owner_team_id is None:
                return f"{away_skaters}v{home_skaters}"
            if event_owner_team_id == home_id:
                return f"{home_skaters}v{away_skaters}"
            if event_owner_team_id == away_id:
                return f"{away_skaters}v{home_skaters}"
            return f"{away_skaters}v{home_skaters}"
        # Fallback to raw code if format unrecognized
        return s

    # Running score tracking (pre-event state). We'll update AFTER mapping current event.
    running_away = 0
    running_home = 0

    # Determine orientation using SUM(x) of shot attempts grouped by (period, event-owner team)
    period_team_sum_x: Dict[Tuple[int, int], float] = {}
    period_sum_all: Dict[int, float] = {}
    for pl in plays_raw:
        pd = ((pl.get('periodDescriptor') or {}).get('number'))
        try:
            pd_key = int(pd) if pd is not None else None
        except Exception:
            pd_key = None
        if not pd_key:
            continue
        tc = pl.get('typeCode')
        if tc not in (505, 506, 507, 508):  # shot attempts
            continue
        d0 = pl.get('details') or {}
        x0 = d0.get('xCoord')
        if x0 is None:
            continue
        try:
            xx = float(x0)
        except Exception:
            continue
        period_sum_all[pd_key] = period_sum_all.get(pd_key, 0.0) + xx
        owner0 = d0.get('eventOwnerTeamId')
        if isinstance(owner0, int) and owner0 in (home_id, away_id):
            key_t = (pd_key, owner0)
            period_team_sum_x[key_t] = period_team_sum_x.get(key_t, 0.0) + xx

    # Precompute shift slices and on-ice players from our own shifts endpoint
    slices: List[Tuple[int, int, int]] = []  # (start, end, ShiftIndex)
    starts: List[int] = []
    slice_players: Dict[int, List[Dict]] = {}
    try:
        resp_shifts = api_game_shifts(game_id)
        js = None
        status_code = 200
        if isinstance(resp_shifts, tuple):
            resp_obj, status_code = resp_shifts
            js = resp_obj.get_json(silent=True) if hasattr(resp_obj, 'get_json') else None
        else:
            js = resp_shifts.get_json(silent=True) if hasattr(resp_shifts, 'get_json') else None
        rows = (js.get('shifts') or []) if isinstance(js, dict) and status_code == 200 else []
        by_idx: Dict[int, Tuple[int, int]] = {}
        for r in rows:
            si = r.get('ShiftIndex'); st = r.get('Start'); en = r.get('End')
            if si is None or st is None or en is None:
                continue
            try:
                sii = int(si); sti = int(st); eni = int(en)
            except Exception:
                continue
            if sii not in by_idx:
                by_idx[sii] = (sti, eni)
            else:
                a, b = by_idx[sii]
                by_idx[sii] = (min(a, sti), max(b, eni))
            # collect players for this slice
            pl_id = r.get('PlayerID'); pl_nm = r.get('Name'); pl_pos = (r.get('Position') or '').upper(); pl_tm = r.get('Team')
            if pl_id is not None and pl_tm is not None:
                slice_players.setdefault(sii, []).append({'PlayerID': pl_id, 'Name': pl_nm, 'Position': pl_pos, 'Team': pl_tm})
        slices = sorted([(v[0], v[1], k) for k, v in by_idx.items()], key=lambda x: x[0])
        starts = [s for s, _, _ in slices]
    except Exception:
        slices = []
        starts = []
        slice_players = {}

    # Precompute on-ice string fields per ShiftIndex
    onice_cache: Dict[int, Dict[str, Optional[str]]] = {}
    for si, plist in slice_players.items():
        # Partition by team and position
        def filter_and_sort(team_abbr: str, pos_code: str):
            flt = [p for p in plist if (p.get('Team') == team_abbr and (p.get('Position') or '').upper() == pos_code)]
            # Ensure numeric PlayerID sort
            def to_int(x):
                try:
                    return int(x)
                except Exception:
                    return 0
            flt_sorted = sorted(flt, key=lambda p: to_int(p.get('PlayerID')))
            ids = ' '.join(str(p.get('PlayerID')) for p in flt_sorted if p.get('PlayerID') is not None)
            names = ' - '.join(str(p.get('Name')) for p in flt_sorted if p.get('Name'))
            return ids or None, names or None

        hf_id, hf_nm = filter_and_sort(str(home_abbrev or ''), 'F')
        hd_id, hd_nm = filter_and_sort(str(home_abbrev or ''), 'D')
        hg_id, hg_nm = filter_and_sort(str(home_abbrev or ''), 'G')
        af_id, af_nm = filter_and_sort(str(away_abbrev or ''), 'F')
        ad_id, ad_nm = filter_and_sort(str(away_abbrev or ''), 'D')
        ag_id, ag_nm = filter_and_sort(str(away_abbrev or ''), 'G')

        onice_cache[si] = {
            'Home_Forwards_ID': hf_id, 'Home_Forwards': hf_nm,
            'Home_Defenders_ID': hd_id, 'Home_Defenders': hd_nm,
            'Home_Goalie_ID': hg_id, 'Home_Goalie': hg_nm,
            'Away_Forwards_ID': af_id, 'Away_Forwards': af_nm,
            'Away_Defenders_ID': ad_id, 'Away_Defenders': ad_nm,
            'Away_Goalie_ID': ag_id, 'Away_Goalie': ag_nm,
        }

    def find_shift_index_for_event(gt: int, event_key: Optional[str]) -> Optional[int]:
        if not slices:
            return None
        ev = (event_key or '').lower()
        ev_norm = ev.replace('-', '_')
        # Check if on a slice start boundary
        k = bisect.bisect_left(starts, gt)
        if k < len(starts) and starts[k] == gt:
            # Boundary rule: faceoff/period_start take later (max) slice; others previous (min) slice
            if ev_norm in ('faceoff', 'period_start'):
                idx = k
            else:
                idx = k - 1 if k > 0 else k
            return slices[idx][2] if 0 <= idx < len(slices) else None
        # Otherwise, select slice with start <= gt < end
        i = bisect.bisect_right(starts, gt) - 1
        if i < 0 or i >= len(slices):
            return None
        s0, e0, si0 = slices[i]
        if gt < e0:
            return si0
        if gt == e0:
            # End boundary equals next start
            if (i + 1) < len(slices) and starts[i + 1] == gt:
                if ev_norm in ('faceoff', 'period_start'):
                    return slices[i + 1][2]
                return si0
        return None

    # StrengthState2 mapping per spec
    def remap_strength_state(s: Optional[str]) -> Optional[str]:
        if not s:
            return s
        s2 = str(s).upper()
        if s2 in ("5V4", "ENF"):
            return "PP1"
        if s2 in ("5V3", "4V3"):
            return "PP2"
        if s2 in ("4V5", "3V5", "3V4"):
            return "SH"
        return s

    # BoxID2 mapping per spec (combines left/right and splits by handedness)
    def compute_boxid2(boxid: Optional[str], shoots: Optional[str]) -> str:
        b = (boxid or '').upper().strip()
        h = (shoots or '').upper().strip()[:1] if shoots else ''
        # Helper shortcuts
        is_r_or_null = (h == 'R' or h == '')
        is_l_or_null = (h == 'L' or h == '')
        if b in ('O01', 'O03'):
            return 'O01'
        if b == 'O02':
            return 'O02'
        if b == 'O04':
            return 'O04-W' if is_r_or_null else 'O04-S'
        if b == 'O05':
            return 'O05-W' if is_r_or_null else 'O05-S'
        if b == 'O06':
            return 'O06-W' if is_r_or_null else 'O06-S'
        if b == 'O07':
            return 'O07'
        if b == 'O08':
            return 'O06-W' if is_l_or_null else 'O06-S'
        if b == 'O09':
            return 'O05-W' if is_l_or_null else 'O05-S'
        if b == 'O10':
            return 'O04-W' if is_l_or_null else 'O04-S'
        if b == 'O11':
            return 'O11'
        if b == 'O12':
            return 'O12-W' if is_r_or_null else 'O12-S'
        if b == 'O13':
            return 'O13-W' if is_r_or_null else 'O13-S'
        if b == 'O14':
            return 'O14-W' if is_r_or_null else 'O14-S'
        if b == 'O15':
            return 'O15'
        if b == 'O16':
            return 'O14-W' if is_l_or_null else 'O14-S'
        if b == 'O17':
            return 'O13-W' if is_l_or_null else 'O13-S'
        if b == 'O18':
            return 'O12-W' if is_l_or_null else 'O12-S'
        if b == 'O19':
            return 'O19-W' if is_r_or_null else 'O19-S'
        if b == 'O20':
            return 'O20-W' if is_r_or_null else 'O20-S'
        if b == 'O21':
            return 'O21'
        if b == 'O22':
            return 'O20-W' if is_l_or_null else 'O20-S'
        if b == 'O23':
            return 'O19-W' if is_l_or_null else 'O19-S'
        if b == 'O24':
            return 'O24-W' if is_r_or_null else 'O24-S'
        if b == 'O25':
            return 'O25'
        if b == 'O26':
            return 'O24-W' if is_l_or_null else 'O24-S'
        return 'D_or_N'

    mapped: List[Dict] = []
    for idx_pl, pl in enumerate(plays_raw):
        period = ((pl.get('periodDescriptor') or {}).get('number'))
        time_in_period = pl.get('timeInPeriod') or ''
        type_code = pl.get('typeCode')
        event_key = pl.get('typeDescKey')
        details = pl.get('details') or {}
        situation = pl.get('situationCode') or ''
        strength = strength_from_situation(situation, details.get('eventOwnerTeamId'))
        x = details.get('xCoord')
        y = details.get('yCoord')
        zone = details.get('zoneCode')
        reason = details.get('reason')
        secondary_reason = details.get('secondaryReason')
        type_code2 = details.get('typeCode') if isinstance(details.get('typeCode'), str) else None
        pen_dur = details.get('duration')
        event_owner = details.get('eventOwnerTeamId')
        event_team_abbrev = away_abbrev if event_owner == away_id else home_abbrev if event_owner == home_id else None
        opponent_abbrev = home_abbrev if event_team_abbrev == away_abbrev else away_abbrev if event_team_abbrev == home_abbrev else None
        goalie_id = details.get('goalieInNetId')
        goalie_name = player_name(goalie_id) if goalie_id else None

        # Collect involved player ids in priority order
        candidate_ids: List[int] = []
        for key in [
            'scoringPlayerId', 'shootingPlayerId', 'playerId', 'hittingPlayerId', 'hitteePlayerId',
            'assist1PlayerId', 'assist2PlayerId', 'blockingPlayerId', 'losingPlayerId', 'winningPlayerId',
            'committedByPlayerId', 'drawnByPlayerId'
        ]:
            pid = details.get(key)
            if pid and pid not in candidate_ids:
                candidate_ids.append(pid)
        p1_id = candidate_ids[0] if len(candidate_ids) > 0 else None
        p2_id = candidate_ids[1] if len(candidate_ids) > 1 else None
        p3_id = candidate_ids[2] if len(candidate_ids) > 2 else None
        p1_name = player_name(p1_id) if p1_id else None
        p2_name = player_name(p2_id) if p2_id else None
        p3_name = player_name(p3_id) if p3_id else None

        # Shot / goal classification
        is_goal = (type_code == 505)
        is_sog = (type_code == 506) or is_goal
        is_miss = (type_code == 507)
        is_block = (type_code == 508)
        # Blocked shots: swap zone O <-> D for display only (coords are shooter-perspective upstream)
        if is_block and zone in ('O', 'D'):
            zone = 'O' if zone == 'D' else 'D'

        # Normalize coordinates so offensive zone is to the right (positive x) for the period
        nx: Optional[float] = None
        ny: Optional[float] = None
        try:
            pd_key2 = int(period) if period is not None else None
        except Exception:
            pd_key2 = None
        sign = 1
        if pd_key2 is not None:
            if isinstance(event_owner, int) and event_owner in (home_id, away_id):
                key = (pd_key2, event_owner)
                if key in period_team_sum_x:
                    sign = 1 if period_team_sum_x[key] >= 0 else -1
                else:
                    opp = home_id if event_owner == away_id else away_id if event_owner == home_id else None
                    if isinstance(opp, int) and (pd_key2, opp) in period_team_sum_x:
                        sign = -1 if period_team_sum_x[(pd_key2, opp)] >= 0 else 1
                    else:
                        sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
            else:
                sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
        try:
            nx = (float(x) * sign) if x is not None else None
        except Exception:
            nx = None
        try:
            ny = (float(y) * sign) if y is not None else None
        except Exception:
            ny = None

        # ScoreState: goal differential from perspective of event team BEFORE applying current event.
        if event_owner == away_id:
            score_state_val = running_away - running_home
        elif event_owner == home_id:
            score_state_val = running_home - running_away
        else:
            score_state_val = running_away - running_home
        # Bounded ScoreState2 per spec
        score_state2_val = -3 if score_state_val < -2 else (3 if score_state_val > 2 else score_state_val)

        # Possession attempts (Corsi/Fenwick)
        corsi = 1 if (is_goal or is_sog or is_miss or is_block) and event_team_abbrev else 0
        fenwick = 1 if (is_goal or is_sog or is_miss) and event_team_abbrev else 0
        shot = 1 if is_sog else 0

        # Position & shoots from primary player if available
        position = None
        shoots = None
        if p1_id and p1_id in roster:
            pos_code = roster[p1_id].get('positionCode')
            if pos_code:
                c = str(pos_code).strip().upper()[:1]
                position = 'F' if c in ('C', 'L', 'R') else c
        # Shoots from bios map (fallback only if present)
        if p1_id and p1_id in shoots_map:
            shoots = shoots_map.get(p1_id)

        # gameTime calculation in seconds
        secs_elapsed = parse_time_to_seconds(time_in_period) or 0
        try:
            game_time = ((period - 1) * 20 * 60 + secs_elapsed) if period else secs_elapsed
        except Exception:
            game_time = secs_elapsed

        # Venue from event team perspective: Home/Away
        venue_ha = 'Home' if event_owner == home_id else ('Away' if event_owner == away_id else '')

        # Shot geometry (feet/degrees) relative to net at (89,0), using normalized coords
        shot_distance = None
        shot_angle = None
        if nx is not None and ny is not None and (is_goal or is_sog or is_miss or is_block):
            try:
                dx = 89.0 - float(nx)
                dy = 0.0 - float(ny)
                dist = (dx * dx + dy * dy) ** 0.5
                ang = math.degrees(math.atan2(abs(dy), dx if dx != 0 else 1e-6))
                shot_distance = round(dist, 2)
                shot_angle = round(ang, 2)
            except Exception:
                pass

        # SeasonState: map gameType to 'regular' or 'playoffs'
        gt = data.get('gameType')
        season_state = 'playoffs' if str(gt) == '3' else 'regular'

        # EventIndex: GameID*10000 + Index (1-based index in plays list)
        try:
            gid_for_idx = int(data.get('id') or game_id)
        except Exception:
            gid_for_idx = int(game_id)
        event_index_val = gid_for_idx * 10000 + (idx_pl + 1)

        # ShiftIndex from slices with boundary rules
        shift_index_val = find_shift_index_for_event(int(game_time), event_key)

        # Prepare on-ice fields from cache by ShiftIndex (if available)
        oi = onice_cache.get(shift_index_val or -1, {})

        # Box geometry left-join on integer grid: round normalized coords and match exactly
        box_id = None
        box_rev = None
        box_size = None
        xi = None
        yi = None
        if nx is not None and ny is not None:
            try:
                xf = float(nx); yf = float(ny)
                # Clamp to rink bounds first
                xf = max(-100.0, min(100.0, xf))
                yf = max(-42.0, min(42.0, yf))
                xi = int(round(xf))
                yi = int(round(yf))
                # Many BoxID grids are defined for x >= 0 and use BoxID_rev for the mirrored side.
                # Use abs(x) to find the grid cell, and keep both BoxID and BoxID_rev from the CSV.
                rec = _get_boxid_map().get((xi, yi))
                if rec:
                    box_id, box_rev, box_size = rec[0], rec[1], rec[2]
            except Exception:
                pass

        # shotType2 normalization per spec
        raw_shot_type = details.get('shotType')
        try:
            shot_type_norm = str(raw_shot_type or '').strip()
        except Exception:
            shot_type_norm = ''
        allowed_types = {"wrist", "tip-in", "snap", "slap", "backhand", "deflected", "wrap-around", ""}
        st_lower = shot_type_norm.lower()
        shot_type2 = shot_type_norm if st_lower in allowed_types else 'other'

        # StrengthState2 remap
        strength2 = remap_strength_state(strength)

        mapped.append({
            'GameID': data.get('id'),
            'Season': data.get('season'),
            'SeasonState': season_state,
            'Venue': venue_ha,
            'Period': period,
            'gameTime': int(game_time),
            'StrengthState': strength,
            'StrengthState2': strength2,
            'typeCode': type_code,
            'Event': event_key,
            'x': nx,
            'y': ny,
            'X': xi,
            'Y': yi,
            'Zone': zone,
            'reason': reason,
            'shotType': details.get('shotType'),
            'shotType2': shot_type2,
            'secondaryReason': secondary_reason,
            'typeCode2': type_code2,
            'PEN_duration': pen_dur,
            'EventTeam': event_team_abbrev,
            'Opponent': opponent_abbrev,
            'Goalie_ID': goalie_id,
            'Goalie': goalie_name,
            'Player1_ID': p1_id,
            'Player1': p1_name,
            'Player2_ID': p2_id,
            'Player2': p2_name,
            'Player3_ID': p3_id,
            'Player3': p3_name,
            'Corsi': corsi,
            'Fenwick': fenwick,
            'Shot': shot,
            'Goal': 1 if is_goal else 0,
            'EventIndex': event_index_val,
            'ShiftIndex': shift_index_val,
            'ScoreState': score_state_val,
            'ScoreState2': score_state2_val,
            'Home_Forwards_ID': oi.get('Home_Forwards_ID'),
            'Home_Forwards': oi.get('Home_Forwards'),
            'Home_Defenders_ID': oi.get('Home_Defenders_ID'),
            'Home_Defenders': oi.get('Home_Defenders'),
            'Home_Goalie_ID': oi.get('Home_Goalie_ID'),
            'Home_Goalie': oi.get('Home_Goalie'),
            'Away_Forwards_ID': oi.get('Away_Forwards_ID'),
            'Away_Forwards': oi.get('Away_Forwards'),
            'Away_Defenders_ID': oi.get('Away_Defenders_ID'),
            'Away_Defenders': oi.get('Away_Defenders'),
            'Away_Goalie_ID': oi.get('Away_Goalie_ID'),
            'Away_Goalie': oi.get('Away_Goalie'),
            'BoxID': box_id,
            'BoxID_rev': box_rev,
            'BoxSize': box_size,
            'BoxID2': None,  # fill after row assembled using Shoots
            # snake_case aliases for downstream consumers expecting these names
            'box_id': box_id,
            'box_rev': box_rev,
            'box_size': box_size,
            'ShotDistance': shot_distance,
            'ShotAngle': shot_angle,
            'Position': position,
            'Shoots': shoots,
            'RinkVenue': rink_venue_value,
            'LastEvent': None,  # to be computed post-pass
            'xG_F': None,
            'xG_S': None,
            'xG_F2': None,
        })

        # AFTER mapping current play, update running score when this is a goal
        if is_goal:
            if 'awayScore' in details and 'homeScore' in details and details.get('awayScore') is not None and details.get('homeScore') is not None:
                try:
                    ra = details.get('awayScore')
                    rh = details.get('homeScore')
                    running_away = int(ra) if ra is not None else running_away
                    running_home = int(rh) if rh is not None else running_home
                except Exception:
                    if event_owner == away_id:
                        running_away += 1
                    elif event_owner == home_id:
                        running_home += 1
            else:
                if event_owner == away_id:
                    running_away += 1
                elif event_owner == home_id:
                    running_home += 1

    # Post-process BoxID2 and LastEvent
    last_event_name: Optional[str] = None
    last_game_time: Optional[int] = None
    for row in mapped:
        # BoxID2 depends on BoxID and Shoots
        row['BoxID2'] = compute_boxid2(row.get('BoxID'), row.get('Shoots'))
        # LastEvent labeling per spec
        if row.get('Fenwick') == 1:
            prev_ev = last_event_name or ''
            gt = row.get('gameTime')
            tsle = (gt - last_game_time) if (gt is not None and last_game_time is not None) else None
            if tsle is not None and tsle < 4 and prev_ev in ('blocked-shot', 'shot-on-goal', 'takeaway', 'giveaway'):
                row['LastEvent'] = 'Rebound'
            elif tsle is not None and tsle < 4:
                row['LastEvent'] = 'Quick'
            else:
                row['LastEvent'] = 'None'
        else:
            row['LastEvent'] = ''
        # Update lag trackers for all events
        last_event_name = row.get('Event') or last_event_name
        last_game_time = row.get('gameTime') if row.get('gameTime') is not None else last_game_time

    # xG computations using pickled models, skipping ENA strength
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'Model')

        # Helper: map season integer like 20142015 to previous, current, next for 3 sliding windows
        def season_prev(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a-1)*10000 + (b-1)
        def season_next(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a+1)*10000 + (b+1)

        # Cache for loaded models keyed by filename
        model_cache: Dict[str, Any] = {}

        def load_model(fname: str):
            if fname in model_cache:
                return model_cache[fname]
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                return None
            try:
                m = joblib.load(path)
                model_cache[fname] = m
                return m
            except Exception:
                return None

        # Build expected dummy column set by inspecting the models or by union of features from data
        # We'll prepare features on the fly similar to training script
        base_feature_cols = [
            "Venue", "shotType2", "ScoreState2", "RinkVenue",
            "StrengthState2", "BoxID2", "LastEvent"
        ]

        # Prepare a pandas DataFrame for plays to compute dummies
        try:
            import pandas as _pd
        except Exception:
            _pd = None

        if _pd is not None and mapped:
            df_all = _pd.DataFrame(mapped)
            # Fill and cast
            for c in base_feature_cols:
                if c in df_all.columns:
                    df_all[c] = df_all[c].fillna('missing').astype(str)
                else:
                    df_all[c] = 'missing'
            # Precompute full dummy matrix; we will slice rows and align columns per model
            X_dummies = _pd.get_dummies(df_all[base_feature_cols]).astype(float)
        else:
            X_dummies = None

        def predict_avg_for_row(idx: int, season_val: Optional[int], model_prefix: str) -> Optional[float]:
            if season_val is None:
                return None
            s_cur = int(season_val)
            # Special-case for 20252026 games: use ..._20222023_20242025.pkl
            if s_cur == 20252026:
                names = [f"{model_prefix}_20222023_20242025.pkl"]
            else:
                s_prev = season_prev(s_cur)
                s_next = season_next(s_cur)
                s_prev2 = season_prev(s_prev)   # s-2
                s_next2 = season_next(s_next)   # s+2
                # Derive three filenames according to spec for season s:
                # 1) (s-2 .. s), 2) (s-1 .. s+1), 3) (s .. s+2)
                names = [
                    f"{model_prefix}_{s_prev2}_{s_cur}.pkl",
                    f"{model_prefix}_{s_prev}_{s_next}.pkl",
                    f"{model_prefix}_{s_cur}_{s_next2}.pkl",
                ]
            models = [load_model(n) for n in names]
            models = [m for m in models if m is not None]
            if not models or X_dummies is None:
                return None
            # Build row feature vector aligned to model's expected features inferred from X_train columns at train time.
            # Since we don't have model.feature_names_, we align by columns used in training code: get_dummies(base_feature_cols)
            x_row = X_dummies.iloc[idx:idx+1].copy()
            preds = []
            for m in models:
                try:
                    x_in = x_row
                    # Some sklearn models may carry feature_names_in_; align in one shot to avoid fragmentation
                    if hasattr(m, 'feature_names_in_'):
                        cols_needed = list(getattr(m, 'feature_names_in_'))
                        x_in = x_row.reindex(columns=cols_needed, fill_value=0.0)
                    # Predict proba
                    p = m.predict_proba(x_in)[:, 1]
                    preds.append(float(p[0]))
                except Exception:
                    # Best effort: try raw predict_proba with current columns
                    try:
                        p = m.predict_proba(x_row)[:, 1]
                        preds.append(float(p[0]))
                    except Exception:
                        continue
            if not preds:
                return None
            return float(sum(preds) / len(preds))

        # Helper for ENA fenwick attempts
        def compute_empty_net_fenwick(sd: Optional[float], sa: Optional[float]) -> Optional[float]:
            if sd is None or sa is None:
                return None
            try:
                val = 1.0 / (1.0 + math.exp(0.013609495*float(sd) + 0.023174225*abs(float(sa)) - 1.97392131))
                return float(val)
            except Exception:
                return None

        # Compute xG_F (Fenwick), xG_S (Shot), xG_F2 (Fenwick) using different model families
        for i_row, row in enumerate(mapped):
            season_val = row.get('Season')
            strength_state = row.get('StrengthState')

            # ENA overrides
            if strength_state == 'ENA':
                # xG_S: 1 for ENA shots on goal
                if row.get('Shot') == 1:
                    row['xG_S'] = 1.0
                # xG_F and xG_F2: logistic formula for ENA fenwick attempts
                if row.get('Fenwick') == 1:
                    val_en = compute_empty_net_fenwick(row.get('ShotDistance'), row.get('ShotAngle'))
                    if val_en is not None:
                        row['xG_F'] = round(val_en, 6)
                        row['xG_F2'] = round(val_en, 6)
                # Skip model predictions for ENA rows
                continue
            # xG_S for shots only
            if row.get('Shot') == 1:
                val = predict_avg_for_row(i_row, season_val, 'xgbs')
                if val is not None:
                    row['xG_S'] = round(val, 6)
            # xG_F and xG_F2 for Fenwick only
            if row.get('Fenwick') == 1:
                val_f = predict_avg_for_row(i_row, season_val, 'xgb')
                if val_f is not None:
                    row['xG_F'] = round(val_f, 6)
                val_f2 = predict_avg_for_row(i_row, season_val, 'xgb2')
                if val_f2 is not None:
                    row['xG_F2'] = round(val_f2, 6)
    except Exception:
        # Fail-safe: don't block PBP if models or pandas are unavailable
        pass

    return jsonify({
        'gameId': data.get('id'),
        'plays': mapped,
    })


@main_bp.route('/api/game/<int:game_id>/shifts')
def api_game_shifts(game_id: int):
    """Scrape HTML TV/TH reports for shifts and map players to playerIds via boxscore.

    Output rows: PlayerID, Name, Team, Period, Start (sec), End (sec), Duration (End-Start)
    """
    gid = str(game_id)
    if len(gid) < 10:
        return jsonify({'error': 'Invalid gameId'}), 400
    try:
        start_year = int(gid[:4])
    except Exception:
        return jsonify({'error': 'Invalid gameId'}), 400
    season_dir = f"{start_year}{start_year+1}"
    suffix = gid[4:]

    urls = {
        'away': f"https://www.nhl.com/scores/htmlreports/{season_dir}/TV{suffix}.HTM",
        'home': f"https://www.nhl.com/scores/htmlreports/{season_dir}/TH{suffix}.HTM",
    }

    def fetch_html(url: str) -> Optional[str]:
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            r = requests.get(url, timeout=25, headers=headers)
            if r.status_code == 200:
                text = r.text
                if text and len(text) > 500:
                    return text
        except Exception:
            return None
        return None

    pages = {side: fetch_html(u) for side, u in urls.items()}

    # Fetch boxscore to map to player IDs
    try:
        r = requests.get(f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore', timeout=20)
        if r.status_code != 200:
            return jsonify({'error': 'Failed to fetch boxscore'}), 502
        box = r.json()
    except Exception:
        return jsonify({'error': 'Failed to fetch boxscore'}), 502

    def unify_roster(team_stats: Dict) -> List[Dict]:
        res: List[Dict] = []
        for grp in ('forwards', 'defense', 'goalies'):
            for p in (team_stats.get(grp) or []):
                nm = p.get('name')
                if isinstance(nm, dict):
                    nm = nm.get('default')
                # Position: prefer explicit position/positionCode, else infer from group.
                raw_pos = (p.get('position') or p.get('positionCode') or '').strip().upper()
                pos = None
                if raw_pos:
                    code = raw_pos[0]
                    pos = 'F' if code in ('C', 'L', 'R') else code
                else:
                    pos = 'F' if grp == 'forwards' else ('D' if grp == 'defense' else 'G')
                res.append({
                    'playerId': p.get('playerId'),
                    'name': nm,
                    'sweaterNumber': str(p.get('sweaterNumber') or p.get('sweater') or p.get('jersey') or '').strip(),
                    'pos': pos,
                })
        return res

    pbg = box.get('playerByGameStats') or {}
    roster_home = unify_roster(pbg.get('homeTeam') or {})
    roster_away = unify_roster(pbg.get('awayTeam') or {})

    def norm_name(s: Optional[str]) -> str:
        if not s:
            return ''
        t = s.replace('\xa0', ' ').replace('\u00a0', ' ').strip()
        if ',' in t:
            parts = [x.strip() for x in t.split(',', 1)]
            if len(parts) == 2:
                t = parts[1] + ' ' + parts[0]
        t = t.replace('.', ' ').replace("'", '').replace('-', ' ')
        t = ' '.join(t.split())
        return t.lower()

    def build_indices(roster: List[Dict]):
        by_num: Dict[str, Dict] = {}
        by_name: Dict[str, Dict] = {}
        by_last: Dict[str, List[Dict]] = {}
        for p in roster:
            num = (p.get('sweaterNumber') or '').lstrip('#')
            if num:
                by_num[str(num)] = p
            nm = norm_name(p.get('name'))
            if nm:
                by_name[nm] = p
                last = nm.split(' ')[-1]
                by_last.setdefault(last, []).append(p)
        return by_num, by_name, by_last

    idx_home = build_indices(roster_home)
    idx_away = build_indices(roster_away)

    def to_seconds(ts: Optional[str]) -> Optional[int]:
        if not ts:
            return None
        ts = ts.strip()
        if '/' in ts:
            ts = ts.split('/', 1)[0].strip()
        m = re.match(r'^(\d{1,2}):(\d{2})$', ts)
        if not m:
            return None
        return int(m.group(1)) * 60 + int(m.group(2))

    def parse_period_value(p: Optional[str]) -> Optional[int]:
        """Map period cell text to integer period.
        - Numeric strings -> int
        - 'OT' -> 4 (overtime starts at 3600s)
        - 'SO' or other text -> None (ignored for shifts)
        """
        if p is None:
            return None
        s = p.strip().upper()
        if not s:
            return None
        if s == 'OT':
            return 4
        if s == 'SO':
            return None
        if s.isdigit():
            try:
                return int(s)
            except Exception:
                return None
        return None

    def proper_name(last_upper: str, first_upper: str) -> str:
        def fix(part: str) -> str:
            part = part.strip().lower()
            return ' '.join(w.capitalize() for w in part.split())
        return f"{fix(first_upper)} {fix(last_upper)}".strip()

    def parse_shifts_from_html(html: str, side: str, idx, team_abbrev: str) -> List[Dict]:
        by_num, by_name, by_last = idx
        out: List[Dict] = []
        if not html:
            return out
        soup_results: List[Dict] = []
        if BeautifulSoup is not None:
            soup = BeautifulSoup(html, 'html.parser')

            # Power Query–style: target content table and iterate rows
            def find_content_table(sp):
                try:
                    trs = sp.find_all('tr')
                    if len(trs) >= 4:
                        td = trs[3].find_all('td')
                        if td:
                            tbl = td[0].find('table')
                            if tbl:
                                return tbl
                except Exception:
                    pass
                # Fallback: heuristic scanning
                for tbl in sp.find_all('table'):
                    rows = tbl.find_all('tr')[:15]
                    for tr in rows:
                        texts = [c.get_text(' ', strip=True).lower() for c in tr.find_all(['th', 'td'])]
                        if not texts:
                            continue
                        if any('shift' in t for t in texts) and (any(t == 'per' or 'period' in t for t in texts) or any(t.startswith('per') for t in texts)):
                            return tbl
                    if tbl.find('td', attrs={'colspan': True}):
                        dense = False
                        for tr in rows:
                            tds = [td for td in tr.find_all('td') if not td.has_attr('colspan') and not td.has_attr('rowspan')]
                            if len(tds) >= 6:
                                dense = True
                                break
                        if dense:
                            return tbl
                return None

            content_tbl = find_content_table(soup)
            if content_tbl is not None:
                current_name = None
                current_jersey = None
                current_pid = None
                current_pos = None
                for tr in content_tbl.find_all('tr'):
                    tds_all = tr.find_all('td')
                    if not tds_all:
                        continue
                    if len(tds_all) == 1 and tds_all[0].has_attr('colspan'):
                        txt = tds_all[0].get_text(' ', strip=True)
                        m1 = re.match(r'^(\d{1,2})\s+([A-Z .\'-]+),\s*([A-Z .\'-]+)$', txt)
                        m2 = re.match(r'^(\d{1,2})\s+([A-Za-z .\'-]+)$', txt)
                        if m1:
                            current_jersey = m1.group(1)
                            last_u = m1.group(2)
                            first_u = m1.group(3)
                            current_name = proper_name(last_u, first_u)
                        elif m2:
                            current_jersey = m2.group(1)
                            name_plain = m2.group(2)
                            parts = name_plain.strip().split()
                            current_name = ' '.join(p.capitalize() for p in parts)
                        else:
                            current_name = None
                            current_jersey = None
                        # Resolve PID on header change
                        current_pid = None
                        current_pos = None
                        if current_jersey:
                            p = by_num.get(current_jersey)
                            if p:
                                current_pid = p.get('playerId')
                                current_pos = p.get('pos')
                        if not current_pid and current_name:
                            p = by_name.get(norm_name(current_name))
                            if p:
                                current_pid = p.get('playerId')
                                current_pos = p.get('pos')
                        if not current_pid and current_name:
                            last_tok = norm_name(current_name).split(' ')[-1]
                            cands = by_last.get(last_tok, [])
                            if cands:
                                if len(cands) == 1:
                                    current_pid = cands[0].get('playerId')
                                    current_pos = cands[0].get('pos')
                                else:
                                    for cand in cands:
                                        if str(cand.get('sweaterNumber')).lstrip('#') == str(current_jersey):
                                            current_pid = cand.get('playerId')
                                            current_pos = cand.get('pos')
                                            break
                        continue

                    # Data rows: ignore colspan/rowspan cells
                    tds = [td for td in tds_all if not td.has_attr('colspan') and not td.has_attr('rowspan')]
                    if len(tds) < 4:
                        continue
                    ctext = [td.get_text(' ', strip=True) for td in tds[:6]]
                    shift_no = ctext[0].strip()
                    per_txt = ctext[1].strip()
                    start_txt = ctext[2].strip()
                    end_txt = ctext[3].strip()
                    per_val = parse_period_value(per_txt)
                    if not (shift_no.isdigit() and per_val is not None):
                        continue
                    start_sec = to_seconds(start_txt)
                    end_sec = to_seconds(end_txt)
                    if start_sec is None or end_sec is None:
                        continue
                    out.append({
                        'PlayerID': current_pid,
                        'Name': current_name,
                        'Position': current_pos,
                        'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                        'Period': int(per_val),
                        'Start': start_sec,
                        'End': end_sec,
                        'Duration': end_sec - start_sec,
                    })
                if out:
                    return out

            # Player-first scan: find header texts and parse the next table
            pat_comma = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-z .\'-]+),\s*([A-Za-z .\'-]+)(\s*)$')
            pat_plain = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-z][A-Za-z .\'-]+)(\s*)$')
            header_nodes = []
            for node in soup.find_all(string=True):
                txt = (node or '').replace('\xa0', ' ').strip()
                if not txt:
                    continue
                if pat_comma.match(txt) or pat_plain.match(txt):
                    header_nodes.append(node)

            for node in header_nodes:
                raw = (node or '').replace('\xa0', ' ').strip()
                m1 = pat_comma.match(raw)
                m2 = pat_plain.match(raw) if not m1 else None
                if m1:
                    jersey = m1.group(2)
                    last_u = m1.group(3)
                    first_u = m1.group(4)
                    disp_name = proper_name(last_u, first_u)
                    last_for_idx = norm_name(last_u).split(' ')[-1]
                elif m2:
                    jersey = m2.group(2)
                    name_plain = m2.group(3)
                    parts = name_plain.strip().split()
                    disp_name = ' '.join(p.capitalize() for p in parts)
                    last_for_idx = norm_name(parts[-1]) if parts else ''
                else:
                    continue

                tbl = node.find_parent().find_next('table') if node else None
                if not tbl:
                    continue
                trs = tbl.find_all('tr')
                if not trs:
                    continue
                header_row_idx = None
                i_shift = i_per = i_start = i_end = -1

                def compute_indexes(cells_text: List[str]):
                    nonlocal i_shift, i_per, i_start, i_end
                    hlow = [h.lower() for h in cells_text]
                    def idx_of(parts: List[str]) -> int:
                        for i, h in enumerate(hlow):
                            if all(p in h for p in parts):
                                return i
                        return -1
                    i_shift = idx_of(['shift'])
                    i_per = idx_of(['per']) if idx_of(['per']) >= 0 else idx_of(['period'])
                    i_start = idx_of(['start'])
                    i_end = idx_of(['end'])

                for ridx, tr in enumerate(trs[:6]):
                    cells = [c.get_text(' ', strip=True) for c in tr.find_all(['th', 'td'])]
                    if not cells:
                        continue
                    compute_indexes(cells)
                    if min(i_shift, i_per, i_start, i_end) >= 0:
                        header_row_idx = ridx
                        break
                if header_row_idx is None:
                    continue

                # Resolve PlayerID
                pid = None
                pos_val = None
                p = by_num.get(jersey)
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
                if not pid:
                    p = by_name.get(norm_name(disp_name))
                    if p:
                        pid = p.get('playerId')
                        pos_val = p.get('pos')
                if not pid and last_for_idx:
                    cands = by_last.get(last_for_idx, [])
                    if cands:
                        if len(cands) == 1:
                            pid = cands[0].get('playerId')
                            pos_val = cands[0].get('pos')
                        else:
                            for cand in cands:
                                if str(cand.get('sweaterNumber')).lstrip('#') == str(jersey):
                                    pid = cand.get('playerId')
                                    pos_val = cand.get('pos')
                                    break

                for tr in trs[header_row_idx + 1:]:
                    tds = [td.get_text(' ', strip=True) for td in tr.find_all('td')]
                    if len(tds) <= max(i_shift, i_per, i_start, i_end):
                        continue
                    per_val = parse_period_value(tds[i_per].strip())
                    if not (tds[i_shift].strip().isdigit() and per_val is not None):
                        continue
                    per = int(per_val)
                    start_sec = to_seconds(tds[i_start])
                    end_sec = to_seconds(tds[i_end])
                    if start_sec is None or end_sec is None:
                        continue
                    soup_results.append({
                        'PlayerID': pid,
                        'Name': disp_name,
                        'Position': pos_val,
                        'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                        'Period': per,
                        'Start': start_sec,
                        'End': end_sec,
                        'Duration': end_sec - start_sec,
                    })
        if soup_results:
            return soup_results

        # Regex fallback (last resort)
        def strip_tags(s: str) -> str:
            s = re.sub(r'<[^>]+>', ' ', s)
            s = re.sub(r'\s+', ' ', s).strip()
            return s

        section_iter = re.finditer(r'<td[^>]*colspan=\"?\d+\"?[^>]*>\s*(.*?)\s*</td>', html, re.I | re.S)
        positions = []
        for m in section_iter:
            positions.append((m.start(), m.end(), m.group(1)))
        positions.append((len(html), len(html), ''))  # sentinel
        for i in range(len(positions) - 1):
            start, end, header_html = positions[i]
            next_start = positions[i + 1][0]
            header_text = strip_tags(header_html)
            jersey = None
            disp_name = None
            last_for_idx = None
            m1 = re.match(r'^(\d{1,2})\s+([A-Z .\'-]+),\s*([A-Z .\'-]+)$', header_text)
            m2 = re.match(r'^(\d{1,2})\s+([A-Za-z .\'-]+)$', header_text)
            if m1:
                jersey = m1.group(1)
                last_u = m1.group(2)
                first_u = m1.group(3)
                disp_name = proper_name(last_u, first_u)
                last_for_idx = norm_name(last_u).split(' ')[-1]
            elif m2:
                jersey = m2.group(1)
                name_plain = m2.group(2)
                parts = name_plain.strip().split()
                disp_name = ' '.join(p.capitalize() for p in parts)
                last_for_idx = norm_name(parts[-1]) if parts else ''
            else:
                continue

            # Resolve PlayerID
            pid = None
            pos_val = None
            if jersey:
                p = by_num.get(jersey)
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
            if not pid and disp_name:
                p = by_name.get(norm_name(disp_name))
                if p:
                    pid = p.get('playerId')
                    pos_val = p.get('pos')
            if not pid and last_for_idx:
                cands = by_last.get(last_for_idx, [])
                if cands:
                    if len(cands) == 1:
                        pid = cands[0].get('playerId')
                        pos_val = cands[0].get('pos')
                    else:
                        for cand in cands:
                            if str(cand.get('sweaterNumber')).lstrip('#') == str(jersey):
                                pid = cand.get('playerId')
                                pos_val = cand.get('pos')
                                break

            section_html = html[end:next_start]
            row_re = re.compile(r'<tr[^>]*>\s*(.*?)\s*</tr>', re.I | re.S)
            cell_re = re.compile(r'<t[dh][^>]*>\s*(.*?)\s*</t[dh]>', re.I | re.S)
            rows = row_re.findall(section_html)
            for row_html in rows:
                if re.search(r'<td[^>]*colspan=', row_html, re.I):
                    continue
                cells_html = cell_re.findall(row_html)
                cells = [strip_tags(c) for c in cells_html]
                if len(cells) < 4:
                    continue
                shift_no = cells[0].strip()
                per_txt = cells[1].strip()
                start_txt = cells[2].strip()
                end_txt = cells[3].strip()
                per_val = parse_period_value(per_txt)
                if not (shift_no.isdigit() and per_val is not None):
                    continue
                start_sec = to_seconds(start_txt)
                end_sec = to_seconds(end_txt)
                if start_sec is None or end_sec is None:
                    continue
                out.append({
                    'PlayerID': pid,
                    'Name': disp_name,
                    'Position': pos_val,
                    'Team': team_abbrev or ('Away' if side == 'away' else 'Home'),
                    'Period': int(per_val),
                    'Start': start_sec,
                    'End': end_sec,
                    'Duration': end_sec - start_sec,
                })
        return out

    away_abbrev = (box.get('awayTeam') or {}).get('abbrev') or 'AWY'
    home_abbrev = (box.get('homeTeam') or {}).get('abbrev') or 'HME'
    shifts_out: List[Dict] = []
    shifts_out += parse_shifts_from_html(pages.get('away') or '', 'away', idx_away, away_abbrev)
    shifts_out += parse_shifts_from_html(pages.get('home') or '', 'home', idx_home, home_abbrev)

    # Transform Start/End to game time (seconds since game start) and build global shift slices
    # Period offset = (Period-1) * 1200 seconds (20-minute periods)
    entries: List[Dict] = []
    boundaries: set[int] = set()
    max_end = 0
    for row in shifts_out:
        try:
            per = int(row.get('Period') or 1)
        except Exception:
            per = 1
        try:
            st = int(row.get('Start') or 0)
            et = int(row.get('End') or 0)
        except Exception:
            continue
        base = (per - 1) * 1200
        gs = base + max(0, st)
        ge = base + max(0, et)
        if ge <= gs:
            continue
        boundaries.add(gs)
        boundaries.add(ge)
        if ge > max_end:
            max_end = ge
        entries.append({
            'gs': gs,
            'ge': ge,
            'PlayerID': row.get('PlayerID'),
            'Name': row.get('Name'),
            'Position': row.get('Position'),
            'Team': row.get('Team'),
        })

    if not entries:
        return jsonify({
            'gameId': game_id,
            'seasonDir': season_dir,
            'suffix': suffix,
            'source': urls,
            'shifts': [],
        })

    # Create sorted unique start times; ensure max_end is included as the last boundary
    times = sorted(t for t in boundaries)
    if not times or times[-1] != max_end:
        times.append(max_end)

    # Build global shift slices and split players into these slices
    split_rows: List[Dict] = []
    for i in range(len(times) - 1):
        s = times[i]
        e = times[i + 1]
        if e <= s:
            continue
        shift_index = int(game_id) * 10000 + (i + 1)

        # Determine active players in [s, e)
        active: List[Dict] = [rec for rec in entries if rec['gs'] <= s < rec['ge']]
        if not active:
            continue

        # Compute skater and goalie counts per team for this slice
        team_counts: Dict[str, Dict[str, int]] = {}
        for rec in active:
            team = rec['Team'] or ''
            pos = (rec.get('Position') or '').upper()
            tc = team_counts.setdefault(team, {'G': 0, 'S': 0})
            if pos == 'G':
                tc['G'] += 1
            else:
                tc['S'] += 1

        # Emit rows with StrengthState
        for rec in active:
            team = rec['Team'] or ''
            # Determine opponent team abbrev
            if team == away_abbrev:
                opp = home_abbrev
            elif team == home_abbrev:
                opp = away_abbrev
            elif team.lower() == 'away':
                opp = 'Home'
            elif team.lower() == 'home':
                opp = 'Away'
            else:
                # Fallback: pick any other team in this slice
                opp = next((t for t in team_counts.keys() if t != team), '')

            my = team_counts.get(team, {'G': 0, 'S': 0})
            their = team_counts.get(opp, {'G': 0, 'S': 0})

            if my['G'] == 0 and (their['G'] or 0) >= 1:
                strength = 'ENF'
            elif their['G'] == 0 and (my['G'] or 0) >= 1:
                strength = 'ENA'
            else:
                strength = f"{my['S']}v{their['S']}"

            period_calc = 1 + (s // 1200)
            split_rows.append({
                'ShiftIndex': shift_index,
                'PlayerID': rec['PlayerID'],
                'Name': rec['Name'],
                'Position': rec.get('Position'),
                'Team': rec['Team'],
                'Period': int(period_calc),
                'Start': int(s),
                'End': int(e),
                'Duration': int(e - s),
                'StrengthState': strength,
            })

    return jsonify({
        'gameId': game_id,
        'seasonDir': season_dir,
        'suffix': suffix,
        'source': urls,
        'shifts': split_rows,
    })
    running_away = 0
    running_home = 0

    # Determine orientation using SUM(x) of shot attempts grouped by (period, event-owner team)
    period_team_sum_x: Dict[Tuple[int, int], float] = {}
    period_sum_all: Dict[int, float] = {}
    for pl in plays_raw:
        pd = ((pl.get('periodDescriptor') or {}).get('number'))
        try:
            pd_key = int(pd) if pd is not None else None
        except Exception:
            pd_key = None
        if not pd_key:
            continue
        tc = pl.get('typeCode')
        if tc not in (505, 506, 507, 508):  # shot attempts
            continue
        d0 = pl.get('details') or {}
        x0 = d0.get('xCoord')
        if x0 is None:
            continue
        try:
            xx = float(x0)
        except Exception:
            continue
        period_sum_all[pd_key] = period_sum_all.get(pd_key, 0.0) + xx
        owner0 = d0.get('eventOwnerTeamId')
        if isinstance(owner0, int) and owner0 in (home_id, away_id):
            key_t = (pd_key, owner0)
            period_team_sum_x[key_t] = period_team_sum_x.get(key_t, 0.0) + xx

    mapped: List[Dict] = []
    for pl in plays_raw:
        period = ((pl.get('periodDescriptor') or {}).get('number'))
        time_in_period = pl.get('timeInPeriod') or ''
        type_code = pl.get('typeCode')
        event_key = pl.get('typeDescKey')
        details = pl.get('details') or {}
        situation = pl.get('situationCode') or ''
        strength = strength_from_situation(situation, details.get('eventOwnerTeamId'))
        x = details.get('xCoord')
        y = details.get('yCoord')
        zone = details.get('zoneCode')
        reason = details.get('reason')
        secondary_reason = details.get('secondaryReason')
        type_code2 = details.get('typeCode') if isinstance(details.get('typeCode'), str) else None
        pen_dur = details.get('duration')
        event_owner = details.get('eventOwnerTeamId')
        event_team_abbrev = away_abbrev if event_owner == away_id else home_abbrev if event_owner == home_id else None
        opponent_abbrev = home_abbrev if event_team_abbrev == away_abbrev else away_abbrev if event_team_abbrev == home_abbrev else None
        goalie_id = details.get('goalieInNetId')
        goalie_name = player_name(goalie_id) if goalie_id else None

        # Collect involved player ids in priority order
        candidate_ids: List[int] = []
        for key in [
            'scoringPlayerId', 'shootingPlayerId', 'playerId', 'hittingPlayerId', 'hitteePlayerId',
            'assist1PlayerId', 'assist2PlayerId', 'blockingPlayerId', 'losingPlayerId', 'winningPlayerId',
            'committedByPlayerId', 'drawnByPlayerId'
        ]:
            pid = details.get(key)
            if pid and pid not in candidate_ids:
                candidate_ids.append(pid)
        p1_id = candidate_ids[0] if len(candidate_ids) > 0 else None
        p2_id = candidate_ids[1] if len(candidate_ids) > 1 else None
        p3_id = candidate_ids[2] if len(candidate_ids) > 2 else None
        p1_name = player_name(p1_id) if p1_id else None
        p2_name = player_name(p2_id) if p2_id else None
        p3_name = player_name(p3_id) if p3_id else None

        # Shot / goal classification
        is_goal = (type_code == 505)
        is_sog = (type_code == 506) or is_goal
        is_miss = (type_code == 507)
        is_block = (type_code == 508)
        # Blocked shots: swap zone O <-> D for display only (coords are shooter-perspective upstream)
        if is_block and zone in ('O', 'D'):
            zone = 'O' if zone == 'D' else 'D'

        # Normalize coordinates so offensive zone is to the right (positive x) for the period
        nx: Optional[float] = None
        ny: Optional[float] = None
        try:
            pd_key2 = int(period) if period is not None else None
        except Exception:
            pd_key2 = None
        sign = 1
        if pd_key2 is not None:
            if isinstance(event_owner, int) and event_owner in (home_id, away_id):
                key = (pd_key2, event_owner)
                if key in period_team_sum_x:
                    sign = 1 if period_team_sum_x[key] >= 0 else -1
                else:
                    opp = home_id if event_owner == away_id else away_id if event_owner == home_id else None
                    if isinstance(opp, int) and (pd_key2, opp) in period_team_sum_x:
                        sign = -1 if period_team_sum_x[(pd_key2, opp)] >= 0 else 1
                    else:
                        sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
            else:
                sign = 1 if period_sum_all.get(pd_key2, 0.0) >= 0 else -1
        try:
            nx = (float(x) * sign) if x is not None else None
        except Exception:
            nx = None
        try:
            ny = (float(y) * sign) if y is not None else None
        except Exception:
            ny = None

        # ScoreState: goal differential from perspective of event team BEFORE applying current event.
        if event_owner == away_id:
            score_state_val = running_away - running_home
        elif event_owner == home_id:
            score_state_val = running_home - running_away
        else:
            score_state_val = running_away - running_home

        # Possession attempts (Corsi/Fenwick)
        corsi = 1 if (is_goal or is_sog or is_miss or is_block) and event_team_abbrev else 0
        fenwick = 1 if (is_goal or is_sog or is_miss) and event_team_abbrev else 0
        shot = 1 if is_sog else 0

        # Position & shoots from primary player if available
        position = None
        shoots = None
        if p1_id and p1_id in roster:
            pos_code = roster[p1_id].get('positionCode')
            position = pos_code[0] if pos_code else None

        # gameTime calculation in seconds
        secs_elapsed = parse_time_to_seconds(time_in_period) or 0
        try:
            game_time = ((period - 1) * 20 * 60 + secs_elapsed) if period else secs_elapsed
        except Exception:
            game_time = secs_elapsed

        # Venue from event team perspective: Home/Away
        venue_ha = 'Home' if event_owner == home_id else ('Away' if event_owner == away_id else '')

        # Shot geometry (feet/degrees) relative to net at (89,0), using normalized coords
        shot_distance = None
        shot_angle = None
        if nx is not None and ny is not None and (is_goal or is_sog or is_miss or is_block):
            try:
                dx = 89.0 - float(nx)
                dy = 0.0 - float(ny)
                dist = (dx * dx + dy * dy) ** 0.5
                ang = math.degrees(math.atan2(abs(dy), dx if dx != 0 else 1e-6))
                shot_distance = round(dist, 2)
                shot_angle = round(ang, 2)
            except Exception:
                pass

        # SeasonState: map gameType to 'regular' or 'playoffs'
        gt = data.get('gameType')
        season_state = 'playoffs' if str(gt) == '3' else 'regular'

        mapped.append({
            'GameID': data.get('id'),
            'Season': data.get('season'),
            'SeasonState': season_state,
            'Venue': venue_ha,
            'Period': period,
            'gameTime': int(game_time),
            'StrengthState': strength,
            'typeCode': type_code,
            'Event': event_key,
            'x': nx,
            'y': ny,
            'Zone': zone,
            'reason': reason,
            'shotType': details.get('shotType'),
            'secondaryReason': secondary_reason,
            'typeCode2': type_code2,
            'PEN_duration': pen_dur,
            'EventTeam': event_team_abbrev,
            'Opponent': opponent_abbrev,
            'Goalie_ID': goalie_id,
            'Goalie': goalie_name,
            'Player1_ID': p1_id,
            'Player1': p1_name,
            'Player2_ID': p2_id,
            'Player2': p2_name,
            'Player3_ID': p3_id,
            'Player3': p3_name,
            'Corsi': corsi,
            'Fenwick': fenwick,
            'Shot': shot,
            'Goal': 1 if is_goal else 0,
            'EventIndex': pl.get('eventId'),
            'ShiftIndex': None,
            'ScoreState': score_state_val,
            'Home_Forwards_ID': None,
            'Home_Forwards': None,
            'Home_Defenders_ID': None,
            'Home_Defenders': None,
            'Home_Goalie_ID': None,
            'Home_Goalie': None,
            'Away_Forwards_ID': None,
            'Away_Forwards': None,
            'Away_Defenders_ID': None,
            'Away_Defenders': None,
            'Away_Goalie_ID': None,
            'Away_Goalie': None,
            'BoxID': None,
            'BoxID_rev': None,
            'BoxSize': None,
            'ShotDistance': shot_distance,
            'ShotAngle': shot_angle,
            'Position': position,
            'Shoots': shoots,
            'xG_F': None,
            'xG_S': None,
            'xG_F2': None,
        })

        # AFTER mapping current play, update running score when this is a goal
        if is_goal:
            if 'awayScore' in details and 'homeScore' in details and details.get('awayScore') is not None and details.get('homeScore') is not None:
                try:
                    ra = details.get('awayScore')
                    rh = details.get('homeScore')
                    running_away = int(ra) if ra is not None else running_away
                    running_home = int(rh) if rh is not None else running_home
                except Exception:
                    if event_owner == away_id:
                        running_away += 1
                    elif event_owner == home_id:
                        running_home += 1
            else:
                if event_owner == away_id:
                    running_away += 1
                elif event_owner == home_id:
                    running_home += 1

    return jsonify({
        'gameId': data.get('id'),
        'plays': mapped,
    })


 
