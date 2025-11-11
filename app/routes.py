from __future__ import annotations

import os
import csv
import re
import math
import bisect
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Tuple, Optional, Any

import requests
import numpy as np  # for numeric handling in model inference
import joblib       # to load pickled models
from flask import Blueprint, jsonify, render_template, request, current_app
import subprocess
import sys
import uuid
import json
import tempfile
try:
    # Python 3.9+: IANA timezones
    from zoneinfo import ZoneInfo  # type: ignore
except Exception:  # pragma: no cover
    ZoneInfo = None  # type: ignore

try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore


main_bp = Blueprint('main', __name__)
# Update page (no link in app)
@main_bp.route('/admin/update', methods=['GET'])
def update_page():
    return render_template('update.html')

# Lightweight in-memory job tracker for admin runs
_ADMIN_JOBS: Dict[str, Dict[str, Any]] = {}

def _jobs_dir() -> str:
    try:
        base = os.getenv('XG_CACHE_DIR') or tempfile.gettempdir()
        d = os.path.join(base, 'nhl_admin_jobs')
        os.makedirs(d, exist_ok=True)
        return d
    except Exception:
        return tempfile.gettempdir()

def _job_status_path(job_id: str) -> str:
    return os.path.join(_jobs_dir(), f'{job_id}.json')

def _persist_job(job_id: str, data: Dict[str, Any]) -> None:
    try:
        with open(_job_status_path(job_id), 'w', encoding='utf-8') as f:
            json.dump(data, f)
    except Exception:
        pass

def _read_job(job_id: str) -> Optional[Dict[str, Any]]:
    # Try memory first
    job = _ADMIN_JOBS.get(job_id)
    if job:
        return job
    # Fallback to disk so other workers can see it
    try:
        p = _job_status_path(job_id)
        if os.path.exists(p):
            with open(p, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        return None
    return None

def _start_admin_job(command: List[str], cwd: str) -> str:
    job_id = str(uuid.uuid4())
    _ADMIN_JOBS[job_id] = {
        'status': 'running',
        'output': '',
        'startedAt': datetime.utcnow().isoformat() + 'Z',
        'command': command,
    }
    _persist_job(job_id, _ADMIN_JOBS[job_id])
    def _runner():
        try:
            res = subprocess.run(command, cwd=cwd, capture_output=True, text=True)
            out = (res.stdout or '') + ("\n" + res.stderr if res.stderr else '')
            _ADMIN_JOBS[job_id]['output'] = out
            _ADMIN_JOBS[job_id]['status'] = 'done' if res.returncode == 0 else 'error'
        except Exception as e:
            _ADMIN_JOBS[job_id]['output'] = str(e)
            _ADMIN_JOBS[job_id]['status'] = 'error'
        finally:
            _ADMIN_JOBS[job_id]['finishedAt'] = datetime.utcnow().isoformat() + 'Z'
            _persist_job(job_id, _ADMIN_JOBS[job_id])
    t = threading.Thread(target=_runner, name=f'admin-job-{job_id}', daemon=True)
    t.start()
    return job_id

@main_bp.route('/admin/job/<job_id>', methods=['GET'])
def get_admin_job(job_id: str):
    job = _read_job(job_id)
    if not job:
        return jsonify({'error': 'job_not_found'}), 404
    return jsonify({'jobId': job_id, **job})

# Run update_data.py with date (async job)
@main_bp.route('/admin/run-update-data', methods=['POST'])
def run_update_data():
    data = request.get_json()
    date = data.get('date')
    if not date:
        return jsonify({'error': 'Missing date'}), 400
    try:
        # Resolve project root reliably in both local and Render environments
        project_root = os.path.abspath(os.path.join(current_app.root_path, '..'))
        script_path = os.path.join(project_root, 'scripts', 'update_data.py')
        cmd = [sys.executable, script_path, '--date', date, '--export']
        job_id = _start_admin_job(cmd, cwd=project_root)
        return jsonify({'jobId': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run lineups.py for all teams (async job)
@main_bp.route('/admin/run-lineups', methods=['POST'])
def run_lineups():
    try:
        project_root = os.path.abspath(os.path.join(current_app.root_path, '..'))
        script_path = os.path.join(project_root, 'scripts', 'lineups.py')
        cmd = [sys.executable, script_path, '--all']
        job_id = _start_admin_job(cmd, cwd=project_root)
        return jsonify({'jobId': job_id})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# --- Module-level caches for performance ---
_MODEL_CACHE: Dict[str, Any] = {}
_FEATURE_COLS_CACHE: Dict[str, List[str]] = {}
_PBP_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_SHIFTS_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}
_BOX_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}

# --- Prestart snapshot config/state ---
_PRESTART_THREAD_STARTED = False
_PRESTART_LOGGED: set[int] = set()  # gameIds captured this process
_PRESTART_CSV_NAME = os.getenv('PRESTART_CSV', 'prestart_snapshots.csv')

def _prestart_csv_path() -> str:
    """Return a writable path for the prestart CSV across environments.
    Priority:
      1) PRESTART_DIR env var
      2) XG_CACHE_DIR (used elsewhere for writable cache on Render)
      3) OS temp dir via _disk_cache_base()
      4) Fallback to project root (may be read-only on some platforms)
    """
    base = os.getenv('PRESTART_DIR') or os.getenv('XG_CACHE_DIR')
    if not base:
        try:
            base = _disk_cache_base()
        except Exception:
            base = None
    if base:
        try:
            os.makedirs(base, exist_ok=True)
        except Exception:
            pass
        return os.path.join(base, _PRESTART_CSV_NAME)
    # Fallbacks
    try:
        return os.path.join(_project_root(), _PRESTART_CSV_NAME)
    except Exception:
        return os.path.join(os.getcwd(), _PRESTART_CSV_NAME)

def _to_decimal_odds(american: Optional[Any]) -> Optional[float]:
    try:
        if american is None:
            return None
        a = float(american)
        if a > 0:
            return 1.0 + (a / 100.0)
        if a < 0:
            return 1.0 + (100.0 / abs(a))
        return None
    except Exception:
        return None

def _bet_fraction_kelly03(prob: Optional[float], american: Optional[Any]) -> Optional[float]:
    try:
        if prob is None:
            return None
        p = float(prob)
        if not (0.0 <= p <= 1.0):
            return None
        dec = _to_decimal_odds(american)
        if dec is None or dec <= 1.0:
            return None
        b = dec - 1.0
        q = 1.0 - p
        f = (b * p - q) / b
        f_scaled = 0.3 * f
        return f_scaled if f_scaled > 0 else 0.0
    except Exception:
        return None

def _append_prestart_row(row: Dict[str, Any]) -> None:
    path = _prestart_csv_path()
    fields = [
        'TimestampUTC','DateET','GameID','StartTimeET',
        'Away','Home',
        'WinAway','WinHome',
        'OddsAway','OddsHome',
        'BetAway','BetHome'
    ]
    try:
        file_exists = os.path.exists(path)
        # Ensure directory exists
        try:
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        except Exception:
            pass
        with open(path, 'a', encoding='utf-8', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not file_exists:
                w.writeheader()
            # coerce missing keys
            rec = {k: row.get(k) for k in fields}
            w.writerow(rec)
    except Exception:
        # best-effort; do not crash app
        pass

def _build_games_for_date(date_et) -> List[Dict[str, Any]]:
    """Internal helper to construct games with projections and odds for a given ET date (date object)."""
    date_str = str(date_et)
    url = f'https://api-web.nhle.com/v1/schedule/{date_str}'
    try:
        r = requests.get(url, timeout=20)
        js = r.json() if r.status_code == 200 else {}
    except Exception:
        js = {}

    def to_et(iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            s = iso_utc.replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            if ZoneInfo is not None:
                et = dt.astimezone(ZoneInfo('America/New_York'))
            else:
                et = dt
            return et.isoformat()
        except Exception:
            return iso_utc

    logo_by_abbrev: Dict[str, str] = {}
    try:
        for tr in TEAM_ROWS:
            ab = (tr.get('Team') or '').upper()
            logo_by_abbrev[ab] = tr.get('Logo') or ''
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    for wk in (js.get('gameWeek') or []):
        if (wk.get('date') or '')[:10] != date_str:
            continue
        for g in (wk.get('games') or []):
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id'),
                'season': g.get('season'),
                'gameType': g.get('gameType'),
                'startTimeUTC': g.get('startTimeUTC'),
                'startTimeET': to_et(g.get('startTimeUTC')),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    if not out and isinstance(js, dict):
        for g in (js.get('games') or []):
            st = g.get('startTimeUTC') or g.get('gameDate')
            if not isinstance(st, str):
                continue
            if st.replace('Z', '').strip()[:10] != date_str:
                continue
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id') or g.get('gamePk') or g.get('gameId'),
                'season': g.get('season'),
                'gameType': g.get('gameType') or g.get('gameTypeId'),
                'startTimeUTC': st,
                'startTimeET': to_et(st),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })

    # Compute B2B set for previous day (reused from API)
    prev_date_et = (date_et - timedelta(days=1)).isoformat()
    prev_set: set[str] = set()
    try:
        r2 = requests.get(f'https://api-web.nhle.com/v1/schedule/{prev_date_et}', timeout=20)
        if r2.status_code == 200:
            js2 = r2.json() or {}
            for wk in (js2.get('gameWeek') or []):
                if (wk.get('date') or '')[:10] != prev_date_et:
                    continue
                for g2 in (wk.get('games') or []):
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
            if not prev_set and isinstance(js2, dict):
                for g2 in (js2.get('games') or []):
                    st2 = g2.get('startTimeUTC') or g2.get('gameDate') or ''
                    if str(st2).replace('Z','').strip()[:10] != prev_date_et:
                        continue
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
    except Exception:
        prev_set = set()

    # Load lineups and player projections
    lineups_all = _load_lineups_all()
    proj_map = _load_player_projections_csv()
    SITUATION = {
        'Away-B2B-B2B': -0.126602018,
        'Away-B2B-Rested': -0.400515738,
        'Away-Rested-B2B': 0.174538991,
        'Away-Rested-Rested': -0.153396566,
    }
    def situation_for(away_abbrev: str, home_abbrev: str) -> float:
        a_b2b = (away_abbrev.upper() in prev_set)
        h_b2b = (home_abbrev.upper() in prev_set)
        if a_b2b and h_b2b:
            key = 'Away-B2B-B2B'
        elif a_b2b and not h_b2b:
            key = 'Away-B2B-Rested'
        elif (not a_b2b) and h_b2b:
            key = 'Away-Rested-B2B'
        else:
            key = 'Away-Rested-Rested'
        return SITUATION.get(key, 0.0)

    for g in out:
        aa = (g.get('awayTeam') or {}).get('abbrev') or ''
        ha = (g.get('homeTeam') or {}).get('abbrev') or ''
        try:
            proj_away = _team_proj_from_lineup(str(aa), lineups_all, proj_map)
            proj_home = _team_proj_from_lineup(str(ha), lineups_all, proj_map)
            dproj = proj_away - proj_home
            sval = situation_for(str(aa), str(ha))
            win_away = 1.0 / (1.0 + math.exp(-(dproj) - sval))
            win_home = 1.0 - win_away
            g['projections'] = {
                'projAway': round(float(proj_away), 6),
                'projHome': round(float(proj_home), 6),
                'dProj': round(float(dproj), 6),
                'situationValue': round(float(sval), 9),
                'winProbAway': round(float(win_away), 6),
                'winProbHome': round(float(win_home), 6),
            }
        except Exception:
            continue

    try:
        odds_map = _fetch_partner_odds_map(date_str)
    except Exception:
        odds_map = {}
    try:
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        for g in out:
            not_started = False
            try:
                st_raw = g.get('startTimeUTC')
                if isinstance(st_raw, str):
                    se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                    if se_utc.tzinfo is None:
                        se_utc = se_utc.replace(tzinfo=_tz.utc)
                    not_started = now_utc < se_utc
            except Exception:
                not_started = False
            gid = None
            try:
                if g.get('id') is not None:
                    gid = int(g.get('id'))
            except Exception:
                gid = None
            if not_started and gid is not None and gid in odds_map:
                g['odds'] = odds_map.get(gid)
    except Exception:
        pass

    return out

def _start_prestart_logger_thread_once():
    global _PRESTART_THREAD_STARTED
    if _PRESTART_THREAD_STARTED:
        return
    _PRESTART_THREAD_STARTED = True

    def _runner():
        # Respect optional window seconds (how many seconds before start qualifies)
        # Default prestart window widened to 3600s (1h) to improve chance of capture
        window_secs = 3600
        try:
            window_secs = max(30, int(os.getenv('PRESTART_WINDOW_SECONDS', str(window_secs))))
        except Exception:
            pass
        # Also capture a "grace" period after start to avoid missing games if app boots late.
        # PRESTART_GRACE_SECONDS overrides default of 300 (5 minutes)
        try:
            grace_secs = max(0, int(os.getenv('PRESTART_GRACE_SECONDS', '300')))
        except Exception:
            grace_secs = 300
        while True:
            try:
                # Determine ET date now
                try:
                    if ZoneInfo is None:
                        raise RuntimeError('zoneinfo_unavailable')
                    now_et = datetime.now(ZoneInfo('America/New_York'))
                except Exception:
                    now_et = datetime.utcnow()
                date_et = now_et.date()
                games = _build_games_for_date(date_et)
                # Current time in UTC
                from datetime import timezone as _tz
                now_utc = datetime.now(_tz.utc)
                for g in games:
                    try:
                        raw_id = g.get('id')
                        gid = int(raw_id) if raw_id is not None else None
                    except Exception:
                        gid = None
                    if gid is None or gid in _PRESTART_LOGGED:
                        continue
                    st_raw = g.get('startTimeUTC')
                    if not isinstance(st_raw, str):
                        continue
                    try:
                        se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                        if se_utc.tzinfo is None:
                            se_utc = se_utc.replace(tzinfo=_tz.utc)
                    except Exception:
                        continue
                    # Capture if within prestart window before start OR within grace window after start
                    delta_before = (se_utc - now_utc).total_seconds()
                    delta_after = (now_utc - se_utc).total_seconds()
                    if (0 <= delta_before <= window_secs) or (0 <= delta_after <= grace_secs):
                        # Prepare row
                        away_ab = (g.get('awayTeam') or {}).get('abbrev') or ''
                        home_ab = (g.get('homeTeam') or {}).get('abbrev') or ''
                        win_away = (g.get('projections') or {}).get('winProbAway')
                        win_home = (g.get('projections') or {}).get('winProbHome')
                        odds_away = (g.get('odds') or {}).get('away') if isinstance(g.get('odds'), dict) else None
                        odds_home = (g.get('odds') or {}).get('home') if isinstance(g.get('odds'), dict) else None
                        bet_away = _bet_fraction_kelly03(win_away, odds_away)
                        bet_home = _bet_fraction_kelly03(win_home, odds_home)
                        # Timestamp in UTC ISO
                        ts_utc = datetime.utcnow().isoformat() + 'Z'
                        # DateET and StartTimeET already in record
                        row = {
                            'TimestampUTC': ts_utc,
                            'DateET': str(date_et),
                            'GameID': gid,
                            'StartTimeET': g.get('startTimeET'),
                            'Away': away_ab,
                            'Home': home_ab,
                            'WinAway': round(float(win_away)*100.0, 3) if isinstance(win_away, (int, float)) else None,
                            'WinHome': round(float(win_home)*100.0, 3) if isinstance(win_home, (int, float)) else None,
                            'OddsAway': odds_away,
                            'OddsHome': odds_home,
                            'BetAway': round(float(bet_away)*100.0, 3) if isinstance(bet_away, (int, float)) else None,
                            'BetHome': round(float(bet_home)*100.0, 3) if isinstance(bet_home, (int, float)) else None,
                        }
                        _append_prestart_row(row)
                        _PRESTART_LOGGED.add(gid)
                # Sleep shorter if there are still games not captured; else back off
                remaining = [g for g in games if isinstance(g.get('id'), int) and g.get('id') not in _PRESTART_LOGGED]
                sleep_secs = 20 if remaining else 120
                time.sleep(sleep_secs)
            except Exception:
                # Never crash; sleep and retry
                try:
                    time.sleep(30)
                except Exception:
                    pass

    t = threading.Thread(target=_runner, name='prestart-logger', daemon=True)
    t.start()

def start_prestart_logger():
    """Public entry to start the background prestart logger thread.
    Safe to call multiple times; only starts once per process.
    """
    _start_prestart_logger_thread_once()

def _cache_get(cache: Dict, key, ttl: int) -> Optional[Any]:
    try:
        import time
        ts, val = cache.get(key, (0, None))
        if ts and (time.time() - ts) < ttl:
            return val
    except Exception:
        return None
    return None

def _cache_set(cache: Dict, key, val) -> None:
    try:
        import time
        cache[key] = (time.time(), val)
    except Exception:
        pass
    
# --- Small on-disk cache utilities (persist across restarts) ---
def _disk_cache_base() -> str:
    base = os.getenv('XG_CACHE_DIR')
    if base:
        return base
    try:
        if os.name == 'nt':
            import tempfile
            return os.path.join(tempfile.gettempdir(), 'nhl_cache')
        return '/tmp/nhl_cache'
    except Exception:
        return '/tmp/nhl_cache'

def _disk_cache_path_pbp(game_id: int) -> str:
    d = _disk_cache_base()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, f'pbp_{int(game_id)}.json')

def _disk_cache_path_shifts(game_id: int) -> str:
    d = _disk_cache_base()
    try:
        os.makedirs(d, exist_ok=True)
    except Exception:
        pass
    return os.path.join(d, f'shifts_{int(game_id)}.json')
@main_bp.route('/')
def index_page():
    """Frontpage Schedule view."""
    return render_template('index.html', teams=TEAM_ROWS, active_tab='Schedule', show_season_state=True)


@main_bp.route('/live')
def live_games_page():
    """Live Games page."""
    return render_template('live.html', teams=TEAM_ROWS, active_tab='Live Games', show_season_state=False)


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


@main_bp.route('/projections')
def game_projections_page():
    """Game Projections page showing today's games by Eastern Time, with toggle to yesterday."""
    return render_template('projections.html', teams=TEAM_ROWS, active_tab='Game Projections', show_season_state=False)


@main_bp.route('/api/projections/games')
def api_projections_games():
    """Return list of games for 'today', 'yesterday', or 'tomorrow' based on Eastern Time.
    Query params:
      - which: 'today' (default) | 'yesterday' | 'tomorrow'
    """
    which = str(request.args.get('which', 'today')).lower().strip()
    # Determine ET date
    try:
        if ZoneInfo is None:
            raise RuntimeError('zoneinfo_unavailable')
        now_et = datetime.now(ZoneInfo('America/New_York'))
    except Exception:
        # Fallback to UTC if ET tz not available
        now_et = datetime.utcnow()
    if which == 'yesterday':
        date_et = (now_et - timedelta(days=1)).date()
    elif which == 'tomorrow':
        date_et = (now_et + timedelta(days=1)).date()
    else:
        date_et = now_et.date()
    date_str = date_et.isoformat()
    # Fetch schedule for ET date
    url = f'https://api-web.nhle.com/v1/schedule/{date_str}'
    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return jsonify({'games': [], 'date': date_str, 'error': 'upstream_error', 'status': r.status_code}), 502
        js = r.json() or {}
    except Exception:
        return jsonify({'games': [], 'date': date_str, 'error': 'fetch_failed'}), 502

    # Build output
    def to_et(iso_utc: Optional[str]) -> Optional[str]:
        if not iso_utc:
            return None
        try:
            # Parse ISO with Z
            s = iso_utc.replace('Z', '+00:00')
            dt = datetime.fromisoformat(s)
            if ZoneInfo is not None:
                et = dt.astimezone(ZoneInfo('America/New_York'))
            else:
                et = dt  # best effort
            return et.isoformat()
        except Exception:
            return iso_utc

    logo_by_abbrev: Dict[str, str] = {}
    try:
        for tr in TEAM_ROWS:
            ab = (tr.get('Team') or '').upper()
            logo_by_abbrev[ab] = tr.get('Logo') or ''
    except Exception:
        pass

    out: List[Dict[str, Any]] = []
    # The API nests games inside gameWeek -> [ { date, games: [...] } ]
    for wk in (js.get('gameWeek') or []):
        day_date = (wk.get('date') or '')[:10]
        if day_date != date_str:
            continue
        for g in (wk.get('games') or []):
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id'),
                'season': g.get('season'),
                'gameType': g.get('gameType'),
                'startTimeUTC': g.get('startTimeUTC'),
                'startTimeET': to_et(g.get('startTimeUTC')),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    # Fallback: some variants include a flat 'games' array
    if not out and isinstance(js, dict):
        for g in (js.get('games') or []):
            st = g.get('startTimeUTC') or g.get('gameDate')
            if not isinstance(st, str):
                continue
            if st.replace('Z', '').strip()[:10] != date_str:
                continue
            home = (g.get('homeTeam') or {})
            away = (g.get('awayTeam') or {})
            ha = (home.get('abbrev') or '').upper()
            aa = (away.get('abbrev') or '').upper()
            out.append({
                'id': g.get('id') or g.get('gamePk') or g.get('gameId'),
                'season': g.get('season'),
                'gameType': g.get('gameType') or g.get('gameTypeId'),
                'startTimeUTC': st,
                'startTimeET': to_et(st),
                'gameState': g.get('gameState') or g.get('gameStatus'),
                'venue': g.get('venue'),
                'homeTeam': { 'abbrev': ha, 'score': home.get('score'), 'logo': logo_by_abbrev.get(ha, '') },
                'awayTeam': { 'abbrev': aa, 'score': away.get('score'), 'logo': logo_by_abbrev.get(aa, '') },
                'periodDescriptor': g.get('periodDescriptor'),
            })
    # Compute B2B status using the previous ET date
    prev_date_et = (date_et - timedelta(days=1)).isoformat()
    prev_url = f'https://api-web.nhle.com/v1/schedule/{prev_date_et}'
    prev_set: set[str] = set()
    try:
        r2 = requests.get(prev_url, timeout=20)
        if r2.status_code == 200:
            js2 = r2.json() or {}
            for wk in (js2.get('gameWeek') or []):
                if (wk.get('date') or '')[:10] != prev_date_et:
                    continue
                for g2 in (wk.get('games') or []):
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
            if not prev_set and isinstance(js2, dict):
                for g2 in (js2.get('games') or []):
                    st2 = g2.get('startTimeUTC') or g2.get('gameDate') or ''
                    if str(st2).replace('Z','').strip()[:10] != prev_date_et:
                        continue
                    home2 = (g2.get('homeTeam') or {})
                    away2 = (g2.get('awayTeam') or {})
                    if home2.get('abbrev'):
                        prev_set.add(str(home2.get('abbrev')).upper())
                    if away2.get('abbrev'):
                        prev_set.add(str(away2.get('abbrev')).upper())
    except Exception:
        prev_set = set()

    # Load lineups and player projections once
    lineups_all = _load_lineups_all()
    proj_map = _load_player_projections_csv()
    # Situation mapping values
    SITUATION = {
        'Away-B2B-B2B': -0.126602018,
        'Away-B2B-Rested': -0.400515738,
        'Away-Rested-B2B': 0.174538991,
        'Away-Rested-Rested': -0.153396566,
    }

    def situation_for(away_abbrev: str, home_abbrev: str) -> tuple[str, float, bool, bool]:
        a_b2b = (away_abbrev.upper() in prev_set)
        h_b2b = (home_abbrev.upper() in prev_set)
        if a_b2b and h_b2b:
            key = 'Away-B2B-B2B'
        elif a_b2b and not h_b2b:
            key = 'Away-B2B-Rested'
        elif (not a_b2b) and h_b2b:
            key = 'Away-Rested-B2B'
        else:
            key = 'Away-Rested-Rested'
        return key, SITUATION.get(key, 0.0), a_b2b, h_b2b

    # Compute projections per game
    for g in out:
        aa = (g.get('awayTeam') or {}).get('abbrev') or ''
        ha = (g.get('homeTeam') or {}).get('abbrev') or ''
        try:
            proj_away = _team_proj_from_lineup(str(aa), lineups_all, proj_map)
            proj_home = _team_proj_from_lineup(str(ha), lineups_all, proj_map)
            dproj = proj_away - proj_home
            key, sval, a_b2b, h_b2b = situation_for(str(aa), str(ha))
            import math
            win_away = 1.0 / (1.0 + math.exp(-(dproj) - sval))
            win_home = 1.0 - win_away
            g['b2bAway'] = bool(a_b2b)
            g['b2bHome'] = bool(h_b2b)
            g['projections'] = {
                'projAway': round(float(proj_away), 6),
                'projHome': round(float(proj_home), 6),
                'dProj': round(float(dproj), 6),
                'situationKey': key,
                'situationValue': round(float(sval), 9),
                'winProbAway': round(float(win_away), 6),
                'winProbHome': round(float(win_home), 6),
            }
        except Exception:
            # If anything fails, still return the game
            continue

    # Attach odds for not-started games only, and attach prestart for started games
    try:
        odds_map = _fetch_partner_odds_map(date_str)
    except Exception:
        odds_map = {}
    # Load prestart snapshots (append-only CSV) and index by GameID
    prestart_map = _load_prestart_snapshots_map()
    try:
        # Determine not-started strictly by comparing schedule startTimeUTC to current UTC
        from datetime import timezone as _tz
        now_utc = datetime.now(_tz.utc)
        for g in out:
            not_started = False
            started = False
            try:
                st_raw = g.get('startTimeUTC')
                if isinstance(st_raw, str):
                    se_utc = datetime.fromisoformat(st_raw.replace('Z', '+00:00'))
                    # If parsed datetime is naive, force UTC
                    if se_utc.tzinfo is None:
                        se_utc = se_utc.replace(tzinfo=_tz.utc)
                    not_started = now_utc < se_utc
                    started = now_utc >= se_utc
            except Exception:
                not_started = False
                started = False
            g['started'] = bool(started)
            gid = None
            try:
                val_id = g.get('id')
                if val_id is not None:
                    gid = int(val_id)
            except Exception:
                gid = None
            if not_started and gid is not None and gid in odds_map:
                g['odds'] = odds_map.get(gid)
            # When started, attach prestart snapshot if available
            if started and gid is not None and gid in prestart_map:
                g['prestart'] = prestart_map.get(gid)
    except Exception:
        pass

    return jsonify({ 'date': date_str, 'timezone': 'ET', 'games': out })


@main_bp.route('/api/roster/<team_code>/current')
def api_roster_current(team_code: str):
    """Proxy NHL roster endpoint to bypass browser CORS.
    Example upstream: https://api-web.nhle.com/v1/roster/TBL/current
    """
    team = (team_code or '').upper().strip()
    if not team:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': []})
    url = f'https://api-web.nhle.com/v1/roster/{team}/current'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
    except Exception:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'fetch_failed'}), 502
    if r.status_code != 200:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'upstream_error', 'status': r.status_code}), 502
    try:
        data = r.json()
    except Exception:
        return jsonify({'forwards': [], 'defensemen': [], 'goalies': [], 'error': 'invalid_upstream'}), 502
    # Normalize expected keys
    out = {
        'forwards': data.get('forwards') or [],
        'defensemen': data.get('defensemen') or [],
        'goalies': data.get('goalies') or [],
    }
    j = jsonify(out)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/seasons/<team_code>')
def api_seasons(team_code: str):
    """Return seasons for a given team using NHL club-stats-season endpoint.
    Shape: [{ "season": 20242025, "gameTypes": ["2", "3"] }, ...]
    """
    team = (team_code or '').upper().strip()
    if not team:
        return jsonify([])
    def _strip_parentheticals(s: Optional[str]) -> str:
        if not s:
            return ''
        try:
            return re.sub(r"\s*\([^)]*\)", '', s).strip()
        except Exception:
            return s or ''

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

    def _strip_parentheticals(s: Optional[str]) -> str:
        if not s:
            return ''
        try:
            return re.sub(r"\s*\([^)]*\)", '', s).strip()
        except Exception:
            return s

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


@main_bp.route('/api/live-games')
def api_live_games():
    """Return list of live games using NHL schedule/now endpoint.
    Filters to gameState indicating in-progress; if none, returns empty list.
    Shape: { games: [ { id, gameState, startTimeUTC, venue, awayTeam, homeTeam, periodDescriptor? } ] }
    """
    url = 'https://api-web.nhle.com/v1/schedule/now'
    try:
        r = requests.get(url, timeout=20)
    except Exception:
        return jsonify({'games': [], 'error': 'Fetch failed'}), 502
    if r.status_code != 200:
        return jsonify({'games': [], 'error': 'Upstream error', 'status': r.status_code}), 502
    try:
        js = r.json()
    except Exception:
        return jsonify({'games': []})
    live_states = {'LIVE', 'INPROGRESS', 'CRIT', 'OT', 'SHOOTOUT'}
    out = []
    for wk in (js.get('gameWeek') or []):
        for g in (wk.get('games') or []):
            st = str(g.get('gameState') or '').upper()
            if st in live_states:
                out.append({
                    'id': g.get('id'),
                    'season': g.get('season'),
                    'gameType': g.get('gameType'),
                    'startTimeUTC': g.get('startTimeUTC'),
                    'gameState': g.get('gameState'),
                    'venue': g.get('venue'),
                    'awayTeam': g.get('awayTeam'),
                    'homeTeam': g.get('homeTeam'),
                    'periodDescriptor': g.get('periodDescriptor'),
                })
    return jsonify({'games': out})


@main_bp.route('/admin/prestart-snapshots')
def admin_prestart_snapshots():
    """Admin endpoint to preview or download the prestart snapshots CSV.
    Query params:
      - mode: 'preview' (default) | 'download'
      - limit: number of rows to return for preview (default: 100, returns last N rows)
      - gameId: optional filter for a specific GameID (int) for preview
    """
    mode = str(request.args.get('mode', 'preview')).strip().lower()
    path = _prestart_csv_path()
    # Download mode
    if mode == 'download':
        try:
            from flask import send_file  # local import to avoid top-level issues
            if not os.path.exists(path):
                return jsonify({'error': 'file_not_found', 'path': path}), 404
            resp = send_file(path, as_attachment=True, download_name=os.path.basename(path))
            try:
                resp.headers['Cache-Control'] = 'no-store'
            except Exception:
                pass
            return resp
        except Exception:
            return jsonify({'error': 'download_failed'}), 500
    # Preview mode
    try:
        limit = int(request.args.get('limit', '100'))
    except Exception:
        limit = 100
    try:
        game_id_filter = request.args.get('gameId')
        game_id_val = int(game_id_filter) if game_id_filter is not None else None
    except Exception:
        game_id_val = None
    if not os.path.exists(path):
        return jsonify({'exists': False, 'path': path, 'rows': [], 'total': 0})
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for r in rdr:
                if game_id_val is not None:
                    try:
                        gid = int(str(r.get('GameID') or '').strip())
                        if gid != game_id_val:
                            continue
                    except Exception:
                        continue
                rows.append(r)
    except Exception:
        return jsonify({'exists': True, 'path': path, 'rows': [], 'total': 0, 'error': 'read_failed'}), 500
    total = len(rows)
    if limit > 0 and total > limit:
        rows = rows[-limit:]
    return jsonify({'exists': True, 'path': path, 'total': total, 'limit': limit, 'rows': rows})


@main_bp.route('/favicon.png')
def favicon_png():
    """Serve favicon.png placed at project root.
    We look in CWD and repo root for a favicon.png and stream it; fallback 404.
    """
    from flask import send_file
    paths = [
        os.path.join(os.getcwd(), 'favicon.png'),
        os.path.join(os.path.dirname(__file__), '..', 'favicon.png'),
    ]
    for p in paths:
        try:
            p2 = os.path.abspath(p)
            if os.path.exists(p2):
                return send_file(p2, mimetype='image/png')
        except Exception:
            continue
    return ('', 404)


@main_bp.route('/api/diag/models')
def api_diag_models():
    """Diagnostics for model loading in production.
    Returns Python and package versions, model directory, and available model files.
    """
    import sys
    info: Dict[str, Any] = {}
    try:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'Model')
        files = []
        try:
            files = sorted([f for f in os.listdir(model_dir) if f.endswith('.pkl')]) if os.path.isdir(model_dir) else []
        except Exception:
            files = []
        # Versions
        def _ver(mod_name: str) -> Optional[str]:
            try:
                mod = __import__(mod_name)
                return getattr(mod, '__version__', 'unknown')
            except Exception:
                return None
        info = {
            'python': sys.version,
            'versions': {
                'numpy': _ver('numpy'),
                'pandas': _ver('pandas'),
                'sklearn': _ver('sklearn'),
                'xgboost': _ver('xgboost'),
                'joblib': _ver('joblib'),
            },
            'model_dir': model_dir,
            'model_count': len(files),
            'models': files,
        }
    except Exception:
        info = {'error': 'diagnostics_failed'}
    return jsonify(info)


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


# --- Projections helpers ---
def _static_path(*parts: str) -> str:
    try:
        base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
        return os.path.join(base, *parts)
    except Exception:
        return os.path.join(os.getcwd(), *parts)

def _load_lineups_all() -> Dict[str, Any]:
    path = _static_path('lineups_all.json')
    try:
        if os.path.exists(path):
            import json
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f) or {}
    except Exception:
        return {}
    return {}

def _load_player_projections_csv() -> Dict[int, Dict[str, Any]]:
    """Load app/static/player_projections.csv into a dict keyed by playerId (int).
    Accepts flexible column casing/names for 'playerId'.
    """
    path = _static_path('player_projections.csv')
    out: Dict[int, Dict[str, Any]] = {}
    try:
        if not os.path.exists(path):
            return {}
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            # Find id column flexibly
            id_col = None
            cols = [c for c in (rdr.fieldnames or [])]
            lowers = { (c or '').lower(): (c or '') for c in cols }
            for cand in ('playerid', 'player_id', 'playerid', 'id'):
                if cand in lowers:
                    id_col = lowers[cand]
                    break
            # If not found, try exact 'playerId'
            if id_col is None and 'playerId' in cols:
                id_col = 'playerId'
            for row in rdr:
                try:
                    pid_raw = row.get(id_col) if id_col else None
                    if pid_raw is None:
                        continue
                    pid = int(str(pid_raw).strip())
                except Exception:
                    continue
                out[pid] = row
    except Exception:
        return {}
    return out

def _load_prestart_snapshots_map() -> Dict[int, Dict[str, Any]]:
    """Load prestart_snapshots.csv into a map keyed by GameID (int) keeping the latest row per game.
    Expected columns: TimestampUTC,DateET,GameID,StartTimeET,Away,Home,WinAway,WinHome,OddsAway,OddsHome,BetAway,BetHome
    """
    path = _prestart_csv_path()
    latest: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(path):
        return latest
    try:
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                try:
                    gid_raw = row.get('GameID')
                    if gid_raw is None:
                        continue
                    gid = int(str(gid_raw).strip())
                except Exception:
                    continue
                ts = row.get('TimestampUTC') or ''
                # Keep the last seen row per GameID (file is append-only)
                latest[gid] = {
                    'TimestampUTC': ts,
                    'DateET': row.get('DateET'),
                    'StartTimeET': row.get('StartTimeET'),
                    'Away': row.get('Away'),
                    'Home': row.get('Home'),
                    # Store numeric percents as floats
                    'winAwayPct': _safe_float(row.get('WinAway')),
                    'winHomePct': _safe_float(row.get('WinHome')),
                    'oddsAway': row.get('OddsAway'),
                    'oddsHome': row.get('OddsHome'),
                    'betAwayPct': _safe_float(row.get('BetAway')),
                    'betHomePct': _safe_float(row.get('BetHome')),
                }
    except Exception:
        return latest
    return latest

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None

def _proj_value_for_player(row: Optional[Dict[str, Any]]) -> float:
    """Sum of (Age + Rookie + EVO + EVD + PP + SH + GSAx) for a projections row.
    Non-numeric values are treated as 0.
    """
    if not row:
        return 0.0
    def f(k: str) -> float:
        try:
            v = row.get(k)
            if v is None:
                return 0.0
            return float(v)
        except Exception:
            # try case-insensitive
            try:
                for key in row.keys():
                    if str(key).lower() == k.lower():
                        vv = row.get(key)
                        return float(vv) if vv is not None else 0.0
            except Exception:
                pass
            return 0.0
    return (
        f('Age') + f('Rookie') + f('EVO') + f('EVD') + f('PP') + f('SH') + f('GSAx')
    )

_ROOKIE_FALLBACK = { 'D': -0.031768511, 'F': -0.024601581, 'G': -0.12 }

def _team_proj_from_lineup(team_abbrev: str, lineups_all: Dict[str, Any], proj_map: Dict[int, Dict[str, Any]]) -> float:
    t = (team_abbrev or '').upper()
    li = lineups_all.get(t) or {}
    total = 0.0
    for sec in ('forwards', 'defense', 'goalies'):
        arr = li.get(sec) or []
        if not isinstance(arr, list):
            continue
        for it in arr:
            try:
                # Exclude all EXT players from projections
                unit_val = str(it.get('unit') or '').upper()
                if unit_val == 'EXT':
                    continue
                pid = it.get('playerId')
                pos = (it.get('pos') or '').upper()[:1]
                # For goalies, include only G1 explicitly (others should be EXT already)
                if pos == 'G':
                    if unit_val and unit_val != 'G1':
                        continue
                if isinstance(pid, int) and pid in proj_map:
                    total += _proj_value_for_player(proj_map.get(pid))
                else:
                    total += _ROOKIE_FALLBACK.get(pos or 'F', _ROOKIE_FALLBACK['F'])
            except Exception:
                continue
    return float(total)

def _fetch_partner_odds_map(date_hint: Optional[str] = None) -> Dict[int, Dict[str, Any]]:
    """Fetch odds from NHL partner endpoint and map by game id to {'away': <odds>, 'home': <odds>}.
    Parsing is best-effort to tolerate upstream shape changes.
    """
    out: Dict[int, Dict[str, Any]] = {}
    # Try date-specific endpoint first if provided, then fallback to 'now'
    urls: List[str] = []
    if date_hint:
        urls.append(f'https://api-web.nhle.com/v1/partner-game/US/{date_hint}')
    urls.append('https://api-web.nhle.com/v1/partner-game/US/now')
    js = None
    for u in urls:
        try:
            r = requests.get(u, timeout=15)
            if r.status_code == 200:
                js = r.json()
                break
        except Exception:
            continue
    if js is None:
        return {}

    def extract_odds_from_node(node: Any) -> Tuple[Optional[Any], Optional[Any]]:
        """Return (away, home) odds from a node if present, else (None, None)."""
        away = None
        home = None
        if isinstance(node, dict):
            # Direct keys
            away = node.get('away') or node.get('awayPrice') or node.get('oddsAway') or node.get('priceAway') or node.get('A')
            home = node.get('home') or node.get('homePrice') or node.get('oddsHome') or node.get('priceHome') or node.get('H')
            # outcomes variants
            if away is None and home is None:
                outcomes = node.get('outcomes') or node.get('selections') or node.get('lines') or []
                if isinstance(outcomes, list):
                    for oc in outcomes:
                        if not isinstance(oc, dict):
                            continue
                        lbl = str(oc.get('label') or oc.get('name') or oc.get('type') or oc.get('outcome') or '').lower()
                        val = oc.get('americanOdds') or oc.get('oddsAmerican') or oc.get('price') or oc.get('american') or oc.get('odds')
                        # Some shapes might use side indicators
                        side = (oc.get('side') or oc.get('team') or oc.get('participant') or '').lower()
                        if 'away' in lbl or side == 'away' or side == 'visitor':
                            away = away if away is not None else val
                        elif 'home' in lbl or side == 'home':
                            home = home if home is not None else val
        return away, home

    def extract_ml_from_team_odds(team_obj: Any) -> Optional[Any]:
        """Given a team object that may contain an 'odds' list as in partner API, pick MONEY_LINE_2_WAY value.
        Fallback to MONEY_LINE_2_WAY_TNB if needed.
        """
        if not isinstance(team_obj, dict):
            return None
        lst = team_obj.get('odds')
        if not isinstance(lst, list):
            return None
        val_ml = None
        try:
            # Prefer exact MONEY_LINE_2_WAY
            for it in lst:
                if not isinstance(it, dict):
                    continue
                desc = str(it.get('description') or '').upper().strip()
                if desc == 'MONEY_LINE_2_WAY':
                    val_ml = it.get('value')
                    break
            # Fallback to MONEY_LINE_2_WAY_TNB
            if val_ml is None:
                for it in lst:
                    if not isinstance(it, dict):
                        continue
                    desc = str(it.get('description') or '').upper().strip()
                    if desc == 'MONEY_LINE_2_WAY_TNB':
                        val_ml = it.get('value')
                        break
        except Exception:
            return None
        return val_ml

    def try_add(gid_val, ml_node):
        if not ml_node:
            return
        try:
            gid = int(gid_val)
        except Exception:
            return
        away, home = extract_odds_from_node(ml_node)
        if away is None and home is None:
            return
        out[gid] = {'away': away, 'home': home}

    # Case 1: top-level list of games
    if isinstance(js, dict) and isinstance(js.get('games'), list):
        for g in (js.get('games') or []):
            gid = g.get('id') or g.get('gameId') or g.get('eventId')
            # First, support partner format where odds are inside team objects
            h = g.get('homeTeam') or {}
            a = g.get('awayTeam') or {}
            h_ml = extract_ml_from_team_odds(h)
            a_ml = extract_ml_from_team_odds(a)
            if h_ml is not None or a_ml is not None:
                try:
                    gid2 = g.get('gameId') or gid
                    try_add(gid2, {'home': h_ml, 'away': a_ml})
                except Exception:
                    pass
            # Also fall back to legacy bets/markets shapes if present
            bets = g.get('bets') or g.get('markets') or g.get('sportsbook') or g.get('sportsbookLines') or {}
            if isinstance(bets, dict):
                ml = bets.get('MONEY_LINE_2_WAY')
                if not ml:
                    for v in bets.values():
                        if isinstance(v, list):
                            for it in v:
                                if isinstance(it, dict) and str(it.get('market') or it.get('type')).upper() == 'MONEY_LINE_2_WAY':
                                    try_add(gid, it)
                        elif isinstance(v, dict) and str(v.get('market') or v.get('type')).upper() == 'MONEY_LINE_2_WAY':
                            try_add(gid, v)
                else:
                    try_add(gid, ml)
            elif isinstance(bets, list):
                for it in bets:
                    if isinstance(it, dict) and str(it.get('market') or it.get('type')).upper() == 'MONEY_LINE_2_WAY':
                        try_add(gid, it)
    # Fallback: recursive search for MONEY_LINE_2_WAY under any node with an id context
    if not out:
        def walk(node, ctx_id=None):
            if isinstance(node, dict):
                gid = node.get('id') or node.get('gameId') or node.get('eventId') or ctx_id
                for k, v in node.items():
                    if k == 'MONEY_LINE_2_WAY':
                        try_add(gid, v if isinstance(v, dict) else None)
                    else:
                        walk(v, gid)
            elif isinstance(node, list):
                for it in node:
                    walk(it, ctx_id)
        walk(js)
    return out


# --- Model utilities ---
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _model_dir() -> str:
    return os.path.join(_project_root(), 'Model')

def load_model_file(fname: str) -> Optional[Any]:
    """Module-level model loader with cache."""
    if fname in _MODEL_CACHE:
        return _MODEL_CACHE[fname]
    path = os.path.join(_model_dir(), fname)
    if not os.path.exists(path):
        return None
    try:
        m = joblib.load(path)
        _MODEL_CACHE[fname] = m
        return m
    except Exception:
        return None

def current_season_id(now: Optional[datetime] = None) -> int:
    d = now or datetime.utcnow()
    y = d.year
    if d.month >= 9:
        start_y = y
        end_y = y + 1
    else:
        start_y = y - 1
        end_y = y
    return start_y * 10000 + end_y

def preload_common_models() -> None:
    """Eager-load central window models for the current season to reduce cold-start latency."""
    try:
        s = current_season_id()
        a = int(str(s)[:4]); b = int(str(s)[4:])
        s_prev = (a-1)*10000 + (b-1)
        s_next = (a+1)*10000 + (b+1)
        middle = f"{s_prev}_{s_next}.pkl"
        for prefix in ('xgbs', 'xgb', 'xgb2'):
            load_model_file(f"{prefix}_{middle}")
    except Exception:
        pass


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
    # Allow bypassing cache for live refreshes
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
    # Serve from cache if available and not forced
    if not force:
        try:
            ttl = int(os.getenv('BOX_CACHE_TTL_SECONDS', '600'))
            cached = _cache_get(_BOX_CACHE, int(game_id), ttl)
            if cached:
                return jsonify(cached)
        except Exception:
            pass
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
    try:
        _cache_set(_BOX_CACHE, int(game_id), data)
    except Exception:
        pass
    resp_json = jsonify(data)
    # Add no-store when forced to help downstream avoid caching
    if force:
        try:
            resp_json.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp_json


@main_bp.route('/api/game/<int:game_id>/right-rail')
def api_game_right_rail(game_id: int):
    """Proxy NHL right-rail endpoint for a game to avoid browser CORS."""
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
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
    j = jsonify(data)
    if force:
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return j


@main_bp.route('/api/game/<int:game_id>/play-by-play')
def api_game_pbp(game_id: int):
    """Fetch NHL play-by-play and map to requested wide schema."""
    # Serve from disk/memory cache when available; for live games use short TTL
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
    live_ttl = 5  # seconds for live
    std_ttl = int(os.getenv('PBP_CACHE_TTL_SECONDS', '600'))
    disk_path = _disk_cache_path_pbp(int(game_id))
    if not force:
        try:
            # Try disk cache first (has metadata such as gameState)
            if os.path.exists(disk_path):
                import json, time
                with open(disk_path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                ts = float(js.get('_cachedAt', 0.0))
                gstate = str(js.get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify({k: v for k, v in js.items() if not k.startswith('_')})
            # Try in-memory cache if disk miss
            cached = _cache_get(_PBP_CACHE, int(game_id), std_ttl)
            if cached:
                return jsonify(cached)
        except Exception:
            pass
    url = f'https://api-web.nhle.com/v1/gamecenter/{game_id}/play-by-play'
    try:
        resp = requests.get(url, timeout=25)
    except Exception:
        return jsonify({'error': 'Fetch failed'}), 502
    if resp.status_code != 200:
        return jsonify({'error': 'Upstream error', 'status': resp.status_code}), 502
    data = resp.json()
    # Capture upstream game state
    game_state = str(data.get('gameState') or data.get('gameStatus') or '').upper()

    # Fetch skater bios for this game to retrieve shoots/catches for players (optional for performance)
    shoots_map: Dict[int, str] = {}
    try:
        if os.getenv('FETCH_BIOS', '0') == '1':
            gid_for_bios = data.get('id') or game_id
            bios_url = f"https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=gameId={gid_for_bios}"
            r_bios = requests.get(bios_url, timeout=15)
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
    compute_xg = (request.args.get('xg', '1') != '0') and (os.getenv('XG_DISABLED', '0') != '1')
    try:
        if not compute_xg:
            raise Exception('xg_disabled')
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        model_dir = os.path.join(project_root, 'Model')

        # Helper: map season integer like 20142015 to previous, current, next for 3 sliding windows
        def season_prev(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a-1)*10000 + (b-1)
        def season_next(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a+1)*10000 + (b+1)

        def load_model(fname: str):
            if fname in _MODEL_CACHE:
                return _MODEL_CACHE[fname]
            path = os.path.join(model_dir, fname)
            if not os.path.exists(path):
                return None
            try:
                m = joblib.load(path)
                _MODEL_CACHE[fname] = m
                return m
            except Exception:
                return None

        # Low-memory per-row one-hot encoding without building a full dummy matrix
        base_feature_cols = [
            "Venue", "shotType2", "ScoreState2", "RinkVenue",
            "StrengthState2", "BoxID2", "LastEvent"
        ]

        def _required_columns_for_model(m: Any) -> Optional[List[str]]:
            try:
                key = f"cols_id_{id(m)}"
                if key in _FEATURE_COLS_CACHE:
                    return _FEATURE_COLS_CACHE[key]
                cols = None
                if hasattr(m, 'feature_names_in_'):
                    cols = list(getattr(m, 'feature_names_in_'))
                elif hasattr(m, 'get_booster'):
                    booster = m.get_booster()
                    cols = getattr(booster, 'feature_names', None)
                if cols:
                    _FEATURE_COLS_CACHE[key] = cols
                return cols
            except Exception:
                return None

        def _vectorize_row_for_model(row_obj: Dict[str, Any], m: Any):
            cols = _required_columns_for_model(m)
            if not cols:
                return None, None  # can't align reliably
            # Build one-hot vector aligned to cols
            vec = [0.0] * len(cols)
            # Precompute string values for the row features
            vals = {}
            for c in base_feature_cols:
                v = row_obj.get(c)
                vals[c] = 'missing' if v is None else str(v)
            # For each required column, parse as prefix_value and set 1.0 if match
            for i, cname in enumerate(cols):
                if '_' not in cname:
                    # Unexpected; leave as 0.0
                    continue
                base, suffix = cname.split('_', 1)
                rv = vals.get(base)
                if rv is None:
                    continue
                if rv == suffix:
                    vec[i] = 1.0
            return vec, cols

        def predict_avg_for_row(row_obj: Dict[str, Any], season_val: Optional[int], model_prefix: str) -> Optional[float]:
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
                # Derive filenames; number of windows configurable via env XG_WINDOWS (default 1 for perf)
                num_windows = 1
                try:
                    num_windows = max(1, min(3, int(os.getenv('XG_WINDOWS', '1'))))
                except Exception:
                    num_windows = 1
                all_names = [
                    f"{model_prefix}_{s_prev2}_{s_cur}.pkl",    # window 1: s-2..s
                    f"{model_prefix}_{s_prev}_{s_next}.pkl",    # window 2: s-1..s+1
                    f"{model_prefix}_{s_cur}_{s_next2}.pkl",    # window 3: s..s+2
                ]
                # Choose middle window first as most centered
                order = [1, 0, 2]
                names = [all_names[i] for i in order[:num_windows]]
            models = [load_model(n) for n in names]
            models = [m for m in models if m is not None]
            if not models:
                return None
            preds = []
            for m in models:
                try:
                    vec, cols = _vectorize_row_for_model(row_obj, m)
                    if vec is None:
                        # Fallback: try tiny pandas DF for this single row (still low-memory)
                        try:
                            import pandas as _pd2  # local import fallback
                            _d = {c: [str(row_obj.get(c) if row_obj.get(c) is not None else 'missing')] for c in base_feature_cols}
                            df1 = _pd2.DataFrame(_d)
                            df1 = _pd2.get_dummies(df1).astype(float)
                            if hasattr(m, 'feature_names_in_'):
                                cols_needed = list(getattr(m, 'feature_names_in_'))
                                df1 = df1.reindex(columns=cols_needed, fill_value=0.0)
                            p = m.predict_proba(df1)[:, 1]
                            preds.append(float(p[0]))
                            continue
                        except Exception:
                            continue
                    else:
                        import numpy as _np2  # local import
                        x_arr = _np2.asarray([vec], dtype=float)
                        p = m.predict_proba(x_arr)[:, 1]
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

        # Compute xG with batched predictions by (family, window) to reduce Python overhead
        # 1) Handle ENA upfront
        for row in mapped:
            if row.get('StrengthState') == 'ENA':
                if row.get('Shot') == 1:
                    row['xG_S'] = 1.0
                if row.get('Fenwick') == 1:
                    val_en = compute_empty_net_fenwick(row.get('ShotDistance'), row.get('ShotAngle'))
                    if val_en is not None:
                        row['xG_F'] = round(val_en, 6)
                        row['xG_F2'] = round(val_en, 6)
        # 2) Group remaining rows by family
        families = {
            'xgbs': [i for i, r in enumerate(mapped) if r.get('Shot') == 1 and r.get('StrengthState') != 'ENA'],
            'xgb':  [i for i, r in enumerate(mapped) if r.get('Fenwick') == 1 and r.get('StrengthState') != 'ENA'],
            'xgb2': [i for i, r in enumerate(mapped) if r.get('Fenwick') == 1 and r.get('StrengthState') != 'ENA'],
        }

        def window_filenames_for_season(s_cur: int, prefix: str) -> List[str]:
            s_prev = season_prev(s_cur)
            s_next = season_next(s_cur)
            s_prev2 = season_prev(s_prev)
            s_next2 = season_next(s_next)
            num_windows = 1
            try:
                num_windows = max(1, min(3, int(os.getenv('XG_WINDOWS', '1'))))
            except Exception:
                num_windows = 1
            all_names = [
                f"{prefix}_{s_prev2}_{s_cur}.pkl",
                f"{prefix}_{s_prev}_{s_next}.pkl",
                f"{prefix}_{s_cur}_{s_next2}.pkl",
            ]
            order = [1, 0, 2]
            return [all_names[i] for i in order[:num_windows]]

        for family, idxs in families.items():
            if not idxs:
                continue
            # Group by season to resolve window filenames once per season
            by_season: Dict[int, List[int]] = {}
            for i in idxs:
                s = mapped[i].get('Season')
                if s is None:
                    continue
                by_season.setdefault(int(s), []).append(i)
            for s_cur, row_idx in by_season.items():
                if s_cur == 20252026:
                    names = [f"{family}_20222023_20242025.pkl"]
                else:
                    names = window_filenames_for_season(s_cur, family)
                models = [load_model(n) for n in names]
                models = [m for m in models if m is not None]
                if not models:
                    continue
                # Precompute required columns per model
                model_cols: List[Tuple[Any, Optional[List[str]]]] = []
                for m in models:
                    cols = _required_columns_for_model(m)
                    model_cols.append((m, cols))
                # Vectorize all rows once per model and predict
                preds_accum = [[] for _ in row_idx]
                for (m, cols) in model_cols:
                    if cols is None:
                        # fallback to per-row tiny pandas
                        try:
                            import pandas as _pd2
                            # Build rows into DF strings
                            data_rows = []
                            for i in row_idx:
                                r = mapped[i]
                                data_rows.append({c: str(r.get(c) if r.get(c) is not None else 'missing') for c in base_feature_cols})
                            df = _pd2.DataFrame(data_rows)
                            df = _pd2.get_dummies(df).astype(float)
                            if hasattr(m, 'feature_names_in_'):
                                cols_needed = list(getattr(m, 'feature_names_in_'))
                                df = df.reindex(columns=cols_needed, fill_value=0.0)
                            p = m.predict_proba(df)[:, 1]
                            for j, val in enumerate(p):
                                preds_accum[j].append(float(val))
                            continue
                        except Exception:
                            continue
                    # Fast vectorization using known columns
                    import numpy as _np2
                    mat = _np2.zeros((len(row_idx), len(cols)), dtype=float)
                    for rpos, i in enumerate(row_idx):
                        r = mapped[i]
                        # Precompute string values
                        vals = {c: ('missing' if r.get(c) is None else str(r.get(c))) for c in base_feature_cols}
                        for cix, cname in enumerate(cols):
                            if '_' not in cname:
                                continue
                            base, suffix = cname.split('_', 1)
                            if vals.get(base) == suffix:
                                mat[rpos, cix] = 1.0
                    p = m.predict_proba(mat)[:, 1]
                    for j, val in enumerate(p):
                        preds_accum[j].append(float(val))
                # Average predictions across windows and write back
                for j, i in enumerate(row_idx):
                    if not preds_accum[j]:
                        continue
                    avgp = float(sum(preds_accum[j]) / len(preds_accum[j]))
                    if family == 'xgbs':
                        mapped[i]['xG_S'] = round(avgp, 6)
                    elif family == 'xgb':
                        mapped[i]['xG_F'] = round(avgp, 6)
                    elif family == 'xgb2':
                        mapped[i]['xG_F2'] = round(avgp, 6)
    except Exception:
        # Fail-safe: don't block PBP if models or pandas are unavailable
        pass

    # Sanitize JSON output to avoid NaN/Inf and numpy types
    def _safe_val(v: Any):
        try:
            import numpy as _np  # type: ignore
            if isinstance(v, (_np.generic,)):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, float):
            try:
                if not math.isfinite(v):
                    return None
            except Exception:
                return None
        return v

    def _sanitize_row(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in r.items():
            out[k] = _safe_val(v)
        return out

    mapped_sanitized = [_sanitize_row(r) for r in mapped]

    out_obj = {
        'gameId': data.get('id'),
        'plays': mapped_sanitized,
        'gameState': game_state,
    }
    try:
        _cache_set(_PBP_CACHE, int(game_id), out_obj)
    except Exception:
        pass
    # Write to disk cache with metadata
    try:
        import json, time
        js = dict(out_obj)
        js['_cachedAt'] = time.time()
        with open(disk_path, 'w', encoding='utf-8') as f:
            json.dump(js, f)
    except Exception:
        pass
    resp = jsonify(out_obj)
    if force:
        try:
            resp.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp


@main_bp.route('/api/game/<int:game_id>/shifts')
def api_game_shifts(game_id: int):
    """Scrape HTML TV/TH reports for shifts and map players to playerIds via boxscore.

    Output rows: PlayerID, Name, Team, Period, Start (sec), End (sec), Duration (End-Start)
    """
    try:
        force = str(request.args.get('force', '')).lower() in ('1', 'true', 'yes', 'y', 'force')
    except Exception:
        force = False
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

    # Disk cache check first (needs gameState awareness from boxscore)
    live_ttl = 5
    std_ttl = int(os.getenv('SHIFTS_CACHE_TTL_SECONDS', '600'))
    disk_path = _disk_cache_path_shifts(int(game_id))

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

    # Try disk cache after knowing the gameState
    if not force:
        try:
            if os.path.exists(disk_path):
                import json, time
                with open(disk_path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                ts = float(js.get('_cachedAt', 0.0))
                gstate = str(js.get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify({k: v for k, v in js.items() if not k.startswith('_')})
        except Exception:
            pass

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
    # Canonical display name by playerId from lineups/boxscore
    name_by_id: Dict[int, str] = {}
    try:
        for p in roster_home + roster_away:
            pid = p.get('playerId')
            nm = p.get('name')
            if isinstance(pid, int) and nm:
                name_by_id[pid] = str(nm)
    except Exception:
        name_by_id = {}

    def canonical_name_for(pid: Optional[int], fallback: Optional[str]) -> Optional[str]:
        try:
            if isinstance(pid, int) and pid in name_by_id:
                return name_by_id[pid]
        except Exception:
            pass
        return fallback

    def _strip_diacritics(text: str) -> str:
        try:
            import unicodedata as _ud
            nfkd = _ud.normalize('NFKD', text)
            return ''.join([c for c in nfkd if not _ud.combining(c)])
        except Exception:
            return text

    def _normalize_jersey(s: Optional[str]) -> str:
        if not s:
            return ''
        # Keep only digits; drop leading zeros for stable compare
        digits = ''.join(ch for ch in str(s) if ch.isdigit())
        return str(int(digits)) if digits.isdigit() else ''

    def _strip_parentheticals_local(s: Optional[str]) -> str:
        if not s:
            return ''
        try:
            return re.sub(r"\s*\([^)]*\)", '', s).strip()
        except Exception:
            return s or ''

    def norm_name(s: Optional[str]) -> str:
        if not s:
            return ''
        t = s.replace('\xa0', ' ').replace('\u00a0', ' ').strip()
        t = _strip_parentheticals_local(t)
        if ',' in t:
            parts = [x.strip() for x in t.split(',', 1)]
            if len(parts) == 2:
                t = parts[1] + ' ' + parts[0]
        t = t.replace('.', ' ').replace("'", '').replace('-', ' ')
        t = ' '.join(t.split())
        t = _strip_diacritics(t)
        return t.lower()

    def last_token_norm(name: Optional[str]) -> str:
        """Return a normalized last-name token: diacritics removed, suffixes stripped.
        E.g., "McDavid Jr." -> "mcdavid"; "Smith III" -> "smith".
        """
        base = norm_name(name)
        if not base:
            return ''
        toks = base.split(' ')
        if not toks:
            return ''
        # Strip common suffixes from the tail
        suffixes = {'jr', 'sr', 'ii', 'iii', 'iv', 'v'}
        while toks and toks[-1].strip('.').lower() in suffixes:
            toks.pop()
        return toks[-1] if toks else ''

    def build_indices(roster: List[Dict]):
        by_num: Dict[str, Dict] = {}
        by_name: Dict[str, Dict] = {}
        by_last: Dict[str, List[Dict]] = {}
        for p in roster:
            num = _normalize_jersey(p.get('sweaterNumber') or '')
            if num:
                by_num[num] = p
            nm = norm_name(p.get('name'))
            if nm:
                by_name[nm] = p
                last = last_token_norm(nm)
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
        # Use Unicode-aware title-casing while preserving hyphens and apostrophes
        def fix(part: str) -> str:
            part = (part or '').strip().replace('\xa0', ' ').replace('\u00a0', ' ')
            # Split on spaces, then title-case each token; keep hyphens/apostrophes intact
            tokens = []
            for tok in part.split():
                subtoks = re.split(r'([-\'])', tok)
                subtoks = [st.title() if st.isalpha() else st for st in subtoks]
                tokens.append(''.join(subtoks))
            return ' '.join(tokens)
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
                        # Allow parenthetical nicknames (e.g., JOHN (JACK)) by stripping them first
                        txt2 = _strip_parentheticals_local(txt)
                        # Support accented Latin letters (e.g., É, è) in names
                        m1 = re.match(r'^(\d{1,2})\s+([A-ZÀ-ÖØ-Þ .\'-]+),\s*([A-ZÀ-ÖØ-Þ .\'-]+)$', txt2)
                        m2 = re.match(r'^(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)$', txt2)
                        if m1:
                            current_jersey = m1.group(1)
                            last_u = m1.group(2)
                            first_u = _strip_parentheticals_local(m1.group(3))
                            current_name = proper_name(last_u, first_u)
                        elif m2:
                            current_jersey = m2.group(1)
                            name_plain = _strip_parentheticals_local(m2.group(2))
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
                            last_tok = last_token_norm(current_name)
                            cands = by_last.get(last_tok, [])
                            if cands:
                                if len(cands) == 1:
                                    current_pid = cands[0].get('playerId')
                                    current_pos = cands[0].get('pos')
                                else:
                                    for cand in cands:
                                        if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(current_jersey):
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
                    # Prefer canonical name from lineups if we have a playerId
                    name_out = canonical_name_for(current_pid, current_name)
                    out.append({
                        'PlayerID': current_pid,
                        'Name': name_out,
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
            # Include Latin-1 accented ranges in character classes
            pat_comma = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+),\s*([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)(\s*)$')
            pat_plain = re.compile(r'^(\s*)(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)(\s*)$')
            header_nodes = []
            for node in soup.find_all(string=True):
                txt = (node or '').replace('\xa0', ' ').strip()
                if not txt:
                    continue
                txt2 = _strip_parentheticals_local(txt)
                if pat_comma.match(txt2) or pat_plain.match(txt2):
                    header_nodes.append(node)

            for node in header_nodes:
                raw = (node or '').replace('\xa0', ' ').strip()
                raw2 = _strip_parentheticals_local(raw)
                m1 = pat_comma.match(raw2)
                m2 = pat_plain.match(raw2) if not m1 else None
                if m1:
                    jersey = m1.group(2)
                    last_u = m1.group(3)
                    first_u = _strip_parentheticals_local(m1.group(4))
                    disp_name = proper_name(last_u, first_u)
                    last_for_idx = last_token_norm(last_u)
                elif m2:
                    jersey = m2.group(2)
                    name_plain = _strip_parentheticals_local(m2.group(3))
                    parts = name_plain.strip().split()
                    disp_name = ' '.join(p.capitalize() for p in parts)
                    last_for_idx = last_token_norm(parts[-1]) if parts else ''
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
                                if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(jersey):
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
                    name_out2 = canonical_name_for(pid, disp_name)
                    soup_results.append({
                        'PlayerID': pid,
                        'Name': name_out2,
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
            header_text2 = _strip_parentheticals_local(header_text)
            jersey = None
            disp_name = None
            last_for_idx = None
            m1 = re.match(r'^(\d{1,2})\s+([A-ZÀ-ÖØ-Þ .\'-]+),\s*([A-ZÀ-ÖØ-Þ .\'-]+)$', header_text2)
            m2 = re.match(r'^(\d{1,2})\s+([A-Za-zÀ-ÖØ-öø-ÿ .\'-]+)$', header_text2)
            if m1:
                jersey = m1.group(1)
                last_u = m1.group(2)
                first_u = _strip_parentheticals_local(m1.group(3))
                disp_name = proper_name(last_u, first_u)
                last_for_idx = last_token_norm(last_u)
            elif m2:
                jersey = m2.group(1)
                name_plain = _strip_parentheticals_local(m2.group(2))
                parts = name_plain.strip().split()
                disp_name = ' '.join(p.capitalize() for p in parts)
                last_for_idx = last_token_norm(parts[-1]) if parts else ''
            else:
                continue

            # Resolve PlayerID
            pid = None
            pos_val = None
            if jersey:
                p = by_num.get(_normalize_jersey(jersey))
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
                            if _normalize_jersey(cand.get('sweaterNumber')) == _normalize_jersey(jersey):
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
                name_out3 = canonical_name_for(pid, disp_name)
                out.append({
                    'PlayerID': pid,
                    'Name': name_out3,
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

    out = {
        'gameId': game_id,
        'seasonDir': season_dir,
        'suffix': suffix,
        'source': urls,
        'shifts': split_rows,
    }
    try:
        ttl = int(os.getenv('SHIFTS_CACHE_TTL_SECONDS', '600'))
        _cache_set(_SHIFTS_CACHE, int(game_id), out)
    except Exception:
        pass
    # Include gameState for disk cache TTL check and persist to disk
    try:
        game_state = str((box.get('gameState') or box.get('gameStatus') or '')).upper()
        out['gameState'] = game_state
    except Exception:
        pass
    try:
        import json, time
        js = dict(out)
        js['_cachedAt'] = time.time()
        with open(disk_path, 'w', encoding='utf-8') as f:
            json.dump(js, f)
    except Exception:
        pass
    resp = jsonify(out)
    if force:
        try:
            resp.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
    return resp
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


 
