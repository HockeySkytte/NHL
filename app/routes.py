from __future__ import annotations

import os
import csv
import re
import math
import bisect
import hashlib
import pickle
import gzip
from datetime import datetime, timedelta
import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Iterable, Iterator

import requests
import joblib       # to load pickled models
from flask import Blueprint, jsonify, render_template, request, current_app, make_response
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

# Optional DB connectivity check for admin use
@main_bp.route('/admin/db-check', methods=['GET'])
def admin_db_check():
    try:
        # Lazy import to avoid app start failures if missing
        try:
            from sqlalchemy import create_engine, text  # type: ignore
        except Exception as e:
            return jsonify({'ok': False, 'error': f'sqlalchemy_import_failed: {e}'}), 500
        # Allow ?mode=ro to test read-only connection
        mode = str(request.args.get('mode','rw')).lower().strip()
        db_url = None
        if mode == 'ro':
            db_url = os.getenv('DATABASE_URL_RO') or os.getenv('DB_URL_RO') or os.getenv('DATABASE_URL')
        else:
            db_url = os.getenv('DATABASE_URL_RW') or os.getenv('DB_URL_RW') or os.getenv('DATABASE_URL')
        if not db_url:
            user = os.getenv('DB_USER', 'root')
            pwd = os.getenv('DB_PASSWORD', '')
            host = os.getenv('DB_HOST', 'localhost')
            port = os.getenv('DB_PORT', '3306')
            name = os.getenv('DB_NAME', 'public')
            db_url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
        else:
            # If DATABASE_URL_* uses localhost but a host override is provided, swap it
            host_override = None
            if mode == 'ro':
                host_override = os.getenv('DB_HOST_RO') or os.getenv('DB_HOST')
            else:
                host_override = os.getenv('DB_HOST_RW') or os.getenv('DB_HOST')
            if host_override and '@localhost' in db_url:
                db_url = db_url.replace('@localhost', f'@{host_override}')
        # Optional SSL
        connect_args: Dict[str, Any] = {}
        if os.getenv('DB_SSL_CA'):
            connect_args['ssl_ca'] = str(os.getenv('DB_SSL_CA') or '')
        if os.getenv('DB_SSL_CERT'):
            connect_args['ssl_cert'] = str(os.getenv('DB_SSL_CERT') or '')
        if os.getenv('DB_SSL_KEY'):
            connect_args['ssl_key'] = str(os.getenv('DB_SSL_KEY') or '')
        if connect_args:
            eng = create_engine(db_url, connect_args=connect_args)
        else:
            eng = create_engine(db_url)
        with eng.connect() as conn:
            conn.execute(text('SELECT 1'))
        return jsonify({'ok': True, 'url': db_url.split('@')[-1] if '@' in db_url else db_url})
    except Exception as e:
        return jsonify({'ok': False, 'error': str(e)}), 500

# Lightweight in-memory job tracker for admin runs
_ADMIN_JOBS: Dict[str, Dict[str, Any]] = {}

# NHL Edge API cache (per-URL)
_EDGE_API_CACHE: Dict[str, Tuple[float, Any]] = {}

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
        export_flag = data.get('export', True)
        replace_flag = data.get('replace_date', False)
        season = str(data.get('season') or os.getenv('NHL_SEASON') or '20252026').strip()

        # Projections -> Google Sheets (local-only friendly)
        projections_to_sheets = bool(data.get('projections_to_sheets', True))
        projections_sheet_id = str(
            data.get('projections_sheet_id')
            or os.getenv('PROJECTIONS_SHEET_ID')
            or ''
        ).strip()
        projections_worksheet = str(
            data.get('projections_worksheet')
            or os.getenv('PROJECTIONS_WORKSHEET')
            or 'Sheets3'
        ).strip()

        # Optional: also run RAPM + context refresh (MySQL + Google Sheets)
        run_rapm = bool(data.get('run_rapm', False))
        rapm_sheet_id = str(
            data.get('rapm_sheet_id')
            or os.getenv('RAPM_SHEET_ID')
            or os.getenv('PROJECTIONS_SHEET_ID')
            or ''
        ).strip()
        rapm_worksheet = str(
            data.get('rapm_worksheet')
            or os.getenv('RAPM_WORKSHEET')
            or 'Sheets4'
        ).strip()
        context_worksheet = str(
            data.get('context_worksheet')
            or os.getenv('CONTEXT_WORKSHEET')
            or 'Sheets5'
        ).strip()

        if projections_to_sheets and not export_flag:
            return jsonify({'error': 'projections_to_sheets requires export=true'}), 400

        if run_rapm and not export_flag:
            return jsonify({'error': 'run_rapm requires export=true'}), 400

        # If export was requested, fail fast with a clear message when DB isn't reachable.
        if export_flag:
            try:
                try:
                    from sqlalchemy import create_engine, text  # type: ignore
                except Exception as e:
                    return jsonify({'error': f'sqlalchemy_import_failed: {e}'}), 500
                db_url = os.getenv('DATABASE_URL_RW') or os.getenv('DB_URL_RW') or os.getenv('DATABASE_URL')
                if not db_url:
                    user = os.getenv('DB_USER', 'root')
                    pwd = os.getenv('DB_PASSWORD', '')
                    host = os.getenv('DB_HOST', 'localhost')
                    port = os.getenv('DB_PORT', '3306')
                    name = os.getenv('DB_NAME', 'public')
                    db_url = f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"
                else:
                    host_override = os.getenv('DB_HOST_RW') or os.getenv('DB_HOST')
                    if host_override and '@localhost' in db_url:
                        db_url = db_url.replace('@localhost', f'@{host_override}')

                connect_args: Dict[str, Any] = {'connection_timeout': 3}
                if os.getenv('DB_SSL_CA'):
                    connect_args['ssl_ca'] = str(os.getenv('DB_SSL_CA') or '')
                if os.getenv('DB_SSL_CERT'):
                    connect_args['ssl_cert'] = str(os.getenv('DB_SSL_CERT') or '')
                if os.getenv('DB_SSL_KEY'):
                    connect_args['ssl_key'] = str(os.getenv('DB_SSL_KEY') or '')

                eng = create_engine(db_url, connect_args=connect_args)
                with eng.connect() as conn:
                    conn.execute(text('SELECT 1'))
            except Exception as e:
                # Common on Render when DB_HOST points to a private LAN address.
                return jsonify({
                    'error': (
                        'MySQL is not reachable from this server. '
                        'If you are running on Render, it cannot connect to a LAN/private IP MySQL host. '
                        'Either configure a publicly reachable DB (DATABASE_URL), or uncheck Export to MySQL / projections / RAPM.'
                    ),
                    'details': str(e),
                }), 502

        cmd = [sys.executable, script_path, '--date', date]
        if export_flag:
            cmd.append('--export')
        if replace_flag:
            cmd.append('--replace-date')

        # Ensure we write projections to Google Sheets instead of app/static/player_projections.csv
        if export_flag and projections_to_sheets:
            if not projections_sheet_id:
                return jsonify({'error': 'Missing projections sheet id (set PROJECTIONS_SHEET_ID env var)'}), 400
            cmd.extend(['--projections-sheets-id', projections_sheet_id])
            if projections_worksheet:
                cmd.extend(['--projections-worksheet', projections_worksheet])

        # Explicit season for table names
        if season:
            cmd.extend(['--season', season])

        if run_rapm:
            if not rapm_sheet_id:
                return jsonify({'error': 'Missing RAPM sheet id (set RAPM_SHEET_ID or PROJECTIONS_SHEET_ID env var)'}), 400
            cmd.append('--run-rapm')
            cmd.extend(['--rapm-sheets-id', rapm_sheet_id])
            if rapm_worksheet:
                cmd.extend(['--rapm-worksheet', rapm_worksheet])
            if context_worksheet:
                cmd.extend(['--context-worksheet', context_worksheet])

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

# Player landing cache: {playerId: (timestamp, json)}
_PLAYER_LANDING_CACHE: Dict[int, Tuple[float, Dict[str, Any]]] = {}

# Skaters player-list cache: {(seasonId, teamAbbrev, seasonState): (timestamp, players)}
_SKATERS_PLAYERS_CACHE: Dict[Tuple[int, str, str], Tuple[float, List[Dict[str, Any]]]] = {}

# Goalies player-list cache: {(seasonId, teamAbbrev, seasonState): (timestamp, players)}
_GOALIES_PLAYERS_CACHE: Dict[Tuple[int, str, str], Tuple[float, List[Dict[str, Any]]]] = {}

# Goalie team map (for trend charts): {(playerId, seasonState): (timestamp, {seasonId: teamAbbrev})}
_GOALIES_TEAM_BY_SEASON_MAP_CACHE: Dict[Tuple[int, str], Tuple[float, Dict[int, str]]] = {}

# Static CSV caches
_RAPM_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_PLAYER_PROJECTIONS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None
_CONTEXT_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_SEASONSTATS_STATIC_CACHE: Optional[Tuple[float, List[Dict[str, Any]]]] = None
_CARD_METRICS_DEF_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_RAPM_PLAYER_STATIC_CACHE: Dict[Tuple[int, Optional[int]], Tuple[float, List[Dict[str, Any]]]] = {}
_CONTEXT_PLAYER_STATIC_CACHE: Dict[Tuple[int, Optional[int]], Tuple[float, List[Dict[str, Any]]]] = {}
_SEASONSTATS_AGG_CACHE: Dict[Tuple[Any, ...], Tuple[float, Dict[int, Dict[str, Any]], Dict[int, str]]] = {}

# Goalies career aggregation helper cache: {(key...): (timestamp, by_pid_season, league_sa_ga)}
_GOALIES_CAREER_MATRIX_CACHE: Dict[
    Tuple[Any, ...],
    Tuple[float, Any, Any],
] = {}
_RAPM_SCALE_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any], Dict[str, Any]]] = {}
_RAPM_CAREER_CACHE: Dict[Tuple[str, str, str], Tuple[float, Dict[str, Any]]] = {}
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
                raw_id = g.get('id')
                if raw_id is not None:
                    gid = int(str(raw_id).strip())
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


def _cache_prune_ttl_and_size(cache: Dict, *, ttl_s: Optional[float] = None, max_items: Optional[int] = None) -> None:
    """Best-effort pruning for caches storing (timestamp, ...) tuples."""
    try:
        now = time.time()

        if ttl_s is not None:
            try:
                ttl_f = float(ttl_s)
            except Exception:
                ttl_f = None
            if ttl_f and ttl_f > 0:
                expired: List[Any] = []
                for k, v in list(cache.items()):
                    try:
                        ts = float(v[0] or 0.0)
                        if ts <= 0 or (now - ts) >= ttl_f:
                            expired.append(k)
                    except Exception:
                        continue
                for k in expired:
                    try:
                        cache.pop(k, None)
                    except Exception:
                        pass

        if max_items is not None:
            try:
                m = int(max_items)
            except Exception:
                m = 0
            if m > 0 and len(cache) > m:
                try:
                    items = sorted(cache.items(), key=lambda kv: float((kv[1] or (0,))[0] or 0.0))
                    to_drop = max(0, len(items) - m)
                    for i in range(to_drop):
                        try:
                            cache.pop(items[i][0], None)
                        except Exception:
                            pass
                except Exception:
                    while len(cache) > m:
                        try:
                            cache.pop(next(iter(cache)), None)
                        except Exception:
                            break
    except Exception:
        return


def _cache_set_multi_bounded(cache: Dict, key, *vals, ttl_s: Optional[float] = None, max_items: Optional[int] = None) -> None:
    """Set cache entry as (timestamp, *vals) and prune by TTL and size."""
    try:
        cache[key] = (time.time(), *vals)
    except Exception:
        return
    _cache_prune_ttl_and_size(cache, ttl_s=ttl_s, max_items=max_items)


def _dict_set_bounded(cache: Dict, key, val, *, max_items: Optional[int] = None) -> None:
    """Best-effort size bounding for plain dict caches (no timestamp tuples).

    Uses insertion order as a cheap LRU approximation.
    """
    try:
        if key in cache:
            try:
                cache.pop(key, None)
            except Exception:
                pass
        cache[key] = val
        if max_items is None:
            return
        try:
            m = int(max_items)
        except Exception:
            m = 0
        if m > 0:
            while len(cache) > m:
                try:
                    cache.pop(next(iter(cache)), None)
                except Exception:
                    break
    except Exception:
        return
    
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


@main_bp.route('/skaters')
def skaters_page():
    """Skaters page (player card + bio/metadata)."""
    return render_template(
        'skaters.html',
        teams=TEAM_ROWS,
        active_tab='Skaters',
        show_season_state=False,
        show_include_historic=False,
    )


@main_bp.route('/goalies')
def goalies_page():
    """Goalies page (player card + bio/metadata)."""
    return render_template(
        'goalies.html',
        teams=TEAM_ROWS,
        active_tab='Goalies',
        show_season_state=False,
        show_include_historic=False,
    )


@main_bp.route('/teams')
def teams_page():
    """Teams page (team card + team table/charts)."""
    return render_template(
        'teams.html',
        teams=TEAM_ROWS,
        active_tab='Teams',
        show_season_state=False,
        show_include_historic=True,
    )


@main_bp.route('/odds/<int:game_id>')
def odds_page(game_id: int):
    """Odds page showing ML history for a game (from Sheet1)."""
    # Keep the primary nav highlighted on Game Projections.
    return render_template('odds.html', teams=TEAM_ROWS, game_id=game_id, active_tab='Game Projections', show_season_state=False)


_SHEET_ROWS_CACHE: Dict[Tuple[str, str], Tuple[float, List[Dict[str, Any]]]] = {}

# Projections enrichment cache: {playerId -> {name, team, pos}}
_ALL_ROSTERS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, str]]]] = None

# NHL skater bios cache by seasonId: {seasonId -> (timestamp, {playerId -> info})}
_SKATER_BIOS_CACHE: Dict[int, Tuple[float, Dict[int, Dict[str, str]]]] = {}


def _load_sheet_rows_cached(sheet_id: str, worksheet: str, ttl_env: str = 'SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl: int = 30) -> List[Dict[str, Any]]:
    """Fetch worksheet rows via Google Sheets API with a short in-memory TTL cache."""
    global _SHEET_ROWS_CACHE
    try:
        ttl_s = max(1, int(os.getenv(ttl_env, str(default_ttl)) or str(default_ttl)))
    except Exception:
        ttl_s = default_ttl
    try:
        max_items = max(1, int(os.getenv('SHEET_ROWS_CACHE_MAX_ITEMS', '12') or '12'))
    except Exception:
        max_items = 12
    now = time.time()
    key = (sheet_id or '', worksheet or '')
    _cache_prune_ttl_and_size(_SHEET_ROWS_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SHEET_ROWS_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        _cache_set_multi_bounded(_SHEET_ROWS_CACHE, key, [], ttl_s=ttl_s, max_items=max_items)
        return []

    info = _load_google_service_account_info_from_env()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    # Avoid gspread numericisation so we can safely parse locale-specific numbers
    # (e.g., decimal commas in Win_Prop).
    try:
        rows = ws.get_all_records(numericise_ignore=['all']) or []
    except Exception:
        rows = ws.get_all_records() or []
    _cache_set_multi_bounded(_SHEET_ROWS_CACHE, key, rows, ttl_s=ttl_s, max_items=max_items)
    return rows


@main_bp.route('/api/odds/history/<int:game_id>')
def api_odds_history(game_id: int):
    """Return ML history time series for both teams for a given game id.

    Data source: Google Sheets worksheet (default Sheet1) with columns like:
      Timestamp, gameId, Team (abbrev), ML
    """
    sheet_id = (os.getenv('STARTED_OVERRIDES_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('STARTED_OVERRIDES_WORKSHEET') or 'Sheet1').strip()
    if not sheet_id:
        j = jsonify({'error': 'missing_sheet_id', 'hint': 'Set PROJECTIONS_SHEET_ID (or STARTED_OVERRIDES_SHEET_ID)'})
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j, 500

    # Team colors from Teams.csv
    color_by_team: Dict[str, str] = {}
    try:
        for row in TEAM_ROWS:
            t = (row.get('Team') or '').strip().upper()
            c = (row.get('Color') or '').strip()
            if t and c:
                color_by_team[t] = c
    except Exception:
        color_by_team = {}

    try:
        rows = _load_sheet_rows_cached(sheet_id, worksheet)
    except Exception as e:
        msg = str(e or '')
        code = 'odds_history_load_failed'
        if 'Missing Google credentials' in msg:
            code = 'missing_google_credentials'
        elif 'Invalid GOOGLE_SERVICE_ACCOUNT' in msg or 'Invalid Google service account JSON' in msg:
            code = 'invalid_google_credentials'
        elif 'SpreadsheetNotFound' in msg:
            code = 'sheet_not_found_or_no_access'
        elif 'WorksheetNotFound' in msg:
            code = 'worksheet_not_found'
        j = jsonify({'error': code, 'hint': 'Check GOOGLE_SERVICE_ACCOUNT_JSON_* and sheet sharing', 'sheet_id': sheet_id, 'worksheet': worksheet})
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j, 500

    def norm_row(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (r or {}).items():
            try:
                kk = str(k).strip().lower()
            except Exception:
                continue
            out[kk] = v
        return out

    def pick(rn: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in rn:
                return rn.get(k)
        return None

    def parse_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            s = str(v).strip()
            if not s:
                return None
            return int(s)
        except Exception:
            return None

    def parse_float(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            s = str(v).strip()
            if not s:
                return None
            # Sheets often contain percent strings like "52.4%".
            s = s.replace('%', '').strip()
            # Handle decimal comma locales: "52,3" => 52.3
            if ',' in s and '.' not in s:
                s = s.replace(',', '.')
            else:
                s = s.replace(',', '')
            return float(s)
        except Exception:
            return None

    want_debug = str(request.args.get('debug', '')).strip() in ('1', 'true', 'yes', 'y')

    series: Dict[str, Dict[str, float]] = {}  # team -> {timestamp -> ml}
    win_series: Dict[str, Dict[str, float]] = {}  # team -> {timestamp -> winProb (0..1)}
    order: Dict[str, List[str]] = {}          # team -> ordered timestamps (for stable ordering)

    debug_sample_raw: Optional[Dict[str, Any]] = None
    debug_sample_norm: Optional[Dict[str, Any]] = None
    for r in rows:
        rn = norm_row(r)
        # Compact key map to handle header variations like "Win Prop" vs "Win_Prop".
        try:
            import re as _re
            rn_compact = {_re.sub(r'[^a-z0-9]+', '', str(k)): v for k, v in rn.items()}
        except Exception:
            rn_compact = {}
        gid = parse_int(pick(rn, ['gameid', 'game_id', 'gamepk', 'game', 'id']))
        if gid != int(game_id):
            continue
        ts = pick(rn, ['timestamp', 'time', 'datetime'])
        ts_s = (str(ts).strip() if ts is not None else '')
        if not ts_s:
            continue
        team = pick(rn, ['team (abbrev)', 'team', 'abbrev', 'team_abbrev'])
        team_s = (str(team).strip().upper() if team is not None else '')
        if not team_s:
            continue

        if want_debug and debug_sample_raw is None:
            try:
                debug_sample_raw = dict(r)
                debug_sample_norm = dict(rn)
            except Exception:
                debug_sample_raw = None
                debug_sample_norm = None
        ml = parse_float(pick(rn, ['ml', 'moneyline', 'odds', 'line']))
        if ml is None:
            continue
        series.setdefault(team_s, {})[ts_s] = ml
        # Optional Win_Prop from sheet. Normalize to 0..1 (accept 0..1 or 0..100).
        wp_raw = pick(rn, ['win_prop', 'win prop', 'winprop', 'win probability', 'win_prob', 'winprobability'])
        if wp_raw is None:
            wp_raw = rn_compact.get('winprop')
        wp = parse_float(wp_raw)
        if wp is not None:
            try:
                if wp > 1.5:
                    wp = wp / 100.0
                if 0.0 <= wp <= 1.0:
                    win_series.setdefault(team_s, {})[ts_s] = wp
            except Exception:
                pass
        order.setdefault(team_s, []).append(ts_s)

    teams_out: List[Dict[str, Any]] = []
    for team_s, ts_map in series.items():
        # Keep stable order but sort timestamps lexically (ISO timestamps sort correctly).
        uniq_ts = list(dict.fromkeys(order.get(team_s, [])))
        try:
            uniq_ts.sort()
        except Exception:
            pass
        wp_map = win_series.get(team_s) or {}
        points = [{'t': ts, 'ml': ts_map[ts], 'winProp': wp_map.get(ts)} for ts in uniq_ts if ts in ts_map]
        teams_out.append({
            'abbrev': team_s,
            'color': color_by_team.get(team_s) or None,
            'points': points,
        })

    payload: Dict[str, Any] = {'gameId': int(game_id), 'teams': teams_out}
    if want_debug:
        payload['debug'] = {
            'sample_row_keys': sorted(list((debug_sample_raw or {}).keys())),
            'sample_row': debug_sample_raw,
            'sample_row_norm_keys': sorted(list((debug_sample_norm or {}).keys())),
            'sample_row_norm': debug_sample_norm,
        }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/lineups/all')
def api_lineups_all():
    """Return lineup data used by the projections lineup selector.

    Source is Google Sheets (see _load_lineups_all()).
    """
    # If sheet id is not configured, fail loudly so the browser/dev logs show a clear signal.
    sheet_id = (os.getenv('LINEUPS_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('LINEUPS_WORKSHEET') or 'Sheets2').strip()
    if not sheet_id:
        j = jsonify({
            'error': 'missing_sheet_id',
            'hint': 'Set LINEUPS_SHEET_ID or PROJECTIONS_SHEET_ID',
        })
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j, 500

    try:
        data = _load_lineups_all()
        j = jsonify(data)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j
    except Exception as e:
        # Log full error server-side, but return only a safe, actionable code to clients.
        try:
            print('[api_lineups_all] load failed:', repr(e))
        except Exception:
            pass
        msg = str(e or '')
        code = 'lineups_load_failed'
        if 'Missing Google credentials' in msg:
            code = 'missing_google_credentials'
        elif 'Invalid GOOGLE_SERVICE_ACCOUNT' in msg or 'Invalid Google service account JSON' in msg:
            code = 'invalid_google_credentials'
        elif 'SpreadsheetNotFound' in msg:
            code = 'sheet_not_found_or_no_access'
        elif 'WorksheetNotFound' in msg:
            code = 'worksheet_not_found'
        j = jsonify({
            'error': code,
            'hint': 'Check GOOGLE_SERVICE_ACCOUNT_JSON_* env vars and that the service account has access to the sheet',
            'sheet_id': sheet_id,
            'worksheet': worksheet,
        })
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j, 500


@main_bp.route('/api/player-projections/sheets')
def api_player_projections_sheets():
    """Fetch player projections from Google Sheets (Sheets3).
    Returns: { playerId: { PlayerID, Position, Age, Rookie, EVO, EVD, PP, SH, GSAx, ... }, ... }
    """
    sheet_id = (os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
    worksheet = (os.getenv('PROJECTIONS_WORKSHEET') or 'Sheets3').strip()
    if not sheet_id:
        return jsonify({'error': 'missing_sheet_id', 'hint': 'Set PROJECTIONS_SHEET_ID env var'}), 500
    
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        return jsonify({'error': 'gspread_unavailable', 'hint': 'Install gspread and google-auth'}), 500
    
    try:
        info = _load_google_service_account_info_from_env()
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly',
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(worksheet)
        # Use numericise_ignore to preserve comma decimal separators (like RAPM does)
        try:
            rows = ws.get_all_records(numericise_ignore=['all']) or []
        except Exception:
            rows = ws.get_all_records() or []
        
        # Build map keyed by PlayerID
        # Values are kept as strings with comma decimal separators for proper parsing
        out = {}
        for r in rows:
            try:
                pid_raw = r.get('PlayerID') or r.get('playerId') or r.get('player_id')
                if pid_raw is None:
                    continue
                pid = _safe_int(pid_raw)
                if not pid or pid <= 0:
                    # tolerate cases like '8479979.0' or '8 479 979'
                    pid = _safe_int(str(pid_raw).replace(' ', '').replace('\u00a0', '').strip())
                if not pid or pid <= 0:
                    continue
                # Return raw row - frontend will parse with _parse_locale_float equivalent
                out[pid] = r
            except Exception:
                continue
        
        j = jsonify(out)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j
    except Exception as e:
        return jsonify({'error': 'fetch_failed', 'hint': str(e)}), 500


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
    proj_map = _load_player_projections_cached()  # Use Google Sheets (same as frontend)
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
    # Started-game overrides from Google Sheets (latest ML + Win_Prop in Sheet1)
    started_sheet_id = (os.getenv('STARTED_OVERRIDES_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    started_ws = (os.getenv('STARTED_OVERRIDES_WORKSHEET') or 'Sheet1').strip()
    started_overrides: Dict[int, Dict[str, Any]] = {}
    if started_sheet_id:
        try:
            started_overrides = _load_started_game_overrides_from_sheet(started_sheet_id, started_ws)
        except Exception:
            started_overrides = {}

    # Fallback prestart snapshots (append-only CSV) and index by GameID
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
            # When started, prefer the latest Sheet1 overrides; else fallback to CSV snapshot
            if started and gid is not None:
                ov = started_overrides.get(gid)
                if isinstance(ov, dict) and ov:
                    # Direct away/home fields
                    if 'oddsAway' in ov or 'oddsHome' in ov or 'winAwayPct' in ov or 'winHomePct' in ov:
                        g['prestart'] = {
                            'oddsAway': ov.get('oddsAway'),
                            'oddsHome': ov.get('oddsHome'),
                            # If Win_Prop is blank, keep None so UI falls back to calculated probability
                            'winAwayPct': ov.get('winAwayPct'),
                            'winHomePct': ov.get('winHomePct'),
                        }
                    # Team-row shape: map Team->(ml, winPct) to away/home using the schedule abbrev
                    elif isinstance(ov.get('_by_team'), dict):
                        tm = ov.get('_by_team') or {}
                        aa = ((g.get('awayTeam') or {}).get('abbrev') or '').upper()
                        ha = ((g.get('homeTeam') or {}).get('abbrev') or '').upper()
                        a_rec = tm.get(aa) if aa else None
                        h_rec = tm.get(ha) if ha else None
                        g['prestart'] = {
                            'oddsAway': a_rec.get('ml') if isinstance(a_rec, dict) else None,
                            'oddsHome': h_rec.get('ml') if isinstance(h_rec, dict) else None,
                            'winAwayPct': a_rec.get('winPct') if isinstance(a_rec, dict) else None,
                            'winHomePct': h_rec.get('winPct') if isinstance(h_rec, dict) else None,
                        }
                elif gid in prestart_map:
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


@main_bp.route('/api/skaters/players')
def api_skaters_players():
    """Return selectable skaters for a given team + season, or full league.

    We want *all* skaters who played for the team in that season (including traded / inactive).

    Query params:
      team=BOS
      season=20252026 (optional; defaults to current season)
      seasonState=regular|playoffs|all (optional; default regular)
      scope=team|league (optional; default team)
    """
    scope = str(request.args.get('scope') or request.args.get('playerScope') or 'team').strip().lower()
    is_league = scope in {'league', 'all', 'full'} or str(request.args.get('league') or '').strip() in {'1', 'true', 'yes'}

    team = str(request.args.get('team') or '').upper().strip()
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or request.args.get('season_state') or 'regular').strip().lower()
    if (not is_league) and (not team):
        return jsonify({'players': []})

    # Prefer NHL stats skater bios (seasonId + currentTeamAbbrev) for any season.
    # Fallback for older seasons: roster/{team}/{season} (historical roster by season).
    # NOTE: skater bios endpoint returns 500 if cayenneExp is omitted.
    season_i = _safe_int(season)
    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0
    if not season_i:
        season_i = current_i

    players: List[Dict[str, Any]] = []

    # Primary source: NHL stats skater summary filtered by teamAbbrev.
    # This includes players who played for the team that season (including multi-team "teamAbbrevs").
    # Docs/shape: https://api.nhle.com/stats/rest/en/skater/summary
    try:
        players_ttl_s = max(60, int(os.getenv('SKATERS_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        players_ttl_s = 21600

    try:
        players_cache_max = max(1, int(os.getenv('SKATERS_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
    except Exception:
        players_cache_max = 12

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'

    cache_key = (int(season_i or 0), '__LEAGUE__' if is_league else team, season_state)
    now = time.time()
    try:
        _cache_prune_ttl_and_size(_SKATERS_PLAYERS_CACHE, ttl_s=players_ttl_s, max_items=players_cache_max)
        cached = _SKATERS_PLAYERS_CACHE.get(cache_key)
        if cached and (now - float(cached[0])) < float(players_ttl_s):
            players = cached[1] or []
            j = jsonify({'players': players})
            try:
                j.headers['Cache-Control'] = 'no-store'
            except Exception:
                pass
            return j
    except Exception:
        pass

    if not players:
        try:
            if is_league:
                if season_state == 'regular':
                    cay = f'seasonId={int(season_i)} and gameTypeId=2'
                elif season_state == 'playoffs':
                    cay = f'seasonId={int(season_i)} and gameTypeId=3'
                else:
                    cay = f'seasonId={int(season_i)} and (gameTypeId=2 or gameTypeId=3)'
            else:
                if season_state == 'regular':
                    cay = f'seasonId={int(season_i)} and gameTypeId=2 and teamAbbrev="{team}"'
                elif season_state == 'playoffs':
                    cay = f'seasonId={int(season_i)} and gameTypeId=3 and teamAbbrev="{team}"'
                else:
                    cay = f'seasonId={int(season_i)} and (gameTypeId=2 or gameTypeId=3) and teamAbbrev="{team}"'

            url = 'https://api.nhle.com/stats/rest/en/skater/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=25,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        pid = _safe_int(row.get('playerId'))
                        if not pid or pid <= 0:
                            continue
                        name = str(row.get('skaterFullName') or '').strip() or str(pid)
                        pos = str(row.get('positionCode') or '').strip().upper()
                        if pos.startswith('G'):
                            continue
                        # For league queries, stats API typically returns 'teamAbbrevs' (sometimes multi-team).
                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': pos}
                        if team_abbrev:
                            rec['team'] = team_abbrev
                        players.append(rec)
        except Exception:
            players = []

        try:
            # De-dupe by playerId
            seen: set[int] = set()
            uniq: List[Dict[str, Any]] = []
            for p in players:
                try:
                    pid_i = int(p.get('playerId') or 0)
                    if pid_i <= 0 or pid_i in seen:
                        continue
                    seen.add(pid_i)
                    uniq.append(p)
                except Exception:
                    continue
            players = uniq
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_SKATERS_PLAYERS_CACHE, cache_key, players, ttl_s=players_ttl_s, max_items=players_cache_max)
        except Exception:
            pass

    # Fallbacks (best-effort) if stats summary fails.
    if (not players) and (not is_league):
        bios_map = _load_skater_bios_season_cached(int(season_i or 0))
        try:
            for pid, info in (bios_map or {}).items():
                try:
                    if not pid:
                        continue
                    t = str((info or {}).get('team') or '').strip().upper()
                    if t != team:
                        continue
                    name = str((info or {}).get('name') or '').strip() or str(pid)
                    pos_code = str((info or {}).get('positionCode') or (info or {}).get('position') or '').strip().upper()
                    players.append({'playerId': int(pid), 'name': name, 'pos': pos_code})
                except Exception:
                    continue
        except Exception:
            players = []

    if (not players) and (not is_league) and season_i and current_i and season_i != current_i:
        url = f'https://api-web.nhle.com/v1/roster/{team.lower()}/{season_i}'
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code == 200:
                data = r.json() if r.content else {}
                if isinstance(data, dict):
                    forwards = data.get('forwards') or []
                    defensemen = data.get('defensemen') or []
                    for p in list(forwards) + list(defensemen):
                        if not isinstance(p, dict):
                            continue
                        pid = _safe_int(p.get('id') or p.get('playerId'))
                        if not pid or pid <= 0:
                            continue
                        fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                        ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                        pos = str(p.get('positionCode') or p.get('position') or '').strip().upper()
                        name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                        players.append({'playerId': int(pid), 'name': name, 'pos': pos})
        except Exception:
            pass

    j = jsonify({'players': players})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/players')
def api_goalies_players():
    """Return selectable goalies for a given team + season, or full league.

    Query params:
      team=BOS
      season=20252026 (optional; defaults to current season)
      seasonState=regular|playoffs|all (optional; default regular)
      scope=team|league (optional; default team)
    """
    scope = str(request.args.get('scope') or request.args.get('playerScope') or 'team').strip().lower()
    is_league = scope in {'league', 'all', 'full'} or str(request.args.get('league') or '').strip() in {'1', 'true', 'yes'}

    team = str(request.args.get('team') or '').upper().strip()
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or request.args.get('season_state') or 'regular').strip().lower()
    if (not is_league) and (not team):
        return jsonify({'players': []})

    season_i = _safe_int(season)
    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0
    if not season_i:
        season_i = current_i

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'

    try:
        players_ttl_s = max(60, int(os.getenv('GOALIES_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        players_ttl_s = 21600

    try:
        players_cache_max = max(1, int(os.getenv('GOALIES_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
    except Exception:
        players_cache_max = 12

    cache_key = (int(season_i or 0), '__LEAGUE__' if is_league else team, season_state)
    # Cache hit: return immediately (avoids extra allocations/work)
    if True:
        try:
            _cache_prune_ttl_and_size(_GOALIES_PLAYERS_CACHE, ttl_s=players_ttl_s, max_items=players_cache_max)
            cached_players = _cache_get(_GOALIES_PLAYERS_CACHE, cache_key, int(players_ttl_s))
            if cached_players is not None:
                j = jsonify({'players': cached_players})
                try:
                    j.headers['Cache-Control'] = 'no-store'
                except Exception:
                    pass
                return j
        except Exception:
            pass

    players: List[Dict[str, Any]] = []
    now = time.time()
    if not players:
        try:
            if is_league:
                if season_state == 'regular':
                    cay = f'seasonId={int(season_i)} and gameTypeId=2'
                elif season_state == 'playoffs':
                    cay = f'seasonId={int(season_i)} and gameTypeId=3'
                else:
                    cay = f'seasonId={int(season_i)} and (gameTypeId=2 or gameTypeId=3)'
            else:
                if season_state == 'regular':
                    cay = f'seasonId={int(season_i)} and gameTypeId=2 and teamAbbrev="{team}"'
                elif season_state == 'playoffs':
                    cay = f'seasonId={int(season_i)} and gameTypeId=3 and teamAbbrev="{team}"'
                else:
                    cay = f'seasonId={int(season_i)} and (gameTypeId=2 or gameTypeId=3) and teamAbbrev="{team}"'

            # NHL stats goalie summary. Shape varies slightly; be defensive.
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=25,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        pid = _safe_int(row.get('playerId') or row.get('goalieId') or row.get('id'))
                        if not pid or pid <= 0:
                            continue
                        name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip() or str(pid)
                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': 'G'}
                        if team_abbrev:
                            rec['team'] = team_abbrev
                        players.append(rec)
        except Exception:
            players = []

        # Fallback for older seasons (team roster endpoint) for team-scoped queries.
        if (not players) and (not is_league) and season_i and current_i and season_i != current_i:
            url = f'https://api-web.nhle.com/v1/roster/{team.lower()}/{season_i}'
            try:
                r = requests.get(url, timeout=20, allow_redirects=True)
                if r.status_code == 200:
                    data = r.json() if r.content else {}
                    if isinstance(data, dict):
                        goalies = data.get('goalies') or []
                        for p in list(goalies):
                            if not isinstance(p, dict):
                                continue
                            pid = _safe_int(p.get('id') or p.get('playerId'))
                            if not pid or pid <= 0:
                                continue
                            fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                            ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                            name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                            players.append({'playerId': int(pid), 'name': name, 'pos': 'G', 'team': team})
            except Exception:
                pass

        try:
            seen: set[int] = set()
            uniq: List[Dict[str, Any]] = []
            for p in players:
                try:
                    pid_i = int(p.get('playerId') or 0)
                    if pid_i <= 0 or pid_i in seen:
                        continue
                    seen.add(pid_i)
                    uniq.append(p)
                except Exception:
                    continue
            players = uniq
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_GOALIES_PLAYERS_CACHE, cache_key, players, ttl_s=players_ttl_s, max_items=players_cache_max)
        except Exception:
            pass

    j = jsonify({'players': players})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/player/<int:player_id>/landing')
def api_player_landing(player_id: int):
    """Proxy NHL player landing endpoint to bypass browser CORS."""
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'error': 'invalid_player_id'}), 400

    try:
        ttl_s = max(10, int(os.getenv('PLAYER_LANDING_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600

    try:
        max_items = max(1, int(os.getenv('PLAYER_LANDING_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512

    _cache_prune_ttl_and_size(_PLAYER_LANDING_CACHE, ttl_s=ttl_s, max_items=max_items)

    cached = _cache_get(_PLAYER_LANDING_CACHE, pid, ttl_s)
    if cached is not None:
        j = jsonify(cached)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    url = f'https://api-web.nhle.com/v1/player/{pid}/landing'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
    except Exception:
        return jsonify({'error': 'fetch_failed'}), 502
    if r.status_code != 200:
        return jsonify({'error': 'upstream_error', 'status': r.status_code}), 502
    try:
        data = r.json()
    except Exception:
        return jsonify({'error': 'invalid_upstream'}), 502
    if not isinstance(data, dict):
        return jsonify({'error': 'invalid_upstream'}), 502

    _cache_set_multi_bounded(_PLAYER_LANDING_CACHE, pid, data, ttl_s=ttl_s, max_items=max_items)
    j = jsonify(data)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/player-projections/<int:player_id>')
def api_player_projections(player_id: int):
    """Return a single projections row for a playerId (Sheets3 preferred; CSV fallback)."""
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'error': 'invalid_player_id'}), 400
    proj_map = _load_player_projections_cached()
    row = proj_map.get(pid)
    if not row:
        return jsonify({'error': 'not_found'}), 404
    j = jsonify({'playerId': pid, 'row': row})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/rapm/player/<int:player_id>')
def api_rapm_player(player_id: int):
    """Return RAPM rows from app/static/rapm/rapm.csv for a player.

    Optional query params:
      season=20252026
    """
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'rows': [], 'error': 'invalid_player_id'}), 400
    season = str(request.args.get('season') or '').strip()
    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None

    rows: List[Dict[str, Any]]
    source = 'static'
    if season_int == 20252026:
        sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
        worksheet = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
        if not sheet_id:
            # Strictly requested to use Sheets4 for 20252026; return a helpful error.
            return jsonify({
                'error': 'missing_sheet_id',
                'hint': 'Set RAPM_SHEET_ID (or PROJECTIONS_SHEET_ID) and share the sheet with the service account',
                'worksheet': worksheet,
            }), 500
        try:
            rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            source = 'sheets'
        except Exception:
            # Fall back to local static CSV if Sheets read fails.
            rows = _load_rapm_static_csv()
            source = 'static'
    else:
        # Static CSVs can be large across seasons; stream+cache only this player's rows.
        rows = _load_rapm_player_rows_static(pid, season_int)
    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if str(r.get('PlayerID') or '').strip() != str(pid):
                continue
            if season_int is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != season_int:
                        continue
                except Exception:
                    continue
            # Keep only a subset needed by the Skaters RAPM tab
            out.append({
                'PlayerID': pid,
                'Season': r.get('Season'),
                'StrengthState': r.get('StrengthState'),
                'Rates_Totals': r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals'),

                'CF': r.get('CF'),
                'CA': r.get('CA'),
                'GF': r.get('GF'),
                'GA': r.get('GA'),
                'xGF': r.get('xGF'),
                'xGA': r.get('xGA'),
                'C_plusminus': r.get('C_plusminus'),
                'G_plusminus': r.get('G_plusminus'),
                'xG_plusminus': r.get('xG_plusminus'),

                'CF_zscore': r.get('CF_zscore'),
                'CA_zscore': r.get('CA_zscore'),
                'GF_zscore': r.get('GF_zscore'),
                'GA_zscore': r.get('GA_zscore'),
                'xGF_zscore': r.get('xGF_zscore'),
                'xGA_zscore': r.get('xGA_zscore'),
                'C_plusminus_zscore': r.get('C_plusminus_zscore'),
                'G_plusminus_zscore': r.get('G_plusminus_zscore'),
                'xG_plusminus_zscore': r.get('xG_plusminus_zscore'),

                'PP_CF': r.get('PP_CF'),
                'PP_GF': r.get('PP_GF'),
                'PP_xGF': r.get('PP_xGF'),
                'PP_CF_zscore': r.get('PP_CF_zscore'),
                'PP_GF_zscore': r.get('PP_GF_zscore'),
                'PP_xGF_zscore': r.get('PP_xGF_zscore'),

                'SH_CA': r.get('SH_CA'),
                'SH_GA': r.get('SH_GA'),
                'SH_xGA': r.get('SH_xGA'),
                'SH_CA_zscore': r.get('SH_CA_zscore'),
                'SH_GA_zscore': r.get('SH_GA_zscore'),
                'SH_xGA_zscore': r.get('SH_xGA_zscore'),
            })
        except Exception:
            continue

    # Stable ordering
    order = {'5v5': 0, 'PP': 1, 'SH': 2}
    out.sort(key=lambda x: (int(x.get('Season') or 0), order.get(str(x.get('StrengthState') or ''), 99)))
    j = jsonify({'playerId': pid, 'rows': out, 'source': source})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/context/player/<int:player_id>')
def api_context_player(player_id: int):
    """Return context rows from app/static/rapm/context.csv (or Sheets5 for 20252026) for a player.

    Optional query params:
      season=20252026
    """
    pid = int(player_id)
    if pid <= 0:
        return jsonify({'rows': [], 'error': 'invalid_player_id'}), 400
    season = str(request.args.get('season') or '').strip()
    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None

    rows: List[Dict[str, Any]]
    source = 'static'
    if season_int == 20252026:
        sheet_id = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
        worksheet = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
        if not sheet_id:
            return jsonify({
                'error': 'missing_sheet_id',
                'hint': 'Set RAPM_SHEET_ID (or PROJECTIONS_SHEET_ID) and share the sheet with the service account',
                'worksheet': worksheet,
            }), 500
        try:
            rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            source = 'sheets'
        except Exception:
            rows = _load_context_static_csv()
            source = 'static'
    else:
        # Static CSVs can be large across seasons; stream+cache only this player's rows.
        rows = _load_context_player_rows_static(pid, season_int)

    out: List[Dict[str, Any]] = []
    for r in rows:
        try:
            if str(r.get('PlayerID') or '').strip() != str(pid):
                continue
            if season_int is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != season_int:
                        continue
                except Exception:
                    continue
            out.append({
                'PlayerID': pid,
                'Season': r.get('Season'),
                'StrengthState': r.get('StrengthState'),
                'Minutes': r.get('Minutes'),
                'QoT_blend_xG67_G33': r.get('QoT_blend_xG67_G33'),
                'QoC_blend_xG67_G33': r.get('QoC_blend_xG67_G33'),
                'ZS_Difficulty': r.get('ZS_Difficulty'),
            })
        except Exception:
            continue

    order = {'5v5': 0, 'PP': 1, 'SH': 2}
    out.sort(key=lambda x: (int(x.get('Season') or 0), order.get(str(x.get('StrengthState') or ''), 99)))
    j = jsonify({'playerId': pid, 'rows': out, 'source': source})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/card')
def api_skaters_card():
    """Player card metrics + league percentiles from SeasonStats (Sheets6).

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or request.args.get('ratesTotals') or 'Totals').strip() or 'Totals'
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()

    scope = str(request.args.get('scope') or 'season').strip().lower()
    min_gp = _safe_int(request.args.get('minGP') or request.args.get('minGp') or request.args.get('min_gp') or 0) or 0
    min_toi_raw = request.args.get('minTOI') or request.args.get('minToi') or request.args.get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]

    # SeasonStats source:
    # - season scope:
    #   - 20252026: Sheets6 (primary)
    #   - other seasons: app/static/nhl_seasonstats.csv
    # - career scope:
    #   - all seasons: app/static/nhl_seasonstats.csv, with 20252026 replaced by Sheets6 when available
    rows_iter: Iterable[Dict[str, Any]]
    source = 'none'
    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()

    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    if scope == 'career':
        # Stream static seasons, optionally replacing 20252026 with Sheets6.
        if sheet_ok and sheet_rows is not None:
            source = 'static+sheets'

            def _it() -> Iterator[Dict[str, Any]]:
                yield from _iter_seasonstats_static_rows(skip_season=20252026)
                for rr in sheet_rows or []:
                    if isinstance(rr, dict):
                        yield rr

            rows_iter = _it()
        else:
            source = 'static'
            rows_iter = _iter_seasonstats_static_rows()
    else:
        # Season scope: prefer Sheets6 only for 20252026; otherwise stream just the requested season.
        if season_int == 20252026 and sheet_ok and sheet_rows is not None:
            source = 'sheets'
            rows_iter = sheet_rows
        else:
            source = 'static'
            rows_iter = _iter_seasonstats_static_rows(season=season_int)

    # Aggregate by player under the requested filters (cached).
    agg, pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    # Apply minimum requirements (affects both the returned player and percentile pools).
    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}
        pos_group_by_pid = {pid_k: g for pid_k, g in pos_group_by_pid.items() if pid_k in eligible}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    # Choose attempts + xG based on the selected xG model.
    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    # Build per-player computed metrics used by cards.
    # Keys are metric IDs: "Category|Metric".
    defs = _load_card_metrics_defs()
    def_map: Dict[str, Dict[str, Any]] = {str(m.get('id')): m for m in (defs.get('metrics') or []) if isinstance(m, dict) and m.get('id')}

    # Backwards-compatible (old UI) mode
    if not metric_ids:
        # Old keys list
        metric_ids = [
            'Ice Time|GP',
            'Ice Time|TOI',
            'Production|iGoals',
            'Production|Assists1',
            'Production|Assists2',
            'Production|Points',
            'Shooting|ixG',
            'Shooting|Sh% or FSh%',
            'Shooting|xSh% or xFS%',
            'Shooting|dSh% or dFSh%',
        ]

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    # For some external metrics (RAPM/context z-scores) we can derive percentiles directly.
    special_pct: Dict[str, Optional[float]] = {}

    def _z_to_pct(z: Optional[float]) -> Optional[float]:
        if z is None:
            return None
        try:
            zz = float(z)
            if not math.isfinite(zz):
                return None
            # Normal CDF via erf
            return 50.0 * (1.0 + math.erf(zz / math.sqrt(2.0)))
        except Exception:
            return None

    def _lower_is_better(metric_id: str) -> bool:
        m = metric_id
        if '|' in metric_id:
            _, m = metric_id.split('|', 1)
        m = str(m or '').strip()
        return m in {
            'CA', 'FA', 'SA', 'GA', 'xGA',
            'PIM_taken', 'PIM_Against',
            'Giveaways',
            'RAPM CA', 'RAPM GA', 'RAPM xGA',
        }

    # Load RAPM/context rows only if requested.
    rapm_row: Optional[Dict[str, Any]] = None
    ctx_row: Optional[Dict[str, Any]] = None
    needs_rapm = any(('|RAPM ' in mid) for mid in metric_ids)
    needs_ctx = any((mid in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) for mid in metric_ids)

    if needs_rapm:
        rapm_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            worksheet = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
            if sheet_id:
                try:
                    rapm_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                except Exception:
                    rapm_rows = _load_rapm_static_csv()
            else:
                rapm_rows = _load_rapm_static_csv()
        else:
            rapm_rows = _load_rapm_player_rows_static(int(pid), season_int)

        # Filter to the requested player/season and pick the requested strength+rates.
        candidates: List[Dict[str, Any]] = []
        for r in rapm_rows:
            try:
                if str(r.get('PlayerID') or '').strip() != str(pid):
                    continue
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != season_int:
                            continue
                    except Exception:
                        continue
                candidates.append(r)
            except Exception:
                continue
        for r in candidates:
            if str(r.get('StrengthState') or '').strip() == want_strength and _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                rapm_row = r
                break
        if rapm_row is None:
            for r in candidates:
                if _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    rapm_row = r
                    break
        if rapm_row is None and candidates:
            rapm_row = candidates[0]

    if needs_ctx:
        ctx_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            sheet_id = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            worksheet = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
            if sheet_id:
                try:
                    ctx_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                except Exception:
                    ctx_rows = _load_context_static_csv()
            else:
                ctx_rows = _load_context_static_csv()
        else:
            ctx_rows = _load_context_player_rows_static(int(pid), season_int)

        candidates2: List[Dict[str, Any]] = []
        for r in ctx_rows:
            try:
                if str(r.get('PlayerID') or '').strip() != str(pid):
                    continue
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != season_int:
                            continue
                    except Exception:
                        continue
                candidates2.append(r)
            except Exception:
                continue
        for r in candidates2:
            if str(r.get('StrengthState') or '').strip() == want_strength:
                ctx_row = r
                break
        if ctx_row is None:
            for r in candidates2:
                if str(r.get('StrengthState') or '').strip() == '5v5':
                    ctx_row = r
                    break
        if ctx_row is None and candidates2:
            ctx_row = candidates2[0]

    def _compute_metric(metric_id: str, v: Dict[str, Any], player_id: int) -> Optional[float]:
        # SeasonStats base vars
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        # NHL Edge metrics (value + percentile come from the NHL Edge API; don't compute percentiles locally).
        if category == 'Edge':
            if int(player_id) != int(pid):
                return None
            if season_int is not None and int(season_int) < 20212022:
                special_pct[metric_id] = None
                return None
            mdef = def_map.get(metric_id) or {}
            link = str(mdef.get('link') or '').strip()
            if not link:
                special_pct[metric_id] = None
                return None

            game_type = _edge_game_type(season_state)
            url = _edge_format_url(link, int(pid), int(season_int or 0), int(game_type))
            if not url:
                special_pct[metric_id] = None
                return None
            payload_edge = _edge_get_cached_json(url)
            if not payload_edge:
                special_pct[metric_id] = None
                return None

            strength_code = None
            if str(mdef.get('strengthCode') or '').strip().lower() == 'strengthcode':
                strength_code = _edge_strength_code(strength_state)

            # Distance Skated: alternate total/per60; PerGame uses distanceTotal/GP but keeps distanceTotal percentile.
            if metric == 'distanceTotal or distancePer60':
                total_val, total_pct = _edge_extract_value_and_pct(payload_edge, 'distanceTotal', strength_code)
                per60_val, per60_pct = _edge_extract_value_and_pct(payload_edge, 'distancePer60', strength_code)
                if rates == 'Per60':
                    if per60_pct is not None:
                        special_pct[metric_id] = per60_pct
                    return per60_val
                if rates == 'PerGame':
                    if total_pct is not None:
                        special_pct[metric_id] = total_pct
                    if total_val is None or gp <= 0:
                        return None
                    return float(total_val) / float(gp)
                # Totals
                if total_pct is not None:
                    special_pct[metric_id] = total_pct
                return total_val

            # Default Edge extraction
            val_e, pct_e = _edge_extract_value_and_pct(payload_edge, str(metric or ''), strength_code)
            if pct_e is not None:
                special_pct[metric_id] = pct_e
            else:
                special_pct[metric_id] = None

            # Percent-of-time fields (zone time) come back as 0..1 fractions.
            try:
                if metric and str(metric).lower().endswith('pctg') and val_e is not None:
                    fv = float(val_e)
                    if 0.0 <= fv <= 1.5:
                        val_e = 100.0 * fv
            except Exception:
                pass
            return val_e

        # External metrics (RAPM/context) for requested player only.
        if player_id == int(pid):
            if metric and str(metric).startswith('RAPM '):
                if not rapm_row:
                    return None
                base = str(metric).replace('RAPM', '', 1).strip()
                # Map to columns + zscore columns
                col = None
                zcol = None
                if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                    col = base
                    zcol = f'{base}_zscore'
                elif base == 'C+/-':
                    col = 'C_plusminus'
                    zcol = 'C_plusminus_zscore'
                elif base == 'G+/-':
                    col = 'G_plusminus'
                    zcol = 'G_plusminus_zscore'
                elif base == 'xG+/-':
                    col = 'xG_plusminus'
                    zcol = 'xG_plusminus_zscore'
                if not col:
                    return None
                val = _parse_locale_float(rapm_row.get(col))
                z = _parse_locale_float(rapm_row.get(zcol)) if zcol else None
                pct = _z_to_pct(z)
                if pct is not None and _lower_is_better(metric_id):
                    pct = 100.0 - pct
                special_pct[metric_id] = pct
                return float(val) if val is not None else None

            if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
                if not ctx_row:
                    return None
                col2 = None
                if metric == 'QoT':
                    col2 = 'QoT_blend_xG67_G33'
                elif metric == 'QoC':
                    col2 = 'QoC_blend_xG67_G33'
                elif metric == 'ZS':
                    col2 = 'ZS_Difficulty'
                val2 = _parse_locale_float(ctx_row.get(col2)) if col2 else None
                pct2 = _z_to_pct(float(val2)) if val2 is not None else None
                special_pct[metric_id] = pct2
                return float(val2) if val2 is not None else None

        # Map common seasonstats metrics
        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        # Individual shooting metrics belong to the Shooting category.
        # (Context has its own on-ice Sh%/PDO derived from on-ice GF/GA/SF/SA.)
        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh% or xFSh%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None

        if metric == 'GAx' and category == 'Shooting':
            # Individual goals above expected
            return _rate_from(gp, toi, (igoals - ixg))

        # On-ice totals
        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        # On-ice percentages / differentials
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            # Convention: if SA=0, treat on-ice Sv% as 100%.
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        # If the definition names a seasonstats column directly, use it.
        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        # Unknown
        _ = def_map.get(metric_id)
        return None

    derived: Dict[int, Dict[str, Optional[float]]] = {}
    for pid_i, v in agg.items():
        per_player: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_player[mid] = _compute_metric(mid, v, pid_i)
        derived[pid_i] = per_player

    # Percentiles (higher is better).
    def _percentile_sorted(values_sorted: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values_sorted:
            return None
        try:
            idx = bisect.bisect_right(values_sorted, float(v))
            return 100.0 * (idx / float(len(values_sorted)))
        except Exception:
            return None

    # Prepare distributions per metric, grouped by position (F vs D).
    # (Skater Card is skaters-only; goalies are excluded above.)
    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    dist_by_pos: Dict[Tuple[str, str], List[float]] = {}
    edge_metric_ids = {mid for mid in metric_ids if str(mid).startswith('Edge|')}
    for pid_i, m in derived.items():
        g = pos_group_by_pid.get(int(pid_i)) or 'F'
        if g not in {'F', 'D'}:
            g = 'F'
        for mid in metric_ids:
            if mid in edge_metric_ids:
                continue
            vv = m.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
                dist_by_pos.setdefault((g, mid), []).append(fv)
            except Exception:
                continue

    # Sort pools once; avoids repeated allocations inside percentile calls.
    for mid, arr in dist_all.items():
        try:
            arr.sort()
        except Exception:
            continue
    for k, arr in dist_by_pos.items():
        try:
            arr.sort()
        except Exception:
            continue

    mine = derived.get(int(pid))
    seasonstats_missing = False
    if mine is None:
        # For some seasons (notably the in-progress 20252026), SeasonStats may be unavailable
        # (e.g., missing Sheets6 config) or incomplete. Don't hard-fail the entire card;
        # instead, fall back to an empty SeasonStats row so Edge/RAPM/Context metrics can render.
        if int(season_int or 0) == 20252026:
            seasonstats_missing = True
            empty_v: Dict[str, Any] = {
                'GP': 0,
                'TOI': 0.0,
                'iGoals': 0.0,
                'Assists1': 0.0,
                'Assists2': 0.0,
                'iShots': 0.0,
                'iFenwick': 0.0,
                'ixG_S': 0.0,
                'ixG_F': 0.0,
                'ixG_F2': 0.0,
                'CA': 0.0,
                'CF': 0.0,
                'FA': 0.0,
                'FF': 0.0,
                'SA': 0.0,
                'SF': 0.0,
                'GA': 0.0,
                'GF': 0.0,
                'xGA_S': 0.0,
                'xGF_S': 0.0,
                'xGA_F': 0.0,
                'xGF_F': 0.0,
                'xGA_F2': 0.0,
                'xGF_F2': 0.0,
                'PIM_taken': 0.0,
                'PIM_drawn': 0.0,
                'PIM_for': 0.0,
                'PIM_against': 0.0,
                'Hits': 0.0,
                'Takeaways': 0.0,
                'Giveaways': 0.0,
            }
            mine = {mid: _compute_metric(mid, empty_v, int(pid)) for mid in metric_ids}
            derived[int(pid)] = mine
        else:
            return jsonify({'error': 'not_found', 'playerId': int(pid), 'source': source}), 404

    out_metrics: Dict[str, Any] = {}
    my_group = pos_group_by_pid.get(int(pid)) or 'F'
    if my_group not in {'F', 'D'}:
        my_group = 'F'
    for mid in metric_ids:
        val = mine.get(mid)
        pct = special_pct.get(mid)
        if pct is None:
            pool = dist_by_pos.get((my_group, mid))
            if not pool:
                pool = dist_all.get(mid) or []
            pct = _percentile_sorted(pool, val)
            if pct is not None and _lower_is_better(mid):
                pct = 100.0 - pct
        out_metrics[mid] = {
            'value': val,
            'pct': pct,
        }

    # Provide dynamic labels used by the UI for some "or" metrics.
    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'playerId': int(pid),
        'season': season_int,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'source': source,
        'seasonStatsMissing': bool(seasonstats_missing),
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'metrics': out_metrics,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/card')
def api_goalies_card():
    """Goalie card metrics + league percentiles from SeasonStats.

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or request.args.get('ratesTotals') or 'Totals').strip() or 'Totals'
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()

    min_gp = _safe_int(request.args.get('minGP') or request.args.get('minGp') or request.args.get('min_gp') or 0) or 0
    min_toi_raw = request.args.get('minTOI') or request.args.get('minToi') or request.args.get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('goalies')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    mine_raw = agg.get(int(pid))
    if mine_raw is None:
        return jsonify({'error': 'not_found', 'playerId': int(pid)}), 404

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    # League average Sv% (weighted by SA) for GSAA.
    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and any(str(mid) in {'Results|GSAA', 'Results|GSAx'} for mid in metric_ids):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
                sheet_id=sheet_id,
                worksheet=worksheet,
                sheet_ok=sheet_ok,
                sheet_rows=sheet_rows,
            )

            for pid_i, vv in agg.items():
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    def _compute_metric(metric_id: str, pid_i: int, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)

        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)

        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)

        return None

    derived: Dict[int, Dict[str, Optional[float]]] = {}
    for pid_i, v in agg.items():
        per_player: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_player[mid] = _compute_metric(mid, int(pid_i), v)
        derived[pid_i] = per_player

    def _percentile(values: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values:
            return None
        import bisect
        arr = sorted(values)
        idx = bisect.bisect_right(arr, float(v))
        return 100.0 * (idx / float(len(arr)))

    def _lower_is_better(mid: str) -> bool:
        m = mid
        if '|' in mid:
            _, m = mid.split('|', 1)
        m = str(m or '').strip()
        return m in {'GA', 'xGA'}

    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    for _pid_i, mm in derived.items():
        for mid in metric_ids:
            vv = mm.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
            except Exception:
                continue

    mine = derived.get(int(pid)) or {}
    out_metrics: Dict[str, Any] = {}
    for mid in metric_ids:
        val = mine.get(mid)
        pool = dist_all.get(mid) or []
        pct = _percentile(pool, val)
        if pct is not None and _lower_is_better(mid):
            pct = 100.0 - pct
        out_metrics[mid] = {'value': val, 'pct': pct}

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'playerId': int(pid),
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'metrics': out_metrics,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


# -----------------------------
# Teams APIs
# -----------------------------

_TEAM_STATS_REST_CACHE: Dict[str, Tuple[float, Any]] = {}


def _team_stats_rest_get(url: str) -> Optional[Dict[str, Any]]:
    """Fetch NHL stats REST JSON with a small in-memory TTL cache."""
    try:
        ttl_s = max(30, int(os.getenv('TEAM_STATS_REST_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600

    try:
        max_items = max(1, int(os.getenv('TEAM_STATS_REST_CACHE_MAX_ITEMS', '128') or '128'))
    except Exception:
        max_items = 128

    now = time.time()
    _cache_prune_ttl_and_size(_TEAM_STATS_REST_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _TEAM_STATS_REST_CACHE.get(url)
    if cached and (now - cached[0]) < ttl_s:
        try:
            return cached[1]
        except Exception:
            return None

    try:
        r = requests.get(url, timeout=25)
        if r.status_code != 200:
            return None
        j = r.json()
        if not isinstance(j, dict):
            return None
        _cache_set_multi_bounded(_TEAM_STATS_REST_CACHE, url, j, ttl_s=ttl_s, max_items=max_items)
        return j
    except Exception:
        return None


def _edge_rank_to_pct(rank: Any, total_teams: int = 32) -> Optional[float]:
    try:
        rr = int(float(rank))
        if total_teams <= 1:
            return None
        if rr < 1 or rr > total_teams:
            return None
        return 100.0 * ((float(total_teams) - float(rr)) / float(total_teams - 1))
    except Exception:
        return None


def _edge_team_extract_value_rank_avg(
    payload: Dict[str, Any],
    metric_key: str,
    strength_code: Optional[str],
    position_code: Optional[str],
) -> Tuple[Optional[float], Optional[int], Optional[float]]:
    """Extract {value, rank, leagueAvg} from NHL Edge team payloads."""
    try:
        # Find a list-of-rows node we can filter by strengthCode/positionCode.
        rows: Optional[List[Dict[str, Any]]] = None
        for v in payload.values():
            if isinstance(v, list) and v and isinstance(v[0], dict):
                keys0 = {str(k).lower() for k in v[0].keys()}
                has_sc = 'strengthcode' in keys0
                # Some team endpoints (e.g. shot speed) use `position` instead of `positionCode`.
                has_pc = ('positioncode' in keys0) or ('position' in keys0)
                if (strength_code and has_sc) or (position_code and has_pc):
                    rows = v  # type: ignore[assignment]
                    break
                if has_sc or has_pc:
                    rows = v  # type: ignore[assignment]

        row: Optional[Dict[str, Any]] = None
        if rows:
            candidates = rows
            sc = str(strength_code).lower() if strength_code else None
            pc = str(position_code).lower() if position_code else None

            def _sc(rr0: Dict[str, Any]) -> str:
                return str(rr0.get('strengthCode') or '').lower()

            def _pc(rr0: Dict[str, Any]) -> str:
                return str(rr0.get('positionCode') or rr0.get('position') or '').lower()

            # Best match: strength + position together.
            if sc and pc:
                for rr in candidates:
                    if _sc(rr) == sc and _pc(rr) == pc:
                        row = rr
                        break
            # Fallbacks: exact strength, exact position, then mixed "all".
            if row is None and sc:
                for rr in candidates:
                    if _sc(rr) == sc:
                        row = rr
                        break
            if row is None and pc:
                for rr in candidates:
                    if _pc(rr) == pc:
                        row = rr
                        break
            if row is None and sc and pc:
                for rr in candidates:
                    if _sc(rr) == sc and _pc(rr) == 'all':
                        row = rr
                        break
            if row is None and sc and pc:
                for rr in candidates:
                    if _sc(rr) == 'all' and _pc(rr) == pc:
                        row = rr
                        break
            if row is None and sc:
                for rr in candidates:
                    if _sc(rr) == 'all':
                        row = rr
                        break
            if row is None and pc:
                for rr in candidates:
                    if _pc(rr) == 'all':
                        row = rr
                        break
            if row is None and candidates:
                row = candidates[0]
        else:
            row = None

        def _to_float(x: Any) -> Optional[float]:
            try:
                if x is None:
                    return None
                f = float(x)
                if not math.isfinite(f):
                    return None
                return f
            except Exception:
                return None

        def _pick_val(node: Any) -> Any:
            if isinstance(node, dict):
                return _ci_get(node, 'imperial') or _ci_get(node, 'value') or _ci_get(node, 'metric')
            return node

        def _pick_avg(node: Any) -> Any:
            if isinstance(node, dict):
                avg = _ci_get(node, 'leagueAvg') or _ci_get(node, 'leagueAverage')
                if isinstance(avg, dict):
                    return _ci_get(avg, 'imperial') or _ci_get(avg, 'value') or _ci_get(avg, 'metric')
                return avg
            return None

        def _pick_rank_from_row(r0: Dict[str, Any]) -> Any:
            mk = str(metric_key)
            # zone time uses offensiveZoneRank for offensiveZonePctg
            if mk.endswith('Pctg'):
                base = mk[:-4]
                return _ci_get(r0, f'{base}Rank') or _ci_get(r0, f'{mk}Rank')
            return _ci_get(r0, f'{mk}Rank')

        # Primary extraction: row-based if present.
        if isinstance(row, dict):
            node = _ci_get(row, metric_key)
            rank_raw = None
            avg_raw = None
            if isinstance(node, dict):
                rank_raw = _ci_get(node, 'rank')
                avg_raw = _pick_avg(node)
            if rank_raw is None:
                rank_raw = _pick_rank_from_row(row)

            val = _pick_val(node)
            out_val = _to_float(val)
            out_avg = _to_float(avg_raw)
            out_rank = None
            try:
                if rank_raw is not None:
                    out_rank = int(float(rank_raw))
            except Exception:
                out_rank = None

            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.5:
                out_val = 100.0 * out_val
            if out_avg is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_avg <= 1.5:
                out_avg = 100.0 * out_avg
            return (out_val, out_rank, out_avg)

        # Fallback: nested dict-of-metrics (rare for team endpoints, but safe)
        for v in payload.values():
            if isinstance(v, dict):
                node = _ci_get(v, metric_key)
                if not isinstance(node, dict):
                    continue
                out_val = _to_float(_pick_val(node))
                out_avg = _to_float(_pick_avg(node))
                out_rank = None
                try:
                    out_rank = int(float(_ci_get(node, 'rank')))
                except Exception:
                    out_rank = None
                if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.5:
                    out_val = 100.0 * out_val
                if out_avg is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_avg <= 1.5:
                    out_avg = 100.0 * out_avg
                return (out_val, out_rank, out_avg)

        return (None, None, None)
    except Exception:
        return (None, None, None)


def _team_id_by_abbrev() -> Dict[str, int]:
    out: Dict[str, int] = {}
    for r in (TEAM_ROWS or []):
        try:
            ab = str(r.get('Team') or '').strip().upper()
            tid = _safe_int(r.get('TeamID'))
            if ab and tid and tid > 0:
                out[ab] = int(tid)
        except Exception:
            continue
    return out


def _build_team_base_stats(*, scope: str, season_int: int, season_state: str) -> Dict[str, Dict[str, Any]]:
    """Return per-team base totals keyed by team abbrev."""
    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'total'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'

    team_id_map = _team_id_by_abbrev()
    abbrev_by_id = {tid: ab for ab, tid in team_id_map.items()}

    def _summary_url(game_type_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summary'
            '?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummary'
            f'&cayenneExp=seasonId={int(season_int)}%20and%20gameTypeId={int(game_type_id)}'
        )

    def _shoot_url(game_type_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summaryshooting'
            '?isAggregate=false&isGame=false&reportType=basic&reportName=teamsummaryshooting'
            f'&cayenneExp=seasonId={int(season_int)}%20and%20gameTypeId={int(game_type_id)}'
        )

    def _summary_url_total(team_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summary'
            '?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummary'
            f'&cayenneExp=teamId={int(team_id)}'
        )

    def _shoot_url_total(team_id: int) -> str:
        return (
            'https://api.nhle.com/stats/rest/en/team/summaryshooting'
            '?isAggregate=true&isGame=false&reportType=basic&reportName=teamsummaryshooting'
            f'&cayenneExp=teamId={int(team_id)}'
        )

    def _flt(x: Any) -> float:
        try:
            f = float(x)
            return f if math.isfinite(f) else 0.0
        except Exception:
            return 0.0

    def _rows_by_teamid(j: Optional[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        try:
            rows = j.get('data') if isinstance(j, dict) else None
            if not isinstance(rows, list):
                return out
            for r in rows:
                if not isinstance(r, dict):
                    continue
                tid = _safe_int(r.get('teamId'))
                if tid and tid > 0:
                    out[int(tid)] = r
        except Exception:
            return out
        return out

    # Season scope: fetch league-wide in a handful of requests.
    if scope_norm == 'season':
        gtypes: List[int]
        if ss_norm == 'regular':
            gtypes = [2]
        elif ss_norm == 'playoffs':
            gtypes = [3]
        else:
            gtypes = [2, 3]

        summary_by_tid: Dict[int, Dict[str, Any]] = {}
        shoot_by_tid: Dict[int, Dict[str, Any]] = {}
        for gt in gtypes:
            js = _team_stats_rest_get(_summary_url(gt)) or {}
            jj = _team_stats_rest_get(_shoot_url(gt)) or {}
            for tid, r in _rows_by_teamid(js).items():
                prev = summary_by_tid.get(tid)
                if not prev:
                    summary_by_tid[tid] = dict(r)
                else:
                    # Additive totals
                    prev['gamesPlayed'] = _flt(prev.get('gamesPlayed')) + _flt(r.get('gamesPlayed'))
                    prev['goalsFor'] = _flt(prev.get('goalsFor')) + _flt(r.get('goalsFor'))
                    prev['goalsAgainst'] = _flt(prev.get('goalsAgainst')) + _flt(r.get('goalsAgainst'))
                    prev['_shotsForTotal'] = _flt(prev.get('_shotsForTotal')) + (_flt(r.get('shotsForPerGame')) * _flt(r.get('gamesPlayed')))
                    prev['_shotsAgainstTotal'] = _flt(prev.get('_shotsAgainstTotal')) + (_flt(r.get('shotsAgainstPerGame')) * _flt(r.get('gamesPlayed')))
            for tid, r in _rows_by_teamid(jj).items():
                prev = shoot_by_tid.get(tid)
                if not prev:
                    shoot_by_tid[tid] = dict(r)
                else:
                    prev['gamesPlayed'] = _flt(prev.get('gamesPlayed')) + _flt(r.get('gamesPlayed'))
                    prev['satFor'] = _flt(prev.get('satFor')) + _flt(r.get('satFor'))
                    prev['satAgainst'] = _flt(prev.get('satAgainst')) + _flt(r.get('satAgainst'))
                    prev['usatFor'] = _flt(prev.get('usatFor')) + _flt(r.get('usatFor'))
                    prev['usatAgainst'] = _flt(prev.get('usatAgainst')) + _flt(r.get('usatAgainst'))

        out: Dict[str, Dict[str, Any]] = {}
        for tid, rsum in summary_by_tid.items():
            ab = abbrev_by_id.get(int(tid))
            if not ab:
                continue
            gp = float(rsum.get('gamesPlayed') or 0.0)
            sf_total = _flt(rsum.get('_shotsForTotal')) if '_shotsForTotal' in rsum else (_flt(rsum.get('shotsForPerGame')) * gp)
            sa_total = _flt(rsum.get('_shotsAgainstTotal')) if '_shotsAgainstTotal' in rsum else (_flt(rsum.get('shotsAgainstPerGame')) * gp)
            out[ab] = {
                'team': ab,
                'teamId': int(tid),
                'scope': 'season',
                'season': int(season_int),
                'seasonState': ss_norm,
                'GP': gp,
                'GF': _flt(rsum.get('goalsFor')),
                'GA': _flt(rsum.get('goalsAgainst')),
                'SF': sf_total,
                'SA': sa_total,
                'xGF': None,
                'xGA': None,
            }

        for tid, rsh in shoot_by_tid.items():
            ab = abbrev_by_id.get(int(tid))
            if not ab or ab not in out:
                continue
            out[ab].update({
                'CF': _flt(rsh.get('satFor')),
                'CA': _flt(rsh.get('satAgainst')),
                'FF': _flt(rsh.get('usatFor')),
                'FA': _flt(rsh.get('usatAgainst')),
            })

        return out

    # Total scope: per-team (teamId-filtered) aggregate endpoints.
    out2: Dict[str, Dict[str, Any]] = {}
    for ab, tid in team_id_map.items():
        js = _team_stats_rest_get(_summary_url_total(tid))
        jj = _team_stats_rest_get(_shoot_url_total(tid))
        try:
            row_s = (js or {}).get('data')
            row_j = (jj or {}).get('data')
            rs = row_s[0] if isinstance(row_s, list) and row_s and isinstance(row_s[0], dict) else {}
            rsh = row_j[0] if isinstance(row_j, list) and row_j and isinstance(row_j[0], dict) else {}

            gp = _flt(rs.get('gamesPlayed'))
            sf_total = _flt(rs.get('shotsForPerGame')) * gp
            sa_total = _flt(rs.get('shotsAgainstPerGame')) * gp
            out2[ab] = {
                'team': ab,
                'teamId': int(tid),
                'scope': 'total',
                'season': None,
                'seasonState': 'all',
                'GP': gp,
                'GF': _flt(rs.get('goalsFor')),
                'GA': _flt(rs.get('goalsAgainst')),
                'SF': sf_total,
                'SA': sa_total,
                'CF': _flt(rsh.get('satFor')),
                'CA': _flt(rsh.get('satAgainst')),
                'FF': _flt(rsh.get('usatFor')),
                'FA': _flt(rsh.get('usatAgainst')),
                'xGF': None,
                'xGA': None,
            }
        except Exception:
            continue
    return out2


@main_bp.route('/api/teams/card')
def api_teams_card():
    """Team card metrics + league percentiles (32 teams) from NHL stats REST + NHL Edge (rank->pct).

    Query params:
      season=<int>
      team=<abbrev>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      rates=Totals|PerGame
      metricIds=<comma-separated Category|Metric ids>
      scope=season|total
    """
    season = str(request.args.get('season') or '').strip()
    team_ab = str(request.args.get('team') or request.args.get('teamAbbrev') or request.args.get('team_abbrev') or '').strip().upper()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    position_code_req = str(request.args.get('positionCode') or request.args.get('posCode') or request.args.get('position') or 'all').strip().lower() or 'all'
    rates = str(request.args.get('rates') or 'Totals').strip() or 'Totals'
    scope = str(request.args.get('scope') or 'season').strip().lower()
    metric_ids_raw = str(request.args.get('metricIds') or request.args.get('metrics') or '').strip()

    try:
        season_int = int(season) if season else 20252026
    except Exception:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    if not team_ab:
        return jsonify({'error': 'missing_team'}), 400

    metric_ids: List[str] = []
    if metric_ids_raw:
        metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('teams')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    base = _build_team_base_stats(scope=scope, season_int=season_int, season_state=season_state)
    mine_base = base.get(team_ab)
    if not mine_base:
        return jsonify({'error': 'not_found', 'team': team_ab}), 404

    defs = _load_card_metrics_defs('teams')
    def_map: Dict[str, Dict[str, Any]] = {str(m.get('id')): m for m in (defs.get('metrics') or []) if isinstance(m, dict) and m.get('id')}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    # For Edge metrics we compute percentiles directly from rank (not from league distributions).
    special_pct: Dict[str, Optional[float]] = {}

    def _compute_metric(metric_id: str, v: Dict[str, Any], team_id: int) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            if int(season_int or 0) < 20212022 and scope == 'season':
                special_pct[metric_id] = None
                return None
            mdef = def_map.get(metric_id) or {}
            link = str(mdef.get('link') or '').strip()
            if not link:
                special_pct[metric_id] = None
                return None
            game_type = _edge_game_type(season_state)
            url = _edge_format_url(link, int(team_id), int(season_int or 0), int(game_type))
            if not url:
                special_pct[metric_id] = None
                return None
            payload_edge = _edge_get_cached_json(url)
            if not payload_edge:
                special_pct[metric_id] = None
                return None

            strength_code = None
            if str(mdef.get('strengthCode') or '').strip().lower() == 'strengthcode':
                strength_code = _edge_strength_code(strength_state)
            pos_code = None
            if str(mdef.get('positionCode') or '').strip().lower() in {'positioncode', 'position'}:
                pos_code = position_code_req or 'all'

            val_e, rank_e, avg_e = _edge_team_extract_value_rank_avg(payload_edge, str(metric or ''), strength_code, pos_code)
            if rank_e is not None:
                special_pct[metric_id] = _edge_rank_to_pct(rank_e, 32)
            else:
                special_pct[metric_id] = None
            # Percent-of-time fields come back as 0..1.
            if val_e is not None and str(metric or '').lower().endswith('pctg') and 0.0 <= float(val_e) <= 1.5:
                val_e = 100.0 * float(val_e)
            # Keep leagueAvg in case the UI wants it later.
            try:
                if avg_e is not None:
                    v.setdefault('_edgeAvg', {})[metric_id] = avg_e
                if rank_e is not None:
                    v.setdefault('_edgeRank', {})[metric_id] = rank_e
            except Exception:
                pass
            return val_e

        # Base totals
        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        # Percentages
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        # Differentials
        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        # Context
        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        return None

    edge_metric_ids = {mid for mid in metric_ids if str(mid).startswith('Edge|')}

    # If the request is Edge-only, do NOT compute Edge for all teams.
    # Edge endpoints already provide league rank, so we can render rank-based bars
    # without expensive 32-team fanout.
    if edge_metric_ids and len(edge_metric_ids) == len(metric_ids):
        team_id_i = int(mine_base.get('teamId') or 0)
        mine_edge: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mine_edge[mid] = _compute_metric(mid, mine_base, team_id_i)

        edge_rank_map = (mine_base.get('_edgeRank') or {}) if isinstance(mine_base, dict) else {}
        edge_avg_map = (mine_base.get('_edgeAvg') or {}) if isinstance(mine_base, dict) else {}
        out_metrics_edge: Dict[str, Any] = {}
        for mid in metric_ids:
            out_metrics_edge[mid] = {
                'value': mine_edge.get(mid),
                'pct': special_pct.get(mid),
                'rank': edge_rank_map.get(mid),
                'avg': edge_avg_map.get(mid),
            }

        payload = {
            'team': team_ab,
            'teamId': team_id_i,
            'season': int(season_int) if scope == 'season' else None,
            'scope': scope,
            'seasonState': season_state,
            'strengthState': strength_state,
            'rates': rates,
            'metrics': out_metrics_edge,
        }
        j = jsonify(payload)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    # Compute derived metrics for all teams (for percentile pools).
    derived: Dict[str, Dict[str, Optional[float]]] = {}
    for ab, v in base.items():
        tid = int(v.get('teamId') or 0)
        per_team: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            per_team[mid] = _compute_metric(mid, v, tid)
        derived[ab] = per_team

    def _percentile(values: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not values:
            return None
        import bisect
        arr = sorted(values)
        idx = bisect.bisect_right(arr, float(v))
        return 100.0 * (idx / float(len(arr)))

    # Build distributions excluding Edge (Edge percentiles come from rank).
    dist_all: Dict[str, List[float]] = {mid: [] for mid in metric_ids}
    for _ab, mm in derived.items():
        for mid in metric_ids:
            if mid in edge_metric_ids:
                continue
            vv = mm.get(mid)
            if vv is None:
                continue
            try:
                fv = float(vv)
                if not math.isfinite(fv):
                    continue
                dist_all[mid].append(fv)
            except Exception:
                continue

    mine = derived.get(team_ab) or {}

    def _lower_is_better_team(metric_id: str) -> bool:
        m = metric_id
        if '|' in metric_id:
            _, m = metric_id.split('|', 1)
        m = str(m or '').strip()
        return m in {
            'CA', 'FA', 'SA', 'GA', 'xGA',
        }

    edge_rank_map = (mine_base.get('_edgeRank') or {}) if isinstance(mine_base, dict) else {}
    edge_avg_map = (mine_base.get('_edgeAvg') or {}) if isinstance(mine_base, dict) else {}

    out_metrics: Dict[str, Any] = {}
    for mid in metric_ids:
        val = mine.get(mid)
        pct = special_pct.get(mid)
        if pct is None and mid not in edge_metric_ids:
            pool = dist_all.get(mid) or []
            pct = _percentile(pool, val)
            if pct is not None and _lower_is_better_team(mid):
                pct = 100.0 - pct
        mm: Dict[str, Any] = {'value': val, 'pct': pct}
        if mid in edge_metric_ids:
            mm['rank'] = edge_rank_map.get(mid)
            mm['avg'] = edge_avg_map.get(mid)
        out_metrics[mid] = mm

    payload = {
        'team': team_ab,
        'teamId': int(mine_base.get('teamId') or 0),
        'season': int(season_int) if scope == 'season' else None,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'rates': rates,
        'metrics': out_metrics,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/table', methods=['GET', 'POST'])
def api_teams_table():
    """Bulk team table metrics for the selected season/scope.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      rates=Totals|PerGame
      scope=season|total
      metricIds=<comma separated Category|Metric>
      includeHistoric=0|1

    Notes:
      - Does not compute NHL Edge metrics in bulk by default (those require upstream calls).
    """
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    rates = str(_get('rates') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()
    include_historic = str(_get('includeHistoric') or _get('include_historic') or '').strip()
    metric_ids_val = _get('metricIds') or _get('metrics')

    try:
        season_int = int(season) if season else 20252026
    except Exception:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('teams')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    base = _build_team_base_stats(scope=scope, season_int=season_int, season_state=season_state)

    # Filter historic teams (based on Teams.csv Active flag) unless requested.
    show_hist = include_historic in {'1', 'true', 'True', 'yes', 'YES'}
    if not show_hist:
        active = {str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if str(r.get('Active') or '').strip() == '1'}
        base = {ab: v for ab, v in base.items() if ab in active}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    def _compute_non_edge(metric_id: str, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            return None

        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        return None

    rows_out: List[Dict[str, Any]] = []
    for ab, v in base.items():
        d = {
            'team': ab,
            'teamId': int(v.get('teamId') or 0),
        }
        for mid in metric_ids:
            d[mid] = _compute_non_edge(mid, v)
        rows_out.append(d)

    # Stable ordering: team name.
    rows_out.sort(key=lambda x: str(x.get('team') or ''))
    payload = {
        'season': int(season_int) if scope == 'season' else None,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'rates': rates,
        'metricIds': metric_ids,
        'rows': rows_out,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/scatter')
def api_teams_scatter():
    """League-wide scatter data for the Teams 'Charts' tab.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      rates=Totals|PerGame
      scope=season|total
      includeHistoric=0|1
      xMetricId=<Category|Metric>
      yMetricId=<Category|Metric>

    Notes:
      - Uses NHL stats REST aggregates (non-Edge metrics only).
      - Does NOT support NHL Edge metrics.
    """
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    rates = str(request.args.get('rates') or 'Totals').strip() or 'Totals'
    scope = str(request.args.get('scope') or 'season').strip().lower()
    include_historic = str(request.args.get('includeHistoric') or request.args.get('include_historic') or '').strip()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    try:
        season_int = int(season) if season else 20252026
    except Exception:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if rates not in {'Totals', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'total'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400
    if str(x_metric_id).startswith('Edge|') or str(y_metric_id).startswith('Edge|'):
        return jsonify({'error': 'edge_not_supported'}), 400

    base = _build_team_base_stats(scope=scope, season_int=season_int, season_state=season_state)

    # Filter historic teams (based on Teams.csv Active flag) unless requested.
    show_hist = include_historic in {'1', 'true', 'True', 'yes', 'YES'}
    if not show_hist:
        active = {str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if str(r.get('Active') or '').strip() == '1'}
        base = {ab: v for ab, v in base.items() if ab in active}

    team_name_by_ab = {str(r.get('Team') or '').strip().upper(): str(r.get('Name') or '').strip() for r in (TEAM_ROWS or [])}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _rate_from(gp: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        if gp <= 0:
            return None
        try:
            return float(vv) / float(gp) if vv is not None else None
        except Exception:
            return None

    def _compute_non_edge(metric_id: str, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = v.get('xGF')
        xga = v.get('xGA')

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Edge':
            return None

        if metric == 'CA':
            return _rate_from(gp, ca)
        if metric == 'FA':
            return _rate_from(gp, fa)
        if metric == 'SA':
            return _rate_from(gp, sa)
        if metric == 'GA':
            return _rate_from(gp, ga)
        if metric == 'xGA':
            return _rate_from(gp, float(xga) if xga is not None else None)

        if metric == 'CF':
            return _rate_from(gp, cf)
        if metric == 'FF':
            return _rate_from(gp, ff)
        if metric == 'SF':
            return _rate_from(gp, sf)
        if metric == 'GF':
            return _rate_from(gp, gf)
        if metric == 'xGF':
            return _rate_from(gp, float(xgf) if xgf is not None else None)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            if xgf is None or xga is None:
                return None
            return _pct(float(xgf), float(xgf) + float(xga))

        if metric == 'C+/-':
            return _rate_from(gp, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, (gf - ga))
        if metric == 'xG+/-':
            if xgf is None or xga is None:
                return None
            return _rate_from(gp, (float(xgf) - float(xga)))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            if xgf is None:
                return None
            return _rate_from(gp, (gf - float(xgf)))
        if category == 'Context' and metric == 'GSAx':
            if xga is None:
                return None
            return _rate_from(gp, (float(xga) - ga))

        return None

    points: List[Dict[str, Any]] = []
    for ab, v in base.items():
        x = _compute_non_edge(x_metric_id, v)
        y = _compute_non_edge(y_metric_id, v)
        try:
            fx = float(x) if x is not None else None
            fy = float(y) if y is not None else None
        except Exception:
            fx = None
            fy = None
        if fx is None or fy is None:
            continue
        if not (math.isfinite(fx) and math.isfinite(fy)):
            continue
        points.append({
            'team': ab,
            'name': team_name_by_ab.get(ab) or ab,
            'x': fx,
            'y': fy,
        })

    payload = {
        'season': int(season_int) if scope == 'season' else None,
        'scope': scope,
        'seasonState': season_state,
        'rates': rates,
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'points': points,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/series')
def api_goalies_series():
    """Per-season GSAA/GSAx series for a single goalie.

    Query params:
      playerId=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2

    Notes:
      - GSAA is computed season-by-season using that season's league-average Sv% baseline.
      - GSAx is 0 before 20102011.
    """
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'

    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    seasons = by_pid_season.get(int(pid)) or {}

    def _goalie_team_map(pid_i: int, ss: str) -> Dict[int, str]:
        """Best-effort {seasonId -> teamAbbrev} from NHL Stats API goalie summary (cached)."""
        try:
            ttl_s = max(60, int(os.getenv('GOALIES_TEAM_BY_SEASON_CACHE_TTL_SECONDS', str(7 * 86400)) or str(7 * 86400)))
        except Exception:
            ttl_s = 7 * 86400
        try:
            max_items = max(1, int(os.getenv('GOALIES_TEAM_BY_SEASON_MAP_CACHE_MAX_ITEMS', '128') or '128'))
        except Exception:
            max_items = 128
        ss_norm = (ss or 'regular').strip().lower()
        if ss_norm not in {'regular', 'playoffs', 'all'}:
            ss_norm = 'regular'
        ck = (int(pid_i), ss_norm)
        now2 = time.time()
        _cache_prune_ttl_and_size(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ttl_s=ttl_s, max_items=max_items)
        cached = _GOALIES_TEAM_BY_SEASON_MAP_CACHE.get(ck)
        if cached and (now2 - float(cached[0])) < float(ttl_s):
            return cached[1] or {}

        def _toi_to_seconds(x: Any) -> int:
            try:
                if isinstance(x, (int, float)):
                    return int(x)
                s = str(x or '').strip()
                if not s:
                    return 0
                if ':' in s:
                    parts = s.split(':')
                    if len(parts) == 2:
                        return int(parts[0]) * 60 + int(parts[1])
                    if len(parts) == 3:
                        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
                return int(float(s))
            except Exception:
                return 0

        # Fetch from NHL stats API.
        try:
            if ss_norm == 'regular':
                cay = f'gameTypeId=2 and playerId={int(pid_i)}'
            elif ss_norm == 'playoffs':
                cay = f'gameTypeId=3 and playerId={int(pid_i)}'
            else:
                cay = f'(gameTypeId=2 or gameTypeId=3) and playerId={int(pid_i)}'
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=20,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list) and rows:
                    best: Dict[int, Tuple[int, str]] = {}
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        sid = row.get('seasonId')
                        season_id = _safe_int(sid)
                        if not season_id:
                            continue
                        season_id_i = int(season_id)

                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = ''
                        if isinstance(team_raw, list) and team_raw:
                            team_abbrev = str(team_raw[0] or '').strip().upper()
                        else:
                            team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        if not team_abbrev:
                            continue

                        gp = row.get('gamesPlayed') or row.get('games') or 0
                        toi = row.get('timeOnIce') or row.get('toi') or 0
                        weight = 0
                        try:
                            weight = int(gp) * 100000 + _toi_to_seconds(toi)
                        except Exception:
                            weight = _toi_to_seconds(toi)

                        prev = best.get(season_id_i)
                        if not prev or weight > int(prev[0]):
                            best[season_id_i] = (int(weight), team_abbrev)

                    team_map: Dict[int, str] = {sid: t for sid, (_, t) in best.items()}
                    _cache_set_multi_bounded(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ck, team_map, ttl_s=ttl_s, max_items=max_items)
                    return team_map
        except Exception:
            pass

        try:
            _cache_set_multi_bounded(_GOALIES_TEAM_BY_SEASON_MAP_CACHE, ck, {}, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        return {}

    def _goalie_primary_team_for_season(pid_i: int, season_i: int, ss: str) -> str:
        try:
            m = _goalie_team_map(pid_i, ss)
            return str(m.get(int(season_i)) or '').strip().upper()
        except Exception:
            return ''

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    out: List[Dict[str, Any]] = []
    for season_id in sorted(int(s) for s in seasons.keys()):
        v = seasons.get(int(season_id)) or {}
        try:
            sa = float(v.get('SA') or 0.0)
            ga = float(v.get('GA') or 0.0)
        except Exception:
            sa = 0.0
            ga = 0.0

        tot_sa, tot_ga = league_sa_ga.get(int(season_id), (0.0, 0.0))
        avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
        sv_s = _sv_frac(float(ga or 0.0), float(sa or 0.0))
        gsaa = (sv_s - avg_sv_s) * float(sa or 0.0)

        gsax = 0.0
        if int(season_id) >= 20102011:
            try:
                xga = float(_xga(v) or 0.0)
                gsax = float(xga or 0.0) - float(ga or 0.0)
            except Exception:
                gsax = 0.0

        out.append({
            'season': int(season_id),
            'team': _goalie_primary_team_for_season(int(pid), int(season_id), season_state),
            'GSAA': float(gsaa),
            'GSAx': float(gsax),
        })

    j = jsonify({
        'playerId': int(pid),
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'seasons': out,
    })
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/table', methods=['GET', 'POST'])
def api_skaters_table():
    """Bulk table metrics for a set of playerIds using the same slicers as the Card tab.

    Query params:
      season=<int>
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      scope=season|career
      minGP=<int>
      minTOI=<float minutes>
      playerIds=<comma separated ints>
      metricIds=<comma separated Category|Metric ids>

    Notes:
      - Computes SeasonStats + RAPM + Context (QoT/QoC/ZS).
      - Does NOT compute NHL Edge metrics in bulk (those require per-player upstream calls).
    """
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    xg_model = str(_get('xgModel', 'xG_F') or 'xG_F').strip()
    rates = str(_get('rates') or _get('ratesTotals') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()

    metric_ids_val = _get('metricIds') or _get('metrics')
    player_ids_val = _get('playerIds') or _get('player_ids')

    min_gp = _safe_int(_get('minGP') or _get('minGp') or _get('min_gp') or 0) or 0
    min_toi_raw = _get('minTOI') or _get('minToi') or _get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    player_ids: List[int] = []
    if isinstance(player_ids_val, list):
        for v in player_ids_val:
            pid_i = _safe_int(v)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    else:
        player_ids_raw = str(player_ids_val or '').strip()
        if not player_ids_raw:
            return jsonify({'error': 'missing_playerIds'}), 400
        for part in player_ids_raw.split(','):
            part = str(part or '').strip()
            if not part:
                continue
            pid_i = _safe_int(part)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    # De-dupe while preserving order
    seen: set[int] = set()
    player_ids = [pid for pid in player_ids if not (pid in seen or seen.add(pid))]
    if not player_ids:
        return jsonify({'error': 'empty_playerIds'}), 400

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        # Best-effort defaults: use card default metrics in definition order.
        defs0 = _load_card_metrics_defs()
        for m in (defs0.get('metrics') or []):
            try:
                if isinstance(m, dict) and m.get('default') and m.get('id'):
                    metric_ids.append(str(m.get('id')))
            except Exception:
                continue

    # SeasonStats source selection (same as Card).
    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    # Aggregate by player under the requested filters (cached).
    agg, _pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    # Apply min requirements.
    if min_gp > 0 or min_toi > 0:
        eligible = {
            pid_k
            for pid_k, d in agg.items()
            if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)
        }
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    # Only keep requested playerIds that are present after filtering.
    # This makes minGP/minTOI (and missing SeasonStats) actually remove rows.
    try:
        eligible_pids = set(int(k) for k in agg.keys())
        player_ids = [int(pid) for pid in player_ids if int(pid) in eligible_pids]
    except Exception:
        pass

    # Determine RAPM/Context needs.
    needs_rapm = any(('|RAPM ' in mid) for mid in metric_ids)
    needs_ctx = any((mid in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) for mid in metric_ids)

    want_pid_set: set[int] = set(int(pid) for pid in player_ids)

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    rapm_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    if needs_rapm:
        rapm_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            rapm_sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            rapm_ws = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
            if rapm_sheet_id:
                try:
                    rapm_rows = _load_sheet_rows_cached(rapm_sheet_id, rapm_ws, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                except Exception:
                    rapm_rows = _load_rapm_static_csv()
            else:
                rapm_rows = _load_rapm_static_csv()
        else:
            # Load whole CSV (TTL cached), then filter by season.
            rapm_rows = _load_rapm_static_csv()

        for r in rapm_rows:
            try:
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != int(season_int):
                            continue
                    except Exception:
                        continue
                pid_i = _safe_int(r.get('PlayerID'))
                if not pid_i or pid_i <= 0:
                    continue
                if int(pid_i) not in want_pid_set:
                    continue
                rapm_by_pid.setdefault(int(pid_i), []).append(r)
            except Exception:
                continue

    ctx_by_pid: Dict[int, List[Dict[str, Any]]] = {}
    if needs_ctx:
        ctx_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            ctx_sheet_id = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            ctx_ws = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
            if ctx_sheet_id:
                try:
                    ctx_rows = _load_sheet_rows_cached(ctx_sheet_id, ctx_ws, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                except Exception:
                    ctx_rows = _load_context_static_csv()
            else:
                ctx_rows = _load_context_static_csv()
        else:
            ctx_rows = _load_context_static_csv()

        for r in ctx_rows:
            try:
                if season_int is not None:
                    try:
                        if int(str(r.get('Season') or '').strip()) != int(season_int):
                            continue
                    except Exception:
                        continue
                pid_i = _safe_int(r.get('PlayerID'))
                if not pid_i or pid_i <= 0:
                    continue
                if int(pid_i) not in want_pid_set:
                    continue
                ctx_by_pid.setdefault(int(pid_i), []).append(r)
            except Exception:
                continue

    def _pick_rapm_row(pid_i: int) -> Optional[Dict[str, Any]]:
        rows = rapm_by_pid.get(int(pid_i)) or []
        if not rows:
            return None
        # Prefer exact strength + rates.
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == want_strength and _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    return r
            except Exception:
                continue
        # Fallback by rates.
        for r in rows:
            try:
                if _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) == want_rapm_rates:
                    return r
            except Exception:
                continue
        return rows[0]

    def _pick_ctx_row(pid_i: int) -> Optional[Dict[str, Any]]:
        rows = ctx_by_pid.get(int(pid_i)) or []
        if not rows:
            return None
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == want_strength:
                    return r
            except Exception:
                continue
        for r in rows:
            try:
                if str(r.get('StrengthState') or '').strip() == '5v5':
                    return r
            except Exception:
                continue
        return rows[0]

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _compute_metric(metric_id: str, v: Optional[Dict[str, Any]], pid_i: int) -> Optional[float]:
        if v is None:
            return None
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        # NHL Edge is not computed in bulk.
        if category == 'Edge':
            return None

        # RAPM + Context from preloaded maps
        if metric and str(metric).startswith('RAPM '):
            row = _pick_rapm_row(pid_i)
            if not row:
                return None
            base = str(metric).replace('RAPM', '', 1).strip()
            col = None
            if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                col = base
            elif base == 'C+/-':
                col = 'C_plusminus'
            elif base == 'G+/-':
                col = 'G_plusminus'
            elif base == 'xG+/-':
                col = 'xG_plusminus'
            if not col:
                return None
            val = _parse_locale_float(row.get(col))
            return float(val) if val is not None else None

        if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
            row = _pick_ctx_row(pid_i)
            if not row:
                return None
            col2 = None
            if metric == 'QoT':
                col2 = 'QoT_blend_xG67_G33'
            elif metric == 'QoC':
                col2 = 'QoC_blend_xG67_G33'
            elif metric == 'ZS':
                col2 = 'ZS_Difficulty'
            val2 = _parse_locale_float(row.get(col2)) if col2 else None
            return float(val2) if val2 is not None else None

        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh% or xFSh%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None
        if metric == 'GAx' and category == 'Shooting':
            return _rate_from(gp, toi, (igoals - ixg))

        # On-ice totals
        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        # On-ice percentages / differentials
        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        return None

    out_players: List[Dict[str, Any]] = []
    for pid_i in player_ids:
        v = agg.get(int(pid_i))
        mm: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mm[mid] = _compute_metric(mid, v, int(pid_i))
        out_players.append({'playerId': int(pid_i), 'metrics': mm})

    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'playerIds': player_ids,
        'metricIds': metric_ids,
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'players': out_players,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/table', methods=['GET', 'POST'])
def api_goalies_table():
    """Bulk table metrics for goalies using the same slicers as the Goalies page."""
    body: Optional[Dict[str, Any]] = None
    try:
        if request.method == 'POST':
            maybe = request.get_json(silent=True)
            if isinstance(maybe, dict):
                body = maybe
    except Exception:
        body = None

    def _get(key: str, default: Any = None) -> Any:
        try:
            if isinstance(body, dict) and key in body and body.get(key) is not None:
                return body.get(key)
        except Exception:
            pass
        return request.args.get(key, default)

    season = str(_get('season') or '').strip()
    season_state = str(_get('seasonState', 'regular') or 'regular').strip().lower()
    strength_state = str(_get('strengthState', '5v5') or '5v5').strip()
    xg_model = str(_get('xgModel', 'xG_F') or 'xG_F').strip()
    rates = str(_get('rates') or _get('ratesTotals') or 'Totals').strip() or 'Totals'
    scope = str(_get('scope', 'season') or 'season').strip().lower()

    metric_ids_val = _get('metricIds') or _get('metrics')
    player_ids_val = _get('playerIds') or _get('player_ids')

    min_gp = _safe_int(_get('minGP') or _get('minGp') or _get('min_gp') or 0) or 0
    min_toi_raw = _get('minTOI') or _get('minToi') or _get('min_toi') or 0
    try:
        min_toi = float(_parse_locale_float(min_toi_raw) or 0.0)
    except Exception:
        min_toi = 0.0
    if min_gp < 0:
        min_gp = 0
    if min_toi < 0:
        min_toi = 0.0

    player_ids: List[int] = []
    if isinstance(player_ids_val, list):
        for v in player_ids_val:
            pid_i = _safe_int(v)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    else:
        player_ids_raw = str(player_ids_val or '').strip()
        if not player_ids_raw:
            return jsonify({'error': 'missing_playerIds'}), 400
        for part in player_ids_raw.split(','):
            part = str(part or '').strip()
            if not part:
                continue
            pid_i = _safe_int(part)
            if pid_i and pid_i > 0:
                player_ids.append(int(pid_i))
    seen: set[int] = set()
    player_ids = [pid for pid in player_ids if not (pid in seen or seen.add(pid))]
    if not player_ids:
        return jsonify({'error': 'empty_playerIds'}), 400

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    metric_ids: List[str] = []
    if isinstance(metric_ids_val, list):
        metric_ids = [str(s).strip() for s in metric_ids_val if s is not None and str(s).strip()]
    else:
        metric_ids_raw = str(metric_ids_val or '').strip()
        if metric_ids_raw:
            metric_ids = [s.strip() for s in metric_ids_raw.split(',') if s and s.strip()]
    if not metric_ids:
        defs0 = _load_card_metrics_defs('goalies')
        metric_ids = [str(m.get('id')) for m in (defs0.get('metrics') or []) if isinstance(m, dict) and m.get('id')]

    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    try:
        eligible_pids = set(int(k) for k in agg.keys())
        player_ids = [int(pid) for pid in player_ids if int(pid) in eligible_pids]
    except Exception:
        pass

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and any(str(mid) in {'Results|GSAA', 'Results|GSAx'} for mid in metric_ids):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
                sheet_id=sheet_id,
                worksheet=worksheet,
                sheet_ok=sheet_ok,
                sheet_rows=sheet_rows,
            )

            pid_set = set(int(pid) for pid in player_ids)
            for pid_i in pid_set:
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    def _compute_metric(metric_id: str, pid_i: int, v: Optional[Dict[str, Any]]) -> Optional[float]:
        if v is None:
            return None
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)
        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)
        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)

        return None

    out_players: List[Dict[str, Any]] = []
    for pid_i in player_ids:
        v = agg.get(int(pid_i))
        mm: Dict[str, Optional[float]] = {}
        for mid in metric_ids:
            mm[mid] = _compute_metric(mid, int(pid_i), v)
        out_players.append({'playerId': int(pid_i), 'metrics': mm})

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'season': int(season_int),
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'playerIds': player_ids,
        'metricIds': metric_ids,
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'players': out_players,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/edge')
def api_skaters_edge():
    """Return a compact set of NHL Edge metrics for the Skaters 'Edge' tab.

    Query params:
      season=20252026
      playerId=<int>
      seasonState=regular|playoffs

    Notes:
      - NHL Edge data availability starts at 20212022.
      - Percentiles are returned by NHL Edge (0..1) and converted to 0..100.
      - Strength filter is applied only to distance + zone time metrics.
    """
    season = str(request.args.get('season') or '').strip()
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()

    pid = _safe_int(player_id_q)
    if not pid or pid <= 0:
        return jsonify({'error': 'missing_playerId'}), 400

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs'}:
        season_state = 'regular'

    # NHL Edge data begins in 20212022.
    if season_int < 20212022:
        j0 = jsonify({
            'playerId': int(pid),
            'season': int(season_int),
            'seasonState': season_state,
            'gameType': _edge_game_type(season_state),
            'available': False,
            'reason': 'edge_unavailable_before_20212022',
            'shotSpeed': {},
            'skatingSpeed': {},
            'zoneTime': {},
            'skatingDistance': {},
        })
        try:
            j0.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j0

    game_type = _edge_game_type(season_state)
    base = 'https://api-web.nhle.com/v1/edge'
    urls = {
        'shotSpeed': f'{base}/skater-shot-speed-detail/{int(pid)}/{int(season_int)}/{int(game_type)}',
        'skatingSpeed': f'{base}/skater-skating-speed-detail/{int(pid)}/{int(season_int)}/{int(game_type)}',
        'zoneTime': f'{base}/skater-zone-time/{int(pid)}/{int(season_int)}/{int(game_type)}',
        'skatingDistance': f'{base}/skater-skating-distance-detail/{int(pid)}/{int(season_int)}/{int(game_type)}',
    }

    payload_shot = _edge_get_cached_json(urls['shotSpeed']) or {}
    payload_skate = _edge_get_cached_json(urls['skatingSpeed']) or {}
    payload_zone = _edge_get_cached_json(urls['zoneTime']) or {}
    payload_dist = _edge_get_cached_json(urls['skatingDistance']) or {}

    def pack(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str] = None) -> Dict[str, Any]:
        v, p, a = _edge_extract_value_pct_avg(payload, metric_key, strength_code)
        return {'value': v, 'pct': p, 'avg': a}

    shot_metrics = {
        'topShotSpeed': pack(payload_shot, 'topShotSpeed'),
        'avgShotSpeed': pack(payload_shot, 'avgShotSpeed'),
        'shotAttempts70to80': pack(payload_shot, 'shotAttempts70to80'),
        'shotAttempts80to90': pack(payload_shot, 'shotAttempts80to90'),
        'shotAttempts90to100': pack(payload_shot, 'shotAttempts90to100'),
        'shotAttemptsOver100': pack(payload_shot, 'shotAttemptsOver100'),
    }

    skating_speed_metrics = {
        'maxSkatingSpeed': pack(payload_skate, 'maxSkatingSpeed'),
        'bursts18to20': pack(payload_skate, 'bursts18to20'),
        'bursts20to22': pack(payload_skate, 'bursts20to22'),
        'burstsOver22': pack(payload_skate, 'burstsOver22'),
    }

    strength_codes = ['all', 'es', 'pp', 'pk']
    zone_time_by_strength: Dict[str, Any] = {}
    skating_distance_by_strength: Dict[str, Any] = {}
    for sc in strength_codes:
        zone_time_by_strength[sc] = {
            'offensiveZonePctg': pack(payload_zone, 'offensiveZonePctg', sc),
            'neutralZonePctg': pack(payload_zone, 'neutralZonePctg', sc),
            'defensiveZonePctg': pack(payload_zone, 'defensiveZonePctg', sc),
        }
        skating_distance_by_strength[sc] = {
            'distanceTotal': pack(payload_dist, 'distanceTotal', sc),
            'distancePer60': pack(payload_dist, 'distancePer60', sc),
        }

    payload = {
        'playerId': int(pid),
        'season': int(season_int),
        'seasonState': season_state,
        'gameType': int(game_type),
        'available': True,
        'shotSpeed': shot_metrics,
        'skatingSpeed': skating_speed_metrics,
        'zoneTime': zone_time_by_strength,
        'skatingDistance': skating_distance_by_strength,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/skaters/scatter')
def api_skaters_scatter():
    """League-wide scatter data for the Skaters 'Charts' tab.

    Query params (match Card slicers):
      season=20252026
      seasonState=regular|playoffs|all
      strengthState=5v5|PP|SH|Other|all
      xgModel=xG_S|xG_F|xG_F2
      rates=Totals|Per60|PerGame
      scope=season|career
      minGP=<int>
      minTOI=<float>

    Scatter params:
      xMetricId=<Category|Metric>
      yMetricId=<Category|Metric>

    Notes:
      - Uses SeasonStats aggregates for all players.
      - Supports RAPM and Context (QoT/QoC/ZS) metrics via static CSVs and/or Sheets for 20252026.
      - Does NOT support NHL Edge metrics.
    """
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or 'Totals').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    try:
        min_gp = int(float(str(request.args.get('minGP') or '0').strip() or '0'))
    except Exception:
        min_gp = 0
    try:
        min_toi = float(str(request.args.get('minTOI') or '0').strip() or '0')
    except Exception:
        min_toi = 0.0

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400
    if str(x_metric_id).startswith('Edge|') or str(y_metric_id).startswith('Edge|'):
        return jsonify({'error': 'edge_not_supported'}), 400

    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    agg, _pos_group_by_pid = _build_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    # Apply minimum requirements.
    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    def _pct(n: Optional[float], d: Optional[float]) -> Optional[float]:
        try:
            if n is None or d is None:
                return None
            if d <= 0:
                return None
            return 100.0 * (float(n) / float(d))
        except Exception:
            return None

    def _attempts(v: Dict[str, Any]) -> float:
        vv = v.get('iShots') if xg_model == 'xG_S' else v.get('iFenwick')
        return float(vv or 0.0)

    def _ixg(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('ixG_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('ixG_F2') or 0.0)
        return float(v.get('ixG_S') or 0.0)

    def _xgf(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGF_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGF_F2') or 0.0)
        return float(v.get('xGF_S') or 0.0)

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _norm_rates_totals(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip() or 'Rates'

    want_strength = strength_state if strength_state in {'5v5', 'PP', 'SH'} else '5v5'
    want_rapm_rates = 'Totals' if rates == 'Totals' else 'Rates'

    # Optional RAPM/context maps for league-wide lookup.
    rapm_by_pid: Dict[int, Dict[str, Any]] = {}
    ctx_by_pid: Dict[int, Dict[str, Any]] = {}

    needs_rapm = ('|RAPM ' in x_metric_id) or ('|RAPM ' in y_metric_id)
    needs_ctx = (x_metric_id in {'Context|QoT', 'Context|QoC', 'Context|ZS'}) or (y_metric_id in {'Context|QoT', 'Context|QoC', 'Context|ZS'})

    if needs_rapm:
        rapm_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            rid = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            rws = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
            if rid:
                try:
                    rapm_rows = _load_sheet_rows_cached(rid, rws, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60) or []
                except Exception:
                    rapm_rows = _load_rapm_static_csv() or []
            else:
                rapm_rows = _load_rapm_static_csv() or []
        else:
            rapm_rows = _load_rapm_static_csv() or []

        for r in rapm_rows:
            try:
                if season_int is not None:
                    if int(str(r.get('Season') or '').strip()) != int(season_int):
                        continue
                st = str(r.get('StrengthState') or '').strip()
                rt = _norm_rates_totals(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals'))
                if st != want_strength or rt != want_rapm_rates:
                    continue
                pid_r = int(str(r.get('PlayerID') or '').strip())
                if pid_r <= 0:
                    continue
                rapm_by_pid[pid_r] = r
            except Exception:
                continue

    if needs_ctx:
        ctx_rows: List[Dict[str, Any]] = []
        if season_int == 20252026:
            cid = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            cws = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
            if cid:
                try:
                    ctx_rows = _load_sheet_rows_cached(cid, cws, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60) or []
                except Exception:
                    ctx_rows = _load_context_static_csv() or []
            else:
                ctx_rows = _load_context_static_csv() or []
        else:
            ctx_rows = _load_context_static_csv() or []

        for r in ctx_rows:
            try:
                if season_int is not None:
                    if int(str(r.get('Season') or '').strip()) != int(season_int):
                        continue
                st = str(r.get('StrengthState') or '').strip()
                if st != want_strength:
                    continue
                pid_r = int(str(r.get('PlayerID') or '').strip())
                if pid_r <= 0:
                    continue
                ctx_by_pid[pid_r] = r
            except Exception:
                continue

    def _compute_metric(metric_id: str, v: Dict[str, Any], player_id: int) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        igoals = float(v.get('iGoals') or 0.0)
        a1 = float(v.get('Assists1') or 0.0)
        a2 = float(v.get('Assists2') or 0.0)
        pts = igoals + a1 + a2
        att = _attempts(v)
        ixg = _ixg(v)

        cf = float(v.get('CF') or 0.0)
        ca = float(v.get('CA') or 0.0)
        ff = float(v.get('FF') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sf = float(v.get('SF') or 0.0)
        sa = float(v.get('SA') or 0.0)
        gf = float(v.get('GF') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xgf = _xgf(v)
        xga = _xga(v)

        pim_taken = float(v.get('PIM_taken') or 0.0)
        pim_drawn = float(v.get('PIM_drawn') or 0.0)
        pim_for = float(v.get('PIM_for') or 0.0)
        pim_against = float(v.get('PIM_against') or 0.0)
        hits = float(v.get('Hits') or 0.0)
        takeaways = float(v.get('Takeaways') or 0.0)
        giveaways = float(v.get('Giveaways') or 0.0)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        # No Edge support here.
        if category == 'Edge':
            return None

        # League-wide RAPM.
        if metric and str(metric).startswith('RAPM '):
            rrow = rapm_by_pid.get(int(player_id))
            if not rrow:
                return None
            base = str(metric).replace('RAPM', '', 1).strip()
            col = None
            if base in {'CF', 'CA', 'GF', 'GA', 'xGF', 'xGA'}:
                col = base
            elif base == 'C+/-':
                col = 'C_plusminus'
            elif base == 'G+/-':
                col = 'G_plusminus'
            elif base == 'xG+/-':
                col = 'xG_plusminus'
            if not col:
                return None
            val = _parse_locale_float(rrow.get(col))
            return float(val) if val is not None else None

        # League-wide Context (QoT/QoC/ZS).
        if category == 'Context' and metric in {'QoT', 'QoC', 'ZS'}:
            crow = ctx_by_pid.get(int(player_id))
            if not crow:
                return None
            col2 = None
            if metric == 'QoT':
                col2 = 'QoT_blend_xG67_G33'
            elif metric == 'QoC':
                col2 = 'QoC_blend_xG67_G33'
            elif metric == 'ZS':
                col2 = 'ZS_Difficulty'
            val2 = _parse_locale_float(crow.get(col2)) if col2 else None
            return float(val2) if val2 is not None else None

        if metric == 'GP':
            return gp
        if metric == 'TOI':
            return toi

        if metric == 'iGoals':
            return _rate_from(gp, toi, igoals)
        if metric == 'Assists1':
            return _rate_from(gp, toi, a1)
        if metric == 'Assists2':
            return _rate_from(gp, toi, a2)
        if metric == 'Points':
            return _rate_from(gp, toi, pts)

        if metric in {'iShots', 'iFenwick', 'iShots or iFenwick'}:
            vv = float(v.get('iShots') or 0.0) if xg_model == 'xG_S' else float(v.get('iFenwick') or 0.0)
            return _rate_from(gp, toi, vv)

        if metric in {'ixG', 'Individual xG'}:
            return _rate_from(gp, toi, ixg)

        if category == 'Shooting' and metric in {'Sh% or FSh%', 'Sh%'}:
            return _pct(igoals, att)
        if category == 'Shooting' and metric in {'xSh% or xFS%', 'xSh%'}:
            return _pct(ixg, att)
        if category == 'Shooting' and metric in {'dSh% or dFSh%'}:
            sh = _pct(igoals, att)
            xsh = _pct(ixg, att)
            return (sh - xsh) if (sh is not None and xsh is not None) else None
        if metric == 'GAx' and category == 'Shooting':
            return _rate_from(gp, toi, (igoals - ixg))

        if metric == 'CF':
            return _rate_from(gp, toi, cf)
        if metric == 'CA':
            return _rate_from(gp, toi, ca)
        if metric == 'FF':
            return _rate_from(gp, toi, ff)
        if metric == 'FA':
            return _rate_from(gp, toi, fa)
        if metric == 'SF':
            return _rate_from(gp, toi, sf)
        if metric == 'SA':
            return _rate_from(gp, toi, sa)
        if metric == 'GF':
            return _rate_from(gp, toi, gf)
        if metric == 'GA':
            return _rate_from(gp, toi, ga)
        if metric == 'xGF':
            return _rate_from(gp, toi, xgf)
        if metric == 'xGA':
            return _rate_from(gp, toi, xga)

        if metric == 'CF%':
            return _pct(cf, (cf + ca))
        if metric == 'FF%':
            return _pct(ff, (ff + fa))
        if metric == 'SF%':
            return _pct(sf, (sf + sa))
        if metric == 'GF%':
            return _pct(gf, (gf + ga))
        if metric == 'xGF%':
            return _pct(xgf, (xgf + xga))
        if metric == 'C+/-':
            return _rate_from(gp, toi, (cf - ca))
        if metric == 'F+/-':
            return _rate_from(gp, toi, (ff - fa))
        if metric == 'S+/-':
            return _rate_from(gp, toi, (sf - sa))
        if metric == 'G+/-':
            return _rate_from(gp, toi, (gf - ga))
        if metric == 'xG+/-':
            return _rate_from(gp, toi, (xgf - xga))

        if category == 'Context' and metric == 'Sh%':
            return _pct(gf, sf)
        if category == 'Context' and metric == 'Sv%':
            if sa <= 0:
                return 100.0 if ga <= 0 else 0.0
            return 100.0 * (1.0 - (ga / sa))
        if category == 'Context' and metric == 'PDO':
            sh_oi = _pct(gf, sf)
            sv_oi = 100.0 if sa <= 0 and ga <= 0 else (0.0 if sa <= 0 else 100.0 * (1.0 - (ga / sa)))
            return (sh_oi + sv_oi) if (sh_oi is not None and sv_oi is not None) else None
        if category == 'Context' and metric == 'GAx':
            return _rate_from(gp, toi, (gf - xgf))
        if category == 'Context' and metric == 'GSAx':
            return _rate_from(gp, toi, (xga - ga))

        if category == 'Penalties' and metric == 'PIM_taken':
            return _rate_from(gp, toi, pim_taken)
        if category == 'Penalties' and metric == 'PIM_drawn':
            return _rate_from(gp, toi, pim_drawn)
        if category == 'Penalties' and metric == 'PIM+/-':
            return _rate_from(gp, toi, (pim_drawn - pim_taken))
        if category == 'Penalties' and metric == 'PIM_For':
            return _rate_from(gp, toi, pim_for)
        if category == 'Penalties' and metric == 'PIM_Against':
            return _rate_from(gp, toi, pim_against)
        if category == 'Penalties' and metric == 'oiPIM+/-':
            return _rate_from(gp, toi, (pim_for - pim_against))

        if category == 'Other' and metric == 'Hits':
            return _rate_from(gp, toi, hits)
        if category == 'Other' and metric == 'Takeaways':
            return _rate_from(gp, toi, takeaways)
        if category == 'Other' and metric == 'Giveaways':
            return _rate_from(gp, toi, giveaways)

        # If the metric names a column directly, use it.
        if metric and metric in v:
            try:
                return _rate_from(gp, toi, float(v.get(metric) or 0.0))
            except Exception:
                return None

        return None

    # Player/team labels via season-aware roster mapping.
    roster_map: Dict[int, Dict[str, Any]] = {}
    try:
        roster_map = _load_all_rosters_for_season_cached(int(season_int or 0)) or {}
    except Exception:
        roster_map = {}

    pts_out: List[Dict[str, Any]] = []
    for pid_i, v in agg.items():
        try:
            xv = _compute_metric(x_metric_id, v, int(pid_i))
            yv = _compute_metric(y_metric_id, v, int(pid_i))
            if xv is None or yv is None:
                continue
            xf = float(xv)
            yf = float(yv)
            if not math.isfinite(xf) or not math.isfinite(yf):
                continue
            info = roster_map.get(int(pid_i)) or {}
            team = str(info.get('team') or '').strip().upper()
            name = str(info.get('name') or '').strip()
            if not name:
                name = str(pid_i)
            pts_out.append({
                'playerId': int(pid_i),
                'name': name,
                'team': team,
                'x': xf,
                'y': yf,
                'gp': float(v.get('GP') or 0.0),
                'toi': float(v.get('TOI') or 0.0),
            })
        except Exception:
            continue

    label_attempts = 'iShots' if xg_model == 'xG_S' else 'iFenwick'
    label_sh = 'Sh%' if xg_model == 'xG_S' else 'FSh%'
    label_xsh = 'xSh%' if xg_model == 'xG_S' else 'xFSh%'
    label_dsh = 'dSh%' if xg_model == 'xG_S' else 'dFSh%'

    payload = {
        'season': season_int,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'labels': {
            'Attempts': label_attempts,
            'Sh': label_sh,
            'xSh': label_xsh,
            'dSh': label_dsh,
        },
        'points': pts_out,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/scatter')
def api_goalies_scatter():
    """League-wide scatter data for the Goalies 'Charts' tab."""
    season = str(request.args.get('season') or '').strip()
    season_state = str(request.args.get('seasonState') or 'regular').strip().lower()
    strength_state = str(request.args.get('strengthState') or '5v5').strip()
    xg_model = str(request.args.get('xgModel') or 'xG_F').strip()
    rates = str(request.args.get('rates') or 'Totals').strip()
    scope = str(request.args.get('scope') or 'season').strip().lower()
    x_metric_id = str(request.args.get('xMetricId') or request.args.get('xMetric') or '').strip()
    y_metric_id = str(request.args.get('yMetricId') or request.args.get('yMetric') or '').strip()

    try:
        min_gp = int(float(str(request.args.get('minGP') or '0').strip() or '0'))
    except Exception:
        min_gp = 0
    try:
        min_toi = float(str(request.args.get('minTOI') or '0').strip() or '0')
    except Exception:
        min_toi = 0.0

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None
    if season_int is None:
        season_int = 20252026

    if season_state not in {'regular', 'playoffs', 'all'}:
        season_state = 'regular'
    if strength_state not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        strength_state = '5v5'
    if xg_model not in {'xG_S', 'xG_F', 'xG_F2'}:
        xg_model = 'xG_F'
    if rates not in {'Totals', 'Per60', 'PerGame'}:
        rates = 'Totals'
    if scope not in {'season', 'career'}:
        scope = 'season'

    if not x_metric_id or not y_metric_id:
        return jsonify({'error': 'missing_metric', 'hint': 'Provide xMetricId and yMetricId'}), 400

    sheet_id = (os.getenv('SEASONSTATS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('SEASONSTATS_WORKSHEET') or 'Sheets6').strip()
    sheet_rows: Optional[List[Dict[str, Any]]] = None
    sheet_ok = False
    if sheet_id:
        try:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='SEASONSTATS_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
            sheet_ok = True
        except Exception:
            sheet_rows = None
            sheet_ok = False

    agg, _pos_group_by_pid = _build_goalies_seasonstats_agg(
        scope=scope,
        season_int=season_int,
        season_state=season_state,
        strength_state=strength_state,
        sheet_id=sheet_id,
        worksheet=worksheet,
        sheet_ok=sheet_ok,
        sheet_rows=sheet_rows,
    )

    if min_gp > 0 or min_toi > 0:
        eligible = {pid_k for pid_k, d in agg.items() if float(d.get('GP') or 0) >= float(min_gp) and float(d.get('TOI') or 0.0) >= float(min_toi)}
        agg = {pid_k: d for pid_k, d in agg.items() if pid_k in eligible}

    def _xga(v: Dict[str, Any]) -> float:
        if xg_model == 'xG_F':
            return float(v.get('xGA_F') or 0.0)
        if xg_model == 'xG_F2':
            return float(v.get('xGA_F2') or 0.0)
        return float(v.get('xGA_S') or 0.0)

    def _rate_from(gp: float, toi: float, vv: Optional[float]) -> Optional[float]:
        if rates == 'Totals':
            return vv
        denom = None
        if rates == 'PerGame':
            denom = gp if gp > 0 else None
        elif rates == 'Per60':
            denom = (toi / 60.0) if toi > 0 else None
        if vv is None or denom is None or denom <= 0:
            return None
        try:
            return float(vv) / float(denom)
        except Exception:
            return None

    def _sv_frac(ga: float, att: float) -> float:
        if att <= 0:
            return 1.0 if ga <= 0 else 0.0
        return 1.0 - (ga / att)

    total_sa = 0.0
    total_ga = 0.0
    for _pid_i, vv in agg.items():
        try:
            total_sa += float(vv.get('SA') or 0.0)
            total_ga += float(vv.get('GA') or 0.0)
        except Exception:
            continue
    avg_sv = _sv_frac(float(total_ga or 0.0), float(total_sa or 0.0))

    career_gsaa_by_pid: Dict[int, float] = {}
    career_gsax_by_pid: Dict[int, float] = {}
    if scope == 'career' and (x_metric_id in {'Results|GSAA', 'Results|GSAx'} or y_metric_id in {'Results|GSAA', 'Results|GSAx'}):
        try:
            by_pid_season, league_sa_ga = _build_goalies_career_season_matrix(
                season_state=season_state,
                strength_state=strength_state,
                sheet_id=sheet_id,
                worksheet=worksheet,
                sheet_ok=sheet_ok,
                sheet_rows=sheet_rows,
            )

            for pid_i in agg.keys():
                seasons = by_pid_season.get(int(pid_i)) or {}
                gsaa_sum = 0.0
                gsax_sum = 0.0
                for s_id, srow in seasons.items():
                    try:
                        sa_s = float(srow.get('SA') or 0.0)
                        ga_s = float(srow.get('GA') or 0.0)
                    except Exception:
                        continue

                    tot_sa, tot_ga = league_sa_ga.get(int(s_id), (0.0, 0.0))
                    avg_sv_s = _sv_frac(float(tot_ga or 0.0), float(tot_sa or 0.0))
                    sv_s = _sv_frac(ga_s, sa_s)
                    gsaa_sum += (sv_s - avg_sv_s) * float(sa_s or 0.0)

                    if int(s_id) >= 20102011:
                        try:
                            xga_s = _xga(srow)
                            gsax_sum += float(xga_s or 0.0) - float(ga_s or 0.0)
                        except Exception:
                            continue

                career_gsaa_by_pid[int(pid_i)] = float(gsaa_sum)
                career_gsax_by_pid[int(pid_i)] = float(gsax_sum)
        except Exception:
            career_gsaa_by_pid = {}
            career_gsax_by_pid = {}

    def _compute_metric(metric_id: str, pid_i: int, v: Dict[str, Any]) -> Optional[float]:
        gp = float(v.get('GP') or 0.0)
        toi = float(v.get('TOI') or 0.0)
        fa = float(v.get('FA') or 0.0)
        sa = float(v.get('SA') or 0.0)
        ga = float(v.get('GA') or 0.0)
        xga = _xga(v)

        category = None
        metric = None
        if '|' in metric_id:
            category, metric = metric_id.split('|', 1)
        else:
            metric = metric_id

        if category == 'Workload' and metric == 'FA':
            return _rate_from(gp, toi, fa)
        if category == 'Workload' and metric == 'SA':
            return _rate_from(gp, toi, sa)
        if category == 'Workload' and metric == 'xGA':
            return _rate_from(gp, toi, xga)
        if category == 'Workload' and metric == 'GA':
            return _rate_from(gp, toi, ga)
        if category == 'Save Percentage' and metric == 'Sv% or FSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(ga, denom)
        if category == 'Save Percentage' and metric == 'xSv% or xFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            return 100.0 * _sv_frac(xga, denom)
        if category == 'Save Percentage' and metric == 'dSv% or dFSv%':
            denom = sa if xg_model == 'xG_S' else fa
            sv = 100.0 * _sv_frac(ga, denom)
            xsv = 100.0 * _sv_frac(xga, denom)
            return (sv - xsv)
        if category == 'Results' and metric == 'GSAx':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsax_by_pid.get(int(pid_i), 0.0)))
            if int(season_int or 0) < 20102011:
                return _rate_from(gp, toi, 0.0)
            return _rate_from(gp, toi, (xga - ga))
        if category == 'Results' and metric == 'GSAA':
            if scope == 'career':
                return _rate_from(gp, toi, float(career_gsaa_by_pid.get(int(pid_i), 0.0)))
            sv = _sv_frac(ga, sa)
            gsaa = (sv - avg_sv) * sa
            return _rate_from(gp, toi, gsaa)
        return None

    def _ensure_league_goalie_map() -> Dict[int, Dict[str, Any]]:
        try:
            ttl_s = max(60, int(os.getenv('GOALIES_PLAYERS_CACHE_TTL_SECONDS', '21600') or '21600'))
        except Exception:
            ttl_s = 21600
        try:
            max_items = max(1, int(os.getenv('GOALIES_PLAYERS_CACHE_MAX_ITEMS', '12') or '12'))
        except Exception:
            max_items = 12
        ck = (int(season_int or 0), '__LEAGUE__', season_state)
        now2 = time.time()
        try:
            _cache_prune_ttl_and_size(_GOALIES_PLAYERS_CACHE, ttl_s=ttl_s, max_items=max_items)
            cached_players = _cache_get(_GOALIES_PLAYERS_CACHE, ck, int(ttl_s))
            if cached_players is not None:
                return {int(r.get('playerId') or 0): r for r in (cached_players or []) if int(r.get('playerId') or 0) > 0}
        except Exception:
            pass

        players: List[Dict[str, Any]] = []
        try:
            if season_state == 'regular':
                cay = f'seasonId={int(season_int)} and gameTypeId=2'
            elif season_state == 'playoffs':
                cay = f'seasonId={int(season_int)} and gameTypeId=3'
            else:
                cay = f'seasonId={int(season_int)} and (gameTypeId=2 or gameTypeId=3)'
            url = 'https://api.nhle.com/stats/rest/en/goalie/summary'
            r = requests.get(
                url,
                params={'limit': -1, 'start': 0, 'cayenneExp': cay},
                headers={'User-Agent': 'Mozilla/5.0'},
                timeout=25,
                allow_redirects=True,
            )
            if r.status_code == 200:
                data = r.json() if r.content else {}
                rows = data.get('data') if isinstance(data, dict) else None
                if isinstance(rows, list):
                    for row in rows:
                        if not isinstance(row, dict):
                            continue
                        pid = _safe_int(row.get('playerId') or row.get('goalieId') or row.get('id'))
                        if not pid or pid <= 0:
                            continue
                        name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip() or str(pid)
                        team_raw = row.get('teamAbbrev') or row.get('teamAbbrevs') or row.get('currentTeamAbbrev') or ''
                        team_abbrev = str(team_raw or '').strip().upper()
                        if '/' in team_abbrev:
                            team_abbrev = team_abbrev.split('/')[0].strip().upper()
                        rec: Dict[str, Any] = {'playerId': int(pid), 'name': name, 'pos': 'G'}
                        if team_abbrev:
                            rec['team'] = team_abbrev
                        players.append(rec)
        except Exception:
            players = []

        try:
            _cache_set_multi_bounded(_GOALIES_PLAYERS_CACHE, ck, players, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        return {int(r.get('playerId') or 0): r for r in players if int(r.get('playerId') or 0) > 0}

    goalie_info_by_pid = _ensure_league_goalie_map()

    pts_out: List[Dict[str, Any]] = []
    for pid_i, v in agg.items():
        try:
            xv = _compute_metric(x_metric_id, int(pid_i), v)
            yv = _compute_metric(y_metric_id, int(pid_i), v)
            if xv is None or yv is None:
                continue
            xf = float(xv)
            yf = float(yv)
            if not math.isfinite(xf) or not math.isfinite(yf):
                continue
            info = goalie_info_by_pid.get(int(pid_i)) or {}
            team = str(info.get('team') or '').strip().upper()
            name = str(info.get('name') or '').strip()
            if not name:
                name = str(pid_i)
            pts_out.append({
                'playerId': int(pid_i),
                'name': name,
                'team': team,
                'x': xf,
                'y': yf,
                'gp': float(v.get('GP') or 0.0),
                'toi': float(v.get('TOI') or 0.0),
            })
        except Exception:
            continue

    label_attempts = 'SA' if xg_model == 'xG_S' else 'FA'
    label_sv = 'Sv%' if xg_model == 'xG_S' else 'FSv%'
    label_xsv = 'xSv%' if xg_model == 'xG_S' else 'xFSv%'
    label_dsv = 'dSv%' if xg_model == 'xG_S' else 'dFSv%'

    payload = {
        'season': season_int,
        'scope': scope,
        'seasonState': season_state,
        'strengthState': strength_state,
        'xgModel': xg_model,
        'rates': rates,
        'minGP': int(min_gp),
        'minTOI': float(min_toi),
        'xMetricId': x_metric_id,
        'yMetricId': y_metric_id,
        'labels': {
            'Attempts': label_attempts,
            'Sv': label_sv,
            'xSv': label_xsv,
            'dSv': label_dsv,
        },
        'points': pts_out,
    }
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/rapm/scale')
def api_rapm_scale():
    """League min/max scales for the Skaters RAPM chart.

    Query params:
      season=20252026
      rates=Rates|Totals
      metric=corsi|xg|goals

    Returns ranges for:
      - fivev5: differential (C+/xG+/G+)
      - pp: PP offense (PP_CF/PP_xGF/PP_GF) from StrengthState=PP rows
      - sh: SH defense (-SH_CA/-SH_xGA/-SH_GA) from StrengthState=SH rows
    """
    season = str(request.args.get('season') or '').strip()
    rates = str(request.args.get('rates') or 'Rates').strip() or 'Rates'
    metric = str(request.args.get('metric') or 'corsi').strip().lower() or 'corsi'
    player_id_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    try:
        player_id_int = int(player_id_q) if player_id_q else None
    except Exception:
        player_id_int = None
    if metric not in {'corsi', 'xg', 'goals'}:
        metric = 'corsi'

    cache_key = (season, rates, metric)
    try:
        ttl_s = max(30, int(os.getenv('RAPM_SCALE_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    try:
        max_items = max(1, int(os.getenv('RAPM_SCALE_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    now = time.time()

    try:
        _cache_prune_ttl_and_size(_RAPM_SCALE_CACHE, ttl_s=ttl_s, max_items=max_items)
    except Exception:
        pass
    cached = _RAPM_SCALE_CACHE.get(cache_key)
    if cached and (now - cached[0]) < ttl_s:
        payload, dists = cached[1], cached[2]
        if player_id_int is not None:
            payload = dict(payload)
            try:
                payload['playerId'] = player_id_int
                payload['player'] = _compute_player_scale_payload(player_id_int, dists)
            except Exception:
                payload['playerId'] = player_id_int
                payload['player'] = {'error': 'player_calc_failed'}
        j = jsonify(payload)
        try:
            j.headers['Cache-Control'] = 'no-store'
        except Exception:
            pass
        return j

    try:
        season_int = int(season) if season else None
    except Exception:
        season_int = None

    # Eligibility thresholds (minutes)
    MIN_5V5 = 100.0
    MIN_PP = 40.0
    MIN_SH = 40.0

    # Load rows (Sheets4 for 20252026; else static)
    rows: List[Dict[str, Any]] = []
    source = 'static'
    if season_int == 20252026:
        sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
        worksheet = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
        if sheet_id:
            try:
                rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                source = 'sheets'
            except Exception:
                rows = _load_rapm_static_csv()
                source = 'static'
        else:
            rows = _load_rapm_static_csv()
            source = 'static'
    else:
        rows = _load_rapm_static_csv()

    # Load context minutes (Sheets5 for 20252026; else static context.csv)
    ctx_rows: List[Dict[str, Any]] = []
    ctx_source = 'static'
    if season_int == 20252026:
        sheet_id = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
        worksheet = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
        if sheet_id:
            try:
                ctx_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60)
                ctx_source = 'sheets'
            except Exception:
                ctx_rows = _load_context_static_csv()
                ctx_source = 'static'
        else:
            ctx_rows = _load_context_static_csv()
            ctx_source = 'static'
    else:
        ctx_rows = _load_context_static_csv()

    minutes_by_pid_strength: Dict[Tuple[int, str], float] = {}
    for r in ctx_rows:
        try:
            if season_int is not None:
                if int(str(r.get('Season') or '').strip()) != season_int:
                    continue
            pid = int(str(r.get('PlayerID') or '').strip())
            st = str(r.get('StrengthState') or '').strip()
            mins = _parse_locale_float(r.get('Minutes'))
            if mins is None:
                continue
            minutes_by_pid_strength[(pid, st)] = float(mins)
        except Exception:
            continue

    def _rt(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip()

    # Columns for the requested metric
    if metric == 'xg':
        diff_col = 'xG_plusminus'
        pp_col = 'PP_xGF'
        sh_col = 'SH_xGA'
        pp_base = 'xGF'
        sh_base = 'xGA'
    elif metric == 'goals':
        diff_col = 'G_plusminus'
        pp_col = 'PP_GF'
        sh_col = 'SH_GA'
        pp_base = 'GF'
        sh_base = 'GA'
    else:
        diff_col = 'C_plusminus'
        pp_col = 'PP_CF'
        sh_col = 'SH_CA'
        pp_base = 'CF'
        sh_base = 'CA'

    # Build per-player values; apply eligibility by minutes
    five_by_pid: Dict[int, float] = {}
    pp_by_pid: Dict[int, float] = {}
    sh_by_pid: Dict[int, float] = {}
    five_off_by_pid: Dict[int, float] = {}
    five_def_by_pid: Dict[int, float] = {}

    for r in rows:
        try:
            if season_int is not None:
                if int(str(r.get('Season') or '').strip()) != season_int:
                    continue
            if _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) != _rt(rates):
                continue
            pid = int(str(r.get('PlayerID') or '').strip())
            st = str(r.get('StrengthState') or '').strip()

            if st == '5v5':
                vdiff = _parse_locale_float(r.get(diff_col))
                if vdiff is not None:
                    five_by_pid[pid] = float(vdiff)

                voff = _parse_locale_float(r.get('CF' if metric == 'corsi' else ('xGF' if metric == 'xg' else 'GF')))
                vdef_raw = _parse_locale_float(r.get('CA' if metric == 'corsi' else ('xGA' if metric == 'xg' else 'GA')))
                if voff is not None:
                    five_off_by_pid[pid] = float(voff)
                if vdef_raw is not None:
                    five_def_by_pid[pid] = -float(vdef_raw)

                # Fallback PP/SH columns on 5v5 row, only if not already present.
                vpp = _parse_locale_float(r.get(pp_col))
                if vpp is not None and pid not in pp_by_pid:
                    pp_by_pid[pid] = float(vpp)
                vsh = _parse_locale_float(r.get(sh_col))
                if vsh is not None and pid not in sh_by_pid:
                    sh_by_pid[pid] = -float(vsh)

            elif st == 'PP':
                vpp = _parse_locale_float(r.get(pp_col))
                if vpp is None:
                    vpp = _parse_locale_float(r.get(pp_base))
                if vpp is not None:
                    pp_by_pid[pid] = float(vpp)

            elif st == 'SH':
                vsh = _parse_locale_float(r.get(sh_col))
                if vsh is None:
                    vsh = _parse_locale_float(r.get(sh_base))
                if vsh is not None:
                    sh_by_pid[pid] = -float(vsh)
        except Exception:
            continue

    def _eligible(pid: int, strength: str) -> bool:
        mins = minutes_by_pid_strength.get((pid, strength))
        if mins is None:
            return False
        if strength == '5v5':
            return mins >= MIN_5V5
        if strength == 'PP':
            return mins >= MIN_PP
        if strength == 'SH':
            return mins >= MIN_SH
        return False

    five_vals = [v for pid, v in five_by_pid.items() if _eligible(pid, '5v5')]
    pp_vals = [v for pid, v in pp_by_pid.items() if _eligible(pid, 'PP')]
    sh_vals = [v for pid, v in sh_by_pid.items() if _eligible(pid, 'SH')]

    five_off_vals = [v for pid, v in five_off_by_pid.items() if _eligible(pid, '5v5')]
    five_def_vals = [v for pid, v in five_def_by_pid.items() if _eligible(pid, '5v5')]

    def _minmax(vals: List[float]) -> Dict[str, Any]:
        if not vals:
            return {'min': None, 'max': None}
        return {'min': float(min(vals)), 'max': float(max(vals))}

    payload = {
        'season': season_int,
        'rates': _rt(rates),
        'metric': metric,
        'source': source,
        'contextSource': ctx_source,
        'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
        'fivev5': _minmax(five_vals),
        'pp': _minmax(pp_vals),
        'sh': _minmax(sh_vals),
    }

    # Build distributions for percentile calcs (sorted for bisect)
    dists: Dict[str, Any] = {
        'fivev5_diff': sorted(five_vals),
        'fivev5_off': sorted(five_off_vals),
        'fivev5_def': sorted(five_def_vals),
        'pp_off': sorted(pp_vals),
        'sh_def': sorted(sh_vals),
        'minutes': minutes_by_pid_strength,
        'values': {
            'fivev5_diff': five_by_pid,
            'fivev5_off': five_off_by_pid,
            'fivev5_def': five_def_by_pid,
            'pp_off': pp_by_pid,
            'sh_def': sh_by_pid,
        },
        'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
    }

    def _bisect_pct(sorted_vals: List[float], v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not sorted_vals:
            return None
        import bisect
        idx = bisect.bisect_right(sorted_vals, v)
        return 100.0 * (idx / float(len(sorted_vals)))

    def _player_payload(pid: int) -> Dict[str, Any]:
        mins5 = minutes_by_pid_strength.get((pid, '5v5'))
        minsp = minutes_by_pid_strength.get((pid, 'PP'))
        minss = minutes_by_pid_strength.get((pid, 'SH'))
        elig5 = mins5 is not None and mins5 >= MIN_5V5
        eligp = minsp is not None and minsp >= MIN_PP
        eligs = minss is not None and minss >= MIN_SH
        v5off = five_off_by_pid.get(pid)
        v5def = five_def_by_pid.get(pid)
        v5diff = five_by_pid.get(pid)
        vpp = pp_by_pid.get(pid)
        vsh = sh_by_pid.get(pid)
        return {
            'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
            'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
            'percentiles': {
                '5v5_off': _bisect_pct(dists['fivev5_off'], v5off) if elig5 else None,
                '5v5_def': _bisect_pct(dists['fivev5_def'], v5def) if elig5 else None,
                '5v5_diff': _bisect_pct(dists['fivev5_diff'], v5diff) if elig5 else None,
                'pp_off': _bisect_pct(dists['pp_off'], vpp) if eligp else None,
                'sh_def': _bisect_pct(dists['sh_def'], vsh) if eligs else None,
            },
        }

    # Expose player percentiles/eligibility when requested
    if player_id_int is not None:
        payload = dict(payload)
        payload['playerId'] = player_id_int
        payload['player'] = _player_payload(player_id_int)

    try:
        payload_base = payload if player_id_int is None else {k: v for k, v in payload.items() if k not in {'player', 'playerId'}}
        _cache_set_multi_bounded(_RAPM_SCALE_CACHE, cache_key, payload_base, dists, ttl_s=ttl_s, max_items=max_items)
    except Exception:
        pass
    j = jsonify(payload)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _compute_player_scale_payload(player_id: int, dists: Dict[str, Any]) -> Dict[str, Any]:
    """Compute eligibility + percentiles for one player from cached distributions."""
    mins = dists.get('minutes') or {}
    thresholds = dists.get('thresholds') or {}
    vmap = (dists.get('values') or {})
    try:
        MIN_5V5 = float(thresholds.get('fivev5', 100.0))
        MIN_PP = float(thresholds.get('pp', 40.0))
        MIN_SH = float(thresholds.get('sh', 40.0))
    except Exception:
        MIN_5V5, MIN_PP, MIN_SH = 100.0, 40.0, 40.0

    mins5 = mins.get((player_id, '5v5'))
    minsp = mins.get((player_id, 'PP'))
    minss = mins.get((player_id, 'SH'))
    elig5 = mins5 is not None and mins5 >= MIN_5V5
    eligp = minsp is not None and minsp >= MIN_PP
    eligs = minss is not None and minss >= MIN_SH

    import bisect
    def _pct(key: str, val: Optional[float]) -> Optional[float]:
        if val is None:
            return None
        arr = dists.get(key) or []
        if not arr:
            return None
        idx = bisect.bisect_right(arr, val)
        return 100.0 * (idx / float(len(arr)))

    v5off = (vmap.get('fivev5_off') or {}).get(player_id)
    v5def = (vmap.get('fivev5_def') or {}).get(player_id)
    v5diff = (vmap.get('fivev5_diff') or {}).get(player_id)
    vpp = (vmap.get('pp_off') or {}).get(player_id)
    vsh = (vmap.get('sh_def') or {}).get(player_id)

    return {
        'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
        'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
        'percentiles': {
            '5v5_off': _pct('fivev5_off', v5off) if elig5 else None,
            '5v5_def': _pct('fivev5_def', v5def) if elig5 else None,
            '5v5_diff': _pct('fivev5_diff', v5diff) if elig5 else None,
            'pp_off': _pct('pp_off', vpp) if eligp else None,
            'sh_def': _pct('sh_def', vsh) if eligs else None,
        },
    }


@main_bp.route('/api/rapm/career')
def api_rapm_career():
    """Career RAPM series for a single player.

    Query params:
      playerId=<int>
      rates=Rates|Totals
      metric=corsi|xg|goals
      strength=5v5|PP|SH

    Output includes per-season values + per-season percentiles filtered by minutes thresholds.
    Scales are league-aware (min/max across eligible players per season).
    """
    pid_q = str(request.args.get('playerId') or request.args.get('player_id') or '').strip()
    try:
        pid = int(pid_q)
    except Exception:
        return jsonify({'error': 'missing_playerId'}), 400

    rates = str(request.args.get('rates') or 'Rates').strip() or 'Rates'
    metric = str(request.args.get('metric') or 'corsi').strip().lower() or 'corsi'
    strength = str(request.args.get('strength') or '5v5').strip() or '5v5'
    if metric not in {'corsi', 'xg', 'goals'}:
        metric = 'corsi'
    if strength not in {'All', '5v5', 'PP', 'SH'}:
        strength = '5v5'

    def _rt(v: Any) -> str:
        s = str(v or '').strip().lower()
        if s.startswith('tot'):
            return 'Totals'
        if s.startswith('rate'):
            return 'Rates'
        return str(v or '').strip()

    # thresholds (minutes)
    MIN_5V5 = 100.0
    MIN_PP = 40.0
    MIN_SH = 40.0

    def _season_int(v: Any) -> Optional[int]:
        """Parse a season into an int like 20252026.

        Accepts common formats from CSV/Sheets:
        - 20252026
        - "20252026"
        - "2025-2026" / "2025/2026"
        - "2025-26" / "2025/26"
        """
        if v is None:
            return None
        s = str(v).strip()
        if not s:
            return None
        # Fast path: all digits
        if s.isdigit():
            try:
                n = int(s)
                return n if n >= 10000000 else None
            except Exception:
                return None

        # Normalize separators
        s2 = re.sub(r"[^0-9]", "", s)
        if len(s2) == 8:
            try:
                return int(s2)
            except Exception:
                return None

        # Handle YYYY-YY / YYYY/YY
        m = re.match(r"^(\d{4})\D+(\d{2})$", s)
        if m:
            try:
                a = int(m.group(1))
                b = int(m.group(2))
                end = (a // 100) * 100 + b
                return int(f"{a}{end:04d}")
            except Exception:
                return None
        return None

    # Build (and cache) league distributions/scales by season for this rates/metric/strength.
    cache_key = (_rt(rates), metric, strength)
    try:
        ttl_s = max(30, int(os.getenv('RAPM_CAREER_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    try:
        max_items = max(1, int(os.getenv('RAPM_CAREER_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    now = time.time()

    league = None
    try:
        _cache_prune_ttl_and_size(_RAPM_CAREER_CACHE, ttl_s=ttl_s, max_items=max_items)
        league = _cache_get(_RAPM_CAREER_CACHE, cache_key, int(ttl_s))
    except Exception:
        league = None

    if league is None:
        # Load RAPM rows: static + (optional) replace 20252026 with Sheets4.
        rapm_rows = _load_rapm_static_csv() or []
        try:
            sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            worksheet = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
            if sheet_id:
                sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60) or []
                # Keep only rows that are NOT 20252026 from static (any format), then append sheets.
                rapm_rows = [r for r in rapm_rows if _season_int(r.get('Season')) != 20252026] + sheet_rows
        except Exception:
            pass

        # Load context rows: static + (optional) replace 20252026 with Sheets5.
        ctx_rows = _load_context_static_csv() or []
        try:
            sheet_id = (os.getenv('CONTEXT_SHEET_ID') or os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or os.getenv('GOOGLE_SHEETS_ID') or '').strip()
            worksheet = (os.getenv('CONTEXT_WORKSHEET') or 'Sheets5').strip()
            if sheet_id:
                sheet_ctx = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='CONTEXT_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60) or []
                ctx_rows = [r for r in ctx_rows if _season_int(r.get('Season')) != 20252026] + sheet_ctx
        except Exception:
            pass

        minutes_by_season_pid_strength: Dict[Tuple[int, int, str], float] = {}
        for r in ctx_rows:
            try:
                season_int = _season_int(r.get('Season'))
                if season_int is None:
                    continue
                pid_i = int(str(r.get('PlayerID') or '').strip())
                st = str(r.get('StrengthState') or '').strip()
                mins = _parse_locale_float(r.get('Minutes'))
                if mins is None:
                    continue
                minutes_by_season_pid_strength[(season_int, pid_i, st)] = float(mins)
            except Exception:
                continue

        def _eligible(season_int: int, pid_i: int, st: str) -> bool:
            mins = minutes_by_season_pid_strength.get((season_int, pid_i, st))
            if mins is None:
                return False
            if st == '5v5':
                return mins >= MIN_5V5
            if st == 'PP':
                return mins >= MIN_PP
            if st == 'SH':
                return mins >= MIN_SH
            return False

        # Column names per metric
        if metric == 'xg':
            diff_col = 'xG_plusminus'
            off_col = 'xGF'
            def_col = 'xGA'
            pp_col = 'PP_xGF'
            pp_base = 'xGF'
            sh_col = 'SH_xGA'
            sh_base = 'xGA'
            z_off = 'xGF_zscore'
            z_def = 'xGA_zscore'
            z_diff = 'xG_plusminus_zscore'
            z_pp = 'PP_xGF_zscore'
            z_sh = 'SH_xGA_zscore'
        elif metric == 'goals':
            diff_col = 'G_plusminus'
            off_col = 'GF'
            def_col = 'GA'
            pp_col = 'PP_GF'
            pp_base = 'GF'
            sh_col = 'SH_GA'
            sh_base = 'GA'
            z_off = 'GF_zscore'
            z_def = 'GA_zscore'
            z_diff = 'G_plusminus_zscore'
            z_pp = 'PP_GF_zscore'
            z_sh = 'SH_GA_zscore'
        else:
            diff_col = 'C_plusminus'
            off_col = 'CF'
            def_col = 'CA'
            pp_col = 'PP_CF'
            pp_base = 'CF'
            sh_col = 'SH_CA'
            sh_base = 'CA'
            z_off = 'CF_zscore'
            z_def = 'CA_zscore'
            z_diff = 'C_plusminus_zscore'
            z_pp = 'PP_CF_zscore'
            z_sh = 'SH_CA_zscore'

        # Aggregate league distributions by season for percentiles + z-score stats
        dist_by_season: Dict[int, Dict[str, List[float]]] = {}
        scale_by_season: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}
        stats_by_season: Dict[int, Dict[str, Dict[str, Optional[float]]]] = {}

        def _push(season_int: int, key: str, v: Optional[float]):
            if v is None:
                return
            dist_by_season.setdefault(season_int, {}).setdefault(key, []).append(float(v))

        # For totals we need to combine strengths per player within a season.
        contrib: Dict[Tuple[int, int], Dict[str, float]] = {}

        # First pass: collect values per season/player/strength
        for r in rapm_rows:
            try:
                season_int = _season_int(r.get('Season'))
                if season_int is None:
                    continue
                if _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) != _rt(rates):
                    continue
                pid_i = int(str(r.get('PlayerID') or '').strip())
                st = str(r.get('StrengthState') or '').strip()

                if st == '5v5' and _eligible(season_int, pid_i, '5v5'):
                    vdiff = _parse_locale_float(r.get(diff_col))
                    voff = _parse_locale_float(r.get(off_col))
                    vdef_raw = _parse_locale_float(r.get(def_col))
                    vdef = (-vdef_raw) if vdef_raw is not None else None
                    _push(season_int, '5v5_diff', vdiff)
                    _push(season_int, '5v5_off', voff)
                    _push(season_int, '5v5_def', vdef)

                    if voff is not None and vdef is not None:
                        contrib.setdefault((season_int, pid_i), {})['5v5_total'] = float(voff) + float(vdef)

                elif st == 'PP' and _eligible(season_int, pid_i, 'PP'):
                    vpp = _parse_locale_float(r.get(pp_col))
                    if vpp is None:
                        vpp = _parse_locale_float(r.get(pp_base))
                    _push(season_int, 'pp_off', vpp)

                    if vpp is not None:
                        contrib.setdefault((season_int, pid_i), {})['pp_off'] = float(vpp)

                elif st == 'SH' and _eligible(season_int, pid_i, 'SH'):
                    vsh = _parse_locale_float(r.get(sh_col))
                    if vsh is None:
                        vsh = _parse_locale_float(r.get(sh_base))
                    vsh2 = (-vsh) if vsh is not None else None
                    _push(season_int, 'sh_def', vsh2)

                    if vsh2 is not None:
                        contrib.setdefault((season_int, pid_i), {})['sh_def'] = float(vsh2)
            except Exception:
                continue

        # Second pass: totals distributions
        for (season_int, _pid_i), d in contrib.items():
            v5 = d.get('5v5_total')
            if v5 is not None:
                _push(season_int, '5v5_total', v5)
            all_total = 0.0
            any_part = False
            for k in ('5v5_total', 'pp_off', 'sh_def'):
                if k in d:
                    all_total += float(d[k])
                    any_part = True
            if any_part:
                _push(season_int, 'all_total', all_total)

        # Compute per-season min/max for scaling
        for season_int, d in dist_by_season.items():
            out: Dict[str, Dict[str, Optional[float]]] = {}
            for k, vals in d.items():
                if not vals:
                    out[k] = {'min': None, 'max': None}
                else:
                    out[k] = {'min': float(min(vals)), 'max': float(max(vals))}
            scale_by_season[season_int] = out

        # Compute per-season mean/std for z-score (for derived totals)
        for season_int, d in dist_by_season.items():
            out2: Dict[str, Dict[str, Optional[float]]] = {}
            for k, vals in d.items():
                if not vals:
                    out2[k] = {'mean': None, 'std': None}
                    continue
                try:
                    n = float(len(vals))
                    mean = float(sum(vals) / n)
                    var = float(sum((x - mean) ** 2 for x in vals) / n)
                    std = float(var ** 0.5)
                    out2[k] = {'mean': mean, 'std': std}
                except Exception:
                    out2[k] = {'mean': None, 'std': None}
            stats_by_season[season_int] = out2

        # Sort distributions for percentile calc
        for season_int, d in dist_by_season.items():
            for k in list(d.keys()):
                d[k] = sorted(d[k])

        league = {
            'dist': dist_by_season,
            'scale': scale_by_season,
            'stats': stats_by_season,
            'minutes': minutes_by_season_pid_strength,
            'thresholds': {'fivev5': MIN_5V5, 'pp': MIN_PP, 'sh': MIN_SH},
            'cols': {
                'diff': diff_col,
                'off': off_col,
                'def': def_col,
                'pp': pp_col,
                'pp_base': pp_base,
                'sh': sh_col,
                'sh_base': sh_base,
                'z_off': z_off,
                'z_def': z_def,
                'z_diff': z_diff,
                'z_pp': z_pp,
                'z_sh': z_sh,
            },
        }
        try:
            _cache_set_multi_bounded(_RAPM_CAREER_CACHE, cache_key, league, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass

    # Pull this player's per-season series from RAPM rows
    # For correctness and simplicity, re-read relevant player rows from sources.
    rapm_rows = _load_rapm_static_csv() or []
    try:
        sheet_id = (os.getenv('RAPM_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
        worksheet = (os.getenv('RAPM_WORKSHEET') or 'Sheets4').strip()
        if sheet_id:
            sheet_rows = _load_sheet_rows_cached(sheet_id, worksheet, ttl_env='RAPM_SHEET_ROWS_CACHE_TTL_SECONDS', default_ttl=60) or []
            rapm_rows = [r for r in rapm_rows if _season_int(r.get('Season')) != 20252026] + sheet_rows
    except Exception:
        pass

    cols = league.get('cols') or {}
    dist = league.get('dist') or {}
    scale = league.get('scale') or {}
    stats = league.get('stats') or {}
    minutes_map = league.get('minutes') or {}
    thresholds = league.get('thresholds') or {'fivev5': 100.0, 'pp': 40.0, 'sh': 40.0}

    def _mins(season_int: int, st: str) -> Optional[float]:
        return minutes_map.get((season_int, pid, st))

    def _elig(season_int: int, st: str) -> bool:
        m = _mins(season_int, st)
        if m is None:
            return False
        if st == '5v5':
            return m >= float(thresholds.get('fivev5', 100.0))
        if st == 'PP':
            return m >= float(thresholds.get('pp', 40.0))
        if st == 'SH':
            return m >= float(thresholds.get('sh', 40.0))
        return False

    import bisect
    def _pct(season_int: int, key: str, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        arr = (dist.get(season_int) or {}).get(key) or []
        if not arr:
            return None
        idx = bisect.bisect_right(arr, v)
        return 100.0 * (idx / float(len(arr)))

    def _z(season_int: int, key: str, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        mm = (stats.get(season_int) or {}).get(key) or {}
        mean = mm.get('mean')
        std = mm.get('std')
        if mean is None or std is None:
            return None
        try:
            std_f = float(std)
            if std_f == 0:
                return None
            return (float(v) - float(mean)) / std_f
        except Exception:
            return None

    series: Dict[int, Dict[str, Any]] = {}
    for r in rapm_rows:
        try:
            if _rt(r.get('Rates_Totals') or r.get('Rates/Totals') or r.get('RatesTotals')) != _rt(rates):
                continue
            if int(str(r.get('PlayerID') or '').strip()) != pid:
                continue
            season_int = _season_int(r.get('Season'))
            if season_int is None:
                continue
            st = str(r.get('StrengthState') or '').strip()
            if st not in {'5v5', 'PP', 'SH'}:
                continue

            item = series.setdefault(season_int, {'Season': season_int})
            if st == '5v5':
                off_key = str(cols.get('off') or '')
                def_key = str(cols.get('def') or '')
                diff_key = str(cols.get('diff') or '')
                z_off_key = str(cols.get('z_off') or '')
                z_def_key = str(cols.get('z_def') or '')
                z_diff_key = str(cols.get('z_diff') or '')

                voff = _parse_locale_float(r.get(off_key))
                vdef_raw = _parse_locale_float(r.get(def_key))
                vdef = (-vdef_raw) if vdef_raw is not None else None
                vdiff = _parse_locale_float(r.get(diff_key))
                item['5v5_off'] = voff
                item['5v5_def'] = vdef
                item['5v5_diff'] = vdiff
                zoff = _parse_locale_float(r.get(z_off_key))
                zdef_raw = _parse_locale_float(r.get(z_def_key))
                zdiff = _parse_locale_float(r.get(z_diff_key))
                item['5v5_off_z'] = zoff
                item['5v5_def_z'] = (-zdef_raw) if zdef_raw is not None else None
                item['5v5_diff_z'] = zdiff
            elif st == 'PP':
                pp_key = str(cols.get('pp') or '')
                pp_base_key = str(cols.get('pp_base') or '')
                z_pp_key = str(cols.get('z_pp') or '')
                vpp = _parse_locale_float(r.get(pp_key))
                if vpp is None:
                    vpp = _parse_locale_float(r.get(pp_base_key))
                item['pp_off'] = vpp
                item['pp_off_z'] = _parse_locale_float(r.get(z_pp_key))
            elif st == 'SH':
                sh_key = str(cols.get('sh') or '')
                sh_base_key = str(cols.get('sh_base') or '')
                z_sh_key = str(cols.get('z_sh') or '')
                vsh = _parse_locale_float(r.get(sh_key))
                if vsh is None:
                    vsh = _parse_locale_float(r.get(sh_base_key))
                item['sh_def'] = (-vsh) if vsh is not None else None
                zsh_raw = _parse_locale_float(r.get(z_sh_key))
                item['sh_def_z'] = (-zsh_raw) if zsh_raw is not None else None
        except Exception:
            continue

    seasons = sorted(series.keys())
    points: List[Dict[str, Any]] = []
    for season_int in seasons:
        row = series.get(season_int) or {'Season': season_int}
        mins5 = _mins(season_int, '5v5')
        minsp = _mins(season_int, 'PP')
        minss = _mins(season_int, 'SH')
        elig5 = _elig(season_int, '5v5')
        eligp = _elig(season_int, 'PP')
        eligs = _elig(season_int, 'SH')
        p: Dict[str, Any] = {
            'Season': season_int,
            'minutes': {'5v5': mins5, 'PP': minsp, 'SH': minss},
            'eligible': {'5v5': elig5, 'PP': eligp, 'SH': eligs},
        }
        if elig5:
            p['5v5_off'] = row.get('5v5_off')
            p['5v5_def'] = row.get('5v5_def')
            p['5v5_diff'] = row.get('5v5_diff')
            p['5v5_off_z'] = row.get('5v5_off_z')
            p['5v5_def_z'] = row.get('5v5_def_z')
            p['5v5_diff_z'] = row.get('5v5_diff_z')
            p['5v5_off_pct'] = _pct(season_int, '5v5_off', row.get('5v5_off'))
            p['5v5_def_pct'] = _pct(season_int, '5v5_def', row.get('5v5_def'))
            p['5v5_diff_pct'] = _pct(season_int, '5v5_diff', row.get('5v5_diff'))

            try:
                v5off = row.get('5v5_off')
                v5def = row.get('5v5_def')
                if v5off is not None and v5def is not None:
                    vtot = float(v5off) + float(v5def)
                    p['5v5_total'] = vtot
                    p['5v5_total_pct'] = _pct(season_int, '5v5_total', vtot)
                    p['5v5_total_z'] = _z(season_int, '5v5_total', vtot)
            except Exception:
                pass
        if eligp:
            p['pp_off'] = row.get('pp_off')
            p['pp_off_z'] = row.get('pp_off_z')
            p['pp_off_pct'] = _pct(season_int, 'pp_off', row.get('pp_off'))
        if eligs:
            p['sh_def'] = row.get('sh_def')
            p['sh_def_z'] = row.get('sh_def_z')
            p['sh_def_pct'] = _pct(season_int, 'sh_def', row.get('sh_def'))

        # All-strength total for this season: sum any eligible components.
        try:
            total_all = 0.0
            any_part = False
            if p.get('5v5_total') is not None:
                total_all += float(p['5v5_total'])
                any_part = True
            if eligp:
                vpp_any = p.get('pp_off')
                if vpp_any is not None:
                    total_all += float(vpp_any)
                    any_part = True
            if eligs:
                vsh_any = p.get('sh_def')
                if vsh_any is not None:
                    total_all += float(vsh_any)
                    any_part = True
            if any_part:
                p['all_total'] = total_all
                p['all_total_pct'] = _pct(season_int, 'all_total', total_all)
                p['all_total_z'] = _z(season_int, 'all_total', total_all)
        except Exception:
            pass
        points.append(p)

    # Global scale across seasons (league min/max per season, then overall min/max)
    def _minmax_over_seasons(key: str) -> Dict[str, Any]:
        mins: List[float] = []
        maxs: List[float] = []
        for season_int, s in scale.items():
            mm = (s or {}).get(key) or {}
            vmin = mm.get('min')
            vmax = mm.get('max')
            if vmin is not None:
                try:
                    mins.append(float(vmin))
                except Exception:
                    pass
            if vmax is not None:
                try:
                    maxs.append(float(vmax))
                except Exception:
                    pass
        return {'min': (min(mins) if mins else None), 'max': (max(maxs) if maxs else None)}

    if strength == 'All':
        league_scale = _minmax_over_seasons('all_total')
    elif strength == '5v5':
        # 5v5 chart uses Off/Def/Total; keep scale wide enough for all three.
        mm1 = _minmax_over_seasons('5v5_off')
        mm2 = _minmax_over_seasons('5v5_def')
        mm3 = _minmax_over_seasons('5v5_total')
        mins = [x for x in [mm1.get('min'), mm2.get('min'), mm3.get('min')] if x is not None]
        maxs = [x for x in [mm1.get('max'), mm2.get('max'), mm3.get('max')] if x is not None]
        league_scale = {'min': (min(mins) if mins else None), 'max': (max(maxs) if maxs else None)}
    elif strength == 'PP':
        league_scale = _minmax_over_seasons('pp_off')
    else:
        league_scale = _minmax_over_seasons('sh_def')

    payload = {
        'playerId': pid,
        'rates': _rt(rates),
        'metric': metric,
        'strength': strength,
        'thresholds': thresholds,
        'seasons': seasons,
        'points': points,
        'scale': league_scale,
    }
    j = jsonify(payload)
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


_TEAM_LOGO_PROXY_CACHE: Dict[str, Tuple[float, bytes]] = {}


def _team_logo_source_url(team_abbrev: str) -> Optional[str]:
    a = (team_abbrev or '').strip().upper()
    if not a:
        return None
    # Prefer Teams.csv mapping when present
    try:
        for row in TEAM_ROWS:
            if (row.get('Team') or '').strip().upper() == a:
                u = (row.get('Logo') or '').strip()
                if u:
                    return u
    except Exception:
        pass
    # Fallback to current NHL assets
    return f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'


def _normalize_svg_dimensions(svg_text: str) -> str:
    """Ensure SVG has width/height when only viewBox is provided.

    Some browsers are picky when drawing SVGs to canvas without intrinsic dimensions.
    """
    try:
        head = svg_text[:2048]
        if 'width=' in head and 'height=' in head:
            return svg_text
        m = re.search(r'viewBox\s*=\s*"\s*[-\d\.]+\s+[-\d\.]+\s+([\d\.]+)\s+([\d\.]+)\s*"', head, re.IGNORECASE)
        if not m:
            return svg_text
        w = m.group(1)
        h = m.group(2)

        def repl(match: re.Match) -> str:
            attrs = match.group(1) or ''
            if 'width=' in attrs.lower() or 'height=' in attrs.lower():
                return match.group(0)
            return f'<svg{attrs} width="{w}" height="{h}">' 

        return re.sub(r'<svg\b([^>]*)>', repl, svg_text, count=1, flags=re.IGNORECASE)
    except Exception:
        return svg_text


@main_bp.route('/api/team-logo/<team_abbrev>.svg')
def api_team_logo_svg(team_abbrev: str):
    """Proxy team SVG logos as same-origin to make canvas rendering reliable."""
    a = (team_abbrev or '').strip().upper()
    if not a or not re.fullmatch(r'[A-Z]{2,4}', a):
        return ('', 404)

    # Cache for 30 days in-process
    ttl_s = 30 * 24 * 3600
    try:
        max_items = max(1, int(os.getenv('TEAM_LOGO_PROXY_CACHE_MAX_ITEMS', '64') or '64'))
    except Exception:
        max_items = 64
    now = time.time()
    try:
        _cache_prune_ttl_and_size(_TEAM_LOGO_PROXY_CACHE, ttl_s=ttl_s, max_items=max_items)
        cached = _cache_get(_TEAM_LOGO_PROXY_CACHE, a, int(ttl_s))
        if cached:
            resp = make_response(cached)
            resp.headers['Content-Type'] = 'image/svg+xml'
            resp.headers['Cache-Control'] = 'public, max-age=86400'
            return resp
    except Exception:
        pass

    src = _team_logo_source_url(a)
    if not src:
        return ('', 404)

    # Whitelist host/path to avoid SSRF from Teams.csv edits
    try:
        from urllib.parse import urlparse
        pu = urlparse(src)
        if pu.scheme not in ('http', 'https') or pu.netloc.lower() not in ('assets.nhle.com',):
            src = f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'
    except Exception:
        src = f'https://assets.nhle.com/logos/nhl/svg/{a}_light.svg'

    try:
        r = requests.get(src, timeout=10)
        if r.status_code != 200 or not (r.content or b''):
            return ('', 404)
        raw = r.content
        try:
            txt = raw.decode('utf-8', errors='replace')
            txt = _normalize_svg_dimensions(txt)
            data = txt.encode('utf-8')
        except Exception:
            data = raw
        try:
            _cache_set_multi_bounded(_TEAM_LOGO_PROXY_CACHE, a, data, ttl_s=ttl_s, max_items=max_items)
        except Exception:
            pass
        resp = make_response(data)
        resp.headers['Content-Type'] = 'image/svg+xml'
        resp.headers['Cache-Control'] = 'public, max-age=86400'
        return resp
    except Exception:
        return ('', 404)


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


def _iter_csv_dict_rows(path: str, *, delimiter: str = ',', encoding: str = 'utf-8-sig') -> Iterator[Dict[str, Any]]:
    """Yield DictReader rows without materializing the full file in memory."""
    try:
        if not path or (not os.path.exists(path)):
            return
        with open(path, 'r', encoding=encoding, newline='') as f:
            rdr = csv.DictReader(f, delimiter=delimiter)
            for row in rdr:
                if isinstance(row, dict):
                    yield row
    except Exception:
        return


def _load_rapm_player_rows_static(player_id: int, season: Optional[int]) -> List[Dict[str, Any]]:
    """Load RAPM rows for a single player from app/static/rapm/rapm.csv (TTL cached)."""
    global _RAPM_PLAYER_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('RAPM_PLAYER_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    try:
        max_items = max(1, int(os.getenv('RAPM_PLAYER_STATIC_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512
    key = (int(player_id), int(season) if season is not None else None)
    now = time.time()
    _cache_prune_ttl_and_size(_RAPM_PLAYER_STATIC_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _RAPM_PLAYER_STATIC_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    path = _static_path('rapm', 'rapm.csv')
    out: List[Dict[str, Any]] = []
    pid_s = str(int(player_id))
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        try:
            if str(r.get('PlayerID') or '').strip() != pid_s:
                continue
            if season is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != int(season):
                        continue
                except Exception:
                    continue
            out.append(r)
        except Exception:
            continue

    _cache_set_multi_bounded(_RAPM_PLAYER_STATIC_CACHE, key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_context_player_rows_static(player_id: int, season: Optional[int]) -> List[Dict[str, Any]]:
    """Load Context rows for a single player from app/static/rapm/context.csv (TTL cached)."""
    global _CONTEXT_PLAYER_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CONTEXT_PLAYER_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    try:
        max_items = max(1, int(os.getenv('CONTEXT_PLAYER_STATIC_CACHE_MAX_ITEMS', '512') or '512'))
    except Exception:
        max_items = 512
    key = (int(player_id), int(season) if season is not None else None)
    now = time.time()
    _cache_prune_ttl_and_size(_CONTEXT_PLAYER_STATIC_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _CONTEXT_PLAYER_STATIC_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    path = _static_path('rapm', 'context.csv')
    out: List[Dict[str, Any]] = []
    pid_s = str(int(player_id))
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        try:
            if str(r.get('PlayerID') or '').strip() != pid_s:
                continue
            if season is not None:
                try:
                    if int(str(r.get('Season') or '').strip()) != int(season):
                        continue
                except Exception:
                    continue
            out.append(r)
        except Exception:
            continue

    _cache_set_multi_bounded(_CONTEXT_PLAYER_STATIC_CACHE, key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _iter_seasonstats_static_rows(*, season: Optional[int] = None, skip_season: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    """Yield SeasonStats rows from app/static/nhl_seasonstats.csv, optionally filtered by season."""
    path = _static_path('nhl_seasonstats.csv')
    season_i = int(season) if season is not None else None
    skip_i = int(skip_season) if skip_season is not None else None
    for r in _iter_csv_dict_rows(path, delimiter=',', encoding='utf-8-sig'):
        if season_i is None and skip_i is None:
            yield r
            continue
        try:
            s = int(str(r.get('Season') or '').strip())
        except Exception:
            continue
        if skip_i is not None and s == skip_i:
            continue
        if season_i is not None and s != season_i:
            continue
        yield r


def _build_seasonstats_agg(
    *,
    scope: str,
    season_int: int,
    season_state: str,
    strength_state: str,
    sheet_id: str,
    worksheet: str,
    sheet_ok: bool,
    sheet_rows: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    """Build (and cache) per-player aggregates for SeasonStats under the requested filters.

    This avoids scanning and materializing the full CSV for every request.
    """
    global _SEASONSTATS_AGG_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    try:
        max_items = max(1, int(os.getenv('SEASONSTATS_AGG_CACHE_MAX_ITEMS', '6') or '6'))
    except Exception:
        max_items = 6

    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'career'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    # When sourcing SeasonStats from Google Sheets (Sheets6), the sheet is overwritten on updates.
    # We stamp a TimestampUTC column in the sheet; include it in the cache key to avoid serving stale
    # aggregates from the in-memory/on-disk cache after a sheet refresh.
    sheet_rev = ''
    try:
        if sheet_ok and sheet_rows:
            sheet_rev = str((sheet_rows[0] or {}).get('TimestampUTC') or '').strip()
    except Exception:
        sheet_rev = ''

    # Cache key includes sheet identity because career scope can splice in Sheets6.
    key = (
        scope_norm,
        int(season_int or 0),
        ss_norm,
        st_norm,
        (sheet_id or '')[:80],
        (worksheet or '')[:80],
        sheet_rev[:40],
    )
    now = time.time()
    _cache_prune_ttl_and_size(_SEASONSTATS_AGG_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SEASONSTATS_AGG_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    # Best-effort on-disk cache (helps on Render where workers may restart).
    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v2').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'seasonstats_agg_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    agg0, pos0 = loaded
                    if isinstance(agg0, dict) and isinstance(pos0, dict):
                        _cache_set_multi_bounded(_SEASONSTATS_AGG_CACHE, key, agg0, pos0, ttl_s=ttl_s, max_items=max_items)
                        return agg0, pos0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        if scope_norm == 'career':
            if sheet_ok and sheet_rows is not None:
                def _it() -> Iterator[Dict[str, Any]]:
                    yield from _iter_seasonstats_static_rows(skip_season=20252026)
                    for rr in sheet_rows or []:
                        if isinstance(rr, dict):
                            yield rr
                return _it()
            return _iter_seasonstats_static_rows()

        # scope == season
        if int(season_int) == 20252026 and sheet_ok and sheet_rows is not None:
            return sheet_rows
        return _iter_seasonstats_static_rows(season=int(season_int))

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    agg: Dict[int, Dict[str, Any]] = {}
    pos_group_by_pid: Dict[int, str] = {}
    gp_max_by_key: Dict[Tuple[int, int, str], int] = {}

    # Stream rows and aggregate.
    for r in _iter_rows():
        try:
            # Sheets-based SeasonStats tabs often omit some columns (e.g. Season) and may vary casing.
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                # Sheets6 can be a single-season tab and may omit Season.
                # For career scope, those rows represent 20252026.
                season_row = 20252026 if scope_norm == 'career' else int(season_int)

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            gp_row = _i(r.get('GP'))
            k = (pid_i, int(season_row), str(ss))
            prev_gp = gp_max_by_key.get(k)
            if prev_gp is None or gp_row > prev_gp:
                gp_max_by_key[k] = gp_row

            if pid_i not in pos_group_by_pid:
                pos_group_by_pid[pid_i] = 'D' if pos.startswith('D') else 'F'

            d = agg.setdefault(pid_i, {
                'GP': 0,
                'TOI': 0.0,
                'iGoals': 0.0,
                'Assists1': 0.0,
                'Assists2': 0.0,
                'iShots': 0.0,
                'iFenwick': 0.0,
                'ixG_S': 0.0,
                'ixG_F': 0.0,
                'ixG_F2': 0.0,
                # on-ice
                'CA': 0.0,
                'CF': 0.0,
                'FA': 0.0,
                'FF': 0.0,
                'SA': 0.0,
                'SF': 0.0,
                'GA': 0.0,
                'GF': 0.0,
                'xGA_S': 0.0,
                'xGF_S': 0.0,
                'xGA_F': 0.0,
                'xGF_F': 0.0,
                'xGA_F2': 0.0,
                'xGF_F2': 0.0,
                # misc
                'PIM_taken': 0.0,
                'PIM_drawn': 0.0,
                'PIM_for': 0.0,
                'PIM_against': 0.0,
                'Hits': 0.0,
                'Takeaways': 0.0,
                'Giveaways': 0.0,
            })

            d['TOI'] = float(d.get('TOI') or 0.0) + _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['iGoals'] = float(d.get('iGoals') or 0.0) + _flt(r.get('iGoals') or _ci_get(r, 'iGoals'))
            d['Assists1'] = float(d.get('Assists1') or 0.0) + _flt(r.get('Assists1') or _ci_get(r, 'Assists1'))
            d['Assists2'] = float(d.get('Assists2') or 0.0) + _flt(r.get('Assists2') or _ci_get(r, 'Assists2'))
            d['iShots'] = float(d.get('iShots') or 0.0) + _flt(r.get('iShots') or _ci_get(r, 'iShots'))
            d['iFenwick'] = float(d.get('iFenwick') or 0.0) + _flt(r.get('iFenwick') or _ci_get(r, 'iFenwick'))
            d['ixG_S'] = float(d.get('ixG_S') or 0.0) + _flt(r.get('ixG_S') or _ci_get(r, 'ixG_S'))
            d['ixG_F'] = float(d.get('ixG_F') or 0.0) + _flt(r.get('ixG_F') or _ci_get(r, 'ixG_F'))
            d['ixG_F2'] = float(d.get('ixG_F2') or 0.0) + _flt(r.get('ixG_F2') or _ci_get(r, 'ixG_F2'))

            d['CA'] = float(d.get('CA') or 0.0) + _flt(r.get('CA') or _ci_get(r, 'CA'))
            d['CF'] = float(d.get('CF') or 0.0) + _flt(r.get('CF') or _ci_get(r, 'CF'))
            d['FA'] = float(d.get('FA') or 0.0) + _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['FF'] = float(d.get('FF') or 0.0) + _flt(r.get('FF') or _ci_get(r, 'FF'))
            d['SA'] = float(d.get('SA') or 0.0) + _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['SF'] = float(d.get('SF') or 0.0) + _flt(r.get('SF') or _ci_get(r, 'SF'))
            d['GA'] = float(d.get('GA') or 0.0) + _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['GF'] = float(d.get('GF') or 0.0) + _flt(r.get('GF') or _ci_get(r, 'GF'))
            d['xGA_S'] = float(d.get('xGA_S') or 0.0) + _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGF_S'] = float(d.get('xGF_S') or 0.0) + _flt(r.get('xGF_S') or _ci_get(r, 'xGF_S'))
            d['xGA_F'] = float(d.get('xGA_F') or 0.0) + _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGF_F'] = float(d.get('xGF_F') or 0.0) + _flt(r.get('xGF_F') or _ci_get(r, 'xGF_F'))
            d['xGA_F2'] = float(d.get('xGA_F2') or 0.0) + _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))
            d['xGF_F2'] = float(d.get('xGF_F2') or 0.0) + _flt(r.get('xGF_F2') or _ci_get(r, 'xGF_F2'))

            d['PIM_taken'] = float(d.get('PIM_taken') or 0.0) + _flt(r.get('PIM_taken') or _ci_get(r, 'PIM_taken'))
            d['PIM_drawn'] = float(d.get('PIM_drawn') or 0.0) + _flt(r.get('PIM_drawn') or _ci_get(r, 'PIM_drawn'))
            d['PIM_for'] = float(d.get('PIM_for') or 0.0) + _flt(r.get('PIM_for') or _ci_get(r, 'PIM_for'))
            d['PIM_against'] = float(d.get('PIM_against') or 0.0) + _flt(r.get('PIM_against') or _ci_get(r, 'PIM_against'))
            d['Hits'] = float(d.get('Hits') or 0.0) + _flt(r.get('Hits') or _ci_get(r, 'Hits'))
            d['Takeaways'] = float(d.get('Takeaways') or 0.0) + _flt(r.get('Takeaways') or _ci_get(r, 'Takeaways'))
            d['Giveaways'] = float(d.get('Giveaways') or 0.0) + _flt(r.get('Giveaways') or _ci_get(r, 'Giveaways'))
        except Exception:
            continue

    gp_sum_by_pid: Dict[int, int] = {}
    for (pid_k, _season_k, _ss_k), gp_k in gp_max_by_key.items():
        gp_sum_by_pid[pid_k] = int(gp_sum_by_pid.get(pid_k, 0) + int(gp_k or 0))
    for pid_k, d in agg.items():
        d['GP'] = int(gp_sum_by_pid.get(pid_k, 0))

    _cache_set_multi_bounded(_SEASONSTATS_AGG_CACHE, key, agg, pos_group_by_pid, ttl_s=ttl_s, max_items=max_items)

    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((agg, pos_group_by_pid), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return agg, pos_group_by_pid


def _build_goalies_career_season_matrix(
    *,
    season_state: str,
    strength_state: str,
    sheet_id: str,
    worksheet: str,
    sheet_ok: bool,
    sheet_rows: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[int, Dict[str, float]]], Dict[int, Tuple[float, float]]]:
    """Build per-goalie per-season aggregates for career calculations.

    Returns:
      - by_pid_season: { playerId: { seasonId: { SA, GA, FA, TOI, xGA_S, xGA_F, xGA_F2 } } }
      - league_sa_ga: { seasonId: (total_sa, total_ga) }
    """
    global _GOALIES_CAREER_MATRIX_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    sheet_rev = ''
    try:
        if sheet_ok and sheet_rows:
            sheet_rev = str((sheet_rows[0] or {}).get('TimestampUTC') or '').strip()
    except Exception:
        sheet_rev = ''

    key = (
        'goalies_career_matrix',
        ss_norm,
        st_norm,
        (sheet_id or '')[:80],
        (worksheet or '')[:80],
        sheet_rev[:40],
    )
    now = time.time()
    cached = _GOALIES_CAREER_MATRIX_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v1').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'goalies_career_matrix_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    a0, b0 = loaded
                    if isinstance(a0, dict) and isinstance(b0, dict):
                        _GOALIES_CAREER_MATRIX_CACHE[key] = (now, a0, b0)
                        return a0, b0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        if sheet_ok and sheet_rows is not None:
            def _it() -> Iterator[Dict[str, Any]]:
                yield from _iter_seasonstats_static_rows(skip_season=20252026)
                for rr in sheet_rows or []:
                    if isinstance(rr, dict):
                        yield rr
            return _it()
        return _iter_seasonstats_static_rows()

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    by_pid_season: Dict[int, Dict[int, Dict[str, float]]] = {}
    league_acc: Dict[int, Dict[str, float]] = {}

    for r in _iter_rows():
        try:
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if not pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                season_row = 20252026

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            pmap = by_pid_season.setdefault(pid_i, {})
            d = pmap.setdefault(int(season_row), {
                'TOI': 0.0,
                'FA': 0.0,
                'SA': 0.0,
                'GA': 0.0,
                'xGA_S': 0.0,
                'xGA_F': 0.0,
                'xGA_F2': 0.0,
            })
            d['TOI'] += _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['FA'] += _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['SA'] += _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['GA'] += _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['xGA_S'] += _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGA_F'] += _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGA_F2'] += _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))

            la = league_acc.setdefault(int(season_row), {'SA': 0.0, 'GA': 0.0})
            la['SA'] += _flt(r.get('SA') or _ci_get(r, 'SA'))
            la['GA'] += _flt(r.get('GA') or _ci_get(r, 'GA'))
        except Exception:
            continue

    league_sa_ga: Dict[int, Tuple[float, float]] = {}
    for s, d in league_acc.items():
        try:
            league_sa_ga[int(s)] = (float(d.get('SA') or 0.0), float(d.get('GA') or 0.0))
        except Exception:
            league_sa_ga[int(s)] = (0.0, 0.0)

    _GOALIES_CAREER_MATRIX_CACHE[key] = (now, by_pid_season, league_sa_ga)
    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((by_pid_season, league_sa_ga), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass

    return by_pid_season, league_sa_ga


def _load_rapm_static_csv() -> List[Dict[str, Any]]:
    """Load app/static/rapm/rapm.csv into memory (TTL cached)."""
    global _RAPM_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('RAPM_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _RAPM_STATIC_CACHE and (now - _RAPM_STATIC_CACHE[0]) < ttl_s:
        return _RAPM_STATIC_CACHE[1]

    path = _static_path('rapm', 'rapm.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if not os.path.exists(path):
            _RAPM_STATIC_CACHE = (now, [])
            return []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        rows = []

    _RAPM_STATIC_CACHE = (now, rows)
    return rows


def _load_context_static_csv() -> List[Dict[str, Any]]:
    """Load app/static/rapm/context.csv into memory (TTL cached)."""
    global _CONTEXT_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CONTEXT_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _CONTEXT_STATIC_CACHE and (now - _CONTEXT_STATIC_CACHE[0]) < ttl_s:
        return _CONTEXT_STATIC_CACHE[1]

    path = _static_path('rapm', 'context.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if not os.path.exists(path):
            _CONTEXT_STATIC_CACHE = (now, [])
            return []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        rows = []

    _CONTEXT_STATIC_CACHE = (now, rows)
    return rows


def _load_seasonstats_static_csv() -> List[Dict[str, Any]]:
    """Load app/static/nhl_seasonstats.csv into memory (TTL cached)."""
    global _SEASONSTATS_STATIC_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_STATIC_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    if _SEASONSTATS_STATIC_CACHE and (now - _SEASONSTATS_STATIC_CACHE[0]) < ttl_s:
        return _SEASONSTATS_STATIC_CACHE[1]

    path = _static_path('nhl_seasonstats.csv')
    rows: List[Dict[str, Any]] = []
    try:
        if not os.path.exists(path):
            _SEASONSTATS_STATIC_CACHE = (now, [])
            return []
        with open(path, 'r', encoding='utf-8', newline='') as f:
            rdr = csv.DictReader(f)
            for row in rdr:
                if isinstance(row, dict):
                    rows.append(row)
    except Exception:
        rows = []

    _SEASONSTATS_STATIC_CACHE = (now, rows)
    return rows


def _load_card_metrics_defs(card: str = 'skaters') -> Dict[str, Any]:
    """Load app/static/card_metrics.csv as card metric definitions.

    Returns:
      {
        'categories': [<category>...],
        'metrics': [
           {
             'id': 'Category|Metric',
             'category': 'Category',
             'metric': 'Metric',
             'name': 'Name',
             'calculation': '...',
             'default': bool,
             'place': 'L1'|'C1'|'R1'|'L2'|'C2'|'R2'|'L3'|'C3'|'R3'|'0',
           },
        ]
      }
    """
    global _CARD_METRICS_DEF_CACHE
    try:
        ttl_s = max(30, int(os.getenv('CARD_METRICS_DEF_CACHE_TTL_SECONDS', '600') or '600'))
    except Exception:
        ttl_s = 600
    now = time.time()
    card_norm = str(card or 'skaters').strip().lower() or 'skaters'

    cached = _CARD_METRICS_DEF_CACHE.get(card_norm)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    path = _static_path('card_metrics.csv')
    out: Dict[str, Any] = {'categories': [], 'metrics': []}
    try:
        if not os.path.exists(path):
            _CARD_METRICS_DEF_CACHE[card_norm] = (now, out)
            return out

        metrics: List[Dict[str, Any]] = []
        cats: List[str] = []
        seen_cat: set[str] = set()
        # Use utf-8-sig to tolerate UTF-8 BOM in headers (common on Windows).
        with open(path, 'r', encoding='utf-8-sig', newline='') as f:
            # card_metrics.csv is sometimes tab-delimited (Excel/Sheets export).
            # Auto-detect delimiter between ';' and '\t' based on the header line.
            try:
                first_line = f.readline()
                delim = '\t' if first_line.count('\t') > first_line.count(';') else ';'
                f.seek(0)
            except Exception:
                delim = ';'
            rdr = csv.DictReader(f, delimiter=delim)
            for row in rdr:
                if not isinstance(row, dict):
                    continue
                card = str(row.get('Card') or '').strip()
                if card and card.lower() != card_norm:
                    continue
                category = str(row.get('Category') or '').strip()
                metric = str(row.get('Metric') or '').strip()
                name = str(row.get('Name') or '').strip()
                calc = str(row.get('Calculation') or '').strip()
                place = str(row.get('Place') or '').strip() or '0'
                default_raw = str(row.get('Default') or '').strip()
                is_default = default_raw in {'1', 'true', 'True', 'YES', 'Yes', 'yes'}
                if not category or not metric:
                    continue

                metric_id = f"{category}|{metric}"
                metrics.append({
                    'id': metric_id,
                    'category': category,
                    'metric': metric,
                    'name': name or metric,
                    'calculation': calc,
                    'default': bool(is_default),
                    'place': place,
                    'link': str(row.get('Link') or row.get('link') or '').strip(),
                    'strengthCode': str(row.get('StrengthCode') or row.get('strengthCode') or '').strip(),
                    'positionCode': str(row.get('PositionCode') or row.get('positionCode') or row.get('') or '').strip(),
                })
                if category not in seen_cat:
                    seen_cat.add(category)
                    cats.append(category)

        out = {'categories': cats, 'metrics': metrics}
    except Exception:
        out = {'categories': [], 'metrics': []}

    _CARD_METRICS_DEF_CACHE[card_norm] = (now, out)
    return out


def _build_goalies_seasonstats_agg(
    *,
    scope: str,
    season_int: int,
    season_state: str,
    strength_state: str,
    sheet_id: str,
    worksheet: str,
    sheet_ok: bool,
    sheet_rows: Optional[List[Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, str]]:
    """Build (and cache) per-goalie aggregates for SeasonStats under the requested filters."""
    global _SEASONSTATS_AGG_CACHE
    try:
        ttl_s = max(30, int(os.getenv('SEASONSTATS_AGG_CACHE_TTL_SECONDS', '1800') or '1800'))
    except Exception:
        ttl_s = 1800

    scope_norm = (scope or 'season').strip().lower()
    if scope_norm not in {'season', 'career'}:
        scope_norm = 'season'
    ss_norm = (season_state or 'regular').strip().lower()
    if ss_norm not in {'regular', 'playoffs', 'all'}:
        ss_norm = 'regular'
    st_norm = (strength_state or '5v5').strip()
    if st_norm not in {'5v5', 'PP', 'SH', 'Other', 'all'}:
        st_norm = '5v5'

    sheet_rev = ''
    try:
        if sheet_ok and sheet_rows:
            sheet_rev = str((sheet_rows[0] or {}).get('TimestampUTC') or '').strip()
    except Exception:
        sheet_rev = ''

    key = (
        'goalies',
        scope_norm,
        int(season_int or 0),
        ss_norm,
        st_norm,
        (sheet_id or '')[:80],
        (worksheet or '')[:80],
        sheet_rev[:40],
    )
    now = time.time()
    cached = _SEASONSTATS_AGG_CACHE.get(key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1], cached[2]

    cache_path = None
    try:
        base = os.getenv('XG_CACHE_DIR')
        if not base:
            base = _disk_cache_base()
        os.makedirs(base, exist_ok=True)
        key_bytes = ('|'.join(map(str, key)) + '|v1').encode('utf-8', errors='ignore')
        h = hashlib.sha1(key_bytes).hexdigest()  # nosec - non-crypto use (filename)
        cache_path = os.path.join(base, f'goalies_seasonstats_agg_{h}.pkl.gz')
        if os.path.exists(cache_path):
            mtime = os.path.getmtime(cache_path)
            if (now - float(mtime)) < float(ttl_s):
                with gzip.open(cache_path, 'rb') as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 2:
                    agg0, pos0 = loaded
                    if isinstance(agg0, dict) and isinstance(pos0, dict):
                        _SEASONSTATS_AGG_CACHE[key] = (now, agg0, pos0)
                        return agg0, pos0
    except Exception:
        cache_path = None

    def _iter_rows() -> Iterable[Dict[str, Any]]:
        if scope_norm == 'career':
            if sheet_ok and sheet_rows is not None:
                def _it() -> Iterator[Dict[str, Any]]:
                    yield from _iter_seasonstats_static_rows(skip_season=20252026)
                    for rr in sheet_rows or []:
                        if isinstance(rr, dict):
                            yield rr
                return _it()
            return _iter_seasonstats_static_rows()

        if int(season_int) == 20252026 and sheet_ok and sheet_rows is not None:
            return sheet_rows
        return _iter_seasonstats_static_rows(season=int(season_int))

    def _flt(v: Any) -> float:
        x = _parse_locale_float(v)
        return float(x) if x is not None else 0.0

    def _i(v: Any) -> int:
        return int(_safe_int(v) or 0)

    agg: Dict[int, Dict[str, Any]] = {}
    pos_group_by_pid: Dict[int, str] = {}
    gp_max_by_key: Dict[Tuple[int, int, str], int] = {}

    for r in _iter_rows():
        try:
            pos = str(
                (r.get('Position') or _ci_get(r, 'Position') or _ci_get(r, 'position') or _ci_get(r, 'positionCode') or _ci_get(r, 'Pos') or '')
            ).strip().upper()
            if not pos.startswith('G'):
                continue

            season_row = None
            try:
                season_row = int(str(r.get('Season') or _ci_get(r, 'Season') or '').strip())
            except Exception:
                season_row = None
            if season_row is None:
                season_row = 20252026 if scope_norm == 'career' else int(season_int)

            ss_raw = str(r.get('SeasonState') or _ci_get(r, 'SeasonState') or _ci_get(r, 'seasonState') or '').strip().lower()
            if ss_raw in {'2', 'reg', 'regular', 'regularseason', 'regular_season'}:
                ss = 'regular'
            elif ss_raw in {'3', 'po', 'playoffs', 'playoff'}:
                ss = 'playoffs'
            else:
                ss = ss_raw or 'regular'

            st = str(r.get('StrengthState') or _ci_get(r, 'StrengthState') or _ci_get(r, 'strengthState') or '').strip() or 'Other'
            if ss_norm != 'all' and ss != ss_norm:
                continue
            if st_norm != 'all' and st != st_norm:
                continue

            pid_i = _i(r.get('PlayerID') or _ci_get(r, 'PlayerID') or _ci_get(r, 'playerId'))
            if pid_i <= 0:
                continue

            gp_row = _i(r.get('GP'))
            k = (pid_i, int(season_row), str(ss))
            prev_gp = gp_max_by_key.get(k)
            if prev_gp is None or gp_row > prev_gp:
                gp_max_by_key[k] = gp_row

            if pid_i not in pos_group_by_pid:
                pos_group_by_pid[pid_i] = 'G'

            d = agg.setdefault(pid_i, {
                'GP': 0,
                'TOI': 0.0,
                'FA': 0.0,
                'SA': 0.0,
                'GA': 0.0,
                'xGA_S': 0.0,
                'xGA_F': 0.0,
                'xGA_F2': 0.0,
            })

            d['TOI'] = float(d.get('TOI') or 0.0) + _flt(r.get('TOI') or _ci_get(r, 'TOI'))
            d['FA'] = float(d.get('FA') or 0.0) + _flt(r.get('FA') or _ci_get(r, 'FA'))
            d['SA'] = float(d.get('SA') or 0.0) + _flt(r.get('SA') or _ci_get(r, 'SA'))
            d['GA'] = float(d.get('GA') or 0.0) + _flt(r.get('GA') or _ci_get(r, 'GA'))
            d['xGA_S'] = float(d.get('xGA_S') or 0.0) + _flt(r.get('xGA_S') or _ci_get(r, 'xGA_S'))
            d['xGA_F'] = float(d.get('xGA_F') or 0.0) + _flt(r.get('xGA_F') or _ci_get(r, 'xGA_F'))
            d['xGA_F2'] = float(d.get('xGA_F2') or 0.0) + _flt(r.get('xGA_F2') or _ci_get(r, 'xGA_F2'))
        except Exception:
            continue

    gp_sum_by_pid: Dict[int, int] = {}
    for (pid_k, _season_k, _ss_k), gp_k in gp_max_by_key.items():
        gp_sum_by_pid[pid_k] = int(gp_sum_by_pid.get(pid_k, 0) + int(gp_k or 0))
    for pid_k, d in agg.items():
        d['GP'] = int(gp_sum_by_pid.get(pid_k, 0))

    _SEASONSTATS_AGG_CACHE[key] = (now, agg, pos_group_by_pid)
    if cache_path:
        try:
            with gzip.open(cache_path, 'wb') as f:
                pickle.dump((agg, pos_group_by_pid), f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            pass
    return agg, pos_group_by_pid


def _edge_game_type(season_state: str) -> int:
    # NHL Edge endpoints use 2=regular, 3=playoffs.
    return 3 if str(season_state).strip().lower() == 'playoffs' else 2


def _edge_strength_code(strength_state: str) -> Optional[str]:
    s = str(strength_state or '').strip()
    if s == '5v5':
        return 'es'
    if s == 'PP':
        return 'pp'
    if s == 'SH':
        return 'pk'
    if s == 'all':
        return 'all'
    return None


def _edge_format_url(example_link: str, player_id: int, season_int: int, game_type: int) -> Optional[str]:
    link = str(example_link or '').strip()
    if not link:
        return None
    if link.startswith('api-web.nhle.com/'):
        link = 'https://' + link
    if link.startswith('http://'):
        link = 'https://' + link[len('http://'):]
    if not link.startswith('https://'):
        return None

    # Replace trailing /<player>/<season>/<gameType> if present.
    try:
        parts = link.rstrip('/').split('/')
        if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit() and parts[-3].isdigit():
            parts[-3] = str(int(player_id))
            parts[-2] = str(int(season_int))
            parts[-1] = str(int(game_type))
            return '/'.join(parts)
    except Exception:
        pass
    return link


def _edge_get_cached_json(url: str) -> Optional[Dict[str, Any]]:
    try:
        ttl_s = max(30, int(os.getenv('EDGE_API_CACHE_TTL_SECONDS', '3600') or '3600'))
    except Exception:
        ttl_s = 3600
    try:
        max_items = max(1, int(os.getenv('EDGE_API_CACHE_MAX_ITEMS', '256') or '256'))
    except Exception:
        max_items = 256
    now = time.time()
    _cache_prune_ttl_and_size(_EDGE_API_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _EDGE_API_CACHE.get(url)
    if cached and (now - cached[0]) < ttl_s:
        try:
            return cached[1]
        except Exception:
            return None

    try:
        r = requests.get(url, timeout=20)
        if r.status_code != 200:
            return None
        j = r.json()
        if not isinstance(j, dict):
            return None
        _cache_set_multi_bounded(_EDGE_API_CACHE, url, j, ttl_s=ttl_s, max_items=max_items)
        return j
    except Exception:
        return None


def _ci_get(d: Dict[str, Any], key: str) -> Any:
    if key in d:
        return d.get(key)
    lk = str(key).lower()
    for k, v in d.items():
        if str(k).lower() == lk:
            return v
    return None


def _edge_pct_to_100(p: Any) -> Optional[float]:
    try:
        if p is None:
            return None
        f = float(p)
        if not math.isfinite(f):
            return None
        # NHL Edge uses 0..1 percentiles
        if 0.0 <= f <= 1.0:
            return 100.0 * f
        # Already 0..100
        if 0.0 <= f <= 100.0:
            return f
        return None
    except Exception:
        return None


def _edge_extract_value_and_pct(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Extract a single metric value + percentile from an NHL Edge JSON payload.

    Supports:
      - dict-of-metrics (e.g. shotSpeedDetails.topShotSpeed)
      - list-of-strength-rows (e.g. zoneTimeDetails with strengthCode)
      - nested dict values with {value|imperial|metric, percentile}
      - scalar values with sibling <metric>Percentile
    """
    # 1) Direct hit in a nested dict of details.
    for v in payload.values():
        if isinstance(v, dict):
            node = _ci_get(v, metric_key)
            if isinstance(node, dict):
                val = _ci_get(node, 'imperial')
                if val is None:
                    val = _ci_get(node, 'value')
                if val is None:
                    val = _ci_get(node, 'metric')
                pct = _edge_pct_to_100(_ci_get(node, 'percentile'))
                try:
                    out_val = float(val) if val is not None else None
                except Exception:
                    out_val = None
                if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                    out_val = 100.0 * out_val
                return (out_val, pct)
            if node is not None and not isinstance(node, (dict, list)):
                try:
                    out_val = float(node)
                    if str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                        out_val = 100.0 * out_val
                    return (out_val, _edge_pct_to_100(_ci_get(v, f'{metric_key}Percentile')))
                except Exception:
                    return (None, _edge_pct_to_100(_ci_get(v, f'{metric_key}Percentile')))

    # 2) Strength-split list.
    rows: Optional[List[Dict[str, Any]]] = None
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(str(k).lower() == 'strengthcode' for k in v[0].keys()):
            rows = v  # type: ignore[assignment]
            break
    if rows:
        wanted = strength_code
        row = None
        if wanted:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == str(wanted).lower():
                    row = rr
                    break
        if row is None:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == 'all':
                    row = rr
                    break
        if row is None:
            row = rows[0]

        # Scalar metric with separate percentile key
        val0 = _ci_get(row, metric_key)
        pct_raw = _ci_get(row, f'{metric_key}Percentile')
        if pct_raw is None and str(metric_key).endswith('Pctg'):
            pct_raw = _ci_get(row, f"{str(metric_key)[:-4]}Percentile")
        pct0 = _edge_pct_to_100(pct_raw)
        if isinstance(val0, dict):
            val = _ci_get(val0, 'imperial')
            if val is None:
                val = _ci_get(val0, 'value')
            if val is None:
                val = _ci_get(val0, 'metric')
            pct = _edge_pct_to_100(_ci_get(val0, 'percentile'))
            try:
                out_val = float(val) if val is not None else None
            except Exception:
                out_val = None
            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                out_val = 100.0 * out_val
            return (out_val, pct)
        try:
            out_val = float(val0) if val0 is not None else None
            if out_val is not None and str(metric_key).endswith('Pctg') and 0.0 <= out_val <= 1.0:
                out_val = 100.0 * out_val
            return (out_val, pct0)
        except Exception:
            return (None, pct0)

    return (None, None)


def _edge_extract_value_pct_avg(payload: Dict[str, Any], metric_key: str, strength_code: Optional[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Extract value + percentile + NHL/league average from an NHL Edge JSON payload.

    Many NHL Edge endpoints include league averages in fields like `<metricBase>LeagueAvg`.
    Example (zone time): `offensiveZonePctg` + `offensiveZoneLeagueAvg`.
    """
    val, pct = _edge_extract_value_and_pct(payload, metric_key, strength_code)

    def _coerce_avg(x: Any) -> Optional[float]:
        try:
            if x is None:
                return None
            f = float(x)
            if not math.isfinite(f):
                return None
            if str(metric_key).endswith('Pctg') and 0.0 <= f <= 1.0:
                return 100.0 * f
            return f
        except Exception:
            return None

    # Strength-split list rows (e.g., zoneTimeDetails).
    rows: Optional[List[Dict[str, Any]]] = None
    for v in payload.values():
        if isinstance(v, list) and v and isinstance(v[0], dict) and any(str(k).lower() == 'strengthcode' for k in v[0].keys()):
            rows = v  # type: ignore[assignment]
            break
    if rows:
        wanted = strength_code
        row = None
        if wanted:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == str(wanted).lower():
                    row = rr
                    break
        if row is None:
            for rr in rows:
                if str(rr.get('strengthCode') or '').lower() == 'all':
                    row = rr
                    break
        if row is None:
            row = rows[0]

        mk = str(metric_key)
        base = mk[:-4] if mk.endswith('Pctg') else mk
        avg_raw = (
            _ci_get(row, f'{mk}LeagueAvg')
            or _ci_get(row, f'{base}LeagueAvg')
            or _ci_get(row, f'{mk}LeagueAverage')
            or _ci_get(row, f'{base}LeagueAverage')
        )
        return (val, pct, _coerce_avg(avg_raw))

    # Nested dict-of-metrics nodes sometimes contain avg-like fields.
    for v in payload.values():
        if isinstance(v, dict):
            node = _ci_get(v, metric_key)
            if isinstance(node, dict):
                avg_raw = (
                    _ci_get(node, 'leagueAvg')
                    or _ci_get(node, 'leagueAverage')
                    or _ci_get(node, 'nhlAvg')
                    or _ci_get(node, 'nhlAverage')
                )
                return (val, pct, _coerce_avg(avg_raw))

    return (val, pct, None)


@main_bp.route('/api/skaters/card/defs')
def api_skaters_card_defs():
    """Return the available Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('skaters')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/goalies/card/defs')
def api_goalies_card_defs():
    """Return the available Goalie Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('goalies')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


@main_bp.route('/api/teams/card/defs')
def api_teams_card_defs():
    """Return the available Team Card categories/metrics (from app/static/card_metrics.csv)."""
    defs = _load_card_metrics_defs('teams')
    j = jsonify(defs)
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _parse_locale_float(v: Any) -> Optional[float]:
    """Parse numbers that may use either decimal comma or decimal dot.

    Handles e.g. '1.234,56' (DK) and '1,234.56' (US).
    Returns None if not parseable.
    """
    try:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        s = str(v).strip()
        if not s:
            return None
        s = s.replace('\u00a0', ' ').replace(' ', '')
        last_dot = s.rfind('.')
        last_comma = s.rfind(',')
        if last_dot != -1 and last_comma != -1:
            if last_comma > last_dot:
                # DK style: 1.234,56
                s = s.replace('.', '').replace(',', '.')
            else:
                # US style: 1,234.56
                s = s.replace(',', '')
        elif last_comma != -1 and last_dot == -1:
            s = s.replace(',', '.')
        return float(s)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    try:
        if v is None:
            return None
        if isinstance(v, bool):
            return None
        if isinstance(v, int):
            return v
        s = str(v).strip()
        if not s:
            return None
        return int(float(s))
    except Exception:
        return None


def _extract_name(obj: Any) -> str:
    try:
        if not isinstance(obj, dict):
            return ''
        fn = obj.get('firstName')
        ln = obj.get('lastName')
        if fn is not None and ln is not None:
            if isinstance(fn, dict):
                fn = fn.get('default') or (next(iter(fn.values())) if fn else '')
            if isinstance(ln, dict):
                ln = ln.get('default') or (next(iter(ln.values())) if ln else '')
            name = f"{fn or ''} {ln or ''}".strip()
            if name:
                return name
        for k in ('fullName', 'name'):
            val = obj.get(k)
            if isinstance(val, str) and val.strip():
                return val.strip()
        person = obj.get('person')
        if isinstance(person, dict):
            val = person.get('fullName')
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ''
    except Exception:
        return ''


def _extract_pos(obj: Any) -> str:
    try:
        if not isinstance(obj, dict):
            return ''
        for k in ('positionCode', 'pos', 'position', 'primaryPosition'):
            val = obj.get(k)
            if isinstance(val, dict):
                val = val.get('abbrev') or val.get('type')
            if isinstance(val, str) and val.strip():
                s = val.strip().upper()
                if s.startswith('G'):
                    return 'G'
                if s.startswith('D'):
                    return 'D'
                return 'F'
        return ''
    except Exception:
        return ''


def _load_skater_bios_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map from NHL stats skater bios for a season.

    Uses: https://api.nhle.com/stats/rest/en/skater/bios
    This provides currentTeamAbbrev and positionCode for skaters.
    """
    global _SKATER_BIOS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('SKATER_BIOS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('SKATER_BIOS_CACHE_MAX_ITEMS', '4') or '4'))
    except Exception:
        max_items = 4

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    now = time.time()
    _cache_prune_ttl_and_size(_SKATER_BIOS_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SKATER_BIOS_CACHE.get(season_i)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    out: Dict[int, Dict[str, str]] = {}
    # NOTE: The endpoint returns 500 if cayenneExp is omitted.
    url = f'https://api.nhle.com/stats/rest/en/skater/bios?limit=-1&start=0&cayenneExp=seasonId={season_i}'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            data = r.json() if r.content else {}
            rows = data.get('data') if isinstance(data, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId'))
                    if not pid or pid <= 0:
                        continue
                    team = str(row.get('currentTeamAbbrev') or '').strip().upper()
                    name = str(row.get('skaterFullName') or '').strip()
                    pos_raw = str(row.get('positionCode') or '').strip().upper()
                    pos = 'D' if pos_raw.startswith('D') else 'F'
                    if pid not in out or (name and not (out.get(pid) or {}).get('name')):
                        out[pid] = {
                            'playerId': str(pid),
                            'name': name,
                            'team': team,
                            'position': pos,
                            'positionCode': pos_raw,
                        }
    except Exception:
        out = {}

    _cache_set_multi_bounded(_SKATER_BIOS_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_goalie_bios_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map from NHL stats goalie bios for a season.

    Uses: https://api.nhle.com/stats/rest/en/goalie/bios
    This provides currentTeamAbbrev and positionCode for goalies.
    """
    global _SKATER_BIOS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('SKATER_BIOS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('SKATER_BIOS_CACHE_MAX_ITEMS', '4') or '4'))
    except Exception:
        max_items = 4

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    cache_key = -int(season_i or 0)
    now = time.time()
    _cache_prune_ttl_and_size(_SKATER_BIOS_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _SKATER_BIOS_CACHE.get(cache_key)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    out: Dict[int, Dict[str, str]] = {}
    url = f'https://api.nhle.com/stats/rest/en/goalie/bios?limit=-1&start=0&cayenneExp=seasonId={season_i}'
    try:
        r = requests.get(url, timeout=20, allow_redirects=True)
        if r.status_code == 200:
            data = r.json() if r.content else {}
            rows = data.get('data') if isinstance(data, dict) else None
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    pid = _safe_int(row.get('playerId'))
                    if not pid or pid <= 0:
                        continue
                    team = str(row.get('currentTeamAbbrev') or '').strip().upper()
                    name = str(row.get('goalieFullName') or row.get('playerFullName') or row.get('skaterFullName') or '').strip()
                    pos_raw = str(row.get('positionCode') or 'G').strip().upper()
                    out[pid] = {
                        'playerId': str(pid),
                        'name': name,
                        'team': team,
                        'position': 'G',
                        'positionCode': pos_raw,
                    }
    except Exception:
        out = {}

    _cache_set_multi_bounded(_SKATER_BIOS_CACHE, cache_key, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _load_all_rosters_cached() -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map by fetching current rosters for all teams.

    Cached with TTL to avoid hammering upstream.
    """
    global _ALL_ROSTERS_CACHE
    try:
        ttl_s = max(60, int(os.getenv('ALL_ROSTERS_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600
    now = time.time()
    if _ALL_ROSTERS_CACHE and (now - _ALL_ROSTERS_CACHE[0]) < ttl_s:
        return _ALL_ROSTERS_CACHE[1]

    # Primary source: NHL stats skater bios for the current season.
    # (We intentionally do NOT use app/static/lineups_all.json here; it may be stale.)
    try:
        season_i = int(current_season_id())
    except Exception:
        season_i = 0
    out = _load_skater_bios_season_cached(season_i)

    # Merge goalie bios (current season)
    try:
        g = _load_goalie_bios_season_cached(season_i) or {}
        if g:
            out = {**out, **g}
    except Exception:
        pass

    _ALL_ROSTERS_CACHE = (now, out)
    return out


# Historical roster cache: {seasonId -> (timestamp, playerId->info)}
_ALL_ROSTERS_BY_SEASON_CACHE: Dict[int, Tuple[float, Dict[int, Dict[str, str]]]] = {}


def _load_all_rosters_for_season_cached(season_id: int) -> Dict[int, Dict[str, str]]:
    """Build a playerId->info map for a specific season.

    For the current season, prefer NHL stats skater bios.
    For other seasons, best-effort fetch team rosters from api-web.nhle.com to recover
    a season-appropriate team abbreviation.
    """
    global _ALL_ROSTERS_BY_SEASON_CACHE
    try:
        ttl_s = max(60, int(os.getenv('ALL_ROSTERS_BY_SEASON_CACHE_TTL_SECONDS', '21600') or '21600'))
    except Exception:
        ttl_s = 21600

    try:
        max_items = max(1, int(os.getenv('ALL_ROSTERS_BY_SEASON_CACHE_MAX_ITEMS', '6') or '6'))
    except Exception:
        max_items = 6

    try:
        season_i = int(season_id)
    except Exception:
        season_i = 0
    if season_i <= 0:
        try:
            season_i = int(current_season_id())
        except Exception:
            season_i = 0

    try:
        current_i = int(current_season_id())
    except Exception:
        current_i = 0

    now = time.time()
    _cache_prune_ttl_and_size(_ALL_ROSTERS_BY_SEASON_CACHE, ttl_s=ttl_s, max_items=max_items)
    cached = _ALL_ROSTERS_BY_SEASON_CACHE.get(season_i)
    if cached and (now - cached[0]) < ttl_s:
        return cached[1]

    # Current season: bios has fast, complete coverage.
    if current_i and season_i == current_i:
        out = _load_skater_bios_season_cached(season_i)
        try:
            g = _load_goalie_bios_season_cached(season_i) or {}
            if g:
                out = {**out, **g}
        except Exception:
            pass
        _cache_set_multi_bounded(_ALL_ROSTERS_BY_SEASON_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
        return out

    out: Dict[int, Dict[str, str]] = {}
    # Historical season: build mapping from per-team roster endpoints.
    teams = [str(r.get('Team') or '').strip().upper() for r in (TEAM_ROWS or []) if isinstance(r, dict)]
    teams = [t for t in teams if t]
    for team in teams:
        url = f'https://api-web.nhle.com/v1/roster/{team}/{season_i}'
        try:
            r = requests.get(url, timeout=20, allow_redirects=True)
            if r.status_code != 200:
                continue
            data = r.json() if r.content else {}
            if not isinstance(data, dict):
                continue
            forwards = data.get('forwards') or []
            defensemen = data.get('defensemen') or []
            goalies = data.get('goalies') or []
            for p in list(forwards) + list(defensemen) + list(goalies):
                if not isinstance(p, dict):
                    continue
                pid = _safe_int(p.get('id') or p.get('playerId'))
                if not pid or pid <= 0:
                    continue
                fn = (p.get('firstName') or {}).get('default') if isinstance(p.get('firstName'), dict) else (p.get('firstName') or '')
                ln = (p.get('lastName') or {}).get('default') if isinstance(p.get('lastName'), dict) else (p.get('lastName') or '')
                name = (str(fn).strip() + ' ' + str(ln).strip()).strip() or str(pid)
                pos_raw = str(p.get('positionCode') or p.get('position') or '').strip().upper()
                pos = 'G' if pos_raw.startswith('G') else ('D' if pos_raw.startswith('D') else 'F')
                out[int(pid)] = {
                    'playerId': str(int(pid)),
                    'name': name,
                    'team': team,
                    'position': pos,
                    'positionCode': pos_raw,
                }
        except Exception:
            continue

    # Fill gaps with bios names (team may be currentTeamAbbrev; keep historical roster team when present).
    try:
        bios = _load_skater_bios_season_cached(season_i) or {}
        for pid_s, info in bios.items():
            try:
                pid_i = int(pid_s)
            except Exception:
                continue
            if pid_i in out:
                if (not out[pid_i].get('name')) and info.get('name'):
                    out[pid_i]['name'] = str(info.get('name') or '')
                continue
            try:
                info_d = info if isinstance(info, dict) else {}
                name_s = str(info_d.get('name') or '').strip() or str(pid_i)
                team_s = str(info_d.get('team') or '').strip().upper()
                pos_raw = str(info_d.get('positionCode') or info_d.get('position') or '').strip().upper()
                pos = 'G' if pos_raw.startswith('G') else ('D' if pos_raw.startswith('D') else ('F' if pos_raw else ''))
                out[pid_i] = {
                    'playerId': str(pid_i),
                    'name': name_s,
                    'team': team_s,
                    'position': pos,
                    'positionCode': pos_raw,
                }
            except Exception:
                out[pid_i] = {
                    'playerId': str(pid_i),
                    'name': str(pid_i),
                    'team': '',
                    'position': '',
                    'positionCode': '',
                }
    except Exception:
        pass

    _cache_set_multi_bounded(_ALL_ROSTERS_BY_SEASON_CACHE, season_i, out, ttl_s=ttl_s, max_items=max_items)
    return out


def _parse_proj_row(row: Dict[str, Any]) -> Dict[str, Any]:
    # player_projections.csv columns: PlayerID,Position,Game_No,Age,Rookie,EVO,EVD,PP,SH,GSAx
    pid = _safe_int(row.get('PlayerID') or row.get('playerId') or row.get('player_id') or row.get('id'))
    pos = str(row.get('Position') or row.get('position') or '').strip().upper()[:1]
    gp = _safe_int(row.get('Game_No') or row.get('GP') or row.get('games') or row.get('gamesPlayed'))
    age = _parse_locale_float(row.get('Age'))
    rookie = _parse_locale_float(row.get('Rookie'))
    evo = _parse_locale_float(row.get('EVO'))
    evd = _parse_locale_float(row.get('EVD'))
    pp = _parse_locale_float(row.get('PP'))
    sh = _parse_locale_float(row.get('SH'))
    gsax = _parse_locale_float(row.get('GSAx') or row.get('gsax') or row.get('Gsax') or row.get('GsaX'))
    # Total excludes GSAx per spec
    total = sum([(age or 0.0), (rookie or 0.0), (evo or 0.0), (evd or 0.0), (pp or 0.0), (sh or 0.0)])
    return {
        'playerId': pid,
        'position': pos,
        'gp': gp,
        'Age': age,
        'Rookie': rookie,
        'EVO': evo,
        'EVD': evd,
        'PP': pp,
        'SH': sh,
        'GSAx': gsax,
        'total': total,
    }


@main_bp.route('/api/player-projections/league')
def api_player_projections_league():
    """League-wide player projections.

    Query params:
      team=EDM (optional)
      include_goalies=1 (optional)
    """
    team = str(request.args.get('team') or '').strip().upper()
    include_goalies = str(request.args.get('include_goalies') or '').strip().lower() in ('1', 'true', 'yes', 'y')

    proj_map = _load_player_projections_cached()
    roster_map = _load_all_rosters_cached()

    out: List[Dict[str, Any]] = []
    for pid, raw in (proj_map or {}).items():
        try:
            row = _parse_proj_row(raw)
            if not row.get('playerId'):
                continue
            pos = str(row.get('position') or '').upper()
            if (not include_goalies) and pos.startswith('G'):
                continue  # skaters only by default
            info = roster_map.get(int(row['playerId'])) or {}
            t = (info.get('team') or '').upper()
            if team and t and t != team:
                continue
            if team and not t:
                # If filtering by team and we don't know the team, skip.
                continue
            out.append({
                **row,
                'name': info.get('name') or '',
                'team': t,
                # prefer CSV position for skaters, but ensure a fallback
                'position': pos if pos in ('F', 'D', 'G') else (info.get('position') or 'F'),
            })
        except Exception:
            continue

    out.sort(key=lambda r: float(r.get('total') or 0.0), reverse=True)
    j = jsonify({'players': out})
    try:
        j.headers['Cache-Control'] = 'no-store'
    except Exception:
        pass
    return j


def _load_player_projections_cached() -> Dict[int, Dict[str, Any]]:
    """Load player projections from Google Sheets (with CSV fallback) into memory (TTL cached)."""
    return _load_player_projections_from_sheets()

_PLAYER_PROJECTIONS_SHEETS_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None

def _load_player_projections_from_sheets() -> Dict[int, Dict[str, Any]]:
    """Load player projections from Google Sheets (Sheets3) into memory."""
    global _PLAYER_PROJECTIONS_SHEETS_CACHE
    try:
        ttl_s = max(30, int(os.getenv('PLAYER_PROJECTIONS_CACHE_TTL_SECONDS', '300') or '300'))
    except Exception:
        ttl_s = 300
    now = time.time()
    if _PLAYER_PROJECTIONS_SHEETS_CACHE and (now - _PLAYER_PROJECTIONS_SHEETS_CACHE[0]) < ttl_s:
        return _PLAYER_PROJECTIONS_SHEETS_CACHE[1]
    
    sheet_id = (os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('PROJECTIONS_WORKSHEET') or 'Sheets3').strip()
    if not sheet_id:
        # Fallback to CSV if no sheet configured
        data = _load_player_projections_csv()
        _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, data)
        return data
    
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        # Fallback to CSV if gspread not available
        data = _load_player_projections_csv()
        _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, data)
        return data
    
    try:
        info = _load_google_service_account_info_from_env()
        scopes = [
            'https://www.googleapis.com/auth/spreadsheets.readonly',
            'https://www.googleapis.com/auth/drive.readonly',
        ]
        creds = Credentials.from_service_account_info(info, scopes=scopes)
        gc = gspread.authorize(creds)
        sh = gc.open_by_key(sheet_id)
        ws = sh.worksheet(worksheet)
        # Use numericise_ignore to preserve comma decimal separators (like RAPM does)
        try:
            rows = ws.get_all_records(numericise_ignore=['all']) or []
        except Exception:
            rows = ws.get_all_records() or []
        
        # Build map keyed by PlayerID
        # Keep values as strings so _parse_locale_float can handle comma decimal separators
        out: Dict[int, Dict[str, Any]] = {}
        for r in rows:
            try:
                pid_raw = r.get('PlayerID') or r.get('playerId') or r.get('player_id')
                if pid_raw is None:
                    continue
                pid = _safe_int(pid_raw)
                if not pid or pid <= 0:
                    pid = _safe_int(str(pid_raw).replace(' ', '').replace('\u00a0', '').strip())
                if not pid or pid <= 0:
                    continue
                # Store raw row with string values preserved
                out[pid] = r
            except Exception:
                continue
        
        _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, out)
        return out
    except Exception:
        # Fallback to CSV on error
        data = _load_player_projections_csv()
        _PLAYER_PROJECTIONS_SHEETS_CACHE = (now, data)
        return data

_LINEUPS_ALL_CACHE: Optional[Tuple[float, Dict[str, Any]]] = None

def _load_google_service_account_info_from_env() -> Dict[str, Any]:
    """Load Google service account JSON.

    Supports (in priority order):
      - GOOGLE_SERVICE_ACCOUNT_JSON_PATH: path to a JSON key file
      - GOOGLE_SERVICE_ACCOUNT_JSON_B64: base64-encoded JSON string
      - GOOGLE_SERVICE_ACCOUNT_JSON: raw JSON string
    """
    raw: Optional[str] = None
    path = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON_PATH')
    if path:
        try:
            p = str(path).strip()
            if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
                p = p[1:-1]
            p = os.path.expandvars(os.path.expanduser(p))
            with open(p, 'r', encoding='utf-8') as f:
                raw = f.read()
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON_PATH: {e}")
    raw_b64 = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON_B64')
    if raw is None and raw_b64:
        try:
            import base64
            s = str(raw_b64).strip()
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1]
            s = ''.join(s.split())
            pad = (-len(s)) % 4
            if pad:
                s = s + ('=' * pad)
            try:
                raw = base64.b64decode(s.encode('utf-8'), validate=False).decode('utf-8')
            except Exception:
                raw = base64.urlsafe_b64decode(s.encode('utf-8')).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON_B64: {e}")
    if raw is None:
        raw = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    if not raw:
        raise RuntimeError(
            'Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON_PATH, GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_SERVICE_ACCOUNT_JSON.'
        )
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid Google service account JSON: {e}")

def _load_lineups_all() -> Dict[str, Any]:
    """Load lineups from Google Sheets.

    Expected columns in the worksheet:
      Timestamp, Team, Unit, Pos, PlayerName, playerId

    Config:
      - LINEUPS_SHEET_ID (defaults to PROJECTIONS_SHEET_ID)
      - LINEUPS_WORKSHEET (default 'Sheets2')
      - LINEUPS_SHEET_CACHE_TTL_SECONDS (default 300)
    """
    global _LINEUPS_ALL_CACHE
    ttl_s = int(os.getenv('LINEUPS_SHEET_CACHE_TTL_SECONDS', '300') or '300')
    now = time.time()
    if _LINEUPS_ALL_CACHE and (now - _LINEUPS_ALL_CACHE[0]) < max(1, ttl_s):
        return _LINEUPS_ALL_CACHE[1]

    sheet_id = (os.getenv('LINEUPS_SHEET_ID') or os.getenv('PROJECTIONS_SHEET_ID') or '').strip()
    worksheet = (os.getenv('LINEUPS_WORKSHEET') or 'Sheets2').strip()
    if not sheet_id:
        # No sheet configured; return empty rather than reading from local JSON.
        _LINEUPS_ALL_CACHE = (now, {})
        return {}

    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        _LINEUPS_ALL_CACHE = (now, {})
        return {}

    info = _load_google_service_account_info_from_env()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    rows = ws.get_all_records() or []

    out: Dict[str, Any] = {}
    latest_ts_by_team: Dict[str, str] = {}

    def _ensure_team(t: str) -> Dict[str, Any]:
        if t not in out:
            out[t] = {'team': t, 'forwards': [], 'defense': [], 'goalies': [], 'generated_at': None}
        return out[t]

    for r in rows:
        try:
            team = str(r.get('Team') or '').strip().upper()
            if not team:
                continue
            unit = str(r.get('Unit') or '').strip().upper()
            pos = str(r.get('Pos') or '').strip().upper()[:1]
            name = str(r.get('PlayerName') or r.get('Name') or '').strip()
            pid_raw = r.get('playerId') if 'playerId' in r else r.get('PlayerId')
            pid = int(str(pid_raw).strip())
            ts = str(r.get('Timestamp') or '').strip()
        except Exception:
            continue

        rec = {'name': name, 'playerId': pid, 'unit': unit, 'pos': ('G' if unit.startswith('G') else pos)}

        bucket = 'forwards'
        if rec['pos'] == 'G' or unit.startswith('G'):
            bucket = 'goalies'
            rec['pos'] = 'G'
        elif rec['pos'] == 'D' or unit.startswith('LD') or unit.startswith('RD'):
            bucket = 'defense'
            rec['pos'] = 'D'
        else:
            rec['pos'] = 'F'

        tnode = _ensure_team(team)
        tnode[bucket].append(rec)
        if ts:
            latest_ts_by_team[team] = max(latest_ts_by_team.get(team, ''), ts)

    # Set generated_at per team
    for t, node in out.items():
        node['generated_at'] = latest_ts_by_team.get(t)

    _LINEUPS_ALL_CACHE = (now, out)
    return out

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


_STARTED_GAME_SHEET_CACHE: Optional[Tuple[float, Dict[int, Dict[str, Any]]]] = None

def _load_started_game_overrides_from_sheet(sheet_id: str, worksheet: str) -> Dict[int, Dict[str, Any]]:
    """Load latest ML (moneyline) and Win_Prop overrides for started games from Google Sheets.

    This is used only for games already started. If Win_Prop is empty, the client should
    fall back to the calculated probability.

    Expected (best-effort) columns in worksheet (supports multiple naming styles):
      - GameID / gameId / id
      - ML_Away / ML_Home (or OddsAway/OddsHome)
      - Win_Prop_Away / Win_Prop_Home (or WinAway/WinHome)
      - OR rows per team with: GameID + Team + ML + Win_Prop

    Config:
      - STARTED_OVERRIDES_SHEET_CACHE_TTL_SECONDS (default 30)
    """
    global _STARTED_GAME_SHEET_CACHE
    try:
        ttl_s = max(1, int(os.getenv('STARTED_OVERRIDES_SHEET_CACHE_TTL_SECONDS', '30') or '30'))
    except Exception:
        ttl_s = 30
    now = time.time()
    if _STARTED_GAME_SHEET_CACHE and (now - _STARTED_GAME_SHEET_CACHE[0]) < ttl_s:
        return _STARTED_GAME_SHEET_CACHE[1]

    if not sheet_id or not worksheet:
        _STARTED_GAME_SHEET_CACHE = (now, {})
        return {}

    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception:
        _STARTED_GAME_SHEET_CACHE = (now, {})
        return {}

    info = _load_google_service_account_info_from_env()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets.readonly',
        'https://www.googleapis.com/auth/drive.readonly',
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(sheet_id)
    ws = sh.worksheet(worksheet)
    rows = ws.get_all_records() or []

    def norm_row(r: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in (r or {}).items():
            try:
                kk = str(k).strip().lower()
            except Exception:
                continue
            out[kk] = v
        return out

    def first_val(rn: Dict[str, Any], keys: List[str]) -> Any:
        for k in keys:
            if k in rn:
                v = rn.get(k)
                if v is None:
                    continue
                s = str(v).strip()
                if s == '':
                    continue
                return v
        return None

    def parse_game_id(rn: Dict[str, Any]) -> Optional[int]:
        v = first_val(rn, ['gameid', 'game_id', 'gamepk', 'game', 'gameid#', 'id', 'gameid '])
        if v is None:
            return None
        try:
            return int(str(v).strip())
        except Exception:
            return None

    def parse_prob_pct(v: Any) -> Optional[float]:
        if v is None:
            return None
        try:
            s = str(v).strip()
            if s == '':
                return None
            x = float(s)
            if not math.isfinite(x):
                return None
            # If it's 0..1, treat as probability; if 1..100 treat as percent.
            if 0.0 <= x <= 1.0:
                return x * 100.0
            if 1.0 < x <= 100.0:
                return x
            return None
        except Exception:
            return None

    def parse_ts(rn: Dict[str, Any]) -> str:
        # Use a timestamp-ish field if present; else empty string (row order will win).
        v = first_val(rn, ['timestamputc', 'timestamp', 'updated', 'updatedutc', 'lastupdated', 'time'])
        return str(v).strip() if v is not None else ''

    def prefer_newer(prev: Optional[Dict[str, Any]], ts: str, row_idx: int) -> bool:
        """Return True if (ts,row_idx) should replace prev based on recency."""
        if prev is None:
            return True
        prev_ts = str(prev.get('_ts') or '')
        prev_row = int(prev.get('_row', -1) or -1)
        # Prefer lexicographically-later timestamp (ISO-8601 sorts correctly)
        if ts and ts > prev_ts:
            return True
        if ts and ts == prev_ts and row_idx >= prev_row:
            return True
        # If no timestamps, prefer later row
        if not ts and not prev_ts and row_idx >= prev_row:
            return True
        return False

    # Two supported shapes:
    #  1) One row per game with Away/Home ML and Win_Prop
    #  2) One row per team with Team + ML + Win_Prop (we later map to away/home)
    by_game: Dict[int, Dict[str, Any]] = {}
    by_game_team: Dict[int, Dict[str, Dict[str, Any]]] = {}

    for idx, r in enumerate(rows):
        rn = norm_row(r)
        gid = parse_game_id(rn)
        if gid is None:
            continue
        ts = parse_ts(rn)
        # Shape 1
        away_ml = first_val(rn, ['ml_away', 'away_ml', 'oddsaway', 'odds_away', 'moneyline_away', 'awaymoneyline'])
        home_ml = first_val(rn, ['ml_home', 'home_ml', 'oddshome', 'odds_home', 'moneyline_home', 'homemoneyline'])
        win_away = first_val(rn, ['win_prop_away', 'away_win_prop', 'winaway', 'win_away', 'winprobaway', 'win_prob_away'])
        win_home = first_val(rn, ['win_prop_home', 'home_win_prop', 'winhome', 'win_home', 'winprobhome', 'win_prob_home'])
        if away_ml is not None or home_ml is not None or win_away is not None or win_home is not None:
            prev = by_game.get(gid)
            prev_ts = str(prev.get('_ts') or '') if prev else ''
            # prefer lexicographically-later timestamp; if none, prefer later row
            choose = (ts and ts >= prev_ts) or (not prev_ts and not ts and prev is not None and idx >= int(prev.get('_row', -1)))
            if prev is None or choose:
                by_game[gid] = {
                    '_ts': ts,
                    '_row': idx,
                    'oddsAway': away_ml,
                    'oddsHome': home_ml,
                    'winAwayPct': parse_prob_pct(win_away),
                    'winHomePct': parse_prob_pct(win_home),
                }
            continue

        # Shape 2 (team row)
        team = first_val(rn, ['team (abbrev)', 'team_abbrev', 'teamabbrev', 'team', 'abbrev', 'club', 'clubabbrev'])
        if team is None:
            continue
        team_ab = str(team).strip().upper()
        ml = first_val(rn, ['ml', 'moneyline', 'odds', 'price'])
        wp = first_val(rn, ['win_prop', 'winprop', 'win%', 'winpct', 'winprob', 'win_probability'])
        if ml is None and wp is None:
            continue
        team_map = by_game_team.setdefault(gid, {})
        prev_team = team_map.get(team_ab)
        next_team = {
            '_ts': ts,
            '_row': idx,
            'ml': ml,
            'winPct': parse_prob_pct(wp),
        }
        if prefer_newer(prev_team, ts, idx):
            team_map[team_ab] = next_team

    # Build final map (prefer shape 1 when present)
    out: Dict[int, Dict[str, Any]] = {}
    for gid, rec in by_game.items():
        out[gid] = {k: v for k, v in rec.items() if not str(k).startswith('_')}

    # Keep team-row data for any games not covered by shape 1
    for gid, team_map in by_game_team.items():
        if gid in out:
            continue
        out[gid] = {
            '_by_team': team_map,
        }

    _STARTED_GAME_SHEET_CACHE = (now, out)
    return out

def _safe_float(v: Any) -> Optional[float]:
    try:
        if v is None or v == '':
            return None
        return float(v)
    except Exception:
        return None

def _proj_value_for_player(row: Optional[Dict[str, Any]]) -> float:
    """Sum of (Age + Rookie + EVO + EVD + PP + SH + GSAx) for a projections row.
    Non-numeric values are treated as 0. Handles comma decimal separators.
    """
    if not row:
        return 0.0
    def f(k: str) -> float:
        try:
            v = row.get(k)
            if v is None:
                return 0.0
            # Use _parse_locale_float to handle comma decimal separators
            parsed = _parse_locale_float(v)
            return parsed if parsed is not None else 0.0
        except Exception:
            # try case-insensitive
            try:
                for key in row.keys():
                    if str(key).lower() == k.lower():
                        vv = row.get(key)
                        if vv is None:
                            return 0.0
                        parsed = _parse_locale_float(vv)
                        return parsed if parsed is not None else 0.0
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
        try:
            max_items = max(1, int(os.getenv('MODEL_CACHE_MAX_ITEMS', '24') or '24'))
        except Exception:
            max_items = 24
        _dict_set_bounded(_MODEL_CACHE, fname, m, max_items=max_items)
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
            try:
                max_items = max(1, int(os.getenv('BOX_CACHE_MAX_ITEMS', '64') or '64'))
            except Exception:
                max_items = 64
            _cache_prune_ttl_and_size(_BOX_CACHE, ttl_s=ttl, max_items=max_items)
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
        ttl = int(os.getenv('BOX_CACHE_TTL_SECONDS', '600'))
    except Exception:
        ttl = 600
    try:
        max_items = max(1, int(os.getenv('BOX_CACHE_MAX_ITEMS', '64') or '64'))
    except Exception:
        max_items = 64
    try:
        _cache_set_multi_bounded(_BOX_CACHE, int(game_id), data, ttl_s=ttl, max_items=max_items)
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
    try:
        max_items = max(1, int(os.getenv('PBP_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
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
            _cache_prune_ttl_and_size(_PBP_CACHE, ttl_s=std_ttl, max_items=max_items)
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

        # Helper: map season integer like 20142015 to previous, current, next for 3 sliding windows
        def season_prev(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a-1)*10000 + (b-1)
        def season_next(s: int) -> int:
            a = int(str(s)[:4]); b = int(str(s)[4:])
            return (a+1)*10000 + (b+1)

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
                    try:
                        max_items = max(1, int(os.getenv('FEATURE_COLS_CACHE_MAX_ITEMS', '512') or '512'))
                    except Exception:
                        max_items = 512
                    _dict_set_bounded(_FEATURE_COLS_CACHE, key, cols, max_items=max_items)
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
            models = [load_model_file(n) for n in names]
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
                models = [load_model_file(n) for n in names]
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
        _cache_set_multi_bounded(_PBP_CACHE, int(game_id), out_obj, ttl_s=std_ttl, max_items=max_items)
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

    # Cache TTLs
    live_ttl = 5
    std_ttl = int(os.getenv('SHIFTS_CACHE_TTL_SECONDS', '600'))
    try:
        max_items = max(1, int(os.getenv('SHIFTS_CACHE_MAX_ITEMS', '24') or '24'))
    except Exception:
        max_items = 24
    disk_path = _disk_cache_path_shifts(int(game_id))

    # Disk cache first (contains gameState so we can pick live vs std TTL without fetching boxscore)
    if not force:
        try:
            if os.path.exists(disk_path):
                import json
                js = None
                with open(disk_path, 'r', encoding='utf-8') as f:
                    js = json.load(f)
                ts = float((js or {}).get('_cachedAt', 0.0) or 0.0)
                gstate = str((js or {}).get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify({k: v for k, v in (js or {}).items() if not str(k).startswith('_')})
        except Exception:
            pass

    # In-memory cache next (also includes gameState)
    if not force:
        try:
            _cache_prune_ttl_and_size(_SHIFTS_CACHE, ttl_s=std_ttl, max_items=max_items)
            cached = _SHIFTS_CACHE.get(int(game_id))
            if cached:
                ts = float((cached[0] or 0.0))
                payload = cached[1]
                gstate = str((payload or {}).get('gameState') or '').upper()
                ttl = live_ttl if gstate in ('LIVE', 'SCHEDULED', 'PREVIEW', 'INPROGRESS') else std_ttl
                if ts and (time.time() - ts) < ttl:
                    return jsonify(payload)
        except Exception:
            pass

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

    # Fetch boxscore to map to player IDs
    try:
        r = requests.get(f'https://api-web.nhle.com/v1/gamecenter/{game_id}/boxscore', timeout=20)
        if r.status_code != 200:
            return jsonify({'error': 'Failed to fetch boxscore'}), 502
        box = r.json()
    except Exception:
        return jsonify({'error': 'Failed to fetch boxscore'}), 502

    pages = {side: fetch_html(u) for side, u in urls.items()}

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
                    if current_pid is None and not current_name:
                        # If we couldn't resolve the current player header, don't emit anonymous shifts.
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
                    if pid is None and not name_out2:
                        continue
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
                if pid is None and not name_out3:
                    continue
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

        # Compute unique on-ice skater/goalie counts per team for this slice.
        # The HTML shift reports can contain duplicate or overlapping rows; counting rows inflates
        # skater counts (e.g., 10v8). Use unique PlayerID (fallback to Name) per side.
        team_players: Dict[str, Dict[str, set]] = {}
        for rec in active:
            team = str(rec.get('Team') or '')
            pos = str(rec.get('Position') or '').upper()
            pid = rec.get('PlayerID')
            name = str(rec.get('Name') or '').strip()
            key = pid if isinstance(pid, int) else (name if name else None)
            if key is None:
                continue
            tp = team_players.setdefault(team, {'G': set(), 'S': set()})
            if pos == 'G':
                tp['G'].add(key)
            else:
                tp['S'].add(key)

        team_counts_raw: Dict[str, Dict[str, int]] = {}
        team_counts_clamped: Dict[str, Dict[str, int]] = {}
        for t, tp in team_players.items():
            try:
                g_raw = int(len(tp.get('G') or set()))
                s_raw = int(len(tp.get('S') or set()))
            except Exception:
                g_raw = 0
                s_raw = 0

            # Clamp to realistic maxima to avoid noise in downstream calculations.
            g = min(max(g_raw, 0), 1)
            sk = min(max(s_raw, 0), 6)

            team_counts_raw[t] = {'G': g_raw, 'S': s_raw}
            team_counts_clamped[t] = {'G': g, 'S': sk}

        def _normalize_strength_state(*, my_s: int, their_s: int, my_g: int, their_g: int) -> str:
            observed_goalies = (my_g + their_g) > 0
            if observed_goalies and my_g == 0 and their_g >= 1:
                return 'ENF'
            if observed_goalies and their_g == 0 and my_g >= 1:
                return 'ENA'

            ms = int(my_s or 0)
            ts = int(their_s or 0)
            ms = max(0, min(ms, 6))
            ts = max(0, min(ts, 6))

            # Preserve empty-net states (6vX / Xv6) explicitly. The expected set includes
            # 6v5/6v4/6v3/6v2 and their inverses.
            if ms == 6 and ts == 6:
                # Extremely rare (both teams with extra attacker) — snap to even strength.
                return '5v5'
            # Requested bucketing for future shifts calcs:
            # - treat 6v5/5v6 as 5v5 (empty net shouldn't split the state)
            # - treat 6v4 as PP and 4v6 as SH
            if (ms, ts) in {(6, 5), (5, 6)}:
                return '5v5'
            if (ms, ts) == (6, 4):
                return 'PP'
            if (ms, ts) == (4, 6):
                return 'SH'
            if ms == 6 or ts == 6:
                return f'{ms}v{ts}'

            # PP / SH
            if ts == 4 and ms == 5:
                return '5v4'
            if ts == 3 and ms == 5:
                return '5v3'
            if ts == 3 and ms == 4:
                return '4v3'
            if ms == 4 and ts == 5:
                return '4v5'
            if ms == 3 and ts == 5:
                return '3v5'
            if ms == 3 and ts == 4:
                return '3v4'

            # Two-man advantage/disadvantage cases that we want to preserve.
            if ms == 2 and ts in (3, 4, 5):
                return f'2v{ts}'
            if ts == 2 and ms in (3, 4, 5):
                return f'{ms}v2'

            # Even strength
            if ms == 4 and ts == 4:
                return '4v4'
            if ms == 3 and ts == 3:
                return '3v3'
            if ms == 5 and ts == 5:
                return '5v5'

            # Extreme low-count fallbacks.
            if ts == 0 and ms >= 1:
                return '1v0'
            if ms == 0 and ts >= 1:
                return '0v1'

            # If we're missing players due to scraping quirks, snap to closest even-strength.
            m = max(ms, ts)
            if m <= 3:
                return '3v3'
            if m == 4:
                return '4v4'
            return '5v5'

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
                opp = next((t for t in team_counts_clamped.keys() if t != team), '')

            my_raw = team_counts_raw.get(team, {'G': 0, 'S': 0})
            their_raw = team_counts_raw.get(opp, {'G': 0, 'S': 0})
            my = team_counts_clamped.get(team, {'G': 0, 'S': 0})
            their = team_counts_clamped.get(opp, {'G': 0, 'S': 0})

            # SeasonStats bucketing: only trust skater counts when both goalies are in.
            # Rules:
            # - both goalies in + both teams have 5+ skaters -> 5v5
            # - both goalies in + opponent has 3/4 skaters -> PP (if we have more skaters)
            # - both goalies in + we have 3/4 skaters -> SH (if opponent has more skaters)
            # - else -> Other
            try:
                my_g = int(my.get('G') or 0)
                their_g = int(their.get('G') or 0)
                # Some shift reports omit goalies entirely; if BOTH sides have 0 goalie rows,
                # treat goalie presence as unknown and assume both goalies are in.
                both_goalies_in = (my_g >= 1 and their_g >= 1) or (my_g == 0 and their_g == 0)
                my_s = int(my.get('S') or 0)
                their_s = int(their.get('S') or 0)
                if both_goalies_in and my_s >= 5 and their_s >= 5:
                    strength_bucket = '5v5'
                elif both_goalies_in and their_s in (3, 4) and my_s > their_s:
                    strength_bucket = 'PP'
                elif both_goalies_in and my_s in (3, 4) and their_s > my_s:
                    strength_bucket = 'SH'
                else:
                    strength_bucket = 'Other'
            except Exception:
                my_g = 0
                their_g = 0
                my_s = 0
                their_s = 0
                strength_bucket = 'Other'

            strength = _normalize_strength_state(my_s=my_s, their_s=their_s, my_g=my_g, their_g=their_g)
            strength_raw = f"{int(my_raw.get('S') or 0)}v{int(their_raw.get('S') or 0)}"

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
                'StrengthStateRaw': strength_raw,
                'StrengthStateBucket': strength_bucket,
                'SkatersOnIceFor': int(my_raw.get('S') or 0),
                'SkatersOnIceAgainst': int(their_raw.get('S') or 0),
                'GoaliesOnIceFor': int(my_raw.get('G') or 0),
                'GoaliesOnIceAgainst': int(their_raw.get('G') or 0),
            })

    out = {
        'gameId': game_id,
        'seasonDir': season_dir,
        'suffix': suffix,
        'source': urls,
        'shifts': split_rows,
    }

    try:
        # Include gameState for disk cache TTL check
        game_state = str((box.get('gameState') or box.get('gameStatus') or '')).upper()
        out['gameState'] = game_state
    except Exception:
        pass

    try:
        _cache_set_multi_bounded(_SHIFTS_CACHE, int(game_id), out, ttl_s=std_ttl, max_items=max_items)
    except Exception:
        pass

    # Persist to disk
    try:
        import json
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


