r"""
update_data.py

Fetch play-by-play and shifts dataframes for all NHL games on a given date.

- Reuses the app's existing logic (routes) via an internal Flask test client,
  so we get identical parsing/mapping as the web app (including robust Shifts parsing).
- Computes xG via the PBP route (xg=1).

Usage (PowerShell):
    pwsh> & .\.venv\Scripts\python.exe .\scripts\update_data.py --date 2025-10-12

Returns three pandas DataFrames: combined PBP, combined Shifts, and per-game Player GameData for the given date.
When run as a script, prints a short summary and shows .head() samples.
"""
from __future__ import annotations

import os
import sys
import argparse
import subprocess
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict, Any, cast
import json
import re

import requests
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

# Ensure the app module is importable when running from repo root or scripts/
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Disable xG model preload when creating the Flask app
os.environ.setdefault('XG_PRELOAD', '0')
# Ensure we populate Shoots by fetching bios inside the PBP route
os.environ.setdefault('FETCH_BIOS', '1')

from app import create_app  # noqa: E402

# Optional HTML parsing for lineup scraping
try:
    from bs4 import BeautifulSoup  # type: ignore
except Exception:
    BeautifulSoup = None  # type: ignore




def _validate_date(d: str) -> str:
    try:
        # Expect YYYY-MM-DD
        dt = datetime.strptime(d, '%Y-%m-%d')
        return dt.strftime('%Y-%m-%d')
    except Exception:
        raise argparse.ArgumentTypeError('Date must be in YYYY-MM-DD format')


def get_game_ids_for_date(date_str: str) -> List[int]:
    """Fetch schedule for the date and extract game ids.
    Uses NHL's public schedule endpoint: https://api-web.nhle.com/v1/schedule/{YYYY-MM-DD}
    """
    url = f'https://api-web.nhle.com/v1/schedule/{date_str}'
    try:
        r = requests.get(url, timeout=25)
        r.raise_for_status()
        js = r.json()
    except Exception as e:
        raise RuntimeError(f'Failed to fetch schedule for {date_str}: {e}')

    ids: List[int] = []

    def _start_date(g: dict) -> Optional[str]:
        st = g.get('startTimeUTC') or g.get('gameDate') or ''
        if not st:
            return None
        try:
            # startTimeUTC is ISO string like '2025-10-12T23:00:00Z'
            # Take the first 10 chars (YYYY-MM-DD)
            return str(st).replace('Z', '').strip()[:10]
        except Exception:
            return None

    # The API returns an array of days under gameWeek; collect only matching date
    weeks = js.get('gameWeek') if isinstance(js, dict) else None
    if isinstance(weeks, list):
        for day in weeks:
            day_date = (day.get('date') or '')[:10]
            for g in (day.get('games') or []):
                gd = _start_date(g)
                if day_date == date_str or gd == date_str:
                    gid = g.get('id') or g.get('gamePk') or g.get('gameId')
                    try:
                        ids.append(int(gid))
                    except Exception:
                        continue
    # If direct 'games' key exists (alternate shape), filter by startTimeUTC
    if isinstance(js, dict) and not ids:
        for g in (js.get('games') or []):
            gd = _start_date(g)
            if gd == date_str:
                gid = g.get('id') or g.get('gamePk') or g.get('gameId')
                try:
                    ids.append(int(gid))
                except Exception:
                    continue
    return sorted(list({i for i in ids if i}))


def _fetch_birthdays_for_game(game_id: int) -> Dict[int, Optional[str]]:
    """Return {playerId: birthDate} for the given game using NHL stats bios endpoints.
    Includes both skaters and goalies to ensure goalies get birthdays too.
    """
    out: Dict[int, Optional[str]] = {}
    base = "https://api.nhle.com/stats/rest/en"
    urls = [
        f"{base}/skater/bios?limit=-1&start=0&cayenneExp=gameId={game_id}",
        f"{base}/goalie/bios?limit=-1&start=0&cayenneExp=gameId={game_id}",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                js = r.json() or {}
                rows = js.get('data') or []
                if isinstance(rows, list):
                    for row in rows:
                        try:
                            pid = row.get('playerId')
                            bday = row.get('birthDate') or row.get('playerBirthDate')
                            if isinstance(pid, int) and pid not in out:
                                out[pid] = bday
                        except Exception:
                            continue
        except Exception:
            continue
    return out


def fetch_day(date_str: str, with_xg: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (df_pbp, df_shifts) for all games played on date_str (YYYY-MM-DD).
    Reuses internal Flask routes via a test client; does not require the dev server to run.
    """
    game_ids = get_game_ids_for_date(date_str)
    if not game_ids:
        # No games on this date
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    app = create_app()
    df_pbp_list: List[pd.DataFrame] = []
    df_shifts_list: List[pd.DataFrame] = []
    df_gamedata_list: List[pd.DataFrame] = []

    # Use a test client so requests are fully handled inside the app context
    with app.test_client() as client:
        for gid in game_ids:
            pbp_df = pd.DataFrame()
            # Play-by-play (optionally compute xG)
            resp_pbp = client.get(
                f'/api/game/{gid}/play-by-play',
                query_string={'xg': ('1' if with_xg else '0')}
            )
            if resp_pbp.status_code == 200:
                js = resp_pbp.get_json(silent=True) or {}
                plays = js.get('plays') or []
                if isinstance(plays, list) and plays:
                    df = pd.DataFrame(plays)
                    df['Date'] = date_str
                    df['GameID'] = gid
                    df_pbp_list.append(df)
                    pbp_df = df
            else:
                # Keep going on failures; print minimal info
                print(f'[warn] play-by-play failed for {gid}: {resp_pbp.status_code}', file=sys.stderr)

            # Shifts
            resp_sh = client.get(f'/api/game/{gid}/shifts', query_string={'force': '1'})
            if resp_sh.status_code == 200:
                js2 = resp_sh.get_json(silent=True) or {}
                rows = js2.get('shifts') or []
                if isinstance(rows, list) and rows:
                    df2 = pd.DataFrame(rows)
                    df2['Date'] = date_str
                    df2['GameID'] = gid
                    df_shifts_list.append(df2)

                    # Build per-player TOI aggregates for df_gamedata
                    try:
                        bdays = _fetch_birthdays_for_game(gid)
                    except Exception:
                        bdays = {}
                    d = df2.copy()
                    # Ensure Position column exists
                    if 'Position' not in d.columns:
                        if 'Pos' in d.columns:
                            d['Position'] = d['Pos']
                        else:
                            d['Position'] = None
                    # Ensure Duration is numeric seconds
                    d['Duration'] = pd.to_numeric(d['Duration'], errors='coerce').fillna(0).astype(float)
                    # Strength buckets
                    if 'StrengthState' not in d.columns:
                        d['StrengthState'] = None
                    ev_set = {'5v5', '4v4', '3v3'}
                    pp_set = {'5v4', '5v3', '4v3'}
                    sh_set = {'4v5', '3v5', '3v4'}
                    d['is_ev'] = d['StrengthState'].isin(ev_set).astype(int)
                    d['is_pp'] = d['StrengthState'].isin(pp_set).astype(int)
                    d['is_sh'] = d['StrengthState'].isin(sh_set).astype(int)
                    # Avoid GroupBy.apply deprecation by computing conditional durations then aggregating
                    d['__dur_all_min'] = d['Duration'] / 60.0
                    d['__dur_ev_min'] = d['Duration'].where(d['is_ev'] == 1, 0.0) / 60.0
                    d['__dur_pp_min'] = d['Duration'].where(d['is_pp'] == 1, 0.0) / 60.0
                    d['__dur_sh_min'] = d['Duration'].where(d['is_sh'] == 1, 0.0) / 60.0
                    agg = (
                        d.groupby(['GameID', 'PlayerID', 'Name', 'Position', 'Team'], dropna=False)
                         .agg(
                             TOI_All=('__dur_all_min', 'sum'),
                             TOI_EV=('__dur_ev_min', 'sum'),
                             TOI_PP=('__dur_pp_min', 'sum'),
                             TOI_SH=('__dur_sh_min', 'sum'),
                         )
                         .reset_index()
                    )
                    agg['Date'] = date_str
                    # Birthday
                    def map_bday(pid):
                        try:
                            return bdays.get(int(pid))
                        except Exception:
                            return None
                    agg['Birthday'] = agg['PlayerID'].map(map_bday)
                    agg = agg.rename(columns={'Name': 'Player'})

                    # Individual statistics from PBP for the same game
                    # Use PBP rows fetched for this game if available
                    ev_set = {'5v5', '4v4', '3v3'}
                    pp_set = {'5v4', '5v3', '4v3'}
                    sh_set = {'4v5', '3v5', '3v4'}
                    indiv_cols = [
                        'iG_All','iG_EV','iG_PP','iG_SH',
                        'A1_All','A1_EV','A1_PP','A1_SH',
                        'A2_All','A2_EV','A2_PP','A2_SH',
                        'PEN_taken','PEN_drawn'
                    ]
                    indiv = pd.DataFrame(columns=['GameID','PlayerID'] + indiv_cols)
                    if not pbp_df.empty:
                        p = pbp_df[pbp_df['GameID'] == gid].copy()
                        # Goals and assists
                        goals = p[p['Goal'] == 1].copy()
                        for label, col in [('iG','Player1_ID'), ('A1','Player2_ID'), ('A2','Player3_ID')]:
                            tmp = goals[[col, 'GameID', 'StrengthState']].rename(columns={col: 'PlayerID'})
                            tmp = tmp.dropna(subset=['PlayerID'])
                            tmp['is_ev'] = tmp['StrengthState'].isin(ev_set)
                            tmp['is_pp'] = tmp['StrengthState'].isin(pp_set)
                            tmp['is_sh'] = tmp['StrengthState'].isin(sh_set)
                            gsum = tmp.groupby(['GameID','PlayerID']).agg(
                                **{
                                    f'{label}_All': ('GameID','count'),
                                    f'{label}_EV': ('is_ev','sum'),
                                    f'{label}_PP': ('is_pp','sum'),
                                    f'{label}_SH': ('is_sh','sum'),
                                }
                            ).reset_index()
                            indiv = gsum if indiv.empty else indiv.merge(gsum, on=['GameID','PlayerID'], how='outer')
                        # Penalties
                        pens = p[p['PEN_duration'].notna()].copy()
                        # PEN_taken: sum of penalty minutes assessed to penalized player (Player1_ID)
                        # PEN_drawn: sum of penalty minutes drawn by drawing player (Player2_ID)
                        if not pens.empty:
                            pens['PEN_duration'] = pd.to_numeric(pens['PEN_duration'], errors='coerce').fillna(0)
                            # Taken minutes
                            taken = pens[['GameID','Player1_ID','PEN_duration']].rename(columns={'Player1_ID':'PlayerID'})
                            taken = taken.dropna(subset=['PlayerID'])
                            taken = taken.groupby(['GameID','PlayerID'])['PEN_duration'].sum().reset_index().rename(columns={'PEN_duration':'PEN_taken'})
                            indiv = taken if indiv.empty else indiv.merge(taken, on=['GameID','PlayerID'], how='outer')
                            # Drawn minutes
                            drawn = pens[['GameID','Player2_ID','PEN_duration']].rename(columns={'Player2_ID':'PlayerID'})
                            drawn = drawn.dropna(subset=['PlayerID'])
                            drawn = drawn.groupby(['GameID','PlayerID'])['PEN_duration'].sum().reset_index().rename(columns={'PEN_duration':'PEN_drawn'})
                            indiv = drawn if indiv.empty else indiv.merge(drawn, on=['GameID','PlayerID'], how='outer')

                        # On-ice FOR/AGAINST metrics per player
                        def _to_ids(v):
                            if v is None:
                                return []
                            if isinstance(v, list):
                                return [int(x) for x in v if x is not None and str(x).strip() != '']
                            s = str(v)
                            if not s:
                                return []
                            # Try JSON array like "[1,2,3]"
                            try:
                                if s.strip().startswith('[') and s.strip().endswith(']'):
                                    arr = json.loads(s)
                                    if isinstance(arr, list):
                                        return [int(x) for x in arr if x is not None and str(x).strip() != '']
                            except Exception:
                                pass
                            # Fallback: split by common separators
                            parts = re.split(r'[\,\|;\s]+', s)
                            out = []
                            for part in parts:
                                part = part.strip()
                                if not part:
                                    continue
                                try:
                                    out.append(int(part))
                                except Exception:
                                    continue
                            return out

                        inv_map = {'5v4':'4v5','5v3':'3v5','4v3':'3v4','4v5':'5v4','3v5':'5v3','3v4':'4v3'}
                        def invert_strength(s):
                            s = str(s or '')
                            return inv_map.get(s, s)

                        for coln in ['Corsi','Fenwick','Shot','Goal','xG_F','xG_S','xG_F2']:
                            if coln not in p.columns:
                                p[coln] = 0
                        p['Corsi'] = pd.to_numeric(p['Corsi'], errors='coerce').fillna(0)
                        p['Fenwick'] = pd.to_numeric(p['Fenwick'], errors='coerce').fillna(0)
                        p['Shot'] = pd.to_numeric(p['Shot'], errors='coerce').fillna(0)
                        p['Goal'] = pd.to_numeric(p['Goal'], errors='coerce').fillna(0)
                        p['xG_F'] = pd.to_numeric(p['xG_F'], errors='coerce').fillna(0.0)
                        p['xG_S'] = pd.to_numeric(p['xG_S'], errors='coerce').fillna(0.0)
                        p['xG_F2'] = pd.to_numeric(p['xG_F2'], errors='coerce').fillna(0.0)

                        def build_onice_lists(row):
                            # Support a few alternate key spellings with hyphens
                            def get_any(row, keys):
                                for k in keys:
                                    if k in row:
                                        return row.get(k)
                                return None
                            h_f = _to_ids(get_any(row, ['Home_Forwards_ID','Home_Forwards-ID','Home_ForwardsIDs','Home_Forwards']))
                            h_d = _to_ids(get_any(row, ['Home_Defenders_ID','Home_Defenders-ID','Home_DefendersIDs','Home_Defenders']))
                            a_f = _to_ids(get_any(row, ['Away_Forwards_ID','Away_Forwards-ID','Away_ForwardsIDs','Away_Forwards']))
                            a_d = _to_ids(get_any(row, ['Away_Defenders_ID','Away_Defenders-ID','Away_DefendersIDs','Away_Defenders']))
                            h_g = []
                            if row.get('Home_Goalie_ID') is not None:
                                try:
                                    h_g = [int(row.get('Home_Goalie_ID'))]
                                except Exception:
                                    h_g = []
                            a_g = []
                            if row.get('Away_Goalie_ID') is not None:
                                try:
                                    a_g = [int(row.get('Away_Goalie_ID'))]
                                except Exception:
                                    a_g = []
                            return (h_f + h_d + h_g, a_f + a_d + a_g)

                        stats_for: dict[int, dict[str, float]] = {}
                        stats_against: dict[int, dict[str, float]] = {}

                        def upd(store: dict[int, dict[str, float]], pid: int, name: str, val: float, strength: str):
                            if pid is None:
                                return
                            d0 = store.setdefault(int(pid), {})
                            d0[name + '_All'] = d0.get(name + '_All', 0.0) + float(val)
                            if strength in ev_set:
                                d0[name + '_EV'] = d0.get(name + '_EV', 0.0) + float(val)
                            elif strength in pp_set:
                                d0[name + '_PP'] = d0.get(name + '_PP', 0.0) + float(val)
                            elif strength in sh_set:
                                d0[name + '_SH'] = d0.get(name + '_SH', 0.0) + float(val)

                        for _, row in p.iterrows():
                            strength = row.get('StrengthState')
                            strength_str = str(strength or '')
                            home_list, away_list = build_onice_lists(row)
                            if not home_list and not away_list:
                                continue
                            venue = row.get('Venue')
                            if venue == 'Home':
                                players_for, players_against = home_list, away_list
                            elif venue == 'Away':
                                players_for, players_against = away_list, home_list
                            else:
                                # Unknown venue; skip
                                continue

                            c = row.get('Corsi', 0) or 0
                            fval = row.get('Fenwick', 0) or 0
                            s = row.get('Shot', 0) or 0
                            g = row.get('Goal', 0) or 0
                            xgf = row.get('xG_F', 0.0) or 0.0
                            xgs = row.get('xG_S', 0.0) or 0.0
                            xgf2 = row.get('xG_F2', 0.0) or 0.0

                            for pid in players_for:
                                upd(stats_for, pid, 'CF', c, strength_str)
                                upd(stats_for, pid, 'FF', fval, strength_str)
                                upd(stats_for, pid, 'SF', s, strength_str)
                                upd(stats_for, pid, 'GF', g, strength_str)
                                upd(stats_for, pid, 'xGF_F', xgf, strength_str)
                                upd(stats_for, pid, 'xGF_S', xgs, strength_str)
                                upd(stats_for, pid, 'xGF_F2', xgf2, strength_str)

                            inv_strength = invert_strength(strength_str)
                            for pid in players_against:
                                upd(stats_against, pid, 'CA', c, inv_strength)
                                upd(stats_against, pid, 'FA', fval, inv_strength)
                                upd(stats_against, pid, 'SA', s, inv_strength)
                                upd(stats_against, pid, 'GA', g, inv_strength)
                                upd(stats_against, pid, 'xGA_F', xgf, inv_strength)
                                upd(stats_against, pid, 'xGA_S', xgs, inv_strength)
                                upd(stats_against, pid, 'xGA_F2', xgf2, inv_strength)

                        def dict_to_df(dct: dict[int, dict[str, float]]) -> pd.DataFrame:
                            if not dct:
                                return pd.DataFrame(columns=['GameID','PlayerID'])
                            try:
                                dfm = pd.DataFrame.from_dict(dct, orient='index')
                                dfm.index.name = 'PlayerID'
                                dfm.reset_index(inplace=True)
                                dfm['PlayerID'] = dfm['PlayerID'].astype(int)
                                dfm.insert(0, 'GameID', gid)
                                return dfm
                            except Exception:
                                return pd.DataFrame(columns=['GameID','PlayerID'])

                        for_df = dict_to_df(stats_for)
                        ag_df = dict_to_df(stats_against)
                        if not for_df.empty:
                            agg = agg.merge(for_df, on=['GameID','PlayerID'], how='left')
                        if not ag_df.empty:
                            agg = agg.merge(ag_df, on=['GameID','PlayerID'], how='left')

                    # Merge indiv stats into agg and fill NA with zeros for numeric
                    if not indiv.empty:
                        # Ensure expected columns exist even if the first merge (iG/A1/A2) created a partial frame
                        # and/or no penalties occurred in the game.
                        for c in indiv_cols:
                            if c not in indiv.columns:
                                indiv[c] = 0
                        agg = agg.merge(indiv, on=['GameID','PlayerID'], how='left')
                    else:
                        for c in indiv_cols:
                            agg[c] = 0

                    # Final column ordering
                    # Ensure on-ice columns exist
                    for c in [
                        'CF_All','CF_EV','CF_PP','CF_SH', 'CA_All','CA_EV','CA_PP','CA_SH',
                        'FF_All','FF_EV','FF_PP','FF_SH', 'FA_All','FA_EV','FA_PP','FA_SH',
                        'SF_All','SF_EV','SF_PP','SF_SH', 'SA_All','SA_EV','SA_PP','SA_SH',
                        'GF_All','GF_EV','GF_PP','GF_SH', 'GA_All','GA_EV','GA_PP','GA_SH',
                        'xGF_F_All','xGF_F_EV','xGF_F_PP','xGF_F_SH', 'xGA_F_All','xGA_F_EV','xGA_F_PP','xGA_F_SH',
                        'xGF_F2_All','xGF_F2_EV','xGF_F2_PP','xGF_F2_SH', 'xGA_F2_All','xGA_F2_EV','xGA_F2_PP','xGA_F2_SH',
                        'xGF_S_All','xGF_S_EV','xGF_S_PP','xGF_S_SH', 'xGA_S_All','xGA_S_EV','xGA_S_PP','xGA_S_SH'
                    ]:
                        if c not in agg.columns:
                            agg[c] = 0.0

                    agg = agg[[
                        'Date','GameID','PlayerID','Player','Position','Team','Birthday',
                        'TOI_All','TOI_EV','TOI_PP','TOI_SH',
                        'CF_All','CF_EV','CF_PP','CF_SH', 'CA_All','CA_EV','CA_PP','CA_SH',
                        'FF_All','FF_EV','FF_PP','FF_SH', 'FA_All','FA_EV','FA_PP','FA_SH',
                        'SF_All','SF_EV','SF_PP','SF_SH', 'SA_All','SA_EV','SA_PP','SA_SH',
                        'GF_All','GF_EV','GF_PP','GF_SH', 'GA_All','GA_EV','GA_PP','GA_SH',
                        'xGF_F_All','xGF_F_EV','xGF_F_PP','xGF_F_SH', 'xGA_F_All','xGA_F_EV','xGA_F_PP','xGA_F_SH',
                        'xGF_F2_All','xGF_F2_EV','xGF_F2_PP','xGF_F2_SH', 'xGA_F2_All','xGA_F2_EV','xGA_F2_PP','xGA_F2_SH',
                        'xGF_S_All','xGF_S_EV','xGF_S_PP','xGF_S_SH', 'xGA_S_All','xGA_S_EV','xGA_S_PP','xGA_S_SH',
                        'iG_All','iG_EV','iG_PP','iG_SH',
                        'A1_All','A1_EV','A1_PP','A1_SH',
                        'A2_All','A2_EV','A2_PP','A2_SH',
                        'PEN_taken','PEN_drawn'
                    ]]
                    # Fill NaNs in numeric columns with zeros
                    num_cols = [
                        c for c in agg.columns
                        if c.startswith(('TOI_', 'iG_', 'A1_', 'A2_', 'PEN_', 'CF_', 'CA_', 'FF_', 'FA_', 'SF_', 'SA_', 'GF_', 'GA_', 'xGF_', 'xGA_'))
                    ]
                    agg[num_cols] = agg[num_cols].fillna(0)
                    df_gamedata_list.append(agg)
            else:
                print(f'[warn] shifts failed for {gid}: {resp_sh.status_code}', file=sys.stderr)

    df_pbp = pd.concat(df_pbp_list, ignore_index=True) if df_pbp_list else pd.DataFrame()
    df_shifts = pd.concat(df_shifts_list, ignore_index=True) if df_shifts_list else pd.DataFrame()
    df_gamedata = pd.concat(df_gamedata_list, ignore_index=True) if df_gamedata_list else pd.DataFrame()
    return df_pbp, df_shifts, df_gamedata


def _create_mysql_engine(desired: str = 'rw') -> Optional[Engine]:
    """Create a SQLAlchemy engine for MySQL using env overrides.
    Precedence (URLs):
      desired=='rw': DATABASE_URL_RW, DB_URL_RW, DATABASE_URL
      desired=='ro': DATABASE_URL_RO, DB_URL_RO, DATABASE_URL
      else: DATABASE_URL
    If URL missing, build from discrete vars (DB_*_RW / DB_*_RO / DB_*).
    SSL: DB_SSL_CA / DB_SSL_CERT / DB_SSL_KEY
    """
    def _first_env(*names: str) -> Optional[str]:
        for n in names:
            v = os.getenv(n)
            if v:
                return v
        return None

    # Pick URL based on desired access level
    db_url = None
    if desired == 'rw':
        db_url = _first_env('DATABASE_URL_RW', 'DB_URL_RW', 'DATABASE_URL')
    elif desired == 'ro':
        db_url = _first_env('DATABASE_URL_RO', 'DB_URL_RO', 'DATABASE_URL')
    else:
        db_url = _first_env('DATABASE_URL')

    # SSL args
    connect_args = {}
    ssl_ca = os.getenv('DB_SSL_CA')
    ssl_cert = os.getenv('DB_SSL_CERT')
    ssl_key = os.getenv('DB_SSL_KEY')
    if ssl_ca or ssl_cert or ssl_key:
        ssl_args = {}
        if ssl_ca:
            ssl_args['ssl_ca'] = ssl_ca
        if ssl_cert:
            ssl_args['ssl_cert'] = ssl_cert
        if ssl_key:
            ssl_args['ssl_key'] = ssl_key
        connect_args.update(ssl_args)

    try:
        if db_url:
            # Replace '@localhost' with LAN/public IP if DB_HOST(_RW/_RO) is set
            host_env = None
            if desired == 'rw':
                host_env = _first_env('DB_HOST_RW') or os.getenv('DB_HOST')
            elif desired == 'ro':
                host_env = _first_env('DB_HOST_RO') or os.getenv('DB_HOST')
            else:
                host_env = os.getenv('DB_HOST')
            if host_env and '@localhost' in db_url:
                db_url = db_url.replace('@localhost', f'@{host_env}')
            return create_engine(db_url, connect_args=connect_args)
        # Build from parts with suffix-specific variables
        def _part(base: str, fallback: Optional[str] = None) -> str:
            return os.getenv(base, fallback or '')
        if desired == 'rw':
            username = _first_env('DB_USER_RW') or _part('DB_USER', 'root')
            password = _first_env('DB_PASSWORD_RW') or _part('DB_PASSWORD', 'Sunesen1')
            host = _first_env('DB_HOST_RW') or _part('DB_HOST', 'localhost')
            port = _first_env('DB_PORT_RW') or _part('DB_PORT', '3306')
            database = _first_env('DB_NAME_RW') or _part('DB_NAME', 'public')
        elif desired == 'ro':
            username = _first_env('DB_USER_RO') or _part('DB_USER', 'root')
            password = _first_env('DB_PASSWORD_RO') or _part('DB_PASSWORD', 'Sunesen1')
            host = _first_env('DB_HOST_RO') or _part('DB_HOST', 'localhost')
            port = _first_env('DB_PORT_RO') or _part('DB_PORT', '3306')
            database = _first_env('DB_NAME_RO') or _part('DB_NAME', 'public')
        else:
            username = _part('DB_USER', 'root')
            password = _part('DB_PASSWORD', 'Sunesen1')
            host = _part('DB_HOST', 'localhost')
            port = _part('DB_PORT', '3306')
            database = _part('DB_NAME', 'public')
        url = f"mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}"
        return create_engine(url, connect_args=connect_args)
    except Exception as e:
        print(f"[error] failed to create MySQL engine: {e}", file=sys.stderr)
        return None


def _load_teams_csv_local() -> List[Dict[str, str]]:
    paths = [
        os.path.join(REPO_ROOT, 'Teams.csv'),
        os.path.join(os.path.dirname(__file__), '..', 'Teams.csv'),
        os.path.join(os.getcwd(), 'Teams.csv'),
    ]
    for p in paths:
        try:
            if os.path.exists(p):
                return pd.read_csv(p, dtype=str).fillna('').to_dict(orient='records')  # type: ignore
        except Exception:
            continue
    return []

TEAMS_ROWS_LOCAL = _load_teams_csv_local()


def infer_team_from_dailyfaceoff_url(url: str) -> str:
    """Infer NHL team abbrev (e.g., ANA) from a DailyFaceoff team URL using Teams.csv.
    Example URL: https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations
    """
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
            # Prefer Active==1
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
        # Build a reasonable default from current date
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
    # Prefer the latest finished game
    finished_states = {'OFF', 'FINAL', 'COMPLETED', 'OFFICIAL'}
    last_game_id = None
    for g in reversed(games):
        st = str(g.get('gameState') or g.get('gameStatus') or '').upper()
        gid = g.get('id') or g.get('gamePk')
        if gid and (st in finished_states or st == 'FUT' or st == 'PREVIEW' or st == 'SCHEDULED'):
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


def scrape_dailyfaceoff_lineup(url: str) -> Dict[str, List[Dict]]:
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

    out = { 'forwards': [], 'defense': [], 'goalies': [] }

    def norm_text(s: str) -> str:
        return ' '.join((s or '').replace('\xa0',' ').strip().split())
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
        # Heuristic: find sections for Forwards / Defense Pairings / Goalies
        # The structure may use headings and repeated rows/cells. Look for player cards/anchors.
        def collect_players(container, unit_label_prefix: str, pos_code: str):
            items = []
            if not container:
                return items
            # Find clickable player names or spans with numbers
            candidates = container.find_all(['a','div','span'], string=True)
            for node in candidates:
                txt = norm_text(node.get_text(' ', strip=True))
                if not txt or len(txt) < 2:
                    continue
                # Filter out generic words
                if txt.lower() in ('forwards','defense','defence','defense pairings','goalies','goalie','line combinations'):
                    continue
                name, j = parse_name_and_jersey(txt)
                # crude name filter: must contain a space (first+last) or at least 3+ letters
                if len(name.split()) >= 1 and any(c.isalpha() for c in name):
                    items.append({ 'name': name, 'jersey': j, 'pos': pos_code, 'unit': unit_label_prefix })
            return items

        # Attempt to locate specific groups by headings
        def find_section_by_heading(soup, heading_texts: List[str]):
            for htag in soup.find_all(['h2','h3','h4','h5','div','span']):
                txt = norm_text(htag.get_text(' ', strip=True)).lower()
                if any(ht in txt for ht in heading_texts):
                    # use next sibling or parent container
                    cont = htag.find_next()
                    parent = htag.parent
                    return cont or parent
            return None

        fwd_cont = find_section_by_heading(soup, ['forwards','forward lines'])
        d_cont = find_section_by_heading(soup, ['defense','defence'])
        g_cont = find_section_by_heading(soup, ['goalies','goalie'])

        out['forwards'] = collect_players(fwd_cont, 'L', 'F')
        out['defense'] = collect_players(d_cont, 'D', 'D')
        out['goalies'] = collect_players(g_cont, 'G', 'G')

    else:
        # Fallback: regex-based extraction
        lines = [l.strip() for l in html.splitlines() if l.strip()]
        for ln in lines:
            low = ln.lower()
            if any(k in low for k in ('forwards','forward lines')):
                # subsequent lines may have numbers and names
                pass
            m = re.findall(r'(?:#?\s*(\d{1,2})\s+)?([A-Z][a-z]+\s+[A-Z][a-z\-\']+)', ln)
            for num, nm in m:
                nm2 = nm.strip(); num2 = num.strip() if num else None
                if nm2:
                    out['forwards'].append({ 'name': nm2, 'jersey': num2, 'pos': 'F', 'unit': 'L' })

    # Deduplicate by (name, jersey)
    def dedup(lst: List[Dict]) -> List[Dict]:
        seen = set(); res = []
        for it in lst:
            key = (it.get('name') or '', it.get('jersey') or '')
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
    roster: list of { playerId, name, sweaterNumber, pos } as from fetch_recent_roster.
    Returns the lineup dict with an added 'playerId' key where matched.
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
    for p in roster:
        num = jersey_norm(p.get('sweaterNumber'))
        if num:
            by_num[num] = p
        nm = norm(p.get('name'))
        if nm:
            by_name[nm] = p
            last = nm.split(' ')[-1]
            by_last.setdefault(last, []).append(p)

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

    mapped = { 'team': team_abbrev, 'forwards': [], 'defense': [], 'goalies': [] }
    for key in ('forwards','defense','goalies'):
        res = []
        for it in lineup.get(key, []):
            pid = match_player(it.get('name'), it.get('jersey'))
            it2 = dict(it)
            it2['playerId'] = pid
            res.append(it2)
        mapped[key] = res
    return mapped


def export_to_mysql(
    df_pbp: pd.DataFrame,
    df_shifts: pd.DataFrame,
    df_gamedata: Optional[pd.DataFrame] = None,
    season: str = "20252026",
    *,
    date_str: Optional[str] = None,
    replace_date: bool = False,
) -> None:
    """Export the dataframes into MySQL tables.
    - df_pbp -> nhl_{season}_pbp
    - df_shifts -> nhl_{season}_shifts
    Creates tables if they don't exist; appends otherwise.
    """
    eng = _create_mysql_engine('rw')
    if eng is None:
        raise RuntimeError("MySQL engine not available")

    tbl_pbp = f"nhl_{season}_pbp"
    tbl_sh = f"nhl_{season}_shifts"
    tbl_gd = f"nhl_{season}_gamedata"

    def _ensure_mysql_column(table_name: str, col_name: str, col_def_sql: str) -> bool:
        """Best-effort: ensure a column exists on an existing MySQL table.

        Returns True if the column exists or was added successfully.
        Returns False if the column could not be ensured (e.g., permissions).
        """
        try:
            with eng.begin() as conn:
                res = conn.execute(
                    text(
                        """
                        SELECT COUNT(*) AS n
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_SCHEMA = DATABASE()
                          AND TABLE_NAME = :t
                          AND COLUMN_NAME = :c
                        """
                    ),
                    {"t": table_name, "c": col_name},
                )
                n = int((res.scalar() or 0))
                if n > 0:
                    return True
        except Exception:
            # Can't inspect schema; don't block exports.
            return False

        try:
            with eng.begin() as conn:
                conn.execute(text(f"ALTER TABLE `{table_name}` ADD COLUMN `{col_name}` {col_def_sql}"))
            return True
        except Exception:
            return False

    def _apply_xg_nulling_rule(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """Ensure xG columns are NULL when reason indicates invalid/short xG.

        Rule: if reason in {'short','failed-bank-attempt'} then xG_F/xG_F2/xG_S must be NULL.
        Applies case-insensitively and only when the needed columns exist.
        """
        if df is None or df.empty:
            return df

        # Find reason column (case-insensitive)
        reason_col = None
        for c in df.columns:
            if str(c).lower() == 'reason':
                reason_col = c
                break
        if not reason_col:
            return df

        reasons = (
            df[reason_col]
            .astype(str)
            .str.strip()
            .str.lower()
        )
        bad = reasons.isin({'short', 'failed-bank-attempt'})
        if not bool(bad.any()):
            return df

        # Map xG columns case-insensitively
        col_by_lower = {str(c).lower(): c for c in df.columns}
        for key in ('xg_f', 'xg_f2', 'xg_s'):
            real = col_by_lower.get(key)
            if real is not None:
                df.loc[bad, real] = float('nan')
        return df

    # Optional: pre-delete by date for idempotent loads (even if today's fetch is empty)
    if replace_date and date_str is not None:
        try:
            with eng.begin() as conn:
                conn.execute(text(f"DELETE FROM {tbl_pbp} WHERE Date = :d"), {"d": date_str})
        except Exception:
            # Table may not exist on first run; ignore
            pass
        try:
            with eng.begin() as conn:
                conn.execute(text(f"DELETE FROM {tbl_sh} WHERE Date = :d"), {"d": date_str})
        except Exception:
            pass
        try:
            with eng.begin() as conn:
                conn.execute(text(f"DELETE FROM {tbl_gd} WHERE Date = :d"), {"d": date_str})
        except Exception:
            pass

    try:
        if not df_pbp.empty:
            # Drop lowercase x/y to avoid case-insensitive collisions with X/Y in MySQL
            dfp = df_pbp.copy()
            _apply_xg_nulling_rule(dfp)
            dfp = dfp.drop(columns=['x', 'y'], errors='ignore')
            dfp.to_sql(tbl_pbp, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_pbp)} rows to {tbl_pbp}")
        else:
            print("[mysql] df_pbp empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_pbp}: {e}", file=sys.stderr)

    try:
        if not df_shifts.empty:
            # Newer shifts exports include StrengthStateBucket; add the column if the table already exists.
            df_sh = df_shifts
            # Best-effort: add new columns when exporting to an existing table.
            if 'StrengthStateBucket' in df_sh.columns:
                ok = _ensure_mysql_column(tbl_sh, 'StrengthStateBucket', 'VARCHAR(16) NULL')
                if not ok:
                    df_sh = df_sh.drop(columns=['StrengthStateBucket'], errors='ignore')
            if 'StrengthStateRaw' in df_sh.columns:
                ok = _ensure_mysql_column(tbl_sh, 'StrengthStateRaw', 'VARCHAR(16) NULL')
                if not ok:
                    df_sh = df_sh.drop(columns=['StrengthStateRaw'], errors='ignore')
            for col in ('SkatersOnIceFor', 'SkatersOnIceAgainst', 'GoaliesOnIceFor', 'GoaliesOnIceAgainst'):
                if col in df_sh.columns:
                    ok = _ensure_mysql_column(tbl_sh, col, 'INT NULL')
                    if not ok:
                        df_sh = df_sh.drop(columns=[col], errors='ignore')
            df_sh.to_sql(tbl_sh, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_sh)} rows to {tbl_sh}")
        else:
            print("[mysql] df_shifts empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_sh}: {e}", file=sys.stderr)

    # GameData export
    try:
        if df_gamedata is not None and not df_gamedata.empty:
            df_gd = df_gamedata.copy()
            _apply_xg_nulling_rule(df_gd)
            df_gd.to_sql(tbl_gd, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_gd)} rows to {tbl_gd}")
        else:
            print("[mysql] df_gamedata empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_gd}: {e}", file=sys.stderr)


def run_player_projections_and_write_csv(csv_path: Optional[str] = None) -> str:
    """Deprecated: projections are written to Google Sheets (Sheets3) only."""
    raise RuntimeError(
        "player_projections.csv updates are disabled. Use run_player_projections_and_write_google_sheet(sheet_id=..., worksheet='Sheets3')."
    )


def _seasonstats_strength_bucket(strength: Any) -> str:
    s = str(strength or '').strip()
    if not s:
        return 'Other'
    s_low = s.lower()
    # Ignore empty-net and other labels here; they bucket as Other.
    if s_low.startswith('en'):
        return 'Other'

    # If StrengthState is already bucketed, pass through.
    if s_low in {'pp', 'sh', 'other'}:
        return s_low.upper() if s_low != 'other' else 'Other'
    # Parse generic "NvM" patterns and apply the SeasonStats rules.
    try:
        if 'v' in s_low:
            left, right = s_low.split('v', 1)
            my_s = int(left.strip())
            their_s = int(right.strip())
            # 5+ v 5+ => 5v5 (includes 6v5, 5v6, etc)
            if my_s >= 5 and their_s >= 5:
                return '5v5'
            # Opponent has 3/4 and we have the advantage => PP
            if their_s in (3, 4) and my_s > their_s:
                return 'PP'
            # We have 3/4 and opponent has the advantage => SH
            if my_s in (3, 4) and their_s > my_s:
                return 'SH'
            return 'Other'
    except Exception:
        pass
    # Backward-compatible exact matches
    if s_low == '5v5':
        return '5v5'
    return 'Other'


def _seasonstats_bucket_from_counts(*, my_skaters: int, their_skaters: int, my_goalies: int, their_goalies: int) -> str:
    """SeasonStats StrengthState bucket from on-ice counts.

    Rules (only when both goalies are in the net):
    - both teams have 5+ skaters -> 5v5
    - opponent has 3/4 skaters -> PP (if we have more skaters)
    - we have 3/4 skaters -> SH (if opponent has more skaters)
    Everything else -> Other.
    """
    try:
        mg = int(my_goalies or 0)
        tg = int(their_goalies or 0)
        ms = int(my_skaters or 0)
        ts = int(their_skaters or 0)
    except Exception:
        return 'Other'

    if mg >= 1 and tg >= 1 and ms >= 5 and ts >= 5:
        return '5v5'
    if mg >= 1 and tg >= 1 and ts in (3, 4) and ms > ts:
        return 'PP'
    if mg >= 1 and tg >= 1 and ms in (3, 4) and ts > ms:
        return 'SH'
    return 'Other'


_INVERT_STRENGTH = {
    '5v4': '4v5',
    '5v3': '3v5',
    '4v3': '3v4',
    '4v5': '5v4',
    '3v5': '5v3',
    '3v4': '4v3',
}


def _invert_strength(strength: Any) -> str:
    s = str(strength or '').strip().lower()
    return _INVERT_STRENGTH.get(s, s)


def _to_ids(v: Any) -> List[int]:
    if v is None:
        return []
    if isinstance(v, list):
        out: List[int] = []
        for x in v:
            if x is None:
                continue
            try:
                out.append(int(x))
            except Exception:
                continue
        return out
    s = str(v)
    if not s:
        return []
    try:
        if s.strip().startswith('[') and s.strip().endswith(']'):
            arr = json.loads(s)
            if isinstance(arr, list):
                out2: List[int] = []
                for x in arr:
                    if x is None:
                        continue
                    try:
                        out2.append(int(x))
                    except Exception:
                        continue
                return out2
    except Exception:
        pass
    parts = re.split(r'[\,\|;\s]+', s)
    out3: List[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            out3.append(int(part))
        except Exception:
            continue
    return out3


def compute_seasonstats_day(
    *,
    df_pbp: pd.DataFrame,
    df_shifts: pd.DataFrame,
    season: str,
    date_str: str,
) -> pd.DataFrame:
    """Compute per-day seasonstats contributions for one season from PBP + Shifts.

    Produces one row per (Season, SeasonState, StrengthState bucket, PlayerID).
    StrengthState buckets: 5v5, PP, SH, Other.
    """
    # Required output columns (no plusMinus / blockedShots)
    out_cols = [
        'Date',
        'Season', 'SeasonState', 'StrengthState', 'PlayerID', 'Position', 'GP', 'TOI',
        'iGoals', 'Assists1', 'Assists2', 'iCorsi', 'iFenwick', 'iShots', 'ixG_F', 'ixG_S', 'ixG_F2',
        'PIM_taken', 'PIM_drawn', 'Hits', 'Takeaways', 'Giveaways', 'SO_Goal', 'SO_Attempt',
        'CA', 'CF', 'FA', 'FF', 'SA', 'SF', 'GA', 'GF',
        'xGA_F', 'xGF_F', 'xGA_S', 'xGF_S', 'xGA_F2', 'xGF_F2',
        'PIM_for', 'PIM_against',
    ]
    strength_buckets = ['5v5', 'PP', 'SH', 'Other']

    # SeasonState per game (from PBP rows)
    game_state: Dict[int, str] = {}
    if not df_pbp.empty and 'GameID' in df_pbp.columns:
        try:
            tmp = df_pbp[['GameID']].copy()
            tmp['SeasonState'] = df_pbp.get('SeasonState')
            tmp = tmp.dropna(subset=['GameID'])
            for gid, grp in tmp.groupby('GameID'):
                ss = None
                try:
                    ss = (grp['SeasonState'].dropna().astype(str).head(1).tolist() or [None])[0]
                except Exception:
                    ss = None
                if ss:
                    try:
                        game_state[int(gid)] = str(ss).strip()
                    except Exception:
                        continue
        except Exception:
            game_state = {}

    # Base player set + Position from shifts
    shifts = df_shifts.copy() if df_shifts is not None else pd.DataFrame()
    if shifts.empty:
        return pd.DataFrame(columns=out_cols)

    # Normalize shift columns
    for col in ['GameID', 'PlayerID', 'StrengthState', 'Position', 'Duration']:
        if col not in shifts.columns:
            shifts[col] = None
    shifts['PlayerID'] = pd.to_numeric(shifts['PlayerID'], errors='coerce')
    shifts = shifts.dropna(subset=['PlayerID'])
    shifts['PlayerID'] = shifts['PlayerID'].astype(int)
    shifts['GameID'] = pd.to_numeric(shifts['GameID'], errors='coerce')
    shifts = shifts.dropna(subset=['GameID'])
    shifts['GameID'] = shifts['GameID'].astype(int)
    shifts['Duration'] = pd.to_numeric(shifts['Duration'], errors='coerce').fillna(0.0).astype(float)
    shifts['Position'] = shifts['Position'].astype(str).str.strip().str.upper().str[:1]
    shifts.loc[shifts['Position'].isin({'C', 'L', 'R'}), 'Position'] = 'F'
    shifts.loc[~shifts['Position'].isin({'F', 'D', 'G'}), 'Position'] = ''
    shifts['SeasonState'] = shifts['GameID'].map(lambda gid: game_state.get(int(gid), 'regular'))
    if 'StrengthStateBucket' in shifts.columns:
        # Prefer upstream bucketing (computed from skater/goalie counts in /api/game/<id>/shifts)
        shifts['StrengthStateBucket'] = (
            shifts['StrengthStateBucket']
            .astype(str)
            .map(lambda x: x.strip() if x is not None else '')
            .map(lambda x: x if x in {'5v5', 'PP', 'SH', 'Other'} else 'Other')
        )
    else:
        # Fallback to legacy parsing from raw "NvM" strings
        shifts['StrengthStateBucket'] = shifts['StrengthState'].map(_seasonstats_strength_bucket)

    # TOI per bucket (minutes)
    toi = (
        shifts.groupby(['SeasonState', 'StrengthStateBucket', 'PlayerID', 'Position'], dropna=False)['Duration']
        .sum()
        .reset_index()
    )
    toi['TOI'] = (toi['Duration'] / 60.0).astype(float)
    toi = toi.drop(columns=['Duration'])

    # GP per player per season state (unique games)
    gp = (
        shifts.groupby(['SeasonState', 'PlayerID'], dropna=False)['GameID']
        .nunique()
        .reset_index()
        .rename(columns={'GameID': 'GP'})
    )

    # Prepare a skeleton of rows: every player x seasonState x bucket
    players = shifts[['SeasonState', 'PlayerID', 'Position']].drop_duplicates()
    rows: List[Dict[str, Any]] = []
    for rec in players.to_dict(orient='records'):
        ss = str(rec.get('SeasonState') or 'regular').strip() or 'regular'
        pid = int(rec.get('PlayerID'))
        pos = str(rec.get('Position') or '').strip().upper()[:1]
        for b in strength_buckets:
            rows.append({
                'Date': date_str,
                'Season': season,
                'SeasonState': ss,
                'StrengthState': b,
                'PlayerID': pid,
                'Position': ('F' if pos in {'C', 'L', 'R'} else pos),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=out_cols)

    # Merge TOI (minutes) and GP from shifts. Don't pre-create these columns to avoid merge suffix collisions.
    out = out.merge(
        toi.rename(columns={'StrengthStateBucket': 'StrengthState'}),
        on=['SeasonState', 'StrengthState', 'PlayerID', 'Position'],
        how='left',
    )
    out['TOI'] = pd.to_numeric(out.get('TOI'), errors='coerce').fillna(0.0)

    out = out.merge(gp, on=['SeasonState', 'PlayerID'], how='left')
    out['GP'] = pd.to_numeric(out.get('GP'), errors='coerce').fillna(0).astype(int)

    # Build PBP-based aggregates using dict accumulator keyed by (SeasonState, StrengthBucket, PlayerID)
    acc: Dict[Tuple[str, str, int], Dict[str, Any]] = {}

    if df_pbp is not None and not df_pbp.empty:
        pbp = df_pbp.copy()
        for col in [
            'GameID', 'SeasonState', 'Period', 'StrengthState', 'Event',
            'Player1_ID', 'Player2_ID', 'Player3_ID',
            'Corsi', 'Fenwick', 'Shot', 'Goal', 'xG_F', 'xG_S', 'xG_F2',
            'PEN_duration', 'Venue',
            'Home_Forwards_ID', 'Home_Defenders_ID', 'Home_Goalie_ID',
            'Away_Forwards_ID', 'Away_Defenders_ID', 'Away_Goalie_ID',
        ]:
            if col not in pbp.columns:
                pbp[col] = None

        pbp['SeasonState'] = pbp['SeasonState'].astype(str).str.strip().replace({'': 'regular'})
        pbp.loc[pbp['SeasonState'].isna(), 'SeasonState'] = 'regular'
        pbp['Period'] = pd.to_numeric(pbp['Period'], errors='coerce').fillna(0).astype(int)
        pbp['Corsi'] = pd.to_numeric(pbp['Corsi'], errors='coerce').fillna(0).astype(int)
        pbp['Fenwick'] = pd.to_numeric(pbp['Fenwick'], errors='coerce').fillna(0).astype(int)
        pbp['Shot'] = pd.to_numeric(pbp['Shot'], errors='coerce').fillna(0).astype(int)
        pbp['Goal'] = pd.to_numeric(pbp['Goal'], errors='coerce').fillna(0).astype(int)
        pbp['xG_F'] = pd.to_numeric(pbp['xG_F'], errors='coerce').fillna(0.0).astype(float)
        pbp['xG_S'] = pd.to_numeric(pbp['xG_S'], errors='coerce').fillna(0.0).astype(float)
        pbp['xG_F2'] = pd.to_numeric(pbp['xG_F2'], errors='coerce').fillna(0.0).astype(float)
        pbp['PEN_duration'] = pd.to_numeric(pbp['PEN_duration'], errors='coerce').fillna(0.0).astype(float)

        for _, row in pbp.iterrows():
            ss = str(row.get('SeasonState') or 'regular').strip() or 'regular'
            period = int(row.get('Period') or 0)

            # Strength bucketing (prefer on-ice counts + goalie presence; fallback to raw StrengthState strings)
            venue = str(row.get('Venue') or '').strip()
            try:
                home_f = _to_ids(row.get('Home_Forwards_ID'))
                home_d = _to_ids(row.get('Home_Defenders_ID'))
                home_g = _to_ids(row.get('Home_Goalie_ID'))
                away_f = _to_ids(row.get('Away_Forwards_ID'))
                away_d = _to_ids(row.get('Away_Defenders_ID'))
                away_g = _to_ids(row.get('Away_Goalie_ID'))
                home_s = len(home_f) + len(home_d)
                away_s = len(away_f) + len(away_d)
                home_goalies = len(home_g)
                away_goalies = len(away_g)

                if venue == 'Home':
                    bucket_for = _seasonstats_bucket_from_counts(
                        my_skaters=home_s,
                        their_skaters=away_s,
                        my_goalies=home_goalies,
                        their_goalies=away_goalies,
                    )
                    bucket_against = _seasonstats_bucket_from_counts(
                        my_skaters=away_s,
                        their_skaters=home_s,
                        my_goalies=away_goalies,
                        their_goalies=home_goalies,
                    )
                elif venue == 'Away':
                    bucket_for = _seasonstats_bucket_from_counts(
                        my_skaters=away_s,
                        their_skaters=home_s,
                        my_goalies=away_goalies,
                        their_goalies=home_goalies,
                    )
                    bucket_against = _seasonstats_bucket_from_counts(
                        my_skaters=home_s,
                        their_skaters=away_s,
                        my_goalies=home_goalies,
                        their_goalies=away_goalies,
                    )
                else:
                    bucket_for = None
                    bucket_against = None
            except Exception:
                bucket_for = None
                bucket_against = None

            if not bucket_for or not bucket_against:
                strength_raw = row.get('StrengthState')
                bucket_for = _seasonstats_strength_bucket(strength_raw)
                bucket_against = _seasonstats_strength_bucket(_invert_strength(strength_raw))

            is_shootout = (period == 5) and (ss == 'regular')
            shooter = row.get('Player1_ID')
            a1 = row.get('Player2_ID')
            a2 = row.get('Player3_ID')

            try:
                shooter_id = int(shooter) if shooter is not None else None
            except Exception:
                shooter_id = None
            try:
                a1_id = int(a1) if a1 is not None else None
            except Exception:
                a1_id = None
            try:
                a2_id = int(a2) if a2 is not None else None
            except Exception:
                a2_id = None

            corsi = int(row.get('Corsi') or 0)
            fenwick = int(row.get('Fenwick') or 0)
            shot = int(row.get('Shot') or 0)
            goal = int(row.get('Goal') or 0)
            xgf = float(row.get('xG_F') or 0.0)
            xgs = float(row.get('xG_S') or 0.0)
            xgf2 = float(row.get('xG_F2') or 0.0)
            pen_min = float(row.get('PEN_duration') or 0.0)
            ev = str(row.get('Event') or '').strip().lower()

            # Individual (non-shootout) stats
            if shooter_id is not None and not is_shootout:
                key = (ss, bucket_for, shooter_id)
                d = acc.setdefault(key, {})
                d['iGoals'] = d.get('iGoals', 0) + goal
                d['iCorsi'] = d.get('iCorsi', 0) + corsi
                d['iFenwick'] = d.get('iFenwick', 0) + fenwick
                d['iShots'] = d.get('iShots', 0) + shot
                d['ixG_F'] = d.get('ixG_F', 0.0) + xgf
                d['ixG_S'] = d.get('ixG_S', 0.0) + xgs
                d['ixG_F2'] = d.get('ixG_F2', 0.0) + xgf2
                if ev == 'hit':
                    d['Hits'] = d.get('Hits', 0) + 1
                elif ev == 'takeaway':
                    d['Takeaways'] = d.get('Takeaways', 0) + 1
                elif ev == 'giveaway':
                    d['Giveaways'] = d.get('Giveaways', 0) + 1

            # Assists (non-shootout)
            if a1_id is not None and not is_shootout and goal == 1:
                key = (ss, bucket_for, a1_id)
                d = acc.setdefault(key, {})
                d['Assists1'] = d.get('Assists1', 0) + 1
            if a2_id is not None and not is_shootout and goal == 1:
                key = (ss, bucket_for, a2_id)
                d = acc.setdefault(key, {})
                d['Assists2'] = d.get('Assists2', 0) + 1

            # Shootout stats (regular season, period 5)
            if shooter_id is not None and is_shootout:
                key = (ss, bucket_for, shooter_id)
                d = acc.setdefault(key, {})
                d['SO_Attempt'] = d.get('SO_Attempt', 0) + fenwick
                d['SO_Goal'] = d.get('SO_Goal', 0) + goal

            # Penalties: individual taken/drawn (all non-zero durations)
            if pen_min > 0:
                if shooter_id is not None:
                    key = (ss, bucket_for, shooter_id)
                    d = acc.setdefault(key, {})
                    d['PIM_taken'] = d.get('PIM_taken', 0.0) + pen_min
                if a1_id is not None:
                    key = (ss, bucket_for, a1_id)
                    d = acc.setdefault(key, {})
                    d['PIM_drawn'] = d.get('PIM_drawn', 0.0) + pen_min

            # On-ice stats (exclude shootout)
            if not is_shootout:
                home_list = _to_ids(row.get('Home_Forwards_ID')) + _to_ids(row.get('Home_Defenders_ID')) + _to_ids(row.get('Home_Goalie_ID'))
                away_list = _to_ids(row.get('Away_Forwards_ID')) + _to_ids(row.get('Away_Defenders_ID')) + _to_ids(row.get('Away_Goalie_ID'))
                if not home_list and not away_list:
                    continue

                if venue == 'Home':
                    players_for, players_against = home_list, away_list
                elif venue == 'Away':
                    players_for, players_against = away_list, home_list
                else:
                    continue

                for pid in players_for:
                    key = (ss, bucket_for, int(pid))
                    d = acc.setdefault(key, {})
                    d['CF'] = d.get('CF', 0) + corsi
                    d['FF'] = d.get('FF', 0) + fenwick
                    d['SF'] = d.get('SF', 0) + shot
                    d['GF'] = d.get('GF', 0) + goal
                    d['xGF_F'] = d.get('xGF_F', 0.0) + xgf
                    d['xGF_S'] = d.get('xGF_S', 0.0) + xgs
                    d['xGF_F2'] = d.get('xGF_F2', 0.0) + xgf2
                    if pen_min > 0:
                        d['PIM_against'] = d.get('PIM_against', 0.0) + pen_min

                for pid in players_against:
                    key = (ss, bucket_against, int(pid))
                    d = acc.setdefault(key, {})
                    d['CA'] = d.get('CA', 0) + corsi
                    d['FA'] = d.get('FA', 0) + fenwick
                    d['SA'] = d.get('SA', 0) + shot
                    d['GA'] = d.get('GA', 0) + goal
                    d['xGA_F'] = d.get('xGA_F', 0.0) + xgf
                    d['xGA_S'] = d.get('xGA_S', 0.0) + xgs
                    d['xGA_F2'] = d.get('xGA_F2', 0.0) + xgf2
                    if pen_min > 0:
                        d['PIM_for'] = d.get('PIM_for', 0.0) + pen_min

    # Apply accumulator into output df
    if acc:
        recs: List[Dict[str, Any]] = []
        for (ss, bucket, pid), d in acc.items():
            rec = {'SeasonState': ss, 'StrengthState': bucket, 'PlayerID': pid}
            rec.update(d)
            recs.append(rec)
        a = pd.DataFrame(recs)
        out = out.merge(a, on=['SeasonState', 'StrengthState', 'PlayerID'], how='left')

    # Fill defaults and enforce dtypes
    numeric_defaults: Dict[str, Any] = {
        'GP': 0,
        'TOI': 0.0,
        'iGoals': 0,
        'Assists1': 0,
        'Assists2': 0,
        'iCorsi': 0,
        'iFenwick': 0,
        'iShots': 0,
        'ixG_F': 0.0,
        'ixG_S': 0.0,
        'ixG_F2': 0.0,
        'PIM_taken': 0.0,
        'PIM_drawn': 0.0,
        'Hits': 0,
        'Takeaways': 0,
        'Giveaways': 0,
        'SO_Goal': 0,
        'SO_Attempt': 0,
        'CA': 0,
        'CF': 0,
        'FA': 0,
        'FF': 0,
        'SA': 0,
        'SF': 0,
        'GA': 0,
        'GF': 0,
        'xGA_F': 0.0,
        'xGF_F': 0.0,
        'xGA_S': 0.0,
        'xGF_S': 0.0,
        'xGA_F2': 0.0,
        'xGF_F2': 0.0,
        'PIM_for': 0.0,
        'PIM_against': 0.0,
    }
    for k, default in numeric_defaults.items():
        if k not in out.columns:
            out[k] = default
        else:
            out[k] = pd.to_numeric(out[k], errors='coerce').fillna(default)

    int_cols = ['iGoals', 'Assists1', 'Assists2', 'iCorsi', 'iFenwick', 'iShots', 'Hits', 'Takeaways', 'Giveaways', 'SO_Goal', 'SO_Attempt',
                'CA', 'CF', 'FA', 'FF', 'SA', 'SF', 'GA', 'GF']
    for c in int_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0).astype(int)
    float_cols = [c for c in out_cols if c not in {'Date', 'Season', 'SeasonState', 'StrengthState', 'PlayerID', 'Position'} and c not in int_cols and c != 'GP']
    for c in float_cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors='coerce').fillna(0.0).astype(float)

    out = cast(pd.DataFrame, out.loc[:, out_cols])
    return out


def update_seasonstats_google_sheet(*, season: str, sheet_id: str, worksheet: str = 'Sheets6') -> None:
    """Rebuild full-season seasonstats for `season` from MySQL and overwrite a Google Sheet worksheet."""
    eng = _create_mysql_engine('ro')
    if eng is None:
        raise RuntimeError('MySQL engine not available for seasonstats')

    tbl_pbp = f"nhl_{season}_pbp"
    tbl_sh = f"nhl_{season}_shifts"

    # Load only needed columns to keep memory reasonable.
    pbp_cols = [
        'GameID', 'SeasonState', 'Period', 'StrengthState', 'Event',
        'Player1_ID', 'Player2_ID', 'Player3_ID',
        'Corsi', 'Fenwick', 'Shot', 'Goal', 'xG_F', 'xG_S', 'xG_F2',
        'PEN_duration', 'Venue',
        'Home_Forwards_ID', 'Home_Defenders_ID', 'Home_Goalie_ID',
        'Away_Forwards_ID', 'Away_Defenders_ID', 'Away_Goalie_ID',
    ]
    sh_cols = ['GameID', 'PlayerID', 'StrengthState', 'Position', 'Duration']

    try:
        df_pbp = pd.read_sql_query(
            f"SELECT {', '.join(pbp_cols)} FROM {tbl_pbp}",
            con=eng,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {tbl_pbp} for seasonstats: {e}")

    try:
        df_shifts = pd.read_sql_query(
            f"SELECT {', '.join(sh_cols)} FROM {tbl_sh}",
            con=eng,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {tbl_sh} for seasonstats: {e}")

    # Compute totals using the same aggregation logic.
    df_tot = compute_seasonstats_day(
        df_pbp=df_pbp,
        df_shifts=df_shifts,
        season=str(season),
        date_str='TOTAL',
    )
    if 'Date' in df_tot.columns:
        df_tot = cast(pd.DataFrame, df_tot.drop(columns=['Date']))

    _write_dataframe_to_google_sheet(df_tot, sheet_id=str(sheet_id).strip(), worksheet=str(worksheet).strip())
    print(f"[sheets] wrote seasonstats to sheetId={sheet_id} worksheet={worksheet}")


def _load_google_service_account_info() -> Dict[str, Any]:
    """Load Google service account JSON from environment.

    Supports either:
      - GOOGLE_SERVICE_ACCOUNT_JSON_PATH: path to a JSON key file
      - GOOGLE_SERVICE_ACCOUNT_JSON: raw JSON string
      - GOOGLE_SERVICE_ACCOUNT_JSON_B64: base64-encoded JSON string
    """
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
    else:
        raw = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON')
    raw_b64 = os.getenv('GOOGLE_SERVICE_ACCOUNT_JSON_B64')
    if raw_b64:
        try:
            import base64
            s = str(raw_b64).strip()
            # Some shells/UI flows add surrounding quotes or newlines.
            if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
                s = s[1:-1]
            s = ''.join(s.split())  # drop all whitespace
            # Fix missing base64 padding (length must be multiple of 4)
            pad = (-len(s)) % 4
            if pad:
                s = s + ('=' * pad)
            try:
                raw = base64.b64decode(s.encode('utf-8'), validate=False).decode('utf-8')
            except Exception:
                # Some users paste URL-safe base64; try that too.
                raw = base64.urlsafe_b64decode(s.encode('utf-8')).decode('utf-8')
        except Exception as e:
            raise RuntimeError(f"Invalid GOOGLE_SERVICE_ACCOUNT_JSON_B64: {e}")
    if not raw:
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON_PATH, GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_SERVICE_ACCOUNT_JSON."
        )
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid Google service account JSON: {e}")


def _write_dataframe_to_google_sheet(df: pd.DataFrame, *, sheet_id: str, worksheet: str) -> None:
    """Overwrite a worksheet with the contents of df (header + rows)."""
    # Optional dependency: only required when using Sheets export
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Google Sheets dependencies missing: {e}. Install: pip install gspread google-auth"
        )

    info = _load_google_service_account_info()
    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive',
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except Exception:
        # Create if missing
        rows = max(2, int(len(df) + 1))
        cols = max(1, int(len(df.columns)))
        ws = sh.add_worksheet(title=worksheet, rows=rows, cols=cols)

    # Prepare values
    df_out = df.copy()

    # Stamp the export time in-worksheet as a column so consumers can see refresh time.
    # Keep the rest of the schema stable (existing columns preserved and order retained aside from this prepend).
    try:
        ts = datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'
    except Exception:
        ts = ''
    try:
        if 'TimestampUTC' in df_out.columns:
            df_out['TimestampUTC'] = ts
        else:
            df_out.insert(0, 'TimestampUTC', ts)
    except Exception:
        pass

    # Replace NaN/NaT with empty string for Sheets
    df_out = df_out.where(pd.notnull(df_out), '')

    def _cell(v: Any) -> Any:
        try:
            # Convert numpy scalars to python
            if hasattr(v, 'item'):
                v = v.item()
        except Exception:
            pass
        # Make datetimes readable
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return v

    values = [list(df_out.columns)]
    for row in df_out.itertuples(index=False, name=None):
        values.append([_cell(v) for v in row])

    # Overwrite worksheet
    ws.clear()
    update_fn = getattr(ws, 'update')
    try:
        update_fn(range_name='A1', values=values)
    except TypeError:
        # Older gspread signature
        update_fn('A1', values)


def run_player_projections_and_write_google_sheet(
    *, sheet_id: str, worksheet: str = 'Sheets3'
) -> None:
    """Run Player_Projections() and overwrite a Google Sheet worksheet."""
    eng = _create_mysql_engine('rw')
    if eng is None:
        raise RuntimeError("MySQL engine not available for projections")

    # Execute stored procedure
    try:
        with eng.begin() as conn:
            conn.execute(text("CALL Player_Projections()"))
        print("[mysql] executed stored procedure Player_Projections()")
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to execute Player_Projections(): {e}")

    # Load table
    try:
        df_proj = pd.read_sql("SELECT * FROM nhl_player_projections", con=eng)
        print(f"[mysql] loaded {len(df_proj)} rows from nhl_player_projections")
    except SQLAlchemyError as e:
        raise RuntimeError(f"Failed to query nhl_player_projections: {e}")

    _write_dataframe_to_google_sheet(df_proj, sheet_id=sheet_id, worksheet=worksheet)
    print(f"[sheets] wrote projections to sheetId={sheet_id} worksheet={worksheet}")


def _resolve_default_sheets_id(*candidates: Optional[str]) -> str:
    for c in candidates:
        if c and str(c).strip():
            return str(c).strip()
    return ''


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fetch PBP and Shifts dataframes for a date')
    parser.add_argument('--date', type=_validate_date, help='Date in YYYY-MM-DD')
    parser.add_argument('--export', action='store_true', help='Export to MySQL after fetching')
    parser.add_argument('--season', default='20252026', help='Season code for table names, e.g., 20252026')
    parser.add_argument('--replace-date', action='store_true', help='Pre-delete rows for this date before insert (idempotent loads)')
    parser.add_argument(
        '--seasonstats-sheets-id',
        default=_resolve_default_sheets_id(os.getenv('GOOGLE_SHEETS_ID'), os.getenv('SEASONSTATS_SHEETS_ID')),
        help='Google Sheets document id to write seasonstats into (default from GOOGLE_SHEETS_ID or SEASONSTATS_SHEETS_ID)'
    )
    parser.add_argument('--seasonstats-worksheet', default='Sheets6', help='Worksheet/tab name for seasonstats output')
    parser.add_argument('--seasonstats-only', action='store_true', help='Skip fetch/export; rebuild seasonstats from MySQL and write to Google Sheets')
    # Projections output options (Sheets3 by default)
    parser.add_argument(
        '--projections-sheets-id',
        default=_resolve_default_sheets_id(os.getenv('GOOGLE_SHEETS_ID'), os.getenv('PROJECTIONS_SHEET_ID')),
        help='Google Sheets document id to write nhl_player_projections into (default from GOOGLE_SHEETS_ID or PROJECTIONS_SHEET_ID)'
    )
    parser.add_argument('--projections-worksheet', default='Sheets3', help='Worksheet/tab name for projections output')
    # RAPM/context post-step options (requires --export)
    parser.add_argument('--run-rapm', action='store_true', help='After export, rebuild RAPM + context and write to MySQL + Google Sheets')
    parser.add_argument('--rapm-sheets-id', help='Google Sheets document id to write RAPM/context into (Sheets4/Sheets5)')
    parser.add_argument('--rapm-worksheet', default='Sheets4', help='Worksheet/tab name for RAPM output')
    parser.add_argument('--context-worksheet', default='Sheets5', help='Worksheet/tab name for context output')
    # Lineup scraping options
    parser.add_argument('--lineup-url', help='DailyFaceoff line combinations URL for a team (e.g., https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations)')
    parser.add_argument('--lineup-save', action='store_true', help='When using --lineup-url, save mapped lineup JSON to app/static/lineup_<TEAM>.json')
    args = parser.parse_args(argv)

    # Standalone seasonstats mode (no fetch_day required)
    if bool(args.seasonstats_only):
        if str(args.season).strip() != '20252026':
            print('[seasonstats] --seasonstats-only is currently intended for --season 20252026', file=sys.stderr)
        ss_id = _resolve_default_sheets_id(
            args.seasonstats_sheets_id,
            os.getenv('GOOGLE_SHEETS_ID'),
            os.getenv('SEASONSTATS_SHEETS_ID'),
        )
        if not ss_id:
            print('[error] seasonstats requires a Sheets doc id via --seasonstats-sheets-id or GOOGLE_SHEETS_ID', file=sys.stderr)
            return 8
        try:
            update_seasonstats_google_sheet(
                season=str(args.season).strip(),
                sheet_id=ss_id,
                worksheet=str(args.seasonstats_worksheet or 'Sheets6').strip(),
            )
            return 0
        except Exception as e:
            print(f"[error] seasonstats-only failed: {e}", file=sys.stderr)
            return 9

    if not args.date:
        print('[error] --date is required unless --seasonstats-only is used', file=sys.stderr)
        return 2

    date_str = args.date
    print(f'Fetching games for {date_str}...')
    try:
        df_pbp, df_shifts, df_gamedata = fetch_day(date_str, with_xg=True)
    except Exception as e:
        print(f'[error] {e}', file=sys.stderr)
        return 2

    print(f'Games: PBP rows={len(df_pbp)} | Shifts rows={len(df_shifts)} | GameData rows={len(df_gamedata)}')
    
    if args.export:
        try:
            export_to_mysql(
                df_pbp,
                df_shifts,
                df_gamedata,
                season=str(args.season),
                date_str=date_str,
                replace_date=bool(args.replace_date)
            )
        except Exception as e:
            print(f"[error] export failed: {e}", file=sys.stderr)
            return 3

        # Post-step: rebuild seasonstats from MySQL and write to Google Sheets (Sheets6)
        if str(args.season).strip() == '20252026':
            # Prefer one shared doc id (GOOGLE_SHEETS_ID), then explicit seasonstats id, then other step ids.
            ss_id = _resolve_default_sheets_id(
                os.getenv('GOOGLE_SHEETS_ID'),
                args.seasonstats_sheets_id,
                args.rapm_sheets_id,
                args.projections_sheets_id,
            )
            if ss_id:
                try:
                    update_seasonstats_google_sheet(
                        season=str(args.season).strip(),
                        sheet_id=ss_id,
                        worksheet=str(args.seasonstats_worksheet or 'Sheets6').strip(),
                    )
                except Exception as e:
                    print(f"[warn] seasonstats sheets step failed: {e}", file=sys.stderr)
            else:
                print('[seasonstats] skipped (no Sheets doc id; set GOOGLE_SHEETS_ID)', file=sys.stderr)
        # After exporting, run projections and write to Google Sheets (Sheets3)
        try:
            proj_id = _resolve_default_sheets_id(
                args.projections_sheets_id,
                os.getenv('GOOGLE_SHEETS_ID'),
                os.getenv('PROJECTIONS_SHEET_ID'),
            )
            if not proj_id:
                raise RuntimeError('projections requires a Sheets doc id via --projections-sheets-id or GOOGLE_SHEETS_ID')
            run_player_projections_and_write_google_sheet(
                sheet_id=str(proj_id).strip(),
                worksheet=str(args.projections_worksheet or 'Sheets3').strip(),
            )
        except Exception as e:
            print(f"[error] projections post-step failed: {e}", file=sys.stderr)
            return 4

        # Optional: run RAPM + context refresh as a post-step.
        if bool(args.run_rapm):
            if not args.rapm_sheets_id:
                print('[error] --run-rapm requires --rapm-sheets-id', file=sys.stderr)
                return 6
            try:
                rapm_script = os.path.join(REPO_ROOT, 'scripts', 'rapm.py')
                rapm_cmd = [
                    sys.executable,
                    rapm_script,
                    '--season',
                    str(args.season).strip(),
                    '--sheets-id',
                    str(args.rapm_sheets_id).strip(),
                    '--worksheet',
                    str(args.rapm_worksheet or 'Sheets4').strip(),
                    '--context-worksheet',
                    str(args.context_worksheet or 'Sheets5').strip(),
                ]
                print('[rapm] starting RAPM + context refresh...')
                res = subprocess.run(rapm_cmd, cwd=REPO_ROOT, capture_output=True, text=True)
                out = (res.stdout or '') + ("\n" + res.stderr if res.stderr else '')
                print(out)
                if res.returncode != 0:
                    print(f'[error] rapm post-step failed with code {res.returncode}', file=sys.stderr)
                    return 7
            except Exception as e:
                print(f'[error] rapm post-step failed: {e}', file=sys.stderr)
                return 7

    # Optional: scrape DailyFaceoff lineup and map to PlayerIDs
    if args.lineup_url:
        try:
            lineup = scrape_dailyfaceoff_lineup(args.lineup_url)
            team_abbrev = infer_team_from_dailyfaceoff_url(args.lineup_url)
            season_int = int(str(args.season)) if args.season else None
            roster = fetch_recent_roster(team_abbrev, season_int)
            mapped = map_lineup_to_player_ids(lineup, roster, team_abbrev)
            # Print a compact summary
            print('\nExpected lineup (mapped to PlayerIDs):')
            import json as _json
            print(_json.dumps(mapped, ensure_ascii=False, indent=2))
            if args.lineup_save:
                out_path = os.path.join(REPO_ROOT, 'app', 'static', f'lineup_{team_abbrev}.json')
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(_json.dumps(mapped, ensure_ascii=False))
                print(f"[file] wrote lineup JSON: {out_path}")
        except Exception as e:
            print(f"[error] lineup scrape/map failed: {e}", file=sys.stderr)
            return 5
    # Show small samples to confirm
    if not df_pbp.empty:
        print('\nPBP sample:')
        print(df_pbp.head(5).to_string(index=False))
    if not df_shifts.empty:
        print('\nShifts sample:')
        print(df_shifts.head(5).to_string(index=False))
    if not df_gamedata.empty:
        print('\nGameData sample:')
        print(df_gamedata.head(10).to_string(index=False))

    # Keep dataframes in memory for interactive use; no file writes here
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
