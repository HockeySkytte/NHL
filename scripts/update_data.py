r"""
update_data.py

Fetch play-by-play and shifts dataframes for all NHL games on a given date.

- Reuses the app's existing logic (routes) via an internal Flask test client,
  so we get identical parsing/mapping as the web app (including robust Shifts parsing).
- Avoids loading heavy xG models by passing xg=0 to the PBP route.

Usage (PowerShell):
    pwsh> & .\.venv\Scripts\python.exe .\scripts\update_data.py --date 2025-10-12

Returns three pandas DataFrames: combined PBP, combined Shifts, and per-game Player GameData for the given date.
When run as a script, prints a short summary and shows .head() samples.
"""
from __future__ import annotations

import os
import sys
import argparse
from datetime import datetime
from typing import List, Tuple, Optional, Dict
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

    try:
        if not df_pbp.empty:
            # Drop lowercase x/y to avoid case-insensitive collisions with X/Y in MySQL
            dfp = df_pbp.copy()
            dfp = dfp.drop(columns=['x', 'y'], errors='ignore')
            # Optional: pre-delete by date for idempotent loads
            if replace_date and date_str is not None:
                try:
                    with eng.begin() as conn:
                        conn.execute(text(f"DELETE FROM {tbl_pbp} WHERE Date = :d"), {"d": date_str})
                except Exception:
                    # Table may not exist on first run; ignore
                    pass
            dfp.to_sql(tbl_pbp, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_pbp)} rows to {tbl_pbp}")
        else:
            print("[mysql] df_pbp empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_pbp}: {e}", file=sys.stderr)

    try:
        if not df_shifts.empty:
            if replace_date and date_str is not None:
                try:
                    with eng.begin() as conn:
                        conn.execute(text(f"DELETE FROM {tbl_sh} WHERE Date = :d"), {"d": date_str})
                except Exception:
                    pass
            df_shifts.to_sql(tbl_sh, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_shifts)} rows to {tbl_sh}")
        else:
            print("[mysql] df_shifts empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_sh}: {e}", file=sys.stderr)

    # GameData export
    try:
        if df_gamedata is not None and not df_gamedata.empty:
            if replace_date and date_str is not None:
                try:
                    with eng.begin() as conn:
                        conn.execute(text(f"DELETE FROM {tbl_gd} WHERE Date = :d"), {"d": date_str})
                except Exception:
                    # table may not exist yet
                    pass
            df_gd = df_gamedata.copy()
            df_gd.to_sql(tbl_gd, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
            print(f"[mysql] wrote {len(df_gd)} rows to {tbl_gd}")
        else:
            print("[mysql] df_gamedata empty; nothing to write")
    except SQLAlchemyError as e:
        print(f"[error] writing {tbl_gd}: {e}", file=sys.stderr)


def run_player_projections_and_write_csv(csv_path: Optional[str] = None) -> str:
    """Run stored procedure Player_Projections() and dump nhl_player_projections to CSV.

    Returns the absolute path to the written CSV.
    """
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

    # Write CSV to static folder
    if csv_path is None:
        # Save under app/static per request
        csv_path = os.path.join(REPO_ROOT, 'app', 'static', 'player_projections.csv')
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df_proj.to_csv(csv_path, index=False)
        print(f"[file] wrote projections CSV: {csv_path}")
        return os.path.abspath(csv_path)
    except Exception as e:
        raise RuntimeError(f"Failed to write projections CSV: {e}")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Fetch PBP and Shifts dataframes for a date')
    parser.add_argument('--date', required=True, type=_validate_date, help='Date in YYYY-MM-DD')
    parser.add_argument('--export', action='store_true', help='Export to MySQL after fetching')
    parser.add_argument('--no-xg', action='store_true', help='Do not compute xG (faster)')
    parser.add_argument('--season', default='20252026', help='Season code for table names, e.g., 20252026')
    parser.add_argument('--replace-date', action='store_true', help='Pre-delete rows for this date before insert (idempotent loads)')
    # Lineup scraping options
    parser.add_argument('--lineup-url', help='DailyFaceoff line combinations URL for a team (e.g., https://www.dailyfaceoff.com/teams/anaheim-ducks/line-combinations)')
    parser.add_argument('--lineup-save', action='store_true', help='When using --lineup-url, save mapped lineup JSON to app/static/lineup_<TEAM>.json')
    args = parser.parse_args(argv)

    date_str = args.date
    print(f'Fetching games for {date_str}...')
    try:
        df_pbp, df_shifts, df_gamedata = fetch_day(date_str, with_xg=(not args.no_xg))
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
        # After exporting, run projections and write CSV
        try:
            out_csv = run_player_projections_and_write_csv()
            print(f"Projections updated and saved to: {out_csv}")
        except Exception as e:
            print(f"[error] projections post-step failed: {e}", file=sys.stderr)
            return 4

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
