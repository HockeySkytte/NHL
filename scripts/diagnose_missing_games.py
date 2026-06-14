"""
diagnose_missing_games.py

Check which NHL games for the 2025-26 season might be missing from the database.

Usage:
  python scripts/diagnose_missing_games.py --season 20252026 [--round R1|R2|R3|R4|all]

Compares the NHL schedule API against Supabase PBP data to find games
that exist in the schedule but have no play-by-play rows.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date, timedelta
from typing import List, Dict, Optional, Set

import requests
import pandas as pd

# Ensure the app module is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('XG_PRELOAD', '0')


def _season_date_bounds(season: str) -> tuple:
    s = str(season).strip()
    start_year = int(s[:4])
    end_year = int(s[4:])
    return date(start_year, 9, 1), date(end_year, 10, 1)


def get_all_games_for_season(season: str) -> List[dict]:
    """Fetch ALL games for a season from the NHL schedule API.
    
    Uses the club-schedule-season endpoint for all 32 teams and deduplicates.
    """
    season_str = str(season).strip()
    all_games: Dict[int, dict] = {}
    
    # 32 NHL team abbreviations (current era)
    teams = [
        'ANA','BOS','BUF','CAR','CBJ','CGY','CHI','COL','DAL','DET',
        'EDM','FLA','LAK','MIN','MTL','NJD','NSH','NYI','NYR','OTT',
        'PHI','PIT','SJS','SEA','STL','TBL','TOR','UTA','VAN','VGK',
        'WPG','WSH',
    ]
    
    for team in teams:
        try:
            url = f'https://api-web.nhle.com/v1/club-schedule-season/{team}/{season_str}'
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            for g in (data.get('games') or []):
                gid = g.get('id')
                if gid and gid not in all_games:
                    all_games[gid] = {
                        'game_id': gid,
                        'game_type': g.get('gameType'),
                        'game_date': g.get('gameDate', '')[:10],
                        'away': (g.get('awayTeam') or {}).get('abbrev', '?'),
                        'home': (g.get('homeTeam') or {}).get('abbrev', '?'),
                        'game_state': g.get('gameState', '?'),
                    }
        except Exception as e:
            print(f'[warn] Failed to fetch schedule for {team}: {e}', file=sys.stderr)
    
    return sorted(all_games.values(), key=lambda g: g['game_date'])


def check_supabase_pbp_games(season: int) -> Set[int]:
    """Return the set of game_ids that have PBP rows in Supabase."""
    from app.supabase_client import get_client
    client = get_client()
    
    # Query distinct game_ids from pbp table for this season
    try:
        resp = client.table('pbp').select('game_id').eq('season', season).execute()
        rows = resp.data or []
        return {int(r['game_id']) for r in rows if r.get('game_id')}
    except Exception as e:
        print(f'[warn] Supabase query failed: {e}', file=sys.stderr)
        return set()


def main():
    parser = argparse.ArgumentParser(description='Diagnose missing NHL games')
    parser.add_argument('--season', default='20252026', help='Season code')
    parser.add_argument('--round', default='all', help='Filter by playoff round (R1/R2/R3/R4/all)')
    parser.add_argument('--check-db', action='store_true', help='Check against Supabase (requires SUPABASE_URL)')
    args = parser.parse_args()
    
    season = str(args.season).strip()
    round_filter = str(args.round).strip().upper()
    
    print(f'Fetching all games for season {season} from NHL API ...')
    games = get_all_games_for_season(season)
    
    # Filter by game type
    regular = [g for g in games if g['game_type'] == 2]
    playoffs = [g for g in games if g['game_type'] == 3]
    
    print(f'\nRegular season games: {len(regular)}')
    print(f'Playoff games: {len(playoffs)}')
    
    # Show playoff games by round
    # Round is encoded in game ID: YYYY03RRGG where RR=round (01-04)
    for round_num in range(1, 5):
        round_games = [g for g in playoffs if (g['game_id'] // 100) % 100 == round_num]
        if round_games:
            print(f'\n  Round {round_num} ({len(round_games)} games):')
            for g in round_games:
                print(f"    {g['game_id']}: {g['away']} @ {g['home']} on {g['game_date']} [{g['game_state']}]")
    
    # Show completed games that haven't been played yet
    completed_playoffs = [g for g in playoffs if g['game_state'] in ('OFF', 'FINAL')]
    future_playoffs = [g for g in playoffs if g['game_state'] not in ('OFF', 'FINAL')]
    
    if future_playoffs:
        print(f'\n  Future playoff games ({len(future_playoffs)}):')
        for g in future_playoffs:
            print(f"    {g['game_id']}: {g['away']} @ {g['home']} on {g['game_date']} [{g['game_state']}]")
    
    # Check against database if requested
    if args.check_db:
        print(f'\nChecking against Supabase PBP data ...')
        season_i = int(season)
        db_game_ids = check_supabase_pbp_games(season_i)
        
        missing = [g for g in games if g['game_id'] not in db_game_ids and g['game_state'] in ('OFF', 'FINAL')]
        if missing:
            print(f'\n  MISSING GAMES ({len(missing)} completed games without PBP data):')
            for g in missing:
                print(f"    {g['game_id']} (type={g['game_type']}): {g['away']} @ {g['home']} on {g['game_date']} [{g['game_state']}]")
        else:
            print('  All completed games have PBP data.')
        
        # Check for xG coverage
        print(f'\nChecking xG coverage for playoff games ...')
        try:
            from app.supabase_client import get_client
            client = get_client()
            for g in completed_playoffs:
                resp = client.table('pbp').select('xg_f').eq('game_id', g['game_id']).eq('season', season_i).not_.is_('xg_f', 'null').limit(1).execute()
                has_xg = bool(resp.data)
                status = '✓ has xG' if has_xg else '✗ NO xG'
                print(f"    {g['game_id']}: {status}")
        except Exception as e:
            print(f'    xG check failed: {e}')


if __name__ == '__main__':
    main()
