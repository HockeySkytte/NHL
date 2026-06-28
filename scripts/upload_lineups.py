"""upload_lineups.py — Upload lineups_all.json + gp_est to the Supabase lineups table.

Reads app/static/lineups_all.json and upserts every player row into the
`lineups` table, marking starters (the 12F + 6D + 1G scraped from Daily Faceoff),
scratches (EXT players), and GP estimates from estimate_gp.py.

Usage:
.\\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py
    .\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py --season 20262027
    .\\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py --dry-run
"""
from __future__ import annotations

import os
import sys
import json
import pathlib
import argparse
from typing import Dict, List, Any

from dotenv import load_dotenv

# Load .env from repo root before importing any app modules that read env vars
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

# Workspace root
REPO_ROOT = str(ROOT)

LINEUPS_PATH = os.path.join(REPO_ROOT, 'app', 'static', 'lineups_all.json')

# Starter units: these are the 12 forwards, 6 defense, and 1 starting goalie
# that Daily Faceoff reports in the projected lineup. EXT = scratch.
STARTER_UNIT_PREFIXES = ('LW', 'C', 'RW', 'LD', 'RD')
STARTER_GOALIE_UNIT = 'G1'  # only the starting goalie


def is_starter(unit: str, pos: str) -> bool:
    """Return True if this slot is in the starting lineup (not a scratch)."""
    u = (unit or '').upper().strip()
    if not u:
        return False
    # Forwards and defense: LW1, C1, RW1, ..., LD3, RD3 → starters
    if any(u.startswith(p) for p in STARTER_UNIT_PREFIXES):
        return True
    # Starting goalie
    if u == STARTER_GOALIE_UNIT:
        return True
    return False


def build_rows(lineups: Dict[str, Any], season: str) -> List[Dict]:
    """Build the list of rows for the lineups table from the JSON data."""
    rows: List[Dict] = []
    for team_abbrev, team_data in lineups.items():
        if not isinstance(team_data, dict):
            continue
        for group_key, default_pos in (('forwards', 'F'), ('defense', 'D'), ('goalies', 'G')):
            for player in team_data.get(group_key, []) or []:
                pid = player.get('playerId')
                if not pid:
                    continue
                unit = (player.get('unit') or 'EXT').upper()
                pos = player.get('pos') or default_pos
                starter = 1 if is_starter(unit, pos) else 0
                gp_est = int(player.get('gp_est') or 0)
                gp_note = str(player.get('gp_est_note') or '')
                rows.append({
                    'team': team_abbrev,
                    'player_id': int(pid),
                    'player_name': player.get('name') or '',
                    'position': pos,
                    'line_unit': unit,
                    'starter': starter,
                    'estimated_gp': gp_est,
                    'gp_note': gp_note,
                    'is_injured': 0,
                    'injury_start': None,
                    'injury_end': None,
                    'replacement_id': None,
                    'replacement_name': '',
                    'season': season,
                    'source': 'dailyfaceoff',
                })
    return rows


def main():
    ap = argparse.ArgumentParser(description='Upload lineups_all.json to Supabase lineups table')
    ap.add_argument('--season', default='20262027', help='Season code (default 20262027)')
    ap.add_argument('--input', help='Input JSON path (default app/static/lineups_all.json)')
    ap.add_argument('--dry-run', action='store_true', help='Preview only, do not write')
    args = ap.parse_args()

    input_path = args.input or LINEUPS_PATH

    print(f"Reading lineups from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lineups = json.load(f)

    rows = build_rows(lineups, args.season)
    print(f"Built {len(rows)} lineup rows for season {args.season}")

    # Summary
    starters = sum(1 for r in rows if r['starter'])
    scratches = len(rows) - starters
    print(f"  Starters: {starters}  Scratches/Extras: {scratches}")
    teams = set(r['team'] for r in rows)
    print(f"  Teams: {len(teams)}")

    if args.dry_run:
        print("\n--- DRY RUN (not writing) ---")
        for team in sorted(teams)[:3]:
            team_rows = [r for r in rows if r['team'] == team]
            print(f"\n{team} ({len(team_rows)} rows):")
            for r in team_rows[:6]:
                flag = '*' if r['starter'] else ' '
                print(f"  {flag} {r['player_name']:25s} {r['line_unit']:5s} gp={r['estimated_gp']:3d}")
        return

    # Upload
    try:
        from app.supabase_client import upsert_lineups, delete_lineups
    except ImportError as e:
        print(f"Cannot import supabase_client: {e}", file=sys.stderr)
        return 1

    # Delete existing rows for this season first (clean slate)
    print(f"\nDeleting existing lineups for season {args.season}...")
    try:
        delete_lineups(season=args.season)
        print("  done")
    except Exception as e:
        print(f"  [warn] delete failed: {e}")

    print(f"Upserting {len(rows)} rows...")
    try:
        count = upsert_lineups(rows)
        print(f"  Uploaded {count} rows to lineups table")
    except Exception as e:
        print(f"  [error] upsert failed: {e}", file=sys.stderr)
        return 2

    print("Done.")


if __name__ == '__main__':
    sys.exit(main() or 0)