"""upload_lineups.py — Upload lineups_all.json + gp_est to the Supabase lineups table.

Reads app/static/lineups_all.json and upserts every player row into the
`lineups` table, marking starters (the 12F + 6D + 1G scraped from Daily Faceoff),
scratches (EXT players), and GP estimates from estimate_gp.py.

Usage:
    .\\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py
    .\\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py --season 20262027
    .\\.venv\\Scripts\\python.exe .\\scripts\\upload_lineups.py --dry-run
"""
from __future__ import annotations

import os
import sys
import json
import pathlib
import argparse
from datetime import datetime, timezone
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

# Refuse to replace the stored season when a scrape returns less than this
# fraction of the rows already stored (see the guard in main()).
MIN_ROW_FRACTION = 0.5


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


def build_rows(lineups: Dict[str, Any], season: str,
               existing: Dict[tuple, Dict] | None = None) -> List[Dict]:
    """Build the list of rows for the lineups table from the JSON data.

    `existing` maps (TEAM, player_id) -> the row already stored for this season.
    A 30-minute re-scrape produces a JSON without gp_est (only
    scripts/estimate_gp.py writes those), so a missing estimate falls back to
    the stored value instead of resetting Estimated Games to 0. The same applies
    to injury bookkeeping, which no scraper writes.
    """
    existing = existing or {}
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
                prev = existing.get((str(team_abbrev).upper(), int(pid)))
                gp_est = player.get('gp_est')
                gp_note = player.get('gp_est_note')
                if gp_est is None and prev:
                    gp_est = prev.get('estimated_gp')
                if gp_note is None and prev:
                    gp_note = prev.get('gp_note')
                row = {
                    'team': team_abbrev,
                    'player_id': int(pid),
                    'player_name': player.get('name') or '',
                    'position': pos,
                    'line_unit': unit,
                    'starter': starter,
                    'estimated_gp': int(gp_est or 0),
                    'gp_note': str(gp_note or ''),
                    'is_injured': 0,
                    'injury_start': None,
                    'injury_end': None,
                    'replacement_id': None,
                    'replacement_name': '',
                    'season': season,
                    'source': 'dailyfaceoff',
                }
                if prev:
                    for col in ('is_injured', 'injury_start', 'injury_end',
                                'replacement_id', 'replacement_name'):
                        if prev.get(col) is not None:
                            row[col] = prev[col]
                rows.append(row)
    return rows


def fetch_existing_rows(season: str) -> Dict[tuple, Dict]:
    """Existing lineups rows for `season`, keyed by (TEAM, player_id)."""
    try:
        from app.supabase_client import read_table
        df = read_table(
            'lineups',
            columns='team,player_id,estimated_gp,gp_note,is_injured,injury_start,'
                    'injury_end,replacement_id,replacement_name',
            filters={'season': f'eq.{season}'},
        )
    except Exception as e:
        print(f"  [warn] could not read existing lineups for carry-over: {e}", file=sys.stderr)
        return {}
    out: Dict[tuple, Dict] = {}
    for rec in df.to_dict(orient='records'):
        try:
            out[(str(rec.get('team') or '').upper(), int(rec.get('player_id')))] = rec
        except Exception:
            continue
    return out


def is_implausible_shrink(new_count: int, existing_count: int) -> bool:
    """True when a scrape lost so many rows that it is probably an outage.

    `lineups.py` leaves sections empty when a team's scrape fails, so a
    DailyFaceoff outage produces a plausible-looking but gutted file. Writing it
    would prune away most of the season's rosters.
    """
    if existing_count <= 0:
        return False
    return new_count < existing_count * MIN_ROW_FRACTION


def prune_stale_rows(season: str, new_keys: set, existing_keys: set) -> int:
    """Delete season rows whose (TEAM, player_id) is no longer in the scrape.

    Called *after* the upsert, so a player who left the team disappears without
    the season ever being empty. Deleting the whole season up front (the old
    behaviour) left the table blank for the duration of the insert, and the
    live app caches lineups for 5 minutes — so readers could latch onto an empty
    lineup. Two writers (the 30-minute cron and the daily Estimated Games job)
    would also race on that window.
    """
    stale = sorted(existing_keys - new_keys)
    if not stale:
        return 0

    from app.supabase_client import get_client
    client = get_client()
    by_team: Dict[str, List[int]] = {}
    for team, pid in stale:
        by_team.setdefault(team, []).append(int(pid))

    removed = 0
    for team, pids in by_team.items():
        for i in range(0, len(pids), 200):
            chunk = pids[i:i + 200]
            (
                client.table('lineups')
                .delete()
                .eq('season', season)
                .eq('team', team)
                .in_('player_id', chunk)
                .execute()
            )
            removed += len(chunk)
    return removed


def main():
    ap = argparse.ArgumentParser(description='Upload lineups_all.json to Supabase lineups table')
    ap.add_argument('--season', default=None, help='Season code (default: current season)')
    ap.add_argument('--input', help='Input JSON path (default app/static/lineups_all.json)')
    ap.add_argument('--dry-run', action='store_true', help='Preview only, do not write')
    ap.add_argument('--allow-shrink', action='store_true',
                    help='Upload even if the scrape lost most of the stored rows')
    args = ap.parse_args()

    if not args.season:
        from scripts.season_for_date import season_code_for
        args.season = season_code_for(datetime.now(timezone.utc).date())

    input_path = args.input or LINEUPS_PATH

    print(f"Reading lineups from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lineups = json.load(f)

    print(f"Reading existing rows for season {args.season} (Estimated Games carry-over)...")
    existing = fetch_existing_rows(args.season)
    print(f"  found {len(existing)} stored rows")

    rows = build_rows(lineups, args.season, existing)
    print(f"Built {len(rows)} lineup rows for season {args.season}")

    # Summary
    starters = sum(1 for r in rows if r['starter'])
    scratches = len(rows) - starters
    print(f"  Starters: {starters}  Scratches/Extras: {scratches}")
    teams = set(r['team'] for r in rows)
    print(f"  Teams: {len(teams)}")
    with_gp = sum(1 for r in rows if r['estimated_gp'])
    print(f"  Rows with Estimated Games: {with_gp}/{len(rows)}")

    # lineups.py leaves sections empty when a team's scrape fails, so a
    # DailyFaceoff outage produces a plausible-looking but gutted file. Writing
    # that would prune away most of the season's rosters, so stop loudly and
    # leave the stored data untouched.
    if not args.allow_shrink and not args.dry_run and is_implausible_shrink(len(rows), len(existing)):
        print(
            f"[error] scrape produced {len(rows)} rows but {len(existing)} are stored "
            f"for season {args.season}; refusing to replace the table "
            f"(re-run with --allow-shrink if this is intentional)",
            file=sys.stderr,
        )
        return 3

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
        from app.supabase_client import upsert_lineups
    except ImportError as e:
        print(f"Cannot import supabase_client: {e}", file=sys.stderr)
        return 1

    print(f"\nUpserting {len(rows)} rows...")
    try:
        count = upsert_lineups(rows)
        print(f"  Upserted {count} rows to lineups table")
    except Exception as e:
        print(f"  [error] upsert failed: {e}", file=sys.stderr)
        return 2

    new_keys = {(r['team'], int(r['player_id'])) for r in rows}
    try:
        removed = prune_stale_rows(args.season, new_keys, set(existing.keys()))
        print(f"  Removed {removed} rows no longer on a roster")
    except Exception as e:
        print(f"  [warn] prune failed: {e}", file=sys.stderr)

    print("Done.")


if __name__ == '__main__':
    sys.exit(main() or 0)