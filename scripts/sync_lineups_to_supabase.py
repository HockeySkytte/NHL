#!/usr/bin/env python3
"""sync_lineups_to_supabase.py — push gp_est/gp_est_note into the Supabase lineups table.

The live app reads the Supabase `lineups` table first, so after `estimate_gp.py`
refreshes `app/static/lineups_all.json` this script updates the `estimated_gp` /
`gp_note` columns for every row of the given season.

Usage:
    .\\.venv\\Scripts\\python.exe .\\scripts\\sync_lineups_to_supabase.py [--season 20262027]
"""
import argparse
import json
import os
import sys
import urllib.request

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LINEUPS_PATH = os.path.join(REPO_ROOT, 'app', 'static', 'lineups_all.json')


def _load_env(path):
    env = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def main():
    ap = argparse.ArgumentParser(description="Sync gp_est from lineups_all.json into Supabase lineups")
    ap.add_argument('--season', default='20262027', help='Season code (default 20262027)')
    ap.add_argument('--input', default=LINEUPS_PATH, help='Lineups JSON path')
    ap.add_argument('--dry-run', action='store_true', help='Print count only, do not PATCH')
    args = ap.parse_args()

    env = _load_env(os.path.join(REPO_ROOT, '.env'))
    url = (env.get('SUPABASE_URL') or '').rstrip('/')
    key = env.get('SUPABASE_SERVICE_KEY') or ''
    if not url or not key:
        print('Missing SUPABASE_URL/SUPABASE_SERVICE_KEY in .env', file=sys.stderr)
        return 1

    with open(args.input, 'r', encoding='utf-8') as f:
        lineups = json.load(f)

    updates = 0
    errors = 0
    for team_abbrev, team_data in lineups.items():
        for group in ('forwards', 'defense', 'goalies'):
            for p in team_data.get(group, []):
                pid = p.get('playerId')
                gp_est = p.get('gp_est')
                if not pid or gp_est is None:
                    continue
                updates += 1
                if args.dry_run:
                    continue
                body = json.dumps({
                    'estimated_gp': int(gp_est),
                    'gp_note': p.get('gp_est_note', ''),
                }).encode('utf-8')
                q = f'season=eq.{args.season}&team=eq.{team_abbrev}&player_id=eq.{pid}'
                req = urllib.request.Request(
                    f'{url}/rest/v1/lineups?{q}',
                    data=body,
                    method='PATCH',
                    headers={
                        'apikey': key,
                        'Authorization': f'Bearer {key}',
                        'Content-Type': 'application/json',
                        'Prefer': 'return=minimal',
                    },
                )
                try:
                    with urllib.request.urlopen(req, timeout=15):
                        pass
                except Exception as e:
                    errors += 1
                    if errors <= 5:
                        print(f'  ERR {team_abbrev}/{pid}: {e}', file=sys.stderr)
                if updates % 100 == 0:
                    print(f'  progress: {updates}', flush=True)

    print(f'{("dry-run: " if args.dry_run else "")}matched {updates} rows, errors: {errors}')
    return 0 if errors == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
