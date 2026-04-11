"""Backfill highlight_url for goal events in the Supabase pbp table.

For each game with goals missing highlight_url:
  1. Fetch the NHL PBP endpoint to learn eventId → event_index mapping.
  2. Fetch the NHL landing endpoint to get eventId → clip URL.
  3. Update matching rows in the pbp table.

Usage:
    python scripts/backfill_highlights.py --season 20242025
    python scripts/backfill_highlights.py --season 20242025 --dry-run
"""

import os, sys, time, argparse
from typing import Dict

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()
os.environ.setdefault('XG_PRELOAD', '0')

from app.supabase_client import get_client

NHL_PBP_URL  = 'https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play'
NHL_LAND_URL = 'https://api-web.nhle.com/v1/gamecenter/{gid}/landing'
PAGE = 1000
DELAY = 0.25  # seconds between API call pairs


def get_games_needing_backfill(sb, season: str) -> set:
    """Return set of game_id values that have goals with NULL highlight_url."""
    rows = []
    offset = 0
    while True:
        batch = (
            sb.table('pbp')
            .select('game_id')
            .eq('season', season)
            .eq('goal', 1)
            .is_('highlight_url', 'null')
            .range(offset, offset + PAGE - 1)
            .execute()
        ).data
        rows.extend(batch)
        if len(batch) < PAGE:
            break
        offset += PAGE
    return set(r['game_id'] for r in rows)


def fetch_goal_eventid_map(game_id: int) -> Dict[int, int]:
    """Fetch PBP and return {eventId: event_index} for goal plays."""
    url = NHL_PBP_URL.format(gid=game_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    plays = r.json().get('plays') or []
    mapping = {}
    for idx, pl in enumerate(plays):
        if pl.get('typeDescKey') == 'goal':
            eid = pl.get('eventId')
            if eid is not None:
                event_index = game_id * 10000 + (idx + 1)
                mapping[int(eid)] = event_index
    return mapping


def fetch_highlight_clips(game_id: int) -> Dict[int, str]:
    """Fetch landing endpoint and return {eventId: clipUrl} for goals."""
    url = NHL_LAND_URL.format(gid=game_id)
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    data = r.json()
    clips = {}
    for per in (data.get('summary', {}).get('scoring') or []):
        for gl in (per.get('goals') or []):
            eid = gl.get('eventId')
            clip = gl.get('highlightClipSharingUrl')
            if eid is not None and clip:
                clips[int(eid)] = clip
    return clips


def backfill_highlights(season: str, dry_run: bool = False):
    sb = get_client()
    game_ids = sorted(get_games_needing_backfill(sb, season))
    total = len(game_ids)
    print(f'[backfill] {total} games with goals missing highlight_url in season {season}')

    updated_total = 0
    errors = 0

    for i, gid in enumerate(game_ids, 1):
        try:
            eid_to_eidx = fetch_goal_eventid_map(gid)
            clips = fetch_highlight_clips(gid)

            game_updates = 0
            for eid, clip_url in clips.items():
                eidx = eid_to_eidx.get(eid)
                if eidx:
                    if dry_run:
                        game_updates += 1
                    else:
                        sb.table('pbp').update(
                            {'highlight_url': clip_url}
                        ).eq('event_index', eidx).eq('season', season).execute()
                        game_updates += 1

            updated_total += game_updates
            if game_updates or i % 100 == 0:
                print(f'  [{i}/{total}] game {gid}: {game_updates} highlights'
                      + (' (dry-run)' if dry_run else ''))
            time.sleep(DELAY)

        except requests.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                # Game may have been cancelled or not available
                print(f'  [{i}/{total}] game {gid}: 404 (skipped)')
            else:
                print(f'  [{i}/{total}] game {gid}: HTTP error {e}')
                errors += 1
        except Exception as e:
            print(f'  [{i}/{total}] game {gid}: error {e}')
            errors += 1

    tag = ' (DRY RUN)' if dry_run else ''
    print(f'\n[backfill] Done{tag}: {updated_total} highlight URLs, {errors} errors')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backfill highlight_url in pbp table')
    parser.add_argument('--season', default='20242025', help='Season code (e.g. 20242025)')
    parser.add_argument('--dry-run', action='store_true', help='Count matches without writing')
    args = parser.parse_args()
    backfill_highlights(args.season, args.dry_run)
