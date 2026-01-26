"""Backfill NHL shifts for a season into MySQL only.

Goal:
- Fetch shifts for games in a season (regular + playoffs by default)
- Export ONLY to MySQL table: nhl_{season}_shifts
- Do NOT write any other outputs (no CSV/JSON files)

Implementation notes:
- Reuses the internal Flask route /api/game/<id>/shifts via create_app().test_client().
- Calls the route with nocache=1&nodisk=1 to avoid both in-memory and on-disk shift caching.
- Uses the public schedule endpoint to discover game IDs by date.

Usage examples:
  pwsh> ./.venv/Scripts/python.exe ./scripts/backfill_shifts_season.py --season 20232024
  pwsh> ./.venv/Scripts/python.exe ./scripts/backfill_shifts_season.py --season 20192020 --start-date 2020-08-01 --end-date 2020-10-01
  pwsh> ./.venv/Scripts/python.exe ./scripts/backfill_shifts_season.py --season 20212022 --replace

MySQL env:
- Uses the same env var conventions as scripts/update_data.py (_create_mysql_engine).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from sqlalchemy import text

# Reuse existing helpers (scripts/ is not a package, so import via sys.path)
SCRIPTS_DIR = os.path.dirname(__file__)
if SCRIPTS_DIR and SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)
import update_data  # type: ignore  # noqa: E402

_create_mysql_engine = update_data._create_mysql_engine
get_game_ids_for_date = update_data.get_game_ids_for_date

# Import Flask app factory (internal route reuse)
os.environ.setdefault('XG_PRELOAD', '0')  # backfills shouldn't preload models
from app import create_app  # noqa: E402


def _validate_date(d: str) -> str:
    try:
        dt = datetime.strptime(d, '%Y-%m-%d')
        return dt.strftime('%Y-%m-%d')
    except Exception:
        raise argparse.ArgumentTypeError('Date must be in YYYY-MM-DD format')


def _season_date_window(season: str) -> tuple[str, str]:
    """Best-effort default window for a seasonId like 20192020.

    We intentionally cast a wide net so we include regular season + playoffs.
    The NHL schedule endpoint will simply return no games for off-days.
    """
    if not (season.isdigit() and len(season) == 8):
        raise ValueError('season must be an 8-digit seasonId like 20192020')
    y0 = int(season[:4])
    y1 = int(season[4:])
    # Start early enough to include early season games; end late enough for playoffs.
    return (f"{y0}-09-01", f"{y1}-07-15")


def _date_range(start: str, end: str) -> Iterable[str]:
    ds = datetime.strptime(start, '%Y-%m-%d')
    de = datetime.strptime(end, '%Y-%m-%d')
    if de < ds:
        raise ValueError('end-date must be >= start-date')
    cur = ds
    while cur <= de:
        yield cur.strftime('%Y-%m-%d')
        cur += timedelta(days=1)


def _coerce_shift_rows(rows: List[Dict[str, Any]], *, date_str: str, game_id: int) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['Date'] = date_str
    df['GameID'] = int(game_id)
    return df


def export_shifts_to_mysql(df_shifts: pd.DataFrame, *, season: str, replace: bool = False) -> int:
    eng = _create_mysql_engine('rw')
    if eng is None:
        raise RuntimeError('MySQL engine not available (check DATABASE_URL / DB_* env vars)')

    tbl_sh = f"nhl_{season}_shifts"

    if df_shifts is None or df_shifts.empty:
        return 0

    # Optional idempotency: delete existing rows for the gameIds we are about to insert.
    if replace:
        try:
            gids = sorted({int(x) for x in df_shifts['GameID'].dropna().tolist()})
        except Exception:
            gids = []
        if gids:
            try:
                with eng.begin() as conn:
                    # Chunk the IN clause to stay safe on very large days.
                    chunk = 250
                    for i in range(0, len(gids), chunk):
                        part = gids[i:i + chunk]
                        params = {f"id{i2}": int(v) for i2, v in enumerate(part)}
                        in_list = ','.join([f":id{i2}" for i2 in range(len(part))])
                        conn.execute(text(f"DELETE FROM `{tbl_sh}` WHERE GameID IN ({in_list})"), params)
            except Exception:
                # Table may not exist; ignore.
                pass

    # Append-only insert; creates table if missing
    df_shifts.to_sql(tbl_sh, con=eng, if_exists='append', index=False, method='multi', chunksize=1000)
    return int(len(df_shifts))


def backfill_one_season(
    *,
    season: str,
    start_date: Optional[str],
    end_date: Optional[str],
    replace: bool,
    sleep_s: float,
    max_games: int,
) -> int:
    """Backfill shifts for a single season. Returns number of rows written."""
    start0, end0 = _season_date_window(season)
    if start_date:
        start0 = start_date
    if end_date:
        end0 = end_date

    print(f"[backfill-shifts] season={season} window={start0}..{end0} replace={bool(replace)}")

    app = create_app()
    total_games = 0
    total_rows = 0
    days_with_games = 0
    stop_after_day = False

    with app.test_client() as client:
        for date_str in _date_range(start0, end0):
            try:
                game_ids = get_game_ids_for_date(date_str)
            except Exception as e:
                print(f"[warn] schedule failed for {date_str}: {e}", file=sys.stderr)
                continue

            if not game_ids:
                continue

            days_with_games += 1
            day_rows: List[pd.DataFrame] = []

            for gid in game_ids:
                if max_games and total_games >= int(max_games):
                    stop_after_day = True
                    break
                total_games += 1

                resp = client.get(
                    f'/api/game/{int(gid)}/shifts',
                    query_string={
                        'force': '1',
                        'nocache': '1',
                        'nodisk': '1',
                    }
                )
                if resp.status_code != 200:
                    print(f"[warn] shifts failed gid={gid} date={date_str}: {resp.status_code}", file=sys.stderr)
                    time.sleep(float(sleep_s))
                    continue

                js = resp.get_json(silent=True) or {}
                rows = js.get('shifts') or []
                if isinstance(rows, list) and rows:
                    df = _coerce_shift_rows(rows, date_str=date_str, game_id=int(gid))
                    if not df.empty:
                        day_rows.append(df)

                time.sleep(float(sleep_s))

            if not day_rows:
                if stop_after_day:
                    print('[backfill-shifts] max-games reached; stopping')
                    break
                continue

            df_day = pd.concat(day_rows, ignore_index=True)
            wrote = export_shifts_to_mysql(df_day, season=season, replace=bool(replace))
            total_rows += wrote
            print(
                f"[mysql] {date_str}: games={len(game_ids)} shifts_rows={len(df_day)} "
                f"wrote={wrote} totals: games={total_games} rows={total_rows}"
            )

            if stop_after_day:
                print('[backfill-shifts] max-games reached; stopping')
                break

    print(f"[done] season={season} days_with_games={days_with_games} games={total_games} rows={total_rows}")
    return int(total_rows)


def _season_step_next(season: str) -> str:
    """Increment an 8-digit seasonId (YYYYYYYY) by one season."""
    if not (season.isdigit() and len(season) == 8):
        raise ValueError('season must be an 8-digit seasonId like 20192020')
    y0 = int(season[:4]) + 1
    y1 = int(season[4:]) + 1
    return f"{y0}{y1}"


def main() -> int:
    ap = argparse.ArgumentParser(description='Backfill shifts for a season into MySQL table nhl_{season}_shifts (no local files).')
    ap.add_argument('--season', default=None, help='Single seasonId like 20192020')
    ap.add_argument('--from-season', dest='from_season', default=None, help='Start seasonId (inclusive), e.g. 20072008')
    ap.add_argument('--to-season', dest='to_season', default=None, help='End seasonId (inclusive), e.g. 20222023')
    ap.add_argument('--start-date', type=_validate_date, default=None, help='Override start date (YYYY-MM-DD)')
    ap.add_argument('--end-date', type=_validate_date, default=None, help='Override end date (YYYY-MM-DD)')
    ap.add_argument('--replace', action='store_true', help='Delete existing rows for inserted GameIDs before insert')
    ap.add_argument('--sleep', type=float, default=0.25, help='Seconds to sleep between game fetches (be polite to NHL endpoints)')
    ap.add_argument('--max-games', type=int, default=0, help='Debug: stop after N games (0 = no limit)')
    ap.add_argument('--continue-on-error', action='store_true', help='When running a season range, keep going if a season fails')

    args = ap.parse_args()

    # Determine run mode.
    season_single = (str(args.season).strip() if args.season else '')
    from_season = (str(args.from_season).strip() if args.from_season else '')
    to_season = (str(args.to_season).strip() if args.to_season else '')

    if season_single and (from_season or to_season):
        raise SystemExit('Provide either --season or (--from-season and --to-season), not both')
    if (from_season and not to_season) or (to_season and not from_season):
        raise SystemExit('Provide both --from-season and --to-season for range runs')
    if not season_single and not (from_season and to_season):
        raise SystemExit('Missing season: provide --season or (--from-season and --to-season)')

    # Single season
    if season_single:
        backfill_one_season(
            season=season_single,
            start_date=args.start_date,
            end_date=args.end_date,
            replace=bool(args.replace),
            sleep_s=float(args.sleep),
            max_games=int(args.max_games or 0),
        )
        return 0

    # Season range (inclusive)
    if int(from_season) > int(to_season):
        raise SystemExit('--from-season must be <= --to-season')

    season_cur = from_season
    grand_rows = 0
    while True:
        t0 = time.time()
        try:
            wrote = backfill_one_season(
                season=season_cur,
                start_date=args.start_date,
                end_date=args.end_date,
                replace=bool(args.replace),
                sleep_s=float(args.sleep),
                max_games=int(args.max_games or 0),
            )
            grand_rows += int(wrote)
            dt = time.time() - t0
            print(f"[range] season={season_cur} wrote={wrote} elapsed={dt:.1f}s grand_rows={grand_rows}")
        except Exception as e:
            print(f"[error] season={season_cur} failed: {e}", file=sys.stderr)
            if not bool(args.continue_on_error):
                return 2

        if season_cur == to_season:
            break
        season_cur = _season_step_next(season_cur)
        if int(season_cur) > int(to_season):
            break

    print(f"[done-range] from={from_season} to={to_season} grand_rows={grand_rows}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
