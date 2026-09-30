"""season_for_date.py — print the NHL season code (e.g. 20262027) for a date.

The GitHub Actions workflows used to hard-code `--season 20252026`. Once the
20262027 season started that silently wrote the new season's games into the old
season's tables (and rebuilt the old season's aggregates), so the app showed
nothing for the games actually played.

`season_code_for` mirrors `app.routes.current_season_id`: a season is named for
the calendar year it starts in, and September is the rollover month.

Usage:
    python scripts/season_for_date.py                       # today (UTC)
    python scripts/season_for_date.py --date 2026-09-28
    python scripts/season_for_date.py --days-ago 1          # yesterday (UTC)
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta, timezone


def season_code_for(day: date) -> str:
    """Season code for a calendar date, e.g. date(2026, 9, 28) -> '20262027'."""
    start_year = day.year if day.month >= 9 else day.year - 1
    return f"{start_year}{start_year + 1}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Print the NHL season code for a date")
    ap.add_argument('--date', help='Date in YYYY-MM-DD (default: today UTC)')
    ap.add_argument('--days-ago', type=int, default=0,
                    help='Use this many days before today UTC (default 0)')
    args = ap.parse_args(argv)

    if args.date:
        day = datetime.strptime(args.date, '%Y-%m-%d').date()
    else:
        day = (datetime.now(timezone.utc) - timedelta(days=args.days_ago)).date()

    print(season_code_for(day))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
