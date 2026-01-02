r"""backfill_range.py

Backfill NHL data into MySQL for an inclusive date range.

This is a thin driver around scripts/update_data.py helpers:
- fetch_day() for data collection (reuses Flask routes parsing)
- export_to_mysql() for writing to MySQL tables with optional replace-by-date

Typical usage (PowerShell):
    pwsh> & .\.venv\Scripts\python.exe .\scripts\backfill_range.py --start 2025-12-17 --end 2026-01-01 --season 20252026 --replace-date --no-xg

Notes:
- --replace-date now deletes rows for the date even if the fetch returns empty.
- By default this script runs projections once at the end (CALL Player_Projections()).
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from typing import List

# Import from update_data.py (same folder)
from update_data import (
    _validate_date,
    export_to_mysql,
    fetch_day,
    run_player_projections_and_write_csv,
)


def _parse_date(d: str) -> date:
    s = _validate_date(d)
    return datetime.strptime(s, "%Y-%m-%d").date()


def _daterange(start_d: date, end_d: date) -> List[date]:
    if end_d < start_d:
        raise argparse.ArgumentTypeError("end must be >= start")
    out: List[date] = []
    cur = start_d
    while cur <= end_d:
        out.append(cur)
        cur = cur + timedelta(days=1)
    return out


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill NHL data for a date range")
    p.add_argument("--start", required=True, type=_parse_date, help="Start date (YYYY-MM-DD), inclusive")
    p.add_argument("--end", required=True, type=_parse_date, help="End date (YYYY-MM-DD), inclusive")
    p.add_argument("--season", default="20252026", help="Season code for table names, e.g. 20252026")
    p.add_argument("--no-xg", action="store_true", help="Do not compute xG (faster)")
    p.add_argument(
        "--replace-date",
        action="store_true",
        help="Delete rows for each date before insert (recommended for backfills)",
    )
    p.add_argument(
        "--skip-projections",
        action="store_true",
        help="Skip CALL Player_Projections() and CSV write at the end",
    )
    args = p.parse_args(argv)

    dates = _daterange(args.start, args.end)
    season = str(args.season)

    total_pbp = 0
    total_shifts = 0
    total_gd = 0

    for d in dates:
        ds = d.isoformat()
        print(f"\n=== {ds} ===")
        df_pbp, df_shifts, df_gd = fetch_day(ds, with_xg=(not args.no_xg))
        print(
            f"Fetched: PBP rows={len(df_pbp)} | Shifts rows={len(df_shifts)} | GameData rows={len(df_gd)}"
        )

        export_to_mysql(
            df_pbp,
            df_shifts,
            df_gd,
            season=season,
            date_str=ds,
            replace_date=bool(args.replace_date),
        )

        total_pbp += int(len(df_pbp))
        total_shifts += int(len(df_shifts))
        total_gd += int(len(df_gd))

    print(
        f"\nDone range {args.start.isoformat()} -> {args.end.isoformat()} | "
        f"PBP={total_pbp} Shifts={total_shifts} GameData={total_gd}"
    )

    if not args.skip_projections:
        out_csv = run_player_projections_and_write_csv()
        print(f"Projections updated and saved to: {out_csv}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
