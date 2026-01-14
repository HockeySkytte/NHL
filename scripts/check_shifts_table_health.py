"""Quick health checks for nhl_{season}_shifts.

Usage:
    pwsh> & ./.venv/Scripts/python.exe ./scripts/check_shifts_table_health.py --season 20252026
    pwsh> & ./.venv/Scripts/python.exe ./scripts/check_shifts_table_health.py --season 20252026 --date 2026-01-10
"""

from __future__ import annotations

import argparse
from typing import Optional

import os
import sys

from sqlalchemy import text

# Allow importing sibling scripts without requiring scripts/ to be a package.
SCRIPTS_DIR = os.path.abspath(os.path.dirname(__file__))
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from update_data import _create_mysql_engine  # type: ignore


def _scalar(conn, sql: str, **params):
    return conn.execute(text(sql), params).scalar()


def check(season: str, date: Optional[str], null_summary: bool, list_states: bool) -> None:
    table = f"nhl_{season}_shifts"

    eng = _create_mysql_engine("ro") or _create_mysql_engine("rw")
    if eng is None:
        raise SystemExit("Could not create MySQL engine (missing env vars?)")

    where = ""
    params = {}
    label = "ALL"
    if date:
        where = "WHERE Date=:date"
        params = {"date": date}
        label = date

    with eng.connect() as conn:
        total = _scalar(conn, f"SELECT COUNT(*) FROM {table} {where}", **params)
        distinct_strength = _scalar(
            conn, f"SELECT COUNT(DISTINCT StrengthState) FROM {table} {where}", **params
        )
        null_playerid_sql = (
            f"SELECT COUNT(*) FROM {table} {where} AND PlayerID IS NULL"
            if where
            else f"SELECT COUNT(*) FROM {table} WHERE PlayerID IS NULL"
        )
        null_name_sql = (
            f"SELECT COUNT(*) FROM {table} {where} AND Name IS NULL"
            if where
            else f"SELECT COUNT(*) FROM {table} WHERE Name IS NULL"
        )
        null_both_sql = (
            f"SELECT COUNT(*) FROM {table} {where} AND PlayerID IS NULL AND Name IS NULL"
            if where
            else f"SELECT COUNT(*) FROM {table} WHERE PlayerID IS NULL AND Name IS NULL"
        )

        null_playerid = _scalar(conn, null_playerid_sql, **params)
        null_name = _scalar(conn, null_name_sql, **params)
        null_both = _scalar(conn, null_both_sql, **params)

        top_states = None
        if list_states and date:
            top_states = conn.execute(
                text(
                    f"""
                    SELECT StrengthState, COUNT(*) AS c
                    FROM {table}
                    WHERE Date=:date
                    GROUP BY StrengthState
                    ORDER BY c DESC
                    LIMIT 50
                    """
                ),
                params,
            ).fetchall()

    print(f"[{label}] rows={total:,}")
    print(f"[{label}] distinct StrengthState={distinct_strength:,}")
    print(f"[{label}] PlayerID NULL={null_playerid:,}")
    print(f"[{label}] Name NULL={null_name:,}")
    print(f"[{label}] PlayerID+Name NULL={null_both:,}")

    if list_states and date:
        if top_states:
            print(f"[{label}] StrengthState distribution (top 50):")
            for st, c in top_states:
                print(f"  {st}: {int(c):,}")
        else:
            print(f"[{label}] StrengthState distribution: (no rows)")

    if null_summary and not date:
        eng = _create_mysql_engine("ro") or _create_mysql_engine("rw")
        if eng is None:
            return
        with eng.connect() as conn:
            first_null_date = _scalar(
                conn,
                f"SELECT MIN(Date) FROM {table} WHERE PlayerID IS NULL AND Name IS NULL",
            )
            print(f"[ALL] first date with anonymous rows={first_null_date}")

            rows = conn.execute(
                text(
                    f"""
                    SELECT Date, COUNT(*) AS c
                    FROM {table}
                    WHERE PlayerID IS NULL AND Name IS NULL
                    GROUP BY Date
                    ORDER BY c DESC
                    LIMIT 10
                    """
                )
            ).fetchall()
        if rows:
            print("[ALL] top dates by anonymous rows:")
            for d, c in rows:
                print(f"  {d}: {int(c):,}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="Season like 20252026")
    ap.add_argument("--date", default=None, help="Optional YYYY-MM-DD")
    ap.add_argument(
        "--null-summary",
        action="store_true",
        help="(ALL only) show earliest + top dates with PlayerID+Name NULL",
    )
    ap.add_argument(
        "--list-states",
        action="store_true",
        help="(requires --date) print StrengthState distribution",
    )
    args = ap.parse_args()

    check(args.season, args.date, bool(args.null_summary), bool(args.list_states))


if __name__ == "__main__":
    main()
