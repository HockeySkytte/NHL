"""Inspect anomalous StrengthState rows in nhl_{season}_shifts.

This is a diagnostics helper to understand why extremely large numeric states
(e.g. '10v8') appear.

Usage:
  python scripts/inspect_shift_strength_anomalies.py --season 20252026 --state 10v8
"""

from __future__ import annotations

import argparse
from typing import Any, Optional

from sqlalchemy import text

import update_data


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", default="20252026")
    ap.add_argument("--state", default="10v8")
    ap.add_argument("--limit", type=int, default=3)
    args = ap.parse_args(argv)

    season = str(args.season)
    tbl = f"nhl_{season}_shifts"

    eng = update_data._create_mysql_engine("ro") or update_data._create_mysql_engine("rw")
    if eng is None:
        raise SystemExit("MySQL engine not available")

    with eng.connect() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT GameID, ShiftIndex, MIN(Date) AS Date, MIN(Start) AS Start, MIN(End) AS End
                FROM {tbl}
                WHERE StrengthState = :st
                GROUP BY GameID, ShiftIndex
                ORDER BY Date DESC
                LIMIT :lim
                """
            ),
            {"st": args.state, "lim": int(args.limit)},
        ).fetchall()

        if not rows:
            print(f"No ShiftIndex groups found for StrengthState={args.state!r}")
            return 0

        for (game_id, shift_index, date_v, start_v, end_v) in rows:
            print("\n=== Example ===")
            print("GameID", game_id, "ShiftIndex", shift_index, "Date", date_v, "Start", start_v, "End", end_v)

            # Count rows/distinct players by team/pos
            agg = conn.execute(
                text(
                    f"""
                    SELECT
                        Team,
                        Position,
                        COUNT(*) AS row_count,
                        COUNT(DISTINCT PlayerID) AS distinct_players
                    FROM {tbl}
                    WHERE GameID = :gid AND ShiftIndex = :sx
                    GROUP BY Team, Position
                    ORDER BY Team, Position
                    """
                ),
                {"gid": game_id, "sx": shift_index},
            ).fetchall()

            print("Team/Position breakdown:")
            for r in agg:
                print(" ", r)

            # Count distinct skaters per team (Position != 'G')
            sk = conn.execute(
                text(
                    f"""
                    SELECT
                        Team,
                        COUNT(*) AS row_count,
                        COUNT(DISTINCT PlayerID) AS distinct_skaters
                    FROM {tbl}
                    WHERE GameID = :gid AND ShiftIndex = :sx AND (Position IS NULL OR Position <> 'G')
                    GROUP BY Team
                    ORDER BY distinct_skaters DESC
                    """
                ),
                {"gid": game_id, "sx": shift_index},
            ).fetchall()
            print("Skaters per team:")
            for r in sk:
                print(" ", r)

            sample = conn.execute(
                text(
                    f"""
                    SELECT Team, PlayerID, Name, Position
                    FROM {tbl}
                    WHERE GameID = :gid AND ShiftIndex = :sx
                    ORDER BY Team, Name
                    LIMIT 30
                    """
                ),
                {"gid": game_id, "sx": shift_index},
            ).fetchall()
            print("Sample rows:")
            for r in sample:
                print(" ", r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
