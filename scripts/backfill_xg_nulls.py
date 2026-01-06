"""Backfill xG NULLing rule in MySQL season tables.

Rule (season 20252026 request): whenever reason is 'short' or 'failed-bank-attempt',
set xG_F, xG_F2, and xG_S to NULL.

This script is safe to re-run (idempotent) and will:
- connect using the same env-driven MySQL config as scripts/update_data.py
- verify tables/columns exist before attempting updates
- print how many rows match before/after

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe .\\scripts\\backfill_xg_nulls.py --season 20252026

Notes:
- Requires RW DB credentials.
"""

from __future__ import annotations

import argparse
import sys
from importlib import util as _importlib_util
from pathlib import Path
from typing import Any, Iterable, Optional, cast

from sqlalchemy import text
from sqlalchemy.engine import Engine


def _lower_trim(col: str) -> str:
    # MySQL-compatible LOWER(TRIM(col))
    return f"LOWER(TRIM({col}))"


def _column_exists(eng: Engine, table_name: str, column_name: str) -> bool:
    sql = text(
        """
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = DATABASE()
          AND table_name = :t
          AND column_name = :c
        LIMIT 1
        """
    )
    with eng.begin() as conn:
        row = conn.execute(sql, {"t": table_name, "c": column_name}).first()
        return row is not None


def _table_exists(eng: Engine, table_name: str) -> bool:
    sql = text(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = DATABASE()
          AND table_name = :t
        LIMIT 1
        """
    )
    with eng.begin() as conn:
        row = conn.execute(sql, {"t": table_name}).first()
        return row is not None


def _count_matching(eng: Engine, table_name: str, reason_col: str) -> int:
    sql = text(
        f"""
        SELECT COUNT(*) AS n
        FROM {table_name}
        WHERE {_lower_trim(reason_col)} IN ('short', 'failed-bank-attempt')
        """
    )
    with eng.begin() as conn:
        return int(conn.execute(sql).scalar() or 0)


def _apply_update(
    eng: Engine,
    table_name: str,
    reason_col: str,
    xg_cols: Iterable[str],
) -> int:
    sets = ", ".join([f"{c} = NULL" for c in xg_cols])
    sql = text(
        f"""
        UPDATE {table_name}
        SET {sets}
        WHERE {_lower_trim(reason_col)} IN ('short', 'failed-bank-attempt')
        """
    )
    with eng.begin() as conn:
        res = conn.execute(sql)
        # rowcount should be available for MySQL connector
        return int(res.rowcount or 0)


def _load_update_data_module() -> Any:
    update_data_path = Path(__file__).resolve().parent / "update_data.py"
    spec = _importlib_util.spec_from_file_location("nhl_update_data", str(update_data_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {update_data_path}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_engine() -> Optional[Engine]:
    # Reuse env parsing and SSL handling from update_data.py
    try:
        module = _load_update_data_module()
        create_eng = getattr(module, "_create_mysql_engine", None)
        if not callable(create_eng):
            raise RuntimeError("_create_mysql_engine not found in update_data.py")
        return cast(Optional[Engine], create_eng("rw"))
    except Exception as e:
        print(f"[error] failed to create MySQL engine: {e}", file=sys.stderr)
        return None


def _impacted_dates(eng: Engine, season: str) -> list[str]:
    tbl = f"nhl_{season}_pbp"
    # Prefer Date column (string) if present.
    date_col = None
    for c in ("Date", "date"):
        if _column_exists(eng, tbl, c):
            date_col = c
            break
    if not date_col:
        return []

    reason_col = None
    for c in ("reason", "Reason"):
        if _column_exists(eng, tbl, c):
            reason_col = c
            break
    if not reason_col:
        return []

    sql = text(
        f"""
        SELECT DISTINCT {date_col} AS d
        FROM {tbl}
        WHERE {_lower_trim(reason_col)} IN ('short', 'failed-bank-attempt')
          AND {date_col} IS NOT NULL
        ORDER BY {date_col}
        """
    )
    with eng.begin() as conn:
        rows = conn.execute(sql).fetchall()
    return [str(r[0]) for r in rows if r and r[0] is not None]


def rebuild_gamedata_for_dates(season: str, dates: list[str]) -> int:
    if not dates:
        print("[skip] no impacted dates detected")
        return 0

    try:
        module = _load_update_data_module()
        fetch_day = getattr(module, "fetch_day", None)
        export_to_mysql = getattr(module, "export_to_mysql", None)
        if not callable(fetch_day) or not callable(export_to_mysql):
            raise RuntimeError("fetch_day/export_to_mysql not found in update_data.py")
        fetch_day = cast(Any, fetch_day)
        export_to_mysql = cast(Any, export_to_mysql)
    except Exception as e:
        print(f"[error] unable to load update_data functions: {e}", file=sys.stderr)
        return 2

    # Re-fetch and re-export each date; replace_date=True deletes & reloads the date slice.
    for d in dates:
        print(f"[rebuild] {d} ...")
        try:
            df_pbp, df_shifts, df_gd = fetch_day(d, with_xg=True)
            export_to_mysql(
                df_pbp,
                df_shifts,
                df_gd,
                season=str(season),
                date_str=str(d),
                replace_date=True,
            )
        except Exception as e:
            print(f"[error] rebuild failed for {d}: {e}", file=sys.stderr)
            return 3
    print(f"[done] rebuilt gamedata (and reloaded date slices): {len(dates)} date(s)")
    return 0


def backfill(season: str, *, rebuild_gamedata: bool = False, date: Optional[str] = None) -> int:
    eng = _get_engine()
    if eng is None:
        print("[error] MySQL engine not available (check env vars)", file=sys.stderr)
        return 2

    tables = [
        f"nhl_{season}_pbp",
        f"nhl_{season}_gamedata",
    ]

    # Columns: tolerate case differences by checking a few common spellings.
    reason_candidates = ["reason", "Reason"]
    xg_candidates = [
        ("xG_F", "xg_f"),
        ("xG_F2", "xg_f2"),
        ("xG_S", "xg_s"),
    ]

    any_updates = 0

    for t in tables:
        if not _table_exists(eng, t):
            print(f"[skip] table not found: {t}")
            continue

        reason_col = None
        for c in reason_candidates:
            if _column_exists(eng, t, c):
                reason_col = c
                break
        if not reason_col:
            print(f"[skip] no reason column in {t}")
            continue

        xg_cols: list[str] = []
        for preferred, alt in xg_candidates:
            if _column_exists(eng, t, preferred):
                xg_cols.append(preferred)
            elif _column_exists(eng, t, alt):
                xg_cols.append(alt)

        if not xg_cols:
            print(f"[skip] no xG columns in {t}")
            continue

        before = _count_matching(eng, t, reason_col)
        updated = _apply_update(eng, t, reason_col, xg_cols)
        after = _count_matching(eng, t, reason_col)

        any_updates += updated
        print(
            f"[ok] {t}: matched={before} updated={updated} matched_after={after} (set {', '.join(xg_cols)} to NULL)"
        )

    if any_updates == 0:
        print("[done] no rows updated")
    else:
        print(f"[done] updated rows (sum across tables): {any_updates}")

    if rebuild_gamedata:
        if date:
            dates = [date]
        else:
            dates = _impacted_dates(eng, season)
        rc = rebuild_gamedata_for_dates(season, dates)
        if rc != 0:
            return rc
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill xG NULLing rule in MySQL")
    p.add_argument("--season", default="20252026", help="Season code, e.g. 20252026")
    p.add_argument(
        "--rebuild-gamedata",
        action="store_true",
        help="Re-fetch and re-export impacted date slices to rebuild gamedata (and keep tables consistent)",
    )
    p.add_argument(
        "--date",
        help="If provided with --rebuild-gamedata, rebuild only this YYYY-MM-DD date slice",
    )
    args = p.parse_args(argv)

    season = str(args.season).strip()
    if not season.isdigit():
        print("[error] --season must be numeric like 20252026", file=sys.stderr)
        return 2

    date = (str(args.date).strip() if args.date else None)
    return backfill(season, rebuild_gamedata=bool(args.rebuild_gamedata), date=date)


if __name__ == "__main__":
    raise SystemExit(main())
