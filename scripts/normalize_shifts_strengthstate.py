"""Normalize StrengthState values in nhl_{season}_shifts.

Problem:
- The shifts scraper emits per-slice StrengthState strings.
- Historical data may include hundreds of distinct numeric states (e.g. '10v8') due to
  duplicate/overlapping shift rows in the HTML reports.

This script collapses numeric NvM StrengthState values into a small expected set:
  5v5, 4v4, 3v3,
  5v4, 5v3, 4v3,
  4v5, 3v5, 3v4,
  1v0, 0v1,
  plus existing ENF/ENA are preserved as-is.

Run:
    python scripts/normalize_shifts_strengthstate.py --season 20252026 --apply

Defaults to dry-run (no updates) unless --apply is provided.
"""

from __future__ import annotations

import argparse
import re
from typing import Dict, Iterable, Optional, Tuple

from sqlalchemy import text

import update_data


_ALLOWED = {
    "5v5",
    "4v4",
    "3v3",
    "5v4",
    "5v3",
    "4v3",
    "4v5",
    "3v5",
    "3v4",
    "ENF",
    "ENA",
    "1v0",
    "0v1",
}


_NUM_RE = re.compile(r"^(\d+)v(\d+)$")


def _normalize_numeric_strength(ms: int, ts: int) -> str:
    ms = max(0, min(int(ms), 6))
    ts = max(0, min(int(ts), 6))

    # Collapse extra-skater overlap (6v5/7v6/...) into expected states.
    if ms >= 5 and ts >= 5:
        return "5v5"

    # PP / SH first so we don't collapse 5v4 into 4v4, etc.
    if ts == 4 and ms >= 5:
        return "5v4"
    if ts == 3 and ms >= 5:
        return "5v3"
    if ts == 3 and ms == 4:
        return "4v3"

    if ms == 4 and ts >= 5:
        return "4v5"
    if ms == 3 and ts >= 5:
        return "3v5"
    if ms == 3 and ts == 4:
        return "3v4"

    # Even strength
    if ms == 4 and ts == 4:
        return "4v4"
    if ms == 3 and ts == 3:
        return "3v3"

    # Extreme low-count fallbacks.
    if ts == 0 and ms >= 1:
        return "1v0"
    if ms == 0 and ts >= 1:
        return "0v1"

    # Snap to closest even-strength when data is incomplete.
    m = max(ms, ts)
    if m <= 3:
        return "3v3"
    if m == 4:
        return "4v4"
    return "5v5"


def _build_mapping(states: Iterable[Optional[str]]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for v in states:
        s = str(v or "").strip()
        if not s:
            continue
        if s in _ALLOWED:
            continue
        m = _NUM_RE.match(s)
        if not m:
            # Leave non-numeric unexpected labels untouched (caller can decide later).
            continue
        ms = int(m.group(1))
        ts = int(m.group(2))
        new = _normalize_numeric_strength(ms, ts)
        if new != s:
            out[s] = new
    return out


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Normalize StrengthState values in nhl_{season}_shifts")
    ap.add_argument("--season", default="20252026")
    ap.add_argument("--apply", action="store_true", help="Apply UPDATEs (default: dry-run)")
    args = ap.parse_args(argv)

    season = str(args.season)
    tbl = f"nhl_{season}_shifts"

    eng = update_data._create_mysql_engine("rw")
    if eng is None:
        raise SystemExit("MySQL engine not available (check env vars)")

    with eng.connect() as conn:
        total = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar() or 0
        distinct = conn.execute(text(f"SELECT COUNT(DISTINCT StrengthState) FROM {tbl}")).scalar() or 0
        print(f"[before] rows={int(total)} distinct StrengthState={int(distinct)}")

        states = [r[0] for r in conn.execute(text(f"SELECT DISTINCT StrengthState FROM {tbl}")).fetchall()]

    mapping = _build_mapping(states)
    print(f"Will normalize {len(mapping)} numeric StrengthState values")

    # Show a small preview
    for i, (old, new) in enumerate(sorted(mapping.items())[:30]):
        print(f"  {old} -> {new}")
        if i >= 29:
            break

    if not args.apply:
        print("Dry-run only. Re-run with --apply to update MySQL.")
        return 0

    # Apply updates by value (fast: small number of distinct values)
    with eng.begin() as conn:
        for old, new in mapping.items():
            conn.execute(
                text(f"UPDATE {tbl} SET StrengthState = :new WHERE StrengthState = :old"),
                {"new": new, "old": old},
            )

    with eng.connect() as conn:
        distinct_after = conn.execute(text(f"SELECT COUNT(DISTINCT StrengthState) FROM {tbl}")).scalar() or 0
        print(f"[after] distinct StrengthState={int(distinct_after)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
