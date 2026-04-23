from __future__ import annotations

import argparse
import pathlib
import sys

from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from scripts.export_preseason_updating_player_projections import CURRENT_TABLE, export_current_player_projections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export current next-game player projections")
    parser.add_argument("--table", default=CURRENT_TABLE, help="Supabase target table")
    parser.add_argument("--dry-run", action="store_true", help="Build rows without upserting")
    parser.add_argument("--refresh-data", action="store_true", help="Reload Moncton base inputs from DB")
    parser.add_argument(
        "--refresh-player-cache",
        action="store_true",
        help="Reload the Moncton player name cache from DB",
    )
    parser.add_argument(
        "--skip-ensure-table",
        action="store_true",
        help="Do not try to create the target table before upsert",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_current_player_projections(
        table=args.table,
        dry_run=args.dry_run,
        refresh_data=args.refresh_data,
        refresh_player_cache=args.refresh_player_cache,
        skip_ensure_table=args.skip_ensure_table,
    )


if __name__ == "__main__":
    main()