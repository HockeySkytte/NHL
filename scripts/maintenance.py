"""
maintenance.py

Run ANALYZE and VACUUM on Supabase tables to update query planner
statistics and reclaim storage.

Usage:
  python scripts/maintenance.py [--analyze] [--vacuum] [--full]
                                 [--table <name>] [--print-sql]

Options:
  --analyze       Run ANALYZE on all tables (default if no flags given)
  --vacuum        Run VACUUM (non-blocking) on all tables
  --full          Run VACUUM FULL (exclusive lock — use with caution)
  --table <name>  Operate on a single table instead of all
  --print-sql     Print SQL instead of executing (useful when direct
                  Postgres connection is unavailable)

When direct Postgres connection fails, the script falls back to printing
the SQL statements for manual execution in the Supabase SQL editor.
"""
from __future__ import annotations

import argparse
import os
import sys
import re
from typing import List, Optional

# Ensure the app module is importable
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# All user-facing tables in dependency-safe order (no FK violations)
ALL_TABLES = [
    'teams',
    'players',
    'pbp',
    'shifts',
    'game_data',
    'season_stats',
    'season_stats_teams',
    'player_projections',
    'player_current_projections',
    'player_game_projections',
    'rapm',
    'rapm_context',
    'rapm_data',
    'odds_history',
    'lineups',
    'started_overrides',
    'last_dates',
    'box_ids',
    'model_fenwick',
    'model_shot',
    'game_model_preseason',
    'card_builder_layouts',
    'user_accounts',
]


def _build_sql(tables: List[str], *, analyze: bool, vacuum: bool, full: bool) -> List[str]:
    """Build the SQL statements for maintenance operations."""
    statements: List[str] = []
    for tbl in tables:
        if analyze:
            statements.append(f'ANALYZE "{tbl}";')
        if vacuum and not full:
            statements.append(f'VACUUM "{tbl}";')
        if full:
            statements.append(f'VACUUM FULL "{tbl}";')
    return statements


def _execute_sql(statements: List[str], db_url: str) -> bool:
    """Execute SQL statements via direct Postgres connection.

    Returns True if all statements succeeded, False otherwise.
    """
    try:
        from sqlalchemy import create_engine, text as sa_text
        from sqlalchemy.engine.url import make_url

        parsed = make_url(db_url)
        eng = create_engine(parsed)

        with eng.connect() as conn:
            # VACUUM can't run inside a transaction
            conn.execute(sa_text("COMMIT;"))
            for stmt in statements:
                print(f'  {stmt}')
                try:
                    conn.execute(sa_text(stmt))
                except Exception as e:
                    print(f'    [warn] {e}')
            print(f'  Done — {len(statements)} statements executed.')
        return True
    except Exception as e:
        print(f'[error] Direct Postgres connection failed: {e}')
        return False


def _print_sql(statements: List[str]) -> None:
    """Print SQL statements for manual execution."""
    print('\n-- Copy the following into the Supabase SQL editor:\n')
    for stmt in statements:
        print(stmt)
    print(f'\n-- {len(statements)} statements total.')


def _redact(s: str) -> str:
    try:
        return re.sub(r'://([^:/?#]+):[^@]*@', r'://\1:***@', s)
    except Exception:
        return '<redacted>'


def main() -> int:
    parser = argparse.ArgumentParser(description='Supabase table maintenance (ANALYZE / VACUUM)')
    parser.add_argument('--analyze', action='store_true', help='Run ANALYZE on tables')
    parser.add_argument('--vacuum', action='store_true', help='Run VACUUM (non-blocking)')
    parser.add_argument('--full', action='store_true', help='Run VACUUM FULL (exclusive lock)')
    parser.add_argument('--table', help='Operate on a single table instead of all')
    parser.add_argument('--db-url', help='Explicit Postgres URL (default: SUPABASE_DB_URL env var)')
    parser.add_argument('--print-sql', action='store_true', help='Print SQL instead of executing')
    args = parser.parse_args()

    # Default: ANALYZE only (safe, non-blocking)
    if not args.analyze and not args.vacuum and not args.full:
        args.analyze = True

    if args.full and not args.vacuum:
        print('[warn] --full implies --vacuum; enabling VACUUM FULL mode.')
        args.vacuum = True

    tables = [args.table] if args.table else ALL_TABLES
    print(f'Target tables: {len(tables)}')
    print(f'Operations: {"ANALYZE " if args.analyze else ""}'
          f'{"VACUUM " if args.vacuum and not args.full else ""}'
          f'{"VACUUM FULL " if args.full else ""}'.strip())

    statements = _build_sql(tables, analyze=args.analyze, vacuum=args.vacuum, full=args.full)

    if args.print_sql:
        _print_sql(statements)
        return 0

    # Try direct Postgres connection
    db_url = args.db_url or os.getenv('SUPABASE_DB_URL', '')
    if db_url:
        print(f'\nConnecting to: {_redact(db_url)}')
        if _execute_sql(statements, db_url):
            return 0
        print('Falling back to printing SQL for manual execution...')

    _print_sql(statements)
    print('\n[info] Set SUPABASE_DB_URL env var or use --db-url for direct execution.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
