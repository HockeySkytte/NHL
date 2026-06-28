"""
Upload CSVs to Supabase via direct Postgres COPY (fastest bulk method).

Usage:
    python _upload_pg.py                    # upload both tables
    python _upload_pg.py --table nhl_current_playerprojections
    python _upload_pg.py --table nhl_player_metrics
"""
import sys, os, time, argparse, io
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, REPO_ROOT)
load_dotenv()

CSV_DIR = os.path.join(REPO_ROOT, 'data', 'player_projections')
DB_URL = os.getenv('DATABASE_HS_URL')
if not DB_URL:
    raise RuntimeError('DATABASE_HS_URL not set in .env')

eng = create_engine(DB_URL, connect_args={'connect_timeout': 60})


def table_exists(cursor, table: str) -> bool:
    cursor.execute(
        "SELECT EXISTS (SELECT FROM information_schema.tables WHERE table_schema = 'public' AND table_name = %s)",
        (table,),
    )
    return cursor.fetchone()[0]


def upload_via_copy(csv_name: str, table: str):
    """Upload CSV to Postgres using COPY (psycopg2 raw connection).

    Uses TRUNCATE + INSERT when the table already exists (preserving schema
    and indexes).  Falls back to CREATE TABLE if the table is missing.
    """
    path = os.path.join(CSV_DIR, csv_name)
    print(f'Reading {csv_name} …', end=' ', flush=True); t0 = time.time()
    df = pd.read_csv(path)
    print(f'{len(df):,} rows ({time.time()-t0:.0f}s)')

    # Fix integer ID columns: convert float to int string, NaN to empty
    for c in ['nhl_api_player_id', 'playerid', 'gameid', 'league', 'prior_games', 'gp']:
        if c in df.columns:
            df[c] = df[c].apply(lambda x: str(int(x)) if pd.notna(x) and x != '' else '')

    df = df.where(df.notna(), '')

    raw_conn = eng.raw_connection()
    cursor = raw_conn.cursor()

    if table_exists(cursor, table):
        # ── Table exists → truncate (fast, preserves schema & indexes) ──
        print(f'  Truncating {table} …', end=' ', flush=True)
        cursor.execute(f'TRUNCATE TABLE {table}')
        raw_conn.commit()
        print('OK')
    else:
        # ── Table missing → create from scratch ──
        col_defs = []
        for col in df.columns:
            if col in ('season', 'team', 'position', 'strengthstate', 'seasonstage', 'nhl_player_name'):
                col_defs.append(f'"{col}" text')
            elif col in ('league', 'playerid', 'gameid', 'prior_games', 'gp', 'nhl_api_player_id'):
                col_defs.append(f'"{col}" bigint')
            else:
                col_defs.append(f'"{col}" double precision')

        print(f'  Creating {table} …', end=' ', flush=True)
        cursor.execute(f'CREATE TABLE {table} ({", ".join(col_defs)})')
        raw_conn.commit()
        print('OK')

        # ── Add indexes (only on first create) ──
        print(f'  Indexing …', end=' ', flush=True); ti = time.time()
        if table == 'nhl_player_metrics':
            cursor.execute('ALTER TABLE nhl_player_metrics ADD PRIMARY KEY (season, playerid, gameid, strengthstate)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_npm_player ON nhl_player_metrics (nhl_api_player_id, season)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_npm_game ON nhl_player_metrics (gameid, season)')
        else:
            cursor.execute('ALTER TABLE nhl_current_playerprojections ADD PRIMARY KEY (season, playerid, strengthstate)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_ncp_player ON nhl_current_playerprojections (nhl_api_player_id, season)')
        raw_conn.commit()
        print(f'done ({time.time()-ti:.0f}s)')

    # ── COPY data in ──
    print(f'  COPY {len(df):,} rows in chunks …', flush=True); t0 = time.time()
    columns = ','.join(f'"{c}"' for c in df.columns)
    copy_sql = f'COPY {table} ({columns}) FROM STDIN WITH (FORMAT CSV)'

    chunk_size = 100000
    chunks = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    for i, chunk in enumerate(chunks):
        buf = io.StringIO()
        chunk.to_csv(buf, index=False, header=False)
        buf.seek(0)
        cursor.copy_expert(copy_sql, buf)
        raw_conn.commit()
        pct = min((i+1)*chunk_size, len(df)) * 100 // len(df)
        elapsed = time.time() - t0
        rate = min((i+1)*chunk_size, len(df)) / elapsed if elapsed > 0 else 0
        print(f'    {min((i+1)*chunk_size, len(df)):,}/{len(df):,} ({pct}%, {rate:.0f} rows/s)', flush=True)
    print(f'  COPY done ({time.time()-t0:.0f}s)')

    cursor.close()
    raw_conn.close()


# ── Parse args ────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='Upload CSVs to Supabase via COPY')
parser.add_argument('--table', type=str, default=None,
                    help='Table name to upload (default: both nhl_player_metrics and nhl_current_playerprojections)')
args = parser.parse_args()

TABLES = {
    'nhl_player_metrics': 'nhl_player_metrics.csv',
    'nhl_current_playerprojections': 'nhl_current_playerprojections.csv',
}

targets = [args.table] if args.table else list(TABLES.keys())
for tbl in targets:
    csv_name = TABLES.get(tbl)
    if not csv_name:
        print(f'Unknown table: {tbl}')
        sys.exit(1)
    upload_via_copy(csv_name, tbl)

# ── Verify ────────────────────────────────────────────────────────
print('\nVerifying …')
with eng.connect() as conn:
    for tbl in targets:
        r = conn.execute(text(f'SELECT COUNT(*) FROM {tbl}'))
        print(f'  {tbl}: {r.scalar():,} rows')

print('Done.')
