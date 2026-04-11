"""
Supabase client helper for the NHL app.

Usage:
    from app.supabase_client import get_client, read_table, upsert_df

    # --- single client ---
    sb = get_client()
    rows = sb.table("teams").select("*").execute()

    # --- pandas helpers ---
    df = read_table("pbp", columns="*", filters={"season": "eq.20252026"})
    upsert_df("pbp", df)

Environment variables (set in .env / Render / Supabase dashboard):
    SUPABASE_URL          – project URL   (e.g. https://xyzxyz.supabase.co)
    SUPABASE_SERVICE_KEY  – service_role key (NOT the anon key – we need write)
"""

import os
import math
import functools

import pandas as pd
from supabase import create_client, Client


# ── Supabase REST client (supabase-py) ───────────────────────────
@functools.lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


# ── Pandas helpers over REST API ─────────────────────────────────

def read_table(table: str, columns: str = "*", filters: dict | None = None,
               limit: int | None = None) -> pd.DataFrame:
    """Read a full table (auto-paginates past the 1 000-row REST limit).

    Args:
        table:   Supabase table name.
        columns: Comma-separated column list (default "*").
        filters: Dict of PostgREST filters, e.g. {"season": "eq.20252026"}.
        limit:   Max rows to return (None = all).
    """
    PAGE = 1000
    sb = get_client()
    rows: list[dict] = []
    offset = 0
    while True:
        q = sb.table(table).select(columns).range(offset, offset + PAGE - 1)
        if filters:
            for col, expr in filters.items():
                op, val = expr.split(".", 1)
                if op == 'in':
                    val_list = [v.strip() for v in val.strip('()').split(',') if v.strip()]
                    q = q.in_(col, val_list)
                else:
                    q = getattr(q, op)(col, val)
        batch = q.execute().data
        rows.extend(batch)
        if len(batch) < PAGE or (limit and len(rows) >= limit):
            break
        offset += PAGE
    if limit:
        rows = rows[:limit]
    return pd.DataFrame(rows)


_BATCH = 500  # Supabase REST max rows per insert


def upsert_df(table: str, df: pd.DataFrame, on_conflict: str = "") -> None:
    """Upsert a DataFrame into *table* in batches of 500.

    Args:
        table:       Supabase table name.
        df:          Data to upsert.
        on_conflict: Comma-separated PK columns for upsert conflict resolution.
                     If empty, does a plain insert.
    """
    sb = get_client()
    # Replace NaN/NaT with None for JSON serialisation
    records = df.where(df.notna(), None).to_dict(orient="records")
    # Convert numpy int/float to native Python types for JSON
    import numpy as np
    import math
    def _clean(v):
        if v is None or v is pd.NA:
            return None
        if isinstance(v, float) and math.isnan(v):
            return None
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            if np.isnan(v):
                return None
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        return v
    records = [{k: _clean(v) for k, v in r.items()} for r in records]
    for i in range(0, len(records), _BATCH):
        chunk = records[i : i + _BATCH]
        if on_conflict:
            sb.table(table).upsert(chunk, on_conflict=on_conflict).execute()
        else:
            sb.table(table).insert(chunk).execute()


def delete_rows(table: str, filters: dict) -> None:
    """Delete rows matching PostgREST filters.

    Args:
        table:   Supabase table name.
        filters: Dict of PostgREST filters, e.g. {"season": "eq.20252026"}.
    """
    sb = get_client()
    q = sb.table(table).delete()
    for col, expr in filters.items():
        op, val = expr.split(".", 1)
        q = getattr(q, op)(col, val)
    q.execute()


def truncate_table(table: str) -> None:
    """Delete ALL rows from a table. Use with care."""
    sb = get_client()
    # PostgREST requires at least one filter; use a tautology
    sb.table(table).delete().neq("ctid", "impossible").execute()
