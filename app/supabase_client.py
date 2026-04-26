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
import re

import pandas as pd
from supabase import create_client, Client


# ── Supabase REST client (supabase-py) ───────────────────────────
@functools.lru_cache(maxsize=1)
def get_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SERVICE_KEY"]
    return create_client(url, key)


def auth_is_configured() -> bool:
    return bool(os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_SERVICE_KEY"))


def create_auth_client(*, admin: bool = False) -> Client:
    url = os.environ["SUPABASE_URL"]
    if admin:
        key = os.environ["SUPABASE_SERVICE_KEY"]
    else:
        key = (
            os.getenv("SUPABASE_ANON_KEY")
            or os.getenv("SUPABASE_PUBLISHABLE_KEY")
            or os.environ["SUPABASE_SERVICE_KEY"]
        )
    return create_client(url, key)


def _to_plain(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {k: _to_plain(v) for k, v in value.items()}

    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _to_plain(model_dump())
        except Exception:
            pass

    dict_method = getattr(value, "dict", None)
    if callable(dict_method):
        try:
            return _to_plain(dict_method())
        except Exception:
            pass

    data = getattr(value, "data", None)
    user = getattr(value, "user", None)
    session = getattr(value, "session", None)
    if any(part is not None for part in (data, user, session)):
        out = {}
        if data is not None:
            out["data"] = _to_plain(data)
        if user is not None:
            out["user"] = _to_plain(user)
        if session is not None:
            out["session"] = _to_plain(session)
        return out

    try:
        raw = vars(value)
    except Exception:
        raw = None
    if isinstance(raw, dict):
        return {k: _to_plain(v) for k, v in raw.items() if not str(k).startswith("_")}

    return value


def _is_missing_table_error(exc: Exception, table: str) -> bool:
    raw = str(_to_plain(exc) or exc or '').lower()
    table_name = table.lower()
    return table_name in raw and (
        'could not find the table' in raw
        or 'schema cache' in raw
        or 'pgrst205' in raw
    )


def _missing_column_name(exc: Exception, table: str) -> str | None:
    raw = str(_to_plain(exc) or exc or '')
    lowered = raw.lower()
    if table.lower() not in lowered:
        return None
    # Typical Postgres error: column "subscription_source" of relation "user_accounts" does not exist
    match = re.search(r'column\s+"([a-zA-Z0-9_]+)"\s+of\s+relation\s+"%s"\s+does\s+not\s+exist' % re.escape(table), raw, flags=re.IGNORECASE)
    if match:
        return str(match.group(1) or '').strip()
    return None


def auth_admin_create_user(email: str, password: str, *, user_metadata: dict | None = None) -> dict:
    client = create_auth_client(admin=True)
    response = client.auth.admin.create_user({
        "email": email,
        "password": password,
        "email_confirm": True,
        "user_metadata": user_metadata or {},
    })
    return _to_plain(response) or {}


def auth_admin_get_user(uid: str) -> dict | None:
    client = create_auth_client(admin=True)
    response = client.auth.admin.get_user_by_id(uid)
    plain = _to_plain(response) or {}
    if isinstance(plain, dict) and isinstance(plain.get("user"), dict):
        return plain.get("user")
    return plain if isinstance(plain, dict) else None


def auth_admin_list_users(page: int | None = None, per_page: int | None = None) -> list[dict]:
    client = create_auth_client(admin=True)
    response = client.auth.admin.list_users(page=page, per_page=per_page)
    plain = _to_plain(response) or []
    return plain if isinstance(plain, list) else []


def auth_admin_update_user(uid: str, attributes: dict) -> dict | None:
    client = create_auth_client(admin=True)
    response = client.auth.admin.update_user_by_id(uid, attributes)
    plain = _to_plain(response) or {}
    if isinstance(plain, dict) and isinstance(plain.get("user"), dict):
        return plain.get("user")
    return plain if isinstance(plain, dict) else None


def auth_admin_delete_user(uid: str, *, should_soft_delete: bool = False) -> None:
    client = create_auth_client(admin=True)
    client.auth.admin.delete_user(uid, should_soft_delete=should_soft_delete)


def auth_sign_in_with_password(email: str, password: str) -> dict:
    client = create_auth_client(admin=False)
    response = client.auth.sign_in_with_password({
        "email": email,
        "password": password,
    })
    return _to_plain(response) or {}


def get_user_account(auth_user_id: str) -> dict | None:
    if not auth_user_id:
        return None
    client = get_client()
    try:
        response = (
            client.table("user_accounts")
            .select("*")
            .eq("auth_user_id", auth_user_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        if _is_missing_table_error(exc, 'user_accounts'):
            return None
        raise
    rows = _to_plain(getattr(response, 'data', None)) or []
    return rows[0] if rows else None


def upsert_user_account(record: dict) -> dict | None:
    auth_user_id = str((record or {}).get('auth_user_id') or '').strip()
    if not auth_user_id:
        raise ValueError('auth_user_id is required for user account upsert')
    payload = dict(record or {})
    client = get_client()
    # Some deployed environments may lag migrations. If a column is missing,
    # retry without that column so core account sync still succeeds.
    for _ in range(10):
        try:
            client.table('user_accounts').upsert(payload, on_conflict='auth_user_id').execute()
            break
        except Exception as exc:
            if _is_missing_table_error(exc, 'user_accounts'):
                return None
            missing_col = _missing_column_name(exc, 'user_accounts')
            if missing_col and missing_col in payload:
                payload.pop(missing_col, None)
                continue
            raise
    return get_user_account(auth_user_id)


def list_user_accounts() -> list[dict]:
    client = get_client()
    try:
        response = client.table('user_accounts').select('*').order('created_at', desc=True).execute()
    except Exception as exc:
        if _is_missing_table_error(exc, 'user_accounts'):
            return []
        raise
    rows = _to_plain(getattr(response, 'data', None)) or []
    return rows if isinstance(rows, list) else []


def delete_user_account(auth_user_id: str) -> None:
    if not auth_user_id:
        return
    client = get_client()
    try:
        client.table('user_accounts').delete().eq('auth_user_id', auth_user_id).execute()
    except Exception as exc:
        if _is_missing_table_error(exc, 'user_accounts'):
            return
        raise


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
