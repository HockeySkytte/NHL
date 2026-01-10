"""One-time exporter for MySQL table -> CSV in app/static.

Defaults:
- Table: nhl_seasonstats
- Output: app/static/nhl_seasonstats.csv

Uses the same env var conventions as app/routes.py admin_db_check:
- DATABASE_URL_RO / DB_URL_RO / DATABASE_URL (for --mode ro)
- DATABASE_URL_RW / DB_URL_RW / DATABASE_URL (for --mode rw)
- Or DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME (default DB_NAME=public)

Optional SSL env vars:
- DB_SSL_CA, DB_SSL_CERT, DB_SSL_KEY
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import tempfile
from pathlib import Path
from typing import Dict, Any


_TABLE_RE = re.compile(r"^[A-Za-z0-9_]+$")


def build_db_url(mode: str) -> str:
    def _first_env(*names: str) -> str | None:
        for n in names:
            v = os.getenv(n)
            if v:
                return v
        return None

    def _part(name: str, fallback: str) -> str:
        return os.getenv(name, fallback)

    mode = (mode or "rw").lower().strip()

    # URL precedence mirrors scripts/update_data.py
    if mode == "ro":
        db_url = _first_env("DATABASE_URL_RO", "DB_URL_RO", "DATABASE_URL")
        host_override = _first_env("DB_HOST_RO") or os.getenv("DB_HOST")
    else:
        db_url = _first_env("DATABASE_URL_RW", "DB_URL_RW", "DATABASE_URL")
        host_override = _first_env("DB_HOST_RW") or os.getenv("DB_HOST")

    # If DATABASE_URL_* uses localhost but a host override is provided, swap it
    if db_url:
        if host_override and "@localhost" in db_url:
            db_url = db_url.replace("@localhost", f"@{host_override}")
        return db_url

    # Build from discrete vars with suffix-specific variables
    if mode == "ro":
        user = _first_env("DB_USER_RO") or _part("DB_USER", "root")
        pwd = _first_env("DB_PASSWORD_RO") or _part("DB_PASSWORD", "Sunesen1")
        host = _first_env("DB_HOST_RO") or _part("DB_HOST", "localhost")
        port = _first_env("DB_PORT_RO") or _part("DB_PORT", "3306")
        name = _first_env("DB_NAME_RO") or _part("DB_NAME", "public")
    else:
        user = _first_env("DB_USER_RW") or _part("DB_USER", "root")
        pwd = _first_env("DB_PASSWORD_RW") or _part("DB_PASSWORD", "Sunesen1")
        host = _first_env("DB_HOST_RW") or _part("DB_HOST", "localhost")
        port = _first_env("DB_PORT_RW") or _part("DB_PORT", "3306")
        name = _first_env("DB_NAME_RW") or _part("DB_NAME", "public")

    return f"mysql+mysqlconnector://{user}:{pwd}@{host}:{port}/{name}"


def redact_db_url(db_url: str) -> str:
    # Best-effort redaction: mysql+driver://user:pass@host/... -> user:***
    try:
        return re.sub(r"://([^:/?#]+):[^@]*@", r"://\1:***@", db_url)
    except Exception:
        return "<redacted>"


def build_connect_args() -> Dict[str, Any]:
    connect_args: Dict[str, Any] = {}
    if os.getenv("DB_SSL_CA"):
        connect_args["ssl_ca"] = os.getenv("DB_SSL_CA")
    if os.getenv("DB_SSL_CERT"):
        connect_args["ssl_cert"] = os.getenv("DB_SSL_CERT")
    if os.getenv("DB_SSL_KEY"):
        connect_args["ssl_key"] = os.getenv("DB_SSL_KEY")
    return connect_args


def export_table_to_csv(*, db_url: str, table: str, out_path: Path) -> None:
    if not _TABLE_RE.match(table):
        raise ValueError(f"Invalid table name: {table!r}")

    try:
        from sqlalchemy import create_engine, text  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "SQLAlchemy import failed. Ensure requirements are installed (SQLAlchemy, mysql-connector-python)."
        ) from e

    connect_args = build_connect_args()
    if connect_args:
        engine = create_engine(db_url, connect_args=connect_args, pool_pre_ping=True)
    else:
        engine = create_engine(db_url, pool_pre_ping=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Write to temp file first, then replace atomically.
    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", delete=False, dir=str(out_path.parent), suffix=".tmp.csv"
    ) as tmp_f:
        tmp_path = Path(tmp_f.name)
        writer = csv.writer(tmp_f)

        with engine.connect() as conn:
            # Stream results where supported by driver.
            result = conn.execution_options(stream_results=True).execute(
                text(f"SELECT * FROM `{table}`")
            )
            headers = list(result.keys())
            writer.writerow(headers)
            for row in result:
                writer.writerow(list(row))

    tmp_path.replace(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export nhl_seasonstats from MySQL to app/static CSV")
    parser.add_argument("--mode", choices=["ro", "rw"], default="ro", help="Which DB URL to use")
    parser.add_argument(
        "--db-url",
        default="",
        help="Optional explicit SQLAlchemy DB URL (overrides env vars). Example: mysql+mysqlconnector://user:pass@host:3306/public",
    )
    parser.add_argument("--table", default="nhl_seasonstats", help="Table name to export")
    parser.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "app" / "static" / "nhl_seasonstats.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    db_url = str(args.db_url).strip() or build_db_url(args.mode)
    out_path = Path(args.out).resolve()

    print(f"Using DB: {redact_db_url(db_url)}")

    export_table_to_csv(db_url=db_url, table=str(args.table).strip(), out_path=out_path)
    print(f"Wrote CSV: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
