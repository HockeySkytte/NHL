"""Build team season stats from MySQL PBP + shifts and write to MySQL + Google Sheets.

Goal:
- Derive a season-level team stats table from raw season tables:
    nhl_{season}_pbp
    nhl_{season}_shifts
- Create/replace:
    seasonstats_team_{season}
- Write the result to a Google Sheets worksheet (default: Sheets7).

Key rule:
- For TOI, do NOT sum all player shifts. We only want ONE Duration per Team per ShiftIndex
  (dedup by GameID+Team+StrengthStateBucket+ShiftIndex and take MAX(Duration)).

Env vars (DB):
- DATABASE_URL_RO / DB_URL_RO / DATABASE_URL (for --mode ro)
- DATABASE_URL_RW / DB_URL_RW / DATABASE_URL (for --mode rw)
- Or DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME (default DB_NAME=public)

Env vars (Google):
- GOOGLE_SERVICE_ACCOUNT_JSON_PATH or GOOGLE_SERVICE_ACCOUNT_JSON(_B64)
- GOOGLE_SHEETS_ID (optional default for --sheet-id)

Usage:
  python scripts/build_seasonstats_team.py --season 20252026 --sheet-id <docId> --worksheet Sheets7
"""

from __future__ import annotations

import argparse
import json
import os
import re
from datetime import datetime, date, timezone
from typing import Any, Dict, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


_SEASON_RE = re.compile(r"^\d{8}$")


def _first_env(*names: str) -> Optional[str]:
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return None


def build_db_url(mode: str) -> str:
    def _part(name: str, fallback: str) -> str:
        return os.getenv(name, fallback)

    mode = (mode or "rw").lower().strip()

    if mode == "ro":
        db_url = _first_env("DATABASE_URL_RO", "DB_URL_RO", "DATABASE_URL")
        host_override = _first_env("DB_HOST_RO") or os.getenv("DB_HOST")
    else:
        db_url = _first_env("DATABASE_URL_RW", "DB_URL_RW", "DATABASE_URL")
        host_override = _first_env("DB_HOST_RW") or os.getenv("DB_HOST")

    if db_url:
        if host_override and "@localhost" in db_url:
            db_url = db_url.replace("@localhost", f"@{host_override}")
        return db_url

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
    try:
        return re.sub(r"://([^:/?#]+):[^@]*@", r"://\1:***@", db_url)
    except Exception:
        return "<redacted>"


def _load_google_service_account_info() -> Dict[str, Any]:
    path = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_PATH")
    if path:
        p = str(path).strip().strip('"').strip("'")
        p = os.path.expandvars(os.path.expanduser(p))
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
    else:
        raw = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

    raw_b64 = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON_B64")
    if raw_b64:
        import base64

        s = str(raw_b64).strip().strip('"').strip("'")
        s = "".join(s.split())
        pad = (-len(s)) % 4
        if pad:
            s = s + ("=" * pad)
        try:
            raw = base64.b64decode(s.encode("utf-8"), validate=False).decode("utf-8")
        except Exception:
            raw = base64.urlsafe_b64decode(s.encode("utf-8")).decode("utf-8")

    if not raw:
        raise RuntimeError(
            "Missing Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON_PATH, GOOGLE_SERVICE_ACCOUNT_JSON_B64, or GOOGLE_SERVICE_ACCOUNT_JSON."
        )
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"Invalid Google service account JSON: {e}")


def _write_dataframe_to_google_sheet(df: pd.DataFrame, *, sheet_id: str, worksheet: str) -> None:
    try:
        import gspread  # type: ignore
        from google.oauth2.service_account import Credentials  # type: ignore
    except Exception as e:
        raise RuntimeError(f"Google Sheets dependencies missing: {e}. Install: pip install gspread google-auth")

    info = _load_google_service_account_info()
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    creds = Credentials.from_service_account_info(info, scopes=scopes)
    gc = gspread.authorize(creds)

    sh = gc.open_by_key(sheet_id)
    try:
        ws = sh.worksheet(worksheet)
    except Exception:
        rows = max(2, int(len(df) + 1))
        cols = max(1, int(len(df.columns)))
        ws = sh.add_worksheet(title=worksheet, rows=rows, cols=cols)

    df_out = df.copy()
    try:
        ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    except Exception:
        ts = ""
    if "TimestampUTC" in df_out.columns:
        df_out["TimestampUTC"] = ts
    else:
        df_out.insert(0, "TimestampUTC", ts)

    df_out = df_out.where(pd.notnull(df_out), "")

    def _cell(v: Any) -> Any:
        try:
            if hasattr(v, "item"):
                v = v.item()
        except Exception:
            pass
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        return v

    values = [list(df_out.columns)]
    for row in df_out.itertuples(index=False, name=None):
        values.append([_cell(v) for v in row])

    ws.clear()
    update_fn = getattr(ws, "update")
    try:
        update_fn(range_name="A1", values=values)
    except TypeError:
        update_fn("A1", values)


def _strength_bucket_sql(col: str) -> str:
    # Mirrors scripts/update_data.py _seasonstats_strength_bucket
    s = f"LOWER(TRIM({col}))"
    left = f"CAST(SUBSTRING_INDEX({s},'v',1) AS UNSIGNED)"
    right = f"CAST(SUBSTRING_INDEX({s},'v',-1) AS UNSIGNED)"
    return f"""
      CASE
        WHEN {col} IS NULL OR TRIM({col})='' THEN 'Other'
        WHEN {s} LIKE 'en%%' THEN 'Other'
        WHEN {s} IN ('pp','sh') THEN UPPER({s})
        WHEN {s}='5v5' THEN '5v5'
        WHEN LOCATE('v', {s})>0 THEN
          CASE
            WHEN {left} >= 5 AND {right} >= 5 THEN '5v5'
            WHEN {right} IN (3,4) AND {left} > {right} THEN 'PP'
            WHEN {left} IN (3,4) AND {right} > {left} THEN 'SH'
            ELSE 'Other'
          END
        ELSE 'Other'
      END
    """.strip()


def rebuild_team_seasonstats(*, eng, season: str) -> pd.DataFrame:
    if not _SEASON_RE.match(season):
        raise ValueError("season must be YYYYYYYY (e.g. 20252026)")

    tbl_pbp = f"nhl_{season}_pbp"
    tbl_sh = f"nhl_{season}_shifts"
    out_tbl = f"seasonstats_team_{season}"

    pbp_bucket = _strength_bucket_sql("p.StrengthState")

    sql = f"""
WITH
  game_ss AS (
    SELECT
      GameID,
            MIN(COALESCE(NULLIF(TRIM(SeasonState),''),'regular')) AS SeasonState
    FROM {tbl_pbp}
    GROUP BY GameID
  ),
  shift_dedup AS (
    SELECT
      s.GameID,
      UPPER(TRIM(s.Team)) AS Team,
      COALESCE(NULLIF(TRIM(s.StrengthStateBucket),''), 'Other') AS StrengthState,
      s.ShiftIndex,
      MAX(COALESCE(s.Duration,0)) AS DurationSec
    FROM {tbl_sh} s
    GROUP BY s.GameID, UPPER(TRIM(s.Team)), COALESCE(NULLIF(TRIM(s.StrengthStateBucket),''), 'Other'), s.ShiftIndex
  ),
  toi AS (
    SELECT
      gm.SeasonState AS SeasonState,
      sd.StrengthState AS StrengthState,
      sd.Team AS Team,
      COUNT(DISTINCT sd.GameID) AS GP,
      SUM(sd.DurationSec)/60.0 AS TOI
    FROM shift_dedup sd
    JOIN game_ss gm ON gm.GameID = sd.GameID
    GROUP BY gm.SeasonState, sd.StrengthState, sd.Team
  ),
  pbp_base AS (
    SELECT
      COALESCE(NULLIF(TRIM(p.SeasonState),''),'regular') AS SeasonState,
      {pbp_bucket} AS StrengthState,
      UPPER(TRIM(p.EventTeam)) AS EventTeam,
      UPPER(TRIM(p.Opponent)) AS Opponent,
      COALESCE(p.Corsi,0) AS Corsi,
      COALESCE(p.Fenwick,0) AS Fenwick,
      COALESCE(p.Shot,0) AS Shot,
      COALESCE(p.Goal,0) AS Goal,
      COALESCE(p.xG_F,0.0) AS xG_F,
      COALESCE(p.xG_S,0.0) AS xG_S,
      COALESCE(p.xG_F2,0.0) AS xG_F2,
      COALESCE(p.PEN_duration,0.0) AS PEN_duration,
      COALESCE(p.Period,0) AS Period
    FROM {tbl_pbp} p
    WHERE p.EventTeam IS NOT NULL AND TRIM(p.EventTeam)<>''
      AND p.Opponent IS NOT NULL AND TRIM(p.Opponent)<>''
      AND NOT (COALESCE(NULLIF(TRIM(p.SeasonState),''),'regular')='regular' AND COALESCE(p.Period,0)=5)
  ),
  pbp_for AS (
    SELECT
      SeasonState,
      StrengthState,
      EventTeam AS Team,
      SUM(CAST(Corsi AS SIGNED)) AS CF,
      SUM(CAST(Fenwick AS SIGNED)) AS FF,
      SUM(CAST(Shot AS SIGNED)) AS SF,
      SUM(CAST(Goal AS SIGNED)) AS GF,
      SUM(CAST(xG_F AS DOUBLE)) AS xGF_F,
      SUM(CAST(xG_S AS DOUBLE)) AS xGF_S,
      SUM(CAST(xG_F2 AS DOUBLE)) AS xGF_F2,
      SUM(CASE WHEN CAST(PEN_duration AS DOUBLE) > 0 THEN CAST(PEN_duration AS DOUBLE) ELSE 0 END) AS PIM_against
    FROM pbp_base
    GROUP BY SeasonState, StrengthState, EventTeam
  ),
  pbp_against AS (
    SELECT
      SeasonState,
      StrengthState,
      Opponent AS Team,
      SUM(CAST(Corsi AS SIGNED)) AS CA,
      SUM(CAST(Fenwick AS SIGNED)) AS FA,
      SUM(CAST(Shot AS SIGNED)) AS SA,
      SUM(CAST(Goal AS SIGNED)) AS GA,
      SUM(CAST(xG_F AS DOUBLE)) AS xGA_F,
      SUM(CAST(xG_S AS DOUBLE)) AS xGA_S,
      SUM(CAST(xG_F2 AS DOUBLE)) AS xGA_F2,
      SUM(CASE WHEN CAST(PEN_duration AS DOUBLE) > 0 THEN CAST(PEN_duration AS DOUBLE) ELSE 0 END) AS PIM_for
    FROM pbp_base
    GROUP BY SeasonState, StrengthState, Opponent
  ),
  teams AS (
    SELECT SeasonState, StrengthState, Team FROM toi
    UNION
    SELECT SeasonState, StrengthState, Team FROM pbp_for
    UNION
    SELECT SeasonState, StrengthState, Team FROM pbp_against
  )
SELECT
  {int(season)} AS Season,
  t.SeasonState,
  t.StrengthState,
  t.Team,
  COALESCE(toi.GP, 0) AS GP,
  COALESCE(toi.TOI, 0.0) AS TOI,
  COALESCE(pf.CF, 0) AS CF,
  COALESCE(pa.CA, 0) AS CA,
  COALESCE(pf.FF, 0) AS FF,
  COALESCE(pa.FA, 0) AS FA,
  COALESCE(pf.SF, 0) AS SF,
  COALESCE(pa.SA, 0) AS SA,
  COALESCE(pf.GF, 0) AS GF,
  COALESCE(pa.GA, 0) AS GA,
  COALESCE(pf.xGF_F, 0.0) AS xGF_F,
  COALESCE(pa.xGA_F, 0.0) AS xGA_F,
  COALESCE(pf.xGF_S, 0.0) AS xGF_S,
  COALESCE(pa.xGA_S, 0.0) AS xGA_S,
  COALESCE(pf.xGF_F2, 0.0) AS xGF_F2,
  COALESCE(pa.xGA_F2, 0.0) AS xGA_F2,
  COALESCE(pa.PIM_for, 0.0) AS PIM_for,
  COALESCE(pf.PIM_against, 0.0) AS PIM_against
FROM teams t
LEFT JOIN toi ON toi.SeasonState=t.SeasonState AND toi.StrengthState=t.StrengthState AND toi.Team=t.Team
LEFT JOIN pbp_for pf ON pf.SeasonState=t.SeasonState AND pf.StrengthState=t.StrengthState AND pf.Team=t.Team
LEFT JOIN pbp_against pa ON pa.SeasonState=t.SeasonState AND pa.StrengthState=t.StrengthState AND pa.Team=t.Team
ORDER BY t.SeasonState, t.StrengthState, t.Team;
"""

    drop_stmt = f"DROP TABLE IF EXISTS {out_tbl}"
    create_stmt = f"""
CREATE TABLE {out_tbl} (
  Season INT NOT NULL,
  SeasonState VARCHAR(8) NOT NULL,
  StrengthState VARCHAR(5) NOT NULL,
  Team TEXT,
  GP BIGINT,
  TOI DOUBLE,
  CF BIGINT,
  CA BIGINT,
  FF BIGINT,
  FA BIGINT,
  SF BIGINT,
  SA BIGINT,
  GF BIGINT,
  GA BIGINT,
  xGF_F DOUBLE,
  xGA_F DOUBLE,
  xGF_S DOUBLE,
  xGA_S DOUBLE,
  xGF_F2 DOUBLE,
  xGA_F2 DOUBLE,
  PIM_for DOUBLE,
  PIM_against DOUBLE
);
""".strip()

    # Fetch the aggregated result first (small result set), then write back.
    # This avoids long-running server-side INSERTs that can drop connections.
    df = pd.read_sql_query(sql, con=eng)

    with eng.begin() as conn:
        conn.execute(text(drop_stmt))
        conn.execute(text(create_stmt))

    # Use chunked inserts for robustness.
    df.to_sql(out_tbl, con=eng, if_exists="append", index=False, chunksize=1000, method="multi")
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Build seasonstats_team_{season} from nhl_{season}_pbp/shifts and write to Sheets")
    parser.add_argument("--season", required=True, help="Season key like 20252026")
    parser.add_argument("--mode", choices=["ro", "rw"], default="rw", help="DB URL mode (needs rw to create table)")
    parser.add_argument("--db-url", default="", help="Optional explicit SQLAlchemy DB URL")
    parser.add_argument("--sheet-id", default="", help="Google Sheets document id (default from GOOGLE_SHEETS_ID)")
    parser.add_argument("--worksheet", default="Sheets7", help="Worksheet/tab name for output")
    parser.add_argument("--no-sheets", action="store_true", help="Only build the MySQL table; skip Sheets write")
    args = parser.parse_args()

    season = str(args.season).strip()
    if not _SEASON_RE.match(season):
        raise SystemExit("--season must be 8 digits like 20252026")

    db_url = str(args.db_url).strip() or build_db_url(str(args.mode))
    print(f"Using DB: {redact_db_url(db_url)}")
    # Batch job: avoid connection pooling (mysql-connector can hit "commands out of sync"
    # if a pooled connection is returned with unread results).
    eng = create_engine(
        db_url,
        poolclass=NullPool,
        pool_pre_ping=True,
        connect_args={"connection_timeout": 600},
    )

    df = rebuild_team_seasonstats(eng=eng, season=season)
    print(f"[mysql] wrote table seasonstats_team_{season} rows={len(df)}")

    if not args.no_sheets:
        sheet_id = str(args.sheet_id).strip() or str(os.getenv("GOOGLE_SHEETS_ID") or "").strip()
        if not sheet_id:
            raise SystemExit("Missing sheet id. Provide --sheet-id or set GOOGLE_SHEETS_ID")
        _write_dataframe_to_google_sheet(df, sheet_id=sheet_id, worksheet=str(args.worksheet).strip())
        print(f"[sheets] wrote seasonstats_team_{season} to sheetId={sheet_id} worksheet={args.worksheet}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
