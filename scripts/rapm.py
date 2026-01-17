"""RAPM (Regularized Adjusted Plus-Minus) from MySQL PBP + shifts tables.

Builds a shift-segment dataset from:
- nhl_{season}_pbp
- nhl_{season}_shifts

Then fits ridge-regression RAPM models (per strength state) similar to the provided reference code.

Outputs:
- Writes intermediate shift dataset to MySQL table `rapm_data_{season}` (replaced).
- Writes RAPM results to MySQL table `nhl_{season}_rapm` (replaced).
- Writes the combined RAPM output (rapm2_all) to a Google Sheets worksheet (default tab: Sheets4).
- Writes the combined context output (context_all_blend_xG67_G33) to a Google Sheets worksheet (default tab: Sheets5).

Usage:
    python .\\scripts\\rapm.py --season 20252026

Notes:
- Requires RW DB credentials to write tables.
- This script treats xG NULLs in PBP (reason short/failed-bank-attempt) as 0 contribution when aggregating.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from importlib import util as _importlib_util
from pathlib import Path
import os
from typing import Any, Iterable, Optional, cast

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.preprocessing import MultiLabelBinarizer
from sqlalchemy import text
from sqlalchemy.engine import Engine


def _load_update_data_module() -> Any:
    update_data_path = Path(__file__).resolve().parent / "update_data.py"
    spec = _importlib_util.spec_from_file_location("nhl_update_data", str(update_data_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module spec from {update_data_path}")
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _create_engine_rw() -> Engine:
    m = _load_update_data_module()
    create_eng = getattr(m, "_create_mysql_engine", None)
    if not callable(create_eng):
        raise RuntimeError("_create_mysql_engine not found in update_data.py")
    eng = cast(Optional[Engine], create_eng("rw"))
    if eng is None:
        raise RuntimeError("MySQL engine not available (check env vars)")
    return eng


def _write_df_to_sheets(df: pd.DataFrame, *, sheet_id: str, worksheet: str) -> None:
    # Reuse Sheets writer from update_data.py so auth/env handling stays consistent.
    m = _load_update_data_module()
    fn = getattr(m, "_write_dataframe_to_google_sheet", None)
    if not callable(fn):
        raise RuntimeError("_write_dataframe_to_google_sheet not found in update_data.py")
    fn(df, sheet_id=sheet_id, worksheet=worksheet)


@dataclass(frozen=True)
class BuildConfig:
    season: str
    data_table: str


_PP_STATES = {"5v4", "5v3", "4v3", "ENF"}
_SH_STATES = {"4v5", "3v5", "3v4", "ENA"}


def _map_strength_side(v: Any) -> str:
    s = str(v or "").strip()
    if s in _PP_STATES:
        return "PP"
    if s in _SH_STATES:
        return "SH"
    return s


def _build_game_map_sql(tbl_pbp: str) -> str:
    # Determine home/away teams per game using Venue labels.
    # Some rows have EventTeam NULL; filter those out.
    return f"""
        SELECT
            GameID,
            MAX(CASE WHEN Venue = 'Home' THEN EventTeam END) AS HomeTeam,
            MAX(CASE WHEN Venue = 'Away' THEN EventTeam END) AS AwayTeam
        FROM {tbl_pbp}
        WHERE EventTeam IS NOT NULL
        GROUP BY GameID
    """.strip()


def build_rapm_data(eng: Engine, cfg: BuildConfig) -> pd.DataFrame:
    """Build shift-segment dataset suitable for RAPM fits."""
    season = str(cfg.season)
    tbl_pbp = f"nhl_{season}_pbp"
    tbl_sh = f"nhl_{season}_shifts"

    # 1) Game-level home/away mapping
    game_map = pd.read_sql(text(_build_game_map_sql(tbl_pbp)), con=eng)
    if game_map.empty:
        raise RuntimeError("No games found in PBP table")

    # 2) Shift segments with on-ice rosters (from shifts table) + home/away mapping.
    # Use GROUP_CONCAT to assemble skater lists by side.
    # Note: PlayerID is stored as float in some rows; cast to UNSIGNED to normalize.
    q_shifts = f"""
        SELECT
            s.ShiftIndex,
            s.GameID,
            MIN(s.Date) AS Date,
            s.Period,
            MAX(CASE WHEN s.Team = gm.HomeTeam THEN s.StrengthState END) AS Home_StrengthState_raw,
            MAX(CASE WHEN s.Team = gm.AwayTeam THEN s.StrengthState END) AS Away_StrengthState_raw,
            MAX(s.Duration) AS Duration,
            gm.HomeTeam,
            gm.AwayTeam,
            GROUP_CONCAT(
                CASE WHEN s.Team = gm.HomeTeam AND s.Position <> 'G' THEN CAST(s.PlayerID AS UNSIGNED) END
                ORDER BY s.PlayerID SEPARATOR ' '
            ) AS Home_Skaters,
            GROUP_CONCAT(
                CASE WHEN s.Team = gm.AwayTeam AND s.Position <> 'G' THEN CAST(s.PlayerID AS UNSIGNED) END
                ORDER BY s.PlayerID SEPARATOR ' '
            ) AS Away_Skaters,
            MAX(CASE WHEN s.Team = gm.HomeTeam AND s.Position = 'G' THEN CAST(s.PlayerID AS UNSIGNED) END) AS Home_Goalie,
            MAX(CASE WHEN s.Team = gm.AwayTeam AND s.Position = 'G' THEN CAST(s.PlayerID AS UNSIGNED) END) AS Away_Goalie
        FROM {tbl_sh} s
        JOIN ({_build_game_map_sql(tbl_pbp)}) gm
          ON gm.GameID = s.GameID
        GROUP BY s.ShiftIndex, s.GameID, s.Period, gm.HomeTeam, gm.AwayTeam
    """.strip()
    seg = pd.read_sql(text(q_shifts), con=eng)
    if seg.empty:
        raise RuntimeError("No shift segments found")

    # Apply the requested PP/SH mapping logic to side strength.
    seg["Home_StrengthState"] = seg["Home_StrengthState_raw"].apply(_map_strength_side)
    seg["Away_StrengthState"] = seg["Away_StrengthState_raw"].apply(_map_strength_side)

    # Segment-level StrengthState: only keep 5v5 / PP / SH (otherwise Mixed).
    def _segment_strength(row: pd.Series) -> str:
        hs = str(row.get("Home_StrengthState") or "").strip()
        a = str(row.get("Away_StrengthState") or "").strip()
        if hs == "5v5" and a == "5v5":
            return "5v5"
        if (hs == "PP" and a == "SH") or (hs == "SH" and a == "PP"):
            # We'll model PP and SH using Off_StrengthState; mark segment as Mixed to avoid direct grouping.
            return "Mixed"
        return "Mixed"

    seg["StrengthState"] = seg.apply(_segment_strength, axis=1)

    # 3) Aggregate event metrics (Corsi/Goal/xG/PEN) from PBP by shift.
    q_metrics = f"""
        SELECT
            p.ShiftIndex,
            p.GameID,
            SUM(CASE WHEN p.EventTeam = gm.HomeTeam THEN COALESCE(p.Corsi, 0) ELSE 0 END) AS Home_Corsi,
            SUM(CASE WHEN p.EventTeam = gm.AwayTeam THEN COALESCE(p.Corsi, 0) ELSE 0 END) AS Away_Corsi,
            SUM(CASE WHEN p.EventTeam = gm.HomeTeam THEN COALESCE(p.Goal, 0) ELSE 0 END) AS Home_Goal,
            SUM(CASE WHEN p.EventTeam = gm.AwayTeam THEN COALESCE(p.Goal, 0) ELSE 0 END) AS Away_Goal,
            SUM(CASE WHEN p.EventTeam = gm.HomeTeam THEN COALESCE(p.xG_F2, 0) ELSE 0 END) AS Home_xG,
            SUM(CASE WHEN p.EventTeam = gm.AwayTeam THEN COALESCE(p.xG_F2, 0) ELSE 0 END) AS Away_xG,
            SUM(CASE WHEN p.EventTeam = gm.HomeTeam THEN COALESCE(p.PEN_duration, 0) ELSE 0 END) AS Home_PEN,
            SUM(CASE WHEN p.EventTeam = gm.AwayTeam THEN COALESCE(p.PEN_duration, 0) ELSE 0 END) AS Away_PEN,
            MAX(CASE WHEN p.Event = 'faceoff' AND p.EventTeam = gm.HomeTeam THEN p.Zone END) AS Home_ZoneStart,
            MAX(CASE WHEN p.Event = 'faceoff' AND p.EventTeam = gm.AwayTeam THEN p.Zone END) AS Away_ZoneStart,
            MAX(CASE WHEN p.EventTeam = gm.HomeTeam THEN p.ScoreState END) AS Home_ScoreState,
            MAX(CASE WHEN p.EventTeam = gm.AwayTeam THEN p.ScoreState END) AS Away_ScoreState
        FROM {tbl_pbp} p
        JOIN ({_build_game_map_sql(tbl_pbp)}) gm
          ON gm.GameID = p.GameID
        GROUP BY p.ShiftIndex, p.GameID
    """.strip()
    met = pd.read_sql(text(q_metrics), con=eng)

    out = seg.merge(met, on=["ShiftIndex", "GameID"], how="left")

    # Fill missing metric aggregates with zeros/defaults.
    for c in [
        "Home_Corsi",
        "Away_Corsi",
        "Home_Goal",
        "Away_Goal",
        "Home_xG",
        "Away_xG",
        "Home_PEN",
        "Away_PEN",
    ]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)

    for c in ["Home_ZoneStart", "Away_ZoneStart"]:
        if c in out.columns:
            out[c] = out[c].fillna("N").astype(str)

    for c in ["Home_ScoreState", "Away_ScoreState"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)

    # Derived columns required by RAPM reference code.
    out["Season"] = int(season)

    # Ensure required columns exist
    for c in [
        "HomeTeam",
        "AwayTeam",
        "Home_Skaters",
        "Away_Skaters",
        "Home_Goalie",
        "Away_Goalie",
        "Home_ZoneStart",
        "Away_ZoneStart",
        "Home_ScoreState",
        "Away_ScoreState",
    ]:
        if c not in out.columns:
            out[c] = ""

    out = out[[
        "ShiftIndex",
        "GameID",
        "Season",
        "StrengthState",
        "Home_StrengthState",
        "Away_StrengthState",
        "Period",
        "Duration",
        "HomeTeam",
        "AwayTeam",
        "Home_Skaters",
        "Away_Skaters",
        "Home_Goalie",
        "Away_Goalie",
        "Home_ZoneStart",
        "Away_ZoneStart",
        "Home_ScoreState",
        "Away_ScoreState",
        "Home_Corsi",
        "Away_Corsi",
        "Home_Goal",
        "Away_Goal",
        "Home_xG",
        "Away_xG",
        "Home_PEN",
        "Away_PEN",
        "Date",
    ]]

    return out


def _split_ids(col: pd.Series) -> pd.Series:
    def _norm(v: Any) -> tuple[str, ...]:
        if v is None:
            return ()
        if isinstance(v, (tuple, list, set)):
            out: list[str] = []
            for x in v:
                s = str(x).strip().strip("\"").strip("'")
                if s:
                    out.append(s)
            return tuple(out)
        s = str(v).strip()
        if not s:
            return ()
        # If something already looks like a tuple string, remove common punctuation.
        s = s.replace("(", " ").replace(")", " ").replace(",", " ")
        parts = [p.strip().strip("\"").strip("'") for p in s.split()]
        return tuple(p for p in parts if p)

    return col.apply(_norm)


def _filter_positive_player_ids(df: pd.DataFrame, col: str = "PlayerID") -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    ids = pd.to_numeric(df[col].astype(str).str.strip().str.strip("\"").str.strip("'"), errors="coerce")
    out = df[ids.notna() & (ids.astype(float) > 0)].copy()
    out[col] = ids.loc[out.index].astype(int).astype(str)
    return out


def _bucket_score(s: pd.Series, cap: int = 3) -> pd.Series:
    v = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
    v = v.clip(lower=-cap, upper=cap)
    return v.astype(str)


def _build_off_def_long(season_df: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["ShiftIndex", "GameID", "Season", "StrengthState", "Period", "Duration"]

    use_side_strength = all(c in season_df.columns for c in ["Home_StrengthState", "Away_StrengthState"])

    home = pd.DataFrame({
        **season_df[base_cols].to_dict(orient="list"),
        "Off_Team": season_df["HomeTeam"].astype(str).tolist(),
        "Def_Team": season_df["AwayTeam"].astype(str).tolist(),
        "Off_Skaters": season_df["Home_Skaters"].tolist(),
        "Def_Skaters": season_df["Away_Skaters"].tolist(),
        "Off_Goalie": season_df["Home_Goalie"].astype(str).tolist(),
        "Def_Goalie": season_df["Away_Goalie"].astype(str).tolist(),
        "Off_ZoneStart": season_df["Home_ZoneStart"].astype(str).tolist(),
        "Def_ZoneStart": season_df["Away_ZoneStart"].astype(str).tolist(),
        "Off_ScoreState": _bucket_score(season_df["Home_ScoreState"]).tolist(),
        "Off_StrengthState": (season_df["Home_StrengthState"] if use_side_strength else season_df["StrengthState"]).astype(str).tolist(),
        "Def_StrengthState": (season_df["Away_StrengthState"] if use_side_strength else season_df["StrengthState"]).astype(str).tolist(),
        "is_home": [1] * len(season_df),
        "CF": pd.to_numeric(season_df["Home_Corsi"], errors="coerce").fillna(0).astype(float).tolist(),
        "GF": pd.to_numeric(season_df["Home_Goal"], errors="coerce").fillna(0).astype(float).tolist(),
        "xGF": pd.to_numeric(season_df["Home_xG"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PEN": pd.to_numeric(season_df["Home_PEN"], errors="coerce").fillna(0.0).astype(float).tolist(),
    })

    away = pd.DataFrame({
        **season_df[base_cols].to_dict(orient="list"),
        "Off_Team": season_df["AwayTeam"].astype(str).tolist(),
        "Def_Team": season_df["HomeTeam"].astype(str).tolist(),
        "Off_Skaters": season_df["Away_Skaters"].tolist(),
        "Def_Skaters": season_df["Home_Skaters"].tolist(),
        "Off_Goalie": season_df["Away_Goalie"].astype(str).tolist(),
        "Def_Goalie": season_df["Home_Goalie"].astype(str).tolist(),
        "Off_ZoneStart": season_df["Away_ZoneStart"].astype(str).tolist(),
        "Def_ZoneStart": season_df["Home_ZoneStart"].astype(str).tolist(),
        "Off_ScoreState": _bucket_score(season_df["Away_ScoreState"]).tolist(),
        "Off_StrengthState": (season_df["Away_StrengthState"] if use_side_strength else season_df["StrengthState"]).astype(str).tolist(),
        "Def_StrengthState": (season_df["Home_StrengthState"] if use_side_strength else season_df["StrengthState"]).astype(str).tolist(),
        "is_home": [0] * len(season_df),
        "CF": pd.to_numeric(season_df["Away_Corsi"], errors="coerce").fillna(0).astype(float).tolist(),
        "GF": pd.to_numeric(season_df["Away_Goal"], errors="coerce").fillna(0).astype(float).tolist(),
        "xGF": pd.to_numeric(season_df["Away_xG"], errors="coerce").fillna(0.0).astype(float).tolist(),
        "PEN": pd.to_numeric(season_df["Away_PEN"], errors="coerce").fillna(0.0).astype(float).tolist(),
    })

    return pd.concat([home, away], ignore_index=True)


def _eligible_players_long(long_df: pd.DataFrame, threshold: int = 300) -> set[str]:
    # Eligibility: total CF involvement while a player is on the ice (offense or defense) in the modeled context.
    from collections import defaultdict

    counts: dict[str, float] = defaultdict(float)
    for row in long_df.itertuples(index=False):
        cf = float(getattr(row, "CF") or 0.0)
        if cf <= 0:
            continue
        for p in getattr(row, "Off_Skaters"):
            counts[str(p)] += cf
        for p in getattr(row, "Def_Skaters"):
            counts[str(p)] += cf
    return {p for p, v in counts.items() if v >= threshold}


def _minutes_by_strength(segments: pd.DataFrame) -> dict[str, dict[str, float]]:
    # Exposure seconds per player per side-mapped strength.
    from collections import defaultdict

    mins: dict[str, dict[str, float]] = {
        "5v5": defaultdict(float),
        "PP": defaultdict(float),
        "SH": defaultdict(float),
    }

    for r in segments.itertuples(index=False):
        dur = float(getattr(r, "Duration") or 0.0)
        if dur <= 0:
            continue
        hs = str(getattr(r, "Home_StrengthState") or "").strip()
        a = str(getattr(r, "Away_StrengthState") or "").strip()
        home_players = getattr(r, "Home_Skaters")
        away_players = getattr(r, "Away_Skaters")
        if hs in mins:
            for p in home_players:
                mins[hs][str(p)] += dur
        if a in mins:
            for p in away_players:
                mins[a][str(p)] += dur
    return mins


def _special_team_counts(long_ppsh: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    """Return (pp_cf_by_player, sh_ca_by_player) for Off=PP/Def=SH rows.

    CA for SH players is opponent CF while they are defending (i.e., CF on these rows).
    """
    from collections import defaultdict

    pp_cf: dict[str, float] = defaultdict(float)
    sh_ca: dict[str, float] = defaultdict(float)
    for r in long_ppsh.itertuples(index=False):
        cf = float(getattr(r, "CF") or 0.0)
        if cf == 0:
            continue
        for p in getattr(r, "Off_Skaters"):
            pp_cf[str(p)] += cf
        for p in getattr(r, "Def_Skaters"):
            sh_ca[str(p)] += cf
    return pp_cf, sh_ca


def _fit_metric(long_df: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series, float]:
    mlb_off = MultiLabelBinarizer(sparse_output=True)
    mlb_def = MultiLabelBinarizer(sparse_output=True)
    off_mat = mlb_off.fit_transform(long_df["Off_Skaters"])
    def_mat = mlb_def.fit_transform(long_df["Def_Skaters"])
    off_names = [f"off_{p}" for p in mlb_off.classes_]
    def_names = [f"def_{p}" for p in mlb_def.classes_]

    off_team = pd.get_dummies(long_df["Off_Team"].astype(str), prefix="off_team")
    def_team = pd.get_dummies(long_df["Def_Team"].astype(str), prefix="def_team")

    z_cols: list[pd.DataFrame] = []
    if "Off_ZoneStart" in long_df.columns:
        z_cols.append(pd.get_dummies(long_df["Off_ZoneStart"].astype(str), prefix="off_zs"))
    if "Def_ZoneStart" in long_df.columns:
        z_cols.append(pd.get_dummies(long_df["Def_ZoneStart"].astype(str), prefix="def_zs"))

    ss = pd.get_dummies(long_df["Off_ScoreState"].astype(str), prefix="ss")

    intercept = np.ones((long_df.shape[0], 1), dtype=float)
    is_home = long_df["is_home"].astype(int).to_numpy().reshape(-1, 1)

    blocks = [
        csr_matrix(off_mat),
        csr_matrix(def_mat),
        csr_matrix(off_team.values),
        csr_matrix(def_team.values),
        *(csr_matrix(b.values) for b in z_cols),
        csr_matrix(ss.values),
        csr_matrix(is_home),
        csr_matrix(intercept),
    ]
    X = cast(Any, hstack(blocks, format="csr"))

    feature_names = (
        off_names
        + def_names
        + off_team.columns.tolist()
        + def_team.columns.tolist()
        + sum((b.columns.tolist() for b in z_cols), [])
        + ss.columns.tolist()
        + ["is_home", "const"]
    )

    dur = np.asarray(pd.to_numeric(long_df["Duration"], errors="coerce").fillna(0.0).to_numpy(), dtype=float)
    denom = np.where(dur > 0, dur / 3600.0, np.nan)
    raw = np.asarray(pd.to_numeric(long_df[metric], errors="coerce").fillna(0.0).to_numpy(), dtype=float)
    y = np.where(np.isfinite(denom), raw / denom, 0.0)
    w = np.where(dur > 0, dur, 0.0)

    alphas = np.logspace(1, 5, num=10)
    cv = RidgeCV(alphas=alphas, cv=5, scoring="neg_mean_squared_error", fit_intercept=False)
    cv.fit(X, y, sample_weight=w)

    model = Ridge(alpha=float(cv.alpha_), fit_intercept=False)
    model.fit(X, y, sample_weight=w)

    coefs = pd.Series(model.coef_, index=feature_names)
    # Only return player coefficients (exclude team/zone/score/is_home/const).
    off = coefs.reindex(off_names).rename(lambda s: s[4:])
    deff = coefs.reindex(def_names).rename(lambda s: s[4:])
    return off, deff, float(cv.alpha_)


def build_rapm(df: pd.DataFrame, *, strength_filter: Optional[list[str]] = None) -> pd.DataFrame:
    # Normalize types
    for col in ["HomeTeam", "AwayTeam", "Home_Goalie", "Away_Goalie", "Home_ZoneStart", "Away_ZoneStart"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    for col in [
        "Duration",
        "Home_Corsi",
        "Away_Corsi",
        "Home_Goal",
        "Away_Goal",
        "Home_xG",
        "Away_xG",
        "Home_PEN",
        "Away_PEN",
    ]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Home_Skaters"] = _split_ids(df["Home_Skaters"])
    df["Away_Skaters"] = _split_ids(df["Away_Skaters"])

    out_frames: list[pd.DataFrame] = []

    season_vals = sorted(pd.unique(df["Season"]))
    if not season_vals:
        return pd.DataFrame()
    if len(season_vals) != 1:
        # Caller should pre-filter to one season.
        df = df[df["Season"] == season_vals[-1]].copy()
    season = int(pd.unique(df["Season"])[0])

    # Compute exposure seconds per player for totals scaling.
    minutes_by_strength = _minutes_by_strength(df)

    # Build full long once, then fit only on contexts we care about.
    long_all = _build_off_def_long(df)
    strengths = ["5v5", "PP", "SH"]
    if strength_filter:
        strengths = [s for s in strengths if s in set(strength_filter)]

    metric_cols = [
        "CF",
        "CA",
        "GF",
        "GA",
        "xGF",
        "xGA",
        "C_plusminus",
        "G_plusminus",
        "xG_plusminus",
        "PEN_drawn",
        "PEN_taken",
        "PEN_plusminus",
    ]

    def _zscore_frames(df_out: pd.DataFrame, *, strength: str) -> pd.DataFrame:
        df_rates = df_out.copy()
        df_rates["Rates_Totals"] = "Rates"
        for col in metric_cols:
            mean = df_rates[col].mean()
            std = df_rates[col].std(ddof=0)
            df_rates[f"{col}_stddev"] = std
            df_rates[f"{col}_zscore"] = (df_rates[col] - mean) / std if std > 0 else 0.0

        df_totals = df_out.copy()
        mins_map = minutes_by_strength.get(strength, {})
        scale = df_totals["PlayerID"].map(lambda p: (mins_map.get(str(p), 0.0) / 3600.0)).fillna(0.0)
        for col in metric_cols:
            df_totals[col] = df_totals[col] * scale
        df_totals["Rates_Totals"] = "Totals"
        for col in metric_cols:
            mean = df_totals[col].mean()
            std = df_totals[col].std(ddof=0)
            df_totals[f"{col}_stddev"] = std
            df_totals[f"{col}_zscore"] = (df_totals[col] - mean) / std if std > 0 else 0.0
        return pd.concat([df_rates, df_totals], ignore_index=True)

    # 5v5 fit (keep existing CF-based eligibility at 300 involvement)
    if "5v5" in strengths:
        long_5v5 = long_all[(long_all["Off_StrengthState"] == "5v5") & (long_all["Def_StrengthState"] == "5v5")].copy()
        if not long_5v5.empty:
            elig_5v5 = _eligible_players_long(long_5v5, threshold=300)
            if elig_5v5:
                long_5v5["Off_Skaters"] = long_5v5["Off_Skaters"].apply(lambda xs: tuple(p for p in xs if str(p) in elig_5v5))
                long_5v5["Def_Skaters"] = long_5v5["Def_Skaters"].apply(lambda xs: tuple(p for p in xs if str(p) in elig_5v5))
                long_5v5 = long_5v5[(long_5v5["Off_Skaters"].str.len() > 0) & (long_5v5["Def_Skaters"].str.len() > 0)]
                if not long_5v5.empty:
                    results: dict[str, pd.Series] = {}
                    alphas: dict[str, float] = {}
                    for metric in ["CF", "GF", "xGF", "PEN"]:
                        off, deff, alpha = _fit_metric(long_5v5, metric)
                        results[f"{metric}_for"] = off
                        results[f"{metric}_against"] = deff
                        alphas[f"Alpha_{metric}"] = alpha

                    players = sorted(set().union(*[s.index for s in results.values()]))
                    df_out = pd.DataFrame({"PlayerID": players})
                    df_out["CF"] = df_out["PlayerID"].map(results["CF_for"]).fillna(0.0)
                    df_out["CA"] = df_out["PlayerID"].map(results["CF_against"]).fillna(0.0)
                    df_out["GF"] = df_out["PlayerID"].map(results["GF_for"]).fillna(0.0)
                    df_out["GA"] = df_out["PlayerID"].map(results["GF_against"]).fillna(0.0)
                    df_out["xGF"] = df_out["PlayerID"].map(results["xGF_for"]).fillna(0.0)
                    df_out["xGA"] = df_out["PlayerID"].map(results["xGF_against"]).fillna(0.0)
                    df_out["PEN_taken"] = df_out["PlayerID"].map(results["PEN_for"]).fillna(0.0)
                    df_out["PEN_drawn"] = df_out["PlayerID"].map(results["PEN_against"]).fillna(0.0)
                    df_out["C_plusminus"] = df_out["CF"] - df_out["CA"]
                    df_out["G_plusminus"] = df_out["GF"] - df_out["GA"]
                    df_out["xG_plusminus"] = df_out["xGF"] - df_out["xGA"]
                    df_out["PEN_plusminus"] = df_out["PEN_drawn"] - df_out["PEN_taken"]
                    df_out["Season"] = int(season)
                    df_out["StrengthState"] = "5v5"
                    for k, a in alphas.items():
                        df_out[k] = a

                    out_frames.append(_zscore_frames(df_out, strength="5v5"))

    # Special teams: fit only on Off=PP / Def=SH, then output both PP offense and SH defense.
    long_ppsh = long_all[(long_all["Off_StrengthState"] == "PP") & (long_all["Def_StrengthState"] == "SH")].copy()
    if not long_ppsh.empty and ("PP" in strengths or "SH" in strengths):
        results: dict[str, pd.Series] = {}
        alphas: dict[str, float] = {}
        for metric in ["CF", "GF", "xGF", "PEN"]:
            off, deff, alpha = _fit_metric(long_ppsh, metric)
            results[f"{metric}_for"] = off
            results[f"{metric}_against"] = deff
            alphas[f"Alpha_{metric}"] = alpha

        pp_cf, sh_ca = _special_team_counts(long_ppsh)
        pp_eligible = {p for p, v in pp_cf.items() if v >= 100.0}
        sh_eligible = {p for p, v in sh_ca.items() if v >= 100.0}

        if "PP" in strengths and pp_eligible:
            players_pp = sorted(pp_eligible)
            df_pp = pd.DataFrame({"PlayerID": players_pp})
            df_pp["CF"] = df_pp["PlayerID"].map(results["CF_for"]).fillna(0.0)
            df_pp["GF"] = df_pp["PlayerID"].map(results["GF_for"]).fillna(0.0)
            df_pp["xGF"] = df_pp["PlayerID"].map(results["xGF_for"]).fillna(0.0)
            df_pp["PEN_taken"] = df_pp["PlayerID"].map(results["PEN_for"]).fillna(0.0)

            df_pp["CA"] = 0.0
            df_pp["GA"] = 0.0
            df_pp["xGA"] = 0.0
            df_pp["PEN_drawn"] = 0.0
            df_pp["C_plusminus"] = df_pp["CF"] - df_pp["CA"]
            df_pp["G_plusminus"] = df_pp["GF"] - df_pp["GA"]
            df_pp["xG_plusminus"] = df_pp["xGF"] - df_pp["xGA"]
            df_pp["PEN_plusminus"] = df_pp["PEN_drawn"] - df_pp["PEN_taken"]
            df_pp["Season"] = int(season)
            df_pp["StrengthState"] = "PP"
            for k, a in alphas.items():
                df_pp[k] = a
            out_frames.append(_zscore_frames(df_pp, strength="PP"))

        if "SH" in strengths and sh_eligible:
            players_sh = sorted(sh_eligible)
            df_sh = pd.DataFrame({"PlayerID": players_sh})
            df_sh["CA"] = df_sh["PlayerID"].map(results["CF_against"]).fillna(0.0)
            df_sh["GA"] = df_sh["PlayerID"].map(results["GF_against"]).fillna(0.0)
            df_sh["xGA"] = df_sh["PlayerID"].map(results["xGF_against"]).fillna(0.0)
            df_sh["PEN_drawn"] = df_sh["PlayerID"].map(results["PEN_against"]).fillna(0.0)

            df_sh["CF"] = 0.0
            df_sh["GF"] = 0.0
            df_sh["xGF"] = 0.0
            df_sh["PEN_taken"] = 0.0
            df_sh["C_plusminus"] = df_sh["CF"] - df_sh["CA"]
            df_sh["G_plusminus"] = df_sh["GF"] - df_sh["GA"]
            df_sh["xG_plusminus"] = df_sh["xGF"] - df_sh["xGA"]
            df_sh["PEN_plusminus"] = df_sh["PEN_drawn"] - df_sh["PEN_taken"]
            df_sh["Season"] = int(season)
            df_sh["StrengthState"] = "SH"
            for k, a in alphas.items():
                df_sh[k] = a
            out_frames.append(_zscore_frames(df_sh, strength="SH"))

    if not out_frames:
        return pd.DataFrame()

    out = pd.concat(out_frames, ignore_index=True)
    return _filter_positive_player_ids(out)


def _zs_to_score(val: Any) -> float:
    v = str(val or "").strip().upper()
    if v in ("OZ", "O"):
        return 1.0
    if v in ("DZ", "D"):
        return -1.0
    return 0.0


def _blended_strength_map(
    rapm_df: pd.DataFrame,
    *,
    season: int,
    strength: str,
    xg_weight: float = 2.0 / 3.0,
    g_weight: float = 1.0 / 3.0,
) -> dict[str, float]:
    sel = rapm_df[(rapm_df["Season"] == int(season)) & (rapm_df["StrengthState"] == str(strength))].copy()
    if "Rates_Totals" in sel.columns:
        sel = sel[sel["Rates_Totals"] == "Rates"]
    if sel.empty:
        return {}
    if not {"xG_plusminus", "G_plusminus"}.issubset(sel.columns):
        return {}
    blend = xg_weight * pd.to_numeric(sel["xG_plusminus"], errors="coerce").fillna(0.0) + g_weight * pd.to_numeric(
        sel["G_plusminus"], errors="coerce"
    ).fillna(0.0)
    return dict(zip(sel["PlayerID"].astype(str), blend.astype(float)))


def compute_context(
    segments_df: pd.DataFrame,
    rapm_df: pd.DataFrame,
    *,
    season: int,
    strengths: Iterable[str] = ("5v5", "PP", "SH"),
    xg_weight: float = 2.0 / 3.0,
    g_weight: float = 1.0 / 3.0,
    blend_label: str = "blend_xG67_G33",
) -> pd.DataFrame:
    """Compute per-player context features from shift segments + RAPM outputs.

    Returns combined context dataframe (one row per player/season/strength).
    """

    if segments_df.empty or rapm_df.empty:
        return pd.DataFrame()

    df = segments_df.copy()
    df = df[df["Season"] == int(season)].copy()
    if df.empty:
        return pd.DataFrame()

    df["Home_Skaters"] = _split_ids(df["Home_Skaters"])
    df["Away_Skaters"] = _split_ids(df["Away_Skaters"])
    duration_series = df["Duration"] if "Duration" in df.columns else pd.Series(0.0, index=df.index)
    df["Duration"] = pd.to_numeric(duration_series, errors="coerce").fillna(0.0)
    df["Home_ZoneStart"] = df.get("Home_ZoneStart", "N")
    df["Away_ZoneStart"] = df.get("Away_ZoneStart", "N")

    strengths_set = set(strengths)

    out_frames: list[pd.DataFrame] = []

    def _build_output(
        tot_secs: dict[str, float],
        qot_num: dict[str, float],
        qoc_num: dict[str, float],
        zs_num: dict[str, float],
        *,
        st: str,
    ) -> Optional[pd.DataFrame]:
        players = sorted(tot_secs.keys())
        if not players:
            return None
        denom = np.array([tot_secs[p] for p in players], dtype=float)
        denom = np.where(denom > 0, denom, 1.0)
        return pd.DataFrame(
            {
                "PlayerID": players,
                "Season": int(season),
                "StrengthState": str(st),
                "Minutes": [tot_secs[p] / 60.0 for p in players],
                f"QoT_{blend_label}": [qot_num.get(p, 0.0) / tot_secs[p] if tot_secs[p] > 0 else 0.0 for p in players],
                f"QoC_{blend_label}": [qoc_num.get(p, 0.0) / tot_secs[p] if tot_secs[p] > 0 else 0.0 for p in players],
                "ZS_Difficulty": [zs_num.get(p, 0.0) / tot_secs[p] if tot_secs[p] > 0 else 0.0 for p in players],
            }
        )

    # --- Even strength context (5v5)
    if "5v5" in strengths_set:
        strength_map_5v5 = _blended_strength_map(rapm_df, season=season, strength="5v5", xg_weight=xg_weight, g_weight=g_weight)
        if strength_map_5v5:
            es = df[(df["Home_StrengthState"].astype(str) == "5v5") & (df["Away_StrengthState"].astype(str) == "5v5")].copy()
            tot_secs: dict[str, float] = {}
            qot_num: dict[str, float] = {}
            qoc_num: dict[str, float] = {}
            zs_num: dict[str, float] = {}
            for row in es.itertuples(index=False):
                dur = float(getattr(row, "Duration") or 0.0)
                if dur <= 0:
                    continue
                home = getattr(row, "Home_Skaters") or ()
                away = getattr(row, "Away_Skaters") or ()
                zs_home = _zs_to_score(getattr(row, "Home_ZoneStart"))
                zs_away = _zs_to_score(getattr(row, "Away_ZoneStart"))

                if len(home) > 1:
                    avg_home_teammate = {
                        p: float(np.mean([strength_map_5v5.get(tp, 0.0) for tp in home if tp != p])) for p in home
                    }
                else:
                    avg_home_teammate = {p: 0.0 for p in home}
                if len(away) > 1:
                    avg_away_teammate = {
                        p: float(np.mean([strength_map_5v5.get(tp, 0.0) for tp in away if tp != p])) for p in away
                    }
                else:
                    avg_away_teammate = {p: 0.0 for p in away}

                avg_home_opp = float(np.mean([strength_map_5v5.get(p, 0.0) for p in away])) if away else 0.0
                avg_away_opp = float(np.mean([strength_map_5v5.get(p, 0.0) for p in home])) if home else 0.0

                for p in home:
                    ps = str(p).strip().strip("\"").strip("'")
                    tot_secs[ps] = tot_secs.get(ps, 0.0) + dur
                    qot_num[ps] = qot_num.get(ps, 0.0) + dur * avg_home_teammate.get(p, 0.0)
                    qoc_num[ps] = qoc_num.get(ps, 0.0) + dur * avg_home_opp
                    zs_num[ps] = zs_num.get(ps, 0.0) + dur * zs_home
                for p in away:
                    ps = str(p).strip().strip("\"").strip("'")
                    tot_secs[ps] = tot_secs.get(ps, 0.0) + dur
                    qot_num[ps] = qot_num.get(ps, 0.0) + dur * avg_away_teammate.get(p, 0.0)
                    qoc_num[ps] = qoc_num.get(ps, 0.0) + dur * avg_away_opp
                    zs_num[ps] = zs_num.get(ps, 0.0) + dur * zs_away

            out = _build_output(tot_secs, qot_num, qoc_num, zs_num, st="5v5")
            if out is not None:
                out_frames.append(out)

    # --- Special teams context (PP and SH) using 5v5 blend map baseline (stable)
    if ("PP" in strengths_set or "SH" in strengths_set) and {"Home_StrengthState", "Away_StrengthState"}.issubset(df.columns):
        base_map = _blended_strength_map(rapm_df, season=season, strength="5v5", xg_weight=xg_weight, g_weight=g_weight)
        st_df = df.copy()
        # Only PP vs SH stints
        mask = (
            (st_df["Home_StrengthState"].astype(str).isin(["PP", "SH"]))
            & (st_df["Away_StrengthState"].astype(str).isin(["PP", "SH"]))
            & (st_df["Home_StrengthState"].astype(str) != st_df["Away_StrengthState"].astype(str))
        )
        st_df = st_df[mask]
        if not st_df.empty:
            from collections import defaultdict

            agg = {
                "PP": {"tot": defaultdict(float), "qot": defaultdict(float), "qoc": defaultdict(float), "zs": defaultdict(float)},
                "SH": {"tot": defaultdict(float), "qot": defaultdict(float), "qoc": defaultdict(float), "zs": defaultdict(float)},
            }
            for row in st_df.itertuples(index=False):
                dur = float(getattr(row, "Duration") or 0.0)
                if dur <= 0:
                    continue
                home_skaters = getattr(row, "Home_Skaters") or ()
                away_skaters = getattr(row, "Away_Skaters") or ()
                h_state = str(getattr(row, "Home_StrengthState") or "")
                a_state = str(getattr(row, "Away_StrengthState") or "")
                if {h_state, a_state} != {"PP", "SH"}:
                    continue
                zs_home = _zs_to_score(getattr(row, "Home_ZoneStart"))
                zs_away = _zs_to_score(getattr(row, "Away_ZoneStart"))
                if h_state == "PP" and a_state == "SH":
                    pp_team = home_skaters
                    sh_team = away_skaters
                    zs_pp = zs_home
                    zs_sh = zs_away
                else:
                    pp_team = away_skaters
                    sh_team = home_skaters
                    zs_pp = zs_away
                    zs_sh = zs_home

                if len(pp_team) > 1:
                    pp_teammate_avg = {
                        p: float(np.mean([base_map.get(str(q).strip().strip("\"").strip("'"), 0.0) for q in pp_team if q != p]))
                        for p in pp_team
                    }
                else:
                    pp_teammate_avg = {p: 0.0 for p in pp_team}
                if len(sh_team) > 1:
                    sh_teammate_avg = {
                        p: float(np.mean([base_map.get(str(q).strip().strip("\"").strip("'"), 0.0) for q in sh_team if q != p]))
                        for p in sh_team
                    }
                else:
                    sh_teammate_avg = {p: 0.0 for p in sh_team}

                pp_opp_avg = (
                    float(np.mean([base_map.get(str(p).strip().strip("\"").strip("'"), 0.0) for p in sh_team])) if sh_team else 0.0
                )
                sh_opp_avg = (
                    float(np.mean([base_map.get(str(p).strip().strip("\"").strip("'"), 0.0) for p in pp_team])) if pp_team else 0.0
                )

                for p in pp_team:
                    ps = str(p).strip().strip("\"").strip("'")
                    agg["PP"]["tot"][ps] += dur
                    agg["PP"]["qot"][ps] += dur * pp_teammate_avg.get(p, 0.0)
                    agg["PP"]["qoc"][ps] += dur * pp_opp_avg
                    agg["PP"]["zs"][ps] += dur * zs_pp
                for p in sh_team:
                    ps = str(p).strip().strip("\"").strip("'")
                    agg["SH"]["tot"][ps] += dur
                    agg["SH"]["qot"][ps] += dur * sh_teammate_avg.get(p, 0.0)
                    agg["SH"]["qoc"][ps] += dur * sh_opp_avg
                    agg["SH"]["zs"][ps] += dur * zs_sh

            for label in ["PP", "SH"]:
                if label not in strengths_set:
                    continue
                out = _build_output(agg[label]["tot"], agg[label]["qot"], agg[label]["qoc"], agg[label]["zs"], st=label)
                if out is not None:
                    out_frames.append(out)

    if not out_frames:
        return pd.DataFrame()
    out = pd.concat(out_frames, ignore_index=True)
    return _filter_positive_player_ids(out)


def write_table(eng: Engine, df: pd.DataFrame, table_name: str) -> None:
    if df.empty:
        print(f"[mysql] {table_name}: empty; nothing to write")
        return
    df.to_sql(table_name, con=eng, if_exists="replace", index=False, method="multi", chunksize=2000)
    print(f"[mysql] wrote {len(df)} rows to {table_name} (replaced)")


def _export_static_csvs(
    eng: Engine,
    *,
    rapm_table: str,
    context_table: str,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    df_rapm = pd.read_sql(text(f"SELECT * FROM {rapm_table}"), con=eng)
    df_ctx = pd.read_sql(text(f"SELECT * FROM {context_table}"), con=eng)

    df_rapm = _filter_positive_player_ids(df_rapm)
    df_ctx = _filter_positive_player_ids(df_ctx)

    rapm_path = out_dir / "rapm.csv"
    ctx_path = out_dir / "context.csv"
    df_rapm.to_csv(rapm_path, index=False)
    df_ctx.to_csv(ctx_path, index=False)
    print(f"[file] wrote {len(df_rapm)} rows -> {rapm_path}")
    print(f"[file] wrote {len(df_ctx)} rows -> {ctx_path}")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Build and fit RAPM from MySQL")
    p.add_argument("--season", default="20252026")
    p.add_argument("--data-table", default=None, help="Intermediate rapm segment table name (default: rapm_data_{season})")
    p.add_argument("--no-build-data", action="store_true", help="Skip building/writing rapm_data")
    p.add_argument("--no-fit", action="store_true", help="Skip fitting RAPM")
    p.add_argument("--strength", action="append", help="Restrict to a StrengthState (repeatable), e.g. --strength 5v5")
    p.add_argument("--sheets-id", help="Google Sheets document id to write RAPM output into (default RAPM_SHEET_ID or PROJECTIONS_SHEET_ID)")
    p.add_argument("--worksheet", default=None, help="Worksheet/tab name for RAPM output (default RAPM_WORKSHEET or Sheets4)")
    p.add_argument("--no-context", action="store_true", help="Skip computing context output")
    p.add_argument("--context-worksheet", default=None, help="Worksheet/tab name for context output (default CONTEXT_WORKSHEET or Sheets5)")
    p.add_argument(
        "--export-static",
        action="store_true",
        help="Export MySQL tables into app/static/rapm as rapm.csv and context.csv",
    )
    p.add_argument(
        "--export-rapm-table",
        default=None,
        help="MySQL table name to export as rapm.csv (default: rapm2_all)",
    )
    p.add_argument(
        "--export-context-table",
        default=None,
        help="MySQL table name to export as context.csv (default: context_all)",
    )
    args = p.parse_args(argv)

    season = str(args.season).strip()
    data_table = (str(args.data_table).strip() if args.data_table else f"rapm_data_{season}")
    cfg = BuildConfig(season=season, data_table=data_table)

    sheet_id = (args.sheets_id or os.getenv("RAPM_SHEET_ID") or os.getenv("PROJECTIONS_SHEET_ID") or "").strip()
    worksheet = (args.worksheet or os.getenv("RAPM_WORKSHEET") or "Sheets4").strip()
    context_worksheet = (args.context_worksheet or os.getenv("CONTEXT_WORKSHEET") or "Sheets5").strip()

    try:
        eng = _create_engine_rw()
    except Exception as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    # Export-only mode: allow exporting existing tables without building/loading/fitting.
    if args.export_static and args.no_fit and args.no_build_data:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            static_dir = repo_root / "app" / "static" / "rapm"
            rapm_table = (args.export_rapm_table or "rapm2_all").strip()
            context_table = (args.export_context_table or "context_all").strip()
            _export_static_csvs(
                eng,
                rapm_table=rapm_table,
                context_table=context_table,
                out_dir=static_dir,
            )
            return 0
        except Exception as e:
            print(f"[error] exporting static CSVs failed: {e}", file=sys.stderr)
            return 11

    df_data: Optional[pd.DataFrame] = None

    if not args.no_build_data:
        print("[rapm] building rapm_data from pbp + shifts...")
        try:
            df_data = build_rapm_data(eng, cfg)
        except Exception as e:
            print(f"[error] build_rapm_data failed: {e}", file=sys.stderr)
            return 3
        try:
            write_table(eng, df_data, cfg.data_table)
        except Exception as e:
            print(f"[error] writing {cfg.data_table} failed: {e}", file=sys.stderr)
            return 4
    else:
        print(f"[rapm] loading existing {cfg.data_table}...")
        try:
            df_data = pd.read_sql(text(f"SELECT * FROM {cfg.data_table} WHERE Season = :s"), con=eng, params={"s": int(season)})
        except Exception as e:
            print(f"[error] reading {cfg.data_table} failed: {e}", file=sys.stderr)
            return 5

    if args.no_fit:
        return 0

    assert df_data is not None
    print("[rapm] fitting RAPM...")
    try:
        # Ensure we only fit the requested season.
        df_data = df_data[df_data["Season"] == int(season)].copy()
        df_res = build_rapm(df_data, strength_filter=(args.strength or None))
    except Exception as e:
        print(f"[error] build_rapm failed: {e}", file=sys.stderr)
        return 6

    if df_res.empty:
        print("[rapm] no results produced")
        return 0

    out_tbl = f"nhl_{season}_rapm"
    try:
        write_table(eng, df_res, out_tbl)
    except Exception as e:
        print(f"[error] writing {out_tbl} failed: {e}", file=sys.stderr)
        return 7

    # Write combined output to Google Sheets (Sheets4 by default).
    if sheet_id:
        try:
            _write_df_to_sheets(df_res, sheet_id=sheet_id, worksheet=worksheet)
            print(f"[sheets] wrote RAPM output to sheetId={sheet_id} worksheet={worksheet}")
        except Exception as e:
            print(f"[error] sheets write failed: {e}", file=sys.stderr)
            return 8
    else:
        print("[warn] no Sheets id provided (set RAPM_SHEET_ID or PROJECTIONS_SHEET_ID); skipping Sheets write")

    # Context output (context_all_blend_xG67_G33) to Sheets5 by default.
    if (not args.no_context) and sheet_id:
        try:
            df_ctx = compute_context(df_data, df_res, season=int(season))
            if df_ctx.empty:
                print("[context] no context rows produced")
            else:
                # Persist context to MySQL
                try:
                    write_table(eng, df_ctx, f"nhl_{season}_context")
                except Exception as e:
                    print(f"[error] writing nhl_{season}_context failed: {e}", file=sys.stderr)
                    return 10
                _write_df_to_sheets(df_ctx, sheet_id=sheet_id, worksheet=context_worksheet)
                print(f"[sheets] wrote context output to sheetId={sheet_id} worksheet={context_worksheet}")
        except Exception as e:
            print(f"[error] context compute/write failed: {e}", file=sys.stderr)
            return 9
    elif (not args.no_context) and (not sheet_id):
        print("[warn] no Sheets id provided; skipping context Sheets write")

    if args.export_static:
        try:
            repo_root = Path(__file__).resolve().parents[1]
            static_dir = repo_root / "app" / "static" / "rapm"
            rapm_table = (args.export_rapm_table or "rapm2_all").strip()
            context_table = (args.export_context_table or "context_all").strip()
            _export_static_csvs(
                eng,
                rapm_table=rapm_table,
                context_table=context_table,
                out_dir=static_dir,
            )
        except Exception as e:
            print(f"[error] exporting static CSVs failed: {e}", file=sys.stderr)
            return 11

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
