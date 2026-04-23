"""
Game Projection Model – Preseason + combined model from Moncton Supabase data.

Step 1: Build a preseason model using previous-season player stats
        + 4 home-perspective situation dummies (no intercept).
Step 2: Compute per-player preseason composite from step 1 coefficients.
Step 3: Build a combined model blending preseason composite with in-season
        per-game stats, weighted by each player's game count.

Usage:
    python scripts/Game_Projection_Model.py                # train + evaluate (uses cached CSVs if available)
    python scripts/Game_Projection_Model.py --save         # also save models to Model/
    python scripts/Game_Projection_Model.py --refresh-data # reload data from DB and update CSVs
"""

import os
import argparse
from collections import deque
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import joblib
from scipy.optimize import minimize
from scipy.special import expit

load_dotenv()

# ── DB connection (lazy – only created when --refresh-data is used) ────
_engine = None

def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    db_url = os.getenv("DATABASE_MONCTON_URL")
    if not db_url:
        raise RuntimeError("DATABASE_MONCTON_URL not set in .env")
    db_url = db_url.replace(":6543/", ":5432/")
    _engine = create_engine(db_url)

    @event.listens_for(_engine, "connect")
    def _set_timeout(dbapi_conn, connection_record):
        cur = dbapi_conn.cursor()
        cur.execute("SET statement_timeout = 0")
        cur.close()

    return _engine


# ── Local CSV cache ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "..", "data", "game_projection")
_BASE_CSV_FILES = ["games.csv", "pvm.csv", "skaters.csv", "goalies.csv"]
_GAMESCORE_CSV = "gamescore.csv"
_SKATER_EV_XG_CSV = "skater_ev_xg.csv"
_REQUIRED_GOALIE_CACHE_COLS = {"manpower", "xga", "sa"}


def _base_csv_exists():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in _BASE_CSV_FILES)


def _gamescore_csv_exists():
    return os.path.exists(os.path.join(DATA_DIR, _GAMESCORE_CSV))


def _skater_ev_xg_csv_exists():
    return os.path.exists(os.path.join(DATA_DIR, _SKATER_EV_XG_CSV))


def _goalie_cache_is_current():
    goalies_path = os.path.join(DATA_DIR, "goalies.csv")
    if not os.path.exists(goalies_path):
        return False
    goalies_cols = set(pd.read_csv(goalies_path, nrows=0).columns)
    return _REQUIRED_GOALIE_CACHE_COLS.issubset(goalies_cols)


def _csv_exists():
    return _base_csv_exists() and _gamescore_csv_exists() and _skater_ev_xg_csv_exists()


def _save_base_csvs(games, pvm, skaters, goalies):
    os.makedirs(DATA_DIR, exist_ok=True)
    games.to_csv(os.path.join(DATA_DIR, "games.csv"), index=False)
    pvm.to_csv(os.path.join(DATA_DIR, "pvm.csv"), index=False)
    skaters.to_csv(os.path.join(DATA_DIR, "skaters.csv"), index=False)
    goalies.to_csv(os.path.join(DATA_DIR, "goalies.csv"), index=False)


def _save_gamescore_csv(gamescore):
    os.makedirs(DATA_DIR, exist_ok=True)
    gamescore.to_csv(os.path.join(DATA_DIR, _GAMESCORE_CSV), index=False)


def _save_skater_ev_xg_csv(skater_ev_xg):
    os.makedirs(DATA_DIR, exist_ok=True)
    skater_ev_xg.to_csv(os.path.join(DATA_DIR, _SKATER_EV_XG_CSV), index=False)


def _save_csvs(games, pvm, skaters, goalies, gamescore=None):
    _save_base_csvs(games, pvm, skaters, goalies)
    if gamescore is not None:
        _save_gamescore_csv(gamescore)
    print(f"  Saved data CSVs to {DATA_DIR}")


def _load_gamescore_csv():
    return pd.read_csv(
        os.path.join(DATA_DIR, _GAMESCORE_CSV),
        parse_dates=["date"],
        dtype={"season": str},
    )


def _load_skater_ev_xg_csv():
    return pd.read_csv(
        os.path.join(DATA_DIR, _SKATER_EV_XG_CSV),
        dtype={"season": str},
    )


def _load_csvs(include_gamescore=True):
    kw = {"dtype": {"season": str}}
    games = pd.read_csv(os.path.join(DATA_DIR, "games.csv"),
                        parse_dates=["date"], **kw)
    pvm = pd.read_csv(os.path.join(DATA_DIR, "pvm.csv"), **kw)
    skaters = pd.read_csv(os.path.join(DATA_DIR, "skaters.csv"), **kw)
    goalies = pd.read_csv(os.path.join(DATA_DIR, "goalies.csv"), **kw)
    print(f"  Games: {len(games)},  PVM: {len(pvm):,},"
          f"  Skaters: {len(skaters):,},  Goalies: {len(goalies):,}")
    if not include_gamescore:
        return games, pvm, skaters, goalies

    gamescore = pd.read_csv(
        os.path.join(DATA_DIR, _GAMESCORE_CSV),
        parse_dates=["date"],
        **kw,
    )
    print(f"  Gamescore rows: {len(gamescore):,}")
    return games, pvm, skaters, goalies, gamescore

GP_THRESHOLD = 41  # half-season; used for GP adjustment + rookie factor
INSEASON_WEIGHTED = False
COMBINED_WEIGHT_SLOPE = 0.015
COMBINED_WEIGHT_CAP_GAMES = 50
COMBINED_PRESEASON_FLOOR = 0.25
COMBINED_INSEASON_CEILING = 0.75
SHRINKAGE_LAMBDA_GRID = [0.0, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

# ── Feature columns ──────────────────────────────────────────────────
# poss_value = faceoffs + defensive + passes + carries + dump_ins_outs
STAT_COLS = ["poss_value", "off_the_puck", "gax", "goalie_gsax"]
ROOKIE_COLS = ["rookie_F", "rookie_D", "rookie_G"]
ALL_FEATURE_COLS = STAT_COLS + ROOKIE_COLS
INSEASON_SKATER_COLS = [
    "ev",
    "pp",
    "sh",
    "ev_xg",
]
GOALIE_METRICS = {
    "gsaa": {"column": "goalie_gsaa", "label": "GSAA"},
    "gsax_on": {"column": "goalie_gsax_on", "label": "xg_on_a - ga"},
    "gsax_all": {"column": "goalie_gsax_all", "label": "xga - ga"},
}
GOALIE_METRIC_COLS = [meta["column"] for meta in GOALIE_METRICS.values()]
DEFAULT_GOALIE_METRIC_KEY = "gsaa"
RECENT_HISTORY_WINDOW = 50
DEFAULT_RUNNING_MODEL_WINDOWS = [40, 50, 60]
DEFAULT_RUNNING_EXP_HALF_LIFE = 15.0
SITUATION_COLS = [
    "sit_b2b_b2b", "sit_b2b_rested",
    "sit_rested_b2b", "sit_rested_rested",
]

# Fixed situation log-odds from prior model (larger sample size)
FIXED_INTERCEPT = 0.1373
FIXED_SIT_COEFS = {
    "sit_b2b_b2b": -0.0766,
    "sit_b2b_rested": -0.2394,
    "sit_rested_b2b": 0.2303,
    "sit_rested_rested": 0.0,
}
FIXED_PRESEASON_COEFS = {
    "poss_value": 0.098010715415619,
    "off_the_puck": 0.070504541853348,
    "gax": 0.063566677671400,
    "goalie_gsax": 0.185349538635838,
    "rookie_F": 0.060698782843077,
    "rookie_D": 0.050160520489441,
    "rookie_G": -0.445447356373330,
}
GOALIE_PRESEASON_LOOKBACK_SEASONS = 2


class FixedCoefficientLogit:
    """Minimal no-intercept logistic model wrapper for a fixed coefficient vector."""

    def __init__(self, feature_cols: list[str], coef: np.ndarray):
        self.feature_names_in_ = np.asarray(feature_cols, dtype=object)
        self.coef_ = np.asarray([coef], dtype=float)
        self.fit_intercept = False

    def decision_function(self, X):
        if isinstance(X, pd.DataFrame):
            X_arr = X.loc[:, list(self.feature_names_in_)].to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)
        return X_arr @ self.coef_[0]


def get_goalie_metric_column(goalie_metric_key: str) -> str:
    if goalie_metric_key not in GOALIE_METRICS:
        raise ValueError(f"Unknown goalie metric key: {goalie_metric_key}")
    return GOALIE_METRICS[goalie_metric_key]["column"]


def get_goalie_metric_label(goalie_metric_key: str) -> str:
    if goalie_metric_key not in GOALIE_METRICS:
        raise ValueError(f"Unknown goalie metric key: {goalie_metric_key}")
    return GOALIE_METRICS[goalie_metric_key]["label"]


def get_inseason_stat_cols(goalie_metric_col: str) -> list[str]:
    return INSEASON_SKATER_COLS + [goalie_metric_col]


def center_metrics_by_position(
    df: pd.DataFrame,
    stat_cols: list[str] | None = None,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Center player features by season and position so values are league-average relative."""
    if stat_cols is None:
        stat_cols = ALL_FEATURE_COLS
    if group_cols is None:
        group_cols = ["season", "position"]

    centered = df.copy()
    means = (
        centered.groupby(group_cols, as_index=False)[list(stat_cols)]
        .mean()
        .rename(columns={col: f"{col}_group_mean" for col in stat_cols})
    )
    centered = centered.merge(means, on=group_cols, how="left")
    for col in stat_cols:
        mean_col = f"{col}_group_mean"
        centered[col] = centered[col].fillna(0.0) - centered[mean_col].fillna(0.0)
    centered = centered.drop(columns=[f"{col}_group_mean" for col in stat_cols])
    return centered


def compute_combined_weights(gp_before: pd.Series):
    """Custom blend schedule based on player games played before the current game."""
    gp_before = gp_before.astype(float)
    pre_w = (1.0 - COMBINED_WEIGHT_SLOPE * gp_before).clip(lower=COMBINED_PRESEASON_FLOOR)
    in_w = (COMBINED_WEIGHT_SLOPE * gp_before).clip(upper=COMBINED_INSEASON_CEILING)
    return pre_w, in_w


# ── 1. Load raw tables ───────────────────────────────────────────────
def load_team_map(conn):
    """Build {teamid: abbrev} and {full_name: abbrev} from teams table."""
    q = text("SELECT teamid, team, teamname FROM teams WHERE competition_id = 1")
    df = pd.read_sql(q, conn)
    id_to_abbr = dict(zip(df["teamid"], df["team"]))
    name_to_abbr = dict(zip(df["teamname"], df["team"]))
    for a in df["team"]:
        name_to_abbr[a] = a
    return id_to_abbr, name_to_abbr


def load_games(conn, id_to_abbr: dict) -> pd.DataFrame:
    """NHL regular-season completed games."""
    q = text("""
        SELECT game_id, season, date,
               away_team_id, home_team_id,
               home_score, away_score
        FROM games
        WHERE competition_id = '1'
          AND season_stage = 'regular'
          AND event_status = 'over'
        ORDER BY date, game_id
    """)
    df = pd.read_sql(q, conn)
    df["awayteam"] = df["away_team_id"].map(id_to_abbr)
    df["hometeam"] = df["home_team_id"].map(id_to_abbr)
    df["home_win"] = (df["home_score"] > df["away_score"]).astype(int)
    df["date"] = pd.to_datetime(df["date"])
    before = len(df)
    df = df.dropna(subset=["awayteam", "hometeam"])
    if len(df) < before:
        print(f"  Warning: dropped {before - len(df)} games with unmapped team IDs")
    return df


def load_pvm(conn, name_to_abbr: dict) -> pd.DataFrame:
    """Possession values per player per game (NHL only)."""
    q = text("""
        SELECT season, playerid, gameid, team, position,
               faceoffs, defensive, passes, carries,
               dump_ins_outs, off_the_puck
        FROM possession_values_master
        WHERE league = '1'
    """)
    print("    Loading PVM …", end=" ", flush=True)
    df = pd.read_sql(q, conn)
    print(f"{len(df):,} rows")
    df["team"] = df["team"].map(name_to_abbr).fillna(df["team"])
    return df


def load_skaters(conn) -> pd.DataFrame:
    """Skater ig/ixg per game (NHL regular season only)."""
    q = text("""
        SELECT season, playerid, gameid, ig, ixg, toi
        FROM skaters_master
        WHERE league = '1'
          AND seasonstage = 'regular'
    """)
    print("    Loading Skaters …", end=" ", flush=True)
    df = pd.read_sql(q, conn)
    print(f"{len(df):,} rows")
    return df


def load_skater_ev_xg(conn) -> pd.DataFrame:
    """Skater EV xGF/xGA per player-game (NHL regular season only)."""
    q = text("""
        SELECT season, playerid, gameid, xgf, xga
        FROM skaters_master
        WHERE league = '1'
          AND seasonstage = 'regular'
          AND manpower = 'EV'
    """)
    print("    Loading Skater EV xG …", end=" ", flush=True)
    df = pd.read_sql(q, conn)
    print(f"{len(df):,} rows")
    return df


def load_goalies(conn) -> pd.DataFrame:
    """Goalie per-game results inputs (NHL regular season only)."""
    q = text("""
        SELECT season, playerid, gameid, manpower, xg_on_a, xga, ga, sa, toi
        FROM goalies_master
        WHERE league = '1'
          AND seasonstage = 'regular'
    """)
    print("    Loading Goalies …", end=" ", flush=True)
    df = pd.read_sql(q, conn)
    print(f"{len(df):,} rows")
    return df


def load_gamescore(conn, name_to_abbr: dict) -> pd.DataFrame:
        """Skater game score by manpower plus goalie rows for roster/team mapping."""
        q = text("""
                SELECT season, gameid, date, playerid, position, team,
                             manpower, toi, gs_def, gs_off
                FROM gamescore_master
                WHERE league = '1'
                    AND seasonstage = 'regular'
                    AND manpower IN ('EV', 'PP', 'SH')
        """)
        print("    Loading Gamescore …", end=" ", flush=True)
        df = pd.read_sql(q, conn)
        print(f"{len(df):,} rows")
        df["team"] = df["team"].map(name_to_abbr).fillna(df["team"])
        df["date"] = pd.to_datetime(df["date"])
        df["manpower"] = df["manpower"].astype(str).str.upper().str.strip()
        return df


def _aggregate_toi_per_player_game(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-player TOI to one row per season/player/game."""
    base_cols = ["season", "playerid", "gameid"]
    toi_col = None
    if "toi" in df.columns:
        toi_col = "toi"
    elif "TOI" in df.columns:
        toi_col = "TOI"

    if toi_col is None:
        out = df[base_cols].drop_duplicates().copy()
        out["toi"] = np.nan
        return out

    return (
        df.groupby(base_cols, as_index=False)
        .agg(toi=(toi_col, "sum"))
    )


def build_inseason_player_game_stats(games: pd.DataFrame,
                                     gamescore: pd.DataFrame,
                                     goalies: pd.DataFrame,
                                     skater_ev_xg: pd.DataFrame) -> pd.DataFrame:
    """Build one row per player-game with skater in-season features and goalie metrics."""
    skater_gamescore = gamescore[gamescore["position"].isin(["F", "D"])].copy()
    skater_gamescore["ev"] = np.where(
        skater_gamescore["manpower"] == "EV",
        skater_gamescore["gs_def"] + skater_gamescore["gs_off"],
        0.0,
    )
    skater_gamescore["pp"] = np.where(skater_gamescore["manpower"] == "PP", skater_gamescore["gs_off"], 0.0)
    skater_gamescore["sh"] = np.where(skater_gamescore["manpower"] == "SH", skater_gamescore["gs_def"], 0.0)

    skater_pg = (
        skater_gamescore.groupby(["season", "playerid", "gameid", "position", "team"], as_index=False)
        .agg(
            ev=("ev", "sum"),
            pp=("pp", "sum"),
            sh=("sh", "sum"),
            toi=("toi", "sum"),
        )
    )

    ev_xg_pg = skater_ev_xg.copy()
    ev_xg_pg["ev_xg"] = ev_xg_pg["xgf"].fillna(0.0) - ev_xg_pg["xga"].fillna(0.0)
    ev_xg_pg = (
        ev_xg_pg.groupby(["season", "playerid", "gameid"], as_index=False)
        .agg(ev_xg=("ev_xg", "sum"))
    )
    skater_pg = skater_pg.merge(
        ev_xg_pg,
        on=["season", "playerid", "gameid"],
        how="left",
    )
    skater_pg["ev_xg"] = skater_pg["ev_xg"].fillna(0.0)

    for col in GOALIE_METRIC_COLS:
        skater_pg[col] = 0.0

    goalie_info = (
        gamescore[gamescore["position"] == "G"]
        [["season", "playerid", "gameid", "team"]]
        .drop_duplicates()
    )

    goalie_stats = goalies.copy()
    for col in ["xg_on_a", "xga", "ga", "sa", "toi"]:
        goalie_stats[col] = goalie_stats[col].fillna(0.0)

    league_sv = (
        goalie_stats.groupby(["season", "manpower"], as_index=False)
        .agg(league_sa=("sa", "sum"), league_ga=("ga", "sum"))
    )
    league_sv["league_sv"] = np.where(
        league_sv["league_sa"] > 0,
        1.0 - (league_sv["league_ga"] / league_sv["league_sa"]),
        1.0,
    )
    goalie_stats = goalie_stats.merge(
        league_sv[["season", "manpower", "league_sv"]],
        on=["season", "manpower"],
        how="left",
    )
    goalie_stats["goalie_gsax_on"] = goalie_stats["xg_on_a"] - goalie_stats["ga"]
    goalie_stats["goalie_gsax_all"] = goalie_stats["xga"] - goalie_stats["ga"]
    goalie_sv = np.where(
        goalie_stats["sa"] > 0,
        1.0 - (goalie_stats["ga"] / goalie_stats["sa"]),
        goalie_stats["league_sv"].fillna(1.0),
    )
    goalie_stats["goalie_gsaa"] = np.where(
        goalie_stats["sa"] > 0,
        (goalie_sv - goalie_stats["league_sv"].fillna(1.0)) * goalie_stats["sa"],
        0.0,
    )
    goalie_pg = (
        goalie_stats.groupby(["season", "playerid", "gameid"], as_index=False)
        .agg(
            goalie_gsax_on=("goalie_gsax_on", "sum"),
            goalie_gsax_all=("goalie_gsax_all", "sum"),
            goalie_gsaa=("goalie_gsaa", "sum"),
            toi=("toi", "sum"),
        )
        .merge(goalie_info, on=["season", "playerid", "gameid"], how="left")
    )
    goalie_pg["position"] = "G"
    for col in INSEASON_SKATER_COLS:
        goalie_pg[col] = 0.0
    goalie_pg["toi"] = goalie_pg["toi"].fillna(0.0)
    goalie_pg["team"] = goalie_pg["team"].fillna("")

    pg = pd.concat(
        [
            skater_pg[["season", "playerid", "gameid", "position", "team", "toi"] + INSEASON_SKATER_COLS + GOALIE_METRIC_COLS],
            goalie_pg[["season", "playerid", "gameid", "position", "team", "toi"] + INSEASON_SKATER_COLS + GOALIE_METRIC_COLS],
        ],
        ignore_index=True,
        sort=False,
    )
    pg = pg[pg["toi"].fillna(0) > 0].copy()

    game_dates = games[["game_id", "date"]].rename(columns={"game_id": "gameid"})
    pg = pg.merge(game_dates, on="gameid", how="inner")
    pg = pg.sort_values(["season", "playerid", "date", "gameid"])
    return pg


# ── 2. Back-to-back + situation dummies ──────────────────────────────
def add_b2b(games: pd.DataFrame) -> pd.DataFrame:
    """Add home_b2b and away_b2b columns."""
    games = games.copy()
    home_dates = games[["hometeam", "date"]].rename(columns={"hometeam": "team"})
    away_dates = games[["awayteam", "date"]].rename(columns={"awayteam": "team"})
    played = set(zip(
        pd.concat([home_dates["team"], away_dates["team"]]),
        pd.concat([home_dates["date"], away_dates["date"]]),
    ))
    prev = games["date"] - pd.Timedelta(days=1)
    games["home_b2b"] = [int((t, d) in played) for t, d in zip(games["hometeam"], prev)]
    games["away_b2b"] = [int((t, d) in played) for t, d in zip(games["awayteam"], prev)]
    return games


def add_situation(games: pd.DataFrame) -> pd.DataFrame:
    """Add 4 situation dummies from the home team's perspective."""
    games = games.copy()
    hb, ab = games["home_b2b"], games["away_b2b"]
    games["sit_b2b_b2b"]       = ((hb == 1) & (ab == 1)).astype(int)
    games["sit_b2b_rested"]    = ((hb == 1) & (ab == 0)).astype(int)
    games["sit_rested_b2b"]    = ((hb == 0) & (ab == 1)).astype(int)
    games["sit_rested_rested"] = ((hb == 0) & (ab == 0)).astype(int)
    return games


def compute_situation_offset(games: pd.DataFrame) -> pd.Series:
    """Fixed situation log-odds per game (home perspective)."""
    offset = pd.Series(FIXED_INTERCEPT, index=games.index)
    for col, val in FIXED_SIT_COEFS.items():
        offset += games[col] * val
    return offset


# ── 3. Player season profiles ────────────────────────────────────────
def prev_season(season: str) -> str:
    """20232024 → 20222023"""
    start = int(season[:4]) - 1
    return f"{start}{start + 1}"


def season_years_ago(season: str, years_back: int) -> str:
    """Return the season string for N years before the given season."""
    out = season
    for _ in range(years_back):
        out = prev_season(out)
    return out


def build_player_profiles(pvm: pd.DataFrame,
                          skaters: pd.DataFrame,
                          goalies: pd.DataFrame) -> pd.DataFrame:
    """
    Per-player per-season profiles with GP-adjusted stats.

    For each stat: adjusted = season_total / max(gp, 41)
    Rookie factor:  (41 - min(gp, 41)) / 41   (1.0 for true rookies, 0 for ≥41 GP)
    """
    pvm = pvm.copy()
    pvm["poss_value"] = (pvm["faceoffs"] + pvm["defensive"] + pvm["passes"]
                         + pvm["carries"] + pvm["dump_ins_outs"])

    pvm_season = (
        pvm.groupby(["season", "playerid"])
        .agg(
            poss_value=("poss_value", "sum"),
            off_the_puck=("off_the_puck", "sum"),
            gp=("gameid", "nunique"),
            position=("position", lambda x: x.mode().iloc[0]),
        )
        .reset_index()
    )

    sk_season = (
        skaters.groupby(["season", "playerid"])
        .agg(ig=("ig", "sum"), ixg=("ixg", "sum"))
        .reset_index()
    )
    sk_season["gax"] = sk_season["ig"] - sk_season["ixg"]

    gl_season = (
        goalies.groupby(["season", "playerid"])
        .agg(
            xg_on_a=("xg_on_a", "sum"),
            ga=("ga", "sum"),
            goalie_gp=("gameid", "nunique"),
        )
        .reset_index()
    )
    gl_season["goalie_gsax_total"] = gl_season["xg_on_a"] - gl_season["ga"]
    gl_season["goalie_gsax_adj"] = (
        gl_season["goalie_gsax_total"]
        / gl_season["goalie_gp"].clip(lower=GP_THRESHOLD)
    )
    gl_season["prev_season"] = gl_season["season"].apply(prev_season)
    gl_prev = gl_season[["season", "playerid", "goalie_gsax_adj"]].rename(
        columns={
            "season": "prev_season",
            "goalie_gsax_adj": "prev_goalie_gsax_adj",
        }
    )
    gl_season = gl_season.merge(gl_prev, on=["prev_season", "playerid"], how="left")
    gl_season["goalie_gsax"] = (
        gl_season["goalie_gsax_adj"] + gl_season["prev_goalie_gsax_adj"].fillna(0.0)
    ) / (1.0 + gl_season["prev_goalie_gsax_adj"].notna().astype(float))

    profiles = (
        pvm_season
        .merge(sk_season[["season", "playerid", "gax"]],
               on=["season", "playerid"], how="left")
        .merge(gl_season[["season", "playerid", "goalie_gsax"]],
               on=["season", "playerid"], how="left")
    )
    profiles["gax"] = profiles["gax"].fillna(0)
    profiles["goalie_gsax"] = profiles["goalie_gsax"].fillna(0)

    gp_adj = profiles["gp"].clip(lower=GP_THRESHOLD)
    for col in ("poss_value", "off_the_puck", "gax"):
        profiles[col] = profiles[col] / gp_adj

    profiles["rookie_factor"] = (
        (GP_THRESHOLD - profiles["gp"].clip(upper=GP_THRESHOLD)) / GP_THRESHOLD
    )

    return profiles


def build_player_game_box_stats(pvm: pd.DataFrame,
                                skaters: pd.DataFrame,
                                goalies: pd.DataFrame,
                                games: pd.DataFrame) -> pd.DataFrame:
    """Build one row per player-game with preseason-style stat columns."""
    pvm_pg = pvm.copy()
    pvm_pg["poss_value"] = (
        pvm_pg["faceoffs"]
        + pvm_pg["defensive"]
        + pvm_pg["passes"]
        + pvm_pg["carries"]
        + pvm_pg["dump_ins_outs"]
    )
    pvm_pg = (
        pvm_pg.groupby(["season", "playerid", "gameid", "team", "position"], as_index=False)
        .agg(
            poss_value=("poss_value", "sum"),
            off_the_puck=("off_the_puck", "sum"),
        )
    )

    skater_pg = (
        skaters.groupby(["season", "playerid", "gameid"], as_index=False)
        .agg(ig=("ig", "sum"), ixg=("ixg", "sum"))
    )
    skater_pg["gax"] = skater_pg["ig"].fillna(0.0) - skater_pg["ixg"].fillna(0.0)

    goalie_pg = (
        goalies.groupby(["season", "playerid", "gameid"], as_index=False)
        .agg(xg_on_a=("xg_on_a", "sum"), ga=("ga", "sum"))
    )
    goalie_pg["goalie_gsax"] = goalie_pg["xg_on_a"].fillna(0.0) - goalie_pg["ga"].fillna(0.0)

    game_dates = games[["game_id", "date"]].rename(columns={"game_id": "gameid"})
    pg = (
        pvm_pg
        .merge(
            skater_pg[["season", "playerid", "gameid", "gax"]],
            on=["season", "playerid", "gameid"],
            how="left",
        )
        .merge(
            goalie_pg[["season", "playerid", "gameid", "goalie_gsax"]],
            on=["season", "playerid", "gameid"],
            how="left",
        )
        .merge(game_dates, on="gameid", how="inner")
    )
    pg["gax"] = pg["gax"].fillna(0.0)
    pg["goalie_gsax"] = pg["goalie_gsax"].fillna(0.0)
    pg = pg.sort_values(["playerid", "date", "gameid"])
    return pg


def format_running_model_half_life(exp_half_life: float) -> str:
    """Return a stable half-life string for filenames and labels."""
    if float(exp_half_life).is_integer():
        return str(int(exp_half_life))
    return str(exp_half_life).replace(".", "p")


def get_running_model_variant_key(window_games: int,
                                  weighting: str = "uniform",
                                  exp_half_life: float | None = None) -> str:
    """Build a stable key for a running-model variant."""
    if window_games <= 0:
        raise ValueError("window_games must be positive")
    if weighting == "uniform":
        return f"recent{window_games}"
    if weighting == "exponential":
        if exp_half_life is None or exp_half_life <= 0:
            raise ValueError("exp_half_life must be positive for exponential weighting")
        return f"recent{window_games}_exp_hl{format_running_model_half_life(exp_half_life)}"
    raise ValueError(f"Unknown running-model weighting: {weighting}")


def get_running_model_variant_label(window_games: int,
                                    weighting: str = "uniform",
                                    exp_half_life: float | None = None) -> str:
    """Build a human-readable label for a running-model variant."""
    if weighting == "uniform":
        return f"Recent-{window_games}"
    if weighting == "exponential":
        if exp_half_life is None or exp_half_life <= 0:
            raise ValueError("exp_half_life must be positive for exponential weighting")
        return f"Recent-{window_games} Exp (half-life {exp_half_life:g})"
    raise ValueError(f"Unknown running-model weighting: {weighting}")


def describe_running_model_source(window_games: int,
                                  weighting: str = "uniform",
                                  exp_half_life: float | None = None) -> str:
    """Describe the data source and weighting scheme for a running-model artifact."""
    if weighting == "uniform":
        return (
            f"latest {window_games} games from previous and current season before each game date"
        )
    if weighting == "exponential":
        if exp_half_life is None or exp_half_life <= 0:
            raise ValueError("exp_half_life must be positive for exponential weighting")
        return (
            f"exponentially weighted latest {window_games} games from previous and current season "
            f"before each game date (half-life {exp_half_life:g} games)"
        )
    raise ValueError(f"Unknown running-model weighting: {weighting}")


def build_running_slot_weights(window_games: int,
                               weighting: str = "uniform",
                               exp_half_life: float | None = None) -> np.ndarray:
    """Build per-slot recency weights from most recent to oldest game."""
    if window_games <= 0:
        raise ValueError("window_games must be positive")
    if weighting == "uniform":
        return np.full(window_games, 1.0 / window_games, dtype=float)
    if weighting == "exponential":
        if exp_half_life is None or exp_half_life <= 0:
            raise ValueError("exp_half_life must be positive for exponential weighting")
        slot_ages = np.arange(window_games, dtype=float)
        raw_weights = np.exp(-np.log(2.0) * slot_ages / exp_half_life)
        return raw_weights / raw_weights.sum()
    raise ValueError(f"Unknown running-model weighting: {weighting}")


def get_running_stat_subset_suffix(stat_cols: list[str] | None = None) -> str:
    """Return a filename-safe suffix for non-default running stat subsets."""
    if stat_cols is None or list(stat_cols) == STAT_COLS:
        return ""
    return "_" + "_".join(stat_cols)


def build_running_rookie_stat_priors(player_game_stats: pd.DataFrame,
                                     window_games: int) -> dict[str, dict[str, float]]:
    """Estimate position-level rookie stat priors on the same per-game scale as running profiles."""
    if window_games <= 0:
        raise ValueError("window_games must be positive")

    season_player = (
        player_game_stats.groupby(["season", "playerid", "position"], as_index=False)
        .agg(
            gp=("gameid", "nunique"),
            **{col: (col, "sum") for col in STAT_COLS},
        )
    )
    season_player["gp"] = season_player["gp"].clip(lower=1)
    season_player["prior_weight"] = (
        (window_games - season_player["gp"].clip(upper=window_games)) / window_games
    )

    for col in STAT_COLS:
        season_player[col] = season_player[col] / season_player["gp"]

    priors: dict[str, dict[str, float]] = {
        pos: {col: 0.0 for col in STAT_COLS}
        for pos in ("F", "D", "G")
    }

    weighted_pool = season_player[season_player["prior_weight"] > 0].copy()
    if weighted_pool.empty:
        return priors

    for pos in priors:
        pos_pool = weighted_pool[weighted_pool["position"] == pos].copy()
        if pos_pool.empty:
            continue
        weights = pos_pool["prior_weight"].to_numpy(dtype=float)
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            continue
        for col in STAT_COLS:
            priors[pos][col] = float(np.dot(pos_pool[col].to_numpy(dtype=float), weights) / weight_sum)

    return priors


def build_running_player_profiles(player_game_stats: pd.DataFrame,
                                  window_games: int = RECENT_HISTORY_WINDOW,
                                  weighting: str = "uniform",
                                  exp_half_life: float | None = None,
                                  rookie_stat_priors: dict[str, dict[str, float]] | None = None) -> pd.DataFrame:
    """Build pregame player profiles from prior current+previous-season games with configurable recency weighting."""
    slot_weights = build_running_slot_weights(
        window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )
    if rookie_stat_priors is None:
        rookie_stat_priors = build_running_rookie_stat_priors(player_game_stats, window_games)
    observed_weight_by_count = np.concatenate(([0.0], np.cumsum(slot_weights)))
    profile_rows = []
    stat_cols = STAT_COLS

    for _, player_games in player_game_stats.groupby("playerid", sort=False):
        history = deque()
        player_games = player_games.sort_values(["date", "gameid"])

        for row in player_games.itertuples(index=False):
            allowed_prev_seasons = {row.season, prev_season(row.season)}
            if history and any(item["season"] not in allowed_prev_seasons for item in history):
                history = deque(
                    item for item in history
                    if item["season"] in allowed_prev_seasons
                )

            games_in_window = len(history)
            observed_weight = float(observed_weight_by_count[games_in_window])
            missing_share = max(0.0, 1.0 - observed_weight)
            weighted_stats = {col: 0.0 for col in stat_cols}
            if games_in_window:
                recent_history = list(reversed(history))
                for weight, item in zip(slot_weights[:games_in_window], recent_history):
                    for col in stat_cols:
                        weighted_stats[col] += float(weight) * item[col]

            rookie_prior = rookie_stat_priors.get(row.position, {})
            for col in stat_cols:
                weighted_stats[col] += missing_share * float(rookie_prior.get(col, 0.0))

            profile_row = {
                "season": row.season,
                "gameid": row.gameid,
                "playerid": row.playerid,
                "team": row.team,
                "position": row.position,
                "date": row.date,
                "games_in_window": games_in_window,
                "rookie_factor": missing_share,
            }
            for col in stat_cols:
                profile_row[col] = weighted_stats[col]
            for pos in ("F", "D", "G"):
                profile_row[f"rookie_{pos}"] = missing_share * float(row.position == pos)
            profile_rows.append(profile_row)

            current_item = {"season": row.season}
            for col in stat_cols:
                current_item[col] = getattr(row, col)
            history.append(current_item)

            if len(history) > window_games:
                history.popleft()

    return pd.DataFrame(profile_rows)


def build_recent_50_player_profiles(player_game_stats: pd.DataFrame,
                                    window_games: int = RECENT_HISTORY_WINDOW) -> pd.DataFrame:
    """Backward-compatible wrapper for the uniform recent-50 running model."""
    return build_running_player_profiles(
        player_game_stats,
        window_games=window_games,
        weighting="uniform",
    )


def build_recent_50_features(games: pd.DataFrame,
                             recent_profiles: pd.DataFrame,
                             stat_cols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate running player profiles into home/away game features."""
    if stat_cols is None:
        stat_cols = STAT_COLS

    game_teams = games[["game_id", "hometeam", "awayteam"]].rename(columns={"game_id": "gameid"})
    roster = recent_profiles.merge(game_teams, on="gameid", how="inner")

    is_home = roster["team"] == roster["hometeam"]
    is_away = roster["team"] == roster["awayteam"]
    roster = roster[is_home | is_away].copy()
    roster["side"] = np.where(roster["team"] == roster["hometeam"], "home", "away")

    agg_cols = list(stat_cols) + ROOKIE_COLS
    team_agg = roster.groupby(["gameid", "side"])[agg_cols].sum().reset_index()

    home = (
        team_agg[team_agg["side"] == "home"]
        .drop(columns=["side"])
        .rename(columns={c: f"home_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )
    away = (
        team_agg[team_agg["side"] == "away"]
        .drop(columns=["side"])
        .rename(columns={c: f"away_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )

    result_base = games[["game_id", "season", "home_win"]].copy()
    result_base["situation_offset"] = compute_situation_offset(games)
    result = (
        result_base
        .merge(home, on="game_id", how="inner")
        .merge(away, on="game_id", how="inner")
    )
    return result


# ── 4. Preseason feature construction ────────────────────────────────
def build_preseason_features(games: pd.DataFrame,
                             pvm: pd.DataFrame,
                             profiles: pd.DataFrame) -> pd.DataFrame:
    """
    For each game, look up every player's previous-season profile,
    apply GP adjustment + rookie constants, and aggregate per team.
    """
    roster = (
        pvm[["gameid", "playerid", "team", "position"]]
        .drop_duplicates(subset=["gameid", "playerid"])
    )

    game_season = games[["game_id", "season"]].rename(columns={"game_id": "gameid"})
    roster = roster.merge(game_season, on="gameid", how="inner")
    roster["prev_season"] = roster["season"].apply(prev_season)

    prev_stats = profiles[["season", "playerid"] + STAT_COLS + ["rookie_factor"]].copy()
    prev_stats = prev_stats.rename(columns={"season": "prev_season"})

    roster = roster.merge(prev_stats, on=["prev_season", "playerid"], how="left")

    for c in STAT_COLS:
        roster[c] = roster[c].fillna(0)
    roster["rookie_factor"] = roster["rookie_factor"].fillna(1.0)

    for pos in ("F", "D", "G"):
        roster[f"rookie_{pos}"] = roster["rookie_factor"] * (roster["position"] == pos).astype(float)

    game_teams = games[["game_id", "hometeam", "awayteam"]].rename(columns={"game_id": "gameid"})
    roster = roster.merge(game_teams, on="gameid", how="inner")
    is_home = roster["team"] == roster["hometeam"]
    is_away = roster["team"] == roster["awayteam"]
    roster = roster[is_home | is_away].copy()
    roster["side"] = np.where(roster["team"] == roster["hometeam"], "home", "away")

    agg_cols = STAT_COLS + ["rookie_F", "rookie_D", "rookie_G"]
    team_agg = roster.groupby(["gameid", "side"])[agg_cols].sum().reset_index()

    home = (
        team_agg[team_agg["side"] == "home"]
        .drop(columns=["side"])
        .rename(columns={c: f"home_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )
    away = (
        team_agg[team_agg["side"] == "away"]
        .drop(columns=["side"])
        .rename(columns={c: f"away_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )

    result_base = games[["game_id", "season", "home_win"]].copy()
    result_base["situation_offset"] = compute_situation_offset(games)
    result = (
        result_base
        .merge(home, on="game_id", how="inner")
        .merge(away, on="game_id", how="inner")
    )
    return result


def build_preseason_updating_features(games: pd.DataFrame,
                                      player_game_box_stats: pd.DataFrame) -> pd.DataFrame:
    """Build preseason-style game features updated with current-season stats before each game.

    For each player-game, the profile is:
      (previous-season totals + current-season cumulative totals before the game)
      / max(previous-season GP + current-season GP before the game, 41)

    Rookie factor follows the same GP-threshold logic as the preseason model.
    """
    profiles = build_preseason_updating_player_profiles(player_game_box_stats)

    game_teams = games[["game_id", "hometeam", "awayteam"]].rename(columns={"game_id": "gameid"})
    pg = profiles.merge(game_teams, on="gameid", how="inner")
    is_home = pg["team"] == pg["hometeam"]
    is_away = pg["team"] == pg["awayteam"]
    pg = pg[is_home | is_away].copy()
    pg["side"] = np.where(pg["team"] == pg["hometeam"], "home", "away")

    agg_cols = STAT_COLS + ROOKIE_COLS
    team_agg = pg.groupby(["gameid", "side"])[agg_cols].sum().reset_index()

    home = (
        team_agg[team_agg["side"] == "home"]
        .drop(columns=["side"])
        .rename(columns={c: f"home_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )
    away = (
        team_agg[team_agg["side"] == "away"]
        .drop(columns=["side"])
        .rename(columns={c: f"away_{c}" for c in agg_cols})
        .rename(columns={"gameid": "game_id"})
    )

    result_base = games[["game_id", "season", "home_win"]].copy()
    result_base["situation_offset"] = compute_situation_offset(games)
    result = (
        result_base
        .merge(home, on="game_id", how="inner")
        .merge(away, on="game_id", how="inner")
    )
    return result


def build_preseason_updating_player_profiles(player_game_box_stats: pd.DataFrame) -> pd.DataFrame:
    """Build one pregame player profile row using prior-season plus in-season cumulative stats."""
    season_totals = player_game_box_stats.groupby(["season", "playerid"], as_index=False).agg(
        gp=("gameid", "nunique"),
        **{col: (col, "sum") for col in STAT_COLS},
    )

    pg = player_game_box_stats.copy().sort_values(["season", "playerid", "date", "gameid"])
    for col in STAT_COLS:
        pg[f"cum_{col}"] = pg.groupby(["season", "playerid"])[col].cumsum() - pg[col]
    pg["gp_before"] = pg.groupby(["season", "playerid"]).cumcount()

    prev_totals = season_totals.rename(
        columns={
            "season": "prev_season",
            "gp": "prev_gp",
            **{col: f"prev_{col}" for col in STAT_COLS},
        }
    )
    pg["prev_season"] = pg["season"].apply(prev_season)
    pg = pg.merge(
        prev_totals[["prev_season", "playerid", "prev_gp"] + [f"prev_{col}" for col in STAT_COLS]],
        on=["prev_season", "playerid"],
        how="left",
    )

    pg["prev_gp"] = pg["prev_gp"].fillna(0)
    for col in STAT_COLS:
        pg[f"prev_{col}"] = pg[f"prev_{col}"].fillna(0.0)

    goalie_prev_totals = season_totals[["season", "playerid", "gp", "goalie_gsax"]].rename(
        columns={
            "season": "goalie_prev_season",
            "gp": "goalie_prev_gp_1",
            "goalie_gsax": "goalie_prev_gsax_1",
        }
    )
    goalie_prev2_totals = season_totals[["season", "playerid", "gp", "goalie_gsax"]].rename(
        columns={
            "season": "goalie_prev2_season",
            "gp": "goalie_prev_gp_2",
            "goalie_gsax": "goalie_prev_gsax_2",
        }
    )
    pg["goalie_prev_season"] = pg["season"].apply(prev_season)
    pg["goalie_prev2_season"] = pg["season"].apply(
        lambda season: season_years_ago(season, GOALIE_PRESEASON_LOOKBACK_SEASONS)
    )
    pg = pg.merge(goalie_prev_totals, on=["goalie_prev_season", "playerid"], how="left")
    pg = pg.merge(goalie_prev2_totals, on=["goalie_prev2_season", "playerid"], how="left")
    pg["prev_goalie_gsax"] = (
        pg["goalie_prev_gsax_1"].fillna(0.0)
        + pg["goalie_prev_gsax_2"].fillna(0.0)
    )
    pg["prev_goalie_gp"] = (
        pg["goalie_prev_gp_1"].fillna(0.0)
        + pg["goalie_prev_gp_2"].fillna(0.0)
    )

    pg["gp_total_before"] = pg["prev_gp"] + pg["gp_before"]
    for col in STAT_COLS:
        if col == "goalie_gsax":
            gp_adj = (pg["prev_goalie_gp"] + pg["gp_before"]).clip(lower=GP_THRESHOLD)
        else:
            gp_adj = pg["gp_total_before"].clip(lower=GP_THRESHOLD)
        pg[col] = (pg[f"prev_{col}"] + pg[f"cum_{col}"]) / gp_adj

    pg["rookie_factor"] = (
        (GP_THRESHOLD - pg["gp_total_before"].clip(upper=GP_THRESHOLD)) / GP_THRESHOLD
    )
    for pos in ("F", "D", "G"):
        pg[f"rookie_{pos}"] = pg["rookie_factor"] * (pg["position"] == pos).astype(float)

    pg["games_in_window"] = pg["gp_total_before"].astype(int)
    return pg[
        [
            "season",
            "gameid",
            "playerid",
            "team",
            "position",
            "date",
            "games_in_window",
            "rookie_factor",
            *STAT_COLS,
            *ROOKIE_COLS,
        ]
    ].copy()


# ── 5. Train preseason logistic regression (no intercept) ─────────────
def train_preseason_model(features_df: pd.DataFrame,
                         seed=None,
                         model_label: str = "Preseason Model",
                         base_feature_cols: list[str] | None = None):
    """Train on home−away stat diffs; situation is fixed (not learned)."""
    if base_feature_cols is None:
        base_feature_cols = ALL_FEATURE_COLS

    diff_cols = []
    for c in base_feature_cols:
        col = f"diff_{c}"
        features_df[col] = features_df[f"home_{c}"] - features_df[f"away_{c}"]
        diff_cols.append(col)

    feature_cols = diff_cols  # situation is fixed, not learned
    X = features_df[feature_cols].copy()
    y = features_df["home_win"].copy()
    sit_offset = features_df["situation_offset"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X, y, sit_offset = X[mask], y[mask], sit_offset[mask]

    X_train, X_test, y_train, y_test, sit_train, sit_test = train_test_split(
        X, y, sit_offset, test_size=0.2, random_state=seed
    )

    model = LogisticRegression(max_iter=5000, random_state=0, fit_intercept=False)
    model.fit(X_train, y_train)

    coef_df = pd.DataFrame({"feature": feature_cols, "coef": model.coef_[0]})
    print(f"\n=== {model_label} Coefficients (learned) ===")
    print(coef_df.to_string(index=False))
    print(f"\n=== Fixed Situation Values ===")
    print(f"  Intercept:          {FIXED_INTERCEPT}")
    for col, val in FIXED_SIT_COEFS.items():
        print(f"  {col:22s} {val:+.4f}")

    # Evaluate with fixed situation offset added to log-odds
    y_prob_train = expit(model.decision_function(X_train) + sit_train.values)
    y_prob_test = expit(model.decision_function(X_test) + sit_test.values)
    print(f"\n=== {model_label} Evaluation ===")
    print(f"Train  LogLoss: {log_loss(y_train, y_prob_train):.4f}  "
          f"AUC: {roc_auc_score(y_train, y_prob_train):.4f}")
    print(f"Test   LogLoss: {log_loss(y_test, y_prob_test):.4f}  "
          f"AUC: {roc_auc_score(y_test, y_prob_test):.4f}")
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    hw_rate = y.mean()
    baseline_ll = log_loss(y, [hw_rate] * len(y))
    print(f"\nBaseline (always predict {hw_rate:.3f}): LogLoss {baseline_ll:.4f}")

    metrics = {
        "train_logloss": log_loss(y_train, y_prob_train),
        "train_auc": roc_auc_score(y_train, y_prob_train),
        "test_logloss": log_loss(y_test, y_prob_test),
        "test_auc": roc_auc_score(y_test, y_prob_test),
        "baseline_logloss": baseline_ll,
        "train_size": len(X_train),
        "test_size": len(X_test),
    }

    return model, feature_cols, coef_df, metrics


def build_fixed_preseason_model(base_feature_cols: list[str] | None = None,
                                fixed_preseason_coefs: dict[str, float] | None = None):
    """Build a model wrapper and coefficient table from the frozen preseason coefficients."""
    if base_feature_cols is None:
        base_feature_cols = ALL_FEATURE_COLS
    if fixed_preseason_coefs is None:
        fixed_preseason_coefs = FIXED_PRESEASON_COEFS

    feature_cols = [f"diff_{col}" for col in base_feature_cols]
    coef = np.array([fixed_preseason_coefs[col] for col in base_feature_cols], dtype=float)
    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    return FixedCoefficientLogit(feature_cols, coef), feature_cols, coef_df


def evaluate_fixed_preseason_model(features_df: pd.DataFrame,
                                   seed=None,
                                   model_label: str = "Preseason Model",
                                   base_feature_cols: list[str] | None = None,
                                   fixed_preseason_coefs: dict[str, float] | None = None):
    """Evaluate a preseason-style feature table with a frozen coefficient vector."""
    if base_feature_cols is None:
        base_feature_cols = ALL_FEATURE_COLS

    features_df = features_df.copy()
    model, feature_cols, coef_df = build_fixed_preseason_model(
        base_feature_cols=base_feature_cols,
        fixed_preseason_coefs=fixed_preseason_coefs,
    )

    for c in base_feature_cols:
        features_df[f"diff_{c}"] = features_df[f"home_{c}"] - features_df[f"away_{c}"]

    X = features_df[feature_cols].copy()
    y = features_df["home_win"].copy()
    sit_offset = features_df["situation_offset"].copy()

    mask = X.notna().all(axis=1) & y.notna()
    X, y, sit_offset = X[mask], y[mask], sit_offset[mask]

    X_train, X_test, y_train, y_test, sit_train, sit_test = train_test_split(
        X, y, sit_offset, test_size=0.2, random_state=seed
    )

    print(f"\n=== {model_label} Coefficients (fixed) ===")
    print(coef_df.to_string(index=False))
    print(f"\n=== Fixed Situation Values ===")
    print(f"  Intercept:          {FIXED_INTERCEPT}")
    for col, val in FIXED_SIT_COEFS.items():
        print(f"  {col:22s} {val:+.4f}")

    train_logits = model.decision_function(X_train) + sit_train.values
    test_logits = model.decision_function(X_test) + sit_test.values
    metrics = print_model_evaluation(model_label, y_train, train_logits, y_test, test_logits, y)
    return model, feature_cols, coef_df, metrics


# ── 6. Preseason composite per player ─────────────────────────────────
def compute_preseason_values(profiles: pd.DataFrame,
                             model: LogisticRegression,
                             feature_cols: list) -> pd.DataFrame:
    """
    Dot-product of preseason model stat coefficients × player stats.
    Produces a single preseason_value per player per season.
    """
    stat_coefs = {}
    for feat, coef in zip(feature_cols, model.coef_[0]):
        if feat.startswith("diff_"):
            col = feat[5:]  # strip "diff_"
            if col in ALL_FEATURE_COLS:
                stat_coefs[col] = coef

    profiles = center_metrics_by_position(profiles)

    profiles["preseason_value"] = sum(
        profiles[col] * coef for col, coef in stat_coefs.items()
    )
    return profiles


def extract_preseason_coefficients(model: LogisticRegression,
                                   feature_cols: list) -> dict:
    """Extract learned preseason coefficients keyed by base feature name."""
    coefs = {}
    for feat, coef in zip(feature_cols, model.coef_[0]):
        if feat.startswith("diff_"):
            coefs[feat[5:]] = coef
    return coefs


# ── 7. Combined model features ───────────────────────────────────────
def build_combined_features(games: pd.DataFrame,
                            pvm: pd.DataFrame,
                            skaters: pd.DataFrame,
                            goalies: pd.DataFrame,
                            gamescore: pd.DataFrame,
                            profiles_with_preseason: pd.DataFrame,
                            goalie_metric_col: str = "goalie_gsax_on",
                            skater_ev_xg: pd.DataFrame | None = None,
                            player_game_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Blend preseason composite + in-season per-game stats per player.

    Preseason weight = max(1 - 0.015 * gp_before, 0.25)
    In-season weight = min(0.015 * gp_before, 0.75)
    where gp_before is the player's prior game count this season.
    """
    stat_cols = get_inseason_stat_cols(goalie_metric_col)
    pg = (
        player_game_stats.copy()
        if player_game_stats is not None
        else build_inseason_player_game_stats(games, gamescore, goalies, skater_ev_xg)
    )

    # Cumulative stats BEFORE this game (exclusive cumsum)
    for col in stat_cols:
        pg[f"cum_{col}"] = (
            pg.groupby(["season", "playerid"])[col].cumsum() - pg[col]
        )
    pg["gp_before"] = pg.groupby(["season", "playerid"]).cumcount()  # 0-based

    # Preseason value from previous season
    pg["prev_season"] = pg["season"].apply(prev_season)
    prev_vals = (
        profiles_with_preseason[["season", "playerid", "preseason_value"]]
        .rename(columns={"season": "prev_season"})
    )
    pg = pg.merge(prev_vals, on=["prev_season", "playerid"], how="left")
    pg["preseason_value"] = pg["preseason_value"].fillna(0)

    # Blending weights
    pg["pre_w"], pg["in_w"] = compute_combined_weights(pg["gp_before"])

    # Weighted contributions
    pg["w_preseason"] = pg["preseason_value"] * pg["pre_w"]
    for col in stat_cols:
        per_game = np.where(pg["gp_before"] > 0,
                            pg[f"cum_{col}"] / pg["gp_before"], 0.0)
        pg[f"w_{col}"] = per_game * pg["in_w"]

    # Home / away assignment
    game_teams = games[["game_id", "hometeam", "awayteam"]].rename(
        columns={"game_id": "gameid"})
    pg = pg.merge(game_teams, on="gameid", how="inner")
    pg["side"] = np.where(
        pg["team"] == pg["hometeam"], "home",
        np.where(pg["team"] == pg["awayteam"], "away", "skip"),
    )
    pg = pg[pg["side"] != "skip"]

    # Aggregate per game × side
    w_cols = ["w_preseason"] + [f"w_{c}" for c in stat_cols]
    team_agg = pg.groupby(["gameid", "side"])[w_cols].sum().reset_index()

    home = (
        team_agg[team_agg["side"] == "home"]
        .drop(columns=["side"])
        .rename(columns={c: f"home_{c}" for c in w_cols})
        .rename(columns={"gameid": "game_id"})
    )
    away = (
        team_agg[team_agg["side"] == "away"]
        .drop(columns=["side"])
        .rename(columns={c: f"away_{c}" for c in w_cols})
        .rename(columns={"gameid": "game_id"})
    )

    result_base = games[["game_id", "season", "home_win"]].copy()
    result_base["situation_offset"] = compute_situation_offset(games)
    result = (
        result_base
        .merge(home, on="game_id", how="inner")
        .merge(away, on="game_id", how="inner")
    )
    return result


# ── 8. In-season-only features ───────────────────────────────────────
def build_inseason_features(games: pd.DataFrame,
                            pvm: pd.DataFrame,
                            skaters: pd.DataFrame,
                            goalies: pd.DataFrame,
                            gamescore: pd.DataFrame,
                            goalie_metric_col: str = "goalie_gsax_on",
                            skater_ev_xg: pd.DataFrame | None = None,
                            player_game_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Pure in-season features: per-player cumulative per-game avg
    from earlier games in the SAME season only.
    No preseason data mixed in.
    """
    stat_cols = get_inseason_stat_cols(goalie_metric_col)
    pg = (
        player_game_stats.copy()
        if player_game_stats is not None
        else build_inseason_player_game_stats(games, gamescore, goalies, skater_ev_xg)
    )

    for col in stat_cols:
        pg[f"cum_{col}"] = (
            pg.groupby(["season", "playerid"])[col].cumsum() - pg[col]
        )
    pg["gp_before"] = pg.groupby(["season", "playerid"]).cumcount()  # 0-based

    if INSEASON_WEIGHTED:
        pg["in_w"] = (pg["gp_before"] / GP_THRESHOLD).clip(upper=1)
    else:
        pg["in_w"] = 1.0

    for col in stat_cols:
        per_game = np.where(pg["gp_before"] > 0,
                            pg[f"cum_{col}"] / pg["gp_before"], 0.0)
        pg[f"is_{col}"] = per_game * pg["in_w"]

    game_teams = games[["game_id", "hometeam", "awayteam"]].rename(
        columns={"game_id": "gameid"})
    pg = pg.merge(game_teams, on="gameid", how="inner")
    pg["side"] = np.where(
        pg["team"] == pg["hometeam"], "home",
        np.where(pg["team"] == pg["awayteam"], "away", "skip"),
    )
    pg = pg[pg["side"] != "skip"]

    is_cols = [f"is_{c}" for c in stat_cols]
    team_agg = pg.groupby(["gameid", "side"])[is_cols].sum().reset_index()

    home = (
        team_agg[team_agg["side"] == "home"]
        .drop(columns=["side"])
        .rename(columns={c: f"home_{c}" for c in is_cols})
        .rename(columns={"gameid": "game_id"})
    )
    away = (
        team_agg[team_agg["side"] == "away"]
        .drop(columns=["side"])
        .rename(columns={c: f"away_{c}" for c in is_cols})
        .rename(columns={"gameid": "game_id"})
    )

    result_base = games[["game_id", "season", "home_win"]].copy()
    result_base["situation_offset"] = compute_situation_offset(games)
    result = (
        result_base
        .merge(home, on="game_id", how="inner")
        .merge(away, on="game_id", how="inner")
    )
    return result


def metrics_from_logits(y: pd.Series, logits: np.ndarray) -> dict:
    """Compute log loss and AUC from log-odds values."""
    y_prob = expit(np.asarray(logits, dtype=float))
    y_arr = np.asarray(y, dtype=int)
    return {
        "logloss": log_loss(y_arr, y_prob),
        "auc": roc_auc_score(y_arr, y_prob),
    }


def split_with_validation(X, y, sit_offset, seed=None):
    """Create outer train/test and inner fit/validation splits."""
    X_train, X_test, y_train, y_test, sit_train, sit_test = train_test_split(
        X, y, sit_offset, test_size=0.2, random_state=seed
    )
    val_seed = None if seed is None else seed + 1
    X_fit, X_val, y_fit, y_val, sit_fit, sit_val = train_test_split(
        X_train, y_train, sit_train, test_size=0.25, random_state=val_seed
    )
    return {
        "fit": (X_fit, y_fit, sit_fit),
        "val": (X_val, y_val, sit_val),
        "train": (X_train, y_train, sit_train),
        "test": (X_test, y_test, sit_test),
    }


def print_model_evaluation(label: str,
                           y_train: pd.Series,
                           train_logits: np.ndarray,
                           y_test: pd.Series,
                           test_logits: np.ndarray,
                           y_all: pd.Series):
    """Print train/test metrics and a constant-rate baseline."""
    train_metrics = metrics_from_logits(y_train, train_logits)
    test_metrics = metrics_from_logits(y_test, test_logits)

    print(f"\n=== {label} Evaluation ===")
    print(f"Train  LogLoss: {train_metrics['logloss']:.4f}  "
          f"AUC: {train_metrics['auc']:.4f}")
    print(f"Test   LogLoss: {test_metrics['logloss']:.4f}  "
          f"AUC: {test_metrics['auc']:.4f}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")

    hw_rate = y_all.mean()
    baseline_ll = log_loss(y_all, [hw_rate] * len(y_all))
    print(f"\nBaseline (always predict {hw_rate:.3f}): LogLoss {baseline_ll:.4f}")
    return {
        "train_logloss": train_metrics["logloss"],
        "train_auc": train_metrics["auc"],
        "test_logloss": test_metrics["logloss"],
        "test_auc": test_metrics["auc"],
        "baseline_logloss": baseline_ll,
        "train_size": len(y_train),
        "test_size": len(y_test),
    }


def fit_logistic_with_optional_bounds(X: pd.DataFrame,
                                      y: pd.Series,
                                      lower_bounds: dict[str, float] | None = None) -> np.ndarray:
    """Fit no-intercept logistic regression with optional lower bounds on selected coefficients."""
    model = LogisticRegression(max_iter=5000, random_state=0, fit_intercept=False)
    model.fit(X, y)

    if not lower_bounds:
        return model.coef_[0]

    X_arr = X.values.astype(float)
    y_arr = y.values.astype(float)
    beta0 = model.coef_[0].astype(float)

    bounds = []
    for feature in X.columns:
        lower = lower_bounds.get(feature) if lower_bounds else None
        bounds.append((lower, None) if lower is not None else (None, None))

    def objective(beta):
        logits = X_arr @ beta
        probs = expit(logits)
        eps = 1e-12
        nll = -np.sum(y_arr * np.log(probs + eps) + (1 - y_arr) * np.log(1 - probs + eps))
        grad = X_arr.T @ (probs - y_arr)
        return nll, grad

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=beta0,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Bounded logistic fit failed: {result.message}")
    return result.x


def fit_shrunk_logit_fixed_target(X: np.ndarray,
                                  y: np.ndarray,
                                  offset: np.ndarray,
                                  beta0: np.ndarray,
                                  shrinkage_lambda: float) -> np.ndarray:
    """Fit logistic regression with ridge penalty toward a fixed target coefficient vector."""
    if shrinkage_lambda == 0.0:
        model = LogisticRegression(max_iter=5000, random_state=0, fit_intercept=False)
        model.fit(X, y)
        return model.coef_[0]

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    beta0 = np.asarray(beta0, dtype=float)

    def objective(beta):
        logits = offset + X @ beta
        probs = expit(logits)
        eps = 1e-12
        nll = -np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        diff = beta - beta0
        penalty = 0.5 * shrinkage_lambda * np.sum(diff ** 2)
        grad = X.T @ (probs - y) + shrinkage_lambda * diff
        return nll + penalty, grad

    result = minimize(
        fun=lambda beta: objective(beta)[0],
        x0=beta0,
        jac=lambda beta: objective(beta)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Non-finite shrinkage solution for lambda={shrinkage_lambda}")
    return result.x


def fit_shape_shrunk_combined(X_pre: np.ndarray,
                              X_stats: np.ndarray,
                              y: np.ndarray,
                              offset: np.ndarray,
                              base_vec: np.ndarray,
                              shrinkage_lambda: float):
    """Fit combined model with free preseason scalar and stats shrunk toward a scaled preseason shape."""
    if shrinkage_lambda == 0.0:
        X_full = np.column_stack([X_pre, X_stats])
        model = LogisticRegression(max_iter=5000, random_state=0, fit_intercept=False)
        model.fit(X_full, y)
        return model.coef_[0][0], model.coef_[0][1:], 1.0

    X_pre = np.asarray(X_pre, dtype=float)
    X_stats = np.asarray(X_stats, dtype=float)
    y = np.asarray(y, dtype=float)
    offset = np.asarray(offset, dtype=float)
    base_vec = np.asarray(base_vec, dtype=float)
    x0 = np.concatenate([np.array([1.0]), base_vec, np.array([1.0])])

    def objective(theta):
        preseason_scalar = theta[0]
        beta_stats = theta[1:-1]
        shape_scale = theta[-1]
        logits = offset + preseason_scalar * X_pre + X_stats @ beta_stats
        probs = expit(logits)
        eps = 1e-12
        nll = -np.sum(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps))
        diff = beta_stats - shape_scale * base_vec
        penalty = 0.5 * shrinkage_lambda * np.sum(diff ** 2)
        resid = probs - y
        grad_pre = np.dot(X_pre, resid)
        grad_stats = X_stats.T @ resid + shrinkage_lambda * diff
        grad_scale = -shrinkage_lambda * np.dot(base_vec, diff)
        grad = np.concatenate([np.array([grad_pre]), grad_stats, np.array([grad_scale])])
        return nll + penalty, grad

    result = minimize(
        fun=lambda theta: objective(theta)[0],
        x0=x0,
        jac=lambda theta: objective(theta)[1],
        method="L-BFGS-B",
        options={"maxiter": 2000, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"Non-finite hybrid shrinkage solution for lambda={shrinkage_lambda}")
    return float(result.x[0]), result.x[1:-1], float(result.x[-1])


def tune_fixed_target_lambda(X_fit, y_fit, sit_fit, X_val, y_val, sit_val, beta0):
    """Choose the fixed-target shrinkage lambda by validation log loss."""
    best = None
    for shrinkage_lambda in SHRINKAGE_LAMBDA_GRID:
        beta = fit_shrunk_logit_fixed_target(
            X_fit.values,
            y_fit.values,
            sit_fit.values,
            beta0,
            shrinkage_lambda,
        )
        val_logits = sit_val.values + X_val.values @ beta
        metrics = metrics_from_logits(y_val, val_logits)
        row = {
            "lambda": shrinkage_lambda,
            "beta": beta,
            **metrics,
        }
        if best is None or row["logloss"] < best["logloss"]:
            best = row
    return best


def tune_shape_combined_lambda(X_pre_fit,
                               X_stats_fit,
                               y_fit,
                               sit_fit,
                               X_pre_val,
                               X_stats_val,
                               y_val,
                               sit_val,
                               base_vec):
    """Choose the hybrid combined shrinkage lambda by validation log loss."""
    best = None
    for shrinkage_lambda in SHRINKAGE_LAMBDA_GRID:
        preseason_scalar, beta_stats, shape_scale = fit_shape_shrunk_combined(
            X_pre_fit.values,
            X_stats_fit.values,
            y_fit.values,
            sit_fit.values,
            base_vec,
            shrinkage_lambda,
        )
        val_logits = (
            sit_val.values
            + preseason_scalar * X_pre_val.values
            + X_stats_val.values @ beta_stats
        )
        metrics = metrics_from_logits(y_val, val_logits)
        row = {
            "lambda": shrinkage_lambda,
            "preseason_scalar": preseason_scalar,
            "shape_scale": shape_scale,
            "beta_stats": beta_stats,
            **metrics,
        }
        if best is None or row["logloss"] < best["logloss"]:
            best = row
    return best


def train_inseason_model(features_df: pd.DataFrame,
                         goalie_metric_col: str = "goalie_gsax_on",
                         seed=None):
    """Train an in-season model using gamescore+xG skater features and a goalie metric."""
    stat_cols = get_inseason_stat_cols(goalie_metric_col)
    feature_cols = []

    print("\n=== In-Season Feature Diagnostics ===")
    for stat in stat_cols:
        diff_col = f"diff_is_{stat}"
        features_df[diff_col] = features_df[f"home_is_{stat}"] - features_df[f"away_is_{stat}"]
        vals = features_df[diff_col]
        feature_cols.append(diff_col)
        print(f"  {diff_col:30s}  mean={vals.mean():+.6f}  std={vals.std():.6f}")

    X = features_df[feature_cols].copy()
    y = features_df["home_win"].copy()
    sit_offset = features_df["situation_offset"].copy()
    mask = X.notna().all(axis=1) & y.notna()
    X, y, sit_offset = X[mask], y[mask], sit_offset[mask]

    X_train, X_test, y_train, y_test, sit_train, sit_test = train_test_split(
        X, y, sit_offset, test_size=0.2, random_state=seed
    )

    coef = fit_logistic_with_optional_bounds(
        X_train,
        y_train,
        lower_bounds={f"diff_is_{goalie_metric_col}": 0.0},
    )

    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    print("\n=== In-Season Model Coefficients (learned) ===")
    print(coef_df.to_string(index=False))
    print(f"\n=== Fixed Situation Values ===")
    print(f"  Intercept:          {FIXED_INTERCEPT}")
    for col, val in FIXED_SIT_COEFS.items():
        print(f"  {col:22s} {val:+.4f}")

    train_logits = X_train.values @ coef + sit_train.values
    test_logits = X_test.values @ coef + sit_test.values
    metrics = print_model_evaluation("In-Season Model", y_train, train_logits, y_test, test_logits, y)

    artifact = {
        "mode": "learned_logistic_gamescore",
        "features": coef_df["feature"].tolist(),
        "coefficients": coef_df.to_dict(orient="records"),
        "skater_feature_source": "gamescore_master + skaters_master",
        "goalie_feature_source": "goalies_master",
        "skater_features": INSEASON_SKATER_COLS,
        "goalie_feature": goalie_metric_col,
        "goalie_coefficient_constraint": ">= 0",
        "fixed_intercept": FIXED_INTERCEPT,
        "fixed_sit_coefs": FIXED_SIT_COEFS,
        "metrics": metrics,
    }
    return artifact, coef_df, metrics


# ── 9. Train combined model ──────────────────────────────────────────
def train_combined_model(features_df: pd.DataFrame,
                         goalie_metric_col: str = "goalie_gsax_on",
                         seed=None):
    """Train combined model using preseason value plus gamescore+xG skater features and a goalie metric."""
    stat_cols = get_inseason_stat_cols(goalie_metric_col)
    features_df["diff_w_preseason"] = (
        features_df["home_w_preseason"] - features_df["away_w_preseason"]
    )
    feature_cols = ["diff_w_preseason"]

    print("\n=== Combined Feature Diagnostics ===")
    pre_vals = features_df["diff_w_preseason"]
    print(f"  {'diff_w_preseason':30s}  mean={pre_vals.mean():+.6f}  std={pre_vals.std():.6f}")

    for stat in stat_cols:
        diff_col = f"diff_w_{stat}"
        features_df[diff_col] = features_df[f"home_w_{stat}"] - features_df[f"away_w_{stat}"]
        vals = features_df[diff_col]
        feature_cols.append(diff_col)
        print(f"  {diff_col:30s}  mean={vals.mean():+.6f}  std={vals.std():.6f}")

    X = features_df[feature_cols].copy()
    y = features_df["home_win"].copy()
    sit_offset = features_df["situation_offset"].copy()
    mask = X.notna().all(axis=1) & y.notna()
    X, y, sit_offset = X[mask], y[mask], sit_offset[mask]

    X_train, X_test, y_train, y_test, sit_train, sit_test = train_test_split(
        X, y, sit_offset, test_size=0.2, random_state=seed
    )

    coef = fit_logistic_with_optional_bounds(
        X_train,
        y_train,
        lower_bounds={f"diff_w_{goalie_metric_col}": 0.0},
    )

    coef_df = pd.DataFrame({"feature": feature_cols, "coef": coef})
    print("\n=== Combined Model Coefficients (learned) ===")
    print(coef_df.to_string(index=False))
    print(f"\n=== Fixed Situation Values ===")
    print(f"  Intercept:          {FIXED_INTERCEPT}")
    for col, val in FIXED_SIT_COEFS.items():
        print(f"  {col:22s} {val:+.4f}")

    train_logits = X_train.values @ coef + sit_train.values
    test_logits = X_test.values @ coef + sit_test.values
    metrics = print_model_evaluation("Combined Model", y_train, train_logits, y_test, test_logits, y)

    artifact = {
        "mode": "learned_logistic_gamescore",
        "features": coef_df["feature"].tolist(),
        "coefficients": coef_df.to_dict(orient="records"),
        "skater_feature_source": "gamescore_master + skaters_master",
        "goalie_feature_source": "goalies_master",
        "goalie_feature": goalie_metric_col,
        "goalie_coefficient_constraint": ">= 0",
        "combined_weight_schedule": {
            "slope": COMBINED_WEIGHT_SLOPE,
            "cap_games": COMBINED_WEIGHT_CAP_GAMES,
            "preseason_floor": COMBINED_PRESEASON_FLOOR,
            "inseason_ceiling": COMBINED_INSEASON_CEILING,
        },
        "fixed_intercept": FIXED_INTERCEPT,
        "fixed_sit_coefs": FIXED_SIT_COEFS,
        "metrics": metrics,
    }
    return artifact, coef_df, metrics


def save_model_outputs(pre_model,
                       pre_feat_cols,
                       is_model,
                       is_coef_df: pd.DataFrame,
                       comb_model,
                       comb_coef_df: pd.DataFrame,
                       metric_key: str | None = None,
                       save_preseason: bool = False):
    """Save trained model artifacts with an optional goalie-metric suffix."""
    os.makedirs("Model", exist_ok=True)

    if save_preseason:
        pre_path = os.path.join("Model", "game_projection_preseason.pkl")
        joblib.dump({
            "model": pre_model,
            "features": pre_feat_cols,
            "gp_threshold": GP_THRESHOLD,
            "fixed_intercept": FIXED_INTERCEPT,
            "fixed_sit_coefs": FIXED_SIT_COEFS,
        }, pre_path)
        print(f"\nPreseason model saved to {pre_path}")

    suffix = "" if metric_key is None else f"_{metric_key}"

    is_path = os.path.join("Model", f"game_projection_inseason{suffix}.pkl")
    joblib.dump(is_model, is_path)
    print(f"In-season model saved to {is_path}")

    is_coef_path = os.path.join("Model", f"game_projection_inseason_coefficients{suffix}.csv")
    is_coef_df.to_csv(is_coef_path, index=False)
    print(f"In-season coefficients saved to {is_coef_path}")

    comb_path = os.path.join("Model", f"game_projection_combined{suffix}.pkl")
    joblib.dump(comb_model, comb_path)
    print(f"Combined model saved to {comb_path}")

    comb_coef_path = os.path.join("Model", f"game_projection_combined_coefficients{suffix}.csv")
    comb_coef_df.to_csv(comb_coef_path, index=False)
    print(f"Combined coefficients saved to {comb_coef_path}")


def save_preseason_updating_model_outputs(model: LogisticRegression,
                                          feature_cols: list[str],
                                          coef_df: pd.DataFrame,
                                          metrics: dict):
    """Save the preseason-style updating model artifact and coefficients."""
    os.makedirs("Model", exist_ok=True)

    model_path = os.path.join("Model", "game_projection_preseason_updating.pkl")
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "feature_source": "prior-season totals + current-season cumulative totals before each game",
        "gp_threshold": GP_THRESHOLD,
        "fixed_intercept": FIXED_INTERCEPT,
        "fixed_sit_coefs": FIXED_SIT_COEFS,
        "metrics": metrics,
    }, model_path)
    print(f"Preseason-updating model saved to {model_path}")

    coef_path = os.path.join("Model", "game_projection_preseason_updating_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"Preseason-updating coefficients saved to {coef_path}")


def save_running_model_outputs(model: LogisticRegression,
                               feature_cols: list[str],
                               coef_df: pd.DataFrame,
                               metrics: dict,
                               window_games: int = RECENT_HISTORY_WINDOW,
                               weighting: str = "uniform",
                               exp_half_life: float | None = None,
                               stat_cols: list[str] | None = None):
    """Save a preseason-style running-model artifact and coefficient export."""
    os.makedirs("Model", exist_ok=True)
    variant_key = get_running_model_variant_key(
        window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )
    variant_label = get_running_model_variant_label(
        window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )
    base_name = (
        "game_projection_recent50"
        if variant_key == "recent50"
        else f"game_projection_{variant_key}"
    )
    base_name = f"{base_name}{get_running_stat_subset_suffix(stat_cols)}"

    model_path = os.path.join("Model", f"{base_name}.pkl")
    joblib.dump({
        "model": model,
        "features": feature_cols,
        "window_games": window_games,
        "weighting": weighting,
        "exp_half_life": exp_half_life,
        "feature_source": describe_running_model_source(
            window_games,
            weighting=weighting,
            exp_half_life=exp_half_life,
        ),
        "fixed_intercept": FIXED_INTERCEPT,
        "fixed_sit_coefs": FIXED_SIT_COEFS,
        "metrics": metrics,
    }, model_path)
    print(f"{variant_label} model saved to {model_path}")

    coef_path = os.path.join("Model", f"{base_name}_coefficients.csv")
    coef_df.to_csv(coef_path, index=False)
    print(f"{variant_label} coefficients saved to {coef_path}")


def save_recent_50_model_outputs(model: LogisticRegression,
                                 feature_cols: list[str],
                                 coef_df: pd.DataFrame,
                                 metrics: dict,
                                 window_games: int = RECENT_HISTORY_WINDOW):
    """Backward-compatible wrapper for the uniform recent-50 artifact paths."""
    save_running_model_outputs(
        model,
        feature_cols,
        coef_df,
        metrics,
        window_games=window_games,
        weighting="uniform",
    )


def train_running_model_variant(games: pd.DataFrame,
                                player_game_box_stats: pd.DataFrame,
                                train_seasons: list[str],
                                window_games: int,
                                weighting: str = "uniform",
                                exp_half_life: float | None = None,
                                stat_cols: list[str] | None = None,
                                seed=None):
    """Train one running-history model variant and return its artifacts and metrics."""
    if stat_cols is None:
        stat_cols = STAT_COLS

    variant_key = get_running_model_variant_key(
        window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )
    variant_label = get_running_model_variant_label(
        window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )

    print("\n" + "=" * 60)
    print(f"{variant_label.upper()} MODEL (single preseason-style model)")
    print("=" * 60)
    print(
        f"  NOTE: each player uses up to the latest {window_games} games from the current and previous season before the game date"
    )
    if weighting == "uniform":
        print(
            f"        missing games flow into rookie_F/D/G as ({window_games} - gp_recent) / {window_games}"
        )
    else:
        print(
            f"        observed games use exponential recency weights with half-life {exp_half_life:g}; missing tail weight flows into rookie_F/D/G"
        )

    print("Building running player profiles...")
    running_profiles = build_running_player_profiles(
        player_game_box_stats,
        window_games=window_games,
        weighting=weighting,
        exp_half_life=exp_half_life,
    )
    print(f"  Player-game profile rows: {len(running_profiles)}")

    print("Building running-model features...")
    running_features = build_recent_50_features(
        games,
        running_profiles,
        stat_cols=stat_cols,
    )
    running_features = running_features[
        running_features["season"].isin(train_seasons)
    ].copy()
    running_features = running_features.drop(columns=["season"])
    print(f"  Training on seasons: {train_seasons}")
    print(f"  Feature rows: {len(running_features)}")

    model, feature_cols, coef_df, metrics = train_preseason_model(
        running_features,
        seed=seed,
        model_label=f"{variant_label} Model",
        base_feature_cols=list(stat_cols) + ROOKIE_COLS,
    )
    return {
        "variant_key": variant_key,
        "variant_label": variant_label,
        "window_games": window_games,
        "weighting": weighting,
        "exp_half_life": exp_half_life,
        "stat_cols": list(stat_cols),
        "model": model,
        "feature_cols": feature_cols,
        "coef_df": coef_df,
        "metrics": metrics,
    }


def run_running_model_comparison(games: pd.DataFrame,
                                 player_game_box_stats: pd.DataFrame,
                                 train_seasons: list[str],
                                 window_games_list: list[int],
                                 seed=None,
                                 include_exponential: bool = False,
                                 exp_window: int = RECENT_HISTORY_WINDOW,
                                 stat_cols: list[str] | None = None,
                                 exp_half_life: float = DEFAULT_RUNNING_EXP_HALF_LIFE):
    """Train and compare running-history model variants."""
    if stat_cols is None:
        stat_cols = STAT_COLS

    outputs = {}
    comparison_rows = []

    variants = [
        {
            "window_games": window_games,
            "weighting": "uniform",
            "exp_half_life": None,
        }
        for window_games in sorted(set(window_games_list))
    ]
    if include_exponential:
        variants.append({
            "window_games": exp_window,
            "weighting": "exponential",
            "exp_half_life": exp_half_life,
        })

    for variant in variants:
        output = train_running_model_variant(
            games,
            player_game_box_stats,
            train_seasons,
            window_games=variant["window_games"],
            weighting=variant["weighting"],
            exp_half_life=variant["exp_half_life"],
            stat_cols=stat_cols,
            seed=seed,
        )
        outputs[output["variant_key"]] = output
        comparison_rows.append({
            "variant_key": output["variant_key"],
            "variant_label": output["variant_label"],
            "weighting": output["weighting"],
            "window_games": output["window_games"],
            "exp_half_life": output["exp_half_life"],
            "train_logloss": output["metrics"]["train_logloss"],
            "train_auc": output["metrics"]["train_auc"],
            "test_logloss": output["metrics"]["test_logloss"],
            "test_auc": output["metrics"]["test_auc"],
        })

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(["test_logloss", "test_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    print("\n=== Running Model Comparison ===")
    print(comparison_df.to_string(index=False))
    return comparison_df, outputs


def run_goalie_metric_comparison(games: pd.DataFrame,
                                 pvm: pd.DataFrame,
                                 skaters: pd.DataFrame,
                                 goalies: pd.DataFrame,
                                 gamescore: pd.DataFrame,
                                 skater_ev_xg: pd.DataFrame,
                                 profiles: pd.DataFrame,
                                 train_seasons: list[str],
                                 seed=None):
    """Train in-season and combined models across goalie metric variants."""
    player_game_stats = build_inseason_player_game_stats(games, gamescore, goalies, skater_ev_xg)
    outputs = {}
    comparison_rows = []

    for metric_key, meta in GOALIE_METRICS.items():
        goalie_metric_col = meta["column"]
        print("\n" + "-" * 60)
        print(f"GOALIE METRIC: {metric_key} ({meta['label']})")
        print("-" * 60)

        is_features = build_inseason_features(
            games,
            pvm,
            skaters,
            goalies,
            gamescore,
            skater_ev_xg=skater_ev_xg,
            goalie_metric_col=goalie_metric_col,
            player_game_stats=player_game_stats,
        )
        is_train_seasons = sorted(is_features["season"].unique())
        is_features = is_features[
            is_features["season"].isin(is_train_seasons)
        ].copy()
        is_features = is_features.drop(columns=["season"])
        print(f"  In-season feature rows: {len(is_features)}")
        is_model, is_coef_df, is_metrics = train_inseason_model(
            is_features,
            goalie_metric_col=goalie_metric_col,
            seed=seed,
        )

        comb_features = build_combined_features(
            games,
            pvm,
            skaters,
            goalies,
            gamescore,
            profiles,
            skater_ev_xg=skater_ev_xg,
            goalie_metric_col=goalie_metric_col,
            player_game_stats=player_game_stats,
        )
        comb_features = comb_features[
            comb_features["season"].isin(train_seasons)
        ].copy()
        comb_features = comb_features.drop(columns=["season"])
        print(f"  Combined feature rows: {len(comb_features)}")
        comb_model, comb_coef_df, comb_metrics = train_combined_model(
            comb_features,
            goalie_metric_col=goalie_metric_col,
            seed=seed,
        )

        outputs[metric_key] = {
            "goalie_metric_key": metric_key,
            "goalie_metric_col": goalie_metric_col,
            "goalie_metric_label": meta["label"],
            "inseason_model": is_model,
            "inseason_coef_df": is_coef_df,
            "inseason_metrics": is_metrics,
            "combined_model": comb_model,
            "combined_coef_df": comb_coef_df,
            "combined_metrics": comb_metrics,
        }
        comparison_rows.append({
            "goalie_metric_key": metric_key,
            "goalie_metric_label": meta["label"],
            "goalie_feature": goalie_metric_col,
            "inseason_test_logloss": is_metrics["test_logloss"],
            "inseason_test_auc": is_metrics["test_auc"],
            "combined_test_logloss": comb_metrics["test_logloss"],
            "combined_test_auc": comb_metrics["test_auc"],
        })

    comparison_df = (
        pd.DataFrame(comparison_rows)
        .sort_values(["combined_test_logloss", "combined_test_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )
    print("\n=== Goalie Metric Comparison ===")
    print(comparison_df.to_string(index=False))
    return comparison_df, outputs


# ── Main ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Game Projection Model")
    parser.add_argument("--save", action="store_true",
                        help="Save trained models to Model/")
    parser.add_argument("--refresh-data", action="store_true",
                        help="Force reload from database (otherwise uses cached CSVs)")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for train/test split (default: random)")
    parser.add_argument("--goalie-metric",
                        choices=sorted(GOALIE_METRICS.keys()),
                        default=DEFAULT_GOALIE_METRIC_KEY,
                        help="Goalie metric to use for single-run in-season and combined models")
    parser.add_argument("--compare-goalie-metrics", action="store_true",
                        help="Train and compare all goalie metric variants")
    parser.add_argument("--run-preseason-updating-model", action="store_true",
                        help="Train a preseason-style model that updates prior-season profiles with current-season data before each game")
    parser.add_argument("--run-recent-50-model", action="store_true",
                        help="Train a single preseason-style model using each player's latest 50 previous games from the current and previous season")
    parser.add_argument("--run-running-models", action="store_true",
                        help="Train running-history model variants on configurable window sizes")
    parser.add_argument("--running-windows", nargs="+", type=int,
                        default=DEFAULT_RUNNING_MODEL_WINDOWS,
                        help="Window sizes for uniform running-history models when --run-running-models is used")
    parser.add_argument("--include-exponential-running-model", action="store_true",
                        help="Also train an exponentially weighted running-history model")
    parser.add_argument("--running-exp-window", type=int,
                        default=RECENT_HISTORY_WINDOW,
                        help="Window size for the exponential running-history model")
    parser.add_argument("--running-exp-half-life", type=float,
                        default=DEFAULT_RUNNING_EXP_HALF_LIFE,
                        help="Half-life in games for exponential running-history weighting")
    parser.add_argument("--running-stat-cols", nargs="+",
                        choices=STAT_COLS,
                        default=STAT_COLS,
                        help="Running-history stat columns to include in preseason-style running models")
    args = parser.parse_args()
    seed = args.seed
    print(f"  Random seed: {seed if seed is not None else 'None (random)'}")
    if not args.compare_goalie_metrics:
        print(f"  Goalie metric: {args.goalie_metric} ({get_goalie_metric_label(args.goalie_metric)})")

    use_cached_base = _base_csv_exists() and not args.refresh_data
    use_cached_goalies = use_cached_base and _goalie_cache_is_current()
    use_cached_gamescore = use_cached_base and _gamescore_csv_exists()
    use_cached_skater_ev_xg = use_cached_base and _skater_ev_xg_csv_exists()

    if use_cached_base and use_cached_goalies and use_cached_gamescore and use_cached_skater_ev_xg:
        print("Loading cached data from CSVs...")
        games, pvm, skaters, goalies, gamescore = _load_csvs(include_gamescore=True)
        skater_ev_xg = _load_skater_ev_xg_csv()
    elif use_cached_base:
        print("Loading cached base tables and refreshing only stale inputs...")
        games, pvm, skaters, goalies = _load_csvs(include_gamescore=False)
        need_team_map = not use_cached_gamescore
        need_goalies = not use_cached_goalies
        need_skater_ev_xg = not use_cached_skater_ev_xg

        if use_cached_gamescore:
            gamescore = _load_gamescore_csv()
        else:
            gamescore = None

        if use_cached_skater_ev_xg:
            skater_ev_xg = _load_skater_ev_xg_csv()
        else:
            skater_ev_xg = None

        eng = None
        conn = None
        try:
            if need_goalies or need_team_map or need_skater_ev_xg:
                eng = _get_engine()
                conn = eng.connect()

            if need_goalies:
                print("Refreshing goalies.csv from database...")
                goalies = load_goalies(conn)
                _save_base_csvs(games, pvm, skaters, goalies)

            if need_skater_ev_xg:
                print("Refreshing skater_ev_xg.csv from database...")
                skater_ev_xg = load_skater_ev_xg(conn)
                _save_skater_ev_xg_csv(skater_ev_xg)

            if need_team_map:
                print("Refreshing gamescore.csv from database...")
                _, name_to_abbr = load_team_map(conn)
                gamescore = load_gamescore(conn, name_to_abbr)
                _save_gamescore_csv(gamescore)
        finally:
            if conn is not None:
                conn.close()
    else:
        print("Loading data from Moncton DB...")
        eng = _get_engine()
        with eng.connect() as conn:
            id_to_abbr, name_to_abbr = load_team_map(conn)
            games = load_games(conn, id_to_abbr)
            pvm = load_pvm(conn, name_to_abbr)
            skaters = load_skaters(conn)
            goalies = load_goalies(conn)
            gamescore = load_gamescore(conn, name_to_abbr)
            skater_ev_xg = load_skater_ev_xg(conn)
        _save_csvs(games, pvm, skaters, goalies, gamescore=gamescore)
        _save_skater_ev_xg_csv(skater_ev_xg)

    print(f"  Games: {len(games)}")
    print(f"  PVM rows: {len(pvm)}")
    print(f"  Skater rows: {len(skaters)}")
    print(f"  Goalie rows: {len(goalies)}")
    print(f"  Gamescore rows: {len(gamescore)}")
    print(f"  Skater EV xG rows: {len(skater_ev_xg)}")

    print("Computing back-to-back + situation status...")
    games = add_b2b(games)
    games = add_situation(games)
    sit_stats = games.groupby(["home_b2b", "away_b2b"]).size()
    print(f"  Situation counts:\n{sit_stats.to_string()}")

    print("Building player season profiles...")
    profiles = build_player_profiles(pvm, skaters, goalies)
    print(f"  Player-season profiles: {len(profiles)}")

    # ── Preseason model ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PRESEASON MODEL")
    print("=" * 60)
    print("Building preseason features...")
    pre_features = build_preseason_features(games, pvm, profiles)

    seasons = sorted(pre_features["season"].unique())
    print(f"  Seasons in data: {seasons}")
    train_seasons = seasons[1:]  # skip first (no prev-season data)
    pre_features = pre_features[pre_features["season"].isin(train_seasons)].copy()
    pre_features = pre_features.drop(columns=["season"])
    print(f"  Training on seasons: {train_seasons}")
    print(f"  Feature rows: {len(pre_features)}")

    print("Training preseason model...")
    print(f"  NOTE: preseason features use player profiles from the PRIOR season only")
    print(f"        e.g. game in {train_seasons[0]} uses profiles from {prev_season(train_seasons[0])}")
    print(f"        goalie_gsax uses trailing {GOALIE_PRESEASON_LOOKBACK_SEASONS} seasons in the source profile")
    pre_model, pre_feat_cols, pre_coef_df, pre_metrics = evaluate_fixed_preseason_model(
        pre_features,
        seed=seed,
        model_label="Preseason Model",
    )

    # ── Preseason composite per player ───────────────────────────────
    print("\nComputing per-player preseason values...")
    profiles = compute_preseason_values(profiles, pre_model, pre_feat_cols)
    print(f"  Preseason value range: [{profiles['preseason_value'].min():.4f}, "
          f"{profiles['preseason_value'].max():.4f}]")

    if args.run_preseason_updating_model:
        print("\n" + "=" * 60)
        print("PRESEASON-UPDATING MODEL")
        print("=" * 60)
        print("Building preseason-style updating features...")
        print("  NOTE: player profiles use prior-season totals plus current-season cumulative totals before each game")
        player_game_box_stats = build_player_game_box_stats(pvm, skaters, goalies, games)
        preseason_updating_features = build_preseason_updating_features(games, player_game_box_stats)
        preseason_updating_features = preseason_updating_features[
            preseason_updating_features["season"].isin(train_seasons)
        ].copy()
        preseason_updating_features = preseason_updating_features.drop(columns=["season"])
        print(f"  Training on seasons: {train_seasons}")
        print(f"  Feature rows: {len(preseason_updating_features)}")
        print(
            f"  NOTE: goalie_gsax uses the prior {GOALIE_PRESEASON_LOOKBACK_SEASONS} seasons plus current-season cumulative totals before the game"
        )

        preseason_updating_model, preseason_updating_feat_cols, preseason_updating_coef_df, preseason_updating_metrics = evaluate_fixed_preseason_model(
            preseason_updating_features,
            seed=seed,
            model_label="Preseason-Updating Model (fixed preseason coefficients)",
        )

        if args.save:
            save_preseason_updating_model_outputs(
                preseason_updating_model,
                preseason_updating_feat_cols,
                preseason_updating_coef_df,
                preseason_updating_metrics,
            )

    # ── In-season-only model ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("IN-SEASON MODEL (pure in-season stats, no preseason)")
    print("=" * 60)
    print("Building in-season and combined features...")
    print(f"  NOTE: skaters use raw per-game gamescore buckets so higher TOI still carries through the totals")
    print(f"        skater features are EV = ev_def + ev_off, PP = pp_off, SH = sh_def, EV xG = xgf - xga")
    print(f"        combined still uses pre_w = max(1 - 0.015 * gp_before, 0.25)")
    print(f"        combined still uses in_w  = min(0.015 * gp_before, 0.75)")

    if args.compare_goalie_metrics:
        comparison_df, metric_outputs = run_goalie_metric_comparison(
            games,
            pvm,
            skaters,
            goalies,
            gamescore,
            skater_ev_xg,
            profiles,
            train_seasons,
            seed=seed,
        )

        if args.save:
            default_outputs = metric_outputs[DEFAULT_GOALIE_METRIC_KEY]
            save_model_outputs(
                pre_model,
                pre_feat_cols,
                default_outputs["inseason_model"],
                default_outputs["inseason_coef_df"],
                default_outputs["combined_model"],
                default_outputs["combined_coef_df"],
                metric_key=None,
                save_preseason=True,
            )

            comparison_path = os.path.join("Model", "game_projection_goalie_metric_comparison.csv")
            comparison_df.to_csv(comparison_path, index=False)
            print(f"Goalie metric comparison saved to {comparison_path}")

            for metric_key, outputs in metric_outputs.items():
                save_model_outputs(
                    pre_model,
                    pre_feat_cols,
                    outputs["inseason_model"],
                    outputs["inseason_coef_df"],
                    outputs["combined_model"],
                    outputs["combined_coef_df"],
                    metric_key=metric_key,
                    save_preseason=False,
                )
    else:
        goalie_metric_col = get_goalie_metric_column(args.goalie_metric)
        player_game_stats = build_inseason_player_game_stats(games, gamescore, goalies, skater_ev_xg)

        is_features = build_inseason_features(
            games,
            pvm,
            skaters,
            goalies,
            gamescore,
            goalie_metric_col=goalie_metric_col,
            skater_ev_xg=skater_ev_xg,
            player_game_stats=player_game_stats,
        )
        is_train_seasons = sorted(is_features["season"].unique())
        is_features = is_features[
            is_features["season"].isin(is_train_seasons)
        ].copy()
        is_features = is_features.drop(columns=["season"])
        print(f"  In-season training seasons: {is_train_seasons}")
        print(f"  In-season feature rows: {len(is_features)}")

        print(f"Training in-season model with goalie metric {args.goalie_metric} ({get_goalie_metric_label(args.goalie_metric)})...")
        is_model, is_coef_df, _ = train_inseason_model(
            is_features,
            goalie_metric_col=goalie_metric_col,
            seed=seed,
        )

        print("\n" + "=" * 60)
        print("COMBINED MODEL (preseason + in-season)")
        print("=" * 60)
        print("Building combined features (this may take a moment)...")
        print(f"  NOTE: preseason_value = prior season profile (independent of current season)")
        comb_features = build_combined_features(
            games,
            pvm,
            skaters,
            goalies,
            gamescore,
            profiles,
            goalie_metric_col=goalie_metric_col,
            skater_ev_xg=skater_ev_xg,
            player_game_stats=player_game_stats,
        )
        comb_features = comb_features[
            comb_features["season"].isin(train_seasons)
        ].copy()
        comb_features = comb_features.drop(columns=["season"])
        print(f"  Combined feature rows: {len(comb_features)}")

        print(f"Training combined model with goalie metric {args.goalie_metric} ({get_goalie_metric_label(args.goalie_metric)})...")
        comb_model, comb_coef_df, _ = train_combined_model(
            comb_features,
            goalie_metric_col=goalie_metric_col,
            seed=seed,
        )

        if args.save:
            save_model_outputs(
                pre_model,
                pre_feat_cols,
                is_model,
                is_coef_df,
                comb_model,
                comb_coef_df,
                metric_key=None,
                save_preseason=True,
            )
            save_model_outputs(
                pre_model,
                pre_feat_cols,
                is_model,
                is_coef_df,
                comb_model,
                comb_coef_df,
                metric_key=args.goalie_metric,
                save_preseason=False,
            )

    if args.run_recent_50_model or args.run_running_models:
        print("\n" + "=" * 60)
        print("RUNNING HISTORY MODELS")
        print("=" * 60)
        print("Building rolling player-game stats...")
        player_game_box_stats = build_player_game_box_stats(pvm, skaters, goalies, games)
        print(f"  Player-game box rows: {len(player_game_box_stats)}")

        running_windows = []
        if args.run_recent_50_model:
            running_windows.append(RECENT_HISTORY_WINDOW)
        if args.run_running_models:
            running_windows.extend(args.running_windows)
        running_windows = sorted(set(running_windows))

        running_comparison_df, running_outputs = run_running_model_comparison(
            games,
            player_game_box_stats,
            train_seasons,
            window_games_list=running_windows,
            seed=seed,
            include_exponential=(
                args.run_running_models and args.include_exponential_running_model
            ),
            exp_window=args.running_exp_window,
            stat_cols=args.running_stat_cols,
            exp_half_life=args.running_exp_half_life,
        )

        if args.save:
            for output in running_outputs.values():
                save_running_model_outputs(
                    output["model"],
                    output["feature_cols"],
                    output["coef_df"],
                    output["metrics"],
                    window_games=output["window_games"],
                    weighting=output["weighting"],
                    exp_half_life=output["exp_half_life"],
                    stat_cols=output.get("stat_cols"),
                )

            if args.run_running_models or len(running_outputs) > 1:
                running_comparison_path = os.path.join(
                    "Model",
                    "game_projection_running_model_comparison.csv",
                )
                running_comparison_df.to_csv(running_comparison_path, index=False)
                print(f"Running model comparison saved to {running_comparison_path}")


if __name__ == "__main__":
    main()
