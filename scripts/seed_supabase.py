"""
Seed Supabase with static reference data that currently lives in CSV files.

Usage:
    python scripts/seed_supabase.py

Ensures the following tables are populated:
    - teams              (from Teams.csv)
    - last_dates         (from Last_date.csv)
    - box_ids            (from BoxID.csv)
    - season_stats       (from nhl_seasonstats.csv)
    - season_stats_teams (from nhl_seasonstats_teams.csv)
    - player_projections (from player_projections.csv)
    - rapm / rapm_context(from rapm/*.csv)

Requires SUPABASE_URL + SUPABASE_SERVICE_KEY env vars.
"""

import os, sys, pathlib, json

# Allow importing from project root
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import pandas as pd
from app.supabase_client import upsert_df


def seed_teams():
    path = ROOT / "Teams.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Team": "team",
        "TeamID": "team_id",
        "Name": "name",
        "Logo": "logo",
        "Color": "color",
        "Active": "active",
    })
    df["active"] = df["active"].astype(bool)
    upsert_df("teams", df, on_conflict="team")
    print(f"  teams: {len(df)} rows")


def seed_last_dates():
    path = ROOT / "Last_date.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Season": "season",
        "Last_Date": "last_date",
    })
    upsert_df("last_dates", df, on_conflict="season")
    print(f"  last_dates: {len(df)} rows")


def seed_box_ids():
    path = ROOT / "BoxID.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "x": "x",
        "y": "y",
        "BoxID": "box_id",
        "BoxID_rev": "box_id_rev",
        "Boxsize": "box_size",
    })
    upsert_df("box_ids", df, on_conflict="x,y")
    print(f"  box_ids: {len(df)} rows")


def seed_season_stats():
    path = ROOT / "app" / "static" / "nhl_seasonstats.csv"
    if not path.exists():
        print("  season_stats: skipped (file not found)")
        return
    df = pd.read_csv(path)
    col_map = {
        "Season": "season", "SeasonState": "season_state",
        "StrengthState": "strength_state", "PlayerID": "player_id",
        "Position": "position", "GP": "gp", "plusMinus": "plus_minus",
        "blockedShots": "blocked_shots", "TOI": "toi",
        "iGoals": "i_goals", "Assists1": "assists1", "Assists2": "assists2",
        "iCorsi": "i_corsi", "iFenwick": "i_fenwick", "iShots": "i_shots",
        "ixG_F": "ixg_f", "ixG_S": "ixg_s", "ixG_F2": "ixg_f2",
        "PIM_taken": "pim_taken", "PIM_drawn": "pim_drawn",
        "Hits": "hits", "Takeaways": "takeaways", "Giveaways": "giveaways",
        "SO_Goal": "so_goal", "SO_Attempt": "so_attempt",
        "CA": "ca", "CF": "cf", "FA": "fa", "FF": "ff",
        "SA": "sa", "SF": "sf", "GA": "ga", "GF": "gf",
        "xGA_F": "xga_f", "xGF_F": "xgf_f",
        "xGA_S": "xga_s", "xGF_S": "xgf_s",
        "xGA_F2": "xga_f2", "xGF_F2": "xgf_f2",
        "PIM_for": "pim_for", "PIM_against": "pim_against",
    }
    df = df.rename(columns=col_map)
    # Convert float columns that should be int (NaN → None handled by upsert_df)
    int_cols = ["gp", "plus_minus", "blocked_shots", "i_goals", "assists1", "assists2",
                "i_corsi", "i_fenwick", "i_shots", "hits", "takeaways", "giveaways",
                "so_goal", "so_attempt", "ca", "cf", "fa", "ff", "sa", "sf", "ga", "gf",
                "pim_against"]
    for c in int_cols:
        if c in df.columns:
            df[c] = df[c].astype("Int64")  # nullable integer
    # Deduplicate on PK (keep last occurrence)
    df = df.drop_duplicates(subset=["season", "season_state", "strength_state", "player_id"], keep="last")
    upsert_df("season_stats", df, on_conflict="season,season_state,strength_state,player_id")
    print(f"  season_stats: {len(df)} rows")


def seed_season_stats_teams():
    path = ROOT / "app" / "static" / "nhl_seasonstats_teams.csv"
    if not path.exists():
        print("  season_stats_teams: skipped (file not found)")
        return
    df = pd.read_csv(path)
    col_map = {
        "Season": "season", "SeasonState": "season_state",
        "StrengthState": "strength_state", "Team": "team",
        "GP": "gp", "TOI": "toi",
        "CF": "cf", "CA": "ca", "FF": "ff", "FA": "fa",
        "SF": "sf", "SA": "sa", "GF": "gf", "GA": "ga",
        "xGF_F": "xgf_f", "xGA_F": "xga_f",
        "xGF_S": "xgf_s", "xGA_S": "xga_s",
        "xGF_F2": "xgf_f2", "xGA_F2": "xga_f2",
        "PIM_for": "pim_for", "PIM_against": "pim_against",
    }
    df = df.rename(columns=col_map)
    upsert_df("season_stats_teams", df, on_conflict="season,season_state,strength_state,team")
    print(f"  season_stats_teams: {len(df)} rows")


def seed_player_projections():
    path = ROOT / "app" / "static" / "player_projections.csv"
    if not path.exists():
        print("  player_projections: skipped (file not found)")
        return
    df = pd.read_csv(path)
    col_map = {
        "PlayerID": "player_id", "Position": "position",
        "Game_No": "game_no", "Age": "age", "Rookie": "rookie",
        "EVO": "evo", "EVD": "evd", "PP": "pp", "SH": "sh", "GSAx": "gsax",
    }
    df = df.rename(columns=col_map)
    upsert_df("player_projections", df, on_conflict="player_id")
    print(f"  player_projections: {len(df)} rows")


def seed_rapm():
    path = ROOT / "app" / "static" / "rapm" / "rapm.csv"
    if not path.exists():
        print("  rapm: skipped (file not found)")
        return
    df = pd.read_csv(path)
    # Core columns that map directly
    core = {
        "PlayerID": "player_id", "Season": "season",
        "StrengthState": "strength_state", "Rates_Totals": "rates_totals",
        "CF": "cf", "CA": "ca", "GF": "gf", "GA": "ga",
        "xGF": "xgf", "xGA": "xga",
        "PEN_taken": "pen_taken", "PEN_drawn": "pen_drawn",
        "C_plusminus": "c_plusminus", "G_plusminus": "g_plusminus",
        "xG_plusminus": "xg_plusminus", "PEN_plusminus": "pen_plusminus",
        "Alpha_CF": "alpha_cf", "Alpha_GF": "alpha_gf",
        "Alpha_xGF": "alpha_xgf", "Alpha_PEN": "alpha_pen",
    }
    import json
    out_rows = []
    for _, row in df.iterrows():
        r = {v: row.get(k) for k, v in core.items() if k in row.index}
        # Pack stddev / zscore / pp_sh into JSONB dicts
        stddev = {c: row[c] for c in df.columns if '_stddev' in c.lower()}
        zscore = {c: row[c] for c in df.columns if '_zscore' in c.lower()}
        pp_sh  = {c: row[c] for c in df.columns if c.startswith('PP_') or c.startswith('SH_')}
        r["stddev"] = {k: None if pd.isna(v) else v for k, v in stddev.items()}
        r["zscore"] = {k: None if pd.isna(v) else v for k, v in zscore.items()}
        r["pp_sh"]  = {k: None if pd.isna(v) else v for k, v in pp_sh.items()}
        out_rows.append(r)
    out = pd.DataFrame(out_rows)
    out = out.drop_duplicates(subset=["player_id", "season", "strength_state"], keep="last")
    upsert_df("rapm", out, on_conflict="player_id,season,strength_state")
    print(f"  rapm: {len(out)} rows")


def seed_rapm_context():
    path = ROOT / "app" / "static" / "rapm" / "context.csv"
    if not path.exists():
        print("  rapm_context: skipped (file not found)")
        return
    df = pd.read_csv(path)
    col_map = {
        "PlayerID": "player_id", "Season": "season",
        "StrengthState": "strength_state", "Minutes": "minutes",
        "QoT_blend_xG67_G33": "qot_blend_xg67_g33",
        "QoC_blend_xG67_G33": "qoc_blend_xg67_g33",
        "ZS_Difficulty": "zs_difficulty",
    }
    df = df.rename(columns=col_map)
    upsert_df("rapm_context", df, on_conflict="player_id,season,strength_state")
    print(f"  rapm_context: {len(df)} rows")


def main():
    print("Seeding Supabase …")
    seed_teams()
    seed_last_dates()
    seed_box_ids()
    seed_season_stats()
    seed_season_stats_teams()
    seed_player_projections()
    seed_rapm()
    seed_rapm_context()
    print("Done.")


if __name__ == "__main__":
    main()
