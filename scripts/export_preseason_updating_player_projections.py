"""Export preseason-updating pregame player projections to Hockey-Statistics.

The export scores the fixed preseason coefficient set on player profiles built
from prior-season totals plus current-season cumulative totals before each game,
matches Moncton player IDs to Hockey-Statistics IDs, and upserts the matched
rows into player_game_projections.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
import unicodedata

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

from app.supabase_client import read_table, upsert_df
from scripts.Game_Projection_Model import (
    _base_csv_exists,
    _get_engine,
    _load_csvs,
    _save_base_csvs,
    GP_THRESHOLD,
    GOALIE_PRESEASON_LOOKBACK_SEASONS,
    ROOKIE_COLS,
    STAT_COLS,
    build_player_game_box_stats,
    build_preseason_updating_player_profiles,
    center_metrics_by_position,
    load_games,
    load_goalies,
    load_pvm,
    load_skaters,
    load_team_map,
    prev_season,
    season_years_ago,
)


DEFAULT_TABLE = "player_game_projections"
CURRENT_TABLE = "player_current_projections"
DEFAULT_MODEL_PATH = ROOT / "Model" / "game_projection_preseason.pkl"
DEFAULT_MODEL_KEY = "preseason_updating"
EXCLUDED_EXPORT_SEASONS = {"20222023", "20232024"}
MONCTON_PLAYER_CACHE = ROOT / "data" / "game_projection" / "moncton_players.csv"
MIGRATION_PATHS = [
    ROOT / "supabase" / "migrations" / "005_create_player_game_projections.sql",
    ROOT / "supabase" / "migrations" / "006_add_source_player_id_to_player_game_projections.sql",
    ROOT / "supabase" / "migrations" / "007_add_raw_metrics_to_player_game_projections.sql",
]
CURRENT_MIGRATION_PATHS = [
    ROOT / "supabase" / "migrations" / "008_create_player_current_projections.sql",
]

NAME_NORMALIZATION_ALIASES = {
    "yegor chinakhov": "egor chinakhov",
    "nikita grebyonkin": "nikita grebenkin",
    "mitch marner": "mitchell marner",
    "chris tanev": "christopher tanev",
    "mathew dumba": "matt dumba",
    "arsenii sergeev": "arseni sergeev",
    "arseny gritsyuk": "arseni gritsyuk",
    "bo groulx": "benoit olivier groulx",
    "danil zhilkin": "danny zhilkin",
    "josh samanski": "joshua samanski",
    "max shabanov": "maxim shabanov",
    "mike benning": "michael benning",
    "samuel blais": "sammy blais",
    "viking gustafsson nyberg": "viking gustavsson nyberg",
    "matt coronato": "matthew coronato",
    "matt rempe": "matthew rempe",
    "matt savoie": "matthew savoie",
    "j j moser": "janis jerome moser",
    "bradly nadeau": "bradley nadeau",
}
PROFILE_METRIC_COLS = STAT_COLS + ROOKIE_COLS


def coerce_nullable_int_ids(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def normalize_person_name(value: object) -> str:
    text_value = str(value or "").strip()
    if not text_value:
        return ""
    text_value = unicodedata.normalize("NFKD", text_value)
    text_value = text_value.encode("ascii", "ignore").decode("ascii")
    text_value = text_value.lower().replace("'", "")
    text_value = re.sub(r"[^a-z0-9]+", " ", text_value)
    text_value = re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", text_value)
    text_value = " ".join(text_value.split())
    return NAME_NORMALIZATION_ALIASES.get(text_value, text_value)


def normalize_team(value: object) -> str:
    team = re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())
    aliases = {
        "utahhockeyclub": "uta",
        "utah": "uta",
        "arizona": "ari",
    }
    return aliases.get(team, team)


def normalize_position_group(value: object) -> str:
    position = str(value or "").strip().upper()
    if position in {"L", "R", "C", "F"}:
        return "F"
    if position.startswith("D"):
        return "D"
    if position.startswith("G"):
        return "G"
    return position


def unique_or_none(series: pd.Series):
    values = pd.Series(series).dropna().unique()
    if len(values) == 1:
        return values[0]
    return None


def build_initial_last_key(value: object) -> str:
    normalized = normalize_person_name(value)
    if not normalized:
        return ""
    parts = normalized.split()
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0][0]} {' '.join(parts[1:])}"


def build_last_name_key(value: object) -> str:
    normalized = normalize_person_name(value)
    if not normalized:
        return ""
    return normalized.split()[-1]


def load_base_inputs(refresh_data: bool = False):
    if _base_csv_exists() and not refresh_data:
        games, pvm, skaters, goalies = _load_csvs(include_gamescore=False)
        return games, pvm, skaters, goalies

    print("Loading projection inputs from Moncton DB...")
    engine = _get_engine()
    with engine.connect() as conn:
        id_to_abbr, name_to_abbr = load_team_map(conn)
        games = load_games(conn, id_to_abbr)
        pvm = load_pvm(conn, name_to_abbr)
        skaters = load_skaters(conn)
        goalies = load_goalies(conn)
    _save_base_csvs(games, pvm, skaters, goalies)
    return games, pvm, skaters, goalies


def load_moncton_players(refresh_cache: bool = False) -> pd.DataFrame:
    if MONCTON_PLAYER_CACHE.exists() and not refresh_cache:
        players = pd.read_csv(MONCTON_PLAYER_CACHE)
    else:
        print("Loading Moncton player names...")
        engine = _get_engine()
        with engine.connect() as conn:
            players = pd.read_sql(
                text(
                    """
                    select
                        player_id as playerid,
                        trim(concat(coalesce(first_name, ''), ' ', coalesce(last_name, ''))) as source_player_name
                    from players
                    """
                ),
                conn,
            )
        MONCTON_PLAYER_CACHE.parent.mkdir(parents=True, exist_ok=True)
        players.to_csv(MONCTON_PLAYER_CACHE, index=False)

    players["playerid"] = coerce_nullable_int_ids(players["playerid"])
    players = players.dropna(subset=["playerid"]).copy()
    players["source_player_name"] = players["source_player_name"].fillna("").astype(str).str.strip()
    players = players.drop_duplicates(subset=["playerid"], keep="last")
    return players


def load_projection_artifact(model_path: pathlib.Path):
    artifact = joblib.load(model_path)
    model = artifact.get("model")
    feature_cols = artifact.get("features") or []
    if model is None or not feature_cols:
        raise ValueError("Artifact is missing model/features")

    coef_map = {}
    for feature, coef in zip(feature_cols, model.coef_[0]):
        if not feature.startswith("diff_"):
            raise ValueError(f"Unexpected feature name: {feature}")
        coef_map[feature[5:]] = float(coef)

    if "window_games" not in artifact:
        artifact["window_games"] = int(artifact.get("gp_threshold", 41))
    if "weighting" not in artifact:
        artifact["weighting"] = DEFAULT_MODEL_KEY

    return artifact, coef_map


def compute_player_projections(
    games: pd.DataFrame,
    pvm: pd.DataFrame,
    skaters: pd.DataFrame,
    goalies: pd.DataFrame,
    moncton_players: pd.DataFrame,
    artifact: dict,
    coef_map: dict[str, float],
) -> pd.DataFrame:
    print("Building preseason-updating player profiles...")
    player_game_stats = build_player_game_box_stats(pvm, skaters, goalies, games)
    raw_profiles = build_preseason_updating_player_profiles(player_game_stats)
    centered_profiles = center_metrics_by_position(raw_profiles)

    profiles = raw_profiles.copy()
    profiles["playerid"] = coerce_nullable_int_ids(profiles["playerid"])
    for col in PROFILE_METRIC_COLS:
        profiles[f"raw_{col}"] = raw_profiles[col]
        profiles[col] = centered_profiles[col]

    profiles = profiles.merge(
        games[["game_id", "season", "date", "hometeam", "awayteam"]].rename(columns={"game_id": "gameid"}),
        on=["season", "gameid", "date"],
        how="left",
    )
    profiles["side"] = np.where(
        profiles["team"] == profiles["hometeam"],
        "home",
        np.where(profiles["team"] == profiles["awayteam"], "away", None),
    )
    profiles = profiles[profiles["side"].notna()].copy()
    profiles["opponent"] = np.where(
        profiles["side"] == "home",
        profiles["awayteam"],
        profiles["hometeam"],
    )
    profiles["raw_projected_value"] = 0.0
    profiles["projected_value"] = 0.0
    for feature_name, coef in coef_map.items():
        if feature_name not in profiles.columns:
            raise KeyError(f"Running profile is missing expected feature {feature_name}")
        raw_feature_name = f"raw_{feature_name}"
        if raw_feature_name not in profiles.columns:
            raise KeyError(f"Running profile is missing expected feature {raw_feature_name}")
        profiles["raw_projected_value"] += profiles[raw_feature_name].fillna(0.0) * coef
        profiles["projected_value"] += profiles[feature_name].fillna(0.0) * coef

    projections = profiles.merge(moncton_players.copy(), on="playerid", how="left")
    projections["source_player_name"] = projections["source_player_name"].fillna("")
    projections["season"] = projections["season"].astype(str)
    return projections


def load_hockey_statistics_lookup(seasons: list[str]) -> pd.DataFrame:
    season_filter = f"in.({','.join(sorted(seasons))})"
    print(f"Loading Hockey-Statistics player lookup for seasons {', '.join(sorted(seasons))}...")
    game_data = read_table(
        "game_data",
        columns="season,game_id,date,player_id,player,position,team",
        filters={"season": season_filter},
    )
    players = read_table(
        "players",
        columns="season,player_id,player",
        filters={"season": season_filter},
    )
    if game_data.empty:
        raise RuntimeError("No Hockey-Statistics game_data rows found for the selected seasons")
    if players.empty:
        raise RuntimeError("No Hockey-Statistics players rows found for the selected seasons")

    game_data["season"] = game_data["season"].astype(str)
    players["season"] = players["season"].astype(str)
    players = players.rename(columns={"player": "players_table_name"})
    lookup = game_data.merge(players, on=["season", "player_id"], how="left")
    lookup["canonical_player_name"] = lookup["players_table_name"].fillna(lookup["player"])
    lookup["team_norm"] = lookup["team"].apply(normalize_team)
    lookup["position_group"] = lookup["position"].apply(normalize_position_group)
    lookup["season"] = lookup["season"].astype(str)
    lookup["game_id"] = pd.to_numeric(lookup["game_id"], errors="coerce").astype("Int64")
    lookup["game_date"] = pd.to_datetime(lookup["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    candidate_rows = []
    for row in lookup.itertuples(index=False):
        candidate_names = {
            normalize_person_name(row.canonical_player_name),
            normalize_person_name(row.player),
        }
        candidate_names.discard("")
        for candidate_name in candidate_names:
            candidate_rows.append(
                {
                    "season": row.season,
                    "game_id": int(row.game_id),
                    "game_date": row.game_date,
                    "player_id": row.player_id,
                    "player": row.canonical_player_name,
                    "team": row.team,
                    "team_norm": row.team_norm,
                    "position_group": row.position_group,
                    "player_name_norm": candidate_name,
                }
            )

    candidate_df = pd.DataFrame(candidate_rows)
    if candidate_df.empty:
        raise RuntimeError("No Hockey-Statistics lookup candidates were built")
    return candidate_df


def load_hockey_statistics_players_lookup(seasons: list[str]) -> pd.DataFrame:
    season_filter = f"in.({','.join(sorted(seasons))})"
    players = read_table(
        "players",
        columns="season,player_id,player",
        filters={"season": season_filter},
    )
    if players.empty:
        raise RuntimeError("No Hockey-Statistics players rows found for the selected seasons")

    players["season"] = players["season"].astype(str)
    players["player_name_norm"] = players["player"].apply(normalize_person_name)
    players["initial_last_key"] = players["player"].apply(build_initial_last_key)
    players["last_name_key"] = players["player"].apply(build_last_name_key)
    return players


def build_unique_lookup(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = (
        df.groupby(keys, as_index=False)
        .agg(
            game_id=("game_id", unique_or_none),
            player_id=("player_id", unique_or_none),
            player=("player", unique_or_none),
            team=("team", unique_or_none),
        )
    )
    return grouped[grouped["player_id"].notna()].copy()


def apply_lookup_matches(
    matched: pd.DataFrame,
    lookup_df: pd.DataFrame,
    keys: list[str],
    match_type: str,
) -> None:
    unresolved = matched[matched["player_id"].isna()].copy()
    if unresolved.empty:
        return

    source = unresolved[keys].copy().reset_index()
    source = source.merge(lookup_df, on=keys, how="left")
    source = source[source["player_id"].notna()].copy()
    if source.empty:
        return

    if "game_id" in source.columns:
        matched.loc[source["index"], "game_id"] = source["game_id"].values
    matched.loc[source["index"], "player_id"] = source["player_id"].values
    if "player" in source.columns:
        matched.loc[source["index"], "player"] = source["player"].values
    matched.loc[source["index"], "match_type"] = match_type


def apply_unique_roster_slot_matches(matched: pd.DataFrame, hs_lookup: pd.DataFrame) -> None:
    unresolved = matched[matched["player_id"].isna()].copy()
    if unresolved.empty:
        return

    resolved_pairs = matched[matched["player_id"].notna()][["game_id", "player_id"]].drop_duplicates()
    hs_remaining = hs_lookup[
        ["season", "game_date", "team_norm", "position_group", "game_id", "player_id", "player"]
    ].drop_duplicates()
    hs_remaining = hs_remaining.merge(
        resolved_pairs,
        on=["game_id", "player_id"],
        how="left",
        indicator=True,
    )
    hs_remaining = hs_remaining[hs_remaining["_merge"] == "left_only"].drop(columns=["_merge"])

    group_keys = ["season", "game_date", "team_norm", "position_group"]
    source_counts = unresolved.groupby(group_keys).size().rename("source_count")
    hs_counts = hs_remaining.groupby(group_keys).size().rename("hs_count")
    eligible_groups = source_counts.to_frame().join(hs_counts, how="left").fillna(0).reset_index()
    eligible_groups = eligible_groups[
        (eligible_groups["source_count"] == 1) & (eligible_groups["hs_count"] == 1)
    ].copy()
    if eligible_groups.empty:
        return

    source = unresolved[group_keys].copy().reset_index().merge(eligible_groups[group_keys], on=group_keys, how="inner")
    source = source.merge(hs_remaining, on=group_keys, how="inner")
    if source.empty:
        return

    matched.loc[source["index"], "game_id"] = source["game_id"].values
    matched.loc[source["index"], "player_id"] = source["player_id"].values
    matched.loc[source["index"], "player"] = source["player"].values
    matched.loc[source["index"], "match_type"] = "unique_roster_slot"


def match_player_ids(projections: pd.DataFrame, hs_lookup: pd.DataFrame):
    matched = projections.copy()
    matched["source_game_id"] = pd.to_numeric(matched["gameid"], errors="coerce").astype("Int64")
    matched["game_date"] = pd.to_datetime(matched["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    matched["team_norm"] = matched["team"].apply(normalize_team)
    matched["position_group"] = matched["position"].apply(normalize_position_group)
    matched["player_name_norm"] = matched["source_player_name"].apply(normalize_person_name)
    matched["initial_last_key"] = matched["source_player_name"].apply(build_initial_last_key)
    matched["last_name_key"] = matched["source_player_name"].apply(build_last_name_key)

    hs_candidates = hs_lookup.copy()
    hs_candidates["initial_last_key"] = hs_candidates["player"].apply(build_initial_last_key)
    hs_candidates["last_name_key"] = hs_candidates["player"].apply(build_last_name_key)

    exact_keys = ["season", "game_date", "team_norm", "player_name_norm", "position_group"]
    exact_lookup = build_unique_lookup(hs_candidates, exact_keys)

    matched = matched.merge(exact_lookup, on=exact_keys, how="left", suffixes=("", "_matched"))
    matched["match_type"] = np.where(
        matched["player_id"].notna(),
        "exact_name_team_position",
        None,
    )

    initial_last_keys = ["season", "game_date", "team_norm", "initial_last_key", "position_group"]
    initial_last_lookup = build_unique_lookup(hs_candidates, initial_last_keys)
    apply_lookup_matches(matched, initial_last_lookup, initial_last_keys, "initial_last_team_position")

    apply_unique_roster_slot_matches(matched, hs_candidates)

    matched["matched_player_name"] = matched["player"].fillna("")
    unmatched = matched[matched["player_id"].isna()].copy()
    return matched, unmatched


def apply_players_table_matches(matched: pd.DataFrame, players_lookup: pd.DataFrame) -> None:
    exact_lookup = (
        players_lookup.groupby(["season", "player_name_norm"], as_index=False)
        .agg(player_id=("player_id", unique_or_none), player=("player", unique_or_none))
    )
    exact_lookup = exact_lookup[exact_lookup["player_id"].notna()].copy()
    apply_lookup_matches(matched, exact_lookup, ["season", "player_name_norm"], "season_exact_name")

    initial_lookup = (
        players_lookup.groupby(["season", "initial_last_key"], as_index=False)
        .agg(player_id=("player_id", unique_or_none), player=("player", unique_or_none))
    )
    initial_lookup = initial_lookup[initial_lookup["player_id"].notna()].copy()
    apply_lookup_matches(matched, initial_lookup, ["season", "initial_last_key"], "season_initial_last")


def apply_season_roster_matches(matched: pd.DataFrame, hs_lookup: pd.DataFrame) -> None:
    season_exact_lookup = build_unique_lookup(
        hs_lookup,
        ["season", "team_norm", "player_name_norm", "position_group"],
    )
    apply_lookup_matches(
        matched,
        season_exact_lookup,
        ["season", "team_norm", "player_name_norm", "position_group"],
        "season_team_position_exact_name",
    )

    hs_with_initial = hs_lookup.assign(initial_last_key=hs_lookup["player"].apply(build_initial_last_key))
    season_initial_lookup = build_unique_lookup(
        hs_with_initial,
        ["season", "team_norm", "initial_last_key", "position_group"],
    )
    apply_lookup_matches(
        matched,
        season_initial_lookup,
        ["season", "team_norm", "initial_last_key", "position_group"],
        "season_team_position_initial_last",
    )

def apply_direct_game_data_matches(matched: pd.DataFrame) -> None:
    unresolved = matched[matched["player_id"].isna()].copy()
    if unresolved.empty:
        return

    query_cache: dict[tuple[str, str, str, str], pd.DataFrame] = {}
    for row in unresolved.itertuples():
        if not row.game_date or not row.team or not row.position:
            continue

        cache_key = (str(row.season), str(row.game_date), str(row.team), str(row.position))
        candidates = query_cache.get(cache_key)
        if candidates is None:
            candidates = read_table(
                "game_data",
                columns="season,game_id,date,player_id,player,position,team",
                filters={
                    "season": f"eq.{row.season}",
                    "date": f"eq.{row.game_date}",
                    "team": f"eq.{row.team}",
                    "position": f"eq.{row.position}",
                },
                limit=100,
            )
            if not candidates.empty:
                candidates["player_name_norm"] = candidates["player"].apply(normalize_person_name)
                candidates["initial_last_key"] = candidates["player"].apply(build_initial_last_key)
                candidates["last_name_key"] = candidates["player"].apply(build_last_name_key)
            query_cache[cache_key] = candidates

        if candidates.empty:
            continue

        name_matches = candidates[
            (candidates["player_name_norm"] == row.player_name_norm)
            | (candidates["initial_last_key"] == row.initial_last_key)
        ].copy()
        if name_matches.empty:
            continue

        player_id = unique_or_none(name_matches["player_id"])
        if pd.isna(player_id):
            continue

        matched.at[row.Index, "game_id"] = unique_or_none(name_matches["game_id"])
        matched.at[row.Index, "player_id"] = player_id
        matched.at[row.Index, "player"] = unique_or_none(name_matches["player"])
        matched.at[row.Index, "match_type"] = "direct_game_data_fallback"


def apply_direct_game_id_matches(matched: pd.DataFrame) -> None:
    unresolved = matched[(matched["player_id"].notna()) & (matched["game_id"].isna())].copy()
    if unresolved.empty:
        return

    query_cache: dict[tuple[str, str, str], float | int | None] = {}
    for row in unresolved.itertuples():
        if not row.game_date or not row.team:
            continue

        cache_key = (str(row.season), str(row.game_date), str(row.team))
        resolved_game_id = query_cache.get(cache_key)
        if cache_key not in query_cache:
            candidates = read_table(
                "game_data",
                columns="game_id",
                filters={
                    "season": f"eq.{row.season}",
                    "date": f"eq.{row.game_date}",
                    "team": f"eq.{row.team}",
                },
                limit=100,
            )
            resolved_game_id = unique_or_none(candidates["game_id"]) if not candidates.empty else None
            query_cache[cache_key] = resolved_game_id

        if pd.isna(resolved_game_id):
            continue

        matched.at[row.Index, "game_id"] = resolved_game_id


def prepare_export_df(matched: pd.DataFrame, artifact: dict, model_key: str) -> pd.DataFrame:
    export_df = matched[matched["player_id"].notna()].copy()
    game_lookup = (
        export_df[export_df["game_id"].notna()]
        .groupby(["season", "source_game_id"], as_index=False)
        .agg(game_id=("game_id", unique_or_none))
        .rename(columns={"game_id": "resolved_game_id"})
    )
    export_df = export_df.merge(game_lookup, on=["season", "source_game_id"], how="left")
    export_df["game_id"] = export_df["game_id"].fillna(export_df["resolved_game_id"])
    export_df = export_df.drop(columns=["resolved_game_id"])

    missing_game_ids = export_df["game_id"].isna().sum()
    if missing_game_ids:
        raise RuntimeError(f"Cannot export {missing_game_ids} rows without a Hockey-Statistics game_id")

    export_df["game_id"] = export_df["game_id"].astype(int)
    export_df["source_game_id"] = export_df["source_game_id"].astype(int)
    export_df["player_id"] = export_df["player_id"].astype(int)
    export_df["source_player_id"] = export_df["playerid"].astype(int)
    export_df["season"] = export_df["season"].astype(int)
    export_df["generated_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")
    export_df["game_date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
    export_df["model_key"] = model_key
    export_df["window_games"] = int(artifact.get("window_games", artifact.get("gp_threshold", 41)))
    export_df["weighting"] = str(artifact.get("weighting", model_key))
    export_df["player"] = export_df["matched_player_name"]

    export_df = export_df.rename(
        columns={
            "rookie_F": "rookie_f",
            "rookie_D": "rookie_d",
            "rookie_G": "rookie_g",
            "raw_rookie_F": "raw_rookie_f",
            "raw_rookie_D": "raw_rookie_d",
            "raw_rookie_G": "raw_rookie_g",
        }
    )
    return export_df[
        [
            "season",
            "game_id",
            "source_game_id",
            "game_date",
            "player_id",
            "source_player_id",
            "player",
            "source_player_name",
            "team",
            "opponent",
            "position",
            "side",
            "model_key",
            "window_games",
            "weighting",
            "games_in_window",
            "rookie_factor",
            "poss_value",
            "off_the_puck",
            "gax",
            "goalie_gsax",
            "rookie_f",
            "rookie_d",
            "rookie_g",
            "raw_poss_value",
            "raw_off_the_puck",
            "raw_gax",
            "raw_goalie_gsax",
            "raw_rookie_f",
            "raw_rookie_d",
            "raw_rookie_g",
            "raw_projected_value",
            "projected_value",
            "match_type",
            "generated_at",
        ]
    ].copy()


def load_current_hockey_statistics_players() -> tuple[str, pd.DataFrame]:
    players = read_table("players", columns="season,player_id,player,position")
    if players.empty:
        raise RuntimeError("No Hockey-Statistics players rows found")

    players["season"] = players["season"].astype(str)
    players["player_id"] = pd.to_numeric(players["player_id"], errors="coerce").astype("Int64")
    players = players[players["player_id"].notna()].copy()
    players["player_id"] = players["player_id"].astype(int)
    players["player"] = players["player"].fillna("").astype(str).str.strip()
    players["position"] = players["position"].apply(normalize_position_group)

    invalid_positions = sorted(set(players.loc[~players["position"].isin({"F", "D", "G"}), "position"].tolist()))
    if invalid_positions:
        raise RuntimeError(f"Unexpected Hockey-Statistics player positions: {invalid_positions}")

    current_season = players["season"].max()
    current_players = players[players["season"] == current_season].copy()
    current_players = current_players.drop_duplicates(subset=["season", "player_id"], keep="last")
    return current_season, current_players


def load_current_projection_player_lookup(current_season: str) -> pd.DataFrame:
    cached_export = ROOT / "data" / "game_projection" / "export_validation" / f"{current_season}_export.pkl"
    if cached_export.exists():
        current_rows = pd.read_pickle(cached_export)
    else:
        current_rows = read_table(
            DEFAULT_TABLE,
            columns="season,player_id,source_player_id,model_key",
            filters={
                "season": f"eq.{current_season}",
                "model_key": f"eq.{DEFAULT_MODEL_KEY}",
            },
        )
    if current_rows.empty:
        return pd.DataFrame(columns=["source_player_id", "player_id"])

    current_rows["source_player_id"] = pd.to_numeric(current_rows["source_player_id"], errors="coerce")
    current_rows["player_id"] = pd.to_numeric(current_rows["player_id"], errors="coerce")
    current_rows = current_rows.dropna(subset=["source_player_id", "player_id"]).copy()
    if current_rows.empty:
        return pd.DataFrame(columns=["source_player_id", "player_id"])

    lookup = (
        current_rows.groupby("source_player_id", as_index=False)
        .agg(player_id=("player_id", unique_or_none))
    )
    lookup = lookup[lookup["player_id"].notna()].copy()
    lookup["source_player_id"] = lookup["source_player_id"].astype(int)
    lookup["player_id"] = lookup["player_id"].astype(int)
    return lookup


def build_moncton_player_name_lookup(moncton_players: pd.DataFrame) -> pd.DataFrame:
    lookup = moncton_players.copy()
    lookup["player_name_norm"] = lookup["source_player_name"].apply(normalize_person_name)
    lookup = lookup[lookup["player_name_norm"] != ""].copy()
    lookup = (
        lookup.groupby("player_name_norm", as_index=False)
        .agg(source_player_id=("playerid", unique_or_none))
    )
    lookup = lookup[lookup["source_player_id"].notna()].copy()
    lookup["source_player_id"] = lookup["source_player_id"].astype(int)
    return lookup


def load_historical_source_player_lookup(current_season: str) -> pd.DataFrame:
    cache_dir = ROOT / "data" / "game_projection" / "export_validation"
    cached_frames: list[pd.DataFrame] = []
    if cache_dir.exists():
        for cache_path in sorted(cache_dir.glob("*_export.pkl")):
            season = cache_path.stem.replace("_export", "")
            if season == current_season:
                continue
            cached_frame = pd.read_pickle(cache_path)
            cached_frames.append(cached_frame[["season", "player_id", "source_player_id", "model_key"]].copy())

    if cached_frames:
        historical_rows = pd.concat(cached_frames, ignore_index=True)
    else:
        historical_rows = read_table(
            DEFAULT_TABLE,
            columns="season,player_id,source_player_id,model_key",
            filters={"model_key": f"eq.{DEFAULT_MODEL_KEY}"},
        )
    if historical_rows.empty:
        return pd.DataFrame(columns=["player_id", "historical_source_player_id"])

    historical_rows["season"] = historical_rows["season"].astype(str)
    historical_rows = historical_rows[historical_rows["season"] != current_season].copy()
    if historical_rows.empty:
        return pd.DataFrame(columns=["player_id", "historical_source_player_id"])

    historical_rows["player_id"] = pd.to_numeric(historical_rows["player_id"], errors="coerce")
    historical_rows["source_player_id"] = pd.to_numeric(historical_rows["source_player_id"], errors="coerce")
    historical_rows = historical_rows.dropna(subset=["player_id", "source_player_id"]).copy()
    if historical_rows.empty:
        return pd.DataFrame(columns=["player_id", "historical_source_player_id"])

    lookup = (
        historical_rows.groupby("player_id", as_index=False)
        .agg(historical_source_player_id=("source_player_id", unique_or_none))
    )
    lookup = lookup[lookup["historical_source_player_id"].notna()].copy()
    lookup["player_id"] = lookup["player_id"].astype(int)
    lookup["historical_source_player_id"] = lookup["historical_source_player_id"].astype(int)
    return lookup


def attach_current_source_player_ids(
    current_players: pd.DataFrame,
    moncton_players: pd.DataFrame,
) -> pd.DataFrame:
    attached = current_players.copy()
    current_season = attached["season"].astype(str).max()
    historical_lookup = load_historical_source_player_lookup(current_season)
    attached = attached.merge(historical_lookup, on="player_id", how="left")
    attached["player_name_norm"] = attached["player"].apply(normalize_person_name)
    moncton_lookup = build_moncton_player_name_lookup(moncton_players)
    attached = attached.merge(moncton_lookup, on="player_name_norm", how="left")
    attached["source_player_id"] = attached["historical_source_player_id"].fillna(attached["source_player_id"])
    return attached.drop(columns=["player_name_norm", "historical_source_player_id"])


def build_current_preseason_updating_raw_profiles(
    player_game_box_stats: pd.DataFrame,
    current_players: pd.DataFrame,
    current_season: str,
) -> pd.DataFrame:
    season_totals = player_game_box_stats.groupby(["season", "playerid"], as_index=False).agg(
        gp=("gameid", "nunique"),
        **{col: (col, "sum") for col in STAT_COLS},
    )

    current_rows = player_game_box_stats[player_game_box_stats["season"].astype(str) == current_season].copy()

    candidate_players = current_players.copy()
    candidate_players["source_player_id"] = pd.to_numeric(candidate_players["source_player_id"], errors="coerce")
    candidate_players = candidate_players.dropna(subset=["source_player_id"]).copy()
    if candidate_players.empty:
        return pd.DataFrame(
            columns=[
                "season",
                "source_player_id",
                "source_game_id",
                "source_game_date",
                "position",
                "games_in_window",
                "rookie_factor",
                *STAT_COLS,
                *ROOKIE_COLS,
            ]
        )
    candidate_players["source_player_id"] = candidate_players["source_player_id"].astype(int)
    candidate_players["position"] = candidate_players["position"].apply(normalize_position_group)
    candidate_players = candidate_players.drop_duplicates(subset=["source_player_id"], keep="last")

    current_totals = season_totals[season_totals["season"].astype(str) == current_season].rename(
        columns={
            "gp": "current_gp",
            **{col: f"current_{col}" for col in STAT_COLS},
        }
    )

    latest_current = (
        current_rows.sort_values(["playerid", "date", "gameid"])
        .groupby("playerid", as_index=False)
        .tail(1)
        [["playerid", "gameid", "date"]]
        .rename(
            columns={
                "playerid": "source_player_id",
                "gameid": "source_game_id",
                "date": "source_game_date",
            }
        )
    )

    prev_season_key = prev_season(current_season)
    prev_totals = season_totals[season_totals["season"].astype(str) == prev_season_key].rename(
        columns={
            "gp": "prev_gp",
            **{col: f"prev_{col}" for col in STAT_COLS},
        }
    )

    goalie_prev1 = season_totals[season_totals["season"].astype(str) == prev_season_key][
        ["playerid", "gp", "goalie_gsax"]
    ].rename(
        columns={
            "playerid": "source_player_id",
            "gp": "goalie_prev_gp_1",
            "goalie_gsax": "goalie_prev_gsax_1",
        }
    )
    goalie_prev2_key = season_years_ago(current_season, GOALIE_PRESEASON_LOOKBACK_SEASONS)
    goalie_prev2 = season_totals[season_totals["season"].astype(str) == goalie_prev2_key][
        ["playerid", "gp", "goalie_gsax"]
    ].rename(
        columns={
            "playerid": "source_player_id",
            "gp": "goalie_prev_gp_2",
            "goalie_gsax": "goalie_prev_gsax_2",
        }
    )

    profiles = candidate_players[["source_player_id", "position"]].merge(
        latest_current,
        on="source_player_id",
        how="left",
    )
    profiles = profiles.merge(
        current_totals[["playerid", "current_gp"] + [f"current_{col}" for col in STAT_COLS]].rename(
            columns={"playerid": "source_player_id"}
        ),
        on="source_player_id",
        how="left",
    )
    profiles = profiles.merge(
        prev_totals[["playerid", "prev_gp"] + [f"prev_{col}" for col in STAT_COLS]].rename(
            columns={"playerid": "source_player_id"}
        ),
        on="source_player_id",
        how="left",
    )
    profiles = profiles.merge(goalie_prev1, on="source_player_id", how="left")
    profiles = profiles.merge(goalie_prev2, on="source_player_id", how="left")

    profiles["season"] = current_season
    profiles["current_gp"] = profiles["current_gp"].fillna(0.0)
    profiles["prev_gp"] = profiles["prev_gp"].fillna(0.0)
    profiles["goalie_prev_gp_1"] = profiles["goalie_prev_gp_1"].fillna(0.0)
    profiles["goalie_prev_gp_2"] = profiles["goalie_prev_gp_2"].fillna(0.0)
    profiles["goalie_prev_gsax_1"] = profiles["goalie_prev_gsax_1"].fillna(0.0)
    profiles["goalie_prev_gsax_2"] = profiles["goalie_prev_gsax_2"].fillna(0.0)
    for col in STAT_COLS:
        profiles[f"current_{col}"] = profiles[f"current_{col}"].fillna(0.0)
        profiles[f"prev_{col}"] = profiles[f"prev_{col}"].fillna(0.0)

    profiles["games_in_window"] = (profiles["prev_gp"] + profiles["current_gp"]).astype(int)
    for col in STAT_COLS:
        if col == "goalie_gsax":
            gp_adj = (profiles["goalie_prev_gp_1"] + profiles["goalie_prev_gp_2"] + profiles["current_gp"]).clip(
                lower=GP_THRESHOLD
            )
            numerator = profiles["goalie_prev_gsax_1"] + profiles["goalie_prev_gsax_2"] + profiles[f"current_{col}"]
        else:
            gp_adj = (profiles["prev_gp"] + profiles["current_gp"]).clip(lower=GP_THRESHOLD)
            numerator = profiles[f"prev_{col}"] + profiles[f"current_{col}"]
        profiles[col] = numerator / gp_adj

    profiles["rookie_factor"] = (
        (GP_THRESHOLD - (profiles["prev_gp"] + profiles["current_gp"]).clip(upper=GP_THRESHOLD)) / GP_THRESHOLD
    )
    for pos in ("F", "D", "G"):
        profiles[f"rookie_{pos}"] = profiles["rookie_factor"] * (profiles["position"] == pos).astype(float)

    return profiles[
        [
            "season",
            "source_player_id",
            "source_game_id",
            "source_game_date",
            "position",
            "games_in_window",
            "rookie_factor",
            *STAT_COLS,
            *ROOKIE_COLS,
        ]
    ].copy()


def prepare_current_projection_export_df(
    current_players: pd.DataFrame,
    current_raw_profiles: pd.DataFrame,
    current_projection_lookup: pd.DataFrame,
    artifact: dict,
    coef_map: dict[str, float],
    model_key: str,
) -> pd.DataFrame:
    current_players = current_players.copy()

    current_profiles = current_raw_profiles.merge(current_projection_lookup, on="source_player_id", how="left")
    current_profiles["player_id"] = pd.to_numeric(current_profiles["player_id"], errors="coerce").astype("Int64")

    export_df = current_players.merge(
        current_profiles,
        on=["season", "player_id", "position"],
        how="left",
        suffixes=("", "_profile"),
    )
    export_df["source_player_id"] = export_df["source_player_id_profile"].fillna(export_df["source_player_id"])
    export_df = export_df.drop(columns=["source_player_id_profile"])

    export_df["games_in_window"] = export_df["games_in_window"].fillna(0).astype(int)
    export_df["rookie_factor"] = export_df["rookie_factor"].fillna(1.0)
    for col in STAT_COLS:
        export_df[col] = export_df[col].fillna(0.0)
    for pos in ("F", "D", "G"):
        export_df[f"rookie_{pos}"] = export_df[f"rookie_{pos}"].fillna(
            export_df["rookie_factor"] * (export_df["position"] == pos).astype(float)
        )

    centered = center_metrics_by_position(export_df[["season", "position", *PROFILE_METRIC_COLS]].copy())
    for col in PROFILE_METRIC_COLS:
        export_df[f"raw_{col}"] = export_df[col]
        export_df[col] = centered[col]

    export_df["raw_projected_value"] = 0.0
    export_df["projected_value"] = 0.0
    for feature_name, coef in coef_map.items():
        raw_feature_name = f"raw_{feature_name}"
        export_df["raw_projected_value"] += export_df[raw_feature_name].fillna(0.0) * coef
        export_df["projected_value"] += export_df[feature_name].fillna(0.0) * coef

    export_df["season"] = export_df["season"].astype(int)
    export_df["player_id"] = export_df["player_id"].astype(int)
    export_df["source_player_id"] = pd.to_numeric(export_df["source_player_id"], errors="coerce").astype("Int64")
    export_df["source_game_id"] = pd.to_numeric(export_df["source_game_id"], errors="coerce").astype("Int64")
    export_df["source_game_date"] = pd.to_datetime(export_df["source_game_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    export_df["model_key"] = model_key
    export_df["window_games"] = int(artifact.get("window_games", artifact.get("gp_threshold", 41)))
    export_df["weighting"] = str(artifact.get("weighting", model_key))
    export_df["generated_at"] = pd.Timestamp.now("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

    export_df = export_df.rename(
        columns={
            "rookie_F": "rookie_f",
            "rookie_D": "rookie_d",
            "rookie_G": "rookie_g",
            "raw_rookie_F": "raw_rookie_f",
            "raw_rookie_D": "raw_rookie_d",
            "raw_rookie_G": "raw_rookie_g",
        }
    )
    return export_df[
        [
            "season",
            "player_id",
            "source_player_id",
            "source_game_id",
            "source_game_date",
            "player",
            "position",
            "model_key",
            "window_games",
            "weighting",
            "games_in_window",
            "rookie_factor",
            "poss_value",
            "off_the_puck",
            "gax",
            "goalie_gsax",
            "rookie_f",
            "rookie_d",
            "rookie_g",
            "raw_poss_value",
            "raw_off_the_puck",
            "raw_gax",
            "raw_goalie_gsax",
            "raw_rookie_f",
            "raw_rookie_d",
            "raw_rookie_g",
            "raw_projected_value",
            "projected_value",
            "generated_at",
        ]
    ].copy()


def export_current_player_projections(
    table: str = CURRENT_TABLE,
    model_path: pathlib.Path | None = None,
    refresh_data: bool = False,
    refresh_player_cache: bool = False,
    dry_run: bool = False,
    skip_ensure_table: bool = False,
) -> pd.DataFrame:
    if model_path is None:
        model_path = pathlib.Path(DEFAULT_MODEL_PATH)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    current_season, current_players = load_current_hockey_statistics_players()
    games, pvm, skaters, goalies = load_base_inputs(refresh_data=refresh_data)
    games, pvm, skaters, goalies = filter_inputs_for_requested_seasons(
        games,
        pvm,
        skaters,
        goalies,
        {current_season},
    )
    moncton_players = load_moncton_players(refresh_cache=refresh_player_cache)
    current_players = attach_current_source_player_ids(current_players, moncton_players)
    artifact, coef_map = load_projection_artifact(model_path)

    player_game_box_stats = build_player_game_box_stats(pvm, skaters, goalies, games)
    current_raw_profiles = build_current_preseason_updating_raw_profiles(
        player_game_box_stats,
        current_players,
        current_season,
    )
    current_projection_lookup = load_current_projection_player_lookup(current_season)
    export_df = prepare_current_projection_export_df(
        current_players,
        current_raw_profiles,
        current_projection_lookup,
        artifact,
        coef_map,
        DEFAULT_MODEL_KEY,
    )

    print(f"Computed current player projections for season {current_season}: {len(export_df):,}")
    print(f"Players with finished-game history: {int(export_df['source_game_id'].notna().sum()):,}")
    print(f"Rookie fallback rows: {int(export_df['source_game_id'].isna().sum()):,}")

    if dry_run:
        return export_df

    if not skip_ensure_table:
        created = apply_sql_statements(os.environ.get("SUPABASE_DB_URL", ""), CURRENT_MIGRATION_PATHS)
        if created:
            print("Ensured target table exists via " + ", ".join(path.name for path in CURRENT_MIGRATION_PATHS))

    upsert_df(table, export_df, on_conflict="season,player_id,model_key")
    print(f"Upserted {len(export_df):,} rows to {table}")
    return export_df


def apply_sql_statements(db_url: str, sql_paths: list[pathlib.Path]) -> bool:
    if not db_url:
        print("SUPABASE_DB_URL is not set; skipping live table creation")
        return False
    engine = create_engine(db_url)
    with engine.begin() as conn:
        for sql_path in sql_paths:
            if not sql_path.exists():
                raise FileNotFoundError(f"Migration not found: {sql_path}")

            raw_sql = sql_path.read_text(encoding="utf-8")
            sql_lines = []
            for line in raw_sql.splitlines():
                if line.strip().startswith("--"):
                    continue
                sql_lines.append(line)
            statements = [stmt.strip() for stmt in "\n".join(sql_lines).split(";") if stmt.strip()]
            for statement in statements:
                conn.exec_driver_sql(statement)
    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export preseason-updating player game projections")
    parser.add_argument("--table", default=DEFAULT_TABLE, help="Supabase target table")
    parser.add_argument(
        "--model-path",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to the saved preseason coefficient artifact",
    )
    parser.add_argument(
        "--season",
        nargs="*",
        default=None,
        help="Optional season filter, e.g. 20252026",
    )
    parser.add_argument("--refresh-data", action="store_true", help="Reload Moncton base inputs from DB")
    parser.add_argument(
        "--refresh-player-cache",
        action="store_true",
        help="Reload the Moncton player name cache from DB",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build and validate rows without upserting")
    parser.add_argument(
        "--skip-ensure-table",
        action="store_true",
        help="Do not try to create the target table before upsert",
    )
    return parser.parse_args()


def get_requested_seasons(args: argparse.Namespace, games: pd.DataFrame) -> set[str]:
    if args.season:
        requested = {str(season).strip() for season in args.season if str(season).strip()}
    else:
        requested = set(games["season"].astype(str).unique())

    skipped = requested & EXCLUDED_EXPORT_SEASONS
    if skipped:
        print(
            "Skipping seasons with insufficient two-year goalie history for export: "
            + ", ".join(sorted(skipped))
        )
    requested -= EXCLUDED_EXPORT_SEASONS
    return requested


def filter_inputs_for_requested_seasons(
    games: pd.DataFrame,
    pvm: pd.DataFrame,
    skaters: pd.DataFrame,
    goalies: pd.DataFrame,
    requested_seasons: set[str],
):
    relevant_seasons = set(requested_seasons)
    relevant_seasons.update(prev_season(season) for season in requested_seasons)
    games = games[games["season"].isin(relevant_seasons)].copy()
    pvm = pvm[pvm["season"].isin(relevant_seasons)].copy()
    skaters = skaters[skaters["season"].isin(relevant_seasons)].copy()
    goalies = goalies[goalies["season"].isin(relevant_seasons)].copy()
    return games, pvm, skaters, goalies


def main() -> None:
    args = parse_args()
    model_path = pathlib.Path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")

    games, pvm, skaters, goalies = load_base_inputs(refresh_data=args.refresh_data)
    requested_seasons = get_requested_seasons(args, games)
    if not requested_seasons:
        raise RuntimeError("No exportable seasons remain after excluding early seasons")
    games, pvm, skaters, goalies = filter_inputs_for_requested_seasons(
        games,
        pvm,
        skaters,
        goalies,
        requested_seasons,
    )
    moncton_players = load_moncton_players(refresh_cache=args.refresh_player_cache)
    artifact, coef_map = load_projection_artifact(model_path)
    projections = compute_player_projections(games, pvm, skaters, goalies, moncton_players, artifact, coef_map)

    projections = projections[projections["season"].isin(requested_seasons)].copy()

    if projections.empty:
        raise RuntimeError("No projections were generated for the requested seasons")

    hs_lookup = load_hockey_statistics_lookup(sorted(requested_seasons))
    hs_players_lookup = load_hockey_statistics_players_lookup(sorted(requested_seasons))
    matched, unmatched = match_player_ids(projections, hs_lookup)
    apply_players_table_matches(matched, hs_players_lookup)
    apply_season_roster_matches(matched, hs_lookup)
    apply_direct_game_data_matches(matched)
    apply_direct_game_id_matches(matched)
    matched["matched_player_name"] = matched["player"].fillna("")
    unmatched = matched[matched["player_id"].isna()].copy()
    export_df = prepare_export_df(matched, artifact, model_key=DEFAULT_MODEL_KEY)

    print(f"Computed player-game projections: {len(projections):,}")
    print(f"Matched Hockey-Statistics player IDs: {len(export_df):,}")
    print(f"Unmatched rows: {len(unmatched):,}")

    if not unmatched.empty:
        unmatched_path = ROOT / "data" / "game_projection" / f"{DEFAULT_MODEL_KEY}_player_projection_unmatched.csv"
        unmatched_path.parent.mkdir(parents=True, exist_ok=True)
        unmatched[
            [
                "season",
                "source_game_id",
                "game_date",
                "playerid",
                "source_player_name",
                "team",
                "opponent",
                "position",
                "games_in_window",
                "projected_value",
            ]
        ].to_csv(unmatched_path, index=False)
        print(f"Wrote unmatched rows to {unmatched_path}")

    match_summary = (
        export_df.groupby("match_type", dropna=False)
        .size()
        .rename("rows")
        .reset_index()
    )
    if not match_summary.empty:
        print("Match summary:")
        print(match_summary.to_string(index=False))

    if args.dry_run:
        return

    if not args.skip_ensure_table:
        created = apply_sql_statements(os.environ.get("SUPABASE_DB_URL", ""), MIGRATION_PATHS)
        if created:
            print(
                "Ensured target table exists via "
                + ", ".join(path.name for path in MIGRATION_PATHS)
            )

    upsert_df(
        args.table,
        export_df,
        on_conflict="season,game_id,player_id,model_key",
    )
    print(f"Upserted {len(export_df):,} rows to {args.table}")


if __name__ == "__main__":
    main()