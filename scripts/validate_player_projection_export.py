from __future__ import annotations

import argparse
import pathlib
import sys

import pandas as pd
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.supabase_client import read_table, upsert_df
from scripts.export_preseason_updating_player_projections import (
    DEFAULT_MODEL_KEY,
    DEFAULT_MODEL_PATH,
    apply_direct_game_data_matches,
    apply_direct_game_id_matches,
    apply_players_table_matches,
    apply_season_roster_matches,
    compute_player_projections,
    filter_inputs_for_requested_seasons,
    load_base_inputs,
    load_hockey_statistics_lookup,
    load_hockey_statistics_players_lookup,
    load_moncton_players,
    load_projection_artifact,
    match_player_ids,
    normalize_person_name,
    normalize_position_group,
    normalize_team,
    prepare_export_df,
)


CACHE_DIR = ROOT / "data" / "game_projection" / "export_validation"
GAME_DATA_DATE_CHUNK = 7


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate staged player projection export workflow")
    parser.add_argument("--season", required=True, help="Single season to validate, e.g. 20252026")
    parser.add_argument(
        "--step",
        required=True,
        choices=["build", "match", "check", "upsert"],
        help="Workflow step to run",
    )
    parser.add_argument(
        "--table",
        default="player_game_projections",
        help="Target table for the upsert step",
    )
    return parser.parse_args()


def cache_path(season: str, suffix: str) -> pathlib.Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{season}_{suffix}.pkl"


def build_hockey_statistics_lookup_for_projections(season: str, projections: pd.DataFrame) -> pd.DataFrame:
    cache = cache_path(season, "hs_lookup")
    if cache.exists():
        return pd.read_pickle(cache)

    dates = sorted(pd.to_datetime(projections["date"], errors="coerce").dt.strftime("%Y-%m-%d").dropna().unique())
    if not dates:
        raise RuntimeError(f"No projection dates available for season {season}")

    print(f"Loading Hockey-Statistics players table for {season}...")
    players = read_table(
        "players",
        columns="season,player_id,player",
        filters={"season": f"eq.{season}"},
    )
    if players.empty:
        raise RuntimeError(f"No Hockey-Statistics players rows found for season {season}")

    game_data_batches: list[pd.DataFrame] = []
    for idx in range(0, len(dates), GAME_DATA_DATE_CHUNK):
        chunk = dates[idx : idx + GAME_DATA_DATE_CHUNK]
        print(
            f"Loading Hockey-Statistics game_data for {season}: chunk {idx // GAME_DATA_DATE_CHUNK + 1}/"
            f"{(len(dates) + GAME_DATA_DATE_CHUNK - 1) // GAME_DATA_DATE_CHUNK}"
        )
        batch = read_table(
            "game_data",
            columns="season,game_id,date,player_id,player,position,team",
            filters={
                "season": f"eq.{season}",
                "date": f"in.({','.join(chunk)})",
            },
        )
        if not batch.empty:
            game_data_batches.append(batch)

    if not game_data_batches:
        raise RuntimeError(f"No Hockey-Statistics game_data rows found for season {season}")

    game_data = pd.concat(game_data_batches, ignore_index=True).drop_duplicates()
    game_data["season"] = game_data["season"].astype(str)
    players["season"] = players["season"].astype(str)
    players = players.rename(columns={"player": "players_table_name"})
    lookup = game_data.merge(players, on=["season", "player_id"], how="left")
    lookup["canonical_player_name"] = lookup["players_table_name"].fillna(lookup["player"])
    lookup["team_norm"] = lookup["team"].apply(normalize_team)
    lookup["position_group"] = lookup["position"].apply(normalize_position_group)
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
        raise RuntimeError(f"No Hockey-Statistics lookup candidates were built for season {season}")

    candidate_df.to_pickle(cache)
    return candidate_df


def build_step(season: str) -> pathlib.Path:
    requested_seasons = {season}
    games, pvm, skaters, goalies = load_base_inputs(refresh_data=False)
    games, pvm, skaters, goalies = filter_inputs_for_requested_seasons(
        games,
        pvm,
        skaters,
        goalies,
        requested_seasons,
    )
    moncton_players = load_moncton_players(refresh_cache=False)
    artifact, coef_map = load_projection_artifact(pathlib.Path(DEFAULT_MODEL_PATH))
    projections = compute_player_projections(
        games,
        pvm,
        skaters,
        goalies,
        moncton_players,
        artifact,
        coef_map,
    )
    projections = projections[projections["season"].astype(str) == season].copy()
    out_path = cache_path(season, "projections")
    projections.to_pickle(out_path)
    print({"step": "build", "season": season, "rows": len(projections), "path": str(out_path)})
    return out_path


def match_step(season: str) -> pathlib.Path:
    projections_path = cache_path(season, "projections")
    if not projections_path.exists():
        raise FileNotFoundError(f"Missing build cache: {projections_path}")

    projections = pd.read_pickle(projections_path)
    hs_lookup = build_hockey_statistics_lookup_for_projections(season, projections)
    hs_players_lookup = load_hockey_statistics_players_lookup([season])
    matched, _ = match_player_ids(projections, hs_lookup)
    apply_players_table_matches(matched, hs_players_lookup)
    apply_season_roster_matches(matched, hs_lookup)
    apply_direct_game_data_matches(matched)
    apply_direct_game_id_matches(matched)
    matched["matched_player_name"] = matched["player"].fillna("")

    unmatched = matched[matched["player_id"].isna()].copy()
    unmatched_path = CACHE_DIR / f"{season}_unmatched.csv"
    unmatched.to_csv(unmatched_path, index=False)

    out_path = cache_path(season, "matched")
    matched.to_pickle(out_path)
    print(
        {
            "step": "match",
            "season": season,
            "rows": len(matched),
            "unmatched_rows": len(unmatched),
            "path": str(out_path),
            "unmatched_path": str(unmatched_path),
        }
    )
    return out_path


def check_step(season: str) -> pathlib.Path:
    matched_path = cache_path(season, "matched")
    if not matched_path.exists():
        raise FileNotFoundError(f"Missing match cache: {matched_path}")

    matched = pd.read_pickle(matched_path)
    artifact, _ = load_projection_artifact(pathlib.Path(DEFAULT_MODEL_PATH))
    export_df = prepare_export_df(matched, artifact, model_key=DEFAULT_MODEL_KEY)

    key_cols = ["season", "game_id", "player_id", "model_key"]
    dupes = export_df[export_df.duplicated(key_cols, keep=False)].copy()
    dupe_path = CACHE_DIR / f"{season}_duplicate_keys.csv"
    dupes.to_csv(dupe_path, index=False)

    out_path = cache_path(season, "export")
    export_df.to_pickle(out_path)
    print(
        {
            "step": "check",
            "season": season,
            "rows": len(export_df),
            "duplicate_rows": len(dupes),
            "path": str(out_path),
            "duplicate_path": str(dupe_path),
        }
    )
    return out_path


def upsert_step(season: str, table: str) -> None:
    export_path = cache_path(season, "export")
    if not export_path.exists():
        raise FileNotFoundError(f"Missing check cache: {export_path}")

    dupe_path = CACHE_DIR / f"{season}_duplicate_keys.csv"
    if dupe_path.exists():
        dupes = pd.read_csv(dupe_path)
        if not dupes.empty:
            raise RuntimeError(f"Refusing to upsert with {len(dupes)} duplicate conflict rows: {dupe_path}")

    export_df = pd.read_pickle(export_path)
    upsert_df(table, export_df, on_conflict="season,game_id,player_id,model_key")
    print({"step": "upsert", "season": season, "rows": len(export_df), "table": table})


def main() -> None:
    args = parse_args()
    season = str(args.season).strip()

    if args.step == "build":
        build_step(season)
        return
    if args.step == "match":
        match_step(season)
        return
    if args.step == "check":
        check_step(season)
        return
    if args.step == "upsert":
        upsert_step(season, args.table)
        return

    raise ValueError(f"Unknown step: {args.step}")


if __name__ == "__main__":
    main()