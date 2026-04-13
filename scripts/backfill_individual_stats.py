"""Backfill individual shooting stats into the game_data Supabase table.

Reads pbp rows for a season, computes per-player individual Corsi/Fenwick/
Shots/xG per strength state, then patches the existing game_data rows.

Usage:
    python scripts/backfill_individual_stats.py --season 20252026
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Dict, List

import pandas as pd

# Ensure project root is on sys.path so we can import app.*
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

os.environ.setdefault("XG_PRELOAD", "0")

from dotenv import load_dotenv
load_dotenv()

from app.supabase_client import read_table, upsert_df


EV_SET = {'5v5', '4v4', '3v3'}
PP_SET = {'5v4', '5v3', '4v3'}
SH_SET = {'4v5', '3v5', '3v4'}


def _strength_suffix(ss: str) -> str | None:
    if ss in EV_SET:
        return '_ev'
    if ss in PP_SET:
        return '_pp'
    if ss in SH_SET:
        return '_sh'
    return None


def compute_individual_stats(pbp: pd.DataFrame) -> pd.DataFrame:
    """Compute individual shooting stats per (game_id, player_id).

    Returns a DataFrame with columns:
        game_id, player_id,
        icf_all, icf_ev, icf_pp, icf_sh,
        iff_all, iff_ev, iff_pp, iff_sh,
        isf_all, isf_ev, isf_pp, isf_sh,
        ixgf_all, ixgf_ev, ixgf_pp, ixgf_sh,
        ixgs_all, ixgs_ev, ixgs_pp, ixgs_sh,
        ixgf2_all, ixgf2_ev, ixgf2_pp, ixgf2_sh,
    """
    # Normalise column names to lowercase for safety
    pbp.columns = [c.lower() for c in pbp.columns]

    # Supabase uses snake_case; also accept camelCase from other sources
    _col_aliases = {
        'strength_state': 'strengthstate',
        'player1_id': 'player1_id',
        'xg_f': 'xg_f',
        'xg_s': 'xg_s',
        'xg_f2': 'xg_f2',
    }
    for old, new in _col_aliases.items():
        if old in pbp.columns and new not in pbp.columns:
            pbp.rename(columns={old: new}, inplace=True)

    needed = ['game_id', 'player1_id', 'strengthstate']
    for c in needed:
        if c not in pbp.columns:
            raise ValueError(f"pbp missing column: {c}")

    # Coerce numeric columns
    for col in ['corsi', 'fenwick', 'shot', 'goal', 'xg_f', 'xg_s', 'xg_f2']:
        if col in pbp.columns:
            pbp[col] = pd.to_numeric(pbp[col], errors='coerce').fillna(0)
        else:
            pbp[col] = 0

    pbp['player1_id'] = pd.to_numeric(pbp['player1_id'], errors='coerce')
    pbp = pbp.dropna(subset=['player1_id'])
    pbp['player1_id'] = pbp['player1_id'].astype(int)

    ss = pbp['strengthstate'].fillna('').astype(str)
    pbp['_suf'] = ss.map(lambda v: _strength_suffix(v))

    # Accumulators per (game_id, player_id)
    acc: Dict[tuple, Dict[str, float]] = {}

    for _, row in pbp.iterrows():
        gid = row['game_id']
        pid = row['player1_id']
        suf = row['_suf']
        key = (gid, pid)

        # Skip non-standard strength states for suffix breakdown
        has_suf = isinstance(suf, str) and suf.startswith('_')

        d = acc.setdefault(key, {
            'icf_all': 0, 'icf_ev': 0, 'icf_pp': 0, 'icf_sh': 0,
            'iff_all': 0, 'iff_ev': 0, 'iff_pp': 0, 'iff_sh': 0,
            'isf_all': 0, 'isf_ev': 0, 'isf_pp': 0, 'isf_sh': 0,
            'ixgf_all': 0.0, 'ixgf_ev': 0.0, 'ixgf_pp': 0.0, 'ixgf_sh': 0.0,
            'ixgs_all': 0.0, 'ixgs_ev': 0.0, 'ixgs_pp': 0.0, 'ixgs_sh': 0.0,
            'ixgf2_all': 0.0, 'ixgf2_ev': 0.0, 'ixgf2_pp': 0.0, 'ixgf2_sh': 0.0,
        })

        corsi = int(row['corsi'])
        fenwick = int(row['fenwick'])
        shot = int(row['shot'])
        xgf = float(row['xg_f'])
        xgs = float(row['xg_s'])
        xgf2 = float(row['xg_f2'])

        if corsi:
            d['icf_all'] += corsi
            if has_suf:
                d[f'icf{suf}'] += corsi
        if fenwick:
            d['iff_all'] += fenwick
            if has_suf:
                d[f'iff{suf}'] += fenwick
        if shot:
            d['isf_all'] += shot
            if has_suf:
                d[f'isf{suf}'] += shot
        if xgf:
            d['ixgf_all'] += xgf
            if has_suf:
                d[f'ixgf{suf}'] += xgf
        if xgs:
            d['ixgs_all'] += xgs
            if has_suf:
                d[f'ixgs{suf}'] += xgs
        if xgf2:
            d['ixgf2_all'] += xgf2
            if has_suf:
                d[f'ixgf2{suf}'] += xgf2

    if not acc:
        return pd.DataFrame()

    rows = []
    for (gid, pid), d in acc.items():
        row = {'game_id': int(gid), 'player_id': int(pid)}
        row.update(d)
        rows.append(row)

    df = pd.DataFrame(rows)
    # Round xG columns
    for c in df.columns:
        if c.startswith('ixg'):
            df[c] = df[c].round(4)
    # Int columns
    for c in df.columns:
        if c.startswith(('icf_', 'iff_', 'isf_')):
            df[c] = df[c].astype(int)
    return df


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Backfill individual shooting stats into game_data")
    p.add_argument("--season", default="20252026", help="Season code")
    p.add_argument("--dry-run", action="store_true", help="Compute but don't write")
    args = p.parse_args(argv)

    season = int(str(args.season).strip())

    # Get distinct game IDs from game_data (much smaller than pbp)
    print(f"[backfill] fetching game IDs for season {season} ...")
    gd = read_table("game_data", columns="game_id", filters={"season": f"eq.{season}"})
    if gd.empty:
        print("[backfill] no game_data rows found; nothing to do")
        return 0
    game_ids = sorted(gd['game_id'].unique())
    print(f"[backfill] found {len(game_ids)} games to process")

    # Load PBP in batches of game IDs using the 'in' filter
    BATCH = 20
    all_indiv: list[pd.DataFrame] = []
    for i in range(0, len(game_ids), BATCH):
        batch_ids = game_ids[i:i + BATCH]
        id_list = ",".join(str(int(g)) for g in batch_ids)
        pbp = read_table("pbp", filters={
            "game_id": f"in.({id_list})",
            "season": f"eq.{season}",
        })
        if pbp.empty:
            continue
        df_indiv = compute_individual_stats(pbp)
        if not df_indiv.empty:
            all_indiv.append(df_indiv)
        n = min(i + BATCH, len(game_ids))
        print(f"  ... processed {n}/{len(game_ids)} games ({len(df_indiv)} player-game rows)")

    if not all_indiv:
        print("[backfill] no individual stats computed; nothing to do")
        return 0

    df_all = pd.concat(all_indiv, ignore_index=True)
    print(f"[backfill] computed stats for {len(df_all)} (game, player) combos across {len(game_ids)} games")

    # Add season column for the upsert conflict key
    df_all['season'] = season

    if args.dry_run:
        print("[backfill] dry run — not writing to Supabase")
        print(df_all.head(20).to_string())
        return 0

    print(f"[backfill] upserting {len(df_all)} rows to game_data ...")
    upsert_df("game_data", df_all, on_conflict="game_id,player_id,season")
    print(f"[backfill] done — {len(df_all)} rows updated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
