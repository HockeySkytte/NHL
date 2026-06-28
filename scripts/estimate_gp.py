"""
estimate_gp.py — Add projected 2026-27 games played to lineups_all.json

For each unique player in the lineups, fetches NHL career stats (seasonTotals
from /v1/player/{id}/landing) and estimates 2026-27 GP based on:
  - Last 2-3 seasons' actual GP
  - Injury history (large GP drops)
  - Age curve adjustments

Writes back to app/static/lineups_all.json with a `gp_est` field per player.
Also adds `gp_est_note` with a brief explanation.

Usage:
    .\.venv\Scripts\python.exe .\scripts\estimate_gp.py
    .\.venv\Scripts\python.exe .\scripts\estimate_gp.py --dry-run   (preview only)
"""

import json
import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

import requests

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
LINEUPS_PATH = os.path.join(REPO_ROOT, 'app', 'static', 'lineups_all.json')

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json',
}

# ── Known injury-prone players (2025-26 major injuries or chronic issues) ──
# These adjustments are applied AFTER the formula-based estimate.
# Format: { playerId: max_gp_estimate }
KNOWN_INJURY_ADJUSTMENTS: Dict[int, int] = {
    # Players who missed significant time in 2025-26 or have chronic issues
    # Connor McDavid - played 63 in 2024-25, 76 in 2023-24 — generally durable
    # Leon Draisaitl - very durable (80+ regularly)
    # Sidney Crosby - aging but still plays 75+
    # Alex Ovechkin - aging
    # Gabriel Landeskog - hasn't played since 2022, attempting comeback
    8476455: 55,  # Landeskog - huge injury risk, missed multiple seasons
    # Evgeni Malkin - aging, declining GP
    8471215: 65,
    # Kris Letang - aging
    8471724: 68,
    # Carey Price - retired/inactive
    # Shea Weber - retired
    # Mark Stone - recurring back issues
    8475913: 55,
    # Max Pacioretty - multiple achilles tears
    8474157: 45,
    # Patrik Laine - mental health / injury history
    8479339: 55,
    # Frederik Andersen - recurring injuries
    8475883: 35,
    # John Gibson - injury history
    8476434: 45,
    # Antti Raanta - injury prone
    # Jake Muzzin - retired/neck
    # Ryan Ellis - career likely over
    # Carey Price - inactive
}

# Known retiring / unlikely to play much in 2026-27
PLAYERS_WINDING_DOWN: set = {
    # Players who may retire or have very limited roles
    8471675,  # Crosby - still going but age 39
    8471215,  # Malkin - age 40
    8470613,  # Brent Burns - age 41
    8471734,  # Jonathan Quick - age 40
    8474590,  # John Carlson - age 36
    8474564,  # Steven Stamkos - age 36
    8475794,  # Tyler Seguin - hip issues
    8473986,  # Alex Killorn - age 36
    8475172,  # Nazem Kadri - age 35
    8475765,  # Vladimir Tarasenko - age 34
    8476468,  # J.T. Miller - age 33
    8473507,  # Jeff Petry - age 37
    8474612,  # Travis Hamonic - age 36
    8474090,  # Brendan Smith - age 37
    8475208,  # Brian Dumoulin - age 35
    8476879,  # Cody Ceci - age 32
    8477496,  # Elias Lindholm - age 31 (not old but decline)
    8476459,  # Mika Zibanejad - age 33
}


def fetch_player_landing(pid: int, timeout: int = 20) -> Optional[Dict]:
    """Fetch /v1/player/{pid}/landing and return JSON or None."""
    url = f"https://api-web.nhle.com/v1/player/{pid}/landing"
    try:
        r = requests.get(url, timeout=timeout, headers=REQUEST_HEADERS)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None


def extract_gp_history(landing: Dict) -> List[int]:
    """Extract regular-season NHL GP from seasonTotals, most recent first.
    
    Filters: leagueAbbrev=='NHL', gameTypeId==2 (regular season).
    Groups by season, taking the max GP within each season.
    """
    season_totals = landing.get('seasonTotals') or []
    
    # Group by season, collect NHL regular season entries
    from collections import defaultdict
    season_gp: Dict[int, int] = defaultdict(int)
    for entry in season_totals:
        if entry.get('leagueAbbrev') != 'NHL':
            continue
        if entry.get('gameTypeId') != 2:  # 2 = regular season
            continue
        season = entry.get('season')
        gp = entry.get('gamesPlayed')
        if season is not None and gp is not None and gp > 0:
            season = int(season)
            # Sum GP for traded players (split across teams in same season)
            season_gp[season] += gp
    
    # Sort by season descending (most recent first)
    gp_list = [gp for season, gp in sorted(season_gp.items(), reverse=True)]
    return gp_list


def estimate_gp_2027(gp_history: List[int], player_id: int, pos: str, age: Optional[int] = None) -> tuple:
    """
    Estimate 2026-27 games played.

    Heuristic:
    1. Average of last 2 seasons (weighted: 0.6 × last + 0.4 × second last)
    2. If only 1 season, use that
    3. If no history, default based on position (F:75, D:72, G:40)
    4. Apply age penalty: over 35, reduce by ~3 games per year
    5. Apply known injury adjustments
    6. Cap at 82

    Returns (gp_est, note_string).
    """
    note_parts = []

    # Base estimate from history
    if len(gp_history) >= 3:
        # Weighted: last season 50%, second 30%, third 20%
        base = int(0.50 * gp_history[0] + 0.30 * gp_history[1] + 0.20 * gp_history[2])
        note_parts.append(f"wtd-avg last 3: {gp_history[:3]}")
    elif len(gp_history) == 2:
        base = int(0.6 * gp_history[0] + 0.4 * gp_history[1])
        note_parts.append(f"wtd-avg last 2: {gp_history[:2]}")
    elif len(gp_history) == 1:
        base = gp_history[0]
        note_parts.append(f"last season: {gp_history[0]}")
    else:
        # No history — rookie or no NHL games
        base = {'F': 72, 'D': 68, 'G': 35}.get(pos, 70)
        note_parts.append(f"no-NHL-history default:{base}")

    # Age adjustment (if we have age)
    if age and age > 32:
        penalty = max(0, (age - 32) * 2)
        base -= penalty
        note_parts.append(f"age{age}:-{penalty}")

    # Floor at 10 (even injury-prone players might play some)
    base = max(10, base)

    # Cap at 82
    base = min(82, base)

    # Known injury adjustments (override if lower)
    if player_id in KNOWN_INJURY_ADJUSTMENTS:
        injury_cap = KNOWN_INJURY_ADJUSTMENTS[player_id]
        if base > injury_cap:
            note_parts.append(f"injury-cap:{injury_cap}")
            base = injury_cap

    # Winding down players
    if player_id in PLAYERS_WINDING_DOWN:
        note_parts.append("aging-vet")

    note = '; '.join(note_parts) if note_parts else 'default-82'

    return base, note


def main():
    ap = argparse.ArgumentParser(description="Add GP estimates to lineups_all.json")
    ap.add_argument('--dry-run', action='store_true', help='Preview only, do not write')
    ap.add_argument('--max-workers', type=int, default=12, help='Parallel fetches (default 12)')
    ap.add_argument('--input', help='Input JSON path (default app/static/lineups_all.json)')
    ap.add_argument('--output', help='Output JSON path (default same as input)')
    args = ap.parse_args()

    input_path = args.input or LINEUPS_PATH
    output_path = args.output or input_path

    print(f"Reading lineups from {input_path}...")
    with open(input_path, 'r', encoding='utf-8') as f:
        lineups = json.load(f)

    # Collect unique player IDs with their position
    player_info: Dict[int, str] = {}  # pid -> pos
    for team_abbrev, team_data in lineups.items():
        for group_key in ('forwards', 'defense', 'goalies'):
            for player in team_data.get(group_key, []):
                pid = player.get('playerId')
                if pid and pid not in player_info:
                    player_info[pid] = player.get('pos', 'F')

    print(f"Unique players to estimate: {len(player_info)}")

    # Fetch all player stats in parallel
    gp_estimates: Dict[int, tuple] = {}  # pid -> (gp_est, note)

    def fetch_and_estimate(pid: int) -> tuple:
        landing = fetch_player_landing(pid)
        gp_history = extract_gp_history(landing) if landing else []
        pos = player_info.get(pid, 'F')
        age = None
        if landing:
            bday = landing.get('birthDate')
            if bday:
                try:
                    birth = datetime.strptime(bday, '%Y-%m-%d')
                    # Age at start of 2026-27 season (Oct 2026)
                    season_start = datetime(2026, 10, 1)
                    age = season_start.year - birth.year - ((season_start.month, season_start.day) < (birth.month, birth.day))
                except Exception:
                    pass
        gp_est, note = estimate_gp_2027(gp_history, pid, pos, age)
        return pid, gp_est, note

    print(f"Fetching stats (workers={args.max_workers})...")
    completed = 0
    total = len(player_info)

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {executor.submit(fetch_and_estimate, pid): pid for pid in player_info}
        for future in as_completed(futures):
            try:
                pid, gp_est, note = future.result()
                gp_estimates[pid] = (gp_est, note)
                completed += 1
                if completed % 50 == 0:
                    print(f"  Progress: {completed}/{total}")
            except Exception as e:
                pid = futures[future]
                gp_estimates[pid] = (82, f"error:{e}")
                completed += 1

    print(f"  Complete: {completed}/{total}")

    # Apply estimates to lineups
    updates = 0
    for team_abbrev, team_data in lineups.items():
        for group_key in ('forwards', 'defense', 'goalies'):
            for player in team_data.get(group_key, []):
                pid = player.get('playerId')
                if pid and pid in gp_estimates:
                    gp_est, note = gp_estimates[pid]
                    player['gp_est'] = gp_est
                    player['gp_est_note'] = note
                    updates += 1

    print(f"Applied GP estimates to {updates} player entries")

    # Update generated_at
    now = datetime.now(timezone.utc).isoformat()
    for team_data in lineups.values():
        team_data['generated_at'] = now

    if args.dry_run:
        print("\n--- DRY RUN (not writing) ---")
        # Show sample
        for team_abbrev in list(lineups.keys())[:3]:
            team_data = lineups[team_abbrev]
            print(f"\n{team_abbrev}:")
            for group_key in ('forwards', 'defense', 'goalies'):
                for p in team_data.get(group_key, [])[:2]:
                    print(f"  {p['name']}: gp_est={p.get('gp_est','?')} ({p.get('gp_est_note','')})")
    else:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # Write with backup
        backup_path = output_path + '.bak'
        if os.path.exists(output_path):
            try:
                os.replace(output_path, backup_path)
                print(f"Backed up to {backup_path}")
            except Exception:
                pass

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(lineups, f, ensure_ascii=False, separators=(',', ':'))
        print(f"Written to {output_path}")

        # Summary stats
        all_gp = [p.get('gp_est', 82) for t in lineups.values()
                  for g in ('forwards', 'defense', 'goalies')
                  for p in t.get(g, [])]
        if all_gp:
            print(f"\nGP estimate distribution ({len(all_gp)} players):")
            print(f"  Min: {min(all_gp)}, Max: {max(all_gp)}, Mean: {sum(all_gp)/len(all_gp):.1f}")


if __name__ == '__main__':
    main()
