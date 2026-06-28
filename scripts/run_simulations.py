#!/usr/bin/env python
"""Run N full-season simulations and save results to CSV.

Usage:
    python scripts/run_simulations.py --sims 100 --season 20252026
    python scripts/run_simulations.py --sims 100 --season 20252026 --seed 42

Outputs (in data/simulations/):
    sim_teams_{timestamp}.csv     — one row per team per sim
    sim_players_{timestamp}.csv   — one row per player per sim (split regular/playoff)

The script imports the Flask app's internal simulation functions directly
(no HTTP). Auth is bypassed by clearing SUPABASE auth env vars before
create_app() — the before_app_request guard checks _auth_enabled() which
returns False without those vars. Data loading still works because
SUPABASE_URL / SUPABASE_ANON_KEY remain set.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

# ── Path & env setup (must happen before importing app) ──────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# Disable auth guard (we call functions directly, not HTTP, so this is
# just a safety net for create_app() side effects).
os.environ.setdefault('XG_PRELOAD', '0')

from app import create_app  # noqa: E402
from app import routes as R  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────
TEAM_CSV_COLS = [
    'sim', 'team', 'conference',
    'gp', 'wins', 'losses', 'ot_losses', 'points',
    'goals_for', 'goals_against', 'goal_differential',
    'playoffs', 'second_round', 'third_round', 'final', 'champion',
]

PLAYER_CSV_COLS = [
    'sim', 'seasonstage', 'pid', 'name', 'team', 'position',
    'goals', 'a1', 'a2', 'points',
]


def _build_base_data(season: int):
    """Load all the data that's constant across simulations (lineups, proj map, schedule).

    Returns (teams, team_proj_map, lineups_all, proj_map, custom_lineups,
             schedule, b2b_sets, team_rosters).
    """
    print('  Loading lineups & projections ...', end=' ', flush=True)
    t0 = time.time()
    lineups_all = R._load_lineups_all()
    proj_map = R._load_v2_player_projections_cached()
    # No custom lineups in batch mode (would need per-user request context).
    custom_lineups = {}
    team_proj_map = R._team_proj_map_for_season(season, lineups_all, proj_map, custom_lineups)
    print(f'{time.time() - t0:.1f}s')

    teams = R._active_team_abbrevs()
    if len(teams) < 2:
        raise RuntimeError('Not enough active teams found.')

    print('  Fetching schedules (parallel) ...', end=' ', flush=True)
    t0 = time.time()
    team_games = R._fetch_all_schedules_parallel(season, teams)
    b2b_sets = R._b2b_date_sets(team_games)
    print(f'{time.time() - t0:.1f}s')

    # Deduped regular-season schedule
    by_id = {}
    for t in teams:
        for g in (team_games.get(t) or []):
            if int(g.get('gameType') or 0) != 2:
                continue
            gid = g.get('id')
            if gid is None or gid in by_id:
                continue
            by_id[gid] = g
    schedule = list(by_id.values())
    schedule.sort(key=lambda x: (str(x.get('date') or ''), str(x.get('id') or '')))
    if not schedule:
        raise RuntimeError('No schedule found for season.')
    print(f'  Schedule: {len(schedule)} regular-season games')

    # Per-team rosters for scorer simulation
    print('  Building team rosters ...', end=' ', flush=True)
    t0 = time.time()
    team_rosters = {}
    for t in teams:
        custom = custom_lineups.get(t)
        team_rosters[t] = R._build_team_roster_rates(t, lineups_all, proj_map, custom)
    print(f'{time.time() - t0:.1f}s')

    return (teams, team_proj_map, lineups_all, proj_map, custom_lineups,
            schedule, b2b_sets, team_rosters)


def _simulate_regular_season(
    schedule, team_proj_map, b2b_sets, team_rosters, season, rng
) -> List[Dict[str, Any]]:
    """Simulate all regular-season games. Returns list of game result dicts."""
    results = []
    lg = R._V2_LG_AVG.get(str(season), 3.0)
    sqrt2 = math.sqrt(2.0)
    for g in schedule:
        home = str(g.get('home') or '').strip().upper()
        away = str(g.get('away') or '').strip().upper()
        date_iso = str(g.get('date') or '').strip()
        if not home or not away or home not in team_proj_map or away not in team_proj_map:
            continue
        home_proj = float(team_proj_map.get(home) or 0.0)
        away_proj = float(team_proj_map.get(away) or 0.0)
        hb = 1 if (date_iso in b2b_sets.get(home, set())) else 0
        ab = 1 if (date_iso in b2b_sets.get(away, set())) else 0
        sit = R._V2_SITUATION.get((hb, ab), 0.0)
        mu = R._V2_CONSERVATIVE_WEIGHT * (home_proj - away_proj + sit)
        # Goal means and win probability use the same shrunk mu as
        # _v2_win_probability so the sim average matches the KPI.
        gf_home = max(0.5, lg + mu / 2.0)
        gf_away = max(0.5, lg - mu / 2.0)
        sigma = math.sqrt(gf_home + gf_away)
        p_home = 0.5 * (1.0 + math.erf(mu / (sigma * sqrt2)))

        gh = R._poisson_draw(gf_home, rng)
        ga = R._poisson_draw(gf_away, rng)
        if gh > ga:
            winner, loser, ot = home, away, False
        elif ga > gh:
            winner, loser, ot = away, home, False
        else:
            ot = True
            if rng.random() < p_home:
                winner, loser = home, away
            else:
                winner, loser = away, home

        # Convert some 1-goal regulation wins to OT wins (game was tied
        # after 60 min).  Poisson ties ~15% but real NHL OT rate ~25%.
        if not ot and abs(gh - ga) == 1 and rng.random() < 0.30:
            ot = True

        home_scorers = R._simulate_goal_scorers(team_rosters.get(home, []), int(gh), rng)
        away_scorers = R._simulate_goal_scorers(team_rosters.get(away, []), int(ga), rng)

        results.append({
            'home': home, 'away': away, 'winner': winner, 'loser': loser,
            'ot': bool(ot),
            'homeGoals': int(gh), 'awayGoals': int(ga),
            'homePoints': 2 if winner == home else (1 if ot else 0),
            'awayPoints': 2 if winner == away else (1 if ot else 0),
            'homeScorers': home_scorers,
            'awayScorers': away_scorers,
        })
    return results


def _aggregate_player_stats(
    results: List[Dict[str, Any]],
    proj_map: Dict[int, Dict[str, Any]],
    seasonstage: str,
) -> Dict[int, Dict[str, Any]]:
    """Aggregate player goal/assist stats from a list of game results."""
    ps: Dict[int, Dict[str, Any]] = {}
    for g in results:
        for scorers_list, team_abbr in ((g.get('homeScorers') or [], g['home']),
                                        (g.get('awayScorers') or [], g['away'])):
            for sg in scorers_list:
                for role, pid in (('scorer', sg.get('scorer')),
                                  ('a1', sg.get('a1')),
                                  ('a2', sg.get('a2'))):
                    if not pid:
                        continue
                    if pid not in ps:
                        row = proj_map.get(pid) or {}
                        ps[pid] = {
                            'pid': int(pid),
                            'name': str(row.get('player') or ''),
                            'team': str(row.get('team') or team_abbr),
                            'position': str(row.get('position') or ''),
                            'goals': 0, 'a1': 0, 'a2': 0, 'points': 0,
                        }
                    p = ps[pid]
                    if role == 'scorer':
                        p['goals'] += 1; p['points'] += 1
                    elif role == 'a1':
                        p['a1'] += 1; p['points'] += 1
                    elif role == 'a2':
                        p['a2'] += 1; p['points'] += 1
    return ps


def _simulate_playoff_with_scorers(
    seeds: Dict[str, List[str]],
    team_proj_map: Dict[str, float],
    team_rosters: Dict[str, List[Dict[str, Any]]],
    season: int,
    rng: random.Random,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Simulate the full playoff bracket and track per-game results with scorers.

    Returns (playoffs_dict, playoff_game_results).
    """
    lg = R._V2_LG_AVG.get(str(season), 3.0)
    sqrt2 = math.sqrt(2.0)
    playoff_games: List[Dict[str, Any]] = []

    def sim_series(top, bottom):
        top_wins = 0
        bottom_wins = 0
        games_played = 0
        home_pattern = [top, top, bottom, bottom, top, bottom, top]
        max_games = 7
        while top_wins < 4 and bottom_wins < 4 and games_played < max_games:
            home = home_pattern[games_played]
            away = bottom if home == top else top
            hp = float(team_proj_map.get(home) or 0.0)
            ap = float(team_proj_map.get(away) or 0.0)
            p_home = 0.5 * (1.0 + math.erf((hp - ap) / (math.sqrt(max(0.5, lg + hp / 2.0) + max(0.5, lg + ap / 2.0)) * sqrt2)))
            gh = R._poisson_draw(max(0.5, lg + hp / 2.0), rng)
            ga = R._poisson_draw(max(0.5, lg + ap / 2.0), rng)
            if gh > ga:
                winner = home
            elif ga > gh:
                winner = away
            else:
                if rng.random() < p_home:
                    winner = home
                else:
                    winner = away
            # Record scorers for this playoff game
            home_scorers = R._simulate_goal_scorers(team_rosters.get(home, []), int(gh), rng)
            away_scorers = R._simulate_goal_scorers(team_rosters.get(away, []), int(ga), rng)
            playoff_games.append({
                'home': home, 'away': away, 'winner': winner,
                'homeGoals': int(gh), 'awayGoals': int(ga),
                'homeScorers': home_scorers, 'awayScorers': away_scorers,
            })
            if winner == top:
                top_wins += 1
            else:
                bottom_wins += 1
            games_played += 1
        winner = top if top_wins == 4 else bottom
        loser = bottom if winner == top else top
        return {
            'top': top, 'bottom': bottom,
            'winner': winner, 'loser': loser,
            'topWins': top_wins, 'bottomWins': bottom_wins,
            'games': games_played,
        }

    def conf_round(team_list):
        return [
            sim_series(team_list[0], team_list[7]),
            sim_series(team_list[1], team_list[6]),
            sim_series(team_list[2], team_list[5]),
            sim_series(team_list[3], team_list[4]),
        ]

    east_r1 = conf_round(seeds['East'])
    west_r1 = conf_round(seeds['West'])

    def reseeds(round_results, original_seeds):
        remain = [s['winner'] for s in round_results]
        idx = {t: i for i, t in enumerate(original_seeds)}
        remain.sort(key=lambda t: idx.get(t, 99))
        return remain

    east_remain = reseeds(east_r1, seeds['East'])
    west_remain = reseeds(west_r1, seeds['West'])
    east_r2 = [sim_series(east_remain[0], east_remain[3]),
               sim_series(east_remain[1], east_remain[2])]
    west_r2 = [sim_series(west_remain[0], west_remain[3]),
               sim_series(west_remain[1], west_remain[2])]

    east_final_seeds = reseeds(east_r2, east_remain)
    west_final_seeds = reseeds(west_r2, west_remain)
    east_conf = sim_series(east_final_seeds[0], east_final_seeds[1])
    west_conf = sim_series(west_final_seeds[0], west_final_seeds[1])
    stanley = sim_series(east_conf['winner'], west_conf['winner'])

    playoffs = {
        'round1': {'East': east_r1, 'West': west_r1},
        'round2': {'East': east_r2, 'West': west_r2},
        'conferenceFinals': {'East': east_conf, 'West': west_conf},
        'stanleyFinal': stanley,
        'champion': stanley['winner'],
    }
    return playoffs, playoff_games


def _playoff_stage_flags(playoffs: Dict[str, Any], team: str) -> Dict[str, int]:
    """Determine how far a team got in the playoffs.

    Returns {playoffs, second_round, third_round, final, champion} as 0/1.
    """
    made_playoffs = False
    second_round = False
    third_round = False
    final = False
    champion = False

    # Round 1
    for conf in ('East', 'West'):
        for s in (playoffs.get('round1') or {}).get(conf) or []:
            if s.get('winner') == team or s.get('loser') == team:
                made_playoffs = True
                break

    # Round 2
    for conf in ('East', 'West'):
        for s in (playoffs.get('round2') or {}).get(conf) or []:
            if s.get('winner') == team or s.get('loser') == team:
                second_round = True
                break

    # Conference Finals (3rd round)
    for conf in ('East', 'West'):
        s = (playoffs.get('conferenceFinals') or {}).get(conf)
        if s and (s.get('winner') == team or s.get('loser') == team):
            third_round = True

    # Stanley Cup Final
    s = playoffs.get('stanleyFinal')
    if s and (s.get('winner') == team or s.get('loser') == team):
        final = True
        if s.get('winner') == team:
            champion = True

    return {
        'playoffs': 1 if made_playoffs else 0,
        'second_round': 1 if second_round else 0,
        'third_round': 1 if third_round else 0,
        'final': 1 if final else 0,
        'champion': 1 if champion else 0,
    }


def _output_dir() -> str:
    d = os.path.join(REPO_ROOT, 'data', 'simulations')
    os.makedirs(d, exist_ok=True)
    return d


def main():
    parser = argparse.ArgumentParser(description='Run N season simulations and save CSVs.')
    parser.add_argument('--sims', type=int, default=100, help='Number of simulations (default 100).')
    parser.add_argument('--season', type=int, default=20252026, help='Season ID (default 20252026).')
    parser.add_argument('--seed', type=int, default=None, help='Base RNG seed (default random).')
    parser.add_argument('--out-suffix', type=str, default='', help='Suffix for output filenames.')
    args = parser.parse_args()

    n_sims = max(1, args.sims)
    season = args.season
    base_seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)

    print(f'=== Season Simulation Batch ===')
    print(f'  sims={n_sims}  season={season}  base_seed={base_seed}')

    # Create app (needed for Supabase client init / app context)
    print('  Creating Flask app ...', end=' ', flush=True)
    t0 = time.time()
    app = create_app()
    print(f'{time.time() - t0:.1f}s')

    # Load base data once (shared across all sims)
    with app.app_context():
        base = _build_base_data(season)
    (teams, team_proj_map, lineups_all, proj_map, custom_lineups,
     schedule, b2b_sets, team_rosters) = base

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    suffix = f'_{args.out_suffix}' if args.out_suffix else ''
    teams_path = os.path.join(_output_dir(), f'sim_teams_{ts}{suffix}.csv')
    players_path = os.path.join(_output_dir(), f'sim_players_{ts}{suffix}.csv')

    team_rows_out: List[Dict[str, Any]] = []
    player_rows_out: List[Dict[str, Any]] = []

    sim_total_t0 = time.time()
    for i in range(n_sims):
        sim_num = i + 1
        sim_seed = base_seed + i
        rng = random.Random(sim_seed)

        if sim_num % 10 == 0 or sim_num == 1:
            elapsed = time.time() - sim_total_t0
            print(f'  sim {sim_num}/{n_sims} ... (avg {elapsed / sim_num:.2f}s/sim)', flush=True)

        # Regular season
        reg_results = _simulate_regular_season(schedule, team_proj_map, b2b_sets, team_rosters, season, rng)

        # Standings + playoff seeding
        standings = R._standings_from_results(teams, reg_results)
        seeds = R._seed_playoffs(standings)
        if len(seeds['East']) < 8 or len(seeds['West']) < 8:
            top16 = [r['team'] for r in standings[:16]]
            seeds = {'East': top16[:8], 'West': top16[8:16]}

        # Playoffs (with scorers)
        playoffs, playoff_games = _simulate_playoff_with_scorers(
            seeds, team_proj_map, team_rosters, season, rng,
        )

        # ── Team CSV rows ──
        for row in standings:
            flags = _playoff_stage_flags(playoffs, row['team'])
            team_rows_out.append({
                'sim': sim_num,
                'team': row['team'],
                'conference': row.get('conference', ''),
                'gp': row['gp'],
                'wins': row['wins'],
                'losses': row['losses'],
                'ot_losses': row['otLosses'],
                'points': row['points'],
                'goals_for': row['goalsFor'],
                'goals_against': row['goalsAgainst'],
                'goal_differential': row.get('goalDifferential', row['goalsFor'] - row['goalsAgainst']),
                **flags,
            })

        # ── Player CSV rows ──
        # Regular season
        reg_ps = _aggregate_player_stats(reg_results, proj_map, 'regular')
        for pid, p in reg_ps.items():
            player_rows_out.append({
                'sim': sim_num,
                'seasonstage': 'regular',
                'pid': p['pid'],
                'name': p['name'],
                'team': p['team'],
                'position': p['position'],
                'goals': p['goals'],
                'a1': p['a1'],
                'a2': p['a2'],
                'points': p['points'],
            })
        # Playoffs
        po_ps = _aggregate_player_stats(playoff_games, proj_map, 'playoff')
        for pid, p in po_ps.items():
            player_rows_out.append({
                'sim': sim_num,
                'seasonstage': 'playoff',
                'pid': p['pid'],
                'name': p['name'],
                'team': p['team'],
                'position': p['position'],
                'goals': p['goals'],
                'a1': p['a1'],
                'a2': p['a2'],
                'points': p['points'],
            })

    total_elapsed = time.time() - sim_total_t0
    print(f'  {n_sims} sims done in {total_elapsed:.1f}s ({total_elapsed / n_sims:.2f}s/sim)')

    # Write CSVs
    print(f'  Writing {teams_path} ...', end=' ', flush=True)
    t0 = time.time()
    with open(teams_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=TEAM_CSV_COLS)
        w.writeheader()
        w.writerows(team_rows_out)
    print(f'{len(team_rows_out)} rows ({time.time() - t0:.1f}s)')

    print(f'  Writing {players_path} ...', end=' ', flush=True)
    t0 = time.time()
    with open(players_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=PLAYER_CSV_COLS)
        w.writeheader()
        w.writerows(player_rows_out)
    print(f'{len(player_rows_out)} rows ({time.time() - t0:.1f}s)')

    print('\nDone.')
    print(f'  Team CSV:   {teams_path}')
    print(f'  Player CSV: {players_path}')


if __name__ == '__main__':
    main()
