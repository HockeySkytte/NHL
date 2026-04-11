"""Quick test: verify league scope logic works by fetching only a few teams."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()
os.environ['XG_PRELOAD'] = '0'

from app import create_app
app = create_app()

with app.app_context():
    from app.routes import _get_lt_shifts, _filter_shifts_season_state, _sb_read

    season = '20242025'
    test_teams = ['EDM', 'TOR', 'NYR']

    # Step 1: Fetch shifts for each team
    for team in test_teams:
        t0 = time.time()
        rows = _get_lt_shifts(team, season)
        t1 = time.time()
        print(f"{team}: {len(rows)} shifts in {t1-t0:.1f}s", flush=True)

    # Step 2: Get player info
    player_rows = _sb_read('players', columns='player_id,player,position',
                           filters={'season': f'eq.{season}'})
    pid_info = {}
    if player_rows:
        for r in player_rows:
            pid_info[str(r.get('player_id', ''))] = {
                'name': r.get('player', ''),
                'position': r.get('position', ''),
            }
    print(f"\nPlayers: {len(pid_info)}", flush=True)

    # Step 3: Build combos (fwd, 5v5, regular)
    target_positions = {'C', 'L', 'R'}
    combo_size = 3
    combo_groups = {}

    for team in test_teams:
        t_rows = _get_lt_shifts(team, season)
        t_rows = _filter_shifts_season_state(t_rows, 'regular')
        t_rows = [s for s in t_rows if str(s.get('strength_state', '')) == '5v5']
        print(f"{team}: {len(t_rows)} filtered shifts", flush=True)

        for s in t_rows:
            pids_on_ice = str(s.get('player_id', '')).split()
            line_pids = sorted(
                pid for pid in pids_on_ice
                if pid_info.get(pid, {}).get('position', '') in target_positions
            )
            if len(line_pids) != combo_size:
                continue
            key = (team, tuple(line_pids))
            grp = combo_groups.get(key)
            if grp is None:
                grp = {'duration': 0, 'game_ids': set(), 'team': team}
                combo_groups[key] = grp
            gid = int(s.get('game_id', 0))
            dur = int(s.get('duration', 0) or 0)
            grp['duration'] += dur
            grp['game_ids'].add(gid)

    print(f"\nTotal combos: {len(combo_groups)}", flush=True)
    teams_found = set(k[0] for k in combo_groups)
    print(f"Teams in combos: {sorted(teams_found)}", flush=True)

    # Show top 3 per team
    for team in test_teams:
        team_combos = [(k, v) for k, v in combo_groups.items() if k[0] == team]
        team_combos.sort(key=lambda x: x[1]['duration'], reverse=True)
        print(f"\n{team} top 3:", flush=True)
        for (t, pids), grp in team_combos[:3]:
            toi = grp['duration'] / 60
            gp = len(grp['game_ids'])
            names = [pid_info.get(p, {}).get('name', p) for p in pids]
            print(f"  {' - '.join(names)}: TOI={toi:.1f} GP={gp}", flush=True)

print("\nDone.", flush=True)
