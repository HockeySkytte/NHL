"""Quick test: verify league scope logic works by fetching only a few teams."""

import os
import sys
import time


def main() -> None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(os.path.join(os.path.dirname(__file__), '..'))
    from dotenv import load_dotenv

    load_dotenv()
    os.environ['XG_PRELOAD'] = '0'

    from app import create_app

    app = create_app()
    with app.app_context():
        from app.routes import _filter_shifts_season_state, _get_lt_shifts, _sb_read

        season = '20242025'
        test_teams = ['EDM', 'TOR', 'NYR']

        for team in test_teams:
            t0 = time.time()
            rows = _get_lt_shifts(team, season)
            t1 = time.time()
            print(f"{team}: {len(rows)} shifts in {t1 - t0:.1f}s", flush=True)

        player_rows = _sb_read('players', columns='player_id,player,position', filters={'season': f'eq.{season}'})
        pid_info = {}
        if player_rows:
            for row in player_rows:
                pid_info[str(row.get('player_id', ''))] = {
                    'name': row.get('player', ''),
                    'position': row.get('position', ''),
                }
        print(f"\nPlayers: {len(pid_info)}", flush=True)

        target_positions = {'C', 'L', 'R'}
        combo_size = 3
        combo_groups = {}

        for team in test_teams:
            team_rows = _get_lt_shifts(team, season)
            team_rows = _filter_shifts_season_state(team_rows, 'regular')
            team_rows = [shift for shift in team_rows if str(shift.get('strength_state', '')) == '5v5']
            print(f"{team}: {len(team_rows)} filtered shifts", flush=True)

            for shift in team_rows:
                pids_on_ice = str(shift.get('player_id', '')).split()
                line_pids = sorted(
                    pid for pid in pids_on_ice if pid_info.get(pid, {}).get('position', '') in target_positions
                )
                if len(line_pids) != combo_size:
                    continue
                key = (team, tuple(line_pids))
                group = combo_groups.get(key)
                if group is None:
                    group = {'duration': 0, 'game_ids': set(), 'team': team}
                    combo_groups[key] = group
                game_id = int(shift.get('game_id', 0))
                duration = int(shift.get('duration', 0) or 0)
                group['duration'] += duration
                group['game_ids'].add(game_id)

        print(f"\nTotal combos: {len(combo_groups)}", flush=True)
        teams_found = set(key[0] for key in combo_groups)
        print(f"Teams in combos: {sorted(teams_found)}", flush=True)

        for team in test_teams:
            team_combos = [(key, val) for key, val in combo_groups.items() if key[0] == team]
            team_combos.sort(key=lambda x: x[1]['duration'], reverse=True)
            print(f"\n{team} top 3:", flush=True)
            for (team_code, pids), group in team_combos[:3]:
                _ = team_code
                toi = group['duration'] / 60
                gp = len(group['game_ids'])
                names = [pid_info.get(pid, {}).get('name', pid) for pid in pids]
                print(f"  {' - '.join(names)}: TOI={toi:.1f} GP={gp}", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
