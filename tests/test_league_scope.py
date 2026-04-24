"""Test league scope using Flask test_client (no server needed)."""

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
    client = app.test_client()

    print("1. Team scope (EDM, fwd)...", flush=True)
    t0 = time.time()
    response = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=team')
    t1 = time.time()
    data = response.get_json()
    count = len(data.get('combos', []))
    print(f"   {t1 - t0:.1f}s  Combos:{count}  Players:{len(data.get('players', {}))}", flush=True)
    if count:
        combo = data['combos'][0]
        print(
            f"   Top: team={combo.get('team')} toi={combo.get('toi')} gp={combo.get('gp')} cf={combo.get('cf')}",
            flush=True,
        )

    print("\n2. League scope (fwd, cold cache)...", flush=True)
    t0 = time.time()
    response = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=league')
    t1 = time.time()
    data = response.get_json()
    count = len(data.get('combos', []))
    print(f"   {t1 - t0:.1f}s  Combos:{count}  Players:{len(data.get('players', {}))}", flush=True)
    if count:
        combo = data['combos'][0]
        print(f"   Top: team={combo.get('team')} toi={combo.get('toi')} gp={combo.get('gp')}", flush=True)
        teams = set(entry.get('team') for entry in data['combos'])
        print(f"   Distinct teams: {len(teams)}", flush=True)
    else:
        print("   *** 0 combos! ***", flush=True)

    print("\n3. League scope (fwd, warm cache)...", flush=True)
    t0 = time.time()
    response = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=league')
    t1 = time.time()
    data = response.get_json()
    count = len(data.get('combos', []))
    print(f"   {t1 - t0:.1f}s  Combos:{count}", flush=True)

    print("\nDone.", flush=True)


if __name__ == '__main__':
    main()
