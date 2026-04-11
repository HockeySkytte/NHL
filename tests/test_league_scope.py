"""Test league scope using Flask test_client (no server needed)."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()
os.environ['XG_PRELOAD'] = '0'

from app import create_app
app = create_app()
client = app.test_client()

# Test 1: Team scope
print("1. Team scope (EDM, fwd)...", flush=True)
t0 = time.time()
r = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=team')
t1 = time.time()
d = r.get_json()
n = len(d.get('combos', []))
print(f"   {t1-t0:.1f}s  Combos:{n}  Players:{len(d.get('players',{}))}", flush=True)
if n:
    c = d['combos'][0]
    print(f"   Top: team={c.get('team')} toi={c.get('toi')} gp={c.get('gp')} cf={c.get('cf')}", flush=True)

# Test 2: League scope (with parallel fetch)
print("\n2. League scope (fwd, cold cache)...", flush=True)
t0 = time.time()
r = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=league')
t1 = time.time()
d = r.get_json()
n = len(d.get('combos', []))
print(f"   {t1-t0:.1f}s  Combos:{n}  Players:{len(d.get('players',{}))}", flush=True)
if n:
    c = d['combos'][0]
    print(f"   Top: team={c.get('team')} toi={c.get('toi')} gp={c.get('gp')}", flush=True)
    teams = set(c2.get('team') for c2 in d['combos'])
    print(f"   Distinct teams: {len(teams)}", flush=True)
else:
    print("   *** 0 combos! ***", flush=True)

# Test 3: Cached league (should be fast)
print("\n3. League scope (fwd, warm cache)...", flush=True)
t0 = time.time()
r = client.get('/api/line-tool/lines?team=EDM&season=20242025&type=fwd&scope=league')
t1 = time.time()
d = r.get_json()
n = len(d.get('combos', []))
print(f"   {t1-t0:.1f}s  Combos:{n}", flush=True)

print("\nDone.", flush=True)
