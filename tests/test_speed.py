"""Speed test: single team shift fetch with PAGE=5000."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
os.chdir(os.path.join(os.path.dirname(__file__), '..'))
from dotenv import load_dotenv
load_dotenv()
os.environ['XG_PRELOAD'] = '0'

from app import create_app
app = create_app()

with app.app_context():
    from app.routes import _get_lt_shifts, _LT_SHIFTS_CACHE
    _LT_SHIFTS_CACHE.clear()
    
    t0 = time.time()
    rows = _get_lt_shifts('EDM', '20242025')
    t1 = time.time()
    print(f"EDM: {len(rows)} shifts in {t1-t0:.1f}s (cold)", flush=True)
    
    _LT_SHIFTS_CACHE.clear()
    t0 = time.time()
    rows = _get_lt_shifts('TOR', '20242025')
    t1 = time.time()
    print(f"TOR: {len(rows)} shifts in {t1-t0:.1f}s (cold)", flush=True)

print("Done.", flush=True)
