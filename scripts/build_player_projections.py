"""
Build nhl_playerprojections and nhl_current_projections CSVs.
Matches MySQL logic exactly, reading from local MySQL tables.

Data: MySQL possession_values_games + player tables.
Coefficients: MySQL model_coefficients.
"""
import os, sys, math, time
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path: sys.path.insert(0, REPO_ROOT)

OUT_DIR = os.path.join(REPO_ROOT, 'data', 'player_projections')
os.makedirs(OUT_DIR, exist_ok=True)

MYSQL_URL = os.getenv('DATABASE_URL', '')
if not MYSQL_URL: raise RuntimeError('DATABASE_URL not set')
eng = create_engine(MYSQL_URL, connect_args={'connect_timeout': 30})

print('Loading data from MySQL …')

# ── 1. possession_values_games (league=1, all seasons) ──────────────
print('  Loading possession_values_games …', end=' ', flush=True); t0 = time.time()
pos = pd.read_sql(text("SELECT * FROM possession_values_games WHERE league = 1"), eng)
# Handle position column: MySQL schema might have both position and position2
if 'position2' in pos.columns and 'position' in pos.columns:
    pos.drop(columns=['position'], inplace=True)
if 'position2' in pos.columns:
    pos.rename(columns={'position2': 'position'}, inplace=True)
# Ensure required columns exist (MySQL table may differ from Supabase)
if 'seasonstage' not in pos.columns:
    pos['seasonstage'] = ''
if 'league' not in pos.columns:
    pos['league'] = 1
print(f'{len(pos):,} rows ({time.time()-t0:.0f}s)')

# Convert gameid to numeric BEFORE any sort/groupby (MySQL stores it as text)
pos['gameid'] = pd.to_numeric(pos['gameid'], errors='coerce').astype('Int64')

# ── 2. player names + NHL API ID mapping ────────────────────────────
print('  Loading player names …', end=' ', flush=True); t0 = time.time()
players = pd.read_sql(text("SELECT player_id, first_name, last_name FROM player"), eng)
players['player_name'] = (players['first_name'].fillna('') + ' ' + players['last_name'].fillna('')).str.strip()
players['player_id'] = pd.to_numeric(players['player_id'], errors='coerce').astype('Int64')
# Build name lookup: playerid -> MySQL name
name_lookup = players.set_index('player_id')['player_name'].to_dict()
print(f'{len(players):,} rows ({time.time()-t0:.0f}s)')

# ── 2b. Match Moncton player IDs to NHL API IDs via Supabase ──────
print('  Matching NHL API player IDs …', end=' ', flush=True); t0 = time.time()
from app.supabase_client import read_table
import re as _re, unicodedata as _uni

_NAME_ALIASES = {
    "yegor chinakhov": "egor chinakhov", "nikita grebyonkin": "nikita grebenkin",
    "mitch marner": "mitchell marner", "chris tanev": "christopher tanev",
    "mathew dumba": "matt dumba", "arsenii sergeev": "arseni sergeev",
    "arseny gritsyuk": "arseni gritsyuk", "bo groulx": "benoit olivier groulx",
    "danil zhilkin": "danny zhilkin", "josh samanski": "joshua samanski",
    "max shabanov": "maxim shabanov", "mike benning": "michael benning",
    "samuel blais": "sammy blais", "matt coronato": "matthew coronato",
    "matt rempe": "matthew rempe", "matt savoie": "matthew savoie",
    "j j moser": "janis jerome moser", "bradly nadeau": "bradley nadeau",
    "viking gustafsson nyberg": "viking gustavsson nyberg",
    "yegor sokolov": "egor sokolov", "tj tynan": "t j tynan",
    "c j suess": "cj suess", "alexander georgiyev": "alexandar georgiev",
    "vladimir tkachyov": "vladimir tkachev", "alexander chmelevski": "sasha chmelevski",
    "cristoval nieves": "boo nieves",
    "mirco muller": "mirco mueller",
    "jean francois berube": "j f berube",
    "turner kent elson": "turner elson",
    "jean christophe beaudin": "j c beaudin",
    "danil yurtaikin": "danil yurtaykin",
    "nikita okhotyuk": "nikita okhotiuk",
    "jeffrey truchon viel": "jeffrey viel",
}

def _norm_name(v):
    t = str(v or "").strip()
    if not t: return ""
    t = _uni.normalize("NFKD", t).encode("ascii", "ignore").decode("ascii")
    t = t.lower().replace("'", "")
    t = _re.sub(r"[^a-z0-9]+", " ", t)
    t = _re.sub(r"\b(jr|sr|ii|iii|iv)\b", " ", t)
    t = " ".join(t.split())
    return _NAME_ALIASES.get(t, t)

def _initial_last(v):
    n = _norm_name(v)
    if not n: return ""
    p = n.split()
    return p[0] if len(p) == 1 else f"{p[0][0]} {' '.join(p[1:])}"

# Load HS players for all relevant seasons
hs_parts = []
for s in ['20192020','20202021','20212022','20222023','20232024','20242025','20252026']:
    try:
        hs = read_table('players', columns='season,player_id,player', filters={'season': f'eq.{s}'})
        if not hs.empty: hs_parts.append(hs)
    except Exception: pass
hs_all = pd.concat(hs_parts, ignore_index=True).drop_duplicates()
hs_all['player_id'] = pd.to_numeric(hs_all['player_id'], errors='coerce').astype('Int64')
hs_all = hs_all.dropna(subset=['player_id'])
hs_all['name_norm'] = hs_all['player'].apply(_norm_name)
hs_all['initial_last'] = hs_all['player'].apply(_initial_last)

# Build per-season lookups
hs_by_season = {s: g for s, g in hs_all.groupby('season')}

# Map Moncton player_id -> (nhl_api_id, nhl_name)
nhl_map = {}
# Pass 1: exact normalized name match
for s, hs in hs_by_season.items():
    lu = hs.groupby('name_norm', as_index=False).agg(nhl_id=('player_id','first'), nhl_name=('player','first'))
    lu.index = lu['name_norm']
    for pid, moncton_name in name_lookup.items():
        if pid in nhl_map: continue
        nn = _norm_name(moncton_name)
        if nn in lu.index:
            nhl_map[pid] = (int(lu.at[nn, 'nhl_id']), lu.at[nn, 'nhl_name'])
# Pass 2: initial+last name
for s, hs in hs_by_season.items():
    lu = hs.groupby('initial_last', as_index=False).agg(nhl_id=('player_id','first'), nhl_name=('player','first'))
    lu.index = lu['initial_last']
    for pid, moncton_name in name_lookup.items():
        if pid in nhl_map: continue
        il = _initial_last(moncton_name)
        if il in lu.index:
            nhl_map[pid] = (int(lu.at[il, 'nhl_id']), lu.at[il, 'nhl_name'])

unmapped = len(name_lookup) - len(nhl_map)
print(f'{len(nhl_map):,} mapped ({unmapped} unmapped) ({time.time()-t0:.0f}s)')
# Build lookup dicts with plain int keys (safe for .map())
nhl_id_dict = {int(k): v[0] for k, v in nhl_map.items()}
nhl_name_dict = {int(k): v[1] for k, v in nhl_map.items()}

# ═══════════════════════════════════════════════════════════════════════
RATE_COLS = ['faceoffs','defensive','passes','carries','dump_ins_outs',
             'off_the_puck','gax','gsax','xgf']
# Raw columns (not centred, summed per player-game-strengthstate then rolled)
RAW_COLS = ['ig','a1','a2','ishots']

for c in ['toi','toi_a'] + RATE_COLS + ['xga'] + RAW_COLS:
    pos[c] = pd.to_numeric(pos[c], errors='coerce').fillna(0)

# ═══ STEP 1: League averages ══════════════════════════════════════════
print('  League averages …', end=' ', flush=True); t0 = time.time()
avgs = pos.groupby(['season','position','strengthstate'], as_index=False).agg(
    toi_sum=('toi','sum'), toi_a_sum=('toi_a','sum'),
    **{f'{c}_sum':(c,'sum') for c in RATE_COLS}, xga_sum=('xga','sum'))
for c in RATE_COLS:
    avgs[c] = (avgs[f'{c}_sum'] / avgs['toi_sum'].replace(0,np.nan)).fillna(0)
avgs['xga'] = (avgs['xga_sum'] / avgs['toi_a_sum'].replace(0,np.nan)).fillna(0)
avgs = avgs[['season','position','strengthstate'] + RATE_COLS + ['xga']]
print(f'{len(avgs)} groups ({time.time()-t0:.0f}s)')

# ═══ STEP 2: Center metrics ═══════════════════════════════════════════
print('  Centering …', end=' ', flush=True); t0 = time.time()
p = pos.merge(avgs, on=['season','position','strengthstate'], how='left', suffixes=('','_avg'))
for c in RATE_COLS + ['xga']:
    denom = p['toi'] if c != 'xga' else p['toi_a']
    per_hr = (p[c] / denom.replace(0,np.nan)).fillna(0)
    p[c] = ((per_hr - p[f'{c}_avg'].fillna(0)) * (p['toi'] if c != 'xga' else p['toi_a'])).fillna(0)
centered = p[['season','league','playerid','gameid','team','position',
               'strengthstate','seasonstage','toi'] + RATE_COLS + ['xga'] + RAW_COLS].copy()
print(f'{len(centered)} rows ({time.time()-t0:.0f}s)')

# ═══ STEP 3: Prior games (distinct player-game pairs) ═════════════════
print('  Prior games …', end=' ', flush=True); t0 = time.time()
pg = centered[['playerid','gameid']].drop_duplicates().sort_values(['playerid','gameid'])
pg['prior_games'] = pg.groupby('playerid').cumcount().clip(upper=41)
pg['gameid'] = pd.to_numeric(pg['gameid'], errors='coerce').astype('Int64')
print(f'{len(pg)} player-games ({time.time()-t0:.0f}s)')

# ═══ STEP 4: Group by (player, game, strengthstate) ══════════════════
# MySQL groups WITHOUT seasonstage — get it separately later
print('  Grouping …', end=' ', flush=True); t0 = time.time()
gc = ['season','league','playerid','gameid','team','position','strengthstate']
cg = centered.groupby(gc, as_index=False)[RATE_COLS + ['xga'] + RAW_COLS].sum()
cg['gameid'] = pd.to_numeric(cg['gameid'], errors='coerce').astype('Int64')
# Carry seasonstage as a separate lookup (MySQL gets it from games table later)
seasonstage_map = centered[['playerid','gameid','seasonstage']].drop_duplicates(['playerid','gameid'])
print(f'{len(cg)} rows ({time.time()-t0:.0f}s)')

# ═══ STEP 5: Rolling 41-game averages (per strength state) ═══════════
# MySQL: ROWS BETWEEN 41 PRECEDING AND 1 PRECEDING over metrics_filled.
# Cross-join every player-game to every non-1v0 strength state so that
# each (playerid, strengthstate) partition has exactly one row per game.
# The window then spans 41 prior GAMES of that strength state.
print('  Rolling averages …', end=' ', flush=True); t0 = time.time()
strengthstates = sorted(cg.loc[cg['strengthstate'] != '1v0', 'strengthstate'].dropna().unique())
player_game_bucket = cg[['season','league','playerid','gameid','team','position']].drop_duplicates()
player_game_bucket = player_game_bucket.merge(seasonstage_map, on=['playerid','gameid'], how='left')
state_df = pd.DataFrame({'strengthstate': strengthstates})
metrics_filled = player_game_bucket.merge(state_df, how='cross')
metrics_filled = metrics_filled.merge(
    cg[['playerid','gameid','strengthstate'] + RATE_COLS + ['xga'] + RAW_COLS],
    on=['playerid','gameid','strengthstate'], how='left')
for c in RATE_COLS + ['xga'] + RAW_COLS:
    metrics_filled[c] = metrics_filled[c].fillna(0)

metrics_filled = metrics_filled.merge(pg, on=['playerid','gameid'], how='left')
metrics_filled['prior_games'] = metrics_filled['prior_games'].fillna(0)
metrics_filled = metrics_filled.sort_values(['playerid','strengthstate','gameid'])

# Window by (playerid, strengthstate) — exactly 41 prior games per partition
# Vectorized: apply rolling sum to all metric columns at once per group
roll_cols = RATE_COLS + ['xga'] + RAW_COLS
metrics_filled = metrics_filled.sort_values(['playerid','strengthstate','gameid'])
grp = metrics_filled.groupby(['playerid','strengthstate'])
rolled = grp[roll_cols].transform(lambda x: x.shift(1).rolling(41, min_periods=0).sum())
for c in roll_cols:
    metrics_filled[f'roll_{c}'] = rolled[c]

rm = metrics_filled.copy()
with np.errstate(divide='ignore', invalid='ignore'):
    rm[roll_cols] = np.where(
        rm['prior_games'].values[:, None] > 0,
        rm[[f'roll_{c}' for c in roll_cols]].fillna(0).values / rm['prior_games'].values[:, None],
        0)
rm['rookie_f'] = np.where(rm['position'] == 'F', (41 - rm['prior_games']) / 41, 0)
rm['rookie_d'] = np.where(rm['position'] == 'D', (41 - rm['prior_games']) / 41, 0)
rm['rookie_g'] = np.where(rm['position'] == 'G', (41 - rm['prior_games']) / 41, 0)
print(f'done ({time.time()-t0:.0f}s)')

# ── Export rolling_metrics CSV for comparison with MySQL ──────────
print('  Exporting rolling_metrics CSV …', end=' ', flush=True); t0 = time.time()
RM_EXPORT_COLS = ['season','league','playerid','gameid','team','position',
                  'strengthstate','seasonstage','prior_games'] + RATE_COLS + ['xga'] + RAW_COLS + [
                  'rookie_f','rookie_d','rookie_g']
rm_out = rm[RM_EXPORT_COLS].sort_values(['playerid','strengthstate','gameid'])
rm_out.to_csv(os.path.join(OUT_DIR, 'rolling_metrics.csv'), index=False)
print(f'{len(rm_out)} rows ({time.time()-t0:.0f}s)')

# ═══ STEP 6: nhl_player_metrics ═══════════════════════════════════
# MySQL: LEFT JOIN rolling_metrics (rolling avg) with metrics_filled
#   (per-game centred values, aliased as gs_*).  No coefficients.
print('  Building nhl_player_metrics …', end=' ', flush=True); t0 = time.time()

# Select rolling-avg columns from rm
RM_KEY_COLS = ['season','league','playerid','gameid','team','position',
               'strengthstate','seasonstage','prior_games']
RM_VAL_COLS = RATE_COLS + ['xga'] + RAW_COLS + ['rookie_f','rookie_d','rookie_g']
r = rm[RM_KEY_COLS + RM_VAL_COLS].copy()

# Select per-game columns from metrics_filled, rename to gs_ prefix
f_val_cols = RATE_COLS + ['xga'] + RAW_COLS
f = metrics_filled[['playerid','gameid','strengthstate'] + f_val_cols].copy()
f.columns = ['playerid','gameid','strengthstate'] + [f'gs_{c}' for c in f_val_cols]

npm = r.merge(f, on=['playerid','gameid','strengthstate'], how='left')
# COALESCE gs_ columns to 0
for c in f_val_cols:
    gs_c = f'gs_{c}'
    npm[gs_c] = npm[gs_c].fillna(0)

# ── Add NHL API player ID and name ──
npm['playerid_num'] = pd.to_numeric(npm['playerid'], errors='coerce').fillna(0).astype(int)
npm['nhl_api_player_id'] = npm['playerid_num'].map(nhl_id_dict).astype('Int64')
npm['nhl_player_name'] = npm['playerid_num'].map(nhl_name_dict).fillna('')
npm.drop(columns=['playerid_num'], inplace=True)

npm = npm.sort_values(['playerid','strengthstate','gameid'])
npm.to_csv(os.path.join(OUT_DIR, 'nhl_player_metrics.csv'), index=False)
print(f'{len(npm)} rows ({time.time()-t0:.0f}s)')

# ═══ STEP 7: nhl_current_playerprojections ═══════════════════════
# Per (playerid, strengthstate): take latest game and compute
#   (rolling_avg * prior_games + gs_current - gs_42nd_oldest) / min(prior_games+1, 41)
# Output: 9 rows per player (one per strengthstate), most recent regular-season game info.
print('  Building nhl_current_playerprojections …', end=' ', flush=True); t0 = time.time()

ALL_METRIC_COLS = RATE_COLS + ['xga'] + RAW_COLS
npm = npm.sort_values(['playerid','strengthstate','gameid'])

# ── Latest regular-season game per (playerid, strengthstate) ──
# Filter to regular season only for the "current" snapshot — playoffs have
# tougher competition which skews the projection downward.
reg_mask = npm['seasonstage'].fillna('').str.strip().str.lower().isin(['regular', 'reg', ''])
npm_reg = npm[reg_mask].sort_values(['playerid','strengthstate','gameid'])

# Use regular-season rows for latest + oldest when available; fall back to full npm otherwise
npm_reg = npm_reg.copy()
npm_full = npm.sort_values(['playerid','strengthstate','gameid']).copy()

# Latest game (prefer regular season)
latest_reg = npm_reg.groupby(['playerid','strengthstate'], as_index=False).last()
latest_all = npm_full.groupby(['playerid','strengthstate'], as_index=False).last()
# Keep regular-season latest for keys that have it; fill remainder from overall
latest_keys = set(zip(latest_reg['playerid'], latest_reg['strengthstate']))
fallback_latest = latest_all[~latest_all.apply(
    lambda r: (r['playerid'], r['strengthstate']) in latest_keys, axis=1
)]
latest = pd.concat([latest_reg, fallback_latest], ignore_index=True)

# ── 42nd-most-recent gs_ value (prefer regular-season window) ──
npm_reg['_rn_desc'] = npm_reg.groupby(['playerid','strengthstate']).cumcount(ascending=False)
oldest_mask_reg = npm_reg['_rn_desc'] == 41
oldest_reg = npm_reg.loc[oldest_mask_reg, ['playerid','strengthstate'] + [f'gs_{c}' for c in ALL_METRIC_COLS]].copy()

npm_full['_rn_desc'] = npm_full.groupby(['playerid','strengthstate']).cumcount(ascending=False)
oldest_mask_full = npm_full['_rn_desc'] == 41
oldest_full = npm_full.loc[oldest_mask_full, ['playerid','strengthstate'] + [f'gs_{c}' for c in ALL_METRIC_COLS]].copy()

# Merge: prefer regular-season oldest, fill gaps from overall
oldest_keys = set(zip(oldest_reg['playerid'], oldest_reg['strengthstate']))
fallback_oldest = oldest_full[~oldest_full.apply(
    lambda r: (r['playerid'], r['strengthstate']) in oldest_keys, axis=1
)]
oldest = pd.concat([oldest_reg, fallback_oldest], ignore_index=True)
oldest.columns = ['playerid','strengthstate'] + [f'{c}_oldest' for c in ALL_METRIC_COLS]

cur = latest.merge(oldest, on=['playerid','strengthstate'], how='left')

# ── Compute current projections ──
for c in ALL_METRIC_COLS:
    gs_c = f'gs_{c}'
    oldest_c = f'{c}_oldest'
    if oldest_c not in cur.columns:
        cur[oldest_c] = 0.0
    if gs_c not in cur.columns:
        cur[gs_c] = 0.0
    pg = cur['prior_games'].fillna(0).astype(float)
    gp = (pg + 1).clip(upper=41)
    # Subtract oldest only if we have >=41 prior games
    oldest_subtract = np.where(pg >= 41, cur[oldest_c].fillna(0), 0.0)
    cur[c] = (cur[c].fillna(0) * pg + cur[gs_c].fillna(0) - oldest_subtract) / gp

# ── Rookie ──
cur['rookie'] = np.where(cur['prior_games'] < 41,
                         (41 - cur['prior_games'] - 1) / 41, 0.0)

cur['gp'] = (cur['prior_games'].fillna(0) + 1).clip(upper=41).astype(int)

# ── Output columns ──
CUR_KEY_COLS = ['season','league','playerid','gameid','team','position','strengthstate']
CUR_OUT_COLS = CUR_KEY_COLS + ['nhl_api_player_id','nhl_player_name','gp'] + ALL_METRIC_COLS + ['rookie']
cur_out = cur[CUR_OUT_COLS].sort_values(['playerid','strengthstate'])
cur_out.to_csv(os.path.join(OUT_DIR, 'nhl_current_playerprojections.csv'), index=False)
print(f'{len(cur_out)} rows ({time.time()-t0:.0f}s)')

print('\nDone. Output in data/player_projections/')
