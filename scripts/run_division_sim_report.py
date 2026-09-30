#!/usr/bin/env python
"""Season Monte-Carlo report: 10,000 simulated seasons, graphed by division.

Runs full-season simulations (regular season + Stanley Cup bracket) and renders
one graphic per division containing:

    Team | Projected Points | Standard Error | Playoff Probability |
    Pinnacle Over/Under | Difference (projection - market)

The simulation deliberately reuses the app's own model functions - Poisson goal
draws (`_poisson_draw`), the V2 situation matrix (`_V2_SITUATION`), the
conservative weight, the schedule/B2B loading, standings, seeding and the
playoff bracket (`_simulate_playoffs`) - so the numbers match the deployed
`/api/projections/simulate-season` path rather than re-implementing the model.

Goal-scorer draws are skipped: this report is team-level only. That changes the
RNG stream (so results will not be bit-identical to scripts/run_simulations.py
for a given seed) but not the distribution of game outcomes.

Usage (PowerShell):
    .\\.venv\\Scripts\\python.exe .\\scripts\\run_division_sim_report.py --sims 10000 --season 20262027

Options:
    --odds PATH    CSV with columns team,pinnacle_ou (season point total).
                   If omitted, a template CSV is written and the two market
                   columns are rendered as "-".
    --workers N    Parallel processes (default: cpu_count-1).
    --out DIR      Output directory (default data/simulations).

Outputs (data/simulations/):
    division_<Division>.png        one graphic per division
    division_report_all.png        all four divisions on one page
    division_report_<ts>.csv       the underlying numbers
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(REPO_ROOT, '.env'))

from app import routes as R  # noqa: E402

# Fallback division map (used only if the NHL standings endpoint is unavailable).
DIVISIONS_FALLBACK: Dict[str, List[str]] = {
    'Atlantic': ['BOS', 'BUF', 'DET', 'FLA', 'MTL', 'OTT', 'TBL', 'TOR'],
    'Metropolitan': ['CAR', 'CBJ', 'NJD', 'NYI', 'NYR', 'PHI', 'PIT', 'WSH'],
    'Central': ['CHI', 'COL', 'DAL', 'MIN', 'NSH', 'STL', 'UTA', 'WPG'],
    'Pacific': ['ANA', 'CGY', 'EDM', 'LAK', 'SEA', 'SJS', 'VAN', 'VGK'],
}
DIVISION_ORDER = ['Atlantic', 'Metropolitan', 'Central', 'Pacific']

# Per-worker globals (set by _worker_init so the big shared payload is pickled once).
_W: Dict[str, Any] = {}


# ── base data ────────────────────────────────────────────────────────────────

def load_base(season: int):
    """Load the data that is constant across simulations."""
    print('Loading lineups & player projections ...', flush=True)
    t0 = time.time()
    lineups_all = R._load_lineups_all()
    proj_map = R._load_v2_player_projections_cached(season)
    team_proj_map = R._team_proj_map_for_season(season, lineups_all, proj_map, {})
    print(f'  {len(proj_map)} projections, {len(team_proj_map)} teams ({time.time() - t0:.1f}s)', flush=True)

    if not proj_map:
        raise RuntimeError('Player projection map is empty - refusing to simulate.')

    teams = R._active_team_abbrevs()
    print(f'Fetching schedules for {len(teams)} teams (parallel) ...', flush=True)
    t0 = time.time()
    team_games = R._fetch_all_schedules_parallel(season, teams)
    b2b_sets = R._b2b_date_sets(team_games)

    by_id: Dict[Any, Dict[str, Any]] = {}
    for t in teams:
        for g in (team_games.get(t) or []):
            if int(g.get('gameType') or 0) != 2:
                continue
            gid = g.get('id')
            if gid is None or gid in by_id:
                continue
            by_id[gid] = g
    schedule = sorted(by_id.values(), key=lambda x: (str(x.get('date') or ''), str(x.get('id') or '')))
    if not schedule:
        raise RuntimeError('No regular-season schedule found.')
    print(f'  schedule: {len(schedule)} games, {len(teams)} teams ({time.time() - t0:.1f}s)', flush=True)
    return teams, team_proj_map, schedule, b2b_sets


def division_map(teams: List[str]) -> Dict[str, str]:
    """team -> division, from the NHL standings endpoint (falls back to the static map)."""
    try:
        import requests
        js = requests.get('https://api-web.nhle.com/v1/standings/now', timeout=30,
                          headers={'User-Agent': 'Mozilla/5.0'}).json()
        out: Dict[str, str] = {}
        for r in (js.get('standings') or []):
            ab = r.get('teamAbbrev')
            if isinstance(ab, dict):
                ab = ab.get('default')
            div = r.get('divisionName')
            if ab and div:
                out[str(ab).upper()] = str(div)
        if all(t in out for t in teams):
            print('Divisions loaded from NHL standings endpoint.', flush=True)
            return out
        print('  [warn] standings endpoint incomplete; using static division map.', flush=True)
    except Exception as e:
        print(f'  [warn] standings endpoint failed ({e}); using static division map.', flush=True)
    static: Dict[str, str] = {}
    for div, tlist in DIVISIONS_FALLBACK.items():
        for t in tlist:
            static[t] = div
    return static


# ── simulation (mirrors scripts/run_simulations.py, minus scorer draws) ──────

def simulate_regular(schedule, team_proj_map, b2b_sets, season: int, rng: random.Random):
    lg = R._V2_LG_AVG.get(str(season), 3.0)
    sqrt2 = math.sqrt(2.0)
    weight = R._V2_CONSERVATIVE_WEIGHT
    situations = R._V2_SITUATION
    poisson = R._poisson_draw

    results: List[Dict[str, Any]] = []
    append = results.append
    for g in schedule:
        home = g.get('home')
        away = g.get('away')
        if home not in team_proj_map or away not in team_proj_map:
            continue
        date_iso = g.get('date')
        sit = situations.get(
            (1 if date_iso in b2b_sets.get(home, ()) else 0,
             1 if date_iso in b2b_sets.get(away, ()) else 0), 0.0)
        mu = weight * (float(team_proj_map.get(home) or 0.0) - float(team_proj_map.get(away) or 0.0) + sit)
        gf_home = max(0.5, lg + mu / 2.0)
        gf_away = max(0.5, lg - mu / 2.0)
        sigma = math.sqrt(gf_home + gf_away)
        p_home = 0.5 * (1.0 + math.erf(mu / (sigma * sqrt2)))

        gh = poisson(gf_home, rng)
        ga = poisson(gf_away, rng)
        if gh > ga:
            winner, ot = home, False
        elif ga > gh:
            winner, ot = away, False
        else:
            ot = True
            winner = home if rng.random() < p_home else away
        # ~30% of 1-goal regulation wins are really OT wins (NHL OT rate ~25%).
        if not ot and abs(gh - ga) == 1 and rng.random() < 0.30:
            ot = True

        if winner == home:
            append({'home': home, 'away': away, 'winner': home, 'loser': away, 'ot': ot,
                    'homeGoals': int(gh), 'awayGoals': int(ga),
                    'homePoints': 2, 'awayPoints': (1 if ot else 0)})
        else:
            append({'home': home, 'away': away, 'winner': away, 'loser': home, 'ot': ot,
                    'homeGoals': int(gh), 'awayGoals': int(ga),
                    'homePoints': (1 if ot else 0), 'awayPoints': 2})
    return results


def simulate_one(seed: int) -> Tuple[Dict[str, int], Dict[str, int]]:
    """One full season. Returns (points_by_team, made_playoffs_by_team)."""
    rng = random.Random(seed)
    results = simulate_regular(_W['schedule'], _W['team_proj_map'], _W['b2b_sets'], _W['season'], rng)
    standings = R._standings_from_results(_W['teams'], results)
    seeds = R._seed_playoffs(standings)
    if len(seeds['East']) < 8 or len(seeds['West']) < 8:
        top16 = [r['team'] for r in standings[:16]]
        seeds = {'East': top16[:8], 'West': top16[8:16]}
    playoffs = R._simulate_playoffs(seeds, _W['team_proj_map'], _W['season'], rng)
    # A team made the playoffs iff it appears in a round-1 series (16 teams).
    po_teams = set()
    for conf in ('East', 'West'):
        for s in (playoffs.get('round1') or {}).get(conf) or []:
            po_teams.add(s.get('winner'))
            po_teams.add(s.get('loser'))
    made = {t: (1 if t in po_teams else 0) for t in _W['teams']}
    return {r['team']: int(r['points']) for r in standings}, made


def _worker_init(teams, team_proj_map, schedule, b2b_sets, season):
    _W['teams'] = teams
    _W['team_proj_map'] = team_proj_map
    _W['schedule'] = schedule
    _W['b2b_sets'] = b2b_sets
    _W['season'] = season


def run_sims(n_sims: int, base_seed: int, workers: int, payload) -> Dict[str, Dict[str, float]]:
    """Return per-team {sum_pts, sum_pts_sq, playoffs, n}."""
    acc: Dict[str, Dict[str, float]] = {
        t: {'sum_pts': 0.0, 'sum_pts_sq': 0.0, 'playoffs': 0.0, 'n': 0.0} for t in payload[0]
    }
    seeds = [base_seed + i for i in range(n_sims)]
    done = 0
    t0 = time.time()
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=payload) as ex:
            for pts, made in ex.map(simulate_one, seeds, chunksize=8):
                for t, p in pts.items():
                    a = acc[t]
                    a['sum_pts'] += p
                    a['sum_pts_sq'] += p * p
                    a['playoffs'] += made.get(t, 0)
                    a['n'] += 1
                done += 1
                if done % 500 == 0:
                    el = time.time() - t0
                    print(f'  {done}/{n_sims} sims ({el:.0f}s, {el / done * 1000:.0f} ms/sim)', flush=True)
    else:
        _worker_init(*payload)
        for s in seeds:
            pts, made = simulate_one(s)
            for t, p in pts.items():
                a = acc[t]
                a['sum_pts'] += p
                a['sum_pts_sq'] += p * p
                a['playoffs'] += made.get(t, 0)
                a['n'] += 1
            done += 1
            if done % 200 == 0:
                el = time.time() - t0
                print(f'  {done}/{n_sims} sims ({el:.0f}s, {el / done * 1000:.0f} ms/sim)', flush=True)
    print(f'  {done} sims in {time.time() - t0:.1f}s', flush=True)
    return acc


# ── stats / odds / rendering ─────────────────────────────────────────────────

def summarise(acc, teams, div_by_team, odds) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for t in teams:
        a = acc[t]
        n = int(a['n']) or 0
        if n == 0:
            continue
        mean = a['sum_pts'] / n
        var = max(0.0, a['sum_pts_sq'] / n - mean * mean)   # population variance
        sd = math.sqrt(var)
        se = sd / math.sqrt(n)
        ou = odds.get(t)
        rows.append({
            'team': t,
            'division': div_by_team.get(t, 'Unknown'),
            'projected_points': mean,
            'std_dev': sd,
            'std_error': se,
            'playoff_prob': a['playoffs'] / n,
            'pinnacle_ou': ou,
            'difference': (mean - ou) if ou is not None else None,
            'sims': n,
        })
    rows.sort(key=lambda r: (-r['projected_points'], r['team']))
    return rows


def read_report_csv(path: str) -> List[Dict[str, Any]]:
    """Load a previously written division_report_*.csv (for re-rendering only)."""
    rows: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            def num(key):
                v = (r.get(key) or '').strip()
                if v == '':
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append({
                'team': str(r.get('team') or '').strip().upper(),
                'division': str(r.get('division') or '').strip(),
                'projected_points': num('projected_points') or 0.0,
                'std_dev': num('std_dev'),
                'std_error': num('std_error') or 0.0,
                'playoff_prob': num('playoff_prob') or 0.0,
                'pinnacle_ou': num('pinnacle_ou'),
                'difference': num('difference'),
                'sims': int(num('sims') or 0),
            })
    return rows


def load_odds(path: Optional[str], teams: List[str]) -> Dict[str, float]:
    if not path or not os.path.exists(path):
        return {}
    out: Dict[str, float] = {}
    with open(path, 'r', encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            key = next((k for k in row if k and k.strip().lower() in ('team', 'team_abbrev', 'abbrev')), None)
            val = next((k for k in row
                        if k and ('pinnacle' in k.strip().lower()
                                  or 'bet365' in k.strip().lower()
                                  or 'market' in k.strip().lower()
                                  or k.strip().lower() in ('ou', 'over_under', 'line', 'total'))), None)
            if key is None or val is None:
                continue
            t = str(row[key]).strip().upper()
            raw = str(row[val]).strip()
            if not t or not raw:
                continue
            try:
                out[t] = float(raw)
            except ValueError:
                continue
    missing = [t for t in teams if t not in out]
    if missing:
        print(f'  [warn] no Pinnacle line for {len(missing)} teams: {", ".join(missing)}', flush=True)
    return out


def write_odds_template(path: str, teams: List[str], div_by_team: Dict[str, str]) -> None:
    with open(path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['team', 'division', 'market_ou'])
        for t in sorted(teams, key=lambda x: (div_by_team.get(x, ''), x)):
            w.writerow([t, div_by_team.get(t, ''), ''])
    print(f'  wrote season-total template: {path}', flush=True)


def _fmt(v: Optional[float], spec: str, dash: str = '-') -> str:
    return dash if v is None else format(v, spec)


# ── theme (mirrors app/templates/base.html + app/static/manager.css) ─────────
THEME = {
    'bg': '#0d0f16',          # --bg
    'panel': '#141826',       # --panel
    'panel_alt': '#1c2233',   # --panel-alt
    'text': '#e6e8ec',        # --text
    'text_dim': '#9aa4b1',    # --text-dim
    'accent': '#4da3ff',      # --accent
    'value': '#f1f5f9',       # --value-text
    'row_alt': '#1a2030',     # .tbl zebra
    'head_from': '#1a2337',   # .tbl th gradient top
    'head_to': '#121a2b',     # .tbl th gradient bottom
    'border': '#1a2230',      # .tbl td border-bottom
    'pos': '#9fe6c4',
    'neg': '#ff8a8a',
    'foot': '#6b7280',
}

_TEAM_COLOR: Dict[str, str] = {}
_LOGO_DIR: Optional[str] = None


def team_colors() -> Dict[str, str]:
    """team abbrev -> primary colour, from Teams.csv."""
    global _TEAM_COLOR
    if _TEAM_COLOR:
        return _TEAM_COLOR
    out: Dict[str, str] = {}
    try:
        with open(os.path.join(REPO_ROOT, 'Teams.csv'), 'r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                ab = (row.get('Team') or '').strip().upper()
                col = (row.get('Color') or '').strip()
                if ab and col.startswith('#') and len(col) in (4, 7):
                    out[ab] = col
    except Exception:
        pass
    _TEAM_COLOR = out
    return out


def team_logo_urls() -> Dict[str, str]:
    """team abbrev -> NHL API logo URL (the same assets the app serves)."""
    out: Dict[str, str] = {}
    try:
        with open(os.path.join(REPO_ROOT, 'Teams.csv'), 'r', encoding='utf-8-sig', newline='') as f:
            for row in csv.DictReader(f):
                ab = (row.get('Team') or '').strip().upper()
                logo = (row.get('Logo') or '').strip()
                if ab and logo and str(row.get('Active') or '').strip() == '1':
                    out[ab] = logo
    except Exception:
        pass
    return out


def _rasterise_svg(svg_bytes: bytes, px: int = 320):
    """SVG -> RGBA PIL image, using a magenta chroma key so logos sit on any bg."""
    import io as _io
    import numpy as np
    from PIL import Image
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM

    key = (255, 0, 255)
    drawing = svg2rlg(_io.BytesIO(svg_bytes))
    scale = float(px) / max(drawing.width, drawing.height)
    drawing.scale(scale, scale)
    drawing.width *= scale
    drawing.height *= scale
    buf = _io.BytesIO()
    renderPM.drawToFile(drawing, buf, fmt='PNG', bg=(key[0] << 16) | (key[1] << 8) | key[2])
    buf.seek(0)
    img = Image.open(buf).convert('RGB')
    arr = np.asarray(img).astype(np.float64)
    k = np.asarray(key, dtype=np.float64)
    alpha = np.clip(np.abs(arr - k).sum(axis=2) / 150.0, 0.0, 1.0)
    # Un-premultiply so anti-aliased edges don't keep a magenta fringe.
    fg = np.zeros_like(arr)
    mask = alpha > 0
    for ch in range(3):
        fg[..., ch][mask] = (arr[..., ch][mask] - k[ch] * (1 - alpha[mask])) / alpha[mask]
    rgba = Image.fromarray(np.dstack([np.clip(fg, 0, 255), alpha * 255.0]).astype(np.uint8), 'RGBA')
    return rgba.crop(rgba.getbbox())


def ensure_logos(teams: List[str], out_dir: str, refresh: bool = False) -> Dict[str, str]:
    """Download + rasterise each team's NHL logo once; returns team -> PNG path."""
    global _LOGO_DIR
    logo_dir = os.path.join(out_dir, 'logos')
    os.makedirs(logo_dir, exist_ok=True)
    _LOGO_DIR = logo_dir
    urls = team_logo_urls()
    paths: Dict[str, str] = {}
    for t in teams:
        path = os.path.join(logo_dir, f'{t}.png')
        if os.path.exists(path) and not refresh and os.path.getsize(path) > 0:
            paths[t] = path
            continue
        url = urls.get(t)
        if not url:
            continue
        try:
            import requests
            r = requests.get(url, timeout=25, headers={'User-Agent': 'Mozilla/5.0'})
            r.raise_for_status()
            _rasterise_svg(r.content).save(path)
            paths[t] = path
        except Exception as e:
            print(f'  [warn] logo failed for {t}: {e}', flush=True)
    print(f'  logos ready: {len(paths)}/{len(teams)}', flush=True)
    return paths


def _logo_image(path: Optional[str]):
    if not path or not os.path.exists(path):
        return None
    try:
        import matplotlib.image as mpimg
        return mpimg.imread(path)
    except Exception:
        return None


def render_division(ax, div: str, rows: List[Dict[str, Any]], n_sims: int,
                    market_label: str = 'Market O/U', market_note: str = '',
                    show_title: bool = True, logos: Optional[Dict[str, str]] = None,
                    season: int = 20262027, footer: bool = True) -> None:
    """Draw one division table in the app's dark theme, with team logos."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage
    from matplotlib.patches import FancyBboxPatch, Rectangle

    logos = logos or {}
    colors = team_colors()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    has_market = any(r['pinnacle_ou'] is not None for r in rows)
    n = len(rows)

    # ── card ──
    ax.add_patch(FancyBboxPatch((0.012, 0.012), 0.976, 0.976,
                                boxstyle='round,pad=0.004,rounding_size=0.02',
                                linewidth=1.0, edgecolor=THEME['border'],
                                facecolor=THEME['panel'], zorder=0))

    top = 0.955
    if show_title:
        ax.text(0.045, top, f'{div} Division', color=THEME['value'],
                fontsize=15.5, fontweight='bold', va='top', ha='left', zorder=3)
        ax.text(0.045, top - 0.055, f'NHL {str(season)[:4]}-{str(season)[4:]} season projections'
                                    f'  \u00b7  {n_sims:,} simulated seasons',
                color=THEME['text_dim'], fontsize=9.5, va='top', ha='left', zorder=3)
        ax.add_patch(Rectangle((0.045, top - 0.108), 0.052, 0.006,
                               facecolor=THEME['accent'], edgecolor='none', zorder=3))

    # ── column layout ──
    cols = [
        ('TEAM', 0.104, 'left'),
        ('PROJ PTS', 0.400, 'right'),
        ('STD DEV', 0.520, 'right'),
        ('PLAYOFF %', 0.665, 'right'),
    ]
    if has_market:
        cols += [(market_label.upper(), 0.840, 'right'), ('DIFF', 0.965, 'right')]

    head_top = top - 0.145
    head_h = 0.058
    body_top = head_top - head_h
    body_bottom = 0.118 if footer else 0.035
    row_h = (body_top - body_bottom) / max(n, 1)

    # ── header row (gradient like .tbl th) ──
    grad = np.linspace(0, 1, 64).reshape(-1, 1)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('hdr', [THEME['head_from'], THEME['head_to']])
    ax.imshow(grad, extent=(0.03, 0.97, head_top - head_h, head_top), aspect='auto',
              cmap=cmap, zorder=1, origin='upper')
    ax.plot([0.03, 0.97], [head_top, head_top], color=THEME['border'], lw=1.0, zorder=2)
    ax.plot([0.03, 0.97], [head_top - head_h, head_top - head_h], color=THEME['border'], lw=1.0, zorder=2)
    for label, x, align in cols:
        ax.text(x, head_top - head_h / 2, label, color=THEME['text_dim'], fontsize=8.0,
                fontweight='bold', va='center', ha=align, zorder=3)

    # OffsetImage zoom is in points, so derive it from the axes height in points.
    fig = ax.figure
    ax_h_pts = ax.get_position().height * fig.get_size_inches()[1] * 72.0

    # ── rows ──
    for i, r in enumerate(rows):
        y_top = body_top - i * row_h
        y_mid = y_top - row_h / 2
        y_bot = y_top - row_h

        if i % 2 == 1:
            ax.add_patch(Rectangle((0.03, y_bot), 0.94, row_h,
                                   facecolor=THEME['row_alt'], edgecolor='none', zorder=1))
        if i > 0:
            ax.plot([0.03, 0.97], [y_top, y_top], color=THEME['border'], lw=0.8, zorder=2)

        # team colour accent bar
        accent = colors.get(r['team'], THEME['accent'])
        ax.add_patch(Rectangle((0.03, y_bot + row_h * 0.16), 0.0045, row_h * 0.68,
                               facecolor=accent, edgecolor='none', zorder=3))

        # logo (kept clear of the team-colour accent bar)
        img = _logo_image(logos.get(r['team']))
        if img is not None:
            # logo height = 62% of the row height, converted to points
            zoom = max(0.02, (row_h * ax_h_pts * 0.62) / float(img.shape[0]))
            ab = AnnotationBbox(OffsetImage(img, zoom=zoom), (0.077, y_mid),
                                xycoords='axes fraction', frameon=False,
                                box_alignment=(0.5, 0.5), zorder=4, pad=0)
            ax.add_artist(ab)

        ax.text(0.104, y_mid, r['team'], color=THEME['value'], fontsize=11.5,
                fontweight='bold', va='center', ha='left', zorder=3)

        ax.text(0.400, y_mid, f"{r['projected_points']:.1f}", color=THEME['value'],
                fontsize=11.5, fontweight='bold', va='center', ha='right', zorder=3)
        ax.text(0.520, y_mid, f"{r['std_dev']:.1f}", color=THEME['text_dim'],
                fontsize=10.5, va='center', ha='right', zorder=3)

        # playoff probability: track + fill sit left of the number
        pct = max(0.0, min(1.0, r['playoff_prob']))
        bar_x, bar_w = 0.535, 0.066
        bar_y = y_mid - row_h * 0.13
        bar_h = row_h * 0.26
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, bar_h,
                               facecolor=THEME['border'], edgecolor='none', zorder=2))
        if pct > 0:
            ax.add_patch(Rectangle((bar_x, bar_y), bar_w * pct, bar_h,
                                   facecolor=THEME['accent'], edgecolor='none', zorder=3))
        ax.text(0.672, y_mid, f'{pct * 100:.1f}%', color=THEME['text'],
                fontsize=10.5, va='center', ha='right', zorder=3)

        if has_market:
            ax.text(0.840, y_mid, _fmt(r['pinnacle_ou'], '.1f'), color=THEME['text'],
                    fontsize=10.5, va='center', ha='right', zorder=3)
            d = r['difference']
            if d is None:
                ax.text(0.965, y_mid, '-', color=THEME['text_dim'], fontsize=10.5,
                        va='center', ha='right', zorder=3)
            else:
                ax.text(0.965, y_mid, f'{d:+.1f}',
                        color=(THEME['pos'] if d > 0 else THEME['neg']),
                        fontsize=11.5, fontweight='bold', va='center', ha='right', zorder=3)

    ax.plot([0.03, 0.97], [body_bottom, body_bottom], color=THEME['border'], lw=1.0, zorder=2)

    if footer:
        # Two short lines so the note always stays inside the card.
        line1 = 'Diff = projection \u2212 market   \u00b7   Std Dev = spread of simulated points'
        ax.text(0.045, 0.072, line1, color=THEME['foot'], fontsize=8.0,
                va='center', ha='left', zorder=3)
        if market_note:
            ax.text(0.045, 0.043, market_note, color=THEME['foot'], fontsize=8.0,
                    va='center', ha='left', zorder=3)


def _save_figure(fig, path: str, **kwargs) -> None:
    """savefig with retries: Windows viewers/indexers transiently lock the target."""
    import time as _time
    last: Optional[Exception] = None
    for attempt in range(5):
        tmp = f'{path}.{os.getpid()}.tmp.png'
        try:
            fig.savefig(tmp, **kwargs)
            os.replace(tmp, path)
            return
        except OSError as e:                      # locked / invalid handle
            last = e
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
            _time.sleep(0.6 * (attempt + 1))
    raise RuntimeError(f'could not write {path}: {last}')


def render_all(out_dir: str, rows_by_div, n_sims: int, season: int, odds_used: bool, stamp: str,
               market_label: str = 'Market O/U', market_note: str = '') -> List[str]:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    teams = sorted({r['team'] for rows in rows_by_div.values() for r in rows})
    logos = ensure_logos(teams, out_dir)

    written: List[str] = []
    for div in DIVISION_ORDER:
        rows = rows_by_div.get(div) or []
        if not rows:
            continue
        fig, ax = plt.subplots(figsize=(8.6, 5.4), dpi=200)
        fig.patch.set_facecolor(THEME['bg'])
        ax.set_facecolor(THEME['bg'])
        render_division(ax, div, rows, n_sims, market_label, market_note,
                        show_title=True, logos=logos, season=season)
        path = os.path.join(out_dir, f'division_{div}.png')
        _save_figure(fig, path, facecolor=THEME['bg'], bbox_inches='tight', pad_inches=0.12)
        plt.close(fig)
        written.append(path)
        print(f'  wrote {path}', flush=True)

    fig, axes = plt.subplots(2, 2, figsize=(17.6, 11.2), dpi=150)
    fig.patch.set_facecolor(THEME['bg'])
    for ax, div in zip(axes.ravel(), DIVISION_ORDER):
        ax.set_facecolor(THEME['bg'])
        render_division(ax, div, rows_by_div.get(div) or [], n_sims, market_label, market_note,
                        show_title=True, logos=logos, season=season, footer=False)
    note = market_note or ('market season point total' if odds_used else 'market over/under not supplied')
    fig.suptitle(f'NHL {str(season)[:4]}-{str(season)[4:]} Season Projections by Division   \u00b7   '
                 f'{n_sims:,} simulations   \u00b7   {note}',
                 fontsize=13.5, color=THEME['value'], y=0.975)
    fig.tight_layout(rect=(0, 0.01, 1, 0.95), h_pad=1.4)
    all_path = os.path.join(out_dir, 'division_report_all.png')
    _save_figure(fig, all_path, facecolor=THEME['bg'], bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    written.append(all_path)
    print(f'  wrote {all_path}', flush=True)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description='Simulate N seasons and graph results by division.')
    ap.add_argument('--sims', type=int, default=10000)
    ap.add_argument('--season', type=int, default=int(R.current_season_id()))
    ap.add_argument('--seed', type=int, default=20260928)
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--odds', default=None, help='CSV with team,pinnacle_ou columns')
    ap.add_argument('--out', default=os.path.join(REPO_ROOT, 'data', 'simulations'))
    ap.add_argument('--render-only', default=None, metavar='CSV',
                    help='Skip simulation; re-render graphics from a previous report CSV')
    ap.add_argument('--market-label', default='Market O/U',
                    help='Header for the betting-market column (default "Market O/U")')
    ap.add_argument('--market-note', default='',
                    help='Footnote naming the market source, e.g. "Bet365, 2026-09-28"')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.render_only:
        rows = read_report_csv(args.render_only)
        if not rows:
            print('No rows in report CSV.', file=sys.stderr)
            return 1
        if args.odds:
            odds = load_odds(args.odds, [r['team'] for r in rows])
            for r in rows:
                ou = odds.get(r['team'])
                r['pinnacle_ou'] = ou
                r['difference'] = (r['projected_points'] - ou) if ou is not None else None
            print(f'  applied market lines for {len(odds)}/{len(rows)} teams from {args.odds}', flush=True)
        n_sims = max(r['sims'] for r in rows)
        rows_by_div: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            rows_by_div.setdefault(r['division'], []).append(r)
        for div in rows_by_div:
            rows_by_div[div].sort(key=lambda x: (-x['projected_points'], x['team']))
        has_odds = any(r['pinnacle_ou'] is not None for r in rows)
        print(f'Re-rendering {len(rows)} teams from {args.render_only} '
              f'(market lines: {"yes" if has_odds else "no"})', flush=True)
        written = render_all(args.out, rows_by_div, n_sims, args.season, has_odds, 'rerender',
                             args.market_label, args.market_note)
        for p in written:
            print(f'  {p}')
        return 0

    print(f'=== Division simulation report ===')
    print(f'  sims={args.sims}  season={args.season}  seed={args.seed}  workers={args.workers}', flush=True)

    teams, team_proj_map, schedule, b2b_sets = load_base(args.season)
    div_by_team = division_map(teams)

    odds_path = args.odds
    if not odds_path:
        odds_path = os.path.join(args.out, 'season_points_ou.csv')
        if not os.path.exists(odds_path):
            write_odds_template(odds_path, teams, div_by_team)
    odds = load_odds(odds_path, teams)
    print(f'  market lines loaded: {len(odds)}/{len(teams)}', flush=True)

    payload = (teams, team_proj_map, schedule, b2b_sets, args.season)
    acc = run_sims(args.sims, args.seed, args.workers, payload)

    rows = summarise(acc, teams, div_by_team, odds)
    rows_by_div: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        rows_by_div.setdefault(r['division'], []).append(r)

    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(args.out, f'division_report_{stamp}.csv')
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        cols = ['team', 'division', 'projected_points', 'std_dev', 'std_error',
                'playoff_prob', 'pinnacle_ou', 'difference', 'sims']
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ('' if r[k] is None else (round(r[k], 6) if isinstance(r[k], float) else r[k]))
                        for k in cols})
    print(f'  wrote {csv_path}', flush=True)

    written = render_all(args.out, rows_by_div, args.sims, args.season, bool(odds), stamp,
                         args.market_label, args.market_note)

    print('\n=== Results ===')
    for div in DIVISION_ORDER:
        print(f'\n{div}')
        print(f'  {"Team":5} {"ProjPts":>8} {"SE":>6} {"Playoff%":>9} {"MktOU":>7} {"Diff":>7}')
        for r in rows_by_div.get(div) or []:
            print(f'  {r["team"]:5} {r["projected_points"]:8.1f} {r["std_error"]:6.2f} '
                  f'{r["playoff_prob"] * 100:8.1f}% {_fmt(r["pinnacle_ou"], "7.1f"):>7} '
                  f'{_fmt(r["difference"], "+7.1f"):>7}')
    print('\nArtifacts:')
    for p in written + [csv_path]:
        print(f'  {p}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
