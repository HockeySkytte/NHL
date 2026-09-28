"""Tests for the season-aware V2 player-projection map.

Regression guard: `nhl_current_playerprojections` only has rows once a season's
possession values have been exported, so the season `current_season_id()` names
is empty from September until that data lands. Loading the map for that empty
season put every lineup player on `_ROOKIE_FALLBACK`, which gave all 32 teams an
identical team projection and made every team project ~84 points. The loader must
fall back to the newest season that actually has rows.
"""
import os

import pytest

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

import app.routes as routes  # noqa: E402

CURRENT_SEASON = 20262027
NEWEST_AVAILABLE = 20252026


@pytest.fixture(autouse=True)
def _clear_latest_season_cache(monkeypatch):
    monkeypatch.setattr(routes, '_V2_FALLBACK_SEASON_CACHE', None)


# ── season resolution ────────────────────────────────────────────────────────

def test_falls_back_to_newest_available_season(monkeypatch):
    built = []

    def fake_build(season=None):
        built.append(season)
        return [{'player_id': 1}] if season == NEWEST_AVAILABLE else []

    monkeypatch.setattr(routes, '_build_v2_player_projections', fake_build)
    monkeypatch.setattr(routes, '_latest_v2_projection_season', lambda: NEWEST_AVAILABLE)

    out = routes._load_v2_player_projections_cached(CURRENT_SEASON)

    assert built == [CURRENT_SEASON, NEWEST_AVAILABLE]
    assert set(out) == {1}


def test_requested_season_is_used_when_it_has_rows(monkeypatch):
    built = []
    probed = []

    def fake_build(season=None):
        built.append(season)
        return [{'player_id': 7}]

    monkeypatch.setattr(routes, '_build_v2_player_projections', fake_build)
    monkeypatch.setattr(routes, '_latest_v2_projection_season',
                        lambda: probed.append(True) or NEWEST_AVAILABLE)

    out = routes._load_v2_player_projections_cached(NEWEST_AVAILABLE)

    assert built == [NEWEST_AVAILABLE]
    assert probed == []          # no fallback lookup when the season has rows
    assert set(out) == {7}


def test_defaults_to_current_season(monkeypatch):
    built = []
    monkeypatch.setattr(routes, 'current_season_id', lambda *a, **k: CURRENT_SEASON)
    monkeypatch.setattr(routes, '_build_v2_player_projections',
                        lambda season=None: built.append(season) or [{'player_id': 3}])

    routes._load_v2_player_projections_cached()

    assert built == [CURRENT_SEASON]


def test_empty_when_no_season_has_rows(monkeypatch):
    monkeypatch.setattr(routes, '_build_v2_player_projections', lambda season=None: [])
    monkeypatch.setattr(routes, '_latest_v2_projection_season', lambda: None)

    assert routes._load_v2_player_projections_cached(CURRENT_SEASON) == {}


def test_no_fallback_rebuild_when_latest_equals_requested(monkeypatch):
    built = []
    monkeypatch.setattr(routes, '_build_v2_player_projections',
                        lambda season=None: built.append(season) or [])
    monkeypatch.setattr(routes, '_latest_v2_projection_season', lambda: CURRENT_SEASON)

    assert routes._load_v2_player_projections_cached(CURRENT_SEASON) == {}
    assert built == [CURRENT_SEASON]     # not rebuilt for the same season


def test_projection_values_are_preserved(monkeypatch):
    monkeypatch.setattr(routes, '_build_v2_player_projections', lambda season=None: [{
        'player_id': 5, 'name': 'Test Player', 'position': 'F', 'team': 'TOR', 'gp': 82,
        'evo': 1.0, 'evd': 2.0, 'pp_raw': 3.0, 'sh_raw': 4.0,
        'gax': 5.0, 'gsax': 6.0, 'rookie': 7.0,
    }])
    monkeypatch.setattr(routes, '_latest_v2_projection_season', lambda: None)

    out = routes._load_v2_player_projections_cached(CURRENT_SEASON)

    assert out[5]['raw_projected_value'] == 28.0      # 1+2+3+4+5+6+7
    assert out[5]['projected_value'] == 28.0          # gp >= 41 so no weighting


# ── newest-season lookup ─────────────────────────────────────────────────────

def test_latest_season_takes_max_and_ignores_junk(monkeypatch):
    captured = {}

    def fake_sb_read(table, **kwargs):
        captured['table'] = table
        captured.update(kwargs)
        return [
            {'season': '20252026'},
            {'season': None},
            {'season': 'junk'},
            {'season': '20242025'},
        ]

    monkeypatch.setattr(routes, '_sb_read', fake_sb_read)

    assert routes._latest_v2_projection_season() == NEWEST_AVAILABLE
    assert captured['table'] == 'nhl_current_playerprojections'
    assert captured['order'] == '-season'
    assert captured['limit'] == 50


def test_latest_season_is_none_when_read_fails(monkeypatch):
    monkeypatch.setattr(routes, '_sb_read', lambda *a, **k: None)
    assert routes._latest_v2_projection_season() is None


def test_latest_season_is_cached(monkeypatch):
    calls = []
    monkeypatch.setattr(routes, '_sb_read',
                        lambda *a, **k: calls.append(1) or [{'season': '20252026'}])

    routes._latest_v2_projection_season()
    routes._latest_v2_projection_season()

    assert len(calls) == 1


# ── _sb_read descending-order convention ─────────────────────────────────────

class _Recorder:
    def __init__(self):
        self.calls = []

    def table(self, name):
        self.calls.append(('table', name))
        return self

    def select(self, cols):
        return self

    def range(self, a, b):
        return self

    def order(self, col, desc=False):
        self.calls.append(('order', col, desc))
        return self

    def eq(self, col, val):
        return self

    def in_(self, col, vals):
        return self

    def execute(self):
        return type('R', (), {'data': []})()


@pytest.mark.parametrize('order,expected', [
    ('-season', ('order', 'season', True)),
    ('playerid,strengthstate', ('order', 'playerid,strengthstate', False)),
])
def test_sb_read_order_convention(monkeypatch, order, expected):
    rec = _Recorder()
    monkeypatch.setattr(routes, '_SUPABASE_OK', True)
    monkeypatch.setattr(routes, '_sb_client', lambda: rec)

    routes._sb_read('nhl_current_playerprojections', columns='season', order=order)

    assert expected in rec.calls
