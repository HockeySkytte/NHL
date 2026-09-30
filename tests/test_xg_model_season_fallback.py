"""Regression guard for the xG model a game report resolves.

The trained xG windows slide (s-2..s, s-1..s+1, s..s+2). When the 20262027
season opened, none of those files existed yet, so `api_game_pbp` loaded no
model at all and left every shot's xG at None. The game report's xG card then
rendered 0.00 / N/A even though Corsi, Fenwick, Shots and Goals were all fine.

A brand-new season must fall back to the newest trained window instead.
"""
import os

import pytest

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

import app.routes as routes  # noqa: E402

NEW_SEASON = 20262027
NEWEST_TRAINED = '20222023_20242025'


@pytest.fixture(autouse=True)
def _clear_latest_cache(monkeypatch):
    monkeypatch.setattr(routes, '_XG_LATEST_MODEL_CACHE', {})


def test_season_window_helpers_round_trip():
    assert routes.season_window_prev(20262027) == 20252026
    assert routes.season_window_next(20262027) == 20272028
    assert routes.season_window_next(routes.season_window_prev(20242025)) == 20242025


def test_latest_model_is_the_newest_end_year(monkeypatch, tmp_path):
    for name in ('xgb_20212022_20232024.pkl', 'xgb_20222023_20242025.pkl',
                 'xgb_20202021_20222023.pkl'):
        (tmp_path / name).write_text('', encoding='utf-8')
    monkeypatch.setattr(routes, '_model_dir', lambda: str(tmp_path))

    assert routes.latest_xg_model_file('xgb') == 'xgb_20222023_20242025.pkl'


def test_latest_model_does_not_cross_prefixes(monkeypatch, tmp_path):
    (tmp_path / 'xgb2_20222023_20242025.pkl').write_text('', encoding='utf-8')
    (tmp_path / 'xgb_20212022_20232024.pkl').write_text('', encoding='utf-8')
    monkeypatch.setattr(routes, '_model_dir', lambda: str(tmp_path))

    assert routes.latest_xg_model_file('xgb') == 'xgb_20212022_20232024.pkl'
    assert routes.latest_xg_model_file('xgb2') == 'xgb2_20222023_20242025.pkl'
    assert routes.latest_xg_model_file('xgbs') is None


def test_latest_model_is_cached(monkeypatch, tmp_path):
    calls = []

    def fake_model_dir():
        calls.append(1)
        return str(tmp_path)

    monkeypatch.setattr(routes, '_model_dir', fake_model_dir)
    routes.latest_xg_model_file('xgb')
    routes.latest_xg_model_file('xgb')
    assert len(calls) == 1


def test_new_season_appends_newest_trained_window(monkeypatch, tmp_path):
    (tmp_path / f'xgb_{NEWEST_TRAINED}.pkl').write_text('', encoding='utf-8')
    monkeypatch.setattr(routes, '_model_dir', lambda: str(tmp_path))

    names = routes.xg_window_filenames(NEW_SEASON, 'xgb')

    assert names[:3] == [
        'xgb_20252026_20272028.pkl',   # centred window first
        'xgb_20242025_20262027.pkl',
        'xgb_20262027_20282029.pkl',
    ]
    assert names[-1] == f'xgb_{NEWEST_TRAINED}.pkl'


def test_centred_window_still_preferred_when_trained(monkeypatch, tmp_path):
    (tmp_path / 'xgb_20232024_20252026.pkl').write_text('', encoding='utf-8')
    (tmp_path / 'xgb_20222023_20242025.pkl').write_text('', encoding='utf-8')
    monkeypatch.setattr(routes, '_model_dir', lambda: str(tmp_path))

    names = routes.xg_window_filenames(20242025, 'xgb')

    assert names[0] == 'xgb_20232024_20252026.pkl'
    # The newest trained window is already a real candidate, so the fallback is
    # not duplicated onto the end of the list.
    assert names[1] == 'xgb_20222023_20242025.pkl'
    assert names.count('xgb_20222023_20242025.pkl') == 1
    assert len(names) == 3


# ── the real Model/ directory must satisfy the current season ────────────────

def test_repo_models_resolve_for_the_open_season():
    """The shipped Model/ tree must give 20262027 a model to score with."""
    for prefix in ('xgb', 'xgb2', 'xgbs'):
        fallback = routes.latest_xg_model_file(prefix)
        assert fallback, f'no trained {prefix} model found in Model/'
        assert routes.load_model_file(fallback) is not None
        assert fallback in routes.xg_window_filenames(NEW_SEASON, prefix)
