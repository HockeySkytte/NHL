"""Tests for `_fetch_club_schedule_games` season handling.

The 20262027 schedule is published by the NHL API (84 games per team). The old
Python behaviour unconditionally downloaded 20252026 and shifted the dates
forward, which silently projected the 84-game season from the 82-game schedule
with the previous season's opponents and dates. The Rust port already tries the
real season first; these tests lock the same behaviour into Python.
"""
import os

import pytest

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

import app.routes as routes  # noqa: E402

SEASON = 20262027


def _payload(season: int, n_games: int = 84):
    return {
        'games': [
            {
                'id': season * 1000 + i,
                'gameType': 2,
                'gameDate': f'{str(season)[:4]}-10-{(i % 28) + 1:02d}',
                'awayTeam': {'abbrev': 'BOS'},
                'homeTeam': {'abbrev': 'TOR'},
            }
            for i in range(n_games)
        ]
    }


class _Resp:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        return self._payload


@pytest.fixture(autouse=True)
def _clear_schedule_cache(monkeypatch):
    monkeypatch.setattr(routes, '_CLUB_SCHEDULE_CACHE', {})


def _mock_requests(monkeypatch, by_season):
    """Record requested seasons and serve canned payloads."""
    asked = []

    def fake_get(url, timeout=None, **kwargs):
        season = int(url.rstrip('/').rsplit('/', 1)[-1])
        asked.append(season)
        payload = by_season.get(season)
        if payload is None:
            return _Resp(404, {})
        return _Resp(200, payload(season))

    monkeypatch.setattr(routes.requests, 'get', fake_get)
    return asked


def test_published_season_is_fetched_directly_without_shifting(monkeypatch):
    asked = _mock_requests(monkeypatch, {SEASON: _payload})

    games = routes._fetch_club_schedule_games('TOR', SEASON)

    assert asked == [SEASON]                      # never fell back to 20252026
    assert len(games) == 84
    assert all(str(g['date']).startswith('2026-') for g in games)
    assert all('_shift' not in str(g['id']) for g in games)


def test_falls_back_to_shifted_previous_season_when_unpublished(monkeypatch):
    asked = _mock_requests(monkeypatch, {20252026: _payload})

    games = routes._fetch_club_schedule_games('TOR', SEASON)

    assert asked == [SEASON, 20252026]            # real season tried first
    assert len(games) == 84
    # 2025-10-xx shifted forward one year -> 2026-10-xx (the real season window).
    assert all(str(g['date']).startswith('2026-') for g in games)
    assert all(str(g['id']).endswith('_shift1') for g in games)


def test_previous_season_is_used_directly_for_a_published_old_season(monkeypatch):
    asked = _mock_requests(monkeypatch, {20252026: _payload})

    games = routes._fetch_club_schedule_games('TOR', 20252026)

    assert asked == [20252026]
    assert all(str(g['date']).startswith('2025-') for g in games)


def test_returns_empty_when_both_seasons_fail(monkeypatch):
    _mock_requests(monkeypatch, {})

    assert routes._fetch_club_schedule_games('TOR', SEASON) == []
