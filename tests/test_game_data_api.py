"""Tests for the /api/game-data endpoint (game_data table access)."""

import os

import pytest

from app import create_app
import app.routes as routes


@pytest.fixture(scope='module')
def app_instance():
    os.environ['XG_PRELOAD'] = '0'
    os.environ['PRESTART_LOGGER'] = '0'
    os.environ['PRELOAD_GM_CACHES'] = '0'
    app = create_app()
    app.config.update(TESTING=True)
    return app


@pytest.fixture()
def client(app_instance):
    return app_instance.test_client()


def test_game_data_requires_season(client):
    response = client.get('/api/game-data?player_id=8478402')
    assert response.status_code == 400
    assert 'season' in (response.get_json() or {}).get('error', '')


def test_game_data_requires_player_id(client):
    response = client.get('/api/game-data?season=20242025')
    assert response.status_code == 400
    assert 'player_id' in (response.get_json() or {}).get('error', '')


def test_game_data_invalid_season(client):
    response = client.get('/api/game-data?season=abc&player_id=8478402')
    assert response.status_code == 400
    assert 'season' in (response.get_json() or {}).get('error', '')


def test_game_data_invalid_player_id(client):
    response = client.get('/api/game-data?season=20242025&player_id=-1')
    assert response.status_code == 400
    assert 'player_id' in (response.get_json() or {}).get('error', '')


def test_game_data_invalid_game_id(client):
    response = client.get('/api/game-data?season=20242025&player_id=8478402&game_id=-5')
    assert response.status_code == 400
    assert 'game_id' in (response.get_json() or {}).get('error', '')


def test_game_data_source_unavailable(monkeypatch, client):
    monkeypatch.setattr(routes, '_sb_read', lambda *a, **k: None)
    response = client.get('/api/game-data?season=20242025&player_id=8478402')
    assert response.status_code == 503


def test_game_data_happy_path(monkeypatch, client):
    captured = {}

    def fake_read(table, *, columns='*', filters=None, col_map=None, order=None, limit=0):
        captured['table'] = table
        captured['columns'] = columns
        captured['filters'] = filters
        captured['order'] = order
        return [
            {
                'game_id': 2024020001,
                'season': 20242025,
                'player_id': 8478402,
                'toi_all': 18.5,
                'cf_ev': 12,
            },
            {
                'game_id': 2024020002,
                'season': 20242025,
                'player_id': 8478402,
                'toi_all': 15.0,
                'cf_ev': 8,
            },
        ]

    monkeypatch.setattr(routes, '_sb_read', fake_read)

    response = client.get('/api/game-data?season=20242025&player_id=8478402')
    assert response.status_code == 200
    assert response.headers.get('Cache-Control') == 'no-store'
    data = response.get_json() or {}
    assert data['season'] == 20242025
    assert data['player_id'] == 8478402
    assert data['game_id'] is None
    assert data['count'] == 2
    assert len(data['games']) == 2
    assert data['games'][0]['game_id'] == 2024020001
    assert captured['table'] == 'game_data'
    assert captured['columns'] == '*'
    assert captured['filters'] == {'season': 'eq.20242025', 'player_id': 'eq.8478402'}
    assert captured['order'] == 'game_id'


def test_game_data_with_game_id(monkeypatch, client):
    captured = {}

    def fake_read(table, *, columns='*', filters=None, col_map=None, order=None, limit=0):
        captured['filters'] = filters
        return []

    monkeypatch.setattr(routes, '_sb_read', fake_read)

    response = client.get('/api/game-data?season=20242025&player_id=8478402&game_id=2024020001')
    assert response.status_code == 200
    data = response.get_json() or {}
    assert data['count'] == 0
    assert data['games'] == []
    assert data['game_id'] == 2024020001
    assert captured['filters'] == {
        'season': 'eq.20242025',
        'player_id': 'eq.8478402',
        'game_id': 'eq.2024020001',
    }


def test_game_data_empty_result_is_not_fallback(monkeypatch, client):
    calls = []

    def fake_read(table, *, columns='*', filters=None, col_map=None, order=None, limit=0):
        calls.append(dict(filters or {}))
        return []

    monkeypatch.setattr(routes, '_sb_read', fake_read)

    # No data for the filter combination must return an honest empty result,
    # never a re-query with relaxed filters.
    response = client.get('/api/game-data?season=20242025&player_id=9999999')
    assert response.status_code == 200
    assert (response.get_json() or {}).get('count') == 0
    assert len(calls) == 1
