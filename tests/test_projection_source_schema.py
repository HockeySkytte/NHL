"""Regression guards for the Moncton source schema the projection model reads.

Two upstream changes broke the daily player-projections job:

1. `skaters_master` / `goalies_master` renamed `manpower` (EV/PP/SH) to
   `strengthstate` with explicit skater counts (5V5, 5V4, 4V5, ENA, ...), so the
   loaders raised `UndefinedColumn: column "manpower" does not exist`.
2. `games.away_team_id` / `home_team_id` became `text` while `teams.teamid` is
   still `bigint`. The int-keyed id map therefore matched nothing and
   `load_games` silently dropped all 8510 games, leaving an empty games frame
   that only failed much later.

The EV bucket includes ENA; that choice was measured against the legacy cached
goalies.csv / skater_ev_xg.csv rather than assumed (see the docstring on
EV_STRENGTH_STATES).
"""
import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')

import scripts.Game_Projection_Model as G  # noqa: E402


# ── strength-state bucketing ─────────────────────────────────────────────────

@pytest.mark.parametrize('state,expected', [
    ('5V5', 'EV'), ('4V4', 'EV'), ('3V3', 'EV'), ('ENA', 'EV'),
    ('5V4', 'PP'), ('5V3', 'PP'), ('4V3', 'PP'),
    ('4V5', 'SH'), ('3V5', 'SH'), ('3V4', 'SH'),
])
def test_strength_bucket_mapping(state, expected):
    assert G.strength_bucket(state) == expected


@pytest.mark.parametrize('state', ['5v5', ' 5v4 ', 'ena', '4v5'])
def test_strength_bucket_is_case_and_space_insensitive(state):
    assert G.strength_bucket(state) is not None


@pytest.mark.parametrize('state', ['ENF', '6V5', 'PP', 'EV', '', None, 'junk'])
def test_unknown_states_return_none(state):
    assert G.strength_bucket(state) is None


def test_every_bucket_set_is_disjoint():
    assert not (G.EV_STRENGTH_STATES & G.PP_STRENGTH_STATES)
    assert not (G.EV_STRENGTH_STATES & G.SH_STRENGTH_STATES)
    assert not (G.PP_STRENGTH_STATES & G.SH_STRENGTH_STATES)


def test_ena_belongs_to_even_strength():
    """Measured choice: ENA in EV matched the legacy cache best by a wide margin."""
    assert 'ENA' in G.EV_STRENGTH_STATES


# ── loaders against the current source shape ─────────────────────────────────

class _FakeReadSql:
    """Serves canned frames in call order and records the SQL/params it was given."""

    def __init__(self, *frames):
        self.frames = list(frames)
        self.queries = []
        self.params = []

    def __call__(self, q, conn, **kwargs):
        self.queries.append(str(q))
        self.params.append(kwargs.get('params') or {})
        return self.frames.pop(0)


def _teams_frame():
    return pd.DataFrame({
        'teamid': [26, 28, 21],
        'team': ['CBJ', 'NYR', 'BOS'],
        'teamname': ['Columbus Blue Jackets', 'New York Rangers', 'Boston Bruins'],
    })


def _games_frame():
    return pd.DataFrame({
        'game_id': [1, 2],
        'season': ['20252026', '20252026'],
        'date': ['2025-10-08', '2025-10-09'],
        'away_team_id': ['26', '21'],
        'home_team_id': ['28', '26'],
        'home_score': [3, 1],
        'away_score': [2, 4],
    })


def test_load_games_maps_text_team_ids(monkeypatch):
    fake = _FakeReadSql(_teams_frame(), _games_frame())
    monkeypatch.setattr(G.pd, 'read_sql', fake)

    id_to_abbr, _ = G.load_team_map(None)
    out = G.load_games(None, id_to_abbr)

    assert len(out) == 2                      # nothing silently dropped
    assert out['awayteam'].tolist() == ['CBJ', 'BOS']
    assert out['hometeam'].tolist() == ['NYR', 'CBJ']
    assert out['home_win'].tolist() == [1, 0]


def test_team_map_is_keyed_by_int():
    """`games.*_team_id` is text and `teams.teamid` is bigint; normalizing both
    to int is what keeps the join working."""
    fake = _FakeReadSql(_teams_frame())
    original = G.pd.read_sql
    G.pd.read_sql = fake
    try:
        id_to_abbr, _ = G.load_team_map(None)
    finally:
        G.pd.read_sql = original

    assert set(id_to_abbr) == {26, 28, 21}
    assert all(isinstance(k, int) for k in id_to_abbr)


def test_team_map_skips_null_team_ids():
    teams = pd.DataFrame({
        'teamid': [26, None],
        'team': ['CBJ', 'ZZZ'],
        'teamname': ['Columbus Blue Jackets', 'Ghost'],
    })
    fake = _FakeReadSql(teams)
    original = G.pd.read_sql
    G.pd.read_sql = fake
    try:
        id_to_abbr, name_to_abbr = G.load_team_map(None)
    finally:
        G.pd.read_sql = original

    assert set(id_to_abbr) == {26}
    assert name_to_abbr['CBJ'] == 'CBJ'      # abbrev aliases itself


# ── id column normalization ──────────────────────────────────────────────────

def test_normalize_coerces_text_ids_to_int():
    df = pd.DataFrame({'playerid': ['8479318', '8478483'], 'gameid': ['2026020001', '2026020002']})
    out = G.normalize_id_columns(df)

    assert out['playerid'].dtype == 'int64'
    assert out['gameid'].dtype == 'int64'
    assert out['playerid'].tolist() == [8479318, 8478483]


def test_normalize_coerces_numeric_ids_to_int():
    df = pd.DataFrame({'playerid': [615.0, 39.0], 'gameid': [91113, 91114]})
    out = G.normalize_id_columns(df)

    assert out['playerid'].dtype == 'int64'
    assert out['playerid'].tolist() == [615, 39]


def test_normalize_leaves_absent_columns_alone():
    df = pd.DataFrame({'season': ['20252026'], 'xgf': [1.0]})
    out = G.normalize_id_columns(df)

    assert list(out.columns) == ['season', 'xgf']
    assert out['xgf'].iloc[0] == 1.0


def test_normalize_fails_loudly_on_non_numeric_ids():
    df = pd.DataFrame({'playerid': ['8479318', 'not-an-id']})

    with pytest.raises(ValueError, match='not-an-id'):
        G.normalize_id_columns(df)


def test_load_goalies_buckets_strength_state(monkeypatch):
    frame = pd.DataFrame({
        'season': ['20252026'] * 4,
        'playerid': ['1'] * 4,
        'gameid': ['9'] * 4,
        'strengthstate': ['5V5', 'ENA', '5V4', '4V5'],
        'xg_on_a': [1.0, 0.1, 0.2, 0.3],
        'xga': [1.1, 0.1, 0.2, 0.3],
        'ga': [1, 0, 0, 1],
        'sa': [10, 2, 3, 5],
        'toi': [30.0, 1.0, 2.0, 3.0],
    })
    monkeypatch.setattr(G.pd, 'read_sql', _FakeReadSql(frame))

    out = G.load_goalies(None)

    assert 'manpower' in out.columns
    assert 'strengthstate' not in out.columns
    assert out['manpower'].tolist() == ['EV', 'EV', 'PP', 'SH']
    assert out['playerid'].dtype == 'int64'


def test_load_goalies_fails_loudly_on_a_new_state(monkeypatch):
    """A future upstream vocabulary change must not silently drop rows."""
    frame = pd.DataFrame({
        'season': ['20252026'], 'playerid': ['1'], 'gameid': ['9'],
        'strengthstate': ['6V5'], 'xg_on_a': [0.1], 'xga': [0.1],
        'ga': [0], 'sa': [1], 'toi': [1.0],
    })
    monkeypatch.setattr(G.pd, 'read_sql', _FakeReadSql(frame))

    with pytest.raises(RuntimeError, match='6V5'):
        G.load_goalies(None)


def test_load_skater_ev_xg_filters_on_strength_state(monkeypatch):
    fake = _FakeReadSql(pd.DataFrame({
        'season': [], 'playerid': [], 'gameid': [], 'xgf': [], 'xga': [],
    }))
    monkeypatch.setattr(G.pd, 'read_sql', fake)

    G.load_skater_ev_xg(None)

    sql = fake.queries[0]
    assert 'manpower' not in sql
    assert 'strengthstate' in sql
    assert 'UPPER(TRIM(strengthstate))' in sql      # case variants exist upstream
    assert set(fake.params[0].values()) == set(G.EV_STRENGTH_STATES)


# ── season selection for the export ──────────────────────────────────────────

import scripts.export_preseason_updating_player_projections as E  # noqa: E402


def _args(season=None, latest=False):
    return type('A', (), {'season': season, 'latest_season': latest})()


def _games(*seasons):
    return pd.DataFrame({'season': list(seasons)})


def test_latest_season_picks_the_newest_with_games():
    """The calendar season may have no games ingested yet."""
    games = _games('20192020', '20242025', '20252026')

    assert E.get_requested_seasons(_args(latest=True), games) == {'20252026'}


def test_latest_season_picks_up_a_new_season_automatically():
    games = _games('20242025', '20252026', '20262027')

    assert E.get_requested_seasons(_args(latest=True), games) == {'20262027'}


def test_latest_season_returns_empty_when_there_are_no_games():
    assert E.get_requested_seasons(_args(latest=True), _games()) == set()


def test_explicit_season_still_wins():
    games = _games('20252026', '20262027')

    assert E.get_requested_seasons(_args(season=['20252026']), games) == {'20252026'}


def test_excluded_seasons_are_removed():
    games = _games('20222023')

    assert E.get_requested_seasons(_args(latest=True), games) == set()


# ── explicit postgres driver ─────────────────────────────────────────────────

@pytest.mark.parametrize('url,expected', [
    ('postgresql://u:p@h:5432/db', 'postgresql+psycopg2://u:p@h:5432/db'),
    ('postgres://u:p@h:5432/db', 'postgresql+psycopg2://u:p@h:5432/db'),
    ('postgresql+psycopg2://u:p@h:5432/db', 'postgresql+psycopg2://u:p@h:5432/db'),
    ('mysql+mysqlconnector://root@localhost:3306/moncton',
     'mysql+mysqlconnector://root@localhost:3306/moncton'),
    ('', ''),
    (None, None),
])
def test_postgres_url_gets_an_explicit_driver(url, expected):
    assert G.postgres_url_with_psycopg2(url) == expected


def test_bare_postgres_url_never_relies_on_the_sqlalchemy_default():
    """SQLAlchemy 2.1 defaults `postgresql://` to psycopg v3, which is not
    installed - that broke every scheduled run the day 2.1 shipped."""
    out = G.postgres_url_with_psycopg2('postgresql://u:p@h:5432/db')

    assert '+psycopg2' in out
    assert out.startswith('postgresql+psycopg2://')


def test_get_engine_uses_the_psycopg2_driver(monkeypatch):
    """create_engine is lazy, so no connection is attempted here."""
    monkeypatch.setattr(G, '_engine', None)
    monkeypatch.setenv('DATABASE_MONCTON_URL', 'postgresql://u:p@h:6543/db')

    engine = G._get_engine()

    assert engine.url.drivername == 'postgresql+psycopg2'
    # The pooler port is still rewritten to the session port.
    assert engine.url.port == 5432
    assert engine.url.host == 'h'


def test_get_engine_requires_the_moncton_url(monkeypatch):
    monkeypatch.setattr(G, '_engine', None)
    monkeypatch.delenv('DATABASE_MONCTON_URL', raising=False)
    monkeypatch.setattr(G, 'load_dotenv', lambda *a, **k: None)

    with pytest.raises(RuntimeError, match='DATABASE_MONCTON_URL'):
        G._get_engine()
