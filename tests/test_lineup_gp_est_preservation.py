"""Estimated Games must survive the 30-minute lineup refresh.

`scripts/lineups.py` re-scrapes line combinations every 30 minutes and
`scripts/upload_lineups.py` rewrites the Supabase `lineups` rows from that
scrape. Only `scripts/estimate_gp.py` produces `gp_est`, and it runs far less
often, so both steps have to carry the stored estimate forward. Otherwise every
scrape reset `estimated_gp` to 0 and the season simulations silently lost their
games-played weighting.
"""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.lineups import carry_over_gp_est  # noqa: E402
from scripts.upload_lineups import (  # noqa: E402
    build_rows,
    is_implausible_shrink,
    prune_stale_rows,
)


def _snapshot(team='TOR', pid=8478402, gp_est=71, note='wtd-avg last 3'):
    player = {'name': 'Auston Matthews', 'playerId': pid, 'unit': 'C1', 'pos': 'F'}
    if gp_est is not None:
        player['gp_est'] = gp_est
        player['gp_est_note'] = note
    return {team: {'team': team, 'forwards': [player], 'defense': [], 'goalies': []}}


# ── scripts/lineups.py carry-over ────────────────────────────────────────────

def test_scrape_carries_gp_est_forward(tmp_path):
    path = tmp_path / 'lineups_all.json'
    path.write_text(json.dumps(_snapshot(gp_est=71)), encoding='utf-8')

    fresh = _snapshot(gp_est=None)          # a scrape never produces gp_est
    carried = carry_over_gp_est(fresh, str(path))

    assert carried == 1
    assert fresh['TOR']['forwards'][0]['gp_est'] == 71
    assert fresh['TOR']['forwards'][0]['gp_est_note'] == 'wtd-avg last 3'


def test_carry_over_does_not_leak_across_teams(tmp_path):
    path = tmp_path / 'lineups_all.json'
    path.write_text(json.dumps(_snapshot(team='TOR', pid=8478402, gp_est=71)), encoding='utf-8')

    fresh = _snapshot(team='BOS', pid=8478402, gp_est=None)
    assert carry_over_gp_est(fresh, str(path)) == 0
    assert 'gp_est' not in fresh['BOS']['forwards'][0]


def test_carry_over_ignores_missing_or_broken_snapshot(tmp_path):
    fresh = _snapshot(gp_est=None)
    assert carry_over_gp_est(fresh, str(tmp_path / 'absent.json')) == 0

    broken = tmp_path / 'broken.json'
    broken.write_text('not json', encoding='utf-8')
    assert carry_over_gp_est(fresh, str(broken)) == 0
    assert 'gp_est' not in fresh['TOR']['forwards'][0]


# ── scripts/upload_lineups.py row building ───────────────────────────────────

def test_missing_gp_est_falls_back_to_stored_value():
    existing = {('TOR', 8478402): {'estimated_gp': 71, 'gp_note': 'stored'}}

    rows = build_rows(_snapshot(gp_est=None), '20262027', existing)

    assert rows[0]['estimated_gp'] == 71
    assert rows[0]['gp_note'] == 'stored'


def test_fresh_gp_est_wins_over_stored_value():
    existing = {('TOR', 8478402): {'estimated_gp': 71, 'gp_note': 'stored'}}

    rows = build_rows(_snapshot(gp_est=64, note='recomputed'), '20262027', existing)

    assert rows[0]['estimated_gp'] == 64
    assert rows[0]['gp_note'] == 'recomputed'


def test_no_stored_row_and_no_gp_est_is_zero():
    rows = build_rows(_snapshot(gp_est=None), '20262027', {})
    assert rows[0]['estimated_gp'] == 0


def test_injury_bookkeeping_is_preserved():
    existing = {('TOR', 8478402): {
        'estimated_gp': 71, 'gp_note': 'stored',
        'is_injured': 1, 'injury_start': '2026-10-01', 'injury_end': '2026-11-01',
        'replacement_id': 8471234, 'replacement_name': 'Call Up',
    }}

    row = build_rows(_snapshot(gp_est=None), '20262027', existing)[0]

    assert row['is_injured'] == 1
    assert row['injury_start'] == '2026-10-01'
    assert row['injury_end'] == '2026-11-01'
    assert row['replacement_id'] == 8471234
    assert row['replacement_name'] == 'Call Up'


def test_rows_without_a_stored_row_keep_defaults():
    row = build_rows(_snapshot(gp_est=None), '20262027', {})[0]

    assert row['is_injured'] == 0
    assert row['injury_start'] is None
    assert row['replacement_id'] is None
    assert row['season'] == '20262027'
    assert row['starter'] == 1


# ── stale-row pruning (must never empty the season) ──────────────────────────

class _DeleteBuilder:
    """Records the delete chain so we can assert the filters used."""

    def __init__(self, log):
        self.log = log

    def table(self, name):
        self.log.append(('table', name))
        return self

    def delete(self):
        self.log.append(('delete',))
        return self

    def eq(self, col, val):
        self.log.append(('eq', col, val))
        return self

    def in_(self, col, vals):
        self.log.append(('in', col, tuple(vals)))
        return self

    def execute(self):
        self.log.append(('execute',))
        return type('R', (), {'data': []})()


def _fake_client(monkeypatch, log):
    import app.supabase_client as sb

    monkeypatch.setattr(sb, 'get_client', lambda: _DeleteBuilder(log))


def test_prune_removes_only_players_missing_from_the_scrape(monkeypatch):
    log = []
    _fake_client(monkeypatch, log)

    removed = prune_stale_rows(
        '20262027',
        new_keys={('TOR', 1), ('TOR', 2)},
        existing_keys={('TOR', 1), ('TOR', 2), ('TOR', 99)},
    )

    assert removed == 1
    assert ('in', 'player_id', (99,)) in log
    assert ('eq', 'team', 'TOR') in log
    assert ('eq', 'season', '20262027') in log


def test_prune_is_a_noop_when_nothing_is_stale(monkeypatch):
    log = []
    _fake_client(monkeypatch, log)

    assert prune_stale_rows('20262027', {('TOR', 1)}, {('TOR', 1)}) == 0
    assert log == []


def test_prune_groups_by_team(monkeypatch):
    log = []
    _fake_client(monkeypatch, log)

    removed = prune_stale_rows(
        '20262027',
        new_keys={('TOR', 1)},
        existing_keys={('TOR', 1), ('TOR', 9), ('BOS', 9)},
    )

    assert removed == 2
    teams = sorted(entry[2] for entry in log if entry[0] == 'eq' and entry[1] == 'team')
    assert teams == ['BOS', 'TOR']


def test_prune_batches_large_deletes(monkeypatch):
    log = []
    _fake_client(monkeypatch, log)

    existing = {('TOR', pid) for pid in range(500)}
    removed = prune_stale_rows('20262027', new_keys=set(), existing_keys=existing)

    assert removed == 500
    batches = [entry for entry in log if entry[0] == 'in']
    assert [len(b[2]) for b in batches] == [200, 200, 100]


# ── gutted-scrape guard ──────────────────────────────────────────────────────

def test_a_gutted_scrape_is_rejected():
    # A DailyFaceoff outage that returns one team must not prune 858 rosters.
    assert is_implausible_shrink(1, 859) is True
    assert is_implausible_shrink(429, 859) is True


def test_a_normal_scrape_is_accepted():
    assert is_implausible_shrink(859, 859) is False
    assert is_implausible_shrink(800, 859) is False
    assert is_implausible_shrink(430, 859) is False


def test_first_upload_has_nothing_to_shrink_from():
    assert is_implausible_shrink(0, 0) is False
    assert is_implausible_shrink(1, 0) is False
