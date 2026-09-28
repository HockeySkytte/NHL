"""Tests for the `lineups_all.json` -> Supabase lineup merge (`app.routes`).

Regression guard: the static snapshot shipped in the deploy image is often older
than the Supabase rows. It may still fill in a missing ``gp_est``, but it must
never *add* players who are no longer on the team - that bug showed up in GM Mode
as duplicate starter slots (e.g. a traded player listed next to the current LW1).
"""
import json
import os

import pytest

os.environ.setdefault('XG_PRELOAD', '0')
os.environ.setdefault('PRESTART_LOGGER', '0')
os.environ.setdefault('PRELOAD_GM_CACHES', '0')

import app.routes as routes  # noqa: E402

# The bundled snapshot from the 2026-08-14 deploy, vs the Supabase rows written
# on 2026-09-28 ~2 minutes after the snapshot was regenerated.
STALE_JSON_GENERATED_AT = '2026-08-14T01:02:26.942467+00:00'
SUPABASE_GENERATED_AT = '2026-09-28T19:04:40.711208+00:00'
# Same pipeline run: the snapshot is generated first, then synced to Supabase.
PIPELINE_JSON_GENERATED_AT = '2026-09-28T19:02:26.000000+00:00'

CURRENT_LW_PID = 111
TRADED_AWAY_PID = 999
EXTRA_D_PID = 222


def _snapshot(generated_at, tmp_path):
    snapshot = {
        'ANA': {
            'team': 'ANA',
            'generated_at': generated_at,
            'forwards': [
                # Still on the team and in Supabase.
                {'name': 'Current LW', 'playerId': CURRENT_LW_PID, 'unit': 'LW1', 'pos': 'F',
                 'gp_est': 75, 'gp_est_note': 'wtd-avg last 3'},
                # Left the team after the snapshot was committed.
                {'name': 'Traded Away', 'playerId': TRADED_AWAY_PID, 'unit': 'LW1', 'pos': 'F',
                 'gp_est': 70, 'gp_est_note': 'stale-note'},
            ],
            'defense': [
                {'name': 'Extra D', 'playerId': EXTRA_D_PID, 'unit': 'EXT', 'pos': 'D'},
            ],
            'goalies': [],
        }
    }
    path = tmp_path / 'lineups_all.json'
    path.write_text(json.dumps(snapshot), encoding='utf-8')
    return str(path)


def _supabase_side(generated_at=SUPABASE_GENERATED_AT):
    return {
        'ANA': {
            'team': 'ANA',
            'generated_at': generated_at,
            'forwards': [{'name': 'Current LW', 'playerId': CURRENT_LW_PID, 'unit': 'LW1', 'pos': 'F'}],
            'defense': [],
            'goalies': [],
        }
    }


def _pids(node, group):
    return [p['playerId'] for p in node[group]]


# ── the regression ───────────────────────────────────────────────────────────

def test_stale_snapshot_does_not_add_players(tmp_path):
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot(STALE_JSON_GENERATED_AT, tmp_path))

    assert _pids(out['ANA'], 'forwards') == [CURRENT_LW_PID]
    assert out['ANA']['defense'] == []


def test_stale_snapshot_does_not_duplicate_a_starter_slot(tmp_path):
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot(STALE_JSON_GENERATED_AT, tmp_path))

    lw1 = [p for p in out['ANA']['forwards'] if p['unit'] == 'LW1']
    assert len(lw1) == 1
    assert lw1[0]['playerId'] == CURRENT_LW_PID


def test_stale_snapshot_still_fills_missing_gp_est(tmp_path):
    """A stale snapshot is still the offline source of GP estimates."""
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot(STALE_JSON_GENERATED_AT, tmp_path))

    rec = out['ANA']['forwards'][0]
    assert rec['gp_est'] == 75
    assert rec['gp_est_note'] == 'wtd-avg last 3'


# ── the behaviour that must keep working ─────────────────────────────────────

def test_fresh_snapshot_completes_the_pool(tmp_path):
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot('2026-10-01T12:00:00+00:00', tmp_path))

    assert _pids(out['ANA'], 'forwards') == [CURRENT_LW_PID, TRADED_AWAY_PID]
    assert _pids(out['ANA'], 'defense') == [EXTRA_D_PID]


def test_snapshot_from_the_current_pipeline_run_still_completes_the_pool(tmp_path):
    """scrape -> estimate -> sync: the sync stamps updated_at *after* the snapshot."""
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot(PIPELINE_JSON_GENERATED_AT, tmp_path))

    assert TRADED_AWAY_PID in _pids(out['ANA'], 'forwards')
    assert _pids(out['ANA'], 'defense') == [EXTRA_D_PID]


@pytest.mark.parametrize('supabase_generated_at', [None, '', 'not-a-timestamp'])
def test_unknown_supabase_timestamp_keeps_legacy_behaviour(tmp_path, supabase_generated_at):
    out = _supabase_side(generated_at=supabase_generated_at)
    routes._merge_gp_est_from_json(out, _snapshot(STALE_JSON_GENERATED_AT, tmp_path))

    assert TRADED_AWAY_PID in _pids(out['ANA'], 'forwards')


def test_unknown_snapshot_timestamp_keeps_legacy_behaviour(tmp_path):
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, _snapshot(None, tmp_path))

    assert TRADED_AWAY_PID in _pids(out['ANA'], 'forwards')


def test_missing_snapshot_is_a_noop(tmp_path):
    out = _supabase_side()
    routes._merge_gp_est_from_json(out, str(tmp_path / 'absent.json'))

    assert _pids(out['ANA'], 'forwards') == [CURRENT_LW_PID]


# ── timestamp helpers ────────────────────────────────────────────────────────

@pytest.mark.parametrize('value', [
    '2026-09-28T19:04:40.711208+00:00',
    '2026-09-28T19:04:40Z',
    '2026-09-28T19:04:40',
])
def test_parse_iso_utc_accepts_offset_z_and_naive(value):
    parsed = routes._parse_iso_utc(value)
    assert parsed is not None
    assert parsed.year == 2026 and parsed.month == 9 and parsed.day == 28
    assert parsed.tzinfo is not None


@pytest.mark.parametrize('value', [None, '', '   ', 'nope'])
def test_parse_iso_utc_rejects_junk(value):
    assert routes._parse_iso_utc(value) is None


def test_tolerance_boundary():
    tolerance = routes._JSON_SNAPSHOT_SYNC_TOLERANCE_SECONDS
    assert routes._json_snapshot_may_add_players(
        '2026-09-28T19:04:40+00:00', '2026-09-28T19:04:40+00:00'
    )
    assert routes._json_snapshot_may_add_players(
        '2026-09-28T19:04:40+00:00', '2026-09-28T19:04:41+00:00'
    )
    # Just past the window -> stale.
    from datetime import timedelta
    boundary = routes._parse_iso_utc(SUPABASE_GENERATED_AT) - timedelta(seconds=tolerance + 1)
    assert not routes._json_snapshot_may_add_players(boundary.isoformat(), SUPABASE_GENERATED_AT)
