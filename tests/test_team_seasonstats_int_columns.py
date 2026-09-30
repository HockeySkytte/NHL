"""season_stats_teams counters must reach Postgres as integers.

`rebuild_team_seasonstats_from_supabase` outer-merges the for/against/TOI
frames. Any key present in only one frame introduces NaN, which upcasts the
INT-declared counting columns to float64; PostgREST then sends "44.0" and
Postgres rejects the whole upsert with 22P02. A fresh season hits this on its
first rebuild, so the team pages stayed empty.
"""
import os
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.update_data import coerce_team_stat_int_columns  # noqa: E402


def test_float_columns_are_cast_to_int():
    df = pd.DataFrame([{
        'season_state': 'regular', 'strength_state': 'SH', 'team': 'BOS',
        'gp': 0.0, 'cf': 0.0, 'ca': 1.0, 'ff': 0.0, 'fa': 1.0,
        'sf': 0.0, 'sa': 1.0, 'gf': 0.0, 'ga': 0.0,
    }])

    out = coerce_team_stat_int_columns(df)

    for col in ('gp', 'cf', 'ca', 'ff', 'fa', 'sf', 'sa', 'gf', 'ga'):
        assert out[col].dtype == 'int64', col
        assert isinstance(out[col].iloc[0], (int,)) or int(out[col].iloc[0]) == out[col].iloc[0]


def test_nan_counters_become_zero():
    df = pd.DataFrame([{
        'season_state': 'regular', 'strength_state': 'PP', 'team': 'CAR',
        'gp': float('nan'), 'cf': float('nan'), 'ca': 3.0,
    }])

    out = coerce_team_stat_int_columns(df)

    assert out['gp'].iloc[0] == 0
    assert out['cf'].iloc[0] == 0
    assert out['ca'].iloc[0] == 3


def test_real_columns_are_left_alone():
    df = pd.DataFrame([{
        'season_state': 'regular', 'strength_state': '5v5', 'team': 'TOR',
        'toi': 55.3833, 'xgf_f': 2.37527, 'xga_f': 2.142, 'pim_for': 4.0,
    }])

    out = coerce_team_stat_int_columns(df)

    assert out['toi'].iloc[0] == pytest.approx(55.3833)
    assert out['xgf_f'].iloc[0] == pytest.approx(2.37527)
    assert out['pim_for'].iloc[0] == pytest.approx(4.0)


def test_missing_columns_are_tolerated():
    df = pd.DataFrame([{'team': 'EDM', 'cf': 1.0}])

    out = coerce_team_stat_int_columns(df)

    assert out['cf'].iloc[0] == 1
    assert list(out.columns) == ['team', 'cf']


def test_int_columns_stay_lossless():
    df = pd.DataFrame([{'team': 'VAN', 'cf': 52, 'ca': 44, 'gp': 1}])

    out = coerce_team_stat_int_columns(df)

    assert out['cf'].iloc[0] == 52
    assert out['ca'].iloc[0] == 44
    assert out['gp'].iloc[0] == 1
