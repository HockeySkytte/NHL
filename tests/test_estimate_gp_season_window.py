"""Estimated Games must ignore the season it is projecting.

`extract_gp_history` used to keep every NHL regular season in `seasonTotals`,
including the one in progress. As soon as the 20262027 opener was played, that
season's 1 game became the most heavily weighted entry in the average and
collapsed every estimate (Auston Matthews: 68 -> 32 games). The season
simulations weight each player's per-game rates by gp_est, so this silently
gutted them.
"""
import os
import sys

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from scripts.estimate_gp import (  # noqa: E402
    TARGET_SEASON,
    estimate_gp_2027,
    extract_gp_history,
)


def _landing(*seasons):
    """`seasons` is a sequence of (season, gamesPlayed)."""
    return {'seasonTotals': [
        {'leagueAbbrev': 'NHL', 'gameTypeId': 2, 'season': s, 'gamesPlayed': gp}
        for s, gp in seasons
    ]}


def test_in_progress_season_is_excluded():
    landing = _landing((20262027, 1), (20252026, 60), (20242025, 67), (20232024, 81))

    assert extract_gp_history(landing) == [60, 67, 81]


def test_future_seasons_are_excluded():
    landing = _landing((20272028, 3), (20262027, 1), (20252026, 60))

    assert extract_gp_history(landing) == [60]


def test_completed_seasons_are_kept_most_recent_first():
    landing = _landing((20252026, 79), (20242025, 78), (20232024, 80))

    assert extract_gp_history(landing) == [79, 78, 80]


def test_non_nhl_and_playoff_rows_are_ignored():
    landing = {'seasonTotals': [
        {'leagueAbbrev': 'AHL', 'gameTypeId': 2, 'season': 20252026, 'gamesPlayed': 70},
        {'leagueAbbrev': 'NHL', 'gameTypeId': 3, 'season': 20252026, 'gamesPlayed': 20},
        {'leagueAbbrev': 'NHL', 'gameTypeId': 2, 'season': 20252026, 'gamesPlayed': 60},
    ]}

    assert extract_gp_history(landing) == [60]


def test_traded_player_games_are_summed():
    landing = {'seasonTotals': [
        {'leagueAbbrev': 'NHL', 'gameTypeId': 2, 'season': 20252026, 'gamesPlayed': 40},
        {'leagueAbbrev': 'NHL', 'gameTypeId': 2, 'season': 20252026, 'gamesPlayed': 39},
    ]}

    assert extract_gp_history(landing) == [79]


def test_zero_game_seasons_are_skipped():
    landing = _landing((20262027, 0), (20252026, 60))

    assert extract_gp_history(landing) == [60]


def test_missing_season_totals_is_empty():
    assert extract_gp_history({}) == []
    assert extract_gp_history({'seasonTotals': None}) == []


def test_estimate_is_not_collapsed_by_the_open_season():
    """The regression that motivated the exclusion."""
    with_open_season = _landing((20262027, 1), (20252026, 60), (20242025, 67), (20232024, 81))
    without_open_season = _landing((20252026, 60), (20242025, 67), (20232024, 81))

    collapsed, _ = estimate_gp_2027(extract_gp_history(with_open_season), 8479318, 'F')
    stable, _ = estimate_gp_2027(extract_gp_history(without_open_season), 8479318, 'F')

    assert collapsed == stable == 68


def test_target_season_constant_matches_the_projected_season():
    assert TARGET_SEASON == 20262027
