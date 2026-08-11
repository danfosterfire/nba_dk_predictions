"""Tests for the availability window x trend x dispersion ladder.

The three things pinned here are the three that would move every figure in
`availability_window.csv` without raising: which rows a window keeps, where the upper tail
is read, and whether a role-graded dispersion still nests the shared one.
"""

import numpy as np
import pandas as pd

from src.models.availability_window import (RoleGradedBetaBinomial, interval_coverage,
                                            paired_bootstrap, restrict_window,
                                            tail_coverage)


def _seasons(labels, per_season=3):
    return pd.DataFrame({"season": np.repeat(labels, per_season),
                         "gp": np.tile(np.arange(per_season), len(labels))})


def test_restrict_window_keeps_only_the_recent_suffix():
    train = _seasons(["2010-11", "2016-17", "2017-18", "2021-22"])
    kept = restrict_window(train, 2017)
    assert sorted(kept["season"].unique()) == ["2017-18", "2021-22"]
    assert len(kept) == 6


def test_restrict_window_with_no_first_year_is_the_identity():
    # `None` has to be the incumbent EXACTLY, or the reference arm the whole ladder
    # bootstraps against is not the shipped head.
    train = _seasons(["1996-97", "2005-06", "2021-22"])
    kept = restrict_window(train, None)
    assert kept.equals(train)


def test_tail_coverage_upper_uses_each_rows_own_schedule():
    # Two rows with DIFFERENT schedule lengths. A 66-game row's full-schedule boundary is
    # at 66, not at the grid's 82 — reading a fixed 82 would score every shortened season
    # against an unreachable event and silently report ~0 for the upper tail.
    max_games = 82
    pmf = np.zeros((2, max_games + 1))
    pmf[0, 66] = 1.0                      # played every one of a 66-game season
    pmf[1, 82] = 1.0                      # played every one of an 82-game season
    y = np.array([66, 82])
    n = np.array([66, 82])

    rows = {r["tail"]: r for r in tail_coverage(pmf, y, n)}
    assert np.isclose(rows["full_schedule"]["predicted"], 1.0)
    assert np.isclose(rows["full_schedule"]["observed"], 1.0)
    assert np.isclose(rows["full_schedule"]["error"], 0.0)


def test_tail_coverage_recovers_a_known_low_tail():
    # One row certain to play 5 games, one certain to play 50. P(GP < 10) is exactly 0.5
    # both predicted and observed; P(GP < 41) is also 0.5; P(GP < 60) is 1.0.
    pmf = np.zeros((2, 83))
    pmf[0, 5] = 1.0
    pmf[1, 50] = 1.0
    rows = {r["tail"]: r for r in tail_coverage(pmf, np.array([5, 50]), np.array([82, 82]))}
    assert np.isclose(rows["below_10"]["predicted"], 0.5)
    assert np.isclose(rows["below_10"]["observed"], 0.5)
    assert np.isclose(rows["below_41"]["predicted"], 0.5)
    assert np.isclose(rows["below_60"]["predicted"], 1.0)


def test_tail_coverage_catches_a_confidently_wrong_model():
    # The failure mode the metric exists for: a model certain nobody misses time, on rows
    # where half of them did. A Brier score would report 0.5; the coverage error reports
    # the full -0.5 gap, which is the number the defect is stated in.
    pmf = np.zeros((2, 83))
    pmf[:, 82] = 1.0
    rows = {r["tail"]: r for r in tail_coverage(pmf, np.array([5, 82]), np.array([82, 82]))}
    assert np.isclose(rows["below_10"]["predicted"], 0.0)
    assert np.isclose(rows["below_10"]["observed"], 0.5)
    assert np.isclose(rows["below_10"]["error"], -0.5)


def test_interval_coverage_is_exact_for_a_point_mass():
    pmf = np.zeros((3, 83))
    pmf[np.arange(3), [10, 40, 70]] = 1.0
    y = np.array([10, 40, 70])
    for level in (0.5, 0.8, 0.95):
        assert np.isclose(interval_coverage(pmf, y, level), 1.0)
    assert np.isclose(interval_coverage(pmf, np.array([11, 41, 71]), 0.95), 0.0)


def test_role_graded_rho_nests_the_shared_dispersion():
    # One dispersion repeated across every bucket must reproduce the shared-rho predictive
    # EXACTLY, the way `S = 0` nests the year effect in the .stan sources. Without this the
    # role arm's gain could be an artifact of a different pmf construction rather than of
    # the grading.
    model = object.__new__(RoleGradedBetaBinomial)
    model.pooled_rho = 0.28
    model.rho_by_role = {label: 0.28 for label in ["<12 mpg", "12-24", "24-30", "30+ mpg"]}
    df = pd.DataFrame({"team_games": [82, 82, 66], "minutes_per_game_lag1": [5.0, 20.0, 34.0]})
    model.predict_mean = lambda frame: np.array([0.6, 0.75, 0.9])[: len(frame)]

    graded = model.predict_pmf(df, 82)

    from src.models.availability import predictive_pmf
    shared = predictive_pmf(df["team_games"].to_numpy(), model.predict_mean(df), 0.28, 82)
    assert np.allclose(graded, shared)
    assert np.allclose(graded.sum(axis=1), 1.0)


def test_role_graded_rho_falls_back_for_an_unseen_bucket():
    model = object.__new__(RoleGradedBetaBinomial)
    model.pooled_rho = 0.28
    model.rho_by_role = {"<12 mpg": 0.31}          # the other three never met MIN_ROLE_ROWS
    df = pd.DataFrame({"team_games": [82, 82], "minutes_per_game_lag1": [5.0, 34.0]})
    model.predict_mean = lambda frame: np.array([0.6, 0.9])[: len(frame)]
    pmf = model.predict_pmf(df, 82)
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert not np.allclose(pmf[0], pmf[1])


def test_paired_bootstrap_brackets_the_observed_difference():
    rng = np.random.default_rng(0)
    reference = rng.normal(10.0, 1.0, 500)
    arm = reference - 0.3                            # a uniform improvement of 0.3
    mean, lo, hi = paired_bootstrap(arm, reference, reps=500)
    assert np.isclose(mean, -0.3)
    # Pairing is the point: a uniform shift has zero variance in the DIFFERENCE, so the
    # interval collapses onto it. An unpaired bootstrap would put it near +-0.12.
    assert lo <= mean <= hi
    assert hi - lo < 1e-9
