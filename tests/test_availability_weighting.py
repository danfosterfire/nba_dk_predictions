"""Tests for the four fitting-window optimizations.

The nesting properties are the point. Each experiment claims to *generalize* the incumbent
rather than replace it, and a generalization that does not reproduce its own special case
is not one — it is a second model whose comparison against the first is meaningless.
"""

import numpy as np
import pandas as pd

from src.models.availability import _neg_loglik
from src.models.availability_weighting import (BLOCKS, block_indices, decay_weights,
                                               fit_dispersion_weighted,
                                               weighted_neg_loglik)


def test_uniform_weights_reproduce_the_unweighted_likelihood():
    # The whole decay experiment rests on this: `lambda = 1.0` must be the incumbent, not
    # something near it.
    rng = np.random.default_rng(0)
    n = np.full(40, 82)
    y = rng.integers(0, 83, 40)
    mu = np.full(40, 0.7)
    w = np.ones(40)
    assert np.isclose(weighted_neg_loglik(y, n, mu, 0.25, w), _neg_loglik(y, n, mu, 0.25))


def test_weighted_likelihood_scales_with_duplicated_rows():
    # A weight of 2 has to be worth exactly two copies of the row, or the decay is not a
    # reweighting of the same likelihood.
    n = np.array([82, 82])
    y = np.array([70, 30])
    mu = np.array([0.8, 0.4])
    doubled = weighted_neg_loglik(y, n, mu, 0.25, np.array([2.0, 2.0]))
    stacked = _neg_loglik(np.tile(y, 2), np.tile(n, 2), np.tile(mu, 2), 0.25)
    assert np.isclose(doubled, stacked)


def test_weighted_dispersion_matches_the_unweighted_fit_at_uniform_weights():
    rng = np.random.default_rng(1)
    n = np.full(200, 82)
    mu = np.full(200, 0.7)
    y = rng.binomial(82, 0.7, 200)
    from src.models.availability import fit_dispersion
    assert np.isclose(fit_dispersion_weighted(y, n, mu, np.ones(200)),
                      fit_dispersion(y, n, mu), atol=1e-6)


def test_decay_weights_are_one_for_the_most_recent_season():
    rows = pd.DataFrame({"season": ["2018-19", "2019-20", "2020-21"]})
    w = decay_weights(rows, origin=2021, lam=0.5)
    # ages are 3, 2, 1 seasons before the origin
    assert np.allclose(w, [0.125, 0.25, 0.5])
    assert np.allclose(decay_weights(rows, 2021, 1.0), 1.0)


def test_decay_of_one_is_uniform_for_any_span():
    rows = pd.DataFrame({"season": [f"{y}-{str(y + 1)[2:]}" for y in range(1996, 2022)]})
    assert np.allclose(decay_weights(rows, 2022, 1.0), 1.0)


def test_block_indices_partition_the_design_and_reserve_column_zero():
    features = ["gp_share_lag1", "gp_share_lag2", "gp_share_lag3",
                "minutes_per_game_lag1", "minutes_per_game_lag2",
                "minutes_per_game_lag3", "total_minutes_lag1",
                "trailing_missed_lag1", "end_play_rate_lag1", "n_spells_lag1",
                "longest_spell_lag1", "age", "age_sq", "career_year",
                "n_prior_seasons", "playoff_games_lag1", "playoff_mpg_lag1",
                "playoff_minutes_share_lag1", "career_minutes_lag1"]
    idx = block_indices(features)
    assert list(idx["intercept"]) == [0]
    covered = np.concatenate([idx[b] for b in BLOCKS])
    # Every design column is claimed exactly once — a block that silently dropped a column
    # would make `all_blocks` differ from a plain short-window fit for the wrong reason.
    assert sorted(covered.tolist()) == list(range(len(features) + 1))
    assert len(set(covered.tolist())) == len(covered)


def test_block_indices_are_offset_by_the_intercept():
    features = ["gp_share_lag1", "age"]
    idx = block_indices(features)
    # `gp_share_lag1` is design column 1, not 0 — an off-by-one here would splice the
    # intercept whenever it meant to splice the first slope.
    assert list(idx["gp_share"]) == [1]
