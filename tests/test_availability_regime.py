"""Tests for the regime axis and for shrinkage toward the long-window fit.

Four things are pinned, and each of them is a claim the round's conclusions rest on rather
than a property of the code:

1. **The indicators are what they say they are.** `regime_target` is a fact about the
   season being predicted and `regime_lag` about the season the features came from, and
   swapping them silently would leave every arm fitting and every number wrong.
2. **The target indicator is scored at zero.** Production predicts an ordinary season, so
   an arm that reads its realized value at scoring time is being handed the answer.
3. **Shrinkage nests both of its endpoints.** `lambda = inf` reproduces the long-window
   coefficients exactly; `lambda = 0` reproduces the plain short-window fit's objective.
   A one-knob extension that cannot reach its own special cases is a second model.
4. **The weight family's zero is exclusion.** `w = 0` and dropping the rows have to be the
   same fit, or "exclude" is not the limit of the knob it is documented as being.
"""

import numpy as np
import pandas as pd
from scipy.stats import betabinom
from sklearn.preprocessing import StandardScaler

from src.models.availability import _ab, _sigmoid
from src.models.availability_regime import (REGIME_CORE, REGIME_LAG_COL, REGIME_TARGET_COL,
                                            REGIME_YEARS, ShrunkBetaBinomialGLM,
                                            add_regime_terms, feasible_lookbacks,
                                            fit_weighted_graded, lambda_vector,
                                            predict_frame, regime_identification,
                                            regime_weights)
from src.models.availability_weighting import WeightedBetaBinomialGLM


def _design(n_per_season: int = 160, seed: int = 0) -> pd.DataFrame:
    """A synthetic availability design with every column the head reads.

    Deliberately spans the regime block and both sides of it, because most of what is
    pinned here is about *which seasons* a row belongs to.
    """
    from src.models.availability import FEATURE_COLS

    rng = np.random.default_rng(seed)
    seasons = [f"{y}-{str(y + 1)[2:]}" for y in range(2012, 2022)]
    rows = pd.DataFrame({"season": np.repeat(seasons, n_per_season)})
    n = len(rows)
    for col in FEATURE_COLS:
        rows[col] = rng.normal(size=n)
    # `minutes_per_game_lag1` cuts the role buckets, so it has to look like minutes.
    rows["minutes_per_game_lag1"] = rng.uniform(2.0, 38.0, n)
    rows["team_games"] = 82
    rows["gp"] = rng.binomial(82, 0.72, n)
    return rows


def test_target_indicator_marks_the_season_being_predicted():
    rows = pd.DataFrame({"season": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]})
    (out,), names = add_regime_terms([rows], "target")
    assert names == [REGIME_TARGET_COL]
    assert out[REGIME_TARGET_COL].tolist() == [0.0, 1.0, 1.0, 1.0, 0.0]


def test_lag_indicator_marks_the_season_the_features_came_from():
    rows = pd.DataFrame({"season": ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]})
    (out,), names = add_regime_terms([rows], "lag")
    assert names == [REGIME_LAG_COL]
    # 2020-21 is predicted from 2019-20, and 2022-23 from 2021-22 — the validation row the
    # lag channel is live on and the target channel is not.
    assert out[REGIME_LAG_COL].tolist() == [0.0, 0.0, 1.0, 1.0, 1.0]


def test_the_two_indicators_are_not_the_same_column():
    rows = pd.DataFrame({"season": ["2019-20", "2022-23"]})
    (out,), names = add_regime_terms([rows], "both")
    assert names == [REGIME_TARGET_COL, REGIME_LAG_COL]
    assert out[REGIME_TARGET_COL].tolist() == [1.0, 0.0]
    assert out[REGIME_LAG_COL].tolist() == [0.0, 1.0]


def test_core_block_excludes_the_bubble_season():
    rows = pd.DataFrame({"season": ["2019-20", "2020-21", "2021-22"]})
    (out,), _ = add_regime_terms([rows], "target", REGIME_CORE)
    assert out[REGIME_TARGET_COL].tolist() == [0.0, 1.0, 1.0]


def test_scoring_forces_the_target_indicator_to_zero_and_leaves_the_lag_alone():
    rows = pd.DataFrame({"season": ["2020-21", "2022-23"]})
    (out,), names = add_regime_terms([rows], "both")
    scored = predict_frame(out, names)
    assert scored[REGIME_TARGET_COL].tolist() == [0.0, 0.0]
    # The prior season is observed before the draft, so it keeps its realized value.
    assert scored[REGIME_LAG_COL].tolist() == [1.0, 1.0]


def test_regime_weights_are_one_off_the_block_and_zero_is_exclusion():
    rows = pd.DataFrame({"season": ["2018-19", "2019-20", "2021-22", "2022-23"]})
    assert np.allclose(regime_weights(rows, 0.5), [1.0, 0.5, 0.5, 1.0])
    assert np.allclose(regime_weights(rows, 1.0), 1.0)
    assert np.allclose(regime_weights(rows, 0.0), [1.0, 0.0, 0.0, 1.0])


def test_weight_zero_is_the_same_fit_as_dropping_the_rows():
    # "Exclude" is documented as the limit of the weight knob rather than a separate arm,
    # and that is only true if a zero-weighted row contributes nothing to the mean OR the
    # dispersion. The scaler is the one thing that legitimately differs, so both fits are
    # handed the same one.
    from src.models.availability import FEATURE_COLS

    rows = _design()
    scaler = StandardScaler().fit(rows[list(FEATURE_COLS)].to_numpy(dtype=float))
    w = regime_weights(rows, 0.0)
    weighted = fit_weighted_graded(rows, w, scaler=scaler)
    dropped = fit_weighted_graded(rows[w > 0], np.ones(int((w > 0).sum())), scaler=scaler)
    assert np.allclose(weighted.beta, dropped.beta, atol=1e-6)
    assert np.isclose(weighted.rho, dropped.rho, atol=1e-8)


def test_lambda_vector_frees_the_named_blocks_and_covers_the_intercept():
    from src.models.availability import FEATURE_COLS

    lam = lambda_vector(list(FEATURE_COLS), 16.0, ("intercept", "workload"))
    assert len(lam) == len(FEATURE_COLS) + 1
    assert lam[0] == 0.0                       # the intercept is block "intercept"
    assert (lam[-4:] == 0.0).all()             # WORKLOAD_COLS are the last four features
    assert (lam[1:-4] == 16.0).all()
    # With no free blocks the penalty reaches every column INCLUDING the intercept, which
    # is where it differs from `l2` and is the point of shrinking toward an anchor.
    assert (lambda_vector(list(FEATURE_COLS), 4.0) == 4.0).all()


def test_infinite_lambda_pins_the_coefficients_to_the_anchor_exactly():
    from src.models.availability import FEATURE_COLS

    rows = _design(seed=3)
    features = list(FEATURE_COLS)
    scaler = StandardScaler().fit(rows[features].to_numpy(dtype=float))
    anchor = np.linspace(-0.5, 0.5, len(features) + 1)
    model = ShrunkBetaBinomialGLM(anchor=anchor,
                                  lam=lambda_vector(features, np.inf),
                                  scaler=scaler, features=features).fit(rows)
    assert np.array_equal(model.beta, anchor)
    assert model.n_pinned == len(features) + 1


def test_free_blocks_move_while_the_shrunk_ones_stay_on_the_anchor():
    from src.models.availability import FEATURE_COLS

    rows = _design(seed=4)
    features = list(FEATURE_COLS)
    scaler = StandardScaler().fit(rows[features].to_numpy(dtype=float))
    anchor = np.full(len(features) + 1, 0.3)
    lam = lambda_vector(features, np.inf, ("intercept", "workload"))
    model = ShrunkBetaBinomialGLM(anchor=anchor, lam=lam, scaler=scaler,
                                  features=features).fit(rows)
    pinned = ~np.isfinite(lam)          # `inf` is the SHRUNK block; the free ones carry 0
    assert np.array_equal(model.beta[pinned], anchor[pinned])
    # The five drifting columns are fitted *in the presence of* the pinned ones, which is
    # the whole difference between this and splicing a coefficient block across two fits.
    assert np.abs(model.beta[~pinned] - anchor[~pinned]).max() > 1e-3


def test_zero_lambda_reaches_the_plain_short_window_objective():
    # Not the coefficient vector: the lag block is collinear by construction, so the two
    # starts land in different places along a flat direction. The objective is what the
    # nesting claim is about.
    from src.models.availability import FEATURE_COLS

    rows = _design(seed=5)
    features = list(FEATURE_COLS)
    plain = WeightedBetaBinomialGLM(features=features).fit(rows)
    shrunk = ShrunkBetaBinomialGLM(anchor=np.zeros(len(features) + 1),
                                   lam=lambda_vector(features, 0.0),
                                   scaler=plain.scaler, features=features).fit(rows)

    def penalized(model) -> float:
        mu = _sigmoid(model._design(rows) @ model.beta)
        a, b = _ab(mu, model.rho)
        y, n = rows["gp"].to_numpy(), rows["team_games"].to_numpy()
        return (-float(betabinom.logpmf(y, n, a, b).sum())
                + float(model.beta[1:] @ model.beta[1:]))

    assert abs(penalized(shrunk) - penalized(plain)) < 1e-2


def test_feasible_lookbacks_drops_a_window_an_exclusion_arm_cannot_fit():
    # 400 rows is the guard (100 per role bucket, four buckets). A three-season lookback
    # at an origin whose window is two-thirds regime seasons cannot clear it once those
    # seasons are excluded, and pooling a ragged arm against a complete reference would
    # compare two different samples.
    rows = pd.DataFrame({"season": np.repeat(
        [f"{y}-{str(y + 1)[2:]}" for y in range(2016, 2022)], 380)})
    arms = {"exclude": {"kind": "weight", "weight": 0.0, "years": REGIME_YEARS}}
    assert feasible_lookbacks(rows, [2021], (8,), arms) == (8,)
    assert feasible_lookbacks(rows, [2021], (3,), arms) == ()
    # With no exclusion arm the same window is fine — the guard is about the arm, not the
    # lookback.
    assert feasible_lookbacks(rows, [2021], (3,), {}) == (3,)


def test_the_lag_flag_implies_the_target_flag_inside_the_fitting_window():
    # The identification fact the lag arm's validation reading turns on: the disrupted
    # seasons are consecutive, so every fitting row with a regime lag also has a regime
    # target, and the two coefficients are separated only by the validation rows.
    fit_rows = pd.DataFrame({"season": [f"{y}-{str(y + 1)[2:]}" for y in range(2012, 2022)]})
    val = pd.DataFrame({"season": ["2022-23", "2023-24"]})
    out = regime_identification(fit_rows, val)
    assert out["lag_implies_target_train"] == 1.0
    assert out["n_regime_target_train"] == len(REGIME_YEARS)
    assert out["n_regime_lag_train"] == 2.0        # 2020-21 and 2021-22
    assert out["n_regime_lag_val"] == 1.0          # 2022-23 only
    assert out["n_regime_target_val"] == 0.0
