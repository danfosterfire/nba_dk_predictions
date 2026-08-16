"""Tests for P4(b) — the preseason against the draft-bucket imputation for no-prior players.

The round compares three arms over one estimator, so the things worth pinning are the ways
the comparison could quietly be about something other than information:

1. **The two endpoints nest exactly.** `k = 0` must be the `preseason` arm and a very large
   `k` must be `draft_bucket`, or a selected interior `k` is not on the ladder the round
   claims to have run — the same nesting discipline `n_rho = 1` and `U_n = 0` carry on the
   fitted heads.
2. **A missing preseason costs nothing rather than imputing zero.** These rows exist and
   `min_pre` is 0 for them, so the blend has to hand them the bucket prior by its own
   arithmetic; a `nan` or a 0 prediction would look like a signal.
3. **Nothing sees the target season.** The bucket prior is an expanding window, and a
   contaminated one would beat the preseason for the wrong reason.
4. **The units match.** The share comparison is `minutes_share` against `mpg_pre / 48`, not
   against the panel's `min_share_pre` — the two differ by a factor of ~5 and comparing
   them measures a unit conversion. This one is a regression test: it was wrong first.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.rookie_priors import (INCUMBENT, MIN_BUCKET, REGULATION_LENGTH, TARGETS,
                                      arm_predictions, attach_preseason, bucket_priors,
                                      paired_absolute, score_rows)


def _rows(spec: list[tuple[str, int, str, float, float]]) -> pd.DataFrame:
    """(season, player_id, draft_bucket, target, min_pre) → no-prior rows."""
    return pd.DataFrame([{"season": s, "player_id": p, "draft_bucket": b,
                          "target": t, "min_pre": m}
                         for s, p, b, t, m in spec])


def _priors(rows: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    return bucket_priors(rows, "target", seasons)


def test_the_two_endpoints_of_the_blend_are_the_two_endpoint_arms_exactly():
    """`k = 0` is `preseason` and `k -> inf` is `draft_bucket`. If either fails, an interior
    `k` selected on the inner carve is an arm the ladder never contained."""
    seasons = ["2018-19", "2019-20"]
    history = _rows([("2018-19", i, "lottery", 0.30, 100.0) for i in range(MIN_BUCKET + 5)])
    target = _rows([("2019-20", 900, "lottery", np.nan, 120.0)]).assign(pre=0.11)
    priors = _priors(history, seasons)
    priors = priors[priors["season"] == "2019-20"]

    at_zero = arm_predictions(target, priors, "pre", 0.0)
    at_inf = arm_predictions(target, priors, "pre", 1e12)
    assert np.allclose(at_zero["shrunk"], at_zero["preseason"])
    assert np.allclose(at_inf["shrunk"], at_inf[INCUMBENT])
    # And the two endpoints are genuinely different, so the equalities above are not both
    # true by the arms having collapsed onto each other.
    assert not np.allclose(at_zero["shrunk"], at_inf["shrunk"])


def test_a_row_with_no_preseason_gets_the_bucket_prior_from_its_own_volume():
    """Not from a special case. `min_pre` is 0 for a player who never took the floor, so
    `w = 0/(0+k)` is already 0 and all three arms agree on him — which is what makes the
    comparison a comparison about *players who have a preseason*."""
    seasons = ["2018-19", "2019-20"]
    history = _rows([("2018-19", i, "undrafted", 0.20, 90.0) for i in range(MIN_BUCKET + 5)])
    target = _rows([("2019-20", 900, "undrafted", np.nan, 0.0)]).assign(pre=np.nan)
    priors = _priors(history, seasons)
    priors = priors[priors["season"] == "2019-20"]
    preds = arm_predictions(target, priors, "pre", 40.0)
    assert np.isfinite(list(preds.values())).all()
    assert np.allclose(preds["shrunk"], preds[INCUMBENT])
    assert np.allclose(preds["preseason"], preds[INCUMBENT])


def test_the_bucket_prior_never_sees_its_own_target_season():
    """Point-in-time by construction, and asserted rather than assumed: adding a wild
    target-season row must not move the prior that season is scored against."""
    seasons = ["2018-19", "2019-20"]
    history = _rows([("2018-19", i, "lottery", 0.30, 80.0) for i in range(MIN_BUCKET + 5)])
    before = _priors(history, seasons)
    contaminated = pd.concat(
        [history, _rows([("2019-20", 900 + i, "lottery", 9.99, 80.0) for i in range(50)])],
        ignore_index=True)
    after = _priors(contaminated, seasons)
    for frame in (before, after):
        frame.set_index(["season", "draft_bucket"], inplace=True)
    assert np.isclose(before.loc[("2019-20", "lottery"), "prior"],
                      after.loc[("2019-20", "lottery"), "prior"])


def test_a_thin_bucket_takes_the_pooled_mean_rather_than_its_own():
    """`MIN_BUCKET` is `rookie_share_priors`' own fallback made explicit. Below it the cell's
    mean is dominated by its sampling error and would read as a gradient that is not there."""
    seasons = ["2018-19", "2019-20"]
    history = _rows([("2018-19", i, "undrafted", 0.10, 80.0) for i in range(MIN_BUCKET + 5)]
                    + [("2018-19", 500 + i, "lottery_top5", 0.90, 80.0) for i in range(3)])
    priors = _priors(history, seasons).set_index(["season", "draft_bucket"])
    thin = priors.loc[("2019-20", "lottery_top5")]
    fat = priors.loc[("2019-20", "undrafted")]
    assert np.isclose(thin["prior"], thin["pooled_prior"])
    assert not np.isclose(thin["prior"], 0.90)
    assert np.isclose(fat["prior"], 0.10)


def test_the_share_target_compares_minutes_over_game_length_on_both_sides():
    """The regression test. `minutes_share` is minutes over GAME LENGTH and the panel's
    `min_share_pre` is a share of the TEAM's preseason minutes — a factor of ~5 apart, so
    pairing them measures a unit conversion and reports the preseason as badly biased low."""
    share_target = [t for t in TARGETS if t[0] == "minutes_share"]
    assert share_target and share_target[0][2] == "pre_minutes_share"

    panel = pd.DataFrame({"season": ["2019-20"], "player_id": [7], "mpg_pre": [24.0],
                          "min_pre": [96.0], "min_share_pre": [0.06],
                          **{c: [1.0] for _, _, c, _ in TARGETS
                             if c != "pre_minutes_share"}})
    out = attach_preseason(pd.DataFrame({"season": ["2019-20"], "player_id": [7]}), panel)
    assert np.isclose(out["pre_minutes_share"].iloc[0], 24.0 / REGULATION_LENGTH)
    assert out["has_preseason"].iloc[0] == 1.0


def test_a_player_with_no_panel_row_is_flagged_and_carries_zero_volume():
    panel = pd.DataFrame({"season": ["2019-20"], "player_id": [7], "mpg_pre": [24.0],
                          "min_pre": [96.0], "min_share_pre": [0.06],
                          **{c: [1.0] for _, _, c, _ in TARGETS
                             if c != "pre_minutes_share"}})
    out = attach_preseason(pd.DataFrame({"season": ["2019-20", "2019-20"],
                                         "player_id": [7, 8]}), panel)
    missing = out[out["player_id"] == 8].iloc[0]
    assert missing["has_preseason"] == 0.0
    assert missing["min_pre"] == 0.0
    assert pd.isna(missing["pre_minutes_share"])


def test_the_minutes_weighted_score_is_a_weight_and_not_a_filter():
    """A per-36 over twelve realized minutes is noise on both sides, and the honest way to
    say so is a weight — dropping those rows would be selection on the outcome. So a
    zero-minute row must still be scored, and must still move the unweighted column."""
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([1.0, 2.0, 9.0])
    heavy = score_rows(y, pred, np.array([100.0, 100.0, 0.0]))
    flat = score_rows(y, pred, np.array([1.0, 1.0, 1.0]))
    assert heavy["n"] == flat["n"] == 3
    assert heavy["mae"] == flat["mae"]                      # unweighted ignores the weights
    assert heavy["mae_minutes_weighted"] < flat["mae_minutes_weighted"]


def test_the_paired_interval_is_zero_width_against_itself():
    """The reference arm's own row must read exactly 0 with a degenerate interval, which is
    what makes `mae_vs_incumbent` readable as a margin rather than as noise around one."""
    err = np.array([0.5, -1.0, 2.0, -0.25])
    delta, lo, hi = paired_absolute(err, err)
    assert delta == pytest.approx(0.0)
    assert (lo, hi) == pytest.approx((0.0, 0.0))
