"""Tests for the role-graded injection sigma — `docs/draw-time-calibration-plan.md`.

None of these draws from the composition head. The search takes a `sigma -> (draws x units)`
closure, so the thing worth pinning is the search itself: that it finds a known per-bucket
answer, that the ship rule is applied from the interval-style bar rather than after seeing
which buckets agreed, and that a shared sigma stays exactly representable.

Two failure modes drove the choices here and both are silent:

- **a search that pools its objective** would let the two largest buckets choose for the two
  smallest, and still print four different numbers;
- **a ship rule read after the numbers** turns four independent chances at a wiggle into
  four "findings". `ship_rule` is code for that reason, and this file pins both sides of it.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import minutes_role_sigma as RS


# ── Synthetic worlds ──────────────────────────────────────────────────────────

def _known_optimum_draw(bins: np.ndarray, n_draws: int = 800, seed: int = 3):
    """A `draw` centred at 0 whose spread is each unit's own bucket sigma.

    The truth lives in `y`, which is drawn once at `N(0, truth[bin])`. CRPS is a proper
    scoring rule, so the expected score is minimized where the predictive equals the
    generating distribution — bucket by bucket, at `sigma = truth[bin]`.

    Centring the predictive on `y` instead would be degenerate: every unit's realized value
    would be its own predictive mean, the realized spread would be zero, and every bucket's
    optimum would be the smallest sigma on the grid regardless of what was planted.
    """
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(size=(n_draws, len(bins)))

    def draw(sigma) -> np.ndarray:
        return base * RS.broadcast_sigma(sigma)[bins][None, :]
    return draw


def _bucketed_truth(n_per_bin: int = 120, truth=(0.6, 0.45, 0.3, 0.225), seed: int = 11):
    """Units in four buckets whose realized deviation sd IS the bucket's true sigma.

    The four values are ON the grid and interior to it, so a recovered optimum is a
    recovered optimum rather than a boundary the search was pinned against.
    """
    bins = np.repeat(np.arange(RS.N_ROLE_BINS), n_per_bin)
    rng = np.random.default_rng(seed)
    y = rng.standard_normal(len(bins)) * np.asarray(truth)[bins]
    return bins, y, np.asarray(truth, dtype=float)


# ── The pieces ────────────────────────────────────────────────────────────────

def test_a_shared_sigma_is_one_cell_and_a_graded_one_is_four():
    """The artifact has to be readable without parsing intent out of a float column."""
    assert RS.format_sigma(0.375) == "0.375"
    assert RS.format_sigma(np.array([0.6, 0.45, 0.375, 0.3])) == "0.6|0.45|0.375|0.3"
    assert RS.scalar_or_nan(0.375) == 0.375
    assert np.isnan(RS.scalar_or_nan(np.array([0.6, 0.45, 0.375, 0.3])))
    np.testing.assert_array_equal(RS.broadcast_sigma(0.375),
                                  np.full(RS.N_ROLE_BINS, 0.375))


def test_sd_ratio_is_realized_error_over_predictive_spread():
    """The number the injection exists to move to 1, and it has to be that way round: a
    predictive twice as wide as the error reads 0.5, which is the over-dispersed end the
    star bucket sits at."""
    y = np.zeros(500)
    rng = np.random.default_rng(0)
    totals = rng.standard_normal(size=(2000, 500)) * 2.0
    # Predictive sd 2.0, realized error ~2.0 (the draw means scatter about 0 far less), so
    # the ratio is well under 1 — the arm is over-dispersed for a target it nails.
    assert RS.sd_ratio(totals, y) < 0.2
    assert RS.sd_ratio(totals + 10.0, y) > RS.sd_ratio(totals, y)


def test_score_block_reports_both_pit_tails_separately():
    """The fringe miss is asymmetric — seasons collapse below the predictive far more often
    than they exceed it — so a symmetric KS or sd_ratio alone would hide the shape."""
    rng = np.random.default_rng(1)
    totals = rng.standard_normal(size=(500, 300))
    y = np.full(300, -3.0)                      # every realized value in the low tail
    block = RS.score_block(totals, y)
    assert block["pit_tail_lo"] > 0.9 and block["pit_tail_hi"] == 0.0
    assert block["n"] == 300 and block["n_draws"] == 500


def test_role_rows_carry_each_buckets_own_sigma_not_the_vector_label():
    """A graded arm's table is read a row at a time; a bucket row carrying the pooled label
    would make every bucket look like it shipped the same value."""
    bins, y, _ = _bucketed_truth(n_per_bin=5)
    rng = np.random.default_rng(2)
    totals = y[None, :] + rng.standard_normal(size=(50, len(y))) * 0.4
    graded = np.array([0.6, 0.45, 0.3, 0.15])

    rows = RS.role_rows(totals, y, bins, graded, "graded", "role_readout", "validation")
    frame = pd.DataFrame(rows).set_index("role")
    assert np.isnan(frame.loc["pooled", "sigma_scalar"])
    for label, sigma in zip(RS.ROLE_LABELS, graded):
        assert frame.loc[label, "sigma_scalar"] == pytest.approx(sigma)
        assert frame.loc[label, "sigma"] == "0.6|0.45|0.3|0.15"


# ── The search ────────────────────────────────────────────────────────────────

def test_the_coordinate_search_recovers_a_known_per_bucket_sigma():
    """The gradient is planted and has to come back. Buckets are searched one at a time
    against their own units, so a search that pooled the objective would flatten this."""
    bins, y, truth = _bucketed_truth()
    draw = _known_optimum_draw(bins)
    start = np.full(RS.N_ROLE_BINS, 0.375)

    found, rows, stable = RS.coordinate_search(draw, bins, y, start, "synthetic")
    assert stable, "a planted, uncoupled optimum must not move on the second pass"
    # Within one grid step of the truth in every bucket, and monotone in role.
    assert np.all(np.abs(found - truth) <= RS.GRID_STEP + 1e-9)
    assert np.all(np.diff(found) <= 0)
    assert {r["split"] for r in rows} == {"synthetic"}
    assert {r["pass"] for r in rows} == {1, 2}


def test_the_search_reuses_a_vector_it_has_already_drawn():
    """The grid re-visits the incumbent vector once per bucket per pass, and a draw is the
    expensive part — a cache miss there is a doubled runtime, not a wrong answer, which is
    exactly the kind of cost that goes unnoticed."""
    bins, y, truth = _bucketed_truth(n_per_bin=20)
    calls = []
    inner = _known_optimum_draw(bins, n_draws=80)

    def counting(sigma):
        calls.append(tuple(np.round(RS.broadcast_sigma(sigma), 6)))
        return inner(sigma)

    RS.coordinate_search(counting, bins, y, np.full(RS.N_ROLE_BINS, 0.375), "synthetic")
    assert len(calls) == len(set(calls)), "the same sigma vector was drawn twice"


def test_the_shared_profile_locates_each_buckets_marginal_optimum():
    """The coordinate search's starting point, and the honest first answer to the falsifier:
    if the per-bucket curves all bottom at the same rung, grading is not going to help."""
    bins, y, truth = _bucketed_truth()
    draw = _known_optimum_draw(bins)

    start, rows = RS.shared_profile(draw, bins, y, "synthetic")
    assert np.all(np.abs(start - truth) <= RS.GRID_STEP + 1e-9)
    # One pooled row plus one per bucket, at every rung of the grid.
    assert len(rows) == len(RS.GRADED_SIGMAS) * (RS.N_ROLE_BINS + 1)
    assert {r["unit"] for r in rows} == {"role_profile"}


def test_the_grid_brackets_the_calibration_target_at_both_ends():
    """A boundary optimum is a censored measurement reported as a result. The scratch
    sd_ratios imply ~0.69 for fringe and ~0.32 for star, so both have to be interior."""
    assert RS.GRADED_SIGMAS[0] < 0.32 and RS.GRADED_SIGMAS[-1] > 0.69
    assert 0.375 in RS.GRADED_SIGMAS, "the shipped scalar must be ON the grid"
    steps = np.diff(RS.GRADED_SIGMAS)
    np.testing.assert_allclose(steps, RS.GRID_STEP)


# ── The ship rule ─────────────────────────────────────────────────────────────

def test_a_bucket_ships_train_when_the_halves_agree_to_a_grid_step():
    """Selection reads the fitting half; validation confirms. So an agreeing bucket ships
    TRAIN's value, never validation's and never a compromise between them."""
    train = np.array([0.600, 0.450, 0.375, 0.300])
    val = np.array([0.675, 0.450, 0.300, 0.300])          # every gap is one step or none
    shipped, notes = RS.ship_rule(train, val, fallback=0.375)
    np.testing.assert_allclose(shipped, train)
    assert notes == ["train"] * RS.N_ROLE_BINS


def test_a_bucket_the_two_halves_disagree_about_keeps_the_shared_scalar():
    """The rule's whole point: four buckets is four chances to read a wiggle as a signal,
    so a bucket that does not replicate is left exactly as it ships today."""
    train = np.array([0.600, 0.450, 0.375, 0.300])
    val = np.array([0.225, 0.450, 0.375, 0.300])          # fringe is five steps away
    shipped, notes = RS.ship_rule(train, val, fallback=0.375)
    np.testing.assert_allclose(shipped, [0.375, 0.450, 0.375, 0.300])
    assert notes[0] == "fallback_disagree" and notes[1:] == ["train"] * 3


def test_the_fallback_is_the_shipped_scalar_rather_than_a_hard_coded_value():
    """`sim.minutes.player_season_sigma` moved once already (0.450 -> 0.375 on 2026-08-14);
    a rule that pinned the old value would quietly ship a sigma nothing selected."""
    train = np.array([0.15, 0.15, 0.15, 0.15])
    val = np.array([0.75, 0.75, 0.75, 0.75])
    shipped, notes = RS.ship_rule(train, val, fallback=0.45)
    np.testing.assert_allclose(shipped, 0.45)
    assert set(notes) == {"fallback_disagree"}
