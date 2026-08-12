"""Tests for the minutes head's fitting-window x dispersion ladder.

Nothing here needs a sampler: the ladder is a point MLE by construction, which is the same
argument `tests/test_availability_window.py` makes one head over.

Five things are pinned, each of which would move every figure in `minutes_window.csv`
without raising:

- **the breakpoint scan's observed statistic and its Monte-Carlo null go through one
  function.** The null is a 5,000-row stack and the observed value is a single series, so
  the vectorized path and the scalar path have to agree exactly — a faster copy of the scan
  calibrating the critical value is how a p-value ends up describing a slightly different
  statistic from the one it is compared against;
- **the role-graded dispersion nests the shared one**, so the ladder's two dispersion arms
  differ by a parameter rather than by an implementation;
- **the composition population is rescaled to the marginal head's unit.** Its native rate is
  a share of `5 x game_length`; compared un-rescaled against a share of `game_length` the
  two era series would sit a factor of five apart, which reads as an enormous era effect and
  is a unit error;
- **a window restricts the fitting half only**, so no arm's basis or coefficients can see a
  validation row;
- **the ladder's fixed variant is the shipped one**, or the table has no incumbent in it.
"""

import numpy as np
import pandas as pd
import pytest

from src.eda.season_effects import ROLE_LABELS
from src.models import minutes_window as MW
from src.models.availability import FEATURE_COLS


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n_per_season: int = 200, seasons=("2018-19", "2019-20", "2020-21"),
            mpg: float = 20.0, seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the point head, its buckets and `variants` read.

    The whole `FEATURE_COLS` block is present because `stan_minutes.variants` builds its
    expansions on top of it; the columns the tests do not exercise are filled with a
    constant rather than dropped, so the design matrix has the shape the real one does.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        share = np.clip(rng.beta(5, 5, n_per_season), 0.05, 0.95)
        trials = np.full(n_per_season, 2000)
        frame = pd.DataFrame({
            "season": season,
            "player_id": np.arange(n_per_season),
            "trials": trials,
            "successes": rng.binomial(trials, share),
            "logit_share_lag1": np.log(share / (1 - share)),
        })
        for column in FEATURE_COLS:
            frame[column] = rng.normal(0.0, 1.0, n_per_season)
        frame["minutes_per_game_lag1"] = mpg
        frame["age"] = rng.uniform(21, 34, n_per_season)
        # The two columns `FloorMinutes.carry_forward` reads. The ladder always scores the
        # no-fit floor beside the fitted arms, so a frame that cannot build the floor cannot
        # exercise the ladder at all.
        frame["minutes_share_lag1"] = share
        frame["length_played"] = trials.astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


# ── The breakpoint scan ───────────────────────────────────────────────────────

def test_sup_f_finds_a_planted_break_and_ignores_a_flat_series():
    rng = np.random.default_rng(11)
    noise = rng.normal(0.0, 0.01, 20)
    flat = 0.5 + noise
    broken = np.concatenate([np.full(10, 0.5), np.full(10, 0.2)]) + noise
    f_flat, _ = MW.sup_f(flat)
    f_broken, idx = MW.sup_f(broken)
    assert f_broken > 100 * max(f_flat, 1e-9)
    assert idx == 10


def test_a_series_with_no_variance_at_all_cannot_break():
    # Both halves constant means zero residual variance and an undefined F. Returning 0
    # rather than an inf keeps a degenerate series out of the Monte-Carlo null's tail.
    assert MW.sup_f(np.full(20, 0.5)) == (0.0, 4)


def test_the_scalar_and_stacked_paths_of_the_scan_agree_exactly():
    # The load-bearing one. `break_scan` takes the observed statistic through the 1-D path
    # and its 5,000-replicate null through the 2-D path, so a divergence between them would
    # calibrate the p-value against a different statistic than the one being tested — and it
    # would do it silently, since both return plausible numbers.
    rng = np.random.default_rng(7)
    stack = rng.normal(0.5, 0.1, size=(25, 22))
    stacked_f, stacked_idx = MW.sup_f(stack)
    for i, row in enumerate(stack):
        f, idx = MW.sup_f(row)
        assert np.isclose(f, stacked_f[i])
        assert idx == stacked_idx[i]


def test_the_scan_respects_the_minimum_side_and_never_breaks_at_the_ends():
    values = np.concatenate([[10.0], np.full(19, 0.5)])   # one outlier at the very start
    _, idx = MW.sup_f(values, min_side=4)
    assert 4 <= idx <= len(values) - 4


def test_break_scan_recovers_a_planted_break_against_its_own_null():
    seasons = [f"{1996 + i}-{str(1997 + i)[-2:]}" for i in range(24)]
    noise = np.random.default_rng(3).normal(0.0, 0.005, 24)
    series = pd.DataFrame({
        "population": "synthetic", "season": seasons, "n": 300,
        "mean_rate": np.array([0.5] * 12 + [0.4] * 12) + noise,
        "sd_rate": np.full(24, 0.2),
        "p_workhorse": np.full(24, 0.05),
    })
    out = MW.break_scan(series, reps=200).set_index("statistic")
    assert out.loc["mean_rate", "break_season"] == seasons[12]
    assert out.loc["mean_rate", "p_value"] < 0.05
    assert out.loc["mean_rate", "shift"] == pytest.approx(-0.1, abs=0.01)
    # A series with no variance at all cannot break, and must not claim to.
    assert out.loc["sd_rate", "sup_f"] == 0.0


# ── The dispersion axis ───────────────────────────────────────────────────────

def test_role_grading_nests_the_shared_dispersion_when_one_bucket_holds_every_row():
    # Every row sits in `12-24`, so the graded head fits exactly one bucket on exactly the
    # rows the shared head fits its scalar on. The two dispersions are then the same
    # `fit_dispersion` call and must agree — that is what makes the ladder's two arms differ
    # by a parameter rather than by an implementation.
    train = _design(mpg=20.0)
    features = ["logit_share_lag1", "age"]
    shared = MW.PointMinutes(features).fit(train)
    graded = MW.RoleGradedPointMinutes(features).fit(train)

    assert list(graded.rho_by_role) == ["12-24"]
    assert np.isclose(graded.rho_by_role["12-24"], shared.rho, atol=1e-6)
    assert graded.rho_spread == pytest.approx(1.0)
    assert np.allclose(graded.rho_for(train), shared.rho_for(train), atol=1e-6)


def test_a_thin_role_bucket_falls_back_to_the_pooled_dispersion():
    # A dispersion fitted on a handful of rows is noise wearing the shape of a parameter,
    # so the bucket must fall back rather than ship a fitted value from 5 rows.
    bulk = _design(n_per_season=200, mpg=20.0, seed=1)
    thin = _design(n_per_season=2, mpg=35.0, seed=2)
    train = pd.concat([bulk, thin], ignore_index=True)
    graded = MW.RoleGradedPointMinutes(["logit_share_lag1", "age"]).fit(train)

    assert "30+ mpg" not in graded.rho_by_role
    assert np.isclose(graded.rho_for(thin)[0], graded.pooled_rho)


def test_the_graded_head_uses_each_rows_own_bucket_dispersion():
    train = pd.concat([_design(mpg=20.0, seed=3), _design(mpg=35.0, seed=4)],
                      ignore_index=True)
    graded = MW.RoleGradedPointMinutes(["logit_share_lag1", "age"]).fit(train)
    assert set(graded.rho_by_role) == {"12-24", "30+ mpg"}

    rho = graded.rho_for(train)
    mid = train["minutes_per_game_lag1"].to_numpy() == 20.0
    assert np.allclose(rho[mid], graded.rho_by_role["12-24"])
    assert np.allclose(rho[~mid], graded.rho_by_role["30+ mpg"])


def test_every_role_label_gets_a_column_even_when_it_was_not_fitted():
    # The artifact's shape must not depend on which buckets happened to clear the row
    # minimum, or two runs of the ladder produce differently-shaped tables.
    val = _design(n_per_season=50)
    samples = np.tile(val["successes"].to_numpy(float), (20, 1))
    row, _ = MW.score_arm("x", samples, val, val, 2, 0.05, 1.0, {"12-24": 0.05})
    assert [f"rho_{label}" in row for label in ROLE_LABELS] == [True] * len(ROLE_LABELS)
    assert np.isnan(row["rho_30+ mpg"])


# ── The era series ────────────────────────────────────────────────────────────

def test_the_composition_population_is_rescaled_to_the_marginal_heads_unit():
    # A player who plays every minute of every game has a rate of 1.0, not 0.2. The
    # composition's native denominator is the team-game pot `5 x game_length`, and comparing
    # that against the marginal head's share of `game_length` puts the two populations a
    # factor of five apart — an enormous apparent era effect that is a unit error.
    frame = pd.DataFrame({"player_id": [1, 1], "season": ["2019-20"] * 2,
                          "y": [48.0, 48.0], "game_length": [48.0, 48.0]})
    out = MW.composition_population(frame)
    assert out["rate"].iloc[0] == pytest.approx(1.0)

    # And a player taking a fifth of the pot — one of five on the floor throughout — reads
    # as 1.0 on the same scale, because that IS every minute.
    fifth = pd.DataFrame({"player_id": [2], "season": ["2019-20"],
                          "y": [48.0], "game_length": [48.0]})
    assert MW.composition_population(fifth)["rate"].iloc[0] == pytest.approx(1.0)


def test_the_era_series_reports_one_cell_per_population_and_season():
    a = pd.DataFrame({"season": ["2018-19"] * 5 + ["2019-20"] * 5,
                      "rate": np.linspace(0.1, 0.9, 10)})
    out = MW.era_series({"one": a, "two": a}, last_season="2019-20")
    assert len(out) == 4
    assert set(out["population"]) == {"one", "two"}
    assert (out["n"] == 5).all()


def test_the_era_series_stops_at_the_last_descriptive_season():
    # 2024-25 and 2025-26 are the held-out split. A descriptive series over them is still a
    # look at the test data, so the cut is enforced rather than remembered.
    frame = pd.DataFrame({"season": ["2023-24", "2023-24", "2024-25", "2024-25"],
                          "rate": [0.4, 0.6, 0.4, 0.6]})
    out = MW.era_series({"p": frame})
    assert list(out["season"]) == ["2023-24"]


def test_the_workhorse_share_counts_at_or_above_the_threshold():
    frame = pd.DataFrame({"season": ["2019-20"] * 4,
                          "rate": [0.74, 0.75, 0.76, 0.10]})
    out = MW.era_series({"p": frame})
    assert out["p_workhorse"].iloc[0] == pytest.approx(0.5)


def test_era_blocks_report_the_endpoint_fold_the_doc_quotes():
    seasons = ["2018-19", "2019-20", "2020-21"]
    series = pd.DataFrame({"population": "p", "season": seasons, "n": [100, 100, 100],
                           "mean_rate": [0.5, 0.5, 0.5],
                           "sd_rate": [0.2, 0.15, 0.1],
                           "sd_logit": [1.0, 0.9, 0.8],
                           "p_workhorse": [0.10, 0.05, 0.01]})
    ends = MW.era_blocks(series, blocks=()).set_index("block").loc["endpoints"]
    assert ends["sd_rate_change"] == pytest.approx(-0.5)
    assert ends["p_workhorse_fold"] == pytest.approx(10.0)
    assert ends["first_season"] == "2018-19" and ends["last_season"] == "2020-21"


def test_a_block_average_weights_seasons_by_their_own_row_count():
    series = pd.DataFrame({"population": "p", "season": ["2018-19", "2019-20"],
                           "n": [300, 100], "mean_rate": [0.5, 0.5],
                           "sd_rate": [0.20, 0.10], "sd_logit": [1.0, 1.0],
                           "p_workhorse": [0.10, 0.02]})
    block = (MW.era_blocks(series, blocks=(("all", "1996-97", "2023-24"),))
             .set_index("block").loc["all"])
    assert block["sd_rate"] == pytest.approx((300 * 0.20 + 100 * 0.10) / 400)
    assert block["n"] == 400


# ── Windows, arms and coverage ────────────────────────────────────────────────

def test_a_window_restricts_the_fitting_half_and_leaves_validation_whole():
    train = _design(seasons=("2010-11", "2015-16", "2020-21"), n_per_season=150)
    val = _design(seasons=("2022-23",), n_per_season=90)
    arm = MW.fit_arm(train, val, "post_2014", "shared", n_knots=3)
    assert len(arm.frame) == len(val)
    assert sorted(arm.frame["season"].unique()) == ["2022-23"]
    # `post_2014` keeps 2015-16 and 2020-21 and drops 2010-11 — 300 rows, not 450.
    assert len(MW.restrict_window(train, MW.WINDOWS["post_2014"])) == 300


def test_each_window_carries_its_own_basis_so_arms_cannot_be_crossed():
    # The knots are fitted on the window's own rows, so two windows hand back differently
    # transformed validation frames. `FittedArm` binds each model to its own, which is what
    # stops one window's coefficients being applied to another window's basis.
    train = _design(seasons=("2010-11", "2015-16", "2020-21"), n_per_season=150)
    val = _design(seasons=("2022-23",), n_per_season=90)
    full = MW.fit_arm(train, val, "full", "shared", n_knots=3)
    short = MW.fit_arm(train, val, "post_2014", "shared", n_knots=3)

    spline_cols = [c for c in full.frame.columns if "__s" in c]
    assert spline_cols and spline_cols == [c for c in short.frame.columns if "__s" in c]
    assert not np.allclose(full.frame[spline_cols].to_numpy(),
                           short.frame[spline_cols].to_numpy())
    assert full.model.features == short.model.features


def test_interval_coverage_is_one_when_every_draw_lands_on_the_observation():
    y = np.array([100.0, 200.0, 300.0])
    samples = np.tile(y, (50, 1))
    for level in MW.COVERAGE_LEVELS:
        assert MW.interval_coverage(samples, y, level) == pytest.approx(1.0)


def test_interval_coverage_is_zero_when_the_predictive_misses_entirely():
    y = np.array([0.0, 0.0])
    samples = np.full((50, 2), 1000.0)
    assert MW.interval_coverage(samples, y, 0.95) == pytest.approx(0.0)


def test_the_reference_arm_is_not_recorded_as_beating_itself():
    # Its interval against itself is degenerately [0, 0], which `verdict` reads as a win.
    # A table claiming the incumbent beat the incumbent is how a ladder loses its baseline.
    train = _design(seasons=("2010-11", "2015-16", "2020-21"), n_per_season=150)
    val = _design(seasons=("2022-23",), n_per_season=90)
    table, arms = MW.ladder(train, val, n_knots=3)

    reference = table[table["arm"] == MW.REFERENCE_ARM].iloc[0]
    assert reference["verdict"] == "reference"
    assert reference["crps_vs_incumbent"] == pytest.approx(0.0)
    assert not bool(reference["beats_incumbent"])
    assert set(arms) == {f"{w}__{m}" for w in MW.WINDOWS for m in MW.RHO_MODES}


def test_the_ladder_scores_the_no_fit_floor_alongside_the_fitted_arms():
    # Every head in this project is quoted against a mandatory no-fit floor, so an arm
    # table without one is not a result. Its own row must also count as clearing it.
    train = _design(seasons=("2010-11", "2015-16", "2020-21"), n_per_season=150)
    val = _design(seasons=("2022-23",), n_per_season=90)
    table, _ = MW.ladder(train, val, n_knots=3)
    assert "carry_forward" in set(table["arm"])
    assert bool(table.loc[table["arm"] == "carry_forward", "beats_floor"].iloc[0])


# ── The guard ─────────────────────────────────────────────────────────────────

def test_the_ladder_refuses_a_variant_the_head_does_not_ship(tmp_path):
    pd.DataFrame({"variant": ["linear", "logit_own_spline"],
                  "selected": [True, False]}).to_csv(
        tmp_path / "stan_minutes_metrics.csv", index=False)
    with pytest.raises(ValueError, match="logit_own_spline"):
        MW.assert_shipped_variant(tmp_path)


def test_the_guard_passes_when_the_variants_agree(tmp_path):
    pd.DataFrame({"variant": ["linear", "logit_own_spline"],
                  "selected": [False, True]}).to_csv(
        tmp_path / "stan_minutes_metrics.csv", index=False)
    assert "matches" in MW.assert_shipped_variant(tmp_path)


def test_a_missing_metrics_artifact_is_unverified_rather_than_fatal(tmp_path):
    # A fresh checkout without `make stan-minutes` still runs the ladder; it just cannot
    # confirm the incumbent, and says so.
    assert "unverified" in MW.assert_shipped_variant(tmp_path)
