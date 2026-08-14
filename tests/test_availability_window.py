"""Tests for the availability window x trend x dispersion x likelihood ladder.

The three things pinned first are the three that would move every figure in
`availability_window.csv` without raising: which rows a window keeps, where the upper tail
is read, and whether a role-graded dispersion still nests the shared one.

The fourth axis adds a fourth: **every likelihood arm reproduces the head it extends at its
nesting parameter values.** That is the discipline `n_rho = 1` and `U_n = 0` already carry
in the .stan sources, and it is the rollback path — an arm that cannot reproduce the
incumbent is a different model rather than an extension of it, and its margin is measuring
something other than the frailty.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.availability import predictive_pmf
from src.models.availability_window import (LIKELIHOODS, BetaBinomFrailty,
                                            BetaRectangularFrailty, FiniteMixtureFrailty,
                                            LogitNormalFrailty, MixtureFrailty,
                                            RoleGradedBetaBinomial, assert_nests,
                                            interval_coverage, paired_bootstrap,
                                            restrict_window, tail_coverage,
                                            tenure_decomposition_pmf)


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


# ── the fourth axis: the likelihood ───────────────────────────────────────────

def _frailty_frame(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """A synthetic availability design wide enough to fit, with all four role buckets filled.

    Plain builders and no fixtures, mirroring `tests/test_preprocess.py`. The columns are
    the ones `FEATURE_COLS` and `PI_COLS` actually read; nothing here has to be realistic,
    only well-conditioned enough that the alternating fit converges.

    **The counts are OVERDISPERSED, and that is load bearing.** Drawing them from a plain
    binomial leaves nothing for `rho` to fit, so every arm here lands on the `RHO_MIN = 1e-6`
    guard rail — where the shape parameters reach ~1e6 and `scipy.stats.betabinom.pmf`
    itself carries ~5e-9 of relative error, because it evaluates two log-betas of magnitude
    1e6 that cancel to a number of order one. The nesting assertions below are equalities
    against that function, so at the guard rail they would be measuring the *yardstick*.
    A frailty draw of the same shape the arms assume puts `rho` at ~0.05 instead, which is
    also what the head actually fits.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.availability_window import PI_COLS

    rng = np.random.default_rng(seed)
    mpg = rng.uniform(2.0, 38.0, n_rows)
    frame = pd.DataFrame({col: rng.normal(size=n_rows) for col in
                          sorted(set(FEATURE_COLS) | set(PI_COLS))})
    frame["minutes_per_game_lag1"] = mpg
    frame["team_games"] = 82
    frame["season"] = np.where(np.arange(n_rows) % 2 == 0, "2018-19", "2019-20")
    mu = np.clip(0.45 + 0.012 * mpg, 0.02, 0.98)
    scale = (1.0 - 0.05) / 0.05
    frame["gp"] = rng.binomial(82, rng.beta(mu * scale, (1.0 - mu) * scale))
    return frame


@pytest.mark.parametrize("name", sorted(LIKELIHOODS))
def test_every_likelihood_arm_reproduces_its_nesting_point(name):
    # Rule 2 of the axis. `pi = 0`, `g = 0`, `theta = 0` and `sigma_u = 0` are each a
    # FINITE, attainable parameter value in these arms rather than a limit, so this is an
    # equality and not a tolerance negotiation — which is exactly why they were
    # parameterized that way.
    train = _frailty_frame()
    model = LIKELIHOODS[name](l2=1.0).fit(train)
    assert assert_nests(model, train) < 1e-8


def test_assert_nests_raises_when_an_arm_does_not_reproduce_the_incumbent():
    # The guard has to fail loudly, or it is decoration. A perturbed nesting point is a
    # different model, and the ladder's whole contrast rests on this not passing silently.
    train = _frailty_frame()
    model = BetaRectangularFrailty(l2=1.0).fit(train)
    model._extra0 = lambda: np.array([0.3])
    with pytest.raises(AssertionError, match="does not reproduce its nesting point"):
        assert_nests(model, train)


def test_finite_mixture_at_one_class_is_the_beta_binomial_arm():
    # K = 1 is the incumbent, and here it is checked on the PREDICTIVE rather than on the
    # log-likelihood: a mixture that nests in likelihood but not in pmf would score
    # identically on train and differently on validation, which is the silent version.
    train = _frailty_frame()
    single = FiniteMixtureFrailty(l2=1.0, k=1).fit(train)
    plain = BetaBinomFrailty(l2=1.0).fit(train)
    assert np.allclose(single.predict_pmf(train, 82), plain.predict_pmf(train, 82))
    assert np.isclose(single.train_loglik, plain.train_loglik)


def test_beta_rectangular_pmf_at_theta_zero_is_the_beta_binomial_pmf():
    model = BetaRectangularFrailty(l2=1.0).fit(_frailty_frame(200))
    df = _frailty_frame(50, seed=1)
    n = df["team_games"].to_numpy(dtype=float)
    eta = model._design(df) @ model.beta
    disp = model._disp_row(df, model.dispersion)
    k = np.arange(83)
    at_zero = model._pmf(n, eta, disp, np.zeros(1), k, df)
    from src.models.availability_window import _bb_pmf_grid, _sigmoid
    assert np.allclose(at_zero, _bb_pmf_grid(n, _sigmoid(eta), disp, k))


def test_beta_rectangular_uniform_component_is_the_discrete_uniform():
    # theta = 1 puts the whole frailty on U(0, 1) = Beta(1, 1), and a binomial mixed over
    # Beta(1, 1) is the discrete uniform on 0..n. That identity is why this arm costs one
    # parameter instead of a quadrature rule.
    model = BetaRectangularFrailty(l2=1.0).fit(_frailty_frame(200))
    df = _frailty_frame(20, seed=2)
    n = df["team_games"].to_numpy(dtype=float)
    k = np.arange(83)
    pmf = model._pmf(n, model._design(df) @ model.beta,
                     model._disp_row(df, model.dispersion), np.ones(1), k, df)
    assert np.allclose(pmf[:, :83], predictive_pmf(n, np.full(len(n), 0.5),
                                                   1.0 / 3.0, 82), atol=1e-12)
    assert np.allclose(pmf.sum(axis=1), 1.0)


def test_logit_normal_at_zero_sigma_is_the_binomial():
    # This arm does NOT nest the incumbent, by design — it is in the ladder because it
    # should fail the other way. What it does nest is the binomial, and that is what pins it.
    from scipy.stats import binom

    from src.models.availability_window import _sigmoid
    train = _frailty_frame(300)
    model = LogitNormalFrailty(l2=1.0).fit(train)
    eta = model._design(train) @ model.incumbent_beta
    ll, _ = model._terms(train["gp"].to_numpy(dtype=float),
                         train["team_games"].to_numpy(dtype=float), eta,
                         np.zeros(len(train)), model._extra0(), train)
    want = binom.logpmf(train["gp"].to_numpy(dtype=float),
                        train["team_games"].to_numpy(dtype=float), _sigmoid(eta))
    assert np.allclose(ll, want)


def test_logit_normal_frailty_cannot_diverge_at_a_boundary():
    model = LogitNormalFrailty(l2=1.0).fit(_frailty_frame(200))
    report = model.shape_report(_frailty_frame(50, seed=3))
    assert report["diverges_at_one"] == 0.0
    assert report["diverges_at_zero"] == 0.0


@pytest.mark.parametrize("name", sorted(LIKELIHOODS))
def test_every_arm_emits_a_proper_predictive(name):
    train = _frailty_frame()
    model = LIKELIHOODS[name](l2=1.0).fit(train)
    pmf = model.predict_pmf(train, 82)
    assert pmf.shape == (len(train), 83)
    assert (pmf >= 0).all()
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-9)
    # Mass above a row's own schedule is impossible, and a mixture is the easy place to
    # leak it — the uniform component's support is 0..n, not 0..max_games.
    assert np.allclose(pmf[:, 83:], 0.0)


@pytest.mark.parametrize("name", sorted(LIKELIHOODS))
def test_no_arm_fits_worse_than_the_point_it_starts_from(name):
    # Every arm starts at the incumbent, so its FITTING log-likelihood can only improve.
    # An arm below its own start means the alternating loop moved backwards, and the
    # validation column would then be measuring the optimizer rather than the likelihood.
    train = _frailty_frame()
    model = LIKELIHOODS[name](l2=1.0).fit(train)
    assert model.train_loglik >= model.nesting_loglik - 1e-6


def test_mixture_starts_are_not_all_the_collapsed_point():
    # The bug this catches is measured and silent: started at `g = 0` a three-class mixture
    # sat on its bound, reported success and reproduced the incumbent to four decimals.
    for arm in (MixtureFrailty(l2=1.0), FiniteMixtureFrailty(l2=1.0, k=3),
                BetaRectangularFrailty(l2=1.0)):
        starts = arm._fit_starts()
        assert len(starts) > 1
        assert any(np.any(np.asarray(s) != np.asarray(arm._extra0())) for s in starts)


def test_finite_mixture_class_means_are_ordered():
    # Ordering is what makes the classes identified; without it the arm label-switches
    # between starts and its fitted offsets mean nothing.
    model = FiniteMixtureFrailty(l2=1.0, k=3).fit(_frailty_frame(400))
    offsets, weights = model._parts(model.extra)
    assert np.all(np.diff(offsets) >= 0)
    assert np.isclose(weights.sum(), 1.0)


def test_tenure_decomposition_arm_is_skipped_when_its_artifact_is_absent(tmp_path):
    # The ladder must not require `make stan-games-played` to have been run.
    val = pd.DataFrame({"season": ["2022-23"], "player_id": [1], "team_games": [82],
                        "gp": [70]})
    assert tenure_decomposition_pmf(val, 82, tmp_path / "missing.csv") is None


def test_tenure_decomposition_arm_aligns_on_season_and_player(tmp_path):
    # The free arm is a join, and a join is where this silently goes wrong: a pmf lined up
    # on the wrong rows still sums to 1 and still scores.
    path = tmp_path / "gp_pmf.csv"
    rows = []
    for (season, player), peak in [(("2022-23", 7), 70), (("2023-24", 9), 20)]:
        for k in range(83):
            rows.append({"arm": "duration_covariates", "season": season,
                         "player_id": player, "team_games": 82, "gp": k,
                         "p": 1.0 if k == peak else 0.0})
    pd.DataFrame(rows).to_csv(path, index=False)

    val = pd.DataFrame({"season": ["2023-24", "2022-23"], "player_id": [9, 7],
                        "team_games": [82, 82], "gp": [20, 70]})
    pmf = tenure_decomposition_pmf(val, 82, path)
    assert pmf.argmax(axis=1).tolist() == [20, 70]


def test_tenure_decomposition_arm_is_skipped_on_partial_coverage(tmp_path):
    # Scoring 400 of 883 rows against arms scored on all 883 is not a comparison, and
    # reindex would hand back NaN rows rather than raising.
    path = tmp_path / "gp_pmf.csv"
    pd.DataFrame([{"arm": "duration_covariates", "season": "2022-23", "player_id": 7,
                   "team_games": 82, "gp": k, "p": 1.0 if k == 70 else 0.0}
                  for k in range(83)]).to_csv(path, index=False)
    val = pd.DataFrame({"season": ["2022-23", "2022-23"], "player_id": [7, 8],
                        "team_games": [82, 82], "gp": [70, 40]})
    assert tenure_decomposition_pmf(val, 82, path) is None


def test_bootstrap_tail_errors_brackets_a_known_boundary_error():
    # The selector is a non-linear statistic — two absolute values of differences of means —
    # so it is resampled and recomputed rather than averaged over a per-row score. A model
    # certain everyone plays a full schedule, on rows where half did not, has a
    # boundary_tail_error of exactly 0.5 with no row-to-row variation in it.
    from src.models.availability_window import _tail_parts, bootstrap_tail_errors

    pmf = np.zeros((200, 83))
    pmf[:, 82] = 1.0
    y = np.where(np.arange(200) % 2 == 0, 5, 82)
    n = np.full(200, 82)
    out = bootstrap_tail_errors(_tail_parts(pmf, y, n), None, reps=200)
    assert out["boundary_tail_error_lo"] <= 0.5 <= out["boundary_tail_error_hi"]


def test_bootstrap_tail_errors_is_paired_against_the_reference():
    # Two arms whose boundary errors differ by a constant on every resample: the paired
    # interval has to collapse onto that constant, the way `paired_bootstrap` does for CRPS.
    from src.models.availability_window import _tail_parts, bootstrap_tail_errors

    n = np.full(100, 82)
    y = np.where(np.arange(100) % 4 == 0, 5, 82)
    perfect = np.zeros((100, 83))
    perfect[np.arange(100), y] = 1.0
    wrong = np.zeros((100, 83))
    wrong[:, 82] = 1.0

    out = bootstrap_tail_errors(_tail_parts(perfect, y, n), _tail_parts(wrong, y, n),
                               reps=200)
    assert out["boundary_vs_betabinom"] < 0.0
    assert out["beats_betabinom_boundary"] is True


# ── the shoulders ─────────────────────────────────────────────────────────────

def test_upper_shoulder_is_counted_in_games_missed_not_games_played():
    # The point of the missed scale: "played nearly everything" has to be the SAME event in
    # a 66-game season and an 82-game one. A fixed `gp > 70` would score the 66-game row
    # against an unreachable threshold and report 0 for a player who missed nothing.
    pmf = np.zeros((2, 86))
    pmf[0, 66] = 1.0                       # played all 66 of a shortened season
    pmf[1, 78] = 1.0                       # missed 4 of 82
    y = np.array([66, 78])
    n = np.array([66, 82])
    rows = {r["tail"]: r for r in tail_coverage(pmf, y, n)}
    assert np.isclose(rows["missed_le_11"]["predicted"], 1.0)
    assert np.isclose(rows["missed_le_11"]["observed"], 1.0)
    # ...and the exclusive band separates the iron man from the near-miss.
    assert np.isclose(rows["band_high_shoulder"]["predicted"], 0.5)
    assert np.isclose(rows["band_full"]["predicted"], 0.5)


def test_bands_partition_the_low_tail_so_opposite_errors_cannot_cancel():
    # The defect this exists for. A head that puts too MUCH mass at exactly zero and too
    # LITTLE at 1-9 has a `below_10` error of zero — which is what `docs/availability-
    # window-plan.md` section 1 measured on the real head, in those two directions.
    pmf = np.zeros((2, 86))
    pmf[:, 0] = 0.5                        # predicts half the rows never play
    pmf[:, 5] = 0.5
    y = np.array([5, 5])                   # both actually played 5
    n = np.array([82, 82])
    rows = {r["tail"]: r for r in tail_coverage(pmf, y, n)}
    assert np.isclose(rows["below_10"]["error"], 0.0)          # cancels, and lies
    assert np.isclose(rows["band_zero"]["error"], +0.5)        # too much mass at zero
    assert np.isclose(rows["band_low_shoulder"]["error"], -0.5)  # too little at 1-9


def test_shoulder_shape_sees_a_misshapen_tail_a_band_total_cannot():
    # A band is one number per region: mass moved WITHIN the region nets to zero. The
    # localized sup distance is what "the shape in the shoulder" actually asks for.
    from src.models.availability_window import shoulder_shape

    n = np.full(200, 82)
    y = np.where(np.arange(200) < 20, 8, 50)      # 10% of rows play exactly 8 games
    pmf = np.zeros((200, 86))
    pmf[:, 1] = 0.1                               # same 10% mass, at 1 game instead of 8
    pmf[:, 50] = 0.9
    out = shoulder_shape(pmf, y, n)
    rows = {r["tail"]: r for r in tail_coverage(pmf, y, n)}
    assert np.isclose(rows["band_low_shoulder"]["error"], 0.0, atol=1e-12)
    assert out["low_shape_ks"] > 0.09             # the shape distance still sees it


def test_shoulder_shape_is_zero_for_a_perfect_predictive():
    from src.models.availability_window import shoulder_shape

    n = np.full(50, 82)
    y = np.arange(50) % 82
    pmf = np.zeros((50, 86))
    pmf[np.arange(50), y] = 1.0
    out = shoulder_shape(pmf, y, n)
    assert out["low_shape_ks"] < 1e-12
    assert out["high_shape_ks"] < 1e-12


def test_mean_abs_tail_error_still_averages_only_its_original_four_thresholds():
    # Adding shoulder rows to `tail_coverage` must not silently redefine a column that is
    # already in `availability_window.csv` and already quoted.
    from src.models.availability_window import score_arm

    n = np.full(60, 82)
    y = np.where(np.arange(60) % 3 == 0, 5, 70)
    pmf = np.zeros((60, 86))
    pmf[np.arange(60), y] = 1.0
    df = pd.DataFrame({"gp": y, "team_games": n,
                       "minutes_per_game_lag1": np.full(60, 20.0),
                       "season": "2022-23"})

    class _Fixed:
        rho = 0.2
        rho_spread = 1.0

        def predict_pmf(self, frame, mg):
            return pmf

        def predict_mean(self, frame):
            return y / n

    row, _ = score_arm("perfect", _Fixed(), df, df, [], 85, 42)
    # A perfect predictive: every threshold error is zero, so every summary is zero.
    assert np.isclose(row["mean_abs_tail_error"], 0.0)
    assert np.isclose(row["boundary_tail_error"], 0.0)
    assert np.isclose(row["shoulder_error"], 0.0)
    assert np.isclose(row["point_mass_error"], 0.0)
    assert np.isclose(row["low_shape_ks"], 0.0)


def test_origin_scores_supplies_every_piece_both_rolling_tables_need():
    # The two rolling tables assemble their rows separately but pool the same per-row
    # dict. Pinning that dict is what stops one of them silently losing a column.
    from src.models.availability_window import TAIL_MISSED, _origin_scores

    train = _frailty_frame(400)
    model = BetaBinomFrailty(l2=1.0).fit(train)
    scored = _origin_scores(model, train, 85)
    for key in ["p_zero", "p_band_low_shoulder", "p_band_high_shoulder",
                *[f"p_missed_le_{m}" for m in TAIL_MISSED]]:
        assert key in scored, f"_origin_scores dropped {key}"
        assert len(scored[key]) == len(train)
        assert np.all((scored[key] >= -1e-9) & (scored[key] <= 1 + 1e-9))


def test_both_rolling_tables_carry_the_shoulder_columns():
    # Measured failure, not a hypothetical: the shoulder columns were added to
    # `rolling_confirmation` and not to `likelihood_rolling`, and the artifact still wrote
    # and still sorted — it simply had no shoulder in it. Run BOTH, shrunk to one arm and
    # the fewest origins each will accept, and read the frames they actually return.
    from src.models import availability_window as mod

    rows = []
    for year in range(2000, 2012):
        frame = _frailty_frame(120, seed=year)
        frame["season"] = f"{year}-{str(year + 1)[-2:]}"
        rows.append(frame)
    train = pd.concat(rows, ignore_index=True)

    saved = (mod.LOOKBACKS, mod.RHO_MODES, mod.FIRST_ORIGIN, mod.LIKELIHOODS)
    try:
        mod.LOOKBACKS = (None,)
        mod.RHO_MODES = ("shared",)
        mod.FIRST_ORIGIN = 2010
        mod.LIKELIHOODS = {"betabinom": BetaBinomFrailty}
        tables = [mod.rolling_confirmation(train, 85), mod.likelihood_rolling(train, 85)]
    finally:
        mod.LOOKBACKS, mod.RHO_MODES, mod.FIRST_ORIGIN, mod.LIKELIHOODS = saved

    for table in tables:
        for column in ("shoulder_error", "err_band_zero", "err_band_low_shoulder",
                       "err_band_high_shoulder", "boundary_tail_error", "body_error"):
            assert column in table.columns
        assert (table["shoulder_error"] >= 0).all()


# ── The `l2` confound (§7d) ───────────────────────────────────────────────────

def test_l2_grid_is_anchored_at_the_unpenalized_mle():
    """The zero end is what makes the sweep a bound rather than a grid search.

    If the reference's optimum sat on the grid's low edge, "sweep it further down" would
    still be an open move and the surviving margin would not be a bound on anything. Zero
    is the unpenalized MLE, so nothing lies below it.
    """
    from src.models.availability_window import L2_GRID

    assert min(L2_GRID) == 0.0
    assert list(L2_GRID) == sorted(L2_GRID)
    assert len(set(L2_GRID)) == len(L2_GRID)
    # The pinned value the ladder scored every arm at has to be *on* the grid, or the
    # sweep cannot report a margin against it.
    assert 1.0 in L2_GRID


def test_l2_confound_reports_each_arm_against_the_reference_at_its_own_best():
    """The verdict's three margins are the three readings §7d distinguishes.

    A sweep that only reported the pinned margin would answer a different question than the
    one asked: what has to be shown is the challenger against a reference regularized *as
    favourably as the grid allows*, which is the reading that can only shrink the margin.
    """
    from src.models import availability_window as mod

    train = _frailty_frame(400, seed=3)
    val = _frailty_frame(200, seed=4)
    grid = (0.0, 1.0, 256.0)
    sweep, verdict = mod.l2_confound(train, val, 85, grid=grid,
                                     arms=("betabinom", "beta_rect"), seed=0)

    assert len(sweep) == len(grid) * 2
    assert set(sweep["l2"]) == set(grid)
    assert sweep["is_pinned"].sum() == 2
    # One verdict row per challenger — the reference is the yardstick, not an arm.
    assert list(verdict["arm"]) == ["beta_rect"]

    row = verdict.iloc[0]
    ref = sweep[sweep["likelihood"] == "betabinom"]
    # The reference's chosen `l2` is its best on the grid, and "best" is lowest CRPS.
    assert row["reference_l2_best"] == ref.sort_values("val_crps").iloc[0]["l2"]
    assert row["reference_crps_best"] <= row["reference_crps_pinned"] + 1e-12
    assert row["arm_crps_best"] <= row["arm_crps_pinned"] + 1e-12
    # A more favourably regularized reference can only move the margin toward zero, which
    # is what makes the surviving share a lower bound rather than a point estimate.
    assert row["margin_vs_reference_best"] >= row["margin_pinned"] - 1e-12
    for name in ("margin_pinned", "margin_vs_reference_best", "margin_matched"):
        assert row[f"{name}_lo"] <= row[name] <= row[f"{name}_hi"]


def test_l2_confound_counts_the_parameters_the_penalty_never_reaches():
    """The confound is the unpenalized block, so the table has to carry its size.

    `l2` multiplies `beta[1:]` and nothing else, so an arm's advantage under a pinned
    penalty is exactly the parameters it adds outside that block. Stating it as a column
    is what turns "the arms are not equally advantaged" into something checkable.
    """
    from src.models.availability_window import L2_UNPENALIZED, LIKELIHOODS

    assert L2_UNPENALIZED["betabinom"] == 0
    # Every arm the confound is measured on is a real arm of the ladder.
    assert set(L2_UNPENALIZED) <= set(LIKELIHOODS)
    # And the ordering is the one §7d states: mixture carries the most, the control one.
    assert (L2_UNPENALIZED["mixture"] > L2_UNPENALIZED["finite_mix"]
            > L2_UNPENALIZED["beta_rect"] > L2_UNPENALIZED["betabinom"])
    # And the mixture's count is `theta`, `mu_low`, `rho_low` and one `gamma` per `pi`
    # column, so it is a fact about `PI_COLS` rather than a literal. The sweep only ever
    # fits the default block, but a `PI_COLS` that grew without this number following it
    # would silently misreport the confound the whole of §7d is about.
    from src.models.availability_window import PI_COLS
    assert L2_UNPENALIZED["mixture"] == 3 + len(PI_COLS)


# ── §12: the absence-composition block and the compound counting process ──────
#
# `src/models/availability_absence.py` crosses a covariate block against a likelihood, and
# both halves are pinned here rather than in a file of their own because they extend the
# same axis these tests already cover: the arm has to reproduce the head it nests, and the
# block has to stay outside the design every other head reaches its rows through.

def test_absence_mix_shares_are_nan_where_the_backfill_has_no_coverage():
    """The reason columns are structurally ZERO before 2006-07, not missing.

    `missed_decomposition` routes every absence to `missed_unknown` where the box-score
    backfill has not run, so the four named kinds are exactly 0 there. Taking shares without
    masking would tell the head there were no healthy scratches in 1997-98 — a fact about
    the backfill wearing the shape of a fact about the players, and one that no downstream
    assertion would catch because 0.0 is a perfectly valid share.
    """
    from src.models.availability import ABSENCE_MIX_SHARE_COLS, absence_mix_shares

    frame = pd.DataFrame({
        "season": ["2000-01", "2015-16"], "player_id": [7, 7],
        "missed_games": [20, 20], "missed_scratch": [0, 5], "missed_inactive": [0, 10],
        "missed_injury": [0, 2], "missed_not_rostered": [0, 3],
        "status_coverage": [0.0, 1.0]})
    shares = absence_mix_shares(frame)

    assert shares.loc[0, ABSENCE_MIX_SHARE_COLS].isna().all()
    assert shares.loc[1, ABSENCE_MIX_SHARE_COLS].notna().all()
    assert np.isclose(shares.loc[1, "missed_inactive_share"], 0.5)
    assert np.isclose(shares.loc[1, "missed_not_rostered_share"], 0.15)


def test_absence_mix_shares_are_zero_for_a_season_with_no_absences():
    # The documented choice at `missed_games == 0`. NaN would drop 4.56% of covered rows for
    # having had a healthy season, and a league mean would assert a composition of absences
    # the player did not have.
    from src.models.availability import ABSENCE_MIX_SHARE_COLS, absence_mix_shares

    frame = pd.DataFrame({
        "season": ["2015-16"], "player_id": [3], "missed_games": [0],
        "missed_scratch": [0], "missed_inactive": [0], "missed_injury": [0],
        "missed_not_rostered": [0], "status_coverage": [1.0]})
    shares = absence_mix_shares(frame)
    assert (shares[ABSENCE_MIX_SHARE_COLS].to_numpy() == 0.0).all()


def test_absence_mix_block_stays_out_of_every_other_heads_design():
    """The block is opt-in, and that is what makes it an ablation.

    `build_design` is imported by `stan_minutes`, `stan_composition`, `stan_games_played`,
    `model_cards`, `sim/season`, `season_terms` and `final_evaluation`. A column that does
    not exist before 2006-07 entering any of them through `LAG_COLS` or `FEATURE_COLS`
    would be silent — it is a valid float everywhere it appears.
    """
    from src.models.availability import (ABSENCE_MIX_COLS, ABSENCE_MIX_SHARE_COLS,
                                         FEATURE_COLS, LAG_COLS)

    assert not set(ABSENCE_MIX_COLS) & set(FEATURE_COLS)
    assert not [c for c in LAG_COLS if c.startswith("missed_")]
    assert ABSENCE_MIX_COLS == [f"{c}_lag1" for c in ABSENCE_MIX_SHARE_COLS]


def test_attach_absence_mix_reads_the_prior_season_only():
    # The block is a season S-1 quantity like every other lag column. A merge that picked up
    # the CURRENT season's composition would be the leak `assert_point_in_time` exists for,
    # and it would score beautifully.
    from src.models.availability import attach_absence_mix

    seasons = ["2014-15", "2015-16"]
    frame = pd.DataFrame({
        "season": ["2014-15", "2015-16"], "player_id": [4, 4],
        "missed_games": [10, 4], "missed_scratch": [10, 0], "missed_inactive": [0, 4],
        "missed_injury": [0, 0], "missed_not_rostered": [0, 0],
        "status_coverage": [1.0, 1.0]})
    design = pd.DataFrame({
        "season": ["2015-16"], "player_id": [4],
        "as_of_date": pd.to_datetime(["2015-10-26"]),
        "season_start_date": pd.to_datetime(["2015-10-27"])})

    out = attach_absence_mix(design, frame, seasons)
    # 2014-15's composition — all scratch — not 2015-16's, which is all inactive.
    assert np.isclose(out.loc[0, "missed_scratch_share_lag1"], 1.0)
    assert np.isclose(out.loc[0, "missed_inactive_share_lag1"], 0.0)


def test_fast_onset_grid_reproduces_scipys_beta_binomial():
    """The compound's inner loop replaces `betabinom.pmf` with a ratio recursion.

    It is ~10x cheaper and it is evaluated thousands of times per fit, so a subtly wrong
    fast path would not raise — it would show up as a likelihood that is merely slightly
    worse, which is indistinguishable from the null this arm is testing for.
    """
    from src.models.availability_absence import _onset_pmf_grid
    from src.models.availability_window import _bb_pmf_grid

    rng = np.random.default_rng(11)
    n = rng.choice([66.0, 72.0, 82.0], 200)
    h = rng.uniform(0.03, 0.7, 200)
    rho = rng.choice([0.05, 0.2, 0.31], 200)
    k = np.arange(83)
    assert np.allclose(_onset_pmf_grid(n, h, rho, k), _bb_pmf_grid(n, h, rho, k),
                       atol=1e-13)
    # Mass above a row's own schedule is impossible, and the recursion's terms there are
    # garbage rather than small — they are masked, not clipped.
    assert (_onset_pmf_grid(n, h, rho, k)[n[:, None] < k[None, :]] == 0.0).all()


def test_compound_at_lambda_one_is_the_beta_binomial_to_machine_precision():
    """`lambda = 1` makes every spell one game, and the beta-binomial's `y -> n - y`
    symmetry then makes `missed ~ BetaBinom(n, 1 - mu, rho)` the incumbent exactly.

    Checked on the log-likelihood AND on the predictive, for the reason the `finite_mix`
    test gives one section up: an arm that nests in likelihood but not in pmf scores
    identically on train and differently on validation, which is the silent version of the
    same failure.
    """
    from src.models.availability_absence import CompoundCountingFrailty
    from src.models.availability_window import _bb_pmf_grid, _sigmoid

    train = _frailty_frame(300)
    model = CompoundCountingFrailty(l2=1.0).fit(train)
    assert assert_nests(model, train) < 1e-8

    df = _frailty_frame(60, seed=5)
    n = df["team_games"].to_numpy(dtype=float)
    eta = model._design(df) @ model.beta
    disp = model._disp_row(df, model.dispersion)
    k = np.arange(83)
    at_one = model._pmf(n, eta, disp, model._extra0(), k, df)
    assert np.allclose(at_one, _bb_pmf_grid(n, _sigmoid(eta), disp, k), atol=1e-12)


def test_compound_piles_truncated_mass_at_a_dead_season_and_still_sums_to_one():
    """A player cannot miss more than his team plays, and the excess is piled rather than
    renormalized.

    Renormalizing would redistribute mass the schedule has already ruled out back across
    the support — moving probability *out* of the `gp = 0` tail this arm exists to fill.
    Piling is also what makes the pmf sum to one by construction instead of by cancellation,
    which is the half a proper-predictive check can see.
    """
    from src.models.availability_absence import CompoundCountingFrailty, _onset_pmf_grid
    from src.models.availability_window import _sigmoid

    train = _frailty_frame(200)
    model = CompoundCountingFrailty(l2=1.0).fit(train)
    df = _frailty_frame(40, seed=6)
    df["team_games"] = 20                      # a short schedule, so long spells run off it
    n = df["team_games"].to_numpy(dtype=float)
    eta = model._design(df) @ model.beta
    disp = model._disp_row(df, model.dispersion)
    # No point mass at one game and a heavy-tailed duration: `sum_j L_j > n` is common here.
    extra = np.array([0.0, float(np.log(0.2 / 0.8)), float(np.log(1.5))])
    k = np.arange(21)

    pmf = model._pmf(n, eta, disp, extra, k, df)
    assert (pmf >= 0).all()
    assert np.allclose(pmf.sum(axis=1), 1.0, atol=1e-12)

    conv, _ = model._tables(extra, 20)
    untruncated = (_onset_pmf_grid(n, 1.0 - _sigmoid(eta), disp, k) @ conv)[:, 20]
    # The pile is strictly more than the mass that landed exactly on `missed == n`, which is
    # the mass from `missed > n` arriving where it belongs.
    assert (pmf[:, 0] > untruncated + 1e-9).all()


def test_compound_profile_pins_lambda_without_moving_its_nesting_point():
    # `lambda_profile` is what separates "the optimizer stopped at the corner" from "the
    # corner is the MLE", so a pinned arm still has to reproduce the incumbent — otherwise
    # the profile is of a different model at every grid point.
    from src.models.availability_absence import CompoundCountingFrailty

    train = _frailty_frame(200, seed=7)
    pinned = CompoundCountingFrailty(l2=1.0, lambda_fixed=0.25).fit(train)
    assert np.isclose(pinned.extra[0], 0.25)
    assert assert_nests(pinned, train) < 1e-8
    assert pinned.name == "compound_lambda0.25"
    # And the duration block can be pinned too, at the spell shape §11b measured.
    both = CompoundCountingFrailty(l2=1.0, lambda_fixed=0.0, duration_fixed=True).fit(train)
    assert np.allclose(both.extra, both._extra0() * np.array([0.0, 1.0, 1.0]))


def test_crossed_arms_differ_only_in_the_feature_block_or_the_likelihood():
    """The 2x2 is only a 2x2 if the four cells vary one thing at a time."""
    from src.models.availability import ABSENCE_MIX_COLS, FEATURE_COLS
    from src.models.availability_absence import CompoundCountingFrailty, arm_spec

    plain_cls, plain_features, plain_kwargs = arm_spec("betabinom")
    mix_cls, mix_features, mix_kwargs = arm_spec("betabinom__absence_mix")
    comp_cls, comp_features, _ = arm_spec("compound")
    both_cls, both_features, _ = arm_spec("compound__absence_mix")

    assert plain_cls is mix_cls is BetaBinomFrailty
    assert comp_cls is both_cls is CompoundCountingFrailty
    assert plain_features == comp_features == list(FEATURE_COLS)
    assert mix_features == both_features == list(FEATURE_COLS) + list(ABSENCE_MIX_COLS)
    # §12's four cells never touch `pi`, so their kwargs stay empty and the arms are the ones
    # that were measured — the §14 machinery must not reach back into the earlier round.
    assert plain_kwargs == mix_kwargs == {}


# ── §14: the same block, crossed against the arm that ships ───────────────────
#
# §12 crossed the block against `betabinom` and the head that ships is `mixture`, so the two
# covariate lists have to be able to move independently: the mean function `beta` and the
# disruption weight `pi` are different questions about the same player. What is pinned here
# is that separation — a widened `pi` must not change the default arm, must still nest the
# incumbent, and must not leak into the arms that have no `pi` at all.

def test_pi_features_defaults_to_pi_cols_on_every_arm():
    """The default is the block every arm measured before 2026-08-12 was fitted with.

    `pi_features` lives on `FrailtyGLM` rather than on `MixtureFrailty` because the scaler and
    `_pi_design` do. That makes it reachable from arms that have no `pi`, so the default has
    to be the thing that reproduces §7 and §12 — otherwise widening one round's block would
    silently re-fit the other's.
    """
    from src.models.availability_window import PI_COLS

    for arm in (MixtureFrailty(l2=1.0), BetaBinomFrailty(l2=1.0),
                BetaRectangularFrailty(l2=1.0)):
        assert arm.pi_features == list(PI_COLS)
    assert len(MixtureFrailty(l2=1.0)._extra0()) == 3 + len(PI_COLS)


def test_a_widened_pi_block_still_nests_the_incumbent():
    """`theta = 0` has to stay the nesting point at any width of `pi`.

    This is the whole reason `pi` is `theta * sigmoid(gamma' z)` rather than
    `sigmoid(gamma_0 + gamma' z)`: the weight is switched off by `theta` alone, so adding
    columns to `z` adds parameters the nesting point does not depend on. If it did depend on
    them, §14's arms would each be a different model rather than an extension of the shipped
    one, and their margins would measure the widening rather than the block.
    """
    from src.models.availability import ABSENCE_MIX_COLS
    from src.models.availability_window import PI_COLS

    train = _frailty_frame(300, seed=8)
    rng = np.random.default_rng(9)
    for col in ABSENCE_MIX_COLS:
        train[col] = rng.uniform(0.0, 1.0, len(train))

    wide = MixtureFrailty(l2=1.0, pi_features=list(PI_COLS) + list(ABSENCE_MIX_COLS))
    fitted = wide.fit(train)
    assert len(fitted._extra0()) == 3 + len(PI_COLS) + len(ABSENCE_MIX_COLS)
    assert assert_nests(fitted, train) < 1e-8
    # And the widening is visible in the artifact rather than inferable from the arm's name,
    # since two §14 arms differ in nothing else.
    report = fitted.shape_report(train)
    assert report["n_pi_features"] == len(PI_COLS) + len(ABSENCE_MIX_COLS)


def test_the_mixture_round_puts_the_block_on_beta_alone_or_on_beta_and_pi():
    """§14's two candidates are different arms, and the block token is what separates them."""
    from src.models.availability import ABSENCE_MIX_COLS, FEATURE_COLS
    from src.models.availability_absence import arm_spec
    from src.models.availability_window import PI_COLS

    beta_cls, beta_features, beta_kwargs = arm_spec("mixture__absence_mix")
    both_cls, both_features, both_kwargs = arm_spec("mixture__absence_mix_pi")

    assert beta_cls is both_cls is MixtureFrailty
    # Same mean function in both — the only difference is whether `pi` sees the block too.
    assert beta_features == both_features == list(FEATURE_COLS) + list(ABSENCE_MIX_COLS)
    assert beta_kwargs == {}
    assert both_kwargs == {"pi_features": list(PI_COLS) + list(ABSENCE_MIX_COLS)}


def test_the_pi_block_is_refused_on_an_arm_with_no_pi():
    # A silently-ignored `pi_features` on `betabinom` would produce a row that looks like a
    # third candidate and is a duplicate of the second.
    from src.models.availability_absence import arm_spec

    with pytest.raises(ValueError, match="only `mixture` has a `pi`"):
        arm_spec("betabinom__absence_mix_pi")
    with pytest.raises(ValueError, match="unknown covariate block"):
        arm_spec("mixture__absence_mixture")


def test_margin_columns_are_named_after_the_arm_they_were_computed_against():
    """The two rounds have different incumbents, and that has to reach the column names.

    §12 quotes `betabinom` and §14 quotes `mixture`. A hard-coded `_vs_betabinom` suffix on a
    round referenced to `mixture` would be a margin naming an arm it was never computed
    against — an error no assertion downstream could see, because the numbers are all valid.
    """
    from src.models.availability_absence import _bootstrap_arms

    rng = np.random.default_rng(4)
    def parts(shift):
        out = {}
        for key in ("full", "low_shoulder", "high_shoulder", "10", "41", "60"):
            p = np.clip(rng.uniform(size=200) * 0.3 + shift, 0.0, 1.0)
            out[f"p_{key}"] = p
            out[f"o_{key}"] = (rng.uniform(size=200) < 0.3).astype(float)
        return out

    tails = {"mixture": parts(0.0), "mixture__absence_mix": parts(0.05)}
    out = _bootstrap_arms(tails, tuple(tails), "mixture", reps=50, seed=0)

    assert "boundary_vs_mixture" in out["mixture__absence_mix"]
    assert "boundary_vs_betabinom" not in out["mixture__absence_mix"]
    # The reference carries its own interval and no margin against itself.
    assert "boundary_tail_error_lo" in out["mixture"]
    assert not [k for k in out["mixture"] if "_vs_" in k]


def test_the_lambda_profiles_margin_columns_are_named_for_the_incumbent():
    """The profile's reference row is `compound_lambda1`, and its columns say `betabinom`.

    That is the nesting identity rather than a shortcut — at `lambda = 1` the compound IS the
    incumbent beta-binomial — and it is load bearing twice: `lambda_profile` reads
    `boundary_vs_betabinom_hi` back to compute D1, and `make docs-audit` re-derives six §12d
    figures from those column names. Deriving the suffix from the reference arm's name, which
    is right for every other caller, would rename them to `boundary_vs_compound_lambda1` and
    turn a published table into a `KeyError` on the next full run.
    """
    from src.models.availability_absence import _bootstrap_arms

    rng = np.random.default_rng(5)
    tails = {}
    for name in ("compound_lambda1", "compound_lambda0"):
        d = {}
        for key in ("full", "low_shoulder", "high_shoulder", "10", "41", "60"):
            d[f"p_{key}"] = rng.uniform(0.0, 0.4, 80)
            d[f"o_{key}"] = (rng.uniform(size=80) < 0.3).astype(float)
        tails[name] = d

    out = _bootstrap_arms(tails, tuple(tails), "compound_lambda1", reps=30, seed=0,
                          suffix="betabinom")
    assert "boundary_vs_betabinom_hi" in out["compound_lambda0"]
    assert not [k for k in out["compound_lambda0"] if "compound_lambda1" in k]
    # Without the override the suffix follows the reference, which is what every other
    # caller needs — the two behaviours are one function and both are pinned.
    derived = _bootstrap_arms(tails, tuple(tails), "compound_lambda1", reps=30, seed=0)
    assert "boundary_vs_compound_lambda1_hi" in derived["compound_lambda0"]


def test_the_interaction_row_is_a_difference_of_differences_and_names_no_arm():
    """The generalized effects table has to reproduce what §12's artifact already holds.

    Two things it would break silently. The interaction is
    `(arm - base) - (other arm - other base)` and a sign error there would read as a plausible
    small number; and the interaction row carries **no** `arm` or `baseline` — §12's artifact
    has them empty, because an effect of two effects is not an arm's row, and `make docs-audit`
    looks the row up by `effect` alone.
    """
    from src.models.availability_absence import interaction_table

    rng = np.random.default_rng(3)
    arms = ("a_block", "a", "b_block", "b")
    per_row = {name: rng.uniform(8.0, 12.0, 120) for name in arms}
    tails = {}
    for name in arms:
        d = {}
        for key in ("full", "low_shoulder", "high_shoulder", "10", "41", "60"):
            d[f"p_{key}"] = rng.uniform(0.0, 0.4, 120)
            d[f"o_{key}"] = (rng.uniform(size=120) < 0.3).astype(float)
        tails[name] = d

    out = interaction_table(
        per_row, tails, reps=40, seed=0,
        contrasts=(("block | a", "a_block", "a"), ("block | b", "b_block", "b")),
        interactions=(("interaction", "a_block", "a", "b_block", "b"),))

    crps = out[out["metric"] == "val_crps"].set_index("effect")
    want = crps.loc["block | a", "delta"] - crps.loc["block | b", "delta"]
    assert np.isclose(crps.loc["interaction", "delta"], want)
    assert crps.loc["interaction", "arm"] == ""
    assert crps.loc["interaction", "baseline"] == ""
    assert np.isnan(crps.loc["interaction", "arm_value"])
    # And the main-effect rows still name theirs, or the table is unreadable.
    assert crps.loc["block | a", "arm"] == "a_block"


def test_an_unknown_round_name_raises_before_anything_is_fitted():
    """The rounds gate exists so §14 can be re-run without touching §12's five artifacts.

    A typo in the config key must fail immediately rather than silently running the default
    pair — which would spend the compound profile and overwrite the files `make docs-audit`
    re-derives sixty-odd figures from. Checked before `load_design`, so the error costs
    nothing.
    """
    from src.models.availability_absence import ROUNDS, run

    cfg = {"evaluation": {"predictions_dir": "outputs/predictions"},
           "features": {"availability": {"absence": {"rounds": ["crossed", "mixtures"]}}}}
    with pytest.raises(ValueError, match="unknown absence round"):
        run(cfg)
    assert ROUNDS == ("crossed", "mixture", "population")


def test_the_mixture_rounds_interaction_is_the_blocks_margin_at_both_likelihoods():
    """The redundancy question is an interaction, and it needs all four arms' rows.

    `[block | mixture] - [block | betabinom]` is what says whether the two-component head
    already had the block's information. Subtracting §12's published margin from §14's would
    give the same point estimate and no interval, because the bootstrap has to resample the
    four arms on ONE set of row indices.
    """
    from src.models.availability_absence import (MIXTURE_ARMS, MIXTURE_CONTRASTS,
                                                 MIXTURE_INTERACTIONS, MIXTURE_REFERENCE)

    label, arm, base, other_arm, other_base = MIXTURE_INTERACTIONS[0]
    assert (arm, base) == ("mixture__absence_mix", "mixture")
    assert (other_arm, other_base) == ("betabinom__absence_mix", "betabinom")
    # Every arm any effect names has to be an arm the ladder actually fits, or the effect is
    # a KeyError at the end of a twenty-minute run.
    named = {a for _, x, y in MIXTURE_CONTRASTS for a in (x, y)}
    named |= {a for spec in MIXTURE_INTERACTIONS for a in spec[1:]}
    assert named <= set(MIXTURE_ARMS)
    assert MIXTURE_REFERENCE in MIXTURE_ARMS


# ── §15: a population is a SCORING restriction, never a refit ─────────────────

def _population_frame(n_rows: int = 400, seed: int = 0) -> pd.DataFrame:
    """`_frailty_frame` plus the two columns §15's round reads.

    `on_season_start_roster` is the draft-pool mask, and `ABSENCE_MIX_COLS` are the block
    `crossed_ladder` refuses to fit without.
    """
    from src.models.availability_absence import ABSENCE_MIX_COLS

    rng = np.random.default_rng(seed + 1)
    frame = _frailty_frame(n_rows, seed)
    for col in ABSENCE_MIX_COLS:
        frame[col] = rng.uniform(0.0, 0.4, n_rows)
    # Deliberately NOT all-ones: a mask that keeps every row would let a bug that ignores
    # the mask entirely pass this file.
    frame["on_season_start_roster"] = (np.arange(n_rows) % 4 != 0).astype(float)
    return frame


def test_the_two_populations_are_one_fit_scored_twice():
    """§15's whole design, and what makes "the same head reads worse on the draft pool" a
    legitimate sentence.

    If the draftable column came from arms refitted on draftable rows it would be a
    different model answering a different question. `fit_arms` takes no validation frame at
    all, which is how that is enforced rather than remembered.
    """
    import inspect

    from src.models.availability_absence import fit_arms, population_ladder

    assert "val" not in inspect.signature(fit_arms).parameters

    train, val = _population_frame(500), _population_frame(240, seed=7)
    arms = ("mixture", "betabinom")
    table = population_ladder(train, val, 82, arms=arms, seed=0)

    assert set(table["population"]) == {"all", "draftable"}
    assert len(table) == len(arms) * 2
    draftable = int((val["on_season_start_roster"] > 0).sum())
    assert set(table.loc[table["population"] == "all", "n_val"]) == {len(val)}
    assert set(table.loc[table["population"] == "draftable", "n_val"]) == {draftable}
    # The fitted objects are shared across populations, so anything read off the FIT rather
    # than off the scored rows has to be identical between the two columns.
    for name in arms:
        rows = table[table["arm"] == name]
        assert rows["train_loglik"].nunique() == 1
        assert rows["n_params"].nunique() == 1


def test_the_population_verdict_reads_the_single_component_arm_against_the_mixture():
    """§7 chose `mixture` over `betabinom` on the boundary, so the quantity that settles
    whether that transfers is the `betabinom` − `mixture` margin — positive and clear of
    zero means the single-component head is ALSO worse on the draft pool."""
    from src.models.availability_absence import population_verdict

    table = pd.DataFrame([
        {"arm": "betabinom", "population": "all", "boundary_vs_mixture": +0.009,
         "boundary_vs_mixture_lo": +0.004, "boundary_vs_mixture_hi": +0.010},
        {"arm": "betabinom", "population": "draftable", "boundary_vs_mixture": -0.003,
         "boundary_vs_mixture_lo": -0.007, "boundary_vs_mixture_hi": -0.002},
    ])
    verdict = population_verdict(table)
    assert verdict["all"]["selection_survives"] is True
    assert verdict["draftable"]["selection_survives"] is False

    # An interval spanning zero is not a survival either — the bar is the interval, not the
    # point estimate, which is the half a reader is most likely to soften later.
    spanning = table.copy()
    spanning.loc[spanning["population"] == "all", "boundary_vs_mixture_lo"] = -0.001
    assert population_verdict(spanning)["all"]["selection_survives"] is False


def test_the_rolling_harness_refuses_a_draftable_reading_it_cannot_take():
    """`absence_rolling` masks the SCORED rows, so it needs the roster column on the frame
    it walks. Failing loudly beats silently returning a pooled number under a draftable
    label — the class of error §15 exists to correct."""
    from src.models.availability_absence import absence_rolling

    with pytest.raises(ValueError, match="on_season_start_roster"):
        absence_rolling(_frailty_frame(80), 82, populations=("all", "draftable"))


# ── §15b: the recency axis on the no-design pool ──────────────────────────────

def test_a_recency_window_keeps_the_last_k_seasons_and_none_keeps_them_all():
    """The knob §15b turns. `None` has to be the shipped estimator EXACTLY — it is the
    reference every margin in that table is taken against, so a `recency` that quietly
    dropped a season would make the whole column measure two changes."""
    from src.models.availability_no_prior import level_tables

    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(4, 20)]
    history = pd.DataFrame({
        "season": np.repeat(seasons, 60),
        "all": "all",
        "gp": np.tile(np.arange(60), len(seasons)),
        "team_games": 82})

    target = seasons[-1]
    full = level_tables(history, target, seasons, "pooled")[0][1]
    last5 = level_tables(history, target, seasons, "pooled", recency=5)[0][1]

    # 15 eligible seasons before the target at 60 rows each; a 5-season cut sees a third.
    assert int(full["rows"].iloc[0]) == 15 * 60
    assert int(last5["rows"].iloc[0]) == 5 * 60
    # And `None` is the identity, not "a very large window".
    assert level_tables(history, target, seasons, "pooled",
                        recency=None)[0][1].equals(full)


def test_the_recency_verdict_needs_both_halves_of_p4s_gate():
    """P4's bar is unchanged by §15b: a rolling interval clear of zero AND a validation one.
    The roster arm has always cleared the first and failed the second, and a recency cut
    that fixed only the bias must not be allowed to read as a pass."""
    from src.models.availability_no_prior import recency_verdict

    def row(recency, split, delta, hi):
        return {"estimator": "roster", "recency": recency, "split": split,
                "population": "draftable", "crps_vs_shipped": delta,
                "crps_vs_shipped_lo": delta - 0.3, "crps_vs_shipped_hi": hi,
                "origins_won_vs_shipped": 13, "origins_compared_vs_shipped": 19,
                "bias": 0.06}

    # What was measured: rolling clears, validation does not.
    table = pd.DataFrame([row("all", "rolling", -0.3486, -0.0183),
                          row("all", "validation", +0.7354, +1.5067)])
    assert recency_verdict(table)["all"]["passes"] is False

    # Both halves clearing is the only thing that passes.
    both = pd.DataFrame([row("all", "rolling", -0.3486, -0.0183),
                         row("all", "validation", -0.5000, -0.1000)])
    assert recency_verdict(both)["all"]["passes"] is True
