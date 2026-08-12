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
    mu = 0.45 + 0.012 * mpg
    frame["gp"] = rng.binomial(82, np.clip(mu, 0.02, 0.98))
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
