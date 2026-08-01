"""Tests for the Stan heads: availability, minutes, and the component rates.

Split deliberately into two kinds. Most of what follows needs no sampler at all — the
closed-form CRPS, the target construction, the floors, the selection logic — and runs in
milliseconds. A handful genuinely need CmdStan, and those are skipped rather than failed
when it is absent, because CmdStan is a ~1 GB toolchain that pip does not manage.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.availability import crps as pmf_crps
from src.models.stan_utils import (crps_from_samples, ks_uniform,
                                   pit_from_samples, prior_sd_for_l2, thin)


def _has_cmdstan() -> bool:
    try:
        import cmdstanpy
        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


needs_cmdstan = pytest.mark.skipif(
    not _has_cmdstan(),
    reason="CmdStan is not installed; "
           "run python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")


# ── The closed-form CRPS ──────────────────────────────────────────────────────

def _crps_double_sum(samples, y):
    """The literal O(S^2) definition, for pinning the O(S log S) form against."""
    out = []
    for j in range(samples.shape[1]):
        x = samples[:, j]
        term1 = np.abs(x - y[j]).mean()
        term2 = np.abs(x[:, None] - x[None, :]).mean()
        out.append(term1 - 0.5 * term2)
    return np.array(out)


def test_crps_from_samples_matches_the_literal_double_sum():
    # The order-statistic identity is exactly the kind of closed form that is quietly
    # wrong by a factor of two, so it is pinned against the definition it replaces.
    rng = np.random.default_rng(0)
    samples = rng.integers(0, 40, size=(120, 7)).astype(float)
    y = rng.integers(0, 40, size=7).astype(float)
    assert np.allclose(crps_from_samples(samples, y), _crps_double_sum(samples, y))


def test_crps_from_samples_matches_the_exact_pmf_crps():
    """Against `availability.crps`, which sums over an explicit pmf grid.

    Two independent implementations of the same quantity, one of which is already used
    for the reported availability numbers. They agree to Monte Carlo error.
    """
    rng = np.random.default_rng(1)
    n, p = 40, 0.35
    grid = np.arange(n + 1)
    from scipy.stats import binom
    pmf = binom.pmf(grid, n, p)[None, :]
    y = np.array([14])
    exact = pmf_crps(pmf, y)[0]

    samples = rng.binomial(n, p, size=(200_000, 1)).astype(float)
    assert abs(crps_from_samples(samples, y.astype(float))[0] - exact) < 0.02


def test_crps_is_zero_for_a_point_mass_on_the_truth():
    samples = np.full((50, 3), 7.0)
    assert np.allclose(crps_from_samples(samples, np.array([7.0, 7.0, 7.0])), 0.0)


def test_crps_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        crps_from_samples(np.zeros((10, 3)), np.zeros(4))


def test_a_tighter_correct_forecast_scores_better_than_a_vague_one():
    rng = np.random.default_rng(2)
    y = np.array([20.0])
    tight = rng.normal(20, 2, size=(5000, 1))
    vague = rng.normal(20, 10, size=(5000, 1))
    assert crps_from_samples(tight, y)[0] < crps_from_samples(vague, y)[0]


# ── PIT ───────────────────────────────────────────────────────────────────────

def test_pit_is_uniform_when_the_predictive_is_calibrated():
    rng = np.random.default_rng(3)
    n, p = 60, 0.4
    y = rng.binomial(n, p, size=800).astype(float)
    samples = rng.binomial(n, p, size=(600, 800)).astype(float)
    assert ks_uniform(pit_from_samples(samples, y, seed=3)) < 0.06


def test_pit_detects_a_biased_predictive():
    rng = np.random.default_rng(4)
    n = 60
    y = rng.binomial(n, 0.4, size=800).astype(float)
    samples = rng.binomial(n, 0.7, size=(600, 800)).astype(float)   # wrong mean
    assert ks_uniform(pit_from_samples(samples, y, seed=4)) > 0.5


# ── Prior / penalty correspondence ────────────────────────────────────────────

def test_prior_sd_reproduces_the_l2_penalty_it_replaces():
    # -loglik + l2*||b||^2  ==  -loglik + ||b||^2/(2*sigma^2)  iff sigma = 1/sqrt(2*l2).
    for l2 in (0.5, 1.0, 100.0):
        sd = prior_sd_for_l2(l2)
        assert np.isclose(1.0 / (2.0 * sd ** 2), l2)


def test_prior_sd_rejects_a_zero_penalty():
    # An L2 of 0 is a flat (improper) prior, not a normal one — refuse rather than return
    # inf and let it surface as a Stan initialization failure three frames later.
    with pytest.raises(ValueError):
        prior_sd_for_l2(0.0)


def test_thin_is_evenly_spaced_and_never_oversamples():
    idx = thin(4000, 400)
    assert len(idx) == 400 and idx[0] == 0 and idx[-1] == 3999
    assert len(np.unique(idx)) == 400
    assert len(thin(50, 400)) == 50          # asking for more than exist keeps all


# ── Availability: the binomial-support guard ─────────────────────────────────

def _availability_rows(gp, team_games):
    return pd.DataFrame({"player_id": range(len(gp)),
                         "season": ["2020-21"] * len(gp),
                         "gp": gp, "team_games": team_games})


def test_binomial_support_guard_rejects_gp_above_team_games():
    """The 13-traded-player trap. A summed log-likelihood is non-finite at EVERY rho if
    any single row violates the support, so this has to fail loudly rather than fit."""
    from src.models.stan_availability import assert_binomial_support

    with pytest.raises(ValueError, match="gp <= team_games"):
        assert_binomial_support(_availability_rows([83, 70], [82, 82]))


def test_binomial_support_guard_passes_the_repaired_frame():
    from src.models.stan_availability import assert_binomial_support

    frame = _availability_rows([83, 70], [82, 82])
    frame["team_games"] = np.maximum(frame["team_games"], frame["gp"])
    assert len(assert_binomial_support(frame)) == 2


# ── Minutes: trials come from game length, never 48 ───────────────────────────

def _player_games(minutes, lengths, season="2020-21", player_id=1):
    """Synthetic played rows plus the matching game-length frame.

    Names and ids stay distinct after any normalization, per `CLAUDE.md` — the P0..P39
    fixture that silently exercised a 40-way key collision and passed is the reason.
    """
    games = np.arange(len(minutes)) + 1
    targets = pd.DataFrame({"player_id": player_id, "season": season,
                            "season_type": "regular", "game_id": games,
                            "min": minutes, "played": 1})
    length = pd.DataFrame({"season": season, "season_type": "regular",
                           "game_id": games, "game_length": lengths, "reliable": True})
    return targets, length


def test_minutes_trials_use_overtime_length_not_48():
    from src.models.stan_minutes import minutes_targets

    # Two regulation games and one double-overtime game. Truncating at 48 would understate
    # the denominator by 10 minutes and inflate the fitted rate.
    targets, lengths = _player_games([40.0, 44.0, 52.0], [48.0, 48.0, 58.0])
    out = minutes_targets(targets, lengths)
    assert out["length_played"].iloc[0] == 154.0
    assert out["minutes_played"].iloc[0] == 136.0
    assert np.isclose(out["minutes_share"].iloc[0], 136.0 / 154.0)


def test_minutes_share_never_exceeds_one_even_over_48():
    """A 52-minute game is legal — 1,650 player-games exceed 48 and the maximum is 63.0."""
    from src.models.stan_minutes import minutes_targets

    targets, lengths = _player_games([52.0], [53.0])
    assert minutes_targets(targets, lengths)["minutes_share"].iloc[0] <= 1.0


def test_minutes_targets_raise_rather_than_drop_an_unmatched_game():
    from src.models.stan_minutes import minutes_targets

    targets, lengths = _player_games([40.0, 44.0], [48.0, 48.0])
    with pytest.raises(ValueError, match="no game length"):
        minutes_targets(targets, lengths.iloc[:1])


def test_minutes_targets_exclude_playoff_rows():
    """Every fitting frame is regular season only; playoff minutes are a role interaction
    whose sign flips with role, so pooling them corrupts the head."""
    from src.models.stan_minutes import minutes_targets

    targets, lengths = _player_games([40.0], [48.0])
    playoff = targets.assign(season_type="playoffs", game_id=99)
    playoff_length = lengths.assign(season_type="playoffs", game_id=99)
    out = minutes_targets(pd.concat([targets, playoff], ignore_index=True),
                          pd.concat([lengths, playoff_length], ignore_index=True))
    assert out["minutes_played"].iloc[0] == 40.0
    assert out["games_played"].iloc[0] == 1


def test_as_trials_rounds_without_ever_clamping_valid_input():
    """Rounding cannot break the support, and it is worth knowing *why* rather than
    trusting the guard: summed game lengths are exact integers (48 + 5k per game), and
    `rint` of a value at or below an integer stays at or below it. So the observed
    clamp count on the real 9,804-row design is 0, and a nonzero one means the
    denominator is wrong rather than that rounding is."""
    from src.models.stan_minutes import as_trials

    frame = pd.DataFrame({"minutes_played": [47.7, 100.2], "length_played": [48.0, 144.0]})
    out = as_trials(frame)
    assert out["successes"].tolist() == [48, 100]
    assert out["trials"].tolist() == [48, 144]
    assert out["rounding_clamped"].tolist() == [0, 0]


def test_as_trials_clamps_an_impossible_row_rather_than_emitting_it():
    """Defence in depth against a corrupted denominator. `successes > trials` makes the
    beta-binomial non-finite, and because the log-likelihood is a sum, one bad row takes
    the whole fit down at every rho — the `gp > team_games` trap, one minute wide."""
    from src.models.stan_minutes import as_trials

    frame = pd.DataFrame({"minutes_played": [50.0], "length_played": [48.0]})
    out = as_trials(frame)
    assert out["successes"].tolist() == [48]
    assert (out["successes"] <= out["trials"]).all()
    assert out["rounding_clamped"].tolist() == [1]


def test_minutes_floor_is_prior_share_times_realized_length():
    from src.models.stan_minutes import carry_forward

    frame = pd.DataFrame({"minutes_share_lag1": [0.5, 0.25],
                          "length_played": [4000.0, 3000.0]})
    assert carry_forward(frame).tolist() == [2000.0, 750.0]


def test_minutes_variants_leave_age_linear_and_replace_the_own_term():
    """Splining age measured WORSE on both splits; `age + age_sq` already fits the arc.
    The flexibility goes to the prior-minutes term and nowhere else."""
    from src.models.availability import FEATURE_COLS
    from src.models.stan_minutes import OWN, variants

    frame = pd.DataFrame({c: np.linspace(0.1, 1.0, 30) for c in FEATURE_COLS})
    frame[OWN] = np.linspace(-2.0, 2.0, 30)
    out = variants(frame, frame, n_knots=4)

    for label in ("logit_own", "logit_own_quadratic", "logit_own_spline"):
        features = out[label][2]
        assert "minutes_per_game_lag1" not in features
        assert "age" in features and "age_sq" in features
        assert not any(f.startswith("age__") for f in features)
    assert any(f.startswith(f"{OWN}__s") for f in out["logit_own_spline"][2])
    assert len(out["logit_own_spline"][2]) > len(out["logit_own"][2])


# ── Components: substitution, selection, floors ──────────────────────────────

def test_substitution_columns_are_exact_and_additive():
    from src.models.stan_components import add_substitution_columns

    design = pd.DataFrame({"fg2a": [500, 300], "fg3a": [200, 100],
                           "fg2a_p36_lag1": [8.0, 6.0], "fg3a_p36_lag1": [2.0, 4.0]})
    out = add_substitution_columns(design)
    assert out["fga"].tolist() == [700, 400]
    # Per-36 rates share a denominator, so the total is the sum exactly.
    assert out["fga_p36_lag1"].tolist() == [10.0, 10.0]
    assert np.allclose(out["fg3a_share_lag1"], [0.2, 0.4])


def test_substitution_share_is_nan_rather_than_zero_when_there_were_no_attempts():
    # "He took no shots last season" and "he took 0% threes" are different statements;
    # collapsing them would tell the imputer a rim-runner is a non-shooter.
    from src.models.stan_components import add_substitution_columns

    design = pd.DataFrame({"fg2a": [0], "fg3a": [0],
                           "fg2a_p36_lag1": [0.0], "fg3a_p36_lag1": [0.0]})
    assert np.isnan(add_substitution_columns(design)["fg3a_share_lag1"].iloc[0])


def _sweep_table():
    return pd.DataFrame([
        {"head": "reb", "variant": "carry_forward", "val_r2": 0.90, "test_r2": 0.94},
        {"head": "reb", "variant": "log_own", "val_r2": 0.93, "test_r2": 0.945},
        {"head": "reb", "variant": "log_own_spline", "val_r2": 0.91, "test_r2": 0.99},
    ])


def test_selection_reads_validation_and_ignores_a_better_test_column():
    """The protocol this project exists to enforce. `log_own_spline` wins on test by a
    mile and loses on validation; selecting it would repeat a false positive whose paired
    bootstrap read [-0.079, -0.015] with P(delta<0) = 99.7% and did not replicate."""
    from src.models.stan_components import _finalize

    out = _finalize(_sweep_table(), "val_r2", "test_r2", higher_is_better=True)
    selected = out.loc[out["selected"], "variant"].tolist()
    assert selected == ["log_own"]


def test_beats_floor_is_a_test_fact_and_selection_is_a_validation_fact():
    from src.models.stan_components import _finalize

    table = _sweep_table()
    table.loc[table["variant"] == "log_own", "test_r2"] = 0.90   # now below the floor
    out = _finalize(table, "val_r2", "test_r2", higher_is_better=True)
    row = out[out["variant"] == "log_own"].iloc[0]
    # Selected on validation, and still correctly reported as failing the floor. A head
    # that does not clear `carry_forward` is not a model, however it was chosen.
    assert bool(row["selected"]) and not bool(row["beats_floor"])
    assert bool(out[out["variant"] == "carry_forward"].iloc[0]["beats_floor"])


def test_lower_is_better_selection_flips_direction_for_the_nll_heads():
    from src.models.stan_components import _finalize

    table = pd.DataFrame([
        {"head": "ftm|fta", "variant": "carry_forward", "val_nll": 3.08, "test_nll": 3.08},
        {"head": "ftm|fta", "variant": "logit_own", "val_nll": 3.05, "test_nll": 3.09},
        {"head": "ftm|fta", "variant": "logit_own_spline", "val_nll": 3.11,
         "test_nll": 3.01},
    ])
    out = _finalize(table, "val_nll", "test_nll", higher_is_better=False)
    assert out.loc[out["selected"], "variant"].tolist() == ["logit_own"]
    # The known case: nothing beats the floor on free-throw percentage.
    assert not bool(out[out["variant"] == "logit_own"].iloc[0]["beats_floor"])


def test_count_variants_spline_nests_the_log_scale():
    """The spline is built on log(own), not raw own, so `log_own_spline` strictly nests
    `log_own` and the contrast isolates curvature GIVEN the right scale."""
    from src.models.stan_components import count_variants

    n = 60
    frame = pd.DataFrame({
        "reb_p36_lag1": np.linspace(0.5, 12.0, n),
        "mpg_lag1": np.linspace(5, 35, n), "total_minutes_lag1": np.linspace(300, 2500, n),
        "gp_lag1": np.linspace(20, 82, n), "age": np.linspace(20, 38, n),
        "age_sq": np.linspace(20, 38, n) ** 2, "career_year": np.arange(n) % 15,
    })
    out = count_variants(frame, frame, "reb", n_knots=4)
    assert "log_reb_p36_lag1" in out["log_own"][2]
    assert "reb_p36_lag1" not in out["log_own"][2]
    assert any(f.startswith("log_reb_p36_lag1__s") for f in out["log_own_spline"][2])
    assert "log_reb_p36_lag1" not in out["log_own_spline"][2]


# ── The Stan sources themselves ───────────────────────────────────────────────

def test_betabinomial_source_uses_the_numerically_safe_complement():
    """`1 - inv_logit(eta)` is exactly 0 in double precision by eta ~ 37, which makes a
    beta shape parameter 0 and rejects the whole target. `inv_logit(-eta)` is
    algebraically identical and stays positive to eta ~ 745. Pinned so it is not
    "simplified" back."""
    from src.models.stan_utils import STAN_DIR

    # Comments are stripped first — the file explains the fragile form at length, and
    # matching that prose instead of the code would make this test pass or fail on the
    # documentation rather than on the model.
    code = "\n".join(line.split("//")[0] for line
                     in (STAN_DIR / "betabinomial_glm.stan").read_text().splitlines())
    assert "inv_logit(-eta)" in code
    assert "1 - inv_logit" not in code


def test_negbinomial_source_offsets_by_log_exposure_in_transformed_data():
    """Exposure is an offset with a fixed coefficient of 1, never a fitted covariate —
    and `log(exposure)` is constant, so recomputing it every gradient evaluation is waste."""
    from src.models.stan_utils import STAN_DIR

    source = (STAN_DIR / "negbinomial_glm.stan").read_text()
    assert "transformed data" in source and "log_exposure = log(exposure)" in source
    # `eta` is assembled once and passed whole, so the offset is checked where it is
    # BUILT rather than inside the likelihood call — the optional year term made the
    # single-expression form untenable and this is the property that actually matters.
    assert "vector[N] eta = log_exposure + alpha + X * beta;" in source
    assert "neg_binomial_2_log(eta, phi)" in source
    # Never a fitted covariate: `exposure` must not reach the design matrix.
    assert "X * beta" in source and "exposure * beta" not in source


# ── End to end, with a real sampler ──────────────────────────────────────────

@needs_cmdstan
def test_stan_posterior_mean_reproduces_the_penalized_mle():
    """The port check, on synthetic data small enough to run in seconds.

    With `beta ~ normal(0, 1/sqrt(2*l2))` the posterior MODE is exactly the penalized MLE,
    so at this n the posterior MEAN has to land within a fraction of a posterior sd of it.
    This is the claim the whole availability module rests on, reduced to something a test
    can assert.
    """
    from src.models.availability import BetaBinomialGLM
    from src.models.stan_availability import StanAvailability

    rng = np.random.default_rng(11)
    n_rows = 1500
    features = ["gp_share_lag1", "minutes_per_game_lag1"]
    frame = pd.DataFrame({
        "gp_share_lag1": rng.uniform(0.2, 1.0, n_rows),
        "minutes_per_game_lag1": rng.uniform(4.0, 36.0, n_rows),
    })
    eta = -0.4 + 1.1 * frame["gp_share_lag1"] + 0.03 * frame["minutes_per_game_lag1"]
    mu = 1.0 / (1.0 + np.exp(-eta))
    rho = 0.2
    scale = (1 - rho) / rho
    frame["team_games"] = 82
    frame["gp"] = rng.binomial(82, rng.beta(mu * scale, (1 - mu) * scale))

    mle = BetaBinomialGLM(l2=1.0, features=features).fit(frame)
    stan = StanAvailability(l2=1.0, features=features, warmup=750, samples=750,
                            chains=4, predictive_draws=100).fit(frame)

    assert stan.diagnostics["divergences"] == 0
    assert stan.diagnostics["max_rhat"] <= 1.01
    posterior_sd = np.r_[stan.alpha_draws.std(ddof=1), stan.beta_draws.std(axis=0, ddof=1)]
    assert np.all(np.abs(stan.beta - mle.beta) < 0.5 * posterior_sd)
    assert abs(stan.rho - mle.rho) < 0.02


@needs_cmdstan
def test_posterior_predictive_obeys_the_law_of_total_variance():
    """The mixture is a real mixture, and its spread decomposes the way it should.

    Deliberately NOT asserting that the integrated predictive is wider per player. It is
    not guaranteed to be: the mixture adds `Var_th(E[Y|th])` but replaces `Var(Y|th_bar)`
    with `E_th[Var(Y|th)]`, and `n*mu*(1-mu)*[1+(n-1)*rho]` is concave in mu, so Jensen
    pushes back. On this fixture the second effect wins and the mixture is *narrower*.
    What is checked instead is the identity itself, evaluated against the pmf — which is
    the thing that would break if the mixture were being formed wrongly.
    """
    from src.models.stan_availability import StanAvailability

    rng = np.random.default_rng(12)
    n_rows = 800
    features = ["gp_share_lag1"]
    frame = pd.DataFrame({"gp_share_lag1": rng.uniform(0.2, 1.0, n_rows)})
    frame["team_games"] = 82
    frame["gp"] = rng.binomial(82, np.clip(frame["gp_share_lag1"], 0.05, 0.95))

    stan = StanAvailability(features=features, warmup=400, samples=400, chains=2,
                            predictive_draws=200).fit(frame)
    plug_in = StanAvailability(features=features, pmf_mode="plug_in")
    plug_in.__dict__.update({k: v for k, v in stan.__dict__.items()
                             if k not in ("pmf_mode", "name")})

    sub = frame.head(50)
    grid = np.arange(83)
    mixed = stan.predict_pmf(sub, 82)
    fixed = plug_in.predict_pmf(sub, 82)
    assert np.allclose(mixed.sum(axis=1), 1.0, atol=1e-6)
    assert np.allclose(fixed.sum(axis=1), 1.0, atol=1e-6)
    # If these were identical, `stan_posterior` would be a mislabelled plug-in.
    assert np.abs(mixed - fixed).max() > 0

    mus, rhos = stan.mu_draws(sub)
    n = sub["team_games"].to_numpy(float)
    conditional = (n * mus * (1 - mus) * (1 + (n - 1) * rhos[:, None])).mean(axis=0)
    law = conditional + (n * mus).var(axis=0)

    mean = (mixed * grid).sum(axis=1)
    pmf_variance = (mixed * grid ** 2).sum(axis=1) - mean ** 2
    assert np.allclose(law, pmf_variance, atol=1e-6)


@needs_cmdstan
def test_shared_beta_induces_board_correlation_a_point_estimate_cannot():
    """The actual argument for fitting this in Stan.

    `Var_th(sum_i E[Y_i|th])` is exactly zero under any point estimate and does not shrink
    with the number of players, because a posterior draw moves every player the same way.
    That is the term a draft portfolio cares about, and no amount of marginal accuracy
    supplies it.
    """
    from src.models.stan_availability import StanAvailability, board_correlation

    rng = np.random.default_rng(14)
    n_rows = 600
    features = ["gp_share_lag1"]
    frame = pd.DataFrame({"gp_share_lag1": rng.uniform(0.2, 1.0, n_rows)})
    frame["team_games"] = 82
    frame["gp"] = rng.binomial(82, np.clip(frame["gp_share_lag1"], 0.05, 0.95))

    stan = StanAvailability(features=features, warmup=400, samples=400, chains=2,
                            predictive_draws=200).fit(frame)
    board = board_correlation(stan, frame, sizes=(15, 150, None), n_subsets=50)
    assert (board["shared_beta_sd"] > 0).all()
    assert (board["inflation"] > 1.0).all()

    # The scaling is the decision-relevant part: the independent term grows as sqrt(N) and
    # the shared-beta term as N, so the ratio grows with portfolio size. A 15-player roster
    # gets almost none of this; the whole board gets meaningfully more.
    ratio = board["shared_beta_sd"] / board["independent_sd"]
    assert ratio.is_monotonic_increasing
    assert board["inflation"].iloc[0] < board["inflation"].iloc[-1]


@needs_cmdstan
def test_negbinomial_head_recovers_a_known_rate_and_respects_exposure():
    """Exposure enters as an offset, so doubling minutes must double the expected count.
    If it were fitted as a covariate instead, this would come out at some other power."""
    from src.models.stan_components import StanCount

    rng = np.random.default_rng(13)
    n_rows = 1200
    minutes = rng.uniform(400, 2800, n_rows)
    log_own = rng.normal(1.5, 0.4, n_rows)
    rate = np.exp(-3.0 + 0.8 * log_own)                 # per minute
    y = rng.negative_binomial(60, 60 / (60 + rate * minutes))

    frame = pd.DataFrame({"log_own": log_own, "total_minutes": minutes, "reb": y})
    model = StanCount(["log_own"], "reb", warmup=400, samples=400, chains=2,
                      predictive_samples=200).fit(frame)

    assert model.diagnostics["divergences"] == 0
    doubled = frame.assign(total_minutes=frame["total_minutes"] * 2)
    assert np.allclose(model.predict_mean(doubled), 2 * model.predict_mean(frame),
                       rtol=1e-6)
    predicted = model.predict_mean(frame)
    assert np.corrcoef(predicted, y)[0, 1] > 0.8
