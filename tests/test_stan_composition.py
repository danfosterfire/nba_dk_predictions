"""Tests for the team-game minutes composition head.

Most of what follows needs no sampler: the frame construction (rounding, ordering,
sequential bounds, offsets), the simulator's hard guarantees, the selection logic and
the OT tail are all pure Python. The multinomial-anchor test pins the identity the
whole model rests on — with caps off, the sequential binomial decomposition IS the
multinomial. Two tests read the Stan source, because the demo this model replaces had
its cap declared but never enforced, and that class of bug is invisible at runtime.
The handful that genuinely need CmdStan are skipped rather than failed.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import stan_utils


def _has_cmdstan() -> bool:
    try:
        import cmdstanpy
        cmdstanpy.cmdstan_path()
        return True
    except Exception:
        return False


needs_cmdstan = pytest.mark.skipif(
    not _has_cmdstan(), reason="CmdStan is not installed; run "
    ".venv/bin/python -c 'import cmdstanpy; cmdstanpy.install_cmdstan()'")


# ── Synthetic builders ────────────────────────────────────────────────────────

def _team_rows(mins, game_id=1, team_id=10, season="2020-21", length=48.0,
               w=None, n_overtimes=0):
    """One team-game block, already in stick-breaking order."""
    k = len(mins)
    w = list(w) if w is not None else list(np.linspace(0.8, 0.1, k))
    return pd.DataFrame({
        "season": season, "game_id": game_id, "team_id": team_id,
        "player_id": np.arange(1, k + 1) + 100 * game_id + 1000 * team_id,
        "min": list(mins),
        "game_length": length,
        "n_overtimes": n_overtimes,
        "N": int(round(5 * length)), "U": int(round(length)),
        "w_share": w, "no_prior": 0.0, "share_stale": 0.0,
        "draft_number": np.nan,
        "position": np.arange(k),
    })


def _sequenced(frames):
    """Rounded + sequential columns, via the real code path."""
    from src.models.stan_composition import largest_remainder, sequential_columns

    frame = pd.concat(frames, ignore_index=True)
    frame["y"] = largest_remainder(frame)
    return sequential_columns(frame)


# ── Rounding ──────────────────────────────────────────────────────────────────

def test_largest_remainder_rounding_preserves_the_team_sum_exactly():
    """The multinomial needs integers summing to exactly N; minutes are recorded to
    the second. The deficit goes to the largest fractional remainders."""
    from src.models.stan_composition import largest_remainder

    # Exact binary fractions, so the remainder ordering is not decided by float dust.
    mins = [40.5, 38.25, 35.75, 30.0, 28.5, 25.75, 17.0, 13.75, 10.5]
    frame = _team_rows(mins)
    y = largest_remainder(frame)
    assert y.sum() == 240
    # Deficit of 4 goes to the largest remainders: the three .75s, then the first .5.
    assert y.tolist() == [41, 38, 36, 30, 28, 26, 17, 14, 10]


def test_rounding_handles_a_negative_deficit_by_taking_from_small_remainders():
    """A handful of real games have box-score minutes summing PAST 5 x game_length
    (worst observed 3.08 minutes); the correction must remove, not only add."""
    from src.models.stan_composition import largest_remainder

    mins = [48.0, 48.0, 48.0, 48.0, 25.1, 23.2]  # sums to 240.3
    frame = _team_rows(mins)
    y = largest_remainder(frame)
    assert y.sum() == 240
    assert y.tolist() == [48, 48, 48, 48, 25, 23]


def test_rounding_never_lifts_a_player_above_the_cap():
    """A player at exactly the cap must not receive the deficit — it would create a
    row the beta-binomial support cannot hold, the `gp > team_games` trap again."""
    from src.models.stan_composition import largest_remainder

    mins = [48.0, 40.5, 40.5, 40.5, 35.5, 35.0]  # deficit lands somewhere; not on #1
    frame = _team_rows(mins)
    y = largest_remainder(frame)
    assert y.sum() == 240
    assert y.max() == 48
    assert y[0] == 48


# ── Ordering ──────────────────────────────────────────────────────────────────

def test_ordering_puts_prior_share_first_and_rookies_last_with_draft_tiebreak():
    from src.models.stan_composition import order_frame

    frame = pd.DataFrame({
        "season": "2020-21", "game_id": 1, "team_id": 10,
        "player_id": [1, 2, 3, 4, 5],
        "w_share": [0.2, 0.7, 0.15, 0.15, 0.4],
        "no_prior": [0.0, 0.0, 1.0, 1.0, 0.0],
        "draft_number": [np.nan, np.nan, 40.0, 3.0, np.nan],
    })
    out = order_frame(frame)
    # Veterans by share (2, 5, 1), then rookies by draft slot (4 before 3).
    assert out["player_id"].tolist() == [2, 5, 1, 4, 3]
    assert out["position"].tolist() == [0, 1, 2, 3, 4]


# ── Sequential columns and the bounds ─────────────────────────────────────────

def test_sequential_columns_make_the_last_player_feasible_by_construction():
    """lo = max(0, R - J*U) is what guarantees the deterministic last step lands in
    [0, U] — the induction the Stan header states."""
    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17])])
    last = frame[frame["is_last"] == 1].iloc[0]
    assert last["y"] == last["R"]
    assert last["y"] <= last["U"]
    live = frame[frame["is_last"] == 0]
    assert (live["y"] <= live["m"]).all()
    assert (live["y"] >= live["lo"]).all()


def test_the_feasibility_lower_bound_binds_in_a_short_rotation_game():
    """K = 6 with N = 240 is the real case (10 of 71,092 team-games): after four
    players the remaining two must absorb what is left, so lo can be positive."""
    frame = _sequenced([_team_rows([47, 46, 45, 44, 30, 28])])
    assert (frame["lo"] > 0).any()


def test_trials_are_remaining_capacity_never_more_than_the_cap():
    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17])])
    assert (frame["m"] <= frame["U"]).all()
    assert (frame["m"] <= frame["R"]).all()
    # Early steps: more remaining than any one player may take.
    assert frame.iloc[0]["m"] == frame.iloc[0]["U"]


def test_stick_offsets_reduce_to_the_multinomial_when_caps_cannot_bind():
    """THE anchor: with U >= N the trials are always the full remainder and the
    sequential binomial decomposition with the carry-forward offsets is EXACTLY the
    multinomial over renormalized prior shares. If this breaks, the offset
    construction no longer matches the model the demo piloted."""
    from scipy.stats import binom, multinomial

    rows = _team_rows([120, 60, 40, 20], length=48.0, w=[0.5, 0.25, 0.15, 0.10])
    rows["U"] = 240   # caps off: U = N
    frame = _sequenced([rows])

    live = frame[frame["is_last"] == 0]
    p = 1 / (1 + np.exp(-live["logit_prior"].to_numpy(float)))
    sequential = binom.logpmf(live["y"].to_numpy(int), live["R"].to_numpy(int),
                              p).sum()
    w = frame["w_share"].to_numpy(float)
    joint = multinomial.logpmf(frame["y"].to_numpy(int), 240, w / w.sum())
    assert abs(sequential - joint) < 1e-9


# ── The simulator's hard guarantees ───────────────────────────────────────────

def test_simulator_draws_respect_cap_and_team_sum_on_every_draw():
    from src.models.stan_composition import simulate_minutes

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                        for g in range(1, 41)])
    eta = np.zeros((len(frame), 50))
    draws = simulate_minutes(frame, eta, np.full(50, 0.08), seed=7)
    assert draws.shape == (50, len(frame))
    assert (draws <= frame["U"].to_numpy()[None, :]).all()
    sums = draws.reshape(50, 40, 8).sum(axis=2)
    assert (sums == 240).all()


def test_binomial_draws_are_visibly_tighter_than_beta_binomial_at_the_same_mean():
    """The mechanism the ladder tests: the measured game-level dispersion is 4.65x
    binomial, so the pure decomposition should be much too tight."""
    from src.models.stan_composition import simulate_minutes

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                        for g in range(1, 41)])
    eta = np.zeros((len(frame), 200))
    tight = simulate_minutes(frame, eta, None, seed=7)
    wide = simulate_minutes(frame, eta, np.full(200, 0.10), seed=7)
    assert wide.std(axis=0).mean() > 1.5 * tight.std(axis=0).mean()


def test_floor_dispersion_fit_recovers_a_known_rho_on_synthetic_data():
    from src.models.stan_composition import (FloorComposition, sequential_columns,
                                             simulate_minutes)

    base = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                       for g in range(1, 301)])
    draw = simulate_minutes(base, np.zeros((len(base), 1)),
                            np.array([0.08]), seed=3)[0]
    observed = base.copy()
    observed["y"] = draw.astype(int)
    observed = sequential_columns(observed)
    rho = FloorComposition().fit(observed).rho
    assert 0.04 < rho < 0.16


# ── Selection logic ───────────────────────────────────────────────────────────

def test_selection_reads_validation_and_ignores_a_better_test_column():
    """The repo has already shipped one false positive selected on test."""
    from src.models.stan_composition import mark_selection

    table = pd.DataFrame([
        {"variant": "carry_forward", "val_crps": 5.0, "test_crps": 5.1,
         "val_r2": 0.5, "test_r2": 0.50},
        {"variant": "binomial", "val_crps": 4.9, "test_crps": 4.0,
         "val_r2": 0.55, "test_r2": 0.60},
        {"variant": "betabinom", "val_crps": 4.5, "test_crps": 4.6,
         "val_r2": 0.56, "test_r2": 0.55},
        {"variant": "independent_comparator", "val_crps": 4.4, "test_crps": 4.4,
         "val_r2": 0.57, "test_r2": 0.57},
    ])
    out = mark_selection(table)
    assert out.loc[out["selected"], "variant"].tolist() == ["betabinom"]
    # The comparator is the incumbent, never selectable, even at the best val CRPS.
    assert not out.loc[out["variant"] == "independent_comparator", "selected"].iloc[0]
    assert out.loc[out["variant"] == "betabinom", "beats_floor"].iloc[0]


# ── Frame construction guards ─────────────────────────────────────────────────

def test_played_frame_raises_rather_than_drop_an_unmatched_game_length():
    """A missing length is a missing denominator; dropping it silently would shrink
    the fitting frame — the same loud-not-dropped rule as `minutes_targets`."""
    from src.models.stan_composition import played_frame

    panel = pd.DataFrame({
        "season": ["2020-21"] * 2, "game_id": [1, 2], "team_id": [10, 10],
        "player_id": [1, 1], "min": [30.0, 31.0], "played": [1, 1]})
    lengths = pd.DataFrame({
        "season": ["2020-21"], "game_id": [1], "season_type": ["regular"],
        "game_length": [48.0], "n_overtimes": [0], "reliable": [True]})
    try:
        played_frame(panel, lengths)
        assert False, "expected a ValueError for the unmatched game"
    except ValueError as e:
        assert "no game" in str(e)


def test_no_prior_players_are_flagged_not_dropped():
    """The composition cannot drop anyone — the sum must be complete. share_lags
    must keep a first-season player, flag him, and leave his share to imputation."""
    from src.models.stan_composition import share_lags

    seasons = ["2019-20", "2020-21"]
    shares = pd.DataFrame({
        "player_id": [1, 1, 2],
        "season": ["2019-20", "2020-21", "2020-21"],
        "minutes_share": [0.6, 0.55, 0.3],
    })
    out = share_lags(shares, seasons)
    rookie = out[(out["player_id"] == 2) & (out["season"] == "2020-21")].iloc[0]
    veteran = out[(out["player_id"] == 1) & (out["season"] == "2020-21")].iloc[0]
    assert rookie["no_prior"] == 1.0 and np.isnan(rookie["prior_share"])
    assert veteran["no_prior"] == 0.0 and veteran["prior_share"] == 0.6


def test_rookie_share_prior_is_point_in_time_and_skips_the_degenerate_first_season():
    """Season S's prior averages seasons strictly BEFORE S. Season index 0 is
    excluded from the observations: with no earlier data everyone there looks like a
    rookie, and folding veterans in would inflate every bucket mean."""
    from src.models.stan_composition import rookie_share_priors

    seasons = ["2018-19", "2019-20", "2020-21"]
    lagged = pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "season": ["2018-19", "2019-20", "2019-20", "2020-21"],
        "season_index": [0, 1, 1, 2],
        "minutes_share": [0.9, 0.4, 0.2, 0.6],
        "no_prior": [1.0, 1.0, 1.0, 1.0],
        "draft_bucket": ["lottery", "lottery", "lottery", "lottery"],
    })
    priors = rookie_share_priors(lagged, seasons)
    p2020 = priors[(priors["season"] == "2020-21")
                   & (priors["draft_bucket"] == "lottery")].iloc[0]
    # Only the two 2019-20 rookies (0.4, 0.2) count: the 0.9 season-0 veteran-lookalike
    # and the 0.6 row from 2020-21 itself must both be excluded.
    assert abs(p2020["rookie_share_prior"] - 0.3) < 1e-12


def test_log_tail_mass_recurrence_matches_scipy_and_survives_tiny_tails():
    """The Stan file computes log P(Y >= lo) by an upward log-space pmf recurrence.
    Two failure modes force that exact shape: beta_binomial_lcdf's grad_F32 autodiff
    made a single gradient cost seconds, and `1 - head_mass` rounds to log1m(1) =
    -inf when the tail is tiny — which killed every chain of the binomial arm's
    first full-data fit at its (finite-target) init. This replicates the recurrence
    line-for-line against scipy, including a tail small enough that the head-mass
    form returns -inf."""
    from scipy.stats import betabinom, binom

    def bb_log_tail(lo, n, a, b):
        log_term = betabinom.logpmf(lo, n, a, b)
        log_total = log_term
        for k in range(lo, n):
            log_term += np.log(((n - k) * (a + k)) / ((k + 1.0) * (b + n - k - 1.0)))
            log_total = np.logaddexp(log_total, log_term)
        return log_total

    def binom_log_tail(lo, n, p):
        log_term = binom.logpmf(lo, n, p)
        log_total = log_term
        for k in range(lo, n):
            log_term += np.log(((n - k) * p) / ((k + 1.0) * (1 - p)))
            log_total = np.logaddexp(log_total, log_term)
        return log_total

    for lo, n, a, b in [(1, 48, 3.0, 7.0), (6, 48, 0.5, 12.0), (21, 53, 19.0, 1.3),
                        (47, 48, 2.0, 2.0), (48, 48, 5.0, 5.0)]:
        expected = np.log(betabinom.sf(lo - 1, n, a, b))
        assert abs(bb_log_tail(lo, n, a, b) - expected) < 1e-8
    for lo, n, p in [(1, 48, 0.3), (8, 48, 0.85), (31, 53, 0.5)]:
        expected = np.log(binom.sf(lo - 1, n, p))
        assert abs(binom_log_tail(lo, n, p) - expected) < 1e-8
    # The case that killed the run: a binomial tail so small the head mass is
    # exactly 1.0 in double precision, where the log-tail form stays finite.
    lo, n, p = 40, 48, 0.1
    assert binom.cdf(lo - 1, n, p) == 1.0
    lt = binom_log_tail(lo, n, p)
    assert np.isfinite(lt)
    assert abs(lt - binom.logsf(lo - 1, n, p)) < 1e-6


def test_variants_carry_no_duplicate_feature_columns():
    """A rookie loses every design column at once, so per-column imputation flags
    are 18 copies of each other — C(18,2) = 153 duplicate standardized columns,
    which the sampler pays for in treedepth. The ladder must carry ONE
    design_missing indicator instead."""
    from src.models.availability import FEATURE_COLS
    from src.models.stan_composition import OWN, variants

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17])])
    rng = np.random.default_rng(5)
    for col in FEATURE_COLS:
        # Distinct values per column, missing TOGETHER for the first three rows —
        # the real pattern: a rookie loses the whole design block at once.
        frame[col] = np.where(frame["position"] < 3, np.nan, rng.normal(size=len(frame)))
    frame[OWN] = rng.normal(size=len(frame))
    frame["no_prior"] = (frame["position"] < 2).astype(float)
    frame["share_stale"] = (frame["position"] == 5).astype(float)

    tr, te, feats, _ = variants(frame, frame)["betabinom"]
    assert "design_missing" in feats
    X = tr[feats].to_numpy(dtype=float)
    dup = sum(np.allclose(X[:, i], X[:, j])
              for i in range(X.shape[1]) for j in range(i + 1, X.shape[1]))
    assert dup == 0


# ── The Stan source ───────────────────────────────────────────────────────────

def _stan_code() -> str:
    """Source with `//` comments stripped — matching prose would make these tests
    pass or fail on the documentation."""
    source = (stan_utils.STAN_DIR / "composition_glm.stan").read_text()
    return "\n".join(line.split("//")[0] for line in source.splitlines())


def test_composition_source_uses_the_numerically_safe_complement():
    code = _stan_code()
    assert "inv_logit(-eta" in code
    assert "1 - inv_logit" not in code


def test_composition_source_enforces_the_sum_and_cap_as_rejects():
    """The demo declared U and never enforced it, and compared an int to a
    probability. The real model must reject bad data loudly in transformed data and
    carry the cap in the TRIALS."""
    code = _stan_code()
    assert code.count("reject(") >= 3
    assert "total != N_total[g]" in code
    assert "y[r] > m[r]" in code
    assert "y[r] < lo[r]" in code
    assert "beta_binomial_lupmf(y[lik] | m[lik]" in code


def test_ot_tail_fits_and_samples_on_the_length_grid():
    from src.models.stan_composition import fit_ot_tail, sample_game_length

    lengths = pd.DataFrame({
        "season": ["2018-19"] * 100,
        "season_type": ["regular"] * 100,
        "n_overtimes": [0] * 94 + [1] * 5 + [2] * 1,
    })
    params = fit_ot_tail(lengths, {"2018-19"})
    assert abs(params["p_any_ot"] - 0.06) < 1e-12
    assert abs(params["p_more_ot"] - 1 / 6) < 1e-12
    drawn = sample_game_length(np.random.default_rng(0), 5000,
                               params["p_any_ot"], params["p_more_ot"])
    assert set(np.unique(drawn)) <= {48, 53, 58, 63, 68, 73}
    assert abs((drawn > 48).mean() - 0.06) < 0.02


# ── Sampler-backed ────────────────────────────────────────────────────────────

@needs_cmdstan
def test_composition_head_recovers_the_floor_on_data_generated_from_it():
    """Data generated at alpha = beta = 0 with a known rho: the fitted head should
    land near zero coefficients and near the true dispersion, with 0 divergences —
    the smallest end-to-end check that the vectorized likelihood, the offset and the
    trials convention agree with the simulator that generated the data."""
    from src.models.stan_composition import (StanComposition, sequential_columns,
                                             simulate_minutes)

    base = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                       for g in range(1, 201)])
    draw = simulate_minutes(base, np.zeros((len(base), 1)),
                            np.array([0.08]), seed=11)[0]
    observed = base.copy()
    observed["y"] = draw.astype(int)
    observed = sequential_columns(observed)
    observed["dummy"] = 0.0

    model = StanComposition(["dummy"], dispersed=1, name="test/synthetic",
                            chains=2, warmup=300, samples=300,
                            predictive_samples=50).fit(observed)
    assert model.diagnostics["divergences"] == 0
    assert abs(float(model.alpha_draws.mean())) < 0.15
    assert 0.05 < model.rho < 0.12

    samples = model.predict_samples(observed, seed=1)
    assert (samples.reshape(50, 200, 8).sum(axis=2) == 240).all()


@needs_cmdstan
def test_composition_head_samples_cleanly_when_the_feasibility_bound_binds():
    """Short-rotation teams put lo > 0 rows into the likelihood, which exercises the
    head-mass truncation function inside the gradient — the code path whose
    beta_binomial_lccdf predecessor cost seconds per gradient."""
    from src.models.stan_composition import StanComposition, sequential_columns

    frames = [_team_rows([47, 46, 45, 44, 30, 28], game_id=g) for g in range(1, 61)]
    observed = _sequenced(frames)
    assert (observed.loc[observed["is_last"] == 0, "lo"] > 0).any()
    observed["dummy"] = 0.0

    model = StanComposition(["dummy"], dispersed=1, name="test/binding-lo",
                            chains=2, warmup=200, samples=200,
                            predictive_samples=20).fit(observed)
    assert model.diagnostics["divergences"] == 0
    assert np.isfinite(model.diagnostics["max_rhat"])
