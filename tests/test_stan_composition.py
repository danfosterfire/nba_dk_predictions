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


def test_rho_bin_edges_come_from_train_only():
    """Same rule as the scaler and the spline knots. Reading the held-out
    distribution to place the bins would leak it into the design, invisibly — and
    here it would be especially quiet, because rho only affects the spread."""
    from src.models.stan_composition import RHO_BIN_COL, rho_bin_edges

    train = pd.DataFrame({RHO_BIN_COL: [0.1, 0.2, 0.3, 0.4]})
    test = pd.DataFrame({RHO_BIN_COL: [10.0, 20.0, 30.0, 40.0]})
    edges = rho_bin_edges(train, n_bins=2)
    pooled = rho_bin_edges(pd.concat([train, test]), n_bins=2)
    assert edges.tolist() == [0.25]
    assert edges.tolist() != pooled.tolist()


def test_rho_bins_are_one_based_and_clamp_outside_the_training_range():
    """A rookie whose imputed share sits below every training quantile is a fringe
    player — bin 1 — not an error and not a NaN, because the composition cannot drop
    a row without breaking the team sum."""
    from src.models.stan_composition import RHO_BIN_COL, assign_rho_bins

    frame = pd.DataFrame({RHO_BIN_COL: [-5.0, 0.15, 0.35, 99.0]})
    (out,) = assign_rho_bins([frame], np.array([0.2, 0.3]))
    assert out["rho_bin"].tolist() == [1, 1, 3, 3]
    assert out["rho_bin"].min() >= 1


def test_the_graded_variant_differs_from_its_twin_in_dispersion_alone():
    """`betabinom_ot_graded` exists to isolate the pilot's one measured
    miscalibration (variance ratio 1.59 fringe against 0.70 star). If it also
    differed in features, the contrast would be confounded and the answer
    uninterpretable."""
    from src.models.availability import FEATURE_COLS
    from src.models.stan_composition import OWN, RHO_BINS, variants

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                        for g in range(1, 6)])
    for col in list(FEATURE_COLS) + [OWN]:
        frame[col] = 1.0
    v = variants(frame, frame)
    shared_tr, _, shared_feats, shared_disp, shared_rho = v["betabinom_ot"]
    graded_tr, _, graded_feats, graded_disp, graded_rho = v["betabinom_ot_graded"]

    assert shared_feats == graded_feats
    assert shared_disp == graded_disp == 1
    assert (shared_rho, graded_rho) == (1, RHO_BINS)
    assert shared_tr["logit_prior"].equals(graded_tr["logit_prior"])


def test_a_single_rho_bin_reproduces_the_shared_rho_simulation_exactly():
    """n_rho = 1 must be the shared-rho model exactly, not approximately — that is
    what makes the graded arm a strict generalization and lets one code path serve
    both. Checked on the simulator, where a silent divergence would surface only as
    a slightly wrong spread.

    The frame carries a MULTI-valued `rho_bin`, because in the sweep it does: every
    variant gets bins assigned and only the model's `n_rho` says whether they are
    used. A shared-rho model handed that frame must ignore the column rather than
    index past its length-1 rho, which is a live bug this pins.
    """
    from src.models.stan_composition import simulate_minutes

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                        for g in range(1, 21)])
    eta = np.zeros((len(frame), 30))

    no_bins = simulate_minutes(frame, eta, np.full((30, 1), 0.08), seed=3)
    binned = frame.copy()
    binned["rho_bin"] = 1 + (binned["position"] % 4)
    one_bin = simulate_minutes(binned, eta, np.full((30, 1), 0.08), seed=3)
    assert np.array_equal(no_bins, one_bin)


def test_graded_rho_widens_only_the_bin_it_grades():
    """The mechanism itself: raising rho in one bin must move that bin's spread and
    leave the others alone, or the per-row gather is wrong."""
    from src.models.stan_composition import simulate_minutes

    frame = _sequenced([_team_rows([40, 38, 36, 34, 30, 25, 20, 17], game_id=g)
                        for g in range(1, 61)])
    frame["rho_bin"] = np.where(frame["position"] < 4, 1, 2)
    eta = np.zeros((len(frame), 120))

    flat = simulate_minutes(frame, eta, np.tile([0.02, 0.02], (120, 1)), seed=5)
    tilted = simulate_minutes(frame, eta, np.tile([0.02, 0.20], (120, 1)), seed=5)

    bin1 = (frame["rho_bin"] == 1).to_numpy()
    bin2 = (frame["rho_bin"] == 2).to_numpy()
    # Bin 2 is much wider; bin 1 moves only through the sequential coupling, which is
    # small relative to a tenfold dispersion change.
    assert tilted[:, bin2].std(axis=0).mean() > 1.5 * flat[:, bin2].std(axis=0).mean()
    assert tilted[:, bin1].std(axis=0).mean() < 1.2 * flat[:, bin1].std(axis=0).mean()


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

    tr, te, feats, _, _ = variants(frame, frame)["betabinom"]
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


def test_composition_source_grades_rho_by_bin_with_a_vectorized_gather():
    """The graded dispersion must stay inside ONE vectorized beta_binomial call —
    a per-row loop over ~100k rows would undo the sampling-cost work. Multiple
    indexing (`rho[rho_bin[lik]]`) is what keeps it vectorized."""
    code = _stan_code()
    assert "rho[rho_bin[lik]]" in code
    assert "vector<lower=1e-6, upper=0.95>[n_rho_par] rho;" in code
    assert "beta_binomial_lupmf(y[lik] | m[lik], s .* inv_logit(eta[lik])" in code


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


def test_the_ot_tail_no_longer_lives_in_this_module():
    """`fit_ot_tail` / `sample_game_length` moved to `src/models/stan_game_length.py`.

    Pinned as an absence, because the failure this guards is a *reintroduction*: the tail
    was parked here once and a second copy beside the real head is how the simulator ends
    up drawing game lengths from a point estimate again.
    `tests/test_stan_game_length.py` carries the same synthetic case against the floor that
    replaced it.
    """
    import src.models.stan_composition as C

    for gone in ("fit_ot_tail", "sample_game_length", "ot_tail_check"):
        assert not hasattr(C, gone), f"{gone} is back in stan_composition"


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


# ── The per-(player, season) random effect ────────────────────────────────────

def test_the_player_season_parameters_are_zero_length_when_U_n_is_zero():
    """`U_n = 0` must disable the effect EXACTLY, not approximately.

    Same bar as `betabinomial_glm`'s year block and the same reason: if the disabled
    parameter space is not literally identical, `base` is not the shipped head and every
    figure the incumbent's artifact carries is describing a different model.
    """
    code = _stan_code()
    assert "int H_u = U_n > 0 ? 1 : 0;" in code
    assert "vector[U_n] u_z;" in code
    assert "vector<lower=0>[H_u] sigma_u;" in code


def test_the_player_season_effect_is_non_centred():
    """Rows per unit run median 57 but p10 11 and minimum 1 — the sparse tail funnels
    under a centred parameterization, so non-centred is the default and divergences are
    the diagnostic that would send it back."""
    code = _stan_code()
    assert "u_z ~ std_normal();" in code
    assert "sigma_u[1] * u_z[unit_idx]" in code


def test_the_disabled_term_passes_a_block_stan_never_reads():
    """Stan has no optional data, so the disabled arm still ships the three keys — with
    `U_n = 0` and every index 0, which the model never dereferences."""
    from src.models.stan_composition import PlayerSeasonTerm

    frame = _team_rows([30, 25, 20, 15, 10, 5])
    block = PlayerSeasonTerm(False).data(frame)
    assert block["U_n"] == 0
    assert block["unit_idx"] == [0] * len(frame)


def test_the_enabled_term_indexes_player_season_and_not_player():
    """A career-long effect would be absorbed by `logit_share_lag1` and the offset, so
    the same player in two seasons must be two units."""
    from src.models.stan_composition import PlayerSeasonTerm

    frame = pd.DataFrame({"player_id": [1, 1, 2, 2], "season": ["2019-20", "2020-21"] * 2})
    block = PlayerSeasonTerm(True).data(frame)
    assert block["U_n"] == 4
    assert sorted(block["unit_idx"]) == [1, 2, 3, 4]


def test_the_shift_is_shared_within_a_unit_and_free_across_draws():
    """The whole mechanism. Per-game noise averages down by ~1/sqrt(G) when summed to a
    season; a shift shared across a player's games passes through in full, which is the
    4.68x `make minutes-unification` measured. A `z` drawn per ROW would reproduce the
    per-game marginal and buy none of the season-level spread."""
    from src.models.stan_composition import PlayerSeasonTerm

    term = PlayerSeasonTerm(True, stream="test")
    term.sigma_draws = np.full(4, 0.5)
    frame = pd.DataFrame({"player_id": [1, 1, 1, 2, 2], "season": "2020-21"})
    shift = term.shift(frame, np.arange(4))

    assert shift.shape == (5, 4)
    assert np.allclose(shift[0], shift[1]) and np.allclose(shift[1], shift[2])
    assert not np.allclose(shift[0], shift[3])
    assert not np.allclose(shift[:, 0], shift[:, 1])


def test_a_disabled_term_shifts_nothing_so_predict_samples_is_unchanged():
    """The nesting, on the Python side of the boundary: a head built without the effect
    must draw exactly what it drew before the parameter existed."""
    from src.models.stan_composition import StanComposition

    frame = _sequenced([_team_rows([30, 25, 20, 15, 10, 5], game_id=g)
                        for g in (1, 2, 3)])
    model = StanComposition(["w_share"], dispersed=1, n_rho=1, predictive_samples=8)
    model.scaler = _identity_scaler()
    model.alpha_draws = np.zeros(8)
    model.beta_draws = np.zeros((8, 1))
    model.rho_draws = np.full((8, 1), 0.1)
    model.rho_by_bin, model.rho = np.full(1, 0.1), 0.1

    assert not model.ps.enabled
    assert np.allclose(model.ps.shift(frame, np.arange(8)), 0.0)
    eta, _ = model._eta_base(frame)
    assert np.allclose(eta, 0.0)


def _identity_scaler():
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(np.zeros((2, 1)))
    scaler.scale_ = np.ones(1)
    scaler.mean_ = np.zeros(1)
    return scaler


def test_eta_base_excludes_the_fitted_effect_so_the_sweep_cannot_double_count_it():
    """`_eta_base` is the DETERMINISTIC predictor, and two consumers depend on that:
    `posteriors._finish` stores it as the round-trip reference (a fresh `z` per call
    would make the gate non-reproducible) and `player_season_effect_sweep` injects its
    own sigma on top."""
    from src.models.stan_composition import StanComposition

    frame = _sequenced([_team_rows([30, 25, 20, 15, 10, 5], game_id=g) for g in (1, 2)])
    model = StanComposition(["w_share"], dispersed=1, n_rho=1, predictive_samples=6,
                            player_season_effect=True)
    model.scaler = _identity_scaler()
    model.alpha_draws = np.zeros(6)
    model.beta_draws = np.zeros((6, 1))
    model.rho_draws = np.full((6, 1), 0.1)
    model.rho_by_bin, model.rho = np.full(1, 0.1), 0.1
    model.ps.sigma_draws = np.full(6, 0.4)

    eta, _ = model._eta_base(frame)
    assert np.allclose(eta, 0.0)
    assert np.abs(model.ps.shift(frame, model._draw_index())).max() > 0


# ── The team-context block ────────────────────────────────────────────────────

def _team_block(pairs, value=0.5):
    from src.models.stan_composition import TEAM_COLS

    block = pd.DataFrame(pairs, columns=["player_id", "season"])
    for i, col in enumerate(TEAM_COLS):
        block[col] = value + i
    return block


def test_the_team_block_gets_ONE_indicator_not_one_per_column():
    """Five per-column flags would be five exact copies of each other — the degenerate
    subspace `design_missing` already exists to avoid, one block over."""
    from src.models.stan_composition import TEAM_COLS, attach_team_context

    train = _sequenced([_team_rows([30, 25, 20, 15, 10, 5], game_id=1)])
    train["design_missing"] = 0.0
    val = train.copy()
    block = _team_block([(pid, "2020-21") for pid in train["player_id"].iloc[:4]])

    tr, te, feats, coverage = attach_team_context(train, val, block)
    assert feats == list(TEAM_COLS) + ["team_missing"]
    assert not [c for c in tr.columns if c.endswith("__miss")]
    assert tr["team_missing"].sum() == 2
    assert coverage["team_share_train"] == pytest.approx(4 / 6)


def test_the_team_block_imputes_from_train_means_only():
    """The scaler rule, applied to the join: reading the validation mean to fill a
    validation hole would leak the held-out distribution into the design invisibly."""
    from src.models.stan_composition import TEAM_COLS, attach_team_context

    train = _sequenced([_team_rows([30, 25, 20, 15, 10, 5], game_id=1)])
    train["design_missing"] = 0.0
    val = _sequenced([_team_rows([30, 25, 20, 15, 10, 5], game_id=2, season="2022-23")])
    val["design_missing"] = 0.0

    block = pd.concat([
        _team_block([(pid, "2020-21") for pid in train["player_id"]], value=1.0),
        _team_block([(pid, "2022-23") for pid in val["player_id"].iloc[:1]], value=99.0),
    ], ignore_index=True)
    # Drop the val rows' block for everyone but one, so the rest are imputed.
    tr, te, _, _ = attach_team_context(train, val, block)
    filled = te.loc[te["team_missing"] == 1, TEAM_COLS[0]].to_numpy(float)
    assert len(filled) and np.allclose(filled, 1.0)


def test_effect_variants_build_four_arms_over_the_shipped_specification():
    """`base` is the shipped arm on this window, not a refit of the incumbent; `ps` and
    `base` must differ in the EFFECT alone and `team` and `base` in the FEATURES alone,
    or neither contrast isolates what it claims to."""
    from src.models.stan_composition import TEAM_COLS, effect_variants

    frames = [_team_rows([30, 25, 20, 15, 10, 5], game_id=g,
                         season="2020-21" if g < 3 else "2022-23") for g in (1, 2, 3)]
    frame = _sequenced(frames)
    from src.models.availability import FEATURE_COLS
    for col in list(FEATURE_COLS) + ["logit_share_lag1"]:
        frame[col] = 0.5
    train = frame[frame["season"] == "2020-21"].reset_index(drop=True)
    val = frame[frame["season"] == "2022-23"].reset_index(drop=True)
    block = _team_block([(pid, "2020-21") for pid in train["player_id"]])

    from src.models import stan_composition as C
    built, _ = effect_variants(train, val, block)
    assert set(built) == {"base", "ps", "team", "ps_team", "ps_centered"}
    assert built["base"][2] == built["ps"][2]
    assert built["base"][5] is False and built["ps"][5] is True
    assert built["team"][2] == built["base"][2] + list(TEAM_COLS) + [C.TEAM_MISSING]
    assert built["team"][5] is False and built["ps_team"][5] is True
    # `ps_centered` is the SAME model in different coordinates — a sampler arm, not a
    # modelling one — so it must differ from `ps` in the parameterization flag alone.
    assert built["ps_centered"][:6] == built["ps"][:6]
    assert built["ps"][6] is False and built["ps_centered"][6] is True
    for arm, (_, _, feats, _, _, _, _) in built.items():
        assert len(feats) == len(set(feats)), f"{arm} carries a duplicate feature"


@needs_cmdstan
def test_U_n_zero_nests_exactly_inside_the_player_season_model():
    """Gate P1, as an identity rather than an assertion about source text.

    At `sigma_u = 0` the effect model's log density must exceed the `U_n = 0` model's by
    *exactly* the std_normal prior on `z` and nothing else — meaning the likelihood, the
    offset, the priors on alpha/beta and the dispersion term are untouched. Stan's `~`
    drops constants, so the check is on the difference, which is constant-free.

    This is the only thing separating "a parameter was added" from "the shipped head was
    silently changed", and the incumbent's whole artifact depends on it.
    """
    rng = np.random.default_rng(0)
    G, K, U, N = 6, 2, 48, 240
    lens = np.full(G, 6)
    P = int(lens.sum())
    y, m, lo, is_last, start = [], [], [], [], []
    row = 1
    for _ in range(G):
        remaining = N
        for j in range(6):
            trials = min(U, remaining)
            last, after = j == 5, 5 - j
            bound = max(0, remaining - after * U)
            value = remaining if last else int(
                min(trials, max(bound, rng.integers(20, 50))))
            y.append(value)
            m.append(trials)
            lo.append(bound)
            is_last.append(int(last))
            remaining -= value
        start.append(row)
        row += 6

    data = {"G": G, "P": P, "K": K, "start": start, "len": lens.tolist(), "y": y,
            "m": m, "lo": lo, "is_last": is_last, "N_total": [N] * G, "U": [U] * G,
            "logit_prior": np.zeros(P).tolist(), "X": rng.normal(size=(P, K)),
            "dispersed": 1, "n_rho": 1, "rho_bin": [1] * P, "beta_scale": 1.0,
            "intercept_scale": 5.0}
    pars = {"alpha": 0.1, "beta": [0.3, -0.2], "rho": [0.08]}
    z = [0.7, -0.3, 1.1, 0.2, -0.9]

    model = stan_utils.compile_model("composition_glm")
    # The non-centred branch, which is the one the nesting claim is about: the centred
    # form's prior at `sigma_u = 0` is `normal(0, 0)` and has no density to compare.
    off = {**data, "U_n": 0, "unit_idx": [0] * P, "u_sd_scale": 1.0, "u_centered": 0}
    on = {**data, "U_n": 5, "unit_idx": ((np.arange(P) % 5) + 1).tolist(),
          "u_sd_scale": 1.0, "u_centered": 0}

    def lp(params, payload):
        return float(model.log_prob(params, data=payload, jacobian=False).iloc[0, 0])

    gap = lp({**pars, "u_z": z, "sigma_u": [0.0]}, on) - lp(pars, off)
    assert gap == pytest.approx(-0.5 * float(np.sum(np.square(z))), abs=1e-4)


def test_the_centred_arm_is_the_same_model_in_different_coordinates():
    """`ps_centered` must be a SAMPLER arm and nothing else.

    Two things follow and both are pinned. The Stan source has to carry both branches with
    the same `sigma_u` prior, and the Python term has to shift IDENTICALLY either way — the
    predictive of a player-season that has not happened is `sigma_u * z` under both
    parameterizations, so a `centered` flag leaking into `shift` would make the two arms
    genuinely different models and their CRPS comparison meaningless.
    """
    from src.models.stan_composition import PlayerSeasonTerm

    code = _stan_code()
    assert "int<lower=0, upper=1> u_centered;" in code
    assert "u_z ~ normal(0, sigma_u[1]);" in code
    assert "u_z ~ std_normal();" in code

    frame = pd.DataFrame({"player_id": [1, 1, 2], "season": "2020-21"})
    shifts = []
    for centered in (False, True):
        term = PlayerSeasonTerm(True, stream="same", centered=centered)
        term.sigma_draws = np.full(3, 0.4)
        shifts.append(term.shift(frame, np.arange(3)))
        assert term.data(frame)["u_centered"] == int(centered)
    np.testing.assert_allclose(shifts[0], shifts[1])


def test_the_dense_metric_needs_the_matrix_to_be_ESTIMABLE_not_merely_to_fit():
    """Two constraints, and the binding one is not memory.

    A dense metric estimates a P x P covariance from the WARMUP draws, so it needs draws on
    the order of the parameter count. At 605 units and 1,000 warmup that is 1.57 draws per
    parameter — rank-deficient, shrunk back toward diagonal by CmdStan, and measured to run
    past an hour on rows `diag_e` finished in 25 minutes. Memory alone would have waved all
    but the full window through, which is exactly the mistake this pins.
    """
    from src.models.stan_composition import (DENSE_DRAWS_PER_PARAM,
                                             DENSE_METRIC_MAX_MB, choose_metric)

    # The effect-free head is the regime `dense_e` was measured to help in.
    assert choose_metric(26, warmup=1000) == "dense_e"
    # Every random-effect arm fails the estimability test, at every window.
    for units in (605, 2204, 12307):
        assert choose_metric(units + 30, warmup=1000) == "diag_e"
    # And more warmup does not rescue it at any realistic budget.
    assert choose_metric(635, warmup=5000) == "diag_e"

    # The estimability boundary is the one that moves first.
    edge = int(1000 / DENSE_DRAWS_PER_PARAM)
    assert choose_metric(edge, warmup=1000) == "dense_e"
    assert choose_metric(edge + 1, warmup=1000) == "diag_e"
    # Memory still vetoes independently, however many draws are available.
    huge = int((DENSE_METRIC_MAX_MB * 1024 ** 2 / 8) ** 0.5) + 1
    assert choose_metric(huge, warmup=10 ** 9) == "diag_e"


def test_an_explicit_metric_overrides_the_sizing():
    """A probe comparing the two metrics on identical data needs to force one; the
    default stays `None` so ordinary callers get the sized choice."""
    from src.models.stan_composition import StanComposition

    assert StanComposition(["x"], dispersed=1).metric is None
    assert StanComposition(["x"], dispersed=1, metric="dense_e").metric == "dense_e"
