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
                           "fg2a_p36_lag1": [8.0, 6.0], "fg3a_p36_lag1": [2.0, 4.0],
                           "fg2a_lag1": [480.0, 310.0], "fg3a_lag1": [190.0, 90.0]})
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
                           "fg2a_p36_lag1": [0.0], "fg3a_p36_lag1": [0.0],
                           "fg2a_lag1": [0.0], "fg3a_lag1": [0.0]})
    assert np.isnan(add_substitution_columns(design)["fg3a_share_lag1"].iloc[0])


def test_substitution_columns_carry_the_raw_lag_counts_the_conversion_floor_needs():
    """`carry_forward_conversion` reads `{made}_lag1` / `{attempted}_lag1`, not per-36
    rates, so `fga_lag1` has to exist or the `fg3a | fga` floor cannot be computed at
    all — and a head with no floor is a head this repo will not accept."""
    from src.models.stan_components import add_substitution_columns

    design = pd.DataFrame({"fg2a": [500], "fg3a": [200],
                           "fg2a_p36_lag1": [8.0], "fg3a_p36_lag1": [2.0],
                           "fg2a_lag1": [480.0], "fg3a_lag1": [190.0]})
    assert add_substitution_columns(design)["fga_lag1"].tolist() == [670.0]


def _conversion_frame():
    rng = np.random.default_rng(4)
    n = 60
    return pd.DataFrame({
        "fg3a_share_lag1": rng.uniform(0.05, 0.7, n),
        "fg3a_pct_lag1": rng.uniform(0.2, 0.5, n),
        "fga_p36_lag1": rng.uniform(8, 20, n),
        "mpg_lag1": rng.uniform(10, 36, n), "total_minutes_lag1": rng.uniform(500, 2500, n),
        "gp_lag1": rng.uniform(40, 82, n), "age": rng.uniform(20, 36, n),
        "age_sq": rng.uniform(400, 1300, n), "career_year": rng.integers(0, 15, n),
    })


def test_conversion_variants_takes_the_own_column_explicitly():
    """The `{made}_pct_lag1` convention does not generalize to the `fg3a | fga` head:
    its own rate is the attempt-MIX share, while `fg3a_pct_lag1` would be three-point
    SHOOTING percentage. Both columns are present here, so a silent fallback to the
    convention would be invisible — which is exactly the failure being guarded."""
    from src.models.stan_components import conversion_variants

    frame = _conversion_frame()
    out = conversion_variants(frame, frame, "fg3a", "fga", own="fg3a_share_lag1")
    features = out["logit_own"][2]
    assert "logit_fg3a_share_lag1" in features
    assert "logit_fg3a_pct_lag1" not in features
    # The own column is replaced by its logit, never carried alongside it.
    assert "fg3a_share_lag1" not in features


def test_conversion_variants_defaults_to_the_made_pct_convention():
    from src.models.stan_components import conversion_variants

    frame = _conversion_frame().rename(columns={"fga_p36_lag1": "fg3a_p36_lag1"})
    out = conversion_variants(frame, frame, "fg3a", "fg3a")
    assert "logit_fg3a_pct_lag1" in out["logit_own"][2]


def test_gate0_arm_b_selects_its_two_factors_independently():
    """Additive separability is not a shortcut: the joint NLL is a SUM of two terms with
    no shared parameters, so minimising it is exactly minimising each term. This pins
    that `_g0_select` picks per factor and that the joint row it marks is the pair of
    per-factor winners — 3+3 fits rather than 9 combinations."""
    from src.models.stan_components import _g0_select

    head_rows = [
        {"analysis": "head", "split": "val", "arm": "fga_x_fg3a_share", "head": "fga",
         "variant": v, "mean_nll": nll, "selected": None}
        for v, nll in (("linear", 5.4), ("log_own", 5.1), ("log_own_spline", 5.2))]
    head_rows += [
        {"analysis": "head", "split": "val", "arm": "fga_x_fg3a_share",
         "head": "fg3a|fga", "variant": v, "mean_nll": nll, "selected": None}
        for v, nll in (("linear", 4.9), ("logit_own", 4.6), ("logit_own_spline", 4.4))]
    head_rows.append({"analysis": "head", "split": "val", "arm": "two_counts",
                      "head": "fg2a", "variant": "log_own", "mean_nll": 5.2,
                      "selected": True})
    joint = [{"analysis": "joint", "split": "val", "arm": "two_counts",
              "head": "fg2a+fg3a", "variant": "selected", "mean_nll": 10.4,
              "selected": None}]
    joint += [{"analysis": "joint", "split": "val", "arm": "fga_x_fg3a_share",
               "head": "fga+fg3a|fga", "variant": f"{a}+{b}",
               "mean_nll": 0.0, "selected": None}
              for a in ("linear", "log_own", "log_own_spline")
              for b in ("linear", "logit_own", "logit_own_spline")]

    out = _g0_select(pd.DataFrame(head_rows + joint))
    picked = out[(out["analysis"] == "head") & out["selected"]
                 & (out["arm"] == "fga_x_fg3a_share")]
    assert sorted(picked["variant"]) == ["log_own", "logit_own_spline"]
    chosen_joint = out[(out["analysis"] == "joint") & out["selected"]
                       & (out["arm"] == "fga_x_fg3a_share")]
    assert chosen_joint["variant"].tolist() == ["log_own+logit_own_spline"]


def test_gate0_arm_a_grid_is_the_best_of_sixteen_from_the_artifact(tmp_path):
    """Arm A's most favourable configuration costs zero fits — it is already on disk.
    Quoting it turns "we removed the handicap" into "arm B wins even against arm A's
    best", which is the version a reviewer cannot argue with."""
    from src.models.stan_components import _arm_a_grid

    pd.DataFrame([{"head": h, "variant": v, "test_nll": nll}
                  for h, rows in (("fg2a", [("linear", 5.4), ("log_own", 5.2)]),
                                  ("fg3a", [("linear", 5.9), ("log_own", 5.3)]))
                  for v, nll in rows]).to_csv(
        tmp_path / "stan_component_metrics.csv", index=False)

    grid = _arm_a_grid(tmp_path)
    assert len(grid) == 4
    best = grid[grid["selected"]]
    assert len(best) == 1
    assert best["variant"].iloc[0] == "log_own+log_own"
    assert best["mean_nll"].iloc[0] == 10.5


def test_gate0_arm_a_grid_is_empty_rather_than_wrong_without_its_artifact(tmp_path):
    from src.models.stan_components import _arm_a_grid

    assert _arm_a_grid(tmp_path).empty


def _sweep_table():
    return pd.DataFrame([
        {"head": "reb", "variant": "carry_forward", "val_r2": 0.90},
        {"head": "reb", "variant": "log_own", "val_r2": 0.93},
        {"head": "reb", "variant": "log_own_spline", "val_r2": 0.91},
    ])


def test_selection_reads_the_validation_column_and_there_is_no_other():
    """The protocol this project exists to enforce.

    This test used to hand `_finalize` a table where `log_own_spline` won on test by a
    mile and lost on validation, and assert that validation won. Since 2026-08-05 the
    stronger statement is available: there is **no test column to prefer**, because the
    sweep never scores those rows. `src/models/held_out.py` raises on them and
    `src/final_evaluation.py` reads them once.
    """
    from src.models.stan_components import _finalize

    out = _finalize(_sweep_table(), "val_r2", higher_is_better=True)
    assert out.loc[out["selected"], "variant"].tolist() == ["log_own"]
    assert not [c for c in out.columns if c.startswith("test_")]


def test_beats_floor_and_selection_now_read_the_same_column():
    """`beats_floor` used to be a TEST fact beside a validation `selected`.

    That meant a head could be chosen on one split and certified on another — which reads
    as rigour and is actually the two halves of a decision disagreeing about what they
    are measured on. Both now read validation; certification against the held-out seasons
    is `src/final_evaluation.py`'s job and nothing else's.
    """
    from src.models.stan_components import _finalize

    table = _sweep_table()
    table.loc[table["variant"] == "log_own", "val_r2"] = 0.89   # now below the floor
    out = _finalize(table, "val_r2", higher_is_better=True)
    row = out[out["variant"] == "log_own_spline"].iloc[0]
    # `log_own_spline` is now the best fitted arm and clears the floor.
    assert bool(row["selected"]) and bool(row["beats_floor"])
    beaten = out[out["variant"] == "log_own"].iloc[0]
    assert not bool(beaten["beats_floor"])
    assert bool(out[out["variant"] == "carry_forward"].iloc[0]["beats_floor"])


def test_lower_is_better_selection_flips_direction_for_the_nll_heads():
    from src.models.stan_components import _finalize

    table = pd.DataFrame([
        {"head": "ftm|fta", "variant": "carry_forward", "val_nll": 3.08},
        {"head": "ftm|fta", "variant": "logit_own", "val_nll": 3.09},
        {"head": "ftm|fta", "variant": "logit_own_spline", "val_nll": 3.11},
    ])
    out = _finalize(table, "val_nll", higher_is_better=False)
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


# ── The role-graded dispersion ───────────────────────────────────────────────
#
# Two levels. These first three go through the head's own python — the bins it builds and
# the predictive it gathers them into — and need no sampler. The three after them evaluate
# the Stan target itself with `log_prob`, which is the only way to check that the `.stan`
# file nests rather than merely that the wrapper does.

def _role_frame(mpg, team_games=82, gp=None):
    frame = pd.DataFrame({"minutes_per_game_lag1": np.asarray(mpg, dtype=float)})
    frame["team_games"] = team_games
    frame["gp"] = team_games if gp is None else gp
    return frame


def test_role_bins_are_one_based_and_follow_the_measured_edges():
    """`ROLE_EDGES` = [0, 12, 24, 30, 60], right-closed, so 12.0 is still the low bucket."""
    from src.models.stan_availability import role_bins

    frame = _role_frame([1.0, 12.0, 12.1, 24.0, 27.5, 30.0, 36.0])
    assert list(role_bins(frame)) == [1, 1, 2, 2, 3, 3, 4]
    # Disabled is all-ones, which with n_rho = 1 is the shared-dispersion head exactly.
    assert set(role_bins(frame, role_rho=False)) == {1}


def test_a_row_outside_the_edges_falls_into_the_widest_bucket():
    """No prior minutes at all, or an implausible value above the top edge.

    The fringe bucket carries the LARGEST dispersion, so this is the conservative
    direction: an unknown player is treated as the most variable rather than the least.
    Silently landing in `30+ mpg` would hand him the model's most reliable availability.
    """
    from src.models.stan_availability import role_bins

    assert list(role_bins(_role_frame([np.nan, 0.0, 99.0, 35.0]))) == [1, 1, 1, 4]


def test_the_head_gathers_each_rows_own_dispersion_into_its_predictive():
    """`rho_bin` has to reach the predictive, not just the fit.

    Built without a sampler by writing known per-bin draws onto a fitted-shaped head: a
    fringe player and a star with the SAME mean must come back with different spreads, and
    the spread each gets must be his own bucket's. If the gather were dropped, both rows
    would take column 0 and this reads as a single beta-binomial.
    """
    from src.models.availability import predictive_pmf
    from src.models.stan_availability import StanAvailability

    head = StanAvailability(features=["minutes_per_game_lag1"], first_season=None,
                            mixture=False)
    frame = _role_frame([6.0, 18.0, 27.0, 36.0])
    # A fitted head, assembled by hand: zero slopes so every row shares one mean, and one
    # posterior draw so the mixture is a single beta-binomial per row.
    head.scaler = type("I", (), {"transform": staticmethod(lambda x: np.zeros_like(x))})()
    head.alpha_draws = np.zeros(1)
    head.beta_draws = np.zeros((1, 1))
    head.rho_draws = np.array([[0.30, 0.25, 0.20, 0.15]])
    head.predictive_draws = 1

    mus, rhos = head.mu_draws(frame)
    assert np.allclose(mus, 0.5)
    assert np.allclose(rhos, [[0.30, 0.25, 0.20, 0.15]])
    assert np.allclose(head.rho_row(frame), [0.30, 0.25, 0.20, 0.15])

    pmf = head.predict_pmf(frame, 82)
    grid = np.arange(83)
    variance = (pmf * grid ** 2).sum(axis=1) - ((pmf * grid).sum(axis=1)) ** 2
    # Strictly decreasing spread across the four buckets, and each row equals the
    # beta-binomial at its OWN rho rather than at any shared one.
    assert np.all(np.diff(variance) < 0)
    for row, rho in enumerate([0.30, 0.25, 0.20, 0.15]):
        expected = predictive_pmf(np.array([82]), np.array([0.5]), rho, 82)
        assert np.allclose(pmf[row], expected[0], atol=1e-10)


def test_the_window_cuts_fitting_rows_and_leaves_scoring_alone():
    """The trap this whole change is arranged around.

    `availability_design` is imported by six other modules, so the window lives on the
    head. `fitting_rows` is the only thing that applies it, and the frames handed to
    `predict_*` are never touched — a shorter fitting window is a bias-variance trade on
    the fit, not a claim about which rows may be predicted.
    """
    from src.models.stan_availability import restrict_window, StanAvailability

    frame = pd.DataFrame({"season": ["2010-11", "2012-13", "2019-20", "2023-24"]})
    assert list(restrict_window(frame, "2012-13")["season"]) == [
        "2012-13", "2019-20", "2023-24"]
    assert len(restrict_window(frame, None)) == 4
    assert len(StanAvailability(first_season="2019-20").fitting_rows(frame)) == 2
    assert len(StanAvailability(first_season=None).fitting_rows(frame)) == 4


# ── The Stan target itself ───────────────────────────────────────────────────

def _lp(model, data, params, **kwargs):
    """One `target` evaluation. `jacobian=False` drops the bounded-parameter transform,
    which is what makes two blocks of DIFFERENT dimension comparable at all."""
    return float(model.log_prob(params, data, jacobian=False, sig_figs=18,
                                **kwargs)["lp__"].iloc[0])


def _betabinomial_fixture(seed=0, n_rows=40, n_bins=3):
    from src.models.stan_utils import pi_block

    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_rows, 2))
    data = {"N": n_rows, "K": 2, "X": X.tolist(),
            "n": [82] * n_rows, "y": rng.integers(0, 83, n_rows).tolist(),
            "beta_scale": 0.7, "intercept_scale": 5.0,
            "S": 0, "season_idx": [0] * n_rows, "year_sd_scale": 0.25,
            # The DISABLED mixture, which is what the other five heads on this file pass.
            **pi_block(n_rows)}
    bins = rng.integers(1, n_bins + 1, n_rows)
    params = {"alpha": 0.1, "beta": [0.2, -0.3], "year_z": [], "sigma_year": [],
              "theta": [], "mu_low": [], "rho_low": [], "gamma": []}
    return data, bins, params, X


@needs_cmdstan
def test_n_rho_one_nests_the_shared_rho_likelihood_exactly():
    """The rollback guarantee, and the reason this was an addition rather than a change.

    Five heads share `betabinomial_glm.stan` and only availability wants a graded
    dispersion. `n_rho = 1` with all-ones bins must therefore be the model those other four
    have always fitted — not approximately, exactly — which is the same discipline `S = 0`
    uses for the year effect. Asserted as an identity on the target, because "the numbers
    look similar" is what this test exists to refuse.
    """
    from src.models.stan_utils import compile_model

    model = compile_model("betabinomial_glm")
    data, bins, params, _ = _betabinomial_fixture()
    rho = 0.23

    shared = _lp(model, {**data, "n_rho": 1, "rho_bin": [1] * data["N"]},
                 {**params, "rho": [rho]})
    # The same number reached the other way: a three-bin block whose entries agree is the
    # shared model too, for ANY bin assignment — so the gather cannot be secretly
    # reordering rows.
    graded = _lp(model, {**data, "n_rho": 3, "rho_bin": bins.tolist()},
                 {**params, "rho": [rho] * 3})
    assert shared == graded


@needs_cmdstan
def test_the_shared_arm_is_still_the_beta_binomial_it_always_was():
    """`n_rho = 1` against scipy, at the level of the likelihood.

    Differences between two values of `rho` are taken rather than the raw target, which
    cancels the `alpha` and `beta` priors — `~` statements drop their normalizing
    constants, so the absolute target is not a quantity scipy can reproduce.
    """
    from scipy.stats import betabinom

    from src.models.stan_utils import compile_model

    model = compile_model("betabinomial_glm")
    data, _, params, X = _betabinomial_fixture()
    block = {**data, "n_rho": 1, "rho_bin": [1] * data["N"]}

    def scipy_loglik(rho_row):
        mu = 1.0 / (1.0 + np.exp(-(params["alpha"] + X @ np.array(params["beta"]))))
        s = (1.0 - rho_row) / rho_row
        return betabinom.logpmf(np.array(data["y"]), 82, s * mu, s * (1.0 - mu)).sum()

    stan_delta = (_lp(model, block, {**params, "rho": [0.31]})
                  - _lp(model, block, {**params, "rho": [0.2]}))
    scipy_delta = (scipy_loglik(np.full(data["N"], 0.31))
                   - scipy_loglik(np.full(data["N"], 0.2)))
    assert abs(stan_delta - scipy_delta) < 1e-6


@needs_cmdstan
def test_rho_bin_gathers_the_dispersion_of_each_rows_own_bin():
    """The gather is row-wise and correct, checked against scipy per row.

    A vectorized `rho[rho_bin]` is exactly the kind of expression that silently applies
    `rho[1]` everywhere, or transposes the bins, and still produces a plausible posterior.
    Moving two of three bins and holding the third fixed pins which rows moved and by how
    much.
    """
    from scipy.stats import betabinom

    from src.models.stan_utils import compile_model

    model = compile_model("betabinomial_glm")
    data, bins, params, X = _betabinomial_fixture()
    block = {**data, "n_rho": 3, "rho_bin": bins.tolist()}
    mu = 1.0 / (1.0 + np.exp(-(params["alpha"] + X @ np.array(params["beta"]))))

    def scipy_loglik(rho_by_bin):
        rho_row = np.asarray(rho_by_bin)[bins - 1]
        s = (1.0 - rho_row) / rho_row
        return betabinom.logpmf(np.array(data["y"]), 82, s * mu, s * (1.0 - mu)).sum()

    graded, flat = [0.31, 0.2, 0.12], [0.2, 0.2, 0.2]
    stan_delta = (_lp(model, block, {**params, "rho": graded})
                  - _lp(model, block, {**params, "rho": flat}))
    assert abs(stan_delta - (scipy_loglik(graded) - scipy_loglik(flat))) < 1e-6

    # And the bins are not interchangeable: permuting the values across bins is a
    # different model, unless the permutation happens to be the identity.
    permuted = _lp(model, block, {**params, "rho": [0.12, 0.2, 0.31]})
    assert permuted != _lp(model, block, {**params, "rho": graded})


# ── The low-availability mixture ─────────────────────────────────────────────
#
# Two levels again, and the split matters more here than it did for `rho_bin`. The Stan
# target is where the nesting has to be exact, because five other heads read this file;
# the python is where the *predictive* has to be the mixture, because a head that fits one
# model and predicts another would pass every log_prob test in this section.

def _pi_fixture(n_rows=40, n_cols=3, seed=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n_rows, n_cols))


@needs_cmdstan
def test_p_zero_and_theta_zero_both_nest_the_current_target_bit_for_bit():
    """The rollback guarantee for the mixture, in both of its forms.

    `betabinomial_glm.stan` serves six heads and only availability wants a mixture, so the
    bar is the one `n_rho = 1` and `S = 0` already meet: not "close", **equal**. Two
    nestings, because two different things could go wrong. `P = 0` is the parameter space
    — the mixture block is zero-length, so the five other heads fit the model they always
    have. `theta = 0` is the likelihood — the block exists and is switched off, which is
    the state a *fitted* head passes through and the reason `theta` is bounded at 0 rather
    than reached as `gamma_0 -> -inf`.

    Equality on doubles, not a tolerance, because the mixture is written as an ADDITIVE
    correction to the untouched beta-binomial statement and that correction is exactly
    `0.0` at `pi = 0`. A tolerance here would hide the difference between "the same target"
    and "a target that rounds to the same 15 digits".
    """
    from src.models.stan_utils import compile_model, pi_block

    model = compile_model("betabinomial_glm")
    data, _, params, _ = _betabinomial_fixture()
    Z = _pi_fixture(data["N"])
    rho = {"rho": [0.23]}
    shared = {**data, "n_rho": 1, "rho_bin": [1] * data["N"]}

    off = _lp(model, shared, {**params, **rho})
    on = _lp(model, {**shared, **pi_block(data["N"], Z)},
             {**params, **rho, "theta": [0.0], "mu_low": [0.1], "rho_low": [0.05],
              "gamma": [0.0] * Z.shape[1]})
    assert off == on

    # `theta = 0` is `pi = 0` for ANY gamma, so switching the mixture off must not depend
    # on where its covariate block happens to sit. What is left is the gamma prior, whose
    # value is known in closed form — so this pins the likelihood contribution at exactly
    # zero rather than at "small".
    gamma = [0.4, -0.2, 0.9]
    with_gamma = _lp(model, {**shared, **pi_block(data["N"], Z)},
                     {**params, **rho, "theta": [0.0], "mu_low": [0.1],
                      "rho_low": [0.05], "gamma": gamma})
    prior = -0.5 * float(np.sum((np.asarray(gamma) / 2.5) ** 2))
    assert abs((with_gamma - off) - prior) < 1e-9


@needs_cmdstan
def test_the_mixture_target_is_the_two_component_mixture_scipy_computes():
    """The likelihood itself, against scipy, at the level of the target.

    Differences between two parameter vectors are taken rather than the raw target, which
    cancels the constants the `~` statements drop — the same device the shared-rho test
    uses one section up. The gamma prior is the one term that does not cancel, so it is
    added back explicitly instead of being absorbed into a tolerance.
    """
    from scipy.special import expit, logsumexp
    from scipy.stats import betabinom

    from src.models.stan_utils import compile_model, pi_block

    model = compile_model("betabinomial_glm")
    data, _, params, X = _betabinomial_fixture()
    Z = _pi_fixture(data["N"])
    rho, y = 0.23, np.asarray(data["y"])
    block = {**data, "n_rho": 1, "rho_bin": [1] * data["N"], **pi_block(data["N"], Z)}
    mu = expit(params["alpha"] + X @ np.asarray(params["beta"]))

    def scipy_loglik(theta, mu_low, rho_low, gamma):
        s = (1 - rho) / rho
        main = betabinom.logpmf(y, 82, s * mu, s * (1 - mu))
        s_low = (1 - rho_low) / rho_low
        low = betabinom.logpmf(y, 82, s_low * mu_low, s_low * (1 - mu_low))
        pi = theta * expit(Z @ np.asarray(gamma))
        mixed = logsumexp(np.vstack([np.log1p(-pi) + main, np.log(pi) + low]), axis=0)
        return mixed.sum() - 0.5 * float(np.sum((np.asarray(gamma) / 2.5) ** 2))

    a = dict(theta=0.12, mu_low=0.10, rho_low=0.044, gamma=[0.4, -0.2, 0.9])
    b = dict(theta=0.30, mu_low=0.22, rho_low=0.150, gamma=[-0.3, 0.5, 0.1])

    def stan_lp(d):
        return _lp(model, block, {**params, "rho": [rho], "theta": [d["theta"]],
                                  "mu_low": [d["mu_low"]], "rho_low": [d["rho_low"]],
                                  "gamma": d["gamma"]})

    assert abs((stan_lp(a) - stan_lp(b))
               - (scipy_loglik(**a) - scipy_loglik(**b))) < 1e-9


def test_pi_block_disabled_is_one_state_rather_than_two_that_look_alike():
    """`Z=None` is the disabled block; an empty `Z` is a mistake and raises.

    The same contract `rho_block` and `year_block` carry — "off" has exactly one spelling,
    so a head cannot half-enable a mixture by handing over a design it built from an empty
    column list.
    """
    from src.models.stan_utils import pi_block

    off = pi_block(7)
    assert off["P"] == 0 and np.asarray(off["Z"]).shape == (7, 0)

    on = pi_block(7, np.zeros((7, 3)))
    assert on["P"] == 3

    with pytest.raises(ValueError, match="DISABLED"):
        pi_block(7, np.zeros((7, 0)))
    with pytest.raises(ValueError, match="matching"):
        pi_block(7, np.zeros((6, 3)))


def test_pis_covariate_block_is_the_one_the_ladder_selected_on():
    """D5, pinned. `PI_FEATURES` is a *shipped choice* that lands in the persisted recipe.

    It is held in `stan_availability` rather than imported, because `availability_window`
    imports `season_terms`, which imports `stan_availability` — a top-level import would be
    a cycle. Two copies of a list is exactly how a head comes to ship a different model
    from the one that was selected, so the equality is asserted rather than trusted.
    """
    from src.models.availability_window import PI_COLS
    from src.models.stan_availability import PI_FEATURES

    assert PI_FEATURES == PI_COLS


def test_the_preseason_block_is_the_arm_the_ladder_selected_on():
    """The same discipline for the preseason block, and for the same cycle reason.

    `PRESEASON_COLS` is a shipped choice that lands in the persisted recipe, held in
    `stan_availability` because importing `availability_preseason` at module scope would
    reach `availability_window` -> `season_terms` -> this module. Two copies of a column
    list is how a head comes to ship a different model from the one that was measured, so
    the equality against the P2 ladder's own declared primary is asserted.
    """
    from src.models.availability_preseason import ARMS, PRIMARY_ARM, SHIPPED_ARM
    from src.models.stan_availability import PRESEASON_COLS

    # The SHIPPED arm, which is not the declared primary: `p1_block` was promoted on the
    # rolling harness (CRPS -0.0977 [-0.1569, -0.0408], boundary -0.0014, 8 of 10 origins),
    # which is a fitting-half decision rather than a validation-driven swap.
    assert SHIPPED_ARM != PRIMARY_ARM
    beta_block, pi_block_cols = ARMS[SHIPPED_ARM]
    assert PRESEASON_COLS == list(beta_block)
    # P2 measured the block on `pi` as a loss with an interval (§14d's verdict, reached by
    # the block with the better prior), so the shipped arm puts nothing there.
    assert pi_block_cols == []


def test_the_minutes_preseason_block_is_the_arm_p3_selected():
    """P3 decision 2: the shipped column is the SEASON-CENTRED delta plus the age split.

    Pinned against `minutes_preseason`'s own constants rather than retyped, because the
    centred column is the one an attribution arm won on the fitting half and the raw one is
    the arm that was declared — shipping the wrong one of those two is a silent reversal of
    the finding, not a typo.
    """
    from src.eda.preseason_value import MISSING_AGE_COLS
    from src.models.minutes_preseason import CENTERED_DELTA, OWN_DELTA
    from src.models.stan_minutes import PRESEASON_COLS

    assert PRESEASON_COLS == [CENTERED_DELTA] + list(MISSING_AGE_COLS)
    assert OWN_DELTA not in PRESEASON_COLS          # the uncentred arm does NOT ship
    # P3 decision 4: the empirical-Bayes volume shrink is a null worth 0.05 CRPS, so no
    # shrunk column reaches the head.
    assert not any(c.endswith("_shrunk") for c in PRESEASON_COLS)


def test_the_minutes_shared_builder_never_carries_the_preseason_block():
    """`build_design` is how `stan_composition`, `minutes_window`, `minutes_unification`
    and `minutes_preseason` reach their rows. The block is a suffix on the head's own
    path, so a variant's column order is unchanged when the flag flips."""
    from src.models.stan_minutes import PRESEASON_COLS, head_features

    base = ["a", "b", "c"]
    assert head_features(base, False) == base
    assert head_features(base, True) == base + list(PRESEASON_COLS)
    assert head_features(base, True)[:len(base)] == base


def test_the_component_preseason_block_is_per_head_and_on_each_heads_own_link():
    """Session 6b's shipped block, as a property of the column lists.

    The one thing that cannot be shared across this family: each head's delta is on **its
    own link** — `log1p` of a per-36 rate for a count, `logit` of a percentage for a
    conversion — so a module-level `PRESEASON_COLS` in the shape `stan_minutes` uses would
    put `reb`'s preseason rebounding on `blk`'s linear predictor. The four indicators ARE
    shared, because a missing preseason row is the same event for every head.

    The shipped delta is the **shrunk** one, which is the opposite of `stan_minutes` on both
    counts: shrunk rather than raw (the fitting half chose the volume weight over P1's
    additive term on every head) and uncentred rather than centred (centring loses on this
    family, because a per-36 rate has already divided the exposure out). Shipping either of
    those the other way round is a silent reversal of a measured finding.
    """
    from src.models.component_rates import CONVERSION_HEADS, COUNT_HEADS
    from src.models.stan_components import (PRESEASON_EXCLUDE, PRESEASON_MISSING_COLS,
                                            head_features, head_preseason_cols)

    seen = set()
    for component in COUNT_HEADS + [m for m, _ in CONVERSION_HEADS]:
        cols = head_preseason_cols(component, True)
        if component in PRESEASON_EXCLUDE:
            assert cols == [], f"{component} is opted out and must carry no block"
            continue
        assert cols[0] == f"pre_d_{component}_shrunk"
        assert cols[1:] == list(PRESEASON_MISSING_COLS)
        assert cols[0] not in seen, "two heads share a delta column"
        seen.add(cols[0])
        # Uncentred, and not the raw unshrunk delta either.
        assert not cols[0].endswith("_centered")
        assert f"pre_d_{component}" != cols[0]

    base = ["a", "b", "c"]
    assert head_features(base, "reb", False) == base
    assert head_features(base, "reb", True) == base + head_preseason_cols("reb", True)
    # Suffix, so a persisted recipe's column order is stable when the flag flips.
    assert head_features(base, "reb", True)[:len(base)] == base


def test_fg3m_given_fg3a_is_opted_out_of_the_preseason_block():
    """The one head 6b measured as WORSE with the block, pinned so it cannot drift back in.

    Ten of eleven heads improve under the posterior at a median retention of 0.991;
    `fg3m|fg3a` goes the other way on CRPS, NLL and PIT KS at once. Three instruments agree
    — P1's attribution said its gain was the shared indicator rather than preseason 3P%, 6b's
    pooled point MLE was already positive, and the posterior control is larger in the same
    direction. Prior-season 3P% over ~200 attempts beats preseason 3P% over ~15, so the block
    adds variance and no signal.

    Pinned as a property rather than left to the constant, because "ship the block on every
    head" is the obvious tidy-up and it would silently undo a measured decision. The exclusion
    also has to be exhaustive in the other direction: no OTHER head may be opted out without
    the same evidence, so the set is asserted whole.
    """
    from src.models.stan_components import (PRESEASON_EXCLUDE, head_features,
                                            head_preseason_cols)

    assert PRESEASON_EXCLUDE == frozenset({"fg3m"})
    assert head_preseason_cols("fg3m", True) == []
    # The head fits exactly what it fitted before the block existed.
    base = ["a", "b", "c"]
    assert head_features(base, "fg3m", True) == base
    assert head_features(base, "fg3m", True) == head_features(base, "fg3m", False)
    # And its sibling conversion heads are unaffected.
    assert head_preseason_cols("ftm", True)
    assert head_preseason_cols("fg2m", True)
    assert head_preseason_cols("fg3a", True)


def test_the_component_shared_builder_never_carries_the_preseason_block():
    """`component_rates.build_design` is how `season_terms`, `posteriors`, the substitution
    sweep and `components_preseason` itself reach their rows. A column that is structurally
    zero before 2004-05 must not enter any of them by accident, which is why `head_design`
    is a separate path — the `attach_absence_mix` precedent, third head family over."""
    import inspect

    from src.models import component_rates
    from src.models.stan_components import head_design

    source = inspect.getsource(component_rates.build_design)
    for token in ("pre_d_", "preseason", "has_preseason"):
        assert token not in source, f"`build_design` reaches for {token}"
    # And the head's own path is the thing that adds them.
    assert "attach_preseason" in inspect.getsource(head_design)


def test_the_component_control_arm_can_never_be_selected():
    """The no-preseason control is what the shipped arm is measured against.

    If `_finalize` could select it, a run that exists to *price* the block would silently
    re-decide it — and `posteriors.selected_specs` would then persist a head with no block
    while the config flag still said `true`. `stan_minutes.sweep` marks its own the same way.
    """
    import pandas as pd

    from src.models.stan_components import CONTROL_SUFFIX, _finalize

    table = pd.DataFrame([
        {"head": "reb", "variant": "carry_forward", "val_r2": 0.10},
        {"head": "reb", "variant": "log_own", "val_r2": 0.50},
        # The control scores BEST and must still lose the `selected` flag.
        {"head": "reb", "variant": f"log_own{CONTROL_SUFFIX}", "val_r2": 0.99},
    ])
    out = _finalize(table, "val_r2", higher_is_better=True)
    assert out.loc[out["variant"] == "log_own", "selected"].iloc[0]
    assert not out.loc[out["variant"].str.endswith(CONTROL_SUFFIX), "selected"].iloc[0]
    assert out.loc[out["variant"].str.endswith(CONTROL_SUFFIX), "is_control"].iloc[0]


def test_the_shared_design_builder_never_carries_the_preseason_block():
    """The guard the whole opt-in design rests on, as a property of the column lists.

    `availability_design` is how the minutes, composition, games-played, exchangeability,
    no-prior and season-term consumers reach their rows. A preseason column there is
    structurally zero before 2004-05 and would re-scope every one of them silently — the
    `attach_absence_mix` precedent, and the reason `head_design` exists as a separate path.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.stan_availability import (PRESEASON_COLS, head_features)

    for column in PRESEASON_COLS:
        assert column not in FEATURE_COLS
    assert head_features(False) == list(FEATURE_COLS)
    assert head_features(True) == list(FEATURE_COLS) + list(PRESEASON_COLS)
    # And the block is a suffix, so a persisted recipe's column order is stable when the
    # flag flips — a consumer that indexes coefficients positionally keeps working.
    assert head_features(True)[:len(FEATURE_COLS)] == list(FEATURE_COLS)


def _mixture_head(n_draws=1, theta=0.0, mu_low=0.10, rho_low=0.05):
    """A fitted-shaped mixture head, assembled by hand — no sampler, no design build."""
    from src.models.stan_availability import PI_FEATURES, StanAvailability

    identity = type("I", (), {"transform": staticmethod(lambda x: np.zeros_like(x))})()
    head = StanAvailability(features=["minutes_per_game_lag1"], first_season=None,
                            mixture=True)
    head.scaler, head.pi_scaler = identity, identity
    head.alpha_draws = np.zeros(n_draws)
    head.beta_draws = np.zeros((n_draws, 1))
    head.rho_draws = np.tile(np.array([[0.30, 0.25, 0.20, 0.15]]), (n_draws, 1))
    head.theta_draws = np.full(n_draws, float(theta))
    head.mu_low_draws = np.full(n_draws, float(mu_low))
    head.rho_low_draws = np.full(n_draws, float(rho_low))
    head.gamma_draws = np.zeros((n_draws, len(PI_FEATURES)))
    head.predictive_draws = n_draws
    frame = _role_frame([6.0, 18.0, 27.0, 36.0])
    for column in PI_FEATURES:
        frame[column] = 0.0
    return head, frame


def test_theta_zero_reproduces_the_single_component_predictive_exactly():
    """The python half of the nesting, and it is a different claim from the Stan half.

    A head can fit the nested target and still *predict* the mixture — `pi` reaches the
    predictive through three separate methods — so the rollback has to hold there too.
    Exact equality: at `theta = 0` the weight is identically zero, so the convex
    combination is the main component and not a blend of it with a negligible other.
    """
    head, frame = _mixture_head(theta=0.0)
    mixed = head.predict_pmf(frame, 82)
    mixed_mean = head.predict_mean(frame)
    head.mixture = False
    assert np.array_equal(mixed, head.predict_pmf(frame, 82))
    assert np.array_equal(mixed_mean, head.predict_mean(frame))


def test_the_predictive_is_the_convex_combination_and_the_mean_is_the_mixtures():
    """`pi` has to reach the pmf, the mean and the moments — and be the SAME `pi` in each.

    The trap this is arranged around is a head whose pmf carries the mixture while
    `predict_mean` returns the main component's mean: `score_arm` turns that mean into MAE
    and R², so the head would be credited with an accuracy its own predictive does not
    have. `availability_window.MixtureFrailty.predict_mean` makes the same choice, which is
    what keeps the ladder row and the port comparable at all.
    """
    from src.models.availability import predictive_pmf

    head, frame = _mixture_head(theta=0.4)
    pi, mu_low, rho_low = head.mixture_draws(frame)
    # gamma = 0, so inv_logit(0) = 0.5 and pi is half of theta on every row.
    assert np.allclose(pi, 0.2)

    main = np.vstack([predictive_pmf(np.array([82]), np.array([0.5]), r, 82)[0]
                      for r in (0.30, 0.25, 0.20, 0.15)])
    low = predictive_pmf(np.full(4, 82), np.full(4, 0.10), 0.05, 82)
    assert np.allclose(head.predict_pmf(frame, 82), 0.8 * main + 0.2 * low, atol=1e-12)
    assert np.allclose(head.predict_mean(frame), 0.8 * 0.5 + 0.2 * 0.10)

    # And the moments the board decomposition reads are the MIXTURE's, which carries a
    # between-component term a single beta-binomial has no way to produce.
    grid = np.arange(83)
    pmf = head.predict_pmf(frame, 82)
    mean, variance = head.predictive_moments(frame)
    assert np.allclose(mean.mean(axis=0), (pmf * grid).sum(axis=1))
    assert np.allclose(variance.mean(axis=0),
                       (pmf * grid ** 2).sum(axis=1) - (pmf * grid).sum(axis=1) ** 2)


def test_the_sampler_draws_a_component_before_it_draws_a_rate():
    """A disrupted season is a different season, not an average of two.

    Drawing a rate from each component and averaging would produce a player who plays 60
    games in every world — precisely the season the arm exists to say does not happen. The
    signature of doing it right is a **bimodal** predictive, so the low mode is counted
    directly rather than checked through a moment that both implementations would match.
    """
    head, frame = _mixture_head(n_draws=4000, theta=0.5)
    samples = head.predict_samples(frame, seed=7)
    assert samples.shape == (4000, 4)
    # pi = 0.25 per row, and the low component puts nearly all of its mass under 20 games
    # where the main one (mu = 0.5) puts a minority of its own. The analytic comparison is
    # the substantive check; the bound beside it only rules out a sampler that ignored the
    # mixture entirely, which would land near the main component's share alone.
    share_low = (samples < 20).mean(axis=0)
    analytic = (head.predict_pmf(frame, 82)[:, :20]).sum(axis=1)
    assert np.allclose(share_low, analytic, atol=0.02)
    main_only = (head.predict_pmf(frame, 82)[:, :20].sum(axis=1) - 0.25) / 0.75
    assert np.all(share_low > main_only + 0.05)


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
    # The incumbent configuration explicitly: `BetaBinomialGLM` is a shared-dispersion
    # model over every row it is given, so a windowed, role-graded head would not be a
    # port of it. The shipped arm's own agreement with its own point MLE is checked by
    # `make stan-availability` against `RoleGradedBetaBinomial`, on the real design.
    stan = StanAvailability(l2=1.0, features=features, warmup=750, samples=750,
                            chains=4, predictive_draws=100,
                            first_season=None, role_rho=False,
                            mixture=False).fit(frame)

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
                            predictive_draws=200, first_season=None,
                            role_rho=False, mixture=False).fit(frame)
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
    # `rhos` comes back gathered per row, the same shape as `mus`, so the two pair up
    # element-wise rather than by broadcasting a per-draw scalar across the board.
    conditional = (n * mus * (1 - mus) * (1 + (n - 1) * rhos)).mean(axis=0)
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
                            predictive_draws=200, first_season=None,
                            role_rho=False, mixture=False).fit(frame)
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


def test_every_stan_head_stamps_which_code_wrote_its_diagnostics(tmp_path):
    """The 2026-08-13 defect, closed.

    The availability head was ported twice that day, six minutes apart, and the *earlier*
    fit's artifacts were written into `docs/preseason-plan.md` and `dashboard/decisions.py`
    as if they described the head that shipped. Nothing objected — the artifacts were
    internally consistent, `make docs-audit` had no claim on the term count, and the numbers
    were plausible. An artifact carried no record of which version of the code wrote it.

    The stamp rides on `diagnostics_frame` rather than on each head, because that is the one
    function every head goes through: a head cannot acquire diagnostics without acquiring
    provenance, which is what stops one of them from quietly lacking it.
    """
    from src.models.stan_utils import (PROVENANCE_COLS, code_provenance,
                                       diagnostics_frame)

    row = {"label": "x", "max_rhat": 1.0, "min_ess_bulk": 400, "min_ess_tail": 400,
           "divergences": 0, "treedepth_saturated": 0, "n_draws": 4000,
           "wall_clock_s": 1.0, "converged": True, "cmdstan": "2.35.0"}
    frame = diagnostics_frame([row])
    assert set(PROVENANCE_COLS) <= set(frame.columns)
    # Appended, never inserted: every consumer reads these artifacts by column NAME, and
    # the diagnostic columns must keep their positions for a human reading the CSV.
    assert list(frame.columns)[:10] == list(row)
    assert list(frame.columns)[-len(PROVENANCE_COLS):] == list(PROVENANCE_COLS)


def test_the_source_digest_moves_when_a_head_does_and_the_commit_alone_would_not(tmp_path):
    """`src_digest` is the field that would actually have caught 2026-08-13.

    The two fits that day shared a commit — the column list was edited between them without
    committing — so `git_commit` matched and `git_dirty` was true for both. Only a digest
    over the source trees separates them.
    """
    from src.models.stan_utils import PROVENANCE_TREES, code_provenance

    root = tmp_path / "src"
    for subdir, pattern in PROVENANCE_TREES:
        (root / subdir).mkdir(parents=True)
        (root / subdir / f"head{pattern[1:]}").write_text("PRESEASON_COLS = [1, 2, 3, 4, 5]")

    before = code_provenance(root)["src_digest"]
    edited = root / PROVENANCE_TREES[0][0] / "head.py"
    edited.write_text("PRESEASON_COLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]")
    after = code_provenance(root)["src_digest"]
    edited.write_text("PRESEASON_COLS = [1, 2, 3, 4, 5]")

    assert before != after                      # the five-column fit and the ten-column one
    assert code_provenance(root)["src_digest"] == before      # and it is content-addressed


def test_provenance_records_rather_than_raises_outside_a_checkout(tmp_path):
    """A record, not a guard. A dirty tree is the normal state during a working session and
    refusing to fit in one would cost more than the defect; an artifact written outside a
    git checkout is still worth writing."""
    from src.models.stan_utils import code_provenance

    (tmp_path / "models").mkdir()
    stamp = code_provenance(tmp_path)
    assert stamp["git_commit"] == ""
    assert stamp["src_digest"] and stamp["written_at"]


def test_merge_heads_replaces_only_the_refitted_heads():
    """`--heads` is a merge, not a hand-patch, and this is the property that makes it sound.

    The eleven heads are fitted separately — the factorization identity the whole module
    rests on — so a head's rows depend on nothing outside itself and refitting one leaves the
    other ten bit-identical at a fixed seed. `posteriors.py --groups` is the standing
    precedent for assembling one artifact from partial runs.

    The safety condition is that selection is **head-local**: `_finalize` picks `selected`
    within a head's own block, so a merged file cannot carry a stale winner from a comparison
    that spanned heads.
    """
    import pandas as pd

    from src.models.stan_components import merge_heads

    existing = pd.DataFrame([
        {"head": "reb", "variant": "log_own", "val_r2": 0.95},
        {"head": "ast", "variant": "log_own", "val_r2": 0.92},
        {"head": "fg3m|fg3a", "variant": "logit_own_spline", "val_r2": 0.12},
    ])
    fresh = pd.DataFrame([
        {"head": "fg3m|fg3a", "variant": "linear", "val_r2": 0.07},
        {"head": "fg3m|fg3a", "variant": "logit_own_spline", "val_r2": 0.13},
    ])
    out = merge_heads(existing, fresh)
    # The untouched heads survive verbatim, values included.
    assert set(out[out["head"] != "fg3m|fg3a"]["head"]) == {"reb", "ast"}
    assert float(out.loc[out["head"] == "reb", "val_r2"].iloc[0]) == 0.95
    # The refitted head is fully replaced, not appended to — no stale row survives.
    block = out[out["head"] == "fg3m|fg3a"]
    assert len(block) == 2
    assert set(block["variant"]) == {"linear", "logit_own_spline"}
    # And an empty/absent baseline is just the fresh rows.
    assert merge_heads(None, fresh).equals(fresh)


def test_the_covered_window_cut_is_per_head_not_family_wide():
    """A head with no preseason column keeps its full fitting window.

    The cut exists solely because a missing-preseason indicator on a pre-2005 row is an era
    dummy. A head that carries no such indicator has nothing to protect against, so cutting
    it discards 2,248 of 8,630 training rows (26%) to buy nothing — which is what a
    family-wide cut did to `fg3m|fg3a` for one afternoon, while the run printed that it
    fitted the pre-2026-08-15 head "exactly".

    Heads on different windows is normal rather than a compromise: they are fitted
    separately, the factorization is exact, and `stan_minutes` (2004-05) already differs
    from `stan_composition` (1996-97).
    """
    import pandas as pd

    from src.models.stan_components import PRESEASON_EXCLUDE, head_fitting_rows

    full = pd.DataFrame({"season": ["1997-98"] * 4 + ["2010-11"] * 4, "x": range(8)})
    covered = full[full["season"] == "2010-11"]

    # A head that carries the block gets the cut frame...
    assert len(head_fitting_rows(full, covered, "reb", True)) == len(covered)
    # ...and an opted-out head keeps everything.
    excluded = next(iter(PRESEASON_EXCLUDE))
    assert len(head_fitting_rows(full, covered, excluded, True)) == len(full)
    # With the block off nobody is cut, which is what makes `false` an exact rollback.
    assert len(head_fitting_rows(full, covered, "reb", False)) == len(full)
