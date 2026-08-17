"""Tests for the minutes-unification gate — does the composition supersede the marginal head?

None of these needs a sampler, which is the same argument `tests/test_posteriors.py` makes:
the gate reads persisted posteriors and refits nothing, so the fit is exactly the part it is
not about. The heads here are the real classes with their draws injected, which is also what
`minutes_unification.rehydrate_*` does against a real artifact.

Three things are pinned that a plausible-looking rewrite would break silently:

- **the collapse to the season unit sorts first.** The composition frame is ordered by
  team-game, so a player's games are scattered through it, and a `reduceat` over the frame's
  own order would sum whichever rows happened to be adjacent;
- **the verdict comes from the interval**, not from the point estimate, so a margin this repo
  could not distinguish from zero cannot be reported as a win in either direction;
- **a team's season minutes are fixed across draws.** That is the structural half of the
  measured verdict — the composition's season-total spread is not merely small, it is
  forbidden at the team level by the constraint the head exists to enforce — and it is the
  reason a dispersion knob cannot fix what the gate found.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import minutes_unification as MU
from src.models.posteriors import DesignRecipe, PosteriorArtifact


# ── Synthetic builders ────────────────────────────────────────────────────────

def _scattered_frame() -> pd.DataFrame:
    """Three players over two games, interleaved the way a team-game frame really is.

    Row order is game-major, so no player's rows are contiguous — which is the case a
    `reduceat` over the raw frame gets wrong while still returning a plausible shape.
    """
    return pd.DataFrame({
        "game_id": [1, 1, 1, 2, 2, 2],
        "player_id": [30, 10, 20, 20, 30, 10],
        "season": "2022-23",
        "y": [5, 1, 3, 4, 6, 2],
    })


def _minutes_artifact(n_draws: int = 16, n_features: int = 3) -> PosteriorArtifact:
    rng = np.random.default_rng(0)
    features = [f"x{j}" for j in range(n_features)]
    scaler = StandardScaler().fit(rng.normal(size=(50, n_features)))
    return PosteriorArtifact(
        head="minutes", head_label="min|available", family="betabinomial",
        response="mean_mu_x_trials",
        recipe=DesignRecipe(variant="logit_own_spline", features=features, scaler=scaler),
        draws={"alpha_draws": rng.normal(size=n_draws),
               "beta_draws": rng.normal(size=(n_draws, n_features)) * 0.1,
               "rho_draws": rng.uniform(0.01, 0.1, size=n_draws)},
        extras={"trials": "trials"}, provenance={"fit_window": "train"})


def _composition_artifact(n_draws: int = 16, n_features: int = 3, n_rho: int = 4
                          ) -> PosteriorArtifact:
    rng = np.random.default_rng(1)
    features = [f"x{j}" for j in range(n_features)]
    scaler = StandardScaler().fit(rng.normal(size=(50, n_features)))
    return PosteriorArtifact(
        head="composition", head_label="minutes composition", family="composition",
        response="eta",
        recipe=DesignRecipe(variant="betabinom_ot_graded", features=features,
                            scaler=scaler),
        draws={"alpha_draws": rng.normal(size=n_draws),
               "beta_draws": rng.normal(size=(n_draws, n_features)) * 0.1,
               "rho_draws": rng.uniform(0.05, 0.2, size=(n_draws, n_rho))},
        extras={"dispersed": 1, "n_rho": n_rho}, provenance={"fit_window": "train"})


# ── Collapsing to the season unit ─────────────────────────────────────────────

def test_season_totals_sums_a_players_games_even_when_they_are_not_contiguous():
    frame = _scattered_frame()
    # One draw whose values are the realized minutes, so the totals are checkable by hand.
    samples = frame["y"].to_numpy(float)[None, :]
    totals, units = MU.season_totals(samples, frame)

    assert list(units["player_id"]) == [10, 20, 30]
    # player 10 played 1 and 2; player 20 played 3 and 4; player 30 played 5 and 6.
    assert totals.shape == (1, 3)
    assert list(totals[0]) == [3.0, 7.0, 11.0]


def test_season_totals_keeps_units_separate_across_seasons():
    frame = _scattered_frame()
    second = frame.copy()
    second["season"] = "2023-24"
    both = pd.concat([frame, second], ignore_index=True)
    samples = both["y"].to_numpy(float)[None, :]

    totals, units = MU.season_totals(samples, both)
    assert len(units) == 6
    assert set(units["season"]) == {"2022-23", "2023-24"}
    assert totals.sum() == both["y"].sum()


def test_season_totals_raises_when_the_frame_and_the_samples_disagree():
    frame = _scattered_frame()
    with pytest.raises(ValueError, match="same player-games"):
        MU.season_totals(np.zeros((4, len(frame) - 1)), frame)


def test_realized_totals_matches_the_collapse_of_the_realized_column():
    frame = _scattered_frame()
    realized = MU.realized_totals(frame, "y")
    totals, units = MU.season_totals(frame["y"].to_numpy(float)[None, :], frame)
    assert list(realized["realized"]) == list(totals[0])
    assert list(realized["player_id"]) == list(units["player_id"])


# ── The decision rule ─────────────────────────────────────────────────────────

def test_verdict_is_decided_by_the_interval_alone():
    # An interval entirely below zero is a win however small the point estimate.
    assert MU.verdict({"crps_delta": -0.01, "ci_lo": -0.05, "ci_hi": -0.001}) == "wins"
    # Straddling zero is a tie, and the plan gives the composition the benefit of it.
    assert MU.verdict({"crps_delta": -5.0, "ci_lo": -9.0, "ci_hi": +1.0}) == "ties"
    assert MU.verdict({"crps_delta": +5.0, "ci_lo": -1.0, "ci_hi": +9.0}) == "ties"
    # Only an interval entirely above zero keeps the marginal head.
    assert MU.verdict({"crps_delta": +25.0, "ci_lo": +19.0, "ci_hi": +33.0}) == "loses"


def test_a_large_point_estimate_with_a_straddling_interval_is_still_a_tie():
    """The failure this guards is the one the availability ladder actually produced."""
    assert MU.verdict({"crps_delta": -0.1297, "ci_lo": -0.3154,
                       "ci_hi": +0.0672}) == "ties"


def test_paired_bootstrap_recovers_a_known_delta_and_brackets_it():
    rng = np.random.default_rng(3)
    b = rng.normal(100.0, 10.0, size=500)
    a = b + rng.normal(4.0, 1.0, size=500)          # a is worse by ~4, paired

    out = MU.paired_bootstrap(a, b, n_boot=500, seed=0)
    assert out["crps_delta"] == pytest.approx(4.0, abs=0.3)
    assert out["ci_lo"] < out["crps_delta"] < out["ci_hi"]
    assert out["ci_lo"] > 0
    assert out["p_delta_negative"] == 0.0
    assert MU.verdict(out) == "loses"


def test_the_pairing_is_what_resolves_the_difference():
    """The between-player spread is what the pairing removes, and it is most of it.

    Same marginals and the same true difference of 3 on both sides; the only change is
    whether the rows correspond. Paired, the delta carries only the within-row noise and
    the interval resolves; unpaired it carries the full 40-minute between-player spread on
    both sides and cannot.
    """
    rng = np.random.default_rng(4)
    b = rng.normal(100.0, 40.0, size=400)
    a = b + rng.normal(3.0, 1.0, size=400)
    paired = MU.paired_bootstrap(a, b, n_boot=400, seed=0)
    assert paired["ci_lo"] > 0                        # resolved
    assert MU.verdict(paired) == "loses"

    shuffled = rng.normal(100.0, 40.0, size=400)
    unpaired = MU.paired_bootstrap(shuffled + 3.0, b, n_boot=400, seed=0)
    assert (unpaired["ci_hi"] - unpaired["ci_lo"]) > \
        20 * (paired["ci_hi"] - paired["ci_lo"])


# ── Rehydration ───────────────────────────────────────────────────────────────

def test_rehydrated_minutes_head_predicts_from_the_injected_draws():
    artifact = _minutes_artifact()
    model = MU.rehydrate_minutes(artifact, keep=8)

    assert model.features == artifact.recipe.features
    assert np.array_equal(model.alpha_draws, artifact.draws["alpha_draws"])
    assert model.predictive_samples == 8
    # The disabled year term the constructor builds — a head that carried one could not
    # have been persisted at all, and `mu_draws` reads it on every call.
    assert not model.year.enabled

    rng = np.random.default_rng(5)
    frame = pd.DataFrame(rng.normal(size=(12, 3)), columns=artifact.recipe.features)
    frame["trials"] = 2000
    mus, rhos = model.mu_draws(frame, 8)
    assert mus.shape == (8, 12)
    assert len(rhos) == 8
    assert ((mus > 0) & (mus < 1)).all()


def test_rehydrated_composition_carries_the_graded_dispersion_per_bin():
    artifact = _composition_artifact(n_rho=4)
    model = MU.rehydrate_composition(artifact, keep=8)

    assert model.dispersed == 1 and model.n_rho == 4
    assert model.rho_by_bin.shape == (4,)
    assert model.rho == pytest.approx(float(model.rho_by_bin.mean()))
    _, _, rho = model.plug_in()
    assert len(rho) == 4


def test_rehydrating_an_undispersed_arm_leaves_no_rho():
    artifact = _composition_artifact(n_rho=1)
    artifact.extras["dispersed"] = 0
    model = MU.rehydrate_composition(artifact, keep=4)
    assert model.rho_draws is None and model.rho_by_bin is None and model.rho == 0.0


# ── Headroom for a season term ────────────────────────────────────────────────

def _headroom_inputs(resid: list[float], seasons: list[str], pred: float = 1000.0):
    """Draws whose mean is `pred` and whose realized values miss it by `resid`."""
    samples = np.full((8, len(resid)), pred)
    units = pd.DataFrame({"player_id": np.arange(len(resid)), "season": seasons})
    return samples, units, np.array([pred + r for r in resid], dtype=float)


def test_a_league_wide_shift_is_visible_as_headroom():
    """The positive control: when a season really does move the level, this finds it."""
    seasons = ["2022-23"] * 6 + ["2023-24"] * 6
    # Every 2022-23 row 100 minutes high, every 2023-24 row 100 low — a pure season effect.
    resid = [100.0] * 6 + [-100.0] * 6
    out = MU.season_effect_headroom(*_headroom_inputs(resid, seasons))

    assert out["league_season_mean_resid_sd"] == pytest.approx(100.0)
    # All of the residual variance here IS the season level, so a year term could take it.
    assert out["share_of_resid_var_reachable"] == pytest.approx(1.0)


def test_residuals_that_sum_to_zero_within_a_season_leave_no_headroom():
    """The composition's actual case, and the reason the answer is structural.

    A head that allocates every minute in the league has residuals summing to zero within
    each season, so there is no league-wide level left for a league-wide term to move —
    however large the individual misses are.
    """
    seasons = ["2022-23"] * 6 + ["2023-24"] * 6
    resid = [300.0, -300.0, 250.0, -250.0, 80.0, -80.0] * 2
    out = MU.season_effect_headroom(*_headroom_inputs(resid, seasons))

    assert out["resid_sd"] > 200.0                      # the misses are large
    assert out["league_season_mean_resid_sd"] == pytest.approx(0.0, abs=1e-9)
    assert out["share_of_resid_var_reachable"] == pytest.approx(0.0, abs=1e-12)


def test_the_log_ratio_skips_rows_below_the_qualification_threshold():
    """A realized zero would make the sd `-inf`, and a 4-minute row is noise, not spread."""
    seasons = ["2022-23"] * 4
    samples = np.full((8, 4), 1000.0)
    units = pd.DataFrame({"player_id": [1, 2, 3, 4], "season": seasons})
    realized = np.array([1200.0, 800.0, 0.0, 4.0])      # two qualified, two not

    out = MU.season_effect_headroom(samples, units, realized)
    assert out["log_ratio_n"] == 2
    assert np.isfinite(out["log_ratio_sd"])
    assert out["n"] == 4                                # the variance figures keep every row


# ── The zero-sum dynamic ──────────────────────────────────────────────────────

# Twelve players summing to exactly 5 x 48, so the per-player cap has real slack. A
# short rotation pins everyone near the cap and leaves nothing to re-allocate, which is a
# true property of the head and a useless fixture for measuring re-allocation.
_ROTATION = [36, 34, 32, 30, 24, 20, 18, 16, 12, 8, 6, 4]


def _two_team_frame(n_games: int = 4):
    """Two teams of twelve, each playing `n_games`, in real team-game blocks.

    Player ids are stable within a team across games and disjoint between teams, so each
    (player, season) unit spans `n_games` rows on exactly one team — which is what
    `teammate_coupling`'s single-team filter and `season_totals`' collapse both need.
    """
    from tests.test_stan_composition import _sequenced, _team_rows

    blocks = []
    for g in range(1, n_games + 1):
        for team in (10, 20):
            block = _team_rows(_ROTATION, game_id=g, team_id=team)
            block["player_id"] = team * 100 + np.arange(len(_ROTATION))
            blocks.append(block)
    return _sequenced(blocks)


def test_teammate_totals_are_negatively_correlated_under_the_constraint():
    """The dynamic the composition exists for, and the identity it has to hit.

    A fixed team total over K players forces the mean pairwise correlation of their season
    totals to exactly -1/(K-1). A head allocating a fixed pot reproduces it; a head drawing
    players independently cannot, and that is invisible in any marginal metric.
    """
    from src.models.stan_composition import simulate_minutes

    frame = _two_team_frame()
    draws = 200
    samples = simulate_minutes(frame, np.zeros((len(frame), draws)),
                               np.full((draws, 1), 0.12), seed=0)
    totals, units = MU.season_totals(samples, frame)

    out = MU.teammate_coupling(totals, units, frame, "composition")
    assert out["roster_size"] == pytest.approx(len(_ROTATION))
    assert out["r_implied_by_fixed_sum"] == pytest.approx(-1.0 / (len(_ROTATION) - 1))
    # Sitting on the identity, not merely somewhere negative.
    assert out["r_teammates"] == pytest.approx(-1.0 / (len(_ROTATION) - 1), abs=0.03)
    assert out["team_season_sum_sd"] == pytest.approx(0.0, abs=1e-9)


def test_independent_draws_show_neither_the_correlation_nor_the_fixed_total():
    """The marginal head's failure mode, on the same rows and the same statistic."""
    frame = _two_team_frame()
    _, units = MU.season_totals(np.zeros((2, len(frame))), frame)

    rng = np.random.default_rng(0)
    independent = rng.normal(1500.0, 300.0, size=(200, len(units)))
    out = MU.teammate_coupling(independent, units, frame, "independent")

    assert abs(out["r_teammates"]) < 0.1               # ~0, where truth is -0.2
    assert out["team_season_sum_sd"] > 100.0           # a fixed quantity, given real spread


# ── Injecting a player-season effect ──────────────────────────────────────────

def test_unit_codes_are_shared_across_a_players_games_and_distinct_across_players():
    frame = _scattered_frame()
    codes = MU.unit_codes(frame)
    by_player = pd.DataFrame({"player_id": frame["player_id"], "code": codes})
    # One code per player, the same in both games.
    assert by_player.groupby("player_id")["code"].nunique().eq(1).all()
    assert len(set(codes)) == 3


def test_injecting_a_player_season_effect_widens_the_season_total():
    """The measured answer to 'is the narrow season total a ceiling or a parameter?'

    The injection is a shock per (player, season) per draw, shared across that player's
    games — so it survives the sum over a season instead of averaging out, which is exactly
    what iid game noise cannot do. The team total stays exact throughout, which
    `simulate_minutes` asserts on every call.

    The fixture runs 40 games rather than a handful because the effect *is* the number of
    games: iid game noise accumulates as sqrt(G) while a persistent shock accumulates as G,
    so the gap between them only opens up over a season. At six games the same injection
    widens by 1.6x and at forty by well over 2x, which is the mechanism rather than a
    tolerance to be tuned.
    """
    from src.models.stan_composition import simulate_minutes

    frame = _two_team_frame(n_games=40)
    draws, codes = 120, MU.unit_codes(frame)
    n_units = int(codes.max()) + 1
    rho = np.full((draws, 1), 0.12)

    widths = []
    for sigma in (0.0, 0.2, 0.4):
        rng = np.random.default_rng(1)
        eta = sigma * rng.normal(size=(n_units, draws))[codes, :]
        totals, _ = MU.season_totals(simulate_minutes(frame, eta, rho, seed=0), frame)
        widths.append(float(totals.std(axis=0).mean()))
        # The constraint holds at every sigma — the spread comes from RE-ALLOCATION, which
        # is the whole reason a per-player effect is available where a shared one is not.
        assert totals.sum(axis=1).std() == pytest.approx(0.0, abs=1e-9)

    assert widths[0] < widths[1] < widths[2], widths       # monotone in sigma
    assert widths[2] > 2.0 * widths[0], widths


# ── The structural half of the verdict ────────────────────────────────────────

def test_a_teams_season_minutes_are_fixed_across_draws():
    """Summed over a team's players and its games, the composition has zero spread.

    The team constraint is per team-game, so it composes over a season: every draw
    allocates exactly `5 x game_length` in each game, and a season is a sum of games. That
    is why the head's season-total under-dispersion cannot be bought back with a dispersion
    parameter — a shared season-level effect is not merely unfitted here, it is forbidden.
    """
    from src.models.stan_composition import simulate_minutes

    from tests.test_stan_composition import _sequenced, _team_rows

    frame = _sequenced([_team_rows([30, 25, 20, 15, 10, 8, 7, 5, 5, 5, 5, 5],
                                   game_id=g) for g in (1, 2, 3)])
    draws = 24
    eta = np.zeros((len(frame), draws))
    rho = np.full((draws, 1), 0.12)
    samples = simulate_minutes(frame, eta, rho, seed=0)

    totals, _ = MU.season_totals(samples, frame)
    # Per player the draws move; summed over the team they cannot.
    assert totals.std(axis=0).max() > 0
    assert samples.sum(axis=1).std() == 0.0
    assert samples.sum(axis=1)[0] == frame.groupby("game_id")["N"].first().sum()


# ── Reading a fitted sigma_u off the artifact ─────────────────────────────────

def test_a_composition_artifact_without_sigma_u_rehydrates_with_the_effect_off():
    """Every artifact written before item 3d carries no `sigma_u_draws`, and must
    rehydrate as the head that shipped rather than as one with a silent zero effect."""
    model = MU.rehydrate_composition(_composition_artifact(), keep=8)
    assert not model.ps.enabled
    assert model.ps.sigma_draws.size == 0


def test_a_fitted_sigma_u_is_picked_up_and_widens_the_predictive():
    """The gate is re-takable rather than re-arguable only if the rehydrated head carries
    the effect the fit found. The stream comes off the artifact too, so the rehydrated
    head draws the same `z` sequence the fitted one would."""
    artifact = _composition_artifact()
    artifact.draws["sigma_u_draws"] = np.full(16, 0.4)
    artifact.extras.update({"player_season_effect": True, "u_sd_scale": 1.0,
                            "u_stream": "effects/ps", "sigma_u": 0.4})

    model = MU.rehydrate_composition(artifact, keep=8)
    assert model.ps.enabled and model.ps.stream == "effects/ps"
    np.testing.assert_allclose(model.ps.sigma_draws, 0.4)

    frame = pd.DataFrame({"player_id": [1, 1, 2, 2], "season": "2022-23"})
    shift = model.ps.shift(frame, np.arange(8))
    assert shift.shape == (4, 8)
    assert np.allclose(shift[0], shift[1]) and not np.allclose(shift[0], shift[2])


def test_the_sweep_marks_the_fitted_sigma_apart_from_the_injected_grid():
    """Once a sigma is fitted the sweep stops being the measurement and becomes the
    calibration check, so the two kinds of row must be distinguishable in the artifact —
    a fitted row silently indistinguishable from a tuned one is exactly the confusion the
    injection's caveat exists to prevent."""
    import inspect

    source = inspect.getsource(MU.player_season_effect_sweep)
    assert '"sigma_source": source' in source
    assert '"fitted"' in source and '"injected_grid"' in source


# ── Grading the injection by role ─────────────────────────────────────────────

def test_a_scalar_sigma_is_the_one_bin_model_rather_than_a_special_case():
    """`docs/draw-time-calibration-plan.md`'s nesting claim, at the arithmetic that carries
    it: a constant vector must be the scalar, not merely close to it."""
    bins = np.array([0, 1, 2, 3, 1, 0])
    np.testing.assert_array_equal(MU.unit_sigma(0.375, bins), np.full(6, 0.375))
    np.testing.assert_array_equal(MU.unit_sigma(np.full(MU.N_ROLE_BINS, 0.375), bins),
                                  MU.unit_sigma(0.375, bins))

    graded = np.array([0.6, 0.5, 0.4, 0.3])
    np.testing.assert_array_equal(MU.unit_sigma(graded, bins),
                                  np.array([0.6, 0.5, 0.4, 0.3, 0.5, 0.6]))


def test_a_sigma_vector_of_the_wrong_length_raises_rather_than_broadcasting():
    """Silently recycling three values over four buckets would produce a plausible table
    describing a grading nobody chose."""
    with pytest.raises(ValueError, match="one value per role bin"):
        MU.unit_sigma(np.array([0.5, 0.4, 0.3]), np.array([0, 1, 2, 3]))


def test_the_injected_shock_is_bit_identical_between_a_scalar_and_a_constant_vector():
    """The claim the whole graded round rests on: every figure the shared-sigma grid wrote
    before 2026-08-16 reproduces exactly, so the new code path is not a re-measurement.

    Checked on the linear predictor `simulate_minutes` is handed, because that is the only
    place sigma enters — everything downstream is the head's own allocation.
    """
    seen = []

    def spy(frame, eta, rho, seed):
        seen.append(np.array(eta, copy=True))
        return np.zeros((eta.shape[1], eta.shape[0]))

    frame = pd.DataFrame({"player_id": [1, 1, 2, 2, 3], "season": "2022-23"})
    codes = MU.unit_codes(frame)
    eta_base = np.arange(15, dtype=float).reshape(5, 3)

    original = MU.simulate_minutes
    MU.simulate_minutes = spy
    try:
        MU.injected_games(frame, eta_base, None, codes, 0.375, z_seed=7)
        MU.injected_games(frame, eta_base, None, codes, np.full(MU.N_ROLE_BINS, 0.375),
                          z_seed=7, bins=np.array([0, 3, 2]))
    finally:
        MU.simulate_minutes = original

    assert len(seen) == 2
    np.testing.assert_array_equal(seen[0], seen[1])
    assert not np.array_equal(seen[0], eta_base), "the shock never reached the predictor"


def test_role_bins_refuses_a_frame_the_recipe_never_binned():
    """`PlayerSeasonTerm._unit_bins` answers all-zeros without `rho_bin`, which is right for
    a scalar broadcasting over units and wrong for a graded sweep — that would fit one
    shared sigma four times and report it as a gradient."""
    frame = pd.DataFrame({"player_id": [1, 2], "season": "2022-23"})
    with pytest.raises(KeyError, match="rho_bin"):
        MU.role_bins(frame, MU.unit_codes(frame), 2)


def test_role_bins_reads_each_units_own_bin_and_is_ordered_by_unit_code():
    """The returned vector indexes `season_totals`' columns, so an ordering that follows the
    frame rather than the codes would grade the wrong players."""
    frame = pd.DataFrame({"player_id": [30, 10, 20, 10], "season": "2022-23",
                          "rho_bin": [4, 1, 2, 1]})
    codes = MU.unit_codes(frame)
    np.testing.assert_array_equal(MU.role_bins(frame, codes, 3), [0, 1, 3])


# ── The shipped injected sigma ────────────────────────────────────────────────

def test_the_shipped_sigma_comes_from_config_with_a_documented_default():
    """One place, so a consumer cannot forget it — and 0.0 has to be supported, because
    the un-injected head is the control every claim about the injection is measured
    against."""
    assert MU.shipped_sigma({}) == MU.SHIPPED_PS_SIGMA
    assert MU.shipped_sigma({"sim": {"minutes": {"player_season_sigma": 0.3}}}) == 0.3
    assert MU.shipped_sigma({"sim": {"minutes": {"player_season_sigma": 0.0}}}) == 0.0


def test_the_module_fallback_agrees_with_the_shipped_config_value():
    """A fallback that disagrees with config is correct and still a trap.

    `SHIPPED_PS_SIGMA` only fires when `sim.minutes.player_season_sigma` is absent, so a
    stale one changes no behaviour — which is exactly why it went unnoticed for two days
    after sigma moved 0.450 -> 0.375 on 2026-08-14, leaving the module constant reading one
    value while every simulator draw used another. The reader it misleads is whoever opens
    this module to change the injection.
    """
    import yaml

    cfg = yaml.safe_load(open("configs/default.yaml"))
    assert MU.shipped_sigma(cfg) == MU.SHIPPED_PS_SIGMA


def test_an_injected_sigma_makes_the_effect_live_on_the_rehydrated_head():
    """The whole point of shipping it here rather than in `src/sim/season.py`: the effect
    arrives by loading the head, not by remembering to apply it afterwards."""
    model = MU.rehydrate_composition(_composition_artifact(), keep=8, injected_sigma=0.45)
    assert model.ps.enabled and model.sigma_source == "injected"
    np.testing.assert_allclose(model.ps.sigma_draws, 0.45)

    frame = pd.DataFrame({"player_id": [1, 1, 2, 2], "season": "2022-23"})
    shift = model.ps.shift(frame, np.arange(8))
    assert np.allclose(shift[0], shift[1]) and not np.allclose(shift[0], shift[2])


def test_a_fitted_sigma_takes_precedence_over_the_injected_constant():
    """If the head ever ships a fitted `sigma_u`, the config constant must not override
    it — the artifact is the better estimate and carries a posterior."""
    artifact = _composition_artifact()
    artifact.draws["sigma_u_draws"] = np.linspace(0.40, 0.50, 16)
    artifact.extras.update({"player_season_effect": True, "sigma_u": 0.45})

    model = MU.rehydrate_composition(artifact, keep=16, injected_sigma=0.30)
    assert model.sigma_source == "fitted"
    np.testing.assert_allclose(model.ps.sigma_draws, np.linspace(0.40, 0.50, 16))


def test_zero_recovers_the_uninjected_head_exactly():
    """`composition_sum` in the artifact is the un-injected control and README quotes it,
    so turning the knob off has to reproduce it rather than merely approximate it."""
    model = MU.rehydrate_composition(_composition_artifact(), keep=8, injected_sigma=0.0)
    assert not model.ps.enabled and model.sigma_source == "none"
    frame = pd.DataFrame({"player_id": [1, 1, 2, 2], "season": "2022-23"})
    assert np.allclose(model.ps.shift(frame, np.arange(8)), 0.0)
