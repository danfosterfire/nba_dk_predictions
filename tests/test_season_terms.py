"""Tests for the season-term ablation: the trend covariate and the year random effect.

The property that has to hold everywhere in this module is that the two forms do
*different jobs* — a trend moves the mean, a year effect moves only the spread — so most
of what follows pins one of them not doing the other's job. Synthetic builders throughout,
with series constructed to have a known trend and a known shock.

Split the same way `test_stan_heads.py` is: almost everything needs no sampler, and the
handful that genuinely need CmdStan are skipped rather than failed when it is absent.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda.season_effects import regime_tests, series_label, year_shock_correlation
from src.models.season_terms import (SEASON_TREND, _finalize, add_role_terms, add_trend,
                                     apply_override, availability_arms,
                                     coverage_from_pmf, coverage_from_samples,
                                     oracle_override, realized_season_dk, season_arms)
from src.models.stan_utils import YearTerm, year_block


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


def _frame(seasons, mpg=None):
    rows = []
    for i, s in enumerate(seasons):
        rows.append({"player_id": 100 + i, "season": s,
                     "minutes_per_game_lag1": 20.0 if mpg is None else mpg[i]})
    return pd.DataFrame(rows)


# ── The trend covariate ───────────────────────────────────────────────────────

def test_trend_is_centred_on_train_so_test_seasons_extrapolate():
    """Centering on the pooled frame would put the held-out seasons INSIDE the fitted
    range, which is not what extrapolating one season forward means."""
    train = _frame(["2016-17", "2017-18", "2018-19"])
    test = _frame(["2019-20", "2020-21"])
    (tr, te), names = add_trend(train, [train, test])

    assert names == [SEASON_TREND]
    assert abs(tr[SEASON_TREND].mean()) < 1e-12
    # Every test row is strictly beyond the training range — the definition of extrapolation.
    assert te[SEASON_TREND].min() > tr[SEASON_TREND].max()


def test_trend_is_one_unit_per_season_so_its_coefficient_is_a_per_season_rate():
    train = _frame(["2016-17", "2017-18", "2018-19", "2019-20"])
    (tr,), _ = add_trend(train, [train])
    steps = np.diff(np.sort(tr[SEASON_TREND].unique()))
    assert np.allclose(steps, 1.0)


def test_trend_handles_the_century_rollover_in_the_season_label():
    """"1999-00" must sort and difference as 1999, not as anything the "-00" suffix
    tempts. The season label is the only shared time key across the designs."""
    train = _frame(["1998-99", "1999-00", "2000-01"])
    (tr,), _ = add_trend(train, [train])
    assert np.allclose(np.sort(tr[SEASON_TREND].to_numpy()), [-1.0, 0.0, 1.0])


# ── The role interaction ──────────────────────────────────────────────────────

def test_role_terms_use_prior_season_mpg_and_leave_the_lowest_bucket_as_reference():
    train = _frame(["2016-17"] * 4, mpg=[5.0, 18.0, 27.0, 35.0])
    (tr,), _ = add_trend(train, [train])
    (out,), names = add_role_terms(tr, [tr])

    # Three dummies for four buckets: the lowest is the reference, so `trend` keeps its
    # meaning as that bucket's slope.
    dummies = [n for n in names if not n.endswith("__x__trend")]
    assert len(dummies) == 3
    assert all(out[d].sum() == 1 for d in dummies)
    # The fringe player is in no dummy at all.
    assert out.loc[0, dummies].sum() == 0


def test_role_interaction_is_exactly_the_dummy_times_the_trend():
    train = _frame(["2016-17"] * 2 + ["2018-19"] * 2, mpg=[5.0, 35.0, 5.0, 35.0])
    (tr,), _ = add_trend(train, [train])
    (out,), names = add_role_terms(tr, [tr])
    for dummy in [n for n in names if not n.endswith("__x__trend")]:
        assert np.allclose(out[f"{dummy}__x__trend"],
                           out[dummy] * out[SEASON_TREND])


def test_role_terms_never_read_the_target_seasons_mpg():
    """Bucketing on the season being forecast would be the outcome in the design."""
    train = _frame(["2016-17"] * 2, mpg=[5.0, 35.0])
    train["minutes_per_game"] = [35.0, 5.0]      # the target season, deliberately reversed
    (tr,), _ = add_trend(train, [train])
    (out,), names = add_role_terms(tr, [tr])
    heavy = [n for n in names if n.startswith("role_30")][0]
    # Row 1 is the heavy-minute player by PRIOR mpg; if the target column were read it
    # would be row 0.
    assert out.loc[1, heavy] == 1.0 and out.loc[0, heavy] == 0.0


# ── The arms ──────────────────────────────────────────────────────────────────

def test_the_year_arm_differs_from_base_ONLY_by_the_year_block():
    """If the feature lists differed too, the contrast would not isolate the year term."""
    train, test = _frame(["2016-17", "2017-18"]), _frame(["2018-19"])
    arms = season_arms(train, test, ["minutes_per_game_lag1"])
    assert arms["base"][2] == arms["year"][2]
    assert arms["base"][3] is None and arms["year"][3] == "season"


def test_the_trend_arm_adds_exactly_one_column_and_no_year_block():
    train, test = _frame(["2016-17", "2017-18"]), _frame(["2018-19"])
    arms = season_arms(train, test, ["minutes_per_game_lag1"])
    assert arms["trend"][2] == ["minutes_per_game_lag1", SEASON_TREND]
    assert arms["trend"][3] is None
    assert arms["trend_year"][2] == arms["trend"][2]
    assert arms["trend_year"][3] == "season"


def test_availability_gets_the_role_arms_and_they_carry_the_trend():
    train = _frame(["2016-17", "2017-18"], mpg=[5.0, 35.0])
    test = _frame(["2018-19"], mpg=[30.0])
    arms = availability_arms(train, test, ["minutes_per_game_lag1"])
    assert "trend_x_role" in arms and "trend_x_role_year" in arms
    features = arms["trend_x_role"][2]
    assert SEASON_TREND in features
    assert any(f.endswith("__x__trend") for f in features)
    assert arms["trend_x_role"][3] is None
    assert arms["trend_x_role_year"][3] == "season"


# ── The year term ─────────────────────────────────────────────────────────────

def test_a_disabled_year_term_is_the_S_equals_zero_block_and_contributes_nothing():
    term = YearTerm(None)
    block = term.data(_frame(["2016-17"] * 5))
    assert block["S"] == 0
    assert set(block["season_idx"]) == {0}
    assert np.allclose(term.shift(np.arange(10)), 0.0)
    assert term.response_multiplier() == 1.0
    assert term.summary()["year_effect"] is False


def test_an_enabled_year_term_indexes_only_the_training_seasons():
    term = YearTerm("season")
    block = term.data(_frame(["2016-17", "2017-18", "2016-17"]))
    assert block["S"] == 2
    assert block["season_idx"] == [1, 2, 1]
    assert term.levels == ["2016-17", "2017-18"]


def test_a_held_out_season_maps_outside_the_fitted_levels_rather_than_into_them():
    """Mapping a forecast season onto a fitted `z` would be a season FIXED effect, which
    is the one form that cannot exist at prediction time."""
    block, levels = year_block(3, np.array(["2016-17", "2017-18", "2025-26"]),
                               levels=["2016-17", "2017-18"])
    assert block["S"] == 2
    assert block["season_idx"] == [1, 2, 0]      # 0 is the index the model never reads
    assert levels == ["2016-17", "2017-18"]


def test_the_year_shift_is_mean_zero_and_one_value_per_draw():
    term = YearTerm("season")
    term.sigma_draws = np.full(4000, 0.05)
    shift = term.shift(np.arange(4000))
    assert shift.shape == (4000,)
    assert abs(shift.mean()) < 0.005          # mean-zero by construction
    assert 0.045 < shift.std() < 0.055        # and its sd IS sigma


def test_two_heads_draw_independent_year_effects():
    """A shared stream would make every head's year effect perfectly correlated, which the
    measured cross-component shock correlation (mean -0.009) does not support."""
    a, b = YearTerm("season", stream="reb"), YearTerm("season", stream="blk")
    same = YearTerm("season", stream="reb")
    for t in (a, b, same):
        t.sigma_draws = np.ones(2000)
    idx = np.arange(2000)
    assert np.allclose(a.shift(idx), same.shift(idx))          # reproducible
    assert abs(np.corrcoef(a.shift(idx), b.shift(idx))[0, 1]) < 0.1   # independent


def test_the_response_multiplier_reports_the_jensen_inflation_rather_than_assuming_it_away():
    """Mean-zero on the linear predictor is NOT mean-zero on the response: under a log
    link E[exp(sigma*z)] = exp(sigma^2/2) > 1."""
    term = YearTerm("season")
    term.sigma_draws = np.full(100, 0.2)
    assert term.response_multiplier() == pytest.approx(np.exp(0.02), rel=1e-9)
    assert term.response_multiplier() > 1.0


# ── Calibration ───────────────────────────────────────────────────────────────

def test_coverage_is_nominal_for_a_calibrated_predictive_and_low_for_a_tight_one():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 4000)
    calibrated = rng.normal(0, 1, size=(2000, 4000))
    too_tight = rng.normal(0, 0.3, size=(2000, 4000))

    good = coverage_from_samples(calibrated, y)
    bad = coverage_from_samples(too_tight, y)
    assert abs(good["coverage_80"] - 0.80) < 0.02
    assert abs(good["coverage_95"] - 0.95) < 0.02
    assert bad["coverage_80"] < 0.4
    assert bad["width_80"] < good["width_80"]


def test_a_discrete_central_interval_is_CONSERVATIVE_rather_than_exact():
    """Worth pinning because it changes how the availability numbers read.

    A discrete interval contains whole atoms, so it cannot hit a nominal level exactly —
    it over-covers, and the coarser the support the more. On a 20-trial binomial the 80%
    interval covers ~89%. So over-coverage on the games-played head is partly discreteness
    and NOT by itself evidence that the predictive is too wide; only the change between
    arms is comparable, since they share the support.
    """
    from scipy.stats import binom
    n, p = 20, 0.4
    pmf = np.tile(binom.pmf(np.arange(n + 1), n, p), (4000, 1))
    y = np.random.default_rng(1).binomial(n, p, 4000)
    got = coverage_from_pmf(pmf, y)["coverage_80"]
    assert got >= 0.80
    assert got < 0.93


def test_pmf_and_sample_coverage_agree_once_the_support_is_fine():
    """The discreteness gap closes as the atoms get small against the spread, which is the
    regime the games-played head is actually in (support 0..82)."""
    from scipy.stats import binom
    n, p = 400, 0.4
    pmf = np.tile(binom.pmf(np.arange(n + 1), n, p), (4000, 1))
    y = np.random.default_rng(1).binomial(n, p, 4000)
    samples = np.random.default_rng(2).binomial(n, p, size=(4000, 4000))
    assert abs(coverage_from_pmf(pmf, y)["coverage_80"]
               - coverage_from_samples(samples, y)["coverage_80"]) < 0.02


# ── The league-level override ─────────────────────────────────────────────────

def test_the_oracle_override_zeroes_each_seasons_bias_independently():
    pred = np.array([10.0, 20.0, 10.0, 20.0])
    y = np.array([12.0, 24.0, 8.0, 16.0])
    seasons = np.array(["a", "a", "b", "b"])
    out = oracle_override(pred, y, seasons)
    for s in ("a", "b"):
        m = seasons == s
        assert out[m].sum() == pytest.approx(y[m].sum())


def test_the_oracle_override_is_a_single_scalar_per_season_and_preserves_ranking():
    """It is the most general form a league-level term can take, which is why it bounds
    the fitted ones — but it cannot reorder players within a season."""
    pred = np.array([5.0, 30.0, 12.0])
    y = np.array([9.0, 20.0, 14.0])
    seasons = np.array(["a", "a", "a"])
    out = oracle_override(pred, y, seasons)
    assert np.allclose(out / pred, (out / pred)[0])
    assert list(np.argsort(out)) == list(np.argsort(pred))


def test_the_manual_override_is_a_no_op_when_unset_and_scopes_to_its_own_season():
    pred = np.array([10.0, 10.0])
    seasons = np.array(["2024-25", "2025-26"])
    assert np.allclose(apply_override(pred, "fta", seasons, {}), pred)
    got = apply_override(pred, "fta", seasons, {"fta": {"2025-26": 1.086}})
    assert got[0] == 10.0 and got[1] == pytest.approx(10.86)
    # A different component is untouched.
    assert np.allclose(apply_override(pred, "blk", seasons, {"fta": {"2025-26": 1.086}}),
                       pred)


# ── Selection discipline ──────────────────────────────────────────────────────

def _table():
    return pd.DataFrame([
        {"head": "reb", "arm": "carry_forward", "val_crps": np.nan, "test_crps": 10.0},
        {"head": "reb", "arm": "base", "val_crps": 9.0, "test_crps": 9.5},
        {"head": "reb", "arm": "year", "val_crps": 9.5, "test_crps": 8.0},
        {"head": "reb", "arm": "oracle_league", "val_crps": np.nan, "test_crps": np.nan},
    ])


def test_selection_reads_validation_and_ignores_a_better_test_column():
    """`year` wins on test and loses on validation; the shipped choice must be `base`."""
    out = _finalize(_table(), "val_crps", "test_crps", higher_is_better=False)
    assert out.loc[out["arm"] == "base", "selected"].item()
    assert not out.loc[out["arm"] == "year", "selected"].item()


def test_beats_floor_is_a_test_fact_and_is_NA_where_there_is_no_test_metric():
    out = _finalize(_table(), "val_crps", "test_crps", higher_is_better=False)
    assert out.loc[out["arm"] == "base", "beats_floor"].item() is True
    assert out.loc[out["arm"] == "year", "beats_floor"].item() is True
    # The oracle is a point prediction with no predictive distribution. Recording it as
    # failing the floor would be a fabricated failure.
    assert pd.isna(out.loc[out["arm"] == "oracle_league", "beats_floor"].item())


def test_the_floor_row_is_never_selected_even_when_it_would_win():
    table = _table()
    table.loc[table["arm"] == "carry_forward", "val_crps"] = 0.1
    out = _finalize(table, "val_crps", "test_crps", higher_is_better=False)
    assert not out.loc[out["arm"] == "carry_forward", "selected"].item()


# ── Season-total reassembly ───────────────────────────────────────────────────

def test_realized_season_dk_uses_the_fg2m_basis_and_matches_dk_weights():
    """`pts = 2*fg2m + 3*fg3m + ftm`. The tempting `2*fgm + 3*fg3m + ftm` pays a made
    three 2 + 3 = 5 and is wrong on every game with a made three."""
    design = pd.DataFrame([{"fg2m": 100.0, "fg3m": 50.0, "ftm": 80.0, "reb": 200.0,
                            "ast": 150.0, "stl": 40.0, "blk": 30.0, "tov": 60.0}])
    pts = 2 * 100 + 3 * 50 + 80
    expected = (pts + 0.5 * 50 + 1.25 * 200 + 1.5 * 150 + 2 * 40 + 2 * 30 - 0.5 * 60)
    assert realized_season_dk(design)[0] == pytest.approx(expected)


def test_realized_season_dk_agrees_with_compute_dk_pts_minus_the_bonus():
    from src.data.preprocess import compute_dk_pts
    design = pd.DataFrame([{"fg2m": 3.0, "fg3m": 1.0, "ftm": 2.0, "reb": 4.0,
                            "ast": 2.0, "stl": 1.0, "blk": 0.0, "tov": 1.0}])
    row = design.assign(pts=2 * design["fg2m"] + 3 * design["fg3m"] + design["ftm"])
    # No category clears 10, so the bonus is 0 and the two must agree exactly.
    assert compute_dk_pts(row)[0] == pytest.approx(realized_season_dk(design)[0])


# ── Composing the eleven heads ────────────────────────────────────────────────

class _StubCount:
    """A count head whose predictive is a fixed mean plus an optional shared shock.

    `shared` is the year effect's defining property: one draw applied to every row alike.
    """

    def __init__(self, mean, rows, shared=0.0, seed=0):
        self.mean, self.rows, self.shared, self.seed = mean, rows, shared, seed
        self.predictive_samples = 100
        self.features = []

    def predict_samples(self, frame, seed=0):
        rng = np.random.default_rng(self.seed)
        d = self.predictive_samples
        z = rng.standard_normal(d) * self.shared
        return np.clip(self.mean * (1 + z)[:, None]
                       + rng.normal(0, 1, size=(d, self.rows)), 0, None)


class _StubConversion:
    def __init__(self, p, rows, seed=0):
        self.p, self.rows, self.seed = p, rows, seed
        self.rho = 0.01

    def p_draws(self, frame, keep):
        return (np.full((keep, self.rows), self.p), np.full(keep, self.rho))


def _stub_models(rows, arms=("base", "year"), shared=(0.0, 0.05)):
    from src.models.season_terms import CONVERSION_HEADS, COUNT_HEADS
    frame = pd.DataFrame({"season": ["2024-25"] * rows, "gp": np.full(rows, 60.0)})
    models = {}
    for arm, s in zip(arms, shared):
        for i, head in enumerate(COUNT_HEADS):
            models[(head, arm)] = (_StubCount(50.0, rows, s, seed=i), frame)
        for j, (m, a) in enumerate(CONVERSION_HEADS):
            models[(f"{m}|{a}", arm)] = (_StubConversion(0.45, rows, seed=j), frame)
    return models, frame


def test_the_composition_refuses_a_frame_that_is_not_row_aligned():
    """Silent misalignment would pair one player's rebounds with another's assists and
    still produce a plausible season total, which is the worst failure available here."""
    from src.models.season_terms import compose_season_dk
    models, frame = _stub_models(20)
    with pytest.raises(ValueError, match="align"):
        compose_season_dk(models, frame.iloc[:10], "base", draws=20, seed=0)


def test_roster_spread_grows_as_sqrt_N_without_a_year_effect_and_as_N_with_one():
    """The reason season effects matter at all: a shared shock does not diversify.

    Independent per-player error grows as sqrt(N), so its per-player sd *falls*. A shared
    league shift grows as N, so its per-player sd is flat — and the ratio between the two
    arms therefore has to widen with roster size.
    """
    from src.models.season_terms import roster_spread
    models, frame = _stub_models(200)
    out = roster_spread(models, frame, ("base", "year"), sizes=(10, 100),
                        draws=400, n_subsets=40, seed=0)
    infl = out.set_index(["arm", "n_players"])["inflation_vs_base"]
    assert infl[("base", 10)] == pytest.approx(1.0)
    assert infl[("year", 10)] > 1.0
    # The year effect's share of the total grows with the roster, which is the whole point.
    assert infl[("year", 100)] > infl[("year", 10)]


def test_compose_season_dk_puts_makes_on_the_DRAWN_attempts():
    """`makes | attempts` is the chain. Conditioning on realized attempts would leak the
    target and understate the spread, since attempt uncertainty is most of the uncertainty
    in points."""
    from src.models.season_terms import _draw_components
    models, frame = _stub_models(30)
    counts, made = _draw_components(models, frame, "base", draws=50, seed=0)
    for m, a in [("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]:
        assert made[m].shape == counts[a].shape
        assert (made[m] <= np.rint(counts[a]) + 1e-9).all()


# ── The Stan sources ──────────────────────────────────────────────────────────

def _source(name: str) -> str:
    return (Path(__file__).resolve().parents[1] / "src" / "stan" / f"{name}.stan").read_text()


@pytest.mark.parametrize("name", ["betabinomial_glm", "negbinomial_glm"])
def test_the_year_parameters_are_zero_length_when_S_is_zero(name):
    """`S = 0` must disable the term EXACTLY, not approximately — the parameter space has
    to be identical to the model without it, or `base` is not the shipped head."""
    text = _source(name)
    assert "int H = S > 0 ? 1 : 0;" in text
    assert "vector[S] year_z;" in text
    assert "vector<lower=0>[H] sigma_year;" in text


@pytest.mark.parametrize("name", ["betabinomial_glm", "negbinomial_glm"])
def test_the_year_effect_is_non_centred(name):
    """Centred, ~28 groups with a small sigma is a funnel and NUTS handles it badly."""
    text = _source(name)
    assert "year_z ~ std_normal();" in text
    assert "sigma_year[1] * year_z[season_idx]" in text


@needs_cmdstan
@pytest.mark.parametrize("name", ["betabinomial_glm", "negbinomial_glm"])
def test_S_zero_nests_exactly_inside_the_year_effect_model(name):
    """The nesting as an identity rather than an assertion about source text.

    At `sigma_year = 0` the year model's log density must exceed the `S = 0` model's by
    *exactly* the std_normal prior on `z` and nothing else — meaning the likelihood, the
    priors on alpha/beta and the dispersion term are untouched. Stan's `~` drops constants,
    so the check is on the difference, which is constant-free.
    """
    from src.models.stan_utils import compile_model

    rng = np.random.default_rng(0)
    N, K = 150, 2
    X = rng.normal(size=(N, K))
    model = compile_model(name)
    if name == "betabinomial_glm":
        data = {"N": N, "K": K, "X": X, "n": np.full(N, 30).tolist(),
                "y": rng.binomial(30, 0.5, N).tolist(),
                "beta_scale": 1.0, "intercept_scale": 5.0}
        pars = {"alpha": 0.3, "beta": [0.5, -0.2], "rho": 0.12}
    else:
        data = {"N": N, "K": K, "X": X, "y": rng.poisson(4.0, N).tolist(),
                "exposure": np.full(N, 100.0).tolist(), "beta_scale": 1.0,
                "intercept_scale": 5.0, "phi_inv_scale": 1.0}
        pars = {"alpha": -3.2, "beta": [0.5, -0.2], "phi_inv": 0.05}

    seasons = np.array([f"s{i % 4}" for i in range(N)])
    z = [0.7, -0.3, 1.1, 0.2]
    off = {**data, **year_block(N)[0]}
    on = {**data, **year_block(N, seasons)[0]}

    lp = lambda p, d: float(model.log_prob(p, data=d, jacobian=False).iloc[0, 0])
    gap = lp({**pars, "year_z": z, "sigma_year": [0.0]}, on) - lp(pars, off)
    assert gap == pytest.approx(-0.5 * float(np.sum(np.square(z))), abs=1e-4)


# ── The league-series regime tests ────────────────────────────────────────────

def _series(values, quantity="fta", start=1996):
    return pd.DataFrame({
        "season": [f"{start + i}-{str(start + i + 1)[2:]}" for i in range(len(values))],
        "quantity": quantity, "rate": values})


def test_a_level_break_is_detected_and_a_clean_trend_is_not():
    """The break test has to separate a real policy step from a smooth series, or it
    cannot do the job it exists for."""
    n = 30
    clean = np.exp(0.02 * np.arange(n))
    stepped = clean.copy()
    broke_at = 27                                    # 2023-24, the PPP season
    stepped[broke_at:] *= 1.15

    out_clean = regime_tests(_series(clean), break_season="2023-24", group="quantity")
    out_step = regime_tests(_series(stepped), break_season="2023-24", group="quantity")
    level = lambda t: t[t["arm"] == "policy_break_level"].iloc[0]

    assert level(out_step)["p_value"] < 0.01
    assert level(out_step)["regime_effect_pct"] == pytest.approx(15.0, abs=1.5)
    assert level(out_clean)["p_value"] > 0.05
    assert abs(level(out_clean)["next_season_shift_pct"]) < 1.0


def test_the_covid_arm_is_an_indicator_that_does_not_carry_into_the_forecast():
    """COVID is a transient regime; modelling it as a permanent shift would put the whole
    post-2021 series on the wrong intercept."""
    n = 30
    values = np.ones(n)
    values[23:25] *= 1.10                            # 2019-20 and 2020-21
    out = regime_tests(_series(values), group="quantity")
    row = out[out["arm"] == "covid_indicator"].iloc[0]
    assert row["regime_effect_pct"] == pytest.approx(10.0, abs=1.0)
    # The forecast season is not in the regime, so the +10% must not appear in it.
    assert abs(row["next_season_shift_pct"]) < 3.0


def test_regime_tests_report_how_many_seasons_the_regime_term_is_fitted_on():
    """`next_season_shift_pct` off three post-break seasons is noise, and the column that
    says so has to be in the artifact rather than in a reader's head."""
    out = regime_tests(_series(np.exp(0.02 * np.arange(30))), group="quantity")
    assert (out[out["arm"] == "policy_break"]["regime_seasons"] == 3).all()


def test_shock_correlation_detrends_so_two_rising_series_do_not_correlate_through_drift():
    n = 30
    t = np.arange(n)
    rng = np.random.default_rng(0)
    a = np.exp(0.05 * t + rng.normal(0, 0.01, n))
    b = np.exp(0.05 * t + rng.normal(0, 0.01, n))     # same trend, independent shocks
    rates = pd.concat([_series(a, "a"), _series(b, "b")], ignore_index=True)
    out = year_shock_correlation(rates, group="quantity")
    assert len(out) == 1
    assert abs(out["corr"].iloc[0]) < 0.5             # drift removed, shocks independent


def test_shock_correlation_finds_a_genuinely_shared_shock():
    n = 30
    rng = np.random.default_rng(1)
    shock = rng.normal(0, 0.03, n)
    rates = pd.concat([_series(np.exp(shock), "a"),
                       _series(np.exp(shock), "b")], ignore_index=True)
    out = year_shock_correlation(rates, group="quantity")
    assert out["corr"].iloc[0] > 0.95


def test_series_label_keeps_the_availability_role_series_apart():
    """`gp_share` is emitted once per role bucket, so `quantity` alone repeats a season
    five times and any per-series regression silently fits a smear of five curves."""
    rates = pd.DataFrame({
        "season": ["2016-17", "2016-17", "2016-17"],
        "quantity": ["gp_share", "gp_share", "fta"],
        "role": ["all", "30+ mpg", np.nan],
        "rate": [0.8, 0.9, 5.0]})
    out = series_label(rates)
    assert out["series"].tolist() == ["gp_share [all]", "gp_share [30+ mpg]", "fta"]
    assert out["series"].nunique() == 3
