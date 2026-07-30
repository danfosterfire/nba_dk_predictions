import numpy as np
import pandas as pd
import pytest

from src.models.availability import (
    FEATURE_COLS,
    LAG_COLS,
    RHO_MAX,
    WORKLOAD_COLS,
    WORKLOAD_SOURCE_COLS,
    BetaBinomialGLM,
    LeagueAgeBaseline,
    RidgeAvailability,
    assert_point_in_time,
    crps,
    fit_dispersion,
    pit_values,
    predictive_pmf,
    expand_basis,
    minutes_nonlinearity_probe,
    nonlinearity_ablation,
    split_seasons,
    workload_ablation,
)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n_players: int = 400, seed: int = 0, rho: float = 0.2) -> pd.DataFrame:
    """Player-seasons where games played really is beta-binomial around a linear mean.

    Availability tracks prior availability with a lot of noise, which is the shape the
    real data has — the point of a synthetic panel is that the true ρ is known, so a
    calibrated model has something to be checked against.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season_index, season in enumerate(SEASONS):
        for player in range(n_players):
            prior = rng.uniform(0.25, 1.0)
            mu = np.clip(0.35 + 0.55 * prior, 0.05, 0.95)
            scale = (1 - rho) / rho
            p = rng.beta(mu * scale, (1 - mu) * scale)
            rows.append({
                "season": season, "season_index": season_index, "player_id": player,
                "gp": int(rng.binomial(82, p)), "team_games": 82,
                "gp_share_lag1": prior, "gp_share_lag2": prior, "gp_share_lag3": prior,
                "minutes_per_game_lag1": 10 + 20 * prior,
                "minutes_per_game_lag2": 10 + 20 * prior,
                "minutes_per_game_lag3": 10 + 20 * prior,
                "total_minutes_lag1": 1600 * prior, "trailing_missed_lag1": 0.0,
                "end_play_rate_lag1": prior, "n_spells_lag1": 3.0,
                "longest_spell_lag1": 4.0,
                "age": 20 + (player % 15), "career_year": season_index,
                "n_prior_seasons": 3,
                # The playoff/mileage block. Deliberately independent of `prior` so these
                # neither carry nor mask the signal the tests below check for — the point
                # is that the head still fits with them present.
                "playoff_games_lag1": float(rng.integers(0, 20)),
                "playoff_mpg_lag1": rng.uniform(0.0, 36.0),
                "playoff_minutes_share_lag1": rng.uniform(0.0, 0.25),
                "career_minutes_lag1": rng.uniform(500.0, 30000.0),
                "season_start_date": pd.Timestamp(f"20{19 + season_index}-10-20"),
                "as_of_date": pd.Timestamp(f"20{19 + season_index}-10-19"),
            })
    df = pd.DataFrame(rows)
    df["age_sq"] = df["age"] ** 2
    return df


# ── The leakage assertion ─────────────────────────────────────────────────────

def test_point_in_time_assertion_rejects_a_row_dated_into_its_own_season():
    """The failure this guards is silent: a design matrix filled from a current-status
    source looks completely normal and simply scores too well."""
    design = _design(20).head(10).copy()
    design.loc[design.index[3], "as_of_date"] = design["season_start_date"].iloc[3]
    with pytest.raises(ValueError, match="as_of_date"):
        assert_point_in_time(design)


def test_point_in_time_assertion_passes_a_clean_design():
    design = _design(20).head(10)
    assert assert_point_in_time(design) is design


def test_split_seasons_holds_out_the_trailing_seasons():
    train, test = split_seasons(_design(50), test_seasons=2)
    assert set(test["season"]) == {"2021-22", "2022-23"}
    assert set(train["season"]) == {"2019-20", "2020-21"}
    # Nothing from a test season leaks into training.
    assert not set(train["season"]) & set(test["season"])


# ── The predictive distribution ───────────────────────────────────────────────

def test_predictive_pmf_is_a_distribution():
    pmf = predictive_pmf(np.array([82, 66]), np.array([0.8, 0.5]), 0.2, max_games=82)
    assert np.allclose(pmf.sum(axis=1), 1.0)
    # A 66-game season puts no mass above 66.
    assert pmf[1, 67:].sum() == pytest.approx(0.0, abs=1e-12)


def test_fit_dispersion_recovers_a_known_rho():
    rng = np.random.default_rng(3)
    n = np.full(4000, 82)
    mu = np.full(4000, 0.75)
    rho = 0.2
    scale = (1 - rho) / rho
    y = rng.binomial(n, rng.beta(mu * scale, (1 - mu) * scale))
    assert fit_dispersion(y, n, mu) == pytest.approx(rho, abs=0.05)


def test_fit_dispersion_stays_finite_when_a_player_beat_the_schedule():
    """13 traded players (0.12%) played more games than their last team's schedule.
    `gp > n` makes the log-density non-finite, and because the likelihood is a sum,
    those rows would take the whole fit down at every value of rho."""
    n = np.full(500, 82)
    y = np.full(500, 70)
    rho = fit_dispersion(y, np.maximum(n, y), np.full(500, 0.8))
    assert np.isfinite(rho) and 0 < rho < RHO_MAX


def test_crps_rewards_a_sharper_correct_distribution():
    y = np.array([60])
    sharp = predictive_pmf(np.array([82]), np.array([60 / 82]), 0.02, max_games=82)
    vague = predictive_pmf(np.array([82]), np.array([60 / 82]), 0.40, max_games=82)
    assert crps(sharp, y)[0] < crps(vague, y)[0]


def test_crps_punishes_a_confidently_wrong_mean():
    y = np.array([30])
    right = predictive_pmf(np.array([82]), np.array([0.37]), 0.05, max_games=82)
    wrong = predictive_pmf(np.array([82]), np.array([0.95]), 0.05, max_games=82)
    assert crps(right, y)[0] < crps(wrong, y)[0]


def test_pit_is_uniform_under_a_correctly_specified_model():
    """The whole reason PIT is randomized: the non-randomized version of a discrete
    predictive is not uniform even when the model is perfect."""
    rng = np.random.default_rng(7)
    n = np.full(4000, 82)
    mu = rng.uniform(0.4, 0.95, size=4000)
    rho = 0.2
    scale = (1 - rho) / rho
    y = rng.binomial(n, rng.beta(mu * scale, (1 - mu) * scale))
    u = pit_values(predictive_pmf(n, mu, rho, 82), y, seed=1)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    assert np.max(np.abs(counts / len(u) - 0.1)) < 0.02


def test_pit_detects_an_overconfident_model():
    rng = np.random.default_rng(9)
    n = np.full(3000, 82)
    mu = np.full(3000, 0.75)
    scale = (1 - 0.25) / 0.25
    y = rng.binomial(n, rng.beta(mu * scale, (1 - mu) * scale))
    # Claim far less dispersion than the data has: the PIT piles up at both ends.
    u = pit_values(predictive_pmf(n, mu, 0.01, 82), y, seed=1)
    counts, _ = np.histogram(u, bins=10, range=(0, 1))
    assert (counts[0] + counts[-1]) / len(u) > 0.4


# ── The models ────────────────────────────────────────────────────────────────

def test_models_fit_and_emit_a_normalized_distribution():
    train, test = split_seasons(_design(300), 2)
    for model in (LeagueAgeBaseline(), RidgeAvailability(), BetaBinomialGLM()):
        model.fit(train)
        pmf = model.predict_pmf(test, 82)
        assert np.allclose(pmf.sum(axis=1), 1.0)
        assert 0 < model.rho < RHO_MAX
        assert ((model.predict_mean(test) > 0) & (model.predict_mean(test) < 1)).all()


def test_the_glm_recovers_signal_a_numeric_gradient_fit_would_miss():
    """The objective is a log-likelihood summed over thousands of rows, so its magnitude
    swamps a finite-difference step and L-BFGS-B stops on gradient noise with
    coefficients near zero. The analytic gradient is what makes this fit work."""
    train, test = split_seasons(_design(400), 2)
    glm = BetaBinomialGLM().fit(train)
    mu = glm.predict_mean(test)
    truth = test["gp"].to_numpy() / test["team_games"].to_numpy()
    # The mean function must track availability, not collapse to a constant.
    assert np.corrcoef(mu, truth)[0, 1] > 0.4
    assert mu.std() > 0.05


def test_the_glm_beats_the_league_age_baseline_on_crps_when_signal_exists():
    """Not a claim about the real data — a control that the comparison can detect a
    difference at all, on a panel built to have one."""
    train, test = split_seasons(_design(400), 2)
    y = test["gp"].to_numpy()
    glm = BetaBinomialGLM().fit(train)
    baseline = LeagueAgeBaseline().fit(train)
    assert (crps(glm.predict_pmf(test, 82), y).mean()
            < crps(baseline.predict_pmf(test, 82), y).mean())


def test_feature_columns_are_all_present_in_the_design():
    design = _design(20)
    assert not set(FEATURE_COLS) - set(design.columns)


# ── The playoff/mileage workload block ────────────────────────────────────────

def test_workload_cols_are_in_feature_cols_and_derive_from_their_sources():
    assert set(WORKLOAD_COLS) <= set(FEATURE_COLS)
    assert WORKLOAD_SOURCE_COLS == [c.removesuffix("_lag1") for c in WORKLOAD_COLS]
    # Every source column must be lagged by build_design, or the lag1 name never exists.
    assert set(WORKLOAD_SOURCE_COLS) <= set(LAG_COLS)


def test_total_minutes_incl_playoffs_is_deliberately_not_a_feature():
    """It is a *worse* predictor than regular-season total_minutes (in-sample R² 0.2816
    vs 0.2843) because it mixes team quality into a workload measure. Recorded as a test
    so it does not get 'fixed' back in."""
    assert "total_minutes_incl_playoffs_lag1" not in FEATURE_COLS
    assert "total_minutes_incl_playoffs" not in LAG_COLS
    assert "total_minutes_lag1" in FEATURE_COLS


def test_workload_ablation_contrasts_only_the_feature_list():
    design = _design(300)
    train, test = split_seasons(design, test_seasons=2)
    out = workload_ablation(train, test, max_games=82, seed=0)

    assert set(out["variant"]) == {"baseline", "plus_playoff_workload",
                                   "plus_playoff_only", "plus_career_minutes_only"}
    counts = out.set_index("variant")["n_features"]
    assert counts["baseline"] == len(FEATURE_COLS) - len(WORKLOAD_COLS)
    assert counts["plus_playoff_workload"] == len(FEATURE_COLS)
    assert counts["plus_career_minutes_only"] == counts["baseline"] + 1
    # The baseline row is the reference, so its own delta is exactly zero.
    assert out.set_index("variant").loc["baseline", "crps_vs_baseline"] == 0.0


def test_workload_ablation_shows_no_gain_when_the_block_is_noise():
    """The fixture's workload columns are independent of the target, so a real ablation
    must not credit them with a material improvement."""
    design = _design(300)
    train, test = split_seasons(design, test_seasons=2)
    out = workload_ablation(train, test, max_games=82, seed=0).set_index("variant")
    assert out.loc["plus_playoff_workload", "crps_vs_baseline"] < 0.25


# ── Nonlinear response ────────────────────────────────────────────────────────

def test_expand_basis_fits_knots_on_train_only():
    """Knot placement from the pooled frame leaks the test distribution into the basis."""
    train = pd.DataFrame({"x": np.linspace(0.0, 1.0, 200)})
    test = pd.DataFrame({"x": np.linspace(5.0, 6.0, 50)})   # far outside train's range
    tr, te, names = expand_basis(train, test, ["x"], kind="spline", n_knots=4)
    assert len(names) == tr[names].shape[1] == te[names].shape[1]
    # Linear extrapolation, so out-of-range rows stay finite and are not clipped to the
    # training basis values.
    assert np.isfinite(te[names].to_numpy()).all()
    assert not np.allclose(te[names].to_numpy()[0], tr[names].to_numpy()[-1])


def test_expand_basis_quadratic_squares_the_column():
    train = pd.DataFrame({"x": np.array([1.0, 2.0, 3.0])})
    test = pd.DataFrame({"x": np.array([4.0])})
    tr, te, names = expand_basis(train, test, ["x"], kind="quadratic")
    assert names == ["x__sq"]
    assert list(tr["x__sq"]) == [1.0, 4.0, 9.0]
    assert list(te["x__sq"]) == [16.0]


def test_expand_basis_rejects_an_unknown_kind():
    df = pd.DataFrame({"x": [1.0, 2.0]})
    with pytest.raises(ValueError, match="quadratic"):
        expand_basis(df, df, ["x"], kind="cubic")


def test_nonlinearity_ablation_selects_on_validation_not_test():
    """The protocol is the point: `selected` must key off the validation column, so a
    variant that only wins on test is not chosen."""
    design = _design(400)
    train, test = split_seasons(design, test_seasons=2)
    out = nonlinearity_ablation(train, test, max_games=82, seed=0,
                                cols=["age", "minutes_per_game_lag1"])
    assert {"linear", "quadratic", "spline_k4", "spline_k5"} == set(out["variant"])
    assert out["selected"].sum() == 1
    best_val = out.loc[out["val_crps_games"].idxmin(), "variant"]
    assert out.loc[out["selected"], "variant"].iloc[0] == best_val
    # The linear row is the reference on both columns.
    lin = out.set_index("variant").loc["linear"]
    assert lin["val_vs_linear"] == 0.0 and lin["test_vs_linear"] == 0.0


def test_nonlinearity_ablation_finds_no_material_gain_on_a_linear_generator():
    """`_design` builds mu linear in `prior`, so a curved basis has nothing real to find.

    The assertion is on *magnitude*, not on which variant is selected. With one small
    validation season a curved variant can still win by chance — which is the exact
    phenomenon this function exists to surface, so asserting "linear is always selected"
    would be asserting that selection is noiseless.
    """
    train, test = split_seasons(_design(400), test_seasons=2)
    out = nonlinearity_ablation(train, test, max_games=82, seed=0,
                               cols=["age", "minutes_per_game_lag1", "gp_share_lag1"])
    # No curved variant may show a large validation improvement over linear.
    curved = out[out["variant"] != "linear"]
    assert curved["val_vs_linear"].min() > -0.25


def test_minutes_probe_reports_variants_and_per_column_attribution():
    design = _design(400)
    design["minutes_per_game"] = (0.8 * design["minutes_per_game_lag1"]
                                  + np.random.default_rng(0).normal(0, 2, len(design)))
    train, test = split_seasons(design, test_seasons=2)
    cols = ["age", "minutes_per_game_lag1"]
    out = minutes_nonlinearity_probe(train, test, cols=cols)

    assert set(out.loc[out["scope"] == "variant", "name"]) == {
        "linear", "quadratic", "spline_k4", "spline_k5"}
    assert set(out.loc[out["scope"] == "column", "name"]) == set(cols)
    # Deltas are only defined for the per-column rows.
    assert out.loc[out["scope"] == "variant", "val_vs_linear"].isna().all()
    assert out.loc[out["scope"] == "column", "val_vs_linear"].notna().all()
    # `replicates` requires both splits to agree in sign.
    col = out[out["scope"] == "column"]
    for _, r in col.iterrows():
        assert bool(r["replicates"]) == (r["val_vs_linear"] > 0 and r["test_vs_linear"] > 0)
