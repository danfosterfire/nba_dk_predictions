import numpy as np
import pandas as pd
import pytest

from src.models.component_rates import (
    ALPHA_VARIANTS,
    COUNT_HEADS,
    CONVERSION_HEADS,
    DERIVED_COUNTS,
    MIN_PRIOR_MINUTES,
    POISSON_ALPHA,
    _r2,
    add_log,
    alpha_feature_sets,
    alpha_sensitivity,
    add_spline,
    build_design,
    carry_forward,
    carry_forward_conversion,
    count_variants,
    evaluate,
    fit_count_head,
    fit_nb_dispersion,
    impute,
    matrix_feature_cols,
    nb_nll,
    season_totals,
    split_seasons,
    walk_forward_pca,
)

SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]


# ── Synthetic builders ────────────────────────────────────────────────────────

def _games(n_players: int = 60, seed: int = 0) -> pd.DataFrame:
    """Player-games whose per-36 rates persist across seasons, as the real data does."""
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(n_players):
        level = rng.uniform(0.4, 1.6)
        for season in SEASONS:
            for game in range(30):
                minutes = float(rng.uniform(12.0, 34.0))
                row = {"player_id": player, "season": season, "min": minutes,
                       "played": 1}
                for c in COUNT_HEADS:
                    row[c] = float(rng.poisson(level * minutes / 12.0))
                # Walk the conversion chain in order, materializing derived trials as it
                # goes — under the shot-attempt basis `fg3a` is drawn as a share of `fga`
                # and `fg2a` is the remainder, so neither is a count head. Written
                # basis-agnostically so the fixture does not have to be rewritten again.
                for made, att in CONVERSION_HEADS:
                    if att not in row and att in DERIVED_COUNTS:
                        total, part = DERIVED_COUNTS[att]
                        row[att] = max(row[total] - row[part], 0.0)
                    row[made] = float(rng.binomial(int(row[att]), 0.45))
                for name, (total, part) in DERIVED_COUNTS.items():
                    row.setdefault(name, max(row[total] - row[part], 0.0))
                rows.append(row)
    return pd.DataFrame(rows)


def _write_bios(tmp_path, n_players: int = 60):
    """`load_ages` reads `player_bio_stats_<slug>.csv`; without it every row's age is NaN
    and `build_design` drops the entire design."""
    from src.data.fetch import _slug
    for season in SEASONS:
        pd.DataFrame({"PLAYER_ID": range(n_players),
                      "AGE": [22 + (i % 14) for i in range(n_players)]}).to_csv(
            tmp_path / f"player_bio_stats_{_slug(season)}.csv", index=False)
    return tmp_path


def _matrix(design_seasons: list[str], n_players: int = 60,
            seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for player in range(n_players):
        for season in design_seasons:
            row = {"player_id": player, "season": season,
                   "season_start_year": int(season[:4]), "gp": 30, "min": 24.0}
            for j in range(12):
                row[f"bas_x{j}"] = float(rng.normal())
            rows.append(row)
    return pd.DataFrame(rows)


# ── The benchmark ─────────────────────────────────────────────────────────────

def test_carry_forward_is_prior_rate_times_actual_minutes():
    frame = pd.DataFrame({"reb_p36_lag1": [9.0, 4.5], "total_minutes": [1800.0, 720.0]})
    got = carry_forward(frame, "reb")
    assert np.allclose(got, [9.0 * 1800 / 36, 4.5 * 720 / 36])


def test_carry_forward_uses_no_training_data_at_all():
    """It takes no `train` argument by design — that is what makes it a floor."""
    frame = pd.DataFrame({"ast_p36_lag1": [5.0], "total_minutes": [1000.0]})
    assert carry_forward(frame, "ast")[0] == pytest.approx(5.0 * 1000 / 36)


def test_carry_forward_never_returns_a_negative_count():
    frame = pd.DataFrame({"blk_p36_lag1": [-1.0, 0.0], "total_minutes": [500.0, 500.0]})
    assert (carry_forward(frame, "blk") >= 0).all()


def test_conversion_floor_shrinks_a_zero_for_three_toward_the_league_mean():
    """The whole reason the conversion floor is not a raw carry-forward: an unshrunk
    0.000 prior on 200 new attempts gives a beta-binomial NLL around 1e9."""
    rng = np.random.default_rng(0)
    n = 400
    train = pd.DataFrame({
        "fg3a_lag1": rng.integers(20, 400, n).astype(float),
        "fg3a": rng.integers(20, 400, n).astype(float)})
    train["fg3m_lag1"] = np.floor(train["fg3a_lag1"] * 0.35)
    train["fg3m"] = np.floor(train["fg3a"] * 0.35)
    test = pd.DataFrame({"fg3m_lag1": [0.0], "fg3a_lag1": [3.0],
                         "fg3m": [70.0], "fg3a": [200.0]})
    p = carry_forward_conversion(train, test, "fg3m", "fg3a")
    assert 0.05 < p[0] < 0.40          # pulled well away from 0.000
    assert np.isfinite(p).all()


def test_conversion_floor_barely_moves_a_high_volume_prior():
    """Shrinkage must be attempt-weighted: 2000 prior attempts should stay near their own
    rate even though the same constant rescues the 3-attempt case above."""
    rng = np.random.default_rng(1)
    n = 400
    train = pd.DataFrame({"fta_lag1": rng.integers(50, 500, n).astype(float),
                          "fta": rng.integers(50, 500, n).astype(float)})
    true = rng.uniform(0.55, 0.92, n)          # real between-player spread in FT%
    train["ftm_lag1"] = np.floor(train["fta_lag1"] * true)
    train["ftm"] = np.floor(train["fta"] * true)
    test = pd.DataFrame({"ftm_lag1": [1800.0], "fta_lag1": [2000.0],
                         "ftm": [400.0], "fta": [500.0]})
    p = carry_forward_conversion(train, test, "ftm", "fta")
    assert p[0] > 0.85                 # own rate is 0.90; shrink must not erase it


# ── The regularization trap ───────────────────────────────────────────────────

def test_poisson_alpha_is_tiny_because_of_the_weighted_objective():
    """`PoissonRegressor` averages the deviance by the weight sum, so with minutes as
    weights an alpha near 1 is an enormous penalty. Pinned so it cannot drift back."""
    assert POISSON_ALPHA <= 1e-6


def test_a_large_alpha_loses_to_the_no_fit_floor(tmp_path):
    """The trap, exercised: over-regularization is silent, and the floor is what exposes
    it. This is the guard `run` prints a warning for."""
    n_players = 60
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    train, test = split_seasons(design, test_seasons=1)
    own = "reb_p36_lag1"
    tr, te, flags = impute(train, test, [own, "mpg_lag1", "total_minutes_lag1", "gp_lag1"])
    feats = [own, "mpg_lag1", "total_minutes_lag1", "gp_lag1"] + flags

    y = te["reb"].to_numpy(float)
    floor = carry_forward(te, "reb")
    def r2(p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()

    mu_ok, _ = fit_count_head(tr, te, feats, "reb", alpha=POISSON_ALPHA)
    mu_bad, _ = fit_count_head(tr, te, feats, "reb", alpha=1.0)
    assert r2(mu_ok) > r2(mu_bad)
    assert r2(mu_bad) < r2(floor)      # the crushed fit loses to arithmetic


# ── Walk-forward PCA ──────────────────────────────────────────────────────────

def test_walk_forward_pca_never_sees_the_target_season(tmp_path):
    """A PCA fitted on all seasons leaks the future into the basis, invisibly. The guard
    is that each target season's basis is fitted only on S-1 and earlier."""
    n_players = 60
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    matrix = _matrix(SEASONS, n_players)

    scored, pc_names = walk_forward_pca(design, matrix, n_components=4)
    assert pc_names == ["pc1", "pc2", "pc3", "pc4"]
    # Corrupting a season's matrix rows must not change PC scores for earlier targets.
    poisoned = matrix.copy()
    last = poisoned["season_start_year"].max()
    cols = [c for c in poisoned.columns if c.startswith("bas_")]
    poisoned.loc[poisoned["season_start_year"] == last, cols] += 500.0
    scored2, _ = walk_forward_pca(design, poisoned, n_components=4)

    key = ["player_id", "season"]
    a = scored[scored["season_start_year"] <= last].set_index(key)["pc1"].dropna()
    b = scored2[scored2["season_start_year"] <= last].set_index(key)["pc1"].dropna()
    early = a.index.intersection(b.index)
    early = [i for i in early if int(i[1][:4]) <= last]
    assert len(early) > 0
    assert np.allclose(a.loc[early].to_numpy(), b.loc[early].to_numpy())


def test_walk_forward_pca_imputes_from_history_only(tmp_path):
    """The fill median is a fitted quantity too, and it used to come from every season.

    The test above cannot see this: its matrix has no missing values, so the imputation
    median is never consulted and poisoning the future changes nothing. Put one NaN in an
    EARLY season's row and the leak becomes reachable — under a whole-matrix median, a
    later season's values move that row's PC score backwards in time.
    """
    n_players = 60
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    matrix = _matrix(SEASONS, n_players)
    cols = [c for c in matrix.columns if c.startswith("bas_")]
    first, last = matrix["season_start_year"].min(), matrix["season_start_year"].max()

    holed = matrix.copy()
    early_rows = holed.index[holed["season_start_year"] == first][:5]
    holed.loc[early_rows, cols[0]] = np.nan

    poisoned = holed.copy()
    poisoned.loc[poisoned["season_start_year"] == last, cols] += 500.0

    a, _ = walk_forward_pca(design, holed, n_components=4)
    b, _ = walk_forward_pca(design, poisoned, n_components=4)

    key = ["player_id", "season"]
    x = a[a["season_start_year"] <= last].set_index(key)["pc1"].dropna()
    y = b[b["season_start_year"] <= last].set_index(key)["pc1"].dropna()
    shared = [i for i in x.index.intersection(y.index) if int(i[1][:4]) <= last]
    assert len(shared) > 0
    assert np.allclose(x.loc[shared].to_numpy(), y.loc[shared].to_numpy())


def test_matrix_feature_cols_excludes_keys_and_volume():
    matrix = _matrix(SEASONS)
    cols = matrix_feature_cols(matrix)
    for banned in ("player_id", "season", "season_start_year", "gp", "min"):
        assert banned not in cols
    assert all(c.startswith("bas_") for c in cols)


def test_walk_forward_pca_raises_rather_than_returning_nothing(tmp_path):
    n_players = 12
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    empty = _matrix(SEASONS, n_players).iloc[:0]
    with pytest.raises(ValueError, match="no rows"):
        walk_forward_pca(design, empty, n_components=4)


# ── Design and helpers ────────────────────────────────────────────────────────

def test_season_totals_collapses_games_and_derives_rates():
    games = _games(n_players=3)
    s = season_totals(games)
    assert len(s) == 3 * len(SEASONS)
    one = s.iloc[0]
    assert np.isclose(one["reb_p36"], one["reb"] / one["total_minutes"] * 36)
    assert np.isclose(one["mpg"], one["total_minutes"] / one["gp"])


def test_design_requires_a_substantial_prior_season(tmp_path):
    n_players = 20
    games = _games(n_players)
    d = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    assert (d["total_minutes_lag1"] >= MIN_PRIOR_MINUTES).all()
    assert (d["total_minutes"] > 0).all()
    # Lag-1 means the first season can never be a target.
    assert SEASONS[0] not in set(d["season"])


def test_impute_uses_train_means_and_flags_the_gap():
    train = pd.DataFrame({"x": [1.0, 3.0, np.nan]})
    test = pd.DataFrame({"x": [np.nan, 5.0]})
    tr, te, flags = impute(train, test, ["x"])
    assert flags == ["x__miss"]
    assert tr["x"].tolist() == [1.0, 3.0, 2.0]      # train mean, not pooled
    assert te["x"].tolist() == [2.0, 5.0]
    assert te["x__miss"].tolist() == [1.0, 0.0]


def test_add_log_is_log1p_so_a_zero_rate_survives():
    train = pd.DataFrame({"r": [0.0, 3.0]})
    tr, _, names = add_log(train, train, ["r"])
    assert names == ["log_r"]
    assert tr["log_r"].tolist() == [0.0, np.log(4.0)]


def test_add_spline_fits_knots_on_train_only():
    train = pd.DataFrame({"x": np.linspace(0, 1, 100)})
    test = pd.DataFrame({"x": np.linspace(9, 10, 20)})
    tr, te, names = add_spline(train, test, ["x"], n_knots=4)
    assert len(names) > 1
    assert np.isfinite(te[names].to_numpy()).all()


def test_nb_dispersion_recovers_a_known_value():
    rng = np.random.default_rng(4)
    mu = np.full(6000, 40.0)
    phi = 8.0
    y = rng.negative_binomial(phi, phi / (phi + mu)).astype(float)
    assert fit_nb_dispersion(y, mu) == pytest.approx(phi, rel=0.25)


def test_nb_nll_is_minimised_at_the_true_mean():
    y = np.array([10.0, 12.0, 11.0])
    good = nb_nll(y, np.full(3, 11.0), 10.0).mean()
    bad = nb_nll(y, np.full(3, 30.0), 10.0).mean()
    assert good < bad


# ── End to end ────────────────────────────────────────────────────────────────

def test_count_variants_keep_the_own_response_raw_under_pca(tmp_path):
    """The PCA replaces every prior-season stat EXCEPT the head's own response variable."""
    n_players = 20
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    train, test = split_seasons(design, test_seasons=1)
    pcs = ["pc1", "pc2"]
    for frame in (train, test):
        for c in pcs:
            frame[c] = 0.0
    out = count_variants(train, test, "reb", pcs)
    _, _, feats = out["pca"]
    assert "log_reb_p36_lag1" in feats          # own response survives, on the log scale
    assert set(pcs) <= set(feats)
    assert "mpg_lag1" not in feats              # context columns are replaced by the PCs


def test_evaluate_always_emits_a_carry_forward_row_per_head(tmp_path):
    n_players = 40
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    train, test = split_seasons(design, test_seasons=1)
    table = evaluate(train, test)
    for head in COUNT_HEADS:
        rows = table[(table["head"] == head) & (table["variant"] == "carry_forward")]
        assert len(rows) == 1
    for made, att in CONVERSION_HEADS:
        rows = table[(table["head"] == f"{made}|{att}")
                     & (table["variant"] == "carry_forward")]
        assert len(rows) == 1
    assert table["beats_floor"].dtype == bool


# ── Regularization sensitivity: the sklearn alpha trap, as a permanent guard ───

def _alpha_split(tmp_path, n_players: int = 60):
    games = _games(n_players)
    design = build_design(games, SEASONS, raw_dir=_write_bios(tmp_path, n_players))
    return split_seasons(design, test_seasons=1)


def test_a_large_alpha_destroys_the_fit_quietly(tmp_path):
    """The trap: the fit converges, the coefficients are finite, and R² collapses."""
    train, test = _alpha_split(tmp_path)
    tr, te, features = alpha_feature_sets(train, test, "reb")["linear"]
    good, _ = fit_count_head(tr, te, features, "reb", alpha=POISSON_ALPHA)
    broken, _ = fit_count_head(tr, te, features, "reb", alpha=1.0)
    y = test["reb"].to_numpy(dtype=float)
    assert np.isfinite(broken).all()                       # no error, no warning
    assert _r2(y, broken) < _r2(y, good)


def test_the_sweep_emits_one_row_per_head_variant_and_alpha(tmp_path):
    train, test = _alpha_split(tmp_path)
    alphas = [1e-8, 1e-2, 1.0]
    out = alpha_sensitivity(train, test, alphas, heads=["reb", "ast"])
    assert len(out) == len(alphas) * 2 * len(ALPHA_VARIANTS)
    assert set(out["alpha"]) == set(alphas)
    assert (out["analysis"] == "alpha_sensitivity").all()
    assert set(out["variant"]) == set(ALPHA_VARIANTS)


def test_r2_falls_monotonically_once_the_penalty_dominates(tmp_path):
    train, test = _alpha_split(tmp_path)
    out = alpha_sensitivity(train, test, [1e-2, 1e-1, 1.0, 10.0], heads=["reb"],
                            variants=("log_own",))
    curve = out.sort_values("alpha")["r2"].to_numpy()
    assert (np.diff(curve) <= 1e-9).all()


def test_the_floor_rides_on_every_row_so_the_crossing_needs_no_join(tmp_path):
    train, test = _alpha_split(tmp_path)
    out = alpha_sensitivity(train, test, [1e-8, 10.0], heads=["reb"],
                            variants=("log_own",))
    assert out["floor_r2"].nunique() == 1                  # one floor per head
    assert (out["beats_floor"] == (out["r2"] > out["floor_r2"])).all()


def test_the_floor_wins_at_a_high_alpha_which_is_why_it_is_mandatory(tmp_path):
    """Arithmetic with no parameters beats an over-penalized GLM."""
    train, test = _alpha_split(tmp_path)
    out = alpha_sensitivity(train, test, [10.0], heads=["reb", "ast"],
                            variants=("log_own",))
    assert not out["beats_floor"].any()


def test_the_sweep_skips_splines_and_pca_so_it_stays_cheap(tmp_path):
    train, test = _alpha_split(tmp_path)
    sets = alpha_feature_sets(train, test, "reb")
    assert set(sets) == {"linear", "log_own"}
    for _, _, features in sets.values():
        assert not any("__s" in f or f.startswith("pc") for f in features)


def test_the_alpha_rows_are_added_beside_the_variant_sweep_not_instead_of_it(tmp_path):
    """The extension must not rewrite `component_rate_metrics.csv`'s existing section."""
    train, test = _alpha_split(tmp_path)
    variants = evaluate(train, test)
    alphas = alpha_sensitivity(train, test, [1e-8], heads=["reb"], variants=("log_own",))
    combined = pd.concat([variants, alphas], ignore_index=True)
    assert (combined["analysis"] == "variant_sweep").sum() == len(variants)
    assert (combined["analysis"] == "alpha_sensitivity").sum() == len(alphas)
    assert set(variants.columns) <= set(combined.columns)
