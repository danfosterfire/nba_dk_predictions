"""Tests for the persisted posteriors — the simulation layer's step zero.

None of these needs a sampler, and that is deliberate rather than a compromise. The gate
`docs/simulations-plan.md` names is "saved draws plus recipe reproduce that head's stored
predictions to numerical tolerance **without refitting**", and the fit is precisely the part
the gate is not about. So the heads here are the real classes with their draws *injected*
rather than sampled: `StanCount`, `StanConversion`, `BetaBinomialHead` and `BetaGeometricHead`
all predict from `alpha_draws` / `beta_draws` / `scaler` / `features` and never look at the
fit that produced them. Injecting the draws exercises the identical prediction path a real fit
would, in milliseconds, and lets the round-trip run in CI on a machine with no CmdStan.

The coverage that matters is **one case per way the heads differ from each other**, since
every bug found while building this was a head that did something the others did not: the
composition ladder's discarded imputation flags, the games-played heads' plug-in mean, and
the duration head's missing `_design` and clipped `mu`.

`make posteriors` runs the same `roundtrip()` on every real head and raises on failure, so
the fitted version of this check is a build gate rather than an untested claim.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import posteriors as P
from src.models.component_rates import BIO_COLS, CONTEXT_COLS
from src.models.stan_components import (StanConversion, StanCount, conversion_variants,
                                        count_variants)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n: int = 240, seed: int = 0, seasons=("2019-20", "2020-21", "2021-22",
                                                  "2022-23", "2023-24")) -> pd.DataFrame:
    """A component-head design frame: prior-season rates, context, bio, and targets."""
    rng = np.random.default_rng(seed)
    per_season = max(n // len(seasons), 1)
    rows = []
    for season in seasons:
        for i in range(per_season):
            minutes = float(rng.uniform(400, 2400))
            rows.append({
                "player_id": i, "season": season,
                "total_minutes": minutes,
                "reb_p36_lag1": float(rng.uniform(2.0, 12.0)),
                "fta_p36_lag1": float(rng.uniform(1.0, 8.0)),
                "mpg_lag1": float(rng.uniform(8.0, 36.0)),
                "total_minutes_lag1": float(rng.uniform(300, 2600)),
                "gp_lag1": float(rng.integers(20, 82)),
                "age": float(rng.integers(20, 38)),
                "career_year": float(rng.integers(0, 15)),
                "ftm_pct_lag1": float(rng.uniform(0.55, 0.92)),
                "fta": float(rng.integers(20, 400)),
                "ftm": 0.0, "reb": 0.0,
            })
    frame = pd.DataFrame(rows)
    frame["age_sq"] = frame["age"] ** 2
    frame["reb"] = np.rint(frame["reb_p36_lag1"] * frame["total_minutes"] / 36.0)
    frame["ftm"] = np.rint(frame["fta"] * frame["ftm_pct_lag1"])
    # Holes on both sides of the split, so the imputation step is not merely created on
    # the fitting frame but actually *applied* on the probe rows the round-trip scores.
    frame.loc[frame.index[::37], "gp_lag1"] = np.nan
    return frame


def _inject(model, features, train_matrix, n_draws=64, seed=0, **extra):
    """Give a real head the draws a fit would have left on it, and nothing else.

    Every predict path reads exactly these five attributes, so a head assembled this way is
    the same object the sampler would have returned as far as prediction is concerned.
    """
    from src.models.stan_utils import YearTerm

    rng = np.random.default_rng(seed)
    model.features = list(features)
    model.scaler = StandardScaler().fit(np.nan_to_num(train_matrix))
    # The disabled year term every head in `make stan` carries. Set explicitly because
    # `__new__` skips `__init__`, and `mu_draws` reads it on every call.
    model.year = YearTerm(None)
    model.alpha_draws = rng.normal(size=n_draws) * 0.05
    model.beta_draws = rng.normal(size=(n_draws, len(features))) * 0.05
    for name, value in extra.items():
        setattr(model, name, value)
    return model


def _raw_matrix(frame, features):
    return frame[features].to_numpy(dtype=float)


# ── The recipe steps ──────────────────────────────────────────────────────────

def test_impute_step_fills_from_train_and_flags():
    frame = pd.DataFrame({"x": [1.0, np.nan, 3.0]})
    step = {"kind": "impute", "means": {"x": 2.0}}
    out = P._apply_step(step, frame)
    assert list(out["x"]) == [1.0, 2.0, 3.0]
    assert list(out["x__miss"]) == [0.0, 1.0, 0.0]
    # The input frame is not mutated — steps compose by copy, as the ladders do.
    assert bool(frame["x"].isna().iloc[1])


def test_log_and_logit_steps_match_the_ladder_expressions():
    frame = pd.DataFrame({"a": [0.0, 3.0], "p": [0.0, 0.5]})
    logged = P._apply_step({"kind": "log1p", "columns": {"a": "log_a"}}, frame)
    assert np.allclose(logged["log_a"], np.log1p([0.0, 3.0]))
    # The clip is the ladder's, and it is load-bearing: an unclipped 0.0 is -inf.
    lo = P._apply_step({"kind": "logit", "columns": {"p": "logit_p"}, "clip": 1e-3},
                       frame)
    assert np.isfinite(lo["logit_p"]).all()
    assert np.isclose(lo["logit_p"].iloc[1], 0.0)


def test_bins_step_puts_out_of_range_values_in_the_end_bins():
    frame = pd.DataFrame({"w": [-5.0, 0.05, 0.5, 99.0]})
    step = {"kind": "bins", "column": "w", "edges": np.array([0.1, 0.3]),
            "name": "rho_bin"}
    out = P._apply_step(step, frame)
    assert list(out["rho_bin"]) == [1, 1, 3, 3]


def test_unknown_step_kind_raises_rather_than_silently_skipping():
    with pytest.raises(KeyError):
        P._apply_step({"kind": "not_a_step"}, pd.DataFrame({"x": [1.0]}))


@pytest.mark.parametrize("role_rho", [True, False])
def test_the_cut_step_reproduces_the_availability_heads_own_role_bins(role_rho):
    """The recipe's whole job: which `rho` applies to which player, without refitting.

    Checked against `role_bins` itself rather than against a hand-written expectation, on
    the values that decide it — a NaN prior season, both sides of every edge, and a value
    above the top one. Getting any of those wrong hands a player another bucket's dispersion
    and renders as a perfectly good-looking card.
    """
    from src.models.stan_availability import (ROLE_BIN_COL, ROLE_COL, role_bins,
                                              role_edges)

    frame = pd.DataFrame({ROLE_COL: [np.nan, 0.0, 11.9, 12.0, 12.1, 24.0, 29.9, 30.0,
                                     59.9, 60.0, 61.0, -3.0]})
    step = {"kind": "cut", "column": ROLE_COL, "edges": role_edges(role_rho),
            "name": ROLE_BIN_COL}
    rebuilt = P._apply_step(step, frame)[ROLE_BIN_COL].to_numpy()
    assert list(rebuilt) == list(role_bins(frame, role_rho))
    assert rebuilt.min() >= 1


def test_a_shared_dispersion_cut_puts_every_row_in_one_bin():
    """`role_rho=False` is `n_rho = 1`, and the recipe has to nest it rather than branch."""
    from src.models.stan_availability import ROLE_COL, role_edges

    frame = pd.DataFrame({ROLE_COL: [np.nan, -10.0, 0.0, 35.0, 1e6]})
    out = P._apply_step({"kind": "cut", "column": ROLE_COL,
                         "edges": role_edges(False), "name": "rho_bin"}, frame)
    assert set(out["rho_bin"]) == {1}


def test_the_fit_window_and_the_season_truncation_are_different_columns():
    """Two season axes that must never be read as one another.

    `fit_window` says which split may be fitted; `fit_first_season` says which suffix of it
    was. A head that fits everything the window offers reports `""` for the second, which is
    what makes a truncated head distinguishable from an untruncated one *in the same
    directory* — the only thing `require_window` cannot tell apart.
    """
    truncated = P.PosteriorArtifact(
        head="availability", head_label="availability", family="betabinomial",
        response="mean_mu", recipe=P.DesignRecipe("base", [], None),
        extras={"fit_first_season": "2012-13"},
        provenance={"fit_window": "train", "first_season": "2012-13"})
    plain = P.PosteriorArtifact(
        head="reb", head_label="reb", family="negbinomial", response="mean_count",
        recipe=P.DesignRecipe("base", [], None), extras={},
        provenance={"fit_window": "train", "first_season": "1997-98"})
    # The composition had the truncation first, under a flatter name; the reader takes both
    # so that renaming its key never becomes the price of adding the column.
    legacy = P.PosteriorArtifact(
        head="composition", head_label="composition", family="composition",
        response="eta", recipe=P.DesignRecipe("base", [], None),
        extras={"first_season": "2016-17"},
        provenance={"fit_window": "train", "first_season": "2016-17"})

    assert P.fit_first_season(truncated) == "2012-13"
    assert P.fit_first_season(plain) == ""
    assert P.fit_first_season(legacy) == "2016-17"
    # The distinction is only visible when the two disagree, which is the untruncated head.
    assert plain.provenance["first_season"] != P.fit_first_season(plain)


# ── The round trip, per family ────────────────────────────────────────────────

def _count_artifact(variant: str, component: str = "reb", n_knots: int = 4):
    frame = _design()
    train, probe = frame[frame["season"] < "2022-23"], frame[frame["season"] >= "2022-23"]
    tr, te, features = count_variants(train, probe, component, n_knots)[variant]
    base = [f"{component}_p36_lag1"] + CONTEXT_COLS + BIO_COLS
    steps = P._count_steps(train, tr, variant, features, base,
                           f"{component}_p36_lag1", n_knots)

    model = _inject(StanCount.__new__(StanCount), features,
                    _raw_matrix(tr, features), phi_draws=np.full(64, 8.0),
                    component=component, predictive_samples=64)
    return P._finish(
        head=component, head_label=component, family="negbinomial",
        response="mean_count", variant=variant, features=features, model=model,
        fit_frame=train, probe_transformed=te, probe_raw=probe, steps=steps,
        builder="test", extras={"component": component, "exposure": "total_minutes",
                                "dispersion": "phi_draws"},
        window="train", cfg_stan={}, draws_kept=64, seconds=0.0)


@pytest.mark.parametrize("variant", ["linear", "log_own", "log_own_spline"])
def test_count_head_round_trips_from_the_raw_frame(variant):
    """The gate: raw rows + recipe + draws reproduce the head's own predictions."""
    artifact = _count_artifact(variant)
    check = artifact.roundtrip()
    assert check["passes"], check
    assert check["max_design_error"] <= P.DESIGN_TOL
    assert check["max_prediction_error"] <= P.PREDICTION_TOL
    # Not a trivial pass: the predictions have real scale to disagree over.
    assert np.ptp(artifact.reference["prediction"]) > 1.0


def test_conversion_head_round_trips_from_the_raw_frame():
    frame = _design()
    train, probe = frame[frame["season"] < "2022-23"], frame[frame["season"] >= "2022-23"]
    variant, n_knots = "logit_own_spline", 4
    tr, te, features = conversion_variants(train, probe, "ftm", "fta", n_knots)[variant]
    base = ["ftm_pct_lag1", "fta_p36_lag1"] + CONTEXT_COLS + BIO_COLS
    steps = P._conversion_steps(train, tr, variant, features, base, "ftm_pct_lag1",
                                n_knots)
    model = _inject(StanConversion.__new__(StanConversion), features,
                    _raw_matrix(tr, features), made="ftm", attempted="fta",
                    rho_draws=np.full(64, 0.01), predictive_samples=64)
    artifact = P._finish(
        head="ftm_given_fta", head_label="ftm|fta", family="betabinomial",
        response="mean_mu", variant=variant, features=features, model=model,
        fit_frame=train, probe_transformed=te, probe_raw=probe, steps=steps,
        builder="test", extras={"made": "ftm", "attempted": "fta",
                                "dispersion": "rho_draws"},
        window="train", cfg_stan={}, draws_kept=64, seconds=0.0)
    check = artifact.roundtrip()
    assert check["passes"], check
    # A probability, so the scale check is that it is one and that it varies.
    p = artifact.reference["prediction"]
    assert ((p > 0.0) & (p < 1.0)).all() and np.ptp(p) > 1e-3


def test_plug_in_response_differs_from_the_posterior_mean_and_both_round_trip():
    """The games-played heads report `sigmoid` at the mean, not the mean of `sigmoid`.

    Storing which one a head uses is the whole reason `response` exists: get it wrong and
    the round-trip still "passes" against a number the head never produced.
    """
    from src.models.stan_games_played import BetaBinomialHead

    frame = _design()
    features = ["mpg_lag1", "total_minutes_lag1", "age"]
    model = _inject(BetaBinomialHead.__new__(BetaBinomialHead), features,
                    _raw_matrix(frame, features), rho_draws=np.full(64, 0.2))
    # Wide draws, so Jensen through the logit is worth more than the tolerance.
    model.beta_draws = model.beta_draws * 40.0
    artifact = P._finish(
        head="gp_onset", head_label="games-played onset", family="betabinomial",
        response="plug_in_mu", variant="duration_covariates", features=features,
        model=model, fit_frame=frame, probe_transformed=frame, probe_raw=frame,
        steps=[], builder="test", extras={"dispersion": "rho_draws"},
        window="train", cfg_stan={}, draws_kept=64, seconds=0.0)
    assert artifact.roundtrip()["passes"]

    plug_in = artifact.predict(artifact.reference["frame"])
    artifact.response = "mean_mu"
    posterior_mean = artifact.predict(artifact.reference["frame"])
    assert np.max(np.abs(plug_in - posterior_mean)) > P.PREDICTION_TOL


def _build_with_steps(steps, features, train, tr, te, probe):
    model = _inject(StanCount.__new__(StanCount), features, _raw_matrix(tr, features),
                    phi_draws=np.full(64, 8.0), component="reb", predictive_samples=64)
    return P._finish(head="reb", head_label="reb", family="negbinomial",
                     response="mean_count", variant="log_own", features=features,
                     model=model, fit_frame=train, probe_transformed=te,
                     probe_raw=probe, steps=steps, builder="test",
                     extras={"exposure": "total_minutes", "dispersion": "phi_draws"},
                     window="train", cfg_stan={}, draws_kept=64, seconds=0.0)


@pytest.mark.parametrize("with_covariates", [True, False])
def test_beta_geometric_duration_head_round_trips(with_covariates):
    """The spell-duration head is the odd one out three times over, so it gets its own test.

    It standardizes *inline* inside `shapes` and exposes no `_design`; it has no
    `predict_mean`, `predict_p` or `mu_draws`, so its mean only exists as `a / (a + b)`
    from the per-draw shapes; and that path runs through `games_played.beta_shapes`, which
    clips `mu` — so the clip is part of the reported mean rather than a guard on the way to
    it. The intercept-only arm is covered too, because `features = []` is legal in Stan and
    leaves the head with no scaler at all.
    """
    from src.models.games_played import MU_MAX, MU_MIN
    from src.models.stan_games_played import BetaGeometricHead

    frame = _design()
    features = ["mpg_lag1", "total_minutes_lag1", "age"] if with_covariates else []
    model = BetaGeometricHead.__new__(BetaGeometricHead)
    rng = np.random.default_rng(11)
    model.features = list(features)
    model.scaler = (StandardScaler().fit(_raw_matrix(frame, features))
                    if features else None)
    model.alpha_draws = rng.normal(size=48) * 0.2 - 1.0
    model.beta_draws = (rng.normal(size=(48, len(features))) * 0.3 if features
                        else np.zeros((48, 0)))
    model.kappa_draws = np.full(48, 3.7)

    artifact = P._finish(
        head="gp_duration", head_label="absence-spell duration",
        family="betageometric", response="mean_mu", variant="duration_covariates",
        features=features, model=model, fit_frame=frame, probe_transformed=frame,
        probe_raw=frame, steps=[], builder="test",
        extras={"dispersion": "kappa_draws", "mu_clip": (MU_MIN, MU_MAX)},
        window="train", cfg_stan={}, draws_kept=48, seconds=0.0)
    assert artifact.roundtrip()["passes"]
    mu = artifact.reference["prediction"]
    assert ((mu > 0.0) & (mu < 1.0)).all()
    # The intercept-only arm is one number repeated; the covariate arm must actually vary.
    assert (np.ptp(mu) > 1e-6) == with_covariates


def test_a_drifted_recipe_fails_the_build_rather_than_writing_a_wrong_artifact():
    """Both drift modes, because they fail differently and only one is loud on its own.

    A dropped step leaves a column missing and the recipe names it. A corrupted *fitted
    value* leaves every column present and every shape right — the silent case, and the
    reason the build compares numbers rather than checking that the recipe ran.
    """
    frame = _design()
    train, probe = frame[frame["season"] < "2022-23"], frame[frame["season"] >= "2022-23"]
    tr, te, features = count_variants(train, probe, "reb", 4)["log_own"]
    base = ["reb_p36_lag1"] + CONTEXT_COLS + BIO_COLS
    steps = P._count_steps(train, tr, "log_own", features, base, "reb_p36_lag1", 4)
    assert _build_with_steps(steps, features, train, tr, te, probe).roundtrip()["passes"]

    with pytest.raises(KeyError, match="log_reb_p36_lag1"):
        _build_with_steps(steps[:1], features, train, tr, te, probe)

    drifted = [dict(steps[0], means={"gp_lag1": 1.0}), steps[1]]
    with pytest.raises(AssertionError, match="do NOT reproduce"):
        _build_with_steps(drifted, features, train, tr, te, probe)


def _availability_block(n: int, rng) -> dict:
    """The availability feature block, which the minutes and composition ladders share."""
    from src.models.availability import FEATURE_COLS

    out = {}
    for col in FEATURE_COLS:
        if col == "age":
            out[col] = rng.integers(20, 38, n).astype(float)
        elif col == "age_sq":
            continue
        else:
            out[col] = rng.uniform(0.05, 30.0, n)
    out["age_sq"] = out["age"] ** 2
    return out


def test_minutes_spline_recipe_reproduces_the_ladders_own_columns():
    """`stan_minutes.variants` splines the prior *logit* share and nothing else."""
    from src.models.stan_minutes import OWN, variants

    rng = np.random.default_rng(3)
    n = 200
    frame = pd.DataFrame({**_availability_block(n, rng),
                          OWN: rng.normal(size=n) * 1.5,
                          "successes": rng.integers(100, 2500, n).astype(float),
                          "trials": rng.integers(2500, 3500, n).astype(float),
                          "season": np.repeat(["2021-22", "2022-23"], n // 2)})
    train, probe = frame.iloc[:120], frame.iloc[120:]
    tr, te, features = variants(train, probe, 4)["logit_own_spline"]
    steps = P._minutes_steps(tr, "logit_own_spline", features, OWN, 4)

    recipe = P.DesignRecipe(variant="logit_own_spline", features=features,
                            scaler=StandardScaler().fit(_raw_matrix(tr, features)),
                            steps=tuple(steps), builder="test")
    # The recipe's basis must equal the ladder's, column for column, from the raw frame.
    rebuilt = recipe.transform(probe)
    for name in features:
        np.testing.assert_allclose(rebuilt[name].to_numpy(float),
                                   te[name].to_numpy(float), rtol=0, atol=0)


def test_composition_recipe_reproduces_the_ladders_own_columns_and_bins():
    """The composition ladder is the awkward one: a flag set *before* the imputation
    erases the evidence, a product term built *after* it, and dispersion bin edges that
    are not a design column at all but which the simulator has to index by."""
    from src.models.availability import FEATURE_COLS
    from src.models.stan_composition import OWN, RHO_BIN_COL, RHO_BINS, variants

    rng = np.random.default_rng(5)
    n = 300
    frame = pd.DataFrame({**_availability_block(n, rng),
                          OWN: rng.normal(size=n) * 1.2,
                          RHO_BIN_COL: rng.uniform(0.0, 0.35, n),
                          "no_prior": rng.integers(0, 2, n).astype(float),
                          "share_stale": rng.integers(0, 2, n).astype(float),
                          "offset_clipped": rng.integers(0, 2, n).astype(float),
                          "n_overtimes": rng.integers(0, 3, n).astype(float),
                          "season": np.repeat(["2021-22", "2022-23"], n // 2)})
    # Rookies lose the whole design block at once, which is what `design_missing` is for.
    rookies = frame.index[::23]
    for col in FEATURE_COLS:
        frame.loc[rookies, col] = np.nan

    train, probe = frame.iloc[:180], frame.iloc[180:]
    tr, te, features, dispersed, n_rho = variants(train, probe,
                                                  RHO_BINS)["betabinom_ot_graded"]
    steps = P._composition_steps(train, tr, features, OWN, RHO_BIN_COL, RHO_BINS)

    rebuilt = P.DesignRecipe(variant="betabinom_ot_graded", features=features,
                             scaler=StandardScaler().fit(_raw_matrix(tr, features)),
                             steps=tuple(steps), builder="test").transform(probe)
    assert "ot_x_own" in features and "design_missing" in features
    for name in features:
        np.testing.assert_allclose(rebuilt[name].to_numpy(float),
                                   te[name].to_numpy(float), rtol=0, atol=0)
    # The graded arm's dispersion is indexed by these, so wrong edges are a wrong model
    # even though every design column is right.
    assert dispersed == 1 and n_rho == RHO_BINS
    np.testing.assert_array_equal(rebuilt["rho_bin"].to_numpy(),
                                  te["rho_bin"].to_numpy())
    assert set(np.unique(rebuilt["rho_bin"])) <= set(range(1, RHO_BINS + 1))


# ── Persistence ───────────────────────────────────────────────────────────────

def test_save_and_load_round_trip_through_disk(tmp_path):
    artifact = _count_artifact("log_own_spline")
    dest = P.save(artifact, tmp_path / "posteriors")
    assert dest.exists()
    loaded = P.load("reb", tmp_path / "posteriors")
    assert loaded.roundtrip()["passes"]
    assert loaded.recipe.variant == "log_own_spline"
    assert loaded.n_draws == 64
    np.testing.assert_allclose(loaded.draws["beta_draws"], artifact.draws["beta_draws"])
    assert set(P.load_all(tmp_path / "posteriors")) == {"reb"}


def test_loading_and_scoring_an_artifact_never_reaches_cmdstan(tmp_path):
    """The module's whole claim: `make posteriors` is the last thing that needs a sampler.

    Checked in a subprocess with `cmdstanpy` made unimportable, because the failure would
    be an *import*, and by the time this test runs the suite has already imported it. A
    module-level `from cmdstanpy import ...` anywhere on the load path would fail here and
    nowhere else.
    """
    import subprocess
    import sys

    P.save(_count_artifact("log_own_spline"), tmp_path / "posteriors")
    script = f'''
import sys
class Blocker:
    def find_module(self, name, path=None):
        return self if name.split(".")[0] == "cmdstanpy" else None
    def load_module(self, name):
        raise ImportError("cmdstanpy must not be needed downstream of make posteriors")
sys.meta_path.insert(0, Blocker())

from src.models.posteriors import load
art = load("reb", {str(tmp_path / "posteriors")!r})
assert art.roundtrip()["passes"]
# Scoring a frame the consumer supplies, not replaying the stored probe.
rows = art.reference["frame"].iloc[:5].copy()
rows["age"] = rows["age"] + 3.0
assert art.predict(rows).shape == (5,)
assert "cmdstanpy" not in sys.modules
print("OK")
'''
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert "OK" in done.stdout


def test_load_names_the_target_that_would_write_the_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="make posteriors"):
        P.load("reb", tmp_path)


def test_windows_get_their_own_directory_and_do_not_overwrite_each_other(tmp_path):
    """All three windows are wanted at once, so a flat layout is a data-loss bug.

    `train` feeds the realized 2022-23 / 2023-24 backtest, `train_val` the one-shot test
    readout, `full` the production board. A flat `<head>.pkl` made whichever ran last
    silently replace the others.
    """
    cfg = {"data": {"features_dir": str(tmp_path)}}
    dirs = {w: P.posteriors_dir(cfg, w) for w in P.WINDOWS}
    assert len(set(dirs.values())) == 3
    assert P.posteriors_dir(cfg).name == P.POSTERIOR_DIR

    for window in ("train", "train_val"):
        artifact = _count_artifact("log_own")
        artifact.provenance["fit_window"] = window
        P.save(artifact, dirs[window])
    for window in ("train", "train_val"):
        loaded = P.load("reb", dirs[window])
        assert loaded.provenance["fit_window"] == window
        P.require_window({"reb": loaded}, window)


def test_require_window_refuses_a_wider_fit_than_the_consumer_allows():
    artifact = _count_artifact("log_own")
    artifact.provenance["fit_window"] = "full"
    with pytest.raises(ValueError, match="train_val"):
        P.require_window({"reb": artifact}, "train_val")
    artifact.provenance["fit_window"] = "train_val"
    P.require_window({"reb": artifact}, "train_val")


# ── Thinning and the split ────────────────────────────────────────────────────

def test_thinning_spans_the_whole_posterior_and_is_not_a_head_slice():
    """`season_terms._draw_components` records why: a front slice is a different posterior.

    Checked as a property of the indices rather than of one head, because every artifact
    goes through `_thinned` and the failure would be invisible in any single head's output.
    """
    model = _inject(StanCount.__new__(StanCount), ["mpg_lag1"],
                    _raw_matrix(_design(), ["mpg_lag1"]), n_draws=4000,
                    phi_draws=np.arange(4000, dtype=float), predictive_samples=4000)
    thinned, draws, n_full = P._thinned(model, 1000)
    assert n_full == 4000 and len(draws["alpha_draws"]) == 1000
    # The last kept draw is the last draw of the chain, which a `[:1000]` slice can never be.
    assert draws["phi_draws"][-1] == 3999.0
    assert draws["phi_draws"][0] == 0.0
    assert thinned.predictive_samples == 1000


def test_windowed_widens_from_train_to_train_val_and_never_reaches_test():
    seasons = ["2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25",
               "2025-26"]
    design = pd.DataFrame({"season": np.repeat(seasons, 4)})
    train, probe = P.windowed(design, "train", 2)
    train_val, probe2 = P.windowed(design, "train_val", 2)
    held = {"2024-25", "2025-26"}
    assert set(train["season"]) == {"2019-20", "2020-21", "2021-22"}
    assert set(probe["season"]) == {"2022-23", "2023-24"}
    assert set(train_val["season"]) == set(train["season"]) | set(probe["season"])
    assert set(probe2["season"]) == set(probe["season"])
    assert not (set(train_val["season"]) & held)


def test_full_window_is_a_capability_not_a_flag():
    """`--window full` must go through the held-out unlock, like `final_evaluation` does."""
    from src.models import held_out

    design = pd.DataFrame({"season": np.repeat(["2021-22", "2022-23", "2023-24",
                                                "2024-25", "2025-26"], 4)})
    previous, held_out._unlocked = held_out._unlocked, False
    try:
        with pytest.raises(held_out.HeldOutLocked):
            P.windowed(design, "full", 2)
        with held_out.unlocked("test"):
            frame, _ = P.windowed(design, "full", 2)
        assert set(frame["season"]) == set(design["season"])
    finally:
        held_out._unlocked = previous


def test_probe_rows_span_the_frame_rather_than_taking_its_head():
    frame = pd.DataFrame({"season": np.repeat(["a", "b", "c"], 100)})
    probe = P.probe_rows(frame, 30)
    assert len(probe) == 30
    assert set(probe["season"]) == {"a", "b", "c"}


# ── The composition's player-season effect ────────────────────────────────────

def test_sigma_u_is_thinned_alongside_the_coefficients_and_the_term_is_copied():
    """A random effect's SCALE has to travel with the draws it was fitted beside.

    Two failures this pins. If `sigma_u_draws` is not thinned on the same index as
    `alpha`/`beta`, a consumer applies draw 700's spread to draw 3's coefficients. And if
    the term object is shared rather than copied, thinning the artifact mutates the live
    head — the sort of aliasing bug that only shows up in a second consumer.
    """
    from src.models.stan_composition import PlayerSeasonTerm, StanComposition

    model = StanComposition.__new__(StanComposition)
    model.alpha_draws = np.arange(100, dtype=float)
    model.beta_draws = np.arange(200, dtype=float).reshape(100, 2)
    model.rho_draws = np.arange(100, dtype=float).reshape(100, 1)
    model.ps = PlayerSeasonTerm(True, stream="live")
    model.ps.sigma_draws = np.arange(100, dtype=float) / 100.0
    model.ps.n_units = 7

    thinned, draws, n_full = P._thinned(model, 10)
    assert n_full == 100 and len(draws["sigma_u_draws"]) == 10
    np.testing.assert_array_equal(draws["sigma_u_draws"],
                                  draws["alpha_draws"] / 100.0)
    assert thinned.ps is not model.ps
    assert len(model.ps.sigma_draws) == 100


def test_a_fitted_random_effect_is_persisted_or_refused_but_never_dropped():
    """The failure mode is silent: an artifact carrying only `alpha` and `beta` for a head
    fitted with a random effect has a *narrower* predictive than the head it claims to
    persist, and a round-trip on the MEAN cannot see it. So the composition's `sigma_u` is
    wired through and a year effect still raises."""
    import inspect

    source = inspect.getsource(P._finish)
    assert "player_season_effect" in source and "sigma_u" in source
    assert "NotImplementedError" in source


def test_the_team_context_join_is_a_recipe_step_with_one_flag_and_train_means():
    """A per-unit block joined at recipe time, so a consumer can score a raw frame. The
    block travels inside the artifact rather than being re-read from parquet — a rebuilt
    `team_context_tierA.parquet` must not silently change what a persisted posterior
    scores."""
    frame = pd.DataFrame({"player_id": [1, 2, 3], "season": "2021-22"})
    block = pd.DataFrame({"player_id": [1, 2], "season": "2021-22",
                          "role_crowding": [0.2, 0.4]})
    step = {"kind": "join", "name": "team_context", "keys": ["player_id", "season"],
            "flag": "team_missing", "block": block, "means": {"role_crowding": 0.3}}

    out = P._apply_step(step, frame)
    assert len(out) == len(frame)
    np.testing.assert_allclose(out["role_crowding"].to_numpy(float), [0.2, 0.4, 0.3])
    np.testing.assert_allclose(out["team_missing"].to_numpy(float), [0.0, 0.0, 1.0])


def test_the_team_context_join_raises_rather_than_duplicating_rows():
    """A block with two rows for one unit would silently double the design frame, which
    every downstream shape check would then agree with."""
    frame = pd.DataFrame({"player_id": [1], "season": "2021-22"})
    block = pd.DataFrame({"player_id": [1, 1], "season": "2021-22",
                          "role_crowding": [0.2, 0.4]})
    step = {"kind": "join", "name": "team_context", "keys": ["player_id", "season"],
            "flag": "team_missing", "block": block, "means": {"role_crowding": 0.3}}

    with pytest.raises(ValueError, match="duplicated rows"):
        P._apply_step(step, frame)


def test_the_composition_team_recipe_reproduces_the_ladders_own_columns():
    """The drift check for the team block, and it is the one that caught a real bug.

    The recipe's `join` step must carry the **raw** block, not the ladder's already-imputed
    frame. Built from `tr` the artifact would persist imputed values as if they were
    observed and leave `team_missing` identically zero on every rebuilt frame — every
    column present, every shape right, and only the numbers wrong, which is exactly the
    failure mode the round-trip exists for and the one that got past the first version of
    the imputation step too.
    """
    from src.models.availability import FEATURE_COLS
    from src.models.stan_composition import (OWN, RHO_BIN_COL, RHO_BINS, TEAM_COLS,
                                             TEAM_MISSING, effect_variants)

    rng = np.random.default_rng(11)
    n = 240
    frame = pd.DataFrame({**_availability_block(n, rng),
                          OWN: rng.normal(size=n) * 1.2,
                          RHO_BIN_COL: rng.uniform(0.0, 0.35, n),
                          "no_prior": rng.integers(0, 2, n).astype(float),
                          "share_stale": rng.integers(0, 2, n).astype(float),
                          "offset_clipped": rng.integers(0, 2, n).astype(float),
                          "n_overtimes": rng.integers(0, 3, n).astype(float),
                          "player_id": np.arange(n),
                          "season": np.repeat(["2021-22", "2022-23"], n // 2)})
    # The block covers two thirds of the units, which is the shape the real file has.
    covered = frame.iloc[::3].index
    block = pd.DataFrame({"player_id": frame.loc[covered, "player_id"].to_numpy(),
                          "season": frame.loc[covered, "season"].to_numpy()})
    for j, col in enumerate(TEAM_COLS):
        block[col] = rng.normal(size=len(block)) + j

    train, probe = frame.iloc[:150], frame.iloc[150:]
    built, _ = effect_variants(train, probe, block, "betabinom_ot_graded", RHO_BINS)
    tr, te, features, _, _, _, _ = built["team"]
    steps = P._composition_steps(train, tr, features, OWN, RHO_BIN_COL, RHO_BINS,
                                 team_block=block)

    rebuilt = P.DesignRecipe(variant="team", features=features,
                             scaler=StandardScaler().fit(_raw_matrix(tr, features)),
                             steps=tuple(steps), builder="test").transform(probe)
    assert TEAM_MISSING in features
    assert 0 < rebuilt[TEAM_MISSING].mean() < 1, "the probe must have holes to check"
    for name in features:
        np.testing.assert_allclose(rebuilt[name].to_numpy(float),
                                   te[name].to_numpy(float), rtol=0, atol=0)
