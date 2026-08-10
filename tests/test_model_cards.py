"""Tests for the model-card emitter — the dashboard's only route to the fitted heads.

None of these fits anything, which is the same stance `tests/test_posteriors.py` takes and
for the same reason: the emitter refits nothing either. The artifacts here are real
`PosteriorArtifact`s with their draws **injected**, so `coefficient_rows` and `verify` walk
the identical code path a persisted head does, in milliseconds.

The coverage that matters is one case per way this module can be wrong *silently*. A
histogram drawn on the wrong edges, a missing-share that resolves to zero because the flag
sits under a third name, a correlation that reports 0 for a constant column, a split label
that is not `train` or `validation` — every one of those renders as a perfectly good-looking
picture. So each gets a test, and the two checks that would fail loudly (the population
anchor and the design tolerance) get one each for the raise.

`make model-cards` runs `verify` on all twenty real heads and raises on failure, so the
fitted version of these checks is a build gate rather than an untested claim.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import model_cards as M
from src.models.posteriors import DesignRecipe, PosteriorArtifact

PREDICTIONS = Path("outputs/predictions")


# ── Synthetic builders ────────────────────────────────────────────────────────

def _raw(n: int = 200, seed: int = 0, missing: int = 20) -> pd.DataFrame:
    """A raw design frame: one skewed rate, one flag-worthy hole, one bio column."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "season": ["2019-20"] * n,
        "x": rng.gamma(2.0, 1.5, n),
        "age": rng.integers(20, 38, n).astype(float),
    })
    frame.loc[: missing - 1, "x"] = np.nan
    return frame


def _design_from(raw: pd.DataFrame, mean: float | None = None) -> pd.DataFrame:
    """The ladder's own transformed frame: impute from the train mean, flag it, log it."""
    out = raw.copy()
    fill = raw["x"].mean() if mean is None else mean
    out["x__miss"] = raw["x"].isna().astype(float)
    out["x"] = raw["x"].fillna(fill)
    out["log_x"] = np.log1p(np.clip(out["x"].to_numpy(dtype=float), 0.0, None))
    return out


def _recipe(train: pd.DataFrame, features: list[str]) -> DesignRecipe:
    design = _design_from(train)
    scaler = StandardScaler().fit(design[features].to_numpy(dtype=float))
    steps = ({"kind": "impute", "means": {"x": float(train["x"].mean())}},
             {"kind": "log1p", "columns": {"x": "log_x"}})
    return DesignRecipe(variant="log_own", features=list(features), scaler=scaler,
                        steps=steps, builder="tests.synthetic")


def _artifact(train: pd.DataFrame, features: list[str], seed: int = 1,
              n_draws: int = 200) -> PosteriorArtifact:
    """A beta-binomial head carrying injected draws and a passing round-trip probe."""
    rng = np.random.default_rng(seed)
    recipe = _recipe(train, features)
    art = PosteriorArtifact(
        head="synthetic", head_label="synthetic", family="betabinomial",
        response="mean_mu", recipe=recipe,
        draws={"alpha_draws": rng.normal(-1.0, 0.05, n_draws),
               "beta_draws": rng.normal(0.3, 0.1, (n_draws, len(features))),
               "rho_draws": rng.gamma(4.0, 0.01, n_draws)},
        extras={"dispersion": "rho_draws"},
        provenance={"n_fit_rows": len(train), "first_season": "2019-20",
                    "last_season": "2019-20", "fit_window": "train"})
    probe = train.iloc[:40]
    art.reference = {"frame": probe.copy(), "design": recipe.matrix(probe),
                     "prediction": art.predict(probe)}
    return art


def _frames(train: pd.DataFrame, val: pd.DataFrame, features: list[str],
            design_train: pd.DataFrame | None = None) -> M.HeadFrames:
    return M.HeadFrames(
        head="synthetic",
        raw={"train": train, "validation": val},
        design={"train": design_train if design_train is not None
                else _design_from(train),
                "validation": _design_from(val, mean=train["x"].mean())},
        features=list(features),
        n_frame={"train": len(train), "validation": len(val)})


FEATURES = ["log_x", "age", "x__miss"]


def _counted(design: pd.DataFrame, trials: int = 60, p: float = 0.3,
             seed: int = 11) -> pd.DataFrame:
    """A design frame with somewhere for a beta-binomial predictive to land."""
    rng = np.random.default_rng(seed)
    out = design.copy()
    out["n"] = float(trials)
    out["y"] = rng.binomial(trials, p, len(out)).astype(float)
    return out


COUNTED = M.ResponseSpec("games", "y", trials="n")


# ── The split vocabulary ──────────────────────────────────────────────────────

def test_the_only_legal_splits_are_train_and_validation():
    """There is no test column, for the same reason the sweeps no longer emit one."""
    assert M.SPLITS == ("train", "validation")
    assert M.WINDOW == "train"


def test_a_third_split_label_is_refused_on_the_way_to_disk():
    legal = pd.DataFrame({"split": ["train", "validation"], "value": [1.0, 2.0]})
    assert M._check_splits(legal, "legal.csv") is legal

    leaked = pd.DataFrame({"split": ["train", "test"], "value": [1.0, 2.0]})
    with pytest.raises(ValueError, match="test"):
        M._check_splits(leaked, "leaked.csv")


def test_the_emitter_reaches_a_split_only_through_selection_split():
    """`split_seasons` hands back the guarded held-out frame; `selection_split` does not.

    Checked on the import graph rather than on a string, so a docstring mentioning the
    wrong function does not fail and an actual import of it cannot pass.
    """
    tree = ast.parse(Path(M.__file__).read_text())
    imported = {alias.name for node in ast.walk(tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "selection_split" in imported
    for forbidden in ("split_seasons", "final_split", "unlocked", "assert_unlocked"):
        assert forbidden not in imported


# ── Terms ─────────────────────────────────────────────────────────────────────

def test_a_spline_basis_groups_under_the_column_it_expands():
    """Twelve bars for one quantity would swamp every real term in the head."""
    assert M.term_family("log_ast_p36_lag1__s3") == ("log_ast_p36_lag1", 3)
    assert M.term_family("logit_share_lag1__s11") == ("logit_share_lag1", 11)
    assert M.term_family("logit_share_lag1__sq") == ("logit_share_lag1", None)
    assert M.term_family("fg3m_pct_lag1__miss") == ("missingness", None)
    assert M.term_family("age") == ("age", None)


def test_source_columns_walks_a_spline_over_a_logit_back_to_the_raw_column():
    assert M.source_columns("logit_fg3m_pct_lag1__s3") == [
        "logit_fg3m_pct_lag1__s3", "logit_fg3m_pct_lag1", "fg3m_pct_lag1"]
    # A column that only looks transformed keeps itself as the nearest candidate, which is
    # what lets `logit_share_lag1` be raw in one head's frame and derived in another's.
    assert M.source_columns("logit_share_lag1")[0] == "logit_share_lag1"
    assert M.source_columns("age") == ["age"]


# ── Coefficients ──────────────────────────────────────────────────────────────

def test_coefficients_carry_the_intercept_every_feature_and_the_dispersion():
    train = _raw()
    art = _artifact(train, FEATURES)
    rows = M.coefficient_rows("synthetic", art)

    assert [r["term"] for r in rows] == ["(intercept)"] + FEATURES + ["rho"]
    assert [r["term_role"] for r in rows] == (
        ["intercept"] + ["coefficient"] * len(FEATURES) + ["dispersion"])
    assert rows[0]["term_family"] == "intercept"
    assert rows[-1]["term_family"] == "dispersion"
    assert all(r["head"] == "synthetic" for r in rows)


def test_a_coefficient_carries_the_scaler_so_it_can_be_unstandardized():
    """The numbers travel; the fitted `StandardScaler` deliberately does not."""
    train = _raw()
    art = _artifact(train, FEATURES)
    rows = {r["term"]: r for r in M.coefficient_rows("synthetic", art)}

    design = _design_from(train)
    assert rows["age"]["scaler_center"] == pytest.approx(design["age"].mean())
    assert rows["age"]["scaler_scale"] == pytest.approx(design["age"].std(ddof=0))
    # An intercept and a dispersion have no design column, so no centre and no scale.
    assert np.isnan(rows["(intercept)"]["scaler_center"])
    assert np.isnan(rows["rho"]["scaler_scale"])


def test_a_graded_dispersion_becomes_one_term_per_bin():
    """The composition fits one `rho` per prior-share bin; a mean would erase the spread."""
    train = _raw()
    art = _artifact(train, FEATURES)
    art.draws["rho_draws"] = np.stack(
        [np.full(200, 0.18), np.full(200, 0.13), np.full(200, 0.09)], axis=1)
    rows = {r["term"]: r for r in M.coefficient_rows("synthetic", art)}

    assert [t for t in rows if t.startswith("rho")] == ["rho[1]", "rho[2]", "rho[3]"]
    assert rows["rho[1]"]["mean"] == pytest.approx(0.18)
    assert rows["rho[3]"]["mean"] == pytest.approx(0.09)


def test_the_quantiles_are_ordered_and_p_positive_agrees_with_them():
    """`p_positive` is the reading a 95% interval crossing zero under-reports.

    The two are consistent rather than identical: an interval clearing zero puts at least
    97.5% of the posterior on one side, and usually more.
    """
    train = _raw()
    art = _artifact(train, FEATURES)
    for row in M.coefficient_rows("synthetic", art):
        assert row["q2.5"] <= row["q25"] <= row["q50"] <= row["q75"] <= row["q97.5"]
        if row["q2.5"] > 0:
            assert row["p_positive"] >= 0.975
        if row["q97.5"] < 0:
            assert row["p_positive"] <= 0.025


# ── Feature bins ──────────────────────────────────────────────────────────────

def test_a_flag_gets_one_bin_per_value_rather_than_a_spike_in_bin_zero():
    edges, kind = M.bin_edges(np.array([0.0, 0.0, 1.0, 0.0]))
    assert kind == "discrete"
    assert list(edges) == [0.0, 1.0]
    counts = M._counts(np.array([0.0, 0.0, 1.0, 0.0]), edges, kind)
    assert list(counts) == [3, 1]


def test_a_continuous_column_gets_linear_bins_and_the_maximum_lands_in_the_top_bar():
    values = np.linspace(0.0, 10.0, 500)
    edges, kind = M.bin_edges(values)
    assert kind == "linear"
    assert len(edges) == M.FEATURE_BINS + 1
    counts = M._counts(values, edges, kind)
    assert counts.sum() == values.size          # nothing falls off the closed top bin
    assert counts[-1] > 0


def test_an_all_nan_column_bins_to_nothing_rather_than_raising():
    edges, kind = M.bin_edges(np.array([np.nan, np.nan]))
    assert kind == "empty" and edges.size == 0
    assert M._counts(np.array([np.nan]), edges, kind).size == 0


def test_both_splits_share_one_edge_set_because_comparing_them_is_the_point():
    """Two histograms drawn on their own edges cannot be compared."""
    train, val = _raw(300, seed=1), _raw(80, seed=2, missing=0)
    val["x"] = val["x"] * 4.0                    # a deliberately wider validation range
    rows = pd.DataFrame(M.feature_rows("synthetic", _frames(train, val, FEATURES)))

    log_x = rows[rows["feature"] == "log_x"]
    edges = {split: tuple(part.sort_values("bin_index")["bin_left"])
             for split, part in log_x.groupby("split")}
    assert edges["train"] == edges["validation"]
    # And the pooled range reaches the wider split, or the top of it would be uncounted.
    assert log_x["count"].groupby(log_x["split"]).sum().tolist() == [len(train), len(val)]


def test_density_is_a_share_so_a_small_split_can_be_drawn_over_a_large_one():
    train, val = _raw(300, seed=1), _raw(80, seed=2)
    rows = pd.DataFrame(M.feature_rows("synthetic", _frames(train, val, FEATURES)))
    totals = rows.groupby(["feature", "split"])["density"].sum()
    assert np.allclose(totals.to_numpy(), 1.0)


# ── Missingness ───────────────────────────────────────────────────────────────

def test_a_spline_or_log_column_inherits_the_missingness_of_the_column_below_it():
    """The failure this prevents renders as a flat zero on a page, not as an error."""
    train = _raw(200, missing=20)
    design = _design_from(train)
    assert M.missing_share("log_x", train, design) == pytest.approx(0.10)
    assert M.missing_share("x", train, design) == pytest.approx(0.10)


def test_an_imputation_flag_is_never_itself_missing_and_its_mean_is_the_share():
    train = _raw(200, missing=20)
    design = _design_from(train)
    assert M.missing_share("x__miss", train, design) == 0.0

    rows = pd.DataFrame(M.feature_rows("synthetic", _frames(train, _raw(50), FEATURES)))
    flag = rows[(rows["feature"] == "x__miss") & (rows["split"] == "train")]
    assert flag["mean"].iloc[0] == pytest.approx(0.10)


def test_missingness_falls_back_to_the_raw_nan_share_when_no_flag_was_minted():
    train = _raw(200, missing=20)
    design = train.copy()                        # a head that imputes nothing
    assert M.missing_share("x", train, design) == pytest.approx(0.10)


# ── Feature relationships ─────────────────────────────────────────────────────

def test_the_correlation_matrix_is_the_whole_square_including_the_diagonal():
    """A heatmap should be a reshape, not a reconstruction."""
    train, val = _raw(300, seed=3), _raw(90, seed=4)
    rows = pd.DataFrame(M.correlation_rows("synthetic", _frames(train, val, FEATURES)))

    assert len(rows) == len(FEATURES) ** 2 * len(M.SPLITS)
    diagonal = rows[rows["feature_x"] == rows["feature_y"]]
    assert np.allclose(diagonal["r"].to_numpy(), 1.0)
    assert sorted(rows["split"].unique()) == ["train", "validation"]


def test_a_constant_column_reads_as_nan_rather_than_as_a_spurious_zero():
    train, val = _raw(300, seed=3), _raw(90, seed=4, missing=0)
    rows = pd.DataFrame(M.correlation_rows("synthetic", _frames(train, val, FEATURES)))

    # `x__miss` is identically 0 on the validation frame above — no row was imputed — so
    # every off-diagonal correlation involving it is undefined.
    off = rows[(rows["split"] == "validation") & (rows["feature_x"] == "x__miss")
               & (rows["feature_y"] != "x__miss")]
    assert off["r"].isna().all()
    assert rows[rows["split"] == "train"]["r"].notna().all()


def test_both_orientations_of_a_pair_carry_the_same_rank_and_the_diagonal_carries_none():
    train, val = _raw(300, seed=5), _raw(90, seed=6)
    rows = pd.DataFrame(M.correlation_rows("synthetic", _frames(train, val, FEATURES)))
    train_rows = rows[rows["split"] == "train"]

    for _, row in train_rows.iterrows():
        mirror = train_rows[(train_rows["feature_x"] == row["feature_y"])
                            & (train_rows["feature_y"] == row["feature_x"])]
        assert mirror["pair_rank"].iloc[0] == row["pair_rank"]
    assert (train_rows[train_rows["feature_x"] == train_rows["feature_y"]]["pair_rank"]
            == -1).all()
    assert set(train_rows[train_rows["pair_rank"] > 0]["pair_rank"]) == {1, 2, 3}


def test_top_pair_marks_at_most_the_cap_and_marks_both_orientations():
    train, val = _raw(300, seed=7), _raw(90, seed=8)
    rows = pd.DataFrame(M.correlation_rows("synthetic", _frames(train, val, FEATURES),
                                           top_pairs=2))
    flagged = rows[(rows["split"] == "train") & rows["top_pair"]]
    assert len(flagged) == 4                      # two pairs, two orientations each
    assert set(flagged["pair_rank"]) == {1, 2}


# ── The joint density behind the heatmap ──────────────────────────────────────

def test_the_density_covers_exactly_the_pairs_the_heatmap_flagged():
    """The menu a page builds and the panels it can draw come from one ranking."""
    train, val = _raw(300, seed=11), _raw(90, seed=12)
    frames = _frames(train, val, FEATURES)
    corr = M.correlation_rows("synthetic", frames, top_pairs=2)
    rows = pd.DataFrame(M.density_rows("synthetic", frames, corr))

    flagged = {(x, y) for x, y, _, _ in M.density_pairs(corr)}
    assert len(flagged) == 2
    assert set(zip(rows["feature_x"], rows["feature_y"])) == flagged
    # One orientation per pair. The square carries both and they are the same picture.
    assert not {(y, x) for x, y in flagged} & set(zip(rows["feature_x"], rows["feature_y"]))


def test_the_pair_menu_is_ranked_on_train_so_flipping_the_split_changes_no_menu():
    """A menu that reshuffles under the split toggle breaks the comparison it exists for."""
    train, val = _raw(400, seed=13), _raw(120, seed=14)
    frames = _frames(train, val, FEATURES)
    corr = M.correlation_rows("synthetic", frames, top_pairs=2)
    rows = pd.DataFrame(M.density_rows("synthetic", frames, corr))

    per_split = {split: set(zip(part["feature_x"], part["feature_y"]))
                 for split, part in rows.groupby("split")}
    assert per_split["train"] == per_split["validation"]
    assert [rank for _, _, rank, _ in M.density_pairs(corr)] == [1, 2]


def test_both_splits_of_a_pair_share_one_grid():
    """Two panels drawn on their own edges cannot be compared, which is the block's job."""
    train, val = _raw(300, seed=15), _raw(90, seed=16)
    frames = _frames(train, val, FEATURES)
    corr = M.correlation_rows("synthetic", frames, top_pairs=3)
    rows = pd.DataFrame(M.density_rows("synthetic", frames, corr))

    # Compared per cell index rather than over the occupied range: one split need not
    # occupy the other's extreme cell, and that difference is the picture, not the grid.
    for (x, y), pair in rows.groupby(["feature_x", "feature_y"]):
        for axis in ("x", "y"):
            edges = pair.groupby([f"{axis}_index", "split"])[
                [f"{axis}_left", f"{axis}_right"]].first().groupby(level=0).nunique()
            assert (edges == 1).all().all(), (x, y, axis)


def test_every_row_of_a_split_lands_in_exactly_one_cell():
    """Counts have to sum to `n`, or the panel is a picture of a subset it does not name."""
    train, val = _raw(300, seed=17), _raw(90, seed=18)
    frames = _frames(train, val, FEATURES)
    corr = M.correlation_rows("synthetic", frames, top_pairs=3)
    rows = pd.DataFrame(M.density_rows("synthetic", frames, corr))

    totals = rows.groupby(["feature_x", "feature_y", "split"])[["count", "density"]].sum()
    sizes = rows.groupby(["feature_x", "feature_y", "split"])["n"].first()
    assert (totals["count"] == sizes).all()
    assert np.allclose(totals["density"].to_numpy(), 1.0)


def test_a_discrete_column_gets_one_cell_per_value_rather_than_a_spike_in_bin_zero():
    """The same rule the 1-D histograms follow, or the two blocks disagree about a flag."""
    train, val = _raw(300, seed=19, missing=30), _raw(90, seed=20, missing=9)
    frames = _frames(train, val, FEATURES)
    corr = M.correlation_rows("synthetic", frames, top_pairs=10)
    rows = pd.DataFrame(M.density_rows("synthetic", frames, corr))

    flag = rows[(rows["feature_x"] == "x__miss") | (rows["feature_y"] == "x__miss")]
    assert len(flag), "the imputation flag should reach the top pairs here"
    for _, part in flag.groupby(["feature_x", "feature_y", "split"]):
        axis = "x" if (part["feature_x"] == "x__miss").all() else "y"
        assert set(part[f"{axis}_index"]) <= {0, 1}
        # A discrete cell is the value itself, not an interval it sits inside.
        assert (part[f"{axis}_left"] == part[f"{axis}_right"]).all()


def test_the_top_of_a_feature_lands_in_the_top_cell_rather_than_outside_the_grid():
    """`np.digitize` would put the maximum one bin past the end; `_counts` does not."""
    edges, kind = M.bin_edges(np.linspace(0.0, 1.0, 500), bins=4)
    assert kind == "linear"
    index = M._cell_index(np.array([0.0, 0.4, 1.0]), edges, kind)
    assert index.tolist() == [0, 1, 3]


def test_a_head_with_no_feature_pair_cards_no_density_rather_than_raising():
    """`game_length_depth` carries no features at all, and its page still has to render."""
    train, val = _raw(60, seed=21), _raw(20, seed=22)
    frames = _frames(train, val, features=[])
    corr = M.correlation_rows("empty", frames)

    assert corr == []
    assert M.density_pairs(corr) == []
    assert M.density_rows("empty", frames, corr) == []


# ── Verification ──────────────────────────────────────────────────────────────

def test_verify_passes_when_the_recipe_and_the_ladder_agree():
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, FEATURES)
    check = M.verify(art, _frames(train, val, FEATURES))

    assert check["verified"] and check["recipe_design_error"] == 0.0
    assert check["design_check"] == "ladder"
    assert check["n_fit"] == len(train) and check["n_validation"] == len(val)


def test_a_drifted_transform_fails_the_build_rather_than_writing_the_artifact():
    """The recipe and the head's ladder are two paths to one matrix; they must agree."""
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, FEATURES)
    drifted = _design_from(train)
    drifted["log_x"] = np.log(drifted["x"] + 2.0)        # log1p became log(x + 2)

    with pytest.raises(AssertionError, match="variant ladder"):
        M.verify(art, _frames(train, val, FEATURES, design_train=drifted))


def test_a_rebuilt_frame_of_the_wrong_size_fails_the_population_anchor():
    """Coefficients describing one population and histograms another is silent otherwise."""
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, FEATURES)
    art.provenance["n_fit_rows"] = 8_630

    with pytest.raises(AssertionError, match="rebuilt fitting frame"):
        M.verify(art, _frames(train, val, FEATURES))


def test_a_ladder_that_stops_producing_a_feature_raises_by_name():
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, FEATURES)
    frames = _frames(train, val, FEATURES)
    frames.design["validation"] = frames.design["validation"].drop(columns=["log_x"])

    with pytest.raises(KeyError, match="log_x"):
        M.verify(art, frames)


def test_the_design_check_is_reported_as_vacuous_when_the_recipe_has_no_steps():
    """A head whose raw frame IS its design frame compares a matrix against itself."""
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, ["age"])
    art.recipe = DesignRecipe(variant="base", features=["age"], scaler=art.recipe.scaler,
                              steps=(), builder="tests.synthetic")
    art.recipe.scaler = StandardScaler().fit(train[["age"]].to_numpy(dtype=float))
    probe = train.iloc[:40]
    art.reference = {"frame": probe.copy(), "design": art.recipe.matrix(probe),
                     "prediction": art.predict(probe)}

    frames = M.HeadFrames(head="synthetic", raw={"train": train, "validation": val},
                          design={"train": train, "validation": val},
                          features=["age"],
                          n_frame={"train": len(train), "validation": len(val)})
    assert M.verify(art, frames)["design_check"] == "vacuous"


# ── The index ─────────────────────────────────────────────────────────────────

def test_every_declared_head_names_a_class_the_dashboard_knows():
    assert set(M.CLASS_LABELS) == {s.model_class for s in M.SPECS.values()}
    assert all(s.unit and s.likelihood and s.description for s in M.SPECS.values())


def test_a_head_with_no_declared_unit_raises_rather_than_shipping_a_blank_column():
    """Every page states its own unit, and it is read from here rather than hard-coded."""
    train, val = _raw(200, seed=9), _raw(60, seed=10)
    art = _artifact(train, FEATURES)
    frames = _frames(train, val, FEATURES)
    check = M.verify(art, frames)

    with pytest.raises(KeyError, match="HeadSpec"):
        M.index_row("synthetic", art, frames, check, n_terms=5)


def test_the_four_component_conversion_heads_declare_their_own_row_filter():
    """`StanConversion.fit` drops rows with no attempts, so `n_fit` < the manifest's."""
    for head in ("fg3a_given_fga", "fg2m_given_fg2a", "fg3m_given_fg3a", "ftm_given_fta"):
        assert M.SPECS[head].model_class == "components"
        assert M.SPECS[head].unit == "player-season"


# ── The predictive: which rows it is drawn over ───────────────────────────────

def test_every_head_that_declares_a_unit_also_declares_what_its_predictive_is_of():
    """A page cannot label an axis it has to guess at."""
    assert set(M.RESPONSES) == set(M.SPECS)
    assert all(r.label and r.observed for r in M.RESPONSES.values())
    assert all(r.check in ("mean", "p_one", "none") for r in M.RESPONSES.values())


def test_a_head_with_no_declared_response_raises_rather_than_shipping_an_empty_ribbon():
    train, val = _raw(120, seed=21), _raw(40, seed=22)
    art = _artifact(train, FEATURES)
    with pytest.raises(KeyError, match="ResponseSpec"):
        M.predictive_tables("synthetic", art, _frames(train, val, FEATURES), cfg={})


def test_a_collapsed_cell_frame_is_expanded_by_its_weight_before_anything_is_drawn():
    """The depth head fits four rows carrying 1,861 games; the ECDF is of the games."""
    cells = pd.DataFrame({"t": [1.0, 2.0, 3.0], "w": [10.0, 4.0, 1.0]})
    spec = M.ResponseSpec("overtime periods", "t", weight="w", check="p_one")
    frame, capped = M.predictive_frame("depth", spec, cells)

    assert len(frame) == 15 and not capped
    assert frame["t"].tolist() == [1.0] * 10 + [2.0] * 4 + [3.0]


def test_a_capped_predictive_frame_spans_the_frame_rather_than_its_first_rows():
    """The frames are season-sorted, so a head slice is one era of the league."""
    frame = pd.DataFrame({"y": np.arange(1000.0), "n": np.full(1000, 5.0)})
    cut, capped = M.predictive_frame("wide", COUNTED, frame, cap=100)

    assert capped and len(cut) == 100
    assert cut["y"].iloc[0] == 0.0 and cut["y"].iloc[-1] == 999.0


def test_a_missing_response_column_raises_by_name_rather_than_drawing_nothing():
    frame = pd.DataFrame({"y": [1.0, 2.0]})
    with pytest.raises(KeyError, match="'n'"):
        M.predictive_frame("wide", COUNTED, frame)


def test_thinning_an_artifact_takes_the_same_draws_across_the_whole_posterior():
    train = _raw(80, seed=23)
    art = _artifact(train, FEATURES, n_draws=200)
    thinned = M.thinned_artifact(art, 20)

    assert thinned.n_draws == 20
    assert np.allclose(thinned.draws["alpha_draws"],
                       art.draws["alpha_draws"][M.thin(200, 20)])
    # A shallow copy: the original keeps every draw it had.
    assert art.n_draws == 200


# ── The predictive: how it is drawn ───────────────────────────────────────────

def test_a_beta_binomial_predictive_respects_its_own_trials_denominator():
    train = _raw(150, seed=24)
    art = _artifact(train, FEATURES)
    frame = _counted(_design_from(train))
    draws = M.family_draws(art, COUNTED, frame, keep=64, seed=1)

    assert draws.shape == (64, len(frame))
    assert draws.min() >= 0 and draws.max() <= 60
    # And it lands on the head's own mean rather than near it by luck.
    expected = art.predict(frame, transformed=True) * frame["n"].to_numpy(float)
    assert draws.mean() == pytest.approx(expected.mean(), rel=0.02)


def test_a_beta_geometric_predictive_starts_at_one_because_a_spell_is_at_least_one_game():
    train = _raw(150, seed=25)
    art = _artifact(train, FEATURES)
    art.family = "betageometric"
    art.draws["kappa_draws"] = np.full(200, 3.7)
    spec = M.ResponseSpec("spell length", "t", weight="w", check="p_one")
    draws = M.family_draws(art, spec, _design_from(train), keep=32, seed=2)

    assert draws.shape[0] == 32 and draws.min() >= 1.0
    # `mu` IS P(T = 1) for this likelihood, which is the check the emitter runs on it.
    mu = float(art.predict(_design_from(train), transformed=True).mean())
    assert (draws == 1).mean() == pytest.approx(mu, abs=0.05)


def test_a_head_that_can_draw_is_drawn_through_its_own_predict_samples():
    """The rule of this half: no second implementation of any head's predictive.

    Checked against the real `StanCount`, rehydrated around injected draws — a
    reimplementation would agree on the mean and disagree on the draws, which is exactly the
    difference a card is supposed to show.
    """
    from src.models.stan_components import StanCount

    train = _raw(150, seed=33)
    art = _artifact(train, FEATURES)
    art.family, art.response = "negbinomial", "mean_count"
    art.extras = {"component": "y", "exposure": "total_minutes",
                  "dispersion": "phi_draws"}
    art.draws["phi_draws"] = np.full(200, 4.0)
    frame = _design_from(train)
    frame["total_minutes"] = 1500.0
    frame["y"] = 120.0

    model = M._rehydrated("counts", art, cfg={}, keep=32)
    assert isinstance(model, StanCount)
    assert np.allclose(model.alpha_draws, art.draws["alpha_draws"])
    assert model.predictive_samples == 32

    spec = M.ResponseSpec("season y", "y")
    drawn = M.draw_predictive("counts", art, spec, frame, cfg={}, keep=32, seed=7)
    assert np.array_equal(drawn, model.predict_samples(frame, 7))
    assert drawn.shape == (32, len(frame))


def test_a_family_with_no_sampling_law_raises_rather_than_being_guessed_at():
    train = _raw(60, seed=26)
    art = _artifact(train, FEATURES)
    art.family = "student_t"
    with pytest.raises(KeyError, match="no predictive"):
        M.family_draws(art, COUNTED, _counted(_design_from(train)), keep=8, seed=3)


def test_a_rate_reporting_head_is_put_on_the_observation_scale_by_its_trials():
    """`predict` gives a probability; the ribbon and the scatter are in games."""
    train = _raw(120, seed=27)
    art = _artifact(train, FEATURES)
    frame = _counted(_design_from(train))
    draws = M.family_draws(art, COUNTED, frame, keep=32, seed=4)
    fitted = M.fitted_values(art, COUNTED, frame, draws)

    rate = art.predict(frame, transformed=True)
    assert np.allclose(fitted, rate * 60.0)
    assert fitted.mean() > 1.0                      # games, not a probability


def test_a_head_with_no_reportable_mean_takes_the_mean_of_its_own_draws():
    """The composition reports `eta`, a step's linear predictor. There is no scale."""
    train = _raw(60, seed=28)
    art = _artifact(train, FEATURES)
    spec = M.ResponseSpec("minutes", "y", trials="n", check="none")
    draws = np.tile(np.arange(60.0), (8, 1))
    fitted = M.fitted_values(art, spec, _counted(_design_from(train)), draws)

    assert np.allclose(fitted, np.arange(60.0))
    assert np.isnan(M.predictive_bias(art, spec, _counted(_design_from(train)),
                                      draws, fitted))


def test_the_mean_gate_catches_a_predictive_drawn_on_the_wrong_scale():
    """A recipe can reproduce a design matrix exactly and still be drawn ten times too big.

    That is the failure this check exists for — a missing exposure or a trials column that
    moved — and it is invisible in every one of the four checks `verify` runs.
    """
    train = _raw(120, seed=29)
    art = _artifact(train, FEATURES)
    frame = _counted(_design_from(train))
    fitted = M.fitted_values(art, COUNTED, frame,
                             M.family_draws(art, COUNTED, frame, 32, 5))

    honest = M.family_draws(art, COUNTED, frame, keep=32, seed=6)
    assert abs(M.predictive_bias(art, COUNTED, frame, honest, fitted)) < 0.05
    assert M.predictive_bias(art, COUNTED, frame, honest * 10.0, fitted) > 5.0

    with pytest.raises(AssertionError, match="wrong scale"):
        M.check_predictive("synthetic", {"predictive_bias": 0.4, "ecdf_band_mc": 0.0,
                                         "ecdf_band_gated": True,
                                         "predictive_draws": 200})


# ── The ECDF ribbon ───────────────────────────────────────────────────────────

def test_a_small_discrete_response_gets_one_grid_point_per_value():
    grid, kind = M.ecdf_grid(np.array([1.0, 1.0, 2.0, 4.0]))
    assert kind == "discrete" and list(grid) == [1.0, 2.0, 4.0]


def test_a_skewed_response_gets_quantiles_so_the_resolution_follows_the_curve():
    """Half a linear grid over season minutes sits in a tail holding a dozen rows."""
    rng = np.random.default_rng(30)
    values = rng.gamma(1.5, 200.0, 5000)
    grid, kind = M.ecdf_grid(values, cap=50)

    assert kind == "quantile" and len(grid) <= 50
    assert grid[0] == pytest.approx(values.min()) and grid[-1] == pytest.approx(values.max())
    # More than half the grid below the mean is the point: that is where the mass is.
    assert (grid < values.mean()).mean() > 0.5


def test_the_observed_curve_reaches_one_at_the_last_grid_point():
    observed = np.array([1.0, 2.0, 2.0, 5.0])
    draws = np.tile(observed, (16, 1))
    rows, _ = M.ecdf_rows("synthetic", "train", observed, draws)

    assert [r["value"] for r in rows] == [1.0, 2.0, 5.0]
    assert [r["observed"] for r in rows] == [0.25, 0.75, 1.0]
    assert rows[-1]["q2.5"] == 1.0 and rows[-1]["q97.5"] == 1.0


def test_the_ribbon_is_one_curve_per_draw_rather_than_the_pooled_predictive():
    """Pooling gives the predictive's own CDF, which has no width to compare against."""
    grid = np.array([0.0, 1.0])
    draws = np.stack([np.zeros(10), np.ones(10)])            # two opposite datasets
    curves = M.ecdf_curves(draws, grid)

    assert curves.shape == (2, 2)
    assert list(curves[0]) == [1.0, 1.0] and list(curves[1]) == [0.0, 1.0]
    rows, _ = M.ecdf_rows("synthetic", "train", np.array([0.0, 1.0]), draws)
    # The band spans both, where a pooled curve would sit at 0.5 with no spread at all.
    assert rows[0]["q2.5"] < 0.1 and rows[0]["q97.5"] > 0.9


def test_the_band_stability_statistic_is_zero_on_identical_halves_and_falls_with_draws():
    """The draw budget is checked rather than asserted, so the statistic has to bite."""
    assert M.band_stability(np.tile(np.linspace(0, 1, 8), (16, 1))) == 0.0

    rng = np.random.default_rng(31)
    grid = np.linspace(0.0, 3.0, 20)
    noisy = M.ecdf_curves(rng.gamma(2.0, 1.0, (64, 400)), grid)
    calm = M.ecdf_curves(rng.gamma(2.0, 1.0, (1024, 400)), grid)
    assert M.band_stability(noisy) > M.band_stability(calm) > 0.0


def test_a_ribbon_that_is_monte_carlo_noise_fails_the_build():
    with pytest.raises(AssertionError, match="Monte"):
        M.check_predictive("synthetic", {"predictive_bias": 0.0, "ecdf_band_mc": 0.2,
                                         "ecdf_band_gated": True,
                                         "predictive_draws": 200})
    # Under `BAND_MIN_ROWS` the statistic is the frame rather than the budget, and the
    # emitter reports it instead of failing: two validation cells give an ECDF of three
    # values, where a half-sample gap of 0.5 is arithmetic.
    M.check_predictive("synthetic", {"predictive_bias": 0.0, "ecdf_band_mc": 0.5,
                                     "ecdf_band_gated": False, "predictive_draws": 200})


# ── Calibration ───────────────────────────────────────────────────────────────

def test_one_outlier_does_not_collapse_the_calibration_grid():
    values = np.append(np.linspace(0.0, 10.0, 999), 1e6)
    edges = M.calibration_edges(values, bins=10)
    assert edges[-1] < 20.0 and len(edges) == 11


def test_the_clipped_tails_are_counted_in_the_end_bins_rather_than_dropped():
    fitted = np.append(np.linspace(0.0, 10.0, 400), [-500.0, 900.0])
    observed = fitted + 1.0
    rows = pd.DataFrame(M.calibration_rows(
        "synthetic", {"train": (fitted, observed),
                      "validation": (fitted[:50], observed[:50])}))

    counted = rows.groupby(["panel", "split"])["count"].sum()
    assert (counted[:, "train"] == len(fitted)).all()
    assert np.allclose(rows.groupby(["panel", "split"])["density"].sum(), 1.0)


def test_both_splits_share_one_calibration_grid_because_comparing_them_is_the_point():
    rng = np.random.default_rng(32)
    train = (rng.normal(50, 10, 800), rng.normal(50, 12, 800))
    val = (rng.normal(80, 10, 200), rng.normal(80, 12, 200))
    rows = pd.DataFrame(M.calibration_rows("synthetic", {"train": train,
                                                         "validation": val}))

    # Empty cells are dropped, so the two splits occupy different cells of one grid — the
    # claim is that it IS one grid, which is the pooled edge set and nothing else.
    pooled = M.calibration_edges(np.concatenate([train[0], val[0]]))
    for split, part in rows[rows["panel"] == "fitted_observed"].groupby("split"):
        assert np.isin(part["x_left"], pooled).all(), split
    # Validation is drawn 30 points to the right of train and lands there, rather than on
    # its own grid where the two panels would look identical.
    assert (rows[(rows["split"] == "validation")
                 & (rows["panel"] == "fitted_observed")]["x_index"].min()
            > rows[(rows["split"] == "train")
                   & (rows["panel"] == "fitted_observed")]["x_index"].min())
    assert set(rows["panel"]) == set(M.PANELS)


def test_the_residual_panel_is_the_residual_and_not_a_second_copy_of_the_observed():
    fitted = np.linspace(1.0, 100.0, 500)
    observed = fitted + 7.0
    rows = pd.DataFrame(M.calibration_rows(
        "synthetic", {"train": (fitted, observed), "validation": (fitted, observed)}))

    residual = rows[rows["panel"] == "residual_fitted"]
    assert residual["y_left"].min() <= 7.0 <= residual["y_right"].max()
    assert residual["y_right"].max() < 20.0          # residuals, not observations


# ── The bounded sample ────────────────────────────────────────────────────────

def test_the_sample_is_bounded_carries_the_residual_and_spans_the_frame():
    fitted = np.arange(10_000.0)
    observed = fitted + 3.0
    sample = M.sample_frame("synthetic", "train", fitted, observed, cap=250)

    assert len(sample) == 250
    assert sample["row"].iloc[0] == 0 and sample["row"].iloc[-1] == 9_999
    assert np.allclose(sample["residual"], 3.0)
    assert sample["fitted"].dtype == np.float32


def test_a_frame_smaller_than_the_cap_is_kept_whole():
    fitted, observed = np.arange(40.0), np.arange(40.0)
    assert len(M.sample_frame("synthetic", "validation", fitted, observed, cap=250)) == 40


# ── The shipped artifacts ─────────────────────────────────────────────────────
#
# The handful that read `outputs/predictions/` keep a *derived* quantity honest against the
# artifact it was derived from — the same job the PCA anchor tests do. Skipped rather than
# failed on a fresh checkout, since `make model-cards` needs `make posteriors` first.

def _shipped(name: str) -> pd.DataFrame:
    path = PREDICTIONS / name
    if not path.exists():
        pytest.skip(f"{path} is missing; run `make model-cards`")
    return pd.read_csv(path)


def _shipped_parquet(name: str) -> pd.DataFrame:
    path = PREDICTIONS / name
    if not path.exists():
        pytest.skip(f"{path} is missing; run `make model-cards`")
    return pd.read_parquet(path)


def test_the_shipped_artifacts_carry_no_split_outside_the_vocabulary():
    for name in ("model_card_features.csv", "model_card_feature_corr.csv"):
        assert set(_shipped(name)["split"].unique()) <= set(M.SPLITS)


def test_the_shipped_index_never_reaches_a_held_out_season():
    """The seasons a head was fitted over must stop before the test split begins."""
    index = _shipped("model_card_index.csv")
    seasons = sorted(s for s in index["last_season"].dropna().astype(str) if s)
    assert seasons, "no head reports a fit season span"
    assert max(seasons) <= "2021-22"
    assert set(index["fit_window"]) == {M.WINDOW}


def test_every_shipped_head_is_verified_and_declares_a_unit():
    index = _shipped("model_card_index.csv")
    assert index["verified"].all()
    assert (index["recipe_design_error"] <= 1e-9).all()
    assert index["unit"].notna().all() and (index["unit"].str.len() > 0).all()
    assert set(index["model_class"]) <= set(M.CLASS_LABELS)


def test_every_shipped_head_carding_features_also_cards_their_correlations():
    index = _shipped("model_card_index.csv").set_index("head")
    features = _shipped("model_card_features.csv")
    corr = _shipped("model_card_feature_corr.csv")

    for head, n_features in index["n_features"].items():
        carded = features[features["head"] == head]["feature"].nunique()
        assert carded == n_features, head
        square = corr[(corr["head"] == head) & (corr["split"] == "train")]
        assert len(square) == n_features ** 2, head


def test_every_shipped_density_pair_is_a_flagged_pair_of_its_own_head():
    """The panel a page can draw and the cell a reader clicked have to be the same pair."""
    index = _shipped("model_card_index.csv").set_index("head")
    corr = _shipped("model_card_feature_corr.csv")
    density = _shipped_parquet("model_card_feature_density.parquet")

    for head, count in index["n_density_pairs"].items():
        flagged = corr[(corr["head"] == head) & (corr["split"] == "train")
                       & corr["top_pair"] & (corr["i"] < corr["j"])]
        pairs = set(zip(flagged["feature_x"], flagged["feature_y"]))
        drawn = density[density["head"] == head]
        assert set(zip(drawn["feature_x"], drawn["feature_y"])) == pairs, head
        assert count == len(pairs), head
        # Two heads carry fewer than two features and legitimately have no pair at all.
        assert count == min(M.TOP_PAIRS, index.loc[head, "n_features"] *
                            (index.loc[head, "n_features"] - 1) // 2), head


def test_every_shipped_density_panel_counts_all_of_its_own_rows():
    density = _shipped_parquet("model_card_feature_density.parquet")
    keys = ["head", "feature_x", "feature_y", "split"]

    totals = density.groupby(keys)["count"].sum()
    sizes = density.groupby(keys)["n"].first()
    assert (totals == sizes).all()
    assert set(density["split"]) <= set(M.SPLITS)
    assert (density.groupby(keys)["x_index"].max() < M.DENSITY_BINS).all()


def test_every_shipped_coefficient_row_belongs_to_a_carded_head():
    index = _shipped("model_card_index.csv").set_index("head")
    coefficients = _shipped("model_card_coefficients.csv")

    assert set(coefficients["head"]) == set(index.index)
    counts = coefficients.groupby("head").size()
    assert (counts == index["n_terms"]).all()
    # One intercept per head, and every head reports a dispersion: the four families here
    # are all two-parameter likelihoods.
    roles = coefficients.groupby("head")["term_role"].apply(set)
    assert all({"intercept", "dispersion"} <= r for r in roles)


def test_every_carded_head_ships_all_three_predictive_artifacts():
    index = _shipped("model_card_index.csv")
    heads = set(index["head"])
    for name in ("model_card_ecdf.csv", "model_card_calibration.csv"):
        assert set(_shipped(name)["head"]) == heads, name
    assert set(_shipped_parquet("model_card_sample.parquet")["head"]) == heads
    assert (index["response_label"].str.len() > 0).all()
    assert set(index["fitted_source"]) <= {"head_predict", "predictive_mean"}


def test_the_shipped_ecdf_is_a_monotone_curve_under_an_ordered_band():
    ecdf = _shipped("model_card_ecdf.csv")
    levels = ["q2.5", "q10", "q25", "q50", "q75", "q90", "q97.5"]

    for (head, split), part in ecdf.groupby(["head", "split"]):
        part = part.sort_values("grid_index")
        assert part["observed"].is_monotonic_increasing, (head, split)
        assert part["observed"].iloc[-1] == pytest.approx(1.0), (head, split)
        for lo, hi in zip(levels, levels[1:]):
            assert (part[lo] <= part[hi] + 1e-12).all(), (head, split, lo)


def test_every_gated_head_ships_a_ribbon_that_is_stable_at_the_draw_budget():
    """The band is checked at 200 draws rather than assumed stable at it."""
    index = _shipped("model_card_index.csv")
    gated = index[index["ecdf_band_gated"]]

    assert len(gated) == len(index) - 1               # only overtime onset is too small
    assert (gated["ecdf_band_mc"] <= M.ECDF_BAND_TOL).all()
    assert (index["predictive_draws"] == M.PRED_DRAWS).all()


def test_every_shipped_predictive_reproduces_its_head_s_own_reported_mean():
    index = _shipped("model_card_index.csv")
    checked = index[index["predictive_check"] != "none"]

    assert len(checked) == len(index) - 1             # only the composition has no scale
    assert (checked["predictive_bias"].abs() <= M.PREDICTIVE_BIAS_TOL).all()
    assert index[index["predictive_check"] == "none"]["predictive_bias"].isna().all()


def test_the_shipped_predictive_stays_inside_its_row_and_draw_budget():
    """The composition is 631,158 rows; what the ribbon was drawn over is declared."""
    index = _shipped("model_card_index.csv").set_index("head")
    sample = _shipped_parquet("model_card_sample.parquet")

    # Blocked frames round up to whole team-games, so the cap is a target rather than a
    # ceiling — but not by a factor.
    assert (index["n_predictive_train"] <= M.PRED_ROWS * 1.2).all()
    assert (index["n_predictive_validation"] <= M.PRED_ROWS * 1.2).all()
    assert index.loc["composition", "predictive_rows_capped"]
    assert (sample.groupby(["head", "split"]).size() <= M.SAMPLE_ROWS).all()
    assert np.allclose(sample["residual"], sample["observed"] - sample["fitted"],
                       atol=1e-2)


def test_the_shipped_calibration_counts_every_row_it_was_given():
    """Tails are clipped into the end bins; nothing is silently outside the grid."""
    calibration = _shipped("model_card_calibration.csv")
    totals = calibration.groupby(["head", "split", "panel"]).agg(
        counted=("count", "sum"), n=("n", "first"))

    assert (totals["counted"] == totals["n"]).all()
    assert set(calibration["panel"]) == set(M.PANELS)
    assert (calibration["count"] > 0).all()           # empty cells are dropped, not shipped
