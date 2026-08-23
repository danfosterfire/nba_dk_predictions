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
from src.models.component_rates import CONVERSION_HEADS, COUNT_HEADS
from src.models.posteriors import DesignRecipe, PosteriorArtifact
from src.sim import season as S

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


def test_component_frames_rebuilds_through_the_preseason_aware_builder():
    """`component_frames` takes `stan_components.head_design`, never the plain builder.

    The fifth instance of the 6b wiring gap, and the one that got furthest: the emitter
    rebuilt all eleven heads at 8,630 rows through `component_rates.build_design` while the
    persisted posteriors were fitted on 6,382 covered-window rows carrying a preseason
    block. `verify`'s row-count check caught it at run time, so nothing shipped — but a card
    is the dashboard's only view of a head's coefficients, and a builder that silently drops
    five columns would have rendered a model nobody fitted.

    Checked on the import graph rather than on a string, the same way the split guard above
    is: `build_design` under any alias fails, and the three helpers the per-head cut needs
    have to actually be imported.
    """
    source = ast.parse(Path(M.__file__).read_text())
    frames = next(node for node in ast.walk(source)
                  if isinstance(node, ast.FunctionDef)
                  and node.name == "component_frames")
    imported = {alias.name for node in ast.walk(frames)
                if isinstance(node, (ast.Import, ast.ImportFrom))
                for alias in node.names}
    assert "build_design" not in imported, (
        "`component_frames` imports the plain `component_rates.build_design`, which carries "
        "no preseason columns and no covered-window cut — use `stan_components.head_design`")
    for needed in ("head_design", "covered_fitting_rows", "head_fitting_rows",
                   "head_features"):
        assert needed in imported, f"`component_frames` no longer imports `{needed}`"


# ── The chain role, pinned against the simulator ──────────────────────────────
#
# `HeadSpec.chain_role` is an *interpretation* — "this head is read when a season is
# drawn" — and this repo's rule for an interpretation on a page is that it carries a
# machine-checkable anchor, the way `pca.orient()` and `COMPONENT_BASIS` do. The anchor
# here is `src/sim/season.py` itself: a head declared to be in the draw path that nothing
# in `src/sim/` reads is exactly the claim that goes stale on the next refactor, and it
# goes stale silently, because the page keeps rendering.

def _sim_artifact_keys() -> set[str]:
    """Every posterior-artifact key `src/sim/` actually subscripts, read with `ast`.

    Static rather than dynamic because the alternative is loading twenty pickles and
    driving a season draw, and because the failure this guards is a *source* change. Two
    forms appear and both are resolved:

    - `artifacts["gp_duration"]` — a literal, taken as itself;
    - `artifacts[artifact_name(head)]` — the component loop, whose head list is
      `component_rates`' own, so it is expanded from that list through the simulator's own
      `artifact_name` rather than from a copy of the names kept here;
    - `artifacts[artifact_key]` — `season.family_artifact`, which since
      `docs/rookie-rates-plan.md` §5f picks one of the two rate families' twin of a
      component head. It expands to BOTH families, because both are read at draw time.

    Anything else raises, so a fourth addressing form fails this test instead of quietly
    widening the declared draw path.
    """
    keys: set[str] = set()
    for path in sorted(Path("src/sim").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Subscript)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "artifacts"):
                continue
            key = node.slice
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                keys.add(key.value)
            elif (isinstance(key, ast.Call) and isinstance(key.func, ast.Name)
                  and key.func.id == "artifact_name"):
                keys |= _component_keys()
            elif isinstance(key, ast.Name) and key.id == "artifact_key":
                keys |= _component_keys()
                keys |= {f"{S.ROOKIE_PREFIX}{name}" for name in _component_keys()}
            else:
                raise AssertionError(
                    f"{path}:{node.lineno} addresses the posterior bundle as "
                    f"`{ast.unparse(node)}`, which this scanner cannot resolve. Teach it "
                    f"the form or the declared draw path stops being checkable.")
    return keys


def _component_keys() -> set[str]:
    """The eleven rate heads' artifact names, through the simulator's own `artifact_name`."""
    return ({S.artifact_name(head) for head in COUNT_HEADS}
            | {S.artifact_name(f"{made}|{attempted}")
               for made, attempted in CONVERSION_HEADS})


def test_every_head_declares_a_chain_role_from_the_closed_vocabulary():
    for head, spec in M.SPECS.items():
        assert spec.chain_role in M.CHAIN_ROLES, (head, spec.chain_role)
    # The vocabulary earns its closure: every term is used, and both sides of the
    # `in_draw_path` split are populated, so the column is a distinction and not a constant.
    used = {spec.chain_role for spec in M.SPECS.values()}
    assert used == set(M.CHAIN_ROLES)
    assert {role.in_draw_path for role in M.CHAIN_ROLES.values()} == {True, False}


def test_an_undeclared_head_has_no_chain_role_rather_than_a_default():
    with pytest.raises(KeyError, match="no `HeadSpec`"):
        M.chain_role("a_head_that_was_never_carded")


def test_the_declared_draw_path_is_what_the_simulator_actually_reads():
    """The anchor. Declared draw path == the artifact keys `src/sim/` subscripts.

    Both directions matter and they fail differently. A head declared in the draw path
    that the simulator never loads is a page overstating what ships; a head the simulator
    loads that is declared out of it is a page understating it, which is how the
    Availability class intro came to describe five heads as two alternates.

    Since §5f of `docs/rookie-rates-plan.md` the simulator also reads a second rate family,
    and those eleven heads deliberately have no `HeadSpec` yet — §5h decides whether they
    get a card. That deferral is checked rather than exempted: the deferred keys must be
    EXACTLY the component heads' rookie twins, so a rookie head the simulator stops
    reading, or a twelfth one it starts reading, fails here.
    """
    keys = _sim_artifact_keys()
    deferred = {key for key in keys if M.deferred(key)}
    assert keys - deferred == M.draw_path_heads()
    assert deferred == {f"{S.ROOKIE_PREFIX}{name}" for name in _component_keys()}


def test_the_availability_chain_is_a_count_head_and_a_layout_head():
    """The specific claim step 2 of `docs/dashboard-revision-plan.md` corrected.

    `_sim_one` takes the games-played *count* from `availability` and lays those misses
    out with `allocate_spells` at `gp_duration`'s shape; the three tenure heads are not
    called at draw time. Asserted per head so a refactor that swapped which head does
    which fails here rather than on a page nobody re-reads.
    """
    assert M.SPECS["availability"].chain_role == "games_played_count"
    assert M.SPECS["gp_duration"].chain_role == "absence_layout"
    for head in ("gp_entry", "gp_exit", "gp_onset"):
        assert not M.chain_role(head).in_draw_path
    # And the layout step it names is a real function, imported by the simulator itself.
    assert S.allocate_spells is not None


def test_the_marginal_minutes_head_is_not_in_the_draw_path_and_the_composition_is():
    """The second correction the column turned up, and the surprising one.

    Both minutes heads ship, but the simulator reads only the composition: the marginal
    head's season-level spread arrives as `sim.minutes.player_season_sigma`, a constant
    `minutes_unification` calibrated against it and `rehydrate_composition` injects. So
    `artifacts["minutes"]` never appears in `src/sim/`, and a page saying "both heads are
    drawn from" would be wrong in a way no marginal metric could show.
    """
    assert not M.chain_role("minutes").in_draw_path
    assert M.chain_role("composition").in_draw_path
    assert "minutes" not in _sim_artifact_keys()


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


def test_a_graded_dispersion_refuses_the_generic_beta_binomial_branch():
    """The failure that moved availability onto `predict_samples`, pinned.

    `family_draws` has one dispersion per draw and no bin assignment to gather on, so a
    (draws x n_rho) `rho_draws` used to `reshape(-1)` into a longer vector that broadcast
    against `mu` — a star's dispersion landing on a fringe player's mean, on a card that
    renders perfectly. Raising by name is the only honest option here: the fix is a head
    with its own predictive, not a guess at the bins.
    """
    train = _raw(120, seed=54)
    art = _artifact(train, FEATURES)
    art.draws["rho_draws"] = np.tile(np.array([0.32, 0.27, 0.25, 0.21]), (200, 1))
    with pytest.raises(ValueError, match="graded by bin"):
        M.family_draws(art, COUNTED, _counted(_design_from(train)), keep=32, seed=1)

    # One column is the shared arm and still goes through, so the guard is on the grading
    # rather than on the shape — an artifact written as (draws x 1) is not a graded head.
    art.draws["rho_draws"] = np.full((200, 1), 0.28)
    assert M.family_draws(art, COUNTED, _counted(_design_from(train)),
                          keep=32, seed=1).shape[0] == 32


def test_availability_is_rehydrated_into_its_own_head_and_gathers_rho_per_row():
    """The card must draw through `StanAvailability`, dispersion bucket by bucket.

    Two things at once, because either alone would pass while the other was broken: that
    `_rehydrated` returns the real head (so `draw_predictive` is byte-identical to the
    head's own `predict_samples`), and that the head hands each row the `rho` of its own
    prior-MPG bucket rather than one number for everybody.
    """
    from src.models.stan_availability import ROLE_BIN_COL, ROLE_COL, StanAvailability

    train = _raw(150, seed=55)
    art = _artifact(train, FEATURES)
    art.head = "availability"
    art.extras = {"dispersion": "rho_draws", "role_rho": True, "n_rho": 4,
                  "rho_bin_column": ROLE_BIN_COL, "rho_bin_source": ROLE_COL,
                  "fit_first_season": "2012-13"}
    # Four wildly separated buckets, so a mis-gather cannot hide inside sampling noise.
    art.draws["rho_draws"] = np.tile(np.array([0.60, 0.30, 0.15, 0.05]), (200, 1))

    frame = _design_from(train)
    frame["team_games"] = 82.0
    frame[ROLE_COL] = np.tile([5.0, 18.0, 27.0, 40.0], len(frame) // 4 + 1)[:len(frame)]

    model = M._rehydrated("availability", art, cfg={}, keep=32)
    assert isinstance(model, StanAvailability)
    assert model.role_rho and model.n_rho == 4

    spec = M.RESPONSES["availability"]
    drawn = M.draw_predictive("availability", art, spec, frame, cfg={}, keep=32, seed=7)
    assert np.array_equal(drawn, model.predict_samples(frame, 7))
    assert drawn.shape == (32, len(frame)) and drawn.max() <= 82

    # The dispersion actually reached the draw: a fringe row's predictive is wider than a
    # star's at the same mean scale, which is the whole content of the graded arm.
    _, rhos = model.mu_draws(frame, 32)
    assert np.allclose(rhos[:, 0], 0.60) and np.allclose(rhos[:, 3], 0.05)


def test_a_rehydrated_head_refuses_a_dispersion_that_does_not_match_its_bins():
    """An artifact and an arm that disagree is a silent mis-gather, so it raises."""
    from src.models.stan_availability import rehydrate_availability

    train = _raw(60, seed=56)
    art = _artifact(train, FEATURES)
    art.extras = {"dispersion": "rho_draws", "role_rho": True}
    art.draws["rho_draws"] = np.full((200, 2), 0.3)      # two columns, four buckets
    with pytest.raises(ValueError, match="dispersion column"):
        rehydrate_availability(art, keep=8)


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


def _summary(**overrides) -> dict:
    """A passing predictive summary, for the gates to be pushed off one at a time."""
    return {"predictive_bias": 0.0, "ecdf_band_mc": 0.001, "ecdf_band_gated": True,
            "predictive_draws": 200, "quantile_ks_mc": 0.001,
            "quantile_ks_gated": True, **overrides}


def test_a_ribbon_that_is_monte_carlo_noise_fails_the_build():
    with pytest.raises(AssertionError, match="Monte"):
        M.check_predictive("synthetic", _summary(ecdf_band_mc=0.2))
    # Under `BAND_MIN_ROWS` the statistic is the frame rather than the budget, and the
    # emitter reports it instead of failing: two validation cells give an ECDF of three
    # values, where a half-sample gap of 0.5 is arithmetic.
    M.check_predictive("synthetic", _summary(ecdf_band_mc=0.5, ecdf_band_gated=False))


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


def test_the_raw_residual_panel_is_gone_rather_than_shipped_and_never_drawn():
    """`residual_fitted` was replaced by the scaled quantile residual on 2026-08-10.

    An artifact half that no page reads is drift, and the raise names its replacement — a
    caller reaching for the old panel gets the reason rather than an empty frame.
    """
    fitted = np.linspace(1.0, 100.0, 500)
    observed = fitted + 7.0
    rows = pd.DataFrame(M.calibration_rows(
        "synthetic", {"train": (fitted, observed), "validation": (fitted, observed)}))

    assert set(rows["panel"]) == {"fitted_observed"} == set(M.PANELS)
    with pytest.raises(KeyError, match="model_card_quantile"):
        M._panel_values("residual_fitted", fitted, observed)


# ── The scaled quantile residual ──────────────────────────────────────────────
#
# Every failure mode here renders as a good-looking picture, which is the reason each one
# gets a case: a QQ that is uniform because the residual was computed against the wrong
# thing, a seed that moves the panel on every rebuild, a rank transform taken inside the
# 2,000-row overlay rather than over the frame it is drawn on, a quartile line through three
# rows, a KS distance that is quietly a reading of the draw budget.

def _calibrated(n: int = 4_000, draws: int = 200, seed: int = 5,
                shift: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """`(draws x rows, observed)` from a Poisson whose observed comes from the same law.

    `shift` moves the *observed* off the predictive, which is what a miscalibrated head
    looks like from here — the residual should then pile up at one end.
    """
    rng = np.random.default_rng(seed)
    rate = rng.gamma(6.0, 1.5, n)
    return (rng.poisson(rate[None, :], (draws, n)).astype(float),
            rng.poisson(rate + shift).astype(float))


def test_a_calibrated_head_puts_its_scaled_residual_on_the_uniform_diagonal():
    """The property the whole block rests on, and it is a property of the arithmetic."""
    draws, observed = _calibrated()
    u = M.scaled_residuals(draws, observed, M.quantile_seed("synthetic", "train"))

    assert u.min() >= 0.0 and u.max() <= 1.0
    assert M.ks_uniform(u) < 0.03
    rows = pd.DataFrame(M.qq_rows(u))
    assert np.abs(rows["x"] - rows["y"]).max() < 0.05
    # The envelope is the exact Beta band on each order statistic, so it brackets the
    # diagonal rather than the data — a band fitted to the points would never flag anything.
    assert (rows["lo"] <= rows["x"]).all() and (rows["hi"] >= rows["x"]).all()


def test_a_head_predicting_the_wrong_level_leaves_the_diagonal_rather_than_hiding():
    """A residual that cannot see a miss is worse than no panel; this one is the control."""
    draws, observed = _calibrated(shift=4.0)
    u = M.scaled_residuals(draws, observed, M.quantile_seed("synthetic", "train"))

    assert M.ks_uniform(u) > 0.2
    assert u.mean() > 0.6                       # observed above the predictive, so u is high


def test_the_randomization_is_seeded_per_head_and_split_so_a_rebuild_does_not_move_it():
    """Otherwise a reader cannot tell a refit from an RNG — the whole point of `_seed`."""
    draws, observed = _calibrated(n=500)
    seed = M.quantile_seed("synthetic", "train")

    assert np.array_equal(M.scaled_residuals(draws, observed, seed),
                          M.scaled_residuals(draws, observed, seed))
    assert M.quantile_seed("synthetic", "train") != M.quantile_seed("synthetic",
                                                                    "validation")
    # And a different stream from the draws themselves: `u`'s uniforms must not be the same
    # sequence the replicate datasets came from.
    assert M.quantile_seed("synthetic", "train") != M._seed("synthetic", "train")


def test_the_randomization_is_what_keeps_a_discrete_predictive_from_looking_miscalibrated():
    """DHARMa's own reason for randomizing, as a measurement rather than a docstring.

    The non-randomized quantile of a discrete predictive is not uniform even under a perfect
    model — it is biased low, because every tie is counted as a miss.
    """
    draws, observed = _calibrated(n=4_000)
    seed = M.quantile_seed("synthetic", "train")
    randomized = M.scaled_residuals(draws, observed, seed)
    plain = (draws < observed[None, :]).mean(axis=0)

    assert M.ks_uniform(plain) > 3 * M.ks_uniform(randomized)


def test_the_ks_stability_statistic_is_the_draw_budget_and_falls_with_draws():
    """The one bar on this half, so it has to bite — and it has to be about the budget."""
    seed = M.quantile_seed("synthetic", "train")
    noisy = M.ks_stability(*_calibrated(n=2_000, draws=8), seed=seed)
    calm = M.ks_stability(*_calibrated(n=2_000, draws=800), seed=seed)

    assert noisy > calm >= 0.0
    assert np.isnan(M.ks_stability(np.zeros((2, 5)), np.zeros(5), seed))


def test_a_quantile_residual_that_is_monte_carlo_noise_fails_the_build():
    with pytest.raises(AssertionError, match="quantized"):
        M.check_predictive("synthetic", _summary(quantile_ks_mc=0.3))
    # Ungated below `BAND_MIN_ROWS`, the same rule the ribbon follows: the overtime-onset
    # head's two validation cells make the statistic a property of the frame.
    M.check_predictive("synthetic", _summary(quantile_ks_mc=0.5, quantile_ks_gated=False))
    # And the KS distance ITSELF is never a bar — a head can sit far from uniform and still
    # ship, because at these sample sizes a uniformity test rejects everything.
    M.check_predictive("synthetic", _summary(quantile_ks_mc=0.001))


def test_the_rank_transform_is_uniform_by_construction_and_averages_its_ties():
    """What makes one panel comparable across a count head and a conversion head."""
    rank = M.rank_uniform(np.array([10.0, 20.0, 30.0, 40.0]))
    assert np.allclose(rank, [0.125, 0.375, 0.625, 0.875])

    # Three ties take the average of ranks 1, 2 and 3, so all three sit at (2 − 0.5)/4 —
    # one column of cells rather than an arbitrary ordering of equals spread across three.
    tied = M.rank_uniform(np.array([5.0, 5.0, 5.0, 9.0]))
    assert np.allclose(tied[:3], 0.375) and tied[3] == 0.875


def test_both_axes_of_the_residual_panel_are_the_unit_square_on_both_splits():
    """The one panel in the contract with no pooled edge set — the transform IS the scale."""
    rng = np.random.default_rng(4)
    train = pd.DataFrame(M.residual_rows(rng.random(3_000), rng.random(3_000)))
    val = pd.DataFrame(M.residual_rows(rng.random(200), rng.random(200)))

    for part in (train, val):
        assert part["x_left"].min() == 0.0 and part["x_right"].max() == 1.0
        assert part["y_left"].min() >= 0.0 and part["y_right"].max() <= 1.0
        assert np.isclose(part["density"].sum(), 1.0)
    assert set(train["x_left"]) >= set(val["x_left"])       # one grid, not two


def test_every_row_lands_in_exactly_one_residual_cell_including_the_two_extremes():
    u = np.array([0.0, 1.0, 0.5, 0.5])
    rows = pd.DataFrame(M.residual_rows(u, np.array([0.0, 1.0, 0.25, 0.75])))
    assert rows["count"].sum() == 4


def test_the_quantile_lines_are_flat_at_their_levels_when_the_residual_is_uniform():
    rng = np.random.default_rng(9)
    u, rank = rng.random(20_000), rng.random(20_000)
    lines = pd.DataFrame(M.quantile_lines(u, rank))

    assert set(lines["level"]) == set(M.QUANTILE_LEVELS)
    assert np.abs(lines["y"] - lines["level"]).max() < 0.05
    assert lines["x_index"].nunique() == M.RESIDUAL_BINS


def test_a_bin_with_too_few_rows_is_a_gap_rather_than_a_line_through_three_points():
    rng = np.random.default_rng(10)
    # Everything in the left tenth of the x axis except five rows on the right.
    rank = np.concatenate([rng.uniform(0, 0.1, 500), rng.uniform(0.9, 1.0, 5)])
    lines = pd.DataFrame(M.quantile_lines(rng.random(len(rank)), rank))

    assert lines["x_index"].max() < M.RESIDUAL_BINS - 1
    assert (lines["count"] >= M.QUANTILE_MIN_ROWS).all()


def test_the_three_panels_share_one_head_split_and_ks_so_a_page_reads_them_together():
    rng = np.random.default_rng(12)
    u, rank = rng.random(2_000), rng.random(2_000)
    rows, ks = M.quantile_tables("synthetic", "validation", u, rank)
    frame = pd.DataFrame(rows)

    assert set(frame["panel"]) == set(M.QUANTILE_PANELS)
    assert (frame["ks"] == ks).all() and (frame["n"] == 2_000).all()
    assert set(frame["split"]) == {"validation"}


def test_a_head_is_declared_out_of_scope_rather_than_drawn_wrongly():
    """The mechanism, exercised against an empty declaration — see `QUANTILE_OUT_OF_SCOPE`.

    The composition is the head this was opened for and it is **in** scope: `u` is a
    function of the draws and the observed, and its `predict_samples` draws minutes in a
    team-game. `predictive_check = none` is about its *fitted* value and does not decide
    this.
    """
    assert M.quantile_scope("composition") == ("drawn", "")
    assert M.RESPONSES["composition"].check == "none"

    reason = "its response is a linear predictor and not the observable"
    M.QUANTILE_OUT_OF_SCOPE["synthetic"] = reason
    try:
        assert M.quantile_scope("synthetic") == ("not_applicable", reason)
    finally:
        del M.QUANTILE_OUT_OF_SCOPE["synthetic"]


# ── The bounded sample ────────────────────────────────────────────────────────

def test_the_sample_carries_both_panels_coordinates_and_spans_the_frame():
    fitted = np.arange(10_000.0)
    observed = fitted + 3.0
    u = np.linspace(0.0, 1.0, 10_000)
    sample = M.sample_frame("synthetic", "train", fitted, observed, u,
                            M.rank_uniform(fitted), cap=250)

    assert len(sample) == 250
    assert sample["row"].iloc[0] == 0 and sample["row"].iloc[-1] == 9_999
    # `u` and the rank are taken at the SAME thinned rows as the fitted value, so a point in
    # the calibration panel and a point in the residual panel are one row of the frame.
    assert np.allclose(sample["u"], u[sample["row"]], atol=1e-6)
    assert np.allclose(sample["predicted_rank"],
                       (sample["row"] + 0.5) / 10_000, atol=1e-6)
    assert sample["fitted"].dtype == np.float32 and sample["u"].dtype == np.float32


def test_a_head_out_of_quantile_scope_still_ships_its_sample_with_empty_columns():
    """A missing column would change the parquet's shape per head; NaN does not."""
    fitted, observed = np.arange(40.0), np.arange(40.0)
    sample = M.sample_frame("synthetic", "validation", fitted, observed, cap=250)

    assert len(sample) == 40
    assert set(sample.columns) == {"head", "split", "row", "fitted", "observed", "u",
                                   "predicted_rank"}
    assert sample["u"].isna().all() and sample["predicted_rank"].isna().all()


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


def test_the_shipped_index_separates_the_fit_window_from_the_season_truncation():
    """Two season axes on one row, and a page must be able to tell them apart.

    `fit_window` is which split may be fitted and is one value for the whole file;
    `fit_first_season` is which suffix of it a head chose, and is empty for the heads that
    fit whatever the window offers. Where a head does truncate, its fitted span has to
    *start* at the truncation — a card whose population began earlier would be describing
    rows the coefficients never saw.
    """
    index = _shipped("model_card_index.csv").set_index("head")
    assert "fit_first_season" in index.columns
    assert index["fit_window"].nunique() == 1

    truncated = index[index["fit_first_season"].astype(str).str.len() > 0]
    for head, row in truncated.iterrows():
        assert str(row["first_season"]) == str(row["fit_first_season"]), head
    if "availability" in index.index:
        assert str(index.loc["availability", "fit_first_season"]) == "2012-13"


def test_every_shipped_head_is_verified_and_declares_a_unit():
    index = _shipped("model_card_index.csv")
    assert index["verified"].all()
    assert (index["recipe_design_error"] <= 1e-9).all()
    assert index["unit"].notna().all() and (index["unit"].str.len() > 0).all()
    assert set(index["model_class"]) <= set(M.CLASS_LABELS)


def test_the_shipped_index_carries_each_head_s_role_in_the_shipped_chain():
    """The column the dashboard reads instead of typing a claim about the simulator.

    Asserted on the *artifact* rather than on `SPECS`, because the page reads the artifact
    — a column emitted under the wrong name, or dropped by an edit to `index_row`, would
    leave the page silently falling back to its empty-string branch.
    """
    index = _shipped("model_card_index.csv").set_index("head")
    for column in ("chain_role", "chain_role_label", "in_draw_path", "chain_role_note"):
        assert column in index.columns, column
        assert index[column].notna().all(), column
    assert set(index["chain_role"]) <= set(M.CHAIN_ROLES)
    assert set(index[index["in_draw_path"]].index) == M.draw_path_heads()
    # Label and note both carry text, since either one empty renders as a dash on the page.
    assert (index["chain_role_label"].str.len() > 0).all()
    assert (index["chain_role_note"].str.len() > 0).all()


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


def test_every_carded_head_ships_all_four_predictive_artifacts():
    index = _shipped("model_card_index.csv")
    heads = set(index["head"])
    for name in ("model_card_ecdf.csv", "model_card_calibration.csv"):
        assert set(_shipped(name)["head"]) == heads, name
    assert set(_shipped_parquet("model_card_sample.parquet")["head"]) == heads
    assert (index["response_label"].str.len() > 0).all()
    assert set(index["fitted_source"]) <= {"head_predict", "predictive_mean"}
    # The quantile artifact carries every head declared in scope, and only those.
    drawn = set(index.loc[index["quantile_scope"] == "drawn", "head"])
    assert set(_shipped("model_card_quantile.csv")["head"]) == drawn
    assert set(index["quantile_scope"]) <= {"drawn", "not_applicable"}


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
    # The overlay's four coordinates: two for the calibration density, two for the residual
    # panel, on the same rows. Both of the latter are transforms onto [0, 1].
    assert sample["u"].between(0.0, 1.0).all()
    assert sample["predicted_rank"].between(0.0, 1.0).all()


def test_the_shipped_calibration_counts_every_row_it_was_given():
    """Tails are clipped into the end bins; nothing is silently outside the grid."""
    calibration = _shipped("model_card_calibration.csv")
    totals = calibration.groupby(["head", "split", "panel"]).agg(
        counted=("count", "sum"), n=("n", "first"))

    assert (totals["counted"] == totals["n"]).all()
    assert set(calibration["panel"]) == set(M.PANELS)
    assert (calibration["count"] > 0).all()           # empty cells are dropped, not shipped


def test_the_shipped_quantile_residual_is_bounded_and_carries_all_three_panels():
    quantile = _shipped("model_card_quantile.csv")
    index = _shipped("model_card_index.csv").set_index("head")

    for (head, split), part in quantile.groupby(["head", "split"]):
        assert set(part["panel"]) >= {"qq", "residual"}, (head, split)
        # A scaled residual is a probability and a rank transform is a share of the frame:
        # both axes are [0, 1] everywhere, which is what puts twenty differently-scaled
        # heads on one pair of panels.
        assert part["x"].between(0.0, 1.0).all(), (head, split)
        assert part["y"].between(0.0, 1.0).all(), (head, split)
        cells = part[part["panel"] == "residual"]
        assert cells["count"].sum() == part["n"].iloc[0], (head, split)
        assert (index.loc[head, f"quantile_ks_{split}"]
                == pytest.approx(part["ks"].iloc[0], abs=1e-6)), (head, split)


def test_every_gated_head_ships_a_ks_distance_that_is_stable_at_the_draw_budget():
    """The bar is on the budget, never on the distance — see `quantile_tables`."""
    index = _shipped("model_card_index.csv")
    gated = index[index["quantile_ks_gated"]]

    assert len(gated) == len(index) - 1               # only overtime onset is too small
    assert (gated["quantile_ks_mc"] <= M.KS_MC_TOL).all()
    # The distances themselves are unconstrained, and several heads sit far from uniform:
    # `fg2m_given_fg2a` reads 0.15 on validation and ships, because a uniformity test at
    # these sample sizes rejects everything and the reading is the size of the miss.
    assert index["quantile_ks_validation"].max() > 0.1
    assert set(index["quantile_weighting"]) == {"unweighted", "expanded"}
    # The two collapsed-cell heads are the expanded ones: their frames are cells carrying a
    # multiplicity, so the residual is one row per spell rather than one per cell.
    assert set(index.loc[index["quantile_weighting"] == "expanded", "head"]) == {
        "gp_duration", "game_length_depth"}


def test_the_availability_cards_coefficients_carry_the_mixture_block():
    """A card that omitted them would describe the single-component head.

    Eleven terms, and the check that matters is on the two that are easy to get subtly
    wrong: `pi:` prefixes keep the mixture's `age` coefficient from being read as the mean's
    (different coefficients on the same column, through different links), and `gamma`
    unstandardizes against `pi`'s OWN scaler rather than the mean's nineteen-column one.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from src.models import model_cards as MC

    art = _artifact(_raw(60, seed=91), FEATURES)
    pi_features = ["age", "gp_share_lag1"]
    pi_scaler = StandardScaler().fit(np.array([[24.0, 0.5], [34.0, 0.9]]))
    art.recipe.pi_features = list(pi_features)
    art.recipe.pi_scaler = pi_scaler
    art.draws["theta_draws"] = np.full(200, 0.11)
    art.draws["mu_low_draws"] = np.full(200, 0.10)
    art.draws["rho_low_draws"] = np.full(200, 0.05)
    art.draws["gamma_draws"] = np.tile(np.array([[0.4, -0.7]]), (200, 1))

    rows = {r["term"]: r for r in MC.coefficient_rows("availability", art)}
    assert {"theta", "mu_low", "rho_low", "pi:age", "pi:gp_share_lag1"} <= set(rows)
    # The roles split by what the term IS, because the dashboard's panel splits on them:
    # the three scalars render beside the dispersion, the eight slopes inside the panel.
    assert rows["theta"]["term_role"] == "dispersion"
    assert rows["pi:age"]["term_role"] == "coefficient"
    assert rows["theta"]["term_family"] == "mixture"
    np.testing.assert_allclose(rows["pi:gp_share_lag1"]["mean"], -0.7)
    # `pi`'s own scaler, not the mean block's — the two are fitted on different columns.
    np.testing.assert_allclose(rows["pi:age"]["scaler_center"], pi_scaler.mean_[0])
    assert rows["pi:age"]["scaler_center"] != rows["age"]["scaler_center"]


def test_a_head_with_no_mixture_emits_no_mixture_terms():
    """Checked on the term NAMES, not on a role — the roles are shared with other blocks.

    `theta` files under `dispersion` and the `pi:` slopes under `coefficient`, so a check on
    `term_role` would pass vacuously on every head in the project.
    """
    from src.models import model_cards as MC

    rows = MC.coefficient_rows("availability", _artifact(_raw(60, seed=92), FEATURES))
    terms = {r["term"] for r in rows}
    assert not (terms & {"theta", "mu_low", "rho_low"})
    assert not [t for t in terms if t.startswith("pi:")]
    assert not [r for r in rows if r["term_family"] == "mixture"]


def test_the_mixture_weights_are_eight_families_not_one_collapsible_basis():
    """The collapse toggle keeps one row per `term_family`, labelled "widest of N bases".

    That is right for a spline basis over one underlying quantity and nonsense for eight
    different covariates on `pi`, so each has to be its own family — exactly how the mean
    block's features are treated.
    """
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    from dashboard import model_cards as DMC
    from src.models import model_cards as MC

    art = _artifact(_raw(60, seed=93), FEATURES)
    pi_features = ["age", "gp_share_lag1", "n_spells_lag1"]
    art.recipe.pi_features = list(pi_features)
    art.recipe.pi_scaler = StandardScaler().fit(np.array([[24.0, 0.5, 1.0],
                                                          [34.0, 0.9, 4.0]]))
    art.draws["theta_draws"] = np.full(200, 0.11)
    art.draws["mu_low_draws"] = np.full(200, 0.10)
    art.draws["rho_low_draws"] = np.full(200, 0.05)
    art.draws["gamma_draws"] = np.tile(np.array([[0.4, -0.7, 0.2]]), (200, 1))

    frame = pd.DataFrame(MC.coefficient_rows("availability", art))
    families = frame.loc[frame["term"].str.startswith("pi:"), "term_family"]
    assert len(set(families)) == len(pi_features)
    collapsed = DMC.coefficient_panel(frame, "availability", collapse=True)
    assert sum(t.startswith("pi:") for t in collapsed["term"]) == len(pi_features)
