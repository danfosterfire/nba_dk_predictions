"""Tests for §5h's recovery readout and the tensor-label plumbing under it.

`docs/rookie-rates-plan.md` §7i measures what the lag ladder and the rookie head give back
of Session 1's floor, and three things here can be wrong while producing a complete,
well-ordered table:

- **the rungs stop being nested**, which turns "what this change added" from a difference of
  two boards into a difference of two unrelated boards, and every increment in the doc
  becomes uninterpretable while still summing to something;
- **the label stops travelling**, so a run reads the shipped tensor, or worse writes over an
  audited artifact — the failure the label exists to prevent, and one that leaves a green
  run and a moved figure;
- **the provenance join silently matches nothing**, which would put every board row in the
  `unpriced` bucket and report a floor of zero at every rung.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sim import bracket as B
from src.sim import draft as D
from src.sim import draft_room as R
from src.sim import rookie_recovery as RR
from src.sim import season as SEASON


# ── Synthetic builders ────────────────────────────────────────────────────────

def _frame(n: int = 12) -> pd.DataFrame:
    return pd.DataFrame({"player_id": np.arange(1, n + 1)})


def _tensor(player_id, family, rung) -> dict:
    return {"player_id": np.asarray(player_id, dtype=np.int64),
            "unit_family": np.asarray(family, dtype=object),
            "lag_rung": np.asarray(rung, dtype=object)}


# ── 1. The provenance join ────────────────────────────────────────────────────

def test_board_rows_the_tensor_does_not_carry_are_the_population_not_an_error():
    """The board is wider than the tensor and that gap IS the floor.

    A left join that raised, or one that dropped, would delete the rows the whole readout
    is about.
    """
    frame = _frame(6)
    tensor = _tensor([2, 4], ["veteran", "rookie"], ["veteran", "true_rookie"])
    prov = RR.board_provenance(frame, tensor)

    assert list(prov["lag_rung"]) == ["unpriced", "veteran", "unpriced", "true_rookie",
                                      "unpriced", "unpriced"]
    assert list(prov["unit_family"]) == ["unpriced", "veteran", "unpriced", "rookie",
                                         "unpriced", "unpriced"]
    assert len(prov) == len(frame)


def test_the_provenance_join_is_positional_rather_than_by_order():
    """The tensor's row order is its own; a join by position would mislabel every row."""
    frame = _frame(4)
    tensor = _tensor([4, 1], ["rookie", "veteran"], ["true_rookie", "veteran"])
    prov = RR.board_provenance(frame, tensor)

    assert prov["lag_rung"].iloc[0] == "veteran"
    assert prov["lag_rung"].iloc[3] == "true_rookie"


# ── 2. The ladder ─────────────────────────────────────────────────────────────

def _prov(rungs: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"lag_rung": rungs,
                         "unit_family": ["x"] * len(rungs)})


def test_the_rungs_are_nested_and_the_last_one_is_the_whole_priceable_board():
    """Nesting is what makes an increment attributable to one change rather than to two."""
    prov = _prov(["veteran", "veteran", "returnee_lag2", "true_rookie", "unpriced"])
    masks = RR.rung_masks(prov, np.ones(5, dtype=bool))

    assert list(masks["veteran"]) == [True, True, False, False, False]
    assert list(masks["ladder"]) == [True, True, True, False, False]
    assert list(masks["rookie"]) == [True, True, True, True, False]
    assert masks["unrestricted"].all()
    for inner, outer in (("veteran", "ladder"), ("ladder", "rookie"),
                         ("rookie", "unrestricted")):
        assert (masks[inner] <= masks[outer]).all(), (inner, outer)


def test_a_row_the_tensor_does_not_price_never_enters_a_restricted_rung():
    """`scorable` and the design label are two different questions and both must hold.

    A row can carry a design and still be padded out of the tensor, and the sweep's own
    restriction reads `scorable` — so a rung built from the label alone would put a
    zero-scored player on our seat's board, which is the bias `priceable_room` exists for.
    """
    prov = _prov(["veteran", "true_rookie", "returnee_lag2"])
    scorable = np.array([True, False, True])
    masks = RR.rung_masks(prov, scorable)

    assert list(masks["rookie"]) == [True, False, True]
    assert masks["unrestricted"].all()


def test_every_admitted_ladder_rung_name_is_the_ladders_own_vocabulary():
    """The rung labels are `component_rates`' — a private copy would drift silently."""
    from src.models.component_rates import LADDER_RUNGS
    from src.models.rookie_rates import ROOKIE_GROUP

    added = dict(RR.RUNGS)
    assert set(added["ladder"]) == set(LADDER_RUNGS)
    assert added["rookie"] == (ROOKIE_GROUP,)
    assert added["veteran"] == ("veteran",)


# ── 3. The census ─────────────────────────────────────────────────────────────

def test_the_census_counts_only_the_market_priced_rows_it_says_it_does():
    frame = _frame(4)
    adp = np.array([1.0, np.nan, 3.0, np.nan])
    realized = np.array([100.0, 200.0, 300.0, 400.0])
    out = RR.census(frame, adp, realized, np.array([True, True, True, False]))

    assert out["n_board"] == 3
    assert out["n_priced"] == 2
    assert out["realized_mean_priced"] == pytest.approx(200.0)
    assert out["realized_total"] == pytest.approx(600.0)


def test_a_rung_with_no_priced_row_reports_nan_rather_than_zero():
    """Zero is a realized total a player can have; `nan` is the absence of one."""
    out = RR.census(_frame(2), np.array([np.nan, np.nan]), np.array([1.0, 2.0]),
                    np.ones(2, dtype=bool))
    assert out["n_priced"] == 0
    assert np.isnan(out["realized_mean_priced"])


# ── 4. The label, which must travel or an audited artifact moves ─────────────

def test_the_tensor_label_reaches_the_filename_and_the_field_cache():
    features = Path("data/features")
    assert (B.load_tensor.__defaults__ or ())[-1] == ""
    assert (R.field_artifact(features, "2022-23", "_x")
            == features / "draft_room_field_2022-23_x.npz")
    assert (R.field_artifact(features, "2022-23")
            == features / "draft_room_field_2022-23.npz")


def test_a_missing_labelled_tensor_names_the_label_in_its_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="--tensor-label _x"):
        B.load_tensor(tmp_path, "2022-23", "_x")


def test_a_tensor_written_before_the_two_families_reads_back_as_one(tmp_path):
    """`load_tensor` must not raise on the tensors already on disk.

    Every artifact the project has shipped was drawn before §5f, and a consumer asking for
    the family column would otherwise have to know the file's vintage.
    """
    np.savez(tmp_path / "sim_tensor_2022-23.npz",
             dk_pts=np.zeros((2, 3, 4), dtype=np.float32),
             player_id=np.array([7, 9]), tournament_round=np.array([1, 1, 2]),
             season=np.array("2022-23"), fit_window=np.array("train"),
             n_sims=np.array(4))
    out = B.load_tensor(tmp_path, "2022-23")

    assert list(out["unit_family"]) == [SEASON.VETERAN_FAMILY] * 2
    assert list(out["lag_rung"]) == ["veteran"] * 2


def test_a_labelled_tensor_run_labels_gate_a_by_default():
    """Gate A is one pooled table `make docs-audit` re-derives extremes from.

    A variant population writing into it moves audited figures by REPLACING that season's
    rows — `merge_gate` merges by season, so nothing would even look wrong. Asserted on the
    source rather than by running the simulator, which needs posteriors and minutes.
    """
    tree = ast.parse(Path("src/sim/season.py").read_text())
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    assigns = [n for n in ast.walk(run)
               if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == "label" for t in n.targets)]
    assert assigns, "`run` no longer defaults the gate label from the tensor label"
    assert ast.unparse(assigns[0].value) == "label or tensor_label"


def test_the_sweeps_artifact_suffix_carries_the_tensor_label():
    """Otherwise a rookie-inclusive replay overwrites the audited `strategy_*.csv` set."""
    source = Path("src/sim/strategy.py").read_text()
    assert "suffix += tensor_label" in source
    tree = ast.parse(source)
    run = next(n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == "run")
    assert "tensor_label" in {a.arg for a in run.args.args + run.args.kwonlyargs}


def test_the_floor_table_reads_the_symmetric_arm_of_its_own_vintage():
    """A labelled asymmetric run against the unlabelled baseline compares two changes."""
    source = Path("src/sim/strategy.py").read_text()
    assert 'f"strategy_realized{tensor_label}.csv"' in source
    assert 'f"strategy_rookie_floor{tensor_label}.csv"' in source


def test_no_function_in_src_sim_reads_a_tensor_label_it_does_not_declare():
    """A NameError that only fires on a flag nothing in the default chain passes.

    Threading the label through five modules meant editing call sites that look identical
    and are not — `board_games_played` is called from both the sweep and the pick-log
    stake, and only one of them has a label to give it. The bug is invisible until somebody
    runs the other target, so it is pinned rather than reviewed.
    """
    def bound(fn: ast.AST) -> set[str]:
        args = getattr(fn, "args", None)
        out = {a.arg for a in (args.args + args.kwonlyargs)} if args else set()
        return out | {n.id for n in ast.walk(fn)
                      if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}

    def check(node: ast.AST, path: Path, enclosing: set[str]) -> None:
        """Depth-first, carrying the enclosing scopes' names — a nested helper closing over
        its parent's `label` is legal Python and must not read as the bug."""
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope = enclosing | bound(child)
                uses = {n.id for n in ast.walk(child)
                        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                for name in ("tensor_label", "label"):
                    if name in uses:
                        assert name in scope, (f"{path}:{child.lineno} {child.name} "
                                               f"reads `{name}` and nothing binds it")
                check(child, path, scope)
            else:
                check(child, path, enclosing)

    for path in sorted(Path("src/sim").glob("*.py")):
        tree = ast.parse(path.read_text())
        module_level = {n.id for node in ast.walk(tree)
                        if isinstance(node, ast.Assign)
                        for n in node.targets if isinstance(n, ast.Name)}
        check(tree, path, module_level)


# ── 5. The split guard, pinned the way the sweep's is ─────────────────────────

def test_the_recovery_readout_reaches_the_split_only_through_selection_split():
    """It scores realized seasons, so a test season leaking in is a spent split."""
    names = set()
    for node in ast.walk(ast.parse(Path("src/sim/rookie_recovery.py").read_text())):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[-1])
    assert "assert_season_allowed" in names
    assert "validation_seasons" in names
