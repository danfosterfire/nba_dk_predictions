"""The end-of-project reading, and the production fit that may only follow it.

Two acts read the held-out seasons and they are not the same act. `src/final_evaluation.py`
*measures* on them once; `make posteriors-production` *fits* on them so the upcoming
season's board is not throwing two years of data away. The ordering between the two is the
thing worth pinning: once the production fit exists, every model that could have been
compared against the test seasons has seen them, and there is no unbiased measurement left
to take.
"""

import ast
from pathlib import Path

import pandas as pd
import pytest

from src import final_evaluation
from src.models import posteriors
from src.models.held_out import HeldOutLocked


def _cfg(tmp_path: Path) -> dict:
    return {"data": {"features_dir": str(tmp_path / "features")},
            "evaluation": {"predictions_dir": str(tmp_path / "predictions")}}


def _manifest(cfg: dict, window: str, rows: list[dict]) -> Path:
    directory = posteriors.posteriors_dir(cfg, window)
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "manifest.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    return dest


def _head(name: str, **overrides) -> dict:
    row = {"head": name, "family": "negbinom", "variant": "log_own_spline",
           "fit_first_season": "", "n_features": 24, "preseason": True,
           "preseason_columns": "own_delta_shrunk", "player_season_effect": False,
           "sigma_u": 0.0}
    return row | overrides


# ── The production window's two gates ─────────────────────────────────────────

def test_the_production_window_needs_the_flag(tmp_path):
    """`--window full` on its own raises. A window is a value that gets passed around;
    an unlock should be an act somebody performs, which is why it takes its own flag."""
    cfg = _cfg(tmp_path)
    with pytest.raises(HeldOutLocked) as excinfo:
        posteriors.assert_production(cfg, production=False)
    assert "make posteriors-production" in str(excinfo.value)


def test_the_production_window_needs_the_measurement_to_exist_first(tmp_path):
    """Deploy before measuring and there is no honest measurement left to take."""
    cfg = _cfg(tmp_path)
    with pytest.raises(HeldOutLocked) as excinfo:
        posteriors.assert_production(cfg, production=True)
    assert "make final-evaluation" in str(excinfo.value)

    dest = Path(cfg["evaluation"]["predictions_dir"])
    dest.mkdir(parents=True, exist_ok=True)
    (dest / posteriors.FINAL_EVALUATION_ARTIFACT).write_text("head,arm\navailability,x\n")
    posteriors.assert_production(cfg, production=True)      # the gate fires once, not always


def test_the_full_window_is_fitted_inside_an_unlock():
    """Structural, because the alternative is a multi-hour fit to observe one branch.

    `windowed` calls `assert_unlocked` for the `full` window, so `run` has to open the
    unlock or the production fit cannot build anything at all — and it has to be the scoped
    `held_out.unlocked`, which re-locks on the way out however the loop ends.
    """
    tree = ast.parse(Path("src/models/posteriors.py").read_text())
    run = next(node for node in tree.body
               if isinstance(node, ast.FunctionDef) and node.name == "run")
    called = {node.func.id for node in ast.walk(run)
              if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}
    assert {"assert_production", "unlocked"} <= called


# ── The deployed model has to be the selected model ───────────────────────────

def test_a_wider_window_carrying_a_different_variant_is_refused(tmp_path):
    """The failure this exists for: `train_val` artifacts built before the preseason block
    shipped carried the right window and the wrong model, and nothing objected."""
    cfg = _cfg(tmp_path)
    _manifest(cfg, "train", [_head("ast"), _head("reb")])
    _manifest(cfg, "train_val", [_head("ast"), _head("reb", preseason=False)])
    with pytest.raises(ValueError) as excinfo:
        posteriors.assert_same_specification(cfg, "train_val")
    assert "reb" in str(excinfo.value) and "preseason" in str(excinfo.value)


def test_a_wider_window_missing_a_head_is_refused(tmp_path):
    """A chain fitted at one window and completed at another is not a model anybody
    selected — and a partial `--groups` run is the normal way to produce one."""
    cfg = _cfg(tmp_path)
    _manifest(cfg, "train", [_head("ast"), _head("reb")])
    _manifest(cfg, "train_val", [_head("ast")])
    with pytest.raises(ValueError) as excinfo:
        posteriors.assert_same_specification(cfg, "train_val")
    assert "reb" in str(excinfo.value)


def test_the_same_specification_at_two_windows_passes(tmp_path):
    """Row counts, season spans and R-hat all differ legitimately across windows, so the
    comparison is on the specification columns and not on the artifact."""
    cfg = _cfg(tmp_path)
    _manifest(cfg, "train", [_head("ast", n_fit_rows=6382, max_rhat=1.002)])
    _manifest(cfg, "train_val", [_head("ast", n_fit_rows=7155, max_rhat=1.006)])
    got = posteriors.assert_same_specification(cfg, "train_val")
    assert list(got.index) == ["ast"]


# ── The registry, and the artifact that must not be retracted ─────────────────

def test_the_chain_is_registered_and_reads_the_deployed_window():
    """The chain is the project's own figure rather than a head's, and it must read the
    window that describes what would deploy — never `full`, which has read the answer."""
    assert "chain" in final_evaluation.HEADS
    assert final_evaluation.CHAIN_WINDOW == "train_val"
    assert final_evaluation.CHAIN_WINDOW != "full"


def test_the_chain_does_not_reselect_the_strategy():
    """It reads `strategy_shipped.csv`. Re-running the sweep here would make the held-out
    reading the largest selection event in the project."""
    source = Path("src/final_evaluation.py").read_text()
    start = source.index("def _chain(")
    body = source[start:source.index("\ndef _report_chain(")]
    assert "strategy_shipped.csv" in body
    assert "strategy_table()" in body          # looked up by name, not swept
    assert "strat.sweep(" not in body and "gate_c(" not in body


def test_taking_one_head_does_not_retract_the_others(tmp_path, monkeypatch):
    """`final_evaluation.csv` merges by head. The chain costs hours and the three heads do
    not, so running them separately is the normal workflow — and this is the one artifact
    in the project that cannot be re-derived on demand, because taking it again is taking
    it a second time.
    """
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(final_evaluation, "HEADS", {
        "availability": lambda cfg, out: pd.DataFrame([{"head": "availability",
                                                        "arm": "mixture", "crps": 1.0}]),
        "chain": lambda cfg, out: pd.DataFrame([{"head": "chain", "arm": "600k",
                                                 "roi": 2.0}]),
    })
    final_evaluation.run(cfg, ["availability"])
    final_evaluation.run(cfg, ["chain"])

    out = pd.read_csv(Path(cfg["evaluation"]["predictions_dir"]) / "final_evaluation.csv")
    assert set(out["head"]) == {"availability", "chain"}
    assert float(out.loc[out["head"] == "availability", "crps"].iloc[0]) == 1.0

    # Re-taking a head replaces its own rows rather than duplicating them.
    final_evaluation.run(cfg, ["chain"])
    out = pd.read_csv(Path(cfg["evaluation"]["predictions_dir"]) / "final_evaluation.csv")
    assert len(out) == 2


# ── The held-out simulation must not touch the audited artifacts ──────────────

def test_the_held_out_simulation_writes_its_own_gate_table():
    """`sim_season_gate_a.csv` is a single pooled table whose extremes `make docs-audit`
    re-derives, so simulating a test season into it would move an audited figure by adding
    rows rather than by changing a result.
    """
    from src.sim import season as sim_season

    source = Path("src/sim/season.py").read_text()
    assert 'f"sim_season_gate_a{label}.csv"' in source
    assert "label" in sim_season.run.__code__.co_varnames
    assert final_evaluation.CHAIN_LABEL
