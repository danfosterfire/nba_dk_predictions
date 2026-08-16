"""Tests for the preseason block's paired contest counterfactual.

Nothing here needs a sampler or a tensor: the module is a comparison over two captured arms,
so the synthetic builders below write the same CSVs the chain does.

The first test is the one this round adds over `test_mixture_value.py`, and it pins a
*structural* fact rather than a number. The block sits on three heads and only two of them
are in the simulator's draw path — `src/sim/` imports neither `StanMinutes` nor
`rehydrate_minutes` and never looks up `artifacts["minutes"]`. `README.md` carried the
opposite claim until 2026-08-14, so the module's docstring is not a safe place to keep it.

The rest are `mixture_value`'s four, one round over, and each is a way the artifact could be
wrong while every number in it still looked plausible:

- **`--capture` refuses unless all THREE keys agree with the arm name.** A pass with the
  block half on is neither arm and would leave no trace in the output;
- **the contest block compares the SAME strategy across arms**, since the two arms need not
  select the same one and the shipped chain already selects a different one at
  `88k_alley_oop`;
- **the resolution row is derived from the sweep's own bootstrap**, not asserted, so it
  tracks `n_sims`;
- **a missing table fails the capture** rather than producing a half-arm.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.sim import preseason_contest as PC


# ── The structural claim the whole comparison is scoped by ────────────────────

def test_the_simulator_never_loads_the_marginal_minutes_head():
    """P3's block cannot reach the contest, and that scopes every number in the artifact.

    Parsed rather than grepped, so a name inside a string or a comment does not count as a
    use. If this ever fails, the module docstring's scope note is wrong and the counterfactual
    is measuring three heads rather than two — which changes what a null here would mean.
    """
    banned = {"StanMinutes", "rehydrate_minutes"}
    for path in sorted(Path("src/sim").glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names = {alias.name for alias in node.names}
                assert not (names & banned), (
                    f"{path} imports {names & banned}; the simulator's minutes come from "
                    f"the composition plus the injected sigma, and "
                    f"src/sim/preseason_contest.py's scope note depends on that")
            # `artifacts["minutes"]` — the other way the fitted head could arrive.
            if (isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                    and node.slice.value == "minutes"
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "artifacts"):
                raise AssertionError(f"{path} loads artifacts['minutes']")


# ── Synthetic builders ────────────────────────────────────────────────────────

def _cfg(tmp_path, availability=True, minutes=True, composition=True,
         components=True) -> dict:
    return {"evaluation": {"predictions_dir": str(tmp_path / "predictions")},
            "data": {"features_dir": str(tmp_path / "features")},
            "sim": {"minutes": {"player_season_sigma": 0.375}},
            "stan": {"availability": {"preseason": availability},
                     "minutes": {"preseason": minutes},
                     "composition": {"preseason": {"adopt": composition}},
                     "components": {"preseason": components}}}


def _sweep(lift: float) -> pd.DataFrame:
    """A sweep carries EVERY strategy — which arm won is `strategy_shipped`'s business."""
    rows = []
    for season in PC.SEASONS:
        for name, value in ((PC.REFERENCE_STRATEGY, lift), ("blend_a70", lift + 0.01),
                            ("adp", lift - 0.12), ("model_mean", lift - 0.05)):
            rows.append({"season": season, "tournament": "600k_shootaround",
                         "strategy": name, "p_advance": 1 / 6 + value,
                         "p_advance_lo": 1 / 6 + value - 0.08,
                         "p_advance_hi": 1 / 6 + value + 0.08,
                         "lift_vs_null": value, "n_sims": 500, "n_worlds": 500})
    return pd.DataFrame(rows)


def _tensor_summary(mean_shift: float = 0.0) -> pd.DataFrame:
    rows = []
    for season in PC.SEASONS:
        for player in range(8):
            rows.append({"season": season, "player_id": 100 + player, "n_sims": 500,
                         "prior_mpg": 5.0 + 4.0 * player,
                         "mean_total": 1000.0 - 50.0 * player + mean_shift,
                         "sd_total": 400.0, "q10_total": 500.0, "q90_total": 1500.0,
                         "mean_gp": 55.0, "p_disrupted": 0.2, "p_iron_man": 0.10,
                         "p_iron_man_strict": 0.03})
    return pd.DataFrame(rows)


def _write_arm(cfg: dict, arm: str, lift: float, strategy: str = PC.REFERENCE_STRATEGY,
               mean_shift: float = 0.0, comp_season: str = "2004-05",
               avail_features: int | None = None) -> None:
    """A complete capture directory, written the way `--capture` would leave one."""
    dest = Path(cfg["evaluation"]["predictions_dir"]) / PC.CAPTURE_DIR / arm
    dest.mkdir(parents=True, exist_ok=True)
    on = (arm == "preseason")

    _sweep(lift).to_csv(dest / "strategy_sweep.csv", index=False)
    _sweep(lift - 0.02).to_csv(dest / "strategy_realized.csv", index=False)
    pd.DataFrame([{"tournament": "600k_shootaround", "strategy": strategy,
                   "sim_lift": lift, "realized_lift": lift - 0.02}]
                 ).to_csv(dest / "strategy_shipped.csv", index=False)
    pd.DataFrame([{"tournament": "600k_shootaround", "materially_different": 0}]
                 ).to_csv(dest / "strategy_gate_d.csv", index=False)
    pd.DataFrame([{"season": s, "rho": 0.42} for s in PC.SEASONS]
                 ).to_csv(dest / "strategy_injection.csv", index=False)
    gate_a = []
    for season in PC.SEASONS:
        gate_a.append({"season": season, "check": "games_played",
                       "pmf_total_variation": 0.06, "simulated_mean_gp": 55.0,
                       "mae": 13.0, "crps": 9.5, "bias": -0.1, "r2": 0.5})
        gate_a.append({"season": season, "check": "season_total_dk",
                       "pmf_total_variation": np.nan, "simulated_mean_gp": np.nan,
                       "mae": 400.0, "crps": 280.0, "bias": -20.0, "r2": 0.70})
    pd.DataFrame(gate_a).to_csv(dest / "sim_season_gate_a.csv", index=False)
    _tensor_summary(mean_shift).to_csv(dest / "tensor_summary.csv", index=False)
    pd.DataFrame([{"arm": arm,
                   "stan.availability.preseason": on,
                   "stan.minutes.preseason": on,
                   "stan.composition.preseason.adopt": on,
                   "stan.components.preseason": on,
                   "captured_at": "2026-08-15T09:00:00",
                   "availability_first_season": "2012-13",
                   "minutes_first_season": "2004-05" if on else "1997-98",
                   "composition_first_season": comp_season,
                   # Availability's window is 2012-13 in BOTH arms — its block is columns
                   # only — so the feature count is the only trace it leaves.
                   "availability_n_features": (avail_features if avail_features is not None
                                               else (29 if on else 19)),
                   "minutes_n_features": 29 if on else 24,
                   # The composition adds no columns: its block blends `w_share`.
                   "composition_n_features": 25,
                   "player_season_sigma": 0.375, "n_sims": 500, "n_players": 8}]
                 ).to_csv(dest / "provenance.csv", index=False)


# ── The capture guard ─────────────────────────────────────────────────────────

def test_capture_refuses_when_the_config_and_the_arm_name_disagree(tmp_path):
    cfg = _cfg(tmp_path)
    with pytest.raises(ValueError, match="preseason keys"):
        PC.capture(cfg, "base")


def test_capture_refuses_a_HALF_ON_block(tmp_path):
    """Two keys off and one on is neither arm — the failure mode a single-key guard misses."""
    cfg = _cfg(tmp_path, availability=False, minutes=False, composition=True)
    with pytest.raises(ValueError, match="stan.composition.preseason.adopt"):
        PC.capture(cfg, "base")
    with pytest.raises(ValueError, match="stan.availability.preseason"):
        PC.capture(cfg, "preseason")


def test_capture_refuses_a_partial_chain(tmp_path):
    """A mid-chain capture freezes half an arm, and every later number would be a blend."""
    cfg = _cfg(tmp_path)
    out = Path(cfg["evaluation"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in PC.ARM_TABLES[:3]:
        pd.DataFrame([{"a": 1}]).to_csv(out / f"{name}.csv", index=False)
    with pytest.raises(FileNotFoundError, match="COMPLETE chain"):
        PC.capture(cfg, "preseason")


def test_a_missing_arm_names_the_procedure_rather_than_the_file(tmp_path):
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "preseason", 0.24)
    with pytest.raises(FileNotFoundError, match="same code"):
        PC.build(cfg)


# ── The comparison ────────────────────────────────────────────────────────────

def test_contest_block_compares_one_strategy_across_both_arms(tmp_path):
    """The arms selected different strategies; the gap must still be like for like."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19, strategy="blend_a70")
    _write_arm(cfg, "preseason", 0.24)
    frame = PC.build(cfg)

    lift = frame[(frame["block"] == "contest") & (frame["measure"] == "sim_lift")]
    assert len(lift) == 1
    assert lift["base"].iloc[0] == pytest.approx(0.19)
    assert lift["preseason"].iloc[0] == pytest.approx(0.24)
    assert lift["delta"].iloc[0] == pytest.approx(0.05)

    picked = frame[frame["measure"] == "selected_reference_strategy"]
    assert picked["base"].iloc[0] == 0.0
    assert picked["preseason"].iloc[0] == 1.0
    assert "blend_a70" in picked["note"].iloc[0]


def test_resolution_is_derived_from_the_bootstrap_not_asserted(tmp_path):
    """Halving the interval has to halve the bar, or the null claim is unfalsifiable."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19)
    _write_arm(cfg, "preseason", 0.24)
    wide = PC.build(cfg)
    bar_wide = wide[wide["measure"] == "min_detectable_lift_gap"]["preseason"].max()

    for arm in PC.ARMS:
        dest = Path(cfg["evaluation"]["predictions_dir"]) / PC.CAPTURE_DIR / arm
        sweep = pd.read_csv(dest / "strategy_sweep.csv")
        sweep["p_advance_lo"] = sweep["p_advance"] - 0.04
        sweep["p_advance_hi"] = sweep["p_advance"] + 0.04
        sweep.to_csv(dest / "strategy_sweep.csv", index=False)
    narrow = PC.build(cfg)
    bar_narrow = narrow[narrow["measure"] == "min_detectable_lift_gap"]["preseason"].max()
    assert bar_narrow == pytest.approx(bar_wide / 2.0)


def test_the_board_block_reports_order_and_the_draw_block_reports_the_level(tmp_path):
    """A pure level shift must leave the ORDER at 1.0 while the draw rows move.

    This is the distinction the preseason round turns on: the composition's gate scored its
    gain at the allocation mean, and a mean that moves every player equally cannot reach a
    ranking. Collapsing the two blocks would report a level shift as a drafting gain.
    """
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19, mean_shift=0.0)
    _write_arm(cfg, "preseason", 0.24, mean_shift=40.0)
    frame = PC.build(cfg)

    order = frame[(frame["block"] == "board")
                  & (frame["measure"] == "spearman_mean_total")]
    assert len(order) == len(PC.SEASONS)
    assert order["preseason"].to_numpy() == pytest.approx(1.0)
    moved = frame[(frame["block"] == "board")
                  & (frame["measure"] == "picks_moving_12plus")]
    assert moved["preseason"].to_numpy() == pytest.approx(0.0)

    level = frame[(frame["block"] == "draw") & (frame["measure"] == "mean_total")]
    assert not level.empty
    assert level["delta"].to_numpy() == pytest.approx(40.0)


def test_the_realized_side_is_priced_by_pairing_not_by_the_simulated_bar(tmp_path):
    """The realized rows must not borrow a bar built from 500 simulated worlds.

    `resolution.min_detectable_lift_gap` is a bootstrap over the simulated side. The
    realized readout has one world per season and two seasons, so that bar never measured
    its uncertainty — stamping RESOLVED with it would overclaim exactly where the evidence
    is thinnest. What the realized side has instead is pairing, so the block reports
    consistency: how many cells agree in sign, and how far apart the seasons put each
    delta.
    """
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19)
    _write_arm(cfg, "preseason", 0.24)
    frame = PC.build(cfg)

    cells = frame[(frame["block"] == "realized")
                  & (frame["measure"] == "delta_positive_cells")]
    assert len(cells) == 1
    # The builder gives every season the same +0.05 shift, so every cell agrees in sign
    # and the denominator rides in the `base` column rather than only in the note.
    assert cells["preseason"].iloc[0] == cells["base"].iloc[0]
    assert "ONE test" in cells["note"].iloc[0]

    spread = frame[(frame["block"] == "realized") & (frame["measure"] == "season_spread")]
    assert not spread.empty
    assert spread["preseason"].to_numpy() == pytest.approx(0.0)

    # And the realized contest row says the bar does not apply to it.
    realized = frame[(frame["block"] == "contest")
                     & (frame["measure"] == "realized_lift")]
    assert "DOES NOT APPLY" in realized["note"].iloc[0]


def test_the_adp_control_is_reported_because_its_board_cannot_move(tmp_path):
    """`adp` never reads the model, so its delta is the world effect with no board in it."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19)
    _write_arm(cfg, "preseason", 0.24)
    frame = PC.build(cfg)
    control = frame[frame["measure"] == "adp_only_lift"]
    assert len(control) == 1
    assert control["delta"].iloc[0] == pytest.approx(0.05)
    assert "CONTROL" in control["note"].iloc[0]
    assert frame[frame["measure"] == "ordering_spearman"]["preseason"].iloc[0] \
        == pytest.approx(1.0)


def _refit(frame: pd.DataFrame, head: str) -> float:
    hit = frame[(frame["block"] == "reach") & (frame["measure"] == "refit_landed")
                & (frame["key"] == head)]
    return float(hit["preseason"].iloc[0])


def test_the_reach_block_shows_which_heads_actually_refitted(tmp_path):
    """A carried-over posterior must surface here, not as a plausible null downstream."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "base", 0.19, comp_season="1996-97")
    _write_arm(cfg, "preseason", 0.24, comp_season="2004-05")
    moved = PC.build(cfg)
    assert _refit(moved, "composition") == 1.0
    assert _refit(moved, "availability") == 1.0

    # And the failure it exists to catch: an identical persisted head on both arms.
    _write_arm(cfg, "base", 0.19, comp_season="2004-05", avail_features=29)
    stalled = PC.build(cfg)
    assert _refit(stalled, "composition") == 0.0
    assert _refit(stalled, "availability") == 0.0

    keys = stalled[(stalled["block"] == "reach") & (stalled["measure"] == "config_key")]
    assert list(keys["key"]) == list(PC.KEY_NAMES)
    assert keys["base"].to_numpy() == pytest.approx(0.0)
    assert keys["preseason"].to_numpy() == pytest.approx(1.0)


def test_reach_needs_BOTH_traces_because_each_head_hides_in_the_other(tmp_path):
    """The trap this block was rewritten to avoid, pinned as two one-sided arms.

    Availability's block is ten columns at a window that does not move; the composition's
    is a blend into `w_share` that adds no columns. Checking `first_season` alone reports
    availability as never having refitted, and checking `n_features` alone does the same to
    the composition — in both cases while every number downstream still looks plausible.
    """
    cfg = _cfg(tmp_path)
    # Both arms at the same composition window: only the FEATURE count can show the refit.
    _write_arm(cfg, "base", 0.19, comp_season="2004-05")
    _write_arm(cfg, "preseason", 0.24, comp_season="2004-05")
    columns_only = PC.build(cfg)
    assert _refit(columns_only, "availability") == 1.0, "the window never moves on this head"
    assert _refit(columns_only, "composition") == 0.0

    # And the mirror image: identical feature counts, only the WINDOW moves.
    _write_arm(cfg, "base", 0.19, comp_season="1996-97", avail_features=29)
    window_only = PC.build(cfg)
    assert _refit(window_only, "availability") == 0.0
    assert _refit(window_only, "composition") == 1.0, "this head adds no columns"
