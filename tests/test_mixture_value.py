"""Tests for the availability mixture's paired contest counterfactual.

Nothing here needs a sampler or a tensor: the module is a comparison over two captured
arms, so the synthetic builders below write the same CSVs the chain does.

Four things are pinned, and each is a way the artifact could be wrong while every number
in it still looked plausible:

- **`--capture` refuses when the config key and the arm name disagree.** Capturing arm A's
  artifacts under arm B's name reproduces the exact confound this module exists to remove
  (`docs/availability-window-plan.md` §7k) and would leave no trace in the output;
- **the contest block compares the SAME strategy across arms.** The two arms need not
  select the same one, and reading `strategy_shipped.csv` per arm would report the gap
  between two different strategies as the mixture's value;
- **the resolution row is derived from the sweep's own bootstrap**, not asserted, so it
  tracks `n_sims` — a row that stayed at 0.09 after the world count changed would license
  a claim the run cannot support;
- **a missing table fails the capture** rather than producing a half-arm, since a
  mid-chain capture is the normal way a partial run gets frozen by accident.
"""

import numpy as np
import pandas as pd
import pytest

from src.sim import mixture_value as MV


# ── Synthetic builders ────────────────────────────────────────────────────────

def _cfg(tmp_path) -> dict:
    return {"evaluation": {"predictions_dir": str(tmp_path / "predictions")},
            "data": {"features_dir": str(tmp_path / "features")},
            "stan": {"availability": {"mixture": True, "first_season": "2012-13",
                                      "role_rho": True}}}


def _sweep(lift: float) -> pd.DataFrame:
    """A sweep carries EVERY strategy — which arm won is `strategy_shipped`'s business."""
    rows = []
    for season in MV.SEASONS:
        for name, value in ((MV.REFERENCE_STRATEGY, lift), ("blend_a70", lift + 0.01),
                            ("adp", lift - 0.12), ("model_mean", lift - 0.05)):
            rows.append({"season": season, "tournament": "600k_shootaround",
                         "strategy": name, "p_advance": 1 / 6 + value,
                         "p_advance_lo": 1 / 6 + value - 0.08,
                         "p_advance_hi": 1 / 6 + value + 0.08,
                         "lift_vs_null": value, "n_sims": 500, "n_worlds": 500})
    return pd.DataFrame(rows)


def _tensor_summary(mean_shift: float = 0.0, iron: float = 0.10) -> pd.DataFrame:
    rows = []
    for season in MV.SEASONS:
        for player in range(8):
            rows.append({"season": season, "player_id": 100 + player, "n_sims": 500,
                         "prior_mpg": 5.0 + 4.0 * player,
                         "mean_total": 1000.0 - 50.0 * player + mean_shift,
                         "sd_total": 400.0, "q10_total": 500.0, "q90_total": 1500.0,
                         "mean_gp": 55.0, "p_disrupted": 0.2, "p_iron_man": iron,
                         "p_iron_man_strict": iron / 3.0})
    return pd.DataFrame(rows)


def _write_arm(cfg: dict, arm: str, lift: float, strategy: str = MV.REFERENCE_STRATEGY,
               iron: float = 0.10) -> None:
    """A complete capture directory, written the way `--capture` would leave one."""
    from pathlib import Path
    dest = Path(cfg["evaluation"]["predictions_dir"]) / MV.CAPTURE_DIR / arm
    dest.mkdir(parents=True, exist_ok=True)

    _sweep(lift).to_csv(dest / "strategy_sweep.csv", index=False)
    _sweep(lift - 0.02).to_csv(dest / "strategy_realized.csv", index=False)
    pd.DataFrame([{"tournament": "600k_shootaround", "strategy": strategy,
                   "sim_lift": lift, "realized_lift": lift - 0.02,
                   "sim_p_advance": 1 / 6 + lift, "sim_roi": 5.0}]
                 ).to_csv(dest / "strategy_shipped.csv", index=False)
    pd.DataFrame([{"tournament": "600k_shootaround", "materially_different": 0}]
                 ).to_csv(dest / "strategy_gate_d.csv", index=False)
    pd.DataFrame([{"season": s, "rho": 0.42} for s in MV.SEASONS]
                 ).to_csv(dest / "strategy_injection.csv", index=False)
    gate_a = []
    for season in MV.SEASONS:
        gate_a.append({"season": season, "check": "games_played",
                       "pmf_total_variation": 0.06, "simulated_mean_gp": 55.0,
                       "mae": 13.0, "crps": 9.5, "bias": -0.1})
        gate_a.append({"season": season, "check": "season_total_dk",
                       "pmf_total_variation": np.nan, "simulated_mean_gp": np.nan,
                       "mae": 400.0, "crps": 280.0, "bias": -20.0})
    pd.DataFrame(gate_a).to_csv(dest / "sim_season_gate_a.csv", index=False)
    _tensor_summary(iron=iron).to_csv(dest / "tensor_summary.csv", index=False)
    pd.DataFrame([{"arm": arm, "mixture": arm == "mixture",
                   "captured_at": "2026-08-12T13:00:00", "first_season": "2012-13",
                   "role_rho": True, "n_sims": 500, "n_players": 8}]
                 ).to_csv(dest / "provenance.csv", index=False)


# ── The capture guard ─────────────────────────────────────────────────────────

def test_capture_refuses_when_the_config_and_the_arm_name_disagree(tmp_path):
    """The one check that stops §7k's confound being reproduced silently."""
    cfg = _cfg(tmp_path)
    assert cfg["stan"]["availability"]["mixture"] is True
    with pytest.raises(ValueError, match="mixture"):
        MV.capture(cfg, "single")


def test_capture_refuses_a_partial_chain(tmp_path):
    """A mid-chain capture freezes half an arm, and every later number would be a blend."""
    from pathlib import Path
    cfg = _cfg(tmp_path)
    out = Path(cfg["evaluation"]["predictions_dir"])
    out.mkdir(parents=True, exist_ok=True)
    for name in MV.ARM_TABLES[:3]:
        pd.DataFrame([{"a": 1}]).to_csv(out / f"{name}.csv", index=False)
    with pytest.raises(FileNotFoundError, match="COMPLETE chain"):
        MV.capture(cfg, "mixture")


def test_a_missing_arm_names_the_procedure_rather_than_the_file(tmp_path):
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "mixture", 0.19)
    with pytest.raises(FileNotFoundError, match="same code"):
        MV.build(cfg)


# ── The comparison ────────────────────────────────────────────────────────────

def test_contest_block_compares_one_strategy_across_both_arms(tmp_path):
    """The arms selected different strategies; the gap must still be like for like.

    `strategy_shipped.csv` holds each arm's OWN selection, so reading it would compare
    `blend_a70` against `lineup_value_blend30` and call the difference the mixture's value.
    The reference strategy is present in both sweeps at a known lift, so the delta is
    pinned exactly.
    """
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "single", 0.16, strategy="blend_a70")
    _write_arm(cfg, "mixture", 0.19)
    frame = MV.build(cfg)

    lift = frame[(frame["block"] == "contest") & (frame["measure"] == "sim_lift")]
    assert len(lift) == 1
    assert lift["single"].iloc[0] == pytest.approx(0.16)
    assert lift["mixture"].iloc[0] == pytest.approx(0.19)
    assert lift["delta"].iloc[0] == pytest.approx(0.03)

    # And the divergent selection is reported rather than absorbed.
    picked = frame[frame["measure"] == "selected_reference_strategy"]
    assert picked["single"].iloc[0] == 0.0
    assert picked["mixture"].iloc[0] == 1.0
    assert "blend_a70" in picked["note"].iloc[0]


def test_resolution_is_derived_from_the_bootstrap_not_asserted(tmp_path):
    """Halving the interval has to halve the bar, or the null claim is unfalsifiable."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "single", 0.16)
    _write_arm(cfg, "mixture", 0.19)
    wide = MV.build(cfg)
    bar_wide = wide[wide["measure"] == "min_detectable_lift_gap"]["mixture"].max()

    from pathlib import Path
    for arm in MV.ARMS:
        dest = Path(cfg["evaluation"]["predictions_dir"]) / MV.CAPTURE_DIR / arm
        sweep = pd.read_csv(dest / "strategy_sweep.csv")
        mid = sweep["p_advance"]
        sweep["p_advance_lo"] = mid - 0.04
        sweep["p_advance_hi"] = mid + 0.04
        sweep.to_csv(dest / "strategy_sweep.csv", index=False)
    narrow = MV.build(cfg)
    bar_narrow = narrow[narrow["measure"] == "min_detectable_lift_gap"]["mixture"].max()
    assert bar_narrow == pytest.approx(bar_wide / 2.0)


def test_board_rows_report_a_between_arm_rank_correlation(tmp_path):
    """The board block asks whether the ORDER moved, which is what a ranking consumes."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "single", 0.16)
    _write_arm(cfg, "mixture", 0.19)
    frame = MV.build(cfg)
    board = frame[(frame["block"] == "board")
                  & (frame["measure"] == "spearman_mean_total")]
    assert len(board) == len(MV.SEASONS)
    # Identical builders, so the ordering is identical and the correlation is exactly 1.
    assert board["mixture"].to_numpy() == pytest.approx(1.0)


def test_the_adp_control_is_reported_because_its_board_cannot_move(tmp_path):
    """`adp` never reads the model, so its delta is the world effect with no board in it.

    Without that row the strategy block reports a level shift with no way to tell a
    drafting gain from a world that spreads rosters further apart — which is the whole
    interpretive question the contest readout turns on.
    """
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "single", 0.16)
    _write_arm(cfg, "mixture", 0.19)
    frame = MV.build(cfg)
    control = frame[frame["measure"] == "adp_only_lift"]
    assert len(control) == 1
    # The builder gives every strategy the same lift shift, so the control moves with it.
    assert control["delta"].iloc[0] == pytest.approx(0.03)
    assert "CONTROL" in control["note"].iloc[0]

    ordering = frame[frame["measure"] == "ordering_spearman"]
    assert ordering["mixture"].iloc[0] == pytest.approx(1.0)


def test_draw_rows_are_graded_by_the_head_s_own_role_buckets(tmp_path):
    """The per-bucket rows line up with the dispersion grading the head actually carries."""
    cfg = _cfg(tmp_path)
    _write_arm(cfg, "single", 0.16, iron=0.10)
    _write_arm(cfg, "mixture", 0.19, iron=0.08)
    frame = MV.build(cfg)
    iron = frame[(frame["block"] == "draw") & (frame["measure"] == "p_iron_man")]
    assert not iron.empty
    assert set(iron["key"]) <= {f"{s} {b}" for s in MV.SEASONS for b in MV.ROLE_LABELS}
    assert iron["delta"].to_numpy() == pytest.approx(-0.02)
