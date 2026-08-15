"""Tests for session 4b — the preseason inside the composition's prior share.

The round's claim is unusual for this project: the preseason enters as an **input** rather
than as a feature, so the things worth pinning are the ways an input change can be wrong in
a way a metric column cannot show.

1. **The incumbent nests exactly.** `k → ∞` must reproduce today's `w_share` bit for bit, or
   every margin is measured against something that never shipped.
2. **The two routes are separable.** `w_share` reaches the head as the offset AND as the
   allocation order; the attribution is the round's main instrument and it is worthless if
   the flags leak into each other.
3. **The pairing is by identity.** Every arm re-runs `order_frame`, so two arms hold the same
   player-games in different row orders. A positional bootstrap over them would return a
   plausible interval around a meaningless difference — this one was a live bug.
4. **`order_frame` is unchanged for every other caller.** A frame with no `order_share`
   column must sort exactly as it did before the hook existed, since eleven other modules
   build their rows through it.
"""

import numpy as np
import pandas as pd
import pytest

import src.models.composition_preseason as CP
import src.models.stan_composition as SC
from src.models.composition_preseason import (INCUMBENT_K, ROUTES, blend_hook, paired,
                                              preseason_share, season_totals)
from src.models.stan_composition import order_frame


def _lagged(rows: list[tuple[str, int, float]]) -> pd.DataFrame:
    """(season, player_id, w_share) → the per-(player, season) frame the hook receives."""
    return pd.DataFrame([{"season": s, "player_id": p, "w_share": w}
                         for s, p, w in rows])


def _panel(rows: list[tuple[str, int, float, float]]) -> pd.DataFrame:
    """(season, player_id, mpg_pre, min_pre) → the panel columns `preseason_share` reads."""
    return preseason_share(pd.DataFrame(
        [{"season": s, "player_id": p, "mpg_pre": mpg, "min_pre": m}
         for s, p, mpg, m in rows]))


def test_a_large_k_reproduces_the_incumbent_share_exactly():
    """The nesting discipline `n_rho = 1`, `U_n = 0` and `theta = 0` already carry. An
    incumbent that is only approximately reproduced makes every margin unreadable."""
    lagged = _lagged([("2020-21", 1, 0.62), ("2020-21", 2, 0.11)])
    panel = _panel([("2020-21", 1, 24.0, 96.0), ("2020-21", 2, 6.0, 30.0)])
    out = blend_hook(panel, 1e12)(lagged)
    assert np.allclose(out["w_share"], lagged["w_share"], rtol=0, atol=1e-6)
    assert np.allclose(out["order_share"], lagged["w_share"], rtol=0, atol=1e-6)


def test_k_zero_is_the_raw_preseason_share_for_anyone_who_has_one():
    lagged = _lagged([("2020-21", 1, 0.62), ("2020-21", 2, 0.11)])
    panel = _panel([("2020-21", 1, 24.0, 96.0), ("2020-21", 2, 6.0, 30.0)])
    out = blend_hook(panel, 0.0)(lagged)
    assert np.allclose(out["w_share"], [24.0 / 48.0, 6.0 / 48.0])


def test_a_player_with_no_preseason_row_keeps_his_incumbent_share_by_arithmetic():
    """Not by a special case. `min_pre` is 0 for him so `omega = 0`, which is what makes the
    blend safe to apply to the whole league rather than to a filtered subset."""
    lagged = _lagged([("2020-21", 1, 0.62), ("2020-21", 2, 0.11)])
    panel = _panel([("2020-21", 1, 24.0, 96.0)])          # player 2 has no row
    for k in (0.0, 20.0, 80.0):
        out = blend_hook(panel, k)(lagged).set_index("player_id")
        assert np.isclose(out.loc[2, "w_share"], 0.11)
        assert np.isclose(out.loc[2, "order_share"], 0.11)


def test_the_blend_weight_is_the_declared_volume_rule():
    lagged = _lagged([("2020-21", 1, 0.20)])
    panel = _panel([("2020-21", 1, 24.0, 80.0)])
    k = 80.0
    out = blend_hook(panel, k)(lagged)
    w = 80.0 / (80.0 + k)
    assert np.isclose(out["w_share"].iloc[0], w * 0.5 + (1 - w) * 0.20)


def test_the_two_routes_do_not_leak_into_each_other():
    """The attribution is the round's main instrument: `offset_only` must leave the ORDER
    column at the incumbent and `order_only` must leave the OFFSET column there."""
    lagged = _lagged([("2020-21", 1, 0.62), ("2020-21", 2, 0.11)])
    panel = _panel([("2020-21", 1, 24.0, 96.0), ("2020-21", 2, 6.0, 30.0)])
    both = blend_hook(panel, 80.0, "both")(lagged)
    offset = blend_hook(panel, 80.0, "offset_only")(lagged)
    order = blend_hook(panel, 80.0, "order_only")(lagged)

    assert np.allclose(offset["w_share"], both["w_share"])
    assert np.allclose(offset["order_share"], lagged["w_share"])
    assert np.allclose(order["order_share"], both["order_share"])
    assert np.allclose(order["w_share"], lagged["w_share"])
    # And the blend is a real change, so the equalities above are not all true by collapse.
    assert not np.allclose(both["w_share"], lagged["w_share"])


def test_an_unknown_route_raises():
    panel = _panel([("2020-21", 1, 24.0, 96.0)])
    with pytest.raises(ValueError, match="route"):
        blend_hook(panel, 80.0, "offset")
    assert set(ROUTES) == {"both", "offset_only", "order_only"}


def test_order_frame_is_unchanged_for_callers_that_set_no_order_share():
    """Eleven modules build their rows through `order_frame`. A frame with no `order_share`
    must sort exactly as it did before the hook existed."""
    frame = pd.DataFrame({
        "season": ["2020-21"] * 4, "game_id": ["0001"] * 4, "team_id": [1] * 4,
        "player_id": [10, 11, 12, 13], "w_share": [0.1, 0.5, 0.3, 0.2],
        "no_prior": [0, 0, 0, 0], "draft_number": [5.0, 2.0, 9.0, np.nan]})
    assert order_frame(frame)["player_id"].tolist() == [11, 12, 13, 10]


def test_order_frame_sorts_on_order_share_when_the_frame_carries_one():
    """And the two columns genuinely differ, so the test cannot pass by them agreeing."""
    frame = pd.DataFrame({
        "season": ["2020-21"] * 4, "game_id": ["0001"] * 4, "team_id": [1] * 4,
        "player_id": [10, 11, 12, 13], "w_share": [0.1, 0.5, 0.3, 0.2],
        "order_share": [0.9, 0.1, 0.3, 0.2],
        "no_prior": [0, 0, 0, 0], "draft_number": [5.0, 2.0, 9.0, np.nan]})
    assert order_frame(frame)["player_id"].tolist() == [10, 12, 13, 11]


def test_the_paired_interval_refuses_two_arms_over_different_rows():
    """The bug this guard exists for: each arm re-sorts, so two arms hold the same
    player-games in different orders. Subtracting them positionally pairs each row against a
    different one and returns a plausible interval around nothing."""
    index = pd.MultiIndex.from_tuples([(1, "a"), (2, "a")], names=["player_id", "season"])
    other = pd.MultiIndex.from_tuples([(1, "a"), (3, "a")], names=["player_id", "season"])
    arm = pd.Series([1.0, 2.0], index=index)
    with pytest.raises(AssertionError, match="same rows"):
        paired(arm, pd.Series([1.0, 2.0], index=other))
    # Reordered but identical rows must be fine, because the round sorts every arm's index.
    reordered = arm.iloc[::-1]
    assert paired(arm, reordered.sort_index())[0] == pytest.approx(0.0)


def test_season_totals_sums_the_right_rows_per_unit():
    """`season` is a string column, so the two-key group cannot go through
    `np.unique(..., axis=0)` — it raises on dtype object, which is how this was written
    first. The replacement has to sum each player-season's own games and no others."""
    frame = pd.DataFrame({"player_id": [1, 1, 2], "season": ["2020-21"] * 3})
    samples = np.array([[1.0, 2.0, 10.0], [3.0, 4.0, 20.0]])
    totals, units = season_totals(samples, frame)
    assert units["player_id"].tolist() == [1, 2]
    assert np.allclose(totals, [[3.0, 10.0], [7.0, 20.0]])


def test_the_incumbent_constant_is_beyond_the_grids_reach():
    """`INCUMBENT_K` has to be large enough that the blend weight is numerically zero at
    realistic preseason volumes, or the 'exact nesting' claim is only approximate."""
    assert 200.0 / (200.0 + INCUMBENT_K) < 1e-6


# ── The adoption — `stan_composition.head_frame`, P5 (2026-08-14) ──────────────
#
# Session 4d's arm ships, so the blend stops being a measurement and becomes the head. What
# is worth pinning changes with it: not whether the hook arithmetic is right (above), but
# whether the blend reaches exactly the consumers that must see it and none of the ones that
# must not. The control-arm trap is the specific failure — `composition_preseason`'s `base`
# arms build through `composition_frame` with no hook, so a blend that leaked into that
# builder would silently delete the contrast three sessions were measured on.


def _cfg(adopt: bool = True, k: float = 80.0, route: str = "offset_only",
         first_season: str = "1996-97", eda_dir: str = "outputs/eda") -> dict:
    return {"data": {"features_dir": "data/features"},
            "eda": {"output_dir": eda_dir},
            "stan": {"composition": {"first_season": first_season,
                                     "preseason": {"adopt": adopt, "blend_k": k,
                                                   "route": route}}}}


def test_adopt_false_is_an_exact_rollback_of_both_the_blend_and_the_window():
    """The house nesting discipline, and here it has two halves rather than one: turning the
    block off must restore the incumbent `w_share` AND the 1996-97 window, because adopting
    it cut the window as a side effect."""
    assert SC.preseason_blend(_cfg(adopt=False)) is None
    assert SC.head_first_season(_cfg(adopt=False)) == "1996-97"


def test_a_nesting_k_is_treated_as_no_blend_at_all():
    """`k -> inf` is the incumbent exactly, so a config that nests must not pay for a hook
    or cut the window — otherwise 'the endpoint is the incumbent' is true of the share and
    false of the fitting rows."""
    assert SC.preseason_blend(_cfg(k=INCUMBENT_K)) is None
    assert SC.head_first_season(_cfg(k=INCUMBENT_K)) == "1996-97"


def test_the_blend_carries_the_declared_k_and_route():
    assert SC.preseason_blend(_cfg()) == (80.0, "offset_only")
    assert SC.preseason_blend(_cfg(k=160.0, route="both")) == (160.0, "both")


def test_the_window_is_read_off_the_coverage_artifact_and_never_hard_coded(tmp_path):
    """P3's rule one head over. 2003-04 holds 369 real preseason rows and is still excluded
    as `tail_missing`, so a year typed into the config would be wrong by one season — and
    the artifact is the only thing that knows that."""
    pd.DataFrame([{"season": "2003-04", "rows_kept": 369, "coverage_class": "tail_missing"},
                  {"season": "2004-05", "rows_kept": 1154, "coverage_class": "covered"},
                  {"season": "2005-06", "rows_kept": 2251, "coverage_class": "covered"}]
                 ).to_csv(tmp_path / "preseason_coverage.csv", index=False)
    assert SC.head_first_season(_cfg(eda_dir=str(tmp_path))) == "2004-05"


def test_the_window_cut_raises_a_window_and_never_lowers_one():
    """`max`, not replace. A head already fitting inside coverage must keep its own window —
    the cut exists to keep an era dummy out, not to re-decide `first_season`."""
    assert SC.head_first_season(_cfg(first_season="2018-19")) == "2018-19"
    assert SC.head_first_season(_cfg(first_season="1996-97")) > "1996-97"


def test_head_frame_passes_no_hook_when_the_block_is_off(monkeypatch):
    seen = {}
    monkeypatch.setattr(SC, "composition_frame",
                        lambda cfg, share_hook=None: seen.setdefault("hook", share_hook))
    SC.head_frame(_cfg(adopt=False))
    assert seen["hook"] is None


def test_head_frame_blends_through_the_measured_hook_rather_than_a_second_copy(monkeypatch,
                                                                               tmp_path):
    """What ships has to be the function sessions 4b-4d scored, at the declared `k` and on
    the declared route. A reimplementation here would be a second chance to get the volume
    weight wrong, and nothing in the repo would compare the two."""
    pd.DataFrame([{"season": "2020-21", "player_id": 1, "mpg_pre": 24.0, "min_pre": 80.0}]
                 ).to_parquet(tmp_path / "preseason.parquet")
    cfg = _cfg()
    cfg["data"]["features_dir"] = str(tmp_path)

    seen = {}
    monkeypatch.setattr(SC, "composition_frame",
                        lambda cfg, share_hook=None: seen.setdefault("hook", share_hook))
    SC.head_frame(cfg)

    lagged = _lagged([("2020-21", 1, 0.20)])
    reference = blend_hook(_panel([("2020-21", 1, 24.0, 80.0)]), 80.0, "offset_only")
    assert np.allclose(seen["hook"](lagged)["w_share"], reference(lagged)["w_share"])
    # `offset_only` is the shipped route, so the ordering must still be the incumbent share.
    assert np.allclose(seen["hook"](lagged)["order_share"], lagged["w_share"])
    assert not np.allclose(seen["hook"](lagged)["w_share"], lagged["w_share"])


def test_the_measurement_modules_never_see_the_configs_blend(monkeypatch):
    """**The control-arm trap, and it is the reason `head_frame` exists as a separate door.**
    `composition_preseason.blended_frame` at a nesting `k` is the `base` control every gate
    arm is measured against; if the head's config could reach it, that control would quietly
    become blended and three sessions of margins would collapse toward zero with nothing
    raising."""
    seen = {}
    monkeypatch.setattr(SC, "composition_frame",
                        lambda cfg, share_hook=None: seen.setdefault("hook", share_hook))
    monkeypatch.setattr(CP, "composition_frame", SC.composition_frame)
    CP.blended_frame(_cfg(), _panel([("2020-21", 1, 24.0, 80.0)]), INCUMBENT_K)
    assert seen["hook"] is None


def test_a_missing_panel_names_the_rollback_flag_rather_than_failing_on_a_join(monkeypatch):
    """The head now needs `preseason.parquet` to build its own offset, so the failure has to
    arrive at the read with the flag that undoes it, not several frames later."""
    cfg = _cfg()
    cfg["data"]["features_dir"] = "does/not/exist"
    with pytest.raises(FileNotFoundError, match="stan.composition.preseason.adopt"):
        SC.head_frame(cfg)
