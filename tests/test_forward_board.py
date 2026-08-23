"""The forward board comparison — the reductions, not the simulator underneath them."""

import numpy as np
import pandas as pd

from src.sim.forward_board import (board_frame, compare_boards, unit_census,
                                   untouched_ids)


def _board(ids, totals):
    frame = pd.DataFrame({"player_id": ids, "mean_total": totals,
                          "sd_total": 1.0})
    frame["rank"] = frame["mean_total"].rank(ascending=False,
                                             method="first").astype(int)
    return frame.sort_values("rank").reset_index(drop=True)


def test_board_frame_ranks_mean_season_totals():
    ctx = {"unit_ids": np.array([7, 8, 9])}
    # units x periods x sims: player 9 is the best in every sim.
    dk = np.zeros((3, 2, 4), dtype=np.float32)
    dk[0] += 10.0
    dk[1] += 5.0
    dk[2] += 20.0
    board = board_frame(ctx, {"dk_pts": dk})
    assert list(board["player_id"]) == [9, 7, 8]
    assert list(board["rank"]) == [1, 2, 3]
    assert float(board.loc[0, "mean_total"]) == 40.0   # 20 x 2 periods


def test_compare_boards_prices_the_missing_by_reference_rank():
    ref = _board([1, 2, 3, 4], [400.0, 300.0, 200.0, 100.0])
    alt = _board([1, 2, 4], [390.0, 310.0, 90.0])       # player 3 (ref rank 3) missing
    row = compare_boards(ref, alt, "x")
    assert row["n_shared"] == 3
    assert row["n_ref_only"] == 1 and row["n_alt_only"] == 0
    assert row["best_missing_rank"] == 3
    assert row["missing_top100"] == 1
    assert row["spearman"] == 1.0, "the shared order is untouched"


def test_compare_boards_overlap_counts_the_top_k_intersection():
    ids = list(range(1, 25))
    ref = _board(ids, [float(100 - i) for i in ids])
    # Swap two players across the top-16 boundary.
    totals = {i: float(100 - i) for i in ids}
    totals[16], totals[17] = totals[17], totals[16]
    alt = _board(ids, [totals[i] for i in ids])
    row = compare_boards(ref, alt, "x")
    assert row["overlap_16"] == 15
    assert row["n_shared"] == 24


# ── §4's acceptance reductions (docs/rookie-rates-plan.md §5g) ────────────────

def _ctx(rows):
    """rows: (player_id, unit_family, lag_rung)."""
    units = pd.DataFrame(rows, columns=["player_id", "unit_family", "lag_rung"])
    return {"units": units, "unit_ids": units["player_id"].to_numpy()}


def test_the_census_separates_the_three_populations_and_prices_them():
    ctx = _ctx([(1, "veteran", "veteran"), (2, "veteran", "veteran"),
                (3, "veteran", "returnee_lag2"), (4, "rookie", "true_rookie")])
    board = _board([1, 2, 3, 4], [400.0, 300.0, 200.0, 100.0])
    out = unit_census(ctx, board, priced={2, 4}).set_index("population")
    assert out.loc["veteran", "units"] == 2
    assert out.loc["lag_recovered", "units"] == 1
    assert out.loc["true_rookie", "units"] == 1
    assert out.loc["veteran", "adp_priced"] == 1 and out.loc["true_rookie",
                                                             "adp_priced"] == 1
    assert out.loc["lag_recovered", "adp_priced"] == 0
    assert out.loc["true_rookie", "best_rank"] == 4


def test_a_population_with_no_units_reports_no_rank_rather_than_raising():
    """The FAIL case §4's acceptance is looking for: a board with no rookies on it. It has
    to come back as a zero rather than as an exception, because the whole point of the
    reading is to say so."""
    ctx = _ctx([(1, "veteran", "veteran")])
    out = unit_census(ctx, _board([1], [400.0]), priced=set()).set_index("population")
    assert out.loc["true_rookie", "units"] == 0
    assert out.loc["true_rookie", "best_rank"] == -1


def test_a_pre_union_frame_is_all_veteran():
    """Every caller that predates the two families hands over a frame with neither column,
    and `component_units`' own default for that case is `veteran`."""
    units = pd.DataFrame({"player_id": [1, 2]})
    out = unit_census({"units": units, "unit_ids": units["player_id"].to_numpy()},
                      _board([1, 2], [2.0, 1.0]), priced=set()).set_index("population")
    assert out.loc["veteran", "units"] == 2
    assert out.loc["lag_recovered", "units"] == 0
    assert untouched_ids({"units": units}) == {1, 2}


def test_untouched_is_rung_zero_veterans_only():
    ctx = _ctx([(1, "veteran", "veteran"), (3, "veteran", "returnee_lag2"),
                (4, "rookie", "true_rookie")])
    assert untouched_ids(ctx) == {1}
