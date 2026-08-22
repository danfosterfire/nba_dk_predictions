"""The forward board comparison — the reductions, not the simulator underneath them."""

import numpy as np
import pandas as pd

from src.sim.forward_board import board_frame, compare_boards


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
