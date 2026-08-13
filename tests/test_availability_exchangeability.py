"""Tests for the exchangeable-trials instrument.

The round's whole argument is that two layouts with the **same** games played are different
seasons, so the things worth pinning are the two halves of that sentence:

1. **Both layouts preserve `gp` exactly, on every row.** If either one moved the count, the
   period-unit gaps would be confounded with a marginal change and the comparison would be
   measuring the wrong thing. `layout_ladder` asserts it at run time; this pins the assertion
   itself, including the two boundary rows a random placement gets wrong most easily — a
   player who missed nothing and one who played nothing.
2. **The layouts differ where they are supposed to and nowhere else.** Exchangeable placement
   has to scatter, spell placement has to clump, and the period statistics have to see the
   difference. A statistic that could not separate a 40-game block from forty single-game
   absences would report a null for the arrangement no matter how wrong the assumption was.

Plus the two guards that keep a null honest: `_attach_gaps` must refuse to divide by a gap
that is not there, and `gp_margin_invariance` must survive a missing artifact rather than
taking the build down.
"""

import numpy as np
import pandas as pd

from src.models.availability_exchangeability import (ARRANGEMENT_TOL, _attach_gaps,
                                                     _longest_run, cell_index,
                                                     gp_margin_invariance,
                                                     missed_decomposition,
                                                     layout_exchangeable, observed_layout,
                                                     period_statistics)
from src.models.games_played import (EdgeResampler, _fit_within_gaps, allocate_spells,
                                     edge_blocks, layout_tenure, missed_share_bin)


def _panel(vectors: list[list[int]], period_size: int = 4) -> pd.DataFrame:
    """One player-season per vector, its games chopped into equal scoring periods."""
    rows = []
    for i, played in enumerate(vectors):
        for g, flag in enumerate(played):
            rows.append({"season": "2022-23", "player_id": 100 + i,
                         "team_game_index": g, "played": flag,
                         "period_index": g // period_size + 1})
    return pd.DataFrame(rows)


def test_exchangeable_layout_preserves_games_played():
    gp = np.array([0, 1, 40, 81, 82])
    team_games = np.full(5, 82)
    played = layout_exchangeable(gp, team_games, seed=3)
    assert np.array_equal(played.sum(axis=1), gp)
    # Nothing is placed past a row's own schedule length, which is what makes the ragged
    # (rows x max_games) rectangle safe to slice per row.
    short = layout_exchangeable(np.array([5]), np.array([10]), seed=1)
    assert short[0, 10:].sum() == 0


def test_spell_layout_preserves_games_played():
    gp = np.array([10, 41, 70, 82])
    team_games = np.full(4, 82)
    played = allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=11)
    assert np.array_equal(played.sum(axis=1), gp)


def test_spell_layout_clumps_and_exchangeable_scatters():
    """The item's own sentence, as a test: same `gp`, very different seasons."""
    gp = np.full(200, 62)
    team_games = np.full(200, 82)
    clustered = allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=5)
    scattered = layout_exchangeable(gp, team_games, seed=5)

    def longest_absence(mat):
        return np.mean([_longest_run(mat[i] == 0) for i in range(len(mat))])

    assert np.array_equal(clustered.sum(axis=1), scattered.sum(axis=1))
    assert longest_absence(clustered) > 2.0 * longest_absence(scattered)


def test_period_statistics_read_the_arrangement_not_the_count():
    """One eight-game block against eight scattered absences, both at `gp = 12`."""
    block = [1] * 6 + [0] * 8 + [1] * 6
    scattered = [1, 0] * 8 + [1] * 4
    assert sum(block) == sum(scattered) == 12
    index = cell_index(_panel([block]))
    blocked = period_statistics(observed_layout(index), index)
    index = cell_index(_panel([scattered]))
    spread = period_statistics(observed_layout(index), index)
    assert blocked["p_dead_period"] > spread["p_dead_period"] == 0.0
    assert blocked["longest_dead_run"] >= 1 and spread["longest_dead_run"] == 0


def test_period_statistics_ignore_periods_with_no_scheduled_game():
    """A bye is not an absence — a period the team does not play must not count as dead."""
    panel = _panel([[1] * 8])
    panel.loc[panel["team_game_index"] >= 4, "period_index"] = 9   # skips periods 2-8
    index = cell_index(panel)
    stats = period_statistics(observed_layout(index), index)
    assert stats["p_dead_period"] == 0.0


def test_longest_run():
    assert _longest_run(np.array([0, 1, 1, 0, 1, 1, 1, 0], dtype=bool)) == 3
    assert _longest_run(np.zeros(5, dtype=bool)) == 0


def test_gaps_refuse_a_denominator_that_is_not_there():
    """A metric the arrangement barely moves gets no `recovered_share`, by design."""
    def row(arm, dead, half):
        return {"analysis": "period_layout", "population": "all", "arm": arm,
                "p_dead_period": dead, "p_half_period": half,
                "longest_dead_run": 1.0, "p_dead_run": 0.1}

    frame = pd.DataFrame([row("observed", 0.20, 0.40), row("clustered", 0.18, 0.40),
                          row("exchangeable", 0.10, 0.40 * (1 + ARRANGEMENT_TOL / 2))])
    gaps = _attach_gaps(frame)
    gaps = gaps[gaps["analysis"] == "period_gap"].set_index("metric")
    assert gaps.loc["p_dead_period", "arrangement_sensitive"]
    assert np.isclose(gaps.loc["p_dead_period", "recovered_share"], 0.8)
    assert not gaps.loc["p_half_period", "arrangement_sensitive"]
    assert np.isnan(gaps.loc["p_half_period", "recovered_share"])


def test_gp_margin_invariance_tolerates_a_missing_artifact(tmp_path):
    """The citation is a convenience, not a dependency — a fresh clone still runs."""
    assert gp_margin_invariance(tmp_path).empty


# ── the tenure factor ─────────────────────────────────────────────────────────

def test_tenure_layout_nests_the_shipped_one_at_zero_tenure():
    """The house nesting discipline: no edge block anywhere must reproduce
    `allocate_spells` exactly, not merely closely, on the same seed."""
    gp = np.array([0, 5, 41, 70, 82])
    team_games = np.full(5, 82)
    zero = np.zeros(5, dtype=np.int64)
    assert np.array_equal(
        layout_tenure(gp, team_games, zero, zero, mu=0.49, kappa=3.9, seed=7),
        allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=7))


def test_tenure_layout_preserves_games_played_and_lays_blocks_at_the_ends():
    gp = np.array([40, 40, 40, 1])
    team_games = np.full(4, 82)
    pre = np.array([20, 0, 10, 40])
    post = np.array([0, 20, 10, 41])
    played = layout_tenure(gp, team_games, pre, post, mu=0.49, kappa=3.9, seed=3)
    assert np.array_equal(played.sum(axis=1), gp)
    # Each drawn block is a run of zeros flush against its end, and the game beside it is
    # played — which is what makes the drawn length the block length rather than a lower
    # bound on it. Row 3 is the degenerate case: one game played, both ends claimed.
    for i in (0, 1, 2):
        assert played[i, :pre[i]].sum() == 0
        assert played[i, 82 - post[i]:].sum() == 0
        if pre[i]:
            assert played[i, pre[i]] == 1
        if post[i]:
            assert played[i, 82 - post[i] - 1] == 1


def test_tenure_layout_refuses_a_block_longer_than_the_missed_total():
    """Silently trimming it would move games played, and every gap in the ladder is only a
    measurement of arrangement because no arm does that."""
    try:
        layout_tenure(np.array([70]), np.array([82]), np.array([10]), np.array([10]),
                      mu=0.49, kappa=3.9)
    except ValueError as exc:
        assert "longer than the missed total" in str(exc)
    else:
        raise AssertionError("a 20-game edge block on 12 missed games was accepted")


def test_overflow_policies_differ_only_in_what_they_do_with_the_excess():
    """Eleven spells and four gaps. `collapse` throws the draw away and lays the missed
    total as one block; `merge` fuses the shortest until it fits, so the count is the
    geometric maximum. Both conserve the total, which is what keeps `gp` exact."""
    lengths = [1, 1, 2, 1, 5, 1, 1, 3, 2, 1, 2]
    rng = np.random.default_rng(0)
    assert _fit_within_gaps(lengths, 4, sum(lengths), "collapse", rng) == [sum(lengths)]
    merged = _fit_within_gaps(lengths, 4, sum(lengths), "merge", rng)
    assert len(merged) == 4 and sum(merged) == sum(lengths)
    # The long tail of the draw survives — merging fuses the shortest, so the 5 is intact.
    assert max(merged) >= 5


def test_overflow_merge_shortens_the_longest_dead_run_on_absent_rows():
    """The whole point at the layout level: a heavily-absent row stops being one block."""
    gp, team_games = np.full(200, 12), np.full(200, 82)
    collapsed = allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=1)
    merged = allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=1, overflow="merge")
    assert np.array_equal(collapsed.sum(axis=1), merged.sum(axis=1))

    def longest(mat):
        return np.mean([_longest_run(mat[i] == 0) for i in range(len(mat))])

    assert longest(merged) < longest(collapsed)


def test_overflow_collapse_is_the_shipped_draw_untouched():
    """The default may not move the rng stream, or every artifact drawn through
    `allocate_spells` would shift on a change that was supposed to be opt-in."""
    gp = np.array([10, 41, 70, 82])
    team_games = np.full(4, 82)
    assert np.array_equal(
        allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=11),
        allocate_spells(gp, team_games, mu=0.49, kappa=3.9, seed=11, overflow="collapse"))


def test_edge_blocks_read_the_leading_and_trailing_runs():
    """The leading run of missed games *is* the pre-tenure block, by construction — that
    identity is what lets the layout draw a block length without a fitted entry head."""
    played = [0, 0, 0, 1, 1, 0, 1, 1, 0, 0]
    panel = pd.DataFrame([
        {"season": "2022-23", "player_id": 1, "team_id": 5, "game_id": g,
         "team_game_index": g, "played": flag, "in_appearance_window": int(3 <= g <= 7),
         "status": "not_rostered" if g < 3 else "inactive"}
        for g, flag in enumerate(played)])
    rows = pd.DataFrame({"season": ["2022-23"], "player_id": [1], "role_bin": [2]})
    cell = edge_blocks(panel, rows).iloc[0]
    assert (cell["pre"], cell["post"], cell["interior"]) == (3, 2, 1)
    assert cell["missed"] == 6 and cell["gp"] == 4
    assert cell["pre_not_rostered"] == 3 and cell["post_not_rostered"] == 0
    assert np.isclose(cell["pre_frac"], 0.5)


def test_missed_share_bins_are_fixed_edges_not_quantiles():
    """A validation row has to land in the bucket a fitting row with the same missed share
    lands in, which is the whole reason the edges are constants."""
    bins = missed_share_bin(np.array([0, 4, 16, 40, 41, 82]), np.full(6, 82))
    # 41/82 is exactly 0.50 and belongs to the top bucket — the bins are left-closed, so a
    # boundary row lands in the same place on both halves of the split.
    assert list(bins) == [1, 1, 2, 3, 4, 4]


def test_edge_resampler_never_draws_more_edge_than_the_player_missed():
    """Two fractions rounded independently can sum past 1.0; the draw has to resolve that
    against the missed total or `layout_tenure` gets a negative interior."""
    cells = pd.DataFrame({
        "missed": [40, 40, 40], "team_games": [82, 82, 82], "gp": [42, 42, 42],
        "role_bin": [1, 1, 1], "pre_frac": [0.5, 0.7, 0.34],
        "post_frac": [0.5, 0.3, 0.67], "pre": [20, 28, 14], "post": [20, 12, 27]})
    edges = EdgeResampler(cells)
    gp = np.array([30, 60, 82, 1])
    team_games = np.full(4, 82)
    pre, post = edges.draw(gp, team_games, np.ones(4, dtype=np.int64),
                           np.random.default_rng(0))
    assert (pre + post <= team_games - gp).all()
    assert (pre >= 0).all() and (post >= 0).all()
    # A player who missed nothing gets no block, whatever the pool holds.
    assert pre[2] == post[2] == 0


def test_missed_decomposition_splits_edge_blocks_by_roster_status():
    """An edge block the player was not rostered for is not an absence, and must not pool
    with one he was rostered through. The two run opposite ways across role, so pooling
    them reports a flat aggregate over two real gradients."""
    games = []
    for g in range(10):
        games.append({"season": "2022-23", "player_id": 1, "team_id": 5, "game_id": g,
                      "team_game_index": g,
                      # signed in game 4: not rostered before, then one interior miss
                      "played": int(g >= 4 and g != 7),
                      "in_appearance_window": int(g >= 4),
                      "status": "not_rostered" if g < 4 else
                                ("inactive" if g == 7 else "played")})
    for g in range(10):
        games.append({"season": "2022-23", "player_id": 2, "team_id": 5, "game_id": g,
                      "team_game_index": g,
                      # one interior miss, then a season-ending injury at game 6 —
                      # rostered throughout, so the same 5 missed games and the same
                      # aggregate edge share as player 1, from a different process
                      "played": int(g < 6 and g != 2),
                      "in_appearance_window": int(g < 6),
                      "status": "inactive" if (g >= 6 or g == 2) else "played"})
    panel = pd.DataFrame(games)
    rows = pd.DataFrame({"season": ["2022-23"] * 2, "player_id": [1, 2],
                         "role_bin": [1, 4]})
    out = missed_decomposition(panel, rows).set_index("population")

    assert out.loc["<12 mpg", "edge_not_rostered_games"] == 4
    assert out.loc["<12 mpg", "edge_still_rostered_games"] == 0
    assert out.loc["<12 mpg", "interior_games"] == 1
    assert out.loc["30+ mpg", "edge_not_rostered_games"] == 0
    assert out.loc["30+ mpg", "edge_still_rostered_games"] == 4
    # The aggregate `edge_share` is identical for the two — 0.80 each — which is exactly
    # the pooling this split exists to undo.
    assert np.isclose(out.loc["<12 mpg", "edge_share"], 0.8)
    assert np.isclose(out.loc["30+ mpg", "edge_share"], 0.8)
    assert out.loc["<12 mpg", "not_rostered_share_of_edge"] == 1.0
    assert out.loc["30+ mpg", "not_rostered_share_of_edge"] == 0.0
