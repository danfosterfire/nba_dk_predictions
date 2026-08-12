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
                                                     layout_exchangeable, observed_layout,
                                                     period_statistics)
from src.models.games_played import allocate_spells


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
