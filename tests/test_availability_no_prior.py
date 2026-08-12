"""Tests for the no-prior population measurement.

The round exists to settle one decision, so the things worth pinning are the three ways the
measurement could quietly answer a different question than the one asked:

1. **The classes are what they say they are.** A "returning veteran" is a player who
   appeared *before* S−1 and not in it — identifiable from the panel and not from the
   design, since the design has no row for him either way. Getting the gap off by one turns
   rookies into veterans and the whole table into a statement about nothing.
2. **The imputed bucket is decision 2's own map.** The decision grades by draft position
   through mean MPG and `season_effects.ROLE_EDGES`; if this module cut on different edges
   it would be refuting a rule nobody wrote.
3. **`rho_imputed_error` points the way the verdict claims.** Negative means "the bucket
   says he is more reliable than he is", which is the failure mode the decision was
   suspected of and the sign the write-up leans on.

Plus the ratio guard: the left-tail axis has a genuine zero, and a spread computed over it
must come back `NaN` rather than `inf`.
"""

import numpy as np
import pandas as pd

from src.eda.season_effects import ROLE_LABELS
from src.models.availability_no_prior import (FITTED_ROLE_RHO, classify, imputed_bucket,
                                              ladder, primary_team_cells, realized,
                                              spreads)


def _panel(rows: list[tuple[str, int, int, int]]) -> pd.DataFrame:
    """(season, player_id, team_id, games_played) → a panel of 4-game seasons."""
    out = []
    for season, player, team, gp in rows:
        for g in range(4):
            out.append({"season": season, "player_id": player, "team_id": team,
                        "game_id": 1000 + g, "game_date": pd.Timestamp("2020-01-01")
                        + pd.Timedelta(days=g), "played": int(g < gp), "min": 20.0})
    return pd.DataFrame(out)


def test_primary_team_cells_attributes_a_traded_player_to_his_last_team():
    panel = pd.concat([_panel([("2020-21", 7, 100, 4)]),
                       _panel([("2020-21", 7, 200, 2)])], ignore_index=True)
    panel.loc[panel["team_id"] == 200, "game_date"] += pd.Timedelta(days=30)
    cells = primary_team_cells(panel)
    assert len(cells) == 1
    # Only the last team's schedule counts, so the denominator is one team's, not both.
    assert cells.iloc[0]["team_games"] == 4 and cells.iloc[0]["gp"] == 2


def test_classify_separates_rookies_from_returning_veterans():
    seasons = ["2018-19", "2019-20", "2020-21"]
    panel = _panel([("2018-19", 1, 10, 3), ("2020-21", 1, 10, 3),   # two seasons out
                    ("2020-21", 2, 10, 3),                          # never appeared before
                    ("2019-20", 3, 10, 3), ("2020-21", 3, 10, 3)])  # one season out
    cells = primary_team_cells(panel)
    # The design covers nobody, so every row is a no-design row and the split is the gap's.
    design = pd.DataFrame({"season": [], "player_id": []})
    out = classify(cells, design, seasons).set_index(["season", "player_id"])
    assert out.loc[("2020-21", 1), "klass"] == "gap_2_seasons"
    assert out.loc[("2020-21", 2), "klass"] == "rookie"
    assert out.loc[("2020-21", 2), "no_prior_appearance"]
    assert out.loc[("2020-21", 3), "klass"] == "gap_1_season"
    assert not out.loc[("2020-21", 1), "no_prior_appearance"]


def test_imputed_bucket_is_decision_2s_own_cut():
    """`ROLE_EDGES` is [0, 12, 24, 30, 60] on prior MPG — decision 3's fixed bins."""
    assert ROLE_LABELS[imputed_bucket(5.0) - 1] == "<12 mpg"
    assert ROLE_LABELS[imputed_bucket(13.4) - 1] == "12-24"
    assert ROLE_LABELS[imputed_bucket(26.8) - 1] == "24-30"
    assert ROLE_LABELS[imputed_bucket(34.0) - 1] == "30+ mpg"
    # Outside the top edge still lands in a real bucket rather than off the end.
    assert ROLE_LABELS[imputed_bucket(90.0) - 1] == "30+ mpg"


def test_realized_recovers_a_binomial_at_no_overdispersion():
    """A group generated with one shared rate has an implied `rho` near zero."""
    rng = np.random.default_rng(4)
    n = 82
    group = pd.DataFrame({"team_games": np.full(4000, n),
                          "gp": rng.binomial(n, 0.8, size=4000),
                          "minutes": 0.0})
    group["mpg"] = 0.0
    out = realized(group)
    assert abs(out["mu"] - 0.8) < 0.01
    assert abs(out["rho_implied"]) < 0.01


def test_imputed_error_is_negative_when_the_bucket_says_too_reliable():
    """The sign the verdict leans on, pinned on a constructed group."""
    rng = np.random.default_rng(9)
    # A group whose MPG maps to `24-30` (rho 0.2535) but which realizes more spread.
    rate = rng.beta(*_ab(0.83, 0.30), size=3000)
    group = pd.DataFrame({"team_games": np.full(3000, 82),
                          "gp": rng.binomial(82, rate), "minutes": 0.0, "mpg": 26.8})
    table = ladder(group.assign(no_prior_appearance=False, klass="rookie",
                                draft_bucket="lottery_top5"), group)
    row = table[table["population"] == "no_design_all"].iloc[0]
    assert row["imputed_bucket"] == "24-30"
    assert row["rho_imputed"] == FITTED_ROLE_RHO[2]
    assert row["rho_imputed_error"] < 0


def _ab(mu: float, rho: float) -> tuple[float, float]:
    scale = (1.0 - rho) / rho
    return mu * scale, (1.0 - mu) * scale


def test_spread_refuses_a_ratio_over_a_genuine_zero():
    draft = pd.DataFrame({"population": ["draft__a", "draft__b"],
                          "mu": [0.25, 0.83], "rho_implied": [0.30, 0.35],
                          "p_gp_below_10": [0.0, 0.43]})
    out = spreads(draft).set_index("axis")
    assert np.isnan(out.loc["left tail", "spread"])
    assert np.isclose(out.loc["left tail", "gap"], 0.43)
    assert np.isclose(out.loc["level", "spread"], 0.83 / 0.25)
