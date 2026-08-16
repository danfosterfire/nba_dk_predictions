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
import pytest

from src.eda.season_effects import ROLE_LABELS
from src.models.availability_no_prior import (FITTED_ROLE_RHO, MIN_CELL, NO_PRESEASON,
                                              PRESEASON_KEY_ARMS, PRESEASON_SHARE_EDGES,
                                              PRESEASON_SHARE_LABELS, SHIPPED_LEVEL_ARM,
                                              _pool, appearance_gap,
                                              attach_preseason_keys, classify, implied_rho,
                                              imputed_bucket, ladder, level_keys,
                                              level_rates, level_tables, preseason_bucket,
                                              primary_team_cells, realized, spreads)


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


# ── §8b, the level ladder ─────────────────────────────────────────────────────
#
# The arms are pooling KEYS over one estimator, so the things worth pinning are the ways a
# key can quietly be the wrong one: the incumbent has to nest exactly, the cross has to
# refuse to hand a returning veteran a rookie's rate, a thin cell has to fall back rather
# than be believed, and nothing may see the target season.

def _history(rows: list[tuple[str, int, int, str, int]]) -> pd.DataFrame:
    """(season, player_id, gp, draft_bucket, gap) → a no-design pooling history."""
    return level_keys(pd.DataFrame(
        [{"season": s, "player_id": p, "gp": gp, "team_games": 82,
          "draft_bucket": bucket, "gap": gap} for s, p, gp, bucket, gap in rows]))


def _bulk(season: str, n: int, gp: int, bucket: str, gap: int,
          first_id: int = 0) -> list[tuple[str, int, int, str, int]]:
    return [(season, first_id + i, gp, bucket, gap) for i in range(n)]


def test_pooled_arm_reproduces_the_single_scalar_it_replaces():
    """The nesting discipline `n_rho = 1` and `U_n = 0` already carry, one layer down.

    `pooled` is not a degenerate case of the graded arm to be checked loosely — it is the
    behaviour that shipped, and an arm that cannot reproduce it exactly makes every recorded
    figure taken under it unreadable.
    """
    history = _history(_bulk("2019-20", 60, 20, "undrafted", 0)
                       + _bulk("2019-20", 60, 70, "lottery_top5", 0, first_id=100))
    rows = _history(_bulk("2020-21", 3, 0, "lottery_top5", 0))
    rate, rung = level_rates(rows, level_tables(history, "2020-21",
                                                ["2019-20", "2020-21"], "pooled"))
    expected = (60 * 20 + 60 * 70) / (120 * 82)
    assert np.allclose(rate, expected)
    assert (rung == 0).all()          # the `all` rung IS the graded rung for this arm


def test_the_cross_refuses_a_returning_veteran_his_draft_buckets_rookie_rate():
    """The measured reason `tenure_draft` exists rather than `draft`.

    A first-overall pick's bucket says 0.85 and a returning ex-lottery pick realizes half
    that; keying both on the bucket alone averages a gradient with a flat. So the two arms
    are asked for the same veteran's rate and required to disagree, in the direction the
    ladder measured.
    """
    history = _history(_bulk("2019-20", 60, 70, "lottery_top5", 0)
                       + _bulk("2019-20", 60, 24, "lottery_top5", 3, first_id=100))
    veteran = _history([("2020-21", 500, 0, "lottery_top5", 3)])
    seasons = ["2019-20", "2020-21"]
    by_draft, _ = level_rates(veteran, level_tables(history, "2020-21", seasons, "draft"))
    by_cross, _ = level_rates(veteran, level_tables(history, "2020-21", seasons,
                                                    "tenure_draft"))
    assert np.isclose(by_draft[0], (60 * 70 + 60 * 24) / (120 * 82))   # both classes pooled
    assert np.isclose(by_cross[0], 24 / 82)                            # returning only
    assert by_cross[0] < by_draft[0]


def test_a_thin_cell_falls_back_a_rung_and_says_so():
    """`MIN_CELL` is a reason to prefer a coarser key, and the rung index is the evidence.

    A graded arm that silently fell back to the pooled rate on every row and a graded arm
    that did nothing are the same table without this.
    """
    history = _history(_bulk("2019-20", MIN_CELL, 70, "lottery_top5", 0)
                       + _bulk("2019-20", MIN_CELL - 1, 20, "undrafted", 0, first_id=200))
    rows = _history([("2020-21", 1, 0, "lottery_top5", 0),
                     ("2020-21", 2, 0, "undrafted", 0)])
    rate, rung = level_rates(rows, level_tables(history, "2020-21",
                                                ["2019-20", "2020-21"], "tenure_draft"))
    assert rung[0] == 0 and np.isclose(rate[0], 70 / 82)     # its own cell clears
    # The undrafted cell is one row short, so it takes `tenure_class` — every rookie pooled.
    assert rung[1] == 1
    pooled_rookies = (MIN_CELL * 70 + (MIN_CELL - 1) * 20) / ((2 * MIN_CELL - 1) * 82)
    assert np.isclose(rate[1], pooled_rookies)


def test_the_terminal_rung_applies_however_thin_it_is():
    """A rate is not optional. Falling off the end of the ladder is an assertion, not a NaN,
    and the last rung is therefore unconditional — a two-row league still returns a rate."""
    history = _history(_bulk("2019-20", 2, 41, "undrafted", 0))
    rows = _history([("2020-21", 9, 0, "lottery_top5", 0)])
    rate, rung = level_rates(rows, level_tables(history, "2020-21",
                                                ["2019-20", "2020-21"], "tenure_draft"))
    assert np.isclose(rate[0], 0.5) and rung[0] == 2


def test_the_rate_cannot_see_the_target_season_or_a_season_after_it():
    """Point-in-time by construction, which is the property the whole estimator rests on.

    Changing the target season's own outcomes — and a later season's — must leave its rate
    bit-identical, or the simulator is being handed a number it could not have had.
    """
    seasons = ["2019-20", "2020-21", "2021-22"]
    base = _bulk("2019-20", 60, 30, "undrafted", 0)
    rows = _history([("2020-21", 7, 0, "undrafted", 0)])
    before, _ = level_rates(rows, level_tables(_history(base), "2020-21", seasons, "draft"))
    contaminated = _history(base + _bulk("2020-21", 60, 82, "undrafted", 0, first_id=300)
                            + _bulk("2021-22", 60, 82, "undrafted", 0, first_id=600))
    after, _ = level_rates(rows, level_tables(contaminated, "2020-21", seasons, "draft"))
    assert np.allclose(before, after)


def test_appearance_gap_labels_a_target_row_from_history_strictly_before_it():
    """The simulator labels a season it has no outcomes for, so the two frames differ.

    Getting this off by one turns every rookie into a returning veteran and hands the whole
    incoming draft class a rate estimated on players who washed out of the league.
    """
    seasons = ["2018-19", "2019-20", "2020-21"]
    history = pd.DataFrame({"season": ["2018-19", "2019-20"], "player_id": [1, 3]})
    rows = pd.DataFrame({"season": "2020-21", "player_id": [1, 2, 3]})
    assert list(appearance_gap(rows, history, seasons)) == [2, 0, 1]


def test_implied_rho_generalizes_the_scalar_form_it_replaces():
    """A graded arm and a flat one have to be measured with the same instrument, so the
    vector form must reduce to the scalar one exactly where they overlap."""
    rng = np.random.default_rng(11)
    group = pd.DataFrame({"team_games": np.full(2000, 82),
                          "gp": rng.binomial(82, rng.beta(*_ab(0.45, 0.3), size=2000)),
                          "minutes": 0.0, "mpg": 8.0})
    mu = float(group["gp"].sum() / group["team_games"].sum())
    inflation, rho = implied_rho(group["gp"].to_numpy(), group["team_games"].to_numpy(),
                                 np.full(len(group), mu))
    assert np.isclose(realized(group)["rho_implied"], rho)
    assert np.isclose(realized(group)["inflation"], inflation)


def test_spread_refuses_a_ratio_over_a_genuine_zero():
    draft = pd.DataFrame({"population": ["draft__a", "draft__b"],
                          "mu": [0.25, 0.83], "rho_implied": [0.30, 0.35],
                          "p_gp_below_10": [0.0, 0.43]})
    out = spreads(draft).set_index("axis")
    assert np.isnan(out.loc["left tail", "spread"])
    assert np.isclose(out.loc["left tail", "gap"], 0.43)
    assert np.isclose(out.loc["level", "spread"], 0.83 / 0.25)


# ── P4(a), the preseason key ──────────────────────────────────────────────────
#
# The ladder gained two axes, and each one can be wrong in a way a CRPS column cannot show.
# The key can silently not exist (a cross whose cells all fall back is indistinguishable
# from the arm it falls back to, which is why `origins_compared` was added); the bucket can
# treat "no preseason row" as a small share rather than as its own event; and the estimator
# population can leak January signings into a rate handed to October rosters.

def _pre(rows, shares):
    """`_history` rows plus a preseason share per player_id — `None` means no row at all."""
    history = _history(rows)
    panel = pd.DataFrame([{"season": s, "player_id": p, "min_share_pre": shares[p]}
                          for s, p, *_ in rows if shares.get(p) is not None])
    if panel.empty:
        panel = pd.DataFrame(columns=["season", "player_id", "min_share_pre"])
    return attach_preseason_keys(history, panel)


def test_no_preseason_row_is_its_own_bucket_and_not_a_small_share():
    """`NO_PRESEASON` is a level. Collapsing it into `bench` would assert that a player who
    never took the floor in October and one who played nine minutes a night are the same
    event, which P1's census measured as false on exactly this population."""
    assert preseason_bucket(pd.Series([np.nan]))[0] == NO_PRESEASON
    assert preseason_bucket(pd.Series([0.0]))[0] == PRESEASON_SHARE_LABELS[0]
    assert preseason_bucket(pd.Series([0.02, 0.05, 0.20])).tolist() == list(
        PRESEASON_SHARE_LABELS)


def test_the_share_cuts_are_the_declared_ones_and_are_closed_on_the_right_side():
    """The edges are a-priori constants, so the only thing to pin is which side each is
    closed on — `pd.cut` is right-closed, so an exactly-even share is `fringe`, not
    `rotation`, and the doc's "at or above an even split" would otherwise be wrong."""
    lo, hi = PRESEASON_SHARE_EDGES
    assert preseason_bucket(pd.Series([lo]))[0] == PRESEASON_SHARE_LABELS[0]
    assert preseason_bucket(pd.Series([hi]))[0] == PRESEASON_SHARE_LABELS[1]
    assert preseason_bucket(pd.Series([hi + 1e-9]))[0] == PRESEASON_SHARE_LABELS[2]


def test_the_cross_nests_inside_tenure_draft_rather_than_replacing_it():
    """The fallback rung has to be a coarsening of the same key. If `tenure_draft_pre` did
    not carry `tenure_draft` as its prefix, a row whose cell is under `MIN_CELL` would drop
    to a *different* partition and the ladder would be four estimators rather than one."""
    rows = _pre([("2020-21", 1, 40, "lottery", 0), ("2020-21", 2, 40, "undrafted", 3)],
                {1: 0.10, 2: 0.01})
    for _, row in rows.iterrows():
        assert row["tenure_draft_pre"].startswith(row["tenure_draft"] + "__")
        assert row["tenure_pre"].startswith(row["tenure_class"] + "__")


def test_a_thin_cross_falls_back_to_tenure_draft_and_reports_that_it_did():
    """The failure this whole column exists for: a graded arm that graded nothing. With one
    row per cell the cross cannot clear `MIN_CELL`, so every row must take the coarser rung
    AND `rung != 0` must say so — otherwise it is indistinguishable from a real grading."""
    history = _pre(_bulk("2019-20", MIN_CELL + 10, 20, "undrafted", 0)
                   + _bulk("2019-20", MIN_CELL + 10, 70, "lottery", 0, first_id=200),
                   {p: 0.10 for p in range(400)})
    # One target row whose preseason cell is `bench` — a cell the history never fills.
    target = _pre([("2020-21", 1, 0, "lottery", 0)], {1: 0.001})
    seasons = ["2019-20", "2020-21"]
    rate, rung = level_rates(target, level_tables(history, "2020-21", seasons,
                                                  "tenure_draft_preseason"))
    coarse, _ = level_rates(target, level_tables(history, "2020-21", seasons,
                                                 "tenure_draft"))
    assert (rung != 0).all()
    assert np.allclose(rate, coarse)


def test_an_arm_without_its_key_raises_rather_than_grading_nothing():
    """A preseason arm run on rows that never saw the panel would fall all the way to `all`
    and score as `pooled` while being labelled `tenure_draft_preseason`. That is the one
    failure mode that produces a plausible number, so it is an exception and not a fallback."""
    history = _history(_bulk("2019-20", MIN_CELL + 10, 20, "undrafted", 0))
    with pytest.raises(KeyError):
        level_tables(history, "2020-21", ["2019-20", "2020-21"], "tenure_draft_preseason")


def test_the_roster_estimator_pools_only_rostered_rows():
    """The axis P4 added outside its charter. The consumer is applied to October rosters, so
    an estimator pooled over January signings estimates the wrong quantity — and the two
    populations realize very different rates, which is what makes this a bug rather than a
    refinement."""
    history = _history(_bulk("2019-20", 60, 70, "undrafted", 0)
                       + _bulk("2019-20", 60, 10, "undrafted", 0, first_id=100))
    history["on_season_start_roster"] = np.where(history["player_id"] < 100, 1.0, 0.0)
    rows = _history(_bulk("2020-21", 1, 0, "undrafted", 0))
    seasons = ["2019-20", "2020-21"]
    pooled, _ = level_rates(rows, level_tables(_pool(history, "all"), "2020-21", seasons,
                                               "pooled"))
    roster, _ = level_rates(rows, level_tables(_pool(history, "roster"), "2020-21", seasons,
                                               "pooled"))
    assert np.allclose(pooled, (60 * 70 + 60 * 10) / (120 * 82))
    assert np.allclose(roster, 70 / 82)
    assert _pool(history, "all").shape[0] == 120


def test_the_simulator_refuses_to_be_configured_with_a_preseason_arm():
    """`sim/season` builds its own keys and never joins the preseason panel, so naming one
    of P4's arms there is a config that cannot work. It fails at the config rather than
    several frames later inside `level_tables`."""
    from src.sim import season as S

    for arm in PRESEASON_KEY_ARMS:
        with pytest.raises(ValueError, match="preseason"):
            S.no_design_level_arm({"sim": {"availability": {"no_design_level": arm}}})
    assert S.no_design_level_arm({}) == SHIPPED_LEVEL_ARM
