"""The season-total readout's composition and its group tuple — §5e of the rookie plan.

Nothing here fits a head or loads an artifact. What is checked is the arithmetic that turns
eleven heads into one number and the bookkeeping that decides which rows that number is
reported on, because those are the two places this readout can be wrong quietly:

- the **chain** — a season dk total assembled in the wrong order (makes drawn on realized
  attempts, `fg2a` taken as a fitted count rather than as `fga - fg3a`) still produces an
  entirely plausible number, so it is checked against the identities that hold on every
  draw and against the one case with no randomness left in it;
- the **groups** — `evaluate` now reports three populations it never used to, and a mask
  that silently covered nothing, or covered everything, would be invisible in the table.

The gate's own inputs (`admitted_rungs`, `ship_arms`) are read out of artifacts written by
earlier sessions, so they are exercised on synthetic tables whose every row is chosen to sit
on one side of the filter.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import season_total as st
from src.models import season_total_rookie as stro


# ── What the readout reads out of earlier sessions' artifacts ─────────────────

def test_admitted_rungs_takes_the_verdict_rows_only():
    table = pd.DataFrame([
        {"gate": "verdict", "group": "returnee_lag2", "metric": "admitted", "value": 1.0},
        {"gate": "verdict", "group": "thin_prior", "metric": "admitted", "value": 0.0},
        {"gate": "verdict", "group": "returnee_thin", "metric": "admitted", "value": 0.0},
        # A non-verdict row with the same group and a passing value must not be read as one.
        {"gate": "crps", "group": "thin_prior", "metric": "wins", "value": 1.0},
        # Nor may a group that is not a ladder rung — rung 0 is the bar, not a rung.
        {"gate": "verdict", "group": "veteran", "metric": "admitted", "value": 1.0},
    ], columns=["gate", "group", "metric", "value"])
    assert stro.admitted_rungs_from(table) == ("returnee_lag2",)


def test_ship_arms_reads_the_decision_population():
    table = pd.DataFrame([
        {"head": "reb", "population": "draftable", "ships": "slot_interaction_spline"},
        {"head": "reb", "population": "all", "ships": "no_fit_floor"},
        {"head": "ast", "population": "draftable", "ships": "no_fit_floor"},
    ])
    arms = stro.ship_arms_from(table)
    assert arms == {"reb": "slot_interaction_spline", "ast": "no_fit_floor"}


# ── The chain ─────────────────────────────────────────────────────────────────

def _arm(n_rows: int = 6, draws: int = 5, p: float = 0.45,
         fga: float = 500.0, fta: float = 100.0) -> tuple[dict, dict]:
    """Deterministic season counts and a conversion block at probability `p`."""
    counts = {c: np.full((draws, n_rows), float(v)) for c, v in
              (("fga", fga), ("fta", fta), ("reb", 300), ("ast", 200), ("stl", 60),
               ("blk", 40), ("tov", 120))}
    conversions = {f"{m}|{a}": (np.full((draws, n_rows), p), np.full(draws, 0.01))
                   for m, a in stro.CONVERSION_HEADS}
    return counts, conversions


def test_compose_holds_the_chain_identities_on_every_draw():
    """`fg2a = fga - fg3a`, and nothing is made that was not attempted.

    These are the two things a chain assembled in the wrong order breaks while still
    returning an entirely plausible season total: taking `fg2a` as a fitted count rather
    than as the difference, or drawing makes on a trials column that is not the one the
    head above it just produced.
    """
    counts, conversions = _arm()
    _, box = stro.compose_linear_dk(counts, conversions)
    assert np.array_equal(box["fg2a"], counts["fga"] - box["fg3a"])
    assert (box["fg3a"] <= counts["fga"]).all()
    assert (box["fg2m"] <= box["fg2a"]).all()
    assert (box["fg3m"] <= box["fg3a"]).all()
    assert (box["ftm"] <= counts["fta"]).all()
    assert np.allclose(box["pts"], 2 * box["fg2m"] + 3 * box["fg3m"] + box["ftm"])


def test_compose_applies_the_dk_weights_exactly():
    """The total is the DK linear sum of the box score it just drew, draw for draw."""
    counts, conversions = _arm()
    total, box = stro.compose_linear_dk(counts, conversions)
    expected = box["pts"] + 0.5 * box["fg3m"]
    for col, weight in (("reb", 1.25), ("ast", 1.5), ("stl", 2.0), ("blk", 2.0),
                        ("tov", -0.5)):
        expected = expected + weight * counts[col]
    assert np.allclose(total, expected)


def test_a_player_who_shoots_nothing_scores_only_the_non_shooting_weights():
    """Zero attempts is the one case with no randomness left in it at all."""
    counts, conversions = _arm(fga=0.0, fta=0.0)
    total, box = stro.compose_linear_dk(counts, conversions)
    assert np.allclose(box["pts"], 0.0)
    assert np.allclose(total, 1.25 * 300 + 1.5 * 200 + 2.0 * 60 + 2.0 * 40 - 0.5 * 120)


def test_compose_refuses_a_head_list_it_cannot_resolve():
    counts, conversions = _arm()
    del counts["fga"]
    with pytest.raises(KeyError):
        stro.compose_linear_dk(counts, conversions)


def test_bonus_is_zero_for_a_player_who_records_nothing():
    zeros = {c: np.zeros((4, 3)) for c in stro.BONUS_CATEGORIES}
    assert np.allclose(stro.season_bonus(zeros, np.full(3, 50.0)), 0.0)


def test_bonus_rises_with_the_counts_and_never_exceeds_its_own_payout():
    """A nightly double-double is worth most of 1.5 a game; a fringe line is worth little.

    Bounded rather than pinned: `expected_bonus` integrates a frailty, so even a 25-and-15
    line misses the threshold on a bad night and the exact value is a Monte-Carlo mean.
    """
    gp = np.full(2, 60.0)

    def line(pts: float, reb: float) -> np.ndarray:
        counts = {"pts": np.full((3, 2), pts * 60), "reb": np.full((3, 2), reb * 60),
                  "ast": np.zeros((3, 2)), "stl": np.zeros((3, 2)),
                  "blk": np.zeros((3, 2))}
        return stro.season_bonus(counts, gp)

    star, fringe = line(25.0, 15.0), line(6.0, 2.0)
    assert (star > 0.6 * 1.5 * 60).all() and (star <= 1.5 * 60).all()
    assert (fringe < 0.05 * 1.5 * 60).all()


def _frame(n: int = 8) -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2022-23"] * n,
        "player_id": np.arange(n),
        st.POPULATION_COLUMN: (["veteran"] * (n - 4) + ["lag_recovered"] * 2
                               + ["rookie"] * 2),
        "dk_total": np.linspace(200, 1600, n),
        "dk_bonus_total": np.full(n, 5.0),
        "gp_played": np.full(n, 50.0),
        "team_games": np.full(n, 82.0),
    })


def _totals(n: int = 8, draws: int = 7) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    floor = rng.uniform(300, 1500, size=(draws, n))
    return {"unserved": np.zeros((draws, n)), "floor": floor, "head": floor * 1.05}


def test_oracle_gp_arm_leaves_the_composed_total_untouched():
    """`gp = gp_played` makes the `gp x rate` decomposition the identity, exactly.

    The readout's whole claim about its own oracle row is that it isolates the rate family
    — so if the round trip through `season_total.evaluate`'s decomposition moved the number
    at all, the oracle column would be measuring the trip rather than the family.
    """
    frame, totals = _frame(), _totals()
    specs = stro.treatments(totals, frame, np.full(len(frame), 41.0))
    gp = frame["gp_played"].to_numpy(dtype=float)
    assert np.allclose(specs["oracle_gp_head"]["samples"], totals["head"])
    assert np.allclose(specs["oracle_gp_head"]["gp"], gp)
    assert np.allclose(specs["head"]["gp"], 41.0)


def test_the_unserved_arm_has_no_oracle_twin_and_predicts_zero():
    frame, totals = _frame(), _totals()
    specs = stro.treatments(totals, frame, np.full(len(frame), 41.0))
    assert set(specs) == set(stro.ORDER)
    assert "oracle_gp_unserved" not in specs
    assert np.allclose(specs["unserved"]["rate"], 0.0)


# ── The group tuple `evaluate` gained ─────────────────────────────────────────

def test_population_groups_appear_only_when_the_frame_names_them():
    frame = _frame()
    named = dict(st.metric_groups(frame))
    assert set(named) == {"all", "rotation"} | set(st.POPULATION_GROUPS)
    assert named["rookie"].sum() == 2 and named["lag_recovered"].sum() == 2
    # The three partition the frame, so nothing is scored twice and nothing is dropped.
    stacked = sum(named[g].astype(int) for g in st.POPULATION_GROUPS)
    assert (stacked == 1).all()

    plain = dict(st.metric_groups(frame.drop(columns=[st.POPULATION_COLUMN])))
    assert set(plain) == {"all", "rotation"}


def test_rotation_is_empty_rather_than_an_error_without_the_lag_columns():
    """The rookie-admitting frame is three designs and only one of them is a lag design."""
    assert not st.rotation_mask(_frame()).any()
    lagged = _frame().assign(minutes_per_game_lag1=30.0, gp_share_lag1=0.9)
    assert st.rotation_mask(lagged).all()


def test_evaluate_scores_every_named_population_and_refuses_a_rateless_treatment():
    frame, totals = _frame(), _totals()
    specs = stro.treatments(totals, frame, np.full(len(frame), 41.0))
    rows, predictions = st.evaluate(frame, None, specs, 82)
    table = pd.DataFrame(rows)

    scored = set(table[table["metric"] == "crps_dk_total"]["group"])
    assert {"all", "rookie", "lag_recovered", "veteran"} <= scored
    assert len(predictions) == len(frame) * len(specs)
    n = table[(table["treatment"] == "head") & (table["group"] == "rookie")
              & (table["metric"] == "mae_dk_total")]["n"]
    assert int(n.iloc[0]) == 2

    with pytest.raises(ValueError):
        st.evaluate(frame, None, {"bare": {"gp": np.ones(len(frame)), "pmf": None}}, 82)


def test_a_shared_rate_still_works_for_the_availability_ladder():
    """`evaluate`'s original contract: one rate model held identical across treatments."""
    frame = _frame().drop(columns=[st.POPULATION_COLUMN])
    rate = np.full(len(frame), 20.0)
    rows, _ = st.evaluate(frame, rate, {"full_season": {"gp": np.full(len(frame), 82.0),
                                                        "pmf": None}}, 82)
    table = pd.DataFrame(rows)
    mae = table[(table["group"] == "all") & (table["metric"] == "mae_dk_total")]
    expected = np.abs(82.0 * 20.0 - frame["dk_total"].to_numpy()).mean()
    assert float(mae["value"].iloc[0]) == pytest.approx(expected)
