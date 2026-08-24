"""The rookie heads' variant ladder and §4's gate as code — `docs/rookie-rates-plan.md` §5d.

Nothing here fits a Stan head: the sampler is the expensive half and it is not the half that
can be wrong quietly. What can be wrong quietly is everything around it — a ladder rung that
drifts from the feature list Session 6 will persist, a paired reading taken on rows the two
arms do not share, and a gate that is a conjunction in the doc and a disjunction in the code.
So the ladder is checked against `rookie_rates.head_features` itself, the pairing is checked
by construction, and the gate is exercised on a synthetic metrics table whose every cell is
chosen to sit on one side of the bar.
"""

import numpy as np
import pandas as pd

from src.models import rookie_rates as rr
from src.models import stan_rookie as sr


# ── The variant ladder ────────────────────────────────────────────────────────

def _frame(n: int = 40, seed: int = 0) -> pd.DataFrame:
    """A design-shaped frame: every column `head_features` asks for, plus the targets."""
    rng = np.random.default_rng(seed)
    out = pd.DataFrame({"player_id": np.arange(n), "season": "2018-19"})
    for component in rr.COUNT_HEADS + [m for m, _ in rr.CONVERSION_HEADS]:
        out[rr.shrunk_column(component)] = rng.normal(size=n)
    for col in list(rr.MISSING_AGE_COLS) + rr.SLOT_COLS:
        out[col] = 0.0
    out[rr.YEARS_SINCE_DRAFT] = rng.integers(0, 4, size=n).astype(float)
    for col in rr.SLOT_INTERACTION_COLS:
        out[col] = 0.0
    out["age"] = rng.uniform(19, 25, size=n)
    out["age_sq"] = out["age"] ** 2
    out["on_season_start_roster"] = (np.arange(n) % 2).astype(float)
    out["total_minutes"] = rng.uniform(200, 2000, size=n)
    out["fga"] = rng.integers(50, 800, size=n).astype(float)
    out["fg3a"] = out["fga"] * 0.4
    return out


def test_the_ladder_is_defined_by_subtraction_from_the_shipped_feature_list():
    """`slot_interaction` IS `head_features`, and `linear` is it minus the interaction.

    The rung that ships has to be the list Session 6's design recipe builds, or a head is
    fitted on columns the forward path does not carry.
    """
    frame = _frame()
    arms = sr.variants(frame, frame, "fga")
    assert list(arms) == list(sr.VARIANTS)
    assert arms["slot_interaction"][2] == rr.head_features("fga")
    assert arms["linear"][2] == [c for c in rr.head_features("fga")
                                 if c not in set(rr.SLOT_INTERACTION_COLS)]
    assert set(arms["linear"][2]) < set(arms["slot_interaction"][2])
    assert len(arms["slot_interaction"][2]) - len(arms["linear"][2]) == len(rr.SLOT_COLS)


def test_the_spline_arm_replaces_the_level_and_nests_the_rung_below_it():
    """Curvature on the level, not curvature beside it — the basis takes its place."""
    frame = _frame()
    arms = sr.variants(frame, frame, "reb")
    level = rr.shrunk_column("reb")
    features = arms["slot_interaction_spline"][2]
    assert level not in features
    basis = [c for c in features if c.startswith(f"{level}__s")]
    assert len(basis) > 1
    # Every non-level column of the rung below survives, so the contrast is the basis alone.
    assert ([c for c in features if c not in basis]
            == [c for c in arms["slot_interaction"][2] if c != level])
    assert set(basis) <= set(arms["slot_interaction_spline"][0].columns)


def test_each_head_splines_its_own_level_and_no_other():
    frame = _frame()
    features = sr.variants(frame, frame, "blk")["slot_interaction_spline"][2]
    assert all(rr.shrunk_column("blk") in c for c in features if "__s" in c)
    assert rr.shrunk_column("fga") not in features


# ── The rows the two arms share ───────────────────────────────────────────────

def test_a_conversion_head_scores_only_rows_with_a_realized_attempt():
    """The restriction is applied ONCE by the caller, so the pairing is a fact."""
    frame = _frame()
    frame.loc[:4, "fg3a"] = 0.0
    live = sr.live_rows(frame, "fga")
    assert len(sr.live_rows(frame, None)) == len(frame)
    assert (live["fga"] > 0).all()
    # `fga` is the trials column for the `fg3a|fga` head, and every row has attempts here.
    assert len(live) == len(frame)
    frame.loc[:4, "fga"] = 0.0
    assert len(sr.live_rows(frame, "fga")) == len(frame) - 5
    assert list(sr.live_rows(frame, "fga").index) == list(range(len(frame) - 5))


def test_a_conversion_targets_makes_are_clamped_to_its_attempts():
    """The box score occasionally records more makes than attempts; a beta-binomial cannot."""
    frame = _frame(n=3)
    frame["fga"] = [10.0, 10.0, 10.0]
    frame["fg3a"] = [3.0, 12.0, 0.0]
    y, n = sr.realized(frame, "fg3a", "fga")
    assert list(y) == [3.0, 10.0, 0.0]
    assert list(n) == [10.0, 10.0, 10.0]
    y_count, n_count = sr.realized(frame, "fga", None)
    assert n_count is None and list(y_count) == [10.0, 10.0, 10.0]


def test_the_decision_population_is_the_season_start_roster_flag():
    frame = _frame(n=10)
    masks = sr.population_masks(frame)
    assert masks["all"].all()
    assert masks[sr.DECISION_POPULATION].sum() == 5
    assert sr.DECISION_POPULATION == "draftable"


def test_per_row_crps_averages_to_the_number_the_scorer_reports():
    """The bootstrap resamples the same quantity the level row quotes, or the two disagree."""
    frame = _frame(n=12)
    rng = np.random.default_rng(1)
    samples = rng.normal(400.0, 50.0, size=(200, len(frame)))
    point = samples.mean(axis=0)
    scored = sr.score(frame, "fga", None, point, samples, 10.0)
    assert np.isclose(sr.per_row_crps(frame, "fga", None, samples).mean(), scored["crps"])


def test_scoring_a_population_restricts_the_rows_and_not_the_fit():
    frame = _frame(n=12)
    rng = np.random.default_rng(2)
    samples = rng.normal(400.0, 50.0, size=(200, len(frame)))
    point = samples.mean(axis=0)
    mask = sr.population_masks(frame)[sr.DECISION_POPULATION]
    restricted = sr.score(frame, "fga", None, point, samples, 10.0, mask)
    direct = sr.score(frame[mask].reset_index(drop=True), "fga", None, point[mask],
                      samples[:, mask], 10.0)
    assert np.isclose(restricted["crps"], direct["crps"])
    assert np.isclose(restricted["mae"], direct["mae"])


# ── The rolling origins ───────────────────────────────────────────────────────

def test_an_origin_needs_a_fitting_frame_before_it():
    """A season with fewer than `MIN_FIT_ROWS` rows behind it is not an origin."""
    train = pd.DataFrame({"season_start_year": ([2010] * 100 + [2011] * 100
                                                + [2012] * 100 + [2013] * 100)})
    assert sr.origins(train, min_fit_rows=100) == [2011, 2012, 2013]
    assert sr.origins(train, min_fit_rows=250) == [2013]
    assert sr.origins(train, min_fit_rows=10_000) == []


# ── The gate, as code ─────────────────────────────────────────────────────────

def test_the_selection_criterion_is_the_shipped_sweeps_and_not_crps():
    """`_finalize`'s rule unchanged, so `selected` means the same thing in both artifacts."""
    assert sr.selection_metric("count") == ("val_r2", True)
    assert sr.selection_metric("conversion") == ("val_nll", False)


#: One rung strictly better than the last on both criteria, so `selected` has a unique
#: answer and a test that moves it is moving the gate rather than a tie-break.
_RUNG_R2 = {"linear": 0.80, "slot_interaction": 0.85, "slot_interaction_spline": 0.90}
_RUNG_NLL = {"linear": 4.5, "slot_interaction": 4.2, "slot_interaction_spline": 4.0}


def _gate_rows(head: str, kind: str, **overrides) -> list[dict]:
    """A head's four variants x two populations, all passing unless overridden.

    The floor row sits at a deliberately worse level on both criteria so `beats_floor` has
    something to be true about, and every fitted arm is handed an interval below zero — the
    tests below then move exactly one cell.
    """
    rows = []
    for variant in (sr.FLOOR_VARIANT,) + sr.VARIANTS:
        floor = variant == sr.FLOOR_VARIANT
        for population in ("all", sr.DECISION_POPULATION):
            row = {"head": head, "kind": kind, "variant": variant,
                   "n_features": 0 if floor else 16, "population": population,
                   "n_scored": 100, "val_crps": 30.0 if floor else 20.0,
                   "val_r2": 0.5 if floor else _RUNG_R2[variant],
                   "val_nll": 5.0 if floor else _RUNG_NLL[variant],
                   "crps_vs_floor": 0.0 if floor else -10.0,
                   "crps_vs_floor_lo": 0.0 if floor else -14.0,
                   "crps_vs_floor_hi": 0.0 if floor else -6.0,
                   "verdict_vs_floor": "floor" if floor else "wins",
                   "rolling_crps": 30.0 if floor else 20.0,
                   "rolling_crps_vs_floor": 0.0 if floor else -8.0,
                   "rolling_lo": 0.0 if floor else -12.0,
                   "rolling_hi": 0.0 if floor else -4.0,
                   "origins_won": 0 if floor else 9, "n_origins": 11, "n_rolling": 500,
                   "rolling_verdict": "floor" if floor else "wins"}
            if population == sr.DECISION_POPULATION:
                row.update(overrides.get(variant, {}))
            rows.append(row)
    return rows


def test_both_halves_of_the_conjunction_are_required():
    """Validation alone ships nothing, and the rolling harness alone ships nothing."""
    winner = "slot_interaction_spline"
    passes = sr.finalize(pd.DataFrame(_gate_rows("fga", "count")))
    assert passes[passes["selected"] & (passes["population"] == sr.DECISION_POPULATION)
                  ]["passes"].all()
    assert (passes["ships"] == winner).all()

    # The validation interval reaches across zero; the rolling half is untouched.
    val_fails = sr.finalize(pd.DataFrame(_gate_rows(
        "fga", "count", **{winner: {"crps_vs_floor_hi": +0.4}})))
    row = val_fails[val_fails["selected"]
                    & (val_fails["population"] == sr.DECISION_POPULATION)].iloc[0]
    assert not row["val_pass"] and row["rolling_pass"] and not row["passes"]
    assert (val_fails["ships"] == sr.FLOOR_VARIANT).all()

    # The rolling interval is clear but a minority of origins won.
    roll_fails = sr.finalize(pd.DataFrame(_gate_rows(
        "fga", "count", **{winner: {"origins_won": 5}})))
    row = roll_fails[roll_fails["selected"]
                     & (roll_fails["population"] == sr.DECISION_POPULATION)].iloc[0]
    assert row["val_pass"] and not row["rolling_pass"] and not row["passes"]
    assert (roll_fails["ships"] == sr.FLOOR_VARIANT).all()


def test_a_head_that_fails_ships_its_floor_rather_than_nothing():
    """§5d's ship rule: the gate decides WHICH arm, never whether the head ships."""
    table = sr.finalize(pd.DataFrame(_gate_rows(
        "ftm|fta", "conversion",
        **{v: {"crps_vs_floor_hi": +1.0, "rolling_hi": +1.0} for v in sr.VARIANTS})))
    assert set(table["ships"]) == {sr.FLOOR_VARIANT}
    assert not table["passes"].any()
    # …and it is still a scored head: the floor's own row is present on both populations.
    assert len(table[table["variant"] == sr.FLOOR_VARIANT]) == 2


def test_the_verdict_is_read_on_the_draftable_population_alone():
    """A head winning on `all` and failing on the rows a board prices ships the floor."""
    rows = _gate_rows("reb", "count",
                      **{v: {"crps_vs_floor_hi": +2.0} for v in sr.VARIANTS})
    table = sr.finalize(pd.DataFrame(rows))
    assert (table["ships"] == sr.FLOOR_VARIANT).all()
    everything = table[table["population"] == "all"]
    # The `all` rows still say the arm won there — the gate ignored them, it did not edit
    # them, which is what makes the gap between the two populations readable.
    assert (everything[everything["variant"] != sr.FLOOR_VARIANT]["crps_vs_floor_hi"]
            < 0).all()


def test_one_variant_per_head_is_selected_and_it_is_carried_onto_both_populations():
    table = sr.finalize(pd.DataFrame(_gate_rows("ast", "count")))
    for population, part in table.groupby("population"):
        assert part["selected"].sum() == 1
    assert set(table[table["selected"]]["variant"]) == {"slot_interaction_spline"}
    assert not table[table["variant"] == sr.FLOOR_VARIANT]["selected"].any()


def test_beats_floor_is_the_shipped_sweeps_point_comparison_and_not_the_gate():
    """A head can improve R2 and still fail a CRPS interval — the two columns disagree."""
    table = sr.finalize(pd.DataFrame(_gate_rows(
        "blk", "count", **{v: {"crps_vs_floor_hi": +1.0} for v in sr.VARIANTS})))
    decision = table[table["population"] == sr.DECISION_POPULATION]
    fitted = decision[decision["variant"] != sr.FLOOR_VARIANT]
    assert fitted["beats_floor"].all()
    assert not fitted["passes"].any()
    assert decision[decision["variant"] == sr.FLOOR_VARIANT]["beats_floor"].all()


def test_two_heads_are_gated_independently():
    """Head-local, which is what makes `merge_heads` safe for a partial refit."""
    rows = (_gate_rows("fga", "count")
            + _gate_rows("stl", "count",
                         **{v: {"rolling_hi": +1.0} for v in sr.VARIANTS}))
    table = sr.finalize(pd.DataFrame(rows))
    assert set(table[table["head"] == "fga"]["ships"]) == {"slot_interaction_spline"}
    assert set(table[table["head"] == "stl"]["ships"]) == {sr.FLOOR_VARIANT}
