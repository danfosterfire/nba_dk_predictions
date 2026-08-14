"""Tests for session 4b's fit — the blended offset under the posterior.

The screen (`tests/test_composition_preseason.py`) pins the blend rule itself. What this
round adds is a **comparison across two estimators**, and every way it can be wrong is a way
the two sides stop being comparable rather than a way a number comes out wrong:

1. **The two arms hold the same rows.** `route = offset_only` leaves the allocation order at
   the incumbent, which is what makes the paired margin legal at all. If a future change
   moved the ordering, `paired` would raise — but only if the two arms are actually built
   from the same window and split, which is what `build_arms` owes.
2. **The floor and the fitted arm are scored by the same functional.** `crps_series` takes
   draws rather than a `(train, val)` pair for exactly this reason; a refactor that gave the
   floor its own scorer would make "the increment shrank under the posterior" a statement
   about two estimators.
3. **The retention is a ratio of the two increments and nothing else.** It is the number the
   round exists to produce and it is assembled from the margin table by string keys.
4. **The gate is code, not prose.** `report` must fail an arm whose interval spans zero and
   an arm that breaks the team constraint, independently.
5. **A partial re-run does not destroy the other arm.** `_flush` merges by identity, the
   rule `composition_effects` learned the hard way.
6. **The configured warmup clears the `dense_e` cliff.** Not a style point: at this arm's
   parameter count the threshold is 600 warmup draws and `diag_e` is ~10x the wall clock, so
   a config that drops below it turns a half-hour round into most of a day with nothing
   raising.
"""

import numpy as np
import pandas as pd
import yaml

from src.models.composition_preseason import crps_series
from src.models.composition_preseason_fit import (ARMS, BASE_VARIANT, COMPARISONS, ROUTE,
                                                  SELECTED_K, _cell, _draftable, _flush,
                                                  announce_metric, arm_rows, margins,
                                                  report, retention)


def _val(n_games: int = 3, n_players: int = 4) -> pd.DataFrame:
    """A frame with the columns `arm_rows` and `crps_series` read, in team-game blocks."""
    rows = []
    for g in range(n_games):
        for k in range(n_players):
            rows.append({"player_id": 10 + k, "season": "2022-23", "game_id": f"00{g}",
                         "team_id": 1, "position": k, "k_players": n_players,
                         "y": 10 + k, "N": 10 * n_players + sum(range(n_players)),
                         "U": 48, "m": 48, "lo": 0,
                         "is_last": int(k == n_players - 1), "stick_ratio": 0.25,
                         "logit_prior": 0.0})
    return pd.DataFrame(rows)


def _samples(val: pd.DataFrame, draws: int, shift: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = val["y"].to_numpy(float)[None, :] + shift
    return base + rng.normal(0.0, 1.0, size=(draws, len(val)))


def test_the_declared_arm_is_the_shipped_variant_on_the_offset_route_at_the_inner_k():
    """Three constants the round is not free to re-decide: `k = 80` came off an inner carve
    of the fitting half (validation prefers 160), the ordering route carried 3.75% of the
    screen's margin and is the expensive half, and the ladder is not re-run here."""
    assert SELECTED_K == 80.0
    assert ROUTE == "offset_only"
    assert BASE_VARIANT == "betabinom_ot_graded"
    assert ARMS == ("base", "preseason")


def test_the_floor_and_the_fitted_arm_go_through_one_scorer():
    """`crps_series` takes DRAWS, so the two sides of the retention are the same functional
    of whatever predictive they are given. Handing it two draw sets from the same frame must
    return series on the same index."""
    val = _val()
    a = crps_series(_samples(val, 16, 0.0), val)
    b = crps_series(_samples(val, 16, 2.0, seed=1), val)
    for left, right in zip(a, b):
        assert left.index.equals(right.index)
    assert (b[0] > a[0]).all()          # the shifted draws are worse on every row


def test_the_draftable_mask_is_the_season_start_roster_and_nothing_else():
    val = _val()
    roster = {("2022-23", 10), ("2022-23", 12)}
    mask = _draftable(val, roster)
    assert mask.sum() == len(val) / 2
    assert set(val.loc[mask, "player_id"]) == {10, 12}


def test_arm_rows_reports_both_units_and_both_populations():
    """Four cells per arm, and the team metric only on the pooled player-game one — a
    draftable subset of a team-game is not a team-game, so asking for the team constraint on
    it is asking a question with no referent."""
    val = _val()
    roster = {("2022-23", 10), ("2022-23", 12)}
    rows = arm_rows(_samples(val, 16, 0.0), val, roster, "base", "fitted")
    assert {(r["unit"], r["population"]) for r in rows} == {
        ("player_game", "pooled"), ("player_game", "draftable"),
        ("player_season", "pooled"), ("player_season", "draftable")}
    assert "team_sum_abs_error" in _cell(rows, "player_game", "pooled")
    assert "team_sum_abs_error" not in _cell(rows, "player_game", "draftable")
    assert "predictive_sd" in _cell(rows, "player_season", "pooled")


def test_the_retention_is_the_fitted_increment_over_the_floors():
    """The round's headline number, and it is assembled from the margin table by string
    keys — a renamed comparison must not silently produce a NaN that reads as 'no result'."""
    frame = pd.DataFrame([
        {"comparison": "fitted_increment", "unit": "player_game",
         "population": "draftable", "crps_delta": -0.10},
        {"comparison": "floor_increment", "unit": "player_game",
         "population": "draftable", "crps_delta": -0.20},
    ])
    rows = {(r["unit"], r["population"]): r for r in retention(frame)}
    assert rows[("player_game", "draftable")]["retention"] == 0.5
    assert np.isnan(rows[("player_season", "pooled")]["retention"])
    assert {c[0] for c in COMPARISONS} >= {"fitted_increment", "floor_increment"}


def test_margins_pair_every_comparison_at_both_units_and_populations():
    val = _val()
    roster = {("2022-23", 10), ("2022-23", 12)}
    series = {"base": crps_series(_samples(val, 16, 1.0), val),
              "preseason": crps_series(_samples(val, 16, 0.0), val),
              "floor_base": crps_series(_samples(val, 16, 2.0), val),
              "floor_preseason": crps_series(_samples(val, 16, 1.5), val)}
    frame = pd.DataFrame(margins(series, roster, SELECTED_K, ROUTE))
    assert len(frame) == len(COMPARISONS) * 2 * 2
    fitted = frame[(frame["comparison"] == "fitted_increment")
                   & (frame["unit"] == "player_game")
                   & (frame["population"] == "draftable")]
    assert float(fitted["crps_delta"].iloc[0]) < 0     # `preseason` is the better arm here
    assert float(fitted["ci_lo"].iloc[0]) <= float(fitted["crps_delta"].iloc[0]) \
        <= float(fitted["ci_hi"].iloc[0])


def test_margins_skip_a_comparison_whose_arm_never_landed():
    """A run cut short after `base` must still produce that arm's rows rather than raising —
    the checkpointing is worthless if the tail of `run` cannot survive a missing arm."""
    val = _val()
    series = {"base": crps_series(_samples(val, 8, 0.0), val),
              "floor_base": crps_series(_samples(val, 8, 1.0), val)}
    frame = pd.DataFrame(margins(series, {("2022-23", 10)}, SELECTED_K, ROUTE))
    assert set(frame["comparison"]) == {"fit_value_base"}


def _gate_frames(hi: float, team_error: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    margin = pd.DataFrame([{"comparison": "fitted_increment", "arm": "preseason",
                            "reference": "base", "unit": "player_game",
                            "population": "draftable", "n": 10,
                            "crps_delta": -0.2, "ci_lo": -0.3, "ci_hi": hi}])
    arms = pd.DataFrame([{"arm": a, "kind": "fitted", "unit": "player_game",
                          "population": "draftable" if p else "pooled", "n": 10,
                          "crps": 4.0, "r2": 0.4, "mae": 6.0, "bias": 0.0, "pit_ks": 0.05,
                          "predictive_sd": np.nan,
                          "team_sum_abs_error": np.nan if p else team_error}
                         for a in ARMS for p in (False, True)])
    return arms, margin


def test_the_gate_fails_an_interval_that_spans_zero():
    arms, margin = _gate_frames(hi=+0.05, team_error=0.0)
    assert report(arms, margin, pd.DataFrame()) is False


def test_the_gate_fails_a_broken_team_constraint_even_when_crps_wins():
    """The two halves are independent: an arm that improves CRPS by no longer summing to the
    team total has given away the one thing this head exists for."""
    arms, margin = _gate_frames(hi=-0.05, team_error=1.5)
    assert report(arms, margin, pd.DataFrame()) is False
    arms_ok, margin_ok = _gate_frames(hi=-0.05, team_error=0.0)
    assert report(arms_ok, margin_ok, pd.DataFrame()) is True


def test_the_configured_warmup_keeps_this_arm_on_the_dense_metric():
    """The cost cliff that cost this round a 32-minute false start.

    `choose_metric` grants `dense_e` only when `warmup >= 20 x parameters`; the declared arm
    is 25 features + intercept + 4 dispersion bins = 30, so the threshold is exactly 600.
    `warmup: 500` — the value the `effects` block uses and the one this round copied — falls
    one step under it and drops the head onto `diag_e`, which `stan_composition`'s own probe
    measured at treedepth 8-9 against 4, roughly 10x the wall clock. Nothing raised: the
    metric only reaches the artifact after the fit it decides the cost of, CmdStan writes no
    draw until warmup ends, and its progress lines are buffered away.
    """
    cfg = yaml.safe_load(open("configs/default.yaml"))
    pre = cfg["stan"]["composition"]["preseason"]
    assert announce_metric(25, 4, int(pre["warmup"])) == "dense_e"
    assert announce_metric(25, 4, 500) == "diag_e"       # what the cliff looks like
    assert announce_metric(25, 4, 600) == "dense_e"      # and where it sits, exactly


def test_flush_merges_by_identity_so_a_partial_rerun_keeps_the_other_arm(tmp_path):
    """`composition_effects._flush`'s rule: the arms are independent and each is a quarter
    of an hour, so re-running one must leave the other in place."""
    dest = tmp_path / "arms.csv"
    keys = ("arm", "unit", "population")
    base = [{"arm": "base", "unit": "player_game", "population": "pooled", "crps": 4.5}]
    pres = [{"arm": "preseason", "unit": "player_game", "population": "pooled",
             "crps": 4.3}]
    _flush(dest, base, keys)
    _flush(dest, pres, keys)
    assert set(pd.read_csv(dest)["arm"]) == {"base", "preseason"}

    _flush(dest, [{**base[0], "crps": 4.4}], keys)
    out = pd.read_csv(dest)
    assert len(out) == 2                                    # replaced, not appended
    assert float(out.loc[out["arm"] == "base", "crps"].iloc[0]) == 4.4
    assert float(out.loc[out["arm"] == "preseason", "crps"].iloc[0]) == 4.3
