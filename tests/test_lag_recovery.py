"""The lag-recovery split — `src/models/lag_recovery.py`.

The module exists to move a population boundary, so what the tests pin is the boundary
itself: which group a row lands in, and that the groups mean what
`component_rates.build_design` means by "has a row".
"""

import numpy as np
import pandas as pd

from src.models import component_rates as CR
from src.models import lag_recovery as LR


def _rows(specs: list[dict]) -> pd.DataFrame:
    """One row per spec, with every lag column the classifier reads."""
    base = {f"total_minutes_lag{i}": np.nan for i in (1, 2, 3)}
    frame = pd.DataFrame([{**base, "season": "2020-21", "first_season": "2010-11",
                           "player_id": i, "age": 25.0, "mpg_lag1": 20.0, **spec}
                          for i, spec in enumerate(specs)])
    return frame.assign(group=LR.classify(frame))


def test_the_six_groups_partition_the_population_and_a_rookie_wins_every_tie():
    """A true rookie is decided by his FIRST SEASON, not by absent lags.

    The ordering matters: a player whose prior seasons fall outside the configured window
    has no lag columns either, and labelling him a rookie would hand the rookie head a row
    it cannot serve — he has an NBA history, the window just does not reach it.
    """
    frame = _rows([
        {"total_minutes_lag1": 1500.0},                       # veteran
        {"total_minutes_lag1": 120.0},                        # thin lag-1
        {"total_minutes_lag2": 1500.0},                       # returnee, usable lag-2
        {"total_minutes_lag2": 90.0},                         # returnee, thin lag-2
        {"total_minutes_lag3": 1500.0},                       # away two seasons
        {"first_season": "2020-21"},                          # true rookie
        {"first_season": "2020-21", "total_minutes_lag1": 1500.0},   # rookie wins the tie
    ])
    assert list(frame["group"]) == ["veteran", "thin_prior", "returnee_lag2",
                                    "returnee_thin", "no_usable_lag", "true_rookie",
                                    "true_rookie"]
    assert set(frame["group"]) <= set(LR.GROUPS)
    assert "true_rookie" not in LR.HAS_HISTORY


def test_the_veteran_group_is_exactly_what_build_design_qualifies():
    """`served_today` in the census has to mean what the shipped design means by a row.

    The census reconciles against the drafting layer — 347 and 359 served on the two
    validation boards — and that reconciliation is only meaningful if the two definitions
    are the same one. `build_design` drops NaN `age`/`mpg_lag1` and then applies
    `MIN_PRIOR_MINUTES`; this pins both halves.
    """
    frame = _rows([
        {"total_minutes_lag1": float(CR.MIN_PRIOR_MINUTES)},        # exactly at it: in
        {"total_minutes_lag1": CR.MIN_PRIOR_MINUTES - 1.0},         # one under: out
        {"total_minutes_lag1": 1500.0, "age": np.nan},              # no age: out
        {"total_minutes_lag1": 1500.0, "mpg_lag1": np.nan},         # no lag mpg: out
    ])
    assert list(LR.scorable_now(frame)) == [True, False, False, False]
    assert list(frame["group"]) == ["veteran", "thin_prior", "veteran", "veteran"]


def test_the_unified_rung_reads_the_nearest_usable_season():
    frame = _rows([
        {"total_minutes_lag1": 900.0, "total_minutes_lag2": 800.0},
        {"total_minutes_lag2": 800.0, "total_minutes_lag3": 700.0},
        {"total_minutes_lag3": 700.0},
        {},
    ])
    assert list(LR.recent_lag(frame)) == [1, 2, 3, 0]


def test_shrinkage_recovers_the_raw_rate_at_k_zero_and_the_mean_when_swamped():
    """Both ends exactly, because the grid brackets `k = 0` and the arm has to be able to
    select 'do not shrink' — the thin-prior raw arm is an anti-model and the fitted `k`
    is what separates the two."""
    frame = _rows([{"total_minutes_lag1": 100.0}])
    frame["reb_p36_lag1"] = 8.0
    assert LR.shrunk_rate(frame, "reb", 0.0, 5.0)[0] == 8.0
    assert abs(LR.shrunk_rate(frame, "reb", 1e9, 5.0)[0] - 5.0) < 1e-6
    half = LR.shrunk_rate(frame, "reb", 100.0, 5.0)[0]
    assert abs(half - 6.5) < 1e-9                     # w = 100/200 = 0.5


def test_a_row_with_no_usable_lag_falls_back_to_the_population_mean_not_nan():
    """The rung has to be total over the population it serves: a NaN rate would propagate
    into a count prediction and silently drop the row from any score it enters."""
    frame = _rows([{}])
    frame["reb_p36_lag1"] = np.nan
    out = LR.recent_shrunk_rate(frame, "reb", 100.0, 5.0)
    assert np.isfinite(out).all() and out[0] == 5.0
