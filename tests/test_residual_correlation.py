import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import FIT_WINDOWS, FULL_WINDOW, TRAIN_VAL_WINDOW
from src.eda.residual_correlation import (
    COMPONENTS,
    LEGACY_BASIS,
    LEGACY_SUBSTITUTION_PAIR,
    MINUTES_CONDITIONED,
    RAW,
    SUBSTITUTION_PAIR,
    conditioned_series,
    correlation_long,
    measure,
    measure_windows,
    raw_series,
    summarize,
    to_matrix,
)
from src.eda.serial_correlation import COUNT_COMPONENTS, CONVERSIONS


# ── Synthetic builders ────────────────────────────────────────────────────────

def _targets(n: int = 600, n_seasons: int = 6, seed: int = 0,
             substitute: bool = False) -> pd.DataFrame:
    """A component-targets-shaped frame carrying every column the eleven heads need.

    `substitute` makes 3PA displace 2PA within a game — the same shot taken from a
    different place — which is the dependence the reparameterization exists to handle and
    the one the matrix must show as negative.
    """
    rng = np.random.default_rng(seed)
    minutes = rng.uniform(12.0, 38.0, n)
    shots = rng.poisson(minutes * 0.45)
    if substitute:
        share = rng.beta(2.0, 4.0, n)
        fg3a = rng.binomial(shots, share).astype(float)
    else:
        fg3a = rng.poisson(minutes * 0.15).astype(float)
    fg2a = np.clip(shots - fg3a, 0, None).astype(float) if substitute else \
        rng.poisson(minutes * 0.30).astype(float)

    # `fga` is the modelled count under the shot-attempt basis and `fg2a` the derived
    # remainder, but both are real columns of `component_targets.parquet`, so the fixture
    # carries the identity rather than choosing a side.
    df = pd.DataFrame({"min": minutes, "fg2a": fg2a, "fg3a": fg3a, "fga": fg2a + fg3a})
    df["fta"] = rng.poisson(minutes * 0.12).astype(float)
    for c, per_min in (("reb", 0.15), ("ast", 0.10), ("stl", 0.03),
                       ("blk", 0.02), ("tov", 0.06)):
        df[c] = rng.poisson(minutes * per_min).astype(float)
    # `fg3a | fga` is already realized above; only the true makes are drawn here.
    for made, att in CONVERSIONS:
        if made in df.columns:
            continue
        df[made] = rng.binomial(df[att].astype(int), 0.45).astype(float)

    df["played"] = 1
    per = max(n // n_seasons, 1)
    df["player_id"] = np.minimum(np.arange(n) // per, n_seasons - 1)
    df["season"] = "2021-22"
    df["ps"] = df["player_id"]
    df["game_index"] = df.groupby("ps").cumcount()
    return df


# ── The component set ─────────────────────────────────────────────────────────

def test_eleven_non_minutes_components_and_no_minutes_row():
    assert len(COMPONENTS) == 11
    assert "min" not in COMPONENTS
    assert set(COUNT_COMPONENTS) <= set(COMPONENTS)
    assert all(f"{m}|{a}" in COMPONENTS for m, a in CONVERSIONS)


# ── The verification the plan asks for ────────────────────────────────────────

def test_diagonal_is_exactly_one_and_the_matrix_is_symmetric():
    table = measure(_targets())
    for basis in (MINUTES_CONDITIONED, RAW):
        R = to_matrix(table, basis)
        assert np.array_equal(np.diag(R.to_numpy()), np.ones(len(R)))
        assert np.allclose(R.to_numpy(), R.to_numpy().T, equal_nan=True)


def test_long_form_carries_both_ordered_pairs_so_a_plain_pivot_is_square():
    table = correlation_long(conditioned_series(_targets()), MINUTES_CONDITIONED)
    assert len(table) == len(COMPONENTS) ** 2
    a, b = SUBSTITUTION_PAIR
    fwd = table[(table.component_a == a) & (table.component_b == b)]["r"].iloc[0]
    rev = table[(table.component_a == b) & (table.component_b == a)]["r"].iloc[0]
    assert fwd == rev


def test_the_legacy_substitution_cell_is_negative_when_threes_displace_twos():
    """The coupling the old two-count basis had to carry in the copula. It is still real
    in the data — that is the point — but under the shot-attempt basis it is removed by
    construction, so it survives as a labelled contrast rather than as a matrix cell."""
    table = measure(_targets(n=1200, substitute=True))
    s = summarize(table, MINUTES_CONDITIONED)
    assert s["legacy_substitution_r"] < 0


def test_the_derived_count_is_not_among_the_modelled_components():
    """`fg2a` is `fga - fg3a` under the shot-attempt basis. If it reappeared in
    `COMPONENTS` the copula would be carrying a cell the basis exists to remove, and the
    eleven-head count would silently become twelve."""
    assert "fg2a" not in COMPONENTS
    assert "fga" in COMPONENTS and "fg3a|fga" in COMPONENTS


def test_the_legacy_pair_stays_out_of_the_matrix():
    """It lives under its own basis label precisely so `to_matrix` can stay strict."""
    table = measure(_targets(n=400))
    legacy = table[table.basis == LEGACY_BASIS]
    assert len(legacy) and set(legacy["component_a"]) == set(LEGACY_SUBSTITUTION_PAIR)
    assert list(to_matrix(table, MINUTES_CONDITIONED).index) == COMPONENTS


def test_to_matrix_raises_rather_than_silently_dropping_a_missing_head():
    """It used to filter with `[c for c in COMPONENTS if c in wide.index]`, so a head-list
    change quietly shrank the copula instead of failing. A matrix missing a head is not a
    smaller copula, it is a wrong one."""
    table = measure(_targets(n=300))
    trimmed = table[~table["component_a"].eq("blk") & ~table["component_b"].eq("blk")]
    with pytest.raises(ValueError, match="does not match COMPONENTS"):
        to_matrix(trimmed, MINUTES_CONDITIONED)


def test_minutes_conditioning_flag_tracks_the_basis():
    table = measure(_targets())
    assert table[table.basis == MINUTES_CONDITIONED]["minutes_conditioned"].all()
    assert not table[table.basis == RAW]["minutes_conditioned"].any()


def test_raw_correlations_are_larger_because_minutes_are_still_in_them():
    """The trap the `minutes_conditioned` column exists to prevent."""
    table = measure(_targets(n=1200))
    cond = summarize(table, MINUTES_CONDITIONED)
    raw = summarize(table, RAW)
    assert raw["off_diagonal_mean"] > cond["off_diagonal_mean"]


def test_independent_components_leave_a_near_zero_conditioned_matrix():
    """Independent Poissons at a shared exposure must not look coupled once minutes go."""
    table = measure(_targets(n=1800, seed=7))
    s = summarize(table, MINUTES_CONDITIONED)
    assert abs(s["off_diagonal_mean"]) < 0.05
    assert s["min_eigenvalue"] > 0


def test_n_games_is_pairwise_complete_not_the_frame_size():
    df = _targets(n=600)
    # Blot out every three-point attempt in the first half: those rows carry no evidence
    # about three-point volume, so any pair involving fg3a must be counted on fewer rows.
    df.loc[df.index[:300], ["fga", "fg2a", "fg3a", "fg2m", "fg3m"]] = 0.0
    table = correlation_long(conditioned_series(df), MINUTES_CONDITIONED)
    # `fg3a | fga` is undefined on a game with no field-goal attempts, so every pair
    # involving it is counted on strictly fewer rows than a pair of plain counts.
    with_share = table[(table.component_a == "fg3a|fga") & (table.component_b == "reb")]
    reb_only = table[(table.component_a == "reb") & (table.component_b == "ast")]
    assert with_share["n_games"].iloc[0] < reb_only["n_games"].iloc[0]


def test_conversion_raw_series_is_a_rate_and_ignores_games_with_no_attempts():
    df = _targets(n=200)
    df.loc[df.index[:50], ["fta", "ftm"]] = 0.0
    series = {name: (resid, valid) for name, _, resid, valid in raw_series(df)}
    resid, valid = series["ftm|fta"]
    assert not valid[:50].any()
    assert ((resid[valid] >= 0) & (resid[valid] <= 1)).all()


# ── The fit window: the copula is an input, so its calibration window matters ──

def _multi_season(n: int = 3000, seasons: int = 5, seed: int = 3) -> pd.DataFrame:
    """`_targets` stamps one season label; the window axis needs several.

    Seasons are assigned per player-season block so that `ps` never spans two of them,
    which is the invariant every residual helper keys on.
    """
    df = _targets(n=n, n_seasons=seasons * 4, seed=seed)
    labels = [f"20{10 + i:02d}-{11 + i:02d}" for i in range(seasons)]
    df["season"] = [labels[p % seasons] for p in df["player_id"]]
    key = df["player_id"].astype(str) + "_" + df["season"]
    df["ps"] = pd.factorize(key)[0]
    df["game_index"] = df.groupby("ps").cumcount()
    return df


def test_measure_windows_emits_both_windows_and_train_val_excludes_the_test_seasons():
    df = _multi_season()
    table = measure_windows(df)
    assert set(table["fit_window"]) == set(FIT_WINDOWS)
    # Same schema in both windows — a consumer switching window must not lose a pair.
    per = table.groupby("fit_window").size()
    assert per[FULL_WINDOW] == per[TRAIN_VAL_WINDOW]
    # And the train_val arm really is measured on fewer games.
    full_n = table[(table.fit_window == FULL_WINDOW)]["n_games"].max()
    tv_n = table[(table.fit_window == TRAIN_VAL_WINDOW)]["n_games"].max()
    assert tv_n < full_n


def test_to_matrix_defaults_to_the_train_val_window_not_the_full_one():
    """The default is what an unthinking consumer gets, so it has to be the safe one.

    A simulator handed the full-window copula is calibrated on the seasons it is later
    backtested against, and nothing in its output would say so.
    """
    table = measure_windows(_multi_season())
    default = to_matrix(table, MINUTES_CONDITIONED)
    train_val = to_matrix(table, MINUTES_CONDITIONED, TRAIN_VAL_WINDOW)
    full = to_matrix(table, MINUTES_CONDITIONED, FULL_WINDOW)
    assert np.allclose(default.to_numpy(), train_val.to_numpy(), equal_nan=True)
    assert not np.allclose(default.to_numpy(), full.to_numpy(), equal_nan=True)


def test_to_matrix_still_pivots_a_frame_that_predates_the_fit_window_axis():
    """`measure` has no window column; older artifacts on disk have none either."""
    table = measure(_targets(n=600))
    assert "fit_window" not in table.columns
    assert list(to_matrix(table, MINUTES_CONDITIONED).index) == COMPONENTS


def test_to_matrix_raises_on_a_window_the_artifact_does_not_carry():
    table = measure_windows(_multi_season(), windows=[FULL_WINDOW])
    with pytest.raises(ValueError, match="no rows for fit_window"):
        to_matrix(table, MINUTES_CONDITIONED, TRAIN_VAL_WINDOW)


def test_summarize_reports_the_window_it_measured():
    table = measure_windows(_multi_season())
    for window in FIT_WINDOWS:
        assert summarize(table, MINUTES_CONDITIONED, window)["fit_window"] == window
