import numpy as np
import pandas as pd

from src.eda.residual_correlation import (
    COMPONENTS,
    MINUTES_CONDITIONED,
    RAW,
    SUBSTITUTION_PAIR,
    conditioned_series,
    correlation_long,
    measure,
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

    df = pd.DataFrame({"min": minutes, "fg2a": fg2a, "fg3a": fg3a})
    df["fta"] = rng.poisson(minutes * 0.12).astype(float)
    for c, per_min in (("reb", 0.15), ("ast", 0.10), ("stl", 0.03),
                       ("blk", 0.02), ("tov", 0.06)):
        df[c] = rng.poisson(minutes * per_min).astype(float)
    for made, att in CONVERSIONS:
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


def test_the_substitution_cell_is_negative_when_threes_displace_twos():
    table = measure(_targets(n=1200, substitute=True))
    s = summarize(table, MINUTES_CONDITIONED)
    assert s["substitution_r"] < 0


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
    df.loc[df.index[:300], ["fg3a", "fg3m"]] = 0.0
    table = correlation_long(conditioned_series(df), MINUTES_CONDITIONED)
    with_fg3a = table[(table.component_a == "fg3a") & (table.component_b == "reb")]
    reb_only = table[(table.component_a == "reb") & (table.component_b == "ast")]
    assert with_fg3a["n_games"].iloc[0] < reb_only["n_games"].iloc[0]


def test_conversion_raw_series_is_a_rate_and_ignores_games_with_no_attempts():
    df = _targets(n=200)
    df.loc[df.index[:50], ["fta", "ftm"]] = 0.0
    series = {name: (resid, valid) for name, _, resid, valid in raw_series(df)}
    resid, valid = series["ftm|fta"]
    assert not valid[:50].any()
    assert ((resid[valid] >= 0) & (resid[valid] <= 1)).all()
