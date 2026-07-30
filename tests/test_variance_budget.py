import numpy as np
import pandas as pd

from src.eda.variance_budget import (
    RESIDUAL,
    TOTAL,
    identity_rows,
    measure,
    minutes_rows,
    opponent_main_rows,
    opponent_rows,
    spread_rows,
)
from src.features.opponent import within_player_season_residuals


# ── Synthetic builders ────────────────────────────────────────────────────────

def _games(mpg: dict[int, float], games_each: int = 120, slope: float = 1.0,
           noise: float = 0.0, seed: int = 0, n_opponents: int = 6,
           n_archetypes: int = 3) -> pd.DataFrame:
    """Player-games where dk_pts is `slope x minutes` plus optional noise.

    `mpg` maps a player id to his average minutes, so two players can share a slope while
    sitting at different levels — which is the configuration that separates the two
    "own minutes" constructions.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for pid, avg in mpg.items():
        minutes = np.clip(rng.normal(avg, 6.0, games_each), 1.0, 47.0)
        dk = slope * minutes + rng.normal(0.0, noise, games_each)
        for i, (m, d) in enumerate(zip(minutes, dk)):
            rows.append({"player_id": pid, "season": "2021-22", "min": float(m),
                         "dk_pts": float(d),
                         "opponent_team_id": i % n_opponents,
                         "is_home": float(i % 2),
                         "archetype": f"arch{pid % n_archetypes}"})
    return _with_residuals(pd.DataFrame(rows))


def _with_residuals(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the within-player-season residuals the budget rows are defined on."""
    df = df.reset_index(drop=True)
    resid = within_player_season_residuals(df, ["dk_pts", "min"])
    df["resid_dk_pts"] = resid["dk_pts"].to_numpy(dtype=float)
    df["resid_min"] = resid["min"].to_numpy(dtype=float)
    return df


def _by_source(rows: list[dict]) -> dict[str, dict]:
    return {r["source"]: r for r in rows}


# ── The two denominators ──────────────────────────────────────────────────────

def test_identity_is_quoted_against_total_and_everything_else_against_the_residual():
    df = _games({1: 16.0, 2: 34.0}, noise=3.0)
    rows = identity_rows(df, "x") + minutes_rows(df, "x") + opponent_main_rows(df, "x")
    for r in rows:
        expected = TOTAL if r["source"].startswith("player_season") else RESIDUAL
        assert r["basis"] == expected, r["source"]


def test_identity_takes_nearly_all_the_variance_when_players_never_overlap():
    """Two players with disjoint dk_pts ranges: identity is the whole story."""
    df = _games({1: 6.0, 2: 44.0}, noise=0.5)
    got = _by_source(identity_rows(df, "x"))["player_season_identity"]
    assert got["share_of_variance"] > 0.9


def test_identity_is_near_zero_when_every_player_has_the_same_distribution():
    df = _games({1: 25.0, 2: 25.0, 3: 25.0}, noise=5.0)
    got = _by_source(identity_rows(df, "x"))["player_season_identity"]
    assert got["share_of_variance"] < 0.05


# ── The minutes correction ────────────────────────────────────────────────────

def test_own_minutes_recovers_a_deterministic_minutes_relationship():
    """dk_pts IS minutes here, so the deviation basis must read ~100%."""
    df = _games({1: 16.0, 2: 34.0}, slope=1.0, noise=0.0)
    rows = _by_source(minutes_rows(df, "x"))
    assert rows["own_minutes"]["share_of_variance"] > 0.99
    assert rows["own_minutes_nonparametric"]["share_of_variance"] > 0.95


def test_the_raw_level_construction_is_attenuated_and_that_is_the_correction():
    """The recorded 18.6% construction, shown failing on data with a known answer.

    Every player gains exactly 1 dk_pt per minute, so own minutes explains 100% of the
    within-player-season residual. Conditioning on the raw minutes *level* pools ten players
    sitting between 10 and 37 mpg, whose residuals at any given minutes count have opposite
    signs and cancel — so the same data reads under half as much.
    """
    df = _games({i + 1: 10.0 + 3 * i for i in range(10)}, slope=1.0, noise=0.0)
    rows = _by_source(minutes_rows(df, "x"))
    deviation = rows["own_minutes"]["share_of_variance"]
    raw_level = rows["own_minutes_raw_level"]["share_of_variance"]
    assert deviation > 0.99
    assert raw_level < 0.5 * deviation
    assert "SUPERSEDED" in rows["own_minutes_raw_level"]["note"]


def test_the_two_constructions_agree_when_every_player_sits_at_the_same_level():
    """The confound is the level spread, so removing it must remove the gap."""
    df = _games({1: 25.0, 2: 25.0, 3: 25.0}, slope=1.0, noise=0.0)
    rows = _by_source(minutes_rows(df, "x"))
    deviation = rows["own_minutes"]["share_of_variance"]
    raw_level = rows["own_minutes_raw_level"]["share_of_variance"]
    assert abs(deviation - raw_level) < 0.1


def test_the_saturated_row_bounds_the_common_slope_row():
    df = _games({1: 16.0, 2: 34.0}, slope=1.0, noise=4.0)
    rows = _by_source(minutes_rows(df, "x"))
    assert (rows["own_minutes"]["share_of_variance"]
            <= rows["own_minutes_saturated"]["share_of_variance"] + 1e-9)


def test_the_slope_rows_expose_the_per_player_heterogeneity():
    """A steeper high-minutes player must show up as a steeper top-tier slope."""
    df = _with_residuals(pd.concat([_games({1: 10.0}, slope=0.5),
                                    _games({2: 36.0}, slope=1.5)], ignore_index=True))
    rows = _by_source(minutes_rows(df, "x"))
    assert rows["minutes_slope_top_mpg_tier"]["units"] == "dk_pts_per_minute"
    assert (rows["minutes_slope_top_mpg_tier"]["value"]
            > rows["minutes_slope_bottom_mpg_tier"]["value"])


# ── Nulls ─────────────────────────────────────────────────────────────────────

def test_both_null_constructions_are_reported_and_named():
    df = _games({1: 16.0, 2: 26.0, 3: 34.0}, games_each=80, noise=6.0)
    rows = opponent_rows(df, "x", n_shuffles=3, seed=0)
    named = {r["null_construction"] for r in rows
             if r["source"].endswith("_above_null")}
    assert named == {"shuffle_opponent_within_season", "shuffle_archetype_within_season"}
    # No row may report an "above null" figure without saying which marginal moved.
    for r in rows:
        if "above_null" in r["source"] or "null_level" in r["source"]:
            assert r["null_construction"] != "none"


def test_the_two_helpers_agree_so_the_reproduction_gap_stays_small():
    """`cell_importance` generalizes `variance_ceiling`; the gap row is the drift guard."""
    df = _games({1: 16.0, 2: 26.0, 3: 34.0}, games_each=80, noise=6.0)
    rows = opponent_rows(df, "x", n_shuffles=3, seed=0)
    gaps = [abs(r["share_of_variance"]) for r in rows
            if r["source"] == "null_reproduction_gap"]
    assert len(gaps) == 2
    assert max(gaps) < 0.01


def test_main_effects_and_interaction_rows_carry_their_own_frame_size():
    df = _games({1: 16.0, 2: 34.0}, noise=5.0)
    panel = df[df["player_id"] == 1]
    rows = opponent_main_rows(df, "x") + opponent_rows(panel, "x", n_shuffles=2, seed=0)
    by = _by_source(rows)
    assert by["opponent_x_season"]["n_games"] == len(df)
    assert by["opponent_x_archetype_x_season"]["n_games"] == len(panel)


# ── Units ─────────────────────────────────────────────────────────────────────

def test_the_residual_sd_is_carried_in_dk_pts_not_as_a_share():
    df = _games({1: 16.0, 2: 34.0}, noise=5.0)
    by = _by_source(spread_rows(df, df, "x"))
    row = by["within_player_season_residual_sd"]
    assert row["units"] == "dk_pts"
    assert row["share_of_variance"] is None
    assert abs(row["value"] - df["resid_dk_pts"].std(ddof=0)) < 1e-9
    assert by["dk_pts_sd"]["value"] > row["value"]


def test_measure_emits_one_row_per_source_and_null_pair():
    df = _games({1: 16.0, 2: 26.0, 3: 34.0}, games_each=80, noise=6.0)
    table = measure(df, df, "2021-22..2021-22", n_shuffles=2, seed=0)
    assert not table.empty
    assert table["seasons"].eq("2021-22..2021-22").all()
    assert set(table["basis"]) <= {TOTAL, RESIDUAL}
    # (source, null_construction) is the key — the same source appears once per null.
    assert not table.duplicated(subset=["source", "null_construction"]).any()
