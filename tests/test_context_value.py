import numpy as np
import pandas as pd

from src.eda.context_value import (
    DESCRIPTION_SOURCES,
    UNDESCRIBED_SOURCES,
    attach_season_minutes,
    cancellation,
    description_source,
    dk_component_weights,
    measure,
    roster_coverage_profile,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

SEASONS = ["2020-21", "2021-22", "2022-23"]


def _frame(rows: list[tuple[int, str, float]]) -> pd.DataFrame:
    """The inclusive roster frame: (player_id, season, realized total minutes)."""
    return pd.DataFrame(rows, columns=["player_id", "season", "min_total"])


def _qualified(rows: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["player_id", "season"])


def _roster(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["player_id", "season", "team_abbreviation"])


# ── The four-way split ────────────────────────────────────────────────────────

def test_the_four_sources_are_assigned_from_the_two_matrices():
    frame = _frame([
        (1, "2020-21", 2000.0), (1, "2021-22", 2000.0),   # described both years
        (2, "2020-21", 120.0), (2, "2021-22", 900.0),      # sub-threshold in 2020-21
        (3, "2020-21", 1500.0), (3, "2022-23", 1500.0),    # absent in 2021-22
        (4, "2021-22", 800.0),                             # debut in 2021-22
    ])
    qualified = _qualified([(1, "2020-21"), (1, "2021-22"), (2, "2021-22"),
                            (3, "2020-21"), (3, "2022-23"), (4, "2021-22")])
    roster = _roster([(1, "2021-22", "LAL"), (2, "2021-22", "LAL"),
                      (3, "2022-23", "LAL"), (4, "2021-22", "LAL")])
    got = list(description_source(roster, frame, qualified, SEASONS))
    assert got == ["prior", "sub_threshold", "returnee", "rookie"]


def test_a_player_absent_the_previous_season_is_a_returnee_not_a_rookie():
    """The gap must not pair across — that is what the returnee category is for."""
    frame = _frame([(1, "2020-21", 1800.0), (1, "2022-23", 1800.0)])
    qualified = _qualified([(1, "2020-21"), (1, "2022-23")])
    roster = _roster([(1, "2022-23", "BOS")])
    assert description_source(roster, frame, qualified, SEASONS).iloc[0] == "returnee"


def test_a_first_season_player_is_a_rookie_even_with_heavy_minutes():
    frame = _frame([(9, "2021-22", 2400.0)])
    qualified = _qualified([(9, "2021-22")])
    roster = _roster([(9, "2021-22", "BOS")])
    assert description_source(roster, frame, qualified, SEASONS).iloc[0] == "rookie"


def test_sub_threshold_needs_the_qualified_matrix_not_just_the_inclusive_frame():
    """A player present in the inclusive frame but absent from the qualified one."""
    frame = _frame([(5, "2020-21", 90.0), (5, "2021-22", 1400.0)])
    roster = _roster([(5, "2021-22", "NYK")])
    inclusive_only = description_source(roster, frame, _qualified([(5, "2021-22")]),
                                        SEASONS).iloc[0]
    now_qualified = description_source(roster, frame,
                                       _qualified([(5, "2020-21"), (5, "2021-22")]),
                                       SEASONS).iloc[0]
    assert inclusive_only == "sub_threshold"
    assert now_qualified == "prior"


# ── The denominator ───────────────────────────────────────────────────────────

def test_the_weight_is_realized_season_s_minutes():
    frame = _frame([(1, "2021-22", 1500.0), (2, "2021-22", 300.0)])
    roster = _roster([(1, "2021-22", "LAL"), (2, "2021-22", "LAL")])
    got = attach_season_minutes(roster, frame)
    assert list(got["season_minutes"]) == [1500.0, 300.0]


def test_a_roster_row_with_no_season_s_row_contributes_zero_rather_than_nan():
    frame = _frame([(1, "2021-22", 1500.0)])
    roster = _roster([(1, "2021-22", "LAL"), (2, "2021-22", "LAL")])
    got = attach_season_minutes(roster, frame)
    assert got["season_minutes"].notna().all()
    assert got.loc[got["player_id"] == 2, "season_minutes"].iloc[0] == 0.0


# ── The profile ───────────────────────────────────────────────────────────────

def _scenario() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One team-season: a described starter, a sub-threshold reserve and a rookie."""
    frame = _frame([
        (1, "2020-21", 2000.0), (1, "2021-22", 2000.0),
        (2, "2020-21", 100.0), (2, "2021-22", 1000.0),
        (3, "2021-22", 1000.0),
    ])
    qualified = _qualified([(1, "2020-21"), (1, "2021-22"), (2, "2021-22"),
                            (3, "2021-22")])
    roster = _roster([(1, "2021-22", "LAL"), (2, "2021-22", "LAL"),
                      (3, "2021-22", "LAL")])
    return roster, frame, qualified


def test_the_four_shares_partition_every_team_season():
    """The verification the plan names: they must sum to exactly 1.0, not approximately."""
    roster, frame, qualified = _scenario()
    out = roster_coverage_profile(roster, frame, qualified, SEASONS, "A")
    check = out[out["scope"] == "checks"]
    assert len(check) == 1
    assert float(check["value"].iloc[0]) < 1e-12


def test_minutes_shares_use_minutes_and_headcount_shares_use_rows():
    """4000 realized minutes: 2000 described, 1000 sub-threshold, 1000 rookie."""
    roster, frame, qualified = _scenario()
    out = roster_coverage_profile(roster, frame, qualified, SEASONS, "A")
    league = out[out["scope"] == "league"].set_index(["key", "metric"])["value"]
    assert abs(league[("prior", "share_of_minutes")] - 0.50) < 1e-12
    assert abs(league[("sub_threshold", "share_of_minutes")] - 0.25) < 1e-12
    assert abs(league[("rookie", "share_of_minutes")] - 0.25) < 1e-12
    assert abs(league[("undescribed", "share_of_minutes")] - 0.50) < 1e-12
    # Head counts weight all three equally, which is exactly why both are reported.
    for source in ("prior", "sub_threshold", "rookie"):
        assert abs(league[(source, "share_of_headcount")] - 1 / 3) < 1e-12


def test_weighting_by_prior_minutes_would_hide_the_rookie_which_is_why_it_is_not_used():
    """A rookie has no S-1 minutes at all, so the circular weighting reports full coverage."""
    roster, frame, qualified = _scenario()
    out = roster_coverage_profile(roster, frame, qualified, SEASONS, "A")
    league = out[out["scope"] == "league"].set_index(["key", "metric"])["value"]
    assert league[("rookie", "share_of_minutes")] > 0.0


def test_the_window_label_is_carried_so_the_two_rosters_stay_distinguishable():
    roster, frame, qualified = _scenario()
    start = roster_coverage_profile(roster, frame, qualified, SEASONS, "A",
                                    window="season_start")
    whole = roster_coverage_profile(roster, frame, qualified, SEASONS, "A",
                                    window="whole_season")
    assert start["window"].eq("season_start").all()
    assert whole["window"].eq("whole_season").all()


def test_a_late_arriving_rookie_raises_the_undescribed_share():
    """Why the two windows disagree: rookies and returnees arrive late."""
    roster, frame, qualified = _scenario()
    league_start = (roster_coverage_profile(roster, frame, qualified, SEASONS, "A")
                    .pipe(lambda d: d[d["scope"] == "league"])
                    .set_index(["key", "metric"])["value"])
    wider_frame = pd.concat([frame, _frame([(4, "2021-22", 1000.0)])], ignore_index=True)
    wider_roster = pd.concat([roster, _roster([(4, "2021-22", "LAL")])],
                             ignore_index=True)
    league_whole = (roster_coverage_profile(wider_roster, wider_frame, qualified,
                                           SEASONS, "A", window="whole_season")
                    .pipe(lambda d: d[d["scope"] == "league"])
                    .set_index(["key", "metric"])["value"])
    assert (league_whole[("undescribed", "share_of_minutes")]
            > league_start[("undescribed", "share_of_minutes")])


def test_worst_team_seasons_are_named_and_ordered():
    frame = _frame([(p, s, 1000.0) for p in range(1, 7) for s in SEASONS])
    qualified = _qualified([(p, s) for p in range(1, 7) for s in SEASONS])
    # LAL fields two described veterans; BOS fields two rookies.
    roster = _roster([(1, "2021-22", "LAL"), (2, "2021-22", "LAL"),
                      (5, "2020-21", "BOS"), (6, "2020-21", "BOS")])
    out = roster_coverage_profile(roster, frame, qualified, SEASONS, "A")
    worst = out[out["scope"] == "worst_team_seasons"]
    assert worst.iloc[0]["key"].startswith("BOS")
    assert list(worst["value"]) == sorted(worst["value"], reverse=True)


def test_every_undescribed_source_is_one_of_the_four():
    assert set(UNDESCRIBED_SOURCES) < set(DESCRIPTION_SOURCES)
    assert "prior" not in UNDESCRIBED_SOURCES


# ── Cross-component cancellation ──────────────────────────────────────────────

def test_dk_weights_cover_the_seven_components_and_exclude_the_aggregate():
    outcomes = [f"{c}_per36" for c in ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")]
    weights = dk_component_weights(outcomes + ["minutes", "dk_pts_per_game"])
    assert len(weights) == 7
    assert "minutes" not in weights and "dk_pts_per_game" not in weights
    assert weights["reb_per36"] == 1.25 and weights["tov_per36"] == -0.5


def test_cancellation_is_one_when_a_single_component_moves():
    got = cancellation({"reb_per36": 0.4}, {"reb_per36": 1.25})
    assert abs(got["gross_dk_movement"] - 0.5) < 1e-12
    assert abs(got["net_dk_movement"] - 0.5) < 1e-12
    assert abs(got["cancellation_ratio"] - 1.0) < 1e-12


def test_cancellation_is_infinite_when_two_components_exactly_offset():
    """The pathological case the ratio exists to name: real movement, zero net."""
    got = cancellation({"reb_per36": 1.2, "ast_per36": -1.0},
                       {"reb_per36": 1.25, "ast_per36": 1.5})
    assert abs(got["gross_dk_movement"] - 3.0) < 1e-12
    assert abs(got["net_dk_movement"]) < 1e-12
    assert got["cancellation_ratio"] == np.inf


def test_cancellation_signs_the_net_but_not_the_gross():
    got = cancellation({"reb_per36": 0.4, "tov_per36": 1.0},
                       {"reb_per36": 1.25, "tov_per36": -0.5})
    assert abs(got["gross_dk_movement"] - (0.5 + 0.5)) < 1e-12
    assert abs(got["net_dk_movement"] - 0.0) < 1e-12


def test_cancellation_ignores_a_component_with_no_measurable_effect():
    got = cancellation({"reb_per36": 0.4, "ast_per36": np.nan},
                       {"reb_per36": 1.25, "ast_per36": 1.5})
    assert abs(got["gross_dk_movement"] - 0.5) < 1e-12


def _panel(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A context panel where one feature raises rebounds and lowers assists equally.

    In DK terms `+0.8 reb` is worth 1.0 and `-0.667 ast` is worth -1.0, so the two cancel
    exactly in the sum while both are real — the configuration the component-head
    architecture exists for.
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 1.0, n)
    df = pd.DataFrame({
        "player_id": np.arange(n), "season": rng.choice(["2020-21", "2021-22"], n),
        "prior_dk_pts_per_game": rng.normal(20.0, 5.0, n),
        "prior_min": rng.normal(28.0, 6.0, n),
        "prior_usg": rng.normal(0.20, 0.04, n),
        "teammate_assist_supply": x,
        "reb_per36": 6.0 + 0.8 * x + rng.normal(0, 0.05, n),
        "ast_per36": 4.0 - (1.0 / 1.5) * x + rng.normal(0, 0.05, n),
        "minutes": rng.normal(28.0, 5.0, n),
    })
    for c in ("pts_per36", "fg3m_per36", "stl_per36", "blk_per36", "tov_per36"):
        df[c] = rng.normal(2.0, 0.05, n)
    df["dk_pts_per_game"] = rng.normal(20.0, 5.0, n)
    return df


def test_measure_emits_per_sd_effects_in_component_units():
    """`effect_c` must equal `r_c * sd(residual y_c)`, which is what makes it DK-weightable."""
    table = measure(_panel(), features=["teammate_assist_supply"], season_fe=True)
    row = table.iloc[0]
    assert abs(row["effect_reb_per36"] - 0.8) < 0.1
    assert abs(row["effect_ast_per36"] - (-1.0 / 1.5)) < 0.1
    # The correlation is dimensionless and much larger — the two are not interchangeable.
    assert abs(row["r_reb_per36"]) > abs(row["effect_reb_per36"])


def test_measure_reports_a_large_cancellation_when_components_offset():
    table = measure(_panel(), features=["teammate_assist_supply"], season_fe=True)
    row = table.iloc[0]
    assert row["gross_dk_movement"] > 1.9
    assert abs(row["net_dk_movement"]) < 0.2
    assert row["cancellation_ratio"] > 8.0


def test_measure_keeps_the_existing_correlation_columns_untouched():
    """The extension must ADD columns, not rewrite what the artifact already carried."""
    table = measure(_panel(), features=["teammate_assist_supply"], season_fe=True)
    for col in ("feature", "n", "r2_controls", "r2_with_feature", "delta_r2",
                "r_reb_per36", "r_dk_pts_per_game"):
        assert col in table.columns
    for col in ("gross_dk_movement", "net_dk_movement", "cancellation_ratio"):
        assert col in table.columns
