import numpy as np
import pandas as pd

from src.features.team_context import (
    RELIABILITY_MINUTES,
    STALENESS_DECAY,
    UNDRAFTED_BUCKET,
    _loo_max,
    _loo_sum,
    _loo_weighted_mean,
    build_team_context,
    coverage_report,
    draft_bucket,
    membership_matrix,
    reliability,
    role_crowding,
    rookie_priors,
    season_start_roster,
)


def _roster(usg=(0.30, 0.20, 0.10), minutes=(2000.0, 1000.0, 1000.0),
            team="LAL", season="2023-24", start_id=1) -> pd.DataFrame:
    n = len(usg)
    return pd.DataFrame({
        "player_id": range(start_id, start_id + n),
        "team_abbreviation": [team] * n,
        "season": [season] * n,
        "adv_usg_pct": list(usg),
        "min_total": list(minutes),
        "prior_minutes": list(minutes),
    })


# ── Leave-one-out primitives ─────────────────────────────────────────────────

def test_loo_sum_excludes_the_player():
    df = _roster(usg=(0.30, 0.20, 0.10))
    out = _loo_sum(df, "adv_usg_pct")
    assert np.allclose(out.to_numpy(), [0.30, 0.40, 0.50])


def test_loo_sum_treats_missing_as_zero():
    df = _roster(usg=(0.30, np.nan, 0.10))
    out = _loo_sum(df, "adv_usg_pct")
    assert np.allclose(out.to_numpy(), [0.10, 0.40, 0.30])


def test_loo_max_returns_runner_up_for_the_leader():
    df = _roster(usg=(0.30, 0.20, 0.10))
    out = _loo_max(df, "adv_usg_pct")
    # the 0.30 player sees 0.20; everyone else sees 0.30
    assert np.allclose(out.to_numpy(), [0.20, 0.30, 0.30])


def test_loo_max_is_nan_for_a_lone_player():
    df = _roster(usg=(0.25,), minutes=(1500.0,))
    assert pd.isna(_loo_max(df, "adv_usg_pct").iloc[0])


def test_loo_weighted_mean_is_minutes_weighted():
    df = _roster(usg=(0.30, 0.20, 0.10), minutes=(2000.0, 1000.0, 1000.0))
    out = _loo_weighted_mean(df, "adv_usg_pct", "min_total")
    # player 1 excluded: (0.20*1000 + 0.10*1000)/2000 = 0.15
    assert abs(out.iloc[0] - 0.15) < 1e-12
    # player 2 excluded: (0.30*2000 + 0.10*1000)/3000
    assert abs(out.iloc[1] - (0.30 * 2000 + 0.10 * 1000) / 3000) < 1e-12


def test_loo_weighted_mean_ignores_weight_of_missing_values():
    df = _roster(usg=(0.30, np.nan, 0.10), minutes=(2000.0, 5000.0, 1000.0))
    out = _loo_weighted_mean(df, "adv_usg_pct", "min_total")
    # the NaN player's 5000 minutes must not dilute anyone's mean
    assert abs(out.iloc[0] - 0.10) < 1e-12
    assert abs(out.iloc[2] - 0.30) < 1e-12


def test_loo_primitives_do_not_cross_team_or_season_boundaries():
    df = pd.concat([
        _roster(usg=(0.30, 0.20), minutes=(2000.0, 1000.0), team="LAL", start_id=1),
        _roster(usg=(0.10, 0.05), minutes=(2000.0, 1000.0), team="BOS", start_id=3),
        _roster(usg=(0.40, 0.35), minutes=(2000.0, 1000.0), team="LAL",
                season="2022-23", start_id=5),
    ], ignore_index=True)
    out = _loo_sum(df, "adv_usg_pct")
    assert np.allclose(out.to_numpy(), [0.20, 0.30, 0.05, 0.10, 0.35, 0.40])


# ── Season-start rosters ─────────────────────────────────────────────────────

def _write_game_logs(tmp_path, rows: list[dict], season="2023-24") -> None:
    """rows: {player, team, day} — one appearance each. Team game order is by day."""
    df = pd.DataFrame([{
        "PLAYER_ID": r["player"],
        "TEAM_ABBREVIATION": r["team"],
        "GAME_ID": f"00{r['team']}{r['day']:04d}",
        "GAME_DATE": f"2023-10-{r['day']:02d}T00:00:00",
        "MIN": r.get("min", 20.0),
    } for r in rows])
    df.to_csv(tmp_path / f"game_logs_{season.replace('-', '_')}.csv", index=False)


def test_season_start_roster_uses_first_appearance_team(tmp_path):
    _write_game_logs(tmp_path, [
        {"player": 1, "team": "LAL", "day": 1},
        {"player": 2, "team": "LAL", "day": 1},
        {"player": 3, "team": "BOS", "day": 1},
    ])
    out = season_start_roster("2023-24", tmp_path).sort_values("player_id")
    assert out["team_abbreviation"].tolist() == ["LAL", "LAL", "BOS"]
    assert out["season"].unique().tolist() == ["2023-24"]


def test_season_start_roster_excludes_midseason_arrivals(tmp_path):
    """A player whose first game is deep into the team's season was not known pre-season."""
    rows = [{"player": 1, "team": "LAL", "day": d} for d in range(1, 21)]
    rows.append({"player": 99, "team": "LAL", "day": 20})      # team game index 19
    _write_game_logs(tmp_path, rows)
    out = season_start_roster("2023-24", tmp_path, window_games=10)
    assert out["player_id"].tolist() == [1]


def test_season_start_roster_window_is_in_team_games_not_calendar_days(tmp_path):
    rows = [{"player": 1, "team": "LAL", "day": d} for d in (1, 3, 5, 7, 9)]
    rows.append({"player": 2, "team": "LAL", "day": 9})        # team game index 4
    _write_game_logs(tmp_path, rows)
    assert set(season_start_roster("2023-24", tmp_path, window_games=4)["player_id"]) == {1, 2}
    assert set(season_start_roster("2023-24", tmp_path, window_games=3)["player_id"]) == {1}


def test_season_start_roster_attributes_an_early_trade_to_the_first_team(tmp_path):
    _write_game_logs(tmp_path, [
        {"player": 1, "team": "LAL", "day": 1},
        {"player": 1, "team": "BOS", "day": 5},
        {"player": 2, "team": "BOS", "day": 1},
    ])
    out = season_start_roster("2023-24", tmp_path)
    assert len(out) == 2
    assert out.set_index("player_id").loc[1, "team_abbreviation"] == "LAL"


def test_season_start_roster_is_empty_for_a_missing_season(tmp_path):
    assert season_start_roster("1998-99", tmp_path).empty


# ── Reliability and shrinkage ────────────────────────────────────────────────

def test_reliability_rises_with_minutes_and_stays_below_one():
    r = reliability([0.0, 200.0, 1000.0, 100_000.0])
    assert r[0] == 0.0
    assert (np.diff(r) > 0).all()
    assert (r < 1.0).all()


def test_reliability_is_one_half_at_the_prior_strength():
    assert abs(reliability([RELIABILITY_MINUTES])[0] - 0.5) < 1e-12


def test_reliability_decays_with_staleness():
    fresh = reliability([1500.0], lag=1)[0]
    stale = reliability([1500.0], lag=3)[0]
    assert abs(stale - fresh * STALENESS_DECAY ** 2) < 1e-12
    assert stale < fresh


def test_reliability_treats_missing_minutes_as_no_evidence():
    assert reliability([np.nan, -5.0])[0] == 0.0
    assert reliability([np.nan, -5.0])[1] == 0.0


# ── Draft buckets and the rookie prior ───────────────────────────────────────

def test_draft_bucket_splits_slots_and_keeps_undrafted_separate():
    out = draft_bucket(pd.Series([1, 5, 6, 14, 15, 30, 31, 60, np.nan]))
    assert out.tolist() == [
        "lottery_top5", "lottery_top5", "lottery", "lottery",
        "late_first", "late_first", "second_round", "second_round", UNDRAFTED_BUCKET,
    ]


def _described(seasons, rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "player_id": r["player"],
        "season_index": seasons.index(r["season"]),
        "debut_index": seasons.index(r["debut"]),
        "draft_bucket": r["bucket"],
        "adv_usg_pct": r["usg"],
        "prior_minutes": r.get("minutes", 1000.0),
        "gmm_p0": r.get("p0", 1.0),
        "gmm_p1": r.get("p1", 0.0),
    } for r in rows])


def test_rookie_priors_only_see_earlier_seasons():
    """An expanding window — a backtest must not learn a draft slot from the future."""
    seasons = ["2021-22", "2022-23", "2023-24"]
    described = _described(seasons, [
        {"player": 1, "season": "2021-22", "debut": "2021-22", "bucket": "lottery", "usg": 0.10},
        {"player": 2, "season": "2022-23", "debut": "2022-23", "bucket": "lottery", "usg": 0.30},
    ])
    priors = rookie_priors(described, seasons, ["gmm_p0", "gmm_p1"])
    lottery = priors[priors["draft_bucket"] == "lottery"].set_index("season")
    # 2022-23 sees only the 2021-22 rookie; 2023-24 sees both.
    assert abs(lottery.loc["2022-23", "adv_usg_pct"] - 0.10) < 1e-12
    assert abs(lottery.loc["2023-24", "adv_usg_pct"] - 0.20) < 1e-12


def test_rookie_priors_fall_back_to_the_pooled_mean_without_history():
    seasons = ["2021-22", "2022-23"]
    described = _described(seasons, [
        {"player": 1, "season": "2021-22", "debut": "2021-22", "bucket": "lottery", "usg": 0.10},
    ])
    priors = rookie_priors(described, seasons, ["gmm_p0", "gmm_p1"])
    first = priors[priors["season"] == "2021-22"]
    # No history at all, so every bucket gets the pooled value rather than NaN.
    assert first["adv_usg_pct"].notna().all()


def test_rookie_priors_back_off_to_all_rookies_for_an_unseen_bucket():
    seasons = ["2021-22", "2022-23"]
    described = _described(seasons, [
        {"player": 1, "season": "2021-22", "debut": "2021-22", "bucket": "lottery", "usg": 0.10},
    ])
    priors = rookie_priors(described, seasons, ["gmm_p0", "gmm_p1"]).set_index(
        ["season", "draft_bucket"])
    assert abs(priors.loc[("2022-23", "second_round"), "adv_usg_pct"] - 0.10) < 1e-12


# ── Role crowding ────────────────────────────────────────────────────────────

def _with_membership(df: pd.DataFrame, memberships: list[list[float]]) -> pd.DataFrame:
    m = pd.DataFrame(memberships, columns=[f"gmm_p{i}" for i in range(len(memberships[0]))])
    return pd.concat([df.reset_index(drop=True), m], axis=1)


def test_role_crowding_is_one_for_identical_players():
    df = _with_membership(_roster(minutes=(1000.0, 1000.0, 1000.0)),
                          [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
    out = role_crowding(df)
    assert np.allclose(out.to_numpy(), 1.0)


def test_role_crowding_is_zero_for_orthogonal_players():
    df = _with_membership(_roster(minutes=(1000.0, 1000.0, 1000.0)),
                          [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    assert np.allclose(role_crowding(df).to_numpy(), 0.0)


def test_role_crowding_is_minutes_weighted():
    """A similar teammate who barely plays crowds less than one who plays a lot."""
    heavy = _with_membership(_roster(minutes=(1000.0, 4000.0, 1000.0)),
                             [[1, 0], [1, 0], [0, 1]])
    light = _with_membership(_roster(minutes=(1000.0, 100.0, 4000.0)),
                             [[1, 0], [1, 0], [0, 1]])
    assert role_crowding(heavy).iloc[0] > role_crowding(light).iloc[0]


def test_role_crowding_excludes_self_and_is_nan_for_a_lone_player():
    df = _with_membership(_roster(usg=(0.25,), minutes=(1500.0,)), [[1, 0, 0]])
    assert pd.isna(role_crowding(df).iloc[0])


def test_role_crowding_does_not_collapse_like_a_mean():
    """Two rosters with identical mean membership must get different crowding scores.

    This is the centroid-collapse case: a minutes-weighted average of membership
    vectors cannot tell these apart, so any encoder built on the mean is blind to the
    difference. Role crowding is nonlinear in the players and separates them.

    Four interchangeable players is *maximal* overlap (every pair similarity is 1), so
    the uniform roster scores higher than two specialist pairs, where each player is
    contested by only one of three teammates (1/3).
    """
    two_pairs = _with_membership(_roster(usg=(0.2,) * 4, minutes=(1000.0,) * 4),
                                 [[1, 0], [1, 0], [0, 1], [0, 1]])
    interchangeable = _with_membership(_roster(usg=(0.2,) * 4, minutes=(1000.0,) * 4),
                                       [[0.5, 0.5]] * 4)
    cols = ["gmm_p0", "gmm_p1"]
    # In raw membership space — what an averaging encoder sees — both are [0.5, 0.5].
    assert np.allclose(two_pairs[cols].mean().to_numpy(),
                       interchangeable[cols].mean().to_numpy())

    assert abs(role_crowding(two_pairs).mean() - 1 / 3) < 1e-9
    assert abs(role_crowding(interchangeable).mean() - 1.0) < 1e-9


def test_role_crowding_without_normalization_respects_an_uncertain_rookie():
    """An imputed rookie's membership is a mean of unit vectors, so its norm is < 1.

    Re-normalizing it would claim a confident role we do not have, and overstate how
    much the rookie crowds his teammates. With `normalize=False` the shorter vector
    correctly contributes less similarity.
    """
    confident = _with_membership(_roster(usg=(0.2, 0.2), minutes=(1000.0, 1000.0)),
                                 [[1.0, 0.0], [1.0, 0.0]])
    uncertain = _with_membership(_roster(usg=(0.2, 0.2), minutes=(1000.0, 1000.0)),
                                 [[1.0, 0.0], [0.6, 0.0]])
    assert abs(role_crowding(confident, normalize=False).iloc[0] - 1.0) < 1e-12
    assert abs(role_crowding(uncertain, normalize=False).iloc[0] - 0.6) < 1e-12
    # ...whereas normalizing erases the distinction entirely.
    assert abs(role_crowding(uncertain, normalize=True).iloc[0] - 1.0) < 1e-12


def test_membership_matrix_rows_are_unit_norm():
    df = _with_membership(_roster(), [[3, 4, 0], [0, 0, 5], [1, 1, 1]])
    M, cols = membership_matrix(df)
    assert cols == ["gmm_p0", "gmm_p1", "gmm_p2"]
    assert np.allclose(np.linalg.norm(M, axis=1), 1.0)


def test_membership_columns_are_ordered_numerically_not_lexically():
    df = _roster(usg=(0.1,) * 1, minutes=(100.0,))
    for i in (0, 2, 10, 11):
        df[f"gmm_p{i}"] = float(i)
    _, cols = membership_matrix(df)
    assert cols == ["gmm_p0", "gmm_p2", "gmm_p10", "gmm_p11"]


# ── Assembly ─────────────────────────────────────────────────────────────────

def _described_roster(sources=("prior", "prior", "prior"),
                      minutes=(2000.0, 1000.0, 1000.0)) -> pd.DataFrame:
    df = _with_membership(_roster(minutes=minutes), [[1, 0], [1, 0], [0, 1]])
    df["adv_ast_pct"] = [0.25, 0.15, 0.05]
    df["adv_pace"] = 100.0
    df["sco_pct_fga_3pt"] = [0.4, 0.3, 0.2]
    df["stats_source"] = list(sources)
    df["stats_lag"] = 1.0
    df["reliability"] = 0.9
    return df


def test_build_team_context_produces_one_row_per_player_season():
    out = build_team_context(_described_roster())
    assert len(out) == 3
    assert not out.duplicated(["player_id", "season"]).any()
    assert out["n_teammates"].tolist() == [2, 2, 2]
    assert np.allclose(out["teammate_minutes"].to_numpy(), [2000.0, 3000.0, 3000.0])
    # pace is a team property, not leave-one-out
    assert np.allclose(out["team_pace"].to_numpy(), 100.0)


def test_build_team_context_usage_load_is_roster_size_invariant():
    """The raw LOO sum grows with an inclusive roster; the load version must not."""
    small = _described_roster()
    big = pd.concat([small, _roster(usg=(0.2, 0.2), minutes=(50.0, 50.0), start_id=10)
                     .assign(adv_ast_pct=0.1, adv_pace=100.0, sco_pct_fga_3pt=0.3,
                             gmm_p0=0.0, gmm_p1=1.0, stats_source="prior",
                             stats_lag=1.0, reliability=0.5)], ignore_index=True)
    s = build_team_context(small).set_index("player_id")
    b = build_team_context(big).set_index("player_id")
    assert b.loc[1, "teammate_usage_sum"] > s.loc[1, "teammate_usage_sum"] + 0.3
    assert abs(b.loc[1, "teammate_usage_load"] - s.loc[1, "teammate_usage_load"]) < 0.05


def test_roster_coverage_reports_the_imputed_share_of_weight():
    described = _described_roster(sources=("prior", "prior", "rookie"))
    out = build_team_context(described)
    # the rookie holds 1000 of 4000 roster minutes
    assert np.allclose(out["roster_coverage"].to_numpy(), 0.75)


def test_roster_coverage_is_one_when_every_member_is_observed():
    out = build_team_context(_described_roster())
    assert np.allclose(out["roster_coverage"].to_numpy(), 1.0)


def test_coverage_report_weights_by_realized_minutes_not_prior_minutes():
    """Prior minutes are zero for rookies, so weighting by them hides the whole gap."""
    described = _described_roster(sources=("prior", "prior", "rookie"))
    described.loc[described["stats_source"] == "rookie", "prior_minutes"] = 0.0
    described["season_minutes"] = [2000.0, 1000.0, 1000.0]

    rep = coverage_report(described).set_index(["team_abbreviation", "season"])
    assert abs(rep.loc[("LAL", "2023-24"), "share_rookie"] - 0.25) < 1e-12
