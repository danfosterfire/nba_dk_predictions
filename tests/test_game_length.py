import numpy as np
import pandas as pd

from src.features.game_length import (
    LENGTH_GRID,
    REGULATION_MINUTES,
    SNAP_TOLERANCE,
    assert_feasible,
    coverage,
    derive_game_length,
    join_lengths,
    load_game_logs,
    minutes_feasibility,
    snap_to_grid,
    team_minutes,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _team_rows(game_id: str, team_id: int, length: float, n_players: int = 10,
               season: str = "2021-22", season_type: str = "regular") -> list[dict]:
    """One team's box score for a game of `length` minutes, minutes split evenly.

    Five players are on the court at all times, so the team's minutes must sum to
    exactly 5 x length however they are distributed among the players.
    """
    total = 5.0 * length
    return [{"season": season, "season_type": season_type, "game_id": game_id,
             "team_id": team_id, "min": total / n_players}
            for _ in range(n_players)]


def _game(game_id: str, length: float, away_length: float | None = None,
          n_players: int = 10, **kw) -> pd.DataFrame:
    """A complete two-team game. `away_length` differing is the disagreement case."""
    rows = _team_rows(game_id, 1, length, n_players, **kw)
    rows += _team_rows(game_id, 2, length if away_length is None else away_length,
                       n_players, **kw)
    return pd.DataFrame(rows)


# ── Snapping ──────────────────────────────────────────────────────────────────

def test_grid_is_regulation_plus_five_minute_periods():
    assert LENGTH_GRID[0] == REGULATION_MINUTES
    assert list(LENGTH_GRID[:4]) == [48.0, 53.0, 58.0, 63.0]


def test_snap_returns_nearest_grid_point_and_signed_residual():
    nearest, residual = snap_to_grid(np.array([48.0, 47.9, 53.2, 58.0]))
    assert list(nearest) == [48.0, 48.0, 53.0, 58.0]
    assert np.allclose(residual, [0.0, -0.1, 0.2, 0.0])


def test_snap_is_a_rounding_correction_not_a_guess():
    """Every observed residual is far inside the 2.5-minute decision boundary."""
    # 0.617 is the worst residual in 30 seasons; 2.5 is where snapping would flip.
    nearest, _ = snap_to_grid(np.array([48.617, 47.383, 53.617]))
    assert list(nearest) == [48.0, 48.0, 53.0]


# ── Derivation ────────────────────────────────────────────────────────────────

def test_regulation_game_is_forty_eight_minutes():
    out = derive_game_length(_game("g1", 48.0))
    assert len(out) == 1
    assert out["game_length"].iloc[0] == 48.0
    assert out["n_overtimes"].iloc[0] == 0
    assert out["n_periods"].iloc[0] == 4
    assert bool(out["reliable"].iloc[0])


def test_overtime_games_recover_their_period_count():
    for length, n_ot in [(53.0, 1), (58.0, 2), (63.0, 3), (68.0, 4)]:
        out = derive_game_length(_game(f"g{n_ot}", length))
        assert out["game_length"].iloc[0] == length
        assert out["n_overtimes"].iloc[0] == n_ot
        assert out["n_periods"].iloc[0] == 4 + n_ot


def test_team_minutes_are_five_times_the_length_regardless_of_roster_size():
    for n in [5, 8, 13, 20]:
        tm = team_minutes(_game("g1", 53.0, n_players=n))
        assert np.allclose(tm["team_minutes"], 5.0 * 53.0)
        assert np.allclose(tm["length_raw"], 53.0)


def test_box_score_rounding_still_snaps_to_the_right_length():
    """Real logs are off by up to 0.6 min; the derivation must absorb that."""
    df = _game("g1", 48.0)
    df.loc[0, "min"] += 0.4       # a rounding artifact in one player's line
    df.loc[1, "min"] -= 0.1
    out = derive_game_length(df)
    assert out["game_length"].iloc[0] == 48.0
    assert bool(out["reliable"].iloc[0])
    assert 0 < out["max_abs_residual"].iloc[0] < SNAP_TOLERANCE


# ── The validation, and the failure modes it catches ──────────────────────────

def test_disagreeing_teams_are_flagged_not_averaged():
    """The two teams are independent estimates; a split is a defect, not a mean."""
    out = derive_game_length(_game("g1", 48.0, away_length=53.0))
    assert not bool(out["teams_agree"].iloc[0])
    assert not bool(out["reliable"].iloc[0])


def test_a_frame_missing_players_is_caught_rather_than_snapped_anyway():
    """The `component_targets.parquet` trap: `clean()` drops low-game players, so a
    team sum over a filtered frame understates the length silently."""
    df = _game("g1", 48.0, n_players=10)
    truncated = df.drop(index=[0, 1, 2])          # three of one team's ten players gone
    out = derive_game_length(truncated)
    assert not bool(out["reliable"].iloc[0])
    assert out["max_abs_residual"].iloc[0] > SNAP_TOLERANCE


def test_one_sided_game_is_unreliable():
    only_home = pd.DataFrame(_team_rows("g1", 1, 48.0))
    out = derive_game_length(only_home)
    assert out["n_teams"].iloc[0] == 1
    assert not bool(out["reliable"].iloc[0])


def test_missing_minutes_count_as_zero_not_as_a_dropped_player():
    """A sub-30-second appearance logs 0.0 or blank; either way n_players must hold."""
    df = _game("g1", 48.0, n_players=10)
    df.loc[0, "min"] = np.nan
    tm = team_minutes(df)
    assert set(tm["n_players"]) == {10}


# ── Season typing and coverage ────────────────────────────────────────────────

def test_game_length_covers_both_season_types(tmp_path):
    """Unlike the fitting frames, this one wants playoff games: length is a property of
    the game, and `prior_playoff_minutes` needs a denominator."""
    cols = ["GAME_ID", "TEAM_ID", "MIN"]
    pd.DataFrame([{"GAME_ID": "g1", "TEAM_ID": 1, "MIN": 24.0}])[cols].to_csv(
        tmp_path / "game_logs_2021_22.csv", index=False)
    pd.DataFrame([{"GAME_ID": "p1", "TEAM_ID": 1, "MIN": 24.0}])[cols].to_csv(
        tmp_path / "game_logs_playoffs_2021_22.csv", index=False)

    logs = load_game_logs(tmp_path)
    assert set(logs["season"]) == {"2021-22"}
    assert set(logs["season_type"]) == {"regular", "playoffs"}
    assert set(logs.columns) >= {"game_id", "team_id", "min"}


def test_coverage_reports_agreement_and_overtime_rate_per_season():
    df = pd.concat([_game("g1", 48.0), _game("g2", 53.0), _game("g3", 48.0)])
    cov = coverage(derive_game_length(df))
    assert len(cov) == 1
    assert cov["games"].iloc[0] == 3
    assert cov["teams_disagree"].iloc[0] == 0
    assert np.isclose(cov["ot_rate"].iloc[0], 1 / 3)


# ── Feasibility: min ~ Binomial(game_length, .) must be well posed everywhere ──

def _lengths(games: dict[str, float], season: str = "2021-22") -> pd.DataFrame:
    """A game_length-shaped frame: one row per game id."""
    return pd.DataFrame([
        {"season": season, "season_type": "regular", "game_id": gid,
         "game_length": length,
         "n_overtimes": int(round((length - REGULATION_MINUTES) / 5.0))}
        for gid, length in games.items()])


def _player_games(rows: list[tuple[str, int, float]],
                  season: str = "2021-22") -> pd.DataFrame:
    """A component-targets-shaped frame: (game_id, player_id, minutes)."""
    return pd.DataFrame([{"season": season, "game_id": gid, "player_id": pid, "min": m}
                         for gid, pid, m in rows])


def test_feasibility_reports_full_coverage_and_no_violations_when_all_is_well():
    lengths = _lengths({"0021": 48.0, "0022": 53.0})
    targets = _player_games([("0021", 1, 36.0), ("0021", 2, 48.0), ("0022", 1, 52.0)])
    out = minutes_feasibility(targets, lengths)
    row = out[out["season"] == "all"].iloc[0]
    assert row["join_coverage"] == 1.0
    assert row["violations"] == 0
    assert row["unmatched"] == 0
    assert abs(row["max_min_over_length"] - 1.0) < 1e-12
    assert_feasible(out)


def test_a_single_violation_is_a_build_failure_not_a_metric():
    """The whole point of item 8: a nonzero count raises rather than being reported."""
    lengths = _lengths({"0021": 48.0})
    targets = _player_games([("0021", 1, 36.0), ("0021", 2, 48.1)])
    out = minutes_feasibility(targets, lengths)
    assert out[out["season"] == "all"].iloc[0]["violations"] == 1
    try:
        assert_feasible(out)
        raise AssertionError("expected assert_feasible to raise on a violation")
    except AssertionError as exc:
        assert "min > game_length" in str(exc)


def test_an_unmatched_player_game_also_raises_rather_than_shrinking_the_frame():
    lengths = _lengths({"0021": 48.0})
    targets = _player_games([("0021", 1, 36.0), ("0099", 2, 30.0)])
    out = minutes_feasibility(targets, lengths)
    assert out[out["season"] == "all"].iloc[0]["unmatched"] == 1
    try:
        assert_feasible(out)
        raise AssertionError("expected assert_feasible to raise on an unmatched row")
    except AssertionError as exc:
        assert "pad_game_id" in str(exc)


def test_the_join_survives_game_ids_stored_as_int_on_one_side():
    """The recorded quirk: int64 in the logs, zero-padded strings in the box scores."""
    lengths = _lengths({"0022100021": 48.0})
    targets = _player_games([(22100021, 1, 36.0)])
    out = minutes_feasibility(targets, lengths)
    assert out[out["season"] == "all"].iloc[0]["unmatched"] == 0


def test_overtime_minutes_above_regulation_are_feasible_not_violations():
    """5.93% of games run long, and truncating at 48 is what this section exists to avoid."""
    lengths = _lengths({"0021": 63.0})
    targets = _player_games([("0021", 1, 60.0)])
    out = minutes_feasibility(targets, lengths)
    row = out[out["season"] == "all"].iloc[0]
    assert row["violations"] == 0
    assert row["player_games_above_regulation"] == 1
    assert row["max_minutes"] == 60.0
    assert_feasible(out)


def test_feasibility_rows_are_added_beside_the_derivation_rows_not_instead_of_them():
    """The extension must not rewrite `game_length_coverage.csv`'s existing section."""
    derivation = coverage(derive_game_length(_game("0021", 48.0)))
    lengths = _lengths({"0021": 48.0})
    feasible = minutes_feasibility(_player_games([("0021", 1, 36.0)]), lengths)
    combined = pd.concat([derivation, feasible], ignore_index=True)
    assert set(derivation.columns) <= set(combined.columns)
    assert (combined["analysis"] == "derivation").sum() == len(derivation)
    assert (combined["analysis"] == "feasibility").sum() == len(feasible)


def test_per_season_rows_accompany_the_overall_row():
    lengths = pd.concat([_lengths({"0021": 48.0}, "2020-21"),
                         _lengths({"0031": 48.0}, "2021-22")], ignore_index=True)
    targets = pd.concat([_player_games([("0021", 1, 30.0)], "2020-21"),
                         _player_games([("0031", 1, 40.0)], "2021-22")],
                        ignore_index=True)
    out = minutes_feasibility(targets, lengths)
    assert set(out["season"]) == {"all", "2020-21", "2021-22"}
    assert out[out["season"] == "all"].iloc[0]["player_games"] == 2
