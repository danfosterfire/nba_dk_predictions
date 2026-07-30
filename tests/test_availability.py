import numpy as np
import pandas as pd
import pytest

from src.eda.availability import (
    PLAYOFF_ROLE_LABELS,
    carryover,
    playoff_scope,
    serial_structure,
    overdispersion,
    persistence,
    spell_distribution,
    weighted_r2,
    window_bracket,
    with_lags,
)
from src.features.availability import (
    _trailing_missed,
    absence_spells,
    attach_workload,
    build_season_panel,
    playoff_workload,
    season_availability,
)

SEASONS = ["2019-20", "2020-21", "2021-22", "2022-23"]


# ── Synthetic builders ────────────────────────────────────────────────────────

def _game_log(rows: list[dict]) -> pd.DataFrame:
    """A minimal game-log-shaped frame, in the raw CSV's upper-case schema."""
    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def _write_log(tmp_path, season: str, rows: list[dict]):
    from src.data.fetch import _slug
    _game_log(rows).to_csv(tmp_path / f"game_logs_{_slug(season)}.csv", index=False)


def _team_season(team_id: int, n_games: int, start_day: int = 1) -> list[dict]:
    """The schedule skeleton: one anchor player who plays every game."""
    return [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 900 + team_id, "TEAM_ID": team_id,
             "GAME_ID": f"{team_id}{g:03d}", "GAME_DATE": f"2021-11-{start_day + g:02d}",
             "MIN": 30.0}
            for g in range(n_games)]


def _panel(rows: list[dict], tmp_path) -> pd.DataFrame:
    _write_log(tmp_path, "2021-22", rows)
    return build_season_panel("2021-22", tmp_path)


# ── The roster-window bracket ─────────────────────────────────────────────────

def test_appearance_window_is_blind_to_season_ending_absence(tmp_path):
    """The artifact this whole module exists to make visible.

    A player who plays games 0-4 of a 10-game season and is never seen again has
    trailing absences the full window sees and the appearance window cannot.
    """
    rows = _team_season(1, 10)
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in range(5)]
    panel = _panel(rows, tmp_path)

    hurt = panel[panel["player_id"] == 7]
    assert hurt["in_appearance_window"].sum() == 5   # only the games he played
    assert len(hurt) == 10                           # full window sees all ten

    appearance = season_availability(panel, "appearance")
    full = season_availability(panel, "full")
    assert appearance.loc[appearance.player_id == 7, "trailing_missed"].iloc[0] == 0
    assert full.loc[full.player_id == 7, "trailing_missed"].iloc[0] == 5


def test_played_is_keyed_on_team_not_just_game(tmp_path):
    """A game_id belongs to both teams, so a traded player must not count as having
    played for his old team in games he played for his new one."""
    rows = _team_season(1, 6) + _team_season(2, 6)
    # Player 7 plays games 0-1 for team 1, then games 2-3 for team 2. Team 1 and team 2
    # share no game ids here, but the merge must still not leak across teams.
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in range(2)]
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 2,
              "GAME_ID": f"2{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in range(2, 4)]
    panel = _panel(rows, tmp_path)

    for team, played_indices in [(1, {0, 1}), (2, {2, 3})]:
        sub = panel[(panel.player_id == 7) & (panel.team_id == team)]
        assert set(sub.loc[sub.played == 1, "team_game_index"]) == played_indices


def test_full_window_is_a_superset_of_appearance_window(tmp_path):
    rows = _team_season(1, 8)
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in (2, 5)]
    panel = _panel(rows, tmp_path)
    assert panel["in_appearance_window"].sum() < len(panel)
    # Every played game is inside the appearance window by construction.
    assert (panel.loc[panel.played == 1, "in_appearance_window"] == 1).all()


# ── Spells ────────────────────────────────────────────────────────────────────

def test_absence_spells_groups_consecutive_missed_games(tmp_path):
    rows = _team_season(1, 12)
    # Plays 0, misses 1-3, plays 4, misses 5, plays 6..11 -> spells of 3 and 1.
    played = [0, 4, 6, 7, 8, 9, 10, 11]
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in played]
    panel = _panel(rows, tmp_path)
    spells = absence_spells(panel, "appearance")
    mine = sorted(spells.loc[spells.player_id == 7, "spell_games"].tolist())
    assert mine == [1, 3]


def test_trailing_missed_counts_only_the_tail():
    assert _trailing_missed(pd.Series([1, 0, 1, 0, 0])) == 2
    assert _trailing_missed(pd.Series([1, 1, 1])) == 0
    assert _trailing_missed(pd.Series([0, 0, 0])) == 3   # never played: all trailing


def test_spell_distribution_reports_the_two_processes(tmp_path):
    rows = _team_season(1, 20)
    played = [0] + list(range(12, 20))          # one 11-game absence
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in played]
    panel = _panel(rows, tmp_path)
    out = pd.DataFrame(spell_distribution(panel, "appearance"))
    share = out[out.metric == "share_missed_games_in_spells_ge10"]["value"].iloc[0]
    assert share == 1.0   # the only spell is 11 long


# ── Season summary ────────────────────────────────────────────────────────────

def test_gp_share_and_minutes_per_game(tmp_path):
    rows = _team_season(1, 10)
    played = [0, 1, 2, 3]
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 24.0}
             for g in played]
    panel = _panel(rows, tmp_path)
    frame = season_availability(panel, "full")
    row = frame[frame.player_id == 7].iloc[0]
    assert row["gp"] == 4
    assert row["team_games"] == 10
    assert np.isclose(row["gp_share"], 0.4)
    assert np.isclose(row["minutes_per_game"], 24.0)


def test_window_bracket_reports_both_constructions(tmp_path):
    rows = _team_season(1, 10)
    rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": 7, "TEAM_ID": 1,
              "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}", "MIN": 20.0}
             for g in range(4)]
    panel = _panel(rows, tmp_path)
    frames = {w: season_availability(panel, w) for w in ("appearance", "full")}
    out = pd.DataFrame(window_bracket(panel, frames))
    rate = out[(out.metric == "played_rate")].set_index("window")["value"]
    assert rate["appearance"] > rate["full"]   # the narrower window is denser


# ── Statistics ────────────────────────────────────────────────────────────────

def test_weighted_r2_recovers_a_clean_linear_fit():
    rng = np.random.default_rng(0)
    x = rng.normal(size=400)
    y = 2.0 * x                       # noiseless
    assert np.isclose(weighted_r2(x, y, np.ones(400)), 1.0)


def test_weighted_r2_is_zero_on_noise():
    rng = np.random.default_rng(1)
    x = rng.normal(size=500)
    y = rng.normal(size=500)
    assert abs(weighted_r2(x, y, np.ones(500))) < 0.05


def test_with_lags_does_not_pair_across_a_missing_season():
    frame = pd.DataFrame({
        "player_id": [1, 1, 2, 2],
        "season": ["2019-20", "2021-22", "2019-20", "2020-21"],
        "gp_share": [0.5, 0.9, 0.4, 0.8],
    })
    out = with_lags(frame, SEASONS, ["gp_share"], max_lag=1)
    # Player 1 skipped 2020-21, so his 2021-22 row must have no lag-1 value.
    p1 = out[(out.player_id == 1) & (out.season == "2021-22")]
    assert p1["gp_share_lag1"].isna().all()
    # Player 2 played consecutive seasons, so his does.
    p2 = out[(out.player_id == 2) & (out.season == "2020-21")]
    assert np.isclose(p2["gp_share_lag1"].iloc[0], 0.4)


# ── Measurements over a synthetic panel ───────────────────────────────────────

def _availability_frame(n_players: int = 300) -> pd.DataFrame:
    """Player-seasons where availability is pure noise but minutes persist strongly.

    The point being that a persistence measurement must separate the two: `gp_share`
    should read ~0 and `minutes_per_game` ~1.
    """
    rng = np.random.default_rng(11)
    skill = rng.uniform(10, 34, size=n_players)
    rows = []
    for season in SEASONS:
        for p in range(n_players):
            rows.append({
                "season": season, "player_id": p,
                "gp_share": rng.uniform(0.2, 1.0),          # no persistence
                "minutes_per_game": skill[p] + rng.normal(0, 0.2),   # strong
                "total_minutes": 1500.0, "team_games": 82,
                "available_rate": rng.uniform(0.2, 1.0), "window_share": 1.0,
                "n_spells": rng.integers(0, 5), "longest_spell": rng.integers(0, 20),
                "single_game_spells": rng.integers(0, 4),
                "long_spells": rng.integers(0, 2), "missed_games": rng.integers(0, 40),
                "trailing_missed": rng.integers(0, 10),
                "start_play_rate": rng.uniform(0.2, 1.0),
            })
    return pd.DataFrame(rows)


def test_persistence_separates_noise_from_signal():
    out = pd.DataFrame(persistence(_availability_frame(), SEASONS, "full"))
    weighted = out[out.metric == "r_within_weighted"].set_index("key")["value"]
    assert abs(weighted["gp_share"]) < 0.15
    assert weighted["minutes_per_game"] > 0.9


def test_persistence_skips_a_constant_column():
    """`window_share` is identically 1 on the full window — undefined, not zero."""
    out = pd.DataFrame(persistence(_availability_frame(), SEASONS, "full"))
    assert "window_share" not in set(out["key"])


def test_carryover_bins_and_reports_incremental_r2():
    out = pd.DataFrame(carryover(_availability_frame(), SEASONS, "full"))
    assert {"incremental"}.issubset(set(out["key"]))
    metrics = set(out[out.key == "incremental"]["metric"])
    assert metrics == {"r2_prior_gp_share", "r2_plus_trailing_missed"}
    # Trailing absence is random here, so it must add essentially nothing.
    vals = out[out.key == "incremental"].set_index("metric")["value"]
    assert vals["r2_plus_trailing_missed"] - vals["r2_prior_gp_share"] < 0.02


def test_overdispersion_is_flat_when_availability_is_binomial():
    """A control: if games played really were independent coin flips, the variance
    ratio would be ~1. It is ~20 on real data, and that gap is the finding."""
    rng = np.random.default_rng(5)
    rows = []
    for season in SEASONS:
        for p in range(600):
            rows.append({
                "season": season, "player_id": p,
                "gp_share": rng.binomial(82, 0.8) / 82,
                "minutes_per_game": 30.0, "total_minutes": 2000.0, "team_games": 82,
            })
    out = pd.DataFrame(overdispersion(pd.DataFrame(rows), SEASONS, "full"))
    ratio = out[out.metric == "variance_ratio"]["value"].iloc[0]
    assert 0.5 < ratio < 2.0


# ── Serial structure ──────────────────────────────────────────────────────────

def test_serial_structure_recovers_a_known_markov_chain(tmp_path):
    """A control with the transition probabilities set by construction: the estimator
    has to return them, or the 3.96x clustering figure means nothing."""
    rng = np.random.default_rng(4)
    p_stay, q_return = 0.90, 0.30
    rows = _team_season(1, 28)
    for player in range(2, 260):
        state, games = 1, []
        for g in range(28):
            state = rng.random() < (p_stay if state == 1 else q_return)
            if state:
                games.append(g)
        rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": player, "TEAM_ID": 1,
                  "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}",
                  "MIN": 20.0} for g in games]
    panel = _panel(rows, tmp_path)
    out = pd.DataFrame(serial_structure(panel, "appearance"))
    got = out.set_index("metric")["value"]
    assert abs(got["p_play_given_played"] - p_stay) < 0.05
    assert abs(got["q_play_given_missed"] - q_return) < 0.08
    # (1+rho)/(1-rho) with rho = p - q
    rho = p_stay - q_return
    assert abs(got["clustering_variance_inflation"] - (1 + rho) / (1 - rho)) < 0.6


def test_serial_structure_reports_the_geometric_null_beside_the_observed_spells(tmp_path):
    """The chain's falsifiable prediction: constant hazard ⇒ geometric spells. Both
    numbers ship together so the comparison cannot be skipped."""
    rows = _team_season(1, 28)
    for player in range(2, 20):                 # every player misses alternate games
        rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": player, "TEAM_ID": 1,
                  "GAME_ID": f"1{g:03d}", "GAME_DATE": f"2021-11-{1 + g:02d}",
                  "MIN": 20.0} for g in range(0, 28, 2)]
    panel = _panel(rows, tmp_path)
    got = pd.DataFrame(serial_structure(panel, "appearance")).set_index("metric")["value"]
    assert "share_single_geometric" in got.index
    assert got["share_single_observed"] == 1.0        # every absence is one game here


# ── Playoff workload: a feature of S-1, never a row to fit ────────────────────

def _playoff_log(tmp_path, season: str, rows: list[dict]):
    from src.data.fetch import _slug
    pd.DataFrame(rows).to_csv(
        tmp_path / f"game_logs_playoffs_{_slug(season)}.csv", index=False)


def _po_rows(player_id: int, n_games: int, minutes: float, season: str = "2021-22"):
    return [{"SEASON_YEAR": season, "PLAYER_ID": player_id, "TEAM_ID": 1,
             "GAME_ID": f"42{g:04d}", "GAME_DATE": f"2022-05-{g + 1:02d}",
             "MIN": minutes}
            for g in range(n_games)]


def test_playoff_workload_aggregates_games_and_minutes(tmp_path):
    _playoff_log(tmp_path, "2021-22", _po_rows(1, 12, 34.0) + _po_rows(2, 4, 8.0))
    out = playoff_workload(["2021-22"], tmp_path).set_index("player_id")
    assert out.loc[1, "playoff_games"] == 12
    assert np.isclose(out.loc[1, "playoff_minutes"], 12 * 34.0)
    assert np.isclose(out.loc[1, "playoff_mpg"], 34.0)
    assert out.loc[2, "playoff_games"] == 4


def test_playoff_workload_omits_non_participants_rather_than_zeroing_them(tmp_path):
    """"No playoff appearance" is a fact, not a missing measurement — but the caller
    decides that, so the frame must not silently invent rows."""
    _playoff_log(tmp_path, "2021-22", _po_rows(1, 5, 20.0))
    out = playoff_workload(["2021-22"], tmp_path)
    assert set(out["player_id"]) == {1}


def test_missing_playoff_logs_raise_rather_than_zero_filling(tmp_path):
    """Zeros here would be indistinguishable from a league where nobody made the
    playoffs — the same collapse `not_rostered` vs `unknown` exists to prevent."""
    _write_log(tmp_path, "2021-22", _team_season(1, 5))
    with pytest.raises(FileNotFoundError, match="make fetch"):
        playoff_workload(["2021-22"], tmp_path)


def test_playoff_workload_is_empty_when_the_requested_season_has_no_rows(tmp_path):
    """A file that exists but covers other seasons is a legitimate empty, not an error."""
    _playoff_log(tmp_path, "2022-23", _po_rows(1, 5, 20.0, season="2022-23"))
    out = playoff_workload(["2021-22"], tmp_path)
    assert out.empty
    assert "playoff_games" in out.columns


def test_playoff_workload_ignores_seasons_outside_the_request(tmp_path):
    _playoff_log(tmp_path, "2021-22", _po_rows(1, 5, 20.0))
    _playoff_log(tmp_path, "2022-23", _po_rows(1, 9, 30.0, season="2022-23"))
    out = playoff_workload(["2021-22"], tmp_path)
    assert set(out["season"]) == {"2021-22"}


def _season_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_attach_workload_fills_non_participants_with_zero():
    frame = _season_frame([
        {"window": "full", "season": "2021-22", "player_id": 1, "total_minutes": 2000.0},
        {"window": "full", "season": "2021-22", "player_id": 2, "total_minutes": 1000.0},
    ])
    playoffs = pd.DataFrame([{"season": "2021-22", "player_id": 1,
                              "playoff_games": 10, "playoff_minutes": 350.0,
                              "playoff_mpg": 35.0}])
    out = attach_workload(frame, playoffs, ["2021-22"]).set_index("player_id")
    assert out.loc[2, "playoff_games"] == 0
    assert out.loc[2, "playoff_minutes"] == 0.0
    assert out.loc[2, "made_playoffs"] == 0
    assert out.loc[1, "made_playoffs"] == 1


def test_total_minutes_incl_playoffs_adds_the_two():
    frame = _season_frame([{"window": "full", "season": "2021-22", "player_id": 1,
                            "total_minutes": 2000.0}])
    playoffs = pd.DataFrame([{"season": "2021-22", "player_id": 1, "playoff_games": 10,
                              "playoff_minutes": 350.0, "playoff_mpg": 35.0}])
    out = attach_workload(frame, playoffs, ["2021-22"]).iloc[0]
    assert np.isclose(out["total_minutes_incl_playoffs"], 2350.0)
    assert np.isclose(out["playoff_minutes_share"], 350.0 / 2350.0)


def test_playoff_minutes_share_is_zero_not_nan_for_a_zero_minute_season():
    frame = _season_frame([{"window": "full", "season": "2021-22", "player_id": 1,
                            "total_minutes": 0.0}])
    out = attach_workload(frame, pd.DataFrame(
        columns=["season", "player_id", "playoff_games", "playoff_minutes",
                 "playoff_mpg"]), ["2021-22"]).iloc[0]
    assert out["playoff_minutes_share"] == 0.0


def test_career_minutes_accumulates_in_season_order_including_playoffs():
    seasons = ["2019-20", "2020-21", "2021-22"]
    frame = _season_frame([{"window": "full", "season": s, "player_id": 1,
                            "total_minutes": 1000.0} for s in seasons])
    playoffs = pd.DataFrame([{"season": "2020-21", "player_id": 1, "playoff_games": 5,
                              "playoff_minutes": 200.0, "playoff_mpg": 40.0}])
    out = attach_workload(frame, playoffs, seasons).sort_values("season")
    assert list(out["career_minutes"]) == [1000.0, 2200.0, 3200.0]
    assert list(out["career_seasons"]) == [1, 2, 3]


def test_career_minutes_does_not_leak_across_the_two_windows():
    """Each window is an independent construction; a cumsum spanning both would double."""
    rows = []
    for w in ("appearance", "full"):
        for s in ["2020-21", "2021-22"]:
            rows.append({"window": w, "season": s, "player_id": 1,
                         "total_minutes": 1000.0})
    out = attach_workload(_season_frame(rows), pd.DataFrame(
        columns=["season", "player_id", "playoff_games", "playoff_minutes",
                 "playoff_mpg"]), ["2020-21", "2021-22"])
    for w in ("appearance", "full"):
        got = out[out["window"] == w].sort_values("season")["career_minutes"]
        assert list(got) == [1000.0, 2000.0]


def test_career_minutes_respects_season_order_not_row_order():
    seasons = ["2019-20", "2020-21", "2021-22"]
    frame = _season_frame([{"window": "full", "season": s, "player_id": 1,
                            "total_minutes": m}
                           for s, m in zip(reversed(seasons), [300.0, 200.0, 100.0])])
    out = attach_workload(frame, pd.DataFrame(
        columns=["season", "player_id", "playoff_games", "playoff_minutes",
                 "playoff_mpg"]), seasons).sort_values("season")
    assert list(out["career_minutes"]) == [100.0, 300.0, 600.0]


# ── Playoff scope: the sign flip that keeps playoffs out of the fitting frame ──

def _scope_inputs(specs: list[tuple[int, float, float, int]], tmp_path,
                  team_id: int = 1, n_games: int = 25):
    """Build (panel, frame, playoffs) for players given as
    (player_id, regular mpg, playoff mpg, playoff games).

    A playoff mpg of 0 with 0 games means he never appeared, which is what the appearance
    rate's denominator has to keep.
    """
    dates = pd.date_range("2021-11-01", periods=n_games).strftime("%Y-%m-%d")
    rows = []
    for pid, reg_mpg in [(900 + team_id, 30.0)] + [(p, m) for p, m, _, _ in specs]:
        rows += [{"SEASON_YEAR": "2021-22", "PLAYER_ID": pid, "TEAM_ID": team_id,
                  "GAME_ID": f"{team_id}{g:03d}", "GAME_DATE": dates[g],
                  "MIN": reg_mpg} for g in range(n_games)]
    panel = _panel(rows, tmp_path)
    frame = season_availability(panel, "full")
    playoffs = pd.DataFrame([
        {"season": "2021-22", "player_id": pid, "playoff_games": games,
         "playoff_minutes": po_mpg * games, "playoff_mpg": po_mpg}
        for pid, _, po_mpg, games in specs if games > 0])
    if playoffs.empty:
        playoffs = pd.DataFrame(columns=["season", "player_id", "playoff_games",
                                         "playoff_minutes", "playoff_mpg"])
    return panel, frame, playoffs


def _scope_table(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows).set_index(["key", "metric"])["value"]


def test_playoff_scope_recovers_the_ratio_per_role_bucket(tmp_path):
    """A bench player halved, a rotation player trimmed, a starter given more."""
    specs = [(1, 8.0, 4.0, 10), (2, 18.0, 14.0, 10), (3, 32.0, 36.0, 10)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    got = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)}))
    assert abs(got[("bench_lt12", "median_playoff_to_regular_mpg")] - 0.5) < 1e-9
    assert abs(got[("rotation_12_24", "median_playoff_to_regular_mpg")]
               - 14.0 / 18.0) < 1e-9
    assert got[("starter_24plus", "median_playoff_to_regular_mpg")] > 1.0


def test_the_buckets_partition_the_qualifying_population(tmp_path):
    """The verification: a player in no bucket would vanish from every row silently."""
    specs = [(1, 8.0, 4.0, 10), (2, 18.0, 14.0, 10), (3, 32.0, 36.0, 10),
             (4, 12.0, 6.0, 10), (5, 24.0, 24.0, 10)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    rows = playoff_scope(panel, frame, playoffs, {("2021-22", 1)})
    got = _scope_table(rows)
    assert got[("all", "unbucketed_rows")] == 0.0
    per_bucket = sum(got[(k, "n_with_playoff_appearance")]
                     for k in ("bench_lt12", "rotation_12_24", "starter_24plus"))
    assert per_bucket == got[("all", "n_with_playoff_appearance")]


def test_the_bucket_edges_are_left_inclusive_so_12_and_24_move_up(tmp_path):
    specs = [(4, 12.0, 6.0, 10), (5, 24.0, 24.0, 10)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    got = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)}))
    assert got[("rotation_12_24", "n_with_playoff_appearance")] == 1.0
    assert got[("starter_24plus", "n_with_playoff_appearance")] == 1.0


def test_the_appearance_rate_keeps_players_who_never_appeared(tmp_path):
    """Two bench players on a playoff team, one of whom never dressed."""
    specs = [(1, 8.0, 4.0, 10), (2, 8.0, 0.0, 0)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    got = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)}))
    assert got[("bench_lt12", "n_on_playoff_teams")] == 2.0
    assert abs(got[("bench_lt12", "playoff_appearance_rate")] - 0.5) < 1e-9
    # ...but the ratio is over appearances only, since a ratio needs a numerator.
    assert got[("bench_lt12", "n_with_playoff_appearance")] == 1.0


def test_players_on_a_non_playoff_team_are_excluded_from_the_appearance_rate(tmp_path):
    """"His team missed the playoffs" is not "he did not dress"."""
    specs = [(1, 8.0, 0.0, 0)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    got = _scope_table(playoff_scope(panel, frame, playoffs, set()))
    assert ("bench_lt12", "playoff_appearance_rate") not in got.index
    assert _scope_table(playoff_scope(panel, frame, playoffs,
                                      {("2021-22", 1)}))[
        ("bench_lt12", "n_on_playoff_teams")] == 1.0


def test_the_regular_games_floor_drops_player_seasons_with_no_role(tmp_path):
    specs = [(1, 30.0, 30.0, 10)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path, n_games=40)
    kept = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)},
                                      min_games=20))
    dropped = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)},
                                         min_games=60))
    assert kept[("all", "n_regular_seasons_ge_min_games")] > 0
    assert dropped[("all", "n_regular_seasons_ge_min_games")] == 0.0


def test_share_playing_more_is_a_share_of_the_appearing_players(tmp_path):
    specs = [(1, 30.0, 36.0, 10), (2, 30.0, 24.0, 10)]
    panel, frame, playoffs = _scope_inputs(specs, tmp_path)
    got = _scope_table(playoff_scope(panel, frame, playoffs, {("2021-22", 1)}))
    assert abs(got[("starter_24plus", "share_playing_more_in_playoffs")] - 0.5) < 1e-9
