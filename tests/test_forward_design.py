"""The forward design — building a season's rows before that season has been played.

The properties pinned here are the ones that would produce a *plausible* wrong board
rather than an error: a denominator taken from a schedule that is quietly short, a forward
row carrying a target it could be fitted on, and a roster snapshot used for a played season
as though it were a pre-opener one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features import forward_design as fd


def test_the_acquired_date_is_parsed_from_the_free_text():
    assert fd.acquired_on("Signed on 12/22/23") == pd.Timestamp("2023-12-22")
    assert fd.acquired_on("Traded from SAS on 02/10/22") == pd.Timestamp("2022-02-10")
    # A draft-pick row carries a year and no date, and is always pre-opener.
    assert fd.acquired_on("#20 Pick in 2021 Draft") is None
    assert fd.acquired_on(None) is None
    assert fd.acquired_on(float("nan")) is None


def test_the_cup_placeholder_does_not_become_a_short_denominator(tmp_path):
    """🔴 The defect this exists for: the published schedule for an UNPLAYED season lists
    80 games per team, not 82, because the NBA Cup knockout opponents are still TBD and
    ride against a placeholder id. `team_games` is the availability head's binomial
    denominator, so 80 would run every player's rate against 2 too few opportunities —
    invisible, and straight into every season total.
    """
    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True)
    # Four teams, every pair meeting 26 times: 78 games each, short of 82 exactly the way
    # the real published schedule is short of it.
    teams = [1610612737 + i for i in range(4)]
    rows = []
    for i, home in enumerate(teams):
        for j, away in enumerate(teams):
            if j <= i:
                continue
            for k in range(26):
                rows.append({"game_id": f"00226{i}{j}{k:02d}", "date": "2026-10-20",
                             "home_team_id": home if k % 2 else away,
                             "away_team_id": away if k % 2 else home,
                             "game_label": ""})
    # The Cup knockout, as the league actually publishes it: six games with BOTH sides
    # TBD, which is why no franchise's count betrays the shortfall.
    for k in range(6):
        rows.append({"game_id": f"0022699{k:02d}", "date": "2026-12-16",
                     "home_team_id": 0, "away_team_id": 0,
                     "game_label": "Emirates NBA Cup"})
    pd.DataFrame(rows).to_csv(raw / "schedule_teams_2026_27.csv", index=False)

    counts, record = fd.schedule_team_games("2026-27", tmp_path)
    assert record["n_placeholder_sides"] == 12      # six games, both sides TBD
    assert record["raw_games_each"] == [78]          # every franchise looks consistent
    assert record["n_franchises"] == 4
    assert record["correction"] > 0, "a short schedule must be corrected, not inherited"
    assert set(counts.unique()) == {fd.SEASON_GAMES}
    assert 0 not in counts.index, "the TBD placeholder is not a team"


def test_a_forward_row_carries_no_target():
    """The sentinel that gets a forward row past the shared builder's `dropna` must not
    survive into the frame. A row with a plausible-looking `gp = 0` is a row something
    could fit on, and it would look exactly like a player who missed the whole season."""
    assert "gp" in fd.TARGET_COLS and "gp_share" in fd.TARGET_COLS
    assert "total_minutes" in fd.TARGET_COLS


def test_the_snapshot_cut_is_a_rehearsal_knob_not_a_production_one(tmp_path):
    """`before` exists so a played season can be rehearsed honestly. For a real forward
    season the snapshot already precedes the opener and there is nothing to cut."""
    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True)
    pd.DataFrame({"PLAYER_ID": [1, 2, 3], "TeamID": [10, 10, 11],
                  "HOW_ACQUIRED": ["#3 Pick in 2024 Draft", "Signed on 12/22/23",
                                   "Traded from SAS on 02/10/22"]}
                 ).to_csv(raw / "team_rosters_2023_24.csv", index=False)

    every = fd.roster_members("2023-24", tmp_path)
    assert len(every) == 3
    cut = fd.roster_members("2023-24", tmp_path, before=pd.Timestamp("2023-10-24"))
    assert set(cut["player_id"]) == {1, 3}, "only the mid-season arrival is removed"


def test_the_missing_roster_file_names_the_admissible_source(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        fd.roster_members("2026-27", tmp_path)
    assert "season_start_roster" in str(excinfo.value)


def test_played_is_not_needed_to_attribute_a_forward_roster(tmp_path):
    """🔴 `roster_grid` picked a traded player's team from his last APPEARANCE, and its
    emptiness check runs *before* that filter — so a forward panel, where nothing has been
    played, returned an EMPTY grid rather than raising. A season that simulates nothing and
    says nothing is the worst available failure.

    In a played season the fallback is a no-op: the panel's (player, team) pairs come from
    the game log, so every pair carries at least one appearance.
    """
    from src.sim.season import roster_grid

    games = pd.DataFrame({"game_id": [f"002260{i:04d}" for i in range(4)],
                          "game_date": pd.to_datetime(
                              ["2026-10-20", "2026-10-22", "2026-10-24", "2026-10-26"]),
                          "team_game_index": [0, 1, 2, 3]})
    panel = pd.concat([games.assign(player_id=p, team_id=10, season="2026-27", played=0)
                       for p in (1, 2, 3)], ignore_index=True)
    panel.to_parquet(tmp_path / "availability_panel.parquet")

    slots = pd.DataFrame({"game_id": games["game_id"], "slot": [0, 0, 1, 1]})
    grid = roster_grid(tmp_path, "2026-27", slots)
    assert len(grid) == 12, "every rostered player keeps every one of his team's games"
    assert set(grid["player_id"]) == {1, 2, 3}
    assert "played" not in grid.columns, "the grid is a schedule, not an outcome"


def test_an_unattributable_player_raises_instead_of_vanishing(tmp_path):
    """The fallback is only safe because a forward roster gives each player one team. If
    one ever sat on two with nothing played, picking either would be a guess."""
    from src.sim.season import roster_grid

    rows = []
    for team in (10, 11):
        for i in range(2):
            rows.append({"game_id": f"00226{team}{i:03d}",
                         "game_date": pd.Timestamp("2026-10-20") + pd.Timedelta(days=i),
                         "team_game_index": i, "player_id": 1, "team_id": team,
                         "season": "2026-27", "played": 0})
    pd.DataFrame(rows).to_parquet(tmp_path / "availability_panel.parquet")

    slots = pd.DataFrame({"game_id": [r["game_id"] for r in rows], "slot": 0})
    with pytest.raises(ValueError) as excinfo:
        roster_grid(tmp_path, "2026-27", slots)
    assert "undecidable" in str(excinfo.value)


def test_primary_team_is_keyed_on_the_season_not_the_player():
    """🔴 The bug this exists for. The fallback checked membership by `player_id` alone,
    so a veteran with appearances in EARLIER seasons was found in the attribution table and
    skipped — leaving exactly the rows that carry lagged features unattributed. It was 496
    of 577 players on 2026-27, and it presented as a NaN binomial denominator rather than
    as an error.
    """
    from src.features.availability import primary_team

    rows = [
        # A veteran who played in 2025-26 and is rostered, unplayed, in 2026-27.
        {"season": "2025-26", "player_id": 1, "team_id": 10, "played": 1,
         "game_date": pd.Timestamp("2025-11-01"), "game_id": "a"},
        {"season": "2026-27", "player_id": 1, "team_id": 11, "played": 0,
         "game_date": pd.Timestamp("2026-11-01"), "game_id": "b"},
        # A rookie who has never played at all.
        {"season": "2026-27", "player_id": 2, "team_id": 11, "played": 0,
         "game_date": pd.Timestamp("2026-11-01"), "game_id": "b"},
    ]
    out = primary_team(pd.DataFrame(rows)).set_index(["season", "player_id"])["team_id"]
    assert out[("2025-26", 1)] == 10, "the played season keeps its appearance rule"
    assert out[("2026-27", 1)] == 11, "the veteran's FORWARD season gets his new team"
    assert out[("2026-27", 2)] == 11


def test_a_played_season_still_attributes_a_traded_player_by_last_appearance():
    """The fallback must not change the rule where appearances exist."""
    from src.features.availability import primary_team

    rows = [{"season": "2023-24", "player_id": 1, "team_id": 10, "played": 1,
             "game_date": pd.Timestamp("2023-11-01"), "game_id": "a"},
            {"season": "2023-24", "player_id": 1, "team_id": 11, "played": 1,
             "game_date": pd.Timestamp("2024-02-01"), "game_id": "b"},
            {"season": "2023-24", "player_id": 1, "team_id": 11, "played": 0,
             "game_date": pd.Timestamp("2024-03-01"), "game_id": "c"}]
    out = primary_team(pd.DataFrame(rows))
    assert len(out) == 1 and out["team_id"].iloc[0] == 11


def test_forward_age_comes_from_the_birth_date_not_the_roster_age_column(tmp_path):
    """🔴 `AGE` on the roster is a FETCH-TIME attribute. On a roster pulled during the
    season it matches the bio column 98.7% exactly; on the 2026-27 roster pulled in August
    it sits 0.617 years low against the season's own reference date. Age and age squared
    are features on every head, so the convenient column would push a systematic error
    through the whole board."""
    from src.eda.availability import AGE_REFERENCE, roster_ages

    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True)
    pd.DataFrame({"PLAYER_ID": [1, 2], "TeamID": [10, 10],
                  "BIRTH_DATE": ["JAN 01, 2000", "JUL 01, 2000"],
                  "AGE": [11.0, 12.0]}                    # deliberately absurd
                 ).to_csv(raw / "team_rosters_2026_27.csv", index=False)

    out = roster_ages("2026-27", tmp_path).set_index("player_id")["age"]
    month, day = AGE_REFERENCE
    assert abs(out[1] - 27.0) < 0.05, "age at the season's reference date, not the column"
    assert abs(out[2] - 26.5) < 0.05
    assert (month, day) == (12, 31)


def test_the_bio_file_still_wins_where_it_exists(tmp_path):
    """The fallback must not move a single played season's ages."""
    from src.eda.availability import load_ages

    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True)
    pd.DataFrame({"PLAYER_ID": [1], "AGE": [30.0]}).to_csv(
        raw / "player_bio_stats_2025_26.csv", index=False)
    pd.DataFrame({"PLAYER_ID": [1], "TeamID": [10], "BIRTH_DATE": ["JAN 01, 1990"],
                  "AGE": [99.0]}).to_csv(raw / "team_rosters_2025_26.csv", index=False)
    out = load_ages(["2025-26"], tmp_path)
    assert float(out["age"].iloc[0]) == 30.0


def _target_columns() -> list[str]:
    """Every column `season_totals` aggregates — the components and the volume family."""
    from src.features.targets import COMPONENTS
    from src.models.component_rates import volume_columns

    return list(dict.fromkeys(list(COMPONENTS) + list(volume_columns())))


def _toy_targets() -> pd.DataFrame:
    rows = []
    for player in (1, 2):
        for game in range(30):
            rows.append({c: 2.0 for c in _target_columns()} | {
                "player_id": player, "season": "2025-26", "season_type": "regular",
                "game_id": f"g{game}", "team_id": 10, "min": 30.0,
                "game_date": pd.Timestamp("2025-11-01") + pd.Timedelta(days=game)})
    return pd.DataFrame(rows)


def _toy_forward(players=(1, 2)) -> pd.DataFrame:
    return pd.DataFrame([{c: np.nan for c in _target_columns()} | {
        "player_id": p, "season": "2026-27", "season_type": "regular",
        "game_id": "z", "team_id": 10, "min": np.nan,
        "game_date": pd.Timestamp("2026-11-01")} for p in players])


def test_both_population_filters_default_to_dropping_forward_rows():
    """🔴 The guard, and it is the default rather than a check.

    A fitting path cannot receive forward rows without naming the season twice — once at
    `build_component_targets`, which drops blank-minute rows, and once at `build_design`,
    whose `total_minutes > 0` drops zero-minute seasons. Neither knob is passed anywhere
    else in the project.
    """
    from src.features.targets import build_component_targets

    forward = _toy_forward(players=(1,))
    kept = build_component_targets(pd.concat([_toy_targets(), forward],
                                             ignore_index=True))
    assert (kept["season"] == "2026-27").sum() == 0, "default drops the forward rows"
    # The marker is absent rather than zero: `component_targets` is 731,906 rows of played
    # basketball, and a column that is zero on all of them is a schema change for nothing.
    assert "is_forward" not in kept.columns

    named = build_component_targets(pd.concat([_toy_targets(), forward],
                                              ignore_index=True),
                                    forward_seasons=["2026-27"])
    assert (named["season"] == "2026-27").sum() == 1
    assert set(named.loc[named["season"] == "2026-27", "is_forward"]) == {1}


def test_clearing_only_one_filter_yields_nothing_rather_than_something_wrong(tmp_path):
    """Two stacked filters, and a half-applied change must fail empty rather than pass a
    frame whose population nobody chose."""
    from src.features.targets import build_component_targets
    from src.models.component_rates import build_design

    forward = _toy_forward()
    targets = build_component_targets(pd.concat([_toy_targets(), forward],
                                                ignore_index=True),
                                      forward_seasons=["2026-27"])
    seasons = ["2025-26", "2026-27"]
    # The second filter left in place: rows exist upstream and are dropped here.
    only_first = build_design(targets, seasons, tmp_path)
    assert (only_first["season"] == "2026-27").sum() == 0


def test_the_fitting_paths_never_ask_for_forward_rows():
    """Structural. The knob is safe because nothing that fits knows it exists."""
    import ast
    from pathlib import Path as _P

    for module in ("src/models/stan_components.py", "src/models/posteriors.py",
                   "src/models/component_rates.py", "src/features/targets.py"):
        tree = ast.parse(_P(module).read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            named = [kw.arg for kw in node.keywords if kw.arg == "forward_seasons"]
            assert not named, (
                f"{module} passes forward_seasons; only src/features/forward_design.py "
                f"may, or a fit could reach rows with no targets")


# ── The composition's per-player frame ────────────────────────────────────────

def _matrix_and_roster(tmp_path, matrix_rows, roster_rows):
    """The two draft-number sources, written where the builders read them."""
    features = tmp_path / "features"
    features.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix_rows, columns=["player_id", "season", "bio_draft_number"]
                 ).to_parquet(features / "season_matrix_roster_tierA.parquet")
    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(roster_rows).to_csv(raw / "team_rosters_2026_27.csv", index=False)
    return {"data": {"features_dir": str(features), "raw_dir": str(tmp_path),
                     "seasons": ["2025-26"]},
            "stan": {"composition": {"preseason": {"adopt": False}}}}


def test_forward_draft_numbers_read_history_first_and_roster_text_for_rookies(tmp_path):
    """The retrospective source (`bio_draft_number`) does not exist before the opener.
    A veteran's slot comes off his own earlier matrix rows — including PAST the target
    season's own row, which in a rehearsal is bio-derived and would grade the rule
    against itself — and a drafted rookie's off the roster's `#N Pick` text. A
    rights-traded rookie carries neither and lands unbucketed, priced by Part D."""
    cfg = _matrix_and_roster(
        tmp_path,
        # Player 1's real slot, and a target-season row that must NOT be read.
        [(1, "2024-25", 7.0), (1, "2026-27", 99.0)],
        [{"PLAYER_ID": 1, "TeamID": 10, "PLAYER": "Vet",
          "HOW_ACQUIRED": "Traded from BOS on 07/06/25"},
         {"PLAYER_ID": 2, "TeamID": 10, "PLAYER": "Rook",
          "HOW_ACQUIRED": "#40 Pick in 2026 Draft"},
         {"PLAYER_ID": 4, "TeamID": 11, "PLAYER": "Rights",
          "HOW_ACQUIRED": "Draft Rights Traded from DAL on 06/24/26"}])
    out = fd.forward_draft_numbers(cfg, "2026-27").set_index("player_id")
    assert out.loc[1, "draft_number"] == 7.0, "history wins, and only pre-target history"
    assert out.loc[2, "draft_number"] == 40.0, "the roster text carries the rookie's slot"
    assert np.isnan(out.loc[4, "draft_number"])
    assert out.loc[4, "draft_bucket"] == "undrafted"
    assert out.loc[2, "draft_bucket"] == "second_round"
    assert (out["season"] == "2026-27").all()


def test_appending_a_forward_season_leaves_every_played_weight_untouched(tmp_path):
    """The extraction's guard property: `season_weights` fed the forward season's rows
    must reproduce every played (player, season) weight exactly, because the same call
    sits under `composition_frame` and therefore under the fit."""
    from src.models.stan_composition import season_weights

    shares = pd.DataFrame({
        "player_id": [1, 1, 2],
        "season": ["2024-25", "2025-26", "2025-26"],
        "minutes_played": [1500.0, 1800.0, 400.0],
        "length_played": [3000.0, 3600.0, 1600.0],
        "games_played": [60, 70, 30],
        "minutes_share": [0.5, 0.5, 0.25]})
    drafts = pd.DataFrame({"player_id": [1, 2], "season": ["2024-25", "2025-26"],
                           "draft_number": [7.0, np.nan],
                           "draft_bucket": ["lottery", "undrafted"]})
    seasons = ["2024-25", "2025-26"]

    base, carried = season_weights(shares, seasons, tmp_path, drafts=drafts)

    forward = pd.concat([shares, pd.DataFrame({"player_id": [1, 3],
                                               "season": "2026-27"})],
                        ignore_index=True)
    fwd_drafts = pd.concat([drafts, pd.DataFrame(
        {"player_id": [1, 3], "season": "2026-27", "draft_number": [7.0, 12.0],
         "draft_bucket": ["lottery", "lottery"]})], ignore_index=True)
    extended, carried2 = season_weights(forward, seasons + ["2026-27"], tmp_path,
                                        drafts=fwd_drafts)

    assert carried == carried2
    keys = ["player_id", "season"]
    a = base[carried].sort_values(keys).reset_index(drop=True)
    b = (extended[extended["season"] != "2026-27"][carried]
         .sort_values(keys).reset_index(drop=True))
    pd.testing.assert_frame_equal(a, b)
    # And the forward rows themselves carry the lag: player 1's 2026-27 weight is his
    # 2025-26 share, exactly as a played 2026-27 row would have read it.
    fwd = extended[extended["season"] == "2026-27"].set_index("player_id")
    assert fwd.loc[1, "w_share"] == 0.5
    assert fwd.loc[1, "no_prior"] == 0.0


def test_forward_composition_players_builds_the_simulator_columns(tmp_path):
    """End to end on a synthetic league: the per-player frame carries `w_share`, the
    ordering keys and the design columns without reading a single target-season game —
    a veteran's weight is his lagged share, a rookie's is the expanding bucket prior
    (the fallback here, with no earlier no-prior observations to average), and a player
    outside the design keeps his row with NaN features for the recipe's impute step."""
    from src.models.availability import FEATURE_COLS as AV_COLS
    from src.models.stan_composition import FALLBACK_ROOKIE_SHARE, OWN

    cfg = _matrix_and_roster(
        tmp_path,
        [(1, "2025-26", 5.0)],
        [{"PLAYER_ID": 1, "TeamID": 10, "PLAYER": "Vet",
          "HOW_ACQUIRED": "Signed on 07/01/25"},
         {"PLAYER_ID": 3, "TeamID": 10, "PLAYER": "Rook",
          "HOW_ACQUIRED": "#12 Pick in 2026 Draft"}])
    features = Path(cfg["data"]["features_dir"])

    games = [f"002250{i:04d}" for i in range(4)]
    panel = pd.DataFrame({
        "season": "2025-26", "player_id": 1, "team_id": 10,
        "game_id": games, "min": 30.0, "played": 1})
    panel.to_parquet(features / "availability_panel.parquet")
    pd.DataFrame({"season": "2025-26", "game_id": games, "game_length": 48.0,
                  "n_overtimes": 0, "reliable": True, "season_type": "regular"}
                 ).to_parquet(features / "game_length.parquet")

    members = pd.DataFrame({"player_id": [1, 3]})
    design = pd.DataFrame({"player_id": [1], "season": ["2026-27"]}
                          | {c: [0.5] for c in AV_COLS})
    players = fd.forward_composition_players(cfg, "2026-27", members, design)

    players = players.set_index("player_id")
    assert players.loc[1, "w_share"] == 30.0 / 48.0
    assert players.loc[1, "no_prior"] == 0.0
    assert players.loc[3, "no_prior"] == 1.0
    assert players.loc[3, "w_share"] == FALLBACK_ROOKIE_SHARE
    assert players.loc[3, "draft_number"] == 12.0
    w = players["w_share"].to_numpy(float)
    assert np.allclose(players[OWN].to_numpy(float), np.log(w / (1 - w)))
    assert float(players.loc[1, "gp_share_lag1"]) == 0.5
    assert players.loc[3, AV_COLS].isna().all(), (
        "a player the design does not qualify keeps NaN features, not zeros")


def test_a_duplicated_member_raises_rather_than_double_weighting(tmp_path):
    cfg = _matrix_and_roster(tmp_path, [(1, "2025-26", 5.0)],
                             [{"PLAYER_ID": 1, "TeamID": 10, "PLAYER": "Vet",
                               "HOW_ACQUIRED": "Signed on 07/01/25"}])
    features = Path(cfg["data"]["features_dir"])
    pd.DataFrame({"season": ["2025-26"], "player_id": [1], "team_id": [10],
                  "game_id": ["0022500001"], "min": [30.0], "played": [1]}
                 ).to_parquet(features / "availability_panel.parquet")
    pd.DataFrame({"season": ["2025-26"], "game_id": ["0022500001"],
                  "game_length": [48.0], "n_overtimes": [0], "reliable": [True],
                  "season_type": ["regular"]}).to_parquet(features
                                                          / "game_length.parquet")
    members = pd.DataFrame({"player_id": [1, 1]})
    design = pd.DataFrame(columns=["player_id", "season"])
    with pytest.raises(ValueError) as excinfo:
        fd.forward_composition_players(cfg, "2026-27", members, design)
    assert "repeats a player_id" in str(excinfo.value)
