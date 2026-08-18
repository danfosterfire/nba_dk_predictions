import pandas as pd
from src.data.fetch import nbastats_dir

from src.eda.availability import decomposition
from src.data.boxscore_status import (
    STATUS_COLS,
    classify_comment,
    load_status,
    pad_game_id,
    season_game_ids,
    uses_v3,
)
from src.features.availability import (
    MISSED_KINDS,
    attach_status,
    build_season_panel,
    missed_decomposition,
    season_availability,
)

SEASON = "2021-22"


# ── Synthetic builders ────────────────────────────────────────────────────────

def _team_season(team_id: int, n_games: int) -> list[dict]:
    """The schedule skeleton: one anchor player who plays every game."""
    return [{"SEASON_YEAR": SEASON, "PLAYER_ID": 900 + team_id, "TEAM_ID": team_id,
             "GAME_ID": 1000 * team_id + g, "GAME_DATE": f"2021-11-{1 + g:02d}",
             "MIN": 30.0}
            for g in range(n_games)]


def _appearances(player_id: int, team_id: int, games: list[int]) -> list[dict]:
    return [{"SEASON_YEAR": SEASON, "PLAYER_ID": player_id, "TEAM_ID": team_id,
             "GAME_ID": 1000 * team_id + g, "GAME_DATE": f"2021-11-{1 + g:02d}",
             "MIN": 20.0}
            for g in games]


def _status_rows(team_id: int, player_id: int,
                 per_game: dict[int, tuple[str, str]]) -> pd.DataFrame:
    rows = [{"season": SEASON, "game_id": pad_game_id(1000 * team_id + g),
             "team_id": team_id, "player_id": player_id, "player_name": "P",
             "status": status, "comment": comment,
             "reason": classify_comment(comment), "start_position": "", "min": ""}
            for g, (status, comment) in per_game.items()]
    return pd.DataFrame(rows, columns=STATUS_COLS)


def _write_log(tmp_path, rows: list[dict]):
    df = pd.DataFrame(rows)
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    dest = nbastats_dir(tmp_path)
    dest.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest / f"game_logs_{SEASON.replace('-', '_')}.csv", index=False)


# ── The comment vocabulary ────────────────────────────────────────────────────

def test_classify_comment_covers_the_stated_vocabulary():
    assert classify_comment("DNP - Coach's Decision") == "coach"
    assert classify_comment("DND - Injury/Illness") == "injury"
    assert classify_comment("NWT - League Suspension") == "suspension"
    assert classify_comment("DNP - G League - Two-Way") == "gleague"
    assert classify_comment("DNP - Personal Reasons") == "personal"


def test_classify_comment_handles_the_older_free_text_form():
    """Pre-2010 seasons write the body part inline with no space after DND."""
    assert classify_comment("DND- Right Quad Strain") == "injury"


def test_an_inactive_player_has_no_stated_reason():
    """`InactivePlayers` publishes no reason. Guessing one would fabricate exactly the
    label the decomposition retest is meant to measure."""
    assert classify_comment("") == ""
    assert classify_comment(None) == ""


def test_endpoint_routing_switches_at_2025_26():
    assert not uses_v3("2024-25")
    assert uses_v3("2025-26")


def test_pad_game_id_matches_the_api_form():
    assert pad_game_id(22300061) == "0022300061"
    assert pad_game_id("0022300061") == "0022300061"


# ── Status attachment ─────────────────────────────────────────────────────────

def test_attach_status_separates_not_rostered_from_unknown(tmp_path):
    """The distinction a half-finished backfill depends on: inside a covered game an
    absent box-score row means "not on the roster"; outside one it means nothing."""
    rows = _team_season(1, 6) + _appearances(7, 1, [0, 1])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)

    # Only games 0-2 are backfilled. Player 7 played 0 and 1, was inactive for 2.
    status = _status_rows(1, 7, {0: ("played", ""), 1: ("played", ""),
                                 2: ("inactive", "")})
    panel = attach_status(panel.drop(columns=["status", "missed_reason",
                                              "status_covered"]), status)
    mine = (panel[panel.player_id == 7]
            .set_index("team_game_index")["status"].to_dict())
    assert mine[0] == "played"
    assert mine[2] == "inactive"
    # Games 3-5 are not in the backfill at all — unknown, not "nobody rostered him".
    assert mine[3] == mine[4] == mine[5] == "unknown"


def test_a_player_missing_from_a_covered_game_is_not_rostered(tmp_path):
    rows = _team_season(1, 4) + _appearances(7, 1, [0])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)
    status = _status_rows(1, 7, {0: ("played", "")})
    # Game 1 is covered (the anchor player has a row there) but player 7 does not.
    status = pd.concat([status, _status_rows(1, 901, {1: ("played", "")})],
                       ignore_index=True)
    panel = attach_status(panel.drop(columns=["status", "missed_reason",
                                              "status_covered"]), status)
    mine = panel[panel.player_id == 7].set_index("team_game_index")["status"].to_dict()
    assert mine[1] == "not_rostered"
    assert mine[2] == "unknown"


def test_the_game_log_wins_over_the_box_score_comment(tmp_path):
    """A player with minutes played, whatever the comment field says."""
    rows = _team_season(1, 3) + _appearances(7, 1, [0])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)
    status = _status_rows(1, 7, {0: ("dnp", "DNP - Coach's Decision")})
    panel = attach_status(panel.drop(columns=["status", "missed_reason",
                                              "status_covered"]), status)
    row = panel[(panel.player_id == 7) & (panel.team_game_index == 0)].iloc[0]
    assert row["played"] == 1 and row["status"] == "played"


def test_status_survives_a_string_versus_int_game_id(tmp_path):
    """The game logs store game ids as ints and the box-score files as zero-padded
    strings, so an un-normalized merge matches nothing — silently, and still returns a
    full panel."""
    rows = _team_season(1, 3) + _appearances(7, 1, [0])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)
    status = _status_rows(1, 7, {1: ("inactive", "")})
    assert status["game_id"].iloc[0] == "0000001001"      # padded string
    out = attach_status(panel.drop(columns=["status", "missed_reason",
                                            "status_covered"]), status)
    assert (out["status_covered"] == 1).any()
    assert (out["status"] == "inactive").sum() == 1


# ── Decomposition ─────────────────────────────────────────────────────────────

def test_missed_decomposition_partitions_missed_games(tmp_path):
    rows = _team_season(1, 6) + _appearances(7, 1, [0, 1])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)
    status = _status_rows(1, 7, {0: ("played", ""), 1: ("played", ""),
                                 2: ("inactive", ""),
                                 3: ("dnp", "DNP - Coach's Decision"),
                                 4: ("dnp", "DND - Injury/Illness")})
    panel = attach_status(panel.drop(columns=["status", "missed_reason",
                                              "status_covered"]), status)
    out = missed_decomposition(panel)
    mine = out[out.player_id == 7].iloc[0]
    assert mine["missed_inactive"] == 1
    assert mine["missed_scratch"] == 1
    assert mine["missed_injury"] == 1
    assert mine["missed_unknown"] == 1               # game 5 is not backfilled
    buckets = sum(mine[f"missed_{k}"] for k in MISSED_KINDS)
    assert buckets == 4                              # every missed game lands in one


def test_season_availability_carries_the_decomposition_and_its_coverage(tmp_path):
    rows = _team_season(1, 6) + _appearances(7, 1, [0, 1])
    _write_log(tmp_path, rows)
    panel = build_season_panel(SEASON, tmp_path)
    frame = season_availability(panel, "full")
    mine = frame[frame.player_id == 7].iloc[0]
    # No backfill on disk for this synthetic season: everything is unknown, coverage 0.
    assert mine["status_coverage"] == 0.0
    assert mine["missed_unknown"] == mine["missed_games"] == 4


def test_load_status_dedupes_a_doubly_appended_chunk(tmp_path):
    """A hard kill can leave a flush appended twice; a player has one status per game."""
    status = _status_rows(1, 7, {0: ("played", ""), 1: ("inactive", "")})
    nbastats_dir(tmp_path).mkdir(parents=True, exist_ok=True)
    dest = nbastats_dir(tmp_path) / "boxscore_status_2021_22.csv"
    pd.concat([status, status], ignore_index=True).to_csv(dest, index=False)
    assert len(load_status(SEASON, tmp_path)) == 2


def test_season_game_ids_reads_regular_season_and_playoffs(tmp_path):
    _write_log(tmp_path, _team_season(1, 3))
    playoffs = pd.DataFrame(_team_season(1, 2))
    playoffs["GAME_ID"] = [40000001, 40000002]
    playoffs.to_csv(nbastats_dir(tmp_path) / "game_logs_playoffs_2021_22.csv", index=False)
    ids = season_game_ids(SEASON, tmp_path)
    assert "0040000001" in ids and "0000001000" in ids
    assert len(ids) == 5


# ── The reason-decomposition retest ───────────────────────────────────────────

SEASONS_4 = ["2019-20", "2020-21", "2021-22", "2022-23"]


def _decomposition_frame(n_players: int = 400, seed: int = 0,
                         persistent: bool = False,
                         coverage: float = 1.0) -> pd.DataFrame:
    """Player-seasons with two causes of absence that behave differently over time.

    This is the structure the real labels have, and the reason an aggregate
    `missed_games` cannot substitute for the split: a player's *scratch* tendency is a
    persistent property of his rotation status, while an *injury* is a transient shock.
    Prior `gp_share` mixes the two together, so knowing which one it was carries
    information the total does not. With `persistent=False` neither cause carries over
    and the split has nothing to find.
    """
    import numpy as np
    rng = np.random.default_rng(seed)
    prone = rng.uniform(0, 1, size=n_players)          # persistent rotation standing
    rows = []
    for season in SEASONS_4:
        for player in range(n_players):
            latent = prone[player] if persistent else rng.uniform(0, 1)
            scratch = int(np.clip(rng.normal(20 * latent, 2), 0, 30))
            injury = int(np.clip(rng.normal(20 * rng.uniform(0, 1), 2), 0, 30))
            gp_share = float(np.clip(1.0 - 0.02 * (scratch + injury)
                                     + rng.normal(0, 0.05), 0.02, 1.0))
            row = {"season": season, "player_id": player, "gp_share": gp_share,
                   "minutes_per_game": 24.0, "total_minutes": 1600.0,
                   "missed_games": scratch + injury, "status_coverage": coverage,
                   "missed_scratch": scratch, "missed_injury": injury}
            for col in ["missed_inactive", "missed_gleague", "missed_not_rostered",
                        "missed_suspension", "missed_personal", "missed_other"]:
                row[col] = int(rng.integers(0, 10))
            rows.append(row)
    return pd.DataFrame(rows)


def _incremental(rows: list[dict]) -> dict:
    out = pd.DataFrame(rows)
    return out[out.key == "incremental"].set_index("metric")["value"].to_dict()


def test_decomposition_refuses_to_answer_without_coverage():
    """A thin result here would be a statement about how far the backfill has run. The
    aggregate version of this null has already been mistaken for a finding once."""
    frame = _decomposition_frame(coverage=0.1)
    out = pd.DataFrame(decomposition(frame, SEASONS_4, "full"))
    assert (out.metric == "insufficient_coverage").any()
    assert not (out.metric == "r2_plus_reason_split").any()


def test_decomposition_reports_a_split_with_its_own_chance_level():
    """Adding 8 columns to a 1-column model raises in-sample R² on noise alone, so the
    split is worthless without the shuffled null beside it."""
    rows = decomposition(_decomposition_frame(persistent=False), SEASONS_4, "full",
                         n_shuffles=10)
    inc = _incremental(rows)
    above = inc["r2_reason_split_above_null"]
    assert inc["r2_plus_reason_split"] > inc["r2_prior_gp_share"]   # rises regardless
    assert abs(above) < 3 * inc["r2_reason_split_null_sd"]          # but not above chance


def test_decomposition_detects_a_reason_that_really_predicts():
    rows = decomposition(_decomposition_frame(persistent=True), SEASONS_4, "full",
                         n_shuffles=10)
    inc = _incremental(rows)
    assert inc["r2_reason_split_above_null"] > 3 * inc["r2_reason_split_null_sd"]


def test_decomposition_drops_a_constant_reason_column():
    """`missed_gleague` is identically zero in the backfilled seasons; a constant column
    only spends a degree of freedom against the null."""
    frame = _decomposition_frame(persistent=True)
    frame["missed_gleague"] = 0
    inc = _incremental(decomposition(frame, SEASONS_4, "full", n_shuffles=5))
    assert inc["n_reason_columns"] == 7
