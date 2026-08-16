import numpy as np
import pandas as pd
import pytest

from src.data.fetch import _season_type_slug, _slug
from src.data.preprocess import (
    ALL_SEASON_TYPES,
    PLAYOFFS,
    PRE_SEASON,
    REGULAR_SEASON,
    _parse_log_filename,
    load_raw,
)
from src.features.preseason import (
    LATE_GAMES,
    build_season_panel,
    clean_rows,
    is_nba_team,
    read_preseason_log,
    season_coverage,
    team_schedule,
)

SEASON = "2021-22"
OPENER = pd.Timestamp("2021-10-19")

# Two NBA franchises and one exhibition opponent (a foreign club, outside the id block).
HOME, AWAY, FOREIGN = 1610612737, 1610612738, 12315


# ── Synthetic builders ────────────────────────────────────────────────────────

def _row(player_id: int, team_id: int, game: int, day: int, minutes: float,
         **box) -> dict:
    """One preseason log row in the raw CSV's upper-case schema."""
    rec = {"SEASON_YEAR": SEASON, "PLAYER_ID": player_id,
           "PLAYER_NAME": f"P{player_id}", "TEAM_ID": team_id,
           "GAME_ID": 10100000 + game, "GAME_DATE": f"2021-10-{day:02d}",
           "MIN": minutes, "FGA": 10.0, "FGM": 5.0, "FG3A": 4.0, "FG3M": 2.0,
           "FTA": 2.0, "FTM": 2.0, "REB": 5.0, "AST": 3.0, "STL": 1.0,
           "BLK": 1.0, "TOV": 2.0, "PTS": 14.0}
    rec.update({k.upper(): v for k, v in box.items()})
    return rec


def _write(tmp_path, rows: list[dict], season: str = SEASON) -> None:
    pd.DataFrame(rows).to_csv(
        tmp_path / f"game_logs_pre_season_{_slug(season)}.csv", index=False)


def _four_game_preseason() -> list[dict]:
    """HOME plays 4 preseason games. Player 1 plays all four, player 2 the first two."""
    rows = []
    for g, day in enumerate([4, 6, 10, 14]):
        rows.append(_row(1, HOME, g, day, 20.0))
        rows.append(_row(3, HOME, g, day, 10.0))   # a teammate, so shares are not 1.0
        if g < 2:
            rows.append(_row(2, HOME, g, day, 10.0))
    return rows


# ── The filename contract: a new season type must not become a pseudo-season ───

def test_pre_season_filename_parses_to_its_own_kind():
    """The failure this guards is silent: an unrecognized prefix returns REGULAR_SEASON
    with an invented season label, so the rows arrive in the DEFAULT fitting frame."""
    assert _parse_log_filename("game_logs_pre_season_2021_22") == (PRE_SEASON, "2021-22")
    assert _parse_log_filename("game_logs_playoffs_2021_22") == (PLAYOFFS, "2021-22")
    assert _parse_log_filename("game_logs_2021_22") == (REGULAR_SEASON, "2021-22")


def test_season_type_slug_folds_the_space():
    """'Pre Season' carries a space that `_slug` does not touch; a filename with a space
    in it cannot be classified by `_parse_log_filename` at all."""
    assert _season_type_slug("Pre Season") == "pre_season"
    assert _season_type_slug("Playoffs") == "playoffs"


def test_load_raw_all_excludes_preseason(tmp_path):
    """`all` means every type that can be a TARGET ROW. `src/features/game_length.py`
    reads it, and would otherwise absorb ~70 exhibition games a season."""
    reg = pd.DataFrame([_row(1, HOME, 0, 20, 30.0)])
    reg.to_csv(tmp_path / f"game_logs_{_slug(SEASON)}.csv", index=False)
    _write(tmp_path, [_row(9, HOME, 0, 4, 30.0)])

    for season_type in (REGULAR_SEASON, ALL_SEASON_TYPES):
        got = load_raw(tmp_path, season_type=season_type)
        assert set(got["season"]) == {SEASON}, "no pseudo-season may appear"
        assert set(got["season_type"]) == {REGULAR_SEASON}
        assert set(got["PLAYER_ID"]) == {1}

    pre = load_raw(tmp_path, season_type=PRE_SEASON)
    assert set(pre["PLAYER_ID"]) == {9}
    assert set(pre["season"]) == {SEASON}


def test_load_raw_rejects_an_unknown_season_type(tmp_path):
    _write(tmp_path, [_row(1, HOME, 0, 4, 30.0)])
    with pytest.raises(ValueError):
        load_raw(tmp_path, season_type="preseason")


# ── The three row filters ─────────────────────────────────────────────────────

def test_rows_after_the_opener_are_filtered(tmp_path):
    """2019-20's file carries the July 2020 bubble scrimmages under the `Pre Season`
    label — nine months AFTER its own opener. Reading them as a feature for 2019-20 is
    not a subtle leak, it is the season itself."""
    rows = _four_game_preseason()
    bubble = _row(1, HOME, 90, 4, 35.0)
    bubble["GAME_DATE"] = "2022-07-28"
    _write(tmp_path, rows + [bubble])

    kept = clean_rows(read_preseason_log(SEASON, tmp_path), OPENER)
    assert len(kept) == len(rows)
    assert kept["game_date"].max() < OPENER

    # With no opener on disk (the production case in October) nothing is date-filtered.
    assert len(clean_rows(read_preseason_log(SEASON, tmp_path), None)) == len(rows) + 1


def test_exhibition_opponents_are_dropped_but_their_games_are_kept(tmp_path):
    """An NBA team playing Real Madrid played a real preseason game. The opponent's own
    rows are not ours to model; the NBA team's rows in that game are."""
    rows = _four_game_preseason()
    rows += [_row(1, HOME, 4, 16, 25.0), _row(3, HOME, 4, 16, 25.0),
             _row(77, FOREIGN, 4, 16, 30.0)]
    _write(tmp_path, rows)

    kept = clean_rows(read_preseason_log(SEASON, tmp_path), OPENER)
    assert 77 not in set(kept["player_id"])
    assert not is_nba_team(pd.Series([FOREIGN])).iloc[0]
    assert team_schedule(kept)["team_pre_games"].max() == 5, \
        "the game against the foreign club still counts for the NBA team"


def test_rows_with_no_player_id_are_dropped(tmp_path):
    """2003-04 carries 96 team-total rows with MIN of exactly 480 (2 x 5 x 48)."""
    junk = _row(1, HOME, 0, 4, 480.0)
    junk["PLAYER_ID"] = None
    _write(tmp_path, _four_game_preseason() + [junk])

    kept = clean_rows(read_preseason_log(SEASON, tmp_path), OPENER)
    assert kept["min"].max() == 20.0
    assert kept["player_id"].notna().all()


# ── The panel ─────────────────────────────────────────────────────────────────

def test_minutes_shares_sum_to_one_within_a_team(tmp_path):
    _write(tmp_path, _four_game_preseason())
    panel = build_season_panel(SEASON, tmp_path, OPENER)
    assert panel.groupby("team_id")["min_share_pre"].sum().round(10).eq(1.0).all()


def test_missed_tail_reads_the_end_of_the_schedule(tmp_path):
    """The health reading the head is after: who sat the LAST games before the opener."""
    _write(tmp_path, _four_game_preseason())
    panel = build_season_panel(SEASON, tmp_path, OPENER).set_index("player_id")

    assert panel.loc[1, "missed_tail"] == 0
    assert panel.loc[1, "played_final_game"] == 1
    assert panel.loc[2, "missed_tail"] == 2, "player 2 played games 0 and 1 of 4"
    assert panel.loc[2, "played_final_game"] == 0
    assert panel.loc[2, "missed_lead"] == 0
    assert (panel["team_pre_games"] == 4).all()


def test_a_player_absent_from_the_late_games_gets_a_zero_share_not_a_null(tmp_path):
    """Zero IS the signal. A null would read as "no information" at the join."""
    _write(tmp_path, _four_game_preseason())
    panel = build_season_panel(SEASON, tmp_path, OPENER).set_index("player_id")

    assert panel.loc[2, "min_share_pre_late"] == 0.0
    assert panel.loc[2, "gp_pre_late"] == 0.0
    assert panel.loc[1, "min_share_pre_late"] > 0.0
    # …but his RANK among players who played stays null, because inventing a last place
    # would make "did not play" look like "played least".
    assert pd.isna(panel.loc[2, "min_rank_pre_late"])


def test_late_window_is_the_last_n_games_of_each_teams_own_schedule(tmp_path):
    """Teams play different numbers of preseason games, so the window cannot be a date."""
    rows = _four_game_preseason()
    rows += [_row(5, AWAY, 20 + g, day, 24.0) for g, day in enumerate([5, 9])]
    _write(tmp_path, rows)

    sched = team_schedule(clean_rows(read_preseason_log(SEASON, tmp_path), OPENER))
    late = sched[sched["is_late"] == 1]
    assert late.groupby("team_id").size().to_dict() == {HOME: LATE_GAMES, AWAY: 2}
    assert sorted(late.loc[late["team_id"] == HOME, "team_game_index"]) == [2, 3]


def test_volume_counts_every_team_but_shares_read_the_last_one(tmp_path):
    """The project convention: a traded player is attributed wholly to his last team."""
    rows = _four_game_preseason()
    rows += [_row(2, AWAY, 20 + g, day, 30.0) for g, day in enumerate([12, 15])]
    _write(tmp_path, rows)

    panel = build_season_panel(SEASON, tmp_path, OPENER).set_index("player_id")
    assert panel.loc[2, "gp_pre"] == 4, "two games for HOME plus two for AWAY"
    assert panel.loc[2, "n_teams_pre"] == 2
    assert panel.loc[2, "team_id"] == AWAY
    assert panel.loc[2, "min_share_pre"] == pytest.approx(1.0), "he is AWAY's only player"


def test_per36_rates_come_off_the_totals(tmp_path):
    _write(tmp_path, _four_game_preseason())
    panel = build_season_panel(SEASON, tmp_path, OPENER).set_index("player_id")

    row = panel.loc[1]
    assert row["min_pre"] == pytest.approx(80.0)
    assert row["reb_pre"] == pytest.approx(20.0)
    assert row["pre_per36_reb"] == pytest.approx(20.0 / 80.0 * 36.0)
    assert row["pre_fg3a_share"] == pytest.approx(16.0 / 40.0)
    assert row["mpg_pre"] == pytest.approx(20.0)
    assert row["gp_share_pre"] == pytest.approx(1.0)


def test_a_player_with_no_appearance_has_no_row(tmp_path):
    """The panel holds appearances, not rosters — the same construction limit
    `src/features/availability.py` documents. `has_preseason` belongs to the attach step;
    what this module owes it is a MEASUREMENT of who is missing."""
    _write(tmp_path, _four_game_preseason())
    panel = build_season_panel(SEASON, tmp_path, OPENER)
    assert set(panel["player_id"]) == {1, 2, 3}


def test_a_season_with_no_file_yields_an_empty_panel(tmp_path):
    assert build_season_panel("1999-00", tmp_path, OPENER).empty


# ── Coverage ──────────────────────────────────────────────────────────────────

def test_coverage_counts_each_drop_reason_separately(tmp_path):
    """A junk row is an API artifact, an exhibition row is a game that is not ours, and a
    post-opener row is a leak that was caught. Collapsing them hides which."""
    rows = _four_game_preseason()
    junk = _row(1, HOME, 0, 4, 480.0)
    junk["PLAYER_ID"] = None
    bubble = _row(1, HOME, 90, 4, 35.0)
    bubble["GAME_DATE"] = "2022-07-28"
    _write(tmp_path, rows + [junk, bubble, _row(77, FOREIGN, 4, 16, 30.0)])

    roster = pd.DataFrame({"player_id": [1, 2, 3, 4], "season": SEASON,
                           "team_abbreviation": "ATL"})
    rec = season_coverage(SEASON, tmp_path, OPENER, roster)

    assert rec["rows_null_player"] == 1
    assert rec["rows_non_nba_team"] == 1
    assert rec["rows_after_opener"] == 1
    assert rec["rows_kept"] == len(rows)
    assert rec["non_nba_team_ids"] == str(FOREIGN)
    assert rec["coverage_class"] == "covered"


def test_coverage_reports_roster_share_and_camp_invitees(tmp_path):
    """Preseason data's first job is the no-prior population, so who is on a season-start
    roster WITHOUT a preseason row is the number the EDA gate needs."""
    _write(tmp_path, _four_game_preseason())
    roster = pd.DataFrame({"player_id": [1, 2, 4, 5], "season": SEASON,
                           "team_abbreviation": "ATL"})
    rec = season_coverage(SEASON, tmp_path, OPENER, roster)

    assert rec["roster_players"] == 4
    assert rec["roster_with_preseason"] == 2
    assert rec["roster_coverage"] == pytest.approx(0.5)
    assert rec["camp_invitees"] == 1, "player 3 played but never reaches a roster"


def test_a_far_opener_gap_is_a_truncated_capture(tmp_path):
    """22 of 23 seasons end 3-5 days before the opener. 2003-04 ends 20 days out, which is
    the signature of a partial capture rather than a short preseason — and the tail is
    exactly what `missed_tail` and the late-weighted shares are read over."""
    _write(tmp_path, _four_game_preseason())
    roster = pd.DataFrame({"player_id": [1], "season": SEASON, "team_abbreviation": "ATL"})

    assert season_coverage(SEASON, tmp_path, OPENER, roster)["coverage_class"] == "covered"
    far = season_coverage(SEASON, tmp_path, pd.Timestamp("2021-11-19"), roster)
    assert far["coverage_class"] == "tail_missing"
    assert far["opener_gap_days"] == 36


def test_coverage_row_exists_for_a_season_with_no_file(tmp_path):
    """A season with no preseason at all is a fact to report, not a row to omit."""
    roster = pd.DataFrame({"player_id": [1, 2], "season": "1999-00",
                           "team_abbreviation": "ATL"})
    rec = season_coverage("1999-00", tmp_path, pd.Timestamp("1999-11-02"), roster)
    assert rec["coverage_class"] == "absent"
    assert rec["rows"] == 0
    assert rec["roster_players"] == 2
    assert np.isnan(rec["roster_coverage"])
