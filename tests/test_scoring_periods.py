import numpy as np
import pandas as pd
import pytest

from src.features.scoring_periods import (
    CUP_FINAL_PREFIX,
    DK_PUBLISHED_ROUND_ENDS,
    OUTSIDE_CONTEST,
    PERIOD_DATE_COLUMN,
    REGULAR_SEASON_PREFIX,
    SCHEDULE_API_FIRST_SEASON,
    assert_contiguous_weeks,
    assert_no_cup_final,
    assert_partition,
    assign_rounds,
    cup_final_mask,
    derive_week_index,
    has_week_numbers,
    round_structure,
    verify_round_structure,
    week_bounds,
)

ROUND_WEEKS = [17, 2, 2, 2]


def _dates(*days: str) -> pd.Series:
    return pd.Series(pd.to_datetime(list(days)))


def _season_dates(n_weeks: int, start: str = "2025-10-21") -> pd.Series:
    """One game on the Tuesday and Friday of each of `n_weeks` consecutive weeks."""
    first = pd.Timestamp(start)
    out = []
    for w in range(n_weeks):
        out += [first + pd.Timedelta(days=7 * w), first + pd.Timedelta(days=7 * w + 3)]
    return pd.Series(out)


def _periods_frame(n_weeks: int, season: str = "2025-26") -> pd.DataFrame:
    dates = _season_dates(n_weeks)
    idx = derive_week_index(dates)
    start, end = week_bounds(dates)
    return pd.DataFrame({
        "season": season,
        "game_id": range(len(dates)),
        "game_id_padded": [str(i).zfill(10) for i in range(len(dates))],
        PERIOD_DATE_COLUMN: dates,
        "period_index": idx,
        "period_name": "Week " + idx.astype(str),
        "period_start": start,
        "period_end": end,
        "tournament_round": assign_rounds(idx.values, ROUND_WEEKS),
        "period_source": "derived",
    })


# ── The week grid ─────────────────────────────────────────────────────────────

def test_week_index_is_monday_anchored():
    # Sun 2025-10-26 closes week 1; Mon 2025-10-27 opens week 2.
    idx = derive_week_index(_dates("2025-10-21", "2025-10-26", "2025-10-27"))
    assert list(idx) == [1, 1, 2]


def test_week_bounds_are_monday_to_sunday():
    start, end = week_bounds(_dates("2025-10-22"))
    assert start.iloc[0] == pd.Timestamp("2025-10-20")
    assert end.iloc[0] == pd.Timestamp("2025-10-26")
    assert start.iloc[0].weekday() == 0 and end.iloc[0].weekday() == 6


def test_all_star_gap_does_not_shift_the_grid():
    """2025-26: week 17's games stop 2/12 and week 18's start 2/19.

    The break spans a whole Monday-Sunday boundary, so a grid keyed on game spacing
    would fuse or shift the weeks. A Monday-anchored one must not notice it.
    """
    idx = derive_week_index(_dates("2026-02-09", "2026-02-12", "2026-02-19",
                                   "2026-02-23"))
    assert list(idx) == [1, 1, 2, 3]


def test_empty_week_is_skipped_not_counted():
    """The COVID-bubble rule: the NBA numbers resumed games consecutively.

    In 2019-20 it labelled the restart weeks 22-24 rather than 41-43, so the index is a
    dense rank over weeks that carry games, not elapsed calendar weeks.
    """
    idx = derive_week_index(_dates("2020-03-09", "2020-03-11", "2020-07-30"))
    assert list(idx) == [1, 1, 2]


def test_week_index_ignores_row_order():
    ordered = derive_week_index(_dates("2025-10-21", "2025-10-27", "2025-11-03"))
    shuffled = derive_week_index(_dates("2025-11-03", "2025-10-21", "2025-10-27"))
    assert list(ordered) == [1, 2, 3]
    assert list(shuffled) == [3, 1, 2]


# ── Rounds ────────────────────────────────────────────────────────────────────

def test_assign_rounds_matches_dk_shape():
    weeks = np.arange(1, 26)
    rounds = assign_rounds(weeks, ROUND_WEEKS)
    assert list(rounds[:17]) == [1] * 17
    assert list(rounds[17:19]) == [2, 2]
    assert list(rounds[19:21]) == [3, 3]
    assert list(rounds[21:23]) == [4, 4]
    # DK's Round 4 ends before the NBA season does; those weeks score for nobody.
    assert list(rounds[23:]) == [OUTSIDE_CONTEST] * 2


def test_assign_rounds_truncates_a_short_season():
    """1998-99 ran 14 weeks, so Round 1 never closes and no later round exists."""
    rounds = assign_rounds(np.arange(1, 15), ROUND_WEEKS)
    assert set(rounds) == {1}


def test_round_structure_flags_incomplete_rounds():
    structure = round_structure(_periods_frame(19), ROUND_WEEKS)
    assert list(structure["tournament_round"]) == [1, 2]
    assert structure["complete"].tolist() == [True, True]
    # 21 weeks reaches round 3 but only half of it.
    structure = round_structure(_periods_frame(20), ROUND_WEEKS)
    short = structure[structure["tournament_round"] == 3].iloc[0]
    assert short["weeks"] == 1 and not short["complete"]


def test_verify_round_structure_accepts_a_full_season():
    periods = _periods_frame(25)
    structure = round_structure(periods, ROUND_WEEKS)
    weeks = pd.Series({"2025-26": 25})
    assert verify_round_structure(structure, ROUND_WEEKS, weeks) is structure
    assert structure["complete"].all()


def test_verify_round_structure_allows_a_short_season_to_truncate():
    """A lockout season loses its tail rounds without failing the build."""
    structure = round_structure(_periods_frame(20), ROUND_WEEKS)
    verify_round_structure(structure, ROUND_WEEKS, pd.Series({"2025-26": 20}))


def test_verify_round_structure_rejects_a_full_season_of_the_wrong_shape():
    """A season with the weeks available must spend all of them on the four rounds."""
    structure = round_structure(_periods_frame(25), ROUND_WEEKS)
    structure.loc[structure["tournament_round"] == 4, "weeks"] = 1
    structure["complete"] = structure["weeks"] == structure["weeks_expected"]
    with pytest.raises(AssertionError, match="does not reproduce DK's shape"):
        verify_round_structure(structure, ROUND_WEEKS, pd.Series({"2025-26": 25}))


def test_verify_round_structure_rejects_an_overlong_round():
    structure = round_structure(_periods_frame(25), ROUND_WEEKS)
    structure.loc[structure["tournament_round"] == 3, "weeks"] = 5
    structure["complete"] = structure["weeks"] == structure["weeks_expected"]
    with pytest.raises(AssertionError, match="longer than DK publishes"):
        verify_round_structure(structure, ROUND_WEEKS, pd.Series({"2025-26": 25}))


def test_verify_round_structure_rejects_a_gap_mid_season():
    """A short round may only ever be the last one — otherwise the map lost its anchor."""
    structure = round_structure(_periods_frame(25), ROUND_WEEKS)
    structure.loc[structure["tournament_round"] == 2, "weeks"] = 1
    structure["complete"] = structure["weeks"] == structure["weeks_expected"]
    with pytest.raises(AssertionError, match="short before the end of the season"):
        verify_round_structure(structure, ROUND_WEEKS, pd.Series({"2025-26": 19}))


# ── The partition, which is the point of the module ───────────────────────────

def test_assert_partition_accepts_a_clean_grid():
    assert_partition(_periods_frame(25))


def test_assert_partition_rejects_a_date_in_two_periods():
    """Two games on one night must never be split across weeks — that double-counts."""
    periods = _periods_frame(25)
    second = periods.iloc[[1]].copy()
    second["game_id"] = 9_999
    second["game_id_padded"] = "0000009999"
    second[PERIOD_DATE_COLUMN] = periods.loc[0, PERIOD_DATE_COLUMN]
    second["period_index"] = 99
    with pytest.raises(AssertionError, match="more than one scoring period"):
        assert_partition(pd.concat([periods, second], ignore_index=True))


def test_assert_partition_rejects_a_duplicated_game():
    periods = _periods_frame(25)
    doubled = pd.concat([periods, periods.iloc[[0]]], ignore_index=True)
    with pytest.raises(AssertionError, match="duplicated"):
        assert_partition(doubled)


def test_assert_partition_rejects_a_missing_period():
    periods = _periods_frame(25)
    periods["period_index"] = periods["period_index"].astype(float)
    periods.loc[3, "period_index"] = np.nan
    with pytest.raises(AssertionError, match="no scoring period"):
        assert_partition(periods)


def test_assert_contiguous_weeks_accepts_a_clean_grid():
    assert_contiguous_weeks(_periods_frame(25))


def test_assert_contiguous_weeks_rejects_a_non_monday_start():
    """Slide the whole week, keeping it 7 days, so only the anchor is wrong."""
    periods = _periods_frame(25)
    periods["period_start"] = periods["period_start"] + pd.Timedelta(days=1)
    periods["period_end"] = periods["period_end"] + pd.Timedelta(days=1)
    with pytest.raises(AssertionError, match="does not start on a Monday"):
        assert_contiguous_weeks(periods)


def test_assert_contiguous_weeks_rejects_a_gap_in_the_index():
    periods = _periods_frame(25)
    periods["period_index"] = periods["period_index"] + 1
    with pytest.raises(AssertionError, match="not contiguous"):
        assert_contiguous_weeks(periods)


# ── The NBA Cup final, which scores nowhere ───────────────────────────────────

def _schedule(**over) -> pd.DataFrame:
    base = pd.DataFrame({
        "game_id_raw": ["0022500001", "0062500001", "0022500002"],
        "week_number": [7, 9, 9],
        "week_name": ["Week 7", "Week 9", "Week 9"],
        "game_label": ["", "Emirates NBA Cup", "Emirates NBA Cup"],
        "game_sub_label": ["", "Championship", "West Group A"],
    })
    return base.assign(**over)


def test_cup_final_identified_by_label():
    mask = cup_final_mask(_schedule())
    assert list(mask) == [False, True, False]
    # It is the *final* that is excluded, not the group games, which score normally.
    assert _schedule().loc[mask, "game_id_raw"].tolist() == ["0062500001"]


def test_cup_final_carries_a_non_regular_season_game_id():
    """The double exclusion: the NBA does not count it in the box score either."""
    sched = _schedule()
    final = sched[cup_final_mask(sched)].iloc[0]
    assert final["game_id_raw"].startswith(CUP_FINAL_PREFIX)
    assert not final["game_id_raw"].startswith(REGULAR_SEASON_PREFIX)


def test_cup_final_label_is_matched_case_insensitively():
    sched = _schedule(game_sub_label=["", "CHAMPIONSHIP", "West Group A"])
    assert list(cup_final_mask(sched)) == [False, True, False]


def test_assert_no_cup_final_rejects_it_reaching_the_output():
    periods = _periods_frame(25)
    periods.loc[0, "game_id_padded"] = "0062500001"
    with pytest.raises(AssertionError, match="Cup championship"):
        assert_no_cup_final(periods, {"0062500001"})


def test_assert_no_cup_final_passes_when_absent():
    assert_no_cup_final(_periods_frame(25), {"0062500001"})


# ── The schedule source ───────────────────────────────────────────────────────

def test_has_week_numbers_detects_the_pre_2017_zeros():
    """ScheduleLeagueV2 returns weekNumber == 0 for every game before 2017-18."""
    assert has_week_numbers(_schedule())
    assert not has_week_numbers(_schedule(week_number=[0, 0, 0]))


def test_has_week_numbers_ignores_non_regular_season_games():
    sched = _schedule(game_id_raw=["0012500001", "0062500001", "0042500002"])
    assert not has_week_numbers(sched)


def test_season_string_ordering_picks_the_api_seasons():
    """The API cutoff is a string comparison, so it has to sort like a season."""
    assert "2025-26" >= SCHEDULE_API_FIRST_SEASON
    assert "2017-18" >= SCHEDULE_API_FIRST_SEASON
    assert not "2016-17" >= SCHEDULE_API_FIRST_SEASON
    assert not "1996-97" >= SCHEDULE_API_FIRST_SEASON


def test_published_round_ends_cover_every_round():
    assert len(DK_PUBLISHED_ROUND_ENDS) == len(ROUND_WEEKS)


from pathlib import Path


# ── The forward season ────────────────────────────────────────────────────────

def _forward_cfg(tmp_path, round_weeks=(17, 2, 2, 2)):
    processed = tmp_path / "processed"
    processed.mkdir(parents=True, exist_ok=True)
    features = tmp_path / "features"
    eda = tmp_path / "eda"
    return {"data": {"processed_dir": str(processed), "features_dir": str(features),
                     "raw_dir": str(tmp_path)},
            "sim": {"scoring_periods": {"round_weeks": list(round_weeks)}},
            "eda": {"output_dir": str(eda)}}


def test_a_played_season_may_not_enter_through_the_forward_path(tmp_path):
    """The knob is for a season nobody has played. A played season's periods come from
    realized dates — 'played, not scheduled' — and letting it through the synthesis
    would silently swap that rule for a plan."""
    from src.features.scoring_periods import run

    cfg = _forward_cfg(tmp_path)
    pd.DataFrame({"season": ["2015-16"], "game_id": ["0021500001"],
                  "game_date": [pd.Timestamp("2015-10-27")]}
                 ).to_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")
    with pytest.raises(ValueError) as excinfo:
        run(cfg, forward_seasons=["2015-16"])
    assert "has been played" in str(excinfo.value)


def test_a_forward_season_enters_the_grid_from_the_synthetic_log(tmp_path):
    """The triples come from `synthetic_game_log`, not the raw schedule, so the period
    grid carries the SAME game ids the roster grid will — filler games included. A
    pre-API season label keeps the test off the network; the week grid is derived either
    way."""
    from src.features.scoring_periods import run

    cfg = _forward_cfg(tmp_path)
    pd.DataFrame({"season": ["2014-15"] * 3,
                  "game_id": [f"002140000{i}" for i in range(3)],
                  "game_date": pd.to_datetime(["2014-10-28", "2014-10-29",
                                               "2014-10-31"])}
                 ).to_parquet(Path(cfg["data"]["processed_dir"]) / "game_logs.parquet")

    # Two franchises meeting daily: 82 games each, nothing for the filler rule to add.
    raw = tmp_path / "nbastats"
    raw.mkdir(parents=True, exist_ok=True)
    teams = (1610612737, 1610612738)
    dates = pd.date_range("2015-10-27", periods=82, freq="D")
    pd.DataFrame({"game_id": [f"00215{i:05d}" for i in range(82)],
                  "date": dates.strftime("%Y-%m-%d"),
                  "home_team_id": [teams[i % 2] for i in range(82)],
                  "away_team_id": [teams[(i + 1) % 2] for i in range(82)],
                  "game_label": ""}).to_csv(raw / "schedule_teams_2015_16.csv",
                                            index=False)
    pd.DataFrame({"PLAYER_ID": [1, 2], "TeamID": list(teams),
                  "PLAYER": ["A", "B"], "HOW_ACQUIRED": ["", ""]}
                 ).to_csv(raw / "team_rosters_2015_16.csv", index=False)

    dest = run(cfg, forward_seasons=["2015-16"])
    periods = pd.read_parquet(dest)
    forward = periods[periods["season"] == "2015-16"]
    assert len(forward) == 82, "every synthetic game carries a period"
    assert forward["period_index"].notna().all()
    assert set(forward["game_id"]) == {f"00215{i:05d}" for i in range(82)}
    assert (periods[periods["season"] == "2014-15"]["period_index"] == 1).all(), (
        "the played seasons build exactly as before")
