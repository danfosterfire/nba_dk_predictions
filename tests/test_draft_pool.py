import numpy as np
import pandas as pd
import pytest

from src.features.draft_pool import (
    DK_CLASSES,
    POSITION_SOURCES,
    _nearest_label,
    _season_label,
    assert_dk_is_single_position,
    assert_drop_rate,
    assert_pool,
    coverage,
    drop_unslottable,
    dual_flags,
    eligibility_flags,
    position_audit,
    position_set,
    primary_letter,
    resolve_positions,
    to_dk_scale,
)


def _rosters(**by_season) -> pd.DataFrame:
    """`{season: {player_id: POSITION}}` → the frame `load_rosters` returns."""
    rows = []
    for season, players in by_season.items():
        for player_id, position in players.items():
            rows.append({"season": season.replace("_", "-"), "player_id": player_id,
                         "roster_name": f"Player {player_id}",
                         "position_nba": position})
    return pd.DataFrame(rows, columns=["season", "player_id", "roster_name",
                                       "position_nba"])


def _boards(**by_season) -> pd.DataFrame:
    """`{season: {player_id: DK position}}` → the frame `load_dk_boards` returns."""
    rows = []
    for season, players in by_season.items():
        for player_id, position in players.items():
            rows.append({"season": season.replace("_", "-"), "player_id": player_id,
                         "dk_player_id": 900000 + abs(player_id),
                         "player_name": f"DK {player_id}", "team_dk": "BOS",
                         "position_dk": position, "adp": np.nan,
                         "capture_date": "2025-10-17", "has_nba_id": player_id > 0})
    return pd.DataFrame(rows, columns=["season", "player_id", "dk_player_id",
                                       "player_name", "team_dk", "position_dk", "adp",
                                       "capture_date", "has_nba_id"])


def _pool(season: str, player_ids: list[int], source: str = "nba_roster"
          ) -> pd.DataFrame:
    return pd.DataFrame({"season": season, "player_id": player_ids, "team": "BOS",
                         "pool_source": source, "has_nba_id": True})


# ── The position map ──────────────────────────────────────────────────────────

def test_primary_letter_takes_the_first_token_of_a_dual():
    got = primary_letter(pd.Series(["G", "G-F", "F-C", "C-F", "F-G", "C"]))
    assert list(got) == ["G", "G", "F", "C", "F", "C"]


def test_primary_letter_rejects_a_class_outside_dk_s_three():
    # A `PG` or `SG` would silently become an eligibility nobody can fill, so it must
    # come back missing and fall through the cascade instead.
    got = primary_letter(pd.Series(["PG", "SG-SF", None, ""]))
    assert got.isna().all()


def test_position_set_expands_duals_and_singles():
    got = position_set(pd.Series(["G-F", "C", None]))
    assert list(got) == [frozenset({"G", "F"}), frozenset({"C"}), frozenset()]


def test_eligibility_flags_are_one_hot_and_dual_flags_are_not():
    letters = pd.Series(["G", "F", "C"])
    flags = eligibility_flags(letters)
    assert flags.sum(axis=1).eq(1).all()

    duals = dual_flags(pd.Series(["G-F", "C", "F-C"]), pd.Series(["G", "C", "F"]))
    assert list(duals.sum(axis=1)) == [2, 1, 2]
    assert duals.loc[0, "dual_g"] and duals.loc[0, "dual_f"] and not duals.loc[0, "dual_c"]


def test_dual_flags_union_the_shipped_letter_so_the_swap_only_widens():
    # Two cases where NBA.com's set alone is NOT a superset of what ships: it names
    # nothing (a 2026 rookie), and it names the other class outright (DK contradicts a
    # single NBA.com letter). Both must keep the shipped slot, or a sensitivity run
    # would be measuring two changes at once.
    duals = dual_flags(pd.Series([None, "G"], dtype="object"),
                       pd.Series(["F", "F"], dtype="string"))
    assert list(duals["dual_f"]) == [True, True]
    assert list(duals["dual_g"]) == [False, True]


def test_eligibility_flag_is_false_rather_than_missing_for_an_unresolved_letter():
    # `assert_pool` counts flags, so a missing letter has to read as "eligible nowhere"
    # rather than as NA, which would sum to something truthy.
    flags = eligibility_flags(pd.Series([None, "G"], dtype="string"))
    assert list(flags.sum(axis=1)) == [0, 1]
    assert flags.dtypes.eq(bool).all()


# ── The claim the whole design rests on ───────────────────────────────────────

def test_a_dual_on_a_dk_board_fails_loudly():
    # The plan assumed DK carried `G-F`. It does not, and the single-letter convention
    # below is only sound while that holds — so a future board that changes must break
    # the build rather than quietly lose the second class.
    assert_dk_is_single_position(_boards(**{"2025_26": {1: "G", 2: "C"}}))
    with pytest.raises(AssertionError, match="single-position"):
        assert_dk_is_single_position(_boards(**{"2025_26": {1: "G-F"}}))


def test_position_audit_separates_containment_from_primary_agreement():
    # Two players NBA.com calls tweeners, and DK picks the *second* letter for both.
    # Containment is perfect and primary agreement is zero — the distinction the
    # headline turns on, and one number cannot express it.
    rosters = _rosters(**{"2025_26": {1: "G-F", 2: "F-C"}})
    boards = _boards(**{"2025_26": {1: "F", 2: "C"}})
    audit = position_audit(rosters, boards, {"rosters_2025-26": ("2025-26", "2025-26")})

    roll = audit[audit["position_nba"] == "ALL"].iloc[0]
    assert roll["n"] == 2
    assert roll["dk_inside_nba_set"] == 1.0
    assert roll["dk_equals_nba_primary"] == 0.0
    assert bool(roll["contemporaneous"])


def test_position_audit_counts_an_outright_contradiction_as_outside_the_set():
    rosters = _rosters(**{"2025_26": {1: "G", 2: "C"}})
    boards = _boards(**{"2025_26": {1: "F", 2: "C"}})
    audit = position_audit(rosters, boards, {"w": ("2024-25", "2024-25")})
    assert audit.empty  # no roster rows in that window at all

    audit = position_audit(rosters, boards, {"w": ("2025-26", "2025-26")})
    roll = audit[audit["position_nba"] == "ALL"].iloc[0]
    assert roll["dk_inside_nba_set"] == 0.5
    # Contemporaneous is read off the season range, not off the window's name.
    assert bool(roll["contemporaneous"])


def test_position_audit_windows_restrict_which_roster_seasons_are_read():
    # The split-clean corroboration: the same board measured against train-only rosters
    # must use only those rows, which is what makes "the finding is not from held-out
    # seasons" checkable rather than asserted.
    rosters = _rosters(**{"2021_22": {1: "G"}, "2025_26": {1: "F", 2: "C"}})
    boards = _boards(**{"2025_26": {1: "F", 2: "C"}})
    audit = position_audit(rosters, boards,
                           {"train": ("1996-97", "2021-22"),
                            "all": ("1996-97", "9999")})
    train = audit[(audit["roster_window"] == "train")
                  & (audit["position_nba"] == "ALL")].iloc[0]
    every = audit[(audit["roster_window"] == "all")
                  & (audit["position_nba"] == "ALL")].iloc[0]
    assert train["n"] == 1 and train["dk_inside_nba_set"] == 0.0
    assert every["n"] == 2 and every["dk_inside_nba_set"] == 1.0


# ── The resolution cascade ────────────────────────────────────────────────────

def test_dk_board_wins_over_the_nba_roster_for_the_same_season():
    pool = _pool("2025-26", [1])
    out = resolve_positions(pool, _rosters(**{"2025_26": {1: "G"}}),
                            _boards(**{"2025_26": {1: "F"}}))
    assert out.loc[0, "position"] == "F"
    assert out.loc[0, "position_source"] == "dk_board"
    assert bool(out.loc[0, "pos_f"]) and not bool(out.loc[0, "pos_g"])
    # The rejected convention reads NBA.com on top of the shipped letter, so where the
    # two contradict it widens to both rather than replacing one with the other.
    assert bool(out.loc[0, "dual_g"]) and bool(out.loc[0, "dual_f"])


def test_the_nba_map_is_used_where_no_board_exists():
    pool = _pool("2022-23", [1])
    out = resolve_positions(pool, _rosters(**{"2022_23": {1: "G-F"}}),
                            _boards(**{"2025_26": {2: "C"}}))
    assert out.loc[0, "position"] == "G"
    assert out.loc[0, "position_source"] == "nba_roster"
    assert bool(out.loc[0, "dual_eligible"])
    assert bool(out.loc[0, "dual_g"]) and bool(out.loc[0, "dual_f"])


def test_a_player_missing_from_his_own_roster_file_carries_a_label_from_another_season():
    # The current-status roster CSV has already dropped ~5% of season-start players by
    # the time it is fetched. Position is static, so the label carries.
    pool = _pool("2022-23", [1])
    out = resolve_positions(pool, _rosters(**{"2023_24": {1: "C-F"}}), _boards())
    assert out.loc[0, "position"] == "C"
    assert out.loc[0, "position_source"] == "nba_roster_carry"


def test_the_carry_prefers_the_nearest_season_and_breaks_ties_toward_the_earlier_one():
    pool = _pool("2010-11", [1])
    out = resolve_positions(pool, _rosters(**{"2009_10": {1: "G"}, "2011_12": {1: "C"}}),
                            _boards())
    assert out.loc[0, "position"] == "G"


def test_a_dk_board_carry_is_the_last_resort_before_unslottable():
    pool = _pool("2018-19", [1])
    out = resolve_positions(pool, _rosters(), _boards(**{"2025_26": {1: "C"}}))
    assert out.loc[0, "position_source"] == "dk_board_carry"
    assert out.loc[0, "position"] == "C"


def test_a_player_no_source_names_is_unslottable_and_dropped():
    pool = _pool("2000-01", [1, 2])
    out = resolve_positions(pool, _rosters(**{"2000_01": {1: "G"}}), _boards())
    assert out.loc[out["player_id"] == 2, "position_source"].iloc[0] == "none"

    kept, dropped = drop_unslottable(out)
    assert list(kept["player_id"]) == [1] and list(dropped["player_id"]) == [2]
    # Keeping him would be worse than dropping him: he would consume a draft pick and
    # never be startable, silently shrinking a 16-man roster.
    assert_pool(kept.assign(team="BOS"))


def test_every_resolved_source_is_in_the_declared_vocabulary():
    pool = _pool("2025-26", [1, 2, 3])
    out = resolve_positions(
        pool,
        _rosters(**{"2025_26": {2: "F"}, "2019_20": {3: "C"}}),
        _boards(**{"2025_26": {1: "G"}}))
    assert set(out["position_source"]) <= set(POSITION_SOURCES)


def test_nearest_label_returns_missing_when_nothing_is_labelled():
    pool = _pool("2000-01", [1]).assign(_start_year=2000)
    got = _nearest_label(pool, pd.DataFrame(columns=["player_id", "_start_year", "x"]),
                         "x")
    assert got.isna().all()


# ── Assertions ────────────────────────────────────────────────────────────────

def _valid_pool() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2022-23", "2022-23"], "player_id": [1, 2],
        "player_name": ["A", "B"], "team": ["BOS", "LAL"],
        "position_source": ["nba_roster", "nba_roster"],
        "pos_g": [True, False], "pos_f": [False, True], "pos_c": [False, False],
        "dual_g": [True, False], "dual_f": [True, True], "dual_c": [False, False],
    })


def test_assert_pool_rejects_a_dual_convention_that_moves_eligibility():
    bad = _valid_pool()
    bad.loc[0, "dual_g"] = False   # shipped at G, dual convention drops him
    with pytest.raises(AssertionError, match="must widen eligibility"):
        assert_pool(bad)


def test_assert_pool_accepts_a_well_formed_board():
    assert_pool(_valid_pool())


def test_assert_pool_rejects_a_duplicated_player():
    bad = _valid_pool()
    bad.loc[1, "player_id"] = 1
    with pytest.raises(AssertionError, match="duplicated"):
        assert_pool(bad)


def test_assert_pool_rejects_a_player_eligible_at_no_class_or_at_two():
    nowhere = _valid_pool()
    nowhere.loc[0, "pos_g"] = False
    with pytest.raises(AssertionError, match="exactly one"):
        assert_pool(nowhere)

    both = _valid_pool()
    both.loc[0, "pos_f"] = True
    with pytest.raises(AssertionError, match="exactly one"):
        assert_pool(both)


def test_assert_pool_rejects_a_missing_team_because_the_roster_rule_needs_two():
    bad = _valid_pool()
    bad.loc[0, "team"] = None
    with pytest.raises(AssertionError, match="no team"):
        assert_pool(bad)


def test_assert_drop_rate_fires_on_a_hollowed_out_pool_not_on_the_known_hole():
    pool = pd.DataFrame({"season": ["2022-23"] * 100})
    assert_drop_rate(pd.DataFrame({"season": ["1996-97"]}), pool)
    with pytest.raises(AssertionError, match="over the 2% bar"):
        assert_drop_rate(pd.DataFrame({"season": ["1996-97"] * 20}), pool)
    with pytest.raises(AssertionError, match="per-season bar"):
        assert_drop_rate(pd.DataFrame({"season": ["1996-97"] * 40}), pool,
                         max_share=1.0, max_per_season=30)


# ── ADP on DK's scale ─────────────────────────────────────────────────────────

def test_dk_sourced_adp_passes_through_the_transfer_unchanged(tmp_path):
    _write_transfer(tmp_path)
    adp = pd.Series([1.0, 1.0])
    source = pd.Series(["draftkings", "fantasypros"])
    got = to_dk_scale(adp, source, tmp_path)
    assert got.iloc[0] == 1.0        # already on DK's scale
    assert got.iloc[1] == 2.0        # recalibrated


def test_the_transfer_is_monotone_and_clips_outside_its_fitted_grid(tmp_path):
    _write_transfer(tmp_path)
    source = pd.Series(["fantasypros"] * 4)
    got = to_dk_scale(pd.Series([0.5, 1.0, 3.0, 99.0]), source, tmp_path)
    assert list(got) == [2.0, 2.0, 6.0, 6.0]
    assert (np.diff(got.to_numpy()) >= 0).all()


def test_a_player_with_no_adp_stays_missing_rather_than_becoming_pick_one(tmp_path):
    _write_transfer(tmp_path)
    got = to_dk_scale(pd.Series([np.nan]), pd.Series(["fantasypros"]), tmp_path)
    assert got.isna().all()


def _write_transfer(tmp_path) -> None:
    pd.DataFrame({"adp_consensus": [1.0, 2.0, 3.0],
                  "adp_dk_fitted": [2.0, 4.0, 6.0]}).to_parquet(
        tmp_path / "adp_transfer.parquet", index=False)


# ── Odds and ends ─────────────────────────────────────────────────────────────

def test_season_label_inverts_the_fetch_helper():
    from src.data.fetch import _season_start_year

    for season in ("1996-97", "1999-00", "2009-10", "2025-26", "2026-27"):
        assert _season_label(_season_start_year(season)) == season


def test_coverage_reports_the_dropped_rows_against_the_season_they_left():
    pool = pd.DataFrame({
        "season": ["2022-23"], "team": ["BOS"], "pool_source": ["nba_roster"],
        "adp": [1.0], "adp_source": ["fantasypros"], "prior_in_matrix": [True],
        "prior_in_roster_matrix": [True], "dual_eligible": [False],
        "position_source": ["nba_roster"],
        "pos_g": [True], "pos_f": [False], "pos_c": [False],
    })
    got = coverage(pool, pd.DataFrame({"season": ["2022-23", "2022-23"]})).iloc[0]
    assert got["players"] == 1 and got["dropped_no_position"] == 2
    assert got["n_g"] == 1 and got["n_f"] == 0
    assert set(DK_CLASSES) == {"G", "F", "C"}
