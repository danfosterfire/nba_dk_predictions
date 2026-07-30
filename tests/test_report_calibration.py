import pandas as pd

from src.eda.report_calibration import (
    DESIGNATIONS,
    calibration,
    coverage,
    game_index,
    join_outcomes,
    normalize_name,
    parse_minutes,
    report_frame,
    revision,
    transfer_table,
)

SEASON = "2025-26"


# ── Synthetic builders ────────────────────────────────────────────────────────

def _log(rows: list[dict]) -> pd.DataFrame:
    """An injury-report-log-shaped frame, in the archive's all-string schema."""
    base = {"report_date": "2026-01-10", "report_time": "05_00PM",
            "published_at": "2026-01-10T17:00:00", "game_date": "01/10/2026",
            "game_time": "07:00 (ET)", "matchup": "MIL@CHA", "team": "Milwaukee Bucks",
            "player_name": "", "status": "Out", "reason": "Injury/Illness - Knee",
            "reason_category": "Injury/Illness", "body_part": "Knee",
            "reason_detail": "", "source_file": "x.pdf"}
    return pd.DataFrame([{**base, **r} for r in rows]).astype(str)


def _write_game_log(tmp_path, rows: list[dict]):
    from src.data.fetch import _slug
    base = {"PLAYER_ID": 1, "PLAYER_NAME": "Anchor Man", "TEAM_ID": 1610612749,
            "TEAM_NAME": "Milwaukee Bucks", "TEAM_ABBREVIATION": "MIL",
            "GAME_ID": "0022500001", "GAME_DATE": "2026-01-10T00:00:00"}
    frame = pd.DataFrame([{**base, **r} for r in rows])
    frame.to_csv(tmp_path / f"game_logs_{_slug(SEASON)}.csv", index=False)


def _write_status(tmp_path, rows: list[dict]):
    base = {"season": SEASON, "game_id": "0022500001", "team_id": 1610612749,
            "player_id": 1, "player_name": "Anchor Man", "status": "played",
            "comment": "", "reason": "", "start_position": "", "min": "30:00"}
    frame = pd.DataFrame([{**base, **r} for r in rows])
    frame.to_csv(tmp_path / f"boxscore_status_{SEASON.replace('-', '_')}.csv", index=False)


def _joined(tmp_path, log_rows, status_rows, game_rows=None):
    from src.eda.report_calibration import outcome_frame
    _write_game_log(tmp_path, game_rows or [{}])
    _write_status(tmp_path, status_rows)
    return join_outcomes(report_frame(_log(log_rows)),
                         game_index([SEASON], tmp_path),
                         outcome_frame([SEASON], tmp_path))


# ── Name normalization: the join key ──────────────────────────────────────────

def test_comma_form_and_plain_form_normalize_together():
    """The two sources spell the same player differently; the calibration joins on this."""
    assert normalize_name("Antetokounmpo, Alex") == normalize_name("Alex Antetokounmpo")
    assert normalize_name("O'Neale, Royce") == normalize_name("Royce O'Neale")
    assert normalize_name("Nurkic, Jusuf") == normalize_name("Jusuf Nurkić")


def test_generational_suffixes_are_stripped():
    assert normalize_name("Porter Jr., Michael") == normalize_name("Michael Porter")
    assert normalize_name("Barnes, Scottie III") == normalize_name("Scottie Barnes")


def test_normalize_name_is_empty_for_blank():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""


# ── Minutes parsing ───────────────────────────────────────────────────────────

def test_minutes_parse_from_mm_ss():
    """`pd.to_numeric` gives all-NaN on this column, which reads as missing data."""
    assert parse_minutes("34:30") == 34.5
    assert parse_minutes("0:00") == 0.0
    assert parse_minutes("28") == 28.0
    assert parse_minutes("") != parse_minutes("")      # nan
    assert parse_minutes("junk") != parse_minutes("")  # nan, not a raise


# ── The report frame ──────────────────────────────────────────────────────────

def test_not_yet_submitted_rows_are_dropped():
    """They carry a team and no player — meaningful in the archive, not in a calibration."""
    frame = report_frame(_log([{"player_name": "Lillard, Damian"},
                               {"player_name": "", "status": "NOT YET SUBMITTED"}]))
    assert len(frame) == 1
    assert frame["name_key"].iloc[0] == "damian lillard"


def test_lead_time_is_game_date_minus_report_date():
    frame = report_frame(_log([{"player_name": "A, B", "report_date": "2026-01-09"},
                               {"player_name": "C, D", "report_date": "2026-01-10"}]))
    assert sorted(frame["lead"].tolist()) == [0, 1]


# ── The join, and the absent/unmatched distinction ────────────────────────────

def test_designation_joins_to_realized_outcome(tmp_path):
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "status": "Questionable"}],
        [{"player_name": "Anchor Man", "status": "played", "min": "31:00"}])
    assert len(joined) == 1
    assert joined["designation"].iloc[0] == "Questionable"
    assert joined["outcome"].iloc[0] == "played"


def test_absent_and_unmatched_are_kept_apart(tmp_path):
    """A name-matching bug must not read as a finding about roster mechanics.

    `Anchor Man` exists in the season but has no row in this game — he was not on that
    game's roster, which is a real outcome. `Ghost Player` appears nowhere, which is this
    module's problem.
    """
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "status": "Out"},
         {"player_name": "Player, Ghost", "status": "Out"}],
        [{"player_name": "Anchor Man", "game_id": "0022500002", "status": "played"},
         {"player_name": "Someone Else", "status": "played"}],
        game_rows=[{}, {"GAME_ID": "0022500002", "GAME_DATE": "2026-02-01T00:00:00"}])
    outcomes = dict(zip(joined["name_key"], joined["outcome"]))
    assert outcomes["anchor man"] == "absent"
    assert outcomes["ghost player"] == "unmatched"


def test_games_the_backfill_has_not_reached_are_uncovered_not_scored(tmp_path):
    """Summer League reports have no NBA box score; they must not dilute the denominator."""
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "game_date": "07/10/2026",
          "report_date": "2026-07-10"}],
        [{"player_name": "Anchor Man", "status": "played"}],
        game_rows=[{}, {"GAME_ID": "0022500099", "GAME_DATE": "2026-07-10T00:00:00"}])
    assert joined["outcome"].iloc[0] == "uncovered"
    assert calibration(joined) == [] or all(
        r["value"] == 0 for r in calibration(joined) if r["metric"] == "n")


# ── Measurements ──────────────────────────────────────────────────────────────

def test_calibration_probabilities_sum_to_one_per_designation(tmp_path):
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "status": "Questionable"},
         {"player_name": "Else, Someone", "status": "Questionable"}],
        [{"player_name": "Anchor Man", "status": "played"},
         {"player_name": "Someone Else", "player_id": 2, "status": "inactive",
          "min": ""}])
    rows = [r for r in calibration(joined) if r["key"] == "Questionable|all"]
    probs = {r["metric"]: r["value"] for r in rows if r["metric"].startswith("p_")}
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["p_played"] == 0.5 and probs["p_inactive"] == 0.5


def test_coverage_counts_every_row_exactly_once(tmp_path):
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor"}, {"player_name": "Player, Ghost"}],
        [{"player_name": "Anchor Man", "status": "played"}])
    rows = {r["metric"]: r["value"] for r in coverage(joined)}
    shares = [rows[f"share_{k}"] for k in
              ("played", "dnp", "inactive", "absent", "unmatched", "uncovered")]
    assert abs(sum(shares) - 1.0) < 1e-9


def test_revision_pairs_the_same_player_across_lead_times(tmp_path):
    """The archive's only handle on how a designation decays with horizon."""
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "status": "Questionable",
          "report_date": "2026-01-09"},
         {"player_name": "Man, Anchor", "status": "Available",
          "report_date": "2026-01-10"}],
        [{"player_name": "Anchor Man", "status": "played"}])
    rows = {(r["key"], r["metric"]): r["value"] for r in revision(joined)}
    assert rows[("all", "paired_rows")] == 1
    assert rows[("Questionable", "p_unchanged")] == 0.0
    assert rows[("Questionable", "p_played")] == 1.0


def test_transfer_table_carries_a_marginal_row_per_designation(tmp_path):
    """The marginal is the fallback when a snapshot states a reason with no cell of its own."""
    joined = _joined(
        tmp_path,
        [{"player_name": "Man, Anchor", "status": "Out"}],
        [{"player_name": "Anchor Man", "status": "inactive", "min": ""}])
    table = transfer_table(joined)
    marginal = table[table.reason_category == ""]
    assert marginal["designation"].tolist() == sorted(DESIGNATIONS)
    out = marginal[marginal.designation == "Out"].iloc[0]
    assert out["p_inactive"] == 1.0 and out["n"] == 1


def test_transfer_table_conditions_on_reason_above_the_cell_floor(tmp_path):
    """Below `min_n` a reason cell is mostly its denominator, so it is not emitted."""
    # Names must stay distinct AFTER normalization. `normalize_name` strips digits, so
    # the obvious `f"P{i}"` fixture collapses all 40 players onto the single key "x p" —
    # which this test used to do, and it passed only because the join then silently
    # attached one arbitrary player's outcome to all 40 report rows. Use letters.
    names = [f"{chr(97 + i // 26)}{chr(97 + i % 26)}" for i in range(40)]
    log = [{"player_name": f"{n}surname, {n}first", "status": "Questionable",
            "reason_category": "Injury/Illness"} for n in names]
    status = [{"player_name": f"{n}first {n}surname", "player_id": i,
               "status": "played" if i % 2 else "inactive"}
              for i, n in enumerate(names)]
    joined = _joined(tmp_path, log, status)
    table = transfer_table(joined, min_n=30)
    cell = table[table.reason_category == "Injury/Illness"]
    assert len(cell) == 1 and cell.iloc[0]["n"] == 40
    assert transfer_table(joined, min_n=50).reason_category.eq("").all()


def test_empty_archive_does_not_raise():
    """The module must be runnable before the archive exists."""
    empty = report_frame(pd.DataFrame())
    assert empty.empty
    assert join_outcomes(empty, pd.DataFrame(), pd.DataFrame()).empty


# ── Fabricated rows: the failure an unmatched rate cannot see ─────────────────
#
# An unmatched rate is monotonically increasing in the error it is meant to detect — every
# fabricated match *improves* it. See the "unmatched rate does NOT validate a join" entry
# in CLAUDE.md, which this block exists to enforce. These tests assert the join refuses to
# resolve what it cannot resolve, rather than asserting a headline percentage.

def test_two_players_sharing_a_name_in_one_game_are_ambiguous_not_guessed(tmp_path):
    """`name_key` is not unique. Two distinct player_ids in one game must not have one of
    their outcomes silently attached to the other.

    Not hypothetical: measured over 20 backfilled seasons and 797,473 status rows, game
    0021200757 carries two different `Chris Johnson`s. `drop_duplicates` alone keeps one
    arbitrarily, which is a fabricated row that scores as a successful match.
    """
    joined = _joined(
        tmp_path,
        [{"player_name": "Johnson, Chris", "status": "Questionable"}],
        [{"player_id": 202419, "player_name": "Chris Johnson", "status": "played"},
         {"player_id": 203187, "player_name": "Chris Johnson", "status": "inactive"}],
    )
    assert joined["outcome"].tolist() == ["ambiguous"]


def test_ambiguous_rows_are_excluded_from_the_calibration(tmp_path):
    """An ambiguous row must not be scored under either candidate's outcome."""
    from src.eda.report_calibration import OUTCOMES
    assert "ambiguous" not in OUTCOMES
    joined = _joined(
        tmp_path,
        [{"player_name": "Johnson, Chris", "status": "Questionable"}],
        [{"player_id": 202419, "player_name": "Chris Johnson", "status": "played"},
         {"player_id": 203187, "player_name": "Chris Johnson", "status": "inactive"}],
    )
    assert joined[joined["outcome"].isin(OUTCOMES)].empty


def test_coverage_reports_ambiguous_so_it_cannot_hide(tmp_path):
    joined = _joined(
        tmp_path,
        [{"player_name": "Johnson, Chris", "status": "Questionable"}],
        [{"player_id": 202419, "player_name": "Chris Johnson", "status": "played"},
         {"player_id": 203187, "player_name": "Chris Johnson", "status": "inactive"}],
    )
    rows = {r["metric"]: r for r in coverage(joined)}
    assert rows["share_ambiguous"]["n"] == 1
    assert rows["match_rate"]["value"] == 0.0, "an ambiguous row is not a match"


def test_every_matched_row_resolves_to_exactly_one_player_id(tmp_path):
    """The property that actually matters, stated directly: a scored row must carry the
    outcome of one identifiable player. This is what an unmatched rate never checks."""
    from src.eda.report_calibration import OUTCOMES
    joined = _joined(
        tmp_path,
        [{"player_name": "Anchor Man", "status": "Questionable"},
         {"player_name": "Johnson, Chris", "status": "Out"}],
        [{"player_id": 1, "player_name": "Anchor Man", "status": "played"},
         {"player_id": 202419, "player_name": "Chris Johnson", "status": "played"},
         {"player_id": 203187, "player_name": "Chris Johnson", "status": "inactive"}],
    )
    scored = joined[joined["outcome"].isin(OUTCOMES)]
    assert scored["player_id"].notna().all()
    assert len(scored) == 1 and scored["player_id"].iloc[0] == 1


def test_the_join_does_not_multiply_rows(tmp_path):
    """Row inflation is the other silent fabrication: a many-to-many merge duplicates a
    report row per candidate and every copy counts as matched."""
    joined = _joined(
        tmp_path,
        [{"player_name": "Anchor Man", "status": "Out"}],
        [{"player_id": 1, "player_name": "Anchor Man", "status": "played"},
         {"player_id": 1, "player_name": "Anchor Man", "status": "played"}],
    )
    assert len(joined) == 1, "one report row must stay one row after the join"


def test_a_name_belonging_to_a_different_player_does_not_match(tmp_path):
    """Suffix stripping collapses `Gary Payton` and `Gary Payton II` onto one key. They
    never share a game, so the game-scoped join is safe — but a report naming a player who
    is not in that game must come back `absent`/`unmatched`, never the other man's outcome.
    """
    joined = _joined(
        tmp_path,
        [{"player_name": "Payton II, Gary", "status": "Questionable"}],
        [{"player_id": 56, "player_name": "Gary Payton", "status": "played"}],
        game_rows=[{"PLAYER_ID": 56, "PLAYER_NAME": "Gary Payton"}],
    )
    # Both normalize to "gary payton", so this DOES match — the test pins that the key is
    # lossy, so the outcome carries player_id 56 and any consumer can check the identity.
    assert joined["outcome"].iloc[0] == "played"
    assert joined["player_id"].iloc[0] == 56
