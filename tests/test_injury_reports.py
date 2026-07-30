from datetime import date

import pandas as pd

from src.data.injuries import (
    _FIELDNAMES,
    missing_days,
    is_captured,
    read_snapshot,
    snapshot_as_of,
    snapshot_dates,
)
from src.data.injury_reports import (
    COLUMNS,
    DEFAULT_COLUMN_X,
    NOT_SUBMITTED,
    _clean,
    _column_lines,
    _column_of,
    _creation_date,
    _page_rows,
    _split_reason,
    date_range,
    header_columns,
    report_filename,
    capture_status,
    report_as_of,
    report_url,
    should_attempt,
)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _chunk(column: str, y: float, text: str, offset: float = 0.0):
    """A text run in a named column, at the x the real PDF puts it."""
    return (DEFAULT_COLUMN_X[COLUMNS.index(column)] + offset, y, text)


def _player_row(y: float, name: str, status: str, reason: str, team: str = ""):
    chunks = [_chunk("player_name", y, name), _chunk("status", y, status),
              _chunk("reason", y, reason)]
    if team:
        chunks.append(_chunk("team", y, team))
    return chunks


# ── URLs ──────────────────────────────────────────────────────────────────────

def test_report_url_and_filename_match_the_cdn_key():
    day = date(2026, 3, 3)
    assert report_url(day) == ("https://ak-static.cms.nba.com/referee/injury/"
                               "Injury-Report_2026-03-03_05_00PM.pdf")
    assert report_filename(day) == "Injury-Report_2026-03-03_05_00PM.pdf"


def test_date_range_is_inclusive_of_both_ends():
    days = date_range(date(2026, 1, 1), date(2026, 1, 4))
    assert days == [date(2026, 1, d) for d in (1, 2, 3, 4)]


def test_a_recorded_403_is_final_except_for_the_last_few_days():
    """The one failure mode that silently loses history: a capture running before the
    5:00 PM ET publication records a 403 for a report that appears that evening."""
    end = date(2026, 3, 10)
    known = {("2026-03-01", "05_00PM"), ("2026-03-09", "05_00PM")}

    def attempt(day):
        key = (day.isoformat(), "05_00PM")
        return should_attempt(day, key, known, pdf_exists=False, end=end)

    assert not attempt(date(2026, 3, 1))     # settled: no report that day, ever
    assert attempt(date(2026, 3, 9))         # inside the recheck window, try again
    assert attempt(date(2026, 3, 5))         # never attempted at all


def test_an_archived_pdf_is_never_re_downloaded():
    key = ("2026-03-03", "05_00PM")
    assert not should_attempt(date(2026, 3, 3), key, set(), pdf_exists=True,
                              end=date(2026, 3, 3))
    # --reparse works entirely from disk and makes no requests.
    assert not should_attempt(date(2026, 3, 3), key, set(), pdf_exists=False,
                              end=date(2026, 3, 3), reparse=True)


def test_creation_date_parses_the_pdf_stamp():
    class _Reader:
        metadata = {"/CreationDate": "D:20260303170004-05'00'"}

    # The stamp matches the filename key exactly, which is what makes reports
    # individually addressable without scraping an index.
    assert _creation_date(_Reader()) == "2026-03-03T17:00:04"


# ── Text repair ───────────────────────────────────────────────────────────────

def test_clean_repairs_a_hyphenated_wrap_but_not_a_real_dash():
    assert _clean("G  League - Two- Way") == "G League - Two-Way"
    # A hyphen with a space on both sides separates the category from the reason.
    assert _clean("Injury/Illness - Left  Knee; Surgery") == \
        "Injury/Illness - Left Knee; Surgery"


def test_split_reason_only_reads_a_body_part_under_injury_illness():
    assert _split_reason("Injury/Illness - Left Knee; Surgery") == \
        ("Injury/Illness", "Left Knee", "Surgery")
    # "Two-Way" is a roster mechanic, not an anatomy.
    assert _split_reason("G League - Two-Way") == ("G League", "", "Two-Way")


def test_split_reason_treats_a_bare_hyphen_as_no_reason_given():
    """The league prints "-" for an Available player with nothing to report."""
    assert _split_reason("-") == ("", "", "")


# ── Column assignment ─────────────────────────────────────────────────────────

def test_a_long_name_stays_in_its_own_column():
    """Cells are left-aligned and wrap, so a chunk belongs to the last column that
    starts at or before it — midpoint boundaries would misfile a long surname."""
    player_x = DEFAULT_COLUMN_X[COLUMNS.index("player_name")]
    status_x = DEFAULT_COLUMN_X[COLUMNS.index("status")]
    midway = (player_x + status_x) / 2 + 20        # past the midpoint, before the column
    assert _column_of(midway, DEFAULT_COLUMN_X) == "player_name"
    assert _column_of(status_x, DEFAULT_COLUMN_X) == "status"


def test_header_columns_are_read_off_the_header_row():
    chunks = []
    for column, label in zip(COLUMNS, ["Game Date", "Game Time", "Matchup", "Team",
                                       "Player Name", "Current Status", "Reason"]):
        for i, word in enumerate(label.split()):
            chunks.append(_chunk(column, 115.2, word, offset=i * 30))
    assert header_columns(chunks) == DEFAULT_COLUMN_X


# ── Row assembly ──────────────────────────────────────────────────────────────

def test_a_wrapped_reason_rejoins_its_own_player():
    """A wrapped reason is typeset centred on its player row — first line above, the
    continuation below — so equal-y grouping would split it across two players."""
    chunks = [
        *_player_row(136.3, "Bagley III, Marvin", "Out", "Injury/Illness - Neck; Sprain"),
        _chunk("reason", 151.2, "Injury/Illness - Right Finger;"),
        *_player_row(158.3, "Marshall, Naji", "Out", ""),
        _chunk("reason", 165.3, "Contusion"),
        *_player_row(180.3, "Martin, Caleb", "Probable", "Injury/Illness - Low Back"),
    ]
    rows, orphan = _page_rows(_column_lines(chunks, DEFAULT_COLUMN_X))
    by_name = {r["player_name"]: r["reason"] for r in rows}
    assert by_name["Marshall, Naji"] == "Injury/Illness - Right Finger; Contusion"
    assert by_name["Bagley III, Marvin"] == "Injury/Illness - Neck; Sprain"
    assert not any(orphan.values())


def test_a_reason_that_wraps_across_a_page_break_is_returned_as_an_orphan():
    """The continuation lands at the top of the next page and belongs to the previous
    page's last row — not to this page's first player."""
    chunks = [
        _chunk("reason", 87.9, "Vein Thrombosis"),          # tail of the previous page
        *_player_row(117.0, "Young, Trae", "Out", "Injury/Illness - Right Knee"),
        *_player_row(147.0, "Black, Anthony", "Out", "Injury/Illness - Right Quad"),
    ]
    rows, orphan = _page_rows(_column_lines(chunks, DEFAULT_COLUMN_X))
    assert orphan["reason"] == "Vein Thrombosis"
    assert rows[0]["player_name"] == "Young, Trae"
    assert "Thrombosis" not in rows[0]["reason"]


def test_an_unfiled_team_anchors_its_own_row():
    """A team that missed the deadline has no player and no status. Without its own
    anchor it is swallowed by the nearest player row and corrupts it."""
    chunks = [
        *_player_row(469.6, "Tatum, Jayson", "Out",
                     "Injury/Illness - Right Achilles; Repair", team="Boston Celtics"),
        _chunk("team", 492.6, "Utah Jazz"),
        _chunk("reason", 492.6, NOT_SUBMITTED),
    ]
    rows, _ = _page_rows(_column_lines(chunks, DEFAULT_COLUMN_X))
    assert len(rows) == 2
    assert rows[0]["team"] == "Boston Celtics"
    assert "Utah" not in rows[0]["team"]
    assert rows[1]["reason"] == NOT_SUBMITTED


def test_page_furniture_is_dropped():
    chunks = [
        _chunk("team", 62.0, "Injury Report:"),
        _chunk("player_name", 62.0, "03/03/26 05:00"),
        _chunk("status", 62.0, "PM"),
        *_player_row(136.3, "Irving, Kyrie", "Out", "Injury/Illness - Left Knee"),
        _chunk("team", 545.4, "Page 7"),
        _chunk("player_name", 545.4, "of 8"),
    ]
    rows, _ = _page_rows(_column_lines(chunks, DEFAULT_COLUMN_X))
    assert [r["player_name"] for r in rows] == ["Irving, Kyrie"]


# ── The ESPN feed's point-in-time discipline ──────────────────────────────────

def _write_snapshot(tmp_path, stamp: str, players: list[str], legacy: bool = False):
    cols = ["scraped_at", "team", "player", "status", "return_date"] if legacy else \
        ["snapshot_date", "scraped_at", "team", "player", "status",
         "return_date_forecast"]
    rows = []
    for player in players:
        row = {"team": "Atlanta Hawks", "player": player, "status": "Day-To-Day",
               "scraped_at": f"{stamp[:10]}T12:00:00"}
        row["return_date" if legacy else "return_date_forecast"] = "2026-10-01"
        if not legacy:
            row["snapshot_date"] = stamp[:10]
        rows.append(row)
    dest = tmp_path / f"injuries_{stamp}.csv"
    pd.DataFrame(rows)[cols].to_csv(dest, index=False)
    return dest


def test_snapshot_as_of_never_returns_a_future_snapshot():
    """The single rule that keeps this feed from leaking the resolved outcome into a
    historical row."""
    log = pd.DataFrame({
        "snapshot_date": ["2026-06-28", "2026-06-28", "2026-07-27"],
        "player": ["A", "B", "A"],
        "status": ["Out", "Out", "Available"],
    })
    early = snapshot_as_of(log, "2026-07-01")
    assert set(early["snapshot_date"]) == {"2026-06-28"}
    assert list(early["status"]) == ["Out", "Out"]
    # Asked before any snapshot exists, it returns nothing rather than the nearest one.
    assert len(snapshot_as_of(log, "2026-01-01")) == 0


def test_read_snapshot_migrates_the_pre_rename_schema(tmp_path):
    """`return_date` is a forecast made on the snapshot date; the rename says so. Files
    on disk are not rewritten — the archive records what was scraped and when."""
    _write_snapshot(tmp_path, "2026-06-28_16-02-19", ["Jock Landale"], legacy=True)
    df = read_snapshot(tmp_path / "injuries_2026-06-28_16-02-19.csv")
    assert list(df.columns) == _FIELDNAMES
    assert df["return_date_forecast"].iloc[0] == "2026-10-01"
    assert df["snapshot_date"].iloc[0] == "2026-06-28"


def test_snapshot_dates_and_daily_skip(tmp_path):
    _write_snapshot(tmp_path, "2026-06-28_16-02-19", ["A"])
    _write_snapshot(tmp_path, "2026-07-27_14-27-51", ["A"])
    assert snapshot_dates(tmp_path) == ["2026-06-28", "2026-07-27"]
    # Re-running the daily capture inside the window is a no-op, not a duplicate.
    assert is_captured(tmp_path, now=pd.Timestamp("2026-07-27 20:00:00").to_pydatetime())
    assert not is_captured(tmp_path,
                           now=pd.Timestamp("2026-07-29 20:00:00").to_pydatetime())


def test_report_as_of_returns_one_report_and_never_a_later_one():
    """The PDF archive is dated at publication, so it is legitimate for history — but
    only read this way. A plain filter would hand a 2026-01-15 row every later report
    about the same injury, which is the resolved outcome rather than a forecast."""
    log = pd.DataFrame({
        "report_date": ["2026-01-15", "2026-01-15", "2026-03-03"],
        "player_name": ["Irving, Kyrie", "Flagg, Cooper", "Irving, Kyrie"],
        "status": ["Out", "Questionable", "Available"],
    })
    at_january = report_as_of(log, "2026-02-01")
    assert set(at_january["report_date"]) == {"2026-01-15"}
    assert list(at_january["status"]) == ["Out", "Questionable"]
    assert len(report_as_of(log, "2025-12-01")) == 0


# ── Seeing a missed run ───────────────────────────────────────────────────────

def test_capture_status_separates_a_missed_run_from_a_day_with_no_report(tmp_path):
    """The distinction the whole report exists for. A 403 is a day with no report to
    have; a day never attempted is a run that did not happen — and is still recoverable
    until it ages out."""
    (tmp_path / "pdf").mkdir()
    (tmp_path / "pdf" / report_filename(date(2026, 3, 8))).write_bytes(b"%PDF")
    pd.DataFrame([{"report_date": "2026-03-09", "report_time": "05_00PM",
                   "http_status": 403, "bytes": 0, "rows": 0, "published_at": "",
                   "fetched_at": "", "pdf_file": ""}]).to_csv(
        tmp_path / "_injury_report_manifest.csv", index=False)

    status = capture_status(tmp_path, end=date(2026, 3, 10), retention_days=2)
    state = status.set_index("report_date")["state"].to_dict()
    assert state["2026-03-08"] == "archived"
    assert state["2026-03-09"] == "no_report_published"   # 403: nothing to fetch
    assert state["2026-03-10"] == "missed"                # never attempted
    assert status["recoverable"].sum() == 1


def test_espn_missing_days_are_reported_as_permanent(tmp_path):
    """Unlike the PDF archive, nothing here can be refetched — the feed keeps no
    history, so a day the capture did not run is gone."""
    _write_snapshot(tmp_path, "2026-06-28_16-02-19", ["A"])
    _write_snapshot(tmp_path, "2026-07-01_16-02-19", ["A"])
    assert missing_days(tmp_path, end="2026-07-01") == ["2026-06-29", "2026-06-30"]
