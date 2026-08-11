"""`make capture-calendar` — the coverage emitter behind the dashboard's page 7.

The emitter re-reads what `make capture-status` and `make adp-status` print, so most of
what could go wrong here is *disagreeing with them*. These tests build the four archives on
disk as synthetic directories and check the states that come back, plus the two rules that
are the artifact's own: a `nothing_to_capture` day is not a failure, and recoverability is a
property of the program rather than of the day.
"""

import gzip
from datetime import date
from pathlib import Path

import pandas as pd

from src.data import capture_calendar as cc
from src.data import injury_reports


# ── Synthetic archives ────────────────────────────────────────────────────────

def _injury_reports(tmp: Path, archived: list[str], attempted_empty: list[str],
                    time_key: str = injury_reports.DEADLINE_TIME_KEY) -> Path:
    """A PDF archive: some days with a file, some attempted and 403'd, the rest untouched."""
    out = tmp / "injury_reports"
    (out / "pdf").mkdir(parents=True, exist_ok=True)
    rows = []
    for day in archived:
        (out / "pdf" / injury_reports.report_filename(date.fromisoformat(day),
                                                      time_key)).write_bytes(b"%PDF-1.4")
        rows.append({"report_date": day, "report_time": time_key, "http_status": 200,
                     "bytes": 8, "rows": 30, "published_at": "", "fetched_at": "",
                     "pdf_file": ""})
    for day in attempted_empty:
        rows.append({"report_date": day, "report_time": time_key, "http_status": 403,
                     "bytes": 0, "rows": 0, "published_at": "", "fetched_at": "",
                     "pdf_file": ""})
    pd.DataFrame(rows).to_csv(out / injury_reports.MANIFEST_NAME, index=False)
    return out


def _espn(tmp: Path, days: list[str]) -> Path:
    out = tmp / "injuries"
    out.mkdir(parents=True, exist_ok=True)
    log = []
    for day in days:
        frame = pd.DataFrame([{"snapshot_date": day, "player_name": "A", "team": "XYZ",
                               "status": "Out"}])
        frame.to_csv(out / f"injuries_{day}_18-30-00.csv", index=False)
        log.append(frame)
    pd.concat(log, ignore_index=True).to_csv(out / "injuries_log.csv", index=False)
    return out


_FP_PAGE = ("<table><tr><th>Rank</th><th>Player</th><th>Yahoo</th><th>AVG</th></tr>"
            "<tr><td>1</td><td>A Player LAL PG</td><td>1.0</td><td>1.0</td></tr></table>")


def _fantasypros(tmp: Path, parsed: list[str], placeholder: list[str]) -> Path:
    """The manifest is what the emitter reads; the pages exist so nothing else trips."""
    out = tmp / "fantasypros"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for day in parsed + placeholder:
        name = f"fp_adp_{day}_040000.html.gz"
        (out / name).write_bytes(gzip.compress(
            _FP_PAGE.encode() if day in parsed else b"<html></html>"))
        rows.append({"captured_at": f"{day}T04:00:00", "capture_date": day, "path": name,
                     "n_rows": 212 if day in parsed else 0, "sources": "Yahoo",
                     "season": "2022-23", "identical_share": "", "overlap": "",
                     "frozen": "", "inherited": "", "parsed": day in parsed})
    pd.DataFrame(rows).to_csv(out / "_fantasypros_manifest.csv", index=False)
    return out


def _draftkings(tmp: Path, boards: dict[str, str]) -> Path:
    """`{filename stem date: 'Mon<D>_<YYYY>'}` — the manual pre-draft CSVs."""
    out = tmp / "dk_draft_rankings"
    out.mkdir(parents=True, exist_ok=True)
    for stamp in boards.values():
        pd.DataFrame([{"ID": 1, "Name": "A Player", "ADP": 1.0, "Team": "LAL",
                       "Position": "G"}]).to_csv(
            out / f"DkPreDraftRankings_{stamp}.csv", index=False)
    return out


def _cfg(tmp: Path) -> dict:
    return {"eda": {"output_dir": str(tmp / "out")},
            "data": {"injury_reports": {"dir": str(tmp / "injury_reports"),
                                        "retention_days": 6},
                     "injuries": {"dir": str(tmp / "injuries")},
                     "adp": {"fantasypros": {"dir": str(tmp / "fantasypros")},
                             "draftkings": {"dir": str(tmp / "dk_draft_rankings")}}}}


# ── The three states, and the one that is not a failure ───────────────────────

def test_an_attempted_day_with_no_report_is_not_counted_as_a_gap(tmp_path):
    """The distinction the whole artifact exists for. A day the CDN 403s is the offseason;
    a day nobody asked about is a run that did not happen, and only the second is an alarm."""
    _injury_reports(tmp_path, archived=["2026-08-01", "2026-08-02"],
                    attempted_empty=["2026-08-03"])
    days = cc.injury_report_days(tmp_path / "injury_reports", end=date(2026, 8, 4),
                                 retention_days=3)
    states = dict(zip(days["capture_date"], days["state"]))
    assert states["2026-08-01"] == cc.CAPTURED
    assert states["2026-08-03"] == cc.NOTHING_TO_CAPTURE
    assert states["2026-08-04"] == cc.MISSED
    # And only the genuine gap is flagged fetchable.
    fetchable = dict(zip(days["capture_date"], days["recoverable"]))
    assert fetchable["2026-08-04"] and not fetchable["2026-08-03"]


def test_the_emitter_reproduces_what_capture_status_prints(tmp_path):
    """It re-reads `injury_reports.capture_status` rather than re-deriving the states, so
    the printout and the artifact cannot drift apart. This pins that they agree."""
    out = _injury_reports(tmp_path, archived=["2026-08-01"],
                          attempted_empty=["2026-08-02"])
    printed = injury_reports.capture_status(out, date(2026, 8, 3), retention_days=2)
    emitted = cc.injury_report_days(out, end=date(2026, 8, 3), retention_days=2)
    assert len(printed) == len(emitted)
    assert dict(zip(emitted["capture_date"], emitted["state"])) == {
        row.report_date: cc._IR_STATES[row.state]
        for row in printed.itertuples(index=False)}


def test_an_espn_gap_is_never_recoverable_and_an_untouched_prefix_is_not_a_gap(tmp_path):
    """The feed keeps no history, so a missed day is permanent — but the archive only
    starts when the first snapshot was taken, and days before that were never owed."""
    out = _espn(tmp_path, ["2026-08-01", "2026-08-04"])
    days = cc.espn_days(out, end=date(2026, 8, 5))
    states = dict(zip(days["capture_date"], days["state"]))
    assert states == {"2026-08-01": cc.CAPTURED, "2026-08-02": cc.MISSED,
                      "2026-08-03": cc.MISSED, "2026-08-04": cc.CAPTURED,
                      "2026-08-05": cc.MISSED}
    assert not days["recoverable"].any()
    # Nothing before the first snapshot appears at all.
    assert min(days["capture_date"]) == "2026-08-01"


def test_an_event_program_emits_captures_and_never_a_missed_day(tmp_path):
    """A board that opens in October has no schedule to have missed the rest of the year,
    so filling its empty days would invent a year of failures a season."""
    fp = _fantasypros(tmp_path, parsed=["2022-10-02"], placeholder=["2022-10-03"])
    dk = _draftkings(tmp_path, {"2025-10-17": "Oct17_2025"})
    for days in (cc.fantasypros_days(fp), cc.draftkings_days(dk)):
        assert cc.MISSED not in set(days["state"])
    # A page that archived but held no board is "nothing to capture", not a capture.
    fp_days = dict(zip(cc.fantasypros_days(fp)["capture_date"],
                       cc.fantasypros_days(fp)["state"]))
    assert fp_days == {"2022-10-02": cc.CAPTURED, "2022-10-03": cc.NOTHING_TO_CAPTURE}


# ── The program table ─────────────────────────────────────────────────────────

def test_a_daily_programs_window_is_its_obligation_not_its_coverage(tmp_path):
    """Inferring the window from the rows would shrink it to whatever was captured, which
    is exactly backwards: the retention window is what the program *owes*."""
    cfg = _cfg(tmp_path)
    _injury_reports(tmp_path, archived=["2026-08-09"], attempted_empty=[])
    _espn(tmp_path, ["2026-08-09"])
    _fantasypros(tmp_path, parsed=["2022-10-02"], placeholder=[])
    _draftkings(tmp_path, {"2025-10-17": "Oct17_2025"})
    _, programs = cc.build(cfg, end=date(2026, 8, 10))
    row = programs.set_index("program").loc[cc.INJURY_REPORTS]
    assert row["window_start"] == "2026-08-04" and row["window_end"] == "2026-08-10"
    assert row["recovery"] == cc.WINDOW and row["recovery_window_days"] == 6
    # An event program takes its own first and last capture instead.
    event = programs.set_index("program").loc[cc.ADP_FANTASYPROS]
    assert event["window_start"] == event["window_end"] == "2022-10-02"
    assert event["cadence"] == cc.EVENT and event["recovery"] == cc.ARCHIVE


def test_the_counts_split_a_gap_into_the_two_that_need_different_actions(tmp_path):
    cfg = _cfg(tmp_path)
    _injury_reports(tmp_path, archived=["2026-08-09"], attempted_empty=["2026-08-08"])
    _espn(tmp_path, ["2026-08-08"])
    _fantasypros(tmp_path, parsed=[], placeholder=[])
    _draftkings(tmp_path, {})
    calendar, programs = cc.build(cfg, end=date(2026, 8, 10))
    reports = programs.set_index("program").loc[cc.INJURY_REPORTS]
    assert reports["n_recoverable"] == reports["n_missed"] and reports["n_lost"] == 0
    espn = programs.set_index("program").loc[cc.ESPN_INJURIES]
    assert espn["n_lost"] == espn["n_missed"] == 2 and espn["n_recoverable"] == 0
    # Every row of the calendar carries one of the three states and nothing else.
    assert set(calendar["state"]) <= set(cc.STATES)


def test_an_absent_archive_is_an_empty_row_rather_than_a_crash(tmp_path):
    """A fresh checkout has none of these directories, and `make capture-calendar` still
    has to write an artifact — a page that cannot open the file cannot say what is missing."""
    cfg = _cfg(tmp_path)
    calendar, programs = cc.build(cfg, end=date(2026, 8, 10))
    assert list(programs["program"]) == [cc.INJURY_REPORTS, cc.ESPN_INJURIES,
                                         cc.ADP_FANTASYPROS, cc.ADP_DRAFTKINGS]
    assert (programs["n_captured"] == 0).all()
    # The injury-report row is still a full retention window of missed days: nothing on
    # disk is not the same as nothing owed.
    assert programs.set_index("program").loc[cc.INJURY_REPORTS, "n_missed"] == 7
    assert len(calendar) == 7


def test_run_writes_both_artifacts_where_the_dashboard_looks_for_them(tmp_path):
    cfg = _cfg(tmp_path)
    _injury_reports(tmp_path, archived=["2026-08-09"], attempted_empty=[])
    _espn(tmp_path, ["2026-08-09"])
    _fantasypros(tmp_path, parsed=["2022-10-02"], placeholder=[])
    _draftkings(tmp_path, {"2025-10-17": "Oct17_2025"})
    dest = cc.run(cfg, end=date(2026, 8, 10))
    assert dest.name == cc.CALENDAR_NAME and dest.exists()
    programs = pd.read_csv(dest.parent / cc.PROGRAMS_NAME)
    assert list(programs.columns) == cc.PROGRAM_COLUMNS
    assert list(pd.read_csv(dest).columns) == cc.CALENDAR_COLUMNS
    assert set(programs["cadence"]) <= set(cc.CADENCES)
    assert set(programs["recovery"]) <= set(cc.RECOVERIES)
