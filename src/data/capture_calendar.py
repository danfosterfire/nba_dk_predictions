"""`make capture-calendar` — what the four capture programs actually have on disk.

`make capture-status` and `make adp-status` **print**. That is the right shape for a person
at a terminal and the wrong one for everything else: the coverage of a perishable feed is an
operational fact, and a printout cannot be drawn, diffed, or checked by anything. This module
re-reads exactly what those two commands read — the archives on disk, **never the network** —
and writes the same picture as two small CSVs.

It is an emitter, not a pipeline stage. It fits nothing, fetches nothing, and re-derives
nothing: every state below comes from `injury_reports.capture_status`, `injuries`'
snapshot/missing-day pair, and the two ADP manifests, which stay the single implementation
of "what does this archive hold".

## Two files, because they answer two different questions

    outputs/eda/capture_calendar.csv   one row per (program, day) that has a state
    outputs/eda/capture_programs.csv   one row per program: cadence, recovery policy, window

The calendar answers *which days*. The program table answers *what happens to a day that is
missing*, which is a fact about the **source** rather than about the archive, and it is the
one thing a reader cannot infer from a grid of cells.

## The state vocabulary is closed, and the third state is the whole point

- `captured` — a capture for that day is on disk.
- `nothing_to_capture` — the day was attempted and the source published nothing. The
  offseason, the All-Star break, a placeholder page. **Not a failure.**
- `missed` — the run did not happen. This is the alarm.

Without the middle state the first and third are indistinguishable from the outside, and a
scheduler that silently stopped firing would only become visible once the days were gone.

## Recoverability is a property of the program, not of the day

Which is why it is a column on the program table and only a derived flag on the calendar:

- `window` — a missed day is recoverable until it ages out of a rolling retention window.
  The NBA's injury-report CDN, at `data.injury_reports.retention_days`.
- `never` — a missed day is gone. The ESPN feed describes *today* and keeps no history; the
  DraftKings board is login-gated, has zero Wayback presence, and is open only while
  contests are (~October).
- `archive` — a third party holds the history and it can be fetched later. FantasyPros, via
  Wayback, which is why it is the one source of the four that is not on a deadline.

Usage:
    python -m src.data.capture_calendar            # write both artifacts
    python -m src.data.capture_calendar --end DATE # as of a fixed day, for tests
"""

import argparse
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yaml

from src.data import adp_draftkings, adp_fantasypros, injuries, injury_reports

CALENDAR_NAME = "capture_calendar.csv"
PROGRAMS_NAME = "capture_programs.csv"

# ── The closed vocabularies ───────────────────────────────────────────────────

CAPTURED = "captured"
NOTHING_TO_CAPTURE = "nothing_to_capture"
MISSED = "missed"
STATES = (CAPTURED, NOTHING_TO_CAPTURE, MISSED)

#: A program that owes a capture every day, against one that captures when a board opens.
#: Only a `daily` program can have a `missed` day — an `event` program has no schedule to
#: have missed, so an empty stretch of its row is not a gap and must not be drawn as one.
DAILY = "daily"
EVENT = "event"
CADENCES = (DAILY, EVENT)

WINDOW = "window"
NEVER = "never"
ARCHIVE = "archive"
RECOVERIES = (WINDOW, NEVER, ARCHIVE)

CALENDAR_COLUMNS = ["program", "capture_date", "state", "recoverable", "records"]
PROGRAM_COLUMNS = ["program", "cadence", "recovery", "recovery_window_days",
                   "window_start", "window_end", "n_days", "n_captured",
                   "n_nothing_to_capture", "n_missed", "n_recoverable", "n_lost",
                   "records"]

INJURY_REPORTS = "injury_reports"
ESPN_INJURIES = "espn_injuries"
ADP_FANTASYPROS = "adp_fantasypros"
ADP_DRAFTKINGS = "adp_draftkings"

# `injury_reports.capture_status` speaks its own three words; they mean these three.
_IR_STATES = {"archived": CAPTURED, "no_report_published": NOTHING_TO_CAPTURE,
              "missed": MISSED}


def _frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=CALENDAR_COLUMNS)
    if out.empty:
        return out
    return out.sort_values(["program", "capture_date"]).reset_index(drop=True)


# ── One reader per program, each over the archive its own module owns ─────────

def injury_report_days(out_dir: str | Path, end: date | None = None,
                       retention_days: int = injury_reports.RETENTION_DAYS,
                       time_key: str = injury_reports.DEADLINE_TIME_KEY) -> pd.DataFrame:
    """The retention window, day by day. `capture_status` decides the states."""
    status = injury_reports.capture_status(out_dir, end, retention_days, time_key)
    if status.empty:
        return _frame([])
    manifest = _read_csv(Path(out_dir) / injury_reports.MANIFEST_NAME)
    rows = (pd.to_numeric(manifest["rows"], errors="coerce")
            .groupby(manifest["report_date"].astype(str)).max().to_dict()
            if not manifest.empty else {})
    return _frame([{"program": INJURY_REPORTS, "capture_date": r.report_date,
                    "state": _IR_STATES[r.state], "recoverable": bool(r.recoverable),
                    "records": float(rows.get(r.report_date, 0) or 0)}
                   for r in status.itertuples(index=False)])


def espn_days(out_dir: str | Path, end: date | None = None) -> pd.DataFrame:
    """Snapshot days, plus every calendar day since the first that has none.

    Those gaps are `recoverable = False` without qualification, and that is the single most
    important cell in this artifact: the feed reports current status and keeps no history,
    so a day the capture did not run is not a backlog item.
    """
    days = injuries.snapshot_dates(out_dir)
    if not days:
        return _frame([])
    end_str = (end or date.today()).isoformat()
    log = injuries.load_log(out_dir)
    records = (log.groupby("snapshot_date").size().to_dict()
               if not log.empty and "snapshot_date" in log.columns else {})
    rows = [{"program": ESPN_INJURIES, "capture_date": day, "state": CAPTURED,
             "recoverable": False, "records": float(records.get(day, 0))}
            for day in days]
    rows += [{"program": ESPN_INJURIES, "capture_date": day, "state": MISSED,
              "recoverable": False, "records": 0.0}
             for day in injuries.missing_days(out_dir, end_str)]
    return _frame(rows)


def fantasypros_days(out_dir: str | Path) -> pd.DataFrame:
    """Archived Wayback/live snapshots, from the manifest `make adp-fantasypros` wrote.

    Read from the manifest rather than by re-parsing the 24 gzipped pages: this module is an
    emitter and `adp_fantasypros.build_panel` is the parser. A snapshot that archived but
    held no board is `nothing_to_capture` — the page existed and had no table on it, which
    is not the same as a day nobody asked.
    """
    manifest = _read_csv(Path(out_dir) / adp_fantasypros.MANIFEST_NAME)
    if manifest.empty:
        return _frame([])
    parsed = manifest["parsed"].astype(str).str.lower().isin(("true", "1"))
    manifest = manifest.assign(
        _parsed=parsed,
        _rows=pd.to_numeric(manifest["n_rows"], errors="coerce").fillna(0.0))
    grouped = manifest.groupby(manifest["capture_date"].astype(str))
    return _frame([{"program": ADP_FANTASYPROS, "capture_date": day,
                    "state": CAPTURED if bool(g["_parsed"].any()) else NOTHING_TO_CAPTURE,
                    "recoverable": False, "records": float(g["_rows"].sum())}
                   for day, g in grouped])


def draftkings_days(in_dir: str | Path,
                    cutoff: int = adp_draftkings.SEASON_MONTH_CUTOFF) -> pd.DataFrame:
    """The manually captured pre-draft boards. Two of them, and no way to make a third."""
    boards = adp_draftkings.load_boards(in_dir, cutoff)
    if boards.empty:
        return _frame([])
    return _frame([{"program": ADP_DRAFTKINGS, "capture_date": str(day),
                    "state": CAPTURED, "recoverable": False, "records": float(len(g))}
                   for day, g in boards.groupby("capture_date")])


def _read_csv(path: Path) -> pd.DataFrame:
    """A manifest, or an empty frame — including when the file exists and is empty.

    A run that archived nothing writes a header-less CSV, which `read_csv` raises on rather
    than returning zero rows. This emitter's whole job is to report on archives that may not
    be there, so it may not be the thing that falls over when one is not.
    """
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=str)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


# ── The program table ─────────────────────────────────────────────────────────

def summarize(calendar: pd.DataFrame, program: str, cadence: str, recovery: str,
              recovery_window_days: int, window_start: str = "",
              window_end: str = "") -> dict:
    """One program's row: its policy, its window, and the counts that go with them.

    `window_start` / `window_end` default to the program's own first and last dated row,
    which is right for an `event` program. A `daily` one passes its obligation window
    explicitly — the injury-report archive owes every day of a rolling 210, whether or not
    it reached any of them, and inferring the window from the rows would quietly shrink it
    to whatever was captured.
    """
    part = calendar[calendar["program"] == program]
    state = part["state"]
    missed = part[state == MISSED]
    days = sorted(part["capture_date"]) or [""]
    return {
        "program": program, "cadence": cadence, "recovery": recovery,
        "recovery_window_days": int(recovery_window_days),
        "window_start": window_start or days[0], "window_end": window_end or days[-1],
        "n_days": int(len(part)),
        "n_captured": int((state == CAPTURED).sum()),
        "n_nothing_to_capture": int((state == NOTHING_TO_CAPTURE).sum()),
        "n_missed": int(len(missed)),
        "n_recoverable": int(missed["recoverable"].astype(bool).sum()),
        "n_lost": int((~missed["recoverable"].astype(bool)).sum()),
        "records": float(part["records"].sum()),
    }


def build(cfg: dict, end: date | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Both frames, from disk only."""
    data = cfg.get("data", {})
    ir_cfg = data.get("injury_reports", {})
    adp_cfg = data.get("adp", {})
    end = end or date.today()
    retention = int(ir_cfg.get("retention_days", injury_reports.RETENTION_DAYS))

    ir_dir = ir_cfg.get("dir", "data/raw/injury_reports")
    espn_dir = data.get("injuries", {}).get("dir", "data/raw/injuries")
    fp_dir = adp_cfg.get("fantasypros", {}).get("dir", "data/raw/adp/fantasypros")
    dk_dir = adp_cfg.get("draftkings", {}).get("dir", "data/raw/dk_draft_rankings")

    calendar = pd.concat([
        injury_report_days(ir_dir, end, retention,
                           ir_cfg.get("time_key", injury_reports.DEADLINE_TIME_KEY)),
        espn_days(espn_dir, end),
        fantasypros_days(fp_dir),
        draftkings_days(dk_dir, int(adp_cfg.get("draftkings", {}).get(
            "season_month_cutoff", adp_draftkings.SEASON_MONTH_CUTOFF))),
    ], ignore_index=True)

    espn_start = min(injuries.snapshot_dates(espn_dir), default="")
    programs = pd.DataFrame([
        summarize(calendar, INJURY_REPORTS, DAILY, WINDOW, retention,
                  window_start=(end - timedelta(days=retention)).isoformat(),
                  window_end=end.isoformat()),
        summarize(calendar, ESPN_INJURIES, DAILY, NEVER, 0,
                  window_start=espn_start, window_end=end.isoformat()),
        summarize(calendar, ADP_FANTASYPROS, EVENT, ARCHIVE, 0),
        summarize(calendar, ADP_DRAFTKINGS, EVENT, NEVER, 0),
    ], columns=PROGRAM_COLUMNS)
    return calendar, programs


def run(cfg: dict, end: date | None = None) -> Path:
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    calendar, programs = build(cfg, end)

    print("Capture coverage — read from disk, no requests made")
    for row in programs.itertuples(index=False):
        span = f"{row.window_start or '—'} → {row.window_end or '—'}"
        print(f"  {row.program:<17} {row.n_captured:>4} captured  "
              f"{row.n_nothing_to_capture:>3} nothing to capture  "
              f"{row.n_missed:>3} missed  ({span})")
    lost = int(programs["n_lost"].sum())
    recoverable = int(programs["n_recoverable"].sum())
    if recoverable:
        print(f"  → {recoverable} missed day(s) still recoverable; "
              f"`make daily-capture` fetches them until they age out.")
    if lost:
        print(f"  → {lost} missed day(s) are PERMANENTLY LOST — no refetch exists.")
    if not (lost or recoverable):
        print("  → No gaps.")

    dest = out_dir / CALENDAR_NAME
    calendar.to_csv(dest, index=False)
    print(f"Wrote {len(calendar):,} program-days → {dest}")
    pdest = out_dir / PROGRAMS_NAME
    programs.to_csv(pdest, index=False)
    print(f"Wrote {len(programs):,} capture programs → {pdest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Write the capture programs' coverage calendar. Makes no requests.")
    parser.add_argument("--end", default=None,
                        help="Treat this day as today (YYYY-MM-DD), for a fixed readout.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, date.fromisoformat(args.end) if args.end else None)
