"""Archive and parse the NBA's official injury-report PDFs.

**This module has a deadline attached.** The CDN keeps roughly seven months of reports on
a rolling basis and there is no index, no archive and no way to ask for an older one: a
date that has aged out returns 403 forever. Every day the archiver does not run is a day
of injury history permanently lost. That is why stage A of `docs/availability-plan.md`
runs before the box-score backfill, which can wait indefinitely because it is
backfillable.

The URL is fully predictable, so no index scraping is needed:

    https://ak-static.cms.nba.com/referee/injury/Injury-Report_YYYY-MM-DD_HH_MMAM.pdf

Reports are published every 30 minutes on game days. One per day is enough for this
project — the **5:00 PM ET** report is the league-mandated deadline report, and capturing
the full 30-minute grid would cost ~8,000 requests a season against ~170 for the daily
one. A 403 means that key does not exist: either the date has aged out, or there were no
games (the All-Star break is absent from the middle of a live window, which is a schedule
artifact, not retention). Neither is retryable, so a 403 is recorded and skipped rather
than backed off.

Each report carries a per-player participation status (Out / Doubtful / Questionable /
Probable / Available) **with a stated reason**, which is richer than anything in the
box-score data and is the only source that gives a *dated forecast* rather than a
resolved outcome.

**Point-in-time discipline.** Every parsed row carries `report_date`, the date the league
published it. That is the row's `as_of_date` and it is what makes this source safe to
build historical features from — unlike the ESPN feed, which describes today and can only
ever describe today. Rows are never revised: a later report is a new row, not an edit.

The raw PDFs are kept alongside the parsed CSVs. Parsing is a guess about a layout that
the league can change; the PDF is the archive, and `--reparse` rebuilds every CSV from
disk without touching the network.

Usage:
    python -m src.data.injury_reports                 # everything still retained → today
    python -m src.data.injury_reports --days 30       # just the last 30 days
    python -m src.data.injury_reports --date 2026-03-03
    python -m src.data.injury_reports --reparse       # re-parse archived PDFs, no network
"""

import argparse
import io
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import yaml
from pypdf import PdfReader

REPORT_URL = ("https://ak-static.cms.nba.com/referee/injury/"
              "Injury-Report_{day}_{time_key}.pdf")

# The league-mandated deadline report. Reports exist every 30 minutes; this is the one
# that matters and the only one worth 170 requests a season.
DEADLINE_TIME_KEY = "05_00PM"

# Retention is ~7 months, rolling. Probed on 2026-07-27: 2025-12-20 was already gone and
# 2025-12-27 was present. 210 days reaches past that edge without wasting many requests.
RETENTION_DAYS = 210

# The CDN throttled during planning, so this is deliberately slower than the nba_api
# delay. A full retention-window sweep is ~210 requests; at 3 s that is ~10 minutes, run
# once. The daily run makes exactly one request.
DELAY = 3.0
MAX_RETRIES = 4
TIMEOUT = 60

# A recorded 403 normally means "this key does not exist" and is never retried — the date
# aged out, or there were no games. The exception is the last few days: a capture that ran
# before 5:00 PM ET sees a 403 for a report that is published later the same evening, and
# without this the archiver would write that day off permanently. Costs at most 3 requests
# a run, against the one failure mode that loses history the archiver exists to keep.
RECHECK_DAYS = 3

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

MANIFEST_NAME = "_injury_report_manifest.csv"
LOG_NAME = "injury_reports_log.csv"

# The participation vocabulary. A parsed row is only kept if its status is one of these,
# which is also what filters out the page header, the title band and the page footer
# without pattern-matching every one of them.
STATUSES = {"Out", "Doubtful", "Questionable", "Probable", "Available"}

# A team that missed the filing deadline gets a row with this in the *reason* column and
# no player and no status. It is kept rather than dropped, because it is the difference
# between "nobody on this team is hurt" and "this team did not say" — without it, a
# missing Out row reads as evidence of health when it is evidence of nothing.
NOT_SUBMITTED = "NOT YET SUBMITTED"

# Column layout. Cells are left-aligned and the header row appears on page 1 only, so the
# x positions are read off that header when present and fall back to these, measured on
# 2026-03-03. A chunk belongs to the *last* column whose left edge is at or before it,
# which is robust to a long name wrapping deep into its cell — midpoint boundaries are
# not.
COLUMNS = ["game_date", "game_time", "matchup", "team",
           "player_name", "status", "reason"]
DEFAULT_COLUMN_X = [23.1, 119.6, 200.0, 264.2, 425.0, 585.4, 666.5]
HEADER_LABELS = ["Game Date", "Game Time", "Matchup", "Team",
                 "Player Name", "Current Status", "Reason"]

# Chunks within this many points of each other are the same visual line. Rows are ~22 pt
# apart and a wrapped reason line sits ~7 pt off its player row, so 3 separates them.
LINE_TOLERANCE = 3.0
COLUMN_TOLERANCE = 2.0

# Half a plain row, used as the top edge of a page's first row when the page holds only
# one. Anything above that edge is the tail of a reason that wrapped across the page
# break, not part of this page's first row.
DEFAULT_HALF_ROW = 11.0

# Lines that are page furniture rather than data. Matched against assembled *lines*, so
# the page footer arrives as "Page 7" in the team column and "of 8" in the player column.
_NOISE = [
    re.compile(r"^Injury\s+Report:?$", re.I),
    re.compile(r"^\d{2}/\d{2}/\d{2}\s+\d{2}:\d{2}$"),
    re.compile(r"^[AP]M$"),
    re.compile(r"^Page(\s+\d+)?$"),
    re.compile(r"^(\d+\s+)?of\s+\d+$"),
] + [re.compile(rf"^{re.escape(lbl)}$") for lbl in HEADER_LABELS]


# ── URLs and dates ────────────────────────────────────────────────────────────

def report_url(day: date, time_key: str = DEADLINE_TIME_KEY) -> str:
    return REPORT_URL.format(day=day.isoformat(), time_key=time_key)


def report_filename(day: date, time_key: str = DEADLINE_TIME_KEY) -> str:
    """The CDN's own filename, kept verbatim so the archive is addressable by key."""
    return f"Injury-Report_{day.isoformat()}_{time_key}.pdf"


def date_range(start: date, end: date) -> list[date]:
    return [start + timedelta(days=i) for i in range((end - start).days + 1)]


def _parse_day(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


# ── Parsing ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Collapse the extractor's token spacing and repair hyphenated wraps.

    GemBox emits each word as its own text operation, so joining with a space turns
    `Two-Way` into `Two- Way`. A hyphen with no space *before* it is always a wrap;
    `Injury/Illness - Left Knee` has spaces on both sides and is left alone.
    """
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"(?<=\S)- (?=\S)", "-", text)


def _chunks(page) -> list[tuple[float, float, str]]:
    """Every text run on the page as (x, y, text).

    The pages are landscape with a flipped text matrix: y *increases* down the page, so
    rows sort ascending. `extract_text` is called only for its visitor side effect.
    """
    out: list[tuple[float, float, str]] = []

    def visit(text, cm, tm, font_dict, font_size):
        stripped = text.strip()
        if stripped:
            out.append((round(tm[4], 1), round(tm[5], 1), stripped))

    page.extract_text(visitor_text=visit)
    return out


def header_columns(chunks: list[tuple[float, float, str]]) -> list[float] | None:
    """Left edge of each column, read off the header row if this page carries one."""
    by_y: dict[float, list[tuple[float, str]]] = {}
    for x, y, text in chunks:
        by_y.setdefault(y, []).append((x, text))
    for y, items in by_y.items():
        items.sort()
        joined = _clean(" ".join(t for _, t in items))
        if joined != " ".join(HEADER_LABELS):
            continue
        # The header labels are multi-token; each column starts at its first token.
        starts, seen = [], 0
        for label in HEADER_LABELS:
            starts.append(items[seen][0])
            seen += len(label.split())
        return starts
    return None


def _column_of(x: float, column_x: list[float]) -> str:
    idx = 0
    for i, left in enumerate(column_x):
        if left <= x + COLUMN_TOLERANCE:
            idx = i
    return COLUMNS[idx]


def _column_lines(chunks: list[tuple[float, float, str]],
                  column_x: list[float]) -> list[tuple[str, float, str]]:
    """Collapse chunks into (column, y, text) lines, one per column per visual row."""
    by_col: dict[str, list[tuple[float, float, str]]] = {}
    for x, y, text in chunks:
        by_col.setdefault(_column_of(x, column_x), []).append((y, x, text))

    lines: list[tuple[str, float, str]] = []
    for col, items in by_col.items():
        items.sort(key=lambda i: (i[0], i[1]))
        line_y, buf = None, []
        for y, _x, text in items:
            if line_y is not None and abs(y - line_y) > LINE_TOLERANCE:
                lines.append((col, line_y, _clean(" ".join(buf))))
                buf = []
                line_y = y
            else:
                line_y = y if line_y is None else line_y
            buf.append(text)
        if buf:
            lines.append((col, line_y, _clean(" ".join(buf))))
    return [ln for ln in lines if not any(p.match(ln[2]) for p in _NOISE)]


def _page_rows(lines: list[tuple[str, float, str]]) -> tuple[list[dict], dict[str, str]]:
    """Attach every line on one page to the row it belongs to.

    A wrapped reason is typeset *centred* on its row — the first line sits above the
    player name and the continuation below — so lines cannot be grouped by equal y.
    Instead each row gets an anchor, and every other line goes to the anchor whose
    midpoint interval contains it. Rows are ~22 pt apart against a ~7 pt wrap offset, so
    the intervals are not close to ambiguous.

    Two kinds of line anchor a row: a player name, and a bare `NOT YET SUBMITTED` in the
    reason column, which is a team-level row with no player at all. Without the second,
    an unfiled team's row is swallowed by whichever player row happens to be nearest and
    corrupts it.

    Returns the page's rows plus any **orphan** lines above the first row — the tail of a
    reason that wrapped across the page break, which belongs to the last row of the
    previous page.
    """
    anchors = sorted({y for col, y, text in lines
                      if col == "player_name"
                      or (col == "reason" and text.upper() == NOT_SUBMITTED)})
    orphan: dict[str, list[str]] = {c: [] for c in COLUMNS}
    if not anchors:
        for col, _y, text in sorted(lines, key=lambda ln: ln[1]):
            orphan[col].append(text)
        return [], {c: _clean(" ".join(v)) for c, v in orphan.items()}

    half = ((anchors[1] - anchors[0]) / 2 if len(anchors) > 1 else DEFAULT_HALF_ROW)
    bounds = []
    for i, y in enumerate(anchors):
        lo = y - half if i == 0 else (anchors[i - 1] + y) / 2
        hi = float("inf") if i == len(anchors) - 1 else (y + anchors[i + 1]) / 2
        bounds.append((lo, hi))

    rows: list[dict] = [{c: [] for c in COLUMNS} for _ in anchors]
    for col, y, text in sorted(lines, key=lambda ln: ln[1]):
        if y < bounds[0][0]:
            orphan[col].append(text)
            continue
        for i, (lo, hi) in enumerate(bounds):
            if lo <= y < hi:
                rows[i][col].append(text)
                break
    return ([{c: _clean(" ".join(v)) for c, v in row.items()} for row in rows],
            {c: _clean(" ".join(v)) for c, v in orphan.items()})


def _split_reason(reason: str) -> tuple[str, str, str]:
    """`Injury/Illness - Left Knee; Surgery` → category, body part, detail.

    Body part is only meaningful under `Injury/Illness`; `G League - Two-Way` has a
    roster mechanic where the body part would be, so it is left empty there rather than
    inventing an anatomy.
    """
    # The league prints a bare hyphen for an Available player with no stated reason.
    # That is genuinely "no reason given", not a category called "-".
    if not reason or reason.strip() == "-":
        return "", "", ""
    category, _, rest = reason.partition(" - ")
    category, rest = category.strip(), rest.strip()
    if category != "Injury/Illness":
        return category, "", rest
    body_part, _, detail = rest.partition(";")
    return category, body_part.strip(), detail.strip()


def parse_report(data: bytes | str | Path, report_date: date | None = None,
                 time_key: str = DEADLINE_TIME_KEY) -> pd.DataFrame:
    """Parse one report PDF into one row per (game, team, player).

    `game_date`, `game_time`, `matchup` and `team` are printed once per block and blank
    on continuation rows — including across a page break, where the matchup reappears but
    the date does not — so they are forward-filled over the whole document.
    """
    if isinstance(data, (str, Path)):
        source = Path(data)
        reader = PdfReader(str(source))
        source_name = source.name
    else:
        reader = PdfReader(io.BytesIO(data))
        source_name = report_filename(report_date, time_key) if report_date else ""

    published_at = _creation_date(reader)
    column_x = None
    records: list[dict] = []
    for page in reader.pages:
        chunks = _chunks(page)
        column_x = header_columns(chunks) or column_x or DEFAULT_COLUMN_X
        page_records, orphan = _page_rows(_column_lines(chunks, column_x))
        # A reason that wrapped across the page break resumes at the top of this page;
        # it is the tail of the previous page's last row, not the head of this one's.
        if records:
            for col, text in orphan.items():
                if text:
                    records[-1][col] = _clean(f"{records[-1][col]} {text}")
        records += page_records

    if not records:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    df = pd.DataFrame(records)
    for col in ["game_date", "game_time", "matchup", "team"]:
        df[col] = df[col].replace("", None).ffill()

    unfiled = (df["reason"].str.upper() == NOT_SUBMITTED)
    df = df[df["status"].isin(STATUSES) | unfiled].copy()
    if df.empty:
        return pd.DataFrame(columns=_OUTPUT_COLS)

    unfiled = df["reason"].str.upper() == NOT_SUBMITTED
    df.loc[unfiled, ["status", "player_name", "reason"]] = [NOT_SUBMITTED, "", ""]

    reasons = df["reason"].map(_split_reason)
    df["reason_category"] = [r[0] for r in reasons]
    df["body_part"] = [r[1] for r in reasons]
    df["reason_detail"] = [r[2] for r in reasons]

    df.insert(0, "report_date",
              report_date.isoformat() if report_date else _report_date(published_at))
    df.insert(1, "report_time", time_key)
    df.insert(2, "published_at", published_at)
    df["source_file"] = source_name
    return df[_OUTPUT_COLS].reset_index(drop=True)


_OUTPUT_COLS = ["report_date", "report_time", "published_at", "game_date", "game_time",
                "matchup", "team", "player_name", "status", "reason", "reason_category",
                "body_part", "reason_detail", "source_file"]


def _creation_date(reader: PdfReader) -> str:
    """The PDF's own `CreationDate`, which matches the filename key exactly.

    `D:20260303170004-05'00'` ↔ `2026-03-03_05_00PM`, verified during planning. It is the
    league's own publication timestamp, so it is stored as the authoritative one.
    """
    raw = (reader.metadata or {}).get("/CreationDate", "")
    m = re.match(r"D:(\d{4})(\d{2})(\d{2})(\d{2})(\d{2})(\d{2})", str(raw))
    if not m:
        return str(raw)
    y, mo, d, h, mi, s = m.groups()
    return f"{y}-{mo}-{d}T{h}:{mi}:{s}"


def _report_date(published_at: str) -> str:
    return published_at[:10] if len(published_at) >= 10 else ""


# ── Fetching ──────────────────────────────────────────────────────────────────

def fetch_pdf(day: date, time_key: str = DEADLINE_TIME_KEY,
              session: requests.Session | None = None, delay: float = DELAY,
              max_retries: int = MAX_RETRIES) -> tuple[int, bytes | None]:
    """Download one report. Returns (http status, bytes or None).

    A 403 is the CDN saying the key does not exist — the date aged out, or there were no
    games. Neither is retryable and backoff does not help, so it returns immediately.
    Throttling (429) and server errors do get backed off, because the CDN throttled
    during planning and a sweep that trips it should slow down rather than give up.
    """
    session = session or requests.Session()
    url = report_url(day, time_key)
    status = 0
    for attempt in range(max_retries + 1):
        try:
            resp = session.get(url, headers=_HEADERS, timeout=TIMEOUT)
        except requests.RequestException as exc:
            status = 0
            print(f"  {day} network error ({exc.__class__.__name__}); backing off")
        else:
            status = resp.status_code
            if status == 200:
                return status, resp.content
            if status in (403, 404):
                return status, None
        if attempt < max_retries:
            wait = delay * 2 ** attempt
            print(f"  {day} HTTP {status}; retrying in {wait:.0f}s")
            time.sleep(wait)
    return status, None


# ── Archive ───────────────────────────────────────────────────────────────────

def _read_manifest(out_dir: Path) -> pd.DataFrame:
    dest = out_dir / MANIFEST_NAME
    if not dest.exists():
        return pd.DataFrame(columns=["report_date", "report_time", "http_status",
                                     "bytes", "rows", "published_at", "fetched_at",
                                     "pdf_file"])
    return pd.read_csv(dest, dtype={"report_date": str})


def _write_manifest(manifest: pd.DataFrame, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / MANIFEST_NAME
    manifest.sort_values(["report_date", "report_time"]).to_csv(dest, index=False)
    return dest


def archive_day(day: date, out_dir: str | Path, time_key: str = DEADLINE_TIME_KEY,
                session: requests.Session | None = None, delay: float = DELAY,
                reparse: bool = False) -> dict | None:
    """Capture and parse one day. Returns a manifest row, or None if nothing was done.

    Idempotent and resumable in both directions: an archived PDF is never re-downloaded,
    and `reparse` rebuilds the CSV from the PDF already on disk without touching the
    network — which is what makes the parser safe to change later.
    """
    out_dir = Path(out_dir)
    pdf_dir = out_dir / "pdf"
    pdf_path = pdf_dir / report_filename(day, time_key)
    csv_path = out_dir / f"injury_report_{day.isoformat()}.csv"

    if pdf_path.exists() and csv_path.exists() and not reparse:
        return None
    if not pdf_path.exists():
        if reparse:
            return None
        status, content = fetch_pdf(day, time_key, session, delay)
        if content is None:
            reason = "no report published" if status == 403 else f"HTTP {status}"
            print(f"  {day}  —  {reason}")
            return {"report_date": day.isoformat(), "report_time": time_key,
                    "http_status": status, "bytes": 0, "rows": 0, "published_at": "",
                    "fetched_at": datetime.now().astimezone().isoformat(),
                    "pdf_file": ""}
        pdf_dir.mkdir(parents=True, exist_ok=True)
        pdf_path.write_bytes(content)
    else:
        status = 200

    df = parse_report(pdf_path, day, time_key)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    published = df["published_at"].iloc[0] if len(df) else ""
    print(f"  {day}  {len(df):>4,} player rows  →  {csv_path}")
    return {"report_date": day.isoformat(), "report_time": time_key,
            "http_status": status, "bytes": pdf_path.stat().st_size, "rows": len(df),
            "published_at": published,
            "fetched_at": datetime.now().astimezone().isoformat(),
            "pdf_file": pdf_path.name}


def should_attempt(day: date, key: tuple[str, str], known: set[tuple[str, str]],
                   pdf_exists: bool, end: date, reparse: bool = False,
                   recheck_days: int = RECHECK_DAYS) -> bool:
    """Whether to spend a request on this day.

    A recorded 403 is normally final — the key does not exist and never will. The
    exception is the last `recheck_days`: a capture that ran before the 5:00 PM ET
    publication sees a 403 for a report that appears later the same evening, and treating
    that as final would write the day off permanently. This archive cannot be backfilled,
    so the asymmetry is deliberate: three wasted requests a run against losing a day.
    """
    if reparse:
        return False
    if pdf_exists:
        return False
    return key not in known or day >= end - timedelta(days=recheck_days)


def rebuild_log(out_dir: str | Path) -> Path:
    """Concatenate the per-day CSVs into one cumulative log.

    Rebuilt rather than appended so a re-run or a re-parse cannot duplicate rows. The
    per-day files are the source of truth; this is a convenience for downstream readers.
    """
    out_dir = Path(out_dir)
    files = sorted(out_dir.glob("injury_report_*.csv"))
    frames = [pd.read_csv(f, dtype=str) for f in files]
    frames = [f for f in frames if len(f)]
    log = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=_OUTPUT_COLS))
    dest = out_dir / LOG_NAME
    log.to_csv(dest, index=False)
    print(f"Rebuilt log: {len(log):,} player-report rows from "
          f"{len(files):,} reports → {dest}")
    return dest


def load_log(out_dir: str | Path) -> pd.DataFrame:
    """The cumulative archive, one row per (report, game, team, player)."""
    path = Path(out_dir) / LOG_NAME
    if not path.exists():
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return pd.read_csv(path, dtype=str).fillna("")


def report_as_of(log: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """The latest report published at or before `as_of` — and nothing after it.

    The supported way to read this archive for a historical row, mirroring
    `injuries.snapshot_as_of`. Unlike the ESPN feed this source *is* dated at publication,
    so it is legitimate for history — but only if it is read that way. A plain filter on
    the log would still hand a row every later report about the same injury, which is the
    resolved outcome rather than a forecast.
    """
    if log.empty:
        return log
    eligible = log[log["report_date"] <= as_of]
    if eligible.empty:
        return eligible
    return eligible[eligible["report_date"] == eligible["report_date"].max()].copy()


def capture_status(out_dir: str | Path, end: date | None = None,
                   retention_days: int = RETENTION_DAYS,
                   time_key: str = DEADLINE_TIME_KEY) -> pd.DataFrame:
    """One row per day in the retention window: archived / no report / **missed**.

    The distinction that matters is the third one. A day the CDN 403s is a day with no
    report to have — the offseason, the All-Star break — and is not a failure. A day that
    was never attempted is a **run that did not happen**, and it is still recoverable
    right up until it ages out of the ~7-month window. Without this report the two look
    identical from the outside, and a scheduler that silently stopped firing would only
    become visible once the days were already gone.
    """
    out_dir = Path(out_dir)
    end = end or date.today()
    manifest = _read_manifest(out_dir)
    attempted = set(zip(manifest["report_date"].astype(str),
                        manifest["report_time"].astype(str)))

    rows = []
    for day in date_range(end - timedelta(days=retention_days), end):
        archived = (out_dir / "pdf" / report_filename(day, time_key)).exists()
        if archived:
            state = "archived"
        elif (day.isoformat(), time_key) in attempted:
            state = "no_report_published"
        else:
            state = "missed"
        rows.append({"report_date": day.isoformat(), "state": state,
                     "recoverable": state == "missed"})
    return pd.DataFrame(rows)


def print_status(out_dir: str | Path, end: date | None = None,
                 retention_days: int = RETENTION_DAYS) -> pd.DataFrame:
    status = capture_status(out_dir, end, retention_days)
    counts = status["state"].value_counts().to_dict()
    print(f"NBA injury-report archive — {retention_days}-day retention window")
    for state in ("archived", "no_report_published", "missed"):
        print(f"  {state:<22} {counts.get(state, 0):>4}")

    missed = status.loc[status["recoverable"], "report_date"].tolist()
    if missed:
        shown = ", ".join(missed[:8]) + (" ..." if len(missed) > 8 else "")
        print(f"  {len(missed)} day(s) never attempted and still recoverable: {shown}")
        print("  → `make injury-reports` fetches them; they are lost only once they age "
              "out of the window.")
    else:
        print("  No gaps: every day in the window is either archived or has no report.")
    return status


def run(cfg: dict, days: int | None = None, start: date | None = None,
        end: date | None = None, reparse: bool = False) -> Path:
    ir_cfg = cfg.get("data", {}).get("injury_reports", {})
    out_dir = Path(ir_cfg.get("dir", "data/raw/injury_reports"))
    time_key = ir_cfg.get("time_key", DEADLINE_TIME_KEY)
    delay = float(ir_cfg.get("delay", DELAY))
    retention = int(ir_cfg.get("retention_days", RETENTION_DAYS))

    end = end or date.today()
    start = start or (end - timedelta(days=(days if days is not None else retention)))
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _read_manifest(out_dir)
    known = set(zip(manifest["report_date"].astype(str),
                    manifest["report_time"].astype(str)))
    days_to_do = date_range(start, end)
    print(f"Injury reports {start} → {end} ({len(days_to_do):,} days, {time_key})"
          + (" [reparse, no network]" if reparse else ""))

    session = requests.Session()
    rows, fetched = [], 0
    for i, day in enumerate(days_to_do):
        pdf_exists = (out_dir / "pdf" / report_filename(day, time_key)).exists()
        key = (day.isoformat(), time_key)
        if not (pdf_exists or should_attempt(day, key, known, pdf_exists, end, reparse)):
            continue                       # a settled 403: the key does not exist
        row = archive_day(day, out_dir, time_key, session, delay, reparse)
        if row is None:
            continue
        rows.append(row)
        if row["bytes"] and not pdf_exists:
            fetched += 1
            if i < len(days_to_do) - 1:
                time.sleep(delay)          # be polite: it throttled during planning

    if rows:
        new = pd.DataFrame(rows)
        manifest = manifest[~manifest.set_index(["report_date", "report_time"]).index.isin(
            new.set_index(["report_date", "report_time"]).index)]
        manifest = pd.concat([manifest, new], ignore_index=True)
        _write_manifest(manifest, out_dir)
    archived = pd.to_numeric(manifest["rows"], errors="coerce").fillna(0) > 0
    print(f"Captured {fetched:,} new reports; {int(archived.sum()):,} archived in total")
    return rebuild_log(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Archive NBA official injury reports.")
    parser.add_argument("--days", type=int, default=None,
                        help="How many days back from today to sweep "
                             f"(default: the {RETENTION_DAYS}-day retention window).")
    parser.add_argument("--start", type=_parse_day, default=None, help="YYYY-MM-DD")
    parser.add_argument("--end", type=_parse_day, default=None, help="YYYY-MM-DD")
    parser.add_argument("--date", type=_parse_day, default=None,
                        help="A single day (shorthand for --start D --end D).")
    parser.add_argument("--reparse", action="store_true",
                        help="Re-parse archived PDFs from disk; makes no requests.")
    parser.add_argument("--status", action="store_true",
                        help="Report which days are archived, have no report, or were "
                             "missed. Makes no requests.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    ir_cfg = cfg.get("data", {}).get("injury_reports", {})
    if args.status:
        print_status(ir_cfg.get("dir", "data/raw/injury_reports"),
                     retention_days=int(ir_cfg.get("retention_days", RETENTION_DAYS)))
        raise SystemExit(0)
    start, end = args.start, args.end
    if args.date:
        start = end = args.date
    run(cfg, days=args.days, start=start, end=end, reparse=args.reparse)
