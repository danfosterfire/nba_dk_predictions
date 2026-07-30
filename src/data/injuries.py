"""Scrape NBA injury reports from the ESPN public API and persist to a timestamped log.

ESPN's /nba/injuries page is JS-rendered, but its underlying JSON API is stable and
public. Each run writes a timestamped snapshot CSV and appends to a cumulative log.

**This feed has no history and can never acquire any.** It describes today, and only
today: there is no dated archive to backfill from, so the cumulative log grows one day at
a time and every day the capture does not run is a hole that stays a hole. That is why it
is on the same daily schedule as the NBA report PDFs — see `make daily-capture`.

## Point-in-time discipline

This is the source most likely to quietly invalidate the availability head, because it is
the most tempting one. It reports a player's status *right now*. Filling a 2019 training
row from a snapshot scraped in 2026 does not tell the model who was hurt in 2019 — it
tells the model how long the injury actually lasted, which is the target. Two rules make
that mistake hard to make:

- **Every row carries `snapshot_date`, and rows are never revised.** A changed status is a
  new row on a new date, not an edit to an old one. `append_to_log` only ever appends.
- **`return_date_forecast` is a forecast made on `snapshot_date`.** ESPN's `returnDate`
  is an expectation, not an outcome, and it must never be overwritten with the realized
  return — store that separately, as the target. The column is named for what it is;
  snapshots written before the rename carry it as `return_date` and `load_log` migrates
  them on read rather than rewriting the archive.
- **Read history through `snapshot_as_of`**, which returns the most recent snapshot at or
  before a date and nothing from after it. A direct read of the log is a leak waiting to
  happen.

Usage:
    python -m src.data.injuries                  # scrape once, append to log
    python -m src.data.injuries --daily          # no-op if today is already captured
    python -m src.data.injuries --snapshot-only  # save snapshot, skip log append
    python -m src.data.injuries --output-dir path/to/dir
"""

import argparse
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests
import yaml

ESPN_API_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

# One snapshot a day is the whole archive; a second one the same evening adds nothing but
# duplicate rows. 20 h rather than 24 so a cron drifting later never skips a day.
MIN_HOURS_BETWEEN_SNAPSHOTS = 20

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_FIELDNAMES = [
    "snapshot_date",
    "scraped_at",
    "team",
    "player",
    "position",
    "status",
    "injury_date",
    "injury_type",
    "injury_location",
    "injury_detail",
    "injury_side",
    "return_date_forecast",
    "fantasy_status",
    "short_comment",
    "long_comment",
]

# Snapshots written before the point-in-time rename. Migrated on read, not on disk — the
# archive is the record of what was scraped and when, and rewriting it would defeat the
# purpose of keeping one.
_LEGACY_COLUMNS = {"return_date": "return_date_forecast"}

_SNAPSHOT_RE = re.compile(r"injuries_(\d{4}-\d{2}-\d{2})_\d{2}-\d{2}-\d{2}\.csv$")


def fetch_injuries(url: str = ESPN_API_URL, timeout: int = 30) -> list[dict]:
    """Call the ESPN NBA injuries API and return a flat list of injury records.

    Returns an empty list on network error or unexpected response shape.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    resp.raise_for_status()
    return _parse_response(resp.json())


def _parse_response(data: dict) -> list[dict]:
    records: list[dict] = []
    for team_block in data.get("injuries", []):
        team_name = team_block.get("displayName", "Unknown")
        for inj in team_block.get("injuries", []):
            athlete = inj.get("athlete", {})
            pos = athlete.get("position", {})
            details = inj.get("details", {})
            fantasy = details.get("fantasyStatus", {})

            records.append({
                "team":             team_name,
                "player":           athlete.get("displayName", ""),
                "position":         pos.get("abbreviation", ""),
                "status":           inj.get("status", ""),
                "injury_date":      inj.get("date", ""),
                "injury_type":      details.get("type", ""),
                "injury_location":  details.get("location", ""),
                "injury_detail":    details.get("detail", ""),
                "injury_side":      details.get("side", ""),
                # A forecast made on the snapshot date, never the realized return.
                "return_date_forecast": details.get("returnDate", ""),
                "fantasy_status":   fantasy.get("abbreviation", ""),
                "short_comment":    inj.get("shortComment", ""),
                "long_comment":     inj.get("longComment", ""),
            })
    return records


# ── Persistence ───────────────────────────────────────────────────────────────

def save_snapshot(records: list[dict], output_dir: Path, scraped_at: datetime) -> Path:
    """Write a single-scrape CSV snapshot named by timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = scraped_at.strftime("%Y-%m-%d_%H-%M-%S")
    dest = output_dir / f"injuries_{ts}.csv"
    df = pd.DataFrame(records)
    df.insert(0, "scraped_at", scraped_at.isoformat())
    df.insert(0, "snapshot_date", scraped_at.date().isoformat())
    df.to_csv(dest, index=False)
    print(f"  Snapshot saved ({len(records)} records) → {dest}")
    return dest


def read_snapshot(path: str | Path) -> pd.DataFrame:
    """One snapshot CSV, migrated to the current schema.

    Snapshots predating `snapshot_date` are dated from their `scraped_at`, and the
    pre-rename `return_date` becomes `return_date_forecast` — the same values under a
    name that says what they are. Files on disk are never rewritten: this is a read-time
    migration, because the archive is the record of what was scraped and when.
    """
    path = Path(path)
    df = pd.read_csv(path, dtype=str).rename(columns=_LEGACY_COLUMNS)
    if "snapshot_date" not in df.columns:
        m = _SNAPSHOT_RE.search(path.name)
        stamped = df["scraped_at"].str.slice(0, 10) if "scraped_at" in df else ""
        df.insert(0, "snapshot_date", m.group(1) if m else stamped)
    for col in _FIELDNAMES:
        if col not in df.columns:
            df[col] = ""
    return df[_FIELDNAMES]


def rebuild_log(output_dir: str | Path) -> Path:
    """Concatenate every snapshot into the cumulative log, oldest first.

    Rebuilt rather than appended so a schema change or a re-run cannot corrupt it — the
    per-snapshot CSVs are the source of truth and are never deleted, so this is lossless.
    The log stays append-only *in content*: each snapshot is a dated observation and a
    changed status appears as a new row, never as an edit to an old one.
    """
    output_dir = Path(output_dir)
    files = sorted(f for f in output_dir.glob("injuries_*.csv")
                   if _SNAPSHOT_RE.search(f.name))
    frames = [read_snapshot(f) for f in files]
    frames = [f for f in frames if len(f)]
    log = (pd.concat(frames, ignore_index=True) if frames
           else pd.DataFrame(columns=_FIELDNAMES))
    dest = output_dir / "injuries_log.csv"
    log.to_csv(dest, index=False)
    print(f"  Rebuilt log: {len(log):,} records from {len(files):,} snapshots → {dest}")
    return dest


# ── Point-in-time reads ───────────────────────────────────────────────────────

def snapshot_dates(output_dir: str | Path) -> list[str]:
    """Dates that have a snapshot on disk, ascending."""
    days = {m.group(1) for f in Path(output_dir).glob("injuries_*.csv")
            if (m := _SNAPSHOT_RE.search(f.name))}
    return sorted(days)


def last_snapshot_at(output_dir: str | Path) -> datetime | None:
    """Timestamp of the most recent snapshot file, or None if there is none."""
    stamps = []
    for f in Path(output_dir).glob("injuries_*.csv"):
        if _SNAPSHOT_RE.search(f.name):
            stamps.append(datetime.strptime(f.stem, "injuries_%Y-%m-%d_%H-%M-%S"))
    return max(stamps) if stamps else None


def is_captured(output_dir: str | Path, now: datetime | None = None,
                min_hours: float = MIN_HOURS_BETWEEN_SNAPSHOTS) -> bool:
    """True if a snapshot is recent enough that another one would add nothing."""
    last = last_snapshot_at(output_dir)
    if last is None:
        return False
    now = (now or datetime.now()).replace(tzinfo=None)
    return now - last < timedelta(hours=min_hours)


def missing_days(output_dir: str | Path, end: str | None = None) -> list[str]:
    """Calendar days since the first snapshot that have none — **irrecoverable holes**.

    Unlike the NBA PDF archive, nothing here can be refetched: the feed describes today
    and keeps no history, so a day the capture did not run is gone permanently. This
    exists so that fact is visible rather than inferred later from a hole in the data.
    """
    days = snapshot_dates(output_dir)
    if not days:
        return []
    end = end or datetime.now().date().isoformat()
    span = pd.date_range(days[0], end, freq="D").strftime("%Y-%m-%d")
    return [d for d in span if d not in set(days)]


def print_status(output_dir: str | Path = "data/raw/injuries") -> None:
    days = snapshot_dates(output_dir)
    missing = missing_days(output_dir)
    print("ESPN injury feed — current-status, forward-only, NOT backfillable")
    if not days:
        print("  No snapshots yet.")
        return
    print(f"  snapshot days          {len(days):>4}  ({days[0]} → {days[-1]})")
    print(f"  missing days           {len(missing):>4}  — permanently lost, no refetch")
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        print(f"  gaps: {shown}")


def load_log(output_dir: str | Path = "data/raw/injuries") -> pd.DataFrame:
    """The cumulative log, migrated to the current schema."""
    path = Path(output_dir) / "injuries_log.csv"
    if not path.exists():
        return pd.DataFrame(columns=_FIELDNAMES)
    return read_snapshot(path)


def snapshot_as_of(log: pd.DataFrame, as_of: str) -> pd.DataFrame:
    """The most recent snapshot at or before `as_of` — and nothing after it.

    The one supported way to read this feed for a historical row. Reading the log
    directly would hand a 2019 training row the resolved 2026 outcome, which is leakage
    of the target rather than a feature.
    """
    if log.empty:
        return log
    eligible = log[log["snapshot_date"] <= as_of]
    if eligible.empty:
        return eligible
    latest = eligible["snapshot_date"].max()
    return eligible[eligible["snapshot_date"] == latest].copy()


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    output_dir: str | Path = "data/raw/injuries",
    snapshot_only: bool = False,
    url: str = ESPN_API_URL,
    daily: bool = False,
    min_hours: float = MIN_HOURS_BETWEEN_SNAPSHOTS,
) -> list[dict]:
    output_dir = Path(output_dir)
    scraped_at = datetime.now(timezone.utc).astimezone()  # local time with tz info

    if daily and is_captured(output_dir, min_hours=min_hours):
        last = last_snapshot_at(output_dir)
        print(f"Already captured within {min_hours:g}h (last snapshot {last}); skipping.")
        return []

    print("Fetching injury data from ESPN API ...")
    records = fetch_injuries(url)
    if not records:
        print("  WARNING: no injury records returned.")
        return records

    save_snapshot(records, output_dir, scraped_at)

    if not snapshot_only:
        rebuild_log(output_dir)

    days = snapshot_dates(output_dir)
    span = f"{days[0]} → {days[-1]}" if days else "none"
    print(f"Archive now holds {len(days):,} snapshot days ({span})")
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ESPN NBA injury reports.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for snapshots and cumulative log "
             "(default: data.injuries.dir from configs/default.yaml)",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Save a snapshot but skip appending to the cumulative log.",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Report snapshot coverage and permanently-missing days. No requests.")
    parser.add_argument(
        "--daily",
        action="store_true",
        help="No-op if a snapshot was already taken inside the daily window. Makes the "
             "scheduled capture safe to re-run.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    inj_cfg = cfg.get("data", {}).get("injuries", {})
    if args.status:
        print_status(args.output_dir or inj_cfg.get("dir", "data/raw/injuries"))
        raise SystemExit(0)
    run(output_dir=args.output_dir or inj_cfg.get("dir", "data/raw/injuries"),
        snapshot_only=args.snapshot_only,
        daily=args.daily,
        min_hours=inj_cfg.get("min_hours_between_snapshots",
                              MIN_HOURS_BETWEEN_SNAPSHOTS))
