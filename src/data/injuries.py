"""Scrape NBA injury reports from the ESPN public API and persist to a timestamped log.

ESPN's /nba/injuries page is JS-rendered, but its underlying JSON API is stable and public.
Each run writes a timestamped snapshot CSV and appends to a cumulative log CSV.

Usage:
    python -m src.data.injuries                  # scrape once, append to log
    python -m src.data.injuries --snapshot-only  # save snapshot, skip log append
    python -m src.data.injuries --output-dir path/to/dir
"""

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ESPN_API_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

_FIELDNAMES = [
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
    "return_date",
    "fantasy_status",
    "short_comment",
    "long_comment",
]


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
                "return_date":      details.get("returnDate", ""),
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
    df.to_csv(dest, index=False)
    print(f"  Snapshot saved ({len(records)} records) → {dest}")
    return dest


def append_to_log(records: list[dict], log_path: Path, scraped_at: datetime) -> None:
    """Append records to the cumulative injury log, stamped with scraped_at."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    ts = scraped_at.isoformat()

    with log_path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        for rec in records:
            writer.writerow({"scraped_at": ts, **rec})

    print(f"  Appended {len(records)} records → {log_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(
    output_dir: str | Path = "data/raw/injuries",
    snapshot_only: bool = False,
    url: str = ESPN_API_URL,
) -> list[dict]:
    output_dir = Path(output_dir)
    scraped_at = datetime.now(timezone.utc).astimezone()  # local time with tz info

    print(f"Fetching injury data from ESPN API ...")
    records = fetch_injuries(url)
    if not records:
        print("  WARNING: no injury records returned.")
        return records

    save_snapshot(records, output_dir, scraped_at)

    if not snapshot_only:
        append_to_log(records, output_dir / "injuries_log.csv", scraped_at)

    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ESPN NBA injury reports.")
    parser.add_argument(
        "--output-dir",
        default="data/raw/injuries",
        help="Directory for snapshots and cumulative log (default: data/raw/injuries)",
    )
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Save a snapshot but skip appending to the cumulative log.",
    )
    args = parser.parse_args()
    run(output_dir=args.output_dir, snapshot_only=args.snapshot_only)
