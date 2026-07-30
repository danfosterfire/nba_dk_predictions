"""FantasyPros NBA consensus ADP — Wayback backfill and live capture.

This is the project's **market-proxy** series: a Yahoo/ESPN/CBS consensus ADP going back
to 2014-15. It is not DraftKings' board — see `src/data/adp_draftkings.py` for that, and
`docs/adp-plan.md` for the measured gap between them (Spearman 0.870, 21.9-pick mean rank
gap, centers drafted 11.9 picks earlier on DK).

## The freeze rule — why season labels are not calendar dates

The page does **not** update continuously. It holds the completed draft season's ADP for
~11 months and flips only when the next season's drafts begin. Measured on consecutive
2025 snapshots, share of players whose `AVG` is byte-identical to the previous snapshot:

    2025-01-19 → 2025-07-10    33.8%     same board, deep tail still drifting
    2025-07-10 → 2025-07-20   100.0%     frozen
    2025-07-20 → 2025-09-06   100.0%     frozen
    2025-09-06 → 2025-10-05     0.9%     FLIPPED to 2025-26

Two consequences, both of which have already produced a wrong answer once:

- **A snapshot's date is not its season.** The 2025-09-06 snapshot looks like a 2025-26
  preseason capture and is in fact 2024-25, unchanged since the previous October.
- **A month cutoff fails outright on 2020-21**, which tipped off 2020-12-22: the
  2020-10-23 snapshot is identical to 2020-09-16, i.e. still 2019-20.

`assign_seasons` handles both with a **calendar-primary, inherit-when-frozen** rule.

> A tempting design that does **not** work, recorded so it is not retried: detecting the
> season *flip* from the identical-AVG share and starting a new run there. Flips and
> ordinary within-draft-window drift are not separable that way. Measured on the archive,
> real flips score 0.0055-0.0871 while October drift on an unchanged board scores
> 0.0667-0.2273 — the ranges overlap, so no threshold splits them. What identity *does*
> decide cleanly is "did this board move at all" (frozen pairs score exactly 1.0000
> against a non-frozen maximum of 0.3360), and that is the one question it is asked.

## Point-in-time discipline

The ADP column is frozen at draft time; the **team and status columns are live**. The
same table that serves 2025-26 ADP in July 2026 lists Giannis on MIA and carries `DTD`
tokens describing that morning. Three columns, three as-of dates. So:

- `adp_as_of` (flip-detected season start) is what the ADP is valid for.
- `captured_at` is when the page was read.
- `status_at_capture` and `team` describe `captured_at`, **never** the draft date. Never
  join `status_at_capture` to a preseason injury feature — that is the leak
  `docs/availability-plan.md` forbids, wearing a disguise.

## Scraping etiquette

`fantasypros.com/robots.txt` permits `/nba/adp/` and sets `Crawl-delay: 5`; `/api/`,
`/json/` and `/nba/ranker/` are disallowed, so the rendered table is parsed rather than
the JSON behind it. The Wayback CDX endpoint needs a browser UA and rate-limits at
roughly one request per 10 s.

Usage:
    python -m src.data.adp_fantasypros                  # live capture, one snapshot
    python -m src.data.adp_fantasypros --backfill       # Wayback sweep, all seasons
    python -m src.data.adp_fantasypros --reparse        # re-parse the archive, offline
    python -m src.data.adp_fantasypros --status         # coverage by season, no requests
"""

import argparse
import gzip
import io
import json
import re
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml
from bs4 import BeautifulSoup

LIVE_URL = "https://www.fantasypros.com/nba/adp/overall.php"
CDX_URL = ("http://web.archive.org/cdx/search/cdx"
           "?url=fantasypros.com/nba/adp/overall.php&output=json&filter=statuscode:200")
# The `id_` modifier returns the originally-archived bytes rather than a Wayback-rewritten
# page — which also means the response is whatever encoding the origin sent. See _decode.
WAYBACK_RAW = "https://web.archive.org/web/{stamp}id_/{url}"

# robots.txt says 5; the Wayback host 503s well before that so it gets its own, longer one.
CRAWL_DELAY = 5.0
WAYBACK_DELAY = 6.0
MAX_RETRIES = 5
TIMEOUT = 60

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

MANIFEST_NAME = "_fantasypros_manifest.csv"
_SNAPSHOT_RE = re.compile(r"fp_adp_(\d{4}-\d{2}-\d{2})_(\d{6})\.html\.gz$")

# Per-snapshot, never assumed: CBS is present in 2014/15/17/19/23 and absent in
# 2020/22/25, and Fantrax appears on other aggregators. Whatever is on the page wins.
KNOWN_SOURCES = ("Yahoo", "ESPN", "CBS", "Fantrax", "Sleeper", "NFBC")

# At or above this share of identical AVG values the board is FROZEN — unchanged since the
# previous snapshot. Measured, on the seeded archive: frozen pairs score exactly 1.0000,
# and every non-frozen pair scores <= 0.3360, so 0.5 sits in a wide empty gap.
#
# What this threshold deliberately does NOT try to do is detect a season *flip*. That is
# not separable from within-draft-window drift by identity alone, and the measurements say
# so: real flips score 0.0055-0.0871 while October drift on the same board scores
# 0.0667-0.2273. The ranges overlap. So identity is used only for the one thing it decides
# cleanly — "did this board change at all" — and the season label comes from the calendar,
# with frozen snapshots inheriting. See `assign_seasons`.
FROZEN_THRESHOLD = 0.5
MIN_OVERLAP = 40

_OUTPUT_COLS = ["captured_at", "capture_date", "season", "adp_rank", "player_name",
                "team", "positions", "status_at_capture", "adp", "source_detail",
                "snapshot_source"]


# ── Player cell ───────────────────────────────────────────────────────────────
# Three formats across 12 years, plus an optional trailing live-status token:
#   2014     "Kevin Durant ( OKC )"           (position in its own column)
#   2015     "LeBron James CLE"
#   2017+    "Russell Westbrook (OKC - PG)"   optionally "... (LAL - PF) OUT"

# The trailing status token is NOT a fixed vocabulary and enumerating it does not work:
# it is `DTD`/`OUT` in 2022, a bare `O` in 2015 (`Kevin Durant OKC O`), and `G-League` —
# hyphenated and mixed-case — in 2025. Anything left after the team is treated as status.
# Inside parentheses that is unambiguous. In the bare 2015 form it stays restricted to
# short uppercase runs, since a loose pattern there would start eating surnames.
_CELL_MODERN = re.compile(
    r"^(?P<name>.+?)\s*\(\s*(?P<team>[A-Z]{2,3})\s*-\s*(?P<pos>[^)]*?)\s*\)"
    r"\s*(?P<status>[A-Za-z][A-Za-z-]{0,15})?\s*$")
_CELL_PARENS = re.compile(
    r"^(?P<name>.+?)\s*\(\s*(?P<team>[A-Z]{2,3})\s*\)"
    r"\s*(?P<status>[A-Za-z][A-Za-z-]{0,15})?\s*$")
_CELL_BARE = re.compile(
    r"^(?P<name>.+?)\s+(?P<team>[A-Z]{2,3})(?:\s+(?P<status>[A-Z]{1,4}))?\s*$")


def parse_player_cell(text: str) -> dict:
    """Split a player cell into name / team / positions / live status.

    Returns empty strings for anything the cell does not carry rather than raising —
    a 2014 cell has no positions, and most cells have no status.
    """
    text = re.sub(r"\s+", " ", (text or "").strip())
    for pattern in (_CELL_MODERN, _CELL_PARENS, _CELL_BARE):
        m = pattern.match(text)
        if m:
            g = m.groupdict()
            return {
                "player_name": g["name"].strip(),
                "team": g["team"],
                "positions": (g.get("pos") or "").replace(" ", ""),
                "status_at_capture": g.get("status") or "",
            }
    return {"player_name": text, "team": "", "positions": "", "status_at_capture": ""}


# ── Fetch ─────────────────────────────────────────────────────────────────────

# Wayback throttles with 503, 429 and **498** — the last is not a standard code and is
# what the CDX endpoint returns once it decides you have been sweeping too hard. All three
# are transient and clear on their own, so they are backed off exponentially rather than
# treated as failures. Nothing here is urgent: the archive is not going anywhere, and the
# module works entirely offline via `--reparse`.
_THROTTLE_CODES = {429, 498, 503}


def _get(url: str, delay: float, timeout: int = TIMEOUT,
         max_retries: int = MAX_RETRIES) -> bytes:
    """GET with exponential backoff on throttling."""
    last = None
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers=_HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - network faults are all retryable here
            last = exc
            code = getattr(exc, "code", None)
            wait = delay * (2 ** attempt) if code in _THROTTLE_CODES else delay
            if attempt < max_retries - 1:
                time.sleep(wait)
    code = getattr(last, "code", None)
    hint = (" — Wayback is throttling; wait a few minutes and re-run, or work offline "
            "with --reparse" if code in _THROTTLE_CODES else "")
    raise RuntimeError(f"GET failed after {max_retries} attempts: {url} ({last}){hint}")


def snapshot_index(cdx_url: str = CDX_URL) -> list[str]:
    """Wayback timestamps for the ADP page with a 200 status, ascending."""
    raw = _get(cdx_url, delay=10.0)
    rows = json.loads(raw.decode("utf-8", "replace"))[1:]
    return sorted(r[1] for r in rows)


def fetch_snapshot(stamp: str, url: str = LIVE_URL) -> bytes:
    return _get(WAYBACK_RAW.format(stamp=stamp, url=url), delay=WAYBACK_DELAY)


def fetch_live(url: str = LIVE_URL) -> bytes:
    return _get(url, delay=CRAWL_DELAY)


def _decode(raw: bytes) -> str:
    """Inflate if gzipped, then decode.

    Wayback's `id_` endpoint hands back the bytes the origin originally sent, so
    snapshots from 2022 on arrive gzip-compressed. Decoding those as UTF-8 yields a page
    with no `<table>` in it, which looks exactly like a JS-rendered page and is not one.
    Sniff the magic number instead of trusting headers.
    """
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", "replace")


# ── Parse ─────────────────────────────────────────────────────────────────────

def parse_board(raw: bytes | str) -> pd.DataFrame:
    """The ADP table from one page, one row per player.

    `table#data` is stable across all 12 archived years — it is the only structural
    element that is. Returns an empty frame for the "Sorry, this report is not available
    at the moment" placeholder FantasyPros serves on some early-September dates, which is
    a page state rather than a parse failure.
    """
    html = _decode(raw) if isinstance(raw, bytes) else raw
    table = BeautifulSoup(html, "html.parser").find("table", id="data")
    if table is None:
        return pd.DataFrame(columns=["adp_rank", "player_name", "team", "positions",
                                     "status_at_capture", "AVG"])

    header = [th.get_text(strip=True) for th in table.find_all("th")]
    body = table.find("tbody") or table
    records = []
    for tr in body.find_all("tr"):
        # recursive=False: html.parser does not auto-close <tr>, so a nested search
        # collects every cell in the table into the first row.
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td", recursive=False)]
        if len(cells) != len(header):
            continue
        row = dict(zip(header, cells))
        player = row.get("Player") or row.get("Player (Team)") or ""
        rec = parse_player_cell(player)
        if not rec["player_name"]:
            continue
        rec["adp_rank"] = pd.to_numeric(row.get("Rank"), errors="coerce")
        # POS is a separate column in 2014-15 only; prefer the in-cell positions.
        if not rec["positions"]:
            rec["positions"] = row.get("POS", "") or ""
        for src in KNOWN_SOURCES:
            if src in row:
                rec[src] = pd.to_numeric(row[src], errors="coerce")
        rec["AVG"] = pd.to_numeric(row.get("AVG"), errors="coerce")
        records.append(rec)

    df = pd.DataFrame(records)
    return df[df["AVG"].notna()].reset_index(drop=True) if len(df) else df


def sources_present(board: pd.DataFrame) -> list[str]:
    return [s for s in KNOWN_SOURCES if s in board.columns]


# ── The freeze rule ───────────────────────────────────────────────────────────

def identical_share(prev: pd.DataFrame, curr: pd.DataFrame) -> tuple[float, int]:
    """Share of players whose AVG is unchanged between two boards, and the overlap size.

    The discriminator behind `assign_seasons`. A frozen board scores ~1.0 and a flipped
    one ~0.0; the deep tail of a live board drifts, which is the 33.8% case.
    """
    if prev.empty or curr.empty:
        return 0.0, 0
    a = prev.set_index("player_name")["AVG"]
    b = curr.set_index("player_name")["AVG"]
    a = a[~a.index.duplicated()]
    b = b[~b.index.duplicated()]
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return 0.0, 0
    return float((a[common] == b[common]).mean()), len(common)


def calendar_season(day: datetime | str) -> str:
    """The season whose drafts most recently ran, by calendar alone.

    Drafts run in October, so a snapshot from October of year Y onwards carries season
    Y/Y+1, and anything from January to September carries the previous one. Correct for
    every observed season except where drafts did not run in October — see
    `assign_seasons`, which is what handles that.
    """
    day = pd.Timestamp(day)
    start = day.year if day.month >= 10 else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def assign_seasons(boards: list[tuple[str, str, pd.DataFrame]],
                   frozen_threshold: float = FROZEN_THRESHOLD,
                   min_overlap: int = MIN_OVERLAP) -> list[dict]:
    """Label each snapshot with the season its ADP is valid for.

    `boards` is [(key, capture_date, board)] ascending, where `key` is unique per
    snapshot — **not** the date, since a date can hold several captures.

    The rule is **calendar-primary, inherit-when-frozen**: take `calendar_season`, unless
    the board is byte-identical to its predecessor, in which case inherit that
    predecessor's season. Inheritance only ever overrides the calendar for a board that
    has not moved, which is exactly the failure the calendar has:

    > 2020-21 tipped off 2020-12-22, so its drafts ran in December. The 2020-10-23
    > snapshot is identical to 2020-09-16 — still the 2019-20 board — and the calendar
    > rule alone would call it 2020-21. Inheritance carries 2019-20 through instead.
    > `tests/test_adp.py::test_frozen_board_inherits_across_the_covid_boundary` pins it.

    Every label ships with the `identical_share` and `overlap` that produced it, so a
    label is auditable rather than asserted.

    **Residual limitation, stated because it is not detectable from identity alone**: if a
    season's drafts open *before* October, the calendar rule mislabels the first snapshots
    of that run. DK opened its 2026-27 board in July 2026, so this is not hypothetical for
    every source — it simply has not yet happened on FantasyPros in 12 archived years.
    The manifest exposes enough to spot it by hand, and `--season-override` forces it.
    """
    out: list[dict] = []
    prev_season: str | None = None
    for i, (key, day, board) in enumerate(boards):
        share, overlap = (0.0, 0) if i == 0 else identical_share(boards[i - 1][2], board)
        frozen = overlap >= min_overlap and share >= frozen_threshold
        season = prev_season if (frozen and prev_season) else calendar_season(day)
        out.append({
            "key": key,
            "capture_date": day,
            "identical_share": round(share, 4),
            "overlap": overlap,
            "frozen": frozen,
            "inherited": bool(frozen and prev_season
                              and season != calendar_season(day)),
            "season": season,
        })
        prev_season = season
    return out


# ── Archive ───────────────────────────────────────────────────────────────────

def snapshot_path(out_dir: Path, captured_at: datetime) -> Path:
    return out_dir / f"fp_adp_{captured_at:%Y-%m-%d_%H%M%S}.html.gz"


def archive_snapshot(raw: bytes, captured_at: datetime, out_dir: str | Path) -> Path:
    """Store the raw page, gzipped. The HTML is the artifact; parsers get rewritten."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = snapshot_path(out_dir, captured_at)
    payload = raw if raw[:2] == b"\x1f\x8b" else gzip.compress(raw)
    dest.write_bytes(payload)
    return dest


def archived_snapshots(out_dir: str | Path) -> list[tuple[datetime, Path]]:
    """(captured_at, path) for every archived snapshot, ascending."""
    found = []
    for f in Path(out_dir).glob("fp_adp_*.html.gz"):
        m = _SNAPSHOT_RE.search(f.name)
        if m:
            found.append((datetime.strptime(f"{m.group(1)}{m.group(2)}",
                                            "%Y-%m-%d%H%M%S"), f))
    return sorted(found)


def _stamp_to_dt(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%d%H%M%S")


# ── Build ─────────────────────────────────────────────────────────────────────

def build_panel(out_dir: str | Path, frozen_threshold: float = FROZEN_THRESHOLD,
                min_overlap: int = MIN_OVERLAP) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse the whole archive into a long panel, plus the per-snapshot manifest.

    Long format: one row per (snapshot, player, source_detail), where `source_detail` is
    `avg` or an individual site. `AVG` is taken verbatim and never recomputed — it does
    not always equal the mean of the displayed columns (measured: exact in 12 of 19
    snapshots), so the displayed sources are not a complete accounting of it.
    """
    snaps = archived_snapshots(out_dir)
    boards = [(dt.isoformat(timespec="seconds"), dt.date().isoformat(),
               parse_board(path.read_bytes())) for dt, path in snaps]
    # Keyed on the full capture timestamp, never the date: a single day can hold several
    # captures, and keying on the date silently collapses them onto one label.
    usable = [b for b in boards if not b[2].empty]
    labels = {lab["key"]: lab
              for lab in assign_seasons(usable, frozen_threshold, min_overlap)}

    rows, manifest = [], []
    for (dt, path), (key, day, board) in zip(snaps, boards):
        srcs = sources_present(board)
        lab = labels.get(key, {})
        manifest.append({
            "captured_at": key,
            "capture_date": day,
            "path": path.name,
            "n_rows": len(board),
            "sources": "|".join(srcs),
            "season": lab.get("season", ""),
            "identical_share": lab.get("identical_share", ""),
            "overlap": lab.get("overlap", ""),
            "frozen": lab.get("frozen", ""),
            "inherited": lab.get("inherited", ""),
            "parsed": bool(len(board)),
        })
        if board.empty:
            continue
        for detail in ["avg"] + [s.lower() for s in srcs]:
            col = "AVG" if detail == "avg" else next(s for s in srcs if s.lower() == detail)
            sub = board[board[col].notna()]
            for r in sub.itertuples(index=False):
                rows.append({
                    "captured_at": dt.isoformat(timespec="seconds"),
                    "capture_date": day,
                    "season": lab.get("season", ""),
                    "adp_rank": getattr(r, "adp_rank"),
                    "player_name": r.player_name,
                    "team": r.team,
                    "positions": r.positions,
                    "status_at_capture": r.status_at_capture,
                    "adp": getattr(r, col),
                    "source_detail": detail,
                    "snapshot_source": "fantasypros",
                })
    panel = pd.DataFrame(rows, columns=_OUTPUT_COLS)
    return panel, pd.DataFrame(manifest)


def print_status(out_dir: str | Path) -> None:
    panel, manifest = build_panel(out_dir)
    print("FantasyPros consensus ADP — frozen between draft seasons, backfillable via Wayback")
    if manifest.empty:
        print("  No snapshots archived yet.")
        return
    ok = manifest[manifest["parsed"]]
    print(f"  snapshots archived     {len(manifest):>4}  "
          f"({manifest.capture_date.min()} → {manifest.capture_date.max()})")
    print(f"  parsed                 {len(ok):>4}  "
          f"({len(manifest) - len(ok)} placeholder/empty pages)")
    if ok.empty:
        return
    print(f"  seasons covered        {ok.season.nunique():>4}")
    print()
    print("  season    snaps  earliest     sources")
    for season, g in ok.groupby("season"):
        srcs = sorted({s for row in g.sources for s in row.split("|") if s})
        print(f"  {season:9s} {len(g):5d}  {g.capture_date.min()}   {','.join(srcs)}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(out_dir: str | Path = "data/raw/adp/fantasypros",
        features_dir: str | Path = "data/features",
        backfill: bool = False,
        reparse: bool = False,
        limit: int | None = None,
        frozen_threshold: float = FROZEN_THRESHOLD,
        min_overlap: int = MIN_OVERLAP) -> pd.DataFrame:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if backfill:
        have = {dt.strftime("%Y%m%d%H%M%S") for dt, _ in archived_snapshots(out_dir)}
        stamps = [s for s in snapshot_index() if s not in have]
        if limit:
            stamps = stamps[:limit]
        print(f"Wayback backfill: {len(stamps):,} snapshots to fetch "
              f"({len(have):,} already archived)")
        for i, stamp in enumerate(stamps, 1):
            try:
                raw = fetch_snapshot(stamp)
            except RuntimeError as exc:
                print(f"  [{i}/{len(stamps)}] {stamp} FAILED: {exc}")
                continue
            dest = archive_snapshot(raw, _stamp_to_dt(stamp), out_dir)
            print(f"  [{i}/{len(stamps)}] {stamp} → {dest.name} ({len(raw):,} bytes)")
            time.sleep(WAYBACK_DELAY)
    elif not reparse:
        captured_at = datetime.now()
        raw = fetch_live()
        dest = archive_snapshot(raw, captured_at, out_dir)
        print(f"  Live snapshot ({len(raw):,} bytes) → {dest}")

    panel, manifest = build_panel(out_dir, frozen_threshold, min_overlap)

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    manifest_dest = out_dir / MANIFEST_NAME
    manifest.to_csv(manifest_dest, index=False)
    panel_dest = features_dir / "adp_fantasypros.parquet"
    panel.to_parquet(panel_dest, index=False)

    seasons = sorted(s for s in panel.season.unique() if s)
    print(f"  Parsed {len(manifest):,} snapshots → {manifest_dest}")
    print(f"  Wrote {len(panel):,} ADP rows over {len(seasons)} seasons "
          f"({seasons[0] if seasons else '-'} → {seasons[-1] if seasons else '-'}) "
          f"→ {panel_dest}")
    return panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FantasyPros NBA consensus ADP: Wayback backfill and live capture.")
    parser.add_argument("--output-dir", default=None,
                        help="Archive directory (default: data.adp.fantasypros.dir).")
    parser.add_argument("--backfill", action="store_true",
                        help="Sweep the Wayback Machine for every archived snapshot. "
                             "Skips anything already on disk.")
    parser.add_argument("--reparse", action="store_true",
                        help="Rebuild the panel from the archive. Offline, no requests.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of backfill fetches (for a trial run).")
    parser.add_argument("--status", action="store_true",
                        help="Report coverage by season. No requests.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    adp_cfg = cfg.get("data", {}).get("adp", {})
    fp_cfg = adp_cfg.get("fantasypros", {})
    out = args.output_dir or fp_cfg.get("dir", "data/raw/adp/fantasypros")

    if args.status:
        print_status(out)
        raise SystemExit(0)

    run(out_dir=out,
        features_dir=cfg.get("data", {}).get("features_dir", "data/features"),
        backfill=args.backfill,
        reparse=args.reparse,
        limit=args.limit,
        frozen_threshold=fp_cfg.get("frozen_threshold", FROZEN_THRESHOLD),
        min_overlap=fp_cfg.get("min_overlap", MIN_OVERLAP))
