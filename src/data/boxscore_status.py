"""Backfill per-game inactive lists and DNP reasons — the one measurement the
availability panel cannot make from game logs.

Game logs contain **only games actually played**, so an absence is not a row you can read
but the gap between a team's schedule and a player's appearances. That gap cannot
distinguish "injured" from "healthy scratch" from "not on the roster", and that single
distinction is what the availability head depends on
(`docs/availability-plan.md`, stage B).

Two endpoints per game close it:

- `BoxScoreSummaryV3` → per-team `inactives`: on the roster, not available to play.
- `BoxScoreTraditionalV2` → `COMMENT` per player: `DNP - Coach's Decision`,
  `DND - Injury/Illness`, `NWT - League Suspension`, and in older seasons free text with
  the body part inline (`DND- Right Quad Strain`).

Together they give a three-way status for every player on every game's roster —
**played / dressed-but-scratched / inactive** — with a stated reason.

## Coverage boundary: 2006-07

The inactive list returns 0 rows for 2005-06 and earlier and is populated from 2006-07
on — probed directly, both sides of the boundary. This is a third coverage line for the
project, alongside Tier A (30 seasons) and Tier B (13). Seasons before it can carry
availability but not its decomposition. The traditional `COMMENT` field does go back
further, so pre-2006-07 seasons still separate played from dressed-but-scratched; they
just cannot see the inactive list.

## Endpoint routing

Both endpoints have a version trap, and they fail in opposite ways:

- `BoxScoreTraditionalV2` returns **0 rows** from 2025-26 on (deprecated, no longer
  published), so that season routes to V3, which exposes the same field as lower-case
  `comment`. This one is loud — an empty frame is obvious.
- `BoxScoreSummaryV2` is the **quiet** one, and it is the reason `_inactive_rows` departs
  from the plan. It keeps returning 200 with a populated `GameSummary` after 2025-04-10
  while silently dropping the `InactivePlayers` set entirely, so a backfill built on it
  records "nobody was inactive" instead of erroring. `BoxScoreSummaryV3` is used for every
  season instead; it agrees with V2 exactly where V2 works and covers the full range.

Per-game inactive counts are recorded in the manifest so a regression of this kind shows
up as a column of zeros rather than as a finding about rosters.

The full backfill is ~24,600 games and 8-14 h at the project's 0.6 s delay, so it is
built to be killed and resumed: work is flushed in chunks and the resume key is the set
of game ids already in the season CSV — the data itself, not a side file that can drift
out of step with it.

Usage:
    python -m src.data.boxscore_status                     # 2006-07 → present
    python -m src.data.boxscore_status --seasons 2023-24 2024-25
    python -m src.data.boxscore_status --limit 20          # smoke test
"""

import argparse
import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug

DELAY = 0.6
FLUSH_EVERY = 25

# InactivePlayers is empty before this season — probed, not assumed.
FIRST_SEASON = "2006-07"
# BoxScoreTraditionalV2 stops returning rows here; V3 carries the same field.
V3_FROM_SEASON = "2025-26"

MANIFEST_NAME = "_boxscore_status_manifest.csv"

STATUS_COLS = ["season", "game_id", "team_id", "player_id", "player_name",
               "status", "comment", "reason", "start_position", "min"]

# Comment vocabulary → reason. Ordered: the first match wins, so the specific categories
# are checked before the generic injury one. The older seasons write free text with the
# body part inline and no space after DND, which is why these match loosely.
_REASON_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("suspension",   re.compile(r"suspen", re.I)),
    ("not_rostered", re.compile(r"not\s*with\s*team|trade|inactive\s*list|"
                                r"not\s*on\s*roster", re.I)),
    ("gleague",      re.compile(r"g\s*league|assignment|two-?way", re.I)),
    ("personal",     re.compile(r"personal|bereavement|family", re.I)),
    ("coach",        re.compile(r"coach", re.I)),
    ("rest",         re.compile(r"\brest\b|load\s*management", re.I)),
    ("injury",       re.compile(r"injur|illness|\bdnd\b|\bsore\b|strain|sprain|"
                                r"health|concussion|surgery", re.I)),
]


def classify_comment(comment: str) -> str:
    """A box-score `COMMENT` → one of the reason categories.

    `InactivePlayers` carries no reason at all, so an inactive player returns `""` and is
    left unsplit rather than guessed at. That is the honest answer: the endpoint conflates
    injury, personal reasons, G-League assignment and roster mechanics, and inventing a
    split here would put a fabricated label into the very decomposition the backfill
    exists to test.
    """
    text = (comment or "").strip()
    if not text:
        return ""
    for reason, pattern in _REASON_PATTERNS:
        if pattern.search(text):
            return reason
    return "other"


# ── Game lists ────────────────────────────────────────────────────────────────

def pad_game_id(game_id) -> str:
    """Game ids are 10-char zero-padded strings; the CSVs store them as ints."""
    text = str(game_id).strip()
    return text.zfill(10) if text.isdigit() else text


def season_game_ids(season: str, raw_dir: str | Path) -> list[str]:
    """Every game id for a season, regular season plus playoffs if fetched."""
    raw_dir = Path(raw_dir)
    ids: set[str] = set()
    for name in (f"game_logs_{_slug(season)}.csv",
                 f"game_logs_playoffs_{_slug(season)}.csv"):
        path = raw_dir / name
        if not path.exists():
            continue
        col = pd.read_csv(path, usecols=["GAME_ID"], low_memory=False)["GAME_ID"]
        ids.update(pad_game_id(g) for g in col.dropna().unique())
    return sorted(ids)


def uses_v3(season: str, v3_from: str = V3_FROM_SEASON) -> bool:
    return _season_start_year(season) >= _season_start_year(v3_from)


# ── One game ──────────────────────────────────────────────────────────────────

def _traditional_rows(game_id: str, use_v3: bool) -> pd.DataFrame:
    """Players who dressed: `played` if they logged minutes, `dnp` if they did not."""
    if use_v3:
        from nba_api.stats.endpoints import boxscoretraditionalv3
        frames = boxscoretraditionalv3.BoxScoreTraditionalV3(
            game_id=game_id).get_data_frames()
        df = frames[0] if frames else pd.DataFrame()
        if df.empty:
            return pd.DataFrame(columns=STATUS_COLS)
        name = (df["firstName"].fillna("") + " " + df["familyName"].fillna("")).str.strip()
        out = pd.DataFrame({
            "team_id": df["teamId"], "player_id": df["personId"], "player_name": name,
            "comment": df.get("comment", "").fillna(""),
            "start_position": df.get("position", "").fillna(""),
            "min": df.get("minutes", "").fillna(""),
        })
    else:
        from nba_api.stats.endpoints import boxscoretraditionalv2
        data = boxscoretraditionalv2.BoxScoreTraditionalV2(
            game_id=game_id).get_normalized_dict()
        df = pd.DataFrame(data.get("PlayerStats", []))
        if df.empty:
            return pd.DataFrame(columns=STATUS_COLS)
        out = pd.DataFrame({
            "team_id": df["TEAM_ID"], "player_id": df["PLAYER_ID"],
            "player_name": df["PLAYER_NAME"].fillna(""),
            "comment": df.get("COMMENT", "").fillna(""),
            "start_position": df.get("START_POSITION", "").fillna(""),
            "min": df.get("MIN", "").fillna(""),
        })

    out["min"] = out["min"].astype(str).replace({"nan": "", "None": ""})
    out["status"] = out["min"].str.strip().ne("").map({True: "played", False: "dnp"})
    return out


def _inactive_rows(game_id: str) -> pd.DataFrame:
    """Players on the roster who were not available. No reason is published.

    Read from `BoxScoreSummaryV3`, **not** the `InactivePlayers` result set of V2 that
    `docs/availability-plan.md` specifies. V2 is correct up to 2025-04-10 and then goes
    silently empty — it still returns 200 with a populated `GameSummary` and simply drops
    the `InactivePlayers` set, so a backfill built on it records "nobody was inactive" for
    every recent game rather than failing. Measured on 2025-26: 223 of 228 games came back
    with zero inactives under V2. V3 covers the whole 2006-07 → 2025-26 range and agrees
    with V2 exactly where V2 works (6 on 2006-07 opening night, 8 on 2023-24's).

    V3 nests the list per team, and the team id lives on the parent block rather than on
    the player, which is why this is hand-unpacked instead of read off a result set.
    """
    from nba_api.stats.endpoints import boxscoresummaryv3
    payload = boxscoresummaryv3.BoxScoreSummaryV3(game_id=game_id).get_dict() or {}
    # `boxScoreSummary` is occasionally present but null rather than absent, so `.get`
    # with a default is not enough — measured on 3 games in 2025-26. They land in the
    # manifest with an error, write no rows, and are retried on the next run.
    summary = payload.get("boxScoreSummary") or {}
    rows = []
    for side in ("homeTeam", "awayTeam"):
        block = summary.get(side) or {}
        for player in block.get("inactives") or []:
            name = f"{player.get('firstName', '')} {player.get('familyName', '')}"
            rows.append({"team_id": block.get("teamId"),
                         "player_id": player.get("personId"),
                         "player_name": name.strip(), "comment": "",
                         "start_position": "", "min": "", "status": "inactive"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=STATUS_COLS)


def fetch_game_status(game_id: str, season: str, use_v3: bool | None = None,
                      delay: float = DELAY) -> tuple[pd.DataFrame, dict]:
    """Both endpoints for one game, as one frame plus a manifest row."""
    use_v3 = uses_v3(season) if use_v3 is None else use_v3
    inactive = _inactive_rows(game_id)
    time.sleep(delay)
    dressed = _traditional_rows(game_id, use_v3)

    frames = [f for f in (dressed, inactive) if not f.empty]
    rows = (pd.concat(frames, ignore_index=True) if frames
            else pd.DataFrame(columns=STATUS_COLS))
    if not rows.empty:
        rows.insert(0, "game_id", game_id)
        rows.insert(0, "season", season)
        rows["reason"] = rows["comment"].map(classify_comment)
        rows = rows[STATUS_COLS]

    manifest = {
        "season": season, "game_id": game_id,
        "dressed_rows": len(dressed), "inactive_rows": len(inactive),
        "played_rows": int((dressed["status"] == "played").sum()) if len(dressed) else 0,
        "endpoint": "traditional_v3" if use_v3 else "traditional_v2",
        "fetched_at": datetime.now().astimezone().isoformat(), "error": "",
    }
    return rows, manifest


# ── Artifacts ─────────────────────────────────────────────────────────────────

def status_path(season: str, raw_dir: str | Path) -> Path:
    return Path(raw_dir) / f"boxscore_status_{_slug(season)}.csv"


def done_game_ids(season: str, raw_dir: str | Path) -> set[str]:
    """Game ids already on disk — the resume key.

    Read from the data rather than the manifest on purpose: the manifest is written after
    the rows, so a run killed between the two writes would otherwise skip games whose rows
    never landed.
    """
    path = status_path(season, raw_dir)
    if not path.exists():
        return set()
    try:
        col = pd.read_csv(path, usecols=["game_id"], low_memory=False)["game_id"]
    except (ValueError, pd.errors.ParserError):
        return set()
    return {pad_game_id(g) for g in col.dropna().unique()}


def _append(rows: pd.DataFrame, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(dest, mode="a", header=not dest.exists(), index=False)


def _append_manifest(rows: list[dict], raw_dir: Path) -> None:
    if not rows:
        return
    dest = raw_dir / MANIFEST_NAME
    dest.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(dest, mode="a", header=not dest.exists(), index=False)


def load_status(season: str, raw_dir: str | Path) -> pd.DataFrame:
    """One season's statuses, de-duplicated.

    A hard kill can leave a chunk appended twice; the natural key is
    (game_id, team_id, player_id), and a player has exactly one status per game.
    """
    path = status_path(season, raw_dir)
    if not path.exists():
        return pd.DataFrame(columns=STATUS_COLS)
    # `on_bad_lines` because the backfill appends for hours: reading a season while it is
    # still being written can catch a half-flushed final line. Skipping it costs one
    # game that the next run picks up anyway, where raising would make every downstream
    # build fail for as long as the backfill is running.
    df = pd.read_csv(path, low_memory=False, on_bad_lines="skip")
    df["game_id"] = df["game_id"].map(pad_game_id)
    return df.drop_duplicates(subset=["game_id", "team_id", "player_id"], keep="last")


# ── Orchestration ─────────────────────────────────────────────────────────────

def backfill_season(season: str, raw_dir: str | Path, delay: float = DELAY,
                    flush_every: int = FLUSH_EVERY, limit: int | None = None) -> int:
    raw_dir = Path(raw_dir)
    game_ids = season_game_ids(season, raw_dir)
    if not game_ids:
        print(f"{season}: no game log on disk, skipping")
        return 0

    done = done_game_ids(season, raw_dir)
    todo = [g for g in game_ids if g not in done]
    if limit is not None:
        todo = todo[:limit]
    print(f"{season}: {len(game_ids):,} games, {len(done):,} done, "
          f"{len(todo):,} to fetch ({'V3' if uses_v3(season) else 'V2'})")
    if not todo:
        return 0

    dest = status_path(season, raw_dir)
    buffer: list[pd.DataFrame] = []
    manifest: list[dict] = []
    written = 0

    for i, game_id in enumerate(todo, start=1):
        try:
            rows, row_manifest = fetch_game_status(game_id, season, delay=delay)
        except Exception as exc:                       # noqa: BLE001 — keep going
            print(f"  ERROR {season} {game_id}: {exc}")
            manifest.append({"season": season, "game_id": game_id, "dressed_rows": 0,
                             "inactive_rows": 0, "played_rows": 0, "endpoint": "",
                             "fetched_at": datetime.now().astimezone().isoformat(),
                             "error": str(exc)[:200]})
            time.sleep(delay)
            continue
        if not rows.empty:
            buffer.append(rows)
        manifest.append(row_manifest)

        if i % flush_every == 0 or i == len(todo):
            if buffer:
                chunk = pd.concat(buffer, ignore_index=True)
                _append(chunk, dest)          # rows first: the resume key is the data
                written += len(chunk)
                buffer = []
            _append_manifest(manifest, raw_dir)
            manifest = []
            print(f"  {season}: {i:,}/{len(todo):,} games, {written:,} status rows "
                  f"→ {dest}")
        time.sleep(delay)

    return written


def run(cfg: dict, seasons: list[str] | None = None, limit: int | None = None) -> Path:
    raw_dir = Path(cfg["data"]["raw_dir"])
    bs_cfg = cfg.get("data", {}).get("boxscore_status", {})
    first = bs_cfg.get("first_season", FIRST_SEASON)
    delay = float(bs_cfg.get("delay", DELAY))
    flush_every = int(bs_cfg.get("flush_every", FLUSH_EVERY))

    # Newest first. The run is long enough that it will be interrupted at least once, and
    # a partial backfill is worth far more on recent seasons — they are the model's test
    # split, and the reason vocabulary is richest there.
    seasons = seasons or sorted(
        (s for s in cfg["data"]["seasons"]
         if _season_start_year(s) >= _season_start_year(first)),
        key=_season_start_year, reverse=True)
    print(f"Box-score status backfill: {len(seasons)} seasons "
          f"({seasons[0]} → {seasons[-1]}), inactive lists from {first}")

    total = 0
    for season in seasons:
        total += backfill_season(season, raw_dir, delay, flush_every, limit)
    print(f"Backfilled {total:,} status rows → {raw_dir}")
    return raw_dir / MANIFEST_NAME


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill per-game inactive lists and DNP reasons.")
    parser.add_argument("--seasons", nargs="+", default=None,
                        help="Seasons to backfill (default: first_season → present).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Fetch at most this many games per season (smoke test).")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.seasons, limit=args.limit)
