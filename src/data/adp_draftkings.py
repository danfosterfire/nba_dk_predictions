"""DraftKings NBA Best Ball ADP — ingest of manually-downloaded pre-draft rankings.

This is the **only** real DK Best Ball ADP that exists anywhere. It is not scrapable and
never will be, so this module ingests rather than fetches:

- DK's best-ball pages have **zero Wayback snapshots** — no history, for any season.
- `api.draftkings.com` exposes nothing relevant unauthenticated (`/draftgroups/v1/…`
  returns 400 `SPO117`; `/bestball/v1/predraftrankings` 404s).
- Every third-party DK ADP tracker found is NFL-only. NBA Best Ball is too small a market.

The CSV is a logged-in download from the draft lobby, and the board is live only while
contests are open. **Nothing captured here can be recovered later**, which makes the
manual capture routine the deadline item in `docs/adp-plan.md`.

Drop files in `data/raw/dk_draft_rankings/` named `DkPreDraftRankings_<Mon><D>_<YYYY>.csv`
(e.g. `DkPreDraftRankings_Oct17_2025.csv`); the date in the filename is the capture date.

## What the file contains

`ID, Name, Position, ADP, Team`, plus a blank column and an `Instructions` column that are
dropped. Only a minority of the pool carries an ADP — 249 of 698 (Oct 2025), 202 of 942
(Jul 2026) — because DK reports it only for players drafted often enough to average.

- **ADP is a live-computed mean, not a rank**: seven significant figures, moving daily.
- **It is right-censored.** ~200-250 players over a 192-pick draft, maxing near 185:
  rarely-drafted players pile up against the boundary rather than extending past it.
  `adp_censored` flags them, because the pile-up otherwise drives any fit through it.
- **Board thickness depends on how long drafts have run.** A July board is thinner and
  less compressed than an October one, so they are not interchangeable observations.

## `ID` is a persistent key — the reason this module also builds the id map

Verified across the two boards: 667 shared ids, and the name agrees on **100.0%** of them
(Jokic = 830650 in both). So DK's side of the join needs a one-time id map rather than
fuzzy name matching forever. `build_id_map` writes it; only the consensus sources in
`src/features/adp.py` still need name normalization.

Usage:
    python -m src.data.adp_draftkings              # ingest every board + build id map
    python -m src.data.adp_draftkings --status     # what is captured, no file writes
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

# DK writes New Orleans as both NO and NOP across boards — one player (Jordan Ford, id
# 955995) carried NO on the 2026-27 board while every other Pelican carried NOP. It is a
# data-entry inconsistency, not a 31st franchise, and it would otherwise read as an
# unmatched team. Aliases are applied on read so downstream never sees the raw code.
TEAM_ALIASES = {"NO": "NOP", "NOR": "NOP", "PHO": "PHX", "GS": "GSW", "SA": "SAS",
                "NY": "NYK", "UTAH": "UTA", "WSH": "WAS", "BRK": "BKN", "CHO": "CHA"}

# DK name → nba_api name, for cases no rule recovers. Keep this list short: every entry is
# a hand-verified claim that two strings are the same person, and it is the one part of the
# join that cannot be checked mechanically.
NAME_ALIASES = {
    "Gregory Jackson": "GG Jackson",
    # Legal name change, 2021 — the same player under both names.
    "Enes Kanter": "Enes Freedom",
    # Nicknames nba_api uses as the canonical name.
    "Mohamed Bamba": "Mo Bamba",
    "Nene Hilario": "Nene",
    "Aleksandar Vezenkov": "Sasha Vezenkov",
}

# DK boards only exist while contests are open, which is roughly June-October for the
# *upcoming* season. So a board captured in month >= 6 is for the season starting that
# year. Verified on both anchors: Jul 2026 -> 2026-27 (2026 draft class present, 178 team
# changes), Oct 2025 -> 2025-26.
SEASON_MONTH_CUTOFF = 6

# At or above this share of the 192-pick draft, ADP is compressed against the boundary
# rather than measuring anything. 54 of 218 matched players sat here on the 2025-26 board,
# and excluding them cut the transfer function's error from 17.0 to 14.4 picks.
DRAFT_PICKS = 192
CENSOR_FRACTION = 0.885   # ~170 of 192

# The month is written both abbreviated and in full across real captures — `Oct17_2025`
# but `July28_2026` — so both spellings are accepted. A too-narrow pattern here does not
# error, it silently skips the board, which is the worst possible failure for a source
# that cannot be re-captured.
_FILENAME_RE = re.compile(r"DkPreDraftRankings_([A-Za-z]{3,9})(\d{1,2})_(\d{4})\.csv$")
_MONTH_FORMATS = ("%b %d %Y", "%B %d %Y")

_OUTPUT_COLS = ["captured_at", "capture_date", "season", "dk_player_id", "player_name",
                "position", "team", "adp", "adp_rank", "adp_censored", "pool_size",
                "source_detail", "snapshot_source"]


def capture_date_from_name(path: str | Path) -> str | None:
    """The capture date encoded in the filename, ISO. None if it does not match."""
    m = _FILENAME_RE.search(Path(path).name)
    if not m:
        return None
    mon, day, year = m.groups()
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(f"{mon} {day} {year}", fmt).date().isoformat()
        except ValueError:
            continue
    return None


def season_of_board(capture_date: str, cutoff: int = SEASON_MONTH_CUTOFF) -> str:
    """The season a board captured on `capture_date` is drafting for."""
    d = pd.Timestamp(capture_date)
    start = d.year if d.month >= cutoff else d.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


def read_board(path: str | Path, capture_date: str | None = None,
               cutoff: int = SEASON_MONTH_CUTOFF) -> pd.DataFrame:
    """One DK pre-draft rankings CSV as a tidy frame.

    Rows without an ADP are kept — they are the draftable pool, which the simulator needs
    when the board runs dry — but carry a null `adp`.
    """
    path = Path(path)
    capture_date = capture_date or capture_date_from_name(path)
    if capture_date is None:
        raise ValueError(f"cannot infer a capture date from {path.name!r}; expected "
                         "DkPreDraftRankings_<Mon><D>_<YYYY>.csv")
    raw = pd.read_csv(path, usecols=["ID", "Name", "Position", "ADP", "Team"])
    df = pd.DataFrame({
        "captured_at": capture_date,
        "capture_date": capture_date,
        "season": season_of_board(capture_date, cutoff),
        "dk_player_id": raw["ID"].astype("int64"),
        "player_name": raw["Name"].astype(str).str.strip(),
        "position": raw["Position"].astype(str).str.strip(),
        "team": raw["Team"].astype(str).str.strip().replace(TEAM_ALIASES),
        "adp": pd.to_numeric(raw["ADP"], errors="coerce"),
    })
    df["adp_rank"] = df["adp"].rank(method="min")
    df["adp_censored"] = df["adp"] >= DRAFT_PICKS * CENSOR_FRACTION
    df["pool_size"] = len(df)
    df["source_detail"] = "dk"
    df["snapshot_source"] = "draftkings"
    return df[_OUTPUT_COLS]


def board_paths(in_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(in_dir).glob("DkPreDraftRankings_*.csv")
                  if capture_date_from_name(p))


def load_boards(in_dir: str | Path, cutoff: int = SEASON_MONTH_CUTOFF) -> pd.DataFrame:
    """Every board on disk, concatenated, ascending by capture date."""
    frames = [read_board(p, cutoff=cutoff) for p in board_paths(in_dir)]
    if not frames:
        return pd.DataFrame(columns=_OUTPUT_COLS)
    return (pd.concat(frames, ignore_index=True)
            .sort_values(["capture_date", "adp"], na_position="last")
            .reset_index(drop=True))


def id_stability(boards: pd.DataFrame) -> dict:
    """Check that `dk_player_id` really is a persistent key before relying on it.

    Returns the number of ids seen on more than one board and the share of those whose
    name is identical throughout. Anything below 1.0 means DK recycles ids and the id map
    below is unsafe — so this is asserted rather than assumed.
    """
    seen = boards.groupby("dk_player_id")["player_name"].agg(["nunique", "count"])
    repeated = seen[seen["count"] > 1]
    if repeated.empty:
        return {"repeated_ids": 0, "name_agreement": 1.0}
    return {"repeated_ids": int(len(repeated)),
            "name_agreement": float((repeated["nunique"] == 1).mean())}


# ── The one-time id map ───────────────────────────────────────────────────────

def normalize_name(name: str) -> str:
    """Fold a player name to a join key: ASCII, no punctuation, no suffix, lowercase."""
    import unicodedata
    n = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    n = re.sub(r"\b(Jr|Sr|II|III|IV|V)\.?\b", "", n)
    n = re.sub(r"[^A-Za-z ]", "", n)
    return re.sub(r"\s+", " ", n).strip().lower()


def _reversed_key(key: str) -> str:
    """`yang hansen` ↔ `hansen yang`. Two-token names only."""
    parts = key.split()
    return " ".join(reversed(parts)) if len(parts) == 2 else key


# A short form must be a genuine *prefix* of the long form, and at least this many
# characters, before two names are called the same person.
#
# ⛔ The obvious cheaper rule — same surname, same first initial — is WRONG, and it fails
# silently in the worst direction: it invents matches. Measured on the two real boards it
# produced 11 false matches out of 12, including `Cameron Boozer` → **Carlos Boozer**,
# `Darryn Peterson` → **Drew Peterson**, `RJ Davis` → **Ricky Davis** and `Javante McCoy`
# → **Jelani McCoy**. It also "achieved" 0.0% unmatched, which is exactly why an unmatched
# *rate* is not a sufficient check: a join that fabricates rows scores perfectly on it.
# Prefix matching accepts `Alexandre Sarr` → `Alex Sarr` and rejects all eleven.
MIN_PREFIX = 3

# Fuzzy steps additionally require the candidate to have been on an NBA roster within this
# many seasons of the board. A draftable board is a *current* player pool, so a candidate
# who last played decades ago is not the same person however well the strings line up.
#
# This guard is not belt-and-braces; prefix matching needs it. Without it `Mikel Brown Jr.`
# (a 2026 draft-class rookie) matches **Mike Brown**, because "mikel" does start with
# "mike". Name similarity alone cannot separate a diminutive from a different name that
# happens to share a stem — era can.
RECENT_SEASONS = 3


def _prefix_match(dk_first: str, ref_first: str) -> bool:
    """True if one first name is a plausible short form of the other."""
    short, long_ = sorted((dk_first, ref_first), key=len)
    return len(short) >= MIN_PREFIX and long_.startswith(short)


def _season_start(season: str) -> int:
    return int(str(season)[:4])


def build_id_map(boards: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Map `dk_player_id` → `nba_api` `player_id`, once, for reuse.

    A cascade, widening only after the stricter methods have had their chance, with the
    method recorded per row so a match is auditable rather than trusted. Each step exists
    because a real player needed it:

    | method | what it fixes | example |
    |---|---|---|
    | `name+team` | the common case | — |
    | `name` | a player who changed teams between the board and the roster file | — |
    | `reversed` | surname-first romanization | `Hansen Yang` → `Yang Hansen` |
    | `prefix` | long vs short first name, **prefix only, one candidate** | `Alexandre Sarr` → `Alex Sarr` |
    | `alias` | nicknames, which no rule recovers | `Gregory Jackson` → `GG Jackson` |

    **`no_nba_history` and `unmatched` are kept apart, and that distinction is the point.**
    A player on a board for a season the roster data does not yet cover may simply never
    have played — every incoming rookie is in that position, and there is no `player_id`
    to find. Collapsing the two would turn "the 2026 draft class exists" into what looks
    like a join failure, exactly as `report_calibration.py` keeps `absent` apart from
    `unmatched` for the same reason. Only `unmatched` is a defect.
    """
    dk = (boards.sort_values("capture_date")
          .groupby("dk_player_id", as_index=False)
          .agg(player_name=("player_name", "last"), team=("team", "last"),
               season=("season", "last")))
    dk["key"] = dk["player_name"].map(normalize_name)

    # `season` is carried through deliberately: dropping it here makes the era guard in
    # `_recent` silently degrade to "always true" rather than fail, which is how
    # `Mikel Brown Jr.` → Mike Brown (last seen 1996-97) survived a first fix.
    keep = ["player_id", "player_name", "team_abbreviation"]
    if "season" in roster.columns:
        keep.append("season")
    ref = roster[keep].copy()
    ref["key"] = ref["player_name"].map(normalize_name)
    ref["key"] = ref["key"].replace({normalize_name(k): normalize_name(v)
                                     for k, v in NAME_ALIASES.items()})

    by_team = (ref.drop_duplicates(["key", "team_abbreviation"])
               .rename(columns={"team_abbreviation": "team"})[["key", "team", "player_id"]])
    by_name = ref.drop_duplicates("key").set_index("key")["player_id"]
    aliases = {normalize_name(k): normalize_name(v) for k, v in NAME_ALIASES.items()}

    # Latest roster season per player, for the era guard on the fuzzy steps.
    last_seen = (ref.groupby("player_id")["season"].max().map(_season_start).to_dict()
                 if "season" in ref.columns else {})
    newest = max(last_seen.values(), default=0)

    def _recent(pid: int) -> bool:
        return not last_seen or newest - last_seen.get(pid, 0) <= RECENT_SEASONS

    # Surname → [(first_name, player_id)], for the prefix step. A surname with two
    # plausible candidates cannot disambiguate them, so it yields nothing.
    surnames: dict[str, list[tuple[str, int]]] = {}
    for k, pid in by_name.items():
        parts = k.split()
        if len(parts) >= 2:
            surnames.setdefault(" ".join(parts[1:]), []).append((parts[0], pid))

    def _by_prefix(key: str):
        parts = key.split()
        if len(parts) < 2:
            return None
        first, last = parts[0], " ".join(parts[1:])
        hits = [pid for f, pid in surnames.get(last, [])
                if _prefix_match(first, f) and _recent(pid)]
        return hits[0] if len(hits) == 1 else None

    def _by_reversed(key: str):
        pid = by_name.get(_reversed_key(key))
        return pid if pid is not None and _recent(pid) else None

    out = dk.merge(by_team, on=["key", "team"], how="left")
    out["match_method"] = out["player_id"].notna().map({True: "name+team", False: ""})

    def _fill(mask: pd.Series, keys: pd.Series, lookup, method: str) -> None:
        if not mask.any():
            return
        hit = keys[mask].map(lookup)
        take = hit.notna()
        out.loc[hit[take].index, "player_id"] = hit[take].values
        out.loc[hit[take].index, "match_method"] = method

    _fill(out["player_id"].isna(), out["key"], by_name, "name")
    _fill(out["player_id"].isna(), out["key"], _by_reversed, "reversed")
    _fill(out["player_id"].isna(), out["key"], _by_prefix, "prefix")
    _fill(out["player_id"].isna(), out["key"].map(lambda k: aliases.get(k, k)),
          by_name, "alias")

    covered = set(roster_seasons(roster)) if "season" in roster.columns else set()
    missing = out["player_id"].isna()
    out.loc[missing, "match_method"] = [
        "no_nba_history" if s not in covered else "unmatched"
        for s in out.loc[missing, "season"]
    ]
    return out[["dk_player_id", "player_name", "team", "season", "key",
                "player_id", "match_method"]]


def roster_seasons(roster: pd.DataFrame) -> list[str]:
    return sorted(roster["season"].unique()) if "season" in roster.columns else []


def print_status(in_dir: str | Path) -> None:
    paths = board_paths(in_dir)
    print("DraftKings Best Ball ADP — manual capture, NOT scrapable, NOT backfillable")
    if not paths:
        print("  No boards captured yet.")
        return
    boards = load_boards(in_dir)
    print(f"  boards captured        {len(paths):>4}")
    print()
    print("  capture      season    pool   with ADP   censored")
    for day, g in boards.groupby("capture_date"):
        has = g["adp"].notna()
        print(f"  {day}   {g.season.iloc[0]:8s} {len(g):5d} {has.sum():9d} "
              f"{int(g.adp_censored.sum()):10d}")
    stab = id_stability(boards)
    print()
    print(f"  dk_player_id seen on >1 board: {stab['repeated_ids']:,}  "
          f"name agreement {stab['name_agreement']:.1%}")


# ── Entry point ───────────────────────────────────────────────────────────────

def run(in_dir: str | Path = "data/raw/dk_draft_rankings",
        features_dir: str | Path = "data/features",
        roster_path: str | Path = "data/features/season_matrix_roster_tierA.parquet",
        cutoff: int = SEASON_MONTH_CUTOFF) -> pd.DataFrame:
    boards = load_boards(in_dir, cutoff)
    if boards.empty:
        print(f"No DK boards found in {in_dir}. Nothing to do.")
        return boards

    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)

    stab = id_stability(boards)
    if stab["repeated_ids"] and stab["name_agreement"] < 1.0:
        print(f"  WARNING: dk_player_id is NOT stable — name agreement "
              f"{stab['name_agreement']:.1%} across {stab['repeated_ids']:,} repeated ids. "
              f"The id map below is unsafe; fall back to name matching.")

    dest = features_dir / "adp_draftkings.parquet"
    boards.to_parquet(dest, index=False)
    seasons = sorted(boards.season.unique())
    print(f"  Read {len(board_paths(in_dir)):,} boards, {len(boards):,} pool rows "
          f"({boards.adp.notna().sum():,} with ADP) over {len(seasons)} seasons "
          f"({', '.join(seasons)}) → {dest}")

    roster_path = Path(roster_path)
    if roster_path.exists():
        roster = pd.read_parquet(
            roster_path,
            columns=["player_id", "player_name", "team_abbreviation", "season"])
        id_map = build_id_map(boards, roster)
        map_dest = features_dir / "adp_dk_id_map.parquet"
        id_map.to_parquet(map_dest, index=False)
        counts = id_map.match_method.value_counts()
        # `no_nba_history` is not a failure — see build_id_map. Only `unmatched` is, and it
        # is the number held to the 0.0% standard `report_calibration.py` set.
        matchable = id_map[id_map.match_method != "no_nba_history"]
        bad = int((matchable.match_method == "unmatched").sum())
        print(f"  Mapped {len(matchable) - bad:,} of {len(matchable):,} matchable DK ids "
              f"→ nba_api player_id ({bad} unmatched, {bad / max(len(matchable), 1):.1%}); "
              f"{int((id_map.match_method == 'no_nba_history').sum()):,} have no NBA "
              f"history yet → {map_dest}")
        print(f"    methods: {', '.join(f'{k} {v:,}' for k, v in counts.items())}")
    else:
        print(f"  Skipped id map: {roster_path} not found (run `make season-matrix`).")
    return boards


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest manually-downloaded DraftKings Best Ball pre-draft rankings.")
    parser.add_argument("--input-dir", default=None,
                        help="Directory of DkPreDraftRankings_*.csv files.")
    parser.add_argument("--status", action="store_true",
                        help="Report what is captured. No writes.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    data_cfg = cfg.get("data", {})
    adp_cfg = data_cfg.get("adp", {})
    dk_cfg = adp_cfg.get("draftkings", {})
    in_dir = args.input_dir or dk_cfg.get("dir", "data/raw/dk_draft_rankings")

    if args.status:
        print_status(in_dir)
        raise SystemExit(0)

    run(in_dir=in_dir,
        features_dir=data_cfg.get("features_dir", "data/features"),
        cutoff=dk_cfg.get("season_month_cutoff", SEASON_MONTH_CUTOFF))
