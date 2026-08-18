"""Draft-board screenshots → validated pick logs (docs/draft-board-ingestion-plan.md).

`docs/entry-collection-plan.md` makes the full 12-entrant pick log of every
data-collection draft the capture target — the field-calibration data the opponent model
has never had. DK's draft room is login-gated with no archive, so the board exists only
as a screenshot taken at draft close. This module turns those screenshots into a tabular
artifact without trusting any single read of the image:

1. `--tile` crops a board into upscaled bands and strips for the two-pass read protocol
   (Claude transcribes; a hand-written transcription is an equal citizen).
2. The default run validates each transcription against the ranking file recorded in the
   manifest: every abbreviated cell ("N. Alexande...", "St. Curry") must resolve to
   EXACTLY ONE player of the same team and position, the snake arithmetic must hold
   twice over, the header's per-seat G/F/C counts must balance, and no player may appear
   twice. Any failure names the cell and blocks the artifact.
3. Green boards land in `data/features/draft_pick_log.parquet`, one row per pick.

The same-date rankings CSV is the only naming authority — cells are never resolved from
basketball knowledge, because the pool contains rookies no such knowledge covers.

Files, per board, in `data/raw/adp_autodrafted_boards_<year>/`:

- `draftboard_<Mon><D>_<YYYY>.png|.jpg`      the screenshot (user, at draft close)
- `draftboard_<date>_header.csv`             seat,entrant,g,f,c            (12 rows)
- `draftboard_<date>_picks.csv`              seat,round,pick_in_round,overall,
                                             cell_name,pos,team           (192 rows)
- `boards_manifest.csv`                      file,contest,our_entrant,mode,
                                             ranking_file,notes         (1 row/board)

The manifest is the provenance record `entry-collection-plan.md` requires — a pod whose
ranking version is unrecorded is uninterpretable, so a board without a manifest row
fails validation rather than guessing.

Usage:
    python -m src.data.draft_boards                    # validate all boards → parquet
    python -m src.data.draft_boards --tile             # write read-protocol tiles
    python -m src.data.draft_boards --cell 13.1 --board draftboard_Aug16_2026
    python -m src.data.draft_boards --status           # captured / transcribed / valid
"""

import argparse
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from PIL import Image

from src.data.adp_draftkings import (
    TEAM_ALIASES,
    _MONTH_FORMATS,
    read_board,
    season_of_board,
)

N_SEATS = 12
N_ROUNDS = 16
N_PICKS = N_SEATS * N_ROUNDS
# DK enforces these at the draft (dk_best_ball_rules.md); they make the autodraft
# best-available comparison cap-aware. Soft-check only — opponents may do anything legal.
ROSTER_CAPS = {"G": 8, "F": 8, "C": 3}

# Month spelled both ways across real captures (Aug16 but July28), same as the rankings
# filenames. A non-matching name silently skips the board, which for a source that cannot
# be re-captured is the worst failure — so `--status` lists unparseable images loudly.
_IMAGE_RE = re.compile(r"draftboard_([A-Za-z]{3,9})(\d{1,2})_(\d{4})\.(png|jpe?g)$",
                       re.IGNORECASE)

_MANIFEST_COLS = ["file", "contest", "our_entrant", "mode", "ranking_file", "notes"]
_HEADER_COLS = ["seat", "entrant", "g", "f", "c"]
_PICKS_COLS = ["seat", "round", "pick_in_round", "overall", "cell_name", "pos", "team"]

_LOG_COLS = ["draft_date", "season", "source_file", "contest", "mode", "ranking_file",
             "seat", "entrant", "our_seat", "overall", "round", "pick_in_round",
             "dk_player_id", "player_name", "pos", "team", "adp", "adp_rank",
             "adp_censored"]


def capture_date_from_name(path: str | Path) -> str | None:
    """The capture date encoded in the image filename, ISO. None if it does not match."""
    m = _IMAGE_RE.search(Path(path).name)
    if not m:
        return None
    mon, day, year = m.groups()[:3]
    for fmt in _MONTH_FORMATS:
        try:
            return datetime.strptime(f"{mon} {day} {year}", fmt).date().isoformat()
        except ValueError:
            continue
    return None


def board_dirs(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Every season's board directory — `adp_autodrafted_boards_<year>`."""
    return sorted(p for p in Path(raw_dir).glob("adp_autodrafted_boards_*")
                  if p.is_dir())


def board_image_paths(in_dir: str | Path) -> list[Path]:
    return sorted(p for p in Path(in_dir).iterdir()
                  if _IMAGE_RE.search(p.name) and capture_date_from_name(p))


# ── Grid detection ────────────────────────────────────────────────────────────
# The board is bright cells on a near-black ground, so gaps between cells are runs of
# dark pixel columns/rows. Detection is re-run per image rather than hard-coded: a DK
# layout change then fails loudly here instead of silently mis-cropping every cell.

_DARK = 40          # mean-luminance threshold for a gap pixel row/column
_SCAN_STARTS = (200, 150, 250, 300)   # y where the grid scan begins, tried in order


def detect_grid(img: Image.Image) -> tuple[list[int], list[int]]:
    """Gap centers `(xs, ys)`: 13 vertical and 17 horizontal, bounding 12 × 16 cells."""
    gray = np.asarray(img.convert("L"), dtype=float)
    H, W = gray.shape

    def _groups(means: np.ndarray, offset: int = 0) -> list[int]:
        dark = [i for i, v in enumerate(means) if v < _DARK]
        out: list[list[int]] = []
        for i in dark:
            if out and i - out[-1][-1] <= 1:
                out[-1].append(i)
            else:
                out.append([i])
        return [int(np.mean(g)) + offset for g in out]

    for y0 in _SCAN_STARTS:
        if y0 >= H:
            continue
        xs = _groups(gray[y0:, :].mean(axis=0))
        ys = _groups(gray[:, 10:W - 10].mean(axis=1)[y0:], offset=y0)
        if len(xs) == N_SEATS + 1 and len(ys) >= N_ROUNDS + 1:
            return xs, ys[-(N_ROUNDS + 1):]
    raise ValueError(
        f"could not find a {N_SEATS}x{N_ROUNDS} grid (needs {N_SEATS + 1} vertical and "
        f"{N_ROUNDS + 1}+ horizontal dark gaps) — has the DK board layout changed?")


# ── Tiling for the two-pass read protocol ─────────────────────────────────────

def tile_board(img_path: str | Path, out_root: str | Path = "outputs/draft_boards/tiles",
               band_scale: int = 2, strip_scale: int = 3) -> Path:
    """Crops for the read protocol: header + 4 row bands (pass 1), 24 half-column
    strips (pass 2). Returns the tile directory."""
    img_path = Path(img_path)
    img = Image.open(img_path)
    xs, ys = detect_grid(img)
    out_dir = Path(out_root) / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    def _save(box: tuple[int, int, int, int], scale: int, name: str) -> None:
        crop = img.crop(box)
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
        crop.save(out_dir / name)

    # pass 1: the entrant header band, then 4-round row bands
    _save((0, max(ys[0] - 115, 0), img.width, ys[0] + 3), band_scale, "header.png")
    for k in range(4):
        r0, r1 = 4 * k + 1, 4 * k + 4
        _save((0, ys[r0 - 1] - 3, img.width, ys[r1] + 3), band_scale,
              f"rows_{r0:02d}_{r1:02d}.png")
    # pass 2: per-seat column strips, split at round 8 so the upscale stays legible
    for s in range(1, N_SEATS + 1):
        x0, x1 = xs[s - 1] - 2, xs[s] + 2
        for half, (ra, rb) in enumerate([(1, 8), (9, 16)]):
            _save((x0, ys[ra - 1] - 3, x1, ys[rb] + 3), strip_scale,
                  f"col_{s:02d}_r{ra:02d}_{rb:02d}.png")
    n = 5 + 2 * N_SEATS
    print(f"... {n:,} tiles → {out_dir}")
    return out_dir


def cell_crop(img_path: str | Path, round_: int, seat: int,
              out_root: str | Path = "outputs/draft_boards/tiles",
              scale: int = 4) -> Path:
    """A single 4x cell crop, for resolving a disagreement or a validator failure.
    Addressed `round.seat` — the seat is the COLUMN, so even rounds keep the address the
    validator prints rather than the mirrored on-screen pick label."""
    img_path = Path(img_path)
    img = Image.open(img_path)
    xs, ys = detect_grid(img)
    out_dir = Path(out_root) / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    box = (xs[seat - 1] - 2, ys[round_ - 1] - 2, xs[seat] + 2, ys[round_] + 2)
    crop = img.crop(box)
    crop = crop.resize((crop.width * scale, crop.height * scale), Image.LANCZOS)
    dest = out_dir / f"cell_r{round_:02d}_s{seat:02d}.png"
    crop.save(dest)
    print(f"... 1 cell crop (round {round_}, seat {seat}) → {dest}")
    return dest


# ── The matcher ───────────────────────────────────────────────────────────────

def split_cell_name(cell_name: str) -> tuple[str, str]:
    """`'St. Curry'` → `('St', 'Curry')`; `'N. Alexande...'` → `('N', 'Alexande')`;
    a name with no dotted first token is all last-prefix."""
    name = cell_name.replace("…", "...").strip()
    if name.endswith("..."):
        name = name[:-3].strip()
    parts = name.split(None, 1)
    if len(parts) == 2 and parts[0].endswith("."):
        return parts[0][:-1], parts[1].strip()
    return "", name


def resolve_cell(cell_name: str, pos: str, team: str,
                 board: pd.DataFrame) -> pd.DataFrame:
    """Rankings rows the displayed cell could be: same team AND position, first name
    starting with the cell's first-prefix, rest of the name starting with its
    last-prefix. The validator requires exactly one."""
    first_prefix, last_prefix = split_cell_name(cell_name)
    team = TEAM_ALIASES.get(team.strip().upper(), team.strip().upper())
    pos = pos.strip().upper()
    cand = board[(board["team"] == team) & (board["position"] == pos)]
    fp, lp = first_prefix.casefold(), last_prefix.casefold()

    def _hit(full_name: str) -> bool:
        parts = full_name.split(None, 1)
        first = parts[0]
        rest = parts[1] if len(parts) == 2 else ""
        if fp and not first.casefold().startswith(fp):
            return False
        target = rest if fp else full_name
        return target.casefold().startswith(lp)

    return cand[cand["player_name"].map(_hit)]


# ── Transcription + manifest IO ───────────────────────────────────────────────

def transcription_paths(img_path: str | Path) -> tuple[Path, Path]:
    img_path = Path(img_path)
    stem = img_path.stem
    return (img_path.with_name(f"{stem}_header.csv"),
            img_path.with_name(f"{stem}_picks.csv"))


def load_manifest(in_dir: str | Path) -> pd.DataFrame:
    path = Path(in_dir) / "boards_manifest.csv"
    if not path.exists():
        return pd.DataFrame(columns=_MANIFEST_COLS)
    df = pd.read_csv(path, dtype=str).fillna("")
    missing = [c for c in _MANIFEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}; expected {_MANIFEST_COLS}")
    return df


def _entrant_matches(header_entrant: str, full_entrant: str) -> bool:
    """DK truncates usernames at ~10 chars, so a header value ending in '...' matches on
    its visible prefix."""
    h = header_entrant.replace("…", "...").strip()
    if h.endswith("..."):
        return full_entrant.strip().casefold().startswith(h[:-3].casefold())
    return h.casefold() == full_entrant.strip().casefold()


# ── Validation ────────────────────────────────────────────────────────────────

def validate_board(img_path: str | Path, manifest: pd.DataFrame,
                   rankings_dir: str | Path = "data/raw/dk_draft_rankings",
                   ) -> tuple[pd.DataFrame | None, list[str], list[str]]:
    """One board's transcription → (pick-log frame or None, hard errors, soft notes).

    Hard errors block the artifact and each names the cell (`round.seat`) it came from.
    Soft notes are printed and never block — opponents may legally deviate from ADP.
    """
    img_path = Path(img_path)
    errors: list[str] = []
    notes: list[str] = []
    date = capture_date_from_name(img_path)

    hdr_path, picks_path = transcription_paths(img_path)
    if not hdr_path.exists() or not picks_path.exists():
        return None, [f"missing transcription ({hdr_path.name}, {picks_path.name})"], []

    # manifest row — provenance is a hard requirement, not metadata
    mrow = manifest[manifest["file"] == img_path.name]
    if len(mrow) != 1:
        return None, [f"needs exactly one boards_manifest.csv row with "
                      f"file={img_path.name!r} (found {len(mrow)})"], []
    mrow = mrow.iloc[0]
    if mrow["mode"] not in ("auto", "live"):
        errors.append(f"manifest mode={mrow['mode']!r}, expected 'auto' or 'live'")
    ranking_path = Path(rankings_dir) / mrow["ranking_file"]
    if not ranking_path.exists():
        return None, errors + [f"manifest ranking_file {ranking_path} not found"], []
    board = read_board(ranking_path)
    if board["capture_date"].iloc[0] != date:
        notes.append(f"ranking file dated {board['capture_date'].iloc[0]} vs board "
                     f"{date} — allowed, but confirm it is the file the seat drafted with")

    header = pd.read_csv(hdr_path)
    picks = pd.read_csv(picks_path)
    for df, cols, name in [(header, _HEADER_COLS, hdr_path.name),
                           (picks, _PICKS_COLS, picks_path.name)]:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            return None, errors + [f"{name} is missing columns {missing}"], []

    # ── structure: the snake arithmetic, twice over ──
    if len(header) != N_SEATS or sorted(header["seat"]) != list(range(1, N_SEATS + 1)):
        errors.append(f"header must be seats 1..{N_SEATS}, one row each")
    if len(picks) != N_PICKS:
        errors.append(f"expected {N_PICKS} pick rows, found {len(picks)}")
    if sorted(picks["overall"]) != list(range(1, N_PICKS + 1)):
        errors.append("`overall` is not a permutation of 1..192")
    for _, p in picks.iterrows():
        addr = f"cell {p['round']}.{p['seat']}"
        want_overall = (p["round"] - 1) * N_SEATS + p["pick_in_round"]
        if p["overall"] != want_overall:
            errors.append(f"{addr}: overall {p['overall']} != (round-1)*12+pick "
                          f"{want_overall}")
        want_pick = p["seat"] if p["round"] % 2 == 1 else N_SEATS + 1 - p["seat"]
        if p["pick_in_round"] != want_pick:
            errors.append(f"{addr}: pick_in_round {p['pick_in_round']} breaks the snake "
                          f"(seat {p['seat']} in round {p['round']} should be {want_pick})")
    counts = picks.groupby("seat").size()
    bad_seats = counts[counts != N_ROUNDS]
    if len(bad_seats):
        errors.append(f"seats without exactly {N_ROUNDS} picks: "
                      f"{dict(bad_seats.astype(int))}")

    # ── resolve every cell against the ranking file ──
    resolved: list[pd.Series | None] = []
    for _, p in picks.iterrows():
        addr = f"cell {p['round']}.{p['seat']}"
        hits = resolve_cell(str(p["cell_name"]), str(p["pos"]), str(p["team"]), board)
        if len(hits) == 1:
            resolved.append(hits.iloc[0])
        elif len(hits) == 0:
            errors.append(f"{addr}: {p['cell_name']!r} ({p['pos']} {p['team']}) matches "
                          f"NO ONE in {ranking_path.name} — re-read the cell")
            resolved.append(None)
        else:
            errors.append(f"{addr}: {p['cell_name']!r} ({p['pos']} {p['team']}) is "
                          f"AMBIGUOUS: {list(hits['player_name'])} — DK's own rendering "
                          f"disambiguates; re-read the cell")
            resolved.append(None)
    ok = [r is not None for r in resolved]
    ids = pd.Series([r["dk_player_id"] if o else None for r, o in zip(resolved, ok)])
    dupes = ids.dropna()[ids.dropna().duplicated(keep=False)]
    if len(dupes):
        names = {board.set_index("dk_player_id")["player_name"].get(i) for i in dupes}
        errors.append(f"players drafted twice: {sorted(names)}")

    # ── the per-seat G/F/C checksum from the header ──
    tally = picks.assign(pos=picks["pos"].str.upper()) \
                 .pivot_table(index="seat", columns="pos", aggfunc="size", fill_value=0)
    for _, h in header.iterrows():
        got = tuple(int(tally.loc[h["seat"], c]) if h["seat"] in tally.index
                    and c in tally.columns else 0 for c in ("G", "F", "C"))
        want = (int(h["g"]), int(h["f"]), int(h["c"]))
        if sum(want) != N_ROUNDS:
            errors.append(f"header seat {h['seat']}: G+F+C = {sum(want)} != {N_ROUNDS}")
        if got != want:
            errors.append(f"seat {h['seat']} ({h['entrant']}): column G/F/C {got} != "
                          f"header {want} — one of the 16 cells or the header is misread")

    # ── which seat is ours ──
    ours = [int(h["seat"]) for _, h in header.iterrows()
            if _entrant_matches(str(h["entrant"]), str(mrow["our_entrant"]))]
    if len(ours) != 1:
        errors.append(f"our_entrant {mrow['our_entrant']!r} matches header seats {ours}; "
                      f"need exactly one")
    our_seat = ours[0] if len(ours) == 1 else -1

    if errors:
        return None, errors, notes

    # ── soft report: pick order vs the market, and our seat vs best-available ──
    log = picks.copy()
    for col in ("dk_player_id", "player_name", "adp", "adp_rank", "adp_censored"):
        log[col] = [r[col] for r in resolved]
    with_adp = log.dropna(subset=["adp"])
    rho = with_adp["overall"].corr(with_adp["adp"], method="spearman")
    notes.append(f"pick order vs ADP: Spearman {rho:.3f} on {len(with_adp)} picks with "
                 f"an ADP; {int(log['adp'].isna().sum())} picks have none, "
                 f"{int(log['adp_censored'].sum())} censored")
    dev = with_adp.assign(gap=(with_adp["overall"] - with_adp["adp"]).abs()) \
                  .nlargest(3, "gap")
    notes.append("largest |pick - ADP| gaps: " + "; ".join(
        f"{r.player_name} pick {int(r.overall)} vs ADP {r.adp:.1f}"
        for r in dev.itertuples()))
    if mrow["mode"] == "auto":
        agree, deviations = autodraft_agreement(log, our_seat, board)
        notes.append(f"our seat vs cap-aware best-available: {agree}/{N_ROUNDS} agree"
                     + (f"; deviations: {deviations}" if deviations else ""))

    log["seat"] = log["seat"].astype(int)
    seat_entrant = header.set_index("seat")["entrant"]
    out = pd.DataFrame({
        "draft_date": date,
        "season": season_of_board(date),
        "source_file": img_path.name,
        "contest": mrow["contest"],
        "mode": mrow["mode"],
        "ranking_file": mrow["ranking_file"],
        "seat": log["seat"],
        "entrant": log["seat"].map(seat_entrant),
        "our_seat": log["seat"] == our_seat,
        "overall": log["overall"].astype(int),
        "round": log["round"].astype(int),
        "pick_in_round": log["pick_in_round"].astype(int),
        "dk_player_id": log["dk_player_id"].astype("int64"),
        "player_name": log["player_name"],
        "pos": log["pos"].str.upper(),
        "team": log["team"].str.upper().replace(TEAM_ALIASES),
        "adp": log["adp"],
        "adp_rank": log["adp_rank"],
        "adp_censored": log["adp_censored"],
    })[_LOG_COLS].sort_values("overall").reset_index(drop=True)
    return out, [], notes


def autodraft_agreement(log: pd.DataFrame, our_seat: int,
                        board: pd.DataFrame) -> tuple[int, list[str]]:
    """How often our seat took the cap-aware best-available row of its own ranking.

    The uploaded CSV's row order IS the ranking (DK's instructions), so best-available is
    the earliest un-drafted row whose position still fits under the 8G/8F/3C caps. Soft
    only: DK's engine may break ties or handle caps differently at the margin.
    """
    order = log.sort_values("overall")
    taken: set[int] = set()
    roster: dict[str, int] = {"G": 0, "F": 0, "C": 0}
    agree, deviations = 0, []
    ranking = board.reset_index(drop=True)
    for _, p in order.iterrows():
        if int(p["seat"]) == our_seat:
            open_pos = [q for q, cap in ROSTER_CAPS.items() if roster[q] < cap]
            avail = ranking[~ranking["dk_player_id"].isin(taken)
                            & ranking["position"].isin(open_pos)]
            expected = avail.iloc[0]
            if expected["dk_player_id"] == p["dk_player_id"]:
                agree += 1
            else:
                deviations.append(f"pick {int(p['overall'])}: took {p['player_name']}, "
                                  f"best-available {expected['player_name']}")
            roster[str(p["pos"]).upper()] += 1
        taken.add(int(p["dk_player_id"]))
    return agree, deviations


# ── Status + entry point ──────────────────────────────────────────────────────

def print_status(raw_dir: str | Path = "data/raw") -> None:
    print("Draft-board pick logs — screenshots at draft close, NOT re-capturable")
    dirs = board_dirs(raw_dir)
    if not dirs:
        print("  No adp_autodrafted_boards_* directory yet.")
        return
    for d in dirs:
        manifest = load_manifest(d)
        unparseable = [p.name for p in d.iterdir()
                       if p.suffix.lower() in (".png", ".jpg", ".jpeg")
                       and not capture_date_from_name(p)]
        for name in unparseable:
            print(f"  WARNING {d.name}/{name}: filename does not parse as "
                  f"draftboard_<Mon><D>_<YYYY> — this board is invisible to validation")
        for img in board_image_paths(d):
            hdr, picks = transcription_paths(img)
            transcribed = hdr.exists() and picks.exists()
            in_manifest = (manifest["file"] == img.name).sum() == 1
            if not transcribed:
                state = "NEEDS TRANSCRIPTION (make draft-boards-tile, then the protocol)"
            elif not in_manifest:
                state = "NEEDS boards_manifest.csv row"
            else:
                _, errs, _ = validate_board(img, manifest)
                state = "valid" if not errs else f"INVALID ({len(errs)} errors)"
            print(f"  {capture_date_from_name(img)}  {d.name}/{img.name:<32} {state}")


def run(raw_dir: str | Path = "data/raw",
        features_dir: str | Path = "data/features",
        rankings_dir: str | Path = "data/raw/dk_draft_rankings") -> pd.DataFrame:
    frames, n_bad = [], 0
    for d in board_dirs(raw_dir):
        manifest = load_manifest(d)
        for img in board_image_paths(d):
            log, errors, notes = validate_board(img, manifest, rankings_dir)
            if errors:
                n_bad += 1
                print(f"  {img.name}: FAILED validation —")
                for e in errors:
                    print(f"    {e}")
            else:
                frames.append(log)
                print(f"  {img.name}: valid ({N_PICKS} picks, "
                      f"{log['entrant'].nunique()} entrants)")
            for note in notes:
                print(f"    note: {note}")
    if frames:
        out = pd.concat(frames, ignore_index=True)
        features_dir = Path(features_dir)
        features_dir.mkdir(parents=True, exist_ok=True)
        dest = features_dir / "draft_pick_log.parquet"
        out.to_parquet(dest, index=False)
        print(f"... {len(out):,} picks from {len(frames)} boards → {dest}")
    else:
        out = pd.DataFrame(columns=_LOG_COLS)
        print("  No validated boards; nothing written.")
    if n_bad:
        raise SystemExit(f"{n_bad} board(s) failed validation — artifact excludes them")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate draft-board transcriptions into the pooled pick log.")
    parser.add_argument("--tile", action="store_true",
                        help="Write read-protocol tiles for boards lacking a "
                             "transcription (or for --board).")
    parser.add_argument("--cell", default=None, metavar="R.S",
                        help="Write one 4x cell crop at round R, seat S (needs --board).")
    parser.add_argument("--board", default=None,
                        help="Image stem or path to restrict --tile / address --cell.")
    parser.add_argument("--status", action="store_true",
                        help="Report captured / transcribed / valid. No writes.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    data_cfg = cfg.get("data", {})
    raw_dir = data_cfg.get("raw_dir", "data/raw")

    def _find_images() -> list[Path]:
        imgs = [p for d in board_dirs(raw_dir) for p in board_image_paths(d)]
        if args.board:
            imgs = [p for p in imgs if args.board in (p.stem, p.name, str(p))]
            if not imgs:
                raise SystemExit(f"no board image matching {args.board!r}")
        return imgs

    if args.status:
        print_status(raw_dir)
    elif args.cell:
        if not args.board:
            raise SystemExit("--cell needs --board")
        r, s = (int(v) for v in args.cell.split("."))
        cell_crop(_find_images()[0], r, s)
    elif args.tile:
        for img in _find_images():
            hdr, picks = transcription_paths(img)
            if args.board or not (hdr.exists() and picks.exists()):
                tile_board(img)
    else:
        run(raw_dir=raw_dir,
            features_dir=data_cfg.get("features_dir", "data/features"),
            rankings_dir=data_cfg.get("adp", {}).get("draftkings", {})
                                 .get("dir", "data/raw/dk_draft_rankings"))
