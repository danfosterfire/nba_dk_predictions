"""The draft-board ingestion module: matcher, snake arithmetic, checksums, tiling."""

import numpy as np
import pandas as pd
from PIL import Image

from src.data.draft_boards import (
    N_PICKS,
    N_ROUNDS,
    N_SEATS,
    autodraft_agreement,
    capture_date_from_name,
    cell_crop,
    detect_grid,
    resolve_cell,
    split_cell_name,
    tile_board,
    validate_board,
)


def _board(rows: list[tuple]) -> pd.DataFrame:
    """A rankings frame shaped like adp_draftkings.read_board output."""
    df = pd.DataFrame(rows, columns=["dk_player_id", "player_name", "position", "team"])
    df["adp"] = np.arange(1.0, len(df) + 1)
    df["adp_rank"] = df["adp"]
    df["adp_censored"] = False
    return df


# ── filename dates ────────────────────────────────────────────────────────────

def test_capture_date_both_month_spellings_and_extensions():
    assert capture_date_from_name("draftboard_Aug16_2026.jpg") == "2026-08-16"
    assert capture_date_from_name("draftboard_July28_2026.png") == "2026-07-28"
    assert capture_date_from_name("draftboard_Oct7_2026.jpeg") == "2026-10-07"
    assert capture_date_from_name("screenshot.png") is None
    assert capture_date_from_name("draftboard_Notamonth1_2026.png") is None


# ── cell-name splitting ───────────────────────────────────────────────────────

def test_split_cell_name_forms():
    assert split_cell_name("N. Jokic") == ("N", "Jokic")
    assert split_cell_name("St. Curry") == ("St", "Curry")
    assert split_cell_name("V. Wemban...") == ("V", "Wemban")
    assert split_cell_name("V. Wemban…") == ("V", "Wemban")   # unicode ellipsis
    assert split_cell_name("M. Porter Jr.") == ("M", "Porter Jr.")
    assert split_cell_name("S. Gilgeous-...") == ("S", "Gilgeous-")
    assert split_cell_name("Nodot Name") == ("", "Nodot Name")


# ── the matcher ───────────────────────────────────────────────────────────────

def test_resolve_simple_and_truncated():
    board = _board([(1, "Nikola Jokic", "C", "DEN"),
                    (2, "Victor Wembanyama", "C", "SAS"),
                    (3, "Shai Gilgeous-Alexander", "G", "OKC"),
                    (4, "Michael Porter Jr.", "F", "BKN"),
                    (5, "Trey Murphy III", "F", "NOP")])
    assert resolve_cell("N. Jokic", "C", "DEN", board)["dk_player_id"].tolist() == [1]
    assert resolve_cell("V. Wemban...", "C", "SAS", board)["dk_player_id"].tolist() == [2]
    assert resolve_cell("S. Gilgeous-...", "G", "OKC", board)["dk_player_id"].tolist() == [3]
    assert resolve_cell("M. Porter Jr.", "F", "BKN", board)["dk_player_id"].tolist() == [4]
    assert resolve_cell("T. Murphy III", "F", "NOP", board)["dk_player_id"].tolist() == [5]


def test_resolve_dk_prefix_disambiguation():
    board = _board([(1, "Stephen Curry", "G", "GSW"),
                    (2, "Seth Curry", "G", "GSW"),
                    (3, "Jalen Williams", "F", "OKC"),
                    (4, "Jaylin Williams", "F", "OKC")])
    # DK's own rendering carries the longer first prefix, which is exact
    assert resolve_cell("St. Curry", "G", "GSW", board)["dk_player_id"].tolist() == [1]
    assert resolve_cell("Se. Curry", "G", "GSW", board)["dk_player_id"].tolist() == [2]
    assert resolve_cell("Jal. Williams", "F", "OKC", board)["dk_player_id"].tolist() == [3]
    # a bare initial is genuinely ambiguous and must return both, never guess
    assert len(resolve_cell("S. Curry", "G", "GSW", board)) == 2
    assert len(resolve_cell("J. Williams", "F", "OKC", board)) == 2


def test_resolve_requires_team_and_pos():
    board = _board([(1, "Donovan Mitchell", "G", "CLE"),
                    (2, "Davion Mitchell", "G", "MIA")])
    assert resolve_cell("D. Mitchell", "G", "CLE", board)["dk_player_id"].tolist() == [1]
    assert resolve_cell("D. Mitchell", "G", "MIA", board)["dk_player_id"].tolist() == [2]
    assert len(resolve_cell("D. Mitchell", "F", "CLE", board)) == 0   # pos misread → loud
    board2 = _board([(3, "Zion Williamson", "F", "NOP")])
    assert resolve_cell("Z. Williamson", "F", "NO", board2)["dk_player_id"].tolist() == [3]


# ── a full synthetic board on disk ────────────────────────────────────────────
# Positions by round (7G/6F/3C for every seat) keep the caps non-binding, and picks run
# in exact ranking order, so the board is a perfect ADP-null pod: validation must be
# green and the autodraft soft check must agree 16/16.

_ROUND_POS = ["G"] * 7 + ["F"] * 6 + ["C"] * 3


def _synthetic_pod(tmp_path):
    first = ["Alpha", "Bravo", "Carlos", "Delta", "Echo", "Fabio", "Golf", "Hotel",
             "India", "Julie", "Kilos", "Limas"]
    players, picks = [], []
    for overall in range(1, N_PICKS + 1):
        rnd = (overall - 1) // N_SEATS + 1
        pick = overall - (rnd - 1) * N_SEATS
        seat = pick if rnd % 2 == 1 else N_SEATS + 1 - pick
        name = f"{first[overall % 12]} Fake{overall:03d}"
        pos = _ROUND_POS[rnd - 1]
        team = f"TM{overall % 30:02d}"
        players.append((1000 + overall, name, pos, overall, team))
        picks.append((seat, rnd, pick, overall,
                      f"{name.split()[0][0]}. {name.split()[1]}", pos, team))
    rankings = tmp_path / "rankings"
    rankings.mkdir()
    pd.DataFrame(players, columns=["ID", "Name", "Position", "ADP", "Team"]) \
        .to_csv(rankings / "DkPreDraftRankings_Jan1_2000.csv", index=False)

    boards = tmp_path / "adp_autodrafted_boards_2000"
    boards.mkdir()
    img = boards / "draftboard_Jan1_2000.png"
    img.touch()
    header = pd.DataFrame({
        "seat": range(1, N_SEATS + 1),
        "entrant": ["sunshinere..." if s == 3 else f"drafter{s:02d}"
                    for s in range(1, N_SEATS + 1)],
        "g": 7, "f": 6, "c": 3})
    header.to_csv(boards / "draftboard_Jan1_2000_header.csv", index=False)
    pd.DataFrame(picks, columns=["seat", "round", "pick_in_round", "overall",
                                 "cell_name", "pos", "team"]) \
        .to_csv(boards / "draftboard_Jan1_2000_picks.csv", index=False)
    manifest = pd.DataFrame([{
        "file": img.name, "contest": "Synthetic Pod", "our_entrant": "sunshinerecorder",
        "mode": "auto", "ranking_file": "DkPreDraftRankings_Jan1_2000.csv", "notes": ""}])
    return img, manifest, rankings, boards


def test_synthetic_pod_validates_green(tmp_path):
    img, manifest, rankings, _ = _synthetic_pod(tmp_path)
    log, errors, notes = validate_board(img, manifest, rankings)
    assert errors == []
    assert len(log) == N_PICKS
    assert log["overall"].tolist() == list(range(1, N_PICKS + 1))
    assert log["our_seat"].sum() == N_ROUNDS
    assert set(log.loc[log["our_seat"], "seat"]) == {3}
    assert log["dk_player_id"].is_unique
    assert log["season"].iloc[0] == "1999-00"
    assert any("16/16 agree" in n for n in notes)


def test_wrong_position_fails_match_and_checksum(tmp_path):
    img, manifest, rankings, boards = _synthetic_pod(tmp_path)
    picks_path = boards / "draftboard_Jan1_2000_picks.csv"
    picks = pd.read_csv(picks_path)
    picks.loc[picks["overall"] == 168, "pos"] = "F"   # cell 14.1 was a C
    picks.to_csv(picks_path, index=False)
    log, errors, _ = validate_board(img, manifest, rankings)
    assert log is None
    assert any("cell 14.1" in e and "matches NO ONE" in e for e in errors)
    assert any("column G/F/C" in e for e in errors)


def test_duplicate_player_fails(tmp_path):
    img, manifest, rankings, boards = _synthetic_pod(tmp_path)
    picks_path = boards / "draftboard_Jan1_2000_picks.csv"
    picks = pd.read_csv(picks_path)
    donor = picks.loc[picks["overall"] == 1]
    for col in ("cell_name", "pos", "team"):
        # overall 13 is the same seat (round 2 snake) and same G round → checksum holds
        picks.loc[picks["overall"] == 13, col] = donor[col].values[0]
    picks.to_csv(picks_path, index=False)
    _, errors, _ = validate_board(img, manifest, rankings)
    assert any("drafted twice" in e for e in errors)


def test_broken_overall_fails_both_identities(tmp_path):
    img, manifest, rankings, boards = _synthetic_pod(tmp_path)
    picks_path = boards / "draftboard_Jan1_2000_picks.csv"
    picks = pd.read_csv(picks_path)
    picks.loc[picks["overall"] == 100, "overall"] = 99   # digit misread
    picks.to_csv(picks_path, index=False)
    _, errors, _ = validate_board(img, manifest, rankings)
    assert any("not a permutation" in e for e in errors)
    assert any("overall 99 !=" in e for e in errors)


def test_missing_manifest_row_fails(tmp_path):
    img, manifest, rankings, _ = _synthetic_pod(tmp_path)
    _, errors, _ = validate_board(img, manifest.iloc[:0], rankings)
    assert any("boards_manifest" in e for e in errors)


def test_autodraft_agreement_flags_a_reach(tmp_path):
    img, manifest, rankings, boards = _synthetic_pod(tmp_path)
    picks = pd.read_csv(boards / "draftboard_Jan1_2000_picks.csv")
    swap = picks["overall"].isin([3, 4])   # seat 3's first pick swapped with seat 4's
    a, b = picks.loc[picks["overall"] == 3].index[0], picks.loc[picks["overall"] == 4].index[0]
    for col in ("cell_name", "pos", "team"):
        picks.loc[a, col], picks.loc[b, col] = picks.loc[b, col], picks.loc[a, col]
    picks.to_csv(boards / "draftboard_Jan1_2000_picks.csv", index=False)
    log, errors, notes = validate_board(img, manifest, rankings)
    assert errors == []
    assert any("15/16 agree" in n for n in notes)


# ── grid detection and tiling on a synthesized image ──────────────────────────

def test_detect_grid_and_tile(tmp_path):
    img = Image.new("RGB", (1220, 1439), (10, 10, 10))
    xs = [4 + round(i * 1210 / 12) for i in range(13)]
    ys = [213 + round(j * 1220 / 16) for j in range(17)]
    px = img.load()
    for x in range(1220):          # the entrant header band above the grid
        for y in range(100, 208):
            px[x, y] = (60, 60, 60)
    for c in range(12):
        for r in range(16):
            for x in range(xs[c] + 3, xs[c + 1] - 2):
                for y in range(ys[r] + 3, ys[r + 1] - 2):
                    px[x, y] = (200, 120, 120)
    dest = tmp_path / "draftboard_Feb2_2001.png"
    img.save(dest)
    gx, gy = detect_grid(Image.open(dest))
    assert len(gx) == 13 and len(gy) == 17
    assert all(abs(a - b) <= 2 for a, b in zip(gx, xs))
    assert all(abs(a - b) <= 2 for a, b in zip(gy, ys))
    out = tile_board(dest, out_root=tmp_path / "tiles")
    names = sorted(p.name for p in out.iterdir())
    assert "header.png" in names and "rows_01_04.png" in names
    assert sum(n.startswith("col_") for n in names) == 24
    cell = cell_crop(dest, 13, 1, out_root=tmp_path / "tiles")
    assert cell.exists()
