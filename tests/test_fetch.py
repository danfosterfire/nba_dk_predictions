import pandas as pd
from src.data.fetch import PLAYER_STAT_MEASURES, _record_fetch, _skip_or_fetch


HEADER = "PLAYER_ID,PLAYER_NAME,GP,MIN,DEF_RATING\n"


def test_skip_or_fetch_skips_file_with_rows(tmp_path):
    dest = tmp_path / "player_stats_defense_2021_22.csv"
    dest.write_text(HEADER + "201939,Stephen Curry,64,34.5,109.8\n")
    assert _skip_or_fetch(dest, "test") is True


def test_skip_or_fetch_refetches_header_only_file(tmp_path):
    """The bug that let five empty defense files survive every re-run."""
    dest = tmp_path / "player_stats_defense_2023_24.csv"
    dest.write_text(HEADER)
    assert _skip_or_fetch(dest, "test") is False


def test_skip_or_fetch_refetches_empty_and_missing_files(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_text("")
    assert _skip_or_fetch(empty, "test") is False
    assert _skip_or_fetch(tmp_path / "absent.csv", "test") is False


def test_skip_or_fetch_honors_two_row_header(tmp_path):
    """Shot locations write a MultiIndex header; both rows are header, not data."""
    dest = tmp_path / "player_shot_locations_2023_24.csv"
    dest.write_text(",,Restricted Area,Restricted Area\n,,FGM,FGA\n")
    assert _skip_or_fetch(dest, "test", header_rows=2) is False

    dest.write_text(",,Restricted Area,Restricted Area\n,,FGM,FGA\n201939,Stephen Curry,1.2,2.4\n")
    assert _skip_or_fetch(dest, "test", header_rows=2) is True


def test_four_factors_removed_from_player_measures():
    """Unsupported for players — the endpoint omits the result set entirely."""
    assert "Four Factors" not in PLAYER_STAT_MEASURES


def test_record_fetch_upserts_by_family_and_season(tmp_path):
    _record_fetch(tmp_path, "player_stats_defense", "2023-24", "PerGame", 0)
    _record_fetch(tmp_path, "player_stats_defense", "2023-24", "Totals", 572)
    _record_fetch(tmp_path, "player_stats_defense", "2015-16", "PerGame", 476)

    manifest = pd.read_csv(tmp_path / "_fetch_manifest.csv")
    assert len(manifest) == 2
    row = manifest[manifest["season"] == "2023-24"].iloc[0]
    assert row["per_mode"] == "Totals"
    assert row["rows"] == 572
