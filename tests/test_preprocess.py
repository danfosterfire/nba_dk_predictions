import pandas as pd
import pytest
from src.data.preprocess import (
    ALL_SEASON_TYPES,
    FULL_WINDOW,
    PLAYOFFS,
    REGULAR_SEASON,
    TEST_SEASONS,
    TRAIN_VAL_WINDOW,
    _parse_log_filename,
    clean,
    fit_window,
    held_out_seasons,
    load_raw,
)


def _make_df(n_games: int = 30, player_id: int = 1) -> pd.DataFrame:
    return pd.DataFrame({
        "PLAYER_ID": [player_id] * n_games,
        "PLAYER_NAME": ["Test Player"] * n_games,
        "TEAM_ABBREVIATION": ["LAL"] * n_games,
        "GAME_ID": range(n_games),
        "GAME_DATE": pd.date_range("2024-01-01", periods=n_games, freq="2D").strftime("%Y-%m-%d"),
        "MATCHUP": ["LAL vs. GSW"] * n_games,
        "WL": ["W"] * n_games,
        "MIN": [30.0] * n_games,
        "PTS": [20.0] * n_games,
        "REB": [5.0] * n_games,
        "AST": [4.0] * n_games,
        "STL": [1.0] * n_games,
        "BLK": [0.5] * n_games,
        "TOV": [2.0] * n_games,
        "FGA": [15.0] * n_games,
        "FGM": [8.0] * n_games,
        "FG3A": [5.0] * n_games,
        "FG3M": [2.0] * n_games,
        "FTA": [4.0] * n_games,
        "FTM": [3.0] * n_games,
        "PLUS_MINUS": [5.0] * n_games,
    })


def test_clean_keeps_expected_columns():
    df = clean(_make_df())
    assert "pts" in df.columns
    assert "game_date" in df.columns
    assert "home" in df.columns


def test_clean_filters_by_min_games():
    df_few = _make_df(n_games=5, player_id=1)
    df_many = _make_df(n_games=30, player_id=2)
    combined = pd.concat([df_few, df_many], ignore_index=True)
    result = clean(combined, min_games=20)
    assert set(result["player_id"].unique()) == {2}


def test_clean_parses_home_flag():
    df = _make_df()
    df["MATCHUP"] = ["LAL vs. GSW"] * 15 + ["LAL @ GSW"] * 15
    result = clean(df, min_games=1)
    assert result["home"].isin([0, 1]).all()


# ── Season type: playoff logs must not become pseudo-seasons ──────────────────

def _write_logs(tmp_path, n_games: int = 30):
    """A regular-season and a playoff log for the same season, as fetch writes them."""
    for name in ("game_logs_2021_22.csv", "game_logs_playoffs_2021_22.csv"):
        _make_df(n_games=n_games).to_csv(tmp_path / name, index=False)


def test_filename_parsing_strips_the_playoffs_prefix():
    assert _parse_log_filename("game_logs_2021_22") == (REGULAR_SEASON, "2021-22")
    assert _parse_log_filename("game_logs_playoffs_2021_22") == (PLAYOFFS, "2021-22")


def test_load_raw_defaults_to_regular_season_only(tmp_path):
    """The default must not pull playoff rows in unannounced — that is the bug."""
    _write_logs(tmp_path)
    df = load_raw(tmp_path)
    assert set(df["season"]) == {"2021-22"}
    assert set(df["season_type"]) == {REGULAR_SEASON}
    assert len(df) == 30


def test_playoff_rows_never_become_a_pseudo_season(tmp_path):
    """`playoffs-2021-22` doubled the season count for every groupby on season."""
    _write_logs(tmp_path)
    df = load_raw(tmp_path, season_type=ALL_SEASON_TYPES)
    assert set(df["season"]) == {"2021-22"}
    assert not any("playoff" in s for s in df["season"])
    assert df["season_type"].value_counts().to_dict() == {REGULAR_SEASON: 30, PLAYOFFS: 30}


def test_playoffs_are_reachable_explicitly_for_workload_features(tmp_path):
    _write_logs(tmp_path)
    df = load_raw(tmp_path, season_type=PLAYOFFS)
    assert set(df["season_type"]) == {PLAYOFFS}
    assert set(df["season"]) == {"2021-22"}


def test_unknown_season_type_raises_rather_than_returning_nothing(tmp_path):
    _write_logs(tmp_path)
    with pytest.raises(ValueError, match="season_type"):
        load_raw(tmp_path, season_type="preseason")


def test_missing_files_for_a_valid_season_type_raise(tmp_path):
    _make_df().to_csv(tmp_path / "game_logs_2021_22.csv", index=False)
    with pytest.raises(FileNotFoundError, match="playoffs"):
        load_raw(tmp_path, season_type=PLAYOFFS)


def test_columns_argument_narrows_the_read(tmp_path):
    _write_logs(tmp_path)
    df = load_raw(tmp_path, columns=["GAME_ID", "TEAM_ABBREVIATION", "MIN"])
    assert set(df.columns) == {"GAME_ID", "TEAM_ABBREVIATION", "MIN",
                               "season", "season_type"}


def test_clean_carries_season_type_through(tmp_path):
    _write_logs(tmp_path)
    out = clean(load_raw(tmp_path, season_type=ALL_SEASON_TYPES), min_games=10)
    assert "season_type" in out.columns
    assert set(out["season_type"]) == {REGULAR_SEASON, PLAYOFFS}


# ── The fit window, and the split constant it has to agree with ──────────────

def _seasons(labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"season": labels, "value": range(len(labels))})


def test_fit_window_drops_the_trailing_seasons():
    frame = _seasons(["2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])
    kept = fit_window(frame, TRAIN_VAL_WINDOW)
    assert sorted(kept["season"]) == ["2021-22", "2022-23", "2023-24"]
    assert held_out_seasons(frame) == ["2024-25", "2025-26"]


def test_full_window_is_the_identity():
    frame = _seasons(["2021-22", "2022-23", "2023-24"])
    assert len(fit_window(frame, FULL_WINDOW)) == len(frame)


def test_fit_window_rejects_an_unknown_window():
    with pytest.raises(ValueError, match="unknown fit window"):
        fit_window(_seasons(["2021-22", "2022-23"]), "trian_val")


def test_fit_window_derives_the_held_out_seasons_rather_than_hard_coding_them():
    """Out-of-order labels and a shorter panel must both still hold out the LAST two."""
    frame = _seasons(["2023-24", "1999-00", "2001-02"])
    assert sorted(fit_window(frame, TRAIN_VAL_WINDOW)["season"]) == ["1999-00"]


def test_test_seasons_agrees_with_every_model_module_that_defines_its_own():
    """Two definitions of "which seasons are held out" that can disagree is worse than
    a duplicated constant, because the disagreement is silent on both sides: the
    calibration frames would exclude a different set of seasons from the one the heads
    actually hold out, and nothing downstream would raise.
    """
    from src.models.availability import TEST_SEASONS as availability_test_seasons
    from src.models.component_rates import TEST_SEASONS as rates_test_seasons
    from src.models.stan_composition import TEST_SEASONS as composition_test_seasons

    assert TEST_SEASONS == availability_test_seasons
    assert TEST_SEASONS == rates_test_seasons
    assert TEST_SEASONS == composition_test_seasons


def test_fit_window_holds_out_the_same_seasons_as_split_seasons():
    """The two helpers key on different frames — a calibration frame against a design
    matrix — so the invariant worth pinning is that they name the same seasons.

    `split_seasons` comes from `models.availability` rather than `component_rates`, which
    used to keep a private copy of it and was the one head the held-out lock could not
    see. There is one definition now.
    """
    from src.models.availability import split_seasons

    frame = _seasons(["2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26"])
    _, held = split_seasons(frame)
    assert sorted(set(held["season"])) == held_out_seasons(frame)
    assert set(fit_window(frame, TRAIN_VAL_WINDOW)["season"]).isdisjoint(held["season"])
