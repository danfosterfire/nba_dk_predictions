import pandas as pd
from src.data.fetch import nbastats_dir
from src.eda.season_matrix import (
    _is_rate,
    _read_shot_locations,
    build_tier,
    feature_cols,
    per36_divisor,
    read_family,
    to_per36,
    FAMILIES,
)


# ── Synthetic raw-directory builders ──────────────────────────────────────────

def _slugged(season: str) -> str:
    return season.replace("-", "_")


def _base_csv(raw_dir, season: str, players: list[dict]) -> None:
    """Write a minimal player_stats_base file — the identity + qualification source."""
    rows = []
    for i, p in enumerate(players):
        rows.append({
            "PLAYER_ID": p["id"], "PLAYER_NAME": p["name"], "NICKNAME": p["name"].split()[0],
            "TEAM_ID": 1610612747, "TEAM_ABBREVIATION": "LAL", "AGE": p.get("age", 27.0),
            "GP": p["gp"], "W": 40, "L": 20, "W_PCT": 0.667, "MIN": p["min"],
            "FGM": 8.0, "FGA": 15.0, "FG_PCT": 0.533, "FG3M": 2.0, "FG3A": 5.0, "FG3_PCT": 0.4,
            "FTM": 3.0, "FTA": 4.0, "FT_PCT": 0.75, "OREB": 1.0, "DREB": 4.0, "REB": p.get("reb", 5.0),
            "AST": p.get("ast", 4.0), "TOV": 2.0, "STL": 1.0, "BLK": 0.5, "BLKA": 0.4,
            "PF": 2.0, "PFD": 2.5, "PTS": p.get("pts", 20.0), "PLUS_MINUS": 3.0,
            "NBA_FANTASY_PTS": 33.0, "DD2": 5, "TD3": 0, "WNBA_FANTASY_PTS": 33.0,
            "TEAM_COUNT": 1, "GP_RANK": i + 1, "MIN_RANK": i + 1,
        })
    pd.DataFrame(rows).to_csv(
        nbastats_dir(raw_dir) / f"player_stats_base_{_slugged(season)}.csv", index=False)


def _game_logs_csv(raw_dir, season: str, players: list[dict]) -> None:
    rows = []
    for p in players:
        for g in range(p["gp"]):
            rows.append({
                "SEASON_YEAR": season, "PLAYER_ID": p["id"], "PLAYER_NAME": p["name"],
                "TEAM_ABBREVIATION": "LAL", "GAME_ID": f"{p['id']}{g:04d}", "MIN": p["min"],
                "PTS": p.get("pts", 20.0), "REB": p.get("reb", 5.0), "AST": p.get("ast", 4.0),
                "STL": 1.0, "BLK": 0.0, "TOV": 2.0, "FG3M": 2.0,
            })
    pd.DataFrame(rows).to_csv(
        nbastats_dir(raw_dir) / f"game_logs_{_slugged(season)}.csv", index=False)


def _tracking_csv(raw_dir, season: str, players: list[dict]) -> None:
    """A Tier-B-only family, so Tier B gains columns Tier A does not have."""
    rows = [{
        "PLAYER_ID": p["id"], "PLAYER_NAME": p["name"], "TEAM_ID": 1610612747,
        "TEAM_ABBREVIATION": "LAL", "GP": p["gp"], "W": 40, "L": 20, "MIN": p["min"],
        "DRIVES": 10.0, "DRIVE_FGM": 3.0, "DRIVE_FGA": 6.0, "DRIVE_FG_PCT": 0.5,
    } for p in players]
    pd.DataFrame(rows).to_csv(
        nbastats_dir(raw_dir) / f"player_tracking_drives_{_slugged(season)}.csv", index=False)


def _make_raw(tmp_path, seasons: list[str], players: list[dict], tracking_from: int = 2013):
    raw = tmp_path / "raw"
    nbastats_dir(raw).mkdir(parents=True)
    for s in seasons:
        _base_csv(raw, s, players)
        _game_logs_csv(raw, s, players)
        if int(s.split("-")[0]) >= tracking_from:
            _tracking_csv(raw, s, players)
    return raw


PLAYERS = [
    {"id": 101, "name": "Qualified Starter", "gp": 60, "min": 30.0},
    {"id": 102, "name": "Qualified Rotation", "gp": 40, "min": 18.0},
]


# ── Per-36 normalization ──────────────────────────────────────────────────────

def test_per36_pergame_is_stat_over_min_times_36():
    df = pd.DataFrame({"AST": [9.0], "REB": [12.4]})
    to_per36(df, ["AST", "REB"], pd.Series([34.6]), "PerGame")
    assert abs(df["AST"].iloc[0] - 9.0 / 34.6 * 36) < 1e-9
    assert abs(df["REB"].iloc[0] - 12.4 / 34.6 * 36) < 1e-9


def test_per36_totals_matches_pergame_for_same_season():
    """The load-bearing invariant for the backfilled Defense seasons.

    MIN always carries the same basis as the stats beside it, so one expression
    covers both modes — a season fetched as Totals must land on the same per-36
    value as the same season fetched PerGame.
    """
    gp, min_pg, stl_pg = 79, 34.6, 1.37
    per_game = pd.DataFrame({"STL": [stl_pg]})
    totals = pd.DataFrame({"STL": [stl_pg * gp]})

    to_per36(per_game, ["STL"], pd.Series([min_pg]), "PerGame")
    to_per36(totals, ["STL"], pd.Series([min_pg * gp]), "Totals")
    assert abs(per_game["STL"].iloc[0] - totals["STL"].iloc[0]) < 1e-9


def test_per36_perminute_is_a_plain_scale_up():
    df = pd.DataFrame({"STL": [0.0315]})
    to_per36(df, ["STL"], pd.Series([3484.1]), "PerMinute")
    assert abs(df["STL"].iloc[0] - 0.0315 * 36) < 1e-9


def test_per36_leaves_rates_and_physicals_untouched():
    df = pd.DataFrame({"FG_PCT": [0.583], "DEF_RATING": [110.0], "PLAYER_HEIGHT_INCHES": [83.0]})
    to_per36(df, list(df.columns), pd.Series([34.6]), "PerGame")
    assert df["FG_PCT"].iloc[0] == 0.583
    assert df["DEF_RATING"].iloc[0] == 110.0
    assert df["PLAYER_HEIGHT_INCHES"].iloc[0] == 83.0


def test_per36_divisor_rejects_modes_with_no_minutes_basis():
    assert per36_divisor("PerGame") == "min"
    assert per36_divisor("Totals") == "min"
    assert per36_divisor("PerMinute") == "scale"
    try:
        per36_divisor("Per100Possessions")
        raise AssertionError("expected ValueError for Per100Possessions")
    except ValueError as exc:
        assert "per-36" in str(exc)


def test_rate_classification_of_tricky_columns():
    # counting stats that must be scaled
    for c in ["DEF_WS", "DEF_WS_RAW", "TIME_OF_POSS", "DIST_MILES", "POSS", "DD2", "PLUS_MINUS"]:
        assert not _is_rate(c), c
    # already-normalized columns that must not be
    for c in ["FG_PCT", "PCT_DREB", "DEF_RATING", "AST_RATIO", "E_PACE", "PIE",
              "AVG_SPEED", "PTS_PER_TOUCH", "FGA_FREQUENCY"]:
        assert _is_rate(c), c
    # physical / draft attributes are neither
    for c in ["PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT", "DRAFT_NUMBER"]:
        assert _is_rate(c), c


# ── Shot-location MultiIndex ──────────────────────────────────────────────────

def test_shot_locations_multiindex_flattening(tmp_path):
    path = tmp_path / "player_shot_locations_2023_24.csv"
    path.write_text(
        ",,Restricted Area,Restricted Area,Restricted Area,In The Paint (Non-RA),Above the Break 3\n"
        "PLAYER_ID,PLAYER_NAME,FGM,FGA,FG_PCT,FGA,FGA\n"
        "101,Test Player,4.0,6.0,0.667,1.5,3.0\n"
    )
    df = _read_shot_locations(path)
    assert list(df.columns) == [
        "PLAYER_ID", "PLAYER_NAME",
        "RESTRICTED_AREA_FGM", "RESTRICTED_AREA_FGA", "RESTRICTED_AREA_FG_PCT",
        "IN_THE_PAINT_NON_RA_FGA", "ABOVE_THE_BREAK_3_FGA",
    ]


def test_shot_location_columns_land_lowercased_and_prefixed(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    (nbastats_dir(raw) / "player_shot_locations_2023_24.csv").write_text(
        ",,Restricted Area\nPLAYER_ID,PLAYER_NAME,FGA\n101,Qualified Starter,6.0\n"
    )
    matrix, _ = build_tier("A", ["2023-24"], raw)
    assert "sl_restricted_area_fga" in matrix.columns


# ── Column hygiene ───────────────────────────────────────────────────────────

def test_read_family_drops_ranks_and_bookkeeping(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    df = read_family(FAMILIES[0], "2023-24", raw)
    for c in df.columns:
        assert not c.endswith("_RANK")
        assert c not in ("TEAM_COUNT", "NICKNAME", "GROUP_SET", "WNBA_FANTASY_PTS")
    # identity columns are carried once from read_identity, not per family
    for c in ("PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP", "W", "L", "W_PCT"):
        assert c not in df.columns


def test_no_rank_or_bookkeeping_columns_survive_into_matrix(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    matrix, _ = build_tier("A", ["2023-24"], raw)
    for c in matrix.columns:
        assert "rank" not in c
        assert c not in ("team_count", "nickname", "group_set")
    # GP/AGE must not appear as per-36'd feature columns
    assert "bas_gp" not in matrix.columns
    assert "bas_age" not in matrix.columns


# ── Assembly ─────────────────────────────────────────────────────────────────

def test_one_row_per_player_season(tmp_path):
    raw = _make_raw(tmp_path, ["2022-23", "2023-24"], PLAYERS)
    matrix, _ = build_tier("A", ["2022-23", "2023-24"], raw)
    assert not matrix.duplicated(["player_id", "season"]).any()
    assert len(matrix) == 4


def test_qualification_filter_excludes_low_gp_and_low_minutes(tmp_path):
    players = PLAYERS + [
        {"id": 103, "name": "Five Game Callup", "gp": 5, "min": 22.0},
        {"id": 104, "name": "Deep Bench", "gp": 55, "min": 4.0},
    ]
    raw = _make_raw(tmp_path, ["2023-24"], players)
    matrix, _ = build_tier("A", ["2023-24"], raw, min_gp=20, min_minutes=10.0)
    assert set(matrix["player_id"]) == {101, 102}


def test_tier_b_starts_in_2013_and_supersets_tier_a_columns(tmp_path):
    seasons = ["2011-12", "2012-13", "2013-14", "2014-15"]
    raw = _make_raw(tmp_path, seasons, PLAYERS)

    a, _ = build_tier("A", seasons, raw)
    b, _ = build_tier("B", seasons, raw)

    assert sorted(a["season"].unique()) == seasons
    assert sorted(b["season"].unique()) == ["2013-14", "2014-15"]
    assert set(a.columns) <= set(b.columns)
    # the tracking family is what Tier B adds
    assert "drv_drives" in b.columns and "drv_drives" not in a.columns


def test_volume_and_target_columns_held_out_of_features(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    matrix, _ = build_tier("A", ["2023-24"], raw)
    feats = feature_cols(matrix)
    for c in ["gp", "min", "min_total", "age", "player_id", "season_start_year",
              "dk_pts_total", "dk_pts_per_game", "dk_pts_std", "games_played"]:
        assert c not in feats
    assert "bas_pts" in feats


def test_all_stat_columns_are_float_not_object(tmp_path):
    """Object-dtype stat columns fall out of `feature_cols` without any error.

    Zero minutes is real (a player with no clutch time), and dividing by a
    pd.NA-holding Series promotes the whole column to object.
    """
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    pd.DataFrame([
        {"PLAYER_ID": 101, "PLAYER_NAME": "Qualified Starter", "GROUP_SET": "Overall",
         "MIN": 2.4, "PTS": 3.0, "FG_PCT": 0.5},
        {"PLAYER_ID": 102, "PLAYER_NAME": "Qualified Rotation", "GROUP_SET": "Overall",
         "MIN": 0.0, "PTS": 0.0, "FG_PCT": 0.0},
    ]).to_csv(nbastats_dir(raw) / "player_clutch_2023_24.csv", index=False)

    matrix, _ = build_tier("A", ["2023-24"], raw)
    assert matrix["clu_pts"].dtype == float
    assert "clu_pts" in feature_cols(matrix)
    # zero clutch minutes yields no rate, not an inf
    zero_min = matrix[matrix["player_id"] == 102].iloc[0]
    assert pd.isna(zero_min["clu_pts"])


def test_target_join_matches_compute_dk_pts(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    matrix, _ = build_tier("A", ["2023-24"], raw)
    row = matrix[matrix["player_id"] == 101].iloc[0]
    # pts 20 + fg3m 2*0.5 + reb 5*1.25 + ast 4*1.5 + stl 1*2 + blk 0 + tov 2*-0.5
    # one double-digit category (pts) so no double-double bonus
    expected = 20 + 1 + 6.25 + 6 + 2 - 1
    assert abs(row["dk_pts_per_game"] - expected) < 1e-9
    assert abs(row["dk_pts_total"] - expected * 60) < 1e-6
    assert row["games_played"] == 60


def test_backfilled_per_mode_from_manifest_is_applied(tmp_path):
    """A Totals-mode Defense file must normalize onto the same scale as PerGame."""
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    gp, min_pg, stl_pg = 60, 30.0, 1.2
    pd.DataFrame([{
        "PLAYER_ID": 101, "PLAYER_NAME": "Qualified Starter", "TEAM_ID": 1, "NICKNAME": "Q",
        "TEAM_ABBREVIATION": "LAL", "AGE": 27.0, "GP": gp, "W": 40, "L": 20, "W_PCT": 0.667,
        "MIN": min_pg * gp, "DEF_RATING": 110.0, "STL": stl_pg * gp, "DEF_WS": 0.07 * gp,
    }]).to_csv(nbastats_dir(raw) / "player_stats_defense_2023_24.csv", index=False)
    pd.DataFrame([{"family": "player_stats_defense", "season": "2023-24",
                   "per_mode": "Totals", "rows": 1}]).to_csv(raw / "_fetch_manifest.csv", index=False)

    matrix, _ = build_tier("A", ["2023-24"], raw)
    row = matrix[matrix["player_id"] == 101].iloc[0]
    assert abs(row["def_stl"] - stl_pg / min_pg * 36) < 1e-9
    assert row["def_def_rating"] == 110.0     # rate passes through unscaled


def test_coverage_report_marks_missing_families(tmp_path):
    raw = _make_raw(tmp_path, ["2023-24"], PLAYERS)
    _, coverage = build_tier("A", ["2023-24"], raw)
    base = coverage[coverage["family"] == "player_stats_base"].iloc[0]
    assert bool(base["available"]) and base["rows"] == 2 and base["matched"] == 2
    missing = coverage[coverage["family"] == "player_clutch"].iloc[0]
    assert not bool(missing["available"]) and missing["rows"] == 0
