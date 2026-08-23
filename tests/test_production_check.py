"""`make production-check` — the readiness readout for a season nothing has seen.

The properties worth pinning are the ones that would make the check *lie*: reporting a
crunch step as done because the artifact file exists, deriving the target season from a
calendar instead of from the board, or treating an unplayed season's missing rows as a
defect rather than as the schedule.
"""

from pathlib import Path

import pandas as pd

from src import production_check


def _cfg(tmp_path: Path) -> dict:
    return {"data": {"features_dir": str(tmp_path / "features"),
                     "raw_dir": str(tmp_path / "raw")},
            "evaluation": {"predictions_dir": str(tmp_path / "predictions")}}


def _pool(tmp_path: Path, seasons: list[str]) -> Path:
    directory = Path(tmp_path / "features")
    directory.mkdir(parents=True, exist_ok=True)
    dest = directory / "draft_pool.parquet"
    pd.DataFrame({"season": seasons, "player_id": range(len(seasons))}).to_parquet(dest)
    return dest


def test_the_target_season_comes_from_the_board_not_the_clock(tmp_path):
    """A DK board exists for exactly one unplayed season at a time, and that is the season
    the production fit is for. Deriving it from today's date would be wrong for the two
    months a year when two boards are up, and would disagree with every other module."""
    cfg = _cfg(tmp_path)
    _pool(tmp_path, ["2024-25", "2025-26", "2026-27"])
    assert production_check.next_season(cfg) == "2026-27"


def test_a_season_with_no_rows_is_missing_even_though_the_file_exists(tmp_path):
    """The check that would otherwise be useless.

    Every per-season artifact here already exists on disk, covering seasons that have been
    played. A file-level check would report the entire October crunch as complete on a
    season with no schedule, no roster and no game log.
    """
    cfg = _cfg(tmp_path)
    features = Path(tmp_path / "features")
    features.mkdir(parents=True, exist_ok=True)
    for name in ("scoring_periods", "availability_panel", "preseason",
                 "component_targets", "draft_pool", "adp_panel"):
        pd.DataFrame({"season": ["2025-26"] * 3}).to_parquet(features / f"{name}.parquet")

    rows = production_check.season_rows(cfg, "2026-27")
    played = production_check.season_rows(cfg, "2025-26")

    assert set(rows["state"]) == {"missing"}, "an unplayed season cannot be ready"
    assert (played["item"].str.endswith(".parquet")
            & (played["state"] == "ready")).sum() == 6
    # Every missing row names the target that would fix it, so the crunch is a checklist.
    assert rows["target"].str.startswith("make").all()


def test_the_inputs_are_listed_in_dependency_order(tmp_path):
    """The first missing row is the one to work on, which is only true if the list is
    ordered — the panel cannot be built before the fetch, the preseason panel cannot be
    built before the preseason is played."""
    items = [name for _, name, _, _ in production_check.SEASON_INPUTS]
    assert production_check.SEASON_INPUTS[0][0] == "raw_dir"
    assert items.index("scoring_periods.parquet") < \
        items.index("availability_panel.parquet")
    assert items.index("preseason.parquet") < items.index("component_targets.parquet")


def test_a_missing_production_fit_lists_every_head_the_train_window_carries(tmp_path):
    """Reported per head rather than as one line, because the normal failure is a partial
    `--groups` run rather than nothing at all."""
    cfg = _cfg(tmp_path)
    from src.models.posteriors import posteriors_dir

    directory = posteriors_dir(cfg, "train")
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"head": ["ast", "reb", "composition"], "variant": "v",
                  "n_fit_rows": 1, "first_season": "1997-98",
                  "last_season": "2021-22"}).to_csv(directory / "manifest.csv",
                                                    index=False)
    rows, note = production_check.model_rows(cfg)
    assert set(rows["item"]) == {"ast", "reb", "composition"}
    assert set(rows["state"]) == {"missing"}
    assert "posteriors-production" in note


def test_capture_is_reported_and_never_gated():
    """There is no freshness threshold that is right in both August and October — the ADP
    boards freeze between draft seasons and the injury feeds run nightly in-season — so
    the check prints the date and lets a person read it."""
    rows = production_check.capture_rows()
    assert set(rows["state"]) <= {"reported", "missing"}
    assert "ready" not in set(rows["state"])


def test_a_raw_csv_input_is_not_read_as_parquet(tmp_path):
    """🔴 The regression this exists for. `SEASON_INPUTS` mixes raw CSVs with feature
    parquets, and the reader tried `read_parquet` on all of them — so the check crashed the
    moment `team_rosters_2026_27.csv` was fetched. The original fixture created only the
    parquets, so every raw row took the `missing` branch and the bug never fired.
    """
    cfg = _cfg(tmp_path)
    raw = Path(cfg["data"]["raw_dir"]) / "nbastats"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "team_rosters_2026_27.csv").write_text("PLAYER_ID,TeamID\n1,10\n2,10\n")
    (raw / "schedule_2026_27.csv").write_text("game_id_raw\n")     # header only

    rows = production_check.season_rows(cfg, "2026-27").set_index("item")
    roster = rows.loc[str(raw / "team_rosters_2026_27.csv")]
    assert roster["state"] == "ready" and roster["detail"] == "2 rows"
    # A header with no data rows is not a fetch — same rule `fetch._skip_or_fetch` applies.
    assert rows.loc[str(raw / "schedule_2026_27.csv"), "state"] == "missing"


def test_the_tensor_row_reads_the_same_staleness_guard_the_room_does(tmp_path):
    """The readiness list used to end one step short of the artifact the draft consumes.

    Three states, because for this artifact "exists" is not "current": missing before the
    build, ready-as-REHEARSAL while no preseason exists anywhere, and STALE the moment
    the October log lands beside a tensor stamped as built without it — read through
    `season.assert_tensor_current` itself, so the checklist and the load-time refusal
    cannot disagree.
    """
    import numpy as np

    cfg = _cfg(tmp_path)
    features = Path(cfg["data"]["features_dir"])
    features.mkdir(parents=True, exist_ok=True)

    row = production_check.tensor_row(cfg, "2026-27").iloc[0]
    assert row["state"] == "missing"
    assert row["target"] == "make simulate-production"

    np.savez(features / "sim_tensor_2026-27.npz",
             season=np.array("2026-27"), fit_window=np.array("full"),
             n_sims=np.array(4), preseason_coverage=np.array(0.0),
             preseason_log_rows=np.array(0))
    row = production_check.tensor_row(cfg, "2026-27").iloc[0]
    assert row["state"] == "ready" and "REHEARSAL" in row["detail"]

    raw = Path(cfg["data"]["raw_dir"]) / "nbastats"
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "game_logs_pre_season_2026_27.csv").write_text(
        "GAME_ID,PLAYER_ID,MIN\n1,7,12.0\n")
    row = production_check.tensor_row(cfg, "2026-27").iloc[0]
    assert row["state"] == "stale"
    assert "simulate-production" in row["target"]
