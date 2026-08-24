"""`make production-check` — is the chain ready to price a season it has never seen?

The October runbook in `docs/preseason-plan.md` is four numbered steps and two or three
days of wall clock, and its defining property is that **the sampler work all lands before
the preseason** while the crunch is numpy over the pickles. That makes readiness a real
question with a wrong answer available: a missing head, a `full` window carrying last
month's specification, or a capture program that stopped firing are all things that look
fine right up until the two days when there is no time to fix them.

So this reads disk and answers it. **It fits nothing, fetches nothing and writes one CSV**
— the same discipline `make dashboard-audit` follows, and for the same reason: a check that
costs an afternoon is a check nobody runs in the week it matters.

## What "ready" means, in two halves

**The model half** is finishable today and is either done or not: twenty posterior heads at
the `full` window, each carrying the specification the `train` window selected. That is
`posteriors.assert_same_specification`, run across the deployment boundary rather than the
measurement one.

**The season half cannot be finished early and that is not a defect.** A target season has
no schedule, no roster, no preseason box scores and no game log until it has them; every
per-season artifact below is *expected* to be missing until its input exists. What this
check buys is that the list is enumerated and ordered, so the crunch is a checklist rather
than a rediscovery — and so a step that has silently stopped working is visible while there
is still time.

The two halves are reported apart for that reason. A red row in the model half is a
problem; a red row in the season half before October is the schedule.

Usage:
    python -m src.production_check                 # the next season, from the ADP board
    python -m src.production_check --season 2026-27
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.posteriors import (FIT_WINDOW, assert_same_specification,
                                   posteriors_dir)

#: The per-season artifacts the crunch produces, in the order it produces them. Each row is
#: `(root, name, the target that writes it, what it cannot be built without)`, where `root`
#: is the config key the directory comes from — the project has no shared config helper and
#: every module reads `cfg` itself, so a hard-coded `data/features` here would be the one
#: path in the project that ignores it.
#:
#: The order is the dependency order, so the first missing row is the one to work on.
SEASON_INPUTS = [
    ("raw_dir", "nbastats/schedule_{slug}.csv", "make scoring-periods",
     "the published schedule — available from ScheduleLeagueV2 as soon as the league "
     "releases it, which for 2026-27 was before 2026-08-21"),
    ("raw_dir", "nbastats/team_rosters_{slug}.csv", "make fetch",
     "the roster snapshot, which is the FORWARD membership rule (see "
     "src/features/forward_design.py)"),
    ("raw_dir", "nbastats/game_logs_pre_season_{slug}.csv", "make fetch",
     "the preseason box scores, so after the final preseason game"),
    ("raw_dir", "nbastats/game_logs_{slug}.csv", "make fetch",
     "the regular-season game log — absent for a season not yet played, and NOT needed "
     "for a forward board"),
    ("features_dir", "scoring_periods.parquet", "make scoring-periods",
     "the schedule — DK's rounds sit on the NBA week grid"),
    ("features_dir", "availability_panel.parquet", "make availability",
     "the schedule and the season-start rosters"),
    ("features_dir", "preseason.parquet", "make preseason",
     "the preseason box scores, so after the final preseason game"),
    ("features_dir", "component_targets.parquet", "make component-targets",
     "the game log — absent for a season that has not been played"),
    ("features_dir", "draft_pool.parquet", "make draft-pool",
     "the DK board, which is the pool for a season with no game log"),
    ("features_dir", "adp_panel.parquet", "make adp",
     "a board observed on or before the opener"),
]

#: Capture programs whose history cannot be backfilled. A gap here is permanent, so the
#: check reports the most recent capture rather than a pass/fail: what matters in August is
#: whether the cron is alive, and the only evidence of that is a recent date.
CAPTURE_CALENDAR = "outputs/eda/capture_calendar.csv"


def next_season(cfg: dict) -> str:
    """The season the production board is for — read from the pool, not from a calendar.

    `draft_pool.parquet` carries the live DK board, and a board exists for exactly one
    unplayed season at a time. Deriving the target from today's date instead would put the
    answer in a place nothing else in the project agrees with, and would be wrong for the
    two months a year when both boards are up.
    """
    pool = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "draft_pool.parquet",
                           columns=["season"])
    return str(sorted(pool["season"].unique())[-1])


def model_rows(cfg: dict) -> tuple[pd.DataFrame, str]:
    """Every `full`-window head, and whether it is the model the `train` window selected."""
    path = posteriors_dir(cfg, "full") / "manifest.csv"
    reference = posteriors_dir(cfg, FIT_WINDOW) / "manifest.csv"
    if not reference.exists():
        return pd.DataFrame(), (f"no {reference} — there is nothing to compare a "
                                f"production fit against")
    expected = sorted(pd.read_csv(reference)["head"])
    if not path.exists():
        return (pd.DataFrame([{"half": "model", "item": head, "state": "missing",
                               "detail": "no production fit"} for head in expected]),
                "the production fit has never been built — `make posteriors-production`")

    got = pd.read_csv(path).set_index("head")
    rows = [{"half": "model", "item": head,
             "state": "ready" if head in got.index else "missing",
             "detail": (f"{got.loc[head, 'variant']} · {got.loc[head, 'n_fit_rows']:,} "
                        f"rows {got.loc[head, 'first_season']}–"
                        f"{got.loc[head, 'last_season']}"
                        if head in got.index else "no production fit")}
            for head in expected]

    try:
        assert_same_specification(cfg, "full")
        note = f"all {len(expected)} heads match the `{FIT_WINDOW}` specification"
    except (ValueError, FileNotFoundError) as problem:
        note = str(problem).splitlines()[0]
    return pd.DataFrame(rows), note


def season_rows(cfg: dict, season: str) -> pd.DataFrame:
    """Which per-season inputs the target season already has, in dependency order.

    A parquet is checked for the season's **rows**, not for the file: every one of these
    artifacts exists already, covering seasons that have been played, and a file-level
    check would report the whole crunch as done.
    """
    rows = []
    for key, name, target, needs in SEASON_INPUTS:
        # Named files rather than a directory glob. `data/raw/nbastats` holds one file per
        # source per season, so "does anything for this season exist" answers `ready` the
        # moment a single one lands — which is exactly the false green this check exists
        # to prevent, and it fired the day the schedule was fetched.
        path = Path(cfg["data"][key]) / name.format(slug=season.replace("-", "_"))
        rel = str(path)
        if not path.exists():
            state, detail = "missing", needs
        elif path.suffix == ".csv":
            # A raw file is named for its season, so existence IS the check — there is no
            # `season` column to count and reading one as parquet is what this branch
            # exists to stop. A header-only file counts as absent, matching
            # `fetch._skip_or_fetch`: an empty response that reached disk should self-heal
            # rather than be reported as done forever.
            with path.open() as handle:
                n = sum(1 for _ in handle) - 1
            state, detail = (("ready", f"{n:,} rows") if n > 0 else ("missing", needs))
        else:
            frame = pd.read_parquet(path, columns=["season"])
            n = int((frame["season"].astype(str) == season).sum())
            state, detail = (("ready", f"{n:,} rows") if n else ("missing", needs))
        rows.append({"half": "season", "item": rel, "state": state,
                     "target": target, "detail": detail})
    return pd.DataFrame(rows)


def tensor_row(cfg: dict, season: str) -> pd.DataFrame:
    """The production tensor itself — the artifact the draft actually consumes.

    Until `docs/rookie-inclusive-tensors-plan.md` §5b the readiness list ended one step
    short of it. Three states rather than two, because for this artifact "exists" is not
    "current": a tensor built before the season's preseason log landed is the August
    board wearing October's name, which is exactly what `season.assert_tensor_current`
    refuses at load time. The row reads the SAME guard, so the checklist and the refusal
    cannot disagree.
    """
    path = Path(cfg["data"]["features_dir"]) / f"sim_tensor_{season}.npz"
    if not path.exists():
        return pd.DataFrame([{
            "half": "season", "item": str(path), "state": "missing",
            "target": "make simulate-production",
            "detail": "the tensor the draft room opens — buildable the moment the "
                      "`full` posteriors and the season's frames exist (a REHEARSAL "
                      "until the preseason lands)"}])

    # Function-level: `src/sim/season.py` is the numpy layer over the posteriors and
    # imports no sampler, but a check that costs a second should not pay for it on the
    # rows that never read a tensor.
    from src.sim.season import assert_tensor_current

    with np.load(path, allow_pickle=False) as z:
        meta = {"season": str(z["season"]), "fit_window": str(z["fit_window"]),
                "n_sims": int(z["n_sims"]),
                "preseason_log_rows": (int(z["preseason_log_rows"])
                                       if "preseason_log_rows" in z else None),
                "preseason_coverage": (float(z["preseason_coverage"])
                                       if "preseason_coverage" in z else None)}
    try:
        assert_tensor_current(cfg["data"]["raw_dir"], meta)
    except RuntimeError:
        return pd.DataFrame([{
            "half": "season", "item": str(path), "state": "stale",
            "target": "make preseason && make simulate-production",
            "detail": f"built with NO {season} preseason while the preseason log now "
                      f"exists on disk — the August board under October's name"}])

    coverage = meta["preseason_coverage"]
    stamped = coverage is not None and np.isfinite(coverage)
    rehearsal = meta["preseason_log_rows"] == 0
    return pd.DataFrame([{
        "half": "season", "item": str(path), "state": "ready",
        "target": "make simulate-production",
        "detail": f"{meta['n_sims']:,} sims at `{meta['fit_window']}`"
                  + (f", preseason coverage {coverage:.0%}" if stamped else "")
                  + (" — REHEARSAL: built before the preseason existed; rebuild after "
                     "the October fetch" if rehearsal else "")}])


def capture_rows() -> pd.DataFrame:
    """The most recent capture per non-backfillable program. Reported, never gated.

    There is no threshold that is right in both August and October — the ADP boards freeze
    between draft seasons and the injury feeds run nightly in-season — so this prints the
    date and lets a person read it. `docs/adp-plan.md` owns what the dates should be.
    """
    path = Path(CAPTURE_CALENDAR)
    if not path.exists():
        return pd.DataFrame([{"half": "capture", "item": "capture_calendar.csv",
                              "state": "missing",
                              "detail": "run `make capture-calendar`"}])
    calendar = pd.read_csv(path)
    captured = calendar[calendar["state"] == "captured"]
    return pd.DataFrame([{"half": "capture", "item": program, "state": "reported",
                          "detail": f"last captured {group['capture_date'].max()}, "
                                    f"{len(group):,} days on record"}
                         for program, group in captured.groupby("program")])


def run(cfg: dict, season: str | None = None) -> Path:
    season = season or next_season(cfg)
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Production readiness — the board for {season}")
    print(f"  Reads disk only: nothing is fitted, nothing is fetched. The runbook this "
          f"checks is\n  `docs/preseason-plan.md`, 'Production runbook'.")

    model, note = model_rows(cfg)
    print(f"\n[model — finishable today, and either done or not]")
    if len(model):
        ready = model[model["state"] == "ready"]
        print(f"  {len(ready)} of {len(model)} heads fitted at the `full` window")
        for row in model[model["state"] != "ready"].itertuples():
            print(f"    MISSING  {row.item:<20} {row.detail}")
    print(f"  {note}")

    season_frame = pd.concat([season_rows(cfg, season), tensor_row(cfg, season)],
                             ignore_index=True)
    print(f"\n[season — cannot be finished early; a missing row before October is the "
          f"schedule]")
    for row in season_frame.itertuples():
        mark = {"ready": "ready  ", "stale": "STALE  "}.get(row.state, "MISSING")
        print(f"    {mark}  {row.item:<42} {row.detail}")
        if row.state != "ready":
            print(f"{'':<15}{row.target}")

    capture = capture_rows()
    print(f"\n[capture — not backfillable, so a stale date is a permanent loss]")
    for row in capture.itertuples():
        print(f"    {row.item:<20} {row.detail}")

    frame = pd.concat([model, season_frame, capture], ignore_index=True)
    frame.insert(0, "season", season)
    dest = out_dir / "production_readiness.csv"
    frame.to_csv(dest, index=False)
    print(f"\nSaved {len(frame):,} readiness rows → {dest}")

    blocked = frame[(frame["half"] == "model") & (frame["state"] != "ready")]
    if len(blocked):
        print(f"  /!\\  {len(blocked)} production head(s) missing. The model half is the "
              f"half with no\n       deadline pressure — build it before the preseason, "
              f"not during the crunch.")
    else:
        print(f"  The model half is DONE. What remains is {season} data, and none of it "
              f"exists\n  until the season does — see the runbook's four crunch steps.")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default=None,
                        help="target season; defaults to the last season the draft pool "
                             "carries, which is the live DK board's")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, season=args.season)
