"""dk_pts at the weekly scoring period — Gate A at the unit the lineup is set at.

`make weekly-scores`. Reads the tensors `make simulate-season` wrote and the realized box
scores, and writes six flat, dashboard-shaped artifacts comparing **observed against
simulated `dk_pts` per player per scoring period**, on the training and validation splits.

## Why this unit, and why it needed an emitter

`make simulate-season`'s Gate A scores four things: season-total `dk_pts`, the games-played
pmf, the per-game bonus rate and the season-total minutes spread. **Nothing scores `dk_pts`
at the weekly unit** — and the weekly unit is where the contest happens. DK seats the best
7 of a 16-man roster **per scoring period** (`docs/dk_best_ball_rules.md`), so every weekly
max, every round total and every elimination cut downstream is a function of the *weekly*
distribution rather than of the season total. A head is only a model at the unit it was
scored at; this project has already paid once for that lesson at the season/team-game
boundary (`make minutes-unification`, one posterior and two opposite verdicts), and this is
the same check one level down from Gate A's own headline row.

Changing the unit needs **no re-simulation**: the tensor's second axis already *is* the
scoring period. What it needs is an emitter, because `dashboard/` may not open a 90 MB
`.npz` and reduce it — that is computing a model quantity, and the artifact rule says no.
So this module stands where `src/models/model_cards.py` stands for the model pages, and it
borrows that module's binning helpers **by import** rather than re-deriving them, so a
ribbon or a binned scatter on the weekly page is the same object a model page draws and a
reader learns the encoding once.

## Twenty periods, and three of them are not a week

`src/features/scoring_periods.py` owns the week grid: DK's Round 1 is **seventeen** weekly
periods and Rounds 2, 3 and 4 are **two weeks each**, which `src/sim/season.py` collapses
onto twenty tensor slots — one per Round-1 week, then one per later round. So the second
axis is weekly for seventeen of its twenty slots and fortnightly for the other three, and a
double week carries roughly twice the games.

**That is why the panels are faceted by period length rather than pooled.** Pooling a
2-week total into a distribution of 1-week totals produces a right tail that is a calendar
fact, and a reader would see it as a model that over-predicts. `PERIOD_TYPES` is the facet
and `weekly_score_period.csv` carries the per-slot readout underneath it.

## The seasons, and why four rather than twenty-nine

Only the two validation seasons had tensors. `season.allowed_seasons` already permits a
training season — it derives the legal set through `held_out.selection_split`, so a train
season needs no unlock and a test season still refuses — so the training side is two more
`make simulate-season` runs at ~80 s and ~80 MB each.

**Two training seasons, to match the validation pair, and they are 2018-19 and 2021-22
rather than the last two.** Measured before committing to the run: `2020-21` has **no
Round 4 at all** — the COVID season started on 2020-12-21 and ran out of weeks, so slot 19
carries 0 games and would contribute a column of structural zeros — and `2019-20`'s Round 4
is the Orlando bubble, 293 players against 373 in Round 3, where a fifth of the pool has an
observed zero that is a schedule fact rather than an availability outcome. The last two
training seasons carrying DK's whole four-round structure are 2018-19 and 2021-22.

## The gate

A pipeline step owes a build gate rather than a browser, and this one raises rather than
writing an artifact that is quietly describing the wrong thing:

| # | check | what it catches |
|---|---|---|
| 1 | **`dk_pts` is `compute_dk_pts`** — the stored column re-derived from the box score it was computed from | a `component_targets.parquet` rebuilt under a different scoring schedule |
| 2 | **the slot map is a partition** — every scored game lands in exactly one slot, every slot the season carries has games, and the tensor's own `tournament_round` agrees with `scoring_periods.parquet` | a tensor built under a different period map |
| 3 | **the periods reconstruct the season total** — the observed per-period sums equal `season.realized_frame`'s `dk_total` per player | a lost join, a drifted slot map, a double-counted game |
| 4 | **provenance** — every tensor at the `train` fit window, every season inside `selection_split` | a tensor whose coefficients read the seasons it is scored on |
| 5 | **the sim budget** — the ribbon and the KS distance re-read on two interleaved halves of the drawn seasons | a picture that is measuring the sampler rather than the model |

Split vocabulary is `model_cards.SPLITS` — the same closed pair, refused on the way to disk
by `_check_splits`, because a page rendering a third split would look like a feature.

Usage:
    python -m src.sim.weekly
    python -m src.sim.weekly --draws 1000
"""

import argparse
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import compute_dk_pts
from src.models.component_rates import build_design as component_build_design
from src.models.held_out import TEST_SEASONS, selection_split
from src.models.model_cards import (BAND_MIN_ROWS, CAL_BINS, ECDF_BAND_TOL, KS_MC_TOL,
                                    SPLITS, band_stability, calibration_rows, ecdf_rows,
                                    ks_stability, quantile_tables, rank_uniform,
                                    sample_frame, scaled_residuals)
from src.models.stan_utils import thin
from src.sim.season import (FIT_WINDOW, N_SCORING_PERIODS, ROUND_1_WEEKS,
                            marginal_metrics, scoring_slots)

# ── The unit ──────────────────────────────────────────────────────────────────

#: The two lengths a tensor slot can be. Seventeen of the twenty are one Round-1 week; the
#: other three are a whole later round, which is two weeks. A double week carries about
#: twice the games, so the two are faceted rather than pooled — see the module docstring.
WEEK = "week"
DOUBLE_WEEK = "double_week"
PERIOD_TYPES = (WEEK, DOUBLE_WEEK)
PERIOD_LABELS = {WEEK: "One week", DOUBLE_WEEK: "Double week"}
PERIOD_WEEKS = {WEEK: 1, DOUBLE_WEEK: 2}

#: What a row of every panel is, written onto the index so a page can label its own axis
#: rather than typing one — `model_cards.ResponseSpec.label`'s job, one artifact over.
RESPONSE_LABEL = "dk_pts in one scoring period"

#: The training pair. Chosen by measurement rather than by recency — see the docstring.
#: Not a config knob: `--season` exists for a probe, and the shipped set is a decision.
TRAIN_SEASONS = ("2018-19", "2021-22")

# ── The predictive ────────────────────────────────────────────────────────────

#: Simulated seasons kept per panel. The tensors carry 2,000 and every one of them is a
#: whole replicate season, so the budget is free in *simulation* terms and paid for in
#: memory and in the CRPS sort: 2,000 x 13,260 rows is a 212 MB float64 array copied three
#: times inside `crps_from_samples`. Thinned across the whole tensor rather than sliced off
#: the front, because sim `s` uses posterior draw `s % n_draws` and a front slice would take
#: a different part of the posterior. `--draws` moves it, and `band_stability` /
#: `ks_stability` are the checks that it is enough.
SIM_DRAWS = 500

#: Reported rather than gated below this many rows, exactly as `model_cards` gates its own
#: ribbon: a half-sample disagreement on a short panel measures the frame, not the budget.
MIN_GATED_ROWS = BAND_MIN_ROWS

#: The randomization's own stream, one per (period type, split). Deterministic across
#: rebuilds — a reader cannot tell an RNG from a re-simulation otherwise — and a different
#: stream from anything `src/sim/season.py` drew with.
QUANTILE_SEED = 20260810

#: What the observed side must reconstruct, in dk_pts. The per-period sums are a partition
#: of the season total the tensor is already gated on, so anything above float32 noise on a
#: ~2,000-point season is a broken join rather than a tolerance question.
SEASON_TOTAL_TOL = 1e-3

#: Ditto for the scoring function itself, re-derived from the box score on every row.
DK_PTS_TOL = 1e-6

ARTIFACTS = ("weekly_score_index.csv", "weekly_score_period.csv",
             "weekly_score_ecdf.csv", "weekly_score_calibration.csv",
             "weekly_score_quantile.csv", "weekly_score_sample.parquet")


# ── The split ─────────────────────────────────────────────────────────────────

def split_of(cfg: dict) -> dict[str, str]:
    """`season -> "train" | "validation"`, through `held_out.selection_split` and nothing else.

    The same choke point every model head goes through, so a test season cannot arrive here
    by being named on the command line: it is simply not in the mapping, and
    `assert_season_allowed` raises for it downstream.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    # The whole frame, exactly as `season.run` reads it: `build_design` collapses every
    # component column to a season total, so a narrowed read cannot produce the *target*
    # seasons — which are the 29 the split is a suffix of, not the 30 data seasons.
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = component_build_design(targets, cfg["data"]["seasons"],
                                    cfg["data"]["raw_dir"])
    train, validation = selection_split(design, TEST_SEASONS)
    out = {season: "train" for season in sorted(set(train["season"]))}
    out.update({season: "validation" for season in sorted(set(validation["season"]))})
    return out


def _check_splits(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """Refuse a split outside the closed vocabulary on the way to disk.

    Mirrors `model_cards._check_splits`, and exists for the same reason: it cannot fire
    today, and the failure it guards is silent by construction — an artifact carrying a
    third split would render as a feature rather than as a bug.
    """
    if "split" not in frame.columns:
        return frame
    unknown = sorted(set(frame["split"]) - set(SPLITS))
    if unknown:
        raise AssertionError(
            f"{name} carries split(s) {unknown}, outside {list(SPLITS)}. The test seasons "
            f"are locked by `held_out.selection_split`; nothing here may widen that.")
    return frame


# ── The period map ────────────────────────────────────────────────────────────

def period_frame(features_dir: Path, season: str) -> pd.DataFrame:
    """One row per tensor slot: its round, how many weeks it spans, and its games.

    Derived from `scoring_periods.parquet` through `season.scoring_slots` — the *same* slot
    map the tensor was written with, never a second copy of the collapse rule. `weeks` is
    the distinct `period_index` count inside the slot, which is what makes the Round-1 /
    later-round distinction a measurement rather than an assumption: nothing here hard-codes
    that Round 1 is seventeen ones and the rest are twos, and a season that failed to be
    either raises.
    """
    periods = pd.read_parquet(features_dir / "scoring_periods.parquet")
    periods = periods[periods["season"] == season]
    slots = scoring_slots(features_dir, season)
    merged = periods.merge(slots[["game_id", "slot"]], on="game_id", how="inner")
    inside = merged[merged["slot"] >= 0]
    frame = (inside.groupby("slot", as_index=False)
             .agg(tournament_round=("tournament_round", "first"),
                  n_rounds=("tournament_round", "nunique"),
                  weeks=("period_index", "nunique"),
                  games=("game_id", "nunique"),
                  start=("game_date", "min"), end=("game_date", "max")))
    if (frame["n_rounds"] != 1).any():
        raise AssertionError(
            f"{season}: a tensor slot spans more than one tournament round. The slot map "
            f"and `scoring_periods.parquet` disagree; re-run `make scoring-periods`.")
    bad = frame[~frame["weeks"].isin(PERIOD_WEEKS.values())]
    if len(bad):
        raise AssertionError(
            f"{season}: slot(s) {bad['slot'].tolist()} span {bad['weeks'].tolist()} weeks, "
            f"and a scoring period is one week (Round 1) or two (Rounds 2-4). "
            f"`src/sim/season.scoring_slots` collapses the later rounds; a third shape "
            f"means that rule has moved.")
    frame["period_type"] = np.where(frame["weeks"] == PERIOD_WEEKS[WEEK],
                                    WEEK, DOUBLE_WEEK)
    frame.insert(0, "season", season)
    return frame.drop(columns="n_rounds")


def assert_covers_the_tensor(frame: pd.DataFrame, tensor_rounds: np.ndarray,
                             season: str) -> None:
    """Gate 2 — the slot map is a partition, and the tensor agrees with it.

    Three ways a tensor can be describing a different calendar from the one this module
    reduces it by, none of which shows up in the numbers: a slot with no games (2020-21's
    Round 4, which is why that season is not in `TRAIN_SEASONS`), a slot the map never
    reaches, and a slot whose round label disagrees with the tensor's own
    `tournament_round` array. All three render as a perfectly good-looking panel.
    """
    have = set(frame["slot"])
    missing = sorted(set(range(N_SCORING_PERIODS)) - have)
    if missing:
        raise AssertionError(
            f"{season}: tensor slot(s) {missing} carry no scheduled game, so the season "
            f"does not cover all {N_SCORING_PERIODS} scoring periods. 2020-21 is the known "
            f"case — it has no Round 4 at all — and a season like it does not belong in a "
            f"pooled panel; drop it from `TRAIN_SEASONS` rather than scoring zeros.")
    declared = np.asarray(tensor_rounds, dtype=int)
    if len(declared) != N_SCORING_PERIODS:
        raise AssertionError(
            f"{season}: the tensor declares {len(declared)} scoring periods against "
            f"{N_SCORING_PERIODS}. It was written by a different `src/sim/season.py`.")
    actual = frame.sort_values("slot")["tournament_round"].to_numpy(int)
    if not np.array_equal(actual, declared):
        raise AssertionError(
            f"{season}: the tensor's round map {declared.tolist()} disagrees with "
            f"`scoring_periods.parquet` {actual.tolist()}. Re-run `make simulate-season` "
            f"for this season — the tensor predates the current period grid.")


# ── The observed side ─────────────────────────────────────────────────────────

def observed_periods(cfg: dict, season: str, players: np.ndarray,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """`(observed, games)` — realized `dk_pts` and games per (player, tensor slot).

    Both are `(players x 20)`, aligned on the tensor's own `player_id` order.

    **`dk_pts` is `preprocess.compute_dk_pts` and is never re-implemented.** The column on
    `component_targets.parquet` is what that function wrote (`features/targets.py`), so
    reading it *is* the reuse — and gate 1 re-derives it from the same box score rather than
    trusting the label, which is the difference between reusing a function and assuming a
    column.

    Regular season only and played games only, matching `season.realized_frame`: a playoff
    row belongs to no contest, and a game a player missed contributes a zero through its
    absence rather than a row.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    rows = targets[(targets["season"] == season)
                   & (targets["season_type"] == "regular")
                   & (targets["played"] == 1)].copy()
    if rows.empty:
        raise ValueError(f"`component_targets.parquet` carries no played regular-season "
                         f"rows for {season}")

    # Gate 1. `compute_dk_pts` reads `pts`, so it is reassembled from the makes the same
    # way `season._sim_one` does — the bonus is a staircase on five components and this is
    # the one place a scoring-schedule change would land silently.
    rows["pts"] = 2 * rows["fg2m"] + 3 * rows["fg3m"] + rows["ftm"]
    error = float((compute_dk_pts(rows) - rows["dk_pts"]).abs().max())
    if error > DK_PTS_TOL:
        raise AssertionError(
            f"{season}: `component_targets.parquet`'s `dk_pts` column is {error:.3e} away "
            f"from `preprocess.compute_dk_pts` re-derived on the same box score (bar "
            f"{DK_PTS_TOL:g}). The scoring schedule moved under the artifact — re-run "
            f"`make component-targets` rather than loosening this.")

    slots = scoring_slots(features_dir, season)
    rows = rows.merge(slots[["game_id", "slot"]], on="game_id", how="left")
    if rows["slot"].isna().any():
        raise AssertionError(
            f"{season}: {int(rows['slot'].isna().sum()):,} played regular-season games are "
            f"not in the scoring-period grid at all. `make scoring-periods` and "
            f"`make component-targets` describe different schedules.")
    rows["slot"] = rows["slot"].astype(int)

    position = {int(pid): i for i, pid in enumerate(players)}
    inside = rows[(rows["slot"] >= 0) & rows["player_id"].isin(position)]
    unit = inside["player_id"].map(position).to_numpy(np.int64)
    flat = unit * N_SCORING_PERIODS + inside["slot"].to_numpy(np.int64)
    size = len(players) * N_SCORING_PERIODS
    observed = np.bincount(flat, weights=inside["dk_pts"].to_numpy(float),
                           minlength=size).reshape(len(players), N_SCORING_PERIODS)
    games = np.bincount(flat, minlength=size).reshape(len(players), N_SCORING_PERIODS)
    return observed, games


def season_games(cfg: dict, season: str, players: np.ndarray) -> np.ndarray:
    """Realized regular-season games played, per tensor unit — the population filter.

    A player who never appeared is twenty observed zeros against a simulated season, and
    `season.gate_a` already drops him from the season-total row for the same reason: his
    zero is a roster fact rather than a weekly outcome. Kept as its own function so the two
    populations are visibly the same rule.
    """
    targets = pd.read_parquet(Path(cfg["data"]["features_dir"])
                              / "component_targets.parquet",
                              columns=["player_id", "season", "season_type", "played"])
    rows = targets[(targets["season"] == season)
                   & (targets["season_type"] == "regular")
                   & (targets["played"] == 1)]
    counts = rows.groupby("player_id")["played"].sum()
    return counts.reindex(players).fillna(0).to_numpy(float)


# ── The tensor ────────────────────────────────────────────────────────────────

def load_tensor(features_dir: Path, season: str) -> dict:
    """One season's `.npz`, with the provenance a consumer needs to refuse the wrong one."""
    path = features_dir / f"sim_tensor_{season}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run `make simulate-season` for {season} — "
            f"`python -m src.sim.season --season {season}` — which needs no CmdStan.")
    z = np.load(path, allow_pickle=False)
    return {"season": season, "dk_pts": z["dk_pts"], "player_id": z["player_id"],
            "tournament_round": z["tournament_round"],
            "fit_window": str(z["fit_window"]), "n_sims": int(z["n_sims"]),
            "n_posterior_draws": int(z["n_posterior_draws"]), "seed": int(z["seed"])}


def assert_window(tensor: dict, window: str = FIT_WINDOW) -> None:
    """Gate 4 — a tensor fitted wider has read the seasons it is about to be scored on.

    `posteriors.require_window`'s argument, one layer up: no frame-level guard can see a
    leak that arrived through the coefficients, so the window is checked where the artifact
    is consumed rather than assumed from where it was produced.
    """
    if tensor["fit_window"] != window:
        raise AssertionError(
            f"{tensor['season']}: the tensor was simulated from `{tensor['fit_window']}` "
            f"posteriors and this readout requires `{window}`. A `train_val` head has read "
            f"2022-23 and 2023-24 through its coefficients, which is exactly the leak the "
            f"validation column here would hide.")


# ── Panels ────────────────────────────────────────────────────────────────────

def panel_rows(tensors: dict, observed: dict, periods: dict, keep: dict,
               period_type: str, seasons: list[str], draws: int) -> dict:
    """Stack one facet's rows across its seasons: observed, the draws, and the row index.

    A row is one **(season, player, scoring period)**. The draw block is
    `(sims x rows)` — one whole simulated season per column-slice, which is what the ECDF
    ribbon needs and why the sims are thinned rather than pooled.
    """
    obs, sims, index = [], [], []
    for season in seasons:
        tensor = tensors[season]
        slots = periods[season]
        chosen = np.sort(slots.loc[slots["period_type"] == period_type,
                                   "slot"].to_numpy(int))
        players = keep[season]
        if not len(chosen) or not players.any():
            continue
        take = thin(tensor["n_sims"], draws)
        block = tensor["dk_pts"][np.ix_(np.flatnonzero(players), chosen)][:, :, take]
        # (players x slots x sims) -> (sims x players*slots), the layout every helper below
        # and `crps_from_samples` expect.
        sims.append(block.transpose(2, 0, 1).reshape(len(take), -1))
        obs.append(observed[season][np.ix_(np.flatnonzero(players), chosen)].ravel())
        index.append(pd.DataFrame({
            "season": season,
            "player_id": np.repeat(tensor["player_id"][players], len(chosen)),
            "slot": np.tile(chosen, int(players.sum()))}))
    if not obs:
        return {"observed": np.zeros(0), "draws": np.zeros((0, 0)),
                "index": pd.DataFrame(columns=["season", "player_id", "slot"])}
    return {"observed": np.concatenate(obs),
            "draws": np.concatenate(sims, axis=1).astype(float),
            "index": pd.concat(index, ignore_index=True)}


def facet_tables(period_type: str, split: str, panel: dict) -> dict:
    """Every artifact row for one (period type, split), cut from **one** draw block.

    The ribbon, the density, the residual and the overlay are four readings of one
    predictive — `model_cards.predictive_tables`' rule, and the reason it is a rule: drawing
    them separately would let the page show a ribbon and a QQ describing two different
    simulations. The helpers are that module's own, by import, so the encodings match the
    model pages exactly; only the key column is renamed, since `head` is not what a row of
    this artifact is keyed by.
    """
    observed, draws = panel["observed"], panel["draws"]
    predicted = draws.mean(axis=0)

    ecdf, band = ecdf_rows(period_type, split, observed, draws)
    seed = facet_seed(period_type, split)
    u = scaled_residuals(draws, observed, seed)
    rank = rank_uniform(predicted)
    quantile, ks = quantile_tables(period_type, split, u, rank)
    sample = sample_frame(period_type, split, predicted, observed, u, rank)
    gated = len(observed) >= MIN_GATED_ROWS
    return {
        "ecdf": _rekey(pd.DataFrame(ecdf)),
        "quantile": _rekey(pd.DataFrame(quantile)),
        "sample": _rekey(sample),
        "predicted": predicted, "observed": observed,
        "summary": {
            "ecdf_band_mc": band, "ecdf_band_gated": gated,
            "ks": ks, "ks_mc": ks_stability(draws, observed, seed), "ks_gated": gated,
            "quantile_seed": seed,
            **marginal_metrics(draws, observed),
            "observed_mean": float(observed.mean()),
            "observed_sd": float(observed.std(ddof=1)) if len(observed) > 1 else float("nan"),
            "predicted_mean": float(predicted.mean()),
            # **Three spreads, because only one of them is comparable to `observed_sd`.**
            # `point_sd` is the spread of the per-row posterior *means* and is smaller than
            # the data by construction — a mean over draws has averaged its own noise away,
            # and a page printing it beside `observed_sd` would report a model that is far
            # too narrow when nothing of the sort has been measured. `predictive_sd` is what
            # the model puts on **one** player-week, and `pooled_sd` is the marginal it
            # implies over every row and draw at once, which is the one `observed_sd`
            # answers. For a best-ball weekly max the spread is the whole question, so all
            # three ship rather than the page picking one.
            "point_sd": float(predicted.std(ddof=1)) if len(predicted) > 1 else float("nan"),
            "predictive_sd": float(draws.std(axis=0, ddof=1).mean()),
            "pooled_sd": float(draws.std(ddof=1)),
            "zero_share": float((observed == 0).mean()),
            "predicted_zero_share": float((draws == 0).mean()),
        },
    }


def facet_seed(period_type: str, split: str) -> int:
    """A stable per-facet seed, so the four randomizations are four streams.

    `zlib.crc32` rather than `hash`, which is salted per process — the same device
    `model_cards._seed` uses, and for the same reason: a residual whose seed moves under a
    rebuild lets a reader mistake an RNG for a re-simulation.
    """
    return QUANTILE_SEED + int(zlib.crc32(f"{period_type}/{split}".encode())) % 100_000


def _rekey(frame: pd.DataFrame) -> pd.DataFrame:
    """`head` -> `period_type`. The helpers are the model cards'; the key is not."""
    return frame.rename(columns={"head": "period_type"})


# ── The build ─────────────────────────────────────────────────────────────────

def check_budget(period_type: str, split: str, summary: dict, draws: int) -> None:
    """Gate 5 — is the picture a reading of the simulator, or of how many seasons were drawn?

    Two bars, both `model_cards`' own and both on the **budget** rather than on the model:
    the ribbon's half-sample disagreement and the KS distance's. The KS distance itself is
    never thresholded — at these sample sizes a uniformity test rejects everything — which
    is the rule `band_distance` already carries one artifact over.
    """
    band = summary["ecdf_band_mc"]
    if summary["ecdf_band_gated"] and np.isfinite(band) and band > ECDF_BAND_TOL:
        raise AssertionError(
            f"{period_type}/{split}: the 95% ECDF ribbon moves by {band:.4f} between two "
            f"halves of the {draws} simulated seasons (bar {ECDF_BAND_TOL}). That is Monte "
            f"Carlo noise at this budget — raise `--draws` rather than shipping it.")
    ks_mc = summary["ks_mc"]
    if summary["ks_gated"] and np.isfinite(ks_mc) and ks_mc > KS_MC_TOL:
        raise AssertionError(
            f"{period_type}/{split}: the KS distance moves by {ks_mc:.4f} between two "
            f"halves of the {draws} simulated seasons (bar {KS_MC_TOL}). At this budget a "
            f"row with no replicate at its observed value carries a residual quantized to "
            f"1/{draws}; raise `--draws` rather than tiling a number that is measuring the "
            f"sampler.")


def period_readout(tensors: dict, observed: dict, games: dict, periods: dict,
                   keep: dict, splits: dict, draws: int) -> pd.DataFrame:
    """One row per (season, scoring period): the same metrics, slot by slot.

    The facets above pool seventeen weeks into one distribution, which is the right unit for
    a calibration panel and the wrong one for "does the simulator drift through the season".
    This is that second question, and it is cheap — 20 rows a season.
    """
    rows = []
    for season, tensor in tensors.items():
        players = np.flatnonzero(keep[season])
        take = thin(tensor["n_sims"], draws)
        slots = periods[season].set_index("slot")
        for slot in range(N_SCORING_PERIODS):
            block = tensor["dk_pts"][np.ix_(players, [slot])][:, 0, :][:, take].T
            y = observed[season][players, slot]
            info = slots.loc[slot]
            rows.append({
                "split": splits[season], "season": season, "slot": int(slot),
                "tournament_round": int(info["tournament_round"]),
                "period_type": str(info["period_type"]),
                "period_label": PERIOD_LABELS[str(info["period_type"])],
                "weeks": int(info["weeks"]), "team_games": int(info["games"]),
                "start": str(info["start"])[:10], "end": str(info["end"])[:10],
                "observed_mean": float(y.mean()),
                "observed_games": float(games[season][players, slot].mean()),
                "predicted_mean": float(block.mean()),
                **marginal_metrics(block.astype(float), y),
            })
    return pd.DataFrame(rows)


def build(cfg: dict, seasons: dict[str, str], draws: int = SIM_DRAWS) -> dict:
    """Every artifact, plus the gate readings, for the given `season -> split` mapping."""
    features_dir = Path(cfg["data"]["features_dir"])
    tensors, observed, games, periods, keep = {}, {}, {}, {}, {}

    for season in sorted(seasons):
        tensor = load_tensor(features_dir, season)
        assert_window(tensor)
        frame = period_frame(features_dir, season)
        assert_covers_the_tensor(frame, tensor["tournament_round"], season)
        obs, gp = observed_periods(cfg, season, tensor["player_id"])
        assert_reconstructs_season(cfg, season, tensor["player_id"], obs)
        tensors[season] = tensor
        observed[season] = obs
        games[season] = gp
        periods[season] = frame
        keep[season] = season_games(cfg, season, tensor["player_id"]) > 0
        print(f"  {season} ({seasons[season]:<10s}) {tensor['dk_pts'].shape[0]:>4,} units, "
              f"{int(keep[season].sum()):>4,} of them with a realized game · "
              f"{tensor['n_sims']:,} sims thinned to {min(draws, tensor['n_sims']):,} · "
              f"{int(frame['games'].sum()):,} scored team-games over "
              f"{int(frame['weeks'].sum())} weeks")

    index, ecdf, calibration, quantile, samples = [], [], [], [], []
    for period_type in PERIOD_TYPES:
        panels = {}
        for split in SPLITS:
            members = sorted(s for s, value in seasons.items() if value == split)
            panel = panel_rows(tensors, observed, periods, keep, period_type, members,
                               draws)
            if not len(panel["observed"]):
                continue
            tables = facet_tables(period_type, split, panel)
            check_budget(period_type, split, tables["summary"], draws)
            ecdf.append(tables["ecdf"])
            quantile.append(tables["quantile"])
            samples.append(tables["sample"])
            panels[split] = (tables["predicted"], tables["observed"])
            slots = {int(s) for season in members
                     for s in periods[season].loc[
                         periods[season]["period_type"] == period_type, "slot"]}
            index.append({
                "period_type": period_type, "period_label": PERIOD_LABELS[period_type],
                "split": split, "seasons": ",".join(members), "n_seasons": len(members),
                "n_periods": len(slots), "weeks": PERIOD_WEEKS[period_type],
                "n_players": int(panel["index"]["player_id"].nunique()),
                "n_player_seasons": int(
                    panel["index"].drop_duplicates(["season", "player_id"]).shape[0]),
                "team_games": int(sum(
                    periods[s].loc[periods[s]["period_type"] == period_type,
                                   "games"].sum() for s in members)),
                "response_label": RESPONSE_LABEL,
                "sim_draws": int(min(draws, min(tensors[s]["n_sims"] for s in members))),
                "n_sims": int(min(tensors[s]["n_sims"] for s in members)),
                "n_posterior_draws": int(min(tensors[s]["n_posterior_draws"]
                                             for s in members)),
                "fit_window": FIT_WINDOW,
                **tables["summary"],
            })
        if panels:
            # Both splits share one edge set per facet — `calibration_rows`' own rule, and
            # the reason the two panels can be read against each other at all.
            calibration.extend(calibration_rows(period_type, panels, bins=CAL_BINS))

    return {
        "weekly_score_index.csv": pd.DataFrame(index),
        "weekly_score_period.csv": period_readout(tensors, observed, games, periods, keep,
                                                  seasons, draws),
        "weekly_score_ecdf.csv": pd.concat(ecdf, ignore_index=True),
        "weekly_score_calibration.csv": _rekey(pd.DataFrame(calibration)),
        "weekly_score_quantile.csv": pd.concat(quantile, ignore_index=True),
        "weekly_score_sample.parquet": pd.concat(samples, ignore_index=True),
    }


def assert_reconstructs_season(cfg: dict, season: str, players: np.ndarray,
                               observed: np.ndarray) -> None:
    """Gate 3 — the twenty periods add up to the season total Gate A already scores.

    The strongest available check on the reduction, because it goes through a *different*
    function on the other side: `season.realized_frame` sums the same games by season and
    this module sums them by period, so a drifted slot map, a lost join or a double-counted
    game shows up as a disagreement rather than as a plausible weekly panel.
    """
    from src.sim.season import realized_frame

    truth = realized_frame(cfg, {"season": season, "unit_ids": players})
    error = float(np.abs(observed.sum(axis=1) - truth["dk_total"].to_numpy(float)).max())
    if error > SEASON_TOTAL_TOL:
        raise AssertionError(
            f"{season}: the per-period observed dk_pts sum to within {error:.4f} of "
            f"`season.realized_frame`'s season total (bar {SEASON_TOTAL_TOL:g}). The "
            f"scoring-period reduction is not a partition of the games Gate A scores.")


def run(cfg: dict, seasons: list[str] | None = None, draws: int = SIM_DRAWS,
        write: bool = True) -> dict[str, Path]:
    started = time.perf_counter()
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Weekly scores — observed against simulated dk_pts, per player per scoring "
          "period.")
    print(f"  Gate A scores the season total; this is the same tensor at the unit a "
          f"best-ball\n  lineup is actually set at. No re-simulation — the period axis is "
          f"already there.")
    mapping = split_of(cfg)
    chosen = list(seasons) if seasons else sorted(
        set(TRAIN_SEASONS) | {s for s, v in mapping.items() if v == "validation"})
    unknown = [s for s in chosen if s not in mapping]
    if unknown:
        raise AssertionError(
            f"{unknown} is not a season `held_out.selection_split` allows. The test split "
            f"is locked and this readout may not reach it — `make final-evaluation` is the "
            f"only thing that unlocks anything.")
    picked = {s: mapping[s] for s in chosen}
    print(f"  {len(picked)} seasons: "
          + " · ".join(f"{s} ({v})" for s, v in sorted(picked.items())))
    print(f"  2020-21 and 2019-20 are deliberately absent — one has no Round 4 at all and "
          f"the\n  other's is the bubble. See the module docstring.\n")

    tables = build(cfg, picked, draws)

    paths = {}
    print()
    for name, table in tables.items():
        _check_splits(table, name)
        dest = out_dir / name
        if not write:
            print(f"Would write {len(table):,} rows x {table.shape[1]} columns → {dest}")
            continue
        if dest.suffix == ".parquet":
            table.to_parquet(dest, index=False)
        else:
            table.to_csv(dest, index=False, float_format="%.6g")
        paths[dest.stem] = dest
        print(f"Wrote {len(table):,} rows x {table.shape[1]} columns → {dest}")

    _report(tables["weekly_score_index.csv"], draws,
            time.perf_counter() - started)
    return paths


def _report(index: pd.DataFrame, draws: int, elapsed: float) -> None:
    """Print every facet explicitly. A distance is a distance, and never a verdict."""
    print("\nGate A at the scoring period — the simulator against the weeks it drew")
    for _, row in index.sort_values(["period_type", "split"]).iterrows():
        print(f"  {row['period_label']:<12s} {row['split']:<10s} n={int(row['n']):>6,}  "
              f"observed {row['observed_mean']:6.2f}  predicted {row['predicted_mean']:6.2f}"
              f"  MAE {row['mae']:6.2f}  bias {row['bias']:+6.2f}  "
              f"R2 {row['r2']:6.4f}  CRPS {row['crps']:6.2f}")
    print("\nSpread — the statistic a weekly max is most sensitive to. `pooled_sd` is the "
          "one\ncomparable to the observed; the point prediction's own spread is narrower "
          "by construction.")
    for _, row in index.sort_values(["period_type", "split"]).iterrows():
        print(f"  {row['period_label']:<12s} {row['split']:<10s} observed sd "
              f"{row['observed_sd']:6.2f}  ·  pooled {row['pooled_sd']:6.2f}  "
              f"({row['pooled_sd'] / row['observed_sd']:.3f}x)  ·  per player-period "
              f"{row['predictive_sd']:6.2f}  ·  point prediction {row['point_sd']:6.2f}"
              f"  ·  zeros {row['zero_share']:.1%} observed against "
              f"{row['predicted_zero_share']:.1%}")
    print(f"\nCalibration readings — DISTANCES, never pass/fail. At these sample sizes a "
          f"uniformity\ntest rejects everything; the only bars here are on the "
          f"{draws} simulated seasons behind them.")
    for _, row in index.sort_values(["period_type", "split"]).iterrows():
        print(f"  {row['period_label']:<12s} {row['split']:<10s} "
              f"ECDF band half-sample {row['ecdf_band_mc']:.4f} "
              f"(bar {ECDF_BAND_TOL})  ·  KS {row['ks']:.4f} "
              f"half-sample {row['ks_mc']:.4f} (bar {KS_MC_TOL})"
              f"{'' if row['ks_gated'] else '  [ungated]'}")
    print(f"\n{len(index)} facets over {index['n'].sum():,} player-periods; {elapsed:.1f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Observed against simulated dk_pts at the weekly scoring period.")
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to the two validation "
                             "seasons plus the training pair.")
    parser.add_argument("--draws", type=int, default=SIM_DRAWS,
                        help="simulated seasons kept per panel, thinned across the tensor")
    parser.add_argument("--no-write", action="store_true",
                        help="build and gate without writing, for a budget measurement")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, draws=args.draws, write=not args.no_write)
