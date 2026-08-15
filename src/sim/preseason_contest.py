"""The preseason block's contest value — a PAIRED counterfactual, not a re-read.

`docs/preseason-plan.md` P5 ran the whole chain behind the adopted composition blend and
every readout improved: Round-1 advance lift went 0.1890 -> 0.2358 simulated and 0.1713 ->
0.204098 realized. **It could attribute none of it.** The composition's posterior,
`sim.minutes.player_season_sigma`, the ADP field and Gate C's error injection all moved in
that one pass, and the previous `strategy_*.csv` was overwritten rather than kept — so the
honest statement was "the chain under the new heads reads higher", which is not the gate's
question. This module is the instrument that answers it, and it is `src/sim/mixture_value.py`
one round over: **two arms of the same chain, captured under the same code, reported side by
side.**

## The procedure this module is the last step of

Two passes, differing in exactly the three config keys that ARE the block:

    # arm A — the counterfactual, run FIRST so the shipped arm is what disk ends on
    stan.availability.preseason: false
    stan.minutes.preseason: false
    stan.composition.preseason.adopt: false
    python -m src.models.posteriors --window train --groups availability,minutes,composition
    make simulate-season bracket draft-sim strategy-sweep
    python -m src.sim.preseason_contest --capture base

    # arm B — the shipped head, all three keys back to true
    ... the same targets ...
    python -m src.sim.preseason_contest --capture preseason

    make preseason-contest

## Only two of the three heads can reach this readout, and that is a result rather than a
## shortcut

The block sits on three heads. `stan_availability.head_design` and
`stan_composition.head_frame` are both read by `src/sim/season.py`, so those two reach the
tensor. **The marginal minutes head does not**: `src/sim/` imports neither `StanMinutes` nor
`rehydrate_minutes` and never looks up `artifacts["minutes"]` — what it takes from that
module is `beta_shapes`, which is arithmetic. `README.md` carried the opposite claim until
2026-08-14. So P3's block — the largest of the three by its own gate, at −4.789 validation
CRPS minutes — is structurally invisible to every number below, and a null here would be
about the other two heads.

Its key is flipped in the counterfactual and its posterior refitted anyway, at 514 s, so the
arm name means "the preseason block off" rather than "two thirds of it off" and no
config/artifact mismatch can hide in the pass. `tests/test_preseason_contest.py` pins the
structural fact rather than trusting this docstring.

## Why the artifact carries both arms rather than a delta against the live one

`mixture_value`'s reason, unchanged: the failure being removed is a comparison against a
figure whose vintage nobody could reconstruct, and a file holding one arm invites exactly
that on the next pass. Both arms and their `captured_at` go in together.

## What the reader has to know before reading the contest block

**The simulated lift is not arm-comparable, and it is a row rather than a footnote.** Each
arm's `sim_lift` is measured against a symmetric-field null *inside a world that arm
generated*, so an arm whose predictive is wider can score a higher lift because its world
spreads rosters further apart, not because its board drafts better. The realized block is the
only one whose truth — actual 2022-23 / 2023-24 box scores — is common to both arms, and it
is two seasons deep.

`resolution` is the other guard: the pooled standard error on an advance probability at
`sim.strategy.n_sims` worlds per season, doubled into an arm-to-arm difference and scaled to
95%. A gap under that row is a null and must be read as one.

**And this block arrives as ORDER, not as shape** — which is the one way it differs from the
mixture's counterfactual and the reason the `board` and `draw.mean_total` rows are load
bearing here. The composition's own gate scored its gain at the allocation MEAN
(`docs/preseason-plan.md` 4d: season MAE −17.70 minutes while the predictive sd *narrowed*),
and a mean shift is exactly what moves a ranking. `§7l`'s precedent — that a head change
arriving as shape can be a measured null on Round-1 advance probability — therefore applies
less cleanly, and the board block is what turns "less cleanly" into a number.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The arm names, and the order every table is written in: counterfactual first, shipped
# second, so `delta` reads "what the preseason block did". They are the composition gate's
# own arm names (`stan.composition.preseason.arms`), so a row here and a row in
# `composition_preseason_fit.csv` mean the same thing by the same word.
ARMS = ("base", "preseason")
CAPTURE_DIR = "preseason_arms"

# The three config keys that ARE the block. `--capture` requires all three to agree with the
# arm name: a pass with two off and one on is neither arm, and it is the shape of mistake
# that leaves every number below looking plausible.
BLOCK_KEYS = (("stan", "availability", "preseason"),
              ("stan", "minutes", "preseason"),
              ("stan", "composition", "preseason", "adopt"))
KEY_NAMES = tuple(".".join(path) for path in BLOCK_KEYS)

# The heads whose POSTERIOR has to be refitted for an arm. `minutes` is here even though it
# cannot reach the tensor (see the module docstring) — the arm is the block, not the subset
# of it that happens to be observable.
REFIT_GROUPS = ("availability", "minutes", "composition")

# The CSVs an arm is captured from — `mixture_value.ARM_TABLES` verbatim, because the two
# counterfactuals ride the same five-target chain. A missing file means the arm was captured
# mid-run and `--capture` refuses.
ARM_TABLES = ("sim_season_gate_a", "bracket_structure", "bracket_null", "bracket_entries",
              "draft_gate_b", "draft_field", "strategy_gate_c", "strategy_injection",
              "strategy_null", "strategy_sweep", "strategy_paired", "strategy_gate_d",
              "strategy_realized", "strategy_shipped")

# The seasons the sweep scores. Validation, and `held_out.selection_split` is what put them
# there — this module never widens them.
SEASONS = ("2022-23", "2023-24")

# The shipped strategy, held fixed across arms so the contest block compares the same object.
# Reading each arm's own selection instead would confound "the block drafts better" with "the
# block selected a different arm"; the selection itself is reported separately.
REFERENCE_STRATEGY = "lineup_value_blend30"

# Prior-minutes-per-game role buckets, matching `src/eda/season_effects.ROLE_EDGES`.
ROLE_EDGES = (-0.01, 12.0, 24.0, 30.0, 99.0)
ROLE_LABELS = ("<12 mpg", "12-24", "24-30", "30+ mpg")

# The tensor's OWN games-played denominator: twenty scoring periods span ~76 games rather
# than 82, so 82-game thresholds do not transfer.
IRON_MAN_GP = 70
IRON_MAN_STRICT_GP = 75
DISRUPTED_GP = 41

# The picks a 12-entry, 16-round snake actually consumes. A rank move beyond this cannot
# reach any roster, so the board block reports the move over the drafted prefix and not over
# a tail nobody sees.
DRAFTED_PICKS = 192


# ── capture ───────────────────────────────────────────────────────────────────

def block_state(cfg: dict) -> dict:
    """The three keys, resolved. Missing reads as the shipped default, which is `true`."""
    state = {}
    for path in BLOCK_KEYS:
        node = cfg
        for part in path[:-1]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        state[".".join(path)] = bool(node.get(path[-1], True))
    return state


def tensor_summary(features_dir: Path, season: str) -> pd.DataFrame:
    """One row per player: the season total's LEVEL and shape, and the games-played tails.

    `mean_total` matters more here than in the mixture's version and is the first column for
    that reason. The composition's gate scored this block at the allocation mean, and a mean
    is what a draft ranking consumes — so the level is the quantity whose movement predicts
    a contest effect, where the mixture's claim was about the tails.
    """
    with np.load(features_dir / f"sim_tensor_{season}.npz", allow_pickle=False) as z:
        total = z["dk_pts"].astype(np.float64).sum(axis=1)
        gp = z["games_played"].astype(np.int32).sum(axis=1)
        player_id = z["player_id"]
        prior_minutes = z["prior_minutes"]
        n_sims = int(z["n_sims"])
    return pd.DataFrame({
        "season": season, "player_id": player_id, "n_sims": n_sims,
        "prior_mpg": np.asarray(prior_minutes, dtype=float) / 82.0,
        "mean_total": total.mean(axis=1), "sd_total": total.std(axis=1),
        "q10_total": np.quantile(total, 0.10, axis=1),
        "q90_total": np.quantile(total, 0.90, axis=1),
        "mean_gp": gp.mean(axis=1),
        "p_disrupted": (gp <= DISRUPTED_GP).mean(axis=1),
        "p_iron_man": (gp >= IRON_MAN_GP).mean(axis=1),
        "p_iron_man_strict": (gp >= IRON_MAN_STRICT_GP).mean(axis=1),
    })


def capture(cfg: dict, arm: str) -> Path:
    """Freeze the arm currently on disk into `outputs/predictions/preseason_arms/<arm>/`."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {ARMS}")
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    dest = out_dir / CAPTURE_DIR / arm
    dest.mkdir(parents=True, exist_ok=True)

    state = block_state(cfg)
    want = (arm == "preseason")
    wrong = {key: value for key, value in state.items() if value != want}
    if wrong:
        raise ValueError(
            f"you are capturing the {arm!r} arm, which needs all three preseason keys "
            f"{want}, and {wrong} disagree. That mismatch is the whole failure this module "
            f"exists to prevent — a pass with the block half on is NEITHER arm and every "
            f"number below would still look plausible. Fix the config or the arm name, and "
            f"re-run the chain if the artifacts on disk were built under the other setting.")

    missing = [t for t in ARM_TABLES if not (out_dir / f"{t}.csv").exists()]
    if missing:
        raise FileNotFoundError(
            f"the {arm!r} arm is missing {missing}. Capture only a COMPLETE chain: "
            f"posteriors -> simulate-season -> bracket -> draft-sim -> strategy-sweep.")

    for name in ARM_TABLES:
        shutil.copy2(out_dir / f"{name}.csv", dest / f"{name}.csv")
    frames = [tensor_summary(features_dir, season) for season in SEASONS]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(dest / "tensor_summary.csv", index=False)

    # The fitted window each refitted head actually used, read off the manifest rather than
    # from config: two of the three keys move a head's `first_season` as a side effect
    # (2004-05 with the block, 1997-98 / 1996-97 without), and an arm that did not refit
    # would show the other arm's window here rather than failing.
    manifest = pd.read_csv(features_dir / "posteriors" / "train" / "manifest.csv"
                           ).set_index("head")
    windows = manifest["first_season"].to_dict()
    features = manifest["n_features"].to_dict()

    stamp = pd.DataFrame([{
        "arm": arm,
        **state,
        "captured_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "availability_first_season": windows.get("availability"),
        "minutes_first_season": windows.get("minutes"),
        "composition_first_season": windows.get("composition"),
        "availability_n_features": features.get("availability"),
        "minutes_n_features": features.get("minutes"),
        "composition_n_features": features.get("composition"),
        "player_season_sigma": float(cfg.get("sim", {}).get("minutes", {})
                                     .get("player_season_sigma", float("nan"))),
        "n_sims": int(summary["n_sims"].iloc[0]),
        "n_players": int(len(summary) / len(SEASONS)),
    }])
    stamp.to_csv(dest / "provenance.csv", index=False)
    print(f"Captured {len(ARM_TABLES) + 1:,} tables for the {arm!r} arm → {dest}")
    return dest


# ── the comparison ────────────────────────────────────────────────────────────

def _arm_dir(out_dir: Path, arm: str) -> Path:
    path = out_dir / CAPTURE_DIR / arm
    if not path.exists():
        raise FileNotFoundError(
            f"no capture for the {arm!r} arm at {path}. Both arms have to be run and "
            f"captured under the same code — see this module's docstring. Comparing one arm "
            f"against a recorded figure is what left `docs/preseason-plan.md` P5 unable to "
            f"attribute its own chain re-run.")
    return path


def _read(out_dir: Path, arm: str, name: str) -> pd.DataFrame:
    return pd.read_csv(_arm_dir(out_dir, arm) / f"{name}.csv")


def _row(block: str, measure: str, key: str, base: float, preseason: float,
         note: str = "") -> dict:
    return {"block": block, "measure": measure, "key": key,
            "base": base, "preseason": preseason, "delta": preseason - base,
            "note": note}


def _buckets(prior_mpg: pd.Series):
    cut = pd.cut(prior_mpg, list(ROLE_EDGES), labels=list(ROLE_LABELS))
    for label in ROLE_LABELS:
        yield label, (cut == label).to_numpy()


def board_rows(out_dir: Path) -> list[dict]:
    """Does the block reach the draft board — and if so, as order or as shape?

    A ranking-based strategy consumes the board's ORDER. If two arms rank the same players
    the same way, the sweep downstream cannot separate them by more than Monte Carlo noise,
    and a contest null is *explained* rather than merely underpowered. So the order
    statistics and the shape statistics are separate rows and are never summed.

    This is the block the preseason round needs most. The composition's own gate measured its
    gain at the allocation mean, so unlike the mixture — whose claim was about tails — this
    arm has a direct mechanism for moving a ranking, and these rows are where it either shows
    up or does not.
    """
    rows = []
    a_all = _read(out_dir, "base", "tensor_summary")
    b_all = _read(out_dir, "preseason", "tensor_summary")
    for season in SEASONS:
        a = a_all[a_all["season"] == season].set_index("player_id").sort_index()
        b = b_all[b_all["season"] == season].set_index("player_id").sort_index()
        if not a.index.equals(b.index):
            raise ValueError(f"{season}: the two arms carry different players; the "
                             f"comparison would be between different boards")
        rank_a = a["mean_total"].rank(ascending=False)
        rank_b = b["mean_total"].rank(ascending=False)
        moved = (rank_a - rank_b).abs()
        drafted = rank_a <= DRAFTED_PICKS
        top100 = set(rank_a.nsmallest(100).index) & set(rank_b.nsmallest(100).index)

        rows += [
            _row("board", "spearman_mean_total", season,
                 float("nan"), float(rank_a.corr(rank_b, method="pearson")),
                 "one number, not a pair — rank correlation BETWEEN the arms"),
            _row("board", "top100_overlap", season, float("nan"), len(top100) / 100.0,
                 "between the arms"),
            _row("board", "mean_abs_rank_move_drafted", season,
                 float("nan"), float(moved[drafted].mean()),
                 f"between the arms, over the {DRAFTED_PICKS} drafted picks"),
            _row("board", "max_abs_rank_move_drafted", season,
                 float("nan"), float(moved[drafted].max()), "between the arms"),
            # The tail of the move distribution, because a mean of ~3 ranks is compatible
            # with either "nothing moved" or "a handful of players moved a round". Those are
            # different findings and the mean alone cannot tell them apart.
            _row("board", "picks_moving_12plus", season, float("nan"),
                 float((moved[drafted] >= 12).sum()),
                 f"of {int(drafted.sum())} drafted picks — a full round of movement"),
        ]
        for label, keep in _buckets(a["prior_mpg"]):
            if not keep.any():
                continue
            for column in ("mean_total", "sd_total", "q10_total", "q90_total", "mean_gp",
                           "p_iron_man", "p_iron_man_strict", "p_disrupted"):
                rows.append(_row("draw", column, f"{season} {label}",
                                 float(a.loc[keep, column].mean()),
                                 float(b.loc[keep, column].mean()),
                                 f"mean over {int(keep.sum())} players"))
    return rows


def gate_a_rows(out_dir: Path) -> list[dict]:
    """The simulator's own marginal checks, which sit between the head and the contest.

    Gate A is the one place a block that improves prediction can be seen improving it before
    any drafting happens. If the contest rows are a null and these are not, the finding is
    that the contest cannot resolve a real prediction gain — which is a different result from
    the block not being one.
    """
    rows = []
    frames = {arm: _read(out_dir, arm, "sim_season_gate_a") for arm in ARMS}
    for season in SEASONS:
        for check, columns in (("games_played", ("pmf_total_variation",
                                                 "simulated_mean_gp")),
                               ("season_total_dk", ("mae", "crps", "bias", "r2"))):
            picked = {arm: f[(f["season"] == season) & (f["check"] == check)]
                      for arm, f in frames.items()}
            for column in columns:
                if any(column not in p.columns or p.empty for p in picked.values()):
                    continue
                rows.append(_row("gate_a", f"{check}.{column}", season,
                                 float(picked["base"][column].iloc[0]),
                                 float(picked["preseason"][column].iloc[0])))
    return rows


def _pooled(frame: pd.DataFrame, tournament: str, column: str) -> float:
    """`ship()`'s own pooling: the plain mean over the scored seasons.

    Recomputed here rather than read from `strategy_shipped.csv`, and that is what
    `REFERENCE_STRATEGY` is for. `strategy_shipped` holds whichever arm each run SELECTED,
    and the two arms need not select the same one — the shipped chain already selects
    `bracket_ev_blend30` at `88k_alley_oop` and `lineup_value_blend30` everywhere else.
    Reading that file would compare two different strategies and report the difference as the
    block's value, which is two changes in one number all over again.
    """
    hit = frame[(frame["strategy"] == REFERENCE_STRATEGY)
                & (frame["tournament"] == tournament)]
    if hit.empty:
        raise KeyError(
            f"{REFERENCE_STRATEGY!r} is not in this arm's sweep at {tournament!r}. It has to "
            f"be present in BOTH arms or the contest block is comparing two different "
            f"strategies; a silent NaN here would read as a missing figure rather than as a "
            f"broken comparison.")
    return float(hit[column].mean())


def contest_rows(out_dir: Path) -> list[dict]:
    """The Round-1 advance readout, and the resolution it has to be read against."""
    rows = []
    sweeps = {arm: _read(out_dir, arm, "strategy_sweep") for arm in ARMS}
    shipped = {arm: _read(out_dir, arm, "strategy_shipped") for arm in ARMS}
    realized = {arm: _read(out_dir, arm, "strategy_realized") for arm in ARMS}

    for tournament in list(shipped["preseason"]["tournament"]):
        for block, frames, column, note in (
                ("sim_lift", sweeps, "lift_vs_null",
                 "SELF-SCORED: each arm is measured in a world it generated"),
                ("sim_p_advance", sweeps, "p_advance",
                 "SELF-SCORED: each arm is measured in a world it generated"),
                ("realized_lift", realized, "lift_vs_null",
                 "common truth: realized box scores, 2 seasons. THE `resolution` BAR DOES "
                 "NOT APPLY — it is the simulated side's bootstrap over 500 worlds, and "
                 "this row has one world per season. Read the `realized` block instead"),
                ("realized_p_advance", realized, "p_advance",
                 "common truth: realized box scores, 2 seasons; see the `realized` block")):
            rows.append(_row("contest", block, tournament,
                             _pooled(frames["base"], tournament, column),
                             _pooled(frames["preseason"], tournament, column), note))
        pick = {arm: f[f["tournament"] == tournament] for arm, f in shipped.items()}
        rows.append(_row("contest", "selected_reference_strategy", tournament,
                         float(pick["base"]["strategy"].iloc[0] == REFERENCE_STRATEGY),
                         float(pick["preseason"]["strategy"].iloc[0] == REFERENCE_STRATEGY),
                         f"1.0 when the arm's own sweep selected {REFERENCE_STRATEGY}; "
                         f"base={pick['base']['strategy'].iloc[0]}, "
                         f"preseason={pick['preseason']['strategy'].iloc[0]}"))

    # The resolution, derived from the sweep's own bootstrap rather than asserted: the
    # per-season standard error of the reference strategy's advance probability, pooled over
    # seasons, doubled into an unpaired arm-to-arm difference, at 95%.
    for arm in ARMS:
        hit = sweeps[arm][(sweeps[arm]["strategy"] == REFERENCE_STRATEGY)
                          & (sweeps[arm]["tournament"] == "600k_shootaround")]
        se = ((hit["p_advance_hi"] - hit["p_advance_lo"]) / 3.92).to_numpy()
        pooled = float(np.sqrt((se ** 2).sum()) / len(se))
        rows.append(_row("resolution", "min_detectable_lift_gap", arm,
                         float("nan"), 1.96 * pooled * np.sqrt(2.0),
                         f"95%, 600k, {int(hit['n_sims'].iloc[0]):,} worlds x {len(se)} "
                         f"seasons; a gap under this is a null"))

    for arm in ARMS:
        hit = realized[arm][(realized[arm]["strategy"] == REFERENCE_STRATEGY)
                            & (realized[arm]["tournament"] == "600k_shootaround")]
        rows.append(_row("resolution", "realized_worlds", arm,
                         float("nan"), float(hit["n_worlds"].sum()),
                         "one realization per season — the realized side selects nothing"))
    return rows


def realized_rows(out_dir: Path) -> list[dict]:
    """The realized side reported in ITS OWN terms, because the sim bar cannot price it.

    `resolution.min_detectable_lift_gap` is a bootstrap over 500 simulated worlds. The
    realized readout has **one world per season and two seasons**, so borrowing that bar
    would stamp RESOLVED on a row whose actual uncertainty it never measured — the same
    class of mistake as a verdict asserted in prose beside numbers that do not support it.

    What the realized side *does* have is **pairing**. Both arms are scored against the
    identical box scores with the identical field, so the enormous season-to-season swing in
    the LEVEL cancels out of the DELTA. `88k_alley_oop` is the demonstration: the base arm's
    lift moves from +0.8273 to −0.1313 across the two seasons — a swing of 0.96 — while the
    arm-to-arm delta stays at +0.0037 and +0.1016.

    So the evidence here is *consistency*, not an interval, and it is reported as such: how
    many of the season x tournament cells move the same way, and how far apart the two
    seasons put each tournament's delta. Two seasons is the ceiling on the honest estimate
    and no resampling manufactures a third — `strategy.py` says so in its own output, and
    nothing in this block pretends otherwise.
    """
    realized = {arm: _read(out_dir, arm, "strategy_realized") for arm in ARMS}
    tournaments = list(realized["preseason"]["tournament"].unique())
    rows, deltas = [], []
    for tournament in tournaments:
        per_season = []
        for season in SEASONS:
            picked = {}
            for arm, frame in realized.items():
                hit = frame[(frame["strategy"] == REFERENCE_STRATEGY)
                            & (frame["tournament"] == tournament)
                            & (frame["season"] == season)]
                picked[arm] = float(hit["lift_vs_null"].iloc[0])
            per_season.append(picked["preseason"] - picked["base"])
        deltas += per_season
        rows.append(_row("realized", "season_spread", tournament,
                         float("nan"), float(abs(per_season[0] - per_season[1])),
                         f"|delta({SEASONS[0]}) - delta({SEASONS[1]})| = "
                         f"|{per_season[0]:+.4f} - {per_season[1]:+.4f}|; the paired "
                         f"delta's season-to-season stability, NOT an interval"))
    rows.append(_row("realized", "delta_positive_cells", "all tournaments",
                     float(len(deltas)), float(sum(d > 0 for d in deltas)),
                     f"of {len(deltas)} season x tournament cells; `base` column is the "
                     f"denominator. The 5 tournaments are ONE test (same worlds, same "
                     f"portfolios) so this is 2 seasons deep, not 10"))
    rows.append(_row("realized", "min_abs_delta", "all tournaments", float("nan"),
                     float(min(abs(d) for d in deltas)),
                     "the weakest cell — the claim is only as strong as this"))
    return rows


def strategy_rows(out_dir: Path, tournament: str = "600k_shootaround") -> list[dict]:
    """The control that turns a plausible contest gain into a measured world artifact.

    The five tournaments are **one test, not five** — same worlds, same portfolios, differing
    only in pod size and payout — so a lift that moves the same way in all five says nothing
    about replication. The sweep's 24 strategies are the axis that can discriminate, because
    they consume the board differently, and one of them consumes it not at all:

    - **`adp` ranks on the market alone.** Its board is byte-identical across the two arms, so
      whatever its lift does is a property of the *world* each arm generated and cannot be a
      drafting gain. That figure is the floor any model-based arm's gain has to clear before
      it means anything.
    - **The spread across all 24** is the noise scale at this world count.
    - **The ordering** is what the sweep actually *decides*. Verdicts survive a shifted level;
      they do not survive a reordering.
    """
    rows = []
    sweeps = {arm: _read(out_dir, arm, "strategy_sweep") for arm in ARMS}
    pooled = {}
    for arm, frame in sweeps.items():
        hit = frame[frame["tournament"] == tournament]
        pooled[arm] = hit.groupby("strategy")["lift_vs_null"].mean()
    base, preseason = pooled["base"], pooled["preseason"]
    if not base.index.equals(preseason.index):
        raise ValueError("the two arms swept different strategy sets")
    delta = preseason - base

    rows += [
        _row("strategy", "lift_delta_mean", tournament, float("nan"), float(delta.mean()),
             f"over {len(delta)} strategies — the level shift, not a drafting gain"),
        _row("strategy", "lift_delta_sd", tournament, float("nan"), float(delta.std()),
             "the noise scale across strategies whose boards are ~0.999 correlated"),
        _row("strategy", "lift_delta_positive", tournament, float("nan"),
             float((delta > 0).sum()), f"of {len(delta)}"),
        _row("strategy", "adp_only_lift", tournament, float(base["adp"]),
             float(preseason["adp"]),
             "THE CONTROL: `adp` never reads the model, so its board is identical across "
             "arms and its delta is pure world effect"),
        _row("strategy", "reference_lift_z", tournament, float("nan"),
             float((delta[REFERENCE_STRATEGY] - delta.mean()) / delta.std()),
             f"{REFERENCE_STRATEGY}'s delta in sds of the across-strategy spread"),
        _row("strategy", "ordering_spearman", tournament, float("nan"),
             float(base.rank().corr(preseason.rank())),
             "between the arms — what the sweep decides, rather than its level"),
        _row("strategy", "top_strategy_matches", tournament, float("nan"),
             float(base.idxmax() == preseason.idxmax()),
             f"base={base.idxmax()}, preseason={preseason.idxmax()}"),
    ]
    return rows


def verdict_rows(out_dir: Path) -> list[dict]:
    """Every gate the sweep carries, so a flipped verdict cannot hide behind a tie."""
    rows = []
    gate_d = {arm: _read(out_dir, arm, "strategy_gate_d") for arm in ARMS}
    rows.append(_row("verdict", "gate_d_materially_different", "all tournaments",
                     float(gate_d["base"]["materially_different"].sum()),
                     float(gate_d["preseason"]["materially_different"].sum()),
                     f"of {len(gate_d['preseason'])} paired comparisons; 0 is Gate D "
                     f"failing, which is itself a result"))
    # The denominator, carried as its own row. A count of zero is the one shape of result
    # that decays silently — the zero stays true while the denominator moves underneath it —
    # so "0 of 6" is two claims here rather than one string.
    rows.append(_row("verdict", "gate_d_comparisons", "all tournaments",
                     float(len(gate_d["base"])), float(len(gate_d["preseason"])),
                     "the denominator behind that zero"))
    injection = {arm: _read(out_dir, arm, "strategy_injection") for arm in ARMS}
    for season in SEASONS:
        pick = {arm: f[f["season"] == season] for arm, f in injection.items()}
        rows.append(_row("verdict", "injection_rho", season,
                         float(pick["base"]["rho"].iloc[0]),
                         float(pick["preseason"]["rho"].iloc[0]),
                         "each arm solves its own — Gate C's rotation is per-world"))
    return rows


def reach_rows(out_dir: Path) -> list[dict]:
    """Which heads the two arms actually differ on, read off the captured provenance.

    The block sits on three heads and only two of them are in the draw path, so an
    arm-to-arm difference in the persisted head is the auditable trace that a refit landed
    rather than a posterior being carried over. A head that reads identical on both arms did
    not move, and every number downstream would be measuring the other two.

    **It takes TWO traces to see that, because the three heads carry the block differently
    and each one is invisible in the other's trace.** Availability's block is ten columns on
    `beta` at a window that does not move (2012-13 either way), so only `n_features` shows
    it: 29 against 19. The composition's is a blend into `w_share` — the `OWN` feature's
    *value*, the offset and the allocation order — which adds no columns at all, so
    `n_features` is 25 in both arms and only the window shows it: 2004-05 against 1996-97.
    Checking either alone would report one of the two heads as never having refitted.
    """
    stamps = {arm: _read(out_dir, arm, "provenance") for arm in ARMS}
    rows = []
    for head in ("availability", "minutes", "composition"):
        seasons = {arm: str(stamps[arm][f"{head}_first_season"].iloc[0]) for arm in ARMS}
        features = {arm: float(stamps[arm][f"{head}_n_features"].iloc[0]) for arm in ARMS}
        moved = (seasons["base"] != seasons["preseason"]
                 or features["base"] != features["preseason"])
        note = (f"base={seasons['base']}/{features['base']:.0f} features, "
                f"preseason={seasons['preseason']}/{features['preseason']:.0f} features")
        if head == "minutes":
            note += ("; this head REACHES NOTHING BELOW — src/sim/ never loads its "
                     "artifact, so its block is invisible to every row after this one")
        rows.append(_row("reach", "refit_landed", head,
                         float("nan"), float(moved),
                         f"1.0 means the persisted head differs between the arms. {note}"))
        rows.append(_row("reach", "n_features", head,
                         features["base"], features["preseason"],
                         "the block's trace on availability and minutes; the composition "
                         "blends `w_share` and adds no columns, so its delta is 0 by design"))
    for key in KEY_NAMES:
        rows.append(_row("reach", "config_key", key,
                         float(stamps["base"][key].iloc[0]),
                         float(stamps["preseason"][key].iloc[0]),
                         "the three keys that ARE the block"))
    return rows


def build(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    rows = (reach_rows(out_dir) + board_rows(out_dir) + gate_a_rows(out_dir)
            + contest_rows(out_dir) + realized_rows(out_dir) + strategy_rows(out_dir)
            + verdict_rows(out_dir))
    frame = pd.DataFrame(rows)
    stamps = pd.concat([_read(out_dir, arm, "provenance") for arm in ARMS],
                       ignore_index=True)
    for _, stamp in stamps.iterrows():
        frame[f"{stamp['arm']}_captured_at"] = stamp["captured_at"]
    return frame


def _report(frame: pd.DataFrame) -> None:
    bar = float(frame[(frame["block"] == "resolution")
                      & (frame["measure"] == "min_detectable_lift_gap")]["preseason"].max())
    print("\nThe preseason block's contest value — paired, both arms on one code")
    print(f"  the instrument resolves a lift gap of {bar:.4f} at 95%; anything under that "
          f"is a null\n")
    for block in ("reach", "board", "draw", "gate_a", "contest", "realized", "strategy",
                  "verdict"):
        picked = frame[frame["block"] == block]
        if picked.empty:
            continue
        print(f"  {block}")
        for _, row in picked.iterrows():
            resolved = ""
            # The bar is the SIMULATED side's bootstrap and is applied only there. A
            # realized row gets `paired/2sn` instead — it has one world per season, so the
            # bar never measured its uncertainty and stamping it would overclaim.
            if block == "contest" and row["measure"] == "sim_lift":
                resolved = "  RESOLVED" if abs(row["delta"]) > bar else "  (null)"
            elif block == "contest" and row["measure"] == "realized_lift":
                resolved = "  paired/2sn"
            base = "      —" if pd.isna(row["base"]) else f"{row['base']:>10.4f}"
            print(f"    {row['measure']:<34} {row['key']:<20} {base} "
                  f"{row['preseason']:>10.4f}  {row['delta']:>+9.4f}{resolved}")
        print()


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build(cfg)
    _report(frame)
    dest = out_dir / "preseason_block_contest.csv"
    frame.to_csv(dest, index=False)
    print(f"Saved {len(frame):,} paired counterfactual rows → {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The preseason block's paired contest counterfactual.")
    parser.add_argument("--capture", choices=list(ARMS), default=None,
                        help="freeze the chain currently on disk as this arm")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    if args.capture:
        capture(cfg, args.capture)
    else:
        run(cfg)
