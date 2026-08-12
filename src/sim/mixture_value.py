"""The availability mixture's contest value — a PAIRED counterfactual, not a re-read.

`docs/availability-window-plan.md` §7k asked what the availability head's tail-calibration
win is worth in Round-1 advance probability and got an answer it had to throw away: the
sweep it compared against had been built before the window round, so `0.2107 -> 0.1890`
moved three things at once and could be attributed to none of them. This module exists so
that never happens again to this question. It holds **two arms of the same chain, captured
under the same code on the same day**, and reports them side by side.

## The procedure this module is the last step of

Two full passes, differing in exactly one config key:

    # arm A — the counterfactual
    stan.availability.mixture: false   in configs/default.yaml
    make posteriors && make simulate-season && make bracket && make draft-sim \\
        && make strategy-sweep
    python -m src.sim.mixture_value --capture single

    # arm B — the shipped head, re-run rather than remembered
    stan.availability.mixture: true
    ... the same five targets ...
    python -m src.sim.mixture_value --capture mixture

    make mixture-value

`--capture` copies the arm's CSVs into `outputs/predictions/mixture_arms/<arm>/` and
reduces its `player x scoring_period x sim` tensors to a per-player summary, because the
comparison needs both arms in memory at once and two tensors are 160 MB. **Only
`--groups availability` of `make posteriors` has to be re-fitted** — the mixture is one
head and the other nineteen are untouched — which is what makes an arm an hour rather
than a day.

## Why the artifact carries both arms rather than a delta against the live one

The shipped arm's figures *are* live in `outputs/predictions/`, so a delta table would be
half-auditable. But the whole failure §7k recorded is a comparison against a figure whose
vintage nobody could reconstruct, and a file holding one arm invites exactly that on the
next pass. Both arms and their `captured_at` go in together, so a stale half is visible
rather than inferable.

## What the reader has to know before reading the contest block

**The simulated lift is not arm-comparable, and the artifact says so in a row rather than
in a footnote.** Each arm's `sim_lift` is measured against a symmetric-field null *inside a
world that arm generated*, so a head with a wider predictive can score a higher lift
because its world spreads rosters further apart, not because its board drafts better. The
realized block is the only one whose truth — actual 2022-23 / 2023-24 box scores — is
common to both arms, and it is two seasons deep.

`resolution` is the other guard: the pooled standard error on an advance probability at
`sim.strategy.n_sims` worlds per season, doubled into an arm-to-arm difference and scaled
to 95%. A gap under that row is a null and must be read as one.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The arm names, and the order every table is written in: counterfactual first, shipped
# second, so `delta` reads "what the mixture did".
ARMS = ("single", "mixture")
CAPTURE_DIR = "mixture_arms"

# The CSVs an arm is captured from. Every one is written by the five-target chain above;
# a missing file means the arm was captured mid-run and `--capture` refuses.
ARM_TABLES = ("sim_season_gate_a", "bracket_structure", "bracket_null", "bracket_entries",
              "draft_gate_b", "draft_field", "strategy_gate_c", "strategy_injection",
              "strategy_null", "strategy_sweep", "strategy_paired", "strategy_gate_d",
              "strategy_realized", "strategy_shipped")

# The seasons the sweep scores. Validation, and `held_out.selection_split` is what put
# them there — this module never widens them.
SEASONS = ("2022-23", "2023-24")

# The shipped strategy, held fixed across arms so the contest block compares the same
# object. Reading each arm's own selection instead would confound "the mixture drafts
# better" with "the mixture selected a different arm", and the selection itself is
# reported separately as `shipped_strategy`.
REFERENCE_STRATEGY = "lineup_value_blend30"

# Prior-minutes-per-game role buckets, matching `src/eda/season_effects.ROLE_EDGES` — the
# same grading the head's own dispersion carries, so a per-bucket row here lines up with
# the `rho` table in `docs/availability-window-plan.md` §7i.
ROLE_EDGES = (-0.01, 12.0, 24.0, 30.0, 99.0)
ROLE_LABELS = ("<12 mpg", "12-24", "24-30", "30+ mpg")

# The upper shoulder, at the tensor's OWN denominator. Twenty scoring periods span ~76
# games rather than 82, so 82-game thresholds do not transfer: `gp >= 70` is this window's
# "missed <= 6" and `gp >= 75` its "missed <= 1". Both are carried because they answer
# different questions — the wide one is the region §7f found the head over-predicts by
# +0.0480, and the strict one is where a star's iron-man season actually lives, which is
# the half a mixture moves most.
IRON_MAN_GP = 70
IRON_MAN_STRICT_GP = 75
DISRUPTED_GP = 41


# ── capture ───────────────────────────────────────────────────────────────────

def tensor_summary(features_dir: Path, season: str) -> pd.DataFrame:
    """One row per player: the season total's shape, and the games-played tails.

    The tensor is reduced here rather than compared here because the comparison needs both
    arms at once. `sd` and `q10` are the quantities the mixture's claim is about — a
    star's healthy season gets tighter and his disruption becomes a separate event — and
    the mean is carried so the board ranking can be rebuilt without the tensor.
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
    """Freeze the arm currently on disk into `outputs/predictions/mixture_arms/<arm>/`."""
    if arm not in ARMS:
        raise KeyError(f"unknown arm {arm!r}; choose from {ARMS}")
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    dest = out_dir / CAPTURE_DIR / arm
    dest.mkdir(parents=True, exist_ok=True)

    mixture = bool(cfg.get("stan", {}).get("availability", {}).get("mixture", True))
    if mixture != (arm == "mixture"):
        raise ValueError(
            f"`stan.availability.mixture` is {mixture} and you are capturing the {arm!r} "
            f"arm. That mismatch is the whole failure this module exists to prevent — "
            f"fix the config or the arm name, and re-run the chain if the artifacts on "
            f"disk were built under the other setting.")

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

    stamp = pd.DataFrame([{
        "arm": arm, "mixture": mixture,
        "captured_at": pd.Timestamp.now().isoformat(timespec="seconds"),
        "first_season": cfg.get("stan", {}).get("availability", {}).get("first_season"),
        "role_rho": cfg.get("stan", {}).get("availability", {}).get("role_rho"),
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
            f"captured under the same code — see this module's docstring. Comparing one "
            f"arm against a recorded figure is what produced the confound in "
            f"docs/availability-window-plan.md §7k.")
    return path


def _read(out_dir: Path, arm: str, name: str) -> pd.DataFrame:
    return pd.read_csv(_arm_dir(out_dir, arm) / f"{name}.csv")


def _row(block: str, measure: str, key: str, single: float, mixture: float,
         note: str = "") -> dict:
    return {"block": block, "measure": measure, "key": key,
            "single": single, "mixture": mixture, "delta": mixture - single,
            "note": note}


def board_rows(out_dir: Path) -> list[dict]:
    """Does the head reach the draft board — and if so, as order or as shape?

    A ranking-based strategy consumes the board's ORDER. If two arms rank the same players
    the same way, the sweep downstream cannot separate them by more than Monte Carlo
    noise, and a contest null is explained rather than merely underpowered. So the order
    statistics and the shape statistics are reported as separate rows and never summed.
    """
    rows = []
    s = _read(out_dir, "single", "tensor_summary")
    m = _read(out_dir, "mixture", "tensor_summary")
    for season in SEASONS:
        a = s[s["season"] == season].set_index("player_id").sort_index()
        b = m[m["season"] == season].set_index("player_id").sort_index()
        if not a.index.equals(b.index):
            raise ValueError(f"{season}: the two arms carry different players; the "
                             f"comparison would be between different boards")
        rank_a = a["mean_total"].rank(ascending=False)
        rank_b = b["mean_total"].rank(ascending=False)
        moved = (rank_a - rank_b).abs()
        drafted = rank_a <= 192                      # 12 entries x 16 rounds
        top100 = set(rank_a.nsmallest(100).index) & set(rank_b.nsmallest(100).index)

        rows += [
            _row("board", "spearman_mean_total", season,
                 float("nan"), float(rank_a.corr(rank_b, method="pearson")),
                 "one number, not a pair — rank correlation BETWEEN the arms"),
            _row("board", "top100_overlap", season, float("nan"), len(top100) / 100.0,
                 "between the arms"),
            _row("board", "mean_abs_rank_move_drafted", season,
                 float("nan"), float(moved[drafted].mean()),
                 "between the arms, over the 192 drafted picks"),
            _row("board", "max_abs_rank_move_drafted", season,
                 float("nan"), float(moved[drafted].max()), "between the arms"),
        ]
        for label, keep in _buckets(a["prior_mpg"]):
            if not keep.any():
                continue
            for measure, column in (("q10_total", "q10_total"), ("sd_total", "sd_total"),
                                    ("p_iron_man", "p_iron_man"),
                                    ("p_iron_man_strict", "p_iron_man_strict"),
                                    ("p_disrupted", "p_disrupted")):
                rows.append(_row("draw", measure, f"{season} {label}",
                                 float(a.loc[keep, column].mean()),
                                 float(b.loc[keep, column].mean()),
                                 f"mean over {int(keep.sum())} players"))
    return rows


def _buckets(prior_mpg: pd.Series):
    cut = pd.cut(prior_mpg, list(ROLE_EDGES), labels=list(ROLE_LABELS))
    for label in ROLE_LABELS:
        yield label, (cut == label).to_numpy()


def gate_a_rows(out_dir: Path) -> list[dict]:
    """The simulator's own marginal checks, which sit between the head and the contest."""
    rows = []
    frames = {arm: _read(out_dir, arm, "sim_season_gate_a") for arm in ARMS}
    for season in SEASONS:
        for check, columns in (("games_played", ("pmf_total_variation",
                                                 "simulated_mean_gp")),
                               ("season_total_dk", ("mae", "crps", "bias"))):
            picked = {arm: f[(f["season"] == season) & (f["check"] == check)]
                      for arm, f in frames.items()}
            for column in columns:
                rows.append(_row("gate_a", f"{check}.{column}", season,
                                 float(picked["single"][column].iloc[0]),
                                 float(picked["mixture"][column].iloc[0])))
    return rows


def _pooled(frame: pd.DataFrame, tournament: str, column: str) -> float:
    """`ship()`'s own pooling: the plain mean over the scored seasons.

    Recomputed here rather than read from `strategy_shipped.csv` for one reason, and it is
    the reason `REFERENCE_STRATEGY` exists. `strategy_shipped` holds whichever arm each run
    SELECTED, and the two arms need not select the same one — they did not, at
    `88k_alley_oop`. Reading that file would compare `lineup_value_blend30` against
    `blend_a70` and report the difference as the mixture's value, which is two changes in
    one number all over again. The selection itself is reported separately.
    """
    hit = frame[(frame["strategy"] == REFERENCE_STRATEGY)
                & (frame["tournament"] == tournament)]
    if hit.empty:
        raise KeyError(
            f"{REFERENCE_STRATEGY!r} is not in this arm's sweep at {tournament!r}. It has "
            f"to be present in BOTH arms or the contest block is comparing two different "
            f"strategies; a silent NaN here would read as a missing figure rather than as "
            f"a broken comparison.")
    return float(hit[column].mean())


def contest_rows(out_dir: Path) -> list[dict]:
    """The Round-1 advance readout, and the resolution it has to be read against."""
    rows = []
    sweeps = {arm: _read(out_dir, arm, "strategy_sweep") for arm in ARMS}
    shipped = {arm: _read(out_dir, arm, "strategy_shipped") for arm in ARMS}
    realized = {arm: _read(out_dir, arm, "strategy_realized") for arm in ARMS}

    tournaments = list(shipped["mixture"]["tournament"])
    for tournament in tournaments:
        for block, frames, column, note in (
                ("sim_lift", sweeps, "lift_vs_null",
                 "SELF-SCORED: each arm is measured in a world it generated"),
                ("sim_p_advance", sweeps, "p_advance",
                 "SELF-SCORED: each arm is measured in a world it generated"),
                ("realized_lift", realized, "lift_vs_null",
                 "common truth: realized box scores, 2 seasons"),
                ("realized_p_advance", realized, "p_advance",
                 "common truth: realized box scores, 2 seasons")):
            rows.append(_row("contest", block, tournament,
                             _pooled(frames["single"], tournament, column),
                             _pooled(frames["mixture"], tournament, column), note))
        pick = {arm: f[f["tournament"] == tournament] for arm, f in shipped.items()}
        rows.append(_row("contest", "selected_reference_strategy", tournament,
                         float(pick["single"]["strategy"].iloc[0] == REFERENCE_STRATEGY),
                         float(pick["mixture"]["strategy"].iloc[0] == REFERENCE_STRATEGY),
                         f"1.0 when the arm's own sweep selected {REFERENCE_STRATEGY}; "
                         f"single={pick['single']['strategy'].iloc[0]}, "
                         f"mixture={pick['mixture']['strategy'].iloc[0]}"))

    # The resolution, derived from the sweep's own bootstrap rather than asserted: the
    # per-season standard error of the reference strategy's advance probability, pooled
    # over seasons, doubled into an unpaired arm-to-arm difference, at 95%.
    for arm in ARMS:
        hit = sweeps[arm][(sweeps[arm]["strategy"] == REFERENCE_STRATEGY)
                          & (sweeps[arm]["tournament"] == "600k_shootaround")]
        se = ((hit["p_advance_hi"] - hit["p_advance_lo"]) / 3.92).to_numpy()
        pooled = float(np.sqrt((se ** 2).sum()) / len(se))
        rows.append(_row("resolution", "min_detectable_lift_gap", arm,
                         float("nan"), 1.96 * pooled * np.sqrt(2.0),
                         f"95%, 600k, {int(hit['n_sims'].iloc[0]):,} worlds x "
                         f"{len(se)} seasons; a gap under this is a null"))

    for arm in ARMS:
        hit = realized[arm][(realized[arm]["strategy"] == REFERENCE_STRATEGY)
                            & (realized[arm]["tournament"] == "600k_shootaround")]
        rows.append(_row("resolution", "realized_worlds", arm,
                         float("nan"), float(hit["n_worlds"].sum()),
                         "one realization per season — the realized side selects nothing"))
    return rows


def strategy_rows(out_dir: Path, tournament: str = "600k_shootaround") -> list[dict]:
    """The control that turns a plausible contest gain into a measured world artifact.

    The five tournaments are **one test, not five** — same worlds, same portfolios,
    differing only in pod size and payout — so a lift that moves the same way in all five
    says nothing about replication. The sweep's 24 strategies are the axis that can
    discriminate, because they consume the board differently, and one of them consumes it
    not at all:

    - **`adp` ranks on the market alone.** Its board is byte-identical across the two arms,
      so whatever its lift does is a property of the *world* each arm generated and cannot
      be a drafting gain. That figure is the floor any model-based arm's gain has to clear
      before it means anything.
    - **The spread across all 24** is the noise scale at this world count. A shift that is
      uniform across strategies is the world moving under all of them; a shift concentrated
      on the strategies that read the head would be the head arriving.
    - **The ordering** is what the sweep actually *decides*. Verdicts survive a shifted
      level; they do not survive a reordering.
    """
    rows = []
    sweeps = {arm: _read(out_dir, arm, "strategy_sweep") for arm in ARMS}
    pooled = {}
    for arm, frame in sweeps.items():
        hit = frame[frame["tournament"] == tournament]
        pooled[arm] = hit.groupby("strategy")["lift_vs_null"].mean()
    single, mixture = pooled["single"], pooled["mixture"]
    if not single.index.equals(mixture.index):
        raise ValueError("the two arms swept different strategy sets")
    delta = mixture - single

    rows += [
        _row("strategy", "lift_delta_mean", tournament, float("nan"), float(delta.mean()),
             f"over {len(delta)} strategies — the level shift, not a drafting gain"),
        _row("strategy", "lift_delta_sd", tournament, float("nan"), float(delta.std()),
             "the noise scale across strategies whose boards are ~0.999 correlated"),
        _row("strategy", "lift_delta_positive", tournament, float("nan"),
             float((delta > 0).sum()), f"of {len(delta)}"),
        _row("strategy", "adp_only_lift", tournament, float(single["adp"]),
             float(mixture["adp"]),
             "THE CONTROL: `adp` never reads the model, so its board is identical across "
             "arms and its delta is pure world effect"),
        _row("strategy", "reference_lift_z", tournament, float("nan"),
             float((delta[REFERENCE_STRATEGY] - delta.mean()) / delta.std()),
             f"{REFERENCE_STRATEGY}'s delta in sds of the across-strategy spread"),
        _row("strategy", "ordering_spearman", tournament, float("nan"),
             float(single.rank().corr(mixture.rank())),
             "between the arms — what the sweep decides, rather than its level"),
        _row("strategy", "top_strategy_matches", tournament, float("nan"),
             float(single.idxmax() == mixture.idxmax()),
             f"single={single.idxmax()}, mixture={mixture.idxmax()}"),
    ]
    return rows


def verdict_rows(out_dir: Path) -> list[dict]:
    """Every gate the sweep carries, so a flipped verdict cannot hide behind a tie."""
    rows = []
    gate_d = {arm: _read(out_dir, arm, "strategy_gate_d") for arm in ARMS}
    rows.append(_row("verdict", "gate_d_materially_different", "all tournaments",
                     float(gate_d["single"]["materially_different"].sum()),
                     float(gate_d["mixture"]["materially_different"].sum()),
                     f"of {len(gate_d['mixture'])} paired comparisons; 0 is Gate D "
                     f"failing, which is itself a result"))
    # The denominator, carried as its own row. A count of zero is the one shape of result
    # that decays silently — the zero stays true while the denominator moves underneath
    # it — so "0 of 6" is two claims here rather than one string.
    rows.append(_row("verdict", "gate_d_comparisons", "all tournaments",
                     float(len(gate_d["single"])), float(len(gate_d["mixture"])),
                     "the denominator behind that zero"))
    injection = {arm: _read(out_dir, arm, "strategy_injection") for arm in ARMS}
    for season in SEASONS:
        pick = {arm: f[f["season"] == season] for arm, f in injection.items()}
        rows.append(_row("verdict", "injection_rho", season,
                         float(pick["single"]["rho"].iloc[0]),
                         float(pick["mixture"]["rho"].iloc[0]),
                         "each arm solves its own — Gate C's rotation is per-world"))
    return rows


def build(cfg: dict) -> pd.DataFrame:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    rows = (board_rows(out_dir) + gate_a_rows(out_dir) + contest_rows(out_dir)
            + strategy_rows(out_dir) + verdict_rows(out_dir))
    frame = pd.DataFrame(rows)
    stamps = pd.concat([_read(out_dir, arm, "provenance") for arm in ARMS],
                       ignore_index=True)
    for _, stamp in stamps.iterrows():
        frame[f"{stamp['arm']}_captured_at"] = stamp["captured_at"]
    return frame


def _report(frame: pd.DataFrame) -> None:
    bar = float(frame[(frame["block"] == "resolution")
                      & (frame["measure"] == "min_detectable_lift_gap")]["mixture"].max())
    print("\nThe availability mixture's contest value — paired, both arms on one code")
    print(f"  the instrument resolves a lift gap of {bar:.4f} at 95%; anything under "
          f"that is a null\n")
    for block in ("board", "draw", "gate_a", "contest", "strategy", "verdict"):
        picked = frame[frame["block"] == block]
        if picked.empty:
            continue
        print(f"  {block}")
        for _, row in picked.iterrows():
            resolved = ""
            if block == "contest" and row["measure"] in ("sim_lift", "realized_lift"):
                resolved = "  RESOLVED" if abs(row["delta"]) > bar else "  (null)"
            single = "      —" if pd.isna(row["single"]) else f"{row['single']:>10.4f}"
            print(f"    {row['measure']:<34} {row['key']:<20} {single} "
                  f"{row['mixture']:>10.4f}  {row['delta']:>+9.4f}{resolved}")
        print()


def run(cfg: dict) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build(cfg)
    _report(frame)
    dest = out_dir / "availability_mixture_contest.csv"
    frame.to_csv(dest, index=False)
    print(f"Saved {len(frame):,} paired counterfactual rows → {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The availability mixture's paired contest counterfactual.")
    parser.add_argument("--capture", choices=list(ARMS), default=None,
                        help="freeze the chain currently on disk as this arm")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    if args.capture:
        capture(cfg, args.capture)
    else:
        run(cfg)
