"""What the two population changes recover of the rookie floor, at the unit that resolves.

`docs/rookie-rates-plan.md` §5h. Session 1 (§7a) priced rookie-lessness in two halves. The
half that **does not** resolve is the Round-1 lift delta: two realized seasons are two
worlds, 21 of 24 arms lost lift and the shipped arm gained, and no amount of resampling
manufactures a third season. The half that **does** resolve is the bar the opponent field
sets — averaged over `n_field_drafts` x 12 entries, it moved **+173.1** dk_pts on 2022-23
and **+111.9** on 2023-24 when the field was allowed to draft players our seat could not
price.

Sessions 2 through 7 made two of those groups priceable — the ladder's recovered returnees
and the true rookies — so this module asks the obvious next question in the same unit:
**how much of that bar rise is left**, and which change removed which part of it.

## Why this is a board ladder and not two more sweeps

The cut line is a function of **which rows the field may draft**, the ADP order it drafts
them in, and what they realized. It reads the tensor for exactly one thing — the `scorable`
mask that says which rows a board carries — and never for a value. So the decomposition is
a ladder of board masks scored by one function, not a ladder of simulated worlds:

    veteran      rung 0 of the lag design — the board every artifact before §5f was built on
    + ladder     plus `returnee_lag2`, the one rung §7c's gate admitted
    + rookie     plus the true-rookie family — today's priceable board
    unrestricted every rostered player, which is what a real field actually drafts

`cut(unrestricted) - cut(rung)` is §7a's floor measured against that rung, so the first row
reproduces Session 1's own number from a different module and the last row is the residual.
The increments in between are what the ladder and the rookie head each recover, and they
are attributable rather than inferred because the rungs are nested by construction.

**This is deliberately not a re-run of the sweep.** Running the 24-arm table three more
times would cost about three hours and produce three more readings of the quantity §7a
already showed does not resolve at N = 2. `make strategy-sweep TENSOR_LABEL=...` is the
rookie-inclusive replay of the contest half, run once, and its caveats are §7a's.

Usage:
    python -m src.sim.rookie_recovery
    python -m src.sim.rookie_recovery --season 2022-23 --tensor-label _rookieinclusive
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.sim import draft_room
from src.sim.bracket import load_tensor
from src.sim.season import (ROOKIE_FAMILY, VETERAN_FAMILY, assert_season_allowed,
                            validation_seasons)
from src.sim.strategy import (N_FIELD_DRAFTS, ROUND_ONE_CUT, SEED, field_cut_line,
                              priceable_room, realized_tensor, split_frame)

# The default variant. `docs/rookie-rates-plan.md` §7g wrote the tensors on disk before the
# two families were unioned and deliberately did not overwrite them, so the rookie-inclusive
# tensor lives beside them under a label rather than over them.
TENSOR_LABEL = "_rookieinclusive"

# `n_sims` only sizes the room's simulated arrays, which nothing here reads: every figure
# below comes off the REALIZED tensor. Kept small so the load is seconds rather than a
# minute, and named rather than defaulted so it is obvious it is not a resolution knob.
N_SIMS_UNUSED = 50

# The ladder, outermost last. Each entry is (name, the `lag_rung` labels it ADDS); a rung
# carries every label above it, which is what makes the increments attributable.
RUNGS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("veteran", ("veteran",)),
    ("ladder", ("thin_prior", "returnee_lag2", "returnee_thin", "no_usable_lag")),
    ("rookie", ("true_rookie",)),
)


def board_provenance(frame: pd.DataFrame, tensor: dict) -> pd.DataFrame:
    """`unit_family` and `lag_rung` for every BOARD row, `unpriced` where the tensor has none.

    The board is `make draft-pool`'s and the tensor is `make simulate-season`'s, and the
    first is the wider of the two — that gap *is* the floor. So the join is a left one and
    the rows it fails to match are the population under measurement rather than an error.
    """
    pos = {int(p): i for i, p in enumerate(tensor["player_id"])}
    index = frame["player_id"].map(pos)
    ok = index.notna().to_numpy()
    family = np.full(len(frame), "unpriced", dtype=object)
    rung = np.full(len(frame), "unpriced", dtype=object)
    take = index[ok].to_numpy().astype(int)
    family[ok] = tensor["unit_family"][take]
    rung[ok] = tensor["lag_rung"][take]
    return pd.DataFrame({"unit_family": family, "lag_rung": rung}, index=frame.index)


def rung_masks(prov: pd.DataFrame, scorable: np.ndarray) -> dict[str, np.ndarray]:
    """The nested board masks, plus `unrestricted`.

    Intersected with the room's own `scorable` rather than trusted from the provenance
    join, because those are two different questions — a row can carry a design and still be
    padded out of the tensor — and the sweep's restriction reads `scorable`.
    """
    scorable = np.asarray(scorable, dtype=bool)
    out: dict[str, np.ndarray] = {}
    cumulative = np.zeros(len(prov), dtype=bool)
    labels = prov["lag_rung"].to_numpy()
    for name, added in RUNGS:
        cumulative = cumulative | (np.isin(labels, added) & scorable)
        out[name] = cumulative.copy()
    out["unrestricted"] = np.ones(len(prov), dtype=bool)
    return out


def census(frame: pd.DataFrame, adp: np.ndarray, realized: np.ndarray,
           mask: np.ndarray) -> dict:
    """Board size, how much of it the market prices, and what those rows realized.

    The ADP-priced count is the one that travels, because §7b's reach figures are quoted in
    it: a long-tail roster row a draft never reaches is not worth the same as a Wembanyama.
    """
    priced = mask & np.isfinite(adp)
    return {"n_board": int(mask.sum()), "n_priced": int(priced.sum()),
            "realized_mean_priced": float(realized[priced].mean()) if priced.any()
            else float("nan"),
            "realized_total": float(realized[mask].sum())}


def run(cfg: dict, seasons: list[str] | None = None, seed: int = SEED,
        n_field_drafts: int | None = None,
        tensor_label: str = TENSOR_LABEL) -> Path:
    """One row per (season, rung): the field's realized Round-1 bar on that board."""
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_field_drafts = int(n_field_drafts or N_FIELD_DRAFTS)

    design = split_frame(cfg)
    seasons = seasons or validation_seasons(design)
    for season in seasons:
        assert_season_allowed(season, design)

    print("Rookie recovery — how much of §7a's floor the two population changes give back")
    print(f"  the tensor variant is `{tensor_label or '(shipped)'}`; every figure here is "
          f"scored on the REALIZED season")
    print(f"  Round 1 is a 2-of-12 cut in all five structures, so the bar is the "
          f"{ROUND_ONE_CUT:.4f} quantile of {n_field_drafts * 12:,} field entries")

    rows = []
    for season in seasons:
        print(f"\n── {season} ──")
        room = draft_room.load_room(cfg, season, n_sims=N_SIMS_UNUSED,
                                    tensor_label=tensor_label)
        tensor = load_tensor(features_dir, season, tensor_label)
        prov = board_provenance(room.frame, tensor)
        masks = rung_masks(prov, room.scorable)
        adp = np.asarray(room.board.adp, dtype=float)
        realized = realized_tensor(cfg, season, room.frame["player_id"].to_numpy(),
                                   len(room.round_of_period)).sum(axis=1)[:, 0]

        print("  board provenance: "
              + ", ".join(f"{k} {v:,}" for k, v in
                          prov["lag_rung"].value_counts().items()))

        cuts = {}
        for name, mask in masks.items():
            # `priceable_room` is the sweep's own restriction, reused rather than
            # reimplemented: it rebuilds the board arrays, the eligibility masks and the
            # field against the restricted population, which is the whole content of a rung.
            sub, _ = priceable_room(cfg, replace(room, scorable=mask), seed,
                                    n_field_drafts, restrict=name != "unrestricted")
            cut = field_cut_line(cfg, sub, season, seed, n_field_drafts)
            cuts[name] = cut
            rows.append({"season": season, "rung": name, "tensor_label": tensor_label}
                        | census(room.frame, adp, realized, mask) | cut)
            print(f"    {name:<13} board {int(mask.sum()):>4,} "
                  f"({int((mask & np.isfinite(adp)).sum()):>3,} priced)  "
                  f"Round-1 bar {cut['field_round1_cut']:>9,.1f}")

        top = cuts["unrestricted"]["field_round1_cut"]
        for row in rows:
            if row["season"] == season:
                row["floor_vs_unrestricted"] = top - row["field_round1_cut"]
        base = cuts["veteran"]["field_round1_cut"]
        print(f"  §7a's floor on this board ladder: "
              f"{top - base:+,.1f} dk_pts at rung 0, residual "
              f"{top - cuts['rookie']['field_round1_cut']:+,.1f} today "
              f"— the ladder gives back "
              f"{cuts['ladder']['field_round1_cut'] - base:+,.1f} and the rookie head "
              f"{cuts['rookie']['field_round1_cut'] - cuts['ladder']['field_round1_cut']:+,.1f}")

    table = pd.DataFrame(rows)
    order = {name: i for i, name in enumerate(list(dict(RUNGS)) + ["unrestricted"])}
    table = table.sort_values(["season", "rung"], key=lambda s: s.map(order)
                              if s.name == "rung" else s,
                              kind="stable").reset_index(drop=True)
    # The increment each change is worth, on the row that adds it — a reader should never
    # have to difference two rows of a CSV by hand to get the headline.
    table["recovered_vs_previous"] = table.groupby("season", sort=False)[
        "field_round1_cut"].diff().fillna(0.0)
    _report(table)

    dest = out_dir / "rookie_recovery.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} rookie recovery rows → {dest}")
    return dest


def _report(table: pd.DataFrame) -> None:
    print("\nThe floor, and what is left of it")
    print(f"  {'season':<9}{'rung':<14}{'board':>7}{'priced':>8}"
          f"{'Round-1 bar':>13}{'floor':>10}{'recovered':>11}")
    for _, r in table.iterrows():
        print(f"  {r['season']:<9}{r['rung']:<14}{r['n_board']:>7,}{r['n_priced']:>8,}"
              f"{r['field_round1_cut']:>13,.1f}{r['floor_vs_unrestricted']:>10,.1f}"
              f"{r['recovered_vs_previous']:>+11,.1f}")
    print("  `floor` is the bar an unrestricted field sets minus this rung's — Session 1's")
    print("  quantity, re-measured. `recovered` is what the rung above gave back.")
    print("  The CONTEST half of the floor is not here and does not resolve at two seasons;")
    print("  `make strategy-sweep TENSOR_LABEL=_rookieinclusive` is its rookie-inclusive replay.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to validation.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--field-drafts", type=int, default=None)
    parser.add_argument("--tensor-label", default=TENSOR_LABEL,
                        help="which tensor variant's `scorable` mask names the board")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, seed=args.seed, n_field_drafts=args.field_drafts,
        tensor_label=args.tensor_label)
