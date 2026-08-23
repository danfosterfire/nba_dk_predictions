"""`make ladder-board` — what the availability lag-recovery ladder does to the BOARD.

`docs/availability-window-plan.md` §16i priced the ladder at the head's own unit (draftable
CRPS in games, 17.6175 -> 8.3662) and at the season-total unit through
`season_total.evaluate`. Neither of those re-allocates minutes, and that is the whole gap
this module closes.

## Why the season-total reading cannot answer it

The minutes allocation is **zero-sum**: a team-game hands out `5 x game_length` minutes and
the composition splits them among the players who played. So when the ladder takes Kawhi
Leonard from the plug-in's 24.7 available games to the head's ~49.7, it does not add
minutes to the league — it moves them off his teammates, in the ~25 team-games he is now
available for, at whatever share the composition already gives him. And the composition
already gives these players real veteran shares (Jamal Murray 0.6164, Leonard 0.6031, Miles
Bridges 0.7312 against a veteran p90 of ~0.60), because their `w_share` comes from
`share_lags` rather than from the availability design the ladder widens.

`season_total.evaluate` holds games played per player and scores each row on its own, so it
is structurally blind to that transfer. `sim/season.py` is not: the same `available` mask
that decides a player's games decides which team-games he competes for minutes in.

## What this measures, and the bar

Both arms, both validation seasons, the same seed and the same sim count — the ladder is a
`build_context` input, so nothing else differs. Four readings, and the third is the gate:

1. **the recovered players** — predicted games and season total against realized, which is
   §16i's win arriving at the board;
2. **their teammates** — the players who lose the minutes, named as a population rather
   than assumed small;
3. **everyone else** — season-total MAE and CRPS against realized on the units neither
   group touches. *This must not degrade.* A ladder that fixes 6 players by making 380
   worse is not a ladder that ships;
4. **the board** — Spearman and top-k overlap against a seed-noise floor, so a rank change
   is read against the noise the tensor already carries rather than against zero.

The seed-noise floor is `forward_board`'s own device: the same context re-simulated at
`seed + 1`, which costs one more simulation and is the only thing that makes "the board
moved" a statement with a unit.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.sim.season import (FIT_WINDOW, N_SCORING_PERIODS, build_context,
                            marginal_metrics, realized_frame, simulate,
                            validation_seasons)

#: Enough sims for a stable mean-total ranking — `forward_board.N_SIMS`, and for its
#: reason: the seed-noise arm is what SAYS whether it was enough.
N_SIMS = 500
SEED = 0

#: The rung the gate admitted (`docs/availability-window-plan.md` §16i). Read as the arm
#: this module turns ON, against the shipped design with the key emptied — so the "off" arm
#: is the pre-ladder simulator exactly, whatever the config happens to say.
LADDER = ("returnee_lag2",)

ARMS = ("off", "on")


def arm_cfg(cfg: dict, arm: str) -> dict:
    """The config with the availability ladder forced on or off.

    Forced rather than read, so the table is a comparison rather than a report of whatever
    the key is set to. `copy.deepcopy` because the nested `stan.availability` block is
    shared otherwise and the second arm would inherit the first's edit.
    """
    import copy

    out = copy.deepcopy(cfg)
    out.setdefault("stan", {}).setdefault("availability", {})["lag_ladder"] = (
        list(LADDER) if arm == "on" else [])
    return out


def recovered_ids(cfg: dict, season: str) -> set:
    """The player-seasons the ladder recovers — read off the design's own `lag_rung`."""
    from src.models.availability import ladder_recovered
    from src.models.stan_availability import availability_design

    design = availability_design(arm_cfg(cfg, "on"))
    rows = design[(design["season"] == season) & ladder_recovered(design)]
    return set(rows["player_id"])


def teammate_ids(ctx: dict, recovered: set) -> set:
    """Everyone who shares a team-game with a recovered player, minus the recovered.

    Read off the simulator's OWN roster grid rather than from a roster file, because the
    grid is what the allocation runs on — a traded player belongs to both his teams there,
    and that is exactly the population whose minutes move.
    """
    grid = ctx["grid"]
    teams = set(grid.loc[grid["player_id"].isin(recovered), "team_id"])
    return set(grid.loc[grid["team_id"].isin(teams), "player_id"]) - recovered


def totals(sim: dict) -> np.ndarray:
    """`(sims x units)` season dk_pts over the contest window."""
    return sim["dk_pts"].sum(axis=1).T.astype(float)


def group_rows(season: str, arm: str, group: str, ids: set, ctx: dict, sim: dict,
               realized: pd.DataFrame) -> list[dict]:
    """One group's simulated-against-realized block, on the units it names."""
    unit_ids = ctx["unit_ids"]
    keep = np.isin(unit_ids, list(ids)) & (realized["gp"].to_numpy() > 0)
    if not keep.any():
        return []
    y = realized["dk_total"].to_numpy()[keep]
    m = marginal_metrics(totals(sim)[:, keep], y)
    gp = sim["games_played"].sum(axis=1).T.astype(float)[:, keep].mean()
    return [{"season": season, "arm": arm, "group": group, "metric": k, "value": float(v)}
            for k, v in {**m, "mean_predicted_gp": gp,
                         "mean_realized_gp": float(realized["gp"].to_numpy()[keep].mean()),
                         "mean_realized_total": float(y.mean()),
                         "mean_predicted_total": float(totals(sim)[:, keep].mean())}.items()]


def board(ctx: dict, sim: dict) -> pd.DataFrame:
    """`forward_board.board_frame`'s ranking, rebuilt here so this module reads alone."""
    total = totals(sim).mean(axis=0)
    out = pd.DataFrame({"player_id": ctx["unit_ids"], "mean_total": total})
    out["rank"] = out["mean_total"].rank(ascending=False, method="first").astype(int)
    return out


def compare(a: pd.DataFrame, b: pd.DataFrame, season: str, arm: str) -> list[dict]:
    """Board `b` against board `a` — Spearman and top-k overlap on the shared units."""
    from scipy.stats import spearmanr

    merged = a.merge(b, on="player_id", suffixes=("_a", "_b"))
    row = {"spearman": float(spearmanr(merged["mean_total_a"],
                                       merged["mean_total_b"]).statistic),
           "n_shared": float(len(merged)),
           "mean_abs_total_diff": float((merged["mean_total_a"]
                                         - merged["mean_total_b"]).abs().mean()),
           "max_rank_move": float((merged["rank_a"] - merged["rank_b"]).abs().max())}
    for k in (16, 48, 100):
        row[f"overlap_{k}"] = float(len(set(a.nsmallest(k, "rank")["player_id"])
                                        & set(b.nsmallest(k, "rank")["player_id"])))
    return [{"season": season, "arm": arm, "group": "board", "metric": k, "value": v}
            for k, v in row.items()]


def run(cfg: dict, seasons: list[str] | None = None, n_sims: int = N_SIMS,
        seed: int = SEED, window: str | None = None) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    window = window or str(cfg.get("sim", {}).get("fit_window", FIT_WINDOW))

    print("Availability lag-recovery ladder — what it does to the BOARD")
    print(f"  Both arms at the `{window}` window, {n_sims:,} sims, seed {seed}. The ladder "
          f"is a\n  `build_context` input, so nothing else differs between them.")
    print(f"  §16i priced this ladder on games played and on a season total that never "
          f"re-allocates\n  minutes. The allocation is zero-sum, so this is the reading "
          f"that can see the transfer.")

    from src.models.stan_components import head_design as component_head_design

    rows: list[dict] = []
    for season in seasons or validation_seasons(component_head_design(cfg)):
        recovered = recovered_ids(cfg, season)
        print(f"\n── {season} — {len(recovered)} recovered player-season(s) ──")

        sims, ctxs, boards = {}, {}, {}
        for arm in ARMS:
            ctx = build_context(arm_cfg(cfg, arm), season, window, n_sims, seed)
            sim = simulate(ctx)
            ctxs[arm], sims[arm], boards[arm] = ctx, sim, board(ctx, sim)
        # The floor: the SAME frames re-drawn. `build_context` is draw-free, so swapping
        # the seed isolates Monte-Carlo noise from the ladder — and it is scored on every
        # group, not only on the board, because the untouched units are the gate and a
        # delta there is unreadable without the noise it has to beat. They are not exactly
        # invariant even in principle: the recovered player's `available` mask changes the
        # length of the minutes-allocation row block, which shifts the stream inside
        # `simulate_minutes` for every team in the league.
        sim_noise = simulate({**ctxs["off"], "seed": seed + 1})
        noise = board(ctxs["off"], sim_noise)

        mates = teammate_ids(ctxs["off"], recovered)
        rest = set(ctxs["off"]["unit_ids"]) - recovered - mates
        print(f"  {len(recovered)} recovered, {len(mates & set(ctxs['off']['unit_ids']))} "
              f"teammates, {len(rest)} untouched units")

        realized = realized_frame(cfg, ctxs["off"])
        for arm, ctx, sim in (("off", ctxs["off"], sims["off"]),
                              ("on", ctxs["on"], sims["on"]),
                              ("seed_noise", ctxs["off"], sim_noise)):
            for group, ids in (("recovered", recovered), ("teammates", mates),
                               ("untouched", rest),
                               ("all", set(ctx["unit_ids"]))):
                rows += group_rows(season, arm, group, ids, ctx, sim, realized)
        rows += compare(boards["off"], boards["on"], season, "on_vs_off")
        rows += compare(boards["off"], noise, season, "seed_noise")

    table = pd.DataFrame(rows)
    table["n_sims"], table["seed"], table["window"] = n_sims, seed, window
    dest = out_dir / "availability_ladder_board.csv"
    table.to_csv(dest, index=False)
    _report(table)
    print(f"\nSaved {len(table):,} rows → {dest}")
    return dest


def _report(table: pd.DataFrame) -> None:
    def cell(season, arm, group, metric):
        hit = table[(table["season"] == season) & (table["arm"] == arm)
                    & (table["group"] == group) & (table["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    for season in table["season"].dropna().unique():
        print(f"\n[{season}] season-total dk_pts against realized, by group")
        print(f"{'group':<12}{'n':>5}{'MAE off':>11}{'MAE on':>11}{'Δ':>9}"
              f"{'CRPS off':>11}{'CRPS on':>11}{'Δ':>9}{'MAE noise Δ':>12}")
        for group in ("recovered", "teammates", "untouched", "all"):
            n = cell(season, "off", group, "n")
            if not np.isfinite(n):
                continue
            mo, mn = cell(season, "off", group, "mae"), cell(season, "on", group, "mae")
            co, cn = cell(season, "off", group, "crps"), cell(season, "on", group, "crps")
            nz = cell(season, "seed_noise", group, "mae") - mo
            print(f"{group:<12}{int(n):>5}{mo:>11.2f}{mn:>11.2f}{mn - mo:>+9.2f}"
                  f"{co:>11.2f}{cn:>11.2f}{cn - co:>+9.2f}{nz:>+11.2f}")
        go = cell(season, "off", "recovered", "mean_predicted_gp")
        gn = cell(season, "on", "recovered", "mean_predicted_gp")
        gr = cell(season, "off", "recovered", "mean_realized_gp")
        print(f"  recovered games played: {go:.2f} → {gn:.2f} against a realized {gr:.2f}")
        print(f"  board: Spearman {cell(season, 'on_vs_off', 'board', 'spearman'):.4f} "
              f"against a {cell(season, 'seed_noise', 'board', 'spearman'):.4f} seed-noise "
              f"floor; top-16 {int(cell(season, 'on_vs_off', 'board', 'overlap_16'))}/16 "
              f"against {int(cell(season, 'seed_noise', 'board', 'overlap_16'))}/16")


if __name__ == "__main__":
    import argparse

    from src.sim.ladder_board import run as _run

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default=None, help="one season instead of both")
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    _run(cfg, seasons=[args.season] if args.season else None, n_sims=args.sims,
         seed=args.seed)
