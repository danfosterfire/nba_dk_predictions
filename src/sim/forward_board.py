"""The forward path's acceptance test — a draft board built without the season's games.

`docs/final-evaluation-plan.md` §6 verified every forward design INPUT against its
retrospective twin, frame by frame: availability (Part A), population (Part B), the
composition's per-player frame (Part D). This module is §6h, the test that matters —
push the forward inputs all the way through `sim/season.py` to the tensor and the board
ranking it implies, on a **played** season, and compare against the board the
retrospective path produces from the same posteriors.

Everything downstream of the tensor — the draft, the bracket, the strategy sweep — is a
pure function of the tensor plus market data that is identical across the two paths, so
the tensor-level ranking IS the board comparison; nothing later can disagree if this
agrees.

## The yardstick is seed noise, not zero

The two paths run the same sampler draws through the same arithmetic, but not on
identical frames: the forward population is the roster snapshot's (cut at the opener —
the rehearsal's honesty knob, `forward_design.roster_members`), `team_games` cannot see
trades, and the composition weights differ on the classified draft-number set. So the
right question is not "are the boards identical" but "is the forward-versus-retro gap
the size of the gap between two retro runs at different seeds". A third simulation —
retro, same frames, seed+1 — supplies that floor.

## Window discipline

The season is a VALIDATION season and the posteriors are the `train` window, exactly as
`make simulate-season` pairs them — `posteriors.require_window` raises on anything
wider. Nothing here touches the test seasons or the `full` window.

Usage:
    python -m src.sim.forward_board                     # 2023-24, 500 sims
    python -m src.sim.forward_board --season 2022-23 --sims 250
"""

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _slug, nbastats_dir
from src.features.availability import build_season_panel, season_availability
from src.features.forward_design import (forward_availability_design,
                                         forward_component_design,
                                         forward_composition_players, roster_members,
                                         schedule_team_games, synthetic_game_log)
from src.models.availability import season_start_dates
from src.models.stan_availability import head_design as availability_head_design
from src.models.stan_components import head_design as component_head_design
from src.models.stan_composition import head_frame
from src.sim.season import (FIT_WINDOW, SEED, build_context, rookie_design,
                            simulate)

#: Enough sims for a stable mean-total ranking; the seed-noise arm is what SAYS whether
#: it was enough, which is why the floor is measured rather than assumed.
N_SIMS = 500


def forward_context(cfg: dict, season: str, window: str, n_sims: int, seed: int,
                    real_design: pd.DataFrame | None = None,
                    real_availability: pd.DataFrame | None = None,
                    members: pd.DataFrame | None = None) -> dict:
    """`build_context` fed from forward-built frames for `season`.

    Four injections, one per harvested artifact: the synthetic panel under the roster
    grid, the forward availability design under the availability head, the forward
    component design under the eleven rate heads, and the forward per-player frame under
    the composition. Each goes through the SAME preseason attachment as its
    retrospective twin (`head_design(design=...)`), so what differs is where the rows
    came from and never what was done to them.

    Only `season`'s rows are forward-built; the other seasons' design rows come from the
    real builders (`real_design` / `real_availability`, rebuilt here when not passed).
    That is the production shape — a 2026-27 board swaps one season's rows into designs
    every played season already has — and in a rehearsal it is also load-bearing: a
    targets frame with `season` cut out erases season+1's design rows entirely (their
    `total_minutes_lag1` is gone), and a missing season SHIFTS `selection_split`'s
    walk-forward labels, so the guard would read the rehearsal season as test and raise.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = cfg["data"]["raw_dir"]

    real_panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    played_target = season in set(real_panel["season"].unique())
    before = (pd.Timestamp(season_start_dates(real_panel)[season])
              if played_target else None)
    if members is None:
        members = roster_members(season, raw_dir, before=before)
        print(f"  membership: {len(members):,} players off the roster snapshot"
              + (f", cut at the {before:%Y-%m-%d} opener (rehearsal knob)"
                 if before is not None else ""))
    else:
        print(f"  membership: {len(members):,} players, population held fixed at the "
              f"retro grid's own — the decomposition arm, not the production case")

    # The synthetic panel, built in a scratch raw dir so the CUT roster is the membership
    # the log is crossed with — `synthetic_game_log` reads the roster file as it stands,
    # and the real one carries the arrivals the cut just removed.
    schedule_team_games(season, raw_dir)          # cache the schedule beside the real raw
    with tempfile.TemporaryDirectory() as tmp:
        tmp_nba = nbastats_dir(tmp)
        tmp_nba.mkdir(parents=True, exist_ok=True)
        shutil.copy(nbastats_dir(raw_dir) / f"schedule_teams_{_slug(season)}.csv",
                    tmp_nba / f"schedule_teams_{_slug(season)}.csv")
        pd.DataFrame({"PLAYER_ID": members["player_id"],
                      "PLAYER": members["player_id"].astype(str),
                      "TeamID": members["team_id"]}).to_csv(
            tmp_nba / f"team_rosters_{_slug(season)}.csv", index=False)
        log, _ = synthetic_game_log(season, tmp,
                                    dest=tmp_nba / f"game_logs_{_slug(season)}.csv")
        fwd_panel = build_season_panel(season, tmp)
    print(f"  synthetic panel: {len(fwd_panel):,} rostered player-games, "
          f"{fwd_panel['player_id'].nunique():,} players")

    panel_full = pd.concat([real_panel[real_panel["season"] != season],
                            fwd_panel[real_panel.columns.intersection(
                                fwd_panel.columns)]], ignore_index=True)
    frame_fwd = season_availability(panel_full, "full")
    avail_design = forward_availability_design(cfg, frame_fwd, panel_full, season,
                                               members[["player_id", "team_id"]])
    avail_fwd = availability_head_design(cfg, design=avail_design)
    if real_availability is None:
        real_availability = availability_head_design(cfg)
    avail_head = pd.concat(
        [real_availability[real_availability["season"] != season],
         avail_fwd[avail_fwd["season"] == season]], ignore_index=True)
    n_av = int((avail_fwd["season"] == season).sum())
    print(f"  availability design: {n_av:,} forward rows through the preseason block")

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    targets_cut = targets[targets["season"] != season] if played_target else targets
    comp_design = forward_component_design(cfg, season, targets=targets_cut,
                                           whole_frame=True, log=log)
    comp_fwd = component_head_design(cfg, design=comp_design)
    if real_design is None:
        real_design = component_head_design(cfg)
    comp_head = pd.concat([real_design[real_design["season"] != season],
                           comp_fwd[comp_fwd["season"] == season]],
                          ignore_index=True)
    n_units = int((comp_fwd["season"] == season).sum())
    print(f"  component design: {n_units:,} forward rows through the preseason block")

    players = forward_composition_players(cfg, season, members[["player_id"]],
                                          avail_design)
    print(f"  composition frame: {len(players):,} per-player rows")

    # **The rookie design's TARGET-SEASON rows are withheld, and that is a leak guard.**
    # `rookie_rates.build_design` carves its population out of `component_targets`, so
    # membership itself — "he played in the NBA this season" — is target-season
    # information, which is exactly what a forward context may not read. A season nobody
    # has played has no rows there anyway, so this changes nothing for a real 2026-27
    # board and everything for the rehearsal on a played one. The forward rookie tier is
    # `docs/rookie-rates-plan.md` §5g; until it exists a forward board carries no rookies,
    # which is the state every board this project has produced.
    rookie_head = rookie_design(cfg)
    withheld = int((rookie_head["season"] == season).sum())
    rookie_head = rookie_head[rookie_head["season"] != season]
    print(f"  rookie design: {withheld:,} target-season rows WITHHELD — this arm has no "
          f"forward rookie tier yet\n    (docs/rookie-rates-plan.md §5g), and their "
          f"membership is target-season information. The\n    population-held-fixed arm "
          f"is held fixed on the ROSTER and cannot fix these.")
    return build_context(cfg, season, window, n_sims, seed, composition=players,
                         design=comp_head, availability=avail_head, panel=fwd_panel,
                         rookie=rookie_head)


def board_frame(ctx: dict, sim: dict) -> pd.DataFrame:
    """One row per scorable unit: mean simulated season total, and the rank it buys.

    The mean is over the contest window the tensor covers, which is what every consumer
    of the tensor ranks on before the market enters. Rank 1 is the best player.
    """
    totals = sim["dk_pts"].sum(axis=1)                      # units x sims
    out = pd.DataFrame({"player_id": ctx["unit_ids"],
                        "mean_total": totals.mean(axis=1),
                        "sd_total": totals.std(axis=1)})
    out["rank"] = out["mean_total"].rank(ascending=False, method="first").astype(int)
    return out.sort_values("rank").reset_index(drop=True)


def compare_boards(a: pd.DataFrame, b: pd.DataFrame, arm: str) -> dict:
    """Board `b` against reference board `a`, on the players both of them rank."""
    from scipy.stats import spearmanr

    merged = a.merge(b, on="player_id", suffixes=("_ref", "_alt"))
    rho = float(spearmanr(merged["mean_total_ref"], merged["mean_total_alt"]).statistic)
    row = {"arm": arm, "n_shared": len(merged),
           "n_ref_only": int((~a["player_id"].isin(b["player_id"])).sum()),
           "n_alt_only": int((~b["player_id"].isin(a["player_id"])).sum()),
           "spearman": rho,
           "mean_abs_total_diff": float((merged["mean_total_ref"]
                                         - merged["mean_total_alt"]).abs().mean())}
    for k in (16, 48, 100):
        top_ref = set(a.nsmallest(k, "rank")["player_id"])
        top_alt = set(b.nsmallest(k, "rank")["player_id"])
        row[f"overlap_{k}"] = len(top_ref & top_alt)
    missing = a[~a["player_id"].isin(b["player_id"])]
    row["missing_top100"] = int((missing["rank"] <= 100).sum())
    row["best_missing_rank"] = (int(missing["rank"].min()) if len(missing)
                                else np.nan)
    return row


def run(cfg: dict, season: str = "2023-24", window: str | None = None,
        n_sims: int = N_SIMS, seed: int = SEED) -> Path:
    window = window or str(cfg.get("sim", {}).get("fit_window", FIT_WINDOW))
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Forward board — {season} at the `{window}` window, {n_sims:,} sims, "
          f"seed {seed}")
    print(f"  The forward arm may not read this season's game log; the retro arm is "
          f"`make simulate-season`'s own path.")

    print(f"\n[retrospective context]")
    real_design = component_head_design(cfg)
    real_availability = availability_head_design(cfg)
    ctx_r = build_context(cfg, season, window, n_sims, seed,
                          composition=head_frame(cfg), design=real_design,
                          availability=real_availability)
    print(f"  {ctx_r['n_units']:,} scorable units over {ctx_r['n_games']:,} games")
    sim_r = simulate(ctx_r)
    board_r = board_frame(ctx_r, sim_r)

    # The seed-noise floor: identical frames, different draws. `build_context` is draw-free
    # so the context is reused with only the seed swapped.
    sim_n = simulate({**ctx_r, "seed": seed + 1})
    board_n = board_frame(ctx_r, sim_n)

    print(f"\n[forward context — the production shape, snapshot membership]")
    ctx_f = forward_context(cfg, season, window, n_sims, seed,
                            real_design=real_design,
                            real_availability=real_availability)
    print(f"  {ctx_f['n_units']:,} scorable units over {ctx_f['n_games']:,} games")
    sim_f = simulate(ctx_f)
    board_f = board_frame(ctx_f, sim_f)

    # The decomposition arm: the same forward machinery on the RETRO grid's own
    # population, so the population bound and the mechanics separate. The minutes
    # allocation is zero-sum, so a missing rotation player does not just vanish — he
    # hands his minutes to every shared teammate, and only this arm can say how much of
    # the headline gap is that redistribution rather than the forward frames themselves.
    print(f"\n[forward context — population held fixed]")
    grid_members = (ctx_r["grid"].drop_duplicates("player_id")
                    [["player_id", "team_id"]].reset_index(drop=True))
    ctx_x = forward_context(cfg, season, window, n_sims, seed,
                            real_design=real_design,
                            real_availability=real_availability,
                            members=grid_members)
    print(f"  {ctx_x['n_units']:,} scorable units over {ctx_x['n_games']:,} games")
    sim_x = simulate(ctx_x)
    board_x = board_frame(ctx_x, sim_x)

    rows = [compare_boards(board_r, board_n, "retro_vs_retro_seed_noise"),
            compare_boards(board_r, board_x, "forward_fixed_population_vs_retro"),
            compare_boards(board_r, board_f, "forward_vs_retro")]
    table = pd.DataFrame(rows)

    print(f"\n[the comparison — forward against retro, with seed noise as the floor]")
    print(table.round(4).to_string(index=False))
    noise, fixed, fwd = rows[0], rows[1], rows[2]
    print(f"\n  Spearman: {fwd['spearman']:.4f} forward-vs-retro against "
          f"{fixed['spearman']:.4f} with the population held fixed and "
          f"{noise['spearman']:.4f} seed-vs-seed on the same frames")
    print(f"  top-16 overlap: {fwd['overlap_16']} / 16 forward against "
          f"{noise['overlap_16']} / 16 noise; top-100: {fwd['overlap_100']} / 100 "
          f"against {noise['overlap_100']} / 100")
    print(f"  population: the forward board is missing {fwd['n_ref_only']} of the "
          f"retro board's players ({fwd['missing_top100']} inside its top 100"
          + (f", best at rank {fwd['best_missing_rank']}" if fwd["n_ref_only"] else "")
          + f") and carries {fwd['n_alt_only']} it does not have — the Part B bound, "
          f"priced at the board")

    dest = out_dir / "forward_board_rehearsal.csv"
    table.assign(season=season, window=window, n_sims=n_sims,
                 seed=seed).to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} comparison rows → {dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="2023-24",
                        help="the played season to rehearse on")
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--window", default=None,
                        help="posterior window (default: the simulator's own)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, season=args.season, n_sims=args.sims, seed=args.seed,
        window=args.window)
