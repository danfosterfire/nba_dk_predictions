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
                                         forward_composition_players,
                                         forward_rookie_design, roster_members,
                                         schedule_team_games, synthetic_game_log)
from src.models.availability import season_start_dates
from src.models.rookie_rates import head_design as rookie_head_design
from src.models.stan_availability import head_design as availability_head_design
from src.models.stan_components import head_design as component_head_design
from src.models.stan_composition import head_frame
from src.sim.season import (FAMILY_COL, FIT_WINDOW, ROOKIE_FAMILY, SEED,
                            VETERAN_FAMILY, build_context, rookie_design, simulate)

#: Enough sims for a stable mean-total ranking; the seed-noise arm is what SAYS whether
#: it was enough, which is why the floor is measured rather than assumed.
N_SIMS = 500


def forward_frames(cfg: dict, season: str,
                   real_design: pd.DataFrame | None = None,
                   real_availability: pd.DataFrame | None = None,
                   real_rookie: pd.DataFrame | None = None,
                   members: pd.DataFrame | None = None) -> dict:
    """Every frame `build_context` needs for `season`, built without its game log.

    Five injections, one per harvested artifact: the synthetic panel under the roster
    grid, the forward availability design under the availability head, the forward
    component design under the eleven veteran rate heads, the forward **rookie** design
    under their eleven true-rookie twins, and the forward per-player frame under the
    composition. Each goes through the SAME shipped attachment as its retrospective twin
    (`head_design(design=...)`), so what differs is where the rows came from and never
    what was done to them.

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

    # **The rookie design's target-season rows are BUILT here, not withheld.** Until
    # 2026-08-22 they were dropped, because `rookie_rates.build_design` carves its
    # population out of `component_targets` and "he played in the NBA this season" is
    # target-season information a forward context may not read. §5g replaces the source
    # rather than the rule: membership now comes off the roster snapshot, the same place
    # the availability and composition frames take it from, and the same synthetic `log`
    # feeds it — so a rehearsal that cuts the population cuts it here too.
    rookie_fwd = rookie_head_design(cfg, design=forward_rookie_design(
        cfg, season, targets=targets_cut, log=log))
    if real_rookie is None:
        real_rookie = rookie_design(cfg)
    rookie_head = pd.concat([real_rookie[real_rookie["season"] != season],
                             rookie_fwd], ignore_index=True)
    print(f"  rookie design: {len(rookie_fwd):,} forward true-rookie rows, "
          f"{int(rookie_fwd['has_preseason'].sum()):,} of them carrying a preseason block "
          f"(0 before the\n    October fetch — the missing indicators carry the rest)")
    return {"composition": players, "design": comp_head, "availability": avail_head,
            "panel": fwd_panel, "rookie": rookie_head,
            "design_forward": comp_fwd[comp_fwd["season"] == season],
            "rookie_forward": rookie_fwd, "members": members}


def forward_context(cfg: dict, season: str, window: str, n_sims: int, seed: int,
                    real_design: pd.DataFrame | None = None,
                    real_availability: pd.DataFrame | None = None,
                    real_rookie: pd.DataFrame | None = None,
                    members: pd.DataFrame | None = None) -> dict:
    """`build_context` fed from `forward_frames`' output for `season`.

    Split from the builders above because a **test** season can be built and cannot be
    simulated: `season.assert_season_allowed` raises for 2026-27 under any window but the
    production one, which is `docs/final-evaluation-plan.md`'s guard doing its job and not
    a forward-design gap. `run(frames_only=True)` stops at the frames for exactly that
    case; everything else calls this.
    """
    f = forward_frames(cfg, season, real_design=real_design,
                       real_availability=real_availability, real_rookie=real_rookie,
                       members=members)
    return build_context(cfg, season, window, n_sims, seed,
                         composition=f["composition"], design=f["design"],
                         availability=f["availability"], panel=f["panel"],
                         rookie=f["rookie"])


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


def unit_census(ctx: dict, board: pd.DataFrame, priced: set) -> pd.DataFrame:
    """Units per population, and how many of them the market prices — §4's acceptance.

    "Present **and draftable**" is two questions and this answers both: a unit reaches the
    tensor (present) and carries an ADP row on the season's board (draftable, in the sense
    that the field will take him and our seat therefore has to price him). The market side
    is read off `adp_panel.parquet` rather than the tensor, because a rookie the model can
    score and the market has never heard of is not the failure this gate is looking for.

    `best_rank` is the strongest thing each population puts on the board, which is what
    turns "there are rookies on it" into "they are somewhere a draft reaches".
    """
    units = ctx["units"]
    family = (units[FAMILY_COL] if FAMILY_COL in units.columns
              else pd.Series(VETERAN_FAMILY, index=units.index))
    rung = (units["lag_rung"] if "lag_rung" in units.columns
            else pd.Series("veteran", index=units.index))
    ranks = board.set_index("player_id")["rank"]

    groups = {"veteran": (family != ROOKIE_FAMILY) & (rung == "veteran"),
              "lag_recovered": (family != ROOKIE_FAMILY) & (rung != "veteran"),
              "true_rookie": family == ROOKIE_FAMILY}
    rows = []
    for name, mask in groups.items():
        ids = units.loc[mask.to_numpy(), "player_id"]
        seen = ranks.reindex(ids).dropna()
        rows.append({"population": name, "units": int(mask.sum()),
                     "adp_priced": int(ids.isin(priced).sum()),
                     "best_rank": int(seen.min()) if len(seen) else -1})
    return pd.DataFrame(rows)


def untouched_ids(ctx: dict) -> set:
    """Rung-0 veteran units — the population neither §5b's ladder nor §5g's rookies added.

    §4's acceptance has two halves and this is the second one's population: the stability
    readings §6h took before either family existed must still hold *on the units they
    describe*. Comparing the whole board instead would fold the new rows' own agreement
    into the number that is supposed to say the old rows did not move.
    """
    units = ctx["units"]
    family = (units[FAMILY_COL] if FAMILY_COL in units.columns
              else pd.Series(VETERAN_FAMILY, index=units.index))
    rung = (units["lag_rung"] if "lag_rung" in units.columns
            else pd.Series("veteran", index=units.index))
    keep = (family == VETERAN_FAMILY) & (rung == "veteran")
    return set(units.loc[keep.to_numpy(), "player_id"])


def priced_ids(cfg: dict, season: str) -> set:
    """Players the market puts a number on for `season` — the draftable half of §4.

    `adp_panel.parquet` is the point-in-time-safe panel and `draft_pool.parquet` is not
    read here on purpose: the pool is rebuilt from the panel plus a membership rule this
    module is in the business of replacing.
    """
    path = Path(cfg["data"]["features_dir"]) / "adp_panel.parquet"
    if not path.exists():
        return set()
    panel = pd.read_parquet(path, columns=["season", "player_id", "adp"])
    rows = panel[(panel["season"] == season) & panel["adp"].notna()]
    return set(rows["player_id"].dropna().astype("int64"))


def frames_census(cfg: dict, season: str, frames: dict) -> pd.DataFrame:
    """The population census off the forward DESIGNS, for a season nobody can simulate.

    `unit_census` reads the tensor's units, which needs a context, which needs the
    held-out unlock. This reads the same three populations one step earlier — a design row
    intersected with the composition's per-player frame, which is what `component_units`
    would do — so a locked season still gets §4's acceptance answered on the rows it has.
    """
    priced = priced_ids(cfg, season)
    allocated = set(frames["composition"]["player_id"])
    vet = frames["design_forward"]
    rung = (vet["lag_rung"] if "lag_rung" in vet.columns
            else pd.Series("veteran", index=vet.index))
    groups = {"veteran": vet.loc[(rung == "veteran").to_numpy(), "player_id"],
              "lag_recovered": vet.loc[(rung != "veteran").to_numpy(), "player_id"],
              "true_rookie": frames["rookie_forward"]["player_id"]}
    rows = []
    for name, ids in groups.items():
        ids = ids[ids.isin(allocated)]
        rows.append({"population": name, "units": int(len(ids)),
                     "adp_priced": int(ids.isin(priced).sum()), "best_rank": -1})
    return pd.DataFrame(rows)


def run(cfg: dict, season: str = "2023-24", window: str | None = None,
        n_sims: int = N_SIMS, seed: int = SEED,
        frames_only: bool = False) -> Path:
    window = window or str(cfg.get("sim", {}).get("fit_window", FIT_WINDOW))
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    if frames_only:
        # The production case, and the one a rehearsal cannot cover: the target season is
        # a TEST season, so every forward frame builds and `build_context` refuses to
        # simulate it. Nothing here is unlocked — the census is taken off the designs.
        print(f"Forward board — {season}, FRAMES ONLY (no simulation)")
        print(f"  A test season cannot be simulated outside `make posteriors-production`; "
              f"this is\n  §4's acceptance answered on the rows the forward builders "
              f"produce for it.")
        print(f"\n[forward frames — the production shape, snapshot membership]")
        frames = forward_frames(cfg, season)
        census = frames_census(cfg, season, frames).assign(arm="forward")
        print(f"\n[§4 acceptance — the two recovered populations, on the forward board]")
        print(census[["population", "units", "adp_priced"]].to_string(index=False))
        missing = census[(census["population"] != "veteran") & (census["units"] == 0)]
        print(f"  acceptance: "
              + ("PASS" if missing.empty else
                 "FAIL: " + ", ".join(missing["population"]) + " absent"))
        dest = out_dir / f"forward_board_population_{_slug(season)}.csv"
        census.assign(season=season).to_csv(dest, index=False)
        print(f"\nSaved {len(census):,} population rows → {dest}")
        return dest

    print(f"Forward board — {season} at the `{window}` window, {n_sims:,} sims, "
          f"seed {seed}")
    print(f"  The forward arm may not read this season's game log; the retro arm is "
          f"`make simulate-season`'s own path.")

    print(f"\n[retrospective context]")
    real_design = component_head_design(cfg)
    real_availability = availability_head_design(cfg)
    real_rookie = rookie_design(cfg)
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
                            real_availability=real_availability,
                            real_rookie=real_rookie)
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
                            real_rookie=real_rookie,
                            members=grid_members)
    print(f"  {ctx_x['n_units']:,} scorable units over {ctx_x['n_games']:,} games")
    sim_x = simulate(ctx_x)
    board_x = board_frame(ctx_x, sim_x)

    # The same three comparisons twice: once on every unit each board carries, and once on
    # the rung-0 veterans alone — §4's "unchanged for units the ladder did not touch". The
    # ranks are re-derived inside the restricted population rather than carried over, so a
    # rookie landing between two veterans cannot move either one's number.
    untouched = untouched_ids(ctx_r) & untouched_ids(ctx_f) & untouched_ids(ctx_x)

    def rung0(board: pd.DataFrame) -> pd.DataFrame:
        out = board[board["player_id"].isin(untouched)].copy()
        out["rank"] = out["mean_total"].rank(ascending=False, method="first").astype(int)
        return out

    pairs = [("retro_vs_retro_seed_noise", board_r, board_n),
             ("forward_fixed_population_vs_retro", board_r, board_x),
             ("forward_vs_retro", board_r, board_f)]
    rows = ([compare_boards(a, b, arm) for arm, a, b in pairs]
            + [compare_boards(rung0(a), rung0(b), f"{arm}__rung0")
               for arm, a, b in pairs])
    table = pd.DataFrame(rows)

    # §4's acceptance, `docs/rookie-rates-plan.md`: rookie AND ladder-recovered units
    # present and draftable on BOTH boards. It is printed beside the stability comparison
    # rather than in place of it, because the two halves fail differently — a population
    # that is absent shows up here and a population that is present but wrongly scored
    # shows up there.
    priced = priced_ids(cfg, season)
    census = pd.concat([unit_census(ctx, board, priced).assign(arm=arm)
                        for arm, ctx, board in
                        (("retrospective", ctx_r, board_r),
                         ("forward", ctx_f, board_f),
                         ("forward_fixed_population", ctx_x, board_x))],
                       ignore_index=True)
    print(f"\n[§4 acceptance — the two recovered populations, on both boards]")
    print(census.pivot(index="population", columns="arm",
                       values=["units", "adp_priced", "best_rank"]).to_string())
    missing = census[(census["population"] != "veteran") & (census["units"] == 0)]
    verdict = ("PASS" if missing.empty else
               "FAIL: " + ", ".join(f"{r.population} absent on {r.arm}"
                                    for r in missing.itertuples()))
    print(f"  acceptance: {verdict} — {len(priced):,} players carry an ADP for {season}")

    print(f"\n[the comparison — forward against retro, with seed noise as the floor]")
    print(table.round(4).to_string(index=False))
    print(f"  `__rung0` is the same comparison on the {len(untouched):,} veteran units "
          f"neither the ladder nor the\n  rookie family added — §4's second acceptance "
          f"half, which asks that they did not move.")
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
    census_dest = out_dir / "forward_board_population.csv"
    census.assign(season=season, window=window, n_sims=n_sims,
                  seed=seed).to_csv(census_dest, index=False)
    print(f"Saved {len(census):,} population rows → {census_dest}")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", default="2023-24",
                        help="the played season to rehearse on")
    parser.add_argument("--sims", type=int, default=N_SIMS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--window", default=None,
                        help="posterior window (default: the simulator's own)")
    parser.add_argument("--frames-only", action="store_true",
                        help="build the forward frames and census them without "
                             "simulating — the only form a TEST season admits")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, season=args.season, n_sims=args.sims, seed=args.seed,
        window=args.window, frames_only=args.frames_only)
