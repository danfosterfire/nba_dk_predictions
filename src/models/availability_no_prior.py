"""The no-prior population: what it realizes, and which axis is worth grading.

`make availability-no-prior` → `availability_no_prior.csv`. numpy only, seconds, no fit.

## The question

The availability head is a lag-1 design, so a player with no prior-season row is not in its
frame at all. He is still on a season-start roster, still consumes a roster spot, and — because
the minutes allocation is zero-sum — whatever availability he is given comes straight out of
his teammates. `sim/season.no_design_availability` is how he reaches the simulator: a **pooled
empirical rate** over the no-design player-seasons strictly before the target, with
`sim/season.availability_rho_bin` sending him to the **lowest** role bucket because his prior
MPG is NaN.

`docs/availability-window-plan.md` §8 **decision 7** — relocated there from the deleted ship
plan, where it was decision 2 — specified replacing that fallback with a three-class
**imputed role bucket**: rookies from draft position, returning veterans from their bucket at
last appearance conditioned on gap length, everyone else the lowest bucket. It was taken on
2026-08-11 and never implemented.

**The role bucket carries dispersion, not level.** So the decision is only worth implementing
if the no-prior classes differ in *dispersion*. This module measures that directly, against
the two things it would be compared to: what the class actually realizes, and what `rho` its
imputed bucket would hand it.

## Scope, and the split

Everything here is descriptive — nothing is fitted, nothing is persisted for a consumer, and
no arm is selected. Rows are restricted to the seasons selection may read (train ∪ validation
via `selection_split`), because a decision is being taken from them.

The dispersion column is an **unconditional** implied `rho`, from each group's own variance
of `gp` around its own pooled mean. That is the right comparison for this population and only
for this population: they reach the head with a *constant* `mu` and no covariates, so the
variance the simulator has to reproduce around that constant is the unconditional one. The
in-design rows are carried on the same footing as the control, which is what makes the
comparison apples-to-apples rather than a fitted `rho` against a raw one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.models.held_out import selection_split
from src.models.stan_availability import availability_design
from src.models.stan_composition import draft_numbers

#: Smallest group worth reporting a variance for. Below it the implied `rho` is dominated by
#: its own sampling error and would read as a gradient that is not there.
MIN_GROUP = 30

#: The shipped head's fitted role dispersions, fringe → star, from
#: `stan_availability`'s persisted `rho_draws`. Carried as a column so the table can be read
#: without opening a second artifact; `docs/availability-window-plan.md` §7i is the source.
FITTED_ROLE_RHO = (0.3176, 0.2701, 0.2535, 0.2067)


def primary_team_cells(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, player): games played, schedule length and realized MPG.

    A traded player is attributed wholly to his **last** team, which is
    `features.availability.season_availability`'s convention and the one
    `sim/season.season_grid` reproduces — taking the panel as it stands would give him a
    denominator of ~93 games against the head's 82.
    """
    last = (panel[panel["played"] == 1].sort_values(["game_date", "game_id"])
            .groupby(["season", "player_id"], as_index=False)
            .agg(team_id=("team_id", "last")))
    primary = panel.merge(last, on=["season", "player_id", "team_id"], how="inner")
    cells = (primary.groupby(["season", "player_id"], as_index=False)
             .agg(gp=("played", "sum"), team_games=("played", "size"),
                  minutes=("min", "sum")))
    cells["mpg"] = cells["minutes"] / cells["gp"].clip(lower=1)
    return cells


def classify(cells: pd.DataFrame, design: pd.DataFrame,
             seasons: list[str]) -> pd.DataFrame:
    """No-design player-seasons, split into the withdrawn decision's three classes.

    The gap is measured from the panel rather than from the design, because a player the
    design has no row for may still have appeared: the design needs a *prior* season, and a
    player who last played three seasons ago has one, just not the one before this. That is
    exactly the returning-veteran class, and it is why the class is identifiable at all.
    """
    order = {season: i for i, season in enumerate(seasons)}
    covered = set(map(tuple, design[["season", "player_id"]].to_numpy()))
    cells = cells.assign(covered=[(s, p) in covered for s, p
                                  in zip(cells["season"], cells["player_id"])])
    appeared: dict[int, list[int]] = {}
    for player, group in cells.groupby("player_id")["season"]:
        appeared[player] = sorted(order[s] for s in group)
    out = cells[~cells["covered"]].copy()
    out["gap"] = [order[s] - max([i for i in appeared.get(p, []) if i < order[s]],
                                 default=order[s])
                  for s, p in zip(out["season"], out["player_id"])]
    out["no_prior_appearance"] = out["gap"] == 0
    out["klass"] = np.where(out["no_prior_appearance"], "rookie",
                            np.where(out["gap"] == 1, "gap_1_season",
                                     np.where(out["gap"] == 2, "gap_2_seasons",
                                              "gap_3plus_seasons")))
    return out


def realized(group: pd.DataFrame) -> dict:
    """Level, dispersion and both tails of a group's realized availability."""
    n = float(group["team_games"].mean())
    mu = float(group["gp"].sum() / group["team_games"].sum())
    inflation = float(np.var(group["gp"] - group["team_games"] * mu, ddof=1)
                      / (n * mu * (1.0 - mu)))
    return {"rows": len(group), "mean_team_games": n, "mu": mu,
            "mean_mpg": float(group["mpg"].mean()),
            "inflation": inflation, "rho_implied": (inflation - 1.0) / (n - 1.0),
            "p_gp_below_10": float((group["gp"] < 10).mean()),
            "p_gp_full": float((group["gp"] >= group["team_games"]).mean())}


def imputed_bucket(mean_mpg: float) -> int:
    """The 1-based role bucket the draft-position map would land a class in."""
    return int(np.clip(np.digitize(mean_mpg, ROLE_EDGES), 1, len(ROLE_LABELS)))


def ladder(no_prior: pd.DataFrame, in_design: pd.DataFrame) -> pd.DataFrame:
    """Every population on one footing, with what the withdrawn rule would hand each one.

    `rho_imputed` is the fitted dispersion the class's mean MPG would map it to, and
    `rho_error` is that minus what it realizes. A **negative** `rho_error` is the failure
    mode the decision was suspected of and never tested for: a bucket that says the class
    is more reliable
    than it is.
    """
    rows = [{"population": "in_design", "klass": "control", **realized(in_design)},
            {"population": "no_design_all", "klass": "all", **realized(no_prior)}]
    for klass, group in no_prior.groupby("klass"):
        if len(group) < MIN_GROUP:
            continue
        rows.append({"population": f"class__{klass}", "klass": klass, **realized(group)})
    rookies = no_prior[no_prior["no_prior_appearance"]]
    for bucket, group in rookies.groupby("draft_bucket"):
        if len(group) < MIN_GROUP:
            continue
        rows.append({"population": f"draft__{bucket}", "klass": "rookie",
                     **realized(group)})
    frame = pd.DataFrame(rows)
    frame["imputed_bucket"] = [ROLE_LABELS[imputed_bucket(m) - 1]
                               for m in frame["mean_mpg"]]
    frame["rho_imputed"] = [FITTED_ROLE_RHO[imputed_bucket(m) - 1]
                            for m in frame["mean_mpg"]]
    # What the population gets TODAY: the lowest bucket, because prior MPG is NaN.
    frame["rho_shipped"] = FITTED_ROLE_RHO[0]
    frame["rho_imputed_error"] = frame["rho_imputed"] - frame["rho_implied"]
    frame["rho_shipped_error"] = frame["rho_shipped"] - frame["rho_implied"]
    return frame


def spreads(frame: pd.DataFrame) -> pd.DataFrame:
    """The one comparison the decision turns on: which axis actually varies.

    The withdrawn rule grades **dispersion** by draft position. If the draft buckets'
    realized dispersion is flat and their realized *level* is not, the rule is grading the
    wrong quantity — and no amount of care in the mapping fixes that.
    """
    draft = frame[frame["population"].str.startswith("draft__")]
    out = []
    for axis, column in (("level", "mu"), ("dispersion", "rho_implied"),
                         ("left tail", "p_gp_below_10")):
        lo, hi = float(draft[column].min()), float(draft[column].max())
        # The left tail's minimum is a genuine zero — no lottery top-5 pick in the window
        # played under ten games — so its ratio is not a number and its `gap` is the
        # readable form. Emitting `inf` would put an unauditable literal in the table.
        out.append({"analysis": "draft_bucket_spread", "axis": axis, "column": column,
                    "min": lo, "max": hi, "gap": hi - lo,
                    "spread": hi / lo if lo else np.nan, "buckets": len(draft)})
    role = np.asarray(FITTED_ROLE_RHO, dtype=float)
    out.append({"analysis": "draft_bucket_spread", "axis": "dispersion (fitted, in-design)",
                "column": "rho_fitted", "min": float(role.min()), "max": float(role.max()),
                "gap": float(role.max() - role.min()),
                "spread": float(role.max() / role.min()), "buckets": len(role)})
    return pd.DataFrame(out)


def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(cfg["data"]["seasons"])

    design = availability_design(cfg)
    train, val = selection_split(design)
    allowed = sorted(set(train["season"]) | set(val["season"]))

    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "game_date",
                 "played", "min"])
    cells = primary_team_cells(panel)
    cells = cells[cells["season"].isin(allowed)]
    if set(cells["season"]) - set(allowed):
        raise AssertionError("the no-prior measurement reached a season selection may "
                             "not read")

    no_prior = classify(cells, design, seasons)
    no_prior = no_prior.merge(draft_numbers(features_dir), on=["player_id", "season"],
                              how="left")
    covered = set(map(tuple, design[["season", "player_id"]].to_numpy()))
    in_design = cells[[(s, p) in covered for s, p
                       in zip(cells["season"], cells["player_id"])]]

    print(f"  {len(no_prior):,} no-design player-seasons of {len(cells):,} in the "
          f"{len(allowed)} seasons selection may read, against {len(in_design):,} "
          f"in-design rows")
    table = ladder(no_prior, in_design)
    axes = spreads(table)
    frame = pd.concat([table.assign(analysis="no_prior_population"), axes],
                      ignore_index=True)
    dest = out_dir / "availability_no_prior.csv"
    frame.to_csv(dest, index=False)

    pd.set_option("display.width", 220)
    print(table[["population", "rows", "mu", "mean_mpg", "rho_implied", "p_gp_below_10",
                 "imputed_bucket", "rho_imputed", "rho_imputed_error",
                 "rho_shipped_error"]].round(4).to_string(index=False))
    print("\n  Which axis is worth grading:")
    print(axes[["axis", "min", "max", "gap", "spread", "buckets"]]
          .round(4).to_string(index=False))
    print(f"\nWrote {len(frame):,} rows → {dest}")
    return {"availability_no_prior": dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
