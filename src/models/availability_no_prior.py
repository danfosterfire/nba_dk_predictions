"""The no-prior population: which axis is worth grading, and the ladder on the one that is.

`make availability-no-prior` → `availability_no_prior.csv`, `availability_no_design_level.csv`.
numpy only, seconds, no fit.

Two rounds live here, in the order they were run. §8a asked which of *dispersion* and *level*
varies across this population and answered "level"; §8b is the ladder on the level, which is
the estimator `sim/season.no_design_availability` hands the simulator.

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

## The level ladder (§8b)

The arms differ **only** in the key a no-design row is pooled on; the estimator itself is
unchanged from the one that ships — the realized `gp / team_games` of rows carrying that key,
over seasons strictly before the target. `pooled` is the incumbent and reproduces
`no_design_availability`'s current scalar exactly, which is the nesting discipline `n_rho = 1`
and `U_n = 0` already carry on the fitted heads.

Every arm is scored through the **same beta-binomial the simulator applies** — the mean from
the arm, the dispersion from the fringe role bucket, which is what `availability_rho_bin`
hands a player with no prior MPG. So a CRPS difference here is a difference in the mean
function and nothing else, and the number is what the simulator's own availability draw
scores rather than a proxy for it.

**Selection reads validation; the rolling origins confirm it.** Every target season in
train ∪ validation is scored, each against an estimator pooled strictly before it, so the
arms have a replication check on the fitting half as well as a reading on the two seasons
selection may decide from — §4b's discipline, for §4b's reason.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.team_context import UNDRAFTED_BUCKET
from src.models.availability import crps, predictive_pmf
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

#: Smallest pooling cell that may carry its own rate; below it the key falls back one rung of
#: its arm's ladder. Set from the arithmetic rather than from the data: 50 player-seasons of
#: ~82 games at this population's `rho` of ~0.35 is an effective ~140 independent games, so a
#: rate near 0.4 carries a standard error of ~0.04 — small against the 0.49 spread the arms
#: are trying to resolve, and the number at which it stops being.
MIN_CELL = 50

#: The keys a no-design row can be pooled on, most specific first, per arm. `pooled` is the
#: incumbent's single key. A row takes the most specific rung whose cell clears `MIN_CELL`,
#: so every arm ends at `all` and no arm can fail to return a rate.
KEY_LADDERS: dict[str, tuple[str, ...]] = {
    "pooled": ("all",),
    "draft": ("draft_bucket", "all"),
    "tenure": ("tenure_class", "all"),
    "tenure_draft": ("tenure_draft", "tenure_class", "all"),
}

#: The arm `sim/season` uses when `sim.availability.no_design_level` is unset.
SHIPPED_LEVEL_ARM = "tenure_draft"

#: The incumbent every arm is bootstrapped against.
LEVEL_REFERENCE = "pooled"


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


def appearance_gap(rows: pd.DataFrame, history: pd.DataFrame,
                   seasons: list[str]) -> np.ndarray:
    """Seasons since each row's player last appeared, or **0** if he never has.

    The gap is measured from the panel rather than from the design, because a player the
    design has no row for may still have appeared: the design needs a *prior* season, and a
    player who last played three seasons ago has one, just not the one before this. That is
    exactly the returning-veteran class, and it is why the class is identifiable at all.

    `history` is the appearance record and `rows` the player-seasons to label, kept as two
    arguments because the simulator's are not the same frame: it labels a *target* season's
    roster from every season before it, while the ladder labels the history's own rows.
    """
    order = {season: i for i, season in enumerate(seasons)}
    appeared: dict[int, list[int]] = {}
    for player, group in history.groupby("player_id")["season"]:
        appeared[player] = sorted(order[s] for s in group)
    return np.array([order[s] - max([i for i in appeared.get(p, []) if i < order[s]],
                                    default=order[s])
                     for s, p in zip(rows["season"], rows["player_id"])], dtype=int)


def classify(cells: pd.DataFrame, design: pd.DataFrame,
             seasons: list[str]) -> pd.DataFrame:
    """No-design player-seasons, split into the withdrawn decision's three classes."""
    covered = set(map(tuple, design[["season", "player_id"]].to_numpy()))
    cells = cells.assign(covered=[(s, p) in covered for s, p
                                  in zip(cells["season"], cells["player_id"])])
    out = cells[~cells["covered"]].copy()
    out["gap"] = appearance_gap(out, cells, seasons)
    out["no_prior_appearance"] = out["gap"] == 0
    out["klass"] = np.where(out["no_prior_appearance"], "rookie",
                            np.where(out["gap"] == 1, "gap_1_season",
                                     np.where(out["gap"] == 2, "gap_2_seasons",
                                              "gap_3plus_seasons")))
    return out


def level_keys(rows: pd.DataFrame) -> pd.DataFrame:
    """The four pooling keys the level arms draw from, as columns on `rows`.

    Two facts about a player the head has no row for, and their cross. `tenure_class` splits
    a first-ever appearance from a return, `draft_bucket` is where the league placed him, and
    `tenure_draft` is the cross — which exists because **the draft bucket does not transfer
    across the two classes**: it is a strong signal for a rookie and a weak one for a
    returning veteran, so pooling them on it averages a gradient with a flat.

    Requires `gap` from `appearance_gap` and `draft_bucket` from `draft_numbers`; a missing
    bucket is `undrafted`, which is that column's own convention and not a filler — a player
    with no draft row went undrafted.
    """
    tenure = np.where(np.asarray(rows["gap"]) == 0, "rookie", "returning")
    bucket = rows["draft_bucket"].fillna(UNDRAFTED_BUCKET).to_numpy()
    return rows.assign(all="all", tenure_class=tenure, draft_bucket=bucket,
                       tenure_draft=[f"rookie__{b}" if t == "rookie" else "returning"
                                     for t, b in zip(tenure, bucket)])


def level_tables(history: pd.DataFrame, season: str, allowed: list[str],
                 arm: str) -> list[tuple[str, pd.DataFrame]]:
    """One `(key column, rate per key)` rung per level of `arm`'s ladder.

    Pooled over no-design rows in the seasons **strictly before** the target and inside the
    seasons selection may read — `no_design_availability`'s existing rule, keyed on one more
    column. Nothing is fitted: each cell's rate is its own realized `gp / team_games`, which
    is the estimator `stan_composition.rookie_share_priors` uses for the same population's
    minutes share.
    """
    if arm not in KEY_LADDERS:
        raise ValueError(f"unknown level arm {arm!r}; expected one of {sorted(KEY_LADDERS)}")
    earlier = [s for s in allowed if s < season]
    past = history[history["season"].isin(earlier)]
    if past.empty:
        raise ValueError(f"no seasons before {season} to estimate a no-design "
                         f"availability rate from")
    out = []
    for column in KEY_LADDERS[arm]:
        table = (past.groupby(column, as_index=False)
                 .agg(gp=("gp", "sum"), team_games=("team_games", "sum"),
                      rows=("gp", "size")))
        table["rate"] = table["gp"] / table["team_games"]
        out.append((column, table.set_index(column)))
    return out


def level_rates(rows: pd.DataFrame,
                tables: list[tuple[str, pd.DataFrame]]) -> tuple[np.ndarray, np.ndarray]:
    """`(rate, rung index)` per row — the most specific cell clearing `MIN_CELL`.

    The rung index is returned rather than discarded because a graded arm that silently fell
    back to the pooled rate on every row is the failure this whole item is about, and it
    would otherwise be indistinguishable from a graded arm that did nothing.
    """
    rate = np.full(len(rows), np.nan)
    rung = np.full(len(rows), -1, dtype=int)
    for level, (column, table) in enumerate(tables):
        keys = rows[column].to_numpy()
        cell_rate = table["rate"].reindex(keys).to_numpy(dtype=float)
        cell_rows = table["rows"].reindex(keys).to_numpy(dtype=float)
        take = np.isnan(rate) & np.isfinite(cell_rate)
        # The last rung is the terminal fallback and applies unconditionally — `MIN_CELL` is
        # a reason to prefer a coarser key, never a reason to return no rate at all.
        if level < len(tables) - 1:
            take &= cell_rows >= MIN_CELL
        rate[take], rung[take] = cell_rate[take], level
    if np.isnan(rate).any():
        raise AssertionError("a no-design row reached the end of its key ladder without a "
                             "rate, which means the `all` cell itself was missing")
    return rate, rung


def implied_rho(y: np.ndarray, n: np.ndarray, mu: np.ndarray) -> tuple[float, float]:
    """`(variance inflation, implied rho)` of `y` around a per-row mean `n·mu`.

    The same arithmetic §8a reports for a group around its own pooled rate, written to take a
    *vector* `mu` so a graded arm can be measured on the identical footing as the flat one.
    That comparison is the point: part of this population's 0.4337 unconditional dispersion
    is between-class variation in the *level*, so an arm that grades the level should shrink
    this number rather than leave it — and if it does, the fringe bucket's `rho` fits the
    residual better than it fitted the pooled arm's.
    """
    y, n, mu = np.asarray(y, float), np.asarray(n, float), np.asarray(mu, float)
    inflation = float(np.var(y - n * mu, ddof=1) / np.mean(n * mu * (1.0 - mu)))
    return inflation, (inflation - 1.0) / (float(n.mean()) - 1.0)


def realized(group: pd.DataFrame) -> dict:
    """Level, dispersion and both tails of a group's realized availability."""
    n = float(group["team_games"].mean())
    mu = float(group["gp"].sum() / group["team_games"].sum())
    inflation, rho = implied_rho(group["gp"].to_numpy(), group["team_games"].to_numpy(),
                                 np.full(len(group), mu))
    return {"rows": len(group), "mean_team_games": n, "mu": mu,
            "mean_mpg": float(group["mpg"].mean()),
            "inflation": inflation, "rho_implied": rho,
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


def score_level_arms(history: pd.DataFrame, allowed: list[str], validation: list[str],
                     rho: float = FITTED_ROLE_RHO[0]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every level arm, rolling over each readable target season, on two splits.

    One pass per `(target season, arm)`: the rates are pooled strictly before the target and
    the season's own no-design rows are scored against them, through the beta-binomial the
    simulator applies — the arm's mean at the **fringe** role bucket's dispersion, which is
    what `availability_rho_bin` hands a player with no prior MPG. Rows accumulate across
    target seasons and are summarized twice, on the validation seasons alone (which is what
    selects) and on every readable origin (which is what confirms).

    A target season is scored only once its pooling window carries `MIN_CELL` rows, so the
    first seasons — where even the pooled rate is a handful of players — are not reported as
    if the arms had been given a fair chance to differ.
    """
    from src.models.availability_window import paired_bootstrap

    order = {season: i for i, season in enumerate(allowed)}
    per_season: list[dict] = []
    kept: dict[str, dict[str, np.ndarray]] = {}
    for arm in KEY_LADDERS:
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("mu", "y", "n", "rung",
                                                              "season")}
        for season in allowed:
            rows = history[history["season"] == season]
            past = history[history["season"].isin([s for s in allowed if s < season])]
            if rows.empty or len(past) < MIN_CELL:
                continue
            mu, rung = level_rates(rows, level_tables(history, season, allowed, arm))
            y = rows["gp"].to_numpy(float)
            n = rows["team_games"].to_numpy(float)
            for key, value in (("mu", mu), ("y", y), ("n", n), ("rung", rung),
                               ("season", np.full(len(rows), order[season]))):
                parts[key].append(value)
            per_season.append({"analysis": "level_by_season", "arm": arm,
                               "season": season, "rows": len(rows), "n_pool": len(past),
                               "mae": float(np.abs(mu * n - y).mean()),
                               "bias": float((mu * n - y).mean()),
                               "crps": float(crps(predictive_pmf(n, mu, rho), y).mean()),
                               "graded_share": float((rung == 0).mean())})
        kept[arm] = {key: np.concatenate(value) for key, value in parts.items()}

    val = {order[s] for s in validation if s in order}
    rows: list[dict] = []
    for split, mask_of in (("validation", lambda k: np.isin(k["season"], list(val))),
                           ("rolling", lambda k: np.ones(len(k["y"]), bool))):
        scores = {arm: crps(predictive_pmf(k["n"][mask_of(k)], k["mu"][mask_of(k)], rho),
                            k["y"][mask_of(k)])
                  for arm, k in kept.items()}
        for arm, k in kept.items():
            m = mask_of(k)
            mu, y, n = k["mu"][m], k["y"][m], k["n"][m]
            pmf = predictive_pmf(n, mu, rho)
            cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
            inflation, resid_rho = implied_rho(y, n, mu)
            share, ss_tot = y / n, float(np.sum((y / n - (y / n).mean()) ** 2))
            delta, lo, hi = paired_bootstrap(scores[arm], scores[LEVEL_REFERENCE])
            rows.append({
                "analysis": "level_arm", "arm": arm, "split": split,
                "keys": " → ".join(KEY_LADDERS[arm]), "rows": int(m.sum()),
                "n_seasons": int(pd.Series(k["season"][m]).nunique()),
                "crps": float(scores[arm].mean()),
                "crps_delta_vs_pooled": delta, "crps_delta_lo": lo, "crps_delta_hi": hi,
                "mae": float(np.abs(mu * n - y).mean()),
                "bias": float((mu * n - y).mean()),
                "r2_gp_share": (1.0 - float(np.sum((share - mu) ** 2)) / ss_tot
                                if ss_tot > 0 else np.nan),
                "mu_min": float(mu.min()), "mu_max": float(mu.max()),
                "mu_spread": float(mu.max() / mu.min()),
                "rho_residual": resid_rho, "inflation": inflation,
                "graded_share": float((k["rung"][m] == 0).mean()),
                # §8a's two boundary statistics, on the same rows and the same predictive.
                "pred_gp_below_10": float(cdf[:, 9].mean()),
                "obs_gp_below_10": float((y < 10).mean()),
                "pred_gp_full": float(pmf[np.arange(len(y)), n.astype(int)].mean()),
                "obs_gp_full": float((y >= n).mean()),
                "rho_scored_at": rho})
        # Every ordered pair, not only the incumbent, because the arms are nested keys and
        # the live question is whether the *extra* key earns its place — `draft` against
        # `tenure_draft` is the comparison that decides that, and it is not a row above.
        by_season = pd.DataFrame(per_season).pivot(index="season", columns="arm",
                                                   values="crps")
        for arm in kept:
            for other in kept:
                if arm == other:
                    continue
                delta, lo, hi = paired_bootstrap(scores[arm], scores[other])
                rows.append({"analysis": "level_pairwise", "arm": arm,
                             "reference": other, "split": split,
                             "rows": int(mask_of(kept[arm]).sum()),
                             "crps_delta": delta, "crps_delta_lo": lo,
                             "crps_delta_hi": hi,
                             # How many ORIGINS the margin holds at, beside how large it is.
                             # A pooled interval says the mean difference is real; this says
                             # whether it is one season carrying twenty-five, which is the
                             # question §4b exists to ask and no bootstrap over pooled rows
                             # can answer.
                             "origins_won": int((by_season[arm]
                                                 < by_season[other]).sum()),
                             "origins": int(len(by_season))})
    return pd.DataFrame(rows), pd.DataFrame(per_season)


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

    # ── §8b: the ladder on the axis §8a left standing ────────────────────────
    history = level_keys(no_prior)
    arms, by_season = score_level_arms(history, allowed, sorted(set(val["season"])))
    level_dest = out_dir / "availability_no_design_level.csv"
    pd.concat([arms, by_season], ignore_index=True).to_csv(level_dest, index=False)

    print(f"\n  The LEVEL ladder — every arm scored through the beta-binomial the "
          f"simulator applies,\n  at the fringe bucket's rho "
          f"{FITTED_ROLE_RHO[0]:.4f}, against the pooled scalar that ships today.")
    for split in ("validation", "rolling"):
        shown = arms[(arms["split"] == split) & (arms["analysis"] == "level_arm")]
        print(f"\n  {split} — {int(shown['rows'].iloc[0]):,} rows over "
              f"{int(shown['n_seasons'].iloc[0])} target seasons:")
        print(shown[["arm", "keys", "crps", "crps_delta_vs_pooled", "crps_delta_lo",
                     "crps_delta_hi", "mae", "bias", "r2_gp_share", "mu_spread",
                     "rho_residual", "graded_share"]].round(4).to_string(index=False))
        pairs = arms[(arms["split"] == split) & (arms["analysis"] == "level_pairwise")
                     & (arms["arm"] == SHIPPED_LEVEL_ARM)]
        print(f"    {SHIPPED_LEVEL_ARM} against each other arm: " + ",  ".join(
            f"{r['reference']} {r['crps_delta']:+.4f} "
            f"[{r['crps_delta_lo']:+.4f}, {r['crps_delta_hi']:+.4f}] "
            f"({int(r['origins_won'])}/{int(r['origins'])} origins)"
            for _, r in pairs.iterrows()))
    print(f"\nWrote {len(arms) + len(by_season):,} rows → {level_dest}")
    return {"availability_no_prior": dest, "availability_no_design_level": level_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
