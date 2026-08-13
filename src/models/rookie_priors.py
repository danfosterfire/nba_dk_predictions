"""P4(b): does a no-prior player's preseason beat his draft slot as a prior?

`make rookie-priors` → `outputs/predictions/rookie_priors.csv`. numpy and pandas only,
seconds, no fit and no CmdStan.

## The question, and the incumbent it is asked against

A player with no usable prior season is invisible to every fitted head in the project — he
clears neither the component heads' `>= 200 prior minutes` filter nor the availability
design's lag-1 requirement. He is nonetheless on an October roster, and the minutes
allocation is zero-sum, so the chain has to say *something* about him. What it says today is
`stan_composition.rookie_share_priors`: **an expanding-window mean of what past no-prior
players in his draft bucket realized**, which becomes his `w_share` — the composition head's
prior minutes share, and through `order_frame` his position in the allocation order too.

That is the `bio_draft_number` imputation `docs/preseason-plan.md` P4(b) names, and P1 sized
the population it serves at **29.7%** of in-scope panel rows. The preseason gives exactly
these players their first real NBA observation, so the question is whether it beats a draft
slot — a fact that is by then one to four years old and was never about this season.

## What is compared

Three arms per target, all point-in-time, all over the same expanding window, and **none of
them fitted** — this is the §8b discipline (`availability_no_prior`), which is that the arms
differ in the *estimator's input* and not in its class:

| arm | the prior it hands a no-prior player |
|---|---|
| `draft_bucket` | the incumbent — past no-prior players in his bucket, mean of the target |
| `preseason` | his own preseason reading of the same quantity, `draft_bucket` where absent |
| `shrunk` | the two blended by preseason volume, `w = m / (m + k)` |

`k` is the one number estimated from data here and it is estimated on an **inner carve of the
fitting half** — the last two training seasons scored against everything before them, which
is `minutes_preseason`'s construction for its own shrinkage constant, for the same reason.
`k = 0` is the `preseason` arm exactly and `k = inf` is `draft_bucket` exactly, so the blend
nests both endpoints and a selected `k` cannot be an arm the ladder did not contain.

## Two target families, because "rate prior" means two different things here

**The share** — `minutes_share` — is the quantity the incumbent actually imputes, on the
scale it is consumed at. The preseason's reading of it is `mpg_pre / 48`, which is *the same
statistic computed on different games*, so this comparison is as close to like-for-like as
the project gets. It is **not** the panel's `min_share_pre`, which is a share of the team's
preseason minutes and differs by a factor of ~5 — see `TARGETS`.

**The rates** — seven per-36 counts and the three-point attempt share — are what P4(b)'s
wording points at, and they have no incumbent consumer today: a no-prior player is not in the
component heads at all, so nothing downstream reads a rate for him. They are measured anyway
because the *same* draft-bucket estimator is what would be reached for if one were needed,
and because a null here is what says the rate half of P4 should stay unbuilt.

## Scope, split and population

Rows are the no-prior player-seasons inside `selection_split`'s train ∪ validation, cut to
the **preseason-covered** window (2004-05 onward) for every arm including the incumbent —
P3's precedent, so that a structurally absent preseason before 2004-05 cannot be credited to
or against the block. Every headline is quoted on the **season-start-roster** population per
P1 decision 5, with the pooled reading carried beside it, because on this population above
all others a missing preseason is mostly a January signing.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.features.team_context import UNDRAFTED_BUCKET
from src.models.held_out import selection_split
from src.models.stan_composition import (composition_frame, draft_numbers, season_shares,
                                         share_lags)

#: The count heads whose per-36 rate the panel carries a preseason twin for, plus the
#: three-point attempt share. `pre_fg3a_share` is a share and the rest are per-36 counts, so
#: they are scored separately rather than pooled into one R2.
RATE_HEADS = ("fga", "fta", "reb", "ast", "stl", "blk", "tov")

#: `(target, realized column, preseason column, family)` — the whole comparison as data.
#: The share target leads because it is the only one with a live consumer.
#:
#: ⚠️ **The share's preseason column is `pre_minutes_share`, not the panel's
#: `min_share_pre`.** They are different quantities and the difference is a factor of ~5:
#: `stan_composition.season_shares` defines `minutes_share` as a player's minutes over
#: **game length** (~0.15-0.35 for a rotation player), while `min_share_pre` is his share of
#: his **team's** preseason minutes (~1/17). Comparing them measures a unit conversion. The
#: matching statistic is `mpg_pre / 48`, which is also the quantity `minutes_preseason`
#: builds its shipped delta from, so the two preseason-minutes measurements in the project
#: are on one scale rather than two.
TARGETS: tuple[tuple[str, str, str, str], ...] = (
    ("minutes_share", "minutes_share", "pre_minutes_share", "share"),
    *((f"per36_{h}", f"per36_{h}", f"pre_per36_{h}", "rate") for h in RATE_HEADS),
    ("fg3a_share", "fg3a_share", "pre_fg3a_share", "rate"),
)

#: Regulation game length, the denominator that puts `mpg_pre` on `minutes_share`'s scale.
#: Preseason overtime exists and the panel carries no length column for it; at ~6% of games
#: and ~10% extra length the approximation is worth ~0.6% of the column and is not what any
#: verdict here turns on.
REGULATION_LENGTH = 48.0

#: The arm that ships today, and the reference every margin is taken against.
INCUMBENT = "draft_bucket"

#: The volume shrinkage grid, in preseason minutes. `0` is the `preseason` arm and the last
#: rung is `draft_bucket` to within a rounding error, so the grid spans both endpoints and
#: an interior optimum is a real one rather than a boundary the grid imposed.
SHRINK_GRID = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 1e6)

#: Target seasons scored in the inner carve that selects `k`, taken off the END of the
#: fitting half. Two, matching `minutes_preseason`'s carve and the validation split's size.
INNER_SCORE_SEASONS = 2

#: Smallest bucket that may carry its own mean; below it the season's pooled no-prior mean is
#: used instead. `rookie_share_priors`' own fallback, made explicit and shared across targets.
MIN_BUCKET = 30


def no_prior_rows(cfg: dict) -> pd.DataFrame:
    """One row per no-prior (player, season): realized targets, draft bucket, preseason.

    "No prior" is `stan_composition.share_lags`' own definition — no minutes share in any of
    the three preceding seasons — rather than a second one written here, because the whole
    point of the comparison is to beat the estimator that population already has.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = list(cfg["data"]["seasons"])

    played = composition_frame(cfg)
    shares = season_shares(played)
    lagged = share_lags(shares, seasons)
    rows = lagged[lagged["no_prior"] == 1][["player_id", "season", "season_index",
                                            "minutes_share"]].copy()

    # The realized rates come from `component_targets.parquet` rather than from the
    # composition frame, which carries minutes and the allocation and no box score at all.
    # Restricted to games he PLAYED on both sides of the ratio, which is
    # `stan_composition.season_shares`' construction and the one the preseason panel's own
    # per-36 columns use — otherwise the two sides of this comparison would differ in their
    # denominators rather than in their information.
    box = pd.read_parquet(features_dir / "component_targets.parquet",
                          columns=["player_id", "season", "season_type", "min", "played",
                                   "fg3a", *RATE_HEADS])
    box = box[(box["season_type"] == "regular") & (box["played"] == 1)]
    totals = (box.groupby(["player_id", "season"], as_index=False)
              .agg(**{"minutes": ("min", "sum"), "fg3a_total": ("fg3a", "sum"),
                      **{h: (h, "sum") for h in RATE_HEADS}}))
    for head in RATE_HEADS:
        totals[f"per36_{head}"] = 36.0 * totals[head] / totals["minutes"].clip(lower=1e-9)
    totals["fg3a_share"] = totals["fg3a_total"] / totals["fga"].clip(lower=1e-9)
    rows = rows.merge(totals.drop(columns=[*RATE_HEADS, "fg3a_total"]),
                      on=["player_id", "season"], how="left")

    rows = rows.merge(draft_numbers(features_dir), on=["player_id", "season"], how="left")
    rows["draft_bucket"] = rows["draft_bucket"].fillna(UNDRAFTED_BUCKET)
    return rows


def attach_preseason(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The panel's readings of every target, plus `min_pre` — the shrinkage's volume."""
    derived = panel.assign(pre_minutes_share=panel["mpg_pre"] / REGULATION_LENGTH)
    keep = ["season", "player_id", "min_pre", "mpg_pre",
            *{p for _, _, p, _ in TARGETS}]
    out = rows.merge(derived[keep], on=["season", "player_id"], how="left")
    if len(out) != len(rows):
        raise AssertionError("the preseason panel duplicated a (season, player_id)")
    out["has_preseason"] = out["mpg_pre"].notna().astype(float)
    out["min_pre"] = out["min_pre"].fillna(0.0)
    return out


def bucket_priors(history: pd.DataFrame, target: str, covered: list[str]) -> pd.DataFrame:
    """`(season, draft_bucket) -> prior`, expanding over seasons strictly before the target.

    `rookie_share_priors`' construction, generalized to any target column and with its
    fallback made a rung rather than an implicit `fillna`: a bucket below `MIN_BUCKET` takes
    the season's pooled no-prior mean, and a season with no history at all is not emitted —
    the caller skips it, rather than being handed a constant that came from nowhere.
    """
    order = {season: i for i, season in enumerate(covered)}
    obs = history[history[target].notna()]
    rows = []
    for season in covered:
        past = obs[obs["season"].map(order) < order[season]]
        if past.empty:
            continue
        overall = float(past[target].mean())
        by_bucket = past.groupby("draft_bucket")[target].agg(["mean", "size"])
        for bucket, cell in by_bucket.iterrows():
            rows.append({"season": season, "draft_bucket": bucket,
                         "prior": float(cell["mean"]) if cell["size"] >= MIN_BUCKET
                         else overall,
                         "prior_n": int(cell["size"]), "pooled_prior": overall})
        for bucket in set(history["draft_bucket"]) - set(by_bucket.index):
            rows.append({"season": season, "draft_bucket": bucket, "prior": overall,
                         "prior_n": 0, "pooled_prior": overall})
    return pd.DataFrame(rows)


def arm_predictions(rows: pd.DataFrame, priors: pd.DataFrame, preseason_col: str,
                    k: float) -> dict[str, np.ndarray]:
    """`{arm: prediction}` for one target, on `rows` already joined to its bucket priors.

    The `shrunk` arm is `w * preseason + (1 - w) * bucket` with `w = m / (m + k)` over
    preseason minutes, and a row with no preseason reading has `w = 0` **by its volume**
    rather than by a special case — `min_pre` is 0 for him, so the blend already hands him
    the bucket prior and the three arms agree on that row by construction.
    """
    joined = rows.merge(priors, on=["season", "draft_bucket"], how="left")
    if joined["prior"].isna().any():
        raise AssertionError("a no-prior row reached no bucket prior at all")
    bucket = joined["prior"].to_numpy(float)
    observed = joined[preseason_col].to_numpy(float)
    seen = np.isfinite(observed)
    minutes = np.where(seen, joined["min_pre"].to_numpy(float), 0.0)
    weight = minutes / (minutes + k) if k > 0 else seen.astype(float)
    filled = np.where(seen, np.nan_to_num(observed), bucket)
    return {INCUMBENT: bucket,
            "preseason": filled,
            "shrunk": weight * filled + (1.0 - weight) * bucket}


def score_rows(y: np.ndarray, pred: np.ndarray, minutes: np.ndarray) -> dict:
    """R2, MAE and bias, unweighted and minutes-weighted.

    Both, because neither alone is the right reading. Unweighted conditions on nothing at
    all; minutes-weighted says what the error is worth where a rate is actually consumed —
    a per-36 over twelve realized minutes is noise on both sides of the comparison. Dropping
    those rows instead would be selection on the outcome, which is why it is a weight.
    """
    err = pred - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    w = minutes / minutes.sum() if minutes.sum() > 0 else np.full(len(y), 1.0 / len(y))
    wy = float(np.sum(w * y))
    w_tot = float(np.sum(w * (y - wy) ** 2))
    return {"n": int(len(y)),
            "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else np.nan,
            "mae": float(np.abs(err).mean()), "bias": float(err.mean()),
            "r2_minutes_weighted": (1.0 - float(np.sum(w * err ** 2)) / w_tot
                                    if w_tot > 0 else np.nan),
            "mae_minutes_weighted": float(np.sum(w * np.abs(err)))}


def paired_absolute(arm: np.ndarray, reference: np.ndarray, reps: int = 2000,
                    seed: int = 42) -> tuple[float, float, float]:
    """`(mean difference, lo, hi)` on |error| — paired inside the row.

    Absolute rather than squared error, because the targets span three orders of magnitude
    across the rate family and a squared-error bootstrap on `per36_fga` would be a statement
    about its two worst rows.
    """
    diff = np.abs(arm) - np.abs(reference)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    boot = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def walk(rows: pd.DataFrame, covered: list[str], target: str, realized: str,
         preseason_col: str, k: float) -> pd.DataFrame:
    """Every scorable target season's predictions, one row per (player-season, arm).

    Each season's bucket priors are pooled strictly before it, so the whole frame is
    point-in-time and the validation and rolling readings are two masks over one pass.
    """
    usable = rows[rows[realized].notna()].assign(_target=lambda d: d[realized])
    # A target with no realized rows at all is a broken join, not an empty ladder — it once
    # was, on a `season_type` label that reads `regular` and not `REGULAR_SEASON`, and it
    # surfaced three frames later as a missing column rather than here as a missing target.
    if usable.empty:
        raise ValueError(f"no realized `{realized}` on any no-prior row — check the join "
                         f"that builds it")
    all_priors = bucket_priors(usable.rename(columns={realized: target})
                               if realized != target else usable, target, covered)
    out = []
    for season in covered:
        target_rows = usable[usable["season"] == season]
        priors = all_priors[all_priors["season"] == season]
        if target_rows.empty or priors.empty:
            continue
        preds = arm_predictions(target_rows, priors, preseason_col, k)
        for arm, pred in preds.items():
            out.append(pd.DataFrame({
                # `row_id` is the pairing key. The bootstrap is paired inside the
                # player-season, so the arms must be aligned by identity rather than by
                # having been built in the same order — the failure that alignment bug class
                # produces is a silently wrong interval, not an exception.
                "row_id": target_rows.index.to_numpy(),
                "season": season, "arm": arm, "target": target,
                "y": target_rows["_target"].to_numpy(float), "pred": pred,
                "minutes": target_rows["minutes"].to_numpy(float),
                "roster": target_rows["on_season_start_roster"].to_numpy(float)}))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def select_k(rows: pd.DataFrame, covered: list[str], train_seasons: list[str],
             inner: int = INNER_SCORE_SEASONS) -> pd.DataFrame:
    """`k` chosen on an inner carve of the FITTING half, per target family.

    The carve scores the last `inner` *training* seasons against everything before them, so
    the grid never touches a season selection may read. One `k` per family rather than per
    target: eight rate targets each picking their own rung off ~120 rows apiece would be
    fitting the grid, and the family is the level the shrinkage claim is made at.
    """
    fitting = [s for s in covered if s in set(train_seasons)]
    scored = set(fitting[-inner:])
    grid = []
    for k in SHRINK_GRID:
        for family in ("share", "rate"):
            errors = []
            for target, realized, preseason_col, fam in TARGETS:
                if fam != family:
                    continue
                frame = walk(rows, fitting, target, realized, preseason_col, k)
                part = frame[(frame["arm"] == "shrunk") & (frame["season"].isin(scored))
                             & (frame["roster"] > 0)]
                if part.empty:
                    continue
                # Standardized so the eight rate targets contribute comparably — `per36_fga`
                # runs ~15 and `per36_stl` ~1, and an unscaled mean would be `fga`'s grid.
                scale = float(part["y"].std()) or 1.0
                standardized = np.abs(part["pred"] - part["y"]).to_numpy() / scale
                errors.append(standardized)
                # **Reported, never selected on.** One `k` per family is the decision (see
                # the docstring); the per-target curve is here so that a family optimum which
                # is wrong for one member is visible rather than buried, which is exactly
                # what happens to `fg3a_share`. Acting on it is a later round's call.
                grid.append({"analysis": "shrinkage_grid_by_target", "family": family,
                             "target": target, "k": k,
                             "inner_mae_standardized": float(standardized.mean()),
                             "n": int(len(standardized)),
                             "scored_seasons": ", ".join(sorted(scored))})
            if errors:
                grid.append({"analysis": "shrinkage_grid", "family": family, "k": k,
                             "inner_mae_standardized": float(np.concatenate(errors).mean()),
                             "n": int(sum(len(e) for e in errors)),
                             "scored_seasons": ", ".join(sorted(scored))})
    return pd.DataFrame(grid)


def summarize(frame: pd.DataFrame, validation: list[str], k: float) -> list[dict]:
    """One row per (target, arm, split, population), with a paired interval vs the incumbent."""
    out = []
    for split in ("validation", "rolling"):
        for population in ("draftable", "pooled"):
            part = frame[frame["season"].isin(validation)] if split == "validation" else frame
            if population == "draftable":
                part = part[part["roster"] > 0]
            if part.empty:
                continue
            for target, group in part.groupby("target"):
                # Indexed by `row_id` so every arm is aligned to the same player-seasons
                # positionally, which is what makes the bootstrap below a paired one.
                by_arm = {arm: g.set_index("row_id").sort_index()
                          for arm, g in group.groupby("arm")}
                reference = by_arm[INCUMBENT]
                for arm, g in by_arm.items():
                    if not g.index.equals(reference.index):
                        raise AssertionError(f"{arm} and {INCUMBENT} scored different rows "
                                             f"for {target} — the pairing is broken")
                    y = g["y"].to_numpy(float)
                    row = {"analysis": "rookie_prior_arm", "target": target, "arm": arm,
                           "split": split, "population": population, "k": k,
                           **score_rows(y, g["pred"].to_numpy(float),
                                        g["minutes"].to_numpy(float))}
                    delta, lo, hi = paired_absolute(
                        g["pred"].to_numpy(float) - y,
                        reference["pred"].to_numpy(float) - reference["y"].to_numpy(float))
                    row["mae_vs_incumbent"] = delta
                    row["mae_vs_incumbent_lo"] = lo
                    row["mae_vs_incumbent_hi"] = hi
                    per_origin = [
                        (float(np.abs(g.loc[m, "pred"] - g.loc[m, "y"]).mean()),
                         float(np.abs(reference.loc[m, "pred"]
                                      - reference.loc[m, "y"]).mean()))
                        for m in (g["season"] == s for s in sorted(set(g["season"])))]
                    row["origins_won"] = int(sum(a < b for a, b in per_origin))
                    row["origins_compared"] = int(sum(not np.isclose(a, b)
                                                      for a, b in per_origin))
                    row["origins"] = len(per_origin)
                    out.append(row)
    return out


def run(cfg: dict) -> dict[str, Path]:
    from src.eda.preseason_value import attach_season_start_roster, covered_seasons

    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(cfg["data"]["seasons"])
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))

    print("Rookie priors — does the preseason beat the draft bucket for a no-prior player?")
    rows = no_prior_rows(cfg)
    train, val = selection_split(rows.assign(dummy=0))
    allowed = sorted(set(train["season"]) | set(val["season"]))
    validation = sorted(set(val["season"]))

    covered = [s for s in covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv"))
               if s in set(allowed)]
    rows = attach_preseason(rows, pd.read_parquet(features_dir / "preseason.parquet"))
    rows = attach_season_start_roster(rows, seasons, cfg["data"]["raw_dir"], window_games)
    scoped = rows[rows["season"].isin(covered)].reset_index(drop=True)

    draftable = scoped[scoped["on_season_start_roster"] > 0]
    print(f"  {len(scoped):,} no-prior player-seasons over {len(covered)} covered "
          f"seasons ({covered[0]} → {covered[-1]});\n  {len(draftable):,} of them on a "
          f"season-start roster ({scoped['on_season_start_roster'].mean():.1%}), "
          f"and {draftable['has_preseason'].mean():.1%} of those have a preseason row.")
    print(f"  The test split is LOCKED — selection reads "
          f"{', '.join(validation)} and the rolling origins confirm.")

    grid = select_k(scoped, covered, sorted(set(train["season"])))
    family_grid = grid[grid["analysis"] == "shrinkage_grid"]
    chosen = {family: float(part.loc[part["inner_mae_standardized"].idxmin(), "k"])
              for family, part in family_grid.groupby("family")}
    print(f"\n  The shrinkage grid, on an inner carve of the FITTING half "
          f"({grid['scored_seasons'].iloc[0]}):")
    pd.set_option("display.width", 220)
    print(family_grid.pivot(index="k", columns="family",
                            values="inner_mae_standardized").round(5).to_string())
    print("  selected k: " + ", ".join(f"{f} = {v:,.0f}" for f, v in chosen.items()))
    per_target = grid[grid["analysis"] == "shrinkage_grid_by_target"]
    own = per_target.loc[per_target.groupby("target")["inner_mae_standardized"].idxmin()]
    print("  each target's OWN inner optimum, reported and NOT selected on: "
          + ", ".join(f"{r['target']} {r['k']:,.0f}" for _, r in own.iterrows()))

    scored: list[dict] = []
    for target, realized, preseason_col, family in TARGETS:
        k = chosen[family]
        frame = walk(scoped, covered, target, realized, preseason_col, k)
        if frame.empty:
            continue
        scored += summarize(frame, validation, k)
    table = pd.DataFrame(scored)

    dest = out_dir / "rookie_priors.csv"
    pd.concat([table, grid], ignore_index=True).to_csv(dest, index=False)

    for split in ("validation", "rolling"):
        shown = table[(table["split"] == split) & (table["population"] == "draftable")]
        print(f"\n  {split} · draftable — MAE against the `{INCUMBENT}` incumbent "
              f"(negative is better):")
        print(shown[["target", "arm", "n", "r2", "mae", "bias", "mae_vs_incumbent",
                     "mae_vs_incumbent_lo", "mae_vs_incumbent_hi", "origins_won",
                     "origins"]].round(4).to_string(index=False))
    print(f"\nWrote {len(table) + len(grid):,} rows → {dest}")
    return {"rookie_priors": dest}


if __name__ == "__main__":
    from src.models.rookie_priors import run as _run

    _run(yaml.safe_load(open("configs/default.yaml")))
