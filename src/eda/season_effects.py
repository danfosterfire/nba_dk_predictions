"""League-level season effects — the shared factor no head currently carries.

Every head in this project is fitted on 30 pooled seasons with **no season term**, which
sits awkwardly against the repo's own standing rule (`CLAUDE.md`: "ALWAYS absorb season
when regressing on 30 pooled seasons", written after pooled `teammate_spacing` and
`team_pace` correlations collapsed by ~an order of magnitude under season fixed effects).

The reason the rule was not simply applied is the prediction-time constraint: **a season
fixed effect for season S does not exist when forecasting S**. There is no fitted dummy for
a season that has not happened. So the question is not "absorb season or not" but which of
two usable forms is right for each quantity:

- a **year-on-year trend** extrapolated one season forward — usable only where the league
  moves smoothly and in one direction; or
- a **year-level random effect** — usable everywhere, and it does not claim to predict the
  direction, only to stop pretending the shift is zero.

This module measures which. For every modeled quantity it separates the league rate's
**drift** (a smooth trend a fixed effect could extrapolate) from its **shock** (residual
year-to-year movement no model can forecast), and it prices the cost of ignoring both by
measuring the no-fit floor's systematic bias on the held-out seasons.

## Why this matters more than an equivalent amount of ordinary error

A league-level shift is **perfectly correlated across every player**, so unlike per-player
prediction error it does not diversify away in a portfolio. A −7% league error on free
throws is −7% on a whole roster's free-throw points. For comparison, the shared-β parameter
uncertainty measured in `stan_availability.board_correlation` is worth +0.2% on a 15-man
roster. Season effects are a much larger non-diversifiable risk than the one the Bayesian
fit was built to capture.

## What it finds, in one line each

- **`fta` is a shock, not a trend.** League FTA/36 oscillates in a 1.21x band with a 4.4%
  year-over-year sd and swings past +/-5% in 9 of 29 transitions — the 2004-05 hand-checking
  crackdown at **+7.6%** and the 2025-26 jump at **+8.6%** are refereeing interventions, not
  drift. A trend term cannot help; a year random effect can.
- **`fg3a` is the opposite** — a smooth 2.97x climb with a trend correlation of +0.94.
  Carry-forward lags a monotone trend by exactly one season, which is a *predictable* bias.
- **The floor pays for it.** `fta` carries a **−7.0%** systematic bias on the held-out
  seasons (**−10.7%** in 2025-26, which is the +8.6% league jump arriving one year late).

Usage:
    python -m src.eda.season_effects
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.component_rates import (COUNT_HEADS, CONVERSION_HEADS, PER36,
                                        build_design, carry_forward,
                                        split_seasons)

# Availability is per player-season rather than a league rate, so it needs a population.
# 10 games matches the bar used in the load-management look in `docs/availability-plan.md`.
AVAILABILITY_MIN_GAMES = 10
ROLE_EDGES = [0, 12, 24, 30, 60]
ROLE_LABELS = ["<12 mpg", "12-24", "24-30", "30+ mpg"]

# A trend is worth extrapolating only if it explains most of the league movement AND is
# large enough to matter. Both bars have to clear: a tiny but perfectly linear drift is not
# worth a parameter, and a large but erratic one cannot be extrapolated at all.
TREND_R2_BAR = 0.70
TREND_SIZE_BAR = 1.0        # percent per season


# ── League rates ──────────────────────────────────────────────────────────────

def _played(targets: pd.DataFrame) -> pd.DataFrame:
    """Regular-season rows the player actually appeared in.

    Regular season only, matching every fitting frame in the project — playoff rates are a
    role interaction whose sign flips, so pooling them would put a spurious step in every
    season's league rate proportional to how far teams advanced.
    """
    return targets[(targets["season_type"] == "regular") & (targets["played"] == 1)]


def league_rates(targets: pd.DataFrame,
                 lengths: pd.DataFrame | None = None) -> pd.DataFrame:
    """One league-wide rate per (season, quantity), plus its year-over-year change.

    Rates are **totals over totals**, not the mean of player rates: the league's FTA per 36
    is `sum(fta) / sum(min) * 36`, which is the actual league rate rather than an average
    weighted by roster churn. A mean of per-player rates would move whenever the population
    of fringe players changed, and that is composition, not refereeing.
    """
    g = _played(targets)
    rows = []
    for season, d in g.groupby("season"):
        minutes = float(d["min"].sum())
        if minutes <= 0:
            continue
        for c in COUNT_HEADS:
            rows.append({"season": season, "quantity": c, "kind": "count_per36",
                         "rate": float(d[c].sum()) / minutes * PER36})
        for made, att in CONVERSION_HEADS:
            attempts = float(d[att].sum())
            if attempts > 0:
                rows.append({"season": season, "quantity": f"{made}_pct",
                             "kind": "conversion",
                             "rate": float(d[made].sum()) / attempts})

    if lengths is not None:
        keys = ["season", "season_type", "game_id"]
        merged = g.merge(lengths[keys + ["game_length"]], on=keys, how="inner")
        share = (merged.groupby("season")
                 .apply(lambda d: float(d["min"].sum()) / float(d["game_length"].sum()),
                        include_groups=False))
        for season, value in share.items():
            rows.append({"season": season, "quantity": "minutes_share",
                         "kind": "minutes", "rate": float(value)})

    out = pd.DataFrame(rows).sort_values(["quantity", "season"]).reset_index(drop=True)
    out["season_index"] = out.groupby("quantity")["season"].rank(method="dense").astype(int) - 1
    out["yoy_pct"] = out.groupby("quantity")["rate"].pct_change() * 100
    return out


def availability_rates(features: pd.DataFrame,
                       min_games: int = AVAILABILITY_MIN_GAMES) -> pd.DataFrame:
    """League mean `gp_share` per season, and the same split by role.

    Availability has no natural totals-over-totals form — "the league played 78% of its
    games" is dominated by whichever fringe players happened to appear — so this reports a
    mean over a qualified population and the role split beside it, because
    `docs/availability-plan.md` finds the era effect is **role-graded** rather than a level
    shift.
    """
    f = features[(features["window"] == "full") & (features["gp"] >= min_games)].copy()
    f["role"] = pd.cut(f["minutes_per_game"], ROLE_EDGES, labels=ROLE_LABELS)

    overall = (f.groupby("season", as_index=False)
               .agg(rate=("gp_share", "mean"), n=("player_id", "size")))
    overall["quantity"] = "gp_share"
    overall["role"] = "all"

    by_role = (f.groupby(["season", "role"], as_index=False, observed=True)
               .agg(rate=("gp_share", "mean"), n=("player_id", "size")))
    by_role["quantity"] = "gp_share"
    by_role["role"] = by_role["role"].astype(str)

    out = pd.concat([overall, by_role], ignore_index=True)
    out = out.sort_values(["role", "season"]).reset_index(drop=True)
    out["yoy_pct"] = out.groupby("role")["rate"].pct_change() * 100
    return out


# ── Drift vs shock ────────────────────────────────────────────────────────────

def drift_vs_shock(rates: pd.DataFrame, group: str = "quantity") -> pd.DataFrame:
    """Separate the part of a league series a trend can fix from the part it cannot.

    A log-linear trend is fitted per quantity, `log(rate) ~ a + b*season_index`. Logs make
    `b` a constant *percentage* per season — the form a multiplicative rate model uses — and
    put every quantity on a comparable scale despite wildly different units.

    **The load-bearing fact, and it is an identity rather than a finding:** subtracting a
    linear trend changes the *mean* of the year-over-year changes and leaves their
    *variance* exactly unchanged, because `diff(a + b*x)` is the constant `b`. So the two
    candidate season terms do different jobs and **neither substitutes for the other**:

    - a **trend fixed effect** removes `yoy_mean_pct` — the systematic drift that makes a
      carry-forward predictor biased in the same direction every single year;
    - a **year-level random effect** is the only thing that addresses `yoy_sd_pct`, the
      irreducible year-to-year spread, which a trend term cannot touch.

    `level_resid_sd_pct` is the sd of the detrended log level and is the σ to give a year
    random effect that sits *on top of* a fitted trend. Where no trend is warranted, use
    `yoy_sd_pct` instead, since the carry-forward lag is then the whole error.
    """
    rows = []
    for name, d in rates.groupby(group):
        d = d.sort_values("season")
        rate = d["rate"].to_numpy(dtype=float)
        keep = np.isfinite(rate) & (rate > 0)
        if keep.sum() < 4:
            continue
        rate = rate[keep]
        y = np.log(rate)
        x = np.arange(len(y), dtype=float)

        slope, intercept = np.polyfit(x, y, 1)
        fitted = intercept + slope * x
        ss_tot = float(((y - y.mean()) ** 2).sum())
        trend_r2 = float(1 - ((y - fitted) ** 2).sum() / ss_tot) if ss_tot > 0 else np.nan

        # Reported as true percentage change, not log-points: over a 24% move the two
        # differ by several points and the log version reads as a larger swing than
        # actually occurred.
        yoy = (rate[1:] / rate[:-1] - 1.0) * 100

        rows.append({
            group: name,
            "seasons": int(len(y)),
            "rate_min": float(rate.min()),
            "rate_max": float(rate.max()),
            "max_over_min": float(rate.max() / rate.min()),
            "trend_pct_per_season": float((np.exp(slope) - 1) * 100),
            "trend_r2": trend_r2,
            "yoy_mean_pct": float(yoy.mean()),
            "yoy_sd_pct": float(yoy.std(ddof=1)),
            "level_resid_sd_pct": float(np.std(y - fitted, ddof=1) * 100),
            "worst_yoy_pct": float(yoy[np.abs(yoy).argmax()]),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["trend_worth_extrapolating"] = ((out["trend_r2"] >= TREND_R2_BAR)
                                        & (out["trend_pct_per_season"].abs() >= TREND_SIZE_BAR))
    # Every quantity has a nonzero `yoy_sd_pct`, so the year effect is never optional; the
    # trend is what varies from quantity to quantity.
    out["recommendation"] = np.where(
        out["trend_worth_extrapolating"], "trend + year effect", "year effect only")
    return out.sort_values("yoy_sd_pct", ascending=False).reset_index(drop=True)


# ── What ignoring it costs ────────────────────────────────────────────────────

def carry_forward_bias(design: pd.DataFrame, test_seasons: int = 2) -> pd.DataFrame:
    """Systematic bias of the no-fit floor on the held-out seasons, per component.

    `carry_forward` is prior per-36 rate x actual minutes, so it lags any league-level move
    by exactly one season. That makes its bias a **direct measurement of the season effect**
    on the quantity the heads are actually scored on — and because neither the floor nor any
    fitted head carries a season term, the fitted heads inherit it.

    Reported as a percentage of the realized total, since the components differ by an order
    of magnitude in scale.
    """
    _, test = split_seasons(design, test_seasons)
    rows = []
    for c in COUNT_HEADS:
        y = test[c].to_numpy(dtype=float)
        f = carry_forward(test, c)
        rows.append({"component": c, "season": "all", "n": len(test),
                     "observed_mean": float(y.mean()),
                     "bias": float((f - y).mean()),
                     "bias_pct": float(100 * (f - y).sum() / y.sum()) if y.sum() else np.nan})
        for season in sorted(test["season"].unique()):
            m = (test["season"] == season).to_numpy()
            rows.append({"component": c, "season": season, "n": int(m.sum()),
                         "observed_mean": float(y[m].mean()),
                         "bias": float((f[m] - y[m]).mean()),
                         "bias_pct": float(100 * (f[m] - y[m]).sum() / y[m].sum())
                         if y[m].sum() else np.nan})
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    lengths_path = features_dir / "game_length.parquet"
    lengths = pd.read_parquet(lengths_path) if lengths_path.exists() else None
    if lengths is None:
        print("  game_length.parquet absent — skipping minutes_share "
              "(run `make game-length`)")

    rates = league_rates(targets, lengths)
    summary = drift_vs_shock(rates)

    availability_path = features_dir / "availability_features.parquet"
    if availability_path.exists():
        avail = availability_rates(pd.read_parquet(availability_path))
        rates = pd.concat([rates, avail.assign(kind="availability")], ignore_index=True)
        summary = pd.concat(
            [summary,
             drift_vs_shock(avail.assign(quantity=avail["quantity"] + " [" + avail["role"] + "]"))],
            ignore_index=True)
    else:
        print("  availability_features.parquet absent — skipping gp_share "
              "(run `make availability`)")

    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    bias = carry_forward_bias(design)

    print(f"Season effects: {rates['season'].nunique()} seasons, "
          f"{rates['quantity'].nunique()} quantities\n")

    print("Drift vs shock — what a TREND fixed effect fixes, and what only a YEAR effect can")
    cols = ["quantity", "max_over_min", "trend_pct_per_season", "trend_r2",
            "yoy_mean_pct", "yoy_sd_pct", "level_resid_sd_pct", "worst_yoy_pct",
            "recommendation"]
    print(summary[cols].round(3).to_string(index=False))
    print(f"\n  A trend is called extrapolable at trend_r2 >= {TREND_R2_BAR} AND "
          f"|slope| >= {TREND_SIZE_BAR}%/season.")
    print("  These are COMPLEMENTARY, not alternatives, and that is an identity rather "
          "than a finding:\n  subtracting a linear trend shifts the MEAN of the "
          "year-over-year changes and leaves their\n  VARIANCE untouched, since "
          "diff(a + b*x) is the constant b. So a trend term removes\n  `yoy_mean_pct` "
          "(the every-year-same-direction bias) and only a year-level random effect\n"
          "  addresses `yoy_sd_pct`. Give that effect `level_resid_sd_pct` when a trend is "
          "also fitted,\n  and `yoy_sd_pct` when it is not.")

    print("\nWhat ignoring it costs — no-fit floor bias on the held-out seasons:")
    piv = bias.pivot_table(index="component", columns="season", values="bias_pct")
    print(piv.round(1).sort_values("all").to_string())
    print("  The floor carries prior per-36 forward, so it lags any league move by exactly "
          "one season.\n  No fitted head corrects this: none of them has a season term.")

    print("\n  A league shift is perfectly correlated across players, so it does NOT "
          "diversify away in a\n  portfolio — unlike the shared-beta term, which is worth "
          "+0.2% on a 15-man roster.")

    artifacts = {
        "league_rates": (rates, out_dir / "season_effects_league_rates.csv"),
        "summary": (summary, out_dir / "season_effects_summary.csv"),
        "carry_forward_bias": (bias, out_dir / "season_effects_carry_forward_bias.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
