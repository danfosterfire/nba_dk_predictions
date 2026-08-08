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
measuring the no-fit floor's systematic bias on the **validation** seasons.

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
- **The three-point MIX is the opposite** — `fg3a_pct` climbs 2.64x with a trend R2 of 0.93
  at +3.58%/season. Carry-forward lags a monotone trend by exactly one season, which is a
  *predictable* bias. (The retired `fg3a` **count** series read a 2.97x climb at +4.07%;
  the shot-attempt basis split it into volume and mix, and the trend is almost all mix.)
- **The floor pays for it, and the bias is a LAG rather than a level.** On the validation
  seasons the sign of every component's bias is the opposite of that season's league move
  in **13 of 14** cells, correlating at **−0.94** — `fta` runs −4.4% into the league's
  +7.3% and +10.7% into its −7.5%. So the cost is real and its *direction reverses with the
  league*, which is exactly why a trend term cannot fix it and a year effect is the only
  candidate that addresses the spread.

Usage:
    python -m src.eda.season_effects
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.component_rates import (COUNT_HEADS, CONVERSION_HEADS, PER36,
                                        build_design, carry_forward)
from src.models.held_out import selection_split

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


# ── Regime confounds: is the trend even a trend? ──────────────────────────────
#
# Two known interventions sit inside the 30-season window, and they are different SHAPES,
# which is why they get different tests rather than one generic "break" scan:
#
#   2019-20 / 2020-21   COVID health protocols — bubble, quarantines, contact tracing.
#                       A **transient regime**: two seasons unlike their neighbours, after
#                       which the league returns. The right treatment is an indicator on
#                       those seasons, and the right question is how much the trend moves
#                       when they are excluded.
#   2023-24             The Player Participation Policy. A **permanent** rule change, so
#                       the right test is a level-and-slope break from that season on.
#
# Extrapolating one smooth line across either is the specific error this measures. The
# decision-relevant output is not the p-value, it is `next_season_shift_pct`: how far the
# one-season-ahead extrapolation moves once the regime is handled. A statistically
# significant break that moves next season's forecast by 0.1% changes nothing.

COVID_SEASONS = ("2019-20", "2020-21")
POLICY_BREAK_SEASON = "2023-24"


def series_label(rates: pd.DataFrame) -> pd.DataFrame:
    """One unique key per league series, folding the availability role into the name.

    `gp_share` is emitted once per role bucket, so `quantity` alone repeats a season five
    times and any per-series regression silently fits a smear of five curves. This is the
    same `gp_share [30+ mpg]` label `run` already builds for the summary table, hoisted so
    every consumer keys on the same thing.
    """
    out = rates.copy()
    role = out["role"] if "role" in out.columns else pd.Series(np.nan, index=out.index)
    labelled = role.notna() & (role.astype(str) != "nan")
    out["series"] = out["quantity"].astype(str)
    out.loc[labelled, "series"] = (out.loc[labelled, "quantity"].astype(str)
                                   + " [" + role[labelled].astype(str) + "]")
    return out


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, float(((y - X @ coef) ** 2).sum())


def _extrapolate(coef: np.ndarray, X_next: np.ndarray) -> float:
    return float(X_next @ coef)


def regime_tests(rates: pd.DataFrame,
                 covid_seasons: tuple[str, ...] = COVID_SEASONS,
                 break_season: str = POLICY_BREAK_SEASON,
                 group: str = "series") -> pd.DataFrame:
    """Per series: does a COVID indicator or a 2023-24 break change the extrapolation?

    Both arms are nested F tests against the same plain log-linear trend, so the
    statistics are comparable to each other. `next_season_shift_pct` is the column that
    decides anything: it is the difference between what the plain trend predicts for the
    season *after* the sample and what the regime-aware model predicts, in percent. A
    trend term is only worth extrapolating across a regime if that number is small.

    The COVID arm is an indicator, not a break, because the two seasons are an
    interruption rather than a new level — modelling them as a permanent shift would
    push the whole post-2021 series onto the wrong intercept. The policy arm is a break
    (level **and** slope from 2023-24 on) because a rule change does not revert.
    """
    if group == "series" and "series" not in rates.columns:
        rates = series_label(rates)
    rows = []
    for name, d in rates.groupby(group):
        d = d.sort_values("season")
        keep = np.isfinite(d["rate"].to_numpy(dtype=float)) & (d["rate"] > 0)
        d = d[keep]
        if len(d) < 8:
            continue
        seasons = d["season"].to_numpy()
        y = np.log(d["rate"].to_numpy(dtype=float))
        t = np.arange(len(y), dtype=float)
        n = len(y)

        base = np.column_stack([np.ones(n), t])
        base_next = np.array([1.0, float(n)])
        coef0, sse0 = _ols(base, y)
        plain_next = _extrapolate(coef0, base_next)

        arms = {}
        covid = np.isin(seasons, list(covid_seasons)).astype(float)
        if 0 < covid.sum() < n:
            # The indicator is 0 for the next season by construction: COVID is over, so
            # the extrapolation uses the trend fitted with those two seasons' influence
            # removed rather than carried forward.
            arms["covid_indicator"] = (np.column_stack([base, covid]),
                                       np.r_[base_next, 0.0])

        after = (seasons >= break_season).astype(float)
        if 0 < after.sum() < n:
            first = t[after.astype(bool)][0]
            since = after * (t - first)
            # Two forms, because they fail differently. The level+slope break is the
            # general one and is what "test for a break" normally means; with only three
            # post-break seasons its slope is fitted on three points and extrapolating it
            # is worse than not testing at all. The level-only arm asks the narrower and
            # far more stable question a rule change actually poses: did the level step?
            arms["policy_break"] = (np.column_stack([base, after, since]),
                                    np.r_[base_next, 1.0, float(n) - first])
            arms["policy_break_level"] = (np.column_stack([base, after]),
                                          np.r_[base_next, 1.0])

        for arm, (X, x_next) in arms.items():
            coef, sse = _ols(X, y)
            df_num = X.shape[1] - base.shape[1]
            df_den = n - X.shape[1]
            f = ((sse0 - sse) / df_num) / (sse / df_den) if sse > 0 and df_den > 0 else np.nan
            shifted = _extrapolate(coef, x_next)
            rows.append({
                group: name, "arm": arm, "seasons": n,
                # How many seasons the regime term is fitted on. A slope estimated from
                # three points and then extrapolated is the reason `policy_break` moves
                # the forecast by 20%: read this column before believing that column.
                "regime_seasons": int(X[:, 2].astype(bool).sum()),
                "trend_pct_per_season": float((np.exp(coef0[1]) - 1) * 100),
                "trend_pct_per_season_adjusted": float((np.exp(coef[1]) - 1) * 100),
                "f_stat": float(f), "df_num": int(df_num), "df_den": int(df_den),
                "p_value": float(_f_sf(f, df_num, df_den)) if np.isfinite(f) else np.nan,
                # exp() of a log-scale difference, minus one: the percentage by which the
                # one-season-ahead forecast moves.
                "next_season_shift_pct": float((np.exp(shifted - plain_next) - 1) * 100),
                "regime_effect_pct": float((np.exp(coef[2]) - 1) * 100),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["significant"] = out["p_value"] < 0.05
    out["abs_shift"] = out["next_season_shift_pct"].abs()
    return (out.sort_values(["arm", "abs_shift"], ascending=[True, False])
            .drop(columns=["abs_shift"]).reset_index(drop=True))


def _f_sf(f: float, df_num: int, df_den: int) -> float:
    from scipy.stats import f as f_dist
    return float(f_dist.sf(f, df_num, df_den))


def year_shock_correlation(rates: pd.DataFrame, group: str = "series") -> pd.DataFrame:
    """Are the league's year-to-year shocks ONE common factor, or many independent ones?

    This is the question a simulator has to answer before it can draw a year effect at
    all. If detrended league movements are strongly correlated across quantities, one
    shared draw per simulated season is right and the eleven heads must not draw
    independently. If they are near-independent, eleven independent draws are right and a
    shared one would invent a league-wide factor that does not exist.

    Measured on the **detrended** log level — the residual after removing each quantity's
    own log-linear trend — because two quantities that both drift upward would otherwise
    correlate at +0.9 through their trends alone, which is drift and not shock.
    """
    if group == "series" and "series" not in rates.columns:
        rates = series_label(rates)
    wide = {}
    for name, d in rates.groupby(group):
        d = d.sort_values("season")
        d = d[np.isfinite(d["rate"].to_numpy(dtype=float)) & (d["rate"] > 0)]
        if len(d) < 8:
            continue
        y = np.log(d["rate"].to_numpy(dtype=float))
        t = np.arange(len(y), dtype=float)
        slope, intercept = np.polyfit(t, y, 1)
        wide[name] = pd.Series(y - (intercept + slope * t), index=d["season"].to_numpy())

    if len(wide) < 2:
        return pd.DataFrame()
    panel = pd.DataFrame(wide).dropna()
    corr = panel.corr()
    rows = []
    names = list(corr.columns)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            rows.append({"series_a": a, "series_b": b,
                         "corr": float(corr.loc[a, b]), "n_seasons": int(len(panel))})
    out = pd.DataFrame(rows).sort_values("corr", key=abs, ascending=False)
    return out.reset_index(drop=True)


# ── What ignoring it costs ────────────────────────────────────────────────────

def carry_forward_bias(design: pd.DataFrame, test_seasons: int = 2,
                       rates: pd.DataFrame | None = None) -> pd.DataFrame:
    """Systematic bias of the no-fit floor on the **validation** seasons, per component.

    `carry_forward` is prior per-36 rate x actual minutes, so it lags any league-level move
    by exactly one season. That makes its bias a **direct measurement of the season effect**
    on the quantity the heads are actually scored on — and because neither the floor nor any
    fitted head carries a season term, the fitted heads inherit it.

    Measured on validation since 2026-08-05, with every head that reads it
    (`src/models/season_terms.py`). It is not a selection, but it is the *premise* of one:
    this is the number that says what a season term would have to recover, and reading it
    off the held-out seasons while the arms compete on validation would mean sizing the
    prize on one split and paying for it on another. `src/models/held_out.py` also simply
    raises on the other frame now.

    Reported as a percentage of the realized total, since the components differ by an order
    of magnitude in scale.

    **Pass `rates` and the lag becomes checkable rather than asserted.** The mechanism
    claimed above — that the floor lags the league by one season — predicts a specific
    thing: the bias on season S should carry the *opposite* sign to the league's own move
    into S. `league_yoy_pct` and `opposes_league_move` put that prediction in the artifact
    per cell, so the headline is a measured relationship instead of a pair of components
    that happened to look biased across whichever two seasons were being scored. That
    distinction is not hypothetical here — the retired held-out reading was quoted as
    "`blk` is biased +6% in *both* seasons, which is drift, not noise", and on the
    validation seasons `blk` reverses sign.
    """
    _, test = selection_split(design, test_seasons)
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
    out = pd.DataFrame(rows)
    return out if rates is None else attach_league_move(out, rates)


def attach_league_move(bias: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """Join each per-season bias cell to that season's league move, and test the lag.

    The `season == "all"` rows are left null rather than joined to a pooled league move:
    averaging a bias across two seasons whose league moves point in opposite directions is
    exactly the summary that made the retired reading look like a standing level bias.
    """
    move = (rates[rates["kind"] == "count_per36"][["season", "quantity", "yoy_pct"]]
            .rename(columns={"quantity": "component", "yoy_pct": "league_yoy_pct"}))
    out = bias.merge(move, on=["component", "season"], how="left")
    out["opposes_league_move"] = np.where(
        out["league_yoy_pct"].notna(),
        out["bias_pct"] * out["league_yoy_pct"] < 0, np.nan)
    return out


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
    bias = carry_forward_bias(design, rates=rates)
    regimes = regime_tests(rates)
    shocks = year_shock_correlation(rates)

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

    print("\nWhat ignoring it costs — no-fit floor bias on the VALIDATION seasons:")
    piv = bias.pivot_table(index="component", columns="season", values="bias_pct")
    print(piv.round(1).sort_values("all").to_string())
    print("  The floor carries prior per-36 forward, so it lags any league move by exactly "
          "one season.\n  No fitted head corrects this: none of them has a season term.")

    per_season = bias[bias["opposes_league_move"].notna()] if \
        "opposes_league_move" in bias.columns else pd.DataFrame()
    if not per_season.empty:
        opposes = int(per_season["opposes_league_move"].sum())
        r = float(np.corrcoef(per_season["bias_pct"],
                              per_season["league_yoy_pct"])[0, 1])
        print(f"\n  The lag is CHECKABLE, not merely asserted: the bias opposes that "
              f"season's league move\n  in {opposes} of {len(per_season)} cells, "
              f"correlating at {r:+.3f}. So the cost is real and its DIRECTION\n  reverses "
              f"with the league — which is why a trend cannot fix it and only a year "
              f"effect\n  addresses the spread. Read a per-season cell, never the pooled "
              f"`all` row on its own:\n  averaging two seasons whose league moves oppose "
              f"each other reports a lag as a level.")

    print("\n  A league shift is perfectly correlated across players, so it does NOT "
          "diversify away in a\n  portfolio — unlike the shared-beta term, which is worth "
          "+0.2% on a 15-man roster.")

    if not regimes.empty:
        print("\nRegime confounds — a smooth trend extrapolated across a policy "
              "discontinuity is wrong.\n"
              "COVID (2019-20/2020-21) is a transient INDICATOR; the Player Participation "
              "Policy (2023-24)\nis a permanent level+slope BREAK. The decisive column is "
              "the last one, not the p-value:")
        cols = [c for c in ["series", "arm", "regime_seasons", "trend_pct_per_season",
                            "trend_pct_per_season_adjusted", "regime_effect_pct",
                            "p_value", "significant", "next_season_shift_pct"]
                if c in regimes.columns]
        for arm, block in regimes.groupby("arm"):
            print(f"\n  arm = {arm}  ({int(block['significant'].sum())} of {len(block)} "
                  f"significant at p < 0.05)")
            print(block[cols].drop(columns=["arm"]).round(3).to_string(index=False))
        worst = regimes.loc[regimes["next_season_shift_pct"].abs().idxmax()]
        print(f"\n  Largest move in the one-season-ahead extrapolation: "
              f"{worst['series']} under {worst['arm']}, "
              f"{worst['next_season_shift_pct']:+.2f}% off {int(worst['regime_seasons'])} "
              f"regime seasons.")
        print("  Read `regime_seasons` before `next_season_shift_pct`: the level+slope arm "
              "fits its slope\n  on the post-break seasons alone, so with three of them "
              "the extrapolation is noise. That\n  is itself the finding — it disqualifies "
              "trend extrapolation across the break rather than\n  supplying a better "
              "trend. `policy_break_level` is the stable form of the same question.")

    if not shocks.empty:
        strongest = shocks.iloc[0]
        print("\nAre the year-to-year shocks ONE common factor or many? "
              "(detrended log level, pairwise)")
        print(f"  {len(shocks)} pairs over {int(shocks['n_seasons'].iloc[0])} seasons: "
              f"mean r {shocks['corr'].mean():+.3f}, mean |r| "
              f"{shocks['corr'].abs().mean():.3f}, range "
              f"{shocks['corr'].min():+.3f} to {shocks['corr'].max():+.3f}")
        print(f"  strongest pair: {strongest['series_a']}–{strongest['series_b']} at "
              f"{strongest['corr']:+.3f}")
        print("  This decides whether a simulator draws ONE year effect per season and "
              "shares it across\n  every head, or one per head. A mean near zero with a "
              "large mean |r| is neither: the\n  shocks are correlated in specific PAIRS "
              "rather than loaded on a common factor, so the\n  year effect needs a "
              "correlation matrix, the same shape the residual copula already takes.")

    artifacts = {
        "league_rates": (rates, out_dir / "season_effects_league_rates.csv"),
        "summary": (summary, out_dir / "season_effects_summary.csv"),
        "carry_forward_bias": (bias, out_dir / "season_effects_carry_forward_bias.csv"),
        "regimes": (regimes, out_dir / "season_effects_regimes.csv"),
        "shock_correlation": (shocks, out_dir / "season_effects_shock_correlation.csv"),
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
