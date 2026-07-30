"""Measure what own-team context is worth, above the player's own prior season.

The question every team-context feature has to answer: given that we already know
what P did last season, does knowing who he plays *with* next season tell us
anything more? Anything that survives that control is real; anything that does not
is a restatement of P's own prior form.

Protocol (unchanged from the measurement it supersedes, so the numbers are
comparable):

    controls   P's own season S-1 dk_pts/game, minutes/game and usage
    features   own-team context for P's season-S roster
    outcomes   P's season-S per-36 components, minutes/game, per-game dk_pts

Reported per (feature, outcome): the partial correlation after residualizing both
sides on the controls, and — for per-game dk_pts — the incremental R² of adding the
feature to the control regression.

The split between rate outcomes and the minutes outcome is the point. Team context
acts on them with opposite signs, so a feature can be strongly predictive of both
and still look like nothing against per-game dk_pts, which is their product.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

CONTROL_COLS = ["prior_dk_pts_per_game", "prior_min", "prior_usg"]

RATE_SUFFIX = "_per36"

# Below this |r| against per-game dk_pts a feature has no sign for `net_dk_movement` to
# agree with, so it is excluded from the sign check rather than failing it. The recorded
# nulls sit at 0.005 (`role_crowding`) and 0.008 (`team_pace`); the features that are not
# nulls sit at 0.09-0.16.
SIGN_CHECK_MIN_R = 0.02

# Per-36 components, as named in the season matrix.
COMPONENT_COLS = {
    "pts_per36": "bas_pts", "fg3m_per36": "bas_fg3m", "reb_per36": "bas_reb",
    "ast_per36": "bas_ast", "stl_per36": "bas_stl", "blk_per36": "bas_blk",
    "tov_per36": "bas_tov",
}

CONTEXT_FEATURES = [
    "role_crowding", "teammate_usage_max", "teammate_usage_sum", "teammate_usage_load",
    "teammate_assist_supply", "teammate_spacing", "team_pace", "roster_coverage",
]


# ── Regression helpers ────────────────────────────────────────────────────────

def _design(X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X])


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    """Residuals of y after least-squares regression on `controls` plus an intercept."""
    A = _design(controls)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return y - A @ beta


def r_squared(y: np.ndarray, X: np.ndarray) -> float:
    A = _design(X)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    tss = float(((y - y.mean()) ** 2).sum())
    return 1.0 - float((resid ** 2).sum()) / tss if tss > 0 else np.nan


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    """Correlation of x and y once both are stripped of what the controls explain."""
    rx, ry = residualize(x, controls), residualize(y, controls)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ── Panel assembly ────────────────────────────────────────────────────────────

def build_panel(matrix: pd.DataFrame, context: pd.DataFrame,
                seasons: list[str]) -> pd.DataFrame:
    """One row per player-season transition: S-1 controls, S context, S outcomes.

    `context` is already keyed on the season being predicted — the corrected
    construction describes P's season-S roster, so no lagging happens here.
    """
    order = {s: i for i, s in enumerate(seasons)}
    nxt = {s: seasons[i + 1] for s, i in order.items() if i + 1 < len(seasons)}

    prior = matrix[["player_id", "season", "dk_pts_per_game", "min", "adv_usg_pct"]].copy()
    prior = prior.rename(columns={"dk_pts_per_game": "prior_dk_pts_per_game",
                                  "min": "prior_min", "adv_usg_pct": "prior_usg"})
    prior["season"] = prior["season"].map(nxt)
    prior = prior.dropna(subset=["season"])

    outcome_cols = ["player_id", "season", "dk_pts_per_game", "min",
                    *COMPONENT_COLS.values()]
    outcomes = matrix[[c for c in outcome_cols if c in matrix.columns]].copy()
    outcomes = outcomes.rename(columns={v: k for k, v in COMPONENT_COLS.items()})
    outcomes = outcomes.rename(columns={"min": "minutes"})

    panel = outcomes.merge(prior, on=["player_id", "season"], how="inner")
    panel = panel.merge(context, on=["player_id", "season"], how="inner")
    return panel.dropna(subset=CONTROL_COLS + ["dk_pts_per_game", "minutes"])


# ── The measurement ───────────────────────────────────────────────────────────

def season_dummies(seasons: pd.Series) -> np.ndarray:
    """One-hot season indicators, first level dropped (the intercept carries it).

    Mandatory on a 30-season panel. Floor spacing and pace both roughly doubled over
    the sample, and so did scoring, so *any* feature built from them correlates with
    *any* rising outcome unless season is absorbed. Pooled estimates here are era
    trends wearing a team-context costume.
    """
    d = pd.get_dummies(seasons, drop_first=True)
    return d.to_numpy(dtype=float)


def control_matrix(sub: pd.DataFrame, season_fe: bool) -> np.ndarray:
    base = sub[CONTROL_COLS].to_numpy(dtype=float)
    if not season_fe:
        return base
    return np.column_stack([base, season_dummies(sub["season"])])


# ── Cross-component cancellation ──────────────────────────────────────────────
#
# The strongest measured argument for the component-head architecture, and the one that
# needs a *unit*. A partial correlation cannot be DK-weighted: it is dimensionless, so
# `1.25 * r_reb + 1.5 * r_ast` adds numbers that are not in the same space. The per-sd
# effect in the component's **own units** can be, and it comes off the same residuals:
#
#     effect_c = slope(residual y_c on residual x) * sd(residual x) = r_c * sd(residual y_c)
#
# so one extra quantity — the residual sd of each component — turns the existing partial
# correlations into DK-commensurable movements. Then:
#
#     gross = sum_c |w_c * effect_c|      what the component heads see
#     net   = |sum_c w_c * effect_c|      what a single dk_pts head sees
#
# in dk_pts per 36 minutes, since the component outcomes are per-36 rates.

# The scoring weights, imported rather than retyped — `targets.py` is the single source.
from src.features.targets import DK_WEIGHTS  # noqa: E402


def dk_component_weights(outcomes: list[str]) -> dict[str, float]:
    """Map the per-36 outcome names onto their DK coefficients.

    Only the seven scoring components appear. `minutes` and `dk_pts_per_game` are outcomes
    of a different kind and must not enter the sum — minutes is the exposure the rates are
    already divided by, and dk_pts_per_game is the aggregate the cancellation is measured
    *against*.
    """
    return {f"{c}{RATE_SUFFIX}": w for c, w in DK_WEIGHTS.items()
            if f"{c}{RATE_SUFFIX}" in outcomes}


def cancellation(effects: dict[str, float], weights: dict[str, float]) -> dict:
    """Gross vs net DK movement, and the ratio between them."""
    terms = [weights[c] * effects[c] for c in weights
             if c in effects and np.isfinite(effects[c])]
    if not terms:
        return {"gross_dk_movement": np.nan, "net_dk_movement": np.nan,
                "cancellation_ratio": np.nan}
    gross = float(np.abs(terms).sum())
    net = float(np.sum(terms))
    return {"gross_dk_movement": gross, "net_dk_movement": net,
            "cancellation_ratio": gross / abs(net) if net != 0 else np.inf}


def measure(panel: pd.DataFrame, features: list[str] = None,
            season_fe: bool = True) -> pd.DataFrame:
    """Partial correlation of every context feature against every outcome.

    Also emits, per feature, the per-sd effect in each component's own units
    (`effect_<outcome>`) and the DK-weighted gross/net/cancellation triple built from them.
    All of it comes off the *same* residuals as the correlations, which is what makes
    `net_dk_movement` and `r_dk_pts_per_game` guaranteed comparable in sign.
    """
    features = features or [f for f in CONTEXT_FEATURES if f in panel.columns]
    outcomes = [c for c in list(COMPONENT_COLS) + ["minutes", "dk_pts_per_game"]
                if c in panel.columns]
    weights = dk_component_weights(outcomes)

    rows = []
    for feat in features:
        sub = panel.dropna(subset=[feat])
        controls = control_matrix(sub, season_fe)
        x = sub[feat].to_numpy(dtype=float)
        base_r2 = r_squared(sub["dk_pts_per_game"].to_numpy(dtype=float), controls)
        with_r2 = r_squared(sub["dk_pts_per_game"].to_numpy(dtype=float),
                            np.column_stack([controls, x]))
        row = {"feature": feat, "n": len(sub),
               "r2_controls": base_r2, "r2_with_feature": with_r2,
               "delta_r2": with_r2 - base_r2}
        effects = {}
        for out in outcomes:
            ok = sub[out].notna().to_numpy()
            y = sub[out].to_numpy(dtype=float)[ok]
            r = partial_corr(x[ok], y, controls[ok])
            row[f"r_{out}"] = r
            # Per one sd of the feature, in the component's own units.
            effects[out] = r * float(residualize(y, controls[ok]).std(ddof=1))
            row[f"effect_{out}"] = effects[out]
        row |= cancellation(effects, weights)
        rows.append(row)
    return pd.DataFrame(rows)


def block_value(panel: pd.DataFrame, features: list[str] = None,
                season_fe: bool = True) -> dict:
    """Incremental R² of the whole context block on per-game dk_pts."""
    features = features or [f for f in CONTEXT_FEATURES if f in panel.columns]
    sub = panel.dropna(subset=features)
    controls = control_matrix(sub, season_fe)
    y = sub["dk_pts_per_game"].to_numpy(dtype=float)
    base = r_squared(y, controls)
    full = r_squared(y, np.column_stack([controls, sub[features].to_numpy(dtype=float)]))
    return {"n": len(sub), "r2_controls": base, "r2_with_block": full,
            "delta_r2": full - base}


# ── Roster description coverage ───────────────────────────────────────────────
#
# `team_context.py` aggregates the season-S roster described by season S-1 stats, and the
# single largest bias available in that construction is dropping the roster members S-1
# cannot describe. `CLAUDE.md` quotes 15.9% of roster minutes, split 9.4% true rookies /
# 5.4% sub-threshold / 1.1% returnees, and until now none of it had a target.
#
# Two things make this more than a `value_counts` on the parquet:
#
# 1. **The denominator must be realized season-S minutes, not the S-1 minutes the aggregate
#    weights by.** A rookie has zero S-1 minutes, so weighting by them reports every roster
#    as fully covered — the failure mode reading as a success, again.
# 2. **"No usable S-1 row" is relative to the QUALIFIED matrix**, not the inclusive one.
#    `stats_source` only knows prior / stale / rookie; a player with 150 sub-threshold
#    minutes in S-1 is `prior` there but is absent from `season_matrix_tier*.parquet`, which
#    is what the PCA, the archetypes and the mechanism scalars are fitted on.

DESCRIPTION_SOURCES = ("prior", "sub_threshold", "returnee", "rookie")
UNDESCRIBED_SOURCES = ("sub_threshold", "returnee", "rookie")

# Team-seasons named individually in the report, worst first.
N_WORST_TEAM_SEASONS = 10

# A window this long admits everyone who played for the team at any point. It is NOT the
# roster the model can know before the season, and it is measured only for the contrast —
# see `roster_coverage_profile`.
WHOLE_SEASON_WINDOW = 82


def description_source(roster: pd.DataFrame, roster_frame: pd.DataFrame,
                       qualified: pd.DataFrame, seasons: list[str]) -> pd.Series:
    """The four-way split the coverage figure is quoted on.

        prior          season S-1 row in the **qualified** matrix — genuinely described
        sub_threshold  an S-1 row in the inclusive frame, but below the matrix's filter
        returnee       no S-1 row at all, but an earlier one — absent the previous season
        rookie         never played

    Derived here from the two matrices rather than read off `team_context`'s `stats_source`,
    for two reasons: `stats_source` cannot see the qualified/sub-threshold line at all (it
    knows only prior / stale / rookie against the *inclusive* frame), and deriving it makes
    the definition testable without rebuilding the whole team-context pipeline. `run`
    cross-checks the two anyway.

    Seasons are matched on the season *index*, never by decrementing the label, so a player
    who missed a whole year cannot pair across the gap — he is a `returnee`, which is the
    category that exists for him.
    """
    order = {s: i for i, s in enumerate(seasons)}
    incl = set(zip(roster_frame["player_id"], roster_frame["season"].map(order)))
    qual = set(zip(qualified["player_id"], qualified["season"].map(order)))
    debut = (roster_frame.assign(_i=roster_frame["season"].map(order))
             .groupby("player_id")["_i"].min().to_dict())

    idx = roster["season"].map(order)
    out = []
    for pid, i in zip(roster["player_id"], idx):
        if pd.isna(i) or debut.get(pid, np.inf) >= i:
            out.append("rookie")
        elif (pid, i - 1) not in incl:
            out.append("returnee")
        else:
            out.append("prior" if (pid, i - 1) in qual else "sub_threshold")
    return pd.Series(out, index=roster.index, name="description_source")


def attach_season_minutes(roster: pd.DataFrame, roster_frame: pd.DataFrame) -> pd.DataFrame:
    """Join each roster row's **realized** season-S minutes — the honest denominator."""
    minutes = (roster_frame[["player_id", "season", "min_total"]]
               .rename(columns={"min_total": "season_minutes"}))
    out = roster.merge(minutes, on=["player_id", "season"], how="left")
    out["season_minutes"] = pd.to_numeric(out["season_minutes"],
                                         errors="coerce").fillna(0.0)
    return out


def roster_coverage_profile(roster: pd.DataFrame, roster_frame: pd.DataFrame,
                            qualified: pd.DataFrame, seasons: list[str], tier: str,
                            window: str = "season_start",
                            n_worst: int = N_WORST_TEAM_SEASONS) -> pd.DataFrame:
    """League shares, the per-team-season distribution, and the named worst offenders.

    Head-count shares ride along beside the minutes-weighted ones deliberately: the gap
    between them is the point. A rookie is ~14% of roster *rows* and ~9% of roster
    *minutes*, so neither number substitutes for the other.

    `window` labels which roster this is — `season_start` is the one `team_context` actually
    aggregates and therefore the one whose coverage describes a real bias; `whole_season`
    admits mid-season arrivals and is measured only for the contrast.
    """
    from src.features.team_context import coverage_report

    df = attach_season_minutes(roster, roster_frame)
    df["description_source"] = description_source(df, roster_frame, qualified, seasons)
    total_minutes = float(df["season_minutes"].sum())

    def _row(scope: str, key: str, metric: str, value: float, n: int) -> dict:
        return {"tier": tier, "window": window, "scope": scope, "key": key,
                "metric": metric, "value": value, "n": int(n)}

    rows = []
    for source in DESCRIPTION_SOURCES:
        m = df["description_source"] == source
        rows += [
            _row("league", source, "share_of_minutes",
                 float(df.loc[m, "season_minutes"].sum() / total_minutes)
                 if total_minutes > 0 else np.nan, int(m.sum())),
            _row("league", source, "share_of_headcount", float(m.mean()), int(m.sum())),
        ]
    undescribed = df["description_source"].isin(UNDESCRIBED_SOURCES)
    rows += [
        _row("league", "undescribed", "share_of_minutes",
             float(df.loc[undescribed, "season_minutes"].sum() / total_minutes)
             if total_minutes > 0 else np.nan, int(undescribed.sum())),
        _row("league", "undescribed", "share_of_headcount",
             float(undescribed.mean()), int(undescribed.sum())),
        _row("league", "all", "roster_rows", float(len(df)), len(df)),
        _row("league", "all", "realized_minutes", total_minutes, len(df)),
        _row("league", "all", "share_rows_matched_to_season_minutes",
             float((df["season_minutes"] > 0).mean()), len(df)),
    ]

    # Per team-season, through team_context's own helper so there is one definition of
    # "share of a roster's minutes by source".
    per_team = coverage_report(df.assign(stats_source=df["description_source"]),
                              sources=DESCRIPTION_SOURCES)
    share_cols = [f"share_{s}" for s in DESCRIPTION_SOURCES]
    per_team["share_undescribed"] = per_team[
        [f"share_{s}" for s in UNDESCRIBED_SOURCES]].sum(axis=1)
    per_team["share_sum"] = per_team[share_cols].sum(axis=1)

    for source in DESCRIPTION_SOURCES + ("undescribed",):
        col = f"share_{source}"
        for q, label in ((0.50, "p50"), (0.90, "p90"), (1.0, "max")):
            rows.append(_row("team_season_distribution", source, label,
                             float(per_team[col].quantile(q)), len(per_team)))
        rows.append(_row("team_season_distribution", source, "mean",
                         float(per_team[col].mean()), len(per_team)))

    worst = per_team.nlargest(n_worst, "share_undescribed")
    for r in worst.itertuples():
        rows.append(_row("worst_team_seasons",
                         f"{r.team_abbreviation} {r.season}", "share_undescribed",
                         float(r.share_undescribed), int(r.total)))

    # The verification: the four shares must partition every team-season exactly.
    deviation = float((per_team["share_sum"] - 1.0).abs().max())
    rows.append(_row("checks", "shares_sum_to_one", "max_abs_deviation", deviation,
                     len(per_team)))

    for r in per_team.itertuples():
        for source in DESCRIPTION_SOURCES:
            rows.append(_row("team_season", f"{r.team_abbreviation} {r.season}",
                             f"share_{source}", float(getattr(r, f"share_{source}")),
                             int(r.total)))
    return pd.DataFrame(rows)


def run(cfg: dict, tier: str = "A") -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = pd.read_parquet(features_dir / f"season_matrix_tier{tier}.parquet")
    context = pd.read_parquet(features_dir / f"team_context_tier{tier}.parquet")
    panel = build_panel(matrix, context, cfg["data"]["seasons"])

    show = ["feature", "n", "r_pts_per36", "r_minutes", "r_dk_pts_per_game", "delta_r2",
            "gross_dk_movement", "net_dk_movement", "cancellation_ratio"]
    print(f"\nTier {tier}: {len(panel):,} player-season transitions")

    for season_fe in (False, True):
        table = measure(panel, season_fe=season_fe)
        block = block_value(panel, season_fe=season_fe)
        label = "with season fixed effects" if season_fe else "pooled (no season control)"
        print(f"\n── {label} ──")
        print(f"Controls alone: R² = {block['r2_controls']:.4f}   "
              f"+ context block: R² = {block['r2_with_block']:.4f} "
              f"(+{block['delta_r2']:.4f}, n={block['n']:,})")
        print(table[[c for c in show if c in table]].round(4).to_string(index=False))
        if season_fe:
            dest = out_dir / f"team_context_value_tier{tier}.csv"
            table.to_csv(dest, index=False)

            # The verification: DK-weighting must not flip a feature's sign against the
            # aggregate it is being compared with. A disagreement means the weights are
            # wrong — but only for a feature that *has* a sign. `role_crowding` (r = -0.005)
            # and `team_pace` (r = +0.008) are recorded nulls, and the sign of a null is
            # noise on both sides, so checking them makes the guard cry wolf on tier B.
            signs = table.dropna(subset=["net_dk_movement", "r_dk_pts_per_game"])
            checkable = signs[signs["r_dk_pts_per_game"].abs() >= SIGN_CHECK_MIN_R]
            skipped = signs[signs["r_dk_pts_per_game"].abs() < SIGN_CHECK_MIN_R]
            flipped = checkable[np.sign(checkable["net_dk_movement"])
                                != np.sign(checkable["r_dk_pts_per_game"])]
            print("\nCross-component cancellation, dk_pts per 36 minutes per sd of the "
                  "feature —\ngross is what the component heads see, net is what a single "
                  "dk_pts head sees:")
            print(table[["feature", "gross_dk_movement", "net_dk_movement",
                         "cancellation_ratio", "r_dk_pts_per_game"]]
                  .round(3).to_string(index=False))
            if len(flipped):
                print(f"  ⚠️  {len(flipped)} feature(s) whose net movement disagrees in "
                      f"sign with r_dk_pts_per_game: "
                      f"{', '.join(flipped['feature'])}\n      The DK weighting is wrong "
                      "— net movement and the aggregate correlation must agree.")
            else:
                print(f"  every net movement agrees in sign with r_dk_pts_per_game across "
                      f"the {len(checkable)} features\n  with "
                      f"|r| >= {SIGN_CHECK_MIN_R:g}, so the DK weighting holds.")
            if len(skipped):
                print(f"  not checked (|r_dk_pts_per_game| < {SIGN_CHECK_MIN_R:g}, i.e. no "
                      f"sign to agree with): {', '.join(skipped['feature'])}")
            worst = table.nlargest(1, "cancellation_ratio")
            if len(worst):
                w = worst.iloc[0]
                print(f"  worst canceller: {w['feature']} at "
                      f"{w['cancellation_ratio']:.1f}x "
                      f"({w['gross_dk_movement']:.2f} gross into "
                      f"{w['net_dk_movement']:+.2f} net)")

    # ── roster description coverage ──────────────────────────────────────────
    from src.features.team_context import season_start_rosters

    seasons = cfg["data"]["seasons"]
    raw_dir = cfg["data"]["raw_dir"]
    window_games = cfg.get("features", {}).get("team_context", {}).get(
        "roster_window_games", 10)
    roster_frame = pd.read_parquet(
        features_dir / f"season_matrix_roster_tier{tier}.parquet",
        columns=["player_id", "season", "min_total"])

    blocks = []
    for label, games in (("season_start", window_games),
                         ("whole_season", WHOLE_SEASON_WINDOW)):
        rosters = season_start_rosters(seasons, raw_dir, games)
        # The first covered season has nothing behind it to describe anyone with, so
        # `describe_roster` drops it; excluded here too or it reads as an all-rookie league.
        rosters = rosters[rosters["season"] != seasons[0]]
        blocks.append(roster_coverage_profile(rosters, roster_frame, matrix, seasons,
                                             tier, label))
    coverage = pd.concat(blocks, ignore_index=True)
    cov_dest = out_dir / f"roster_coverage_profile_tier{tier}.csv"
    coverage.to_csv(cov_dest, index=False)

    print(f"\n── roster description coverage, tier {tier} ──")
    for label in ("season_start", "whole_season"):
        sub = coverage[coverage["window"] == label]
        league = sub[sub["scope"] == "league"].set_index(["key", "metric"])["value"]
        note = (f"the roster team_context aggregates — first {window_games} team games"
                if label == "season_start"
                else "everyone who played for the team, INCLUDING mid-season arrivals")
        print(f"\n  {label} ({note}):")
        for source in DESCRIPTION_SOURCES + ("undescribed",):
            print(f"    {source:<14} {league[(source, 'share_of_minutes')]:>7.2%} of "
                  f"minutes   {league[(source, 'share_of_headcount')]:>7.2%} of roster rows")
        dist = sub[(sub["scope"] == "team_season_distribution")
                   & (sub["key"] == "undescribed")].set_index("metric")["value"]
        print(f"    per team-season undescribed: mean {dist['mean']:.1%}, "
              f"p50 {dist['p50']:.1%}, p90 {dist['p90']:.1%}, max {dist['max']:.1%}")
        worst = sub[sub["scope"] == "worst_team_seasons"]
        print("    worst rosters: " + ", ".join(f"{r.key} {r.value:.0%}"
                                                for r in worst.head(5).itertuples()))
    print("\n  Weighted by REALIZED season-S minutes. Weighting by the S-1 minutes the "
          "aggregate\n  actually uses is circular — a rookie has none, so every roster "
          "reads as fully covered.\n  The two windows differ because rookies and returnees "
          "arrive LATE: the recorded 15.9%\n  is the whole-season roster, but the shipped "
          "aggregate only ever sees the season-start one.")
    check = coverage[coverage["scope"] == "checks"]
    print(f"  the four shares partition every team-season: max deviation from 1.0 "
          f"is {check['value'].max():.2e}")

    # The head-count mix must agree with `stats_source` on the parquet, which is the
    # independent construction. `stats_source` cannot see the sub-threshold line, so
    # prior+sub_threshold is what corresponds to its `prior`.
    if "stats_source" in context.columns:
        mix = context["stats_source"].value_counts(normalize=True)
        start = coverage[(coverage["window"] == "season_start")
                         & (coverage["scope"] == "league")
                         & (coverage["metric"] == "share_of_headcount")]
        got = start.set_index("key")["value"]
        pairs = [("prior", got.get("prior", 0.0) + got.get("sub_threshold", 0.0)),
                 ("stale", got.get("returnee", 0.0)), ("rookie", got.get("rookie", 0.0))]
        print("  cross-check against team_context's own `stats_source` head-count mix:")
        for name, derived in pairs:
            print(f"    {name:<8} parquet {mix.get(name, 0.0):.3%}  derived here "
                  f"{derived:.3%}  gap {derived - mix.get(name, 0.0):+.3%}")

    print(f"\n→ {dest}")
    print(f"→ {cov_dest}  ({len(coverage):,} rows)")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    for t in ("A", "B"):
        run(cfg, t)
