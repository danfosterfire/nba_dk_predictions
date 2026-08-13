"""P1 — the EDA gate: does a preseason delta carry anything the prior season does not?

`docs/preseason-plan.md` P1. Three measurements and one decision, for ~zero sampler cost,
**on the training seasons alone** — this module never calls anything that materializes
validation, let alone test, because its whole job is to say which heads are worth spending
P2–P4 on and a screen that reads the selection split has already spent it.

    (a) redundancy      how much of each preseason quantity is its prior-season equivalent
                        wearing a shorter sample
    (b) incremental     what the difference-coded block adds to each head's own metric,
                        above that head's own prior-season feature block
    (c) missingness     who has no preseason row — by age and by role — and whether the
                        hole predicts the season, because "rested veteran" and "injured
                        star" are different absences

**The rates bar is stated before the run and is quoted in the rate heads' own unit.** A
count head ships against a no-fit floor it beats by +0.0013 (`reb`) to +0.0334 (`stl`) of
validation R² *on the season total* (`outputs/predictions/component_rate_metrics.csv`). So
the preseason increment is measured the same way — `component_rates.fit_count_head`, R² on
the season total — and anything that cannot reach the bottom of that range does not earn an
arm. Comparing an R²-on-a-per-36-rate against an R²-on-a-season-total bar would have been
the same category error the project's own docs flag elsewhere, and it is the reason this
module refits the actual heads rather than running a generic partial-R².

## Three things that make a small ΔR² readable

1. **The score rows are not the fit rows.** The block is fitted on all but the last two
   *training* seasons and scored on those two, so an in-sample ΔR² inflated by the block's
   own column count cannot be mistaken for signal. Both are reported; the out-of-sample one
   is the decision column. The inner score split lands on 2020-21 and 2021-22 — the COVID
   72-game season with its December preseason, and the season after it — which is an
   awkward pair and a fair stress test, and is why the in-sample twin rides along.
2. **A shuffled null gives the ΔR² a zero point.** The block is permuted across rows within
   season, which keeps its marginal distribution and its season composition and destroys
   only the pairing with the player. `delta_r2_null_mean` / `null_sd` is what a block of
   this shape earns by chance at this n, and `z_vs_null` is the reading.
3. **Only covered seasons are in scope.** The panel is empty before 2003-04 and truncated
   in it, so a block that is structurally zero on eight training seasons would dilute every
   ΔR² by a fact about the API. `covered_seasons` restricts to `coverage_class == "covered"`
   and the restriction is reported rather than assumed — `docs/preseason-plan.md`'s
   "coverage interactions" risk, answered here rather than deferred.

## Difference coding, and why the fill value for a missing row does not matter

Every preseason quantity with a prior-season equivalent enters as a **delta on the head's
own link scale** — logit for shares, `log1p` for per-36 rates — so zero means "the preseason
agrees with the prior season", and a coefficient path through zero recovers the shipped head
exactly. Participation has no prior-season equivalent and enters as a level.

A player with no preseason appearance has no panel row, and every column of his block is
filled with a **constant**. With `has_preseason` in the block, that fill barely matters: a
column filled at constant `c` on the missing rows equals `x·1{present} + c·(1 −
has_preseason)`, so any two constants span the same column space and an *unpenalized* fit
is exactly invariant to the choice — which is what `tests/test_preseason_value.py` pins.
Under an L2 penalty the invariance is only up to the penalty's basis, since standardizing a
column whose fill moved is a different parameterization; that residue is second-order and is
the reason zero is chosen rather than being merely arbitrary, because zero is the value the
*nesting* argument wants. What no fill can do is tell a rested veteran from an injured star.
That is measurement (c), not encoding — and (c) says the two are separable, by age.

Usage:
    python -m src.eda.preseason_value
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.eda.availability import with_lags
from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.availability import build_panel as build_availability_panel
from src.features.availability import season_availability
from src.models.availability import (
    FEATURE_COLS,
    TEST_SEASONS,
    build_design,
    season_start_dates,
)
from src.models.availability import split_seasons
from src.models.component_rates import (
    CONVERSION_HEADS,
    COUNT_HEADS,
    _r2,
    conversion_variants,
    count_variants,
    fit_conversion_head,
    fit_count_head,
)
from src.models.component_rates import build_design as build_rate_design
from src.models.held_out import as_plain, selection_split

# ── Constants ─────────────────────────────────────────────────────────────────

#: Clip for the logit link. Matches `availability.EPS`; a preseason share is genuinely 0
#: for a player who sat every late game, which is the signal rather than a missing value.
EPS = 1e-3

#: Trailing TRAINING seasons scored rather than fitted, so the reported ΔR² is out of
#: sample. Two to match the project's own split arity — never the validation seasons, which
#: this module does not materialize.
INNER_SCORE_SEASONS = 2

#: Permutations of the block within season, for the ΔR² noise floor. Twenty rather than a
#: handful because `null_sd` is the denominator of `z_vs_null` and the whole gate runs in
#: seconds — the sampler-cost argument that keeps other ladders short does not apply here.
N_SHUFFLES = 20
SEED = 42

#: The bar, stated before the run: the range by which the best fitted count head beats its
#: no-fit floor on validation R² of the season total. A preseason increment below the bottom
#: of this does not earn an arm, and the null is recorded instead.
#: Reproduced by `make component-rates` → `component_rate_metrics.csv`.
RATE_BAR_LOW, RATE_BAR_HIGH = 0.0013, 0.0334

#: Ridge penalty for the availability/minutes probes. `availability.MINUTES_RIDGE_ALPHA`,
#: reused so the two probes are on the same scale as the shipped minutes probe.
RIDGE_ALPHA = 10.0

#: Age cells for the missingness census. Coarse on purpose — the question is whether the
#: hole is concentrated in the veterans a coach rests, not the shape of the age curve.
AGE_EDGES = [0.0, 24.0, 28.0, 32.0, 99.0]
AGE_LABELS = ["<24", "24-27", "28-31", "32+"]

#: Prior-availability cells. A player who already missed a third of last season is the one
#: whose preseason absence is most likely to be the same injury.
GP_SHARE_EDGES = [-0.01, 0.50, 0.75, 0.90, 1.01]
GP_SHARE_LABELS = ["<0.50", "0.50-0.75", "0.75-0.90", "0.90+"]

OUT_COLS = ["measurement", "family", "key", "target", "metric", "value", "n"]


# ── Link scales ───────────────────────────────────────────────────────────────

def logit(x, eps: float = EPS) -> np.ndarray:
    """Logit of a share, clipped off both boundaries.

    Clipping rather than masking: a zero share is a real observation here (he played none
    of his team's late preseason games), so dropping it would delete exactly the rows the
    participation signal lives on.
    """
    p = np.clip(np.asarray(x, dtype=float), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def log_rate(x) -> np.ndarray:
    """`log1p` of a non-negative rate — the scale `component_rates.add_log` fits on."""
    return np.log1p(np.clip(np.asarray(x, dtype=float), 0.0, None))


# ── The prior-season equivalent of a within-team minutes share ────────────────

#: What `team_minutes_shares` writes, un-lagged.
TEAM_SHARE_COLS = ["min_share", "min_rank"]

#: The lag-1 twins a delta is taken against.
TEAM_SHARE_LAG_COLS = [f"{c}_lag1" for c in TEAM_SHARE_COLS]


def team_minutes_shares(panel: pd.DataFrame) -> pd.DataFrame:
    """Within-team minutes share and rank per (season, player), over his **last** team.

    The prior-season quantity `min_share_pre` is a delta against, and it has to be built
    here because no existing design carries it: `FEATURE_COLS` holds `total_minutes_lag1`
    and `minutes_per_game_lag1`, which are levels. A share is what survives the preseason's
    minutes compression, so it is the only unit on which the two sides are comparable at
    all.

    Deliberately the same construction as `preseason._shares` — share of the team's summed
    minutes, rank descending within team, attributed to the player's last team — so the
    difference between the two sides is the player's role and not two definitions of share.

    Built from the availability panel rather than `component_targets.parquet`, for the
    reason `game_length` gives: `preprocess.clean` drops players below `data.min_games`, and
    a team sum over a filtered frame silently understates the denominator and inflates every
    share on that roster.
    """
    played = panel[panel["played"] == 1]
    by_pt = (played.groupby(["season", "team_id", "player_id"], as_index=False)
             .agg(min_total=("min", "sum")))
    team_total = by_pt.groupby(["season", "team_id"])["min_total"].transform("sum")
    by_pt["min_share"] = by_pt["min_total"] / team_total.replace(0, np.nan)
    by_pt["min_rank"] = (by_pt.groupby(["season", "team_id"])["min_share"]
                         .rank(ascending=False, method="min").astype(float))

    last_team = (played.sort_values(["game_date", "game_id"])
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(team_id=("team_id", "last")))
    out = last_team.merge(by_pt, on=["season", "team_id", "player_id"], how="left")
    return out[["season", "player_id"] + TEAM_SHARE_COLS]


def attach_prior_shares(design: pd.DataFrame, panel: pd.DataFrame,
                        seasons: list[str]) -> pd.DataFrame:
    """`min_share_lag1` / `min_rank_lag1` merged onto a design matrix.

    Season S-1 quantities like every other lag column, so the design stays point-in-time.
    """
    lagged = with_lags(team_minutes_shares(panel), seasons, TEAM_SHARE_COLS, max_lag=1)
    keep = ["season", "player_id"] + TEAM_SHARE_LAG_COLS
    return design.merge(lagged[keep], on=["season", "player_id"], how="left")


# ── The difference-coded blocks ───────────────────────────────────────────────

#: The availability / minutes block. Two deltas on the link scale, three participation
#: levels that have no prior-season equivalent, one reliability term, one indicator.
AVAIL_BLOCK = ["pre_d_min_share_late", "pre_d_mpg", "pre_gp_share",
               "pre_missed_tail_share", "pre_played_final_game",
               "pre_log_min", "has_preseason"]

#: What a rate head adds: its own delta, the reliability term, the indicator. Three columns,
#: because the increment has to be attributable to *that head's* preseason rate.
RATE_BLOCK_SHARED = ["pre_log_min", "has_preseason"]

#: Panel columns every block needs, before any delta is taken. The raw made/attempted
#: totals are here because a conversion head's preseason percentage is not a panel column —
#: `fg2m` and `fg2a` are not even panel columns, for the same reason they are not heads.
PANEL_COLS = (["season", "player_id", "min_pre", "mpg_pre", "gp_share_pre",
               "min_share_pre", "min_share_pre_late", "min_rank_pre",
               "missed_tail", "played_final_game", "team_pre_games",
               "pre_fg3a_share", "fga_pre", "fgm_pre", "fg3a_pre", "fg3m_pre",
               "fta_pre", "ftm_pre"]
              + [f"pre_per36_{c}" for c in COUNT_HEADS])

#: `(made total, attempted total)` in the panel's own names, for each conversion head.
#: `fg2m_pre` / `fg2a_pre` are derived exactly as `component_rates.DERIVED_COUNTS` derives
#: them — `fgm - fg3m` and `fga - fg3a`.
CONVERSION_PRE_SOURCES = {"fg3a": ("fg3a_pre", "fga_pre"),
                          "fg2m": ("fg2m_pre", "fg2a_pre"),
                          "fg3m": ("fg3m_pre", "fg3a_pre"),
                          "ftm": ("ftm_pre", "fta_pre")}


def _reliability(frame: pd.DataFrame) -> pd.DataFrame:
    """`pre_log_min` and `has_preseason` — the two columns every block carries.

    Volume matters: a delta over 60 preseason minutes is noisier than one over 140, and
    `log1p(min_pre)` is the cheapest form of the reliability weight the plan leaves open
    between empirical-Bayes shrinkage and an interaction.
    """
    out = frame.copy()
    out["has_preseason"] = out["min_pre"].notna().astype(float)
    out["pre_log_min"] = log_rate(out["min_pre"].fillna(0.0))
    return out


def attach_availability_block(design: pd.DataFrame,
                              panel: pd.DataFrame) -> pd.DataFrame:
    """`AVAIL_BLOCK` on a design matrix carrying `TEAM_SHARE_LAG_COLS`.

    `pre_missed_tail_share` is a share of the team's own preseason rather than the raw
    count, because `team_pre_games` runs from 2 (the 2011-12 lockout) to 8 and a count
    would make a lockout season's ceiling the same number as an ordinary season's floor.
    """
    out = _reliability(design.merge(panel[PANEL_COLS], on=["season", "player_id"],
                                    how="left"))

    out["pre_d_min_share_late"] = (logit(out["min_share_pre_late"])
                                   - logit(out["min_share_lag1"]))
    out["pre_d_mpg"] = log_rate(out["mpg_pre"]) - log_rate(out["minutes_per_game_lag1"])
    out["pre_gp_share"] = out["gp_share_pre"]
    out["pre_missed_tail_share"] = (out["missed_tail"]
                                    / out["team_pre_games"].replace(0, np.nan))
    out["pre_played_final_game"] = out["played_final_game"]

    # See the module docstring: with `has_preseason` in the block the constant is arbitrary,
    # and zero is the one the nesting argument wants.
    out[AVAIL_BLOCK] = out[AVAIL_BLOCK].fillna(0.0)
    return out


def rate_block(component: str) -> list[str]:
    return [f"pre_d_{component}"] + RATE_BLOCK_SHARED


def attach_rate_block(design: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """One delta per rate head, on that head's own link scale.

    The count heads take a `log1p` per-36 delta and the conversions a logit delta on the
    percentage, matching what each head's link actually is — a conversion head is a
    beta-binomial on made-out-of-attempted, so its preseason twin has to be a percentage
    and not a rate.
    """
    out = _reliability(design.merge(panel[PANEL_COLS], on=["season", "player_id"],
                                    how="left"))
    out["fg2m_pre"] = out["fgm_pre"] - out["fg3m_pre"]
    out["fg2a_pre"] = out["fga_pre"] - out["fg3a_pre"]

    for c in COUNT_HEADS:
        out[f"pre_d_{c}"] = log_rate(out[f"pre_per36_{c}"]) - log_rate(out[f"{c}_p36_lag1"])
    for made, _ in CONVERSION_HEADS:
        num, den = CONVERSION_PRE_SOURCES[made]
        share = out[num] / out[den].replace(0, np.nan)
        out[f"pre_d_{made}"] = logit(share) - logit(out[f"{made}_pct_lag1"])

    cols = ([f"pre_d_{c}" for c in COUNT_HEADS]
            + [f"pre_d_{m}" for m, _ in CONVERSION_HEADS] + RATE_BLOCK_SHARED)
    out[cols] = out[cols].fillna(0.0)
    return out


# ── Scope ─────────────────────────────────────────────────────────────────────

def covered_seasons(coverage: pd.DataFrame) -> list[str]:
    """Seasons whose preseason is present **and** whose tail is intact.

    `tail_missing` (2003-04 alone) is excluded rather than down-weighted: the late-weighted
    shares and `missed_tail` are read over exactly the games that season does not hold, so
    its block is not a noisy version of the feature — it is a different one.
    """
    have = coverage[(coverage["rows_kept"] > 0)
                    & (coverage["coverage_class"] == "covered")]
    return sorted(have["season"].astype(str))


def training_frames(design: pd.DataFrame, seasons_in_scope: list[str],
                    test_seasons: int = TEST_SEASONS,
                    inner: int = INNER_SCORE_SEASONS
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(fit, score)` — both inside the TRAINING half, both restricted to covered seasons.

    `selection_split` is called for its left-hand frame only. Validation is never read here:
    P1 decides which heads get built, and a screen that spends the selection split leaves
    P2 nothing to select on.

    The inner carve nests `split_seasons` inside its own training frame, which is the case
    `held_out.as_plain` exists for — the right-hand side of the inner split is a training
    season, not the held-out one, and has to be readable.
    """
    train, _ = selection_split(design, test_seasons)
    scoped = train[train["season"].isin(seasons_in_scope)]
    n_seasons = scoped["season"].nunique()
    if n_seasons <= inner:
        raise ValueError(f"need more than {inner} covered training seasons; got {n_seasons}")
    fit, score = split_seasons(scoped, inner)
    return fit.reset_index(drop=True), as_plain(score).reset_index(drop=True)


# ── Regression helpers ────────────────────────────────────────────────────────

def _matrix(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return np.nan_to_num(frame[cols].to_numpy(dtype=float), nan=0.0, posinf=0.0,
                         neginf=0.0)


def ridge_r2(fit: pd.DataFrame, score: pd.DataFrame, features: list[str],
             target: str) -> float:
    """Out-of-sample R² of a standardized ridge — the availability/minutes probe.

    A probe, not a head: the number says whether the block moves the target, not how good
    a minutes or availability model can be. `stan_availability` is where a head is scored.
    """
    scaler = StandardScaler().fit(_matrix(fit, features))
    model = Ridge(alpha=RIDGE_ALPHA).fit(scaler.transform(_matrix(fit, features)),
                                         fit[target].to_numpy(dtype=float))
    pred = model.predict(scaler.transform(_matrix(score, features)))
    y = score[target].to_numpy(dtype=float)
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def shuffle_within_season(frame: pd.DataFrame, cols: list[str],
                          rng: np.random.Generator) -> pd.DataFrame:
    """Permute the block across rows, within season.

    Within season rather than pooled, so the null keeps the block's era composition —
    otherwise a shuffled 2011-12 lockout row could land on a 2019-20 player and the null
    would be measuring the calendar.
    """
    out = frame.reset_index(drop=True).copy()
    block = np.array(out[cols].to_numpy(dtype=float), copy=True)
    for _, pos in out.groupby("season", sort=False).indices.items():
        block[pos] = block[rng.permutation(pos)]
    out[cols] = block
    return out


def residualize(y: np.ndarray, controls: np.ndarray) -> np.ndarray:
    design = np.column_stack([np.ones(len(y)), controls])
    beta, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ beta


def partial_corr(x: np.ndarray, y: np.ndarray, controls: np.ndarray) -> float:
    rx, ry = residualize(x, controls), residualize(y, controls)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


# ── (a) Redundancy ────────────────────────────────────────────────────────────

#: `(family, preseason column, prior-season column, link)`. The pairs whose correlation
#: answers "is the preseason quantity just a shorter sample of the prior season".
REDUNDANCY_PAIRS = (
    [("availability", "min_share_pre", "min_share_lag1", "logit"),
     ("availability", "min_share_pre_late", "min_share_lag1", "logit"),
     ("availability", "min_rank_pre", "min_rank_lag1", "raw"),
     ("availability", "mpg_pre", "minutes_per_game_lag1", "log"),
     ("availability", "gp_share_pre", "gp_share_lag1", "raw")]
    + [("rates", f"pre_per36_{c}", f"{c}_p36_lag1", "log") for c in COUNT_HEADS]
    + [("rates", "pre_fg3a_share", "fg3a_pct_lag1", "logit")]
)

_LINKS = {"logit": logit, "log": log_rate, "raw": lambda x: np.asarray(x, dtype=float)}


def redundancy(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Correlation of each preseason quantity with its prior-season equivalent.

    Reported on the **link scale the delta is taken on**, because that is the scale the
    coefficient would live on: a Pearson r on raw per-36 rates is dominated by the
    right tail of `fga` and says nothing about what a log-scale delta carries.

    `sd_delta` rides along and is the number that actually matters. A high correlation
    with a wide residual spread is not redundancy — it is a strong common component plus
    real new information, which is exactly what difference coding isolates.
    """
    rows = []
    for family, pre_col, prior_col, link in REDUNDANCY_PAIRS:
        frame = frames[family]
        if pre_col not in frame or prior_col not in frame:
            continue
        f = _LINKS[link]
        sub = frame[[pre_col, prior_col]].dropna()
        if len(sub) < 2:
            continue
        x, y = f(sub[pre_col]), f(sub[prior_col])
        delta = x - y
        for metric, value in (("pearson_r", float(np.corrcoef(x, y)[0, 1])),
                              ("spearman_r", float(pd.Series(x).corr(pd.Series(y),
                                                                     method="spearman"))),
                              ("mean_delta", float(delta.mean())),
                              ("sd_delta", float(delta.std(ddof=1))),
                              ("sd_prior", float(y.std(ddof=1)))):
            rows.append({"measurement": "redundancy", "family": family, "key": pre_col,
                         "target": f"{prior_col} [{link}]", "metric": metric,
                         "value": value, "n": len(sub)})
    return rows


# ── (b) Incremental signal ────────────────────────────────────────────────────

def _increment_rows(family: str, key: str, target: str, base: float, with_block: float,
                    in_sample: float, nulls: list[float], n: int,
                    metric: str) -> list[dict]:
    null = np.asarray(nulls, dtype=float)
    delta = with_block - base
    sd = float(null.std(ddof=1)) if len(null) > 1 else np.nan
    values = {
        f"r2_base_{metric}": base,
        f"r2_with_block_{metric}": with_block,
        "delta_r2": delta,
        "delta_r2_in_sample": in_sample,
        "delta_r2_null_mean": float(null.mean()) if len(null) else np.nan,
        "delta_r2_null_sd": sd,
        "z_vs_null": (delta - float(null.mean())) / sd if sd and sd > 0 else np.nan,
    }
    return [{"measurement": "incremental", "family": family, "key": key,
             "target": target, "metric": m, "value": v, "n": n}
            for m, v in values.items()]


def _attribution(family: str, key: str, target: str, base: float, n: int,
                 delta_only: float, shared_only: float) -> list[dict]:
    """Split a head's increment into its own preseason rate and the shared two columns."""
    return [{"measurement": "incremental", "family": family, "key": key,
             "target": target, "metric": m, "value": v, "n": n}
            for m, v in (("delta_r2_delta_only", delta_only - base),
                         ("delta_r2_shared_only", shared_only - base))]


def availability_increment(fit: pd.DataFrame, score: pd.DataFrame,
                           block: list[str] = None, n_shuffles: int = N_SHUFFLES,
                           seed: int = SEED, family: str = "availability") -> list[dict]:
    """What the preseason block adds to `gp_share` and `minutes_per_game`.

    The base is `availability.FEATURE_COLS` — the shipped head's own prior-season block —
    so the contrast is a nested increment rather than a different model. Both targets are
    scored, because the plan splits the head that way: the availability arm is about
    games played and the minutes arm is about `min | available`, and the preseason is a
    priori much louder about the second.

    **Run twice, on two populations, and the gap between them is a finding rather than a
    robustness check.** The design frame holds every player who appeared in season S,
    including the ones who signed *after* the preseason — a January 10-day contract has no
    preseason row and a tiny `gp_share` for a reason that has nothing to do with health.
    That is not leakage (at the draft we do know he is on no roster), but it is a
    population the head never predicts on in production, where the draft pool is the
    season-start roster. `family="availability_draftable"` is the number that describes
    what the block would buy where it would actually be used.
    """
    block = block or AVAIL_BLOCK
    base_cols = [c for c in FEATURE_COLS if c in fit.columns]
    rng = np.random.default_rng(seed)
    rows = []
    for target in ("gp_share", "minutes_per_game"):
        base = ridge_r2(fit, score, base_cols, target)
        with_block = ridge_r2(fit, score, base_cols + block, target)
        in_sample = (ridge_r2(fit, fit, base_cols + block, target)
                     - ridge_r2(fit, fit, base_cols, target))
        nulls = []
        for _ in range(n_shuffles):
            f_s = shuffle_within_season(fit, block, rng)
            s_s = shuffle_within_season(score, block, rng)
            nulls.append(ridge_r2(f_s, s_s, base_cols + block, target) - base)
        rows += _increment_rows(family, "block", target, base, with_block,
                                in_sample, nulls, len(score), "ridge")

        # Attribution: one column at a time, against the same base.
        controls = _matrix(fit, base_cols)
        y = fit[target].to_numpy(dtype=float)
        for col in block:
            rows.append({"measurement": "incremental", "family": family,
                         "key": col, "target": target, "metric": "partial_r",
                         "value": partial_corr(_matrix(fit, [col]).ravel(), y, controls),
                         "n": len(fit)})
            rows.append({"measurement": "incremental", "family": family,
                         "key": col, "target": target, "metric": "delta_r2_alone",
                         "value": ridge_r2(fit, score, base_cols + [col], target) - base,
                         "n": len(score)})
    return rows


def rate_increment(fit: pd.DataFrame, score: pd.DataFrame,
                   n_shuffles: int = N_SHUFFLES, seed: int = SEED) -> list[dict]:
    """What each rate head's own preseason delta adds, **on that head's own metric**.

    Counts are scored by R² on the season total and conversions by R² on the realized
    percentage, exactly as `component_rates.evaluate` does, because the bar this is read
    against is a column of that artifact. The base variant is `log_own` for counts and
    `linear` for conversions — the plain fitted arm, not each head's selected one: what is
    being priced is the *increment*, and a nested increment on top of the shipped variant
    is P-2-and-later work that costs sampler time this gate exists to avoid spending.
    """
    rng = np.random.default_rng(seed)
    rows = []

    for component in COUNT_HEADS:
        block = rate_block(component)
        y = score[component].to_numpy(dtype=float)
        y_fit = fit[component].to_numpy(dtype=float)
        tr, te, feats = count_variants(fit, score, component)["log_own"]
        base = _r2(y, fit_count_head(tr, te, feats, component)[0])
        with_block = _r2(y, fit_count_head(tr, te, feats + block, component)[0])
        in_sample = (_r2(y_fit, fit_count_head(tr, tr, feats + block, component)[0])
                     - _r2(y_fit, fit_count_head(tr, tr, feats, component)[0]))
        nulls = []
        for _ in range(n_shuffles):
            f_s = shuffle_within_season(tr, block, rng)
            s_s = shuffle_within_season(te, block, rng)
            nulls.append(_r2(s_s[component].to_numpy(dtype=float),
                             fit_count_head(f_s, s_s, feats + block, component)[0]) - base)
        rows += _increment_rows("rates_count", component, f"{component} season total",
                                base, with_block, in_sample, nulls, len(score), "total")
        # Attribution, and the question the gate actually asks: the head earns an arm on
        # its own preseason RATE, not on the fact that a preseason row exists. The two
        # halves of the block are scored separately so a gain that is really
        # `has_preseason` cannot be recorded as a rate finding.
        rows += _attribution("rates_count", component, f"{component} season total",
                             base, len(score),
                             delta_only=_r2(y, fit_count_head(
                                 tr, te, feats + [f"pre_d_{component}"], component)[0]),
                             shared_only=_r2(y, fit_count_head(
                                 tr, te, feats + RATE_BLOCK_SHARED, component)[0]))

    for made, attempted in CONVERSION_HEADS:
        block = rate_block(made)
        n_score = score[attempted].to_numpy(dtype=float)
        tr, te, feats = conversion_variants(fit, score, made, attempted)["linear"]

        def _score(f, t, features):
            mask, p, _ = fit_conversion_head(f, t, features, made, attempted)
            n_t = t[attempted].to_numpy(dtype=float)[mask]
            return _r2(t[made].to_numpy(dtype=float)[mask] / n_t, p)

        base = _score(tr, te, feats)
        with_block = _score(tr, te, feats + block)
        in_sample = _score(tr, tr, feats + block) - _score(tr, tr, feats)
        nulls = [_score(shuffle_within_season(tr, block, rng),
                        shuffle_within_season(te, block, rng), feats + block) - base
                 for _ in range(n_shuffles)]
        rows += _increment_rows("rates_conversion", f"{made}|{attempted}",
                                f"{made}/{attempted} realized pct", base, with_block,
                                in_sample, nulls, int((n_score > 0).sum()), "pct")
        rows += _attribution("rates_conversion", f"{made}|{attempted}",
                             f"{made}/{attempted} realized pct", base,
                             int((n_score > 0).sum()),
                             delta_only=_score(tr, te, feats + [f"pre_d_{made}"]),
                             shared_only=_score(tr, te, feats + RATE_BLOCK_SHARED))
    return rows


# ── (c) The missingness census ────────────────────────────────────────────────

def _cells(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """The cuts the census is read on, plus the pooled row.

    `season_start_roster` is a cut and not a filter here on purpose: the two sides answer
    different questions. Off-roster rows are players who signed *during* season S, whose
    missing preseason is a contract fact rather than a health one, and separating them is
    what stops the health reading from being credited with a roster reading.
    """
    return {
        "all": pd.Series("all", index=frame.index),
        "role": pd.cut(frame["minutes_per_game_lag1"], ROLE_EDGES,
                       labels=ROLE_LABELS).astype(str),
        "age": pd.cut(frame["age"], AGE_EDGES, labels=AGE_LABELS).astype(str),
        "prior_gp_share": pd.cut(frame["gp_share_lag1"], GP_SHARE_EDGES,
                                 labels=GP_SHARE_LABELS).astype(str),
        "season_start_roster": np.where(frame["on_season_start_roster"] > 0,
                                        "on", "off"),
    }


#: The cuts read on the draftable population. `season_start_roster` is not among them
#: because it is constant there — it is the cut that *defines* it.
CENSUS_DIMENSIONS = ("all", "role", "age", "prior_gp_share")


def missingness_census(frame: pd.DataFrame, family: str = "availability",
                       dimensions: tuple[str, ...] = CENSUS_DIMENSIONS) -> list[dict]:
    """Who has no preseason row, and whether the hole predicts the season.

    The share missing is the easy half. The half that decides whether `has_preseason` needs
    **splitting** is the outcome gap *within a cell*: a rested veteran and an injured star
    are both missing, and only one of them is about to lose the season. If the gap is flat
    across role and age, one indicator is enough; if it concentrates, the indicator is
    hiding two populations and P2 has to separate them.

    Read the role and age cuts on the **draftable** frame only. Pooled over everyone who
    appeared in season S they are unreadable: half the missing rows are mid-season signings
    whose absence is a contract fact, and that mixes a roster story into every cell of a
    table meant to isolate a health one.
    """
    rows = []
    present = frame["has_preseason"] > 0
    cells = {k: v for k, v in _cells(frame).items() if k in dimensions}
    for dimension, cell in cells.items():
        for key, idx in frame.groupby(np.asarray(cell), sort=True).indices.items():
            sub = frame.iloc[idx]
            here = present.iloc[idx]
            with_pre, without = sub[here.values], sub[~here.values]
            vals = {"n": float(len(sub)), "share_missing": float((~here).mean())}
            for target in ("gp_share", "minutes_per_game"):
                a = float(with_pre[target].mean()) if len(with_pre) else np.nan
                b = float(without[target].mean()) if len(without) else np.nan
                vals[f"mean_{target}_with_preseason"] = a
                vals[f"mean_{target}_no_preseason"] = b
                vals[f"gap_{target}"] = b - a
            rows += [{"measurement": "missingness", "family": family,
                      "key": str(key), "target": dimension, "metric": m, "value": v,
                      "n": len(sub)} for m, v in vals.items()]
    return rows


def attach_season_start_roster(design: pd.DataFrame, seasons: list[str],
                               raw_dir: str | Path, window_games: int) -> pd.DataFrame:
    """`on_season_start_roster` — was the player on an opening-roster before the season?

    The draft pool, and therefore the only population the availability head is ever applied
    to in production. `season_start_rosters` reads a player's team in his earliest games of
    season S, which `docs/project-spec.md` establishes is knowable before the season and so
    is not leakage in a backtest.
    """
    from src.features.team_context import season_start_rosters

    rosters = season_start_rosters(seasons, raw_dir, window_games)
    keys = set(zip(rosters["season"], rosters["player_id"]))
    out = design.copy()
    out["on_season_start_roster"] = [
        float((s, p) in keys) for s, p in zip(out["season"], out["player_id"])]
    return out


def no_prior_reach(panel: pd.DataFrame, design: pd.DataFrame,
                   seasons_in_scope: list[str]) -> list[dict]:
    """How many preseason rows belong to players the availability design cannot reach.

    P4's population, counted here because P1 is where its size stops being an assertion.
    A player with no usable prior season has no design row at all, so he is invisible to
    measurement (b) — and he is precisely the player for whom a preseason game is the
    *first* NBA observation rather than a marginal update.
    """
    pre = panel[panel["season"].isin(seasons_in_scope)]
    keys = set(zip(design["season"], design["player_id"]))
    covered = [(s, p) in keys for s, p in zip(pre["season"], pre["player_id"])]
    n_out = int(len(pre) - sum(covered))
    return [{"measurement": "missingness", "family": "no_prior", "key": "panel_rows",
             "target": "reach", "metric": m, "value": v, "n": len(pre)}
            for m, v in (("panel_rows", float(len(pre))),
                         ("rows_with_design_row", float(sum(covered))),
                         ("rows_without_design_row", float(n_out)),
                         ("share_without_design_row",
                          float(n_out / len(pre)) if len(pre) else np.nan))]


# ── Entry point ───────────────────────────────────────────────────────────────

def _pivot(rows: list[dict], measurement: str, family: str, index: str,
           columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    sub = frame[(frame["measurement"] == measurement) & (frame["family"] == family)]
    piv = sub.pivot_table(index=index, columns="metric", values="value")
    return piv[[c for c in columns if c in piv.columns]]


def run(cfg: dict) -> Path:
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", TEST_SEASONS))
    p_cfg = cfg.get("eda", {}).get("preseason_value", {})
    n_shuffles = int(p_cfg.get("n_shuffles", N_SHUFFLES))
    seed = int(p_cfg.get("seed", SEED))

    panel = pd.read_parquet(features_dir / "preseason.parquet")
    coverage = pd.read_csv(out_dir / "preseason_coverage.csv")
    scope = covered_seasons(coverage)

    print(f"Preseason EDA gate (P1): {len(panel):,} panel rows over "
          f"{panel['season'].nunique()} seasons")
    print(f"  in scope: {len(scope)} seasons with an intact tail "
          f"({scope[0]} → {scope[-1]}); "
          f"{int((coverage['coverage_class'] == 'tail_missing').sum())} excluded as "
          f"tail_missing, {int((coverage['rows_kept'] == 0).sum())} as absent")
    print("  TRAIN SEASONS ONLY. Validation is never materialized here — P1 decides which "
          "heads\n  get built, and a screen that spends the selection split leaves P2 "
          "nothing to select on.")

    # ── the availability / minutes family ────────────────────────────────────
    window_games = cfg.get("features", {}).get("team_context", {}).get(
        "roster_window_games", 10)
    av_panel = build_availability_panel(seasons, raw_dir)
    frame = season_availability(av_panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(av_panel))
    design = attach_prior_shares(design, av_panel, seasons)
    design = attach_availability_block(design, panel)
    design = attach_season_start_roster(design, seasons, raw_dir, window_games)

    av_fit, av_score = training_frames(design, scope, test_seasons, INNER_SCORE_SEASONS)
    print(f"\nAvailability design: {len(design):,} player-seasons, "
          f"{len(av_fit):,} fit / {len(av_score):,} score "
          f"({', '.join(sorted(av_score['season'].unique()))} as the inner score split)")
    print(f"  {av_fit['has_preseason'].mean():.1%} of fit rows have a preseason row")
    # A player with a preseason row but no prior-season *share* played zero games in S-1,
    # so the delta has no left-hand side. Reported rather than filled silently: the fill
    # is arbitrary only for rows the indicator separates, and these are not those rows.
    orphan = int(((av_fit["has_preseason"] > 0)
                  & av_fit["min_share_lag1"].isna()).sum())
    print(f"  {orphan:,} fit rows have a preseason row but no prior-season minutes share "
          "(zero games\n  played in S-1, so the share delta has no left-hand side and its "
          "zero is not separated\n  by the indicator)")

    av_all = pd.concat([av_fit, av_score], ignore_index=True)
    rows = availability_increment(av_fit, av_score, n_shuffles=n_shuffles, seed=seed)
    rows += missingness_census(av_all, "availability", ("all", "season_start_roster"))
    rows += no_prior_reach(panel, design, scope)

    # The same increment on the population the head is actually applied to. A player who
    # signed in January has no preseason row and a tiny `gp_share` for a contract reason,
    # and crediting the health signal with that is the one way this measurement could
    # flatter itself without leaking anything.
    dr_fit = av_fit[av_fit["on_season_start_roster"] > 0].reset_index(drop=True)
    dr_score = av_score[av_score["on_season_start_roster"] > 0].reset_index(drop=True)
    print(f"  draftable subset (on a season-start roster): {len(dr_fit):,} fit / "
          f"{len(dr_score):,} score, {dr_fit['has_preseason'].mean():.1%} with a "
          f"preseason row")
    rows += availability_increment(dr_fit, dr_score, n_shuffles=n_shuffles, seed=seed,
                                   family="availability_draftable")
    rows += missingness_census(av_all[av_all["on_season_start_roster"] > 0]
                               .reset_index(drop=True), "availability_draftable")

    # ── the rate family ──────────────────────────────────────────────────────
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    rate_design = attach_season_start_roster(
        attach_rate_block(build_rate_design(targets, seasons, raw_dir), panel),
        seasons, raw_dir, window_games)
    rt_fit, rt_score = training_frames(rate_design, scope, test_seasons,
                                       INNER_SCORE_SEASONS)
    print(f"Rate design: {len(rate_design):,} player-seasons, {len(rt_fit):,} fit / "
          f"{len(rt_score):,} score")
    print(f"  {rt_fit['on_season_start_roster'].mean():.1%} of rate-design fit rows are "
          f"on a season-start roster — the >= 200 prior-minute filter already removes "
          f"most of\n  the mid-season-signing population, which is why the rate family is "
          f"measured on one scope")
    rows += [{"measurement": "incremental", "family": "rates_population",
              "key": "rate_design", "target": "season_start_roster",
              "metric": "share_on_roster",
              "value": float(rt_fit["on_season_start_roster"].mean()), "n": len(rt_fit)}]
    rows += rate_increment(rt_fit, rt_score, n_shuffles=n_shuffles, seed=seed)
    rows += redundancy({"availability": av_fit, "rates": rt_fit})

    table = pd.DataFrame(rows)[OUT_COLS]

    # ── readout ──────────────────────────────────────────────────────────────
    print("\n── (a) redundancy: is the preseason quantity a shorter sample of the prior "
          "season? ──")
    for family in ("availability", "rates"):
        piv = _pivot(rows, "redundancy", family, "key",
                     ["pearson_r", "spearman_r", "mean_delta", "sd_delta", "sd_prior"])
        print(f"\n  {family}:")
        print(piv.round(4).to_string())
    print("\n  A high r with a wide `sd_delta` is not redundancy — it is a strong common "
          "component\n  plus real new information, which is what the difference coding "
          "isolates.")

    print("\n── (b) incremental signal, out of sample on held-back TRAINING seasons ──")
    for family, label in (
            ("availability", "every player who appeared in season S"),
            ("availability_draftable", "ON A SEASON-START ROSTER — the production "
                                       "population, and the number that counts")):
        piv = _pivot(rows, "incremental", family, "target",
                     ["r2_base_ridge", "r2_with_block_ridge", "delta_r2",
                      "delta_r2_in_sample", "delta_r2_null_mean", "delta_r2_null_sd",
                      "z_vs_null"])
        print(f"\n  availability / minutes, {label} "
              f"(ridge probe, block = {len(AVAIL_BLOCK)} columns):")
        print(piv.round(4).to_string())
    print("\n  The gap between the two is a population effect, not leakage: a player who "
          "signed in\n  January has no preseason row and a tiny gp_share for a contract "
          "reason. Both facts are\n  knowable at the draft; only the second scope is the "
          "population the head is applied to.")

    attribution = pd.DataFrame(rows)
    attribution = attribution[(attribution["measurement"] == "incremental")
                              & (attribution["family"] == "availability_draftable")
                              & (attribution["metric"].isin(["partial_r",
                                                             "delta_r2_alone"]))]
    print("\n  per column on the draftable population (partial r against the same base, "
          "and its ΔR² alone):")
    print(attribution.pivot_table(index="key", columns=["target", "metric"],
                                  values="value").round(4).to_string())

    for family, label in (("rates_count", "count heads — R² on the SEASON TOTAL, the "
                                          "bar's own unit"),
                          ("rates_conversion", "conversion heads — R² on the realized "
                                               "percentage")):
        piv = _pivot(rows, "incremental", family, "key",
                     ["r2_base_total", "r2_base_pct", "delta_r2", "delta_r2_delta_only",
                      "delta_r2_shared_only", "delta_r2_in_sample",
                      "delta_r2_null_mean", "delta_r2_null_sd", "z_vs_null"])
        print(f"\n  {label}:")
        print(piv.round(4).to_string())

    counts = _pivot(rows, "incremental", "rates_count", "key", ["delta_r2"])
    clears = counts[counts["delta_r2"] >= RATE_BAR_LOW]
    print(f"\n  THE RATES BAR, stated before the run: ΔR² ≥ {RATE_BAR_LOW:+.4f} — the "
          f"margin by which\n  the weakest shipped count head (`reb`) beats its no-fit "
          f"floor. Range to {RATE_BAR_HIGH:+.4f}.")
    print(f"  {len(clears)} of {len(counts)} count heads clear it"
          + (f": {', '.join(clears.index)}" if len(clears) else " — recorded as a null."))

    print("\n── (c) the missingness census ──")
    census_cols = ["n", "share_missing", "mean_gp_share_with_preseason",
                   "mean_gp_share_no_preseason", "gap_gp_share",
                   "mean_minutes_per_game_with_preseason",
                   "mean_minutes_per_game_no_preseason", "gap_minutes_per_game"]
    for family, dimensions in (("availability", ("all", "season_start_roster")),
                               ("availability_draftable", CENSUS_DIMENSIONS)):
        for dimension in dimensions:
            sub = table[(table["measurement"] == "missingness")
                        & (table["family"] == family)
                        & (table["target"] == dimension)]
            piv = sub.pivot_table(index="key", columns="metric", values="value")
            scope_label = ("" if family == "availability" else " [draftable only]")
            print(f"\n  by {dimension}{scope_label}:")
            print(piv[[c for c in census_cols if c in piv.columns]].round(3).to_string())
    print("\n  A flat gap across role and age means one `has_preseason` indicator is "
          "enough. A gap\n  that concentrates means the indicator is hiding two "
          "populations — a rested veteran\n  and an injured star — and P2 has to split it. "
          "The role and age cuts are read on the\n  draftable frame alone, because pooled "
          "over everyone half the missing rows are\n  mid-season signings and the cell is "
          "then a contract fact wearing a health costume.")

    reach = table[(table["measurement"] == "missingness")
                  & (table["family"] == "no_prior")].set_index("metric")["value"]
    print(f"\n  P4's population, sized: {int(reach['rows_without_design_row']):,} of "
          f"{int(reach['panel_rows']):,} in-scope panel rows "
          f"({reach['share_without_design_row']:.1%}) belong to players with no usable "
          f"prior\n  season, so measurement (b) cannot see them at all — they are the "
          f"rows for whom a\n  preseason game is the FIRST NBA observation rather than a "
          f"marginal update.")

    dest = out_dir / "preseason_value.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
