"""Fitting-window x season-trend ladder for the availability head, scored on the TAILS.

`make availability-window`. One row per arm in
`outputs/predictions/availability_window.csv`.

## Why this exists

`model_card_ecdf.csv` says the shipped beta-binomial misses both ends of its own
distribution, and misses them in opposite directions. On validation it puts **5.95%** of
player-seasons at a full schedule against an observed **2.72%**, and **5.21%** below ten
games against an observed **8.15%** — both outside the posterior-predictive band, on the
head that `README.md` calls the largest lever on the season total. Neither miss is visible
in CRPS, MAE or PIT KS, which is why the head has cleared every gate it has ever been
scored on while being wrong about the two events a best-ball roster actually turns on: an
iron man and a dead pick.

## What the two arms are for, and why they are not alternatives

A **shorter fitting window** and a **season trend** answer different halves of the same
measurement, so this ladder crosses them rather than choosing between them.

The league moved. On the head's own design rows, mean `gp_share` sits flat near 0.710 for
twenty seasons and then falls to 0.605-0.638; a sup-F scan over every candidate breakpoint
puts the change at **2017-18** (F = 91.8 against a Monte-Carlo null 95th percentile of 9.2)
and BIC prefers a *broken trend* over a level step. So the movement is a slope change, not
a step:

- a **window** is the right instrument for a completed level shift. It throws away the old
  regime and costs training rows, which for `post_break` is severe — 2017-18 through
  2021-22 is five target seasons, because the training half ends at 2021-22 and validation
  may not be fitted on.
- a **trend** is the right instrument for a continuing slope. It costs one column and
  extrapolates by construction, which is exactly what a walk-forward forecast needs and
  exactly what makes it dangerous: `season_terms` warns that a trend fitted on many seasons
  and extrapolated one or two forward can fit two validation seasons by accident.

Crossing them is what separates the two explanations. If the window alone carries it, the
change was a step and the trend is noise; if the trend alone carries it, truncating was
throwing away usable rows; if they only work together, it is a broken trend and both terms
are doing real work.

`trend_x_role` is here because `docs/availability-plan.md` measured the era effect as
**role-graded** rather than a level shift — heavy-minute players lost -0.101 of games-played
share against -0.037 for fringe players — so a single league-wide slope is the wrong shape
if that holds. Role is bucketed on **prior-season** MPG, known before opening night.

## Why the tails are in the metric set

Because the defect is invisible without them, and because the contest is a threshold
machine: a Round-1 knockout is decided by the best 7 of 16 in a week, so a player who plays
four games is a dead roster slot and a player who plays every game is a ceiling the model
either has or does not. `tail_coverage` reports **predicted against observed** at each
threshold rather than a Brier score, because a Brier score is minimised by a model that is
confidently wrong in the same direction on every row and would not have caught this.

The upper threshold is `GP == team_games` per row rather than a fixed 82, so the shortened
seasons contribute their own boundary instead of an unreachable one.

## What this ladder is and is not

It is a **specification ladder on the point MLE**, which is what `availability.py::run`
itself selects on and what `stan_availability` is verified against (21/21 coefficients
inside the 95% credible interval). It runs in seconds and needs no CmdStan, so an arm can
be rejected before anyone spends sampler time — the same reason `games_played.py` is numpy
only. An arm that wins here earns a Stan port; it does not ship from here.

Selection reads **validation and nothing else**. `selection_split` never materializes the
held-out rows, and every window is a restriction of the *training* half only.

## The fourth axis, added 2026-08-11: the likelihood

The three axes above are all instruments on the *mean* or on one dispersion scalar, and §4
settled that neither closes the boundary. `docs/potential-to-dos.md` item 5 says why, and
the mechanism is arithmetic rather than a hypothesis: under `a = mu(1-rho)/rho` and
`b = (1-mu)(1-rho)/rho` the frailty's **shape** and its **variance** are the same parameter,
so `b < 1` — a Beta density that diverges at `p = 1`, sitting exactly on "played every
game" — is forced whenever `rho > (1-mu)/(2-mu)`. That holds on 52.7% of validation rows and
82.2% of the `24-30 mpg` bucket. Moving `rho` is why the shipped arm could only halve the
miss.

So the fourth axis varies the **frailty**, holding the other three at the shipped arm
(`three_point_era`, `none`, `role`). It is not crossed with the full grid: 19 arms was
already the multiplicity problem §4b exists to answer, and crossing would make it 95.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize
from scipy.special import digamma, logsumexp
from scipy.stats import betabinom, binom
from sklearn.preprocessing import StandardScaler

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.availability import build_panel, season_availability
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN, AvailabilityModel,
                                     BetaBinomialGLM, LeagueAgeBaseline, _ab, _sigmoid,
                                     build_design, crps, fit_dispersion, pit_values,
                                     predictive_pmf, season_start_dates)
from src.models.held_out import selection_split
from src.models.season_terms import add_role_terms, add_trend, season_start_year

# First season *start year* kept in the fitting half. `None` is the incumbent's 1996-97.
# `three_point_era` is `docs/potential-to-dos.md`'s measured 3PA breakpoint; `post_break`
# is where the sup-F scan puts the availability change, and it is the expensive one —
# five target seasons, because training stops at 2021-22.
WINDOWS: dict[str, int | None] = {
    "full": None,
    "three_point_era": 2012,
    "post_break": 2017,
}

# The season term, on top of a fixed feature block so the contrast is the term alone.
TREND_SPECS = ("none", "trend", "trend_x_role")

# How the beta-binomial's dispersion is allowed to vary. `shared` is the incumbent's one
# scalar for every player in every season; `role` grades it on **prior-season** MPG, the
# device `stan_composition` already ships (0.1768 fringe against 0.0855 star, a 2.07x
# spread that cut its calibration error by 35%). Two candidate explanations for the
# boundary-mass defect — one dispersion for 25 seasons, or one dispersion for every role —
# and crossing them with the window separates era pooling from population pooling.
RHO_MODES = ("shared", "role")

# Low-tail thresholds. 10 is `docs/potential-to-dos.md` item 4's "very few games"; 41 and
# 60 are Gate D's, carried so this table is readable beside `stan_games_played_gates.csv`.
TAIL_BELOW = (10, 41, 60)

# ── The upper shoulder, added 2026-08-11 ──────────────────────────────────────
#
# The metric set above is **asymmetric**, and the asymmetry hid half the defect. `below_10`
# is a ten-game-wide shoulder; `full_schedule` is a single point mass. So between 60 games
# and a full schedule nothing was measured at all — while `docs/availability-window-plan.md`
# §1 states the defect as "too little in the shoulders at 2-15 and **70-80** games". The
# shoulder the defect names on the high side was never on the table.
#
# Thresholds here are counted in **games missed** rather than games played, because that is
# the only schedule-invariant way to say "played nearly everything": `gp > 70` is a
# different event in a 66-game season than in an 82-game one, where `missed <= 11` is the
# same event in both. 11 and 5 are ">70 of 82" and ">76 of 82".
TAIL_MISSED = (11, 5)

# The exclusive bands the two cumulative families decompose into. A cumulative threshold
# lets errors of OPPOSITE sign inside the same tail cancel, which is exactly what happens
# here: §1 measured the head *over*-predicting P(GP <= 1) while *under*-predicting
# P(GP < 10). That is the U-shape, and `below_10` alone nets it out. Each band is
# (name, lower, upper) on the relevant scale, half-open at the top.
BANDS: tuple[tuple[str, str, int, int], ...] = (
    ("zero", "played", 0, 1),              # GP == 0, the exact lower boundary
    ("low_shoulder", "played", 1, 10),     # 1-9 games: alive, but a dead roster slot
    ("high_shoulder", "missed", 1, 12),    # missed 1-11: nearly an iron man, not one
    ("full", "missed", 0, 1),              # missed == 0, the exact upper boundary
)

# How far into each tail the localized shape distance looks. A band is one number per
# region; this is the largest gap anywhere in the region's calibration curve, which is what
# "the shape of the distribution in the shoulders" actually asks for.
SHAPE_DEPTH = 15

# Central predictive intervals whose realized coverage is reported, matching
# `season_terms.COVERAGE_LEVELS` so the two tables read the same way.
COVERAGE_LEVELS = (0.5, 0.8, 0.95)

BOOTSTRAP_REPS = 2000
REFERENCE_ARM = "full__none__shared"   # the shipped head: whole window, no season term,
                                       # one dispersion for every player in every season


# Fewest training rows a role bucket needs before it gets its own dispersion. Below this
# the bucket falls back to the pooled scalar, because a dispersion fitted on a handful of
# rows is noise wearing the shape of a parameter.
MIN_ROLE_ROWS = 100


class RoleGradedBetaBinomial(BetaBinomialGLM):
    """`BetaBinomialGLM` with one dispersion per prior-MPG role bucket.

    The mean function is fitted exactly as the incumbent's; `rho` is then re-fitted inside
    each bucket holding that mean fixed, which is the same two-stage device
    `AvailabilityModel.fit` already uses to give each model the dispersion its own errors
    warrant — applied one level down, to each role's own errors.

    Buckets come from `season_effects.ROLE_EDGES`, the edges `docs/availability-plan.md`
    measured the role-graded era effect on, so a dispersion split here is comparable to the
    level split measured there. Role is **prior-season** MPG: known before opening night,
    so this is not the target grading itself.
    """

    name = "beta_binomial_role_rho"

    @staticmethod
    def _buckets(df: pd.DataFrame) -> np.ndarray:
        return np.asarray(pd.cut(df["minutes_per_game_lag1"].to_numpy(dtype=float),
                                 ROLE_EDGES, labels=ROLE_LABELS).astype(str))

    def fit(self, train: pd.DataFrame) -> "RoleGradedBetaBinomial":
        super().fit(train)
        self.pooled_rho = float(self.rho)
        y, n, mu = (train["gp"].to_numpy(), train["team_games"].to_numpy(),
                    self.predict_mean(train))
        buckets = self._buckets(train)
        self.rho_by_role = {}
        for label in ROLE_LABELS:
            mask = buckets == label
            if mask.sum() >= MIN_ROLE_ROWS:
                self.rho_by_role[label] = fit_dispersion(y[mask], n[mask], mu[mask])
        return self

    @property
    def rho_spread(self) -> float:
        vals = list(self.rho_by_role.values())
        return max(vals) / min(vals) if vals else 1.0

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        n, mu = df["team_games"].to_numpy(), self.predict_mean(df)
        buckets = self._buckets(df)
        pmf = np.zeros((len(df), max_games + 1))
        for label in np.unique(buckets):
            mask = buckets == label
            pmf[mask] = predictive_pmf(n[mask], mu[mask],
                                       self.rho_by_role.get(label, self.pooled_rho),
                                       max_games)
        return pmf


def restrict_window(train: pd.DataFrame, first_year: int | None) -> pd.DataFrame:
    """The fitting half, cut to a recent suffix of seasons.

    Applied to the **training rows only**. The design itself is built over every season
    regardless, because the lag columns reach back three seasons and trimming the frame
    would silently drop each window's own first cohort.
    """
    if first_year is None:
        return train
    return train[season_start_year(train) >= first_year].copy()


def tail_coverage(pmf: np.ndarray, y: np.ndarray, n: np.ndarray) -> list[dict]:
    """Predicted against observed frequency at each tail, as rates rather than scores.

    The predicted rate is the *mean of the per-row probabilities*, which is what the
    predictive says the population rate is; the observed rate is the realized one. A model
    can hold CRPS and PIT while getting these wrong by a factor of two, which is the whole
    reason this function exists.
    """
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    rows = np.arange(len(y))
    n_int = np.asarray(n, dtype=int)
    missed = n_int - np.asarray(y, dtype=int)
    out = []
    for threshold in TAIL_BELOW:
        out.append({"tail": f"below_{threshold}",
                    "predicted": float(cdf[:, threshold - 1].mean()),
                    "observed": float((y < threshold).mean())})
    # The upper boundary is each row's OWN schedule, so a 66- or 72-game season
    # contributes the boundary it actually had.
    out.append({"tail": "full_schedule",
                "predicted": float(pmf[rows, n_int].mean()),
                "observed": float((y == n).mean())})
    # The upper SHOULDER, counted in games missed so a shortened season contributes the
    # same event an 82-game one does.
    for m in TAIL_MISSED:
        out.append({"tail": f"missed_le_{m}",
                    "predicted": float(_p_missed_le(pmf, cdf, n_int, m).mean()),
                    "observed": float((missed <= m).mean())})
    # The exclusive bands, so a U-shape cannot net itself out inside one cumulative.
    for name, scale, lo, hi in BANDS:
        if scale == "played":
            predicted = _cdf_at(cdf, hi - 1) - (_cdf_at(cdf, lo - 1) if lo else 0.0)
            observed = (np.asarray(y) >= lo) & (np.asarray(y) < hi)
        else:
            predicted = (_p_missed_le(pmf, cdf, n_int, hi - 1)
                         - (_p_missed_le(pmf, cdf, n_int, lo - 1) if lo else 0.0))
            observed = (missed >= lo) & (missed < hi)
        out.append({"tail": f"band_{name}", "predicted": float(predicted.mean()),
                    "observed": float(observed.mean())})
    for row in out:
        row["error"] = row["predicted"] - row["observed"]
        row["abs_error"] = abs(row["error"])
    return out


def _cdf_at(cdf: np.ndarray, k: int) -> np.ndarray:
    """`P(GP <= k)` per row, clamped to the grid."""
    if k < 0:
        return np.zeros(len(cdf))
    return cdf[:, min(k, cdf.shape[1] - 1)]


def _p_missed_le(pmf: np.ndarray, cdf: np.ndarray, n: np.ndarray, m: int) -> np.ndarray:
    """`P(team_games - GP <= m)` per row — the upper tail on the schedule-invariant scale.

    Equal to `1 - P(GP <= n - m - 1)`, taken at each row's own `n` rather than at a shared
    grid point, which is the whole reason the upper thresholds are counted in games missed.
    """
    if m < 0:
        return np.zeros(len(pmf))
    idx = np.clip(n - m - 1, -1, cdf.shape[1] - 1)
    below = np.where(idx < 0, 0.0,
                     np.take_along_axis(cdf, np.maximum(idx, 0)[:, None], axis=1).ravel())
    return np.clip(1.0 - below, 0.0, 1.0)


def shoulder_shape(pmf: np.ndarray, y: np.ndarray, n: np.ndarray,
                   depth: int = SHAPE_DEPTH) -> dict[str, float]:
    """The largest calibration gap anywhere *inside* each shoulder, not at one point in it.

    A band is one number per region and can be right on average while the distribution
    inside it is the wrong shape — too much mass at 1 game and too little at 8 sums to a
    band error of zero. This walks the region's calibration curve and reports the worst gap
    on it, which is what "the shape of the predicted distribution in the shoulders" asks
    for: a Kolmogorov-style sup distance restricted to a region rather than taken over the
    whole line, where the body's 60 percentage points of mass would swamp it.

    The low side walks `P(GP <= g)` on the absolute grid; the high side walks
    `P(missed <= m)`, for the same schedule-invariance reason `TAIL_MISSED` exists. Both
    compare the *mean predicted* curve against the *observed* one — the population-level
    reading `tail_coverage` uses, not a per-row one.
    """
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    n_int = np.asarray(n, dtype=int)
    missed = n_int - np.asarray(y, dtype=int)
    low = max(abs(float(_cdf_at(cdf, g).mean()) - float((y <= g).mean()))
              for g in range(depth + 1))
    high = max(abs(float(_p_missed_le(pmf, cdf, n_int, m).mean())
                   - float((missed <= m).mean()))
               for m in range(depth + 1))
    return {"low_shape_ks": low, "high_shape_ks": high,
            "shape_ks": max(low, high)}


def interval_coverage(pmf: np.ndarray, y: np.ndarray, level: float) -> float:
    """Realized coverage of the central `level` predictive interval."""
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    alpha = (1.0 - level) / 2.0
    lo = (cdf < alpha).sum(axis=1)
    hi = (cdf < 1.0 - alpha).sum(axis=1)
    return float(((y >= lo) & (y <= hi)).mean())


def build_arm(train: pd.DataFrame, val: pd.DataFrame, spec: str
              ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """`(train, val, features)` for one season-term spec.

    `add_trend` centres on **train**, so each window centres on its own fitting rows and
    the validation seasons sit outside the fitted range in every arm — which is what
    extrapolating one season forward is supposed to mean.
    """
    if spec == "none":
        return train, val, list(FEATURE_COLS)
    (tr, va), trend_names = add_trend(train, [train, val])
    if spec == "trend":
        return tr, va, list(FEATURE_COLS) + trend_names
    (tr, va), role_names = add_role_terms(tr, [tr, va])
    return tr, va, list(FEATURE_COLS) + trend_names + role_names


def paired_bootstrap(arm: np.ndarray, reference: np.ndarray, reps: int = BOOTSTRAP_REPS,
                     seed: int = 42) -> tuple[float, float, float]:
    """`(mean difference, lo, hi)` on arm − reference CRPS, resampling rows in pairs.

    Paired inside the row, because both arms score the same validation player-seasons and
    the between-player variance is an order of magnitude larger than the between-arm one.
    This is the discipline `docs/potential-to-dos.md` names as the difference between a
    finding and a prompt.
    """
    diff = arm - reference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def score_arm(name: str, model, train: pd.DataFrame, val: pd.DataFrame,
              features: list[str], max_games: int, seed: int = 42) -> tuple[dict, np.ndarray]:
    """One fitted arm's row, and its per-row CRPS for the paired bootstrap."""
    y = val["gp"].to_numpy()
    n = val["team_games"].to_numpy()
    mu = model.predict_mean(val)
    # Through the model rather than through `predictive_pmf` directly, so a head that
    # varies its dispersion per row supplies its own predictive instead of being flattened
    # back to a scalar by the scorer.
    pmf = model.predict_pmf(val, max_games)
    scores = crps(pmf, y)

    share, pred_share = y / n, mu
    ss_res = float(np.sum((share - pred_share) ** 2))
    ss_tot = float(np.sum((share - share.mean()) ** 2))
    u = pit_values(pmf, y, seed)
    grid = np.linspace(0, 1, 101)
    ks = float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid)))

    row = {
        "arm": name,
        "n_train": len(train),
        "n_train_seasons": int(pd.Series(train["season"]).nunique()),
        "first_train_season": min(train["season"]),
        "n_features": len(features),
        "val_crps": float(scores.mean()),
        "val_mae": float(np.abs(y - mu * n).mean()),
        "val_r2_gp_share": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "val_pit_ks": ks,
        "rho": float(model.rho),
        "rho_spread": float(getattr(model, "rho_spread", 1.0)),
        "implied_overdispersion": float(1 + (np.median(n) - 1) * model.rho),
    }
    for level in COVERAGE_LEVELS:
        row[f"coverage_{int(level * 100)}"] = interval_coverage(pmf, y, level)
    tails = tail_coverage(pmf, y, n)
    by_name = {t["tail"]: t for t in tails}
    for t in tails:
        row[f"pred_{t['tail']}"] = t["predicted"]
        row[f"obs_{t['tail']}"] = t["observed"]
        row[f"err_{t['tail']}"] = t["error"]
    # Two summaries, because the first one is misleading on its own and that is worth
    # keeping visible rather than deleting. `mean_abs_tail_error` averages all four
    # thresholds and is dominated by 41 and 60, which sit in the BODY of an 82-game
    # distribution, not its tails — an arm that shifts location can improve both
    # boundaries while wrecking those two and still look worse. `boundary_tail_error` is
    # the pair the defect is actually about: a dead roster slot and an iron man.
    # Explicitly the ORIGINAL four thresholds, so adding the shoulder rows below cannot
    # silently change what this number means from one artifact version to the next.
    row["mean_abs_tail_error"] = float(np.mean(
        [by_name[t]["abs_error"] for t in ("below_10", "below_41", "below_60",
                                           "full_schedule")]))
    row["boundary_tail_error"] = float(np.mean([by_name["below_10"]["abs_error"],
                                                by_name["full_schedule"]["abs_error"]]))
    # Reported BESIDE the selector, never averaged into it. `rolling_confirmation` has
    # carried this since the trend arms showed an arm can buy both boundaries by wrecking
    # the middle; the validation table needs it for the same reason.
    row["body_error"] = float(np.mean([by_name["below_41"]["abs_error"],
                                       by_name["below_60"]["abs_error"]]))
    # The SHOULDERS, on the same never-averaged-in footing, added 2026-08-11.
    # `boundary_tail_error` is **asymmetric** — a ten-game-wide low shoulder against a
    # single-point upper boundary — so it can move a long way while 60-81 games, which
    # nothing else in this function looked at, does not move at all.
    row["shoulder_error"] = float(np.mean([by_name["band_low_shoulder"]["abs_error"],
                                           by_name["band_high_shoulder"]["abs_error"]]))
    # And the exact point masses, separated from the shoulders they were pooled with,
    # because §1 measured the two halves of the low tail erring in OPPOSITE directions:
    # P(GP <= 1) over-predicted while P(GP < 10) is under-predicted. Pooled, they cancel.
    row["point_mass_error"] = float(np.mean([by_name["band_zero"]["abs_error"],
                                             by_name["band_full"]["abs_error"]]))
    row.update(shoulder_shape(pmf, y, n))
    return row, scores


def ladder(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
           l2: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Every (window x season-term) arm, plus the league/age floor, scored on validation."""
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}

    floor = LeagueAgeBaseline().fit(train)
    row, scores = score_arm("league_age_floor", floor, train, val, [], max_games, seed)
    rows.append(row)
    per_row["league_age_floor"] = scores

    for window, first_year in WINDOWS.items():
        cut = restrict_window(train, first_year)
        for spec in TREND_SPECS:
            for rho_mode in RHO_MODES:
                name = f"{window}__{spec}__{rho_mode}"
                tr, va, features = build_arm(cut, val, spec)
                cls = BetaBinomialGLM if rho_mode == "shared" else RoleGradedBetaBinomial
                model = cls(l2=l2, features=features).fit(tr)
                row, scores = score_arm(name, model, tr, va, features, max_games, seed)
                rows.append(row)
                per_row[name] = scores
                print(f"  {name:<40} {len(tr):>6,} rows / "
                      f"{row['n_train_seasons']:>2} seasons   CRPS {row['val_crps']:.4f}   "
                      f"rho {row['rho']:.4f} ({row['rho_spread']:.2f}x)   "
                      f"boundary err {row['boundary_tail_error']:.4f}")

    reference = per_row[REFERENCE_ARM]
    for row in rows:
        d, lo, hi = paired_bootstrap(per_row[row["arm"]], reference, seed=seed)
        row["crps_vs_incumbent"] = d
        row["crps_vs_incumbent_lo"] = lo
        row["crps_vs_incumbent_hi"] = hi
        row["beats_incumbent"] = bool(hi < 0.0)
    return pd.DataFrame(rows)


# ── Rolling-origin confirmation ───────────────────────────────────────────────
#
# The ladder above scores 19 arms on the 883 validation rows, which is a multiplicity
# problem and a power problem at once. This block answers it **without spending validation
# twice**: it walks an origin across the training half, fits on the seasons before it and
# scores the season itself, and pools. Every row it touches is a fitting-half row, so the
# validation reading stays the arbiter it was — this asks only whether that reading
# replicates.
#
# It also reparameterizes the window in the way that actually matters. `WINDOWS` names an
# absolute first season, which is a fact about the past; a **lookback length** is a policy
# that will still mean something when the production fit runs for 2026-27. Comparing
# lookbacks across origins is the bias-variance curve directly.

# Lookback lengths in target seasons. `None` is "every season before the origin", the
# incumbent's rule. 12 is the longest that is genuine at the earliest origin below.
LOOKBACKS: tuple[int | None, ...] = (None, 12, 8, 5, 3)

# First origin. Chosen so the longest finite lookback is real rather than silently
# truncated to `None` — the training half starts at target season 1997-98, so 1997 + 12.
FIRST_ORIGIN = 2009

ROLLING_REFERENCE = "all__none__shared"


def _origin_scores(model, score: pd.DataFrame, max_games: int) -> dict[str, np.ndarray]:
    """Per-row quantities for pooling across origins.

    The **body** thresholds are carried as well as the boundaries, because the whole
    verdict on a season trend turns on whether it pays for its boundaries out of the
    middle. Reporting only the boundaries here would have made the trend look free.
    """
    y = score["gp"].to_numpy()
    n = score["team_games"].to_numpy()
    pmf = model.predict_pmf(score, max_games)
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    out = {"crps": crps(pmf, y), "pit": pit_values(pmf, y),
           "p_full": pmf[np.arange(len(y)), n.astype(int)], "y": y, "n": n}
    for threshold in TAIL_BELOW:
        out[f"p_below_{threshold}"] = cdf[:, threshold - 1]
    # The upper shoulder, on the games-missed scale, plus the exact zero — the two regions
    # the original threshold set could not see. Carried per row so they pool across
    # origins exactly as the others do.
    n_int = n.astype(int)
    out["p_zero"] = pmf[:, 0]
    for m in TAIL_MISSED:
        out[f"p_missed_le_{m}"] = _p_missed_le(pmf, cdf, n_int, m)
    out["p_band_low_shoulder"] = cdf[:, 9] - pmf[:, 0]
    out["p_band_high_shoulder"] = (_p_missed_le(pmf, cdf, n_int, 11)
                                   - _p_missed_le(pmf, cdf, n_int, 0))
    return out


def rolling_confirmation(train: pd.DataFrame, max_games: int, l2: float = 1.0,
                         seed: int = 42) -> pd.DataFrame:
    """Walk-forward over the fitting half: one fit per (origin, arm), pooled.

    The origin is a *target* season; the arm fits on target seasons strictly before it.
    Nothing here reads the validation or held-out rows, so this is a confirmation of the
    ladder rather than a second selection on the same data.
    """
    years = np.asarray(sorted(np.unique(season_start_year(train))))
    origins = [int(y) for y in years if y >= FIRST_ORIGIN]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        year = season_start_year(train)
        score = train[year == origin]
        if score.empty:
            continue
        for lookback in LOOKBACKS:
            floor_year = -np.inf if lookback is None else origin - lookback
            fit_rows = train[(year < origin) & (year >= floor_year)]
            if len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                continue
            for spec in ("none", "trend"):
                for rho_mode in RHO_MODES:
                    name = f"{'all' if lookback is None else lookback}__{spec}__{rho_mode}"
                    tr, sc, features = build_arm(fit_rows, score, spec)
                    cls = BetaBinomialGLM if rho_mode == "shared" else RoleGradedBetaBinomial
                    model = cls(l2=l2, features=features).fit(tr)
                    scored = _origin_scores(model, sc, max_games)
                    scored["origin"] = np.full(len(sc), origin)
                    slot = per_arm.setdefault(name, {"n_fit": []})
                    for key, values in scored.items():
                        slot.setdefault(key, []).append(values)
                    slot["n_fit"].append(len(tr))
        print(f"  origin {origin}: scored {len(score):,} rows")

    pooled = {k: {m: np.concatenate(v) for m, v in d.items() if m != "n_fit"}
              for k, d in per_arm.items()}
    reference = pooled[ROLLING_REFERENCE]["crps"]
    grid = np.linspace(0, 1, 101)
    rows = []
    for name, d in pooled.items():
        y, n, org = d["y"], d["n"], d["origin"]
        mean, lo, hi = paired_bootstrap(d["crps"], reference, seed=seed)
        # Origins are the independent replicates, so a win count over them is the
        # multiplicity-robust statement a pooled interval is not.
        wins = sum(1 for o in np.unique(org)
                   if d["crps"][org == o].mean() < reference[org == o].mean())
        u = d["pit"]
        rows.append({
            "arm": name,
            "lookback": name.split("__")[0],
            "spec": name.split("__")[1],
            "rho_mode": name.split("__")[2],
            "n_origins": int(len(np.unique(org))),
            "n_scored": int(len(y)),
            "mean_fit_rows": float(np.mean(per_arm[name]["n_fit"])),
            "crps": float(d["crps"].mean()),
            "crps_vs_all": mean,
            "crps_vs_all_lo": lo,
            "crps_vs_all_hi": hi,
            "origins_won": wins,
            "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid))),
            "pred_full_schedule": float(d["p_full"].mean()),
            "obs_full_schedule": float((y == n).mean()),
            **{f"pred_below_{t}": float(d[f"p_below_{t}"].mean()) for t in TAIL_BELOW},
            **{f"obs_below_{t}": float((y < t).mean()) for t in TAIL_BELOW},
            **{f"pred_missed_le_{m}": float(d[f"p_missed_le_{m}"].mean())
               for m in TAIL_MISSED},
            **{f"obs_missed_le_{m}": float(((n - y) <= m).mean()) for m in TAIL_MISSED},
            "pred_band_zero": float(d["p_zero"].mean()),
            "obs_band_zero": float((y == 0).mean()),
            "pred_band_low_shoulder": float(d["p_band_low_shoulder"].mean()),
            "obs_band_low_shoulder": float(((y >= 1) & (y < 10)).mean()),
            "pred_band_high_shoulder": float(d["p_band_high_shoulder"].mean()),
            "obs_band_high_shoulder": float((((n - y) >= 1) & ((n - y) < 12)).mean()),
        })
    out = pd.DataFrame(rows)
    out["err_full_schedule"] = out["pred_full_schedule"] - out["obs_full_schedule"]
    for threshold in TAIL_BELOW:
        out[f"err_below_{threshold}"] = (out[f"pred_below_{threshold}"]
                                         - out[f"obs_below_{threshold}"])
    for m in TAIL_MISSED:
        out[f"err_missed_le_{m}"] = (out[f"pred_missed_le_{m}"]
                                     - out[f"obs_missed_le_{m}"])
    for band in ("zero", "low_shoulder", "high_shoulder"):
        out[f"err_band_{band}"] = out[f"pred_band_{band}"] - out[f"obs_band_{band}"]
    out["boundary_tail_error"] = (out["err_below_10"].abs()
                                  + out["err_full_schedule"].abs()) / 2
    # The body, carried beside the boundaries so a location shift cannot look free.
    out["body_error"] = (out["err_below_41"].abs() + out["err_below_60"].abs()) / 2
    # The shoulders, beside both, for the reason `score_arm` documents: the boundary pair
    # is asymmetric and 60-81 games was measured by nothing.
    out["shoulder_error"] = (out["err_band_low_shoulder"].abs()
                             + out["err_band_high_shoulder"].abs()) / 2
    return out.sort_values("crps").reset_index(drop=True)


# ── The fourth axis: the likelihood ───────────────────────────────────────────
#
# Every arm below is the same mean function under a different frailty, fitted on the same
# windowed rows with the same `l2`, and scored by the same `score_arm`. Three design rules
# hold across all of them.
#
# **1. The dispersion stays role-graded**, because the axis is the likelihood alone and the
# shipped arm grades it. What "dispersion" means is arm-specific — `rho` for the Beta
# families, `sigma_u` for the logit-normal — so the base class carries a *dispersion vector
# per role bucket* with an arm-supplied meaning, rather than a `rho` the logit-normal would
# have to fake.
#
# **2. Every arm starts at the incumbent and is asserted to reproduce it there.**
# `assert_nests` evaluates the arm's own log-likelihood at its nesting parameter values and
# compares it against `RoleGradedBetaBinomial`'s on the same rows. This is the discipline
# `n_rho = 1` and `U_n = 0` already carry in the Stan sources, and it is the rollback path:
# an arm that cannot reproduce the head it extends is a different model, not an extension.
# The optimizer starts from that point too, so no arm can score worse than the incumbent on
# the *fitting* rows — which is exactly why the validation column is the one that decides.
#
# **3. `l2 = 1.0` is pinned across arms, and that is NOT neutral between them.** The penalty
# reaches `beta[1:]` only, so the mixture arms carry 1 to 11 extra *unpenalized* parameters
# against the incumbent's zero. A fixed penalty therefore favours the larger arms, and any
# win here is an upper bound on the likelihood's own contribution. `docs/availability-window-
# plan.md` §5.6 records the same confound one axis over, where `l2` was pinned across
# lookbacks of very different row counts. Stated rather than corrected, because correcting
# it means a second selection axis and this ladder already has a multiplicity problem.

# Gauss-Hermite nodes for the logit-normal's integral over the frailty. 24 is far past
# convergence for a smooth unimodal integrand — the log-likelihood moves by < 1e-9 between
# 24 and 48 nodes — and costs 24 binomial evaluations per row.
GH_NODES = 24

# Covariates on the disrupted-season probability. `docs/potential-to-dos.md` item 5 names
# three families — age, prior absence, playoff workload — and this is that list made
# concrete, not the whole 19-column block: pi carries its own coefficient per column, and
# nineteen more unpenalized parameters on 4,027 rows would be measuring the confound in
# rule 3 rather than the mechanism.
PI_COLS = ["age", "age_sq", "gp_share_lag1", "trailing_missed_lag1",
           "n_spells_lag1", "longest_spell_lag1",
           "playoff_games_lag1", "career_minutes_lag1"]

# A "disrupted season" is one where the player misses more than half the schedule. The
# bound is what stops the two components from label-switching — with a scalar mean against a
# covariate-driven one they are already distinguishable, but a bound makes it structural
# rather than hopeful. Reported in the output so a fit that pins against it is visible.
MU_LOW_MAX = 0.5

# Latent durability classes in the finite mixture. 3 is the arm; 2 and 4 are fitted beside
# it as a sensitivity on K rather than as competitors for selection.
FINITE_MIX_K = 3
FINITE_MIX_K_SENSITIVITY = (2, 4)

# The prior-MPG column the role buckets cut on, and the guarded value `_neg_ll` returns
# where the likelihood is non-finite — the same device `availability._neg_loglik` uses, for
# the same reason: an unguarded nan makes L-BFGS-B's numeric gradient nan and it stops at
# wherever it had reached, which looks exactly like a fitted model.
ROLE_COL_LADDER = "minutes_per_game_lag1"
_LARGE = 1e12


def _bb_logpmf(y: np.ndarray, n: np.ndarray, mu: np.ndarray,
               rho: np.ndarray) -> np.ndarray:
    """Beta-binomial log-pmf with a per-row dispersion, finite everywhere."""
    a, b = _ab_row(mu, rho)
    return np.nan_to_num(betabinom.logpmf(y, n, a, b), nan=-700.0,
                         neginf=-700.0, posinf=-700.0)


def _ab_row(mu: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`availability._ab`, vectorized over a per-row `rho` instead of a scalar."""
    mu = np.clip(np.asarray(mu, dtype=float), EPS, 1 - EPS)
    rho = np.clip(np.asarray(rho, dtype=float), RHO_MIN, RHO_MAX)
    scale = (1.0 - rho) / rho
    return mu * scale, (1.0 - mu) * scale


def _bb_dlogpmf_dmu(y: np.ndarray, n: np.ndarray, mu: np.ndarray,
                    rho: np.ndarray) -> np.ndarray:
    """d log BetaBinom(y; n, mu, rho) / d mu — the same expression `BetaBinomialGLM`
    already differentiates, with `a + b = scale` free of `mu` so its digamma terms cancel."""
    a, b = _ab_row(mu, rho)
    scale = a + b
    return np.nan_to_num(scale * (digamma(y + a) - digamma(a)
                                  - digamma(n - y + b) + digamma(b)))


def _bb_pmf_grid(n: np.ndarray, mu: np.ndarray, rho: np.ndarray,
                 k: np.ndarray) -> np.ndarray:
    a, b = _ab_row(mu, rho)
    return np.nan_to_num(betabinom.pmf(k[None, :], np.asarray(n)[:, None],
                                       a[:, None], b[:, None]))


class FrailtyGLM(AvailabilityModel):
    """One logit mean function, one role-graded dispersion, a pluggable frailty.

    Subclasses supply four things and inherit everything else, so the ladder's contrast is
    the frailty and never a second implementation of the fit, the pmf grid or the scorer:

    - `_extra0()`  — the nesting parameter values, and the optimizer's start
    - `_bounds()`  — box constraints for the non-beta block
    - `_terms()`   — per-row log-likelihood and `d log L / d eta`
    - `_pmf()`     — the (rows x games+1) predictive

    **The fit is alternating, for the reason `BetaBinomialGLM` documents.** A joint
    numeric-gradient fit over 20-plus parameters does not work on an objective of magnitude
    1e4: a finite-difference step moves it by ~1e-3 and L-BFGS-B stops on gradient noise,
    producing coefficients that are essentially the ones it started with. So `beta` is
    optimized with an analytic gradient holding the frailty block fixed, and the frailty
    block — never more than 12 parameters, all on bounded or O(1) scales — is optimized
    numerically holding `beta` fixed, until the log-likelihood stops moving.
    """

    name = "frailty"
    #: Human-readable dispersion name, for the output table.
    dispersion_label = "rho"

    def __init__(self, l2: float = 1.0, features: list[str] | None = None,
                 max_rounds: int = 12, tol: float = 1e-6):
        self.l2 = l2
        self.features = list(features or FEATURE_COLS)
        self.max_rounds = max_rounds
        self.tol = tol

    # ── design ────────────────────────────────────────────────────────────────
    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[self.features].to_numpy(dtype=float))
        return np.column_stack([np.ones(len(X)), X])

    def _pi_design(self, df: pd.DataFrame) -> np.ndarray:
        return self.pi_scaler.transform(df[PI_COLS].to_numpy(dtype=float))

    def _buckets(self, df: pd.DataFrame) -> np.ndarray:
        """1-based prior-MPG role bucket, the same cut `RoleGradedBetaBinomial` uses."""
        idx = pd.cut(df[ROLE_COL_LADDER].to_numpy(dtype=float), ROLE_EDGES, labels=False)
        return np.nan_to_num(np.asarray(idx, dtype=float), nan=0.0).astype(int)

    def _disp_row(self, df: pd.DataFrame, disp: np.ndarray) -> np.ndarray:
        return disp[self._buckets(df)]

    # ── the arm's contract ────────────────────────────────────────────────────
    def _extra0(self) -> np.ndarray:
        return np.zeros(0)

    def _extra_bounds(self) -> list[tuple[float | None, float | None]]:
        return []

    def _fit_starts(self) -> list[np.ndarray]:
        """Where the optimizer *starts*, which is not the same as where the arm *nests*.

        The nesting point is a collapsed mixture, and a collapsed mixture is the worst
        possible start for a mixture likelihood: every component sees the same responsibility,
        so the surface is flat in the direction that separates them and L-BFGS-B can sit on
        the bound it started on and report success. Measured — a three-class `finite_mix`
        started at `g = 0` fitted offsets of 0.007 and 0.009 and reproduced the incumbent to
        four decimals, which is an optimizer artifact wearing a null's clothes.

        So the arms with separable components hand back several starts and `fit` keeps the
        best by *training* log-likelihood. `_extra0` stays the nesting point and is what
        `assert_nests` checks, unchanged.
        """
        return [np.asarray(self._extra0(), dtype=float)]

    def _terms(self, y, n, eta, disp_row, extra, df) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _pmf(self, n, eta, disp_row, extra, k, df) -> np.ndarray:
        raise NotImplementedError

    def _nesting_disp(self) -> np.ndarray:
        """The dispersion at the arm's nesting point.

        The incumbent's own fitted value for the Beta families, where the extra parameters
        are what switch the arm off. `LogitNormalFrailty` overrides it with zero, because
        what *it* nests is the binomial and `sigma_u = 0` is where that lives.
        """
        return self.incumbent_disp

    def shape_report(self, df: pd.DataFrame) -> dict:
        """The diagnostic the axis exists to move: does the frailty diverge at a boundary?

        For the Beta families this is `b < 1` at `p = 1` and `a < 1` at `p = 0`. For the
        logit-normal it is 0 by construction, which is the whole reason that arm is in the
        ladder.

        **Two versions, and the weighted one is the one to read.** `diverges_at_one` is the
        share of rows where *any* component diverges, which is the statistic
        `docs/potential-to-dos.md` item 5 measured on a single-component head. On a
        multi-component arm it is close to mechanical — spread four classes across the mean
        and one of them will sit high enough — so `diverges_at_one_weighted` reports the
        share of predictive *mass* sitting under a divergent frailty, which is the same
        number on the incumbent and a real one on the alternatives.
        """
        raise NotImplementedError

    # ── fit ───────────────────────────────────────────────────────────────────
    def _neg_ll(self, beta, disp_raw, extra, y, n, X, df) -> float:
        ll, _ = self._terms(y, n, X @ beta, self._disp_row(df, self._disp(disp_raw)),
                            extra, df)
        return _LARGE if not np.all(np.isfinite(ll)) else -float(ll.sum())

    @staticmethod
    def _disp(raw: np.ndarray) -> np.ndarray:
        """Bucket dispersions from their unconstrained parameterization."""
        return np.clip(_sigmoid(np.asarray(raw, dtype=float)), RHO_MIN, RHO_MAX)

    @staticmethod
    def _disp_raw(values: np.ndarray) -> np.ndarray:
        v = np.clip(np.asarray(values, dtype=float), RHO_MIN, RHO_MAX)
        return np.log(v / (1.0 - v))

    def _start(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """`(beta, dispersion-per-bucket)` at the incumbent — the shipped arm's own fit.

        Every arm starts here, which is what makes the nesting assertion a statement about
        the *fitted* incumbent rather than about an arbitrary point in its family.
        """
        base = RoleGradedBetaBinomial(l2=self.l2, features=self.features).fit(train)
        self.scaler = base.scaler
        disp = np.array([base.rho_by_role.get(label, base.pooled_rho)
                         for label in ROLE_LABELS], dtype=float)
        # The common yardstick: every arm's fitted log-likelihood is reported against the
        # *incumbent's* on the same rows, including the one arm that does not nest it.
        self.incumbent_loglik = float(
            _bb_logpmf(train["gp"].to_numpy(dtype=float),
                       train["team_games"].to_numpy(dtype=float),
                       base.predict_mean(train), disp[self._buckets(train)]).sum())
        return np.asarray(base.beta, dtype=float), disp

    def fit(self, train: pd.DataFrame) -> "FrailtyGLM":
        self.pi_scaler = StandardScaler().fit(train[PI_COLS].to_numpy(dtype=float))
        beta, disp = self._start(train)
        self.incumbent_beta, self.incumbent_disp = beta.copy(), disp.copy()
        X = self._design(train)
        y = train["gp"].to_numpy(dtype=float)
        n = train["team_games"].to_numpy(dtype=float)

        disp0 = self._disp_raw(disp)
        nest_ll, _ = self._terms(y, n, X @ beta, self._disp_row(train, self._nesting_disp()),
                                 np.asarray(self._extra0(), dtype=float), train)
        self.nesting_loglik = float(nest_ll.sum())

        n_disp = len(disp0)
        bounds = [(-14.0, 3.0)] * n_disp + list(self._extra_bounds())

        def alternate(start: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
            b, disp_raw, extra = beta.copy(), disp0.copy(), np.asarray(start, dtype=float)

            def beta_objective(bb: np.ndarray) -> tuple[float, np.ndarray]:
                ll, deta = self._terms(y, n, X @ bb,
                                       self._disp_row(train, self._disp(disp_raw)),
                                       extra, train)
                if not np.all(np.isfinite(ll)):
                    return _LARGE, np.zeros_like(bb)
                value = -float(ll.sum()) + self.l2 * float(bb[1:] @ bb[1:])
                grad = -(X.T @ deta)
                grad[1:] += 2.0 * self.l2 * bb[1:]
                return value, np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)

            def frailty_objective(theta: np.ndarray) -> float:
                return self._neg_ll(b, theta[:n_disp], theta[n_disp:], y, n, X, train)

            previous = np.inf
            for _ in range(self.max_rounds):
                b = minimize(beta_objective, b, jac=True, method="L-BFGS-B").x
                theta0 = np.concatenate([disp_raw, extra])
                best = minimize(frailty_objective, theta0, method="L-BFGS-B",
                                bounds=bounds, options={"eps": 1e-6, "maxiter": 500})
                if best.fun <= frailty_objective(theta0):
                    disp_raw, extra = best.x[:n_disp], best.x[n_disp:]
                value = self._neg_ll(b, disp_raw, extra, y, n, X, train)
                if abs(previous - value) < self.tol:
                    break
                previous = value
            return b, disp_raw, extra, value

        fits = [alternate(start) for start in self._fit_starts()]
        beta, disp_raw, extra, value = min(fits, key=lambda f: f[3])
        self.n_starts = len(fits)
        self.start_spread = float(max(f[3] for f in fits) - min(f[3] for f in fits))

        self.beta, self.extra = beta, extra
        self.dispersion = self._disp(disp_raw)
        self.train_loglik = -value
        self.n_params = int(len(beta) + n_disp + len(extra))
        # `score_arm` reads these two; for the graded arms `rho` is the population-weighted
        # dispersion rather than a fitted scalar, and is labelled as such in the table.
        counts = np.bincount(self._buckets(train), minlength=len(ROLE_LABELS))
        self.rho = float(np.average(self.dispersion, weights=np.maximum(counts, 1)))
        self.rho_spread = float(self.dispersion.max() / self.dispersion.min())
        return self

    # ── predict ───────────────────────────────────────────────────────────────
    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        k = np.arange(int(max_games) + 1)
        return self._pmf(df["team_games"].to_numpy(dtype=float), self._design(df) @ self.beta,
                         self._disp_row(df, self.dispersion), self.extra, k, df)

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        """The **predictive** mean share, not the mean of the main component.

        A mixture's mean is a weighted mean, and `score_arm` turns this into MAE and R² in
        games. Returning the beta-binomial component's mean would credit a mixture arm with
        an accuracy its own predictive does not have.
        """
        n = df["team_games"].to_numpy(dtype=float)
        pmf = self.predict_pmf(df, int(n.max()))
        k = np.arange(pmf.shape[1])
        return np.clip((pmf * k[None, :]).sum(axis=1) / np.maximum(n, 1), EPS, 1 - EPS)


class BetaBinomFrailty(FrailtyGLM):
    """The incumbent's likelihood, refitted through this class — the ladder's reference row.

    It exists so the reference is produced by the *same* fit loop as the alternatives.
    `RoleGradedBetaBinomial` fits `beta` under a shared `rho` and then profiles `rho` per
    bucket holding the mean fixed; this fits both jointly. Scoring a profiled reference
    against joint alternatives would confound the likelihood with the estimator, which is
    the one thing this axis is supposed to isolate.

    The two therefore agree at the **nesting point** and not at the fitted one — `assert_nests`
    checks the former, and `likelihood_ladder` carries the latter as its own row so the size
    of the estimator difference is visible rather than absorbed.
    """

    name = "betabinom"

    def _terms(self, y, n, eta, disp_row, extra, df):
        mu = _sigmoid(eta)
        ll = _bb_logpmf(y, n, mu, disp_row)
        return ll, _bb_dlogpmf_dmu(y, n, mu, disp_row) * mu * (1.0 - mu)

    def _pmf(self, n, eta, disp_row, extra, k, df):
        return _bb_pmf_grid(n, _sigmoid(eta), disp_row, k)

    def shape_report(self, df: pd.DataFrame) -> dict:
        mu = _sigmoid(self._design(df) @ self.beta)
        a, b = _ab_row(mu, self._disp_row(df, self.dispersion))
        # One component, so weighted and unweighted are the same number by construction.
        return {"diverges_at_one": float((b < 1).mean()),
                "diverges_at_zero": float((a < 1).mean()),
                "diverges_at_one_weighted": float((b < 1).mean()),
                "diverges_at_zero_weighted": float((a < 1).mean()),
                "rho": ";".join(f"{r:.4f}" for r in self.dispersion)}


class MixtureFrailty(FrailtyGLM):
    """`pi_i * BetaBinom(mu_low, rho_low) + (1 - pi_i) * BetaBinom(mu_i, rho_i)`.

    The hypothesis is that the low tail is **not a frailty at all**: an Achilles rupture in
    October is a different event, not an extreme draw of a per-game rate. So the disrupted
    season gets its own component with its own mean and dispersion, and `pi` carries
    covariates — the arm can say *who* is at risk, which a wider frailty cannot.

    `pi_i = theta * sigmoid(gamma' z_i)` with `theta` bounded in [0, 1] rather than the
    plain `sigmoid(gamma_0 + gamma' z_i)`, for one reason: **`theta = 0` is attainable
    exactly**, so `pi = 0` reproduces the incumbent at a finite parameter value instead of
    in a limit. Mixture weights sit on the boundary of their parameter space and a nesting
    assertion that has to be taken as `gamma_0 -> -inf` is a statement about floating point
    rather than about the model. `z` has no intercept because `theta` is the scale.
    """

    name = "mixture"

    def _extra0(self) -> np.ndarray:
        # [theta, mu_low (logit), rho_low (logit), gamma...]
        return np.concatenate([[0.0, float(np.log(0.15 / 0.85)), self._disp_raw([0.3])[0]],
                               np.zeros(len(PI_COLS))])

    def _extra_bounds(self):
        return ([(0.0, 1.0),
                 (None, float(np.log(MU_LOW_MAX / (1 - MU_LOW_MAX)))),
                 (-14.0, 3.0)] + [(-5.0, 5.0)] * len(PI_COLS))

    def _fit_starts(self):
        # The nesting point, plus two live disruption rates. 8% and 25% bracket the
        # plausible range: the observed P(GP < 10) is 8.15% on validation and 5.70% on the
        # fitting half, and a quarter of the league is far more than any reading of it.
        starts = [np.asarray(self._extra0(), dtype=float)]
        for theta in (0.08, 0.25):
            s = starts[0].copy()
            s[0] = theta
            starts.append(s)
        return starts

    def _parts(self, extra, df):
        theta = float(np.clip(extra[0], 0.0, 1.0))
        mu_low = float(np.clip(_sigmoid(extra[1]), EPS, MU_LOW_MAX))
        rho_low = float(self._disp(extra[2]))
        pi = theta * _sigmoid(self._pi_design(df) @ np.asarray(extra[3:], dtype=float))
        return pi, mu_low, rho_low

    def _terms(self, y, n, eta, disp_row, extra, df):
        pi, mu_low, rho_low = self._parts(extra, df)
        mu = _sigmoid(eta)
        ll_main = _bb_logpmf(y, n, mu, disp_row)
        ll_low = _bb_logpmf(y, n, np.full_like(mu, mu_low), np.full_like(mu, rho_low))
        stack = np.vstack([np.log(np.maximum(1.0 - pi, 1e-300)) + ll_main,
                           np.log(np.maximum(pi, 1e-300)) + ll_low])
        ll = logsumexp(stack, axis=0)
        # Only the main component depends on eta, so the gradient is the incumbent's,
        # weighted by that component's posterior responsibility.
        resp = np.exp(stack[0] - ll)
        return ll, resp * _bb_dlogpmf_dmu(y, n, mu, disp_row) * mu * (1.0 - mu)

    def _pmf(self, n, eta, disp_row, extra, k, df):
        pi, mu_low, rho_low = self._parts(extra, df)
        mu = _sigmoid(eta)
        main = _bb_pmf_grid(n, mu, disp_row, k)
        low = _bb_pmf_grid(n, np.full_like(mu, mu_low), np.full_like(mu, rho_low), k)
        return (1.0 - pi)[:, None] * main + pi[:, None] * low

    def shape_report(self, df: pd.DataFrame) -> dict:
        pi, mu_low, rho_low = self._parts(self.extra, df)
        mu = _sigmoid(self._design(df) @ self.beta)
        a, b = _ab_row(mu, self._disp_row(df, self.dispersion))
        a_low, b_low = _ab_row(np.full_like(mu, mu_low), np.full_like(mu, rho_low))
        return {"diverges_at_one": float(((b < 1) | (b_low < 1)).mean()),
                "diverges_at_zero": float(((a < 1) | (a_low < 1)).mean()),
                "diverges_at_one_weighted": float(((1 - pi) * (b < 1)
                                                   + pi * (b_low < 1)).mean()),
                "diverges_at_zero_weighted": float(((1 - pi) * (a < 1)
                                                    + pi * (a_low < 1)).mean()),
                "pi_mean": float(pi.mean()), "theta": float(np.clip(self.extra[0], 0, 1)),
                # Whether `pi`'s covariates carry anything is the arm's own claim — that it
                # can say WHO is at risk, which a wider frailty cannot. A flat `pi` would
                # make this a two-component mixture with a constant weight, i.e. `K = 2`
                # of the arm below, and the spread is what distinguishes the two.
                "pi_sd": float(pi.std()),
                "pi_p10": float(np.percentile(pi, 10)),
                "pi_p90": float(np.percentile(pi, 90)),
                "mu_low": mu_low, "rho_low": rho_low,
                "mu_low_at_bound": bool(mu_low >= MU_LOW_MAX - 1e-6),
                "rho": ";".join(f"{r:.4f}" for r in self.dispersion)}


class FiniteMixtureFrailty(FrailtyGLM):
    """K latent durability classes: `sum_k w_k * BetaBinom(sigmoid(eta + delta_k), rho_i)`.

    The Heckman-Singer device — replace the continuous mixing distribution with a discrete
    one on K support points — with one change forced by the house nesting rule. The pure
    version puts **binomials** at the support points, and at K = 1 that is a binomial rather
    than the incumbent. Here the components are beta-binomials with **ordered mean offsets**,
    which nests the incumbent exactly at `delta = 0` and contains the pure version as the
    `rho -> 0` corner of the same family. So the fitted `rho` is itself the readout: if the
    discrete classes are doing the frailty's job, `rho` should collapse toward zero, and if
    it does not, the shoulders wanted a continuous frailty as well as a discrete one.

    Ordering is `delta_1 = 0`, `delta_k = delta_{k-1} + g_k` with `g_k >= 0` as a *bounded*
    parameter rather than `exp(s_k)` — for the same reason `MixtureFrailty` bounds `theta`.
    `g = 0` is the nesting point and has to be reachable, and `exp(s) = 0` is not.
    Label-switching is impossible under an ordering constraint, which is the trap
    `docs/potential-to-dos.md` item 5 names for this arm.
    """

    name = "finite_mix"

    def __init__(self, *args, k: int = FINITE_MIX_K, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = int(k)
        self.name = f"finite_mix_k{self.k}"

    def _extra0(self) -> np.ndarray:
        # [g_2..g_K, weight logits for classes 2..K]. All classes identical at g = 0,
        # which is the incumbent regardless of the weights.
        return np.zeros(2 * (self.k - 1))

    def _extra_bounds(self):
        return [(0.0, 6.0)] * (self.k - 1) + [(-8.0, 8.0)] * (self.k - 1)

    def _fit_starts(self):
        # Separated classes, because the collapsed point is a saddle for this arm: at
        # `g = 0` every class carries the same responsibility and the gradient that would
        # pull them apart is the one the surface is flat in. Gaps of 0.5 and 1.5 on the
        # logit scale are roughly 12 and 30 games of separation at the population mean.
        starts = [np.asarray(self._extra0(), dtype=float)]
        for gap in (0.5, 1.5):
            s = starts[0].copy()
            s[:self.k - 1] = gap
            starts.append(s)
        return starts

    def _parts(self, extra):
        g = np.clip(np.asarray(extra[:self.k - 1], dtype=float), 0.0, None)
        offsets = np.concatenate([[0.0], np.cumsum(g)])
        logits = np.concatenate([[0.0], np.asarray(extra[self.k - 1:], dtype=float)])
        weights = np.exp(logits - logsumexp(logits))
        return offsets, weights

    def _terms(self, y, n, eta, disp_row, extra, df):
        offsets, weights = self._parts(extra)
        mus = _sigmoid(eta[None, :] + offsets[:, None])
        stack = np.vstack([np.log(max(w, 1e-300)) + _bb_logpmf(y, n, m, disp_row)
                           for w, m in zip(weights, mus)])
        ll = logsumexp(stack, axis=0)
        resp = np.exp(stack - ll[None, :])
        deta = sum(resp[j] * _bb_dlogpmf_dmu(y, n, mus[j], disp_row)
                   * mus[j] * (1.0 - mus[j]) for j in range(len(weights)))
        return ll, deta

    def _pmf(self, n, eta, disp_row, extra, k, df):
        offsets, weights = self._parts(extra)
        out = np.zeros((len(n), len(k)))
        for w, off in zip(weights, offsets):
            out += w * _bb_pmf_grid(n, _sigmoid(eta + off), disp_row, k)
        return out

    def shape_report(self, df: pd.DataFrame) -> dict:
        offsets, weights = self._parts(self.extra)
        eta = self._design(df) @ self.beta
        disp_row = self._disp_row(df, self.dispersion)
        one = np.zeros(len(df), dtype=bool)
        zero = np.zeros(len(df), dtype=bool)
        one_w = np.zeros(len(df))
        zero_w = np.zeros(len(df))
        for off, w in zip(offsets, weights):
            a, b = _ab_row(_sigmoid(eta + off), disp_row)
            one |= b < 1
            zero |= a < 1
            one_w += w * (b < 1)
            zero_w += w * (a < 1)
        return {"diverges_at_one": float(one.mean()), "diverges_at_zero": float(zero.mean()),
                "diverges_at_one_weighted": float(one_w.mean()),
                "diverges_at_zero_weighted": float(zero_w.mean()),
                "class_offsets": ";".join(f"{o:.3f}" for o in offsets),
                "class_weights": ";".join(f"{w:.3f}" for w in weights),
                "class_means": ";".join(f"{m:.3f}" for m in
                                        _sigmoid(float(eta.mean()) + offsets)),
                "rho": ";".join(f"{r:.4f}" for r in self.dispersion)}


class LogitNormalFrailty(FrailtyGLM):
    """A binomial GLMM: `y ~ Binomial(n, sigmoid(eta + sigma_u * x))`, `x ~ N(0, 1)`.

    **This arm does not nest the incumbent, and it is here because of that.** Its tails are
    lighter than the Beta's at both ends — the logit-normal density *vanishes* at `p = 0`
    and `p = 1` rather than diverging — so if the boundary mass is a shape defect it should
    fix the high tail by construction and worsen the low one. An arm that fails the *other
    way* is what separates "the frailty's shape is wrong" from "a component is missing",
    which no single arm can do on its own.

    What it does nest is the **binomial**, at `sigma_u = 0`, which is the `rho -> 0` corner
    of the beta-binomial. `assert_nests` checks that instead, exactly, so the arm is still
    pinned to something rather than trusted.

    `sigma_u` is graded by role bucket, because the axis holds the dispersion at `role` and
    `sigma_u` is this family's dispersion. The integral is Gauss-Hermite over `GH_NODES`.
    """

    name = "logitnormal"
    dispersion_label = "sigma_u"

    @staticmethod
    def _disp(raw):
        # sigma_u on a log scale rather than a logit one: it is a positive scale, not a
        # proportion, and `_disp_raw` is inverted to match.
        return np.clip(np.exp(np.asarray(raw, dtype=float)), 1e-8, 5.0)

    @staticmethod
    def _disp_raw(values):
        return np.log(np.clip(np.asarray(values, dtype=float), 1e-8, 5.0))

    def _nesting_disp(self) -> np.ndarray:
        # Exactly zero, not `_disp`'s floor: at 1e-8 the Gauss-Hermite sum differs from the
        # binomial in the seventh decimal per row and by ~4e-4 summed, which would turn an
        # equality assertion into a tolerance negotiation.
        return np.zeros(len(ROLE_LABELS))

    def _start(self, train):
        beta, rho = super()._start(train)
        # Match the beta-binomial's frailty variance on the logit scale rather than
        # starting at an arbitrary scale: Var[p] = mu(1-mu)rho, and the delta method sends
        # that to Var[logit p] = rho / (mu(1-mu)) at the population mean.
        share = float((train["gp"].sum()) / train["team_games"].sum())
        return beta, np.sqrt(np.asarray(rho) / (share * (1 - share)))

    def _nodes(self):
        x, w = np.polynomial.hermite_e.hermegauss(GH_NODES)
        return x, w / np.sqrt(2.0 * np.pi)

    def _terms(self, y, n, eta, disp_row, extra, df):
        x, w = self._nodes()
        lin = eta[None, :] + disp_row[None, :] * x[:, None]
        p = _sigmoid(lin)
        stack = np.log(w)[:, None] + np.nan_to_num(binom.logpmf(y[None, :], n[None, :], p),
                                                   nan=-700.0, neginf=-700.0)
        ll = logsumexp(stack, axis=0)
        resp = np.exp(stack - ll[None, :])
        return ll, (resp * (y[None, :] - n[None, :] * p)).sum(axis=0)

    def _pmf(self, n, eta, disp_row, extra, k, df):
        x, w = self._nodes()
        out = np.zeros((len(n), len(k)))
        for xq, wq in zip(x, w):
            p = _sigmoid(eta + disp_row * xq)
            out += wq * np.nan_to_num(binom.pmf(k[None, :], np.asarray(n)[:, None],
                                                p[:, None]))
        return out

    def shape_report(self, df: pd.DataFrame) -> dict:
        # Zero at both ends by construction: the logit-normal density carries a
        # `exp(-logit(p)^2 / 2sigma^2)` factor that beats every polynomial term.
        return {"diverges_at_one": 0.0, "diverges_at_zero": 0.0,
                "diverges_at_one_weighted": 0.0, "diverges_at_zero_weighted": 0.0,
                "sigma_u": ";".join(f"{s:.3f}" for s in self.dispersion)}


class BetaRectangularFrailty(FrailtyGLM):
    """`theta * U(0, 1) + (1 - theta) * Beta(a, b)` as the frailty — the one-parameter control.

    `U(0, 1)` is `Beta(1, 1)`, and a binomial mixed over it is the **discrete uniform** on
    `0..n`, so the marginal is `theta / (n + 1) + (1 - theta) * BetaBinom`, exactly. That is
    why this arm is the control: it adds tail mass at both ends *symmetrically* and with one
    parameter, so it can only trade the two boundaries against the body, never against each
    other. If it matches an arm that costs eleven parameters, the eleven bought nothing.

    `theta = 0` is a bounded parameter value, so the nesting is exact.
    """

    name = "beta_rect"

    def _extra0(self) -> np.ndarray:
        return np.zeros(1)

    def _extra_bounds(self):
        return [(0.0, 1.0)]

    def _fit_starts(self):
        return [np.zeros(1), np.array([0.05]), np.array([0.20])]

    def _terms(self, y, n, eta, disp_row, extra, df):
        theta = float(np.clip(extra[0], 0.0, 1.0))
        mu = _sigmoid(eta)
        ll_bb = _bb_logpmf(y, n, mu, disp_row)
        stack = np.vstack([np.log(max(1.0 - theta, 1e-300)) + ll_bb,
                           np.full_like(ll_bb, np.log(max(theta, 1e-300)))
                           - np.log(n + 1.0)])
        ll = logsumexp(stack, axis=0)
        resp = np.exp(stack[0] - ll)
        return ll, resp * _bb_dlogpmf_dmu(y, n, mu, disp_row) * mu * (1.0 - mu)

    def _pmf(self, n, eta, disp_row, extra, k, df):
        theta = float(np.clip(extra[0], 0.0, 1.0))
        bb = _bb_pmf_grid(n, _sigmoid(eta), disp_row, k)
        flat = (k[None, :] <= np.asarray(n)[:, None]) / (np.asarray(n)[:, None] + 1.0)
        return (1.0 - theta) * bb + theta * flat

    def shape_report(self, df: pd.DataFrame) -> dict:
        mu = _sigmoid(self._design(df) @ self.beta)
        a, b = _ab_row(mu, self._disp_row(df, self.dispersion))
        # The uniform component is finite at both ends, so only the Beta one can diverge.
        theta = float(np.clip(self.extra[0], 0.0, 1.0))
        return {"diverges_at_one": float((b < 1).mean()),
                "diverges_at_zero": float((a < 1).mean()),
                "diverges_at_one_weighted": float((1 - theta) * (b < 1).mean()),
                "diverges_at_zero_weighted": float((1 - theta) * (a < 1).mean()),
                "theta": theta,
                "rho": ";".join(f"{r:.4f}" for r in self.dispersion)}


#: The fourth axis. `betabinom` is the reference row; every other arm is asserted to
#: reproduce it (or, for `logitnormal`, the binomial) at its nesting parameter values.
LIKELIHOODS: dict[str, callable] = {
    "betabinom": BetaBinomFrailty,
    "mixture": MixtureFrailty,
    "finite_mix": lambda **kw: FiniteMixtureFrailty(k=FINITE_MIX_K, **kw),
    "logitnormal": LogitNormalFrailty,
    "beta_rect": BetaRectangularFrailty,
}

#: The likelihood axis holds the other three here — the arm that ships.
LIKELIHOOD_WINDOW = "three_point_era"
LIKELIHOOD_REFERENCE = "betabinom"


def assert_nests(arm: FrailtyGLM, train: pd.DataFrame, tol: float = 1e-8) -> float:
    """Every arm reproduces the head it extends, at its own nesting parameter values.

    Rule 2 of the axis, and the rollback path: `pi = 0`, `g = 0`, `theta = 0` and
    `sigma_u = 0` are each a *finite, attainable* parameter value in the arms above, so this
    is an equality rather than a limit. Returns the absolute log-likelihood gap.

    `logitnormal` is checked against the **binomial** rather than the beta-binomial, because
    it does not nest the incumbent and is in the ladder for that reason.
    """
    y = train["gp"].to_numpy(dtype=float)
    n = train["team_games"].to_numpy(dtype=float)
    disp_row = arm._disp_row(train, arm._nesting_disp())
    eta = arm._design(train) @ arm.incumbent_beta
    got, _ = arm._terms(y, n, eta, disp_row, arm._extra0(), train)
    if isinstance(arm, LogitNormalFrailty):
        want = binom.logpmf(y, n, _sigmoid(eta))
    else:
        want = _bb_logpmf(y, n, _sigmoid(eta), disp_row)
    gap = abs(float(got.sum()) - float(want.sum()))
    if not gap < tol:
        raise AssertionError(
            f"{arm.name} does not reproduce its nesting point: log-likelihood "
            f"{got.sum():.10f} against {want.sum():.10f} (gap {gap:.3e} > {tol:.0e}). "
            f"An arm that cannot reproduce the head it extends is a different model.")
    return gap


def tenure_decomposition_pmf(val: pd.DataFrame, max_games: int,
                             path: Path) -> np.ndarray | None:
    """The games-played head's composite pmf, aligned onto the availability rows.

    **The free arm.** `stan_games_played` is the structural alternative to a frailty — entry
    index x exit index x a within-tenure two-state chain with beta-geometric spells, built
    because "a departure is an absorbing hitting time, not a low recovery rate" — and it had
    never been compared on this statistic, because Gate D was CRPS-shaped. The pmf is
    already on disk, so the comparison costs a file read.

    `duration_covariates` is the arm that head *selects*; `within_tenure` is flagged
    `oracle_tenure` and covers 751 rows rather than 883, so the 7.2265 that
    `docs/availability-window-plan.md` §5.3 quotes is not a forecast and is not a candidate.

    Returns `None` when the artifact is absent, so this ladder does not require
    `make stan-games-played` to have been run.
    """
    if not path.exists():
        print(f"  no {path.name} on disk — skipping the tenure-decomposition arm")
        return None
    gp = pd.read_csv(path)
    gp = gp[gp["arm"] == "duration_covariates"]
    wide = gp.pivot_table(index=["season", "player_id"], columns="gp", values="p",
                          fill_value=0.0)
    keys = pd.MultiIndex.from_arrays([val["season"], val["player_id"]],
                                     names=["season", "player_id"])
    have = wide.reindex(keys)
    if have.isna().all(axis=1).any():
        missing = int(have.isna().all(axis=1).sum())
        print(f"  tenure pmf covers {len(val) - missing:,}/{len(val):,} rows — skipping")
        return None
    pmf = np.zeros((len(val), max_games + 1))
    cols = [c for c in wide.columns if int(c) <= max_games]
    pmf[:, [int(c) for c in cols]] = np.nan_to_num(have[cols].to_numpy(dtype=float))
    return pmf


def likelihood_ladder(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
                      l2: float = 1.0, seed: int = 42,
                      gp_pmf_path: Path | None = None) -> pd.DataFrame:
    """The fourth axis, with window / season term / dispersion held at the shipped arm."""
    cut = restrict_window(train, WINDOWS[LIKELIHOOD_WINDOW])
    y_val = val["gp"].to_numpy()
    n_val = val["team_games"].to_numpy()
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}
    tails: dict[str, dict[str, np.ndarray]] = {}

    # §4's arm exactly as §4 fitted it, carried as a non-selectable row. Every arm below is
    # a JOINT MLE of its own likelihood, while `RoleGradedBetaBinomial` fits `beta` under a
    # shared `rho` and then profiles `rho` per bucket holding the mean fixed. Scoring a
    # profiled reference against joint alternatives would confound the likelihood with the
    # estimator, so the reference is refitted jointly — and this row is what says how much
    # that alone is worth.
    two_stage = RoleGradedBetaBinomial(l2=l2, features=list(FEATURE_COLS)).fit(cut)
    row, scores = score_arm("betabinom_two_stage", two_stage, cut, val,
                            list(FEATURE_COLS), max_games, seed)
    row.update({"likelihood": "betabinom", "family": "betabinom", "selectable": False,
                "dispersion_label": "rho", "estimator": "two_stage"})
    rows.append(row)
    per_row["betabinom_two_stage"] = scores
    tails["betabinom_two_stage"] = _tail_parts(two_stage.predict_pmf(val, max_games),
                                               y_val, n_val)

    builders = dict(LIKELIHOODS)
    for k in FINITE_MIX_K_SENSITIVITY:
        builders[f"finite_mix_k{k}"] = (lambda k=k, **kw: FiniteMixtureFrailty(k=k, **kw))

    for name, build in builders.items():
        model = build(l2=l2, features=list(FEATURE_COLS)).fit(cut)
        gap = assert_nests(model, cut)
        row, scores = score_arm(name, model, cut, val, list(FEATURE_COLS), max_games, seed)
        row.update({"likelihood": name, "n_params": model.n_params,
                    "nesting_loglik_gap": gap,
                    "train_loglik": model.train_loglik,
                    "train_loglik_at_nesting": model.nesting_loglik,
                    "train_loglik_incumbent": model.incumbent_loglik,
                    "dispersion_label": model.dispersion_label,
                    "family": "finite_mix" if name.startswith("finite_mix") else name,
                    "selectable": name in LIKELIHOODS, "estimator": "joint",
                    "n_starts": model.n_starts, "start_loglik_spread": model.start_spread})
        row.update({f"shape_{k}": v for k, v in model.shape_report(val).items()})
        rows.append(row)
        per_row[name] = scores
        tails[name] = _tail_parts(model.predict_pmf(val, max_games), y_val, n_val)
        print(f"  {name:<16} {model.n_params:>3} params  CRPS {row['val_crps']:.4f}  "
              f"PIT {row['val_pit_ks']:.4f}  boundary {row['boundary_tail_error']:.4f}  "
              f"body {row['body_error']:.4f}  "
              f"diverges@1 {row['shape_diverges_at_one_weighted']:.1%} (mass)")

    if gp_pmf_path is not None:
        pmf = tenure_decomposition_pmf(val, max_games, gp_pmf_path)
        if pmf is not None:
            row, scores = _score_pmf("tenure_decomposition", pmf, val, max_games, seed)
            row.update({"likelihood": "tenure_decomposition", "family": "structural",
                        "selectable": False, "dispersion_label": "n/a",
                        # Fitted by `stan_games_played` on its own rows, not by this
                        # ladder — so the fitting-half columns describe nothing here.
                        "n_train": np.nan, "n_train_seasons": np.nan,
                        "first_train_season": ""})
            rows.append(row)
            per_row["tenure_decomposition"] = scores
            tails["tenure_decomposition"] = _tail_parts(pmf, y_val, n_val)
            print(f"  {'tenure_decomp':<16} {'—':>3}         "
                  f"CRPS {row['val_crps']:.4f}  PIT {row['val_pit_ks']:.4f}  "
                  f"boundary {row['boundary_tail_error']:.4f}  "
                  f"body {row['body_error']:.4f}")

    reference = per_row[LIKELIHOOD_REFERENCE]
    for row in rows:
        d, lo, hi = paired_bootstrap(per_row[row["arm"]], reference, seed=seed)
        row["crps_vs_betabinom"] = d
        row["crps_vs_betabinom_lo"] = lo
        row["crps_vs_betabinom_hi"] = hi
        row["beats_betabinom"] = bool(hi < 0.0)
        # The selector gets an interval too. A boundary margin quoted bare is the thing
        # `docs/potential-to-dos.md` calls a prompt rather than a finding, and this axis is
        # decided on the boundary rather than on CRPS.
        row.update(bootstrap_tail_errors(
            tails[row["arm"]],
            None if row["arm"] == LIKELIHOOD_REFERENCE else tails[LIKELIHOOD_REFERENCE],
            seed=seed))
    return pd.DataFrame(rows).sort_values("val_crps").reset_index(drop=True)


def _tail_parts(pmf: np.ndarray, y: np.ndarray, n: np.ndarray) -> dict[str, np.ndarray]:
    """The per-row pieces the three summary errors are means of."""
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    n_int = np.asarray(n, dtype=int)
    missed = n_int - np.asarray(y, dtype=int)
    parts = {"p_full": pmf[np.arange(len(y)), n_int],
             "o_full": (y == n).astype(float)}
    for threshold in TAIL_BELOW:
        parts[f"p_{threshold}"] = cdf[:, threshold - 1]
        parts[f"o_{threshold}"] = (y < threshold).astype(float)
    parts["p_low_shoulder"] = cdf[:, 9] - pmf[:, 0]
    parts["o_low_shoulder"] = ((y >= 1) & (y < 10)).astype(float)
    parts["p_high_shoulder"] = (_p_missed_le(pmf, cdf, n_int, 11)
                                - _p_missed_le(pmf, cdf, n_int, 0))
    parts["o_high_shoulder"] = ((missed >= 1) & (missed < 12)).astype(float)
    return parts


def _tail_errors(parts: dict[str, np.ndarray],
                 idx: np.ndarray) -> tuple[float, float, float]:
    """`(boundary_tail_error, body_error, shoulder_error)` on one resample of the rows."""
    def err(key: str) -> float:
        return abs(parts[f"p_{key}"][idx].mean() - parts[f"o_{key}"][idx].mean())
    return ((err("10") + err("full")) / 2.0,
            (err("41") + err("60")) / 2.0,
            (err("low_shoulder") + err("high_shoulder")) / 2.0)


def bootstrap_tail_errors(parts: dict[str, np.ndarray],
                          reference: dict[str, np.ndarray] | None,
                          reps: int = BOOTSTRAP_REPS, seed: int = 42) -> dict:
    """Intervals on the selector, and on its difference against the reference arm.

    `boundary_tail_error` decides this axis, and §4b is the standing argument that a margin
    without an interval is a prompt rather than a finding. It is a **non-linear** statistic —
    two absolute values of differences of means — so the bootstrap resamples rows and
    recomputes it rather than resampling a per-row score, and it is paired inside the row so
    the between-player variance cancels the way it does for CRPS.
    """
    n_rows = len(parts["p_full"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_rows, size=(reps, n_rows))
    boundary = np.empty(reps)
    body = np.empty(reps)
    shoulder = np.empty(reps)
    d_boundary = np.empty(reps)
    d_shoulder = np.empty(reps)
    for r in range(reps):
        boundary[r], body[r], shoulder[r] = _tail_errors(parts, idx[r])
        if reference is not None:
            ref_b, _, ref_s = _tail_errors(reference, idx[r])
            d_boundary[r] = boundary[r] - ref_b
            d_shoulder[r] = shoulder[r] - ref_s
    out = {"boundary_tail_error_lo": float(np.percentile(boundary, 2.5)),
           "boundary_tail_error_hi": float(np.percentile(boundary, 97.5)),
           "body_error_lo": float(np.percentile(body, 2.5)),
           "body_error_hi": float(np.percentile(body, 97.5)),
           "shoulder_error_lo": float(np.percentile(shoulder, 2.5)),
           "shoulder_error_hi": float(np.percentile(shoulder, 97.5))}
    if reference is not None:
        out.update({"boundary_vs_betabinom": float(d_boundary.mean()),
                    "boundary_vs_betabinom_lo": float(np.percentile(d_boundary, 2.5)),
                    "boundary_vs_betabinom_hi": float(np.percentile(d_boundary, 97.5)),
                    "beats_betabinom_boundary": bool(np.percentile(d_boundary, 97.5) < 0.0),
                    "shoulder_vs_betabinom": float(d_shoulder.mean()),
                    "shoulder_vs_betabinom_lo": float(np.percentile(d_shoulder, 2.5)),
                    "shoulder_vs_betabinom_hi": float(np.percentile(d_shoulder, 97.5)),
                    "beats_betabinom_shoulder": bool(np.percentile(d_shoulder, 97.5) < 0.0)})
    return out


def _score_pmf(name: str, pmf: np.ndarray, val: pd.DataFrame, max_games: int,
               seed: int) -> tuple[dict, np.ndarray]:
    """`score_arm` for an arm that arrives as a pmf rather than as a fitted object.

    The tenure decomposition is fitted in Stan by another module; what this ladder can do
    with it is score its predictive on the same rows by the same code, which is the whole
    point of the free arm.
    """
    class _Fixed:
        rho = np.nan
        rho_spread = np.nan

        def predict_pmf(self, df, mg):
            return pmf

        def predict_mean(self, df):
            n = df["team_games"].to_numpy(dtype=float)
            k = np.arange(pmf.shape[1])
            return np.clip((pmf * k[None, :]).sum(axis=1) / np.maximum(n, 1),
                           EPS, 1 - EPS)

    return score_arm(name, _Fixed(), val, val, [], max_games, seed)


# The likelihood axis's rolling-origin confirmation runs at this lookback: §4b's interior
# CRPS optimum, and the closest fitting-half analogue of the 2012-13 window the validation
# ladder holds. An absolute first season means nothing at a 2011 origin.
LIKELIHOOD_LOOKBACK = 8


def likelihood_rolling(train: pd.DataFrame, max_games: int, l2: float = 1.0,
                       seed: int = 42) -> pd.DataFrame:
    """The fourth axis on §4b's harness — 13 origins, fitting half only.

    Same instrument, same reason: 8 arms on 883 validation rows is a multiplicity problem,
    and §5b is the standing evidence that an arm can win one of these two readings and lose
    the other. Nothing here touches a validation or held-out row.
    """
    years = season_start_year(train)
    origins = [int(y) for y in np.unique(years) if y >= FIRST_ORIGIN]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        score = train[years == origin]
        fit_rows = train[(years < origin) & (years >= origin - LIKELIHOOD_LOOKBACK)]
        if score.empty or len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
            continue
        for name, build in LIKELIHOODS.items():
            model = build(l2=l2, features=list(FEATURE_COLS)).fit(fit_rows)
            assert_nests(model, fit_rows)
            scored = _origin_scores(model, score, max_games)
            scored["origin"] = np.full(len(score), origin)
            slot = per_arm.setdefault(name, {})
            for key, values in scored.items():
                slot.setdefault(key, []).append(values)
        print(f"  origin {origin}: {len(fit_rows):,} fit / {len(score):,} scored")

    pooled = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in per_arm.items()}
    reference = pooled[LIKELIHOOD_REFERENCE]["crps"]

    def parts_of(d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        missed = d["n"].astype(int) - d["y"].astype(int)
        out = {"p_full": d["p_full"], "o_full": (d["y"] == d["n"]).astype(float),
               "p_low_shoulder": d["p_band_low_shoulder"],
               "o_low_shoulder": ((d["y"] >= 1) & (d["y"] < 10)).astype(float),
               "p_high_shoulder": d["p_band_high_shoulder"],
               "o_high_shoulder": ((missed >= 1) & (missed < 12)).astype(float)}
        for threshold in TAIL_BELOW:
            out[f"p_{threshold}"] = d[f"p_below_{threshold}"]
            out[f"o_{threshold}"] = (d["y"] < threshold).astype(float)
        return out

    ref_parts = parts_of(pooled[LIKELIHOOD_REFERENCE])
    grid = np.linspace(0, 1, 101)
    rows = []
    for name, d in pooled.items():
        y, n, org = d["y"], d["n"], d["origin"]
        mean, lo, hi = paired_bootstrap(d["crps"], reference, seed=seed)
        wins = sum(1 for o in np.unique(org)
                   if d["crps"][org == o].mean() < reference[org == o].mean())
        u = d["pit"]
        row = {"arm": name, "n_origins": int(len(np.unique(org))), "n_scored": int(len(y)),
               "crps": float(d["crps"].mean()), "crps_vs_betabinom": mean,
               "crps_vs_betabinom_lo": lo, "crps_vs_betabinom_hi": hi,
               "origins_won": wins,
               "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u)
                                             - grid))),
               "err_full_schedule": float(d["p_full"].mean() - (y == n).mean())}
        for threshold in TAIL_BELOW:
            row[f"err_below_{threshold}"] = float(d[f"p_below_{threshold}"].mean()
                                                  - (y < threshold).mean())
        row["boundary_tail_error"] = (abs(row["err_below_10"])
                                      + abs(row["err_full_schedule"])) / 2
        row["body_error"] = (abs(row["err_below_41"]) + abs(row["err_below_60"])) / 2
        # The shoulders and the exact point masses, on the same footing as the other two —
        # the boundary pair is asymmetric (a ten-game low shoulder against a single upper
        # point) and 60-81 games was measured by nothing before 2026-08-11.
        missed = d["n"].astype(int) - d["y"].astype(int)
        for band, predicted, observed in (
                ("zero", d["p_zero"], (d["y"] == 0)),
                ("low_shoulder", d["p_band_low_shoulder"],
                 (d["y"] >= 1) & (d["y"] < 10)),
                ("high_shoulder", d["p_band_high_shoulder"],
                 (missed >= 1) & (missed < 12))):
            row[f"pred_band_{band}"] = float(predicted.mean())
            row[f"obs_band_{band}"] = float(observed.mean())
            row[f"err_band_{band}"] = row[f"pred_band_{band}"] - row[f"obs_band_{band}"]
        for m in TAIL_MISSED:
            row[f"pred_missed_le_{m}"] = float(d[f"p_missed_le_{m}"].mean())
            row[f"obs_missed_le_{m}"] = float((missed <= m).mean())
            row[f"err_missed_le_{m}"] = (row[f"pred_missed_le_{m}"]
                                         - row[f"obs_missed_le_{m}"])
        row["shoulder_error"] = (abs(row["err_band_low_shoulder"])
                                 + abs(row["err_band_high_shoulder"])) / 2
        # The selector carries an interval here too. The two readings can disagree — §5b
        # is the standing evidence — and a disagreement is only readable if both sides of
        # it are quoted with their uncertainty.
        row.update(bootstrap_tail_errors(
            parts_of(d), None if name == LIKELIHOOD_REFERENCE else ref_parts, seed=seed))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", 42))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    panel = build_panel(seasons, raw_dir)
    frame = season_availability(panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(panel))
    train, val = selection_split(design)
    max_games = int(design["team_games"].max())

    print(f"Availability window ladder: {len(train):,} train / {len(val):,} validation "
          f"({', '.join(sorted(val['season'].unique()))})")
    print("  The held-out split is LOCKED (src/models/held_out.py) and is never "
          "materialized here.\n  Every window restricts the FITTING half only.")
    table = ladder(train, val, max_games, l2=l2, seed=seed)

    dest = out_dir / "availability_window.csv"
    table.to_csv(dest, index=False)
    print(f"\nWrote {len(table):,} arms → {dest}")

    print("\nRolling-origin confirmation on the FITTING HALF ONLY "
          f"(origins from {FIRST_ORIGIN}, lookbacks {LOOKBACKS}):")
    rolling = rolling_confirmation(train, max_games, l2=l2, seed=seed)
    roll_dest = out_dir / "availability_window_rolling.csv"
    rolling.to_csv(roll_dest, index=False)
    print(f"\nWrote {len(rolling):,} arms × "
          f"{int(rolling['n_scored'].max()):,} scored rows → {roll_dest}")

    print(f"\nThe fourth axis — the LIKELIHOOD, with window / season term / dispersion "
          f"held at\n  the shipped arm ({LIKELIHOOD_WINDOW}, none, role). "
          f"`l2` is pinned at {l2} across arms, which is\n  NOT neutral: the mixture arms "
          f"carry unpenalized parameters the incumbent does not.")
    likelihood = likelihood_ladder(train, val, max_games, l2=l2, seed=seed,
                                   gp_pmf_path=out_dir / "stan_games_played_gp_pmf.csv")
    lik_dest = out_dir / "availability_likelihood.csv"
    likelihood.to_csv(lik_dest, index=False)
    print(f"\nWrote {len(likelihood):,} arms → {lik_dest}")

    print(f"\nRolling-origin confirmation of the likelihood axis "
          f"(lookback {LIKELIHOOD_LOOKBACK}, fitting half only):")
    lik_rolling = likelihood_rolling(train, max_games, l2=l2, seed=seed)
    lik_roll_dest = out_dir / "availability_likelihood_rolling.csv"
    lik_rolling.to_csv(lik_roll_dest, index=False)
    print(f"\nWrote {len(lik_rolling):,} arms × "
          f"{int(lik_rolling['n_scored'].max()):,} scored rows → {lik_roll_dest}")

    return {"availability_window": dest, "availability_window_rolling": roll_dest,
            "availability_likelihood": lik_dest,
            "availability_likelihood_rolling": lik_roll_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
