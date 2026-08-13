"""P3 — the preseason increment on the MARGINAL minutes head, as a nested arm.

`make minutes-preseason`. Three artifacts in `outputs/predictions/`:
`minutes_preseason.csv`, `minutes_preseason_rolling.csv`,
`minutes_preseason_shrinkage.csv`.

## Why this head, and why first

`docs/preseason-plan.md` P1 measured the preseason block's incremental out-of-sample R² on
the **draftable** population — every player on a season-start roster — and the two candidate
heads swapped places under that restriction: `gp_share` fell 6x to +0.0198 while
`minutes_per_game` *rose* 2.3x to +0.0492. So the plan's a-priori ordering was inverted by
its own gate and minutes runs first. One column carries almost all of it: P1's `pre_d_mpg`
alone scores **+0.0519**, more than the seven-column block it sits in, at partial r 0.334.
Every other column in that block is worth <= +0.0028 alone.

That is a screen on a point estimate under a ridge. This module is the same question asked
at the unit the head actually ships on — CRPS of the season-total minutes predictive — on
the `minutes_window` point MLE, which needs no CmdStan and so can reject an arm before
anyone spends sampler time. Nothing here ships a head; an arm that wins earns a Stan port,
exactly as `docs/minutes-window-plan.md` §5 puts it.

## The bar, stated before the run

An arm ships onward only if **both** readings agree:

1. **Validation.** Its CRPS paired-bootstrap interval against the covered-window incumbent
   lies entirely below zero, at the head's own season-total unit, on the draftable
   population.
2. **The rolling-origin harness on the fitting half.** Same sign, interval below zero, and a
   majority of origins won.

Validation alone ships nothing, and that is not caution for its own sake:
`docs/availability-window-plan.md` §12e and §14f record two blocks that won a validation
reading and shrank 4-6x on the rolling harness, and `docs/minutes-window-plan.md` §3 records
this head's own fitting-window axis doing the same thing — -2.687 on validation, -0.079
rolling, 6 of 13 origins.

`predictive_sd` rides beside CRPS in every row and is **reported rather than barred**. It is
the quantity `make minutes-unification` ships this head for (season-total predictive sd
302.75 against the composition's 64.65), so an arm that improves CRPS by narrowing the one
thing this head exists to supply is not obviously an improvement — the same warning
`minutes_window.score_arm` carries, one axis over.

## The coverage restriction is load-bearing here, unlike on availability

The preseason panel begins at 2004-05 (2003-04 is `tail_missing` and out of scope), and this
head fits from **1997-98**. That is 2,154 of 8,306 training rows with no preseason row for a
reason that is a fact about the NBA's API rather than about the player — and the
missing-preseason indicator would read as an era dummy on every one of them.
`docs/preseason-plan.md`'s "coverage interactions" risk names exactly this case and says the
availability head's 2012-13 window dodges it. This head does not.

So **every arm, including the reference, fits the covered window only**, and the first
covered season is read off `preseason_coverage.csv` rather than hard-coded. The
full-window incumbent is scored beside it as a context row, so the restriction's own cost
is visible and cannot be silently credited to — or charged against — the preseason block.

## Difference coding on this head's own link

`docs/preseason-plan.md`'s design decision is that a delta is taken **on the model's own
link scale**. This head is a beta-binomial on season minutes out of season game-length, so
its link is a logit on the per-game minutes *share*, and the canonical column is

    pre_d_logit_share = logit(mpg_pre / 48) - logit(minutes_share_lag1)

which is P1's `pre_d_mpg` moved onto the scale the coefficient will actually live on. It is
the **primary** arm, declared before the run; P1's own `log1p`-MPG column rides as a
sensitivity arm so the gate's finding is re-read in this head's unit rather than inherited
from a ridge probe on a different one.

The 48 is the one approximation in the block: the panel carries no preseason game lengths,
so a preseason overtime inflates that row's share by ~10%. It is a *delta* against a prior
share whose denominator is realized length, so a constant part of the bias lands in the
intercept, and the residual is far below the delta's own sampling noise at 4-6 games.

**Zero recovers the incumbent exactly.** A player whose preseason agrees with his prior
season carries a zero delta, a player with no preseason row carries zero deltas plus his
indicator, and the coefficient path through zero is the shipped head — the house nesting
discipline (`pi = 0`, `K = 1`, `U_n = 0`, `n_rho = 1`), pinned by a test.

**The indicator is split by age, not carried whole.** P1 decision 3: on the draftable frame
a 32-plus player with no preseason row loses 0.247 of the season and 4.93 MPG, while a
24-to-27 player with no preseason row loses 0.34 MPG — a rested veteran and an injured star
are the same indicator and different events. Four `pre_missing__<age>` columns, one per P1's
own age cells, coded on the **missing** side so they are sparse (3.8% of draftable rows) and
so zero still means "the incumbent".

## The shrinkage constant the plan left open

`docs/preseason-plan.md` leaves the volume question open between an empirical-Bayes shrink
and a reliability interaction, and says the gate decides on train. P1 shipped the cheapest
form — an additive `pre_log_min` — without ruling on it. `fit_shrinkage` closes it here, on
an inner carve of the **fitting half** (the last two training seasons scored, the rest
fitted), never on validation: the delta is multiplied by `min_pre / (min_pre + k)` over a
grid of `k`, and the selected `k` is a fitted quantity estimated where every other fitted
quantity in this project is.

Usage:
    python -m src.models.minutes_preseason
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year
from src.eda.preseason_value import (AGE_EDGES, AGE_LABELS, EPS, attach_availability_block,
                                     attach_prior_shares, attach_season_start_roster,
                                     covered_seasons, logit)
from src.models.availability import split_seasons
from src.models.availability_window import restrict_window
from src.models.held_out import as_plain, selection_split
from src.models.minutes_unification import paired_bootstrap, verdict
from src.models.minutes_window import (BOOTSTRAP_REPS, VARIANT, PointMinutes,
                                       assert_shipped_variant, score_arm)
from src.models.season_terms import season_start_year
from src.models.stan_minutes import OWN, SPLINE_KNOTS, FloorMinutes, variants
from src.models.stan_minutes import build_design as minutes_build_design
from src.models.stan_utils import crps_from_samples, ks_uniform, pit_from_samples

# Regulation game length in minutes, the denominator that puts a preseason MPG on the same
# share scale as `minutes_share_lag1`. The panel carries no preseason game lengths — see the
# module docstring for why the approximation is affordable and where its bias lands.
PRESEASON_GAME_LENGTH = 48.0

# The head's own link-scale delta, and the two derived forms of it. `OWN_DELTA` is the
# primary; `SHRUNK_DELTA` is it multiplied by the fitted reliability weight; `MPG_DELTA` is
# P1's column verbatim, carried as the sensitivity that re-reads the gate in this unit.
OWN_DELTA = "pre_d_logit_share"
SHRUNK_DELTA = "pre_d_logit_share_shrunk"
CENTERED_DELTA = "pre_d_logit_share_centered"
MPG_DELTA = "pre_d_mpg"
SHARE_LATE_DELTA = "pre_d_min_share_late"

# The additive reliability term P1 shipped, and the columns of its seven-column block that
# are not the delta itself. Kept as a named list so the `p1_block` arm is P1's block and not
# a paraphrase of it.
RELIABILITY = "pre_log_min"
P1_PARTICIPATION = ["pre_d_min_share_late", "pre_gp_share", "pre_missed_tail_share",
                    "pre_played_final_game"]

# The missing-preseason indicator, split on P1's own age cells (decision 3). Coded on the
# MISSING side: 3.8% of draftable rows, and zero across the block still means "incumbent".
MISSING_AGE_COLS = [f"pre_missing__{label}" for label in AGE_LABELS]

# Pseudo-minutes for the empirical-Bayes weight `min_pre / (min_pre + k)`. 0 is no shrinkage
# — the weight is identically 1 and the arm is `own_delta` exactly, which is what makes the
# grid's left edge a control rather than a guess. The upper edge is well past a full
# preseason (~140 minutes for a starter), so the grid brackets its own optimum or says the
# delta should be shrunk away entirely.
SHRINKAGE_GRID: tuple[float, ...] = (0.0, 20.0, 40.0, 80.0, 160.0, 320.0)

# Trailing TRAINING seasons scored rather than fitted when `k` is selected. Two, matching
# `preseason_value.INNER_SCORE_SEASONS` and the project's own split arity — and never the
# validation seasons, which the shrinkage constant must not be allowed to read.
INNER_SCORE_SEASONS = 2

# The reference every interval is taken against: the shipped head's variant and dispersion,
# refit on the covered window, with no preseason column at all.
REFERENCE_ARM = "incumbent"

# Declared before the run. The gate's verdict is read off this arm; everything else on the
# ladder is a sensitivity or an attribution.
PRIMARY_ARM = "own_delta"

# The population every verdict is read on — P1 decision 5. A figure quoted on the head's
# whole validation frame is a population statement rather than a model one, so both are
# written and the draftable one decides.
DECISION_POPULATION = "draftable"

# Rolling-origin harness. `FIRST_ORIGIN` matches `minutes_window.FIRST_ORIGIN` so the two
# confirmations are read side by side; with coverage starting at 2004-05 it leaves five
# seasons of fitting rows at the earliest origin. `MIN_FIT_ROWS` matches that module's
# `MIN_ROLE_ROWS * len(ROLE_LABELS)` floor.
FIRST_ORIGIN = 2009
MIN_FIT_ROWS = 400

SEED = 0


# ── The block ─────────────────────────────────────────────────────────────────

def preseason_arms(shrunk: bool = True) -> dict[str, list[str]]:
    """`arm name -> the preseason columns it adds to the shipped feature list`.

    Every arm is the shipped variant **plus** these columns, so each is a nested increment
    on the incumbent rather than a different head, and the empty list is the incumbent
    itself. The ladder is deliberately short: P1 already screened the columns, and this
    module's job is to price the survivor at the head's own unit rather than to run a second
    feature selection on the selection split.
    """
    arms: dict[str, list[str]] = {
        REFERENCE_ARM: [],
        # PRIMARY — the delta on this head's own link, plus the age-split indicator.
        PRIMARY_ARM: [OWN_DELTA] + MISSING_AGE_COLS,
        # P1's additive reliability term, the form the gate shipped without ruling on it.
        "own_delta_reliability": [OWN_DELTA, RELIABILITY] + MISSING_AGE_COLS,
        # P1's seven-column block, with `has_preseason` replaced by its age split per
        # decision 3. On the gate's own reading this should not beat the primary — the block
        # scored +0.0492 against the single column's +0.0519 — and an arm that exists to be
        # beaten is how that reading gets checked rather than assumed.
        "p1_block": ([OWN_DELTA, MPG_DELTA, RELIABILITY] + P1_PARTICIPATION
                     + MISSING_AGE_COLS),
        # SENSITIVITY — P1's column verbatim, on `log1p` MPG rather than the head's logit
        # share. The two are near-duplicates by construction; a disagreement between them is
        # a statement about the link, not about the preseason.
        "mpg_scale": [MPG_DELTA] + MISSING_AGE_COLS,
        # ATTRIBUTION — the indicator alone. If this carries the primary arm's gain then the
        # finding is "a missing preseason is a signal", not "the preseason delta is", which
        # is the split P1's own attribution columns exist to catch one family over.
        "missing_only": list(MISSING_AGE_COLS),
        # ATTRIBUTION — the delta with each season's own mean removed. Preseason minutes are
        # COMPRESSED (`docs/preseason-plan.md`: starters play 15-20 minutes), so the delta's
        # mean is far from zero and varies by season, and a coefficient on it is partly a
        # league-level shift the point head has no season term to absorb. Centering leaves
        # only the cross-player part. If the gain survives, the block is a player-specific
        # update; if it collapses, it was a level the intercept should have owned. The
        # centering constant is each season's OWN preseason mean, which is on disk before
        # the opener, so this stays point-in-time on a validation season.
        "own_delta_centered": [CENTERED_DELTA] + MISSING_AGE_COLS,
        # ATTRIBUTION — the within-team late-preseason share, which is the unit P0 argued
        # for precisely because a share divides the compression out. It is inside
        # `p1_block`; alone it says whether the compression-free unit can carry the block on
        # its own.
        "share_late_only": [SHARE_LATE_DELTA] + MISSING_AGE_COLS,
    }
    if shrunk:
        arms["own_delta_shrunk"] = [SHRUNK_DELTA] + MISSING_AGE_COLS
    return arms


def reliability_weight(frame: pd.DataFrame, k: float) -> np.ndarray:
    """`min_pre / (min_pre + k)` — the empirical-Bayes weight on a preseason delta.

    A delta over 60 preseason minutes is noisier than one over 140, and this is the shrink
    `docs/preseason-plan.md` leaves open against the additive-reliability alternative. `k`
    is in the same units as `min_pre`, so it reads directly as "how many preseason minutes
    before the delta is believed half way".

    `k = 0` is the weight 1 everywhere, which makes the un-shrunk arm the grid's own left
    edge rather than a separate model.
    """
    minutes = np.nan_to_num(frame["min_pre"].to_numpy(dtype=float), nan=0.0)
    if k <= 0:
        return np.ones(len(frame))
    return minutes / (minutes + float(k))


def attach_preseason(design: pd.DataFrame, panel: pd.DataFrame, av_panel: pd.DataFrame,
                     seasons: list[str]) -> pd.DataFrame:
    """Every preseason column this ladder can read, merged onto the minutes design.

    **Opt-in, and outside `stan_minutes.build_design` deliberately** — the
    `availability.attach_absence_mix` precedent, for the identical reason. That builder is
    the route every head in the chain takes to its rows, and a column that is structurally
    zero before 2004-05 must not be able to enter the composition head, the games-played
    spell process or the simulator by accident.

    The P1 delta builders are **reused rather than forked**, as `docs/preseason-plan.md`
    specifies: `attach_prior_shares` supplies the within-team share the late-share delta is
    taken against, and `attach_availability_block` supplies P1's seven columns on P1's own
    scales. What this function adds is the two things P1 had no reason to build — the delta
    on *this head's* link, and the age-split indicator P1's census decided on.
    """
    out = attach_prior_shares(design, av_panel, seasons)
    return add_own_columns(attach_availability_block(out, panel))


def add_own_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The three columns P1 had no reason to build, on a frame P1's attach step has run over.

    Split out of `attach_preseason` so the tests exercise this function rather than a copy of
    it — the failure mode `docs/minutes-window-plan.md` §1 names one module over, where a
    population rebuilt by hand picked the wrong window and nothing raised.

    Requires `has_preseason`, `mpg_pre` and `min_pre` from
    `preseason_value.attach_availability_block`, and `OWN` and `age` from the head's own
    design.
    """
    out = frame.copy()
    present = out["has_preseason"].to_numpy(dtype=float) > 0
    share = np.clip(out["mpg_pre"].to_numpy(dtype=float) / PRESEASON_GAME_LENGTH,
                    EPS, 1.0 - EPS)
    # `OWN` is `logit(minutes_share_lag1)`, built by `stan_minutes.build_design` — the same
    # quantity on the same scale, so the difference is the preseason's own contribution and
    # not two definitions of a minutes share.
    delta = logit(share) - out[OWN].to_numpy(dtype=float)
    out[OWN_DELTA] = np.where(present, np.nan_to_num(delta, nan=0.0), 0.0)

    # The same delta with each season's own mean removed, over the rows that actually have a
    # preseason — the zeros are a fill and averaging them in would shrink the centring by
    # the missing rate. A missing row keeps its zero, so the nesting argument still holds.
    seasons_present = out["season"].where(present)
    means = out[OWN_DELTA].where(present).groupby(seasons_present).transform("mean")
    out[CENTERED_DELTA] = np.where(present,
                                   out[OWN_DELTA] - means.fillna(0.0).to_numpy(float), 0.0)

    # Needed as a level by the reliability weight, and zero is the value the nesting
    # argument wants for a row with no preseason.
    out["min_pre"] = out["min_pre"].fillna(0.0)

    cells = pd.cut(out["age"].to_numpy(dtype=float), AGE_EDGES,
                   labels=AGE_LABELS).astype(str)
    for label in AGE_LABELS:
        out[f"pre_missing__{label}"] = ((~present) & (cells == label)).astype(float)
    return out


def with_shrunk_delta(frame: pd.DataFrame, k: float) -> pd.DataFrame:
    """`SHRUNK_DELTA` written onto a copy, at a given shrinkage constant."""
    out = frame.copy()
    out[SHRUNK_DELTA] = out[OWN_DELTA].to_numpy(dtype=float) * reliability_weight(frame, k)
    return out


# ── The shrinkage constant, fitted on the fitting half ────────────────────────

def fit_shrinkage(train: pd.DataFrame, grid: tuple[float, ...] = SHRINKAGE_GRID,
                  n_knots: int = SPLINE_KNOTS, variant: str = VARIANT,
                  inner: int = INNER_SCORE_SEASONS, seed: int = SEED
                  ) -> tuple[float, pd.DataFrame]:
    """`(selected k, the grid)` — the reliability shrink, chosen inside the fitting half.

    The inner carve nests `split_seasons` inside the *training* frame, which is the case
    `held_out.as_plain` exists for: the right-hand side is a training season and has to be
    readable, and validation is never materialized. Anything fitted from data is fitted on
    the fitting half — the rule `docs/preseason-plan.md` states for this constant by name.

    Selection is on CRPS at the head's own unit rather than on an R², because that is what
    the ladder downstream is scored on and a constant chosen against a different metric
    would be optimizing something the gate does not read.
    """
    fit, score = split_seasons(train, inner)
    fit, score = fit.reset_index(drop=True), as_plain(score).reset_index(drop=True)

    rows = []
    for k in grid:
        tr, sc, features = variants(with_shrunk_delta(fit, k),
                                    with_shrunk_delta(score, k), n_knots)[variant]
        model = PointMinutes(features + [SHRUNK_DELTA] + MISSING_AGE_COLS).fit(tr)
        samples = model.predict_samples(sc, seed)
        y = sc["successes"].to_numpy(dtype=float)
        rows.append({"k": float(k), "n_fit": len(tr), "n_score": len(sc),
                     "mean_weight": float(reliability_weight(sc, k).mean()),
                     "inner_crps": float(crps_from_samples(samples, y).mean())})
    table = pd.DataFrame(rows)
    return float(table.loc[table["inner_crps"].idxmin(), "k"]), table


# ── The ladder ────────────────────────────────────────────────────────────────

class FittedArm:
    """A fitted arm bound to the validation frame its own spline basis was built on.

    `minutes_window.FittedArm` with one addition: this ladder scores the same arm on two
    populations, so the frame it carries is the full validation frame and a caller selects
    rows out of it by position. Pairing one arm's coefficients with another's basis is the
    failure both classes exist to make impossible.
    """

    def __init__(self, name: str, model: PointMinutes, train: pd.DataFrame,
                 frame: pd.DataFrame, extra: list[str]):
        self.name, self.model, self.train, self.frame = name, model, train, frame
        self.extra = extra


def fit_arm(train: pd.DataFrame, val: pd.DataFrame, name: str, extra: list[str],
            n_knots: int = SPLINE_KNOTS, variant: str = VARIANT) -> FittedArm:
    """One nested arm: the shipped variant's features plus `extra`.

    The spline basis is fitted on this arm's own fitting rows, which for every arm here are
    the same covered-window rows — so unlike `minutes_window.fit_arm` the basis does not
    move between arms, and the only thing that differs is the column list.
    """
    tr, va, features = variants(train, val, n_knots)[variant]
    return FittedArm(name, PointMinutes(features + extra).fit(tr), tr, va, extra)


def ladder(train: pd.DataFrame, val: pd.DataFrame, arms: dict[str, list[str]],
           n_knots: int = SPLINE_KNOTS, variant: str = VARIANT,
           seed: int = SEED) -> dict[str, FittedArm]:
    """Every arm fitted once, on the covered window."""
    fitted = {}
    for name, extra in arms.items():
        arm = fit_arm(train, val, name, extra, n_knots, variant)
        fitted[name] = arm
        print(f"  {name:<24} {len(arm.model.features):>3} features "
              f"({len(extra):>2} preseason)   rho {arm.model.rho:.5f}"
              f"{'' if arm.model.converged else '   NOT CONVERGED'}")
    return fitted


def score_population(fitted: dict[str, FittedArm], mask: np.ndarray, population: str,
                     floor_crps: float | None = None, seed: int = SEED) -> pd.DataFrame:
    """Every fitted arm scored on one population of validation rows.

    The arms are **not** refitted per population. The head fits what it fits; P1 decision 5
    is about where a figure is *read*, and restricting the fitting rows as well would
    confound
    a population statement with a smaller training set.
    """
    rows, per_row = [], {}
    for name, arm in fitted.items():
        frame = arm.frame.loc[mask].reset_index(drop=True)
        samples = arm.model.predict_samples(frame, seed)
        row, scores = score_arm(name, samples, frame, arm.train,
                                len(arm.model.features), arm.model.rho,
                                arm.model.rho_spread, seed=seed)
        row["population"] = population
        row["n_preseason_cols"] = len(arm.extra)
        row["preseason_cols"] = "|".join(arm.extra)
        rows.append(row)
        per_row[name] = scores

    reference, primary = per_row[REFERENCE_ARM], per_row[PRIMARY_ARM]
    for row in rows:
        delta = paired_bootstrap(per_row[row["arm"]], reference, n_boot=BOOTSTRAP_REPS,
                                 seed=seed)
        row["crps_vs_incumbent"] = delta["crps_delta"]
        row["crps_vs_incumbent_lo"] = delta["ci_lo"]
        row["crps_vs_incumbent_hi"] = delta["ci_hi"]
        # The reference's own row is a comparison against itself, where the interval is
        # degenerately [0, 0] and `verdict` would read it as a win. `minutes_window.ladder`
        # names that case for the same reason.
        row["verdict"] = "reference" if row["arm"] == REFERENCE_ARM else verdict(delta)
        row["beats_floor"] = (bool(row["val_crps"] <= floor_crps)
                              if floor_crps is not None else None)
        # And against the DECLARED PRIMARY, so "arm X beats the primary" is an interval
        # rather than two point estimates read side by side. Every arm here is an
        # attribution or a sensitivity on that arm, and the only thing that could promote
        # one of them over it is a paired comparison against it.
        against = paired_bootstrap(per_row[row["arm"]], primary, n_boot=BOOTSTRAP_REPS,
                                   seed=seed)
        row["crps_vs_primary"] = against["crps_delta"]
        row["crps_vs_primary_lo"] = against["ci_lo"]
        row["crps_vs_primary_hi"] = against["ci_hi"]
        row["verdict_vs_primary"] = ("primary" if row["arm"] == PRIMARY_ARM
                                     else verdict(against))
    return pd.DataFrame(rows)


def floor_row(train: pd.DataFrame, val: pd.DataFrame, population: str,
              seed: int = SEED) -> tuple[dict, float]:
    """The mandatory no-fit carry-forward, on one population.

    `docs/minutes-window-plan.md` §2 carries it for the same reason `component_rates` does:
    a head that does not clear a prior-season carry-forward is not a model, and at the season
    unit this head's own floor is close enough (161.29 against 144.23) that the check is not
    a formality.
    """
    floor = FloorMinutes().fit(train)
    samples = floor.predict_samples(val, seed)
    row, _ = score_arm("carry_forward", samples, val, train, 0, floor.rho, 1.0, seed=seed)
    row["population"] = population
    row["n_preseason_cols"] = 0
    row["preseason_cols"] = ""
    row["verdict"] = "floor"
    return row, float(row["val_crps"])


# ── Rolling-origin confirmation, fitting half only ────────────────────────────

def rolling_confirmation(train: pd.DataFrame, arms: dict[str, list[str]],
                         n_knots: int = SPLINE_KNOTS, variant: str = VARIANT,
                         first_origin: int = FIRST_ORIGIN, seed: int = SEED
                         ) -> pd.DataFrame:
    """Walk-forward over the covered fitting half: one fit per (origin, arm).

    **This is half the gate, not a post-hoc check.** The ladder scores seven arms on a few
    hundred validation player-seasons, which is a multiplicity problem and a power problem at
    once, and this project has been wrong in exactly that spot three times — twice on the
    availability head (`docs/availability-window-plan.md` §12e, §14f) and once on this head's
    own fitting-window axis (`docs/minutes-window-plan.md` §3, -2.687 on validation against
    -0.079 rolling). Every row touched here is a fitting-half row.

    Unlike `minutes_window.rolling_confirmation` the **window is not an axis**: that ladder
    settled it, and re-crossing it here would price the preseason block against a window the
    fitting half already refused. Each origin fits on every covered season before it.
    """
    year = season_start_year(train)
    origins = [int(y) for y in sorted(np.unique(year)) if y >= first_origin]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        score = train[year == origin]
        fit_rows = train[year < origin]
        if score.empty or len(fit_rows) < MIN_FIT_ROWS:
            continue
        tr, sc, features = variants(fit_rows, score, n_knots)[variant]
        y = sc["successes"].to_numpy(dtype=float)
        for name, extra in arms.items():
            model = PointMinutes(features + extra).fit(tr)
            samples = model.predict_samples(sc, seed)
            slot = per_arm.setdefault(name, {})
            slot.setdefault("crps", []).append(crps_from_samples(samples, y))
            slot.setdefault("pit", []).append(pit_from_samples(samples, y, seed))
            slot.setdefault("sd", []).append(samples.std(axis=0))
            slot.setdefault("y", []).append(y)
            slot.setdefault("origin", []).append(np.full(len(sc), origin))
            slot.setdefault("n_fit", []).append(len(tr))
        print(f"  origin {origin}: fitted {len(tr):,} rows, scored {len(sc):,}")

    pooled = {name: {k: np.concatenate(v) for k, v in d.items() if k != "n_fit"}
              for name, d in per_arm.items()}
    reference = pooled[REFERENCE_ARM]["crps"]
    primary = pooled[PRIMARY_ARM]["crps"]
    rows = []
    for name, d in pooled.items():
        origin = d["origin"]
        delta = paired_bootstrap(d["crps"], reference, n_boot=BOOTSTRAP_REPS, seed=seed)
        against = paired_bootstrap(d["crps"], primary, n_boot=BOOTSTRAP_REPS, seed=seed)
        primary_wins = sum(1 for o in np.unique(origin)
                           if d["crps"][origin == o].mean() < primary[origin == o].mean())
        # Origins are the independent replicates, so a win count over them is the
        # multiplicity-robust statement a pooled interval is not.
        wins = sum(1 for o in np.unique(origin)
                   if d["crps"][origin == o].mean() < reference[origin == o].mean())
        rows.append({
            "arm": name,
            "n_preseason_cols": len(arms[name]),
            "n_origins": int(len(np.unique(origin))),
            "n_scored": int(len(d["y"])),
            "mean_fit_rows": float(np.mean(per_arm[name]["n_fit"])),
            "crps": float(d["crps"].mean()),
            "crps_vs_incumbent": delta["crps_delta"],
            "crps_vs_incumbent_lo": delta["ci_lo"],
            "crps_vs_incumbent_hi": delta["ci_hi"],
            "origins_won": wins,
            "verdict": "reference" if name == REFERENCE_ARM else verdict(delta),
            "crps_vs_primary": against["crps_delta"],
            "crps_vs_primary_lo": against["ci_lo"],
            "crps_vs_primary_hi": against["ci_hi"],
            "origins_won_vs_primary": primary_wins,
            "verdict_vs_primary": ("primary" if name == PRIMARY_ARM
                                   else verdict(against)),
            "pit_ks": ks_uniform(d["pit"]),
            "predictive_sd": float(d["sd"].mean()),
            "realized_sd": float(d["y"].std()),
        })
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


# ── The gate ──────────────────────────────────────────────────────────────────

def gate(validation: pd.DataFrame, rolling: pd.DataFrame, arm: str = PRIMARY_ARM,
         population: str = DECISION_POPULATION) -> dict:
    """Both halves of the bar, evaluated as code rather than as a judgement after the fact.

    The composition head is priced **only if this passes** — `docs/preseason-plan.md` P3 —
    so the go/no-go is a column in the artifact rather than a sentence in a printout.
    """
    val = validation[(validation["arm"] == arm)
                     & (validation["population"] == population)].iloc[0]
    roll = rolling[rolling["arm"] == arm].iloc[0]
    val_pass = bool(val["crps_vs_incumbent_hi"] < 0.0)
    roll_pass = bool(roll["crps_vs_incumbent_hi"] < 0.0
                     and roll["origins_won"] * 2 > roll["n_origins"])

    # Whether any sensitivity or attribution arm beats the declared primary — and crucially
    # whether it does so on the FITTING HALF, where a promotion can be justified without
    # spending the selection split on a choice made after seeing it. An arm that beats the
    # primary on validation alone is a validation-driven swap and is recorded as such.
    challenger = rolling[(rolling["verdict_vs_primary"] == "wins")
                         & (rolling["origins_won_vs_primary"] * 2 > rolling["n_origins"])]
    best = str(challenger.iloc[0]["arm"]) if len(challenger) else ""
    val_challenger = validation[(validation["arm"] == best)
                                & (validation["population"] == population)]

    return {"arm": arm, "population": population,
            "challenger": best,
            "challenger_rolling_delta": (float(challenger.iloc[0]["crps_vs_primary"])
                                         if best else np.nan),
            "challenger_rolling_hi": (float(challenger.iloc[0]["crps_vs_primary_hi"])
                                      if best else np.nan),
            "challenger_val_delta": (float(val_challenger.iloc[0]["crps_vs_primary"])
                                     if len(val_challenger) else np.nan),
            "challenger_val_hi": (float(val_challenger.iloc[0]["crps_vs_primary_hi"])
                                  if len(val_challenger) else np.nan),
            "challenger_confirmed_on_fitting_half": bool(best),
            "val_crps_delta": float(val["crps_vs_incumbent"]),
            "val_lo": float(val["crps_vs_incumbent_lo"]),
            "val_hi": float(val["crps_vs_incumbent_hi"]),
            "val_pass": val_pass,
            "rolling_crps_delta": float(roll["crps_vs_incumbent"]),
            "rolling_lo": float(roll["crps_vs_incumbent_lo"]),
            "rolling_hi": float(roll["crps_vs_incumbent_hi"]),
            "origins_won": int(roll["origins_won"]),
            "n_origins": int(roll["n_origins"]),
            "rolling_pass": roll_pass,
            "shrinkage_vs_validation": float(val["crps_vs_incumbent"]
                                             / roll["crps_vs_incumbent"])
            if roll["crps_vs_incumbent"] else np.nan,
            "passes": bool(val_pass and roll_pass),
            "price_composition": bool(val_pass and roll_pass)}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = Path(cfg["eda"]["output_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    seasons = cfg["data"]["seasons"]
    n_knots = int(cfg.get("stan", {}).get("minutes", {}).get("spline_knots", SPLINE_KNOTS))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))

    variant = assert_shipped_variant(out_dir)
    print("Minutes preseason increment (P3) — the preseason block as a nested arm on "
          "`min | available`")
    print(f"  variant held fixed at {variant}")
    print("  The BAR, stated before the run: validation CRPS interval clear of zero on the "
          "draftable\n  population AND the rolling-origin harness agreeing (interval clear "
          "of zero, majority of\n  origins). Validation alone ships nothing — two blocks on "
          "this project won a validation\n  reading and shrank 4-6x rolling.")

    # ── the design ───────────────────────────────────────────────────────────
    from src.features.availability import load_artifacts
    panel = pd.read_parquet(features_dir / "preseason.parquet")
    coverage = pd.read_csv(eda_dir / "preseason_coverage.csv")
    scope = covered_seasons(coverage)
    first_covered = _season_start_year(scope[0])

    design = minutes_build_design(cfg)
    av_panel, _ = load_artifacts(features_dir)
    design = attach_preseason(design, panel, av_panel, seasons)
    design = attach_season_start_roster(design, seasons, raw_dir, window_games)

    train_full, val = selection_split(design, test_seasons)
    train = restrict_window(train_full, first_covered).reset_index(drop=True)
    val = val.reset_index(drop=True)

    print(f"\n  {len(design):,} player-seasons; {len(train_full):,} fit / {len(val):,} "
          f"select ({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  Preseason coverage begins {scope[0]}, and this head fits from "
          f"{min(train_full['season'])}. So EVERY arm — the\n  reference included — fits "
          f"the covered window only: {len(train):,} of {len(train_full):,} training rows "
          f"({len(train) / len(train_full):.1%}).\n  The {len(train_full) - len(train):,} "
          f"rows dropped have no preseason row for a reason that is a fact about\n  the "
          f"API, and the missing indicator would read as an era dummy on every one of them.")
    print(f"  Validation rows inside coverage: {int(val['season'].isin(scope).sum()):,} of "
          f"{len(val):,}.")

    draftable = val["on_season_start_roster"].to_numpy(dtype=float) > 0
    print(f"  Draftable validation rows (on a season-start roster): {int(draftable.sum()):,}"
          f" of {len(val):,} ({draftable.mean():.1%}); "
          f"{val['has_preseason'].mean():.1%} of validation rows carry a preseason row.")

    # ── the shrinkage constant, on the fitting half ──────────────────────────
    print(f"\nStep 1 — the reliability shrink `min_pre / (min_pre + k)`, which "
          f"`docs/preseason-plan.md`\n  leaves open. Selected on an inner carve of the "
          f"FITTING half ({INNER_SCORE_SEASONS} trailing training seasons\n  scored), never "
          f"on validation:")
    k, shrink_table = fit_shrinkage(train, n_knots=n_knots, variant=VARIANT, seed=SEED)
    shrink_dest = out_dir / "minutes_preseason_shrinkage.csv"
    shrink_table.assign(selected=shrink_table["k"] == k).to_csv(shrink_dest, index=False)
    print(shrink_table.round(4).to_string(index=False))
    print(f"  selected k = {k:.0f} preseason minutes"
          + ("  (the grid's left edge — no shrink, so the shrunk arm IS the primary arm)"
             if k <= 0 else ""))

    train = with_shrunk_delta(train, k)
    val = with_shrunk_delta(val, k)

    # ── the ladder ───────────────────────────────────────────────────────────
    arms = preseason_arms()
    print(f"\nStep 2 — the ladder: {len(arms)} nested arms on the covered window "
          f"(the reference carries no\n  preseason column, and every other arm is it plus "
          f"columns, so zero recovers it exactly):")
    fitted = ladder(train, val, arms, n_knots, VARIANT, SEED)

    blocks = []
    for population, mask in (("all", np.ones(len(val), dtype=bool)),
                             (DECISION_POPULATION, draftable)):
        floor, floor_crps = floor_row(train, val.loc[mask].reset_index(drop=True),
                                      population, SEED)
        scored = score_population(fitted, mask, population, floor_crps, SEED)
        blocks.append(pd.concat([pd.DataFrame([floor]), scored], ignore_index=True))
    validation = pd.concat(blocks, ignore_index=True)

    # The full-window incumbent, scored as CONTEXT so the coverage restriction's own cost is
    # visible. It is not on the ladder: it fits different rows from every other arm, so an
    # interval against it would carry two changes at once.
    context = fit_arm(train_full, val, "incumbent_full_window", [], n_knots, VARIANT)
    ctx_rows = []
    for population, mask in (("all", np.ones(len(val), dtype=bool)),
                             (DECISION_POPULATION, draftable)):
        frame = context.frame.loc[mask].reset_index(drop=True)
        row, _ = score_arm("incumbent_full_window",
                           context.model.predict_samples(frame, SEED), frame,
                           context.train, len(context.model.features),
                           context.model.rho, 1.0, seed=SEED)
        row.update({"population": population, "n_preseason_cols": 0,
                    "preseason_cols": "", "verdict": "context"})
        ctx_rows.append(row)
    validation = pd.concat([validation, pd.DataFrame(ctx_rows)], ignore_index=True)

    dest = out_dir / "minutes_preseason.csv"
    validation.to_csv(dest, index=False)

    show = ["arm", "n_preseason_cols", "n_val", "val_crps",
            "crps_vs_incumbent", "crps_vs_incumbent_lo", "crps_vs_incumbent_hi",
            "crps_vs_primary", "crps_vs_primary_lo", "crps_vs_primary_hi",
            "val_pit_ks", "val_mae", "val_bias", "predictive_sd", "verdict"]
    for population, label in (
            (DECISION_POPULATION, "ON A SEASON-START ROSTER — the production population, "
                                  "and the one the verdict is read on"),
            ("all", "every row the head fits — a population statement, not a model one")):
        part = validation[validation["population"] == population]
        print(f"\n  {label}:")
        print(part[show].round(4).to_string(index=False))
    print(f"\nWrote {len(validation):,} scored rows → {dest}")

    # ── rolling-origin confirmation ──────────────────────────────────────────
    print(f"\nStep 3 — rolling-origin confirmation on the COVERED FITTING HALF ONLY "
          f"(origins from {FIRST_ORIGIN}).\n  This is half the gate, not a post-hoc check.")
    rolling = rolling_confirmation(train, arms, n_knots, VARIANT, FIRST_ORIGIN, SEED)
    roll_dest = out_dir / "minutes_preseason_rolling.csv"
    rolling.to_csv(roll_dest, index=False)
    print()
    print(rolling[["arm", "mean_fit_rows", "n_scored", "crps", "crps_vs_incumbent",
                   "crps_vs_incumbent_lo", "crps_vs_incumbent_hi", "origins_won",
                   "n_origins", "crps_vs_primary", "crps_vs_primary_lo",
                   "crps_vs_primary_hi", "origins_won_vs_primary", "pit_ks",
                   "predictive_sd", "verdict"]]
          .round(4).to_string(index=False))
    print(f"\nWrote {len(rolling):,} arms × {int(rolling['n_scored'].max()):,} scored rows "
          f"→ {roll_dest}")

    # ── the gate ─────────────────────────────────────────────────────────────
    result = gate(validation, rolling, PRIMARY_ARM, DECISION_POPULATION)
    print(f"\nStep 4 — the gate on the PRIMARY arm (`{PRIMARY_ARM}`), both halves:")
    print(f"    validation      {result['val_crps_delta']:+8.3f} "
          f"[{result['val_lo']:+.2f}, {result['val_hi']:+.2f}]   "
          f"{'PASS' if result['val_pass'] else 'FAIL'}")
    print(f"    rolling origin  {result['rolling_crps_delta']:+8.3f} "
          f"[{result['rolling_lo']:+.2f}, {result['rolling_hi']:+.2f}]   "
          f"{result['origins_won']}/{result['n_origins']} origins   "
          f"{'PASS' if result['rolling_pass'] else 'FAIL'}")
    print(f"    {'GATE PASSES' if result['passes'] else 'GATE FAILS'} — the composition "
          f"head is priced only if this passes\n    (`docs/preseason-plan.md` P3), so "
          f"`price_composition` = {result['price_composition']}.")
    if result["challenger"]:
        print(f"\n    And an attribution arm beats the declared primary on BOTH readings: "
              f"`{result['challenger']}`\n    reads "
              f"{result['challenger_rolling_delta']:+.3f}"
              f" [.., {result['challenger_rolling_hi']:+.2f}] against it on the fitting "
              f"half and\n    {result['challenger_val_delta']:+.3f} "
              f"[.., {result['challenger_val_hi']:+.2f}] on validation. The rolling reading "
              f"is what makes promoting it\n    a fitting-half decision rather than a swap "
              f"made after seeing the selection split.")

    validation = pd.concat(
        [validation, pd.DataFrame([{**result, "arm": f"gate__{result['arm']}",
                                    "population": result["population"],
                                    "verdict": "gate"}])], ignore_index=True)
    validation.to_csv(dest, index=False)

    return {"ladder": dest, "rolling": roll_dest, "shrinkage": shrink_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
