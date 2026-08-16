"""Session 6b — the preseason increment on every component RATE head, as nested arms.

`make components-preseason`. Three artifacts in `outputs/predictions/`:
`components_preseason.csv`, `components_preseason_rolling.csv`,
`components_preseason_shrinkage.csv`.

## Why these six heads, and why a session at all

`docs/preseason-plan.md` P1 decision 2. The plan's a-priori expectation was that the rate
half would collapse — the count heads already sit 0.81-0.95 validation R² against a no-fit
carry-forward, so there was very little room for six preseason games to add anything. **Five
of seven count heads cleared the bar instead**, so the gate *added* a session rather than
shrinking one:

| head | P1 ΔR² | of which its own delta | of which the shared pair |
|---|---|---|---|
| `ast` | +0.0122 | +0.0123 | -0.0001 |
| `fga` | +0.0051 | +0.0054 | -0.0002 |
| `stl` | +0.0021 | +0.0023 | -0.0001 |
| `tov` | +0.0017 | +0.0019 | -0.0002 |
| `reb` | +0.0016 | +0.0036 | -0.0010 |
| `ftm\\|fta` | +0.0176 (own R², delta-carried) | | |

**All eleven heads are armed, and the five P1 left out are the second run's new information.**
The first pass carried P1's short list alone. That exclusion did not survive the round: the
screen misranked the heads it *admitted* in both directions — its largest increment
(`ftm|fta`) is a tie here and its smallest clearing one (`reb`) is the round's biggest
block-to-fit ratio — and two of the five it excluded were never negative to begin with
(`fg3m|fg3a` +0.01488 at z = 7.53, excluded because the gain is the shared indicator pair
rather than preseason 3P%; `fg3a|fga` +0.00467 delta-carried at z = 6.70, better than two
armed heads). Three heads *do* carry a negative screen — `blk`, `fg2m|fg2a`, `fta` — and a
permutation z that negative is evidence of **harm** rather than absence of gain, which is a
claim worth pricing at a paired interval rather than inheriting. See `COUNT_ARMS`.

## The bar, stated before the run — P1's own "what it does not settle"

P1's bar was an **R² screen on a point estimate**, and it is explicit that this is a filter
for what is worth fitting rather than evidence that anything ships. So this module re-asks the
same question at the bar every other arm in the round has been held to (P2, P3), and an arm
survives only if **both** readings agree:

1. **Validation.** Its CRPS paired-bootstrap interval against the covered-window incumbent
   lies entirely below zero, at the head's own unit (the season total for a count head, made
   free throws for the conversion head), on the **draftable** population — P1 decision 5.
2. **The rolling-origin harness on the fitting half.** Same sign, interval below zero, and a
   majority of origins won.

Validation alone ships nothing: `docs/availability-window-plan.md` §12e and §14f record two
blocks that won a validation reading and shrank 4-6x rolling. And the multiplicity here is
larger than any round before it — **six heads** rather than one — so the rolling harness is
carrying more weight than it did in P2 or P3, not less.

Nothing here ships a head. An arm that clears both halves earns a Stan port in
`stan_components`, and — unlike the marginal minutes head — these six sit in the simulator's
own draw path, so a survivor is priceable with `make preseason-contest` at `--groups
components`.

## Point MLE on the shipped features, which is the P3 template exactly

Each arm is the head's **shipped variant** (read off `stan_component_metrics.csv`, never
hard-coded) plus preseason columns, fitted by `component_rates.fit_count_head` /
`fit_conversion_head` — the sklearn point references the Stan heads are checked against. So
the features are the shipped basis and the fit costs seconds, which is what lets an arm be
rejected before any sampler time is spent. `minutes_preseason` does the same thing one head
over: `stan_minutes.variants` for the design, `minutes_window.PointMinutes` for the fit.

The predictive is the **plug-in** one — NB(mu, phi) for a count, beta-binomial(p, rho) for a
conversion — so it carries no parameter uncertainty, exactly as `PointMinutes` does. That
narrows every arm's predictive equally and is therefore fair across the ladder; what it means
is that a CRPS *level* here is not comparable to `stan_component_metrics.csv`'s, only the
*differences* between arms are.

## The coverage restriction bites here, as it did on minutes

The preseason panel begins at 2004-05 and this design fits from 1997-98, so a missing-preseason
indicator would read as an era dummy on the pre-2005 rows. **Every arm, the reference
included, fits the covered window only**, and the first covered season is read off
`preseason_coverage.csv` rather than hard-coded. The full-window incumbent is scored beside it
as a context row so the restriction's own cost is visible and cannot be credited to the block
— P3 paid a quarter of its increment there and session 4d paid 8.5%.

## Difference coding on each head's own link

A count head is a log-link rate, so its delta is `log1p(per-36 preseason) - log1p(per-36 prior
season)`; a conversion head is a logit-link proportion, so its delta is
`logit(preseason pct) - logit(prior pct)`. Both are built by
`preseason_value.attach_rate_block`, **reused rather than forked** — the rule
`docs/preseason-plan.md` states for the delta builders, and the reason
`preseason_value.season_centered` and `attach_missing_age_indicators` live there too.

**Zero recovers the incumbent exactly.** A player whose preseason agrees with his prior season
carries a zero delta; a player with no preseason row carries a zero delta plus exactly one age
indicator; and the coefficient path through zero is the shipped head. The house nesting
discipline (`pi = 0`, `K = 1`, `U_n = 0`, `n_rho = 1`), pinned by a test.

Usage:
    python -m src.models.components_preseason
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year
from src.eda.preseason_value import (MISSING_AGE_COLS, attach_missing_age_indicators,
                                     attach_rate_block, attach_season_start_roster,
                                     covered_seasons, season_centered)
from src.models.availability import fit_dispersion
from src.models.availability_window import restrict_window
from src.models.component_rates import (build_design, carry_forward,
                                        carry_forward_conversion, fit_conversion_head,
                                        fit_count_head, fit_nb_dispersion)
from src.models.held_out import selection_split
from src.models.minutes_preseason import reliability_weight
from src.models.minutes_unification import paired_bootstrap, verdict
from src.models.minutes_window import BOOTSTRAP_REPS
from src.models.season_terms import season_start_year
from src.models.stan_components import (PREDICTIVE_SAMPLES, SPLINE_KNOTS, _beta_shapes,
                                        conversion_variants, count_variants)
from src.models.stan_utils import crps_from_samples, ks_uniform, pit_from_samples

# ⚙️ **Every component head is armed, since 2026-08-15.** The first run of this module
# carried P1 decision 2's short list alone — `ast`, `fga`, `stl`, `tov`, `reb` and
# `ftm|fta` — and the five it left out were left out on P1's ΔR² screen. That exclusion did
# not survive its own round, for two reasons recorded here rather than in a commit message:
#
#   1. **The screen misranked the heads it did admit, in both directions.** Its largest rate
#      increment (`ftm|fta`, +0.01755 at z = 18.8) is a tie at both readings of this ladder,
#      and its smallest clearing count head (`reb`, +0.00165) has the round's largest
#      block-to-fit ratio. A screen that cannot order the heads it passed is not evidence
#      about the heads it failed.
#   2. **Two of the five were never negative at all.** `fg3m|fg3a` reads +0.01488 at z = 7.53
#      — a LARGER ΔR² than four armed heads — and was excluded because the gain is the shared
#      indicator pair rather than preseason 3P% (own delta −0.00237). `fg3a|fga` reads
#      +0.00467 delta-carried at z = 6.70, better than `stl` or `tov`, and sat out only
#      because `ftm|fta` was the standout among conversions.
#
# Three heads DO carry a genuinely negative screen — `blk` (−0.00847, z = −3.18), `fg2m|fg2a`
# (−0.00845, z = −9.44) and `fta` (−0.00161, z = −2.32) — and a permutation z that negative
# means the real block scores worse out of sample than a SHUFFLED block of the same shape,
# which is evidence of harm rather than absence of gain. They are armed anyway, because this
# ladder is the instrument that can price that claim at a paired interval and it costs five
# minutes: excluding a head on a screen this one has already contradicted is the same error
# in the other direction. An arm that comes back positive-and-unresolvable is a null; an arm
# that comes back with an interval clear of zero ON THE WRONG SIDE is the screen confirmed,
# and either is worth more than the assumption.
COUNT_ARMS = ["ast", "fga", "stl", "tov", "reb", "blk", "fta"]
CONVERSION_ARMS = [("ftm", "fta"), ("fg3a", "fga"), ("fg2m", "fg2a"), ("fg3m", "fg3a")]

# P1's screen for every head, as data, so the artifact carries what the gate is being read
# against. Positive entries were admitted by P1; the three negative ones are the claim this
# round is testing. `fg3m|fg3a`'s +0.01488 is the trap the attribution split caught — its own
# delta is −0.00237 and the whole gain is the shared indicator pair.
P1_SCREEN = {"ast": 0.01223, "fga": 0.00511, "stl": 0.00205, "tov": 0.00170,
             "reb": 0.00165, "blk": -0.00847, "fta": -0.00161,
             "ftm|fta": 0.01755, "fg3a|fga": 0.00467, "fg2m|fg2a": -0.00845,
             "fg3m|fg3a": 0.01488}

# The three whose screen is negative — armed here for the first time, and the heads whose
# gate reading is the new information in the second run.
P1_NEGATIVE_HEADS = {h: v for h, v in P1_SCREEN.items() if v < 0}

# The reference every interval is taken against: the shipped variant refit on the covered
# window with no preseason column at all.
REFERENCE_ARM = "incumbent"

# Declared before the run. The gate's verdict is read off this arm on every head; everything
# else on the ladder is a sensitivity or an attribution.
PRIMARY_ARM = "own_delta"

# P1 decision 5. Both populations are written; the draftable one decides.
DECISION_POPULATION = "draftable"

# P1's additive reliability term. `own_delta_reliability` below is P1's three-column rate
# block exactly, with `has_preseason` replaced by its age split per decision 3.
RELIABILITY = "pre_log_min"

# Pseudo-minutes for the empirical-Bayes weight `min_pre / (min_pre + k)`, shared with P3 so
# the two heads' selected constants are read on one scale. 0 is no shrinkage — the weight is
# identically 1 and the shrunk arm IS the primary — which makes the grid's left edge a control
# rather than a guess.
SHRINKAGE_GRID: tuple[float, ...] = (0.0, 20.0, 40.0, 80.0, 160.0, 320.0)

# Trailing TRAINING seasons scored rather than fitted when `k` is selected. Two, matching
# `preseason_value.INNER_SCORE_SEASONS` and the project's split arity — and never validation,
# which a fitted constant must not be allowed to read.
INNER_SCORE_SEASONS = 2

# Rolling-origin harness. `FIRST_ORIGIN` matches `minutes_preseason.FIRST_ORIGIN` so the two
# confirmations are read side by side; with coverage starting at 2004-05 it leaves five
# seasons of fitting rows at the earliest origin.
FIRST_ORIGIN = 2009
MIN_FIT_ROWS = 1000

SEED = 0


# ── Heads, as one object rather than two code paths ───────────────────────────

class Head:
    """One rate head: its name, its likelihood, and the three things a ladder needs of it.

    Counts and conversions differ in the variant builder, the fit, and the predictive — but
    not in anything the ladder does with them, so they are one class with two constructors
    rather than two parallel loops. The alternative is the shape `component_rates.evaluate`
    has, where every scoring change has to be made twice and can be made inconsistently once.
    """

    def __init__(self, name: str, kind: str, component: str, attempted: str | None,
                 delta: str):
        self.name, self.kind = name, kind
        self.component, self.attempted, self.delta = component, attempted, delta

    # -- the design -------------------------------------------------------
    def variants(self, train: pd.DataFrame, test: pd.DataFrame, n_knots: int):
        if self.kind == "count":
            return count_variants(train, test, self.component, n_knots)
        return conversion_variants(train, test, self.component, self.attempted, n_knots)

    def live(self, frame: pd.DataFrame) -> np.ndarray:
        """Rows the head can score. A conversion head needs at least one attempt."""
        if self.kind == "count":
            return np.ones(len(frame), dtype=bool)
        return frame[self.attempted].to_numpy(dtype=float) > 0

    def target(self, frame: pd.DataFrame) -> np.ndarray:
        y = frame[self.component].to_numpy(dtype=float)
        if self.kind == "count":
            return y
        n = frame[self.attempted].to_numpy(dtype=float)
        return np.minimum(np.rint(y), np.rint(n)).astype(float)

    # -- the fit and its plug-in predictive --------------------------------
    def fit_predict(self, train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                    n_samples: int = PREDICTIVE_SAMPLES, seed: int = SEED
                    ) -> tuple[np.ndarray, np.ndarray, float]:
        """`(samples, point prediction, dispersion)` on the rows `live` keeps.

        The point MLE the Stan head is checked against, and its plug-in predictive. No
        parameter uncertainty, exactly as `minutes_window.PointMinutes` — which narrows every
        arm alike, so ladder *differences* are fair even though the CRPS *level* is not
        comparable to `stan_component_metrics.csv`.
        """
        if self.kind == "count":
            mu, phi = fit_count_head(train, test, features, self.component)
            return count_samples(mu, phi, n_samples, seed), mu, phi
        mask, p, rho = fit_conversion_head(train, test, features, self.component,
                                           self.attempted)
        n = np.rint(test[self.attempted].to_numpy(dtype=float)).astype(int)[mask]
        return conversion_samples(p, rho, n, n_samples, seed), p, rho

    def floor_predict(self, train: pd.DataFrame, test: pd.DataFrame,
                      n_samples: int = PREDICTIVE_SAMPLES, seed: int = SEED
                      ) -> tuple[np.ndarray, np.ndarray, float]:
        """The mandatory no-fit floor, wrapped in the head's own likelihood.

        `component_rates` makes this non-optional and the reason is on the record: the rate
        side is close to saturated from prior-season information alone, so a head that does
        not clear arithmetic is not a model. It matters more here than usual because
        `ftm|fta` **does not clear its floor today** — it is the project's known conversion
        null — so "does the block get it over the line" is a live question rather than a
        formality.
        """
        if self.kind == "count":
            mu = np.clip(carry_forward(test, self.component), 1e-6, None)
            phi = fit_nb_dispersion(
                train[self.component].to_numpy(dtype=float),
                np.clip(carry_forward(train, self.component), 1e-6, None))
            return count_samples(mu, phi, n_samples, seed), mu, phi
        mask = self.live(test)
        live = self.live(train)
        p = carry_forward_conversion(train, test, self.component, self.attempted)[mask]
        rho = fit_dispersion(
            np.rint(train[self.component].to_numpy(dtype=float)[live]).astype(int),
            np.rint(train[self.attempted].to_numpy(dtype=float)[live]).astype(int),
            carry_forward_conversion(train, train, self.component,
                                     self.attempted)[live])
        n = np.rint(test[self.attempted].to_numpy(dtype=float)).astype(int)[mask]
        return conversion_samples(p, rho, n, n_samples, seed), p, rho


def heads() -> list[Head]:
    """Every component head as a `Head` object, counts first — see `COUNT_ARMS`."""
    out = [Head(c, "count", c, None, f"pre_d_{c}") for c in COUNT_ARMS]
    out += [Head(f"{m}|{a}", "conversion", m, a, f"pre_d_{m}") for m, a in CONVERSION_ARMS]
    return out


def count_samples(mu: np.ndarray, phi: float, n_samples: int, seed: int) -> np.ndarray:
    """NB2 predictive draws at a plug-in `(mu, phi)` — `(draws x rows)`."""
    rng = np.random.default_rng(seed)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
    p = phi / (phi + mu)
    return rng.negative_binomial(np.full((n_samples, len(mu)), float(phi)),
                                 np.repeat(p[None, :], n_samples, axis=0)).astype(float)


def conversion_samples(p: np.ndarray, rho: float, n: np.ndarray, n_samples: int,
                       seed: int) -> np.ndarray:
    """Beta-binomial predictive draws of makes at a plug-in `(p, rho)` — `(draws x rows)`."""
    rng = np.random.default_rng(seed)
    a, b = _beta_shapes(np.repeat(np.asarray(p, dtype=float)[None, :], n_samples, axis=0),
                        np.full((n_samples, 1), float(rho)))
    return rng.binomial(np.asarray(n, dtype=int)[None, :], rng.beta(a, b)).astype(float)


# ── The block ─────────────────────────────────────────────────────────────────

def centered_column(delta: str) -> str:
    return f"{delta}_centered"


def shrunk_column(delta: str) -> str:
    return f"{delta}_shrunk"


def preseason_arms(delta: str) -> dict[str, list[str]]:
    """`arm name -> the preseason columns it adds to the shipped feature list`.

    Every arm is the shipped variant **plus** these columns, so each is a nested increment on
    the incumbent rather than a different head, and the empty list is the incumbent itself.
    The ladder is deliberately short — P1 already screened the columns, and re-running a
    feature selection on the selection split is what the gate exists to avoid.
    """
    return {
        REFERENCE_ARM: [],
        # PRIMARY — the head's own delta on its own link, plus the age-split indicator.
        PRIMARY_ARM: [delta] + MISSING_AGE_COLS,
        # P1's rate block verbatim, with decision 3 applied to its indicator. On P1's own
        # attribution the reliability pair contributes ~0.000 to every count head, so this
        # arm should NOT beat the primary — an arm that exists to be beaten is how that
        # reading is checked rather than assumed.
        "own_delta_reliability": [delta, RELIABILITY] + MISSING_AGE_COLS,
        # ATTRIBUTION — the delta with each season's own mean removed. P3 found preseason
        # quantities carry a league-level shift that varies by season (the calendar runs from
        # 2 games a team in the 2011-12 lockout to 8 in an ordinary year), and P2 found the
        # same on a level rather than a delta. Neither of these heads has a season term to
        # absorb it. Whether that replicates on a RATE is the open question this arm asks:
        # a per-36 rate is already a ratio, so the compression argument that motivated
        # centring on minutes does not obviously carry.
        "own_delta_centered": [centered_column(delta)] + MISSING_AGE_COLS,
        # The cross of the two things that won on the other heads — P2 shipped the wider
        # block, P3 shipped the centred column, and nothing has yet tested them together.
        "own_delta_centered_reliability": ([centered_column(delta), RELIABILITY]
                                           + MISSING_AGE_COLS),
        # The empirical-Bayes shrink at the constant fitted on the fitting half. `k = 0` makes
        # this the primary arm exactly.
        "own_delta_shrunk": [shrunk_column(delta)] + MISSING_AGE_COLS,
        # ATTRIBUTION — the indicator alone. If this carries the primary arm's gain then the
        # finding is "a missing preseason is a signal", not "the preseason rate delta is",
        # which is the split P1's own attribution columns exist to catch.
        "missing_only": list(MISSING_AGE_COLS),
    }


def attach_preseason(design: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """Every preseason column this ladder can read, merged onto the rate design.

    **Opt-in, and outside `component_rates.build_design` deliberately** — the
    `availability.attach_absence_mix` and `minutes_preseason.attach_preseason` precedent, for
    the identical reason. That builder is the route `stan_components`, `season_terms`,
    `posteriors` and the substitution sweep all take to their rows, and a column that is
    structurally zero before 2004-05 must not be able to enter any of them by accident.

    P1's delta builder is reused rather than forked: `attach_rate_block` writes every count
    head's `log1p` per-36 delta and every conversion head's logit percentage delta on P1's own
    scales. What this adds is the two devices P1's later decisions produced — the age-split
    indicator (decision 3) and the season-centred twin (P3's finding).
    """
    out = attach_missing_age_indicators(attach_rate_block(design, panel))
    for head in heads():
        out[centered_column(head.delta)] = season_centered(out, head.delta)
    out["min_pre"] = out["min_pre"].fillna(0.0)
    return out


def with_shrunk_delta(frame: pd.DataFrame, delta: str, k: float) -> pd.DataFrame:
    """`shrunk_column(delta)` written onto a copy, at a given shrinkage constant."""
    out = frame.copy()
    out[shrunk_column(delta)] = (out[delta].to_numpy(dtype=float)
                                 * reliability_weight(frame, k))
    return out


# ── The shipped variant, read rather than assumed ─────────────────────────────

def shipped_variants(out_dir: Path, required: list[str] | None = None) -> dict[str, str]:
    """`head -> the variant `make stan-components` selected`, from its own artifact.

    A ladder holding a variant the head does not ship has no incumbent in it, and the
    `crps_vs_incumbent` column would then be measuring two changes at once.
    `minutes_window.assert_shipped_variant` makes the same argument one head over; this is
    the six-headed version, which cannot be a module constant because the six do not agree
    (`log_own_spline` on `fga`/`ast`/`stl`, `log_own` on `tov`/`reb`).
    """
    required = required or [h.name for h in heads()]
    path = Path(out_dir) / "stan_component_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing — run `make stan-components` first. The ladder holds each "
            f"head's SHIPPED variant fixed and reads it from that artifact rather than "
            f"hard-coding it.")
    table = pd.read_csv(path)
    selected = (table[table["selected"]].set_index("head")["variant"].astype(str).to_dict())
    missing = [h for h in required if h not in selected]
    if missing:
        raise ValueError(f"no selected variant in {path} for: {', '.join(missing)}")
    return {h: selected[h] for h in required}


# ── The shrinkage constant, fitted on the fitting half ────────────────────────

def fit_shrinkage(head: Head, train: pd.DataFrame, variant: str,
                  grid: tuple[float, ...] = SHRINKAGE_GRID, n_knots: int = SPLINE_KNOTS,
                  inner: int = INNER_SCORE_SEASONS, seed: int = SEED
                  ) -> tuple[float, pd.DataFrame]:
    """`(selected k, the grid)` — the reliability shrink, chosen inside the fitting half.

    `docs/preseason-plan.md` leaves the volume question open between an empirical-Bayes shrink
    and a reliability interaction, and says the gate decides on train. P3 closed it on the
    minutes head at `k = 20`; whether a per-36 *rate* wants the same is a separate question,
    since a rate over 60 preseason minutes is noisier than a minutes total over the same 60.

    Selection is on CRPS at the head's own unit, because that is what the ladder downstream is
    scored on — a constant chosen against an R² would be optimizing something the gate does
    not read. The inner carve nests inside the *training* frame and validation is never
    materialized.
    """
    years = season_start_year(train)
    cut = sorted(np.unique(years))[-inner]
    fit = train[years < cut].reset_index(drop=True)
    score = train[years >= cut].reset_index(drop=True)

    rows = []
    for k in grid:
        tr, sc, features = head.variants(with_shrunk_delta(fit, head.delta, k),
                                         with_shrunk_delta(score, head.delta, k),
                                         n_knots)[variant]
        extra = [shrunk_column(head.delta)] + MISSING_AGE_COLS
        samples, _, _ = head.fit_predict(tr, sc, features + extra, seed=seed)
        y = head.target(sc[head.live(sc)])
        rows.append({"head": head.name, "k": float(k), "n_fit": len(tr), "n_score": len(sc),
                     "mean_weight": float(reliability_weight(sc, k).mean()),
                     "inner_crps": float(crps_from_samples(samples, y).mean())})
    table = pd.DataFrame(rows)
    return float(table.loc[table["inner_crps"].idxmin(), "k"]), table


# ── The ladder ────────────────────────────────────────────────────────────────

class FittedArm:
    """A fitted arm bound to the validation frame its own basis was built on.

    `minutes_preseason.FittedArm`, one family over. The ladder scores each arm on two
    populations, so what the object carries is the whole validation frame and a caller selects
    rows out of it by position — pairing one arm's coefficients with another's spline basis is
    the failure this class exists to make impossible.
    """

    def __init__(self, name: str, head: Head, train: pd.DataFrame, frame: pd.DataFrame,
                 features: list[str], extra: list[str], samples: np.ndarray,
                 prediction: np.ndarray, dispersion: float):
        self.name, self.head, self.train, self.frame = name, head, train, frame
        self.features, self.extra = features, extra
        self.samples, self.prediction, self.dispersion = samples, prediction, dispersion


def fit_arm(head: Head, train: pd.DataFrame, val: pd.DataFrame, name: str,
            extra: list[str], variant: str, n_knots: int = SPLINE_KNOTS,
            seed: int = SEED) -> FittedArm:
    """One nested arm: the head's shipped variant plus `extra`."""
    tr, va, features = head.variants(train, val, n_knots)[variant]
    samples, prediction, dispersion = head.fit_predict(tr, va, features + extra, seed=seed)
    return FittedArm(name, head, tr, va, features, extra, samples, prediction, dispersion)


def ladder(head: Head, train: pd.DataFrame, val: pd.DataFrame, arms: dict[str, list[str]],
           variant: str, n_knots: int = SPLINE_KNOTS,
           seed: int = SEED) -> dict[str, FittedArm]:
    """Every arm of one head, fitted once on the covered window."""
    fitted = {}
    for name, extra in arms.items():
        arm = fit_arm(head, train, val, name, extra, variant, n_knots, seed)
        fitted[name] = arm
        print(f"    {name:<32} {len(arm.features) + len(extra):>3} features "
              f"({len(extra):>2} preseason)   dispersion {arm.dispersion:.5f}")
    return fitted


def score_arm(head: Head, name: str, arm: FittedArm, mask: np.ndarray, population: str,
              seed: int = SEED) -> tuple[dict, np.ndarray]:
    """One arm's row on one population, and its per-row CRPS for the paired bootstrap.

    `mask` is over the arm's **live** rows, which for a count head is every validation row and
    for a conversion head is the rows with at least one attempt.
    """
    frame = arm.frame[head.live(arm.frame)].reset_index(drop=True)
    y = head.target(frame)[mask]
    samples = arm.samples[:, mask]
    prediction = np.asarray(arm.prediction, dtype=float)[mask]
    if head.kind == "conversion":
        # A conversion head predicts a percentage and draws makes, so its R² and MAE are read
        # on the realized percentage — `stan_components.score_conversion`'s convention, kept
        # so a figure here is comparable to that artifact's.
        n = np.rint(frame[head.attempted].to_numpy(dtype=float)).astype(float)[mask]
        realized, point = y / n, prediction
    else:
        realized, point = y, prediction
    scores = crps_from_samples(samples, y)
    row = {
        "head": head.name, "kind": head.kind, "arm": name, "population": population,
        "n_train": len(arm.train),
        "n_train_seasons": int(pd.Series(arm.train["season"]).nunique()),
        "first_train_season": min(arm.train["season"]),
        "n_features": len(arm.features) + len(arm.extra),
        "n_preseason_cols": len(arm.extra),
        "preseason_cols": "|".join(arm.extra),
        "n_val": int(mask.sum()),
        "val_crps": float(scores.mean()),
        "val_mae": float(np.abs(realized - point).mean()),
        "val_r2": float(1 - ((realized - point) ** 2).sum()
                        / ((realized - realized.mean()) ** 2).sum()),
        "val_bias": float((point - realized).mean()),
        "val_pit_ks": ks_uniform(pit_from_samples(samples, y, seed)),
        "predictive_sd": float(samples.std(axis=0).mean()),
        "realized_sd": float(y.std()),
        "dispersion": float(arm.dispersion),
    }
    return row, scores


def score_population(head: Head, fitted: dict[str, FittedArm], mask: np.ndarray,
                     population: str, floor_crps: float | None = None,
                     seed: int = SEED) -> pd.DataFrame:
    """Every fitted arm of one head, scored on one population of validation rows.

    The arms are **not** refitted per population. The head fits what it fits; P1 decision 5 is
    about where a figure is *read*, and restricting the fitting rows too would confound a
    population statement with a smaller training set.
    """
    rows, per_row = [], {}
    for name, arm in fitted.items():
        row, scores = score_arm(head, name, arm, mask, population, seed)
        rows.append(row)
        per_row[name] = scores

    reference, primary = per_row[REFERENCE_ARM], per_row[PRIMARY_ARM]
    for row in rows:
        delta = paired_bootstrap(per_row[row["arm"]], reference, n_boot=BOOTSTRAP_REPS,
                                 seed=seed)
        row["crps_vs_incumbent"] = delta["crps_delta"]
        row["crps_vs_incumbent_lo"] = delta["ci_lo"]
        row["crps_vs_incumbent_hi"] = delta["ci_hi"]
        # The reference compared against itself has a degenerate [0, 0] interval, which
        # `verdict` would read as a win — the case `minutes_window.ladder` names.
        row["verdict"] = "reference" if row["arm"] == REFERENCE_ARM else verdict(delta)
        row["beats_floor"] = (bool(row["val_crps"] <= floor_crps)
                              if floor_crps is not None else None)
        against = paired_bootstrap(per_row[row["arm"]], primary, n_boot=BOOTSTRAP_REPS,
                                   seed=seed)
        row["crps_vs_primary"] = against["crps_delta"]
        row["crps_vs_primary_lo"] = against["ci_lo"]
        row["crps_vs_primary_hi"] = against["ci_hi"]
        row["verdict_vs_primary"] = ("primary" if row["arm"] == PRIMARY_ARM
                                     else verdict(against))
    return pd.DataFrame(rows)


def floor_row(head: Head, train: pd.DataFrame, val: pd.DataFrame, mask: np.ndarray,
              population: str, seed: int = SEED) -> tuple[dict, float]:
    """The mandatory no-fit carry-forward, on one population."""
    samples, prediction, dispersion = head.floor_predict(train, val, seed=seed)
    arm = FittedArm("carry_forward", head, train, val, [], [], samples, prediction,
                    dispersion)
    row, _ = score_arm(head, "carry_forward", arm, mask, population, seed)
    row["verdict"] = "floor"
    return row, float(row["val_crps"])


# ── Rolling-origin confirmation, fitting half only ────────────────────────────

def rolling_confirmation(head: Head, train: pd.DataFrame, arms: dict[str, list[str]],
                         variant: str, n_knots: int = SPLINE_KNOTS,
                         first_origin: int = FIRST_ORIGIN, seed: int = SEED
                         ) -> pd.DataFrame:
    """Walk-forward over the covered fitting half: one fit per (origin, arm).

    **This is half the gate, not a post-hoc check**, and it is doing more work here than in
    P2 or P3 because six heads are being read at once. Every row touched is a fitting-half
    row; each origin fits on every covered season before it.

    Scored on **both** populations, unlike `minutes_preseason.rolling_confirmation` which
    scores its rolling reading pooled. The fits are shared between the two, so the draftable
    reading — the one P1 decision 5 says the verdict is read on — costs nothing extra, and
    having the pair on the fitting half is what makes the validation pooled/draftable gap
    interpretable rather than a single unreplicated number.
    """
    years = season_start_year(train)
    origins = [int(y) for y in sorted(np.unique(years)) if y >= first_origin]
    slots: dict[tuple[str, str], dict[str, list]] = {}

    for origin in origins:
        score = train[years == origin]
        fit_rows = train[years < origin]
        if score.empty or len(fit_rows) < MIN_FIT_ROWS:
            continue
        tr, sc, features = head.variants(fit_rows, score, n_knots)[variant]
        live = head.live(sc)
        y = head.target(sc[live])
        draftable = sc["on_season_start_roster"].to_numpy(dtype=float)[live] > 0
        for name, extra in arms.items():
            samples, _, _ = head.fit_predict(tr, sc, features + extra, seed=seed)
            crps = crps_from_samples(samples, y)
            pit = pit_from_samples(samples, y, seed)
            sd = samples.std(axis=0)
            for population, mask in (("all", np.ones(len(y), dtype=bool)),
                                     (DECISION_POPULATION, draftable)):
                slot = slots.setdefault((name, population), {})
                slot.setdefault("crps", []).append(crps[mask])
                slot.setdefault("pit", []).append(pit[mask])
                slot.setdefault("sd", []).append(sd[mask])
                slot.setdefault("y", []).append(y[mask])
                slot.setdefault("origin", []).append(np.full(int(mask.sum()), origin))
                slot.setdefault("n_fit", []).append(len(tr))
        print(f"    origin {origin}: fitted {len(tr):,} rows, scored {len(y):,} "
              f"({int(draftable.sum()):,} draftable)")

    pooled = {key: {k: np.concatenate(v) for k, v in d.items() if k != "n_fit"}
              for key, d in slots.items()}
    rows = []
    for population in ("all", DECISION_POPULATION):
        reference = pooled[(REFERENCE_ARM, population)]["crps"]
        primary = pooled[(PRIMARY_ARM, population)]["crps"]
        for name in arms:
            d = pooled[(name, population)]
            origin = d["origin"]
            delta = paired_bootstrap(d["crps"], reference, n_boot=BOOTSTRAP_REPS, seed=seed)
            against = paired_bootstrap(d["crps"], primary, n_boot=BOOTSTRAP_REPS, seed=seed)
            # Origins are the independent replicates, so a win count over them is the
            # multiplicity-robust statement a pooled interval is not.
            wins = sum(1 for o in np.unique(origin)
                       if d["crps"][origin == o].mean() < reference[origin == o].mean())
            primary_wins = sum(1 for o in np.unique(origin)
                               if d["crps"][origin == o].mean() < primary[origin == o].mean())
            rows.append({
                "head": head.name, "kind": head.kind, "arm": name,
                "population": population,
                "n_preseason_cols": len(arms[name]),
                "n_origins": int(len(np.unique(origin))),
                "n_scored": int(len(d["y"])),
                "mean_fit_rows": float(np.mean(slots[(name, population)]["n_fit"])),
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
    return pd.DataFrame(rows)


# ── The gate ──────────────────────────────────────────────────────────────────

def gate(head: str, validation: pd.DataFrame, rolling: pd.DataFrame,
         arm: str = PRIMARY_ARM, population: str = DECISION_POPULATION) -> dict:
    """Both halves of the bar for one head, evaluated as code rather than after the fact.

    P1→P2: the gate is a **conjunction**, and it stays one. P2's round is the standing
    precedent that a bar re-read after seeing which side an arm landed on is not a bar — a
    rolling-only win there was recorded as a failure and shipped separately, by an owner
    decision taken with the failing half in view.
    """
    val = validation[(validation["head"] == head) & (validation["arm"] == arm)
                     & (validation["population"] == population)].iloc[0]
    roll = rolling[(rolling["head"] == head) & (rolling["arm"] == arm)
                   & (rolling["population"] == population)].iloc[0]
    val_pass = bool(val["crps_vs_incumbent_hi"] < 0.0)
    roll_pass = bool(roll["crps_vs_incumbent_hi"] < 0.0
                     and roll["origins_won"] * 2 > roll["n_origins"])

    # Whether any sensitivity or attribution arm beats the declared primary ON THE FITTING
    # HALF, which is where a promotion can be justified without spending the selection split
    # on a choice made after seeing it. P3's rule, and the one that reversed P1 decision 4.
    roll_head = rolling[(rolling["head"] == head)
                        & (rolling["population"] == population)]
    challenger = roll_head[(roll_head["verdict_vs_primary"] == "wins")
                           & (roll_head["origins_won_vs_primary"] * 2
                              > roll_head["n_origins"])]
    best = str(challenger.iloc[0]["arm"]) if len(challenger) else ""
    val_challenger = validation[(validation["head"] == head) & (validation["arm"] == best)
                                & (validation["population"] == population)]

    return {
        "head": head, "arm": arm, "population": population,
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
        "shrinkage_vs_validation": (float(val["crps_vs_incumbent"]
                                          / roll["crps_vs_incumbent"])
                                    if roll["crps_vs_incumbent"] else np.nan),
        "passes": bool(val_pass and roll_pass),
        "earns_stan_port": bool(val_pass and roll_pass),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eda_dir = Path(cfg["eda"]["output_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    seasons = cfg["data"]["seasons"]
    n_knots = int(cfg.get("stan", {}).get("components", {})
                  .get("spline_knots", SPLINE_KNOTS))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))

    print("Component-rate preseason increments (session 6b) — the preseason block as a "
          "nested arm\non the five surviving count heads plus `ftm|fta`")
    print("  The BAR, stated before the run: validation CRPS interval clear of zero on the "
          "draftable\n  population AND the rolling-origin harness agreeing (interval clear "
          "of zero, majority of\n  origins). P1's screen was an R² on a point estimate and "
          "its own text says so — it is a\n  filter for what is worth fitting, never "
          "evidence that anything ships.")

    variants = shipped_variants(out_dir)
    print("\n  Each head's SHIPPED variant, read from stan_component_metrics.csv rather "
          "than hard-coded:")
    print("    " + ", ".join(f"{h} → {v}" for h, v in variants.items()))
    print(f"  ALL {len(heads())} component heads are armed. P1's ΔR² screen admitted six of "
          f"them and its\n  ranking did not survive this ladder in either direction, so the "
          f"five it excluded are\n  measured rather than assumed. The three carrying a "
          f"genuinely NEGATIVE screen —\n  "
          + ", ".join(f"{h} ({v:+.5f})" for h, v in P1_NEGATIVE_HEADS.items())
          + " — are the claim this run tests:\n  a permutation z that negative says the real "
            "block scores worse out of sample than a\n  SHUFFLED one, which is evidence of "
            "harm rather than absence of gain, and this is the\n  instrument that can price "
            "it at a paired interval.")

    # ── the design ───────────────────────────────────────────────────────────
    panel = pd.read_parquet(features_dir / "preseason.parquet")
    coverage = pd.read_csv(eda_dir / "preseason_coverage.csv")
    scope = covered_seasons(coverage)
    first_covered = _season_start_year(scope[0])

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = attach_preseason(build_design(targets, seasons, raw_dir), panel)
    design = attach_season_start_roster(design, seasons, raw_dir, window_games)

    train_full, val = selection_split(design, test_seasons)
    train = restrict_window(train_full, first_covered).reset_index(drop=True)
    val = val.reset_index(drop=True)

    print(f"\n  {len(design):,} player-seasons; {len(train_full):,} fit / {len(val):,} "
          f"select ({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  Preseason coverage begins {scope[0]}, and this design fits from "
          f"{min(train_full['season'])}. So EVERY arm — the\n  reference included — fits "
          f"the covered window only: {len(train):,} of {len(train_full):,} training rows "
          f"({len(train) / len(train_full):.1%}).")
    draftable = val["on_season_start_roster"].to_numpy(dtype=float) > 0
    print(f"  Draftable validation rows (on a season-start roster): {int(draftable.sum()):,}"
          f" of {len(val):,} ({draftable.mean():.1%}) — the >= 200 prior-minute filter "
          f"already removes\n  most of the mid-season-signing population, which is P1's own "
          f"note on this family. "
          f"{val['has_preseason'].mean():.1%} of validation rows carry a preseason row.")

    # ── the ladder, head by head ─────────────────────────────────────────────
    val_blocks, roll_blocks, shrink_blocks, gates = [], [], [], []
    for head in heads():
        variant = variants[head.name]
        print(f"\n── {head.name} ({head.kind}, shipped variant `{variant}`) "
              f"{'─' * max(0, 40 - len(head.name) - len(variant))}")

        k, shrink_table = fit_shrinkage(head, train, variant, n_knots=n_knots, seed=SEED)
        shrink_blocks.append(shrink_table.assign(selected=shrink_table["k"] == k))
        print(f"  shrink `min_pre / (min_pre + k)` on an inner carve of the FITTING half: "
              f"k = {k:.0f}"
              + ("  (the grid's left edge — the shrunk arm IS the primary)" if k <= 0
                 else ""))

        head_train = with_shrunk_delta(train, head.delta, k)
        head_val = with_shrunk_delta(val, head.delta, k)
        arms = preseason_arms(head.delta)
        fitted = ladder(head, head_train, head_val, arms, variant, n_knots, SEED)

        live = head.live(head_val)
        live_draftable = draftable[live]
        for population, mask in (("all", np.ones(int(live.sum()), dtype=bool)),
                                 (DECISION_POPULATION, live_draftable)):
            floor, floor_crps = floor_row(head, head_train, head_val, mask, population,
                                          SEED)
            scored = score_population(head, fitted, mask, population, floor_crps, SEED)
            val_blocks.append(pd.concat([pd.DataFrame([floor]), scored], ignore_index=True))

        # The full-window incumbent, scored as CONTEXT so the coverage restriction's own cost
        # is visible. Not on the ladder: it fits different rows from every other arm, so an
        # interval against it would carry two changes at once.
        context = fit_arm(head, with_shrunk_delta(train_full, head.delta, k), head_val,
                          "incumbent_full_window", [], variant, n_knots, SEED)
        for population, mask in (("all", np.ones(int(live.sum()), dtype=bool)),
                                 (DECISION_POPULATION, live_draftable)):
            row, _ = score_arm(head, "incumbent_full_window", context, mask, population,
                               SEED)
            row["verdict"] = "context"
            val_blocks.append(pd.DataFrame([row]))

        print(f"  rolling-origin confirmation on the COVERED FITTING HALF (origins from "
              f"{FIRST_ORIGIN}):")
        roll_blocks.append(rolling_confirmation(head, head_train, arms, variant, n_knots,
                                               FIRST_ORIGIN, SEED))

    validation = pd.concat(val_blocks, ignore_index=True)
    rolling = pd.concat(roll_blocks, ignore_index=True)
    shrinkage = pd.concat(shrink_blocks, ignore_index=True)

    shrink_dest = out_dir / "components_preseason_shrinkage.csv"
    shrinkage.to_csv(shrink_dest, index=False)
    roll_dest = out_dir / "components_preseason_rolling.csv"
    rolling.to_csv(roll_dest, index=False)

    # ── readout ──────────────────────────────────────────────────────────────
    show = ["head", "arm", "n_preseason_cols", "n_val", "val_crps", "crps_vs_incumbent",
            "crps_vs_incumbent_lo", "crps_vs_incumbent_hi", "crps_vs_primary",
            "val_pit_ks", "val_r2", "beats_floor", "verdict"]
    part = validation[validation["population"] == DECISION_POPULATION]
    print(f"\n\n══ VALIDATION, on a season-start roster — the population the verdict is "
          f"read on ══")
    for head in heads():
        print(f"\n  {head.name}:")
        print(part[part["head"] == head.name][show[1:]].round(4).to_string(index=False))

    print(f"\n══ ROLLING ORIGIN, covered fitting half, draftable rows ══")
    roll_show = ["arm", "mean_fit_rows", "n_scored", "crps", "crps_vs_incumbent",
                 "crps_vs_incumbent_lo", "crps_vs_incumbent_hi", "origins_won",
                 "n_origins", "crps_vs_primary", "origins_won_vs_primary", "pit_ks",
                 "verdict"]
    roll_part = rolling[rolling["population"] == DECISION_POPULATION]
    for head in heads():
        print(f"\n  {head.name}:")
        print(roll_part[roll_part["head"] == head.name][roll_show]
              .round(4).to_string(index=False))

    # ── the gate, head by head ───────────────────────────────────────────────
    print(f"\n\n══ THE GATE on the PRIMARY arm (`{PRIMARY_ARM}`), both halves, per head ══")
    for head in heads():
        result = gate(head.name, validation, rolling, PRIMARY_ARM, DECISION_POPULATION)
        gates.append(result)
        print(f"\n  {head.name}:")
        print(f"    validation      {result['val_crps_delta']:+9.4f} "
              f"[{result['val_lo']:+.4f}, {result['val_hi']:+.4f}]   "
              f"{'PASS' if result['val_pass'] else 'FAIL'}")
        print(f"    rolling origin  {result['rolling_crps_delta']:+9.4f} "
              f"[{result['rolling_lo']:+.4f}, {result['rolling_hi']:+.4f}]   "
              f"{result['origins_won']}/{result['n_origins']} origins   "
              f"{'PASS' if result['rolling_pass'] else 'FAIL'}")
        print(f"    {'GATE PASSES' if result['passes'] else 'GATE FAILS'} — "
              f"`earns_stan_port` = {result['earns_stan_port']}")
        if result["challenger"]:
            print(f"    and `{result['challenger']}` beats the declared primary on the "
                  f"FITTING half: {result['challenger_rolling_delta']:+.4f} "
                  f"[.., {result['challenger_rolling_hi']:+.4f}], "
                  f"validation {result['challenger_val_delta']:+.4f} "
                  f"[.., {result['challenger_val_hi']:+.4f}]")

    passed = [g["head"] for g in gates if g["passes"]]
    print(f"\n  {len(passed)} of {len(gates)} heads clear the conjunction"
          + (f": {', '.join(passed)}" if passed else " — the round is a null."))
    print("  A head that clears both halves earns a Stan port in `stan_components`; these "
          "six sit in\n  the simulator's own draw path, so a survivor is priceable with "
          "`make preseason-contest`\n  at `--groups components`. Nothing is shipped by this "
          "module.")

    gate_rows = pd.DataFrame(gates).assign(arm=lambda d: "gate__" + d["arm"],
                                           verdict="gate")
    dest = out_dir / "components_preseason.csv"
    pd.concat([validation, gate_rows], ignore_index=True).to_csv(dest, index=False)
    print(f"\nWrote {len(validation):,} scored rows + {len(gate_rows)} gate rows → {dest}")
    print(f"Wrote {len(rolling):,} rolling rows → {roll_dest}")
    print(f"Wrote {len(shrinkage):,} shrinkage rows → {shrink_dest}")

    return {"ladder": dest, "rolling": roll_dest, "shrinkage": shrink_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
