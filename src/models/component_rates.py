"""Component rate heads, season-collapsed, benchmarked against a no-fit floor.

Predicts each box-score component's **season total** from prior-season information, using
the collapsed likelihood argued for in `docs/predictions-plan.md`: for a log-link Poisson
with minutes exposure and a linear predictor constant within player-season, the game-level
and season-collapsed likelihoods differ only by a multinomial factor free of β, so fitting
10,194 player-seasons recovers the same coefficients as fitting 731,906 player-games. The
same identity holds for the binomial conversion heads with a hypergeometric factor.

## The benchmark this module exists to enforce

**`carry_forward`: prior per-36 rate × actual minutes / 36. No fitting at all.** It scores
validation R² of **0.82–0.95** across the seven count heads, and the best fitted model here
beats it by only +0.001 to +0.019. Any proposed component head must be quoted against it —
a head that does not clear it is not a model, it is a worse version of arithmetic. This is
the sharpest available statement of the project's "attempts persist" finding (`fg3a` 0.908
in `persistence.csv`) and it says the rate side is close to saturated from prior-season
information alone, which is why availability carries the larger share of season-total error.

`beats_floor` is on every output row, and `run` prints a loud warning for any head that
fails it, because that is the signature of the bug described next.

## The trap that produced a false finding here

`sklearn`'s `PoissonRegressor` minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the data term
is **averaged by the weight sum**. Fitting a rate with `sample_weight = minutes` makes
`Σw ≈ 1e7`, so `alpha=1.0` is an enormous penalty that shrinks every coefficient to nearly
zero, *quietly*: the fit converges and a flexible basis partially compensates, so splines
and interactions appear to buy real signal. Measured on `reb`: R² **0.662 at alpha=1.0
against 0.928 at alpha ≤ 0.01**. `LogisticRegression` is the reverse — `Σ w·logloss +
‖coef‖²/(2C)`, not averaged, so the same weights make `C=1.0` weak. Hence `POISSON_ALPHA`
below, and hence the floor being mandatory rather than optional.

## What the specification should be

**Scale, not curvature.** A log link wants a multiplicative predictor: `log E[rate] =
β·log(prior rate)` gives `rate ∝ prior_rate^β`. Linear-in-raw-rate inside `exp()` is
misspecified, catastrophically for the zero-heavy skewed heads (`fg3a` 0.520, `blk` 0.638
against 0.879 / 0.820 on the log scale). Splines add a further +0.030 (`fg3a`) and +0.041
(`blk`) and ≤ +0.003 elsewhere; `age × own` and `mpg × own` interactions are a null once the
scale is right.

## Which rows everything here is scored on

**Validation**, since 2026-08-05, and this module needed the change more than any other:
it defined its **own** `split_seasons` rather than importing the shared one, so it was the
single head in the project with no held-out guard at all. `src/models/held_out.py` puts the
lock inside `availability.split_seasons` precisely so that a head cannot forget to add it —
and a private copy of the same six lines routes straight around that. The copy was
behaviourally identical, so folding it onto the shared function changes nothing but the
guard.

Everything here is a **sweep**: seven count heads and four conversion heads across up to
seven feature variants each, plus a seven-point regularization grid, all of it ranked. That
is selection, so it belongs on validation. The `alpha_sensitivity` curve is the same in kind
— it exists to catch an over-penalized fit, and catching it on the held-out seasons would
mean the fix was tuned there too.

Usage:
    python -m src.models.component_rates
"""

from dataclasses import dataclass
from pathlib import Path

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.preprocessing import SplineTransformer, StandardScaler

from src.eda.availability import load_ages, with_lags
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.availability import fit_dispersion
from src.models.held_out import selection_split

# The shot-attempt basis: total attempts as a COUNT and the three-point mix as a SHARE,
# rather than two independent attempt counts. A three substitutes for a two, so the two
# counts are not independent, and this basis enforces the substitution by construction
# instead of leaving it for the residual copula. Adopted 2026-08-03 on the Gate 0
# measurement in `docs/shot-attempt-basis-plan.md`: -0.493549 nats per player-season on
# test against the canonical pair at each head's own selected variant, and -0.390814 at
# the two bases' NO-FIT FLOORS, i.e. the coordinate change alone beats the canonical
# basis's best fitted configuration.
#
# The head count is still ELEVEN. `fg2a` stops being a head and becomes DERIVED
# (`fga - fg3a`), exactly as `pts` already is — it is still needed as the *trials* for
# `fg2m | fg2a`, which is why `season_totals` carries it explicitly below.
COUNT_HEADS = ["fga", "fta", "reb", "ast", "stl", "blk", "tov"]
# Ordered as the generative chain reaches them: `fga` is drawn, then the share splits it,
# then `fg2a` falls out, then the makes are drawn on their attempts.
CONVERSION_HEADS = [("fg3a", "fga"), ("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]

# name -> (total, part), with value = total - part. Derived counts are not heads and are
# never fitted; they exist because a later head needs them as trials. `_draw_components`
# materializes them in chain order at draw time.
DERIVED_COUNTS = {"fg2a": ("fga", "fg3a")}
PER36 = 36.0

# See the module docstring: this is *not* a tuning choice, it is the value at which the
# penalty stops dominating a minutes-weighted fit. Anything at or above ~0.01 silently
# destroys the coefficients.
POISSON_ALPHA = 1e-8
LOGISTIC_C = 1e6

MIN_PRIOR_MINUTES = 200      # the project's qualification threshold; reliability ~0.75
TEST_SEASONS = 2
SPLINE_KNOTS = 5
PCA_COMPONENTS = 10

# The prior-season columns the design carries. The own-rate column is per head.
CONTEXT_COLS = ["mpg_lag1", "total_minutes_lag1", "gp_lag1"]
BIO_COLS = ["age", "age_sq", "career_year"]


# ── Design ────────────────────────────────────────────────────────────────────

def volume_columns() -> list[str]:
    """Every column that needs a season total: the count heads, plus every make and every
    set of trials a conversion head references.

    The union matters because **an attempted column need not be a count head**. Under the
    shot-attempt basis `fg2a` is derived rather than fitted, yet `fg2m | fg2a` still needs
    it as trials, and `fg3a` is a *make* (of `fga`) that is simultaneously the trials for
    `fg3m | fg3a`. Deriving the list from both head lists rather than from `COUNT_HEADS`
    alone is what makes the module basis-agnostic — in the two-count basis every attempted
    column is already a count head, so this is exactly the old behaviour there.
    """
    made = [m for m, _ in CONVERSION_HEADS]
    attempted = [a for _, a in CONVERSION_HEADS]
    return list(dict.fromkeys(COUNT_HEADS + made + attempted))


def rate_columns() -> list[str]:
    """Columns that get a per-36 rate — the count heads and every trials column.

    `build_design` asks for `{attempted}_p36_lag1` as each conversion head's volume
    feature, so a trials column that is not a count head still needs its rate.
    """
    return list(dict.fromkeys(COUNT_HEADS + [a for _, a in CONVERSION_HEADS]))


def season_totals(targets: pd.DataFrame) -> pd.DataFrame:
    """Collapse player-games to player-seasons: totals, per-36 rates, conversion pcts."""
    agg = {c: "sum" for c in volume_columns()}
    agg |= {"min": "sum", "played": "sum"}
    out = (targets.groupby(["player_id", "season"], as_index=False).agg(agg)
           .rename(columns={"min": "total_minutes", "played": "gp"}))
    for c in rate_columns():
        out[f"{c}_p36"] = out[c] / out["total_minutes"].replace(0, np.nan) * PER36
    # Under the shot-attempt basis this makes `fg3a_pct` the three-point *share of
    # attempts* (`fg3a / fga`), which is the head's own rate — NOT three-point shooting
    # percentage, which remains `fg3m_pct` (`fg3m / fg3a`). Two different columns, and the
    # naming convention `conversion_variants` uses resolves to the right one for each.
    for m, a in CONVERSION_HEADS:
        out[f"{m}_pct"] = out[m] / out[a].replace(0, np.nan)
    out["mpg"] = out["total_minutes"] / out["gp"].replace(0, np.nan)
    return out


# ── The lag-recovery ladder (docs/rookie-rates-plan.md §5b) ───────────────────

# The rungs the ladder can admit, named as `lag_recovery.classify` names them so the
# design's own `lag_rung` column reconciles with §7b's board census without a mapping
# table. `veteran` is rung 0 and is not listed: it is the shipped design and is always in.
LADDER_RUNGS = ("thin_prior", "returnee_lag2", "returnee_thin", "no_usable_lag")

# The provenance a recovered row carries. Present only when the ladder is on, so a
# ladder-off frame is byte-for-byte the frame every one of the twelve importers has
# always received.
LADDER_COLS = ("lag_source", "lag_rung", "lag_minutes")


@dataclass(frozen=True)
class LagLadder:
    """The admitted rungs plus the constants the imputation needs, all fitted elsewhere.

    Nothing here is estimated at build time and that is deliberate: `build_design` is
    called by twelve modules on frames that are sometimes a single forward season, so a
    constant fitted from whatever rows happen to be in front of it would differ between
    callers. `lag_recovery.ladder_constants` reads them from the artifact that fitted them
    on the fitting half, which is the `components_preseason_shrinkage.csv` precedent.
    """

    rungs: tuple[str, ...]
    shrinkage: Mapping[str, float]                    # per rate column, prior-season minutes
    means: Mapping[str, float]                        # per rate column, fitting-half per-36
    conversion: Mapping[str, tuple[float, float]]     # per made column, (pseudo-attempts, league)
    max_lag: int = 3


def lag_ladder(cfg: dict, rungs: Sequence[str] | None = None) -> LagLadder | None:
    """The configured ladder, or `None` for the shipped pre-ladder design.

    `stan.components.lag_ladder` is a **list of admitted rungs** rather than a boolean,
    because §4's gate is per rung and came back per rung: an empty list is today's design
    exactly, and the key is what the gate's verdict is written into. `rungs` overrides it
    for the gate itself, which has to build the widest design in order to score the rungs
    it is deciding about.
    """
    from src.models.lag_recovery import ladder_constants

    configured = (cfg.get("stan", {}).get("components", {}).get("lag_ladder")
                  if rungs is None else rungs)
    admitted = tuple(configured or ())
    if not admitted:
        return None
    unknown = [r for r in admitted if r not in LADDER_RUNGS]
    if unknown:
        raise ValueError(f"`stan.components.lag_ladder` names unknown rung(s) {unknown}; "
                         f"the vocabulary is {list(LADDER_RUNGS)}")
    path = Path(cfg["evaluation"]["predictions_dir"]) / "lag_recovery.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing and the ladder's shrinkage is fitted there — run "
            f"`make lag-recovery`, or set `stan.components.lag_ladder: []` to build the "
            f"pre-ladder design exactly.")
    return LagLadder(rungs=admitted, **ladder_constants(path))


def apply_lag_ladder(d: pd.DataFrame, ladder: LagLadder) -> pd.DataFrame:
    """Fill an unusable lag-1 block from the nearest usable season, in place of it.

    **Imputation only.** The recovered row is written into the *same* `_lag1` columns the
    shipped heads already fit coefficients on, so those coefficients score it unchanged and
    no head is refitted — §3 constraint 4, which is the whole reason the ladder takes this
    form rather than admitting the rows to the fit with a staleness feature.

    Three things are carried, and they are carried differently because they are different
    kinds of quantity:

    - **rates** (`{c}_p36_lag1`) are shrunk toward the fitting-half population mean by the
      source season's own reliability weight `m / (m + k)` — `lag_recovery.recent_shrunk_rate`
      verbatim. §7b is emphatic that the shrink is not optional: raw thin lag-1 scores a
      pooled R² of −1.1451 and is an anti-model, against +0.8566 shrunk.
    - **conversion percentages** (`{m}_pct_lag1`) are empirical-Bayes shrunk on their own
      *attempts*, because that is where a proportion's reliability lives — the form
      `carry_forward_conversion` already settled.
    - **volumes and context** (`mpg_lag1`, `total_minutes_lag1`, `gp_lag1`, every
      `{c}_lag1`) are carried **raw**. They are not estimates of a per-minute quantity;
      they are what the player's most recent season actually was, and `lag_source` records
      how stale that is.

    ## The threshold discontinuity is named, not solved

    Qualification is tested on the **pre-imputation** lag-1, so a player at 201 prior
    minutes keeps a raw rate and one at 199 gets a shrunk one — at the fitted `k` the jump
    in weight is about 0.2. Removing it means shrinking everyone continuously, which
    changes the *fitting* population and costs eleven refits at two windows. That is the
    escalation §3 constraint 4 names; this function is the cheap form it gates first.
    """
    from src.models.lag_recovery import classify, recent_conversion, recent_lag, \
        recent_shrunk_rate

    prior = d["total_minutes_lag1"].to_numpy(dtype=float)
    # Computed BEFORE anything is written, which is what makes rung 0 bit-identical: the
    # test is the shipped one, on the column the shipped design read.
    qualified = np.isfinite(prior) & (prior >= MIN_PRIOR_MINUTES)
    lag = recent_lag(d, ladder.max_lag)
    recovered = ~qualified & (lag >= 1)

    # `classify`'s six-way label, with the design's own boundary winning rung 0 — the two
    # agree on every real row (a qualified lag-1 implies a prior season, so no qualified
    # row can be a first-season player), and pinning it here means bit-identity does not
    # depend on that argument staying true.
    rung = np.where(qualified, "veteran", classify(d).to_numpy())

    # Every shrunk column is computed on the untouched frame first: `recent_lag` reads
    # `total_minutes_lag1`, so writing the carried block before reading the rates would
    # make every lag-2 row look like a lag-1 row.
    shrunk = {c: recent_shrunk_rate(d, c, ladder.shrinkage[c], ladder.means[c])
              for c in rate_columns()}
    pct = {m: recent_conversion(d, m, a, *ladder.conversion[m])
           for m, a in CONVERSION_HEADS}

    carry = list(dict.fromkeys(volume_columns() + ["mpg", "total_minutes", "gp"]))
    carried = {c: d[f"{c}_lag1"].to_numpy(dtype=float).copy() for c in carry}
    source_minutes = np.full(len(d), np.nan)
    for L in range(1, ladder.max_lag + 1):
        hit = lag == L
        if not hit.any():
            continue
        source_minutes[hit] = d[f"total_minutes_lag{L}"].to_numpy(dtype=float)[hit]
        for c in carry:
            carried[c][hit] = d[f"{c}_lag{L}"].to_numpy(dtype=float)[hit]

    out = d.copy()
    for c in carry:
        out[f"{c}_lag1"] = np.where(recovered, carried[c],
                                    d[f"{c}_lag1"].to_numpy(dtype=float))
    for c in rate_columns():
        out[f"{c}_p36_lag1"] = np.where(recovered, shrunk[c],
                                        d[f"{c}_p36_lag1"].to_numpy(dtype=float))
    for m, _ in CONVERSION_HEADS:
        out[f"{m}_pct_lag1"] = np.where(recovered, pct[m],
                                        d[f"{m}_pct_lag1"].to_numpy(dtype=float))

    # Provenance on EVERY row, not only the recovered ones: a board that cannot say a unit
    # was scored off a two-year-old season cannot be audited, and "the column is absent"
    # is not the same statement as "the row came from lag 1". The per-head reliability
    # weight is `lag_minutes / (lag_minutes + k)` at that head's `k` in `lag_recovery.csv`
    # — one row-level column and nine constants, rather than nine redundant columns.
    out["lag_source"] = np.where(lag >= 1, np.char.add("lag", lag.astype(str)), "none")
    out["lag_rung"] = rung
    out["lag_minutes"] = source_minutes
    return out


def ladder_recovered(design: pd.DataFrame) -> np.ndarray:
    """Which rows the ladder admitted that the shipped design would have dropped."""
    if "lag_rung" not in design.columns:
        return np.zeros(len(design), dtype=bool)
    return (design["lag_rung"].to_numpy() != "veteran")


def fitting_rows(design: pd.DataFrame) -> pd.DataFrame:
    """Rung 0 only — the fitting population, which the ladder does not widen.

    §3 constraint 4 in one function. The ladder is a **scoring** change: the recovered rows
    are imputed into columns the existing coefficients already read, and admitting them to
    the fit as well would be the escalation the constraint holds in reserve. Every caller
    that fits rather than scores routes its training frame through here.
    """
    return design[~ladder_recovered(design)]


def build_design(targets: pd.DataFrame, seasons: list[str], raw_dir: str | Path,
                 forward_seasons: Sequence[str] = (),
                 ladder: "LagLadder | None" = None) -> pd.DataFrame:
    """One row per (player, target season) with lag-1 prior-season columns.

    Rows need a prior season of at least `MIN_PRIOR_MINUTES`, so every own-rate feature is
    measured over enough minutes to be worth something, and a current season with minutes,
    since minutes are the exposure.

    ## `ladder`, and why the default is `None`

    `None` builds the pre-ladder design **exactly** — `max_lag=1`, no provenance columns,
    the same prior-minutes test — because this builder is how twelve other modules reach
    their rows and none of them asked for a wider population. `lag_ladder(cfg)` turns
    `stan.components.lag_ladder` into the object; a caller that wants the ladder passes it
    by name, and `apply_lag_ladder` states what it does to the block.

    Rung 0 is bit-identical under both settings and `tests/test_component_rates.py` pins
    it, which is the only claim that lets §3 constraint 4 hold: the recovered rows are
    scored by coefficients fitted on a population that did not move.

    ## `forward_seasons`, and why the second condition cannot apply to one

    `total_minutes > 0` is the **target** season's minutes, and it is a fitting-population
    rule: a player-game with no minutes contributes nothing to a count likelihood whose
    exposure *is* minutes. A season that has not been played has no minutes for anybody, so
    the rule empties it — which is correct when fitting and wrong when building the design
    a production board is scored from.

    Naming a season here exempts it from that condition **only**. The prior-minutes
    qualification still binds, because it is a statement about the features rather than
    about the target, and it is exactly as knowable in September as in June.

    **The default is empty**, so a fitting path cannot receive forward rows without asking
    for them by name. `features/targets.build_component_targets` carries the same knob for
    the same reason and has to be told too, since it drops blank-minute rows first — two
    stacked population filters, and a caller that clears one and not the other simply gets
    nothing, which is the safe way for a half-applied change to fail.
    """
    s = season_totals(targets)
    lag_cols = ([f"{c}_p36" for c in rate_columns()]
                + [f"{m}_pct" for m, _ in CONVERSION_HEADS]
                + volume_columns()
                + ["mpg", "total_minutes", "gp"])
    lag_cols = list(dict.fromkeys(lag_cols))
    d = with_lags(s, seasons, lag_cols, max_lag=1 if ladder is None else ladder.max_lag)

    ages = load_ages(seasons, raw_dir)
    d = (d.merge(ages, on=["season", "player_id"], how="left") if not ages.empty
         else d.assign(age=np.nan))
    d["age_sq"] = d["age"] ** 2
    d["career_year"] = d.sort_values("season_index").groupby("player_id").cumcount()

    if ladder is not None:
        # `classify` needs to know a player's first played season, for the reason it tests
        # that first: a player whose history falls outside the window has no lag columns
        # either, and calling him a rookie would hand the rookie head a row it cannot
        # serve. Merged from `s` — every season on the frame, not just the design's.
        first = s.groupby("player_id")["season"].min().rename("first_season")
        d = apply_lag_ladder(d.merge(first, on="player_id", how="left"), ladder)

    # Before the qualification test, and unchanged by the ladder: a recovered row has had
    # its `mpg_lag1` and `total_minutes_lag1` filled from the source season by now, so it
    # survives here for the same reason a veteran does.
    d = d.dropna(subset=["age", "mpg_lag1", "total_minutes_lag1"])
    if ladder is None:
        qualified = d["total_minutes_lag1"] >= MIN_PRIOR_MINUTES
    else:
        # The single prior-minutes test becomes the rung admission the gate decided.
        # `veteran` IS that test, computed pre-imputation, so this is a widening and never
        # a substitution — no row today's design carries can fail it.
        qualified = d["lag_rung"].isin(("veteran",) + tuple(ladder.rungs))
    if forward_seasons:
        forward = d["season"].isin(set(forward_seasons))
        d = d[qualified & ((d["total_minutes"] > 0) | forward)]
        d = d.assign(is_forward=forward[d.index].astype(int))
    else:
        d = d[qualified & (d["total_minutes"] > 0)]
    return d.reset_index(drop=True)


#
# `split_seasons` used to be defined here, and that is why this module was the only head in
# the project with no held-out guard at all: `src/models/availability.py::split_seasons` is
# the choke point `held_out.py` wraps, and a private copy routes around it. The two were
# otherwise **behaviourally identical** — same `sorted(unique)`, same trailing-`n` slice,
# same two frames — so deleting the copy changes nothing except that the held-out side is
# now guarded. Nothing is lost with it.


# ── The benchmark ─────────────────────────────────────────────────────────────

def carry_forward(frame: pd.DataFrame, component: str) -> np.ndarray:
    """The no-fit floor: prior per-36 rate × actual minutes / 36.

    No parameters, no fitting, no training data. Everything else in this module has to
    beat it, and most of the achievable skill is already here.
    """
    rate = frame[f"{component}_p36_lag1"].to_numpy(dtype=float)
    return np.clip(rate * frame["total_minutes"].to_numpy(dtype=float) / PER36, 0.0, None)


def carry_forward_conversion(train: pd.DataFrame, frame: pd.DataFrame, made: str,
                             attempted: str) -> np.ndarray:
    """The conversion floor: prior percentage **shrunk** toward the training league mean.

    The raw carry-forward is not a usable floor here and that is a fact about proportions,
    not a coding choice. A player who went 0-for-3 from three has a prior 3P% of exactly
    0.000; carrying it forward onto 200 attempts gives a beta-binomial NLL of ~1.3e9 and
    makes the benchmark meaningless. `CLAUDE.md` already settles the fix — conversion
    percentages must be **shrunk hard toward player/league means**, unlike the share and
    count columns. So the floor is empirical-Bayes:

        p_hat = (made_{S-1} + k·league_mean) / (attempts_{S-1} + k)

    with `k` pseudo-attempts and `league_mean` both estimated on **train** only. Still no
    regression and no features — one shrinkage constant — which is the honest proportion
    analogue of "carry the rate forward".
    """
    lag_made, lag_att = f"{made}_lag1", f"{attempted}_lag1"
    m_train = train[lag_made].to_numpy(dtype=float)
    a_train = train[lag_att].to_numpy(dtype=float)
    ok = np.isfinite(m_train) & np.isfinite(a_train) & (a_train > 0)
    league = float(m_train[ok].sum() / a_train[ok].sum())

    def nll_for(k: float) -> float:
        p = (m_train[ok] + k * league) / (a_train[ok] + k)
        y = train[made].to_numpy(dtype=float)[ok]
        n = train[attempted].to_numpy(dtype=float)[ok]
        live = n > 0
        rho = fit_dispersion(y[live].astype(int), n[live].astype(int), p[live])
        return float(beta_binomial_nll(y[live].astype(int), n[live].astype(int),
                                       p[live], rho))

    best = minimize_scalar(nll_for, bounds=(1.0, 2000.0), method="bounded")
    k = float(best.x)
    m = frame[lag_made].to_numpy(dtype=float)
    a = frame[lag_att].to_numpy(dtype=float)
    p = np.where(np.isfinite(m) & np.isfinite(a),
                 (np.nan_to_num(m) + k * league) / (np.nan_to_num(a) + k), league)
    return np.clip(p, 1e-3, 1 - 1e-3)


# ── Feature construction ──────────────────────────────────────────────────────

def impute(train: pd.DataFrame, test: pd.DataFrame,
           cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Fill NaN from **train** means and flag it.

    A player with no prior 3PA has a genuinely undefined prior 3P%; the indicator keeps
    that distinguishable from "league-average shooter", and splines cannot take NaN.
    """
    tr, te, flags = train.copy(), test.copy(), []
    for col in cols:
        if not train[col].isna().any() and not test[col].isna().any():
            continue
        mean = float(train[col].mean())
        flag = f"{col}__miss"
        tr[flag] = train[col].isna().astype(float)
        te[flag] = test[col].isna().astype(float)
        tr[col], te[col] = train[col].fillna(mean), test[col].fillna(mean)
        flags.append(flag)
    return tr, te, flags


def add_log(train: pd.DataFrame, test: pd.DataFrame,
            cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """`log1p` of a non-negative column — the multiplicative scale a log link wants."""
    tr, te, names = train.copy(), test.copy(), []
    for col in cols:
        name = f"log_{col}"
        for frame, src in ((tr, train), (te, test)):
            frame[name] = np.log1p(np.clip(src[col].to_numpy(dtype=float), 0.0, None))
        names.append(name)
    return tr, te, names


def add_spline(train: pd.DataFrame, test: pd.DataFrame, cols: list[str],
               n_knots: int = SPLINE_KNOTS
               ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Cubic B-spline basis with linear extrapolation, knots from **train** quantiles."""
    tr, te, names = train.copy(), test.copy(), []
    for col in cols:
        st = SplineTransformer(n_knots=n_knots, degree=3, extrapolation="linear",
                               include_bias=False)
        basis_tr = st.fit_transform(train[[col]].to_numpy(dtype=float))
        basis_te = st.transform(test[[col]].to_numpy(dtype=float))
        for j in range(basis_tr.shape[1]):
            name = f"{col}__s{j}"
            tr[name], te[name] = basis_tr[:, j], basis_te[:, j]
            names.append(name)
    return tr, te, names


def add_interactions(train: pd.DataFrame, test: pd.DataFrame,
                     pairs: list[tuple[str, str]]
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr, te, names = train.copy(), test.copy(), []
    for a, b in pairs:
        name = f"{a}__x__{b}"
        tr[name] = tr[a].to_numpy(dtype=float) * tr[b].to_numpy(dtype=float)
        te[name] = te[a].to_numpy(dtype=float) * te[b].to_numpy(dtype=float)
        names.append(name)
    return tr, te, names


# ── Walk-forward PCA ──────────────────────────────────────────────────────────

def matrix_feature_cols(matrix: pd.DataFrame) -> list[str]:
    """Numeric season-matrix feature columns, excluding keys and volume."""
    drop = {"player_id", "season", "season_start_year", "team_id", "gp", "min",
            "reliability", "stats_source", "archetype", "draft_bucket",
            "team_abbreviation", "player_name", "roster_coverage"}
    return [c for c in matrix.columns
            if c not in drop and pd.api.types.is_numeric_dtype(matrix[c])]


def walk_forward_pca(design: pd.DataFrame, matrix: pd.DataFrame,
                     n_components: int = PCA_COMPONENTS
                     ) -> tuple[pd.DataFrame, list[str]]:
    """PC scores of each row's **prior-season** stat vector, fitted point-in-time.

    For a target season S the features describe season S-1, and everything through S-1 is
    observable at the prediction date — so the scaler and the PCA are refitted for every
    target season on matrix rows from **S-1 and earlier only**. Fitting one PCA on all 30
    seasons would leak the future into the basis, which is the same class of error as
    filling a historical row from a current-status source, and it is invisible in the
    output: the scores look completely normal and simply score too well.

    This is why the repo's cached `pca_features_tier*.pkl` artifacts cannot be used here —
    they are fitted pooled or within-season across the whole sample.
    """
    feats = matrix_feature_cols(matrix)
    m = matrix[["player_id", "season_start_year"] + feats].copy()
    m[feats] = m[feats].apply(pd.to_numeric, errors="coerce")
    m = m.dropna(subset=["season_start_year"])

    design = design.copy()
    design["prior_start_year"] = design["season_start_year"] - 1
    pc_names = [f"pc{i + 1}" for i in range(n_components)]
    out = []
    for target_year, block in design.groupby("season_start_year", sort=True):
        prior_year = int(target_year) - 1
        history = m[m["season_start_year"] <= prior_year]
        rows = block.merge(
            m[m["season_start_year"] == prior_year],
            left_on=["player_id", "prior_start_year"],
            right_on=["player_id", "season_start_year"],
            how="left", suffixes=("", "_m"))
        if len(history) <= n_components or rows[feats].isna().all(axis=None):
            continue

        # The imputation median is a FITTED QUANTITY, exactly like the scaler and the
        # PCA, and it belongs inside this loop for the same reason. It used to be taken
        # once over the whole matrix before the loop — so a 1998 row's missing column was
        # filled from a median that had seen 2025. Small in effect (the PCA arm is a
        # measured +/-0.003 null) and precisely the class of error this function's
        # docstring exists to warn about, which is why it is fixed rather than noted.
        # A column with no history at all leaves a NaN median; standardizing a constant
        # gives 0 in sklearn, so the 0.0 fallback contributes nothing rather than
        # dropping the row through `keep`.
        medians = history[feats].median()
        history = history.assign(**{c: history[c].fillna(medians[c]).fillna(0.0)
                                    for c in feats})
        scaler = StandardScaler().fit(history[feats].to_numpy(dtype=float))
        pca = PCA(n_components=n_components, random_state=0).fit(
            scaler.transform(history[feats].to_numpy(dtype=float)))

        # Only rows that actually matched a prior-season row are imputed. An unmatched
        # row must stay NaN so `keep` drops it — filling it would invent a whole
        # league-median stat line for a player who has none.
        matched = rows["season_start_year_m"].notna().to_numpy()
        filled = rows[feats].fillna(medians).fillna(0.0).to_numpy(dtype=float)
        X = np.where(matched[:, None], filled, np.nan)
        keep = ~np.isnan(X).any(axis=1)
        scores = np.full((len(rows), n_components), np.nan)
        if keep.any():
            scores[keep] = pca.transform(scaler.transform(X[keep]))
        block = block.copy()
        for j, name in enumerate(pc_names):
            block[name] = scores[:, j]
        out.append(block)
    if not out:
        raise ValueError("walk-forward PCA produced no rows; check season_start_year")
    return pd.concat(out, ignore_index=True), pc_names


# ── Heads ─────────────────────────────────────────────────────────────────────

def _design_matrix(frame: pd.DataFrame, features: list[str]) -> np.ndarray:
    X = frame[features].to_numpy(dtype=float)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def nb_nll(y: np.ndarray, mu: np.ndarray, phi: float) -> np.ndarray:
    """Negative binomial (NB2) negative log-likelihood, `var = mu + mu^2/phi`."""
    mu = np.clip(mu, 1e-9, None)
    return -(gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
             + phi * np.log(phi / (phi + mu))
             + y * np.log(np.where(y > 0, mu / (phi + mu), 1.0)))


def fit_nb_dispersion(y: np.ndarray, mu: np.ndarray) -> float:
    r = minimize_scalar(lambda lp: nb_nll(y, mu, float(np.exp(lp))).sum(),
                        bounds=(-3.0, 8.0), method="bounded")
    return float(np.exp(r.x))


def fit_count_head(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                   component: str, alpha: float = POISSON_ALPHA
                   ) -> tuple[np.ndarray, float]:
    """Poisson rate GLM with minutes exposure; returns test means and NB dispersion.

    Exposure is handled by the standard identity — fitting `y/M` with `sample_weight = M`
    is exactly a Poisson with offset `log M`.
    """
    m_train = train["total_minutes"].to_numpy(dtype=float)
    m_test = test["total_minutes"].to_numpy(dtype=float)
    y_train = train[component].to_numpy(dtype=float)
    scaler = StandardScaler().fit(_design_matrix(train, features))
    model = PoissonRegressor(alpha=alpha, max_iter=5000).fit(
        scaler.transform(_design_matrix(train, features)), y_train / m_train,
        sample_weight=m_train)
    mu_train = np.clip(
        model.predict(scaler.transform(_design_matrix(train, features))) * m_train,
        1e-6, None)
    mu_test = np.clip(
        model.predict(scaler.transform(_design_matrix(test, features))) * m_test,
        1e-6, None)
    return mu_test, fit_nb_dispersion(y_train, mu_train)


def fit_conversion_head(train: pd.DataFrame, test: pd.DataFrame, features: list[str],
                        made: str, attempted: str, C: float = LOGISTIC_C
                        ) -> tuple[np.ndarray, np.ndarray, float]:
    """Grouped-binomial logistic GLM; returns the test mask, probabilities and rho.

    A grouped binomial is exactly a weighted Bernoulli on two rows per group — one with
    weight `made`, one with weight `attempted - made` — which is what lets `sklearn`'s
    solver fit it without expanding to one row per shot.
    """
    n_train = train[attempted].to_numpy(dtype=float)
    y_train = train[made].to_numpy(dtype=float)
    ok_train = n_train > 0
    ok_test = test[attempted].to_numpy(dtype=float) > 0

    scaler = StandardScaler().fit(_design_matrix(train, features)[ok_train])
    X = scaler.transform(_design_matrix(train, features)[ok_train])
    X2 = np.vstack([X, X])
    labels = np.r_[np.ones(int(ok_train.sum())), np.zeros(int(ok_train.sum()))]
    weights = np.r_[y_train[ok_train], (n_train - y_train)[ok_train]]
    keep = weights > 0
    model = LogisticRegression(C=C, max_iter=5000).fit(X2[keep], labels[keep],
                                                       sample_weight=weights[keep])
    p_train = model.predict_proba(X)[:, 1]
    p_test = model.predict_proba(
        scaler.transform(_design_matrix(test, features)[ok_test]))[:, 1]
    rho = fit_dispersion(y_train[ok_train].astype(int), n_train[ok_train].astype(int),
                         p_train)
    return ok_test, p_test, rho


# ── Evaluation ────────────────────────────────────────────────────────────────

def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def count_variants(train: pd.DataFrame, test: pd.DataFrame, component: str,
                   pc_names: list[str] | None = None
                   ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """Feature sets to compare for one count head, raw and PCA."""
    own = f"{component}_p36_lag1"
    base = [own] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    out = {"linear": (tr, te, list(base))}

    tl, el, log_names = add_log(tr, te, [own])
    log_base = [c for c in base if c != own] + log_names
    out["log_own"] = (tl, el, list(log_base))

    ts, es, spline_names = add_spline(tr, te, [own])
    out["log_own_spline"] = (ts, es, [c for c in base if c != own] + spline_names)

    ti, ei, inter_names = add_interactions(tl, el, [("age", log_names[0]),
                                                   ("mpg_lag1", log_names[0])])
    out["log_own_inter"] = (ti, ei, log_base + inter_names)

    if pc_names:
        # PCA replaces every prior-season stat EXCEPT the head's own response variable.
        pca_base = [c for c in log_base if c not in CONTEXT_COLS] + pc_names
        tp, ep, _ = impute(tl, el, pc_names)
        out["pca"] = (tp, ep, list(pca_base))
        tps, eps, spl = add_spline(tp, ep, [own]) if own in tp else (tp, ep, [])
        out["pca_spline"] = (tps, eps,
                             [c for c in pca_base if c not in log_names] + spl)
        tpi, epi, pinter = add_interactions(
            tp, ep, [(log_names[0], pc_names[0]), (log_names[0], pc_names[1])])
        out["pca_inter"] = (tpi, epi, pca_base + pinter)
    return out


def conversion_variants(train: pd.DataFrame, test: pd.DataFrame, made: str,
                        attempted: str, pc_names: list[str] | None = None
                        ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    own = f"{made}_pct_lag1"
    vol = f"{attempted}_p36_lag1"
    base = [own, vol] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    out = {"linear": (tr, te, list(base))}
    ts, es, spline_names = add_spline(tr, te, [own])
    out["spline_own"] = (ts, es, [c for c in base if c != own] + spline_names)
    ti, ei, inter = add_interactions(tr, te, [("mpg_lag1", own), (vol, own)])
    out["inter"] = (ti, ei, base + inter)
    if pc_names:
        tp, ep, _ = impute(tr, te, pc_names)
        pca_base = [c for c in base if c not in CONTEXT_COLS + [vol]] + pc_names
        out["pca"] = (tp, ep, list(pca_base))
        tpi, epi, pinter = add_interactions(tp, ep, [(own, pc_names[0])])
        out["pca_inter"] = (tpi, epi, pca_base + pinter)
    return out


def evaluate(train: pd.DataFrame, test: pd.DataFrame,
             pc_names: list[str] | None = None) -> pd.DataFrame:
    """Every head × variant, with the no-fit floor always present as a row."""
    rows = []
    for component in COUNT_HEADS:
        y = test[component].to_numpy(dtype=float)
        floor = carry_forward(test, component)
        floor_r2 = _r2(y, floor)
        floor_phi = fit_nb_dispersion(train[component].to_numpy(dtype=float),
                                      np.clip(carry_forward(train, component), 1e-6, None))
        rows.append({"analysis": "variant_sweep", "head": component,
                     "kind": "count", "variant": "carry_forward",
                     "n_features": 0, "r2": floor_r2,
                     "mae": float(np.abs(y - floor).mean()),
                     "nll": float(nb_nll(y, np.clip(floor, 1e-6, None), floor_phi).mean()),
                     "dispersion": floor_phi, "beats_floor": True})
        for label, (tr, te, features) in count_variants(train, test, component,
                                                        pc_names).items():
            mu, phi = fit_count_head(tr, te, features, component)
            r2 = _r2(y, mu)
            rows.append({"analysis": "variant_sweep", "head": component,
                         "kind": "count", "variant": label,
                         "n_features": len(features), "r2": r2,
                         "mae": float(np.abs(y - mu).mean()),
                         "nll": float(nb_nll(y, mu, phi).mean()),
                         "dispersion": phi, "beats_floor": bool(r2 > floor_r2)})

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        n_test = test[attempted].to_numpy(dtype=float)
        y_test = test[made].to_numpy(dtype=float)
        ok = n_test > 0
        p_floor = carry_forward_conversion(train, test, made, attempted)[ok]
        live = train[attempted].to_numpy(dtype=float) > 0
        rho_floor = fit_dispersion(
            train[made].to_numpy(dtype=float)[live].astype(int),
            train[attempted].to_numpy(dtype=float)[live].astype(int),
            carry_forward_conversion(train, train, made, attempted)[live])
        floor_nll = float(beta_binomial_nll(y_test[ok].astype(int), n_test[ok].astype(int),
                                           p_floor, rho_floor) / int(ok.sum()))
        realised = y_test[ok] / n_test[ok]
        rows.append({"analysis": "variant_sweep", "head": head,
                     "kind": "conversion", "variant": "carry_forward",
                     "n_features": 0, "r2": _r2(realised, p_floor),
                     "mae": float(np.abs(realised - p_floor).mean()),
                     "nll": floor_nll, "dispersion": rho_floor, "beats_floor": True})
        for label, (tr, te, features) in conversion_variants(train, test, made, attempted,
                                                             pc_names).items():
            mask, p, rho = fit_conversion_head(tr, te, features, made, attempted)
            nll = float(beta_binomial_nll(y_test[mask].astype(int),
                                          n_test[mask].astype(int), p, rho)
                        / int(mask.sum()))
            rows.append({"analysis": "variant_sweep", "head": head,
                         "kind": "conversion", "variant": label,
                         "n_features": len(features),
                         "r2": _r2(y_test[mask] / n_test[mask], p),
                         "mae": float(np.abs(y_test[mask] / n_test[mask] - p).mean()),
                         "nll": nll, "dispersion": rho, "beats_floor": bool(nll < floor_nll)})
    return pd.DataFrame(rows)


# ── Regularization sensitivity ────────────────────────────────────────────────
#
# The `sklearn` alpha trap cost a full set of published figures, and it failed *quietly*:
# `PoissonRegressor` averages the deviance by the weight sum, so with `sample_weight =
# minutes` (Sum w ~ 1e7) `alpha=1.0` crushes every coefficient while the fit still converges
# and a flexible basis partially compensates. `reb` read R² 0.662 instead of 0.928, and the
# spline and interaction variants looked like they were buying real signal.
#
# So the trap gets a curve rather than an anecdote, and the curve is a regression guard: a
# future refactor that reintroduces a default-looking alpha shows up as the high-alpha end of
# the fitted line dropping below the no-fit floor. That crossing is the visual statement of
# why the floor is mandatory — arithmetic beats an over-penalized GLM.

# Spans the correct and the broken regimes. `POISSON_ALPHA` sits at the bottom; 1.0 is the
# value that looks like a default and is not one.
ALPHA_GRID = [1e-8, 1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]

# Swept on both the misspecified raw-rate spec (which is what the recorded 0.662 was measured
# on) and the shipped log spec, because "does over-regularization hurt the spec we ship" is
# the question the guard actually needs to answer.
ALPHA_VARIANTS = ("linear", "log_own")


def alpha_feature_sets(train: pd.DataFrame, test: pd.DataFrame, component: str
                       ) -> dict[str, tuple[pd.DataFrame, pd.DataFrame, list[str]]]:
    """The two feature sets the alpha sweep runs on — no splines, no PCA.

    A deliberately trimmed `count_variants`: the sweep is about the penalty, so building a
    spline basis and a walk-forward PCA for every alpha would cost time to measure nothing.
    """
    own = f"{component}_p36_lag1"
    base = [own] + CONTEXT_COLS + BIO_COLS
    tr, te, flags = impute(train, test, base)
    base = base + flags
    tl, el, log_names = add_log(tr, te, [own])
    return {"linear": (tr, te, list(base)),
            "log_own": (tl, el, [c for c in base if c != own] + log_names)}


def alpha_sensitivity(train: pd.DataFrame, test: pd.DataFrame,
                      alphas: list[float] = None, heads: list[str] = None,
                      variants: tuple[str, ...] = ALPHA_VARIANTS) -> pd.DataFrame:
    """Validation R² per count head across an alpha grid, with the no-fit floor beside it.

    `floor_r2` rides on every row so the crossing is readable without a join, and
    `beats_floor` keeps the same meaning it has everywhere else in this module.
    """
    alphas = list(ALPHA_GRID if alphas is None else alphas)
    heads = list(COUNT_HEADS if heads is None else heads)
    rows = []
    for component in heads:
        y = test[component].to_numpy(dtype=float)
        floor_r2 = _r2(y, carry_forward(test, component))
        sets = alpha_feature_sets(train, test, component)
        for variant in variants:
            if variant not in sets:
                continue
            tr, te, features = sets[variant]
            for alpha in alphas:
                mu, phi = fit_count_head(tr, te, features, component, alpha=alpha)
                r2 = _r2(y, mu)
                rows.append({"analysis": "alpha_sensitivity", "head": component,
                             "kind": "count", "variant": variant, "alpha": alpha,
                             "n_features": len(features), "r2": r2,
                             "mae": float(np.abs(y - mu).mean()),
                             "nll": float(nb_nll(y, mu, phi).mean()),
                             "dispersion": phi, "floor_r2": floor_r2,
                             "beats_floor": bool(r2 > floor_r2)})
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = cfg["data"]["raw_dir"]
    seasons = cfg["data"]["seasons"]

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, seasons, raw_dir)
    design["season_start_year"] = design["season"].str.slice(0, 4).astype(int)
    print(f"Component rates: {len(design):,} player-seasons "
          f"(prior season >= {MIN_PRIOR_MINUTES} min), "
          f"{design['season'].nunique()} target seasons")

    matrix = pd.read_parquet(features_dir / "season_matrix_roster_tierA.parquet")
    with_pcs, pc_names = walk_forward_pca(design, matrix)
    print(f"  walk-forward PCA: {len(pc_names)} components, refitted per target season on "
          f"S-1 and earlier only")
    print(f"  {with_pcs[pc_names[0]].notna().mean():.1%} of rows have PC scores "
          f"({len(matrix_feature_cols(matrix))} season-matrix columns in)")

    train, val = selection_split(with_pcs, TEST_SEASONS)
    print(f"  The test split is LOCKED — every variant here is RANKED, so the whole "
          f"sweep\n  runs on VALIDATION (src/models/held_out.py).")
    print(f"  {len(train):,} fit / {len(val):,} score "
          f"({', '.join(sorted(val['season'].unique()))} as validation)\n")

    table = evaluate(train, val, pc_names)
    alpha_cfg = cfg.get("evaluation", {}).get("alpha_grid", ALPHA_GRID)
    alphas = alpha_sensitivity(train, val, alpha_cfg)
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "component_rate_metrics.csv"
    # `split` rather than a `val_` prefix on every metric: these columns were never
    # split-tagged, so there is no ambiguity to remove by renaming — only the question of
    # which rows they describe, which a column answers once for the whole file.
    pd.concat([table, alphas], ignore_index=True).assign(
        split="validation").to_csv(dest, index=False)

    for kind, metric in (("count", "r2"), ("conversion", "nll")):
        sub = table[table["kind"] == kind]
        piv = sub.pivot_table(index="head", columns="variant", values=metric)
        cols = [c for c in ["carry_forward", "linear", "log_own", "log_own_spline",
                            "log_own_inter", "spline_own", "inter", "pca", "pca_spline",
                            "pca_inter"] if c in piv.columns]
        print(f"Validation {metric} by {kind} head "
              f"({'higher' if metric == 'r2' else 'lower'} is better):")
        print(piv[cols].round(4).to_string())
        gain = piv.drop(columns=["carry_forward"]).sub(piv["carry_forward"], axis=0)
        best = (gain.max(axis=1) if metric == "r2" else -gain.min(axis=1))
        print(f"  best variant vs the no-fit floor: "
              f"{', '.join(f'{h} {v:+.4f}' for h, v in best.items())}\n")

    # A single variant losing to the floor is often a real finding — `linear` loses on
    # most count heads because linear-in-raw-rate inside exp() is misspecified.
    # What would signal the regularization trap is the *best* variant for a head losing.
    fitted = table[table["variant"] != "carry_forward"]
    beaten = sorted(set(table["head"]) - set(fitted.loc[fitted["beats_floor"], "head"]))
    if beaten:
        print(f"⚠️  NO variant beats the no-fit floor for: {', '.join(beaten)}")
        print("   Two possible causes, and they need different responses. Either the floor "
              "is genuinely\n   unbeatable for that head — `ftm|fta` is the known case, "
              "since free-throw percentage is\n   pure player skill with no context to "
              "add — or the fit is over-regularized. Check alpha\n   against the "
              "module docstring before concluding it is a null.")
    else:
        print("Every head has at least one variant beating the no-fit floor.")
    lost = fitted[~fitted["beats_floor"]]
    if len(lost):
        print(f"\n{len(lost)} individual fits lose to the floor (expected for `linear`, "
              "which is misspecified on the raw rate scale):")
        print(lost.pivot_table(index="head", columns="variant", values="r2",
                              aggfunc="first").round(4).to_string())

    # ── the alpha curve, as a permanent regression guard ─────────────────────
    print("\nRegularization sensitivity — validation R² by alpha under exposure weights.\n"
          "The floor is arithmetic with no parameters, so a fitted line dropping BELOW it "
          "is the\nsignature of the penalty dominating the data term:")
    for variant in ALPHA_VARIANTS:
        sub = alphas[alphas["variant"] == variant]
        if sub.empty:
            continue
        piv = sub.pivot_table(index="head", columns="alpha", values="r2")
        piv.insert(0, "floor", sub.groupby("head")["floor_r2"].first())
        print(f"\n  variant = {variant}")
        print(piv.round(4).to_string())
    top = max(alpha_cfg)
    at_top = alphas[alphas["alpha"] == top]
    print(f"\n  at alpha={top:g}, {int((~at_top['beats_floor']).sum())} of "
          f"{len(at_top)} fits fall below the no-fit floor.")

    # The guard is each head against its OWN optimum on the grid, not against the floor:
    # `blk` and `fg3a` lose to the floor at every alpha because they need splines, which is
    # a documented modelling finding rather than the penalty misbehaving.
    best = alphas.loc[alphas.groupby(["variant", "head"])["r2"].idxmax()]
    worst_best_alpha = float(best["alpha"].max())
    print(f"  every head's best alpha on the grid is <= {worst_best_alpha:g}, and the loss "
          "from the shipped\n  "
          f"alpha={POISSON_ALPHA:g} to alpha=1.0 is:")
    for variant in ALPHA_VARIANTS:
        sub = alphas[alphas["variant"] == variant]
        shipped = sub[sub["alpha"] == min(alpha_cfg)].set_index("head")["r2"]
        broken = sub[sub["alpha"] == 1.0].set_index("head")["r2"]
        if shipped.empty or broken.empty:
            continue
        loss = (shipped - broken).sort_values(ascending=False)
        print(f"    {variant:9s} median {loss.median():.3f} R², worst "
              f"{loss.index[0]} {loss.iloc[0]:.3f}")
    reb = alphas[(alphas["head"] == "reb") & (alphas["variant"] == "linear")]
    lo = reb[reb["alpha"] <= 0.01]
    one = reb[reb["alpha"] == 1.0]
    if len(lo) and len(one):
        print(f"  `reb` linear reproduces the recorded pair: R² "
              f"{lo['r2'].max():.4f} at alpha <= 0.01 (recorded 0.928) against "
              f"{float(one['r2'].iloc[0]):.4f} at alpha=1.0 (recorded 0.662)\n  — the trap "
              "as a curve rather than an anecdote.")

    print(f"\nSaved {len(table) + len(alphas):,} metric rows "
          f"({len(alphas):,} of them the alpha sweep) → {dest}")
    return {"metrics": dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
