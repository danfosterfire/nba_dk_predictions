"""The rookie rate design and its eleven no-fit floors — `docs/rookie-rates-plan.md` §5c.

`component_rates.build_design` is a lag design: it drops any row whose lag-1 block is NaN
before `MIN_PRIOR_MINUTES` is even consulted, so a player with no NBA season at all is not
a noisy row, he is **no row**. §5b's ladder widened that boundary as far as it can go — the
nearest *usable* season, shrunk by its own reliability — and what is left over is the one
population for which "the features do not exist" is literally true. This module builds a
design for exactly that population and nothing else.

## The population, and why it is smaller than the one this family was named for

`docs/rookie-rates-plan.md` §7b measured that a player with *any* prior NBA season is
better served by the veteran heads with his nearest usable lag imputed than by a head built
for players with no history: a returnee's two-year-old rate beats the volume-shrunk
preseason estimator on 7 of 7 heads (0.8977 against 0.8092), and that estimator is missing
for 28.5% of returnees against 9.2% of true rookies. So §3 constraint 2 was withdrawn and
the boundary became *is any prior NBA season constructible*. Here that is one line —
`lag_recovery.classify`'s `true_rookie`, which tests **first played season** rather than
absent lag columns, so a player whose history falls outside the configured window is not
mislabelled into a head that cannot serve him.

Disjointness from the veteran design is therefore by construction rather than by
subtraction, and `run` asserts it against a freshly built ladder design every time.

## Why the delta convention becomes a level

Every veteran head fits a *delta*: the preseason per-36 against the prior-season per-36, on
the head's own link. A rookie has no prior season to difference against, so the same column
becomes **level against the fitting-population mean** on that same link scale — which is
the delta with the population standing in for the player's own history. Centring is not
cosmetic: uncentred, `log1p(pre_per36) = 0` would mean "he scored nothing in the preseason"
and a player with no preseason row would be indistinguishable from one who did nothing,
which is exactly the conflation the missing indicators exist to prevent.

## Zero-recovery, which every rung of the ladder has to satisfy

The Stan sources put a `normal(0, 1)` L2 prior on every standardized coefficient, so the
prior's mode is "this block says nothing". A block is **zero-recovering** when its columns
are exactly 0 for a row whose information is absent — then the prior's mode and the row's
absence agree, and no coefficient can invent a value for a player nobody measured.

- **Preseason**: the shrunk level is `w * (level - center)` at `w = min_pre / (min_pre + k)`,
  so a row with no preseason minutes has `w = 0` and the column is 0 by its own volume
  rather than by a special case. The four age-split missing indicators then say *who* is
  missing, coded on the missing side exactly as `preseason_value` established.
- **Draft slot**: `undrafted` is the reference cell, so an undrafted rookie is all-zeros
  across the slot block and its interaction. That is the honest zero for a block whose
  whole content is *where he was taken* — and it is the largest single cell in the
  population, so the prior's mode is also its mode.
- **Years since draft** is forced to 0 for an undrafted player, whose draft year does not
  exist; a lottery pick arriving immediately is still distinguishable from him by the
  bucket indicator.

`has_preseason` is **not** a twelfth column, and that is a departure from §5c's literal
wording with a mechanical reason: it is `1 - sum(MISSING_AGE_COLS)` exactly — the four
indicators partition the missing rows, as their own docstring says — so adding it makes the
block rank-deficient against the intercept while breaking the missing-side coding that
keeps zero meaning "no information". It is on the frame as a population column and out of
every head's feature list.

## The floors, which are P4(b)'s selected estimator and not a new one

`make rookie-priors` already asked whether a no-prior player's preseason beats his draft
slot and answered it on 19 rolling origins: the volume-shrunk blend wins on 5 of 8 rate
targets, `fta`/`tov`/`stl` are nulls, and **the draft bucket alone is an anti-model**
(R2 -0.043..+0.046). P4 decision 4 is this program's mandate — *"if he is ever put in, his
preseason per-36 is the prior to use and a draft bucket is not"* — so the no-fit floor here
is that estimator, `rookie_priors.arm_predictions`' own `shrunk` arm, wrapped in the head's
likelihood at a train-fitted dispersion so its CRPS is comparable to a fitted head's. That
mirrors `stan_components.count_floor` / `conversion_floor`, which is what Session 4's gate
will read it against.

**Nothing here fits a head.** Four constant blocks are estimated, all on the fitting half
per §3 constraint 6: the conversion block's `(k pseudo-attempts, league)`, the volume shrink
`k`, the centring means, and the floors' dispersions. `make rookie-rates` writes them beside
the floor's own scores so Sessions 4-7 read them from the artifact rather than from a
constant pinned in a module that can stop matching the half it was fitted on.

Usage:
    python -m src.models.rookie_rates
"""

from dataclasses import dataclass
from pathlib import Path

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

from src.data.fetch import _season_start_year
from src.eda.preseason_value import (EPS, MISSING_AGE_COLS, CONVERSION_PRE_SOURCES,
                                     attach_missing_age_indicators, log_rate, logit)
from src.features.team_context import DRAFT_BUCKETS, UNDRAFTED_BUCKET, draft_bucket
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.availability import fit_dispersion
from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, TEST_SEASONS,
                                        PER36, fit_nb_dispersion)
from src.models.held_out import selection_split
from src.models.minutes_preseason import reliability_weight
from src.models.rookie_priors import SHRINK_GRID, arm_predictions, bucket_priors
from src.models.stan_components import (PREDICTIVE_SAMPLES, _beta_shapes, head_label,
                                        score_conversion, score_count)

#: `lag_recovery.classify`'s label for the population this family serves, and the whole of
#: the boundary §3 constraint 2′ drew. Named rather than re-tested here so the design cannot
#: drift from the census that sized it.
ROOKIE_GROUP = "true_rookie"

#: The two arms every floor is read against, and the reference P4(b) took its margins
#: against. `draft_bucket` is the incumbent (`stan_composition.rookie_share_priors`' own
#: estimator, pointed at a rate) and is the measured anti-model; `preseason` is the same
#: blend at `k = 0`. Reporting all three is what shows the shrink doing the work.
INCUMBENT = "draft_bucket"
FLOOR_ARM = "shrunk"

#: The slot block. Four indicators and `undrafted` as the reference cell — see the module
#: docstring on zero-recovery. `team_context.DRAFT_BUCKETS` is the vocabulary, so a board
#: reconciles with `stan_composition`'s minutes-share priors without a mapping table.
SLOT_NAMES = [name for _, _, name in DRAFT_BUCKETS]
SLOT_COLS = [f"draft_{name}" for name in SLOT_NAMES]
YEARS_SINCE_DRAFT = "years_since_draft"
SLOT_INTERACTION_COLS = [f"{c}__x__{YEARS_SINCE_DRAFT}" for c in SLOT_COLS]
BIO_COLS = ["age", "age_sq"]

#: A draft year more than a decade before a first NBA season is a data defect rather than a
#: very patient prospect — the covered window holds one such row at 29 years. Clipping names
#: it; the honest tail (a 2019 second-rounder arriving in 2023) runs to about 9.
MAX_YEARS_SINCE_DRAFT = 10

#: Trailing TRAINING seasons scored rather than fitted when the volume `k` is selected. Two,
#: matching `rookie_priors.INNER_SCORE_SEASONS` and `components_preseason`'s carve, and
#: never validation — a fitted constant must not be allowed to read the selection split.
INNER_SCORE_SEASONS = 2

#: Pseudo-attempts bracket for the conversion block's empirical-Bayes shrink, the bounds
#: `component_rates.carry_forward_conversion` already uses for the same device.
CONVERSION_K_BOUNDS = (1.0, 2000.0)

SEED = 0


# ── Heads ─────────────────────────────────────────────────────────────────────

def head_list() -> list[tuple[str, str, str | None]]:
    """`(label, component, attempted)` for all eleven heads, counts first.

    `lag_ladder.head_list`'s shape, so the two gates in this program iterate their heads in
    one order and a per-head table from either can be read against the other.
    """
    return ([(c, c, None) for c in COUNT_HEADS]
            + [(head_label(m, a), m, a) for m, a in CONVERSION_HEADS])


def level_column(component: str) -> str:
    return f"pre_level_{component}"


def shrunk_column(component: str) -> str:
    return f"{level_column(component)}_shrunk"


def head_family(attempted: str | None) -> str:
    """`count` or `conversion` — the level the volume `k` is selected at."""
    return "count" if attempted is None else "conversion"


def head_features(component: str) -> list[str]:
    """One head's feature ladder, §5c's four blocks in order.

    The preseason level leads because it is the block the family exists for; the slot block
    follows it because P4(b) set the expectation that slot alone is an anti-model and it
    "earns its place in interaction or not at all", which is only readable if the
    interaction is in the list beside it.

    Session 4 adds curvature on top of this (linear -> +interaction -> +spline, mirroring
    `component_rates.count_variants`); this is the linear rung every variant is built from.
    """
    return ([shrunk_column(component)] + list(MISSING_AGE_COLS) + list(SLOT_COLS)
            + [YEARS_SINCE_DRAFT] + list(SLOT_INTERACTION_COLS) + list(BIO_COLS))


# ── The population ────────────────────────────────────────────────────────────

def rookie_rows(targets: pd.DataFrame, seasons: Sequence[str], raw_dir: str | Path,
                covered: Sequence[str] | None = None) -> pd.DataFrame:
    """Every true-rookie player-season, with the lag block **removed**.

    Built on `lag_recovery.population_frame` rather than on a second census: that function
    is the frame `component_rates.build_design` is carved out of, so the rows it drops are
    still there to be selected, and `classify` is the same six-way split §7b's board census
    and §7c's ladder both report in. Reusing it is what makes disjointness a property of one
    labelling rather than an agreement between two.

    **The lag columns are dropped rather than carried as NaN.** Every one of them is
    structurally missing for this population — that is the definition of it — so a caller
    who reached for `reb_p36_lag1` here would be reading a column that cannot exist, and a
    silent NaN is a worse answer than a `KeyError`.

    `covered` cuts to the seasons the preseason panel actually reaches (2004-05 onward).
    Unlike the veteran heads, which cut only their *fitting* rows because they still hold a
    prior-season rate, a rookie row before 2004-05 has no preseason block at all — the head
    would be slot and age alone, a different regime, and its missing indicator would read as
    an era dummy rather than as a player fact.
    """
    from src.models.lag_recovery import population_frame

    frame = population_frame(targets, list(seasons), raw_dir)
    rows = frame[(frame["group"] == ROOKIE_GROUP) & (frame["total_minutes"] > 0)]
    rows = rows.dropna(subset=["age"])
    if covered is not None:
        rows = rows[rows["season"].isin(set(covered))]
    lagged = [c for c in rows.columns if c.endswith(("_lag1", "_lag2", "_lag3"))]
    out = rows.drop(columns=lagged + ["group"]).reset_index(drop=True)
    out["season_start_year"] = out["season"].map(_season_start_year)
    out["age_sq"] = out["age"].to_numpy(dtype=float) ** 2
    return out


def draft_slots(features_dir: Path | str) -> pd.DataFrame:
    """`(player_id, season, draft_number, draft_year)` off the inclusive roster matrix.

    `stan_composition.draft_numbers`' source and its verified coverage (every played
    player-season), with the draft *year* carried too — years-since-draft is half of §5c's
    slot block and the matrix is the only place it exists.

    `bio_draft_number` is NaN for an undrafted player **by construction**, which is the
    reason `team_context.DRAFT_BUCKETS` has an `undrafted` category at all: the alternative
    is imputing a 61st pick for a quarter of this population.
    """
    matrix = pd.read_parquet(Path(features_dir) / "season_matrix_roster_tierA.parquet",
                             columns=["player_id", "season", "bio_draft_number",
                                      "bio_draft_year"])
    return matrix.rename(columns={"bio_draft_number": "draft_number",
                                  "bio_draft_year": "draft_year"})


def attach_draft_slot(rows: pd.DataFrame, draft: pd.DataFrame) -> pd.DataFrame:
    """The slot block: four bucket indicators, years-since-draft, and their interaction.

    Undrafted is the reference cell and years-since-draft is **forced to zero** for him:
    his draft year does not exist, and a stray one in the matrix would otherwise give an
    undrafted row a non-zero column in a block that is supposed to be his zero. A drafted
    rookie who arrives immediately also sits at zero years, but the bucket indicator
    separates the two — which is the whole reason the interaction is a product rather than
    a single "staleness" column.
    """
    out = rows.merge(draft, on=["player_id", "season"], how="left")
    if len(out) != len(rows):
        raise AssertionError("the season matrix duplicated a (season, player_id)")
    bucket = draft_bucket(out["draft_number"])
    drafted = (bucket != UNDRAFTED_BUCKET).to_numpy()
    for name, col in zip(SLOT_NAMES, SLOT_COLS):
        out[col] = (bucket == name).to_numpy().astype(float)
    years = (out["season_start_year"].to_numpy(dtype=float)
             - out["draft_year"].to_numpy(dtype=float))
    out[YEARS_SINCE_DRAFT] = np.where(
        drafted & np.isfinite(years),
        np.clip(np.nan_to_num(years, nan=0.0), 0.0, MAX_YEARS_SINCE_DRAFT), 0.0)
    for col, inter in zip(SLOT_COLS, SLOT_INTERACTION_COLS):
        out[inter] = out[col].to_numpy(dtype=float) * out[YEARS_SINCE_DRAFT].to_numpy(float)
    out["draft_bucket"] = bucket.to_numpy()
    return out


def attach_preseason(rows: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The panel's raw readings plus `has_preseason` and the four missing indicators.

    Raw rather than levelled, because the conversion levels need a constant that is fitted
    on the fitting half and this function is called before the split exists. `with_levels`
    is the second half.

    **`has_preseason` is `min_pre > 0`, not "a panel row exists".** A player can appear in
    the panel having played none of his team's preseason games; his per-36 rates are then
    0/0 and his reliability weight is 0 anyway, so counting him as present would only put a
    NaN where the block promises a number.
    """
    # `fg2m_pre`/`fg2a_pre` are the two names in `CONVERSION_PRE_SOURCES` the panel does
    # not carry — they are derived below — so `fgm_pre` joins the list to derive them.
    raw = sorted({c for pair in CONVERSION_PRE_SOURCES.values() for c in pair}
                 - {"fg2m_pre", "fg2a_pre"} | {"fgm_pre"})
    keep = (["season", "player_id", "min_pre"]
            + [f"pre_per36_{c}" for c in COUNT_HEADS] + raw)
    out = rows.merge(panel[keep], on=["season", "player_id"], how="left")
    if len(out) != len(rows):
        raise AssertionError("the preseason panel duplicated a (season, player_id)")
    # Exactly `component_rates.DERIVED_COUNTS`' derivation, on the preseason side:
    # `fg2a` is not a head, it is `fga - fg3a`, and `fg2m|fg2a` still needs it as trials.
    out["fg2m_pre"] = out["fgm_pre"] - out["fg3m_pre"]
    out["fg2a_pre"] = out["fga_pre"] - out["fg3a_pre"]
    out["min_pre"] = out["min_pre"].fillna(0.0)
    out["has_preseason"] = (out["min_pre"].to_numpy(dtype=float) > 0).astype(float)
    return attach_missing_age_indicators(out)


def build_design(targets: pd.DataFrame, seasons: Sequence[str], raw_dir: str | Path,
                 panel: pd.DataFrame, draft: pd.DataFrame,
                 covered: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per true-rookie player-season, with every block that needs no constant.

    Frames in rather than paths in, for the reason `component_rates.build_design` takes
    `targets`: the forward path (§5g) assembles rookie rows from a roster snapshot for a
    season nobody has played, and it has to reach the same three attachments without going
    through a parquet of realized box scores.
    """
    rows = rookie_rows(targets, seasons, raw_dir, covered)
    return attach_preseason(attach_draft_slot(rows, draft), panel)


# ── The levels, and the constants they need ───────────────────────────────────

@dataclass(frozen=True)
class RookieConstants:
    """Everything fitted on the fitting half that the design and the floors consume.

    Read from the artifact rather than pinned in the module, for the reason
    `stan_components.shrinkage_constants` reads `components_preseason_shrinkage.csv` and
    `component_rates.lag_ladder` reads `lag_recovery.csv`: these are estimated quantities,
    and a constant copied into a module is a constant that can silently stop matching the
    half it was fitted on.
    """

    volume: Mapping[str, float]                    # per head, preseason minutes
    center: Mapping[str, float]                    # per head, on that head's link scale
    conversion: Mapping[str, tuple[float, float]]  # per made column, (pseudo-attempts, league)


def conversion_pct(frame: pd.DataFrame, made: str, k: float, league: float) -> np.ndarray:
    """The preseason conversion percentage, empirical-Bayes shrunk on its own attempts.

    `component_rates.carry_forward_conversion`'s device pointed at preseason attempts, and
    it is a *different* shrink from the volume weight on purpose: a proportion's reliability
    lives in its attempts, not in the minutes that produced them. A rookie who went 0-for-2
    from three in October has a raw preseason 3P% of exactly 0.000, and carrying that onto
    200 regular-season attempts is not a weak estimator but a broken one.

    A row with no preseason attempts lands on `league` exactly, which is the same number the
    centring then subtracts most of — so the column such a row contributes is near zero
    before its volume weight makes it exactly zero.
    """
    num, den = CONVERSION_PRE_SOURCES[made]
    m = np.nan_to_num(frame[num].to_numpy(dtype=float), nan=0.0)
    a = np.nan_to_num(frame[den].to_numpy(dtype=float), nan=0.0)
    return np.clip((m + k * league) / (a + k), EPS, 1.0 - EPS)


def with_levels(frame: pd.DataFrame, conversion: Mapping[str, tuple[float, float]]
                ) -> pd.DataFrame:
    """`pre_level_{component}` for all eleven heads, on each head's own link.

    Counts take `log1p` of the preseason per-36 and conversions the logit of the shrunk
    percentage — the scales `component_rates.add_log` and `conversion_variants` fit on, and
    the scales `preseason_value.attach_rate_block` already builds the veteran deltas on. A
    shared scale would put `reb`'s preseason rebounding on `blk`'s linear predictor.

    **NaN where the player has no preseason minutes**, not 0. Zero is a rate, and the
    distinction between "measured at zero" and "not measured" is the one thing this block
    must not lose; `with_shrunk_level` turns the NaN into an exact 0 by way of a zero
    volume weight, which is a statement about volume rather than about the rate.
    """
    out = frame.copy()
    present = out["min_pre"].to_numpy(dtype=float) > 0
    for _, component, attempted in head_list():
        if attempted is None:
            level = log_rate(out[f"pre_per36_{component}"].to_numpy(dtype=float))
        else:
            level = logit(conversion_pct(out, component, *conversion[component]))
        out[level_column(component)] = np.where(present, level, np.nan)
    return out


def with_shrunk_level(frame: pd.DataFrame, component: str, k: float,
                      center: float) -> pd.DataFrame:
    """`w * (level - center)` written onto a copy — the shipped column, and the only one.

    Two devices in one column, and they answer different questions. The **centring** makes
    zero mean "the population" instead of "a rate of zero", which is what lets a missing row
    share the coefficient's origin with an average one. The **volume weight** then says how
    much of his own reading to believe: `min_pre / (min_pre + k)`, at a median ~68 preseason
    minutes and the `k = 160` P4(b) selected for rates, is about 0.30. §2 of the plan is
    emphatic that this is not optional — raw preseason per-36 loses to the *anti-model* on 6
    of 8 targets.

    `k = 0` is the centred level unshrunk and a large `k` is a column of zeros, so the grid
    that selects `k` brackets both endpoints and an interior optimum is a real one.
    """
    out = frame.copy()
    level = out[level_column(component)].to_numpy(dtype=float)
    weight = reliability_weight(frame, k)
    out[shrunk_column(component)] = np.where(np.isfinite(level),
                                             weight * (level - float(center)), 0.0)
    return out


def with_shipped_block(frame: pd.DataFrame, constants: RookieConstants) -> pd.DataFrame:
    """Every head's shrunk level at the constants that were fitted for it."""
    out = with_levels(frame, constants.conversion)
    for _, component, _ in head_list():
        out = with_shrunk_level(out, component, constants.volume[component],
                                constants.center[component])
    return out


# ── Fitting the constants, on the fitting half alone ──────────────────────────

def fit_conversion_constants(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """`(k pseudo-attempts, league)` per conversion head, on the fitting half's rookies.

    The league mean is this population's own realized ratio rather than the league's, and
    that is deliberate: the constant is what a rookie's preseason percentage is pulled
    *toward*, so it has to be the mean of the players it is pulling. Rookies shoot worse
    than veterans, and shrinking a rookie's 2-for-6 toward a veteran mean would move it in
    the wrong direction.

    `carry_forward_conversion`'s criterion verbatim — minimize the beta-binomial NLL of the
    realized season at a dispersion refitted for each candidate `k`, so the constant is not
    chosen against a dispersion that assumed a different one. Chosen on the rows that have a
    preseason, which is the population the constant is for; a row without one lands on
    `league` for every `k` and carries no information about which `k` is right.
    """
    out: dict[str, tuple[float, float]] = {}
    present = train["min_pre"].to_numpy(dtype=float) > 0
    for made, attempted in CONVERSION_HEADS:
        y_all = np.rint(train[made].to_numpy(dtype=float))
        n_all = np.rint(train[attempted].to_numpy(dtype=float))
        ok = np.isfinite(y_all) & np.isfinite(n_all) & (n_all > 0)
        league = float(y_all[ok].sum() / n_all[ok].sum()) if ok.any() else 0.5
        live = ok & present
        if not live.any():
            out[made] = (float(np.mean(CONVERSION_K_BOUNDS)), league)
            continue
        y = np.minimum(y_all, n_all)[live].astype(int)
        n = n_all[live].astype(int)

        def nll_for(k: float, made=made, league=league, live=live, y=y, n=n) -> float:
            p = conversion_pct(train, made, float(k), league)[live]
            return float(beta_binomial_nll(y, n, p, fit_dispersion(y, n, p)))

        best = minimize_scalar(nll_for, bounds=CONVERSION_K_BOUNDS, method="bounded")
        out[made] = (float(best.x), league)
    return out


def fit_centers(train: pd.DataFrame) -> dict[str, float]:
    """Each head's centring constant — the fitting half's mean level, present rows only.

    Present rows only for `season_centered`'s reason one family over: the absent rows'
    levels are a fill, and averaging them in would drag the origin toward whatever the fill
    happens to be and make the block's zero mean something other than "the population".
    """
    present = train["min_pre"].to_numpy(dtype=float) > 0
    out = {}
    for _, component, _ in head_list():
        level = train[level_column(component)].to_numpy(dtype=float)
        ok = present & np.isfinite(level)
        out[component] = float(level[ok].mean()) if ok.any() else 0.0
    return out


# ── The no-fit floors ─────────────────────────────────────────────────────────

def floor_targets() -> list[tuple[str, str]]:
    """`(head label, the column its bucket prior is an expanding mean of)`.

    A count head's prior is over the **per-36 rate** and a conversion head's over the
    **percentage**, because those are the quantities that are comparable across players
    whose minutes differ by an order of magnitude — which is the same reason
    `rookie_priors` measures per-36 targets rather than totals.
    """
    return [(label, f"{component}_p36" if attempted is None else f"{component}_pct")
            for label, component, attempted in head_list()]


def bucket_prior_table(rows: pd.DataFrame, target: str,
                       covered: Sequence[str]) -> pd.DataFrame:
    """`(season, draft_bucket) -> expanding-window prior`, `rookie_priors`' own builder.

    Point-in-time by construction: season S's prior averages true rookies from seasons
    strictly before S, so it is knowable in September. The first covered season gets no
    prior at all and is dropped by `scorable_rows` rather than handed a constant that came
    from nowhere — `bucket_priors` declines to emit it, and that is the right behaviour to
    inherit rather than to paper over.
    """
    return bucket_priors(rows[rows[target].notna()], target, list(covered))


def scorable_rows(rows: pd.DataFrame, priors: pd.DataFrame) -> np.ndarray:
    """Rows whose season has an expanding prior — everything after the first covered one."""
    return rows["season"].isin(set(priors["season"])).to_numpy()


def floor_arms(rows: pd.DataFrame, priors: pd.DataFrame, preseason_col: str,
               k: float) -> dict[str, np.ndarray]:
    """`{arm: predicted rate or percentage}` — `rookie_priors.arm_predictions` verbatim.

    Imported rather than restated because the floor **is** P4(b)'s selected estimator, and a
    second implementation of it is a place where the head's benchmark and the measurement
    that chose the benchmark can silently disagree.
    """
    return arm_predictions(rows, priors, preseason_col, k)


def count_floor_predictive(train: pd.DataFrame, test: pd.DataFrame, priors: pd.DataFrame,
                           component: str, k: float, arm: str = FLOOR_ARM,
                           seed: int = SEED
                           ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """`(realized, mu, draws x rows samples, phi)` — the floor's predictive, undigested.

    Split out of `count_floor` for §5d's gate, which needs the **per-row** CRPS the paired
    bootstrap resamples rather than the mean `score_count` returns. One implementation with
    two readers, so the number the gate is read against and the number §7d's table quotes
    cannot come apart.
    """
    rate = floor_arms(test, priors, f"pre_per36_{component}", k)[arm]
    mu = np.clip(rate * test["total_minutes"].to_numpy(dtype=float) / PER36, 1e-6, None)
    train_rate = floor_arms(train, priors, f"pre_per36_{component}", k)[arm]
    train_mu = np.clip(train_rate * train["total_minutes"].to_numpy(dtype=float) / PER36,
                       1e-6, None)
    phi = fit_nb_dispersion(train[component].to_numpy(dtype=float), train_mu)
    rng = np.random.default_rng(seed)
    samples = rng.negative_binomial(
        np.full((PREDICTIVE_SAMPLES, len(mu)), phi),
        np.repeat((phi / (phi + mu))[None, :], PREDICTIVE_SAMPLES, axis=0)).astype(float)
    return test[component].to_numpy(dtype=float), mu, samples, phi


def count_floor(train: pd.DataFrame, test: pd.DataFrame, priors: pd.DataFrame,
                component: str, k: float, arm: str = FLOOR_ARM,
                seed: int = SEED) -> dict:
    """The shrunk rate x realized minutes, wrapped in an NB at a train-fitted dispersion.

    `stan_components.count_floor`'s shape with a different prior rate — the wrapping is what
    makes a CRPS from arithmetic comparable to a CRPS from a sampler, and without it §5d's
    gate would be comparing a point prediction against a predictive distribution.
    """
    return score_count(*count_floor_predictive(train, test, priors, component, k, arm,
                                               seed), seed)


def conversion_floor(train: pd.DataFrame, test: pd.DataFrame, priors: pd.DataFrame,
                     made: str, attempted: str, volume_k: float, k_attempts: float,
                     league: float, arm: str = FLOOR_ARM, seed: int = SEED) -> dict:
    """The shrunk percentage on realized trials, beta-binomial at a train-fitted rho.

    `stan_components.conversion_floor`'s shape. The preseason column it blends is the
    EB-shrunk percentage rather than a raw one, so the blend's two ends are both usable
    estimators — a raw preseason 3P% over four attempts is the broken thing
    `carry_forward_conversion` exists to explain.

    **Two shrinkages, and they must not be confused for one another.** `k_attempts` pulls
    his preseason percentage toward the rookie league mean by how many shots he took;
    `volume_k` then blends the result with his draft bucket's expanding prior by how many
    minutes he played. The first is a property of the proportion and the second of the
    reading — passing one where the other belongs divides by zero at `volume_k = 0`, which
    is the grid's own left edge, so the mistake announces itself as a NaN rather than as a
    quietly wrong constant.
    """
    return score_conversion(*conversion_floor_predictive(
        train, test, priors, made, attempted, volume_k, k_attempts, league, arm, seed),
        seed)


def conversion_floor_p(frame: pd.DataFrame, priors: pd.DataFrame, made: str,
                       volume_k: float, k_attempts: float, league: float,
                       arm: str = FLOOR_ARM) -> np.ndarray:
    """The floor's predicted percentage on **every** row, whatever its realized attempts.

    Split out of `conversion_floor_predictive` because §5e composes a season dk total from
    the eleven heads and draws makes on **drawn** trials — so it needs a percentage for a
    row whose realized attempts are zero, which the scoring path deliberately drops. One
    implementation with two readers, so the percentage the gate scores and the percentage
    the composition draws through cannot come apart.
    """
    pre_col = f"pre_pct_{made}"
    scored = frame.assign(**{pre_col: np.where(
        frame["min_pre"].to_numpy(dtype=float) > 0,
        conversion_pct(frame, made, k_attempts, league), np.nan)})
    return np.clip(floor_arms(scored, priors, pre_col, volume_k)[arm], EPS, 1.0 - EPS)


def conversion_floor_predictive(train: pd.DataFrame, test: pd.DataFrame,
                                priors: pd.DataFrame, made: str, attempted: str,
                                volume_k: float, k_attempts: float, league: float,
                                arm: str = FLOOR_ARM, seed: int = SEED
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray, float]:
    """`(y, n, p, draws x rows samples, rho)` on the rows with at least one attempt.

    `count_floor_predictive`'s role for the beta-binomial half, and split out for the same
    reason: §5d's gate resamples per-row CRPS. **The row filter is part of the return** —
    only rows with a realized attempt are scored, which is the same restriction
    `stan_components._score_conv` applies to a fitted head, so a caller pairing the two
    must hand both the same frame.
    """
    tr, te = train, test
    ok = te[attempted].to_numpy(dtype=float) > 0
    live = tr[attempted].to_numpy(dtype=float) > 0
    p = conversion_floor_p(te, priors, made, volume_k, k_attempts, league, arm)[ok]
    p_train = conversion_floor_p(tr, priors, made, volume_k, k_attempts, league, arm)[live]
    y_train = np.minimum(np.rint(tr[made].to_numpy(dtype=float)),
                         np.rint(tr[attempted].to_numpy(dtype=float)))[live].astype(int)
    n_train = np.rint(tr[attempted].to_numpy(dtype=float))[live].astype(int)
    rho = fit_dispersion(y_train, n_train, p_train)
    n = np.rint(te[attempted].to_numpy(dtype=float))[ok].astype(int)
    y = np.minimum(np.rint(te[made].to_numpy(dtype=float))[ok].astype(int), n)
    a, b = _beta_shapes(np.repeat(p[None, :], PREDICTIVE_SAMPLES, axis=0),
                        np.full((PREDICTIVE_SAMPLES, 1), rho))
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n[None, :], rng.beta(a, b)).astype(float)
    return y.astype(float), n.astype(float), p, samples, rho


def head_floor(train: pd.DataFrame, test: pd.DataFrame, priors: pd.DataFrame,
               component: str, attempted: str | None, k: float,
               conversion: Mapping[str, tuple[float, float]], arm: str = FLOOR_ARM,
               seed: int = SEED) -> dict:
    """One head's floor, whichever likelihood it wears. `k` is always the VOLUME shrink."""
    if attempted is None:
        return count_floor(train, test, priors, component, k, arm, seed)
    return conversion_floor(train, test, priors, component, attempted, k,
                            *conversion[component], arm, seed)


# ── Selecting the volume shrink, inside the fitting half ──────────────────────

def fit_volume_k(train: pd.DataFrame, conversion: Mapping[str, tuple[float, float]],
                 grid: Sequence[float] = SHRINK_GRID, inner: int = INNER_SCORE_SEASONS,
                 seed: int = SEED) -> tuple[dict[str, float], pd.DataFrame]:
    """`(head -> k, the grid)` — one `k` per FAMILY, chosen on an inner carve of train.

    Per family rather than per head, which is `rookie_priors`' own decision and for its own
    reason: the carve scores ~120 player-seasons, and eleven heads each picking their own
    rung off that many rows is fitting the grid rather than fitting a shrinkage. Each head's
    own optimum is written to the artifact and **reported, never selected on**, so a family
    optimum that is wrong for one member is visible rather than buried.

    Selection is on **CRPS**, standardized by the head's own target spread so that `fga`
    (season totals in the hundreds) and `blk` (in the tens) contribute comparably. CRPS
    because that is what §5d's gate reads — a constant chosen against an R2 would be
    optimizing something the gate does not look at.
    """
    years = train["season_start_year"].to_numpy(dtype=int)
    cut = sorted(np.unique(years))[-inner]
    fit = train[years < cut].reset_index(drop=True)
    score = train[years >= cut].reset_index(drop=True)
    covered = sorted(set(train["season"]))

    rows = []
    for label, component, attempted in head_list():
        target = f"{component}_p36" if attempted is None else f"{component}_pct"
        priors = bucket_prior_table(train, target, covered)
        keep = scorable_rows(score, priors)
        live = keep & ((score[attempted].to_numpy(dtype=float) > 0) if attempted
                       else np.ones(len(score), dtype=bool))
        scored = score[live].reset_index(drop=True)
        fitted = fit[scorable_rows(fit, priors)].reset_index(drop=True)
        if len(scored) < 2 or not len(fitted):
            continue
        y = (scored[component].to_numpy(dtype=float) if attempted is None
             else np.minimum(np.rint(scored[component].to_numpy(dtype=float)),
                             np.rint(scored[attempted].to_numpy(dtype=float))))
        spread = float(np.std(y)) or 1.0
        for k in grid:
            crps = head_floor(fitted, scored, priors, component, attempted, float(k),
                              conversion, seed=seed)["crps"]
            rows.append({"head": label, "family": head_family(attempted), "k": float(k),
                         "inner_crps": float(crps),
                         "inner_crps_standardized": float(crps) / spread,
                         "n_fit": len(fitted), "n_score": len(scored),
                         "mean_weight": float(reliability_weight(scored, float(k)).mean())})
    table = pd.DataFrame(rows)
    by_family = (table.groupby(["family", "k"])["inner_crps_standardized"].mean()
                 .reset_index())
    chosen = {family: float(part.loc[part["inner_crps_standardized"].idxmin(), "k"])
              for family, part in by_family.groupby("family")}
    return ({component: chosen[head_family(attempted)]
             for _, component, attempted in head_list()}, table)


def fit_constants(train: pd.DataFrame, grid: Sequence[float] = SHRINK_GRID,
                  inner: int = INNER_SCORE_SEASONS, seed: int = SEED
                  ) -> tuple[RookieConstants, pd.DataFrame]:
    """Every constant this family needs, in the one order their dependencies allow.

    The conversion block comes first because the volume grid scores a floor that reads its
    percentages; the centring means come last because they are means of levels the
    conversion block defines.
    """
    conversion = fit_conversion_constants(train)
    volume, table = fit_volume_k(train, conversion, grid, inner, seed)
    center = fit_centers(with_levels(train, conversion))
    return RookieConstants(volume=volume, center=center, conversion=conversion), table


def rookie_constants(path: Path | str) -> RookieConstants:
    """`rookie_rate_floors.csv` -> the constants the design consumes.

    The artifact is the interface, exactly as `lag_recovery.ladder_constants` is for the
    ladder: Sessions 4-7 read this and nothing else, so a head cannot drift from the run
    that fitted its shrink.
    """
    table = pd.read_csv(path)
    hit = table[table["measurement"] == "constants"]

    def block(metric: str) -> dict[str, float]:
        sub = hit[hit["metric"] == metric]
        return {str(h): float(v) for h, v in zip(sub["head"], sub["value"])}

    volume, center = block("k_minutes"), block("center")
    k_att, league = block("k_attempts"), block("league_pct")
    components = [c for _, c, _ in head_list()]
    missing = ([f"{c}/k_minutes" for c in components if c not in volume]
               + [f"{c}/center" for c in components if c not in center]
               + [f"{m}/k_attempts" for m, _ in CONVERSION_HEADS if m not in k_att]
               + [f"{m}/league_pct" for m, _ in CONVERSION_HEADS if m not in league])
    if missing:
        raise ValueError(f"{path} is missing the rookie constants {missing} — re-run "
                         f"`make rookie-rates`.")
    return RookieConstants(volume=volume, center=center,
                           conversion={m: (k_att[m], league[m])
                                       for m, _ in CONVERSION_HEADS})


def head_design(cfg: dict, design: pd.DataFrame | None = None) -> pd.DataFrame:
    """The shipped rookie design — **this family's path, no other's**.

    `stan_components.head_design`'s role one family over, and separate from `build_design`
    for the same reason: the shrunk block needs constants that were fitted on the fitting
    half, and a caller who builds rows without them would get a design whose zero means
    something different.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    if design is None:
        from src.eda.preseason_value import covered_seasons

        eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
        covered = covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv"))
        design = build_design(
            pd.read_parquet(features_dir / "component_targets.parquet"),
            cfg["data"]["seasons"], cfg["data"]["raw_dir"],
            pd.read_parquet(features_dir / "preseason.parquet"),
            draft_slots(features_dir), covered)
    path = out_dir / "rookie_rate_floors.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing and the rookie heads ship a volume-shrunk preseason level "
            f"— run `make rookie-rates`.")
    return with_shipped_block(design, rookie_constants(path))


# ── The readout ───────────────────────────────────────────────────────────────

def _row(measurement: str, split: str, population: str, head: str, arm: str,
         metric: str, value: float, n: int, k: float = float("nan")) -> dict:
    return {"measurement": measurement, "split": split, "population": population,
            "head": head, "arm": arm, "metric": metric, "value": float(value),
            "n": int(n), "k": float(k)}


def population_rows(frames: Mapping[str, pd.DataFrame]) -> list[dict]:
    """The census: rows, draftable rows, preseason coverage, and the slot distribution."""
    out = []
    for split, frame in frames.items():
        for population, part in (("all", frame),
                                 ("draftable",
                                  frame[frame["on_season_start_roster"] > 0])):
            if not len(part):
                continue
            out += [
                _row("population", split, population, "", "", "n_rows", len(part),
                     len(part)),
                _row("population", split, population, "", "", "preseason_coverage",
                     float(part["has_preseason"].mean()), len(part)),
                _row("population", split, population, "", "", "median_min_pre",
                     float(part.loc[part["has_preseason"] > 0, "min_pre"].median()),
                     len(part)),
                _row("population", split, population, "", "", "mean_years_since_draft",
                     float(part[YEARS_SINCE_DRAFT].mean()), len(part)),
            ]
            for name in SLOT_NAMES + [UNDRAFTED_BUCKET]:
                out.append(_row("population", split, population, "", name, "n_slot",
                                int((part["draft_bucket"] == name).sum()), len(part)))
    return out


def constant_rows(constants: RookieConstants) -> list[dict]:
    """The four fitted blocks, one row each, so the artifact is the interface."""
    out = []
    for _, component, _ in head_list():
        out += [_row("constants", "train", "all", component, FLOOR_ARM, "k_minutes",
                     constants.volume[component], 0),
                _row("constants", "train", "all", component, FLOOR_ARM, "center",
                     constants.center[component], 0)]
    for made, (k, league) in constants.conversion.items():
        out += [_row("constants", "train", "all", made, FLOOR_ARM, "k_attempts", k, 0),
                _row("constants", "train", "all", made, FLOOR_ARM, "league_pct",
                     league, 0)]
    return out


def grid_rows(table: pd.DataFrame) -> list[dict]:
    """The volume grid, per head and pooled per family — the family row is what selects."""
    out = []
    for _, r in table.iterrows():
        out.append(_row("shrinkage_grid_by_head", "train", "all", str(r["head"]),
                        FLOOR_ARM, "inner_crps_standardized",
                        float(r["inner_crps_standardized"]), int(r["n_score"]),
                        float(r["k"])))
    by_family = table.groupby(["family", "k"]).agg(
        value=("inner_crps_standardized", "mean"), n=("n_score", "max")).reset_index()
    for _, r in by_family.iterrows():
        out.append(_row("shrinkage_grid", "train", "all", "", str(r["family"]),
                        "inner_crps_standardized", float(r["value"]), int(r["n"]),
                        float(r["k"])))
    return out


def floor_rows(train: pd.DataFrame, frames: Mapping[str, pd.DataFrame],
               covered: Sequence[str], constants: RookieConstants,
               seed: int = SEED) -> list[dict]:
    """Every head x arm x split x population, with the incumbent beside the floor.

    The `draft_bucket` arm is carried because P4(b) measured it as an **anti-model** for
    rates and this is the reading that says whether that still holds on the population the
    head actually fits. A floor that did not beat it would not be a floor.
    """
    out = []
    history = pd.concat(list(frames.values()), ignore_index=True)
    for label, component, attempted in head_list():
        target = f"{component}_p36" if attempted is None else f"{component}_pct"
        priors = bucket_prior_table(history, target, covered)
        fit = train[scorable_rows(train, priors)].reset_index(drop=True)
        k = constants.volume[component]
        for split, frame in frames.items():
            for population, part in (("all", frame),
                                     ("draftable",
                                      frame[frame["on_season_start_roster"] > 0])):
                rows = part[scorable_rows(part, priors)].reset_index(drop=True)
                if attempted is not None:
                    rows = rows[rows[attempted].to_numpy(dtype=float) > 0]
                if len(rows) < 2 or not len(fit):
                    continue
                for arm in (INCUMBENT, "preseason", FLOOR_ARM):
                    scored = head_floor(fit, rows, priors, component, attempted, k,
                                        constants.conversion, arm, seed)
                    for metric, value in scored.items():
                        out.append(_row("floor", split, population, label, arm, metric,
                                        value, len(rows), k))
    return out


# ── Entry point ───────────────────────────────────────────────────────────────

def assert_disjoint(rookie: pd.DataFrame, cfg: dict, targets: pd.DataFrame) -> int:
    """No (player, season) is served by both families — §5c's construction, checked.

    Against the **widest** ladder rather than the shipped one, because the claim is about
    the design's boundary and not about which rungs happen to be admitted today: if the
    ladder could ever reach a row this head also carries, the two would double-count a unit
    in the simulator's `units` union and the tensor would hold him twice.

    Disjoint is all this claims. The two families do not *cover* every rostered player —
    §7c left 84 long-tail board rows served by neither, and a player whose only prior
    season falls outside the lag window is in that hole rather than in this head.
    """
    from src.models.component_rates import LADDER_RUNGS, build_design as veteran_design
    from src.models.component_rates import lag_ladder

    ladder = lag_ladder(cfg, rungs=LADDER_RUNGS)
    veteran = veteran_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"],
                             ladder=ladder)
    overlap = (set(zip(rookie["player_id"], rookie["season"]))
               & set(zip(veteran["player_id"], veteran["season"])))
    if overlap:
        raise AssertionError(
            f"{len(overlap)} player-seasons are in BOTH the rookie design and the widest "
            f"veteran design; the two families must never overlap")
    return len(veteran)


def disjoint_rows(rookie: pd.DataFrame, n_veteran: int) -> list[dict]:
    """The two design sizes and the empty intersection, so the check is auditable.

    `assert_disjoint` raises rather than reports, which is right for a guarantee and wrong
    for a record: a figure nobody can re-derive from the artifact is a figure that drifts
    silently once the assertion is deleted.
    """
    return [_row("disjoint", "design", "all", "", "", "n_rookie_design", len(rookie),
                 len(rookie)),
            _row("disjoint", "design", "all", "", "", "n_veteran_design", n_veteran,
                 n_veteran),
            _row("disjoint", "design", "all", "", "", "n_overlap", 0, len(rookie))]


def run(cfg: dict) -> Path:
    from src.eda.preseason_value import attach_season_start_roster, covered_seasons

    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = list(cfg["data"]["seasons"])
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    covered = covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv"))
    design = build_design(targets, seasons, cfg["data"]["raw_dir"],
                          pd.read_parquet(features_dir / "preseason.parquet"),
                          draft_slots(features_dir), covered)
    design = attach_season_start_roster(design, seasons, cfg["data"]["raw_dir"],
                                        window_games)

    print("Rookie rates — the true-rookie design and its eleven no-fit floors")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")
    n_veteran = assert_disjoint(design, cfg, targets)
    print(f"  DISJOINT: {len(design):,} true-rookie rows share no (player, season) with "
          f"the {n_veteran:,}-row\n  veteran design at EVERY ladder rung, admitted or not "
          f"— which is what Session 6's units\n  union needs. Disjoint, not exhaustive: "
          f"§7c leaves 84 long-tail board rows to neither.")

    train, val = selection_split(design, TEST_SEASONS)
    frames = {"train": train, "validation": val}
    # Every coverage figure is quoted on train+validation, never on the whole design: the
    # design is built over every covered season so Session 6 can persist it at the `full`
    # window, and a census that averaged the held-out seasons in would be a figure nobody
    # is allowed to have read.
    scoped = pd.concat([train, val], ignore_index=True)
    print(f"  covered window {covered[0]} → {covered[-1]}; on train+validation "
          f"{scoped['has_preseason'].mean():.1%} of rows carry a preseason row, "
          f"{scoped.loc[scoped['on_season_start_roster'] > 0, 'has_preseason'].mean():.1%} "
          f"of the draftable ones")
    print(f"  {len(train):,} fit / {len(val):,} score "
          f"({', '.join(sorted(val['season'].unique()))} as validation); "
          f"{int((val['on_season_start_roster'] > 0).sum()):,} of the validation rows "
          f"are draftable")

    constants, grid = fit_constants(train)
    leveled = {name: with_levels(frame, constants.conversion)
               for name, frame in frames.items()}
    print("\n  The volume grid, on an inner carve of the FITTING half — one k per FAMILY, "
          "and\n  each head's own optimum reported but never selected on:")
    family = (grid.groupby(["family", "k"])["inner_crps_standardized"].mean()
              .unstack("family"))
    print(family.round(5).to_string())
    print("  selected k: " + ", ".join(
        f"{f} = {constants.volume[c]:,.0f}"
        for f, c in (("count", COUNT_HEADS[0]), ("conversion", CONVERSION_HEADS[0][0]))))
    own = grid.loc[grid.groupby("head")["inner_crps_standardized"].idxmin()]
    print("  each head's OWN optimum: "
          + ", ".join(f"{r['head']} {r['k']:,.0f}" for _, r in own.iterrows()))
    print("  centring means, on each head's own link: "
          + ", ".join(f"{c} {constants.center[c]:.3f}" for _, c, _ in head_list()))
    print("  conversion block, pseudo-ATTEMPTS not minutes: "
          + ", ".join(f"{m} k={k:.1f} league={p:.3f}"
                      for m, (k, p) in constants.conversion.items()))

    rows = (population_rows(frames) + disjoint_rows(design, n_veteran)
            + constant_rows(constants) + grid_rows(grid)
            + floor_rows(train, leveled, covered, constants))
    table = pd.DataFrame(rows)
    _report(table, with_shipped_block(val, constants))

    dest = out_dir / "rookie_rate_floors.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} rookie-rate rows → {dest}")
    return dest


def _report(t: pd.DataFrame, shipped: pd.DataFrame) -> None:
    def floor(split: str, population: str, head: str, arm: str, metric: str) -> float:
        hit = t[(t["measurement"] == "floor") & (t["split"] == split)
                & (t["population"] == population) & (t["head"] == head)
                & (t["arm"] == arm) & (t["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    def n_of(split: str, population: str, head: str) -> int:
        hit = t[(t["measurement"] == "floor") & (t["split"] == split)
                & (t["population"] == population) & (t["head"] == head)
                & (t["metric"] == "crps")]
        return int(hit["n"].max()) if len(hit) else 0

    for population in ("draftable", "all"):
        print(f"\n  Validation · {population} — the no-fit floor against the incumbent "
              f"draft-bucket mean\n  (P4(b) measured the bucket alone as an ANTI-MODEL "
              f"for rates; R2 here says whether that holds)")
        print(f"  {'head':<10}{'n':>6}{'bucket R2':>11}{'preseason':>11}{'shrunk':>10}"
              f"{'floor CRPS':>12}{'bucket CRPS':>13}{'PIT KS':>9}")
        for label, _, _ in head_list():
            print(f"  {label:<10}{n_of('validation', population, label):>6}"
                  f"{floor('validation', population, label, INCUMBENT, 'r2'):>11.4f}"
                  f"{floor('validation', population, label, 'preseason', 'r2'):>11.4f}"
                  f"{floor('validation', population, label, FLOOR_ARM, 'r2'):>10.4f}"
                  f"{floor('validation', population, label, FLOOR_ARM, 'crps'):>12.4f}"
                  f"{floor('validation', population, label, INCUMBENT, 'crps'):>13.4f}"
                  f"{floor('validation', population, label, FLOOR_ARM, 'pit_ks'):>9.4f}")

    print("\n  The design's zero-recovery, on the validation frame — every block is "
          "exactly 0\n  where its information is absent, which is where the Stan L2 prior's "
          "mode is:")
    absent = shipped["has_preseason"].to_numpy(dtype=float) == 0
    undrafted = (shipped["draft_bucket"] == UNDRAFTED_BUCKET).to_numpy()
    pre_cols = [shrunk_column(c) for _, c, _ in head_list()]
    slot_cols = SLOT_COLS + [YEARS_SINCE_DRAFT] + SLOT_INTERACTION_COLS
    print(f"    preseason block  {int(absent.sum()):>4} rows with no preseason, "
          f"max |column| = "
          f"{float(np.abs(shipped.loc[absent, pre_cols].to_numpy(float)).max(initial=0.0)):.1e}")
    print(f"    slot block       {int(undrafted.sum()):>4} undrafted rows, max |column| = "
          f"{float(np.abs(shipped.loc[undrafted, slot_cols].to_numpy(float)).max(initial=0.0)):.1e}")
    print(f"    {len(head_features(COUNT_HEADS[0]))} features per head: 1 shrunk level, "
          f"{len(MISSING_AGE_COLS)} missing indicators, {len(SLOT_COLS)} slot indicators, "
          f"years-since-draft,\n    {len(SLOT_INTERACTION_COLS)} interactions, "
          f"{len(BIO_COLS)} bio. NO Stan fit here — Session 4 (§5d) runs the gate.")


if __name__ == "__main__":
    from src.models.rookie_rates import run as _run

    _run(yaml.safe_load(open("configs/default.yaml")))
