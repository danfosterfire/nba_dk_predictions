"""Model cards — every fitted head's specification, coefficients and inputs, as flat CSVs.

`docs/dashboard-plan.md` finding 2: **the model detail pages are blocked on artifacts, not
on UI.** Eighteen of twenty heads publish no coefficients anywhere, nothing publishes the
features a head was fed, and the one place all of it already exists —
`data/features/posteriors/<window>/*.pkl` — is the one place the dashboard may not look.
Unpickling a `PosteriorArtifact` imports `src.models.posteriors`, which the `ast`-based
purity guard cannot see because it walks static imports only, and the object it hands back
carries a fitted `StandardScaler` plus the ordered design steps: the capability to score an
arbitrary frame. A dashboard holding that is a dashboard that can silently disagree with the
fit it is describing.

So this module stands between them. It reads the pickles and each head's **own variant
ladder**, and writes long-format tables the dashboard reads and nothing else. All nine the
contract names land here:

| artifact | grain | what it is for |
|---|---|---|
| `model_card_index.csv` | head | the head selector, the *unit* every page must state, and each head's role in the shipped chain |
| `model_card_coefficients.csv` | head × term | the sorted credible-interval panel |
| `model_card_features.csv` | head × feature × split × bin | the small-multiple histograms and the n/mean/sd/missing table |
| `model_card_feature_corr.csv` | head × split × feature × feature | the correlation heatmap, and which pairs earn a density |
| `model_card_feature_density.parquet` | head × pair × split × 2-D bin | the joint behind the heatmap, for the pairs that earn one |
| `model_card_ecdf.csv` | head × split × grid point | the observed ECDF over a posterior-predictive ribbon |
| `model_card_calibration.csv` | head × split × 2-D bin | fitted-against-observed, as density |
| `model_card_quantile.csv` | head × split × panel × row | the scaled quantile residual — a QQ-uniform and the residual against rank-transformed predicted |
| `model_card_sample.parquet` | head × split × row | a bounded subsample, for texture over both densities |

## Three rules this module inherits, and the mechanism for each

**The test split cannot reach an artifact.** Every frame is carved by
`held_out.selection_split`, which drops the last two target seasons entirely rather than
returning them and trusting nobody to read them. `SPLITS` is the closed vocabulary and
`_check_splits` refuses anything outside it on the way to disk, so a fifth split label is a
raised exception rather than a column on a page.

**The `train` posterior window, never `train_val`.** At `train_val` the validation rows were
in the fit, and a "validation" histogram or scatter drawn from those coefficients is an
in-sample picture wearing the wrong label. `WINDOW` is a module constant rather than a flag
for that reason — there is no correct second value — and `posteriors.require_window` refuses
an artifact fitted wider instead of letting it be discovered in a picture.

**The recipe is verified at build time, the way `posteriors.py` verifies it.** That module
reproduces each head's design matrix and predictions on a 400-row probe and fails the build
rather than writing a wrong artifact. This one has a *second*, larger drift surface: it
**re-derives the frames themselves**, so a `build_design` that changed shape, a split that
moved, or a filter that drifted would leave the coefficients describing one population and
the histograms describing another. `verify` therefore runs four checks per head and raises on
any of them:

1. **the population anchor** — the rebuilt fitting frame has exactly the row count and season
   span `posteriors.py` recorded in the artifact's provenance;
2. **the recipe against the ladder** — the persisted recipe applied to the raw frame equals
   the head's own variant ladder put through the head's own scaler, on both splits, to
   `posteriors.DESIGN_TOL`. This is the load-bearing check for the eleven heads that carry
   design steps and is *tautological* for the nine that carry none (their raw frame is their
   design frame), which is why check 4 is run too rather than assumed redundant;
3. **the feature block is producible** — every column `recipe.features` names exists on the
   rebuilt frame;
4. **the artifact's own round-trip** — `PosteriorArtifact.roundtrip()`, which compares the
   recipe against the design matrix and the predictions the *fitted head itself* stored, and
   is therefore non-vacuous for every head including the nine above.

The predictive half adds a fifth, in the same shape: **the drawn predictive must reproduce
the head's own reported mean** (`predictive_bias`), which is what catches a missing exposure,
a wrong trials column or a link applied twice — the three ways a plausible-looking ribbon can
describe a different model from the one the coefficients above it came from.

Two further bars in `check_predictive` are about the **draw budget** rather than the head,
and the difference matters: `ecdf_band_mc` and `quantile_ks_mc` each read their statistic on
two interleaved halves of the draws and raise if the two disagree. Neither thresholds a
*fit* statistic — the ribbon's distance and the residual's KS are reported and never
rendered as a verdict.

Nothing here refits and nothing here samples from Stan, so `make model-cards` needs no
CmdStan — only the head modules, for their variant ladders and their own predictive.

Usage:
    python -m src.models.model_cards
"""

from __future__ import annotations

import copy
import re
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.held_out import selection_split
from src.models.posteriors import (DESIGN_TOL, fit_first_season, team_game_probe, load_all,
                                   posteriors_dir, require_window)
from src.models.stan_utils import ks_uniform, pit_from_samples, thin

# The only window whose coefficients may describe a validation row. See the module
# docstring; deliberately a constant rather than a flag, because there is no second
# correct value and `--window train_val` would be an invitation.
WINDOW = "train"

# The closed split vocabulary. There is no test column, for the same reason the sweeps no
# longer emit one (`src/models/held_out.py`).
SPLITS = ("train", "validation")

# Histogram resolution for a continuous feature. Below ~20 the shape of a skewed prior-rate
# column stops being readable; above ~40 a validation split with 742 rows is mostly empty
# bins. A feature with few enough distinct values gets one bin per value instead, so an
# imputation flag reads as two bars rather than as a spike in bin 0.
FEATURE_BINS = 30
MAX_DISCRETE = 12

# Pairs that earn an on-demand 2-D density, per head and split. Ranked by |r| over the
# distinct off-diagonal pairs, so the flagged set is "where the joint is worth looking at"
# rather than an arbitrary prefix — at rank 20 |r| is still 0.62 on the availability block and
# 0.23 on the smallest count head.
TOP_PAIRS = 20

# The joint density's grid. Coarser than the feature histograms' 30 because a 2-D cell holds
# 1/n_bins of the rows a 1-D bar does: at 18 x 18 the median panel occupies ~180 of its 324
# cells on a training frame and ~110 on a validation one, and a finer grid buys resolution
# nothing has the rows to fill. The **pair list is ranked on the training split** and both
# splits are drawn for it, so flipping the split changes the picture and not the menu.
DENSITY_BINS = 18

# Rows the design check runs on. The composition's 631,158-row frame times ~40 intermediate
# columns is ~200 MB per `frame.copy()` and the recipe makes one per step, so the full-frame
# comparison is bounded rather than free. Evenly spaced through the frame via `thin` — every
# design step is row-wise or uses stored fitted state, so a row subset is exactly equivalent
# to the full frame, and a head slice would be one era of the league.
VERIFY_ROWS = 25_000

# ── The predictive budget ─────────────────────────────────────────────────────
#
# Every one of these is a cost decision, and the composition is the head that sets them: at
# 631,158 training rows times the 1,000 persisted draws its predictive alone is 631M numbers
# and hours of sequential allocation. None of that buys a better picture.
#
# 200 draws over 20,000 rows is 4M per head and seconds. The draw budget is checked rather
# than asserted — `band_stability` re-reads the 95% ribbon on two interleaved halves of the
# draws and `run` refuses a head whose two readings disagree by more than
# `ECDF_BAND_TOL` in ECDF units. The row cap is a *subsample of the population*, so it moves
# Monte Carlo error and not the estimand; the index carries the row count per split so a page
# can say what it drew over.
# 400 since 2026-08-12, up from 200: the availability head became a two-component mixture,
# whose predictive is genuinely wider, and the same 200 draws stopped buying a stable ribbon
# — `ecdf_band_mc` read **0.0216** against a 0.02 bar, so the gate below failed rather than
# shipping a band that was measuring the sampler. The budget is measured rather than
# extrapolated: at 300/400/600/800/1000 draws the availability band reads
# 0.0083 / 0.0136 / 0.0059 / 0.0091 / 0.0079, which is the 1/sqrt(D) fall plus the statistic's
# own noise, and 400 is the smallest power-of-two step that clears the bar for every head.
PRED_DRAWS = 400
PRED_ROWS = 20_000

# Grid points on the ECDF. A response with few enough distinct values gets one point per
# value; everything else gets quantiles of its own observed distribution, which puts the
# resolution where the curve actually moves rather than spreading it evenly over a range a
# season-minutes total or a spell length skews hard inside.
ECDF_GRID = 100
BAND_LEVELS = (2.5, 10.0, 25.0, 50.0, 75.0, 90.0, 97.5)

# The half-sample disagreement the 95% ribbon may show, in ECDF units. **Two, because the
# statistic is roughly twice what it bounds**: two independent D/2 readings differ by about
# 2x the standard error of the D-draw estimate they average to, and `band_stability` takes a
# max over grid points on top of that. So a bar of 0.02 says the shipped ribbon is good to
# about one ECDF point, which is the resolution one of these panels is drawn at. Measured
# worst at 200 draws is 0.0145 (`game_length_depth`), with `availability` next at 0.0135.
#
# Heads below `BAND_MIN_ROWS` are reported and not gated: the overtime-onset head's two
# validation cells give an ECDF that takes three values, and a half-sample gap of 0.5 there
# is the *frame*, not the draw budget.
#
# Worst gated head at the shipped 400 draws is `availability` at 0.0136, which is also the
# head that forced the budget up — see `PRED_DRAWS`. The figure this comment used to quote
# (0.0145 at 200 draws, `game_length_depth`) held until the availability head's likelihood
# changed under it, which is the argument for gating the budget rather than asserting it.
ECDF_BAND_TOL = 0.02
BAND_MIN_ROWS = 500

# The 2-D calibration grid. 30 x 30 with empty cells dropped is a few hundred rows per panel
# rather than 900, and the edges span the pooled 0.5-99.5% range with the tails clipped INTO
# the end bins — one heavy-tailed residual would otherwise collapse the grid to a single cell
# while nothing was formally lost.
#
# **One panel, since 2026-08-10.** This file used to carry `residual_fitted` beside it; the
# raw residual against the fitted value is what `model_card_quantile.csv` replaced, because
# the raw residual of a negative binomial on a season total and of a beta-binomial on a rate
# are not on one scale and cannot be read the same way. The scaled quantile residual is.
CAL_BINS = 30
CAL_SPAN = (0.005, 0.995)
PANELS = ("fitted_observed",)

# ── The scaled quantile residual ──────────────────────────────────────────────
#
# DHARMa's device, in this repo's own functions: simulate replicate responses from the fitted
# model (`draw_predictive`, already there for the ribbon), take each observation's quantile
# inside its own replicate distribution, and randomize across the probability mass AT the
# observed value (`stan_utils.pit_from_samples`, `below + U*at`), because the non-randomized
# quantile of a discrete predictive is not uniform even under a perfect model. Every response
# on these pages is discrete, so the randomization is required rather than optional.
#
# One difference from R's DHARMa, worth stating on the page: DHARMa simulates at the fitted
# model's point estimate, and these draws integrate over the posterior. The residual is
# therefore a Bayesian PIT residual — the same reading, carrying parameter uncertainty rather
# than conditioning it away.

# Order statistics drawn on the QQ panel. A grid rather than every row for the reason the
# whole contract is binned: at 20,000 rows the sorted `u` is 20,000 points that overplot into
# a line. A head with fewer distinct positions than this gets one point per order statistic.
QQ_POINTS = 100

# The residual-against-rank grid. Much coarser than the calibration's 30, and **the smallest
# split sets it rather than the largest**: both axes are [0, 1] by construction — the rank
# transform on x, the PIT on y — so a calibrated head spreads its rows evenly over every cell
# instead of concentrating them on a diagonal, and the panel is read for *departures* from
# that. At 20 x 20 a 742-row validation split puts 1.9 rows in a cell and the picture is
# Poisson noise drawn as structure; at 10 x 10 it puts 7.4, and the quantile lines below get
# 74 rows a bin rather than 37. Measured by rendering it, which is the only layer that can
# see the difference. This is also the one panel in the contract that needs no pooled edge
# set: the transform IS the shared scale, so train and validation are on one grid without
# being put there.
RESIDUAL_BINS = 10

# The three lines DHARMa draws, and the rows a bin needs before its quantiles are worth
# drawing. Below this a "quartile" is two rows and a line drawn through it is noise wearing
# the shape of a finding — the bin is dropped and the line has a gap instead.
QUANTILE_LEVELS = (0.25, 0.5, 0.75)
QUANTILE_MIN_ROWS = 10

# The disagreement the KS distance may show between two interleaved halves of the draws, in
# the units the tile is printed in. Same device and same bar as `ECDF_BAND_TOL`, and gated on
# the same `BAND_MIN_ROWS`, because it is the same question one statistic over: is the number
# on the page a reading of the head or of the draw budget? Measured worst at the shipped 200
# draws is 0.0105 (`gp_entry`, validation); `game_length_ot` reads 0.04 on a two-cell
# validation split and is reported rather than gated, exactly as its ribbon is.
KS_MC_TOL = 0.02

#: Heads whose drawn predictive is not on the observed value's own scale, so a quantile
#: residual would be a picture of nothing — mapped to the reason, which the index carries.
#:
#: **Empty today, and the composition is the head it was opened for.** That head reports
#: `response = eta`, the linear predictor of one *step* in a sequential allocation, and
#: `predictive_check = none` because there is no scale to compare that mean on. Neither fact
#: reaches the quantile residual: `u` is a function of the **draws** and the observed, and
#: `StanComposition.predict_samples` draws minutes in a team-game — the same column
#: `RESPONSES` declares as its observable and the same draws the ribbon is already cut from.
#: `stan_composition.score_samples` settles it, since the head's own scoring computes
#: `ks_uniform(pit_from_samples(samples, y, seed))` — this exact statistic, on this exact
#: predictive. `predictive_check` governs the *fitted* value, not the draws, and does not
#: decide scope.
#:
#: It exists because the failure it guards is silent: a residual drawn against a response on
#: the wrong scale is a perfectly good-looking uniform-ish cloud. Prefer declaring a head
#: here over shipping a panel that is quietly wrong.
QUANTILE_OUT_OF_SCOPE: dict[str, str] = {}

#: The randomization gets its own stream per (head, split), namespaced off the draw's. Same
#: rule as `_seed`: `u` and the draws it is computed from must not share a sequence, and a
#: rebuild must not move the picture — a reader cannot tell a refit from an RNG.
QUANTILE_STREAM = "quantile"

QUANTILE_PANELS = ("qq", "residual", "quantile")

# Points kept for the scatter overlay. Past a couple of thousand a scatter is a blob, so this
# is a legibility cap as much as a size one; `thin` spreads it through the frame.
SAMPLE_ROWS = 2_000

# The predictive's own gate: how far the drawn mean may sit from the head's own reported mean,
# as a share of that mean. Monte Carlo noise on this statistic is ~1/sqrt(rows x draws)
# because it averages over both, so the bar is loose in those units and tight against the
# failures it exists for — a missing exposure, a wrong trials column, a link applied twice —
# every one of which is an order of magnitude, not a percent.
PREDICTIVE_BIAS_TOL = 0.05

# One RNG stream per (head, split), the way `stan_utils.YearTerm` splits its own: two heads
# drawn under one seed would otherwise share a sequence, which is a claim about them.
PRED_SEED = 42

ARTIFACTS = ("model_card_index.csv", "model_card_coefficients.csv",
             "model_card_features.csv", "model_card_feature_corr.csv",
             "model_card_feature_density.parquet", "model_card_ecdf.csv",
             "model_card_calibration.csv", "model_card_quantile.csv",
             "model_card_sample.parquet")


# ── What each head is, declared rather than derived ───────────────────────────

@dataclass(frozen=True)
class ChainRole:
    """What one head does when `make simulate-season` draws a season.

    **A head being fitted, converged and carded says nothing about whether the simulator
    calls it**, and until this vocabulary existed the only way to find out was to read
    `src/sim/season.py`. Sixteen of the twenty heads are read out of the posterior bundle
    by `_sim_one`; four are not, and one whole model page was describing its five heads as
    alternates because nothing on the page could say which.

    `label` is a verb phrase, because the page writes it into a sentence whose subject is
    the head and whose auxiliary is chosen by `in_draw_path` — "in the simulated season it
    **draws the games-played count**" against "it is **not called at draw time**". `note`
    is the mechanism, and like `HeadSpec.description` it is specification prose and never a
    result.

    `in_draw_path` is the machine-checkable half. `draw_path_heads()` returns the heads it
    claims, and `tests/test_model_cards.py` walks `src/sim/` with `ast` to assert that set
    against the artifact keys the simulator actually subscripts — a declared "in the draw
    path" that nothing in `src/sim/` reads is exactly the kind of interpretation that goes
    stale on the next refactor, which is the argument `pca.orient()` and `COMPONENT_BASIS`
    already make one level up.
    """

    label: str
    in_draw_path: bool
    note: str


#: The closed vocabulary. Six roles over twenty heads: five things a season draw is
#: assembled from, and one for a head the draw never touches.
CHAIN_ROLES: dict[str, ChainRole] = {
    "games_played_count": ChainRole(
        "draws the games-played count", True,
        "`_sim_one` draws one beta-binomial rate per player from this head's posterior and "
        "a binomial count of games played per player-team cell. It decides how many games "
        "are missed and nothing about which ones."),
    "absence_layout": ChainRole(
        "lays the absences out", True,
        "`games_played.allocate_spells` places that count of missed games as spells at this "
        "head's fitted `(mu, kappa)`, one pair per posterior draw. It decides how the "
        "misses clump and nothing about how many there are — which is the axis a "
        "best-7-of-16 knockout turns on and the one a games-played marginal cannot see."),
    "minutes_allocation": ChainRole(
        "allocates the team-game's minutes", True,
        "`stan_composition.simulate_minutes` splits each team-game's `5 x game_length` "
        "among the players available for it, with the shipped per-(player, season) effect "
        "injected so the season-level spread comes with the head rather than being "
        "remembered by the consumer."),
    "game_length": ChainRole(
        "draws how long the game is", True,
        "`stan_game_length.sample_game_length`, once per game and shared by both teams, "
        "because overtime is a property of the game. It is the one input a forward "
        "simulation cannot look up."),
    "box_score_component": ChainRole(
        "draws one component of the box score", True,
        "Its per-minute rate or conversion probability is evaluated once per player and "
        "drawn per game against the minutes the allocation gave him, in `DRAW_ORDER`. "
        "`dk_pts` is `compute_dk_pts` over the eleven heads' drawn integers."),
    "not_at_draw_time": ChainRole(
        "not called at draw time", False,
        "Fitted, converged and carded, and `src/sim/season.py` never reads it out of the "
        "posterior bundle. What it supplies instead is a **bar the simulated draw is scored "
        "against** in Gate A: the games-played pmf for the tenure decomposition, the "
        "season-total minutes spread for the marginal minutes head."),
}


@dataclass(frozen=True)
class HeadSpec:
    """The page-facing description of one head.

    **`unit` is here rather than in the dashboard on purpose.** Every model page has to
    state its own unit prominently — the component heads are fitted season-collapsed on
    ~10,000 player-season rows while the composition is per player-game on ~631,000, and a
    reader comparing an R² across those pages without knowing that is being misled. A unit
    string hard-coded in a view goes stale the first time a head is refitted at a different
    grain; read from the artifact it cannot.

    `description` is specification only — what the head models and how — never a result.
    That is the one kind of typed prose `docs/dashboard-plan.md` allows on a model page,
    and it lives here for the same reason `unit` does.

    `chain_role` is the same argument again, one question further on: *what the head does
    in the chain that ships*, keyed into `CHAIN_ROLES`. It is declared beside the head
    rather than in the dashboard for the reason `unit` is, and pinned against `src/sim/`
    rather than merely written down for the reason `COMPONENT_BASIS` is anchored.
    """

    model_class: str
    unit: str
    likelihood: str
    description: str
    chain_role: str


CLASS_LABELS: dict[str, str] = {
    "availability": "Availability",
    "minutes": "Minutes",
    "components": "Box-score components",
    "game_length": "Game length",
}

# The four classes come from the likelihood and the unit, not from the `.stan` source:
# `betabinomial_glm.stan` serves availability, minutes, four conversion heads and overtime
# onset, so grouping by source would put unrelated things behind one selector.
SPECS: dict[str, HeadSpec] = {
    "availability": HeadSpec(
        "availability", "player-season", "beta-binomial",
        "Games played out of team games, shrunk hard toward a league/age baseline.",
        "games_played_count"),
    "gp_entry": HeadSpec(
        "availability", "player-season", "beta-binomial",
        "Share of the schedule before a player's tenure with the team begins.",
        "not_at_draw_time"),
    "gp_exit": HeadSpec(
        "availability", "player-season", "beta-binomial",
        "Share of the schedule after a player's tenure with the team ends.",
        "not_at_draw_time"),
    "gp_onset": HeadSpec(
        "availability", "player-season", "beta-binomial",
        "Per-game hazard of starting an absence spell while at risk inside tenure.",
        "not_at_draw_time"),
    "gp_duration": HeadSpec(
        "availability", "absence spell", "beta-geometric",
        "How long an absence spell lasts — a geometric hazard with a Beta frailty "
        "integrated out.",
        "absence_layout"),
    # The marginal minutes head ships and is **not** in the draw path, which is the second
    # thing this column turned up. Its season-level spread reaches the simulator as
    # `sim.minutes.player_season_sigma`, a constant `minutes_unification` calibrated
    # against it and `rehydrate_composition` injects into the composition — so what
    # `season.py` reads is the composition, and this head is Gate A's minutes-spread bar.
    "minutes": HeadSpec(
        "minutes", "player-season", "beta-binomial",
        "Season minutes as successes out of real game length, given availability.",
        "not_at_draw_time"),
    "composition": HeadSpec(
        "minutes", "player-game inside a team-game", "sequential beta-binomial",
        "A team-game's 5 x game_length minutes allocated among the players who played, "
        "as sequential binomial trials ordered by prior-season minutes share.",
        "minutes_allocation"),
    "game_length_ot": HeadSpec(
        "game_length", "season cell of games", "beta-binomial",
        "Whether a game goes to overtime, collapsed to season cells.",
        "game_length"),
    "game_length_depth": HeadSpec(
        "game_length", "overtime-depth cell", "beta-geometric",
        "How many overtime periods a game that reaches one goes on to play.",
        "game_length"),
}

_COUNT_DESCRIPTIONS = {
    "fga": "Total field-goal attempts", "fta": "Free-throw attempts",
    "reb": "Rebounds", "ast": "Assists", "stl": "Steals", "blk": "Blocks",
    "tov": "Turnovers",
}
for _head, _what in _COUNT_DESCRIPTIONS.items():
    SPECS[_head] = HeadSpec(
        "components", "player-season", "negative binomial",
        f"{_what} as a count with season minutes as the exposure.",
        "box_score_component")

_CONVERSION_DESCRIPTIONS = {
    "fg3a_given_fga": "The three-point share of total field-goal attempts.",
    "fg2m_given_fg2a": "Two-point makes out of two-point attempts.",
    "fg3m_given_fg3a": "Three-point makes out of three-point attempts.",
    "ftm_given_fta": "Free-throw makes out of free-throw attempts.",
}
for _head, _what in _CONVERSION_DESCRIPTIONS.items():
    SPECS[_head] = HeadSpec("components", "player-season", "beta-binomial", _what,
                            "box_score_component")


def chain_role(head: str) -> ChainRole:
    """The head's declared role in the shipped chain, or a raise naming the vocabulary."""
    spec = SPECS.get(head)
    if spec is None:
        raise KeyError(f"no `HeadSpec` for {head!r}; declared heads are {sorted(SPECS)}")
    if spec.chain_role not in CHAIN_ROLES:
        raise KeyError(
            f"{head!r} declares chain role {spec.chain_role!r}, which is not in the closed "
            f"vocabulary {sorted(CHAIN_ROLES)}. The vocabulary is closed so a page can "
            f"group by it and a test can check it against `src/sim/`.")
    return CHAIN_ROLES[spec.chain_role]


def draw_path_heads() -> set[str]:
    """The heads declared to be read when a season is drawn.

    The claim `tests/test_model_cards.py` checks against the artifact keys `src/sim/`
    actually subscripts, rather than against this file a second time.
    """
    return {head for head in SPECS if chain_role(head).in_draw_path}


# Import-time, so a head added with a typo'd or missing role fails on import rather than
# on the page — the same stance `_check_splits` takes on a split label.
for _head in SPECS:
    chain_role(_head)


@dataclass(frozen=True)
class ResponseSpec:
    """What one row of a head's predictive *is*, and how the card draws it.

    Declared per head rather than derived, for the same reason `HeadSpec.unit` is: the
    twenty heads do not share an observable. `label` is the axis a page writes on its ECDF
    and its scatter, and it is the difference between a chart of "the response" and a chart
    of games played.

    - `observed` is the realized column on the head's **own** design frame;
    - `trials` is the binomial denominator, and it is also what puts a rate-reporting head's
      `predict` on the observation scale — `mean_mu x trials`, exactly as
      `PosteriorArtifact.predict` does for the heads that already report it that way;
    - `weight` marks a frame of **collapsed cells**: the duration heads fit one row per
      distinct `(length, covariates)` with a multiplicity, so the card expands by that
      weight before drawing. Without the expansion the depth head's ECDF would be four
      points carrying 1,861 games between them;
    - `check` names which reading of the head's own mean the drawn predictive is verified
      against — see `predictive_bias`;
    - `blocked` marks the composition, whose rows are contiguous team-game blocks that a
      row-wise subsample would cut through.
    """

    label: str
    observed: str
    trials: str = ""
    weight: str = ""
    check: str = "mean"
    blocked: bool = False


RESPONSES: dict[str, ResponseSpec] = {
    "availability": ResponseSpec("games played", "gp", trials="team_games"),
    "gp_entry": ResponseSpec("games before tenure began", "pre_tenure",
                             trials="entry_trials"),
    "gp_exit": ResponseSpec("games after tenure ended", "post_tenure",
                            trials="exit_trials"),
    "gp_onset": ResponseSpec("absence spells started", "onsets",
                             trials="at_risk_played"),
    "gp_duration": ResponseSpec("absence-spell length (games)", "t", weight="w",
                                check="p_one"),
    "minutes": ResponseSpec("season minutes", "successes", trials="trials"),
    "composition": ResponseSpec("minutes in one team-game", "y", trials="m",
                                check="none", blocked=True),
    "game_length_ot": ResponseSpec("overtime games in the cell", "y", trials="n"),
    "game_length_depth": ResponseSpec("overtime periods", "t", weight="w",
                                      check="p_one"),
}

_COUNT_LABELS = {"fga": "season field-goal attempts", "fta": "season free-throw attempts",
                 "reb": "season rebounds", "ast": "season assists", "stl": "season steals",
                 "blk": "season blocks", "tov": "season turnovers"}
for _head, _label in _COUNT_LABELS.items():
    # No `trials`: a count head's exposure is already inside its own `predict` (`mean_count`
    # is mu x minutes) and inside its own `predict_samples`, which reads `total_minutes`.
    RESPONSES[_head] = ResponseSpec(_label, _head)

for _made, _attempted, _label in (("fg3a", "fga", "three-point attempts"),
                                  ("fg2m", "fg2a", "two-point makes"),
                                  ("fg3m", "fg3a", "three-point makes"),
                                  ("ftm", "fta", "free-throw makes")):
    RESPONSES[f"{_made}_given_{_attempted}"] = ResponseSpec(
        f"{_label} in a season", _made, trials=_attempted)


# ── The frames each head was fitted on, rebuilt ───────────────────────────────

@dataclass
class HeadFrames:
    """One head's two splits, in both the raw and the design-column basis.

    `raw` and `design` carry the rows the head **actually fitted** — after any filter the
    head applies to itself. `n_frame` carries the row count *before* that filter, because
    that is what `posteriors.py` recorded in the provenance and therefore what the
    population anchor compares against. The two differ on exactly the four conversion
    heads, whose `StanConversion.fit` drops rows with no attempts internally: `fg3m|fg3a`
    fits 7,695 of the 8,630 rows the manifest reports.
    """

    head: str
    raw: dict[str, pd.DataFrame]
    design: dict[str, pd.DataFrame]
    features: list[str]
    n_frame: dict[str, int]
    row_filter: str = ""
    ladder_features: list[str] = field(default_factory=list)


def _split_pair(design: pd.DataFrame, test_seasons: int
                ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, validation)` and nothing else — the one door to a split in this module.

    Wraps `held_out.selection_split` rather than calling `availability.split_seasons`,
    which would hand back the guarded held-out frame and make the test seasons one
    attribute access away.
    """
    return selection_split(design, test_seasons)


def _frames(head: str, raw_train, raw_val, design_train, design_val, features,
            n_frame_train: int | None = None, row_filter: str = "",
            ladder_features: list[str] | None = None) -> HeadFrames:
    return HeadFrames(
        head=head,
        raw={"train": raw_train, "validation": raw_val},
        design={"train": design_train, "validation": design_val},
        features=list(features),
        n_frame={"train": int(n_frame_train if n_frame_train is not None
                              else len(raw_train)),
                 "validation": int(len(raw_val))},
        row_filter=row_filter,
        ladder_features=list(ladder_features if ladder_features is not None else features))


def availability_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """The beta-binomial games-played-out-of-team-games head. No variant ladder.

    **Two season axes, and only one of them is the split.** `_split_pair` decides which
    seasons are eligible; the head then truncates its *fitting* rows to a recent suffix of
    them (`stan.availability.first_season`, 2012-13 since 2026-08-11), and that truncation is
    re-applied here through the head's own `restrict_window`. Skipping it would not fail
    quietly: the population anchor compares this frame against the row count and season span
    `posteriors.py` recorded, so it would raise. That is the point — the anchor is what makes
    "which rows did this head see" a checked claim rather than a convention.

    `fit_first_season` comes from the artifact rather than from config, for the reason
    `composition_frames` reads its own truncation the same way: the artifact records what was
    fitted and the config is a knob that can move under it.

    **The truncation is on the fitting rows only.** Validation is scored unfiltered, which is
    the head's own rule — a shorter fitting window is a bias-variance trade on the fit, not a
    claim about which rows may be predicted — so the card's validation half covers every
    season the split allows.
    """
    art = artifacts.get("availability")
    if art is None:
        return {}
    # `head_design` rather than the shared builder: the card's population anchor compares a
    # rebuilt frame against the persisted recipe's own feature list, and that list carries
    # the preseason block this head ships (docs/preseason-plan.md P2). Rebuilding from
    # `availability_design` would fail `verify` on five missing columns rather than on
    # anything being wrong.
    from src.models.stan_availability import head_design, restrict_window

    test_seasons = _test_seasons(cfg)
    train, val = _split_pair(head_design(cfg), test_seasons)
    train = restrict_window(train, str(art.extras.get("fit_first_season") or "") or None)
    return {"availability": _frames("availability", train, val, train, val,
                                    art.recipe.features)}


def games_played_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """Entry, exit, onset and the spell-duration head of the tenure decomposition.

    `fittable` is part of the *frame* here rather than an internal head filter —
    `posteriors.games_played_artifacts` applies it before fitting — so it is baked into
    `raw` and the anchor is the filtered count.
    """
    if not {"gp_entry", "gp_exit", "gp_onset", "gp_duration"} & set(artifacts):
        return {}
    from src.models.stan_games_played import (games_played_design, onset_rate_lags,
                                              spell_rows_for)

    test_seasons = _test_seasons(cfg)
    features_dir = Path(cfg["data"]["features_dir"])
    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    design = games_played_design(cfg, panel)
    design = design.merge(onset_rate_lags(design, cfg["data"]["seasons"]),
                          on=["season", "player_id"], how="left")
    train, val = _split_pair(design, test_seasons)
    train, val = train[train["fittable"]], val[val["fittable"]]

    out = {}
    for head in ("gp_entry", "gp_exit", "gp_onset"):
        art = artifacts.get(head)
        if art is None:
            continue
        out[head] = _frames(head, train, val, train, val, art.recipe.features,
                            row_filter="fittable")

    art = artifacts.get("gp_duration")
    if art is not None:
        # The duration arm's covariates are the availability block joined onto collapsed
        # spell rows, so its design frame is a *spell* frame — a different unit from the
        # three heads above, and the reason the index declares one per head.
        cols = list(art.recipe.features) or None
        spell_train = spell_rows_for(panel, train, cols)
        spell_val = spell_rows_for(panel, val, cols)
        out["gp_duration"] = _frames("gp_duration", spell_train, spell_val, spell_train,
                                     spell_val, art.recipe.features,
                                     row_filter="fittable player-seasons, collapsed to "
                                                "absence spells")
    return out


def minutes_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """`min | available`, season-collapsed as successes out of real game length."""
    art = artifacts.get("minutes")
    if art is None:
        return {}
    # `head_design` plus the covered-window cut, so the rebuilt frame is the one the
    # persisted recipe describes — the minutes head ships a preseason block and fits from
    # 2004-05 (docs/preseason-plan.md P3).
    from src.models.stan_minutes import (SPLINE_KNOTS, covered_fitting_rows, head_design,
                                         head_features, variants)

    test_seasons = _test_seasons(cfg)
    n_knots = int(cfg.get("stan", {}).get("minutes", {}).get("spline_knots", SPLINE_KNOTS))
    train, val = _split_pair(head_design(cfg), test_seasons)
    train = covered_fitting_rows(train, cfg)
    tr, va, base_features = variants(train, val, n_knots)[art.recipe.variant]
    features = head_features(base_features)
    return {"minutes": _frames("minutes", train, val, tr, va, art.recipe.features,
                               ladder_features=features)}


def component_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """The seven negative-binomial counts and four beta-binomial conversions.

    One `build_design` for all eleven, because it is the same frame; the ladder differs per
    head. The conversion heads' `attempted > 0` filter is applied here so the histograms
    describe the rows the head saw, while `n_frame` keeps the unfiltered count the manifest
    recorded — see `HeadFrames`.
    """
    from src.models.component_rates import CONVERSION_HEADS, COUNT_HEADS, build_design
    from src.models.stan_components import (SPLINE_KNOTS, conversion_variants,
                                            count_variants)

    wanted = set(COUNT_HEADS) | {f"{made}_given_{att}" for made, att in CONVERSION_HEADS}
    if not wanted & set(artifacts):
        return {}
    test_seasons = _test_seasons(cfg)
    n_knots = int(cfg.get("stan", {}).get("components", {})
                  .get("spline_knots", SPLINE_KNOTS))
    targets = pd.read_parquet(Path(cfg["data"]["features_dir"])
                              / "component_targets.parquet")
    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    train, val = _split_pair(design, test_seasons)

    out = {}
    for component in COUNT_HEADS:
        art = artifacts.get(component)
        if art is None:
            continue
        tr, va, features = count_variants(train, val, component,
                                          n_knots)[art.recipe.variant]
        out[component] = _frames(component, train, val, tr, va, art.recipe.features,
                                 ladder_features=features)

    for made, attempted in CONVERSION_HEADS:
        head = f"{made}_given_{attempted}"
        art = artifacts.get(head)
        if art is None:
            continue
        tr, va, features = conversion_variants(train, val, made, attempted,
                                               n_knots)[art.recipe.variant]
        live_tr = train[attempted].to_numpy(dtype=float) > 0
        live_va = val[attempted].to_numpy(dtype=float) > 0
        out[head] = _frames(head, train[live_tr], val[live_va], tr[live_tr], va[live_va],
                            art.recipe.features, n_frame_train=len(train),
                            row_filter=f"{attempted} > 0",
                            ladder_features=features)
    return out


def composition_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """The team-game minutes allocation, at whatever window the persisted head was fitted.

    `first_season` comes from the artifact's own extras rather than from config, because the
    artifact is the record of what was fitted and the config is a knob that can move under
    it. The full frame is built over every season regardless — the lags and the expanding
    rookie prior need the history — and the cut only decides which rows were fitted.
    """
    art = artifacts.get("composition")
    if art is None:
        return {}
    from src.models.stan_composition import (PILOT_FIRST_SEASON, RHO_BINS,
                                             composition_frame, variants)

    test_seasons = _test_seasons(cfg)
    variant = art.recipe.variant
    if "+" in variant:
        raise NotImplementedError(
            f"the persisted composition was fitted at `{variant}`, an arm from "
            f"`stan_composition.effect_variants` (the player-season effect, the "
            f"team-context block, or both). Its ladder takes the team block and returns a "
            f"seven-tuple, so `composition_frames` has to call `effect_variants` rather "
            f"than `variants` before this head can be carded.")

    first_season = str(art.extras.get("first_season")
                       or cfg.get("stan", {}).get("composition", {})
                       .get("first_season", PILOT_FIRST_SEASON))
    frame = composition_frame(cfg)
    pilot = frame[frame["season"] >= first_season].reset_index(drop=True)
    train, val = _split_pair(pilot, test_seasons)
    ladder = variants(train, val, RHO_BINS)
    tr, va, features = ladder[variant][0], ladder[variant][1], ladder[variant][2]
    return {"composition": _frames("composition", train, val, tr, va,
                                   art.recipe.features, ladder_features=features)}


def game_length_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """Overtime onset on season cells, and overtime depth on collapsed depth rows.

    Two heads on two frames at two units, which is why they are two artifacts rather than
    one — and why the depth head's four training rows are the honest population rather than
    a sampling accident.
    """
    if not {"game_length_ot", "game_length_depth"} & set(artifacts):
        return {}
    from src.models.stan_game_length import (ARMS, MATCHUP_BINS, MATCHUP_COL, depth_rows,
                                             game_frame, matchup_edges, overtime_cells)

    test_seasons = _test_seasons(cfg)
    cfg_gl = cfg.get("stan", {}).get("game_length", {})
    train, val = _split_pair(game_frame(cfg), test_seasons)

    out = {}
    art = artifacts.get("game_length_ot")
    if art is not None:
        arm = str(art.extras.get("arm", art.recipe.variant))
        features = ARMS[arm]["features"]
        edges = matchup_edges(train, int(cfg_gl.get("matchup_bins", MATCHUP_BINS)))
        cell_train, cell_val = train, val
        if MATCHUP_COL in features:
            cell_train = train[train[MATCHUP_COL].notna()]
            cell_val = val[val[MATCHUP_COL].notna()]
        cells_tr = overtime_cells(cell_train, features, edges)
        cells_va = overtime_cells(cell_val, features, edges)
        out["game_length_ot"] = _frames("game_length_ot", cells_tr, cells_va, cells_tr,
                                        cells_va, art.recipe.features,
                                        row_filter="games collapsed to season cells")

    art = artifacts.get("game_length_depth")
    if art is not None:
        depth_tr, depth_va = depth_rows(train), depth_rows(val)
        out["game_length_depth"] = _frames("game_length_depth", depth_tr, depth_va,
                                           depth_tr, depth_va, art.recipe.features,
                                           row_filter="overtime games collapsed to depth "
                                                      "rows")
    return out


BUILDERS = (availability_frames, games_played_frames, minutes_frames, component_frames,
            composition_frames, game_length_frames)


def _test_seasons(cfg: dict) -> int:
    return int(cfg.get("features", {}).get("availability", {}).get("test_seasons", 2))


def build_frames(cfg: dict, artifacts: dict) -> dict[str, HeadFrames]:
    """Every head's rebuilt frames, keyed by head. Nothing is fitted."""
    out: dict[str, HeadFrames] = {}
    for builder in BUILDERS:
        out.update(builder(cfg, artifacts))
    unknown = sorted(set(artifacts) - set(out))
    if unknown:
        raise KeyError(
            f"no frame builder for {unknown}. A head on disk with no builder here would "
            f"card its coefficients and silently omit its features; add it to `BUILDERS`.")
    return out


# ── Verification ──────────────────────────────────────────────────────────────

def design_error(art, raw: pd.DataFrame, design: pd.DataFrame,
                 cap: int = VERIFY_ROWS) -> float:
    """Max absolute disagreement between the persisted recipe and the head's own ladder.

    Two genuinely different paths to the same matrix: the recipe applied to the **raw**
    builder output, against the ladder's already-transformed frame put through the head's
    scaler. Both go through `DesignRecipe.matrix`, so the scaler and the column order are
    shared and what is being compared is the *transform* — the imputation means, the log and
    logit scales, the spline knots, the interaction and the bin edges.

    Zero rows or zero features is reported as agreement rather than as an error: the depth
    head's design is genuinely `(rows x 0)` and `roundtrip` carries its check instead.
    """
    if not len(raw) or not art.recipe.features:
        return 0.0
    idx = thin(len(raw), min(cap, len(raw)))
    left = art.recipe.matrix(raw.iloc[idx])
    right = art.recipe.matrix(design.iloc[idx], transformed=True)
    if left.shape != right.shape:
        return float("inf")
    return float(np.max(np.abs(left - right)))


def verify(art, frames: HeadFrames) -> dict:
    """Four checks per head; raises rather than writing an artifact that disagrees.

    The four are listed in the module docstring. Check 2 is reported as `vacuous` for a head
    whose recipe carries no design steps, because its raw frame *is* its design frame and
    the comparison is an identity — saying so is the difference between a gate and a green
    tick that means nothing.
    """
    provenance = art.provenance
    expected = int(provenance.get("n_fit_rows", -1))
    got = int(frames.n_frame["train"])
    if got != expected:
        raise AssertionError(
            f"{frames.head}: the rebuilt fitting frame has {got:,} rows and the persisted "
            f"posterior was fitted on {expected:,}. The design builder, the split or the "
            f"head's own row filter has moved since `make posteriors` ran; re-run it "
            f"rather than carding a population the coefficients do not describe.")

    seasons = frames.raw["train"].get("season")
    if seasons is not None and provenance.get("first_season"):
        span = (str(seasons.min()), str(seasons.max()))
        recorded = (str(provenance["first_season"]), str(provenance["last_season"]))
        if span != recorded:
            raise AssertionError(
                f"{frames.head}: the rebuilt fitting frame spans {span[0]}–{span[1]} and "
                f"the posterior was fitted over {recorded[0]}–{recorded[1]}.")

    for split in SPLITS:
        missing = [c for c in art.recipe.features
                   if c not in frames.design[split].columns]
        if missing:
            raise KeyError(
                f"{frames.head}: the {split} design frame has no column for {missing}; the "
                f"variant ladder no longer produces the features the recipe names.")

    errors = {split: design_error(art, frames.raw[split], frames.design[split])
              for split in SPLITS}
    worst = max(errors.values())
    if worst > DESIGN_TOL:
        raise AssertionError(
            f"{frames.head}: the persisted design recipe and the head's own variant ladder "
            f"disagree by {worst:.3e} (bar {DESIGN_TOL:.0e}) — train {errors['train']:.3e}, "
            f"validation {errors['validation']:.3e}. One of the two has drifted; do not "
            f"loosen the tolerance.")

    check = art.roundtrip()
    if not check["passes"]:
        raise AssertionError(
            f"{frames.head}: the persisted posterior does not round-trip against its own "
            f"stored predictions — design {check['max_design_error']:.3e}, prediction "
            f"{check['max_prediction_error']:.3e}. Re-run `make posteriors`.")

    return {
        "head": frames.head,
        "n_fit": int(len(frames.raw["train"])),
        "n_validation": int(len(frames.raw["validation"])),
        "n_frame_rows": got,
        "recipe_design_error": worst,
        "roundtrip_design_error": float(check["max_design_error"]),
        "roundtrip_prediction_error": float(check["max_prediction_error"]),
        "design_check": "ladder" if art.recipe.steps else "vacuous",
        "ladder_features_agree": list(frames.ladder_features) == list(art.recipe.features),
        "verified": True,
    }


def _check_splits(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    """The last thing between a split label and disk.

    `selection_split` already makes a test row unreachable, so this cannot fire today. It
    exists because the failure it guards is silent by construction — a page rendering a
    third split would look like a feature — and because a future builder that reaches for a
    frame some other way should hit an exception here rather than a chart.
    """
    if "split" not in frame.columns:
        return frame
    seen = set(frame["split"].astype(str).unique())
    illegal = sorted(seen - set(SPLITS))
    if illegal:
        raise ValueError(
            f"{name} carries the split label(s) {illegal}; the model cards emit "
            f"{list(SPLITS)} and nothing else (src/models/held_out.py).")
    return frame


# ── Terms ─────────────────────────────────────────────────────────────────────

_SPLINE_TERM = re.compile(r"^(?P<base>.+)__s(?P<index>\d+)$")


def term_family(term: str) -> tuple[str, int | None]:
    """`(family, basis index)` — the grouping that keeps a 12-knot basis off the panel.

    A spline basis is `k` coefficients on one underlying quantity, and drawn as `k`
    independent bars it swamps every real term in the head. So `log_ast_p36_lag1__s3` groups
    under `log_ast_p36_lag1` and carries its basis index, and the panel can collapse the
    family to one row or expand it. Imputation flags group together rather than under the
    column they flag: `__miss` terms answer "what does not knowing cost", which is one
    question across the block rather than one per feature.
    """
    match = _SPLINE_TERM.match(term)
    if match:
        return match["base"], int(match["index"])
    if term.endswith("__miss"):
        return "missingness", None
    if term.endswith("__sq"):
        return term[: -len("__sq")], None
    return term, None


def _summary(name: str, draws: np.ndarray, role: str, **extra) -> dict:
    draws = np.asarray(draws, dtype=float).reshape(-1)
    family, basis = term_family(name)
    quantiles = np.quantile(draws, [0.025, 0.25, 0.5, 0.75, 0.975])
    return {
        "term": name, "term_family": family, "term_role": role,
        "basis_index": basis if basis is not None else -1,
        "n_draws": int(draws.size),
        "mean": float(draws.mean()), "sd": float(draws.std(ddof=1)),
        "q2.5": float(quantiles[0]), "q25": float(quantiles[1]),
        "q50": float(quantiles[2]), "q75": float(quantiles[3]),
        "q97.5": float(quantiles[4]),
        # The share of the posterior on one side of zero — the panel's "is this term
        # resolved" reading, and the thing a 95% interval crossing zero under-reports.
        "p_positive": float((draws > 0).mean()),
        **extra,
    }


def coefficient_rows(head: str, art) -> list[dict]:
    """Intercept, one row per feature, and the family's dispersion.

    **Every coefficient is on the standardized design scale**, because every head fits a
    `StandardScaler`'d matrix — which is what makes a sorted bar chart across terms a
    legitimate comparison rather than a plot of measurement units. The scaler's centre and
    scale ride along per term so a consumer can unstandardize without the fitted object
    (`stan_game_length._unstandardized` does the same arithmetic), which is the whole point
    of the emitter: the numbers travel, the capability to score does not.
    """
    rows = [_summary("(intercept)", art.draws["alpha_draws"], "intercept",
                     scaler_center=float("nan"), scaler_scale=float("nan"))]
    rows[0]["term_family"] = "intercept"

    beta = np.asarray(art.draws["beta_draws"], dtype=float)
    scaler = art.recipe.scaler
    centres = np.asarray(getattr(scaler, "mean_", []), dtype=float)
    scales = np.asarray(getattr(scaler, "scale_", []), dtype=float)
    for j, name in enumerate(art.recipe.features):
        rows.append(_summary(
            name, beta[:, j], "coefficient",
            scaler_center=float(centres[j]) if j < centres.size else float("nan"),
            scaler_scale=float(scales[j]) if j < scales.size else float("nan")))

    name = str(art.extras.get("dispersion", ""))
    values = art.draws.get(name)
    if values is not None:
        values = np.asarray(values, dtype=float)
        label = name.replace("_draws", "")
        # The composition's graded arm fits one `rho` per prior-share bin, so its
        # dispersion is a vector and each bin is its own term. Flattening it to a mean
        # would erase the 2.07x fringe-to-star spread that is the arm's whole point.
        columns = [(label, values)] if values.ndim == 1 else [
            (f"{label}[{k + 1}]", values[:, k]) for k in range(values.shape[1])]
        for term, draws in columns:
            rows.append(_summary(term, draws, "dispersion",
                                 scaler_center=float("nan"),
                                 scaler_scale=float("nan")))
            rows[-1]["term_family"] = "dispersion"

    sigma = art.draws.get("sigma_u_draws")
    if sigma is not None:
        rows.append(_summary("sigma_u", sigma, "dispersion", scaler_center=float("nan"),
                             scaler_scale=float("nan")))
        rows[-1]["term_family"] = "dispersion"

    rows += _mixture_rows(art)
    for row in rows:
        row["head"] = head
    return rows


def _mixture_rows(art) -> list[dict]:
    """The availability head's low-availability mixture — eleven terms, or none.

    Emitted because a card that showed `alpha`, `beta` and `rho` for a head fitted with a
    mixture would describe the **single-component** model: `theta` scales the whole weight,
    `gamma` says which players carry it, and `mu_low` / `rho_low` are the disrupted season
    itself. Those are parameters of the shipped head, not diagnostics of it.

    `gamma` rides on `pi`'s OWN scaler — the recipe's second block — so its centre and scale
    columns come from there rather than from the mean's, which would unstandardize eight
    coefficients against the wrong nineteen-column fit. The terms are prefixed `pi:` so a
    reader cannot mistake `pi:age` for the mean's `age`; they are different coefficients on
    the same column through different links.
    """
    theta = art.draws.get("theta_draws")
    if theta is None:
        return []
    out = []
    for name, key in [("theta", "theta_draws"), ("mu_low", "mu_low_draws"),
                      ("rho_low", "rho_low_draws")]:
        # `dispersion`, not a role of their own: these three are scalar summaries of the low
        # component, and the dashboard's `SCALAR_ROLES` is what keeps a scalar out of the
        # sorted slope panel. A new role would render them as bars with no design column
        # behind them.
        out.append(_summary(name, art.draws[key], "dispersion",
                            scaler_center=float("nan"), scaler_scale=float("nan")))
        out[-1]["term_family"] = "mixture"

    gamma = np.asarray(art.draws.get("gamma_draws", np.zeros((len(theta), 0))), dtype=float)
    scaler = getattr(art.recipe, "pi_scaler", None)
    centres = np.asarray(getattr(scaler, "mean_", []), dtype=float)
    scales = np.asarray(getattr(scaler, "scale_", []), dtype=float)
    for j, name in enumerate(getattr(art.recipe, "pi_features", []) or []):
        # `coefficient`, because that is what they are — slopes on a standardized design,
        # just through a different link — so the panel sorts them beside the mean block's
        # and a reader can see which players `pi` picks out.
        # `term_family` is left as `_summary` derives it — one family per term, exactly how
        # the mean block's features are treated. Grouping the eight under a shared
        # "mixture weight" family would be wrong in a way the page makes visible: the panel's
        # collapse toggle keeps one row per family and labels it "widest of N bases", which
        # is right for a spline basis over ONE quantity and nonsense for eight different
        # covariates.
        out.append(_summary(
            f"pi:{name}", gamma[:, j], "coefficient",
            scaler_center=float(centres[j]) if j < centres.size else float("nan"),
            scaler_scale=float(scales[j]) if j < scales.size else float("nan")))
    return out


# ── Features ──────────────────────────────────────────────────────────────────

def bin_edges(values: np.ndarray, bins: int = FEATURE_BINS,
              max_discrete: int = MAX_DISCRETE) -> tuple[np.ndarray, str]:
    """`(edges, kind)` for one feature, from the values of **both** splits pooled.

    Pooled rather than per split, because comparing the two histograms is the block's entire
    job and two histograms drawn on their own edges cannot be compared. A feature with few
    enough distinct values gets one bin per value — an imputation flag then reads as two
    bars at 0 and 1 instead of as a spike in the first of thirty linear bins.
    """
    finite = values[np.isfinite(values)]
    if not finite.size:
        return np.zeros(0), "empty"
    unique = np.unique(finite)
    if unique.size <= max_discrete:
        return unique, "discrete"
    lo, hi = float(finite.min()), float(finite.max())
    if not hi > lo:                                                # pragma: no cover
        return unique, "discrete"
    return np.linspace(lo, hi, bins + 1), "linear"


def _counts(values: np.ndarray, edges: np.ndarray, kind: str) -> np.ndarray:
    finite = values[np.isfinite(values)]
    if kind == "empty":
        return np.zeros(0, dtype=int)
    if kind == "discrete":
        return np.array([int((finite == edge).sum()) for edge in edges])
    # `np.histogram`'s last bin is closed on the right, which is what puts the maximum in
    # the top bar rather than nowhere.
    return np.histogram(finite, bins=edges)[0].astype(int)


_TRANSFORM_PREFIXES = ("log_", "logit_")


def source_columns(feature: str) -> list[str]:
    """The design column and the raw columns it was built from, nearest first.

    A design column is usually two transforms away from the column whose missingness it
    inherits: `logit_fg3m_pct_lag1__s3` is a spline basis over a logit over
    `fg3m_pct_lag1`, and only the last of those three names is what the builder produced or
    the head flagged. Peeling the basis index, then `__sq`, then a `log_` / `logit_` prefix
    walks back to it, and every intermediate name is kept because the head may have flagged
    or built any one of them — `logit_share_lag1` is a *raw* column in the composition
    frame and a *derived* one in the minutes ladder.
    """
    names, current = [feature], feature
    match = _SPLINE_TERM.match(current)
    if match:
        current = match["base"]
        names.append(current)
    if current.endswith("__sq"):
        current = current[: -len("__sq")]
        names.append(current)
    for prefix in _TRANSFORM_PREFIXES:
        if current.startswith(prefix):
            names.append(current[len(prefix):])
            break
    return list(dict.fromkeys(names))


def missing_share(feature: str, raw: pd.DataFrame, design: pd.DataFrame) -> float:
    """What share of this feature's rows the head did not actually observe.

    Read off the head's **own** imputation flag whenever it minted one, because that is its
    record of what it filled rather than a re-derivation of it; otherwise the NaN share of
    the raw column the feature descends from. Resolving through `source_columns` rather than
    on the feature's own name is what stops a spline basis and a `log_` column from
    reporting a flat zero while the missingness they carry sits under a third name.

    An imputation flag is itself never missing, and returns 0. Its **mean** is the share
    being asked about, which is how `docs/dashboard-plan.md`'s "imputation flags shown as
    their own share" lands: the flag is a feature row like any other and its mean is the
    number.
    """
    if feature.endswith("__miss"):
        return 0.0
    candidates = source_columns(feature)
    for name in candidates:
        flag = f"{name}__miss"
        if flag in design.columns:
            return float(np.asarray(design[flag], dtype=float).mean())
    for name in candidates:
        if name in raw.columns:
            return float(pd.isna(raw[name]).mean())
    if feature in design.columns:
        return float(pd.isna(design[feature]).mean())
    return float("nan")                                            # pragma: no cover


def feature_rows(head: str, frames: HeadFrames) -> list[dict]:
    """`head x feature x split x bin` — the histograms and the summary table in one table.

    The per-feature statistics repeat on every bin row rather than living in a fifth file:
    the two are always read together, a page that groups by feature gets both from one
    `read_table`, and the repetition costs ~18,000 rows across every head in the project.
    """
    rows = []
    for feature in frames.features:
        pooled = np.concatenate([
            np.asarray(frames.design[split][feature], dtype=float) for split in SPLITS])
        edges, kind = bin_edges(pooled)
        left = edges if kind == "discrete" else edges[:-1]
        right = edges if kind == "discrete" else edges[1:]
        family, basis = term_family(feature)

        for split in SPLITS:
            values = np.asarray(frames.design[split][feature], dtype=float)
            finite = values[np.isfinite(values)]
            counts = _counts(values, edges, kind)
            total = max(int(finite.size), 1)
            stats = {
                "head": head, "feature": feature, "term_family": family,
                "basis_index": basis if basis is not None else -1,
                "split": split, "n": int(values.size), "n_finite": int(finite.size),
                "mean": float(finite.mean()) if finite.size else float("nan"),
                "sd": float(finite.std(ddof=1)) if finite.size > 1 else float("nan"),
                "min": float(finite.min()) if finite.size else float("nan"),
                "q05": float(np.quantile(finite, 0.05)) if finite.size else float("nan"),
                "q50": float(np.quantile(finite, 0.50)) if finite.size else float("nan"),
                "q95": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
                "max": float(finite.max()) if finite.size else float("nan"),
                "missing_share": missing_share(feature, frames.raw[split],
                                               frames.design[split]),
                "bin_kind": kind, "n_bins": int(len(counts)),
            }
            for index, count in enumerate(counts):
                rows.append({**stats, "bin_index": index,
                             "bin_left": float(left[index]),
                             "bin_right": float(right[index]),
                             "count": int(count),
                             # Share rather than count, so a 773-row validation histogram
                             # can be drawn over an 8,630-row training one.
                             "density": float(count) / total})
    return rows


# ── Feature relationships ─────────────────────────────────────────────────────

def correlation_rows(head: str, frames: HeadFrames,
                     top_pairs: int = TOP_PAIRS) -> list[dict]:
    """The full square correlation matrix per split, plus the pairs that earn a density.

    **The whole square, including the diagonal**, so a heatmap is a reshape rather than a
    reconstruction — and so a constant column shows up as an empty row instead of vanishing
    from the axis. Pearson pairwise-complete via `DataFrame.corr`, which is what leaves a
    constant column as NaN rather than as a spurious zero.

    Both splits, which is one column more than `docs/dashboard-plan.md`'s sketch: the
    collinearity question is a training-frame question, but every other block on these pages
    shows train beside validation, and a block that shrank on the validation split would say
    something worth seeing. `pair_rank` and `top_pair` are computed inside each split, over
    the distinct off-diagonal pairs only.
    """
    rows = []
    for split in SPLITS:
        frame = frames.design[split][frames.features].astype(float)
        matrix = frame.corr()
        order = {name: i for i, name in enumerate(frames.features)}

        pairs = []
        for i, x in enumerate(frames.features):
            for y in frames.features[i + 1:]:
                r = float(matrix.at[x, y])
                if np.isfinite(r):
                    pairs.append((abs(r), x, y))
        pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
        rank = {(x, y): n + 1 for n, (_, x, y) in enumerate(pairs)}

        for x in frames.features:
            for y in frames.features:
                r = float(matrix.at[x, y])
                position = rank.get((x, y)) or rank.get((y, x))
                rows.append({
                    "head": head, "split": split, "feature_x": x, "feature_y": y,
                    "i": order[x], "j": order[y],
                    "r": r, "abs_r": abs(r) if np.isfinite(r) else float("nan"),
                    "n": int(len(frame)),
                    "pair_rank": int(position) if position else -1,
                    "top_pair": bool(position is not None and position <= top_pairs),
                })
    return rows


def density_pairs(correlations: list[dict],
                  split: str = "train") -> list[tuple[str, str, int, float]]:
    """`(x, y, rank, r)` for the pairs that earn a density, in rank order.

    Read back off the rows `correlation_rows` just produced rather than re-ranked here, so
    the flag a page filters the menu on and the panels it can actually draw cannot disagree.
    One orientation per pair — the square carries both and they are the same picture.

    **Ranked on the training split**, one list for both panels. The alternative is a menu
    that changes when the reader flips the split, which breaks the only thing the two panels
    are side by side for.
    """
    seen, out = set(), []
    for row in correlations:
        if row["split"] != split or not row["top_pair"] or row["i"] >= row["j"]:
            continue
        key = (row["feature_x"], row["feature_y"])
        if key in seen:                                            # pragma: no cover
            continue
        seen.add(key)
        out.append((row["feature_x"], row["feature_y"], int(row["pair_rank"]),
                    float(row["r"])))
    return sorted(out, key=lambda item: item[2])


def _cell_index(values: np.ndarray, edges: np.ndarray, kind: str) -> np.ndarray:
    """Which bin each value falls in, for edges `bin_edges` produced.

    `discrete` edges are the values themselves, so the index is a lookup; `linear` edges are
    `bins + 1` boundaries and the top bin is closed on the right, matching `_counts` — the
    two grids are read against each other on a page and an off-by-one at the maximum would
    put the extreme row of a feature in a different cell in each block.
    """
    if kind == "discrete":
        return np.searchsorted(edges, values)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, len(edges) - 2)


def density_rows(head: str, frames: HeadFrames, correlations: list[dict],
                 bins: int = DENSITY_BINS) -> list[dict]:
    """`head x pair x split x 2-D bin` — the joint behind the most correlated cells.

    The heatmap answers "is anything in this block collinear" at a glance and cannot answer
    "what does the joint actually look like"; a full pair-plot matrix answers the second at
    150–400 panels nobody reads. So the density is precomputed for the pairs the first
    question points at, and the page draws one of them at a time
    (`feature-correlation-not-pair-plots`).

    **Both splits share one edge set per pair**, pooled, for the reason the feature
    histograms do: comparing the two panels is the block's job and two grids cannot be
    compared. Empty cells are dropped, which is most of them for a spline basis pair.
    """
    rows = []
    for x, y, rank, _ in density_pairs(correlations):
        grids = {}
        for axis in (x, y):
            pooled = np.concatenate([
                np.asarray(frames.design[split][axis], dtype=float) for split in SPLITS])
            grids[axis] = bin_edges(pooled, bins=bins)

        (x_edges, x_kind), (y_edges, y_kind) = grids[x], grids[y]
        if x_kind == "empty" or y_kind == "empty":                 # pragma: no cover
            continue
        nx = len(x_edges) if x_kind == "discrete" else len(x_edges) - 1
        ny = len(y_edges) if y_kind == "discrete" else len(y_edges) - 1

        for split in SPLITS:
            xs = np.asarray(frames.design[split][x], dtype=float)
            ys = np.asarray(frames.design[split][y], dtype=float)
            keep = np.isfinite(xs) & np.isfinite(ys)
            counts = np.zeros((nx, ny), dtype=np.int64)
            np.add.at(counts, (_cell_index(xs[keep], x_edges, x_kind),
                               _cell_index(ys[keep], y_edges, y_kind)), 1)
            total = max(int(counts.sum()), 1)
            # The r this panel is a picture of, on this panel's own rows — not the training
            # r reused as a label for a validation picture.
            r = float(pd.Series(xs[keep]).corr(pd.Series(ys[keep]))) if keep.sum() > 1 \
                else float("nan")
            for i, j in np.argwhere(counts > 0):
                rows.append({
                    "head": head, "feature_x": x, "feature_y": y,
                    "pair_rank": int(rank), "split": split, "r": r,
                    "x_index": int(i), "y_index": int(j),
                    "x_left": float(x_edges[i]),
                    "x_right": float(x_edges[i] if x_kind == "discrete" else x_edges[i + 1]),
                    "y_left": float(y_edges[j]),
                    "y_right": float(y_edges[j] if y_kind == "discrete" else y_edges[j + 1]),
                    "count": int(counts[i, j]),
                    "density": float(counts[i, j]) / total,
                    "n": int(keep.sum()),
                })
    return rows


# ── The posterior predictive ──────────────────────────────────────────────────
#
# Blocks 5 and 6 of a model page — the ECDF ribbon, and predicted-vs-observed — need
# something the four artifacts above do not: a *drawn* predictive. The rule is that it comes
# from the head's own code, because a card describing a differently-drawn model is worse than
# no card. Thirteen of the twenty heads expose a `predict_samples`, and those are rehydrated
# and called exactly as `src/models/minutes_unification.py` does. The other seven never draw
# at all — they score through an explicit pmf or a log-likelihood — so there is no
# `predict_samples` to call, and the card takes the per-draw *parameters* from the artifact's
# own `mu_draws` and adds the one sampling call the family implies. `predictive_bias` is what
# keeps that honest: the drawn mean has to reproduce the head's own reported mean.

def thinned_artifact(art, keep: int = PRED_DRAWS):
    """A shallow copy of the artifact carrying `keep` of its draws, thinned across all.

    `PosteriorArtifact.mu_draws` uses every draw it holds, which at 1,000 draws x 20,000 rows
    is 160 MB of intermediate for a picture drawn from 200. Thinning the *artifact* rather
    than the result keeps the arithmetic the artifact's own — the same trick
    `posteriors._thinned` plays on a live head, and evenly spaced for the same reason: a head
    slice takes one part of the posterior.
    """
    idx = thin(art.n_draws, keep)
    out = copy.copy(art)
    out.draws = {name: (np.asarray(values)[idx]
                        if np.asarray(values).shape[:1] == (art.n_draws,) else values)
                 for name, values in art.draws.items()}
    return out


def _seed(head: str, split: str) -> int:
    """One stream per (head, split). `YearTerm._rng` splits its own seed the same way."""
    return int(zlib.crc32(f"{head}/{split}".encode())) ^ PRED_SEED


def predictive_frame(head: str, spec: ResponseSpec, design: pd.DataFrame,
                     cap: int = PRED_ROWS) -> tuple[pd.DataFrame, bool]:
    """`(rows, capped)` — the rows this head's predictive is drawn over.

    Weight-expanded first, then capped. The expansion matters for the two duration heads and
    only for them: their frames are *collapsed cells* carrying a multiplicity, so drawing one
    spell per row and weighting it afterwards would put a whole cell's mass on a single draw.
    Expanding gives one row per spell, which is the population the ECDF is of.

    The cap is `thin` — evenly spaced through the frame, because the frames are season-sorted
    and a head slice is one era of the league — except on the composition, where a row-wise
    subsample would cut through a team-game block and `ragged_arrays` would reject the frame
    the artifact exists to feed.
    """
    for name in (spec.observed, spec.trials, spec.weight):
        if name and name not in design.columns:
            raise KeyError(
                f"{head}: the design frame has no column {name!r}, which `RESPONSES` "
                f"declares as this head's observed value, trials or weight. The builder has "
                f"moved; the predictive cannot be drawn against a column that is not there.")
    frame = design
    if spec.weight:
        counts = np.clip(np.rint(frame[spec.weight].to_numpy(dtype=float)), 0, None)
        frame = frame.iloc[np.repeat(np.arange(len(frame)), counts.astype(int))]
    frame = frame.reset_index(drop=True)
    if len(frame) <= cap:
        return frame, False
    if spec.blocked:
        return team_game_probe(frame, cap), True
    return frame.iloc[thin(len(frame), cap)].reset_index(drop=True), True


def _rehydrated(head: str, art, cfg: dict, keep: int):
    """The head class itself, carrying the artifact's draws — or `None` if it cannot draw.

    Deliberately the real head rather than a reimplementation of its predictive: the
    negative-binomial draw, the beta-binomial's `p ~ Beta` then `Binomial`, the composition's
    sequential capped allocation and the thinning in front of each all live in the head, and
    the card is supposed to describe the head as it ships.

    **Availability moved onto this path on 2026-08-11 and it was not cosmetic.** It used to
    fall through to `family_draws`, whose beta-binomial branch takes one dispersion per draw —
    correct while `rho` was a scalar, and wrong the moment the head graded it by prior-MPG
    role. `StanAvailability.mu_draws` gathers each row's own bucket, so drawing through the
    head is what keeps a star's dispersion off a fringe player's mean.

    **The composition is rehydrated with the shipped per-(player, season) sigma**, because
    that is what `rehydrate_composition` gives every other consumer and the whole point of it
    living there is that nobody has to remember to apply it. The index carries the value.
    """
    if art.family == "negbinomial":
        from src.models.stan_components import StanCount

        model = StanCount(list(art.recipe.features), str(art.extras["component"]),
                          name=f"model_cards/{head}", predictive_samples=keep)
        model.scaler = art.recipe.scaler
        model.alpha_draws = np.asarray(art.draws["alpha_draws"], dtype=float)
        model.beta_draws = np.asarray(art.draws["beta_draws"], dtype=float)
        model.phi_draws = np.asarray(art.draws["phi_draws"], dtype=float)
        model.phi = float(model.phi_draws.mean())
        return model
    if "made" in art.extras:
        from src.models.stan_components import StanConversion

        model = StanConversion(list(art.recipe.features), str(art.extras["made"]),
                               str(art.extras["attempted"]),
                               name=f"model_cards/{head}", predictive_samples=keep)
        model.scaler = art.recipe.scaler
        model.alpha_draws = np.asarray(art.draws["alpha_draws"], dtype=float)
        model.beta_draws = np.asarray(art.draws["beta_draws"], dtype=float)
        model.rho_draws = np.asarray(art.draws["rho_draws"], dtype=float)
        model.rho = float(model.rho_draws.mean())
        return model
    if head == "availability":
        from src.models.stan_availability import rehydrate_availability

        return rehydrate_availability(art, keep)
    if head == "minutes":
        from src.models.minutes_unification import rehydrate_minutes

        return rehydrate_minutes(art, keep)
    if head == "composition":
        from src.models.minutes_unification import rehydrate_composition, shipped_sigma

        return rehydrate_composition(art, keep, injected_sigma=shipped_sigma(cfg))
    return None


def family_draws(art, spec: ResponseSpec, frame: pd.DataFrame, keep: int,
                 seed: int) -> np.ndarray:
    """`(draws x rows)` for the seven heads that expose no `predict_samples`.

    Availability, the three games-played binomial heads and overtime onset all score through
    `predict_pmf`, and the two beta-geometric heads through a log-likelihood, so none of them
    has a sampler to call. What they *do* have is the parameters: `mu` per draw comes from
    the artifact's own `mu_draws`, the dispersion from its own persisted block, and the shape
    transform from the head family's own function — `stan_minutes.beta_shapes` for the
    `rho`-parameterized beta-binomial and `games_played.beta_shapes` for the `kappa`
    -parameterized frailty. Only the draw from the sampling law is written here, one line per
    family, and `predictive_bias` checks it against the head's own mean.
    """
    thinned = thinned_artifact(art, keep)
    mu = thinned.mu_draws(frame, transformed=True)
    rng = np.random.default_rng(seed)

    if art.family == "betabinomial":
        from src.models.stan_minutes import beta_shapes

        rho = np.asarray(thinned.draws["rho_draws"], dtype=float)
        if rho.ndim > 1 and rho.shape[1] > 1:
            raise ValueError(
                f"{art.head} carries a {rho.shape[1]}-column `rho_draws` — a dispersion "
                f"graded by bin — and this branch has one dispersion per draw and no bin "
                f"assignment to gather on. Flattening it would apply an arbitrary bucket's "
                f"dispersion to every row. Give the head a `predict_samples` and rehydrate "
                f"it in `_rehydrated`, as `availability` does.")
        a, b = beta_shapes(mu, rho.reshape(-1)[:, None])
        trials = np.rint(frame[spec.trials].to_numpy(dtype=float)).astype(np.int64)
        return rng.binomial(trials[None, :], rng.beta(a, b)).astype(float)

    if art.family == "betageometric":
        from src.models.games_played import beta_shapes as frailty_shapes

        clip = art.extras.get("mu_clip")
        if clip is not None:
            mu = np.clip(mu, float(clip[0]), float(clip[1]))
        kappa = np.asarray(thinned.draws["kappa_draws"], dtype=float).reshape(-1)
        a, b = frailty_shapes(mu, kappa[:, None])
        # A hazard of exactly 0 is a spell of infinite length and a `ValueError` from numpy.
        # The floor bounds the support at a million games rather than letting one underflowed
        # Beta draw decide the axis of the chart.
        hazard = np.clip(rng.beta(a, b), 1e-6, 1.0)
        return rng.geometric(hazard).astype(float)

    raise KeyError(
        f"{art.head}: no predictive for the `{art.family}` family. A head that neither "
        f"exposes `predict_samples` nor names a family this knows cannot be carded — teach "
        f"`family_draws` its sampling law rather than drawing it somewhere else.")


def draw_predictive(head: str, art, spec: ResponseSpec, frame: pd.DataFrame, cfg: dict,
                    keep: int = PRED_DRAWS, seed: int = 0) -> np.ndarray:
    """`(draws x rows)` — the head's own predictive, however that head draws."""
    model = _rehydrated(head, art, cfg, keep)
    if model is None:
        return family_draws(art, spec, frame, keep, seed)
    return np.asarray(model.predict_samples(frame, seed), dtype=float)


def fitted_values(art, spec: ResponseSpec, frame: pd.DataFrame,
                  draws: np.ndarray) -> np.ndarray:
    """The head's own reported mean, on the response's own scale.

    Three heads have no such mean and take the mean of their own draws instead, which the
    index declares as `fitted_source = predictive_mean` rather than leaving a reader to
    assume: the composition reports a linear predictor of a *step* (`response = eta`), and a
    beta-geometric's mean is `E[1/p]` under a Beta frailty, which is not the `mu` the head
    reports and is not finite for every parameter it can take.
    """
    if spec.check != "mean":
        return draws.mean(axis=0)
    mean = np.asarray(art.predict(frame, transformed=True), dtype=float)
    if art.response in ("mean_mu", "plug_in_mu", "mixture_mean_mu"):
        if not spec.trials:
            raise KeyError(
                f"{art.head} reports `{art.response}` — a rate — and its `ResponseSpec` "
                f"names no `trials`, so there is nothing to put it on the observation "
                f"scale with. The scatter would be a probability against a count.")
        return mean * frame[spec.trials].to_numpy(dtype=float)
    return mean


def predictive_bias(art, spec: ResponseSpec, frame: pd.DataFrame, draws: np.ndarray,
                    fitted: np.ndarray) -> float:
    """Signed relative gap between the drawn predictive and the head's own reported mean.

    The fifth build-time check, and the one the four in `verify` cannot make: the recipe can
    reproduce a design matrix perfectly while the predictive drawn from it is on the wrong
    scale. A missing exposure, a trials column that moved, a link applied twice — each is an
    order of magnitude here, against a bar of `PREDICTIVE_BIAS_TOL`.

    The beta-geometric heads are checked on a different reading of the same parameter, and a
    sharper one: `mu` **is** `P(T = 1)` for that likelihood, so the share of drawn spells
    that end after one game is a prediction rather than a restatement. The composition is not
    checked — `eta` is the mean of a step in a sequential allocation, and there is no scale
    to compare it on.
    """
    if spec.check == "mean":
        scale = float(np.abs(fitted).mean())
        return float((draws.mean() - fitted.mean()) / max(scale, 1e-12))
    if spec.check == "p_one":
        mu = float(np.asarray(art.predict(frame, transformed=True), dtype=float).mean())
        return float(((draws == 1).mean() - mu) / max(mu, 1e-12))
    return float("nan")


# ── The ECDF ribbon ───────────────────────────────────────────────────────────

def ecdf_grid(observed: np.ndarray, cap: int = ECDF_GRID) -> tuple[np.ndarray, str]:
    """`(grid, kind)` — where the ECDF is evaluated, from the observed values alone.

    Quantiles of the observed rather than a linear span, because every response here is
    skewed and half a linear grid over season minutes or spell length would sit in a tail
    that holds a dozen rows. A response with few enough distinct values gets one point per
    value instead, which is what makes the overtime-depth curve four honest steps.

    Drawn from the **observed** and not from the draws on purpose: an over-wide predictive
    then shows as a ribbon that has not reached 1 at the last grid point, which is the
    finding. A grid stretched to cover it would hide that in the axis.
    """
    finite = np.asarray(observed, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        return np.zeros(0), "empty"
    unique = np.unique(finite)
    if unique.size <= cap:
        return unique, "discrete"
    return np.unique(np.quantile(finite, np.linspace(0.0, 1.0, cap))), "quantile"


def _ecdf_at(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.searchsorted(np.sort(values), grid, side="right") / max(len(values), 1)


def ecdf_curves(draws: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """`(draws x grid)` — one replicate dataset's ECDF per posterior draw.

    Per draw rather than pooled over all of them: pooling gives the predictive's own CDF,
    which has no width and cannot be compared against a single realized sample. The ribbon
    asks "how far from the observed curve would a dataset drawn from this posterior fall",
    and that needs one whole replicate dataset per draw.
    """
    out = np.empty((draws.shape[0], grid.size))
    for d in range(draws.shape[0]):
        out[d] = _ecdf_at(draws[d], grid)
    return out


def band_stability(curves: np.ndarray) -> float:
    """Max disagreement between the 95% ribbon read on two interleaved halves of the draws.

    The draw budget's own check, in the units the chart is drawn in: if one half of the draws
    and the other half put the ribbon edge in the same place, the budget is enough and the
    number says so rather than the comment. It bounds about twice the error of the shipped
    full-budget ribbon — see `ECDF_BAND_TOL`, which is set from that.

    Interleaved rather than split down the middle, because the draws arrive chain-major and
    the first half is not the same posterior as the second.
    """
    if curves.shape[0] < 4:
        return float("nan")
    lo = np.percentile(curves[0::2], [2.5, 97.5], axis=0)
    hi = np.percentile(curves[1::2], [2.5, 97.5], axis=0)
    return float(np.max(np.abs(lo - hi)))


def ecdf_rows(head: str, split: str, observed: np.ndarray,
              draws: np.ndarray) -> tuple[list[dict], float]:
    """`(rows, band_mc)` — the observed ECDF and the predictive band, one row per grid point.

    Seven quantiles rather than the three bands the page draws, so the 50 / 80 / 95% ribbons
    are `q25`–`q75`, `q10`–`q90` and `q2.5`–`q97.5` and the median line is `q50`. Emitting
    the edges rather than the bands keeps the file a reshape away from any of them.
    """
    grid, kind = ecdf_grid(observed)
    if not grid.size:                                              # pragma: no cover
        return [], float("nan")
    curves = ecdf_curves(draws, grid)
    quantiles = np.percentile(curves, BAND_LEVELS, axis=0)
    observed_curve = _ecdf_at(observed, grid)

    rows = []
    for i, value in enumerate(grid):
        row = {"head": head, "split": split, "grid_index": int(i), "value": float(value),
               "observed": float(observed_curve[i]), "grid_kind": kind,
               "n_rows": int(len(observed)), "n_draws": int(draws.shape[0])}
        row.update({f"q{level:g}": float(quantiles[j, i])
                    for j, level in enumerate(BAND_LEVELS)})
        rows.append(row)
    return rows, band_stability(curves)


# ── Fitted against observed, as density ───────────────────────────────────────

def calibration_edges(values: np.ndarray, bins: int = CAL_BINS,
                      span: tuple[float, float] = CAL_SPAN) -> np.ndarray:
    """Bin edges over the pooled 0.5–99.5% range of one axis.

    Robust rather than min-to-max because a single season-minutes residual or one 79-game
    spell puts every other row in the first cell — a panel that is technically complete and
    shows nothing. Values outside are clipped *into* the end bins by `calibration_rows`, so
    the tail is visible at the edge rather than dropped.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not finite.size:                                            # pragma: no cover
        return np.linspace(0.0, 1.0, bins + 1)
    lo, hi = (float(x) for x in np.quantile(finite, span))
    if not hi > lo:
        lo, hi = float(finite.min()), float(finite.max())
    if not hi > lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, bins + 1)


def _panel_values(panel: str, fitted: np.ndarray,
                  observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if panel == "fitted_observed":
        return fitted, observed
    raise KeyError(
        f"unknown calibration panel {panel!r}. `residual_fitted` was removed on "
        f"2026-08-10 — the raw residual against the fitted value is what "
        f"`model_card_quantile.csv` replaced, on a scale four differently-distributed "
        f"heads can actually be read on.")


def calibration_rows(head: str, panels: dict[str, tuple[np.ndarray, np.ndarray]],
                     bins: int = CAL_BINS) -> list[dict]:
    """`head x split x panel x 2-D bin`, empty cells dropped.

    Binned rather than per-row for the reason the whole contract is binned: the composition's
    scatter is 631,158 points per panel, which is not an artifact but a copy of the data. And
    **both splits share one edge set per panel**, computed on the pooled values, for the same
    reason the feature histograms do — comparing train against validation is the panel's job
    and two grids cannot be compared.
    """
    rows = []
    for panel in PANELS:
        pooled = [_panel_values(panel, *panels[split]) for split in SPLITS]
        x_edges = calibration_edges(np.concatenate([x for x, _ in pooled]), bins)
        y_edges = calibration_edges(np.concatenate([y for _, y in pooled]), bins)
        for split in SPLITS:
            x, y = _panel_values(panel, *panels[split])
            counts, _, _ = np.histogram2d(
                np.clip(x, x_edges[0], x_edges[-1]),
                np.clip(y, y_edges[0], y_edges[-1]), bins=[x_edges, y_edges])
            total = max(float(counts.sum()), 1.0)
            for i, j in np.argwhere(counts > 0):
                rows.append({
                    "head": head, "split": split, "panel": panel,
                    "x_index": int(i), "y_index": int(j),
                    "x_left": float(x_edges[i]), "x_right": float(x_edges[i + 1]),
                    "y_left": float(y_edges[j]), "y_right": float(y_edges[j + 1]),
                    "count": int(counts[i, j]),
                    "density": float(counts[i, j]) / total,
                    "n": int(len(x)),
                })
    return rows


def sample_frame(head: str, split: str, fitted: np.ndarray, observed: np.ndarray,
                 u: np.ndarray | None = None, rank: np.ndarray | None = None,
                 cap: int = SAMPLE_ROWS) -> pd.DataFrame:
    """A bounded subsample of real points, for texture over the binned densities.

    `thin` rather than a random sample, so the overlay spans the frame and re-running the
    emitter does not move the points around under a reader. `float32` because these are
    coordinates on a scatter and the parquet is 80,000 rows of them.

    **`u` and `predicted_rank` ride here rather than in the quantile artifact**, for the
    reason the fitted and observed pair already does: the artifact is binned so that a
    631,158-row panel is a picture rather than a copy of the data, and the overlay is the
    bounded exception. Both are taken at the *same* thinned rows as `fitted`, and the rank is
    computed over the **whole** predictive frame before the thinning — a rank recomputed
    inside a 2,000-row subsample would be a different transform from the one the panel
    underneath it is binned on. `None` on a head declared out of quantile scope, which ships
    the columns as NaN rather than dropping them and changing the file's shape per head.
    """
    idx = thin(len(fitted), min(cap, len(fitted)))
    empty = np.full(len(fitted), np.nan)
    return pd.DataFrame({
        "head": head, "split": split, "row": idx.astype(np.int32),
        "fitted": fitted[idx].astype(np.float32),
        "observed": observed[idx].astype(np.float32),
        "u": (empty if u is None else np.asarray(u, dtype=float))[idx].astype(np.float32),
        "predicted_rank": (empty if rank is None
                           else np.asarray(rank, dtype=float))[idx].astype(np.float32)})


# ── The scaled quantile residual ──────────────────────────────────────────────

def quantile_scope(head: str) -> tuple[str, str]:
    """`(scope, reason)` — whether this head admits a quantile residual at all.

    `drawn` for every head today. The alternative is `not_applicable` with the reason on the
    index, which is the shape this takes rather than a missing head: a page that finds no
    rows for a head cannot tell "out of scope" from "the build broke", and an absent panel
    with no reason beside it is the same defect one level up. See `QUANTILE_OUT_OF_SCOPE`.
    """
    reason = QUANTILE_OUT_OF_SCOPE.get(head)
    return ("not_applicable", reason) if reason else ("drawn", "")


def quantile_seed(head: str, split: str) -> int:
    """The randomization's own stream, namespaced off the draw's `_seed`."""
    return _seed(f"{head}/{QUANTILE_STREAM}", split)


def scaled_residuals(draws: np.ndarray, observed: np.ndarray, seed: int) -> np.ndarray:
    """`u` per row — `stan_utils.pit_from_samples`, verbatim and by import.

    Named rather than inlined so the panel below and the KS beside it are visibly one
    quantity, and *not* reimplemented for the same reason `compute_dk_pts` is never
    reimplemented: `below + U*at` is the whole of DHARMa's scaled residual, this repo already
    had it, and a second copy is a second thing to keep right.
    """
    return pit_from_samples(draws, observed, seed)


def ks_stability(draws: np.ndarray, observed: np.ndarray, seed: int) -> float:
    """|KS(half) − KS(other half)| on two **interleaved** halves of the draws.

    `band_stability`'s device, one statistic over, and it answers the question the draw
    budget actually raises here: at `PRED_DRAWS` draws a row with no replicate landing
    exactly on its observed value has `at = 0`, so its `u` is `below` alone — a multiple of
    1/D rather than a continuous number. Two independent D/2 readings disagreeing by less
    than `KS_MC_TOL` is the evidence that the tile on the page is a reading of the head and
    not of the budget.

    Interleaved rather than split down the middle, because the draws arrive chain-major. The
    **same** seed on both halves, so the randomization is held fixed and what moves is the
    draws.
    """
    if draws.shape[0] < 4:
        return float("nan")
    return float(abs(ks_uniform(pit_from_samples(draws[0::2], observed, seed))
                     - ks_uniform(pit_from_samples(draws[1::2], observed, seed))))


def rank_uniform(values: np.ndarray) -> np.ndarray:
    """`(rank − 0.5) / n` — DHARMa's rank transform of the predicted value.

    What makes the residual panel comparable across heads: a count head's predicted season
    rebounds and a conversion head's predicted makes share no axis, and their ranks do. Ties
    take the average rank, so a head whose fitted values are a handful of cells reads as a
    few columns rather than as an arbitrary ordering of equals.
    """
    values = np.asarray(values, dtype=float)
    if not values.size:                                            # pragma: no cover
        return values
    return (pd.Series(values).rank(method="average").to_numpy() - 0.5) / len(values)


def qq_rows(u: np.ndarray, points: int = QQ_POINTS) -> list[dict]:
    """The QQ-uniform panel: order statistics against their expected uniform quantiles.

    **The envelope is exact and pointwise**: the k-th of n order statistics of a uniform
    sample is `Beta(k, n − k + 1)`, so `lo`/`hi` are that distribution's 2.5 and 97.5
    percentiles rather than a simulated band. Pointwise, which is the reading it has to be
    given: about 5 of 100 grid points fall outside a pointwise 95% envelope under a *correct*
    model, and the composition's rows are not independent inside a team-game either. It is a
    sense of scale beside the curve, never a test — the same rule `band_distance` carries.
    """
    from scipy.stats import beta

    u = np.asarray(u, dtype=float)
    u = u[np.isfinite(u)]
    n = u.size
    if not n:                                                      # pragma: no cover
        return []
    order = np.unique(np.clip(np.rint(np.linspace(1, n, min(points, n))), 1, n)
                      ).astype(int)
    values = np.sort(u)[order - 1]
    # k/(n+1) rather than k/n: the expectation of the k-th order statistic, so the last point
    # sits below 1 instead of pinning the top of the panel to the largest observation.
    expected = order / (n + 1.0)
    lo = beta.ppf(0.025, order, n - order + 1)
    hi = beta.ppf(0.975, order, n - order + 1)
    return [{"panel": "qq", "x_index": int(i), "y_index": -1,
             "x": float(expected[i]), "y": float(values[i]),
             "lo": float(lo[i]), "hi": float(hi[i])}
            for i in range(len(order))]


def _bin_index(values: np.ndarray, bins: int) -> np.ndarray:
    """Which of `bins` equal cells on [0, 1] each value falls in, top bin closed."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    return np.clip(np.digitize(values, edges[1:-1], right=False), 0, bins - 1)


def residual_rows(u: np.ndarray, rank: np.ndarray,
                  bins: int = RESIDUAL_BINS) -> list[dict]:
    """The scaled residual against rank-transformed predicted, as a binned density.

    Both axes are `[0, 1]` by construction, which is why this is the one panel in the
    contract with no pooled edge set to compute: the rank transform and the PIT *are* the
    shared scale, so train and validation are already on one grid. Empty cells are dropped,
    as everywhere else.
    """
    keep = np.isfinite(u) & np.isfinite(rank)
    xs, ys = np.asarray(rank)[keep], np.asarray(u)[keep]
    counts = np.zeros((bins, bins), dtype=np.int64)
    np.add.at(counts, (_bin_index(xs, bins), _bin_index(ys, bins)), 1)
    total = max(int(counts.sum()), 1)
    edges = np.linspace(0.0, 1.0, bins + 1)
    return [{"panel": "residual", "x_index": int(i), "y_index": int(j),
             "x": float((edges[i] + edges[i + 1]) / 2),
             "y": float((edges[j] + edges[j + 1]) / 2),
             "x_left": float(edges[i]), "x_right": float(edges[i + 1]),
             "y_left": float(edges[j]), "y_right": float(edges[j + 1]),
             "count": int(counts[i, j]), "density": float(counts[i, j]) / total}
            for i, j in np.argwhere(counts > 0)]


def quantile_lines(u: np.ndarray, rank: np.ndarray, bins: int = RESIDUAL_BINS,
                   levels: tuple[float, ...] = QUANTILE_LEVELS,
                   min_rows: int = QUANTILE_MIN_ROWS) -> list[dict]:
    """The 0.25 / 0.5 / 0.75 quantiles of `u` inside each column of that panel.

    **Flat at their own levels iff calibrated**, which is the quantitative half of the
    picture: a density can look reasonable while its middle drifts, and three lines against
    three references say so in the units the axis is already in. Binned on the density's own
    x edges, so the lines and the cells beneath them cannot disagree.

    A column with fewer than `min_rows` rows is left out rather than drawn, so a line has a
    gap where the evidence does — a quartile of three rows is noise in the shape of a
    finding.
    """
    keep = np.isfinite(u) & np.isfinite(rank)
    xs, ys = np.asarray(rank)[keep], np.asarray(u)[keep]
    cells = _bin_index(xs, bins)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        inside = ys[cells == i]
        if inside.size < min_rows:
            continue
        for level, value in zip(levels, np.quantile(inside, levels)):
            rows.append({"panel": "quantile", "x_index": int(i), "y_index": -1,
                         "x": float((edges[i] + edges[i + 1]) / 2),
                         "y": float(value), "level": float(level),
                         "x_left": float(edges[i]), "x_right": float(edges[i + 1]),
                         "count": int(inside.size)})
    return rows


def quantile_tables(head: str, split: str, u: np.ndarray,
                    rank: np.ndarray) -> tuple[list[dict], float]:
    """`(rows, ks)` — the three panels of one head's residual, and its KS distance.

    One long table with a `panel` column, the way `model_card_calibration.csv` carries its
    own: every row is a location `(x, y)` on that panel's axes, with the columns a panel does
    not have left empty. `ks` is repeated on every row — the same deliberate repetition
    `model_card_features.csv` makes, so a page filtering to one head and split gets the panel
    *and* the number printed above it from one read.

    **The KS distance is a distance.** `ks_uniform` over the scaled residuals, reported and
    never thresholded: at n ≈ 10⁴ a strict uniformity test rejects everything, so a page
    rendering in-or-out would report that twenty heads out of twenty fail. The only bar in
    this module's quantile half is `KS_MC_TOL`, and that one is about the *draw budget*
    rather than about the head.
    """
    finite = np.isfinite(u)
    ks = ks_uniform(np.asarray(u)[finite]) if finite.any() else float("nan")
    stamp = {"head": head, "split": split, "ks": float(ks), "n": int(finite.sum())}
    return ([{**stamp, **row} for row in
             qq_rows(u) + residual_rows(u, rank) + quantile_lines(u, rank)],
            float(ks))


def predictive_tables(head: str, art, frames: HeadFrames, cfg: dict,
                      keep: int = PRED_DRAWS, cap: int = PRED_ROWS) -> dict:
    """Draw each split once and cut all three predictive artifacts from the same draws.

    One draw per split, not one per artifact: the ECDF band, the calibration density and the
    scatter overlay are three readings of one predictive, and drawing them separately would
    let a page show a ribbon and a scatter that disagree.
    """
    spec = RESPONSES.get(head)
    if spec is None:
        raise KeyError(
            f"no `ResponseSpec` for {head!r}. Every head declares what its predictive is an "
            f"ECDF *of* — a page cannot label an axis it has to guess at, and a head with no "
            f"declared observable would silently ship an empty ribbon.")

    ecdf, calibration, quantile, samples, panels = [], [], [], [], {}
    counts, band, bias, ks, ks_mc = {}, {}, {}, {}, {}
    scope, reason = quantile_scope(head)
    for split in SPLITS:
        frame, capped = predictive_frame(head, spec, frames.design[split], cap)
        observed = frame[spec.observed].to_numpy(dtype=float)
        draws = draw_predictive(head, art, spec, frame, cfg, keep, _seed(head, split))
        fitted = fitted_values(art, spec, frame, draws)

        rows, band[split] = ecdf_rows(head, split, observed, draws)
        ecdf.extend(rows)

        u = rank = None
        if scope == "drawn":
            # The randomization and the draws are one object here: `u` is computed from the
            # draws the ribbon above was cut from, so a page cannot show a ribbon and a QQ
            # that describe two different predictives.
            seed = quantile_seed(head, split)
            u = scaled_residuals(draws, observed, seed)
            rank = rank_uniform(fitted)
            rows, ks[split] = quantile_tables(head, split, u, rank)
            quantile.extend(rows)
            ks_mc[split] = ks_stability(draws, observed, seed)

        samples.append(sample_frame(head, split, fitted, observed, u, rank))
        panels[split] = (fitted, observed)
        counts[split] = (len(frame), capped)
        bias[split] = predictive_bias(art, spec, frame, draws, fitted)
    calibration.extend(calibration_rows(head, panels))

    finite_band = [b for b in band.values() if np.isfinite(b)]
    gated = [b for split, b in band.items()
             if np.isfinite(b) and counts[split][0] >= BAND_MIN_ROWS]
    sigma = 0.0
    if head == "composition":
        # Recorded rather than implied: the shipped head a consumer loads carries the
        # injected per-(player, season) effect, so the ribbon does too, and a page comparing
        # this card against `stan_composition_metrics.csv` has to be able to see that.
        from src.models.minutes_unification import shipped_sigma

        sigma = float(shipped_sigma(cfg))
    gated_ks = [ks_mc[split] for split in ks_mc
                if np.isfinite(ks_mc[split]) and counts[split][0] >= BAND_MIN_ROWS]
    return {
        "ecdf": ecdf, "calibration": calibration, "quantile": quantile,
        "sample": samples,
        "summary": {
            "response_label": spec.label,
            "predictive_draws": int(keep),
            "n_predictive_train": counts["train"][0],
            "n_predictive_validation": counts["validation"][0],
            "predictive_rows_capped": bool(counts["train"][1]
                                           or counts["validation"][1]),
            # Why the depth head draws over 1,861 rows when it was fitted on 4: its frame is
            # collapsed cells and the predictive is over the spells they carry.
            "predictive_weighted": bool(spec.weight),
            "fitted_source": ("head_predict" if spec.check == "mean"
                              else "predictive_mean"),
            "predictive_check": spec.check,
            "predictive_bias": max(bias.values(), key=abs),
            "ecdf_band_mc": max(finite_band) if finite_band else float("nan"),
            "ecdf_band_gated": bool(gated),
            "player_season_sigma": sigma,
            # The quantile half. Both splits' distances ship because both are read on the
            # page — unlike `predictive_bias`, which is one gate collapsed to its worst
            # split — and `quantile_ks_mc` is the only bar here, on the draw budget rather
            # than on the head.
            "quantile_scope": scope,
            "quantile_reason": reason,
            "quantile_ks_train": ks.get("train", float("nan")),
            "quantile_ks_validation": ks.get("validation", float("nan")),
            "quantile_ks_mc": max(gated_ks) if gated_ks else (
                max(ks_mc.values(), key=abs) if ks_mc else float("nan")),
            "quantile_ks_gated": bool(gated_ks),
            # How a collapsed-cell frame's multiplicity enters: `predictive_frame` expands
            # it BEFORE the draw, so the residual is one row per spell and the KS is
            # unweighted over spells. The alternative — one residual per cell, weighted —
            # would put 1,861 games' worth of mass on four `u` values.
            "quantile_weighting": "expanded" if spec.weight else "unweighted",
        },
    }


def check_predictive(head: str, summary: dict) -> None:
    """Raise rather than write a ribbon drawn from a different model than the coefficients.

    Two bars, and they fail differently. A predictive whose mean has drifted from the head's
    own is a *wrong* picture — a missing exposure or a trials column that moved. A ribbon
    that is not stable across half the draws is a *noisy* one, and the answer to it is more
    draws rather than a different model.
    """
    bias = summary["predictive_bias"]
    if np.isfinite(bias) and abs(bias) > PREDICTIVE_BIAS_TOL:
        raise AssertionError(
            f"{head}: the drawn predictive sits {bias:+.2%} from the head's own reported "
            f"mean (bar {PREDICTIVE_BIAS_TOL:.0%}). The recipe can reproduce a design matrix "
            f"exactly and still be drawn on the wrong scale — check the exposure, the trials "
            f"column and the link before loosening this.")
    band = summary["ecdf_band_mc"]
    if summary["ecdf_band_gated"] and np.isfinite(band) and band > ECDF_BAND_TOL:
        raise AssertionError(
            f"{head}: the 95% ECDF ribbon moves by {band:.4f} between two halves of the "
            f"{summary['predictive_draws']} draws (bar {ECDF_BAND_TOL}). The band is Monte "
            f"Carlo noise at this budget; raise `PRED_DRAWS` rather than shipping it.")
    # The third bar, and the only one on the quantile half. **Not** a bar on the KS distance
    # itself — that is reported and never thresholded — but on whether the distance is a
    # reading of the head or of the draw budget, which is what the 1/D quantization of `u`
    # puts in question.
    ks_mc = summary["quantile_ks_mc"]
    if summary["quantile_ks_gated"] and np.isfinite(ks_mc) and ks_mc > KS_MC_TOL:
        raise AssertionError(
            f"{head}: the KS distance of the scaled quantile residual moves by {ks_mc:.4f} "
            f"between two halves of the {summary['predictive_draws']} draws (bar "
            f"{KS_MC_TOL}). At this budget a row with no replicate at its observed value "
            f"carries a `u` quantized to 1/{summary['predictive_draws']}; raise "
            f"`PRED_DRAWS` rather than printing a tile that is measuring the sampler.")


# ── The index ─────────────────────────────────────────────────────────────────

def index_row(head: str, art, frames: HeadFrames, check: dict,
              n_terms: int, predictive: dict | None = None,
              n_density_pairs: int = 0) -> dict:
    spec = SPECS.get(head)
    if spec is None:
        raise KeyError(
            f"no `HeadSpec` for {head!r}. Every head declares its own unit, so the page can "
            f"read it rather than hard-coding a unit string that goes stale on the next "
            f"refit — see `HeadSpec`.")
    provenance = art.provenance
    role = chain_role(head)
    return {
        "head": head,
        "label": art.head_label,
        "model_class": spec.model_class,
        "class_label": CLASS_LABELS[spec.model_class],
        "unit": spec.unit,
        # Beside the unit, and for the same reason: a page must be able to say what the
        # head does in the chain that ships, and a view that types that in is a view that
        # goes stale on the next refactor of `src/sim/season.py`. `in_draw_path` is the
        # half a test can check; `chain_role` is the closed key a page may group by.
        "chain_role": spec.chain_role,
        "chain_role_label": role.label,
        "in_draw_path": role.in_draw_path,
        "chain_role_note": role.note,
        "family": art.family,
        "likelihood": spec.likelihood,
        "description": spec.description,
        "variant": art.recipe.variant,
        "n_features": len(art.recipe.features),
        # The recipe's SECOND design block, which today only the availability mixture has:
        # `pi`'s covariates enter through their own link and their own scaler, so they are
        # not more columns in `n_features` and the coefficient panel carries both blocks.
        # A page reading `n_features` alone would under-count the panel by exactly this.
        "n_pi_features": len(getattr(art.recipe, "pi_features", []) or []),
        "n_terms": int(n_terms),
        # How many of this head's feature pairs a page can open a joint density on. Zero is
        # a real value — a one-feature head has no pair — and a page reading it can say so
        # rather than rendering an empty selector.
        "n_density_pairs": int(n_density_pairs),
        "n_fit": check["n_fit"],
        "n_validation": check["n_validation"],
        "n_frame_rows": check["n_frame_rows"],
        "row_filter": frames.row_filter,
        "n_draws": art.n_draws,
        "fit_window": provenance.get("fit_window", ""),
        # The second season axis, and NOT the one above. `fit_window` is which split may be
        # fitted (`train` / `train_val` / `full`); this is which recent suffix of that split
        # the head chose to fit, empty for the heads that fit everything the window offers.
        # `first_season` below is neither — it is the fitted frame's observed span, which
        # equals the truncation when there is one and predates it when there is not.
        "fit_first_season": fit_first_season(art),
        "first_season": provenance.get("first_season", ""),
        "last_season": provenance.get("last_season", ""),
        "response": art.response,
        "dispersion": str(art.extras.get("dispersion", "")).replace("_draws", ""),
        "max_rhat": float(provenance.get("max_rhat", float("nan"))),
        "divergences": int(provenance.get("divergences", -1)),
        "converged": bool(provenance.get("converged", False)),
        "coefficient_scale": "standardized",
        "recipe_design_error": check["recipe_design_error"],
        "roundtrip_prediction_error": check["roundtrip_prediction_error"],
        "design_check": check["design_check"],
        "verified": check["verified"],
        # The predictive half's own columns: what the ribbon is an ECDF *of*, how many rows
        # and draws it was cut from, where the fitted value came from, and both of its
        # build-time checks. A page that draws the ribbon without saying it was drawn over a
        # 20,000-row subsample of 631,158 is overstating it.
        **{key: (predictive or {}).get(key, default) for key, default in (
            ("response_label", ""), ("predictive_draws", 0),
            ("n_predictive_train", 0), ("n_predictive_validation", 0),
            ("predictive_rows_capped", False), ("predictive_weighted", False),
            ("fitted_source", ""),
            ("predictive_check", ""), ("predictive_bias", float("nan")),
            ("ecdf_band_mc", float("nan")), ("ecdf_band_gated", False),
            ("player_season_sigma", 0.0),
            ("quantile_scope", ""), ("quantile_reason", ""),
            ("quantile_ks_train", float("nan")),
            ("quantile_ks_validation", float("nan")),
            ("quantile_ks_mc", float("nan")), ("quantile_ks_gated", False),
            ("quantile_weighting", ""))},
        "git_sha": provenance.get("git_sha", ""),
        "built_at": provenance.get("built_at", ""),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict, heads: tuple[str, ...] | None = None,
        write: bool = True) -> dict[str, Path]:
    """Card every head on disk, verify each one, and write the nine artifacts.

    `heads` is a **verification** subset, not a rebuild subset, which is why `write`
    defaults off with it at the CLI. Unlike `make posteriors` — a day of sampler time,
    where re-running one group and merging the manifest by head is the only sane
    workflow — this target is a couple of minutes for all twenty, so a partial rebuild buys
    nothing and would quietly leave `model_card_index.csv` describing three heads.
    """
    started = time.perf_counter()
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    source = posteriors_dir(cfg, WINDOW)

    print(f"Model cards — the dashboard-shaped view of every fitted head.\n"
          f"  reading persisted posteriors from {source}; nothing is refitted and no "
          f"sampler runs.")
    artifacts = load_all(source, heads=list(heads) if heads else None)
    if not artifacts:
        raise FileNotFoundError(
            f"no posterior artifacts under {source}. Run `make posteriors` first — it is "
            f"the only target in the simulation layer that needs a CmdStan toolchain.")
    # The window is a capability check, not a label check: a head fitted at `train_val` has
    # read 2022-23 and 2023-24 through its coefficients, and no frame-level guard can see
    # that in a histogram drawn from them.
    require_window(artifacts, WINDOW)
    print(f"  {len(artifacts)} heads at the `{WINDOW}` window. The test split is LOCKED; "
          f"every row carries `split` in {list(SPLITS)}.")

    print("\n── rebuilding each head's own design frames ──")
    frames = build_frames(cfg, artifacts)

    index, coefficients, features, correlations, densities = [], [], [], [], []
    ecdf, calibration, quantile, samples = [], [], [], []
    print(f"\n── verifying the recipe against each head's variant ladder, and drawing "
          f"{PRED_DRAWS} predictive\n   draws per head over at most {PRED_ROWS:,} rows a "
          f"split ──")
    for head in sorted(artifacts):
        art, head_frames = artifacts[head], frames[head]
        check = verify(art, head_frames)
        terms = coefficient_rows(head, art)
        coefficients.extend(terms)
        features.extend(feature_rows(head, head_frames))
        head_corr = correlation_rows(head, head_frames)
        correlations.extend(head_corr)
        head_density = density_rows(head, head_frames, head_corr)
        densities.extend(head_density)

        predictive = predictive_tables(head, art, head_frames, cfg)
        check_predictive(head, predictive["summary"])
        ecdf.extend(predictive["ecdf"])
        calibration.extend(predictive["calibration"])
        quantile.extend(predictive["quantile"])
        samples.extend(predictive["sample"])

        index.append(index_row(head, art, head_frames, check, len(terms),
                               predictive["summary"],
                               n_density_pairs=len(density_pairs(head_corr))))
        if not check["ladder_features_agree"]:
            # `impute` mints a flag when EITHER frame carries a hole, and `posteriors.py`
            # ran the ladder against a 400-row probe where this module runs it against the
            # whole validation frame — so the two can disagree about the flag set without
            # either being wrong. The recipe stays the authority and the design still has
            # to reproduce; this is a note, not a failure.
            print(f"  /!\\  {head}: the variant ladder's feature list differs from the "
                  f"persisted recipe's ({len(head_frames.ladder_features)} against "
                  f"{len(art.recipe.features)}). The recipe is the authority and the "
                  f"design reproduces exactly, so nothing is wrong — but the ladder is "
                  f"minting or dropping a column against the frame the fit saw.")
        summary = predictive["summary"]
        print(f"  {head:<18s} {check['n_fit']:>7,} fit / {check['n_validation']:>6,} val "
              f"· {len(art.recipe.features):>2d} features · "
              f"{len(density_pairs(head_corr)):>2d} joint densities · design "
              f"{check['recipe_design_error']:.1e} ({check['design_check']}) · "
              f"round-trip {check['roundtrip_prediction_error']:.1e} · PASS")
        bias = summary["predictive_bias"]
        print(f"  {'':<18s} predictive over {summary['n_predictive_train']:>7,}"
              f" / {summary['n_predictive_validation']:>6,} rows · "
              f"{summary['response_label']} · band MC {summary['ecdf_band_mc']:.4f}"
              f"{'' if summary['ecdf_band_gated'] else ' (ungated)'} · mean "
              f"{f'{bias:+.2%}' if np.isfinite(bias) else 'not checkable'}"
              f" ({summary['predictive_check']})")
        if summary["quantile_scope"] == "drawn":
            print(f"  {'':<18s} quantile residual KS "
                  f"{summary['quantile_ks_train']:.4f} train / "
                  f"{summary['quantile_ks_validation']:.4f} val — a DISTANCE, never a "
                  f"pass/fail · half-sample {summary['quantile_ks_mc']:.4f}"
                  f"{'' if summary['quantile_ks_gated'] else ' (ungated)'}")
        else:
            print(f"  {'':<18s} quantile residual NOT DRAWN — {summary['quantile_reason']}")

    tables = {
        "model_card_index.csv": pd.DataFrame(index),
        "model_card_coefficients.csv": pd.DataFrame(coefficients),
        "model_card_features.csv": pd.DataFrame(features),
        "model_card_feature_corr.csv": pd.DataFrame(correlations),
        "model_card_feature_density.parquet": pd.DataFrame(densities),
        "model_card_ecdf.csv": pd.DataFrame(ecdf),
        "model_card_calibration.csv": pd.DataFrame(calibration),
        "model_card_quantile.csv": pd.DataFrame(quantile),
        "model_card_sample.parquet": pd.concat(samples, ignore_index=True),
    }
    leading = {"model_card_coefficients.csv": ["head", "term", "term_family", "term_role"],
               "model_card_features.csv": ["head", "feature", "split", "bin_index"],
               "model_card_feature_corr.csv": ["head", "split", "feature_x", "feature_y"],
               "model_card_feature_density.parquet": ["head", "feature_x", "feature_y",
                                                      "split", "x_index", "y_index"],
               "model_card_ecdf.csv": ["head", "split", "grid_index", "value"],
               "model_card_calibration.csv": ["head", "split", "panel", "x_index",
                                              "y_index"],
               "model_card_quantile.csv": ["head", "split", "panel", "x_index",
                                           "y_index", "x", "y"]}

    paths = {}
    print()
    for name, table in tables.items():
        _check_splits(table, name)
        columns = leading.get(name)
        if columns:
            table = table[columns + [c for c in table.columns if c not in columns]]
        dest = out_dir / name
        if not write:
            print(f"Would write {len(table):,} rows x {table.shape[1]} columns → {dest}")
            continue
        if dest.suffix == ".parquet":
            # The two binary artifacts, for the same reason from opposite directions. The
            # sample is 54,000 rows of three float columns and nothing else, and parquet
            # keeps the `float32` a scatter overlay needs where a CSV would widen every one
            # back to text. The density is the mirror image — two long feature names
            # repeated on every one of its 93,608 cells — and dictionary encoding is the
            # measured difference between 10.5 MB of restated column names and 0.55 MB.
            table.to_parquet(dest, index=False)
        else:
            # Six significant digits. The per-feature statistics repeat on every bin row —
            # deliberately, so a page groups by feature and gets the histogram and the
            # summary table from one read — and full float64 repr triples the file for
            # precision no histogram can draw.
            table.to_csv(dest, index=False, float_format="%.6g")
        paths[dest.stem] = dest
        print(f"Wrote {len(table):,} rows x {table.shape[1]} columns → {dest}")

    worst_band = max((r["ecdf_band_mc"] for r in index
                      if r["ecdf_band_gated"] and np.isfinite(r["ecdf_band_mc"])),
                     default=float("nan"))
    worst_bias = max((r["predictive_bias"] for r in index
                      if np.isfinite(r["predictive_bias"])), key=abs, default=float("nan"))
    ungated = sorted(r["head"] for r in index if not r["ecdf_band_gated"])
    unchecked = sorted(r["head"] for r in index if r["predictive_check"] == "none")
    drawn = sorted(r["head"] for r in index if r["in_draw_path"])
    print(f"\n{len(index)} heads carded across "
          f"{len(set(r['model_class'] for r in index))} model classes, "
          f"{len(drawn)} of them in the simulator's draw path "
          f"({', '.join(sorted(r['head'] for r in index if not r['in_draw_path']))} "
          f"are not); worst recipe design error "
          f"{max(r['recipe_design_error'] for r in index):.2e} against a "
          f"{DESIGN_TOL:.0e} bar; {time.perf_counter() - started:.1f}s.")
    print(f"Predictive: {PRED_DRAWS} draws per head; worst 95%-ribbon half-sample "
          f"disagreement {worst_band:.4f} against a {ECDF_BAND_TOL} bar, worst drawn-mean "
          f"gap {worst_bias:+.2%} against a\n{PREDICTIVE_BIAS_TOL:.0%} bar. The band is "
          f"checked at this budget rather than assumed stable at it — "
          f"{', '.join(ungated) or 'no head'} sits under {BAND_MIN_ROWS:,} rows and is "
          f"reported rather than gated,\nand {', '.join(unchecked) or 'no head'} reports a "
          f"mean the drawn predictive cannot be compared against.")

    drawn_ks = [r for r in index if r["quantile_scope"] == "drawn"]
    ks_values = [r[column] for r in drawn_ks
                 for column in ("quantile_ks_train", "quantile_ks_validation")
                 if np.isfinite(r[column])]
    worst_ks_mc = max((r["quantile_ks_mc"] for r in drawn_ks
                       if r["quantile_ks_gated"] and np.isfinite(r["quantile_ks_mc"])),
                      default=float("nan"))
    out_of_scope = sorted(r["head"] for r in index if r["quantile_scope"] != "drawn")
    span = (f"{min(ks_values):.4f}–{max(ks_values):.4f}" if ks_values else "no head")
    print(f"Quantile residuals: {len(drawn_ks)} of {len(index)} heads drawn "
          f"({', '.join(out_of_scope) or 'no head'} out of scope); KS distance spans "
          f"{span} across heads and splits and is "
          f"REPORTED, never\nthresholded — at these sample sizes a uniformity test rejects "
          f"every head. The one bar is on the draw budget: worst half-sample KS "
          f"disagreement {worst_ks_mc:.4f} against a {KS_MC_TOL} bar.")
    return paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Emit the dashboard's model-card artifacts. No CmdStan, no refit.")
    # A verification subset, deliberately not a rebuild subset: the full run is ~5 s, so a
    # partial write would only ever leave the shipped artifacts describing three heads.
    parser.add_argument("--check", default="",
                        help="comma-separated heads to build and verify WITHOUT writing; "
                             "for debugging one head's design check")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    subset = tuple(h.strip() for h in args.check.split(",") if h.strip())
    run(cfg, heads=subset or None, write=not subset)
