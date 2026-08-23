"""The availability head as a full Bayesian fit — the same likelihood, in Stan.

`src/models/availability.py::BetaBinomialGLM` is a validated point-MLE beta-binomial:
held-out CRPS 10.795 games on 911 test rows, 19 features, beating ridge (10.896), a GBM
(10.888) and a league/age baseline (13.614), and worth −211 dk_pts of season-total MAE.
This module ports **that exact likelihood** to Stan and draws `(beta, rho)` from the
posterior instead of fixing them at the optimum.

## Why, given the marginal metric will barely move

Stated plainly because the wrong justification is easy to reach for: at ~10,300 training
rows against 20 parameters the posterior is sharply concentrated, so posterior-mean
coefficients and the MLE agree closely and GP-marginal CRPS should improve very little. If
the case were argued on CRPS it would fail.

**The argument is the joint distribution across players.** Every player shares `beta`, so
one posterior draw shifts the whole board's availability together — a correctly calibrated
source of cross-player correlation that arrives free with the fit rather than as an
invented copula. For a draft portfolio "how wrong could my entire board be at once" is a
different and more important question than "how wrong is this one player", and only the
posterior answers it. `docs/predictions-plan.md` needs exactly that and nothing in this
repo previously supplied it.

## What this module checks, and the three rows it prints

The port is verified rather than assumed, by fitting both and scoring them with the *same*
code — `evaluate`, `crps` and `pit_values` are imported from the MLE module, not
reimplemented, so a metric difference cannot be a metric-implementation difference:

- `beta_binomial_mle` — the existing head, unchanged. The reference.
- `stan_plug_in` — posterior means substituted for the MLE's optimum, one beta-binomial
  per row. This is the row that should *match* the MLE, because the prior is set to
  `1/sqrt(2*l2)`, which makes the posterior mode exactly the penalized MLE.
- `stan_posterior` — the predictive distribution properly integrated over the posterior,
  `p(y) = mean_s BetaBinom(y | n, mu_s, rho_s)`. The row the simulator should consume.

**The integrated predictive is not necessarily *wider* per player, and expecting it to be
is a trap.** By the law of total variance the mixture adds `Var_th(E[Y|th])`, but it also
replaces `Var(Y|th_bar)` with `E_th[Var(Y|th)]`, and the conditional variance
`n*mu*(1-mu)*[1 + (n-1)*rho]` is **concave in mu** — so Jensen pushes the other way.
Measured on this design the two nearly cancel (+0.046 against -0.082 games^2, verified
against the pmf to 1.3e-10). The marginal width is a red herring; see
`board_correlation` for the quantity that is not.

## What ships, since 2026-08-11: a 2012-13 fitting window and a role-graded dispersion

`make availability-window` laddered the fitting window against a season term against the
dispersion's grading, and `three_point_era__none__role` won: validation CRPS **9.8247**
against the incumbent's 10.0057, a paired gap of −0.181 with a 95% interval of
[−0.257, −0.103], PIT KS 0.0588 against 0.0939, and boundary error cut 43%. Both halves are
defaults here — `first_season="2012-13"` and `role_rho=True` — so every consumer of this
head gets the shipped configuration by loading it rather than by remembering to ask.
`docs/availability-window-plan.md` is the evidence and should not be re-litigated here.

Two things about that are worth stating where the code is, because both are easy to get
wrong:

**The window is applied to the FITTING rows, inside `fit`, and nowhere else.**
`availability_design` is imported by `stan_minutes`, `stan_composition`,
`stan_games_played`, `model_cards`, `sim/season`, `season_terms` and `final_evaluation`,
and filtering it would silently re-scope six heads that never asked for a window. Scoring
stays on the full frame it is handed: a shorter fitting window is a bias-variance trade on
the fit, not a claim about which rows may be predicted.

**A season trend is a measured NULL, and the reason is worth keeping.** The trend is the
only instrument that closes both tails, and it does so by shifting the whole distribution
down — so the body blows out and both CRPS and PIT degrade. A location instrument cannot
fix a shape defect. Do not add one back.

## And since 2026-08-12: a second component for the disrupted season

`mixture=True` adds `pi_i * BetaBinom(mu_low, rho_low)` beside the main component, with
eight covariates on `pi` — age, prior absence, playoff workload. The hypothesis it encodes
is that the low tail is not a frailty at all: an Achilles rupture in October is a different
*event*, not an extreme draw of a per-game rate, and under `a = mu(1-rho)/rho` the frailty's
shape and its variance are one parameter, so no value of `rho` can put the mass where the
data wants it. `docs/availability-window-plan.md` §7 laddered five likelihoods and this one
halved the boundary error while tying on CRPS; §7h is the port.

Three things about it belong where the code is.

**`False` recovers the single-component head EXACTLY, and that is asserted rather than
intended.** In `betabinomial_glm.stan` the mixture is `P = 0`, whose parameters are then
zero-length; and even with `P > 0`, `theta = 0` leaves the target **bit for bit** unchanged,
because the mixture enters as an additive correction to the untouched beta-binomial
statement. Both are pinned on Stan's own `log_prob` in `tests/test_stan_heads.py`. That is
the rollback path for a file six heads share.

**The head is selected on tail calibration with a CRPS guard**, which is a change of rule
recorded as D1 in `docs/availability-window-plan.md` §8 and stated *before* this arm was
ported. Do not re-decide it on mean CRPS; the mixture ties there by construction.

**`predict_mean` returns the PREDICTIVE mean, not the main component's.** A mixture's mean is
a weighted mean, and every scorer turns that number into MAE and R² in games — returning the
main component's would credit the head with an accuracy its own predictive does not have.

## The trap that carries over, and the one that does not

**`n = max(team_games, gp)` is still required.** 13 traded player-seasons (0.12%) have
`gp > team_games`, because their two teams' schedules overlap. `beta_binomial_lpmf` is
undefined there, and since the log-likelihood is a sum those rows take the whole fit down
at *every* value of rho. Under HMC this is worse than under L-BFGS-B, not better: a
non-finite target poisons the trajectory rather than merely stopping an optimizer.
`build_design` already applies the fix, and `assert_binomial_support` re-checks it here
because a silent violation is fatal and cheap to rule out.

**The numeric-gradient failure does not carry over.** The MLE needs an analytic gradient
and an alternating fit because L-BFGS-B on a ~1e5-magnitude objective stops on
finite-difference noise. Stan differentiates the model exactly, so `beta` and `rho` are
sampled jointly with no alternation.

## Which rows the three numbers are measured on

**Validation**, since 2026-08-05. Every figure printed here — the CRPS triple, the board
correlation, the PIT deciles — describes the fit on `train` scored against the validation
seasons, and `src/models/held_out.py` raises on anything that reaches past them.

That is a demotion in what the numbers *are*, and worth stating plainly. A port check is
not a selection, so scoring it on the held-out seasons was never the failure mode the lock
was built for. But it is also not the end-of-project measurement, and a module that reads
test "just to see" is exactly how the test column stops feeling special — which is how the
games-played head came to settle a shipping decision on it. `board_correlation` is the
sharper case: it is a **simulator input**, a statement about how much a whole draft board
moves together, so calibrating it on the seasons the simulator is later backtested against
is leakage the split cannot catch, in the same way `game_level_dispersion` and the residual
copula already take `train_val`.

`fit_and_score` is the whole measurement, and `src/final_evaluation.py` calls **the same
function** on `(full_train, test)` when the workflow is finished. So the held-out reading is
not a reimplementation of this one; it is this one, run once, on the other frame.

Usage:
    python -m src.models.stan_availability
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import betabinom

from src.data.fetch import _season_start_year
from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.availability import build_panel, season_availability
from src.models.availability import (EPS, FEATURE_COLS, RHO_MAX, RHO_MIN,
                                     AvailabilityModel, BetaBinomialGLM,
                                     _sigmoid, build_design, evaluate,
                                     pit_table, rung_zero, season_start_dates)
from src.models.held_out import selection_split
from src.models.stan_utils import (GAMMA_SCALE, MU_LOW_MAX, YearTerm, chain_summary,
                                   compile_model, diagnostics_frame, pi_block,
                                   posterior, prior_sd_for_l2, rho_block, sample,
                                   standardized, thin, warn_if_unconverged)

MODEL = "betabinomial_glm"

# ── The shipped configuration ────────────────────────────────────────────────────────
# Mirrored in `configs/default.yaml`'s `stan.availability` block, which is what `run`
# reads. These are the CLASS defaults so that every other consumer of the head —
# `posteriors`, `model_cards`, `season_terms`, `final_evaluation` — gets what ships
# without having to remember to pass it. `docs/availability-window-plan.md` §4 is the
# evidence; `None` / `False` recover the incumbent exactly.

# First season kept in the FITTING rows. 2012-13 is `availability_window.WINDOWS`'s
# `three_point_era`, which won the ladder on CRPS, PIT and boundary coverage at once.
# Deliberately not 2017-18, where the sup-F scan puts the break: the break's location and
# the best fitting window are different questions, and five target seasons is too few.
FIRST_SEASON = "2012-13"

# One dispersion per prior-MPG role bucket rather than one for every player. Milder than
# the composition head's 2.07x — 1.26x on the full window, 1.47x on this one — and it
# never hurt a single metric in the ladder.
ROLE_RHO = True

# The column the buckets are cut on, and the column the assignment is written to. Named
# constants because they are a *contract* rather than an implementation detail: the
# persisted posterior carries the cut as a recipe step, so `posteriors.py` and every
# consumer downstream of it addresses those two columns by name.
ROLE_COL = "minutes_per_game_lag1"
ROLE_BIN_COL = "rho_bin"

# ── The low-availability mixture, ported 2026-08-11 ──────────────────────────────────
#
# `docs/availability-window-plan.md` §7c laddered five likelihoods and `mixture` won the
# selector: `boundary_tail_error` **0.0109** against the single-component head's 0.0201,
# `shoulder_error` −0.0021 [−0.0086, −0.0010], and a **tie** on CRPS (+0.011, interval
# spanning zero). D1 in `docs/availability-window-plan.md` §8 is the rule that makes that a
# ship: this head is selected on tail calibration with a CRPS non-inferiority guard, stated
# before the arm was measured rather than reverse-engineered from it.
#
# `False` recovers the single-component head EXACTLY — `P = 0` in the Stan source, whose
# parameter block is then zero-length. That is the rollback path and it is asserted on
# Stan's own `log_prob`, not merely intended.
MIXTURE = True

# `pi`'s covariate block. `docs/potential-to-dos.md` item 5 names three families — age,
# prior absence, playoff workload — and this is that list made concrete. It is a **shipped
# choice** (D5) rather than a default: it lands in the persisted `DesignRecipe`, and the
# fitted `pi` runs from 1.2% to 10.8% across the 10th and 90th percentiles of players, so
# the block carries real signal. Deliberately not the whole 19-column feature block:
# nineteen more parameters on 4,027 rows would be measuring the penalty rather than the
# mechanism.
#
# Held here rather than imported from `availability_window`, which imports `season_terms`,
# which imports this module — a top-level import would be a cycle. A test pins the two
# lists equal, so the ladder that selected the arm and the head that ships it cannot drift.
PI_FEATURES = ["age", "age_sq", "gp_share_lag1", "trailing_missed_lag1",
               "n_spells_lag1", "longest_spell_lag1",
               "playoff_games_lag1", "career_minutes_lag1"]

# ── The preseason block, adopted 2026-08-13 ──────────────────────────────────────────
#
# `docs/preseason-plan.md` P2. Current-season preseason games are a legitimate input now
# that the draft happens after the preseason, and they enter this head as **ten columns on
# `beta`** — P1's full block: preseason minutes volume, three participation levels, two
# difference-coded deltas, and the missing-preseason indicator split on P1's own age cells.
# Zero across the block recovers the head that shipped before today exactly, which is what
# makes `preseason: false` a real rollback and not a different model.
#
# **This is the arm the FITTING HALF selected, and shipping it reverses P1 decision 4** —
# which said the availability block should be *smaller* than P1's seven columns because a
# ridge overfit them. At this head's own unit the wider block is better: on the rolling
# harness it beats the arm declared before the run by −0.0977 CRPS [−0.1569, −0.0408] and
# −0.0014 of `boundary_tail_error` [−0.0019, −0.0010], both clear of zero, at 8 of 10
# origins. The promotion rests on the fitting half rather than on validation, which is what
# keeps it off the selection split.
#
# **The whole block is adopted against a gate that FAILED as written**, and that is on the
# record rather than smoothed over. P2's bar was a conjunction — a validation CRPS interval
# clear of zero with the boundary held, AND the rolling harness agreeing — and validation
# could not resolve the arm while the rolling harness passed both halves. The two readings do
# not conflict: validation's interval contains the rolling estimate and is 2.4x wider on 4.6x
# fewer rows. The bar was written against the opposite failure and had no clause for this
# one. The call was taken explicitly by the project owner on 2026-08-13.
#
# ⚠️ **A complete preseason is now a production PRECONDITION.** `pre_missed_tail_share` and
# `pre_played_final_game` are read over the preseason's tail and do not exist until it is
# over. A centred-volume arm that survives a truncated capture was adopted and withdrawn the
# same day: the premise behind it — DK contests filling before the final preseason game — was
# contradicted by the owner's experience drafting after the 2025-26 preseason ended. The
# runbook's Oct 17-20 window is load-bearing rather than advisory, and if that ever turns,
# this block is the thing that has to change.
#
# ⚠️ Nothing here may be pushed into `availability_design`. Six other modules import that
# builder — the minutes, composition and games-played heads, the exchangeability and
# no-prior ladders, and the season-term ablation — and a column that is structurally zero
# before 2004-05 must not enter any of them by accident. `head_design` below is this head's
# own path, and the `attach_absence_mix` precedent is the same shape.
#
# Held here rather than imported from `availability_preseason`, which reaches
# `availability_window` -> `season_terms` -> this module: a top-level import is a cycle, the
# identical reason `PI_FEATURES` is duplicated above. A test pins the two lists equal.
PRESEASON = True

PRESEASON_COLS = ["pre_log_min", "pre_gp_share", "pre_missed_tail_share",
                  "pre_played_final_game", "pre_d_mpg", "pre_d_min_share_late",
                  "pre_missing__<24", "pre_missing__24-27", "pre_missing__28-31",
                  "pre_missing__32+"]

# Where the chains start on `theta`, one per chain. The point MLE of this likelihood needed
# **multi-start** — begun at its own nesting point a three-class mixture sat on the bound
# and reproduced the incumbent to four decimals — so four chains launched from the same
# place would make per-chain agreement a statement about the initializer. 0 is the nesting
# point, 0.08 and 0.25 bracket the plausible disruption rate (observed P(GP < 10) is 8.15%
# on validation), and 0.50 is past any reading of it. `chain_summary` is what reads the
# answer back.
THETA_INITS = (0.02, 0.08, 0.25, 0.50)

# Draws kept for the posterior-predictive mixture. Each draw costs one (rows x games+1)
# beta-binomial evaluation, so this trades wall clock against Monte Carlo error in the
# predictive. 400 puts the MC error on a CRPS of ~10.8 games well below 0.001.
PREDICTIVE_DRAWS = 400
# Draws pushed through scipy at once. Purely a memory knob: 60 x 911 x 84 doubles is
# ~37 MB, where the full 400 at once would be ~245 MB.
CHUNK = 60

INTERCEPT_SCALE = 5.0


def assert_binomial_support(design: pd.DataFrame) -> pd.DataFrame:
    """`0 <= gp <= team_games` on every row, or the log-likelihood is non-finite.

    `build_design` already takes `n = max(team_games, gp)`, so this should never fire. It
    is here because the failure is silent in the data and catastrophic in the sampler:
    thirteen bad rows out of ten thousand make the target non-finite everywhere, and the
    symptom surfaces as an uninterpretable initialization error rather than as a data
    problem.
    """
    y = design["gp"].to_numpy(int)
    n = design["team_games"].to_numpy(int)
    bad = (y < 0) | (n < 0) | (y > n)
    if bad.any():
        rows = design.loc[bad, ["player_id", "season", "gp", "team_games"]].head()
        raise ValueError(
            f"{int(bad.sum())} rows violate 0 <= gp <= team_games — the beta-binomial "
            f"likelihood is undefined there and the summed target is non-finite at every "
            f"rho:\n{rows.to_string(index=False)}")
    return design


def availability_design(cfg: dict) -> pd.DataFrame:
    """The point-in-time-checked player-season design, from cache when it exists.

    `make availability` already writes `availability_panel.parquet` and
    `availability_features.parquet`; rebuilding them from 30 seasons of CSVs costs a
    couple of minutes and produces the same frames. The cache is used when present and
    the source is printed either way, because a silently stale artifact would move every
    number downstream of it.

    Shared with `stan_minutes.py`, which needs the identical feature block — the minutes
    head is `min | available`, the next link in the same chain.

    ## The §16 lag-recovery ladder enters HERE, and that is the whole of its blast radius

    `stan.availability.lag_ladder` is read at this one point, because this function is the
    choke point `stan_minutes`, `stan_composition`, `stan_games_played`, `model_cards`,
    `season_terms`, `availability_no_prior` and the simulator all reach their rows through.
    An empty list — the default — builds the pre-ladder design exactly and nothing below
    can tell the difference.

    ⚠️ **A non-empty list moves all seven of them at once**, which is deliberate and is why
    the key defaults off: a recovered player is a real roster spot, and because the minutes
    allocation is zero-sum he takes minutes from his teammates rather than appearing beside
    them. `docs/availability-window-plan.md` §16c states the list; turning the key on is a
    separate decision from measuring the ladder.
    """
    from src.features.availability import load_artifacts
    from src.models.availability import lag_ladder

    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]
    ladder = lag_ladder(cfg)

    panel_path = features_dir / "availability_panel.parquet"
    try:
        panel, frame = load_artifacts(features_dir)
        frame = frame[frame["window"] == "full"].drop(columns=["window"])
        stamp = pd.Timestamp(panel_path.stat().st_mtime, unit="s")
        print(f"  availability artifacts from cache ({stamp:%Y-%m-%d %H:%M}) — "
              f"run `make availability` if the panel is stale")
    except FileNotFoundError:
        print("  no cached availability artifacts; rebuilding the panel from raw logs")
        panel = build_panel(seasons, raw_dir)
        frame = season_availability(panel, "full")

    if ladder is not None:
        print(f"  §16 lag-recovery ladder ON at rung(s) {list(ladder.rungs)} — this "
              f"widens the design for EVERY consumer of it")
    design = build_design(frame, seasons, raw_dir, season_start_dates(panel),
                          ladder=ladder)
    return assert_binomial_support(design)


def head_features(preseason: bool | None = None) -> list[str]:
    """The availability head's feature list — `FEATURE_COLS`, plus the preseason block.

    One expression of it, because five consumers need the same answer: this module's port
    check, `posteriors.availability_artifact`, `model_cards`, `final_evaluation` and
    `src/sim/season.py`. A second copy is a place where the persisted recipe and the frame
    rebuilt against it can disagree, and `model_cards.verify` would be the thing that
    finally noticed.
    """
    on = PRESEASON if preseason is None else bool(preseason)
    return list(FEATURE_COLS) + (list(PRESEASON_COLS) if on else [])


def head_design(cfg: dict, preseason: bool | None = None,
                design: pd.DataFrame | None = None) -> pd.DataFrame:
    """`availability_design` plus the preseason block — **this head's path and no other's**.

    Separate from `availability_design` deliberately and permanently. That builder is how
    six other modules reach their rows, and the preseason columns are structurally zero
    before 2004-05; putting them there would re-scope the minutes head, the composition
    head, the games-played spell process, the exchangeability and no-prior ladders and the
    season-term ablation, none of which asked for them and none of which would raise. The
    `attach_absence_mix` precedent, applied to a block that actually ships.

    The block is built by `availability_preseason.attach_preseason`, which is the function
    the P2 ladder itself used — not a reimplementation of it, so the coefficients this head
    fits are coefficients on the columns that were measured.

    Both guards are assertions rather than fills. A missing column means the panel is stale
    (`make preseason`), and a NaN means a row reached the head with an undefined block,
    which under a beta-binomial likelihood is a silent non-fit rather than an error.
    """
    # `design` lets the forward path (`features/forward_design.py`) bring its own rows
    # through the SAME preseason attachment; `None` is today's behaviour exactly.
    design = availability_design(cfg) if design is None else design
    if not (PRESEASON if preseason is None else bool(preseason)):
        return design

    # Function-level for the cycle `PRESEASON_COLS` documents, and because a consumer that
    # only wants the shared builder should not pay for the import.
    from src.features.availability import load_artifacts
    from src.models.availability_preseason import attach_preseason

    features_dir = Path(cfg["data"]["features_dir"])
    panel_path = features_dir / "preseason.parquet"
    if not panel_path.exists():
        raise FileNotFoundError(
            f"{panel_path} is missing and the availability head ships a preseason block — "
            f"run `make preseason`, or set `stan.availability.preseason: false` to fit the "
            f"pre-2026-08-13 head exactly.")
    av_panel, _ = load_artifacts(features_dir)
    out = attach_preseason(design, pd.read_parquet(panel_path), av_panel,
                           cfg["data"]["seasons"])

    missing = [c for c in PRESEASON_COLS if c not in out.columns]
    if missing:
        raise ValueError(f"`attach_preseason` did not produce {missing}; the panel or the "
                         f"shipped block list has moved")
    if out[PRESEASON_COLS].isna().any().any():
        bad = [c for c in PRESEASON_COLS if out[c].isna().any()]
        raise ValueError(f"NaN in the shipped preseason block: {bad}. Zero means 'no new "
                         f"information' and is a *value*; NaN is a build failure.")
    return out


def restrict_window(frame: pd.DataFrame, first_season: str | None) -> pd.DataFrame:
    """The fitting rows, cut to a recent suffix of seasons.

    **Applied to the fitting rows only, and never to `availability_design`.** Six other
    modules import that builder and a filter inside it would silently re-scope the minutes
    head, the composition head, the games-played spell process and the simulator, none of
    which asked for a window and none of which would raise.

    The design is still *built* over every season regardless of this, because the lag
    columns reach back three seasons — trimming the frame earlier would drop the window's
    own first cohort instead of windowing it.

    `availability_window.restrict_window` is the same cut keyed on a start year; this one
    takes the season label the config carries, so `first_season: 2012-13` is written the
    way every other season in this repo is written.
    """
    if first_season is None:
        return frame
    first_year = _season_start_year(str(first_season))
    years = frame["season"].astype(str).map(_season_start_year).to_numpy()
    return frame[years >= first_year]


def role_edges(role_rho: bool = ROLE_RHO) -> list[float]:
    """The bucket edges `role_bins` cuts on — the fitted state the artifact has to carry.

    Under a shared dispersion this is one interval covering the whole line, which is
    `n_rho = 1` and reproduces the all-ones assignment exactly. Written as edges rather than
    special-cased downstream so the persisted recipe has one shape in both arms: a consumer
    reconstructs `rho_bin` by cutting on these, and never has to know which arm it holds.
    """
    return [float(e) for e in ROLE_EDGES] if role_rho else [-np.inf, np.inf]


def role_bins(frame: pd.DataFrame, role_rho: bool = ROLE_RHO) -> np.ndarray:
    """1-based prior-MPG role bucket per row — the `rho_bin` the Stan source gathers on.

    `season_effects.ROLE_EDGES` are constants rather than quantiles of anything, so there
    is no split concern and a validation row lands in the same bucket it would have landed
    in during fitting. Role is **prior-season** MPG: known before opening night, so this
    grades the dispersion on information the head already holds rather than on the target.

    `role_rho=False` returns all-ones, which with `n_rho = 1` is the shared-dispersion head
    exactly.

    A row outside the edges — no prior minutes at all, or an implausible one above the top
    edge — falls into the **lowest** bucket, the conservative direction since the fringe
    bucket carries the widest dispersion. Today the design has no such row, so this is a
    guard rather than a live branch — but the *simulator* reaches it constantly, through
    `sim/season.availability_rho_bin`, for every rostered player the head has no row for.

    **That fallback is measured rather than assumed** (`make availability-no-prior`,
    `docs/availability-window-plan.md` §8a). A three-class imputed bucket for that population
    was specified and is **withdrawn**: across draft buckets the no-prior population's
    realized dispersion spans 1.1557× while its realized *level* spans 3.3260×, so the rule
    graded the flat axis — and applied as written it would hand a lottery top-5 pick a
    narrower `rho` than he realizes. The lowest bucket is too narrow on eight of the nine
    measured no-design groups and too wide on one, by 0.0184 — which is the argument for
    keeping it, since too wide is the safe direction and it is barely that.
    """
    if not role_rho:
        return np.ones(len(frame), dtype=int)
    mpg = frame[ROLE_COL].to_numpy(dtype=float)
    idx = pd.cut(mpg, ROLE_EDGES, labels=False)
    return np.nan_to_num(np.asarray(idx, dtype=float), nan=0.0).astype(int) + 1


class StanAvailability(AvailabilityModel):
    """Beta-binomial availability head, fitted by NUTS.

    Subclasses the MLE module's `AvailabilityModel` on purpose: `evaluate` then scores it
    through exactly the same code path as the other four candidates, so the comparison is
    about the fit and not about two implementations of CRPS.
    """

    def __init__(self, l2: float = 1.0, features: list[str] | None = None,
                 pmf_mode: str = "posterior", name: str | None = None,
                 chains: int = 4, warmup: int = 1000, samples: int = 1000,
                 seed: int = 42, predictive_draws: int = PREDICTIVE_DRAWS,
                 year_column: str | None = None, metric: str | None = None,
                 first_season: str | None = FIRST_SEASON,
                 role_rho: bool = ROLE_RHO, mixture: bool = MIXTURE,
                 pi_features: list[str] | None = None,
                 gamma_scale: float = GAMMA_SCALE, mu_low_max: float = MU_LOW_MAX):
        if pmf_mode not in ("posterior", "plug_in"):
            raise ValueError(f"pmf_mode must be 'posterior' or 'plug_in'; got {pmf_mode!r}")
        self.l2 = l2
        self.features = features or FEATURE_COLS
        self.pmf_mode = pmf_mode
        self.name = name or f"stan_{pmf_mode}"
        self.chains, self.warmup, self.samples, self.seed = chains, warmup, samples, seed
        self.predictive_draws = predictive_draws
        self.metric = metric
        self.first_season = first_season
        self.role_rho = bool(role_rho)
        self.mixture = bool(mixture)
        self.pi_features = list(pi_features or PI_FEATURES)
        self.gamma_scale, self.mu_low_max = float(gamma_scale), float(mu_low_max)
        self.year = YearTerm(year_column, seed=seed, stream=self.name)

    # ── The dispersion bins ───────────────────────────────────────────────────

    @property
    def n_rho(self) -> int:
        return len(ROLE_LABELS) if self.role_rho else 1

    @property
    def rho_labels(self) -> list[str]:
        return list(ROLE_LABELS) if self.role_rho else ["shared"]

    def bins(self, df: pd.DataFrame) -> np.ndarray:
        return role_bins(df, self.role_rho)

    def fitting_rows(self, train: pd.DataFrame) -> pd.DataFrame:
        """The rows this head fits on — rung 0, then the window, and nowhere upstream.

        **`rung_zero` first, and it is the whole of §16's imputation-only claim as code.**
        `stan.availability.lag_ladder` widens `availability_design` for every consumer of
        it, and §16i's shipped arm scores those rows with a posterior fitted **before** the
        ladder existed — the arm that admitted them to the fit (`staleness`) was built,
        priced and rejected at +0.7310 [−1.4152, +2.8585]. So the recovered rows must never
        reach a fit, and this method is the one place every path that fits this head goes
        through: `run`, `fit_and_score`'s two point-MLE references,
        `posteriors.availability_artifact`, `model_cards` and `src/final_evaluation.py`
        alike. `availability_lag.fit_arms` states the same restriction for its own `impute`
        arm, which is what this reproduces in the shipped head.

        A no-op while `stan.availability.lag_ladder` is `[]`: `ladder_recovered` returns
        all-False on a frame with no `lag_rung` column.
        """
        return restrict_window(rung_zero(train), self.first_season)

    # ── Fitting ───────────────────────────────────────────────────────────────

    def fit(self, train: pd.DataFrame) -> "StanAvailability":
        # The window bites FIRST, so the scaler, the year block and the dispersion bins
        # are all built from the rows that are actually fitted. Announced rather than
        # silent: a head that quietly drops two thirds of its training rows is exactly the
        # failure `availability_design` must never be allowed to have.
        n_offered = len(train)
        train = self.fitting_rows(train)
        if len(train) != n_offered:
            print(f"  window {self.first_season}+: fitting on {len(train):,} of "
                  f"{n_offered:,} player-seasons (scoring is unfiltered)")
        assert_binomial_support(train)
        (X,), self.scaler = standardized(train, [train], self.features)
        y = train["gp"].to_numpy(int)
        n = train["team_games"].to_numpy(int)

        bins = self.bins(train)
        self.bin_counts = np.bincount(bins, minlength=self.n_rho + 1)[1:]
        # `pi`'s design is standardized on the SAME fitting rows the mean's is, and by the
        # same helper, so the two scalers describe one population. Kept as its own scaler
        # rather than folded into the feature block: `PI_FEATURES` is a strict subset of a
        # different list and its columns enter through a different link.
        Z = None
        if self.mixture:
            (Z,), self.pi_scaler = standardized(train, [train], self.pi_features)
        data = {
            "N": len(train), "K": X.shape[1], "X": X,
            "n": n.tolist(), "y": y.tolist(),
            # The identity that makes this a port: an L2 penalty of `l2` on the
            # standardized coefficients IS a normal(0, 1/sqrt(2*l2)) prior, so the
            # posterior mode here is the MLE module's optimum rather than a nearby
            # quantity that happens to resemble it.
            "beta_scale": prior_sd_for_l2(self.l2),
            "intercept_scale": INTERCEPT_SCALE,
            **self.year.data(train),
            **rho_block(len(train), bins, self.n_rho),
            **pi_block(len(train), Z, self.gamma_scale, self.mu_low_max),
        }
        share = float(np.clip(y.sum() / max(n.sum(), 1), EPS, 1 - EPS))
        inits = {"alpha": float(np.log(share / (1 - share))),
                 "beta": np.zeros(X.shape[1]).tolist(),
                 # The measured overdispersion, as a starting point — one per bin, since
                 # `rho` is a vector[n_rho] even when the vector has one entry.
                 "rho": [0.23] * self.n_rho}
        if self.mixture:
            # One init dict PER CHAIN, dispersed on `theta`. A mixture posterior can be
            # multimodal — the point MLE of this arm needed multi-start — and four chains
            # from one starting point would answer "did the chains agree with each other"
            # with "they were never given the chance to disagree".
            inits = [{**inits, "theta": [float(t)],
                      # The point MLE's fitted values: a low component at about 8 games of
                      # 82, tight around it. A start, not a prior — neither is stated.
                      "mu_low": [0.10], "rho_low": [0.05],
                      "gamma": np.zeros(len(self.pi_features)).tolist()}
                     for t in np.resize(np.asarray(THETA_INITS, dtype=float),
                                        self.chains)]

        model = compile_model(MODEL)
        fit, self.diagnostics = sample(
            model, data, chains=self.chains, warmup=self.warmup,
            samples=self.samples, seed=self.seed, label=self.name, inits=inits,
            metric=self.metric)
        warn_if_unconverged(self.diagnostics)

        draws = posterior(fit, ["alpha", "beta", "rho"])
        self.alpha_draws = draws["alpha"].reshape(-1)
        self.beta_draws = draws["beta"].reshape(len(self.alpha_draws), -1)
        # (draws x n_rho), and (draws x 1) when the dispersion is shared — so every
        # consumer indexes it the same way in both arms rather than branching.
        self.rho_draws = draws["rho"].reshape(len(self.alpha_draws), -1)
        self._absorb_mixture(fit)

        # Posterior means, in the same layout as `BetaBinomialGLM.beta` (intercept first)
        # so the two coefficient vectors can be diffed element-wise.
        self.beta = np.r_[self.alpha_draws.mean(), self.beta_draws.mean(axis=0)]
        self.rho_by_bin = dict(zip(self.rho_labels, self.rho_draws.mean(axis=0)))
        # `self.rho` stays a SCALAR because `availability.evaluate` and the season-term
        # ablation both read it as one. Row-weighted over the fitting rows, so it is the
        # average dispersion the head actually applies — and exactly `rho_draws.mean()`
        # when the dispersion is shared.
        self.rho = float(np.average(self.rho_draws.mean(axis=0), weights=self.bin_counts))
        self.year.absorb(fit)
        return self

    def _absorb_mixture(self, fit) -> None:
        """Keep the mixture block's draws, and the per-chain record of how it mixed.

        `chain_summary` is stored rather than printed because R-hat is the wrong instrument
        here: four chains each stuck in a different mode can post a respectable R-hat while
        describing four different models, and this likelihood is the one in the project
        where that is a live possibility.
        """
        n_draws = len(self.alpha_draws)
        if not self.mixture:
            self.theta_draws = np.zeros(n_draws)
            self.mu_low_draws = np.zeros(n_draws)
            self.rho_low_draws = np.full(n_draws, RHO_MIN)
            self.gamma_draws = np.zeros((n_draws, 0))
            self.chains_table = pd.DataFrame()
            return
        draws = posterior(fit, ["theta", "mu_low", "rho_low", "gamma"])
        self.theta_draws = draws["theta"].reshape(-1)
        self.mu_low_draws = draws["mu_low"].reshape(-1)
        self.rho_low_draws = draws["rho_low"].reshape(-1)
        self.gamma_draws = draws["gamma"].reshape(n_draws, -1)
        self.chains_table = chain_summary(fit, ["theta", "mu_low", "rho_low"])

    # ── Prediction ────────────────────────────────────────────────────────────

    def _design(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.features].to_numpy(dtype=float)
        return self.scaler.transform(np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0))

    def _pi_design(self, df: pd.DataFrame) -> np.ndarray:
        Z = df[self.pi_features].to_numpy(dtype=float)
        return self.pi_scaler.transform(np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0))

    def _idx(self, keep: int | None = None) -> np.ndarray:
        """The kept draw indices. One definition, because `mu_draws` and the mixture
        block below must index the SAME draws — pairing a draw's `mu` with another
        draw's `pi` would be a silent averaging of the posterior."""
        return thin(len(self.alpha_draws), keep or self.predictive_draws)

    def mu_draws(self, df: pd.DataFrame, keep: int | None = None) -> tuple[np.ndarray,
                                                                          np.ndarray]:
        """(draws x rows) mean and (draws x rows) dispersion.

        `rho` is returned already **gathered by each row's own bin**, matching `mu`'s
        shape, so a caller pairs the two element-wise and cannot silently apply one
        bucket's dispersion to another bucket's player. Under a shared dispersion every
        column of the returned matrix is the same number, which is what the scalar it
        replaced used to be.

        **This is the MAIN component, not the predictive**, when the head carries a
        mixture. `mixture_draws` supplies the rest and `predict_pmf` combines them; a
        caller that wants "the head's mean" wants `predict_mean`.
        """
        idx = self._idx(keep)
        # (rows x K) @ (K x draws) -> (rows x draws), then transposed to draws-major.
        eta = (self._design(df) @ self.beta_draws[idx].T
               + self.alpha_draws[idx][None, :] + self.year.shift(idx)[None, :])
        return _sigmoid(eta).T, self.rho_draws[np.ix_(idx, self.bins(df) - 1)]

    def mixture_draws(self, df: pd.DataFrame, keep: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """`(pi, mu_low, rho_low)` on the same draws `mu_draws` returns.

        `pi` is (draws x rows) because it carries covariates — the arm's distinguishing
        claim is that it can say *who* is at risk. `mu_low` and `rho_low` are (draws,):
        the low component is one disrupted season, not a per-player one. Without a
        mixture `pi` is exactly zero and the two scalars are inert, so every consumer
        below takes one path rather than branching.
        """
        idx = self._idx(keep)
        if not self.mixture:
            # Built rather than gathered, so a head assembled without a mixture block —
            # a hand-built fixture, an artifact from before this existed — has one path
            # through the predictive rather than an attribute it must remember to carry.
            return (np.zeros((len(idx), len(df))), np.zeros(len(idx)),
                    np.full(len(idx), RHO_MIN))
        pi = self.theta_draws[idx][:, None] * _sigmoid(self.gamma_draws[idx]
                                                       @ self._pi_design(df).T)
        return pi, self.mu_low_draws[idx], self.rho_low_draws[idx]

    def rho_row(self, df: pd.DataFrame) -> np.ndarray:
        """Posterior-mean dispersion per row — the plug-in counterpart of `mu_draws`."""
        return self.rho_draws.mean(axis=0)[self.bins(df) - 1]

    def pi_row(self, df: pd.DataFrame) -> np.ndarray:
        """`pi` at the posterior mean — the plug-in counterpart of `mixture_draws`."""
        if not self.mixture:
            return np.zeros(len(df))
        return float(self.theta_draws.mean()) * _sigmoid(
            self._pi_design(df) @ self.gamma_draws.mean(axis=0))

    def predict_mean(self, df: pd.DataFrame) -> np.ndarray:
        """The **predictive** mean share — a weighted mean when there is a mixture.

        `score_arm` and `evaluate` both turn this into MAE and R² in games, so returning
        the main component's mean would credit the head with an accuracy its own
        predictive does not have. `availability_window.MixtureFrailty.predict_mean` makes
        the same choice for the same reason, and that is what keeps the two comparable.

        Posterior mean of mu, or mu at the posterior mean, depending on the mode. The
        distinction is small but real — `E[mu]` and `mu(E[beta])` differ by Jensen's
        inequality through `inv_logit` — and keeping them separate is what lets the
        plug-in row be a like-for-like comparison against the MLE.
        """
        if self.pmf_mode == "plug_in":
            mu = _sigmoid(self.beta[0] + self._design(df) @ self.beta[1:])
            if not self.mixture:
                return mu
            pi = self.pi_row(df)
            return (1.0 - pi) * mu + pi * float(self.mu_low_draws.mean())
        mus, _ = self.mu_draws(df)
        pi, mu_low, _ = self.mixture_draws(df)
        return ((1.0 - pi) * mus + pi * mu_low[:, None]).mean(axis=0)

    def predictive_moments(self, df: pd.DataFrame, keep: int | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """(draws x rows) `E[Y | theta]` and `Var(Y | theta)` in games.

        The law-of-total-variance decomposition `board_correlation` performs needs both
        conditional moments, and under a mixture neither is the main component's: the
        variance picks up the *between-component* term `pi(1-pi)(m_low - m_main)^2`, which
        is exactly the extra spread the arm was adopted for. Computing it here rather than
        inline keeps one expression of the mixture's moments.
        """
        mus, rhos = self.mu_draws(df, keep)
        pi, mu_low, rho_low = self.mixture_draws(df, keep)
        n = df["team_games"].to_numpy(dtype=float)

        def moments(mu, rho):
            return n * mu, n * mu * (1.0 - mu) * (1.0 + (n - 1.0) * rho)

        m_main, v_main = moments(mus, rhos)
        if not self.mixture:
            return m_main, v_main
        m_low, v_low = moments(mu_low[:, None], rho_low[:, None])
        mean = (1.0 - pi) * m_main + pi * m_low
        second = (1.0 - pi) * (v_main + m_main ** 2) + pi * (v_low + m_low ** 2)
        return mean, second - mean ** 2

    def predict_samples(self, df: pd.DataFrame, seed: int = 0) -> np.ndarray:
        """(draws x rows) games played, drawn from the posterior predictive.

        The head scores through `predict_pmf` — an explicit 0..83 grid, which is affordable
        here and is not at the minutes head's 0..4,000 — so nothing in the fitting path ever
        needed a sampler. `make model-cards` does: an ECDF ribbon is one *replicate dataset*
        per posterior draw, which is a draw and not a pmf.

        It lives here rather than in the card for the reason `docs/model-cards-plan.md`
        makes load-bearing: no second implementation of any head's predictive. That mattered
        the moment `rho` became a vector — the card's generic beta-binomial branch takes one
        dispersion per draw, and applying a star's `rho` to a fringe player's mean is exactly
        the silent failure a card renders as a good-looking picture. `mu_draws` already
        gathers each row's own bucket, so drawing through it cannot make that mistake.

        Sampled as `p ~ Beta(a, b)` then `y ~ Binomial(n, p)`, the same two lines
        `StanMinutes.predict_samples` uses and orders of magnitude faster than
        `betabinom.rvs` at this shape.

        Under a mixture the component is drawn **first**, per (draw, row), and the rate
        comes from whichever one won. Drawing a rate from each and averaging would produce
        a season somewhere between healthy and disrupted, which is precisely the season the
        arm exists to say does not happen.
        """
        mus, rhos = self.mu_draws(df, self.predictive_draws)
        pi, mu_low, rho_low = self.mixture_draws(df, self.predictive_draws)
        rng = np.random.default_rng(seed)
        a, b = _shapes(mus, rhos)
        p = rng.beta(a, b)
        if self.mixture:
            a_low, b_low = _shapes(np.broadcast_to(mu_low[:, None], mus.shape),
                                   np.broadcast_to(rho_low[:, None], mus.shape))
            p = np.where(rng.random(p.shape) < pi, rng.beta(a_low, b_low), p)
        n = df["team_games"].to_numpy(int)
        return rng.binomial(n[None, :], p).astype(float)

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        n = df["team_games"].to_numpy(int)
        k = np.arange(int(max_games) + 1)
        if self.pmf_mode == "plug_in":
            mu = _sigmoid(self.beta[0] + self._design(df) @ self.beta[1:])
            if not self.mixture:
                return _plug_in_pmf(n, mu, self.rho_row(df), k)
            return _plug_in_pmf(n, mu, self.rho_row(df), k, pi=self.pi_row(df),
                                mu_low=float(self.mu_low_draws.mean()),
                                rho_low=float(self.rho_low_draws.mean()))

        # The predictive properly integrated over the posterior: a mixture of one
        # beta-binomial per draw, not one beta-binomial at the average parameter. This is
        # the whole reason for fitting in Stan, and it is strictly wider than the plug-in.
        mus, rhos = self.mu_draws(df)
        pi, mu_low, rho_low = self.mixture_draws(df)
        out = np.zeros((len(df), len(k)))
        for lo in range(0, len(mus), CHUNK):
            hi = lo + CHUNK
            a, b = _shapes(mus[lo:hi], rhos[lo:hi])
            pmf = np.nan_to_num(betabinom.pmf(k[None, None, :], n[None, :, None],
                                              a[:, :, None], b[:, :, None]))
            if self.mixture:
                # The low component varies by draw and by `n`, but not by player — one
                # disrupted season, not one per row — so its shapes carry no row axis and
                # broadcast against the schedule instead.
                a_low, b_low = _shapes(mu_low[lo:hi], rho_low[lo:hi])
                low = np.nan_to_num(betabinom.pmf(k[None, None, :], n[None, :, None],
                                                  a_low[:, None, None],
                                                  b_low[:, None, None]))
                w = pi[lo:hi][:, :, None]
                pmf = (1.0 - w) * pmf + w * low
            out += pmf.sum(axis=0)
        return out / len(mus)


def rehydrate_availability(artifact, keep: int) -> StanAvailability:
    """A `StanAvailability` carrying a persisted artifact's draws, scaler and bin arm.

    The counterpart of `minutes_unification.rehydrate_minutes`, and here for the same
    reason: a consumer that wants this head's predictive should get *this head*, not a
    re-derivation of it beside the artifact. `make model-cards` is the caller.

    **`role_rho` comes from the artifact, never from the class default.** The default is
    what ships *today*; an artifact is a record of what was fitted, and a shared-dispersion
    posterior rehydrated under a graded default would index a one-column `rho_draws` with
    bucket 2 and raise — or, worse, would not. `rho_draws` is (draws x n_rho) in both arms,
    which is what keeps the two on one path.

    **The mixture block comes from the artifact too, and it is all-or-nothing.** `pi` needs
    four draw arrays *and* its own covariate list and scaler — the recipe's second block,
    `PI_FEATURES` not being a subset of `FEATURE_COLS`. A head rehydrated with any of those
    missing would be the single-component model wearing the mixture's name and would raise
    nowhere, so a mixture artifact that carries an incomplete block raises here instead.

    `bin_counts` and the row-weighted scalar `rho` are deliberately absent: both are
    properties of the *fitting* rows, which an artifact does not carry, and a plausible-looking
    stand-in computed from the scored frame would be a different number wearing the same name.
    """
    recipe = artifact.recipe
    mixture = bool(artifact.extras.get("mixture", False))
    model = StanAvailability(features=list(recipe.features),
                             pmf_mode="posterior", name="rehydrated/availability",
                             predictive_draws=keep,
                             first_season=str(artifact.extras.get("fit_first_season") or "")
                             or None,
                             role_rho=bool(artifact.extras.get("role_rho", False)),
                             mixture=mixture,
                             pi_features=list(getattr(recipe, "pi_features", []) or [])
                             or None)
    model.scaler = recipe.scaler
    model.alpha_draws = np.asarray(artifact.draws["alpha_draws"], dtype=float)
    model.beta_draws = np.asarray(artifact.draws["beta_draws"], dtype=float)
    model.rho_draws = np.asarray(artifact.draws["rho_draws"], dtype=float)
    if not model.mixture:
        model._absorb_mixture(None)      # the inert block: `pi` is exactly zero
    else:
        missing = [name for name in ("theta_draws", "mu_low_draws", "rho_low_draws",
                                     "gamma_draws") if name not in artifact.draws]
        if missing or getattr(recipe, "pi_scaler", None) is None \
                or not getattr(recipe, "pi_features", []):
            raise ValueError(
                f"this artifact declares the low-availability mixture and its block is "
                f"incomplete — missing draws {missing or 'none'}, "
                f"{len(getattr(recipe, 'pi_features', []) or [])} pi covariate(s), scaler "
                f"{'present' if getattr(recipe, 'pi_scaler', None) is not None else 'absent'}"
                f". `pi` cannot be rebuilt for a new frame, and a head assembled from what "
                f"is here would silently be the single-component one. Re-run "
                f"`make posteriors --groups availability`.")
        model.pi_scaler = recipe.pi_scaler
        model.theta_draws = np.asarray(artifact.draws["theta_draws"], dtype=float).reshape(-1)
        model.mu_low_draws = np.asarray(artifact.draws["mu_low_draws"],
                                        dtype=float).reshape(-1)
        model.rho_low_draws = np.asarray(artifact.draws["rho_low_draws"],
                                         dtype=float).reshape(-1)
        model.gamma_draws = np.asarray(artifact.draws["gamma_draws"], dtype=float).reshape(
            len(model.alpha_draws), -1)
        model.chains_table = pd.DataFrame()
        if model.gamma_draws.shape[1] != len(model.pi_features):
            raise ValueError(
                f"the artifact carries {model.gamma_draws.shape[1]} `gamma` coefficient(s) "
                f"and {len(model.pi_features)} pi covariate(s). The persisted mixture and "
                f"its design block disagree, so `pi` would be built from the wrong columns.")
    if model.rho_draws.ndim == 1:
        model.rho_draws = model.rho_draws[:, None]
    if model.rho_draws.shape[1] != model.n_rho:
        raise ValueError(
            f"the artifact carries {model.rho_draws.shape[1]} dispersion column(s) and its "
            f"`role_rho={model.role_rho}` arm gathers {model.n_rho}. The persisted head and "
            f"the bin assignment disagree; re-run `make posteriors --groups availability` "
            f"rather than reading one bucket's dispersion onto another bucket's players.")
    return model


def _shapes(mu: np.ndarray, rho: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = np.clip(mu, EPS, 1 - EPS)
    scale = (1.0 - np.clip(rho, RHO_MIN, RHO_MAX)) / np.clip(rho, RHO_MIN, RHO_MAX)
    return mu * scale, (1.0 - mu) * scale


def _plug_in_pmf(n: np.ndarray, mu: np.ndarray, rho: np.ndarray | float,
                 k: np.ndarray, pi: np.ndarray | float = 0.0,
                 mu_low: float = 0.0, rho_low: float = RHO_MIN) -> np.ndarray:
    """`rho` is per row, so the plug-in view carries the grading the posterior one does.

    And `pi` is per row, so it carries the mixture too. `pi = 0` returns the single
    beta-binomial exactly, which is what the head does when the mixture is off.
    """
    a, b = _shapes(np.asarray(mu, dtype=float), np.asarray(rho, dtype=float))
    pmf = np.nan_to_num(betabinom.pmf(k[None, :], n[:, None], a[:, None], b[:, None]))
    pi = np.asarray(pi, dtype=float)
    if not np.any(pi):
        return pmf
    a_low, b_low = _shapes(np.full(len(n), mu_low), np.full(len(n), rho_low))
    low = np.nan_to_num(betabinom.pmf(k[None, :], n[:, None],
                                      a_low[:, None], b_low[:, None]))
    w = np.broadcast_to(pi, (len(n),))[:, None]
    return (1.0 - w) * pmf + w * low


# ── The port check ────────────────────────────────────────────────────────────

def coefficient_comparison(mle: BetaBinomialGLM, stan: StanAvailability,
                           features: list[str],
                           rho_mle: dict[str, float] | None = None) -> pd.DataFrame:
    """MLE optimum against the posterior, coefficient by coefficient.

    Both are fitted on the identical standardized design — the **windowed** rows, or the
    comparison would be between two different models rather than two fits of one — so the
    vectors are directly comparable entry for entry. `z_from_mle` — how many posterior
    standard deviations the MLE sits from the posterior mean — is the column that matters:
    agreement means the port is faithful, and a large value on any single coefficient
    localizes a discrepancy that an aggregate norm would average away.

    The dispersion contributes one row **per bin**. `rho_mle` supplies the point-MLE value
    per bin label — `availability_window.RoleGradedBetaBinomial.rho_by_role` — and falls
    back to the shared scalar, which is the right reference when `n_rho = 1`. The
    agreement is looser here than for the coefficients and expected to be: that estimate
    re-fits each bucket's dispersion holding the mean fixed, a two-stage profile, where
    Stan samples the whole vector jointly with `beta`.
    """
    names = ["intercept"] + list(features)
    posterior_sd = np.r_[stan.alpha_draws.std(ddof=1),
                         stan.beta_draws.std(axis=0, ddof=1)]
    lo = np.r_[np.percentile(stan.alpha_draws, 2.5),
               np.percentile(stan.beta_draws, 2.5, axis=0)]
    hi = np.r_[np.percentile(stan.alpha_draws, 97.5),
               np.percentile(stan.beta_draws, 97.5, axis=0)]
    out = pd.DataFrame({
        "term": names, "mle": mle.beta, "posterior_mean": stan.beta,
        "posterior_sd": posterior_sd, "q2_5": lo, "q97_5": hi,
    })
    out["difference"] = out["posterior_mean"] - out["mle"]
    out["z_from_mle"] = out["difference"] / out["posterior_sd"].replace(0, np.nan)
    out["mle_inside_95"] = (out["mle"] >= out["q2_5"]) & (out["mle"] <= out["q97_5"])
    rho_mle = rho_mle or {}
    rho = pd.DataFrame([{
        "term": f"rho[{label}]",
        "mle": float(rho_mle.get(label, mle.rho)),
        "posterior_mean": float(stan.rho_draws[:, j].mean()),
        "posterior_sd": float(stan.rho_draws[:, j].std(ddof=1)),
        "q2_5": float(np.percentile(stan.rho_draws[:, j], 2.5)),
        "q97_5": float(np.percentile(stan.rho_draws[:, j], 97.5)),
        "n_fit_rows": int(stan.bin_counts[j]),
    } for j, label in enumerate(stan.rho_labels)])
    rho["difference"] = rho["posterior_mean"] - rho["mle"]
    rho["z_from_mle"] = rho["difference"] / rho["posterior_sd"]
    rho["mle_inside_95"] = (rho["mle"] >= rho["q2_5"]) & (rho["mle"] <= rho["q97_5"])
    return pd.concat([out, rho], ignore_index=True)


PORTFOLIO_SIZES = (12, 15, 30, 150, None)      # None = the whole held-out board


def board_correlation(stan: StanAvailability, frame: pd.DataFrame,
                      sizes: tuple = PORTFOLIO_SIZES, n_subsets: int = 200,
                      seed: int = 0) -> pd.DataFrame:
    r"""The thing the posterior buys that a point estimate cannot: whole-board covariance.

    Every player's mean is a function of one shared `beta`, so a posterior draw moves the
    entire board **together**. That is a genuine, correctly calibrated source of
    cross-player correlation which arrives free with the fit, rather than an invented
    copula — and for a draft portfolio "how wrong could my whole board be at once" is a
    different and more important question than "how wrong is this one player".

    Decomposed exactly by the law of total variance on the board total `T = sum_i Y_i`:

        Var(T) = E_th[ sum_i Var(Y_i | th) ]  +  Var_th( sum_i E[Y_i | th] )
                 \_____ independent _____/       \_____ shared beta _____/

    The first term is all a plug-in model has: with parameters fixed, what is left is
    independent across players and grows as **sqrt(N)**. The second is exactly zero under a
    point estimate and grows as **N**, because it is perfectly correlated across players.

    **So the size of the portfolio decides whether this matters at all, and it is reported
    across sizes rather than as one number.** The ratio of the two terms scales as sqrt(N):
    measured here it is a fraction of a percent on a 15-player roster and several percent
    across the whole board. Quoting only the board figure would badly oversell what the
    posterior does for a single draft roster; quoting only the roster figure would miss
    that it is the dominant term for board-wide exposure across many lineups.

    Subsets are drawn at random and averaged, so `inflation` is measured on real players
    rather than extrapolated from the full-board number under an equal-variance assumption.

    **`frame` is the validation board, not the held-out one.** This is a simulator input —
    "how much does my whole board move together" is a number the simulator is *given* — so
    measuring it on the seasons the simulator is later backtested against would be leakage
    of the kind the split cannot catch, exactly as for `stan_minutes.game_level_dispersion`
    and the residual copula. The board size therefore tracks the validation seasons' player
    count rather than the held-out one's.
    """
    # Through the head's own moments, so a mixture contributes its between-component
    # variance rather than being read as its main component. `pi = 0` recovers the single
    # beta-binomial's `n mu (1-mu) [1 + (n-1) rho]` exactly.
    means, conditional = stan.predictive_moments(frame)   # per draw per player

    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size is None or size >= len(frame):
            size, subsets = len(frame), [np.arange(len(frame))]
        else:
            subsets = [rng.choice(len(frame), size, replace=False)
                       for _ in range(n_subsets)]
        independent = np.array([np.sqrt(conditional[:, s].sum(axis=1).mean())
                                for s in subsets])
        shared = np.array([means[:, s].sum(axis=1).std(ddof=1) for s in subsets])
        total = np.hypot(independent, shared)
        rows.append({
            "n_players": size,
            "n_subsets": len(subsets),
            "expected_total_games": float(np.mean([means[:, s].sum(axis=1).mean()
                                                   for s in subsets])),
            "independent_sd": float(independent.mean()),
            "shared_beta_sd": float(shared.mean()),
            "total_sd": float(total.mean()),
            "inflation": float((total / independent).mean()),
        })
    return pd.DataFrame(rows)


# ── The mixture's own port check ──────────────────────────────────────────────
#
# The single-component port is checked coefficient by coefficient against the point MLE,
# because the prior makes the posterior MODE exactly the penalized MLE and agreement is a
# check with a defined answer. The mixture cannot be checked that way and it would be a
# mistake to pretend otherwise: `theta`, `mu_low` and `rho_low` are BOUNDED parameters, the
# ladder fits them inside a box with no penalty, `gamma` carries a prior here and none
# there, and a bounded posterior mean is not its mode. So the check moves up one level —
# to the *predictive*, scored by `availability_window.score_arm`, which is the same code
# that produced the row this is a port of.
#
# Three arms, and each answers a different question:
#
#   `mixture_point_mle`    — the ladder's own arm, refitted here on the same rows. Should
#                            reproduce `availability_likelihood.csv`'s `mixture` row to
#                            four decimals; if it does not, the two frames differ and
#                            nothing below is comparable.
#   `stan_mixture_plug_in` — posterior means substituted for the optimum. The row that
#                            should land NEAR the point MLE.
#   `stan_mixture_posterior` — the predictive integrated over the posterior. The row a
#                            consumer gets, and the one that may legitimately differ.

MIXTURE_METRICS = ("val_crps", "val_pit_ks", "boundary_tail_error", "body_error",
                   "shoulder_error", "point_mass_error")


def mixture_parameters(mle, stan: StanAvailability) -> pd.DataFrame:
    """The mixture block, point MLE against posterior — `theta`, `mu_low`, `rho_low`, `pi`.

    `mle` is a fitted `availability_window.MixtureFrailty`, whose `extra` vector is
    `[theta, logit mu_low, logit rho_low, gamma...]`. Reported term by term rather than as
    one distance, because these four are on different scales and a single norm would be
    dominated by whichever happened to be largest.
    """
    from src.models.availability_window import MU_LOW_MAX

    rows = [{"term": "theta", "mle": float(np.clip(mle.extra[0], 0.0, 1.0)),
             "draws": stan.theta_draws},
            {"term": "mu_low",
             "mle": float(np.clip(_sigmoid(mle.extra[1]), EPS, MU_LOW_MAX)),
             "draws": stan.mu_low_draws},
            {"term": "rho_low", "mle": float(mle._disp(mle.extra[2])),
             "draws": stan.rho_low_draws}]
    for j, name in enumerate(stan.pi_features):
        rows.append({"term": f"gamma[{name}]", "mle": float(mle.extra[3 + j]),
                     "draws": stan.gamma_draws[:, j]})
    out = []
    for row in rows:
        draws = np.asarray(row["draws"], dtype=float)
        out.append({
            "term": row["term"], "mle": row["mle"],
            "posterior_mean": float(draws.mean()),
            "posterior_sd": float(draws.std(ddof=1)),
            "q2_5": float(np.percentile(draws, 2.5)),
            "q97_5": float(np.percentile(draws, 97.5))})
    frame = pd.DataFrame(out)
    frame["difference"] = frame["posterior_mean"] - frame["mle"]
    frame["z_from_mle"] = frame["difference"] / frame["posterior_sd"].replace(0, np.nan)
    frame["mle_inside_95"] = ((frame["mle"] >= frame["q2_5"])
                              & (frame["mle"] <= frame["q97_5"]))
    return frame


def pi_profile(model, frame: pd.DataFrame) -> dict:
    """What `pi` says, on whichever frame it is handed.

    The arm's distinguishing claim is that it can name *who* is at risk — a flat `pi` would
    make it a two-component mixture with a constant weight, which
    `docs/availability-window-plan.md` §7c measured as worth nothing. So the spread across
    players is reported, not only the mean.
    """
    pi = (model.pi_row(frame) if isinstance(model, StanAvailability)
          else model._parts(model.extra, frame)[0])
    return {"pi_mean": float(np.mean(pi)), "pi_sd": float(np.std(pi)),
            "pi_p10": float(np.percentile(pi, 10)),
            "pi_p90": float(np.percentile(pi, 90)),
            "pi_spread": float(np.percentile(pi, 90) / max(np.percentile(pi, 10), 1e-12))}


def mixture_port_check(cfg: dict, train: pd.DataFrame, val: pd.DataFrame,
                       max_games: int, l2: float, seed: int,
                       first_season: str | None = FIRST_SEASON,
                       role_rho: bool = ROLE_RHO) -> dict:
    """Fit the ladder's `mixture` arm and its Stan port on the same rows, score both.

    Scored through `availability_window.score_arm` — the ladder's own scorer, imported
    rather than reimplemented — so a difference between this table and
    `availability_likelihood.csv` cannot be a difference between two implementations of
    CRPS or of `boundary_tail_error`.
    """
    from src.models.availability_window import (LIKELIHOODS, MixtureFrailty, _tail_parts,
                                                assert_nests, bootstrap_tail_errors,
                                                paired_bootstrap, score_arm)

    stan = StanAvailability(
        l2=l2, pmf_mode="posterior", name="stan_mixture_posterior",
        chains=int(cfg.get("chains", 4)), warmup=int(cfg.get("warmup", 1000)),
        samples=int(cfg.get("samples", 1000)), seed=seed,
        predictive_draws=int(cfg.get("predictive_draws", PREDICTIVE_DRAWS)),
        first_season=first_season, role_rho=role_rho, mixture=True,
        gamma_scale=float(cfg.get("availability", {}).get("pi_gamma_scale", GAMMA_SCALE)),
        mu_low_max=float(cfg.get("availability", {}).get("mu_low_max", MU_LOW_MAX)))
    fit_rows = stan.fitting_rows(train)

    print(f"\nFitting the point-MLE arms on the same {len(fit_rows):,} windowed rows "
          f"({', '.join(sorted(pd.Series(fit_rows['season']).unique()))[:20]}...):")
    reference = LIKELIHOODS["betabinom"](l2=l2, features=list(FEATURE_COLS)).fit(fit_rows)
    assert_nests(reference, fit_rows)
    mle = MixtureFrailty(l2=l2, features=list(FEATURE_COLS)).fit(fit_rows)
    gap = assert_nests(mle, fit_rows)
    print(f"  `mixture` nests `betabinom` at pi = 0 to {gap:.3e} log-likelihood, from "
          f"{mle.n_starts} starts (spread {mle.start_spread:.3f})")
    print(f"  theta {float(np.clip(mle.extra[0], 0, 1)):.4f}, "
          f"mu_low {float(_sigmoid(mle.extra[1])):.4f}, "
          f"rho_low {float(mle._disp(mle.extra[2])):.4f}, "
          f"train log-likelihood {mle.train_loglik:,.1f} against the reference's "
          f"{mle.incumbent_loglik:,.1f}")

    print(f"\nFitting the mixture in Stan ({stan.chains} chains x {stan.samples} draws)...")
    stan.fit(train)
    d = stan.diagnostics
    print(f"  max R-hat {d['max_rhat']:.4f}, min ESS "
          f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}, "
          f"{d['divergences']} divergences, {d['wall_clock_s']:.1f}s wall clock")

    plug_in = StanAvailability(l2=l2, pmf_mode="plug_in", predictive_draws=1)
    plug_in.__dict__.update({k: v for k, v in stan.__dict__.items()
                             if k not in ("pmf_mode", "name")})
    plug_in.pmf_mode, plug_in.name = "plug_in", "stan_mixture_plug_in"

    rows, per_row, tails = [], {}, {}
    y_val, n_val = val["gp"].to_numpy(), val["team_games"].to_numpy()
    for name, model, source in (("betabinom_point_mle", reference, "point_mle"),
                                ("mixture_point_mle", mle, "point_mle"),
                                ("stan_mixture_plug_in", plug_in, "stan"),
                                ("stan_mixture_posterior", stan, "stan")):
        row, scores = score_arm(name, model, fit_rows, val, list(FEATURE_COLS),
                                max_games, seed)
        row["source"] = source
        if name != "betabinom_point_mle":
            row.update(pi_profile(model, val))
        rows.append(row)
        per_row[name] = scores
        tails[name] = _tail_parts(model.predict_pmf(val, max_games), y_val, n_val)

    # D1 evaluated on the object that SHIPS, not on the ladder row it is a port of. The
    # rule is "improves the calibration metrics AND is CRPS non-inferior", and both halves
    # are interval statements — a boundary margin quoted bare is what this project calls a
    # prompt rather than a finding. Paired against the same single-component reference the
    # ladder used, so the two tables are read the same way.
    ref = "betabinom_point_mle"
    for row in rows:
        d, lo, hi = paired_bootstrap(per_row[row["arm"]], per_row[ref], seed=seed)
        row.update({"crps_vs_betabinom": d, "crps_vs_betabinom_lo": lo,
                    "crps_vs_betabinom_hi": hi})
        row.update(bootstrap_tail_errors(tails[row["arm"]],
                                         None if row["arm"] == ref else tails[ref],
                                         seed=seed))
    return {"stan": stan, "plug_in": plug_in, "mle": mle, "reference": reference,
            "fit_rows": fit_rows, "scores": pd.DataFrame(rows),
            "parameters": mixture_parameters(mle, stan),
            "chains": stan.chains_table,
            "diagnostics": diagnostics_frame([stan.diagnostics])}


def ladder_row(path: Path, arm: str) -> pd.Series | None:
    """The recorded `availability_likelihood.csv` row this port is a port of.

    Read from disk rather than refitted, because the point of the comparison is that the
    number in the artifact — the one `docs/availability-window-plan.md` §7c quotes and D1
    was taken on — is the number this reproduces.
    """
    if not path.exists():
        print(f"  no {path.name} on disk — run `make availability-window` for the "
              f"recorded ladder row")
        return None
    table = pd.read_csv(path)
    hit = table[table["arm"] == arm]
    return None if hit.empty else hit.iloc[0]


# ── The measurement, on whichever pair of frames it is handed ─────────────────

def fit_and_score(train: pd.DataFrame, frame: pd.DataFrame, max_games: int,
                  cfg_stan: dict, l2: float, seed: int,
                  first_season: str | None = FIRST_SEASON,
                  role_rho: bool = ROLE_RHO, mixture: bool = MIXTURE,
                  features: list[str] | None = None) -> dict:
    """Fit the point MLEs and the Stan head on `train`, score all four on `frame`.

    Split-agnostic on purpose. `run` hands it `(train, validation)`; when the workflow is
    finished `src/final_evaluation.py` hands it `(train + validation, test)`. The held-out
    number is therefore produced by *this* code rather than by a second implementation of
    it that could drift — the same reason the port check imports `evaluate` and `crps` from
    the MLE module instead of reimplementing them.

    **Both point MLEs are fitted on the same windowed rows the Stan head fits**, obtained
    from the head's own `fitting_rows` rather than re-derived here. A port check against a
    reference fitted on a different population is not a port check; it is a comparison of
    two models, and it would read as a failure of the port when it was a difference in the
    data. Scoring is on the whole of `frame` for all four.

    Two references rather than one, because the shipped head changed two things at once:

    - `beta_binomial` — one shared dispersion, the incumbent likelihood on the new window.
      Reproduces the ladder's `three_point_era__none__shared`.
    - `beta_binomial_role_rho` — the dispersion graded by prior-MPG bucket, which is the
      arm `docs/availability-window-plan.md` §4 actually selected
      (`three_point_era__none__role`). This is what the Stan head is a port *of*, so the
      two should agree, and the ladder's own CRPS is a third-party check on both.

    A third arrives with the mixture, for the same reason: `mixture_mle` is
    `availability_window.MixtureFrailty` on the same rows, which is the arm §7c selected
    and the one the Stan head is then a port of. Scoring a mixture posterior only against
    single-component point MLEs would read a *likelihood* change as a port discrepancy.
    """
    stan = StanAvailability(
        l2=l2, pmf_mode="posterior", chains=int(cfg_stan.get("chains", 4)),
        warmup=int(cfg_stan.get("warmup", 1000)),
        samples=int(cfg_stan.get("samples", 1000)), seed=seed,
        predictive_draws=int(cfg_stan.get("predictive_draws", PREDICTIVE_DRAWS)),
        first_season=first_season, role_rho=role_rho, mixture=mixture,
        gamma_scale=float(cfg_stan.get("availability", {})
                          .get("pi_gamma_scale", GAMMA_SCALE)),
        mu_low_max=float(cfg_stan.get("availability", {})
                         .get("mu_low_max", MU_LOW_MAX)),
        features=list(features) if features else None)
    fit_rows = stan.fitting_rows(train)

    print(f"\nFitting the point MLEs on the same {len(fit_rows):,} windowed rows "
          f"(the reference this ports)...")
    # NOTE both single-component references keep `FEATURE_COLS`. They are context rows;
    # the arm this head is a port OF is `mixture_mle`, which takes `stan.features` below and
    # therefore carries the same preseason block the posterior does. A port check whose
    # reference had a different design matrix would read a feature change as a port
    # discrepancy — the argument this module already makes about the fitting window.
    mle = BetaBinomialGLM(l2).fit(fit_rows)
    print(f"  shared rho: converged={mle.converged}, rho={mle.rho:.4f}")
    role_mle = None
    if role_rho:
        # Imported here rather than at module scope: `availability_window` imports
        # `season_terms`, which imports this module, so a top-level import is a cycle.
        from src.models.availability_window import RoleGradedBetaBinomial
        role_mle = RoleGradedBetaBinomial(l2).fit(fit_rows)
        print("  role-graded rho: "
              + ", ".join(f"{k} {v:.4f}" for k, v in role_mle.rho_by_role.items())
              + f" ({role_mle.rho_spread:.2f}x spread)")
    mixture_mle = None
    if mixture:
        from src.models.availability_window import MixtureFrailty, assert_nests
        mixture_mle = MixtureFrailty(l2, features=list(stan.features)).fit(fit_rows)
        mixture_mle.name = "mixture_mle"
        gap = assert_nests(mixture_mle, fit_rows)
        print(f"  mixture: theta {float(np.clip(mixture_mle.extra[0], 0, 1)):.4f}, "
              f"mu_low {float(_sigmoid(mixture_mle.extra[1])):.4f}, "
              f"rho_low {float(mixture_mle._disp(mixture_mle.extra[2])):.4f}, "
              f"nests at pi = 0 to {gap:.3e}")

    print(f"\nFitting in Stan ({cfg_stan.get('chains', 4)} chains x "
          f"{cfg_stan.get('samples', 1000)} draws)...")
    stan.fit(train)
    d = stan.diagnostics
    print(f"  max R-hat {d['max_rhat']:.4f}, min ESS "
          f"{min(d['min_ess_bulk'], d['min_ess_tail']):.0f}, "
          f"{d['divergences']} divergences, {d['wall_clock_s']:.1f}s wall clock "
          f"({d['cmdstan']})")
    print("  posterior mean rho by bin: "
          + ", ".join(f"{k} {v:.4f}" for k, v in stan.rho_by_bin.items()))

    # The plug-in view shares the fit; only the predictive differs.
    plug_in = StanAvailability(l2=l2, pmf_mode="plug_in", predictive_draws=1)
    plug_in.__dict__.update({k: v for k, v in stan.__dict__.items()
                             if k not in ("pmf_mode", "name")})

    rows, pit_frames, prediction_frames = [], [], []
    for model in [m for m in (mle, role_mle, mixture_mle, plug_in, stan) if m is not None]:
        model_rows, predictions = evaluate(model, frame, max_games, seed)
        rows += model_rows
        prediction_frames.append(predictions)
        pit_frames.append(pit_table(predictions["pit"].to_numpy(), model.name))

    # **The port check's reference is the arm the head is a port OF.** Under a mixture that
    # is `mixture_mle`, not `mle`: the two fit different likelihoods, so the main
    # component's `beta` and `rho` are genuinely different parameters and comparing them
    # would read a likelihood change as a port discrepancy — the same argument this module
    # already makes for fitting both on the same windowed rows, one axis over. Measured:
    # against `mle` the shipped mixture posterior puts the MLE inside 19 of 24 intervals
    # and `rho[30+ mpg]` at z = -6.34, which says the mixture pulled the main component's
    # dispersion down (0.2261 against 0.2627) and nothing at all about the port.
    reference = mixture_mle or mle
    rho_reference = (dict(zip(ROLE_LABELS, np.asarray(mixture_mle.dispersion).reshape(-1)))
                     if mixture_mle is not None
                     else getattr(role_mle, "rho_by_role", None))
    coefficients = coefficient_comparison(reference, stan, stan.features,
                                          rho_mle=rho_reference)
    if mixture_mle is not None:
        # The eleven mixture terms, in the same columns, so one table is the whole check.
        coefficients = pd.concat([coefficients,
                                  mixture_parameters(mixture_mle, stan)],
                                 ignore_index=True)
    return {
        "mle": mle, "role_mle": role_mle, "mixture_mle": mixture_mle,
        "reference": reference,
        "stan": stan, "plug_in": plug_in, "fit_rows": fit_rows,
        "metrics": pd.DataFrame(rows),
        "coefficients": coefficients,
        "board": board_correlation(stan, frame),
        "pit": pd.concat(pit_frames, ignore_index=True),
        "predictions": pd.concat(prediction_frames, ignore_index=True),
        "diagnostics": diagnostics_frame([stan.diagnostics]),
    }


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    cfg_stan = cfg.get("stan", {})
    cfg_head = cfg_stan.get("availability", {})
    test_seasons = int(cfg_av.get("test_seasons", 2))
    seed = int(cfg_stan.get("seed", cfg_av.get("seed", 42)))
    l2 = float(cfg_av.get("glm_l2", 1.0))
    first_season = cfg_head.get("first_season", FIRST_SEASON)
    role_rho = bool(cfg_head.get("role_rho", ROLE_RHO))
    mixture = bool(cfg_head.get("mixture", MIXTURE))
    preseason = bool(cfg_head.get("preseason", PRESEASON))
    features = head_features(preseason)

    design = head_design(cfg, preseason)
    train, val = selection_split(design, test_seasons)
    max_games = int(design["team_games"].max())

    print(f"Stan availability: {len(design):,} player-seasons. The test split is LOCKED — "
          f"this port\n  check fits and scores VALIDATION only "
          f"(src/models/held_out.py); the held-out reading is\n  taken once, by "
          f"`make final-evaluation`, through this module's own `fit_and_score`.")
    print(f"  {len(train):,} fit / {len(val):,} score "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    print(f"  {len(features)} features, n = max(team_games, gp) — the 13 traded "
          f"player-seasons with gp > team_games would otherwise make the summed\n"
          f"  log-likelihood non-finite at every rho, which under HMC poisons the "
          f"trajectory rather than just stopping an optimizer.")
    print(f"  Shipped configuration: fitting window {first_season or 'full'}+, "
          f"dispersion {'graded by prior-MPG role' if role_rho else 'shared'}, "
          f"likelihood {'2-component mixture' if mixture else 'beta-binomial'}, "
          f"preseason block {'ON' if preseason else 'off'} "
          f"— docs/availability-window-plan.md §4 and §7, docs/preseason-plan.md P2.\n"
          f"  The window cuts the FITTING rows only; `availability_design` is untouched, "
          f"because six other\n  modules import it and would be silently re-scoped — and "
          f"the preseason block reaches this head\n  through `head_design`, which is this "
          f"head's own path for exactly the same reason.")

    scored = fit_and_score(train, val, max_games, cfg_stan, l2, seed,
                           first_season=first_season, role_rho=role_rho,
                           mixture=mixture, features=features)
    mle, stan, plug_in = scored["mle"], scored["stan"], scored["plug_in"]

    metrics = scored["metrics"]
    table = (metrics[metrics.group == "all"]
             .pivot_table(index="model", columns="metric", values="value"))
    order = ["crps_games", "mae_games", "r2_gp_share", "pit_ks_distance",
             "dispersion_rho", "implied_overdispersion"]
    print("\nValidation scores (CRPS in games, lower is better):")
    print(table[order].sort_values("crps_games").round(4).to_string())

    coefs = scored["coefficients"]
    worst = coefs.loc[coefs["z_from_mle"].abs().idxmax()]
    print(f"\nPort check — `{scored['reference'].name}`'s optimum against the posterior "
          f"({len(coefs)} terms),\n  both fitted on the same "
          f"{len(scored['fit_rows']):,} windowed rows. The reference is the arm this head "
          f"is a\n  port OF, so a likelihood change is not read as a port discrepancy:")
    print(f"  max |posterior mean - MLE| = "
          f"{coefs['difference'].abs().max():.5f}")
    print(f"  largest gap in posterior sds: {worst['term']} at "
          f"z = {worst['z_from_mle']:+.3f}")
    print(f"  MLE inside the 95% credible interval for "
          f"{int(coefs['mle_inside_95'].sum())}/{len(coefs)} terms")
    print("  The prior is normal(0, 1/sqrt(2*l2)), so the posterior MODE is exactly the\n"
          "  penalized MLE — agreement here is a defined check, not a coincidence.")
    if role_rho:
        print("  Under a GRADED rho that identity holds for the model but not for this\n"
              "  reference: the point estimate profiles each bucket's dispersion against a\n"
              "  fixed mean rather than optimizing jointly, so the two agree closely rather\n"
              "  than exactly, and they differ most on the intercept. The exact nesting is\n"
              "  pinned where it can be — `n_rho = 1` against the shared-rho target, in\n"
              "  tests/test_stan_heads.py.")

    dispersion = coefs[coefs["term"].str.startswith("rho[")]
    print("\n  The dispersion, bin by bin (point MLE against the posterior):")
    print(dispersion[["term", "n_fit_rows", "mle", "posterior_mean", "posterior_sd",
                      "z_from_mle"]].round(4).to_string(index=False))
    print("  Looser agreement than the coefficients is expected: the point estimate "
          "re-fits each\n  bucket's rho holding the mean fixed, where Stan samples the "
          "whole vector jointly.")

    mixed = coefs[coefs["term"].isin(["theta", "mu_low", "rho_low"])
                  | coefs["term"].str.startswith("gamma[")]
    if len(mixed):
        print(f"\n  The mixture block ({len(mixed)} terms), point MLE against the "
              f"posterior:")
        print(mixed[["term", "mle", "posterior_mean", "posterior_sd",
                     "z_from_mle", "mle_inside_95"]].round(4).to_string(index=False))
        print("  Weaker evidence than the coefficient block's, and the difference "
              "matters: the\n  normal(0, 1/sqrt(2*l2)) prior makes the posterior mode "
              "exactly the penalized MLE for\n  `beta`, where `theta`, `mu_low` and "
              "`rho_low` are bounded and unpenalized in the ladder\n  and `gamma` carries "
              "a prior here and none there. The predictive table is the check.")
        print("  pi on the scored rows: "
              + ", ".join(f"{k} {v:.4f}" for k, v in pi_profile(stan, val).items()))

    crps_mle = float(table.loc[mle.name, "crps_games"])
    crps_plug = float(table.loc[plug_in.name, "crps_games"])
    crps_post = float(table.loc[stan.name, "crps_games"])
    print(f"\n  CRPS: MLE {crps_mle:.4f} | Stan plug-in {crps_plug:.4f} "
          f"({crps_plug - crps_mle:+.4f}) | Stan posterior {crps_post:.4f} "
          f"({crps_post - crps_mle:+.4f})")
    print("  The marginal metric was never the argument — at ~10,000 rows against 20\n"
          "  parameters the posterior is sharp, so this is expected to be a wash.")
    role_mle = scored["role_mle"]
    if role_mle is not None:
        crps_role = float(table.loc[role_mle.name, "crps_games"])
        print(f"  Against the ladder: the shared-rho MLE on this window scores "
              f"{crps_mle:.4f} where\n  `make availability-window` read 9.8444, and the "
              f"role-graded one {crps_role:.4f} against 9.8247.\n  Two independent "
              f"reproductions of the arm this head ports.")

    board = scored["board"]
    print("\nWhat the posterior actually buys — shared-beta correlation, by portfolio size:")
    print(board.round(3).to_string(index=False))
    small = board.iloc[0]
    whole = board.iloc[-1]
    print(f"  The shared-beta term is EXACTLY ZERO under any point estimate. It grows as N "
          f"while the\n  independent term grows as sqrt(N), so the ratio scales as sqrt(N) "
          f"and the SIZE OF THE\n  PORTFOLIO decides whether it matters: assuming "
          f"independent marginals understates the\n  spread by "
          f"{small['inflation'] - 1:.1%} on {int(small['n_players'])} players and "
          f"{whole['inflation'] - 1:.1%} across all "
          f"{int(whole['n_players'])}.")
    print("  So this is real for board-wide exposure and near-irrelevant for one roster — "
          "do not\n  quote the board figure as if it applied to a 15-man team.")
    print("  Note it is a statement about the JOINT, not the marginal: per player the\n"
          "  integrated predictive is not necessarily wider, because Jensen on the concave\n"
          "  conditional variance pushes back against the parameter spread.")

    print("\nPIT calibration (share per decile; 0.100 is uniform):")
    pit = scored["pit"]
    print(pit.pivot_table(index="model", columns="bin_low", values="share")
          .round(3).to_string())

    artifacts = {
        "metrics": (metrics, out_dir / "stan_availability_metrics.csv"),
        "coefficients": (coefs, out_dir / "stan_availability_coefficients.csv"),
        "diagnostics": (scored["diagnostics"],
                        out_dir / "stan_availability_diagnostics.csv"),
        "board": (board, out_dir / "stan_availability_board.csv"),
        "pit": (pit, out_dir / "stan_availability_pit.csv"),
        "predictions": (scored["predictions"],
                        out_dir / "stan_availability_predictions.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    return paths


def run_mixture_check(cfg: dict) -> dict[str, Path]:
    """`make stan-availability-mixture` — the port check for the low-availability mixture.

    A separate target from `make stan-availability`, and deliberately so. That one writes
    the head's shipped metrics, which every quoted port figure in
    `docs/availability-plan.md` and `docs/facts-archive.md` is audited against; this one
    answers a different question —
    does the Stan mixture reproduce the point-MLE arm `docs/availability-window-plan.md` §7c
    selected — and writes its own artifacts, so a port check cannot silently move a
    published headline.
    """
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    cfg_stan = cfg.get("stan", {})
    cfg_head = cfg_stan.get("availability", {})
    seed = int(cfg_stan.get("seed", cfg_av.get("seed", 42)))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    # `availability_design`, NOT `head_design`, and deliberately: this check scores the Stan
    # mixture against a RECORDED point-MLE ladder row (`availability_likelihood.csv`'s
    # `mixture`), which was fitted without the preseason block. Carrying the block here
    # would compare two different design matrices and read a feature change as a port
    # discrepancy — the same argument this module makes about the fitting window. The
    # preseason block's own port evidence is `run`'s table, where the reference is a mixture
    # MLE on the identical columns.
    design = availability_design(cfg)
    train, val = selection_split(design, int(cfg_av.get("test_seasons", 2)))
    max_games = int(design["team_games"].max())

    print(f"Availability mixture port check: {len(train):,} offered / {len(val):,} scored "
          f"({', '.join(sorted(val['season'].unique()))} as validation).")
    print("  The held-out split is LOCKED and never materialized here. The window cuts the "
          "FITTING\n  rows only, exactly as `make availability-window` cut them, or the "
          "two tables would\n  describe different populations rather than different fits.")
    missing = [c for c in PI_FEATURES if c not in design.columns]
    if missing:
        raise KeyError(f"pi's covariate block is missing {missing} from the design")
    print(f"  pi's block: {len(PI_FEATURES)} columns, "
          f"{int(design[PI_FEATURES].isna().sum().sum())} missing values")

    scored = mixture_port_check(
        cfg_stan, train, val, max_games, l2, seed,
        first_season=cfg_head.get("first_season", FIRST_SEASON),
        role_rho=bool(cfg_head.get("role_rho", ROLE_RHO)))
    table = scored["scores"]

    recorded = ladder_row(out_dir / "availability_likelihood.csv", "mixture")
    recorded_ref = ladder_row(out_dir / "availability_likelihood.csv", "betabinom")
    if recorded is not None:
        for arm, row in (("mixture", recorded), ("betabinom", recorded_ref)):
            if row is None:
                continue
            table = pd.concat([table, pd.DataFrame([{
                "arm": f"ladder_{arm}", "source": "availability_likelihood.csv",
                **{m: float(row[m]) for m in MIXTURE_METRICS if m in row}}])],
                ignore_index=True)

    print("\nThe port, against the row it is a port of "
          "(`availability_likelihood.csv`'s `mixture`):")
    print(table[["arm", "source", *MIXTURE_METRICS]].round(4).to_string(index=False))

    print("\nD1's rule, applied to what SHIPS — calibration wins with a CRPS "
          "non-inferiority guard,\n  every margin against the single-component reference "
          "on the same 883 rows:")
    guard = table[table["crps_vs_betabinom"].notna()]
    print(guard[["arm", "crps_vs_betabinom", "crps_vs_betabinom_lo",
                 "crps_vs_betabinom_hi", "boundary_vs_betabinom",
                 "boundary_vs_betabinom_lo", "boundary_vs_betabinom_hi",
                 "shoulder_vs_betabinom", "shoulder_vs_betabinom_lo",
                 "shoulder_vs_betabinom_hi"]].round(4).to_string(index=False))

    if recorded is not None:
        got = table.loc[table["arm"] == "mixture_point_mle"].iloc[0]
        gaps = {m: abs(float(got[m]) - float(recorded[m])) for m in MIXTURE_METRICS
                if m in recorded}
        worst = max(gaps, key=gaps.get)
        print(f"\n  The point MLE refitted here against the recorded ladder row: largest "
              f"gap {gaps[worst]:.6f} on {worst}.")
        print("  Four decimals is the bar — the ladder and this module fit the same "
              "likelihood on the\n  same rows with the same penalty, so anything larger "
              "means the frames differ and the\n  Stan rows below are not comparable to "
              "§7c at all.")

    chains = scored["chains"]
    if not chains.empty:
        print("\nPer chain, on the parameters R-hat is the wrong instrument for:")
        print(chains[["parameter", "chain", "mean", "sd", "spread_in_sds"]]
              .round(4).to_string(index=False))
        worst = chains["spread_in_sds"].max()
        print(f"  Largest between-chain gap: {worst:.3f} pooled posterior sds. Four chains "
              f"started at\n  theta = {THETA_INITS}, so this is a statement about the "
              f"posterior rather than about\n  where the sampler was pointed. A multimodal "
              f"fit shows up here and can hide from R-hat.")

    print("\nThe mixture block, point MLE against the posterior:")
    print(scored["parameters"][["term", "mle", "posterior_mean", "posterior_sd",
                                "z_from_mle", "mle_inside_95"]]
          .round(4).to_string(index=False))
    print("  These are BOUNDED parameters fitted inside a box by the ladder and given a "
          "prior here,\n  so the mode/MLE identity that pins the coefficient block does "
          "not hold for them. The\n  predictive table above is the check; this is where a "
          "disagreement would be localized.")

    artifacts = {
        "scores": (table, out_dir / "stan_availability_mixture.csv"),
        "parameters": (scored["parameters"],
                       out_dir / "stan_availability_mixture_parameters.csv"),
        "chains": (chains, out_dir / "stan_availability_mixture_chains.csv"),
        "diagnostics": (scored["diagnostics"],
                        out_dir / "stan_availability_mixture_diagnostics.csv"),
    }
    paths = {}
    for name, (frame, dest) in artifacts.items():
        frame.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(frame):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    import sys

    cfg = yaml.safe_load(open("configs/default.yaml"))
    if "--mixture-check" in sys.argv:
        run_mixture_check(cfg)
    else:
        run(cfg)
