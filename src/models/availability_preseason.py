"""P2 — the preseason increment on the AVAILABILITY head, as a nested arm on what ships.

`make availability-preseason`. Four artifacts in `outputs/predictions/`:
`availability_preseason.csv`, `availability_preseason_rolling.csv`,
`availability_preseason_effects.csv`, `availability_preseason_block.csv`.

## Why this head, and why the block is small

`docs/preseason-plan.md` P1 measured the preseason block's incremental out-of-sample R² on
the **draftable** population — every player on a season-start roster, the only population
this head is ever applied to — and got **+0.0198** on `gp_share` against +0.0492 on minutes
per game. That inversion is why P3 ran first. It also found the shape of what is left, and
it is the mirror image of P3's:

| column, alone, above the head's own prior-season block | ΔR² |
|---|---|
| `pre_log_min` — how many preseason minutes he logged | **+0.0504** |
| `pre_gp_share` | +0.0183 |
| `pre_played_final_game` | +0.0166 |
| `pre_missed_tail_share` | +0.0146 |
| `pre_d_mpg` — the delta P3 ships | +0.0126 |
| `pre_d_min_share_late` | +0.0027 |
| `has_preseason` | +0.0024 |
| *P1's seven-column block together* | *+0.0198* |

**Five of the seven columns beat the block that contains them**, which is P1 decision 4:
seven columns overfit a ridge on this target. So the declared primary here is **one column
plus the age split** — `pre_log_min` and P1's four `pre_missing__<age>` indicators, five
columns, exactly the arity P3's primary carried. The wider blocks ride as sensitivities so
decision 4 is re-read at this head's own unit rather than inherited from a ridge probe.

And the two heads want **different halves of the same panel**. Minutes wants the *delta*
(`missing_only` and `share_late_only` were ties there); availability wants the *level* of
participation and volume. That is not a contradiction — a preseason minutes delta says how
the coach is using him and a preseason minutes total says whether he is on the floor at all.

## The two places a preseason column can go, and they are different questions

`docs/availability-window-plan.md` §14d is the standing lesson: the shipped head is a
two-component mixture, `pi_i * BetaBinom(mu_low, rho_low) + (1 - pi_i) * BetaBinom(mu_i,
rho_i)`, and its mean function `beta` and its disrupted-season weight `pi` are different
questions about the same player. §14 measured a block that helped `beta` and *cost* CRPS on
`pi` — four parameters that fit and did not predict — and that is the cleanest null in that
document.

The preseason has an a-priori claim on `pi` that the absence composition did not:
"who missed the *tail* of the preseason, days before the opener" is a direct reading of who
is about to lose the season, where "what kind of absence he had last year" is a fact about
last year. So `pi` gets its own arm rather than being ruled out by precedent, and the two
arms are separate rows because they answer separate questions.

**The `l2` confound is stated rather than corrected, as §7d and §14b do.** The penalty
reaches `beta[1:]` only, so the block's columns *are* penalized on `beta` and are **not** on
`pi`: the `pi` arms carry 7 unpenalized parameters the shipped head does not. §7d swept
eight penalties from 0 to 256 and moved the reference by 0.00034 CRPS at its best, so the
confound is bounded — and it can only flatter the `pi` arms, which is the direction that
does not need correcting if they lose.

## The bar, stated before the run

`docs/preseason-plan.md` P2 states it, and it is §14's `wins_crps_holds_boundary` — D1 with
its halves swapped, because `mixture` already spent the boundary gain (0.0201 → 0.0109) and
what an arm bolted onto it has to buy is the CRPS the shipped head gave up:

1. **Validation.** The paired-bootstrap CRPS interval against the shipped head lies entirely
   below zero, **and** the `boundary_tail_error` margin's interval does not establish a
   loss — on the draftable population.
2. **The rolling-origin harness on the fitting half.** The same conjunction, plus a majority
   of origins won.

**Validation alone ships nothing, and on this head that is not caution for its own sake.**
§12e and §14f record two blocks that won a validation CRPS reading on these exact rows and
shrank **5.8×** and **4.2×** on the rolling harness, reopening their intervals across zero.
This is the third block to be asked the same question of the same head, and the first two
answers were "unconfirmed".

## The population is load-bearing here, unlike the coverage window

Two restrictions matter on this head and only one of them binds.

**Coverage does not.** The shipped head fits the `three_point_era` window (2012-13 onward)
and the preseason panel begins at 2004-05, so every fitting row and every validation row is
inside coverage. P3 had to cut its own window and pay 1.19 CRPS minutes for it; here there
is nothing to cut. The rolling harness's first origin is derived from the panel rather than
hard-coded, for the same reason.

**Population does.** P1's first reading on this head was +0.1171 R² and **6× of it was
mid-season signings** — players with no preseason row for a contract reason and a small
`gp_share` for the same one. Restricting to the season-start roster took the same block to
+0.0198. So every arm is scored on both populations, the artifact carries the column, and
the verdict is read on `draftable` (P1 decision 5). The arms are **not** refitted per
population: the head fits what it fits, and restricting the fitting rows as well would
confound a population statement with a smaller training set.

Usage:
    python -m src.models.availability_preseason
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year
from src.eda.preseason_value import (MISSING_AGE_COLS, attach_availability_block,
                                     attach_missing_age_indicators, attach_prior_shares,
                                     attach_season_start_roster, covered_seasons,
                                     season_centered)
from src.features.availability import build_panel, season_availability
from src.models.availability import (FEATURE_COLS, build_design, season_start_dates)
from src.models.availability_absence import (_bootstrap_arms, _d1, _rolling_parts, _suffix,
                                             _wins_crps_holds_boundary, interaction_table)
from src.models.availability_window import (BOOTSTRAP_REPS, LIKELIHOOD_LOOKBACK,
                                            LIKELIHOOD_WINDOW, MIN_ROLE_ROWS, PI_COLS,
                                            WINDOWS, MixtureFrailty, _origin_scores,
                                            _tail_errors, _tail_parts, assert_nests,
                                            paired_bootstrap, restrict_window, score_arm)
from src.eda.season_effects import ROLE_LABELS
from src.models.held_out import selection_split
from src.models.season_terms import season_start_year

# ── The block ─────────────────────────────────────────────────────────────────

#: The single column P1 measured as worth more alone (+0.0504) than its whole block
#: (+0.0198): `log1p` of the player's total preseason minutes. A level rather than a delta,
#: which is what this head turns out to want — see the module docstring.
VOLUME = "pre_log_min"

#: The same column with each season's own preseason mean removed. P3's finding asked of a
#: different quantity: preseason minutes totals carry a **season-level** shift (2 games a
#: team in the 2011-12 lockout, a December preseason in 2020-21, 8 games in an ordinary
#: year), and this head has no season term to absorb it (`docs/availability-window-plan.md`
#: §7i). Whether that level is nuisance or signal is a measurement, not a guess.
VOLUME_CENTERED = "pre_log_min_centered"

#: Participation, with no prior-season equivalent and therefore no delta: what share of his
#: team's preseason he played, what share of its *tail* he missed, and whether he was on the
#: floor for the last one. These are the columns with an a-priori claim on `pi`.
PARTICIPATION = ["pre_gp_share", "pre_missed_tail_share", "pre_played_final_game"]

#: The two difference-coded columns P1's block carries, kept named so `p1_block` is P1's
#: block and not a paraphrase of it.
P1_DELTAS = ["pre_d_mpg", "pre_d_min_share_late"]

#: The arms, as `(columns added to beta, columns added to pi)`. Every arm is the shipped
#: head plus these, so a zero coefficient recovers it exactly — the house nesting discipline,
#: and here it is stronger than usual on the `pi` side: `pi = theta * sigmoid(gamma' z)`
#: switches off through `theta` alone, so widening `z` adds parameters the nesting point does
#: not depend on and `assert_nests` still reads 0.0 (§14b).
ARMS: dict[str, tuple[list[str], list[str]]] = {
    # The reference: the head that ships, with no preseason column anywhere.
    "mixture": ([], []),
    # PRIMARY — one column plus the age split. P1 decision 4 (the block overfits this
    # target) and decision 3 (the indicator splits on age), together.
    "mixture__volume": ([VOLUME] + MISSING_AGE_COLS, []),
    # ATTRIBUTION — the indicators alone. `pre_log_min` is 0 on a missing row, so it is
    # collinear with the indicator by construction and the primary's gain could in principle
    # be "a preseason row exists" wearing a volume column's name. This is the row that says
    # otherwise, and P1 predicts it is nearly nothing: `has_preseason` alone scored +0.0024.
    "mixture__missing_only": (list(MISSING_AGE_COLS), []),
    # ATTRIBUTION — the same volume column with each season's own mean removed. If the gain
    # survives, the block is a cross-player reading; if it collapses, it was a league-level
    # calendar fact the intercept should have owned. P3 found the second on its own head.
    "mixture__volume_centered": ([VOLUME_CENTERED] + MISSING_AGE_COLS, []),
    # SENSITIVITY — volume plus the three participation levels on `beta`. Four of P1's top
    # five columns; if the primary is right about decision 4 this should not beat it.
    "mixture__participation": ([VOLUME] + PARTICIPATION + MISSING_AGE_COLS, []),
    # SENSITIVITY — P1's seven-column block with `has_preseason` replaced by its age split.
    # An arm that exists to be beaten, which is how P1's own reading gets checked at this
    # head's unit rather than assumed.
    "mixture__p1_block": ([VOLUME] + PARTICIPATION + P1_DELTAS + MISSING_AGE_COLS, []),
    # The `pi` question — participation on the disrupted-season weight and nothing on the
    # mean. "Who missed the tail of the preseason" is a direct reading of who is about to
    # lose the season, which is what `pi` is for.
    "mixture__pi": ([], PARTICIPATION + MISSING_AGE_COLS),
    # And both, so the incremental effect of `pi` GIVEN `beta` is a paired interval rather
    # than two point estimates read side by side — §14d's arithmetic, on this block.
    "mixture__volume_pi": ([VOLUME] + MISSING_AGE_COLS, PARTICIPATION + MISSING_AGE_COLS),
}

#: The head that ships (`docs/availability-window-plan.md` §7i), and therefore what every
#: margin is quoted against. Quoting `betabinom` would price the block against a head that
#: was retired — the mistake §12 made and §14 exists to correct.
REFERENCE_ARM = "mixture"

#: Declared before the run. The gate's verdict is read off this arm and nothing else; every
#: other row is a sensitivity or an attribution on it.
PRIMARY_ARM = "mixture__volume"

#: What actually ships, and it is **not** the arm declared before the run: the fitting half
#: selected P1's full block and that is what ships.
#:
#: On the rolling harness `p1_block` beats `PRIMARY_ARM` on CRPS at −0.0977
#: [−0.1569, −0.0408] and on `boundary_tail_error` at −0.0014 [−0.0019, −0.0010] — both
#: intervals clear of zero, 8 of 10 origins — under P3's promotion rule: an arm preferred
#: after seeing validation is a validation-driven swap, preferred on the fitting half it is a
#: decision the selection split never paid for. It **reverses P1 decision 4**, which said
#: this head's block should be smaller than P1's seven columns because a ridge overfit them.
#: At this head's own unit the wider block is measurably better.
#:
#: ⚠️ **This makes a COMPLETE preseason a production precondition, not a preference.**
#: `pre_missed_tail_share` and `pre_played_final_game` are read over the preseason's tail and
#: are undefined until it is over. A centred-volume arm that survives a truncated capture was
#: adopted and then **withdrawn on 2026-08-13**: the operational premise behind it — that DK
#: contests might fill before the final preseason game — was contradicted by the owner's own
#: experience drafting after the 2025-26 preseason ended and securing entries. So the head
#: takes the better model and the runbook's Oct 17-20 window becomes load-bearing rather than
#: advisory. What that costs if the premise ever turns: measured 7 days early, 3.4% of
#: players have no panel row against a ~4.2% base, and two of these ten columns do not exist
#: at all.
SHIPPED_ARM = "mixture__p1_block"

#: The population every verdict is read on — P1 decision 5. `all` is written beside it and
#: is a population statement rather than a model one.
DECISION_POPULATION = "draftable"
POPULATIONS = ("all", DECISION_POPULATION)

#: The effects the ladder is read as, `(label, arm, baseline)`. Negative is better on all
#: four metrics, since three are absolute calibration errors and the fourth is CRPS.
CONTRASTS: tuple[tuple[str, str, str], ...] = (
    ("preseason on beta | mixture", "mixture__volume", "mixture"),
    ("preseason on pi | mixture", "mixture__pi", "mixture"),
    ("preseason on beta+pi | mixture", "mixture__volume_pi", "mixture"),
    ("preseason on pi | already on beta", "mixture__volume_pi", "mixture__volume"),
    ("the age-split indicator alone", "mixture__missing_only", "mixture"),
    ("centring the volume column", "mixture__volume_centered", "mixture__volume"),
    ("participation on beta | volume", "mixture__participation", "mixture__volume"),
    ("P1's whole block | volume", "mixture__p1_block", "mixture__volume"),
)

#: `mixture`'s validation `boundary_tail_error` (§7c/§14c). Carried as a level so the round
#: is readable against the defect it inherited rather than only against its own reference.
MIXTURE_BOUNDARY = 0.0109

SEED = 42


def arm_columns(name: str) -> tuple[list[str], list[str]]:
    """`(features, pi_features)` for one arm — the shipped lists plus that arm's block."""
    beta_extra, pi_extra = ARMS[name]
    return list(FEATURE_COLS) + list(beta_extra), list(PI_COLS) + list(pi_extra)


def attach_preseason(design: pd.DataFrame, panel: pd.DataFrame, av_panel: pd.DataFrame,
                     seasons: list[str]) -> pd.DataFrame:
    """Every preseason column this ladder can read, merged onto the availability design.

    **Opt-in, and outside `availability.build_design` deliberately** — the
    `attach_absence_mix` precedent, for the identical reason. That builder is the route
    seven modules take to their rows (`stan_minutes`, `stan_composition`,
    `stan_games_played`, `model_cards`, `sim/season`, `season_terms`, `final_evaluation`),
    and a column that is structurally zero before 2004-05 must not be able to enter any of
    them by accident. A caller opts in here and widens its own feature list, which is what
    keeps this an ablation rather than a change of head.

    P1's builders are **reused rather than forked**, as `docs/preseason-plan.md` specifies:
    `attach_prior_shares` supplies the within-team share the late-share delta is taken
    against, `attach_availability_block` supplies P1's seven columns on P1's own scales, and
    `attach_missing_age_indicators` / `season_centered` are the two encodings P3 built and
    P1 decided on.
    """
    out = attach_availability_block(attach_prior_shares(design, av_panel, seasons), panel)
    out = attach_missing_age_indicators(out)
    out[VOLUME_CENTERED] = season_centered(out, VOLUME)
    return out


#: Every preseason column any arm can read — what `run` asserts is finite before fitting.
BLOCK_COLS = ([VOLUME, VOLUME_CENTERED] + PARTICIPATION + P1_DELTAS
              + list(MISSING_AGE_COLS))


# ── The ladder ────────────────────────────────────────────────────────────────

def fit_arms(train: pd.DataFrame, l2: float = 1.0,
             names: tuple[str, ...] | None = None) -> dict[str, MixtureFrailty]:
    """Every arm fitted once, on the shipped window's rows.

    One fit per arm and not one per population: the head fits what it fits, and P1 decision 5
    is about where a figure is *read*. Refitting on the draftable rows alone would confound a
    population statement with a smaller training set.
    """
    fitted: dict[str, MixtureFrailty] = {}
    for name in (names or tuple(ARMS)):
        features, pi_features = arm_columns(name)
        missing = [c for c in features + pi_features if c not in train.columns]
        if missing:
            raise ValueError(f"{name} needs {missing}; call `attach_preseason` first")
        if train[features + pi_features].isna().any().any():
            bad = [c for c in features + pi_features if train[c].isna().any()]
            raise ValueError(f"{name} has NaN in {bad} on the fitting window")
        model = MixtureFrailty(l2=l2, features=features, pi_features=pi_features).fit(train)
        model.nesting_gap = assert_nests(model, train)
        fitted[name] = model
        print(f"  {name:<28} {model.n_params:>3} params  "
              f"({len(ARMS[name][0]):>2} on beta, {len(ARMS[name][1]):>2} on pi)  "
              f"train ll {model.train_loglik:,.2f}  nesting gap {model.nesting_gap:.1e}")
    return fitted


def score_population(fitted: dict[str, MixtureFrailty], train: pd.DataFrame,
                     val: pd.DataFrame, mask: np.ndarray, population: str,
                     max_games: int, reference: str = REFERENCE_ARM,
                     seed: int = SEED) -> tuple[pd.DataFrame, dict, dict]:
    """Every fitted arm scored on one population of validation rows.

    Returns `(table, per-row CRPS, tail parts)` — the last two because the effects table
    needs the arms' scores together, and a paired bootstrap cannot be reconstructed from a
    summary row.
    """
    frame = val.loc[mask].reset_index(drop=True)
    y, n = frame["gp"].to_numpy(), frame["team_games"].to_numpy()
    rows, per_row, tails = [], {}, {}

    for name, model in fitted.items():
        features, pi_features = arm_columns(name)
        row, scores = score_arm(name, model, train, frame, features, max_games, seed)
        beta_extra, pi_extra = ARMS[name]
        row.update({"population": population, "n_val": len(frame),
                    "n_preseason_beta": len(beta_extra), "n_preseason_pi": len(pi_extra),
                    "preseason_beta": "|".join(beta_extra),
                    "preseason_pi": "|".join(pi_extra),
                    "n_params": model.n_params, "nesting_loglik_gap": model.nesting_gap,
                    "train_loglik": model.train_loglik,
                    "train_loglik_incumbent": model.incumbent_loglik,
                    "selectable": name != reference,
                    "share_has_preseason": float(frame["has_preseason"].mean())})
        row.update({f"shape_{k}": v for k, v in model.shape_report(frame).items()})
        rows.append(row)
        per_row[name] = scores
        tails[name] = _tail_parts(model.predict_pmf(frame, max_games), y, n)

    suffix = _suffix(reference)
    ref_scores = per_row[reference]
    boundary = _bootstrap_arms(tails, tuple(fitted), reference, reps=BOOTSTRAP_REPS,
                               seed=seed)
    # And a second set of margins against the DECLARED PRIMARY, so "arm X beats the arm we
    # said we would ship" is an interval rather than two point estimates read side by side.
    # `minutes_preseason` carries this and P3 used it to promote an attribution arm on a
    # fitting-half decision; without it a ladder can only say that two arms both beat the
    # reference, which is not the same question.
    vs_primary = _bootstrap_arms(tails, tuple(fitted), PRIMARY_ARM, reps=BOOTSTRAP_REPS,
                                 seed=seed, suffix="primary")
    for row in rows:
        delta, lo, hi = paired_bootstrap(per_row[row["arm"]], ref_scores, seed=seed)
        row[f"crps_vs_{suffix}"] = delta
        row[f"crps_vs_{suffix}_lo"] = lo
        row[f"crps_vs_{suffix}_hi"] = hi
        row[f"beats_{suffix}"] = bool(hi < 0.0)
        against = paired_bootstrap(per_row[row["arm"]], per_row[PRIMARY_ARM], seed=seed)
        row["crps_vs_primary"] = against[0]
        row["crps_vs_primary_lo"] = against[1]
        row["crps_vs_primary_hi"] = against[2]
        row["beats_primary"] = bool(row["arm"] != PRIMARY_ARM and against[2] < 0.0)
        row.update({k: v for k, v in vs_primary[row["arm"]].items()
                    if k.endswith(("_vs_primary", "_vs_primary_lo", "_vs_primary_hi"))
                    or k.startswith("beats_primary_")})
        row["beats_mixture_boundary"] = bool(row["boundary_tail_error"] < MIXTURE_BOUNDARY)
        row.update(boundary[row["arm"]])
        # Both predicates on every row, per §14a. D1 asks for a calibration gain and settles
        # for CRPS non-inferiority; this round's bar is its mirror image, because the head
        # being extended has already spent the calibration gain. An arm that passes one and
        # fails the other is the reading rather than an anomaly to be resolved.
        row["d1_passes"] = _d1(row.get(f"boundary_vs_{suffix}_hi", np.nan),
                               row[f"crps_vs_{suffix}_lo"])
        row["wins_crps_holds_boundary"] = _wins_crps_holds_boundary(
            row.get(f"boundary_vs_{suffix}_lo", np.nan), row[f"crps_vs_{suffix}_hi"])
    return pd.DataFrame(rows), per_row, tails


# ── Rolling-origin confirmation, fitting half only ────────────────────────────

def rolling_confirmation(train: pd.DataFrame, max_games: int, first_origin: int,
                         l2: float = 1.0, lookback: int = LIKELIHOOD_LOOKBACK,
                         names: tuple[str, ...] | None = None,
                         reference: str = REFERENCE_ARM,
                         seed: int = SEED) -> pd.DataFrame:
    """Walk-forward over the fitting half: one fit per (origin, arm), pooled.

    **This is half the gate, not a post-hoc check**, and on this head it is the half that has
    decided the last two rounds. §12e and §14f both record a block that won validation on
    these rows and shrank 4-6x here with its interval reopened across zero; §10e is the
    standing rule that a fresh winner fails until it replicates.

    `first_origin` is derived from the preseason coverage artifact rather than hard-coded:
    at a lookback of `lookback` seasons, the earliest origin whose *whole* fitting window
    carries a preseason is the first covered season plus the lookback. Restricting the
    origins rather than dropping rows is what keeps every arm on identical rows.

    Both populations come out of one pass. The scored rows are masked *after* the fit, so the
    draftable reading is the same fitted arm on fewer rows rather than a different model.
    """
    years = season_start_year(train)
    origins = [int(y) for y in np.unique(years) if y >= first_origin]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        score = train[years == origin]
        fit_rows = train[(years < origin) & (years >= origin - lookback)]
        if score.empty or len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
            continue
        for name in (names or tuple(ARMS)):
            features, pi_features = arm_columns(name)
            model = MixtureFrailty(l2=l2, features=features,
                                   pi_features=pi_features).fit(fit_rows)
            assert_nests(model, fit_rows)
            scored = _origin_scores(model, score, max_games)
            scored["origin"] = np.full(len(score), origin)
            scored["draftable"] = score["on_season_start_roster"].to_numpy(dtype=float)
            slot = per_arm.setdefault(name, {})
            for key, values in scored.items():
                slot.setdefault(key, []).append(values)
        print(f"  origin {origin}: {len(fit_rows):,} fit / {len(score):,} scored "
              f"({int(score['on_season_start_roster'].sum()):,} draftable)")

    pooled = {k: {m: np.concatenate(v) for m, v in d.items()} for k, d in per_arm.items()}
    suffix = _suffix(reference)
    rows = []
    for population in POPULATIONS:
        keep = (np.ones(len(pooled[reference]["y"]), dtype=bool) if population == "all"
                else pooled[reference]["draftable"] > 0)
        cut = {name: {m: v[keep] for m, v in d.items()} for name, d in pooled.items()}
        parts = {name: _rolling_parts(d) for name, d in cut.items()}
        ref_scores = cut[reference]["crps"]
        primary_scores = cut[PRIMARY_ARM]["crps"]
        margins = _bootstrap_arms(parts, tuple(cut), reference, reps=BOOTSTRAP_REPS,
                                  seed=seed)
        vs_primary = _bootstrap_arms(parts, tuple(cut), PRIMARY_ARM, reps=BOOTSTRAP_REPS,
                                     seed=seed, suffix="primary")
        grid = np.linspace(0, 1, 101)
        for name, d in cut.items():
            y, n, org, u = d["y"], d["n"], d["origin"], d["pit"]
            delta, lo, hi = paired_bootstrap(d["crps"], ref_scores, seed=seed)
            boundary, body, shoulder = _tail_errors(parts[name], np.arange(len(y)))
            beta_extra, pi_extra = ARMS[name]
            row = {
                "arm": name, "population": population,
                "n_preseason_beta": len(beta_extra), "n_preseason_pi": len(pi_extra),
                "n_origins": int(len(np.unique(org))), "n_scored": int(len(y)),
                "crps": float(d["crps"].mean()),
                f"crps_vs_{suffix}": delta, f"crps_vs_{suffix}_lo": lo,
                f"crps_vs_{suffix}_hi": hi,
                "origins_won": sum(1 for o in np.unique(org)
                                   if d["crps"][org == o].mean()
                                   < ref_scores[org == o].mean()),
                "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u)
                                              - grid))),
                "err_below_10": float(d["p_below_10"].mean() - (y < 10).mean()),
                "err_full_schedule": float(d["p_full"].mean() - (y == n).mean()),
                "boundary_tail_error": boundary, "body_error": body,
                "shoulder_error": shoulder,
            }
            row.update(margins[name])
            # Against the declared primary, on the fitting half — the only reading that can
            # promote an arm over it without spending the selection split on a choice made
            # after seeing it. P3's rule, and the column P2's first run did not have.
            against = paired_bootstrap(d["crps"], primary_scores, seed=seed)
            row["crps_vs_primary"] = against[0]
            row["crps_vs_primary_lo"] = against[1]
            row["crps_vs_primary_hi"] = against[2]
            row["origins_won_vs_primary"] = sum(
                1 for o in np.unique(org)
                if d["crps"][org == o].mean() < primary_scores[org == o].mean())
            row["beats_primary"] = bool(name != PRIMARY_ARM and against[2] < 0.0)
            row.update({k: v for k, v in vs_primary[name].items()
                        if k.endswith(("_vs_primary", "_vs_primary_lo", "_vs_primary_hi"))
                        or k.startswith("beats_primary_")})
            row["d1_passes"] = _d1(row.get(f"boundary_vs_{suffix}_hi", np.nan),
                                   row[f"crps_vs_{suffix}_lo"])
            row["wins_crps_holds_boundary"] = _wins_crps_holds_boundary(
                row.get(f"boundary_vs_{suffix}_lo", np.nan), row[f"crps_vs_{suffix}_hi"])
            rows.append(row)
    return pd.DataFrame(rows)


# ── The gate ──────────────────────────────────────────────────────────────────

def gate(validation: pd.DataFrame, rolling: pd.DataFrame, arm: str = PRIMARY_ARM,
         population: str = DECISION_POPULATION, reference: str = REFERENCE_ARM) -> dict:
    """Both halves of the bar, evaluated as code rather than as a judgement after the fact.

    A Stan port is earned **only if this passes** — `docs/preseason-plan.md` P2 — so the
    go/no-go is a column in the artifact rather than a sentence in a printout.

    The challenger block is P3's device: an arm that beats the declared primary on the
    **fitting half** can be promoted without spending the selection split on a choice made
    after seeing it, and an arm that beats it on validation alone is a validation-driven
    swap and is recorded as such.
    """
    suffix = _suffix(reference)
    val = validation[(validation["arm"] == arm)
                     & (validation["population"] == population)].iloc[0]
    roll = rolling[(rolling["arm"] == arm)
                   & (rolling["population"] == population)].iloc[0]
    val_pass = bool(val["wins_crps_holds_boundary"])
    roll_pass = bool(roll["wins_crps_holds_boundary"]
                     and roll["origins_won"] * 2 > roll["n_origins"])

    # A challenger is an arm that beats the declared primary **on the fitting half**, with
    # an interval and a majority of origins. P3's rule: an arm preferred after seeing
    # validation is a validation-driven swap; preferred on the rolling harness it is a
    # fitting-half decision and may be carried forward.
    roll_pop = rolling[rolling["population"] == population]
    beaten = roll_pop[(roll_pop["arm"] != arm) & (roll_pop["arm"] != reference)
                      & (roll_pop["crps_vs_primary_hi"] < 0.0)
                      & (roll_pop["origins_won_vs_primary"] * 2 > roll_pop["n_origins"])]
    best = str(beaten.sort_values("crps_vs_primary").iloc[0]["arm"]) if len(beaten) else ""

    return {
        "arm": arm, "population": population, "reference": reference,
        "val_crps_delta": float(val[f"crps_vs_{suffix}"]),
        "val_lo": float(val[f"crps_vs_{suffix}_lo"]),
        "val_hi": float(val[f"crps_vs_{suffix}_hi"]),
        "val_boundary_delta": float(val[f"boundary_vs_{suffix}"]),
        "val_boundary_lo": float(val[f"boundary_vs_{suffix}_lo"]),
        "val_boundary_hi": float(val[f"boundary_vs_{suffix}_hi"]),
        "val_pass": val_pass,
        "rolling_crps_delta": float(roll[f"crps_vs_{suffix}"]),
        "rolling_lo": float(roll[f"crps_vs_{suffix}_lo"]),
        "rolling_hi": float(roll[f"crps_vs_{suffix}_hi"]),
        "rolling_boundary_delta": float(roll[f"boundary_vs_{suffix}"]),
        "rolling_boundary_lo": float(roll[f"boundary_vs_{suffix}_lo"]),
        "rolling_boundary_hi": float(roll[f"boundary_vs_{suffix}_hi"]),
        "origins_won": int(roll["origins_won"]), "n_origins": int(roll["n_origins"]),
        "rolling_pass": roll_pass,
        # The shrinkage §12e and §14f both measured, as a number rather than a sentence.
        # Above 1 means validation claimed more than the fitting half replicates.
        "shrinkage_vs_validation": (float(val[f"crps_vs_{suffix}"]
                                          / roll[f"crps_vs_{suffix}"])
                                    if roll[f"crps_vs_{suffix}"] else np.nan),
        "challenger": best,
        "challenger_rolling_delta": (float(beaten.iloc[0]["crps_vs_primary"])
                                     if best else np.nan),
        "challenger_rolling_hi": (float(beaten.iloc[0]["crps_vs_primary_hi"])
                                  if best else np.nan),
        "challenger_confirmed_on_fitting_half": bool(best),
        "passes": bool(val_pass and roll_pass),
        "earns_stan_port": bool(val_pass and roll_pass),
    }


# ── The block's own diagnostics ───────────────────────────────────────────────

def block_diagnostics(train: pd.DataFrame, val: pd.DataFrame,
                      scope: list[str]) -> pd.DataFrame:
    """What the block is made of, as an artifact rather than a printout.

    Four things this round asserts rather than assumes, none of which survives being left in
    a log line: that the shipped window sits entirely inside preseason coverage (so nothing
    had to be cut, unlike P3), how much of each population carries a preseason row, how far
    the season means of the volume column move (which is what the centred arm exists to
    remove), and the range of `team_pre_games` inside the window (the 2011-12 lockout is the
    reason `pre_missed_tail_share` is a share and not a count).
    """
    rows: list[dict] = []
    uncovered = sorted(set(train["season"].unique()) - set(scope))
    rows += [
        {"scope": "fitting window", "statistic": "n_rows", "value": float(len(train))},
        {"scope": "fitting window", "statistic": "n_seasons",
         "value": float(train["season"].nunique())},
        {"scope": "fitting window", "statistic": "n_seasons_outside_coverage",
         "value": float(len(uncovered))},
        {"scope": "fitting window", "statistic": "share_has_preseason",
         "value": float(train["has_preseason"].mean())},
        {"scope": "fitting window", "statistic": "share_draftable",
         "value": float(train["on_season_start_roster"].mean())},
    ]
    for label, frame in (("validation", val),
                         ("validation draftable",
                          val[val["on_season_start_roster"] > 0])):
        rows += [
            {"scope": label, "statistic": "n_rows", "value": float(len(frame))},
            {"scope": label, "statistic": "share_has_preseason",
             "value": float(frame["has_preseason"].mean())},
        ]
    present = train[train["has_preseason"] > 0]
    by_season = present.groupby("season")[VOLUME].mean()
    rows += [
        {"scope": "fitting window", "statistic": "volume_season_mean_min",
         "value": float(by_season.min())},
        {"scope": "fitting window", "statistic": "volume_season_mean_max",
         "value": float(by_season.max())},
        {"scope": "fitting window", "statistic": "volume_season_mean_sd",
         "value": float(by_season.std(ddof=1))},
        {"scope": "fitting window", "statistic": "volume_sd_within_season",
         "value": float((present[VOLUME] - present["season"].map(by_season)).std(ddof=1))},
        {"scope": "fitting window", "statistic": "team_pre_games_min",
         "value": float(present["team_pre_games"].min())},
        {"scope": "fitting window", "statistic": "team_pre_games_max",
         "value": float(present["team_pre_games"].max())},
    ]
    # The share of a missing-preseason row's mass the age split actually separates: if one
    # cell carried everything the split would be an indicator with extra columns.
    missing = train[train["has_preseason"] <= 0]
    for col in MISSING_AGE_COLS:
        rows.append({"scope": "fitting window", "statistic": f"share_missing_{col}",
                     "value": float(missing[col].mean()) if len(missing) else np.nan})
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def load_design(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, int, list[str]]:
    """`(train, validation, max_games, covered seasons)` with the preseason block attached.

    The held-out split is never materialized: `selection_split` hands back the two frames
    this ladder is allowed to see and nothing else.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["eda"]["output_dir"])
    seasons = cfg["data"]["seasons"]
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))

    panel = pd.read_parquet(features_dir / "preseason.parquet")
    scope = covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv"))

    av_panel = build_panel(seasons, raw_dir)
    frame = season_availability(av_panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(av_panel))
    design = attach_preseason(design, panel, av_panel, seasons)
    design = attach_season_start_roster(design, seasons, raw_dir, window_games)

    train, val = selection_split(design)
    return train, val.reset_index(drop=True), int(design["team_games"].max()), scope


def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", SEED))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    train, val, max_games, scope = load_design(cfg)
    cut = restrict_window(train, WINDOWS[LIKELIHOOD_WINDOW]).reset_index(drop=True)
    first_covered = _season_start_year(scope[0])
    first_origin = first_covered + LIKELIHOOD_LOOKBACK

    print("Availability preseason increment (P2) — the preseason block as a nested arm on "
          "the head\nthat ships (`mixture`, three_point_era window, no season term, "
          "role-graded rho)")
    print(f"  {len(train):,} train / {len(val):,} validation "
          f"({', '.join(sorted(val['season'].unique()))}); the held-out split is LOCKED and "
          f"is never materialized here.")
    print(f"  Shipped window ({LIKELIHOOD_WINDOW}): {len(cut):,} fitting rows, "
          f"{min(cut['season'])} → {max(cut['season'])}.")
    print(f"  Preseason coverage begins {scope[0]}, so the whole window is inside it and "
          f"NOTHING is cut —\n  P3 had to restrict its own window and paid 1.19 CRPS "
          f"minutes for it before any preseason\n  column existed.")
    draftable = val["on_season_start_roster"].to_numpy(dtype=float) > 0
    print(f"  Draftable validation rows (on a season-start roster): {int(draftable.sum()):,}"
          f" of {len(val):,} ({draftable.mean():.1%}); "
          f"{val['has_preseason'].mean():.1%} of validation rows carry a preseason row "
          f"({val.loc[draftable, 'has_preseason'].mean():.1%} of the draftable ones).")
    print("  The BAR, stated before the run: a validation CRPS interval clear of zero with "
          "the\n  boundary HELD, on the draftable population, AND the rolling harness "
          "agreeing.\n  Validation alone ships nothing — §12e and §14f are two blocks that "
          "won this exact\n  reading on these exact rows and shrank 5.8x and 4.2x rolling.")

    if cut[BLOCK_COLS].isna().any().any():
        bad = [c for c in BLOCK_COLS if cut[c].isna().any()]
        raise ValueError(f"NaN preseason columns on the fitting window: {bad}")

    block = block_diagnostics(cut, val, scope)
    block_dest = out_dir / "availability_preseason_block.csv"
    block.to_csv(block_dest, index=False)
    print(f"\nStep 1 — the block's own diagnostics → {block_dest}")
    print(block.round(4).to_string(index=False))

    print(f"\nStep 2 — the ladder: {len(ARMS)} nested arms. The reference carries no "
          f"preseason column;\n  every other arm is it plus columns, on `beta`, on `pi` or "
          f"on both, so zero recovers it\n  exactly (`theta = 0` nests at any width of "
          f"`pi`):")
    fitted = fit_arms(cut, l2=l2)

    blocks, per_row, tails = [], {}, {}
    for population in POPULATIONS:
        mask = (np.ones(len(val), dtype=bool) if population == "all" else draftable)
        table, scores, parts = score_population(fitted, cut, val, mask, population,
                                                max_games, seed=seed)
        blocks.append(table)
        if population == DECISION_POPULATION:
            per_row, tails = scores, parts
    validation = pd.concat(blocks, ignore_index=True)

    show = ["arm", "n_preseason_beta", "n_preseason_pi", "n_val", "train_loglik",
            "val_crps", "crps_vs_mixture", "crps_vs_mixture_lo", "crps_vs_mixture_hi",
            "val_pit_ks", "boundary_tail_error", "boundary_vs_mixture",
            "boundary_vs_mixture_lo", "boundary_vs_mixture_hi", "body_error",
            "shoulder_error", "wins_crps_holds_boundary"]
    for population, label in (
            (DECISION_POPULATION, "ON A SEASON-START ROSTER — the production population, "
                                  "and the one the verdict is read on"),
            ("all", "every row the head fits — a population statement, not a model one")):
        part = validation[validation["population"] == population]
        print(f"\n  {label}:")
        print(part[show].round(4).to_string(index=False))

    dest = out_dir / "availability_preseason.csv"
    validation.to_csv(dest, index=False)
    print(f"\nWrote {len(validation):,} scored rows → {dest}")

    print("\nStep 3 — the ladder read as effects, on the draftable population. The `pi` "
          "rows are\n  §14d's question asked of a block that has an a-priori claim on it:")
    effects = interaction_table(per_row, tails, reps=BOOTSTRAP_REPS, seed=seed,
                               contrasts=CONTRASTS, interactions=())
    for metric in ("val_crps", "boundary_tail_error"):
        for _, r in effects[effects["metric"] == metric].iterrows():
            print(f"  {metric:<19} {r['effect']:<36} {r['delta']:+.5f} "
                  f"[{r['delta_lo']:+.5f}, {r['delta_hi']:+.5f}]")
    eff_dest = out_dir / "availability_preseason_effects.csv"
    effects.to_csv(eff_dest, index=False)
    print(f"Wrote {len(effects):,} effect rows → {eff_dest}")

    print(f"\nStep 4 — rolling-origin confirmation on the FITTING HALF ONLY (lookback "
          f"{LIKELIHOOD_LOOKBACK},\n  origins from {first_origin} = the first covered "
          f"season {scope[0]} plus the lookback). This is half\n  the gate, not a post-hoc "
          f"check.")
    rolling = rolling_confirmation(train, max_games, first_origin, l2=l2, seed=seed)
    roll_dest = out_dir / "availability_preseason_rolling.csv"
    rolling.to_csv(roll_dest, index=False)
    roll_show = ["arm", "n_scored", "crps", "crps_vs_mixture", "crps_vs_mixture_lo",
                 "crps_vs_mixture_hi", "origins_won", "n_origins", "pit_ks",
                 "boundary_tail_error", "boundary_vs_mixture", "boundary_vs_mixture_lo",
                 "boundary_vs_mixture_hi", "wins_crps_holds_boundary"]
    print(f"\n  {DECISION_POPULATION}:")
    print(rolling[rolling["population"] == DECISION_POPULATION][roll_show]
          .round(4).to_string(index=False))
    print(f"\nWrote {len(rolling):,} arm x population rows → {roll_dest}")

    result = gate(validation, rolling, PRIMARY_ARM, DECISION_POPULATION)
    print(f"\nStep 5 — the gate on the PRIMARY arm (`{PRIMARY_ARM}`), both halves:")
    print(f"    validation      CRPS {result['val_crps_delta']:+8.4f} "
          f"[{result['val_lo']:+.4f}, {result['val_hi']:+.4f}]   "
          f"boundary {result['val_boundary_delta']:+.5f} "
          f"[{result['val_boundary_lo']:+.5f}, {result['val_boundary_hi']:+.5f}]   "
          f"{'PASS' if result['val_pass'] else 'FAIL'}")
    print(f"    rolling origin  CRPS {result['rolling_crps_delta']:+8.4f} "
          f"[{result['rolling_lo']:+.4f}, {result['rolling_hi']:+.4f}]   "
          f"boundary {result['rolling_boundary_delta']:+.5f} "
          f"[{result['rolling_boundary_lo']:+.5f}, {result['rolling_boundary_hi']:+.5f}]   "
          f"{result['origins_won']}/{result['n_origins']} origins   "
          f"{'PASS' if result['rolling_pass'] else 'FAIL'}")
    print(f"    {'GATE PASSES' if result['passes'] else 'GATE FAILS'} — a Stan port is "
          f"earned only if this passes\n    (`docs/preseason-plan.md` P2), so "
          f"`earns_stan_port` = {result['earns_stan_port']}.")
    if not result["passes"]:
        print("    A failed gate is a result and is recorded as one. §10e is the rule: a "
              "fresh winner\n    fails until it replicates, and the session bank shrinks "
              "rather than the bar.")

    validation = pd.concat(
        [validation, pd.DataFrame([{**result, "arm": f"gate__{result['arm']}",
                                    "verdict": "gate"}])], ignore_index=True)
    validation.to_csv(dest, index=False)

    return {"availability_preseason": dest, "availability_preseason_rolling": roll_dest,
            "availability_preseason_effects": eff_dest,
            "availability_preseason_block": block_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
