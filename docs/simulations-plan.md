# Simulations Plan: Drafting and Scoring Best-Ball Portfolios

This is a planning doc, not a measurement report. It records direction, not results — update it
in place as pieces get built, the way `availability-plan.md` was.

**Status: specification settled 2026-08-08, nothing built.** The prediction layer is complete
and its heads all ship; this layer is the reason they exist. Before 2026-08-08 this doc named
four things to build and said nothing about the order, the interfaces between them, or the
artifacts — it was the only plan doc in `docs/` a session could not start work from. That
planning session settled six decisions, found one hard prerequisite nobody had noticed, and
put the whole thing on a calendar. All three are recorded below.

## Purpose

Closes the loop `~/Documents/nba_stats` never finished: a draft simulator, a season/tournament
outcome simulator, and a backtesting harness, so draft strategies (ADP-blended ranking,
exposure caps, posterior-sampled diversification, stacking) can be **compared on real historical
seasons** instead of learned live. Consumes `docs/predictions-plan.md`'s posterior draws.

**Unlike every layer before it, this one goes live this year.** The 2026-27 season is a
production run, not a research exercise — see "The deadline" below.

---

## What was settled on 2026-08-08

Six decisions, taken in a planning session against the economics derived in
`dashboard/economics.py` and the artifact survey recorded under "The prerequisite nobody
noticed". Each has a registry entry in `dashboard/decisions.py`.

| # | Decision | Why |
|---|---|---|
| 1 | **Two strategies, two tiers**: 10 entries at $20 (`600k_shootaround`) and 4 entries at $52 (`20k_spin_move`) | Near-equal stake ($200 / $208) across structures whose objectives differ in *shape*, not just scale. Comparing them is itself a result |
| 2 | **The test split is a pure readout** | The strategy is frozen on validation and written to an artifact; the test backtest is run once, reported, and changes **nothing** — not the strategy, not the stake, not the entry decision |
| 3 | **Reactive live-pick is the primary draft mode** | Fewer, higher-conviction entries drafted manually. Ranking-submission is the fallback for any fast draft that outruns the clock. This resolves a contradiction the doc used to carry in two places |
| 4 | **Draft state arrives through a local draft-room UI** | A Streamlit page over the precomputed board, one click per pick. Works at a 30-second clock and needs no external access. Reading the DK page is not on the critical path |
| 5 | **The in-draft objective is payout-weighted EV over the full bracket** | Simulate all four rounds against an ADP field and pick the player maximizing expected payout. It is the actual money objective, and only modestly more work once the bracket sim exists |
| 6 | **Strategy tuning runs on simulated truth with model-error injection** | The only surface with enough resolution to rank strategies — but the injection is mandatory, for the reason in "Why an uninjected simulated backtest cannot price ADP" |

---

## The deadline, and the ordering it forces

The DraftKings 2026-27 board was open and carrying ADP on **2026-07-28**, roughly 12 weeks
before the season. Drafts happen before October. That is the binding constraint on everything
below, and it inverts the natural build order: a **defensible draftable board and a working
draft room must exist before the full strategy sweep does**, not fall out of it.

Two production inputs were checked on 2026-08-08:

- ✅ **2026-27 rosters are live.** `commonteamroster` returns them (19 players for BOS on the
  probe, with `POSITION` null for unsigned/two-way players — the DK board covers that gap).
- ❌ **The 2026-27 regular-season schedule is not published.** `ScheduleLeagueV2` returns 20
  rows for 2026-27 — 19 preseason games plus the 12/11/2026 NBA Cup final that
  `docs/dk_best_ball_rules.md` says does not score — against 1,400 rows for 2025-26. The
  schedule is normally released in mid-August. **Poll for it**; the simulator cannot run a
  production season without per-team game dates.

Because the schedule is not yet available and the backtest seasons are, **every backtest piece
can be built now and the production run is schedule-gated at the very end.** That is the
ordering the to-do list follows.

---

## The prerequisite nobody noticed: no head persists its posterior

`make stan` fits eleven-plus heads and writes metrics, diagnostics and per-row predictions.
**It writes no coefficient draws.** The single exception is `stan_composition`'s crash
checkpoint, which pickles `alpha_draws` / `beta_draws` / `rho_draws` / `scaler` / `features`
per arm — incidentally, to survive a 9.9-hour loop, not as a consumable artifact.

So "simulate from the joint posterior" currently means *refit*: ~137 minutes for the component
heads and ~9.9 hours for the composition head, every run. That is not a thing a draft room can
do, and it is not a thing a strategy sweep can do a hundred times.

**Step zero of this layer is therefore a posterior artifact**, and it is a small module rather
than a research question. One file per head carrying:

- **thinned coefficient draws** (`alpha_draws`, `beta_draws`, `rho_draws`, dispersion), 1,000
  draws by default, thinned across the whole posterior via `stan_utils.thin` — never sliced off
  the front, for the reason `season_terms._draw_components` already records;
- the **design recipe** needed to score an arbitrary frame: feature list, the fitted
  `StandardScaler`, spline knots, imputation means, and the selected variant name;
- **provenance**: the fit window (`train`, `train_val`, or `full`), the CmdStan version, the
  git SHA, and the timestamp.

`stan_composition._checkpoint` is the precedent for the mechanics; this generalizes it to every
head, promotes it from a crash artifact to a contract, and gives it a loader. Once it exists,
`make posteriors` is the only thing that ever needs a CmdStan toolchain, and everything
downstream is numpy.

**This also unblocks the walk-forward question.** Fitting one posterior per fit window is what
makes it affordable to widen the realized backtest later without re-deciding anything.

---

## The output contract that collapses the compute problem

**The drafting layer never sees a player-game.** Best ball scores by *scoring period* — the
best 7 of your 16 by slot, summed over the games in that week — so the simulator's deliverable
to everything downstream is a single tensor:

```
sim_tensor[player, scoring_period, sim]  ->  dk_pts     float32
```

Round 1 is 17 weeks and Rounds 2–4 are one double week each, so there are **20 scoring
periods**. At ~550 draftable players and 2,000 sims that is 550 × 20 × 2,000 × 4 bytes ≈
**88 MB** — small enough to hold in memory for the whole strategy sweep and small enough to
load into a draft room in under a second.

Per-game draws still happen, because the double-double bonus is a per-game threshold on five
components simultaneously and `E[bonus] ≠ bonus(E[x])`. They happen *inside* the simulator and
are summed into periods immediately. Nothing outside the simulator ever materializes a
`player × game × sim` array.

**Fixing this contract is the difference between a draft sweep that runs in minutes and one
that runs in hours**, and it is what makes the sub-second in-draft recompute achievable.

A second, smaller tensor rides alongside it and must not be forgotten:

```
availability[player, scoring_period, sim]  ->  games played that period    uint8
```

Because a zero-game week is not the same as a bad week for lineup selection, and because the
frozen-roster risk in Rounds 2–4 is entirely a games-played story.

---

## Contest mechanics — ground truth is `docs/dk_best_ball_rules.md`

- **Roster**: 16 players, at least 2 NBA teams.
- **Weekly lineup**: 7 starters auto-selected as the highest scorers by eligible slot — 2 G,
  2 F, 1 C, 2 UTIL (G/F/C) — 9 bench, bench points do not count. The scoring period is the date
  of the first game through the last game of that week's game set; a rescheduled or suspended
  game counts for the period it is **played** in.
- **Scoring**: matches `compute_dk_pts` exactly. Verified against the rules doc; **no scoring
  changes needed.**
- **Draft**: snake, order randomized once the lobby fills, 16 rounds, one live pod of **12
  entries** in Round 1 for every tournament.
- **Auto-draft**: queue first, then pre-draft ranking, with default caps of **8 G / 8 F / 3 C**
  unless overridden, plus an exclusion list. Those caps are DK's own defaults — `nba_stats`'s
  `make_pick` re-implemented them rather than inventing them, and the opponent model should use
  them as given.
- **Tournament**: 4 rounds, one frozen roster throughout, **no redraft**. Round 1 is 17 weeks
  (10/20–2/14) cumulative; Rounds 2–4 are each one double week (2/15–3/7, 3/8–3/21, 3/22–4/4).
  The NBA Cup championship game (12/11/2026) does not score. Ties break on best single week,
  cascading down through the round's weeks, then on best individual player score, cascading
  the same way.

### Scoring periods are NBA weeks — derive, do not hand-enter

`ScheduleLeagueV2` carries `weekNumber` and `weekName` natively. Checked on 2025-26: the week
numbering runs Monday–Sunday, partitions game dates with **zero** dates in more than one week,
and Week 17 closes 2026-02-12 against DK's stated Round-1 close of 2/14. DK's rounds are NBA
week ranges.

So the bucketing module reads the NBA week grid and maps DK's published round windows onto it,
rather than re-deriving weeks from raw dates. It must still own three edge cases once, not
per-use: the played-not-scheduled rule for postponed games, the NBA Cup final exclusion, and
the all-star gap (2025-26's Week 17 ends 2/12 and Week 18 opens 2/19).

---

## The two target tournaments

Derived live by `dashboard/economics.py` from `data/raw/dk_best_ball_tournament_*.csv`.

| | `600k_shootaround` | `20k_spin_move` |
|---|---|---|
| entry fee | $20 | $52 |
| **entries this year** | **10** ($200) | **4** ($208) |
| max per player | 150 | 12 |
| field | 35,280 | 432 |
| rake / break-even hurdle | 14.97% / **+17.60%** | 10.97% / **+12.32%** |
| R1 → R2 | 2 of 12 | 2 of 12 |
| R2 → R3 | 1 of 12, 11 cash ≥$30 | 2 of 6, 4 cash ≥$80 |
| R3 → R4 | 1 of 10, 9 cash ≥$100 | 2 of 6, 4 cash ≥$250 |
| R4 | 49 paid, min $1,000 | 8 paid, min $750 |
| first prize | $200,000 = **10,000×** | $5,000 = **96×** |
| P(reach R4) at random | 0.139% | 1.852% |
| E[entries reaching R4] | 0.014 | 0.074 |

**Round 1 is the only zero-consolation round, and that reframes the objective.** Surviving it
guarantees a cash in both tournaments — `600k_shootaround` pays 11 of 12 R2 entries at $30
minimum on a $20 entry, and `20k_spin_move` pays or advances all 6 at $80 minimum on a $52
entry. So `P(any return) = P(top 2 of 12)` exactly, and everything past Round 1 sets the *size*
of the return rather than its sign. The earlier framing of this doc — "convex, therefore chase
the tail" — is right only for `600k_shootaround`, and only above the R2 floor.

**The two tiers differ in what strategy should buy.** `20k_spin_move`'s path is three
successive shallow cuts (2/12 → 2/6 → 2/6) into a nearly flat final table, so it rewards
**survival probability** and durability. `600k_shootaround`'s path narrows brutally after
Round 1 (2/12 → 1/12 → 1/10) into a 10,000× top prize, so above the R2 floor it rewards
**correlated upside and differentiation from the field**. Confirming that the sweep actually
selects different rosters for the two is Gate D.

**Field quality is unmeasured and is a live risk.** A $1 field plausibly holds far more
autodraft entries than a $450 one, in the opposite direction from the rake math. The opponent
model must let field composition vary by tier rather than assume one field everywhere, even
though nothing calibrates it today.

---

## Architecture

Five modules, each writing one artifact, in dependency order. Every one is numpy over the
posterior artifact — **only `make posteriors` needs CmdStan.**

```
make posteriors      src/models/posteriors.py      thinned draws + design recipe per head
                                                   -> data/features/posteriors/<head>.pkl

make scoring-periods src/features/scoring_periods.py   NBA week grid -> DK round windows
                                                   -> data/features/scoring_periods.parquet

make draft-pool      src/features/draft_pool.py    the board: player x season, DK position
                                                   eligibility, team, ADP, prior-season row
                                                   -> data/features/draft_pool.parquet

make simulate-season src/sim/season.py             THE tensor
                                                   -> data/features/sim_tensor_<season>.npz

make draft-sim       src/sim/draft.py              snake draft vs an ADP field
make bracket         src/sim/bracket.py            4 rounds, advancement, ties, payouts
make strategy-sweep  src/sim/strategy.py           the sweep -> outputs/predictions/strategy_*.csv
make draft-room      dashboard/draft_room.py       the live recommender
```

`src/sim/` is a new package, parallel to `src/models/` and `src/eda/`, because these are
neither models nor analyses. Conventions carry over unchanged: `python -m src.sim.<module>`
entry points, the duplicated `yaml.safe_load` idiom in `__main__`, `mkdir(parents=True,
exist_ok=True)` before every write, `f"... {n:,} ... → {dest}"` progress lines.

### `src/sim/season.py` — the season simulator

Assembly, not invention. Every piece already exists and is measured:

| Step | Source | Number |
|---|---|---|
| which games he plays | `stan_games_played.sequences` | entry × exit × within-tenure chain, all gates pass |
| minutes, team-constrained | `stan_composition.simulate_minutes` | CRPS 4.4945 vs 4.7842 independent |
| minutes, game-level noise | `stan_minutes_dispersion.csv` | **4.65×** binomial |
| minutes, serial dependence | `serial_correlation.csv` | **2.43×** ten-game block inflation |
| the eleven component heads | `season_terms._draw_components` | already materializes `fga → fg3a\|fga → fg2a → makes` |
| cross-component dependence | `residual_correlation.csv` | Gaussian copula, mean +0.013, max 0.157 |
| the bonus | `targets.expected_bonus` / `compute_dk_pts` | overdispersion **0.025** at the player-game unit |

Four rules the assembly must not violate, all of them already argued elsewhere and repeated
here because this is the module that could quietly break them:

1. **Draw, never plug in.** `E[min]` and `E[gp]` are wrong inputs to a threshold bonus.
2. **One shared `min` draw per player-game** feeds all eleven heads as exposure. Minutes are
   46.4% of within-player residual variance; this is where the correlation comes from.
3. **Sequential structure goes on minutes and nowhere else.** Both conversion heads are
   measured nulls for a hot hand.
4. **One posterior draw moves the whole board.** Players share `β`, so the sim index must be
   the *outer* loop over posterior draws — not resampled per player. That shared-`β` sweep is
   the cross-player correlation this layer wants, and drawing it per player destroys it.

Consumers default to the `train_val` fit window, matching the four numbers already calibrated
that way.

### `src/sim/draft.py` — the draft simulator

A snake draft over 12 entries and 16 rounds. Opponents autodraft off recalibrated-DK ADP with
rank noise, subject to DK's real 8G/8F/3C caps. Two modes over one engine:

- **reactive** (primary) — a pick function sees the current board state and returns a ranked
  recommendation. This is what the draft room calls and what the strategy sweep exercises.
- **ranking-submission** (fallback) — a static pre-draft ranking plus position limits and an
  exclusion list, executed by DK's documented autodraft logic. Needed because a 30-second
  clock can outrun a human, and needed for the opponent model regardless.

**The field model has a real calibration target, which the plan previously assumed it did
not.** Observed ADP *is* the field's realized aggregate behaviour, so simulating many drafts
and measuring the resulting average draft position must reproduce the observed ADP curve. That
is Gate B, and it is what "calibrated to reproduce ADP" means concretely.

### `src/sim/bracket.py` — rounds, advancement, ties, payouts

Reads the tournament spec from the two CSVs via `dashboard.economics` — round count, pod size,
advance count and cash table — so pointing this at the real 2026-27 numbers is a data change.
It must get three things exactly right:

- **the cascading tie-break**, because the 2-vs-3 boundary in a 12-entry pod over 17 weeks will
  be close often, and the rules are explicit: best single week, then second-best, down through
  the round, then best individual player score, cascading the same way;
- **wildcards**, which fill any shortfall from the highest-scoring non-advancing entries;
- **the selected survivor field in Rounds 2–4.** Opponents there are not an ADP field — they
  are the population that already cleared a 2-of-12 cut. Simulating the whole bracket gets this
  for free; scoring rounds independently against a fresh ADP field would systematically
  overstate continuation value.

### `src/sim/strategy.py` — the sweep

A strategy is a config object, so exploration is a table rather than a rewrite:

```yaml
ranking:      model_mean | model_quantile:<q> | adp | blend
alpha:        scalar, or per-round (rounds 1-2 vs 9+ — see below)
position_caps: {G: 8, F: 8, C: 3}          # DK defaults; overridable
exposure_caps: per-player share across the portfolio's entries
stacking:     same-team pair bonus, 0 = off
objective:    bracket_ev | p_advance | expected_score
n_entries:    10 (600k_shootaround) | 4 (20k_spin_move)
```

**Select on P(advance) lift, report ROI.** The two are not equally measurable. ROI is dominated
by rare deep runs — `600k_shootaround` reaches Round 4 on 0.139% of entries — so its Monte
Carlo error is enormous. `P(top 2 of 12)` is a 16.67% event and resolves orders of magnitude
faster on the same simulation budget. Since surviving Round 1 is exactly the condition for any
return at all, the lift in `P(top 2 of 12)` over an ADP-drafted entry is both the statistic the
sweep can resolve and a defensible headline. ROI against the break-even hurdle is reported
alongside it, with its interval.

`docs/adp-plan.md` binds two things on the blend: use the **DK-recalibrated** consensus, not a
raw one (a monotone recalibration cuts cross-validated error from 24.0 to 17.0 picks, and DK
drafts centers 11.9 picks earlier because category-league ADP discounts them for FT%), and
expect `α` to want to vary by round, since disagreement is 5.1 picks in rounds 1–2 against 30.8
in rounds 9+ — which is where 9 of the 16 roster spots are filled.

---

## The backtest, and why its two halves do different jobs

### Simulated truth — the tuning surface

Draw a season from the posterior, call it truth, draft against it, score the bracket. Unlimited
resolution, and the only surface with enough power to separate strategies.

**Why an uninjected simulated backtest cannot price ADP.** In a world drawn from the model's own
posterior, the model is perfectly calibrated by construction. ADP can then only add noise, so
the sweep will drive `α → 0` for reasons that have nothing to do with whether the market knows
something. The same failure hits every strategy that hedges model error: exposure caps,
differentiation, shrinkage toward consensus.

So the truth draw is **perturbed to reproduce the model's measured out-of-sample miss** before
anything is scored against it. The calibration targets are already on disk: the availability
head's validation CRPS (10.006 games), the component heads' validation R² against their no-fit
floors (0.81–0.95), and the season-total MAE (400.5 dk_pts). The injected world must reproduce
those, not the model's in-sample calibration. **An uninjected sweep is not a conservative
version of this — it is a sweep that answers a different question**, and its α is not
transportable.

This is the one genuinely new piece of statistical machinery in the layer, and it deserves its
own gate (Gate C).

### Realized truth — the honest readout

Replay simulated portfolios against real box scores. Two seasons: **2022-23 and 2023-24**, the
validation split, both of which happen to carry FantasyPros ADP.

N = 2 seasons of correlated pods will not distinguish `α = 0.3` from `α = 0.5`, and the plan
should not pretend otherwise. Its job is to catch a strategy that is broken in a way the
simulated world cannot see, and to put an honest — wide — interval on the measured edge.

**A cheap widening exists if it turns out to be needed.** ADP also covers 2014-15, 2017-18,
2018-19 and 2019-20. The fitted heads are in-sample on those, but the **no-fit carry-forward
floor is out-of-sample by construction** and scores only 0.001–0.03 R² below the fitted heads.
A floor-ranked backtest across six seasons is a legitimate robustness check on strategy
*shape*, even though it cannot price the fitted model's edge. Build it only if the two-season
result is ambiguous.

### The test split — a pure readout, mechanized as one

Settled 2026-08-08: the test seasons (2024-25, 2025-26) get **one** backtest run before going
live, and it changes **nothing**. Not the strategy, not the stake, not the entry decision.

The rule is enforced the way `src/models/held_out.py` enforces it for the model heads, because
prose already failed once here:

- the strategy sweep goes through `selection_split` and never materializes the test rows;
- the shipped strategy is written to an artifact by the validation sweep, and the test runner
  **reads which strategy shipped rather than re-deciding**, exactly as `src/final_evaluation.py`
  does;
- the test runner is a **report generator** — ROI distribution, P(advance), P(cash), worst-case
  drawdown across the 10 + 4 entries — and emits no ranking, no selection and no recommendation;
- it runs inside `held_out.unlocked("pre-season risk readout")` so the unlock is visible in the
  log.

---

## The live draft room

`dashboard/draft_room.py`, a Streamlit page separate from the walkthrough app. It loads the
precomputed sim tensor and the draft pool, shows the board, and takes **one click per pick** to
mark a player gone. No external access, no OCR, no page scraping on the critical path.

**Latency is the design constraint and it should be treated as a gate.** A 30-second fast-draft
clock means a recompute budget well under a second. That is achievable because the expensive
work is precomputed: the in-draft calculation is the marginal bracket EV of adding each
remaining player to the current roster, which is array math over a `16 × 20 × n_sims` slice.
Two things make it fit:

- **drop to `n_sims = 500` in-draft** and keep 2,000 for the sweep. The in-draft decision is a
  ranking of candidates, not an estimate of a level;
- **best-7-by-slot is a partial sort**, not a full one — never recompute the whole lineup
  selection when one player is added.

If the budget cannot be met, the fallback is a precomputed static ranking with an exclusion
list, which is the ranking-submission mode already being built.

Reading the DK draft page directly — via the browser extension or a pasted pick log — is worth
exploring for 8-hour slow drafts, where the clock is not the constraint. It is explicitly not
on the critical path and should not gate the draft room shipping.

---

## Data the layer needs, and where it comes from

- **DK position eligibility** — `data/raw/team_rosters_*.csv` carries `POSITION` with DK-shaped
  dual eligibility (`G-F`, `F-C`, `C-F`, `F-G`) for all 30 seasons with **zero** nulls on the
  historical files. Without it no lineup can be filled at all. The two
  `data/raw/dk_draft_rankings/*.csv` boards carry DK's *own* positions for 698 and 942 players,
  which is what the NBA.com → DK mapping is validated against. 2026-27 rosters carry nulls for
  unsigned players; the DK board covers them.
- **ADP** — `data/features/adp_panel.parquet`, 15,012 rows. Coverage is 2014-15, 2017-18,
  2018-19, 2019-20, 2022-23, 2023-24, 2024-25, 2025-26 (both sources) and 2026-27 (DK only).
  Both validation seasons are covered, which is the coverage that matters.
- **Schedule** — realized game dates from `data/processed/game_logs.parquet` for backtests;
  `ScheduleLeagueV2` for the production season, **once it is published**.
- **Tournament structure** — the two `dk_best_ball_tournament_*.csv` files, through
  `dashboard.economics`. Re-verify against the live 2026-27 contests before any backtest
  number is treated as load-bearing.

---

## Gates

Every gate is judged on validation or on simulated truth. None reads the test split.

| Gate | Pass condition | Why this bar |
|---|---|---|
| **A** | the season simulator reproduces the **marginals it was built from**: season-total dk_pts distribution against `season_total_metrics.csv`, GP pmf against `stan_games_played_gp_pmf.csv`, and per-game bonus rate against `bonus_calibration.csv` | An assembly bug is silent. Every input head is already calibrated, so a simulator that misses a marginal it was handed has a wiring fault, not a modelling one |
| **B** | simulated drafts reproduce the **observed ADP curve** — mean absolute rank gap under the 17.0-pick recalibration error, so the field model is no worse than the market proxy it consumes | The field model's only real calibration target. Failing it means the opponent model is not a field |
| **C** | the **error-injected** simulated world reproduces the model's measured out-of-sample miss: availability CRPS ≈ 10.006 games, component R² in 0.81–0.95, season-total MAE ≈ 400.5 dk_pts | Without this the sweep cannot price ADP, exposure caps, or any other hedge against model error |
| **D** | the sweep selects **materially different** rosters for the two tiers | If the $20 and $52 strategies converge, either the objective is not doing its job or the tier difference is smaller than the economics imply. Either way it needs to be known before entering |
| **E** | in-draft recompute **under 1.0 s** at `n_sims = 500` on the full remaining pool | The 30-second clock. Failing it drops the draft room to ranking-submission mode |
| **F** | measured edge, expressed as **lift in P(top 2 of 12)** over an ADP-drafted entry, is reported with a bootstrap interval against the break-even hurdle (+17.60% / +12.32%) | Not a pass/fail on the edge itself — a requirement that the number is quoted in comparable units with its uncertainty, rather than as a point estimate |

---

## Config

```yaml
sim:
  posterior_draws: 1000          # thinned across the whole posterior, never sliced
  fit_window: train_val          # matches the four already-calibrated simulator inputs
  n_sims: 2000                   # strategy sweep
  n_sims_draft: 500              # in-draft; a ranking, not a level
  scoring_periods: 20            # R1's 17 weeks + three double weeks
  pod_size: 12
  seed: 0

  tournaments:
    600k_shootaround: {entries: 10}
    20k_spin_move:    {entries: 4}

  field:
    adp_source: dk_recalibrated  # docs/adp-plan.md: never the raw consensus
    rank_noise_sd: null          # fitted by Gate B, not chosen
    position_caps: {G: 8, F: 8, C: 3}   # DK's own autodraft defaults

  error_injection:
    enabled: true                # Gate C. Disabling it changes the question being asked
    targets: [availability_crps, component_r2, season_total_mae]
```

---

## Tests

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.

- **lineup selection** — best 7 by slot from a hand-built 16 with known scores, including the
  UTIL fallback and a dual-eligibility player who must be placed to maximize the total rather
  than greedily.
- **tie-breaks** — two entries with identical round totals and different weekly maxima, then
  identical weekly maxima and different best-player scores, cascading.
- **bracket arithmetic** — the advance chain must reproduce `economics.advance_table` field
  sizes exactly for all five tournaments.
- **scoring-period bucketing** — a postponed game scores in the period it is played; the NBA Cup
  final scores nowhere; every game date lands in exactly one period.
- **the split guard** — the strategy sweep raises if it reaches the test seasons, pinned the way
  `component_rates`' guard is pinned.
- **posterior round-trip** — a head's saved draws and design recipe reproduce its stored
  predictions to numerical tolerance, without refitting.
- **draw order** — `fga → fg3a|fga → fg2a → makes` is materialized in that order; reordering it
  must fail loudly.

---

## Open questions and risks

- **The 2026-27 schedule is not published.** Blocks the production run only. Poll `ScheduleLeagueV2`.
- **Field skill by tournament tier is unmeasured**, and will bias any opponent model that
  assumes one field composition across a $20 and a $52 contest.
- **The DK ADP capture deadline is live** — an early-to-mid October 2026 board is the second
  anchor the recalibration needs, and it cannot be backfilled. See `docs/adp-plan.md`.
- **No-redraft risk is not in the ranking.** A Round-1 pick who is traded or suffers a
  season-ending injury is frozen dead weight through Rounds 2–4. The simulator captures this if
  the spell process runs the full season, but whether the *ranking* should carry an explicit
  durability term distinct from expected value is open.
- **Mid-season trades are not modelled** — 13.6% of players appeared for 2+ teams in 2023-24.
  A known limit inherited from the prediction layer.
- **N = 2 realized seasons** is the ceiling on the honest edge estimate. Named, not solved.
- **Tournament structures may change** for the live 2026-27 contests. Re-verify the metadata
  and prize CSVs before treating any backtest result as load-bearing.
- **This doc is not yet in `make docs-audit`.** Add it once it carries measured figures rather
  than specification — see `docs/docs-audit.md`.

---

## The build order — one session per item, with its opening prompt

Ordered so the **live-draft path closes at item 7**. Items 2 and 3 depend on nothing and can
run in any order alongside item 1. Every prompt assumes the reader starts by reading
`CLAUDE.md`, `docs/project-spec.md` and this doc, which is why none of them repeat the
conventions.

### 1. `make posteriors` — persist the fitted posteriors ⛔ blocks 4, 6, 7, 8

> Read `CLAUDE.md`, `docs/project-spec.md` and `docs/simulations-plan.md` (the section "The
> prerequisite nobody noticed"). Build `src/models/posteriors.py` + `make posteriors`, which
> loads each fitted Stan head and writes one artifact per head to
> `data/features/posteriors/<head>.pkl`: thinned coefficient draws (1,000, via
> `stan_utils.thin`, spread across the whole posterior — never sliced off the front), the design
> recipe needed to score an arbitrary frame (feature list, fitted scaler, spline knots,
> imputation means, selected variant), and provenance (fit window, CmdStan version, git SHA,
> timestamp). `stan_composition._checkpoint` is the precedent for the mechanics — generalize it
> from a crash artifact to a contract and give it a loader. Cover every head in `make stan` plus
> `stan-games-played`. Add a round-trip test: saved draws + recipe reproduce that head's stored
> predictions to numerical tolerance **without refitting**. Default the fit window to
> `train_val`. This is the one target that needs CmdStan; everything downstream is numpy.

### 2. `make scoring-periods` — the DK week grid

> Read `docs/simulations-plan.md` ("Scoring periods are NBA weeks") and
> `docs/dk_best_ball_rules.md`. Build `src/features/scoring_periods.py` + `make scoring-periods`
> writing `data/features/scoring_periods.parquet`: one row per (season, game_id) with its
> scoring period and tournament round. Read the NBA week grid from
> `nba_api.stats.endpoints.ScheduleLeagueV2` (`weekNumber` / `weekName`) for seasons that have
> it and derive it from realized `game_date` in `data/processed/game_logs.parquet` otherwise;
> map DK's published round windows onto that grid. Own three edge cases once: a postponed game
> scores in the period it is **played**, the NBA Cup final does not score, and the all-star gap
> breaks week adjacency. Assert every game date lands in exactly one period. Verify Round 1 = 17
> weeks and Rounds 2–4 = one double week each. Flip the `scoring-periods-are-nba-weeks` registry
> entry from `open` to `built`.

### 3. `make draft-pool` — the board, with DK position eligibility

> Read `docs/simulations-plan.md` ("Data the layer needs") and `docs/adp-plan.md`. Build
> `src/features/draft_pool.py` + `make draft-pool` writing `data/features/draft_pool.parquet`:
> one row per (season, player) with team, **DK position eligibility** (G / F / C flags, duals
> allowed), ADP from `adp_panel.parquet`, and the prior-season key the model heads need. Take
> positions from `data/raw/team_rosters_*.csv` (`POSITION`, 30 seasons, zero nulls, already
> DK-shaped duals) and **validate the NBA.com → DK mapping** against the two
> `data/raw/dk_draft_rankings/*.csv` boards, which carry DK's own positions for 698 and 942
> players — report the disagreement rate rather than assuming the mapping. Fall back to the DK
> board for the 2026-27 rows whose `POSITION` is null (unsigned / two-way). Reuse
> `adp_dk_id_map.parquet` for the id join; never name-match where an id exists.

### 4. `make simulate-season` — the tensor, and Gate A

> Read `docs/simulations-plan.md` ("The output contract", "`src/sim/season.py`") and
> `docs/predictions-plan.md`. Create the `src/sim/` package and build `src/sim/season.py` +
> `make simulate-season`, writing `data/features/sim_tensor_<season>.npz`: a
> `player x scoring_period x sim` float32 tensor of dk_pts plus a `uint8` games-played twin.
> Assemble, do not invent — `stan_games_played.sequences` for which games, `stan_composition
> .simulate_minutes` plus the 4.65x game-level dispersion and 2.43x block inflation for minutes,
> `season_terms._draw_components` for the eleven heads in the order `fga → fg3a|fga → fg2a →
> makes`, the conditioned matrix from `residual_correlation.csv` as the copula, and
> `compute_dk_pts` for scoring. Four rules that must not be violated: draw never plug in; one
> shared `min` draw per player-game feeds all eleven heads; sequential structure goes on minutes
> only; **the posterior draw is the outer loop**, shared across all players, because that shared
> `β` is the cross-player correlation this layer exists for. **Gate A**: the simulator must
> reproduce the marginals it was handed — season-total dk_pts against `season_total_metrics.csv`,
> the GP pmf against `stan_games_played_gp_pmf.csv`, per-game bonus rate against
> `bonus_calibration.csv`. A miss here is a wiring fault, not a modelling one.

### 5. `make bracket` — lineups, ties, advancement, payouts

> Read `docs/simulations-plan.md` ("`src/sim/bracket.py`", "Tests") and
> `docs/dk_best_ball_rules.md`. Build `src/sim/bracket.py`: best-7-of-16 by slot per scoring
> period (2 G / 2 F / 1 C / 2 UTIL, dual eligibility placed to maximize the total rather than
> greedily), Round-1 to Round-4 advancement, wildcards, and the **cascading tie-break** — best
> single week, then second-best down through the round, then best individual player score,
> cascading the same way. Read every structural number (round count, pod size, advance count,
> cash table) from `dashboard.economics`, never hardcoded. The Round 2–4 field must be the
> **selected survivor population**, not a fresh ADP field; simulating the whole bracket gets
> this for free and scoring rounds independently would overstate continuation value. Tests:
> hand-built lineups with known scores, both tie-break cascades, and the advance chain
> reproducing `economics.advance_table` field sizes exactly for all five tournaments.

### 6. `make draft-sim` — the snake draft and the ADP field, and Gate B

> Read `docs/simulations-plan.md` ("`src/sim/draft.py`") and `docs/adp-plan.md`. Build
> `src/sim/draft.py`: a 12-entry, 16-round snake draft over one engine with two modes —
> **reactive** (a pick function sees board state and returns a ranked recommendation; this is
> primary) and **ranking-submission** (static ranking + position limits + exclusion list,
> executed by DK's documented autodraft logic: queue → ranking → 8G/8F/3C caps). Opponents
> autodraft off the **DK-recalibrated** consensus from `adp_transfer.parquet` — never the raw
> consensus — with rank noise. **Gate B**: simulate many drafts, measure the resulting average
> draft position, and require it to reproduce the observed ADP curve within the 17.0-pick
> recalibration error. Fit `rank_noise_sd` to that target rather than choosing it. Let field
> composition vary by tournament tier even though nothing calibrates that yet.

### 7. `dashboard/draft_room.py` — the live recommender, and Gate E 🎯 live-draft ready

> Read `docs/simulations-plan.md` ("The live draft room"). Build a Streamlit page separate from
> the walkthrough app: load the precomputed sim tensor and draft pool, show the board, take
> **one click per pick** to mark a player gone, and return a ranked recommendation by marginal
> bracket EV. **Gate E is a hard latency bar: under 1.0 s per recompute on the full remaining
> pool.** Two things make it fit — drop to `n_sims = 500` in-draft (the decision is a ranking,
> not a level) and use a partial sort for best-7-by-slot rather than recomputing the whole
> lineup selection per candidate. Measure and report the actual latency. If the bar cannot be
> met, fall back to exporting a static ranking + exclusion list in DK's pre-draft-rankings CSV
> format, which is the ranking-submission mode item 6 already built.

### 8. `make strategy-sweep` — the sweep, error injection, and Gates C and D

> Read `docs/simulations-plan.md` ("`src/sim/strategy.py`", "The backtest"). Build
> `src/sim/strategy.py` + `make strategy-sweep`. A strategy is a config object (ranking source,
> `α` overall and per-round, position caps, exposure caps, stacking, objective, entry count), so
> the sweep is a table. **Gate C first, because it gates the sweep's validity**: perturb the
> simulated truth to reproduce the model's *measured* out-of-sample miss — availability CRPS
> 10.006 games, component R² 0.81–0.95 against the no-fit floors, season-total MAE 400.5 dk_pts
> — and verify it does. Without it the sweep drives `α → 0` for reasons that have nothing to do
> with the market. **Select on lift in P(top 2 of 12), report ROI** against the break-even
> hurdle with a bootstrap interval; ROI alone is dominated by 0.139%-probability deep runs and
> will not resolve. Sweep both tiers (10 entries at $20, 4 at $52). **Gate D**: confirm the two
> tiers select materially different rosters. Go through `selection_split` — the sweep must
> raise if it reaches the test seasons, pinned by a test the way `component_rates`' guard is.
> Write the shipped strategy to an artifact. Then replay against realized 2022-23 / 2023-24 and
> report it with honest, wide intervals as a readout, not a selector.

### 9. Production run for 2026-27 ⏳ schedule-gated

> Read `docs/simulations-plan.md` ("The deadline"). Poll
> `nba_api.stats.endpoints.ScheduleLeagueV2(season="2026-27")` — as of 2026-08-08 it returns 20
> rows (19 preseason plus the 12/11/2026 Cup final) against 1,400 for 2025-26, so the regular
> season is not published. Once it is: refresh `make fetch` for 2026-27 rosters and schedule,
> refit the heads on all 30 seasons, rebuild the posterior artifacts at the full window, build
> the 2026-27 draft pool and sim tensor, and produce the board. Flip
> `production-schedule-not-published` from `blocked` to resolved.

### 10. The test-split risk readout — once, before going live

> Read `docs/simulations-plan.md` ("The test split") and `src/final_evaluation.py`. Extend the
> final evaluation to backtest the **already-shipped** strategy on 2024-25 and 2025-26 inside
> `held_out.unlocked("pre-season risk readout")`. It must read which strategy shipped from the
> artifact rather than re-deciding, and emit only a risk report — ROI distribution, P(advance),
> P(cash), worst-case drawdown across the 10 + 4 entries. **It changes nothing**: not the
> strategy, not the stake, not the entry decision. Run it once.

### 11. Dashboard and docs refresh

> Read `docs/dashboard-plan.md`, `docs/provenance-plan.md` and `docs/docs-audit.md`. Repoint
> `dashboard/tabs/simulations.py::CHAIN` and `dashboard/tabs/drafting.py` at the real artifacts
> — both currently render "not built" warnings. Flip the 2026-08-08 registry entries from
> `settled` / `open` to `built` with their artifacts in `reproduce`. Update README §2's
> "Simulation — planned" and "Ranking, drafting and tournaments — planned" sections and §3
> Results with the measured figures. **Add `docs/simulations-plan.md` to `src/docs_audit.py`**
> now that it carries measured figures rather than specification. Run `make docs-audit` and
> `make dashboard-audit`.

## Doc and registry obligations

Six decisions from 2026-08-08 are registered in `dashboard/decisions.py` under topics
`simulations` and `drafting`. As pieces get built: flip each entry's `status` from `settled` to
`built` with its artifact in `reproduce`, update this doc in place, and repoint
`dashboard/tabs/simulations.py::CHAIN` and `dashboard/tabs/drafting.py` at the real artifacts —
both tabs currently render "not built" warnings that will otherwise go stale.
