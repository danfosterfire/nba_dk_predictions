# Simulations Plan: Drafting and Scoring Best-Ball Portfolios

This is a planning doc, not a measurement report. It records direction, not results — update it
in place as pieces get built, the way `availability-plan.md` was.

## Purpose

Closes the loop `~/Documents/nba_stats` never finished: a draft simulator, a season/tournament
outcome simulator, and a backtesting harness, so draft strategies (ADP-blended ranking,
exposure caps, posterior-sampled diversification, stacking) can be **compared on real historical
seasons** instead of learned live. Consumes `docs/predictions-plan.md`'s posterior draws.
Strategy tuning itself (payout shape, exposure caps, stacking intensity) is explicitly staged
*after* this machinery works — that ordering is deliberate, not an oversight.

## Contest mechanics — ground truth is `docs/dk_best_ball_rules.md`

- **Roster**: 16 players total, must include players from at least 2 different NBA teams.
- **Weekly lineup**: 7 starters auto-selected from your 16 as the highest scorers by eligible
  slot — 2 G, 2 F, 1 C, 2 UTIL (G/F/C) — 9 bench, bench points don't count. The scoring period
  each week is the date of the first game through the last game of that week's game set; a
  rescheduled or suspended game counts for the period it's actually **played** in, not the one
  originally scheduled — matters for replaying real historical schedules in a backtest.
- **Scoring**: matches `compute_dk_pts` exactly — 1 pt/point, +0.5 bonus per made 3, 1.25/reb,
  1.5/ast, 2/stl, 2/blk, −0.5/tov, +1.5 DD, +3 TD. Verified against the rules doc; **no scoring
  changes needed.**
- **Draft**: snake, draft order randomized once the lobby fills, 16 rounds, one live draft pod
  of **12 entries**. Fast drafts (30 s clock, minutes to hours) or slow drafts (8 h clock,
  days) — not all tournaments/leagues are fast; the user plans to use slow drafts to draft
  manually rather than lean on autodraft. Overall field size and per-round advance counts vary
  by tournament — see "Five real tournament examples" below, which now grounds this in actual
  numbers rather than a placeholder.
- **Auto-draft mechanics — the key simplification for simulator design.** A 30-second fast-draft
  clock makes live manual play across ~100 entries impractical. In practice, executing a
  strategy at that scale means submitting a **personalized pre-draft ranking** (plus optional
  position-limit overrides and an exclusion list) and letting DK's autodraft engine execute it:
  queue first, then pre-draft-ranking order, with default caps of **8 G / 8 F / 3 C** unless
  overridden. This is exactly what `nba_stats/run_tournaments.R`'s `make_pick` re-implemented —
  its 8/8/3 caps are DK's own defaults, not an independent heuristic. **Implication: the primary
  simulator mode should be "submit a ranking + limits," not "react pick-by-pick."** A reactive
  live-pick mode is worth building too, for slow drafts or small-scale manual play, but is
  secondary.
- **Player pool**: active-roster players as of contest creation; trades/signings update team
  eligibility.
- **Tournament structure**: 4 rounds, one frozen roster throughout — **no redraft between
  rounds**. Round 1 is 17 weeks (10/20–2/14), cumulative. Rounds 2–4 are each one "double week"
  (2/15–3/7, 3/8–3/21, 3/22–4/4). The NBA Cup championship game (12/11/2026) doesn't score. A
  specified number of entrants advance per contest per round (numbers TBD, vary by tournament);
  wildcards fill any shortfall; ties break on best single week, cascading down through the
  round's weeks, then on best individual player score, cascading the same way.
- **Payout**: magnitude and curve unknown right now. Treated as a lever below, not hardcoded.

## What the prediction layer now guarantees this one — settled 2026-07-29

Four results from the modeling side change what this plan has to build, and all four reduce it:

- **Correlated draws do not need a joint model fit.** The heads are fitted separately (the
  posterior factorizes exactly — see `CLAUDE.md`), and the correlation this layer needs arrives
  at *draw* time from two cheap sources: a **shared `min` draw** per player-game pushed through
  all eleven component heads as exposure, and — only if that is not enough — a **Gaussian copula**
  on residuals whose off-diagonals are measured at mean **+0.013**, max **0.157** (`fg2a`–`reb`),
  with the 3PA/2PA substitution at **−0.110**. So "sample one shared team/player state, then draw
  conditionally" was the right instinct below, and it is now the *only* mechanism needed rather
  than one of two candidates.
- **Parameter uncertainty is the cross-player correlation source this plan actually wants.**
  All players share `β`, so a posterior draw shifts *every* player together. That answers "how
  wrong could my whole board be at once", which under a zero-consolation Round-1 knockout is a
  different and more important question than any single player's marginal. This is why the
  availability head is being refitted in Stan (`docs/availability-plan.md`) — not for CRPS.
- **Minutes carries the sequential structure, and it is where simulated season variance is
  won or lost.** `make serial-correlation`: 10-game block variance inflation is **2.43×** for
  `min` against 1.01–1.11× for the conversion heads. An iid-across-games simulator will visibly
  understate the spread of a simulated season through minutes and essentially nowhere else, so the
  spell process and the minutes process should be one sequential layer.
- **Never plug in `E[min]` or `E[gp]` — draw them.** The DK bonus is a threshold on five
  components simultaneously, so the mean of the product is not the product of the means. This is
  the same reason `targets.py::expected_bonus` needs its shared frailty term.

Also settled and relevant to the backtest harness: **every fitting frame is regular season
only**, matching this plan's contest window (Round 4 ends 4/4, before the playoffs begin), so a
historical replay never needs playoff box scores. Game length — the trials denominator for
minutes, 48 or 53/58/63/68 — is available for every game in `data/features/game_length.parquet`,
which matters here because **5.93% of games go to overtime** and a simulator that caps minutes at
48 will systematically understate the top of the minutes distribution.

## What must be built

1. **Draft simulator** — player pool + rankings (ideally posterior draws from
   `docs/predictions-plan.md`, not point ranks) + an opponent-behavior model (start simple:
   opponents draft off an ADP/consensus proxy plus noise, or off DK's own default board plus
   noise, mirroring real autodraft: queue → ranking → position caps) + snake order over 16
   rounds, pods of 12. **Build both modes, and do not default to treating ranking-submission
   as primary**: the user is deliberately moving away from many-parallel-cheap-entries this
   year specifically because it forced autodraft-like behavior and precluded real strategy
   (see below), so a reactive live-pick mode — including a real-time recommendation assist for
   slow (8 h/pick) drafts, which are genuinely playable manually — is at least as important a
   deliverable as the batch ranking-submission mode, not a secondary one.
2. **Season/tournament outcome simulator** — simulate player performance from the joint
   posterior in `docs/predictions-plan.md` (not independent marginals), select the best 7-of-16
   lineup by slot each scoring week exactly per the eligibility rules above, accumulate Round-1
   standings over 17 weeks, apply the advancement and tie-break logic into Rounds 2–4 with the
   same frozen roster, and score under a **data-driven tournament spec** (see below) rather than
   a hardcoded payout function.
3. **Portfolio layer sized to a small number of higher-conviction entries, not ~100** — the
   user is entering fewer, bigger bets this year, bounded hard by each tournament's
   `max_entries_per_player` (6 to 150 across the five real examples below). This changes the
   portfolio problem from "spray 100 cheap bets and rely on the law of large numbers" to
   something closer to Kelly-style bankroll allocation across a handful of deliberately
   differentiated entries — exposure caps still matter, but at N=6–20 each entry is a real
   decision, not a diversification unit.
4. This reuses the skeleton `nba_stats/01-scripts/run_tournaments.R` started (`run_draft`,
   `make_pick`, the week-bucketing in `run_season`) but finishes what it left undone: ADP-aware
   opponent modeling (the old ranking object has no ADP column at all), lineup selection +
   standings + payout scoring (`run_season` computes `team_position_rank` but never finishes
   selecting a lineup or accumulating standings — the `run tournaments` section was never
   filled in), and portfolio-level exposure control (didn't exist).

## Payout structure as an explicit, adjustable lever

Unknown for now — this section is the explainer for why it matters and how to keep it
adjustable rather than baked in.

Two archetypes bound the design space:

- **Flat / cash-like** (a smooth payout up to a broad cutoff — e.g. top half doubles up):
  marginal prize value is roughly constant across a wide range of scores. Optimal play is
  classical mean-variance: maximize expected score for a given variance, i.e. favor
  consistency and avoid unforced correlated risk. This is what "draft the highest-value team"
  naively optimizes for, and it's the right target *only* under this payout shape.
- **Top-heavy / GPP-like** (payout concentrated in the extreme right tail, and the pool is
  split among ties): marginal prize value is near zero except near the very top, so the right
  objective is **P(extreme outcome)**, not E[score]. Two consequences that feel unintuitive
  coming from cash-game thinking:
  - You want variance, and specifically **correlated** upside. Uncorrelated high-variance picks
    average out across a 16-man roster (that's the law of large numbers working against you);
    correlated boom picks — same team, same game-environment thesis — don't average out, they
    either boom together or bust together, which is what it actually takes to reach the tail.
  - You want **differentiation from the field**, independent of accuracy. If your model and the
    consensus agree on a player's value, holding that player is fine in a cash game and
    actively costly in a GPP, because a chalky combination that booms is split among everyone
    who held it — a less-owned combination that also booms wins more. This is a second, distinct
    reason to care about ADP beyond "is my model right": even when you agree with the market,
    you may not want to hold what the market holds.
- **DK's actual structure is a hybrid, and it already leans convex even before the $ curve is
  known — confirmed below, and more starkly than this section originally speculated.** Round 1
  turns out to be not merely convex but a literal zero-consolation knockout in every real
  tournament checked (see "Five real tournament examples"), and because the roster is frozen
  after Round 1, that decision must also hedge the shorter, higher-variance Rounds 2–4 — a
  **dual-horizon problem**, not a single safe/spiky dial.
- **Design implication**: the outcome simulator's objective must be a swept parameter — a
  mean-variance trade-off knob for now, the literal payout table once it's known — so strategy
  exploration (the part explicitly staged for later) is a config change, not a rewrite.

A second, separate lever: **your own entry portfolio** can diversify even when each individual
entry wants to be "spiky," but only along a dimension that helps. The original failure was
accidental correlation across entries on the *wrong* dimension — a shared model blind spot
(injury-blindness) rather than a deliberately chosen boom thesis. A good tournament
portfolio differentiates entries on purpose (e.g. different stacking theses, exposure caps by
risk tier), not by accident.

### Five real tournament examples — this is no longer hypothetical

`data/raw/dk_best_ball_tournament_metadata.csv` and `dk_best_ball_tournament_prize_structure.csv`
give five actual DK Best Ball tournaments spanning the exact axis the user described — entry fee
from $1 to $450, field from 216 to 35,280 entries. The live tournaments run closer to the 2026-27
season may differ slightly; this is enough to build and validate the machinery now, then re-point
it at real numbers later. Derived by walking entries through each round (script-verified):

| tournament | entries | max/player | fee | 1st prize | rake | R2 advance | R3 advance | R4 finalists paid |
|---|---|---|---|---|---|---|---|---|
| `600k_shootaround` | 35,280 | 150 | $20 | $200,000 (×10,000) | 15.0% | 1 of 12 | 1 of 10 | 49/49 |
| `50k_four_pt_play` | 14,688 | 20 | $4 | $5,000 (×1,250) | 14.9% | 2 of 12 | 2 of 12 | 68/68 |
| `15k_and_one` | 17,640 | 150 | $1 | $1,500 (×1,500) | 15.0% | 1 of 10 | 1 of 7 | 24/42 |
| `20k_spin_move` | 432 | 12 | $52 | $5,000 (×96) | 11.0% | 2 of 6 | 2 of 6 | 8/8 |
| `88k_alley_oop` | 216 | 6 | $450 | $20,000 (×44) | 9.5% | 2 of 6 | 2 of 6 | 4/4 |

Findings that sharpen the payout discussion above:

- **Round 1 is a hard, zero-consolation cutoff in every tournament, no exception.** All five use
  a 12-entry live draft pod for Round 1 (`total_entries / 12` divides evenly in every case), and
  every one pays **nothing** to ranks 3–12 — only the top 2 advance, $0 consolation otherwise. So
  the entire 17-week Round 1 isn't merely "convex," it is literal binary elimination with no
  partial credit for 10 of every 12 entries. That raises the tie-break mechanics in
  `docs/dk_best_ball_rules.md` (best single week, cascading) from a footnote to something the
  simulator needs to get right — the 2-vs-3 boundary in a 12-team pod over 17 weeks will be
  close often.
- **Rounds 2–3 advance rates diverge sharply by tournament, independent of the Round-1 rule.**
  Harsh on the cheap, mass-entry tournaments (`600k_shootaround` 1/12 then 1/10; `15k_and_one`
  1/10 then 1/7) versus generous on the pricier, small-field ones (`50k_four_pt_play`,
  `20k_spin_move`, `88k_alley_oop`, all 2 of 6–12, i.e. ~17–33%).
- **Payout convexity scales with entry-fee tier, not with "is this a tournament."** All five
  share the same 4-round advance/cash shape, yet `600k_shootaround`'s 1st place is 10,000× the
  entry fee and 33% of the entire pool, while `88k_alley_oop`'s 1st place is only 44× entry fee
  and the top 4 split almost evenly (2.3× max/min). Cheap mass-entry tournaments are
  lottery-like at the top; expensive small-field ones are close to flat once you reach the final
  table — the flat-vs-GPP dial from the conceptual discussion above is largely **a function of
  which tournament tier you enter**, not one project-wide constant.
- **Rake is lower on the pricier, smaller-field tournaments** — 9.5–11.0% on
  `20k_spin_move`/`88k_alley_oop` against 14.9–15.0% on the three cheaper ones. A real,
  quantifiable input to tournament selection, independent of any edge estimate.
  **Express it as the break-even edge hurdle `1/(1 − rake) − 1`** — the amount by which an entry
  must beat the average entrant just to return its fee: **+10.45%** (`88k_alley_oop`), +12.32%
  (`20k_spin_move`), +17.50% (`50k_four_pt_play`), **+17.60%** (`600k_shootaround`,
  `15k_and_one`). The cheap tournaments demand **68% more edge to break even**. This is the form
  the outcome simulator's objective should compare against, because it is denominated in the same
  units as a measured edge — a raw rake percentage is not.
- **`max_entries_per_player` hard-bounds portfolio size per tournament** — 150 down to 6 across
  the five — and lines up directly with this year's plan: the $52/$450 tournaments are
  structurally built around a small number of entries, the $1/$20 ones around mass entry.
- **Open, unquantified tension worth flagging rather than assuming away**: field *quality*
  plausibly scales with stakes in the opposite direction from the math above — a $1
  tournament's field likely has far more casual/autodraft entries than a $450 one's, even
  though its Round-1 cut is exactly as harsh. The opponent-behavior model in the draft
  simulator should let field skill distribution vary by tournament tier rather than assume one
  field composition everywhere.

**Design implication**: the tournament model in the outcome simulator should be a **generic,
data-driven spec** — round count, pod size, advance count, and cash table all read from a table
shaped exactly like these two CSVs — rather than hardcoded per tournament. Comparing strategies
across all five examples now, and swapping in the real end-of-season numbers later, should both
be a data change, not a rewrite.

## Validation: backtest against real seasons

Draft simulated portfolios, sized to each tournament's `max_entries_per_player`, under candidate
strategies (greedy-by-model, ADP-blended, exposure-capped, posterior-sampled) using only
information available as of each historical draft date, replay them against realized box scores
under each tournament's real payout structure, and compare. This is the harness
`run_tournaments.R` never finished, and it replaces "learn the lesson live" with "learn it in a
backtest" — directly the priority the user set: machinery
first, strategy exploration second.

## ADP-blended ranking — the immediate next discussion

**Sourcing is resolved — see `docs/adp-plan.md`** for the measured survey, storage schema and
point-in-time rules. Three of its findings bind directly on the mechanics below:

- **ADP lives in this layer, not in the prediction layer.** Confirmed as a decision, so `α` below
  sweeps a real axis rather than a weight the GLMM already absorbed.
- **Blend against a DK-*recalibrated* consensus, not a raw one.** The only real DK ADP is a single
  2025-10-17 snapshot; a Yahoo/ESPN consensus correlates with it at ρ = 0.870 but with a
  **21.9-pick mean absolute rank gap**, and DK drafts centers **11.9 picks earlier** because
  category-league ADP discounts them for FT% while DK Best Ball pays rebounds and blocks flat.
  Uncorrected, that scoring-system artifact reads as model edge on exactly one position. A
  monotone recalibration cuts cross-validated error 24.0 → 17.0 picks.
- **Disagreement is concentrated in rounds 9+** — mean absolute gap 5.1 picks in rounds 1–2
  against **30.8 in rounds 9+**, which is where 9 of the 16 roster spots are filled. Any blend
  tuned on early-round agreement will be tuned on the easy half; `α` most likely wants to vary by
  round.

The blending mechanic itself remains open. Candidates, none decided yet:

- **Continuous blend**: rank by `α · model_z + (1 − α) · adp_z`, with `α` tuned against backtest
  performance — the lever the old draft bot never had at all (its ranking object carried no ADP
  column).
- **Confidence-weighted blend**: `α` varies per player with model confidence (e.g. shrink toward
  ADP for players with thin games-played history or volatile injury-report status) — the
  Bayesian-prior framing from `docs/predictions-plan.md`, treating ADP as the market's implied
  prior.
- **Hard disagreement audit**: cap the effective rank gap versus ADP rather than smoothly
  blending, with an explicit override list for cases where the gap is *explained* (e.g. the
  availability head already prices in injury news the market is still catching up to).

## Open questions / risks

- Payout curve and pod size / advance count are now grounded in five real examples (above), but
  the live tournament run closer to the 2026-27 season may differ — re-verify against the actual
  metadata/prize-structure files before a backtest result is treated as load-bearing.
- Field skill distribution by tournament tier is unmeasured (flagged above) and will bias any
  opponent-behavior model that assumes one field composition across all five tournament types.
- **The opponent model's ADP input has a capture deadline** (`docs/adp-plan.md`): DK's board is
  login-gated, has zero Wayback presence, and is live only while contests run — ~October 2026.
  Missing it leaves the DK↔consensus recalibration fit to a single anchor for another year.
- **No-redraft risk**: a Round-1 pick who gets traded or suffers a long-term injury is frozen
  dead weight through Rounds 2–4. This is a robustness consideration for ranking that a per-game
  dk_pts model alone doesn't capture and may warrant an explicit term distinct from raw expected
  value.
- Rescheduled/suspended-game and stat-correction edge cases in the rules doc matter for
  faithfully replaying historical schedules in a backtest, and should be handled once by the
  code that builds the weekly scoring-period bucketing, not reinvented ad hoc per use.
