# The final evaluation, and the production fit behind it

**Opened and taken 2026-08-21.** Two things happened on one day and they are different
acts: the held-out seasons were **measured** once, and then **fitted on** so the 2026-27
board is not throwing two years of data away. This doc is the record of both — what was
unlocked, what it said, and what it does not say.

`docs/train-validate-test-split.md` is the discipline this round spends; read it first. It
explains why the test split is a capability rather than a convention, and why a
validation-versus-test gap here is **not** a replication check: every figure below comes
from a model refitted on train **plus** validation, so the two columns differ in training
data as well as in evaluation rows.

---

## 1. Two acts, and why the production fit needed its own unlock

Before 2026-08-21 `src/final_evaluation.py` was the only thing permitted to unlock
`src/models/held_out.py`. That was right for measuring and wrong for deploying, and the gap
showed up as a command in `docs/preseason-plan.md`'s October runbook that could not be run:
`make posteriors --window full` reached `assert_unlocked` and raised, because nothing
unlocked it.

So there are now two unlockers, with different reasons, and the difference is worth stating
because a second unlocker is exactly the kind of thing that erodes a guard:

| | `src/final_evaluation.py` | `posteriors.assert_production` |
|---|---|---|
| what it does | scores the shipped spec on the test seasons, once | refits the shipped spec on every season there is |
| produces | numbers | coefficients |
| reason string | `end-of-project evaluation of the whole workflow` | `the production fit for the upcoming season's draft board` |
| can a decision read its output? | yes — that is the risk it is disciplined against | **no, because it takes no measurement** |

The production fit is safe to add precisely because it does not score. `tests/test_held_out.py`
pins that as a property rather than an intention: `posteriors.py` must not name a scoring
function, or the production window becomes a way to read the test split that nothing calls
a measurement — which is the shape of the Gate D failure that produced `held_out.py` in the
first place.

### The ordering gate, and why it is a hard requirement

`assert_production` wants two things: `--production` typed, and
`outputs/predictions/final_evaluation.csv` already on disk.

The second is the load-bearing one. **Deploy before measuring and there is no honest
measurement left to take** — from the moment a production fit exists, every candidate model
has seen 2024-25 and 2025-26, and the comparison that would have priced the workflow can
only be made against a model that already read the answer. The file is written once and
stays, so this gate fires on the first production fit and never again.

`make posteriors-production` is the target. `make posteriors WINDOW=full` on its own still
raises, and its message names the target — a window is a value that gets passed around, and
an unlock should be an act somebody performs.

---

## 2. What was refit, and the stale window the check caught

The held-out reading has to describe the model that would actually deploy, so every head is
refitted on train **plus** validation before it scores. That rule was already followed by
the three registered heads. It was **not** being followed by the persisted posteriors, and
nothing could see it.

`data/features/posteriors/train_val/` had been on disk since 2026-08-08 and was carrying a
different model from the one that ships: it predated the preseason block on ten of eleven
rate heads (2026-08-15) and the composition's adopted offset (2026-08-14). Right window,
wrong model — and every column that would have said so was already in the manifest.

**`posteriors.assert_same_specification` is the check that was missing.** `require_window`
stops a consumer reading coefficients that saw too much; this is the other half, and it
compares two windows on the eight columns that define a *specification* rather than a fit —
`family`, `variant`, `fit_first_season`, `n_features`, `preseason`, `preseason_columns`,
`player_season_effect`, `sigma_u`. Row counts, season spans, R-hat and the draws themselves
differ legitimately across windows and are not compared. A head missing from the wider
window is refused too, because a partial `--groups` run is the normal way to produce one.

The chain readout calls it before it simulates anything.

---

## 3. The heads

`make final-evaluation`, run 2026-08-21. Availability and games played score **911**
held-out player-seasons; the season total scores **896**, the difference being the inner
join with the rate frame.

### 3a. Availability — the ordering reproduces and the mixture's margin *widens*

The head the simulator consumes is `stan_posterior`. All five arms are scored because
`stan_availability` compares one likelihood fitted two ways, and which row the simulator
reads was settled by the argument rather than by a metric.

| arm | CRPS (val) | CRPS (test) | MAE (test) | R² (test) | PIT KS (test) |
|---|---|---|---|---|---|
| `beta_binomial` | 9.8444 | **10.7254** | 15.4239 | 0.2856 | 0.0599 |
| `beta_binomial_role_rho` | 9.8247 | **10.7083** | 15.4239 | 0.2856 | 0.0573 |
| `mixture_mle` | 9.1390 | **9.8759** | 14.3878 | 0.3873 | 0.0356 |
| `stan_plug_in` | 9.1329 | **9.8792** | 14.3866 | 0.3869 | 0.0367 |
| **`stan_posterior`** | **9.1289** | **9.8771** | **14.3861** | **0.3873** | **0.0367** |

Three readings, in order of how much they are worth.

**The two-component mixture's margin over the plain beta-binomial reproduces and grows** —
**0.7154** CRPS on validation, **0.8483** on test. That is the §7 selection in
`docs/availability-window-plan.md`, which was taken on a boundary error rather than on CRPS,
and it holds on seasons nothing in it has seen.

**The Stan port still reproduces the point MLE.** The three ports spread **0.00332** CRPS on
test against **0.01003** on validation — tighter, on a frame the port never touched. This is
the fourth time that reproduction has been checked and the first time on held-out rows.

**Calibration barely moves and accuracy does.** PIT KS goes 0.0344 → 0.0367 while R² falls
0.4709 → 0.3873. The head is about as well-calibrated on unseen seasons as on validation and
explains less of the variance in them, which is the expected shape: shrinkage buys
calibration and costs sharpness, and the test seasons are the ones it was shrunk without.

The board-correlation table comes back too, in
`outputs/predictions/final_evaluation_availability_board.csv` — "how much does my whole
board move together" is the quantity the posterior exists to supply. On a 12-man board the
shared-`beta` inflation is **1.0046**; across all 911 held-out player-seasons it is
**1.1256**. The number a drafter cares about is the first one, and it is small.

### 3b. Games played — the head does not ship, and the incumbent row is the answer

⚠️ **A read of the artifact was wrong here and is fixed.** `_games_played` took
`spell_process.csv`'s `arm` column as "which arm shipped". That column is set to the best
**fitted** arm even when *neither* candidate clears Gate D — `stan_games_played` prints "the
head does not ship; the incumbent stands" beside it — so the module reported
`duration_covariates` as selected and then declined to score it. The verdict is now read
from the artifact's `gate_d/passes` row, which is **0**.

So the held-out reading of this head is one row, and it is the incumbent availability head:

| | CRPS | MAE | R² | PIT KS | P(gp<41) pred/obs | P(gp<60) pred/obs |
|---|---|---|---|---|---|---|
| `incumbent`, held out | **10.7952** | 15.3909 | 0.2831 | 0.0963 | **0.1499** / **0.1180** | **0.3478** / **0.3687** |

✅ **Those four tail figures re-derive a reading the project retired two weeks ago, and it
reproduces.** `docs/availability-plan.md` carries "(Test reading, retired: 15.0% / 34.8%
predicted against 11.8% / 36.9% observed)" — a pre-conversion figure taken back when Gate D's
bars were specified on the test split, kept beside its validation replacement (15.5% / 35.2%
against 10.0% / 32.6%) rather than deleted. The held-out run lands on **0.1499 / 0.3478**
predicted against **0.1180 / 0.3687** observed: the same four numbers to three significant
figures.

The observed pair has to match — same rows, same target. The **predicted** pair matching is
the finding, because it comes from a model refitted on train **plus** validation rather than
the train-only fit that produced the retired reading. Two extra seasons of training data
move this head's left tail by less than a tenth of a point, which is what a head shrunk this
hard is supposed to do, and it is the first time that has been checked rather than assumed.

`docs/games-played-plan.md`'s Gate D row quotes the same pair as its bar. That line is the
incident record — the gate specified on test figures — and it stays as written.

A second consequence, and it is the reason this matters beyond bookkeeping. The season
total's `spell_process` treatment means "the games-played module's pmf, composed through the
rate model". With nothing shipping there is no such pmf, so writing the **incumbent's** pmf
under that name would have reported one model twice and made the held-out Gate E a
comparison of a model against itself. The treatment is now skipped loudly instead, which is
the honest shape of "this arm does not ship". The first run of this evaluation did report
that row (MAE 435.1674, against `beta_binomial`'s 435.1352 — the giveaway) and it is
withdrawn.

### 3c. The season total — the project's headline, reproduced

`season_total = gp × dk_pts_per_game_played`, with the rate model held identical across rows
so the contrast is the availability treatment and nothing else. The fixed rate model scores
held-out R² **0.7702** on dk_pts per game played.

| treatment | MAE (val) | MAE (test) | R² (test) | bias (test) | CRPS (test) |
|---|---|---|---|---|---|
| `full_season` | 610.7564 | **646.2640** | 0.1414 | 541.9020 | — |
| `prior_gp` | 417.8882 | **475.0156** | 0.4933 | 41.8615 | — |
| `league_age` | 461.6116 | **476.0361** | 0.5588 | 23.1869 | 340.8155 |
| **`beta_binomial`** | **400.4586** | **435.1352** | **0.5952** | **6.1199** | **316.9385** |
| `oracle_rate` | 261.9324 | **302.6747** | 0.7726 | 5.3446 | — |
| `oracle_gp` | 214.3973 | **221.3128** | 0.8850 | −33.1500 | — |

🔴 **These six MAE figures are not new, and saying so is the point.**
`docs/availability-plan.md` already carries them: "⚠️ This table was a TEST evaluation until
2026-08-05 and read 646.3 / 475.0 / 476.0 / **435.1** / 302.7 / 221.3 MAE on 10,294 train /
896 test." The 2026-08-21 run reproduces every one of them, on the same 10,294 / 896 row
counts, and `src/docs_audit.py`'s `SEASON_TOTAL_HISTORICAL` list has been presence-checking
them as retired values for two weeks.

They reproduce because **this table is not the shipped availability head.**
`season_total.compare` composes `availability.BetaBinomialGLM` — the plain incumbent — with
a fixed rate model, and nothing in that composition has changed since 2026-08-05. The
mixture, the 2012-13 window, the role-graded dispersion and the preseason block all live in
`stan_availability`, which §3a covers and which this table has never read. So the honest
statement is that the composition did not drift across three weeks of work one head over,
which is worth knowing and is not a held-out surprise.

**What is new is the comparison, and it is the replication the round was for.** The
*validation* column is post-conversion — it did not exist before 2026-08-05, because until
then this table had no validation column at all. So:

> `README.md` says the availability head is worth −210 dk_pts of season-total MAE against
> assuming a full season. On validation that is **210.2978**. On the held-out seasons it is
> **211.1288**.

One of those two numbers was taken today and the other was recoverable before the round
started; what the round establishes is that they agree. That is a weaker claim than "the
headline was confirmed on unseen data today" and it is the one the evidence supports.

⚠️ Read §3a for the figures that *are* new. The availability head's held-out CRPS has never
been computed before under any discipline — every arm in that table postdates the
conversion.

**The oracle comparison replicates and widens.** An oracle on games played beats an oracle
on the scoring rate by **47.5351** dk_pts of MAE on validation and by **81.3619** on test.
"The availability distribution is where the effort belongs" is not an artifact of the
validation seasons.

**Bias is the one row that improves.** The shipped treatment carries −3.0569 on validation
and **+6.1199** on test: still small, and it changes sign, which is what an unbiased
treatment is supposed to do across samples. `full_season`'s bias of **541.9020** is the
comparison that makes the point — assuming everyone plays every game is not a slightly worse
model, it is a differently-shaped one.

R² falls 0.7073 → 0.5952 across the same move, which is the availability head's own R² fall
showing through the composition. The MAE, which is what a drafter loses, does not.

---

## 4. The chain — the workflow rather than a head

The three heads above are the model's figures. The chain is the project's: a board built
from the deployed posterior, drafted under the **shipped** strategy against an ADP field,
and scored on the box scores that actually happened.

It is registered as a fourth head, `python -m src.final_evaluation chain`, and it costs
hours where the others cost minutes.

### 4a. Three things it deliberately does not do

**It does not select.** The strategy comes out of `strategy_shipped.csv` — the rule
`strategy.ship` was written for, and the one `_games_played` already follows. Re-running the
24-arm sweep on the test seasons would turn the held-out reading into the largest selection
event in the project.

**It does not simulate a world.** `make strategy-sweep` reports a simulated-truth sweep and
a realized replay, and only the realized half belongs here. The simulated half needs Gate
C's error injection, which solves two nuisance parameters — `rho` from the market-minus-model
skill gap and `g` from the season-total MAE — **against the realized season it is then
scored on**. On validation that is a tuning surface and says so. On test it would be a
held-out figure calibrated on the held-out answer.

**It does not manufacture a market.** See below.

### 4b. 🔴 One of the two test seasons has no admissible ADP, and it is a capture gap

`src/features/draft_pool.py` already records that under `adp.training_rows` only five of the
nine ADP seasons survive point-in-time discipline: a board is admissible only if it was
*observed* on or before the season's first game. **2024-25 is not one of them.** Every
archived snapshot of that season's board postdates the opener, so the board that season
drafted on was never captured.

Measured at the pool, on 2026-08-21:

| season | draftable players | carrying a legal ADP |
|---|---|---|
| 2024-25 | 458 | **0** |
| 2025-26 | 456 | **258** |

So the contest is replayed on **one** season. Two was already the ceiling on this estimate —
`strategy.replay_realized`'s own docstring says N = 2 correlated pods cannot distinguish
`alpha = 0.3` from `alpha = 0.5` — and one is what the ADP capture actually left.

This is checked **before** a field is drafted, because a field drafted from an all-`nan`
board does not fail. It drafts something, and every number downstream of it would be a
statement about that something.

The gap is not recoverable: the DK board is login-gated with no archive and the FantasyPros
snapshots for that window postdate the opener. It is the strongest existing argument for the
`make daily-capture` cron, which is what stops the same hole opening again.

### 4c. What still runs on both seasons

Gate A needs no market at all — it asks whether a season drawn from the deployed posterior
lands where the season that happened landed — so **both** test seasons are simulated and
gated. Only the contest is restricted to 2025-26.

The held-out simulation writes `sim_season_gate_a_final.csv` rather than adding rows to the
shipped `sim_season_gate_a.csv`. That is not tidiness: `make docs-audit` re-derives the
extremes of the pooled table, so a held-out run writing into it would move an audited figure
by adding rows rather than by changing a result.

### 4d. Results

Run 2026-08-21, 20:21 → 20:38 — `assert_same_specification` passed on all 20 heads
before anything simulated, both test seasons drew 2,000-sim tensors at the `train_val`
window, and the contest replayed on 2025-26 alone (§4b).

**Gate A — the simulator against what happened, on seasons nothing in it has seen.** The
bars beside each figure are the validation run's own values, which is what makes this a
walk-forward reading rather than a self-comparison:

| | 2024-25 | 2025-26 | validation bar |
|---|---|---|---|
| scorable units | 386 | 405 | 873 pooled |
| season-total MAE | **367.3581** | **421.9977** | 400.46 |
| season-total CRPS | 257.6995 | 299.2200 | 287.26 |
| season-total R² | 0.6911 | 0.5267 | 0.7073 |
| season-total bias | −17.4584 | −8.0504 | −3.06 |
| games-played CRPS | **9.2840** | **10.2700** | the head's own floor, 10.0057 |
| minutes spread, conditional on gp | 270.89 | 269.63 | 277.23 |
| bonus per game | 0.1857 vs 0.1700 | 0.1831 vs 0.1410 | — |

The held-out seasons **bracket the validation bar** — one better, one worse, on the
season total and on games played alike — and the like-for-like minutes spread lands
within 8 minutes of it on both. The bonus runs hot on 2025-26 (+0.0421) and the
conditional-on-realized-minutes read is −0.0066, so the excess is minutes-shaped rather
than conversion-shaped. These rows live in `sim_season_gate_a_final.csv`, never in the
pooled validation table the audit re-derives (§4c).

**The contest — one season, one world.** The field is the only thing that can be
resampled, so no interval below is a season interval; `strategy.replay_realized` already
records that N = 2 cannot separate nearby strategies, and N = 1 is what the ADP capture
left. The shipped strategy (`lineup_value_blend30`) against its in-world comparators,
Round-1 lift over the 1/6 null and realized ROI, per captured payout structure:

| structure | shipped lift | shipped ROI | `adp` lift | `model_mean` lift | break-even hurdle |
|---|---|---|---|---|---|
| 600k_shootaround | **−0.0725** | −0.8578 | +0.0056 | −0.0593 | +0.1760 |
| 20k_spin_move | −0.1143 | −0.7537 | +0.1021 | +0.0795 | +0.1232 |
| 50k_four_pt_play | +0.0349 | −0.6951 | +0.1287 | −0.0593 | +0.1750 |
| 15k_and_one | +0.0691 | −0.6325 | +0.0358 | −0.0423 | +0.1760 |
| 88k_alley_oop | +0.0356 | −0.2674 | +0.8138 | +0.8138 | +0.1045 |

(The 88k row is one $450 entry in one world — its comparator column is degenerate and
quoted only because leaving a hole would look like an omission.)

**The honest reading: the held-out chain does not confirm the drafting edge, and that is
the result rather than a caveat.** The sweep's simulated worlds put the shipped
strategy's Round-1 lift at ~0.24 and the validation-season replay at ~0.20; the one
admissible held-out world puts it **negative on two structures including the 600k
flagship** and at +0.03 to +0.07 on the other three, with every ROI negative against
hurdles of +10.45% to +17.60% — and the plain-ADP ranking outperforms the shipped
strategy on four of five structures in this world. One world cannot separate strategy
from variance, so this neither refutes the simulated edge nor supports it; what it does
establish is that the project's realized evidence for the edge ends where it stood after
the validation replay, and the held-out season bought no confirmation.

One structural note the contest surfaced: the board was restricted to the 361 pool
players the tensor prices — 95 dropped, 25 of them carrying ADP — which is §6h's
rookie-and-fringe scope limitation showing up as market names the model cannot rank
(`docs/potential-to-dos.md` §16).

---

## 5. The production fit

`make posteriors-production` — the same twenty chain heads at the `full` window, fitted on
every season there is, written to `data/features/posteriors/full/`. It is the artifact
`docs/preseason-plan.md`'s October runbook consumes, and the runbook's key structural fact
is why it can be built in August at all: **the heads fit on historical seasons, so the
2026-27 preseason enters only as prediction-time design rows, never as fitting data.** Every
sampler-hour lands before the preseason; the crunch itself is numpy over the pickles.

Consumers must treat these artifacts as production-only. `require_window` is what enforces
it — a backtest reading `full` heads has read the seasons it is about to be scored on,
through the coefficients, and no frame-level split guard can see that.

Taken 2026-08-21; the last sampler (the composition, the long pole) exited at 23:15.

**All 20 heads are on disk at `full`, and every one matches the `train` window's
specification on the 8 columns** — `assert_same_specification` passes for `train_val`
and `full` alike, which is the §2 check doing the job it was built for on the day it
was built. Across the 20 heads the manifest reads max R-hat **1.00608**, **0**
divergences, **0** round-trip failures.

**The manifest holds 31 rows since 2026-08-22**, not 20: Session 6 of
`docs/rookie-rates-plan.md` (§5f) added the `rookie-components` group — eleven true-rookie
rate heads, ten of them deterministic plug-ins and one fitted — at all three windows, and
the specification comparison gained `family_population` and `deterministic` so a plug-in
cannot be deployed under a fitted head's name. It changes nothing above: the group is
additive, the twenty chain heads are untouched, and `1.00608` is still the worst R-hat
because the one rookie head that samples reads 1.00418. The composition fitted **553,716** player-rows
over 2004-05–2025-26 — against **500,759** at `train_val`, the two held-out seasons'
rows being the difference and the point — at R-hat 1.0026.

`make production-check` after the fit: **the model half is done.** The season half is
October data by nature — the preseason box scores gate `make preseason` and the final
crunch — with one exception closed tonight: the 2026-27 schedule is published, so the
scoring-period grid is already built through the §6g forward knob (1,230 games, DK's
17/2/2/2 shape asserted, R1 closing on the rules copy's own date), to be rebuilt near
the opener when dates move. The 2026-27 rows of the availability panel and component
targets are the runbook's forward-synthesis steps, not missing artifacts.

---

## 6. Scoring a season nobody has played

**Status on 2026-08-21 (end of day): every design input builds forward and is verified —
availability, components, the composition's per-player frame, the scoring-period
calendar — and the acceptance test (§6h) is taken: pushed all the way through the
simulator to a board, the forward path is indistinguishable from seed noise once the
rehearsal's population bound is held aside.** Everything below is the current position. The reasoning that got here
went wrong twice and both corrections are recorded at the end, because the wrong diagnoses
each suggested a wrong fix.

### 6a. The approach — one synthetic artifact, not five rewrites

Every season-keyed frame in this project is built by aggregating
`game_logs_<season>.csv`. `features/availability.build_season_panel` reads it for
membership *and* calls `team_schedule`, which reads the same file for each team's game
sequence — then constructs the panel as **roster × schedule**, which is already the shape a
forward season needs.

So the forward path is that one file, synthesized: `forward_design.synthetic_game_log`
crosses the published schedule with the roster snapshot. The existing builders then work
unchanged, instead of five modules learning a second way to build themselves.

Both inputs exist **today**: `ScheduleLeagueV2` returns 1,271 games for 2026-27 with team
ids on every one, and `CommonTeamRoster` returns 577 players over 30 teams.

### 6b. What it produces, measured

| stage | result |
|---|---|
| synthetic log | **47,314** player-game rows, 577 players, 30 teams, **82** games each (80 published + **30** filler games) |
| `build_season_panel` | 47,314 rows, `played` 0 throughout — no fabricated target |
| `season_availability` | 577 player-seasons, `team_games` **82**, 0 NaN |
| `availability.build_design` | **489** design rows, `gp_share_lag1` 489/489, mean age **27.13** |

### 6b-ii. The membership rule, validated on a season where the answer is known

`make forward-rehearsal` builds the design for a **played** season without that season's
game log and compares it against the design as it is built today — the October runbook's
own advice, because *the real window is too short to debug a join in*. Run on **2023-24**,
a validation season.

**Part A — the mechanics, population held fixed. 23 columns, 450 shared players, 0
unexplained disagreements.** Exactly one column differs on exactly one player, and it is the
one that should: `team_games` by 2. `build_design` ends on `max(team_games, gp)` so a traded
player whose two teams' schedules overlap does not make `betabinom.logpmf` non-finite. That
is **the one place a target-season quantity reaches this head's input side**, and a forward
design cannot reproduce it — in September nobody knows who will be traded. One row in 450,
and the project already documents it as 13 rows in thirty seasons.

**Part B — the population, one-directional.** Both sides are *design* populations, so the
prior-minutes and age qualification is held constant and only the membership rule varies:

| | players | realized dk_pts | over 500 |
|---|---|---|---|
| in both | **401** | 496,255 | 288 |
| snapshot only (spurious) | **0** | 0 | 0 |
| game log only (missed) | **49** | 31,247 | 17 |

**Zero spurious rows** is the production-relevant half: the snapshot carries 532 names
against the design's 450, and the design's own qualification absorbs every extra one. The
forward rule does not invent players.

The 49 misses are **entirely a rehearsal artifact**, decomposed rather than asserted: 29 are
absent from the roster file altogether (traded or waived away before the snapshot was taken,
which nothing in the file records), and 20 were removed by the `HOW_ACQUIRED` cut because
the snapshot records a mid-season team for them — and all 20 appear in the panel inside some
team's first ten games, so all 20 were on an October roster under a different team. Neither
contamination exists for a snapshot taken before an opener, which is the production case.

That asymmetry is why `draft_pool.py` **rejects** `team_rosters_<season>.csv` as a
membership source — a current-status snapshot puts February signings in an October pool —
and why the objection **inverts** for an unplayed season, where February has not happened.
`team_context.season_start_roster` is not the substitute: it reads game logs too, joining a
player to team T at the index of T's game where he first appears.

### 6c. Three things the synthesis does not hand you free

**The two games the schedule does not have.** An unplayed season lists **80** games per
team, not 82 — §6f. Six of the missing games are the Cup knockout, published with the
placeholder id `0` on *both* sides; the other twenty-four are replacement games for teams
that do not advance and are not published at all until December. Leaving them out is not a
2.4% approximation, it is an inconsistency: `team_games` is both the availability head's
binomial denominator **and** the length of the grid absences are laid out on, and 80 on one
side with 82 on the other is a defect. `synthetic_game_log` fills each team to
`SEASON_GAMES`, paired into real two-team games and dated inside the knockout window, and
records the approximation — *which* teams meet on *which* day in that window is not
knowable in October and neither materially moves a season total.

**Age comes from the wrong file.** `load_ages` read `player_bio_stats_<season>.csv`, a
season-*statistics* endpoint that does not exist until games are played. `roster_ages` is
the fallback and deliberately does **not** use the roster's own `AGE` column: that is a
*fetch-time* attribute, matching the bio column 98.7% exactly on a roster pulled during the
season and sitting **0.617 years low** on the 2026-27 roster pulled in August. Age and age
squared are features on every head. `BIRTH_DATE` is exact, so age is recomputed at 31
December of the start year — which reproduces the bio column to a mean of **+0.018** years,
inside half a year on 98.7% of 525 shared 2025-26 players.

**✅ The component design, built 2026-08-21 — it was blocked TWICE, stacked.** The eleven
rate heads score from `component_rates.build_design`, so without it there is no box score to
reassemble `dk_pts` from and therefore no board, whatever the availability and minutes heads
say.

*First, upstream:* `features/targets.build_component_targets` ended on
`dropna(subset=["min"] + COMPONENTS)`. The synthetic log leaves `MIN` blank, so every
forward row was dropped there and `component_targets.parquet` — which `build_design` reads —
never received a 2026-27 row at all.

*Then, downstream:* `build_design` closes on `total_minutes > 0`, and `total_minutes` is the
**target** season's. Blank minutes sum to `0.0`, so every row was dropped again.

Both are correct rules for every use this project has ever had — a player-game with no
minutes contributes nothing to a count likelihood whose exposure *is* minutes — and both are
population rules rather than feature computations, which is why neither was a modelling
problem. Each now takes a `forward_seasons` argument that exempts a named season, and
`forward_design.forward_component_design` is the single call that chains them:

| | result |
|---|---|
| forward component design, 2026-27 | **419** rows |
| `reb_p36_lag1`, `mpg_lag1`, `total_minutes_lag1`, `age` | 419/419 populated |
| the same call with no `forward_seasons` | **0** rows |

419 against the availability head's 489, because this design's `total_minutes_lag1 >= 200`
qualification is stricter than a prior gp-share and an age.

**The default is the guard, rather than a check bolted beside one.** Both knobs default to
empty, so a fitting path cannot receive rows with no targets without naming the season twice
— and `tests/test_forward_design.py` AST-walks `stan_components`, `posteriors`,
`component_rates` and `targets` to assert none of them ever passes it. Clearing one filter
and not the other yields an **empty** frame rather than a wrong one, which is the safe way
for a half-applied change to fail, and `forward_component_design` raises with that
explanation rather than returning nothing.

Played artifacts are untouched and that is checked rather than asserted: recomputed over the
real logs, `component_targets` is **bit-identical** at 731,906 rows with the same columns —
`is_forward` is added only when a forward season is named, because a column that is zero on
every played row is a schema change bought for nothing.

⚠️ Two earlier readings of this were wrong. The `total_minutes > 0` filter was called a red
herring, which was right when nothing reached it and wrong once the synthetic log supplied
rows; and the section then named only that filter, because the toy that measured it built
the targets frame directly and stepped over `build_component_targets` without noticing.

### 6d. What is genuinely October-gated

The preseason box scores, and nothing else. Schedule, rosters and 2025-26 lagged features
exist now; the production posteriors are building.

`make rosters SEASON=2026-27` always re-fetches, because the roster changes with every
signing up to the opener and it is the forward membership rule — a board drafted in October
off a roster cached in August is drafting last summer's league. `fetch_team_rosters` also
read its team list from the season's game log and so would have skipped 2026-27 entirely;
it now falls back to the published schedule.

### 6e. ⚠️ Two corrections, kept because the wrong diagnosis suggested the wrong fix

**"A target-season filter blocks the design."** Wrong for the availability head. Every
feature is prediction-time legal, and the exposure is a **draw** —
`sim/season.py::draw_components` computes `mu = rate[unit] * minutes` from the shared
minutes draw, and `total_minutes` appears once in the whole simulator, lagged. The real
blocker was that rows are *harvested from the game log*, which is upstream of any filter.

**"`played` has to be faked, so every target is fabricated and needs a guard."** Also
wrong, and withdrawn the same day. `played` is used in exactly two places —
`season_availability` and `roster_grid` — for exactly one purpose: attributing a **traded**
player to his last team, because in a played season he has rows under both. `roster_grid`
then drops the column. Given roster membership there is nothing to disambiguate. Measured:
the design built with `played` faked to 1 and with `played` never set gives **489 rows each,
identical on all 19 feature columns, maximum difference 0.00e+00**. Nothing is fabricated
and no guard is required.

🔴 **Checking that turned up a real defect in both callers.** The team-attribution rule is
now shared as `features/availability.primary_team`, and it had two faults. Its emptiness
check ran *before* the `played == 1` filter it depends on, so a forward season returned an
**empty grid** and **all-NaN denominators** rather than raising — a season that simulates
nothing and reports nothing. And the first fix keyed the fallback on `player_id` alone, so
a veteran with appearances in *earlier* seasons was skipped: 496 of 577 players on 2026-27,
presenting as a NaN denominator rather than an error. It is now keyed on the
`(season, player_id)` pair, falls back to a player's only team, and raises when he is
genuinely on two rosters with nothing played.

Played seasons are untouched, and that is checked rather than asserted:
`season_availability` recomputed over the real panel is **bit-identical** to the stored
artifact across 14,569 rows and 30 columns.

### 6f. The Cup finding, and the scoring rule that is separate from it

| season | regular-season games | franchises | games each | TBD placeholder |
|---|---|---|---|---|
| 2025-26 (played) | 1,230 | 30 | **82** | none |
| 2026-27 (published) | 1,206 | 30 | **80** | 6 games, both sides TBD |

Same class of defect `sim/season.py::roster_grid` already records from the other direction,
where a traded player's denominator came out at 92.6 against 82.0 and "every simulated
season would have run the head's rate against ~13% too many opportunities". That one was
found in the numbers; this one was found in August.

**The Cup championship game is a separate matter and is excluded three times over.** The
NBA does not count it as a regular-season game — 2026-27's is `0062600001` on 2026-12-11,
a `006` prefix, so it never enters the game logs and `component_targets` holds `002` rows
only, all 731,906 of them. DraftKings excludes it independently
(`docs/dk_best_ball_rules.md`, same date). And `scoring_periods.cup_final_mask` keys on the
*label* rather than the id prefix on purpose, because the prefix is the NBA's claim about
regular-season status and could change without DK's rule changing; verified against
2026-27, where the game carries `gameLabel = "Emirates NBA Cup"` and
`gameSubLabel = "Championship"`. Group-stage games are regular-season games and are scored.

### 6g. The last two inputs — the composition's per-player frame and the calendar

Both built 2026-08-21, later the same day as the rest of §6.

**The composition's per-player frame builds forward and reproduces the shipped head's.**
Everything the simulator reads off the composition's fitting frame is constant within a
player-season — `w_share`, the allocation order's keys, the design columns behind
`composition_eta` — so `forward_design.forward_composition_players` builds exactly those
columns, through `season_weights` and `shipped_share_hook`, both **extracted from**
`stan_composition` rather than copied out of it, so the values come from the code that
fits. Rehearsed on 2023-24 as Part D, population held fixed at the shipped
`head_frame` path's own players: **25 columns × 572 players, 0 unexplained
disagreements**. This is also the first forward-built row through the preseason blend
join — 2023-24 has a real preseason panel, so the join carried real weight and still
agreed — which closes the open question the first version of this section recorded.

The classified difference is the **draft number**, whose retrospective source
(`bio_draft_number`) is a season-statistics file that does not exist before the opener.
The forward rule reads the player's own earlier matrix rows first — past the target
season's own row, which in a rehearsal is bio-derived and would grade the rule against
itself — and the roster snapshot's "#N Pick in YYYY Draft" text for rookies; the residue
lands in the undrafted bucket, which is what a missing draft number does everywhere else
(`forward_draft_numbers`'s docstring carries the 2026-27 coverage counts). On the
rehearsal the mismatches decompose exactly: **33** players with no pre-2023-24 matrix row
— the production case, text-sourced — and **19** who are no longer on the snapshot at
all, Part B's contamination one column over, impossible in production where membership
IS the snapshot. `w_share` (and so its logit and `order_share`) moves only inside the
first set, by at most **0.0950**, because a no-prior rookie's weight is his bucket's
expanding prior and the bucket is what the text sometimes cannot supply.

**Scoring periods build forward from the synthetic log.** `scoring_periods.run` takes a
default-empty `forward_seasons` — the same guard shape as the component knobs — and
appends the named season's `(season, game_id, game_date)` triples from
`synthetic_game_log` rather than from the raw schedule. That sourcing is the point: the
raw schedule is 26 games per team short of usable (§6f), and the synthetic log carries
the SAME game ids the roster grid will, filler games included — periods keyed to the raw
schedule would leave the filler games slotless, and a game with no slot scores for
nobody. Every forward date is a plan rather than a realized date, so the October runbook
should rebuild the artifact near the opener rather than trust an August build of it.

### 6h. The acceptance test — the forward board against the retrospective one

`make forward-board`, run 2026-08-21: 2023-24 at the `train` window (the pairing
`make simulate-season` itself uses — nothing here touches the test seasons or the `full`
posteriors), 500 sims, boards ranked on mean simulated season total, which is the
quantity every consumer of the tensor ranks on before the market enters. Two control
arms sit beside the forward run: the same retro frames re-simulated at seed+1 — the
noise floor the gap is judged against — and the forward machinery on the retro grid's
own population, because the minutes allocation is zero-sum: a missing player is not a
missing row, he is minutes handed to every shared teammate, and only a fixed-population
arm can separate that from the forward frames themselves.

⚠️ **Re-run 2026-08-22** after Session 6 of `docs/rookie-rates-plan.md` (§5f) put two more
populations on the retrospective board — the lag-recovery ladder's admitted returnees and
the true-rookie rate family. The table below is that run; the numbers it supersedes are
kept in the paragraph after it, because what moved and what did not is the reading.

| arm | shared players | Spearman | mean \|Δ season total\| | top-16 | top-48 | top-100 |
|---|---|---|---|---|---|---|
| retro vs retro, seed noise | 467 | 0.9990 | 20.35 | 15 | 45 | 98 |
| **forward, population held fixed** | **387** | **0.9988** | 23.25 | **15** | **47** | **98** |
| forward, snapshot population | 354 | 0.9828 | 105.11 | 14 | 43 | 89 |

**The mechanics reading did not move and the population bound widened by exactly the new
units.** On the 387 units both arms share, Spearman is **0.9988** against a **0.9990**
floor and a mean season-total gap of **23.25** dk_pts against **20.35** — against 0.9989,
0.9990, 22.17 and 21.03 on the 2026-08-21 run, which is seed noise on seed noise. Every
input difference the earlier parts classified — the draft-number sourcing, the filler
games, the schedule-sourced denominator — still amounts to nothing the board can see.

What *did* move is the population column: the retro board is **467** units where it was
387, and the fixed-population arm is missing **80** of them (74 true rookies and 6
lag-recovered returnees in 2023-24). That arm holds the **roster** fixed, and these rows
are not a roster question — `rookie_rates.build_design` carves its population out of
`component_targets`, so "he played in the NBA this season" is target-season information a
forward context may not read. `forward_context` therefore withholds the target season's
rookie rows and says how many, on both forward arms. The forward rookie tier is
`docs/rookie-rates-plan.md` §5g; until it lands, a forward board carries no rookies, which
is the state every board this project has produced.

⚠️ The first run of this arm reported 18 units it could not share, and the sentence
written for them — a permanent qualification fringe — was wrong. Profiling the 18 found
every one absent from the roster snapshot: `forward_component_design` synthesized its
own log from the roster *file*, so the population override reached three of the four
frames and not the fourth. The membership frame now flows through the component log
too, the arm shares all 387, and the wrong diagnosis is recorded here because it
briefly claimed a production limitation that does not exist.

The whole headline gap (**0.9828**, **89**/100) is therefore the population bound Part B
measured, priced here at the board: **113** players the forward arm does not carry, **6**
of them inside the retro top 100, reaching it partly through the redistribution of their
minutes onto everyone else. **80** of the 113 are the two new populations above and are
a missing forward *builder*; the remaining 33 are what the 2026-08-21 run measured — the
players the 2023-24 snapshot no longer lists, traded or waived away before it was taken
and invisible in the file. Neither contamination exists for a snapshot taken before the
opener, which is the production case — and **0** spurious players, in both arms, again.

⚠️ **The sentence that used to stand here — "no board this project produces contains a
true rookie" — was true until 2026-08-22 and is now half true.** The retrospective board
carries **74** of them in 2023-24 and 72 in 2022-23, because `docs/rookie-rates-plan.md`
§5f made the simulator's scorable units the union of the veteran rate design and the
true-rookie one. The forward board still carries none, which is the id-map and
forward-design work §5g owns, and it is the reason the 2026-27 production board cannot
take a 2026 draftee **yet** — no longer because the features structurally do not exist.

---

## 7. What this round does not settle

- **The chain reading is one season, not two.** §4b: the 2024-25 ADP board was never
  captured, so N = 1 for the contest. The heads are unaffected — they need no market.
- **A val/test gap is still not a replication check**, and nothing here should be read as
  one. Every held-out figure comes from a refit on train + validation, so the two columns
  differ in training data as well as rows. Where a *reproduction* is claimed above it is
  because a margin or an ordering reproduced, not because two levels are close.
- **The season-total table's held-out figures were already knowable** (§3c) and the round's
  contribution there is the validation column and the agreement, not the test column.
- **Forward scoring is wired and accepted at the board** (§6h) — but on ONE rehearsal
  season, with the population bound measured rather than removed, and the production
  window's own risks (roster churn to the opener, a schedule that can move) are not
  things a rehearsal can price. The October runbook still rebuilds every forward input
  near the draft.
- **The ESPN injury feed is down**, returning `403 Forbidden` as of 2026-08-21, with 48
  permanently-lost days since 2026-06-28. That feed is current-status only and has no
  history, so those days do not come back. It does not touch anything measured here — no
  shipped head reads it — but it is the capture program with the least slack and the check
  now surfaces it.
- **The split is spent.** Nothing above may be used to change a modelling decision. If a
  figure here motivates a change, the change is un-priced: there is no second held-out
  reading behind it, and the honest thing to record is that the estimate no longer applies
  to whatever ships next.
