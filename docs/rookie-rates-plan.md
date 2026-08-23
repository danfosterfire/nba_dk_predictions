# Rookie rate heads — scoring the structurally-missing population

**Opened 2026-08-22.** Converts `docs/potential-to-dos.md` §16 into a scheduled program:
a component-rate head family for the players the veteran heads structurally cannot score
— true rookies, returnees, and thin-prior fringe — so they can appear on a board and be
drafted. Seven sessions, each with its own prompt (§6), each appending its results here.

## ✅ STATUS: ALL EIGHT SESSIONS RUN 2026-08-22 (§7a, §7c, §7d, §7e, §7f, §7g, §7h, §7i). DESIGN REVISED 2026-08-22 (§7b). THE INTERLEAVED ROUND — `docs/availability-window-plan.md` §16 — RAN 2026-08-22 AND §7f HAS ITS NUMBER. THE PROGRAM IS CLOSED.

**The program is eight sessions, not seven, and the rookie head serves a smaller
population than the one it was scheduled for.** §7b measured that a player with *any* prior
NBA season is better served by the veteran heads with his nearest usable lag imputed than
by a head built for players with no history at all — so §3 constraint 2 was withdrawn, a
new Session 2 built the veteran design's lag-recovery ladder, and the rookie rate heads
serve **true rookies only**. Everything downstream renumbered by one.

**The ladder is built and gated, and one rung of four survived** (§7c): rung A's
full-lag-2 returnee is admitted, rung B's thin prior and the two tails are not. The ladder
is **inert until Session 6** — `stan.components.lag_ladder` is still `[]` — because turning
it on means routing the fitting paths through `component_rates.fitting_rows`, which is
Session 6's job. The rookie head's population is unchanged by that verdict: it was always
true rookies only.

**The rookie design is built and its floors are measured** (§7d): `make rookie-rates`
carries 1,495 true-rookie rows disjoint from the veteran design at every rung, sixteen
zero-recovering features per head, and eleven dispersion-wrapped floors that beat the
shipping draft-bucket incumbent on CRPS on **11 of 11 heads**.

**The heads are fitted and the gate admitted one of eleven** (§7e): `make stan-rookie`
fits three variants per head and runs §4's conjunction. **`reb` ships FITTED** at the spline
arm (validation −2.8869 [−5.0460, −0.7950], rolling −2.4642 [−3.8744, −1.0523], 9 of 11
origins); the other ten ship §7d's floor as a plug-in. Two heads clear validation, two clear
rolling, and the intersection is one — `fg2m|fg2a` wins validation and **reverses** on the
fitting half, which is the case the rolling half was put there to catch. Every unit still
carries all eleven quantities: the gate decides which arm, never whether.

**The heads are composed into a season total and §16's own gate is answered** (§7f): `make
season-total-rookie` reports `veteran`, `lag_recovered` and `rookie` through
`season_total.evaluate`'s widened group tuple, and the two answers point opposite ways. The
rate family is worth **520.27 dk_pts** of draftable season-total MAE on true rookies and
**519.15** on the ladder's returnees, against the zero they score today — that is the number
§16 asked for, and the program's mandate is discharged. The *fitted* family against the
*floor* family is **null** (+0.9738 [−3.3629, +5.3300] CRPS at the shipped games treatment,
−2.8084 [−6.0123, +0.2479] at `oracle_gp`), which is §7e's one-of-eleven verdict arriving one
level down. The session's largest incidental finding is elsewhere: the ladder's returnees are
given **24.738** games against a realized **46.857**, and an oracle on games takes their MAE
from 543.3167 to 64.4025.

**The family is persisted and the simulator draws it** (§7g): `stan.components.lag_ladder`
is `[returnee_lag2]`, the eleven `rookie-components` posteriors exist at all three windows
(ten deterministic plug-ins, `reb` fitted), and `build_context`'s scorable units are the
**union** of the two rate families — **471** and **467** units on the two validation seasons
against 386 and 387, with Victor Wembanyama on a board for the first time. `make
production-check` is green on 31 of 31 heads. **The tensors on disk were not regenerated**:
what the change is worth in contest units is §5h's replay against §7a's floor.

**The floor is closed at the unit that resolves** (§7i): `make rookie-recovery` re-reads
§7a's Round-1 bar against a **nested ladder of board masks** on a labelled rookie-inclusive
tensor, and rung 0 reproduces `make rookie-floor`'s own **+173.1** / **+111.9** to the digit
— which is what licenses reading the rest of it. The floor falls to **+25.4** on 2022-23 and
**−29.0** on 2023-24; the ladder gives back **+77.5** / **+15.8** and the rookie head
**+70.1** / **+125.1**. The negative residual is a result rather than noise: past the rookie
rung the rows still outside our board realize *less* than what the field would otherwise
take, so the remaining hole is a handicap on the field. `make strategy-sweep-rookie` is the
contest half, and it settles a different thing than it was asked to: the sweep **re-selects
`lineup_value_blend30` on all four multi-entry structures** with simulated lift rising
0.2369 → **0.2645** on the 600k flagship, while the realized lift delta (**+0.0160** for the
shipped arm, 17 of 24 arms losing at a median of −0.0683) refuses to resolve exactly as §7a
said it would — and this time is not even paired, because the tensor moved under it. The
eleven rookie heads are
**declared in the chain and deliberately not carded** (`docs/model-cards-plan.md`), and
`docs/final-evaluation-plan.md` §7 records that the held-out figures measured the
rookie-less workflow.

**The forward path is wired and the production board carries rookies** (§7h): the DK id
map gains a **roster-snapshot** tier (`no_nba_history` 162 → **91**, 71 rows resolving to
real `nba_api` ids, no existing match moved), `forward_rookie_design` builds true-rookie rows
from the roster snapshot for a season nobody has played, and `forward_component_design` now
asks for the ladder it was silently not getting. **2026-27 carries 419 veteran, 6
lag-recovered and 116 true-rookie units, 13 of the rookies ADP-priced.** §4's second
acceptance half holds: on the 354 rung-0 veterans, Spearman **0.9986** against a **0.9988**
seed-noise floor. A test season cannot be *simulated* outside `make posteriors-production`,
so its reading is `make forward-board SEASON=2026-27 FRAMES_ONLY=1`.

⚠️ **Session 6 left one thing broken for the length of a session and
`docs/availability-window-plan.md` §16j fixed it**: the ladder's recovered returnees went
into the tensor at the availability plug-in's 24.7 games. That key is now on too, so both
ladders ship — see §7g's closing item.

**§7a, §7b, §7c, §7d, §7e, §7f, §7h and §7i are in `make docs-audit`** (`_rookie_floor`,
`_lag_recovery`, `_lag_ladder`, `_rookie_rates`, `_rookie_heads` and `_season_total_rookie`,
plus `_final_evaluation` for §7h's board figures, which live in
`forward_board_rehearsal.csv` and `forward_board_population.csv`, and `_rookie_recovery`
for §7i,
against `strategy_rookie_floor.csv`, the
`_rookiefloor` artifact set, `lag_recovery.csv`, `lag_ladder.csv`, `rookie_rate_floors.csv`,
`rookie_rate_metrics.csv`, `season_total_rookie.csv`, `rookie_recovery.csv` and
`sim_season_gate_a_rookieinclusive.csv`). Every
other figure here is either audited from its own artifact elsewhere (`rookie_priors.csv`,
`strategy_injection.csv`, `preseason_coverage.csv`) or labelled scratch — §7a labels its two
scratch measurements inline. Each session that produces a quotable number should add it to
the audit the same day (see `docs/docs-audit.md`).

## 1. The problem

*Written before the program ran, and kept as the problem statement it is. The
retrospective half was closed on 2026-08-22 by §7g — a 2023-24 board now carries 74 true
rookies — and the forward half is §5g's. Everything below describes the state the program
was opened against.*

Every board this project has produced, forward or retrospective, contains **zero true
rookies**, and the mechanism is structural: all eleven component rate heads are
lag-designs. `component_rates.build_design` drops any row with NaN lag columns
(`src/models/component_rates.py:205`) before `MIN_PRIOR_MINUTES = 200` (`:206`) is even
consulted — below it the own-rate features *do not exist* rather than being noisy. The
2023-24 retrospective design carries 0 players whose first played season is the target;
Wembanyama is not among the 387 scorable units the shipped path ranked that season.

The consequence is asymmetric. Rookies DO absorb minutes (the composition's expanding
draft-bucket priors) and DO carry an availability rate (the graded no-design level), so
their teammates' totals are right — but their own dk_pts are never scored. The simulator's
scorable units are `component design ∩ composition players` (`sim/season.build_context`,
which §7g replaced with `component_units`);
a player missing from the design stays in the grid, drains teammate minutes, and never
enters the tensor. Downstream, `priceable_room` (`src/sim/strategy.py:698-763`) restricts
the sweep's board symmetrically and *measures* the hole: 101/105 board rows dropped in the
two validation seasons, and an unrestricted ADP field drafts **~1.2 zero-scoring players
per entry** (73-75% of entries carry at least one). Round-1 rookies routinely carry
top-100 ADP; our seat can never take one.

The population that fails the test is wider than rookies, and **the three groups inside it
do not want the same treatment** — which is what §7b measured and §3 constraint 2′ records.
A returnee's lag-1 is missing only because he sat out; his lag-2 is a full season and
`with_lags` pairs on season index, so the row exists and is never built. A thin-prior
player's lag-1 exists and is noisy rather than absent. Only a **true rookie** has no
own-rate feature at any lag. So the veteran design widens to serve the first two through
the lag-recovery ladder (§5b), and the rookie rate heads serve the third (§5c) — half the
unserved board rows, and the half for which "the features do not exist" is literally true.

## 2. What already exists — do not rebuild

- **P4(b), `make rookie-priors`** (`src/models/rookie_priors.py`, readout in
  `docs/preseason-plan.md`): on the no-prior population, the volume-shrunk preseason
  per-36 (`w = min_pre/(min_pre+k)`) beats the draft-bucket mean on **5 of 8** rate
  targets at 19/19 rolling origins (`reb`, `fga`, `ast`, `blk`, `fg3a_share`); `fta`,
  `tov`, `stl` are nulls; the draft bucket alone is an **anti-model** for rates
  (R² −0.043..+0.046). Selected `k = 160` for rates, 20 for the share, 10 for
  `fg3a_share`. P4 decision 4 is this program's mandate: *"if he is ever put in, his
  preseason per-36 is the prior to use and a draft bucket is not."* Note §16's original
  text ("transfer … never measured for players with no prior season") predates this — the
  transfer HAS been measured, at the estimator level; what is unmeasured is a fitted head.
- **Preseason data**: `data/features/preseason.parquet`, 23 seasons on disk, covered
  window 2004-05 → 2025-26 (`preseason_coverage.csv`; 2003-04 is `tail_missing` and
  excluded). Validation seasons 96.9% / 95.7% roster coverage. Scratch measurement
  2026-08-22: rookies on a season-start roster carry a preseason row **97.25%** of the
  time (n = 1,126, 2005-06 onward); without the roster restriction this falls to ~91-93%,
  the gap being January signings — the population P1 decision 5 exists to exclude.
- **Volume caution**: rookies play more preseason *games* (4.46 vs 4.03) but the same
  median ~68 total minutes. At k = 160 that is weight 0.30 — the reliability shrink is
  not optional; raw preseason per-36 is worse than the anti-model on 6 of 8 targets.
- **The lag-recovery split, `make lag-recovery`** (`src/models/lag_recovery.py`, readout in
  §7b): the six-way population census, the carry-forward power of each rung, and the
  head-to-head against P4(b)'s preseason estimator. It fits no head — every arm is
  `carry_forward`'s functional form with a different prior rate — and Session 2's builder
  imports its `classify`, `recent_lag` and `recent_shrunk_rate` rather than restating them.
- **Stan sources**: `negbinomial_glm.stan` / `betabinomial_glm.stan` are generic in `K`.
  A new head is a feature matrix + fitting population + artifact name; **no new `.stan`
  file, no recompile**.
- **The plug-in precedent**: `sim/season.py::no_design_availability` (`:365-437`) is the
  shape to mirror for anything unfitted — full-length vector with the caller's mask
  deciding, constant across draws ("the honest shape of a plugged-in empirical prior"),
  graded by key ladder with coarsening fallback, arm name persisted into the artifact.
- **Reusable pieces**: `preseason_value.attach_missing_age_indicators` and
  `attach_season_start_roster`; `minutes_preseason.reliability_weight`;
  `components_preseason.fit_shrinkage` (the inner-carve pattern);
  `stan_components.count_floor` / `conversion_floor` (dispersion-wrapped floors);
  `merge_heads` + `run(cfg, heads=...)` (head-local partial refits);
  `team_context.DRAFT_BUCKETS`; `forward_design.forward_draft_numbers`.

## 3. Constraints the program inherits

Decided 2026-08-22 (owner), plus standing rules. **Constraint 2 was reversed the same
day** — the reversal is kept in place rather than edited away, because it is what moved
the rookie head's population and Sessions 2-3 are built on it.

1. **Price first, build regardless.** Session 1 measures what rookie-lessness costs in
   contest units; a noisy 2-season interval is context for the head's later value, not a
   kill switch. (§16's literal "not worth building if within noise" is superseded by this
   decision.) **Held** — §7a's contest half did not resolve and the program continued.
2. ~~**One design for the whole structurally-missing population**~~ — **WITHDRAWN
   2026-08-22, replaced by constraint 2′.** The original reasoning was that draft slot ×
   years-since-draft would let one design serve the 2026 lottery pick and the 24-year-old
   arriving from Europe. It does not follow for the players who have an NBA season to
   draw on, and §7b measured why: a returnee's two-year-old rate carries as well as a
   veteran's one-year-old one and beats the rookie head's own preseason estimator on 7 of
   7 heads. A design that serves him with preseason and draft slot is throwing away his
   best feature.
2′. **The boundary is *is any prior NBA season constructible*, not *is the immediately-
   prior one big enough*.** Three populations, three treatments:
   - anyone with a played season inside the window is served by the **veteran heads**,
     through the lag-recovery ladder of §5b — the nearest usable lag, shrunk by its own
     reliability;
   - the **rookie rate heads** serve players with **no NBA season at all**, and nothing
     else. That is the only group for which "the features do not exist" is literally
     true, which is what the family was named for;
   - the availability plug-in keeps the name "no-design" and is untouched.
3. **The test split is SPENT** (2026-08-21). Selection reads validation only, through
   `held_out.selection_split`. The heads ship **un-priced against the held-out seasons**,
   and Session 8 records in `docs/final-evaluation-plan.md` that the final-evaluation
   figures measured the rookie-less workflow.
4. **No refit of the existing 20 heads.** Parameter blocks are disjoint; the chain
   factorizes; the rookie heads are additive artifacts. `assert_same_specification`
   refuses a reference head missing from a wider window, so the new group must be fitted
   at **both** `train_val` and `full` (production is already fitted; existing artifacts
   untouched).
   **Constraint 2′ does not breach this, and that is the reason it takes the form it
   does.** The ladder widens the veteran heads' *scoring* population, never their fitting
   population: the imputation happens at design time and the existing coefficients score
   it unchanged. A version that admitted these rows to the fit — with a staleness feature
   the head could learn a slope for — would be cleaner statistically and would cost
   eleven refits at two windows. §5b gates the cheap form first and names the escalation.
5. **Scope runs through the forward board** — the id-map tier (§5g) is in scope, so the
   2026-27 draft can actually take a rookie.
6. Anything fitted from data — shrinkage k, dispersion, imputation means — is estimated
   on the fitting half alone. Every preseason figure is quoted on the
   season-start-roster population.

## 4. The gates, stated before any result

- **Ladder rung gate (Session 2)** — a rung is admitted only if, on the population it
  serves: **(1)** its validation paired-bootstrap CRPS interval against the *unserved*
  status quo lies entirely below zero at the player-game unit; **and (2)** its carry-
  forward R² lands inside the shipped no-fit floor's published band (0.81-0.95 on the
  count heads), so a rung cannot be admitted on the strength of beating nothing. The
  scoring-only form is gated before the refit form is considered at all.
- **Per-head ship gate (Session 4)** — the standing conjunction:
  **(1)** validation paired-bootstrap CRPS interval entirely below zero against the
  no-fit floor, at the player-game unit, on the **draftable season-start-roster**
  population; **and (2)** the rolling-origin harness on the fitting half agreeing — same
  sign, interval below zero, majority of origins won.
- **Ship rule**: a head that clears ships fitted; a head that fails ships its **no-fit
  floor estimator as the plug-in**. Every unit must carry all eleven quantities to be
  scorable at all, so the gate decides fitted-vs-floor, never scored-vs-unscored.
  Expectation set by P4(b): `fta`/`tov`/`stl` may well ship the floor.
- **Program-level settling gate (Session 5, §16's own)**: the head family against the
  floor family at the **season-total dk_pts unit** on the true-rookie population,
  validation seasons.
- **Acceptance (Session 7)**: rookie *and* ladder-recovered units present and draftable
  on retro and forward boards; the §6h population-held-fixed stability readings unchanged
  for units the ladder did not touch.

## 5. Design

### 5a. Session 1 — price the floor (contest units)

What rookie-lessness costs: the field drafts the **unrestricted** board, our seat stays
masked to scorable rows, and the gap to the audited symmetric-restricted run is the floor.

- A variant run of `src/sim/strategy.py` using the non-default artifact suffix
  (`strategy.py:1469-1474`) → `_rookiefloor` artifacts; the audited CSVs stay untouched.
- Skip `priceable_room` for the **field side only**; our seat keeps its `scorable` mask
  (`strategy.py:861`, `draft_room.py:675`).
- Honest scoring for unrestricted picks: the **realized replay** is the primary readout
  (all players have realized dk_pts in validation seasons). The simulated-world arm scores
  unscorable rows as literal zero, which overstates the field's handicap — either fill
  those rows with realized totals or report replay only, and say which.
- Readout: Round-1 `lift_vs_null` delta vs the symmetric run, both validation seasons,
  paired where the machinery allows. New target `make rookie-floor`; result recorded here
  and in `dashboard/decisions.py`. **Run 2026-08-22 — see §7a.** The lift readout came back
  unresolved as §3 decision 1 allowed for, so the session also measured the floor at the
  unit that does resolve: the Round-1 cut line the opponent field sets.
- Context to carry: a rookie head arrives as **new board rows** (order-changing), which
  is the more promising case than the shape-only changes §7l measured as contest nulls.

### 5b. Session 2 — the veteran design's lag-recovery ladder

**This session runs before the rookie head is built, because it decides how much of the
hole the rookie head is left to serve.** Measured in §7b; the design below is what that
measurement selected.

`component_rates.build_design` asks for one thing — a lag-1 season of at least
`MIN_PRIOR_MINUTES` — and drops everything else before any modelling decision is taken.
The ladder replaces that single test with three rungs, all of them **design-time
imputation into the existing lag-1 columns**, so the fitted heads score them unchanged
(§3 constraint 4).

- **Rung 0 — qualified.** `total_minutes_lag1 >= 200`. Untouched, bit-identical to today.
- **Rung A — returnee.** `total_minutes_lag1` missing, a usable season at lag 2 or 3.
  `with_lags` pairs on season *index*, deliberately, so the row exists and is simply never
  built at `max_lag=1`; the ladder raises the builder to `max_lag=3` and fills lag-1 from
  the nearest usable season.
- **Rung B — thin prior.** `total_minutes_lag1` present and under the threshold. The rate
  is shrunk toward the fitting-population mean by its own reliability weight
  `m / (m + k)`, `k` in prior-season minutes and fitted per head on the fitting half.
  **The shrink is not optional and §7b is emphatic about it**: raw thin lag-1 scores a
  pooled R² of −1.1451 and is an anti-model, against +0.8566 shrunk.

Rungs A and B are the same estimator once stated generally — *the nearest usable season,
shrunk by its own reliability* — and §7b measures that unified form too, so the builder
implements one rung and the three-way split is a readout rather than three code paths.

- **Module**: extend `src/models/component_rates.py` rather than forking it, with the
  ladder behind a config key so the shipped design is recoverable by flipping it. The
  measurement module `src/models/lag_recovery.py` (`make lag-recovery`) already carries
  `classify`, `recent_lag` and `recent_shrunk_rate`; the builder imports them rather than
  restating them.
- **A `lag_source` column travels with every row** — the lag it was recovered from, and
  its reliability weight — because a scored unit whose provenance is not recorded cannot
  be audited on a board, and Session 7's acceptance test reads it.
- **The threshold discontinuity is real and is named rather than solved.** A player at 201
  prior minutes gets a raw rate and one at 199 gets a shrunk one; at the fitted `k` the
  jump in weight is about 0.2. Removing it means shrinking everyone continuously, which
  changes the *fitting* population and costs eleven refits — the escalation §3 constraint
  4 names, not this session's job.
- **Gate**: §4's ladder rung gate, per rung, on validation. **Run 2026-08-22 — see §7c.**
- **What it could reach** (§7b): 47 of 101 unserved board rows in 2022-23 and 51 of 105 in
  2023-24, including 7 and 6 of the ADP-priced ones.
- **What the gate actually admitted** (§7c): rung A's full-lag-2 returnee alone — **10 and
  4** board rows, but **7 of 16 and 3 of 21 ADP-priced** ones, which is every priced row
  the ladder could have recovered in 2022-23 and half of them in 2023-24. Reach and
  admission are different quantities; rungs B and the two tails are long-tail roster rows a
  draft rarely reaches, and they failed the band half of the gate.

### 5c. Session 3 — the rookie design builder and the no-fit floors

New module `src/models/rookie_rates.py` (`python -m src.models.rookie_rates`,
`make rookie-rates`; conventions: config idiom, package-path imports for pickled classes).
**Built and measured 2026-08-22 — see §7d**, which records two mechanical departures from
the wording below: `has_preseason` is not a twelfth feature (it is `1 − Σ(missing
indicators)` and would make the block rank-deficient), and `undrafted` is the slot block's
reference cell rather than a fifth indicator, so an undrafted rookie is that block's exact
zero.

- **Population**: player-seasons whose **target season is their first played season** —
  no NBA history at any lag, so no own-rate feature is constructible. The complement of
  `component_rates.build_design`'s qualified set *after* the §5b ladder has taken every row
  it can serve. Disjoint from the veteran design by construction (a test pins this).
  Fit on all such rows; gate and quote on the season-start-roster draftable subpopulation.
  Covered window 2004-05 onward.
- **Feature ladder** (all rungs zero-recovering under the Stan L2-at-zero prior):
  - **Volume-shrunk preseason rate on the head's own link scale**, centered against the
    fitting-population mean — a rookie has no prior season to difference against, so the
    delta convention becomes level-vs-population on the same link scale. Counts:
    `log1p(pre_per36)`; conversions: EB-shrunk `logit((pre_made + k·league)/(pre_att + k))`
    (the `carry_forward_conversion` device pointed at preseason attempts). Reliability
    weight `min_pre/(min_pre + k)` via `minutes_preseason.reliability_weight`; k fitted on
    an inner carve of the fitting half, P4(b)'s selections as the starting grid.
  - `has_preseason` + the four age-split missing indicators (reused, not forked).
  - **Draft slot** (`team_context.DRAFT_BUCKETS`; undrafted is an indicator, never an
    imputed number), **years-since-draft**, and the **slot × years-since-draft
    interaction**. Forward-safe by construction (slot is roster text for an unplayed
    season). Expectations set by P4(b): slot alone is an anti-model; it earns its place
    in interaction or not at all.
  - `age`, `age_sq`.
  - ~~Variant arm: shrunk thin-prior lag for sub-200-minute returnees who have one.~~
    **Moved to §5b rung B**, where it belongs: a player with a thin lag has an own-rate
    feature and the veteran heads are fitted on that feature's scale.
- **The population is now homogeneous, and that is the point of the change.** Every row
  the head fits has exactly the same information — preseason, slot, age, and nothing else —
  so the fitted coefficients describe one regime instead of averaging over three. It is
  also smaller and its features are better covered: preseason reaches 90.8% of played true
  rookies against 71.5% of returnees (§7b).
- **No-fit floor per head** = P4(b)'s shrunk estimator (expanding draft-bucket mean
  blended with preseason per-36 at fitted k), wrapped in NB at a train-fitted φ /
  beta-binomial at a fitted ρ so CRPS is comparable — mirroring
  `stan_components.count_floor` / `conversion_floor`.
- Targets from `component_targets.parquet` (rookies' realized rows exist; only the design
  excluded them). DK scoring is never reimplemented.

### 5d. Session 4 — Stan arms and the gate

**Run 2026-08-22 — see §7e**, which records the ladder's shortened shape (the scale rung
does not exist, because a rookie head's own feature is already on its link) and the dense
mass matrix that made the rolling half affordable in Stan.

- All eleven heads: 7 NB counts with minutes exposure, `fg3a|fga` + 3 conversions
  beta-binomial, on the two existing compiled sources. Ladder mirrors
  `count_variants`/`conversion_variants`: linear → +interaction → +spline.
- Metrics via the existing scorers (`val_r2`/`val_nll`/CRPS/PIT); artifact
  `outputs/predictions/rookie_rate_metrics.csv` in the shape of
  `stan_component_metrics.csv`, with `selected` and `beats_floor` head-local. §7e adds a
  `population` column and the gate block to that shape, because §4 reads the verdict on the
  draftable subpopulation and one row per (head, variant) could carry only one population.
- Run the §4 gate per head; record fitted-vs-floor ship decisions here and in the
  registry. Sampler scale: eleven small-population fits — minutes each, nothing like the
  veteran sweeps. **396 fits in about fourteen minutes, zero convergence failures.**

### 5e. Session 5 — the season-total readout (§16's settling gate)

- A rookie-admitting frame builder alongside `season_total.build_frame` (the current
  frame drops the population twice over on lag columns). GP treatment = the shipped
  graded `no_design_availability` level; rate side = the rookie heads vs the floor
  family.
- Add **two** groups to `season_total.evaluate`'s group tuple, not one: `rookie` for the
  true-rookie population Session 3's head serves, and `lag_recovered` for the rows §5b's
  ladder admitted. They are different populations served by different heads and pooling
  them would report an average of two regimes — the mistake §3 constraint 2′ exists to
  undo. `compare` stays split-agnostic (one code path — the held-out unlock is spent and
  stays spent).
- Readout: season-total dk_pts MAE/CRPS, head vs floor, validation seasons, recorded
  here.

### 5f. Session 6 — persistence and the simulator

- New group `"rookie-components"` in `posteriors.GROUPS`; one `PosteriorArtifact` per
  head with a `DesignRecipe` whose `builder` is `src.models.rookie_rates.head_design`;
  floor-shipping heads persist a deterministic recipe, constant across draws (the honest
  plug-in shape). Roundtrip-gated like every artifact. Fit at `train_val` **and** `full`
  (`make posteriors WINDOW=… --groups rookie-components`); `src/production_check.py`
  green afterwards.
- `src/sim/season.py::build_context`: `units` becomes the **union** of the veteran and
  rookie designs (disjoint populations) with a per-unit family indicator;
  `component_rates(artifacts, units)` branches by family. `row_scorable`, `row_flat`,
  the tensor and everything downstream are unchanged. Tests pin disjointness + union
  coverage; a seeded spot-check shows existing veteran units bit-identical.
- Drafting code needs no structural change — more rows become scorable; `priceable_room`
  drops fewer; the `scorable` masks already do the right thing.
- **§5b's rows need no new artifact at all**, which is the dividend of the imputation-only
  form: they are veteran-head units with a wider design, so they ride the existing
  `components` posterior group. Only the design recipe's builder changes, and the
  roundtrip gate covers it. `lag_source` travels as a unit column so a board can say which
  rung scored a player.
- **Turning the ladder on is this session's first act, and it is two edits, not one.**
  `stan.components.lag_ladder: [returnee_lag2]` (§7c's verdict), and every path that
  *fits* rather than scores must take its training frame through
  `component_rates.fitting_rows` — `posteriors.component_artifacts`, `stan_components.run`,
  `season_terms`, `components_preseason`. `windowed()` splits on season and knows nothing
  about rungs, so wiring `head_design` without the second edit would silently widen the
  fitting population and breach §3 constraint 4. The bit-identity test §7c ships checks
  the design; a seeded veteran-unit spot-check is what checks the artifacts.

### 5g. Session 7 — forward wiring, through the 2026-27 board

- `forward_design.forward_rookie_design`: rows for an unplayed season from the roster
  snapshot — draft slot via the existing `forward_draft_numbers` text tier (note its 25
  rights-traded-draftee misbuckets), preseason columns from the target season's panel
  **when it exists**. Before the October fetch the block is all-missing and the head
  degrades to slot + indicators — that is the missing-indicator path working, not a
  special case. Wire into `forward_board.forward_context` as a fifth injected frame.
- **The id-map blocker**: DK board rows for never-played players carry a negative
  surrogate id (`draft_pool.py:266-271`) because `adp_draftkings.build_id_map` matches
  only against the season matrix. Add a **roster-snapshot reference tier**, revisit
  `adp.NON_DEFECT_METHODS`, and pin with a test (board name → snapshot row → real
  PLAYER_ID; the AJ-Dybantsa shape).
- ~~**The ladder is forward-safe without new work and the doc should say why**: a
  returnee's lag-2 and a thin-prior's lag-1 are both prior-season NBA statistics, which is
  exactly what `forward_design` already assembles for veterans. Rung A and rung B need no
  forward-specific tier — only the true-rookie head does, which is the whole of the
  id-map blocker above.~~ 🔴 **Wrong, and §7h says how.** The features need no new tier and
  the *builder* did: `forward_component_design` reaches `component_rates.build_design`
  directly, whose default is the pre-ladder design, so the forward board dropped every
  recovered returnee until `ladder=lag_ladder(cfg)` was passed.
- Acceptance per §4; `make forward-board` on 2026-27 shows draftable rookies **and**
  lag-recovered returnees.

### 5h. Session 8 — closeout

**Run 2026-08-22 — see §7i.** The readout was taken on the half of §7a's floor that
resolves rather than as three more sweeps, and the wording below is what it was scheduled
against.

- Optional readout: symmetric rookie-inclusive sweep replay (suffix mechanism) — how much
  of Session 1's floor the two changes recover, reported **separately for the ladder and
  the rookie head**, since they land in different parts of the board: the ladder recovers
  7 and 6 ADP-priced rows against the rookie head's 9 and 15 (§7b). *Taken as `make
  rookie-recovery`, a nested ladder of board masks — the ladder's admitted reach is 7 and 3
  rather than §7b's 7 and 6, because §7c admitted one rung of four.*
- `docs/final-evaluation-plan.md`: record that the held-out figures measured the
  rookie-less workflow and do not transfer to the rookie-inclusive one.
- docs-audit coverage for every figure this doc quotes from its own artifacts;
  `make dashboard-audit` green; `docs/model-cards-plan.md` check — whether the new heads
  get model cards and what `chain_role` they declare.

## 6. Session runbook — the prompts

One prompt per session, for a fresh session each. Sessions 2-8 end by appending results
here and updating `dashboard/decisions.py` where a ship decision lands.

**One interleaved session sits between 5 and 6 and is not part of this program's eight.**
§7f found a defect on the *availability* head — it drops exactly the rows §5b's ladder
recovers, and the plug-in that catches them hands every returning player in the league one
scalar. It is scheduled as `docs/availability-window-plan.md` §16 and it runs **before**
Session 6, because Session 6 is what wires these rows into the simulator and would otherwise
wire them in with 24.7 games each. The rookie sessions keep their numbers; nothing here is
renumbered.

1. ✅ **Done 2026-08-22 — §7a.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 1 (§5a): price the rookie floor** —
   > the asymmetric-board realized-replay run with the `_rookiefloor` artifact suffix and
   > a new `make rookie-floor` target. Record the figure in the plan doc and the
   > decisions registry. Do not start building the head.
2. ✅ **Done 2026-08-22 — §7c.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 2 (§5b): the veteran design's
   > lag-recovery ladder** — raise `component_rates.build_design` to `max_lag=3`, fill a
   > missing or thin lag-1 from the nearest usable season shrunk by its own reliability
   > (importing `lag_recovery`'s helpers, not restating them), behind a config key, with
   > a `lag_source` column on every row. Gate each rung per §4 on validation. **Imputation
   > only — no head is refitted.** Tests for the rung boundaries, the threshold
   > discontinuity, and veteran rows coming back bit-identical.
3. ✅ **Done 2026-08-22 — §7d.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 3 (§5c): the rookie design builder
   > and no-fit floors** — `src/models/rookie_rates.py` for the TRUE-ROOKIE population
   > only (first played season = target season), with the feature ladder per the doc plus
   > the eleven dispersion-wrapped floor estimators. Tests for disjointness from the
   > post-ladder veteran design, shrinkage weight, and zero-recovery. No Stan fits yet.
4. ✅ **Done 2026-08-22 — §7e.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 4 (§5d): fit the eleven rookie
   > arms and run the stated gate** — validation paired-bootstrap CRPS vs floor plus
   > rolling origins, write `rookie_rate_metrics.csv`, record fitted-vs-floor ship
   > decisions in the doc and registry.
5. ✅ **Done 2026-08-22 — §7f.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 5 (§5e): the season-total
   > readout** — rookie-admitting frame, `rookie` AND `lag_recovered` groups in
   > `season_total.evaluate`, head-vs-floor at the season-total unit on validation. This
   > is §16's settling gate; record it.

✅ **Done 2026-08-22 — `docs/availability-window-plan.md` §16i, not a session of this
program.** One rung of three admitted (`returnee_lag2`), imputation only, and §7f's open
item measured at 543.3167 → 305.7897 draftable season-total MAE. The prompt below is §16h
verbatim; keep the two in step if either is edited.

   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/availability-window-plan.md and execute **§16: the returnee gap** — raise
   > `availability.build_design` to recover a missing `gp_share_lag1` block from the nearest
   > usable season (importing `lag_recovery`'s helpers and `component_rates`' rung
   > vocabulary, not restating them), behind a config key, with `lag_source` / `lag_rung` /
   > `lag_minutes` on every row. Run **both** arms of §16e — imputation-only scored by the
   > shipped posterior first, then the refit with a staleness column at `train_val` and
   > `full` — against §16f's gate on the draftable and `all` populations, with rung 0 as the
   > bar. Tests for the rung boundaries and for rung 0 coming back bit-identical. Record the
   > result there and in the decisions registry, add the quotable figures to
   > `make docs-audit`, and re-run `make season-total-rookie` so §7f of
   > docs/rookie-rates-plan.md gets its number.

6. ✅ **Done 2026-08-22 — §7g.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 6 (§5f): persistence and simulator
   > integration** — the `rookie-components` posterior group at `train_val` and `full`,
   > deterministic recipes for floor-shipped heads, the `build_context` units union with
   > per-family branching, veteran units bit-identical spot-check, `production_check`
   > green.
7. ✅ **Done 2026-08-22 — §7h.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 7 (§5g): forward wiring** —
   > `forward_rookie_design`, the ADP id-map roster-snapshot tier with its test, rookie
   > and lag-recovered rows on `make forward-board`, and the §4 acceptance checks.
8. ✅ **Done 2026-08-22 — §7i.**
   > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 8 (§5h): closeout** — the optional
   > symmetric rookie-inclusive sweep replay against Session 1's floor, the
   > final-evaluation-plan note, docs-audit and dashboard-audit green, model-cards check.

## 7. Results

### 7a. Session 1 — the rookie floor, priced in contest units (run 2026-08-22)

**`make rookie-floor`** (`src/sim/strategy.py --field-board unrestricted`) runs the shipped
sweep asymmetrically: the opponent field drafts **every** rostered player, our seat stays
masked to the rows the tensor prices. Artifacts carry the `_rookiefloor` suffix plus the
floor's own `strategy_rookie_floor.csv`; the audited `strategy_*.csv` set is untouched, the
same discipline `--field` uses.

**The two arms are the same code on the same worlds, and that is checked rather than
assumed.** Gate C is computed on the scorable population in *both* modes, so
`strategy_gate_c_rookiefloor.csv` must reproduce `strategy_gate_c.csv` exactly — it does, to
**0.000e+00 across all 12 rows**, and the injection records agree on `rho`, `scale_g` and
`achieved_mae` to the digit. That check is what licenses reading the symmetric side off the
2026-08-16 artifact instead of paying an hour to re-run it;
`docs/availability-window-plan.md` §7l's lesson — *a baseline of unknown vintage is not a
baseline* — is answered by a measurement, not by trust. `rookie_floor_table` raises if the
file is absent and the readout says outright that it means nothing if Gate C ever diverges.

#### The board, and who is on the wrong side of it

| season  | field board | our seat | unpriceable | of those, ADP-priced | field takes /entry | entries holding ≥1 |
|---------|------------:|---------:|------------:|---------------------:|-------------------:|-------------------:|
| 2022-23 | 347 → **448** | 347 | 101 | 16 | 1.2556 | 73.06% |
| 2023-24 | 359 → **464** | 359 | 105 | 21 | 1.1861 | 74.72% |

The population is not marginal and it is not only rookies (**scratch measurement,
2026-08-22** — realized totals joined to the board outside `make rookie-floor`, so these
figures are not in the audit). The rows our seat can never take
include **Victor Wembanyama — 3,244 realized dk_pts at ADP 21.9** — with Chet Holmgren
(2,720, ADP 56.3) and Brandon Miller (1,952, ADP 124) in 2023-24; and Paolo Banchero (2,445,
ADP 63.0), **Jamal Murray (2,234, ADP 72.2)** and **Kawhi Leonard (1,917, ADP 24.3)** in
2022-23. The last two are returnees rather than rookies, which is §5b's one-design-for-the-
whole-population argument arriving as data rather than as a preference. On the ADP-priced
subset those rows realize **1,398.6** (2022-23) and **1,173.3** (2023-24) season dk_pts
against **1,271.8 / 1,284.5** for the board we *can* price: the players we are structurally
unable to draft are, on average, better than the ones we can.

Every figure below this line comes out of `make rookie-floor` and is audited
(`src/docs_audit.py::_rookie_floor`), except the two blocks explicitly marked scratch.

#### 1. The bar the field sets — the half of the floor that resolves

Round 1 is a 2-of-12 cut in all five structures, so the bar is the 10/12 quantile of the
field's entries, scored on the season that actually happened.

| season  | field drafts the priceable board | …the whole board | delta |
|---------|---------------------------------:|-----------------:|------:|
| 2022-23 | 15,505.8 | 15,678.9 | **+173.1** |
| 2023-24 | 15,393.5 | 15,505.4 | **+111.9** |

**Letting the field have the players we cannot price raises the bar our entries must clear
by 112 to 173 dk_pts**, roughly +0.7% to +1.1% of the cut. This is averaged over 1,200
field entries per board, so unlike everything below it, it resolves. It is also the quantity
a rookie rate head would actually recover: the head does not lower the bar, it lets our seat
reach the same players the bar is made of.

#### 2. The contest lift — the half that does not resolve

The requested readout: Round-1 `lift_vs_null` on the realized replay, asymmetric minus
symmetric, paired on the season and on the one world that season was.

| tournament | 2022-23 | 2023-24 | pooled |
|------------|--------:|--------:|-------:|
| 15k_and_one      | +0.0952 | +0.0205 | +0.0579 |
| 20k_spin_move    | +0.0430 | +0.0889 | +0.0659 |
| 50k_four_pt_play | +0.1349 | −0.1649 | −0.0150 |
| 600k_shootaround | +0.1035 | −0.0680 | +0.0178 |
| *88k_alley_oop*  | *−0.0065* | *+0.6851* | *reported, never pooled* |

The shipped arm `lineup_value_blend30` averages **+0.0316** over the eight multi-entry
readings, which run from **−0.1649 to +0.1349** and lose lift in 2 of 8. `88k_alley_oop` is
excluded from every pooled figure because it is a **single $450 entry**: its symmetric
realized reading is 0.9940 in 2022-23 against 0.0654 in 2023-24, so one roster's coin flip
would otherwise set the headline.

Across the **whole 24-arm table** (4 multi-entry structures × 2 seasons each) the sign
reverses: **21 of 24 arms lose lift, median −0.0765**, range −0.2033 (`blend_a70`) to
+0.1385 (`lineup_value`). The shipped arm is one of only three that gain. There is no
resolved ordering in `alpha` behind that spread — `blend_a70` is worst but `blend_a85` and
`blend_a50` are middling — so it is dispersion, not a mechanism.

**So the contest half does not resolve, and §3 decision 1 already assumed it would not.**
Two realized seasons is one world each; the module says so about its own headline edge and
it is no more true here.

#### Why the contest half is so noisy — the decomposition

Three channels, and only one of them is small.

1. **The field gets better**: +173.1 / +111.9 dk_pts on the cut, as above.
2. **The board around our seat changes**: the field spends ~1.2 picks per entry on players
   we could never take, so across eleven opponents roughly thirteen *scorable* players
   survive to us who otherwise would not. Scratch measurement, 10 entries at 600k on the
   realized world: our own Round-1 total moves **+234.5** (2022-23) and **−41.4** (2023-24)
   for the shipped arm, **+666.3 / −454.0** for a pure-ADP entry of ours. That is one draw
   per season and it swamps channel 1 by three to five times.
3. **Our own ranking is *not* reordered** — the one channel that could have been a
   measurement artifact, and it is not. `value_rank` is unchanged on the scorable rows
   (unscorable rows carry dk_pts identically 0 and sort below everyone), and while
   `adp_rank` shifts by 0–16 / 0–21 places, the resulting blend key correlates with its
   restricted twin at Spearman **≥ 0.99990** at every `alpha` from 0 to 1, with 49–50 of
   the top 50 shared. Scratch measurement.

Channel 2 is real — a real field really does spend picks on rookies — but at N = 2 it is a
coin flip, and it is why the lift table straddles zero while the cut line does not.

#### The simulated arm, and why it is not the readout

`strategy_sweep_rookiefloor.csv` is uniformly positive: **+0.0603** mean for the shipped
arm, 0 of 8 readings negative, all 24 arms positive. That is the artefact §5a predicted in
advance — in the simulated world an unscorable row is a **literal zero**, so an unrestricted
field spends 1.2 picks per entry on players who score nothing and hands us a handicap of the
tensor's making. It confirms the caveat rather than adding evidence, and the realized replay
is the readout. `make rookie-floor` prints this in the run header.

#### What Session 1 settles, and what it does not

- **Settled**: the floor is not zero and it is not small at the unit that resolves. The
  field's Round-1 bar rises **+173.1 / +111.9 dk_pts** when it may draft what we cannot,
  and the ADP-priced part of that population out-realizes our own board.
- **Settled**: the two arms are the same code on the same worlds (Gate C to 0.000e+00), and
  our seat's ranking is not distorted by the wider board (Spearman ≥ 0.99990) — so the
  measurement is about the board and nothing else.
- **Not settled**: what it is worth in Round-1 advance probability. 21 of 24 arms lose
  (median −0.0765) while the shipped arm gains +0.0316 on readings spanning −0.1649 to
  +0.1349. Two realized seasons cannot separate those, exactly as §3 decision 1 anticipated.
- **Unchanged**: the program proceeds. Decision 1 is *price first, build regardless*, and
  the floor is context for what the head is worth, not a gate on building it. Session 2
  (§5b) is next.

### 7b. The design reversal — where the veteran design's boundary belongs (2026-08-22)

**`make lag-recovery`** (`src/models/lag_recovery.py` → `outputs/predictions/lag_recovery.csv`).
Fits no head: every arm is `component_rates.carry_forward`'s own functional form (prior
per-36 × realized minutes ÷ 36) scored as R² against the realized count, which puts each
number on the same scale as the shipped no-fit floor's published **0.81–0.95** band. Nine
rate targets, not all eleven — the four beta-binomial heads are a different metric and they
inherit whatever population decision this settles rather than informing it, so their gate
stays Session 4's job. Train+validation only; the test split is spent and stays spent.

#### Why one test was hiding three populations

`build_design` asks for a lag-1 season of at least `MIN_PRIOR_MINUTES`, and
`with_lags` pairs on season *index* — deliberately, so that "a player who missed a year"
cannot silently pair across the gap. Together those two make one filter do three different
jobs. Split apart, over played player-seasons in the window:

| group | played seasons | what own-rate signal exists |
|---|---:|---|
| veteran | 9,403 | lag-1, qualified |
| thin prior | 891 | lag-1, under threshold |
| returnee (usable lag-2) | 177 | a full season, one year staler |
| returnee (thin lag-2) | 75 | a thin season, one year staler |
| away 2+ seasons | 118 | lag-3 |
| **true rookie** | **2,176** | **none at any lag** |

Counted on train+validation, which is why each row equals the `n` its score is reported on
below — the test seasons are not ours to count either.

#### The measurement

Mean R² over the nine rate targets:

| arm | train | validation | pooled | n pooled |
|---|---:|---:|---:|---:|
| veteran lag-1 — today's design | 0.9081 | 0.8989 | 0.9080 | 9,403 |
| **returnee lag-2 — rung A** | 0.9108 | **0.8976** | **0.9157** | 177 |
| thin prior, raw lag-1 | −1.2555 | 0.2165 | **−1.1451** | 891 |
| **thin prior, shrunk lag-1 — rung B** | 0.8575 | **0.8020** | **0.8566** | 891 |
| **any history, nearest lag shrunk** | 0.8601 | **0.8845** | 0.8630 | 1,261 |

Three things follow, and each one moved a decision.

**A returnee's two-year-old rate is not degraded.** It lands at 0.8976 on validation
against the veteran floor's 0.8989 — a difference of 0.0013 — and inside the published
band. This is consistent with the project's own architecture rather than surprising: rates
are the persistent quantity and availability is the volatile one, so a player who lost a
season to injury comes back with his per-minute profile intact. The right treatment is to
carry it forward, not to model him as though he had never played.

**The thin-prior shrink is the whole of rung B.** Raw thin lag-1 is not a weak
estimator, it is an **anti-model** at −1.1451 pooled — a hundred noisy minutes carried
forward onto a full season is worse than knowing nothing. Shrunk by reliability it reaches
0.8566. The fitted `k` runs 25 (`fg3a`, `ast`) to 300 (`tov`) prior-season minutes, per
head, on the fitting half — the same ordering P4(b) found for preseason, where the
low-volume counts need the most shrinking.

**One rung serves the whole has-history population**, at validation 0.8845 pooled over
1,261 rows. The two weakest subgroups sit below the band and are named rather than hidden:
thin lag-2 returnees at 0.7813 pooled (n = 75) and players away two or more seasons at
0.7497 (n = 118). Both are 1–2 board rows per season, so the ladder admits them with the
same estimator and Session 2's gate decides per rung.

#### Against the alternative: what the rookie head would have given these players

On the 108 returnee rows that have a preseason row, their own lag-2 against P4(b)'s
volume-shrunk preseason per-36 at k = 160 — the estimator §5c's head would serve them
with:

- mean R² **0.8977** for lag-2 against **0.8092** for shrunk preseason, and **lag-2 wins
  7 of 7 heads**. The gap is widest on `blk` and `stl`, the low-volume counts where a
  ~68-minute preseason sample carries almost nothing.
- and preseason is *missing more often for exactly this group*: **71.5%** coverage against
  **90.8%** for played true rookies. Returnees are frequently still rehabbing in October.

So a design serving returnees with preseason and draft slot would be discarding their best
feature and would be missing its own replacement feature nearly a third of the time. That
is what withdrew §3 constraint 2.

#### What each change recovers on a real board

Counted on `draft_pool.parquet`, the same file the drafting layer builds its board from.
**The census reconciles exactly with the drafting layer** — 347 and 359 rows served today,
and 16 and 21 ADP-priced rows unserved, which are `strategy_injection.csv`'s own figures
arrived at from the opposite direction.

| | 2022-23 rows (ADP-priced) | 2023-24 rows (ADP-priced) |
|---|---:|---:|
| served today | 347 | 359 |
| rung B — thin prior | 27 (0) | 35 (3) |
| rung A — returnee, usable lag-2 | 10 (7) | 4 (3) |
| rung A — returnee, thin lag-2 | 1 (0) | 2 (0) |
| rung A — away 2+ seasons | 9 (0) | 10 (0) |
| **left to the rookie head** | **54 (9)** | **54 (15)** |

The ladder recovers **47 of 101** unserved rows in 2022-23 and **51 of 105** in 2023-24 —
and **7 of 16** and **6 of 21** of the ADP-priced ones, which is the part a draft actually
reaches. The rookie head keeps the true rookies: half the rows and, in 2023-24, most of
the draftable ones.

#### What this does not settle

The arms above are carry-forward power, not the §4 gate. A rung is admitted on a
validation paired-bootstrap CRPS interval at the player-game unit, which is Session 2's
job; this measurement chose the *shape* of the ladder and sized its population, and a rung
that fails its gate ships nothing rather than shipping on the strength of these R²s. The
conversion heads are unmeasured here by design. And the imputation-only form carries a
known discontinuity at 200 prior minutes (§5b), which is priced only if the refit form is
ever escalated to.

### 7c. Session 2 — the ladder built, and its gate run per rung (2026-08-22)

**`make lag-ladder`** (`src/models/lag_ladder.py` → `outputs/predictions/lag_ladder.csv`),
against the ladder `component_rates.build_design` now carries behind
`stan.components.lag_ladder`. §7b chose the ladder's *shape*; this is §4's gate deciding
which rungs ship, and it came back **one rung of four**.

#### What was built

`build_design(..., ladder=...)` raises `with_lags` to `max_lag=3` and fills an unusable
lag-1 block from the nearest usable season, **into the same `_lag1` columns the eleven
shipped heads already fit coefficients on**. Rates are shrunk by the source season's own
reliability weight (`lag_recovery.recent_shrunk_rate`, imported rather than restated);
conversion percentages are empirical-Bayes shrunk on their own *attempts*, the form
`carry_forward_conversion` settled, at a `k` fitted on the fitting half and persisted
beside the rate constants; volumes and context (`mpg_lag1`, `total_minutes_lag1`,
`gp_lag1`, every `{c}_lag1`) are carried **raw**, because they are not estimates of a
per-minute quantity — they are what the player's most recent season actually was.

Every row carries `lag_source` (`lag1`/`lag2`/`lag3`), `lag_rung` (§7b's own group
vocabulary, so a board reconciles with the census directly) and `lag_minutes`. The
per-head reliability weight is `lag_minutes / (lag_minutes + k)` at that head's `k` in
`lag_recovery.csv` — one row-level column and nine constants rather than nine redundant
columns, which is the only place this departs from §5b's literal wording.

`stan.components.lag_ladder` is a **list of admitted rungs**, not a boolean, because the
gate is per rung and came back per rung. `[]` rebuilds the pre-ladder design exactly —
`max_lag=1`, no provenance columns, the same 200-minute test — and
`tests/test_lag_ladder.py` pins rung 0 coming back **bit-identical**, NaN for NaN, on
every shared column. That test is the whole of §3 constraint 4: the recovered rows are
scored by coefficients fitted before the ladder existed, which is only legitimate if the
rows those coefficients were fitted on did not move. `component_rates.fitting_rows` is the
same statement as code, for the fitting paths Session 6 will thread it through.

The ladder reaches **1,350** rows the shipped design drops, taking it from **10,194** to
**11,544**; **910** of those are validation rows, split 773 veteran / 100 thin-prior /
19 returnee-with-a-full-lag-2 / 11 returnee-with-a-thin-lag-2 / 7 away-two-seasons.

#### The gate fits nothing, and that is the claim under test

The eleven heads are read off `data/features/posteriors/train/` and score the recovered
rows with the coefficients they already have. `train` ends at 2021-22, so the validation
seasons are outside it. Nothing here is fitted, which is what "imputation only, no head is
refitted" has to mean if it means anything.

**Gate 1 — against the unserved status quo.** A row missing from the design is missing
from the tensor, so today it contributes a literal zero to every draw: its predictive is a
point mass at 0 and its CRPS is `|y − 0| = y`. Deltas are averaged over the eleven heads
below, which mixes units (`fga` in attempts, `blk` in blocks) — read the column as a
direction and the head count as the result.

| rung | n | head CRPS | unserved | mean delta | heads with the interval below zero |
|---|---:|---:|---:|---:|---:|
| veteran (rung 0, the bar) | 773 | 13.1114 | 154.0600 | −140.9486 | **11 of 11** |
| thin prior (rung B) | 100 | 4.5890 | 32.1362 | −27.5471 | **11 of 11** |
| returnee, full lag-2 (rung A) | 19 | 7.3274 | 95.7589 | −88.4315 | **11 of 11** |
| returnee, thin lag-2 | 11 | 2.5457 | 13.1506 | −10.6049 | 10 of 11 (`fta` reaches +0.2376) |
| away 2+ seasons | 7 | 4.4994 | 29.4134 | −24.9141 | 10 of 11 (`blk` reaches +2.6648) |

**Gate 2 — inside the shipped floor's published band.** `carry_forward` on the ladder's
own imputed column, which *is* `recent_shrunk_rate` by construction.

| rung | fga | fta | reb | ast | stl | blk | tov | count-head mean | 9-rate mean | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--|
| veteran | 0.951 | 0.876 | 0.950 | 0.919 | 0.839 | 0.810 | 0.897 | **0.8919** | **0.8989** | in |
| thin prior | 0.954 | 0.573 | 0.930 | 0.697 | 0.781 | 0.747 | 0.828 | **0.7873** | 0.7999 | **OUT** |
| returnee, full lag-2 | 0.980 | 0.839 | 0.990 | 0.900 | 0.963 | 0.522 | 0.961 | **0.8791** | 0.9013 | in |
| returnee, thin lag-2 | 0.981 | 0.093 | 0.858 | 0.755 | 0.863 | 0.307 | 0.248 | **0.5866** | 0.6402 | **OUT** |
| away 2+ seasons | 0.975 | 0.859 | 0.951 | 0.760 | 0.432 | −0.632 | 0.839 | **0.5977** | 0.6748 | **OUT** |

**The gate is the mean, and the per-head cells are diagnosis — because the per-head
reading rejects the shipped design itself.** On the same 773 veteran rows only **5 of 7**
count heads sit inside [0.81, 0.95], `fga` (0.951) and `reb` (0.950) being *above* it. A
test the incumbent fails is not a test of anything: the published band summarizes a pooled
per-head spread, not a tolerance every head sits in on every subpopulation. So the gate is
§4's own domain read as one number — the rung's mean over the count heads — with rung 0
carried alongside as the bar. The veteran row is also a **cross-check on the whole
pipeline**: its 9-rate mean of 0.8989 reproduces §7b's audited veteran validation figure
to the digit, from a different module on a different frame.

#### The verdict

| rung | gate 1 | gate 2 | admitted |
|---|---|---|:--|
| thin prior (rung B) | 11 of 11 | 0.7873, below the band | **no** |
| returnee, full lag-2 (rung A) | 11 of 11 | 0.8791, in band | **YES** |
| returnee, thin lag-2 | 10 of 11 | 0.5866 | no |
| away 2+ seasons | 10 of 11 | 0.5977 | no |

`stan.components.lag_ladder: [returnee_lag2]` is the shipped setting, and it is **inert
until Session 6** wires `head_design` and routes the fitting paths through
`fitting_rows` — the ladder is built and gated here, not turned on. Nothing downstream
changes this session, which is why the key is left at `[]` in `configs/default.yaml` with
the verdict recorded in the comment; Session 6 flips it as its first act.

Three things this settles, and each is a smaller answer than §5b expected.

**Rung A is the rung, and it is the ADP-priced half.** §7b sized the ladder's *reach* at
47 of 101 unserved board rows in 2022-23 and 51 of 105 in 2023-24. The gate admits **10
and 4** of those — but **7 of 16 and 3 of 21 ADP-priced rows**, which is every priced row
the ladder could ever have recovered in 2022-23 and half of them in 2023-24. Rung A was
always where the draftable players were (Jamal Murray at ADP 72.2, Kawhi Leonard at 24.3
are both this shape); rungs B and the two tails are long-tail roster rows that a draft
rarely reaches. Reach and admission are different quantities and §5b's line now says which
is which.

**Rung B fails on the count heads and was already borderline in §7b.** Its 0.7873 is not a
surprise reading: §7b's own thin-prior-shrunk validation figure was 0.8020 over the nine
rate targets, itself under the band's lower edge, and the count-head restriction §4 names
removes the three attempt columns (`fga`, `fg2a`, `fg3a`) the project has always predicted
best. What sinks it is `fta` (0.573), `ast` (0.697) and `blk` (0.747) — a hundred noisy
minutes shrunk toward the mean is a working estimator for volume and not yet one for the
low-rate counts. It is the largest rung by rows (996 in the design), so it is also the one
worth re-opening if the escalation §3 constraint 4 names is ever taken: a continuous shrink
that admitted these rows to the *fit* would let the head learn the slope this imputation
has to assume.

**Half the hole stays open, and it is the long-tail half.** After rung A, **37 of 101**
unserved 2022-23 board rows and **47 of 105** in 2023-24 are served by neither the ladder
nor the rookie head — 27/35 thin-prior, 1/2 thin-lag-2 returnees, 9/10 away two seasons.
Only **3** of those 84 rows carry an ADP at all (all in 2023-24), which is the reason to
leave them rather than to force a rung through its own gate.

**The two tails fail on both halves and should stop being discussed as a rung.** Eleven and
seven validation rows produce intervals that reach across zero on a head each, and
count-head means of 0.5866 and 0.5977 with `blk` at −0.632 for the away-two-seasons group —
an anti-model on that head. §7b admitted them to the measurement "with the same estimator"
and said the gate would decide; it decided.

#### What this does not settle

The gate is at the **season-total unit**, not §4's stated player-game unit, and that is a
departure with a reason rather than a shortcut: these heads are season-collapsed by
construction, so a player-game predictive does not exist without inventing a per-game
dispersion nothing has fitted. Every component-head figure in the project is at this unit,
which is the only way a number here is comparable to `stan_component_metrics.csv`.

Nothing has been measured about what rung A is worth **in contest units** — that is
Session 8's optional replay, and Session 1's floor is the thing it would be read against.
And the 200-minute discontinuity is still there by design: at the fitted `k` a player at
201 prior minutes gets a raw rate and one at 199 gets a shrunk one, a weight jump of about
0.2, pinned by a test rather than smoothed.

### 7d. Session 3 — the rookie design and its eleven no-fit floors (run 2026-08-22)

**`make rookie-rates`** (`src/models/rookie_rates.py` → `outputs/predictions/rookie_rate_floors.csv`).
The population §7c left over, given a design and a benchmark. **No head is fitted** — four
constant blocks are estimated on the fitting half and written to the artifact, and Session 4
(§5d) is what puts a sampler on it.

#### The population, and that it is disjoint by construction

**1,495** true-rookie design rows over the covered window (2004-05 → 2025-26), and **zero**
of them share a `(player_id, season)` with the **11,544**-row veteran design built at *every*
ladder rung — admitted or not, because the claim is about where the boundary is rather than
about which rungs happen to ship today. `run` asserts it on every invocation rather than
trusting it, since Session 6 makes the simulator's scorable units the **union** of the two
families and a player in both would enter the tensor twice.

Disjoint is not exhaustive and the doc should not read as though it were: a player whose only
prior season falls outside `max_lag=3` is in neither family, which is the long-tail half §7c
left open. `tests/test_rookie_rates.py` pins that shape too — the rookie head must not
quietly absorb him.

| | train | of those draftable | validation | of those draftable |
|---|---:|---:|---:|---:|
| rows | 1,218 | 961 | 146 | 108 |
| carry a preseason row | 91.4% | **97.1%** | 93.8% | **99.1%** |
| median preseason minutes | 70.5 | 81.2 | 65.3 | 70.4 |

The draftable restriction is doing exactly what P1 decision 5 said it would: on the whole
population a missing preseason is mostly a January signing, and on the season-start-roster
population — the only one a board ever prices — it nearly disappears. Draft slot on the
training half runs 90 top-5 picks, 162 lottery, 282 late-first, 347 second-round and **337
undrafted**, which is why `undrafted` is the slot block's reference cell rather than a fifth
indicator: it is the largest single cell, so the L2 prior's mode and the population's mode
are the same row.

#### What the design carries, and the two departures from §5c's wording

Sixteen features per head: the head's own volume-shrunk preseason **level**, four age-split
missing indicators, four slot indicators, years-since-draft, four slot × years-since-draft
interactions, and `age`/`age_sq`. Every block is **zero-recovering** — exactly 0, not
approximately, where its information is absent — which is checked on the validation frame at
the end of every run and pinned by test:

- 9 validation rows have no preseason reading, by either route (no panel row at all, or a
  panel row with no minutes), and their eleven shrunk levels are 0 **by their own volume
  weight** rather than by a special case;
- 43 validation rows are undrafted, and their nine slot columns are 0 — including
  years-since-draft, which is forced to zero for an undrafted player even where the roster
  matrix carries a stray draft year.

Two things depart from §5c as literally written, both mechanically rather than as
preferences:

1. **`has_preseason` is not a twelfth feature.** It is `1 − Σ(missing indicators)` exactly —
   the four partition the missing rows, which is their own docstring's claim — so adding it
   makes the block rank-deficient against the intercept while breaking the missing-side
   coding that keeps zero meaning "no information". It stays on the frame as a population
   column and out of every head's feature list.
2. **`undrafted` is the slot block's reference cell** rather than a fifth indicator, for the
   zero-recovery reason above. §5c's "undrafted is an indicator, never an imputed number" is
   honoured in the sense `team_context.DRAFT_BUCKETS` means it: he is a category, not a 61st
   pick.

The **level, not a delta**, is the substantive change and it is what the population forced.
Every veteran head fits the preseason as a difference against the prior season; a rookie has
no prior season, so the same column becomes level-against-the-fitting-population-mean on the
same link. Centring is not cosmetic — uncentred, `log1p(pre_per36) = 0` says "he did nothing
in October" and would be indistinguishable from "nobody measured him", which is the exact
conflation the missing indicators exist to prevent.

#### The constants, all fitted on the fitting half

| block | value |
|---|---|
| volume shrink `k`, counts | **160** preseason minutes |
| volume shrink `k`, conversions | **10** preseason minutes |
| centring means (`fga`, `reb`, `ast`, `blk`) | 2.4755, 1.8623, 1.1363, 0.4294 |
| conversion EB `(k` pseudo-attempts`, league)` | `fg3a` (2.07, 0.2754), `fg2m` (107.93, 0.4789), `fg3m` (401.32, 0.3346), `ftm` (34.89, 0.7282) |

`k` is chosen per **family** on an inner carve of the fitting half — the last two training
seasons scored against everything before them — on standardized CRPS, which is
`rookie_priors`' own construction and for its own reason: the carve scores ~120
player-seasons and eleven heads each picking a rung off that many rows is fitting the grid.
The count family lands on exactly P4(b)'s selected `k = 160`, reached independently at a
different unit on a different population, and the curve is a clean interior minimum
(0.72140 at `k = 0`, 0.20216 at 160, 0.22233 at the `draft_bucket` endpoint). The conversion
family is nearly flat from 0 to 20 and takes 10.

Each head's **own** optimum is written to the artifact and **reported, never selected on**:
`ast` 40, `reb`/`fga` 80, `blk`/`fta` 160, `tov` 320, and **`stl` at the grid's right
endpoint** — for `stl` the preseason carries nothing at all and the bucket mean is the best
available prior, which is P4(b)'s null arriving again from a different direction.

#### The floors, validation · draftable (n = 108, except `fg3m|fg3a` 106 and `ftm|fta` 105)

Three arms of one estimator: `draft_bucket` is the incumbent
(`stan_composition.rookie_share_priors`' own device pointed at a rate), `preseason` is the
blend at `k = 0`, and `shrunk` is the floor.

| head | bucket R² | preseason R² | **floor R²** | floor CRPS | bucket CRPS |
|---|---:|---:|---:|---:|---:|
| `fga` | 0.9127 | 0.8585 | **0.9531** | 28.9885 | 37.7341 |
| `reb` | 0.7274 | 0.7868 | **0.8449** | 25.5257 | 34.0359 |
| `ast` | 0.6132 | 0.6563 | **0.8378** | 14.3171 | 21.3704 |
| `tov` | 0.7611 | 0.5682 | **0.8082** | 9.6004 | 10.2502 |
| `fta` | 0.6750 | 0.2643 | **0.7668** | 18.0836 | 20.0037 |
| `stl` | 0.7871 | −0.5164 | **0.7851** | 4.6727 | 4.8078 |
| `blk` | 0.2966 | 0.7019 | **0.6711** | 7.1199 | 9.7919 |
| `fg3a\|fga` | −0.5989 | 0.6109 | **0.5743** | 17.8601 | 40.8617 |
| `fg3m\|fg3a` | −0.0415 | −0.0963 | **−0.0388** | 3.3429 | 5.8696 |
| `ftm\|fta` | −0.0441 | 0.0120 | **0.0027** | 3.2919 | 3.7188 |
| `fg2m\|fg2a` | −0.1649 | −0.0356 | **−0.0508** | 7.9526 | 9.1621 |

**The floor beats the incumbent on CRPS on 11 of 11 heads**, and on R² on 10 of 11 — the
exception is `stl`, which loses by 0.0020 and is one of P4(b)'s three declared nulls. That is
the shape a benchmark should have: it is better than what ships today for this population on
every head, which is what makes "the head must clear the floor" a real bar rather than a
formality.

Three readings follow, and the third is the one that changes what Session 4 expects.

**P4(b)'s anti-model finding reproduces at the head's own unit.** The draft bucket alone is
negative on all four conversion heads (−0.5989 to −0.0415) and reaches only 0.2966 on `blk`.
P4 decision 4 — *"if he is ever put in, his preseason per-36 is the prior to use and a draft
bucket is not"* — was measured on per-36 rates at the estimator level; here it is measured on
season totals through a fitted likelihood, on the population the head actually fits, and it
holds.

**The raw preseason arm is unusable at the CRPS unit, and that is what the shrink is for.**
Its point estimates are respectable (`fga` R² 0.8585, `blk` 0.7019) but its NB dispersion
**pins at the optimizer's lower bound of 0.050 on 5 of 7 count heads** — `fga` sits just off
it at 0.0634 and `reb` at 0.0962 — so the predictive is as wide as the parameterization
allows. `fga`'s CRPS is **207.0254** against the floor's 28.9885 and its PIT KS **0.8200**
against 0.0578: the point prediction is fine and the *distribution* is not, which is the
half of a predictive a season-total R² cannot see. A boundary hit is a caveat rather than a
measurement — the exact CRPS is a bound on how bad the arm is, not a number — but the
direction is not in doubt, and it is §2's "raw preseason per-36 is worse than the anti-model
on 6 of 8 targets" arriving as a calibration failure rather than as an R².

**Three conversion heads are already near-nulls and one is not.** `fg2m|fg2a`, `fg3m|fg3a`
and `ftm|fta` sit within ±0.06 R² of zero for *every* arm — shooting percentage is the least
persistent quantity in the box score and a rookie has no prior season to carry it forward
from — so §5c's expectation that some heads ship their floor is, if anything, understated:
those three have very little for a fitted head to beat, and beating a floor that is itself
near-zero is not the same as scoring the head well. `fg3a|fga` is the opposite case and the
one to watch: the **share** of attempts taken from three is a stable, coachable quantity, the
preseason reads it well (0.6109), and the incumbent is actively harmful (−0.5989).

#### What Session 3 settles, and what it does not

- **Settled**: the two families are disjoint by construction and the assertion runs every
  time, which is the property Session 6's `units` union needs.
- **Settled**: the design exists, is zero-recovering on every block, and its constants are
  fitted on the fitting half and read from the artifact rather than pinned in a module.
- **Settled**: the floor is a real bar. It beats the shipping incumbent on CRPS on 11 of 11
  heads and the count family independently re-selects P4(b)'s `k = 160`.
- **Not settled**: whether a fitted head beats it anywhere. That is Session 4's gate, on 108
  draftable validation rows — a small population, and the interval widths on it are the
  reason §4 pairs validation with a rolling-origin confirmation on the fitting half.
- **Not measured**: anything about the four conversion heads' *fitted* prospects. Three of
  them have a floor near zero on every arm; a head that clears a near-zero floor has cleared
  very little, and Session 4 should read `beats_floor` beside the level rather than alone.


### 7e. Session 4 — the eleven arms fitted, and the gate that admitted one (run 2026-08-22)

**`make stan-rookie`** (`src/models/stan_rookie.py` → `outputs/predictions/rookie_rate_metrics.csv`).
§7d built the design and eleven no-fit floors and fitted nothing; this puts a sampler on it
and runs §4's per-head ship gate. **One head of eleven ships fitted: `reb`.** The other ten
ship §7d's floor estimator as a plug-in — which is a decision about *which arm*, never about
whether the head ships, because a unit must carry all eleven quantities to enter the tensor
at all.

#### What was fitted

Three variants per head, §5d's ladder with its scale rung removed: the veteran ladder walks
raw → log → spline because a veteran head's own feature is a prior-season rate that must be
put on the link scale first, and a rookie head's own feature is **already** on it (§7d's
level, not a delta). So the rungs are `linear` (the shrunk level, four missing indicators,
four slot indicators, years-since-draft, age/age_sq — 12 features), `slot_interaction`
(**§5c's shipped list exactly**, 16), and `slot_interaction_spline` (the level's B-spline
basis replacing its linear term, 21). Each is built by *subtraction from*
`rookie_rates.head_features` rather than restated, so a rung cannot drift from the list
Session 6 will persist.

**The gate is the conjunction §4 stated, and both halves are computed as code.** Validation
is the selected variant's paired-bootstrap CRPS interval against the floor on the 108
draftable rows; the rolling half walks 11 origins over the fitting half (2011-12 → 2021-22,
≥ 300 fitting rows each, 629 draftable rows scored) and asks for the same sign, an interval
below zero, and a majority of origins won. Unlike `make components-preseason`, **the rolling
half is Stan rather than a point-MLE stand-in** — at this population size it costs minutes,
so both halves are the same estimator against the same floor.

Two mechanical points worth recording:

- **`metric=dense_e`, and it is a 30× speedup rather than a preference.** The slot block's
  four products are collinear with their own indicators by construction, and `age`/`age_sq`
  add a second such pair. On `fga` the shipped diagonal metric saturated treedepth on
  **793 of 4,000** draws and took 90.5 s; `dense_e` saturates **0**, takes 3.0 s, and lands
  on the same answer (R² 0.9633 against 0.9631, CRPS 22.82 against 22.71). All **396** fits
  ran with **zero** convergence warnings — no R-hat above 1.01, no ESS below 400, no
  divergence — in about fourteen minutes.
- **Both arms fit the same 1,161 rows**, not the design's 1,218. The floor's dispersion needs
  an expanding bucket prior and 2004-05 has none, so handing the sampler 57 rows the floor
  cannot have would have made this a comparison of two fitting populations.
- **The spline rung is the one that is not literally zero-recovering**, and it is named
  rather than hidden: a B-spline basis at a level of exactly 0 is a *constant* vector, not a
  zero one. The four missing indicators partition exactly the rows with no preseason reading
  and absorb that constant, so no row's prediction depends on a level nobody measured — but
  the column is not zero and §7d's blanket claim does not extend to it.

#### The ladder, validation · draftable (n = 108, except `fg3m|fg3a` 106 and `ftm|fta` 105)

CRPS then its paired delta against §7d's floor; `*` marks an interval entirely below zero;
**bold** is the variant `selected` picked (R² for a count, NLL for a conversion —
`_finalize`'s rule, unchanged).

| head | floor | linear | + slot × yrs | + spline |
|---|---:|---:|---:|---:|
| `fga` | 28.7864 | **27.7258** (−1.0606) | 27.8000 (−0.9863) | 27.9197 (−0.8667) |
| `fta` | 18.1564 | 16.9755 (−1.1809) | 16.9849 (−1.1715) | **16.7022** (−1.4542) |
| `reb` | 25.4441 | 23.6888 (−1.7553\*) | 23.7146 (−1.7294\*) | **22.5571** (−2.8869\*) |
| `ast` | 14.1020 | 13.2504 (−0.8515) | 13.4451 (−0.6569) | **12.9649** (−1.1370) |
| `stl` | 4.6717 | **4.2233** (−0.4484) | 4.2753 (−0.3964) | 4.3672 (−0.3045) |
| `blk` | 7.1148 | 6.3973 (−0.7176) | **6.4654** (−0.6494) | 6.8952 (−0.2196) |
| `tov` | 9.4824 | 8.7768 (−0.7056) | 8.9735 (−0.5089) | **8.6699** (−0.8125) |
| `fg3a\|fga` | 17.9441 | 18.4253 (+0.4811) | 18.5143 (+0.5701) | **18.0213** (+0.0772) |
| `fg2m\|fg2a` | 7.9819 | 7.8057 (−0.1762\*) | 7.8399 (−0.1420\*) | **7.7869** (−0.1950\*) |
| `fg3m\|fg3a` | 3.3672 | **3.4257** (+0.0584) | 3.4328 (+0.0656) | 3.4232 (+0.0560) |
| `ftm\|fta` | 3.3054 | 3.3323 (+0.0268) | **3.3332** (+0.0278) | 3.3453 (+0.0399) |

The spline is the selected rung on **6** heads, `linear` on 3 and `slot_interaction` on 2 —
so P4(b)'s expectation that draft slot "earns its place in interaction or not at all" comes
back *not at all* on nine of eleven heads.

#### The gate

| head | selected | validation delta [95%] | rolling delta [95%] | origins | ships |
|---|---|---:|---:|---:|:--|
| `fga` | linear | −1.0606 [−2.5734, +0.4606] | −0.2366 [−0.9897, +0.5090] | 7/11 | floor |
| `fta` | +spline | −1.4542 [−2.8534, +0.0573] | −0.3919 [−1.1009, +0.3892] | 6/11 | floor |
| **`reb`** | **+spline** | **−2.8869 [−5.0460, −0.7950]** | **−2.4642 [−3.8744, −1.0523]** | **9/11** | **FITTED** |
| `ast` | +spline | −1.1370 [−3.0730, +0.5660] | −0.9014 [−1.7982, +0.0280] | 8/11 | floor |
| `stl` | linear | −0.4484 [−1.1565, +0.1271] | −0.1882 [−0.4204, +0.0505] | 7/11 | floor |
| `blk` | + slot × yrs | −0.6494 [−1.9045, +0.2934] | +0.3094 [−0.0608, +0.7130] | 4/11 | floor |
| `tov` | +spline | −0.8125 [−1.7536, +0.1287] | −0.5426 [−0.9532, −0.1326] | 9/11 | floor |
| `fg3a\|fga` | +spline | +0.0772 [−0.6119, +0.8766] | +0.7197 [−0.0565, +1.8455] | 2/11 | floor |
| `fg2m\|fg2a` | +spline | −0.1950 [−0.3797, −0.0254] | +0.3365 [+0.1120, +0.6336] | 3/11 | floor |
| `fg3m\|fg3a` | linear | +0.0584 [−0.1656, +0.2858] | −0.0529 [−0.1318, +0.0242] | 8/11 | floor |
| `ftm\|fta` | + slot × yrs | +0.0278 [−0.1538, +0.2363] | +0.0764 [−0.0304, +0.1999] | 5/11 | floor |

**Two heads clear validation, two clear rolling, and the intersection is one.** That is what
a conjunction is for, and this round is the cleanest illustration the project has produced:
the two halves disagree about *which* head, not merely about how much.

#### Five readings, and the second is why the rolling half exists

**Every count head points the right way and only one of them resolves.** All seven have a
favourable validation CRPS delta (−0.4484 to −2.8869) and all seven beat the floor on R²;
`beats_floor` is true on **8 of 11** selected arms. The direction is unanimous and the
evidence is thin — 108 rows — which is exactly the shape §7d predicted when it said the
interval widths on this population are the reason §4 pairs validation with a rolling
confirmation.

**`fg2m|fg2a` is the head the rolling half was put there to catch.** Its validation interval
is entirely below zero — −0.1950 [−0.3797, −0.0254], a clean win — and its rolling interval
is entirely below zero *on the wrong side*: **+0.3365 [+0.1120, +0.6336]**, with **3 of 11**
origins won. A validation-only gate would have shipped a fitted head that 1,161 fitting-half
rows say is worse than arithmetic. `docs/availability-window-plan.md` §12e and §14f record
blocks that won validation and *shrank* 4-6× rolling; this one does not shrink, it reverses.

**`tov` is the mirror image and the head to revisit first.** Rolling −0.5426 [−0.9532,
−0.1326] with **9 of 11** origins is as clear as `reb`'s, and validation −0.8125 [−1.7536,
+0.1287] reaches across zero on 108 rows by a margin smaller than the point estimate. `ast`
is the other near miss, with a rolling upper edge of **+0.0280**. Neither ships, and neither
should on the bar §4 set in advance — but both are failures of *power* rather than of
direction, and the population is the thing that would fix them.

**§7d's three near-nulls are confirmed as nulls, and the head it said to watch is the
worst.** `fg3m|fg3a`, `ftm|fta` and `fg3a|fga` all have *positive* validation deltas — the
fitted head is worse than the floor, not merely not better — which answers §7d's caution
("beating a floor that is itself near-zero is not the same as scoring the head well") more
sharply than it was posed: they do not beat it. `fg3a|fga` was the opposite case §7d flagged
to watch, because the share of attempts taken from three is stable and coachable and the
preseason reads it at R² 0.6109. It is the **worst of the eleven** on the fitting half:
+0.7197 rolling, **2 of 11** origins. The share is readable by an *estimator* and is not
improved by giving a GLM the same information plus draft slot.

**The fitted arms are better calibrated than the floors on 7 of 11 heads**, including heads
that fail the gate: `fta` PIT KS 0.2641 → 0.1919, `tov` 0.2456 → 0.1393, `stl` 0.1252 →
0.0700, and `reb` 0.1044 → **0.0459**. That is a real property of the fitted family and it is
not what the gate reads, so it changes nothing about what ships — but it is the reason the
count heads are worth re-opening when the population grows rather than being written off.

#### What `reb` actually buys

Validation · draftable: CRPS **25.4441 → 22.5571**, R² **0.8449 → 0.8781**, PIT KS
**0.1044 → 0.0459** — better on level, spread and calibration at once, and the only head
where all three move together with an interval to support it. On the fitting half it wins
**9 of 11** origins at −2.4642 [−3.8744, −1.0523].

Why this head and not another is an interpretation rather than a measurement, but the
obvious candidate is in §7d's own table: rebounding is one of only **three** count heads
whose *raw* preseason arm already beats the draft bucket on R² (0.7868 against 0.7274;
`ast` and `blk` are the others), so the level the spline bends is one that carries real
signal before any shrink is applied. What that does not explain is why `ast` and `blk` did
not follow, and nothing here settles it.

#### What Session 4 settles, and what it does not

- **Settled**: the design fits. Eleven heads, three variants, 396 fits, zero convergence
  failures, and a dense metric that makes the rolling half of the gate affordable in Stan
  rather than in a point-MLE stand-in.
- **Settled**: the ship split. **`reb` fitted at `slot_interaction_spline`; ten heads on
  §7d's floor.** Session 6 persists ten deterministic recipes and one posterior.
- **Settled**: the draftable restriction is not what is blocking the other ten. On the wider
  `all` population (146 validation rows) the verdict is **the same single head** — `fta` and
  `fg2m|fg2a` clear validation there and still fail rolling, `tov` and `fg3m|fg3a` clear
  rolling there and still fail validation.
- **Not settled**: whether `tov` and `ast` are real. Both fail on power rather than
  direction, and the honest reading is that 108 draftable rows cannot separate a −0.8 CRPS
  effect from zero. Nothing here licenses shipping them; §5h's closeout is where a wider
  population or a refit form would be argued for.
- **Not measured**: anything at the season-total dk_pts unit, which is §16's own settling
  gate and Session 5's (§5e) job. A per-head CRPS win of −2.9 rebounds is not yet a
  statement about a draftable player's season total, and a family that ships ten floors and
  one fitted head has to be read against the floor family *as a family*.


### 7f. Session 5 — the season-total readout, and what §16's own gate answers (run 2026-08-22)

**`make season-total-rookie`** (`src/models/season_total_rookie.py` →
`outputs/predictions/season_total_rookie.csv`). Sessions 3 and 4 measured the family in
rebounds and made threes; this composes all eleven heads into the number a board is drafted
on and asks §16's own question. It answers two, and **they point in opposite directions**:
the family is worth **520 dk_pts of season-total MAE** against the hole it fills, and the
*fitted* arm is worth **nothing measurable** against the floor arm.

#### What was built

`season_total.build_frame` cannot carry this population and drops it **twice over on lag
columns** — the availability design is lag-1 and has no row for a player with no prior
season, and `RATE_FEATURES` then wants three more lag-1 columns. So `season_total_rookie.py`
is that builder's rookie-admitting twin, and it shares exactly one thing with it: the
scorer. `season_total.evaluate`'s group tuple now carries `veteran`, `lag_recovered` and
`rookie` off a `population` column, `rotation_mask` returns all-False on a frame that has no
lag block rather than raising, and a treatment may supply its own **rate** and its own
predictive **samples** — which is what lets one scorer serve a ladder that varies games and
a readout that varies the rate family. The availability ladder's own artifact
(`season_total_metrics.csv`) comes back **bit-identical** after the change.

**Three rate arms per group, each the family's own version of the same thing.** `unserved`
is what a row scores today — it is not in the design, so it is not in the tensor, and every
draw scores it at zero; `floor` is §7d's eleven dispersion-wrapped preseason blends for the
rookie family and `carry_forward` / `carry_forward_conversion` for the veteran one; `head`
is what ships — §7e's split read out of `rookie_rate_metrics.csv`'s own `ships` column, and
the persisted `train`-window posteriors for the veteran family. **One head is fitted here**
(`reb`, because no rookie posterior exists until Session 6); everything else is read off
disk. The `lag_recovered` group is the rungs §7c **admitted**, read from `lag_ladder.csv`'s
verdict rows rather than from `stan.components.lag_ladder`, which is still deliberately `[]`.

The chain is `season_terms.compose_season_dk`'s: counts from their heads, `fg3a` drawn on
the **drawn** `fga`, `fg2a` as the difference, makes on drawn trials. The bonus is a
per-game threshold that a season total cannot carry, so each arm gets its own
`expected_bonus` at `BONUS_OVERDISPERSION` — the constant calibrated at exactly this unit,
a player-season's mean per-game counts — constant across draws, the same honest shape as the
availability plug-in. It is **0.66%** of the realized season total on these rows, so the
target is `dk_pts` outright rather than its linear part.

The games-played side is the shipped treatment per row, and which one a row gets is a
measured fact rather than a choice: the availability design covers **773 of 773** veteran
rows and **0 of 165** rookie and returnee rows, so `no_design_availability`'s graded
`tenure_draft` level is the treatment for exactly the population §5e names it for. Every arm
is read at that level **and** at `oracle_gp`, which is `season_total`'s own oracle device and
is what separates a rate-family error from an availability error.

#### The table, validation · draftable (MAE then CRPS, dk_pts, lower is better)

| arm | veteran (706) | lag_recovered (14) | rookie (108) |
|---|---:|---:|---:|
| `unserved` — today | 1393.4919 · 1393.4919 | 1062.4643 · 1062.4643 | 739.1505 · 739.1505 |
| floor family | 284.6165 · 228.8490 | 543.3167 · 510.9946 | **218.8826 · 179.4336** |
| head family | 280.5043 · 228.2814 | 550.0307 · 521.4856 | **219.2185 · 180.4075** |
| floor · oracle GP | 118.8089 · 85.4694 | 64.4025 · 47.1678 | 91.1718 · 67.3352 |
| head · oracle GP | 106.3415 · 76.5554 | 63.6246 · 44.9040 | 88.7267 · 64.5268 |

On the unrestricted population (773 / 19 / 146) the rookie column reads 579.1849 unserved,
195.2182 floor, 195.3603 head, and 70.0754 for the head at oracle GP.

#### The gate §4 stated, paired, on the draftable rows

| group | games treatment | CRPS delta [95%] | MAE delta |
|---|---|---:|---:|
| veteran | shipped | −0.5676 [−5.5140, +4.4607] | −4.1122 |
| veteran | oracle GP | **−8.9141 [−12.3735, −5.3473]** | −12.4674 |
| lag_recovered | shipped | +10.4909 [−3.4576, +26.7757] | +6.7140 |
| lag_recovered | oracle GP | −2.2638 [−7.9743, +3.8861] | −0.7780 |
| **rookie** | **shipped** | **+0.9738 [−3.3629, +5.3300]** | **+0.3358** |
| **rookie** | **oracle GP** | **−2.8084 [−6.0123, +0.2479]** | **−2.4451** |

**§16's settling gate does not resolve, and that is the result.** The rookie head family is
not measurably better than the rookie floor family at the season-total dk_pts unit on either
games treatment: the interval spans zero at the shipped level and reaches +0.2479 at
`oracle_gp`. It is also not measurably *worse*. That is exactly what one fitted head of
eleven should be worth at a unit where the other ten arms are literally identical between
the two families — the head arm differs from the floor arm in `reb` and in nothing else —
and it is §7e's verdict arriving one level down rather than a new finding.

#### Four readings, and the first is the number §16 asked for

**The family is worth ~520 dk_pts of season-total MAE, and the choice of arm is worth
nothing.** A draftable rookie scores **739.1505** MAE today, because he is not in the tensor
and every draw prices him at zero; the floor family takes that to **218.8826** and CRPS from
739.1505 to **179.4336**. The ladder's returnees move **1062.4643 → 543.3167**. Those two
numbers are the answer to *"is a rate head for this population worth building"* and they are
three orders of magnitude larger than the fitted-vs-floor gap the per-head gate spent
Session 4 on. **Which arm a head ships is a small decision inside a large one**, and §4's
ship rule — the gate decides which arm, never whether — is what makes it so.

**The availability plug-in, not the rate family, is what limits the recovered returnees.**
Their draftable MAE falls from **543.3167 to 64.4025** the moment games played is an oracle,
an 8.4× reduction, because `no_design_availability` gives them **24.738** games against a
realized **46.857**. The plug-in is not malfunctioning — `make availability-no-prior` §8b
measured that an uncovered returnee's realized rate collapses toward 0.30 whatever his draft
bucket says, and 24.7/82 is 0.30 to the digit. What it is, is estimated on a *different*
population: it pools every uncovered returnee, most of them fringe roster rows, while rung A
is specifically the player who missed a whole season having played a full one before it —
Jamal Murray and Kawhi Leonard, who then play 47 games. **The ladder recovered these rows'
rate side and their availability side is still being served by a level built for someone
else.** That is the largest single thing this session found and it is a Session 7 item, not
a rookie-head item: the rate arms are already at 63-64 MAE on a 1062 dk_pts season.

**The rookie plug-in, by contrast, is doing its job.** 42.715 predicted games against 44.852
realized on the draftable rows, and the oracle only takes the floor family from 218.8826 to
91.1718 — a 2.4× reduction against the returnees' 8.4×. `no_design_availability`'s graded
level was selected on the no-prior population and it is the one place in this table where
the plug-in is scoring the population it was measured on.

**The head-vs-floor gap does not resolve for the SHIPPED population either, and that reframes
the rookie row.** The veteran family's eleven fitted heads beat their own no-fit floor by
**−8.9141 [−12.3735, −5.3473]** CRPS at `oracle_gp` — a clean win — and by **−0.5676
[−5.5140, +4.4607]** at the shipped games treatment, which is no win at all. The same
availability noise that hides the rookie family's arm choice hides the veteran family's, on
706 rows instead of 108. This is README §3's "availability is the largest measured win"
arriving from a direction nothing had taken it from before: at the season-total unit the
games-played treatment dominates the rate family *even where the rate family is eleven
fitted Stan heads*. A reader who takes the rookie row as evidence that the rookie design is
weak has to explain why the veteran row says the same thing.

#### The cross-check, and what it is worth

The veteran `head · oracle GP` cell on the unrestricted 773 rows is MAE **101.0178**, CRPS
**72.7774**. `season_term_season_total.csv`'s `base` arm is the same composition on the same
rows from a different module, with heads it refits itself rather than the persisted ones and
with the bonus excluded: MAE 105.710, CRPS 76.055. Two independent paths to the same number
within 4%, and the residual is the direction the bonus and the persisted-versus-refitted
posteriors would move it. The chain is composing what it says it is composing.

#### What Session 5 settles, and what it does not

- **Settled**: §16's gate, in the sense that matters. The rate family is worth **520.27
  dk_pts** of draftable season-total MAE on true rookies and **519.15** on the ladder's
  returnees, against the zero they score today. The program's mandate is discharged.
- **Settled**: the fitted-vs-floor question at the deliverable unit is **null**, on both
  games treatments, and Session 4's per-head split is not overturned by it. Ten of the
  eleven arms are identical between the two families by construction.
- **Settled**: the readout is one scorer. `season_total.evaluate` serves both ladders, its
  group tuple never pools `rookie` with `lag_recovered`, and `season_total_metrics.csv`
  reproduces bit-identically after the change.
- **Not settled**: whether the fitted `reb` arm is worth anything downstream. 108 rows
  cannot separate a −2.8 CRPS effect from zero, and §5h's contest replay is where a
  season-total effect this size would or would not become a bracket effect.
- **Measured 2026-08-22 by `docs/availability-window-plan.md` §16i, and it is worth about
  half the prize.** That round gave the availability head its own lag-recovery ladder and
  admitted the rung these rows sit on, so they are scored by the head instead of by
  `no_design_availability`: predicted games go **24.7376 → 49.6575** against a realized
  46.8571, and the `lag_recovered` draftable season-total MAE falls **543.3167 → 305.7897**
  (CRPS 510.9946 → 249.8303) against the same unmoved oracle floor of 64.4025 — **49.60% of
  the gap**. `make season-total-rookie-lagladder` is that arm and writes its own
  `_lagladder` artifact; `season_total_rookie.csv` re-runs bit-identical, and every
  `oracle_gp`, `veteran` and `rookie` cell is unchanged to the digit, which is the check
  that the ladder moved only the population it was built for. The residual 241.3872 dk_pts
  is not availability-recoverable on fourteen rows.

#### Why every recovered returnee gets the same number of games, and the fork that follows

Not a bug and not a tie: `availability_no_prior.level_keys` builds the shipped
`tenure_draft` key as `f"rookie__{bucket}"` for a first appearance and the bare string
**`"returning"`** for everything else, so the draft bucket is dropped for a returnee by
construction and one pooled scalar per season is applied to all of them. The 14 draftable
rows get **24.8** games in 2022-23 and **24.6** in 2023-24 — Kawhi Leonard, Jamal Murray and
Miles Bridges alongside Armoni Brooks — against realized values from 10 to 70. §8b of
`make availability-no-prior` chose that collapse on evidence (the bucket is a gradient for a
rookie and a flat for a returnee, and crossing it in would average the two), but it chose it
on the **pooled** uncovered-returnee population, which is mostly fringe roster churn. Rung A
is the opposite shape: a full season played, one missed, a rotation role waiting.

**The rate side of these rows is already right**, which is what makes the games side the
whole story. Predicted dk_pts per game played against realized, on the same 14: Bridges 40.3
vs 38.6, Leonard 42.9 vs 41.9, Williamson 42.2 vs 43.6, Wiseman 19.1 vs 19.7, Gallinari 10.8
vs 11.0, Thompson 10.1 vs 10.1. That is §7b's imputed lag-2 doing exactly what it was
measured to do.

**The availability head has the identical structural gap and no ladder, and it is closer to
free than the component one was.** `availability.build_design` already calls
`with_lags(..., max_lag=3)`, so `gp_share_lag2/lag3` and `minutes_per_game_lag2/lag3` are
already on the row; `FEATURE_COLS` already **fits coefficients on all six**, so the head
already reads two- and three-year-old availability for a covered veteran. The row is dropped
by `dropna(subset=["gp_share_lag1", ...])` — on lag-1 **alone** — and the three lines under
it already fill lag2/lag3 *from* lag1 when those are missing. The mirror fill is simply not
written, and that asymmetry is the entire gap.

*Scratch measurement, 2026-08-22, not in `make docs-audit`*: replacing the flat plug-in with
the ladder's own carried `gp_lag1` (capped at team games, no shrink, no fit) takes the
`lag_recovered` draftable season-total MAE from **543.32 to 342.39** against the oracle's
64.40 — about **42%** of the gap — and the games error from 24.2 to 14.1. Correlation between
the carried lag and realized games is only **+0.352** on the 14 rows (+0.624 on all 19), so
most of the residual is not carry-forward-recoverable.

**Which is why the fork here does not resolve the way the component one did.** §3 constraint
4 gated the cheap imputation-only form first because for a *rate* head the nearest real rate
is a good estimate of this year's rate. For an *availability* head **the missing season is
the signal**, and imputing lag-1 from a healthy lag-2 tells the head "he played 80 games last
year" — the single most misleading thing available about Kawhi Leonard in 2022-23. The
absence block goes with it: `trailing_missed_lag1`, `n_spells_lag1`, `longest_spell_lag1` and
`ABSENCE_MIX_COLS` all describe S-1 and would be imputed away too. So the arm worth measuring
is probably the **refit** form — the recovered rows admitted to the fit with a staleness
column the head can learn a slope for — which the component program declined on cost at
eleven heads and two windows. **The availability head is one head.** That asymmetry is the
argument, and it is a measurement for `docs/availability-window-plan.md` to take, not this
program.

**Scheduled there as §16 on 2026-08-22 and run the same day, ahead of Session 6** — because
the rows it recovers are the ones §5f is about to wire into the simulator, and it would
otherwise have wired them in at 24.7 games each. §16i is the readout. **The fork resolved
the opposite way to the expectation above**: the refit-with-a-staleness-column arm was built
and priced and does not beat the imputation (+0.7310 [−1.4152, +2.8585] on the draftable
rows), because it learns its discount from the pooled recovered population and over-applies
it to the draftable one — the plug-in's own defect, one level up. The imputation-only arm
ships at one rung of three, `returnee_lag2`, exactly as §7c's component ladder did on the
same vocabulary. `n_prior_seasons` now counts real seasons rather than depth, and
`lag_interior_gaps` sees the 130 `(1,0,1)` rows.

### 7g. Session 6 — the family persisted, the ladder turned on, and a board with rookies on it (run 2026-08-22)

**`make posteriors --groups rookie-components` at all three windows**, plus two edits that
turn §7c's admitted rung on and one that makes the simulator's scorable units the union of
the two rate families. Nothing is measured here: this session is **plumbing**, and its
result is that a 2023-24 board now contains Victor Wembanyama.

#### The ladder, on — and it is two edits because one of them is the guard

`stan.components.lag_ladder: [returnee_lag2]` is §7c's verdict, and it reaches the heads
through `stan_components.head_design` alone — never through the bare `build_design` that
twelve modules call. The design goes from **10,194** rows to **10,383**, which is
`lag_ladder.csv`'s own `returnee_lag2` census of **189** arriving where the heads can see
it.

The second edit is the one §3 constraint 4 depends on: `selection_split` and
`windowed` split on **season** and know nothing about rungs, so every path that *fits*
rather than scores takes its training frame through `component_rates.fitting_rows` first.
Three do — `stan_components.run`, `posteriors.component_artifacts` and
`model_cards.component_frames`, the last because it re-derives each head's fitted state
(imputation means, spline knots) and checks the persisted recipe against it at 1e-9, so a
widened training frame would fail heads that are correct. `season_terms` and
`components_preseason` reach `component_rates.build_design` directly and are pre-ladder by
construction; `tests/test_lag_ladder.py` pins all five by parsing, so a fourth fitting path
fails there rather than silently.

**Rung 0 comes back bit-identical on every column any head reads** — 48 feature and target
columns over 10,194 rows, checked through `head_design` rather than through `build_design`,
which is one layer further out than §7c's own test. Eleven columns *do* move and they are
named rather than glossed: `components_preseason.attach_preseason` writes a season-**centred**
twin of every preseason delta, and a season mean is a property of the frame, so 189 new rows
shift it. None of the eleven is in any head's feature list — the shipped block is the
volume-shrunk delta, a per-row product that cannot move — and `own_delta_centered` is an
ablation arm that ships nowhere. A test asserts no head's preseason block names a `_centered`
column, so if one is ever shipped this fails and the fix is to centre on rung 0.

#### The eleven artifacts, and what a plug-in looks like when it is one

`posteriors.GROUPS` gains `rookie-components`; the heads are namespaced `rookie_*` because
both families share a window directory and a rookie `reb` landing on `reb.pkl` would
overwrite the head that scores everybody else.

**Ten of the eleven are deterministic and one samples**, which is §7e's ship split read out
of `rookie_rate_metrics.csv` rather than pinned. A floor-shipped head is persisted as a
`DesignRecipe` with one step and one feature: the step writes §7d's floor **on the head's own
link** — `log(rate / 36)` for a count, `logit(p)` for a conversion — and the artifact carries
`beta = 1`, `alpha = 0`, identical on every draw, so the family's own inverse link hands the
floor's prediction straight back. That is what makes a plug-in and a fitted head the same
object to `component_rates`, and it is `sim/season.no_design_availability`'s shape one family
over: no posterior, so the predictive does not integrate over one. The draws are 1,000
identical rows rather than one, because `build_context` takes `min(a.n_draws …)` across the
bundle and a single-row head would collapse the twenty that do have a posterior to one draw.

The dispersion is **not** constant in the same sense: `phi` / `rho` are fitted on the head's
own fitting rows by `count_floor_predictive` / `conversion_floor_predictive`, so the plug-in
is a distribution rather than a point — a point estimate *of a dispersion*, repeated per
draw, which is exactly what §7d scored the floor's CRPS with.

The expanding draft-bucket prior table travels **inside** the step, for the reason the
composition's team-context block does: a rebuilt `rookie_rate_floors.csv` must not silently
change what a persisted posterior scores. It is point-in-time by construction — season S
averages true rookies from seasons strictly before S — so carrying it is not carrying an
answer, and its **history is the window's own**: the `train` artifact's floors have never
read a test season. A season the table does not cover **raises** rather than extrapolating,
which is how a 2026-27 forward row will announce that §5g's tier is missing.

| window | rookie fit rows | fitted head | its R-hat | divergences | worst round-trip |
|---|---:|---|---:|---:|---:|
| `train` | 1,161 | `reb` @ `slot_interaction_spline` | 1.0014 | 0 | 3.41e-13 |
| `train_val` | 1,307 | same | 1.0032 | 0 | 3.41e-13 |
| `full` | 1,438 | same | 1.00418 | 0 | 3.41e-13 |

The 1,161 is §7e's own fitting population to the row. The round-trip error is not zero and
should not be: a floor head's design is `log(rate/36)` and its prediction is `exp` of that,
so the gate measures a log/exp round trip against a **1e-8** bar. The design half is exactly
0.0 on all eleven.

**The whole group builds in 4.7 seconds at the `full` window** — `reb` is 4.5 of them and
the ten floors are 0.01–0.02 each, which is two dispersion fits and a round-trip rather
than a sampler. `make posteriors` was a day; this adds nothing to it.

#### The manifest, and the two columns that stop a plug-in wearing a fitted head's name

Every window now carries **31** manifest rows, and `manifest_row` gains
`family_population` (`veteran` / `true_rookie`) and `deterministic`. Both are in
`SPECIFICATION_COLUMNS`, so `assert_same_specification` compares on **10** columns rather
than 8 — which is the point: a head deployed at `full` as a fitted arm while `train`
selected the floor is exactly the class of failure that check exists for, and neither
`variant` alone nor `n_features` alone would catch every case of it.

`make production-check` is **green**: 31 of 31 heads at `full`, all matching the `train`
specification on those ten columns. The `full` manifest's worst R-hat is still **1.00608**
and its divergence count still **0**, because the one rookie head that samples reads
1.00418.

`make model-cards` still cards **20** heads and prints the eleven it does not, by name,
against a `DEFERRED_PREFIXES` rule — §5h decides whether the rookie family gets cards and
what `chain_role` it declares. `tests/test_model_cards.py` checks the deferral rather than
exempting it: the deferred keys must be **exactly** the component heads' rookie twins, so a
rookie head the simulator stops reading, or a twelfth one it starts reading, fails there.

#### The union, and where a board gains rows

`build_context`'s `units` is now `component_units(veteran design, rookie design, season,
composition players)` — both families intersected with the minutes allocation, concatenated
veteran-first, and labelled with `unit_family`. Disjointness is **asserted on every build**
rather than trusted, because the failure is silent: a player in both families takes two rows
in `units`, two entries in the tensor and two slots on a board, and every marginal still
looks plausible. The rookie rows inherit the ladder's own provenance vocabulary —
`lag_rung = true_rookie`, `lag_source = none` — so a board can say which rung scored a
player, and `total_minutes_lag1 = 0`, which is literally true rather than a fill and puts a
rookie in the lowest `prior_minutes` bucket every tensor consumer already has.

`component_rates` branches by family and scatters the blocks back. Scoring the whole frame
through one recipe would **not** raise — `DesignRecipe._block` fills a missing column with
zero and then standardizes it, so a rookie would be priced off the veteran design's mean
prior season — which is why the branch is on rows rather than on a `try`.

| season | veteran | `returnee_lag2` | true rookie | units | ADP-priced |
|---|---:|---:|---:|---:|---:|
| 2022-23 | 386 | 13 | 72 | **471** (was 386) | **211** (was 194) |
| 2023-24 | 387 | 6 | 74 | **467** (was 387) | **237** (was 216) |

*Scratch measurement, 2026-08-22, not in `make docs-audit`* — reproduce with
`build_context(cfg, season, "train", …)` and `adp_panel.parquet`. The ladder's 7 and 3
ADP-priced rows are §7c's admitted count to the digit. The rookie head's 10 and 18 are
larger than §7b's 9 and 15 because the denominators differ: §7b counted draft-board rows and
this counts units that reach the tensor and carry a panel row.

**The 2023-24 board's top five rookies are Chet Holmgren, Scoot Henderson, Victor
Wembanyama, Brandon Miller and Ausar Thompson**, and the recovered returnees are Miles
Bridges, Danilo Gallinari, Isaiah Thomas and Tristan Thompson — §7f's own named rows,
arriving in the tensor rather than in a table beside it. The ordering inside the rookie
block is coarse and should be: ten of the eleven heads are a preseason-per-36 blend, so the
family separates a lottery pick from a second-rounder far better than it separates
Wembanyama from Holmgren.

#### What the veteran units did not do

**§5f's spot-check is at the rates, not at the tensor, and that is forced.** Adding rows
moves the RNG stream — `_sim_one` draws a per-unit gamma and a per-unit beta sized by
`n_units` — so no simulated season can be bit-identical across a population change. The
claim that *is* well-posed is that the union does not change what a unit the veteran design
already carried is **scored with**, and it holds exactly: on 2023-24's 393 veteran-family
rows, every count rate, every conversion probability and every dispersion is bit-identical
to scoring those rows alone. A single-family frame also keeps the pre-union **shapes** —
`phi` stays `(draws,)` rather than being promoted to `(draws × n_units)` — which is not
cosmetic, since numpy's scalar and array gamma paths are different code and a promoted
shape would move every retrospective tensor for no modelling reason.

`make forward-board` re-run at 500 sims says the same thing from the board's end: on the 387
units both arms share, Spearman **0.9988** against a **0.9990** seed-noise floor, where
2026-08-21 read 0.9989 against 0.9990. The population column is where everything moved — the
retro board is **467** units and the forward arms carry **80** fewer, because
`rookie_rates.build_design` carves its population out of `component_targets` and "he played
in the NBA this season" is target-season information a forward context may not read.
`forward_context` withholds those rows and prints how many. **That is a leak guard, not a
deferral**, and it is why §5g is the session that puts a rookie on a *forward* board.

#### What Session 6 settles, and what it does not

- **Settled**: the family is persisted and reproducible. Eleven artifacts at three windows,
  every one round-tripping from its saved draws and recipe alone, and `production_check`
  green on 31 of 31 heads at `full` under a specification check that now reads the two
  columns telling a plug-in from a fit.
- **Settled**: the ladder is on, and rung 0's design is bit-identical on every column any
  head reads. The fitting population did not move, which is the only thing that makes
  scoring these rows with coefficients fitted before the ladder existed legitimate.
- **Settled**: rookies and lag-recovered returnees are in the tensor. 471 and 467 scorable
  units on the two validation seasons against 386 and 387, and 17 and 21 more ADP-priced
  rows — the hole §1 measured, closed on the retrospective path.
- **Not measured**: what any of it is worth. **The tensors on disk were NOT regenerated**,
  deliberately: every downstream artifact — the strategy sweep, Gate A, the mixture arms,
  the draft room's board — is built on the rookie-less ones, and re-running them is §5h's
  symmetric replay against §7a's floor. Until then `make simulate-season` and everything
  after it will produce different numbers from the ones on disk, and that is the wiring
  being live rather than a regression.
- **Not measured**: `priceable_room`'s drop, for the same reason — it reads the tensor's
  `scorable` mask, so the "101 of 105 board rows dropped" figure §1 quotes moves only when
  the tensor is rebuilt.
- **Not settled**: the forward board. It carries no rookies and cannot until §5g builds
  their rows from a roster snapshot and closes the ADP id-map's negative-surrogate tier.
- **⚠️ Was open for the length of this session, and is now CLOSED by
  `docs/availability-window-plan.md` §16j.** Session 6 wired the ladder's returnees into
  the tensor while `stan.availability.lag_ladder` was still `[]`, so they were given
  **24.7** games against a realized 46.9 — exactly what the §6 runbook interleaved §16
  ahead of this session to prevent, and it happened anyway because §16c's blast-radius
  argument had left the key off pending its own decision. §16j took that decision: the key
  is on, `availability.rung_zero` is threaded through all eight paths that fit, and the
  zero-sum transfer is measured at the board rather than assumed. On these rows the
  simulator now gives **39.71** and **35.88** games against realized 38.00 and 33.33, worth
  **−214.76** and **−82.23** dk_pts of season-total MAE, and the teammates who lose the
  minutes pay +3.93 and +3.88 — inside the seed noise on the same rows. `make ladder-board`
  is that reading.

### 7h. Session 7 — the forward wiring, and a production board with rookies on it (run 2026-08-22)

`make forward-board` on 2023-24 and `make forward-board SEASON=2026-27 FRAMES_ONLY=1`,
plus the id-map tier that had to land first. §7g put rookies in the *retrospective* tensor
and said exactly why it could not put them in the forward one; this session removes that
reason. **The 2026-27 production board now carries 116 true rookies and 6 lag-recovered
returnees**, and 13 of the rookies are players the DK market prices.

#### The id-map blocker, closed with a reference tier rather than a fallback

`adp_draftkings.build_id_map` matched only against the season matrix, which is built from
played seasons — so a player who has never played had no `player_id` to find and
`draft_pool.load_dk_boards` gave him a **negative surrogate**. That is the correct answer to
the question it was asked and the wrong one for a rookie head: a design row is keyed on the
real `nba_api` id, so no amount of modelling could reach a board row wearing `-830650`.

The source that has the id is the one `forward_design` already leans on —
`team_rosters_<season>.csv` carries a real `PLAYER_ID` for the 2026 draft class the day it
is published. `roster_snapshot_reference` reads it and a new cascade step consumes it,
**keyed on `(season, normalized name)`**, sitting above the three fuzzy steps and below the
two exact ones. Above the fuzzy steps because it is stronger evidence than any of them: the
era guard `prefix` and `reversed` need is free when the reference *is* the era, and an
ambiguous key yields nothing rather than a coin flip, which is `_by_prefix`'s own rule.

| | before | after |
|---|---:|---:|
| matchable DK ids | 811 | **882** |
| `no_nba_history` | 162 | **91** |
| `roster_snapshot` | — | **71** |
| cascade unmatched rate | 0.50% | **0.49%** |

**No existing match moved**, which is the property that made the placement safe to choose
on evidence rather than on argument. The residual 91 are DK's deep pool below the
577-player snapshot — camp and two-way names, **none of which carries an ADP** — so
`adp.NON_DEFECT_METHODS` was revisited and deliberately left alone: promoting them to
`unmatched` would put a permanent 91-row failure on a join with nothing left to find. All
**13** ADP-priced never-played players resolve, AJ Dybantsa at 41.8 among them, which is the
shape `tests/test_adp.py` now pins along with the cross-season and ambiguous-key refusals.

#### `forward_rookie_design`, and the two filters a forward season cannot pass

`rookie_rates.build_design` gains a `forward_seasons` knob with the same default-empty guard
`component_rates.build_design` and `build_component_targets` carry, and it clears **two**
conditions rather than one:

- `total_minutes > 0` — the target season's own minutes, which a scheduled season has for
  nobody. Same rule, same reason as the veteran design's.
- the `covered` cut — an **era** rule about seasons with no preseason panel at all, where a
  fitted row from 1998 would carry its missing indicator as a decade dummy. A forward season
  is the other case: its panel arrives with `make preseason` in October. Cutting it would
  leave a production board with no rookie rows every year until the fetch ran.

Everything else the design needs already existed forward. Membership is
`lag_recovery.classify`'s `true_rookie` over the **same** stacked targets frame the veteran
design is carved from — `forward_targets`, extracted this session so both families read one
frame, because two separate synthesies could disagree about whose first season this is. The
slot block comes from `forward_draft_slots`, which is `forward_draft_numbers`' own two
preseason-legal sources plus the draft **year** the rookie design needs for
years-since-draft; `_DRAFT_PICK` now captures it, and the 25 rights-traded draftees that
function documents still land in the undrafted bucket. Age comes from `load_ages`' roster
fallback. On 2026-27 that is **116** rows, **429** of the roster's 577 resolving a draft
slot, and **0** carrying a preseason block — the missing-indicator path working, not a gap.

#### 🔴 The ladder was not forward-safe, and §5g's third bullet was wrong about why

§5g reasoned that rung A needs no forward-specific tier because a returnee's lag-2 is a
prior-season statistic like any other. That is true and it is not sufficient:
`stan_components.head_design` carries the ladder **only on the path where it builds the
design itself**, and the forward path brings its own design and reaches
`component_rates.build_design`, whose default is the pre-ladder frame. So the forward board
was silently dropping every recovered returnee *and* handing the rows it did carry a null
`lag_rung`.

It was found by the census below, which read every forward veteran as lag-recovered. The fix
is one argument — `ladder=lag_ladder(cfg)` — and `tests/test_forward_design.py` pins it by
inspecting the call, because the failure mode is a board that is quietly narrower rather than
an error. The forward design goes from 419 to **425** rows on 2026-27.

#### §4's acceptance, both halves

**Present and draftable** — `forward_board_population.csv`, one row per (arm, population),
2023-24 at the `train` window, 500 sims:

| population | retro units | retro priced | forward units | forward priced | forward best rank |
|---|---:|---:|---:|---:|---:|
| veteran (rung 0) | 387 | 216 | 354 | 202 | 1 |
| lag-recovered | 6 | 3 | 4 | 3 | 267 |
| true rookie | 74 | 18 | 91 | 18 | 66 |

and on the **production** board, `make forward-board SEASON=2026-27 FRAMES_ONLY=1`:
**419** veteran units (180 priced), **6** lag-recovered (4 priced), **116** true rookies
(**13** priced). Both recovered populations are present and priced on every board. The
forward counts are the roster snapshot's rather than the game log's, so a returnee who never
actually came back is not in it (4 against 6) and a rookie who never played is (91 against
74) — the Part B bound, running in the direction it is known to.

**Unchanged for units the ladder did not touch** — the second half, and the reason
`forward_board.py` now runs every comparison twice. On the **354** rung-0 veterans all three
contexts carry, the fixed-population arm scores Spearman **0.9986** against a **0.9988**
seed-noise floor on the same units, at mean season-total gaps of **25.35** and **22.64**
dk_pts. The whole-board fixed-population figure *did* move, 0.9988 → **0.9929**, and that is
the 33 new units it carries and the retro board does not — rookies the snapshot lists who
never played. A Spearman over a union reports exactly that, which is why the rung-0 row sits
beside it rather than instead of it.

The population bound fell with the builders: the fixed-population arm now shares **all 467**
retro units against 80 missing before, and the snapshot arm is missing **36** against 113.

#### What Session 7 settles, and what it does not

- **Settled**: a forward board can carry a true rookie, end to end — board row → real
  `player_id` → design row → posterior → tensor → rank. The 2026-27 board carries 116 of
  them and prices 13.
- **Settled**: the ladder reaches the forward path, which it did not before this session
  and which nothing would have raised about.
- **Settled**: the untouched veterans did not move. Two population changes in two sessions,
  and the rung-0 mechanics reading is still inside seed noise.
- **Not measured**: what the forward rookies are *worth*. Nothing here scores a board — the
  acceptance is presence, pricing and stability. §5h's symmetric replay against §7a's floor
  is the contest-unit reading, and it is retrospective.
- **⚠️ The 2026-27 board cannot be SIMULATED and that is not a §5g gap.** 2026-27 is a
  **test** season, so `season.assert_season_allowed` refuses it under any window but the
  production one — `docs/final-evaluation-plan.md`'s guard doing its job. `FRAMES_ONLY=1`
  is the form a test season admits: every forward frame builds and is censused, and nothing
  is unlocked. The production board itself comes after `make posteriors-production`.
- **⚠️ Four DK boards are ingested nowhere.** `data/raw/dk_draft_rankings/` holds captures
  from Aug 12, 16, 17 and 20 that the ADP artifact chain (last built 2026-07-29) has never
  read. This session rebuilt the id map from the **same two boards** the rest of the chain
  uses, so its only delta is the snapshot tier. Ingesting the other four is a separate round:
  it moves roughly twenty audited ADP figures and `docs_audit._dk_name_agreement` assumes one
  board per season and raises on more.
- **Not settled**: the rights-traded draftees. 25 of the 2026-27 roster carry no slot in
  `HOW_ACQUIRED` and sit in the undrafted bucket until their first bio row corrects them.
  That is the largest cell and the block's honest zero, so nothing is invented — but a
  lottery pick priced as undrafted is a real mis-bucket and the fix is a draft-results
  source this project does not capture.

### 7i. Session 8 — the floor re-read, and what the two changes gave back (run 2026-08-22)

`make rookie-recovery` and `make strategy-sweep-rookie`, both on a **labelled tensor**.
§7g wrote the union into `build_context` and deliberately left the tensors on disk alone,
because every audited downstream artifact was drawn on the rookie-less ones. This session
draws the rookie-inclusive pair beside them — `sim_tensor_<season>_rookieinclusive.npz`,
471 and 467 units against 386 and 387 — and reads the floor against it. `--tensor-label`
suffixes the tensor, the draft room's cached field, Gate A **and** every sweep artifact,
which is `--field`'s discipline one layer down; a labelled tensor labels the gate by default,
because Gate A is one pooled table whose extremes `make docs-audit` re-derives and a variant
population writing into it would move audited figures by replacing rows.

#### The readout is a ladder of boards, and it is not three more sweeps

§7a split the floor in two and only one half resolved. The half that does: the Round-1 bar
the opponent field sets, averaged over **1,200** realized entries, which rose **+173.1** and
**+111.9** dk_pts when the field could draft players our seat could not price. The half that
does not: the lift delta, 21 of 24 arms losing at a median of −0.0765 while the shipped arm
gained +0.0316 — two realized seasons are two worlds, and §3 decision 1 said so in advance.

So the recovery is measured on the half that resolves, and measured as a **nested ladder of
board masks** rather than as two more asymmetric runs:

| rung | who the field may draft |
|---|---|
| `veteran` | rung 0 of the lag design — the board every artifact before §5f was built on |
| `ladder` | plus `returnee_lag2`, the one rung §7c's gate admitted |
| `rookie` | plus the true-rookie family — today's priceable board |
| `unrestricted` | every rostered player, which is what a real field actually drafts |

`cut(unrestricted) − cut(rung)` is §7a's floor measured against that rung. The rungs are
nested by construction, so each change's share is **attributable rather than inferred** —
and because the cut line reads the tensor for one thing only, the `scorable` mask, and is
otherwise a function of ADP order and realized totals, three more 53-minute sweeps would
have bought three more readings of the half that already refused to resolve.

**The rung-0 row reproduces `make rookie-floor`'s own number to the digit** — **+173.1** and
**+111.9**, out of a different module, on a different tensor, through a different code path.
That is the check that licenses reading the rest of the ladder as the same quantity §7a
priced, and it is the direct analogue of §7a's own Gate C reproduction.

#### The board ladder, validation seasons, realized

| season | rung | board | ADP-priced | Round-1 bar | floor left | recovered |
|---|---|---:|---:|---:|---:|---:|
| 2022-23 | `veteran` | 347 | 180 | 15,505.8 | **173.1** | — |
| 2022-23 | `ladder` | 357 | 187 | 15,583.3 | 95.5 | **+77.5** |
| 2022-23 | `rookie` | 411 | 196 | 15,653.5 | **25.4** | **+70.1** |
| 2022-23 | `unrestricted` | 448 | 196 | 15,678.9 | 0.0 | — |
| 2023-24 | `veteran` | 359 | 207 | 15,393.5 | **111.9** | — |
| 2023-24 | `ladder` | 363 | 210 | 15,409.3 | 96.1 | **+15.8** |
| 2023-24 | `rookie` | 417 | 225 | 15,534.4 | **−29.0** | **+125.1** |
| 2023-24 | `unrestricted` | 464 | 228 | 15,505.4 | 0.0 | — |

#### Four readings, and the second is the one that was not predicted

**1. The floor is closed, and on one season it is past closed.** 173.1 → **25.4** on
2022-23 and 111.9 → **−29.0** on 2023-24. On 2022-23 the two changes gave back **147.7 of
173.1**, 85% of a quantity Session 1 measured before any of the machinery existed.

**2. The 2023-24 residual is negative, and that is a real result rather than noise in a
figure that averages 1,200 entries.** The bar the field sets on our *rookie-inclusive*
board is **29.0 dk_pts higher** than the bar it sets on the whole rostered board. The
mechanism is §7a's own channel 2, arriving with its sign flipped: the 47 rows still outside
our board realize less than what the field would otherwise have taken, so letting the field
have them makes its entries *worse*. Past the rookie rung, the remaining hole is no longer a
handicap on us — it is a handicap on the field, and one we do not need. The floor was never
monotone in board width and this is the first measurement that shows it.

**3. The split lands where §5h predicted, on the count it predicted it on.** The ladder adds
**10** board rows and **7** ADP-priced ones in 2022-23 and **4** and **3** in 2023-24 —
§7c's admitted-rung census to the row, out of a completely separate code path. The rookie
head adds **54** and **54** board rows, **9** and **15** ADP-priced — §5h's own expectation,
written down before this session ran. What the two are *worth* does not follow the row
counts: in 2022-23 the ladder's 7 priced rows are worth **+77.5** against the rookie head's
9 for **+70.1**, and in 2023-24 the ladder's 3 are worth **+15.8** against the rookie head's
15 for **+125.1**. The ladder's rows are few, expensive and drafted early — Jamal Murray and
Kawhi Leonard are the §7a examples — and the rookie head's are many and cheaper.

**4. On 2022-23 the market-priced hole is completely closed.** At the `rookie` rung the
board carries **196** ADP-priced rows against the unrestricted board's **196**: every player
the DK market prices is now one our seat can rank. 2023-24 gets 225 of 228. The residual on
both seasons is made almost entirely of market-silent rows, which is the population a draft
reaches last and a real field reaches by accident.

#### The incidental Gate A reading, which is a population change and not a win

The labelled run wrote `sim_season_gate_a_rookieinclusive.csv`, and the season-total row
**improves**: MAE 360.80 → **345.43** on 2022-23 and 373.63 → **362.11** on 2023-24, CRPS
250.68 → 238.67 and 256.92 → 250.35, R² 0.7116 → 0.7323 and 0.7106 → 0.7209. **None of that
is a model improvement and it must not be quoted as one.** The added 85 and 80 units are
rookies and recovered returnees who score little and are easy to get roughly right, so the
denominator changed under a mean. It is recorded because the two Gate A tables now differ
and a reader holding both needs to be told which comparison they support: the shipped table
is the one the bars were set against, and the labelled one describes a different population.

#### The contest half, replayed — and it refuses to resolve exactly as §7a said it would

`make strategy-sweep-rookie` is the shipped sweep, symmetric, on the rookie-inclusive
tensor: 24 strategies × 5 structures × 2 seasons at 500 simulated worlds each, plus the
realized replay, writing `strategy_*_rookieinclusive.csv` and leaving the audited set alone.

**The comparison against the shipped symmetric arm is NOT paired, and that is the first
thing to say about it.** §7a could read its symmetric side off the 2026-08-16 artifact
because Gate C reproduced to **0.000e+00** across all 12 rows, which proved the two arms
were the same code on the same worlds. Here the tensor itself moved — new units, a moved
RNG stream, and the availability ladder of `docs/availability-window-plan.md` §16j — and
Gate C says so: the uninjected world differs by up to **14.57** dk_pts, the solved rotation
moves `rho` 0.3582 → 0.3443 and 0.3172 → 0.2825, and the scale moves 1.1337 → 1.1875 and
1.1405 → 1.1811. The injection still hits its target on both arms (`achieved_mae`
**400.4586** either side), so the gate is working; what is gone is the licence to treat the
difference as a board change alone. This is why the recovery readout above lives at the cut
line and not here.

Read with that caveat, the shipped arm's realized Round-1 lift moves **+0.0160** on average
over the eight multi-entry readings, spanning **−0.0682 to +0.1839** and losing lift in 3 of
8. Across the whole 24-arm table the sign reverses again — **17 of 24 arms lose, median
−0.0683**, from −0.3059 (`blend_a30`) to +0.2382 (`lineup_value`) — which is the same shape
§7a reported at 21 of 24 and median −0.0765. The single-entry `88k_alley_oop` reads −0.0396
and **+0.8823**, one roster's coin flip, and is excluded from every pooled figure for the
reason §7a excludes it.

The simulated arm is positive and uninformative for the reason §7a's was: **+0.0329** for
the shipped arm with 0 of 8 readings negative. It is a wider board scored by a model that
believes itself, not evidence.

#### What the replay does settle, and it is not the lift

**The sweep re-selects the same strategy on every tier that selects.** All four
multi-entry structures come back at `lineup_value_blend30` — the same ranking, the same
`alpha = 0.3`, the same objective — with simulated lift rising rather than falling (0.2369
→ 0.2645 on the 600k flagship). Two population changes that put 85 and 80 new units in the
tensor and 64 and 58 new rows on the board did **not** move the arm the project drafts
under, and that is a stability result the cut-line ladder cannot give.

The one flip is `88k_alley_oop`, which selects `lineup_value` (`alpha = 0`) instead. It is a
**single $450 entry** whose realized readings §7a already refuses to pool, and a tier with
one entry has no portfolio for a hedge to act on — so the flip is the noisiest cell in the
table changing hands, not a finding about `alpha`.

#### What the plug-ins actually contain, because "no-fit floor" invites the wrong reading

**Ten of eleven heads shipping a no-fit floor is not a finding that preseason says nothing
— the floor *is* the preseason estimator.** Recorded here because the plug-in's persisted
shape (`beta = 1`, `alpha = 0`, one synthetic column) reads like an intercept, and it is not.

That column carries `rookie_priors`' `shrunk` arm on the head's own link:

    w * preseason_per36 + (1 - w) * draft_bucket_prior,   w = min_pre / (min_pre + k)

with `k` = **160** preseason minutes on the seven count heads. At the draftable population's
median **70.4** preseason minutes that is `w` = **0.3057** — so a median rookie's plug-in
prediction is about 31% his own October per-36 and 69% the prior for players taken where he
was taken. **Preseason coverage on that population is 99.1%**, so the blend is live for
essentially all of them; a rookie who played no preseason gets `w = 0` by his own volume
rather than by a special case, and lands on the bucket exactly.

The three arms, scored apart (validation · draftable, `rookie_rate_floors.csv`):

| | `draft_bucket` | `preseason` | `shrunk` |
|---|---:|---:|---:|
| heads where it wins R² against `draft_bucket` | — | **6 of 11** | 8 of 11 |
| heads where `shrunk` beats it on CRPS | **11 of 11** | 8 of 11 | — |

Three readings, and the second is the one the shape of the artifact hides:

1. **Preseason carries information the draft slot cannot**, and the clearest case is
   `fg3a|fga`: R² **0.6109** from preseason against **−0.5989** from the bucket. Where a
   rookie shoots from is visible in October and is not a function of where he was drafted.
   `blk` is the same story more mildly (0.7019 against 0.2966).
2. **Raw preseason is not usable on its own** — 70 minutes is a small sample and it fails
   badly where the rate is rare or minutes-driven: `stl` R² **−0.5164** against the bucket's
   0.7871, and `fga` CRPS **207.03** against 37.73. So preseason wins on 6 of 11 by R² and
   loses catastrophically on the rest.
3. **The volume shrink is what turns one usable arm and one unusable arm into an estimator
   that wins everywhere.** `shrunk` beats the shipping draft-bucket incumbent on CRPS on
   **11 of 11** heads and beats raw preseason on **8 of 11**. That is P4(b)'s selected
   estimator doing the job it was selected for, and it is why the floor is a high bar rather
   than a null one.

So §7e's verdict — one fitted arm of eleven — says that a Stan head on top of this blend
buys nothing beyond what the blend already extracts, on ten of eleven quantities. It does
**not** say preseason is uninformative; it says the preseason information is already in the
floor. That is the same shape the veteran side reports, where the preseason block is worth
more than fitting itself on several heads (`docs/preseason-plan.md`).

#### What Session 8 settles, and what it does not

- **Settled**: the half of §7a's floor that resolves is closed. **173.1 → +25.4** and
  **111.9 → −29.0**, with rung 0 reproducing Session 1's own figure to the digit out of a
  separate module, and the per-change split landing on §7c's admitted-rung census to the row.
- **Settled**: the shipped strategy survives both population changes. Same arm on all four
  multi-entry structures, simulated lift up rather than down.
- **Settled**: the eleven rookie heads are **declared in the chain and deliberately not
  carded**, and the anchor test got stronger rather than weaker for it — see
  `docs/model-cards-plan.md`.
- **Not settled, and not settleable here**: what the changes are worth in Round-1 advance
  probability. The realized replay moves **+0.0160** for the shipped arm across readings
  spanning −0.0682 to +0.1839, while 17 of 24 arms lose at a median of −0.0683. Two realized
  seasons could not separate those in §7a and cannot now — and this time the arms are not
  even paired, because the tensor moved.
- **⚠️ The shipped tensors on disk are still the rookie-less ones.** §7i drew a labelled
  pair beside them rather than over them, deliberately, so `make simulate-season` and every
  target after it will still produce different numbers from the artifacts on disk.
  Regenerating the shipped set moves the whole downstream chain — the audited sweep, Gate A,
  the mixture arms, the draft room's cached board — and is a separate, unscheduled round.
  Nothing in this program requires it: the production board is built by
  `make posteriors-production` and `make forward-board`, which read the posteriors and not
  these tensors.
- **⚠️ The held-out figures do not transfer.** `docs/final-evaluation-plan.md` §7 now records
  it: every figure of the 2026-08-21 round measured the rookie-less workflow, the head
  coefficients are untouched but the population they describe is narrower than the one that
  ships, and §4d's **−0.0725** chain reading was drafted off a board with no rookie on it.
  The split is spent and there is no second unlock.
