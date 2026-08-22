# Rookie rate heads — scoring the structurally-missing population

**Opened 2026-08-22.** Converts `docs/potential-to-dos.md` §16 into a scheduled program:
a component-rate head family for the players the veteran heads structurally cannot score
— true rookies, returnees, and thin-prior fringe — so they can appear on a board and be
drafted. Seven sessions, each with its own prompt (§6), each appending its results here.

## 🚧 STATUS: SESSIONS 1-3 RUN 2026-08-22 (§7a, §7c, §7d). DESIGN REVISED 2026-08-22 (§7b). SESSION 4 IS NEXT.

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
shipping draft-bucket incumbent on CRPS on **11 of 11 heads**. **No head is fitted yet** —
Session 4 runs the §4 gate on 108 draftable validation rows.

**§7a, §7b, §7c and §7d are in `make docs-audit`** (`_rookie_floor`, `_lag_recovery`,
`_lag_ladder` and `_rookie_rates`, against `strategy_rookie_floor.csv`, the `_rookiefloor`
artifact set, `lag_recovery.csv`, `lag_ladder.csv` and `rookie_rate_floors.csv`). Every
other figure here is either audited from its own artifact elsewhere (`rookie_priors.csv`,
`strategy_injection.csv`, `preseason_coverage.csv`) or labelled scratch — §7a labels its two
scratch measurements inline. Each session that produces a quotable number should add it to
the audit the same day (see `docs/docs-audit.md`).

## 1. The problem

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
scorable units are `component design ∩ composition players` (`src/sim/season.py:978-982`);
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

- All eleven heads: 7 NB counts with minutes exposure, `fg3a|fga` + 3 conversions
  beta-binomial, on the two existing compiled sources. Ladder mirrors
  `count_variants`/`conversion_variants`: linear → +interaction → +spline.
- Metrics via the existing scorers (`val_r2`/`val_nll`/CRPS/PIT); artifact
  `outputs/predictions/rookie_rate_metrics.csv` in the shape of
  `stan_component_metrics.csv`, with `selected` and `beats_floor` head-local.
- Run the §4 gate per head; record fitted-vs-floor ship decisions here and in the
  registry. Sampler scale: eleven small-population fits — minutes each, nothing like the
  veteran sweeps.

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
- **The ladder is forward-safe without new work and the doc should say why**: a
  returnee's lag-2 and a thin-prior's lag-1 are both prior-season NBA statistics, which is
  exactly what `forward_design` already assembles for veterans. Rung A and rung B need no
  forward-specific tier — only the true-rookie head does, which is the whole of the
  id-map blocker above.
- Acceptance per §4; `make forward-board` on 2026-27 shows draftable rookies **and**
  lag-recovered returnees.

### 5h. Session 8 — closeout

- Optional readout: symmetric rookie-inclusive sweep replay (suffix mechanism) — how much
  of Session 1's floor the two changes recover, reported **separately for the ladder and
  the rookie head**, since they land in different parts of the board: the ladder recovers
  7 and 6 ADP-priced rows against the rookie head's 9 and 15 (§7b).
- `docs/final-evaluation-plan.md`: record that the held-out figures measured the
  rookie-less workflow and do not transfer to the rookie-inclusive one.
- docs-audit coverage for every figure this doc quotes from its own artifacts;
  `make dashboard-audit` green; `docs/model-cards-plan.md` check — whether the new heads
  get model cards and what `chain_role` they declare.

## 6. Session runbook — the prompts

One prompt per session, for a fresh session each. Sessions 2-8 end by appending results
here and updating `dashboard/decisions.py` where a ship decision lands.

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
4. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 4 (§5d): fit the eleven rookie
   > arms and run the stated gate** — validation paired-bootstrap CRPS vs floor plus
   > rolling origins, write `rookie_rate_metrics.csv`, record fitted-vs-floor ship
   > decisions in the doc and registry.
5. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 5 (§5e): the season-total
   > readout** — rookie-admitting frame, `rookie` AND `lag_recovered` groups in
   > `season_total.evaluate`, head-vs-floor at the season-total unit on validation. This
   > is §16's settling gate; record it.
6. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 6 (§5f): persistence and simulator
   > integration** — the `rookie-components` posterior group at `train_val` and `full`,
   > deterministic recipes for floor-shipped heads, the `build_context` units union with
   > per-family branching, veteran units bit-identical spot-check, `production_check`
   > green.
7. > I'm working on the NBA prediction project (CLAUDE.md). Read
   > docs/rookie-rates-plan.md and execute **Session 7 (§5g): forward wiring** —
   > `forward_rookie_design`, the ADP id-map roster-snapshot tier with its test, rookie
   > and lag-recovered rows on `make forward-board`, and the §4 acceptance checks.
8. > I'm working on the NBA prediction project (CLAUDE.md). Read
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

