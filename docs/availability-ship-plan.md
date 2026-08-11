# Shipping the windowed, role-graded availability head — a seven-session work plan

**What ships:** `stan_availability.py` moves its fitting window to **2012-13 onward** and its
dispersion from one shared scalar to a **role-graded vector** over prior-MPG buckets. On
validation that reads CRPS **9.8247** against the incumbent's **10.0057** — a paired gap of
**−0.181, interval [−0.257, −0.103]** — PIT KS **0.0588** against 0.0939, and boundary error
cut 43%. The evidence is `docs/availability-window-plan.md`; do not re-litigate it here.

**This doc is scaffolding.** It is a sequencing aid, not a record. When the round lands, the
"as built" content belongs in `availability-window-plan.md` and this file should be deleted,
exactly as `dashboard-build-prompts.md` was.

---

## The three things that will break if nobody reads this first

**1. `availability_design` must NOT be filtered.** It is imported by `stan_minutes`,
`stan_composition`, `stan_games_played`, `model_cards`, `sim/season`, `season_terms` and
`final_evaluation`. Putting a season filter inside it silently re-scopes the minutes head,
the composition head, the games-played spell process and the simulator — six heads, none of
which asked for it, and nothing would raise. **The window is applied to the fitting rows
inside the availability head only.** This is the same trap `docs/potential-to-dos.md` item 1
records for `stan.composition.first_season`.

**2. The Stan change is a transplant, not an invention.** `composition_glm.stan` already
carries exactly the pattern needed — `int<lower=1> n_rho`, `array[P] int rho_bin`,
`vector<lower=1e-6, upper=0.95>[n_rho_par] rho`, and the vectorized gather
`rho[rho_bin[lik]]` (lines 117-123, 179, 205-207). Its comment states the invariant to
preserve: *"n_rho = 1 with rho_bin all-ones is EXACTLY the shared-rho model."* Copy that
discipline. `n_rho = 1` must reproduce the incumbent posterior bit for bit, the same way
`S = 0` disables the year effect and `U_n = 0` disables `sigma_u`.

**3. Every session ends green.** `make docs-audit` is a gate *and* a pytest test, so the
moment a refit moves a quoted figure the repo is red until the prose is updated. Therefore
**each session updates the figures it moves**, in the same session. Session 7 is a narrative
and registry pass, not a catch-up pass. If you defer figures, six sessions sit red.

## Decisions taken 2026-08-11

**1. `make availability-model`'s GLM/GBM/ridge ladder does not move, and is not re-run.**
Production is the Stan/Bayesian head and that is not being reconsidered at this stage, so a
comparison whose only purpose was to establish which mean function to adopt has no live
decision behind it. It stays on the full window **as documentation of how the project got
here**, not as a live gate. Do not refresh it, and do not read its figures as describing
what ships.

**2. The no-prior population gets an imputed role bucket in three classes, not a pooled
default.** These are players on a season-start roster with no prior-season row at all — 106
of 539 rostered players in 2022-23, and per `README.md` about **14.7% of season-start roster
minutes**. Assigning them the pooled `rho` would be a silent shrug at a sixth of the league.
The rule, in order:

- **Rookies → bucket from draft position.** A first overall pick reaches the court more than
  a thirtieth, and the repo already carries this: `stan_composition.rookie_share_priors` is
  an expanding-window, point-in-time mean realized minutes share per draft bucket for
  no-prior players, with `DRAFT_BUCKETS` and `UNDRAFTED_BUCKET` from
  `features/team_context.py`. Reuse it rather than building a second draft-to-minutes map.
- **Returning veterans → the bucket they usually land in, given where they left.** A player
  whose last appearance is before S−1 is identifiable from the panel. Measure the transition
  from his bucket at last appearance to his bucket in the first season back, and condition
  on the length of the gap — one season out and three seasons out are not the same event.
  This is the only genuinely new measurement in the round.
- **Everyone else → the lowest bucket.** Undrafted rookies, G-League call-ups, anyone the
  first two classes miss. `bio_draft_number` is ~15% NaN by construction, so this branch is
  load-bearing rather than a rare fallback.

**Both estimates are point-in-time expanding-window quantities, not full-history averages.**
That is stricter than "estimated on the fitting half" and it is the house pattern:
`no_design_availability` and `rookie_share_priors` both pool only seasons strictly *before*
the target, intersected with the seasons selection may read. Using every season including
the target's own would leak, and using the held-out seasons would leak worse.

**3. The role bins themselves stay fixed.** `season_effects.ROLE_EDGES` is
`[0, 12, 24, 30, 60]` on `minutes_per_game_lag1` — constants, so no split concern. If anyone
later makes them quantile-derived they must come from the fitting rows only, which is what
`composition_glm.stan`'s "bin edges from TRAIN quantiles only" comment is about.

## What is deliberately NOT re-run

`stan-components` (137 min), `stan-composition` (~9 h) and `stan-minutes` are **separate
factorized heads**. The chain `availability → min | available → counts | min` has disjoint
parameter blocks, so changing availability does not change their posteriors. Re-running them
would burn hours to reproduce identical numbers — see the standing note in
`CLAUDE.md`/memory about not refitting for record-keeping. `make final-evaluation` is **not**
run at any point: it is the end-of-project reading.

---

## Session 1 — the Stan source, the head, the config

**Cost:** `make stan-availability` was 254 s at the full window; a 2012-13 window is smaller,
so budget under 10 minutes including the MLE.

```
Read docs/availability-ship-plan.md, then docs/availability-window-plan.md §4 and §5.

Ship the windowed, role-graded availability head into Stan.

1. src/stan/betabinomial_glm.stan — replace the scalar `rho` with a role-indexed vector,
   transplanting the pattern from src/stan/composition_glm.stan (lines 117-123, 179,
   205-207): `int<lower=1> n_rho`, `array[N] int<lower=1,upper=n_rho> rho_bin`, and
   `vector<lower=1e-6,upper=0.95>[n_rho] rho` gathered as `rho[rho_bin]`. `n_rho = 1` with
   all-ones bins MUST reproduce the current posterior exactly — same nesting discipline as
   `S = 0` for the year effect. Keep the `inv_logit(-eta)` form and the existing comment
   about why it is not `1 - inv_logit(eta)`.

2. src/models/stan_availability.py — `StanAvailability` gains a role-bucket argument and
   passes `n_rho` / `rho_bin` through. Buckets come from
   src/eda/season_effects.ROLE_EDGES on `minutes_per_game_lag1` (prior season, so it is
   legitimate point-in-time information). `self.rho_draws` becomes (draws x n_rho) and
   `predict_pmf` / `mu_draws` must gather per row. Default the head to the shipped
   configuration.

3. Add a `stan.availability` block to configs/default.yaml — it does not exist yet. Carry
   `first_season: 2012-13` and the role-bin setting, with a comment on each saying what it
   is and pointing at docs/availability-window-plan.md.

CRITICAL: do NOT put the season filter inside `availability_design`. Six other modules
import it and would be silently re-scoped. Filter the fitting rows inside the head's own
fit path, and leave scoring on the full validation set.

Then: run `make stan-availability`. Verify (a) the port check still holds — the MLE must be
fitted on the SAME windowed rows or the comparison is meaningless; (b) the fitted rho vector
is near the point-MLE values 0.3084 fringe / 0.2456 star; (c) validation CRPS lands near
9.82, not 10.01.

Add tests to tests/test_stan_heads.py (plain assert, synthetic builders): n_rho=1 nests the
shared-rho likelihood, and rho_bin gathers the right dispersion per row.

Finish with `.venv/bin/python -m pytest tests/` and `make docs-audit` BOTH green — update
any figure in docs/ or README.md that this refit moved, in this session.
```

**Commit:** `Ship windowed role-graded rho into stan_availability`

---

## Session 2 — posteriors and model cards, and the payoff check

**Cost:** `make posteriors` for the availability head only; `make model-cards` is minutes.

```
Read docs/availability-ship-plan.md and docs/model-cards-plan.md.

Propagate the windowed, role-graded availability head to the persisted posteriors and the
dashboard's model cards.

1. src/models/posteriors.py::availability_artifact — its `extras` currently declares
   {"dispersion": "rho_draws"} and `rho_draws` is now a (draws x n_rho) matrix rather than a
   vector. Persist the bin edges and the per-row bin assignment recipe alongside it, so a
   consumer can reconstruct which rho applies to which player WITHOUT refitting. Keep the
   fit-window axis (`train` / `train_val` / `full`) orthogonal to the new season-truncation
   axis and name them so nobody confuses the two.

2. src/models/model_cards.py — the availability card must draw its predictive through the
   head's own `predict_samples`, per the four rules in docs/model-cards-plan.md, and its
   mean must still reproduce the head's own (`predictive_bias` near zero).

Then run `make posteriors` (availability only if the target supports it) and
`make model-cards`.

THE PAYOFF CHECK, and report it explicitly: read outputs/predictions/model_card_ecdf.csv for
head == "availability" and compare against the pre-change reading. The defect this whole
round exists to fix was P(GP<10) predicted 5.21% against an observed 8.15%, and
P(GP>=82) predicted 5.95% against an observed 2.72%, both outside the 95% band. Say what
those two numbers are now and whether they are inside the band. If they have barely moved,
say so plainly — the CRPS and PIT wins would still stand, but the headline claim would not.

Finish with pytest and `make docs-audit` both green, updating figures this moved.
```

**Commit:** `Propagate windowed availability head to posteriors and model cards`

---

## Session 3 — the imputed role bucket for players with no prior season

**Cost:** numpy only, minutes. No Stan, no simulator run.

```
Read docs/availability-ship-plan.md — decision 2 especially — then
src/sim/season.py::no_design_availability and
src/models/stan_composition.py::rookie_share_priors.

Build the imputed role bucket for players with no prior-season row. The availability head is
a lag-1 design, so these players are not in its frame at all: 106 of 539 rostered players in
2022-23, and about 14.7% of season-start roster minutes. They still consume roster spots, and
because the minutes allocation is zero-sum whatever they are given comes straight out of
their teammates.

Three classes, in this order:

1. Rookies — bucket from draft position. `rookie_share_priors` already computes an
   expanding-window, point-in-time mean realized minutes SHARE per draft bucket for no-prior
   players, with DRAFT_BUCKETS and UNDRAFTED_BUCKET from src/features/team_context.py.
   Convert that share to an MPG equivalent and map it through season_effects.ROLE_EDGES.
   Reuse that function and its vocabulary; do not write a second draft-to-minutes map.

2. Returning veterans — a player whose last appearance precedes S-1, identifiable from
   availability_panel.parquet. Measure the transition from his role bucket at last
   appearance to his bucket in the first season back, CONDITIONED ON GAP LENGTH — one season
   out and three seasons out are not the same event. Emit this as an artifact with counts,
   one row per (gap length, from bucket, to bucket): it is a fact about the league worth
   reading on its own, not just an internal lookup.

3. Everyone else — the lowest bucket. `bio_draft_number` is ~15% NaN by construction, so
   this branch is load-bearing rather than a rare fallback.

HARD REQUIREMENT: both estimates are point-in-time expanding-window quantities. Pool only
seasons strictly BEFORE the target, intersected with the seasons selection may read, exactly
as `no_design_availability` and `rookie_share_priors` already do. Do not average over the
whole history, and do not touch the held-out seasons.

THEN THE CHECK THAT COULD INVALIDATE THE IDEA — do it before anything is wired anywhere. The
buckets carry DISPERSION, not level. Putting a first overall pick in the `30+ mpg` bucket
hands him the LOWEST rho (0.2456), the most reliable availability in the model. That may be
exactly backwards for rookies, who plausibly have more availability variance than their
minutes suggest. So measure the realized availability dispersion of each imputed class
directly and compare it against the rho its assigned bucket would give it. If rookies are
more dispersed than their imputed minutes bucket implies, say so and propose the fix — a
rookie-specific bucket or an offset — rather than shipping the mapping quietly.

Report the transition table and the dispersion check, write the artifact, and STOP. No
simulator changes this session.

Finish with `.venv/bin/python -m pytest tests/` and `make docs-audit` both green.
```

**Commit:** `Impute a role bucket for the no-prior-season population`

---

## Session 4 — the simulator draw path

**Cost:** `make simulate-season` is the expensive one here — measure it before committing to
a full re-run, and say what it cost.

```
Read docs/availability-ship-plan.md and docs/simulations-plan.md.

Carry the role-graded dispersion into the season simulator.

src/sim/season.py draws availability at roughly line 645:

    a, b = beta_shapes(ctx["avail_mu"][draw], np.full(ctx["n_players"], ctx["avail_rho"][draw]))

That `np.full` broadcasts ONE dispersion to every player and is now wrong. Give each player
the rho of his own prior-MPG bucket. The precedent is already in this file: the composition
head's draw a few lines below uses `ctx["row_rho_bin"]` for exactly this, so mirror its
naming and its context-building rather than inventing a second convention.

The no-prior population's buckets come from session 3's artifact, not from a default invented
here — wire that lookup in. If session 3's dispersion check found rookies more variable than
their imputed minutes bucket implies, apply whatever it proposed rather than the raw mapping.

Then run `make simulate-season` and `make weekly-scores`, and report Gate A at both units
against its previous reading (outputs/predictions/sim_season_gate_a.csv,
weekly_score_*.csv).

Finish with pytest and `make docs-audit` both green, updating figures this moved.
```

**Commit:** `Draw availability with role-graded dispersion in the simulator`

---

## Session 5 — the heads that hold availability as a floor

**Cost:** `make stan-games-played` was 42.7 min. `make season-total` is minutes.

```
Read docs/availability-ship-plan.md, docs/games-played-plan.md and the Gate E section of
docs/availability-plan.md.

Two consumers hold the availability head as their permanent floor, and both of their gates
move now that the floor moved.

1. `make season-total` — Gate E. `season_total.gate_e` reads the incumbent's own row out of
   whichever table it is scoring, so the bar moves with the head by construction and must
   not be hard-coded. Confirm that still holds, then report the new MAE and CRPS against the
   previous 400.46 / 287.26.

2. `make stan-games-played` — its floor arm IS the availability head, so every arm's
   "vs floor" column shifts. Re-run and report whether any verdict changes. The standing
   verdict is that no arm clears Gate D and the incumbent stands; say explicitly whether
   that survives, because a better floor makes it HARDER to clear, not easier.

Do NOT re-run stan-components, stan-composition or stan-minutes. They are separate
factorized heads with disjoint parameter blocks; their posteriors do not depend on
availability, and re-running them would spend hours reproducing identical numbers.

Finish with pytest and `make docs-audit` both green, updating figures this moved.
```

**Commit:** `Re-run games-played and season-total against the new availability floor`

---

## Session 6 — the contest layer

**Cost:** unknown and potentially large. **Measure `make strategy-sweep` before running it**
— it is 24 strategies x 5 structures x 2 seasons x 500 worlds. If it is hours, say so and
decide with the user rather than starting it.

```
Read docs/availability-ship-plan.md and docs/simulations-plan.md.

Re-run the drafting and tournament layer on the new simulated seasons, and report what moved.

Order: `make bracket`, `make draft`, then `make strategy-sweep`. Time the sweep's first
structure before committing to the whole grid and report the projected cost; if it is more
than an hour, stop and report rather than running it to completion.

What to report, specifically: the shipped arm's Round-1 advance lift against the
symmetric-field null in the 600k Shootaround (previously 0.2107 simulated, 0.1268 realized),
and whether Gate D's "the two buy-in tiers do not select materially different rosters in 0
of 6 paired comparisons" still holds.

The interesting question is whether a better-calibrated availability tail changes the
strategy verdicts at all. Both boundary errors made rosters look MORE reliable than they
are, so any axis trading ceiling against reliability — stacking, handcuffing,
diversification — is where a change would show up. If nothing moves, that is a result worth
stating plainly.

Finish with pytest and `make docs-audit` both green, updating figures this moved.
```

**Commit:** `Re-run the contest layer on the recalibrated availability draws`

---

## Session 7 — the narrative, the registry, the guards

**Cost:** minutes.

```
Read docs/availability-ship-plan.md, docs/availability-window-plan.md and
docs/docs-audit.md.

Close out the availability shipping round.

1. README.md — §2's availability paragraph and §3's headline still describe a full-window
   head with one shared dispersion. Rewrite both to describe what ships, and update every
   quoted figure. §3's "The availability head is the largest measured win" paragraph and the
   `board_correlation` figure in §2 are the two that matter most.

2. docs/availability-window-plan.md — add an "as built" section recording what shipped, what
   each downstream session measured, and anything that came out differently from the plan.
   This doc becomes the permanent record.

3. dashboard/decisions.py — the entries `availability-misses-both-boundaries` and
   `rho-is-graded-by-role-not-by-era` are status "measured" and should become "built" with
   their shipped figures. Check whether any existing availability entry is now falsified; a
   reversal becomes "withdrawn" and KEEPS its entry.

4. Delete docs/availability-ship-plan.md — it is scaffolding and the round has landed.

5. CLAUDE.md — trim the availability-window-plan entry to describe the shipped state rather
   than the investigation.

Do NOT touch `make availability-model`'s GLM/GBM/ridge ladder or any figure it produces —
decision 1 in the ship plan settles that it stays on the full window as a record of how the
project got here. Where README.md or a plan doc quotes it, make sure the surrounding prose
says it is development history rather than a description of what ships.

Finish with pytest, `make docs-audit` and `make dashboard-audit` all green.
```

**Commit:** `Close out the availability shipping round`

---

## Rollback

Every session is one commit. The change is additive at the Stan level — `n_rho = 1` with
all-ones bins is the incumbent exactly — so reverting the config alone (`first_season` back
to the full window, `n_rho` to 1) restores the shipped head without touching the `.stan`
source, provided session 1's nesting test is real. That test is the rollback guarantee;
write it first.

## Standing invariants for every session

- `.venv/bin/python -m pytest tests/` — never `.venv/bin/pytest`, whose shebang is broken.
- Selection reads validation only. `selection_split` never materializes the held-out rows,
  and nothing in this round may call `final_split`.
- Anything fitted from data — bin edges, scalers, shrinkage constants — comes from the
  fitting rows alone.
- A load-bearing decision gets a `dashboard/decisions.py` entry alongside its plan-doc edit.
