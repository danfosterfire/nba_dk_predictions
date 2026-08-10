# Dashboard expansion — one prompt per session

Scaffolding, not a plan and not a result. The plan is
[dashboard-plan.md](dashboard-plan.md), section **"The expansion — from one view to nine
pages"**; this file is that plan's build order cut into ten copy-pasteable prompts, one per
fresh session. **Delete this file when the expansion lands** — it describes work, not the
project.

The ten sessions map onto the plan's eight steps, with the model-card emitter split in two
(it is the largest) and the three remaining model pages split in two (the minutes page is
bespoke, the other two are configuration).

| session | step | deliverable | new pipeline work |
|---|---|---|---|
| 1 | 1 | the `st.navigation` shell + Player fingerprints | no |
| 2 | 2 | Tournament & strategy page | no |
| 3 | 3a | `make model-cards` — index, coefficients, features, correlations | **yes** |
| 4 | 3b | `make model-cards` — predictive ECDF, calibration, residuals | **yes** |
| 5 | 4 | the generic model renderer + Availability page | no |
| 6 | 5a | Box-score components + Game length pages | no |
| 7 | 5b | Minutes page, incl. the two-unit comparison | no |
| 8 | 6 | Inputs beyond the heads + the capture-calendar emitter | small |
| 9 | 7 | Draft board as a page | no |
| 10 | 8 | Overview | no |

Run them in order. Sessions 3 and 4 are the only ones that must not be reordered against
anything downstream — sessions 5–7 cannot start until 4 finishes.

---

## Session 1 — the multipage shell

```
Implement step 1 of the dashboard expansion: the multipage shell.

Read first: docs/dashboard-plan.md (the section "The expansion — from one view to nine
pages", especially finding 1 on st.tabs vs st.navigation) and dashboard/README.md.

Convert dashboard/app.py from a single-view script into an st.navigation / st.Page
entrypoint. Move today's PCA view verbatim into dashboard/views/fingerprints.py behind a
render() function, named "Player fingerprints" in the nav. Lift the light/dark appearance
toggle into shared state so it survives navigation instead of resetting per page. Add a
placeholder page or two only if the nav needs more than one entry to render sensibly —
otherwise ship with the one real page.

Constraints that will bite: `make dashboard` must keep working unchanged; nothing in
dashboard/ may import from src/ (SRC_IMPORTERS still names exactly one file); the view's
pure layer stays in dashboard/pca.py with no Streamlit import.

Verify with all three layers from dashboard/README.md — AppTest in both appearance modes,
a figure rendered to PNG, and the live page driven in a real browser. kaleido and
playwright are installed and reach the system Chrome via
p.chromium.launch(channel="chrome"); no playwright install is needed.

Update docs/dashboard-plan.md in place and add any load-bearing decision to
dashboard/decisions.py.
```

## Session 2 — Tournament & strategy

```
Implement step 2 of the dashboard expansion: the Tournament & strategy page (page 8).

Read first: docs/dashboard-plan.md ("The expansion", the "Page 8" subsection),
dashboard/README.md, docs/dk_best_ball_rules.md, and the relevant parts of
docs/simulations-plan.md.

This page needs no new pipeline work — every artifact exists and nothing reads it. Four
blocks: the contest structure from bracket_structure.csv and dashboard/economics.py; the
strategy sweep from strategy_sweep.csv drawn as lift-vs-null with intervals, faceted by
axis, with the break-even hurdle as a reference line; simulated versus realized from
strategy_shipped.csv beside strategy_realized.csv, labelled so the reader knows which is
the tuning surface and which is the honest readout; and the paired comparisons from
strategy_paired.csv, where a gap whose interval crosses zero is the most useful thing on
the page and should be styled as such rather than buried.

Constraints: read artifacts only, no src/ import, pure logic in its own module with no
Streamlit import so it is testable directly. Follow dashboard/theme.py unmodified —
ALL_PAIRS_CAP is 3 for any chart where non-adjacent series sit side by side, and no value
may be reachable by colour alone.

Verify with the three layers. Update the plan doc and the decision registry.
```

## Session 3 — the model-card emitter, part A

```
Implement step 3a of the dashboard expansion: the first half of the model-card artifact
emitter.

Read first: docs/dashboard-plan.md ("The expansion", finding 2 and "The model-card
artifact contract"), src/models/posteriors.py in full, and src/models/held_out.py.

Create src/models/model_cards.py and a `make model-cards` target, plus
docs/model-cards-plan.md for the contract. This session writes four of the seven
artifacts: model_card_index.csv (one row per head — unit, family, variant, n_fit, label,
class), model_card_coefficients.csv (head x term, posterior mean/sd/q2.5/q25/q75/q97.5,
with a term_family column so a 12-knot spline basis can be grouped rather than swamping a
panel), model_card_features.csv (head x feature x split x bin — binned counts plus
per-feature n/mean/sd/missing-share), and model_card_feature_corr.csv (head x feature x
feature, plus the top ~20 correlated pairs flagged for the on-demand density that session 4
will bin).

Three things that are easy to get wrong and must not be: go through
held_out.selection_split so the only legal split values are train and validation and no
test row can reach an artifact; read the `train` posterior window, not train_val, since at
train_val the validation rows were in the fit; and verify the design recipe at build time
the way posteriors.py already does — reproduce each head's own design matrix exactly and
fail the build rather than write a wrong artifact.

No dashboard work at all this session. Tests in the repo's plain-assert synthetic-builder
style. Register the decisions and update both plan docs.
```

## Session 4 — the model-card emitter, part B

```
Implement step 3b of the dashboard expansion: the predictive half of the model-card
emitter.

Read first: docs/model-cards-plan.md and docs/dashboard-plan.md ("The model-card artifact
contract"), plus src/models/model_cards.py as session 3 left it.

Add three artifacts to `make model-cards`: model_card_ecdf.csv (head x split x grid point —
the observed ECDF and the 50/80/95% quantiles of the posterior-predictive ECDF across
draws), model_card_calibration.csv (head x split x 2-D bin — fitted-vs-observed density and
residual-vs-fitted density, binned rather than per-row), and model_card_sample.parquet (a
bounded per-head-per-split subsample carrying fitted, observed and residual, so the pages
can overlay real points for texture on top of the binned density).

Cost is the thing to get right here. The composition head at full row count times a
thousand draws is not affordable and is not needed — cap the draws at ~200 over a
subsample, and check that the ECDF band is stable at that budget rather than assuming it.
Total artifact footprint should stay in single-digit MB.

Same three rules as session 3: selection_split only, the `train` posterior window, and the
build-time recipe check. Each head declares its own unit in model_card_index.csv — the
component heads are season-collapsed player-seasons, the composition is per team-game,
game length is per game — and the dashboard reads the unit from there rather than
hard-coding it.

Tests, plan docs, registry.
```

## Session 5 — the model renderer and the Availability page

```
Implement step 4 of the dashboard expansion: the generic model-detail renderer, proved
against one model class.

Read first: docs/dashboard-plan.md ("Pages 3-6 — the model detail views"),
docs/model-cards-plan.md, dashboard/README.md, and docs/availability-plan.md.

Build the seven-block renderer once, generically, then instantiate it as the Availability
page (page 3) with a head selector covering the availability head and the games-played
tenure decomposition (gp_entry, gp_exit, gp_onset, gp_duration). The seven blocks in order:
what this head is; the features it was fed; feature relationships as a correlation heatmap
plus one on-demand 2-D density for a selected pair (NOT a pair plot — that fork is settled,
see feature-correlation-not-pair-plots in the registry); coefficients as sorted posterior
means with 95% intervals, spline bases grouped by term_family; predictive calibration as
the observed ECDF over the posterior-predictive ribbon, train and validation side by side;
predicted-vs-observed and residual-vs-fitted, four panels, train and validation; and
diagnostics read from the existing stan_*_diagnostics.csv and the posteriors manifest.

The page states its unit prominently, read from model_card_index.csv. Long scrolling pages
are fine and expected — do not cram. Typed prose is allowed only in block 1 and only to
describe the specification, never a result.

Build the renderer so sessions 6 and 7 are configuration rather than a rewrite. Verify with
all three layers, and test the pure layer directly.
```

## Session 6 — Box-score components and Game length

```
Implement the first half of step 5: the Box-score components and Game length pages (5 and
6).

Read first: docs/dashboard-plan.md ("Pages 3-6"), docs/predictions-plan.md,
docs/shot-attempt-basis-plan.md, and dashboard/views/ as session 5 left it.

Both pages are instantiations of session 5's renderer. Box-score components carries eleven
heads behind one dropdown — the seven negative-binomial counts and the four beta-binomial
conversions. Game length carries two: overtime onset (beta-binomial) and overtime depth
(beta-geometric).

Two things specific to these pages. The components page should surface each head's no-fit
floor alongside its fitted score, because every head in this project is quoted against that
floor and one of them (ftm|fta) does not clear it — that is a finding, not an embarrassment
to hide. And the fg3a|fga head is a share of attempts, not a shooting percentage; the page
must not let a reader think otherwise, since that confusion is exactly what
component_rates.conversion_variants guards against in code.

If the renderer needs changes to accommodate either page, change it once and re-verify the
Availability page too rather than forking a second renderer.

Verify with the three layers. Update the plan doc and registry.
```

## Session 7 — the Minutes page

```
Implement the second half of step 5: the Minutes page (page 4).

Read first: docs/dashboard-plan.md ("Pages 3-6"), docs/minutes-composition-plan.md, and
docs/simulations-plan.md's minutes-unification sections.

Two heads behind the selector — the marginal min|available head and the team-game
composition — rendered through session 5's renderer. Then the block that makes this page
different from the other three: minutes_unification.csv, which is the repo's cleanest
demonstration that a head is only a model at the unit it was scored at. The same posterior
wins decisively at the per-team-game unit and loses at the season unit, against the same
two floors. Draw that as a comparison rather than burying it in a table, and show the
predictive spread that is the actual failure (4.68x too narrow) rather than only the means,
which barely differ.

Worth showing alongside it: the injected per-player-season effect at sigma = 0.450 that
closes the gap to a tie, and the zero-sum team constraint the composition carries and the
marginal head does not. The second of those lands directly on drafting strategy — a
same-team stack's minutes are anti-correlated, not independent — so it belongs on a page a
reader reaches before the strategy page.

Verify with the three layers. Update the plan doc and registry.
```

## Session 8 — Inputs beyond the heads

```
Implement step 6 of the dashboard expansion: the "Inputs beyond the heads" page (page 7),
plus a small capture-status emitter.

Read first: docs/dashboard-plan.md ("Page 7"), docs/adp-plan.md, and
docs/availability-plan.md's capture sections.

The through-line is everything the simulator consumes that is not a fitted coefficient.
Three blocks. ADP: the point-in-time-safe panel and its three-dates-per-row discipline, the
DK/FantasyPros agreement from adp_profile.csv, the name-matching audit, and the four of nine
seasons that point-in-time safety costs. The capture programs on a deadline: injury-report
PDFs, the ESPN feed and the DK board, drawn as a coverage calendar — this one is worth
building precisely because the data is perishable and a gap in the calendar is
unrecoverable, so the picture is an operational alarm rather than a decoration. And the four
calibrated simulator inputs: the residual copula as a heatmap, the block variance inflation
from serial_correlation.csv, the bonus overdispersion from bonus_calibration.csv, and the
game-level minutes dispersion — each showing its fit_window rather than hiding it, because
which window to consume is decided by what the number will be scored against.

`make adp-status` and `make capture-status` print rather than write, so the calendar needs a
small emitter writing a real artifact first. Keep it small; this is not a new pipeline
stage.

Verify with the three layers. Update the plan doc and registry.
```

## Session 9 — the Draft board page

```
Implement step 7 of the dashboard expansion: the draft board as a page (page 9).

Read first: docs/dashboard-plan.md ("Page 9"), dashboard/README.md's section on the one
exempt page, and docs/simulations-plan.md's "The live draft room".

Add dashboard/draft_room.py to the nav as a page, exposing its body behind a render() the
entrypoint imports. This is feasible only because the shell is st.navigation rather than
st.tabs — the page's script does not run until the reader navigates to it, and
st.cache_resource keeps the ~40 MB reference field warm across navigation. Confirm that
holds in a browser rather than assuming it: navigate away and back and check the field is
not rebuilt.

Two conditions the session must not break. `make draft-room` keeps working as a standalone
single-page launch, because draft night is a thirty-second clock and should not share a
process with anything — the file stays directly runnable. And the src/ exemption stays
narrowed rather than widened: SRC_IMPORTERS still names exactly one file, still held to
src.sim, and moving the draft room into the app must not become the precedent that lets a
model page import a model.

Responsiveness is the acceptance criterion here, not completeness. If navigation makes the
room measurably slower to first paint than `make draft-room` does today, say so with the
measurement and leave them separate.
```

## Session 10 — Overview

```
Implement step 8 of the dashboard expansion: the Overview page (page 1). Last on purpose —
its tiles link into the other pages, so it could not be written before they existed.

Read first: docs/dashboard-plan.md (the "Charter amendment 2026-08-10" subsection, which
sets the terms this page is allowed to exist on), README.md, and the
dashboard-overview-page-exemption entry in dashboard/decisions.py.

A portfolio reader arrives at a URL with no context and will not open a repository. Give
them: what the problem is, the pipeline in one diagram, four to six hero numbers, and a
route into the pages that show the work. Emphasize brevity — this is a portfolio item, not
a walkthrough.

The three bounds are the terms of the exemption and are not negotiable. One page, one
screen: if it scrolls, it has become the nine-tab walkthrough this charter deleted on
2026-08-08. Every number on it is read from an artifact — typed prose may say what the
project does, it may not state a result, so a tile showing season-total MAE reads
season_total_metrics.csv like every other figure on the site. And no decision registry, no
provenance links, no reversal log.

Separately and worth doing while you are here: README.md still describes ranking, drafting
and tournaments as "planned" when they shipped. Fix that, and check `make docs-audit` still
passes.
```
