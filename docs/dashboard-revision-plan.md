# Dashboard Revision Plan — the second round

Planning doc, not a measurement report. `docs/dashboard-plan.md` is the charter and the
record of the nine-page expansion that shipped 2026-08-10; **this doc is the round after
it**, and it inherits every rule that one set. Update it in place as steps land, with a
"Step N, as built" section each, the way the expansion doc was.

Five items, requested 2026-08-10, each a self-contained session. The build order below is a
dependency ordering rather than a preference.

**Two of the five have a different answer than the request assumed**, and both differences
came out of reading the code rather than the docs:

- The availability page does **not** carry two alternate models where one ships. It carries
  five heads, of which **two are in the simulator's draw path and three are not** — see
  [step 2](#step-2--the-availability-page-says-which-head-ships).
- The DHARMa work is **mostly already built**. `stan_utils.pit_from_samples` *is* the scaled
  quantile residual, and `model_cards.draw_predictive` already produces the replicate draws
  it needs — see [step 3](#step-3--scaled-quantile-residuals-in-block-6).

---

## What every step inherits

Not negotiable; these are the mechanisms that keep the surface from drifting into
documentation again. All are from `dashboard/README.md` and `docs/dashboard-plan.md`.

- **Every figure is read from an artifact a `make` target produced.** If a number is not in
  an artifact it does not go on the page. Typed prose may say what something *is*; it may
  not state a *result*.
- **Nothing in `dashboard/` imports from `src/`.** `draft_room.py` is the one pinned
  exemption, held to `src.sim` by `SRC_IMPORTERS`. A step that needs a computed quantity
  writes an emitter in `src/` and has the dashboard read its artifact — that is what
  `src/models/model_cards.py` exists for, and step 4 needs the same move.
- **The three verification layers.** `AppTest` in both appearance modes proves the page
  runs; a figure rendered to PNG through kaleido proves the figure is legible; the live page
  driven in Chrome through Playwright proves the page is. A pipeline-only step owes a
  **build gate** instead. Both tools are in `requirements.txt` and neither needs a browser
  download.
- **Tests in `tests/test_dashboard.py`** (and `tests/test_model_cards.py` for emitter work),
  plain `assert` with synthetic builders, exercising the pure layer directly.
- **A `dashboard/decisions.py` entry for every load-bearing fork**, per the standing
  instruction in `CLAUDE.md`, alongside the plan-doc edit.
- **This doc updated in place**, and any new artifact accounted for so `make
  dashboard-audit`'s orphan count moves the right way. If a step quotes a new figure in a
  doc, `make docs-audit` has to be able to re-derive it.

---

## The build order

| # | step | user item | new pipeline work | why here |
|---|---|---|---|---|
| 1 | [Appearance reaches the whole page](#step-1--the-appearance-mode-reaches-the-whole-page) | 4 | none | global chrome; every later step is then verified once, under the real theme |
| 2 | [The availability page says which head ships](#step-2--the-availability-page-says-which-head-ships) | 2 | one column on `model_card_index.csv` | small, self-contained, and it corrects a claim now on the page |
| 3 | [Scaled quantile residuals](#step-3--scaled-quantile-residuals-in-block-6) | 3 | one artifact from the existing predictive | touches all four model pages through one renderer |
| 4 | [dk_pts at the tournament round](#step-4--dk_pts-at-the-tournament-round-the-new-page) | 5 | an emitter, plus training-season tensors | the large step; it also adds a page, which moves the route table |
| 5 | [The Overview as a paper](#step-5--the-overview-rewritten-as-a-paper) | 1 | none | last, because step 4 gives it a ninth route — the same reason it was built last the first time |

---

## Step 1 · The appearance mode reaches the whole page

**User item 4.** The sidebar radio changes the plot surfaces and nothing else.

### What is actually going on

This is not an oversight; it is a decision that has stopped being right.
`.streamlit/config.toml` says so in a comment:

> The chart surfaces are *not* set here on purpose. `dashboard/app.py` pins its plot
> surfaces to the exact values the colour palette was validated against and offers its own
> light/dark toggle, so Streamlit's page chrome is left to follow the viewer's own theme
> setting.

So there are **two independent controls**: Streamlit's own appearance setting owns the page
chrome, and `shell.appearance_control()` owns the plots. A reader whose Streamlit is in dark
mode and who picks "light" gets a hybrid, which is exactly the reported symptom.

### The finding that changes the fix

**Streamlit 1.60 carries per-mode theme config**, which the current `config.toml` predates.
`[theme.light]` and `[theme.dark]` both accept `backgroundColor`,
`secondaryBackgroundColor`, `textColor`, `linkColor`, `borderColor`,
`dataframeBorderColor`, `dataframeHeaderBackgroundColor`, `codeBackgroundColor` and a
sidebar sub-table, plus `theme.chartCategoricalColors` / `chartSequentialColors` /
`chartDivergingColors` at the top level.

That means the honest fix is largely a **data change**, not a CSS fight: define both modes
from `dashboard/theme.THEMES` so that whichever mode Streamlit is in, the chrome is the
palette that was validated — including the one surface CSS cannot reach.

### The fork this step has to settle, and how

**`st.dataframe` renders to a canvas** (already recorded in `dashboard/README.md`), so its
cells are not in the DOM and an injected CSS rule cannot repaint them. Config can. That is
the boundary between the two options:

- **A — the radio wins.** Keep `shell.appearance_control()` and add a CSS block, rendered by
  the entrypoint from `shell.py` beside it, that repaints `stAppViewContainer`, `stHeader`,
  `stSidebar`, headings, body text and captions from `THEMES[mode]`. Everything in the DOM
  follows the radio; the canvas-rendered tables follow Streamlit's base and can disagree.
- **B — Streamlit's own setting wins.** Retire the radio. `shell.current_theme()` already
  reads `st.context.theme.type` through `detected_mode()`, so the reader switches appearance
  in Streamlit's settings menu and chrome, widgets, dataframes and plots all move together.
  One control, no hybrid state possible, and one fewer thing in the sidebar.

**Recommended: build the config half first, measure, then decide.** The `[theme.light]` /
`[theme.dark]` tables are worth having under either option and cost nothing under both. With
them in place, drive the page in Chrome in both modes and *list* what still fails to follow
the radio. If the list is only the dataframes, A is defensible with a documented boundary;
if it is wider, take B. Do not decide this from reasoning — the last three surprises on this
dashboard were all found by looking at the running page.

Two constraints on either path:

- **The palette stays the validated instance, used unmodified.** Generate the config values
  from `theme.THEMES` rather than retyping them, or add a test that parses `config.toml` and
  asserts the two agree. A hand-typed hex that drifts from `THEMES` puts the measured
  contrast figures in `dashboard/README.md` out of date silently.
- **`headless = true` is load-bearing** and must survive the edit — without it `make
  dashboard` stops at an interactive onboarding prompt and exits 255.

### Verification

Browser layer is mandatory and is the only one that can see this at all; `AppTest` has no
DOM. Check in both modes on at least the Overview, one model page (for the tables) and the
draft room: app background, header, sidebar, body text, a `st.dataframe` header, and the
plot surface. Drive the nav link rather than `page.goto` — a hard reload is a new session
and a legitimate reset to `detected_mode()`.

---

## Step 2 · The availability page says which head ships

**User item 2**, and the premise needs correcting before the page does.

### What the code actually does

`src/sim/season.py::_sim_one` draws availability in two moves:

```python
p_available = rng.beta(a, b)                    # the `availability` head's beta-binomial
gp          = rng.binomial(cell_games, p_available[...])
played      = allocate_spells(gp, cell_games, dur_mu, dur_kappa, ...)   # `gp_duration`
```

`spell_shape(artifacts["gp_duration"], ...)` supplies that `(mu, kappa)`. So **two heads are
in the draw path**: `availability` says *how many* games a player misses, `gp_duration` says
*how they clump*, and `games_played.allocate_spells` places the spells.

`gp_entry`, `gp_exit` and `gp_onset` are **not called at draw time**. Outside
`stan_games_played.py`, the only modules that name them are `posteriors.py` (which persists
them) and `model_cards.py` (which cards them). `season.py` states the reason it does not call
`HybridProcess.sequences`: that path draws its count from a pmf already marginalized over the
posterior, which is right for a marginal metric and wrong for a simulator whose whole point
is that one posterior draw moves the entire board together.

They are still load-bearing, though, which is why the recommendation below is not deletion:
`stan_games_played_gp_pmf.csv` — what the tenure decomposition produces — is one of Gate A's
four bars in `make simulate-season`.

So the page's current class intro, *"Two ways of predicting the same quantity"*, is the
thing to fix. It is true of the two modelling **approaches** and it reads as "here are two
alternates, pick one", when the shipped chain uses one head from each.

### What to build

Keep all five heads and make the chain role **visible and read from an artifact**:

- Add a chain-role column to `model_card_index.csv` — declared in `src/models/model_cards.py`
  beside `HeadSpec`, where the head is described, and emitted per head. Something in a closed
  vocabulary: the head supplies the count, supplies the spell shape, or is fitted and not
  called at draw time.
- Surface it on the model page next to the unit, so it applies to every model class rather
  than being an availability-page special case — the minutes page has the same question
  (both minutes heads ship, in different halves) and the box-score page does not.
- Rewrite `model_cards.CLASSES["availability"].intro` to say the chain rather than the
  dichotomy: the head that gives the count, the head that lays it out, and the tenure
  decomposition that exists as the process model and the Gate A bar.

**A test must pin the claim against reality**, in the spirit of `pca.orient()` and
`COMPONENT_BASIS`'s anchors: a declared "in the draw path" that nothing in `src/sim/` reads
is exactly the kind of interpretation that goes stale on the next refactor. Assert the
declared draw-path heads against what `src/sim/season.py` actually loads.

Alternative considered and not recommended: drop the three heads from the page. It hides
three fitted, converged, carded heads, and the reader who wants to know why the simulator
does not use the tenure chain then has nowhere to find out.

**A coincidence worth using.** `make dashboard-audit`'s baseline on 2026-08-10 is **1**
orphaned artifact, and it is `outputs/predictions/stan_games_played_spell_shape.csv` — the
realized spell shape, i.e. exactly the quantity `allocate_spells` produces and the reason
`gp_duration` is in the draw path at all. `docs/dashboard-plan.md` treats the orphan count as
how it picks what to build next, and this step is pointed at the last one. Drawing it, or
naming it in a registry entry, would take the count to 0; check it on the way in and out
either way.

---

## Step 3 · Scaled quantile residuals in block 6

**User item 3 — "how feasible is this?" Answer: high, and most of it is already in the
repo.** This is an artifact plus two figures, not a method.

### Why it is nearly free

DHARMa does three things: simulate replicate responses from the fitted model; compute each
observation's quantile within its own replicate distribution; randomize across the
probability mass at the observed value, because the non-randomized quantile of a discrete
predictive is not uniform even under a perfect model.

All three already exist here:

| DHARMa step | what already does it |
|---|---|
| `simulate()` | `model_cards.draw_predictive` — `(200 draws × rows)` per head per split, drawn through **each head's own** `predict_samples` and gated by `predictive_bias` |
| the scaled residual | `stan_utils.pit_from_samples` — `below + U·at`, and its docstring gives DHARMa's own reason for the randomization |
| uniformity statistic | `stan_utils.ks_uniform` |

Reuse both `stan_utils` functions verbatim rather than reimplementing — the same convention
that governs `compute_dk_pts`. Every response on these pages is discrete (beta-binomial,
negative binomial, beta-geometric), so the randomized form is required, not optional.

One difference from DHARMa worth stating on the page in one line: R's DHARMa simulates from
the fitted model at its point estimate, while these draws integrate over the posterior. The
residual is therefore a Bayesian PIT residual — same reading, and if anything the better
behaved of the two, since it carries parameter uncertainty rather than conditioning it away.

### What to build

A new artifact beside the existing seven, following `docs/model-cards-plan.md`'s contract —
long format, keyed by `head`, `split`, and the same closed split vocabulary (no test
column). Binned like `model_card_calibration.csv` rather than per row, for the same reason:
a 631,158-row composition panel is a copy of the data. Carry the per-row values on
`model_card_sample.parquet` for the bounded overlay.

The user asked to replace **residual-vs-predicted**, so block 6 keeps its fitted-vs-observed
panel and swaps the other:

1. **QQ-uniform** — sorted `u` against expected uniform quantiles, with an envelope, and the
   KS distance as a tile. This is the panel that makes the residual readable at all: a
   correct model puts the points on the diagonal regardless of the head's likelihood, which
   is the property four differently-scaled heads on one page need.
2. **Scaled residual against predicted**, with predicted **rank-transformed** to a uniform x
   axis (DHARMa's default), and the 0.25 / 0.5 / 0.75 quantile lines drawn — flat at those
   levels iff calibrated. The rank transform is what makes the panel comparable between a
   count head on a season total and a conversion head on a rate.

### The things that are easy to get wrong

In the spirit of the emitter's existing gate list — every one of these renders as a
perfectly good-looking picture:

- **The randomization needs a fixed seed per (head, split)**, matching `model_cards._seed`,
  or a rebuild silently moves the picture and a reader cannot tell a refit from an RNG.
- **200 draws quantizes `u` to steps of 1/200.** That is fine for a QQ plot and marginal for
  a gated KS statistic. Decide whether the residual artifact needs its own, larger draw
  budget, and measure it the way `band_stability` measures the ribbon's — two interleaved
  halves, and a build failure if they disagree.
- **The composition head may be out of scope, and should say so rather than be drawn.** Its
  response is `eta`, the mean of a *step* in a sequential allocation; `fitted_source` is
  `predictive_mean` and `spec.check` is neither `mean` nor `p_one`. Check whether a quantile
  residual on that response means anything. If it does not, emit the head with an explicit
  "not applicable" reason — a wrong panel is worse than an absent one.
- **`gp_duration` carries a weight** (`ResponseSpec(..., weight="w")`). Decide how weights
  enter the residual and the KS, and say which in the artifact.
- **A KS distance is not a pass/fail.** `model_cards.band_distance`'s docstring already
  makes this argument for block 5 — at n ≈ 10⁴ everything fails a strict uniformity test —
  and the same rule applies here: report the distance, and never render in-or-out as a
  verdict.

Update `docs/model-cards-plan.md` (the emitter contract), add the artifact to `ARTIFACTS`,
and register the decision.

---

## Step 4 · dk_pts at the tournament round — the new page

**User item 5.** A page between "Inputs beyond the heads" and "Tournament & strategy",
comparing observed against simulated `dk_pts` per player per tournament round, on train and
validation.

### Why this is the right page to add

Gate A today scores season-total `dk_pts`, the games-played pmf, the per-game bonus rate and
the season-total minutes spread. **Nothing scores `dk_pts` at the round unit** — which is the
unit the contest is actually decided at, since each of the four elimination rounds is its own
cut. This project has already paid once for the lesson that *a head is only a model at the
unit it was scored at* (`make minutes-unification`, one posterior and two opposite verdicts).
This page is that check at the unit that matters for the deliverable, and it should be framed
as an extension of Gate A rather than as a picture.

### What already exists — more than expected

**The tensor already carries the round map.** `data/features/sim_tensor_<season>.npz`:

```
dk_pts           (386, 20, 2000)  float32
games_played     (386, 20, 2000)  uint8
player_id        (386,)           int64
scoring_periods  (20,)            int64
tournament_round (20,)            int64      ← the four rounds, per scoring period
prior_minutes    (386,)           float32
```

So the four round totals are one grouped sum over the period axis. **No re-simulation is
needed to change the unit** — and `outputs/eda/scoring_periods_rounds.csv` carries the round
→ week/date map for every season independently.

### What is missing, and what it costs

- **Training-season tensors.** Only `2022-23` and `2023-24` are on disk. `season.allowed_seasons`
  already permits train seasons — it derives the legal set through `held_out.selection_split`,
  so a train season needs no unlock and a test season still refuses. `draft_pool_coverage.csv`
  covers all 30 seasons, so the roster side is there.
  **Scope decision: do not simulate all 25 training seasons.** Take the last two
  (`2020-21`, `2021-22`) to match the validation pair, and say why in the doc. Measure one
  season's wall clock and disk at a small `--n-sims` before committing to a full run —
  each shipped tensor is ~90 MB.
- **Observed `dk_pts` per player per round.** No artifact carries it. Derive from the game
  logs joined to the scoring-period grid, reusing `preprocess.compute_dk_pts` and
  `features/scoring_periods.py`'s own slot map verbatim. Never reimplement either.
- **An emitter.** The dashboard may not open a 90 MB `.npz` and reduce it — that is computing
  a model quantity, and the artifact rule says no. So this step needs the same move step 3 of
  the expansion made: a module in `src/sim/` behind a `make` target, writing flat,
  dashboard-shaped artifacts. Reuse `model_cards`' binning helpers (`bin_edges`,
  `calibration_edges`, `ecdf_grid`, `ecdf_curves`, `sample_frame`) rather than re-deriving
  them, so the new page's panels are the same objects as a model page's and a reader learns
  the encoding once.

### What is on the page

Mirroring block 5 and block 6 of a model page, which is what the user asked for:

- **Observed ECDF over the posterior-predictive ribbon**, faceted by tournament round, with
  train and validation distinguished. Read as a *distance* from the median replicate, not as
  in-or-out — the same rule `model_cards.band_distance` already carries.
- **Observed against predicted**, binned density with a bounded subsample overlaid, per
  round, per split.
- The round structure itself stated from `scoring_periods_rounds.csv`: Round 1 is seventeen
  weeks and Rounds 2–4 are two each, so the four panels are not four equal units and a
  reader comparing their spreads without knowing that is being misled. Same argument the
  model pages make about the unit, one level up.

Consider carrying the scaled quantile residual from step 3 here too, at the round unit — it
is the same computation over a different predictive, and it is the panel that makes four
differently-sized rounds comparable.

### Consequences elsewhere

- The page slots in at **position 8**, moving Tournament & strategy to 9 and Draft board to
  10. `app.VIEWS` order changes; the pinned `url_path`s do not.
- `overview.ROUTES` gains a row. That is why step 5 follows this one.
- `dashboard/README.md`'s page count and layout tree need updating, as does the nine-page
  language in `docs/dashboard-plan.md` and `CLAUDE.md`.

---

## Step 5 · The Overview, rewritten as a paper

**User item 1.** "The random stats about the project in the current version come across as
numbers-vomit rather than a concise overview."

### What to change and what not to

The five hero tiles are a consequence of the charter's **bound 2** — every number read from
an artifact — and that bound is not the problem and does not get relaxed. `overview.Spec`
stays exactly as it is: a reading is a `build` over the frames it `needs`, so a figure cannot
be typed into the view even by accident.

What changes is the **surface**. A tile row is five numbers and no sentence, which is
precisely the "numbers-vomit" reading. Restructure as four short sections mirroring
`README.md` — Introduction, Methods, Results, Discussion — of two to four sentences each,
with the artifact-read figures *inline in the prose* rather than stacked in tiles. The
lookups stay; `Reading` becomes a prose fragment rather than a metric label, value and delta.

Keep: the pipeline diagram (it is the one figure that carries the whole shape at a glance),
and the route block (it is why the page was built last).

### The bound this step has to renegotiate explicitly

**Bound 1 is "one page, one screen"**, measured — the first draft came in at 1,144 px on a
900 px viewport and was cut to 702. Four prose sections will not fit a 900 px viewport
alongside the diagram and now nine route links. So this step must either:

- **(a) amend bound 1** to "opens above the fold; scrolls no further than one screen more",
  written into `docs/dashboard-plan.md`'s charter amendment section rather than assumed; or
- **(b) hold one screen** and drop the diagram or the routes to pay for the prose.

**Recommended: (a).** The reason bound 1 existed was "if it scrolls it has become the
walkthrough again", and the walkthrough was nine tabs of rendered decision registry — four
short paragraphs is not that. But the amendment is a charter change and must be written and
registered, not taken silently. The page should still open with the Introduction and its
first result visible, and the route block must stay reachable without hunting.

Bound 3 does not move: no decision registry, no provenance links, no reversal log. The
existing tests that parse the module for a `decisions` import, for `docs/` in reader-visible
strings, and for digits in route labels all stay green.

### Verification

The browser layer is the only one that can measure the fold, and it is the layer that
failed this page's first draft. Measure the rendered height in Chrome at 1440×900 and record
it in the "as built" section, as the first round did.

---

# Appendix · The five session prompts

**Ephemeral scaffolding.** Delete this appendix when the round lands, the way
`docs/dashboard-build-prompts.md` was deleted when the expansion did. Each prompt is meant
to be pasted into a fresh session on its own.

---

### Prompt 1 — the appearance mode

> Read `CLAUDE.md`, `docs/project-spec.md`, `dashboard/README.md` and
> `docs/dashboard-revision-plan.md` (step 1). Do not read the other steps' sections; they
> are other sessions' work.
>
> The dashboard's light/dark radio changes only the plot surfaces — the page background,
> text, sidebar and tables follow Streamlit's own appearance setting instead. Make the
> appearance a single, coherent choice across the whole page.
>
> Start from the finding in step 1: Streamlit 1.60 supports per-mode theme config
> (`[theme.light]` / `[theme.dark]`, including `backgroundColor`, `textColor`,
> `borderColor`, `dataframeHeaderBackgroundColor` and a sidebar sub-table), which the
> current `.streamlit/config.toml` predates and deliberately declines to set. Build that
> half first, from `dashboard/theme.THEMES` rather than retyped hex — the palette is the
> validated reference instance and is used unmodified. Keep `headless = true`.
>
> Then **measure in a browser, in both modes**, what still fails to follow the sidebar radio,
> and settle the fork step 1 names: keep the radio and accept a documented boundary (option
> A), or retire it in favour of Streamlit's own appearance setting, which
> `shell.detected_mode()` already reads (option B). Decide on the measurement, not on
> reasoning — record what you found either way.
>
> Owed on the way out: the three verification layers (the browser one is mandatory and is
> the only one that can see this — `AppTest` has no DOM); tests in `tests/test_dashboard.py`,
> including one pinning `config.toml` against `theme.THEMES` if you generate one from the
> other; a `dashboard/decisions.py` entry; a "Step 1, as built" section in
> `docs/dashboard-revision-plan.md`; and `dashboard/README.md` updated if the sidebar's
> contents change.

---

### Prompt 2 — which availability head ships

> Read `CLAUDE.md`, `docs/project-spec.md`, `dashboard/README.md`,
> `docs/availability-plan.md`, `docs/games-played-plan.md` and
> `docs/dashboard-revision-plan.md` (step 2).
>
> The Availability page's class intro says "Two ways of predicting the same quantity", which
> reads as two alternates where one ships. That is wrong in a specific way, established in
> step 2 by reading `src/sim/season.py::_sim_one`: the simulator draws the games-played
> **count** from the `availability` head and lays the misses out with
> `games_played.allocate_spells` at `gp_duration`'s fitted spell shape. `gp_entry`,
> `gp_exit` and `gp_onset` are never called at draw time — `season.py` states why it does
> not call `HybridProcess.sequences` — but the tenure decomposition still produces
> `stan_games_played_gp_pmf.csv`, which is one of Gate A's four bars.
>
> Verify all of that yourself before building anything; if it has moved, follow the code.
>
> Then make each head's role in the shipped chain visible **and read from an artifact**: a
> chain-role column in a closed vocabulary, declared in `src/models/model_cards.py` beside
> `HeadSpec` where the head is described, emitted on `model_card_index.csv`, and surfaced by
> `dashboard/views/model_page.py` beside the unit so every model class gets it rather than
> just this one. Rewrite `model_cards.CLASSES["availability"].intro` to describe the chain.
> Do not delete the three heads.
>
> Pin the claim with a test, the way `pca.orient()` and `COMPONENT_BASIS` pin theirs: a head
> declared to be in the draw path that nothing in `src/sim/` reads is exactly the
> interpretation that goes stale on the next refactor.
>
> Owed on the way out: `make model-cards` re-run and its build gate green; the three
> verification layers; tests; a `dashboard/decisions.py` entry; `docs/model-cards-plan.md`
> updated for the new column; and a "Step 2, as built" section.

---

### Prompt 3 — scaled quantile residuals

> Read `CLAUDE.md`, `docs/project-spec.md`, `docs/model-cards-plan.md`,
> `dashboard/README.md` and `docs/dashboard-revision-plan.md` (step 3).
>
> Replace block 6's residual-against-predicted panel on the model pages with **scaled
> quantile residuals**, following R's DHARMa. Keep the fitted-against-observed panel.
>
> Most of this is already built and the step is an artifact plus two figures, not a method.
> `model_cards.draw_predictive` already returns `(200 draws × rows)` per head per split
> through each head's own `predict_samples`; `stan_utils.pit_from_samples` already computes
> the randomized quantile residual (`below + U·at`) that DHARMa calls a scaled residual, and
> `stan_utils.ks_uniform` the uniformity statistic. Reuse both verbatim — the same
> convention that governs `compute_dk_pts`.
>
> Emit a new long-format artifact following `docs/model-cards-plan.md`'s contract: binned
> like `model_card_calibration.csv`, keyed by head and split, no test column, with the
> per-row values riding on `model_card_sample.parquet` for the overlay. Draw two panels: a
> QQ-uniform with an envelope and the KS distance as a tile, and the scaled residual against
> **rank-transformed** predicted with the 0.25/0.5/0.75 quantile lines.
>
> Step 3 lists five things that are easy to get wrong here, every one of which renders as a
> good-looking picture — the seed, the 1/200 quantization of `u`, whether the composition
> head's `eta` response admits a quantile residual at all, `gp_duration`'s weight, and the
> rule that a KS distance is reported as a distance and never as a pass/fail. Handle each
> explicitly, and prefer an artifact that declares a head out of scope over a panel that is
> quietly wrong.
>
> Owed on the way out: the emitter's build gate (a pipeline step owes a gate rather than a
> browser, but this one also ships a page change, so it owes the three layers too); tests in
> `tests/test_model_cards.py` and `tests/test_dashboard.py`; a `dashboard/decisions.py`
> entry; `docs/model-cards-plan.md` and `dashboard/README.md` updated; and a "Step 3, as
> built" section.

---

### Prompt 4 — dk_pts at the tournament round

> Read `CLAUDE.md`, `docs/project-spec.md`, `docs/dk_best_ball_rules.md`,
> `docs/simulations-plan.md`, `docs/model-cards-plan.md`, `dashboard/README.md` and
> `docs/dashboard-revision-plan.md` (step 4). This is the largest step in the round.
>
> Build a new page, between "Inputs beyond the heads" and "Tournament & strategy",
> comparing **observed against simulated `dk_pts` per player per tournament round**, on both
> the training and the validation splits. Frame it as Gate A at a unit Gate A does not
> currently cover: the round is the unit the contest is decided at, and nothing scores
> `dk_pts` there today.
>
> Step 4 records what already exists and what does not. The tensor already carries
> `tournament_round` per scoring period, so changing the unit is a grouped sum and needs no
> re-simulation. What is missing is training-season tensors (only the two validation seasons
> are on disk; `season.allowed_seasons` already permits train seasons through
> `held_out.selection_split`), an observed per-player-per-round `dk_pts` artifact, and an
> emitter — the dashboard may not open a 90 MB `.npz` and reduce it.
>
> **Measure before you commit to a run.** Time one training season at a small `--n-sims`
> first; each shipped tensor is ~90 MB. Take the last two training seasons to match the
> validation pair rather than all 25, and say why in the doc.
>
> Reuse rather than re-derive: `preprocess.compute_dk_pts` and
> `features/scoring_periods.py`'s slot map for the observed side, and `model_cards`' binning
> helpers for the panels, so this page's ECDF ribbon and binned scatter are the same objects
> a model page draws and the reader learns the encoding once. Read the ribbon as a distance
> from the median replicate, never as in-or-out. State that Round 1 is seventeen weeks and
> Rounds 2–4 are two each, so the four panels are not four equal units.
>
> Adding a page moves Tournament & strategy to 9 and Draft board to 10 in `app.VIEWS`; the
> pinned `url_path`s do not change. Update the page counts in `dashboard/README.md`,
> `docs/dashboard-plan.md` and `CLAUDE.md`.
>
> Owed on the way out: the emitter's build gate; the three verification layers; tests; a
> `dashboard/decisions.py` entry; `make dashboard-audit`'s orphan count checked on the way
> in and out; and a "Step 4, as built" section.

---

### Prompt 5 — the Overview as a paper

> Read `CLAUDE.md`, `README.md`, `docs/dashboard-plan.md` (especially "Charter amendment
> 2026-08-10" and "Step 8, as built"), `dashboard/README.md` and
> `docs/dashboard-revision-plan.md` (step 5). Run this step **after** the new round-level
> page has landed, because it adds a route.
>
> Rewrite the Overview page in the publication structure `README.md` uses — Introduction,
> Methods, Results, Discussion — two to four sentences each, brief. The current five hero
> tiles read as numbers-vomit rather than an overview; the figures should live *inside* the
> prose instead of stacked as tiles.
>
> **What does not change:** bound 2 of the charter — every number on the page is read from
> an artifact and none is typed — and `overview.Spec`, the mechanism that enforces it. The
> lookups stay; `Reading` becomes a prose fragment rather than a tile's label, value and
> delta. Bound 3 does not move either: no decision registry, no provenance links, no
> reversal log, and the tests that parse for those stay green. Keep the pipeline diagram and
> the route block.
>
> **What you have to renegotiate explicitly:** bound 1, "one page, one screen", measured at
> 702 px on a 900 px viewport. Four prose sections plus the diagram plus nine route links
> will not fit it. Step 5 recommends amending the bound to "opens above the fold, scrolls no
> further than one screen more" and writing that amendment into `docs/dashboard-plan.md`'s
> charter section with a registry entry — not taking it silently. If you disagree, hold one
> screen and cut the diagram or the routes to pay for the prose, and record why.
>
> Owed on the way out: the three verification layers, with the rendered page height measured
> in Chrome at 1440×900 and recorded (the browser layer is what caught this page's first
> draft at 1,144 px); tests; a `dashboard/decisions.py` entry for the charter amendment;
> and a "Step 5, as built" section.
