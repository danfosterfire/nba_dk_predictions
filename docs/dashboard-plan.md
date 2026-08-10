# Dashboard Plan: A Visualization Surface

This is a planning doc, not a measurement report. It records direction; update it in place
as pieces land, the way `availability-plan.md` was.

> **Superseded 2026-08-08.** The previous version of this doc specified a nine-tab
> *project walkthrough* — a browsable rendering of the decision registry, hero numbers and
> provenance links, aimed at "the project architect wanting a birds-eye view of the
> decisions". It shipped in full on 2026-07-30 and is now removed. That work is not lost:
> the decisions themselves live in `dashboard/decisions.py`, `CLAUDE.md` and the plan docs,
> which were always the source of truth the dashboard distilled. Commit `e8e58b0` holds the
> nine tabs if a panel is ever wanted back verbatim. See
> [What the walkthrough was, and why it went](#what-the-walkthrough-was-and-why-it-went).

## Purpose and audience

**The dashboard shows data. Prose about the project belongs in the docs.**

The walkthrough answered "what was decided and why". That question turned out to be
answered better by the documents that were already answering it — nine tabs of rendered
markdown is a worse `README.md`, and keeping it truthful cost a registry, an audit, a
launchd job and a standing instruction. What no document can do is let you *look* at a
player, so that is what the dashboard is for now.

The rule that replaces the old altitude-and-reversals pair is simpler:

- **One view does one thing, and the thing is a picture.** A view answers a question you
  would otherwise answer by opening a parquet in pandas. If a panel's content is a
  paragraph, it belongs in `docs/`.
- **Interpretation is allowed, but it must be pinned.** A chart that says "this axis is
  interior bigs" is more useful than one that says "PC1", and it is also a claim that can
  go stale. Where the dashboard interprets, the interpretation carries a machine-checkable
  anchor — see [Pinning the interpretation](#pinning-the-interpretation).

### Charter amendment 2026-08-10 — one page of prose, capped and pinned

The expansion below adds an **Overview** page, which is prose, which is the thing this
charter deleted. The exemption is granted deliberately and bounded, because the audience
changed: the walkthrough was for the project architect and lost to `docs/`; the Overview
page is for a **portfolio reader who arrives at a URL with no context** and will not open a
repository. That reader is not served by any document, because they will not read one.

Three bounds make it a different object from the walkthrough, and they are the terms of the
exemption:

1. **One page, one screen.** If it scrolls, it has become the walkthrough again.
2. **Every number on it is read from an artifact.** Typed prose may say *what the project
   does*; it may not state a *result*. A hero tile showing 400.5 dk_pts of season-total MAE
   reads `season_total_metrics.csv` like every other figure on the site. This is the
   mechanism, not a preference — a chart of an artifact cannot disagree with the artifact,
   and the walkthrough died of hand-typed claims drifting from the documents that made them.
3. **No decision registry, no provenance links, no reversal log.** Those are what made the
   walkthrough a documentation surface, and they stay in `decisions.py` and `docs/`.

Registered as `dashboard-overview-page-exemption` in `dashboard/decisions.py`.

What carries over unchanged: **every figure is read from an artifact a `make` target
produced**, and **nothing in `dashboard/` imports from `src/`** (pinned by
`test_the_dashboard_imports_nothing_from_src`). The dashboard cannot refit, re-project or
re-cluster anything; if a number is not in an artifact, it does not go on the page.

---

## View 1 · The PCA player-style fingerprint

**Shipped 2026-08-08.** `dashboard/app.py`, with its pure layer in `dashboard/pca.py` and
its two figures in `dashboard/charts.py`.

### The question

Where does one player-season sit on the axes the season matrix actually varies along, and
who else has been there?

### What is on screen

Two dropdowns — season **S** and player **P** — then three things:

| element | what it shows |
|---|---|
| **The radial chart**, left | P's score on each of the first ten principal components. Each component is a spoke; the radius is that component's score in **standard deviations from the league mean**. |
| **One component panel**, right | The component whose spoke was last clicked: its strongest loadings as a diverging bar chart, P's score on it, the rotation player-season at each end of the axis, and a paragraph on what the loadings say. |
| **Nearest neighbours**, below | The three player-seasons closest to P·S in PCA space, optionally overlaid on the radial chart. |

**One component at a time, opened by clicking its spoke.** The first version arrayed all
ten loadings panels around the chart, which crammed the page badly enough that each
panel's explanation had to be hidden in an expander to fit — and an explanation behind a
click is an explanation nobody reads. Showing the one component the reader asked about
buys the room to print it. The selected spoke is enlarged and ringed so the chart says
what the panel is explaining, and a `Component` dropdown mirrors the same choice for
anyone not using a mouse.

### The decisions inside it

Recorded here because each was a real fork, and several are the kind that would otherwise
be silently re-decided by whoever edits next.

**Ten components, Tier A, `within_season`.** Ten reaches 67.3% cumulative variance and
keeps the spokes 36° apart, which is about as tight as a labelled radial chart gets.
`within_season` z-scores inside each season, so a player is described relative to his own
league rather than to the 3-point revolution — the era-neutral mode that already feeds
archetypes and modeling. The `pooled` and Tier B twins exist on disk and are **deliberately
not offered as a toggle**: the ten titles are read off *these* loadings, and a title written
for one decomposition is wrong for another. Offering the toggle would mean writing forty
titles and checking all four sets on every refit.

**The radial axis is fixed at ±2 SD for every player and every season, and off-axis scores
are pinned rather than rescaled.** An axis that adapts to its player makes every fingerprint
look equally extreme, which destroys the only thing the chart is for — comparing shapes.
Pinning has a cost: at −2 SD the point lands on the centre and several spokes can collapse
into the origin. It is paid explicitly. A pinned marker is drawn as an **open** circle
rather than a filled one, the hover reports the true score rather than the clipped one, and
a caption under the chart lists every pinned component with its real value.

**PC scores are divided by each component's own standard deviation.** A raw PC score is in
the units of the standardized feature space, so PC1's spread is 5.87 and PC10's is 1.68 —
the same raw number is a different distance from average on each spoke, and a radial chart
that plotted raw scores would be reading ten different axes as one. One scale across all 30
seasons, not one per season: the per-season spreads differ by about 5%, well inside the
width of a plotted line.

**Nearest neighbours use raw Euclidean distance, not the SD-scaled scores.** Distance on the
raw scores over the retained subspace *is* distance in the standardized feature space the
PCA was fitted on, so the high-variance style axes dominate as they should. Scaling every
component to one SD first is a Mahalanobis distance and gives PC10 — team pace, 1.9% of
variance — the same say as PC1. A test pins the difference with a case where the two
metrics disagree.

**A player's own other seasons are always excluded from the neighbour list.** They are
usually his three nearest neighbours, which is true and tells you nothing.

**Loadings panels are balanced across the sign, not a top-`n` by magnitude.** This is the
one that was caught by rendering the page rather than by reasoning about it. PC1's eight
largest loadings are *all positive* — rebounding, paint scoring — and its negative end, the
3-point attempt and scoring shares, lands ninth and tenth by a thousandth. A plain top-8
panel therefore read "rebounding big" and dropped the "not a shooter" half of an axis that
is defined by the opposition. Each side now gets half the slots, a short side is backfilled
by magnitude, and the rows are sorted by signed loading so the bars diverge around zero.

**The chart is drawn on cartesian axes, not on a plotly `polar` subplot, because
Streamlit cannot report a click on a polar trace.** This was measured, not assumed: a
probe app rendering a `Scatterpolar` and a `Scatter` side by side, both with
`on_select="rerun", selection_mode="points"`, returns `[]` for every click on the polar
markers and a full point payload for the cartesian ones. Since clicking a spoke *is* the
interaction, the polar subplot had to go. The disc is now placed by hand — rings and
spokes as shapes, spoke and ring labels as annotations, the polygon as a `Scatter` in
`(r·cosθ, r·sinθ)` — which costs about sixty lines and buys back two things beyond the
click: exact control over the ring labels, which a `polar` axis will not give either
(it rotates radial tick text by `angle - tickangle` and ignores `ticksuffix` outright),
and a grid made of shapes, so nothing but a data point can be picked up as a click.

An invisible marker trace with a 30px radius sits under the visible one in the same
point order, so a click *near* a vertex still lands on it.

**The first spoke is at the top and the rest run counterclockwise**, so components 1–5
descend the left of the disc and 6–10 climb the right. That was chosen when ten panels
were arrayed around the chart and their column order had to follow the circle; it is kept
because the ordering is a property readers learn once, and a fixed layout is worth more
than the reason it was originally fixed.

### Pinning the interpretation

The ten titles are the only typed content on the page. Everything else — the exemplar
player-seasons at each end of an axis, the variance shares, the loadings, the scores — is
derived live from the three `pca_tierA_within_season_*` artifacts.

That leaves one failure mode, and it is a real one: **a PCA's signs are arbitrary.**
`sklearn` is free to return −PC1, and a refit could flip any axis without changing the fit
at all, at which point "Paint big, not shooter" would be labelling the shooters. So each
component carries an **anchor** feature whose loading defines the labelled direction, and
`pca.orient()` flips the score *and* the loading column together whenever a refit lands on
the other sign. Today every anchor already loads positive, so orientation is a no-op — which
is the point. It is a guard, not a correction, and a test asserts the no-op so that a refit
which flips an axis is *noticed* rather than silently absorbed.

| PC | share | title | anchor |
|---|---|---|---|
| 1 | 23.3% | Paint big, not shooter | `adv_reb_pct` |
| 2 | 15.2% | On-ball scoring load | `adv_usg_pct` |
| 3 | 7.0% | Assisted, efficient, not creating | `sco_pct_ast_fgm` |
| 4 | 5.9% | Leaky defence, negative impact | `adv_def_rating` |
| 5 | 3.5% | Mid-range at a slow pace | `sco_pct_pts_2pt_mr` |
| 6 | 3.2% | Double-double passing hub | `adv_ast_ratio` |
| 7 | 2.9% | Clean finishing, few turnovers | `adv_ts_pct` |
| 8 | 2.4% | Mid-range big who steals | `bas_stl` |
| 9 | 2.1% | Clutch free-throw volume | `clu_fta` |
| 10 | 1.9% | Fast team pace | `adv_pace` |

Two of these are worth reading as findings rather than as labels, and the view shows them
rather than hiding them:

- **PC9 is a small-sample artifact, and it looks like one.** It is almost entirely the
  clutch family — a few hundred possessions a season — so its tail is heavy and its
  unfiltered extremes reach ±18 SD on players nobody would recognise. That is why the
  exemplars are drawn from rotation seasons only (`min ≥ 24`, `gp ≥ 45`) rather than from
  the raw argmax.
- **PC10 is a team attribute reaching a player row through the roster join.** Pace, three
  ways, and little else. Worth seeing precisely because it is not a skill.

### Verification, as run

Three layers, and each one caught something the layer above it could not.

**`AppTest`** — `app.py` executed for real in both appearance modes, plus a season switch,
a player switch, the neighbour overlay, the component selector and the same-season toggle.
Final state: **2 charts, 6 tiles, 3 tables, 4 selectors, 0 exceptions, 0 missing-artifact
warnings**, light and dark.

**Figures rendered to PNG** through kaleido and looked at. This caught the
balanced-loadings problem above and a legend that listed the comparison player before the
selected one.

**The live page screenshotted and driven in a real browser**, via Playwright against the
installed Chrome. This is the layer that earned its keep:

- The radar rendered a literal **"undefined"** where its title would be. `apply_theme` set
  `title_font` on a figure with no title, which leaves plotly.js a title object with no
  text; kaleido draws nothing for that, so the PNG check could not see it. Fixed in
  `apply_theme` for every figure, and pinned by a test.
- **Streamlit returns no selection for a click on a polar trace.** A probe app comparing
  polar and cartesian traces in the same page settled it, and the radar was rebuilt on
  cartesian axes as a result — see above. Then the fix itself was verified by clicking
  spoke 8 and spoke 6 in the browser and asserting the panel heading followed.
- The header's five metric tiles clipped their own values inside a narrow column. Moving
  the row to full width plus a small style block (`stMetricValue` to 1.4rem, the label to
  0.75rem — the test IDs were checked against this Streamlit's frontend bundle rather than
  assumed) fixed it.

The rule this leaves behind is in `dashboard/README.md`: `AppTest` proves the page runs, a
PNG proves the figure is legible, and only a browser proves the page is.

---

# The expansion — from one view to nine pages

**Planned 2026-08-10. Nothing below is built.** This section is the design; each numbered
step in [The build order](#the-build-order) is meant to be handed to a fresh session on its
own.

The goal is a single surface that presents the whole project — the PCA view keeps its
content under the name **Player fingerprints**, and eight pages join it covering the model
layer, the non-model inputs, the tournament layer, and the live draft board.

## The three findings that shape the design

Established by reading the code and the artifacts on disk before writing any of this down.
Each one changes what gets built rather than how it looks.

### 1. `st.tabs` is the wrong container, and `st.navigation` is the right one

**Streamlit executes the body of every tab on every rerun.** Tabs are a client-side
affordance: the Python inside each `with tab:` block runs whether or not that tab is
visible, and the inactive content is hidden with CSS. Nine tabs would therefore mean every
interaction anywhere re-runs all nine, including whichever one loads the 90 MB simulation
tensor. The page would be unusable, and no amount of caching fixes it because the cost is
the *rendering*, not only the I/O.

`st.navigation` / `st.Page` (present in the installed Streamlit 1.60) runs **only the
selected page's script**. Pages live in one server process, so `@st.cache_data` and
`@st.cache_resource` are shared across them — a tensor loaded by the draft board stays warm
if the reader navigates away and back.

So the deliverable is a **sidebar-navigated multipage app**, and the user-facing word "tab"
maps to a page. This is not a cosmetic substitution: it is the only structure in which item
5 below is feasible at all.

### 2. The model pages are blocked on artifacts, not on UI

This is the large piece of work in the expansion, and it is pipeline work rather than
dashboard work.

The dashboard's binding rule is that it reads artifacts and never imports `src/`. Against
the model detail pages, here is what is actually on disk today:

| what a model page needs | what exists | gap |
|---|---|---|
| coefficient posteriors | `stan_availability_coefficients.csv`, `stan_games_played_coefficients.csv` | **18 of 20 heads have none** |
| diagnostics table | `stan_*_diagnostics.csv` for every head, plus the posteriors `manifest.csv` | none — reuse directly |
| PIT / calibration | `stan_availability_pit.csv`, `stan_games_played_pit.csv`; components carry a scalar `val_pit_ks` in `stan_component_metrics.csv` | **no ECDF band for any head** |
| predicted vs observed rows | `availability_predictions.csv`, `stan_availability_predictions.csv`, `season_total_predictions.csv`, `stan_games_played_predictions.csv` | **no component, minutes, composition or game-length rows; and every one of these is validation-only** |
| the features fed to each head | nothing | **all of it** |

The tempting shortcut is `data/features/posteriors/{train,train_val}/*.pkl`, which does
carry thinned coefficient draws and a design recipe for twenty heads. **It must not be read
from the dashboard.** Unpickling it imports `src.models.posteriors` — invisible to the
`ast`-based guard, which only sees static imports, so the guard would pass while the rule
broke — and the object it returns carries a fitted `StandardScaler` and the ordered design
steps, which is precisely the capability to score an arbitrary frame. A dashboard holding
that is a dashboard that can silently disagree with the fit it is describing, which is the
one failure the rule exists to prevent.

**So a new emitter stands between them**: `src/models/model_cards.py`, `make model-cards`,
reading the posterior pickles and each head's own variant ladder and writing flat,
long-format, dashboard-shaped artifacts. The dashboard reads those and only those. The
contract is specified in [The model-card artifact contract](#the-model-card-artifact-contract)
below and gets its own `docs/model-cards-plan.md` when it is built.

Two rules the emitter inherits and must not be allowed to quietly break:

- **It goes through `held_out.selection_split`.** The user asked for train *and* validation
  scatters, which is exactly right and is also the whole split surface. Every emitted row
  carries a `split` column whose only legal values are `train` and `validation`. There is no
  test column, for the same reason the sweeps no longer emit one.
- **It reads the `train` posterior window, not `train_val`.** At `train_val` the validation
  rows were in the fit, and a "validation" scatter drawn from those coefficients is an
  in-sample scatter wearing the wrong label. `posteriors.require_window` is there to refuse
  the wrong one rather than discover it in a picture.

### 3. Items 3 and 4 need almost no new pipeline work

The non-model inputs page and the tournament page are the cheap half of the expansion,
because their artifacts already exist and nothing reads them. `make dashboard-audit`'s
orphaned-artifact check — "what has the pipeline written that nothing looks at?" — was
already named in this doc as the right way to choose the next view, and it points here.

On disk and unread: `adp_panel.parquet`, `adp_profile.csv`, `adp_match_audit.csv`,
`residual_correlation.csv`, `serial_correlation.csv`, `bonus_calibration.csv`,
`game_length_coverage.csv`, `draft_pool_coverage.csv`, `variance_budget.csv`, and the
entire strategy family — `strategy_sweep.csv` (88 rows over the full axis grid),
`strategy_paired.csv` (440), `strategy_shipped`, `strategy_realized`, `strategy_null`,
gates C and D, `bracket_structure`, `bracket_entries`, `bracket_null`, `draft_field`,
`draft_adp_curve`, `draft_gate_b`, `sim_season_gate_a`, plus `economics.py`'s derived
contest arithmetic.

That is the richest material in the repo and it has never been drawn.

---

## The pages

Nine, in sidebar order. "Class" pages carry a head selector; the others do not.

| # | page | source | new artifacts? |
|---|---|---|---|
| 1 | **Overview** | hero tiles from existing metrics CSVs | no |
| 2 | **Player fingerprints** | today's `app.py`, moved unchanged | no |
| 3 | **Availability** | model cards + `stan_availability_*`, `stan_games_played_*` | **yes** |
| 4 | **Minutes** | model cards + `stan_minutes_*`, `stan_composition_*`, `minutes_unification.csv` | **yes** |
| 5 | **Box-score components** | model cards + `stan_component_*` | **yes** |
| 6 | **Game length** | model cards + `stan_game_length_*` | **yes** |
| 7 | **Inputs beyond the heads** | ADP, injury capture, copula, serial correlation, bonus overdispersion | small |
| 8 | **Tournament & strategy** | the strategy / bracket / draft families, `economics.py` | no |
| 9 | **Draft board** | today's `draft_room.py`, as a page | no |

### Pages 3–6 — the model detail views, and the head selector

The user's item 2 asks for one tab per **model class** with a dropdown for the specific head
within a class. The four classes fall out of the likelihood and the unit, not out of the
`.stan` file — `betabinomial_glm.stan` serves availability, minutes, four conversion heads
and overtime onset, so grouping by source would put unrelated things together:

- **Availability** — `availability` (beta-binomial over games played out of team games) and
  the games-played tenure decomposition (`gp_entry`, `gp_exit`, `gp_onset`, `gp_duration`).
- **Minutes** — the marginal `min | available` head and the team-game composition. This page
  has a second job the others do not: it is where `minutes_unification.csv` renders, and that
  comparison is the repo's cleanest demonstration that *a head is only a model at the unit it
  was scored at*. One posterior, two units, opposite verdicts.
- **Box-score components** — the seven negative-binomial counts and four beta-binomial
  conversions, eleven heads behind one dropdown.
- **Game length** — overtime onset (beta-binomial) and overtime depth (beta-geometric).

**Every page states its own unit, prominently.** The component heads are fitted
season-collapsed on ~10,000 player-season rows; the composition is per team-game on ~984,000;
game length is per game; availability is per player-season. A reader comparing an R² across
pages without knowing that is being misled, and this repo has already paid for that lesson.

Each head renders the same seven blocks, in this order, on one long scrolling page — the user
was explicit that scrolling is fine and cramming is not:

1. **What this head is** — three or four sentences and the likelihood. The one place typed
   prose is allowed on these pages, and it describes the *specification*, never a result.
2. **The features it was fed** — a small-multiple grid of histograms, one per feature, plus a
   table of n / mean / sd / missing-share. Imputation flags shown as their own share.
3. **Feature relationships** — see the scope note below; a correlation heatmap, not a full
   pair plot.
4. **Coefficients** — a horizontal bar of posterior means with 95% credible intervals,
   sorted, and spline bases grouped by `term_family` so a 12-knot basis does not swamp the
   panel.
5. **Predictive calibration** — the observed ECDF drawn over a ribbon of posterior-predictive
   ECDF quantiles (50 / 80 / 95%), train and validation side by side.
6. **Predicted vs observed, and residuals** — four panels: fitted-vs-observed and
   residual-vs-fitted, each for train and validation, drawn from the binned density with a
   bounded subsample overlaid for texture.
7. **Diagnostics** — R̂, ESS bulk/tail, divergences, treedepth saturation, draws, wall clock,
   CmdStan version and git SHA, read from the existing `stan_*_diagnostics.csv` and the
   posteriors manifest. No new artifact.

**Block 3 is a correlation heatmap, not a pair plot. Settled 2026-08-10.** The original
sketch asked for pair plots of the features. A full pairwise matrix over 12–20 features is
150–400 panels — unreadable at any size that fits a page, and an artifact carrying every
pairwise 2-D binning is large for something nobody reads. What the pair plot is actually
being asked ("is anything in here collinear, and what does the joint look like where it
matters") survives the substitution intact: a **feature correlation heatmap**, for which
there is precedent in `feature_correlation_tierA.parquet`, plus **one on-demand 2-D
density** for a reader-selected pair, precomputed only for the top ~20 most-correlated
pairs per head. Registered as `feature-correlation-not-pair-plots`, at status `open` until
the emitter writes it — the fork is decided, the block is not built.

### Page 7 — Inputs beyond the heads

The user's item 3. The through-line is **everything the simulator consumes that is not a
fitted coefficient**, which is a genuinely distinct kind of input and is currently invisible:

- **ADP** — the `adp_panel` and its three-dates-per-row point-in-time discipline, the
  DK↔FantasyPros agreement from `adp_profile.csv`, the name-matching audit, and the four of
  nine seasons that point-in-time safety costs.
- **The capture programs on a deadline** — injury-report PDFs, the ESPN feed and the DK
  board, drawn as a coverage calendar. This one is worth building precisely because the data
  is *perishable*: a gap in the calendar is unrecoverable, and a picture of it is an
  operational alarm, not a decoration. Likely needs a small emitter, since `make adp-status`
  and `make capture-status` print rather than write.
- **The four calibrated simulator inputs** — the residual copula as a heatmap, the block
  variance inflation from `serial_correlation.csv`, the bonus overdispersion from
  `bonus_calibration.csv`, and the game-level minutes dispersion. Each carries its
  `fit_window`, and the page should show the window rather than hide it, because *which
  window to consume is decided by what the number will be scored against* is a real project
  finding.

### Page 8 — Tournament & strategy

The user's item 4, and the page the whole project builds toward. No new artifacts. Four
blocks:

1. **The contest structure** — `bracket_structure.csv` and `economics.py`: five real
   tournaments, four elimination rounds, Round 1 a zero-consolation knockout in every one,
   and rake as a break-even edge hurdle because that is the unit an edge compares in.
2. **The sweep** — `strategy_sweep.csv` over ranking source, blend weight, position and
   exposure caps, stacking, objective and entry count. The natural figure is lift versus the
   null with confidence intervals, faceted by axis, with the break-even hurdle drawn as a
   reference line.
3. **Simulated versus realized** — `strategy_shipped.csv` beside `strategy_realized.csv`.
   The two halves do different jobs and the page must say which is the tuning surface and
   which is the honest readout.
4. **The paired comparisons** — `strategy_paired.csv`, 440 rows of gap with intervals and a
   `resolved` flag. A gap whose interval crosses zero is the most useful thing on the page
   and should be styled as such rather than buried.

### Page 9 — The draft board

The user's item 5, and the answer is **yes, it is feasible, and it is feasible only because
of finding 1.** As a `st.tabs` child it would be a disaster; as an `st.Page` its script does
not run until the reader navigates to it, and `@st.cache_resource` keeps the ~40 MB reference
field warm across navigation.

Two conditions:

- **`make draft-room` keeps working as a standalone launch.** Draft night is a
  thirty-second clock and should not share a process with anything. The page file stays
  directly runnable; the unified app imports its `render()`.
- **The `src/` exemption stays narrowed rather than widened.** `SRC_IMPORTERS` in
  `tests/test_dashboard.py` continues to name exactly one file, held to `src.sim`. Moving the
  draft room into the app must not become the precedent that lets page 3 import a model.

---

## The model-card artifact contract

Sketch, to be firmed up in `docs/model-cards-plan.md` when step 3 is built. All files
long-format, keyed by `head`, under `outputs/predictions/`. Sizes are the reason for every
binning decision — a 984,000-row composition scatter is not an artifact, it is a copy of the
data.

| artifact | grain | approx rows |
|---|---|---|
| `model_card_index.csv` | head | 20 — unit, family, variant, n_fit, label, class |
| `model_card_coefficients.csv` | head × term | ~600 — mean, sd, q2.5/25/75/97.5, `term_family` |
| `model_card_features.csv` | head × feature × split × bin | ~30,000 — binned counts plus per-feature n/mean/sd/missing |
| `model_card_feature_corr.csv` | head × feature × feature | ~5,000 |
| `model_card_ecdf.csv` | head × split × grid point | ~8,000 — observed ECDF and predictive quantile band |
| `model_card_calibration.csv` | head × split × 2-D bin | ~4,000 — fitted vs observed density, and residual density |
| `model_card_sample.parquet` | head × split × row | ~200,000 capped — bounded subsample carrying fitted, observed, residual |

Three things the emitter must do that are easy to get wrong:

- **Cap the draws and the rows before generating a posterior predictive.** The composition
  head at full row count times a thousand draws is not affordable; 200 draws over a
  subsample is, and the ECDF band is stable well before that.
- **Verify the recipe the way `posteriors.py` already does.** That module reproduces each
  head's own design matrix and predictions exactly at build time and fails the build rather
  than writing a wrong artifact. The model cards are downstream of the same recipe and get
  the same check, or they will drift silently.
- **Declare the unit per head in `model_card_index.csv`** and let the page read it, rather
  than hard-coding a unit string in the dashboard where it can go stale.

---

## The build order

Eight steps. Each is a self-contained session with its own deliverable and its own
verification; the ordering is a dependency ordering, not a preference.

**Step 1 · The multipage shell.** Convert `app.py` into an `st.navigation` entrypoint;
move the PCA view verbatim into `dashboard/views/fingerprints.py` behind a `render()`;
lift the appearance toggle into shared state so it survives navigation; keep `make
dashboard` pointing at the same entrypoint. Ships with one real page, so the shell is
proved before anything depends on it.

**Step 2 · Tournament & strategy (page 8).** Deliberately second: it is the richest page,
needs zero new pipeline work, and it exercises the multipage shell with a genuinely
different layout before the expensive step lands.

**Step 3 · The model-card emitter.** `src/models/model_cards.py`, `make model-cards`,
`docs/model-cards-plan.md`, and the tests. No dashboard work at all. The largest step and
the one most worth handing a fresh session with the whole context budget.

**Step 4 · The generic model renderer plus the Availability page (3).** Build the seven
blocks once, against one class, so the renderer is proved before it is reused three times.

**Step 5 · Minutes, Box-score components, Game length (pages 4–6).** Mostly configuration
against the step-4 renderer, plus the `minutes_unification` two-unit comparison, which is
bespoke.

**Step 6 · Inputs beyond the heads (page 7).** Includes the small capture-calendar emitter.

**Step 7 · The draft board as a page (9).** Plus keeping `make draft-room` standalone.

**Step 8 · Overview (page 1).** Last, on purpose: its hero tiles link into the pages, so it
cannot be written until they exist, and writing it first would make it a table of contents
for pages that do not.

## What each step owes on the way out

Not negotiable, since these are the mechanisms that keep the surface from drifting into
being documentation again:

- **The three verification layers** from `dashboard/README.md`: `AppTest` in both appearance
  modes proves the page runs, a figure rendered to PNG proves the figure is legible, a real
  browser proves the page is. All three are runnable as of 2026-08-10 — `kaleido` and
  `playwright` were added to `requirements.txt`, having been used ad hoc during the PCA
  build and never installed. **Neither needs a browser download**: kaleido finds the system
  Chrome by itself, and playwright reaches it with
  `p.chromium.launch(channel="chrome")`, so `playwright install` and its ~150 MB of
  bundled browsers are not required. Both were probed end to end when they were added.
- **Tests in `tests/test_dashboard.py`**, plain `assert` with synthetic builders, exercising
  the pure layer directly rather than through a rendered page.
- **A `dashboard/decisions.py` entry** for every load-bearing fork, per the standing
  instruction in `CLAUDE.md`.
- **This doc updated in place**, and the new artifacts accounted for so `make
  dashboard-audit`'s orphan count moves the right way.

---

## Structure

```
dashboard/
  README.md       # the rules a new view has to follow
  __init__.py
  app.py          # the PCA fingerprint view: page, controls, layout
  pca.py          # its pure layer — orientation, SD scaling, loadings, neighbours
  charts.py       # fig_radar / fig_loadings
  theme.py        # SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  artifacts.py    # load_cfg, features_dir, read_table, optional
  decisions.py    # NOT the dashboard — the project decision registry (see below)
  economics.py    # NOT the dashboard — contest arithmetic for the drafting layer
  audit.py        # NOT the dashboard — make dashboard-audit
```

`app.py` and `artifacts.py` are the only modules that import Streamlit. Everything else is
pure, which is what lets `tests/test_dashboard.py` exercise the palette rules, the component
spec, the scaling, the neighbour metric and both figures as plain functions.

### Three files that stayed, and are not the dashboard

The instinct on an overhaul is to delete everything in the directory. These three are load
bearing elsewhere and only live here for historical reasons:

- **`decisions.py`** — the project's decision registry, wired to a standing instruction in
  `CLAUDE.md` and to `make dashboard-audit`. It was *distilled from* the docs for the
  walkthrough to render; the docs remain the source of truth and the registry outlives its
  renderer.
- **`audit.py`** — `make dashboard-audit`, a report on registry drift. Its orphaned-artifact
  check counts an artifact as accounted for if either a registry entry names it *or* a
  string literal in `dashboard/` reads it. Removing nine tabs removed a lot of literals, so
  the orphan count is expected to have risen; that is a registry-coverage question, not a
  dashboard one.
- **`economics.py`** — rake, break-even hurdle and advance rates derived from the two real
  tournament CSVs. Pure arithmetic the drafting and backtest layers will consume, quoted in
  `README.md`, and never a rendering concern.

Moving them out of `dashboard/` is defensible and is **not** being done as part of this
overhaul: it would touch the `Makefile`, `CLAUDE.md`, `src/docs_audit.py` and two dozen
tests for a filing improvement.

---

## What the walkthrough was, and why it went

Kept short, and kept at all because the reasons it existed were good ones and the next
person to want a decisions tab should see them.

It was nine topic tabs over 117 hero numbers, 50 charts and 57 tables, every figure read
from an artifact, organised around a 97-entry decision registry with a closed status
vocabulary in which `withdrawn` was first class. It worked. Its machinery caught three real
problems in its first hours: two registry entries overturned by a fit that landed
mid-build, and an R² column that had been wrong in two documents because it was hand-typed
and never recomputed.

It went for one reason, which is not a fault in the execution: **it was documentation
rendered as an app.** Every claim on it had to be kept in sync with a document that already
made the claim, and the four anti-drift mitigations — a precedence rule, a `CLAUDE.md`
instruction, a make target and a weekly launchd job — were the cost of that duplication
rather than a feature. A dashboard that reads artifacts and draws pictures has no such
drift surface, because a chart of an artifact cannot disagree with the artifact.

Two mechanisms from it are worth keeping in mind for future views:

- **The provenance rule** — a figure that cannot be reproduced from a `make` target does not
  go on the page — carries over verbatim and is cheap now that the page is charts.
- **The orphaned-artifact check** was the most valuable thing the audit did, because it asks
  the inverse question: what has the pipeline written that nothing looks at? That is a good
  question for choosing view 2.

The pieces that are gone: `dashboard/tabs/` (nine modules), `dashboard/layout.py`
(`decision_card`, `status_badge`, `pending_marker`, `stat_tiles`, `table_view`), and the
generic figure builders `fig_heatmap` / `fig_bars` / `fig_lines` / `fig_scatter`, which no
view calls. All are in commit `e8e58b0`.

---

## Tests

`tests/test_dashboard.py`. The palette, registry, audit and economics tests carry over
unchanged — the check that the overhaul changed no behaviour outside the view. The tab
tests, the four generic figure builders' tests and the `pipeline_health` / `inventory`
tests are removed with the code they covered.

New, in the same plain-`assert` synthetic-builder style:

- **Orientation.** A flipped anchor is detected; `orient()` flips score and loading
  *together* and leaves the caller's frames alone; a missing anchor defaults to no flip.
- **Scaling.** SD units put every spoke on one scale; a degenerate component does not
  divide by zero; `clamp` pins rather than rescales.
- **The fingerprint.** The true score survives beside the pinned radius, so a hover cannot
  read 2.0 for a 17.8; an absent player-season raises. The hover string is preformatted to
  one decimal rather than left to a d3 format spec, which does not apply to a value
  arriving from a mixed-dtype `customdata` array and let a `1.7174781203` onto the screen.
- **The click.** A payload resolves by spoke label or by point index under any of the four
  key spellings; the polygon's repeated closing vertex wraps to the first component; an
  empty or unrecognisable selection changes nothing.
- **Loadings.** Sorted signed; the balance rule is pinned against the *real* PC1, including
  an assertion that the naive top-8 rule would have shown only positives; a one-sided
  component backfills rather than shrinking the panel.
- **Neighbours.** Own seasons excluded; season restriction; sorted with a distance; and the
  raw-versus-Mahalanobis case where the two metrics pick different players.
- **Figures.** The score-to-radius map puts the centre at −2 SD and the rim at +2; the axis
  is fixed regardless of how extreme the player is and every vertex stays inside the rim;
  the polygon closes; the grid is shapes and the only traces are the hit layer and the
  data, so nothing else can be reported as a clicked point; a pinned spoke is drawn open
  and hovers its true score; the selected spoke is enlarged and ringed; an overlay takes
  slot 2 and turns the legend on; loading bars take the diverging ends rather than two
  categorical slots; and an untitled figure carries an empty title rather than a bare font.
- **The artifact contract.** Every component's anchor is a real feature in the shipped
  loadings, every component names two real player-seasons, and the shipped decomposition
  still points the labelled way.

---

## Next views

**Superseded 2026-08-10 by [The expansion](#the-expansion--from-one-view-to-nine-pages)**,
which specifies the next eight pages concretely. Two of the four sketches below were taken
up there — posterior draws for a player-season became the model-card predictive blocks, and
the draft board became page 9. The two that were not are kept, because they are still good
and still unclaimed:

- **The player-season trajectory.** The same fingerprint over a career, as a small-multiple
  or an animated path through the first two components. `pooled` mode is the right artifact
  for this one, because era is the axis you want visible.
- **The archetype membership vector.** `archetypes_*` is a soft membership over a continuum
  with a silhouette peaking at 0.181, so the honest picture is a stacked bar of memberships,
  never a hard label.

## Publishing

Not a goal. Worth noting that this view needs exactly three files —
`pca_tierA_within_season_{scores.parquet, loadings.parquet, variance.csv}`, about 4.3 MB
total — so a deploy bundle is a filtering job rather than a rewrite. `component_targets.parquet`
(37 MB) is not read and must not start being read.
