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

1. ~~**One page, one screen.** If it scrolls, it has become the walkthrough again.~~
   **Amended 2026-08-10, see below:** *opens above the fold, and scrolls no further than
   one screen more.*
2. **Every number on it is read from an artifact.** Typed prose may say *what the project
   does*; it may not state a *result*. A sentence quoting 400.5 dk_pts of season-total MAE
   reads `season_total_metrics.csv` like every other figure on the site. This is the
   mechanism, not a preference — a chart of an artifact cannot disagree with the artifact,
   and the walkthrough died of hand-typed claims drifting from the documents that made them.
   **A typed sentence carries no digit at all**, which is the same bound one level down —
   see the amendment below.
3. **No decision registry, no provenance links, no reversal log.** Those are what made the
   walkthrough a documentation surface, and they stay in `decisions.py` and `docs/`.

Registered as `dashboard-overview-page-exemption` in `dashboard/decisions.py`.

#### The amendment to bound 1 — 2026-08-10, and it is a ceiling rather than a waiver

The page was rewritten the same day into `README.md`'s four sections — Introduction,
Methods, Results, Discussion — because five hero tiles are five numbers with no argument
around them, which is an overview of nothing. Four sections of two-to-four sentences do not
fit 900 px beside the pipeline diagram and nine route links, so **either the bound moved or
the page did**, and the only things left to cut were the diagram (the one figure that
carries the whole shape at a glance) and the route block (the reason the page was built
last). Amending is the smaller loss, and it is taken here rather than silently:

> **Bound 1, as amended.** The page **opens above the fold** — the title, the Introduction
> and its first artifact-read figure are visible without scrolling — and **ends inside one
> more screen**. Two viewports is a ceiling, not a target, and it is still a browser
> measurement rather than an intention.

The bound's own reason survives the change, which is why the ceiling is hard. "If it
scrolls it has become the walkthrough again" was shorthand for a real failure: the
walkthrough was nine tabs of rendered decision registry, and four paragraphs that open above
the fold are not that. Two screens is close enough to the old bound that the page still
cannot grow a registry, a reversal log or a fifth section without visibly breaking it, and
far enough that the prose the rewrite asked for is affordable.

**Measured on the way out, since that is what makes this a bound at all: 1,036 px on a
900 px viewport and 1,161 px on a 1280×800 laptop** — inside the new ceiling by 764 and 439
px, and *below* the 1,144 px first draft this bound rejected when the page was tiles. The
paper is cheaper than the tile row it replaced because two sections to a row is both a
readable measure and half the height of a full-width column.

Bounds 2 and 3 do not move. Bound 2 **gains a half**, and it is the half the rewrite made
necessary: a hero tile had nowhere to put a typed number — its value came from a lookup and
its label was a label — while a paragraph has room for one mid-sentence, which is exactly
how the walkthrough's claims drifted. So a typed fragment now carries **no digit at all**
(`test_typed_prose_carries_no_digit`), and the contest's own rules are spelled in words:
*sixteen players* is a rule, `46.4%` is a measurement, and on this page a digit means the
second kind.

Registered as `overview-bound-one-amended-for-the-paper`, with the three height readings on
`overview-fits-one-screen-by-measurement`.

What carries over unchanged: **every figure is read from an artifact a `make` target
produced**, and **nothing in `dashboard/` imports from `src/`** (pinned by
`test_the_dashboard_imports_nothing_from_src`). The dashboard cannot refit, re-project or
re-cluster anything; if a number is not in an artifact, it does not go on the page.

---

## View 1 · The PCA player-style fingerprint

**Shipped 2026-08-08.** `dashboard/views/fingerprints.py`, with its pure layer in
`dashboard/pca.py` and its two figures in `dashboard/charts.py`. It was the whole of
`app.py` until 2026-08-10, when [step 1](#step-1-as-built--the-multipage-shell) moved it
into the multipage shell unchanged; `app.py` is now the `st.navigation` entrypoint.

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

**Planned 2026-08-10. Steps 1, 2 and 3 shipped the same day; the rest is not built.** This
section is the design; each numbered step in
[The build order](#the-build-order) is meant to be handed to a fresh session on its own.
What the shipped steps actually landed, and the things they measured that the design did not
anticipate, are in [Step 1, as built](#step-1-as-built--the-multipage-shell),
[Step 2, as built](#step-2-as-built--tournament--strategy) and
[Step 3a, as built](#step-3a-as-built--the-model-card-emitter).

The goal is a single surface that presents the whole project — the PCA view keeps its
content under the name **Player fingerprints**, and eight pages join it covering the model
layer, the non-model inputs, the tournament layer, and the live draft board. **A ninth
joined them on 2026-08-10** — the weekly-scores page, from step 4 of
`docs/dashboard-revision-plan.md` — so the surface is ten pages; the nine-page language in
this section is the expansion's own scope and is left as written.

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

**Built 2026-08-10 and confirmed in a browser**, with one correction the design missed:
`st.navigation` renders *no navigation widget at all* for a single-page app, so the shell
shipped a placeholder beside the one real page until step 2 replaced it the same day. The
two-page floor is a standing constraint rather than a property of the placeholder, and a
test carries it. See [Step 1, as built](#step-1-as-built--the-multipage-shell).

### 2. The model pages are blocked on artifacts, not on UI

This is the large piece of work in the expansion, and it is pipeline work rather than
dashboard work.

The dashboard's binding rule is that it reads artifacts and never imports `src/`. Against
the model detail pages, here is what is actually on disk today:

| what a model page needs | what exists | gap |
|---|---|---|
| coefficient posteriors | ✅ `model_card_coefficients.csv`, all 20 heads (was: 2) | closed 2026-08-10 |
| diagnostics table | `stan_*_diagnostics.csv` for every head, plus the posteriors `manifest.csv` | none — reuse directly |
| PIT / calibration | `stan_availability_pit.csv`, `stan_games_played_pit.csv`; components carry a scalar `val_pit_ks` in `stan_component_metrics.csv` | **no ECDF band for any head** |
| predicted vs observed rows | `availability_predictions.csv`, `stan_availability_predictions.csv`, `season_total_predictions.csv`, `stan_games_played_predictions.csv` | **no component, minutes, composition or game-length rows; and every one of these is validation-only** |
| the features fed to each head | ✅ `model_card_features.csv`, `model_card_feature_corr.csv` | closed 2026-08-10 |
| the unit each head is fitted at | ✅ `model_card_index.csv` | closed 2026-08-10 |

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
long-format, dashboard-shaped artifacts. The dashboard reads those and only those.
**All seven artifacts shipped 2026-08-10**, and step 4 added an eighth; the contract now lives in
[docs/model-cards-plan.md](model-cards-plan.md); the sketch in
[The model-card artifact contract](#the-model-card-artifact-contract) below is kept for what
the shipped row counts say about the guesses.

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

**Closed 2026-08-10, when the expansion landed.** Page 7 took the ADP and calibrated-input
families, page 8 took the strategy and bracket families, and page 1 took the last three that
no page had claimed — `variance_budget.csv`, `game_length_coverage.csv` and
`sim_season_gate_a.csv`, each as one hero figure. The list above is kept as the record of
what the expansion was chosen against, not as a to-do.

---

## The pages

**Ten as of 2026-08-10**, in sidebar order — nine when the expansion landed, plus the
weekly-scores page step 4 of `docs/dashboard-revision-plan.md` inserted at position 8.
"Class" pages carry a head selector; the others do not.

| # | page | source | new artifacts? |
|---|---|---|---|
| 1 | **Overview** | ✅ `views/overview.py`, figures read from existing metrics CSVs, shipped 2026-08-10 (hero tiles; rewritten as four sections of prose the same day) | no |
| 2 | **Player fingerprints** | ✅ `views/fingerprints.py`, moved unchanged 2026-08-10 | no |
| 3 | **Availability** | ✅ `views/availability.py` over the generic renderer, shipped 2026-08-10 | one — the joint density |
| 4 | **Minutes** | ✅ `views/minutes.py` over the generic renderer plus `minutes_unification.csv`, shipped 2026-08-10 | no — step 3 had already written them |
| 5 | **Box-score components** | ✅ `views/components.py` over the generic renderer, shipped 2026-08-10 | no |
| 6 | **Game length** | ✅ `views/game_length.py` over the generic renderer, shipped 2026-08-10 | no |
| 7 | **Inputs beyond the heads** | ADP, injury capture, copula, serial correlation, bonus overdispersion, the availability layout | small |
| 8 | **Weekly scores** | ✅ `views/weekly.py` over `make weekly-scores`, shipped 2026-08-10 — Gate A at the scoring period | **six**, from a new emitter |
| 9 | **Tournament & strategy** | ✅ `views/tournament.py`, shipped 2026-08-10 | no |
| 10 | **Draft board** | today's `draft_room.py`, as a page | no |

**Inserting page 8 moved the last two rows and moved no `url_path`.** That is the whole
reason `app.VIEWS` declares each path rather than deriving it from the title or the
position: a link into `/tournament` written before the insertion still resolves after it.

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
   bounded subsample overlaid for texture. **Superseded 2026-08-10 by step 3 of
   `docs/dashboard-revision-plan.md`**: the raw residual-vs-fitted half is replaced by a
   scaled quantile residual (a QQ-uniform and the residual against rank-transformed
   predicted), because a raw residual is not on one scale across the four classes this one
   renderer serves and a quantile residual is uniform iff calibrated whatever the
   likelihood. `model_card_calibration.csv` carries one panel from that date.
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

**✅ Shipped 2026-08-10.** The design below stands as written; what it cost, the one emitter
it needed, and the four things it did not anticipate are in
[Step 6, as built](#step-6-as-built--inputs-beyond-the-heads).

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

**✅ Shipped 2026-08-10.** The design below stands as written; what it cost, what it had to
derive that the design did not anticipate, and what the three verification layers caught
are in [Step 2, as built](#step-2-as-built--tournament--strategy).

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

**✅ Shipped 2026-08-10.** The design below stands, and the feasibility argument came back
as a measurement rather than an expectation — see
[Step 7, as built](#step-7-as-built--the-draft-board-as-page-9).

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

**Superseded by [docs/model-cards-plan.md](model-cards-plan.md), 2026-08-10 — all seven
artifacts ship.** The sketch below is kept because the shipped row counts are worth reading
against the guesses, and because two of the three "easy to get wrong" notes under it became
build-time gates rather than advice. All files long-format, keyed by `head`, under
`outputs/predictions/`. Sizes are the reason for every binning decision — a 984,000-row
composition scatter is not an artifact, it is a copy of the data.

| artifact | grain | sketched rows | **as shipped** |
|---|---|---|---|
| `model_card_index.csv` | head | 20 — unit, family, variant, n_fit, label, class | ✅ **20** × 31 columns |
| `model_card_coefficients.csv` | head × term | ~600 — mean, sd, q2.5/25/75/97.5, `term_family` | ✅ **311** (268 coefficients, 20 intercepts, 23 dispersion) |
| `model_card_features.csv` | head × feature × split × bin | ~30,000 — binned counts plus per-feature n/mean/sd/missing | ✅ **14,892** over 92 distinct features |
| `model_card_feature_corr.csv` | head × feature × feature | ~5,000 | ✅ **8,920** — the sketch omitted the split, and both are emitted |
| *(unsketched)* `model_card_feature_density.parquet` | head × pair × split × 2-D bin | — | ✅ **93,608** — the sketch flagged the pairs and never binned one; added by step 4 |
| `model_card_ecdf.csv` | head × split × grid point | ~8,000 — observed ECDF and predictive quantile band | ✅ **2,977** — the grid follows the observed quantiles, so it is denser where the curve moves and shorter overall |
| `model_card_calibration.csv` | head × split × 2-D bin | ~4,000 — fitted vs observed density, and residual density | ✅ **28,709** — the sketch omitted the panel; two panels on a 30 × 30 grid with empty cells dropped. **11,689 since 2026-08-10**, when the residual panel moved to the artifact below |
| *(unsketched)* `model_card_quantile.csv` | head × split × panel × row | — | ✅ **8,768** — the scaled quantile residual that replaced the raw one; added by step 3 of `docs/dashboard-revision-plan.md` |
| `model_card_sample.parquet` | head × split × row | ~200,000 capped — bounded subsample carrying fitted, observed, residual | ✅ **54,375** — 2,000 per head and split, which is where a scatter stops being a scatter. Carries `u` and `predicted_rank` in place of `residual` since 2026-08-10 |

Three things the emitter must do that are easy to get wrong. All three held, and the first
two are now gates rather than intentions:

- **Cap the draws and the rows before generating a posterior predictive.** The composition
  head at full row count times a thousand draws is not affordable; 200 draws over a
  subsample is, and the ECDF band is stable well before that. **Shipped at 200 draws over at
  most 20,000 rows a split, and the stability is measured rather than assumed** — the ribbon
  is re-read on two interleaved halves of the draws and the build fails if they disagree by
  more than 0.02 in ECDF units. Worst gated head reads 0.0145.
- **Verify the recipe the way `posteriors.py` already does.** That module reproduces each
  head's own design matrix and predictions exactly at build time and fails the build rather
  than writing a wrong artifact. The model cards are downstream of the same recipe and get
  the same check, or they will drift silently. **The predictive half needed a fifth check on
  top**, because all four pass on a design matrix that is then drawn from on the wrong scale:
  the drawn mean must reproduce the head's own reported mean, within 5%.
- **Declare the unit per head in `model_card_index.csv`** and let the page read it, rather
  than hard-coding a unit string in the dashboard where it can go stale. **`response_label`
  is the same argument for the axis** — "games played", "minutes in one team-game",
  "absence-spell length (games)" — since the twenty heads share no observable either.

Total footprint **7.4 MB** across the seven artifacts, and **8.6 MB** across the eight that ship once step 4 added the density.

---

## The build order

Eight steps. Each is a self-contained session with its own deliverable and its own
verification; the ordering is a dependency ordering, not a preference.

**Step 1 · The multipage shell. ✅ Shipped 2026-08-10.** Convert `app.py` into an
`st.navigation` entrypoint; move the PCA view verbatim into
`dashboard/views/fingerprints.py` behind a `render()`; lift the appearance toggle into
shared state so it survives navigation; keep `make dashboard` pointing at the same
entrypoint. Ships with one real page, so the shell is proved before anything depends on
it — *and one placeholder, because Streamlit will not draw a navigation for a single page,*
which step 2 then replaced. See [Step 1, as built](#step-1-as-built--the-multipage-shell).

**Step 2 · Tournament & strategy (page 8). ✅ Shipped 2026-08-10.** Deliberately second: it
is the richest page, needs zero new pipeline work, and it exercises the multipage shell
with a genuinely different layout before the expensive step lands. All three of those held.
See [Step 2, as built](#step-2-as-built--tournament--strategy).

**Step 3 · The model-card emitter. ✅ Shipped 2026-08-10.**
`src/models/model_cards.py`, `make model-cards`, `docs/model-cards-plan.md`, and 65 tests.
No dashboard work at all. The largest step and the one most worth handing a fresh session
with the whole context budget — which is why the build order cut it in two (that order
lived in `docs/dashboard-build-prompts.md`, deleted 2026-08-10 when the expansion landed;
commit history holds it). **Step 3a landed the index, the coefficients, the
features and the feature correlations; step 3b the predictive half** — the ECDF ribbon, the
binned calibration density and the bounded sample — plus the fifth build-time check that
goes with drawing anything. Seven artifacts, 7.4 MB, ten seconds, no CmdStan. See
[Step 3a, as built](#step-3a-as-built--the-model-card-emitter).

**Step 4 · The generic model renderer plus the Availability page (3). ✅ Shipped
2026-08-10.** The seven blocks once, in `dashboard/views/model_page.py` over
`dashboard/model_cards.py`, with `views/availability.py` as four lines that name the class.
It needed one artifact the plan had assumed was already there — the joint density block 3
draws — see [Step 4, as built](#step-4-as-built--the-model-renderer-and-availability).

**Step 5 · Minutes, Box-score components, Game length (pages 4–6). ✅ Shipped 2026-08-10.**
Mostly configuration against the step-4 renderer, plus the `minutes_unification` two-unit
comparison, which is bespoke. The "mostly configuration" premise held for all three — each
is a view module that names a class plus its own named blocks — and the bespoke half needed
**three** named blocks rather than one, four new figures and no new artifact. See
[Step 5a, as built](#step-5a-as-built--box-score-components-and-game-length) and
[Step 5b, as built](#step-5b-as-built--the-minutes-page).

**Step 6 · Inputs beyond the heads (page 7). ✅ Shipped 2026-08-10.** Including the small
capture-calendar emitter, which is the step's only pipeline work: `make capture-calendar`
writes the two CSVs behind block 1, since `make capture-status` and `make adp-status` print
rather than write. See [Step 6, as built](#step-6-as-built--inputs-beyond-the-heads).

**Step 7 · The draft board as a page (9). ✅ Shipped 2026-08-10.** Both conditions held, and
the one that was written as a risk — that navigation might cost the room its
responsiveness — measured the other way: the page is 0.46 s *faster* to first paint than a
cold `make draft-room`, and 0.31 s on a return. See
[Step 7, as built](#step-7-as-built--the-draft-board-as-page-9).

**Step 8 · Overview (page 1). ✅ Shipped 2026-08-10.** Last, on purpose: it links into the
pages, so it could not be written until they existed, and writing it first would have made
it a table of contents for pages that did not. The three bounds all held, and the first one
had to be *measured* rather than intended — the page came in at 1,144 px on a 900 px
viewport before it was cut to 702. See
[Step 8, as built](#step-8-as-built--the-overview), and the
[amendment to bound 1](#the-amendment-to-bound-1--2026-08-10-and-it-is-a-ceiling-rather-than-a-waiver)
taken when the page was rewritten as a paper later the same day.

## What each step owes on the way out

Not negotiable, since these are the mechanisms that keep the surface from drifting into
being documentation again:

- **The three verification layers** from `dashboard/README.md`: `AppTest` in both appearance
  modes proves the page runs, a figure rendered to PNG proves the figure is legible, a real
  browser proves the page is. They apply to a step that ships a *page*; the two pipeline-only
  steps (3a, 3b) owe a **build gate** instead, since they have no page, no figure and no
  browser — see [Step 3a, as built](#step-3a-as-built--the-model-card-emitter).
  All three are runnable as of 2026-08-10 — `kaleido` and
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

## Step 1, as built — the multipage shell

**2026-08-10.** `make dashboard` and `make draft-room` are unchanged, `SRC_IMPORTERS` still
names exactly one file, and `dashboard/pca.py` still imports no Streamlit. The PCA view
moved verbatim: `AppTest` reports the same 2 charts, 6 tiles, 3 tables and 4 selectors it
did before the move.

### What the shell owns

`app.py` is now the entrypoint and holds three things: `st.set_page_config`, the
`VIEWS` tuple that is the sidebar, and `pages()`, which turns each row into an `st.Page`
with the first as default. A page is a `render()` in `dashboard/views/`; a row carries its
title, icon and a pinned `url_path`, so a deep link outlives a retitling.

The appearance mode moved into `dashboard/shell.py`, rendered by the entrypoint and read by
a view through `shell.current_theme()`. **Superseded 2026-08-10** — the radio is retired and
the appearance is Streamlit's own setting; `shell.current_theme()` is unchanged and the
navigation argument below still stands. See `docs/dashboard-revision-plan.md`, "Step 1, as
built". The rest of this subsection is the record of what shipped in step 1 of the
expansion. **This is forced rather than tidy.** The
entrypoint's body runs on every rerun while a `render()` runs only when its page is
selected, and Streamlit clears `st.session_state` for widgets the current page did not
render — so a mode switch declared inside a view is destroyed the moment the reader
navigates away. Driven under `AppTest`, a round trip to the second page and back resets the
fingerprint view's own `component` key from `pc8` to `pc1` while `appearance` holds. Same
session, same navigation, opposite outcomes; that contrast is both the demonstration and
the reason. Registered as `appearance-lives-in-the-shell`.

Sidebar order follows ownership: the navigation, then the shell's controls, then whatever
the page writes for itself — measured in the DOM rather than assumed, since all three land
in one column.

### The one thing the design got wrong

The step was specified to "ship with one real page, so the shell is proved before anything
depends on it". **That is not possible: Streamlit draws no navigation widget at all for a
single-page app.** The Python side still sends `Position.SIDEBAR`; the frontend renders
nothing, and `[data-testid="stSidebarNav"]` is simply absent from the DOM. A shell shipped
alone would therefore have been byte-for-byte indistinguishable to a reader from the
single-page script it replaced, and neither the navigation nor the cross-page state could
have been verified in a browser at all.

`AppTest` could not have caught this — it has no DOM — which makes it a clean example of
the three-layer rule earning its keep rather than a formality.

So the shell shipped **one placeholder beside the one real page**, and it was the *next*
page in the build order (Tournament & strategy, step 2) rather than a lorem-ipsum tab, so
step 2 replaced its row in `app.VIEWS` instead of adding to it. Two rules kept it from
lying to a machine that is checking, and the next placeholder will need them again: it
showed **no numbers**, and it named **`make` targets rather than artifact filenames** —
`audit.py`'s orphaned-artifact check counts an artifact as read when any string literal in
`dashboard/` names it, so a placeholder listing `strategy_sweep.csv` would report a file as
drawn that nothing draws, and the orphan count is how this doc picks what to build next.
Checked on the way in: the orphan count is unchanged at **1** and nothing is newly masked.
Registered as `one-page-renders-no-navigation`.

**`dashboard/views/placeholder.py` was deleted on 2026-08-10 when step 2 landed, and the
constraint it stood for did not go with it.** `test_the_navigation_carries_at_least_two_
entries` still holds the floor, and `test_every_navigation_row_is_a_real_view_module` now
holds the other half — `pages()` hands `st.Page` a bare callable, so a row whose `render`
came from a closure or a stub would navigate perfectly well and put something on a URL that
no module owns.

### Verification, as run

All three layers from `dashboard/README.md`, and the third is again the one that earned its
keep.

**`AppTest`**, both appearance modes, both pages: 2 charts, 6 tiles, 3 tables, 4 selectors,
0 exceptions, 0 missing-artifact warnings on the fingerprint page, and 0 charts on the
other — which is the assertion that the unselected page's script did *not* run. Plus the
page-switch round trip described above.

**Six figures rendered to PNG** through kaleido and looked at, built from the view's new
home: the radar in both modes, with and without a neighbour overlay, and a loadings panel.
The balanced-loadings rule still holds (PC1 draws five positive and five negative bars),
the selected spoke is still ringed, and pinned components still draw as open markers.

**The live page driven in Chrome** through Playwright. It confirmed the navigation renders
with both entries in declared order, that sidebar ordering is nav → Appearance → page
controls, that switching mode repaints the plots to the pinned surfaces (`#fcfcfb` →
`#1a1a19`, read off `.main-svg`'s inline style — plotly does not put `paper_bgcolor` on
`rect.bg`, which sits at `fill-opacity: 0`), that the appearance survives a navigation to
the placeholder and back, that `/tournament` deep-links, and that clicking spoke 8 still
drives the component panel (`PC1 · Paint big, not shooter` → `PC8 · Mid-range big who
steals`). No literal `"undefined"` anywhere.

---

## Step 2, as built — Tournament & strategy

**2026-08-10.** `dashboard/views/tournament.py` behind a `render()`, its pure layer in
`dashboard/strategy.py`, and five figures in `charts.py`. It replaced the placeholder's row
in `app.VIEWS` rather than adding to it, and `dashboard/views/placeholder.py` was deleted
with it — the *constraint* the placeholder existed for outlives it and is now carried by
`test_the_navigation_carries_at_least_two_entries` plus a new
`test_every_navigation_row_is_a_real_view_module`.

**The step's premise held exactly.** No new artifact, no pipeline run, no `src/` import,
and the orphan count is unchanged at **1** — the strategy and bracket families were already
accounted for by registry entries, so drawing them moved nothing. The page reads five
files: `bracket_structure.csv` from `make bracket`, and `strategy_{sweep,paired,shipped,
realized}.csv` from `make strategy-sweep`, plus `economics.py`'s arithmetic over the two
captured DraftKings CSVs.

### The one thing the design did not specify, and it is a units problem

The plan says to draw "lift versus the null with confidence intervals, faceted by axis,
**with the break-even hurdle as a reference line**". Those two halves are in different
units. `select-on-p-advance-report-roi` put the sweep's headline in *survival* — lift in
`P(top 2 of 12)` — precisely because ROI does not resolve at any affordable budget, while
the hurdle is a *return*: +17.60% and +12.32%. Drawing +17.60% on a lift axis is a units
error; drawing nothing leaves the page's central chart with no answer to "is this edge
worth entering on".

So the hurdle is converted, and the conversion is stated on the page rather than buried in
the line. An exchangeable entry advances at `p_null` and is worth `1 − rake` of its fee; if
expected payout scaled with `P(advance)`, returning the whole fee needs
`p_null/(1 − rake) = p_null·(1 + hurdle)`, so the lift is **`p_null · hurdle`** — **+0.0293**
at `600k_shootaround` and **+0.0205** at `20k_spin_move`.

**The assumption is conservative, and that is measured rather than asserted.** The
elasticity of the sweep's own ROI with respect to its own survival —
`log(payout ratio) / log(survival ratio)`, both against the null the same row carries — has
a median of **5.40** at `600k_shootaround` and **2.01** at `20k_spin_move`, and exceeds 1 on
**all 88** swept rows. Payout compounds through four cuts into a 10,000× top prize, so a
strategy that survives twice as often is worth far more than twice as much, and the drawn
line therefore sits *above* the lift a real break-even needs. The page prints the elasticity
beside the line. Registered as `hurdle-is-drawn-in-survival-units`.

### What the four blocks became

1. **Contest structure** — six tiles for the selected tier, then the survival curve
   (`P(reach round r)` for an exchangeable entry, log axis, five tournaments) beside the
   break-even hurdle bars, and a table twin carrying the round ladder and all five
   contests. Round 1 carries its own marker line, because five of every six entries are
   gone in a round that pays nothing.
2. **The sweep** — 22 arms on 7 facets, two seasons as two series, sharing an x axis with
   the two reference lines. Arms are ordered by their mean lift *inside* their own facet
   and *per tier*, not on a fixed order: the two tiers disagree about which `α` wins, and a
   shared order would hide it. A test pins that they still disagree.
3. **Simulated versus realized** — both surfaces on one `P(top 2 of 12)` axis so their
   **widths** can be compared, which is the whole point. The realized intervals are
   **2.9×** wider at `600k_shootaround`, and that ratio is a tile.
4. **The paired comparisons** — sorted gaps against a chosen baseline, with the unresolved
   ones in their own slot. At the default view (`600k_shootaround`, `p_advance`, against
   `model_mean`) that is **8 of 21**. Registered as
   `unresolved-gaps-are-styled-not-buried`.

Three decisions inside those worth recording:

- **Five tournaments is two over `ALL_PAIRS_CAP`, so both block-1 figures use
  highlight-and-gray** with the same encoding — the two swept tiers take slots 0 and 1 in
  `TARGET_TIERS` order, the other three take `muted`. Using one encoding twice means the
  reader learns it once, and the slot comes from the declared order rather than from row
  order so the two charts cannot disagree. It also surfaced a real coincidence: `20k Spin
  Move` and `88k Alley Oop` have *identical* advance chains, so their curves overlap
  exactly and one is drawn on top of the other. That is captioned and the table twin lists
  both, rather than being hidden by drawing order.
- **`crosses_zero` is derived from the interval the chart draws, not read from the
  artifact's `resolved` column.** The styling has to follow the bar the reader is looking
  at; a flag that drifted from its own interval would put a filled marker on a gap visibly
  straddling zero. A test asserts the two agree across every tournament × metric ×
  baseline on the shipped artifact, so a future run that stops agreeing is caught.
- **The tile CSS moved from the fingerprint view into `shell.TILE_CSS`, as an opt-in.**
  Two pages now want the type scale that fits more than four metrics across a row. It is a
  function a page *calls* rather than something the entrypoint applies, which keeps the
  original reason intact: a page that has not been laid out yet should not inherit a scale
  chosen for somebody else's header.

### Verification, as run

**`AppTest`**, both appearance modes: **5 charts, 13 tiles, 5 tables, 3 selectors, 4
subheaders, 4 expanders, 0 exceptions, 0 missing-artifact warnings**, identical in light
and dark. Plus the tier switch, both block-4 selectors, and a fingerprints → tournament →
fingerprints round trip in which `appearance` holds and the fingerprint page still reports
its own 2 charts, 6 tiles, 3 tables, 4 selectors — the check that moving the tile CSS
changed nothing there.

**Sixteen figures rendered to PNG** through kaleido and looked at: all five figures in both
modes, and the sweep, surfaces and paired figures for both tiers. This layer caught three
things a test would not have:

- **Plotly's y zeroline drew a rule through the top row of every panel.** These charts put
  rows on a *numeric* y axis so two series can be nudged off a shared row, which makes row
  0 an arm rather than an origin. Pinned by `test_a_row_position_axis_carries_no_zero_line`.
- **Two reference-line labels collided** on the sweep, since the null and the break-even
  line are 0.029 apart. Shortened to `null` and `break-even`, with the derivation in the
  caption.
- **The `20k_spin_move` null sits mid-axis**, where the surfaces chart's top-anchored label
  landed under the legend. Reference labels can now be anchored at the plot floor instead,
  and that tier's is.

A fourth was found by looking at the rendered page rather than the figure: **row heights
were not comparable across facets.** Making a subplot taller without making its *range*
taller just spreads its rows apart, so a one-arm facet was twice as airy as a five-arm one
and each facet title sat on the last row of the facet above. The header room now comes out
of the range.

**The live page driven in Chrome** through Playwright — **25 checks, all passing**. The
navigation renders with both entries and no placeholder; `/tournament` deep-links; five
plots draw; all four block headings are present; there is no `stException` and no
missing-artifact alert; no literal `"undefined"` anywhere, before or after an interaction;
sidebar order is nav → Appearance → page controls (checked by *geometry*, since the nav's
own link is titled "Tournament & strategy" and a text search for "Tournament" finds the
navigation instead); the mode switch repaints to the pinned surfaces; switching tier
reaches both the tiles (`$20` → `$52`) and the reference line (`+0.0293` → `+0.0205`); and
after an in-app navigation to the fingerprint page the dark surface holds and all six
metric tiles render at 22.4px with **no** element clipping its own value.

One thing that layer taught about the *test*, not the page: an appearance check must click
the nav link rather than `goto` the URL. A hard reload is a new session and resets the mode
to `detected_mode()` legitimately — what `shell.py` claims to survive is an in-app
navigation.

**25 new tests** in `tests/test_dashboard.py`, all of the pure layer and the figures as
plain functions; 143 in that file and **1,168** across the suite.

---

## Step 3a, as built — the model-card emitter

**2026-08-10.** `src/models/model_cards.py` and `make model-cards`, writing four of the
seven artifacts in about 5 seconds with no CmdStan and no refit. The contract is
[docs/model-cards-plan.md](model-cards-plan.md), which is what step 4 reads; this
section records only what the *design above* got wrong or left out. The other three
artifacts followed the same day — see
[Step 3b, as built](#step-3b-as-built--the-predictive-half).

**No dashboard work, as specified, and the orphan count is unchanged at 1.** The four new
artifacts are accounted for by their registry entries' `reproduce` links rather than by a
string literal in `dashboard/`, which is the honest state: the pipeline has written them and
no page reads them yet. That is check 3 of `make dashboard-audit` doing exactly its job —
a deliberately deferred family is *recorded* rather than rendered.

### The verification is bigger than the design asked for, and that is the finding

The step was specified as "verify the recipe the way `posteriors.py` already does". That is
the right rule at the wrong size. `posteriors.py` checks a 400-row probe because its own
drift surface is the recipe; **the emitter re-derives the frames**, so a `build_design` that
changed shape, a split that moved or a filter that drifted would leave the coefficients
describing one population and the histograms describing another — and both files would look
perfectly well-formed. So `verify` runs four checks and raises on any:

1. the rebuilt fitting frame matches `provenance.n_fit_rows` and the recorded season span;
2. the persisted recipe on the raw frame equals the head's **own variant ladder** through the
   head's own scaler, both splits, to 1e-9;
3. every column `recipe.features` names exists on the rebuilt frame;
4. the artifact's own `roundtrip()`.

**Check 2 is tautological for the nine heads whose recipe carries no design steps** — their
raw frame *is* their design frame — which is why 4 is run rather than assumed redundant, and
why `model_card_index.csv` carries `design_check` per head (`ladder` or `vacuous`). A green
tick that cannot fail is worse than no tick. As shipped, all twenty heads pass check 2 at
exactly 0.0.

### Three things the contract sketch did not say

- **`n_fit` and `n_frame_rows` differ on four heads, and neither is wrong.**
  `StanConversion.fit` drops rows with no attempts *internally*, so `fg3m_given_fg3a` fits
  7,695 of the 8,630 rows `posteriors/train/manifest.csv` reports. Both ship, with a
  `row_filter` column naming the filter, because a reader comparing the two files would
  otherwise find an unexplained discrepancy.
- **The correlation artifact carries a `split` column**, which the sketched grain omitted.
  It costs 4,460 rows and it earned them immediately: three features are **identically
  constant on the validation split** — two leftmost spline bases with no validation row in
  their knot span, and `ftm_pct_lag1__miss`, since nothing was imputed there — which is
  visible as an empty row in the heatmap and invisible in any training-frame-only view.
- **The feature histograms are the head's design columns, and `missing_share` has to walk
  back to find the column it came from.** `logit_fg3m_pct_lag1__s3` is a spline over a logit
  over `fg3m_pct_lag1`, and only the last of those three names is what the head flagged.
  Reading the flag off the feature's own name would report a flat zero for every spline basis
  in the project, which renders as a perfectly good-looking page.

### Verification, as run

**The three layers from `dashboard/README.md` do not apply** — there is no page, no figure
and no browser in this step, and running `AppTest` would have proved only that the pages
built in earlier sessions still work. What stands in their place is the build gate above,
which is stronger in the one direction that matters here: it is the only step so far whose
`make` target *fails* rather than reports.

**34 tests** in `tests/test_model_cards.py`, plain `assert` with synthetic builders, on real
`PosteriorArtifact`s with injected draws. The coverage rule is **one case per way this module
can be wrong silently**, because every one of those renders as a good-looking picture: a
histogram on the wrong edges, a missing-share resolving to zero because the flag sits under a
third name, a correlation reporting 0 for a constant column, a split label outside the
vocabulary, a spline basis drawn as six unrelated bars. The two checks that fail *loudly* —
the population anchor and the design tolerance — get one test each for the raise. Five read
the shipped artifacts to keep a derived quantity honest against the artifact it came from,
and skip rather than fail without `make posteriors`.

## Step 3b, as built — the predictive half

**2026-08-10.** The remaining three artifacts, from one 200-draw predictive per head per
split. Seven artifacts now, 7.4 MB, ten seconds, still no CmdStan. Again no dashboard work,
and the orphan count is again unchanged. Three things worth carrying into step 4.

**The draw budget is a measurement, not a setting.** The prompt asked for ~200 draws *and*
for the band to be checked at that budget rather than assumed stable there, which turned out
to be the more useful half of the instruction. `band_stability` re-reads the 95% ribbon on two
interleaved halves of the draws; the statistic falls as `1/sqrt(D)` across 100 / 200 / 400,
which is the confirmation that it is measuring Monte Carlo error and not misfit, and at the
shipped 200 the worst gated head sits at **0.0145** — roughly 0.007 on the shipped ribbon,
under the resolution these panels draw at. 400 draws would buy a third of a pixel for double
the cost. One head is reported rather than gated: `game_length_ot` has two validation cells,
so its ECDF takes three values and a half-sample gap of 0.5 is arithmetic.

**A fifth check was needed, and the four existing ones could not have caught what it
catches.** All four of `verify`'s checks pass on a design matrix that is then drawn from on
the wrong scale — a missing exposure, a trials column that moved, a link applied twice — so
`predictive_bias` compares the drawn mean against the head's own reported mean. Worst is
+1.20%. Three heads cannot take that check as stated and say so in the index rather than
appearing to pass it: the composition reports `eta`, a step's linear predictor, and the two
beta-geometrics report `mu`, which is `P(T = 1)` and not a mean — so those two are checked on
`P(T = 1)` instead, which is a sharper reading of the same parameter.

**Block 5 needs to be drawn as a distance, not as a verdict.** At n ≈ 10⁴ the
posterior-predictive ribbon is ±1–2 ECDF points and every head in the project falls outside
it somewhere — the observed curve is inside the 95% band at 22% of grid points for
`availability` and 7% for the composition. That is what a PPC does at this sample size. The
reading the page should render is the vertical distance from `q50` (0.037 for availability,
0.055 for minutes, 0.051 for the composition), not in-or-out.

**31 more tests**, 65 in the file. The new ones follow the same rule and add the failure modes
this half introduces: a collapsed cell frame drawn one row per cell rather than per spell, a
band pooled across draws instead of read per draw, a calibration grid collapsed by one
outlier, a residual panel that is a second copy of the observed one. One of them builds a real
`StanCount` and asserts `draw_predictive` returns byte-identical output to the head's own
`predict_samples`, because "no second implementation of any head's predictive" is this half's
load-bearing rule and a docstring cannot hold it.

---

## Step 4, as built — the model renderer and Availability

**2026-08-10.** `dashboard/views/model_page.py` is the seven blocks; `dashboard/model_cards.py`
is their pure layer; `dashboard/views/availability.py` is four lines that name a class. Six
new figures in `charts.py`, 38 new tests, and one new artifact the plan had not costed.

**The step's premise held: pages 4–6 are configuration.** `model_cards.CLASSES` already
carries all four pages' title, icon, `url_path`, head order and specification intro, and
`app.model_view()` builds a navigation row from that table, so sessions 6 and 7 add a view
module of four lines and one row. What is *not* declared there is the unit — that is read per
head from `model_card_index.csv`, because the five availability heads are not all at one unit
(`gp_duration` is per absence spell) and a page that stated one unit at the top would be
wrong about one of its own heads. Registered as
`model-pages-are-one-renderer-and-a-class-table`.

### The one thing the plan assumed was already there

**Block 3 had no artifact.** `feature-correlation-not-pair-plots` was recorded at status
`open` with the unblocking condition written out — "the emitter binning a 2-D density for
each `top_pair`" — and step 3 shipped the flags without the densities. The dashboard cannot
compute a joint from marginals, so the block was undrawable, and this session added the
eighth artifact rather than substituting something the heatmap already says.
`model_card_feature_density.parquet` is 93,608 cells over 720 panels for 536 KB; the contract
is in [model-cards-plan.md](model-cards-plan.md#model_card_feature_densityparquet--the-joint-behind-the-heatmap).

Two decisions inside it. The pair menu is ranked on the **training** split and both panels are
drawn for it, so flipping the split changes the picture and not the menu. And it ships as
**parquet**, which is the second exception to this family being CSVs and for the opposite
reason to the first: the sample is three float columns where a CSV would widen every
`float32`, and the density is two long feature names restated on every cell where dictionary
encoding is the measured difference between 10.5 MB and 0.55 MB. As a CSV the block would
have cost more than the other seven artifacts together. Registered as
`model-card-density-is-parquet-not-csv`.

### What each block became

1. **What this head is** — five tiles led by the **unit**, the head's own `description`, and
   the specification as a table. Typed prose is confined to this block and to
   `ModelClass.intro`, and describes the specification only.
2. **The features it was fed** — one histogram per design column on the edge set both splits
   share, train as filled bars and validation as a step line, plus the n / mean / sd /
   imputed-share table. Two marks as well as two colours, so the comparison survives the
   relief rule. 19 columns for availability, and the widest head in the project (the
   composition, 25) comes out as seven rows of a tall figure rather than a crammed one.
3. **Feature relationships** — the correlation heatmap pinned to [−1, +1] around zero so two
   heads' heatmaps mean the same thing, beside one joint density for a pair the reader picks.
4. **Coefficients** — posterior means with 95% intervals, sorted, families kept together with
   their bases in order, and a collapse toggle for the nine heads that carry a six-column
   basis. The **intercept and dispersion are tiled rather than drawn**: they are not on the
   standardized slope scale the bars share, and the intercept would set the axis.
5. **Predictive calibration** — the ribbon, with the **largest distance from the median
   replicate tiled** and coverage demoted to a footnote, exactly as 3b's finding requires.
6. **Predicted against observed** — four panels of binned density with the bounded sample
   over them.
7. **Diagnostics** — two rows, `make posteriors` and `make stan`, each naming itself, with an
   em dash wherever a source carries nothing; then the four build-time checks with
   `design_check` beside them. Registered as `two-sampler-runs-are-two-rows`.

### Verification, as run

**`AppTest`**, both appearance modes, all five heads: **6 charts, 10 tiles, 8 tables, 4
selectors, 7 subheaders, 6 expanders, 0 exceptions, 0 missing-artifact warnings**, identical
in light and dark and identical across the five heads. Plus both block-3 controls, and a
fingerprints → availability → tournament round trip in which `appearance` holds and the other
two pages still report their own counts.

**Thirty figures rendered to PNG** through kaleido and looked at — every figure in both modes
for two heads, plus the composition's 25-feature grid and `ast` collapsed and expanded, which
is the spline path the availability class cannot exercise and pages 5 and 6 depend on. Three
things this layer caught that a test would not have:

- **The coefficient panel opened on its weakest term.** Sorting ascending put the strongest
  family at the bottom of a figure whose row 0 is the top.
- **One colourbar was labelling four panels that do not share a scale.** A 751-row validation
  panel puts an order of magnitude more share into each cell than an 8,232-row training one,
  so the four panels are now each scaled to their own densest cell under a colourbar that
  says so, with the raw share still in the hover.
- **The scatter overlay was undoing the emitter's tail clipping.** `gp_duration`'s density
  spans 2 to 9 games and one 62-game spell in the overlay stretched the axis until the
  density was a sliver. Both splits of a panel now share one range, taken from the grid.
  Registered as `calibration-panels-are-scaled-to-their-own-densest-cell`.

**The live page driven in Chrome** through Playwright — **27 checks, all passing**: the
navigation carries three entries in declared order, `/availability` deep-links, six plots
draw, all seven block headings are present, no `stException` and no missing-artifact alert,
no literal `"undefined"` before or after an interaction, sidebar order is nav → Appearance →
page controls by geometry, no metric tile clips its own value, the head selector reaches a
head at a different unit, the mode switch repaints to the pinned surfaces, the pair selector
redraws the density, and dark holds across an in-app navigation in both directions.

A fourth thing was found by reading the rendered page rather than a figure, and it is the
same class of problem step 2 hit: **three tables were truncating the column that carried
their content.** The specification table's `Note`, and both diagnostics tables, were cut off
mid-sentence. The specification table went full width, the two sentences explaining the
sampler rows moved into the caption, and the diagnostics columns were renamed short enough
that every one of them — including the git SHA — fits without a horizontal scroll.

**38 new tests** in `tests/test_dashboard.py`, 181 in that file and **1,280** across the
suite. Two of them are artifact-contract tests rather than unit tests, and they are the ones
that will catch a session-6 or session-7 mistake before a browser does: every carded head
belongs to exactly one declared page, and every head's diagnostics label resolves to a real
row in the real `stan_*_diagnostics.csv` — the failure mode there is a block that renders
*empty* rather than wrong.

---

## Step 5a, as built — Box-score components and Game length

**2026-08-10.** `dashboard/views/components.py` and `dashboard/views/game_length.py`, two
rows in `app.VIEWS`, two figures in `charts.py`, and 24 new tests. **No new artifact and no
pipeline run** — the orphan count is unchanged at **1**.

**Step 4's premise held all the way.** A model page really is a view module that names a
class: `model_cards.CLASSES` already carried both pages' title, icon, `url_path`, head order
and intro, so `app.model_view()` took each in one line and nothing else in `app.py` moved.
What each page adds is one callable.

### A page's own block is named, not numbered

The renderer's `extra` hook was built in step 4 for the minutes page and is used here first.
Both pages key theirs on **block 1**, and both are called *Against the no-fit floor*.

The rule the two of them settle is that **the seven blocks are numbered because they are
shared, and a page's own block is named**. Inserting page-specific material into the sequence
would mean block 5 was a different block on two of the four pages, which is the one property
the numbering buys. A named block also goes where its question is asked: "what did this head
buy over doing nothing" is the first thing to know about a component head, not the eighth.
Registered as `a-pages-own-block-is-named-not-numbered`.

### Page 5 · the floor is the point, and one head is below it

Every head in this project is quoted against a no-fit floor, so a fitted score with no
reference point on the page is unreadable. The block reads `stan_component_metrics.csv` —
the head's own variant ladder, in which the floor is a row — rather than the model cards,
which describe the arm that shipped and carry no record of what it beat.

Four tiles for the open head, then its own five-row ladder, then **every head's margin over
its own floor** as one chart with the open head highlighted. On the shipped Stan ladder ten
heads clear and **`ftm|fta` does not, at −0.0190 R²**. It is drawn, not omitted: an
empirical-Bayes shrink of a prior free-throw percentage is already close to optimal for a
quantity that is nearly pure player skill, which is a finding about the target rather than a
defect in the head. The zero line *is* the floor, so position carries the verdict; a
non-clearing bar is also outlined and every bar prints its own margin, because no value here
may be reachable by colour alone. The caption states that R² is each head's own on its own
response, so a bar reads as *how much the fit added* and never as one head beating another.

> These are the **Stan** ladder's margins and they are not the `+0.0013 to +0.0334` in
> `README.md`, which is `make component-rates`' sklearn probe over the count heads only. Two
> instruments, two artifacts; the page draws the one that describes the heads that shipped.

Registered as `every-component-head-is-drawn-against-its-floor`.

### Page 5 · the share head says what it is a share of

`fg3a | fga` models `fg3a / fga` — the three-point share of a player's shot diet — and its
own prior-season term is named **`logit_fg3a_pct_lag1`**, where `_pct_` is `fg3a / fga` and
*not* `fg3m / fg3a`. Block 4 puts that name at the top of the panel as the head's strongest
term, so the page was about to hand a reader a shooting-percentage label for an attempt mix.
That is precisely what `stan_components.conversion_variants` takes an explicit `own=`
parameter to prevent in the fitting code.

So `model_cards.COMPONENT_BASIS` declares each head's role — count, attempt share,
conversion — with the ratio written as arithmetic and **anchored** to the design column
carrying its own rate, in the sense `pca.orient()` uses the word: a test asserts every
declared `own_family` is a real `term_family` for that head in `model_card_features.csv`, so
a refit that renames the column fails instead of mislabelling. Every head prints a one-line
basis note in block 1; the share head prints the long form naming both columns and the head
the other one lives on. Registered as `the-share-head-is-anchored-to-its-own-column`.

### Page 6 · a two-point ECDF is not a calibration reading

The onset head is fitted on **26 season cells** and the depth head on **4 depth cells** — the
two smallest heads in the project — so each has a **two-point validation ECDF**. Block 5
renders it without complaint and it is an arithmetic shape: a filled triangle between two
points. The page says so, with the grid count read from the artifact rather than typed.

What stands in for it is the unit the heads are actually consumed at, which
`make stan-game-length` already writes: a predictive **count per game class** for every arm
of the ladder including its no-fit floor. `regulation` is `n_games` minus the other three by
construction for every arm alike, so it is carried in the table twin and left off the figure,
where it is a 2,300-long bar that flattens the three classes the arms differ on. The
observed is drawn as an **outlined** bar rather than a third filled series — it is the target
the two arms are measured against, and ink is what block 5 already uses for an observed
curve. For the depth head the block also tiles nats per overtime game for the geometric and
the beta-geometric, which are a wash on 138 validation games; the frailty earns its keep on
the absence-spell head that shares the Stan source, not here. Registered as
`game-length-is-read-at-the-games-unit`.

### Verification, as run

**`AppTest`**, both appearance modes, **all thirteen heads on the two new pages and all five
on Availability again**. Components: **7 charts, 14 tiles, 10 tables, 4–5 selectors, 8
subheaders, 6–7 expanders, 0 exceptions, 0 missing-artifact warnings**, identical in light
and dark and identical across the eleven heads except for the collapse toggle, which the four
heads with no spline basis correctly do not draw. Game length: **5 charts** for onset and
**3** for depth, the difference being the blocks that degrade rather than draw. Availability
is unchanged at its step-4 counts — 6 charts, 10 tiles, 8 tables, 4 selectors, 7 subheaders,
6 expanders — and a five-page round trip holds `appearance` while fingerprints and tournament
still report their own counts.

**Fourteen figures rendered to PNG** through kaleido and looked at. This layer earned its
keep four times over, and every fix is in the shared layer:

- **Plotly drew a ribbon's markers in its own default colorway.** `go.Scatter` infers
  `lines+markers` for a trace of 20 points or fewer *and* infers the marker colour from the
  default palette, so the onset head's two-point band came out with stray cyan and red dots
  on a surface whose whole point is a validated palette. `mode` is now explicit.
- **A one-column feature grid still laid out four columns**, drawing its single histogram in
  the leftmost quarter with three empty cells beside it, and a single-row grid put its
  subplot title under the shared legend.
- **A value printed outside its own bar rendered against the plot edge.** Both new figures
  now take an axis range with room for the label, which is not assertable from the trace:
  the text is laid out by plotly.js.
- **A solid ink bar for the observed read as the largest quantity on the chart** rather than
  as the reference the two arms are measured against.

**The live pages driven in Chrome** through Playwright — **52 checks, all passing**: five
navigation rows in declared order, `/components` deep-links, seven plots and all eight
headings on the components page, five and three on game length, no `stException`, no
missing-artifact warning, no literal `"undefined"` **and no literal `nan`** anywhere before
or after an interaction, no metric tile clipping its own value, sidebar order by geometry, the
mode switch repainting to the pinned surfaces and holding across an in-app navigation, the
head selector reaching the share head and the failing head and moving both the tiles and the
prose, and Availability and Tournament still drawing their own six and five plots.

A fifth and sixth defect came from **reading** the rendered page rather than a figure, which
is the same lesson steps 2 and 4 recorded:

- **Block 2 raised on every head that imputed anything.** `DataFrame.itertuples` renames a
  column whose name is not an identifier, so `Imputed share` arrives as `_10` and reading it
  back by name is a `KeyError`. Nine heads and the whole availability page render fine
  because nothing in them was imputed; `ftm|fta` and `fg3m|fg3a` are the first that were.
  The formatting moved into `model_cards.imputed_shares` so a test can hold it.
- **The depth head printed the literal `nan`.** It is fitted on depth cells and has no season
  span at all, and an f-string over an absent CSV cell prints four letters that look like a
  value — the same class of defect as the `undefined` a plotly title with no text renders as.
  `model_cards.text()` and `season_span()` now return an em dash, and the browser layer
  checks for `nan` alongside `undefined` from here on.

All six are recorded together as `the-smallest-heads-fixed-the-shared-renderer`, because what
they have in common is the argument for not forking a renderer: **the two smallest heads in
the project and the first heads to impute anything reached branches the shipped pages could
not, and every fix landed on the shipped pages too.**

**24 new tests**, 205 in `tests/test_dashboard.py` and **1,304** across the suite.

---

## Step 5b, as built — the Minutes page

**2026-08-10.** `dashboard/views/minutes.py`, one row in `app.VIEWS`, four figures in
`charts.py`, ~500 lines of pure layer in `model_cards.py`, and 20 new tests. **No new
artifact and no pipeline run**; the orphan count is unchanged at **1**.

Step 4's premise held for the fourth and last time — `model_cards.CLASSES` already carried
this page's title, icon, `url_path`, head order and intro — but this is the page the build
order called *bespoke*, and the bespoke half is bigger than the configuration half. The
seven numbered blocks are the same seven; what the page adds is three named blocks, drawn
from `minutes_unification.csv` and the composition's own `make stan-composition` ladder.

### A page's own block is named — and a page may have three

Pages 5 and 6 settled that a page's own block is named rather than numbered and is keyed on
the numbered block whose question it extends. Both keyed theirs on block 1. This page uses
**three keys**, and the reason is worth keeping: each of its three questions is asked at a
different point on the page, and burying all three under block 1 would have put the answer
before two of the questions.

- **Under block 1 · "Two units, two verdicts."** Which unit a head is a model at is the
  first thing to know about either minutes head, exactly as "what did this head buy over
  doing nothing" is on the box-score page.
- **Under block 5 · "The missing parameter."** What fails at the season unit is
  *calibration*, and the injected effect is what moves the PIT KS. It follows the block that
  draws the ribbon.
- **Under block 6 · "What no marginal panel can see."** Block 6 is four panels of marginal
  residuals, and the zero-sum team constraint is precisely the thing no marginal metric can
  see. The block follows them and says so in its title.

Registered as `three-named-blocks-go-where-their-questions-are`.

### The two units are made commensurable by the floor they already had

The page's reason to exist is that **one posterior gives opposite verdicts at two units**,
and the obstacle to drawing it is that the two CRPS are 4.4945 minutes per player-game and
170.06 per player-season. They cannot share an axis.

What makes them comparable is the reference each already carries: every head in this project
is quoted against a **no-fit floor**, so the axis is the *ratio* to the floor of that unit
and the zero line is the floor — the encoding page 5 already uses. On the shipped artifacts
the composition reads **+3.9%** against its per-game floor and **−5.4%** against the season
one, and the marginal head reads **−2.3%** and **+10.5%**. Two panels, four bars, and the
reversal is the picture rather than a sentence under a table.

The second thing that made this drawable is that **the two heads meet at both units in
artifacts where nothing was fitted twice**: at the season unit `make minutes-unification`
rehydrates both around their persisted posteriors, and at the per-game unit the marginal
head is already there as `independent_comparator`, the control the composition's own ladder
refits. `model_cards.unit_board` owns that arm→head map so the view never has to know it.
Registered as `two-units-share-an-axis-only-through-their-floors`.

### The means are the control and the spread is the finding

Drawing only MAE would say the two heads are the same model; drawing only the predictive sd
would leave a reader wondering whether the composition is simply worse. So the block draws
**four readings in a 2 × 2**, split by what each is a statement about — `MAE` and `bias` for
where the predictive sits, `predictive sd` and `PIT KS` for how wide it is. The top row is a
tie (200.28 against 200.12, and the composition is the *less biased* at +2.41 against
−14.09); the bottom row separates them **4.68×** (64.65 against 302.75).

The sd panel is the only one with a target value, so it is the only one carrying a reference
line: the head's own **residual sd, 243.50**. Without it a reader cannot tell whether 64.65
is too narrow or 302.75 too wide, which is the entire diagnosis. Every other panel is drawn
bare rather than given a decorative zero.

### Every figure on this page is a comparison, so nothing is highlight-and-gray

Pages 5 and 6 use highlight-and-gray, where a slot marks the head the reader has open among
eleven. Three of this page's four figures have exactly **two** series and both are the
point, so a colour that followed the selector would mean two different things on one screen.
Each head therefore keeps one fixed slot for the whole page (`model_cards.MINUTES_SLOTS`,
two slots — well inside `ALL_PAIRS_CAP`), and what follows the selector is the **tiles**,
which read the open head's own side of each comparison in the open head's own direction.
That last part is where the sign errors live: `season_gap` flips the interval's *ends* along
with the gap, because negating both without swapping them is wrong in exactly one of the two
branches and looks fine in the other. Registered as `a-comparison-page-fixes-its-colours`.

### The σ block, and the one number the page refuses to type

The injected per-player-season effect is drawn twice, because it answers two questions. The
**gaps against the marginal head** go through `fig_paired` — the tournament page's own
builder, unchanged — because "an interval straddling zero is a tie" is the same idea that
page already carries three redundant ways, and a second encoding for one idea is worse than
either. And the **two σ grids** get a panel each, never a shared y axis: they score disjoint
rows (742 validation player-seasons against 1,145 training ones), so their CRPS *levels* are
not comparable and only the location of each optimum is. Both optima are interior and one
grid step apart, which is the whole reason the shipped σ owes the evaluation rows nothing.

The shipped σ itself is read from `player_season_sigma` on the composition's own card, not
typed — and a test holds the load-bearing property that it **is the train grid's optimum**,
so a refit that moved the grid without moving the persisted value fails rather than leaving
the page claiming a σ nothing selected.

### Verification, as run

**`AppTest`**, both appearance modes, both heads: **11 charts, 20–23 tiles, 11 tables, 2
selectors + 2 radios, 10 subheaders (seven numbered, three named), 9 expanders, 0
exceptions, 0 missing-artifact warnings**, identical in light and dark. The two tile counts
differ by exactly the three extra scalar terms the composition carries — its dispersion is
role-graded, so block 4 tiles `rho[1]…rho[4]` rather than one `rho` — and the collapse
toggle appears only on the marginal head, which is the one with a spline basis. The four
already-shipped pages still report their step-4 and step-5a counts, and `appearance` holds
across a five-page round trip.

**Ten figures rendered to PNG** through kaleido, in both modes, and looked at. Two defects,
and **the more serious one was in the shared builder**:

- **`fig_paired`'s baseline label collided with the legend.** Both live in the strip above
  the plot, so whether they overlap depends on *where zero falls on the x axis* — nothing in
  the trace can see it, and neither placement `_reference_line` offers is safe in general
  (at the top it hits the legend, at the floor it hits the bottom row's interval). The fix is
  that the label was redundant: the axis title already reads "gap … against `<baseline>`", so
  the line is now drawn bare. `_reference_line` takes an empty label for it, the tournament
  page was re-rendered to confirm the only change there is one fewer duplicate, and a test
  pins it.
- **A reference label's own opaque chip cut the curve it labelled in half.** The marginal
  head's CRPS line on the σ grid was labelled at the right end, where the validation curve
  turns back up through it. It moved to the left end, where the curve is at its highest and
  the strip under the line is empty.

Two smaller things the render settled rather than caught: the verdict figure's axis room is
a **percentage**-width constant rather than the wide one the count-labelled bars need, and
the shipped σ is deliberately *not* marked on the grid figure, because an enlarged marker
there already means "the optimum on this grid" and a second mark meaning something else is
worse than a caption.

**The live page driven in Chrome** through Playwright — **45 checks, all passing**: six
navigation rows in declared order, `/minutes` deep-links, eleven plots draw, all ten block
headings are present, no `stException`, no missing-artifact alert, no literal `"undefined"`
and no literal `nan` before or after an interaction, every one of the fifteen numbers the
three named blocks exist to show is on the page, no metric tile clips its own value, sidebar
order is nav → Appearance → page controls by geometry, the head selector reaches the
composition at its own unit and the two-unit block still draws for it, the mode switch
repaints to the pinned dark surface and holds across an in-app navigation in both
directions, and Tournament and Availability still draw their own five and six plots.

A third defect came from **reading** the rendered page rather than asserting it, which is now
four sessions in a row: **the σ block's tiles are all readings of the composition** — shipped
σ, the gap, the predictive sd, the PIT KS — and they render identically whichever head the
selector has open, so with `min|available` open a skimmer could read "Predictive sd 280.87"
as the marginal head's. The caption now says the block is the composition's throughout and
the two ambiguous tiles are labelled `· injected`.

**20 new tests**, 225 in `tests/test_dashboard.py` and **1,324** across the suite. Three of
them are artifact-contract tests: the reversal across the two units still holds on the real
artifacts, the shipped σ is still the train grid's optimum, and both minutes heads still
render all seven blocks.

---

## Step 6, as built — Inputs beyond the heads

**2026-08-10.** `dashboard/views/beyond_heads.py` over a new pure layer
`dashboard/inputs.py`, four new figures in `charts.py` and two reused, one new emitter
(`src/data/capture_calendar.py`, `make capture-calendar`) writing two small CSVs, and 38 new
tests. The page is one row in `app.VIEWS` at `url_path="inputs"`, sitting between Game
length and Tournament & strategy so the navigation reads in page order.

The design in "Page 7" above stands as written — three blocks, the calendar as an
operational alarm, every calibrated input showing its `fit_window`. Four things it did not
anticipate are below, and three of the four were found by *looking at a rendered figure*
rather than by asserting about one.

### Block 4 · the availability layout — added 2026-08-12

**A fifth simulator input, and the first one on this page that is not a number.** The
availability head draws *how many* games a player misses; where they fall is a separate
choice made at draw time, because games played is invariant to the arrangement and no
likelihood over it can carry one. `sim.availability.layout` ships `tenure_merge`
(`docs/availability-window-plan.md` §13), and the block draws what it does and how close it
lands: three scoring-period statistics against the realized season, per role bucket, plus
the realized absence-spell distribution and the two keys the edge-block draw is resampled
on.

It belongs on this page rather than on Availability (page 3) for the same reason the copula
does: **page 3 draws posteriors and this is not one.** It is chosen rather than fitted.

**One charter rule the block needed, and it is new to this page.** The ladder artifact
carries four *drawn* arms — the shipped one and the three that selected it — and the block
draws exactly one of them, beside `observed`. The rejected arms are an argument for a
choice, and an argument is what the walkthrough was removed for being; they live in the plan
doc and in `dashboard/decisions.py`. Two consequences fall out of stating it:

- **The block reads levels rather than the ladder's own `recovered_share`.** That statistic
  is a ratio against the *exchangeable* arm, so quoting it would put a rejected arm on
  screen through the denominator while appearing not to. Levels in each metric's own unit —
  two shares and a run length — carry the same reading without one.
- **`observed` is not an arm and stays.** It is the realized played/missed vector the layout
  exists to reproduce, so it takes the hollow-ink reference marker `fig_coupling` and the
  game-length page already use for "this is the target, not a rival model", and
  `fig_layout_exposure` is a faceted dumbbell for that reason.

`test_no_rejected_layout_arm_reaches_the_page_source` pins the rule as a **source scan**
rather than a frame assertion, which is the level it has to be at: the failure mode is not a
wrong figure but a caption that narrates the comparison, and a typed arm name would pass
every check on the data.

**One metric is deliberately absent.** `p_half_period` is nearly arrangement-invariant —
`availability_exchangeability._attach_gaps` already declines to score it — so drawing it
would read as a fourth diagnostic the layout fails rather than as a fact about that metric.

### The blocks are drawn in the opposite order to the way the plan lists them

The plan lists ADP first and the capture programs second. On the page the capture block is
**block 1**, because it is the only block on this dashboard that is an *alarm* rather than a
result: everything else here is a measurement that will still be there tomorrow, and a
missing capture day is a thing that stops being fixable. Putting it below a six-tile ADP
block would have made a reader scroll past the one thing that is time-sensitive. The two
ADP blocks and the calibrated block are then in the plan's order.

### The emitter is two files, because the calendar cannot carry the recovery policy

`make capture-calendar` writes `capture_calendar.csv` — one row per (program, day) that has
a state — and `capture_programs.csv`, one row per program. The split is not tidiness. The
calendar answers *which days*; the program table answers *what happens to a day that is
missing*, which is a fact about the **source** rather than about the archive, and it is the
one thing a reader cannot infer from a grid of cells. The emitter re-reads
`injury_reports.capture_status` and `injuries`' snapshot/missing-day pair rather than
re-deriving the states, so the printout and the artifact cannot drift apart, and a test pins
that they agree. It runs at the end of `make daily-capture`: a scheduler that stops firing
is only visible in the artifact it stops refreshing.

**The state vocabulary is three words and the middle one is load-bearing.** `captured`,
`nothing_to_capture`, `missed`. A day the CDN 403s is the offseason and is *not* a failure;
a day nobody attempted is a run that did not happen. On the live archive that distinction is
47 days out of 211, so colouring the middle state as a gap would cry wolf on a quarter of the
calendar. Registered as `a-day-with-no-report-is-not-a-gap`.

**Recoverability is a property of the program and rides on the row label.** An
injury-report gap is fetchable until it ages out and an ESPN gap never is, and neither varies
along its own row. A fourth cell colour would also have put orange beside red — the exact
pair `theme.py`'s validation rejects — so the axis carries it and the cells stay at two slots
and a neutral. Registered as `recoverability-rides-on-the-row-not-the-cell`.

**An `event` program has no schedule to have missed**, so the DK and FantasyPros rows are
drawn as sparse captures with nothing in between rather than as a year of failures a year.
That is why `cadence` is a column. Registered as
`an-event-programs-empty-days-are-not-gaps`.

### The window disagreement is the finding, not a bug to reconcile

`residual_correlation.to_matrix` defaults to `train_val` and `src/sim/season.py` overrides it
to `train`, and **both are right**. `train_val` is the window that is never *wrong* — it
excludes the test seasons and nothing else — so it is the safe default for an unthinking
caller. The shipped simulation is scored against a backtest on 2022-23 and 2023-24, which
are *inside* `train_val`, so a run at that window would calibrate itself on the seasons it is
about to be marked on. The page draws all four inputs at all three windows and names both.
The measurement that makes the block worth drawing: **the largest of the four moves 3.9%
across the three windows and the rest by less** — the residual copula's count-block mean r
reads +0.0225 / +0.0222 / +0.0216, the ten-game block inflation 2.4321× / 2.4206× / 2.4167×,
the player-game bonus overdispersion 0.0248 / 0.0248 / 0.0249, and the game-level minutes
dispersion 4.648× / 4.684× / 4.716×. A number that moved visibly would have been caught years
ago; one that does not is the kind that gets consumed at the wrong window forever.
Registered as `the-fit-window-is-shown-rather-than-chosen-for-the-reader`.

### Two figures were reused rather than written, and one of them needed a one-line fix

The three-window panel is `fig_metric_facets`, the minutes page's builder — the same shape in
both places: a handful of rows compared inside each facet, facets in different units. Reusing
it needed `_head_colors` to fall back to `muted` for a row its slot map does not name,
instead of to the next categorical slot. That is what lets one map serve both a *fixed
pairing* (the minutes page names both its heads) and *highlight-and-gray* (the inputs page
names only the window the simulator consumes). Handing an unnamed row a colour nobody chose
was the worse behaviour either way. Registered as
`an-unnamed-row-grays-out-rather-than-taking-a-slot`.

The copula heatmap is `fig_correlation`, which gained one `limit` parameter — see below.

### Verification, as run

**`AppTest`**, both appearance modes × all three fit windows: **6 charts, 14 tiles, 11
tables, 1 selector, 1 warning, 12 captions, 0 exceptions, 0 missing-artifact warnings**,
identical in every one of the six combinations. The other pages still report their own
counts.

**Twelve figures rendered to PNG** through kaleido, in both modes, and looked at. **Three
defects, all invisible in the trace:**

- 🔴 **The copula heatmap showed its own diagonal and nothing else.** `fig_correlation` pins
  its scale to [−1, +1], which is right on a model page — feature correlations run the whole
  range, and pinning them is what makes two heads' heatmaps comparable. The copula's largest
  off-diagonal cell is **+0.133**, so every real cell rendered as the neutral midpoint and
  the figure reported *no dependence* about a matrix that exists to carry some. The diagonal
  is what forces the scale (1.0 by construction, no information), so narrowing to ±0.15 and
  masking the diagonal are one decision, and the caller states the limit rather than the
  builder guessing it. Registered as `a-correlation-heatmap-is-scaled-to-what-it-carries`.
- **The calendar read as a barcode.** At 150 days a two-pixel `xgap` is nearly as wide as a
  cell, so an unbroken run of captures came out striped and a genuinely missing day was
  indistinguishable from the gutter between two present ones — on the one chart whose whole
  job is making a hole visible. Days are now flush and only the rows are separated.
- **The block-inflation reference line was a gridline.** Eleven of its twelve bars end within
  half a unit of 1.0, and at `_reference_line`'s hairline weight the reference was
  indistinguishable from the gridlines beside it. Drawn `layer="below"` it also survived only
  in the gutters between bars, which *reads as a dashed line* — and `theme.py` bans dashes
  because a dash is supposed to mean something. It is now its own line, above the bars, at
  weight 2.

Two smaller things the render settled rather than caught: the calendar's row labels went to
two lines (on one line the longest is 62 characters, and plotly gives a tick label whatever
width it asks for, so the axis was taking a third of the plot away from the calendar it
labels), and the ADP timeline's zero line is drawn bare because its axis title already reads
"days from the season's first game — negative is before it".

**The live page driven in Chrome** through Playwright — **28 checks, all passing**:
`/inputs` deep-links, the navigation lists eight rows with page 7 sixth, all three block
headings are present, six plots draw, no `stException`, no missing-artifact alert, no literal
`"undefined"` and no literal `nan` before or after an interaction, all fourteen tiles render
without clipping their own values, the four numbers the ADP block exists to show are on the
page, sidebar order is nav → Appearance → Fit window by geometry, switching the fit window
changes the numbers rather than decorating them, the mode switch repaints to the pinned dark
surface and holds across an in-app round trip to the tournament page and back, and the
calendar's table twin opens.

**A fourth defect came from reading the rendered page rather than asserting it**, which is
now five sessions in a row: **the capture block had four stacked bold paragraphs describing
each source**, which is prose about the project on a surface whose charter says prose belongs
in `docs/`. The block had turned back into a document. They moved inside the expander, beside
the table they annotate; the main surface keeps the alarm, the calendar and its two captions.

**38 new tests** — 30 in `tests/test_dashboard.py` (262 there) and 8 in a new
`tests/test_capture_calendar.py` — and **1,369** across the suite. One of the emitter tests
caught a real bug: a manifest that exists and is empty made `read_csv` raise, so
`make capture-calendar` fell over on exactly the archive it exists to report as missing.
Four are artifact-contract tests: the shipped panel still loses four of its nine seasons, the
shipped ladder still names the arm that ships, the calendar and its program table are still
one measurement, and all four calibrated inputs still read at all three windows within 5%.

---

## Step 7, as built — the draft board as page 9

**2026-08-10.** One row in `app.VIEWS`, a three-line `dashboard/views/draft_room.py`, a
`render()` / `main()` split inside `dashboard/draft_room.py`, `recall` / `remember` in
`shell.py`, and 7 new tests. **No new artifact, no pipeline run and no change to what the
room computes** — `src/sim/draft_room.py` was not touched.

### The condition written as a risk measured the other way

The step was allowed to leave the two launches separate if navigation made the room slower
to first paint. It does not: it makes it *faster*, and a return visit is not close.
Playwright driving Chrome, "first paint" being the moment the recommendation's `Recompute`
tile is on screen, three rounds against freshly restarted servers:

| launch | run 1 | run 2 | run 3 |
|---|---|---|---|
| `make draft-room`, cold process | 3.57 s | 3.66 s | 3.67 s |
| page 9 from the navigation, cold process | 3.18 s | 3.18 s | 3.15 s |
| page 9, away to another page and back | 0.31 s | 0.32 s | 0.33 s |
| page 9, away and back a second time | 0.31 s | 0.33 s | 0.31 s |
| `make draft-room`, second browser session on the same process | 0.45 s | 0.44 s | 0.43 s |

Both cold rows pay the same 2.06 s `load_room` and the same 0.89 s
`import src.sim.draft_room`. What separates them is that a `goto` against a cold server
also downloads Streamlit's frontend bundle and opens a fresh websocket, where a navigation
click has both already; the last row isolates that, being a warm cache reached through a
fresh page load and costing 0.12 s more than the in-app return. **The row that matters is
the third: 0.31 s to re-enter a room that costs 3.6 s to launch**, which is finding 1
arriving as a number rather than as an argument.

So the standalone launch stays for the reason it was specified and not for speed. Draft
night is a thirty-second clock, and what a separate process buys is that nothing else can
raise, block or allocate inside it — `main()` is `st.set_page_config` plus `render()`, and
that is the whole of the split.

### The field is not rebuilt, and the page says so itself

`open_room` is `@st.cache_resource(show_spinner="Drafting the reference field — once per
season…")`, so a cache **miss is visible in the DOM** and the browser layer can assert on it
instead of inferring it from a stopwatch. Polling every 20 ms across each navigation, the
spinner appears on both cold launches and on **neither** return. `AppTest` checks the same
claim by counting rather than by looking — `src.sim.draft_room.load_room` patched to count
its calls, out to the fingerprint page and back: **one call, zero rebuilds.**

### The move exposed a defect that exists only because it moved

`appearance-lives-in-the-shell` recorded that Streamlit clears widget state for a page the
reader has left, and the escape it used — render the control from the entrypoint — is
not available to a control that belongs to *one* page. The room is where that stops being
cosmetic, because its state is split down the middle: the **pick log is a plain
session-state key and survives** a navigation, while the season, tournament, seat and
objective are widgets and do not. A reader who steps away mid-draft therefore does not lose
the draft. They keep it, and re-read it under different assumptions.

Measured under `AppTest` with every control moved off its default, one pick taken, out to
the fingerprint page and back:

| | season | tournament | seat | objective | picks |
|---|---|---|---|---|---|
| before | 2022-23 | `600k_shootaround` | 5 | `p_advance` | `[2]` |
| after, widget state only | **2023-24** | `600k_shootaround` | **1** | **`bracket_ev`** | `[2]` |
| after, as shipped | 2022-23 | `600k_shootaround` | 5 | `p_advance` | `[2]` |

The seat is the sharp one — the page's own docstring calls wrong-seat bookkeeping "the
failure that silently invalidates everything below it" — but the **season is the one that
changes what the log says**: board index 2 is Luka Dončić on the 2022-23 board and Giannis
Antetokounmpo on the 2023-24 one, so the identical pick log names a different pod.

The fix is `shell.recall` / `shell.remember`, a namespaced plain key the page writes as it
runs: not widget state, so nothing clears it. It is deliberately **not** a general escape
hatch from the shell — genuinely global state still belongs in the entrypoint, and this is
for the narrower case where a page's own control says what that page's own surviving state
means. `top` (candidates shown) is left to reset, because it is cosmetic, and that is the
line. Registered as `a-page-control-that-names-the-state-must-outlive-the-page`.

### The `src/` exemption came out narrower than it went in

The room needed a row-owning module under `views/`, and naming it `views/draft_room.py`
would have handed it the exemption for free: `SRC_IMPORTERS` matched on **basename**, so a
second file called `draft_room.py` anywhere in the package was exempt the moment it
existed. The keys are now paths. One file is exempt, it is still held to `src.sim`, and the
wrapper is held to the ordinary rule — it imports nothing from `src/` at all, and a test
pins each half of that.

The wrapper also **defers** its import of the room to inside `render()`. `app.py` imports
every view module before it draws anything, so a module-level import would have put 0.89 s
of `src.sim` on the startup path of every page including the ones that never touch the
simulation layer; a subprocess test pins that importing the wrapper pulls in no `src.*` at
all.

### The middle verification layer has nothing to render

Page 9 draws **no figure** — it is buttons, metric tiles, captions and two tables — so the
PNG layer is skipped for the first time in the expansion, and the browser layer does the
work of both. One consequence is worth stating rather than filing as a bug: **the
appearance toggle is inert on this page**, because what the mode selects is a chart palette
and there are no charts. It still *holds* across the page, which is the part that matters,
and the tournament page repaints to the pinned dark surface after a round trip through the
room. **No longer true as of 2026-08-10**, and in the useful direction: the toggle is gone,
the appearance is Streamlit's own, and the page chrome comes from the same palette — so the
room now changes with the mode in both of its launches, charts or no charts. See
`docs/dashboard-revision-plan.md`, "Step 1, as built".

### Verification, as run

**`AppTest`**, both appearance modes: **36 buttons, 9 metric tiles, 1 table, 3 selectboxes,
10 captions, 3 expanders, 1 warning, 0 errors, 0 exceptions**, identical in light and dark.
The warning is the room's own injury-relevance banner — the feeds describe today and the
board is 2023-24 — which is the page working rather than a defect. The counted round trip
above holds `picks`, the seat and `appearance`, and the fingerprint page still drew its two
charts on the way through.

**The live pages driven in Chrome** through Playwright — **28 checks, all passing**: eight
navigation rows in the declared order with the draft board last, `/draft-room` deep-links,
the room paints from the navigation, its own 15 px root scale applies **and does not leak**
(16 px → 15 px → 16 px across the trip), sidebar order is nav → Appearance → Draft room by
geometry, no `stException`, no literal `undefined` and no literal `nan` before or after a
pick, all six of the room's landmarks are on the page, a click puts a pick in the log, the
log and the seat survive a navigation while the player taken stays off the board, the
return does not rebuild the field, dark mode leaves the room intact and still repaints the
tournament page to `rgb(26, 26, 25)`, and `make draft-room` serves a page with no
navigation and no exception.

Reading the rendered navigation rather than asserting on it caught one thing: the row had
taken `:material/sports_basketball:`, which the box-score components page already owns. Two
identical icons in one sidebar is a row the reader has to read twice, so the draft board
draws a ranked list instead.

Two things the browser layer cost that are about *reading* a Streamlit page rather than
about this page, recorded in `dashboard/README.md` with the rest of that list:

- **Streamlit streams a page's blocks**, so `inner_text` at first paint sees the top of the
  page only. The recommendation exists seconds before the board, the roster and the pick
  log below it; three checks failed against a page that was fine.
- **The first `button` in the main container is a zero-size chrome element** and clicking it
  silently does nothing — a pick check that had never taken a pick, and passed its
  "the player is gone" companion for the wrong reason. A pick button is the one carrying
  `·` separators.

**7 new tests**, and with step 6's landing in the same working tree `tests/test_dashboard.py`
reads **262** and the suite **1,369**.

---

## Step 8, as built — the Overview

> **Partly superseded 2026-08-10**, the same day, by step 5 of
> `docs/dashboard-revision-plan.md`: the five hero tiles are now four sections of prose and
> bound 1 is amended to "opens above the fold, scrolls no further than one screen more"
> (see [the amendment](#the-amendment-to-bound-1--2026-08-10-and-it-is-a-ceiling-rather-than-a-waiver)).
> Everything below is the record of what shipped first, and three of its four subsections
> are untouched by the rewrite: `Spec` is unchanged, bound 3 is unchanged, and
> `st.page_link` still needs the entrypoint's own page objects. What moved is the tile row
> and the height.

**2026-08-10, and the expansion lands with it.** `dashboard/overview.py` (the pure layer),
`dashboard/views/overview.py` (the page), `charts.fig_pipeline`, a page registry in
`shell.py`, one row at the **front** of `app.VIEWS`, and 19 new tests. No new artifact, no
pipeline run: the page reads eight CSVs that five earlier `make` targets had already
written, three of which — `variance_budget.csv`, `game_length_coverage.csv` and
`sim_season_gate_a.csv` — this dashboard had never drawn.

Five hero tiles, one five-box pipeline diagram, eight route links, and one paragraph of
prose. That is the whole page, and the shape of it was set by the first bound rather than
chosen.

### Bound 1 turned out to be a measurement, and the first draft failed it

"One page, one screen" is the term the exemption was granted on, and it is not assertable
from Python — `AppTest` has no DOM and reports the same five tiles and eight links whatever
they are laid out as. Driven in Chrome at 1440×900, the first complete draft came in at
**1,144 px against a 900 px viewport**: it scrolled, which by the charter's own words meant
it had become the walkthrough again.

It ships at **702 px**, with 198 px of headroom, and at **769 px** on a 1280×800 laptop.
Where the 442 px came from, largest first:

| change | why it was there | saved |
|---|---|---|
| Streamlit's default chrome | 6 rem of padding above the first element, a 2.5 rem `h1`, 1 rem between every vertical block — defaults laid out for a scrolling document | ~150 px |
| Route blurbs cut to one line | eight three-line blurbs is 200 px of the screen; they now say what a page holds in one line each | ~90 px |
| The diagram, 200 px → 150 px | the stage notes were a wrapped sentence each; the arrow chain already carries the meaning they were spelling out | 50 px |
| The intro paragraph, 6 lines → 4 | it explained the contest twice | ~52 px |
| The tile caption, 2 lines → 1 | — | 22 px |

Nothing was *removed* to make it fit — the same five tiles, five stages and eight routes are
on the page — which is the part worth recording. The bound cost the page its verbosity and
not its content, and that is the argument for having written the bound as a hard one.

**The compression lives in the view, not in `shell.py`.** `COMPACT_CSS` is scoped to the
main container so it cannot reach the sidebar, and it is deliberately not shared: a model
page is *supposed* to scroll, and inheriting a type scale chosen for a landing page would
only make it scroll further. That is the same argument `shell.compact_tiles` is opt-in for.

### Bound 2 is a table, so a figure cannot reach the page by being typed

Every number on the page arrives through `overview.Spec` — a `build` callable over the
frames it `needs` — so the view holds layout and the pure layer holds every lookup, and the
season-total MAE is read from `season_total_metrics.csv` exactly as the tournament page
reads its own artifacts. Three consequences fell out of writing it that way rather than
inlining five `read_csv` calls:

- **A half-built repo loses readings, not the page.** `_resolve` drops a reading whose
  source is missing and keeps the rest, and a test removes each of the eight sources in turn
  and asserts the page is neither empty nor complete. A landing page that hard-fails on the
  one machine where `make strategy-sweep` has not run yet is a landing page nobody sees.
- **Derived numbers are derived, not retyped.** The advance tile prints a rate, a lift and a
  field null; the null is `rate − lift` rather than a typed 1/6, so the three numbers on the
  tile add up in front of the reader and cannot drift apart across a re-run. A test asserts
  the identity rather than the value.
- **The floor band is filtered where a reader can see the filter.** `R² 0.81–0.95` is the
  **count** heads' no-fit floor; read off the conversion heads too it opens to [0.13, 0.96],
  which `docs/simulations-plan.md` has already had to re-derive once. `FLOOR_KIND` is a
  named constant with a test on it.

A missing row **raises** rather than returning NaN, which is the opposite of
`model_cards.text()`'s em dash and deliberately so: a model page with one blank cell is
still a model page, and a landing page whose hero tile reads `nan` is not.

### Bound 3 is a grep, and the place it would have been broken is the route labels

No decision registry, no provenance links, no reversal log. The test parses both modules
and checks that neither imports `decisions`, and that no string literal reaching a reader
contains `docs/` or `withdrawn` — docstrings exempt, since both files cite the charter they
live under in their own headers. The route blurbs are held to the same rule and to one more:
**no digits**. Eight one-line labels sitting next to eight links is exactly where "the
composition is 4.68× too narrow" would have crept back in as a teaser.

### `st.page_link` needs the entrypoint's own page objects

The route block is why this step was last, and it needed one piece of plumbing.
`st.page_link` accepts only a `st.Page` that `st.navigation` was actually handed, and those
are constructed in `app.main()` — so a page linking to its siblings must be *given* them.
Rebuilding them inside the view would collide on `url_path`, and importing `app` from a view
is a cycle. `shell.publish_pages` / `shell.page` are the third instance of the asymmetry
that module already exists for: the entrypoint runs on every rerun and a `render()` does
not.

**Keyed by the `url_path` `app.VIEWS` declares, not by `StreamlitPage.url_path`** — Streamlit
rewrites the *default* page's own path to `""` so it can serve `/`. The Overview is now the
default page, so reading the key back off the object would silently lose exactly one row,
and it would lose the one row nothing links to. A test pins it with a fake page that reports
`""`, because nothing in the app would have failed.

### Three defects only the rendering could see, which is now six sessions in a row

- **A metric's delta neither wraps nor truncates honestly.** Five tiles across, the
  comparison rendered as `-210.3 against assuming a f…` — a phrase cut mid-word, which is
  worse than no comparison at all. Fixed by a `stMetricDelta` font rule (the test ID read
  off this Streamlit's bundle, as `shell.TILE_CSS`'s were) *and* by writing deltas that fit:
  `-210.3 vs a full season` and `+12.7% vs a 16.7% field`.
- **A column is a vertical stack, so an 8-cell grid filled column-major comes out ragged.**
  One blurb wrapping to two lines in the top row pushed only *its* column's second link
  down. The routes are now one `st.columns` call per row, so a wrap can cost alignment
  inside its own row and nowhere else. `AppTest` counts eight page links either way.
- **Shrinking `h1` clipped its own ascenders.** Removing the heading's top padding to buy
  vertical space cropped the top of the type; it needs an explicit `line-height` beside the
  smaller `font-size`, not just less padding.

A fourth, from the PNG layer: with zero horizontal margin the outer boxes' 1 px borders land
exactly on the paper edge and are clipped, so the padding lives in the axis range instead.

### The diagram plots no data, and takes no colour

`fig_pipeline` is the only figure in `charts.py` that holds no traces — it is five rects,
fifteen annotations and four arrowheads. It is **uncoloured on purpose**: five steps is past
`ALL_PAIRS_CAP`, and more to the point the steps are not a scale and not categories being
compared, so five hues would be an encoding that decodes to nothing. Every box takes
`th["neutral"]` and prints its own figure, which satisfies the relief rule trivially. The
notes are wrapped in the pure layer at `NOTE_WIDTH`, sized for the **narrowest** checked
viewport, because a plotly annotation neither wraps nor clips — an over-long note simply
runs out over its own box, and the figure spec cannot see it.

### Verification, as run

**`AppTest`**, both appearance modes: **1 chart, 5 metric tiles, 8 page links, 9 captions,
0 warnings, 0 errors, 0 exceptions**, identical in light and dark. Every page link resolves
to a registered page, and `/` serves the Overview.

**The figure rendered to PNG** in both modes through kaleido, which caught the clipped outer
borders.

**The live page driven in Chrome** through Playwright — **19 checks, all passing** at both
1440×900 and 1280×800: five tiles, eight routes, the page fits the viewport with no scroll
at either size, no `stException`, no literal `undefined`, `NaN` or `Traceback`, dark mode
repaints to the pinned `rgb(26, 26, 25)` surface, and clicking the Tournament & strategy
route navigates to `/tournament` and paints it.

**19 new tests** in `tests/test_dashboard.py` (**281** there) and **1,388** across the suite.
One existing test changed rather than being added to: `test_the_first_page_is_the_real_one`
now names the Overview, which is the assertion that `/` serves the page written for a reader
with no context rather than a radial chart of a player-season they did not choose.

---

## Structure

```
dashboard/
  README.md       # the rules a new view has to follow
  overview.py     # page 1's pure layer — the four sections' sentences, the five
                  #   pipeline stages and the nine routes, each a lookup into an artifact
  __init__.py
  app.py          # the entrypoint — st.navigation, and VIEWS, the sidebar
  shell.py        # cross-page state: the appearance mode, current_theme(), the
                  #   opt-in metric-tile type scale, and the page registry the
                  #   Overview's st.page_link rows come out of
  views/
    overview.py       # page 1 — the one page of prose, under the charter amendment's
                      #   three bounds; its CSS is what holds it to one screen
    fingerprints.py   # the PCA fingerprint page — controls, layout, render()
    availability.py   # page 3 — four lines that name a model class
    minutes.py        # page 4 — the class, plus three named blocks: one posterior at
                      #   two units, the injected effect, and the zero-sum constraint
    components.py     # page 5 — the class, plus the eleven heads against their floor
    game_length.py    # page 6 — the class, plus both heads read in games
    model_page.py     # the seven-block model detail page, once, for all four classes
    tournament.py     # the contest, the sweep, both backtests, the paired gaps
    draft_room.py     # page 9's row — three lines that defer to the room below, so the
                      #   import that reaches src.sim happens on demand and not at start
  draft_room.py   # the live draft room — page 9 AND its own app (make draft-room), the
                  #   one file allowed to import from src/, bounded at src.sim
  pca.py          # the fingerprint view's pure layer — orientation, SD scaling,
                  #   loadings, neighbours
  strategy.py     # the tournament view's pure layer — the contest summary, the
                  #   hurdle-in-survival-units conversion, the sweep facets, the
                  #   two surfaces, the paired gaps
  model_cards.py  # the model pages' pure layer — the class table, the seven blocks
                  #   as frames, where each head's sampler row lives, and the
                  #   page-specific blocks (the component floor board, the game-length
                  #   class counts, and the minutes page's two-unit board, spread
                  #   panel, sigma sweep and teammate coupling)
  charts.py       # fig_radar / fig_loadings, the five tournament figures, the six
                  #   model-page figures, the six a single page owns, page 7's four,
                  #   and fig_pipeline — the one figure here that plots no data
  theme.py        # SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  artifacts.py    # load_cfg, features_dir, predictions_dir, read_table, optional
  decisions.py    # NOT the dashboard — the project decision registry (see below)
  economics.py    # NOT the dashboard — contest arithmetic for the drafting layer
  audit.py        # NOT the dashboard — make dashboard-audit
```

`app.py`, `shell.py`, `artifacts.py`, `draft_room.py` and everything under `views/` are the
Streamlit surface; everything else is pure, which is what lets `tests/test_dashboard.py`
exercise the palette rules, the component spec, the scaling, the neighbour metric and both
figures as plain functions. That guard is now a **denylist over the whole package** rather
than an allowlist of six filenames, so each of the seven pages still to come is pure by
default and has to be named before it can import Streamlit.

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

Added with the shell on 2026-08-10, exercising `app.VIEWS` as data — `st.Page` is only
constructed inside `app.pages()`, so the navigation registry is testable without a runtime:

- **The navigation carries at least two entries**, with the measured reason in the
  docstring, so a future edit back to one page fails rather than silently rendering no nav.
- **Every page has a unique `url_path` and a callable**; a duplicate path is a
  `StreamlitAPIException` raised at nav-build time, i.e. in a browser.
- **The first page is the real one**, since `pages()` makes index 0 the default and `/`
  must not serve a placeholder.
- **Every navigation row is a real view module** — added with step 2, when the placeholder
  went. `pages()` hands `st.Page` a bare callable, so a closure or a stub would navigate
  fine and own a URL no module does.
- **The shell offers exactly the modes the palette defines**, `shell.MODES` against
  `theme.THEMES` — a mode with no palette entry is a `KeyError` inside `theme()`.
- **The Streamlit-purity guard became a denylist**, plus a test that every name on it still
  exists, since a denylist naming a deleted file silently stops guarding a real one.

Added with the tournament page on 2026-08-10 — 25 tests over `dashboard/strategy.py` and
the five new figures. Most use synthetic builders; the handful that read the real
`outputs/predictions/` artifacts are the ones keeping a *derived* quantity honest against
the artifact it is derived from, which is the same job the PCA anchor tests do:

- **The hurdle conversion, end to end.** `p_null + break_even_lift(p_null, hurdle)` equals
  `p_null/(1 − rake)` exactly, so the reference line is the definition rather than a
  coefficient. A null outside `(0, 1)` and a hurdle at or below −1 raise.
- **The elasticity that makes it conservative.** Reads exactly 1 on a synthetic
  proportional payout and exactly 3 on a cubed one; an arm sitting *on* the null is dropped
  rather than dividing by `log(1)`; and on the shipped sweep it is above 1 on every row of
  both tiers, which is the measurement the page prints.
- **Two routes to the same survival.** The economics chain's `p_reach_final` matches
  `bracket_structure.csv`'s own `p_reach_analytic` for every tournament — the page tiles
  one and draws the other, so a disagreement would put two numbers for one quantity on one
  screen.
- **Arm ordering is per tier and on the mean.** A synthetic arm that wins one season
  outright but loses on the mean stays below; and on the real sweep the two tiers' `α`
  orderings still differ, which is why the order is not fixed.
- **`crosses_zero` is the interval, not the flag.** A deliberately drifted `resolved`
  column does not change the styling, and on the shipped artifact the derived flag agrees
  with `resolved` across every tournament × metric × baseline.
- **The palette rules, as arithmetic.** Five tournaments use at most `ALL_PAIRS_CAP`
  categorical slots with the rest at `muted`; a target's slot comes from `TARGET_TIERS`
  order rather than row order, so the two block-1 figures cannot disagree; the sweep's two
  seasons plus its break-even line are exactly three slots; and every hurdle bar prints its
  own value.
- **The figures' geometry.** One subplot per facet with both reference lines in each; the
  best arm on the top row; the two seasons nudged off their shared row by an equal
  offset; no zeroline on a row-position axis; a rule between the two backtest surfaces; a
  hollow marker only where the interval actually covers zero; and no series at all for
  "does not resolve" when nothing does. Every figure carries an explicit title, the
  `"undefined"` guard.

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
