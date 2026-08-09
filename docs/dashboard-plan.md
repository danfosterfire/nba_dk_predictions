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

Not built, and deliberately unspecified beyond a sentence each — the point of shipping one
view is to see what the next one should be.

- **The player-season trajectory.** The same fingerprint over a career, as a small-multiple
  or an animated path through the first two components. `pooled` mode is the right artifact
  for this one, because era is the axis you want visible.
- **The archetype membership vector.** `archetypes_*` is a soft membership over a continuum
  with a silhouette peaking at 0.181, so the honest picture is a stacked bar of memberships,
  never a hard label.
- **Posterior draws for a player-season.** Once the simulator lands, the deliverable is a
  distribution — the natural view is a density of simulated `dk_pts` with the realized
  season marked on it.
- **The draft board.** Model ranking against ADP, which is the plot the whole project is
  building toward.

## Publishing

Not a goal. Worth noting that this view needs exactly three files —
`pca_tierA_within_season_{scores.parquet, loadings.parquet, variance.csv}`, about 4.3 MB
total — so a deploy bundle is a filtering job rather than a rewrite. `component_targets.parquet`
(37 MB) is not read and must not start being read.
