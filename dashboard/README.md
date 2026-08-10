# The dashboard

Data visualizations over the artifacts the pipeline wrote. Three views today — the PCA
player-style fingerprint, the Availability model detail page, and the tournament & strategy
page — inside a multipage shell that six more pages plug into. Run it with `make dashboard`;
the plan is `docs/dashboard-plan.md`.

**The dashboard shows data. Prose about the project belongs in `docs/`.** The nine-tab
project walkthrough that used to live here was documentation rendered as an app, and every
claim on it had to be kept in sync with a document that already made the claim. It was
removed on 2026-08-08; commit `e8e58b0` holds it.

## The two rules

> **Every figure is read from an artifact that a `make` target produced.** If a number is
> not in an artifact, it does not go on the page.

> **Nothing in this package imports from `src/`.** The dashboard cannot refit, re-project
> or re-cluster anything — it opens files. `test_the_dashboard_imports_nothing_from_src`
> walks the package with `ast` and fails if one appears.

**One page is exempt and the exemption is pinned rather than waived.** `draft_room.py` is
not a view — it drives a live draft under a thirty-second clock — and it needs
`bracket.best_lineup` and `draft.legal_mask`. The alternative to importing them is
reimplementing the matroid that seats a weekly lineup and the rules that decide which
players are legal, which is the drift the rule exists to prevent arriving through the
other door. So `SRC_IMPORTERS` in `tests/test_dashboard.py` names the one file and
`test_the_one_exempt_page_reaches_no_further_than_the_simulation_layer` holds it to
`src.sim` — numpy over the artifacts, importing no CmdStan — so an exempt page still
cannot refit anything. Everything it computes lives in `src/sim/draft_room.py`; this file
is the surface. Registered as `draft-room-imports-src-sim`.

Where a view *interprets* an artifact — naming a principal component, say — the
interpretation carries a machine-checkable anchor so it cannot silently invert. See
`pca.COMPONENTS` and `pca.orient()`.

## Layout

```
dashboard/
  README.md       this file
  app.py          the entrypoint — st.navigation over the pages, and VIEWS, the sidebar
  shell.py        cross-page state: the appearance mode, current_theme(), and the
                  opt-in metric-tile type scale
  views/
    fingerprints.py   the PCA fingerprint page — controls, layout, render()
    availability.py   page 3 — four lines that name a model class
    model_page.py     the seven-block model detail page, written once and shared by
                      the four model classes (pages 3-6)
    tournament.py     the contest structure, the strategy sweep, simulated against
                      realized, and the paired gaps
  draft_room.py   the live draft room — `make draft-room`. Its own app, not a page of
                  app.py, and the one file that imports src/ (see above)
  pca.py          the fingerprint view's pure layer — orientation, SD scaling, loadings,
                  neighbours
  strategy.py     the tournament view's pure layer — the contest summary, the hurdle
                  converted into survival units, the sweep facets, the two backtest
                  surfaces, the paired gaps
  model_cards.py  the model pages' pure layer — the class table (which heads make a
                  page, in which order), the seven blocks as frames, and where each
                  head's `make stan` diagnostics row lives
  charts.py       fig_radar / fig_loadings, the five tournament figures, and the six
                  model-page figures
  theme.py        SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  artifacts.py    load_cfg, features_dir, predictions_dir, read_table, optional
  decisions.py    ─┐
  economics.py     ├ not the dashboard — see below
  audit.py        ─┘
```

`app.py`, `shell.py`, `artifacts.py`, `draft_room.py` and everything under `views/` are the
Streamlit surface; **everything else is pure**, which is what lets
`tests/test_dashboard.py` exercise the palette rules, the component spec, the scaling, the
neighbour metric and both figures as plain functions rather than through a rendered page.
That is a denylist rather than an allowlist, so a new module is pure by default and
`test_pure_modules_do_not_import_streamlit` fails until it is either kept pure or named.

## The shell

`st.navigation` rather than `st.tabs`, and the reason is structural: **Streamlit executes
the body of every tab on every rerun**, so nine tabs would re-run all nine on every
interaction, including whichever one loads the 90 MB simulation tensor. `st.navigation`
runs only the selected page's script, in one server process, so `@st.cache_data` and
`@st.cache_resource` stay shared and a tensor loaded by one page is still warm after a
navigation. See `docs/dashboard-plan.md`, finding 1.

Two consequences worth knowing before adding a page:

- **The entrypoint runs on every rerun; a `render()` runs only when its page is
  selected.** Anything that must survive navigation goes in `shell.py` and is rendered by
  `app.py` — today the appearance mode, which a page reads through
  `shell.current_theme()`. A widget a page declares is torn down when the reader leaves
  it: driven under `AppTest`, a round trip resets the fingerprint view's own `component`
  key from `pc8` to `pc1` while `appearance` holds.
- **Sidebar order follows that ownership**: the navigation, then the shell's controls,
  then whatever the page writes to `st.sidebar` for itself.

A page is one module in `views/` exposing `render() -> None`, plus one row in `app.VIEWS`
carrying its title, icon and `url_path`. **The four model detail pages are one renderer and
a class table**: `views/model_page.py` draws the seven blocks for whichever class it is
handed, `model_cards.CLASSES` carries each page's title, icon, `url_path`, head order and
specification intro, and `app.model_view()` builds the navigation row from that table — so
pages 4-6 are a four-line view module and one row, not a second renderer. The first row is the default page, i.e. what `/`
serves. **`st.navigation` renders nothing at all for a single-page app** — measured, not
assumed — so the shell keeps at least two rows, and two tests hold that: one on the row
count, one asserting every row's `render` comes from a module under `views/`. A
`views/placeholder.py` filled the second row until the tournament page took it; if the
build order ever needs one again, it shows **no numbers** and names **`make` targets rather
than artifact filenames**, because `audit.py` counts an artifact as read when any string
literal in `dashboard/` names it.

### Three files that are not the dashboard

They live here for historical reasons and are load bearing elsewhere. Do not delete them
with the rendering code:

- **`decisions.py`** — the project's decision registry, wired to a standing instruction in
  `CLAUDE.md` and to `make dashboard-audit`. `CLAUDE.md` and the `docs/*-plan.md` files are
  the source of truth; the registry is a distillation of them, and where the two disagree
  the docs are right.
- **`audit.py`** — `make dashboard-audit`, a report on registry drift. A **report, not a
  gate**: it exits 0 with findings, because failing the suite when somebody edits a doc
  trains people to ignore the suite. Only its missing-artifact check is also a `pytest`
  test.
- **`economics.py`** — rake, break-even hurdle and advance rates derived from the two real
  tournament CSVs. Pure arithmetic the drafting and backtest layers will consume.

## Colour

`theme.py` is the validated reference palette instance, used **unmodified**. Two rules from
that validation constrain every chart, and neither is a style preference:

- The eight categorical slots clear the colour-blind gates on *adjacent* pairs (bars, lines,
  stacks), but only the **first three** clear them on *all* pairs. Any chart where
  non-adjacent series sit side by side caps at `ALL_PAIRS_CAP = 3` and uses
  highlight-and-gray past that. The radial chart draws at most two series for this reason,
  and the tournament page's five-tournament figures highlight the two swept tiers and gray
  the other three — the same encoding in both, so the reader learns it once.
- Three light-mode slots fall below 3:1 contrast on the light surface, which obliges the
  relief rule: **no value is reachable by colour alone.** The radial chart is read off a
  labelled axis with rings at −2/−1/0/+1/+2 SD and ships a table twin; every loadings panel
  ships one too. On the tournament page every bar prints its own value, every interval is
  read off a labelled axis, and "this gap does not resolve" is carried three ways at once —
  the interval visibly straddles the zero line, the marker is hollow, and the legend and
  table twin both say so.

Sequential is a single blue hue; diverging is blue↔red with a **neutral gray** midpoint,
never a rainbow. A loading's sign is a direction on one axis, so the loadings bars take the
two ends of the diverging scale rather than two categorical slots. Plot surfaces are pinned
to the exact surfaces the palette was validated against (`#fcfcfb` / `#1a1a19`) rather than
inherited from Streamlit's chrome, so the measured contrast figures apply as documented.

## Adding a view

1. Put the pure logic in its own module with no Streamlit import, and test it directly.
2. Read artifacts through `artifacts.optional()`, which names the `make` target when a file
   is missing instead of raising.
3. Build figures in `charts.py` and hand them to `st.plotly_chart` — a figure builder takes
   the theme dict and returns a `go.Figure`, so it stays testable.
4. Put the page in `views/<name>.py` behind `render()`, add its row to `app.VIEWS`, and
   take the palette from `shell.current_theme()` rather than reading a mode yourself.
5. Verify in three layers, because each one sees what the one above it cannot.
   **`AppTest`** in both appearance modes proves the page runs. **A figure rendered to PNG**
   proves the figure is legible — it caught a loadings panel that silently dropped the
   negative half of an axis, and a plotly zeroline drawn through the top row of every
   panel of a dot plot. **The live page in a real browser** proves the page is, and
   is the only layer that can: it caught a plotly `title_font` with no text rendering as
   the literal string "undefined", metric tiles clipping their own values, and a click
   handler that never fired.

Two mechanics of the first layer, both of which cost a while to find:

- **`AppTest.switch_page` cannot reach these pages.** It resolves a *file* path and hashes
  the filename; `st.Page` over a bare callable hashes its `url_path` instead. Navigate with
  `at._page_hash = streamlit.util.calc_hash("<url_path>")`, which is the field
  `switch_page` sets anyway.
- **A selectbox with a `format_func` stores raw values and exposes formatted options**, so
  `.select("600k_shootaround")` raises and `.set_value("600k_shootaround")` is what works.

## Seven things plotly and Streamlit do that cost a day

Recorded because none is discoverable from the docs and all were found by looking at the
running page:

- **Streamlit reports no selection for a click on a `polar` trace.** `on_select="rerun",
  selection_mode="points"` returns `[]` for every click on a `Scatterpolar` and a full
  payload for a `Scatter` in the same app. The radial chart is therefore drawn on cartesian
  axes with its grid as shapes; see the header comment in `charts.py`.
- **A title object with a font and no text renders as "undefined".** Only in a browser —
  kaleido draws nothing — so `apply_theme` sets `title.text` explicitly.
- **`st.navigation` renders no navigation widget for a one-page app.** The Python side
  still sends `Position.SIDEBAR`; the frontend simply draws nothing, so
  `[data-testid="stSidebarNav"]` is absent from the DOM. `AppTest` cannot see this — it
  has no DOM — which is why the shell was verified in a browser before it was believed.
- **Widget state does not survive navigation.** Streamlit clears `st.session_state` for
  widgets the current page did not render, so anything global belongs in `shell.py`, where
  the entrypoint renders it on every rerun. Handy for the browser layer: plotly writes
  `paper_bgcolor` onto the `.main-svg` element's inline style, not onto its `rect.bg`
  (which sits at `fill-opacity: 0`), so the pinned surfaces are readable as
  `background: rgb(252, 252, 251)` and `background: rgb(26, 26, 25)`. It survives an
  **in-app** navigation; a `page.goto` in a browser test is a hard reload, a new session,
  and a legitimate reset to `detected_mode()` — drive the nav link instead.
- **A `go.Bar` with a zero-width or NaN `width` raises**, it does not fall back. That is
  reachable from an artifact rather than from a typo: a discrete feature's bin edges are its
  own values, so `bin_left == bin_right`, and the width has to be *derived* from the gap to
  the next value. `model_cards.histogram_panel` does it, and a test pins it.
- **A colourbar over more than one panel is a claim that they share a scale.** Four
  calibration panels binned as *shares of their own split* do not — a 751-row validation
  panel puts an order of magnitude more into each cell than an 8,232-row training one — so
  each panel is scaled to its own densest cell and the colourbar says so, with the raw share
  in the hover. Only visible by rendering the figure.
- **`st.dataframe` truncates the column that carries the content**, quietly and with no
  ellipsis in the DOM. A `column_config` width is a hint rather than a guarantee, and a wide
  table simply loses its right-hand columns off the edge. The fix that works is fewer and
  shorter columns, plus full width for a table whose long column is the point — checked by
  reading the rendered page, since `AppTest` reports a `dataframe` element either way.
- **Plotly does not offset grouped scatter, only grouped bars.** Two series at the same
  categorical y sit exactly on top of each other, so every dot-and-interval chart here puts
  its rows on a *numeric* y axis and nudges each series off the row by hand, restoring the
  names as tick text. Two consequences: turn `zeroline` off, since row 0 is a category and
  not an origin, and if a subplot is given extra height for a facet header, give its
  **range** the same extra or the rows just spread apart instead.
