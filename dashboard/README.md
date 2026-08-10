# The dashboard

Data visualizations over the artifacts the pipeline wrote. One view today — the PCA
player-style fingerprint — inside a multipage shell that seven more pages plug into. Run
it with `make dashboard`; the plan is `docs/dashboard-plan.md`.

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
  shell.py        cross-page state: the appearance mode and current_theme()
  views/
    fingerprints.py   the PCA fingerprint page — controls, layout, render()
    placeholder.py    a page the build order has specified and not yet built
  draft_room.py   the live draft room — `make draft-room`. Its own app, not a page of
                  app.py, and the one file that imports src/ (see above)
  pca.py          the fingerprint view's pure layer — orientation, SD scaling, loadings,
                  neighbours
  charts.py       fig_radar / fig_loadings
  theme.py        SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  artifacts.py    load_cfg, features_dir, read_table, optional
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
carrying its title, icon and `url_path`. The first row is the default page, i.e. what `/`
serves. **`st.navigation` renders nothing at all for a single-page app** — measured, not
assumed — so the shell keeps at least two rows; `views/placeholder.py` fills the gap with
the next page in the build order.

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
  highlight-and-gray past that. The radial chart draws at most two series for this reason.
- Three light-mode slots fall below 3:1 contrast on the light surface, which obliges the
  relief rule: **no value is reachable by colour alone.** The radial chart is read off a
  labelled axis with rings at −2/−1/0/+1/+2 SD and ships a table twin; every loadings panel
  ships one too.

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
   negative half of an axis. **The live page in a real browser** proves the page is, and
   is the only layer that can: it caught a plotly `title_font` with no text rendering as
   the literal string "undefined", metric tiles clipping their own values, and a click
   handler that never fired.

## Four things plotly and Streamlit do that cost a day

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
  `background: rgb(252, 252, 251)` and `background: rgb(26, 26, 25)`.
