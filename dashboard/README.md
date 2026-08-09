# The dashboard

Data visualizations over the artifacts the pipeline wrote. One view today — the PCA
player-style fingerprint. Run it with `make dashboard`; the plan is
`docs/dashboard-plan.md`.

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

Where a view *interprets* an artifact — naming a principal component, say — the
interpretation carries a machine-checkable anchor so it cannot silently invert. See
`pca.COMPONENTS` and `pca.orient()`.

## Layout

```
dashboard/
  README.md       this file
  app.py          the PCA fingerprint view: page, controls, layout
  pca.py          its pure layer — orientation, SD scaling, loadings, neighbours
  charts.py       fig_radar / fig_loadings
  theme.py        SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  artifacts.py    load_cfg, features_dir, read_table, optional
  decisions.py    ─┐
  economics.py     ├ not the dashboard — see below
  audit.py        ─┘
```

`app.py` and `artifacts.py` are the only modules that import Streamlit. Everything else is
pure, which is what lets `tests/test_dashboard.py` exercise the palette rules, the component
spec, the scaling, the neighbour metric and both figures as plain functions rather than
through a rendered page.

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
4. Verify with a Streamlit `AppTest` in both appearance modes, **and look at the rendered
   figure.** `AppTest` proves the page runs; it cannot prove the page is legible. Rendering
   view 1's charts to PNG is what caught a loadings panel that silently dropped the negative
   half of an axis.
