# The dashboard

A nine-tab walkthrough of the project, for a reader who wants a birds-eye view of the
decisions rather than an EDA explorer. Run it with `make dashboard`.

## Precedence — read this before trusting a number on the page

> `CLAUDE.md` and the `docs/*-plan.md` files are the source of truth for every claim on this
> dashboard. The registry in `decisions.py` is a **distillation** of them for browsing, not an
> authority. Where the two disagree, the docs are right and the registry is stale — fix the
> registry.

Drift is the main risk in this package, and it gets four mitigations: this rule, the standing
instruction in `CLAUDE.md`'s Conventions section, `make dashboard-audit`, and a weekly job that
appends the audit to `outputs/dashboard_audit.log`.

## The provenance rule

> **Every figure on the dashboard is read from an artifact that a `make` target produced.**
> Where no such artifact exists yet, the figure does not go on the page until one does — see
> `docs/provenance-plan.md`. The only exception is an `incident` entry, which carries a date
> and a doc reference instead of a number.

A panel that needs an unbacked figure renders `layout.pending_marker(...)` naming the target
that will supply it, rather than the typed number. `make dashboard-audit` counts the markers,
so the count trends to zero. It is **zero today** — all ten items in `docs/provenance-plan.md`
landed on 2026-07-29.

## Layout

```
dashboard/
  README.md       this file
  app.py          page config, sidebar, tab dispatch — thin
  theme.py        SERIES, THEMES, ALL_PAIRS_CAP, theme(), apply_theme(), ordinal_colors()
  charts.py       fig_heatmap / fig_bars / fig_lines / fig_scatter
  layout.py       table_view, note, stat_tiles, decision_card, status_badge, pending_marker
  artifacts.py    load_cfg, features_dir, eda_dir, read_table, read_pickle, optional, inventory
  decisions.py    the registry — pure data, no streamlit import
  economics.py    tournament rake / hurdle / advance-rate derivations — pure, no streamlit
  audit.py        make dashboard-audit — pure logic + a __main__ block
  tabs/           one render(ctx) per tab, in project order
```

Three constraints hold this shape:

- **`decisions.py`, `economics.py` and `audit.py` must not import Streamlit.** They hold the
  only new logic worth testing, and keeping them pure means `tests/test_dashboard.py` exercises
  them directly rather than through an `importlib` file-loading trick.
- **The dashboard reads artifacts and nothing else.** There is no import from `src/` anywhere
  in the package. Dropping tab 3's `season_pairs()` helper — the single such import — turned
  that from a convention into an invariant, and a test pins it.
- Each tab exposes one `render(ctx)`, where `ctx` carries the theme dict, tier, era mode and
  resolved artifact paths. One context object stops nine signatures from growing nine ways.

## Colour

`theme.py` is the validated reference palette instance, used **unmodified**. Two rules from that
validation constrain every chart, and neither is a style preference:

- The eight categorical slots clear the colour-blind gates on *adjacent* pairs (bars, lines,
  stacks), but only the **first three** clear them on *all* pairs. Any scatter, and any chart
  where non-adjacent series sit side by side, caps at `ALL_PAIRS_CAP = 3` and uses
  highlight-and-gray past that.
- Three light-mode slots fall below 3:1 contrast on the light surface, which obliges the relief
  rule: **every chart ships a table-view twin in an expander**, so no value is reachable by
  colour alone.

Sequential is a single blue hue; diverging is blue↔red with a **neutral gray** midpoint, never a
rainbow. Plot surfaces are pinned to the exact surfaces the palette was validated against
(`#fcfcfb` / `#1a1a19`) rather than inherited from Streamlit's chrome, so the measured contrast
figures apply as documented.

## `make dashboard-audit`

Four checks, run as `python -m dashboard.audit`:

| check | catches |
|---|---|
| every `reproduce` artifact exists | an entry citing a target that was renamed or never built |
| no `source` doc has a commit newer than the entry's `reviewed` date | the docs-folder sweep the drift risk needs |
| artifacts on disk that no tab and no entry references | the inverse check — orphaned families accumulate silently |
| pending provenance markers remaining | a rising count is a regression |

It is a **report, not a gate** — it exits 0 with findings, because failing the suite when
somebody edits a doc trains people to ignore the suite. Only the first check is also a `pytest`
test, so the anti-drift guard cannot rot.

## Adding a decision

Add the `Decision` to `decisions.py` alongside the `CLAUDE.md` / plan-doc edit, with `source`
naming the doc it was distilled from and `reviewed` set to the date you checked it. Statuses
come from a closed vocabulary (`built`, `settled`, `measured`, `null`, `withdrawn`, `open`,
`blocked`, `deadline`, `incident`). A reversal becomes `withdrawn` and **keeps its entry** —
deleting it throws away the most useful row on the decision-log tab.

`null` is for nulls that are *measurements* and therefore carry an artifact. A dated diagnosis
of an external system is an `incident`: it renders without a live-number claim, because
re-deriving it would mean re-probing a third party to no purpose. That boundary is
`docs/provenance-plan.md`'s figures-versus-incidents line.
