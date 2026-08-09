
**`make docs-audit` is the guard against prose drifting away from its artifact**, which has
happened twice, both times silently and both times in a table that was *partially* refreshed:
the season-total R² column, and the report-calibration block. `src/docs_audit.py` holds one
`Claim` per quoted figure and runs three checks — the figure equals its artifact **to the
precision it is quoted at** (`22.7` to ±0.05, `0.0635` to ±0.00005); the quoted string still
appears in the doc, so a claim cannot rot into describing nothing; and coverage, so "how much
of this doc is audited" is a number. Unlike `dashboard-audit` this one **exits non-zero** —
a doc contradicting its artifact is a defect, not a preference — and it is also a `pytest`
test. Missing artifacts are *skipped*, so a fresh checkout without `make eda` is clean.
**Rebuilding an artifact will fail it until the docs are updated. That is the point.**
It guards the artifact→prose direction tightly and the prose→artifact direction loosely;
see the module docstring for exactly what it cannot catch.

**It covers twelve docs with 2,360 claims and one builder per doc**: `README.md`,
`availability-plan`, `minutes-composition-plan`, `predictions-plan`, `games-played-plan`,
`shot-attempt-basis-plan`, `adp-plan`, and the five files the 2026-08-08 reorganization
split `CLAUDE.md` into — `facts-archive`, `model-development-notes`, `data-quirks`,
`project-spec` and `train-validate-test-split`.
Coverage of measured figures: 79% (README), 79% (shot-attempt basis), 67% (model
development notes), 64% (composition), 63% (predictions), 60% (adp), 58% (availability),
46% (games played), 40% (facts archive), 33% (project spec), 26% (train/validate/test),
21% (data quirks) — `make docs-audit` prints them live, so treat the
printout rather than this line as current. The uncovered remainder is prose-only figures
(`docs/provenance-plan.md` lists all fourteen), costing estimates, and counts of things
rather than measurements.
**668 of the claims are superseded values held for the record**, which is the number that
grows fastest as heads move off the test split: each conversion retires a measurement
without deleting it. The season-term conversion alone added **161** — the largest single
jump so far, because that ablation quotes four arms across thirteen heads and five
downstream tables, so a change of split retires figures by the table rather than by the
line.

**`_established_facts` is the one builder that is not one-doc-one-function**, because the
established-facts section it claims was split across five docs *by subject* rather than
moved as a block. It names a destination per section with `into(...)`, overrides that
per claim where an individual figure landed elsewhere, and
`test_the_established_facts_builder_spans_exactly_its_declared_docs` pins the resulting
set — so a mistyped destination fails a test rather than surfacing as a stale claim
nobody reads. Nothing claims `CLAUDE.md` any more: it is a router carrying no
measurements, so there is nothing in it to drift.

**`README.md` was added last and is the doc the guard fits best**, which is why its coverage
is among the highest: it holds no measurements of its own, only a selection of headlines
copied from the established facts and the plan docs, and it is the most-read and least-maintained file in the
repo — the exact conditions under which a figure goes stale unnoticed. Two things its builder
does that the others do not: it claims **roundings** (`58%` against 57.96%, `86%` against an
R² of 0.859), because an overview should round and `implied_tolerance` already handles that
correctly; and it claims **shipped constants against their fitted optima** (the bonus
overdispersions 0.10 and 0.025 against `bonus_calibration.csv`'s `analysis == "fitted"`
rows), so a re-calibration that moves an optimum away from the constant fails here rather
than passing silently. The tournament break-even hurdles have no `outputs/` artifact —
`dashboard/economics.py` derives them at render time — so they are claimed against the
checked-in raw boards, with `_break_even_hurdle` duplicating one line of `economics.py`
rather than importing it, since nothing in `src/` imports the dashboard package. A test pins
the two copies together.

**A block quoted in two docs is claimed from both against the one artifact**, because
"current in one doc and stale in the other" is the failure that has already happened twice
here (the season-total R² column, the report-calibration block). `_regime_claims` and
`_season_term_claims` / `_season_term_summary_claims` are shared builders for exactly that
reason — `docs/facts-archive.md` carries a summary of the season-term verdict and the plan
doc carries it in full, so the summary claims the subset it quotes rather than being forced
to carry every cell. The availability ladder is now quoted in four docs and claimed from
all four, at three different scopes.

**Some quoted figures must NOT agree with the artifact, and `Claim(historical=True)` is how
they survive.** Two kinds: a superseded value preserved beside its correction ("corrected
2026-07-30 from 0.664 / 0.838 / 0.922"), and a scratch-session measurement kept beside the
promoted one — `docs/adp-plan.md` is built on the second, quoting ρ **0.8704** on 218 pairs in
its planning section and **0.8675** on 226 in its implementation section, *both correct*. A
historical claim is excluded from the value check and still presence-checked, so the failure
mode it guards is **deletion**, not drift. There are 56 of them. Without the flag the only
options are to "correct" a reversal out of existence or to leave it unprotected.

**Extending it to the three new docs found drift in all three**, which is the argument for
having built it: the serial-correlation table in `predictions-plan.md` (twelve rows, refreshed
in `CLAUDE.md` and not here), the roster-coverage and residual-correlation figures corrected in
two other files and missed here, the ADP position offset (−0.3 → **−2.0**, a real change on the
larger matched set), and — the second occurrence of the exact failure named above — the
**report-calibration block, stale in `CLAUDE.md` while current in `availability-plan.md`**. It
is now claimed from both docs against the one artifact so that cannot recur.

**The dashboard reads artifacts and nothing else — that is an invariant, not a
convention.** It never refits, and there is **no import from `src/`** anywhere in the
package; a test walks it with `ast` and fails if one appears. Since 2026-08-08 it is a
visualization surface rather than a project walkthrough — one view, the PCA player-style
fingerprint — so the claims it used to render live only in the docs and in
`dashboard/decisions.py`. `.streamlit/config.toml` sets `headless = true`, without which Streamlit's first-run email
prompt makes `make dashboard` exit 255 instead of serving.

**Every figure on the dashboard is read from an artifact a `make` target produced.** Where
none exists, the panel renders `layout.pending_marker(...)` naming the target rather than a
typed number, and `make dashboard-audit` counts the markers. It is **0** today, because all
ten items in `docs/provenance-plan.md` landed first. The only exception to the rule is an
`incident` entry, which carries a date and a doc reference instead of a number.

`make dashboard-audit` runs four checks — every cited artifact exists (also a `pytest`
test), no source doc has a git commit newer than an entry's `reviewed` date, no artifact on
disk goes unreferenced by both tabs and registry, and no pending markers remain. A weekly
launchd job (`com.nba-deep-learning.dashboard-audit`, Mondays 09:00) appends it to
`outputs/dashboard_audit.log`. **The orphan check is the one that earns its keep**: nine
artifact families had accumulated unreachable from the dashboard purely because nothing was
looking, and removing a tab silently re-creates that — so bring it back to zero deliberately,
either by rendering the family or by naming it in a registry entry with its make target.

**Verify dashboard changes with Streamlit's `AppTest`, not with `curl`.** A request to port
8501 returns 200 from the HTML shell even when the script raises on every tab; `AppTest`
executes `app.py` for real, and because `st.tabs` renders all its children, an exception in
any tab surfaces.
