"""Tab 3 · Exploratory data analysis — narrative only in this pass.

The eight EDA figure sections the pre-split app carried are **not** ported. The PCA /
archetype line is not currently feeding the pipeline, and porting charts nothing
consumes would have been the largest block of work in the revamp for the least
return. Later, the subset of visuals that turn out to matter comes back — chosen
then, on evidence, rather than inherited wholesale now. Commit `e1a24c4` holds the
renderers verbatim if a section is wanted back.

`charts.py` keeps every figure builder, so that is a one-file change rather than a
rewrite. One side effect worth naming: `season_pairs()` was the dashboard's only
import from `src/`, so dropping it made **"the dashboard reads artifacts and nothing
else"** an invariant rather than a convention, and a test pins it.

What the tab does carry is the EDA findings that changed the build, as decision
cards, each linked to the artifact and make target behind it.
"""

import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx, optional
from dashboard.charts import fig_bars
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "eda"

# The findings that changed the build, in the order they are worth reading. Every
# other `eda` entry still renders below, under "the rest".
HEADLINE = [
    "absorb-season-always",
    "minutes-weight-per36-rates",
    "share-versus-conversion",
    "archetypes-partition-a-continuum",
    "feature-matrix-is-singular",
    "aging-lives-in-availability",
]


def render(ctx: Ctx) -> None:
    tab_header(
        "Exploratory data analysis",
        "Narrative in this pass. The season-level sweep produced 40-odd artifacts and "
        "a handful of findings that actually changed how the model is built; those "
        "are below as decision cards, each linked to the artifact and make target "
        "behind it.")

    st.info(
        ":material/info: **The figure sections are deliberately deferred, not lost.** "
        "The PCA / archetype line is not currently feeding the pipeline, so porting "
        "its charts would have been the largest block of work in this revamp for the "
        "least return. `charts.py` keeps every figure builder and commit `e1a24c4` "
        "holds the eight renderers verbatim, so bringing a section back is a one-file "
        "change. `make dashboard-audit`'s orphaned-artifact check is the standing "
        "nag: anything the sweep writes that no tab reads and no decision names shows "
        "up in it every week.")

    _scale(ctx)

    st.markdown("---")
    st.markdown("### The six findings that changed the build")
    by_id = {d.id: d for d in D.REGISTRY}
    decision_cards([by_id[i] for i in HEADLINE if i in by_id])

    st.markdown("---")
    st.markdown("### The rest of the EDA record")
    note("Method rules, feature verdicts and recorded nulls. The nulls matter as much "
         "as the findings — they are what stops a dead end from being rebuilt.")
    decision_cards([d for d in D.by_topic(TOPIC) if d.id not in set(HEADLINE)])


# ── What the sweep actually produced ──────────────────────────────────────────

def _scale(ctx: Ctx) -> None:
    st.markdown("### What the sweep produced")

    cov = optional(ctx.features("coverage_report.csv"), target="make season-matrix")
    if cov is None:
        return

    scoped = cov[cov["tier"].isin(["A", "B"] if ctx.tier == "B" else ["A"])]
    stat_tiles([
        ("Seasons covered", f"{cov['season'].nunique()}",
         "Tier A is the 30-season box-score frame; Tier B adds tracking and hustle "
         "from 2013-14 and is a strict column superset."),
        ("Stat families", f"{cov['family'].nunique()}",
         "Fetched per season and joined into one row per (player, season)."),
        ("Families in tier scope", f"{scoped['family'].nunique()}",
         f"Under the sidebar's tier {ctx.tier} selection."),
        ("Rows in the coverage report", f"{len(cov):,}",
         "One per family × season × tier, with how many players matched."),
    ])

    per_family = (scoped.groupby("family")["season"].nunique()
                  .rename("seasons").reset_index().sort_values("seasons"))
    st.plotly_chart(
        fig_bars(per_family, "family", ["seasons"], ctx.th,
                 f"Seasons available per stat family (tier {ctx.tier} scope)",
                 axis_title="seasons", height=460),
        width="stretch")
    note("The four coverage boundaries are visible directly: core box-score families "
         "run the full window, tracking and `pt_shot` start in 2013-14, estimated in "
         "2014-15 and hustle in 2015-16. The tier selector in the sidebar scopes this "
         "panel and the coverage heatmap on **Data collection** — nothing else on the "
         "dashboard has a tier, because each head is fitted on one frame.")
    with detail("Coverage — every family × season"):
        st.dataframe(cov.sort_values(["season", "family"]), width="stretch",
                     hide_index=True)
    table_view(per_family, "Families by season — table view")
    provenance("`make season-matrix` → `data/features/coverage_report.csv`")
