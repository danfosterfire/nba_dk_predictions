"""The model detail page, once — seven blocks over any head the emitter carded.

`docs/dashboard-plan.md` specifies four model pages (availability, minutes, box-score
components, game length) that differ only in which heads sit behind their selector. This
module is that page; a view under `views/` is one call into `render()` with a class key,
and `dashboard/model_cards.py` holds the class table and every frame shape below.

**The selector carries only heads in the simulator's draw path** (since 2026-08-23): a
model page discusses the heads a simulated season is assembled from, and a head that is
fitted, converged and carded but never read at draw time — the games-played tenure
decomposition, the marginal minutes head — stays carded and off the page. `heads_of`
reads the artifact's own `in_draw_path` for the cut, so the pages follow
`src/sim/season.py` through `make model-cards` rather than restating it. The marginal
minutes head's one draw-time role — calibrating the injected per-(player, season) σ — is
noted on the *Inputs beyond the heads* page, where that constant is inventoried.

The seven blocks, in the order the plan fixes them:

1. **What this head is** — its declared specification, its **unit**, and its **role in the
   shipped chain**, stated at the top of the page. The one block where typed prose is
   allowed, and only ever about the specification: the component heads are season-collapsed
   player-seasons while the composition is per player-game, and a reader comparing an R²
   across those pages without knowing that is being misled. The chain role is the same
   argument one question on — a head being fitted, converged and carded says nothing about
   whether `make simulate-season` calls it, and four of the twenty are never read at draw
   time — so it too is read from `model_card_index.csv` rather than typed here.
2. **The features it was fed** — one histogram per design column with train and validation
   on shared edges, plus the n / mean / sd / imputed-share table.
3. **Feature relationships** — the correlation heatmap, and one joint density for a pair
   the reader picks. **Not a pair plot**: 12–20 features is 150–400 panels
   (`feature-correlation-not-pair-plots`).
4. **Coefficients** — posterior means with 95% credible intervals, sorted, spline bases
   grouped under their `term_family`.
5. **Predictive calibration** — the observed ECDF over the posterior-predictive ribbon,
   train beside validation, read as a *distance* rather than as a verdict.
6. **Predicted against observed, and the scaled quantile residual** — the binned density
   with its bounded sample overlaid, then DHARMa's residual: a QQ-uniform with a pointwise
   envelope, and the residual against **rank-transformed** predicted with its quartile
   lines. The raw residual-against-predicted panel this replaced was not readable across
   the four classes, since a negative binomial's residual on a season total and a
   beta-geometric's on a spell length are not on one scale. A quantile residual is uniform
   iff calibrated whatever the likelihood, which is what one shared renderer needs.
7. **Diagnostics** — the two sampler runs behind the card, and the four build-time checks
   `make model-cards` had to pass before it was allowed to write it.

Every number on the page is read from an artifact `make model-cards`, `make posteriors` or
`make stan` wrote. Nothing here imports `src/`, and nothing here computes a model quantity:
the three derived readings on the page — the ribbon's distance from its median, the binned
panel means, and how far a quantile line sits from its own level — are computed in
`model_cards.py`, from the same cells the figure beside them draws. The KS distance itself
is *not* one of them: it is a model quantity, so `make model-cards` computes it through
`stan_utils.ks_uniform` and this page reads it off the artifact.
"""

import numpy as np
import pandas as pd
import streamlit as st

from dashboard import model_cards as mc
from dashboard import shell
from dashboard.artifacts import ROOT, load_cfg, optional, predictions_dir, rel
from dashboard.charts import (fig_calibration, fig_coefficients, fig_correlation,
                              fig_ecdf, fig_features, fig_joint, fig_qq,
                              fig_quantile_residual)

#: Features past this many are behind an expander rather than drawn on arrival. The
#: composition's 25 columns are seven rows of small multiples, which is a lot of figure to
#: put between a reader and block 3 on every rerun.
FEATURE_GRID_LIMIT = 8


# ── Loading ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Reading the model cards…")
def load_cards() -> dict[str, pd.DataFrame] | None:
    """The ten model-card artifacts, or `None` once a missing one has been named."""
    directory = predictions_dir()
    frames = {}
    for key, name in (("index", mc.INDEX_FILE),
                      ("coefficients", mc.COEFFICIENTS_FILE),
                      ("features", mc.FEATURES_FILE),
                      ("correlations", mc.CORRELATION_FILE),
                      ("density", mc.DENSITY_FILE),
                      ("ecdf", mc.ECDF_FILE),
                      ("calibration", mc.CALIBRATION_FILE),
                      ("quantile", mc.QUANTILE_FILE),
                      ("sample", mc.SAMPLE_FILE),
                      ("stan", mc.STAN_FILE)):
        frame = optional(directory / name, target=mc.MAKE_CARDS)
        if frame is None:
            return None
        frames[key] = frame
    return frames


@st.cache_data(show_spinner=False)
def load_diagnostics(filename: str) -> pd.DataFrame | None:
    """One `make stan` diagnostics table, or `None` — block 7 degrades, it does not stop."""
    path = predictions_dir() / filename
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_manifest() -> pd.DataFrame | None:
    """`make posteriors`'s own manifest — a flat CSV beside the pickles, never a pickle.

    The pickles in that directory are the one thing this package may not open: unpickling
    one imports `src.models.posteriors` and hands back a fitted scaler and the ordered
    design steps, i.e. the capability to score a frame. The manifest is a CSV of what each
    fit cost and converged to, and reading it needs none of that.
    """
    cfg = load_cfg()
    path = (ROOT / cfg["data"]["features_dir"] / "posteriors" / mc.POSTERIOR_WINDOW
            / mc.MANIFEST_FILE)
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── Block 1 · what this head is ───────────────────────────────────────────────

def specification_block(row: pd.Series) -> None:
    tiles = [
        ("Unit", str(row["unit"]), "One row of this head's fit. Every figure below is at "
                                  "this unit, and an R² read across two pages at "
                                  "different units is not a comparison"),
        ("Fitting rows", f"{int(row['n_fit']):,}", "Rows the head actually fitted"),
        ("Features", f"{int(row['n_features'])}", "Design columns, post-transform"),
        ("Likelihood", str(row["likelihood"]), "The observation model"),
        ("Posterior draws", f"{int(row['n_draws']):,}",
         "Thinned draws every block below is cut from"),
    ]
    for col, (label, value, helptext) in zip(st.columns(len(tiles)), tiles):
        col.metric(label, value, help=helptext)

    st.markdown(f"**{row['description']}**")
    # What the head does when a season is drawn, beside what it *is*. Read from the index
    # rather than typed: sixteen of the twenty heads are loaded by `src/sim/season.py` and
    # four are not, and a page that decided which for itself would go stale the first time
    # the simulator was refactored — the same argument `unit` above already carries.
    role = mc.chain_role_phrase(row)
    if role:
        st.caption(role)
    # Full width rather than half: the `Note` column is the half of this table worth
    # reading, and in a half-width column Streamlit truncated every one of them.
    st.dataframe(mc.specification(row), hide_index=True, width="stretch",
                 column_config={"Note": st.column_config.TextColumn(width="large")})

    left, right = st.columns(2, gap="large")
    # `last_season` is absent on a head that is not fitted over seasons at all — the
    # game-length depth head is fitted on four depth cells — and an f-string over a missing
    # CSV cell prints the literal `nan`, which is the same class of defect as `undefined`.
    last = mc.text(row["last_season"], "")
    left.caption(
        f"**The response is `{row['response_label']}`,** and the head's own selected arm "
        f"is `{row['variant']}` — chosen on the validation split by its own variant "
        f"ladder, never on the test seasons"
        + (f", which end after `{last}` and are" if last else ", which are")
        + " not materialized anywhere in this file.")
    right.caption(
        f"Coefficients are on the **{row['coefficient_scale']} design scale**: every "
        f"column is centred and scaled before fitting, which is what makes block 4 a "
        f"comparison between terms rather than a plot of measurement units. The scaler's "
        f"centre and scale ride along per term in the artifact, so a reader can "
        f"unstandardize without the fitted object.")


# ── Block 2 · the features it was fed ─────────────────────────────────────────

def features_block(cards: dict, head: str, th: dict) -> None:
    features = cards["features"]
    order = mc.feature_order(features, head)
    if not order:
        st.info("This head carries no design columns — it fits an intercept and a "
                "dispersion only, so there is nothing to draw here.")
        return

    summary = mc.feature_summary(features, head)
    panel = mc.histogram_panel(features, head, order)
    imputed = mc.imputed_shares(summary)

    st.caption(
        f"**{len(order)} design column{'s' if len(order) != 1 else ''}**, taken from the "
        f"head's own variant ladder "
        f"post-transform and pre-standardization — a spline basis and an imputation flag "
        f"are features here because they are what the head was actually fed. Train and "
        f"validation share one edge set per column, computed on the pooled values, and "
        f"each bar is the **share** of its own split so the two are comparable at "
        f"different row counts.")

    grid = fig_features(panel, th, title="")
    if len(order) > FEATURE_GRID_LIMIT:
        with st.expander(f"The {len(order)} feature histograms", expanded=True):
            st.plotly_chart(grid, width="stretch", key=f"features-{head}",
                            config={"displayModeBar": False})
    else:
        st.plotly_chart(grid, width="stretch", key=f"features-{head}",
                        config={"displayModeBar": False})

    if imputed:
        st.caption(
            "**Imputed shares, as their own rows:** " + imputed
            + ". An imputation flag is never itself missing — its *mean* is the share "
              "being asked about — and a spline basis inherits the share of the raw "
              "column it descends from rather than reporting zero.")
    else:
        st.caption("**Nothing in this block was imputed:** every design column was "
                   "observed on every row of both splits.")

    with st.expander("Table view — n, mean, sd and imputed share per feature"):
        st.dataframe(summary, hide_index=True, width="stretch")


# ── Block 3 · feature relationships ───────────────────────────────────────────

def relationships_block(cards: dict, head: str, th: dict) -> None:
    correlations = cards["correlations"]
    square = mc.correlation_square(correlations, head, "train")
    if square.empty:
        st.info("This head has no feature block, so there is nothing to correlate.")
        return
    if len(square) < 2:
        # A correlation needs a pair. The game-length onset head's whole design is one
        # season term, and drawing its 1 × 1 square would put a heatmap of a single r = 1
        # cell on the page — worse than nothing, because it looks like a measurement.
        st.info(f"This head has one design column — `{square.index[0]}` — so there is no "
                f"pair to correlate and no joint to draw. The block needs two.")
        return

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.plotly_chart(fig_correlation(square, th, title="Training split"),
                        width="stretch", key=f"corr-{head}",
                        config={"displayModeBar": False})
        constant = mc.constant_features(correlations, head, "validation")
        st.caption(
            "Pearson, pairwise-complete, **the whole square including the diagonal** — so "
            "this is a reshape of the artifact rather than a reconstruction of it, and a "
            "column that is constant on a split stays on the axis as a gap instead of "
            "disappearing from it."
            + (f" On the validation split, {len(constant)} column"
               f"{'s are' if len(constant) != 1 else ' is'} identically constant and "
               f"correlates with nothing: " + ", ".join(f"`{c}`" for c in constant) + "."
               if constant else ""))

    menu = mc.pair_menu(correlations, head)
    with right:
        if menu.empty:
            st.info("No feature pair is correlated strongly enough to have earned a "
                    "precomputed density.")
            return
        options = list(range(len(menu)))
        choice = st.selectbox(
            "Joint density for one pair", options,
            format_func=lambda i: mc.pair_label(menu.iloc[i]),
            key=f"pair-{head}",
            help="The most correlated pairs, precomputed. A full pair-plot matrix over "
                 "this block would be 150–400 panels; the heatmap says which pair is "
                 "worth opening and this draws that one.")
        pair = menu.iloc[choice]
        split = st.radio("Split", list(mc.SPLITS), horizontal=True,
                         format_func=lambda s: mc.SPLIT_LABELS[s], key=f"pair-split-{head}")
        cells = mc.density_panel(cards["density"], head, str(pair["feature_x"]),
                                 str(pair["feature_y"]), split)
        if cells.empty:
            st.info("No rows land in this pair on the selected split.")
            return
        st.plotly_chart(
            fig_joint(cells, th, str(pair["feature_x"]), str(pair["feature_y"]),
                      title=f"r = {float(cells['r'].iloc[0]):+.3f} over "
                            f"{int(cells['n'].iloc[0]):,} rows"),
            width="stretch", key=f"joint-{head}", config={"displayModeBar": False})
        st.caption(
            "Both splits are binned on one shared grid, so flipping the split changes the "
            "picture and not the axes. The pair menu is ranked on the training split for "
            "the same reason.")

    with st.expander("Table view — the correlation matrix"):
        st.dataframe(square.round(3), width="stretch")


# ── The Stan program — a named block, between blocks 3 and 4 ──────────────────

def stan_block(cards: dict, row: pd.Series, th: dict) -> None:
    """The head's own Stan program, verbatim, keyed on block 3 so it lands before the
    coefficients it defines.

    Named rather than numbered, like every page-owned block, and shared by the pages that
    ask for it rather than built into the sequence — the numbered blocks are the contract
    every model page keeps. The source is read from `model_card_stan.csv`, the emitter's
    snapshot of `src/stan/`, so the code shown is the code the cards ship with and the
    page still reads artifacts only. Code is the one thing on these pages that is neither
    a figure nor a result: it is the specification itself, which is what block 1's prose
    is allowed to describe and this block simply shows.
    """
    head = str(row["head"])
    program = mc.stan_row(cards["stan"], head)
    if program is None:
        # An older artifact loses the block rather than raising — the same degradation
        # `chain_role_phrase` chooses when its column is absent.
        return

    st.markdown("---")
    st.subheader("The Stan program")
    siblings = mc.stan_siblings(cards["stan"], cards["index"], head)
    st.caption(
        f"**`{program['stan_file']}`, verbatim** — the source this head's likelihood is "
        f"compiled from, snapshotted into `{mc.STAN_FILE}` by `{mc.MAKE_CARDS}` beside "
        f"every card above. "
        + (f"The same program serves {len(siblings)} other carded head"
           f"{'s' if len(siblings) != 1 else ''} "
           f"({', '.join(siblings)}) with different data — optional blocks are switched "
           f"by the data it is handed, so a zero-length block is disabled exactly rather "
           f"than approximately."
           if siblings else
           "No other carded head compiles from this program."))
    with st.expander(f"`{program['stan_file']}` · {int(program['n_lines']):,} lines",
                     expanded=False):
        st.code(str(program["source"]), language=None)

def coefficients_block(cards: dict, head: str, th: dict) -> None:
    coefficients = cards["coefficients"]
    scalars = mc.scalar_terms(coefficients, head)
    families = coefficients[(coefficients["head"] == head)
                            & (~coefficients["term_role"].isin(mc.SCALAR_ROLES))]
    has_bases = bool((families["basis_index"] >= 0).any())

    collapse = False
    if has_bases:
        collapse = st.toggle(
            "Collapse spline bases to one row per feature", value=False,
            key=f"collapse-{head}",
            help="Nine heads in this project carry a six-column basis over one underlying "
                 "quantity. Collapsed, each family keeps its widest basis and says so.")
    panel = mc.coefficient_panel(coefficients, head, collapse=collapse)
    if panel.empty:
        st.info("This head fits no slope terms — only an intercept and a dispersion.")
    else:
        st.plotly_chart(fig_coefficients(panel, th, title=""), width="stretch",
                        key=f"coef-{head}", config={"displayModeBar": False})

    if len(scalars):
        tiles = [(str(r["term"]), f"{float(r['mean']):+.4f}",
                  f"95% interval [{float(r['q2.5']):+.4f}, {float(r['q97.5']):+.4f}] · "
                  f"{r['term_role']}")
                 for _, r in scalars.iterrows()]
        for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 4)), tiles):
            col.metric(label, value, help=helptext)

    st.caption(
        "**The intercept and the dispersion are tiled rather than drawn.** They are not on "
        "the standardized slope scale the bars share, and letting the intercept into the "
        "panel would set the axis and flatten every term the panel exists to compare. "
        + ("Spline bases are grouped under their family and ordered by basis index, so a "
           "six-column basis reads as one feature's shape rather than as six unrelated "
           "bars. " if has_bases else "")
        + "`P(> 0)` in the table is the share of draws above zero, which a 95% interval "
          "crossing zero under-reports: a term at 0.94 and a term at 0.50 both cross it "
          "and are not the same claim.")

    if not panel.empty:
        with st.expander("Table view — every term, with its interval"):
            st.dataframe(mc.coefficient_table(panel), hide_index=True, width="stretch")


# ── Block 5 · predictive calibration ──────────────────────────────────────────

def calibration_block(cards: dict, row: pd.Series, head: str, th: dict) -> None:
    panels = {mc.SPLIT_LABELS[split]: mc.ecdf_panel(cards["ecdf"], head, split)
              for split in mc.SPLITS}
    panels = {name: part for name, part in panels.items() if not part.empty}
    if not panels:
        st.info("No predictive ribbon was cut for this head.")
        return

    distance = mc.band_distance(cards["ecdf"], head)
    tiles = []
    for _, part in distance.iterrows():
        tiles.append((f"Largest gap · {part['label'].lower()}", f"{part['max_gap']:.3f}",
                      f"The furthest the observed ECDF sits from the median replicate, in "
                      f"ECDF units, at {part['value_at_max']:,.3g} "
                      f"{row['response_label']}"))
    tiles.append(("Inside the 95% band",
                  " / ".join(f"{p['inside_95']:.0%}" for _, p in distance.iterrows()),
                  "Share of grid points where the observed curve is inside the widest "
                  "band, train / validation. Not the reading — see the caption"))
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(
        fig_ecdf(panels, th, str(row["response_label"]), title=""),
        width="stretch", key=f"ecdf-{head}", config={"displayModeBar": False})

    st.caption(
        f"**Read the size of the miss, not in-or-out.** One replicate dataset is drawn per "
        f"posterior draw and its ECDF recorded, so the band is the spread of *datasets* "
        f"the model thinks it could have produced — at these sample sizes it is one to two "
        f"ECDF points wide and every head in this project leaves it somewhere. The useful "
        f"statistic is the largest vertical distance from the median replicate, tiled "
        f"above. {mc.predictive_provenance(row)}")
    st.caption(
        "The grid is quantiles of the **observed**, not of the draws: an over-wide "
        "predictive then shows up as a ribbon that has not reached 1 at the last grid "
        "point, where a grid stretched to cover the draws would hide that in the axis.")

    with st.expander("Table view — the distance, per split"):
        st.dataframe(
            distance.rename(columns={
                "label": "Split", "max_gap": "Largest gap from median",
                "mean_gap": "Mean gap", "inside_95": "Inside 95% band",
                "n_grid": "Grid points", "n_rows": "Rows", "n_draws": "Draws",
                "value_at_max": "Value at the largest gap"})
            .drop(columns="split")
            .round({"Largest gap from median": 4, "Mean gap": 4, "Inside 95% band": 3,
                    "Value at the largest gap": 3}),
            hide_index=True, width="stretch")


# ── Block 6 · predicted against observed, and the scaled quantile residual ────

def residuals_block(cards: dict, row: pd.Series, head: str, th: dict) -> None:
    cells = {(panel, split): mc.calibration_panel(cards["calibration"], head, panel, split)
             for panel in mc.PANELS for split in mc.SPLITS}
    if all(frame.empty for frame in cells.values()):
        st.info("No calibration density was cut for this head.")
        return
    points = {(panel, split): mc.sample_points(cards["sample"], head, split)
              for panel in mc.PANELS for split in mc.SPLITS}

    st.plotly_chart(
        fig_calibration(cells, points, th, mc.PANEL_AXES, mc.PANEL_LABELS, title=""),
        width="stretch", key=f"calibration-{head}", config={"displayModeBar": False})

    source = ("the head's own reported mean" if row["fitted_source"] == "head_predict"
              else "the mean of its drawn predictive")
    st.caption(
        f"The shading is a 30 × 30 binned density, each panel against its own densest "
        f"cell, with empty cells left transparent; the dots are a bounded subsample laid "
        f"over it for texture. Binned rather than per row because a panel of this head's "
        f"{int(row['n_fit']):,} rows is a copy of the data rather than a picture of it. "
        f"Both splits share one grid **and one axis range**, which is what "
        f"makes them comparable — the range spans the pooled 0.5–99.5% of the values with "
        f"the tails clipped into the end bins, so one heavy-tailed value cannot "
        f"collapse the grid and a handful of overlay points sit outside the view. The "
        f"predicted axis is **{source}** (`fitted_source` in the index says which).")

    with st.expander("Table view — the panel's binned means"):
        st.caption("Weighted off the same cells the panels draw, so the figure and the "
                   "table are two readings of one object. Binned, therefore approximate "
                   "to the width of a cell.")
        st.dataframe(mc.calibration_summary(cards["calibration"], head).round(3),
                     hide_index=True, width="stretch")

    quantile_block(cards, row, head, th)


def quantile_block(cards: dict, row: pd.Series, head: str, th: dict) -> None:
    """The scaled quantile residual — the half of block 6 that used to be a raw residual.

    Two panels, because they answer different questions: the QQ-uniform says whether the
    residual is uniform *overall*, and the residual against rank-transformed predicted says
    whether it is uniform *everywhere along the fit*. A head can pass the first and drift
    badly on the second, which is exactly what a marginal statistic cannot see.
    """
    st.markdown("**Scaled quantile residuals**")
    note = mc.quantile_note(row)
    if note:
        # Declared out of scope by the emitter rather than silently absent. A wrong panel
        # here would be a good-looking uniform cloud, which is worse than no panel.
        st.info(f"This head ships no quantile residual: {note}")
        return

    qq = {mc.SPLIT_LABELS[split]: mc.qq_panel(cards["quantile"], head, split)
          for split in mc.SPLITS}
    qq = {name: part for name, part in qq.items() if not part.empty}
    if not qq:
        st.info("No quantile residual was cut for this head.")
        return

    distance = mc.quantile_distance(cards["quantile"], head)
    tiles = [(f"KS distance · {part['label'].lower()}", f"{part['ks']:.3f}",
              f"How far the {int(part['n']):,} scaled residuals sit from uniform, at their "
              f"furthest point. A distance in probability units — never a pass or a fail")
             for _, part in distance.iterrows()]
    tiles.append(("Furthest quantile line",
                  " / ".join(f"{p['line_gap']:.3f}" if np.isfinite(p["line_gap"]) else "—"
                             for _, p in distance.iterrows()),
                  "The largest gap between a binned 0.25 / 0.5 / 0.75 line and its own "
                  "level, train / validation — where a KS distance says how much, this "
                  "says where"))
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(fig_qq(qq, th, title=""), width="stretch", key=f"qq-{head}",
                    config={"displayModeBar": False})
    st.caption(
        f"**DHARMa's residual, on this project's own draws.** Each row's residual is its "
        f"randomized quantile inside its own {int(row['predictive_draws'])}-draw replicate "
        f"distribution — `below + U·at`, randomized across the probability mass at the "
        f"observed value because the plain quantile of a *discrete* predictive is not "
        f"uniform even under a perfect model, and every response on these pages is "
        f"discrete. Uniform iff calibrated, **whatever the head's likelihood is**, which is "
        f"what makes one panel readable across four model pages where a raw residual is not. "
        f"R's DHARMa simulates at the fitted point estimate; these draws integrate over the "
        f"posterior, so this is a Bayesian PIT residual — the same reading, carrying "
        f"parameter uncertainty rather than conditioning it away.")

    residual = {mc.SPLIT_LABELS[split]: mc.residual_cells(cards["quantile"], head, split)
                for split in mc.SPLITS}
    residual = {name: part for name, part in residual.items() if not part.empty}
    if residual:
        lines = {mc.SPLIT_LABELS[split]: mc.quantile_lines(cards["quantile"], head, split)
                 for split in mc.SPLITS}
        overlay = {mc.SPLIT_LABELS[split]: mc.sample_points(cards["sample"], head, split)
                   for split in mc.SPLITS}
        st.plotly_chart(
            fig_quantile_residual(residual, lines, overlay, th, mc.QUANTILE_LEVELS,
                                  title=""),
            width="stretch", key=f"quantile-{head}", config={"displayModeBar": False})
        st.caption(
            "**Predicted is rank-transformed**, which is what makes this panel comparable "
            "between a count head on a season total and a conversion head on a rate — the "
            "predicted values share no axis and their ranks do. Both axes are then [0, 1] "
            "by construction, so the two splits are on one grid without being put there. "
            "The three lines are the binned 0.25 / 0.5 / 0.75 quantiles of the residual "
            "and are **flat at those levels iff calibrated**; the dashed references are "
            "the levels themselves, and a bin with too few rows for a quartile is a gap "
            "rather than a line drawn through three points. The shading is the *departure* "
            "from an even spread rather than the mass — a calibrated residual fills this "
            "square evenly by construction, so red is where rows pile up and blue is where "
            "they thin out.")

    ungated = ("" if bool(row["quantile_ks_gated"]) else
               ", and is reported rather than gated because this head has too few rows for "
               "the statistic to mean anything")
    st.caption(
        f"**The KS distance is a distance.** At {int(distance['n'].max()):,} rows a strict "
        f"uniformity test rejects every head in this project, so it is tiled as a size and "
        f"never as a verdict — the same rule block 5's ribbon is read under. What *is* "
        f"gated is the draw budget behind it: `{mc.MAKE_CARDS}` re-reads the distance on "
        f"two interleaved halves of the draws and fails the build if the two disagree by "
        f"more than {mc.KS_MC_TOL}, because at {int(row['predictive_draws'])} draws a row "
        f"with no replicate landing on its observed value carries a residual quantized to "
        f"1/{int(row['predictive_draws'])}. This head reads "
        f"{float(row['quantile_ks_mc']):.4f}{ungated}.")

    with st.expander("Table view — the distance, per split"):
        st.dataframe(
            distance.rename(columns={
                "label": "Split", "ks": "KS distance from uniform", "n": "Rows",
                "line_gap": "Furthest quantile line from its level",
                "n_bins": "Bins with enough rows to draw"})
            .drop(columns="split")
            .round({"KS distance from uniform": 4,
                    "Furthest quantile line from its level": 4}),
            hide_index=True, width="stretch")


# ── Block 7 · diagnostics ─────────────────────────────────────────────────────

def diagnostics_block(row: pd.Series, head: str) -> None:
    filename, label = mc.diagnostics_source(row)
    runs = mc.sampler_runs(row, load_manifest(), load_diagnostics(filename))

    st.dataframe(runs, hide_index=True, width="stretch")
    st.caption(
        f"**Two fits of one specification, and the page says which is which.** The "
        f"persisted row is `{mc.MAKE_POSTERIORS}`, which refits the head once at its "
        f"shipped variant and keeps the draws every block above is cut from. The selection "
        f"row is `{mc.MAKE_STAN}`, one arm of the ladder that chose the variant, read from "
        f"`{filename}` under `{label}` — it recorded the full diagnostic block and threw "
        f"its draws away, which is why ESS and treedepth are known for it and not for the "
        f"other, and why only one of them kept any draws. R̂ is known for both and need "
        f"not agree; they are different chains. `{mc.ABSENT}` is a quantity the source "
        f"does not carry, not a zero.")

    st.markdown("**What the build checked before it wrote this card**")
    st.dataframe(mc.build_checks(row), hide_index=True, width="stretch",
                 column_config={
                     "What it catches": st.column_config.TextColumn(width="medium")})
    st.caption(
        f"`{mc.MAKE_CARDS}` fails rather than writing an artifact that disagrees with the "
        f"head it describes, so these are green by construction — which is why the first "
        f"row says `{row['design_check']}`. It reads `vacuous` for the heads whose recipe "
        f"carries no design steps, where that check compares a frame with itself, and a "
        f"green tick that cannot fail is worth less than no tick.")


# ── The page ──────────────────────────────────────────────────────────────────

def render(class_key: str, extra: dict | None = None) -> None:
    """One model page: a head selector, then the same seven blocks for whichever is picked.

    `extra` maps a block number to a callable taking `(cards, row, theme)`, for a page that
    owns something the other three do not — the box-score page's no-fit floor, the game
    length page's per-class predictive check, and the minutes page's two-unit comparison.
    It is rendered after the numbered block it is keyed on.

    **The seven blocks are numbered and a page's own block is named**, which is the whole
    distinction: the numbers are the contract every model page keeps, so inserting a page's
    own material into the sequence would mean block 5 was a different block on two pages. A
    named block also goes where its question is asked rather than at the end — the
    floor-and-verdict blocks are keyed on block 1, because "what did this head buy over
    doing nothing" is the first thing to know about a head and not the eighth, and the
    shared `stan_block` is keyed on block 3, so the program lands just before the
    coefficients it defines.
    """
    extra = extra or {}
    spec = mc.model_class(class_key)
    shell.compact_tiles()
    st.title(spec.title)

    cards = load_cards()
    if cards is None:
        st.stop()
    heads = mc.heads_of(cards["index"], class_key)
    if not heads:
        st.warning(f"`{mc.INDEX_FILE}` carries no head of this class — run "
                   f"`{mc.MAKE_CARDS}`.")
        st.stop()

    with st.sidebar:
        st.header("Head")
        head = st.selectbox(
            "Head", heads, label_visibility="collapsed",
            format_func=lambda h: str(mc.head_row(cards["index"], h)["label"]),
            key=f"head-{class_key}",
            help="Every head on this page shares a likelihood family or a role, and each "
                 "states its own unit — they are not all fitted at the same one. Only "
                 "heads a simulated season actually reads at draw time appear here.")
        st.markdown("---")
        st.caption(
            f"Read from `{rel(predictions_dir())}` — reproduce with `{mc.MAKE_CARDS}`, "
            f"which reads the persisted posteriors and refits nothing. The sampler "
            f"diagnostics come from `{mc.MAKE_STAN}` and `{mc.MAKE_POSTERIORS}`.")

    row = mc.head_row(cards["index"], head)
    th = shell.current_theme()

    st.caption(spec.intro)
    # The chain role sits on the header beside the unit, and is *dropped* rather than
    # dashed when an older index does not carry it — a bare `·  —` in a heading reads as a
    # field the page failed to fill rather than as one it does not have.
    role_label = mc.text(row.get("chain_role_label"), "")
    st.markdown(
        f"### {row['label']} · fitted per **{row['unit']}**"
        + (f" · {role_label}" if role_label else "")
        + f"  \n`{row['family']}` · {int(row['n_fit']):,} rows · {mc.season_span(row)}")

    blocks = (
        (1, "What this head is", None,
         lambda: specification_block(row)),
        (2, "The features it was fed", None,
         lambda: features_block(cards, head, th)),
        (3, "Feature relationships", None,
         lambda: relationships_block(cards, head, th)),
        (4, "Coefficients", "Posterior means with 95% credible intervals, on the "
                            "standardized design scale.",
         lambda: coefficients_block(cards, head, th)),
        (5, "Predictive calibration", "The observed distribution against the one the "
                                      "posterior says it should have produced.",
         lambda: calibration_block(cards, row, head, th)),
        (6, "Predicted against observed", "Where the fit lands, and what it leaves "
                                          "behind on a scale every head shares.",
         lambda: residuals_block(cards, row, head, th)),
        (7, "Diagnostics", "What the sampler did, and what the build checked.",
         lambda: diagnostics_block(row, head)),
    )
    for number, title, caption, body in blocks:
        st.markdown("---")
        st.subheader(f"{number} · {title}")
        if caption:
            st.caption(caption)
        body()
        if number in extra:
            extra[number](cards, row, th)
