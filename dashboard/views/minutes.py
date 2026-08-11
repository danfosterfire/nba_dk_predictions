"""Minutes — two heads over the same quantity, fitted and scored at two different units.

Page 4 of `docs/dashboard-plan.md`'s expansion, and the fourth instantiation of the generic
renderer in `dashboard/views/model_page.py`. Behind the selector: the marginal
`min | available` head, fitted season-collapsed as successes out of real game length, and
the team-game composition, which allocates each team-game's `5 x game_length` minutes among
the players who played as sequential beta-binomial trials ordered by prior-season share.

**This page has a job the other three model pages do not**, and it is the reason the build
order called it bespoke. `minutes_unification.csv` scores *both* heads at the season unit off
their persisted posteriors, and the result is the cleanest demonstration in this repo that
**a head is only a model at the unit it was scored at**: the composition clears its own
per-player-game no-fit floor decisively and fails the season-unit one, on the same posterior
and the same draws, while the marginal head does the reverse.

Three named blocks carry that, each keyed on the numbered block whose question it extends:

- **under block 1**, the two-unit verdict — because "which unit is this a model at" is the
  first thing to know about either head, exactly as "what did this head buy over doing
  nothing" is on the box-score page. It is followed by the four readings that say *what*
  fails: the two heads tie on where the predictive sits and separate 4.68x on how wide it is.
- **under block 5**, the injected per-player-season effect — because what fails at the season
  unit is calibration, and sigma is what moves the PIT KS. The shipped value is read from the
  composition's own card rather than typed.
- **under block 6**, the zero-sum team constraint — because block 6 is four panels of
  *marginal* residuals and no marginal metric can see whether a head carries it. This is
  also the block that lands on drafting: a same-team stack's minutes are anti-correlated
  rather than independent, and a handcuff is a hedge that exists only if the model has the
  sign. A reader reaches this page before the strategy page.

Every figure here is a comparison **between** the two heads, so unlike pages 5 and 6 nothing
is highlight-and-gray: each head keeps one palette slot across the whole page
(`model_cards.MINUTES_SLOTS`) and the tiles say which one the selector has open.
"""

import pandas as pd
import streamlit as st

from dashboard import model_cards as mc
from dashboard.artifacts import optional, predictions_dir
from dashboard.charts import (fig_coupling, fig_metric_facets, fig_paired,
                              fig_sigma_grids, fig_unit_verdict)
from dashboard.views import model_page

CLASS_KEY = "minutes"


@st.cache_data(show_spinner=False)
def load_unification() -> pd.DataFrame | None:
    """`make minutes-unification` — both heads at the season unit, neither refitted."""
    return optional(predictions_dir() / mc.MINUTES_UNIFICATION_FILE,
                    target=mc.MAKE_MINUTES_UNIFICATION)


@st.cache_data(show_spinner=False)
def load_ladder() -> pd.DataFrame | None:
    """The composition's own variant ladder, which is where the two heads meet per game."""
    return optional(predictions_dir() / mc.COMPOSITION_METRICS_FILE,
                    target=mc.MAKE_STAN_COMPOSITION)


def units_block(cards: dict, row: pd.Series, th: dict) -> None:
    """This page's headline, under block 1: the same posterior at two units.

    Named rather than numbered, like the blocks pages 5 and 6 own. It renders the same on
    both heads because it *is* a comparison between them — what follows the selector is the
    tile row, which reads the open head's own side of it in the open head's own direction.
    """
    unification = load_unification()
    ladder = load_ladder()
    board = mc.unit_board(unification, ladder, cards["index"])
    if board.empty:
        return
    head = str(row["head"])
    gap = mc.season_gap(unification, head)
    mine = board[board["head"] == head]

    st.markdown("---")
    st.subheader("Two units, two verdicts")
    st.caption(
        "Both heads are quoted against a **no-fit floor**, and at two different units: the "
        "composition's own fitted unit, where its ladder refits the marginal head as its "
        f"control (`{mc.COMPARATOR_ARM}`), and the season total, where "
        f"`{mc.MAKE_MINUTES_UNIFICATION}` sums the composition's per-game draws and scores "
        "both heads on the player-seasons they share. Nothing was refitted for either "
        "reading — both heads are rehydrated around their persisted posteriors and score "
        "through their own `predict_samples`.")

    tiles = []
    for _, part in mine.iterrows():
        tiles.append((f"CRPS per {part['unit_label']}", f"{part['crps']:,.4f}",
                      f"Read from the `{part['arm']}` arm, against a no-fit floor of "
                      f"{part['floor_crps']:,.4f} — "
                      + ("clears it" if part["clears"] else "**does not clear it**")))
    if gap is not None:
        other = (mc.HEAD_MARGINAL if head == mc.HEAD_COMPOSITION else mc.HEAD_COMPOSITION)
        tiles.append((
            "Season-unit gap", f"{float(gap['crps_delta']):+,.2f}",
            f"Paired bootstrap over {int(gap['n']):,} player-seasons against "
            f"{mc.head_label(cards['index'], other, other)}, in CRPS minutes. 95% interval "
            f"[{float(gap['ci_lo']):+,.2f}, {float(gap['ci_hi']):+,.2f}] over "
            f"{int(gap['n_bootstrap']):,} resamples. Negative is better"))
        tiles.append((
            "Season-total spread", f"{float(gap['sd_ratio_minutes_over_composition']):.2f}×",
            "How much narrower the composition's season-total predictive is than the "
            "marginal head's. This is the whole failure — see the four panels below"))
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(fig_unit_verdict(board, th, mc.MINUTES_SLOTS, title=""),
                    width="stretch", key=f"units-{head}",
                    config={"displayModeBar": False})
    # The two floors, read rather than typed: they are the reason the axis is a ratio, so a
    # caption that rounded them by hand would be arguing from numbers that are not on the
    # page anywhere else.
    floors = " and ".join(
        f"{part['floor_crps']:,.4g} per {part['unit_label']}"
        for _, part in board.drop_duplicates("unit").iterrows())
    st.caption(
        f"**The zero line is the floor, and the axis is a ratio to it** — a floor of "
        f"{floors} cannot share an axis, and the floor is what "
        "every head in this project is quoted against anyway, so the two units are made "
        "commensurable by the reference they already had. The finding is the reversal: the "
        "same posterior is on opposite sides of the line in the two panels, and so is the "
        "head it is drawn against. Per-game the composition beats its floor and the "
        "independent draw does not; per season the composition fails the carry-forward "
        "floor the marginal head clears. **A head is only a model at the unit it was "
        "scored at.**")

    spread = mc.spread_panel(unification, cards["index"])
    if not spread.empty:
        st.markdown("**What fails at the season unit is the spread, not the fit**")
        st.plotly_chart(fig_metric_facets(spread, th, mc.MINUTES_SLOTS, title=""),
                        width="stretch", key=f"spread-{head}",
                        config={"displayModeBar": False})
        coverage = mc.coverage_row(unification)
        season = board[board["unit"] == "season"]
        st.caption(
            "The top row is where the predictive sits and the bottom row is how wide it is."
            + (f" On the {int(season['n'].iloc[0]):,} player-seasons both heads cover they "
               f"are a tie on the top row — and the composition is the **less biased** of "
               f"the two — while the bottom row separates them by "
               f"{float(gap['sd_ratio_minutes_over_composition']):.2f}×."
               if len(season) and gap is not None else "")
            + " Summing draws that are iid across games cannot manufacture season-level "
              "heterogeneity: per-game noise averages down by roughly 1/√G while a "
              "season-level multiplier passes through in full. The rule on the "
              "predictive-sd panel is the head's own residual sd — the spread a calibrated "
              "season total has to cover — and it is why the failure reads as *too narrow* "
              "rather than as the other head being too wide."
            + (f" Coverage cuts the other way and is reported on its own row rather than "
               f"folded in: the composition also scores "
               f"{int(coverage['n']):,} validation player-seasons against the marginal "
               f"head's {int(season['n'].iloc[0]):,}, the rookies and low-minute players a "
               f"prior-minutes filter drops — who are draftable, and who are also easier "
               f"to predict."
               if coverage is not None and len(season) else ""))

    with st.expander("Table view — both heads at both units"):
        st.dataframe(mc.unit_table(board), hide_index=True, width="stretch")
        st.caption(
            "`Against the floor` is the quantity the figure draws: how much of the floor's "
            "CRPS the head removes, positive when it clears. The two CRPS columns are in "
            "the units of their own row and are not comparable down the table — which is "
            "the reason the figure normalizes rather than plotting them.")


def sigma_block(cards: dict, row: pd.Series, th: dict) -> None:
    """Under block 5: the missing parameter, since what failed above is calibration.

    The injection is `σ·z` per player-season per draw, shared across that player's games and
    pushed back through the head's **own** sequential allocation — so the team total stays
    exact. `σ = 0` recovers the un-injected head, which is why the sweep's first row is the
    same 170.06 the block above reports.
    """
    unification = load_unification()
    sweep = mc.sigma_sweep(unification)
    if sweep.empty:
        return
    head = str(row["head"])
    shipped = mc.shipped_sigma(cards["index"])
    gaps = mc.sigma_gaps(sweep, shipped)
    marginal = mc.season_reading(unification, mc.HEAD_MARGINAL, "crps_minutes")
    at_shipped = mc.sigma_row(sweep, shipped)

    st.markdown("---")
    st.subheader("The missing parameter")
    st.caption(
        "A season-level term cannot be *shared*: a league-wide shift re-tilts the "
        "allocation and leaves the team total where it was, and against a head that "
        "allocates every minute in the league it has no residual variance to reach. A "
        "per-**(player, season)** effect is not shared. Injecting one — `σ·z` per "
        "player-season per posterior draw, shared across that player's games and pushed "
        "back through the head's own allocation — is what a fitted random effect's "
        "predictive integrates to, and it needs no refit. **Every number in this block is "
        "the composition's**, whichever head the selector has open: the marginal head "
        "appears only as the baseline the gap is measured against, and takes no effect.")

    if at_shipped is not None:
        marginal_ks = mc.season_reading(unification, mc.HEAD_MARGINAL, "pit_ks")
        marginal_sd = mc.season_reading(unification, mc.HEAD_MARGINAL, "predictive_sd")
        # Read rather than typed, like every other number on the surface: the two the
        # shipped sigma is worth comparing against are the same head without the effect and
        # the head it is trying to reach.
        un_injected = mc.sigma_row(sweep, 0.0)
        against = ""
        if un_injected is not None and marginal_sd is not None:
            against = (f" — against {float(un_injected['val_sd']):,.2f} without it, and "
                       f"{marginal_sd:,.2f} for the marginal head")
        tiles = [
            ("Shipped σ", f"{float(at_shipped['sigma']):.3f}",
             "Read from the composition's own card (`player_season_sigma`), not typed here. "
             "It is estimated on the training seasons, and applied by "
             "`minutes_unification.rehydrate_composition` so a consumer gets the effect by "
             "loading the head"),
            ("Gap at that σ", f"{float(at_shipped['val_delta']):+,.2f}",
             f"Against the marginal head in season-total CRPS minutes, 95% interval "
             f"[{float(at_shipped['val_ci_lo']):+,.2f}, "
             f"{float(at_shipped['val_ci_hi']):+,.2f}] — an interval straddling zero is a "
             f"tie, which is what `{at_shipped['verdict']}` in the artifact records"),
            ("Predictive sd · injected", f"{float(at_shipped['val_sd']):,.2f}",
             "The injected composition's season-total predictive sd, in minutes, at the "
             "shipped σ" + against),
        ]
        if marginal_ks is not None:
            tiles.append((
                "PIT KS · injected", f"{float(at_shipped['val_pit_ks']):.4f}",
                f"Against the marginal head's {marginal_ks:.4f} at the same unit — the "
                f"injected arm is the better calibrated of the two, with the team "
                f"constraint still exact"))
        for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
            col.metric(label, value, help=helptext)

    if not gaps.empty:
        st.plotly_chart(
            fig_paired(gaps, th, baseline="the marginal head",
                       unit="season-total CRPS minutes"),
            width="stretch", key=f"sigma-gap-{head}", config={"displayModeBar": False})
        st.caption(
            "The same encoding the tournament page uses for a comparison that does not "
            "resolve, because it is the same idea: **an interval straddling zero is a "
            "tie**, carried three ways at once — the interval visibly crosses the "
            "baseline, the marker is hollow, and the legend and the table below both say "
            "so. Here a tie is the *result*: the un-injected head loses by "
            f"{float(gaps['gap'].iloc[0]):+,.2f} and one parameter closes it.")

    st.plotly_chart(
        fig_sigma_grids(sweep, th, marginal_crps=marginal),
        width="stretch", key=f"sigma-grid-{head}", config={"displayModeBar": False})
    st.caption(
        "**Two grids, disjoint rows, and that is the point.** σ read off the validation "
        "split would be tuned on the split it is later scored against, so the identical "
        "grid was re-run on the last two *training* seasons. Both optima are interior and "
        "they are one grid step apart, which is the evidence that the effect size was not "
        "moved by the evaluation data — so the shipped σ owes those rows nothing. The two "
        "CRPS levels are **not** comparable to each other: the panels score different "
        "player-seasons, and only the location of each minimum is being read across them. "
        "MAE barely moves across the whole sweep, so what this buys is spread and not fit.")

    with st.expander("Table view — the whole sweep, both grids"):
        st.dataframe(mc.sigma_table(sweep, shipped), hide_index=True, width="stretch")
        st.caption(
            "`σ = 0.000` is the un-injected head exactly, so the first row of this table "
            "and the season-unit row of the block above are the same measurement. The "
            "effect is built into `composition_glm.stan` as an optional fitted parameter as "
            "well, and is not fitted: at 12× the shipped arm's cost it did not converge "
            "inside the budget.")


def constraint_block(cards: dict, row: pd.Series, th: dict) -> None:
    """Under block 6: the one thing four panels of marginal residuals cannot show.

    A team's season minutes are a fixed pot, so teammates' totals are negatively correlated
    by arithmetic. Whether a head carries that is invisible in every marginal metric on this
    page — and it is the half of the comparison that decides two drafting strategies.
    """
    unification = load_unification()
    panel = mc.teammate_coupling(unification, cards["index"])
    if panel.empty:
        return
    head = str(row["head"])

    st.markdown("---")
    st.subheader("What no marginal panel can see")
    st.caption(
        "The four panels above are marginals, one row at a time. A team's season minutes "
        "are a fixed pot, though, so two teammates' season totals are negatively correlated "
        "by arithmetic rather than by fitting: a fixed sum over K players forces a mean "
        "pairwise **r = −1/(K−1)**. Each head is drawn against that value at its **own** "
        "measured roster size, because the two cover different numbers of teammates.")

    tiles = []
    for _, part in panel.iterrows():
        tiles.append((f"{part['label']} · team season sd", f"{part['team_sd']:,.1f}",
                      f"Predictive sd, in minutes, of a whole team's season minutes — a "
                      f"quantity that is physically fixed. Measured over "
                      f"{int(part['n']):,} single-team player-seasons at a mean roster of "
                      f"{part['roster']:.2f}"))
    for col, (label, value, helptext) in zip(st.columns(max(len(tiles), 3)), tiles):
        col.metric(label, value, help=helptext)

    st.plotly_chart(fig_coupling(panel, th, mc.MINUTES_SLOTS, title=""),
                    width="stretch", key=f"coupling-{head}",
                    config={"displayModeBar": False})
    st.caption(
        "The hollow marker is what the fixed pot forces and the filled one is what the head "
        "puts there; the rule between them is the gap, which is the reading. The "
        "composition sits on its own constraint. The marginal head reads essentially zero "
        "against a forced value of "
        f"{float(panel.loc[panel['head'] == mc.HEAD_MARGINAL, 'forced'].iloc[0]):+.4f} "
        "and puts a four-figure predictive sd on a team season total that cannot move, so "
        "it assigns real probability to outcomes that cannot happen. Neither measured "
        "figure is exactly zero because dropping traded players leaves a *subset* of each "
        "roster and a subset of a fixed-sum set has no fixed sum — the ratio is what "
        "carries, not the absolute values."
        if (panel["head"] == mc.HEAD_MARGINAL).any() else "")
    st.caption(
        "**This is the block that lands on drafting, and it lands before the strategy "
        "page on purpose.** Two swept axes depend on the sign directly. *Stacking*: a "
        "same-team pair's minutes are anti-correlated, so under independent draws a "
        "minutes-driven stack is mispriced, plausibly with the wrong sign. *Handcuffing*: "
        "drafting a starter's backup is a hedge that **only exists** if the model carries "
        "the negative correlation — a head without it cannot discover the strategy, so a "
        "sweep over it would silently never propose one.")

    with st.expander("Table view — the coupling, and what forces it"):
        st.dataframe(mc.coupling_table(panel), hide_index=True, width="stretch")


def render() -> None:
    model_page.render(CLASS_KEY,
                      extra={1: units_block, 5: sigma_block, 6: constraint_block})
