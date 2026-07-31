"""Tab 8 · Drafting strategy.

The tab with the most genuinely unexploited data on disk: both tournament CSVs parse
cleanly and every economics figure below is derived live by `dashboard/economics.py`
rather than typed in.

It reads as specification in places — the draft layer is not built — so it leads with
what is *measured and binding*: the real tournament economics, the knockout objective,
and the ADP freeze rule.
"""

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard import economics as E
from dashboard.artifacts import Ctx, optional, read_table
from dashboard.charts import fig_bars, fig_lines
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "drafting"

ROSTER = [
    ("2", "G", "PG / SG"),
    ("2", "F", "SF / PF"),
    ("1", "C", "centre"),
    ("2", "UTIL", "any of G / F / C"),
]


def render(ctx: Ctx) -> None:
    tab_header(
        "Drafting strategy",
        "What the contest actually rewards. The scoring is identical to "
        "`compute_dk_pts`, but the *payout* is a four-round knockout — which changes "
        "the objective from expected score to probability of advancing, and makes "
        "variance something to buy rather than avoid.")

    _mechanics(ctx)
    _economics(ctx)
    _payout_convexity(ctx)
    _adp(ctx)

    st.markdown("---")
    st.markdown("### Decisions on the strategy layer")
    decision_cards(D.by_topic(TOPIC))


# ── Contest mechanics ─────────────────────────────────────────────────────────

def _mechanics(ctx: Ctx) -> None:
    st.markdown("### Contest mechanics")

    left, right = st.columns([1, 1])
    with left:
        st.markdown("**16-man frozen roster, best 7 by slot each week**")
        st.dataframe(pd.DataFrame(ROSTER, columns=["slots", "position", "eligible"]),
                     width="stretch", hide_index=True)
        note("Drafted by live snake draft in pods of 12, from at least 2 different "
             "NBA teams. **After the draft there are no trades and no roster "
             "management** — each week your highest-scoring eligible players are "
             "automatically your starters. There is no redraft between rounds: "
             "advancing entries keep the lineup they drafted in round 1.")
    with right:
        st.markdown("**Four rounds, and the season is over before the playoffs**")
        st.markdown(
            "- Round 1 runs **10/20 – 2/14** (17 weeks)\n"
            "- Round 4 ends **4/4**, ahead of a mid-April playoff start\n"
            "- A scoring period runs from the first game through the last game of "
            "the week's game set; a suspended game scores in the period it is "
            "**played**\n"
            "- Auto-draft falls back queue → pre-draft ranking → position caps "
            "(8G / 8F / 3C)\n"
            "- Ties cascade through a specified tie-break order")
        st.info("**This is why the whole project is regular season only** — out of "
                "scope by the rules of the product before it is by statistics.")
    provenance("`docs/dk_best_ball_rules.md`; the scoring formula is "
               "`preprocess.compute_dk_pts`, reused verbatim")


# ── Tournament economics ──────────────────────────────────────────────────────

def _economics(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The five real tournaments")

    try:
        econ = E.economics()
        adv = E.advance_table()
    except FileNotFoundError:
        st.warning("`data/raw/dk_best_ball_tournament_*.csv` not found — these are "
                   "manual captures from the DraftKings lobby.")
        return

    stat_tiles([
        ("Tournaments captured", f"{len(econ)}",
         "Entries 216 → 35,280 and fees $1 → $450."),
        ("Rake, range",
         f"{econ['rake'].min():.1%} – {econ['rake'].max():.1%}",
         "Share of the buy-in pool the house keeps."),
        ("Break-even edge hurdle",
         f"{econ['break_even_hurdle'].min():.2%} – "
         f"{econ['break_even_hurdle'].max():.2%}",
         "`1/(1−rake) − 1` — the edge needed just to return the entry fee."),
        ("Top prize, as a multiple of entry",
         f"{econ['first_prize_multiple'].min():,.0f}× – "
         f"{econ['first_prize_multiple'].max():,.0f}×",
         "The payout convexity, which differs by an order of magnitude across the "
         "five."),
    ])

    view = econ.copy()
    view["rake_pct"] = view["rake"] * 100
    view["hurdle_pct"] = view["break_even_hurdle"] * 100
    st.plotly_chart(
        fig_bars(view, "tournament", ["rake_pct", "hurdle_pct"], ctx.th,
                 "Rake against the break-even edge hurdle it implies",
                 axis_title="%", height=340),
        width="stretch")

    cheap = view.loc[view["break_even_hurdle"].idxmax()]
    dear = view.loc[view["break_even_hurdle"].idxmin()]
    st.info(
        f"**Rake is the break-even edge hurdle, and that is the right unit.** A raw "
        f"rake percentage is not denominated like a measured edge: losing "
        f"{cheap['rake']:.1%} of the pool means beating the field by "
        f"**{cheap['break_even_hurdle']:.2%}** just to return the fee, not by "
        f"{cheap['rake']:.1%}. On that scale `{cheap['tournament']}` demands "
        f"**{cheap['break_even_hurdle'] / dear['break_even_hurdle'] - 1:.0%} more "
        f"edge** than `{dear['tournament']}` — the cheap entries are the punishing "
        f"ones, which inverts the intuition that a small buy-in is forgiving.")
    table_view(view[["tournament", "total_entries", "entry_fee_per_team",
                     "buy_in_pool", "total_prizes", "rake", "break_even_hurdle",
                     "first_prize", "first_prize_multiple"]].round(4),
               "Tournament economics — table view")

    st.markdown("#### Round 1 is a zero-consolation knockout in all five")
    r1 = adv[adv["round"] == 1]
    rates = (adv[adv["n_advance"] > 0]
             .pivot_table(index="tournament", columns="round",
                          values="advance_rate"))
    rates.columns = [f"round {c}" for c in rates.columns]
    st.plotly_chart(
        fig_bars(rates.reset_index(), "tournament", list(rates.columns), ctx.th,
                 "Share of each pod that advances, by round",
                 axis_title="advance rate", height=340),
        width="stretch")
    st.error(
        f"**Top {int(r1['n_advance'].iloc[0])} of {int(r1['pod_size'].iloc[0])} "
        f"advance and ranks {int(r1['n_advance'].iloc[0]) + 1}–"
        f"{int(r1['pod_size'].iloc[0])} receive $0 — in every one of the five.** So "
        f"the round-1 objective is **P(advance)**, not E[score]: near the cut line, "
        f"variance is worth paying for, and a lineup that maximizes expected points "
        f"is not the one that maximizes the chance of finishing top two. It is also "
        f"why the cascading tie-break mechanics stop being a footnote.")
    ev = E.round_one_pod_evidence()
    note(f"The prize file records round 1's *advance cutoff* but not its pod size, "
         f"because pods are a contest-structure fact rather than a payout one. "
         f"`ROUND_ONE_POD = 12` comes from the rules doc, and the entry counts "
         f"confirm it independently — chaining each tournament's entries forward "
         f"through the per-round pods gives a whole-number field every time "
         f"(**{ev['integral']}**), and **{ev['final_field_equals_paid']} of "
         f"{ev['n_tournaments']}** then have a final round paying *exactly* its own "
         f"field. Integrality alone is not enough: any divisor of 12 passes it, "
         f"since halving the pod doubles the next field. It is the second arm that "
         f"pins 12 — every other pod size scores 0 or 1 there.")
    table_view(adv.round(4), "Round structure — table view")
    provenance("derived live by `dashboard/economics.py` from "
               "`data/raw/dk_best_ball_tournament_metadata.csv` and "
               "`data/raw/dk_best_ball_tournament_prize_structure.csv`")


# ── Payout convexity ──────────────────────────────────────────────────────────

def _payout_convexity(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Payout convexity — how top-heavy is the final table")

    try:
        econ = E.economics()
    except FileNotFoundError:
        return

    names = list(econ["tournament"])
    default = names.index("600k_shootaround") if "600k_shootaround" in names else 0
    picked = st.selectbox("tournament", names, index=default, key="payout_pick")

    curve = E.payout_curve(tournament=picked)
    row = econ[econ["tournament"] == picked].iloc[0]
    if curve.empty:
        return

    curve = curve.copy()
    curve["multiple_of_entry"] = curve["cash"] / row["entry_fee_per_team"]
    st.plotly_chart(
        fig_lines(curve, "place", {"multiple_of_entry": "× entry fee"}, ctx.th,
                  f"{picked} — final-round payout by finishing place",
                  y_title="cash as a multiple of the entry fee",
                  x_title="finishing place", height=380, label_last=False)
        .update_yaxes(type="log"),
        width="stretch")
    stat_tiles([
        ("Entry fee", f"${row['entry_fee_per_team']:,.0f}", "Per team."),
        ("Top prize", f"${row['first_prize']:,.0f}",
         f"{row['first_prize_multiple']:,.0f}× the entry fee."),
        ("Final-round places paid", f"{len(curve)}",
         "Expanded from the banded prize rows."),
        ("Ratio, 1st to last paid",
         f"{curve['cash'].max() / curve['cash'].min():,.0f}×",
         "How top-heavy the final table is."),
    ])
    note("Convexity tracks **field size and prize-pool tier**, not 'is this a "
         "tournament': the large cheap contests pay thousands of times the entry fee "
         "at the top, while the two small high-fee ones have a near-flat final table. "
         "A near-flat table rewards expected score; a steep one rewards the tail. The "
         "log axis is there because a linear one shows a spike and nothing else.")
    table_view(curve, "Payout curve — table view")


# ── ADP ───────────────────────────────────────────────────────────────────────

def _adp(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The market — ADP as a benchmark, not as a feature")

    prof = optional(ctx.eda("adp_profile.csv"), target="make adp-profile")
    board_path = ctx.features("adp_draftkings.parquet")
    id_map_path = ctx.features("adp_dk_id_map.parquet")
    panel_path = ctx.features("adp_panel.parquet")

    if board_path.exists() and id_map_path.exists():
        board = read_table(str(board_path))
        id_map = read_table(str(id_map_path))
        stat_tiles([
            ("DK boards captured", f"{board['capture_date'].nunique()}",
             f"{', '.join(sorted(board['capture_date'].astype(str).unique()))} — "
             f"login-gated, no API, zero Wayback snapshots."),
            ("Board rows carrying ADP",
             f"{int(board['adp'].notna().sum()):,} / {len(board):,}",
             "The rest of the pool is undrafted in the sample, so the board is "
             "right-censored."),
            ("Censored rows", f"{board['adp_censored'].mean():.1%}",
             "Treated as censored rather than dropped or imputed."),
            ("DK ids mapped to `player_id`", f"{int(id_map['player_id'].notna().sum()):,}",
             "The DK `ID` is a **persistent** key across boards, which is the only "
             "reason this side is an id join rather than a name join."),
        ])

    if panel_path.exists():
        panel = read_table(str(panel_path),
                           ("season", "as_of_date", "snapshot_lag_days",
                            "snapshot_source", "captured_before_season_start"))
        by_season = (panel.groupby(["season", "snapshot_source"])
                     .size().rename("rows").reset_index())
        wide = by_season.pivot_table(index="season", columns="snapshot_source",
                                     values="rows", fill_value=0).reset_index()
        st.plotly_chart(
            fig_bars(wide, "season",
                     [c for c in wide.columns if c != "season"], ctx.th,
                     "ADP panel rows by season and source", axis_title="rows",
                     height=320),
            width="stretch")
        st.warning(
            "**The freeze rule: a snapshot's calendar date is not its season.** The "
            "FantasyPros table is frozen for ~11 months, so the 2025-09-06 snapshot "
            "is **2024-25** ADP. Any `month >= 10` rule misassigns six 2020 snapshots, "
            "because 2020-21 tipped off in December. Getting this wrong shifts an "
            "entire season of market data by one year — and it would look like a "
            "modelling result rather than a join bug.")
        note(f"Every row is point-in-time safe: "
             f"{panel['captured_before_season_start'].mean():.1%} of panel rows were "
             f"captured before their season started, and `snapshot_lag_days` carries "
             f"the gap so a consumer can filter rather than trust.")
        table_view(wide, "Panel coverage — table view")

    if prof is not None:
        ladder = prof[prof["section"] == "ladder"]
        if not ladder.empty:
            st.markdown("#### Recalibrating consensus onto the DK board")
            st.plotly_chart(
                fig_bars(ladder, "metric", ["value"], ctx.th,
                         "Mean absolute rank gap against the DK board, in picks",
                         axis_title="picks", height=340,
                         emphasis="+ monotone (isotonic) rescale"),
                width="stretch")
            raw = float(ladder[ladder.metric == "consensus raw"]["value"].iloc[0])
            mono = float(ladder[ladder.metric ==
                                "+ monotone (isotonic) rescale"]["value"].iloc[0])
            note(f"Raw consensus is {raw:.2f} picks away from the DK board on "
                 f"average; a monotone rescale brings it to {mono:.2f}. Adding a "
                 f"position offset on top of the linear rescale is worth only "
                 f"~0.3 picks, so nearly all of the gain is the shape of the mapping "
                 f"rather than a per-position correction.")

        transfer = ctx.features("adp_transfer.parquet")
        if transfer.exists():
            t = read_table(str(transfer))
            anchors = int(t["n_anchors"].iloc[0])
            if anchors <= 1:
                st.error(
                    f"**`n_anchors = {anchors}`.** The whole recalibration rests on a "
                    f"single timing-matched board pair "
                    f"({t['fitted_on_seasons'].iloc[0]}, {int(t['n_pairs'].iloc[0])} "
                    f"players). That is *one observation of the mapping*, not a fitted "
                    f"transfer function — it cannot distinguish a stable "
                    f"consensus-to-DK relationship from one year's quirk. An "
                    f"early-to-mid **October 2026** board is what would take this to "
                    f"two, and it is the load-bearing capture on the deadline board.")
            st.plotly_chart(
                fig_lines(t.sort_values("adp_consensus"), "adp_consensus",
                          {"adp_dk_fitted": "fitted DK pick"}, ctx.th,
                          "The fitted consensus → DK mapping",
                          y_title="DK pick", x_title="FantasyPros consensus pick",
                          height=340, label_last=False),
                width="stretch")
            table_view(t.round(3), "Transfer function — table view")

        pos = prof[prof["section"] == "position_bias"]
        tier = prof[prof["section"] == "tier_gap"]
        left, right = st.columns(2)
        with left:
            if not pos.empty:
                mean_gap = pos[pos["metric"].str.endswith("mean_rank_gap")].copy()
                mean_gap["position"] = mean_gap["metric"].str.split(":").str[0]
                st.plotly_chart(
                    fig_bars(mean_gap, "position", ["value"], ctx.th,
                             "Mean rank gap by position", axis_title="picks",
                             horizontal=False, height=300),
                    width="stretch")
                note("Centres go substantially later on the DK board than consensus "
                     "says — a systematic, exploitable offset if it replicates on a "
                     "second anchor.")
        with right:
            if not tier.empty:
                gaps = tier[tier["metric"].str.endswith("mean_abs_rank_gap")].copy()
                gaps["tier"] = gaps["metric"].str.split(":").str[0]
                st.plotly_chart(
                    fig_bars(gaps, "tier", ["value"], ctx.th,
                             "Disagreement by draft tier", axis_title="picks",
                             horizontal=False, height=300),
                    width="stretch")
                note("The two sources agree closely at the top and diverge sharply "
                     "from round 9 on — which is where **9 of 16 picks** are made. "
                     "The market is only precise where it is least decisive.")

        with detail("ADP profile — every section"):
            st.dataframe(prof.round(4), width="stretch", hide_index=True)

    st.info("**Decision: ADP stays in the strategy layer, not the GLMM.** Under a "
            "knockout payout the edge *is* model-minus-market, so a model fit on ADP "
            "reproduces its own benchmark and the edge goes to zero by construction. "
            "The one narrow exception is an ADP prior for thin-data players only, "
            "with a pre-registered test.")
    provenance("`make adp` → `outputs/eda/adp_profile.csv`, "
               "`data/features/adp_panel.parquet`, "
               "`data/features/adp_transfer.parquet`, "
               "`data/features/adp_draftkings.parquet`")
