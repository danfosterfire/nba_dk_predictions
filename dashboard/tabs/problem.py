"""Tab 1 · Problem & constraints.

The frame: what is being predicted, what is knowable when, and where the signal is.

Every figure here is read from an artifact. The variance budget in particular had no
runnable source until `make variance-budget` landed — it was the most-quoted table in
the project and the least checkable, which is the whole reason
`docs/provenance-plan.md` exists.
"""

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx, optional
from dashboard.charts import fig_bars, fig_lines
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "problem"

# `preprocess.compute_dk_pts`, mirrored for display only — never reimplemented.
SCORING = [
    ("Point", "+1"), ("Made 3-pointer", "+0.5 bonus"), ("Rebound", "+1.25"),
    ("Assist", "+1.5"), ("Steal", "+2"), ("Block", "+2"), ("Turnover", "−0.5"),
    ("Double-double", "+1.5"), ("Triple-double", "+3"),
]

CONTRACT = [
    ("`min | available`", "successes / trials",
     "**game length** — 48, or 53/58/… in OT", "exposure only"),
    ("`fg2a` · `fg3a` · `fta`", "counts", "`min`", "exposure only"),
    ("`fg2m | fg2a`", "successes / trials", "`fg2a`", "2 pts"),
    ("`fg3m | fg3a`", "successes / trials", "`fg3a`", "3.5 pts"),
    ("`ftm | fta`", "successes / trials", "`fta`", "1 pt"),
    ("`reb` · `ast` · `stl` · `blk` · `tov`", "counts", "`min`",
     "1.25 / 1.5 / 2 / 2 / −0.5"),
]


def render(ctx: Ctx) -> None:
    tab_header(
        "Problem & constraints",
        "Predict a player's DraftKings fantasy points for each game of an "
        "**upcoming** season, plus the running season total — from information "
        "available before the season starts.")

    _scoring(ctx)
    _prediction_time(ctx)
    _variance_budget(ctx)
    _output_contract(ctx)
    _playoff_scope(ctx)
    _honest_caveat(ctx)

    st.markdown("---")
    st.markdown("### Decisions that set the frame")
    decision_cards(D.by_topic(TOPIC))


# ── Scoring ───────────────────────────────────────────────────────────────────

def _scoring(ctx: Ctx) -> None:
    st.markdown("### What is being scored")
    left, right = st.columns([1, 1])
    with left:
        st.dataframe(pd.DataFrame(SCORING, columns=["event", "dk_pts"]),
                     width="stretch", hide_index=True)
        note("Mirrored from `preprocess.compute_dk_pts` for display. The scoring "
             "function itself is reused verbatim everywhere and never "
             "reimplemented.")
    with right:
        st.markdown(
            "**The two bonuses are what make this hard.** A double-double is a "
            "simultaneous threshold on *five* components — points, rebounds, assists, "
            "steals, blocks — so it cannot be recovered from marginal means: "
            "`E[bonus] ≠ bonus(E[x])`.\n\n"
            "That single fact drives the architecture. It is why the deliverable is a "
            "**joint draw** rather than twelve marginals, why the components are "
            "modeled instead of `dk_pts` directly, and why the simulator must draw "
            "minutes rather than plug in their expectation.")
    provenance("`docs/dk_best_ball_rules.md`; formula mirrors "
               "`src/data/preprocess.py::compute_dk_pts`")


# ── The prediction-time constraint ────────────────────────────────────────────

def _prediction_time(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### What is knowable before the season starts")

    known, unknown = st.columns(2)
    with known:
        st.success(
            "**Known**\n\n"
            "- The season schedule\n"
            "- **Season-start rosters** — definitively which team each player is on\n"
            "- *Previous-season* stats for every team and player")
    with unknown:
        st.error(
            "**Not known**\n\n"
            "- Within-season roster changes (mid-season trades)\n"
            "- Current-season minutes\n"
            "- Injuries\n"
            "- Form")

    st.info(
        "**So the information set is a cross-season join: current-season roster "
        "membership × prior-season statistics.** Three consequences follow, and each "
        "is a place the obvious implementation is wrong:\n\n"
        "1. **Team composition aggregates the season-S roster described by S-1 "
        "stats** — not the S-1 roster. Same for opponents.\n"
        "2. **Minutes weights must come from S-1**, since season-S minutes are "
        "unknown.\n"
        "3. **Rookies and returnees are on the known roster but have no S-1 stats.** "
        "Impute, or exclude *and renormalize* — dropping them silently biases every "
        "aggregate toward veterans.")
    note("Team identity is known, so team embeddings and fixed effects are legitimate "
         "inputs. Season-start rosters are derivable from disk as a player's team in "
         "his earliest game of season S, and are known before the season, so they are "
         "not leakage in a backtest. **Every feature except the schedule-derived ones "
         "is constant within a player-season** — the only per-game variation available "
         "is opponent, home/away and rest. Only mid-season churn is irreducible, and "
         "it is out of scope for now.")


# ── The variance budget ───────────────────────────────────────────────────────

def _variance_budget(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Where the variance is")

    budget = optional(ctx.eda("variance_budget.csv"), target="make variance-budget")
    if budget is None:
        return

    def share(source: str) -> float:
        hit = budget[budget["source"] == source]
        return float(hit["share_of_variance"].iloc[0]) if len(hit) else float("nan")

    def value(source: str) -> float:
        hit = budget[budget["source"] == source]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    stat_tiles([
        ("Player-season identity", f"{share('player_season_identity'):.1%}",
         "Of **total** per-game variance. Who the player is, and which season."),
        ("Own minutes played", f"{share('own_minutes'):.1%}",
         "Of the **within-player residual** — and unknowable in advance."),
        ("Opponent × season", f"{share('opponent_x_season'):.2%}",
         "Of the within-player residual."),
        ("Home / away", f"{share('home_away'):.2%}",
         "Of the within-player residual."),
    ])
    st.warning(
        f"**These are in-sample ceilings, not achievable gains — and the two bases "
        f"must not be conflated.** Player-season identity is a share of *total* "
        f"per-game variance; everything else is a share of the *within-player "
        f"residual*, whose sd is **{value('within_player_season_residual_sd'):.2f} "
        f"dk_pts**. The artifact carries a `basis` column per row so misreading the "
        f"denominator is impossible.")

    headline = budget[budget["source"].isin([
        "player_season_identity", "own_minutes", "opponent_x_season", "home_away",
        "opponent_x_archetype_x_season"])].copy()
    headline["pct"] = headline["share_of_variance"] * 100
    st.plotly_chart(
        fig_bars(headline.sort_values("pct", ascending=False), "source", ["pct"],
                 ctx.th, "Share of variance by source — note the two bases",
                 axis_title="% of variance", height=320),
        width="stretch")

    st.markdown("#### Ceiling against achievable")
    matchup = optional(ctx.eda("opponent_matchup_tierA.csv"), target="make opponent")
    if matchup is not None:
        dk = matchup[matchup["outcome"] == "dk_pts"]
        if not dk.empty:
            row = dk.iloc[0]
            main = float(row["r2_main_effect"])
            inter = float(row["interaction_above_null"])
            stat_tiles([
                ("Opponent ceiling, in sample", f"{share('opponent_x_season'):.3%}",
                 "The ANOVA on contemporaneous opponent identity."),
                ("Opponent main effect, held out", f"{main:.3%}",
                 "On 2024-25/2025-26 with prior-season-only inputs. This is the "
                 "achievable number."),
                ("Interaction above its null, held out", f"{inter:+.3%}",
                 "Which is why the old 'encode opponent as an interaction' guidance "
                 "is withdrawn."),
                ("Main effect vs interaction",
                 f"{main / max(inter, 1e-9):.0f}×", "Out of sample, on dk_pts."),
            ])
            note("The ceiling and the achievable gain are different quantities on "
                 "different frames, and comparing one against the other is exactly "
                 "the error that produced the withdrawn opponent guidance. They sit "
                 "side by side here so the gap is visible rather than implied. The "
                 "interaction does earn its place **per component** — it nearly "
                 "doubles the main effect on blocks — just not on the aggregate.")
            table_view(matchup.round(6), "Opponent — held-out contrast, per outcome")

    st.markdown("#### Both null constructions, because the choice changes the answer")
    nulls = budget[budget["source"].str.contains("above_null", na=False)].copy()
    nulls = nulls[~nulls["source"].str.contains("variance_ceiling")]
    if not nulls.empty:
        nulls["pct"] = nulls["share_of_variance"] * 100
        nulls["permuted"] = nulls["null_construction"].fillna("—")
        st.plotly_chart(
            fig_bars(nulls, "permuted", ["pct"], ctx.th,
                     "Opponent × archetype × season, above a shuffled null",
                     axis_title="% of within-player residual variance", height=280),
            width="stretch")
        note("Same data, same cells: permuting **opponent** within season and "
             "permuting **archetype** give answers that differ by ~3×. With ~2,700 "
             "cells over 198k rows the expected chance R² is ~1.4% against a raw "
             "statistic of 2.31%, so the headline was largely cell count. Quoting one "
             "without naming the permuted marginal is a recorded past error, and the "
             "artifact carries a `null_construction` column so it cannot recur.")

    with detail("The variance budget — every row, with basis and null construction"):
        st.dataframe(budget.round(6), width="stretch", hide_index=True)
    provenance("`make variance-budget` → `outputs/eda/variance_budget.csv`; "
               "`make opponent` → `outputs/eda/opponent_matchup_tierA.csv`")


# ── The output contract ───────────────────────────────────────────────────────

def _output_contract(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The output contract")
    st.dataframe(
        pd.DataFrame(CONTRACT, columns=["component", "likelihood",
                                        "exposure / trials", "DK weight"]),
        width="stretch", hide_index=True)
    note("Twelve quantities per player-game, each with its own likelihood. Only "
         "**eight** reach the scoring function — `fg2m`, `fg3m`, `ftm`, `reb`, `ast`, "
         "`stl`, `blk`, `tov`. `min` and the three attempt counts matter solely "
         "through the exposure and trials they supply. Availability sits **upstream** "
         "of all twelve: it gates whether the player-game exists at all.")
    st.info("**`pts` is not a primitive count and is never modeled directly.** It is a "
            "weighted sum: `pts = 2·fg2m + 3·fg3m + ftm`. Writing "
            "`2·fgm + 3·fg3m + ftm` is the tempting error — the stored `fgm` already "
            "includes threes, so that pays a made three 5 points and matches `pts` on "
            "only the ~60% of games with `fg3m = 0`.")


# ── Regular season only ───────────────────────────────────────────────────────

def _playoff_scope(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Regular season only — the product decides it before the "
                "statistics do")

    st.info("**Round 1 runs 10/20 – 2/14 and Round 4 ends 4/4**, ahead of a mid-April "
            "playoff start. Predicting playoff games is out of scope by the rules of "
            "the product, not by modelling convenience. The statistics agree, and "
            "they agree in a specific way worth seeing.")

    prof = optional(ctx.eda("availability_profile.csv"),
                    target="make availability-profile")
    if prof is None:
        return
    scope = prof[prof["measurement"] == "playoff_scope"]
    buckets = scope[scope["key"] != "all"]
    if buckets.empty:
        return

    wide = buckets.pivot_table(index="key", columns="metric", values="value")
    order = ["bench_lt12", "rotation_12_24", "starter_24+"]
    wide = wide.reindex([o for o in order if o in wide.index]).reset_index()

    st.plotly_chart(
        fig_bars(wide, "key",
                 ["median_playoff_to_regular_mpg", "share_playing_more_in_playoffs",
                  "playoff_appearance_rate"], ctx.th,
                 "Playoff minutes and availability by prior-season role",
                 axis_title="ratio / share", height=340),
        width="stretch")
    st.error(
        "**Playoff minutes are a role interaction with a *sign change*, not a level "
        "shift.** Bench players lose about half their minutes while a majority of "
        "starters play *more*. A pooled playoff indicator would fit one coefficient "
        "to a −50% effect and a +5% effect at once, and bench players are numerous "
        "enough to drag it toward compression — distorting exactly the star minutes "
        "the model most needs right. Availability shifts too: rotations shorten, "
        "which is a different process from the one the availability head models.")
    note("**The playoff logs are still worth their fetch, as prior-season workload** "
         "— and that block earns its place on the availability head, though by "
         "selection rather than fatigue. Missing playoff logs raise rather than "
         "zero-fill, because 'nobody made the playoffs in 30 seasons' and 'the logs "
         "were never fetched' would otherwise be the same frame.")
    table_view(wide.round(4), "Playoff scope — table view")
    provenance("`make availability-profile` → `outputs/eda/availability_profile.csv` "
               "(`playoff_scope`); `docs/dk_best_ball_rules.md`")


# ── The honest caveat ─────────────────────────────────────────────────────────

def _honest_caveat(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The honest caveat")

    prof = optional(ctx.eda("target_profile.csv"), target="make target-profile")
    if prof is None:
        return

    first_k = prof[prof["analysis"] == "season_total"].copy()
    decomp = prof[prof["analysis"] == "season_total_decomposition"]

    if not first_k.empty:
        first_k["k"] = first_k["bucket"].astype(int)
        first_k = first_k.sort_values("k")
        five = first_k[first_k["k"] == 5]
        stat_tiles([
            ("R² after 5 games",
             f"{float(five['r2_extrapolated'].iloc[0]):.3f}" if len(five) else "—",
             "Extrapolating the first-k mean over the games actually played — so "
             "this isolates the *rate*, holding availability known."),
            ("MAE after 5 games",
             f"{float(five['mae'].iloc[0]):.0f}" if len(five) else "—",
             f"dk_pts, on a season total averaging "
             f"{float(five['mean'].iloc[0]):,.0f}." if len(five) else ""),
            ("Whole own-team context block", "+0.0086",
             "ΔR² on next-season per-game dk_pts, from every own-team feature "
             "combined — see the EDA tab."),
        ])
        st.plotly_chart(
            fig_lines(first_k, "k",
                      {"r2_extrapolated": "held-in R²", "r": "correlation"}, ctx.th,
                      "How much of the season total the first k games settle",
                      y_title="", x_title="games observed", height=320,
                      label_last=False),
            width="stretch")
        st.warning(
            "**Five current-season games settle 86% of the season total, against "
            "+0.0086 R² for the entire own-team context block.** That is the measured "
            "price of the prior-season-only constraint, and it is the number to keep "
            "in view before reading any later tab as a large win. It is not an "
            "argument against the work — the constraint *is* the product — but it "
            "does say where the ceiling is.")

    if not decomp.empty:
        def cell(bucket: str, col: str) -> float:
            hit = decomp[decomp["bucket"] == bucket]
            return float(hit[col].iloc[0]) if len(hit) else float("nan")

        spread = decomp[decomp["bucket"] == "games_ge_10"]
        st.markdown("#### The season total's two factors")
        stat_tiles([
            ("log(rate) alone", f"{cell('log_rate', 'r2'):.1%}",
             "Share of log(season total) explained."),
            ("log(games) alone", f"{cell('log_games', 'r2'):.1%}",
             "The availability half."),
            ("Both together", f"{cell('log_rate_and_games', 'r2'):.4f}",
             "Exactly 1.0 by arithmetic — a free internal check that the "
             "decomposition is of the right identity."),
            ("Games played spread",
             f"{float(spread['mean'].iloc[0]):.0f} ± "
             f"{float(spread['sd'].iloc[0]):.1f}" if len(spread) else "—",
             f"mean ± sd; p10 {float(spread['p10'].iloc[0]):.0f}, "
             f"p90 {float(spread['p90'].iloc[0]):.0f}" if len(spread) else ""),
        ])
        note(f"The two factors correlate at "
             f"**{cell('log_rate_vs_log_games', 'r'):+.3f}**, which is why the shares "
             f"sum past 1: they are *overlapping* shares, not a partition, and "
             f"quoting them as though they partitioned would double-count. The "
             f"population is player-seasons with ≥10 games — a load-bearing floor, "
             f"since the unfiltered frame carries one- and two-game seasons whose "
             f"per-game rate is noise, and it **reverses which factor dominates**.")

    with detail("Season-total analysis — first-k and the log decomposition"):
        st.dataframe(prof[prof["analysis"].str.startswith("season_total")].round(4),
                     width="stretch", hide_index=True)
    provenance("`make target-profile` → `outputs/eda/target_profile.csv`")
