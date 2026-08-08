"""Tab 7 · Season simulations.

**Nothing here is built**, and the tab says so at the top rather than reading as
though it were. Its job is to show that the *specification* is already pinned by
measurements — which is the honest status and, for this reader, the interesting part.
Model diagnostics arrive once there is a simulator to diagnose.
"""

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import ROOT, Ctx, optional, window
from dashboard.charts import fig_bars, fig_heatmap
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "simulations"

# The generative chain, in evaluation order. `artifact` is checked live, so this
# doubles as the build tracker.
CHAIN = [
    ("availability", "gp | team games", "beta-binomial",
     "team games", "outputs/predictions/stan_availability_metrics.csv"),
    ("minutes", "min | available", "beta-binomial",
     "game length", "outputs/predictions/stan_minutes_metrics.csv"),
    ("counts", "fg2a · fg3a · fta · reb · ast · stl · blk · tov | min",
     "negative binomial", "min", "outputs/predictions/stan_component_metrics.csv"),
    ("conversions", "fg2m | fg2a · fg3m | fg3a · ftm | fta", "beta-binomial",
     "attempts", "outputs/predictions/stan_component_metrics.csv"),
    ("spell process", "which games he misses", "entry x exit x beta-geometric spells",
     "team games", "outputs/predictions/stan_games_played_metrics.csv"),
    ("residual copula", "cross-component dependence", "Gaussian copula",
     "—", "outputs/predictions/simulator_draws.parquet"),
]


def render(ctx: Ctx) -> None:
    tab_header(
        "Season simulations",
        "Composing the heads into a joint draw over a whole season, then over a "
        "16-man roster, then over a four-round knockout.")

    st.warning(
        ":material/construction: **Half the simulator is built.** The spell process — "
        "which games a player misses — shipped on 2026-08-05 (`make stan-games-played`); "
        "the residual copula that composes eleven marginal posteriors into one correlated "
        "season has not. The point worth making about the rest is that its specification "
        "is already **pinned by measurements** rather than by preference: every number "
        "below is an input the simulator has to honour, measured before it exists.")

    _chain(ctx)
    _correlation(ctx)
    _block_inflation(ctx)
    _bonus(ctx)
    _spells(ctx)
    _spell_process(ctx)
    _contest(ctx)

    st.markdown("---")
    st.markdown("### Decisions on the simulator")
    decision_cards(D.by_topic(TOPIC))


# ── The generative chain ──────────────────────────────────────────────────────

def _chain(ctx: Ctx) -> None:
    st.markdown("### The generative chain, and what is built")

    rows = []
    for stage, quantity, likelihood, trials, artifact in CHAIN:
        # `artifact` is repo-relative, so build status is a live check rather than a
        # maintained list — this table doubles as the build tracker.
        built = (ROOT / artifact).exists()
        rows.append({
            "stage": stage, "quantity": quantity, "likelihood": likelihood,
            "exposure / trials": trials,
            "status": "✅ built" if built else "⬜ not built",
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.info(
        "**The posterior factorizes exactly, so the heads are fitted separately.** "
        "The chain has distinct parameter blocks and independent priors, which means "
        "eleven separate fits recover the *identical* posterior a joint fit would — "
        "an identity, not an approximation. `megamodel.stan` is the proof from this "
        "project's own history: no parameter was shared between any two heads, so it "
        "paid the full joint-fit price and ran on `sample_frac(0.01)`. **99% of the "
        "data was given up for a coupling that was not in the model.**")
    note("Which is why what remains is not a fitting problem but a *draw-time* one: "
         "the heads exist, and the missing pieces are the spell process and the "
         "composition step that turns eleven marginal posteriors into one correlated "
         "season.")


# ── Where correlation comes from ──────────────────────────────────────────────

def _correlation(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Where the correlation comes from")

    res = window(optional(ctx.eda("residual_correlation.csv"),
                          target="make residual-correlation"))
    if res is None:
        return

    conditioned = res[res["minutes_conditioned"] == True]        # noqa: E712
    raw = res[res["minutes_conditioned"] == False]               # noqa: E712
    off = conditioned[conditioned["component_a"] != conditioned["component_b"]]
    off_raw = raw[raw["component_a"] != raw["component_b"]]

    stat_tiles([
        ("Off-diagonal mean, minutes-conditioned", f"{off['r'].mean():+.4f}",
         "What is left after a shared minutes draw. Small — which is the finding."),
        ("Off-diagonal mean, raw", f"{off_raw['r'].mean():+.4f}"
         if len(off_raw) else "—",
         "Before conditioning. Two shot-volume counts both scale with the minutes "
         "they were accumulated over."),
        ("Largest conditioned cell", f"{off['r'].max():+.4f}",
         f"{off.loc[off['r'].idxmax(), 'component_a']}–"
         f"{off.loc[off['r'].idxmax(), 'component_b']}"),
        ("The 3PA/2PA substitution", f"{_pair(off, 'fg3a', 'fg2a'):+.4f}",
         "Negative, exactly as the reparameterization predicts."),
    ])

    if len(off_raw):
        ratio = off_raw["r"].mean() / max(off["r"].mean(), 1e-9)
        st.error(
            f"**Build the copula on the minutes-conditioned matrix, never the raw "
            f"one — a {ratio:.0f}× difference.** Building on the raw matrix would "
            f"impose {ratio:.0f}× the intended coupling *on top of* the shared "
            f"minutes draw that produced it. That is why `minutes_conditioned` is an "
            f"explicit column rather than an implicit filename convention.")

    grid = conditioned.pivot_table(index="component_a", columns="component_b",
                                   values="r")
    st.plotly_chart(
        fig_heatmap(grid, ctx.th,
                    "Residual cross-component correlation, minutes conditioned out",
                    "r", diverging=True, zmid=0.0, height=520,
                    hover="%{y} · %{x}<br>r %{z:+.3f}"),
        width="stretch")
    note("Both ordered pairs are emitted so a plain `pivot` returns a square matrix, "
         "and the long form is deliberate: the simulator reads this "
         "programmatically, and a wide matrix with component names as columns is a "
         "schema that breaks the moment a head is added. The matrix is symmetric with "
         "a diagonal of exactly 1.0 and a minimum eigenvalue comfortably positive, so "
         "it is usable as a copula input **without** a nearest-PSD correction.")
    st.info("**Draw `min` once per player-game and push it through all eleven heads "
            "as exposure.** Minutes are by far the largest common factor — 46% of the "
            "within-player residual. What is left over after that is small enough "
            "that a Gaussian copula on these residuals is a correction, not the main "
            "mechanism.")
    table_view(conditioned.round(4), "Conditioned matrix — table view")
    provenance("`make residual-correlation` → "
               "`outputs/eda/residual_correlation.csv`")


def _pair(frame: pd.DataFrame, a: str, b: str) -> float:
    hit = frame[(frame["component_a"] == a) & (frame["component_b"] == b)]
    return float(hit["r"].iloc[0]) if len(hit) else float("nan")


# ── Block inflation ───────────────────────────────────────────────────────────

def _block_inflation(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### What an independent-draws simulator would get wrong")

    serial = window(optional(ctx.eda("serial_correlation.csv"),
                             target="make serial-correlation"))
    if serial is None:
        return

    st.plotly_chart(
        fig_bars(serial.sort_values("block_inflation", ascending=False), "component",
                 ["block_inflation"], ctx.th,
                 "Ten-game block variance inflation, per component",
                 axis_title="× an independent-draws simulator", height=400,
                 emphasis="min"),
        width="stretch")
    note("**This is the decision-relevant column**: it is the factor by which drawing "
         "independently understates the variance of an *aggregate* — which is exactly "
         "what a season total and a weekly best-ball lineup are. Minutes at 2.43×, "
         "shot volume at ~1.46× on top of minutes, and the conversion heads at "
         "1.01–1.10×. **Put the sequential model on minutes, beside the availability "
         "spell process, and leave the other eleven heads collapsed.**")
    table_view(serial.round(4), "Serial correlation — table view")
    provenance("`make serial-correlation` → `outputs/eda/serial_correlation.csv`")


# ── The bonus ─────────────────────────────────────────────────────────────────

def _bonus(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The bonus needs the joint — and the per-game value is not the "
                "per-season one")

    cal = window(optional(ctx.eda("bonus_calibration.csv"),
                          target="make component-targets"))
    if cal is None:
        return

    # The artifact carries both units and a per-bucket break; the headline tiles are
    # the pooled player-season rows, which is the unit the shipped constant is for.
    season = cal[(cal["unit"] == "player_season") & (cal["bucket"] == "all")]
    game = cal[(cal["unit"] == "player_game") & (cal["bucket"] == "all")]
    indep = season[season["is_independent"]]
    shipped = season[season["is_shipped"]]
    game_best = game[game["expected_mean_bonus"].notna()].copy()

    tiles = []
    if len(indep):
        r = indep.iloc[0]
        tiles.append(("Independent sampling", f"{float(r['relative_bias']):.1%}",
                      f"E[bonus] {float(r['expected_mean_bonus']):.4f} against a "
                      f"realized {float(r['realized_mean_bonus']):.4f} — too low, "
                      f"because independence destroys the co-occurrence the "
                      f"threshold needs."))
    if len(shipped):
        r = shipped.iloc[0]
        tiles.append(("Shipped overdispersion (season unit)",
                      f"{float(r['overdispersion']):.3f}",
                      f"bias {float(r['bias']):+.4f} dk_pts/game — the season-unit "
                      f"optimum to two figures."))
    if len(game_best):
        r = game_best.loc[game_best["bias"].abs().idxmin()]
        tiles.append(("Best at the player-game unit",
                      f"{float(r['overdispersion']):.3f}",
                      f"bias {float(r['bias']):+.4f} — what the *simulator* needs, "
                      f"because it draws per game. Roughly 4× smaller, since minutes "
                      f"are no longer hidden inside the frailty."))
    if tiles:
        stat_tiles(tiles)

    st.error(
        "**The simulator needs overdispersion 0.025, not 0.10.** The shipped 0.10 is "
        "correct for its documented unit — the player *season* — and the aggregate "
        "calibration reproduces. But at the player-**game** unit the fitted optimum "
        "is 0.0248, and only there does the fit hold across *every* minutes bucket "
        "rather than only in aggregate. Using 0.10 per game over-predicts the bonus "
        "by **+0.036 dk_pts/game for 30+ minute players** — exactly the players it is "
        "worth most for.")
    note("`BONUS_OVERDISPERSION` is the variance of the shared per-game Gamma "
         "frailty, which does two jobs at once: it makes each category's marginal "
         "negative binomial *and* induces the positive dependence the bonus needs. It "
         "is not a dispersion of `dk_pts` and it is not fitted by any head. "
         "`BONUS_GAME_OVERDISPERSION = 0.025` sits beside it with the unit stated in "
         "both docstrings.")

    buckets = cal[(cal["unit"] == "player_game") & (cal["bucket"] != "all")
                  & cal["expected_mean_bonus"].notna()]
    if not buckets.empty:
        order = ["0-12", "12-18", "18-24", "24-30", "30-48"]
        grid = buckets.pivot_table(index="bucket", columns="overdispersion",
                                   values="bias")
        grid = grid.reindex([o for o in order if o in grid.index])
        st.plotly_chart(
            fig_heatmap(grid, ctx.th,
                        "Bonus bias by minutes bucket × overdispersion "
                        "(player-game unit)",
                        "bias (dk_pts/game)", diverging=True, zmid=0.0, height=320,
                        hover="%{y} min · overdispersion %{x}<br>bias %{z:+.4f}"),
            width="stretch")
        note("**At 0.10 the buckets do not fit — their errors offset.** Read the "
             "columns: at 0 every bucket is biased low, at 0.10 every bucket is "
             "biased high and the 30–48 minute bucket worst of all, and only at "
             "~0.025 is every bucket near zero at once. At the player-*season* unit "
             "the same test fails differently — the per-bucket bias runs from "
             "negative at low minutes to positive at high and cancels in aggregate — "
             "and the reason is structural rather than a tuning miss: one scalar "
             "frailty variance is standing in for minutes variation whose *relative* "
             "size differs by bucket.")

    with detail("Bonus calibration — every unit × bucket × overdispersion"):
        st.dataframe(cal.round(5), width="stretch", hide_index=True)
    provenance("`make component-targets` → `outputs/eda/bonus_calibration.csv`")


# ── The spell process ─────────────────────────────────────────────────────────

def _spells(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The spell process — the simple version is already falsified")

    prof = optional(ctx.eda("availability_profile.csv"),
                    target="make availability-profile")
    if prof is None:
        return

    def look(measurement: str, key: str, metric: str, window: str = "appearance"):
        hit = prof[(prof["measurement"] == measurement) & (prof["window"] == window)
                   & (prof["key"] == key) & (prof["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    stat_tiles([
        ("P(play | played)", f"{look('serial_structure', 'transition', 'p_play_given_played'):.3f}",
         "The stay-in state."),
        ("P(play | missed)", f"{look('serial_structure', 'transition', 'q_play_given_missed'):.3f}",
         "The recovery hazard."),
        ("Markov variance inflation",
         f"{look('serial_structure', 'markov', 'clustering_variance_inflation'):.2f}×",
         "What a 2-state chain generates — against ~20× measured."),
        ("Mean absence spell",
         f"{look('serial_structure', 'spell_shape', 'mean_spell_observed'):.2f} games",
         "Which the geometric model matches exactly, by construction."),
    ])

    shape = pd.DataFrame({
        "statistic": ["share of spells that are 1 game", "share of spells 10+ games"],
        "observed": [look("serial_structure", "spell_shape", "share_single_observed"),
                     look("serial_structure", "spell_shape", "share_ge10_observed")],
        "geometric": [look("serial_structure", "spell_shape", "share_single_geometric"),
                      look("serial_structure", "spell_shape", "share_ge10_geometric")],
    })
    st.plotly_chart(
        fig_bars(shape, "statistic", ["observed", "geometric"], ctx.th,
                 "Absence spells against a constant-hazard model",
                 axis_title="share of spells", height=280),
        width="stretch")
    st.error(
        "**A constant hazard implies geometric spells, which matches the mean and "
        "misses both tails.** Far more one-game absences and far more long ones than "
        "the model allows — so absences are a **mixture**. The shipped duration head is "
        "a **beta-geometric**: a geometric hazard with a Beta frailty integrated out, "
        "which beats the geometric by 11,278 log-likelihood points at one extra "
        "parameter.")
    st.info(
        "**⚠️ 'Clustering is only about a sixth of the overdispersion' was withdrawn on "
        "2026-08-05.** Two errors compounded: the composition is **additive**, "
        "`inflation = C + ρ(n − C)`, so dividing 22.7 by 3.96 decomposes nothing; and the "
        "two figures were measured on different windows *and* different populations. "
        "Matched — full window, established rotation players — the same population reads "
        "`P(play|played) = 0.9443` and `P(play|missed) = 0.1333`, giving **C = 9.58**. "
        "Clustering supplies **42%** of the budget, not 17%, and the practical consequence "
        "is the opposite of what the old reading implied: there is **no dispersion hole to "
        "fill, there is a surplus to avoid**.")
    note("Build it for the season-total joint distribution and the preseason initial "
         "state — not for games-played CRPS, which the beta-binomial head already "
         "handles.")

    spells = prof[(prof["measurement"] == "spell_distribution")
                  & (prof["window"] == "appearance")
                  & (prof["metric"] == "spell_games")]
    if not spells.empty:
        st.plotly_chart(
            fig_bars(spells.sort_values("key"), "key", ["value"], ctx.th,
                     "Absence-spell length percentiles", axis_title="games",
                     horizontal=False, height=280),
            width="stretch")
    table_view(prof[prof["measurement"].isin(["serial_structure",
                                              "spell_distribution"])].round(4),
               "Spell structure — table view")
    provenance("`make availability-profile` → `outputs/eda/availability_profile.csv` "
               "(`serial_structure`, `spell_distribution`)")


# ── The fitted spell process ──────────────────────────────────────────────────

def _spell_process(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The spell process, fitted — entry × exit × a within-tenure chain")

    gate = optional(ctx.predictions("stan_games_played_gate.csv"),
                    target="make games-played")
    if gate is None:
        return

    tail = gate[gate["metric"] == "p_below_41"].set_index("arm")
    plain = tail.loc["full_window_chain"] if "full_window_chain" in tail.index else None
    tenure = (tail.loc["tenure_decomposition"]
              if "tenure_decomposition" in tail.index else None)
    if plain is not None and tenure is not None:
        stat_tiles([
            ("Observed P(GP < 41)", f"{plain['observed']:.3f}",
             "Established rotation players, single-team seasons."),
            ("Plain full-window chain", f"{plain['simulated']:.3f}",
             f"{plain['simulated'] / plain['observed'] - 1:+.1%} — the left tail, "
             f"over-predicted. This arm is REJECTED."),
            ("Tenure decomposition", f"{tenure['simulated']:.3f}",
             f"{tenure['simulated'] / tenure['observed'] - 1:+.1%} — the shipped "
             f"process class."),
        ])

    st.error(
        "**A departure is an absorbing hitting time, not a low recovery rate.** A waived "
        "player's cell has a recovery hazard of about zero, so a recurrent two-state "
        "chain makes him absorbing from his *first* absence rather than from the game he "
        "was actually cut — relocating the departure earlier in the season and dragging "
        "the left tail out with it. Gate 0 measures that at its most generous possible "
        "parameterization, each cell running at its own observed hazards, so the failure "
        "is a property of the **process class** and no feature block can rescue it.")
    st.info(
        "**The fix keeps all 30 seasons.** `team_games = pre-tenure + tenure + "
        "post-tenure` and `gp` = games played inside the tenure, an identity that "
        "reconstructs full-window `gp_share` with no residual. Each factor answers a "
        "separate question — when he joined, when he stopped, how often he missed while "
        "there — and it is identifiable structurally from `in_appearance_window`, which "
        "exists on every season. Two things fall out free: within a tenure every absence "
        "spell is **interior**, so the duration head needs no censoring branch; and the "
        "initial state is **known**, which is the one thing the collapse to sufficient "
        "statistics cannot tell you.")

    table_view(gate.round(4), "Gate 0 — simulated against observed, per arm")
    provenance("`make games-played` → `outputs/predictions/stan_games_played_gate.csv`")

    spells = optional(ctx.predictions("stan_games_played_spells.csv"),
                      target="make games-played")
    if spells is not None:
        bias = spells[spells["analysis"] == "censoring_bias"]
        if not bias.empty:
            st.markdown("#### Censoring is 8.8% of spells and 28.6% of the missed games")
            st.plotly_chart(
                fig_bars(bias, "treatment", ["p_ge_26"], ctx.th,
                         "P(spell ≥ 26 games) under three treatments of censoring",
                         axis_title="probability", height=300),
                width="stretch")
            note("**Both wrong answers fail silently.** Dropping censored spells "
                 "understates the extreme tail by ~2×; treating them as complete still "
                 "understates it. Neither raises, neither looks wrong, and both change "
                 "exactly the tail the head exists to get right. With proper censoring "
                 "the fitted `a` falls below 1, so the beta-geometric has **no finite "
                 "mean** — the simulator does not care, because a spell is truncated by "
                 "the remaining schedule anyway, but `E[T]` must never be quoted from "
                 "that fit.")
        classes = spells[spells["analysis"] == "spell_class"]
        if not classes.empty:
            table_view(classes.round(4), "Spell classes — table view")
        provenance("`make games-played` → "
                   "`outputs/predictions/stan_games_played_spells.csv`")

    metrics = optional(ctx.predictions("stan_games_played_metrics.csv"),
                       target="make stan-games-played")
    if metrics is None:
        return
    st.markdown("#### The arm ladder")
    st.dataframe(metrics.round(4), width="stretch", hide_index=True)
    note("**Selection reads the validation column only.** This head has already produced "
         "one test-selected false positive and caught a second, so the test column "
         "confirms and never decides. `floor` is the incumbent season-level "
         "beta-binomial and stays permanently as the bar; `within_tenure` holds the "
         "tenure at its observed value, so it is an oracle and cannot ship however well "
         "it scores.")

    gates = optional(ctx.predictions("stan_games_played_gates.csv"),
                     target="make stan-games-played")
    if gates is not None:
        table_view(gates.round(4), "Gates A–D — table view")
    process = optional(ctx.predictions("spell_process.csv"),
                       target="make stan-games-played")
    if process is not None:
        table_view(process.round(4),
                   "What the simulator is GIVEN — spell_process.csv")
        provenance("`make stan-games-played` → `outputs/predictions/spell_process.csv`, "
                   "`stan_games_played_metrics.csv`, `stan_games_played_gates.csv`, "
                   "`stan_games_played_coefficients.csv`, "
                   "`stan_games_played_diagnostics.csv`, `stan_games_played_pit.csv`, "
                   "`stan_games_played_predictions.csv`, `stan_games_played_gp_pmf.csv`, "
                   "`stan_games_played_collapse.csv`")


# ── What the contest requires ─────────────────────────────────────────────────

def _contest(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### What the contest requires the simulator to implement")

    left, right = st.columns(2)
    with left:
        st.markdown(
            "- **16-man frozen roster**, from ≥2 NBA teams, no in-season management\n"
            "- **Best 7 of 16 by slot each week** — 2 G / 2 F / 1 C / 2 UTIL\n"
            "- A scoring period runs **first game through last game** of the week's "
            "game set\n"
            "- A suspended game scores in the period it is **played**\n"
            "- **Four rounds**, no redraft between them\n"
            "- Cascading tie-breaks")
    with right:
        st.info(
            "**The weekly best-7-of-16 is why the simulator cannot be a set of "
            "marginals.** A max over a correlated set is not a function of the "
            "marginals — two players who miss the same week are a very different "
            "roster from two who miss different weeks, at identical marginal "
            "availability. That is the same argument as the bonus threshold, one "
            "level up.")
    st.warning("**Never plug in `E[min]` or `E[gp]` — draw them. And never cap "
               "minutes at 48.**")
    note("**Validation plan:** posterior-predictive checks on held-out team-total "
         "variance and same-team pairwise covariance — *not* point accuracy. A "
         "simulator that gets the mean right and the covariance wrong would score "
         "well on MAE and lose money, because the payout is a knockout on rank.")
    provenance("`docs/dk_best_ball_rules.md`; `docs/simulations-plan.md`")
