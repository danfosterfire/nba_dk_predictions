"""Tab 5 · The minutes head.

Small, self-contained, and the most instructive contrast in the project: its
specification answer is the **opposite** of the component heads'. There, scale is
everything and curvature is nearly nothing. Here the logit scale is a dead wash and
curvature is what pays — and unlike the games-played arm, it replicates on both
splits.
"""

import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx, optional
from dashboard.charts import fig_bars, fig_lines
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "minutes"


def render(ctx: Ctx) -> None:
    tab_header(
        "The minutes head",
        "`min | available` — drawn conditional on the player being available, then "
        "pushed through all eleven component heads as exposure. It is the largest "
        "single common factor in the model, which is why it gets its own head rather "
        "than being folded into the rates.")

    _why_separate(ctx)
    _trials_denominator(ctx)
    _variant_ladder(ctx)
    _where_the_curvature_is(ctx)
    _dispersions(ctx)

    st.markdown("---")
    st.markdown("### Decisions on this head")
    decision_cards(D.by_topic(TOPIC))


# ── Why minutes gets its own head ─────────────────────────────────────────────

def _why_separate(ctx: Ctx) -> None:
    st.markdown("### Why it is separate")

    budget = optional(ctx.eda("variance_budget.csv"), target="make variance-budget")
    if budget is None:
        return
    own = budget[budget["source"] == "own_minutes"]
    if own.empty:
        return

    def share(source: str) -> float:
        hit = budget[budget["source"] == source]
        return float(hit["share_of_variance"].iloc[0]) if len(hit) else float("nan")

    stat_tiles([
        ("Own minutes, share of residual", f"{share('own_minutes'):.1%}",
         "Of within-player-season residual variance in dk_pts, conditioning on the "
         "within-player minutes *deviation* — the contrast the residual is defined "
         "by."),
        ("Nonparametric check", f"{share('own_minutes_nonparametric'):.1%}",
         "Cell means over 2-minute bins. Agreeing with the linear row rules out a "
         "linearity artifact."),
        ("Saturated upper bound", f"{share('own_minutes_saturated'):.1%}",
         "Cells are (player-season × minutes bin) — a per-player minutes profile, "
         "the nonparametric ceiling."),
        ("Superseded construction", f"{share('own_minutes_raw_level'):.1%}",
         "Conditioning on the raw minutes *level*, pooled across players. Attenuated "
         "by more than half."),
    ])
    st.info(
        "**The long-recorded figure was 18.6%, and it turned out to be the wrong "
        "construction.** Conditioning on the raw minutes level pools a 30-minute game "
        "that is *below* average for a 34-mpg starter with one far *above* average "
        "for an 18-mpg reserve, so their residuals cancel inside the cell. The "
        "superseded row ships beside the corrected one rather than being quietly "
        "overwritten. Direction is safe: minutes matter **more** than the project "
        "believed, which strengthens every argument built on drawing `min` once and "
        "pushing it through all eleven heads.")
    table_view(budget[budget["source"].str.startswith("own_minutes")]
               [["source", "share_of_variance", "basis", "n_games", "note"]].round(4),
               "Own-minutes constructions — table view")
    provenance("`make variance-budget` → `outputs/eda/variance_budget.csv`")


# ── The trials denominator ────────────────────────────────────────────────────

def _trials_denominator(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The trials denominator is derivable exactly — never 48")

    cov = optional(ctx.eda("game_length_coverage.csv"), target="make game-length")
    if cov is None:
        return

    deriv = cov[cov["analysis"] == "derivation"]
    games = int(deriv["games"].sum())
    disagree = int(deriv["teams_disagree"].sum())
    worst = float(deriv["max_abs_residual"].max())
    ot = float((deriv["ot_rate"] * deriv["games"]).sum() / deriv["games"].sum())

    stat_tiles([
        ("Games derived", f"{games:,}",
         "Regular season and playoffs, 1996-97 → 2025-26. Length is a property of a "
         "game rather than of a target, so both season types are covered."),
        ("Team disagreements", f"{disagree}",
         "Five players are on court at every moment, so a team's summed minutes are "
         "exactly 5 × game length. **The two teams agreeing is the validation.**"),
        ("Worst rounding residual", f"{worst:.3f} min",
         "Against a 2.5-minute decision boundary between grid points."),
        ("Overtime rate", f"{ot:.2%}",
         "Capping at 48 would discard these games and censor the top of the minutes "
         "distribution exactly where stars play most."),
    ])

    regular = deriv[deriv["season_type"] == "regular"].sort_values("season")
    st.plotly_chart(
        fig_lines(regular, "season", {"ot_rate": "overtime rate"}, ctx.th,
                  "Share of games going to overtime, by season",
                  y_title="OT rate", x_title="season", height=320, label_last=False),
        width="stretch")

    feas = cov[(cov["analysis"] == "feasibility") & (cov["season"] == "all")]
    if not feas.empty:
        f = feas.iloc[0]
        st.success(
            f"**The specification is feasible on every row, which is the check that "
            f"matters.** Joining game length to the component targets covers "
            f"**{float(f['join_coverage']):.1%}** of {int(f['player_games']):,} "
            f"player-games and yields **{int(f['violations'])}** rows with "
            f"`min > game_length`; the maximum ratio is exactly "
            f"**{float(f['max_min_over_length']):.4f}**, a player who played every "
            f"minute. So `min ~ Binomial(game_length, ·)` is well posed everywhere, "
            f"with no clipping and no boundary hack. "
            f"{int(f['player_games_above_regulation']):,} player-games exceed 48 "
            f"minutes and the observed maximum is {float(f['max_minutes']):.1f}.")
        note("The zero-violations row is **asserted, not reported** — a nonzero count "
             "is a build failure. The assertion has two arms because they fail "
             "differently and both are otherwise silent: a violation means the "
             "binomial support is wrong, and an *unmatched* player-game means the "
             "join is broken, which is the `pad_game_id` trap and would show up only "
             "as a quietly smaller fitting frame.")

    with detail("Game length — derivation and feasibility, per season"):
        st.dataframe(cov.round(4), width="stretch", hide_index=True)
    provenance("`make game-length` → `outputs/eda/game_length_coverage.csv`, "
               "`data/features/game_length.parquet`")


# ── The variant ladder ────────────────────────────────────────────────────────

def _variant_ladder(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The variant ladder against the no-fit floor")
    note("Season-collapsed: `y` is season minutes and `n` is summed game length "
         "**over the games he played**, so this is the conditional `min | available` "
         "and it composes with the availability head rather than double-counting "
         "absences. Zero rows were clamped by rounding.")

    m = optional(ctx.predictions("stan_minutes_metrics.csv"),
                 target="make stan-minutes")
    if m is None or m.empty:
        return

    floor = m[m["variant"] == "carry_forward"].iloc[0]
    best = m[m["selected"]].iloc[0] if m["selected"].any() else m.iloc[-1]

    stat_tiles([
        ("Selected variant", str(best["variant"]),
         "Chosen on **validation** CRPS, per the lesson from the games-played arm."),
        ("Test CRPS", f"{float(best['test_crps']):.2f}",
         f"Against the no-fit floor's {float(floor['test_crps']):.2f} — "
         f"{float(best['test_crps']) - float(floor['test_crps']):+.1f} minutes."),
        ("Test R²", f"{float(best['test_r2']):.4f}",
         f"Against the floor's {float(floor['test_r2']):.4f} — "
         f"{float(best['test_r2']) - float(floor['test_r2']):+.4f}."),
        ("Clears the floor", "yes" if bool(best["beats_floor"]) else "no",
         "Every head is quoted against a no-fit carry-forward baseline."),
    ])

    st.plotly_chart(
        fig_bars(m, "variant", ["val_crps", "test_crps"], ctx.th,
                 "CRPS by variant and split", axis_title="CRPS (minutes)",
                 height=340),
        width="stretch")

    linear = m[m["variant"] == "linear"].iloc[0]
    logit = m[m["variant"] == "logit_own"].iloc[0]
    st.info(
        f"**The specification answer is the OPPOSITE of the count heads', and that is "
        f"the finding.** There, scale is everything and curvature is nearly nothing. "
        f"Here the logit scale is a **dead wash** — test R² "
        f"{float(logit['test_r2']):.4f} against linear's "
        f"{float(linear['test_r2']):.4f}, and it is *worse* on validation CRPS — "
        f"while **curvature is what pays**. Both splits move the same way, so unlike "
        f"the games-played arm this replicates. Do not generalize 'put it on the "
        f"link's scale' from the counts to the minutes head.")

    st.warning(
        f"**`open` defect — the fitted heads carry a held-out bias the floor does "
        f"not.** {float(best['test_bias']):.1f} minutes against the floor's "
        f"{float(floor['test_bias']):.1f}, about −2.7% on a ~1,500-minute mean. It is "
        f"the price of shrinkage on a held-out season: the floor is unbiased because "
        f"it does not shrink. It costs nothing on R², MAE or CRPS here, but it is a "
        f"real calibration defect and **it would compound through the eleven "
        f"component heads that take these minutes as exposure**. Worth a bias "
        f"correction before the simulator consumes it.")

    diag = optional(ctx.predictions("stan_minutes_diagnostics.csv"),
                    target="make stan-minutes")
    if diag is not None:
        note(f"Sampler: max R̂ {diag['max_rhat'].max():.4f}, "
             f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
             f"{diag['wall_clock_s'].sum():.0f} s total. The spline variants cost "
             f"several times what the linear ones do — **B-spline bases are badly "
             f"conditioned for HMC**, which is valid but expensive; an orthogonalized "
             f"basis is the fix if a spline variant ever becomes the shipped spec.")
        with detail("Sampler diagnostics per fit"):
            st.dataframe(diag.round(4), width="stretch", hide_index=True)

    table_view(m.round(4), "Variant ladder — table view")
    provenance("`make stan-minutes` → `outputs/predictions/stan_minutes_metrics.csv`, "
               "`outputs/predictions/stan_minutes_diagnostics.csv`")


# ── Where the curvature actually is ───────────────────────────────────────────

def _where_the_curvature_is(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Where the curvature is — a floor at the bottom, not a ceiling "
                "at the top")

    probe = optional(ctx.predictions("availability_minutes_nonlinearity.csv"),
                     target="make availability-model")
    if probe is None:
        return

    cols = probe[probe["scope"] == "column"].sort_values("val_vs_linear",
                                                         ascending=False)
    st.plotly_chart(
        fig_bars(cols, "name", ["val_vs_linear", "test_vs_linear"], ctx.th,
                 "Δ R² from splining one column at a time, next-season MPG",
                 axis_title="Δ R² against the linear spec", height=420),
        width="stretch")
    note("Splining one column at a time attributes the gain. Only "
         "**`minutes_per_game_lag1`** moves both splits the same way, and "
         "`total_minutes_lag1` marginally. A spline on **`age` is actively worse**.")

    st.info(
        "**So the intuitive story is real but already absorbed.** Young players ramp "
        "up, prime players play heavy minutes, veterans get load-managed — MPG peaks "
        "at 27 and falls to 0.515 by 37 — but `age + age_sq` in the linear baseline "
        "already carries it. The nonlinearity that pays is a **floor at the bottom of "
        "the prior-MPG range**: mean next-season MPG runs 4.3 → 10.5 (**+6.1**), "
        "9.3 → 12.4 (+3.1), 15.1 → 15.8 (+0.7), then a roughly parallel decline of "
        "−2.0 from 26 mpg up. Part of that is survivorship — a 4-mpg player needs a "
        "next-season row to appear at all — the same bias that makes cross-sectional "
        "age curves worthless here.")

    with detail("Variant sweep and per-column probe — table view"):
        st.dataframe(probe.round(4), width="stretch", hide_index=True)
    provenance("`make availability-model` → "
               "`outputs/predictions/availability_minutes_nonlinearity.csv`")


# ── Two dispersions, and the one the fit does not estimate ────────────────────

def _dispersions(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The simulator needs three minutes numbers, and they compose")

    disp = optional(ctx.predictions("stan_minutes_dispersion.csv"),
                    target="make stan-minutes")
    serial = optional(ctx.eda("serial_correlation.csv"),
                      target="make serial-correlation")
    if disp is None or serial is None:
        return

    season_rho = float(disp[disp["metric"] == "season_level_rho"]["rho"].iloc[0])
    game = disp[disp["metric"] == "game_level_rho"].iloc[0]
    block = float(serial[serial["component"] == "min"]["block_inflation"].iloc[0])

    stat_tiles([
        ("1 · Season-level ρ", f"{season_rho:.4f}",
         "What the head fits — the mean level for a player-season."),
        ("2 · Game-level ρ", f"{float(game['rho']):.4f}",
         f"{float(game['implied_overdispersion']):.2f}× binomial at a 48-minute "
         f"game: the marginal spread of a *single* game, over "
         f"{int(game['n_player_games']):,} player-games."),
        ("3 · Ten-game block inflation", f"{block:.2f}×",
         "The serial dependence *between* games — the factor by which an "
         "independent-draws simulator understates the variance of an aggregate."),
    ])

    st.error(
        f"**The season-level ρ is NOT the number the simulator needs, and they differ "
        f"by more than the fit does.** A season total cannot separate a per-game "
        f"random effect from a per-season one: iid game noise is diluted by ~1/G "
        f"while a shared season multiplier passes through in full. Drawing per-game "
        f"minutes from {season_rho:.4f} would make every simulated game far too close "
        f"to the player's average. The three numbers **compose, they do not "
        f"substitute** — using only the game-level ρ gives independent draws with the "
        f"right marginal and too little variance in any aggregate; using only the "
        f"block inflation gets the clustering right and each game wrong.")
    note("The game-level figure is in-sample against each player-season's own mean, "
         "and is therefore a **floor**.")

    st.markdown("#### Serial structure per component — minutes is the outlier")
    st.plotly_chart(
        fig_bars(serial.sort_values("block_inflation", ascending=False), "component",
                 ["block_inflation"], ctx.th, "Ten-game block variance inflation",
                 axis_title="× an independent-draws simulator", height=380,
                 emphasis="min"),
        width="stretch")
    note("Minutes at 2.43× against ~1.01–1.10× for the conversion heads. **There is "
         "no shooting hot hand**, so the successes/trials heads collapse for free — "
         "what is autocorrelated is the *exposure*, not the conversion. Decay is "
         "slower than AR(1), and removing a within-season linear trend drops lag-1 "
         "substantially, so roughly a third is slow role drift and two-thirds a shock "
         "with a 3–5 game e-folding. Rotation churn and injury ramps, not form.")
    table_view(serial.round(4), "Serial correlation — table view")
    provenance("`make stan-minutes` → "
               "`outputs/predictions/stan_minutes_dispersion.csv`; "
               "`make serial-correlation` → `outputs/eda/serial_correlation.csv`")
