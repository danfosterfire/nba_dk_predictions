"""Tab 4 · The availability head.

The most complete story in the project: measurement → ceiling → baselines →
downstream value → Stan port. Fully live-backed, no gated panels.

Games played is simultaneously the **largest lever on the season total and the least
predictable input**, which is the whole reason this head exists and the reason it
emits a distribution rather than a point estimate.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx, optional
from dashboard.charts import fig_bars, fig_lines
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)
from dashboard.theme import apply_theme

TOPIC = "availability"

PROFILE = "availability_profile.csv"
WINDOW = "full"          # the window CLAUDE.md quotes; the appearance twin sits beside it


def _lookup(prof: pd.DataFrame, measurement: str, key: str, metric: str,
            window: str = WINDOW) -> float:
    """One value out of the long profile artifact, or NaN if the row is absent."""
    hit = prof[(prof["measurement"] == measurement) & (prof["window"] == window)
               & (prof["key"] == key) & (prof["metric"] == metric)]
    return float(hit["value"].iloc[0]) if len(hit) else float("nan")


def _metric_frame(metrics: pd.DataFrame, metric: str,
                  group: str = "all") -> pd.DataFrame:
    sub = metrics[(metrics["metric"] == metric) & (metrics["group"] == group)]
    return sub[["model", "value"]].reset_index(drop=True)


def _value(frame: pd.DataFrame, **where) -> float:
    mask = pd.Series(True, index=frame.index)
    for col, val in where.items():
        mask &= frame[col] == val
    hit = frame[mask]
    return float(hit["value"].iloc[0]) if len(hit) else float("nan")


def render(ctx: Ctx) -> None:
    tab_header(
        "The availability head",
        "How many of his team's games a player is available for. It is the largest "
        "lever on the season total and the least predictable input the project has — "
        "which is why the head emits a distribution, not a point estimate.")

    prof = optional(ctx.eda(PROFILE), target="make availability-profile")
    if prof is not None:
        _why_it_matters(ctx, prof)
        _overdispersion(ctx, prof)
        _windows_and_nulls(ctx, prof)
        _ceiling(ctx, prof)

    _baselines(ctx)
    _season_total(ctx)
    _ablations(ctx)

    if prof is not None:
        _reasons(ctx, prof)

    _stan(ctx)
    _report_calibration(ctx)

    st.markdown("---")
    st.markdown("### Decisions on this head")
    decision_cards(D.by_topic(TOPIC))


# ── Why it matters ────────────────────────────────────────────────────────────

def _why_it_matters(ctx: Ctx, prof: pd.DataFrame) -> None:
    st.markdown("### Why it needs its own head")

    rows = ["gp_share", "minutes_per_game", "total_minutes", "available_rate",
            "n_spells", "longest_spell"]
    pers = pd.DataFrame({
        "quantity": rows,
        "r_unweighted": [_lookup(prof, "persistence", r, "r_within_unweighted")
                         for r in rows],
        "r_weighted": [_lookup(prof, "persistence", r, "r_within_weighted")
                       for r in rows],
    })

    stat_tiles([
        ("Games played, year over year",
         f"{_lookup(prof, 'persistence', 'gp_share', 'r_within_weighted'):.3f}",
         "Season-absorbed within-season r. The least persistent quantity in the "
         "project."),
        ("Minutes per game",
         f"{_lookup(prof, 'persistence', 'minutes_per_game', 'r_within_weighted'):.3f}",
         "The same measurement on the rate side, for contrast."),
        ("Longest absence spell",
         f"{_lookup(prof, 'persistence', 'longest_spell', 'r_within_weighted'):.3f}",
         "There is no durability latent to extract — see the nulls below."),
    ])
    note("Availability barely persists while the rate side persists strongly. That "
         "asymmetry is the argument for shrinking the head hard toward a league/age "
         "baseline rather than toward the player's own prior games played.")

    st.plotly_chart(
        fig_bars(pers.sort_values("r_weighted"), "quantity",
                 ["r_weighted", "r_unweighted"], ctx.th,
                 "Year-over-year persistence of the availability quantities",
                 axis_title="r (season-absorbed)", height=360),
        width="stretch")
    note("**Unweighted is the column to read here** — the exception to this project's "
         "standing minutes-weighting rule. Weighting exists to suppress per-36 rates "
         "measured over a few garbage-time minutes, which is real measurement error. "
         "'He played 12 games' has none, so weighting would down-weight exactly the "
         "injured seasons the head exists to predict.")
    table_view(pers.round(4), "Persistence — table view")
    provenance("`make availability-profile` → `outputs/eda/availability_profile.csv` "
               "(`persistence`)")


# ── Overdispersion ────────────────────────────────────────────────────────────

def _overdispersion(ctx: Ctx, prof: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("### Games played is ~20× overdispersed against a binomial")

    ratio = _lookup(prof, "overdispersion", "rotation_players", "variance_ratio")
    stat_tiles([
        ("Variance ratio", f"{ratio:.1f}×",
         "Observed variance in games-played share against what a binomial with the "
         "same mean would give. A point estimate cannot represent this."),
        ("Mean games-played share",
         f"{_lookup(prof, 'overdispersion', 'rotation_players', 'mean_gp_share'):.3f}",
         "Established rotation players only."),
        ("Below 60 games",
         f"{_lookup(prof, 'overdispersion', 'rotation_players', 'share_below_60_games'):.1%}",
         "Of established rotation players — the left tail the histogram shows."),
        ("Below 41 games",
         f"{_lookup(prof, 'overdispersion', 'rotation_players', 'share_below_41_games'):.1%}",
         "Missing more than half a season."),
    ])

    bins = prof[(prof["measurement"] == "overdispersion")
                & (prof["window"] == WINDOW)
                & (prof["metric"] == "count")
                & (prof["key"].str.startswith("gp_"))].copy()
    if not bins.empty:
        bins["low"] = bins["key"].str.extract(r"gp_(\d+)_")[0].astype(int)
        bins = bins.sort_values("low")
        bins["games"] = bins["key"].str.removeprefix("gp_").str.replace("_", "–")
        st.plotly_chart(
            fig_bars(bins, "games", ["value"], ctx.th,
                     "Games played, established rotation players",
                     axis_title="player-seasons", horizontal=False, height=340),
            width="stretch")
        note("A mode at the top of the range with a long left tail. The mass on the "
             "left is the part a point estimate cannot carry, and it is where the "
             "season total is decided.")
        table_view(bins[["games", "value"]].rename(columns={"value": "player_seasons"}),
                   "Histogram — table view")
    provenance("`make availability-profile` → `availability_profile.csv` "
               "(`overdispersion`)")


# ── The two windows, and the two nulls ────────────────────────────────────────

def _windows_and_nulls(ctx: Ctx, prof: pd.DataFrame) -> None:
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### The two roster windows bracket the truth")
        appearance = _lookup(prof, "window_bracket", "panel", "played_rate",
                             window="appearance")
        full = _lookup(prof, "window_bracket", "panel", "played_rate", window="full")
        stat_tiles([
            ("Appearance window", f"{appearance:.3f}",
             "First to last appearance — blind to a season-ending absence, so it "
             "reads too high."),
            ("Full window", f"{full:.3f}",
             "Every team game — counts 'not on an NBA roster' as 'missed', so it "
             "reads too low."),
        ])
        note("Neither window resolves it, and that is the honest statement: the "
             "played rate is bracketed, not measured. Every figure on this tab ships "
             "with both windows in its table view.")

    with right:
        st.markdown("#### Two nulls — there is no durability latent")
        one = _lookup(prof, "multiyear", "gp_share_lag1", "r_unweighted")
        three = _lookup(prof, "multiyear", "gp_share_mean3", "r_unweighted")
        spell = _lookup(prof, "persistence", "longest_spell", "r_within_unweighted")
        stat_tiles([
            ("1-year prior", f"{one:.3f}", "Prior season's games-played share."),
            ("3-year average", f"{three:.3f}",
             "A longer history does not beat one year."),
            ("Longest spell", f"{spell:.3f}",
             "Persistence of a player's worst absence."),
        ])
        note("Averaging three years of availability does **not** beat one, and a "
             "player's longest absence barely predicts his next one. Both are "
             "recorded so a 'durability score' is not rebuilt.")

    provenance("`make availability-profile` → `availability_profile.csv` "
               "(`window_bracket`, `multiyear`, `persistence`)")


# ── The in-sample ceiling ─────────────────────────────────────────────────────

def _ceiling(ctx: Ctx, prof: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("### The in-sample ceiling ladder")

    ladder = prof[(prof["measurement"] == "predictor_r2")
                  & (prof["window"] == WINDOW)
                  & (prof["metric"] == "r2_in_sample_unweighted")].copy()
    if ladder.empty:
        return
    order = ["prior_gp_share", "prior_mpg", "gp_lags_1_3", "gp_and_mpg_lag1",
             "gp_and_mpg_lags_1_3", "plus_age_and_career"]
    ladder["rank"] = ladder["key"].map({k: i for i, k in enumerate(order)})
    ladder = ladder.sort_values("rank")

    st.plotly_chart(
        fig_bars(ladder, "key", ["value"], ctx.th,
                 "In-sample R² on next-season games-played share",
                 axis_title="R² (unweighted, season-absorbed)", height=320,
                 emphasis="plus_age_and_career"),
        width="stretch")

    gp_only = _lookup(prof, "predictor_r2", "prior_gp_share",
                      "r2_in_sample_unweighted")
    mpg_only = _lookup(prof, "predictor_r2", "prior_mpg", "r2_in_sample_unweighted")
    note(f"Everything the project knows about a player's history reaches "
         f"**R² ≈ {ladder['value'].max():.2f}** in sample — a ceiling, not an "
         f"achievable target, and *not* comparable to the head's held-out R². The "
         f"striking row is the first pair: prior minutes per game predicts "
         f"next-season games played ({mpg_only:.3f}) about as well as prior games "
         f"played does ({gp_only:.3f}). Availability is closer to a function of role "
         f"than of durability.")
    table_view(ladder[["key", "value", "n"]].round(4), "Ceiling ladder — table view")
    provenance("`make availability-profile` → `availability_profile.csv` "
               "(`predictor_r2`)")


# ── Baselines ─────────────────────────────────────────────────────────────────

def _baselines(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The baseline ladder, and the stopping rule it triggered")

    metrics = optional(ctx.predictions("availability_metrics.csv"),
                       target="make availability-model")
    if metrics is None:
        return

    crps = _metric_frame(metrics, "crps_games").sort_values("value")
    stat_tiles([
        ("Beta-binomial GLM", f"{_value(crps, model='beta_binomial'):.3f}",
         "Held-out CRPS in games (2024-25 / 2025-26). Lower is better."),
        ("Gradient boosting", f"{_value(crps, model='gbm'):.3f}",
         "A fully nonparametric learner on the same features."),
        ("Ridge", f"{_value(crps, model='ridge'):.3f}", "A linear point predictor."),
        ("League/age baseline", f"{_value(crps, model='league_age'):.3f}",
         "No player history at all."),
    ])
    st.plotly_chart(
        fig_bars(crps, "model", ["value"], ctx.th,
                 "Held-out CRPS in games", axis_title="CRPS (games)", height=280,
                 emphasis="beta_binomial"),
        width="stretch")
    note("**Gradient boosting does not beat a 19-feature GLM, so the plan's own "
         "stopping rule says stop.** Two things confirm the *distribution* is the "
         "working part rather than the mean: the fitted dispersion independently "
         "recovers the ~20× overdispersion measured above, and PIT is near-uniform.")

    left, right = st.columns([3, 2])
    with left:
        pit = optional(ctx.predictions("availability_pit.csv"),
                       target="make availability-model")
        if pit is not None:
            fig = go.Figure()
            for i, model in enumerate(["beta_binomial", "gbm", "ridge", "league_age"]):
                g = pit[pit["model"] == model]
                if g.empty:
                    continue
                fig.add_bar(x=g["bin_low"] + 0.05, y=g["share"], name=model,
                            marker=dict(color=ctx.th["series"][i],
                                        line=dict(color=ctx.th["surface"], width=2)),
                            hovertemplate=f"{model}<br>PIT %{{x:.2f}} · "
                                          "%{y:.1%}<extra></extra>")
            fig.add_hline(y=0.1, line=dict(color=ctx.th["axis"], width=2))
            fig.update_layout(title="PIT histogram — flat is calibrated", bargap=0.2)
            fig.update_xaxes(title="probability integral transform", showgrid=False)
            fig.update_yaxes(title="share of held-out player-seasons",
                             tickformat=".0%")
            st.plotly_chart(apply_theme(fig, ctx.th, 340), width="stretch")
            table_view(pit.round(4), "PIT — table view")
    with right:
        ks = _metric_frame(metrics, "pit_ks_distance").sort_values("value")
        st.plotly_chart(
            fig_bars(ks, "model", ["value"], ctx.th,
                     "PIT deviation from uniform (KS)", axis_title="KS distance",
                     height=280, emphasis="beta_binomial"),
            width="stretch")
        disp = _metric_frame(metrics, "implied_overdispersion")
        note("Implied overdispersion recovered by each fit — " + " · ".join(
            f"{r.model} {r.value:.1f}×" for r in disp.itertuples()))

    with detail("Every held-out metric, all four models"):
        st.dataframe(metrics.round(4), width="stretch", hide_index=True)
    provenance("`make availability-model` → "
               "`outputs/predictions/availability_metrics.csv`, "
               "`outputs/predictions/availability_pit.csv`")


# ── What it is worth on the deliverable ───────────────────────────────────────

def _season_total(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### What the head is worth on the actual deliverable")
    note("`season_total = gp × dk_per_game_played`, holding a **fixed** rate model "
         "and varying only the games-played treatment — so the contrast is the head "
         "and nothing else. The two oracle rows replace one factor with its realized "
         "value, which is what settles which half of the remaining error is larger.")

    st_metrics = optional(ctx.predictions("season_total_metrics.csv"),
                          target="make season-total")
    if st_metrics is None:
        return

    mae = st_metrics[(st_metrics["metric"] == "mae_dk_total")
                     & (st_metrics["group"] == "all")]
    order = ["full_season", "prior_gp", "league_age", "beta_binomial",
             "oracle_rate", "oracle_gp"]
    mae = mae.set_index("treatment").reindex(
        [o for o in order if o in set(mae["treatment"])]).reset_index()

    naive = _value(mae, treatment="full_season")
    head = _value(mae, treatment="beta_binomial")
    prior = _value(mae, treatment="prior_gp")
    o_gp = _value(mae, treatment="oracle_gp")
    o_rate = _value(mae, treatment="oracle_rate")
    bias = _value(st_metrics[st_metrics["metric"] == "bias_dk_total"],
                  treatment="beta_binomial")
    naive_bias = _value(st_metrics[st_metrics["metric"] == "bias_dk_total"],
                        treatment="full_season")

    stat_tiles([
        ("Against a full season", f"−{naive - head:.1f}",
         f"dk_pts of season-total MAE, {(head - naive) / naive:.1%}. The naive "
         f"treatment assumes 82 games for everyone."),
        ("Against carrying prior GP", f"−{prior - head:.1f}",
         "dk_pts of MAE against the obvious alternative."),
        ("Head bias", f"{bias:+.1f}",
         f"Near zero — the distribution doing its job. The naive treatment's "
         f"{naive_bias:+.1f} is most of its error."),
    ])

    st.plotly_chart(
        fig_bars(mae, "treatment", ["value"], ctx.th,
                 "Season-total MAE by games-played treatment",
                 axis_title="MAE (dk_pts)", height=340, emphasis="beta_binomial"),
        width="stretch")
    note(f"**The oracles settle which half dominates.** Perfect games played gives "
         f"{o_gp:.1f} MAE against perfect rate's {o_rate:.1f}, so availability "
         f"carries {o_rate - o_gp:.1f} dk_pts more of the remaining error than the "
         f"rate does. `oracle_gp` is invariant to the head by construction, which "
         f"makes it the internal check that a change to the GP treatment moved only "
         f"what it should.")

    rot = st_metrics[(st_metrics["metric"] == "mae_dk_total")
                     & (st_metrics["group"] == "rotation")]
    if not rot.empty:
        r_naive = _value(rot, treatment="full_season")
        r_head = _value(rot, treatment="beta_binomial")
        st.info(f"**Gains shrink on established rotation players, but the ordering "
                f"holds** — {r_naive:.1f} naive → {r_head:.1f} with the head "
                f"({(r_head - r_naive) / r_naive:.1%}), against "
                f"{(head - naive) / naive:.1%} overall, because regulars miss less "
                f"time. Do not quote the aggregate figure as if it applied to the "
                f"players a DFS user cares about most.")

    with detail("Season-total metrics — every treatment and group"):
        st.dataframe(st_metrics.round(3), width="stretch", hide_index=True)
    provenance("`make season-total` → `outputs/predictions/season_total_metrics.csv`")


# ── The two ablations ─────────────────────────────────────────────────────────

def _ablations(ctx: Ctx) -> None:
    st.markdown("---")
    left, right = st.columns(2)

    with left:
        st.markdown("#### Playoff workload — selection, not fatigue")
        abl = optional(ctx.predictions("availability_workload_ablation.csv"),
                       target="make availability-model")
        if abl is not None:
            st.plotly_chart(
                fig_bars(abl.sort_values("crps_games"), "variant", ["crps_games"],
                         ctx.th, "Held-out CRPS by feature set",
                         axis_title="CRPS (games)", height=280,
                         emphasis="plus_playoff_workload"),
                width="stretch")
            note("**Every sign is the opposite of the mechanism the block was built "
                 "for.** Playoff columns predict *better* next-season availability, "
                 "because playoff participation marks a good player on a good team "
                 "and that selection effect beats fatigue outright. The only column "
                 "pointing the way fatigue would is `career_minutes` — cumulative "
                 "mileage. Same inversion as `missed_injury` below.")
            table_view(abl.round(4), "Workload ablation — table view")

    with right:
        st.markdown("#### Nonlinearity — a false positive, kept as a method card")
        non = optional(ctx.predictions("availability_nonlinearity.csv"),
                       target="make availability-model")
        if non is not None:
            st.plotly_chart(
                fig_bars(non, "variant", ["val_crps_games", "test_crps_games"],
                         ctx.th, "CRPS by split — validation selects, test confirms",
                         axis_title="CRPS (games)", height=280),
                width="stretch")
            st.warning(
                "**The test split prefers every curved variant and none of them "
                "replicate.** A *paired* bootstrap on the 911 test rows put the "
                "quadratic gain at −0.047, 95% CI [−0.079, −0.015], P(Δ<0) = 99.7% — "
                "and it was still a false positive, because a paired interval says a "
                "difference is consistent *within one sample*, not that the sample "
                "was representative. **Select on validation; quote test for "
                "confirmation only.**")
            table_view(non.round(4), "Nonlinearity ablation — table view")

    provenance("`make availability-model` → "
               "`outputs/predictions/availability_workload_ablation.csv`, "
               "`outputs/predictions/availability_nonlinearity.csv`")


# ── Absence reasons ───────────────────────────────────────────────────────────

def _reasons(ctx: Ctx, prof: pd.DataFrame) -> None:
    st.markdown("---")
    st.markdown("### Splitting absences by reason — and the intuition inverts")

    dec = prof[(prof["measurement"] == "decomposition") & (prof["window"] == WINDOW)]
    reasons = dec[dec["key"].str.startswith("missed_")].copy()
    if reasons.empty:
        return
    wide = (reasons.pivot_table(index="key", columns="metric", values="value")
            .reset_index().rename(columns={"key": "reason"}))
    wide["reason"] = wide["reason"].str.removeprefix("missed_")
    wide = wide.sort_values("r_vs_next_gp_share")

    inc = dec[dec["key"] == "incremental"]
    base = _value(inc, metric="r2_prior_gp_share")
    plus_agg = _value(inc, metric="r2_plus_missed_games")
    plus_split = _value(inc, metric="r2_plus_reason_split")
    above = _value(inc, metric="r2_reason_split_above_null")
    null_sd = _value(inc, metric="r2_reason_split_null_sd")

    stat_tiles([
        ("Prior GP share alone", f"{base:.4f}", "In-sample R², season-absorbed."),
        ("+ aggregate missed games", f"{plus_agg:.4f}",
         f"Worth only +{plus_agg - base:.4f}."),
        ("+ the 8-way reason split", f"{plus_split:.4f}",
         "The split is essentially the entire effect."),
        ("Above a shuffled null", f"+{above:.4f}",
         f"≈{above / null_sd:.0f} sd. Not the null the plan expected."),
    ])

    st.plotly_chart(
        fig_bars(wide, "reason", ["r_vs_next_gp_share", "r_persistence_of_column"],
                 ctx.th,
                 "Absence reasons — against next-season availability, and their own "
                 "persistence", axis_title="r", height=380),
        width="stretch")
    note("**The carrying reasons invert the intuition.** `scratch` and `inactive` "
         "predict next-season availability strongly and negatively, while "
         "`injury` is slightly **positive** — the *rotation* reasons predict "
         "availability and the injury reason does not. `scratch` also persists more "
         "than any other availability column including games played itself, which is "
         "the strongest single statement of 'much of what looks like availability is "
         "rotation status'. `missed_injury` counts only injury-flagged players who "
         "**dressed**, so it partly marks 'was a rotation player'; the genuinely "
         "unavailable are `inactive`, for which the endpoint states no reason at all.")
    table_view(wide.round(4), "Reason decomposition — table view")
    st.info("**`models.availability.FEATURE_COLS` does not consume any of these "
            "columns yet** — the one evidence-backed feature change outstanding on "
            "the head.")
    provenance("`make availability-profile` → `availability_profile.csv` "
               "(`decomposition`)")


# ── The Stan port ─────────────────────────────────────────────────────────────

def _stan(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The Stan port — a defined check, not a hopeful comparison")
    note("An L2 penalty of `l2` on standardized coefficients **is** a "
         "`normal(0, 1/sqrt(2·l2))` prior, so the posterior *mode* is exactly the "
         "penalized optimum the point MLE finds. That identity is what makes this a "
         "verification rather than a comparison of two similar numbers.")

    coefs = optional(ctx.predictions("stan_availability_coefficients.csv"),
                     target="make stan-availability")
    diag = optional(ctx.predictions("stan_availability_diagnostics.csv"),
                    target="make stan-availability")

    if coefs is not None and diag is not None:
        d = diag.iloc[0]
        stat_tiles([
            ("MLE inside the 95% interval",
             f"{int(coefs['mle_inside_95'].sum())} / {len(coefs)}",
             "Every coefficient, including the dispersion ρ."),
            ("Largest gap", f"{coefs['z_from_mle'].abs().max():.3f} sd",
             "Biggest MLE-to-posterior-mean difference, in posterior sd."),
            ("R̂ / divergences",
             f"{d['max_rhat']:.4f} / {int(d['divergences'])}",
             f"min ESS {d['min_ess_bulk']:,.0f} over {int(d['n_draws']):,} draws, "
             f"{d['wall_clock_s']:.0f} s wall clock."),
        ])

        forest = coefs[coefs["term"] != "rho"]
        fig = go.Figure()
        fig.add_scatter(
            x=forest["posterior_mean"], y=forest["term"], mode="markers",
            name="posterior mean",
            error_x=dict(type="data", symmetric=False,
                         array=forest["q97_5"] - forest["posterior_mean"],
                         arrayminus=forest["posterior_mean"] - forest["q2_5"],
                         color=ctx.th["axis"], thickness=1.5, width=0),
            marker=dict(size=9, color=ctx.th["series"][0],
                        line=dict(color=ctx.th["surface"], width=2)),
            hovertemplate="%{y}<br>posterior %{x:+.3f}<extra></extra>")
        fig.add_scatter(
            x=forest["mle"], y=forest["term"], mode="markers", name="penalized MLE",
            marker=dict(size=7, color=ctx.th["series"][1], symbol="x"),
            hovertemplate="%{y}<br>MLE %{x:+.3f}<extra></extra>")
        fig.add_vline(x=0, line=dict(color=ctx.th["axis"], width=1))
        fig.update_layout(title="Coefficients — posterior 95% interval against the "
                                "penalized MLE")
        fig.update_xaxes(title="standardized coefficient")
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(apply_theme(fig, ctx.th, 620), width="stretch")
        table_view(coefs.round(4), "Coefficients — table view")

    board = optional(ctx.predictions("stan_availability_board.csv"),
                     target="make stan-availability")
    if board is not None and len(board):
        st.markdown("#### What the posterior buys, and the correction")
        st.plotly_chart(
            fig_lines(board, "n_players",
                      {"independent_sd": "independent term",
                       "shared_beta_sd": "shared-β term"},
                      ctx.th, "Season-total uncertainty by board size",
                      y_title="sd (games)", x_title="players on the board",
                      height=340, label_last=False),
            width="stretch")
        small, whole = board.iloc[0], board.iloc[-1]
        st.warning(
            f"**It is worth almost nothing on one roster, and the size of the "
            f"portfolio is what decides.** The independent term grows as √N and the "
            f"shared-β term as N, so their ratio scales as √N: inflation is "
            f"**{small['inflation']:.4f}× at {int(small['n_players'])} players** "
            f"against **{whole['inflation']:.4f}× across all "
            f"{int(whole['n_players'])}**. Quoting the full-board figure as if it "
            f"applied to a 15-man roster is the over-claim to avoid — it was made and "
            f"corrected in the session that built this.")
        note("What the posterior buys is `Var_θ(Σ E[Y|θ])`: every player shares β, so "
             "one draw moves the whole board together, and that term is exactly 0 "
             "under any point estimate. Do **not** argue for it on marginal CRPS — "
             "that is a wash by construction, and the integrated predictive is "
             "measurably *narrower* per player, because `n·μ(1−μ)·[1+(n−1)ρ]` is "
             "concave in μ and Jensen pushes against the mixture's added variance.")
        table_view(board.round(4), "Board correlation — table view")

    provenance("`make stan-availability` → "
               "`outputs/predictions/stan_availability_coefficients.csv`, "
               "`outputs/predictions/stan_availability_diagnostics.csv`, "
               "`outputs/predictions/stan_availability_board.csv`")


# ── The injury-report transfer function ───────────────────────────────────────

def _report_calibration(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The injury-report transfer function")

    rep = optional(ctx.eda("report_calibration.csv"),
                   target="make report-calibration")
    if rep is None:
        return

    order = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
    cal = rep[rep["measurement"] == "calibration"].copy()
    # The key is `designation|lead`: `all` pools the report leads, `lead_0` is the
    # same-day report and `lead_1` a day-stale one.
    cal[["designation", "lead"]] = cal["key"].str.split("|", expand=True)
    played = cal[(cal["metric"] == "p_played") & (cal["lead"] == "all")]
    played = (played.set_index("designation")
              .reindex([o for o in order if o in set(played["designation"])])
              .reset_index())

    st.plotly_chart(
        fig_bars(played, "designation", ["value"], ctx.th,
                 "P(played) by injury-report designation", axis_title="P(played)",
                 horizontal=False, height=320),
        width="stretch")
    st.warning("**The designation scale is not monotone**: `Available` plays *less* "
               "than `Probable`. That inversion is reason mix, not label noise — "
               "excluding G-League rows the scale reads 0.001 / 0.016 / 0.559 / 0.920 "
               "/ 0.903 and the residual sits inside a ~1.6 pp standard error. "
               "**Treat Probable and Available as one designation**, and condition on "
               "`reason_category` rather than on the five-level scale.")

    cov = rep[rep["measurement"] == "coverage"]
    stat_tiles([
        ("Report rows joined", f"{_value(cov, metric='match_rate'):.1%}",
         f"{int(_value(cov, metric='scored_rows')):,} of "
         f"{int(_value(cov, metric='report_rows')):,} archive rows matched to a "
         f"realized box-score outcome."),
        ("Unmatched names", f"{_value(cov, metric='share_unmatched'):.1%}",
         "Exact-match join on `game_id` + `name_key` with no fuzzy tier at all — "
         "which is why this rate is trustworthy here and is not elsewhere."),
        ("Ambiguous cells", f"{_value(cov, metric='share_ambiguous'):.1%}",
         "Same-game name collisions, excluded rather than guessed."),
        ("Players / dates",
         f"{int(_value(cov, metric='players'))} / "
         f"{int(_value(cov, metric='game_dates'))}",
         "Coverage of the forward-only PDF archive."),
    ])

    left, right = st.columns(2)
    with left:
        rev = rep[(rep["measurement"] == "revision")
                  & (rep["metric"] == "p_unchanged")]
        if not rev.empty:
            rev = (rev.set_index("key")
                   .reindex([o for o in order if o in set(rev["key"])]).reset_index())
            st.plotly_chart(
                fig_bars(rev, "key", ["value"], ctx.th,
                         "Unchanged in the next day's report",
                         axis_title="P(unchanged)", horizontal=False, height=300),
                width="stretch")
            note("`Out` is near-deterministic and sticky; `Questionable` is a coin "
                 "flip that resolves.")
    with right:
        lead = cal[(cal["metric"] == "p_played") & (cal["lead"].isin(["lead_0",
                                                                      "lead_1"]))]
        if not lead.empty:
            wide = (lead.pivot_table(index="designation", columns="lead",
                                     values="value").reset_index())
            wide = (wide.set_index("designation")
                    .reindex([o for o in order if o in set(wide["designation"])])
                    .reset_index())
            st.plotly_chart(
                fig_bars(wide, "designation", ["lead_0", "lead_1"], ctx.th,
                         "P(played) — same-day report against a day-stale one",
                         axis_title="P(played)", horizontal=False, height=300),
                width="stretch")
            note("A **stale** Questionable is worth about what a fresh one is, which "
                 "is the encouraging read for a preseason snapshot — that is read "
                 "weeks ahead.")

    # Read live rather than typed: these two moved when the box-score backfill closed
    # the missing 2025-26 games, and the prose figures went stale unnoticed.
    mins = cal[(cal["metric"] == "mean_min_given_played") & (cal["lead"] == "all")]
    if not mins.empty:
        by = mins.set_index("designation")["value"]
        note(f"Minutes barely move either: a Questionable who plays gets "
             f"{by.get('Questionable', float('nan')):.1f} against a Probable's "
             f"{by.get('Probable', float('nan')):.1f}, so there is no meaningful "
             f"minutes haircut to model. The designation acts on the play/not-play "
             f"margin, not on workload.")

    with detail("Transfer function — every designation × reason cell"):
        st.dataframe(rep.round(4), width="stretch", hide_index=True)
    provenance("`make report-calibration` → `outputs/eda/report_calibration.csv`, "
               "`data/features/report_transfer.parquet`")
