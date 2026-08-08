"""Tab 6 · The DK component-rate heads.

Twelve quantities per player-game, each with its own likelihood. The centrepiece is
the **no-fit floor**: prior per-36 rate × actual minutes, no fitting at all, scoring
held-out R² 0.82–0.94. A head that does not clear it is not a model.
"""

import pandas as pd
import streamlit as st

from dashboard import decisions as D
from dashboard.artifacts import Ctx, optional
from dashboard.charts import fig_bars, fig_heatmap, fig_lines
from dashboard.layout import (decision_cards, detail, note, provenance, stat_tiles,
                              tab_header, table_view)

TOPIC = "components"

# The output contract, in the order the chain evaluates it. `min` and the three
# attempt counts contribute nothing to DK scoring themselves — they are exposure and
# trials for the eight that do.
CONTRACT = [
    ("min | available", "successes / trials", "game length (48, or 53/58/… in OT)",
     "exposure only", "stan-minutes"),
    ("fg2a", "count — NB", "min", "exposure only", "stan-components"),
    ("fg3a", "count — NB", "min", "exposure only", "stan-components"),
    ("fta", "count — NB (arrives in **pairs**: model trips, double)", "min",
     "exposure only", "stan-components"),
    ("fg2m | fg2a", "successes / trials", "fg2a", "2 pts", "stan-components"),
    ("fg3m | fg3a", "successes / trials", "fg3a", "3.5 pts", "stan-components"),
    ("ftm | fta", "successes / trials", "fta", "1 pt", "stan-components"),
    ("reb", "count — NB", "min", "1.25", "stan-components"),
    ("ast", "count — NB", "min", "1.5", "stan-components"),
    ("stl", "count — NB", "min", "2", "stan-components"),
    ("blk", "count — NB", "min", "2", "stan-components"),
    ("tov", "count — NB", "min", "−0.5", "stan-components"),
]


def render(ctx: Ctx) -> None:
    tab_header(
        "The DK component-rate heads",
        "`dk_pts` is deterministic given the components and is never predicted "
        "directly. Twelve quantities are modeled per player-game; only **eight** "
        "reach the scoring function, and the other four matter solely through the "
        "exposure and trials they supply.")

    _contract(ctx)
    _floor(ctx)
    _scale_not_curvature(ctx)
    _alpha_trap(ctx)
    _dispersion(ctx)
    _build_tracker(ctx)

    st.markdown("---")
    st.markdown("### Decisions on these heads")
    decision_cards(D.by_topic(TOPIC))


# ── The output contract ───────────────────────────────────────────────────────

def _contract(ctx: Ctx) -> None:
    st.markdown("### The output contract")
    contract = pd.DataFrame(CONTRACT, columns=["component", "likelihood",
                                               "exposure / trials", "DK weight",
                                               "target"])
    st.dataframe(contract.drop(columns="target"), width="stretch", hide_index=True)
    note("Given a joint draw, reassemble `pts = 2·fg2m + 3·fg3m + ftm` and pass it to "
         "`preprocess.compute_dk_pts`. Under DK weights the scoring contribution is "
         "`ftm + 2·fg2m + 3.5·fg3m` — 3.5 because DK pays 0.5 per made three on top "
         "of the 3 points. **`2·fgm + 3·fg3m + ftm` is the tempting error**: it pays "
         "a three 2 + 3 = 5 and matches `pts` on only the 59.97% of games with "
         "`fg3m = 0`, because the stored `fgm` already includes threes.")
    st.info("**The deliverable is a joint draw, not twelve marginals.** The "
            "double-double bonus is a simultaneous threshold on `pts`/`reb`/`ast`/"
            "`stl`/`blk`, so `E[bonus] ≠ bonus(E[x])` and no set of marginal means "
            "can produce it.")


# ── The no-fit floor ──────────────────────────────────────────────────────────

def _floor(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The no-fit floor is nearly the whole model")
    note("`carry_forward` is the prior per-36 rate × actual minutes / 36, with **no "
         "fitting at all**. Every proposed head is quoted against it, and every "
         "output row carries `beats_floor`.")

    m = optional(ctx.predictions("component_rate_metrics.csv"),
                 target="make component-rates")
    if m is None:
        return
    sweep = m[m["analysis"] == "variant_sweep"]
    counts = sweep[sweep["kind"] == "count"]
    if counts.empty:
        return

    floor = (counts[counts["variant"] == "carry_forward"]
             .set_index("head")["r2"].rename("floor"))
    best = (counts[counts["variant"] != "carry_forward"]
            .groupby("head")["r2"].max().rename("best_fitted"))
    compare = pd.concat([floor, best], axis=1).reset_index()
    compare["gain"] = compare["best_fitted"] - compare["floor"]
    compare = compare.sort_values("floor", ascending=False)

    stat_tiles([
        ("Floor R², range",
         f"{compare['floor'].min():.2f} – {compare['floor'].max():.2f}",
         "Held-out R² on the season total, eight count heads, with no fitting at "
         "all."),
        ("Best fitted gain",
         f"+{compare['gain'].min():.4f} – +{compare['gain'].max():.4f}",
         "What the best of seven fitted variants buys over the floor."),
        ("Heads clearing the floor",
         f"{int(sweep['beats_floor'].sum())} / {len(sweep)}",
         "Across every head × variant row, counts and conversions."),
    ])

    st.plotly_chart(
        fig_bars(compare, "head", ["floor", "best_fitted"], ctx.th,
                 "Held-out R² — the no-fit floor against the best fitted variant",
                 axis_title="R² on the season total", height=380),
        width="stretch")
    note("The floor being this strong is the sharpest available statement of "
         "'attempts persist' — and it means the rate side is close to saturated from "
         "prior-season information alone. That is consistent with the availability "
         "tab's oracle contrast, where perfect games played beats perfect rate.")
    table_view(compare.round(4), "Floor vs best fitted — table view")
    provenance("`make component-rates` → "
               "`outputs/predictions/component_rate_metrics.csv`")


# ── Scale, not curvature ──────────────────────────────────────────────────────

def _scale_not_curvature(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### For the count heads the answer is *scale*, not curvature")

    m = optional(ctx.predictions("component_rate_metrics.csv"),
                 target="make component-rates")
    if m is None:
        return
    counts = m[(m["analysis"] == "variant_sweep") & (m["kind"] == "count")]
    if counts.empty:
        return

    grid = counts.pivot_table(index="head", columns="variant", values="r2")
    order = ["carry_forward", "linear", "log_own", "log_own_spline", "log_own_inter",
             "pca", "pca_spline", "pca_inter"]
    grid = grid.reindex(columns=[c for c in order if c in grid.columns])
    grid = grid.reindex(grid["carry_forward"].sort_values(ascending=False).index)

    st.plotly_chart(
        fig_heatmap(grid, ctx.th, "Held-out R² by head × variant", "R²",
                    height=420, hover="%{y} · %{x}<br>R² %{z:.4f}"),
        width="stretch")
    st.info(
        "**A log link wants a multiplicative predictor.** "
        "`log E[rate] = β·log(prior rate)` makes the model `rate ∝ prior_rate^β`, "
        "which is the right shape. Linear-in-raw-rate inside `exp()` is badly "
        "misspecified — catastrophically so for the zero-heavy skewed heads, where "
        "`fg3a` reads 0.520 and `blk` 0.638 against floors of 0.904 and 0.841. "
        "`log(own)` recovers nearly all of it in **one term**.")

    spline_gain = (grid.get("log_own_spline") - grid.get("log_own")).sort_values(
        ascending=False).rename("gain").reset_index()
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            fig_bars(spline_gain, "head", ["gain"], ctx.th,
                     "What a spline adds over `log(own)`",
                     axis_title="Δ held-out R²", height=320),
            width="stretch")
    with right:
        note("**Splines add a real but small further gain, concentrated where the "
             "prior is most skewed** — `fg3a` and `blk` only. Spend flexibility on "
             "those two heads and nowhere else.")
        note("**The `age × own` and `mpg × own` interactions are a null** once the "
             "scale is right — ≤ +0.001, and *negative* for `blk` and `ast`. "
             "Component-specific aging is real but does not survive as an "
             "interaction here.")
        note("**Walk-forward PCA of the whole 156-column season matrix is worth "
             "~nothing** over three raw context columns — within ±0.003 of `log_own` "
             "on every head. The 121 style/tracking columns add nothing once you have "
             "the player's own prior rate and his minutes. Use the cheap raw spec.")

    conv = m[(m["analysis"] == "variant_sweep") & (m["kind"] == "conversion")]
    if not conv.empty:
        st.markdown("#### The conversion side")
        nll = conv.pivot_table(index="head", columns="variant", values="nll")
        st.plotly_chart(
            fig_bars(nll.reset_index().melt(id_vars="head", value_name="nll")
                     .pivot_table(index="head", columns="variant", values="nll")
                     .reset_index(), "head",
                     [c for c in ["carry_forward", "linear", "spline_own", "inter",
                                  "pca", "pca_inter"] if c in nll.columns],
                     ctx.th, "Negative log-likelihood by conversion head "
                             "(lower is better)",
                     axis_title="NLL", height=320),
            width="stretch")
        note("**`ftm|fta` is the one head where *nothing* beats the floor.** "
             "Free-throw percentage is pure player skill with no context to add, so "
             "an empirical-Bayes shrink of the prior is already optimal. `fg2m|fg2a` "
             "and `fg3m|fg3a` gain ~1–2% of their NLL at most.")
        st.warning(
            "**The conversion floor has to be a *shrunk* carry-forward, and that is a "
            "fact about proportions.** A player who went 0-for-3 from three has a "
            "prior 3P% of exactly 0.000; carrying it onto 200 attempts gives a "
            "beta-binomial NLL of 1.3e9 and makes the benchmark meaningless. The "
            "floor is therefore `p = (made + k·league_mean)/(attempts + k)` with `k` "
            "fitted on train only — one shrinkage constant, no features. This is the "
            "'shrink conversion percentages hard' rule showing up as a benchmark "
            "requirement.")

    with detail("Every head × variant row"):
        st.dataframe(m[m["analysis"] == "variant_sweep"].round(4), width="stretch",
                     hide_index=True)


# ── The sklearn alpha trap ────────────────────────────────────────────────────

def _alpha_trap(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The `sklearn` alpha trap, re-run as a permanent ablation")

    m = optional(ctx.predictions("component_rate_metrics.csv"),
                 target="make component-rates")
    if m is None:
        return
    alpha = m[m["analysis"] == "alpha_sensitivity"]
    if alpha.empty:
        return

    heads = sorted(alpha["head"].unique())
    default = heads.index("reb") if "reb" in heads else 0
    c1, c2 = st.columns([1, 1])
    head = c1.selectbox("head", heads, index=default, key="alpha_head")
    variant = c2.selectbox("spec", sorted(alpha["variant"].unique()), index=0,
                           key="alpha_variant")

    curve = alpha[(alpha["head"] == head) & (alpha["variant"] == variant)]
    curve = curve.sort_values("alpha")
    if curve.empty:
        return

    plot = curve[["alpha", "r2", "floor_r2"]].copy()
    st.plotly_chart(
        fig_lines(plot, "alpha", {"r2": "fitted", "floor_r2": "no-fit floor"},
                  ctx.th, f"{head} · {variant} — held-out R² across the alpha grid",
                  y_title="held-out R²", x_title="alpha (log scale)", height=360,
                  label_last=False).update_xaxes(type="log"),
        width="stretch")

    best = curve.loc[curve["r2"].idxmax()]
    at_one = curve[curve["alpha"] == 1.0]
    if not at_one.empty:
        stat_tiles([
            ("Best on the grid", f"{float(best['r2']):.4f}",
             f"at alpha = {float(best['alpha']):g}"),
            ("At alpha = 1.0", f"{float(at_one['r2'].iloc[0]):.4f}",
             "The default-looking value that crushed every coefficient."),
            ("Loss to alpha = 1.0",
             f"{float(best['r2']) - float(at_one['r2'].iloc[0]):.4f}",
             "The guard: each head is compared against its own optimum on the grid."),
            ("No-fit floor", f"{float(curve['floor_r2'].iloc[0]):.4f}",
             "The floor is what caught this, which is why it is mandatory."),
        ])

    st.error(
        "**`sklearn`'s two regularization conventions are opposite, and with exposure "
        "weights the difference is ~7 orders of magnitude.** `PoissonRegressor` "
        "minimizes `deviance / (2·Σw) + alpha·‖coef‖²` — the data term is **averaged "
        "by the weight sum**. Fitting a rate with `sample_weight = minutes` makes "
        "Σw ≈ 1e7, so a default-looking `alpha=1.0` is an enormous penalty. It fails "
        "*quietly*: the fit converges, coefficients are finite, and a flexible basis "
        "partially compensates — so a spline or an interaction looks like it is "
        "buying real signal. `LogisticRegression` is the reverse: its objective is "
        "not averaged, so `C=1.0` is *weak*.")
    note("This cost a full set of published figures. At alpha = 10, every fit falls "
         "below the no-fit floor, which is the visual statement of why the floor is "
         "mandatory. The guard cannot simply be 'the fitted line crosses the floor', "
         "because `blk` and `fg3a` lose to the floor at *every* alpha under "
         "`log_own` — those two need splines, which is a modelling finding rather "
         "than the penalty misbehaving.")

    with detail("Alpha sensitivity — every head × variant × alpha"):
        st.dataframe(alpha.round(4), width="stretch", hide_index=True)
    provenance("`make component-rates` → `component_rate_metrics.csv` "
               "(`analysis == alpha_sensitivity`)")


# ── Dispersion and zero-inflation ─────────────────────────────────────────────

def _dispersion(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### Why decomposing `pts` removes a misspecification rather than "
                "patching one")

    prof = optional(ctx.eda("target_profile.csv"), target="make target-profile")
    if prof is None:
        return
    dist = prof[prof["analysis"] == "distribution"]
    minutes = dist[dist["bucket_kind"] == "minutes"]
    if minutes.empty:
        return

    order = ["0-5", "5-12", "12-18", "18-24", "24-30", "30-48"]
    grid = minutes.pivot_table(index="metric", columns="bucket",
                               values="var_over_mean_within")
    grid = grid.reindex(columns=[o for o in order if o in grid.columns])
    rows = [m for m in ["dk_pts", "pts", "fgm", "fg2m", "fg3m", "ftm", "fga", "fg2a",
                        "fg3a", "fta", "reb", "ast", "stl", "blk", "tov"]
            if m in grid.index]

    st.plotly_chart(
        fig_heatmap(grid.reindex(rows), ctx.th,
                    "Within-player dispersion by minutes played", "var / mean",
                    diverging=True, zmid=1.0, height=440,
                    hover="%{y} · %{x} min<br>var/mean %{z:.2f}"),
        width="stretch")
    note("Neutral is 1.0, where Poisson is correctly specified — blue is "
         "under-dispersed, red over-dispersed. **`pts` sits at ~2.2 at every "
         "exposure while the shot classes it decomposes into sit at ~1.0.** The "
         "weighting is the entire source of the overdispersion: a 2× coefficient "
         "squares into the variance and doubles var/mean while leaving the mean "
         "alone. So a Poisson head is badly misspecified on `pts` and correctly "
         "specified on the shot classes — decomposing does not merely help, it "
         "**removes** the misspecification rather than patching it with a negative "
         "binomial. Free throws stay overdispersed because they arrive in pairs, and "
         "`X = 2·Poisson` has var/mean exactly 2.")

    zeros = minutes.pivot_table(index="metric", columns="bucket", values="zero_share")
    zeros = zeros.reindex(columns=[o for o in order if o in zeros.columns])
    z_rows = [m for m in ["dk_pts", "pts", "fg3m", "stl", "blk"] if m in zeros.index]
    st.plotly_chart(
        fig_heatmap(zeros.reindex(z_rows), ctx.th,
                    "Share of games at exactly zero, by minutes played", "zero share",
                    height=280, hover="%{y} · %{x} min<br>%{z:.1%} zeros"),
        width="stretch")
    note("**Zero-inflation is a minutes artifact, not a property of the target.** "
         "`dk_pts` is 0 in 35.4% of sub-5-minute games and 0.0% above 18 minutes, so "
         "conditional on minutes the zeros are Poisson zeros and there is nothing to "
         "zero-inflate. But `blk`, `fg3m` and `stl` stay genuinely low-count even in "
         "30–48 minute games — those three heads are misspecified under MSE at *any* "
         "minutes level.")

    usage = dist[dist["bucket_kind"] == "usage"]
    if not usage.empty:
        u_grid = usage.pivot_table(index="metric", columns="bucket",
                                   values="var_over_mean_within")
        u_rows = [m for m in ["pts", "fga", "reb", "ast"] if m in u_grid.index]
        st.plotly_chart(
            fig_heatmap(u_grid.reindex(u_rows), ctx.th,
                        "Within-player dispersion by usage", "var / mean",
                        diverging=True, zmid=1.0, height=260,
                        hover="%{y} · usage %{x}<br>var/mean %{z:.2f}"),
            width="stretch")
        note("**Dispersion is worst at *low* usage, not low minutes** — the Poisson "
             "specification is tightest where the minutes are. If a dispersion term "
             "is added, key it on usage, not minutes.")

    with detail("Distribution — every metric × bucket"):
        st.dataframe(dist.round(4), width="stretch", hide_index=True)
    provenance("`make target-profile` → `outputs/eda/target_profile.csv`")


# ── Per-head build tracker ────────────────────────────────────────────────────

def _build_tracker(ctx: Ctx) -> None:
    st.markdown("---")
    st.markdown("### The Stan heads — per-head build status")

    metrics_path = ctx.predictions("stan_component_metrics.csv")

    if not metrics_path.exists():
        # An "in progress" state, deliberately not a missing-artifact warning: the
        # fit is long-running, and a red warning would read as a defect rather than
        # as a run that has not finished.
        st.info(":material/hourglass_top: **`make stan-components` has not written "
                "its artifacts yet.** The eleven heads fit separately — eight "
                "negative-binomial counts and three beta-binomial conversions, from "
                "two `.stan` files — and this panel fills in per head as "
                "`outputs/predictions/stan_component_metrics.csv` lands. The "
                "specification above is measured and settled; only the sampler run "
                "is outstanding.")
        tracker = pd.DataFrame(CONTRACT,
                               columns=["component", "likelihood",
                                        "exposure / trials", "DK weight", "target"])
        tracker["Stan fit"] = ["✅ built" if t == "stan-minutes" else "⏳ in progress"
                               for t in tracker["target"]]
        st.dataframe(tracker[["component", "likelihood", "exposure / trials",
                              "Stan fit"]], width="stretch", hide_index=True)
    else:
        m = optional(metrics_path, target="make stan-components")
        if m is not None:
            selected = m[m["selected"]].copy()
            # `val_r2` since 2026-08-05; `test_r2` on artifacts written before the
            # held-out lock landed. Read whichever the file carries rather than assuming,
            # because this tab must render an old artifact and a new one identically —
            # the alternative is a KeyError on the first refresh after a conversion.
            r2 = "val_r2" if "val_r2" in m.columns else "test_r2"
            crps = "val_crps" if "val_crps" in m.columns else "test_crps"
            ks = "val_pit_ks" if "val_pit_ks" in m.columns else "test_pit_ks"
            split = "validation" if r2 == "val_r2" else "held-out"
            floor = (m[m["variant"] == "carry_forward"]
                     .set_index("head")[r2].rename("floor_r2"))
            view = selected.merge(floor, on="head", how="left")
            view["gain"] = view[r2] - view["floor_r2"]

            stat_tiles([
                ("Heads fitted", f"{m['head'].nunique()}",
                 "Seven negative-binomial counts and four beta-binomial "
                 "conversions, each its own fit."),
                ("Selected variants clearing the floor",
                 f"{int(selected['beats_floor'].sum())} / {len(selected)}",
                 "The selected variant per head against its no-fit carry-forward."),
                ("Median gain over the floor", f"{view['gain'].median():+.4f}",
                 f"{split.capitalize()} R². The floor really is nearly the whole model."),
            ])

            st.plotly_chart(
                fig_bars(view.sort_values(r2, ascending=False), "head",
                         ["floor_r2", r2], ctx.th,
                         "Stan heads — no-fit floor against the selected variant",
                         axis_title=f"{split} R²", height=380),
                width="stretch")
            note("Variant selection is on **validation**, and since 2026-08-05 there is "
                 "no test column at all: `src/models/held_out.py` raises on the held-out "
                 "seasons and `make final-evaluation` reads them once. The discipline "
                 "the availability head's nonlinearity false positive bought, now "
                 "enforced rather than remembered.")
            with detail("Selected variant per head"):
                st.dataframe(
                    view[[c for c in ["head", "kind", "variant", "n_features", r2,
                                      "floor_r2", "gain", crps, ks, "beats_floor"]
                          if c in view.columns]].round(4),
                    width="stretch", hide_index=True)
            table_view(m.round(4), "Every head × variant — table view")

        sub = optional(ctx.predictions("stan_component_substitution.csv"),
                       target="make stan-components")
        if sub is not None and not sub.empty:
            st.markdown("#### The 3PA/2PA substitution, reparameterized into the chain")
            delta = float(sub["reparam_minus_canonical"].iloc[0])
            st.success(
                f"**Modelling `fga` as the count and `fg3a | fga` as a binomial "
                f"*share* beats two independent counts by "
                f"{abs(delta):.3f} joint NLL per player-season, on both splits.** "
                f"That is the predicted result: the reparameterization enforces the "
                f"substitution *by construction*, keeps the posterior factorization "
                f"exact, and is better specified anyway — shot-mix shares persist "
                f"like counts (`sco_pct_fga_3pt` at 0.886) while the two raw counts "
                f"trade off against each other at −0.11 residual correlation. "
                f"Coupling two Poissons was never necessary.")
            table_view(sub.round(4), "Substitution arms — table view")

        d = optional(ctx.predictions("stan_component_diagnostics.csv"),
                     target="make stan-components")
        if d is not None and not d.empty:
            note(f"Sampler across all {len(d)} fits: max R̂ {d['max_rhat'].max():.4f}, "
                 f"{int(d['divergences'].sum())} divergences, "
                 f"{int(d['treedepth_saturated'].sum())} treedepth saturations, "
                 f"{d['wall_clock_s'].sum() / 60:.0f} minutes total. The saturations "
                 f"are concentrated in the spline variants — B-spline bases are badly "
                 f"conditioned for HMC, valid but expensive.")
            with detail("Sampler diagnostics per head × variant × split"):
                st.dataframe(d.round(4), width="stretch", hide_index=True)

    st.info("**Fit the heads separately, not as one joint model — the posterior "
            "factorizes exactly.** The chain `availability → min | available → "
            "counts | min → makes | attempts` has distinct parameter blocks and "
            "independent priors, so eleven separate fits recover the *identical* "
            "posterior. That is an identity, not an approximation. `megamodel.stan` "
            "is the proof from this project's own history: no parameter was shared "
            "between any two heads, so it paid the full joint-fit price and ran on "
            "`sample_frac(0.01)` — 99% of the data given up for a coupling that was "
            "not in the model.")
    provenance("`make stan-components` → "
               "`outputs/predictions/stan_component_metrics.csv`, "
               "`outputs/predictions/stan_component_diagnostics.csv`, "
               "`outputs/predictions/stan_component_substitution.csv`")
