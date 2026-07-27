"""Streamlit explorer over the precomputed EDA artifacts.

Reads parquet/CSV/pickle that `make eda` already wrote — nothing here fits a PCA,
clusters anything, or refits a model. Startup is a set of cached reads, so the
tier and era toggles switch artifacts rather than recomputing them. Two places do
a *join* on load (the persistence t-vs-t+1 scatter pairs seasons; the archetype
composition heatmap pivots shares); those are reshapes of stored columns, not fits.

Run with `make dashboard`.

## Colour

The palette is the validated reference instance, used unmodified: eight categorical
slots in fixed order, a single-hue blue ramp for magnitude, and blue↔red with a
neutral gray midpoint for polarity. Two rules from that validation constrain the
charts here and are worth knowing before editing one:

- The eight slots clear the CVD gates on *adjacent* pairs (bars, lines, stacks),
  but only the **first three** clear them on *all* pairs. So any scatter, and any
  chart where non-adjacent series sit side by side, caps at three coloured series
  and uses highlight-and-gray past that.
- Three light-mode slots fall below 3:1 contrast on the light surface, which
  obliges the relief rule: every chart ships a table-view twin in an expander.

Plot surfaces are pinned to the exact surfaces the palette was validated against
(`#fcfcfb` / `#1a1a19`) rather than inherited from Streamlit's page chrome, so the
measured contrast and separation figures apply as documented.
"""

import pickle
import sys
from pathlib import Path

# Project root on the path so `src.*` imports resolve, matching the convention in
# notebooks/fetch_game_logs.ipynb.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

# ── Palette ───────────────────────────────────────────────────────────────────

# Categorical slots, in the fixed validated order. Never cycled, never reordered.
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500",
             "#d55181", "#008300", "#9085e9", "#e66767"],
}
# All-pairs forms (scatter, bubble) may only use this many slots; past it, the
# fourth slot puts yellow beside orange and the pair fails the floors.
ALL_PAIRS_CAP = 3

# Single blue hue, light → dark. Reversed on the dark surface so "near zero"
# recedes toward the surface in both modes.
BLUE_RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
             "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
             "#0d366b"]
# Ordinal (discrete ordered) marks must stay clear of the surface: no lighter than
# step 250 on light, no darker than step 600 on dark.
ORDINAL_SLICE = {"light": slice(3, None), "dark": slice(None, 10)}

THEMES = {
    "light": {
        "surface": "#fcfcfb", "plane": "#f9f9f7",
        "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
        "grid": "#e1e0d9", "axis": "#c3c2b7", "neutral": "#f0efec",
        "series": SERIES["light"],
        "diverging": [[0.0, "#2a78d6"], [0.5, "#f0efec"], [1.0, "#e34948"]],
    },
    "dark": {
        "surface": "#1a1a19", "plane": "#0d0d0d",
        "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
        "grid": "#2c2c2a", "axis": "#383835", "neutral": "#383835",
        "series": SERIES["dark"],
        "diverging": [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]],
    },
}

FONT = 'system-ui, -apple-system, "Segoe UI", sans-serif'


def theme(mode: str) -> dict:
    th = dict(THEMES[mode])
    ramp = BLUE_RAMP if mode == "light" else list(reversed(BLUE_RAMP))
    th["sequential"] = [[i / (len(ramp) - 1), c] for i, c in enumerate(ramp)]
    th["ordinal"] = (BLUE_RAMP[ORDINAL_SLICE["light"]] if mode == "light"
                     else list(reversed(BLUE_RAMP[ORDINAL_SLICE["dark"]])))
    return th


def ordinal_colors(th: dict, n: int) -> list[str]:
    """`n` evenly spaced steps of the ordinal ramp, for ordered categories."""
    ramp = th["ordinal"]
    if n <= 1:
        return [ramp[len(ramp) // 2]]
    idx = np.linspace(0, len(ramp) - 1, n).round().astype(int)
    return [ramp[i] for i in idx]


def apply_theme(fig: go.Figure, th: dict, height: int = 420,
                legend: bool = True) -> go.Figure:
    """Recessive hairline chrome, pinned surfaces, no dashes anywhere."""
    fig.update_layout(
        height=height,
        paper_bgcolor=th["surface"], plot_bgcolor=th["surface"],
        font=dict(family=FONT, size=13, color=th["ink2"]),
        title_font=dict(family=FONT, size=15, color=th["ink"]),
        margin=dict(l=8, r=8, t=48, b=8),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(color=th["ink2"], size=12)),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    )
    axis = dict(gridcolor=th["grid"], gridwidth=1, griddash="solid",
                linecolor=th["axis"], linewidth=1, zerolinecolor=th["axis"],
                zerolinewidth=1, tickfont=dict(color=th["muted"], size=11),
                title_font=dict(color=th["ink2"], size=12))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def table_view(df: pd.DataFrame, label: str = "Table view") -> None:
    """The WCAG-clean twin every chart ships with — values never live in colour alone."""
    with st.expander(label):
        st.dataframe(df, width="stretch", hide_index=True)


def note(text: str) -> None:
    st.caption(text)


# ── Generic figure builders ───────────────────────────────────────────────────

def fig_heatmap(z: pd.DataFrame, th: dict, title: str, colorbar: str,
                diverging: bool = False, zmid: float | None = None,
                height: int = 460, hover: str = "%{y} · %{x}<br>%{z:.3f}") -> go.Figure:
    """Sequential for magnitude, diverging (two hues + neutral) for polarity."""
    fig = go.Figure(go.Heatmap(
        z=z.to_numpy(dtype=float), x=[str(c) for c in z.columns],
        y=[str(i) for i in z.index],
        colorscale=th["diverging"] if diverging else th["sequential"],
        zmid=zmid if diverging else None,
        colorbar=dict(title=dict(text=colorbar, side="right", font=dict(size=11)),
                      thickness=10, outlinewidth=0,
                      tickfont=dict(color=th["muted"], size=10)),
        hovertemplate=hover + "<extra></extra>",
    ))
    fig.update_layout(title=title)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    return apply_theme(fig, th, height, legend=False)


def fig_bars(df: pd.DataFrame, category: str, series: list[str], th: dict,
             title: str, axis_title: str = "", horizontal: bool = True,
             height: int = 420, emphasis: str | None = None) -> go.Figure:
    """One bar series per column in `series`, in fixed slot order.

    `emphasis` names a single category to keep in slot 1 while the rest go muted —
    the one-number story told without eight hues.
    """
    fig = go.Figure()
    cats = df[category].astype(str)
    for i, col in enumerate(series):
        if emphasis is not None:
            colors = [th["series"][0] if c == emphasis else th["muted"] for c in cats]
        else:
            colors = th["series"][i % len(th["series"])]
        common = dict(name=col, marker=dict(color=colors, line=dict(width=0)),
                      hovertemplate="%{customdata}<br>" + col + " %{value:.4g}"
                                    "<extra></extra>",
                      customdata=cats)
        if horizontal:
            fig.add_bar(y=cats, x=df[col], orientation="h", **common)
        else:
            fig.add_bar(x=cats, y=df[col], **common)

    fig.update_layout(title=title, bargap=0.35, bargroupgap=0.12)
    # A 2px surface gap between adjacent fills rather than a border around marks.
    fig.update_traces(marker_line_color=th["surface"], marker_line_width=2)
    if horizontal:
        fig.update_xaxes(title=axis_title)
        fig.update_yaxes(autorange="reversed", showgrid=False)
    else:
        fig.update_yaxes(title=axis_title)
        fig.update_xaxes(showgrid=False)
    return apply_theme(fig, th, height, legend=len(series) > 1)


def fig_lines(df: pd.DataFrame, x: str, series: dict[str, str], th: dict,
              title: str, y_title: str = "", x_title: str = "",
              colors: list[str] | None = None, height: int = 420,
              label_last: bool = True) -> go.Figure:
    """2px lines, ≥8px markers, endpoint direct labels rather than a value per point."""
    palette = colors or th["series"]
    fig = go.Figure()
    for i, (col, label) in enumerate(series.items()):
        if col not in df:
            continue
        colour = palette[i % len(palette)]
        g = df[[x, col]].dropna()
        fig.add_scatter(x=g[x], y=g[col], mode="lines+markers", name=label,
                        line=dict(color=colour, width=2),
                        marker=dict(size=8, color=colour,
                                    line=dict(color=th["surface"], width=2)),
                        hovertemplate=f"{label}<br>%{{x}} · %{{y:.4g}}<extra></extra>")
        if label_last and len(g):
            fig.add_annotation(x=g[x].iloc[-1], y=g[col].iloc[-1], text=label,
                               showarrow=False, xanchor="left", xshift=8,
                               font=dict(color=th["ink2"], size=11))
    fig.update_layout(title=title)
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return apply_theme(fig, th, height, legend=len(series) > 1)


def fig_scatter(df: pd.DataFrame, x: str, y: str, th: dict, title: str,
                color_by: str | None = None, highlight: list | None = None,
                hover_cols: list[str] | None = None, continuous: bool = False,
                height: int = 520, x_title: str = "", y_title: str = "") -> go.Figure:
    """A scatter is an all-pairs form, so identity colour caps at three slots.

    Past three, or for any high-cardinality key, `highlight` switches to
    emphasis: the chosen entities take the leading slots and everything else goes
    muted and translucent. Continuous keys (age, season) use the sequential ramp,
    which has no pair limit.
    """
    hover_cols = hover_cols or []
    fig = go.Figure()

    def _hover(sub: pd.DataFrame) -> tuple[str, np.ndarray]:
        if not hover_cols:
            return f"{x} %{{x:.3g}}<br>{y} %{{y:.3g}}<extra></extra>", None
        tpl = "<br>".join(f"%{{customdata[{i}]}}" for i in range(len(hover_cols)))
        return (tpl + f"<br>{x} %{{x:.3g}}<br>{y} %{{y:.3g}}<extra></extra>",
                sub[hover_cols].to_numpy())

    if continuous and color_by:
        tpl, cd = _hover(df)
        fig.add_scattergl(
            x=df[x], y=df[y], mode="markers", customdata=cd, hovertemplate=tpl,
            marker=dict(size=6, opacity=0.75, color=df[color_by],
                        colorscale=th["sequential"], showscale=True,
                        colorbar=dict(title=dict(text=color_by, side="right",
                                                 font=dict(size=11)),
                                      thickness=10, outlinewidth=0,
                                      tickfont=dict(color=th["muted"], size=10))))
        legend = False
    elif color_by and highlight:
        rest = df[~df[color_by].isin(highlight)]
        tpl, cd = _hover(rest)
        fig.add_scattergl(x=rest[x], y=rest[y], mode="markers", name="everything else",
                          customdata=cd, hovertemplate=tpl,
                          marker=dict(size=5, color=th["muted"], opacity=0.30))
        for i, key in enumerate(highlight[:ALL_PAIRS_CAP]):
            sub = df[df[color_by] == key]
            tpl, cd = _hover(sub)
            fig.add_scattergl(x=sub[x], y=sub[y], mode="markers", name=str(key),
                              customdata=cd, hovertemplate=tpl,
                              marker=dict(size=7, color=th["series"][i], opacity=0.85,
                                          line=dict(color=th["surface"], width=1)))
        legend = True
    else:
        tpl, cd = _hover(df)
        fig.add_scattergl(x=df[x], y=df[y], mode="markers", name=y, customdata=cd,
                          hovertemplate=tpl,
                          marker=dict(size=5, color=th["series"][0], opacity=0.5))
        legend = False

    fig.update_layout(title=title, hovermode="closest")
    fig.update_xaxes(title=x_title or x)
    fig.update_yaxes(title=y_title or y)
    return apply_theme(fig, th, height, legend=legend)


def stat_tiles(items: list[tuple[str, str, str]]) -> None:
    """A row of hero numbers — the right form when the story is one value."""
    cols = st.columns(len(items))
    for col, (label, value, helptext) in zip(cols, items):
        col.metric(label, value, help=helptext)


# ── Artifact loading ──────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def features_dir() -> Path:
    return ROOT / load_cfg()["data"]["features_dir"]


def eda_dir() -> Path:
    return ROOT / load_cfg()["eda"]["output_dir"]


@st.cache_data(show_spinner=False)
def read_table(path_str: str, columns: tuple[str, ...] | None = None) -> pd.DataFrame:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return pd.read_parquet(path, columns=list(columns) if columns else None)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def read_pickle(path_str: str):
    with open(path_str, "rb") as f:
        return pickle.load(f)


def optional(path: Path, columns: tuple[str, ...] | None = None) -> pd.DataFrame | None:
    """Read an artifact, or explain which make target produces it and return None."""
    if not path.exists():
        st.warning(f"`{path.relative_to(ROOT)}` not found — run `make eda` to build it.")
        return None
    return read_table(str(path), columns)


@st.cache_data(show_spinner=False)
def archetype_names(tier: str) -> dict[int, str]:
    path = features_dir() / f"archetype_profiles_tier{tier}.parquet"
    if not path.exists():
        return {}
    prof = read_table(str(path), ("archetype", "archetype_name"))
    return {int(r.archetype): r.archetype_name for r in prof.itertuples()}


@st.cache_data(show_spinner=False)
def season_pairs(tier: str, column: str) -> pd.DataFrame:
    """Season t vs season t+1 for one column — a join over stored values, not a fit."""
    from src.eda.persistence import lagged_pairs

    path = features_dir() / f"season_matrix_roster_tier{tier}.parquet"
    keep = ("player_id", "player_name", "season", "min_total", column)
    frame = read_table(str(path), tuple(dict.fromkeys(keep)))
    pairs = lagged_pairs(frame, load_cfg()["data"]["seasons"], lag=1)
    return pairs.dropna(subset=[column, f"{column}_next"])


# ── Tab 1 · Coverage ──────────────────────────────────────────────────────────

def tab_coverage(th: dict, tier: str) -> None:
    st.subheader("Data coverage")
    note("Which raw families exist for which season, and how much of each roster the "
         "prior-season description actually reaches. The two holes this is built to "
         "surface are the tracking-era boundary and the share of roster minutes with "
         "no usable S-1 row.")

    cov = optional(features_dir() / "coverage_report.csv")
    if cov is not None:
        cov = cov[cov["tier"].isin(["A", "B"] if tier == "B" else ["A"])]
        grid = cov.pivot_table(index="family", columns="season", values="matched",
                               aggfunc="max", fill_value=0)
        st.plotly_chart(
            fig_heatmap(grid, th, "Players matched per family × season",
                        "players", height=520,
                        hover="%{y}<br>%{x} · %{z:.0f} players"),
            width="stretch")
        table_view(cov.sort_values(["season", "family"]), "Coverage — table view")

    st.markdown("---")
    st.markdown("**Roster coverage** — the share of each team-season's roster weight "
                "that is *observed* prior-season data rather than an imputed rookie prior.")
    ctx = optional(features_dir() / f"team_context_tier{tier}.parquet",
                   ("team_abbreviation", "season", "roster_coverage", "stats_source"))
    if ctx is None:
        return

    per_team = (ctx.groupby(["team_abbreviation", "season"])["roster_coverage"]
                .mean().reset_index())
    stat_tiles([
        ("Mean roster coverage", f"{per_team['roster_coverage'].mean():.1%}",
         "Share of roster weight described by a real prior season."),
        ("10th percentile", f"{per_team['roster_coverage'].quantile(0.10):.1%}",
         "The thin end — young, high-turnover rosters."),
        ("Worst team-season", f"{per_team['roster_coverage'].min():.1%}",
         "Never dropped; carried with a reliability weight instead."),
    ])
    grid = per_team.pivot_table(index="team_abbreviation", columns="season",
                                values="roster_coverage")
    st.plotly_chart(
        fig_heatmap(grid, th, "Roster coverage by team × season", "coverage",
                    height=620, hover="%{y} · %{x}<br>coverage %{z:.1%}"),
        width="stretch")
    table_view(per_team.sort_values("roster_coverage"), "Roster coverage — table view")


# ── Tab 2 · PCA ───────────────────────────────────────────────────────────────

def tab_pca(th: dict, tier: str, mode: str) -> None:
    st.subheader("PCA — style axes")
    note(f"Tier {tier}, `{mode}` standardization. `within_season` is era-neutral and "
         "is what feeds archetypes and modeling; `pooled` lets era show up as a "
         "visible trajectory.")

    tag = f"tier{tier}_{mode}"
    variance = optional(features_dir() / f"pca_{tag}_variance.csv")
    scores = optional(features_dir() / f"pca_{tag}_scores.parquet")
    loadings = optional(features_dir() / f"pca_{tag}_loadings.parquet")
    if variance is None or scores is None or loadings is None:
        return

    head = variance.head(25)
    fig = go.Figure()
    fig.add_bar(x=head["component"], y=head["explained_variance_ratio"],
                name="per component", marker=dict(color=th["series"][0],
                                                  line=dict(color=th["surface"], width=2)),
                hovertemplate="PC%{x}<br>%{y:.2%}<extra></extra>")
    fig.add_scatter(x=head["component"], y=head["cumulative"], name="cumulative",
                    mode="lines+markers", line=dict(color=th["series"][1], width=2),
                    marker=dict(size=8, color=th["series"][1],
                                line=dict(color=th["surface"], width=2)),
                    hovertemplate="PC%{x}<br>cumulative %{y:.1%}<extra></extra>")
    fig.update_layout(title="Scree — share of variance explained", bargap=0.35)
    fig.update_xaxes(title="component", showgrid=False)
    fig.update_yaxes(title="share of variance", tickformat=".0%")
    st.plotly_chart(apply_theme(fig, th, 380), width="stretch")
    table_view(variance.head(40), "Variance — table view")

    st.markdown("---")
    pcs = [c for c in scores.columns if c.startswith("pc")]
    c1, c2, c3 = st.columns(3)
    x_pc = c1.selectbox("x axis", pcs, index=0)
    y_pc = c2.selectbox("y axis", pcs, index=min(1, len(pcs) - 1))
    colour_by = c3.selectbox("colour by", ["archetype", "age", "season", "team"])

    arch_path = features_dir() / f"archetypes_tier{tier}.parquet"
    plot = scores.copy()
    if arch_path.exists():
        arch = read_table(str(arch_path), ("player_id", "season", "archetype"))
        plot = plot.merge(arch, on=["player_id", "season"], how="left")
        names = archetype_names(tier)
        plot["archetype_name"] = plot["archetype"].map(
            lambda a: names.get(int(a), str(a)) if pd.notna(a) else "unlabelled")

    hover = [c for c in ("player_name", "season", "archetype_name") if c in plot]
    if colour_by == "age":
        fig = fig_scatter(plot.dropna(subset=["age"]), x_pc, y_pc, th,
                          f"{x_pc} vs {y_pc}, coloured by age", color_by="age",
                          continuous=True, hover_cols=hover)
    elif colour_by == "season":
        fig = fig_scatter(plot, x_pc, y_pc, th,
                          f"{x_pc} vs {y_pc}, coloured by season",
                          color_by="season_start_year", continuous=True,
                          hover_cols=hover)
    elif colour_by == "team" and "team_abbreviation" in plot:
        teams = sorted(plot["team_abbreviation"].dropna().unique())
        picked = st.multiselect("highlight teams (max 3)", teams,
                                default=teams[:1], max_selections=ALL_PAIRS_CAP)
        fig = fig_scatter(plot, x_pc, y_pc, th, f"{x_pc} vs {y_pc}, teams highlighted",
                          color_by="team_abbreviation", highlight=picked,
                          hover_cols=hover)
    else:
        options = sorted(plot.get("archetype_name", pd.Series(dtype=str)).dropna().unique())
        picked = st.multiselect("highlight archetypes (max 3)", options,
                                default=options[:2], max_selections=ALL_PAIRS_CAP)
        note("A scatter compares every pair of colours at once, where only three "
             "slots clear the colour-blind separation floors — so this highlights "
             "up to three and mutes the rest rather than colouring all nine.")
        fig = fig_scatter(plot, x_pc, y_pc, th,
                          f"{x_pc} vs {y_pc}, archetypes highlighted",
                          color_by="archetype_name", highlight=picked, hover_cols=hover)
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    left, right = st.columns([1, 1])
    with left:
        pc = st.selectbox("loadings for", pcs, index=0, key="loadings_pc")
        top = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).index).head(15)
        st.plotly_chart(
            fig_bars(top[["feature", pc]], "feature", [pc], th,
                     f"What defines {pc}", axis_title="loading", height=460),
            width="stretch")
        table_view(loadings.reindex(
            loadings[pc].abs().sort_values(ascending=False).index)[["feature", pc]].head(40),
            "Loadings — table view")
    with right:
        players = sorted(scores["player_name"].dropna().unique()) if "player_name" in scores else []
        if players:
            who = st.selectbox("career trajectory", players,
                               index=players.index("LeBron James")
                               if "LeBron James" in players else 0)
            path = plot[plot["player_name"] == who].sort_values("season")
            fig = go.Figure()
            fig.add_scatter(x=path[x_pc], y=path[y_pc], mode="lines+markers+text",
                            text=path["season"].str[:4], textposition="top center",
                            textfont=dict(color=th["muted"], size=10),
                            line=dict(color=th["series"][0], width=2),
                            marker=dict(size=9, color=th["series"][0],
                                        line=dict(color=th["surface"], width=2)),
                            name=who,
                            hovertemplate="%{text}<br>" + f"{x_pc} %{{x:.2f}}"
                                          f"<br>{y_pc} %{{y:.2f}}<extra></extra>")
            fig.update_layout(title=f"{who} through style space")
            fig.update_xaxes(title=x_pc)
            fig.update_yaxes(title=y_pc)
            st.plotly_chart(apply_theme(fig, th, 460, legend=False),
                            width="stretch")
            table_view(path[[c for c in ("season", "age", x_pc, y_pc) if c in path]],
                       "Trajectory — table view")


# ── Tab 3 · Archetypes ────────────────────────────────────────────────────────

def tab_archetypes(th: dict, tier: str) -> None:
    st.subheader("Archetypes")
    note("k-means over the leading era-adjusted PCs, with k chosen from a GMM-BIC "
         "sweep. A team is then expressible as a minutes-weighted distribution over "
         "these — the input the eventual model consumes for own-team and opponent.")

    profiles = optional(features_dir() / f"archetype_profiles_tier{tier}.parquet")
    labelled = optional(features_dir() / f"archetypes_tier{tier}.parquet")
    if profiles is None or labelled is None:
        return

    names = archetype_names(tier)
    options = [f"{k} · {v}" for k, v in sorted(names.items())]
    picked = st.selectbox("archetype", options)
    cid = int(picked.split(" · ")[0])

    left, right = st.columns([1, 1])
    with left:
        row = profiles[profiles["archetype"] == cid].drop(
            columns=["archetype", "archetype_name"]).iloc[0]
        top = row.reindex(row.abs().sort_values(ascending=False).index).head(16)
        prof = pd.DataFrame({"feature": top.index, "z": top.to_numpy(dtype=float)})
        fig = go.Figure(go.Bar(
            y=prof["feature"], x=prof["z"], orientation="h",
            marker=dict(color=prof["z"], colorscale=th["diverging"],
                        cmid=0.0, line=dict(color=th["surface"], width=2)),
            hovertemplate="%{y}<br>z %{x:+.2f}<extra></extra>"))
        fig.update_layout(title=f"Cluster {cid} profile (z vs league)", bargap=0.35)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title="standardized deviation")
        st.plotly_chart(apply_theme(fig, th, 520, legend=False),
                        width="stretch")
        table_view(prof, "Cluster profile — table view")
    with right:
        members = labelled[labelled["archetype"] == cid]
        roster = (members.sort_values("min_total", ascending=False)
                  .drop_duplicates("player_name")
                  [[c for c in ("player_name", "season", "age", "min_total",
                                "dk_pts_per_game") if c in members]].head(25))
        st.markdown(f"**Representative members** — {len(members):,} player-seasons")
        st.dataframe(roster, width="stretch", hide_index=True)

    st.markdown("---")
    comp = optional(features_dir() / f"team_composition_tier{tier}.parquet")
    if comp is None:
        return
    share_cols = [c for c in comp.columns if c.startswith("share_arch")]
    teams = sorted(comp["team_abbreviation"].unique())
    team = st.selectbox("team composition", teams,
                        index=teams.index("LAL") if "LAL" in teams else 0)

    sub = comp[comp["team_abbreviation"] == team].sort_values("season")
    grid = sub.set_index("season")[share_cols].T
    grid.index = [names.get(int(c.removeprefix("share_arch")),
                            c) for c in grid.index]
    note("A heatmap rather than a stacked bar: nine archetypes exceed the eight "
         "categorical slots, and a single-hue magnitude ramp has no such limit.")
    st.plotly_chart(
        fig_heatmap(grid, th, f"{team} — minutes share by archetype", "share",
                    height=420, hover="%{y}<br>%{x} · %{z:.1%}"),
        width="stretch")
    table_view(sub[["season", "roster_size", "mean_age", "mean_height_inches",
                    "pace", "usage_hhi", *share_cols]], "Composition — table view")


# ── Tab 4 · Persistence ───────────────────────────────────────────────────────

def tab_persistence(th: dict, tier: str) -> None:
    st.subheader("Persistence — what survives a year")
    note("Pooled and within-season correlations side by side. Where they diverge "
         "sharply the pooled figure is measuring the calendar, not the player: "
         "three-point volume roughly doubled over the sample, so anything built "
         "from it tracks anything else that rose.")

    table = optional(eda_dir() / "persistence.csv")
    if table is None:
        return
    t = table[table["tier"] == tier].copy()
    if t.empty:
        st.info(f"No persistence rows for tier {tier}.")
        return

    stat_tiles([
        ("Features ranked", f"{len(t):,}", "Numeric columns with enough season pairs."),
        ("Season pairs", f"{int(t['n_pairs'].max()):,}",
         "Consecutive-season pairs on the inclusive roster frame."),
        ("Median fitted r∞", f"{t[t['minutes_weighted']]['reliability_r_inf'].median():.3f}",
         "Across per-36 columns. team_context's single fit uses 0.924."),
    ])

    n = st.slider("show top / bottom", 5, 30, 12)
    view = st.radio("ranked by", ["most persistent", "least persistent",
                                  "largest era gap"], horizontal=True)
    if view == "most persistent":
        show = t.nlargest(n, "r_within_season")
    elif view == "least persistent":
        show = t.nsmallest(n, "r_within_season")
    else:
        show = t.reindex(t["era_gap"].abs().sort_values(ascending=False).index).head(n)

    st.plotly_chart(
        fig_bars(show, "feature", ["r_pooled", "r_within_season"], th,
                 "Year-over-year correlation — pooled vs season-absorbed",
                 axis_title="r", height=max(360, 26 * len(show) + 140)),
        width="stretch")
    table_view(t[["feature", "n_pairs", "minutes_weighted", "r_pooled",
                  "r_within_season", "era_gap", "reliability_r_inf",
                  "reliability_m0", "minutes_for_r75"]].round(4),
               "Persistence — table view")

    st.markdown("---")
    feats = sorted(t["feature"])
    default = "adv_usg_pct" if "adv_usg_pct" in feats else feats[0]
    col = st.selectbox("season t vs t+1 for", feats, index=feats.index(default))
    row = t[t["feature"] == col].iloc[0]

    left, right = st.columns([3, 2])
    with left:
        try:
            pairs = season_pairs(tier, col)
        except Exception as exc:                        # missing column in the frame
            st.warning(f"Could not pair seasons for `{col}`: {exc}")
            pairs = None
        if pairs is not None and len(pairs):
            sample = pairs.sample(min(6000, len(pairs)), random_state=0)
            fig = fig_scatter(sample, col, f"{col}_next", th,
                              f"{col}: season t vs t+1",
                              hover_cols=[c for c in ("player_name", "season")
                                          if c in sample],
                              x_title="season t", y_title="season t+1", height=460)
            lo = float(np.nanpercentile(sample[col], 1))
            hi = float(np.nanpercentile(sample[col], 99))
            fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
                            line=dict(color=th["axis"], width=2), hoverinfo="skip")
            st.plotly_chart(fig, width="stretch")
    with right:
        st.metric("pooled r", f"{row['r_pooled']:.3f}")
        st.metric("within-season r", f"{row['r_within_season']:.3f}",
                  delta=f"{-row['era_gap']:+.3f} vs pooled")
        if np.isfinite(row.get("reliability_r_inf", np.nan)):
            m = np.geomspace(20, 3000, 60)
            curve = pd.DataFrame({
                "minutes": m,
                "reliability": row["reliability_r_inf"] * m / (m + row["reliability_m0"]),
            })
            st.plotly_chart(
                fig_lines(curve, "minutes", {"reliability": "fitted r(m)"}, th,
                          "Reliability vs prior-season minutes",
                          y_title="r", x_title="total minutes", height=300,
                          label_last=False),
                width="stretch")
            if np.isfinite(row.get("minutes_for_r75", np.nan)):
                note(f"Reaches r = 0.75 at {row['minutes_for_r75']:,.0f} total minutes.")


# ── Tab 5 · Aging ─────────────────────────────────────────────────────────────

def tab_aging(th: dict, tier: str) -> None:
    st.subheader("Aging curves")
    note("Delta method: the mean within-player change between consecutive seasons, "
         "era-absorbed and integrated. A cross-sectional curve would turn *up* at "
         "the top because weak veterans leave the league — both are plotted so the "
         "gap between them is visible rather than asserted.")

    curves = optional(eda_dir() / "aging_curves.csv")
    if curves is None:
        return
    c = curves[curves["tier"] == tier]
    if c.empty:
        st.info(f"No aging rows for tier {tier}.")
        return

    metrics = sorted(c["metric"].unique())
    default = "dk_linear_per36" if "dk_linear_per36" in metrics else metrics[0]
    metric = st.selectbox("metric", metrics, index=metrics.index(default))
    overall = c[(c["metric"] == metric) & (c["archetype_name"] == "all")].sort_values("age")

    if not overall.empty:
        anchor = overall["cross_sectional_mean"].iloc[
            int(np.argmin(np.abs(overall["cumulative"].to_numpy())))]
        compare = pd.DataFrame({
            "age": overall["age"],
            "delta method": overall["cumulative"],
            "cross-sectional": overall["cross_sectional_mean"] - anchor,
        })
        st.plotly_chart(
            fig_lines(compare, "age",
                      {"delta method": "delta method", "cross-sectional": "cross-sectional"},
                      th, f"{metric} — change from the anchor age, measured two ways",
                      y_title=f"{metric} vs anchor", x_title="age", height=440),
            width="stretch")
        note("Same units, one axis. The cross-sectional line staying flat or rising "
             "where the delta line falls *is* the survivorship bias.")

    st.markdown("---")
    arch_options = sorted(a for a in c["archetype_name"].unique() if a != "all")
    picked = st.multiselect("overlay archetypes (max 3)", arch_options,
                            default=arch_options[:2], max_selections=ALL_PAIRS_CAP)
    wide = overall[["age", "cumulative"]].rename(columns={"cumulative": "all"})
    series = {"all": "all players"}
    for name in picked[:ALL_PAIRS_CAP]:
        g = c[(c["metric"] == metric) & (c["archetype_name"] == name)][["age", "cumulative"]]
        wide = wide.merge(g.rename(columns={"cumulative": name}), on="age", how="left")
        series[name] = name
    st.plotly_chart(
        fig_lines(wide, "age", series, th, f"{metric} — integrated age curve",
                  y_title="cumulative change", x_title="age", height=460,
                  label_last=False),
        width="stretch")

    table_view(c[c["metric"] == metric][
        ["archetype_name", "age", "n_pairs", "n_players", "mean_delta",
         "mean_delta_era_adj", "se", "cumulative", "cumulative_ratio",
         "cross_sectional_mean"]].round(4), "Aging — table view")


# ── Tab 6 · Target ────────────────────────────────────────────────────────────

def tab_target(th: dict) -> None:
    st.subheader("Target — distribution and the running season total")
    note("The mean-variance relationship per component decides a GLM's family, a "
         "GBM's objective, and whether multihead.py's Poisson-with-minutes-offset "
         "is correctly specified. The column that decides it is the *within* one: "
         "dispersion around each player-season's own mean, at fixed exposure.")

    prof = optional(eda_dir() / "target_profile.csv")
    if prof is None:
        return

    dist = prof[prof["analysis"] == "distribution"]
    overall = dist[dist["bucket_kind"] == "all"]
    minutes = dist[dist["bucket_kind"] == "minutes"]

    if not minutes.empty:
        cfg = load_cfg()["eda"].get("target", {})
        edges = cfg.get("minutes_buckets", [0, 5, 12, 18, 24, 30, 48])
        order = [f"{edges[i]:g}-{edges[i + 1]:g}" for i in range(len(edges) - 1)]
        grid = minutes.pivot_table(index="metric", columns="bucket",
                                   values="var_over_mean_within")
        grid = grid.reindex(columns=[o for o in order if o in grid.columns])
        metrics_order = [m for m in ["dk_pts", "pts", "fgm", "fg2m", "fg3m", "ftm",
                                     "fga", "fg2a", "fg3a", "fta", "reb", "ast",
                                     "stl", "blk", "tov"] if m in grid.index]
        st.plotly_chart(
            fig_heatmap(grid.reindex(metrics_order), th,
                        "Within-player dispersion by minutes played", "var / mean",
                        diverging=True, zmid=1.0, height=400,
                        hover="%{y} · %{x} min<br>var/mean %{z:.2f}"),
            width="stretch")
        note("Neutral = 1.0, where Poisson is correctly specified. Blue is "
             "under-dispersed, red over-dispersed. `pts` sits at ~2.2–2.5 at every "
             "exposure, but the shot classes it decomposes into — `fg2m`, `fg3m`, "
             "`fg3a` — sit at ~1.0: the overdispersion is the 2× weight on field "
             "goals squaring into the variance, not a property of scoring. Free "
             "throws stay near 2 because they arrive in pairs.")

    if not overall.empty:
        st.plotly_chart(
            fig_bars(overall.sort_values("var_over_mean_within"), "metric",
                     ["var_over_mean_within"], th,
                     "Within-player dispersion, pooled", axis_title="var / mean",
                     height=360),
            width="stretch")
    table_view(dist[["metric", "bucket_kind", "bucket", "n", "mean", "var",
                     "var_over_mean", "var_over_mean_within",
                     "dispersion_alpha_within", "zero_share", "skew"]].round(4),
               "Distribution — table view")

    st.markdown("---")
    st.markdown("**Running season total** — how much of the final total is already "
                "settled by game *k*.")
    pred = prof[prof["analysis"] == "season_total"]
    traj = optional(eda_dir() / "target_trajectories.parquet")

    left, right = st.columns([1, 1])
    with left:
        if not pred.empty:
            p = pred.copy()
            p["k"] = p["bucket"].astype(int)
            p = p.sort_values("k")
            st.plotly_chart(
                fig_lines(p, "k", {"r2_extrapolated": "held-in R²", "r": "correlation"},
                          th, "First k games vs the final total",
                          y_title="", x_title="games observed", height=380,
                          label_last=False),
                width="stretch")
            table_view(p[["bucket", "n", "r", "r2_extrapolated", "mae", "mean"]].round(4),
                       "Predictiveness — table view")
    with right:
        if traj is not None and not traj.empty:
            deciles = sorted(traj["decile"].dropna().unique())
            colours = ordinal_colors(th, len(deciles))
            fig = go.Figure()
            for colour, d in zip(colours, deciles):
                g = traj[traj["decile"] == d].sort_values("game_index")
                fig.add_scatter(x=g["game_index"], y=g["mean_cumulative"],
                                mode="lines", name=f"decile {int(d) + 1}",
                                line=dict(color=colour, width=2),
                                hovertemplate=f"decile {int(d) + 1}<br>"
                                              "game %{x}<br>%{y:,.0f} dk_pts"
                                              "<extra></extra>")
            fig.update_layout(title="Cumulative dk_pts by final-total decile")
            fig.update_xaxes(title="game index")
            fig.update_yaxes(title="cumulative dk_pts")
            st.plotly_chart(apply_theme(fig, th, 380), width="stretch")
            note("Deciles are ordered, so this uses the ordinal ramp rather than ten "
                 "categorical hues.")
            table_view(traj.round(2), "Trajectories — table view")


# ── Tab 7 · Team context ──────────────────────────────────────────────────────

def tab_team_context(th: dict, tier: str) -> None:
    st.subheader("Own-team context")
    note("What knowing P's *next*-season teammates adds, once his own prior season "
         "is controlled for — and with season absorbed, without which spacing and "
         "pace read as era drift in a team-context costume.")

    value = optional(eda_dir() / f"team_context_value_tier{tier}.csv")
    if value is not None:
        st.plotly_chart(
            fig_bars(value.sort_values("delta_r2", ascending=False), "feature",
                     ["delta_r2"], th,
                     "Incremental R² on per-game dk_pts, above the controls",
                     axis_title="ΔR²", height=380,
                     emphasis="teammate_usage_load"),
            width="stretch")
        note("`teammate_usage_load` carries the block; it exists only under the "
             "roster(S) × stats(S-1) construction.")

        r_cols = [c for c in value.columns if c.startswith("r_")]
        grid = value.set_index("feature")[r_cols]
        grid.columns = [c.removeprefix("r_") for c in grid.columns]
        st.plotly_chart(
            fig_heatmap(grid, th, "Partial correlation per feature × outcome",
                        "partial r", diverging=True, zmid=0.0, height=420,
                        hover="%{y} vs %{x}<br>r %{z:+.3f}"),
            width="stretch")
        note("Signs disagree across components and largely cancel in the DK sum — "
             "the measured argument for component heads over a single dk_pts head.")
        table_view(value.round(4), "Context value — table view")

    st.markdown("---")
    ctx = optional(features_dir() / f"team_context_tier{tier}.parquet",
                   ("player_id", "season", "team_abbreviation", "stats_source",
                    "reliability", "roster_coverage"))
    if ctx is None:
        return
    mix = (ctx.groupby(["season", "stats_source"]).size()
           .rename("n").reset_index())
    mix["share"] = mix["n"] / mix.groupby("season")["n"].transform("sum")
    wide = mix.pivot_table(index="season", columns="stats_source",
                           values="share", fill_value=0.0).reset_index()

    fig = go.Figure()
    for i, source in enumerate([s for s in ("prior", "stale", "rookie") if s in wide]):
        fig.add_bar(x=wide["season"], y=wide[source], name=source,
                    marker=dict(color=th["series"][i],
                                line=dict(color=th["surface"], width=2)),
                    hovertemplate=f"{source}<br>%{{x}} · %{{y:.1%}}<extra></extra>")
    fig.update_layout(title="Roster rows by prior-data source", barmode="stack",
                      bargap=0.25)
    fig.update_yaxes(title="share of roster rows", tickformat=".0%")
    fig.update_xaxes(showgrid=False)
    st.plotly_chart(apply_theme(fig, th, 400), width="stretch")
    table_view(wide.round(4), "Source mix — table view")


# ── Tab 8 · Opponent ──────────────────────────────────────────────────────────

def tab_opponent(th: dict, tier: str) -> None:
    st.subheader("Opponent encoding")
    note("Team defensive quality read straight off the team families — no player "
         "aggregation, so none of the roster-coverage bias — plus a low-rank "
         "matchup interaction between player style and opponent defense.")

    # `make opponent` fits tier A only, and the tier is not really a property of
    # this analysis: the defensive profile comes from the team families, which have
    # no tier at all. Only the player-style half reads a tier's PCA. So rather than
    # claim a missing artifact, say which tier is on screen.
    if not (eda_dir() / f"opponent_matchup_tier{tier}.csv").exists():
        st.info(f"The opponent encoder is fitted on tier A, so tier {tier} has no "
                "artifacts of its own — showing tier A. The defensive profile is "
                "built from the team families and is tier-independent; only the "
                "player-style side of the matchup term uses a tier's PCA.")
        tier = "A"

    matchup = optional(eda_dir() / f"opponent_matchup_tier{tier}.csv")
    if matchup is not None:
        m = matchup.copy()
        for c in ("r2_main_effect", "interaction_above_null"):
            m[c] = m[c] * 100
        st.plotly_chart(
            fig_bars(m.sort_values("r2_main_effect", ascending=False), "outcome",
                     ["r2_main_effect", "interaction_above_null"], th,
                     "Held-out % of within-player-season residual variance",
                     axis_title="% of residual variance", height=420),
            width="stretch")
        note("The main effect beats the interaction ~17× on dk_pts, but the "
             "interaction nearly doubles the main effect on blocks — it earns its "
             "place per component, not on the aggregate.")
        table_view(matchup.round(6), "Matchup — table view")

    st.markdown("---")
    pca_path = features_dir() / f"opponent_pca_tier{tier}.pkl"
    cols_path = features_dir() / f"opponent_profile_cols_tier{tier}.pkl"
    if pca_path.exists() and cols_path.exists():
        pca = read_pickle(str(pca_path))
        prof_cols = read_pickle(str(cols_path))
        loadings = pd.DataFrame(
            pca.components_.T, index=prof_cols,
            columns=[f"opp_pc{i + 1}" for i in range(pca.n_components_)])
        pc = st.selectbox("defensive profile axis", list(loadings.columns))
        share = pca.explained_variance_ratio_[list(loadings.columns).index(pc)]
        top = loadings.reindex(loadings[pc].abs().sort_values(ascending=False).index).head(14)
        bar = pd.DataFrame({"column": top.index, "loading": top[pc].to_numpy()})
        fig = go.Figure(go.Bar(
            y=bar["column"], x=bar["loading"], orientation="h",
            marker=dict(color=bar["loading"], colorscale=th["diverging"], cmid=0.0,
                        line=dict(color=th["surface"], width=2)),
            hovertemplate="%{y}<br>loading %{x:+.3f}<extra></extra>"))
        fig.update_layout(title=f"{pc} — {share:.1%} of profile variance", bargap=0.35)
        fig.update_yaxes(autorange="reversed", showgrid=False)
        fig.update_xaxes(title="loading")
        st.plotly_chart(apply_theme(fig, th, 460, legend=False), width="stretch")
        table_view(loadings.reset_index().rename(columns={"index": "column"}).round(4),
                   "Defensive loadings — table view")

    profile = features_dir() / f"opponent_profile_tier{tier}.parquet"
    if not profile.exists():
        st.info("Team positions in defensive-profile space need "
                f"`{profile.name}` — run `make opponent`.")
        return
    scores = read_table(str(profile))
    pcs = [c for c in scores.columns if c.startswith("opp_pc")]
    c1, c2 = st.columns(2)
    x_pc = c1.selectbox("x", pcs, index=0, key="opp_x")
    y_pc = c2.selectbox("y", pcs, index=min(1, len(pcs) - 1), key="opp_y")

    seasons = sorted(scores["season"].unique())
    picked_seasons = st.multiselect("seasons", seasons, default=seasons[-3:])
    sub = scores[scores["season"].isin(picked_seasons)] if picked_seasons else scores
    st.plotly_chart(
        fig_scatter(sub, x_pc, y_pc, th, "Teams in defensive-profile space",
                    hover_cols=[c for c in ("team_id", "season") if c in sub],
                    height=480),
        width="stretch")
    table_view(sub.round(3), "Team profiles — table view")


# ── Tab 9 · Feature diagnostics ───────────────────────────────────────────────

def tab_feature_diagnostics(th: dict, tier: str) -> None:
    st.subheader("Feature diagnostics")
    note("Not about a finding — about which model families can safely use which "
         "inputs. Collinearity is a GLM problem, cold-start cardinality a GBM and "
         "embedding problem, and an importance number without its own chance level "
         "is a problem for all three.")

    diag = optional(eda_dir() / "feature_diagnostics.csv")
    if diag is None:
        return

    summary = diag[(diag["analysis"] == "collinearity_summary") & (diag["tier"] == tier)]
    if not summary.empty:
        s = summary.iloc[0]
        stat_tiles([
            ("Features", f"{int(s['n_features'])}",
             "Within-season z-scored columns in the block."),
            ("Matrix rank", f"{int(s['matrix_rank'])}",
             "Below the feature count means exactly redundant columns."),
            ("VIF ≥ flag", f"{int(s['n_vif_above_flag'])}",
             "A GLM must regularize or drop; a tree does not care."),
            ("Clusters at |r| ≥ 0.95", f"{int(s['n_clusters'])}",
             f"Largest holds {int(s['largest_cluster'])} columns."),
        ])

    col = diag[(diag["analysis"] == "collinearity") & (diag["tier"] == tier)]
    if not col.empty:
        top_n = st.slider("features in the correlation heatmap", 10, 60, 30,
                          help="Ranked by VIF — the columns a GLM has to resolve.")
        R = optional(eda_dir() / f"feature_correlation_tier{tier}.parquet")
        if R is not None:
            picked = [f for f in col["name"].head(top_n) if f in R.columns]
            st.plotly_chart(
                fig_heatmap(R.loc[picked, picked], th,
                            f"Correlation among the {len(picked)} highest-VIF features",
                            "r", diverging=True, zmid=0.0, height=620,
                            hover="%{y}<br>%{x}<br>r %{z:+.3f}"),
                width="stretch")
            note("Correlation is polar, so this is the diverging pair with a neutral "
                 "midpoint at zero — never a rainbow.")
        table_view(col[["name", "vif", "max_abs_corr", "max_corr_partner",
                        "cluster_id", "cluster_size"]], "Collinearity — table view")

    st.markdown("---")
    card = diag[(diag["analysis"] == "cardinality") & (diag["tier"] == tier)]
    if not card.empty:
        st.plotly_chart(
            fig_bars(card, "name", ["n_categories", "n_effective"], th,
                     "Nominal vs effective category count", axis_title="categories",
                     height=340),
            width="stretch")
        note("`team_abbreviation` splits relocations into separate cold-start "
             "categories; `team_id` keeps SEA→OKC and NJN→BKN joined to their own "
             "history, which is why the opponent encoder keys on the id.")
        table_view(card[["name", "n_categories", "n_effective", "min_n", "median_n",
                         "n_below_threshold", "min_seasons_spanned", "cold_start"]]
                   .round(2), "Cardinality — table view")

    st.markdown("---")
    st.markdown("**The shuffled-null helper**, checked against the one-off it "
                "generalizes from.")
    repro = diag[diag["analysis"] == "null_reproduction"]
    if not repro.empty:
        r = repro.copy()
        for c in ("statistic", "null_mean", "above_null",
                  "variance_ceiling_above_null"):
            r[c] = r[c] * 100
        st.plotly_chart(
            fig_bars(r, "name", ["above_null", "variance_ceiling_above_null"], th,
                     "Opponent × archetype × season, above a shuffled null",
                     axis_title="% of within-player-season residual variance",
                     height=320, horizontal=True),
            width="stretch")
        note("Same data, same cells: permuting opponent within season gives ~+0.96%, "
             "permuting the archetype label ~+0.31%. That 3× swing is why the helper "
             "makes naming the permuted marginal mandatory — and the second bar is "
             "`opponent.py::variance_ceiling` on identical rows, so the "
             "generalization cannot silently drift from its source.")
        table_view(repro.round(6), "Null reproduction — table view")

    abl = diag[diag["analysis"] == "sequence_ablation"]
    if not abl.empty:
        a = abl.iloc[0]
        st.markdown("**Sequence vs. aggregate** — is the NN's LSTM trunk over "
                    "prior-season game logs earning its complexity?")
        stat_tiles([
            ("Aggregates only", f"{a['r2_aggregate']:.4f}",
             "Held-out R² on next-season dk_pts/game."),
            ("+ order features", f"{a['r2_sequence']:.4f}",
             f"{a['delta_sequence']:+.4f} over aggregates alone."),
            ("Shuffled order", f"{a['r2_sequence_shuffled']:.4f}",
             "Same features, same parameter count, order destroyed."),
            ("In the ordering", f"{a['delta_above_null']:+.4f}",
             "What order carries once the parameter count is matched."),
        ])
        bars = pd.DataFrame({
            "arm": ["aggregates only", "+ order features", "shuffled order"],
            "r2": [a["r2_aggregate"], a["r2_sequence"], a["r2_sequence_shuffled"]],
        })
        st.plotly_chart(
            fig_bars(bars, "arm", ["r2"], th, "Held-out R² by feature set",
                     axis_title="R²", height=280, emphasis="+ order features"),
            width="stretch")
        table_view(abl.round(6), "Sequence ablation — table view")


# ── App ───────────────────────────────────────────────────────────────────────

TABS = ["Coverage", "PCA", "Archetypes", "Persistence", "Aging", "Target",
        "Team context", "Opponent", "Feature diagnostics"]


def detected_mode() -> str:
    """Follow Streamlit's own theme where it exposes one."""
    for get in (lambda: st.context.theme.type, lambda: st.get_option("theme.base")):
        try:
            value = get()
        except Exception:
            continue
        if value in ("light", "dark"):
            return value
    return "light"


def main() -> None:
    st.set_page_config(page_title="NBA season-level EDA", layout="wide",
                       page_icon="🏀")
    st.title("NBA season-level EDA")
    st.caption("Reads the artifacts `make eda` produced. Nothing is refitted here.")

    # One filter row above everything it scopes, rather than per-chart controls.
    with st.sidebar:
        st.header("Scope")
        tier = st.radio("Tier", ["A", "B"], horizontal=True,
                        help="A: 30 seasons, box-score families. "
                             "B: 13 seasons (2013-14+), adds tracking and hustle.")
        mode = st.radio("Era mode", ["within_season", "pooled"],
                        help="within_season is era-neutral and feeds modeling; "
                             "pooled makes era a visible axis.")
        default = detected_mode()
        appearance = st.radio("Appearance", ["light", "dark"],
                              index=0 if default == "light" else 1,
                              horizontal=True,
                              help="Chart steps are selected per mode, not flipped.")
        st.markdown("---")
        st.caption("Artifacts: `data/features/` (model inputs) and `outputs/eda/` "
                   "(analysis reports).")

    th = theme(appearance)
    tabs = st.tabs(TABS)
    with tabs[0]:
        tab_coverage(th, tier)
    with tabs[1]:
        tab_pca(th, tier, mode)
    with tabs[2]:
        tab_archetypes(th, tier)
    with tabs[3]:
        tab_persistence(th, tier)
    with tabs[4]:
        tab_aging(th, tier)
    with tabs[5]:
        tab_target(th)
    with tabs[6]:
        tab_team_context(th, tier)
    with tabs[7]:
        tab_opponent(th, tier)
    with tabs[8]:
        tab_feature_diagnostics(th, tier)


if __name__ == "__main__":
    main()
