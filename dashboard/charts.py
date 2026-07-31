"""Generic figure builders, moved verbatim from the pre-split `app.py`.

Kept whole even though tab 3's figures are deferred: they are tested, reusable, and
needed the moment a figure section comes back. Like `theme.py` this imports no
Streamlit — a figure is built here and handed to `st.plotly_chart` by a tab.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import ALL_PAIRS_CAP, apply_theme


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
