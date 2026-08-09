"""The two figures this dashboard draws: the radial fingerprint and a loadings panel.

Both obey the palette rules `theme.py` documents. Two of those bind here in particular:

- **Colour never carries a value alone.** The radial chart is read off a fixed radial
  axis with labelled rings, and every loadings panel ships a table twin in an expander.
- **Polarity gets the diverging pair, not two categorical slots.** A loading's sign is
  a direction on one axis, so the bars take the two ends of `th["diverging"]` — the
  same encoding the project's heatmaps use for a signed quantity.

Like `theme.py` this imports no Streamlit: a figure is built here and handed to
`st.plotly_chart` by `app.py`.
"""

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import apply_theme

#: The first spoke sits at the top and the rest run counterclockwise, so components
#: 1–5 fall down the left of the circle in order and 6–10 climb the right. The panels
#: beside the chart are laid out to match, which is the only reason the direction is
#: pinned here rather than left to plotly's default.
ANGULAR_ROTATION = 90
ANGULAR_DIRECTION = "counterclockwise"

#: The radial tick arm, placed halfway between the first and last spokes so the ring
#: labels never sit on top of a data line.
TICK_ARM_ANGLE = 72


def fig_radar(frame: pd.DataFrame, th: dict, name: str, limit: float = 2.0,
              overlay: pd.DataFrame | None = None, overlay_name: str = "",
              height: int = 620) -> go.Figure:
    """One player-season's component scores as a closed polygon on a fixed axis.

    `frame` is `pca.fingerprint()`: one row per component with `label`, `title`, `sd`
    (the true score) and `radius` (that score pinned to ±`limit`). The polygon is drawn
    at `radius` and the hover reports `sd`, so a pinned spoke cannot quietly read as a
    2.0. An `overlay` frame of the same shape draws a second, unfilled polygon —
    two series, an adjacent pair in the validated slot order.
    """
    fig = go.Figure()

    def close(seq):
        values = list(seq)
        return values + values[:1]

    labels = close(frame["label"])

    # The league-average ring, drawn first so both polygons sit above it. Not a data
    # series: it is the zero line of a radial axis, which plotly will not draw itself,
    # and it is the reference every radius on the chart is measured against.
    fig.add_trace(go.Scatterpolar(
        r=[0.0] * len(labels), theta=labels, mode="lines", name="league average",
        line=dict(color=th["muted"], width=1.5), hoverinfo="skip", showlegend=False))

    if overlay is not None and len(overlay):
        fig.add_trace(go.Scatterpolar(
            r=close(overlay["radius"]), theta=close(overlay["label"]),
            mode="lines+markers", name=overlay_name or "comparison",
            line=dict(color=th["series"][1], width=2), legendrank=2,
            marker=dict(size=6, color=th["series"][1]),
            customdata=close(overlay[["title", "sd"]].to_numpy()),
            hovertemplate="<b>%{theta}</b> %{customdata[0]}<br>"
                          "%{customdata[1]:+.2f} SD<extra>" +
                          (overlay_name or "comparison") + "</extra>"))

    # Open markers where the true score is off the axis, so a pinned spoke is visible
    # as pinned without reading the hover.
    symbols = ["circle-open" if p else "circle" for p in frame["pinned"]]
    fig.add_trace(go.Scatterpolar(
        r=close(frame["radius"]), theta=labels, mode="lines+markers", name=name,
        fill="toself", fillcolor=_translucent(th["series"][0], 0.22), legendrank=1,
        line=dict(color=th["series"][0], width=2),
        marker=dict(size=9, color=th["series"][0], symbol=close(symbols),
                    line=dict(color=th["surface"], width=1)),
        customdata=close(frame[["title", "sd"]].to_numpy()),
        hovertemplate="<b>%{theta}</b> %{customdata[0]}<br>"
                      "%{customdata[1]:+.2f} SD<extra>" + name + "</extra>"))

    fig.update_layout(polar=dict(
        bgcolor=th["surface"],
        # The tick arm sits between two spokes (72° = halfway from PC1 to PC10) so the
        # numbers never sit on top of a data line.
        # Two plotly-isms on the radial axis, both found by rendering it. A polar tick
        # label is rotated by `angle - tickangle`, so upright text needs the two equal
        # rather than `tickangle=0`; and `ticksuffix` is ignored here, so the unit is
        # written into `ticktext` instead of appended.
        radialaxis=dict(range=[-limit, limit],
                        tickvals=[-limit, -limit / 2, 0, limit / 2, limit],
                        ticktext=[f"−{limit:g} SD", f"−{limit / 2:g}", "0",
                                  f"+{limit / 2:g}", f"+{limit:g} SD"],
                        angle=TICK_ARM_ANGLE, tickangle=TICK_ARM_ANGLE,
                        gridcolor=th["grid"], linecolor=th["axis"],
                        tickfont=dict(color=th["muted"], size=10)),
        angularaxis=dict(rotation=ANGULAR_ROTATION, direction=ANGULAR_DIRECTION,
                         gridcolor=th["grid"], linecolor=th["axis"],
                         tickfont=dict(color=th["ink2"], size=12)),
    ))
    fig = apply_theme(fig, th, height, legend=overlay is not None and len(overlay) > 0)
    fig.update_layout(margin=dict(l=56, r=56, t=56, b=32))
    return fig


def fig_loadings(frame: pd.DataFrame, th: dict, title: str,
                 height: int | None = None) -> go.Figure:
    """A component's largest loadings as signed horizontal bars, largest first.

    `frame` is `pca.top_loadings()`. Positive and negative take the two ends of the
    diverging scale rather than two categorical slots, because the sign is a direction
    on one axis and not a second category.
    """
    negative, positive = th["diverging"][0][1], th["diverging"][-1][1]
    colors = [positive if v >= 0 else negative for v in frame["loading"]]
    fig = go.Figure(go.Bar(
        y=frame["pretty"], x=frame["loading"], orientation="h",
        marker=dict(color=colors, line=dict(color=th["surface"], width=2)),
        customdata=frame["feature"],
        hovertemplate="%{customdata}<br>loading %{x:+.3f}<extra></extra>"))
    fig.update_layout(title=title, bargap=0.3)
    fig.update_xaxes(title="", zeroline=True, zerolinecolor=th["axis"],
                     zerolinewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = apply_theme(fig, th, height or (58 + 22 * len(frame)), legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=42, b=24),
                      title_font=dict(size=13))
    return fig


def _translucent(hex_color: str, alpha: float) -> str:
    """`#2a78d6` → `rgba(42,120,214,0.22)`, for a fill under a solid stroke."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
