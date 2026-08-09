"""The two figures this dashboard draws: the radial fingerprint and a loadings panel.

Both obey the palette rules `theme.py` documents. Two of those bind here in particular:

- **Colour never carries a value alone.** The radial chart is read off labelled rings at
  −2/−1/0/+1/+2 SD and ships a table twin; every loadings panel ships one too.
- **Polarity gets the diverging pair, not two categorical slots.** A loading's sign is
  a direction on one axis, so the bars take the two ends of `th["diverging"]` — the
  same encoding the project's heatmaps use for a signed quantity.

**The radial chart is drawn in cartesian coordinates rather than on a `polar` subplot,
and that is not a style choice.** Streamlit's `st.plotly_chart(on_select=...)` returns an
empty selection for every click on a polar trace — verified against a cartesian scatter
in the same app, which selects normally — so a `Scatterpolar` fingerprint cannot be
clicked, and clicking a spoke to open its component is the whole interaction. Placing the
points by hand costs the polar grid, which is redrawn here as shapes and annotations, and
buys back exact control over the ring labels that `polar` would not give either: it
rotates radial tick text by `angle - tickangle` and ignores `ticksuffix` outright.

Like `theme.py` this imports no Streamlit: a figure is built here and handed to
`st.plotly_chart` by `app.py`.
"""

import math

import pandas as pd
import plotly.graph_objects as go

from dashboard.theme import apply_theme

#: The first spoke sits at the top and the rest run counterclockwise, so components
#: 1–5 fall down the left of the circle in order and 6–10 climb the right.
ANGULAR_ROTATION = 90
ANGULAR_DIRECTION = "counterclockwise"

#: The radial tick arm, placed halfway between the first and last spokes so the ring
#: labels never sit on top of a data line.
TICK_ARM_ANGLE = 72

#: Marker sizes. The spokes are click targets — clicking one opens its loadings panel —
#: so the base marker is larger than a purely decorative one would be, and the selected
#: spoke is enlarged again so the chart says what the panel beside it is explaining.
MARKER = 11
SELECTED_MARKER = 17
#: An invisible trace under the visible one, so a click *near* a vertex still lands.
HIT_MARKER = 30

#: Unit-disc radii: the plot is drawn on a disc of radius 1, with the outer ring at
#: `+limit` SD and the centre at `−limit`. Labels sit outside the rim.
RIM = 1.0
SPOKE_LABEL_RADIUS = 1.17
AXIS_EXTENT = 1.34


def spoke_angle(index: int, n: int) -> float:
    """Degrees for spoke `index` — the first at the top, the rest counterclockwise."""
    return ANGULAR_ROTATION + 360.0 * index / n


def unit_radius(score: float, limit: float) -> float:
    """A clamped score in [−limit, +limit] onto the unit disc: centre 0, rim 1."""
    return (score + limit) / (2.0 * limit)


def _xy(radius: float, angle_deg: float) -> tuple[float, float]:
    radians = math.radians(angle_deg)
    return radius * math.cos(radians), radius * math.sin(radians)


def fig_radar(frame: pd.DataFrame, th: dict, name: str, limit: float = 2.0,
              overlay: pd.DataFrame | None = None, overlay_name: str = "",
              selected: str | None = None, height: int = 620) -> go.Figure:
    """One player-season's component scores as a closed polygon on a fixed axis.

    `frame` is `pca.fingerprint()`: one row per component with `label`, `title`, `sd`
    (the true score) and `radius` (that score pinned to ±`limit`). The polygon is drawn
    at `radius` and the hover reports `sd`, so a pinned spoke cannot quietly read as a
    2.0. An `overlay` frame of the same shape draws a second, unfilled polygon — two
    series, an adjacent pair in the validated slot order.

    `selected` names the spoke whose panel is open beside the chart; its marker is
    enlarged and ringed so the chart says which component it is currently explaining.
    """
    n = len(frame)
    angles = [spoke_angle(i, n) for i in range(n)]
    fig = go.Figure()

    _draw_grid(fig, th, angles, frame["label"], limit)
    if overlay is not None and len(overlay):
        _add_polygon(fig, overlay, th, angles, limit, overlay_name or "comparison",
                     colour=th["series"][1], fill=False, legendrank=2)
    _add_hit_layer(fig, frame, angles, limit)
    _add_polygon(fig, frame, th, angles, limit, name, colour=th["series"][0],
                 fill=True, legendrank=1, selected=selected)

    fig = apply_theme(fig, th, height,
                      legend=overlay is not None and len(overlay) > 0)
    axis = dict(range=[-AXIS_EXTENT, AXIS_EXTENT], showgrid=False, zeroline=False,
                showticklabels=False, showline=False, visible=False,
                constrain="domain")
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, scaleanchor="x", scaleratio=1)
    fig.update_layout(margin=dict(l=8, r=8, t=48, b=8), hovermode="closest",
                      dragmode=False)
    return fig


def _draw_grid(fig: go.Figure, th: dict, angles: list[float], labels, limit: float
               ) -> None:
    """The rings, the spokes and their labels — what a `polar` subplot would supply.

    Shapes and annotations rather than traces, so nothing here can be picked up as a
    clicked point.
    """
    ticks = [-limit, -limit / 2, 0.0, limit / 2, limit]
    for score in ticks:
        radius = unit_radius(score, limit)
        if radius <= 0:
            continue
        zero = score == 0
        fig.add_shape(type="circle", xref="x", yref="y",
                      x0=-radius, y0=-radius, x1=radius, y1=radius,
                      line=dict(color=th["muted"] if zero else th["grid"],
                                width=1.5 if zero else 1),
                      layer="below")

    for angle in angles:
        x, y = _xy(RIM, angle)
        fig.add_shape(type="line", x0=0, y0=0, x1=x, y1=y,
                      line=dict(color=th["grid"], width=1), layer="below")

    for angle, label in zip(angles, labels):
        x, y = _xy(SPOKE_LABEL_RADIUS, angle)
        fig.add_annotation(x=x, y=y, text=str(label), showarrow=False,
                           font=dict(color=th["ink2"], size=13))

    # Ring labels on their own arm, over an opaque chip so a data line crossing them
    # does not make them unreadable.
    for score in ticks:
        x, y = _xy(unit_radius(score, limit), TICK_ARM_ANGLE)
        unit = " SD" if abs(score) == limit else ""
        fig.add_annotation(x=x, y=y, text=f"{score:+.0f}{unit}" if score else "0",
                           showarrow=False, font=dict(color=th["muted"], size=10),
                           bgcolor=th["surface"], borderpad=2)


def _add_hit_layer(fig: go.Figure, frame: pd.DataFrame, angles: list[float],
                   limit: float) -> None:
    xs, ys = zip(*[_xy(unit_radius(r, limit), a)
                   for r, a in zip(frame["radius"], angles)])
    fig.add_trace(go.Scatter(
        x=list(xs), y=list(ys), mode="markers", name="", showlegend=False,
        hoverinfo="skip",
        marker=dict(size=HIT_MARKER, color="rgba(0,0,0,0)",
                    line=dict(width=0))))


def _add_polygon(fig: go.Figure, frame: pd.DataFrame, th: dict, angles: list[float],
                 limit: float, name: str, colour: str, fill: bool, legendrank: int,
                 selected: str | None = None) -> None:
    points = [_xy(unit_radius(r, limit), a)
              for r, a in zip(frame["radius"], angles)]
    xs = [p[0] for p in points] + [points[0][0]]
    ys = [p[1] for p in points] + [points[0][1]]

    # Score to one decimal, formatted here rather than by a d3 spec: the spec would be
    # applied to a value arriving from a mixed-dtype customdata array, which is how a
    # 1.7174781203 ends up on screen at full precision.
    custom = [[label, title, f"{sd:+.1f}"] for label, title, sd
              in zip(frame["label"], frame["title"], frame["sd"])]
    custom = custom + custom[:1]

    marker = dict(size=MARKER, color=colour)
    if fill:
        chosen = [label == selected for label in frame["label"]] + [
            frame["label"].iloc[0] == selected]
        symbols = ["circle-open" if p else "circle" for p in frame["pinned"]]
        symbols = symbols + symbols[:1]
        marker = dict(
            size=[SELECTED_MARKER if c else MARKER for c in chosen], color=colour,
            symbol=symbols,
            line=dict(color=[th["ink"] if c else th["surface"] for c in chosen],
                      width=[2 if c else 1 for c in chosen]))

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=name, legendrank=legendrank,
        fill="toself" if fill else None,
        fillcolor=_translucent(colour, 0.22) if fill else None,
        line=dict(color=colour, width=2), marker=marker, customdata=custom,
        hovertemplate="<b>%{customdata[0]}</b> %{customdata[1]}<br>"
                      "%{customdata[2]} SD<extra>" + name + "</extra>"))


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
    fig.update_xaxes(title="loading", zeroline=True, zerolinecolor=th["axis"],
                     zerolinewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig = apply_theme(fig, th, height or (76 + 26 * len(frame)), legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=44, b=32),
                      title=dict(font=dict(size=14)))
    return fig


def _translucent(hex_color: str, alpha: float) -> str:
    """`#2a78d6` → `rgba(42,120,214,0.22)`, for a fill under a solid stroke."""
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
