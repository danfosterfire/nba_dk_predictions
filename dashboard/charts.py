"""Every figure this dashboard draws — the fingerprint pair, then the tournament page.

All of them obey the palette rules `theme.py` documents. Three of those bind here in
particular:

- **Colour never carries a value alone.** The radial chart is read off labelled rings at
  −2/−1/0/+1/+2 SD; the tournament figures are read off labelled axes and carry printed
  values or a stated interval. Every one of them ships a table twin in the view.
- **Polarity gets the diverging pair, not two categorical slots.** A loading's sign is
  a direction on one axis, so the bars take the two ends of `th["diverging"]` — the
  same encoding the project's heatmaps use for a signed quantity.
- **`ALL_PAIRS_CAP` is three, and two of these charts would otherwise exceed it.** The
  survival curve carries five tournaments and the hurdle bar five bars, which are
  non-adjacent series in the sense that validation measured. Both use highlight-and-gray:
  the two tiers the sweep actually drafts into take slots 0 and 1, the other three take
  `th["muted"]`. That is also the right *editorial* encoding, since only two tiers have a
  portfolio, and using one encoding for both figures means the reader learns it once.

**The radial chart is drawn in cartesian coordinates rather than on a `polar` subplot,
and that is not a style choice.** Streamlit's `st.plotly_chart(on_select=...)` returns an
empty selection for every click on a polar trace — verified against a cartesian scatter
in the same app, which selects normally — so a `Scatterpolar` fingerprint cannot be
clicked, and clicking a spoke to open its component is the whole interaction. Placing the
points by hand costs the polar grid, which is redrawn here as shapes and annotations, and
buys back exact control over the ring labels that `polar` would not give either: it
rotates radial tick text by `angle - tickangle` and ignores `ticksuffix` outright.

**The dot-and-interval figures place their rows on a numeric y axis, not a categorical
one.** Two series at the same category collide exactly — plotly offsets grouped *bars* and
does not offset grouped scatter — so each row is an integer position and each series is
nudged off it by `SERIES_NUDGE`, with the arm names restored as tick text. The axis is
then inverted by range rather than by `autorange="reversed"`, so the best arm is the top
row and the nudges keep their sign.

Like `theme.py` this imports no Streamlit: a figure is built here and handed to
`st.plotly_chart` by a view.
"""

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from dashboard.theme import FONT, apply_theme

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


# ══ The tournament page ═══════════════════════════════════════════════════════

#: How far a series is nudged off its integer row so two intervals do not overlap.
SERIES_NUDGE = 0.22
#: Vertical room a facet header needs, in units of one row, so a one-arm facet does not
#: come out as a sliver of a subplot with its title on top of its only point.
FACET_HEADER_ROWS = 1.2
DOT = 9
INTERVAL_WIDTH = 2


def _tier_colors(th: dict, tournaments, targets: tuple[str, ...]) -> list[str]:
    """Highlight-and-gray over the five captured tournaments.

    `ALL_PAIRS_CAP` is 3 and these are five non-adjacent series, so only the two tiers the
    sweep drafts into take a categorical slot; the rest recede to `muted`. The slot a
    target takes is its position in `targets`, not its position in the data, so the two
    block-1 figures agree with each other whatever order the artifact happens to be in.
    """
    slots = {name: i for i, name in enumerate(targets)}
    return [th["series"][slots[t]] if t in slots else th["muted"]
            for t in tournaments]


def _reference_line(fig: go.Figure, x: float, th: dict, label: str,
                    color: str | None = None, faceted: bool = False,
                    at_floor: bool = False) -> None:
    """A labelled vertical reference. Solid, because `theme.py` bans dashes outright.

    `faceted` spans every subplot of a `make_subplots` figure; the label is added once, at
    the top of the paper, rather than once per facet. `at_floor` drops the label inside
    the plot instead, which is what a line near the middle of the x range needs — the
    legend sits on the top edge and a label there collides with whichever entry happens to
    be under it. Caught in a rendered PNG, on the tier whose null lands mid-axis.
    """
    placement = dict(row="all", col=1) if faceted else {}
    fig.add_vline(x=x, line=dict(color=color or th["axis"], width=1), layer="below",
                  **placement)
    fig.add_annotation(x=x, y=0.0 if at_floor else 1.0, yref="paper", text=label,
                       showarrow=False, xanchor="left",
                       yanchor="bottom", xshift=4, yshift=3 if at_floor else 0,
                       font=dict(color=color or th["ink2"], size=11),
                       bgcolor=th["surface"], borderpad=2)


def fig_survival(frame: pd.DataFrame, th: dict, targets: tuple[str, ...],
                 pretty=str, height: int = 400) -> go.Figure:
    """`P(reach round r)` for an exchangeable entry, five tournaments, log scale.

    `frame` is `strategy.survival_frame()`. The log axis is not decoration: the curve falls
    from 1 to 0.0014 over four rounds at `600k_shootaround`, and on a linear axis every
    round after the first is a flat line on the floor. What the picture is for is the
    **first** cut — 5 of 6 entries are gone in a round that pays nothing at all, in every
    one of the five captured structures — so Round 1 carries its own marker line.
    """
    fig = go.Figure()
    names = list(dict.fromkeys(frame["tournament"]))
    # Non-targets first, so a highlighted tier is never painted over by a gray one —
    # `20k_spin_move` and `88k_alley_oop` share an advance chain exactly (2/12 → 2/6 →
    # 2/6) and their curves coincide at every round.
    names = [n for n in names if n not in targets] + [n for n in targets if n in names]
    for rank, (name, color) in enumerate(zip(names, _tier_colors(th, names, targets))):
        sub = frame[frame["tournament"] == name]
        is_target = name in targets
        fig.add_trace(go.Scatter(
            x=sub["round"], y=sub["p_reach"], mode="lines+markers",
            name=pretty(name),
            legendrank=targets.index(name) if is_target else 10 + rank,
            line=dict(color=color, width=3 if is_target else 1.5),
            marker=dict(size=DOT if is_target else 6, color=color),
            hovertemplate="round %{x}<br>reaches %{y:.3%}<extra>"
                          + pretty(name) + "</extra>"))

    fig.add_vline(x=1, line=dict(color=th["axis"], width=1), layer="below")
    # Anchored at the floor rather than the top: the curves all start at 100%, so the
    # top-left corner is the one place on this chart a label cannot go.
    fig.add_annotation(
        x=1, y=0.02, yref="paper", text="Round 1 · zero consolation", showarrow=False,
        xanchor="left", yanchor="bottom", xshift=5,
        font=dict(color=th["ink2"], size=11), bgcolor=th["surface"], borderpad=2)

    fig.update_xaxes(title="round", tickmode="array", tickvals=[1, 2, 3, 4],
                     range=[0.85, 4.3])
    fig.update_yaxes(title="P(reach the round)", type="log", tickformat=".2%")
    return apply_theme(fig, th, height)


def fig_hurdle(frame: pd.DataFrame, th: dict, targets: tuple[str, ...],
               pretty=str, height: int = 300) -> go.Figure:
    """Rake expressed as the edge that just returns the entry fee, per tournament.

    `frame` is `strategy.contest_summary()`. The hurdle rather than the rake because a raw
    rake share is not denominated the way a measured edge is: losing 15% of the pool means
    beating the field by 17.6% to break even, not by 15%. Every bar prints its own value,
    so nothing here is reachable by colour alone.
    """
    frame = frame.sort_values("break_even_hurdle")
    colors = _tier_colors(th, frame["tournament"], targets)
    labels = [pretty(t) for t in frame["tournament"]]
    fig = go.Figure(go.Bar(
        y=labels, x=frame["break_even_hurdle"], orientation="h",
        marker=dict(color=colors, line=dict(color=th["surface"], width=2)),
        text=[f"{v:+.2%}" for v in frame["break_even_hurdle"]],
        textposition="outside", textfont=dict(color=th["ink2"], size=12),
        customdata=list(zip(frame["rake"], frame["entry_fee_per_team"])),
        hovertemplate="rake %{customdata[0]:.2%} at a $%{customdata[1]:,.0f} entry"
                      "<br>break-even edge %{x:+.2%}<extra>%{y}</extra>"))
    fig.update_xaxes(title="break-even edge hurdle", tickformat=".0%",
                     range=[0, float(frame["break_even_hurdle"].max()) * 1.28])
    fig.update_yaxes(showgrid=False)
    fig.update_layout(bargap=0.35)
    return apply_theme(fig, th, height, legend=False)


def fig_sweep(panel: pd.DataFrame, facets: list, th: dict, break_even: float,
              null_label: str = "null", row_height: int = 34) -> go.Figure:
    """Lift over the symmetric-field null for every swept arm, faceted by axis.

    `panel` is `strategy.sweep_panel()` and `facets` is `strategy.facets()`. One subplot
    per axis on a shared x, arms ordered best-first inside their own facet, and one series
    per swept season — two of them, an adjacent pair in the validated slot order.

    Two reference lines, and the second one needed a derivation. Zero is the null an
    exchangeable entry sits on. `break_even` is `strategy.break_even_lift()`, the hurdle
    converted out of return units into the survival units this axis is in, which the
    caller is expected to caption with the assumption that conversion makes.

    **The intervals here are unpaired and will cover most of the table**; that is a
    property of the level rather than of the differences, since a kind simulated season is
    kind to every arm at once. The paired block is where arms are separated from each
    other, and this one is where they are separated from the null.
    """
    heights = [len(arms) + FACET_HEADER_ROWS for _, _, arms in facets]
    total = sum(heights)
    fig = make_subplots(
        rows=len(facets), cols=1, shared_xaxes=True, vertical_spacing=0.012,
        row_heights=[h / total for h in heights],
        subplot_titles=[label for _, label, _ in facets])
    for note in fig.layout.annotations:                     # the facet headers
        note.update(font=dict(color=th["ink"], size=13), x=0, xanchor="left")

    seasons = sorted(panel["season"].astype(str).unique())
    offsets = _nudges(len(seasons))
    for r, (axis, _, arms) in enumerate(facets, start=1):
        position = {arm: i for i, arm in enumerate(arms)}
        for s, season in enumerate(seasons):
            sub = panel[(panel["axis"] == axis)
                        & (panel["season"].astype(str) == season)]
            fig.add_trace(_interval_trace(
                x=sub["lift_vs_null"], lo=sub["lift_lo"], hi=sub["lift_hi"],
                y=[position[a] + offsets[s] for a in sub["strategy"]],
                name=season, color=th["series"][s], showlegend=(r == 1),
                text=list(sub["strategy"]), unit="lift",
                legendgroup=season), row=r, col=1)
        # The header room is taken out of the *range*, not just the row height: a
        # subplot that is taller without a taller range simply spreads its rows apart,
        # which leaves the facet title sitting on the last row of the facet above and
        # makes a one-arm facet twice as airy as a five-arm one. Extending the top of
        # the range by the same amount the row height was extended by keeps every row
        # the same number of pixels tall across all seven facets.
        fig.update_yaxes(tickmode="array", tickvals=list(position.values()),
                         ticktext=arms,
                         range=[len(arms) - 0.5, -0.5 - FACET_HEADER_ROWS],
                         showgrid=False, zeroline=False, row=r, col=1)

    fig = apply_theme(fig, th, height=int(total * row_height) + 60)
    fig.update_xaxes(title="", showticklabels=False)
    fig.update_xaxes(title="lift in P(top 2 of 12) over the null", tickformat=".2f",
                     showticklabels=True, row=len(facets), col=1)
    _reference_line(fig, 0.0, th, null_label, faceted=True)
    _reference_line(fig, break_even, th, "break-even", color=th["series"][2],
                    faceted=True)
    fig.update_layout(margin=dict(l=8, r=8, t=64, b=8), hovermode="closest")
    return fig


def fig_surfaces(panel: pd.DataFrame, th: dict, p_null: float, arms: tuple[str, ...],
                 height: int = 380) -> go.Figure:
    """The same arms on both backtest surfaces, on one `P(top 2 of 12)` axis.

    `panel` is `strategy.surfaces_panel()`. The two halves share an axis so their
    **widths** can be compared, which is the whole point: the simulated rows pool 500 drawn
    worlds a season and the realized rows have exactly one, so the honest readout's
    intervals are several times wider and cover the null in most rows. A figure that showed
    only the point estimates would invite exactly the comparison the block exists to
    forbid.

    A rule between the two blocks says which is which, and the row labels repeat it, so
    the split is never carried by position alone.
    """
    labels = list(dict.fromkeys(panel["label"]))
    position = {label: i for i, label in enumerate(labels)}
    offsets = _nudges(len(arms))
    fig = go.Figure()
    for a, arm in enumerate(arms):
        sub = panel[panel["strategy"] == arm]
        fig.add_trace(_interval_trace(
            x=sub["p_advance"], lo=sub["lo"], hi=sub["hi"],
            y=[position[label] + offsets[a] for label in sub["label"]],
            name=arm, color=th["series"][a], showlegend=True,
            text=list(sub["label"]), unit="P(top 2 of 12)"))

    n_simulated = sum(1 for label in labels if label.startswith("Simulated"))
    if 0 < n_simulated < len(labels):
        fig.add_hline(y=n_simulated - 0.5, line=dict(color=th["axis"], width=1),
                      layer="below")

    fig.update_yaxes(tickmode="array", tickvals=list(position.values()),
                     ticktext=labels, range=[len(labels) - 0.5, -0.5],
                     showgrid=False, zeroline=False)
    fig.update_xaxes(title="P(top 2 of 12) — surviving Round 1", tickformat=".0%")
    fig = apply_theme(fig, th, height)
    _reference_line(fig, p_null, th, "the null · 2 of 12", at_floor=True)
    return fig


def fig_paired(panel: pd.DataFrame, th: dict, baseline: str, unit: str,
               row_height: int = 26) -> go.Figure:
    """Paired gaps against one baseline, with the ones that do not resolve picked out.

    `panel` is `strategy.paired_panel()`. Differencing inside the simulated season removes
    the common term the unpaired sweep intervals are dominated by, so almost everything
    here separates — and the arms that still do not are the useful ones, because an
    unresolved gap is a decision the budget cannot make rather than a decision made.

    So they are **styled, not buried**: a second slot, an open marker echoing the
    fingerprint's pinned convention, and a heavier interval. The encoding is triply
    redundant — the interval visibly straddles the zero line, the marker is hollow, and the
    legend and table twin both name it — because no value on this dashboard may be
    reachable by colour alone.
    """
    arms = list(panel["strategy"])
    position = {arm: i for i, arm in enumerate(arms)}
    fig = go.Figure()
    groups = (("resolves", ~panel["crosses_zero"], th["series"][0], "circle", 2),
              ("does not resolve", panel["crosses_zero"], th["series"][1],
               "circle-open", 3))
    for name, mask, color, symbol, width in groups:
        sub = panel[mask]
        if sub.empty:
            continue
        fig.add_trace(_interval_trace(
            x=sub["gap"], lo=sub["gap_lo"], hi=sub["gap_hi"],
            y=[position[a] for a in sub["strategy"]],
            name=name, color=color, showlegend=True, text=list(sub["strategy"]),
            unit="gap", symbol=symbol, width=width))

    fig.update_yaxes(tickmode="array", tickvals=list(position.values()),
                     ticktext=arms, range=[len(arms) - 0.5, -0.5],
                     showgrid=False, zeroline=False)
    fig.update_xaxes(title=f"gap in {unit} against {baseline}", tickformat=".2f")
    fig = apply_theme(fig, th, height=int(len(arms) * row_height) + 110)
    _reference_line(fig, 0.0, th, f"{baseline} — the baseline")
    return fig


def _nudges(n: int) -> list[float]:
    """Row offsets for `n` series sharing one row, centred on it."""
    if n <= 1:
        return [0.0]
    return [(i - (n - 1) / 2.0) * SERIES_NUDGE for i in range(n)]


def _interval_trace(x, lo, hi, y, name: str, color: str, showlegend: bool,
                    text: list, unit: str, symbol: str = "circle",
                    width: int = INTERVAL_WIDTH, legendgroup: str | None = None
                    ) -> go.Scatter:
    """One dot-and-interval series: a point estimate with an asymmetric 95% interval."""
    x = list(x)
    lo, hi = list(lo), list(hi)
    return go.Scatter(
        x=x, y=list(y), mode="markers", name=name, showlegend=showlegend,
        legendgroup=legendgroup or name,
        marker=dict(size=DOT, color=color, symbol=symbol,
                    line=dict(color=color, width=2)),
        error_x=dict(type="data", symmetric=False,
                     array=[h - v for h, v in zip(hi, x)],
                     arrayminus=[v - l for v, l in zip(x, lo)],
                     color=color, thickness=width, width=0),
        customdata=[[t, f"{l:+.4f}", f"{h:+.4f}"] for t, l, h in zip(text, lo, hi)],
        hovertemplate="<b>%{customdata[0]}</b><br>" + unit +
                      " %{x:+.4f}<br>95% interval "
                      "[%{customdata[1]}, %{customdata[2]}]<extra>" + name + "</extra>")


# ══ The model pages ═══════════════════════════════════════════════════════════
#
# Six figures, one per block that has one, and every one of them is a *class* figure rather
# than a head figure: the same builder draws availability, the composition and the eleven
# component heads, because everything that differs between them arrives in the frame or in
# an axis label read from `model_card_index.csv`. That is what makes pages 4–6 configuration
# rather than a rewrite.
#
# Two palette rules bind here in particular. A **density is magnitude**, so it takes the
# single-hue sequential ramp with a colorbar, never a rainbow; a **coefficient is polarity**,
# so it takes the two ends of the diverging scale, the same encoding a PCA loading gets. And
# because three of these figures carry a colourbar, each one ships a table twin in the view:
# a reader must be able to get the number without reading a shade.

#: Small-multiple geometry for the feature grid. Four across is the widest that keeps a
#: 30-bin histogram legible in a page column, and 25 features (the composition) then come
#: out as seven rows of a tall figure — which is what "long scrolling pages are fine" buys.
HIST_COLUMNS = 4
HIST_ROW_HEIGHT = 150
#: A 2-D density's overlay: past a couple of thousand points a scatter is a blob, so the
#: emitter caps the sample and this draws it small and translucent under the reader's eye.
OVERLAY_SIZE = 3
OVERLAY_ALPHA = 0.22


def fig_features(panel: pd.DataFrame, th: dict, columns: int = HIST_COLUMNS,
                 title: str = "") -> go.Figure:
    """One histogram per design column, train filled and validation stepped over it.

    `panel` is `model_cards.histogram_panel()`. Both splits are drawn on the **one edge set
    the emitter pooled them onto** — comparing them is the block's whole job and two
    histograms on their own edges cannot be compared — and both are drawn as `density`, the
    share of that split's own rows, so a 751-row validation frame is comparable with an
    8,232-row training one.

    The two splits are an adjacent pair in the validated slot order, and they are also
    different *marks* — bars against a step line — so the comparison survives without
    colour, which is the relief rule the palette obliges.
    """
    features = list(dict.fromkeys(panel["feature"]))
    rows = max(1, math.ceil(len(features) / columns))
    # Spacing in *fractions*, so a taller grid needs a smaller one to leave the same gap in
    # pixels — but not so small that a subplot title lands on the tick labels of the row
    # above it, which is what 25 features did at the first cut.
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=features,
                        vertical_spacing=min(0.07, 0.45 / max(rows, 1)),
                        horizontal_spacing=0.06)

    for index, feature in enumerate(features):
        row, col = divmod(index, columns)
        row += 1
        col += 1
        for slot, split in enumerate(("train", "validation")):
            part = panel[(panel["feature"] == feature) & (panel["split"] == split)]
            if part.empty:
                continue
            color = th["series"][slot]
            common = dict(name=split.capitalize(), legendgroup=split,
                          showlegend=index == 0)
            if split == "train":
                fig.add_trace(go.Bar(
                    x=part["bin_center"], y=part["density"], width=part["bin_width"],
                    marker=dict(color=_translucent(color, 0.55),
                                line=dict(width=0)),
                    hovertemplate="%{x:.3g}<br>%{y:.1%} of train<extra></extra>",
                    **common), row=row, col=col)
            else:
                fig.add_trace(go.Scatter(
                    x=part["bin_center"], y=part["density"], mode="lines",
                    line=dict(color=color, width=1.5, shape="hvh"),
                    hovertemplate="%{x:.3g}<br>%{y:.1%} of validation<extra></extra>",
                    **common), row=row, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=11, color=th["ink2"])
    fig.update_xaxes(showgrid=False, tickfont=dict(size=9))
    fig.update_yaxes(showgrid=True, tickformat=".0%", tickfont=dict(size=9),
                     rangemode="tozero")
    fig.update_layout(title=title, bargap=0.02, barmode="overlay",
                      hovermode="closest")
    return apply_theme(fig, th, height=rows * HIST_ROW_HEIGHT + 90)


def fig_correlation(square: pd.DataFrame, th: dict, title: str = "",
                    height: int = 520) -> go.Figure:
    """The feature correlation matrix, diverging around zero, diagonal included.

    `square` is `model_cards.correlation_square()`. **Diverging rather than sequential**,
    because the sign of a correlation is a direction and not a magnitude, and pinned to
    [−1, +1] rather than to the data's own range so two heads' heatmaps mean the same thing.
    A constant column arrives as a row of `NaN` and is drawn as a gap on the surface — which
    is the point of shipping the whole square: the empty row stays on the axis and says so.
    """
    fig = go.Figure(go.Heatmap(
        z=square.to_numpy(dtype=float), x=list(square.columns), y=list(square.index),
        zmin=-1.0, zmax=1.0, zmid=0.0, colorscale=th["diverging"],
        hoverongaps=False,
        colorbar=dict(title=dict(text="r", font=dict(color=th["ink2"], size=11)),
                      tickfont=dict(color=th["muted"], size=10), thickness=12,
                      outlinewidth=0, len=0.85),
        hovertemplate="%{y}<br>%{x}<br>r = %{z:+.3f}<extra></extra>"))
    fig.update_xaxes(showgrid=False, tickangle=-55, tickfont=dict(size=9),
                     constrain="domain")
    fig.update_yaxes(showgrid=False, tickfont=dict(size=9), autorange="reversed",
                     scaleanchor="x", constrain="domain")
    fig.update_layout(title=title)
    return apply_theme(fig, th, height, legend=False)


def _density_heatmap(cells: pd.DataFrame, th: dict, colorbar: bool, hover: str,
                     relative: bool = False) -> go.Heatmap:
    """A binned 2-D density from long cell rows, empty cells left transparent.

    The artifacts drop empty cells rather than shipping a dense grid, so the matrix is
    rebuilt here and the holes stay `NaN` — a zero would be painted as the bottom of the
    ramp, which reads as "measured and low" rather than as "no rows landed here".

    `relative` divides each panel by its own densest cell, which is what makes **one**
    colourbar legitimate over several panels that do not share a scale: a validation panel
    over 751 rows puts a far larger share in each cell than a training panel over 8,232, so
    a shared absolute scale washes the larger split out and a per-panel absolute scale means
    the same shade twice with two meanings. The raw share stays in the hover either way.
    """
    x_values = np.sort(cells["x_center"].unique())
    y_values = np.sort(cells["y_center"].unique())
    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}
    grid = np.full((len(y_values), len(x_values)), np.nan)
    for x, y, value in zip(cells["x_center"], cells["y_center"], cells["density"]):
        grid[y_index[y], x_index[x]] = value
    share = grid.copy()
    if relative:
        peak = np.nanmax(grid)
        grid = grid / peak if peak > 0 else grid
    title, fmt = ("share of the\ndensest cell", ".0%") if relative else ("share", ".1%")
    return go.Heatmap(
        z=grid, x=x_values, y=y_values, colorscale=th["sequential"],
        hoverongaps=False, showscale=colorbar, customdata=share,
        colorbar=dict(title=dict(text=title, font=dict(color=th["ink2"], size=11)),
                      tickformat=fmt, tickfont=dict(color=th["muted"], size=10),
                      thickness=12, outlinewidth=0, len=0.85),
        hovertemplate=hover)


def fig_joint(cells: pd.DataFrame, th: dict, x_label: str, y_label: str,
              title: str = "", height: int = 460) -> go.Figure:
    """One feature pair's joint density — the on-demand half of block 3.

    `cells` is `model_cards.density_panel()`. This is the substitute for a pair plot and
    the reason there is no pair plot: a full matrix over 12–20 features is 150–400 panels,
    where the question being asked ("what does the joint look like where it matters") is
    answered by the heatmap for *which* pair and by one of these for *that* pair.
    """
    fig = go.Figure(_density_heatmap(
        cells, th, colorbar=True,
        hover=f"{x_label} %{{x:.3g}}<br>{y_label} %{{y:.3g}}"
              "<br>%{z:.2%} of rows<extra></extra>"))
    # One panel at a time here, so the colourbar can stay in the artifact's own units.
    fig.update_xaxes(title=x_label)
    fig.update_yaxes(title=y_label)
    fig.update_layout(title=title)
    return apply_theme(fig, th, height, legend=False)


def fig_coefficients(panel: pd.DataFrame, th: dict, title: str = "",
                     row_height: int = 26) -> go.Figure:
    """Posterior means with 95% credible intervals, sorted, families kept together.

    `panel` is `model_cards.coefficient_panel()`. Bars from zero because a coefficient's
    **sign** is the first thing to read, coloured off the two ends of the diverging scale
    for the same reason a PCA loading is — polarity is a direction on one axis, not two
    categories. The interval is drawn as a rule through each bar in the ink colour, so the
    uncertainty is legible against either end of the scale.

    The intercept and the dispersion are deliberately absent: they are not on the
    standardized slope scale the other terms share, and letting the intercept into the
    panel would set the axis and flatten every term the panel exists to compare.
    """
    negative, positive = th["diverging"][0][1], th["diverging"][-1][1]
    rows = list(range(len(panel)))
    labels = list(panel["label"] if "label" in panel else panel["term"])
    means = panel["mean"].to_numpy(dtype=float)
    lo = panel["q2.5"].to_numpy(dtype=float)
    hi = panel["q97.5"].to_numpy(dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=means, y=rows, orientation="h", showlegend=False,
        marker=dict(color=[positive if m >= 0 else negative for m in means],
                    line=dict(color=th["surface"], width=1)),
        customdata=list(zip(panel["term_family"], panel["p_positive"])),
        hovertemplate="mean %{x:+.4f}<br>P(> 0) %{customdata[1]:.2f}"
                      "<extra>%{customdata[0]}</extra>"))
    for row, (l, h) in enumerate(zip(lo, hi)):
        fig.add_shape(type="line", x0=l, x1=h, y0=row, y1=row,
                      line=dict(color=th["ink"], width=1.5), layer="above")
    # The interval ends as their own trace, so a reader can hover the number rather than
    # only see the rule — colour and length are not the only route to the value.
    fig.add_trace(go.Scatter(
        x=list(lo) + list(hi), y=rows + rows, mode="markers", showlegend=False,
        marker=dict(size=5, color=th["ink"], symbol="line-ns-open"),
        hovertemplate="95% interval bound %{x:+.4f}<extra></extra>"))

    fig.update_xaxes(title="posterior mean, standardized design scale", zeroline=True,
                     zerolinewidth=1)
    fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=labels,
                     showgrid=False, zeroline=False,
                     range=[len(panel) - 0.5, -0.5])
    fig.update_layout(title=title, bargap=0.3)
    return apply_theme(fig, th, height=max(240, len(panel) * row_height + 110),
                       legend=False)


#: The three nested bands the emitter ships as column pairs, widest first so it is painted
#: first and the narrowest overlap ends up darkest, under the median line.
_ECDF_BANDS = ((("q2.5", "q97.5"), "95%"), (("q10", "q90"), "80%"),
               (("q25", "q75"), "50%"))


def fig_ecdf(panels: dict, th: dict, value_label: str, title: str = "",
             height: int = 420) -> go.Figure:
    """The observed ECDF over the posterior-predictive ribbon, one subplot per split.

    `panels` maps a split label to `model_cards.ecdf_panel()`. Three nested bands — 50, 80
    and 95% of the per-draw replicate ECDFs — under one observed curve, on a y axis both
    subplots share so the two splits are read against each other.

    **The reading is the vertical distance, not in-or-out.** At n ≈ 10⁴ the band is one to
    two ECDF points wide and every head in the project leaves it somewhere; the view prints
    the largest gap from the median replicate beside this figure, which is the statistic
    that separates a small miss from a large one.
    """
    names = list(panels)
    fig = make_subplots(rows=1, cols=len(names), subplot_titles=names,
                        shared_yaxes=True, horizontal_spacing=0.06)
    ribbon = th["series"][0]

    for col, name in enumerate(names, start=1):
        part = panels[name]
        first = col == 1
        for alpha, ((low, high), label) in zip((0.14, 0.20, 0.26), _ECDF_BANDS):
            fig.add_trace(go.Scatter(
                x=list(part["value"]) + list(part["value"])[::-1],
                y=list(part[high]) + list(part[low])[::-1],
                fill="toself", fillcolor=_translucent(ribbon, alpha),
                line=dict(width=0), hoverinfo="skip", name=f"{label} band",
                legendgroup=label, showlegend=first), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=part["value"], y=part["q50"], mode="lines", name="median replicate",
            legendgroup="median", showlegend=first,
            line=dict(color=ribbon, width=1.5),
            hovertemplate="%{x:.3g}<br>median replicate %{y:.3f}<extra></extra>"),
            row=1, col=col)
        fig.add_trace(go.Scatter(
            x=part["value"], y=part["observed"], mode="lines", name="observed",
            legendgroup="observed", showlegend=first,
            line=dict(color=th["ink"], width=2),
            hovertemplate="%{x:.3g}<br>observed %{y:.3f}<extra></extra>"),
            row=1, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_xaxes(title=value_label)
    fig.update_yaxes(title="F(x)", range=[0, 1.02], tickformat=".0%", col=1)
    fig.update_layout(title=title)
    return apply_theme(fig, th, height)


#: Room left around a calibration panel's own grid, as a share of its span.
PANEL_PAD = 0.02


def _panel_range(cells: dict, panel: str) -> tuple[tuple[float, float],
                                                   tuple[float, float]]:
    """The x and y extent of one panel type's grid, pooled over both splits."""
    frames = [frame for (name, _), frame in cells.items()
              if name == panel and frame is not None and not frame.empty]
    if not frames:
        return (0.0, 1.0), (0.0, 1.0)
    grid = pd.concat(frames)
    out = []
    for axis in ("x", "y"):
        lo = float(grid[f"{axis}_left"].min())
        hi = float(grid[f"{axis}_right"].max())
        pad = (hi - lo) * PANEL_PAD or 1.0
        out.append((lo - pad, hi + pad))
    return out[0], out[1]


def fig_calibration(cells: dict, points: dict, th: dict, axis_labels: dict,
                    panel_labels: dict, title: str = "",
                    height: int = 760) -> go.Figure:
    """Four panels: predicted-vs-observed and residual-vs-predicted, on both splits.

    `cells` and `points` are keyed by `(panel, split)` — the binned density from
    `model_cards.calibration_panel()` and the bounded subsample from
    `model_cards.sample_points()`. The density carries the mass, which a 631,158-point
    scatter cannot; the sample carries the texture, which a 30 × 30 grid cannot.

    Each panel gets its reference where one exists: the identity line on the
    predicted-against-observed panels, zero on the residual ones. Without it a reader has
    to infer the target line from the data, which is precisely the bias the panel is for.
    """
    keys = [(panel, split) for panel in panel_labels for split in ("train", "validation")]
    titles = [f"{panel_labels[panel]} · {split}" for panel, split in keys]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles,
                        vertical_spacing=0.11, horizontal_spacing=0.08)
    # One axis range per panel *type*, spanning the grid both splits share. Two reasons,
    # and the first was caught by rendering the figure rather than by reading the code: the
    # emitter clips the density's tails into its end bins, and letting the unclipped overlay
    # set the axis undoes that — `gp_duration` has one 62-game spell that squashed a
    # 2-to-9-game density into a sliver. And a train panel and a validation panel on
    # different ranges cannot be compared, which is the whole reason they are side by side.
    ranges = {panel: _panel_range(cells, panel) for panel in panel_labels}

    for index, key in enumerate(keys):
        row, col = divmod(index, 2)
        row += 1
        col += 1
        panel, _ = key
        grid = cells.get(key)
        if grid is None or grid.empty:
            continue
        x_label, y_label = axis_labels[panel]
        # Each panel against its own densest cell, with one colourbar for all four. The
        # four are not on one scale — a 751-row validation panel puts an order of magnitude
        # more share in each cell than an 8,232-row training one — so an absolute colourbar
        # drawn from the first panel would label the other three wrongly. Caught by
        # rendering the figure; the raw share is still in every cell's hover.
        fig.add_trace(_density_heatmap(
            grid, th, colorbar=index == 0, relative=True,
            hover=f"{x_label} %{{x:.3g}}<br>{y_label} %{{y:.3g}}"
                  "<br>%{customdata:.2%} of this panel's rows<extra></extra>"),
            row=row, col=col)

        sample = points.get(key)
        if sample is not None and len(sample):
            y = (sample["observed"] if panel == "fitted_observed"
                 else sample["residual"])
            fig.add_trace(go.Scatter(
                x=sample["fitted"], y=y, mode="markers", showlegend=False,
                marker=dict(size=OVERLAY_SIZE, color=_translucent(th["ink"],
                                                                 OVERLAY_ALPHA)),
                hovertemplate=f"{x_label} %{{x:.3g}}<br>{y_label} %{{y:.3g}}"
                              "<extra>one row</extra>"), row=row, col=col)

        x_range, y_range = ranges[panel]
        if panel == "fitted_observed":
            span = [max(x_range[0], y_range[0]), min(x_range[1], y_range[1])]
            fig.add_trace(go.Scatter(
                x=span, y=span, mode="lines", showlegend=False, hoverinfo="skip",
                line=dict(color=th["ink2"], width=1)), row=row, col=col)
        else:
            fig.add_hline(y=0.0, line=dict(color=th["ink2"], width=1), row=row, col=col)
        fig.update_xaxes(title=x_label, range=list(x_range), row=row, col=col)
        fig.update_yaxes(title=y_label, range=list(y_range), row=row, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_layout(title=title)
    return apply_theme(fig, th, height, legend=False)
