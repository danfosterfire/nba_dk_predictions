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
    """A vertical reference, labelled unless the axis already says what it is.

    Solid, because `theme.py` bans dashes outright.

    `faceted` spans every subplot of a `make_subplots` figure; the label is added once, at
    the top of the paper, rather than once per facet. `at_floor` drops the label inside
    the plot instead, which is what a line near the middle of the x range needs — the
    legend sits on the top edge and a label there collides with whichever entry happens to
    be under it. Caught in a rendered PNG, on the tier whose null lands mid-axis.

    **An empty `label` draws the line and no annotation**, which is the other half of that
    same problem: neither placement is safe in general, because whether the label lands on
    the legend or on a data interval depends on *where zero falls on the x axis*, and
    nothing in the trace can see that. A caller whose axis title already names the reference
    — `fig_paired`'s is literally "gap … against `<baseline>`" — has nothing to lose by
    dropping the second copy, and gains a placement that cannot collide.

    Drawn **below** the data, which is right for every caller here because their marks are
    dots and intervals that a line behind stays readable through. A caller whose marks are
    solid bars *crossing* the reference needs it above them instead, and draws its own —
    see `fig_block_inflation`.
    """
    placement = dict(row="all", col=1) if faceted else {}
    fig.add_vline(x=x, line=dict(color=color or th["axis"], width=1), layer="below",
                  **placement)
    if not label:
        return
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
    # Bare, because the axis title already reads "gap … against <baseline>" and so zero can
    # only be the baseline. It used to carry a label at the top of the paper, which collided
    # with the legend on any frame where zero happened to land under it — visible on the
    # minutes page's sigma sweep, latent here, and unreachable from the trace either way.
    _reference_line(fig, 0.0, th, "")
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
    # Never more columns than features. The composition's 25 columns are what this grid was
    # laid out for; the game-length onset head has **one**, and a four-wide grid drew it in
    # the leftmost quarter with three empty cells beside it.
    columns = max(1, min(columns, len(features)))
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
    fig = apply_theme(fig, th, height=rows * HIST_ROW_HEIGHT + 90)
    if rows == 1:
        # A single-row grid puts its subplot title where the shared legend already sits.
        # Taller grids have the legend clear of the first row's title by construction.
        fig.update_layout(margin=dict(l=8, r=8, t=76, b=8))
    return fig


def fig_correlation(square: pd.DataFrame, th: dict, title: str = "",
                    height: int = 520, limit: float = 1.0) -> go.Figure:
    """A correlation matrix, diverging around zero.

    `square` is `model_cards.correlation_square()` on a model page and
    `inputs.copula_square()` on page 7. **Diverging rather than sequential**, because the
    sign of a correlation is a direction and not a magnitude. A constant column arrives as a
    row of `NaN` and is drawn as a gap on the surface — which is the point of shipping the
    whole square: the empty row stays on the axis and says so.

    `limit` is the scale's half-range and **defaults to the full [−1, +1]**, because a model
    page's feature correlations run the whole range and pinning them means two heads'
    heatmaps mean the same thing. A caller whose matrix does not is obliged to say so: the
    residual copula's largest off-diagonal cell is +0.133, so on the pinned scale every cell
    that is not the diagonal renders as the neutral midpoint and the figure reports "no
    dependence" about a matrix that exists precisely to carry some. Narrowing the scale is
    then not cosmetic, and neither is masking the diagonal that forces it.
    """
    fig = go.Figure(go.Heatmap(
        z=square.to_numpy(dtype=float), x=list(square.columns), y=list(square.index),
        zmin=-limit, zmax=limit, zmid=0.0, colorscale=th["diverging"],
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

#: Where a horizontal legend sits on a figure that *also* carries subplot titles, and the
#: top margin that makes room for it. `apply_theme`'s default 1.02 puts the legend in the
#: same strip `make_subplots` writes its titles into: both are paper-referenced just above
#: the plot area, so a legend of four or five entries runs straight through the first
#: subplot's title. Invisible in the trace and obvious in a rendered PNG, which is the
#: layer that found it — on this project's longest legend, the ECDF ribbon's five.
LEGEND_ABOVE_TITLES = 1.11
LEGEND_TITLE_MARGIN = 78


def _legend_above_titles(fig: go.Figure) -> go.Figure:
    """Lift the legend clear of the subplot-title strip. Call **after** `apply_theme`."""
    fig.update_layout(legend=dict(y=LEGEND_ABOVE_TITLES),
                      margin=dict(t=LEGEND_TITLE_MARGIN))
    return fig


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
            # `mode` is explicit because plotly infers `lines+markers` for a trace with 20
            # points or fewer, and infers the marker colour from its *own* default
            # colorway — so a head whose grid is short draws stray cyan and red dots that
            # are in no palette this project validated. The onset head's validation grid is
            # two points, which is the first place in the repo that threshold is crossed.
            fig.add_trace(go.Scatter(
                x=list(part["value"]) + list(part["value"])[::-1],
                y=list(part[high]) + list(part[low])[::-1], mode="lines",
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
    return _legend_above_titles(apply_theme(fig, th, height))


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
                    height: int = 420) -> go.Figure:
    """Predicted against observed, train beside validation — one row per panel type.

    `cells` and `points` are keyed by `(panel, split)` — the binned density from
    `model_cards.calibration_panel()` and the bounded subsample from
    `model_cards.sample_points()`. The density carries the mass, which a 631,158-point
    scatter cannot; the sample carries the texture, which a 30 × 30 grid cannot.

    Each panel gets the identity line as its reference. Without it a reader has to infer the
    target line from the data, which is precisely the bias the panel is for.

    **The grid is `len(panel_labels)` rows by two splits rather than a fixed 2 × 2.** It drew
    four panels until 2026-08-10, when the residual half moved to `fig_quantile_residual`;
    a hard-coded two rows would have left two empty cells and a figure twice as tall as its
    content, which `AppTest` cannot see and a rendered PNG can.
    """
    keys = [(panel, split) for panel in panel_labels for split in ("train", "validation")]
    titles = [f"{panel_labels[panel]} · {split}" for panel, split in keys]
    fig = make_subplots(rows=max(len(panel_labels), 1), cols=2, subplot_titles=titles,
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
            fig.add_trace(go.Scatter(
                x=sample["fitted"], y=sample["observed"], mode="markers",
                showlegend=False,
                marker=dict(size=OVERLAY_SIZE, color=_translucent(th["ink"],
                                                                 OVERLAY_ALPHA)),
                hovertemplate=f"{x_label} %{{x:.3g}}<br>{y_label} %{{y:.3g}}"
                              "<extra>one row</extra>"), row=row, col=col)

        x_range, y_range = ranges[panel]
        span = [max(x_range[0], y_range[0]), min(x_range[1], y_range[1])]
        fig.add_trace(go.Scatter(
            x=span, y=span, mode="lines", showlegend=False, hoverinfo="skip",
            line=dict(color=th["ink2"], width=1)), row=row, col=col)
        fig.update_xaxes(title=x_label, range=list(x_range), row=row, col=col)
        fig.update_yaxes(title=y_label, range=list(y_range), row=row, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_layout(title=title)
    return apply_theme(fig, th, height, legend=False)


#: Marker size on the QQ panel. Smaller than a scatter's, because 100 order statistics on a
#: 400 px square is one point every 4 px and anything larger reads as a band.
QQ_MARKER = 4


def fig_qq(panels: dict, th: dict, title: str = "", height: int = 400) -> go.Figure:
    """The scaled quantile residual against the uniform it should be, one panel per split.

    `panels` maps a split label to `model_cards.qq_panel()`. **A correct model puts the
    points on the diagonal whatever its likelihood is**, which is the property that lets one
    renderer draw a negative binomial count head and a beta-geometric spell head on the same
    axes — and the reason this panel replaced the raw residual one.

    The envelope is the emitter's pointwise Beta band on each order statistic, drawn as a
    ribbon rather than as two lines so it reads as background. Pointwise: about 5 of 100
    points sit outside a 95% pointwise envelope under a *correct* model, so it is a sense of
    scale beside the curve rather than a test, and the KS distance is tiled beside the figure
    as the actual reading.
    """
    names = list(panels)
    fig = make_subplots(rows=1, cols=max(len(names), 1), subplot_titles=names,
                        shared_yaxes=True, horizontal_spacing=0.06)
    ribbon = th["series"][0]

    for col, name in enumerate(names, start=1):
        part = panels[name]
        first = col == 1
        # `mode` is explicit on every trace here: plotly infers `lines+markers` for a trace
        # of 20 points or fewer and takes the marker colour from its own default colorway,
        # and the overtime-onset head's validation QQ is two points.
        fig.add_trace(go.Scatter(
            x=list(part["expected"]) + list(part["expected"])[::-1],
            y=list(part["hi"]) + list(part["lo"])[::-1], mode="lines", fill="toself",
            fillcolor=_translucent(ribbon, 0.20), line=dict(width=0), hoverinfo="skip",
            name="95% pointwise envelope", legendgroup="band", showlegend=first),
            row=1, col=col)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="uniform", legendgroup="uniform",
            showlegend=first, hoverinfo="skip",
            line=dict(color=th["ink2"], width=1, dash="dash")), row=1, col=col)
        fig.add_trace(go.Scatter(
            x=part["expected"], y=part["observed"], mode="markers",
            name="observed residual", legendgroup="observed", showlegend=first,
            marker=dict(size=QQ_MARKER, color=th["ink"]),
            hovertemplate="expected %{x:.3f}<br>observed %{y:.3f}<extra></extra>"),
            row=1, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_xaxes(title="expected uniform quantile", range=[-0.02, 1.02])
    fig.update_yaxes(title="scaled residual", range=[-0.02, 1.02], col=1)
    fig.update_layout(title=title)
    return _legend_above_titles(apply_theme(fig, th, height))


def _excess_heatmap(cells: pd.DataFrame, th: dict, colorbar: bool) -> go.Heatmap:
    """A binned density coloured by its **departure from an even spread**, not by its mass.

    The sequential ramp every other density here uses is the wrong encoding for this one,
    and rendering it is what showed why: a calibrated scaled residual against a rank
    transform fills the unit square *evenly by construction*, so a share-of-the-densest-cell
    ramp paints a wall of near-equal blue in which nothing is legible and Poisson noise
    between cells reads as structure. What the panel is for is the departure — a genuinely
    signed quantity, which takes the diverging scale with its neutral midpoint at "exactly
    its share", the same rule a correlation heatmap follows.

    `z` is `density x cells - 1`: 0 where a cell holds exactly its share, +1 where it holds
    double, clamped at ±1 so one dense corner cannot flatten the rest. Empty cells stay NaN
    and paint nothing, because a zero here would read as "an even spread" rather than as
    "no rows landed".
    """
    x_values = np.sort(cells["x_center"].unique())
    y_values = np.sort(cells["y_center"].unique())
    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}
    share = np.full((len(y_values), len(x_values)), np.nan)
    for x, y, value in zip(cells["x_center"], cells["y_center"], cells["density"]):
        share[y_index[y], x_index[x]] = value
    n_cells = len(x_values) * len(y_values)
    return go.Heatmap(
        z=share * n_cells - 1.0, x=x_values, y=y_values, customdata=share,
        colorscale=th["diverging"], zmin=-1.0, zmax=1.0, zmid=0.0,
        hoverongaps=False, showscale=colorbar,
        colorbar=dict(title=dict(text="against an\neven spread",
                                 font=dict(color=th["ink2"], size=11)),
                      tickformat="+.0%", tickfont=dict(color=th["muted"], size=10),
                      thickness=12, outlinewidth=0, len=0.85),
        hovertemplate="rank of predicted %{x:.2f}<br>scaled residual %{y:.2f}"
                      "<br>%{customdata:.2%} of this panel's rows, %{z:+.0%} against an "
                      "even spread<extra></extra>")


def fig_quantile_residual(cells: dict, lines: dict, points: dict, th: dict,
                          levels: tuple[float, ...], title: str = "",
                          height: int = 420) -> go.Figure:
    """The scaled residual against rank-transformed predicted, one panel per split.

    Three objects on one panel and each answers a different question: the binned density
    says where the rows sit *against an even spread*, the bounded subsample carries the
    texture, and the three quantile lines carry the number — **flat at 0.25 / 0.5 / 0.75 iff
    calibrated**, drawn against dashed references at those levels so "flat" is read off the
    axis rather than judged by eye.

    The x axis is a rank transform, which is what makes this panel comparable between a count
    head on a season total and a conversion head on a rate: the predicted values themselves
    share no axis and their ranks do. Both axes are [0, 1] by construction, so the two splits
    are on one grid without needing to be put there.

    **The empirical lines are drawn in `ink` rather than in a series colour**, which is a
    legibility fix rather than a preference: over a diverging field a mid-scale line is
    invisible at one end of it, and these three lines are the panel's quantitative content.
    All three share one colour and all three references another — `theme.ALL_PAIRS_CAP` is 3
    — and each line's level is legible from the reference directly under it and from the
    hover, so nothing here is reachable by colour alone.
    """
    names = list(cells)
    fig = make_subplots(rows=1, cols=max(len(names), 1), subplot_titles=names,
                        shared_yaxes=True, horizontal_spacing=0.06)

    for col, name in enumerate(names, start=1):
        first = col == 1
        grid = cells[name]
        if grid is not None and len(grid):
            fig.add_trace(_excess_heatmap(grid, th, colorbar=first), row=1, col=col)
        sample = points.get(name)
        if sample is not None and len(sample):
            fig.add_trace(go.Scatter(
                x=sample["predicted_rank"], y=sample["u"], mode="markers",
                showlegend=False,
                marker=dict(size=OVERLAY_SIZE,
                            color=_translucent(th["ink"], OVERLAY_ALPHA)),
                hovertemplate="rank of predicted %{x:.3f}<br>scaled residual %{y:.3f}"
                              "<extra>one row</extra>"), row=1, col=col)

        for level in levels:
            fig.add_trace(go.Scatter(
                x=[0, 1], y=[level, level], mode="lines", hoverinfo="skip",
                name="expected level", legendgroup="expected",
                showlegend=first and level == levels[0],
                line=dict(color=th["ink2"], width=1, dash="dash")), row=1, col=col)
        line = lines.get(name)
        if line is not None and len(line):
            for index, level in enumerate(levels):
                part = line[line["level"] == level]
                if part.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=part["x"], y=part["y"], mode="lines+markers",
                    name="empirical quantile", legendgroup="empirical",
                    showlegend=first and index == 0,
                    line=dict(color=th["ink"], width=2.5),
                    marker=dict(size=5, color=th["ink"]),
                    customdata=[level] * len(part),
                    hovertemplate="rank of predicted %{x:.2f}<br>%{customdata:.0%} "
                                  "quantile %{y:.3f}<extra></extra>"), row=1, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_xaxes(title="predicted, rank-transformed", range=[0, 1])
    fig.update_yaxes(title="scaled residual", range=[0, 1], col=1)
    fig.update_layout(title=title)
    return _legend_above_titles(apply_theme(fig, th, height))


# ── The two figures a model page owns for itself ──────────────────────────────
#
# Everything above is a *class* figure — one builder over any head. These two are not: they
# draw the ladder a head was selected from, which lives beside the model cards in each
# class's own `make stan` metrics table, and the two classes that carry a page-specific block
# so far shape that ladder differently. Both obey the same two rules as everything else:
# every bar prints its own value, and the view ships a table twin.

#: A bar's own value, printed outside it. The relief rule in its most literal form — three
#: light-mode slots fall under 3:1 on the light surface, so no bar's height may be the only
#: route to its number.
BAR_TEXT_SIZE = 11
#: Room past the longest bar for the label that sits outside it. Without it the widest bar's
#: own value renders hard against the plot edge — caught by rendering the figure, since the
#: text is laid out by plotly.js and no assertion about the trace can see where it lands.
BAR_TEXT_ROOM = 0.14


def _bar_text_range(values, floor_at_zero: bool = True,
                    room: float = BAR_TEXT_ROOM) -> list[float]:
    """An axis range with room at each end for a value printed outside its bar.

    `room` is a parameter rather than the constant because the label's *length* decides how
    much is enough and only the caller knows it: `+0.0334` and `200.28 minutes` do not need
    the same margin, and neither is measurable from the trace.
    """
    lo, hi = float(min(values)), float(max(values))
    span = (hi - lo) or (abs(hi) or 1.0)
    return [min(0.0, lo) - (0.0 if floor_at_zero and lo >= 0 else span * room),
            hi + span * room]


def fig_floor_margin(board: pd.DataFrame, th: dict, highlight: str = "",
                     title: str = "", row_height: int = 30) -> go.Figure:
    """Every head on a page against its own no-fit floor, as the margin between them.

    `board` is `model_cards.floor_board()`. **The zero line is the floor**, so a head that
    did not clear it is on the other side of it — position, not colour, carries the finding,
    which is the only encoding that survives the palette's relief rule. Two more routes to
    the same fact ride along: a non-clearing bar is outlined in the ink colour, and every bar
    prints its own margin.

    Colour is highlight-and-gray rather than a scale: eleven bars is well past
    `ALL_PAIRS_CAP`, and the one distinction worth making is which head the reader currently
    has open, so `highlight` takes slot 0 and the rest take `muted`.
    """
    panel = board.sort_values("margin", ascending=False).reset_index(drop=True)
    rows = list(range(len(panel)))
    clears = panel["clears"].to_numpy(dtype=bool)
    fig = go.Figure(go.Bar(
        x=panel["margin"], y=rows, orientation="h", showlegend=False,
        marker=dict(
            color=[th["series"][0] if head == highlight else th["muted"]
                   for head in panel["head"]],
            line=dict(color=[th["surface"] if ok else th["ink"] for ok in clears],
                      width=[1 if ok else 2 for ok in clears])),
        text=[f"{m:+.4f}" for m in panel["margin"]], textposition="outside",
        textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
        customdata=list(zip(panel["floor_r2"], panel["shipped_r2"], panel["variant"])),
        hovertemplate="floor R² %{customdata[0]:.4f}<br>shipped R² %{customdata[1]:.4f}"
                      "<br>margin %{x:+.4f}<extra>%{customdata[2]}</extra>"))
    fig.update_xaxes(title="validation R² above the head's own no-fit floor",
                     range=_bar_text_range(panel["margin"], floor_at_zero=False))
    fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=list(panel["label"]),
                     showgrid=False, zeroline=False, range=[len(panel) - 0.5, -0.5])
    fig.update_layout(title=title, bargap=0.35)
    fig = apply_theme(fig, th, height=max(240, len(panel) * row_height + 120),
                      legend=False)
    # After `apply_theme`, which sets the hairline zeroline every other chart wants. Here
    # the zero line *is* the floor and a bar's side of it is the finding, so it is drawn at
    # the weight of a reference line rather than of a gridline.
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor=th["ink2"])
    return fig


def fig_class_counts(panel: pd.DataFrame, th: dict, value_label: str = "games",
                     title: str = "", row_height: int = 30) -> go.Figure:
    """Observed counts against a fitted arm's and its floor's, per outcome class.

    `panel` is `model_cards.class_counts()`, already filtered to the classes worth drawing.
    **The observed is not a third model, it is the target**, so it is drawn as an outlined
    bar in the ink colour — a different *mark* as well as a different colour, the same
    relief the feature grid gets from bars against a step line — and the two arms fill
    toward it in slots 0 and 1. Ink for the observed is the encoding block 5's ribbon
    already uses, so a reader who scrolled past it has learned this once. A solid ink bar
    was the first cut and read as the largest quantity on the chart rather than as the
    reference the other two are measured against.
    """
    classes = list(dict.fromkeys(panel["class_label"]))
    series = list(dict.fromkeys(panel["series"]))
    position = {name: i for i, name in enumerate(classes)}
    colors = {name: th["series"][i] for i, name in enumerate(
        [s for s in series if s != "observed"])}

    fig = go.Figure()
    for name in series:
        part = panel[panel["series"] == name]
        observed = name == "observed"
        fig.add_trace(go.Bar(
            x=part["count"], y=[position[c] for c in part["class_label"]],
            orientation="h", name=name,
            marker=dict(color="rgba(0,0,0,0)" if observed else colors[name],
                        line=dict(color=th["ink"] if observed else colors[name],
                                  width=2 if observed else 0)),
            text=[f"{v:,.1f}" for v in part["count"]], textposition="outside",
            textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
            hovertemplate="%{x:,.2f} " + value_label + "<extra>" + name + "</extra>"))

    fig.update_xaxes(title=value_label, range=_bar_text_range(panel["count"]))
    fig.update_yaxes(tickmode="array", tickvals=list(position.values()), ticktext=classes,
                     showgrid=False, zeroline=False, range=[len(classes) - 0.5, -0.5])
    fig.update_layout(title=title, barmode="group", bargap=0.3, bargroupgap=0.08)
    return apply_theme(fig, th,
                       height=max(260, len(classes) * len(series) * row_height + 130))


# ── The one figure the weekly-scores page owns ────────────────────────────────
#
# Everything else that page draws is reused *unmodified* from the model pages above —
# `fig_ecdf`, `fig_calibration`, `fig_qq` and `fig_quantile_residual`, because
# `make weekly-scores` cuts its artifacts with `src/models/model_cards.py`'s own binning
# helpers and hands back the same frame shapes. This is the one question those four cannot
# answer: they pool seventeen weeks into one distribution, and "does the simulator drift
# *through* the season" is a reading along the period axis rather than across it.

#: Where the seventeen one-week periods end and the three double weeks begin. Drawn as a
#: divider rather than left to the tick labels, because the unit changes there: the last
#: three points are two weeks of games each and are roughly twice the height for that
#: reason alone.
UNIT_BREAK_WIDTH = 1


def fig_period_profile(panels: dict, th: dict, value_label: str = "mean dk_pts per player",
                       title: str = "", height: int = 380) -> go.Figure:
    """Observed against simulated mean `dk_pts`, period by period, one subplot per split.

    `panels` maps a split label to `weekly.profile_panel()`. Two series per subplot, which
    is inside `ALL_PAIRS_CAP`, and both are read off a labelled axis with a table twin in
    the view — the relief rule, which matters here because the whole content is the *gap*
    between two lines.

    **The x axis is not a number line.** Seventeen one-week periods are followed by three
    double weeks, so the ticks read `W1`…`W17` then `R2`/`R3`/`R4` and a divider is drawn
    where the unit changes. A bare 1…20 would invite reading the step up at the end as a
    model artifact when it is the calendar.
    """
    names = list(panels)
    fig = make_subplots(rows=1, cols=max(len(names), 1), subplot_titles=names,
                        shared_yaxes=True, horizontal_spacing=0.06)

    for col, name in enumerate(names, start=1):
        part = panels[name]
        first = col == 1
        if part is None or not len(part):
            continue
        breaks = part.index[part["period_type"] != part["period_type"].iloc[0]]
        for index, (column, label, color) in enumerate(
                (("observed", "observed", th["ink"]),
                 ("predicted", "simulated", th["series"][0]))):
            # `mode` is explicit for the reason every `go.Scatter` here sets it: plotly
            # infers `lines+markers` at 20 points or fewer and takes the marker colour from
            # its own default colorway, and this trace is exactly 20 points.
            fig.add_trace(go.Scatter(
                x=part["label"], y=part[column], mode="lines+markers", name=label,
                legendgroup=label, showlegend=first,
                line=dict(color=color, width=2, dash="solid"),
                marker=dict(size=6, color=color),
                customdata=np.stack([part["games"], part["n"]], axis=-1),
                hovertemplate=f"%{{x}}<br>{label} %{{y:.1f}} dk_pts"
                              "<br>%{customdata[0]:.2f} games played, "
                              "%{customdata[1]:,} rows<extra></extra>"),
                row=1, col=col)
        if len(breaks):
            # Between the last one-week tick and the first double week, so it separates the
            # two units rather than sitting on a data point.
            fig.add_vline(x=float(breaks[0]) - 0.5, line=dict(
                color=th["axis"], width=UNIT_BREAK_WIDTH, dash="dot"), row=1, col=col)

    for annotation in fig.layout.annotations:
        annotation.font = dict(family=FONT, size=12, color=th["ink2"])
    fig.update_xaxes(title="scoring period", type="category")
    fig.update_yaxes(title=value_label, rangemode="tozero", col=1)
    fig.update_layout(title=title)
    return _legend_above_titles(apply_theme(fig, th, height))


# ── The four figures the minutes page owns ────────────────────────────────────
#
# The other three model pages compare a head against a floor. This one compares **two heads
# at two units**, which changes what the colour has to do: on pages 5 and 6 a slot marks the
# head the reader has open among many, and here every figure has exactly two series that are
# both the point. So the head→slot map is fixed for the whole page (`MINUTES_SLOTS`) and
# nothing on it is highlight-and-gray — two series is well inside `ALL_PAIRS_CAP`, the
# reader learns the pairing once, and the tiles rather than the palette say which head is
# open.
#
# The zero line means something different in each: the unit's own no-fit floor in the first,
# no bias in the second, the marginal head in the third (drawn by `fig_paired`, reused), and
# independence in the fourth. Each one is labelled, because "zero" is not self-describing
# when it is a different reference four times on one page.

#: Room past the longest bar on the two figures whose printed value is a formatted number
#: rather than a bare coefficient. Wider than `BAR_TEXT_ROOM` because "200.28" and "+10.5%"
#: are two to three times the width of "+0.03", and plotly.js lays the text out itself.
WIDE_BAR_TEXT_ROOM = 0.30
#: The same, for a bar printing a signed percentage — six characters rather than a
#: formatted count, so a third of the extra room is enough and the rest is dead axis.
PERCENT_BAR_TEXT_ROOM = 0.18


def _head_colors(th: dict, heads, slots: dict) -> list[str]:
    """The fixed per-row slot a page shares across its figures.

    **A row not in `slots` recedes to `muted` rather than taking the next categorical
    slot.** That is what makes the map usable as highlight-and-gray as well as as a fixed
    pairing: the minutes page names every head it draws, so the fallback never fires there,
    and the inputs page names only the window the simulator consumes and lets the other two
    gray out. Handing an unnamed row a colour nobody chose is the failure mode either way.
    """
    return [th["series"][slots[head]] if head in slots else th["muted"]
            for head in heads]


def fig_unit_verdict(board: pd.DataFrame, th: dict, slots: dict, title: str = "",
                     row_height: int = 44) -> go.Figure:
    """One posterior at two units, each against the no-fit floor **of that unit**.

    `board` is `model_cards.unit_board()`. The two units cannot share a CRPS axis — 4.5
    minutes per player-game against 170 per season — so the axis is the *ratio* to each
    unit's own floor, which is dimensionless and therefore comparable. That is not a
    convenience: the floor is what every head in this project is quoted against, so "the
    zero line is the floor" already means something on this dashboard, and the finding is
    that the same head sits on opposite sides of it in the two panels.

    Position carries the verdict and colour carries only which head is which, so the
    reversal survives the palette's relief rule intact. Every bar prints its own value.
    """
    units = list(dict.fromkeys(board["unit_label"]))
    fig = make_subplots(rows=len(units), cols=1, shared_xaxes=True,
                        vertical_spacing=0.16,
                        subplot_titles=[f"scored per {unit}" for unit in units])
    for note in fig.layout.annotations:
        note.update(font=dict(color=th["ink"], size=13), x=0, xanchor="left")

    for r, unit in enumerate(units, start=1):
        part = board[board["unit_label"] == unit]
        rows = list(range(len(part)))
        fig.add_trace(go.Bar(
            x=part["improvement"], y=rows, orientation="h", showlegend=False,
            marker=dict(color=_head_colors(th, part["head"], slots),
                        line=dict(color=th["surface"], width=1)),
            text=[f"{v:+.1%}" for v in part["improvement"]], textposition="outside",
            textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
            customdata=list(zip(part["crps"], part["floor_crps"], part["arm"])),
            hovertemplate="CRPS %{customdata[0]:.4f} against a floor of "
                          "%{customdata[1]:.4f}<br>%{x:+.2%} against the floor"
                          "<extra>%{customdata[2]}</extra>"), row=r, col=1)
        fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=list(part["label"]),
                         showgrid=False, zeroline=False,
                         range=[len(part) - 0.5, -0.5], row=r, col=1)

    fig.update_xaxes(title="", showticklabels=False)
    fig.update_xaxes(title="CRPS against that unit's own no-fit floor", tickformat="+.0%",
                     showticklabels=True, row=len(units), col=1)
    fig = apply_theme(fig, th, height=max(300, len(board) * row_height + 150),
                      legend=False)
    fig.update_xaxes(range=_bar_text_range(board["improvement"], floor_at_zero=False,
                                           room=PERCENT_BAR_TEXT_ROOM))
    # After `apply_theme`, like `fig_floor_margin`: here the zero line *is* the floor and a
    # bar's side of it is the whole finding, so it is drawn at reference weight. It carries
    # no annotation of its own — the axis title already says what zero is, and a label at
    # the top of the paper would land on a facet header that starts at the same edge.
    fig.update_xaxes(zeroline=True, zerolinewidth=2, zerolinecolor=th["ink2"])
    fig.update_layout(title=title, bargap=0.4, margin=dict(l=8, r=8, t=60, b=8))
    return fig


def fig_metric_facets(panel: pd.DataFrame, th: dict, slots: dict, columns: int = 2,
                      title: str = "", row_height: int = 150) -> go.Figure:
    """The same few rows read several ways, one metric per facet on its own axis.

    `panel` is `model_cards.spread_panel()` on the minutes page and
    `inputs.window_facets()` on the inputs page — the same shape in both, which is why there
    is one builder: a small fixed set of rows, compared inside each facet, where the facets
    are in different units. Every facet has its own x range for that reason; on the minutes
    page they are minutes, minutes and a KS statistic, and on the inputs page a correlation,
    two variance ratios and a frailty variance. What is shared is the *rows*, so a reader
    reads **down** the facets to see the same rows change places.

    A facet whose frame carries a `reference` draws it as a labelled line: the predictive-sd
    panel is the only one with a target value, and without it a reader cannot tell whether
    64.65 is too narrow or 302.75 too wide.
    """
    metrics = list(dict.fromkeys(panel["metric_label"]))
    columns = max(1, min(columns, len(metrics)))
    rows = max(1, math.ceil(len(metrics) / columns))
    fig = make_subplots(rows=rows, cols=columns, subplot_titles=metrics,
                        vertical_spacing=min(0.18, 0.55 / max(rows, 1)),
                        horizontal_spacing=0.16)
    for note in fig.layout.annotations:
        note.update(font=dict(color=th["ink"], size=13))

    for index, metric in enumerate(metrics):
        row, col = divmod(index, columns)
        row, col = row + 1, col + 1
        part = panel[panel["metric_label"] == metric]
        positions = list(range(len(part)))
        fig.add_trace(go.Bar(
            x=part["value"], y=positions, orientation="h", showlegend=False,
            marker=dict(color=_head_colors(th, part["head"], slots),
                        line=dict(color=th["surface"], width=1)),
            text=list(part["text"]), textposition="outside",
            textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
            customdata=list(part["label"]),
            hovertemplate="%{customdata}<br>" + metric + " %{x:,.4f}<extra></extra>"),
            row=row, col=col)

        values = list(part["value"]) + [0.0]
        reference = part["reference"].dropna()
        if len(reference):
            target = float(reference.iloc[0])
            values.append(target)
            fig.add_vline(x=target, line=dict(color=th["ink2"], width=1), layer="below",
                          row=row, col=col)
            # Above the top bar rather than beside the line: the y axis is reversed, so the
            # free strip a facet has is the one between its own top edge and row 0.
            fig.add_annotation(x=target, y=-0.85, text=f"{target:,.1f} — residual sd",
                               showarrow=False, xanchor="right", yanchor="top",
                               xshift=-4, font=dict(color=th["ink2"], size=10),
                               bgcolor=th["surface"], borderpad=2, row=row, col=col)
        fig.update_xaxes(range=_bar_text_range(values, room=WIDE_BAR_TEXT_ROOM),
                         row=row, col=col)
        fig.update_yaxes(tickmode="array", tickvals=positions,
                         ticktext=list(part["label"]), showgrid=False, zeroline=False,
                         range=[len(part) - 0.5, -0.9], row=row, col=col)

    fig.update_layout(title=title, bargap=0.45)
    fig = apply_theme(fig, th, height=rows * row_height + 90, legend=False)
    fig.update_layout(margin=dict(l=8, r=8, t=60, b=8))
    return fig


def fig_sigma_grids(sweep: pd.DataFrame, th: dict, marginal_crps: float | None = None,
                    height: int = 380) -> go.Figure:
    """The injected effect's CRPS against sigma, on the two grids that were run.

    `sweep` is `model_cards.sigma_sweep()`. Two panels rather than two series on one axis,
    and that is the whole care this figure needs: the grids score **disjoint rows** — 742
    validation player-seasons against 1,145 training ones — so their CRPS *levels* are not
    comparable and only the location of each minimum is. Two series on a shared axis would
    invite exactly the comparison the panels forbid.

    Each panel marks its own optimum, and the shipped sigma is deliberately **not** marked
    here: an enlarged marker already means "the optimum on this grid", and a second mark on
    one figure meaning something else is worse than a caption. The gap chart above it names
    the shipped row instead.

    The validation panel also carries the marginal head's CRPS as a reference line, which is
    the level the injection has to reach; the train panel has no such line, because the
    marginal head was never scored on those rows.
    """
    panels = (("val_crps", "val_n", 1, marginal_crps, "validation"),
              ("train_crps", "train_n", 2, None, "train"))
    # The row count goes into the facet header, and it is built before the subplots rather
    # than patched into `layout.annotations` afterwards — every `add_annotation` below
    # appends to that same tuple, so an index into it is only stable by accident.
    titles = []
    for column, count, _, _, name in panels:
        rows = sweep.loc[sweep[column].notna(), count].dropna()
        titles.append(f"{name} · {int(rows.iloc[0]):,} player-seasons" if len(rows)
                      else name)
    fig = make_subplots(rows=1, cols=2, shared_xaxes=True, horizontal_spacing=0.10,
                        subplot_titles=titles)
    for column, count, col, reference, _ in panels:
        part = sweep[sweep[column].notna()]
        if part.empty:
            continue
        best = part.loc[part[column].idxmin()]
        color = th["series"][0] if col == 1 else th["series"][1]
        fig.add_trace(go.Scatter(
            x=part["sigma"], y=part[column], mode="lines+markers", showlegend=False,
            line=dict(color=color, width=2),
            marker=dict(size=[DOT + 5 if abs(s - float(best["sigma"])) < 1e-9 else DOT
                              for s in part["sigma"]], color=color,
                        line=dict(color=th["surface"], width=1)),
            hovertemplate="σ = %{x:.3f}<br>CRPS %{y:.2f}<extra></extra>"), row=1, col=col)
        fig.add_annotation(x=float(best["sigma"]), y=float(best[column]),
                           text=f"σ = {float(best['sigma']):.3f}", showarrow=False,
                           yanchor="top", yshift=-12,
                           font=dict(color=th["ink"], size=11), bgcolor=th["surface"],
                           borderpad=2, row=1, col=col)
        if reference is not None:
            fig.add_hline(y=float(reference), line=dict(color=th["ink2"], width=1),
                          layer="below", row=1, col=col)
            # At the left end, where the curve is at its highest and the strip under the
            # line is empty. Against the right end the label's own opaque chip cut the
            # curve in half, which a rendered PNG shows and a trace assertion cannot.
            fig.add_annotation(x=float(part["sigma"].min()), y=float(reference),
                               text="the marginal head", showarrow=False, xanchor="left",
                               yanchor="bottom", yshift=3,
                               font=dict(color=th["ink2"], size=10),
                               bgcolor=th["surface"], borderpad=2, row=1, col=col)

    for note in fig.layout.annotations[:2]:
        note.update(font=dict(color=th["ink"], size=13))
    fig.update_xaxes(title="injected per-player-season effect σ")
    fig.update_yaxes(title="CRPS (season minutes)")
    return apply_theme(fig, th, height, legend=False)


def fig_coupling(panel: pd.DataFrame, th: dict, slots: dict, title: str = "",
                 height: int = 300) -> go.Figure:
    """Each head's teammate correlation against the one a fixed team total forces on it.

    `panel` is `model_cards.teammate_coupling()`. Drawn as a **dumbbell**, one row per head,
    because the quantity is the gap: the reference is not a rival series, it is what the
    physics of a fixed pot requires at that head's own roster size, and it differs between
    the two rows for the ordinary reason that they cover different numbers of teammates.

    The measured value is a filled dot in the head's own slot and the forced value is a
    hollow one in the ink colour — the same "this is the reference, not a third model"
    encoding the game-length page uses for an observed count, and the same hollow marker the
    tournament page uses for a value that is not a measurement to be beaten.
    """
    rows = list(range(len(panel)))
    fig = go.Figure()
    for row, (_, part) in zip(rows, panel.iterrows()):
        fig.add_trace(go.Scatter(
            x=[part["forced"], part["measured"]], y=[row, row], mode="lines",
            showlegend=False, hoverinfo="skip",
            line=dict(color=th["axis"], width=2)))
    fig.add_trace(go.Scatter(
        x=panel["forced"], y=rows, mode="markers", name="forced by a fixed team total",
        marker=dict(size=DOT + 2, color="rgba(0,0,0,0)", symbol="circle",
                    line=dict(color=th["ink"], width=2)),
        customdata=list(panel["roster"]),
        hovertemplate="−1/(K−1) at K = %{customdata:.2f} is %{x:+.4f}"
                      "<extra>forced</extra>"))
    fig.add_trace(go.Scatter(
        x=panel["measured"], y=rows, mode="markers", name="what the head puts there",
        marker=dict(size=DOT + 2, color=_head_colors(th, panel["head"], slots),
                    line=dict(color=th["surface"], width=1)),
        customdata=list(panel["label"]),
        hovertemplate="%{customdata}<br>mean pairwise r %{x:+.4f}<extra>measured</extra>"))

    values = list(panel["measured"]) + list(panel["forced"])
    fig.update_xaxes(title="mean pairwise correlation between two teammates' season minutes",
                     range=_bar_text_range(values, floor_at_zero=False,
                                           room=WIDE_BAR_TEXT_ROOM))
    fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=list(panel["label"]),
                     showgrid=False, zeroline=False, range=[len(panel) - 0.5, -0.5])
    fig.update_layout(title=title)
    fig = apply_theme(fig, th, height)
    _reference_line(fig, 0.0, th, "independent draws", at_floor=True)
    return fig


def fig_layout_exposure(panel: pd.DataFrame, th: dict, slots: dict, title: str = "",
                        row_height: int = 132) -> go.Figure:
    """What the shipped availability layout puts a roster through, against what happened.

    `panel` is `inputs.layout_exposure()`. One facet per scoring-period statistic, because
    the three are in different units — two shares and a run length — and one row per role
    bucket inside each, so a reader reads **down** the facets to see the same four buckets
    change places.

    Drawn as a **dumbbell** for `fig_coupling`'s reason: the quantity a reader wants is the
    gap, and the second series is not a rival model. `observed` is the realized played/missed
    vector — the thing the layout exists to reproduce — so it takes the hollow ink marker
    every reference on this site takes, and the shipped layout takes the page's colour slot.
    Nothing else is drawn: the arms that selected this one are an argument, not an input.
    """
    metrics = list(dict.fromkeys(panel["metric_label"]))
    fig = make_subplots(rows=len(metrics), cols=1, subplot_titles=metrics,
                        vertical_spacing=min(0.16, 0.5 / max(len(metrics), 1)))
    for note in fig.layout.annotations:
        note.update(font=dict(color=th["ink"], size=13))

    for index, metric in enumerate(metrics):
        part = panel[panel["metric_label"] == metric]
        rows = list(range(len(part)))
        row = index + 1
        for y, item in zip(rows, part.itertuples(index=False)):
            fig.add_trace(go.Scatter(
                x=[item.observed, item.shipped], y=[y, y], mode="lines",
                showlegend=False, hoverinfo="skip",
                line=dict(color=th["axis"], width=2)), row=row, col=1)
        fig.add_trace(go.Scatter(
            x=part["observed"], y=rows, mode="markers", name="what actually happened",
            showlegend=index == 0, legendgroup="observed",
            marker=dict(size=DOT + 2, color="rgba(0,0,0,0)", symbol="circle",
                        line=dict(color=th["ink"], width=2)),
            customdata=list(part["population"]),
            hovertemplate="%{customdata}<br>realized %{x:,.4f}<extra>observed</extra>"),
            row=row, col=1)
        fig.add_trace(go.Scatter(
            x=part["shipped"], y=rows, mode="markers", name="what the simulator draws",
            showlegend=index == 0, legendgroup="shipped",
            marker=dict(size=DOT + 2,
                        color=_head_colors(th, part["head"], slots),
                        line=dict(color=th["surface"], width=1)),
            customdata=list(part["population"]),
            hovertemplate="%{customdata}<br>simulated %{x:,.4f}<extra>shipped</extra>"),
            row=row, col=1)

        values = list(part["shipped"]) + list(part["observed"])
        fig.update_xaxes(range=_bar_text_range(values, room=WIDE_BAR_TEXT_ROOM),
                         row=row, col=1)
        fig.update_yaxes(tickmode="array", tickvals=rows,
                         ticktext=list(part["population"]), showgrid=False,
                         zeroline=False, range=[len(part) - 0.5, -0.5], row=row, col=1)

    fig.update_layout(title=title)
    fig = apply_theme(fig, th, height=len(metrics) * row_height + 120)
    fig.update_layout(margin=dict(l=8, r=8, t=64, b=8))
    return fig


# ── The four figures the inputs page owns ─────────────────────────────────────
#
# Page 7 draws things that are not model outputs, so two of these encode something no other
# figure here does: a **calendar of days** and a **ladder of corrections**. The other two —
# the copula heatmap and the three-window panel — reuse `fig_correlation` and
# `fig_metric_facets` rather than growing near-copies, which is why they are absent below.
#
# The palette rules land on this page the same way they land everywhere else, with one
# wrinkle worth stating: the calendar's cell states are *categorical*, and the one
# distinction a reader most needs — whether a missed day can still be fetched — is
# deliberately **not** a fourth colour. It is constant along a row, so it rides on the row
# label instead. Two coloured states and a neutral is well inside `ALL_PAIRS_CAP`; four
# would not have been, and orange-beside-red is the exact pair the validation rejects.

#: Height of one program's row on the calendar, and of one snapshot row on the ADP timeline.
CALENDAR_ROW = 34
#: A capture is one day wide on an axis measured in days, so the marks are drawn as a
#: heatmap rather than as markers: a marker has a size in pixels and would misreport a
#: single missed day as a week at one zoom level and hide it at another.
#:
#: **Only the rows are separated, not the days**, and that is a rendered-figure finding: at
#: 150 days a two-pixel `xgap` is nearly as wide as a cell, so an unbroken run of captures
#: came out as a barcode and a genuinely missing day was indistinguishable from the gutter
#: between two present ones. A calendar's whole job is to make a hole visible.
CALENDAR_ROW_GAP = 3


def _stepped_colorscale(colors: list[str]) -> list[list]:
    """A discrete colorscale over `len(colors)` integer codes, with hard edges.

    Plotly has no categorical heatmap: a colorscale is continuous, so each band is written
    twice — once at its own floor and once at the next one — which is what makes the
    boundary a step rather than a gradient. Codes are then read against
    `zmin = -0.5, zmax = n - 0.5`, so code `k` lands in the middle of band `k`.
    """
    n = len(colors)
    scale = []
    for i, color in enumerate(colors):
        scale.append([i / n, color])
        scale.append([(i + 1) / n, color])
    return scale


def fig_calendar(grid: pd.DataFrame, labels: list[str], th: dict, states: tuple,
                 state_labels: dict, title: str = "",
                 row_height: int = CALENDAR_ROW) -> go.Figure:
    """Every capture program's coverage, one cell per day — the operational alarm.

    `grid` is `inputs.calendar_grid()` and `labels` is `inputs.row_labels()`, which carries
    the recoverability of a gap. **The row label is load-bearing rather than decorative**:
    the cells encode three states in two slots and a neutral, and whether a missed cell is a
    backlog item or an incident is a property of the *program*, so encoding it in the cell
    would spend a colour the palette cannot spare on a fact that never varies along a row.

    A day a program has no row for is left as a gap on the surface rather than painted as
    "captured nothing" — an `event` program has no schedule to have missed, and painting its
    empty stretches would invent 118 failures a year for a board that opens in October.

    A heatmap draws no legend, so the three states arrive as marker-only traces with no
    points in them. That is the only way to name a categorical colour in plotly without
    putting a continuous colourbar over three integers and calling it a scale.
    """
    days = sorted(grid["capture_date"].unique())
    day_index = {d: i for i, d in enumerate(days)}
    z = np.full((len(labels), len(days)), np.nan)
    hover = np.empty((len(labels), len(days)), dtype=object)
    hover[:] = ""
    for cell in grid.itertuples(index=False):
        if not 0 <= cell.row < len(labels):
            continue
        column = day_index[cell.capture_date]
        z[cell.row, column] = cell.code
        hover[cell.row, column] = (f"{cell.label}<br>{cell.capture_date}<br>"
                                   f"<b>{cell.state_label}</b><br>"
                                   f"{cell.records:,.0f} records")

    colors = [th["series"][0], th["neutral"], th["series"][1]]
    fig = go.Figure(go.Heatmap(
        z=z, x=days, y=labels, zmin=-0.5, zmax=len(states) - 0.5,
        colorscale=_stepped_colorscale(colors), showscale=False, hoverongaps=False,
        xgap=0, ygap=CALENDAR_ROW_GAP, customdata=hover,
        hovertemplate="%{customdata}<extra></extra>"))
    for state, color in zip(states, colors):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", name=state_labels.get(state, state),
            marker=dict(symbol="square", size=11, color=color,
                        line=dict(color=th["axis"], width=1)),
            hoverinfo="skip", showlegend=True))

    fig.update_xaxes(showgrid=False, tickfont=dict(size=10))
    fig.update_yaxes(showgrid=False, zeroline=False, tickfont=dict(size=11),
                     autorange="reversed")
    fig.update_layout(title=title)
    return apply_theme(fig, th, height=len(labels) * row_height + 150)


def fig_adp_lag(snaps: pd.DataFrame, th: dict, title: str = "",
                row_height: int = CALENDAR_ROW) -> go.Figure:
    """Every ADP board on the axis that decides whether it may be used: days from tip-off.

    One row per season, one mark per captured board, `x` measured from that season's first
    game. **Everything at or left of zero is legal and everything right of it is not**, so
    the finding is a position on the axis rather than a colour — the two series are then a
    second route to the same fact and the marker symbol is a third, which is what the relief
    rule asks for on a page where the distinction decides whether a season exists at all.

    The zero line is drawn bare. The x axis title already says what zero is, and a labelled
    reference has no placement that is safe in general — see `_reference_line`.
    """
    seasons = sorted(snaps["season"].unique(), reverse=True)
    position = {s: i for i, s in enumerate(seasons)}
    fig = go.Figure()
    for legal, name, symbol, color in (
            (True, "observed before the opener — usable", "circle", th["series"][0]),
            (False, "observed after it — dropped by `training_rows`", "x-thin",
             th["series"][1])):
        part = snaps[snaps["legal"] == legal]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["lag_days"], y=[position[s] for s in part["season"]],
            mode="markers", name=name,
            marker=dict(size=DOT + 1, color=color, symbol=symbol,
                        line=dict(color=color if symbol.startswith("x") else th["surface"],
                                  width=2 if symbol.startswith("x") else 1)),
            customdata=list(zip(part["source"], part["as_of_date"], part["rows"])),
            hovertemplate="%{customdata[0]} · %{customdata[1]}<br>"
                          "%{x:+.0f} days from tip-off<br>%{customdata[2]:,} rows"
                          "<extra></extra>"))

    fig.update_xaxes(title="days from the season's first game — negative is before it")
    fig.update_yaxes(tickmode="array", tickvals=list(position.values()),
                     ticktext=seasons, showgrid=False, zeroline=False,
                     range=[len(seasons) - 0.5, -0.5])
    fig.update_layout(title=title)
    fig = apply_theme(fig, th, height=max(280, len(seasons) * row_height + 130))
    _reference_line(fig, 0.0, th, "")
    return fig


def fig_ladder(ladder: pd.DataFrame, th: dict, title: str = "",
               row_height: int = CALENDAR_ROW) -> go.Figure:
    """The consensus→DK recalibration ladder, in the order it was built.

    Measured order rather than sorted order, because each rung is a correction *added to*
    the one above it; sorting by score would turn a ladder into a menu and quietly invite
    reading the best row as the shipped one, which it is not.

    Highlight-and-gray on the shipped arm: seven rungs is past `ALL_PAIRS_CAP`, and the one
    distinction worth a slot is which rung `adp_transfer.parquet` actually holds.
    """
    rows = list(range(len(ladder)))
    fig = go.Figure(go.Bar(
        x=ladder["mean_abs_rank_gap"], y=rows, orientation="h", showlegend=False,
        marker=dict(color=[th["series"][0] if s else th["muted"]
                           for s in ladder["shipped"]],
                    line=dict(color=th["surface"], width=1)),
        text=[f"{v:.2f}" for v in ladder["mean_abs_rank_gap"]], textposition="outside",
        textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
        customdata=[("shipped" if s else "measured, not shipped")
                    for s in ladder["shipped"]],
        hovertemplate="%{y}<br>%{x:.3f} picks<extra>%{customdata}</extra>"))
    fig.update_xaxes(title="mean absolute rank gap against the DK board, in picks",
                     range=_bar_text_range(ladder["mean_abs_rank_gap"],
                                           room=WIDE_BAR_TEXT_ROOM))
    fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=list(ladder["arm"]),
                     showgrid=False, zeroline=False, range=[len(ladder) - 0.5, -0.5])
    fig.update_layout(title=title, bargap=0.35)
    return apply_theme(fig, th, height=max(260, len(ladder) * row_height + 120),
                       legend=False)


def fig_block_inflation(panel: pd.DataFrame, th: dict, highlight: str = "min",
                        title: str = "", row_height: int = 26) -> go.Figure:
    """Ten-game block variance inflation per component — where sequential structure is.

    One is independence, and the axis title says so, so the reference line is drawn bare.
    `highlight` takes slot 0 because exactly one of these rows is an *input*: minutes carry
    the serial dependence and every other head is drawn to show that it does not, which is
    the measured null the simulator's third rule rests on.
    """
    rows = list(range(len(panel)))
    fig = go.Figure(go.Bar(
        x=panel["block_inflation"], y=rows, orientation="h", showlegend=False,
        marker=dict(color=[th["series"][0] if c == highlight else th["muted"]
                           for c in panel["component"]],
                    line=dict(color=th["surface"], width=1)),
        text=[f"{v:.3f}×" for v in panel["block_inflation"]], textposition="outside",
        textfont=dict(size=BAR_TEXT_SIZE, color=th["ink2"]), cliponaxis=False,
        customdata=list(zip(panel["kind"], panel["lag1"])),
        hovertemplate="%{y}<br>%{x:.4f}× independent draws<br>lag-1 r "
                      "%{customdata[1]:+.4f}<extra>%{customdata[0]}</extra>"))
    fig.update_xaxes(title="ten-game block variance, × independent draws",
                     range=_bar_text_range(panel["block_inflation"],
                                           room=PERCENT_BAR_TEXT_ROOM))
    fig.update_yaxes(tickmode="array", tickvals=rows, ticktext=list(panel["component"]),
                     showgrid=False, zeroline=False, range=[len(panel) - 0.5, -0.5])
    fig.update_layout(title=title, bargap=0.3)
    fig = apply_theme(fig, th, height=max(280, len(panel) * row_height + 120),
                      legend=False)
    # Its own line rather than `_reference_line`, for two reasons a rendered PNG supplied.
    # At the shared helper's hairline weight it was indistinguishable from the gridlines it
    # sits among, and eleven of these twelve bars end within half a unit of it — so the
    # question the chart exists to answer was answered for nobody. And it goes **above** the
    # bars: below them it survives only in the gutters between rows, which reads as a dashed
    # line, and `theme.py` bans dashes because a dash is supposed to mean something.
    fig.add_vline(x=1.0, line=dict(color=th["ink2"], width=2), layer="above")
    return fig


# ── The Overview page ─────────────────────────────────────────────────────────
#
# One figure, and it is the only chart here that plots no data. The pipeline diagram is a
# *layout* — five boxes in a row with the reader's eye pushed left to right — so it is
# built out of shapes and annotations like the radial grid, for the same reason: everything
# on it is chrome, and nothing on it should be pickable or hoverable as if it were a point.
#
# Its figures still come from artifacts (`dashboard/overview.py` builds the strings), which
# is the charter bound that page exists under. What is typed here is the geometry.

#: Box geometry on a unit canvas. The gap is where the arrow between two stages goes, so it
#: is wide enough to read as a connector rather than as a seam.
STAGE_GAP = 0.035
#: Where the three lines of a box sit, as a fraction of its height. The figure is the tall
#: one and takes the middle; the title labels it from above and the note explains from below.
STAGE_TITLE_Y = 0.79
STAGE_FIGURE_Y = 0.50
STAGE_NOTE_Y = 0.20
STAGE_HEIGHT = 150


def fig_pipeline(stages: list, th: dict, title: str = "") -> go.Figure:
    """The project in one row of boxes: data → heads → seasons → strategies → contests.

    Deliberately uncoloured. A five-step chain is past `ALL_PAIRS_CAP` and, more to the
    point, the steps are not a scale and not categories being compared — giving them five
    hues would be an encoding that decodes to nothing. The boxes take the neutral fill and
    every figure prints itself, so the relief rule is satisfied trivially.

    `stages` is a list of `overview.Stage`; the notes must already be `<br>`-wrapped, since
    a plotly annotation does not wrap and silently runs off the end of its box.
    """
    fig = go.Figure()
    n = max(len(stages), 1)
    width = (1.0 - STAGE_GAP * (n - 1)) / n

    for i, stage in enumerate(stages):
        x0 = i * (width + STAGE_GAP)
        fig.add_shape(type="rect", x0=x0, x1=x0 + width, y0=0.0, y1=1.0,
                      line=dict(color=th["axis"], width=1), fillcolor=th["neutral"],
                      layer="below")
        centre = x0 + width / 2
        for y, text, size, color in (
                (STAGE_TITLE_Y, stage.title, 12, th["ink2"]),
                (STAGE_FIGURE_Y, stage.figure, 25, th["ink"]),
                (STAGE_NOTE_Y, stage.note, 11, th["muted"])):
            fig.add_annotation(x=centre, y=y, text=text, showarrow=False,
                               xanchor="center", yanchor="middle",
                               font=dict(family=FONT, size=size, color=color))
        if i:
            # Drawn as an annotation rather than a shape because plotly puts an arrowhead
            # on an annotation and not on a line shape, and the head is what makes the row
            # a chain rather than five tiles that happen to be adjacent.
            fig.add_annotation(x=x0, y=0.5, ax=x0 - STAGE_GAP, ay=0.5,
                               xref="x", yref="y", axref="x", ayref="y",
                               text="", showarrow=True, arrowhead=2, arrowsize=1.1,
                               arrowwidth=1.4, arrowcolor=th["axis"])

    # The margins below are zero so the row spans the column, which puts the outer boxes'
    # own 1px borders exactly on the paper edge and clips them — visible only in a
    # rendering. The range carries the padding instead.
    fig.update_xaxes(visible=False, range=[-0.012, 1.012], fixedrange=True)
    fig.update_yaxes(visible=False, range=[-0.03, 1.03], fixedrange=True)
    fig.update_layout(title=title)
    fig = apply_theme(fig, th, height=STAGE_HEIGHT, legend=False)
    # The theme leaves 48px of top margin for a title; this figure has none, and on a page
    # whose whole constraint is one screen that is 48px of nothing.
    fig.update_layout(margin=dict(l=0, r=0, t=4, b=4), hovermode=False)
    return fig
