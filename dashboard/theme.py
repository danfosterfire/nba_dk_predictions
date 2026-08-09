"""Palette and chart chrome — the validated reference instance, used unmodified.

Eight categorical slots in fixed order, a single-hue blue ramp for magnitude, and
blue↔red with a neutral gray midpoint for polarity. Two rules from that validation
constrain every chart in `charts.py` and are worth knowing before editing one:

- The eight slots clear the CVD gates on *adjacent* pairs (bars, lines, stacks),
  but only the **first three** clear them on *all* pairs. So any scatter, and any
  chart where non-adjacent series sit side by side, caps at three coloured series
  and uses highlight-and-gray past that.
- Three light-mode slots fall below 3:1 contrast on the light surface, which
  obliges the relief rule: every chart ships a table-view twin in an expander.

Plot surfaces are pinned to the exact surfaces the palette was validated against
(`#fcfcfb` / `#1a1a19`) rather than inherited from Streamlit's page chrome, so the
measured contrast and separation figures apply as documented.

This module imports no Streamlit — it is plotly and numpy only, so the colour rules
are testable without a runtime.
"""

import numpy as np
import plotly.graph_objects as go

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
    """Recessive hairline chrome, pinned surfaces, no dashes anywhere.

    The title is set as `text` **and** font rather than font alone: styling
    `title_font` on an untitled figure leaves plotly.js a title object with no text,
    which it renders as the literal string "undefined" in a browser. Kaleido does not,
    so this is only ever seen on the page.
    """
    fig.update_layout(
        height=height,
        paper_bgcolor=th["surface"], plot_bgcolor=th["surface"],
        font=dict(family=FONT, size=13, color=th["ink2"]),
        title=dict(text=fig.layout.title.text or "",
                   font=dict(family=FONT, size=15, color=th["ink"])),
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
