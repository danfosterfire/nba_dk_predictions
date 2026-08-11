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

from pathlib import Path

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


# ── The page chrome, derived from the palette ─────────────────────────────────

# Streamlit 1.60 carries per-mode theme config — `[theme.light]` and `[theme.dark]`,
# each with a `sidebar` sub-table — which the project config predated. Before it, the
# page background, sidebar, body text and tables followed Streamlit's own defaults
# while only the plot surfaces came from this palette, so a reader in dark mode read
# light charts on a dark page.
#
# These tables close that. They are *derived* rather than retyped: a hand-copied hex
# that drifted from `THEMES` would put the measured contrast figures in
# `dashboard/README.md` silently out of date, so `config_toml()` renders the file and a
# test parses the checked-in copy back against this mapping.
#
# The two roles the mapping rests on, both of which hold in *both* modes:
#   * `surface` is the page's ground, and is the same value `apply_theme` paints a
#     figure's paper with — so a chart has no visible edge against the page it is on.
#   * `plane` is the recessive panel behind that ground: the sidebar, a dataframe
#     header, a code block. Inside the sidebar the pair inverts, so a widget sitting on
#     `surface` reads as raised off the `plane` behind it.
# `neutral` is deliberately *not* used here — it is the diverging scale's midpoint, a
# chart colour, and chrome borrowing it would make a data value look like furniture.
#: Each Streamlit setting written as the palette **role** it takes, never as a colour.
#: The indirection is what makes the rule below assertable in both modes — in dark,
#: `neutral` and `axis` happen to be the same hex, so "the chrome does not borrow the
#: diverging midpoint" is not a claim any comparison of *values* could check.
CHROME_ROLES: dict[str, str] = {
    "primaryColor": "accent",
    "backgroundColor": "surface",
    "secondaryBackgroundColor": "plane",
    "textColor": "ink",
    "linkColor": "accent",
    "borderColor": "grid",
    "dataframeBorderColor": "axis",
    "dataframeHeaderBackgroundColor": "plane",
    "codeBackgroundColor": "plane",
    "codeTextColor": "ink2",
    "font": "font",
}

#: The sidebar inverts the surface/plane pair, so a widget inside it reads as raised off
#: the panel behind it rather than flush with it, and takes the heavier hairline for the
#: one border a reader actually sees as an edge.
SIDEBAR_ROLES: dict[str, str] = {
    **CHROME_ROLES,
    "backgroundColor": "plane",
    "secondaryBackgroundColor": "surface",
    "dataframeHeaderBackgroundColor": "surface",
    "codeBackgroundColor": "surface",
    "borderColor": "axis",
}


def _role(palette: dict, role: str) -> str:
    """One palette role as a colour. `accent` is the first categorical slot, which is the
    only place a *data* colour is deliberately reused as chrome: a link and a selected
    control are the page saying "this one", which is what slot 1 says in every chart."""
    if role == "accent":
        return palette["series"][0]
    if role == "font":
        return FONT
    return palette[role]


def streamlit_theme(mode: str) -> dict:
    """The `[theme.<mode>]` table for `.streamlit/config.toml`, from `THEMES[mode]`.

    Colour-only, plus the one font stack — no radii, no widget-border switches. Nothing
    here is a style preference the palette does not already carry.
    """
    th = THEMES[mode]
    table = {key: _role(th, role) for key, role in CHROME_ROLES.items()}
    table["sidebar"] = {key: _role(th, role) for key, role in SIDEBAR_ROLES.items()}
    return table


# `theme.chartCategoricalColors` / `chartSequentialColors` / `chartDivergingColors` are
# deliberately absent: they exist only at the top level, i.e. once for both modes, while
# `SERIES` is selected per mode rather than flipped. Setting them would push one mode's
# eight slots onto the other. Nothing here draws with them anyway — every chart on the
# site is plotly through `charts.py`, which takes the palette as an argument.
CONFIG_HEADER = """\
# Project-level Streamlit config for `make dashboard`.
#
# GENERATED — `make dashboard-config` writes this file from `dashboard/theme.py`, and
# `tests/test_dashboard.py` parses it back against `theme.streamlit_theme()`. Edit the
# palette or `config_toml()`, never the hex below.
#
# headless = true is the load-bearing line: without it Streamlit's first run on a
# machine stops at an interactive "Email:" onboarding prompt, and `make dashboard`
# exits 255 instead of serving.
#
# The [theme.light] / [theme.dark] tables are the whole page in the palette the charts
# were validated against, so one appearance choice moves the background, the sidebar,
# the body text, the tables and the plots together. Streamlit's own appearance setting
# is that choice — see `dashboard/shell.py`.
"""


def _toml_value(value: str) -> str:
    """A TOML literal string, so the font stack's own double quotes need no escaping."""
    return f"'{value}'"


def config_toml() -> str:
    """The full text of `.streamlit/config.toml`, palette included."""
    lines = [CONFIG_HEADER, "[server]", "headless = true", "",
             "[browser]", "gatherUsageStats = false"]
    for mode in THEMES:
        table = streamlit_theme(mode)
        flat = {k: v for k, v in table.items() if not isinstance(v, dict)}
        lines += ["", f"[theme.{mode}]"]
        lines += [f"{k} = {_toml_value(v)}" for k, v in flat.items()]
        for sub, values in ((k, v) for k, v in table.items() if isinstance(v, dict)):
            lines += ["", f"[theme.{mode}.{sub}]"]
            lines += [f"{k} = {_toml_value(v)}" for k, v in values.items()]
    return "\n".join(lines) + "\n"


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


def write_config(root: Path) -> Path:
    """Render `.streamlit/config.toml` from the palette. `make dashboard-config`."""
    dest = root / ".streamlit" / "config.toml"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(config_toml())
    n = sum(len(v) if isinstance(v, dict) else 1
            for mode in THEMES for v in streamlit_theme(mode).values())
    print(f"wrote {n:,} theme settings across {len(THEMES)} modes → {dest}")
    return dest


if __name__ == "__main__":
    write_config(Path(__file__).resolve().parent.parent)
