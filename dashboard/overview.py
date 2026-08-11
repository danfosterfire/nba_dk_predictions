"""The Overview page's pure layer — five hero readings, five pipeline stages, one route.

Page 1, and the one page of prose on this dashboard. The charter it lives under
(`docs/dashboard-plan.md`, "Charter amendment 2026-08-10") grants that exemption on three
terms, and **the second of them is what this module is for**: typed prose may say what the
project does, it may not state a *result*. So every figure the page shows arrives here as a
lookup into an artifact — the season-total MAE is read from `season_total_metrics.csv` like
every other number on the site, and a re-run that moves it moves the page.

The mechanism is `Spec`: a reading is a `build` over the frames it `needs`, so a figure
cannot be typed into the view even by accident, and a repo missing one artifact drops the
readings that depend on it rather than printing a stale constant or taking the page down.

Nothing here imports Streamlit or `src/`, so the whole page is testable as data:
`stage_readings` and `tile_readings` take frames and hand back strings.
"""

import textwrap
from typing import Callable, NamedTuple

import pandas as pd

from dashboard.strategy import pretty_tournament

# ── The artifacts, by the target that writes them ─────────────────────────────
#
# `directory` keys the two artifact roots the view resolves (`artifacts.eda_dir` and
# `artifacts.predictions_dir`), so this table stays free of paths and the view stays free
# of filenames. Eight small CSVs: the page is the cheapest one on the dashboard, which is
# what a landing page has to be.

EDA = "eda"
PREDICTIONS = "predictions"


class Source(NamedTuple):
    key: str
    directory: str
    filename: str
    target: str


SOURCES: tuple[Source, ...] = (
    Source("coverage", EDA, "game_length_coverage.csv", "make game-length"),
    Source("budget", EDA, "variance_budget.csv", "make variance-budget"),
    Source("cards", PREDICTIONS, "model_card_index.csv", "make model-cards"),
    Source("season_total", PREDICTIONS, "season_total_metrics.csv", "make season-total"),
    Source("components", PREDICTIONS, "stan_component_metrics.csv",
           "make stan-components"),
    Source("simulation", PREDICTIONS, "sim_season_gate_a.csv", "make simulate-season"),
    Source("sweep", PREDICTIONS, "strategy_sweep.csv", "make strategy-sweep"),
    Source("shipped", PREDICTIONS, "strategy_shipped.csv", "make strategy-sweep"),
    Source("bracket", PREDICTIONS, "bracket_structure.csv", "make bracket"),
)

#: The flagship tier, and the only tournament this page names. `strategy.TARGET_TIERS`
#: carries both swept tiers; the Overview shows one, because a landing page that makes the
#: reader pick a tournament has already stopped being a landing page.
HEADLINE_TIER = "600k_shootaround"

#: The season-total ladder's two ends: the shipped availability head, and what the season
#: total costs if you assume nobody misses a game. The gap between them is the largest
#: single measured win in the project, which is why it is the first tile.
SHIPPED_TREATMENT = "beta_binomial"
NAIVE_TREATMENT = "full_season"

#: The no-fit floor in `stan_component_metrics.csv`'s vocabulary, and the head family the
#: quoted band comes from. **The counts only.** Read off the conversion heads too, the band
#: opens to [0.13, 0.96] — three of the four conversions score under 0.35 — and a floor that
#: wide is not a floor. `docs/simulations-plan.md` had to re-derive this once already.
FLOOR_VARIANT = "carry_forward"
FLOOR_KIND = "count"


# ── Lookups ───────────────────────────────────────────────────────────────────

class MissingRow(LookupError):
    """The artifact exists and the row this page quotes does not."""


def _one(frame: pd.DataFrame, column: str, **filters) -> float:
    """The single value at `column` under `filters`, or raise naming what was asked for.

    Loud rather than NaN: a landing page printing `nan` where a result belongs is the
    failure mode `model_cards.text()` exists for, and here there is no honest fallback —
    the tile has nothing to say without its row.
    """
    sub = frame
    for key, value in filters.items():
        sub = sub[sub[key] == value]
    if sub.empty:
        raise MissingRow(f"no row for {filters} in a frame of {len(frame)}")
    return float(sub.iloc[0][column])


def player_games(coverage: pd.DataFrame) -> float:
    return _one(coverage, "player_games", analysis="feasibility", season="all",
                season_type="regular")


def seasons(coverage: pd.DataFrame) -> int:
    """Seasons on disk, counted from the per-season rows rather than from the config.

    `all` is the pooled row and is not a season.
    """
    rows = coverage[(coverage["analysis"] == "feasibility")
                    & (coverage["season_type"] == "regular")
                    & (coverage["season"] != "all")]
    return int(rows["season"].nunique())


def minutes_share(budget: pd.DataFrame) -> float:
    """Own minutes as a share of the **within-player-season** residual, not of the total."""
    return _one(budget, "share_of_variance", source="own_minutes")


def heads(cards: pd.DataFrame) -> int:
    return int(len(cards))


def divergences(cards: pd.DataFrame) -> int:
    return int(cards["divergences"].sum())


def season_total_mae(metrics: pd.DataFrame, treatment: str) -> float:
    return _one(metrics, "value", treatment=treatment, group="all",
                metric="mae_dk_total")


def floor_band(components: pd.DataFrame) -> tuple[float, float]:
    """The no-fit floor's validation R² across the count heads, lowest and highest."""
    rows = components[(components["variant"] == FLOOR_VARIANT)
                      & (components["kind"] == FLOOR_KIND)]
    if rows.empty:
        raise MissingRow(f"no {FLOOR_VARIANT}/{FLOOR_KIND} rows in a frame of "
                         f"{len(components)}")
    return float(rows["val_r2"].min()), float(rows["val_r2"].max())


def simulated_worlds(simulation: pd.DataFrame) -> int:
    return int(simulation["n_sims"].max())


def strategies(sweep: pd.DataFrame) -> int:
    return int(sweep["strategy"].nunique())


def contests(bracket: pd.DataFrame) -> int:
    return int(bracket["tournament"].nunique())


def advance(shipped: pd.DataFrame, tournament: str = HEADLINE_TIER) -> dict[str, float]:
    """The shipped arm's realized Round-1 advance rate, its lift, and the field null.

    The null is *derived* rather than read: the artifact carries the rate and the lift over
    a symmetric field, and their difference is that field's own rate. Deriving it keeps the
    three numbers on the page arithmetically consistent with each other — a rounded 1/6
    typed in beside them would not be.
    """
    rate = _one(shipped, "realized_p_advance", tournament=tournament)
    lift = _one(shipped, "realized_lift", tournament=tournament)
    return {"rate": rate, "lift": lift, "null": rate - lift,
            "seasons": _one(shipped, "realized_seasons", tournament=tournament)}


# ── Readings ──────────────────────────────────────────────────────────────────

class Reading(NamedTuple):
    """One hero tile. `delta` is empty where there is nothing to compare against."""

    label: str
    value: str
    note: str
    delta: str = ""


class Stage(NamedTuple):
    """One box of the pipeline diagram: a figure, what it counts, and one line of why."""

    title: str
    figure: str
    note: str


class Spec(NamedTuple):
    """A reading, plus the source keys it cannot be built without."""

    needs: tuple[str, ...]
    build: Callable[[dict[str, pd.DataFrame]], Reading | Stage]


def _resolve(specs: tuple[Spec, ...], frames: dict[str, pd.DataFrame]) -> list:
    """Every reading whose sources are present, in declaration order.

    A partially built repo loses the readings it cannot support and keeps the rest, which
    is the behaviour a landing page wants: the alternative is a blank screen because one of
    eight CSVs has not been generated yet.
    """
    return [spec.build(frames) for spec in specs
            if all(key in frames for key in spec.needs)]


#: Stage notes are kept to one wrapped line each, and that is a layout constraint rather
#: than a house style: five boxes across a sidebar-narrowed column leaves each one about
#: 140 px, so a note long enough to wrap twice pushes the diagram tall enough to cost the
#: page its screen. The arrow chain carries the meaning the notes would otherwise spell out.
STAGE_SPECS: tuple[Spec, ...] = (
    Spec(("coverage",), lambda f: Stage(
        "Box scores",
        f"{player_games(f['coverage']):,.0f}",
        f"player-games, {seasons(f['coverage'])} seasons")),
    Spec(("cards",), lambda f: Stage(
        "Bayesian heads",
        f"{heads(f['cards'])}",
        "fitted in Stan")),
    Spec(("simulation",), lambda f: Stage(
        "Simulated seasons",
        f"{simulated_worlds(f['simulation']):,.0f}",
        "drawn per player-season")),
    Spec(("sweep",), lambda f: Stage(
        "Drafting strategies",
        f"{strategies(f['sweep'])}",
        "swept over seven axes")),
    Spec(("bracket",), lambda f: Stage(
        "Real contests",
        f"{contests(f['bracket'])}",
        "captured from DraftKings")),
)


def _season_total(f: dict[str, pd.DataFrame]) -> Reading:
    shipped = season_total_mae(f["season_total"], SHIPPED_TREATMENT)
    naive = season_total_mae(f["season_total"], NAIVE_TREATMENT)
    return Reading(
        "Season-total error",
        f"{shipped:,.1f} dk_pts",
        "Mean absolute error on a player's season `dk_pts` total, on the validation "
        "seasons. The comparison is the same total with everyone assumed to play every "
        f"game, which reads {naive:,.1f}. Read from `season_total_metrics.csv` "
        "(`make season-total`).",
        # Kept under ~26 characters, which is what a tile five across can show: a metric's
        # delta neither wraps nor ellipsizes honestly, so a longer phrase renders as
        # `-210.3 against assuming a f…` — a comparison cut mid-word, which is worse than
        # none. The full framing is in the note above, which the tile carries as its help.
        delta=f"-{naive - shipped:,.1f} vs a full season")


def _minutes(f: dict[str, pd.DataFrame]) -> Reading:
    return Reading(
        "Minutes, unknown at draft",
        f"{minutes_share(f['budget']):.1%}",
        "Share of the *within-player-season* variance in per-game `dk_pts` that the "
        "player's own minutes explain — the largest single thing nobody knows before the "
        "season starts, which is why minutes get two heads of their own. Read from "
        "`variance_budget.csv` (`make variance-budget`).")


def _floor(f: dict[str, pd.DataFrame]) -> Reading:
    low, high = floor_band(f["components"])
    return Reading(
        "Floor with nothing fitted",
        f"R² {low:.2f}–{high:.2f}",
        "Validation R² of a no-fit floor across the seven count heads: last season's "
        "per-36 rate times this game's minutes, nothing fitted. Every head in the project "
        "is quoted against it, and that band is why the effort goes into availability "
        "rather than into rate features. Read from `stan_component_metrics.csv` "
        "(`make stan-components`).")


def _sampler(f: dict[str, pd.DataFrame]) -> Reading:
    return Reading(
        f"Divergences, {heads(f['cards'])} fits",
        f"{divergences(f['cards']):,}",
        f"Across all {heads(f['cards'])} persisted fits, every one of which also clears "
        "its R̂ and effective-sample-size bars. Read from `model_card_index.csv` "
        "(`make model-cards`).")


def _advance(f: dict[str, pd.DataFrame]) -> Reading:
    got = advance(f["shipped"])
    return Reading(
        "Round 1 advance rate",
        f"{got['rate']:.1%}",
        "The shipped strategy's rosters in the "
        f"{pretty_tournament(HEADLINE_TIER)}, replayed against realized box scores over "
        f"{got['seasons']:.0f} validation seasons. A symmetric field advances "
        f"{got['null']:.1%} by construction, so this is a backtest on two seasons rather "
        "than a track record. Read from `strategy_shipped.csv` (`make strategy-sweep`).",
        # Points on the advance-rate axis, not a percentage change. Printing the null right
        # beside it is what disambiguates that in the width a delta has: the tile reads
        # 29.4%, +12.7% and 16.7%, and those three add up in front of the reader.
        delta=f"+{got['lift']:.1%} vs a {got['null']:.1%} field")


TILE_SPECS: tuple[Spec, ...] = (
    Spec(("season_total",), _season_total),
    Spec(("budget",), _minutes),
    Spec(("components",), _floor),
    Spec(("cards",), _sampler),
    Spec(("shipped",), _advance),
)


def stage_readings(frames: dict[str, pd.DataFrame]) -> list[Stage]:
    return _resolve(STAGE_SPECS, frames)


def tile_readings(frames: dict[str, pd.DataFrame]) -> list[Reading]:
    return _resolve(TILE_SPECS, frames)


# ── The route out ─────────────────────────────────────────────────────────────
#
# The page's last job, and the reason it was built last: a reader who arrives at the URL
# with no context needs somewhere to go, and there was nowhere to send them until the eight
# pages existed. Keyed by `app.VIEWS[i].url_path` so a retitled page keeps its blurb, and
# **the blurbs state no results** — that is the charter's second bound again, and a route
# label is exactly where a stray headline would try to creep back in.

class Route(NamedTuple):
    url_path: str
    blurb: str


#: One line each, for the same reason the stage notes are one line: eight blurbs at three
#: lines apiece is 200 px of the screen this page is not allowed to exceed. Each says what
#: its page holds, in the page's own terms, and none of them states a result.
ROUTES: tuple[Route, ...] = (
    Route("fingerprints", "Player style on the league's axes."),
    Route("availability", "Games played, and the tenure model."),
    Route("minutes", "Two units, two opposite verdicts."),
    Route("components", "Eleven heads against a no-fit floor."),
    Route("game-length", "Overtime — whether, and how deep."),
    Route("inputs", "The market and the fixed constants."),
    Route("tournament", "The sweep: simulated vs realized."),
    Route("draft-room", "The live board. One click per pick."),
)


# ── Text ──────────────────────────────────────────────────────────────────────

#: Characters per line inside a diagram box. Plotly annotations do not wrap, so the note is
#: broken here — in the layer a test can read — rather than by eye against a rendering.
#: Sized for the *narrowest* container the page is checked at rather than the widest, since
#: the figure is drawn at `width="stretch"` and Python never learns how wide that was: at a
#: 1280 px viewport the sidebar leaves each of the five boxes about 140 px, which is 26
#: characters at the note's font size.
NOTE_WIDTH = 26


def wrap(text: str, width: int = NOTE_WIDTH) -> str:
    """`text` broken into plotly's `<br>`-separated lines."""
    return "<br>".join(textwrap.wrap(text, width=width)) if text else ""
