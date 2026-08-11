"""The Overview page's pure layer — four sections of a paper, five stages, nine routes.

Page 1, and the one page of prose on this dashboard. The charter it lives under
(`docs/dashboard-plan.md`, "Charter amendment 2026-08-10") grants that exemption on three
terms, and **the second of them is what this module is for**: typed prose may say what the
project does, it may not state a *result*. So every figure the page shows arrives here as a
lookup into an artifact — the season-total MAE is read from `season_total_metrics.csv` like
every other number on the site, and a re-run that moves it moves the page.

The mechanism is `Spec`: a reading is a `build` over the frames it `needs`, so a figure
cannot be typed into the view even by accident, and a repo missing one artifact drops the
readings that depend on it rather than printing a stale constant or taking the page down.

**The page is prose now rather than a tile row, and that moved where the bound has to
bite.** A hero tile had nowhere to put a typed number: its value came from a `Spec` and its
label was a label. A paragraph has room for a typed one in the middle of a sentence, which
is exactly how the deleted walkthrough drifted. So a `Section`'s body is an ordered mix of
two kinds of fragment — a `str` is typed prose and **carries no digit at all**, a `Spec` is
a lookup — and `test_typed_prose_carries_no_digit` is bound 2 restated at the sentence.
That is why the contest rules below are spelled in words: *sixteen players* is a rule and
`46.4%` is a measurement, and on this page a digit means the second kind.

Nothing here imports Streamlit or `src/`, so the whole page is testable as data:
`paragraphs` and `stage_readings` take frames and hand back strings.
"""

import textwrap
from typing import Callable, NamedTuple

import pandas as pd

from dashboard.strategy import pretty_tournament

# ── The artifacts, by the target that writes them ─────────────────────────────
#
# `directory` keys the two artifact roots the view resolves (`artifacts.eda_dir` and
# `artifacts.predictions_dir`), so this table stays free of paths and the view stays free
# of filenames. Ten small CSVs: the page is the cheapest one on the dashboard, which is
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
    Source("weekly", PREDICTIONS, "weekly_score_index.csv", "make weekly-scores"),
    Source("sweep", PREDICTIONS, "strategy_sweep.csv", "make strategy-sweep"),
    Source("shipped", PREDICTIONS, "strategy_shipped.csv", "make strategy-sweep"),
    Source("bracket", PREDICTIONS, "bracket_structure.csv", "make bracket"),
)

#: The flagship tier, and the only tournament this page names. `strategy.TARGET_TIERS`
#: carries both swept tiers; the Overview shows one, because a landing page that makes the
#: reader pick a tournament has already stopped being a landing page.
HEADLINE_TIER = "600k_shootaround"

#: The season-total ladder's four ends. The first two are the Results sentence — the
#: shipped availability head against what the season total costs if you assume nobody
#: misses a game, which is the largest single measured win in the project. The two oracles
#: are the Discussion sentence: they say which *half* of the remaining error is worth
#: attacking, and they disagree with the intuition that the rate is the hard part.
SHIPPED_TREATMENT = "beta_binomial"
NAIVE_TREATMENT = "full_season"
ORACLE_GP_TREATMENT = "oracle_gp"
ORACLE_RATE_TREATMENT = "oracle_rate"

#: The no-fit floor in `stan_component_metrics.csv`'s vocabulary, and the head family the
#: quoted band comes from. **The counts only.** Read off the conversion heads too, the band
#: opens to [0.13, 0.96] — three of the four conversions score under 0.35 — and a floor that
#: wide is not a floor. `docs/simulations-plan.md` had to re-derive this once already.
FLOOR_VARIANT = "carry_forward"
FLOOR_KIND = "count"

#: The weekly facet the Discussion sentence reads. `weekly_score_index.csv` carries four
#: rows, because three of the twenty scoring slots are **double** weeks and a double week
#: carries about twice the games — so the two period types are two units and pooling them
#: would report a calendar fact as a model miss. One week is the unit seventeen of the
#: twenty slots are, and `validation` is the split every other figure on this page is read
#: on.
WEEKLY_PERIOD = "week"
WEEKLY_SPLIT = "validation"


# ── Lookups ───────────────────────────────────────────────────────────────────

class MissingRow(LookupError):
    """The artifact exists and the row this page quotes does not."""


def _one(frame: pd.DataFrame, column: str, **filters) -> float:
    """The single value at `column` under `filters`, or raise naming what was asked for.

    Loud rather than NaN: a landing page printing `nan` where a result belongs is the
    failure mode `model_cards.text()` exists for, and here there is no honest fallback —
    the sentence has nothing to say without its row.
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


def scoreless_weeks(weekly: pd.DataFrame) -> tuple[float, float]:
    """Observed and simulated share of player-weeks that score nothing at all.

    Both halves come off the same row, so the pair is a comparison the artifact already
    made rather than two lookups this page divides.
    """
    return (_one(weekly, "zero_share", period_type=WEEKLY_PERIOD, split=WEEKLY_SPLIT),
            _one(weekly, "predicted_zero_share", period_type=WEEKLY_PERIOD,
                 split=WEEKLY_SPLIT))


def advance(shipped: pd.DataFrame, tournament: str = HEADLINE_TIER) -> dict[str, float]:
    """The shipped arm's realized Round-1 advance rate, its lift, and the field null.

    The null is *derived* rather than read: the artifact carries the rate and the lift over
    a symmetric field, and their difference is that field's own rate. Deriving it keeps the
    three numbers in the sentence arithmetically consistent with each other — a rounded 1/6
    typed in beside them would not be.
    """
    rate = _one(shipped, "realized_p_advance", tournament=tournament)
    lift = _one(shipped, "realized_lift", tournament=tournament)
    return {"rate": rate, "lift": lift, "null": rate - lift,
            "seasons": _one(shipped, "realized_seasons", tournament=tournament)}


# ── Readings ──────────────────────────────────────────────────────────────────

class Reading(NamedTuple):
    """One sentence of the paper, with its figures already read out of a frame.

    A sentence rather than a tile's label / value / delta since 2026-08-10: five tiles were
    five numbers and no argument, which is the "numbers-vomit" the rewrite was asked for.
    The `Spec` around it is unchanged, so what a figure has to go through to reach the page
    did not move — only what it looks like when it gets there.
    """

    text: str


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
    ten CSVs has not been generated yet.
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


# ── The four sections ─────────────────────────────────────────────────────────
#
# `README.md`'s structure, at a landing page's length: Introduction, Methods, Results,
# Discussion, three sentences each. Every sentence that states a result is a `Spec`; every
# `str` is typed prose that says what the project *does* and carries no digit.

def _minutes(f: dict[str, pd.DataFrame]) -> Reading:
    return Reading(
        f"A player's own minutes decide **{minutes_share(f['budget']):.1%}** of the "
        "within-player-season variance in `dk_pts`, and minutes are exactly what nobody "
        "knows in advance — so what is worth having is not a sharper guess at an average "
        "but an honest joint distribution over a whole roster.")


def _sampler(f: dict[str, pd.DataFrame]) -> Reading:
    # "convergence and effective-sample-size bars" rather than the R̂ every model page
    # names, for two reasons found by rendering this one: the combining circumflex sits
    # badly on the `R` at body size in this font, and a reader who arrives at a URL with no
    # context does not know what R-hat is. The model pages keep the symbol, because a
    # reader who has navigated to one has asked for that level.
    return Reading(
        f"All **{heads(f['cards'])}** persisted fits clear their convergence and "
        f"effective-sample-size bars, with **{divergences(f['cards']):,}** divergences "
        "between them.")


def _season_total(f: dict[str, pd.DataFrame]) -> Reading:
    shipped = season_total_mae(f["season_total"], SHIPPED_TREATMENT)
    naive = season_total_mae(f["season_total"], NAIVE_TREATMENT)
    return Reading(
        "Availability is the largest measured win: mean absolute error on a player's "
        f"season `dk_pts` total is **{shipped:,.1f}** on the validation seasons, against "
        f"**{naive:,.1f}** with everyone assumed to play every game.")


def _floor(f: dict[str, pd.DataFrame]) -> Reading:
    low, high = floor_band(f["components"])
    return Reading(
        "The rate side is nearly saturated before anything is fitted — last season's "
        "per-36 rate times this game's minutes scores validation R² "
        f"**{low:.2f}–{high:.2f}** across the count heads, the floor every fitted head is "
        "quoted against.")


def _advance(f: dict[str, pd.DataFrame]) -> Reading:
    got = advance(f["shipped"])
    return Reading(
        "Rosters drafted against a market field and replayed on realized box scores clear "
        f"the {pretty_tournament(HEADLINE_TIER)}'s first elimination round "
        f"**{got['rate']:.1%}** of the time, against **{got['null']:.1%}** for a symmetric "
        f"field, over **{got['seasons']:.0f}** validation seasons.")


def _oracles(f: dict[str, pd.DataFrame]) -> Reading:
    gp = season_total_mae(f["season_total"], ORACLE_GP_TREATMENT)
    rate = season_total_mae(f["season_total"], ORACLE_RATE_TREATMENT)
    return Reading(
        "Two oracles say where the headroom is: perfect knowledge of games played would "
        f"leave **{gp:,.1f}** dk_pts of season-total error and a perfect per-minute rate "
        f"would leave **{rate:,.1f}**, so availability is worth more than better rate "
        "features.")


def _scoreless(f: dict[str, pd.DataFrame]) -> Reading:
    observed, simulated = scoreless_weeks(f["weekly"])
    return Reading(
        "Shape is what the contest reads and a season total averages it away — about a "
        f"fifth of player-weeks score nothing at all, **{observed:.1%}** observed against "
        f"**{simulated:.1%}** simulated.")


class Section(NamedTuple):
    """One section of the paper: a heading, and an ordered body of two kinds of fragment.

    A `str` is typed prose and may not carry a digit; a `Spec` is a lookup into an
    artifact. See the module docstring — that split is bound 2 restated at the sentence,
    which is where it has to be stated once the page is prose rather than tiles.
    """

    heading: str
    body: tuple  # of str | Spec


SECTIONS: tuple[Section, ...] = (
    Section("Introduction", (
        "DraftKings best ball is a snake draft of sixteen players, frozen before opening "
        "night: the best seven by roster slot score each week, and four elimination "
        "rounds cut the field to one.",
        "Nothing can be traded, started or benched afterwards, and all anyone knows on "
        "draft night is last season's box scores plus this season's rosters and schedule.",
        Spec(("budget",), _minutes),
    )),
    Section("Methods", (
        "That distribution is a chain of Bayesian models fitted in Stan — availability, "
        "then minutes given availability, then each box-score component given minutes — "
        "and `dk_pts` is never predicted directly, because it is a deterministic function "
        "of those components.",
        "The heads are fitted separately and that is exact rather than convenient: their "
        "parameter blocks are disjoint, so the joint posterior factorizes and separate "
        "fits recover what one joint model would.",
        Spec(("cards",), _sampler),
    )),
    Section("Results", (
        Spec(("season_total",), _season_total),
        Spec(("components",), _floor),
        Spec(("shipped",), _advance),
    )),
    Section("Discussion", (
        Spec(("season_total",), _oracles),
        Spec(("weekly",), _scoreless),
        "The frame is a harder limit than the fit: the draft happens before a single game "
        "is played, and the readout above is a backtest rather than a track record.",
    )),
)

#: Every lookup the four sections make, for the tests that hold each `Spec` to declaring
#: the sources it reads. Derived rather than listed, so a sentence added to a section
#: cannot escape the check by not being written down twice.
PROSE_SPECS: tuple[Spec, ...] = tuple(
    fragment for section in SECTIONS for fragment in section.body
    if isinstance(fragment, Spec))


class Paragraph(NamedTuple):
    """One rendered section: its heading, and its surviving sentences joined up."""

    heading: str
    text: str


def paragraphs(frames: dict[str, pd.DataFrame]) -> list[Paragraph]:
    """The sections that have something to say, in declaration order.

    A section keeps its typed sentences whatever is on disk and loses only the readings
    whose artifacts are missing, so a half-built repo reads as a shorter paper rather than
    a broken one. A section left with nothing at all — `Results` is every-sentence-a-lookup
    — drops its heading too, since a heading over an empty paragraph is worse than an
    absent section.
    """
    out = []
    for section in SECTIONS:
        parts = [fragment if isinstance(fragment, str) else fragment.build(frames).text
                 for fragment in section.body
                 if isinstance(fragment, str)
                 or all(key in frames for key in fragment.needs)]
        if parts:
            out.append(Paragraph(section.heading, " ".join(parts)))
    return out


def stage_readings(frames: dict[str, pd.DataFrame]) -> list[Stage]:
    return _resolve(STAGE_SPECS, frames)


# ── The route out ─────────────────────────────────────────────────────────────
#
# The page's last job, and the reason it was built last: a reader who arrives at the URL
# with no context needs somewhere to go, and there was nowhere to send them until the nine
# pages existed. Keyed by `app.VIEWS[i].url_path` so a retitled page keeps its blurb, and
# **the blurbs state no results** — that is the charter's second bound again, and a route
# label is exactly where a stray headline would try to creep back in.

class Route(NamedTuple):
    url_path: str
    blurb: str


#: One line each, for the same reason the stage notes are one line: nine blurbs at three
#: lines apiece is 200 px of a page that is still not allowed to run long. Each says what
#: its page holds, in the page's own terms, and none of them states a result.
ROUTES: tuple[Route, ...] = (
    Route("fingerprints", "Player style on the league's axes."),
    Route("availability", "Games played, and the tenure model."),
    Route("minutes", "Two units, two opposite verdicts."),
    Route("components", "Eleven heads against a no-fit floor."),
    Route("game-length", "Overtime — whether, and how deep."),
    Route("inputs", "The market and the fixed constants."),
    Route("weekly", "Scored at the unit a lineup is set."),
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
