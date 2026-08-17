"""Page 7's pure layer — everything the simulator consumes that is not a coefficient.

Four families of input, and the only thing they have in common is the thing that makes them
a page: **none of them is a fitted coefficient, and every one of them is load-bearing.**

1. **ADP** — the market's forecast of the same target the model predicts, and therefore the
   most leakage-prone input in the repo. Its discipline is three dates per row, and what
   that discipline *costs* is the number worth drawing: the panel holds nine seasons and
   only five of them are point-in-time legal.
2. **The capture programs** — four archives, three of them perishable. This is the only
   block on the dashboard that is an operational alarm rather than a result: a missing day
   is not a worse measurement, it is a day that no longer exists.
3. **The calibrated numbers, split by whether the draw actually reads them.** ⚠️ This block
   said "the four calibrated simulator inputs" until 2026-08-16 and that framing was wrong in
   three ways at once: **two of the four are diagnostics** the draws are *checked against*
   rather than given (the game-level minutes dispersion, by an explicit decision that the
   composition's fitted role-graded rho owns it; and the ten-game block inflation, which is
   produced by the season-constant frailties rather than imposed), and the injected
   per-(player, season) sigma — which *is* consumed, on every minutes draw the simulator
   makes — was **not on the list at all**. Five rows now, in two groups, each saying where it
   enters the draw or what it is checked against. The three that carry a `fit_window` still
   show it rather than hiding it; the sigma does not, because it is a config constant rather
   than an artifact measured per window. `docs/sim-inputs-plan.md`.
4. **The availability layout** — where a player's missed games fall, which the availability
   head cannot say because games played is invariant to the arrangement. The one input here
   that is not a number, and the one whose diagnostic is at the scoring period rather than
   the season. **Only the shipped arm and the observed target are read**; see
   `SHIPPED_LAYOUT`.

**No Streamlit import**, so every rule here is exercised as a plain function in
`tests/test_dashboard.py` rather than through a rendered page, and no function reads a file:
the view opens the artifacts and hands frames in.

## Why the window is on screen rather than chosen for the reader

`residual_correlation.to_matrix` defaults to `train_val` and `src/sim/season.py` overrides it
to `train`, and **both are right**. `train_val` is the window that is never *wrong* — it
excludes the test seasons and nothing else — so it is the safe default for an unthinking
caller. But the shipped simulation is scored against a backtest on 2022-23 and 2023-24,
which are *inside* `train_val`, so a run at that window would calibrate itself on the seasons
it is about to be marked on. Which window to consume is decided by what the number will be
scored against, not by which is widest, and the three-window panel exists because the
differences are small enough that the leak would never have announced itself.
"""

from typing import NamedTuple

import pandas as pd

# ── The artifacts, by the target that writes them ─────────────────────────────

PANEL_FILE = "adp_panel.parquet"
PROFILE_FILE = "adp_profile.csv"
MATCH_AUDIT_FILE = "adp_match_audit.csv"
CALENDAR_FILE = "capture_calendar.csv"
PROGRAMS_FILE = "capture_programs.csv"
RESIDUAL_FILE = "residual_correlation.csv"
SERIAL_FILE = "serial_correlation.csv"
BONUS_FILE = "bonus_calibration.csv"
DISPERSION_FILE = "stan_minutes_dispersion.csv"
# The injected sigma's own home. Read for the `shipped_configuration` row, which is the one
# place an artifact records what `sim.minutes.player_season_sigma{,_by_role}` is set to —
# this package may not import `src/`, so config is not readable from here and an artifact is
# the only honest source. That row reached the file on 2026-08-16; before then it was
# computed, printed and dropped.
MIN_UNIF_FILE = "minutes_unification.csv"
LAYOUT_FILE = "availability_exchangeability.csv"
LAYOUT_PROFILE_FILE = "availability_clustering.csv"

MAKE_ADP = "make adp"
MAKE_CALENDAR = "make capture-calendar"
MAKE_RESIDUAL = "make residual-correlation"
MAKE_SERIAL = "make serial-correlation"
MAKE_BONUS = "make component-targets"
MAKE_DISPERSION = "make stan-minutes"
MAKE_MIN_UNIF = "make minutes-unification"
MAKE_LAYOUT = "make availability-exchangeability"

#: The availability layout `sim.availability.layout` ships — `src/sim/season.py::LAYOUT_ARMS`.
#: The ladder artifact carries the arms that were compared to select it; **this page draws
#: only this one and the observed target**, because a dashboard shows what the pipeline does
#: and the rejected arms are a plan-doc argument (`docs/availability-window-plan.md` §13).
SHIPPED_LAYOUT = "tenure_merge"
#: The realized played/missed vectors — what the layout is trying to reproduce. Not a rival
#: arm, which is why it is drawn as a hollow reference wherever it appears.
OBSERVED_LAYOUT = "observed"

#: The three fit windows every calibrated input is measured at, widest first. Pinned here
#: rather than read off an artifact so a window silently disappearing is a failing test.
FIT_WINDOWS = ("full", "train_val", "train")
#: What an unthinking caller gets — never wrong, sometimes wider than a caller may read.
SAFE_WINDOW = "train_val"
#: What `make simulate-season` actually reads, because its backtest scores the two seasons
#: `train_val` contains. `src/sim/season.py::FIT_WINDOW`.
SIM_WINDOW = "train"

MINUTES_CONDITIONED = "minutes_conditioned"

# ── ADP · the point-in-time panel ─────────────────────────────────────────────

#: The panel's dating columns, which are the point of the panel. Three dates per row:
#: when the board was observed, when the season it describes began, and the gap.
_SNAPSHOT_KEYS = ["snapshot_source", "season", "as_of_date", "season_start_date",
                  "snapshot_lag_days", "captured_before_season_start"]


def snapshots(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per captured board: source, season, when it was observed, and its lag.

    A board is a *snapshot*, not a row — the panel is long over players and sources, so
    counting rows counts pool sizes rather than captures.
    """
    if panel.empty:
        return pd.DataFrame(columns=["source", "season", "as_of_date", "lag_days",
                                     "legal", "rows"])
    grouped = (panel.groupby(_SNAPSHOT_KEYS, dropna=False).size()
               .reset_index(name="rows"))
    out = pd.DataFrame({
        "source": grouped["snapshot_source"],
        "season": grouped["season"],
        "as_of_date": grouped["as_of_date"],
        "season_start_date": grouped["season_start_date"],
        "lag_days": grouped["snapshot_lag_days"],
        "legal": grouped["captured_before_season_start"].astype(bool),
        "rows": grouped["rows"],
    })
    return out.sort_values(["season", "as_of_date", "source"]).reset_index(drop=True)


def datable(snaps: pd.DataFrame) -> pd.DataFrame:
    """Snapshots whose lag is known, i.e. everything but a board for an unplayed season.

    The 2026-27 board has no `season_start_date` because that season has no game logs yet.
    It is legal — a board for a season that has not begun cannot postdate it — but it has no
    position on a days-from-tip-off axis, so it is kept out of the figure and named in the
    table rather than being drawn at a zero it was not measured at.
    """
    return snaps[snaps["lag_days"].notna()].reset_index(drop=True)


def undated(snaps: pd.DataFrame) -> pd.DataFrame:
    """The other half of `datable` — captures for a season that has not been played."""
    return snaps[snaps["lag_days"].isna()].reset_index(drop=True)


def season_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    """Per season: the captures held, and how many survive `adp.training_rows`.

    The verdict column is `legal`, and it is an **any**, not an **all**: one board observed
    at or before the opener makes the season usable, however many late ones sit beside it.
    """
    snaps = snapshots(panel)
    if snaps.empty:
        return pd.DataFrame(columns=["season", "sources", "n_snapshots", "n_legal",
                                     "legal", "earliest_legal", "rows", "legal_rows"])
    rows = []
    for season, part in snaps.groupby("season"):
        legal = part[part["legal"]]
        rows.append({
            "season": season,
            "sources": ", ".join(sorted(part["source"].unique())),
            "n_snapshots": int(len(part)),
            "n_legal": int(len(legal)),
            "legal": bool(len(legal)),
            "earliest_legal": (str(legal["as_of_date"].min()) if len(legal) else ""),
            "rows": int(part["rows"].sum()),
            "legal_rows": int(legal["rows"].sum()),
        })
    return pd.DataFrame(rows).sort_values("season").reset_index(drop=True)


def point_in_time_cost(panel: pd.DataFrame) -> dict:
    """What the discipline costs, as the four numbers the block is built around.

    A season with no board dated at or before its opener is not a season with a worse ADP
    input — it is a season with **none**, because a frozen board observed mid-season still
    carries a preseason *value* and this project counts the qualifying observation rather
    than the qualifying value.
    """
    coverage = season_coverage(panel)
    if coverage.empty:
        return {"seasons_held": 0, "seasons_legal": 0, "seasons_lost": 0, "lost": (),
                "rows": 0, "legal_rows": 0, "row_share": 0.0}
    lost = tuple(coverage.loc[~coverage["legal"], "season"])
    rows = int(coverage["rows"].sum())
    legal_rows = int(coverage["legal_rows"].sum())
    return {
        "seasons_held": int(len(coverage)),
        "seasons_legal": int(coverage["legal"].sum()),
        "seasons_lost": len(lost),
        "lost": lost,
        "rows": rows,
        "legal_rows": legal_rows,
        "row_share": (legal_rows / rows) if rows else 0.0,
    }


# ── ADP · the profile, the ladder and the audit ───────────────────────────────

#: The ladder arm `adp_profile.fit_transfer` actually ships: a monotone map from consensus
#: ADP onto DK's scale, with no position term. The C/F/G offset scores better and is not
#: shipped, because the offset needs a position label DK and NBA.com disagree about on
#: 9.02% of players — so the page flags the shipped arm rather than the best one.
SHIPPED_LADDER_ARM = "+ monotone (isotonic) rescale"


def _section(profile: pd.DataFrame, section: str) -> pd.DataFrame:
    return profile[profile["section"] == section]


def agreement(profile: pd.DataFrame) -> dict:
    """The DK↔consensus agreement block, as `{metric: value}`."""
    part = _section(profile, "agreement")
    return {str(r.metric): float(r.value) for r in part.itertuples(index=False)}


def ladder(profile: pd.DataFrame) -> pd.DataFrame:
    """The recalibration ladder, best last, with the shipped arm flagged.

    Ordered as it was measured rather than by score: it is a ladder of *corrections*, each
    added to the one above it, so sorting it by value would make it read as a menu.
    """
    part = _section(profile, "ladder")
    if part.empty:
        return pd.DataFrame(columns=["arm", "mean_abs_rank_gap", "shipped"])
    return pd.DataFrame({
        "arm": part["metric"].astype(str).to_numpy(),
        "mean_abs_rank_gap": part["value"].astype(float).to_numpy(),
        "shipped": (part["metric"].astype(str) == SHIPPED_LADDER_ARM).to_numpy(),
    })


def _keyed_section(profile: pd.DataFrame, section: str, index: str) -> pd.DataFrame:
    """A `key:metric` section pivoted so each key is a row and each metric a column."""
    part = _section(profile, section)
    if part.empty:
        return pd.DataFrame(columns=[index])
    keys = part["metric"].astype(str).str.split(":", n=1, expand=True)
    wide = (pd.DataFrame({index: keys[0], "metric": keys[1],
                          "value": part["value"].astype(float)})
            .pivot(index=index, columns="metric", values="value").reset_index())
    wide.columns.name = None
    return wide


def position_bias(profile: pd.DataFrame) -> pd.DataFrame:
    """Mean rank gap by position — the reason a raw consensus is never used directly."""
    return _keyed_section(profile, "position_bias", "position")


def tier_gap(profile: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute rank gap by draft-round tier, early rounds first."""
    out = _keyed_section(profile, "tier_gap", "tier")
    order = ["R1-2", "R3-4", "R5-8", "R9+"]
    if out.empty or "tier" not in out.columns:
        return out
    out["_order"] = [order.index(t) if t in order else len(order) for t in out["tier"]]
    return out.sort_values("_order").drop(columns="_order").reset_index(drop=True)


def match_summary(audit: pd.DataFrame) -> dict:
    """The cascade's counts beside the rejected rule's, as `{rule: {metric: value}}`.

    Both are kept because the *comparison* is the finding: the rejected surname+initial rule
    scores a **better** unmatched rate than the cascade while fabricating 23 matches, so an
    unmatched rate is not a sufficient check on a join. It is monotonically improving in the
    error it is supposed to detect.
    """
    part = audit[audit["section"] == "summary"]
    out: dict[str, dict[str, float]] = {}
    for row in part.itertuples(index=False):
        out.setdefault(str(row.rule), {})[str(row.metric)] = float(row.value)
    return out


def fuzzy_matches(audit: pd.DataFrame) -> pd.DataFrame:
    """Every surviving non-exact match, with both names — small enough to read.

    A fuzzy tier too large to eyeball is a fuzzy tier too large to trust, so this is the
    table rather than a count of it.
    """
    part = audit[audit["section"] == "fuzzy_match"]
    return pd.DataFrame({
        "source": part["source"], "rule": part["rule"],
        "board name": part["board_name"], "matched to": part["matched_name"],
        "season": part["board_season"], "seasons apart": part["seasons_apart"],
    }).reset_index(drop=True)


def ablation_false_matches(audit: pd.DataFrame) -> pd.DataFrame:
    """What the rejected rule invents, kept running forever as an ablation."""
    part = audit[(audit["section"] == "ablation_match")
                 & (audit["metric"] == "agrees_with_cascade")
                 & (audit["value"].astype(float) == 0.0)]
    out = pd.DataFrame({
        "source": part["source"],
        "board name": part["board_name"], "matched to": part["matched_name"],
        "season": part["board_season"], "seasons apart": part["seasons_apart"],
    })
    return out.sort_values("seasons apart", ascending=False).reset_index(drop=True)


# ── The capture programs ──────────────────────────────────────────────────────

class ProgramNote(NamedTuple):
    """What a program is and what happens to a day it misses.

    Interpretation, not measurement — so it lives here rather than in the artifact, and
    `test_every_capture_program_has_a_note` holds it against `capture_programs.csv` in both
    directions. Every *number* on the block comes from the artifact.
    """

    label: str
    what: str
    stake: str


PROGRAM_NOTES: dict[str, ProgramNote] = {
    "injury_reports": ProgramNote(
        label="NBA injury-report PDFs",
        what="The league-mandated 5:00 PM ET deadline report, one PDF a day.",
        stake="Ages out of the CDN after ~7 months and 403s forever after. A gap inside "
              "the window is a backlog item; past it, it is gone."),
    "espn_injuries": ProgramNote(
        label="ESPN injury feed",
        what="A current-status feed with no dated archive behind it.",
        stake="Describes today and keeps no history, so the archive can only grow "
              "forward. **Every missed day is permanent** — there is nothing to refetch."),
    "adp_draftkings": ProgramNote(
        label="DraftKings draft board",
        what="The pre-draft rankings CSV, downloaded by hand from the contest lobby.",
        stake="Login-gated, zero Wayback presence, and live only while contests are "
              "(~October). It is the only anchor the consensus→DK map has."),
    "adp_fantasypros": ProgramNote(
        label="FantasyPros consensus ADP",
        what="The consensus board, live plus a Wayback backfill.",
        stake="The one source of the four that is **not** on a deadline: Wayback holds "
              "the history, so a snapshot not taken today can be taken later."),
}

RECOVERY_LABELS = {
    "window": "recoverable until it ages out",
    "never": "permanently lost",
    "archive": "recoverable from a third-party archive",
}

CAPTURED = "captured"
NOTHING_TO_CAPTURE = "nothing_to_capture"
MISSED = "missed"
#: Draw order and colour order. `captured` and `missed` are the two that take a palette
#: slot; `nothing_to_capture` is neutral because it is **not a failure** — the offseason and
#: the All-Star break land in it, and colouring it as a gap would cry wolf on 47 days out of
#: 211. Recoverability is not encoded in the cell at all: it is a property of the *program*,
#: so it rides on the row label, which is a second channel and satisfies the relief rule
#: without spending a third colour on a pair the palette cannot separate.
STATE_ORDER = (CAPTURED, NOTHING_TO_CAPTURE, MISSED)
STATE_LABELS = {CAPTURED: "captured", NOTHING_TO_CAPTURE: "nothing to capture",
                MISSED: "missed"}


def program_label(program: str) -> str:
    note = PROGRAM_NOTES.get(program)
    return note.label if note else program


def program_table(programs: pd.DataFrame) -> pd.DataFrame:
    """The program table for display: policy, window and counts, most perishable first."""
    if programs.empty:
        return programs
    out = pd.DataFrame({
        "Program": [program_label(p) for p in programs["program"]],
        "Cadence": programs["cadence"],
        "A missed day is": [RECOVERY_LABELS.get(r, r) for r in programs["recovery"]],
        "Window": [f"{a} → {b}" for a, b in zip(programs["window_start"],
                                                programs["window_end"])],
        "Captured": programs["n_captured"],
        "Nothing to capture": programs["n_nothing_to_capture"],
        "Missed": programs["n_missed"],
        "Recoverable": programs["n_recoverable"],
        "Lost": programs["n_lost"],
    })
    return (out.assign(_risk=-programs["n_lost"].to_numpy())
            .sort_values(["_risk", "Program"]).drop(columns="_risk")
            .reset_index(drop=True))


def calendar_alarm(programs: pd.DataFrame) -> dict:
    """The block's headline: how many days are still fetchable, and how many are not."""
    if programs.empty:
        return {"recoverable": 0, "lost": 0, "captured": 0, "worst": ""}
    lost = programs.sort_values("n_lost", ascending=False)
    worst = lost.iloc[0]
    return {
        "recoverable": int(programs["n_recoverable"].sum()),
        "lost": int(programs["n_lost"].sum()),
        "captured": int(programs["n_captured"].sum()),
        "worst": program_label(str(worst["program"])) if int(worst["n_lost"]) else "",
    }


def calendar_grid(calendar: pd.DataFrame, programs: pd.DataFrame,
                  days: int = 120) -> pd.DataFrame:
    """The last `days` of every program's row, as one long frame the heatmap pivots.

    **A shared date axis over the whole archive is the wrong picture**, and that is a
    measurement rather than a preference: FantasyPros reaches back to 2014-10-15, so a
    common axis over every capture would compress the two daily programs — the only two that
    can *have* a gap — into a few pixels at the right-hand edge. The window therefore ends at
    the latest day any program covers and reaches back a fixed span, and the ADP boards
    appear on it as the sparse events they are.

    Days a program has no row for are dropped rather than filled: an `event` program has no
    schedule to have missed, so an empty stretch of its row means "nothing was due", not
    "nothing was captured".
    """
    if calendar.empty or programs.empty:
        return pd.DataFrame(columns=["program", "row", "label", "capture_date", "state",
                                     "state_label", "code", "recoverable", "records"])
    end = pd.to_datetime(programs["window_end"].replace("", pd.NA).dropna()).max()
    if pd.isna(end):
        end = pd.to_datetime(calendar["capture_date"]).max()
    start = end - pd.Timedelta(days=max(days, 1) - 1)
    dates = pd.to_datetime(calendar["capture_date"])
    part = calendar[(dates >= start) & (dates <= end)].copy()

    # The row index is carried on the frame rather than recovered from the label in the
    # figure: `row_labels` decorates a program's name with its recovery policy, so matching
    # a cell to a row by its label means matching a prefix, and two programs whose names
    # share one would silently land on the same row.
    order = {p: i for i, p in enumerate(programs["program"])}
    part["row"] = [order.get(p, len(order)) for p in part["program"]]
    part["label"] = [program_label(p) for p in part["program"]]
    part["state_label"] = [STATE_LABELS.get(s, s) for s in part["state"]]
    part["code"] = [STATE_ORDER.index(s) if s in STATE_ORDER else -1
                    for s in part["state"]]
    return (part.sort_values(["row", "capture_date"])
            [["program", "row", "label", "capture_date", "state", "state_label", "code",
              "recoverable", "records"]].reset_index(drop=True))


def row_labels(programs: pd.DataFrame) -> list[str]:
    """One label per calendar row, carrying the recoverability the cells do not.

    The cells encode three states in two colours and a neutral; whether a missed cell can
    still be fetched is the difference between a chore and an incident, and it belongs on
    the axis because it is constant along the row.

    Two lines rather than one, which a rendered figure decided: on one line the longest of
    these is 62 characters and plotly gives a tick label whatever width it asks for, so the
    axis took a third of the plot away from the calendar it is labelling.
    """
    return [f"{program_label(p)}<br>{RECOVERY_LABELS.get(r, r)}"
            for p, r in zip(programs["program"], programs["recovery"])]


def gap_table(calendar: pd.DataFrame) -> pd.DataFrame:
    """Every missed day, newest first, with what can still be done about it."""
    part = calendar[calendar["state"] == MISSED]
    if part.empty:
        return pd.DataFrame(columns=["Program", "Day", "Still fetchable"])
    out = pd.DataFrame({
        "Program": [program_label(p) for p in part["program"]],
        "Day": part["capture_date"].astype(str),
        "Still fetchable": part["recoverable"].astype(bool),
    })
    return out.sort_values(["Day", "Program"], ascending=[False, True]).reset_index(
        drop=True)


# ── The calibrated numbers: consumed by the draw, or diagnostic ───────────────

#: The two roles, and the distinction the block exists to make. CONSUMED means the draw
#: reads the number — change it and the tensor changes. DIAGNOSTIC means the number is
#: measured and reported and the draw never sees it; it is a target the draws are scored
#: against. Both are "calibrated" and only one is an input, which is precisely what the old
#: "four calibrated simulator inputs" framing lost.
CONSUMED, DIAGNOSTIC = "consumed", "diagnostic"


class Calibrated(NamedTuple):
    """One calibrated number, and whether the draw actually reads it."""

    key: str
    label: str
    artifact: str
    target: str
    unit: str
    what: str
    fmt: str
    role: str
    #: Where it enters the draw (CONSUMED) or what it is checked against (DIAGNOSTIC).
    where: str
    #: Whether it is measured per `fit_window`. False for the injected sigma, which is a
    #: config constant — so it is shown in the inventory and left out of the window panel
    #: rather than given three identical rows that would imply it had been measured thrice.
    windowed: bool = True


CALIBRATED: tuple[Calibrated, ...] = (
    Calibrated(
        key="copula", label="Residual copula", artifact=RESIDUAL_FILE,
        target=MAKE_RESIDUAL, unit="mean r, count block",
        what="Cross-component dependence left after the shared minutes draw, imposed as a "
             "Gaussian copula on the seven count heads' per-game frailties.",
        fmt="{:+.4f}", role=CONSUMED,
        where="`sim/season.count_copula` inverts it to the frailty scale, then correlates "
              "the per-game lognormal frailties."),
    Calibrated(
        key="block_inflation", label="Block variance inflation", artifact=SERIAL_FILE,
        target=MAKE_SERIAL, unit="× binomial, ten-game blocks",
        what="How much more a ten-game block of minutes varies than independent draws "
             "would. **A diagnostic, never imposed** — the serial dependence the draw has "
             "is *produced* by the season-constant per-(player, season) frailties, which "
             "carry most of it, rather than read from this number.",
        fmt="{:.4f}×", role=DIAGNOSTIC,
        where="Gate A reports the drawn value against this target; nothing reads it."),
    Calibrated(
        key="bonus_overdispersion", label="Bonus overdispersion", artifact=BONUS_FILE,
        target=MAKE_BONUS, unit="frailty variance, player-game",
        what="The variance of the shared per-game frailty that makes the double-double "
             "threshold fire at the realized rate. Unit-specific: ~0.10 at the season "
             "unit, ~0.025 at the player-game unit the simulator draws at.",
        fmt="{:.4f}", role=CONSUMED,
        where="Two roles, one mechanism: it is the variance of the count heads' per-game "
              "frailties AND the scale the copula above is inverted at."),
    Calibrated(
        key="minutes_dispersion", label="Game-level minutes dispersion",
        artifact=DISPERSION_FILE, target=MAKE_DISPERSION, unit="× binomial, per game",
        what="Within-player-season per-game overdispersion of minutes against the "
             "player's own mean. A **diagnostic** the composition's draws are checked "
             "against, not an input to them — superseded as an input by decision, because "
             "the composition fits its own role-graded rho and only one of the two can "
             "govern a draw.",
        fmt="{:.3f}×", role=DIAGNOSTIC,
        where="Gate A reports the drawn value against this target; the composition's own "
              "fitted rho is what the draw uses."),
    Calibrated(
        key="player_season_sigma", label="Injected per-(player, season) sigma",
        artifact=MIN_UNIF_FILE, target=MAKE_MIN_UNIF, unit="logit-scale sd, per unit",
        what="The season-level spread the composition cannot manufacture from draws that "
             "are iid across games. Graded by role since 2026-08-16 — a constant "
             "logit-scale sigma left fringe player-seasons 1.73x under-dispersed while "
             "stars were over-dispersed at 0.86x.",
        fmt="{:.3f}", role=CONSUMED, windowed=False,
        where="`minutes_unification.rehydrate_composition` puts `sigma * z` on the linear "
              "predictor once per (player, season) per draw, shared across that player's "
              "games, and re-runs the head's own allocation."),
)

CALIBRATED_BY_KEY = {c.key: c for c in CALIBRATED}

#: The subset the window panel covers. Not `CALIBRATED` — the injected sigma is a config
#: constant, so it has no per-window reading and three identical rows would claim it had
#: been measured three times.
WINDOWED = tuple(c for c in CALIBRATED if c.windowed)


def copula_components(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                      basis: str = MINUTES_CONDITIONED) -> list[str]:
    """The matrix's component order, taken from the artifact rather than retyped.

    First-appearance order, which is the order `residual_correlation.COMPONENTS` wrote:
    the seven counts, then the four conversions. A test pins that split, because the count
    block is the only part of the matrix the simulator imposes and reading it off the wrong
    corner would be silent.
    """
    part = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)]
    return list(dict.fromkeys(part["component_a"]))


def copula_square(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                  basis: str = MINUTES_CONDITIONED,
                  mask_diagonal: bool = False) -> pd.DataFrame:
    """The long form pivoted back to a square, in the artifact's own component order.

    `mask_diagonal` blanks the self-correlations, which carry no information and cost the
    whole colour scale: they are 1.0 by construction while the largest real cell is +0.133,
    so a scale that accommodates them renders every real cell as the midpoint. The figure
    masks; the table twin does not, because there the diagonal is a free check that the
    pivot landed the way round it was meant to.
    """
    part = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)]
    if part.empty:
        return pd.DataFrame()
    order = copula_components(residual, window, basis)
    square = part.pivot(index="component_a", columns="component_b", values="r")
    square = square.reindex(index=order, columns=order)
    if mask_diagonal:
        square = square.copy()
        for name in order:
            square.loc[name, name] = float("nan")
    return square


def copula_limit(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                 basis: str = MINUTES_CONDITIONED, step: float = 0.05) -> float:
    """A symmetric colour-scale half-range covering every off-diagonal cell.

    Rounded up to a `step` so the scale is a readable number rather than the largest cell to
    six places, and symmetric so zero stays at the neutral midpoint — a diverging scale
    whose midpoint drifts off zero turns a sign into a shade of the wrong colour.
    """
    cells = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)
                     & (residual["component_a"] != residual["component_b"])]
    if cells.empty:
        return step
    peak = float(cells["r"].abs().max())
    return max(step, step * -(-peak // step))


def copula_kinds(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                 basis: str = MINUTES_CONDITIONED) -> dict[str, str]:
    """`{component: count | conversion}`, so a block can be named rather than sliced."""
    part = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)]
    return dict(zip(part["component_a"], part["kind_a"]))


def copula_cells(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                 basis: str = MINUTES_CONDITIONED, top: int = 6) -> pd.DataFrame:
    """The largest off-diagonal cells, each pair once."""
    part = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)
                    & (residual["component_a"] != residual["component_b"])].copy()
    if part.empty:
        return pd.DataFrame(columns=["pair", "r", "kind", "n_games"])
    part["pair"] = [" · ".join(sorted((a, b))) for a, b in
                    zip(part["component_a"], part["component_b"])]
    part["kind"] = [a if a == b else "mixed"
                    for a, b in zip(part["kind_a"], part["kind_b"])]
    part = part.drop_duplicates("pair")
    part["_abs"] = part["r"].abs()
    out = part.nlargest(top, "_abs")
    return pd.DataFrame({"pair": out["pair"], "r": out["r"], "kind": out["kind"],
                         "n_games": out["n_games"]}).reset_index(drop=True)


def copula_mean(residual: pd.DataFrame, window: str = SAFE_WINDOW,
                basis: str = MINUTES_CONDITIONED, kind: str = "count") -> float:
    """Mean off-diagonal `r` inside one block — the copula's one summary number."""
    part = residual[(residual["fit_window"] == window) & (residual["basis"] == basis)
                    & (residual["kind_a"] == kind) & (residual["kind_b"] == kind)
                    & (residual["component_a"] != residual["component_b"])]
    return float(part["r"].mean()) if len(part) else float("nan")


def block_inflation(serial: pd.DataFrame, window: str = SAFE_WINDOW,
                    detrended: bool = False) -> pd.DataFrame:
    """Ten-game block variance inflation per component, largest first.

    `min_detrended` is dropped by default: it is the same series with the season trend
    removed, so drawing both puts one component on the chart twice.
    """
    part = serial[serial["fit_window"] == window]
    if not detrended:
        part = part[~part["component"].astype(str).str.endswith("_detrended")]
    out = pd.DataFrame({
        "component": part["component"].astype(str),
        "kind": part["kind"].astype(str),
        "block_inflation": part["block_inflation"].astype(float),
        "lag1": part["lag1"].astype(float),
        "n_pairs": part["n_pairs"],
    })
    return out.sort_values("block_inflation", ascending=False).reset_index(drop=True)


def bonus_rows(bonus: pd.DataFrame, window: str = SAFE_WINDOW) -> pd.DataFrame:
    """The overdispersion that zeroes the bias, per unit, beside the shipped constant.

    Two units and two different answers, which is the point: the same threshold calibrated
    at the season unit reads ~4× the value it needs at the player-game unit, because a
    season-level frailty is diluted across ~70 games and a per-game one is not.
    """
    fitted = bonus[(bonus["fit_window"] == window) & (bonus["analysis"] == "fitted")]
    shipped = bonus[(bonus["fit_window"] == window) & (bonus["analysis"] == "calibration")
                    & (bonus["bucket"] == "all") & bonus["is_shipped"].astype(bool)]
    ship_by_unit = dict(zip(shipped["unit"], shipped["overdispersion"]))
    bias_by_unit = dict(zip(shipped["unit"], shipped["relative_bias"]))
    return pd.DataFrame({
        "unit": fitted["unit"].astype(str).to_numpy(),
        "fitted": fitted["overdispersion"].astype(float).to_numpy(),
        "shipped_constant": [float(ship_by_unit.get(u, float("nan")))
                             for u in fitted["unit"]],
        "shipped_relative_bias": [float(bias_by_unit.get(u, float("nan")))
                                  for u in fitted["unit"]],
        "n": fitted["n"].to_numpy(),
    }).reset_index(drop=True)


def minutes_dispersion(dispersion: pd.DataFrame,
                       window: str = SAFE_WINDOW) -> pd.DataFrame:
    """The game-level rho and what it implies, at every window the artifact carries."""
    part = dispersion[dispersion["metric"] == "game_level_rho"]
    out = pd.DataFrame({
        "fit_window": part["fit_window"].astype(str),
        "rho": part["rho"].astype(float),
        "implied_overdispersion": part["implied_overdispersion"].astype(float),
        "n_player_games": part["n_player_games"],
    })
    return out.reset_index(drop=True)


def injected_sigma(unification: pd.DataFrame) -> tuple[float, list[float]]:
    """`(shared sigma, per-role vector)` from the artifact's `shipped_configuration` row.

    The vector is empty when the injection is shared, which is how a reader tells "not
    graded" from "a grading that happens to be flat". Parsed from the `sigma_by_role` cell
    rather than recomputed, because this package may not import `src/` and the config it
    would need to read lives there.
    """
    row = unification[unification["unit"] == "ps_effect_shipped"]
    if row.empty:
        return float("nan"), []
    shared = float(row["sigma"].iloc[0])
    cell = str(row["sigma_by_role"].iloc[0] or "").strip() if "sigma_by_role" in row else ""
    graded = ([float(part) for part in cell.split("|")]
              if cell and cell.lower() != "nan" else [])
    return shared, graded


def _window_value(frames: dict, key: str, window: str) -> float:
    """One calibrated number's value at one window, from whichever artifact owns it.

    The injected sigma is not windowed, so it answers with its mean over role buckets at
    every window and `window_panel` leaves it out entirely.
    """
    if key == "player_season_sigma":
        shared, graded = injected_sigma(frames["unification"])
        return float(sum(graded) / len(graded)) if graded else shared
    if key == "copula":
        return copula_mean(frames["residual"], window)
    if key == "block_inflation":
        rows = block_inflation(frames["serial"], window)
        rows = rows[rows["component"] == "min"]
        return float(rows["block_inflation"].iloc[0]) if len(rows) else float("nan")
    if key == "bonus_overdispersion":
        rows = bonus_rows(frames["bonus"], window)
        rows = rows[rows["unit"] == "player_game"]
        return float(rows["fitted"].iloc[0]) if len(rows) else float("nan")
    rows = minutes_dispersion(frames["dispersion"])
    rows = rows[rows["fit_window"] == window]
    return (float(rows["implied_overdispersion"].iloc[0]) if len(rows)
            else float("nan"))


def calibrated_text(frames: dict, spec: Calibrated, value: float) -> str:
    """The metric's display string — the graded sigma as its vector, everything else as one
    number. A mean over four role buckets is a value nothing selected, so it is never what
    the tile shows."""
    if spec.key == "player_season_sigma":
        _, graded = injected_sigma(frames["unification"])
        if graded:
            return " / ".join(spec.fmt.format(v) for v in graded)
    return spec.fmt.format(value)


def calibrated_table(frames: dict, window: str = SAFE_WINDOW) -> pd.DataFrame:
    """Every calibrated number at one window, carrying its ROLE and where it enters.

    One table rather than two, with `role` as a column, so a caller that groups them cannot
    disagree with a caller that does not — and so a row can never be silently dropped from
    one group without appearing in the other.
    """
    rows = []
    for spec in CALIBRATED:
        value = _window_value(frames, spec.key, window)
        rows.append({"key": spec.key, "label": spec.label, "value": value,
                     "text": calibrated_text(frames, spec, value), "unit": spec.unit,
                     "artifact": spec.artifact, "target": spec.target,
                     "what": spec.what, "role": spec.role, "where": spec.where,
                     "windowed": spec.windowed,
                     "fit_window": window if spec.windowed else ""})
    return pd.DataFrame(rows)


def by_role(table: pd.DataFrame, role: str) -> pd.DataFrame:
    """One role's rows, in `CALIBRATED` order. The page's two sections come from here."""
    return table[table["role"] == role].reset_index(drop=True)


def window_panel(frames: dict, windows: tuple[str, ...] = FIT_WINDOWS) -> pd.DataFrame:
    """Each calibrated input at each window — the panel that shows the leak is small.

    The three windows differ by two seasons out of thirty, so every one of these moves by
    less than the precision it is quoted at. That is the whole argument for showing the
    window: a number that changed visibly would have been caught long ago, and one that does
    not is exactly the kind that gets consumed at the wrong window forever.
    """
    rows = []
    for spec in CALIBRATED:
        if not spec.windowed:
            continue
        values = {w: _window_value(frames, spec.key, w) for w in windows}
        finite = [v for v in values.values() if v == v]
        span = (max(finite) - min(finite)) if finite else float("nan")
        base = max((abs(v) for v in finite), default=0.0)
        for window in windows:
            value = values[window]
            rows.append({
                "key": spec.key, "label": spec.label, "fit_window": window,
                "value": value, "text": spec.fmt.format(value), "unit": spec.unit,
                "spread": span,
                "relative_spread": (span / base) if base else float("nan"),
                "is_safe_default": window == SAFE_WINDOW,
                "is_simulator": window == SIM_WINDOW,
            })
    return pd.DataFrame(rows)


#: Highlight-and-gray on the window `make simulate-season` actually consumes, matching the
#: other two bar charts on this page so the reader learns one encoding.
#:
#: Three windows would be exactly `ALL_PAIRS_CAP` and colouring all three is legal — but it
#: would spend the colour on an identity the y axis already carries, and leave the one
#: question a reader brings to this figure ("which of these ships?") answered nowhere on it.
WINDOW_SLOTS = {SIM_WINDOW: 0}


def window_facets(panel: pd.DataFrame) -> pd.DataFrame:
    """`window_panel` in the schema `charts.fig_metric_facets` reads.

    Reshaped rather than redrawn: the minutes page's spread panel and this one are the same
    figure — a handful of rows compared inside each facet, facets in different units — so
    they share a builder. The `reference` column is all-null here because none of the four
    inputs has a target value to draw; the minutes page's predictive-sd facet does.
    """
    if panel.empty:
        return pd.DataFrame(columns=["metric_label", "head", "label", "value", "text",
                                     "reference"])
    return pd.DataFrame({
        "metric_label": [f"{r.label} ({r.unit})" for r in panel.itertuples(index=False)],
        "head": panel["fit_window"].to_numpy(),
        "label": panel["fit_window"].to_numpy(),
        "value": panel["value"].to_numpy(),
        "text": panel["text"].to_numpy(),
        "reference": [None] * len(panel),
    })


def widest_relative_spread(panel: pd.DataFrame) -> tuple[str, float]:
    """The input that moves most across the three windows, and by how much.

    The block's headline number, and it is deliberately the *worst* case rather than the
    mean: "the largest of the four moves by 1.5%" is a statement a reader can act on, and an
    average over four would hide whichever one did not behave.
    """
    if panel.empty:
        return "", float("nan")
    per_input = (panel.drop_duplicates("key")
                 .sort_values("relative_spread", ascending=False))
    top = per_input.iloc[0]
    return str(top["label"]), float(top["relative_spread"])


# ── The availability layout ───────────────────────────────────────────────────
#
# A fifth simulator input, and the one that is not a scalar. The availability head says *how
# many* games a player misses; it cannot say *where* they fall, because `gp` is invariant to
# the arrangement. So the layout is chosen at draw time rather than fitted, which is exactly
# what puts it on this page rather than on the Availability model page.
#
# **Only the shipped arm and the observed target are read here.** The ladder artifact carries
# the arms that selected `tenure_merge`; those are a plan-doc argument and a decision-log
# entry, not a data visualization. What a reader of this page needs is what the simulator
# does and how close it lands.

#: The scoring-period statistics the layout is answerable for, in the order the block reads
#: them, with the unit each is measured in. `p_half_period` is deliberately absent: the
#: arrangement barely moves it, which is a fact about that metric rather than about the
#: layout, and `availability_exchangeability._attach_gaps` already declines to score it.
LAYOUT_METRICS = (
    ("p_dead_period", "Share of scoring periods with zero games", "share"),
    ("longest_dead_run", "Longest run of consecutive dead periods", "periods"),
    ("p_dead_run", "Share of player-seasons hitting a 3-period dead run", "share"),
)

#: Highlight-and-gray again, and the same encoding the copula and window figures use: the
#: thing that ships takes the colour, and the reference it is measured against stays ink.
LAYOUT_SLOTS = {SHIPPED_LAYOUT: 0}

#: The edge-profile figure's single row key. Every bin takes the same slot because none of
#: them *ships* — the gradient across them is the mechanism, not a choice between them.
EDGE_KEY = "edge_share"
EDGE_SLOTS = {EDGE_KEY: 0}

#: Role buckets bottom-to-top, so a reader reads *up* the axis toward the players a roster is
#: built around — `eda.season_effects.ROLE_LABELS` order, which every other role-graded
#: figure in the project uses.
LAYOUT_ROLES = ("<12 mpg", "12-24", "24-30", "30+ mpg")


def _layout_rows(ladder: pd.DataFrame, analysis: str) -> pd.DataFrame:
    if ladder.empty or "analysis" not in ladder.columns:
        return ladder.iloc[:0]
    return ladder[ladder["analysis"] == analysis]


def layout_exposure(ladder: pd.DataFrame) -> pd.DataFrame:
    """Shipped layout against the realized one, per role bucket, per period statistic.

    The block's main figure. Both series are levels in the metric's own unit rather than a
    recovered share, because a share is a ratio against an arm this page does not draw.
    """
    rows = _layout_rows(ladder, "period_layout")
    if rows.empty:
        return pd.DataFrame(columns=["metric", "metric_label", "unit", "population",
                                     "shipped", "observed", "ratio"])
    keyed = rows.set_index(["population", "arm"])
    out = []
    for metric, label, unit in LAYOUT_METRICS:
        for population in LAYOUT_ROLES:
            if (population, SHIPPED_LAYOUT) not in keyed.index:
                continue
            shipped = float(keyed.loc[(population, SHIPPED_LAYOUT), metric])
            observed = float(keyed.loc[(population, OBSERVED_LAYOUT), metric])
            out.append({"metric": metric, "metric_label": label, "unit": unit,
                        "population": population, "shipped": shipped,
                        "observed": observed,
                        # `charts._head_colors` keys the page's colour slot off this, the
                        # way every other builder here does — the chart module never reads
                        # this one, so the name travels on the frame rather than by import.
                        "head": SHIPPED_LAYOUT,
                        "ratio": shipped / observed if observed else float("nan")})
    return pd.DataFrame(out)


def layout_headline(ladder: pd.DataFrame) -> pd.DataFrame:
    """The three pooled readings, as tiles: what ships, what is real, and the ratio."""
    rows = _layout_rows(ladder, "period_layout")
    if rows.empty:
        return pd.DataFrame(columns=["label", "text", "unit", "what"])
    keyed = rows.set_index(["population", "arm"])
    out = []
    for metric, label, unit in LAYOUT_METRICS:
        shipped = float(keyed.loc[("all", SHIPPED_LAYOUT), metric])
        observed = float(keyed.loc[("all", OBSERVED_LAYOUT), metric])
        digits = 4 if unit == "share" else 3
        out.append({"label": label, "unit": unit,
                    "text": f"{shipped:,.{digits}f}",
                    "observed": observed,
                    "what": f"realized {observed:,.{digits}f} on the same rows"})
    return pd.DataFrame(out)


def layout_spell_shape(ladder: pd.DataFrame) -> pd.DataFrame:
    """The absence-spell lengths the shipped layout **realizes**, against the observed ones.

    The block's second figure, and it does a different job from the first: the layout was
    chosen on scoring-period exposure, so the spell-length distribution is a statistic it was
    never selected against and reproducing it is independent evidence rather than a
    restatement.
    """
    rows = _layout_rows(ladder, "layout_spell_shape")
    if rows.empty:
        return pd.DataFrame(columns=["metric", "label", "shipped", "observed"])
    keyed = rows.set_index(["population", "arm"])
    wanted = (("spells_per_season", "Absence spells per season"),
              ("mean_spell", "Mean spell length (games)"),
              ("p_spell_ge10", "Share of spells reaching 10 games"),
              ("p_spell_ge30", "Share of spells reaching 30 games"))
    return pd.DataFrame([
        {"metric": metric, "label": label,
         "shipped": float(keyed.loc[("all", SHIPPED_LAYOUT), metric]),
         "observed": float(keyed.loc[("all", OBSERVED_LAYOUT), metric])}
        for metric, label in wanted])


def layout_edge_profile(clustering: pd.DataFrame) -> pd.DataFrame:
    """What the shipped layout conditions its edge-block draw on.

    The mechanism, in one figure: the share of a player-season's missed games that sits in a
    tenure edge block is a function of **how much he missed**, so that is the key the draw is
    resampled within. Pooled over roles, because the role split is what the *ends* are keyed
    on rather than the amount.
    """
    if clustering.empty or "analysis" not in clustering.columns:
        return pd.DataFrame(columns=["missed_share_bin", "mean_edge_frac",
                                     "player_seasons"])
    rows = clustering[(clustering["analysis"] == "edge_profile")
                      & (clustering["population"] == "all")
                      & (clustering["missed_share_bin"] != "all")]
    return (rows[["missed_share_bin", "mean_edge_frac", "p_no_edge", "player_seasons"]]
            .reset_index(drop=True))


def layout_edge_ends(clustering: pd.DataFrame) -> pd.DataFrame:
    """Which *end* the block sits at, per role — the second key the draw is resampled on.

    The two run opposite ways and that is the whole reason role is a key at all: a fringe
    player's edge games are mostly a late first appearance, a star's are mostly a season that
    ended early.
    """
    if clustering.empty or "analysis" not in clustering.columns:
        return pd.DataFrame(columns=["population", "mean_pre_frac", "mean_post_frac"])
    rows = clustering[(clustering["analysis"] == "edge_profile")
                      & (clustering["missed_share_bin"] == "all")
                      & (clustering["population"].isin(LAYOUT_ROLES))]
    order = {label: i for i, label in enumerate(LAYOUT_ROLES)}
    return (rows[["population", "mean_pre_frac", "mean_post_frac", "player_seasons"]]
            .assign(_o=lambda f: f["population"].map(order))
            .sort_values("_o").drop(columns="_o").reset_index(drop=True))


def layout_rows_scored(ladder: pd.DataFrame) -> int:
    """Player-seasons behind the block, for the caption to state rather than imply."""
    rows = _layout_rows(ladder, "period_layout")
    if rows.empty:
        return 0
    pooled = rows[(rows["population"] == "all") & (rows["arm"] == OBSERVED_LAYOUT)]
    return int(pooled["player_seasons"].iloc[0]) if len(pooled) else 0
