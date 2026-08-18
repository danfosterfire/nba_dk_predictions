"""Reconstruct who was available for each game, and summarize it per player-season.

Game logs contain **only games actually played** — 10.7 rows per team-game against a
~15-man roster — and `AVAILABLE_FLAG` is uniformly 1 in every season, so it carries no
information. An absence is therefore not a row you can read; it is the *gap* between a
team's schedule and a player's appearances, and finding it requires knowing when the
player was on the roster. The game logs only reveal that through appearances, which is
circular.

Two constructions are possible and both are biased, in opposite directions:

- **Appearance window** — from a player's first to his last appearance for a team. It is
  blind to season-ending injuries *by construction*: the window always ends on a game he
  played, so a player who tears an ACL in March has no trailing absence at all. Measured
  on 2016-17 → 2024-25, trailing missed games came out 0 for **all 1,889** player-seasons.
- **Full season** — the whole schedule if he played at least one game for the team. This
  sees the season-ending injury, but counts waived, traded and 10-day players as rostered
  all year.

Played rates are 0.727 and 0.509 respectively. They bracket the truth and **neither can
separate "injured" from "not on the roster" from "healthy scratch"** — the one distinction
the availability head depends on. That separation needs `BoxScoreSummaryV2.InactivePlayers`
and the box-score DNP `COMMENT` field, which are not yet fetched; see
`docs/availability-plan.md`, stage B.

So this module deliberately ships **both** windows rather than choosing. The panel is built
on the full-season window with an `in_appearance_window` flag, since the appearance window
is a strict subset — one artifact, both constructions, and the gap between them is itself
the honest error bar.

**The `status` column replaces the bracket with a measurement** wherever
`src/data/boxscore_status.py` has been backfilled. Each panel row resolves to one of:

| status | meaning |
|---|---|
| `played` | logged minutes |
| `dnp` | dressed and available, not used — a *rotation* fact, not a health one |
| `inactive` | on the roster, not available to play; the endpoint states no reason |
| `not_rostered` | not on that game's roster at all — the waived/traded/10-day case that contaminates the full window |
| `unknown` | that game is not in the backfill (pre-2006-07, or not yet fetched) |

`not_rostered` and `unknown` are kept apart deliberately. Collapsing them would let an
un-backfilled season read as a season of players nobody rostered, which is the failure
mode most likely to be mistaken for a finding.

Per-season aggregates follow the project convention that a traded player is attributed
wholly to his **last** team: `gp` counts games for every team, but the schedule-relative
measures (start, end, and trailing absences) are read over his last team's timeline.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.boxscore_status import load_status as load_boxscore_status
from src.data.boxscore_status import pad_game_id
from src.data.fetch import _season_start_year, _slug, nbastats_dir
from src.data.preprocess import PLAYOFFS, load_raw

# Columns needed off the raw game log. Everything else is rank noise.
LOG_COLS = {"SEASON_YEAR", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID",
            "GAME_ID", "GAME_DATE", "MIN"}

# How a missed game is classified once the box-score backfill has landed. `dnp` is split
# out from the injury reasons on purpose: a player who dressed and was not used was
# *available*, and folding him in with the injured would be the same conflation the
# schedule-minus-appearances construction already makes.
MISSED_KINDS = ["injury", "scratch", "suspension", "gleague", "personal",
                "inactive", "not_rostered", "other", "unknown"]

# A spell of this many missed games is a "major" absence. 36.6% of all missed games sit
# in spells at least this long, against 47.6% of spells being a single game — the two
# ends are different processes (season-ending injury vs. rest and minor knocks) and want
# different models.
LONG_SPELL_GAMES = 10

# Windows the season-edge measures are read over.
SEASON_EDGE_GAMES = 10   # "did his season end with him unavailable"
SEASON_OPEN_GAMES = 20   # "did he start the next one unavailable"


# ── Schedules ─────────────────────────────────────────────────────────────────

def team_schedule(season: str, raw_dir: str | Path) -> pd.DataFrame:
    """Every game each team played in `season`, indexed chronologically.

    Keyed on `team_id`, never `team_abbreviation` — the abbreviation has 36 categories
    to the id's 30 and splits relocated franchises from their own history.
    """
    path = nbastats_dir(raw_dir) / f"game_logs_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "team_id", "game_id",
                                     "game_date", "team_game_index"])

    gl = pd.read_csv(path, usecols=lambda c: c in LOG_COLS, low_memory=False)
    gl.columns = [c.lower() for c in gl.columns]
    gl["game_date"] = pd.to_datetime(gl["game_date"], errors="coerce")
    gl = gl.dropna(subset=["game_date"])

    games = (gl[["team_id", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["team_id", "game_date", "game_id"]))
    games["team_game_index"] = games.groupby("team_id").cumcount()
    games.insert(0, "season", season)
    return games.reset_index(drop=True)


def playoff_workload(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """Per (player, season) playoff games and minutes — prior-season *workload*.

    A deep run is up to ~28 extra high-intensity games that the regular-season logs
    cannot see, so `total_minutes` undercounts the real mileage of exactly the players
    who carried the most. This is the only sanctioned use of the playoff logs: the games
    are a **feature of season S-1**, never a row to fit (see `CLAUDE.md`, "Scope").

    The timing is clean without any special handling. Season S-1's playoffs finish in
    June and season S opens in October, so the lag-1 value of every column here is known
    at `as_of_date` — it goes through the same `with_lags` machinery as everything else
    and inherits the same point-in-time guarantee.

    Players with no playoff appearance are **absent from the returned frame**, not zero:
    the caller decides, and for availability zero is the right fill because "played no
    playoff games" is a fact rather than a missing measurement. Kept separate so the
    distinction stays visible at the join.

    Missing playoff logs **raise**, they do not degrade to zeros. "Nobody made the
    playoffs in 30 seasons" and "the logs were never fetched" would otherwise be the same
    frame, which is the `not_rostered`-vs-`unknown` collapse this module already refuses
    to make elsewhere. Run `make fetch` first.
    """
    try:
        logs = load_raw(raw_dir, season_type=PLAYOFFS,
                        columns=["PLAYER_ID", "GAME_ID", "MIN"])
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"No playoff game logs in {raw_dir}. Playoff workload cannot be zero-filled "
            "— that is indistinguishable from a league where nobody reached the "
            "playoffs. Run `make fetch` to write game_logs_playoffs_*.csv.") from exc
    logs = logs[logs["season"].isin(seasons)]
    if logs.empty:
        return pd.DataFrame(columns=["season", "player_id", "playoff_games",
                                     "playoff_minutes", "playoff_mpg"])

    logs["min"] = pd.to_numeric(logs["MIN"], errors="coerce").fillna(0.0)
    out = (logs.rename(columns={"PLAYER_ID": "player_id"})
           .groupby(["season", "player_id"], as_index=False)
           .agg(playoff_games=("GAME_ID", "nunique"), playoff_minutes=("min", "sum")))
    out["playoff_mpg"] = out["playoff_minutes"] / out["playoff_games"]
    return out


def attach_workload(seasons_frame: pd.DataFrame, playoffs: pd.DataFrame,
                    season_order: list[str]) -> pd.DataFrame:
    """Merge playoff workload on, then build the totals that need a career history.

    Three derived columns, each answering a different question the raw counts do not:

    - `total_minutes_incl_playoffs` — the workload total the availability head should
      have been using all along, since `total_minutes` stops at game 82.
    - `playoff_minutes_share` — how much of the season's mileage came after it, which
      separates a starter on a first-round exit from one on a finals run.
    - `career_minutes` — cumulative mileage through the end of that season, playoffs
      included. `make aging` puts the availability arc at -54% peak-to-37 while the rate
      arc is +/-15%, and mileage is the mechanism age is standing in for.

    `career_minutes` is **left-censored at the first season on disk** (1996-97), exactly
    as `career_year` in `models.availability.build_design` already is, so it understates
    for players who debuted earlier. `career_seasons` ships beside it so a consumer can
    see how much history each row actually has rather than assuming it is complete.
    """
    out = seasons_frame.merge(playoffs, on=["season", "player_id"], how="left")
    # Coerce explicitly. A merge against an *empty* playoff frame leaves these columns
    # `object`, and `fillna(0.0)` does not fix that — the first symptom is `cumsum is not
    # supported for object dtype` several lines below, which reads like a pandas problem
    # rather than an unfetched-source one.
    for c in ("playoff_minutes", "playoff_mpg", "total_minutes"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0).astype(float)
    out["playoff_games"] = (pd.to_numeric(out["playoff_games"], errors="coerce")
                            .fillna(0).astype(int))
    out["made_playoffs"] = (out["playoff_games"] > 0).astype(int)

    out["total_minutes_incl_playoffs"] = out["total_minutes"] + out["playoff_minutes"]
    denom = out["total_minutes_incl_playoffs"].replace(0, np.nan)
    out["playoff_minutes_share"] = (out["playoff_minutes"] / denom).fillna(0.0)

    # Cumulative, in season order, within window so the two windows stay independent.
    order = {s: i for i, s in enumerate(season_order)}
    out["_i"] = out["season"].map(order)
    out = out.sort_values(["window", "player_id", "_i"])
    g = out.groupby(["window", "player_id"])
    out["career_minutes"] = g["total_minutes_incl_playoffs"].cumsum()
    out["career_seasons"] = g.cumcount() + 1
    return out.drop(columns="_i").reset_index(drop=True)


def _read_logs(season: str, raw_dir: str | Path) -> pd.DataFrame:
    path = nbastats_dir(raw_dir) / f"game_logs_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame()
    gl = pd.read_csv(path, usecols=lambda c: c in LOG_COLS, low_memory=False)
    gl.columns = [c.lower() for c in gl.columns]
    gl["game_date"] = pd.to_datetime(gl["game_date"], errors="coerce")
    return gl.dropna(subset=["game_date"])


# ── The panel ─────────────────────────────────────────────────────────────────

def build_season_panel(season: str, raw_dir: str | Path) -> pd.DataFrame:
    """One row per (player, team, team game) on the full-season window.

    `played` marks an appearance; `in_appearance_window` marks the rows the narrower
    construction would keep. Filtering to the flag gives the appearance-window panel
    exactly, so both bracketing constructions come off one artifact.
    """
    gl = _read_logs(season, raw_dir)
    if gl.empty:
        return pd.DataFrame()

    sched = team_schedule(season, raw_dir)
    # Keyed on team as well as game: a `game_id` belongs to *both* teams in the game, so
    # merging on (player, game) alone marks a traded player as having played for his old
    # team in games he played for his new one — which also stretches his appearance
    # window to the end of the season.
    appeared = (gl[["player_id", "team_id", "game_id", "min"]]
                .drop_duplicates(subset=["player_id", "team_id", "game_id"]))

    # A player is carried on team T's full-season window if he appeared for T at all.
    pairs = gl[["player_id", "team_id"]].drop_duplicates()
    panel = pairs.merge(sched.drop(columns="season"), on="team_id", how="inner")
    panel = panel.merge(appeared, on=["player_id", "team_id", "game_id"], how="left")
    panel["played"] = panel["min"].notna().astype(int)

    # The appearance window: first to last appearance for that team.
    played_rows = panel[panel["played"] == 1]
    span = (played_rows.groupby(["player_id", "team_id"])["team_game_index"]
            .agg(first_index="min", last_index="max").reset_index())
    panel = panel.merge(span, on=["player_id", "team_id"], how="left")
    panel["in_appearance_window"] = (
        (panel["team_game_index"] >= panel["first_index"])
        & (panel["team_game_index"] <= panel["last_index"])
    ).astype(int)

    panel = attach_status(panel, load_boxscore_status(season, raw_dir))

    panel.insert(0, "season", season)
    panel.insert(1, "season_start_year", _season_start_year(season))
    return (panel.drop(columns=["first_index", "last_index"])
            .sort_values(["player_id", "team_id", "team_game_index"])
            .reset_index(drop=True))


# ── The three-way status ──────────────────────────────────────────────────────

def attach_status(panel: pd.DataFrame, status: pd.DataFrame) -> pd.DataFrame:
    """Resolve every panel row to played / dnp / inactive / not_rostered / unknown.

    Keyed on `(player_id, team_id, game_id)` for the same reason the panel itself is: a
    `game_id` belongs to both teams, so a two-key merge credits a traded player's new-team
    games to his old team.

    Coverage is tracked per *game*, not per season. A game the backfill has not reached
    yet leaves its rows `unknown`; only inside a covered game does the absence of a
    box-score row mean the player was genuinely not on that roster. Without that
    distinction a half-finished backfill reads as a league-wide roster collapse.

    Game ids are normalized to the API's zero-padded 10-character form on both sides.
    The game-log CSVs store them as integers and the box-score files as strings, so a
    raw merge matches nothing at all — silently, and while still producing a full panel.
    """
    panel = panel.copy()
    panel["status_covered"] = 0
    panel["status"] = np.where(panel["played"] == 1, "played", "unknown")
    panel["missed_reason"] = ""

    if status.empty:
        return panel

    keys = ["player_id", "team_id", "_game_key"]
    panel["_game_key"] = panel["game_id"].map(pad_game_id)
    obs = status.assign(_game_key=status["game_id"].map(pad_game_id))
    obs = (obs[keys + ["status", "reason"]]
           .rename(columns={"status": "_status", "reason": "_reason"})
           .drop_duplicates(subset=keys, keep="last"))
    for key in ("player_id", "team_id"):
        obs[key] = pd.to_numeric(obs[key], errors="coerce")
        panel[key] = pd.to_numeric(panel[key], errors="coerce")

    panel = panel.merge(obs, on=keys, how="left")
    covered = panel["_game_key"].isin(set(obs["_game_key"]))
    panel["status_covered"] = covered.astype(int)

    # Inside a covered game, no box-score row means he was not on that game's roster.
    resolved = panel["_status"].where(panel["_status"].notna(), "not_rostered")
    panel["status"] = np.where(covered, resolved, panel["status"])
    panel["missed_reason"] = np.where(covered & panel["_reason"].notna(),
                                      panel["_reason"].fillna(""), "")
    # The game log is the authority on whether he played: a player with minutes is
    # `played` whatever the box score's comment field says.
    panel.loc[panel["played"] == 1, "status"] = "played"
    return panel.drop(columns=["_status", "_reason", "_game_key"])


def _missed_kind(status: pd.Series, reason: pd.Series) -> pd.Series:
    """One bucket per missed game, from the status and the comment vocabulary."""
    kind = pd.Series("unknown", index=status.index, dtype=object)
    kind[status == "not_rostered"] = "not_rostered"
    kind[status == "inactive"] = "inactive"

    dnp = status == "dnp"
    kind[dnp] = "other"
    for name, bucket in [("coach", "scratch"), ("injury", "injury"),
                         ("suspension", "suspension"), ("gleague", "gleague"),
                         ("personal", "personal"), ("rest", "personal")]:
        kind[dnp & (reason == name)] = bucket
    return kind


def missed_decomposition(window: pd.DataFrame) -> pd.DataFrame:
    """Missed games split by reason, per (season, player) — the point of the backfill.

    The buckets partition the missed games exactly, so they sum to `missed_games`.
    `status_coverage` rides along because the split is only interpretable in proportion
    to it: a season with 0.0 coverage produces `missed_unknown` and nothing else, which
    is a statement about the backfill rather than about the players.
    """
    cols = ["season", "player_id"] + [f"missed_{k}" for k in MISSED_KINDS]
    if window.empty or "status" not in window.columns:
        return pd.DataFrame(columns=cols + ["status_coverage"])

    coverage = (window.groupby(["season", "player_id"], as_index=False)
                .agg(status_coverage=("status_covered", "mean")))

    missed = window[window["played"] == 0]
    if missed.empty:
        out = coverage
    else:
        kinds = _missed_kind(missed["status"], missed["missed_reason"])
        counts = (missed.assign(_kind=kinds)
                  .groupby(["season", "player_id", "_kind"]).size()
                  .unstack("_kind", fill_value=0).reset_index()
                  .rename(columns={k: f"missed_{k}" for k in MISSED_KINDS}))
        out = coverage.merge(counts, on=["season", "player_id"], how="left")

    for kind in MISSED_KINDS:
        if f"missed_{kind}" not in out.columns:
            out[f"missed_{kind}"] = 0
    return out[cols + ["status_coverage"]]


def build_panel(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    frames = [build_season_panel(s, raw_dir) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No game log CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


# ── Absence spells ────────────────────────────────────────────────────────────

def absence_spells(panel: pd.DataFrame, window: str = "appearance") -> pd.DataFrame:
    """Consecutive runs of missed games, one row per spell.

    `window` selects the construction: `appearance` (blind to season-ending absences)
    or `full` (admits players who were no longer on the roster).
    """
    df = _select_window(panel, window)
    keys = ["season", "player_id", "team_id"]
    df = df.sort_values(keys + ["team_game_index"])

    missed = 1 - df["played"]
    # A new block starts wherever the played/missed state flips.
    block = (missed != missed.groupby([df[k] for k in keys]).shift()).cumsum()
    spells = (df.assign(_block=block)[missed == 1]
              .groupby(keys + ["_block"], as_index=False)
              .agg(spell_games=("played", "size"),
                   start_index=("team_game_index", "min"),
                   end_index=("team_game_index", "max")))
    return spells.drop(columns="_block")


def _select_window(panel: pd.DataFrame, window: str) -> pd.DataFrame:
    if window == "appearance":
        return panel[panel["in_appearance_window"] == 1]
    if window == "full":
        return panel
    raise ValueError(f"window must be 'appearance' or 'full', got {window!r}")


# ── The spell process: sufficient statistics and spell classes ────────────────
#
# Both of these exist for `src/models/games_played.py`. They are *reductions* of the
# panel, not features: the game-level Markov likelihood and the game-level duration
# likelihood are exactly recoverable from them, so a head fitted on these rows has the
# identical posterior a head fitted on 1.3M panel rows would. See
# `docs/games-played-plan.md`.

SPELL_CLASSES = ["interior", "left_truncated", "right_censored"]
TENURE_POSITIONS = ["pre", "interior", "post"]
CELL_KEYS = ["season", "player_id", "team_id"]


def collapse_transitions(panel: pd.DataFrame, window: str = "full") -> pd.DataFrame:
    """The four transition counts per (season, player, team) — sufficient statistics.

    Every feature the availability head uses is **constant within a player-season** (all
    lag-1/2/3 plus age), which is the prediction-time constraint rather than a modelling
    choice. For a two-state chain with observed states and cell-constant transition
    probabilities the game-level likelihood is therefore

        L = h^onsets (1-h)^(at_risk_played - onsets)
            * r^recoveries (1-r)^(at_risk_missed - recoveries)

    up to a factor free of the parameters, so `(onsets, at_risk_played, recoveries,
    at_risk_missed)` are sufficient. This is the same algebraic collapse the count heads
    already use, not an approximation, and `tests/test_games_played.py` pins it against an
    explicit product over transitions.

    **The collapse is conditional on the initial state**, which it deliberately does not
    carry into the likelihood: the first game of a cell contributes no transition, so `s0`
    needs its own head. The tenure decomposition supplies it — a tenure begins on a game
    the player appeared in, by construction.

    Right-censoring needs no term at all on this side: the product runs over *observed*
    transitions and the terminal state contributes no factor.
    """
    df = _select_window(panel, window)
    if df.empty:
        return pd.DataFrame(columns=CELL_KEYS + [
            "n_games", "played_games", "initial_state", "onsets", "at_risk_played",
            "recoveries", "at_risk_missed"])
    df = df.sort_values(CELL_KEYS + ["team_game_index"])

    played = df["played"].to_numpy(dtype=np.int64)
    prior = df.groupby(CELL_KEYS, sort=False)["played"].shift()
    seen = prior.notna().to_numpy()
    prior_played = np.nan_to_num(prior.to_numpy(), nan=0.0).astype(np.int64)

    work = df[CELL_KEYS].copy()
    work["n_games"] = 1
    work["played_games"] = played
    work["at_risk_played"] = np.where(seen & (prior_played == 1), 1, 0)
    work["onsets"] = np.where(seen & (prior_played == 1) & (played == 0), 1, 0)
    work["at_risk_missed"] = np.where(seen & (prior_played == 0), 1, 0)
    work["recoveries"] = np.where(seen & (prior_played == 0) & (played == 1), 1, 0)

    out = work.groupby(CELL_KEYS, as_index=False).sum()
    first = (df.groupby(CELL_KEYS, as_index=False)
             .agg(initial_state=("played", "first"),
                  first_index=("team_game_index", "min"),
                  last_index=("team_game_index", "max")))
    return out.merge(first, on=CELL_KEYS, how="left")


def spell_classes(panel: pd.DataFrame, window: str = "full") -> pd.DataFrame:
    """`absence_spells` plus the censoring class and where the spell sits in the tenure.

    Three classes, and the split matters because the duration likelihood treats them
    differently and because two of them are mostly roster mechanics rather than health:

    | class | meaning | likelihood contribution |
    |---|---|---|
    | `interior` | starts and ends inside the timeline | `P(T = t)` |
    | `left_truncated` | already in progress at the cell's first game | residual duration |
    | `right_censored` | still running at the cell's last game | `P(T >= t)` |

    On the **appearance** window every spell is interior by construction — the window
    opens and closes on games he played — which is exactly why the tenure decomposition
    removes censoring from the within-tenure duration head rather than having to model it.

    `tenure_position` is the second, independent split: whether the spell falls before a
    player's first appearance, after his last, or between them. `not_rostered_share`
    rides along per spell because 61.4% of full-window missed games sit in edge spells and
    roughly half of *those* games are `not_rostered` — so a duration head fitted on the
    full window is fitting roster mechanics, and this column is what makes that visible
    beside every coefficient rather than argued about.
    """
    spells = absence_spells(panel, window)
    if spells.empty:
        return spells.assign(spell_class=[], tenure_position=[], censored=[],
                             truncated=[], not_rostered_share=[])

    df = _select_window(panel, window)
    bounds = (df.groupby(CELL_KEYS, as_index=False)
              .agg(cell_first=("team_game_index", "min"),
                   cell_last=("team_game_index", "max")))
    appeared = df[df["played"] == 1]
    tenure = (appeared.groupby(CELL_KEYS, as_index=False)
              .agg(first_appearance=("team_game_index", "min"),
                   last_appearance=("team_game_index", "max")))

    out = spells.merge(bounds, on=CELL_KEYS, how="left").merge(
        tenure, on=CELL_KEYS, how="left")
    out["truncated"] = (out["start_index"] <= out["cell_first"]).astype(int)
    out["censored"] = (out["end_index"] >= out["cell_last"]).astype(int)
    # A cell only exists for a team the player appeared for, so a spell cannot be both.
    out["spell_class"] = np.where(
        out["truncated"] == 1, "left_truncated",
        np.where(out["censored"] == 1, "right_censored", "interior"))
    out["tenure_position"] = np.where(
        out["end_index"] < out["first_appearance"], "pre",
        np.where(out["start_index"] > out["last_appearance"], "post", "interior"))

    if "status" in df.columns:
        missed = df[df["played"] == 0]
        share = _spell_status_share(missed, out)
        out["not_rostered_share"] = share
    else:
        out["not_rostered_share"] = np.nan
    return out.drop(columns=["cell_first", "cell_last"])


def _spell_status_share(missed: pd.DataFrame, spells: pd.DataFrame) -> np.ndarray:
    """Share of each spell's games whose box-score status is `not_rostered`.

    Joined on the interval rather than on a spell id, because `absence_spells` does not
    carry one back to the panel. An interval join over ~83k spells and ~580k missed rows
    is done per cell with `searchsorted`, which keeps it linear.
    """
    flag = (missed["status"] == "not_rostered").to_numpy(dtype=float)
    frame = missed[CELL_KEYS + ["team_game_index"]].copy()
    frame["_flag"] = flag
    frame = frame.sort_values(CELL_KEYS + ["team_game_index"])

    keyed: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}
    for key, grp in frame.groupby(CELL_KEYS, sort=False):
        idx = grp["team_game_index"].to_numpy()
        keyed[key] = (idx, np.concatenate([[0.0], np.cumsum(grp["_flag"].to_numpy())]))

    out = np.full(len(spells), np.nan)
    starts = spells["start_index"].to_numpy()
    ends = spells["end_index"].to_numpy()
    lengths = spells["spell_games"].to_numpy(dtype=float)
    for i, key in enumerate(zip(*(spells[k] for k in CELL_KEYS))):
        hit = keyed.get(key)
        if hit is None:
            continue
        idx, cum = hit
        lo = int(np.searchsorted(idx, starts[i], side="left"))
        hi = int(np.searchsorted(idx, ends[i], side="right"))
        out[i] = (cum[hi] - cum[lo]) / max(lengths[i], 1.0)
    return out


def tenure_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """Per (season, player, team): the entry index, the exit index, and what is inside.

    `team_games = pre_tenure + tenure_games + post_tenure` exactly, and
    `gp = played games inside the tenure`, so the full-window games-played share is
    reconstructed by the three factors with no residual. That identity is what lets the
    two-state chain be confined to the interval where it demonstrably works — a departure
    is an absorbing hitting time, not a low recovery rate, and a chain that has to
    represent it as the latter relocates it to the player's first absence.
    """
    keys = CELL_KEYS
    appeared = panel[panel["played"] == 1]
    tenure = (appeared.groupby(keys, as_index=False)
              .agg(entry_index=("team_game_index", "min"),
                   exit_index=("team_game_index", "max"),
                   gp=("played", "sum")))
    schedule = (panel.groupby(keys, as_index=False)
                .agg(team_games=("team_game_index", lambda s: int(s.max()) + 1)))
    out = tenure.merge(schedule, on=keys, how="left")
    out["pre_tenure"] = out["entry_index"]
    out["post_tenure"] = out["team_games"] - 1 - out["exit_index"]
    out["tenure_games"] = out["exit_index"] - out["entry_index"] + 1
    return out


def _trailing_missed(played: pd.Series) -> int:
    """Consecutive missed games at the very end of the timeline.

    Zero by construction on the appearance window — that is the artifact this exists to
    make visible, not a finding about players. A timeline with no appearance at all is
    unreachable from the panel (a player is only carried for teams he played for), but
    the literal answer there is every game, not none.
    """
    arr = played.to_numpy()
    nonzero = np.flatnonzero(arr)
    return len(arr) if len(nonzero) == 0 else int(len(arr) - nonzero[-1] - 1)


# ── Per player-season summary ─────────────────────────────────────────────────

def season_availability(panel: pd.DataFrame, window: str = "full",
                        long_spell_games: int = LONG_SPELL_GAMES,
                        edge_games: int = SEASON_EDGE_GAMES,
                        open_games: int = SEASON_OPEN_GAMES) -> pd.DataFrame:
    """One row per (player, season): games played, spell structure, season edges.

    Volume totals (`gp`, `total_minutes`) count every team the player appeared for.
    Schedule-relative measures are read over his **last** team's timeline, matching the
    project convention that a traded player is attributed wholly to his last team.
    """
    sel = _select_window(panel, window)
    if sel.empty:
        return pd.DataFrame()

    # Volume across all teams.
    totals = (sel.groupby(["season", "season_start_year", "player_id"], as_index=False)
              .agg(gp=("played", "sum"), total_minutes=("min", "sum")))

    # Last team = the team of the player's final appearance that season.
    played_rows = sel[sel["played"] == 1]
    last_team = (played_rows.sort_values(["game_date", "game_id"])
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(team_id=("team_id", "last")))
    primary = sel.merge(last_team, on=["season", "player_id", "team_id"], how="inner")
    primary = primary.sort_values(["season", "player_id", "team_game_index"])

    grp = primary.groupby(["season", "player_id"])
    edges = grp.agg(
        team_games=("team_game_index", lambda s: int(s.max()) + 1),
        window_games=("played", "size"),
        played_in_window=("played", "sum"),
        trailing_missed=("played", _trailing_missed),
    ).reset_index()

    n_edge, n_open = edge_games, open_games
    edge_rate = (primary[primary["team_game_index"]
                         >= primary.groupby(["season", "player_id"])["team_game_index"]
                         .transform("max") - (n_edge - 1)]
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(end_play_rate=("played", "mean")))
    open_rate = (primary[primary["team_game_index"] < n_open]
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(start_play_rate=("played", "mean")))

    spells = absence_spells(panel, window)
    if spells.empty:
        spell_agg = pd.DataFrame(columns=["season", "player_id", "n_spells",
                                          "longest_spell", "single_game_spells",
                                          "long_spells", "missed_games"])
    else:
        spell_agg = (spells.groupby(["season", "player_id"], as_index=False)
                     .agg(n_spells=("spell_games", "size"),
                          longest_spell=("spell_games", "max"),
                          single_game_spells=("spell_games", lambda s: int((s == 1).sum())),
                          long_spells=("spell_games",
                                       lambda s: int((s >= long_spell_games).sum())),
                          missed_games=("spell_games", "sum")))

    out = (totals.merge(edges, on=["season", "player_id"], how="left")
           .merge(edge_rate, on=["season", "player_id"], how="left")
           .merge(open_rate, on=["season", "player_id"], how="left")
           .merge(spell_agg, on=["season", "player_id"], how="left")
           .merge(missed_decomposition(sel), on=["season", "player_id"], how="left"))

    for c in (["n_spells", "longest_spell", "single_game_spells", "long_spells",
               "missed_games"] + [f"missed_{k}" for k in MISSED_KINDS]):
        out[c] = out[c].fillna(0).astype(int)
    out["status_coverage"] = out["status_coverage"].fillna(0.0)

    out["gp_share"] = out["gp"] / out["team_games"]
    out["minutes_per_game"] = out["total_minutes"] / out["gp"].replace(0, np.nan)
    # Availability *while rostered* vs. share of the season spent on a roster. On the
    # full window the second is ~1 by construction; it only bites on the appearance one.
    out["window_share"] = out["window_games"] / out["team_games"]
    out["available_rate"] = out["played_in_window"] / out["window_games"].replace(0, np.nan)
    out.insert(0, "window", window)
    return out


# ── Artifacts ─────────────────────────────────────────────────────────────────

def save_artifacts(panel: pd.DataFrame, seasons_frame: pd.DataFrame,
                   features_dir: str | Path) -> dict[str, Path]:
    features_dir = Path(features_dir)
    features_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "panel": features_dir / "availability_panel.parquet",
        "seasons": features_dir / "availability_features.parquet",
    }
    panel.to_parquet(paths["panel"], index=False)
    seasons_frame.to_parquet(paths["seasons"], index=False)
    return paths


def load_artifacts(features_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_dir = Path(features_dir)
    panel = pd.read_parquet(features_dir / "availability_panel.parquet")
    seasons = pd.read_parquet(features_dir / "availability_features.parquet")
    return panel, seasons


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]
    a_cfg = cfg.get("features", {}).get("availability", {})
    long_spell = a_cfg.get("long_spell_games", LONG_SPELL_GAMES)
    edge = a_cfg.get("season_edge_games", SEASON_EDGE_GAMES)
    open_g = a_cfg.get("season_open_games", SEASON_OPEN_GAMES)

    panel = build_panel(seasons, raw_dir)
    print(f"Built availability panel: {len(panel):,} player-games over "
          f"{panel['season'].nunique()} seasons")

    frames = []
    for window in ("appearance", "full"):
        sel = _select_window(panel, window)
        rate = sel["played"].mean()
        frame = season_availability(panel, window, long_spell, edge, open_g)
        frames.append(frame)
        print(f"  {window:<10} window: {len(sel):>9,} rows, played rate {rate:.3f}, "
              f"{len(frame):,} player-seasons, "
              f"mean trailing missed {frame['trailing_missed'].mean():.2f}")

    seasons_frame = pd.concat(frames, ignore_index=True)

    playoffs = playoff_workload(seasons, raw_dir)
    seasons_frame = attach_workload(seasons_frame, playoffs, seasons)
    ran = seasons_frame[seasons_frame["window"] == "full"]
    played = ran[ran["playoff_games"] > 0]
    print(f"  playoff workload: {len(playoffs):,} player-seasons with a playoff "
          f"appearance, {ran['made_playoffs'].mean():.1%} of rows")
    print(f"    mean playoff games {played['playoff_games'].mean():.1f} "
          f"(max {int(ran['playoff_games'].max())}), "
          f"mean playoff minutes {played['playoff_minutes'].mean():.0f}")
    print(f"    total_minutes undercounts mileage by "
          f"{played['playoff_minutes_share'].mean():.1%} on those rows")

    paths = save_artifacts(panel, seasons_frame, features_dir)
    for name, dest in paths.items():
        n = len(panel) if name == "panel" else len(seasons_frame)
        print(f"Saved {n:,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
