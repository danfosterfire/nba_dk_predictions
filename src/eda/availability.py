"""What is knowable about games played, measured rather than assumed.

Games played is simultaneously the **largest lever on the season DK total and the least
predictable input in the project** — `log(season total)` is 73.4% explained by `log(games)`
alone, and games played persists year over year at r = 0.316 against 0.779 for minutes per
game. This module measures how much of that gap is closeable with data already on disk, so
the availability head is designed against numbers instead of intuition.

Nine measurements, each of which changed the plan in `docs/availability-plan.md`:

1. **`window_bracket`** — the two roster-window constructions and the gap between them.
   The appearance window is blind to season-ending absences *by construction*; this reports
   the artifact (trailing missed games ≡ 0) rather than letting it look like a finding.
2. **`persistence`** — year-over-year *r* for every availability quantity. Games played
   reaches ~0.32; minutes per game ~0.79.
3. **`predictor_r2`** — an R² ladder for predicting next-season games-played share. **These
   are in-sample fits**: read them as an ordering and a ceiling, not as achievable gains.
   The distinction matters here because this project has already been burned once by
   comparing an in-sample ceiling against a held-out gain (see the opponent section of
   `README.md`), and the conclusion drawn from this ladder — that the internal ceiling is
   low — is only safe in the pessimistic direction.
4. **`multiyear`** — whether averaging more history helps. It does not, which is the single
   most design-relevant null here: there is no durability latent to extract.
5. **`spell_distribution`** — absences are two processes, not one.
6. **`carryover`** — what ending a season unavailable implies for the next.
7. **`overdispersion`** — season GP against a binomial. ~20×, with a heavy left tail, which
   is why the head must emit a distribution rather than a point estimate.
8. **`decomposition`** — whether splitting absences by *reason* beats the aggregate, on
   the real `status` labels from `src/data/boxscore_status.py`. On the schedule-derived
   labels this was a null (`available_rate` persists at 0.217, `missed_games` at 0.137,
   against `gp_share`'s 0.317), but those labels cannot tell injury from scratch from
   waiver, so the null was untestable rather than settled. This retests it on labels that
   can, and reports `status_coverage` beside every figure — a decomposition measured where
   the backfill has not reached is a statement about the backfill.
9. **`serial_structure`** — whether a 2-state Markov chain is the right model for
   game-to-game availability. Serial correlation is strong (ρ = 0.60) and the geometric
   spell distribution it implies gets the mean right while missing both tails. Reported
   because "absences are clustered, so use an AR model" is the obvious instinct and it is
   only half correct. It is emitted **twice**: once on the appearance window over all
   players (the recorded rows, unchanged) and once on the frame `overdispersion` uses —
   full window, established rotation players — under `*_rotation` keys, because the two
   figures were being divided into each other across a window *and* a population change.

Season is absorbed throughout and pairs are minutes-weighted, reusing
`persistence.py`'s `demean_within` / `pair_weights` / `weighted_corr` so the numbers sit on
the same footing as the rest of the project's persistence work.

Output is deliberately long and un-fitted — one row per (measurement, window, key, metric) —
matching `aging_curves.csv`, because a GLM, a GBM and the simulator each want a different
slice of it.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug, nbastats_dir
from src.eda.feature_diagnostics import above_null
from src.eda.persistence import demean_within, pair_weights, weighted_corr
from src.data.preprocess import PLAYOFFS
from src.features.availability import (
    _select_window,
    absence_spells,
    build_panel,
    playoff_workload,
    season_availability,
)

# Quantities whose year-over-year persistence is worth a row.
PERSISTENCE_COLS = [
    "gp_share", "minutes_per_game", "total_minutes", "available_rate", "window_share",
    "n_spells", "longest_spell", "single_game_spells", "long_spells", "missed_games",
]

# The population the overdispersion figure is quoted on: last season's regulars.
ROTATION_MIN_MPG = 20.0
ROTATION_MIN_GP_SHARE = 0.70

# The reason split the box-score backfill makes possible. `scratch` (DNP - Coach's
# Decision) is the one that should behave differently from the rest: a scratched player
# was available, so it measures rotation status rather than health.
DECOMPOSITION_COLS = ["missed_injury", "missed_scratch", "missed_inactive",
                      "missed_gleague", "missed_not_rostered", "missed_suspension",
                      "missed_personal", "missed_other"]

# A player-season needs at least this share of its games covered by the backfill on both
# sides of a pair before it can say anything about reasons.
MIN_STATUS_COVERAGE = 0.9
MIN_DECOMPOSITION_PAIRS = 200

SPELL_QUANTILES = [0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
CARRYOVER_BINS = [-1, 0, 2, 5, 10, 20, 10_000]
CARRYOVER_LABELS = ["0", "1-2", "3-5", "6-10", "11-20", "21+"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _row(measurement: str, window: str, key: str, metric: str,
         value: float, n: int | None = None) -> dict:
    return {"measurement": measurement, "window": window, "key": key,
            "metric": metric, "n": n, "value": value}


def weighted_r2(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    """In-sample weighted R² of an OLS fit with an intercept.

    In-sample by design — these are ceilings for ranking predictor sets, not estimates of
    out-of-sample skill. `run` labels them as such in the printed summary.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(y, dtype=float)
    w = np.asarray(w, dtype=float)
    keep = np.isfinite(y) & np.isfinite(w) & np.isfinite(X).all(axis=1)
    X, y, w = X[keep], y[keep], w[keep]
    if len(y) < X.shape[1] + 2:
        return np.nan
    design = np.column_stack([np.ones(len(X)), X])
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(design * sw[:, None], y * sw, rcond=None)
    resid = y - design @ beta
    ybar = np.average(y, weights=w)
    denom = np.average((y - ybar) ** 2, weights=w)
    return float(1 - np.average(resid ** 2, weights=w) / denom) if denom > 0 else np.nan


#: The date the bio family's `age` is referenced to, measured rather than assumed. Age
#: recomputed from `BIRTH_DATE` at 31 December of the season's start year reproduces
#: `player_bio_stats`' own column to a mean of +0.018 years, inside half a year on 98.7%
#: of 525 shared 2025-26 players; 1 October and 1 February both miss by more.
AGE_REFERENCE = (12, 31)


def roster_ages(season: str, raw_dir: str | Path) -> pd.DataFrame:
    """Age from the roster snapshot, for a season the bio family cannot cover yet.

    🔴 **Not the roster's own `AGE` column, and the difference is a real bias.** That
    column is a *fetch-time* attribute: on a roster pulled during the season it matches the
    bio column exactly (98.7% of 525 players), and on the 2026-27 roster pulled in August
    2026 it sits **0.617 years low** against the season's own reference date, inside half a
    year on only 38.1% of players. Age and age squared are features on every head, so
    taking the convenient column would push a systematic error through the whole board.

    `BIRTH_DATE` is exact and does not move, so it is recomputed at the reference date and
    the `AGE` column is used only where a birth date is missing.
    """
    path = nbastats_dir(raw_dir) / f"team_rosters_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["season", "player_id", "age"])
    roster = pd.read_csv(path)
    if "PLAYER_ID" not in roster.columns:
        return pd.DataFrame(columns=["season", "player_id", "age"])

    month, day = AGE_REFERENCE
    reference = pd.Timestamp(year=_season_start_year(season), month=month, day=day)
    birth = pd.to_datetime(roster.get("BIRTH_DATE"), errors="coerce", format="mixed")
    age = (reference - birth).dt.days / 365.25
    if "AGE" in roster.columns:
        age = age.fillna(pd.to_numeric(roster["AGE"], errors="coerce"))
    out = pd.DataFrame({"season": season, "player_id": roster["PLAYER_ID"], "age": age})
    return out.dropna(subset=["age"]).drop_duplicates("player_id")


def load_ages(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """Player age per season, off the bio family. 100% coverage in practice.

    `player_bio_stats_<season>.csv` is a season-*statistics* endpoint, so it does not exist
    for a season that has not been played — which is the one season a production board is
    for. `roster_ages` is the fallback and is used only where the bio file is absent, so no
    played season's ages move.
    """
    frames = []
    for season in seasons:
        path = nbastats_dir(raw_dir) / f"player_bio_stats_{_slug(season)}.csv"
        if not path.exists():
            fallback = roster_ages(season, raw_dir)
            if len(fallback):
                frames.append(fallback)
            continue
        df = pd.read_csv(path, low_memory=False)
        df.columns = [c.lower() for c in df.columns]
        if "age" not in df.columns or "player_id" not in df.columns:
            continue
        frames.append(pd.DataFrame({"season": season,
                                    "player_id": df["player_id"].values,
                                    "age": pd.to_numeric(df["age"], errors="coerce")}))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def with_lags(frame: pd.DataFrame, seasons: list[str], cols: list[str],
              max_lag: int = 3) -> pd.DataFrame:
    """Attach lag-1..max_lag values of `cols`, pairing on the season *index*.

    Indexing rather than differencing the label stops a player who missed a year from
    silently pairing across the gap, exactly as `persistence.lagged_pairs` does.
    """
    order = {s: i for i, s in enumerate(seasons)}
    df = frame[frame["season"].isin(order)].copy()
    df["season_index"] = df["season"].map(order)
    out = df.copy()
    for lag in range(1, max_lag + 1):
        prior = df[["player_id", "season_index"] + cols].copy()
        prior["season_index"] = prior["season_index"] + lag
        prior = prior.rename(columns={c: f"{c}_lag{lag}" for c in cols})
        out = out.merge(prior, on=["player_id", "season_index"], how="left")
    return out


# ── 1. The two roster windows ─────────────────────────────────────────────────

def window_bracket(panel: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Played rate and season-edge visibility under each construction.

    The headline is that `trailing_missed` is identically zero on the appearance window.
    That is the construction talking, not the players, and it is reported so nobody reads
    a null into it later.
    """
    rows = []
    for window, frame in frames.items():
        sel = panel if window == "full" else panel[panel["in_appearance_window"] == 1]
        rows += [
            _row("window_bracket", window, "panel", "rows", float(len(sel)), len(sel)),
            _row("window_bracket", window, "panel", "played_rate",
                 float(sel["played"].mean()), len(sel)),
            _row("window_bracket", window, "season_edge", "mean_trailing_missed",
                 float(frame["trailing_missed"].mean()), len(frame)),
            _row("window_bracket", window, "season_edge", "share_trailing_missed_gt0",
                 float((frame["trailing_missed"] > 0).mean()), len(frame)),
            _row("window_bracket", window, "season_edge", "mean_end_play_rate",
                 float(frame["end_play_rate"].mean()), len(frame)),
        ]
    return rows


# ── 2. Persistence ────────────────────────────────────────────────────────────

def persistence(frame: pd.DataFrame, seasons: list[str], window: str) -> list[dict]:
    """Year-over-year r for each availability quantity, season-absorbed."""
    cols = list(dict.fromkeys(PERSISTENCE_COLS + ["total_minutes"]))
    lagged = with_lags(frame, seasons, cols, max_lag=1)
    lagged = lagged.dropna(subset=["gp_share_lag1"])
    if lagged.empty:
        return []
    w = pair_weights(lagged["total_minutes_lag1"].to_numpy(),
                     lagged["total_minutes"].to_numpy())
    groups = lagged["season"].to_numpy()

    rows = []
    for col in PERSISTENCE_COLS:
        prior, now = f"{col}_lag1", col
        if prior not in lagged:
            continue
        pair = lagged[[prior, now]].astype(float)
        keep = pair.notna().all(axis=1).to_numpy()
        if keep.sum() < 50:
            continue
        x = demean_within(pair[prior].to_numpy(), groups, w)[keep]
        y = demean_within(pair[now].to_numpy(), groups, w)[keep]
        r_w = weighted_corr(x, y, w[keep])
        r_u = weighted_corr(x, y)
        # `window_share` is identically 1 on the full window, so its correlation is
        # undefined rather than zero. A NaN row is not a measurement — skip it.
        if not np.isfinite(r_w) and not np.isfinite(r_u):
            continue
        rows += [
            _row("persistence", window, col, "r_within_weighted", r_w, int(keep.sum())),
            _row("persistence", window, col, "r_within_unweighted", r_u, int(keep.sum())),
        ]
    return rows


# ── 3. The predictor ladder, and 4. the multi-year null ───────────────────────

PREDICTOR_SETS = {
    "prior_gp_share":          ["gp_share_lag1"],
    "prior_mpg":               ["minutes_per_game_lag1"],
    "gp_lags_1_3":             ["gp_share_lag1", "gp_share_lag2", "gp_share_lag3"],
    "gp_and_mpg_lag1":         ["gp_share_lag1", "minutes_per_game_lag1"],
    "gp_and_mpg_lags_1_3":     ["gp_share_lag1", "gp_share_lag2", "gp_share_lag3",
                                "minutes_per_game_lag1", "minutes_per_game_lag2",
                                "minutes_per_game_lag3"],
    "plus_age_and_career":     ["gp_share_lag1", "gp_share_lag2", "gp_share_lag3",
                                "minutes_per_game_lag1", "minutes_per_game_lag2",
                                "minutes_per_game_lag3", "age", "age_sq", "career_year"],
}


def predictor_ladder(frame: pd.DataFrame, seasons: list[str], raw_dir: str | Path,
                     window: str) -> tuple[list[dict], pd.DataFrame]:
    """In-sample weighted R² for predicting next-season games-played share.

    Restricted to players with three prior seasons so every set is compared on identical
    rows — otherwise the richer sets would be scored on an easier, more-established
    population and the ladder would measure sample selection.
    """
    cols = ["gp_share", "minutes_per_game", "total_minutes"]
    lagged = with_lags(frame, seasons, cols, max_lag=3)

    ages = load_ages(seasons, raw_dir)
    if not ages.empty:
        lagged = lagged.merge(ages, on=["season", "player_id"], how="left")
    else:
        lagged["age"] = np.nan
    lagged["age_sq"] = lagged["age"] ** 2
    lagged["career_year"] = (lagged.sort_values("season_index")
                             .groupby("player_id").cumcount())

    need = ["gp_share_lag1", "gp_share_lag2", "gp_share_lag3",
            "minutes_per_game_lag1", "minutes_per_game_lag2", "minutes_per_game_lag3",
            "age", "gp_share"]
    d = lagged.dropna(subset=need).copy()
    if len(d) < 100:
        return [], d

    w = pair_weights(d["total_minutes_lag1"].to_numpy(), d["total_minutes"].to_numpy())
    groups = d["season"].to_numpy()
    y = demean_within(d["gp_share"].to_numpy(), groups, w)

    # Both weightings, because for *availability* the choice is not the settled question
    # it is for per-36 rates. Minutes weighting exists to stop a rate measured over three
    # garbage-time minutes from dominating a regression — real measurement error. Games
    # played has no such error: "he played 12 games" is exact. Weighting by minutes
    # therefore down-weights precisely the injured seasons the head exists to predict, so
    # the unweighted figure is arguably the honest one here even though the weighted one
    # is what `persistence.csv` reports elsewhere.
    ones = np.ones(len(d))
    unweighted_y = demean_within(d["gp_share"].to_numpy(), groups, None)

    rows = []
    for name, feature_cols in PREDICTOR_SETS.items():
        X = np.column_stack([demean_within(d[c].to_numpy(), groups, w)
                             for c in feature_cols])
        Xu = np.column_stack([demean_within(d[c].to_numpy(), groups, None)
                              for c in feature_cols])
        rows += [
            _row("predictor_r2", window, name, "r2_in_sample_weighted",
                 weighted_r2(X, y, w), len(d)),
            _row("predictor_r2", window, name, "r2_in_sample_unweighted",
                 weighted_r2(Xu, unweighted_y, ones), len(d)),
        ]

    # The multi-year null: does averaging three years of availability beat one?
    d["gp_share_mean3"] = d[["gp_share_lag1", "gp_share_lag2",
                             "gp_share_lag3"]].mean(axis=1)
    for key, col in [("gp_share_lag1", "gp_share_lag1"),
                     ("gp_share_mean3", "gp_share_mean3")]:
        x = demean_within(d[col].to_numpy(), groups, w)
        xu = demean_within(d[col].to_numpy(), groups, None)
        rows += [
            _row("multiyear", window, key, "r_weighted",
                 weighted_corr(x, y, w), len(d)),
            _row("multiyear", window, key, "r_unweighted",
                 weighted_corr(xu, unweighted_y), len(d)),
        ]
    return rows, d


# ── 5. Spell structure ────────────────────────────────────────────────────────

def spell_distribution(panel: pd.DataFrame, window: str,
                       long_spell_games: int = 10) -> list[dict]:
    """Absence spell lengths. Two processes: single games, and season-wreckers."""
    spells = absence_spells(panel, window)
    if spells.empty:
        return []
    lengths = spells["spell_games"]
    n = len(lengths)
    rows = [_row("spell_distribution", window, "all", "n_spells", float(n), n)]
    for q in SPELL_QUANTILES:
        rows.append(_row("spell_distribution", window, f"p{int(q * 100)}",
                         "spell_games", float(lengths.quantile(q)), n))
    rows += [
        _row("spell_distribution", window, "all", "share_single_game",
             float((lengths == 1).mean()), n),
        _row("spell_distribution", window, "all",
             f"share_missed_games_in_spells_ge{long_spell_games}",
             float(lengths[lengths >= long_spell_games].sum() / lengths.sum()), n),
        _row("spell_distribution", window, "all", "mean_spell_games",
             float(lengths.mean()), n),
    ]
    return rows


# ── 6. Carryover ──────────────────────────────────────────────────────────────

def carryover(frame: pd.DataFrame, seasons: list[str], window: str) -> list[dict]:
    """What ending season S-1 on an unresolved absence implies for season S.

    Meaningless on the appearance window, where trailing absences cannot exist; run it
    there anyway so the output shows the bracket rather than hiding it.
    """
    cols = ["gp_share", "trailing_missed", "start_play_rate", "total_minutes"]
    lagged = with_lags(frame, seasons, cols, max_lag=1)
    d = lagged.dropna(subset=["trailing_missed_lag1", "gp_share"]).copy()
    if d.empty:
        return []

    d["bucket"] = pd.cut(d["trailing_missed_lag1"], CARRYOVER_BINS,
                         labels=CARRYOVER_LABELS)
    rows = []
    for bucket, grp in d.groupby("bucket", observed=True):
        n = len(grp)
        rows += [
            _row("carryover", window, str(bucket), "next_season_gp_share",
                 float(grp["gp_share"].mean()), n),
            _row("carryover", window, str(bucket), "next_season_start_play_rate",
                 float(grp["start_play_rate"].mean()), n),
        ]

    # Incremental value over prior GP — the point being that it is small, because a
    # player who missed the end of the season already has a low prior GP share.
    w = pair_weights(d["total_minutes_lag1"].to_numpy(), d["total_minutes"].to_numpy())
    groups = d["season"].to_numpy()
    y = demean_within(d["gp_share"].to_numpy(), groups, w)
    base = demean_within(d["gp_share_lag1"].to_numpy(), groups, w)
    tail = demean_within(d["trailing_missed_lag1"].to_numpy(), groups, w)
    rows += [
        _row("carryover", window, "incremental", "r2_prior_gp_share",
             weighted_r2(base, y, w), len(d)),
        _row("carryover", window, "incremental", "r2_plus_trailing_missed",
             weighted_r2(np.column_stack([base, tail]), y, w), len(d)),
    ]
    return rows


# ── 7. Overdispersion ─────────────────────────────────────────────────────────

def overdispersion(frame: pd.DataFrame, seasons: list[str], window: str,
                   min_mpg: float = ROTATION_MIN_MPG,
                   min_gp_share: float = ROTATION_MIN_GP_SHARE,
                   bin_width: int = 8) -> list[dict]:
    """Season GP against a binomial, on established rotation players.

    Conditioning on *last* season's regulars is what makes the comparison fair: it fixes
    a population that a naive model would call reliably available, then measures how wide
    the realized distribution actually is.
    """
    cols = ["gp_share", "minutes_per_game", "total_minutes", "team_games"]
    lagged = with_lags(frame, seasons, cols, max_lag=1)
    rot = lagged[(lagged["minutes_per_game_lag1"] >= min_mpg)
                 & (lagged["gp_share_lag1"] >= min_gp_share)].dropna(subset=["gp_share"])
    n = len(rot)
    if n < 50:
        return []

    p, sd = float(rot["gp_share"].mean()), float(rot["gp_share"].std())
    games = float(rot["team_games"].median())
    binom_sd = float(np.sqrt(p * (1 - p) / games))
    rows = [
        _row("overdispersion", window, "rotation_players", "n", float(n), n),
        _row("overdispersion", window, "rotation_players", "mean_gp_share", p, n),
        _row("overdispersion", window, "rotation_players", "sd_gp_share", sd, n),
        _row("overdispersion", window, "rotation_players", "binomial_sd", binom_sd, n),
        _row("overdispersion", window, "rotation_players", "variance_ratio",
             float((sd ** 2) / (binom_sd ** 2)) if binom_sd > 0 else np.nan, n),
        _row("overdispersion", window, "rotation_players", "share_below_60_games",
             float((rot["gp_share"] * games < 60).mean()), n),
        _row("overdispersion", window, "rotation_players", "share_below_41_games",
             float((rot["gp_share"] * games < 41).mean()), n),
    ]
    gp = rot["gp_share"] * games
    edges = np.arange(0, games + bin_width, bin_width)
    counts, _ = np.histogram(gp, bins=edges)
    for lo, count in zip(edges[:-1], counts):
        rows.append(_row("overdispersion", window,
                         f"gp_{int(lo)}_{int(lo + bin_width)}", "count",
                         float(count), n))
    return rows


# ── 8. Does splitting absences by reason beat the aggregate? ──────────────────

def decomposition(frame: pd.DataFrame, seasons: list[str], window: str,
                  min_coverage: float = MIN_STATUS_COVERAGE,
                  min_pairs: int = MIN_DECOMPOSITION_PAIRS,
                  n_shuffles: int = 20, seed: int = 42) -> list[dict]:
    """Retest the decomposition null on the real box-score `status` labels.

    The question is narrow and worth stating precisely: **given last season's games-played
    share, does knowing *why* he missed those games add anything?** The aggregate version
    of this was already a null, but on labels that could not separate an injury from a
    healthy scratch from a waiver, so it was untestable rather than answered.

    Both sides of a pair must clear `min_coverage`, otherwise the split is measuring how
    far the backfill has run. Coverage and pair count are emitted first, so a thin result
    reads as thin rather than as a finding.
    """
    cols = (["gp_share", "minutes_per_game", "total_minutes", "missed_games",
             "status_coverage"] + DECOMPOSITION_COLS)
    have = [c for c in cols if c in frame.columns]
    if "status_coverage" not in have:
        return []

    lagged = with_lags(frame, seasons, have, max_lag=1)
    covered = lagged.dropna(subset=["gp_share", "gp_share_lag1"])
    covered = covered[(covered["status_coverage_lag1"] >= min_coverage)
                      & (covered["status_coverage"] >= min_coverage)]

    rows = [
        _row("decomposition", window, "coverage", "mean_status_coverage",
             float(frame["status_coverage"].mean()), len(frame)),
        _row("decomposition", window, "coverage", "player_seasons_above_threshold",
             float((frame["status_coverage"] >= min_coverage).sum()), len(frame)),
        _row("decomposition", window, "coverage", "usable_pairs",
             float(len(covered)), len(covered)),
    ]
    if len(covered) < min_pairs:
        # Not "no effect" — no measurement. Said out loud so the row is not read as a null.
        rows.append(_row("decomposition", window, "coverage", "insufficient_coverage",
                         1.0, len(covered)))
        return rows

    groups = covered["season"].to_numpy()
    ones = np.ones(len(covered))
    # Unweighted: the documented exception. Minutes weighting down-weights exactly the
    # injured seasons this measurement is about.
    y = demean_within(covered["gp_share"].to_numpy(), groups, None)

    present = [c for c in DECOMPOSITION_COLS if c in covered.columns]
    for col in present + ["missed_games"]:
        prior = f"{col}_lag1"
        if prior not in covered:
            continue
        x = demean_within(covered[prior].to_numpy(), groups, None)
        now = demean_within(covered[col].to_numpy(), groups, None)
        rows += [
            # What it says about next season, and whether it is stable enough to say it.
            _row("decomposition", window, col, "r_vs_next_gp_share",
                 weighted_corr(x, y), len(covered)),
            _row("decomposition", window, col, "r_persistence_of_column",
                 weighted_corr(x, now), len(covered)),
        ]

    base = demean_within(covered["gp_share_lag1"].to_numpy(), groups, None)
    aggregate = np.column_stack(
        [base, demean_within(covered["missed_games_lag1"].to_numpy(), groups, None)])
    # A column that is constant across the covered rows carries no information and would
    # only spend a degree of freedom against the null below.
    block = np.column_stack([demean_within(covered[f"{c}_lag1"].to_numpy(), groups, None)
                             for c in present])
    informative = block.std(axis=0) > 0
    block, present = block[:, informative], [c for c, k in zip(present, informative) if k]
    split = np.column_stack([base, block])

    # These are **in-sample** R², and adding k regressors to n rows raises it by roughly
    # k(1-R²)/(n-k-1) on pure noise alone — here ~+0.012 against an observed gain of the
    # same order. So the split ships with its own chance level, routed through the same
    # helper as every other importance number in the project. `rows` names what is
    # permuted, because "above a shuffled null" is meaningless without it.
    def statistic(values: dict) -> float:
        return weighted_r2(np.column_stack([base, block[values["rows"]]]), y, ones)

    null = above_null(statistic, {"rows": np.arange(len(covered))}, "rows",
                      n_shuffles=n_shuffles, seed=seed)

    rows += [
        _row("decomposition", window, "incremental", "r2_prior_gp_share",
             weighted_r2(base, y, ones), len(covered)),
        _row("decomposition", window, "incremental", "r2_plus_missed_games",
             weighted_r2(aggregate, y, ones), len(covered)),
        _row("decomposition", window, "incremental", "r2_plus_reason_split",
             null["statistic"], len(covered)),
        _row("decomposition", window, "incremental", "r2_reason_split_shuffled_null",
             null["null_mean"], len(covered)),
        _row("decomposition", window, "incremental", "r2_reason_split_null_sd",
             null["null_sd"], len(covered)),
        _row("decomposition", window, "incremental", "r2_reason_split_above_null",
             null["statistic"] - null["null_mean"], len(covered)),
        _row("decomposition", window, "incremental", "n_reason_columns",
             float(len(present)), len(covered)),
    ]
    return rows


# ── 10. Why the fitting frame is regular season only ──────────────────────────
#
# The scope decision rests on the product first — `docs/dk_best_ball_rules.md` puts the end
# of Round 4 at 4/4, before the playoffs start — and on a *sign flip* second. The sign flip
# is the statistical half and it was prose only.
#
# What it says: playoff minutes are not a level shift, they are a role interaction whose
# sign reverses. Bench players lose about half their minutes; starters gain. A pooled playoff
# indicator would fit one coefficient to a -50% effect and a +5% effect at once, and bench
# players are numerous enough to drag it toward compression — distorting exactly the star
# minutes the model most needs right.
#
# **Unweighted, per the recorded exception.** This is an availability measurement: minutes
# weighting would down-weight the bench players whose ratio is the whole point, and the
# statistics here are medians and shares, which have no exposure to weight by anyway.

# Role buckets in regular-season minutes per game. The boundaries are the conventional
# bench / rotation / starter lines, chosen before the measurement rather than fitted to it.
PLAYOFF_ROLE_BUCKETS = [0.0, 12.0, 24.0, 1e9]
PLAYOFF_ROLE_LABELS = ["bench_lt12", "rotation_12_24", "starter_24plus"]

# Regular-season games a player-season needs before its mpg describes a role at all.
PLAYOFF_MIN_REGULAR_GAMES = 20


def playoff_teams(seasons: list[str], raw_dir: str | Path) -> set[tuple[str, int]]:
    """(season, team_id) pairs that played at least one playoff game.

    Needed because the appearance rate's denominator is "players on a **playoff team**".
    Measuring it over every team would fold "his team missed the playoffs" into "he did not
    dress", which are different facts — the same `not_rostered`-vs-`unknown` distinction this
    module keeps everywhere else.
    """
    from src.data.preprocess import load_raw

    logs = load_raw(raw_dir, season_type=PLAYOFFS, columns=["TEAM_ID"])
    logs = logs[logs["season"].isin(seasons)]
    return set(zip(logs["season"], pd.to_numeric(logs["TEAM_ID"], errors="coerce")))


def playoff_scope(panel: pd.DataFrame, frame: pd.DataFrame, playoffs: pd.DataFrame,
                  teams: set[tuple[str, int]],
                  min_games: int = PLAYOFF_MIN_REGULAR_GAMES) -> list[dict]:
    """Playoff-to-regular MPG ratio and playoff appearance rate, by regular-season role.

    Two populations, deliberately different, because they answer different questions:

    - the **ratio** is over player-seasons that actually appeared in the playoffs, since a
      ratio needs a numerator;
    - the **appearance rate** is over `min_games`+ player-seasons on a playoff team,
      appeared or not, since that is what "did rotations shorten" asks.

    Both are bucketed on the *same* season's regular-season mpg, which is what makes
    "playoff-to-regular ratio by role" a coherent comparison.

    **The appearance rate is keyed on the player's LAST team, not on every team he appeared
    for**, and that is the difference from the superseded figures (0.664 / 0.838 / 0.922
    against 0.711 / 0.905 / 0.958 here). Keying on every team emits one row per
    (player, team) and records a player traded away from a playoff team in February as
    having "not appeared" for it — which is the roster-churn-as-unavailability conflation
    this module refuses to make everywhere else. It also double-counts him. The ordering, and
    therefore the finding, is unchanged either way; the level is not.
    """
    played = panel[panel["played"] == 1]
    last_team = (played.sort_values(["game_date", "game_id"])
                 .groupby(["season", "player_id"], as_index=False)
                 .agg(team_id=("team_id", "last")))
    d = frame.merge(last_team, on=["season", "player_id"], how="left")
    d = d[d["gp"] >= min_games].copy()
    d = d.merge(playoffs[["season", "player_id", "playoff_games", "playoff_mpg"]],
                on=["season", "player_id"], how="left")
    d["playoff_games"] = pd.to_numeric(d["playoff_games"], errors="coerce").fillna(0)
    d["on_playoff_team"] = [(s, t) in teams
                            for s, t in zip(d["season"],
                                            pd.to_numeric(d["team_id"], errors="coerce"))]
    if d.empty:
        # An empty population is not a measurement of zero — say so with the count rows
        # and emit no bucket rows at all.
        return [_row("playoff_scope", "full", "all", "n_regular_seasons_ge_min_games",
                     0.0, 0),
                _row("playoff_scope", "full", "all", "min_regular_games",
                     float(min_games), 0)]
    d["role"] = pd.cut(d["minutes_per_game"], PLAYOFF_ROLE_BUCKETS,
                       labels=PLAYOFF_ROLE_LABELS, right=False)

    appeared = d[d["playoff_games"] > 0]
    eligible = d[d["on_playoff_team"]]
    rows = [
        _row("playoff_scope", "full", "all", "n_regular_seasons_ge_min_games",
             float(len(d)), len(d)),
        _row("playoff_scope", "full", "all", "n_with_playoff_appearance",
             float(len(appeared)), len(appeared)),
        _row("playoff_scope", "full", "all", "n_on_playoff_teams",
             float(len(eligible)), len(eligible)),
        _row("playoff_scope", "full", "all", "min_regular_games",
             float(min_games), len(d)),
    ]
    for label in PLAYOFF_ROLE_LABELS:
        sub = appeared[appeared["role"] == label]
        elig = eligible[eligible["role"] == label]
        n = len(sub)
        if n:
            ratio = sub["playoff_mpg"] / sub["minutes_per_game"].replace(0, np.nan)
            rows += [
                _row("playoff_scope", "full", label, "n_with_playoff_appearance",
                     float(n), n),
                _row("playoff_scope", "full", label, "median_playoff_to_regular_mpg",
                     float(ratio.median()), n),
                _row("playoff_scope", "full", label, "share_playing_more_in_playoffs",
                     float((ratio > 1.0).mean()), n),
                _row("playoff_scope", "full", label, "median_regular_mpg",
                     float(sub["minutes_per_game"].median()), n),
                _row("playoff_scope", "full", label, "median_playoff_mpg",
                     float(sub["playoff_mpg"].median()), n),
            ]
        if len(elig):
            rows += [
                _row("playoff_scope", "full", label, "n_on_playoff_teams",
                     float(len(elig)), len(elig)),
                _row("playoff_scope", "full", label, "playoff_appearance_rate",
                     float((elig["playoff_games"] > 0).mean()), len(elig)),
            ]
    # The buckets must partition the qualifying population exactly — a player with no role
    # would silently vanish from all three rows while still sitting in the total.
    assigned = int(d["role"].notna().sum())
    rows.append(_row("playoff_scope", "full", "all", "unbucketed_rows",
                     float(len(d) - assigned), len(d)))
    return rows


# ── 9. Is availability a Markov chain? ────────────────────────────────────────

def rotation_players(frame: pd.DataFrame, seasons: list[str],
                     min_mpg: float = ROTATION_MIN_MPG,
                     min_gp_share: float = ROTATION_MIN_GP_SHARE
                     ) -> set[tuple[str, int]]:
    """(season, player_id) pairs that were regulars *last* season.

    Exactly the population `overdispersion` quotes its 22.7x variance ratio on, factored
    out so `serial_structure` can be measured on the same rows. That matching is the whole
    point: the recorded 3.96x clustering figure is measured on the appearance window over
    **all** players while the 22.7x runs on the full window over **rotation** players, so
    dividing one by the other compares two different frames.
    """
    lagged = with_lags(frame, seasons, ["minutes_per_game", "gp_share"], max_lag=1)
    rot = lagged[(lagged["minutes_per_game_lag1"] >= min_mpg)
                 & (lagged["gp_share_lag1"] >= min_gp_share)]
    return set(zip(rot["season"], rot["player_id"]))


def serial_structure(panel: pd.DataFrame, window: str,
                     long_spell_games: int = 10,
                     population: set[tuple[str, int]] | None = None,
                     key_suffix: str = "") -> list[dict]:
    """Game-to-game transition structure, against the 2-state Markov chain it implies.

    Absences are obviously clustered, so an autoregressive availability process is the
    natural model. This measures how far the *simplest* one — a 2-state chain, play/miss,
    with constant transition probabilities — actually gets, because it makes two
    predictions that can be checked directly:

    - **Variance.** A chain with lag-1 autocorrelation ρ = p − q inflates the variance of
      a game count by `(1+ρ)/(1-ρ)` over binomial. Compare that against the ~22.7×
      measured in `overdispersion`: whatever is left over is *between-player*
      heterogeneity, which serial correlation cannot produce and which is what the
      beta-binomial head already models.
    - **Spell lengths.** A constant-hazard chain implies **geometric** absences, so
      `P(spell = 1) = q` and `P(spell ≥ k) = (1-q)^(k-1)` exactly. The observed
      distribution is a mixture, so this is the specific, falsifiable way the simple
      chain is wrong.

    `population` restricts the rows to a set of (season, player_id) pairs and
    `key_suffix` tags the emitted keys, so the same measurement can be reported on the
    **matched** frame the overdispersion figure uses. The unsuffixed rows are unchanged.
    """
    sel = _select_window(panel, window)
    if population is not None:
        sel = sel[[(s, p) in population
                   for s, p in zip(sel["season"], sel["player_id"])]]
    if sel.empty:
        return []
    keys = ["season", "player_id", "team_id"]
    sel = sel.sort_values(keys + ["team_game_index"])

    played = sel["played"].to_numpy()
    prior = sel.groupby(keys, sort=False)["played"].shift()
    seen = prior.notna().to_numpy()
    prior, now = prior.to_numpy()[seen].astype(int), played[seen]
    if seen.sum() < 100:
        return []

    p = float(now[prior == 1].mean())          # P(play | played last game)
    q = float(now[prior == 0].mean())          # P(play | missed last game)
    rho = p - q                                # lag-1 autocorrelation of the chain
    clustering = (1 + rho) / (1 - rho) if rho < 1 else np.nan
    stationary = q / (1 - p + q) if (1 - p + q) > 0 else np.nan

    n = int(seen.sum())
    trans, markov, shape = (f"transition{key_suffix}", f"markov{key_suffix}",
                            f"spell_shape{key_suffix}")
    rows = [
        _row("serial_structure", window, trans, "p_play_given_played", p, n),
        _row("serial_structure", window, trans, "q_play_given_missed", q, n),
        _row("serial_structure", window, trans, "lag1_autocorrelation", rho, n),
        _row("serial_structure", window, markov, "clustering_variance_inflation",
             clustering, n),
        _row("serial_structure", window, markov, "stationary_play_rate", stationary, n),
        _row("serial_structure", window, markov, "observed_play_rate",
             float(played.mean()), n),
    ]

    spells = absence_spells(panel, window)
    if population is not None and not spells.empty:
        spells = spells[[(s, pid) in population
                         for s, pid in zip(spells["season"], spells["player_id"])]]
    if spells.empty:
        return rows
    lengths = spells["spell_games"]
    m = len(lengths)
    # The geometric null the chain implies, given the same q.
    rows += [
        _row("serial_structure", window, shape, "share_single_observed",
             float((lengths == 1).mean()), m),
        _row("serial_structure", window, shape, "share_single_geometric", q, m),
        _row("serial_structure", window, shape,
             f"share_ge{long_spell_games}_observed",
             float((lengths >= long_spell_games).mean()), m),
        _row("serial_structure", window, shape,
             f"share_ge{long_spell_games}_geometric",
             float((1 - q) ** (long_spell_games - 1)), m),
        _row("serial_structure", window, shape, "mean_spell_observed",
             float(lengths.mean()), m),
        _row("serial_structure", window, shape, "mean_spell_geometric",
             1.0 / q if q > 0 else np.nan, m),
    ]
    return rows


# ── Entry point ───────────────────────────────────────────────────────────────

def measure(panel: pd.DataFrame, seasons: list[str], raw_dir: str | Path,
            cfg_eda: dict | None = None) -> pd.DataFrame:
    cfg_eda = cfg_eda or {}
    min_mpg = cfg_eda.get("rotation_min_mpg", ROTATION_MIN_MPG)
    min_gp_share = cfg_eda.get("rotation_min_gp_share", ROTATION_MIN_GP_SHARE)
    bin_width = cfg_eda.get("gp_histogram_bin_width", 8)
    min_coverage = cfg_eda.get("min_status_coverage", MIN_STATUS_COVERAGE)
    min_pairs = cfg_eda.get("min_decomposition_pairs", MIN_DECOMPOSITION_PAIRS)
    n_shuffles = cfg_eda.get("n_shuffles", 20)
    seed = cfg_eda.get("seed", 42)

    frames = {w: season_availability(panel, w) for w in ("appearance", "full")}
    rows = window_bracket(panel, frames)

    for window, frame in frames.items():
        rows += persistence(frame, seasons, window)
        rows += spell_distribution(panel, window)
        rows += carryover(frame, seasons, window)
        ladder, _ = predictor_ladder(frame, seasons, raw_dir, window)
        rows += ladder
        rows += overdispersion(frame, seasons, window, min_mpg, min_gp_share, bin_width)
        rows += decomposition(frame, seasons, window, min_coverage, min_pairs,
                              n_shuffles, seed)
        rows += serial_structure(panel, window)
        # The same measurement on the rows `overdispersion` quotes its 22.7x on. The
        # unsuffixed rows above are untouched — the two are different frames, and the
        # error this exists to prevent is reading them as one.
        rows += serial_structure(panel, window,
                                 population=rotation_players(frame, seasons, min_mpg,
                                                             min_gp_share),
                                 key_suffix="_rotation")

    # Once, not per window: the scope evidence is about season-level role, and the two
    # roster windows do not change a player's regular-season minutes per game.
    playoffs = playoff_workload(seasons, raw_dir)
    rows += playoff_scope(panel, frames["full"], playoffs,
                          playoff_teams(seasons, raw_dir),
                          cfg_eda.get("playoff_min_regular_games",
                                      PLAYOFF_MIN_REGULAR_GAMES))

    return pd.DataFrame(rows)[["measurement", "window", "key", "metric", "n", "value"]]


def run(cfg: dict) -> Path:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_eda = cfg["eda"].get("availability", {})

    panel = build_panel(seasons, raw_dir)
    print(f"Availability panel: {len(panel):,} player-games over "
          f"{panel['season'].nunique()} seasons")

    table = measure(panel, seasons, raw_dir, cfg_eda)

    def pick(measurement: str, window: str, key: str, metric: str) -> float:
        m = table[(table.measurement == measurement) & (table.window == window)
                  & (table.key == key) & (table.metric == metric)]
        return float(m["value"].iloc[0]) if len(m) else float("nan")

    print("\nThe two roster windows bracket the truth; neither separates injury from "
          "roster churn:")
    for w in ("appearance", "full"):
        print(f"  {w:<10} played rate {pick('window_bracket', w, 'panel', 'played_rate'):.3f}"
              f"   mean trailing missed "
              f"{pick('window_bracket', w, 'season_edge', 'mean_trailing_missed'):.2f}")
    print("  Trailing missed is 0 on the appearance window by construction — the window "
          "ends on a game he played.")

    print("\nYear-over-year persistence (full window, season-absorbed, minutes-weighted):")
    pers = table[(table.measurement == "persistence") & (table.window == "full")
                 & (table.metric == "r_within_weighted")]
    print(pers[["key", "n", "value"]].round(3).to_string(index=False))

    print("\nPredicting next-season games-played share — IN-SAMPLE R², a ceiling:")
    ladder = (table[(table.measurement == "predictor_r2") & (table.window == "full")]
              .pivot_table(index="key", columns="metric", values="value")
              .reindex(PREDICTOR_SETS.keys()))
    print(ladder.round(4).to_string())
    print("  Minutes weighting is not the settled choice it is for per-36 rates: games "
          "played is measured\n  exactly, so weighting down-weights the injured seasons "
          "this head exists to predict.")

    print("\nDoes more history help? (the durability-latent null)")
    my = (table[(table.measurement == "multiyear") & (table.window == "full")]
          .pivot_table(index="key", columns="metric", values="value"))
    print(my.round(3).to_string())
    print("  If mean3 does not beat lag1, there is no stable durability signal to extract.")

    print("\nSeason GP against a binomial (established rotation players, full window):")
    for k in ("n", "mean_gp_share", "sd_gp_share", "binomial_sd", "variance_ratio",
              "share_below_60_games", "share_below_41_games"):
        print(f"  {k:<22} {pick('overdispersion', 'full', 'rotation_players', k):.3f}")
    print("  A point estimate cannot represent this; the head must emit a distribution.")

    print("\nIs availability a Markov chain? (appearance window)")
    for k, metric in [("transition", "p_play_given_played"),
                      ("transition", "q_play_given_missed"),
                      ("transition", "lag1_autocorrelation"),
                      ("markov", "clustering_variance_inflation")]:
        print(f"  {metric:<32} "
              f"{pick('serial_structure', 'appearance', k, metric):.3f}")
    obs1 = pick("serial_structure", "appearance", "spell_shape", "share_single_observed")
    geo1 = pick("serial_structure", "appearance", "spell_shape", "share_single_geometric")
    obsl = pick("serial_structure", "appearance", "spell_shape", "share_ge10_observed")
    geol = pick("serial_structure", "appearance", "spell_shape", "share_ge10_geometric")
    print(f"  spells of 1 game    observed {obs1:.3f} vs geometric {geo1:.3f}")
    print(f"  spells of 10+ games observed {obsl:.4f} vs geometric {geol:.4f}")
    print("  Serial correlation is real, but a constant-hazard chain gets the MEAN spell "
          "right and\n  both tails wrong — absences are a mixture of two processes, not "
          "one.")
    c_matched = pick("serial_structure", "full", "markov_rotation",
                     "clustering_variance_inflation")
    print(f"\n  MATCHED to the frame the 22.7x is measured on (full window, established "
          f"rotation\n  players): P(play|played) "
          f"{pick('serial_structure', 'full', 'transition_rotation', 'p_play_given_played'):.4f}, "
          f"P(play|missed) "
          f"{pick('serial_structure', 'full', 'transition_rotation', 'q_play_given_missed'):.4f}, "
          f"C = {c_matched:.2f}.")
    print("  Do NOT divide 22.7 by 3.96: they are different windows AND different "
          "populations, and\n  the composition is additive rather than multiplicative — "
          "inflation = C + rho*(n - C).")

    print("\nDoes splitting absences by reason beat the aggregate? (full window)")
    dec = table[(table.measurement == "decomposition") & (table.window == "full")]
    pairs = pick("decomposition", "full", "coverage", "usable_pairs")
    print(f"  mean status coverage "
          f"{pick('decomposition', 'full', 'coverage', 'mean_status_coverage'):.3f}, "
          f"{pairs:,.0f} usable season pairs")
    if dec[dec.metric == "insufficient_coverage"].empty and pairs > 0:
        inc = (dec[dec.key == "incremental"].set_index("metric")["value"])
        print(f"  R²  prior gp_share {inc.get('r2_prior_gp_share', float('nan')):.4f}"
              f"  → + missed_games {inc.get('r2_plus_missed_games', float('nan')):.4f}"
              f"  → + reason split {inc.get('r2_plus_reason_split', float('nan')):.4f}")
        print(f"  IN-SAMPLE: {inc.get('n_reason_columns', 0):.0f} added columns raise R² "
              f"by {inc.get('r2_reason_split_shuffled_null', float('nan')) - inc.get('r2_prior_gp_share', float('nan')):+.4f} "
              f"on shuffled rows alone (sd {inc.get('r2_reason_split_null_sd', float('nan')):.4f});"
              f"\n  the split is {inc.get('r2_reason_split_above_null', float('nan')):+.4f} "
              f"ABOVE that chance level.")
        vs = (dec[dec.metric == "r_vs_next_gp_share"]
              .set_index("key")["value"].sort_values())
        print("  r vs next-season gp share, by reason:")
        print(vs.round(3).to_string())
    else:
        print("  Not enough backfilled coverage to answer yet — this is a statement "
              "about the\n  backfill, not about the players. Re-run after "
              "`make boxscore-status` completes.")

    print("\nWhy the fitting frame is regular season only — the sign flip "
          "(unweighted, by regular-season role):")
    scope = table[table.measurement == "playoff_scope"]
    piv = (scope[scope.key != "all"]
           .pivot_table(index="key", columns="metric", values="value")
           .reindex([k for k in PLAYOFF_ROLE_LABELS
                     if k in set(scope["key"])]))
    cols = ["n_with_playoff_appearance", "median_regular_mpg", "median_playoff_mpg",
            "median_playoff_to_regular_mpg", "share_playing_more_in_playoffs",
            "n_on_playoff_teams", "playoff_appearance_rate"]
    print(piv[[c for c in cols if c in piv.columns]].round(3).to_string())
    starter = pick("playoff_scope", "full", "starter_24plus",
                   "median_playoff_to_regular_mpg")
    bench = pick("playoff_scope", "full", "bench_lt12", "median_playoff_to_regular_mpg")
    unbucketed = pick("playoff_scope", "full", "all", "unbucketed_rows")
    print(f"  Starters play MORE in the playoffs ({starter:.3f}x) while the bench plays "
          f"about half ({bench:.3f}x).\n  One pooled playoff coefficient would have to fit "
          "both, and the bench is numerous enough to\n  drag it toward compression — which "
          "is why playoff rows are features, never targets.")
    print(f"  Rotations shorten too: the appearance rate rises across the same buckets.")
    if unbucketed:
        print(f"  ⚠️  {unbucketed:,.0f} qualifying player-seasons fall in no role bucket — "
              "the buckets must partition.")

    dest = out_dir / "availability_profile.csv"
    table.to_csv(dest, index=False)
    print(f"\nAvailability profile: {len(table):,} measurement rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
