"""Clean and merge raw game log CSVs into a single processed DataFrame."""

from pathlib import Path

import pandas as pd
import yaml


KEEP_COLS = [
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GAME_ID",
    "GAME_DATE",
    "MATCHUP",
    "WL",
    "MIN",
    "PTS",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "FGA",
    "FGM",
    "FG3A",
    "FG3M",
    "FTA",
    "FTM",
    "PLUS_MINUS",
]

RENAME = {c: c.lower() for c in KEEP_COLS}


REGULAR_SEASON = "regular"
PLAYOFFS = "playoffs"
PRE_SEASON = "pre_season"
# "Every type that can be a TARGET ROW" — regular season and playoffs, not preseason.
# See `load_raw` for why the odd one out is excluded by construction.
ALL_SEASON_TYPES = "all"

# Which file prefix maps to which kind. `game_logs_{season}.csv` has no prefix and is the
# regular season; everything else announces itself.
_LOG_PREFIXES = {"playoffs_": PLAYOFFS, "pre_season_": PRE_SEASON}


# ── The evaluation split, for the measurements that feed the simulator ────────
#
# Every fitted head splits temporally by target season: the trailing `TEST_SEASONS` are
# held out, and the two before them are the validation split carved out of the training
# half. That lives in `models.availability.TEST_SEASONS` and its copies, which describe a
# *design matrix*.
#
# This is the same number for a different population. Four quantities the simulator will
# take as direct inputs — the residual copula, the game-level minutes dispersion, the
# block variance inflation and the bonus overdispersion — are **calibrations, not fits**,
# so nothing stops them from being measured over every season including the held-out ones.
# Doing that would calibrate the simulator on the seasons it is later scored against.
# `fit_window` is the one-line fix, and the artifacts carry every window so the size of
# the difference is on disk rather than assumed.
#
# **There are three windows, not two, and the third was added on 2026-08-08.** `train_val`
# is clean for a *test-split* readout and not for a *validation* one — it contains 2022-23
# and 2023-24, which is exactly what the realized backtest in `docs/simulations-plan.md`
# scores against. Once `make posteriors` started emitting coefficients per window, a
# backtest could have clean coefficients and a noise shape calibrated on the seasons it was
# scoring, which is a leak wearing the previous fix's clothes. So the windows now mirror
# the three consumers one-for-one: `train` for anything scored on validation, `train_val`
# for the one-shot test readout, `full` for production.
#
# `tests/test_preprocess.py` pins this against the model modules' copies: two definitions
# of "which seasons are held out" that could disagree is a worse failure than a duplicated
# constant, because a disagreement is silent on both sides.
TEST_SEASONS = 2

FULL_WINDOW = "full"
TRAIN_VAL_WINDOW = "train_val"
TRAIN_WINDOW = "train"
# Widest first, so a table pivoted on this column reads left-to-right as "progressively
# more held out".
FIT_WINDOWS = [FULL_WINDOW, TRAIN_VAL_WINDOW, TRAIN_WINDOW]

# How many trailing season labels each window drops. `train` drops twice `TEST_SEASONS`
# because the validation split is carved out of the training half — the same two-step
# `models.held_out.selection_split` performs, expressed here as one count.
_WINDOW_DROP = {FULL_WINDOW: 0, TRAIN_VAL_WINDOW: 1, TRAIN_WINDOW: 2}


def fit_window(frame: pd.DataFrame, window: str = TRAIN_VAL_WINDOW,
               test_seasons: int = TEST_SEASONS) -> pd.DataFrame:
    """Restrict `frame` to a fit window, keyed on its `season` column.

    `full` returns everything; `train_val` drops the trailing `test_seasons` season labels;
    `train` drops twice that many, so the validation seasons go too. All three derive the
    labels the same way `split_seasons` does — sort the labels present, take the last N —
    rather than hard-coding them, so a change to the data window moves everything together.

    **Which one to consume is decided by what the number will be scored against**, not by
    which is widest. A statistic used while scoring 2022-23 / 2023-24 must come from
    `train`, because `train_val` contains those seasons; `train_val` is for the one-shot
    test readout; `full` is production.

    Note this keys on the season a row is *from*, which for these calibration frames is
    the season being held out. A design matrix's `season` is its *target* season and its
    features describe S-1; the two coincide here because these frames are realized
    outcomes, not lagged features.
    """
    if window not in FIT_WINDOWS:
        raise ValueError(f"unknown fit window {window!r}; expected one of {FIT_WINDOWS}")
    if window == FULL_WINDOW or not test_seasons:
        return frame
    test_seasons = test_seasons * _WINDOW_DROP[window]
    order = sorted(frame["season"].unique())
    # An empty fitting half is the failure this repo has already shipped once in another
    # costume — a frame that silently shrank rather than raising. Every downstream
    # statistic here degrades to NaN, which reads as "no dependence" rather than as "no
    # data", so it has to be loud. `models.held_out.selection_split` raises for the same
    # reason on the same shape of input.
    if len(order) <= test_seasons:
        raise ValueError(
            f"cannot hold out {test_seasons} of {len(order)} seasons ({order}) — the "
            f"{window!r} window would be empty. Every correlation measured on "
            "it would be NaN, which reads as a null rather than as missing data.")
    return frame[~frame["season"].isin(set(order[-test_seasons:]))]


def held_out_seasons(frame: pd.DataFrame,
                     test_seasons: int = TEST_SEASONS,
                     window: str = TRAIN_VAL_WINDOW) -> list[str]:
    """The season labels `fit_window` drops for `window` — for printing what was excluded."""
    if window not in FIT_WINDOWS:
        raise ValueError(f"unknown fit window {window!r}; expected one of {FIT_WINDOWS}")
    dropped = test_seasons * _WINDOW_DROP[window]
    return sorted(frame["season"].unique())[-dropped:] if dropped else []


def _parse_log_filename(stem: str) -> tuple[str, str]:
    """`game_logs_2021_22` -> ("regular", "2021-22");
    `game_logs_playoffs_2021_22` -> ("playoffs", "2021-22");
    `game_logs_pre_season_2021_22` -> ("pre_season", "2021-22").

    All three season types share the same `season` label. Deriving the season from the slug
    without stripping the prefix produced the pseudo-season `playoffs-2021-22`, which
    silently doubled the season count for every downstream
    `groupby(["player_id", "season"])` — and an unrecognized prefix does not just invent a
    label, it also returns `REGULAR_SEASON`, so the rows arrive in the *default* frame.
    That is why every new prefix belongs in `_LOG_PREFIXES` rather than in a caller's glob.
    """
    slug = stem.replace("game_logs_", "")
    for prefix, kind in _LOG_PREFIXES.items():
        if slug.startswith(prefix):
            return kind, slug[len(prefix):].replace("_", "-")
    return REGULAR_SEASON, slug.replace("_", "-")


def load_raw(raw_dir: str | Path, season_type: str = REGULAR_SEASON,
             columns: list[str] | None = None) -> pd.DataFrame:
    """Raw player-game rows, tagged with `season` and `season_type`.

    **Defaults to regular season only, and that is the modelling decision, not a
    convenience.** `game_logs_*.csv` also matches the 30 `game_logs_playoffs_*.csv`
    files, so this used to pull playoff rows in unannounced. Playoff games are excluded
    from every fitting frame because the DK best-ball contest ends 4/4 — before the
    playoffs begin — and because playoff minutes are a *role interaction with a sign
    change*, not a level shift (see `CLAUDE.md`). The playoff logs stay on disk and stay
    useful as **prior-season workload features**, which is a different role entirely.

    `season_type` is one of "regular", "playoffs", "pre_season" or "all". Anything else
    raises rather than silently returning an empty frame.

    **"all" means regular + playoffs, and deliberately excludes the preseason.** A
    preseason game is a *forecast covariate* for the season about to start and never a
    target row (`docs/preseason-plan.md`), so it must be asked for by name. The default
    the other way round would be silent: `src/features/game_length.py` reads "all" to
    derive every game's length from summed team minutes, and would have absorbed ~70
    exhibition games per season into an artifact whose whole claim is that its two
    independent estimates disagree on none of them.
    """
    if season_type not in (REGULAR_SEASON, PLAYOFFS, PRE_SEASON, ALL_SEASON_TYPES):
        raise ValueError(
            "season_type must be one of 'regular', 'playoffs', 'pre_season', 'all'; "
            f"got {season_type!r}")

    wanted = ({REGULAR_SEASON, PLAYOFFS} if season_type == ALL_SEASON_TYPES
              else {season_type})
    raw_dir = Path(raw_dir)
    frames = []
    for f in sorted(raw_dir.glob("game_logs_*.csv")):
        kind, season = _parse_log_filename(f.stem)
        if kind not in wanted:
            continue
        df = pd.read_csv(f, usecols=columns, low_memory=False)
        df["season"] = season
        df["season_type"] = kind
        frames.append(df)
    if not frames:
        raise FileNotFoundError(
            f"No {season_type} game log CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


def compute_dk_pts(df: pd.DataFrame) -> pd.Series:
    """DraftKings NBA fantasy points scoring."""
    dd_cats = (df[["pts", "reb", "ast", "stl", "blk"]] >= 10).sum(axis=1)
    bonus = dd_cats.map({0: 0, 1: 0, 2: 1.5, 3: 4.5, 4: 4.5, 5: 4.5})
    return (
        df["pts"] * 1.0
        + df["fg3m"] * 0.5
        + df["reb"] * 1.25
        + df["ast"] * 1.5
        + df["stl"] * 2.0
        + df["blk"] * 2.0
        + df["tov"] * -0.5
        + bonus
    )


def clean(df: pd.DataFrame, min_games: int = 20) -> pd.DataFrame:
    # Keep only the columns we care about (drop silently if absent); preserve season
    cols = [c for c in KEEP_COLS if c in df.columns]
    extra = [c for c in ("season", "season_type") if c in df.columns]
    df = df[cols + extra].rename(columns=RENAME).copy()

    df["game_date"] = pd.to_datetime(df["game_date"])
    df["home"] = df["matchup"].str.contains("vs\\.").astype(int)
    df["min"] = pd.to_numeric(df["min"], errors="coerce")
    df = df.dropna(subset=["min", "pts", "reb", "ast"])

    df["dk_pts"] = compute_dk_pts(df)

    # Drop players with too few games
    counts = df.groupby("player_id")["game_id"].transform("count")
    df = df[counts >= min_games]

    return df.sort_values(["player_id", "game_date"]).reset_index(drop=True)


def run(cfg: dict) -> Path:
    df_raw = load_raw(cfg["data"]["raw_dir"])
    df = clean(df_raw, min_games=cfg["data"]["min_games"])

    out_dir = Path(cfg["data"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "game_logs.parquet"
    df.to_parquet(dest, index=False)
    print(f"Processed {len(df):,} rows → {dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
