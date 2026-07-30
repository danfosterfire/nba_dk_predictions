"""Game length in minutes — the trials denominator for the minutes head.

`min` is not a count. A player's minutes are bounded above by the length of the game he
played in, so the natural likelihood is successes-out-of-trials with trials = that
length (see the component contract in `CLAUDE.md` and `docs/predictions-plan.md`). The
denominator is 48 in regulation and 5 more per overtime period, and **nothing in the
raw game logs states it** — which is why the prior attempt truncated minutes at 48 and
dropped every overtime game instead.

It does not need to be fetched. Five players are on the court for every second of a
basketball game, so a team's summed minutes are exactly `5 x game length`, and the logs
carry minutes to the second (98.4% of rows are non-integer). Summing and dividing by
five recovers the length.

**The two teams in a game are two independent estimates of the same quantity, and that
is the validation** — not a coverage percentage. Per `CLAUDE.md`, an unmatched or
agreement rate that can be bought by fabricating rows is worthless; this one cannot be,
because the two sums are computed from disjoint sets of players and a construction
error would have to corrupt both identically to go unnoticed.

Measured over all 37,986 games (regular season + playoffs, 1996-97 -> 2025-26):
**0 games where the two teams disagree**, 100% of team-games within 1 minute of the
48 + 5k grid and 99.9% exactly on it. The residual is box-score rounding — worst in the
early sample (max 0.617 min, 2002-03) and identically zero from 2014-15 on. Snapping to
the grid is therefore a rounding correction, not a guess: the largest observed residual
is 0.617 against a 2.5-minute decision boundary, a 4x margin.

**Build this from the raw logs, never from `component_targets.parquet`.**
`preprocess.clean` drops players below `data.min_games`, and a team sum over a filtered
frame is missing whole players' minutes — it understates the length, silently, while
still snapping to a plausible grid point. `team_minutes` therefore takes raw logs, and
`derive_game_length` flags any game whose residual exceeds the tolerance rather than
snapping it anyway.

This module covers **both** season types, unlike the fitting frames. Game length is a
property of the game rather than a modelling target, so withholding playoff rows would
buy nothing and would leave `prior_playoff_minutes` without a denominator.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import ALL_SEASON_TYPES, load_raw

REGULATION_MINUTES = 48.0
OVERTIME_MINUTES = 5.0
PLAYERS_ON_COURT = 5
MAX_OVERTIMES = 6

# Half the gap between grid points is 2.5 minutes; the largest rounding residual ever
# observed is 0.617. 1.5 sits well outside the noise and well inside the boundary, so a
# row that trips it is a data problem (missing players), not a rounding artifact.
SNAP_TOLERANCE = 1.5

LENGTH_GRID = REGULATION_MINUTES + OVERTIME_MINUTES * np.arange(MAX_OVERTIMES + 1)


def load_game_logs(raw_dir: str | Path) -> pd.DataFrame:
    """Raw player-game rows for **both** season types, lower-cased.

    This is the one place that deliberately asks for playoff rows too. Game length is a
    property of a game, not a modelling target, so there is no reason to withhold it —
    and the playoff lengths are what `prior_playoff_minutes` needs to be interpretable.
    Consumers filter on `season_type`; the fitting frames take `regular` only.
    """
    logs = load_raw(raw_dir, season_type=ALL_SEASON_TYPES,
                    columns=["GAME_ID", "TEAM_ID", "MIN"])
    return logs.rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id",
                                "MIN": "min"})


def team_minutes(logs: pd.DataFrame) -> pd.DataFrame:
    """Summed minutes per (season, season_type, game_id, team_id).

    `min` is coerced and zero-filled rather than dropped: a player who appeared for
    under 30 seconds can log 0.0, and dropping him would not change the sum but would
    change `n_players`, which is the diagnostic for a truncated frame.
    """
    df = logs.copy()
    df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
    keys = ["season", "season_type", "game_id", "team_id"]
    out = df.groupby(keys, as_index=False).agg(
        team_minutes=("min", "sum"), n_players=("min", "size")
    )
    out["length_raw"] = out["team_minutes"] / PLAYERS_ON_COURT
    return out


def snap_to_grid(length: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nearest 48 + 5k grid point, and the signed residual against it."""
    length = np.asarray(length, dtype=float)
    nearest = LENGTH_GRID[np.abs(length[:, None] - LENGTH_GRID).argmin(axis=1)]
    return nearest, length - nearest


def derive_game_length(logs: pd.DataFrame,
                       tolerance: float = SNAP_TOLERANCE) -> pd.DataFrame:
    """One row per game: length in minutes, overtime count, and the audit columns.

    `reliable` is the flag consumers should gate on. It is False when the two teams
    disagree, when a game does not have exactly two team rows, or when either team's
    raw sum sits further than `tolerance` from the grid — the signature of a frame that
    has lost players rather than of a game that ran long.
    """
    tm = team_minutes(logs)
    tm["length_snapped"], tm["residual"] = snap_to_grid(tm["length_raw"].values)

    keys = ["season", "season_type", "game_id"]
    g = tm.groupby(keys)
    out = g.agg(
        n_teams=("team_id", "nunique"),
        n_players=("n_players", "sum"),
        length_raw_mean=("length_raw", "mean"),
        max_abs_residual=("residual", lambda s: float(np.abs(s).max())),
        distinct_lengths=("length_snapped", "nunique"),
        game_length=("length_snapped", "max"),
    ).reset_index()

    out["teams_agree"] = out["distinct_lengths"] == 1
    out["n_overtimes"] = ((out["game_length"] - REGULATION_MINUTES)
                          / OVERTIME_MINUTES).round().astype(int)
    out["n_periods"] = 4 + out["n_overtimes"]
    out["reliable"] = (out["teams_agree"] & (out["n_teams"] == 2)
                       & (out["max_abs_residual"] <= tolerance))
    return out.drop(columns=["distinct_lengths"])


def coverage(lengths: pd.DataFrame) -> pd.DataFrame:
    """Per-season audit: agreement, worst rounding residual, overtime rate."""
    g = lengths.groupby(["season", "season_type"])
    out = g.agg(
        games=("game_length", "size"),
        reliable=("reliable", "mean"),
        teams_disagree=("teams_agree", lambda s: int((~s).sum())),
        max_abs_residual=("max_abs_residual", "max"),
        exact_on_grid=("max_abs_residual", lambda s: float((s < 1e-3).mean())),
        ot_rate=("n_overtimes", lambda s: float((s > 0).mean())),
    ).reset_index()
    out.insert(0, "analysis", "derivation")
    return out


# ── Feasibility: is `min ~ Binomial(game_length, .)` well posed everywhere? ────
#
# The derivation above validates the *lengths*. It says nothing about whether the
# specification the lengths exist for actually holds on every row, and that is a separate,
# stronger claim: a single player-game with `min > game_length` would make the binomial
# likelihood undefined there and force a clip or a boundary hack.
#
# So this section is a **check, not a metric**. `assert_feasible` raises; a nonzero violation
# count is a build failure. Reporting it as a percentage would be the same mistake as
# reporting an unmatched rate — a number that reads like success while the thing it measures
# is broken.

# The two sides store game ids differently, so both go through `pad_game_id`: the game logs
# (and therefore `game_length.parquet` and `component_targets.parquet`) carry int64, the
# box-score files carry zero-padded 10-character strings. Normalizing both is cheap and
# means a future consumer joining from either side cannot silently match nothing.
FEASIBILITY_KEY = "_game_key"


def join_lengths(targets: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Attach each player-game's game length, keyed on a normalized game id."""
    from src.data.boxscore_status import pad_game_id

    left = targets[["season", "player_id", "game_id", "min"]].copy()
    left[FEASIBILITY_KEY] = left["game_id"].map(pad_game_id)
    right = lengths[["game_id", "game_length", "n_overtimes"]].copy()
    right[FEASIBILITY_KEY] = right["game_id"].map(pad_game_id)
    right = right.drop(columns="game_id").drop_duplicates(subset=FEASIBILITY_KEY)
    return left.merge(right, on=FEASIBILITY_KEY, how="left")


def minutes_feasibility(targets: pd.DataFrame, lengths: pd.DataFrame) -> pd.DataFrame:
    """Join coverage, violations and the ratio ceiling, overall and per season.

    `max_min_over_length` reaching exactly 1.0 is the interesting part: someone played every
    minute of a game, so the specification is tight rather than merely satisfied — there is
    no headroom hiding a construction error.
    """
    joined = join_lengths(targets, lengths)
    joined["ratio"] = joined["min"] / joined["game_length"]

    def _block(label: str, season_type: str, df: pd.DataFrame) -> dict:
        matched = df[df["game_length"].notna()]
        return {
            "analysis": "feasibility", "season": label, "season_type": season_type,
            "player_games": int(len(df)),
            "join_coverage": float(len(matched) / len(df)) if len(df) else np.nan,
            "unmatched": int(len(df) - len(matched)),
            "violations": int((matched["min"] > matched["game_length"]).sum()),
            "max_min_over_length": float(matched["ratio"].max()) if len(matched) else np.nan,
            "player_games_above_regulation": int((df["min"] > REGULATION_MINUTES).sum()),
            "max_minutes": float(df["min"].max()) if len(df) else np.nan,
        }

    rows = [_block("all", "regular", joined)]
    for season, block in joined.groupby("season", sort=True):
        rows.append(_block(season, "regular", block))
    return pd.DataFrame(rows)


def assert_feasible(table: pd.DataFrame) -> None:
    """Raise unless every player-game joined and none exceeds its game length.

    Both halves matter and they fail differently. An unmatched player-game means the join is
    broken — which `pad_game_id` exists to prevent and which would otherwise show up only as
    a silently smaller fitting frame. A violation means the binomial support is wrong.
    """
    overall = table[(table["analysis"] == "feasibility") & (table["season"] == "all")]
    if overall.empty:
        raise AssertionError("no feasibility row to check")
    row = overall.iloc[0]
    if int(row["violations"]):
        worst = float(row["max_min_over_length"])
        raise AssertionError(
            f"{int(row['violations']):,} player-games have min > game_length "
            f"(worst ratio {worst:.4f}). `min ~ Binomial(game_length, .)` is not well posed "
            "on those rows, so the minutes head would need a clip. This is a build failure, "
            "not a metric — check that game_length was derived from the RAW logs rather "
            "than from the filtered component_targets frame.")
    if int(row["unmatched"]):
        raise AssertionError(
            f"{int(row['unmatched']):,} of {int(row['player_games']):,} player-games have no "
            "game length. Game ids are int64 in the game logs and zero-padded strings in the "
            "box-score files — join through `boxscore_status.pad_game_id`.")


def run(cfg: dict) -> Path:
    logs = load_game_logs(cfg["data"]["raw_dir"])
    lengths = derive_game_length(logs)

    out_dir = Path(cfg["data"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / "game_length.parquet"
    lengths.to_parquet(dest, index=False)

    cov = coverage(lengths)

    # ── the feasibility check ────────────────────────────────────────────────
    targets_path = out_dir / "component_targets.parquet"
    feasible = pd.DataFrame()
    if targets_path.exists():
        targets = pd.read_parquet(targets_path,
                                  columns=["season", "player_id", "game_id", "min"])
        feasible = minutes_feasibility(targets, lengths)
        assert_feasible(feasible)
        cov = pd.concat([cov, feasible], ignore_index=True)

    eda_dir = Path(cfg["eda"]["output_dir"])
    eda_dir.mkdir(parents=True, exist_ok=True)
    cov_dest = eda_dir / "game_length_coverage.csv"
    cov.to_csv(cov_dest, index=False)

    disagree = int((~lengths["teams_agree"]).sum())
    unreliable = int((~lengths["reliable"]).sum())
    print(f"Game length: {len(lengths):,} games → {dest}")
    print(f"  the validation — games where the two teams disagree: {disagree}")
    print(f"  unreliable (disagreement, wrong team count, or residual > "
          f"{SNAP_TOLERANCE}): {unreliable}")
    print(f"  worst rounding residual: {lengths['max_abs_residual'].max():.3f} min "
          f"against a {OVERTIME_MINUTES / 2:.1f} min decision boundary")
    dist = lengths["n_overtimes"].value_counts().sort_index()
    print("  overtime periods: " +
          ", ".join(f"{k}→{v:,}" for k, v in dist.items()))
    print(f"  overall OT rate: {(lengths['n_overtimes'] > 0).mean():.2%}")

    if feasible.empty:
        print(f"  skipping the feasibility check — {targets_path} not found "
              "(run `make component-targets`)")
    else:
        row = feasible[feasible["season"] == "all"].iloc[0]
        print(f"\nFeasibility of `min ~ Binomial(game_length, ·)` on "
              f"{int(row['player_games']):,} player-games:")
        print(f"  join coverage: {row['join_coverage']:.1%} "
              f"({int(row['unmatched']):,} unmatched)")
        print(f"  rows with min > game_length: {int(row['violations'])} "
              "— asserted, not reported")
        print(f"  max min / game_length: {row['max_min_over_length']:.4f} "
              "(1.0000 = someone played every minute, so the bound is tight)")
        print(f"  player-games above {REGULATION_MINUTES:g} minutes: "
              f"{int(row['player_games_above_regulation']):,}, "
              f"observed max {row['max_minutes']:.1f} — truncating at "
              f"{REGULATION_MINUTES:g} would censor these")
    print(f"Coverage: {len(cov):,} rows → {cov_dest}")
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
