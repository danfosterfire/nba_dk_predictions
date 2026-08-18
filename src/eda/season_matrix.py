"""Build one row per (player_id, season) from the season-level raw CSV families.

This is the foundation the rest of the EDA consumes. Two tiers are produced:

    Tier A  30 seasons (1996-97+), box-score derived families only
    Tier B  13 seasons (2013-14+), Tier A plus tracking / hustle / estimated

Counting stats are converted to per-36 so that style is comparable across
players with different playing time; rates and percentages pass through. MIN,
GP and total minutes are carried as explicit volume columns so downstream PCA
can hold them out — otherwise PC1 is just "minutes played".

**Two consumers, two row sets.** The qualified matrices (`GP>=20 & MIN>=10`) serve
the PCA, where per-36 rates from a handful of garbage-time minutes would be wild
outliers. Roster aggregation wants the opposite: a 14-minute-a-night bench player is
a real teammate consuming real minutes, and dropping him biases every team aggregate
toward veterans. So `run` also writes an unfiltered `season_matrix_roster_tier*`
frame; its consumers are expected to down-weight thin seasons rather than exclude
them (see `src/features/team_context.py::reliability`).
"""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import MANIFEST_NAME, _season_start_year, _slug, nbastats_dir
from src.data.preprocess import compute_dk_pts


# ── Column classification ─────────────────────────────────────────────────────

# Dropped from every family: ranks are derived, the rest are bookkeeping.
COMMON_DROP = ("NICKNAME", "TEAM_COUNT", "GROUP_SET")

# Dropped from every family, base included — these are identity and team-context
# columns, carried once from the base file by `read_identity` rather than repeated
# per family. Leaving them in a family's stat set would silently per-36 them.
IDENTITY_DROP = (
    "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "AGE", "GP", "G", "W", "L", "W_PCT",
)

# Identity and volume columns taken from the base family, in output order.
ID_COLS = ["player_id", "player_name", "season", "season_start_year",
           "team_id", "team_abbreviation", "age"]
VOLUME_COLS = ["gp", "min", "min_total"]

# Substrings marking a column as already rate-like — passes through un-normalized.
# Covers *_PCT / PCT_*, *_RATING, *_RATIO, *_FREQUENCY, PACE, PIE, AVG_SPEED,
# AVG_SEC_PER_TOUCH, PTS_PER_TOUCH, AST_TO_PASS_PCT, ...
RATE_MARKERS = ("_PCT", "PCT_", "_FREQUENCY", "_RATING", "_RATIO", "PACE", "PIE",
                "AVG_", "_PER_", "AST_TO")

# Neither counting stats nor rates — physical and draft attributes that must
# never be divided by minutes.
NEVER_SCALE = ("PLAYER_HEIGHT_INCHES", "PLAYER_WEIGHT",
               "DRAFT_YEAR", "DRAFT_ROUND", "DRAFT_NUMBER")


def _is_rate(col: str) -> bool:
    """True if `col` is already normalized and must not be converted to per-36."""
    if col in NEVER_SCALE:
        return True
    return any(m in col for m in RATE_MARKERS)


# ── Family registry ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Family:
    """One raw CSV family and how to fold it into the season matrix.

    `drop` lists columns this family duplicates from an earlier family — the
    same stat measured the same way. They are removed rather than kept under a
    distinct prefix so PCA does not double-count a direction.
    """

    name: str                                  # also the filename prefix
    prefix: str                                # column prefix, e.g. "adv"
    tier: str                                  # "A" (30 seasons) or "B" (2013-14+)
    first_year: int = 1996
    header_rows: int = 1
    has_min: bool = True                       # carries its own MIN for the per-36 basis
    drop: tuple[str, ...] = field(default_factory=tuple)


FAMILIES: list[Family] = [
    # ── Tier A: box-score derived, all 30 seasons ────────────────────────────
    Family("player_stats_base", "bas", "A", drop=("WNBA_FANTASY_PTS",)),
    Family("player_stats_advanced", "adv", "A", drop=(
        # FG columns and sp_work_* duplicate base / the plain rating columns
        "FGM", "FGA", "FG_PCT", "FGM_PG", "FGA_PG",
        "sp_work_OFF_RATING", "sp_work_DEF_RATING", "sp_work_NET_RATING", "sp_work_PACE",
    )),
    Family("player_stats_defense", "def", "A"),
    Family("player_stats_misc", "misc", "A", drop=(
        # OPP_PTS_* are carried by defense; the rest duplicate base
        "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE", "OPP_PTS_FB", "OPP_PTS_PAINT",
        "BLK", "BLKA", "PF", "PFD", "NBA_FANTASY_PTS",
    )),
    Family("player_stats_scoring", "sco", "A", drop=("FGM", "FGA", "FG_PCT")),
    Family("player_stats_usage", "usg", "A", drop=("USG_PCT",)),  # advanced has it
    Family("player_bio_stats", "bio", "A", has_min=False, drop=(
        # Keep only physique and draft position; every stat here repeats base/advanced
        "PLAYER_HEIGHT", "COLLEGE", "COUNTRY",
        "PTS", "REB", "AST", "NET_RATING",
        "OREB_PCT", "DREB_PCT", "USG_PCT", "TS_PCT", "AST_PCT",
    )),
    Family("player_clutch", "clu", "A", drop=("WNBA_FANTASY_PTS",)),
    Family("player_shot_locations", "sl", "A", header_rows=2, has_min=False),

    # ── Tier B: tracking era, 2013-14+ ───────────────────────────────────────
    Family("player_pt_shot", "ptsh", "B", first_year=2013, has_min=False, drop=(
        # Only the shot-mix frequencies are new; the totals duplicate base
        "PLAYER_LAST_TEAM_ID", "PLAYER_LAST_TEAM_ABBREVIATION",
        "FGM", "FGA", "FG_PCT", "FG2M", "FG2A", "FG2_PCT", "FG3M", "FG3A", "FG3_PCT",
    )),
    Family("player_hustle", "hus", "B", first_year=2015),
    Family("player_estimated_metrics", "est", "B", first_year=2014, drop=(
        # advanced already carries these E_* columns
        "E_OFF_RATING", "E_DEF_RATING", "E_NET_RATING", "E_TOV_PCT", "E_USG_PCT", "E_PACE",
    )),
    Family("player_tracking_speeddistance", "spd", "B", first_year=2013, drop=("DIST_FEET",)),
    Family("player_tracking_possessions", "poss", "B", first_year=2013, drop=("POINTS",)),
    Family("player_tracking_catchshoot", "cs", "B", first_year=2013),
    Family("player_tracking_pullupshot", "pu", "B", first_year=2013),
    Family("player_tracking_drives", "drv", "B", first_year=2013),
    Family("player_tracking_passing", "pass", "B", first_year=2013, drop=("AST",)),
    # The touch families each repeat possessions' TOUCHES and its own touch count
    Family("player_tracking_elbowtouch", "elb", "B", first_year=2013,
           drop=("TOUCHES", "ELBOW_TOUCHES")),
    Family("player_tracking_posttouch", "post", "B", first_year=2013,
           drop=("TOUCHES", "POST_TOUCHES")),
    Family("player_tracking_painttouch", "pnt", "B", first_year=2013,
           drop=("TOUCHES", "PAINT_TOUCHES")),
]

TIER_FIRST_YEAR = {"A": 1996, "B": 2013}


# ── Per-36 normalization ──────────────────────────────────────────────────────

def per36_divisor(per_mode: str) -> str:
    """How a file's counting stats reach per-36, given the per_mode it was fetched in.

    `MIN` always carries the same basis as the stats beside it, so `stat / MIN * 36`
    is correct for both PerGame (both per-game) and Totals (both season totals).
    PerMinute is the exception: stats are already per-minute while MIN stays a
    season total, so it is a plain scale-up.
    """
    if per_mode in ("PerGame", "Totals"):
        return "min"
    if per_mode == "PerMinute":
        return "scale"
    raise ValueError(
        f"Cannot convert per_mode={per_mode!r} to per-36: it carries no minutes basis. "
        f"Delete the affected CSV and re-run the fetch to land on another mode."
    )


def to_per36(df: pd.DataFrame, cols: list[str], minutes: pd.Series, per_mode: str = "PerGame") -> None:
    """Convert `cols` of `df` to per-36 in place, leaving rate columns untouched."""
    mode = per36_divisor(per_mode)
    # np.nan, not pd.NA — the latter promotes the Series to object dtype, which
    # then silently drops the column out of `feature_cols`. Zero minutes happens
    # for real in the clutch family (a player with no clutch time has no rate).
    safe_minutes = minutes.replace(0.0, np.nan)
    for c in cols:
        if _is_rate(c):
            continue
        if mode == "scale":
            df[c] = df[c] * 36.0
        else:
            df[c] = df[c] / safe_minutes * 36.0


# ── Loading ───────────────────────────────────────────────────────────────────

def read_family(fam: Family, season: str, raw_dir: Path) -> pd.DataFrame | None:
    """Read one family for one season, or None if the file is absent/empty.

    Returns columns keyed on PLAYER_ID with every kept stat prefixed, already
    converted to per-36 where applicable.
    """
    path = nbastats_dir(raw_dir) / f"{fam.name}_{_slug(season)}.csv"
    if not path.exists():
        return None

    if fam.header_rows == 2:
        df = _read_shot_locations(path)
    else:
        df = pd.read_csv(path)
    if df.empty:
        return None

    df = df.drop(columns=[c for c in df.columns if c.endswith("_RANK")], errors="ignore")
    df = df.drop(columns=list(COMMON_DROP) + list(fam.drop) + list(IDENTITY_DROP),
                 errors="ignore")
    df = df.drop_duplicates(subset="PLAYER_ID", keep="first")
    return df


BASE_FAMILY = FAMILIES[0]


def read_identity(season: str, raw_dir: Path, min_gp: int, min_minutes: float) -> pd.DataFrame:
    """Identity and volume columns for a season's qualified players, from base.

    `TEAM_ABBREVIATION` is the player's *last* team — the season files carry one
    row per player, so a traded player's earlier teams are not represented here.
    """
    path = nbastats_dir(raw_dir) / f"{BASE_FAMILY.name}_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame()

    base = pd.read_csv(path).drop_duplicates(subset="PLAYER_ID", keep="first")
    base = base[(base["GP"] >= min_gp) & (base["MIN"] >= min_minutes)]
    return pd.DataFrame({
        "player_id": base["PLAYER_ID"].values,
        "player_name": base["PLAYER_NAME"].values,
        "season": season,
        "season_start_year": _season_start_year(season),
        "team_id": base["TEAM_ID"].values,
        "team_abbreviation": base["TEAM_ABBREVIATION"].values,
        "age": base["AGE"].values,
        "gp": base["GP"].values,
        "min": base["MIN"].values,
        "min_total": (base["MIN"] * base["GP"]).values,
    })


def _read_shot_locations(path: Path) -> pd.DataFrame:
    """Read the one family with a two-row header, flattening to `zone_stat` names.

    ('Restricted Area', 'FGA') → 'RESTRICTED_AREA_FGA'; the identity columns sit
    under an 'Unnamed: N_level_0' top level and keep their bare names.
    """
    df = pd.read_csv(path, header=[0, 1])
    flat = []
    for top, sub in df.columns:
        if str(top).startswith("Unnamed"):
            flat.append(str(sub))
        else:
            zone = str(top).upper().replace(" ", "_").replace("-", "_")
            zone = zone.replace("(", "").replace(")", "")
            flat.append(f"{zone}_{sub}")
    df.columns = flat
    return df


def load_manifest(raw_dir: Path) -> dict[tuple[str, str], str]:
    """Map (family, season) → per_mode from the fetch manifest.

    Files fetched before the manifest existed are absent from it and default to
    PerGame, which is what fetch.py hardcoded at the time.
    """
    path = raw_dir / MANIFEST_NAME
    if not path.exists():
        return {}
    m = pd.read_csv(path)
    return {(r.family, r.season): r.per_mode for r in m.itertuples()}


# ── Target ────────────────────────────────────────────────────────────────────

GAME_LOG_COLS = ["PLAYER_ID", "GAME_ID", "MIN", "PTS", "REB", "AST", "STL", "BLK", "TOV", "FG3M"]


def build_target(seasons: list[str], raw_dir: Path) -> pd.DataFrame:
    """Aggregate raw game logs into per-(player, season) dk_pts summaries.

    Read straight from data/raw rather than the processed parquet so this does
    not depend on `src.data.preprocess` having been run.
    """
    frames = []
    for season in seasons:
        path = nbastats_dir(raw_dir) / f"game_logs_{_slug(season)}.csv"
        if not path.exists():
            continue
        gl = pd.read_csv(path, usecols=lambda c: c in GAME_LOG_COLS)
        gl.columns = [c.lower() for c in gl.columns]
        gl["min"] = pd.to_numeric(gl["min"], errors="coerce")
        gl = gl.dropna(subset=["min", "pts", "reb", "ast"])
        gl["dk_pts"] = compute_dk_pts(gl)

        agg = gl.groupby("player_id").agg(
            dk_pts_total=("dk_pts", "sum"),
            dk_pts_per_game=("dk_pts", "mean"),
            dk_pts_std=("dk_pts", "std"),
            games_played=("game_id", "count"),
        ).reset_index()
        agg["season"] = season
        frames.append(agg)

    if not frames:
        raise FileNotFoundError(f"No game log CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_season(season: str, families: list[Family], raw_dir: Path,
                 manifest: dict[tuple[str, str], str],
                 min_gp: int = 20, min_minutes: float = 10.0) -> tuple[pd.DataFrame, list[dict]]:
    """Assemble one season's row block, plus its coverage records."""
    out = read_identity(season, raw_dir, min_gp, min_minutes)
    if out.empty:
        return pd.DataFrame(), []

    keep_ids = set(out["player_id"])
    minutes = out.set_index("player_id")["min"]

    coverage: list[dict] = []
    for fam in families:
        df = read_family(fam, season, raw_dir)
        rows = 0 if df is None else len(df)
        if df is None:
            coverage.append({"season": season, "family": fam.name, "tier": fam.tier,
                             "available": False, "rows": 0, "matched": 0})
            continue

        df = df[df["PLAYER_ID"].isin(keep_ids)].set_index("PLAYER_ID")
        stat_cols = [c for c in df.columns if c != "MIN"]

        # Per-36 basis: the family's own MIN when it has one, else base's. They
        # agree except where a family was fetched in a different per_mode.
        per_mode = manifest.get((fam.name, season), "PerGame")
        fam_minutes = df["MIN"] if (fam.has_min and "MIN" in df.columns) else minutes
        matched = len(df)
        df = df[stat_cols].apply(pd.to_numeric, errors="coerce")
        to_per36(df, stat_cols, fam_minutes.reindex(df.index), per_mode)

        df.columns = [f"{fam.prefix}_{c.lower()}" for c in df.columns]
        out = out.merge(df.reset_index().rename(columns={"PLAYER_ID": "player_id"}),
                        on="player_id", how="left")

        coverage.append({"season": season, "family": fam.name, "tier": fam.tier,
                         "available": True, "rows": rows, "matched": matched})

    return out, coverage


def build_tier(tier: str, seasons: list[str], raw_dir: str | Path,
               min_gp: int = 20, min_minutes: float = 10.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full season matrix for one tier, with its coverage report."""
    raw_dir = Path(raw_dir)
    manifest = load_manifest(raw_dir)
    first_year = TIER_FIRST_YEAR[tier]
    # Tier A is box-score only; Tier B is a strict superset of Tier A's columns.
    families = FAMILIES if tier == "B" else [f for f in FAMILIES if f.tier == "A"]
    tier_seasons = [s for s in seasons if _season_start_year(s) >= first_year]

    blocks, coverage = [], []
    for season in tier_seasons:
        block, cov = build_season(season, families, raw_dir, manifest, min_gp, min_minutes)
        if not block.empty:
            blocks.append(block)
        coverage.extend(cov)
        print(f"  {season}: {len(block):,} qualified players, {block.shape[1]} columns")

    if not blocks:
        raise FileNotFoundError(f"No season data found in {raw_dir}")

    matrix = pd.concat(blocks, ignore_index=True)
    target = build_target(tier_seasons, raw_dir)
    matrix = matrix.merge(target, on=["player_id", "season"], how="left")
    return matrix, pd.DataFrame(coverage)


def feature_cols(matrix: pd.DataFrame) -> list[str]:
    """Numeric style features — identity, volume and target columns held out."""
    exclude = set(ID_COLS + VOLUME_COLS + [
        "dk_pts_total", "dk_pts_per_game", "dk_pts_std", "games_played",
    ])
    return [c for c in matrix.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(matrix[c])]


def run(cfg: dict) -> dict[str, Path]:
    """Build both tiers plus the coverage report, writing to data/features/."""
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["data"]["features_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    eda = cfg["eda"]

    written: dict[str, Path] = {}
    coverage_frames = []
    for tier in ("A", "B"):
        print(f"\nTier {tier}:")
        matrix, coverage = build_tier(
            tier, cfg["data"]["seasons"], raw_dir,
            min_gp=eda["min_gp"], min_minutes=eda["min_minutes"],
        )
        dest = out_dir / f"season_matrix_tier{tier}.parquet"
        matrix.to_parquet(dest, index=False)
        written[tier] = dest
        coverage_frames.append(coverage)
        n_feat = len(feature_cols(matrix))
        print(f"Tier {tier}: {len(matrix):,} player-seasons, {n_feat} features, "
              f"{matrix['season'].nunique()} seasons → {dest}")

        # Unfiltered twin for roster aggregation. Same columns, every player who
        # took the floor — the qualification filter is a PCA concern, not a
        # description of who was on the roster.
        roster_cfg = eda.get("roster_frame", {"min_gp": 1, "min_minutes": 0})
        roster_matrix, _ = build_tier(
            tier, cfg["data"]["seasons"], raw_dir,
            min_gp=roster_cfg["min_gp"], min_minutes=roster_cfg["min_minutes"],
        )
        roster_dest = out_dir / f"season_matrix_roster_tier{tier}.parquet"
        roster_matrix.to_parquet(roster_dest, index=False)
        written[f"roster_{tier}"] = roster_dest
        extra = len(roster_matrix) - len(matrix)
        print(f"Tier {tier} roster frame: {len(roster_matrix):,} player-seasons "
              f"(+{extra:,} sub-threshold, {extra / len(roster_matrix):.1%}) → {roster_dest}")

    cov = pd.concat(coverage_frames, ignore_index=True).drop_duplicates(
        subset=["season", "family"], keep="last")
    cov_dest = out_dir / "coverage_report.csv"
    cov.sort_values(["season", "family"]).to_csv(cov_dest, index=False)
    written["coverage"] = cov_dest
    print(f"Coverage: {len(cov):,} season × family rows → {cov_dest}")
    return written


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
