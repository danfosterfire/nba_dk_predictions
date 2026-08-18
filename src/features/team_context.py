"""Own-team context features, computed leaving the target player out.

Every feature here answers "what does the *rest* of my roster look like?". Player P
must be excluded, otherwise the feature partly encodes P's own style and the model
can read part of its answer off its own input.

The headline feature is `role_crowding`: how much of P's role is already occupied by
teammates, measured as minutes-weighted similarity between P's archetype membership
and each teammate's. It is deliberately nonlinear in the players, so unlike a
minutes-weighted mean of PC scores it does not centroid-collapse — a roster of two
extremes and a roster of two averages give different answers.

## The construction: roster(S) x stats(S-1)

Season-start rosters are known before the season starts, so team context aggregates
the **season-S roster** described by **season S-1 stats**, weighting teammates by
their S-1 minutes. (An earlier version aggregated the S-1 roster and lagged the
result forward, which described a lineup that no longer exists.) Rosters come from
the game logs: a player is on team T's season-start roster if his first appearance
for T falls inside T's first `roster_window_games` games.

## Roster members with no usable S-1 row

15.9% of season-S roster minutes are played by someone the S-1 qualified matrix does
not describe — 9.4% true rookies, 5.4% sub-threshold, 1.1% returnees — and that
reaches 45%+ on the young, high-turnover rosters where team context is supposed to
matter most. Silently dropping them is the single largest bias available here, so
all three are carried explicitly, each with a reliability weight:

    prior     an S-1 row in the *inclusive* roster frame. Shrunk toward the league
              average by `reliability(minutes)`.
    stale     no S-1 row but an earlier one — the most recent is used, decayed a
              further STALENESS_DECAY per season of lag.
    rookie    never played. Imputed from an expanding-window prior over historical
              rookies at the same draft slot, with reliability 0 (i.e. the pure
              prior) and imputed minutes.

`roster_coverage` reports the share of each roster's weight that is observed rather
than imputed, so a model can discount thin aggregates instead of being misled.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _slug, nbastats_dir
from src.eda.archetypes import assign_archetypes

KEYS = ["player_id", "season"]
TEAM_KEYS = ["team_abbreviation", "season"]

# Prior-season stats the mechanism scalars are built from. Each targets a distinct
# channel by which a teammate suppresses or supports P's box score.
USAGE_COL = "adv_usg_pct"        # competition for shots
ASSIST_COL = "adv_ast_pct"       # supply of assists to P
PACE_COL = "adv_pace"            # possessions, which scale every component
SPACING_COL = "sco_pct_fga_3pt"  # floor spacing around P
MECHANISM_COLS = (USAGE_COL, ASSIST_COL, PACE_COL, SPACING_COL)

# Reliability of a per-36 rate as a function of prior-season total minutes:
# r(m) = r_inf * m / (m + RELIABILITY_MINUTES). Fitted to the observed
# season-to-season correlation of eight style stats across minutes bins on the
# inclusive frame (11,272 consecutive-season pairs): r_inf = 0.924, m0 = 66.
# So rates stabilize fast — the 200-minute qualification threshold already sits at
# 0.75 — which is why sub-threshold players are worth including at all.
RELIABILITY_MINUTES = 66.0

# Extra decay per season of staleness, from the lag-k persistence of the same eight
# stats on well-measured seasons (>=1000 min): r = 0.921 / 0.881 / 0.853 / 0.826 at
# lags 1-4, i.e. ~0.96 of the previous lag each year.
STALENESS_DECAY = 0.96

# Draft-slot buckets for the rookie prior. Undrafted players are their own category
# rather than an imputed number — bio_draft_number is ~15% NaN by construction.
DRAFT_BUCKETS = [(1, 5, "lottery_top5"), (6, 14, "lottery"),
                 (15, 30, "late_first"), (31, 60, "second_round")]
UNDRAFTED_BUCKET = "undrafted"

OUTPUT_COLS = [
    "role_crowding", "teammate_usage_max", "teammate_usage_sum", "teammate_usage_load",
    "teammate_assist_supply", "teammate_spacing", "team_pace",
    "teammate_minutes", "n_teammates", "roster_coverage",
]


# ── Season-start rosters ──────────────────────────────────────────────────────

ROSTER_LOG_COLS = ["PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN"]


def season_start_roster(season: str, raw_dir: str | Path,
                        window_games: int = 10) -> pd.DataFrame:
    """Who was on each team at the start of `season`, from that season's game logs.

    A player joins team T at the index of T's game in which he first appears. Keeping
    everyone inside T's first `window_games` recovers a ~15-man roster covering ~96%
    of the team's realized minutes. The two failure modes trade off against each
    other and neither is separable from game logs alone: a hard opening-night cut
    (index 0) drops players injured in October, leaving 11 players and 84% of
    minutes, while a season-long cut admits February signings who were never known
    pre-season.
    """
    path = nbastats_dir(raw_dir) / f"game_logs_{_slug(season)}.csv"
    if not path.exists():
        return pd.DataFrame(columns=["player_id", "season", "team_abbreviation"])

    gl = pd.read_csv(path, usecols=lambda c: c in ROSTER_LOG_COLS)
    gl.columns = [c.lower() for c in gl.columns]
    gl["game_date"] = pd.to_datetime(gl["game_date"], errors="coerce")
    gl = gl.dropna(subset=["game_date"])

    # Index each team's games chronologically, then read off when a player first
    # shows up for that team.
    games = (gl[["team_abbreviation", "game_id", "game_date"]].drop_duplicates()
             .sort_values(["team_abbreviation", "game_date", "game_id"]))
    games["team_game_index"] = games.groupby("team_abbreviation").cumcount()
    gl = gl.merge(games[["team_abbreviation", "game_id", "team_game_index"]],
                  on=["team_abbreviation", "game_id"], how="left")

    first = (gl.sort_values(["player_id", "game_date", "game_id"])
             .groupby(["player_id", "team_abbreviation"], as_index=False)
             .agg(team_game_index=("team_game_index", "min")))
    first = first[first["team_game_index"] <= window_games]

    # A player who appears for two teams inside the window (an early trade) is
    # attributed to the first — the roster he was known to start on.
    first = first.sort_values(["player_id", "team_game_index"]).drop_duplicates("player_id")
    return pd.DataFrame({
        "player_id": first["player_id"].values,
        "season": season,
        "team_abbreviation": first["team_abbreviation"].values,
    })


def season_start_rosters(seasons: list[str], raw_dir: str | Path,
                         window_games: int = 10) -> pd.DataFrame:
    frames = [season_start_roster(s, raw_dir, window_games) for s in seasons]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"No game log CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True)


# ── Reliability ───────────────────────────────────────────────────────────────

def reliability(minutes, prior_minutes: float = RELIABILITY_MINUTES,
                lag: int = 1) -> np.ndarray:
    """How much to trust a player's prior-season description, in [0, 1).

    `m / (m + m0)` is the standard empirical-Bayes reliability of a mean measured
    over `m` minutes against a prior of strength `m0`. Multiplying a z-scored
    feature by it *is* shrinkage toward the league average, so no per-feature means
    are needed. Staleness decays it a further STALENESS_DECAY per extra season.
    """
    m = np.asarray(minutes, dtype=float)
    m = np.where(np.isfinite(m) & (m > 0), m, 0.0)
    r = m / (m + prior_minutes)
    lag = np.asarray(lag, dtype=float)
    return r * STALENESS_DECAY ** np.maximum(lag - 1.0, 0.0)


# ── Prior-season descriptions ─────────────────────────────────────────────────

def draft_bucket(draft_number: pd.Series) -> pd.Series:
    """Map bio_draft_number to a coarse slot category, NaN → 'undrafted'."""
    n = pd.to_numeric(draft_number, errors="coerce")
    out = pd.Series(UNDRAFTED_BUCKET, index=n.index, dtype=object)
    for lo, hi, name in DRAFT_BUCKETS:
        out[(n >= lo) & (n <= hi)] = name
    return out


def _normalize_rows(M: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    return np.divide(M, norms, out=np.zeros_like(M), where=norms > 0)


def rookie_priors(described: pd.DataFrame, seasons: list[str],
                  membership_cols: list[str]) -> pd.DataFrame:
    """Expected first-season profile per (season, draft bucket), from past rookies.

    Fitted on an **expanding window** — the prior used for season S sees only
    rookies from seasons before S — so a backtest never learns a draft slot's value
    from the future. Seasons with no history fall back to the pooled prior.

    Memberships are averaged *after* row-normalization, which makes the result exact
    rather than approximate: role crowding is a dot product of unit vectors, so
    E[cos(m_P, m_rookie)] = m̂_P · E[m̂_rookie]. Normalizing the mean instead would
    give a different, wrong number.
    """
    order = {s: i for i, s in enumerate(seasons)}
    rookies = described[described["season_index"] == described["debut_index"]].copy()

    value_cols = list(MECHANISM_COLS) + membership_cols + ["prior_minutes"]
    value_cols = [c for c in value_cols if c in rookies.columns]

    rows = []
    pooled = rookies[value_cols].mean()
    for season in seasons:
        past = rookies[rookies["season_index"] < order[season]]
        by_bucket = past.groupby("draft_bucket")[value_cols].mean() if len(past) else None
        overall = past[value_cols].mean() if len(past) else pooled
        for bucket in [b for _, _, b in DRAFT_BUCKETS] + [UNDRAFTED_BUCKET]:
            if by_bucket is not None and bucket in by_bucket.index:
                vals = by_bucket.loc[bucket]
            else:
                vals = overall
            rows.append({"season": season, "draft_bucket": bucket, **vals.to_dict()})
    return pd.DataFrame(rows)


def describe_roster(rosters: pd.DataFrame, frame: pd.DataFrame, seasons: list[str],
                    features_dir: str | Path, tier: str) -> pd.DataFrame:
    """Attach each season-S roster member's best available prior-season description.

    `frame` is the inclusive roster season matrix (every player who took the floor).
    Returns one row per (player_id, season) carrying the mechanism scalars, archetype
    memberships, a `prior_minutes` weight, and `stats_source` / `stats_lag` /
    `reliability` describing where the numbers came from.
    """
    # Only seasons the frame actually covers can describe anyone. Tier B starts at
    # 2013-14, so without this every pre-tracking veteran would look like a rookie.
    covered = [s for s in seasons if s in set(frame["season"])]
    order = {s: i for i, s in enumerate(covered)}
    frame = frame[frame["season"].isin(order)].copy()
    frame["season_index"] = frame["season"].map(order)

    # Archetype memberships for every row of the inclusive frame, shrunk toward the
    # league average in proportion to minutes played, then row-normalized once here
    # so downstream cosine similarity is a plain dot product.
    rel = reliability(frame["min_total"])
    assigned = assign_archetypes(frame, features_dir, tier, reliability=rel)
    membership_cols = [c for c in assigned.columns if c.startswith("gmm_p")]
    frame[membership_cols] = _normalize_rows(assigned[membership_cols].to_numpy(dtype=float))

    frame["debut_index"] = frame.groupby("player_id")["season_index"].transform("min")
    frame["draft_bucket"] = draft_bucket(frame.get("bio_draft_number", pd.Series(np.nan,
                                                                                index=frame.index)))
    frame = frame.rename(columns={"min_total": "prior_minutes"})

    described_cols = ["player_id", "season_index", "debut_index", "draft_bucket",
                      "prior_minutes", *[c for c in MECHANISM_COLS if c in frame],
                      *membership_cols]
    described = frame[described_cols]

    # ── match each roster row to the most recent prior season available ───────
    # The first covered season has nothing behind it, so its rosters are not
    # describable at all — dropped rather than emitted as an all-rookie team.
    roster = rosters[rosters["season"].isin(order)].copy()
    roster["season_index"] = roster["season"].map(order)
    roster = roster[roster["season_index"] > 0]

    prior = described.rename(columns={"season_index": "prior_index"})
    merged = roster.merge(prior.drop(columns=["debut_index", "draft_bucket"]),
                          on="player_id", how="left")
    merged = merged[merged["prior_index"] < merged["season_index"]]
    merged["stats_lag"] = merged["season_index"] - merged["prior_index"]
    merged = (merged.sort_values(["player_id", "season", "stats_lag"])
              .drop_duplicates(["player_id", "season"], keep="first"))

    out = roster.merge(
        merged.drop(columns=["team_abbreviation", "season_index", "prior_index"]),
        on=["player_id", "season"], how="left")
    out["stats_source"] = np.where(out["stats_lag"].isna(), "rookie",
                                   np.where(out["stats_lag"] > 1, "stale", "prior"))

    # ── impute the rookies from the draft-slot prior ──────────────────────────
    priors = rookie_priors(described, seasons, membership_cols)
    own = described[["player_id", "season_index", "draft_bucket"]].rename(
        columns={"season_index": "season_index_own"})
    out = out.merge(own, left_on=["player_id", "season_index"],
                    right_on=["player_id", "season_index_own"], how="left")
    out["draft_bucket"] = out["draft_bucket"].fillna(UNDRAFTED_BUCKET)
    out = out.drop(columns=["season_index_own"])

    fill_cols = [c for c in MECHANISM_COLS if c in out] + membership_cols + ["prior_minutes"]
    out = out.merge(priors, on=["season", "draft_bucket"], how="left", suffixes=("", "_prior"))
    is_rookie = out["stats_source"] == "rookie"
    for c in fill_cols:
        pc = f"{c}_prior"
        if pc in out:
            out[c] = out[c].where(~is_rookie, out[pc])
    out = out.drop(columns=[c for c in out.columns if c.endswith("_prior")])

    out["stats_lag"] = out["stats_lag"].fillna(1.0)
    # A rookie's description is the prior itself, carrying no player-specific
    # evidence: reliability 0. His *weight* is the prior's expected minutes, which is
    # what stops him from silently vanishing from the aggregate.
    out["reliability"] = np.where(is_rookie, 0.0,
                                  reliability(out["prior_minutes"], lag=out["stats_lag"]))
    out["prior_minutes"] = out["prior_minutes"].fillna(0.0)
    return out.drop(columns=["season_index", "debut_index"], errors="ignore")


# ── Leave-one-out helpers ─────────────────────────────────────────────────────

def _loo_weighted_mean(df: pd.DataFrame, value: str, weight: str) -> pd.Series:
    """Minutes-weighted mean of `value` over everyone on the team except each player.

    O(1) per player: subtract the player's own contribution from the team total rather
    than regrouping once per player. Rows where `value` is missing contribute no weight,
    so they neither shift nor dilute the mean.
    """
    keys = [df[k] for k in TEAM_KEYS]
    w = df[weight].where(df[value].notna(), 0.0)
    vw = df[value].fillna(0.0) * w
    num = vw.groupby(keys).transform("sum") - vw
    den = w.groupby(keys).transform("sum") - w
    return (num / den).where(den > 0)


def _loo_sum(df: pd.DataFrame, value: str) -> pd.Series:
    """Total of `value` across teammates. NaN contributes 0, matching groupby-sum."""
    keys = [df[k] for k in TEAM_KEYS]
    return df[value].groupby(keys).transform("sum") - df[value].fillna(0.0)


def _loo_max(df: pd.DataFrame, value: str) -> pd.Series:
    """Max of `value` among teammates — "is there another primary option?".

    The leave-one-out max is the team's top value, except for the player who *is* that
    top value, who gets the runner-up. A lone player has no runner-up and yields NaN.
    """
    keys = [df[k] for k in TEAM_KEYS]
    ranked = df[value].groupby(keys).rank(method="first", ascending=False)
    top1 = df[value].groupby(keys).transform("max")
    top2 = df[value].where(ranked == 2).groupby(keys).transform("max")
    return top1.where(ranked != 1, top2)


# ── Role crowding ─────────────────────────────────────────────────────────────

def membership_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """L2-normalized GMM soft-membership vectors, for cosine similarity."""
    cols = sorted([c for c in df.columns if c.startswith("gmm_p")],
                  key=lambda c: int(c.removeprefix("gmm_p")))
    if not cols:
        raise KeyError("no gmm_p* membership columns — run `make archetypes` first")
    M = df[cols].to_numpy(dtype=float)
    return _normalize_rows(M), cols


def role_crowding(df: pd.DataFrame, weight: str = "min_total",
                  normalize: bool = True) -> pd.Series:
    """Minutes-weighted similarity between P and every teammate.

    ``crowd(P) = Σ_{i≠P} w̃ᵢ · cos(m_P, mᵢ)``  with ``w̃`` renormalized over teammates.

    0 means nobody on the roster plays P's role; values near 1 mean P's minutes are
    contested by players with the same style. A lone qualified player on a team has no
    teammates and yields NaN, not 0 — "no data" and "no crowding" are different.

    `normalize=False` takes the membership rows as already unit-length. Imputed
    rookies carry a *mean of normalized* vectors, whose norm is deliberately below 1;
    re-normalizing it would overstate how strongly a rookie of unknown role crowds.
    """
    if normalize:
        M, _ = membership_matrix(df)
    else:
        cols = sorted([c for c in df.columns if c.startswith("gmm_p")],
                      key=lambda c: int(c.removeprefix("gmm_p")))
        M = df[cols].to_numpy(dtype=float)
    w = df[weight].to_numpy(dtype=float)
    out = np.full(len(df), np.nan)

    for _, idx in df.groupby(TEAM_KEYS).indices.items():
        if len(idx) < 2:
            continue
        m, wi = M[idx], w[idx]
        sim = m @ m.T                       # cosine similarity, rows already normalized
        np.fill_diagonal(sim, 0.0)          # exclude self
        # weight each teammate by its share of the *other* players' minutes
        denom = wi.sum() - wi
        with np.errstate(invalid="ignore", divide="ignore"):
            out[idx] = np.where(denom > 0, (sim * wi).sum(axis=1) / denom, np.nan)
    return pd.Series(out, index=df.index)


# ── Assembly ──────────────────────────────────────────────────────────────────

def build_team_context(described: pd.DataFrame, weight: str = "prior_minutes",
                       normalize_membership: bool = True) -> pd.DataFrame:
    """One row per (player_id, season) of leave-one-out own-team context.

    `described` is the output of `describe_roster`: the season-S roster with each
    member's season S-1 description already attached and weighted.
    """
    df = described.sort_values(TEAM_KEYS + ["player_id"]).reset_index(drop=True)

    out = df[KEYS + TEAM_KEYS[:1]].copy()
    for col in ("stats_source", "stats_lag", "reliability"):
        if col in df:
            out[col] = df[col].values

    out["role_crowding"] = role_crowding(df, weight, normalize=normalize_membership)
    out["teammate_usage_max"] = _loo_max(df, USAGE_COL) if USAGE_COL in df else np.nan
    # Usage is close to zero-sum across the five on-court players, so the total
    # claimed by everyone else is the direct competition P faces.
    out["teammate_usage_sum"] = _loo_sum(df, USAGE_COL) if USAGE_COL in df else np.nan
    # ...but a raw sum over an inclusive roster grows with roster size, which is a
    # bookkeeping artifact rather than a basketball fact. The minutes-weighted mean
    # scaled to a five-man lineup is the roster-size-invariant version.
    out["teammate_usage_load"] = (5.0 * _loo_weighted_mean(df, USAGE_COL, weight)
                                  if USAGE_COL in df else np.nan)
    out["teammate_assist_supply"] = (_loo_weighted_mean(df, ASSIST_COL, weight)
                                     if ASSIST_COL in df else np.nan)
    out["teammate_spacing"] = (_loo_weighted_mean(df, SPACING_COL, weight)
                               if SPACING_COL in df else np.nan)
    # Pace is a property of the whole team including P; it is not leave-one-out.
    if PACE_COL in df:
        w = df[weight].where(df[PACE_COL].notna(), 0.0)
        num = (df[PACE_COL].fillna(0.0) * w).groupby([df[k] for k in TEAM_KEYS]).transform("sum")
        den = w.groupby([df[k] for k in TEAM_KEYS]).transform("sum")
        out["team_pace"] = (num / den).where(den > 0)
    else:
        out["team_pace"] = np.nan
    out["teammate_minutes"] = _loo_sum(df, weight)
    out["n_teammates"] = df.groupby(TEAM_KEYS)["player_id"].transform("count") - 1

    # Share of the roster's weight that is observed rather than imputed. Thin
    # aggregates are visible to the model instead of silently misleading it.
    if "stats_source" in df:
        observed = df[weight].where(df["stats_source"] != "rookie", 0.0)
        keys = [df[k] for k in TEAM_KEYS]
        total = df[weight].groupby(keys).transform("sum")
        out["roster_coverage"] = (observed.groupby(keys).transform("sum") / total).where(total > 0)
    else:
        out["roster_coverage"] = np.nan
    return out


STATS_SOURCES = ("prior", "stale", "rookie")


def coverage_report(described: pd.DataFrame,
                    sources: tuple[str, ...] = STATS_SOURCES) -> pd.DataFrame:
    """Per team-season share of roster minutes by prior-data source.

    Weighted by *realized* season-S minutes where available, because that is the
    exposure the aggregate is failing to describe. Weighting by the S-1 minutes the
    aggregate actually uses would be circular: a rookie has zero of them, so every
    roster would look fully covered.

    `sources` is a parameter because `stats_source`'s three-way split is not the split the
    15.9% figure is quoted on: that one separates a *sub-threshold* prior season from a
    qualified one, which is a fact about the qualified matrix rather than about this frame.
    See `src/eda/context_value.py::description_source`.
    """
    df = described.copy()
    w = "season_minutes" if "season_minutes" in df else "prior_minutes"
    tot = df.groupby(TEAM_KEYS)[w].sum().rename("total")
    by = df.groupby(TEAM_KEYS + ["stats_source"])[w].sum().unstack("stats_source").fillna(0.0)
    by = by.join(tot)
    for c in sources:
        if c not in by:
            by[c] = 0.0
        by[f"share_{c}"] = (by[c] / by["total"]).where(by["total"] > 0)
    return by.reset_index()


def run(cfg: dict, tier: str = "A") -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    seasons = cfg["data"]["seasons"]
    tc_cfg = cfg.get("features", {}).get("team_context", {})
    window = tc_cfg.get("roster_window_games", 10)

    frame_path = features_dir / f"season_matrix_roster_tier{tier}.parquet"
    if not frame_path.exists():
        raise FileNotFoundError(f"{frame_path} not found — run `make season-matrix` first")
    frame = pd.read_parquet(frame_path)

    rosters = season_start_rosters(seasons, raw_dir, window)
    described = describe_roster(rosters, frame, seasons, features_dir, tier)
    context = build_team_context(described, normalize_membership=False)

    dest = features_dir / f"team_context_tier{tier}.parquet"
    context.to_parquet(dest, index=False)

    n_crowd = int(context["role_crowding"].notna().sum())
    mix = described["stats_source"].value_counts(normalize=True)
    print(f"Team context tier {tier}: {len(context):,} player-seasons "
          f"({n_crowd:,} with role_crowding) → {dest}")
    print(f"  roster rows by source: " +
          ", ".join(f"{k} {v:.1%}" for k, v in mix.items()))
    print(f"  roster_coverage (share of weight observed): "
          f"mean {context['roster_coverage'].mean():.3f}, "
          f"p10 {context['roster_coverage'].quantile(0.1):.3f}")
    print(context[OUTPUT_COLS].describe().loc[["mean", "std", "min", "max"]].round(3).to_string())
    return dest


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    for t in ("A", "B"):
        run(cfg, t)
