"""Opponent encoding: team defensive quality, plus a low-rank matchup interaction.

Two blocks, in the order their measured value justifies.

**(a) Team-level defensive quality, read straight off disk.** The seven `team_stats_*`
families and `team_estimated_metrics` are one row per team per season, 30 seasons, and
were previously unused. `team_stats_opponent` is the one that matters most: for team T
its `OPP_*` columns are *what T's opponents recorded against T*, which is exactly the
question "if my player faces T, what does he get?" — already resolved per DK component.
Nothing here is rebuilt by aggregating player rows, which would also inherit the
roster-coverage bias documented in README.

**(b) A rank-1/2 bilinear matchup term.** The interaction between player style and
opponent defense measured +1.00% of within-player residual variance against a shuffled
null, versus 0.69% for the opponent main effect, so the interaction is the point:

    matchup_k = (u_k · s_P) * (v_k · o_T)        k = 1..rank

`s_P` is P's prior-season style (PCA scores), `o_T` the opponent's prior-season
defensive profile. At rank 2 with 8 style and 5 defense dims that is 26 parameters —
deliberately tiny, because the effective support is 892 team-seasons, not 700k
player-games. Fitted by alternating least squares, which is a ridge regression in
closed form on each side.

## Basis quirk in the team files

`fetch_team_stats` requests `PerMode="PerGame"`, and the counting stats do come back
per game — but `MIN` and `POSS` are **season totals** (MIN ~3966 = 48.4 * 82). The
`stat / MIN * 36` rule that holds for the player families is therefore wrong here.
Everything is put on a per-100-possessions basis instead, using `POSS / GP`, which is
the right normalization for a defense anyway: pace and stinginess are separate axes and
must not be collapsed.
"""

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA

from src.data.fetch import _slug
from src.data.preprocess import compute_dk_pts
from src.features.targets import COMPONENTS

TEAM_KEYS = ["team_id", "season"]

# Families to fold in, and the columns worth keeping from each. Ranks, records and
# names are dropped; anything repeated across families is kept once, from the first
# family that carries it.
TEAM_FAMILIES: dict[str, tuple[str, ...]] = {
    "team_stats_advanced": ("DEF_RATING", "OFF_RATING", "PACE", "POSS",
                            "DREB_PCT", "OREB_PCT", "EFG_PCT", "TM_TOV_PCT"),
    # OPP_* here is "what opponents recorded against this team" — the per-component
    # opponent profile, no aggregation required.
    "team_stats_opponent": ("OPP_PTS", "OPP_FG3M", "OPP_FGA", "OPP_FG3A", "OPP_FTA",
                            "OPP_OREB", "OPP_DREB", "OPP_REB", "OPP_AST", "OPP_TOV",
                            "OPP_STL", "OPP_BLK", "OPP_FG_PCT", "OPP_FG3_PCT"),
    "team_stats_four_factors": ("OPP_EFG_PCT", "OPP_FTA_RATE", "OPP_TOV_PCT",
                                "OPP_OREB_PCT"),
    "team_stats_defense": ("OPP_PTS_PAINT", "OPP_PTS_FB", "OPP_PTS_2ND_CHANCE",
                           "OPP_PTS_OFF_TOV"),
    # 2014-15+ only; a model-estimated view that is less sensitive to lineup noise.
    "team_estimated_metrics": ("E_DEF_RATING", "E_PACE"),
}

# Counting stats in the team files arrive per game and are converted to per-100
# possessions. Anything matching these markers is already a rate and passes through.
TEAM_RATE_MARKERS = ("_PCT", "_RATING", "PACE", "_RATE")

# Volume/identity columns held out of the defensive profile itself.
PROFILE_EXCLUDE = ("poss", "gp", "off_rating", "efg_pct", "tm_tov_pct")


def _is_team_rate(col: str) -> bool:
    return any(m in col.upper() for m in TEAM_RATE_MARKERS)


# ── Loading the team families ─────────────────────────────────────────────────

def read_team_family(family: str, season: str, raw_dir: str | Path) -> pd.DataFrame | None:
    """One team family for one season, keyed on TEAM_ID with the kept columns only."""
    path = Path(raw_dir) / f"{family}_{_slug(season)}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty or "TEAM_ID" not in df.columns:
        return None

    keep = [c for c in TEAM_FAMILIES[family] if c in df.columns]
    out = df[["TEAM_ID", *keep]].drop_duplicates(subset="TEAM_ID", keep="first")
    if "GP" in df.columns:
        out = out.assign(GP=df["GP"].values[:len(out)])
    return out


def build_team_seasons(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """One row per (team_id, season) across every team family, per-100 normalized."""
    frames = []
    for season in seasons:
        block: pd.DataFrame | None = None
        for family in TEAM_FAMILIES:
            df = read_team_family(family, season, raw_dir)
            if df is None:
                continue
            if block is None:
                block = df
            else:
                new = [c for c in df.columns if c not in block.columns or c == "TEAM_ID"]
                block = block.merge(df[new], on="TEAM_ID", how="left")
        if block is None or block.empty:
            continue
        block["season"] = season
        frames.append(block)

    if not frames:
        raise FileNotFoundError(f"No team stat CSVs found in {raw_dir}")
    out = pd.concat(frames, ignore_index=True)
    out.columns = [c.lower() for c in out.columns]
    return to_per100(out)


def to_per100(team_seasons: pd.DataFrame) -> pd.DataFrame:
    """Convert per-game counting stats to per-100 possessions, rates untouched.

    Possessions per game is `POSS / GP` — POSS is a season total in these files even
    though the counting stats are per game. Rate columns are already possession- or
    attempt-normalized and must not be divided again.
    """
    out = team_seasons.copy()
    if "poss" not in out or "gp" not in out:
        raise KeyError("team_stats_advanced must supply POSS and GP for per-100 scaling")

    poss_per_game = (out["poss"] / out["gp"]).replace(0.0, np.nan)
    out["poss_per_game"] = poss_per_game
    for col in out.columns:
        if col in ("team_id", "season", "gp", "poss", "poss_per_game"):
            continue
        if _is_team_rate(col) or not pd.api.types.is_numeric_dtype(out[col]):
            continue
        out[col] = out[col] / poss_per_game * 100.0
    return out


# ── The defensive profile ─────────────────────────────────────────────────────

def profile_cols(team_seasons: pd.DataFrame) -> list[str]:
    """Columns describing what a defense concedes, in era-neutral form."""
    return [c for c in team_seasons.columns
            if c not in TEAM_KEYS and c not in PROFILE_EXCLUDE
            and not c.startswith("poss")
            and pd.api.types.is_numeric_dtype(team_seasons[c])]


def standardize_within_season(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Z-score inside each season, so era drift is not mistaken for defensive quality.

    Defensive ratings rose ~10 points and pace ~8 possessions over the sample. Without
    this, "faces a good defense" and "plays in 2004" are the same direction — the exact
    confound that invented the previously-reported team-context effects.
    """
    out = df[TEAM_KEYS].copy()
    g = df.groupby("season")[cols]
    z = (df[cols] - g.transform("mean")) / g.transform("std").replace(0.0, np.nan)
    # A season where a column is constant or absent carries no information; 0 is that
    # season's league average in z-space.
    return pd.concat([out, z.fillna(0.0)], axis=1)


def fit_profile_pca(profile: pd.DataFrame, cols: list[str], n_components: int = 5,
                    seed: int = 42) -> tuple[pd.DataFrame, PCA]:
    """Reduce the standardized defensive profile to a handful of axes.

    Held to `n_components` deliberately: the effective sample is 892 team-seasons, and
    the matchup interaction is worth ~1% of within-player residual variance. A richer
    opponent representation would be fitting noise with extra steps.
    """
    X = profile[cols].to_numpy(dtype=float)
    pca = PCA(n_components=min(n_components, X.shape[1]), random_state=seed).fit(X)
    scores = pd.DataFrame(pca.transform(X),
                          columns=[f"opp_pc{i + 1}" for i in range(pca.n_components_)],
                          index=profile.index)
    return pd.concat([profile[TEAM_KEYS], scores], axis=1), pca


def nxt_map(seasons: list[str]) -> dict[str, str]:
    """season S → the season it predicts, S+1. Gaps in the list never pair across."""
    return {s: seasons[i + 1] for i, s in enumerate(seasons) if i + 1 < len(seasons)}


def lag_profiles(profile: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Shift a team's profile forward one season: season S carries S-1's numbers.

    Keyed on `team_id` rather than abbreviation, so relocations (SEA→OKC, NJN→BKN)
    stay joined to their own history instead of silently becoming expansion teams.
    """
    order = {s: i for i, s in enumerate(seasons)}
    nxt = {s: seasons[i + 1] for s, i in order.items() if i + 1 < len(seasons)}
    out = profile[profile["season"].isin(nxt)].copy()
    out["profile_season"] = out["season"]
    out["season"] = out["season"].map(nxt)
    return out


# ── Per-game opponents ────────────────────────────────────────────────────────

GAME_LOG_COLS = ["SEASON_YEAR", "PLAYER_ID", "TEAM_ID", "TEAM_ABBREVIATION", "GAME_ID",
                 "GAME_DATE", "MATCHUP", "MIN", "PTS", "REB", "AST", "STL", "BLK",
                 "TOV", "FG3M"]


def opponent_abbreviation(matchup: pd.Series) -> pd.Series:
    """'LAL @ NOP' or 'LAL vs. NOP' → 'NOP'."""
    return matchup.str.split(r"\s+(?:@|vs\.)\s+", regex=True).str[-1].str.strip()


def is_home(matchup: pd.Series) -> pd.Series:
    """'vs.' is a home game, '@' is away."""
    return matchup.str.contains("vs.", regex=False).astype(float)


def load_player_games(seasons: list[str], raw_dir: str | Path) -> pd.DataFrame:
    """Player-game rows with opponent, home flag, dk_pts and per-36 component rates."""
    frames = []
    for season in seasons:
        path = Path(raw_dir) / f"game_logs_{_slug(season)}.csv"
        if not path.exists():
            continue
        gl = pd.read_csv(path, usecols=lambda c: c in GAME_LOG_COLS)
        gl.columns = [c.lower() for c in gl.columns]
        gl["season"] = season
        frames.append(gl)
    if not frames:
        raise FileNotFoundError(f"No game log CSVs found in {raw_dir}")

    gl = pd.concat(frames, ignore_index=True)
    gl["min"] = pd.to_numeric(gl["min"], errors="coerce")
    gl = gl.dropna(subset=["min", *COMPONENTS])
    gl = gl[gl["min"] > 0]

    gl["opponent_abbreviation"] = opponent_abbreviation(gl["matchup"])
    gl["is_home"] = is_home(gl["matchup"])
    gl["dk_pts"] = compute_dk_pts(gl)

    exposure = gl["min"] / 36.0
    for c in COMPONENTS:
        gl[f"{c}_per36"] = gl[c] / exposure

    # team_id is only in the file for the player's own team, so the opponent's id
    # comes from the season's own abbreviation → id map.
    ids = (gl[["season", "team_abbreviation", "team_id"]].drop_duplicates()
           .rename(columns={"team_abbreviation": "opponent_abbreviation",
                            "team_id": "opponent_team_id"}))
    return gl.merge(ids, on=["season", "opponent_abbreviation"], how="left")


def within_player_season_residuals(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Subtract each player-season's own mean — the residual the budget is defined on.

    Player-season identity is 58% of total per-game variance and is *not* what an
    opponent encoder is for. Everything below is measured against the 9.4 dk_pts of
    within-player-season noise that remains.
    """
    g = df.groupby(["player_id", "season"])[cols]
    return df[cols] - g.transform("mean")


# ── Low-rank bilinear matchup ─────────────────────────────────────────────────

@dataclass
class BilinearMatchup:
    """`y ≈ alpha·o + Σ_k (u_k·s)(v_k·o)`, plus an intercept.

    U is (rank, n_style), V is (rank, n_defense). The two are identified only up to a
    per-k scale (u_k c, v_k / c give the same product), so V's rows are normalized
    after fitting to make the artifact reproducible.
    """

    U: np.ndarray
    V: np.ndarray
    alpha: np.ndarray
    intercept: float
    rank: int

    def interaction(self, S: np.ndarray, O: np.ndarray) -> np.ndarray:
        """The (n, rank) elementwise products — the features a model consumes."""
        return (S @ self.U.T) * (O @ self.V.T)

    def predict(self, S: np.ndarray, O: np.ndarray) -> np.ndarray:
        return self.intercept + O @ self.alpha + self.interaction(S, O).sum(axis=1)


def _ridge(X: np.ndarray, y: np.ndarray, lam: float,
           w: np.ndarray | None = None) -> np.ndarray:
    """Closed-form (weighted) ridge. No intercept is added; pass one in X."""
    if w is None:
        A, b = X.T @ X, X.T @ y
    else:
        Xw = X * w[:, None]
        A, b = Xw.T @ X, Xw.T @ y
    return np.linalg.solve(A + lam * np.eye(X.shape[1]), b)


def minutes_weights(minutes: np.ndarray) -> np.ndarray:
    """Weights for regressions on per-36 rates, normalized to mean 1.

    A per-36 rate from a 3-minute garbage-time appearance has ~12x the sd of one from
    a 30-minute game (measured: sd(pts_per36) is 42.4 under 5 minutes against 6.6
    above 24). Unweighted, those rows dominate every rate regression and the opponent
    signal disappears under them. Weighting by minutes is the exposure the rate was
    actually measured over.
    """
    m = np.asarray(minutes, dtype=float)
    m = np.where(np.isfinite(m) & (m > 0), m, 0.0)
    return m / m.mean() if m.mean() > 0 else m


def fit_bilinear(S: np.ndarray, O: np.ndarray, y: np.ndarray, rank: int = 2,
                 ridge: float = 1.0, iters: int = 25, seed: int = 42,
                 tol: float = 1e-9, w: np.ndarray | None = None) -> BilinearMatchup:
    """Alternating least squares for the low-rank matchup term.

    The objective is biconvex: with V fixed the model is linear in U and vice versa,
    so each half-step is a ridge regression in closed form. The opponent main effect
    `alpha` is refit each sweep alongside the U step, so the interaction is only ever
    credited with what the main effect could not explain.
    """
    S = np.asarray(S, dtype=float)
    O = np.asarray(O, dtype=float)
    y = np.asarray(y, dtype=float)
    rng = np.random.default_rng(seed)

    d_s, d_o = S.shape[1], O.shape[1]
    V = rng.normal(scale=1.0 / np.sqrt(d_o), size=(rank, d_o))
    U = np.zeros((rank, d_s))
    ones = np.ones((len(y), 1))

    prev = np.inf
    for _ in range(iters):
        # ── U step: features are s ⊗ (v_k · o), stacked over k, next to o and 1 ──
        proj_o = O @ V.T                                    # (n, rank)
        blocks = [S * proj_o[:, [k]] for k in range(rank)]
        X = np.hstack([ones, O, *blocks])
        beta = _ridge(X, y, ridge, w)
        intercept, alpha = float(beta[0]), beta[1:1 + d_o]
        U = beta[1 + d_o:].reshape(rank, d_s)

        # ── V step: features are o ⊗ (u_k · s) ─────────────────────────────────
        proj_s = S @ U.T                                    # (n, rank)
        blocks = [O * proj_s[:, [k]] for k in range(rank)]
        X = np.hstack([ones, O, *blocks])
        beta = _ridge(X, y, ridge, w)
        intercept, alpha = float(beta[0]), beta[1:1 + d_o]
        V = beta[1 + d_o:].reshape(rank, d_o)

        resid = y - (intercept + O @ alpha + ((S @ U.T) * (O @ V.T)).sum(axis=1))
        sse = float(resid @ resid if w is None else (w * resid) @ resid)
        if abs(prev - sse) <= tol * max(prev, 1.0):
            break
        prev = sse

    # Fix the scale indeterminacy so refits are comparable.
    norms = np.linalg.norm(V, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return BilinearMatchup(U=U * norms, V=V / norms, alpha=alpha,
                           intercept=intercept, rank=rank)


# ── Measurement ───────────────────────────────────────────────────────────────

def variance_explained(y: np.ndarray, pred: np.ndarray,
                       w: np.ndarray | None = None) -> float:
    """Share of the variance of y explained by `pred`, relative to y's own mean."""
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones(len(y))
    mean = float(np.average(y, weights=w))
    tss = float((w * (y - mean) ** 2).sum())
    rss = float((w * (y - pred) ** 2).sum())
    return 1.0 - rss / tss if tss > 0 else np.nan


def main_effect_only(O_tr, y_tr, O_te, ridge: float = 1.0,
                     w: np.ndarray | None = None) -> np.ndarray:
    """Opponent main effect alone — the baseline the interaction has to beat."""
    X = np.hstack([np.ones((len(y_tr), 1)), O_tr])
    beta = _ridge(X, y_tr, ridge, w)
    return np.hstack([np.ones((len(O_te), 1)), O_te]) @ beta


def evaluate_matchup(S_tr, O_tr, y_tr, S_te, O_te, y_te, rank: int = 2,
                     ridge: float = 1.0, n_shuffles: int = 3, seed: int = 42,
                     w_tr: np.ndarray | None = None,
                     w_te: np.ndarray | None = None) -> dict:
    """Held-out variance explained: main effect, interaction, and a shuffled null.

    The null permutes the *player style* rows against a fixed opponent column,
    breaking the player↔opponent pairing while leaving both marginals intact. A rank-2
    interaction fitted on half a million rows will explain something by chance; the
    null says how much.

    Held-out evaluation is the point. The often-quoted "opponent x archetype x season"
    figure is an in-sample ANOVA over ~2,700 cells whose expected chance R² is ~1.4%,
    and its "above null" value swings from +0.29% to +0.97% depending only on which
    marginal the null shuffles. Nothing that fragile should size a design decision.
    """
    base_te = main_effect_only(O_tr, y_tr, O_te, ridge, w_tr)
    fit = fit_bilinear(S_tr, O_tr, y_tr, rank=rank, ridge=ridge, seed=seed, w=w_tr)
    full_te = fit.predict(S_te, O_te)

    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(n_shuffles):
        perm = rng.permutation(len(S_tr))
        null_fit = fit_bilinear(S_tr[perm], O_tr, y_tr, rank=rank, ridge=ridge,
                                seed=seed, w=w_tr)
        perm_te = rng.permutation(len(S_te))
        nulls.append(variance_explained(y_te, null_fit.predict(S_te[perm_te], O_te), w_te))

    return {
        "r2_main_effect": variance_explained(y_te, base_te, w_te),
        "r2_with_interaction": variance_explained(y_te, full_te, w_te),
        "r2_shuffled_null": float(np.mean(nulls)),
        "n_train": len(y_tr), "n_test": len(y_te), "rank": rank,
        "fit": fit,
    }


def _cell_share(y: np.ndarray, key: np.ndarray) -> float:
    """In-sample share of variance explained by cell means — a plain one-way ANOVA."""
    df = pd.DataFrame({"y": np.asarray(y, dtype=float), "g": np.asarray(key)})
    m = df.groupby("g")["y"].transform("mean")
    tss = float(((df["y"] - df["y"].mean()) ** 2).sum())
    return 1.0 - float(((df["y"] - m) ** 2).sum()) / tss if tss > 0 else np.nan


def variance_ceiling(games: pd.DataFrame, archetype: pd.Series,
                     outcome: str = "dk_pts", n_shuffles: int = 5,
                     seed: int = 42) -> dict:
    """The in-sample ANOVA that the quoted opponent numbers come from.

    Reproduced here so the figures in README have a source. **These are ceilings, not
    achievable gains**: they use *contemporaneous* opponent identity and are fitted in
    sample, whereas a usable encoder knows only the opponent's prior season.

    Two nulls are reported because the answer depends on which one is chosen, and that
    dependence is itself the finding. `null_shuffle_opponent` permutes opponent within
    season (the construction behind the quoted +1.00%); `null_shuffle_archetype`
    permutes the style label instead. Both preserve the marginals, and they disagree
    by a factor of three.
    """
    y = games[outcome].to_numpy(dtype=float)
    opp = games["opponent_team_id"].astype(str).to_numpy()
    season = games["season"].to_numpy()
    arch = archetype.astype(str).to_numpy()

    joined = np.char.add(np.char.add(opp, arch), season)
    rng = np.random.default_rng(seed)

    null_o, null_a = [], []
    for _ in range(n_shuffles):
        perm_opp = opp.copy()
        for s in np.unique(season):
            m = season == s
            perm_opp[m] = rng.permutation(opp[m])
        null_o.append(_cell_share(y, np.char.add(np.char.add(perm_opp, arch), season)))
        perm_arch = arch.copy()
        for s in np.unique(season):
            m = season == s
            perm_arch[m] = rng.permutation(arch[m])
        null_a.append(_cell_share(y, np.char.add(np.char.add(opp, perm_arch), season)))

    return {
        "n": len(games),
        "opponent_x_season": _cell_share(y, np.char.add(opp, season)),
        "home_away": _cell_share(y, games["is_home"].astype(str).to_numpy()),
        "opponent_x_archetype_x_season": _cell_share(y, joined),
        "null_shuffle_opponent": float(np.mean(null_o)),
        "null_shuffle_archetype": float(np.mean(null_a)),
        "n_cells": int(len(np.unique(joined))),
    }


# ── Artifacts ─────────────────────────────────────────────────────────────────

def save_artifacts(pca: PCA, cols: list[str], matchups: dict,
                   out_dir: str | Path, tier: str) -> None:
    """Mirror of src/eda/pca.py::save_artifacts."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"opponent_pca_tier{tier}.pkl", "wb") as f:
        pickle.dump(pca, f)
    with open(out_dir / f"opponent_profile_cols_tier{tier}.pkl", "wb") as f:
        pickle.dump(cols, f)
    with open(out_dir / f"opponent_matchup_tier{tier}.pkl", "wb") as f:
        pickle.dump(matchups, f)


def load_artifacts(out_dir: str | Path, tier: str) -> tuple[PCA, list[str], dict]:
    out_dir = Path(out_dir)
    with open(out_dir / f"opponent_pca_tier{tier}.pkl", "rb") as f:
        pca = pickle.load(f)
    with open(out_dir / f"opponent_profile_cols_tier{tier}.pkl", "rb") as f:
        cols = pickle.load(f)
    with open(out_dir / f"opponent_matchup_tier{tier}.pkl", "rb") as f:
        matchups = pickle.load(f)
    return pca, cols, matchups


# ── Orchestration ─────────────────────────────────────────────────────────────

STYLE_DIMS = 8
DEFENSE_DIMS = 5


def build_panel(seasons: list[str], raw_dir: str | Path, features_dir: str | Path,
                tier: str, defense_dims: int = DEFENSE_DIMS,
                style_dims: int = STYLE_DIMS, seed: int = 42):
    """Player-games joined to prior-season style and prior-season opponent defense."""
    team_seasons = build_team_seasons(seasons, raw_dir)
    cols = profile_cols(team_seasons)
    standardized = standardize_within_season(team_seasons, cols)
    profile, pca = fit_profile_pca(standardized, cols, defense_dims, seed)
    lagged = lag_profiles(profile, seasons)

    games = load_player_games(seasons, raw_dir)
    outcomes = ["dk_pts"] + [f"{c}_per36" for c in COMPONENTS]
    games[outcomes] = within_player_season_residuals(games, outcomes)

    # Opponent defense: the opponent's *prior* season profile.
    opp = lagged.rename(columns={"team_id": "opponent_team_id"})
    opp_cols = [c for c in opp.columns if c.startswith("opp_pc")]
    games = games.merge(opp[["opponent_team_id", "season", *opp_cols]],
                        on=["opponent_team_id", "season"], how="inner")

    # Player style: his prior-season PCA scores, lagged the same way.
    scores = pd.read_parquet(
        Path(features_dir) / f"pca_tier{tier}_within_season_scores.parquet")
    style_cols = [f"pc{i + 1}" for i in range(style_dims)]
    style = scores[["player_id", "season", *style_cols]].copy()
    order = {s: i for i, s in enumerate(seasons)}
    nxt = {s: seasons[i + 1] for s, i in order.items() if i + 1 < len(seasons)}
    style["season"] = style["season"].map(nxt)
    style = style.dropna(subset=["season"])

    games = games.merge(style, on=["player_id", "season"], how="inner")
    return games, style_cols, opp_cols, pca, cols


def run(cfg: dict, tier: str = "A") -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["eda"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    opp_cfg = cfg.get("features", {}).get("opponent", {})
    rank = opp_cfg.get("rank", 2)
    ridge = opp_cfg.get("ridge", 100.0)
    seed = cfg["training"]["seed"]

    games, style_cols, opp_cols, pca, prof_cols = build_panel(
        seasons, cfg["data"]["raw_dir"], features_dir, tier,
        opp_cfg.get("defense_dims", DEFENSE_DIMS),
        opp_cfg.get("style_dims", STYLE_DIMS), seed)

    print(f"\nTier {tier}: {len(games):,} player-games with prior-season style "
          f"and prior-season opponent profile")
    print(f"Defensive profile: {len(prof_cols)} team columns → {len(opp_cols)} PCs "
          f"({pca.explained_variance_ratio_.sum():.1%} of profile variance)")

    # Temporal walk-forward: the last two seasons are held out.
    test_seasons = set(seasons[-2:])
    tr = games[~games["season"].isin(test_seasons)]
    te = games[games["season"].isin(test_seasons)]

    S_tr, O_tr = tr[style_cols].to_numpy(float), tr[opp_cols].to_numpy(float)
    S_te, O_te = te[style_cols].to_numpy(float), te[opp_cols].to_numpy(float)

    # Per-36 rates are weighted by minutes; per-game dk_pts is already a per-game
    # quantity and is left alone.
    w_tr_min = minutes_weights(tr["min"].to_numpy(float))
    w_te_min = minutes_weights(te["min"].to_numpy(float))

    rows, matchups = [], {}
    for outcome in ["dk_pts"] + [f"{c}_per36" for c in COMPONENTS]:
        weighted = outcome != "dk_pts"
        res = evaluate_matchup(S_tr, O_tr, tr[outcome].to_numpy(float),
                               S_te, O_te, te[outcome].to_numpy(float),
                               rank=rank, ridge=ridge, seed=seed,
                               w_tr=w_tr_min if weighted else None,
                               w_te=w_te_min if weighted else None)
        matchups[outcome] = res.pop("fit")
        rows.append({"outcome": outcome, **res, "minutes_weighted": weighted,
                     "interaction_above_null": res["r2_with_interaction"] - res["r2_shuffled_null"]})

    table = pd.DataFrame(rows)
    dest = out_dir / f"opponent_matchup_tier{tier}.csv"
    table.to_csv(dest, index=False)
    save_artifacts(pca, prof_cols, matchups, features_dir, tier)

    # The fitted profile scores themselves, so consumers (the dashboard's opponent
    # tab) can read teams' positions in defensive-profile space instead of
    # rebuilding the team families and refitting the PCA to see them.
    profile_dest = features_dir / f"opponent_profile_tier{tier}.parquet"
    rebuilt = build_team_seasons(seasons, cfg["data"]["raw_dir"])
    standardized = standardize_within_season(rebuilt, prof_cols)
    scores = pd.DataFrame(pca.transform(standardized[prof_cols].to_numpy(dtype=float)),
                          columns=[f"opp_pc{i + 1}" for i in range(pca.n_components_)])
    pd.concat([rebuilt[TEAM_KEYS].reset_index(drop=True), scores], axis=1
              ).to_parquet(profile_dest, index=False)
    print(f"→ {profile_dest}  ({len(scores):,} team-seasons in defensive-profile space)")

    show = ["outcome", "r2_main_effect", "r2_with_interaction", "r2_shuffled_null",
            "interaction_above_null"]
    print(f"\nHeld-out ({len(te):,} player-games in {sorted(test_seasons)}), "
          f"% of within-player-season residual variance:")
    print((table[show].set_index("outcome") * 100).round(3).to_string())

    # Rank is a design choice with a budget attached, so show the evidence for it.
    print("\nRank sweep (held-out % of residual variance, above the shuffled null):")
    for outcome in ("dk_pts", "pts_per36"):
        weighted = outcome != "dk_pts"
        parts = []
        for r in (1, 2, 3):
            res = evaluate_matchup(S_tr, O_tr, tr[outcome].to_numpy(float),
                                   S_te, O_te, te[outcome].to_numpy(float),
                                   rank=r, ridge=ridge, seed=seed,
                                   w_tr=w_tr_min if weighted else None,
                                   w_te=w_te_min if weighted else None)
            parts.append(f"rank {r}: {(res['r2_with_interaction'] - res['r2_shuffled_null']) * 100:+.3f}%")
        print(f"  {outcome:12} {'   '.join(parts)}")

    # The in-sample ceiling the quoted figures describe, for contrast.
    arch = pd.read_parquet(features_dir / f"archetypes_tier{tier}.parquet",
                           columns=["player_id", "season", "archetype"])
    arch["season"] = arch["season"].map(nxt_map(seasons))
    ceiling_games = games.merge(arch.dropna(subset=["season"]),
                                on=["player_id", "season"], how="inner")
    ceil = variance_ceiling(ceiling_games, ceiling_games["archetype"], seed=seed)
    print(f"\nIn-sample ceiling on the same panel ({ceil['n']:,} player-games, "
          f"{ceil['n_cells']:,} cells) — contemporaneous opponent identity:")
    print(f"  opponent x season                    {ceil['opponent_x_season'] * 100:6.3f}%")
    print(f"  home / away                          {ceil['home_away'] * 100:6.3f}%")
    print(f"  opponent x archetype x season        {ceil['opponent_x_archetype_x_season'] * 100:6.3f}%")
    print(f"    above null (shuffle opponent)      "
          f"{(ceil['opponent_x_archetype_x_season'] - ceil['null_shuffle_opponent']) * 100:+6.3f}%")
    print(f"    above null (shuffle archetype)     "
          f"{(ceil['opponent_x_archetype_x_season'] - ceil['null_shuffle_archetype']) * 100:+6.3f}%")

    print(f"\n→ {dest}")
    return dest


if __name__ == "__main__":
    # Imported through the package path so BilinearMatchup pickles as
    # src.features.opponent.BilinearMatchup — see CLAUDE.md.
    from src.features.opponent import run as _run

    cfg = yaml.safe_load(open("configs/default.yaml"))
    _run(cfg)
