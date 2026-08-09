"""The PCA fingerprint's pure layer — orientation, scaling, loadings, neighbours.

No Streamlit here, so every rule below is exercised directly by `tests/test_dashboard.py`
rather than through a rendered page. `app.py` supplies the frames; this module never
reads a file.

Three things are worth knowing before editing anything here.

**The artifact is one specific decomposition.** Tier A, `within_season`: 30 seasons,
box-score families only, z-scored inside each season so a player is described relative
to his own league rather than to the 3-point revolution. The pooled and Tier B twins
exist on disk and are deliberately not offered — the component titles below are read off
*these* loadings, and a title written for one decomposition is wrong for another.

**A PCA's signs are arbitrary, and the titles are not.** `sklearn` is free to return
−PC1, and a refit could flip any axis without changing the fit at all. So every component
carries an `anchor` feature whose loading defines the labelled direction, and
`orient()` flips the score *and* the loading column whenever a refit lands on the other
sign. Today every anchor already loads positive, so orientation is a no-op — which is the
point: it is a guard, not a correction.

**Only the titles are typed.** Exemplars are derived live from the scores
(`exemplars()`), variance shares from the variance CSV, loadings from the loadings
frame. The provenance rule the old walkthrough enforced still holds; the interpretation
sentence is the one thing no artifact can supply.
"""

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# ── The artifact this view is written against ─────────────────────────────────

SCORES_FILE = "pca_tierA_within_season_scores.parquet"
LOADINGS_FILE = "pca_tierA_within_season_loadings.parquet"
VARIANCE_FILE = "pca_tierA_within_season_variance.csv"

#: Components shown on the radial chart. Ten reaches 67.3% cumulative variance and
#: keeps the spokes 36° apart, which is about as tight as a labelled radial gets.
N_COMPONENTS = 10

#: The radial axis, fixed for every player and season so two fingerprints can be
#: compared by shape alone. Scores outside it are pinned rather than rescaled.
AXIS_LIMIT = 2.0

#: A player-season has to be a real rotation season to name an axis. Without this the
#: extremes are all sub-500-minute players whose rate stats are noise — the clutch
#: components below reach ±18 SD on players nobody would recognise.
EXEMPLAR_MIN_MPG = 24.0
EXEMPLAR_MIN_GP = 45


@dataclass(frozen=True)
class Component:
    """One principal component, with the part a human had to supply.

    `title` is an interpretation of the loadings, in 3–5 words, naming the *positive*
    direction. `anchor` pins which direction that is; `reads` says which loadings the
    title came from, so a reader can check the claim against the bars beside it.
    """

    pc: str
    title: str
    anchor: str
    reads: str


COMPONENTS: tuple[Component, ...] = (
    Component(
        "pc1", "Paint big, not shooter", "adv_reb_pct",
        "Offensive and total rebound share, paint points and 2-point attempt share on "
        "the positive side; 3-point attempt and 3-point scoring share on the negative. "
        "The single largest axis in the matrix, and it is position."),
    Component(
        "pc2", "On-ball scoring load", "adv_usg_pct",
        "Usage rate, points, field-goal attempts and free-throw makes, all loading the "
        "same way. This is volume of offensive responsibility, not efficiency at it."),
    Component(
        "pc3", "Assisted, efficient, not creating", "sco_pct_ast_fgm",
        "Assisted-field-goal share, effective and true shooting, net rating and "
        "plus-minus against assist ratio and assist usage. Finishes possessions other "
        "people made; the negative end starts them."),
    Component(
        "pc4", "Leaky defence, negative impact", "adv_def_rating",
        "Defensive rating and opponent paint points up, defensive win shares, net "
        "rating and plus-minus down. Higher is worse — a defensive rating is points "
        "conceded, so the axis points at the porous end."),
    Component(
        "pc5", "Mid-range at a slow pace", "sco_pct_pts_2pt_mr",
        "Mid-range scoring share, makes and attempts, against team pace and steals. "
        "The half-court two-point game, and the era it belongs to."),
    Component(
        "pc6", "Double-double passing hub", "adv_ast_ratio",
        "Assist ratio, assist-to-turnover and double-doubles up; blocked attempts and "
        "defensive win shares down. A player the offence runs through who is not "
        "getting his own shot swatted."),
    Component(
        "pc7", "Clean finishing, few turnovers", "adv_ts_pct",
        "Field-goal percentage, effective and true shooting, restricted-area accuracy "
        "and fast-break points up; turnover rate and turnovers down. Efficiency "
        "stripped of the volume PC2 already holds."),
    Component(
        "pc8", "Mid-range big who steals", "bas_stl",
        "Mid-range share and attempts alongside steal rate, against true shooting and "
        "free-throw scoring share. The mobile power forward, opposite the small guard "
        "who lives at the line."),
    Component(
        "pc9", "Clutch free-throw volume", "clu_fta",
        "Almost entirely the clutch family — clutch free throws made and attempted, "
        "clutch points, fouls drawn. A few hundred possessions a season, so the tail "
        "is heavy and the extremes are usually small samples rather than players."),
    Component(
        "pc10", "Fast team pace", "adv_pace",
        "Pace, three ways, and little else. A *team* attribute reaching a player row "
        "through the roster join — worth seeing precisely because it is not a skill."),
)

PC_NAMES: tuple[str, ...] = tuple(c.pc for c in COMPONENTS)
BY_PC: dict[str, Component] = {c.pc: c for c in COMPONENTS}

#: Identity columns carried through the PCA, used for the header and the neighbour list.
ID_COLS = ("player_id", "player_name", "season", "team_abbreviation", "age", "gp",
           "min", "dk_pts_per_game")


# ── Orientation ───────────────────────────────────────────────────────────────

def orientation(loadings: pd.DataFrame) -> dict[str, int]:
    """`+1` where the anchor already loads positive, `-1` where a refit flipped it.

    `loadings` is the artifact as written: one row per feature, one column per `pcN`,
    plus a `feature` column.
    """
    indexed = loadings.set_index("feature")
    signs = {}
    for c in COMPONENTS:
        if c.pc not in indexed.columns or c.anchor not in indexed.index:
            signs[c.pc] = 1
            continue
        signs[c.pc] = -1 if float(indexed.at[c.anchor, c.pc]) < 0 else 1
    return signs


def orient(scores: pd.DataFrame, loadings: pd.DataFrame
           ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Flip any component whose anchor loads negative, in both frames together.

    Scores and loadings must be flipped as a pair or the bars stop explaining the
    radius. Distances are unaffected either way, so the neighbour list does not care.
    """
    signs = orientation(loadings)
    scores, loadings = scores.copy(), loadings.copy()
    for pc, sign in signs.items():
        if sign == 1:
            continue
        if pc in scores.columns:
            scores[pc] = scores[pc] * sign
        if pc in loadings.columns:
            loadings[pc] = loadings[pc] * sign
    return scores, loadings, signs


# ── Scaling to standard deviations ────────────────────────────────────────────

def sd_scale(scores: pd.DataFrame, pcs: tuple[str, ...] = PC_NAMES) -> pd.Series:
    """Each component's standard deviation across every player-season.

    A PC score is in the units of the standardized feature space, so PC1's spread is
    ~5.9 and PC10's is ~1.7 — the same raw number means something different on each
    spoke. Dividing by these makes the radial axis one unit, which is what "±2 SD" on
    the chart is denominated in. One scale for all 30 seasons, not one per season:
    `within_season` standardization already removed the era, and the per-season spreads
    differ by ~5%, well inside the width of a plotted line.
    """
    return scores[list(pcs)].std(ddof=1)


def in_sd_units(scores: pd.DataFrame, pcs: tuple[str, ...] = PC_NAMES) -> pd.DataFrame:
    """Every component centred and scaled, so the radial axis reads in SD from the mean.

    The centring is a no-op on the shipped artifact — PCA scores are mean-zero by
    construction — and is done anyway so the axis label is true of whatever frame it is
    handed rather than true by luck.
    """
    columns = list(pcs)
    scale = sd_scale(scores, pcs).replace(0.0, 1.0)
    return (scores[columns] - scores[columns].mean()) / scale


def clamp(values: pd.Series, limit: float = AXIS_LIMIT) -> pd.Series:
    """Pin to the rim rather than rescale — the axis is fixed across all P and S."""
    return values.clip(-limit, limit)


def fingerprint(scores: pd.DataFrame, player_id: int, season: str,
                pcs: tuple[str, ...] = PC_NAMES,
                limit: float = AXIS_LIMIT) -> pd.DataFrame:
    """One player-season as a plottable row per component.

    Carries both the true SD score and the clamped radius, because the chart draws the
    clamped one and the hover has to admit to it.
    """
    row = scores[(scores["player_id"] == player_id) & (scores["season"] == season)]
    if row.empty:
        raise KeyError(f"no player-season {player_id!r} / {season!r} in the scores")
    sd = in_sd_units(scores, pcs).loc[row.index[0]]
    frame = pd.DataFrame({
        "pc": list(pcs),
        "label": [p.upper() for p in pcs],
        "title": [BY_PC[p].title if p in BY_PC else p for p in pcs],
        "sd": sd.reindex(list(pcs)).to_numpy(dtype=float),
    })
    frame["radius"] = clamp(frame["sd"], limit)
    frame["pinned"] = frame["sd"].abs() > limit
    return frame


# ── Loadings ──────────────────────────────────────────────────────────────────

FAMILY = {
    "adv": "advanced", "bas": "base", "bio": "bio", "clu": "clutch",
    "def": "defense", "misc": "misc", "sco": "scoring", "sl": "shot loc",
    "usg": "usage",
}

TOKENS = {
    "pct": "%", "fgm": "FGM", "fga": "FGA", "fg": "FG", "fg3m": "FG3M",
    "fg3a": "FG3A", "ftm": "FTM", "fta": "FTA", "ft": "FT", "oreb": "OREB",
    "dreb": "DREB", "reb": "REB", "ast": "AST", "uast": "unassisted", "stl": "STL",
    "blk": "BLK", "blka": "BLKA", "tov": "TOV", "pf": "PF", "pfd": "PFD",
    "pts": "PTS", "ts": "TS", "efg": "eFG", "dd2": "dbl-dbl", "td3": "trpl-dbl",
    "fb": "fast break", "ra": "restricted area", "mr": "mid-range", "2pt": "2PT",
    "3pt": "3PT", "tm": "team", "ws": "win shares", "e": "est.", "min": "MIN",
    "nba": "NBA", "opp": "opp", "tov_pct": "TOV%",
}


def pretty_feature(name: str) -> str:
    """`usg_pct_oreb` → `usage · % OREB`. Cosmetic; the raw name stays in the hover."""
    head, _, rest = name.partition("_")
    family = FAMILY.get(head)
    if family is None:
        family, rest = "", name
    words = [TOKENS.get(tok, tok.replace("_", " ")) for tok in rest.split("_")]
    body = re.sub(r" %", "%", " ".join(w for w in words if w))
    return f"{family} · {body}" if family else body


def top_loadings(loadings: pd.DataFrame, pc: str, n: int = 8) -> pd.DataFrame:
    """The strongest loadings on `pc`, **balanced across the sign** and sorted signed.

    A plain top-`n` by absolute value is the obvious rule and it is wrong here. PC1's
    eight largest are all positive — rebounding and paint scoring — and its negative
    end, the 3-point attempt and scoring shares, lands ninth and tenth by a thousandth.
    The panel would then read "rebounding big" and drop the "not a shooter" half of an
    axis that is *defined* by the opposition, which is precisely what a component means.

    So each side gets `n // 2` slots and a short side is backfilled by magnitude from
    the other. The result is sorted by signed loading, which makes the bars a diverging
    chart around zero rather than a ranking.
    """
    frame = loadings[["feature", pc]].rename(columns={pc: "loading"})
    ranked = frame.reindex(frame["loading"].abs().sort_values(ascending=False).index)
    half = n // 2
    picked = pd.concat([ranked[ranked["loading"] > 0].head(half),
                        ranked[ranked["loading"] < 0].head(n - half)])
    if len(picked) < n:
        picked = pd.concat([picked, ranked.drop(picked.index).head(n - len(picked))])
    picked = picked.sort_values("loading", ascending=False).reset_index(drop=True)
    picked["pretty"] = picked["feature"].map(pretty_feature)
    return picked


def variance_share(variance: pd.DataFrame) -> dict[str, float]:
    """`{"pc1": 0.233, ...}` from the variance CSV's 1-indexed `component` column."""
    return {f"pc{int(r.component)}": float(r.explained_variance_ratio)
            for r in variance.itertuples()}


# ── Exemplars ─────────────────────────────────────────────────────────────────

def rotation_seasons(scores: pd.DataFrame, min_mpg: float = EXEMPLAR_MIN_MPG,
                     min_gp: int = EXEMPLAR_MIN_GP) -> pd.DataFrame:
    keep = scores[(scores["min"] >= min_mpg) & (scores["gp"] >= min_gp)]
    return keep if len(keep) else scores


def exemplars(scores: pd.DataFrame, pc: str, **kw) -> tuple[str, str]:
    """The rotation player-season at each end of `pc`, as `"Name · season"`.

    Derived rather than typed, so a refit or a new season moves them on its own — the
    component titles are the only hand-written thing on the page.
    """
    pool = rotation_seasons(scores, **kw)
    if pool.empty or pc not in pool.columns:
        return "—", "—"
    hi, lo = pool.loc[pool[pc].idxmax()], pool.loc[pool[pc].idxmin()]
    return (f"{hi['player_name']} · {hi['season']}",
            f"{lo['player_name']} · {lo['season']}")


# ── Nearest neighbours ────────────────────────────────────────────────────────

def neighbors(scores: pd.DataFrame, player_id: int, season: str, k: int = 3,
              pcs: tuple[str, ...] = PC_NAMES,
              same_season_only: bool = False) -> pd.DataFrame:
    """The `k` closest other player-seasons in PCA space.

    Euclidean distance on the **raw** scores, not the SD-scaled ones. Raw distance in
    the retained subspace is distance in the standardized feature space the PCA was fitted
    on, so the high-variance style axes dominate as they should. Scaling each component to
    one SD first would be a Mahalanobis distance and would give PC10 — team pace, 1.9% of
    variance — the same say as PC1.

    The player's own other seasons are always excluded. They are usually his three
    nearest neighbours, which is true and tells you nothing.
    """
    pool = scores[scores["player_id"] != player_id]
    if same_season_only:
        pool = pool[pool["season"] == season]
    target = scores[(scores["player_id"] == player_id) & (scores["season"] == season)]
    if target.empty:
        raise KeyError(f"no player-season {player_id!r} / {season!r} in the scores")
    if pool.empty:
        return pool.assign(distance=[]).head(0)

    cols = [p for p in pcs if p in scores.columns]
    delta = pool[cols].to_numpy(dtype=float) - target[cols].to_numpy(dtype=float)
    out = pool.copy()
    out["distance"] = np.sqrt((delta ** 2).sum(axis=1))
    keep = [c for c in ID_COLS if c in out.columns] + ["distance"]
    return out.nsmallest(k, "distance")[keep].reset_index(drop=True)
