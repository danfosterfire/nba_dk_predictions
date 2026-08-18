"""The game-length head: overtime as a random variable, not a lookup.

Every backtest in this repo reads `game_length` from `data/features/game_length.parquet`,
because in a replay the games already happened. **In a forward simulation nothing knows how
long a game will be**, and both minutes heads need it: `stan_composition` allocates exactly
`5 x game_length` minutes per team-game and `stan_minutes` uses it as binomial trials. So
the simulator needs a game-length *draw*, and it sits upstream of everything else in the
chain.

A point-MLE version used to live in `stan_composition` — `fit_ot_tail` / `sample_game_length`,
two floats fitted on the train seasons. It worked, and it was in the wrong module (the
simulator would have had to import a 9.9-hour head to draw a game length), in the wrong form
(a hardcoded pair of floats is the one simulator input that never went through the posterior
artifact contract), and unconditional (there is a real season trend it could not see). This
module replaces it; the pooled pair survives here as the mandatory no-fit **floor**, so the
gate is against the thing that shipped rather than against nothing.

## Two heads, two existing `.stan` sources, no new one

    does a game go to OT   beta-binomial     betabinomial_glm.stan     season cells
    how many overtimes     beta-geometric    betageometric_duration.stan  one row per depth

Which is the project's factorization argument again: `P(length) = P(any OT) x P(depth | OT)`,
disjoint parameter blocks, independent priors, so the two fits recover the posterior a joint
model would. The classes are reused too — `stan_games_played.BetaBinomialHead` and
`BetaGeometricHead` are exactly these likelihoods with different data, so this module builds
frames and scores them rather than defining a third and fourth head class.

**The beta-binomial's `rho` is the trend-versus-wander answer, measured rather than
asserted.** `src/eda/season_effects.py` exists to draw that distinction; here it falls out of
the same fit that estimates the trend, as the residual season-to-season dispersion once the
slope is removed. **The beta-geometric's Beta frailty is the fix for the plain geometric's
one miss**: pooled, the geometric over-predicts 3OT+ by ~3 games in 2,460, and a frailty on
the continuation hazard is the same device the absence-spell process uses one level down.

## What the ladder is, and what it is not

1. `floor` — the pooled league rate and the pooled continuation, no fitting. The incumbent.
2. `season_trend` — the OT rate on a season index. Expected to win, and this is the second
   head in the project to ship a season term after `min`.
3. `season_trend_matchup` — plus `|prior-season net rating difference|` between the two
   scheduled teams, which is known before the season given the schedule. Speculative.

A season *fixed* effect would be unusable at prediction time; a season *slope* is not, which
is the whole reason the trend is admissible here. It also cannot be a year random effect —
`YearTerm` widens the predictive without shifting it, and the movement measured here is a
shift.

## Size it honestly. This is a tail story, not a mean-effects story

Expected extra minutes per game is `5 x p / (1 - 0.1382)`: 0.345 min at the pooled rate
against 0.302 at the trend-extrapolated 2026-27 rate, so **getting the trend right is worth
0.09% of total minutes**. Nobody should build this expecting a mean effect.

The reason to build it properly is the tail. Overtime is where 40+ minute games come from —
1,650 player-games exceed 48 minutes and the observed maximum is 63.0 — and under a best-ball
weekly max plus a threshold bonus, a star's ceiling week is what decides a 2-of-12 pod. An OT
frequency 17% too high inflates every star's simulated ceiling, which is a shape error in
exactly the statistic the tournament objective is most sensitive to. Roughly 30 collapsed rows
and 2-4 parameters: the cheapest head in the project, and the ladder above is the whole scope.

## The draw is per GAME, shared by both teams

`sample_game_length` draws once per game. Overtime is a property of the *game* — every player
on the floor gets the extra minutes together — so drawing it per team-game would produce a
correct marginal and a wrong joint, and would silently destroy the correlated upside a
same-team stack exists to buy.

Usage:
    python -m src.models.stan_game_length
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.fetch import _season_start_year, _slug, nbastats_dir
from src.features.game_length import OVERTIME_MINUTES, REGULATION_MINUTES
from src.models.games_played import (KAPPA_MAX, KAPPA_MIN, MU_MAX, MU_MIN,
                                     beta_geometric_logpmf, beta_shapes)
from src.models.held_out import selection_split
from src.models.stan_games_played import (GLM_L2, KAPPA_SCALE, BetaBinomialHead,
                                          BetaGeometricHead)
from src.models.stan_utils import diagnostics_frame, thin

TEST_SEASONS = 2

# The season index, and the matchup covariate built off `team_estimated_metrics_*.csv`.
SEASON_COL = "season_start_year"
MATCHUP_COL = "net_rating_gap"

# Cells for the matchup arm. The beta-binomial needs `n > 1` per row for `rho` to be
# identified at all, so the per-game covariate is binned rather than left at one game per
# row; the edges come from the fitting half only.
MATCHUP_BINS = 10

# Classes the PPC reports, matching `stan_composition.ot_tail_check` exactly so the new
# head's counts are directly comparable to the incumbent's 2,310.5 / 128.4 / 18.1 / 2.96.
PPC_CLASSES = ("regulation", "1OT", "2OT", "3OT+")

# Deepest overtime the depth tables enumerate. Nothing in 30 seasons exceeds 4OT, and the
# draw is deliberately NOT capped there: the fitted hazard puts 5OT+ at ~1 in 20,500 games,
# about one every seventeen seasons, which is the right answer for a league whose record is
# a 6OT game. Truncating a tail is how a simulator ends up with a ceiling the real game does
# not have, and the ceiling is the statistic this head exists for.
MAX_DEPTH = 4

# Slates redrawn through `sample_game_length` for the slate-level PPC. 500 x 2,460 games is
# a fifth of a second and puts the Monte Carlo error on the mean OT count near 0.5 games,
# well inside the ~11-game predictive sd it is there to report.
SLATE_SIMS = 500

# Arms in ladder order. `selectable` marks the ones that could ship: the floor is the
# incumbent, and `season_trend_covered` exists only to isolate the matchup term on the
# window where the matchup covariate exists, so neither is a candidate however well it
# scores.
ARMS = {
    "floor": {"selectable": False, "features": ()},
    "season_trend": {"selectable": True, "features": (SEASON_COL,)},
    "season_trend_covered": {"selectable": False, "features": (SEASON_COL,)},
    "season_trend_matchup": {"selectable": True, "features": (SEASON_COL, MATCHUP_COL)},
}
# Arms fitted on the reduced window where prior-season net ratings exist. Kept together so
# the pair is obviously a matched comparison rather than two arms that happen to agree.
COVERED_ARMS = ("season_trend_covered", "season_trend_matchup")

EPS = 1e-12


# ── The frame ─────────────────────────────────────────────────────────────────

def game_frame(cfg: dict) -> pd.DataFrame:
    """One row per regular-season game: its season, its depth, and the matchup gap.

    Regular season only, matching every other fitting frame in this project and the contest
    window — the DK tournament ends 4/4. Playoff overtimes are a different process (no
    back-to-backs, shortened rotations) and 2,440 games of it would tilt a 6% base rate.

    `n_overtimes` is *derived*, not fetched: `make game-length` recovers every game's length
    from summed team minutes / 5, and the two teams are two independent estimates that
    disagree on 0 of 37,986 games.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    season_type = str(cfg.get("stan", {}).get("game_length", {})
                      .get("season_type", "regular"))
    lengths = pd.read_parquet(features_dir / "game_length.parquet")
    reg = lengths[lengths["season_type"] == season_type].copy()
    if not len(reg):
        raise ValueError(f"no {season_type} games in game_length.parquet — run "
                         f"`make game-length`")

    out = reg[["season", "game_id", "n_overtimes"]].reset_index(drop=True)
    out[SEASON_COL] = out["season"].map(_season_start_year).astype(float)
    out["ot"] = (out["n_overtimes"] >= 1).astype(int)
    gap = matchup_gaps(cfg, sorted(out["season"].unique()))
    out = out.merge(gap, on=["season", "game_id"], how="left")
    return out


def matchup_gaps(cfg: dict, seasons: list[str]) -> pd.DataFrame:
    """`|prior-season net rating difference|` per game — legal before the season starts.

    Evenly matched teams are likelier to be tied at the buzzer, and the schedule plus the
    previous season's team ratings are both known in September. That is the *whole* case for
    the covariate, and it is thin, which is why the arm is labelled speculative.

    `team_estimated_metrics_*.csv` starts at 2014-15, so the gap exists for target seasons
    2015-16 on and is NaN before. That is a real coverage limit rather than a missing value
    to impute: filling it with the league mean over eighteen seasons would hand the arm a
    constant column that the season trend then has to share an era with. The arms that use it
    are fitted on the covered window and compared against a same-window control instead.
    """
    raw = Path(cfg["data"]["raw_dir"])
    ratings = {}
    for season in seasons:
        path = nbastats_dir(raw) / f"team_estimated_metrics_{_slug(season)}.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path, usecols=["TEAM_ID", "E_NET_RATING"])
        ratings[season] = dict(zip(frame["TEAM_ID"], frame["E_NET_RATING"]))

    rows = []
    for season in seasons:
        prior = ratings.get(_previous_season(season))
        path = nbastats_dir(raw) / f"game_logs_{_slug(season)}.csv"
        if prior is None or not path.exists():
            continue
        pairs = (pd.read_csv(path, usecols=["GAME_ID", "TEAM_ID"])
                 .drop_duplicates()
                 .rename(columns={"GAME_ID": "game_id", "TEAM_ID": "team_id"}))
        pairs["rating"] = pairs["team_id"].map(prior)
        agg = pairs.groupby("game_id").agg(hi=("rating", "max"), lo=("rating", "min"),
                                           n_teams=("team_id", "size"),
                                           n_rated=("rating", "count"))
        # A game whose two teams are not both rated has no gap. Two teams per game is the
        # invariant the whole construction rests on, so a row that misses it is dropped
        # rather than averaged over however many rows turned up.
        agg = agg[(agg["n_teams"] == 2) & (agg["n_rated"] == 2)]
        rows.append(pd.DataFrame({"season": season, "game_id": agg.index,
                                  MATCHUP_COL: (agg["hi"] - agg["lo"]).to_numpy(float)}))
    if not rows:
        return pd.DataFrame(columns=["season", "game_id", MATCHUP_COL])
    return pd.concat(rows, ignore_index=True)


def _previous_season(season: str) -> str:
    """'2015-16' -> '2014-15'. Season keys go through `fetch`, never through slicing."""
    year = _season_start_year(season) - 1
    return f"{year}-{str(year + 1)[2:]}"


# ── Collapsing to the two fitting frames ──────────────────────────────────────

def matchup_edges(train: pd.DataFrame, n_bins: int = MATCHUP_BINS) -> np.ndarray:
    """Interior quantile edges of the matchup gap, from the FITTING half only.

    Fitted from data, so it goes on the fitting side of the split like every spline knot and
    imputation mean in the repo.
    """
    values = train[MATCHUP_COL].dropna().to_numpy(float)
    if not len(values):
        return np.zeros(0)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    return np.unique(np.quantile(values, quantiles))


def overtime_cells(frame: pd.DataFrame, features: tuple[str, ...] = (SEASON_COL,),
                   edges: np.ndarray | None = None) -> pd.DataFrame:
    """OT games out of games, collapsed to one row per distinct covariate cell.

    Without covariates the beta-binomial likelihood depends on a game only through whether it
    went to overtime, so 35,546 games collapse to ~30 season rows and the head costs seconds.
    That collapse is exact for the mean; `rho` is then the dispersion *between cells*, which
    for season cells is exactly the quantity "is the era movement a trend or a wander" asks
    about.

    The matchup arm bins its covariate for a mechanical reason: at one game per row the
    beta-binomial has `n = 1` everywhere, where it degenerates to a Bernoulli and `rho` is not
    identified at all.
    """
    work = frame.copy()
    keys = ["season", SEASON_COL]
    if MATCHUP_COL in features:
        if edges is None:
            raise ValueError("the matchup arm needs bin edges fitted on the training half")
        work = work[work[MATCHUP_COL].notna()]
        work["matchup_bin"] = np.searchsorted(np.asarray(edges, dtype=float),
                                              work[MATCHUP_COL].to_numpy(float),
                                              side="right")
        keys = keys + ["matchup_bin"]

    agg = {"y": ("ot", "sum"), "n": ("ot", "size")}
    if MATCHUP_COL in features:
        # The cell's own mean gap, so the covariate the head sees is the covariate the games
        # in that cell actually carried rather than the bin's midpoint.
        agg[MATCHUP_COL] = (MATCHUP_COL, "mean")
    out = work.groupby(keys, as_index=False, sort=True).agg(**agg)
    out["y"] = out["y"].astype(int)
    out["n"] = out["n"].astype(int)
    return out.reset_index(drop=True)


def depth_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct overtime depth, with a multiplicity weight.

    `t` is the number of overtime periods, so the support starts at 1 by construction and the
    beta-geometric reads directly: `mu = P(T = 1)` is the probability that a game which went
    to overtime ends after one. Nothing is censored — a game that finished, finished — and
    nothing is left-truncated, so `betageometric_duration.stan`'s `H_open = 0` branch disables
    the in-progress offset exactly.
    """
    ot = frame[frame["n_overtimes"] >= 1]
    out = (ot.groupby("n_overtimes", as_index=False).size()
           .rename(columns={"n_overtimes": "t", "size": "w"}))
    out["w"] = out["w"].astype(float)
    out["censored"] = 0.0
    out["truncated"] = 0.0
    return out.sort_values("t").reset_index(drop=True)


# ── The no-fit floor ──────────────────────────────────────────────────────────

class FloorGameLength:
    """The pooled league rate and the pooled continuation. No fitting, no covariates.

    This *is* the retired `stan_composition.fit_ot_tail`, moved rather than reimplemented, so
    "does the fitted head beat the thing that shipped" and "does it clear the no-fit floor"
    are the same question with one answer. Two parameters cover 3OT and 4OT for free, because
    the continuation probability is near-constant in depth (0.1382 / 0.1438 / 0.1190 over the
    full 35,546 games), which is what makes a geometric in depth the right shape.
    """

    name = "floor"

    def fit(self, frame: pd.DataFrame) -> "FloorGameLength":
        k = frame["n_overtimes"].to_numpy(int)
        n_ot = int((k >= 1).sum())
        self.p_any_ot = float((k >= 1).mean()) if len(k) else 0.0
        self.p_more_ot = float((k >= 2).sum() / max(n_ot, 1))
        self.n_games, self.n_ot_games = int(len(k)), n_ot
        return self

    def p_ot(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), self.p_any_ot)

    def log_depth_pmf(self, t: np.ndarray) -> np.ndarray:
        """`log P(T = t)` for a plain geometric in depth, support t >= 1."""
        t = np.asarray(t, dtype=float)
        c = float(np.clip(self.p_more_ot, EPS, 1 - EPS))
        return (t - 1.0) * np.log(c) + np.log1p(-c)


# ── The fitted arms ───────────────────────────────────────────────────────────

def fit_overtime(cells: pd.DataFrame, features: tuple[str, ...], name: str,
                 cfg_stan: dict, l2: float = GLM_L2) -> BetaBinomialHead:
    """One beta-binomial GLM over OT games out of games.

    `stan_games_played.BetaBinomialHead` is that likelihood with different data — the same
    reuse `betabinomial_glm.stan` itself represents — so this is a call, not a fifth head
    class. It fits with `S = 0`: the season term here is a *slope on a season index*, which is
    usable at prediction time, and not a year random effect, which would widen the predictive
    without shifting it.
    """
    return BetaBinomialHead(list(features), name, l2=l2,
                            chains=int(cfg_stan.get("chains", 4)),
                            warmup=int(cfg_stan.get("warmup", 1000)),
                            samples=int(cfg_stan.get("samples", 1000)),
                            seed=int(cfg_stan.get("seed", 42))).fit(cells, "y", "n")


def fit_depth(rows: pd.DataFrame, cfg_stan: dict, kappa_scale: float = KAPPA_SCALE,
              name: str = "game_length/depth") -> BetaGeometricHead:
    """The overtime-depth head — intercept-only, shared by every fitted arm.

    Depth does not depend on the arm: the ladder is about *whether* a game goes to overtime,
    and the continuation hazard is measured to be flat in depth. One fit, reused, so the
    ladder's arms differ in the one thing they are supposed to differ in.
    """
    return BetaGeometricHead([], name, kappa_scale=kappa_scale,
                             chains=int(cfg_stan.get("chains", 4)),
                             warmup=int(cfg_stan.get("warmup", 1000)),
                             samples=int(cfg_stan.get("samples", 1000)),
                             seed=int(cfg_stan.get("seed", 42))).fit(rows)


def ot_probabilities(head: BetaBinomialHead, cells: pd.DataFrame,
                     idx: np.ndarray | None = None) -> np.ndarray:
    """(draws x rows) `P(a single game on this row goes to OT)`.

    A beta-binomial at `n = 1` has `P(y = 1) = a / (a + b) = mu` exactly, so the season-level
    frailty drops out of a *single game's* marginal and only shows up in the joint over a
    season. That is the right split: `rho` is what makes a whole simulated season's OT rate
    wander, and it must not also move the per-game probability.

    Works on cells or on raw games — the head reads only its feature columns, and both frames
    carry them.
    """
    draws = np.arange(len(head.alpha_draws)) if idx is None else np.asarray(idx)
    mus = []
    for draw in draws:
        a, b = head.shapes(cells, int(draw))
        mus.append(a / (a + b))
    return np.asarray(mus)


def depth_pmf_draws(head: BetaGeometricHead, t: np.ndarray) -> np.ndarray:
    """(draws x len(t)) `P(T = t)` under the beta-geometric, per posterior draw.

    Through the head's own `shapes` and `games_played.beta_geometric_logpmf`, which is the
    Stan target verbatim and is already pinned by `tests/test_games_played.py` — there is no
    second implementation of the density here.
    """
    t = np.asarray(t, dtype=float)
    probe = pd.DataFrame({"t": t})
    out = np.empty((len(head.alpha_draws), len(t)))
    for draw in range(len(head.alpha_draws)):
        a, b = head.shapes(probe, draw)
        out[draw] = np.exp(beta_geometric_logpmf(t, a / (a + b), a + b))
    return out


def depth_pmf(head: BetaGeometricHead, max_depth: int) -> np.ndarray:
    """Posterior-mean `P(T = d)` for `d = 1..max_depth`, indexable by depth.

    Evaluated on the handful of distinct depths rather than on every row: the head is
    intercept-only, so a per-row evaluation would be tens of millions of `betaln` calls for
    four distinct answers.
    """
    return depth_pmf_draws(head, np.arange(1, max_depth + 1)).mean(axis=0)


# ── Scoring ───────────────────────────────────────────────────────────────────

def log_lik(p_ot: np.ndarray, depth_logpmf: np.ndarray,
            k: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`(onset, depth)` log-density per game, the two exact factors of `log P(length)`.

    `log P(length) = log P(any OT) + 1{OT} log P(depth | OT)` factorizes with no remainder, so
    keeping the halves apart costs nothing and says which one moved. The depth term is zero
    on a regulation game because there is no depth to score, not because it was dropped.
    """
    k = np.asarray(k, dtype=int)
    p = np.clip(np.asarray(p_ot, dtype=float), EPS, 1 - EPS)
    ot = k >= 1
    onset = np.where(ot, np.log(p), np.log1p(-p))
    return onset, np.where(ot, np.asarray(depth_logpmf, dtype=float), 0.0)


def score(p_ot: np.ndarray, depth_logpmf: np.ndarray, k: np.ndarray) -> dict:
    """Held-out log-likelihood per game, split into the two factors it is made of.

    The onset half is per *game*; the depth half is per *overtime game*, since that is the
    population it is estimated on and a per-game average of it would mostly measure the OT
    rate a second time.
    """
    k = np.asarray(k, dtype=int)
    p = np.clip(np.asarray(p_ot, dtype=float), EPS, 1 - EPS)
    ot = k >= 1
    onset, depth = log_lik(p_ot, depth_logpmf, k)
    return {
        "n_games": int(len(k)),
        "n_ot_games": int(ot.sum()),
        "ll_per_game": float((onset + depth).mean()) if len(k) else float("nan"),
        "ll_onset_per_game": float(onset.mean()) if len(k) else float("nan"),
        "ll_depth_per_ot_game": (float(depth[ot].mean()) if ot.any() else float("nan")),
        "pred_ot_rate": float(p.mean()),
        "obs_ot_rate": float(ot.mean()) if len(k) else float("nan"),
    }


def ppc(p_ot: np.ndarray, depth_pmf: dict[int, float], k: np.ndarray,
        arm: str) -> pd.DataFrame:
    """Predicted vs observed counts in the four OT classes `ot_tail_check` reported.

    Same four classes, same population, so the row for `floor` reproduces the incumbent's
    2,310.5 / 128.4 / 18.1 / 2.96 and the fitted arms are read straight against it.
    """
    k = np.asarray(k, dtype=int)
    n = len(k)
    expected_ot = float(np.sum(np.clip(p_ot, 0.0, 1.0)))
    rows = []
    for label in PPC_CLASSES:
        if label == "regulation":
            predicted, observed = n - expected_ot, int((k == 0).sum())
        elif label == "3OT+":
            tail = 1.0 - sum(depth_pmf[d] for d in (1, 2))
            predicted, observed = expected_ot * tail, int((k >= 3).sum())
        else:
            depth = int(label[0])
            predicted, observed = expected_ot * depth_pmf[depth], int((k == depth).sum())
        rows.append({"variant": arm, "class": label, "observed": observed,
                     "predicted": float(predicted), "n_games": n})
    out = pd.DataFrame(rows)
    out["abs_error"] = (out["observed"] - out["predicted"]).abs()
    return out


def bootstrap_delta(ll_arm: np.ndarray, ll_floor: np.ndarray, seed: int = 42,
                    draws: int = 2000) -> tuple[float, float]:
    """95% interval on the per-game log-likelihood margin over the floor, paired by game.

    The margin here is small enough that quoting it as a point estimate would be the mistake
    this repo has already made four times — every one of those reversals turned on a
    sub-1% margin nobody had put an interval around. Paired, because the two models score the
    *same* games and the pairing removes almost all of the variance.
    """
    delta = np.asarray(ll_arm, dtype=float) - np.asarray(ll_floor, dtype=float)
    if not len(delta):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = delta[rng.integers(0, len(delta), size=(draws, len(delta)))].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def simulated_slate(ot_head: BetaBinomialHead, depth_head: BetaGeometricHead,
                    games: pd.DataFrame, seed: int = 42,
                    sims: int = SLATE_SIMS) -> dict:
    """Redraw the whole scoring slate `sims` times **through `sample_game_length` itself**.

    Two jobs, and they are why this goes through the shipped sampler rather than a closed
    form beside it.

    It is the only check on `rho`. A per-game log-likelihood cannot see the season frailty —
    the frailty leaves the marginal alone by construction — so without a slate-level draw the
    head's overdispersion parameter would be fitted, reported and never compared to anything.

    And it is the only check on the *sampler*. Everything else here scores probabilities; the
    simulator will call `sample_game_length`, and a wiring fault in it (a wrong grid step, a
    depth drawn per team, a frailty applied twice) would leave every probability in this
    module correct and every simulated season wrong. One posterior draw per simulated slate,
    which is how the season simulator will use it.
    """
    inputs = draw_inputs(ot_head, depth_head, games, keep=sims)
    rng = np.random.default_rng(seed)
    n_games = len(games)
    counts = np.zeros((sims, len(PPC_CLASSES)))
    total = np.zeros(sims)
    for s in range(sims):
        length = sample_game_length(rng, n_games, inputs, draw=s)
        k = np.rint((length - REGULATION_MINUTES) / OVERTIME_MINUTES).astype(int)
        counts[s] = [(k == 0).sum(), (k == 1).sum(), (k == 2).sum(), (k >= 3).sum()]
        total[s] = float((k >= 1).sum())
    return {"classes": counts.mean(axis=0),
            "lo90": float(np.quantile(total, 0.05)),
            "hi90": float(np.quantile(total, 0.95)),
            "mean_ot": float(total.mean()), "sims": sims}


# ── The simulator's entry point ───────────────────────────────────────────────

def draw_inputs(ot_head, depth_head, frame: pd.DataFrame,
                keep: int | None = None) -> dict:
    """Everything `sample_game_length` needs, as arrays indexed by posterior draw.

    `frame` carries one row per *game to be simulated* — for the shipped arm that is a single
    row per season, since the only covariate is the season index. Built once per simulated
    season and reused across every game in it.

    The two heads are thinned to the same count so that draw `d` means one thing. They are
    independent posteriors — the factorization is exact — so any pairing is legitimate; what
    is not legitimate is a pairing that silently changes length between the two.
    """
    n_ot = len(ot_head.alpha_draws)
    n_depth = len(depth_head.alpha_draws)
    kept = int(keep) if keep else min(n_ot, n_depth)
    ot_idx, depth_idx = thin(n_ot, kept), thin(n_depth, kept)
    alpha = np.asarray(depth_head.alpha_draws, dtype=float)[depth_idx]
    return {
        "p_ot": ot_probabilities(ot_head, frame)[ot_idx],
        "rho": np.asarray(ot_head.rho_draws, dtype=float)[ot_idx],
        "mu_depth": 1.0 / (1.0 + np.exp(-np.clip(alpha, -30, 30))),
        "kappa_depth": np.asarray(depth_head.kappa_draws, dtype=float)[depth_idx],
    }


def posterior_inputs(ot, depth, frame: pd.DataFrame) -> dict:
    """`draw_inputs`, from the two persisted artifacts instead of two live heads.

    This is the path the season simulator takes: `make posteriors` writes `game_length_ot`
    and `game_length_depth` to `data/features/posteriors/<window>/`, and everything
    downstream is numpy. `ot` and `depth` are `posteriors.PosteriorArtifact`s, duck-typed
    rather than imported so this module stays importable without them.
    """
    return {
        "p_ot": ot.mu_draws(frame),
        "rho": np.asarray(ot.draws["rho_draws"], dtype=float),
        # The depth head is intercept-only, so one row is the whole answer; taking a column
        # of a (draws x rows) block of identical values would just be wasted work.
        "mu_depth": depth.mu_draws(frame.iloc[:1])[:, 0],
        "kappa_depth": np.asarray(depth.draws["kappa_draws"], dtype=float),
    }


def forward_cells(season: str, n_games: int = 1,
                  matchup_gap: float | None = None) -> pd.DataFrame:
    """The design frame a FORWARD season needs — the covariates, with no outcomes.

    The whole claim of a season slope is that it extrapolates, and this is what a consumer
    hands the head to collect on that: `forward_cells("2026-27", 1230)` is the 2026-27
    regular season as far as this head is concerned. Nothing here reads a game that has
    happened, which is why the production board can be built in September.
    """
    out = pd.DataFrame({"season": [season] * int(n_games)})
    out[SEASON_COL] = float(_season_start_year(season))
    if matchup_gap is not None:
        out[MATCHUP_COL] = float(matchup_gap)
    return out


def sample_game_length(rng: np.random.Generator, n_games: int, draws: dict,
                       draw: int | None = None) -> np.ndarray:
    """Game lengths on the 48 / 53 / 58 / ... grid — **one draw per game**.

    Call this **once per game and share the result across both teams**. Overtime is a property
    of the game: every player on the floor gets the extra minutes together, so a per-team-game
    draw would give a correct marginal and a wrong joint, and would destroy the correlated
    upside a same-team stack is drafted for.

    Three levels of randomness, and they are different things:

    - **the posterior draw** — which coefficients this simulated season believes;
    - **the season frailty** — `rho`, drawn once per call, so a whole simulated season's OT
      rate wanders together the way the fitted between-season dispersion says it does;
    - **the per-game frailty** on the continuation hazard, drawn per overtime game, which is
      the beta-geometric's Beta integrated back in.

    `draws` comes from `draw_inputs`. `p_ot` may be one probability for the slate or one per
    game; the season frailty is applied as a shared shift on the logit either way, which is
    exactly a redrawn rate in the constant case.
    """
    n_draws = len(draws["rho"])
    d = int(rng.integers(n_draws)) if draw is None else int(draw) % n_draws

    p = np.atleast_1d(np.asarray(draws["p_ot"][d], dtype=float))
    if p.size not in (1, n_games):
        # Loudly, because the quiet version of this is a simulator that truncates a
        # 1,230-game slate to the handful of cells the head was fitted on and reports a
        # perfectly plausible season.
        raise ValueError(
            f"`p_ot` carries {p.size} probabilities for {n_games} games. It must be one "
            f"probability for the slate or one per game — a covariate arm's draw needs "
            f"`draw_inputs` on a per-GAME frame (`forward_cells`), not on the collapsed "
            f"cells the head was fitted from.")
    p = np.clip(np.broadcast_to(p, (n_games,)), EPS, 1 - EPS)
    rho = float(np.clip(draws["rho"][d], EPS, 1 - EPS))
    scale = (1.0 - rho) / rho
    bar = float(np.clip(p.mean(), EPS, 1 - EPS))
    season_p = float(np.clip(rng.beta(bar * scale, (1.0 - bar) * scale), EPS, 1 - EPS))
    shift = np.log(season_p / (1 - season_p)) - np.log(bar / (1 - bar))
    p_game = 1.0 / (1.0 + np.exp(-(np.log(p / (1 - p)) + shift)))

    n_ot = np.zeros(n_games, dtype=int)
    any_ot = rng.random(n_games) < p_game
    if any_ot.any():
        mu = float(np.clip(draws["mu_depth"][d], MU_MIN, MU_MAX))
        kappa = float(np.clip(draws["kappa_depth"][d], KAPPA_MIN, KAPPA_MAX))
        a, b = beta_shapes(np.full(int(any_ot.sum()), mu),
                           np.full(int(any_ot.sum()), kappa))
        # The frailty is per overtime game, not per season: it is that game's own
        # propensity to stay tied. Clipped away from 0 only because a beta draw of exactly
        # 0.0 would make `geometric` diverge, never to bound the model.
        hazard = np.clip(rng.beta(a, b), 1e-6, 1.0)
        n_ot[any_ot] = rng.geometric(hazard)
    return REGULATION_MINUTES + OVERTIME_MINUTES * n_ot


# ── The sweep ─────────────────────────────────────────────────────────────────

def sweep(train: pd.DataFrame, val: pd.DataFrame, cfg_stan: dict) -> tuple:
    """Every arm on the VALIDATION split. The test seasons are never materialized.

    `selection_split` upstream never hands them over, so there is no test column to be
    tempted by — the discipline failure this repo has on record was a gate specified with
    test figures as its bars.
    """
    cfg_gl = cfg_stan.get("game_length", {})
    n_bins = int(cfg_gl.get("matchup_bins", MATCHUP_BINS))
    kappa_scale = float(cfg_gl.get("kappa_scale", KAPPA_SCALE))
    arms = [a for a in cfg_gl.get("arms", list(ARMS)) if a in ARMS]
    if "season_trend_matchup" in arms and "season_trend_covered" not in arms:
        # The matchup arm is only interpretable against a same-window control, since its
        # frame is eleven seasons and the trend arm's is thirty.
        arms.insert(arms.index("season_trend_matchup"), "season_trend_covered")

    seed = int(cfg_stan.get("seed", 42))
    k_val = val["n_overtimes"].to_numpy(int)
    edges = matchup_edges(train, n_bins)
    covered = train[MATCHUP_COL].notna()

    floor = FloorGameLength().fit(train)
    geometric = np.exp(floor.log_depth_pmf(np.arange(1, MAX_DEPTH + 1)))
    floor_pmf = {d: float(geometric[d - 1]) for d in (1, 2, 3)}
    depth = fit_depth(depth_rows(train), cfg_stan, kappa_scale)
    frailty = depth_pmf(depth, MAX_DEPTH)
    fitted_pmf = {d: float(frailty[d - 1]) for d in (1, 2, 3)}
    # Indexed off the four distinct depths rather than evaluated per row: both depth models
    # are intercept-only, so a per-row pass would be millions of `betaln` calls for four
    # answers. The `- 1` is the shift from a 1-based depth to a 0-based index.
    def depth_ll(k: np.ndarray, pmf: np.ndarray) -> np.ndarray:
        return np.log(np.clip(pmf[np.maximum(np.asarray(k, int), 1) - 1], EPS, None))

    base_onset, base_depth = log_lik(floor.p_ot(val),
                                     depth_ll(k_val, geometric), k_val)
    base_total = base_onset + base_depth

    rows, checks, diagnostics, heads, slates = [], [], [depth.diagnostics], {}, {}
    for arm in arms:
        features = ARMS[arm]["features"]
        if arm == "floor":
            p_val = floor.p_ot(val)
            scored = score(p_val, depth_ll(k_val, geometric), k_val)
            rows.append({"variant": arm, "n_features": 0,
                         "n_fit_games": floor.n_games,
                         "n_fit_cells": 0, "p_any_ot": floor.p_any_ot,
                         "p_more_ot": floor.p_more_ot, "rho": np.nan,
                         "rho_p05": np.nan, "rho_p95": np.nan,
                         "season_dispersion": np.nan, "slope_per_season": np.nan,
                         "matchup_per_point": np.nan, "p_ot_2026_27": np.nan,
                         "ot_lo90": np.nan, "ot_hi90": np.nan,
                         "sim_ot_games": np.nan,
                         "ll_delta_lo95": 0.0, "ll_delta_hi95": 0.0, **scored})
            checks.append(ppc(p_val, floor_pmf, k_val, arm))
            continue

        fit_frame = train[covered] if arm in COVERED_ARMS else train
        val_frame = val[val[MATCHUP_COL].notna()] if arm in COVERED_ARMS else val
        cells_fit = overtime_cells(fit_frame, features, edges)
        cells_val = overtime_cells(val_frame, features, edges)
        head = fit_overtime(cells_fit, features, f"game_length/{arm}", cfg_stan)
        diagnostics.append(head.diagnostics)
        heads[arm] = head

        per_cell = ot_probabilities(head, cells_val).mean(axis=0)
        p_val = _expand(per_cell, cells_val, val_frame, features, edges)
        k_arm = val_frame["n_overtimes"].to_numpy(int)
        scored = score(p_val, depth_ll(k_arm, frailty), k_arm)
        onset, dep = log_lik(p_val, depth_ll(k_arm, frailty), k_arm)
        paired = base_total if len(val_frame) == len(val) else None
        lo, hi = (bootstrap_delta(onset + dep, paired, seed) if paired is not None
                  else (float("nan"), float("nan")))
        slate = simulated_slate(head, depth, val_frame, seed=seed)
        slates[arm] = slate
        rows.append({"variant": arm, "n_features": len(features),
                     "n_fit_games": int(cells_fit["n"].sum()),
                     "n_fit_cells": len(cells_fit),
                     "p_any_ot": float(np.mean(p_val)), "p_more_ot": np.nan,
                     "ot_lo90": slate["lo90"], "ot_hi90": slate["hi90"],
                     "sim_ot_games": slate["mean_ot"],
                     "ll_delta_lo95": lo, "ll_delta_hi95": hi,
                     **_dispersion(head, cells_val),
                     **_unstandardized(head, features),
                     "p_ot_2026_27": _forward_rate(head, cells_val, features),
                     **scored})
        checks.append(ppc(p_val, fitted_pmf, k_arm, arm))

    table = pd.DataFrame(rows).merge(
        pd.concat(checks, ignore_index=True)
        .groupby("variant", as_index=False)["abs_error"].sum()
        .rename(columns={"abs_error": "class_abs_error"}), on="variant", how="left")

    # Selected on the ONSET half, because that is the only half the arms differ in — every
    # fitted arm shares one depth head. Scoring the complete model would let depth noise on
    # 138 overtime games decide a comparison about 2,460 games' worth of OT rate.
    candidates = table[table["variant"].map(lambda a: ARMS[a]["selectable"])]
    best = candidates.loc[candidates["ll_onset_per_game"].idxmax(), "variant"]
    table["selected"] = table["variant"] == best
    floor_row = table[table["variant"] == "floor"].iloc[0]
    # The gate is on the COMPLETE model against the complete floor, which is what the
    # simulator draws from, and on the OT-class counts against the incumbent's own row.
    table["beats_floor"] = table["ll_per_game"] > floor_row["ll_per_game"]
    table.loc[table["variant"] == "floor", "beats_floor"] = True
    table["beats_incumbent_counts"] = (table["class_abs_error"]
                                       <= floor_row["class_abs_error"])
    return (table, pd.concat(checks, ignore_index=True), diagnostics, heads, depth,
            floor, slates)


def _dispersion(head: BetaBinomialHead, cells: pd.DataFrame) -> dict:
    """`rho` with its posterior spread, and what it implies at a season's game count.

    Reported with quantiles rather than as a point, because on ~30 cells this parameter is
    weakly identified and the uniform prior `betabinomial_glm.stan` puts on `rho` leans
    upward. Quoting the posterior mean alone would turn a prior into a measurement.
    """
    r = np.asarray(head.rho_draws, dtype=float)
    n = float(cells["n"].mean()) if len(cells) else 1.0
    return {"rho": float(r.mean()), "rho_p05": float(np.quantile(r, 0.05)),
            "rho_p95": float(np.quantile(r, 0.95)),
            "season_dispersion": float(1.0 + (n - 1.0) * np.median(r))}


def _forward_rate(head: BetaBinomialHead, cells: pd.DataFrame,
                  features: tuple[str, ...]) -> float:
    """What this fit extrapolates to for 2026-27 — the season the project is aimed at.

    Not a production number: production refits at the `full` window through
    `make posteriors`. It is here because a season slope's whole claim is that it
    extrapolates, and a head that ships one should print where it lands.
    """
    if SEASON_COL not in features or not len(cells):
        return float("nan")
    probe = cells.iloc[[0]].copy()
    probe[SEASON_COL] = float(_season_start_year("2026-27"))
    if MATCHUP_COL in features:
        weights = cells["n"].to_numpy(float)
        probe[MATCHUP_COL] = float(np.average(cells[MATCHUP_COL], weights=weights))
    return float(ot_probabilities(head, probe).mean())


def _expand(per_cell: np.ndarray, cells: pd.DataFrame, frame: pd.DataFrame,
            features: tuple[str, ...], edges: np.ndarray) -> np.ndarray:
    """Cell-level probabilities pushed back out to one value per game."""
    work = frame.copy()
    if MATCHUP_COL in features:
        work["matchup_bin"] = np.searchsorted(np.asarray(edges, dtype=float),
                                              work[MATCHUP_COL].to_numpy(float),
                                              side="right")
        keys = ["season", "matchup_bin"]
    else:
        keys = ["season"]
    lookup = cells[keys].copy()
    lookup["p"] = per_cell
    return work.merge(lookup, on=keys, how="left")["p"].to_numpy(float)


def _unstandardized(head: BetaBinomialHead, features: tuple[str, ...]) -> dict:
    """The fitted slopes back on their natural scale — per season, and per rating point.

    The head standardizes its design, so `beta` is per standard deviation of a training
    column and is not comparable to the −0.00893 per season the plan measured. Dividing by
    the scaler's own scale is the whole conversion, and doing it here means the artifact
    carries a number a reader can check against the plan.
    """
    beta = head.beta_draws.mean(axis=0)
    scale = np.asarray(head.scaler.scale_, dtype=float)
    named = {name: float(beta[i] / scale[i]) for i, name in enumerate(features)}
    return {"slope_per_season": named.get(SEASON_COL, np.nan),
            "matchup_per_point": named.get(MATCHUP_COL, np.nan)}


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_stan = cfg.get("stan", {})
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", TEST_SEASONS))

    print("Stan game-length head — overtime is a random variable forward, not a lookup")
    frame = game_frame(cfg)
    train, val = selection_split(frame, test_seasons)
    print(f"  {len(frame):,} regular-season games over "
          f"{frame['season'].nunique()} seasons. The test split is LOCKED — this sweep "
          f"fits and\n  scores VALIDATION only (src/models/held_out.py).")
    print(f"  {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    covered = int(train[MATCHUP_COL].notna().sum())
    print(f"  matchup covariate covers {covered:,} of {len(train):,} training games "
          f"({covered / max(len(train), 1):.0%}) —\n  `team_estimated_metrics_*.csv` "
          f"starts at 2014-15, so the two matchup arms fit a shorter window\n  and are "
          f"compared against `season_trend_covered` on exactly those rows.")

    table, checks, diagnostics, heads, depth, floor, slates = sweep(train, val, cfg_stan)

    print("\nArm ladder (validation log-likelihood per game, higher is better; the arms "
          "differ\nonly in the ONSET half, which is what selection reads):")
    print(table[["variant", "n_features", "n_fit_games", "ll_onset_per_game",
                 "ll_per_game", "pred_ot_rate", "class_abs_error",
                 "selected", "beats_floor"]].round(5).to_string(index=False))
    selected = table.loc[table["selected"], "variant"].iloc[0]
    chosen = table[table["variant"] == selected].iloc[0]
    floor_row = table[table["variant"] == "floor"].iloc[0]
    print(f"\n  Selected on VALIDATION: {selected}. There is no test column.")
    print(f"  vs the no-fit floor, onset half: {chosen['ll_onset_per_game']:.6f} against "
          f"{floor_row['ll_onset_per_game']:.6f} "
          f"({chosen['ll_onset_per_game'] - floor_row['ll_onset_per_game']:+.6f} nats "
          f"per game)")
    print(f"  vs the no-fit floor, complete model: {chosen['ll_per_game']:.6f} against "
          f"{floor_row['ll_per_game']:.6f} "
          f"({chosen['ll_per_game'] - floor_row['ll_per_game']:+.6f}), 95% paired "
          f"bootstrap\n  [{chosen['ll_delta_lo95']:+.6f}, "
          f"{chosen['ll_delta_hi95']:+.6f}] — a margin this small is not a result on its "
          f"own,\n  which is why the gate is the OT-class counts below.")
    print(f"  predicted OT rate {chosen['pred_ot_rate']:.4f} against the floor's "
          f"{floor_row['pred_ot_rate']:.4f} and an observed "
          f"{chosen['obs_ot_rate']:.4f}")
    if not bool(chosen["beats_floor"]):
        print("  /!\\  The selected arm does NOT clear the no-fit floor. Per CLAUDE.md "
              "that is not a model.")

    print(f"\nOT-class counts on {int(floor_row['n_games']):,} validation games — the "
          f"gate is the incumbent's own row:")
    wide = checks.pivot(index="class", columns="variant", values="predicted") \
        .reindex(PPC_CLASSES)
    wide.insert(0, "observed",
                checks[checks["variant"] == "floor"].set_index("class")["observed"]
                .reindex(PPC_CLASSES).astype(int))
    print(wide.round(1).to_string())
    print(f"  summed |observed - predicted|: "
          f"{', '.join(f'{r.variant} {r.class_abs_error:.2f}' for r in table.itertuples())}")
    if table["n_games"].nunique() > 1:
        # Only possible if a matchup arm lost validation games to a missing prior-season
        # rating, which would make the summed errors above populations of different sizes.
        print("  /!\\  the arms did NOT all score the same validation games "
              f"({dict(zip(table['variant'], table['n_games']))}) — the class errors are "
              "directly comparable")

    matchup = table[table["variant"] == "season_trend_matchup"]
    control = table[table["variant"] == "season_trend_covered"]
    if len(matchup) and len(control):
        gain = float(matchup["ll_per_game"].iloc[0] - control["ll_per_game"].iloc[0])
        print(f"\nThe matchup arm against its same-window control: "
              f"{gain:+.6f} nats per game\n  (coefficient "
              f"{float(matchup['matchup_per_point'].iloc[0]):+.5f} per net-rating point). "
              f"Speculative by construction —\n  evenly matched teams are likelier to be "
              f"tied at the buzzer, and that is the whole case.")

    print(f"\nTrend or wander — the beta-binomial's own overdispersion answers it:")
    print(f"  slope {chosen['slope_per_season']:+.5f} logit per season, against a "
          f"residual season rho of\n  {chosen['rho']:.2e} "
          f"[{chosen['rho_p05']:.2e}, {chosen['rho_p95']:.2e}] — "
          f"{chosen['season_dispersion']:.2f}x binomial at the posterior median. The era "
          f"movement is in the\n  SLOPE, not in the residual spread, which is why a season "
          f"slope ships and a year random\n  effect does not: a slope extrapolates to a "
          f"season that has not happened and a fitted\n  `year_z` does not.")
    print(f"  Honest about that rho: on {int(chosen['n_fit_cells'])} cells it is weakly "
          f"identified and `betabinomial_glm.stan`\n  leaves it a uniform prior on "
          f"(1e-6, 0.95), which leans upward. Read it as an upper bound\n  on the wander "
          f"rather than a measurement of it.")
    print(f"  Extrapolated to 2026-27 this fit reads {chosen['p_ot_2026_27']:.4f} "
          f"against the floor's {floor_row['p_any_ot']:.4f}\n  (the production board "
          f"refits at the `full` window through `make posteriors`).")

    slate = slates[selected]
    analytic = (checks[checks["variant"] == selected].set_index("class")["predicted"]
                .reindex(PPC_CLASSES).to_numpy(float))
    print(f"\nSlate PPC — the validation season redrawn {slate['sims']:,} times THROUGH "
          f"`sample_game_length`,\none posterior draw per slate, which is how the season "
          f"simulator will call it:")
    print(pd.DataFrame({"class": list(PPC_CLASSES),
                        "observed": wide["observed"].to_numpy(),
                        "analytic": analytic,
                        "sampled": slate["classes"]}).round(1).to_string(index=False))
    print(f"  90% interval on the OT count, WITH the season frailty: "
          f"[{slate['lo90']:.0f}, {slate['hi90']:.0f}] against {int(val['ot'].sum())} "
          f"observed.\n  Agreement between the analytic and sampled columns is the check "
          f"on the SAMPLER — a per-team\n  draw or a wrong grid step would leave every "
          f"probability above right and every season wrong.")

    depth_table = _depth_table(train, val, floor, depth)
    print("\nOvertime depth — the plain geometric against the Beta frailty:")
    print(depth_table.round(4).to_string(index=False))
    _depth_verdict(depth_table, depth)

    p_sel = float(chosen["pred_ot_rate"])
    extra = OVERTIME_MINUTES * p_sel / (1.0 - float(floor.p_more_ot))
    print(f"\nSizing, so nobody mistakes this for a mean-effects result: at the selected "
          f"arm's\n  {p_sel:.4f} OT rate the expected extra length is {extra:.3f} min a "
          f"game, {extra / REGULATION_MINUTES:.2%} of regulation,\n  against "
          f"{OVERTIME_MINUTES * floor.p_any_ot / (1 - floor.p_more_ot):.3f} min at the "
          f"floor's rate. The head exists for the TAIL: overtime is where\n  40+ minute "
          f"games come from, and a best-ball weekly max is a function of the ceiling.")

    diag = diagnostics_frame(diagnostics)
    artifacts = {
        "metrics": (table, out_dir / "stan_game_length_metrics.csv"),
        "ppc": (checks, out_dir / "stan_game_length_ppc.csv"),
        "depth": (depth_table, out_dir / "stan_game_length_depth.csv"),
        "diagnostics": (diag, out_dir / "stan_game_length_diagnostics.csv"),
    }
    paths = {}
    for name, (df, dest) in artifacts.items():
        df.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(df):,} {name} rows → {dest}")
    print(f"\nSampler: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
          f"{diag['wall_clock_s'].sum():.1f} s total — the cheapest head in the project.")
    return paths


def _depth_verdict(table: pd.DataFrame, depth: BetaGeometricHead) -> None:
    """Say plainly which depth model validation prefers, and why the frailty ships anyway.

    `docs/simulations-plan.md` motivated the Beta frailty as "the natural fix for the plain
    geometric's one miss — it over-predicts 3OT+ by 3 games in 2,460". **That reason is
    wrong, and this is where it gets corrected**: those 3 games are a *validation*
    over-prediction, and the frailty makes it very slightly worse, because a frailty puts
    more mass in the tail rather than less. What the frailty actually fixes is the opposite
    miss on the fitting half, where the plain geometric under-predicts 3OT.
    """
    val = table[(table["split"] == "val") & (table["depth"] == 0)]
    if not len(val):
        return
    row = val.iloc[0]
    gap = float(row["beta_geometric"] - row["geometric"])
    tails = table[(table["split"] == "train") & (table["depth"] == 3)]
    print(f"  On validation the plain geometric is ahead by {-gap:.5f} nats per overtime "
          f"game over {int(row['n_ot_games'])} of them —\n  around one 2OT game's worth of "
          f"evidence, so the two are not distinguishable there.")
    if len(tails):
        t = tails.iloc[0]
        print(f"  The plan's stated reason for the frailty is BACKWARDS and is corrected "
              f"here: the 3 spurious\n  3OT+ games are a validation OVER-prediction and a "
              f"frailty makes that marginally worse. What it\n  fixes is the fitting "
              f"half's UNDER-prediction — {t['observed']:.0f} observed 3OT games against "
              f"{t['geometric']:.1f} geometric\n  and {t['beta_geometric']:.1f} "
              f"beta-geometric, on 13x the overtime games validation carries.")
    print(f"  It ships regardless, on a reason that is not fit: the beta-geometric NESTS "
          f"the geometric\n  (kappa -> infinity), it is fitted here at kappa "
          f"{depth.kappa:.1f}, and it is the only form of the depth\n  model that carries "
          f"a POSTERIOR — which is the entire point of moving this out of "
          f"`fit_ot_tail`.")


def _depth_table(train: pd.DataFrame, val: pd.DataFrame, floor: FloorGameLength,
                 depth: BetaGeometricHead) -> pd.DataFrame:
    """The two depth models side by side: counts by depth, and validation log-likelihood.

    The frailty's whole job is the tail, and the tail is three games in 2,460 — so it is shown
    as counts by depth rather than summarized into a number that rounds it away. The `nats`
    rows put the same comparison on the selection metric, which is how the choice between the
    two is settled rather than asserted.
    """
    depths = np.arange(1, MAX_DEPTH + 1)
    geometric = np.exp(floor.log_depth_pmf(depths))
    frailty = depth_pmf(depth, MAX_DEPTH)
    rows = []
    for split, frame in (("train", train), ("val", val)):
        k = frame["n_overtimes"].to_numpy(int)
        ot = k[k >= 1]
        n_ot = int(len(ot))
        for i, d in enumerate(depths):
            observed = int((k == d).sum()) if d < MAX_DEPTH else int((k >= d).sum())
            rows.append({"split": split, "depth": int(d),
                         "label": f"{d}OT" if d < MAX_DEPTH else f"{d}OT+",
                         "n_ot_games": n_ot, "observed": observed,
                         "geometric": float(n_ot * geometric[i]),
                         "beta_geometric": float(n_ot * frailty[i])})
        # `depth = 0` is the summary row: mean log-density per overtime game under each
        # model, which is the half of the arm ladder's metric the two differ in.
        index = np.clip(ot, 1, MAX_DEPTH) - 1
        rows.append({
            "split": split, "depth": 0, "label": "nats/OT game", "n_ot_games": n_ot,
            "observed": np.nan,
            "geometric": float(np.log(np.clip(geometric[index], EPS, None)).mean())
            if n_ot else np.nan,
            "beta_geometric": float(np.log(np.clip(frailty[index], EPS, None)).mean())
            if n_ot else np.nan})
    out = pd.DataFrame(rows)
    counts = out["depth"] > 0
    for name in ("geometric", "beta_geometric"):
        out[f"{name}_abs_error"] = np.where(
            counts, (out["observed"] - out[name]).abs(), np.nan)
    return out


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
