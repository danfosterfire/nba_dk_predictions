"""§16 — the returnee gap: the availability head's own lag-recovery ladder, and its gate.

`make availability-lag` → `outputs/predictions/availability_lag.csv`.

## The defect, and why it is this head's

`availability.build_design` drops a player-season on **`gp_share_lag1` alone**, so a player
who missed all of S-1 has no row no matter how complete S-2 is. He falls to
`sim/season.no_design_availability`, whose shipped `tenure_draft` key is the bare string
`"returning"` for every non-rookie — **one scalar per season for the whole population**.
`docs/rookie-rates-plan.md` §7f found it from the other end: the component ladder recovered
these rows' rate side, their predicted dk_pts *per game played* came out within a point of
realized, and their season totals were still 8.4x off because the games came from a level
fitted on fringe roster churn.

`docs/availability-window-plan.md` §16 is that finding turned into a round on the head that
owns it, and this module is the round.

## What varies, and what does not

The design change is in `availability.build_design(..., ladder=...)`, behind
`stan.availability.lag_ladder`, on `component_rates.LADDER_RUNGS`' vocabulary and reusing
`lag_recovery`'s helpers. This module fits the ladder's two constants, builds the widest
design it can, and scores four arms on the rows that design adds:

| arm | what it is | refits? |
|---|---|---|
| `plugin` | `no_design_availability`'s graded level at the fringe role dispersion — **the bar** | no |
| `shipped` | the persisted `train`-window Stan posterior, on the imputed row | no |
| `impute` | §16e arm 1 — the same specification as a point MLE, fitted on rung 0 alone | rung 0 |
| `staleness` | §16e arm 2 — the recovered rows enter the fit with `STALENESS_COLS` | rung 0 + recovered |

`shipped` is the literal reading §16e asks for first: *nothing refitted, one design edit*,
scored by coefficients fitted before the ladder existed — `lag_ladder.py`'s discipline, one
head over. `impute` is the same arm as a point MLE, and it exists because it is the
**control that prices the staleness column**: a refit that beats a plug-in has beaten
nothing in particular, and a refit that beats the same rows imputed has beaten the
imputation. It is also the arm the rolling harness can carry, since that refits per origin.

## The gate, stated in §16f before any result

- **Population.** The recoverable rows, reported `all` and **`draftable`** separately, with
  rung 0 carried beside them as the shipped bar — `lag_ladder.py`'s arrangement.
- **Primary.** Validation paired-bootstrap CRPS **in games** against `plugin`, interval
  entirely below zero on the draftable rows. Games, not season-total dk_pts: this is the
  head's own unit and the one every figure in `docs/availability-window-plan.md` is
  comparable to.
- **Confirmation.** The rolling-origin harness on the fitting half agreeing — same sign,
  interval below zero, a majority of origins won. §12e and §14f are why: two blocks on this
  head won a validation reading and did not survive it.
- **Non-regression, and it is hard.** Rung 0 comes back bit-identical, design and posterior.
  `tests/test_availability_lag.py` pins the design; `shipped` never refits, so its posterior
  cannot move; `staleness` does refit, and there the requirement is on the design and on a
  seeded spot-check.

## What this module does NOT do

It does not turn the key on. §16c: `build_design` is imported by `stan_minutes`,
`stan_composition`, `stan_games_played`, `model_cards`, `sim/season`, `season_terms` and
`final_evaluation`, and because the minutes allocation is zero-sum a recovered player takes
minutes from his teammates rather than appearing beside them. Measuring the ladder and
shipping it are separate decisions, and this module owns only the first.

Usage:
    python -m src.models.availability_lag
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize_scalar

from src.eda.preseason_value import attach_season_start_roster
from src.features.availability import load_artifacts
from src.models.availability import (AvailabilityLagLadder, FEATURE_COLS, STALENESS_COLS,
                                     build_design, crps, fit_dispersion, ladder_recovered,
                                     predictive_pmf, season_start_dates)
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.availability_no_prior import FITTED_ROLE_RHO, MIN_CELL, ROSTER_WINDOW_GAMES
from src.models.availability_window import (BOOTSTRAP_REPS, LIKELIHOOD_LOOKBACK,
                                            LIKELIHOOD_WINDOW, MIN_ROLE_ROWS, WINDOWS,
                                            MixtureFrailty, _origin_scores, assert_nests,
                                            paired_bootstrap, restrict_window)
from src.models.component_rates import LADDER_RUNGS
from src.models.held_out import selection_split
from src.models.posteriors import load_all, posteriors_dir
from src.models.season_terms import season_start_year
from src.sim.season import (allowed_seasons, no_design_availability, no_design_level_arm)

SEED = 42

#: The window whose posterior scores the `shipped` arm. `train` ends at 2021-22, so the
#: validation seasons the gate reads are outside its fit — the same discipline every
#: selection reading in the project follows, and the reason `shipped` can refit nothing.
SCORING_WINDOW = "train"

#: The arms, in report order. `plugin` is the bar every margin is quoted against.
ARMS: tuple[str, ...] = ("plugin", "shipped", "impute", "impute_raw", "staleness")
REFERENCE = "plugin"

#: The arms that are a fitted head rather than a plug-in, and so can be scored on rung 0.
HEAD_ARMS: tuple[str, ...] = ("shipped", "impute", "impute_raw", "staleness")

#: The arms scored on a design whose carried `gp_share` is **raw** rather than shrunk.
#: `impute_raw` is the same fitted model as `impute` reading a `k = 0` frame, so the pair
#: prices the one constant the design change contains — the shrink — separately from the
#: rows it recovers. Neither §16f's gate nor §16g's falsifiers cover that, and without it a
#: verdict on arm 1 could not say which half of arm 1 earned it.
RAW_ARMS: tuple[str, ...] = ("impute_raw",)

#: What the rolling harness carries. `shipped` is absent by construction: it is a persisted
#: posterior fitted once on the whole fitting half, and re-scoring it at each origin would
#: read coefficients that saw the origin's own season. `impute` is its refit-per-origin
#: twin and is what confirms it.
ROLLING_ARMS: tuple[str, ...] = ("plugin", "impute", "impute_raw", "staleness")

#: Which arm's rolling reading confirms which arm's validation reading. `shipped` has no
#: rolling row of its own **by construction** — it is one persisted posterior fitted on the
#: whole fitting half, and re-scoring it at each origin would read coefficients that saw the
#: origin's own season — so it is confirmed by `impute`, which is the same specification
#: refitted per origin. The two agree to 0.0034 CRPS on rung 0's 883 validation rows, which
#: is what makes the substitution a reading of the same arm rather than of a different one.
CONFIRMED_BY: dict[str, str] = {"shipped": "impute", "impute": "impute",
                                "impute_raw": "impute_raw", "staleness": "staleness"}

#: `veteran` is rung 0 — not a gated rung, but scored beside them so every figure has the
#: shipped bar next to it rather than a remembered one. `thin_prior` is structurally empty
#: on this head and is carried anyway, because "the rung exists and recovers nothing here"
#: is a statement worth being able to read off the artifact.
GROUPS: tuple[str, ...] = ("veteran",) + LADDER_RUNGS

#: The rungs that can recover a row on THIS head. A thin lag-1 still has a `gp_share_lag1`,
#: so `thin_prior` has always been inside the availability design — the asymmetry with
#: `component_rates`, whose boundary is a 200-minute test rather than a presence test.
RECOVERABLE: tuple[str, ...] = ("returnee_lag2", "returnee_thin", "no_usable_lag")

#: §15's two populations, unchanged. `all` is what §7, §12 and §14 scored; the draft pool is
#: the only one the head is ever applied to, and §16f reads the verdict there.
POPULATIONS: tuple[str, ...] = ("all", "draftable")

#: Bounds on the carried share's shrinkage, in pseudo-games. An 82-game season is the
#: natural scale, so a `k` of 82 is "trust the carried season half as much as a real one"
#: and the upper bound is five of them — far past any reading of the 0.746x level gap.
K_BOUNDS = (0.0, 400.0)

#: The dispersion the plug-in is scored at: the fringe role bucket, which is what
#: `sim/season.availability_rho_bin` hands a player with no prior MPG. Taken from
#: `availability_no_prior.FITTED_ROLE_RHO` rather than restated.
PLUGIN_RHO = FITTED_ROLE_RHO[0]

#: The rolling harness's first origin. The recoverable population needs three lag seasons
#: to exist and the fitting window opens at 2012-13, so an origin before this has almost no
#: recovered rows to score and would report a majority won on single digits.
FIRST_ORIGIN = 2015


# ── 1. The designs ────────────────────────────────────────────────────────────

def load_designs(cfg: dict, ladder: AvailabilityLagLadder | None
                 ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(the shipped design, the widest design)`, both carrying the draft-pool column.

    Built from the cached availability artifacts, which is `availability_design`'s own
    source. The two frames are returned together because every population statement in this
    module is a difference between them, and rebuilding one of them separately is how the
    bit-identity claim would quietly stop being tested.
    """
    seasons = list(cfg["data"]["seasons"])
    raw_dir = Path(cfg["data"]["raw_dir"])
    panel, frame = load_artifacts(Path(cfg["data"]["features_dir"]))
    frame = frame[frame["window"] == "full"].drop(columns=["window"])
    starts = season_start_dates(panel)

    plain = build_design(frame, seasons, raw_dir, starts)
    wide = build_design(frame, seasons, raw_dir, starts, ladder=ladder)
    return (attach_season_start_roster(plain, seasons, raw_dir, ROSTER_WINDOW_GAMES),
            attach_season_start_roster(wide, seasons, raw_dir, ROSTER_WINDOW_GAMES))


def head_frame(cfg: dict, design: pd.DataFrame) -> pd.DataFrame:
    """The design with the shipped preseason block attached, through the head's own path.

    `stan_availability.head_design` and never a second attachment: the coefficients the
    `shipped` arm scores with are coefficients on the columns `attach_preseason` built, and
    a recovered row has to arrive on the identical block or the posterior is being read
    against a design it never saw.
    """
    from src.models.stan_availability import head_design
    return head_design(cfg, design=design)


# ── 2. The ladder's two constants, fitted on the fitting half ─────────────────

def fit_constants(recovered: pd.DataFrame) -> dict[str, float]:
    """`{k_games, anchor_share}` — the shrink for a carried `gp_share`, and its target.

    ## The anchor is NOT the league mean, and that is the measurement

    `lag_recovery.recent_conversion`'s `league` argument is a league rate, because a
    conversion percentage carried forward is unbiased and only noisy. **A carried
    `gp_share` is neither.** §16b measured it: the recoverable rows realize 0.336 of a
    schedule and their carried lag-2 says 0.451, so the carry is biased **high by exactly
    the fact it erased** — a season the player missed, replaced by the one before it. The
    veteran population's own mean is 0.685, higher still, so shrinking toward it moves the
    level the wrong way and no `k` could rescue it.

    So the anchor is the **recoverable population's own pooled rate on the fitting half**,
    which is the quantity a returning player's level actually shrinks toward, and `k` says
    how fast. Both are fitted here, on the fitting half only, and persisted — the
    `components_preseason_shrinkage.csv` precedent, for its reason: a constant copied into a
    module is a constant that can silently stop matching the half it was fitted on.

    `k` is chosen by `fit_recent_conversion`'s own criterion — minimize the beta-binomial
    NLL of realized games at a dispersion **refitted for each candidate `k`**, so the
    constant is not selected against a dispersion that assumed a different one.
    """
    gp = recovered["gp_lag1"].to_numpy(dtype=float)
    trials = recovered["team_games_lag1"].to_numpy(dtype=float)
    y = np.rint(recovered["gp"].to_numpy(dtype=float)).astype(int)
    n = np.rint(recovered["team_games"].to_numpy(dtype=float)).astype(int)
    anchor = float(y.sum() / max(n.sum(), 1))

    live = np.isfinite(gp) & np.isfinite(trials) & (trials > 0)

    def nll_for(k: float) -> float:
        mu = np.clip((gp + k * anchor) / (trials + k), 1e-3, 1 - 1e-3)[live]
        rho = fit_dispersion(y[live], n[live], mu)
        return float(beta_binomial_nll(y[live], n[live], mu, rho))

    best = minimize_scalar(nll_for, bounds=K_BOUNDS, method="bounded")
    return {"k_games": float(best.x), "anchor_share": anchor}


def ladder_constants(path: Path | str) -> dict:
    """`availability_lag.csv` -> the constants `availability.lag_ladder` needs.

    The artifact is the interface, exactly as `lag_recovery.ladder_constants` is for the
    component ladder: `availability.lag_ladder` reads this and nothing else, so the builder
    cannot drift from the measurement that chose its shape.
    """
    table = pd.read_csv(path)
    hit = table[table["measurement"] == "constants"].set_index("metric")["value"]
    missing = [c for c in ("k_games", "anchor_share") if c not in hit.index]
    if missing:
        raise ValueError(
            f"{path} is missing the ladder constants {missing} — it predates §16's "
            f"ladder. Re-run `make availability-lag`.")
    return {"k_games": float(hit["k_games"]),
            "league_share": float(hit["anchor_share"])}


# ── 3. The bar: what one of these rows is given today ─────────────────────────

def plugin_rates(cfg: dict, plain: pd.DataFrame, wide: pd.DataFrame) -> pd.Series:
    """`no_design_availability`'s rate for every recovered row, keyed `(season, player_id)`.

    The shipped path called with the shipped arguments: the covered set is the **plain**
    design, which is exactly what these players fall outside of today, so the number here is
    the number the simulator gives them and not a reconstruction of it. Seasons whose
    pooling window has not yet filled are skipped rather than scored against a handful of
    players — `level_tables`' own rule, surfaced as a skip instead of an exception.
    """
    seasons = list(cfg["data"]["seasons"])
    features_dir = Path(cfg["data"]["features_dir"])
    arm = no_design_level_arm(cfg)
    allowed = allowed_seasons(plain)

    recovered = wide[ladder_recovered(wide)]
    out: dict[tuple[str, int], float] = {}
    for season, block in recovered.groupby("season", sort=True):
        ids = block["player_id"].to_numpy()
        try:
            level = no_design_availability(features_dir, str(season), plain, allowed,
                                           seasons, ids, arm)
        except ValueError:
            continue
        for pid, rate in level.items():
            out[(str(season), int(pid))] = float(rate)
    index = pd.MultiIndex.from_tuples(out.keys(), names=["season", "player_id"])
    return pd.Series(list(out.values()), index=index, name="plugin_rate")


# ── 4. The arms ───────────────────────────────────────────────────────────────

def arm_features(arm: str, features: list[str]) -> list[str]:
    """The covariate list one arm fits on. `staleness` is the only one that widens."""
    return list(features) + (list(STALENESS_COLS) if arm == "staleness" else [])


def fit_arms(train: pd.DataFrame, features: list[str], l2: float = 1.0,
             arms: tuple[str, ...] = ("impute", "staleness")) -> dict[str, object]:
    """`{arm: fitted MixtureFrailty}` — the shipped likelihood, two fitting populations.

    **`impute` fits rung 0 alone and `staleness` fits rung 0 plus the recovered rows.**
    That is the whole of §16e's fork: arm 1 leaves the fitting population exactly where the
    shipped head left it and imputes into columns it already has coefficients for; arm 2
    admits the rows and hands the head three columns that say the block is stale, so it can
    learn the shrink rather than have it asserted.
    """
    out: dict[str, object] = {}
    for arm in arms:
        rows = train if arm == "staleness" else train[~ladder_recovered(train)]
        cols = arm_features(arm, features)
        if rows[cols].isna().any().any():
            bad = [c for c in cols if rows[c].isna().any()]
            raise ValueError(f"{arm}: NaN in {bad} on the fitting window")
        model = MixtureFrailty(l2=l2, features=cols).fit(rows)
        assert_nests(model, rows)
        out[arm] = model
    return out


def arm_pmf(arm: str, frames: dict[str, pd.DataFrame], models: dict, shipped,
            plugin: np.ndarray, max_games: int) -> np.ndarray:
    """One arm's predictive over games played.

    Every arm returns a `(rows x games+1)` pmf on the same grid, which is what makes the
    paired bootstrap a comparison of forecasts rather than of implementations. The plug-in
    is a scalar mean at the **fringe** role dispersion, which is the predictive the
    simulator actually applies to one of these players — not a point estimate dressed up.

    `frames` carries the shrunk design and its raw-carry twin under the same row order, so
    `impute_raw` is the `impute` model reading a `k = 0` block rather than a second fit.
    """
    frame = frames["raw" if arm in RAW_ARMS else "wide"]
    n = frame["team_games"].to_numpy(dtype=float)
    if arm == "plugin":
        return predictive_pmf(n, np.clip(plugin, 1e-3, 1 - 1e-3), PLUGIN_RHO, max_games)
    if arm == "shipped":
        return shipped.predict_pmf(frame, max_games)
    return models[arm.removesuffix("_raw")].predict_pmf(frame, max_games)


# ── 5. The gate ───────────────────────────────────────────────────────────────

def _row(measurement: str, arm: str, group: str, population: str, metric: str,
         value: float, n: int, split: str = "validation") -> dict:
    return {"measurement": measurement, "split": split, "arm": arm, "group": group,
            "population": population, "metric": metric, "value": float(value),
            "n": int(n)}


def population_rows(plain: pd.DataFrame, wide: pd.DataFrame,
                    val: pd.DataFrame) -> list[dict]:
    """How many rows each rung is worth, in the design and on the validation split."""
    rows = [_row("population", "", "all", "all", "n_design_today", len(plain), len(plain)),
            _row("population", "", "all", "all", "n_design_ladder", len(wide), len(wide)),
            _row("population", "", "all", "all", "n_validation", len(val), len(val))]
    for group in GROUPS:
        in_design = int((wide["lag_rung"] == group).sum())
        sub = val[val["lag_rung"] == group]
        draftable = int((sub["on_season_start_roster"] > 0).sum())
        rows += [_row("population", "", group, "all", "n_design", in_design, len(wide)),
                 _row("population", "", group, "all", "n_validation", len(sub), len(sub)),
                 _row("population", "", group, "draftable", "n_validation", draftable,
                      draftable)]
    return rows


def carry_diagnostics(val: pd.DataFrame) -> list[dict]:
    """§16b's table, on the rows that ship: does the carry have the ordering, and the level?

    Reported and not gated. The claim the whole design brief rests on is that the carried
    season **orders** these players — correlation with realized share — while its level is
    biased high, and the two are separate quantities that a single CRPS number cannot
    separate. Rung 0's own lag-1 correlation sits beside them as the bar.
    """
    rows: list[dict] = []
    for group in ("veteran",) + RECOVERABLE + ("recovered",):
        for population in POPULATIONS:
            sub = (val[ladder_recovered(val)] if group == "recovered"
                   else val[val["lag_rung"] == group])
            if population == "draftable":
                sub = sub[sub["on_season_start_roster"] > 0]
            if len(sub) < 3:
                continue
            realized = (sub["gp"] / sub["team_games"]).to_numpy(dtype=float)
            carried = (sub["gp_lag1"] / sub["team_games_lag1"]).to_numpy(dtype=float)
            ok = np.isfinite(realized) & np.isfinite(carried)
            if ok.sum() < 3 or np.std(carried[ok]) == 0:
                continue
            rows += [
                _row("carry", "", group, population, "realized_share",
                     float(realized[ok].mean()), int(ok.sum())),
                _row("carry", "", group, population, "carried_share",
                     float(carried[ok].mean()), int(ok.sum())),
                _row("carry", "", group, population, "level_ratio",
                     float(realized[ok].mean() / carried[ok].mean()), int(ok.sum())),
                _row("carry", "", group, population, "corr_carried_realized",
                     float(np.corrcoef(carried[ok], realized[ok])[0, 1]), int(ok.sum())),
            ]
    return rows


def gate_table(frames: dict[str, pd.DataFrame], models: dict, shipped,
               plugin: np.ndarray, max_games: int,
               seed: int = SEED) -> tuple[list[dict], dict]:
    """Every arm's validation CRPS per group per population, and its margin over `plugin`.

    The margin is bootstrapped **within** each population rather than differenced across
    them, `population_ladder`'s rule: the recovered rows are a tenth of a percent of the
    frame, so a pooled margin would be a statement about rung 0 wearing a rung's name.
    """
    val = frames["wide"]
    rows: list[dict] = []
    per_row: dict[tuple[str, str, str], np.ndarray] = {}
    for group in ("veteran",) + RECOVERABLE + ("recovered",):
        mask = (ladder_recovered(val) if group == "recovered"
                else (val["lag_rung"] == group).to_numpy())
        for population in POPULATIONS:
            keep = mask & ((val["on_season_start_roster"].to_numpy(dtype=float) > 0)
                           if population == "draftable"
                           else np.ones(len(val), dtype=bool))
            cut = {k: f.loc[keep] for k, f in frames.items()}
            frame = cut["wide"]
            if len(frame) < 2:
                continue
            y = frame["gp"].to_numpy()
            scores = {}
            for arm in ARMS:
                if arm == "plugin" and group == "veteran":
                    continue           # rung 0 is never served by the plug-in
                pmf = arm_pmf(arm, cut, models, shipped, plugin[keep], max_games)
                s = crps(pmf, y)
                scores[arm] = s
                per_row[(arm, group, population)] = s
                mu = (pmf * np.arange(pmf.shape[1])[None, :]).sum(axis=1)
                rows += [
                    _row("gate", arm, group, population, "crps", s.mean(), len(frame)),
                    _row("gate", arm, group, population, "mae",
                         float(np.abs(mu - y).mean()), len(frame)),
                    _row("gate", arm, group, population, "bias",
                         float((mu - y).mean()), len(frame)),
                    _row("gate", arm, group, population, "mean_games",
                         float(mu.mean()), len(frame)),
                    _row("gate", arm, group, population, "realized_games",
                         float(y.mean()), len(frame)),
                ]
            for arm, reference in ((a, REFERENCE) for a in HEAD_ARMS if a in scores):
                if reference not in scores:
                    continue
                d = paired_bootstrap(scores[arm], scores[reference], seed=seed)
                rows += _margin("gate", arm, group, population, reference, d, len(frame))
            # §16g's second falsifier, stated as a column: if arm 2 beats the plug-in but
            # not arm 1, the staleness block is not what is doing the work — the rows are.
            if "staleness" in scores and "impute" in scores:
                d = paired_bootstrap(scores["staleness"], scores["impute"], seed=seed)
                rows += _margin("gate", "staleness", group, population, "impute", d,
                                len(frame))
    return rows, per_row


def _margin(measurement: str, arm: str, group: str, population: str, reference: str,
            d: tuple[float, float, float], n: int, split: str = "validation") -> list[dict]:
    """One paired-bootstrap margin as four rows.

    `d` is `availability_window.paired_bootstrap`'s `(delta, lo, hi)`, the tuple every other
    ladder on this head reads. `measurement` is a parameter rather than the string `gate`
    because the rolling half writes the identical four rows against the identical reference,
    and a margin filed under the wrong measurement is a margin `verdicts` cannot find.
    """
    delta, lo, hi = d
    tag = f"vs_{reference}"
    return [_row(measurement, arm, group, population, f"crps_{tag}", delta, n, split),
            _row(measurement, arm, group, population, f"crps_{tag}_lo", lo, n, split),
            _row(measurement, arm, group, population, f"crps_{tag}_hi", hi, n, split),
            _row(measurement, arm, group, population, f"beats_{reference}",
                 float(hi < 0.0), n, split)]


# ── 6. The confirmation §12e and §14f are the reason for ──────────────────────

def rolling(frames: dict[str, pd.DataFrame], plugin: pd.Series, features: list[str],
            max_games: int, l2: float = 1.0, seed: int = SEED,
            first_origin: int = FIRST_ORIGIN) -> list[dict]:
    """The rolling-origin harness on the fitting half — refit per origin, score forward.

    Validation carries 32 recoverable rows and 17 draftable ones, so it can measure a
    direction and not a size. This is the half that says whether the direction replicates,
    and on this head it is part of the bar rather than a follow-up: §12e and §14f record two
    covariate blocks that won a validation CRPS reading on these rows and shrank 5.8x and
    4.2x here, reopening their intervals across zero.

    The plug-in is **not** refitted per origin either — it is `no_design_availability`'s own
    expanding-window estimator, already pooled strictly before each target season, so the
    value a row carries here is the value it carried in the validation table.
    """
    train = frames["wide"]
    years = season_start_year(train)
    origins = [int(y) for y in np.unique(years) if y >= first_origin]
    parts: dict[str, dict[str, list[np.ndarray]]] = {}
    origin_means: list[dict] = []

    for origin in origins:
        at_origin = np.asarray(years == origin)
        fits = np.asarray((years < origin) & (years >= origin - LIKELIHOOD_LOOKBACK))
        score = train.loc[at_origin]
        fit_rows = train.loc[fits]
        if score.empty or len(fit_rows) < MIN_ROLE_ROWS:
            continue
        models = fit_arms(fit_rows, features, l2=l2)
        keys = list(zip(score["season"].astype(str), score["player_id"].astype(int)))
        rate = plugin.reindex(keys).to_numpy(dtype=float)
        keep = ladder_recovered(score) & np.isfinite(rate)
        if keep.sum() < 2:
            continue
        cut = {k: f.loc[at_origin].loc[keep] for k, f in frames.items()}
        frame = cut["wide"]
        y = frame["gp"].to_numpy()
        draftable = frame["on_season_start_roster"].to_numpy(dtype=float) > 0
        per_arm = {}
        for arm in ROLLING_ARMS:
            pmf = arm_pmf(arm, cut, models, None, rate[keep], max_games)
            per_arm[arm] = crps(pmf, y)
            slot = parts.setdefault(arm, {"crps": [], "draftable": []})
            slot["crps"].append(per_arm[arm])
            slot["draftable"].append(draftable.astype(float))
        for population in POPULATIONS:
            m = np.ones(len(frame), bool) if population == "all" else draftable
            if m.sum() < 1:
                continue
            for arm in per_arm:
                origin_means.append({"origin": origin, "population": population,
                                     "arm": arm, "crps": float(per_arm[arm][m].mean()),
                                     "n": int(m.sum())})
        print(f"  origin {origin}: {len(fit_rows):,} fit / {int(keep.sum()):,} recovered "
              f"scored ({int(draftable.sum()):,} draftable)")

    if not parts:
        return []
    pooled = {a: {k: np.concatenate(v) for k, v in d.items()} for a, d in parts.items()}
    means = pd.DataFrame(origin_means)
    rows: list[dict] = []
    for population in POPULATIONS:
        m = (np.ones(len(pooled[REFERENCE]["crps"]), bool) if population == "all"
             else pooled[REFERENCE]["draftable"] > 0)
        if m.sum() < 2:
            continue
        wide_means = means[means["population"] == population].pivot(
            index="origin", columns="arm", values="crps")
        for arm in ROLLING_ARMS:
            s = pooled[arm]["crps"][m]
            rows.append(_row("rolling", arm, "recovered", population, "crps",
                             float(s.mean()), int(m.sum()), split="rolling"))
        for arm, reference in (("impute", REFERENCE), ("impute_raw", REFERENCE),
                               ("staleness", REFERENCE), ("staleness", "impute")):
            d = paired_bootstrap(pooled[arm]["crps"][m], pooled[reference]["crps"][m],
                                 reps=BOOTSTRAP_REPS, seed=seed)
            rows += _margin("rolling", arm, "recovered", population, reference, d,
                            int(m.sum()), split="rolling")
            won = int((wide_means[arm] < wide_means[reference]).sum())
            rows += [_row("rolling", arm, "recovered", population,
                          f"origins_won_vs_{reference}", won, len(wide_means),
                          split="rolling"),
                     _row("rolling", arm, "recovered", population,
                          f"origins_vs_{reference}", len(wide_means), len(wide_means),
                          split="rolling")]
    return rows


# ── 7. The verdict, as code ───────────────────────────────────────────────────

def verdicts(table: pd.DataFrame) -> pd.DataFrame:
    """§16f's conjunction per arm — validation on the draftable rows, then the rolling half.

    Written as a function of the artifact rather than as a judgement made after reading it,
    which is the discipline every gate in `docs/availability-window-plan.md` follows.
    """
    def cell(measurement: str, arm: str, population: str, metric: str,
             group: str = "recovered") -> float:
        hit = table[(table["measurement"] == measurement) & (table["arm"] == arm)
                    & (table["group"] == group) & (table["population"] == population)
                    & (table["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    rows: list[dict] = []
    for group in ("recovered",) + RECOVERABLE:
        for arm in HEAD_ARMS:
            n = cell("gate", arm, "draftable", "crps", group)
            val_hi = cell("gate", arm, "draftable", "crps_vs_plugin_hi", group)
            val_delta = cell("gate", arm, "draftable", "crps_vs_plugin", group)
            twin = CONFIRMED_BY[arm]
            roll_hi = cell("rolling", twin, "draftable", "crps_vs_plugin_hi")
            roll_delta = cell("rolling", twin, "draftable", "crps_vs_plugin")
            won = cell("rolling", twin, "draftable", "origins_won_vs_plugin")
            origins = cell("rolling", twin, "draftable", "origins_vs_plugin")
            if not np.isfinite(n):
                continue
            primary = bool(val_hi < 0.0)
            # The confirmation is a conjunction of three things and not of the interval
            # alone: a sign flip with a wide interval is the failure §12e recorded, and "a
            # majority of origins won" is what separates a real margin from one season
            # carrying eight. It is pooled over the rungs, because the rolling half scores
            # 2-8 draftable rows at an origin and a per-rung majority would be noise.
            confirm = bool(roll_hi < 0.0 and np.sign(roll_delta) == np.sign(val_delta)
                           and origins > 0 and won > origins / 2.0)
            rows += [
                _row("verdict", arm, group, "draftable", "primary_passes",
                     float(primary), 0),
                _row("verdict", arm, group, "draftable", "rolling_confirms",
                     float(confirm), 0),
                _row("verdict", arm, group, "draftable", "admitted",
                     float(primary and confirm), 0),
            ]
    # §16g's second falsifier as a column rather than as a reading: "arm 2 beats the
    # plug-in but not arm 1 — then the staleness column is not what is doing the work, the
    # rows are, and the cheap form ships." Positive means arm 2 is the worse of the two.
    for population in POPULATIONS:
        d = cell("gate", "staleness", population, "crps_vs_impute")
        lo = cell("gate", "staleness", population, "crps_vs_impute_lo")
        hi = cell("gate", "staleness", population, "crps_vs_impute_hi")
        if not np.isfinite(d):
            continue
        rows += [_row("verdict", "staleness", "recovered", population,
                      "staleness_earns_its_place", float(hi < 0.0), 0),
                 _row("verdict", "staleness", "recovered", population,
                      "staleness_is_worse", float(lo > 0.0), 0)]
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", SEED))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    print("§16 — the returnee gap: the availability head's lag-recovery ladder")
    print("  the test split is LOCKED — seasons go through `held_out.selection_split`")

    # Pass 1: a RAW carry (`k = 0`), which is the frame the constants are fitted from. The
    # shrink cannot be fitted before the rows it shrinks exist, and building them twice is
    # cheaper than a constant estimated from whatever rows a caller happens to hold.
    raw = AvailabilityLagLadder(rungs=tuple(LADDER_RUNGS), k_games=0.0, league_share=0.0)
    plain, wide_raw = load_designs(cfg, raw)
    train_raw, _ = selection_split(wide_raw)
    fitting = train_raw[season_start_year(train_raw) >= WINDOWS[LIKELIHOOD_WINDOW]]
    constants = fit_constants(fitting[ladder_recovered(fitting)])
    print(f"  {len(plain):,} design rows today -> {len(wide_raw):,} with every rung "
          f"(+{len(wide_raw) - len(plain):,})")
    print(f"  shrink fitted on {int(ladder_recovered(fitting).sum()):,} fitting-half "
          f"recovered rows: k = {constants['k_games']:.4f} pseudo-games toward an anchor "
          f"of {constants['anchor_share']:.4f}")

    ladder = AvailabilityLagLadder(rungs=tuple(LADDER_RUNGS),
                                   k_games=constants["k_games"],
                                   league_share=constants["anchor_share"])
    plain, wide = load_designs(cfg, ladder)
    wide, wide_raw = head_frame(cfg, wide), head_frame(cfg, wide_raw)
    for col in STALENESS_COLS:
        if col not in wide.columns:
            raise ValueError(f"the preseason attachment dropped {col}")
    # The two designs differ in ONE column — the carried `gp_share_lag1` — so they carry the
    # same rows in the same order, and the raw-carry sensitivity is a second reading of the
    # same fit rather than a second population. Asserted rather than assumed, because a
    # silent misalignment would pair one player's block with another's outcome.
    key = ["season", "player_id"]
    if not wide[key].reset_index(drop=True).equals(wide_raw[key].reset_index(drop=True)):
        raise AssertionError("the shrunk and raw-carry designs are not row-aligned")
    wide_raw = wide_raw.reset_index(drop=True)
    wide = wide.reset_index(drop=True)

    plugin = plugin_rates(cfg, plain, wide)
    max_games = int(wide["team_games"].max())

    from src.models.stan_availability import head_features
    features = head_features(cfg.get("stan", {}).get("availability", {}).get("preseason"))

    train, val = selection_split(wide)
    # Rung 0 has no plug-in rate by construction — it is not a no-design row — so a NaN here
    # is only a defect on a recovered one. `gate_table` never reads the reference on rung 0,
    # and a recovered row the estimator cannot reach is dropped rather than scored at zero.
    keys = list(zip(val["season"].astype(str), val["player_id"].astype(int)))
    plugin_val = plugin.reindex(keys).to_numpy(dtype=float)
    drop = ladder_recovered(val) & ~np.isfinite(plugin_val)
    if drop.any():
        print(f"  {int(drop.sum())} recovered validation row(s) have no plug-in rate and "
              f"are dropped rather than scored against nothing")
    idx = val.index[~drop]
    frames = {"wide": wide.loc[idx].reset_index(drop=True),
              "raw": wide_raw.loc[idx].reset_index(drop=True)}
    plugin_val = np.nan_to_num(plugin_val[~drop], nan=0.0)
    val = frames["wide"]

    artifacts = load_all(posteriors_dir(cfg, SCORING_WINDOW), heads=["availability"])
    from src.models.stan_availability import PREDICTIVE_DRAWS, rehydrate_availability
    shipped = rehydrate_availability(artifacts["availability"], PREDICTIVE_DRAWS)
    print(f"  `shipped` scores with data/features/posteriors/{SCORING_WINDOW}/"
          f"availability.pkl — NOTHING is refitted for it")

    fit_mask = season_start_year(train) >= WINDOWS[LIKELIHOOD_WINDOW]
    cut = train.loc[fit_mask]
    models = fit_arms(cut, features, l2=l2)
    print(f"  MLE arms fitted on {len(cut):,} rows from "
          f"{cut['season'].min()} ({int((~ladder_recovered(cut)).sum()):,} at rung 0, "
          f"{int(ladder_recovered(cut).sum()):,} recovered)")

    rows = population_rows(plain, wide, val) + carry_diagnostics(val)
    rows += [_row("constants", "", "recovered", "all", k, v,
                  int(ladder_recovered(fitting).sum()))
             for k, v in constants.items()]
    gate, per_row = gate_table(frames, models, shipped, plugin_val, max_games, seed)
    rows += gate

    # The design's non-regression claim, checked at the SCORE rather than at the frame:
    # the shrunk and raw-carry designs differ in one column on the recovered rows only, so
    # a fitted head must score rung 0 identically through both. If this ever fires, the
    # ladder has reached a row it was not supposed to be able to reach.
    for population in POPULATIONS:
        a = per_row.get(("impute", "veteran", population))
        b = per_row.get(("impute_raw", "veteran", population))
        if a is not None and b is not None and not np.array_equal(a, b):
            raise AssertionError(
                f"the raw-carry design moved rung 0 on the {population} population — the "
                f"shrink reached a row the shipped design already carried")

    print("\nThe rolling-origin harness on the fitting half")
    train_idx = train.index
    rows += rolling({"wide": wide.loc[train_idx].reset_index(drop=True),
                     "raw": wide_raw.loc[train_idx].reset_index(drop=True)},
                    plugin, features, max_games, l2=l2, seed=seed)

    table = pd.DataFrame(rows)
    table = pd.concat([table, verdicts(table)], ignore_index=True)
    _report(table)

    dest = out_dir / "availability_lag.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} §16 gate rows → {dest}")
    return dest


def _report(t: pd.DataFrame) -> None:
    def cell(measurement: str, arm: str, group: str, population: str,
             metric: str) -> float:
        hit = t[(t["measurement"] == measurement) & (t["arm"] == arm)
                & (t["group"] == group) & (t["population"] == population)
                & (t["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    print("\nThe carry — does it have the ordering, and does it have the level?")
    print(f"  {'group':<16}{'population':<11}{'realized':>10}{'carried':>10}"
          f"{'ratio':>8}{'corr':>8}")
    for group in ("veteran",) + RECOVERABLE + ("recovered",):
        for population in POPULATIONS:
            r = cell("carry", "", group, population, "realized_share")
            if not np.isfinite(r):
                continue
            print(f"  {group:<16}{population:<11}{r:>10.4f}"
                  f"{cell('carry', '', group, population, 'carried_share'):>10.4f}"
                  f"{cell('carry', '', group, population, 'level_ratio'):>8.3f}"
                  f"{cell('carry', '', group, population, 'corr_carried_realized'):>8.3f}")

    for population in POPULATIONS:
        print(f"\nValidation CRPS in games — {population}")
        print(f"  {'group':<16}{'arm':<11}{'n':>5}{'CRPS':>9}{'MAE':>8}{'pred gp':>9}"
              f"{'real gp':>9}{'vs plugin [95%]':>28}")
        for group in ("veteran",) + RECOVERABLE + ("recovered",):
            for arm in ARMS:
                c = cell("gate", arm, group, population, "crps")
                if not np.isfinite(c):
                    continue
                d = cell("gate", arm, group, population, "crps_vs_plugin")
                margin = ("" if not np.isfinite(d) else
                          f"{d:+9.4f} [{cell('gate', arm, group, population, 'crps_vs_plugin_lo'):+.4f}, "
                          f"{cell('gate', arm, group, population, 'crps_vs_plugin_hi'):+.4f}]")
                n = t[(t['measurement'] == 'gate') & (t['arm'] == arm)
                      & (t['group'] == group) & (t['population'] == population)]["n"]
                print(f"  {group:<16}{arm:<11}{int(n.max()) if len(n) else 0:>5}{c:>9.4f}"
                      f"{cell('gate', arm, group, population, 'mae'):>8.3f}"
                      f"{cell('gate', arm, group, population, 'mean_games'):>9.3f}"
                      f"{cell('gate', arm, group, population, 'realized_games'):>9.3f}"
                      f"{margin:>28}")

    print("\nThe rolling half — recovered rows, refit per origin")
    for population in POPULATIONS:
        for arm in ("impute", "impute_raw", "staleness"):
            d = cell("rolling", arm, "recovered", population, "crps_vs_plugin")
            if not np.isfinite(d):
                continue
            print(f"  {population:<10}{arm:<11}{d:+9.4f} "
                  f"[{cell('rolling', arm, 'recovered', population, 'crps_vs_plugin_lo'):+.4f}, "
                  f"{cell('rolling', arm, 'recovered', population, 'crps_vs_plugin_hi'):+.4f}]"
                  f"   {int(cell('rolling', arm, 'recovered', population, 'origins_won_vs_plugin'))}"
                  f" of {int(cell('rolling', arm, 'recovered', population, 'origins_vs_plugin'))}"
                  f" origins")

    print("\nThe verdict — §16f's conjunction, stated before the result, draftable rows")
    for group in ("recovered",) + RECOVERABLE:
        for arm in HEAD_ARMS:
            admitted = cell("verdict", arm, group, "draftable", "admitted")
            if not np.isfinite(admitted):
                continue
            print(f"  {group:<16}{arm:<11}"
                  f"{'ADMITTED' if admitted == 1.0 else 'rejected':<10}"
                  f" primary "
                  f"{'pass' if cell('verdict', arm, group, 'draftable', 'primary_passes') == 1.0 else 'FAIL'}"
                  f", rolling "
                  f"{'confirms' if cell('verdict', arm, group, 'draftable', 'rolling_confirms') == 1.0 else 'does not confirm'}")
    for population in POPULATIONS:
        earns = cell("verdict", "staleness", "recovered", population,
                     "staleness_earns_its_place")
        if not np.isfinite(earns):
            continue
        print(f"  §16g falsifier 2, {population}: the staleness column "
              f"{'EARNS its place over arm 1' if earns == 1.0 else 'does not beat arm 1'}"
              f" ({cell('gate', 'staleness', 'recovered', population, 'crps_vs_impute'):+.4f} "
              f"[{cell('gate', 'staleness', 'recovered', population, 'crps_vs_impute_lo'):+.4f}, "
              f"{cell('gate', 'staleness', 'recovered', population, 'crps_vs_impute_hi'):+.4f}])")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
