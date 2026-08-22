"""How much own-rate signal is recoverable for the players the veteran design drops.

`component_rates.build_design` asks for one thing — a lag-1 season of at least
`MIN_PRIOR_MINUTES` — and everything that fails it is dropped before any modelling
decision is taken. That single test bundles three populations with very different
information, and this module measures them apart:

- **returnee** — lag-1 is missing entirely because he missed the season, but lag-2 is a
  full one. `eda/availability.with_lags` pairs on season *index* deliberately ("stops a
  player who missed a year from silently pairing across the gap"), so the lag-2 row exists
  in the data and is simply never built at `max_lag=1`.
- **thin prior** — lag-1 exists and is under the threshold, so the rate is noisy rather
  than absent.
- **true rookie** — no NBA season at all, and no own-rate feature is constructible at any
  lag. This is the only group for which "the features do not exist" is literally true.

The measurement decides where the boundary of the veteran design belongs, which is a
population question rather than a fitting one — so **nothing here fits a head**. Every
estimator is `component_rates.carry_forward`'s own functional form (prior per-36 x realized
minutes / 36), scored as R2 against the realized count, which puts every number on the same
scale as the shipped no-fit floor's published 0.81-0.95 band.

**Scope: the nine rate targets, not all eleven heads.** The four beta-binomial heads are a
different metric (NLL against a fitted dispersion, not R2) and they inherit whatever
population decision this settles rather than informing it; their per-head gate is Session 3's
job. `docs/rookie-rates-plan.md` §7b says so where it quotes these figures.

Two things are fitted for the *arms*, both on the fitting half alone per §3 constraint 6:
the thin-prior shrinkage `k`, and the population mean it shrinks toward. The returnee arm
fits nothing — raw lag-2, carried forward — so its figure is quoted pooled over
train+validation, where the n is large enough to read.

**The module is also where the shipped ladder's constants live** (`ladder_constants_rows`,
`ladder_constants`), because `component_rates.build_design` must not fit anything at build
time and a constant pinned in a module is a constant that can stop matching the half it was
fitted on — the `components_preseason_shrinkage.csv` precedent. That includes the four
conversion heads' `(k pseudo-attempts, league mean)`, which is a *feature* block rather
than a score: the scope note above still holds, and no conversion head is scored here.

Usage:
    python -m src.models.lag_recovery
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from scipy.optimize import minimize_scalar

from src.eda.availability import with_lags
from src.models.availability import _neg_loglik as beta_binomial_nll
from src.models.availability import fit_dispersion
from src.models.component_rates import (CONVERSION_HEADS, MIN_PRIOR_MINUTES, PER36,
                                        load_ages, rate_columns, season_totals,
                                        volume_columns)
from src.models.held_out import selection_split

# The shrinkage grid for the thin-prior arm, in pseudo-minutes. `0` is the raw lag-1 and
# a large `k` is the population mean, so the grid brackets the decision rather than
# assuming shrinkage helps.
THIN_K_GRID = (0.0, 25.0, 50.0, 100.0, 150.0, 200.0, 300.0, 500.0, 800.0, 1500.0)

GROUPS = ("veteran", "thin_prior", "returnee_lag2", "returnee_thin", "true_rookie",
          "no_usable_lag")

# Every group except `true_rookie` has SOME played NBA season inside the window, which is
# the whole of the revised boundary: the rookie head serves the players for whom no
# own-rate feature is constructible at any lag, and nobody else.
HAS_HISTORY = ("thin_prior", "returnee_lag2", "returnee_thin", "no_usable_lag")


# ── 1. The population, split by WHY the design has no row ────────────────────

def population_frame(targets: pd.DataFrame, seasons: list[str],
                     raw_dir: str) -> pd.DataFrame:
    """Every player-season with lags 1-3 attached and a `group` label.

    Deliberately **not** `build_design`: this is the frame that design is carved out of,
    so the rows it drops are still here to be counted and scored.
    """
    s = season_totals(targets)
    # `volume_columns` rather than `rate_columns`: the conversion block's shrinkage is
    # fitted from lagged MAKES over lagged ATTEMPTS, and a make (`fg2m`, `ftm`) is not a
    # rate target. The arms below are unaffected — these are extra columns, not extra rows.
    lag_cols = list(dict.fromkeys([f"{c}_p36" for c in rate_columns()]
                                  + volume_columns()
                                  + ["mpg", "total_minutes", "gp"]))
    d = with_lags(s, seasons, lag_cols, max_lag=3)
    ages = load_ages(seasons, raw_dir)
    d = (d.merge(ages, on=["season", "player_id"], how="left") if not ages.empty
         else d.assign(age=np.nan))
    first = s.groupby("player_id")["season"].min().rename("first_season")
    d = d.merge(first, on="player_id", how="left")
    return d.assign(group=classify(d)).reset_index(drop=True)


def classify(d: pd.DataFrame) -> pd.Series:
    """The six-way split, in priority order — first match wins.

    `true_rookie` is tested first and by *first played season* rather than by absent lags,
    because a player whose only prior seasons fall outside the configured window would
    otherwise be mislabelled as a rookie and hand the rookie head a row it cannot serve.
    """
    m1, m2, m3 = (d["total_minutes_lag1"], d["total_minutes_lag2"],
                  d["total_minutes_lag3"])
    return pd.Series(
        np.select(
            [d["season"] == d["first_season"],
             m1.notna() & (m1 >= MIN_PRIOR_MINUTES),
             m1.notna(),
             m2.notna() & (m2 >= MIN_PRIOR_MINUTES),
             m2.notna(),
             m3.notna()],
            ["true_rookie", "veteran", "thin_prior", "returnee_lag2", "returnee_thin",
             "no_usable_lag"],
            default="no_usable_lag"),
        index=d.index)


def scorable_now(d: pd.DataFrame) -> pd.Series:
    """Whether today's shipped design would carry the row — the baseline to widen from."""
    return (d["group"] == "veteran") & d["age"].notna() & d["mpg_lag1"].notna()


# ── 2. The estimators, all one functional form ───────────────────────────────

def carry_count(frame: pd.DataFrame, rate: np.ndarray) -> np.ndarray:
    """`component_rates.carry_forward`'s prediction, given any prior rate."""
    return np.clip(np.asarray(rate, dtype=float)
                   * frame["total_minutes"].to_numpy(dtype=float) / PER36, 0.0, None)


def shrunk_rate(frame: pd.DataFrame, comp: str, k: float, mu: float,
                lag: int = 1) -> np.ndarray:
    """A thin lag pulled toward `mu` by its own reliability weight `m / (m + k)`.

    The same device `minutes_preseason.reliability_weight` uses and P4(b) selected for the
    no-prior population, pointed at prior-season minutes instead of preseason minutes.
    """
    m = frame[f"total_minutes_lag{lag}"].to_numpy(dtype=float)
    r = frame[f"{comp}_p36_lag{lag}"].to_numpy(dtype=float)
    w = np.where(np.isfinite(m), m / (m + k), 0.0) if k > 0 else np.ones(len(frame))
    return np.where(np.isfinite(r), w * r + (1.0 - w) * mu, mu)


def recent_lag(frame: pd.DataFrame, max_lag: int = 3) -> np.ndarray:
    """Which lag each row's most recent played season sits at — 0 where there is none.

    The revised design does not care *why* lag-1 is unusable, only what the nearest usable
    season is. A returnee's is at 2, a thin-prior's is at 1 (thin but present), and a
    player away two years is at 3 — one rung serves all three.
    """
    out = np.zeros(len(frame), dtype=int)
    for lag in range(max_lag, 0, -1):
        m = frame[f"total_minutes_lag{lag}"].to_numpy(dtype=float)
        out = np.where(np.isfinite(m) & (m > 0), lag, out)
    return out


def recent_shrunk_rate(frame: pd.DataFrame, comp: str, k: float,
                       mu: float) -> np.ndarray:
    """The proposed rung: the most recent usable rate, shrunk by its own reliability.

    One estimator for the whole has-history population, so the design's boundary becomes
    *is there any prior season at all* rather than *is the immediately-prior one big
    enough*. `k` is in prior-season minutes and is fitted on the fitting half.
    """
    lag = recent_lag(frame)
    rate = np.full(len(frame), np.nan)
    minutes = np.zeros(len(frame))
    for L in (1, 2, 3):
        hit = lag == L
        if hit.any():
            rate[hit] = frame[f"{comp}_p36_lag{L}"].to_numpy(float)[hit]
            minutes[hit] = frame[f"total_minutes_lag{L}"].to_numpy(float)[hit]
    w = np.where(minutes > 0, minutes / (minutes + k), 0.0) if k > 0 else np.ones(len(frame))
    return np.where(np.isfinite(rate), w * rate + (1.0 - w) * mu, mu)


def r2(y: np.ndarray, yhat: np.ndarray) -> float:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    ok = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[ok], yhat[ok]
    if len(y) < 2 or not np.any(y != y[0]):
        return float("nan")
    return float(1.0 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def fit_recent_k(train: pd.DataFrame, mu: dict[str, float]) -> dict[str, float]:
    """Shrinkage for the unified rung, fitted on the fitting half's has-history rows.

    Fitted on the population it serves rather than on veterans, because the whole point of
    the rung is that these rows carry less information than a qualified lag-1 and the
    weight has to know it.
    """
    sub_ = train[train["group"].isin(HAS_HISTORY)]
    out = {}
    for comp in rate_columns():
        scores = [(r2(sub_[comp],
                      carry_count(sub_, recent_shrunk_rate(sub_, comp, k, mu[comp]))), k)
                  for k in THIN_K_GRID]
        scores = [(sc, k) for sc, k in scores if np.isfinite(sc)]
        out[comp] = float(max(scores)[1]) if scores else float("nan")
    return out


def fit_thin_k(train: pd.DataFrame, mu: dict[str, float]) -> dict[str, float]:
    """Per-head shrinkage, chosen on the FITTING half's thin-prior rows only.

    Fitted per head rather than pooled because the heads differ by an order of magnitude in
    how much a sub-threshold sample tells you — a hundred minutes pins a rebound rate far
    better than a block rate, which is the same ordering P4(b) found for preseason.
    """
    thin = train[train["group"] == "thin_prior"]
    out = {}
    for comp in rate_columns():
        scores = [(r2(thin[comp], carry_count(thin, shrunk_rate(thin, comp, k, mu[comp]))),
                   k) for k in THIN_K_GRID]
        scores = [(s, k) for s, k in scores if np.isfinite(s)]
        out[comp] = float(max(scores)[1]) if scores else float("nan")
    return out


def recent_conversion(frame: pd.DataFrame, made: str, attempted: str, k: float,
                     league: float) -> np.ndarray:
    """The nearest usable season's conversion percentage, empirical-Bayes shrunk.

    `recent_shrunk_rate`'s twin for a proportion, and deliberately a *different* device:
    a percentage's reliability is carried by its **attempts**, not by the minutes that
    produced them, and `component_rates.carry_forward_conversion` already settled the form
    on the record — `(made + k*league) / (attempts + k)` with `k` in pseudo-attempts,
    because a player who went 0-for-3 has a prior of exactly 0.000 and carrying that
    forward raw is not a weak estimator but a broken one.

    Fitted here rather than left to the head because the four conversion features are part
    of the same lag block the rates are: §7b scored the nine rate targets and left the
    conversions to inherit the population decision, which they can only do if the block
    they inherit is built.
    """
    lag = recent_lag(frame)
    m = np.full(len(frame), np.nan)
    a = np.full(len(frame), np.nan)
    for L in (1, 2, 3):
        hit = lag == L
        if hit.any():
            m[hit] = frame[f"{made}_lag{L}"].to_numpy(float)[hit]
            a[hit] = frame[f"{attempted}_lag{L}"].to_numpy(float)[hit]
    ok = np.isfinite(m) & np.isfinite(a)
    p = np.where(ok, (np.nan_to_num(m) + k * league) / (np.nan_to_num(a) + k), league)
    return np.clip(p, 1e-3, 1 - 1e-3)


def fit_recent_conversion(train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """`(k pseudo-attempts, league mean)` per conversion head, on the FITTING half.

    The league mean comes from the veteran rows — the fitting population, unchanged — and
    `k` is chosen on the has-history rows, which is the population the constant is for.
    `carry_forward_conversion`'s own criterion is reused verbatim: minimize the
    beta-binomial NLL at a dispersion refitted for each candidate `k`, so the constant is
    not chosen against a dispersion that assumed a different one.
    """
    vet = train[train["group"] == "veteran"]
    sub = train[train["group"].isin(HAS_HISTORY)]
    out = {}
    for made, attempted in CONVERSION_HEADS:
        mv = vet[made].to_numpy(float)
        av = vet[attempted].to_numpy(float)
        ok = np.isfinite(mv) & np.isfinite(av) & (av > 0)
        league = float(mv[ok].sum() / av[ok].sum()) if ok.any() else 0.5
        y = np.rint(sub[made].to_numpy(float)).astype(float)
        n = np.rint(sub[attempted].to_numpy(float)).astype(float)
        live = np.isfinite(y) & np.isfinite(n) & (n > 0)
        y, n = np.minimum(y, n)[live].astype(int), n[live].astype(int)
        if not len(y):
            out[made] = (float("nan"), league)
            continue

        def nll_for(k: float, made=made, attempted=attempted, league=league,
                    y=y, n=n, live=live) -> float:
            p = recent_conversion(sub, made, attempted, float(k), league)[live]
            rho = fit_dispersion(y, n, p)
            return float(beta_binomial_nll(y, n, p, rho))

        best = minimize_scalar(nll_for, bounds=(1.0, 2000.0), method="bounded")
        out[made] = (float(best.x), league)
    return out


# ── 3. The measurements ──────────────────────────────────────────────────────

def _row(measurement: str, split: str, group: str, head: str, estimator: str,
         metric: str, value: float, n: int, season: str = "") -> dict:
    return {"measurement": measurement, "split": split, "season": season, "group": group,
            "head": head, "estimator": estimator, "metric": metric,
            "value": float(value), "n": int(n)}


def population_census(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """How many played player-seasons each group holds, per split.

    Counted on the split rather than on every season on disk: the test seasons are not
    ours to count, and a census that included them would disagree with every `n` reported
    beside it.
    """
    return [_row("population", split, group, "", "", "n_played",
                 int((frame["group"] == group).sum()), len(frame))
            for split, frame in frames.items() for group in GROUPS]


def board_census(pool: pd.DataFrame, frame: pd.DataFrame,
                 seasons: list[str]) -> list[dict]:
    """Of the players on a real draft board, how many does each group account for?

    The board rather than the roster, because "who could we have drafted and did not" is
    the question the floor priced, and `draft_pool.parquet` is the same file the drafting
    layer builds its board from.
    """
    rows = []
    for season in seasons:
        board = pool[pool["season"] == season]
        sub = frame[frame["season"] == season].copy()
        sub["served"] = scorable_now(sub)
        sub = sub.drop_duplicates("player_id").set_index("player_id")
        group = board["player_id"].map(sub["group"]).fillna("no_usable_lag").to_numpy()
        served = board["player_id"].map(sub["served"]).fillna(False).astype(bool)
        priced = board["adp"].notna().to_numpy()
        rows.append(_row("board_census", "board", "all", "", "", "n_board",
                         len(board), len(board), season))
        rows.append(_row("board_census", "board", "all", "", "", "n_priced",
                         int(priced.sum()), len(board), season))
        rows.append(_row("board_census", "board", "served_today", "", "", "n",
                         int(served.sum()), len(board), season))
        for name in GROUPS:
            hit = (group == name) & ~served.to_numpy()
            rows.append(_row("board_census", "board", name, "", "", "n_unserved",
                             int(hit.sum()), len(board), season))
            rows.append(_row("board_census", "board", name, "", "", "n_unserved_priced",
                             int((hit & priced).sum()), len(board), season))
    return rows


def carry_power(frames: dict[str, pd.DataFrame], mu: dict[str, float],
                thin_k: dict[str, float],
                recent_k: dict[str, float] | None = None) -> list[dict]:
    """Every group's best carry-forward, per head, per split.

    The four arms are the four boundaries of the design: what the veteran rows get today,
    what a returnee's untouched lag-2 gets, and what a thin lag-1 gets raw versus shrunk.
    """
    arms = [("veteran", "lag1", lambda f, c: f[f"{c}_p36_lag1"].to_numpy(float)),
            ("returnee_lag2", "lag2", lambda f, c: f[f"{c}_p36_lag2"].to_numpy(float)),
            ("thin_prior", "lag1_raw", lambda f, c: f[f"{c}_p36_lag1"].to_numpy(float)),
            ("thin_prior", "lag1_shrunk",
             lambda f, c: shrunk_rate(f, c, thin_k[c], mu[c]))]
    rows = []
    # The unified rung, scored on each has-history group separately AND pooled, because
    # "one estimator serves them all" is a claim about the worst group as much as the mean.
    if recent_k is not None:
        for split, frame in frames.items():
            pool = frame[frame["group"].isin(HAS_HISTORY)]
            for group, sub_ in [("has_history", pool)] + [
                    (g, frame[frame["group"] == g]) for g in HAS_HISTORY]:
                for comp in rate_columns():
                    rows.append(_row(
                        "carry_power", split, group, comp, "recent_shrunk", "r2",
                        r2(sub_[comp],
                           carry_count(sub_, recent_shrunk_rate(sub_, comp,
                                                                recent_k[comp],
                                                                mu[comp]))),
                        len(sub_)))
        for comp, k in recent_k.items():
            rows.append(_row("carry_power", "train", "has_history", comp,
                             "recent_shrunk", "k_minutes", k, 0))
    for split, frame in frames.items():
        for group, estimator, rate_of in arms:
            sub = frame[frame["group"] == group]
            for comp in rate_columns():
                rows.append(_row("carry_power", split, group, comp, estimator, "r2",
                                 r2(sub[comp], carry_count(sub, rate_of(sub, comp))),
                                 len(sub)))
    for comp, k in thin_k.items():
        rows.append(_row("carry_power", "train", "thin_prior", comp, "lag1_shrunk",
                         "k_minutes", k, 0))
    return rows


def ladder_constants_rows(mu: dict[str, float],
                          conv: dict[str, tuple[float, float]]) -> list[dict]:
    """The fitted constants `component_rates`' ladder consumes, persisted beside the arms.

    `k_minutes` is already written by `carry_power`; what was missing is everything else
    the imputation needs to be reproducible from the artifact alone — the population mean
    each rate is shrunk toward, and the conversion block's `(k, league)` pair.

    Read rather than pinned, for the reason `stan_components.shrinkage_constants` reads
    `components_preseason_shrinkage.csv`: these are fitted quantities, estimated on the
    fitting half per §3 constraint 6, and a constant copied into a module is a constant
    that can silently stop matching the half it was fitted on.
    """
    rows = [_row("carry_power", "train", "has_history", comp, "recent_shrunk",
                 "mu_p36", value, 0) for comp, value in mu.items()]
    for made, (k, league) in conv.items():
        rows.append(_row("carry_power", "train", "has_history", made, "recent_shrunk",
                         "k_attempts", k, 0))
        rows.append(_row("carry_power", "train", "has_history", made, "recent_shrunk",
                         "league_pct", league, 0))
    return rows


def ladder_constants(path: Path | str) -> dict:
    """`lag_recovery.csv` -> the three constant blocks the ladder builder needs.

    The artifact is the interface: `component_rates.lag_ladder` reads this and nothing
    else, so the builder cannot drift from the measurement that chose its shape.
    """
    table = pd.read_csv(path)
    hit = table[(table["measurement"] == "carry_power") & (table["split"] == "train")
                & (table["group"] == "has_history")
                & (table["estimator"] == "recent_shrunk")]

    def block(metric: str) -> dict[str, float]:
        sub = hit[hit["metric"] == metric]
        return {str(h): float(v) for h, v in zip(sub["head"], sub["value"])}

    shrinkage, means = block("k_minutes"), block("mu_p36")
    k_att, league = block("k_attempts"), block("league_pct")
    missing = ([f"{c}/k_minutes" for c in rate_columns() if c not in shrinkage]
               + [f"{c}/mu_p36" for c in rate_columns() if c not in means]
               + [f"{m}/k_attempts" for m, _ in CONVERSION_HEADS if m not in k_att]
               + [f"{m}/league_pct" for m, _ in CONVERSION_HEADS if m not in league])
    if missing:
        raise ValueError(
            f"{path} is missing the ladder constants {missing} — it predates the "
            f"lag-recovery ladder. Re-run `make lag-recovery`.")
    return {"shrinkage": shrinkage, "means": means,
            "conversion": {m: (k_att[m], league[m]) for m, _ in CONVERSION_HEADS}}


def preseason_head_to_head(frames: dict[str, pd.DataFrame], features_dir: Path,
                           mu: dict[str, float], k_pre: float = 160.0) -> list[dict]:
    """Returnees' lag-2 against the estimator the rookie head would give them instead.

    P4(b) (`make rookie-priors`) selected the volume-shrunk preseason per-36 at `k = 160`
    for rates on the no-prior population. If the rookie head is to serve returnees, that is
    what it serves them with — so this is the comparison that decides whether a returnee
    belongs in that head at all. Coverage is reported beside it, because an estimator that
    is missing is not an estimator.
    """
    pre = pd.read_parquet(features_dir / "preseason.parquet")
    cols = ["season", "player_id", "min_pre"] + [c for c in pre.columns
                                                 if c.startswith("pre_per36_")]
    covered = set(pre["season"].unique())
    rows = []
    for split, frame in frames.items():
        for group in ("returnee_lag2", "true_rookie"):
            sub = frame[(frame["group"] == group)
                        & frame["season"].isin(covered)].merge(pre[cols],
                                                               on=["season", "player_id"],
                                                               how="left")
            has = sub["min_pre"].notna() & (sub["min_pre"] > 0)
            rows.append(_row("preseason", split, group, "", "", "coverage",
                             float(has.mean()) if len(sub) else float("nan"), len(sub)))
            hit = sub[has]
            if not len(hit):
                continue
            w = hit["min_pre"].to_numpy(float) / (hit["min_pre"].to_numpy(float) + k_pre)
            for comp in rate_columns():
                col = f"pre_per36_{comp}"
                if col not in hit.columns:
                    continue
                shrunk = w * hit[col].to_numpy(float) + (1.0 - w) * mu[comp]
                rows.append(_row("preseason", split, group, comp, "preseason_shrunk",
                                 "r2", r2(hit[comp], carry_count(hit, shrunk)), len(hit)))
                if group == "returnee_lag2":
                    rows.append(_row("preseason", split, group, comp, "lag2", "r2",
                                     r2(hit[comp],
                                        carry_count(hit,
                                                    hit[f"{comp}_p36_lag2"].to_numpy(float))),
                                     len(hit)))
    return rows


# ── 4. The entry point ───────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    frame = population_frame(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    played = frame[frame["total_minutes"] > 0]
    train, val = selection_split(played, 2)
    pooled = pd.concat([train, val], ignore_index=True)
    print("Lag recovery — where the veteran design's boundary belongs")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")
    print(f"  {len(train):,} fit / {len(val):,} select "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")
    for name in GROUPS:
        n = int((pooled["group"] == name).sum())
        print(f"    {name:<16} {n:>7,} played player-seasons (train+validation)")

    mu = {c: float(train.loc[train["group"] == "veteran", f"{c}_p36"].mean())
          for c in rate_columns()}
    thin_k = fit_thin_k(train, mu)
    recent_k = fit_recent_k(train, mu)
    conv = fit_recent_conversion(train)
    print("  thin-prior shrinkage fitted on TRAIN only: "
          + ", ".join(f"{c} {k:g}" for c, k in thin_k.items()))
    print("  unified-rung shrinkage fitted on TRAIN only: "
          + ", ".join(f"{c} {k:g}" for c, k in recent_k.items()))
    print("  conversion-block shrinkage, pseudo-ATTEMPTS not minutes: "
          + ", ".join(f"{m} {k:.1f}" for m, (k, _) in conv.items()))

    frames = {"train": train, "validation": val, "train+validation": pooled}
    rows = population_census(frames)
    rows += board_census(pd.read_parquet(features_dir / "draft_pool.parquet"),
                         played, sorted(val["season"].unique()))
    rows += carry_power(frames, mu, thin_k, recent_k)
    rows += ladder_constants_rows(mu, conv)
    rows += preseason_head_to_head(frames, features_dir, mu)
    table = pd.DataFrame(rows)

    _report(table)
    dest = out_dir / "lag_recovery.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} lag recovery rows → {dest}")
    return dest


def _report(t: pd.DataFrame) -> None:
    def mean_r2(measurement: str, split: str, group: str, estimator: str) -> float:
        hit = t[(t["measurement"] == measurement) & (t["split"] == split)
                & (t["group"] == group) & (t["estimator"] == estimator)
                & (t["metric"] == "r2")]
        return float(hit["value"].mean())

    def n_of(measurement: str, split: str, group: str, estimator: str) -> int:
        hit = t[(t["measurement"] == measurement) & (t["split"] == split)
                & (t["group"] == group) & (t["estimator"] == estimator)
                & (t["metric"] == "r2")]
        return int(hit["n"].max()) if len(hit) else 0

    print("\nCarry-forward power, mean R2 over the nine rate targets")
    print(f"  {'arm':<34}{'train':>9}{'validation':>12}{'pooled':>9}{'n pooled':>10}")
    for group, estimator, label in (
            ("veteran", "lag1", "veteran lag-1 (today's design)"),
            ("returnee_lag2", "lag2", "returnee lag-2 (proposed)"),
            ("thin_prior", "lag1_raw", "thin prior, raw lag-1"),
            ("thin_prior", "lag1_shrunk", "thin prior, shrunk lag-1 (proposed)"),
            ("has_history", "recent_shrunk", "ANY history, most-recent shrunk lag"),
            ("returnee_thin", "recent_shrunk", "  of which returnee, thin lag-2"),
            ("no_usable_lag", "recent_shrunk", "  of which away 2+ seasons")):
        print(f"  {label:<34}"
              f"{mean_r2('carry_power', 'train', group, estimator):>9.4f}"
              f"{mean_r2('carry_power', 'validation', group, estimator):>12.4f}"
              f"{mean_r2('carry_power', 'train+validation', group, estimator):>9.4f}"
              f"{n_of('carry_power', 'train+validation', group, estimator):>10,}")

    print("\nReturnees: their own lag-2 against what the rookie head would give them")
    pooled = t[(t["measurement"] == "preseason") & (t["split"] == "train+validation")
               & (t["group"] == "returnee_lag2") & (t["metric"] == "r2")]
    wide = pooled.pivot_table(index="head", columns="estimator", values="value")
    if {"lag2", "preseason_shrunk"} <= set(wide.columns):
        wins = int((wide["lag2"] > wide["preseason_shrunk"]).sum())
        print(f"  mean R2 — lag-2 {wide['lag2'].mean():.4f} against shrunk preseason "
              f"{wide['preseason_shrunk'].mean():.4f}; lag-2 wins {wins} of {len(wide)}")
    for group in ("returnee_lag2", "true_rookie"):
        cov = t[(t["measurement"] == "preseason") & (t["split"] == "train+validation")
                & (t["group"] == group) & (t["metric"] == "coverage")]
        if len(cov):
            print(f"  preseason coverage, {group:<14} "
                  f"{float(cov['value'].iloc[0]) * 100:5.1f}%  "
                  f"(n = {int(cov['n'].iloc[0]):,})")

    print("\nThe draft board, by why the shipped design has no row for him")
    census = t[t["measurement"] == "board_census"]
    for season in sorted(census["season"].unique()):
        s = census[census["season"] == season]
        def val(group: str, metric: str) -> int:
            hit = s[(s["group"] == group) & (s["metric"] == metric)]
            return int(hit["value"].iloc[0]) if len(hit) else 0
        print(f"  {season}: {val('all', 'n_board')} board rows, "
              f"{val('served_today', 'n')} served today")
        for name in GROUPS:
            n, priced = val(name, "n_unserved"), val(name, "n_unserved_priced")
            if n:
                print(f"    {name:<16} {n:>4} unserved, {priced:>3} of them ADP-priced")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
