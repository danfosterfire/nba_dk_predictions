"""Fitting-window x season-trend ladder for the availability head, scored on the TAILS.

`make availability-window`. One row per arm in
`outputs/predictions/availability_window.csv`.

## Why this exists

`model_card_ecdf.csv` says the shipped beta-binomial misses both ends of its own
distribution, and misses them in opposite directions. On validation it puts **5.95%** of
player-seasons at a full schedule against an observed **2.72%**, and **5.21%** below ten
games against an observed **8.15%** — both outside the posterior-predictive band, on the
head that `README.md` calls the largest lever on the season total. Neither miss is visible
in CRPS, MAE or PIT KS, which is why the head has cleared every gate it has ever been
scored on while being wrong about the two events a best-ball roster actually turns on: an
iron man and a dead pick.

## What the two arms are for, and why they are not alternatives

A **shorter fitting window** and a **season trend** answer different halves of the same
measurement, so this ladder crosses them rather than choosing between them.

The league moved. On the head's own design rows, mean `gp_share` sits flat near 0.710 for
twenty seasons and then falls to 0.605-0.638; a sup-F scan over every candidate breakpoint
puts the change at **2017-18** (F = 91.8 against a Monte-Carlo null 95th percentile of 9.2)
and BIC prefers a *broken trend* over a level step. So the movement is a slope change, not
a step:

- a **window** is the right instrument for a completed level shift. It throws away the old
  regime and costs training rows, which for `post_break` is severe — 2017-18 through
  2021-22 is five target seasons, because the training half ends at 2021-22 and validation
  may not be fitted on.
- a **trend** is the right instrument for a continuing slope. It costs one column and
  extrapolates by construction, which is exactly what a walk-forward forecast needs and
  exactly what makes it dangerous: `season_terms` warns that a trend fitted on many seasons
  and extrapolated one or two forward can fit two validation seasons by accident.

Crossing them is what separates the two explanations. If the window alone carries it, the
change was a step and the trend is noise; if the trend alone carries it, truncating was
throwing away usable rows; if they only work together, it is a broken trend and both terms
are doing real work.

`trend_x_role` is here because `docs/availability-plan.md` measured the era effect as
**role-graded** rather than a level shift — heavy-minute players lost -0.101 of games-played
share against -0.037 for fringe players — so a single league-wide slope is the wrong shape
if that holds. Role is bucketed on **prior-season** MPG, known before opening night.

## Why the tails are in the metric set

Because the defect is invisible without them, and because the contest is a threshold
machine: a Round-1 knockout is decided by the best 7 of 16 in a week, so a player who plays
four games is a dead roster slot and a player who plays every game is a ceiling the model
either has or does not. `tail_coverage` reports **predicted against observed** at each
threshold rather than a Brier score, because a Brier score is minimised by a model that is
confidently wrong in the same direction on every row and would not have caught this.

The upper threshold is `GP == team_games` per row rather than a fixed 82, so the shortened
seasons contribute their own boundary instead of an unreachable one.

## What this ladder is and is not

It is a **specification ladder on the point MLE**, which is what `availability.py::run`
itself selects on and what `stan_availability` is verified against (21/21 coefficients
inside the 95% credible interval). It runs in seconds and needs no CmdStan, so an arm can
be rejected before anyone spends sampler time — the same reason `games_played.py` is numpy
only. An arm that wins here earns a Stan port; it does not ship from here.

Selection reads **validation and nothing else**. `selection_split` never materializes the
held-out rows, and every window is a restriction of the *training* half only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.season_effects import ROLE_EDGES, ROLE_LABELS
from src.features.availability import build_panel, season_availability
from src.models.availability import (FEATURE_COLS, BetaBinomialGLM, LeagueAgeBaseline,
                                     build_design, crps, fit_dispersion, pit_values,
                                     predictive_pmf, season_start_dates)
from src.models.held_out import selection_split
from src.models.season_terms import add_role_terms, add_trend, season_start_year

# First season *start year* kept in the fitting half. `None` is the incumbent's 1996-97.
# `three_point_era` is `docs/potential-to-dos.md`'s measured 3PA breakpoint; `post_break`
# is where the sup-F scan puts the availability change, and it is the expensive one —
# five target seasons, because training stops at 2021-22.
WINDOWS: dict[str, int | None] = {
    "full": None,
    "three_point_era": 2012,
    "post_break": 2017,
}

# The season term, on top of a fixed feature block so the contrast is the term alone.
TREND_SPECS = ("none", "trend", "trend_x_role")

# How the beta-binomial's dispersion is allowed to vary. `shared` is the incumbent's one
# scalar for every player in every season; `role` grades it on **prior-season** MPG, the
# device `stan_composition` already ships (0.1768 fringe against 0.0855 star, a 2.07x
# spread that cut its calibration error by 35%). Two candidate explanations for the
# boundary-mass defect — one dispersion for 25 seasons, or one dispersion for every role —
# and crossing them with the window separates era pooling from population pooling.
RHO_MODES = ("shared", "role")

# Low-tail thresholds. 10 is `docs/potential-to-dos.md` item 4's "very few games"; 41 and
# 60 are Gate D's, carried so this table is readable beside `stan_games_played_gates.csv`.
TAIL_BELOW = (10, 41, 60)

# Central predictive intervals whose realized coverage is reported, matching
# `season_terms.COVERAGE_LEVELS` so the two tables read the same way.
COVERAGE_LEVELS = (0.5, 0.8, 0.95)

BOOTSTRAP_REPS = 2000
REFERENCE_ARM = "full__none__shared"   # the shipped head: whole window, no season term,
                                       # one dispersion for every player in every season


# Fewest training rows a role bucket needs before it gets its own dispersion. Below this
# the bucket falls back to the pooled scalar, because a dispersion fitted on a handful of
# rows is noise wearing the shape of a parameter.
MIN_ROLE_ROWS = 100


class RoleGradedBetaBinomial(BetaBinomialGLM):
    """`BetaBinomialGLM` with one dispersion per prior-MPG role bucket.

    The mean function is fitted exactly as the incumbent's; `rho` is then re-fitted inside
    each bucket holding that mean fixed, which is the same two-stage device
    `AvailabilityModel.fit` already uses to give each model the dispersion its own errors
    warrant — applied one level down, to each role's own errors.

    Buckets come from `season_effects.ROLE_EDGES`, the edges `docs/availability-plan.md`
    measured the role-graded era effect on, so a dispersion split here is comparable to the
    level split measured there. Role is **prior-season** MPG: known before opening night,
    so this is not the target grading itself.
    """

    name = "beta_binomial_role_rho"

    @staticmethod
    def _buckets(df: pd.DataFrame) -> np.ndarray:
        return np.asarray(pd.cut(df["minutes_per_game_lag1"].to_numpy(dtype=float),
                                 ROLE_EDGES, labels=ROLE_LABELS).astype(str))

    def fit(self, train: pd.DataFrame) -> "RoleGradedBetaBinomial":
        super().fit(train)
        self.pooled_rho = float(self.rho)
        y, n, mu = (train["gp"].to_numpy(), train["team_games"].to_numpy(),
                    self.predict_mean(train))
        buckets = self._buckets(train)
        self.rho_by_role = {}
        for label in ROLE_LABELS:
            mask = buckets == label
            if mask.sum() >= MIN_ROLE_ROWS:
                self.rho_by_role[label] = fit_dispersion(y[mask], n[mask], mu[mask])
        return self

    @property
    def rho_spread(self) -> float:
        vals = list(self.rho_by_role.values())
        return max(vals) / min(vals) if vals else 1.0

    def predict_pmf(self, df: pd.DataFrame, max_games: int) -> np.ndarray:
        n, mu = df["team_games"].to_numpy(), self.predict_mean(df)
        buckets = self._buckets(df)
        pmf = np.zeros((len(df), max_games + 1))
        for label in np.unique(buckets):
            mask = buckets == label
            pmf[mask] = predictive_pmf(n[mask], mu[mask],
                                       self.rho_by_role.get(label, self.pooled_rho),
                                       max_games)
        return pmf


def restrict_window(train: pd.DataFrame, first_year: int | None) -> pd.DataFrame:
    """The fitting half, cut to a recent suffix of seasons.

    Applied to the **training rows only**. The design itself is built over every season
    regardless, because the lag columns reach back three seasons and trimming the frame
    would silently drop each window's own first cohort.
    """
    if first_year is None:
        return train
    return train[season_start_year(train) >= first_year].copy()


def tail_coverage(pmf: np.ndarray, y: np.ndarray, n: np.ndarray) -> list[dict]:
    """Predicted against observed frequency at each tail, as rates rather than scores.

    The predicted rate is the *mean of the per-row probabilities*, which is what the
    predictive says the population rate is; the observed rate is the realized one. A model
    can hold CRPS and PIT while getting these wrong by a factor of two, which is the whole
    reason this function exists.
    """
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    rows = np.arange(len(y))
    out = []
    for threshold in TAIL_BELOW:
        out.append({"tail": f"below_{threshold}",
                    "predicted": float(cdf[:, threshold - 1].mean()),
                    "observed": float((y < threshold).mean())})
    # The upper boundary is each row's OWN schedule, so a 66- or 72-game season
    # contributes the boundary it actually had.
    out.append({"tail": "full_schedule",
                "predicted": float(pmf[rows, np.asarray(n, dtype=int)].mean()),
                "observed": float((y == n).mean())})
    for row in out:
        row["error"] = row["predicted"] - row["observed"]
        row["abs_error"] = abs(row["error"])
    return out


def interval_coverage(pmf: np.ndarray, y: np.ndarray, level: float) -> float:
    """Realized coverage of the central `level` predictive interval."""
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    alpha = (1.0 - level) / 2.0
    lo = (cdf < alpha).sum(axis=1)
    hi = (cdf < 1.0 - alpha).sum(axis=1)
    return float(((y >= lo) & (y <= hi)).mean())


def build_arm(train: pd.DataFrame, val: pd.DataFrame, spec: str
              ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """`(train, val, features)` for one season-term spec.

    `add_trend` centres on **train**, so each window centres on its own fitting rows and
    the validation seasons sit outside the fitted range in every arm — which is what
    extrapolating one season forward is supposed to mean.
    """
    if spec == "none":
        return train, val, list(FEATURE_COLS)
    (tr, va), trend_names = add_trend(train, [train, val])
    if spec == "trend":
        return tr, va, list(FEATURE_COLS) + trend_names
    (tr, va), role_names = add_role_terms(tr, [tr, va])
    return tr, va, list(FEATURE_COLS) + trend_names + role_names


def paired_bootstrap(arm: np.ndarray, reference: np.ndarray, reps: int = BOOTSTRAP_REPS,
                     seed: int = 42) -> tuple[float, float, float]:
    """`(mean difference, lo, hi)` on arm − reference CRPS, resampling rows in pairs.

    Paired inside the row, because both arms score the same validation player-seasons and
    the between-player variance is an order of magnitude larger than the between-arm one.
    This is the discipline `docs/potential-to-dos.md` names as the difference between a
    finding and a prompt.
    """
    diff = arm - reference
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    boot = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def score_arm(name: str, model, train: pd.DataFrame, val: pd.DataFrame,
              features: list[str], max_games: int, seed: int = 42) -> tuple[dict, np.ndarray]:
    """One fitted arm's row, and its per-row CRPS for the paired bootstrap."""
    y = val["gp"].to_numpy()
    n = val["team_games"].to_numpy()
    mu = model.predict_mean(val)
    # Through the model rather than through `predictive_pmf` directly, so a head that
    # varies its dispersion per row supplies its own predictive instead of being flattened
    # back to a scalar by the scorer.
    pmf = model.predict_pmf(val, max_games)
    scores = crps(pmf, y)

    share, pred_share = y / n, mu
    ss_res = float(np.sum((share - pred_share) ** 2))
    ss_tot = float(np.sum((share - share.mean()) ** 2))
    u = pit_values(pmf, y, seed)
    grid = np.linspace(0, 1, 101)
    ks = float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid)))

    row = {
        "arm": name,
        "n_train": len(train),
        "n_train_seasons": int(pd.Series(train["season"]).nunique()),
        "first_train_season": min(train["season"]),
        "n_features": len(features),
        "val_crps": float(scores.mean()),
        "val_mae": float(np.abs(y - mu * n).mean()),
        "val_r2_gp_share": 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan,
        "val_pit_ks": ks,
        "rho": float(model.rho),
        "rho_spread": float(getattr(model, "rho_spread", 1.0)),
        "implied_overdispersion": float(1 + (np.median(n) - 1) * model.rho),
    }
    for level in COVERAGE_LEVELS:
        row[f"coverage_{int(level * 100)}"] = interval_coverage(pmf, y, level)
    tails = tail_coverage(pmf, y, n)
    by_name = {t["tail"]: t for t in tails}
    for t in tails:
        row[f"pred_{t['tail']}"] = t["predicted"]
        row[f"obs_{t['tail']}"] = t["observed"]
        row[f"err_{t['tail']}"] = t["error"]
    # Two summaries, because the first one is misleading on its own and that is worth
    # keeping visible rather than deleting. `mean_abs_tail_error` averages all four
    # thresholds and is dominated by 41 and 60, which sit in the BODY of an 82-game
    # distribution, not its tails — an arm that shifts location can improve both
    # boundaries while wrecking those two and still look worse. `boundary_tail_error` is
    # the pair the defect is actually about: a dead roster slot and an iron man.
    row["mean_abs_tail_error"] = float(np.mean([t["abs_error"] for t in tails]))
    row["boundary_tail_error"] = float(np.mean([by_name["below_10"]["abs_error"],
                                                by_name["full_schedule"]["abs_error"]]))
    return row, scores


def ladder(train: pd.DataFrame, val: pd.DataFrame, max_games: int,
           l2: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Every (window x season-term) arm, plus the league/age floor, scored on validation."""
    rows: list[dict] = []
    per_row: dict[str, np.ndarray] = {}

    floor = LeagueAgeBaseline().fit(train)
    row, scores = score_arm("league_age_floor", floor, train, val, [], max_games, seed)
    rows.append(row)
    per_row["league_age_floor"] = scores

    for window, first_year in WINDOWS.items():
        cut = restrict_window(train, first_year)
        for spec in TREND_SPECS:
            for rho_mode in RHO_MODES:
                name = f"{window}__{spec}__{rho_mode}"
                tr, va, features = build_arm(cut, val, spec)
                cls = BetaBinomialGLM if rho_mode == "shared" else RoleGradedBetaBinomial
                model = cls(l2=l2, features=features).fit(tr)
                row, scores = score_arm(name, model, tr, va, features, max_games, seed)
                rows.append(row)
                per_row[name] = scores
                print(f"  {name:<40} {len(tr):>6,} rows / "
                      f"{row['n_train_seasons']:>2} seasons   CRPS {row['val_crps']:.4f}   "
                      f"rho {row['rho']:.4f} ({row['rho_spread']:.2f}x)   "
                      f"boundary err {row['boundary_tail_error']:.4f}")

    reference = per_row[REFERENCE_ARM]
    for row in rows:
        d, lo, hi = paired_bootstrap(per_row[row["arm"]], reference, seed=seed)
        row["crps_vs_incumbent"] = d
        row["crps_vs_incumbent_lo"] = lo
        row["crps_vs_incumbent_hi"] = hi
        row["beats_incumbent"] = bool(hi < 0.0)
    return pd.DataFrame(rows)


# ── Rolling-origin confirmation ───────────────────────────────────────────────
#
# The ladder above scores 19 arms on the 883 validation rows, which is a multiplicity
# problem and a power problem at once. This block answers it **without spending validation
# twice**: it walks an origin across the training half, fits on the seasons before it and
# scores the season itself, and pools. Every row it touches is a fitting-half row, so the
# validation reading stays the arbiter it was — this asks only whether that reading
# replicates.
#
# It also reparameterizes the window in the way that actually matters. `WINDOWS` names an
# absolute first season, which is a fact about the past; a **lookback length** is a policy
# that will still mean something when the production fit runs for 2026-27. Comparing
# lookbacks across origins is the bias-variance curve directly.

# Lookback lengths in target seasons. `None` is "every season before the origin", the
# incumbent's rule. 12 is the longest that is genuine at the earliest origin below.
LOOKBACKS: tuple[int | None, ...] = (None, 12, 8, 5, 3)

# First origin. Chosen so the longest finite lookback is real rather than silently
# truncated to `None` — the training half starts at target season 1997-98, so 1997 + 12.
FIRST_ORIGIN = 2009

ROLLING_REFERENCE = "all__none__shared"


def _origin_scores(model, score: pd.DataFrame, max_games: int) -> dict[str, np.ndarray]:
    """Per-row quantities for pooling across origins.

    The **body** thresholds are carried as well as the boundaries, because the whole
    verdict on a season trend turns on whether it pays for its boundaries out of the
    middle. Reporting only the boundaries here would have made the trend look free.
    """
    y = score["gp"].to_numpy()
    n = score["team_games"].to_numpy()
    pmf = model.predict_pmf(score, max_games)
    cdf = np.clip(np.cumsum(pmf, axis=1), 0.0, 1.0)
    out = {"crps": crps(pmf, y), "pit": pit_values(pmf, y),
           "p_full": pmf[np.arange(len(y)), n.astype(int)], "y": y, "n": n}
    for threshold in TAIL_BELOW:
        out[f"p_below_{threshold}"] = cdf[:, threshold - 1]
    return out


def rolling_confirmation(train: pd.DataFrame, max_games: int, l2: float = 1.0,
                         seed: int = 42) -> pd.DataFrame:
    """Walk-forward over the fitting half: one fit per (origin, arm), pooled.

    The origin is a *target* season; the arm fits on target seasons strictly before it.
    Nothing here reads the validation or held-out rows, so this is a confirmation of the
    ladder rather than a second selection on the same data.
    """
    years = np.asarray(sorted(np.unique(season_start_year(train))))
    origins = [int(y) for y in years if y >= FIRST_ORIGIN]
    per_arm: dict[str, dict[str, list]] = {}

    for origin in origins:
        year = season_start_year(train)
        score = train[year == origin]
        if score.empty:
            continue
        for lookback in LOOKBACKS:
            floor_year = -np.inf if lookback is None else origin - lookback
            fit_rows = train[(year < origin) & (year >= floor_year)]
            if len(fit_rows) < MIN_ROLE_ROWS * len(ROLE_LABELS):
                continue
            for spec in ("none", "trend"):
                for rho_mode in RHO_MODES:
                    name = f"{'all' if lookback is None else lookback}__{spec}__{rho_mode}"
                    tr, sc, features = build_arm(fit_rows, score, spec)
                    cls = BetaBinomialGLM if rho_mode == "shared" else RoleGradedBetaBinomial
                    model = cls(l2=l2, features=features).fit(tr)
                    scored = _origin_scores(model, sc, max_games)
                    scored["origin"] = np.full(len(sc), origin)
                    slot = per_arm.setdefault(name, {"n_fit": []})
                    for key, values in scored.items():
                        slot.setdefault(key, []).append(values)
                    slot["n_fit"].append(len(tr))
        print(f"  origin {origin}: scored {len(score):,} rows")

    pooled = {k: {m: np.concatenate(v) for m, v in d.items() if m != "n_fit"}
              for k, d in per_arm.items()}
    reference = pooled[ROLLING_REFERENCE]["crps"]
    grid = np.linspace(0, 1, 101)
    rows = []
    for name, d in pooled.items():
        y, n, org = d["y"], d["n"], d["origin"]
        mean, lo, hi = paired_bootstrap(d["crps"], reference, seed=seed)
        # Origins are the independent replicates, so a win count over them is the
        # multiplicity-robust statement a pooled interval is not.
        wins = sum(1 for o in np.unique(org)
                   if d["crps"][org == o].mean() < reference[org == o].mean())
        u = d["pit"]
        rows.append({
            "arm": name,
            "lookback": name.split("__")[0],
            "spec": name.split("__")[1],
            "rho_mode": name.split("__")[2],
            "n_origins": int(len(np.unique(org))),
            "n_scored": int(len(y)),
            "mean_fit_rows": float(np.mean(per_arm[name]["n_fit"])),
            "crps": float(d["crps"].mean()),
            "crps_vs_all": mean,
            "crps_vs_all_lo": lo,
            "crps_vs_all_hi": hi,
            "origins_won": wins,
            "pit_ks": float(np.max(np.abs(np.searchsorted(np.sort(u), grid) / len(u) - grid))),
            "pred_full_schedule": float(d["p_full"].mean()),
            "obs_full_schedule": float((y == n).mean()),
            **{f"pred_below_{t}": float(d[f"p_below_{t}"].mean()) for t in TAIL_BELOW},
            **{f"obs_below_{t}": float((y < t).mean()) for t in TAIL_BELOW},
        })
    out = pd.DataFrame(rows)
    out["err_full_schedule"] = out["pred_full_schedule"] - out["obs_full_schedule"]
    for threshold in TAIL_BELOW:
        out[f"err_below_{threshold}"] = (out[f"pred_below_{threshold}"]
                                         - out[f"obs_below_{threshold}"])
    out["boundary_tail_error"] = (out["err_below_10"].abs()
                                  + out["err_full_schedule"].abs()) / 2
    # The body, carried beside the boundaries so a location shift cannot look free.
    out["body_error"] = (out["err_below_41"].abs() + out["err_below_60"].abs()) / 2
    return out.sort_values("crps").reset_index(drop=True)


def run(cfg: dict) -> dict[str, Path]:
    raw_dir = Path(cfg["data"]["raw_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seasons = cfg["data"]["seasons"]
    cfg_av = cfg.get("features", {}).get("availability", {})
    seed = int(cfg_av.get("seed", 42))
    l2 = float(cfg_av.get("glm_l2", 1.0))

    panel = build_panel(seasons, raw_dir)
    frame = season_availability(panel, "full")
    design = build_design(frame, seasons, raw_dir, season_start_dates(panel))
    train, val = selection_split(design)
    max_games = int(design["team_games"].max())

    print(f"Availability window ladder: {len(train):,} train / {len(val):,} validation "
          f"({', '.join(sorted(val['season'].unique()))})")
    print("  The held-out split is LOCKED (src/models/held_out.py) and is never "
          "materialized here.\n  Every window restricts the FITTING half only.")
    table = ladder(train, val, max_games, l2=l2, seed=seed)

    dest = out_dir / "availability_window.csv"
    table.to_csv(dest, index=False)
    print(f"\nWrote {len(table):,} arms → {dest}")

    print("\nRolling-origin confirmation on the FITTING HALF ONLY "
          f"(origins from {FIRST_ORIGIN}, lookbacks {LOOKBACKS}):")
    rolling = rolling_confirmation(train, max_games, l2=l2, seed=seed)
    roll_dest = out_dir / "availability_window_rolling.csv"
    rolling.to_csv(roll_dest, index=False)
    print(f"\nWrote {len(rolling):,} arms × "
          f"{int(rolling['n_scored'].max()):,} scored rows → {roll_dest}")
    return {"availability_window": dest, "availability_window_rolling": roll_dest}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
