"""What the availability head is worth on the number the project actually predicts.

`docs/availability-plan.md` names this as "the downstream metric that actually matters":
season DK total MAE with the availability head wired in, against the current implicit
treatment. Everything upstream is measured in *games* — CRPS 10.91 against a baseline's
13.61 — and games are not the deliverable. This module composes

```
season_total = games_played x dk_pts_per_game_played
```

holding the **rate model fixed** and varying only the games-played treatment, so the
difference between rows is the availability head and nothing else.

Five treatments, two of which are oracles that exist to bound the answer:

| treatment | games played |
|---|---|
| `full_season` | every game on the schedule — the naive "no availability model" case |
| `prior_gp` | last season's games-played share, carried forward (persists at r = 0.316) |
| `league_age` | the shrink-to-league/age baseline the README standing instruction asks for |
| `beta_binomial` | the fitted head |
| `oracle_gp` | **realized** games played, with the predicted rate |
| `oracle_rate` | predicted games played, with the **realized** rate |

The two oracles are the point of the design. Season-total error decomposes into an
availability part and a rate part, and without them a headline MAE cannot say which one it
is dominated by — the plan's whole thesis is that availability is the larger and less
tractable half, and that is a claim this module can check rather than repeat.

**The distribution, not just the mean.** A point GP estimate makes the total a point too.
The head emits a pmf over games, so the predicted total is its pushforward through
`k -> k * rate`: a discrete distribution with atoms at `k * rate` and the pmf's own weights.
That is scored with CRPS in *dk_pts*, which is the season-total analogue of the CRPS in
games that stage E reported.

> **What this is not.** The rate model here is deliberately plain — a ridge on prior-season
> rate, minutes, availability and age. It is not the project's eventual component-head rate
> model, and its R² should not be read as one. It is held identical across every row of the
> comparison, which is all that is required for the availability contrast to be clean.
> `E[GP x rate] = E[GP] x E[rate]` also assumes the two are conditionally independent given
> the features; they are not exactly, so `oracle_rate` carries the residual.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.eda.availability import with_lags
from src.features.availability import build_panel, season_availability
from src.models.availability import (
    TEST_SEASONS,
    BetaBinomialGLM,
    LeagueAgeBaseline,
    build_design,
    season_start_dates,
    split_seasons,
)

# Prior-season inputs to the fixed rate model. Rate persists at 0.869 per CLAUDE.md, so
# this is the easy half; minutes and availability are here because rotation status moves
# the rate too, and age because the aging work says the arc is real if small (+-15%).
#
# Only `dk_per_game` is lagged here — `build_design` already carries `gp_share_lag1`,
# `minutes_per_game_lag1`, `age`, `age_sq` and `career_year`, and re-deriving them would
# collide on the merge and silently produce `_x`/`_y` columns.
RATE_LAG_COLS = ["dk_per_game"]
RATE_FEATURES = ["dk_per_game_lag1", "minutes_per_game_lag1", "gp_share_lag1",
                 "age", "age_sq", "career_year"]
RATE_RIDGE_ALPHA = 10.0

# Established rotation players — the population the plan quotes its tail figures on, and
# the one where a season-total forecast is worth the most money.
ROTATION_MIN_MPG = 20.0
ROTATION_MIN_GP_SHARE = 0.70


# ── The season-total frame ────────────────────────────────────────────────────

def season_rates(targets: pd.DataFrame) -> pd.DataFrame:
    """Realized per (player, season): games, DK total, and the per-game rate.

    Keyed on games *played* rather than on the schedule, because the rate this composes
    with is "dk_pts on a night he plays". A zero-minute row scores zero and would drag the
    rate toward zero while telling us nothing about it.
    """
    played = targets[targets["played"] == 1]
    out = (played.groupby(["player_id", "season"])
           .agg(gp_played=("dk_pts", "size"), dk_total=("dk_pts", "sum"),
                minutes_total=("min", "sum"))
           .reset_index())
    out["dk_per_game"] = out["dk_total"] / out["gp_played"]
    out["minutes_per_game"] = out["minutes_total"] / out["gp_played"]
    return out


def build_frame(cfg: dict) -> pd.DataFrame:
    """One row per (player, target season) carrying both halves and the GP design.

    The availability design and the rate frame are built independently and joined, so the
    games-played model is fitted on exactly the rows and denominators stage E used.

    That leaves two games-played columns: the design's `gp` (from the availability panel,
    which the head is fitted against) and `gp_played` (from the component targets, which
    the realized rate divides by). Checked rather than assumed — they agree on **99.7%** of
    rows and never differ by more than one game, so the pairing is safe. `oracle_gp` still
    uses `gp_played` so that the two oracles together reproduce `dk_total` exactly.
    """
    raw_dir = Path(cfg["data"]["raw_dir"])
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = cfg["data"]["seasons"]

    targets = pd.read_parquet(features_dir / "component_targets.parquet",
                              columns=["player_id", "season", "dk_pts", "min", "played"])
    rates = season_rates(targets)

    panel = build_panel(seasons, raw_dir)
    availability = season_availability(panel, "full")
    design = build_design(availability, seasons, raw_dir, season_start_dates(panel))

    lagged = with_lags(rates, seasons, RATE_LAG_COLS, max_lag=1)
    keep = (["player_id", "season", "gp_played", "dk_total", "dk_per_game"]
            + [f"{c}_lag1" for c in RATE_LAG_COLS])
    frame = design.merge(lagged[keep], on=["player_id", "season"], how="inner")
    return frame.dropna(subset=RATE_FEATURES + ["dk_total", "gp_played"]).reset_index(
        drop=True)


# ── The fixed rate model ──────────────────────────────────────────────────────

class RateModel:
    """Predicted dk_pts on a night he plays. Identical across every GP treatment."""

    def fit(self, train: pd.DataFrame) -> "RateModel":
        X = train[RATE_FEATURES].to_numpy(dtype=float)
        self.scaler = StandardScaler().fit(X)
        self.model = Ridge(alpha=RATE_RIDGE_ALPHA).fit(
            self.scaler.transform(X), train["dk_per_game"].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(df[RATE_FEATURES].to_numpy(dtype=float))
        return np.clip(self.model.predict(X), 0.0, None)


# ── Scoring a distribution over the total ─────────────────────────────────────

def crps_from_atoms(atoms: np.ndarray, weights: np.ndarray, y: np.ndarray) -> np.ndarray:
    """CRPS of a discrete predictive distribution, in the units of `atoms`.

    Kernel form, `E|X - y| - 0.5 E|X - X'|`, which is exact for an atomic distribution and
    avoids choosing a grid. `atoms` is (rows x K) and must be sorted ascending along K —
    it is, because it is `k * rate` for increasing k and a non-negative rate.
    """
    atoms = np.asarray(atoms, dtype=float)
    weights = np.asarray(weights, dtype=float)
    y = np.asarray(y, dtype=float)[:, None]
    weights = weights / np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    term1 = np.sum(weights * np.abs(atoms - y), axis=1)
    # E|X - X'| = 2 * sum_k w_k x_k (2 W_k - w_k - 1), with W the cumulative weight.
    # Derived by splitting the double sum at j < k: the O(K) reduction, since the literal
    # K x K form would be (rows x 83 x 83).
    cw = np.cumsum(weights, axis=1)
    term2 = 2.0 * np.sum(weights * atoms * (2.0 * cw - weights - 1.0), axis=1)
    return term1 - 0.5 * term2


# ── Treatments ────────────────────────────────────────────────────────────────

def gp_treatments(train: pd.DataFrame, test: pd.DataFrame,
                  max_games: int) -> dict[str, dict]:
    """Predicted games played under each treatment, with a pmf where one exists.

    Returns `{name: {"gp": array, "pmf": array | None}}`. The oracles carry the realized
    value and no distribution — they are bounds, not forecasts.
    """
    out: dict[str, dict] = {}
    n = test["team_games"].to_numpy(dtype=float)

    out["full_season"] = {"gp": n.copy(), "pmf": None}
    out["prior_gp"] = {"gp": np.clip(test["gp_share_lag1"].to_numpy(dtype=float), 0, 1) * n,
                       "pmf": None}

    for model in (LeagueAgeBaseline(), BetaBinomialGLM()):
        model.fit(train)
        pmf = model.predict_pmf(test, max_games)
        out[model.name] = {"gp": model.predict_mean(test) * n, "pmf": pmf}

    # `gp_played`, not the design's `gp`: the realized rate is `dk_total / gp_played`, so
    # this pairing is the one that reproduces `dk_total` exactly when both halves are
    # oracles. Using the other column would leave a residual in a row labelled "perfect".
    out["oracle_gp"] = {"gp": test["gp_played"].to_numpy(dtype=float), "pmf": None}
    return out


def _row(treatment: str, group: str, metric: str, value: float, n: int) -> dict:
    return {"treatment": treatment, "group": group, "metric": metric,
            "n": n, "value": value}


def evaluate(test: pd.DataFrame, rate: np.ndarray, treatments: dict[str, dict],
             max_games: int) -> tuple[list[dict], pd.DataFrame]:
    """Season-total metrics per treatment, plus a tidy prediction frame."""
    y = test["dk_total"].to_numpy(dtype=float)
    rotation = ((test["minutes_per_game_lag1"] >= ROTATION_MIN_MPG)
                & (test["gp_share_lag1"] >= ROTATION_MIN_GP_SHARE)).to_numpy()
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    rows, frames = [], []
    for name, spec in treatments.items():
        # `oracle_rate` swaps the rate rather than the games, so it is built here where
        # both halves are in scope.
        r = test["dk_per_game"].to_numpy(dtype=float) if name == "oracle_rate" else rate
        pred = spec["gp"] * r
        err = np.abs(pred - y)

        for group, mask in (("all", np.ones(len(y), bool)), ("rotation", rotation)):
            if not mask.any():
                continue
            rows.append(_row(name, group, "mae_dk_total", float(err[mask].mean()),
                             int(mask.sum())))
            rows.append(_row(name, group, "rmse_dk_total",
                             float(np.sqrt(((pred - y)[mask] ** 2).mean())),
                             int(mask.sum())))
        rows.append(_row(name, "all", "r2_dk_total",
                         float(1 - np.sum((pred - y) ** 2) / ss_tot), len(y)))
        rows.append(_row(name, "all", "bias_dk_total", float((pred - y).mean()), len(y)))

        if spec["pmf"] is not None:
            atoms = np.arange(max_games + 1)[None, :] * r[:, None]
            score = crps_from_atoms(atoms, spec["pmf"], y)
            rows.append(_row(name, "all", "crps_dk_total", float(score.mean()), len(y)))
            if rotation.any():
                rows.append(_row(name, "rotation", "crps_dk_total",
                                 float(score[rotation].mean()), int(rotation.sum())))

        frames.append(pd.DataFrame({
            "season": test["season"].values, "player_id": test["player_id"].values,
            "treatment": name, "predicted_gp": spec["gp"], "predicted_rate": r,
            "predicted_dk_total": pred, "dk_total": y, "abs_error": err,
            "rotation": rotation}))
    return rows, pd.concat(frames, ignore_index=True)


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    test_seasons = int(cfg_av.get("test_seasons", TEST_SEASONS))

    frame = build_frame(cfg)
    train, test = split_seasons(frame, test_seasons)
    max_games = int(frame["team_games"].max())

    print(f"Season-total design: {len(frame):,} player-seasons, "
          f"{len(train):,} train / {len(test):,} test "
          f"({', '.join(sorted(test['season'].unique()))} held out)")

    rate_model = RateModel().fit(train)
    rate = rate_model.predict(test)
    rate_r2 = 1 - (np.sum((rate - test["dk_per_game"]) ** 2)
                   / np.sum((test["dk_per_game"] - test["dk_per_game"].mean()) ** 2))
    print(f"  Fixed rate model: held-out R² {rate_r2:.4f} on dk_pts per game played. "
          f"Identical across\n  every treatment below, so the contrast is the games-played "
          f"model and nothing else.")

    treatments = gp_treatments(train, test, max_games)
    treatments["oracle_rate"] = {"gp": treatments["beta_binomial"]["gp"], "pmf": None}

    rows, predictions = evaluate(test, rate, treatments, max_games)
    metrics = pd.DataFrame(rows)

    order = ["full_season", "prior_gp", "league_age", "beta_binomial",
             "oracle_rate", "oracle_gp"]
    table = (metrics[metrics.group == "all"]
             .pivot_table(index="treatment", columns="metric", values="value")
             .reindex([t for t in order if t in set(metrics.treatment)]))
    cols = [c for c in ["mae_dk_total", "rmse_dk_total", "r2_dk_total",
                        "bias_dk_total", "crps_dk_total"] if c in table.columns]
    print("\nSeason DK total, held out (MAE in dk_pts, lower is better):")
    print(table[cols].round(1).to_string())

    naive = float(table.loc["full_season", "mae_dk_total"])
    head = float(table.loc["beta_binomial", "mae_dk_total"])
    prior = float(table.loc["prior_gp", "mae_dk_total"])
    print(f"\nThe availability head is worth {naive - head:+.1f} dk_pts of season-total MAE "
          f"against\n  assuming a full season ({naive:.1f} → {head:.1f}, "
          f"{(naive - head) / naive:.1%}), and {prior - head:+.1f} against carrying prior "
          f"games\n  forward ({prior:.1f}).")

    o_gp = float(table.loc["oracle_gp", "mae_dk_total"])
    o_rate = float(table.loc["oracle_rate", "mae_dk_total"])
    print(f"\nWhich half dominates? Perfect games played would give {o_gp:.1f} MAE; "
          f"perfect rate\n  would give {o_rate:.1f}, against the achievable "
          f"{head:.1f}. Availability is worth\n  {head - o_gp:.1f} dk_pts of the remaining "
          f"error and the rate {head - o_rate:.1f}.")

    rot = (metrics[metrics.group == "rotation"]
           .pivot_table(index="treatment", columns="metric", values="value")
           .reindex([t for t in order if t in set(metrics.treatment)]))
    print("\nEstablished rotation players only — where a season total is worth the most:")
    print(rot[[c for c in cols if c in rot.columns]].round(1).to_string())

    paths = {}
    for name, (data, dest) in {
            "metrics": (metrics, out_dir / "season_total_metrics.csv"),
            "predictions": (predictions, out_dir / "season_total_predictions.csv"),
    }.items():
        data.to_csv(dest, index=False)
        paths[name] = dest
        print(f"\nSaved {len(data):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
