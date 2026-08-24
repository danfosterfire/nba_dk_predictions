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

## Which rows the table is scored on

**Validation**, since 2026-08-05. This module is where a games-played treatment is *chosen*
— the whole design is a ladder of six of them scored against each other, and Gate E of
`docs/games-played-plan.md` asks it to arbitrate between the incumbent beta-binomial and
the spell process. A comparison that decides something has to be run on the split that is
allowed to decide, and until this change it was not: the entire headline table was a test
evaluation, quoted in four documents as the project's deliverable-level result.

`compare` is the whole measurement and takes whichever pair of frames it is handed.
`src/final_evaluation.py` calls it on `(train + validation, test)` once, at the end. The
`oracle_gp` / `oracle_rate` bound and the naive/prior-GP ladder are unaffected in kind by
the move — they are arithmetic on whichever rows they are given — but every number moves,
because the rows and the training frame both change.
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
)
from src.models.held_out import selection_split
from src.models.stan_utils import crps_from_samples

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

#: Extra populations `evaluate` reports when the frame names them — the rookie program's
#: season-total readout (`docs/rookie-rates-plan.md` §5e), which is §16's settling gate.
#:
#: **Two rather than one, and they must never be pooled.** `rookie` is the true-rookie
#: population Session 3's head serves and `lag_recovered` is the rows §5b's ladder admitted;
#: they are different populations served by different heads, and an average of the two would
#: be an average of two regimes — the mistake §3 constraint 2' of that plan exists to undo.
#: `veteran` rides beside them as the bar, which is `lag_ladder`'s own arrangement for rung 0:
#: a season-total MAE in dk_pts is unreadable without the shipped population's own number
#: next to it.
#:
#: The frame this module's own `build_frame` returns carries no `population` column, so on
#: the availability ladder every one of these masks is empty and `evaluate` skips it —
#: `season_total_metrics.csv` is unchanged by their existence.
POPULATION_COLUMN = "population"
POPULATION_GROUPS = ("veteran", "lag_recovered", "rookie")


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

SPELL_PMF_FILE = "stan_games_played_gp_pmf.csv"
# The test-side twin, written by `src/final_evaluation.py::_games_played` and by nothing
# else. Two filenames rather than one, because a single path would let a validation-side
# pmf silently answer a held-out question the moment the row keys happened to overlap.
FINAL_SPELL_PMF_FILE = "final_evaluation_gp_pmf.csv"


def spell_process_pmf(out_dir: Path, frame: pd.DataFrame, max_games: int,
                      pmf_file: str = SPELL_PMF_FILE) -> np.ndarray | None:
    """The spell process's games-played pmf, if `make stan-games-played` has run.

    Read from an artifact rather than imported, because `src/models/stan_games_played.py`
    needs a CmdStan toolchain and this module does not — the same separation
    `games_played.py` keeps from its own Stan half. Returns `None` when the artifact is
    absent, so a fresh checkout scores the original five treatments and says so.

    **Partial coverage is refused rather than filled.** This frame is an inner join of the
    availability design with the rate frame, so it is a near-subset of the head's own
    rows; a row the head did not predict would otherwise get a silent zero pmf, which is
    an infinitely confident forecast of zero games rather than a missing one. That refusal
    is also what keeps the two pmf files from crossing splits: handed a validation pmf and
    a test frame, the key join covers nothing and the treatment is skipped loudly.
    """
    path = Path(out_dir) / pmf_file
    if not path.exists():
        return None
    long = pd.read_csv(path)
    keys = frame[["season", "player_id"]].copy()
    keys["_row"] = np.arange(len(frame))
    joined = long.merge(keys, on=["season", "player_id"], how="inner")

    covered = joined["_row"].nunique()
    if covered < len(frame):
        print(f"  /!\\  {pmf_file} covers {covered:,} of {len(frame):,} "
              f"season-total rows; skipping the spell-process treatment rather than "
              f"zero-filling {len(frame) - covered:,} of them")
        return None

    pmf = np.zeros((len(frame), max_games + 1))
    keep = joined["gp"] <= max_games
    pmf[joined.loc[keep, "_row"].to_numpy(int),
        joined.loc[keep, "gp"].to_numpy(int)] = joined.loc[keep, "p"].to_numpy(float)
    return pmf / np.clip(pmf.sum(axis=1, keepdims=True), 1e-12, None)


def gp_treatments(train: pd.DataFrame, frame: pd.DataFrame, max_games: int,
                  out_dir: Path | None = None,
                  pmf_file: str = SPELL_PMF_FILE) -> dict[str, dict]:
    """Predicted games played under each treatment, with a pmf where one exists.

    Returns `{name: {"gp": array, "pmf": array | None}}`. The oracles carry the realized
    value and no distribution — they are bounds, not forecasts.
    """
    out: dict[str, dict] = {}
    n = frame["team_games"].to_numpy(dtype=float)

    out["full_season"] = {"gp": n.copy(), "pmf": None}
    out["prior_gp"] = {"gp": np.clip(frame["gp_share_lag1"].to_numpy(dtype=float), 0, 1) * n,
                       "pmf": None}

    for model in (LeagueAgeBaseline(), BetaBinomialGLM()):
        model.fit(train)
        pmf = model.predict_pmf(frame, max_games)
        out[model.name] = {"gp": model.predict_mean(frame) * n, "pmf": pmf}

    # Gate E: the same composition, with games played coming from the spell process
    # instead of the season-level beta-binomial. Everything else — the rate model, the
    # rows, the scoring — is held identical, so the contrast is the games-played
    # treatment and nothing else, exactly as it is for the four rows above.
    if out_dir is not None:
        spell = spell_process_pmf(out_dir, frame, max_games, pmf_file)
        if spell is not None:
            k = np.arange(max_games + 1)[None, :]
            out["spell_process"] = {"gp": (spell * k).sum(axis=1), "pmf": spell}

    # `gp_played`, not the design's `gp`: the realized rate is `dk_total / gp_played`, so
    # this pairing is the one that reproduces `dk_total` exactly when both halves are
    # oracles. Using the other column would leave a residual in a row labelled "perfect".
    out["oracle_gp"] = {"gp": frame["gp_played"].to_numpy(dtype=float), "pmf": None}
    return out


def _row(treatment: str, group: str, metric: str, value: float, n: int) -> dict:
    return {"treatment": treatment, "group": group, "metric": metric,
            "n": n, "value": value}


def rotation_mask(frame: pd.DataFrame) -> np.ndarray:
    """Established rotation players, or all-False on a frame that cannot say.

    The rookie-admitting frame is three designs concatenated and only one of them is a
    lag design, so the two prior-season columns this mask reads do not exist on it. That
    is a population without an answer rather than a population of size zero, and an
    all-False mask is what `evaluate` already skips.
    """
    cols = ("minutes_per_game_lag1", "gp_share_lag1")
    if not set(cols) <= set(frame.columns):
        return np.zeros(len(frame), dtype=bool)
    return ((frame["minutes_per_game_lag1"] >= ROTATION_MIN_MPG)
            & (frame["gp_share_lag1"] >= ROTATION_MIN_GP_SHARE)).to_numpy()


def metric_groups(frame: pd.DataFrame) -> list[tuple[str, np.ndarray]]:
    """`(name, mask)` for every population a table reports.

    `all` and `rotation` always; `POPULATION_GROUPS` when the frame names them. Read off a
    column rather than re-derived here, because the label that separates a true rookie from
    a ladder-recovered returnee is `lag_recovery.classify`'s and a second implementation of
    it is a place where a board and this table could silently disagree.
    """
    groups = [("all", np.ones(len(frame), dtype=bool)),
              ("rotation", rotation_mask(frame))]
    if POPULATION_COLUMN in frame.columns:
        labels = frame[POPULATION_COLUMN].to_numpy()
        groups += [(g, labels == g) for g in POPULATION_GROUPS]
    return groups


def predictive_crps(spec: dict, rate: np.ndarray, y: np.ndarray,
                    max_games: int) -> np.ndarray | None:
    """Per-row CRPS of a treatment's predictive over the season total, or `None`.

    Two shapes, because the two ladders put their distribution on different halves of
    `gp x rate`. The availability ladder varies **games** and carries a pmf over them, so
    the total's predictive is that pmf pushed through `k -> k * rate`. The rookie readout
    (`docs/rookie-rates-plan.md` §5e) varies the **rate** and carries draws of the total
    itself, because eleven component heads composed through the chain do not collapse to a
    pmf on a 0-83 grid. A treatment carrying neither is a point forecast and scores no CRPS
    — which is the honest shape for the two oracles and for a plugged-in level.
    """
    if spec.get("samples") is not None:
        return crps_from_samples(np.asarray(spec["samples"], dtype=float), y)
    if spec.get("pmf") is None:
        return None
    atoms = np.arange(max_games + 1)[None, :] * rate[:, None]
    return crps_from_atoms(atoms, spec["pmf"], y)


def evaluate(frame: pd.DataFrame, rate: np.ndarray | None, treatments: dict[str, dict],
             max_games: int) -> tuple[list[dict], pd.DataFrame]:
    """Season-total metrics per treatment, plus a tidy prediction frame.

    `rate` is the shared rate model — the availability ladder's whole design is that it is
    held identical across every row. `None` says there isn't one, which is the rookie
    readout's case: there the rate family is what varies and every treatment carries its
    own, so a treatment that forgot to is an error rather than a silent zero.
    """
    test = frame
    y = test["dk_total"].to_numpy(dtype=float)
    groups = metric_groups(test)
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    rows, frames = [], []
    for name, spec in treatments.items():
        # A treatment may swap the RATE rather than the games — `oracle_rate` does, and so
        # does every arm of the rookie readout, where the rate family is the thing under
        # test and the games treatment is what is held fixed.
        supplied = spec.get("rate")
        if supplied is None and rate is None:
            raise ValueError(f"treatment {name!r} supplies no rate and `evaluate` was "
                             f"handed no shared one")
        r = rate if supplied is None else np.asarray(supplied, dtype=float)
        pred = spec["gp"] * r
        err = np.abs(pred - y)

        for group, mask in groups:
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

        score = predictive_crps(spec, r, y, max_games)
        if score is not None:
            for group, mask in groups:
                if not mask.any():
                    continue
                rows.append(_row(name, group, "crps_dk_total",
                                 float(score[mask].mean()), int(mask.sum())))

        frames.append(pd.DataFrame({
            "season": test["season"].values, "player_id": test["player_id"].values,
            "treatment": name, "predicted_gp": spec["gp"], "predicted_rate": r,
            "predicted_dk_total": pred, "dk_total": y, "abs_error": err,
            "rotation": groups[1][1]}))
    return rows, pd.concat(frames, ignore_index=True)


# ── The comparison, on whichever pair of frames it is handed ──────────────────

ORDER = ["full_season", "prior_gp", "league_age", "beta_binomial", "spell_process",
         "oracle_rate", "oracle_gp"]


def gate_e(table: pd.DataFrame) -> dict:
    """Gate E of `docs/games-played-plan.md`: is the spell process worth it downstream?

    The games-played arms are compared on CRPS in *games*, which is a marginal metric and
    which the plan's own Gate D showed cannot see the thing the spell process exists to
    fix — how absences are *shaped* into runs. Gate E asks the question one level down,
    where the answer is worth money: does swapping the games-played treatment improve the
    season DK total, holding the rate model and the rows identical?

    **The bars are the incumbent's own row on whichever split this is scored on**, read out
    of the table rather than written down. A gate with hard-coded thresholds is a test-set
    number in disguise — `stan_games_played._gate_d` had exactly that shape and it is how
    the games-played decision came to be settled on test.

    Returns `{"ran": False}` when `make stan-games-played` has not written a pmf, so a
    fresh checkout reports "not run" instead of a silent pass.
    """
    if "spell_process" not in set(table.index):
        return {"ran": False}
    spell, incumbent = table.loc["spell_process"], table.loc["beta_binomial"]
    out = {"ran": True,
           "mae": float(spell["mae_dk_total"]),
           "incumbent_mae": float(incumbent["mae_dk_total"]),
           "crps": float(spell.get("crps_dk_total", np.nan)),
           "incumbent_crps": float(incumbent.get("crps_dk_total", np.nan)),
           "bias": float(spell["bias_dk_total"]),
           "incumbent_bias": float(incumbent["bias_dk_total"])}
    out["mae_improves"] = bool(out["mae"] < out["incumbent_mae"])
    out["crps_improves"] = bool(out["crps"] < out["incumbent_crps"])
    out["passes"] = bool(out["mae_improves"] and out["crps_improves"])
    return out


def compare(train: pd.DataFrame, frame: pd.DataFrame, max_games: int,
            out_dir: Path | None = None, pmf_file: str = SPELL_PMF_FILE
            ) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Fit the rate model and every GP treatment on `train`, score them on `frame`.

    Split-agnostic, so the validation ladder and the one end-of-project reading are the
    same code rather than two copies of it. `run` passes `(train, validation)`;
    `src/final_evaluation.py` passes `(train + validation, test)` and the test-side pmf.
    """
    rate_model = RateModel().fit(train)
    rate = rate_model.predict(frame)
    rate_r2 = float(1 - (np.sum((rate - frame["dk_per_game"]) ** 2)
                         / np.sum((frame["dk_per_game"]
                                   - frame["dk_per_game"].mean()) ** 2)))

    treatments = gp_treatments(train, frame, max_games, out_dir, pmf_file)
    treatments["oracle_rate"] = {"gp": treatments["beta_binomial"]["gp"], "pmf": None,
                                 "rate": frame["dk_per_game"].to_numpy(dtype=float)}
    rows, predictions = evaluate(frame, rate, treatments, max_games)
    return pd.DataFrame(rows), predictions, rate_r2


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_av = cfg.get("features", {}).get("availability", {})
    test_seasons = int(cfg_av.get("test_seasons", TEST_SEASONS))

    frame = build_frame(cfg)
    train, val = selection_split(frame, test_seasons)
    max_games = int(frame["team_games"].max())

    print(f"Season-total design: {len(frame):,} player-seasons. The test split is "
          f"LOCKED — this\n  ladder CHOOSES a games-played treatment (Gate E of "
          f"docs/games-played-plan.md), so it\n  runs on VALIDATION; the deliverable-level "
          f"held-out number is taken once, by\n  `make final-evaluation`.")
    print(f"  {len(train):,} fit / {len(val):,} score "
          f"({', '.join(sorted(val['season'].unique()))} as validation)")

    metrics, predictions, rate_r2 = compare(train, val, max_games, out_dir)
    print(f"  Fixed rate model: validation R² {rate_r2:.4f} on dk_pts per game played. "
          f"Identical across\n  every treatment below, so the contrast is the games-played "
          f"model and nothing else.")

    order = ORDER
    table = (metrics[metrics.group == "all"]
             .pivot_table(index="treatment", columns="metric", values="value")
             .reindex([t for t in order if t in set(metrics.treatment)]))
    cols = [c for c in ["mae_dk_total", "rmse_dk_total", "r2_dk_total",
                        "bias_dk_total", "crps_dk_total"] if c in table.columns]
    print("\nSeason DK total on VALIDATION (MAE in dk_pts, lower is better):")
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

    gate = gate_e(table)
    print("\nGate E — the spell process on the deliverable, against the incumbent "
          "beta-binomial:")
    if not gate["ran"]:
        print("  NOT RUN: no spell-process pmf on disk. Run `make stan-games-played` "
              "first; its\n  validation-side pmf is what this treatment reads.")
    else:
        print(f"  MAE  {gate['mae']:.1f} against {gate['incumbent_mae']:.1f} "
              f"({gate['mae'] - gate['incumbent_mae']:+.1f})")
        print(f"  CRPS {gate['crps']:.1f} against {gate['incumbent_crps']:.1f} "
              f"({gate['crps'] - gate['incumbent_crps']:+.1f})")
        print(f"  bias {gate['bias']:+.1f} against {gate['incumbent_bias']:+.1f}")
        print(f"  => {'PASSES' if gate['passes'] else 'FAILS'}. Both bars are the "
              f"incumbent's own row on these\n     rows, so the gate cannot be passed by "
              f"changing what it is compared against.")

    paths = {}
    for name, (data, dest) in {
            "metrics": (metrics, out_dir / "season_total_metrics.csv"),
            "predictions": (predictions, out_dir / "season_total_predictions.csv"),
            "gate_e": (pd.DataFrame([gate]), out_dir / "season_total_gate_e.csv"),
    }.items():
        data.to_csv(dest, index=False)
        paths[name] = dest
        print(f"\nSaved {len(data):,} {name} rows → {dest}")
    return paths


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
