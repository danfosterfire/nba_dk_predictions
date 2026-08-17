"""Item 3d: does a fitted per-(player, season) effect close the composition's season gap?

`make minutes-unification` scored the minutes composition's summed draws at the **season**
unit and found them **4.68x too narrow** — predictive sd 64.65 minutes against the marginal
head's 302.75, CRPS 170.06 against 144.35, and short of the season-unit carry-forward floor
at 161.29 — on the same posterior that clears its own per-team-game floor decisively. It
then traced that to a *missing parameter* rather than to a ceiling: a league-wide season
term has **0.000000%** of the residual variance to reach (the head allocates every minute
in the league, so its residuals sum to zero within a season), but a per-(player, season)
shift is not shared, and injecting `sigma * z` per unit per posterior draw into the fitted
posterior moved the season-total sd to 239.45 and **tied** the marginal head at
sigma = 0.375.

That injection read sigma off validation CRPS, which is the split it is scored against. This
module fits it instead — `sigma_u` as a Stan parameter in `composition_glm.stan` — and
sweeps a team-context feature block alongside it, since a head whose entire job is dividing
a fixed team pot among teammates currently carries **nothing about the teammates**.

## Why this is a separate target from `make stan-composition`

Three reasons, and the first is a build gate rather than a preference.

1. **`outputs/predictions/stan_composition_*.csv` is the incumbent's record and
   `make docs-audit` re-derives eleven quoted figures from it.** A partial run — three new
   arms, no `binomial`, no `betabinom`, and at a pilot window — would overwrite those rows
   with figures that do not answer the same question, and the audit would fail on a
   *bookkeeping* change rather than a measured one.
2. **The incumbent is deliberately not refitted.** `docs/simulations-plan.md` says its
   posterior is on disk at `data/features/posteriors/<window>/composition.pkl` and is the
   comparison baseline; this ladder is three new arms plus a same-window control.
3. **The full-window commitment is a separate decision from the arm ordering.** The pilot
   window is 6.5x smaller in rows and 5.6x smaller in units, which is the path this head
   already took once from Gate A to Gate E.

The head-level capability — the `U_n` block in `composition_glm.stan`, `PlayerSeasonTerm`,
`effect_variants` and the team join — all lives in `src/models/stan_composition.py`, where
it belongs. This module is the driver and the gate.

## What it measures, and at which unit

**Both units, in one run, because the head's whole lesson is that a verdict belongs to a
unit.** Per team-game (Gate P3, against the incumbent's 4.4945) and per player-season
(Gate P2, against the persisted marginal head on the rows both cover). The marginal head is
read from its `train`-window artifact and never refitted, so the season-unit comparison here
is the same comparison `make minutes-unification` takes, on the same validation rows.

Selection reads validation only, through `held_out.selection_split`.

## The arms, and the one that is not a modelling arm

`base` `ps` `ps_team` `team` are the ladder. `stan_composition.effect_variants` also offers
**`ps_centered`**, which is `ps` in different coordinates — a *sampler* arm, and it exists
because Gate A found the non-centred fit taking 12.0x the shipped arm's wall clock with a
step size of 0.0094 and 17 treedepth-saturated draws. It must never be selected against `ps`
on CRPS: the two have the same posterior, so any difference between them is Monte Carlo
error or a convergence failure, and the only thing worth comparing is the cost and the
diagnostics.

**`mq` and `mq_graded` are the third representation** (`docs/composition-quadrature-plan.md`,
opened 2026-08-16): the same effect with each unit's latent INTEGRATED OUT by Gauss-Hermite
quadrature inside the model block, so the posterior never contains one. `mq` bears the same
relation to `ps` that `ps_centered` does — same posterior, different route, and a
disagreement between them is a bug rather than a result, which `_report_gates` checks
explicitly before any metric is worth reading. What it buys is what `ps` has twice failed to
deliver: a *converged* fit, and a parameter block (~35) that does not grow with the window,
which is the only way the full-window sigma — the one that would actually ship — gets
estimated. `mq_graded` is the one genuinely new model here: sigma per `rho_bin` rather than
shared, motivated by the injection's own per-role calibration finding a ~2x gradient in what
sigma wants across roles.

Usage:
    python -m src.models.composition_effects
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.held_out import selection_split
from src.models.stan_composition import (GH_INFLATE, GROUP_KEYS, PILOT_FIRST_SEASON,
                                         Q_NODES, TEAM_COLS, TEST_SEASONS, UNIT_KEYS,
                                         FloorComposition, StanComposition,
                                         announce_metric, composition_frame,
                                         effect_variants, score_samples, team_context)
from src.models.stan_utils import diagnostics_frame

ARMS = ("base", "ps", "ps_team", "team")

# The arms that carry the effect at all, by either representation — what P2/P3/P5 are read
# over. `mq`/`mq_graded` marginalize it; `ps`/`ps_team` sample it.
EFFECT_ARMS = ("ps", "ps_team", "ps_centered", "mq", "mq_graded")

# The incumbent's published per-team-game CRPS, which Gate P3 forbids regressing past. A
# constant rather than a re-read, because the incumbent is not refitted here and the figure
# is the one `outputs/predictions/stan_composition_metrics.csv` carries.
INCUMBENT_CRPS = 4.4945

# The injection's CRPS-optimal sigma and the calibration-optimal one, from
# `outputs/predictions/minutes_unification.csv`. Gate P5 asks the fit to land near them.
INJECTED_SIGMA = (0.375, 0.45)

SEED = 0


def artifact_stem(label: str) -> str:
    """`composition_effects[_label]` — a BUILD GATE rather than tidiness.

    `make docs-audit` re-derives the pilot `base` arm's per-team-game CRPS (**4.45614**) from
    `outputs/predictions/composition_effects_metrics.csv`, and `_flush` merges by arm name —
    so a later round writing `base` there under a different offset, window or ladder would
    answer a different question under that figure's name and the audit would fail on a
    bookkeeping change. It is the same guard `composition_preseason_fit.artifact_stem`
    carries one module over, and for the same reason: the round that measured a number and
    the round that reuses its name must not be able to collide.

    An empty label restores the original paths exactly, which is what keeps re-running the
    2026-08-09 ladder a config edit rather than a code edit.
    """
    return "composition_effects" + (f"_{label}" if label else "")

# Gate A's correction factor. This head's own Gate A read 12.8 h against an actual 20.9 h at
# the full window — a 1.63x miss — because per-row cost is superlinear in rows: more data
# sharpens the posterior, shrinks the step size and buys more leapfrog steps per iteration.
# `stan_games_played` keeps the same constant deliberately, as a conservative factor rather
# than a current estimate.
PROBE_MISS = 1.63

# Realized minutes below which a logit share says nothing — `stan_minutes.MIN_PRIOR_MINUTES`
# reused rather than re-chosen, as `minutes_unification` already does.
MIN_QUALIFIED = 200


# ── The deviation the whole item is about ─────────────────────────────────────

def primary_team(frame: pd.DataFrame) -> pd.DataFrame:
    """One team per (player, season): where he played the most minutes.

    Needed because roster churn is a team-level quantity and 13.6% of players appear for
    two teams in a season. Attributing a traded player to both teams would double-count him
    in every departure and arrival aggregate.
    """
    by_team = (frame.groupby(UNIT_KEYS + ["team_id"], as_index=False)["y"].sum()
               .sort_values(UNIT_KEYS + ["y"], ascending=[True, True, False]))
    return by_team.drop_duplicates(UNIT_KEYS)[UNIT_KEYS + ["team_id"]]


def roster_churn(units: pd.DataFrame, seasons: list[str]) -> pd.DataFrame:
    """Departed / arrived / net prior-season minutes share per (player, season).

    For player `i` on team `T` in season `S`: `departed` sums the S-1 shares of players who
    were on `T` in S-1 and are not in S, `arrived` sums the S-1 shares of players on `T` in
    S who were not in S-1, and `net_opened` is their difference. All three are S-1
    quantities over the S roster, which is the project's information set exactly — the
    season-S roster is known before the season and the shares are last year's.

    `i` is excluded from neither aggregate because he is in neither set by construction: he
    is on `T` in both seasons whenever the lag exists at all.
    """
    order = {s: i for i, s in enumerate(seasons)}
    units = units.assign(_idx=units["season"].map(order))

    rows = []
    for (season, team), block in units.groupby(["season", "team_id"], sort=False):
        idx = int(block["_idx"].iloc[0])
        if idx == 0:
            continue
        last = seasons[idx - 1]
        was = units[(units["season"] == last) & (units["team_id"] == team)]
        now_ids, was_ids = set(block["player_id"]), set(was["player_id"])
        # Departed players' share is their share in the season they left, which for a
        # season-(S-1) roster row IS `minutes_share` — the realized one, not the lag.
        departed = float(was.loc[~was["player_id"].isin(now_ids), "minutes_share"].sum())
        newcomers = block.loc[~block["player_id"].isin(was_ids)]
        arrived = float(newcomers["prior_share"].fillna(0.0).sum())
        rows.append({"season": season, "team_id": team, "departed_share": departed,
                     "arrived_share": arrived, "net_opened": departed - arrived})
    churn = pd.DataFrame(rows)
    return units.merge(churn, on=["season", "team_id"], how="left")


def deviation_signals(frame: pd.DataFrame, block: pd.DataFrame, seasons: list[str],
                      cols: list[str] = TEAM_COLS) -> pd.DataFrame:
    """One row per (player, season): the deviation and everything meant to predict it.

    The deviation is `logit(realized minutes share) - logit(prior share)`, on the fitting
    half only. It is the exact quantity the player-season effect is a parameter for, so
    "what is the team block worth" is answerable without fitting the head at all.
    """
    units = (frame.groupby(UNIT_KEYS, as_index=False)
             .agg(minutes=("min", "sum"), length=("game_length", "sum"),
                  prior_share=("w_share", "first"), no_prior=("no_prior", "first")))
    units["minutes_share"] = units["minutes"] / units["length"].clip(lower=1e-9)
    units = units.merge(primary_team(frame), on=UNIT_KEYS)

    def logit(x: np.ndarray) -> np.ndarray:
        x = np.clip(np.asarray(x, dtype=float), 1e-4, 1 - 1e-4)
        return np.log(x / (1 - x))

    units["deviation"] = logit(units["minutes_share"]) - logit(units["prior_share"])
    units = roster_churn(units, seasons)

    order = {s: i for i, s in enumerate(seasons)}
    lag = units[UNIT_KEYS + ["deviation"]].copy()
    lag["_idx"] = lag["season"].map(order) + 1
    lag = lag.assign(season=lag["_idx"].map({i: s for i, s in enumerate(seasons)}))
    units = units.merge(lag[UNIT_KEYS + ["deviation"]].rename(
        columns={"deviation": "deviation_lag1"}), on=UNIT_KEYS, how="left")

    units = units.merge(block, on=UNIT_KEYS, how="left")
    keep = (units["no_prior"] == 0) & (units["minutes"] >= MIN_QUALIFIED)
    return units[keep].reset_index(drop=True)


def _r2(y: np.ndarray, X: np.ndarray) -> float:
    """In-sample R² of an OLS fit with an intercept, NaN rows dropped pairwise-complete."""
    good = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[good], X[good]
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ beta
    return float(1 - resid.var() / y.var())


def deviation_table(signals: pd.DataFrame, cols: list[str] = TEAM_COLS) -> pd.DataFrame:
    """The four scratch correlations the plan doc quotes, re-derived into an artifact.

    `docs/simulations-plan.md` records them as prose from a scratch session — own lag-1
    deviation **-0.201**, departed **+0.041**, arrived **-0.063**, net opened **+0.088**,
    with in-sample R² **0.040 -> 0.052**. This is the same measurement inside the run, on
    the fitting half, so the figures stop being unreproducible.

    **Expect the block to be worth about a point of R², and record a null as a null.** The
    deviation is only ~5% predictable from pre-season information at all — the same wall the
    whole project runs into, where availability persists at r = 0.317 and five games of the
    real season settle 86% of the season total. That is the finding, not a disappointment.
    """
    y = signals["deviation"].to_numpy(float)
    rows = [{"analysis": "correlation", "signal": name, "n": int(np.isfinite(v).sum()),
             "value": float(pd.Series(y).corr(pd.Series(v)))}
            for name, v in (("own_deviation_lag1",
                             signals["deviation_lag1"].to_numpy(float)),
                            ("departed_share", signals["departed_share"].to_numpy(float)),
                            ("arrived_share", signals["arrived_share"].to_numpy(float)),
                            ("net_opened", signals["net_opened"].to_numpy(float)))]

    own = signals[["deviation_lag1"]].to_numpy(float)
    churn = signals[["departed_share", "arrived_share", "net_opened"]].to_numpy(float)
    team = signals[cols].to_numpy(float)
    for name, X in (("own_history", own),
                    ("own_plus_churn", np.column_stack([own, churn])),
                    ("own_plus_churn_plus_team_block",
                     np.column_stack([own, churn, team])),
                    ("team_block_alone", team)):
        rows.append({"analysis": "in_sample_r2", "signal": name, "n": len(signals),
                     "value": _r2(y, X)})
    rows.append({"analysis": "spread", "signal": "deviation_sd", "n": len(signals),
                 "value": float(np.nanstd(y))})
    return pd.DataFrame(rows)


# ── The season unit — Gate P2, against the persisted marginal head ────────────

def marginal_head_season(cfg: dict, keep: int):
    """The marginal head's validation season totals, from its `train` artifact.

    Never refitted, and never the composition's own window: `stan_minutes` ships fitted on
    `train`, `make posteriors` persisted exactly that, and this is the identical object
    `make minutes-unification` scores. Reusing it is what makes the pilot's season-unit
    reading comparable to the gate's.
    """
    from src.models.minutes_unification import (FIT_WINDOW, rehydrate_minutes,
                                                validation_frames)
    from src.models.posteriors import load, posteriors_dir, require_window

    window = str(cfg.get("sim", {}).get("fit_window", FIT_WINDOW))
    artifact = load("minutes", posteriors_dir(cfg, window))
    require_window({"minutes": artifact}, window)
    _, minutes_val, _, _ = validation_frames(cfg)
    model = rehydrate_minutes(artifact, keep)
    totals = model.predict_samples(artifact.recipe.transform(minutes_val), SEED)
    units = minutes_val[UNIT_KEYS].copy()
    units["realized"] = minutes_val["successes"].to_numpy(float)
    return totals, units, window


def season_arm(samples: np.ndarray, val: pd.DataFrame, mins_totals: np.ndarray,
               mins_units: pd.DataFrame, label: str) -> dict:
    """One arm's season-unit row: its own metrics plus the paired gap to the marginal head."""
    from src.models.minutes_unification import (paired_bootstrap, realized_totals,
                                                score_season, season_totals, verdict)
    from src.models.stan_utils import crps_from_samples

    totals, units = season_totals(samples, val)
    units = units.merge(realized_totals(val, "y"), on=UNIT_KEYS)
    common = mins_units.merge(units[UNIT_KEYS + ["realized"]], on=UNIT_KEYS, how="inner",
                              suffixes=("_minutes", "_composition"))
    pos_c = {k: i for i, k in enumerate(map(tuple, units[UNIT_KEYS].to_numpy()))}
    pos_m = {k: i for i, k in enumerate(map(tuple, mins_units[UNIT_KEYS].to_numpy()))}
    keys = list(map(tuple, common[UNIT_KEYS].to_numpy()))
    idx_c = np.array([pos_c[k] for k in keys])
    idx_m = np.array([pos_m[k] for k in keys])

    y_c = common["realized_composition"].to_numpy(float)
    y_m = common["realized_minutes"].to_numpy(float)
    row = score_season(totals[:, idx_c], y_c, label, "season_total")
    delta = paired_bootstrap(crps_from_samples(totals[:, idx_c], y_c),
                             crps_from_samples(mins_totals[:, idx_m], y_m))
    row.update({k: delta[k] for k in ("crps_delta", "ci_lo", "ci_hi", "p_delta_negative")})
    row["verdict_vs_marginal"] = verdict(delta)
    return row


# ── Gate A, on the arm that carries the cost risk ─────────────────────────────

def probe_effects(train: pd.DataFrame, built: dict, cfg_stan: dict, iters: dict,
                  max_hours: float, arms: tuple[str, ...]) -> dict:
    """One-season `ps` fit, extrapolated to the sweep, aborting loudly past the budget.

    `stan_composition.probe_timing` probes `betabinom` — the ~25-parameter arm under
    `dense_e` — which is exactly the wrong thing to size here. The whole cost risk of this
    item is the random effect: `dense_e` is forced off, and the parameter count goes from
    ~25 to one per player-season unit. So the probe fits the `ps` arm, and the extrapolation
    scales by rows **and** by units, because a random-effect fit's cost is not linear in
    rows alone.

    **The extrapolation is a lower bound and should be read as one.** That head's own
    Gate A read 12.8 h against an actual 20.9 h at the full window, a 1.63x miss, because
    per-row cost is superlinear in rows: more data sharpens the posterior, shrinks the step
    size and buys more leapfrog steps. Nothing here fixes that; the factor is applied
    explicitly instead of pretending to a linear law.
    """
    last = sorted(train["season"].unique())[-1]
    # Whichever random-effect arm the run actually plans to fit, not `ps` by name: a ladder
    # that swapped in `ps_centered` would otherwise be sized by the parameterization it is
    # not using, which is the one thing this probe exists to measure.
    probe_arm = next((a for a in arms
                      if built[a].player_season_effect or built[a].quadrature), "ps")
    arm = built[probe_arm]
    tr = arm.train[arm.train["season"] == last].reset_index(drop=True)

    model = StanComposition(arm.features, arm.dispersed, arm.n_rho,
                            name=f"probe/{probe_arm}-one-season",
                            chains=int(cfg_stan.get("chains", 4)),
                            seed=int(cfg_stan.get("seed", 42)),
                            warmup=int(cfg_stan.get("probe_warmup", 200)),
                            samples=int(cfg_stan.get("probe_samples", 200)),
                            player_season_effect=arm.player_season_effect,
                            u_centered=arm.centered, quadrature=arm.quadrature,
                            q_nodes=int(cfg_stan.get("composition", {}).get("effects", {})
                                        .get("quadrature", {}).get("nodes", Q_NODES)),
                            gh_inflate=float(cfg_stan.get("composition", {})
                                             .get("effects", {}).get("quadrature", {})
                                             .get("inflate", GH_INFLATE)),
                            n_sigma=arm.n_sigma).fit(tr)
    seconds = float(model.diagnostics["wall_clock_s"])

    probe_units = tr.groupby(UNIT_KEYS, sort=False).ngroups
    full_units = train.groupby(UNIT_KEYS, sort=False).ngroups
    row_scale = len(train) / max(len(tr), 1)
    unit_scale = full_units / max(probe_units, 1)
    iter_scale = ((iters["warmup"] + iters["samples"])
                  / (int(cfg_stan.get("probe_warmup", 200))
                     + int(cfg_stan.get("probe_samples", 200))))
    # Rows drive the likelihood, units drive the parameter block; the geometric mean of the
    # two is the honest middle between "cost is linear in rows" (which ignores 12,307 new
    # parameters) and "linear in parameters" (which ignores a 631k-row likelihood).
    #
    # **The marginal arm is scaled by ROWS ALONE, and that is the whole point of it.** Its
    # parameter block is ~35 wide at every window, so units buy it no cost at all and folding
    # a 5.6x unit scale into its extrapolation would charge it for the exact thing it was
    # built to stop paying. Getting this wrong would price `mq` at the full window as if it
    # were `ps`, which is the comparison the item exists to settle.
    scale = (row_scale if arm.quadrature else float(np.sqrt(row_scale * unit_scale))
             ) * iter_scale
    n_effect_arms = sum(bool(built[a].player_season_effect or built[a].quadrature)
                        for a in arms)
    n_plain_arms = len(arms) - n_effect_arms
    # The plain arms keep `dense_e` and are the cheap ones; charge them at a quarter of a
    # random-effect arm rather than at zero.
    hours = seconds * scale * (n_effect_arms + 0.25 * n_plain_arms) * PROBE_MISS / 3600
    print(f"  Gate A: `{probe_arm}` probe fit {len(tr):,} rows / {probe_units:,} units "
          f"({last}) in {seconds:.0f}s at {model.diagnostics['metric']}, "
          f"{model.diagnostics['parameterization']}\n"
          f"    -> sweep extrapolates to {hours:.1f}h "
          f"(rows x{row_scale:.1f}, units x{unit_scale:.1f}"
          f"{' — NOT charged, the marginal arm is ~35 params at any window'
             if arm.quadrature else ''}, iters x{iter_scale:.1f}, "
          f"x{PROBE_MISS:.2f} for the measured under-prediction)")
    if hours > max_hours:
        raise RuntimeError(
            f"Gate A: extrapolated sweep {hours:.1f}h exceeds the {max_hours:.0f}h budget. "
            f"Fallbacks, in order: drop `team`/`ps_team` from `stan.composition.effects."
            f"arms`, shorten the chains, move `stan.composition.effects.first_season` later "
            f"— and if none of those fit, take the fallback in docs/simulations-plan.md and "
            f"ship the injection with sigma estimated on `train`.")
    return {"probe_season": last, "probe_rows": len(tr), "probe_units": probe_units,
            "probe_seconds": seconds, "extrapolated_hours": float(hours),
            "diagnostics": model.diagnostics}


# ── The sweep ─────────────────────────────────────────────────────────────────

def _flush(dest: Path, row: dict, key: str) -> None:
    """Merge one completed arm over whatever is on disk, keyed by arm name.

    **Merge rather than truncate-then-append**, which is the rule `posteriors.flush`
    already follows for its manifest and for the same reason: a random-effect arm is hours
    and the arms are independent, so re-running a subset — `arms: [ps]` after `base` has
    already landed — must leave the other rows in place. Truncating at the start of `run`
    made a partial re-run silently destroy the arms it was not fitting, which is exactly
    the composability the per-arm checkpointing exists to provide.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    built = pd.DataFrame([row])
    if dest.exists():
        existing = pd.read_csv(dest)
        if len(existing) and key in existing.columns:
            built = pd.concat([existing[existing[key] != row[key]], built],
                              ignore_index=True)
    built.to_csv(dest, index=False)


def _flush_frame(dest: Path, frame: pd.DataFrame) -> None:
    """Append-only, for the diagnostics log where every fit is its own row."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, mode="a", header=not dest.exists(), index=False)


def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(cfg["data"]["features_dir"])
    seasons = list(cfg["data"]["seasons"])
    cfg_stan = cfg.get("stan", {})
    comp_cfg = cfg_stan.get("composition", {})
    eff = comp_cfg.get("effects", {})
    first_season = str(eff.get("first_season", PILOT_FIRST_SEASON))
    arms = tuple(eff.get("arms", ARMS))
    keep = int(comp_cfg.get("predictive_samples", 200))
    iters = {"warmup": int(eff.get("warmup", cfg_stan.get("select_warmup", 500))),
             "samples": int(eff.get("samples", cfg_stan.get("select_samples", 500)))}
    max_hours = float(eff.get("max_extrapolated_hours", 12.0))
    seed = int(cfg_stan.get("seed", 42))
    stem = artifact_stem(str(eff.get("label", "") or ""))

    print("Composition effects — the per-(player, season) random effect and the team block")
    print(f"  window: {first_season} on; arms: {', '.join(arms)}; "
          f"{iters['warmup']}+{iters['samples']} x {cfg_stan.get('chains', 4)} chains")
    print(f"  writing {stem}_*.csv → {out_dir}")

    frame = composition_frame(cfg)
    windowed = frame[frame["season"] >= first_season].reset_index(drop=True)
    train, val = selection_split(windowed, TEST_SEASONS)
    print("  The test split is LOCKED — this sweep fits and scores VALIDATION only "
          "(src/models/held_out.py).")
    for name, part in (("train", train), ("val", val)):
        print(f"  {name}: {len(part):,} rows / "
              f"{part.groupby(GROUP_KEYS, sort=False).ngroups:,} team-games / "
              f"{part.groupby(UNIT_KEYS, sort=False).ngroups:,} player-season units "
              f"({', '.join(sorted(part['season'].unique()))})")

    block = team_context(features_dir)

    # ── The deviation, before any fitting ─────────────────────────────────────
    # Measured on the FULL window's fitting half rather than this run's, deliberately: the
    # deviation is a property of the data and not of whichever window the head is fitted
    # on, and the plan's scratch figures are full-window. Restricting it to the pilot's
    # four training seasons would cut it to ~1,500 player-seasons and make the
    # re-derivation incomparable to the prose it exists to replace.
    dev_train, _ = selection_split(frame, TEST_SEASONS)
    signals = deviation_signals(dev_train, block, seasons)
    dev = deviation_table(signals)
    dev["window"] = "train"
    print(f"\nThe deviation the effect is a parameter for — logit(realized share) - "
          f"logit(prior share),\n  {len(signals):,} train player-seasons over the full "
          f"window, sd "
          f"{float(dev.loc[dev['signal'] == 'deviation_sd', 'value'].iloc[0]):.4f}:")
    print(dev.round(4).to_string(index=False))
    # Written before anything samples. It needs no fit at all, it is the artifact that
    # replaces the plan doc's scratch prose, and a run aborted by Gate A must still leave it.
    dev_dest = out_dir / f"{stem}_deviation.csv"
    dev.to_csv(dev_dest, index=False)
    print(f"Saved {len(dev):,} deviation rows → {dev_dest}")

    built, coverage = effect_variants(train, val, block)
    print(f"\n  team block: {coverage['team_share_train']:.1%} of train rows covered "
          f"({coverage['team_share_val']:.1%} of val).\n"
          f"  {coverage['team_missing_with_design_present']:.1%} of train rows are missing "
          f"the team block but DO have an\n  availability design, which is why "
          f"`team_missing` ships as its own indicator rather than\n  reusing "
          f"`design_missing` "
          f"({coverage['team_missing_and_design_missing']:.1%} of rows carry both).")

    diag_dest = out_dir / f"{stem}_diagnostics.csv"
    diag_dest.unlink(missing_ok=True)
    probe = probe_effects(train, built, cfg_stan, iters, max_hours, arms)
    # Flushed the moment Gate A returns, not at the end of a run that is hours long. The
    # probe row IS the cost measurement — it is what decides whether the full window is
    # affordable — so it must survive the run being cut short, which is the same reason
    # the deviation table is written before any sampling.
    _flush_frame(diag_dest, diagnostics_frame([probe["diagnostics"]]))

    floor = score_samples(FloorComposition(keep).fit(train).predict_samples(val, seed),
                          val, "carry_forward", seed)
    mins_totals, mins_units, window = marginal_head_season(cfg, keep)
    print(f"\n  season-unit comparator: the persisted `{window}` marginal head, "
          f"{len(mins_units):,} player-seasons. Nothing is refitted.")

    # NOT unlinked: `_flush` merges by arm, so re-running a subset leaves the arms it is
    # not fitting in place. A random-effect arm is hours and the arms are independent.
    game_dest = out_dir / f"{stem}_metrics.csv"
    season_dest = out_dir / f"{stem}_season.csv"

    quad_cfg = eff.get("quadrature", {})
    q_nodes = int(quad_cfg.get("nodes", Q_NODES))
    gh_inflate = float(quad_cfg.get("inflate", GH_INFLATE))

    rows, season_rows, diagnostics = [], [], []
    for arm in arms:
        spec = built[arm]
        tr, te, feats, n_rho = spec.train, spec.val, spec.features, spec.n_rho
        how = ("MARGINALIZED by quadrature" if spec.quadrature
               else "sampled" if spec.player_season_effect else "off")
        print(f"\n  fitting `{arm}` — {len(feats)} features, player-season effect {how}"
              f"{' (centred)' if spec.centered else ''}"
              f"{f', Q={q_nodes}, {spec.n_sigma} sigma bin(s)' if spec.quadrature else ''}")
        # Before the sampler, not after it. This block sat one warmup draw under the
        # `dense_e` cliff for four days and nothing said so — the metric only reaches the
        # artifact after the fit whose cost it decides.
        announce_metric(len(feats), n_rho, iters["warmup"],
                        (tr.groupby(UNIT_KEYS, sort=False).ngroups
                         if spec.player_season_effect else 0),
                        spec.n_sigma if spec.quadrature else 0)
        started = time.perf_counter()
        model = StanComposition(feats, spec.dispersed, n_rho, name=f"effects/{arm}",
                                chains=int(cfg_stan.get("chains", 4)), seed=seed,
                                predictive_samples=keep,
                                player_season_effect=spec.player_season_effect,
                                u_sd_scale=float(eff.get("u_sd_scale", 1.0)),
                                u_centered=spec.centered, quadrature=spec.quadrature,
                                q_nodes=q_nodes, gh_inflate=gh_inflate,
                                n_sigma=spec.n_sigma, **iters).fit(tr)
        diagnostics.append(model.diagnostics)
        _flush_frame(diag_dest, diagnostics_frame([model.diagnostics]))
        samples = model.predict_samples(te, seed)
        row = {**score_samples(samples, te, arm, seed), **model.ps.summary(),
               "n_features": len(feats), "metric": model.diagnostics["metric"],
               "parameterization": model.diagnostics["parameterization"],
               "wall_clock_s": float(time.perf_counter() - started)}
        rows.append(row)
        _flush(game_dest, row, "variant")

        srow = season_arm(samples, te, mins_totals, mins_units, arm)
        srow.update({k: row[k] for k in ("sigma_u", "sigma_u_lo", "sigma_u_hi")})
        season_rows.append(srow)
        _flush(season_dest, srow, "arm")
        print(f"    per-team-game CRPS {row['crps_minutes']:.4f} "
              f"(incumbent {INCUMBENT_CRPS:.4f}), team-sum error "
              f"{row['team_sum_abs_error']:.2f}; season CRPS "
              f"{srow['crps_minutes']:.2f} vs marginal {srow['crps_delta']:+.2f} "
              f"[{srow['ci_lo']:+.2f}, {srow['ci_hi']:+.2f}] → "
              f"{srow['verdict_vs_marginal']}; sd {srow['predictive_sd']:.1f}; "
              f"sigma_u {row['sigma_u']:.4f}")

    table = pd.DataFrame(rows)
    season = pd.DataFrame(season_rows)
    table["floor_crps"] = floor["crps_minutes"]
    table["incumbent_crps"] = INCUMBENT_CRPS
    table["first_season"] = first_season
    season["first_season"] = first_season

    print("\nPer team-game (Gate P3 — the incumbent's win must not be traded away):")
    print(table[["variant", "n_features", "crps_minutes", "r2_minutes", "pit_ks",
                 "team_sum_abs_error", "sigma_u", "metric", "parameterization",
                 "wall_clock_s"]]
          .round(4).to_string(index=False))
    print(f"  floor {floor['crps_minutes']:.4f}, incumbent (full window, not refitted) "
          f"{INCUMBENT_CRPS:.4f}")

    print("\nPer player-season (Gate P2 — the gate this whole line of work exists to move):")
    print(season[["arm", "n", "crps_minutes", "mae_minutes", "bias_minutes", "pit_ks",
                  "predictive_sd", "crps_delta", "ci_lo", "ci_hi",
                  "verdict_vs_marginal"]].round(4).to_string(index=False))

    _report_gates(table, season)

    diag = diagnostics_frame([probe["diagnostics"]] + diagnostics)
    # `_flush` has already merged each arm as it landed, so the incremental files are the
    # authority; re-writing `table`/`season` wholesale here would drop arms from earlier
    # runs. Re-read them instead.
    artifacts = {"metrics": (pd.read_csv(game_dest), game_dest),
                 "season": (pd.read_csv(season_dest), season_dest),
                 "deviation": (dev, dev_dest), "diagnostics": (diag, diag_dest)}
    paths = {}
    for name, (df, dest) in artifacts.items():
        df.to_csv(dest, index=False)
        paths[name] = dest
        print(f"Saved {len(df):,} {name} rows → {dest}")
    print(f"\nSampler: max R-hat {diag['max_rhat'].max():.4f}, "
          f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
          f"{diag['wall_clock_s'].sum() / 3600:.2f} h total")
    return paths


def _report_gates(table: pd.DataFrame, season: pd.DataFrame) -> None:
    """P2-P5 stated as code, so the outcome is not a judgement made after seeing them."""
    by_arm = table.set_index("variant")
    by_season = season.set_index("arm")
    print("\nGates:")
    for arm in table["variant"]:
        if arm in by_season.index:
            v = by_season.loc[arm]
            ok2 = v["verdict_vs_marginal"] in ("wins", "ties")
            print(f"  P2 {arm:>8}: season unit {v['verdict_vs_marginal']:>5} "
                  f"({'PASS' if ok2 else 'FAIL'}); team-sum error "
                  f"{by_arm.loc[arm, 'team_sum_abs_error']:.2f} "
                  f"({'PASS' if by_arm.loc[arm, 'team_sum_abs_error'] == 0 else 'FAIL'})")
        crps = by_arm.loc[arm, "crps_minutes"]
        print(f"  P3 {arm:>8}: per-team-game CRPS {crps:.4f} against "
              f"{INCUMBENT_CRPS:.4f} "
              f"({'PASS' if crps <= INCUMBENT_CRPS else 'FAIL'})")
    if {"ps", "ps_team"} <= set(by_arm.index):
        a, b = float(by_arm.loc["ps", "sigma_u"]), float(by_arm.loc["ps_team", "sigma_u"])
        print(f"  P4 ps_team: sigma_u {b:.4f} against ps's {a:.4f} "
              f"({'PASS — the block explains part of the deviation' if b < a else 'FAIL — a block that improves CRPS without shrinking sigma_u is explaining something else'})")
    for arm in EFFECT_ARMS:
        if arm in by_arm.index and bool(by_arm.loc[arm, "ps_effect"]):
            s = float(by_arm.loc[arm, "sigma_u"])
            near = INJECTED_SIGMA[0] * 0.5 <= s <= INJECTED_SIGMA[1] * 2.0
            bins = sorted(c for c in by_arm.columns if c.startswith("sigma_u_bin"))
            graded = ("" if not bins else "  per bin: " + ", ".join(
                f"{by_arm.loc[arm, c]:.4f}" for c in bins))
            print(f"  P5 {arm:>10}: sigma_u {s:.4f} against the injection's "
                  f"{INJECTED_SIGMA[0]}-{INJECTED_SIGMA[1]} "
                  f"({'PASS' if near else 'CHECK — a wide disagreement with a measurement on the same rows is a bug until explained'})"
                  f"{graded}")
    # The correctness check the whole representation rests on: `mq` and `ps` are the SAME
    # posterior by two routes, so a disagreement is a bug and not a finding.
    if {"ps", "mq"} <= set(by_arm.index):
        a, b = float(by_arm.loc["ps", "sigma_u"]), float(by_arm.loc["mq", "sigma_u"])
        agree = abs(a - b) <= 0.05 * max(a, b)
        print(f"  AGREEMENT ps/mq: sigma_u {a:.4f} sampled against {b:.4f} marginalized "
              f"({'PASS' if agree else 'FAIL — two representations of one posterior must agree; do not read the metrics'})")


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
