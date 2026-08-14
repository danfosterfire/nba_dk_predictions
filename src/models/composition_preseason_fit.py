"""Session 4b's fit: does the blended offset survive the posterior?

`make composition-preseason-fit` → `outputs/predictions/composition_preseason_fit.csv`.
Two pilot-window `stan-composition` fits, ~15 min each.

## What this closes and what it does not

`make composition-preseason` measured the blend on the head's **no-fit floor**, which sets
`eta = 0` and so isolates the two routes `w_share` takes that no coefficient can reach. It
passed decisively at the head's own selection unit — **−0.19972 [−0.21661, −0.18225]** CRPS
minutes per player-game on the draft pool — and it stated its own limit in one sentence:
*"the floor is a screen, not a substitute: `beta` can correct an offset the floor cannot, so
the increment could shrink under a fitted head."* That is the question here, and it has a
precedent on each side. P3's own increment **grew** when integrated over `beta` (−4.789 at
the point MLE, −5.911 under the posterior); the composition's per-team-game floor gap, by
contrast, is exactly the kind of level a fitted `beta` is good at absorbing.

**It does not settle the full window.** Everything here is 2018-19 onward, which is what P3
decision 3 specified and what `potential-to-dos.md` item 1 measured at ~6× cheaper than the
shipped head's 9.92 h. Nor does it price the arm in the contest, which is P5.

## The bar, stated before the run

The blended arm beats the **same-window control** at the **per-player-game** unit on the
**draftable** population, with a paired-bootstrap interval clear of zero, and the team
constraint stays exact (`team_sum_abs_error == 0`). That is the unit `stan_composition`
selects on and the population P1 decision 5 fixed, so the bar is the screen's own bar read
one layer up.

Reported beside it rather than barred: the season unit (the screen found a tie there and a
better offset *cannot* manufacture season-level spread — that is what
`sim.minutes.player_season_sigma` exists for), and the **retention** — the fitted increment
as a fraction of the floor increment measured in this same run.

## Why the floors are re-measured here

Because the screen ran at 120 predictive draws and this runs at the head's own 200, and
because its floor was fitted on a frame cut the same way but scored by a different process.
Quoting "the fitted increment is X% of the floor increment" across two artifacts would put
the draw budget inside the ratio. Both floors are therefore refitted on these exact frames
at this exact draw count, and the retention is a **within-artifact** quantity.

## Why this is not `make stan-composition` and not `make composition-preseason`

The first reason is the one `composition_effects` already carries and it is a build gate:
`outputs/predictions/stan_composition_*.csv` is the incumbent's record and `make docs-audit`
re-derives eleven quoted figures from it, so a two-arm pilot-window run must not write there.
The second is cost — `composition_preseason` is 48 s of numpy with no CmdStan, and that
property is the whole design of the screen. Folding a half-hour sampler run into it would
cost the screen its re-runnability for no gain.

## The arms

| arm | `w_share` on the offset | allocation order | variant |
|---|---|---|---|
| `base` | incumbent | incumbent | `betabinom_ot_graded` |
| `preseason` | blended at `k = 80` | **incumbent** | `betabinom_ot_graded` |

`base` is a **same-window control**, never the full-window incumbent —
`composition_effects`' rule, for the reason `stan_game_length`'s `season_trend_covered`
carries: a pilot-window arm ordering read against a full-window baseline is uninterpretable.

`route = offset_only` is session 4b decision 2: the ordering route carried **3.75%** of the
screen's margin and is the expensive half to change, since it permutes the sequential
decomposition and therefore the whole likelihood's block structure. `k = 80` is decision 3,
off an inner carve of the fitting half; validation's own optimum is 160 and reading it would
be selecting on the split the arm is scored against.

⚠️ **In a fitted arm the blend reaches two places the floor could not show.** The floor's
`eta = 0` switched off the feature route, and its dispersion is a single shared `rho`. Here
`OWN = logit_share_lag1` is `logit` of the *blended* share, and `RHO_BIN_COL` is `w_share`,
so the graded dispersion's bin edges are quantiles of the blended column too. Both are the
right behaviour — the bins grade on whatever column orders the sequence and sets the offset —
but it means the fitted increment is not the floor increment plus a coefficient, and a
retention above 1.0 has somewhere to come from.

Usage:
    python -m src.models.composition_preseason_fit
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.composition_preseason import (INCUMBENT_K, crps_series, frame_at, paired,
                                              preseason_share, season_start_roster,
                                              season_totals)
from src.models.held_out import selection_split
from src.models.stan_composition import (DENSE_DRAWS_PER_PARAM, GROUP_KEYS,
                                         PILOT_FIRST_SEASON, TEST_SEASONS,
                                         FloorComposition, StanComposition, choose_metric,
                                         score_samples, variants)
from src.models.stan_utils import (crps_from_samples, diagnostics_frame, ks_uniform,
                                   pit_from_samples)

#: The shipped variant, and the only one fitted here. The ladder that chose it
#: (`stan_composition.FITTED_VARIANTS`) is not re-run: this round asks what one input change
#: is worth to the head that ships, not which head ships.
BASE_VARIANT = "betabinom_ot_graded"

#: Session 4b decision 3 — selected on an inner carve of the FITTING half.
SELECTED_K = 80.0

#: Session 4b decision 2 — the offset carries 103% of the screen's margin and the ordering
#: 3.75%, so only the offset is fitted.
ROUTE = "offset_only"

ARMS = ("base", "preseason")

#: Draws for the predictive and for the bootstrap that reads it.
SEED = 0
N_BOOTSTRAP = 2000


def build_arms(cfg: dict, pre: pd.DataFrame, first_season: str, k: float, route: str,
               test_seasons: int = TEST_SEASONS) -> dict:
    """`{arm: (train, val, features, dispersed, n_rho)}` — one frame built per arm.

    Each arm goes through `variants` on **its own** frame, so its scaler, its imputation
    means and its `rho_bin` edges are all estimated on its own fitting half. Sharing the
    control's would score a blended offset through a design fitted for a different one,
    which is the mistake `composition_preseason.score_arm` already refuses one layer down.
    """
    out = {}
    for arm in ARMS:
        frame = frame_at(cfg, pre, INCUMBENT_K if arm == "base" else k, first_season, route)
        train, val = selection_split(frame, test_seasons)
        tr, te, feats, dispersed, n_rho = variants(train, val)[BASE_VARIANT]
        out[arm] = (tr, te, feats, dispersed, n_rho)
    return out


def arm_rows(samples: np.ndarray, val: pd.DataFrame, roster: set, arm: str, kind: str,
             seed: int = SEED) -> list[dict]:
    """One arm's metric rows: both units × both populations.

    **The team metric exists only on the pooled frame**, and that is a fact about the unit
    rather than a limitation — `score_samples` reads the team-sum error off contiguous
    team-game blocks, and a draftable subset of a team-game is not a team-game.
    `composition_preseason.score_arm` states the same thing and this mirrors its layout so
    the two artifacts stack.
    """
    pooled = score_samples(samples, val, arm, seed)
    rows = [{"analysis": "arm", "arm": arm, "kind": kind, "unit": "player_game",
             "population": "pooled", "n": len(val),
             "crps": pooled["crps_minutes"], "r2": pooled["r2_minutes"],
             "mae": pooled["mae_minutes"], "bias": pooled["bias_minutes"],
             "pit_ks": pooled["pit_ks"],
             "team_sum_abs_error": pooled["team_sum_abs_error"]}]

    totals, units = season_totals(samples, val)
    realized = (val.groupby(["player_id", "season"], as_index=False)["y"].sum()
                .rename(columns={"y": "realized"}))
    units = units.merge(realized, on=["player_id", "season"], how="left")

    y_game = val["y"].to_numpy(float)
    game_pred = samples.mean(axis=0)
    game_mask = _draftable(val, roster)

    y_m, pred_m = y_game[game_mask], game_pred[game_mask]
    ss = float(np.sum((y_m - y_m.mean()) ** 2))
    rows.append({"analysis": "arm", "arm": arm, "kind": kind, "unit": "player_game",
                 "population": "draftable", "n": int(game_mask.sum()),
                 "crps": float(crps_from_samples(samples[:, game_mask], y_m).mean()),
                 "r2": 1.0 - float(np.sum((y_m - pred_m) ** 2)) / ss if ss > 0 else np.nan,
                 "mae": float(np.abs(y_m - pred_m).mean()),
                 "bias": float((pred_m - y_m).mean()),
                 "pit_ks": ks_uniform(pit_from_samples(samples[:, game_mask], y_m, seed))})

    for population, mask in (("draftable", _draftable(units, roster)),
                             ("pooled", np.ones(len(units), bool))):
        y_season = units.loc[mask, "realized"].to_numpy(float)
        drawn = totals[:, mask]
        pred = drawn.mean(axis=0)
        ss_tot = float(np.sum((y_season - y_season.mean()) ** 2))
        rows.append({
            "analysis": "arm", "arm": arm, "kind": kind, "unit": "player_season",
            "population": population, "n": int(mask.sum()),
            "crps": float(crps_from_samples(drawn, y_season).mean()),
            "r2": 1.0 - float(np.sum((y_season - pred) ** 2)) / ss_tot
            if ss_tot > 0 else np.nan,
            "mae": float(np.abs(y_season - pred).mean()),
            "bias": float((pred - y_season).mean()),
            "pit_ks": ks_uniform(pit_from_samples(drawn, y_season, seed)),
            "predictive_sd": float(drawn.std(axis=0).mean())})
    return rows


def announce_metric(n_features: int, n_rho: int, warmup: int) -> str:
    """The metric this arm will get, printed **before** the sampler starts.

    ⚠️ **A cost cliff, not a preference, and it is invisible in the artifact until the fit
    is over.** `choose_metric` grants `dense_e` only when `warmup >= 20 × parameters`, and
    this arm is 25 features + intercept + 4 dispersion bins = 30 — so the threshold is
    exactly 600 warmup draws. `stan_composition`'s own probe measured NUTS held at treedepth
    8–9 under `diag_e` against treedepth 4 under `dense_e`, ~10× the wall clock. A first
    attempt at `warmup: 500` — copied from the `effects` block, which sits under the same
    cliff — ran **32 minutes without completing its 500 warmup draws**, against
    `composition_effects`' **880 s** for this same arm at this same window for the whole
    500+500 fit under `dense_e`, and nothing said so until it was killed. CmdStan writes no
    draw until warmup ends and its progress lines are buffered away, so the *only* early
    signal is the metric itself. This prints it in the first second.
    """
    n_params = n_features + 1 + n_rho
    metric = choose_metric(n_params, warmup)
    print(f"    {n_params} parameters, {warmup} warmup draws → {metric}")
    if metric != "dense_e":
        print(f"    /!\\  `diag_e` on this head is ~10× the wall clock of `dense_e` "
              f"(treedepth 8–9 against 4 on its own probe). `dense_e` needs warmup ≥ "
              f"{DENSE_DRAWS_PER_PARAM * n_params:.0f} and this run has {warmup}.")
    return metric


def _draftable(frame: pd.DataFrame, roster: set) -> np.ndarray:
    """Season-start-roster mask over a frame carrying `player_id` and `season`."""
    return np.array([(s, p) in roster for p, s
                     in zip(frame["player_id"], frame["season"])])


def _cell(rows: list[dict], unit: str, population: str) -> dict:
    return next(r for r in rows if r["unit"] == unit and r["population"] == population)


#: The four comparisons, as `(label, arm, reference)`. The first two are the round's
#: question — the same contrast measured under the posterior and under the floor, on the
#: same frames at the same draw budget, so the retention between them is within-artifact.
#: The last two are the control the screen could not run: fitting has to still be worth
#: something on top of the better offset, or the arm has improved the floor by making the
#: head redundant.
COMPARISONS = (("fitted_increment", "preseason", "base"),
               ("floor_increment", "floor_preseason", "floor_base"),
               ("fit_value_base", "base", "floor_base"),
               ("fit_value_preseason", "preseason", "floor_preseason"))


def margins(series: dict, roster: set, k: float, route: str) -> list[dict]:
    """Every comparison × unit × population, paired and aligned by row identity."""
    rows = []
    for label, arm, reference in COMPARISONS:
        if arm not in series or reference not in series:
            continue
        for unit, position in (("player_game", 0), ("player_season", 1)):
            a, b = series[arm][position], series[reference][position]
            draft = pd.Index([(s, p) in roster for p, s, *_ in a.index])
            for population, mask in (("pooled", None), ("draftable", draft)):
                point, lo, hi = paired(a, b, mask, reps=N_BOOTSTRAP)
                rows.append({
                    "analysis": "margin", "comparison": label, "arm": arm,
                    "reference": reference, "unit": unit, "population": population,
                    "n": int(len(a) if mask is None else mask.to_numpy().sum()),
                    "crps_delta": point, "ci_lo": lo, "ci_hi": hi,
                    "k": k, "route": route})
    return rows


def retention(margin_frame: pd.DataFrame) -> list[dict]:
    """Fitted increment ÷ floor increment, per unit and population.

    The number session 4b's open question resolves to. Above 1.0 the posterior *grew* the
    increment, as P3's own did; below it, `beta` absorbed part of what the better offset was
    supplying. Reported rather than barred, because no bar for it was stated before the run
    and inventing one after seeing which side it landed on is exactly what P2 records as not
    being a bar.
    """
    rows = []
    for unit in ("player_game", "player_season"):
        for population in ("draftable", "pooled"):
            def pick(comparison: str) -> float:
                part = margin_frame[(margin_frame["comparison"] == comparison)
                                    & (margin_frame["unit"] == unit)
                                    & (margin_frame["population"] == population)]
                return float(part["crps_delta"].iloc[0]) if len(part) else np.nan

            fitted, floor = pick("fitted_increment"), pick("floor_increment")
            rows.append({"analysis": "retention", "unit": unit, "population": population,
                         "fitted_increment": fitted, "floor_increment": floor,
                         "retention": fitted / floor if floor else np.nan})
    return rows


def _flush(dest: Path, rows: list[dict], keys: tuple[str, ...]) -> None:
    """Merge one completed arm's rows over whatever is on disk, keyed by identity.

    `composition_effects._flush`'s rule and its reason: the arms are independent and each is
    a quarter-hour, so re-running one must leave the other in place. Truncating at the start
    of `run` made a partial re-run silently destroy the arms it was not fitting.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    built = pd.DataFrame(rows)
    if dest.exists():
        existing = pd.read_csv(dest)
        if len(existing) and set(keys) <= set(existing.columns):
            landing = set(map(tuple, built[list(keys)].astype(str).to_numpy()))
            keep = ~existing[list(keys)].astype(str).apply(tuple, axis=1).isin(landing)
            built = pd.concat([existing[keep], built], ignore_index=True)
    built.to_csv(dest, index=False)


def report(arms: pd.DataFrame, margin_frame: pd.DataFrame, ret: pd.DataFrame) -> bool:
    """The bar, as code, so the outcome is not a judgement made after seeing the numbers."""
    pd.set_option("display.width", 220)
    for unit in ("player_game", "player_season"):
        for population in ("draftable", "pooled"):
            shown = arms[(arms["unit"] == unit) & (arms["population"] == population)]
            print(f"\n  {unit} · {population}:")
            print(shown[["arm", "kind", "n", "crps", "r2", "mae", "bias", "pit_ks",
                         "predictive_sd", "team_sum_abs_error"]]
                  .round(5).to_string(index=False))

    print("\n  Paired margins (negative favours the left arm):")
    print(margin_frame[["comparison", "unit", "population", "n", "crps_delta",
                        "ci_lo", "ci_hi"]].round(5).to_string(index=False))
    print("\n  Retention — the fitted increment as a fraction of the floor's:")
    print(ret.round(5).to_string(index=False))

    bar = margin_frame[(margin_frame["comparison"] == "fitted_increment")
                       & (margin_frame["unit"] == "player_game")
                       & (margin_frame["population"] == "draftable")]
    team = arms[(arms["unit"] == "player_game") & (arms["population"] == "pooled")]
    exact = bool((team["team_sum_abs_error"].fillna(0) == 0).all())
    if bar.empty:
        print("\nGate: NOT EVALUABLE — both arms have to land in one run.")
        return False
    point, lo, hi = (float(bar["crps_delta"].iloc[0]), float(bar["ci_lo"].iloc[0]),
                     float(bar["ci_hi"].iloc[0]))
    clear = hi < 0
    print(f"\nGate — the bar stated before the run:\n"
          f"  per-player-game CRPS on the draftable population, blended vs the same-window "
          f"control:\n    {point:+.5f} [{lo:+.5f}, {hi:+.5f}] "
          f"({'PASS — interval clear of zero' if clear else 'FAIL — the interval spans zero'})\n"
          f"  team_sum_abs_error exactly 0 on every arm: "
          f"{'PASS' if exact else 'FAIL'}")
    print(f"  → session 4b's fit {'PASSES' if clear and exact else 'FAILS'}")
    return clear and exact


def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
    cfg_stan = cfg.get("stan", {})
    comp_cfg = cfg_stan.get("composition", {})
    pre_cfg = comp_cfg.get("preseason", {})

    first_season = str(pre_cfg.get("first_season", PILOT_FIRST_SEASON))
    k = float(pre_cfg.get("blend_k", SELECTED_K))
    route = str(pre_cfg.get("route", ROUTE))
    arms_to_fit = tuple(pre_cfg.get("arms", ARMS))
    keep = int(comp_cfg.get("predictive_samples", 200))
    iters = {"warmup": int(pre_cfg.get("warmup", cfg_stan.get("select_warmup", 500))),
             "samples": int(pre_cfg.get("samples", cfg_stan.get("select_samples", 500)))}
    seed = int(cfg_stan.get("seed", 42))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))

    print("Composition preseason FIT — does the blended offset survive the posterior?")
    print(f"  window {first_season} on; arms {', '.join(arms_to_fit)}; "
          f"k = {k:g} on the {route} route; {iters['warmup']}+{iters['samples']} × "
          f"{cfg_stan.get('chains', 4)} chains, {keep} predictive draws")

    from src.eda.preseason_value import covered_seasons

    covered = set(covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv")))
    if first_season < min(covered):
        raise ValueError(f"the fit window starts at {first_season}, before the preseason "
                         f"panel's first covered season {min(covered)} — the blend would "
                         f"be a structural no-op on the early rows")

    pre = preseason_share(pd.read_parquet(features_dir / "preseason.parquet"))
    roster = season_start_roster(cfg)
    built = build_arms(cfg, pre, first_season, k, route, test_seasons)
    tr0, te0 = built["base"][0], built["base"][1]
    print(f"  {len(tr0):,} fit / {len(te0):,} select player-games over "
          f"{te0.groupby(GROUP_KEYS, sort=False).ngroups:,} validation team-games; "
          f"{len(roster):,} season-start-roster keys")
    print("  The test split is LOCKED — this round fits and scores VALIDATION only "
          "(src/models/held_out.py).")

    arm_dest = out_dir / "composition_preseason_fit_arms.csv"
    diag_dest = out_dir / "composition_preseason_fit_diagnostics.csv"
    diag_dest.unlink(missing_ok=True)

    rows: list[dict] = []
    series: dict[str, tuple[pd.Series, pd.Series]] = {}
    diagnostics: list[dict] = []
    for arm in arms_to_fit:
        tr, te, feats, dispersed, n_rho = built[arm]

        # The floor first, on this arm's own frames — cheap, and it is what makes the
        # retention a within-artifact ratio rather than a comparison across two rounds'
        # draw budgets.
        floor_label = f"floor_{arm}"
        floor_samples = FloorComposition(keep).fit(tr).predict_samples(te, seed)
        floor_rows = arm_rows(floor_samples, te, roster, floor_label, "floor", seed)
        for row in floor_rows:
            row.update({"k": INCUMBENT_K if arm == "base" else k, "route": route,
                        "first_season": first_season, "n_features": 0})
        rows += floor_rows
        series[floor_label] = crps_series(floor_samples, te)
        _flush(arm_dest, floor_rows, ("arm", "unit", "population"))

        print(f"\n  fitting `{arm}` — {len(feats)} features, {n_rho} dispersion bins")
        announce_metric(len(feats), n_rho, iters["warmup"])
        started = time.perf_counter()
        model = StanComposition(feats, dispersed, n_rho, name=f"preseason/{arm}",
                                chains=int(cfg_stan.get("chains", 4)), seed=seed,
                                predictive_samples=keep, **iters).fit(tr)
        diagnostics.append(model.diagnostics)
        samples = model.predict_samples(te, seed)
        fitted_rows = arm_rows(samples, te, roster, arm, "fitted", seed)
        for row in fitted_rows:
            row.update({"k": INCUMBENT_K if arm == "base" else k, "route": route,
                        "first_season": first_season, "n_features": len(feats),
                        "rho": float(model.rho),
                        "metric": model.diagnostics["metric"],
                        "wall_clock_s": float(time.perf_counter() - started)})
        rows += fitted_rows
        series[arm] = crps_series(samples, te)
        _flush(arm_dest, fitted_rows, ("arm", "unit", "population"))
        diagnostics_frame([model.diagnostics]).to_csv(
            diag_dest, mode="a", header=not diag_dest.exists(), index=False)
        fitted_cell = _cell(fitted_rows, "player_game", "draftable")
        floor_cell = _cell(floor_rows, "player_game", "draftable")
        print(f"    per-player-game CRPS {fitted_cell['crps']:.5f} draftable "
              f"(its own floor {floor_cell['crps']:.5f}), rho {model.rho:.4f}, "
              f"{time.perf_counter() - started:.0f}s")

    margin_rows = margins(series, roster, k, route)
    margin_frame = pd.DataFrame(margin_rows)
    ret = pd.DataFrame(retention(margin_frame)) if len(margin_frame) else pd.DataFrame()

    arms = pd.DataFrame(rows)
    passed = report(arms, margin_frame, ret) if len(margin_frame) else False

    table = pd.concat([arms, margin_frame, ret], ignore_index=True)
    table["gate_passed"] = passed
    dest = out_dir / "composition_preseason_fit.csv"
    table.to_csv(dest, index=False)
    print(f"\nWrote {len(table):,} rows → {dest}")

    if diagnostics:
        diag = diagnostics_frame(diagnostics)
        print(f"Sampler: max R-hat {diag['max_rhat'].max():.4f}, "
              f"{int(diag['divergences'].sum())} divergences over {len(diag)} fits, "
              f"{diag['wall_clock_s'].sum() / 3600:.2f} h total")
    return {"composition_preseason_fit": dest, "arms": arm_dest,
            "diagnostics": diag_dest}


if __name__ == "__main__":
    from src.models.composition_preseason_fit import run as _run

    _run(yaml.safe_load(open("configs/default.yaml")))
