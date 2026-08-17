"""Grade the injected per-(player, season) sigma by role — `docs/draw-time-calibration-plan.md`.

`sim.minutes.player_season_sigma` is one number, applied to every player the composition
allocates minutes to. `make minutes-unification` selects it by CRPS at the season unit and
it wins there — but a **pooled** optimum can be the wrong value for every bucket at once,
and the scratch measurement that opened this plan says it is:

    role        n     sigma=0 sd_ratio   at sigma=0.375
    1 fringe    456   4.42               1.73   (still under-dispersed)
    2 bench     275   4.30               1.17
    3 starter   191   3.98               0.93
    4 star      189   3.83               0.85   (over-dispersed)

The raw deficit is nearly role-flat; what is *not* flat is where a constant **logit-scale**
sigma lands once it has been through the allocation and summed to a season. So this module
asks the obvious next question — what does each bucket want — and answers it with the grid
that already chose the scalar, run one bucket at a time.

## What is being calibrated, and what is not

This is a **draw-time** constant, not a likelihood. Nothing here refits a head, touches
`composition_glm.stan` or reads a sampler. The knob nests exactly: a one-element vector is
the shipped scalar (`unit_sigma`), and an absent `sim.minutes.player_season_sigma_by_role`
key leaves the scalar in charge.

The role axis is the composition head's **own** `rho_bin` — train quantiles of `w_share` —
so it is the same axis its fitted per-game dispersion is graded over (0.140 fringe to 0.0735
star, a 1.91x spread) and the same axis the quadrature line's fitted `sigma_u` came back
graded on (0.609 fringe to 0.291 star, 2.09x — `docs/composition-quadrature-plan.md` §8).
**That fitted vector is a reference for the shape of what comes out here, never a source of
values**: it is one probe-scale season, and a likelihood-scale sigma and a draw-time
calibration sigma answer different questions.

## Two grids, and only one of them selects

Selection reads the fitting half. The per-bucket optima are searched on the **last two
training seasons** — the same rows `estimate_sigma_on_train` uses to choose the scalar — and
the identical search is run on validation only so the two can be compared. **Validation
never chooses**: a bucket ships its train optimum when the two agree to a grid step, and
keeps the shared scalar when they do not. That rule is `docs/draw-time-calibration-plan.md`
step 2 and it is written down before the numbers, because a per-bucket sigma is four chances
to read a wiggle as a signal.

The search is coordinate-wise. The buckets are not independent — a fringe player's shock
takes minutes from a starter through the team constraint — so each bucket is optimized
against **its own** units' CRPS with the others held, and the whole cycle is run twice. A
bucket whose optimum moves between passes is the coupling talking, and it is reported rather
than smoothed over.

Usage:
    python -m src.models.minutes_role_sigma
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.minutes_unification import (FIT_WINDOW, N_ROLE_BINS, POSTERIOR_DRAWS,
                                            SEED, TRAIN_SIGMA_SEASONS, UNIT_KEYS,
                                            format_role_sigma, injected_games,
                                            injected_totals, paired_bootstrap,
                                            realized_totals, rehydrate_composition,
                                            rehydrate_minutes, role_bins, shipped_sigma,
                                            shipped_sigma_by_role, teammate_coupling,
                                            unit_codes, validation_frames, verdict)
from src.models.posteriors import load, posteriors_dir, require_window
from src.models.stan_utils import crps_from_samples, ks_uniform, pit_from_samples

# The graded grid. **Regular**, unlike the scalar sweep's `PS_SIGMAS`, because the ship rule
# is "train and validation optima agree to a grid step" and an irregular grid makes that
# sentence mean a different thing at each rung. The shipped scalar 0.375 is ON it, so
# "this bucket wants what it already has" is an outcome the grid can express rather than
# something it has to round to.
#
# The span is set by what the calibration target implies, not by taste. Season-total
# predictive variance goes as `V0 + c*sigma^2`, so the scratch sd_ratios above solve for the
# sigma that would put each bucket at 1.0: ~0.69 for fringe and ~0.32 for star. A grid
# stopping at 0.6 would have censored the fringe optimum and reported a boundary as a result.
GRID_STEP = 0.075
GRADED_SIGMAS = tuple(round(0.15 + GRID_STEP * i, 4) for i in range(9))

# Coordinate cycles. Two, because one cannot show stability and three costs another 12
# minutes to confirm what the second already showed; a bucket still moving after two is a
# reported failure of the coordinate assumption, not something to iterate away.
COORD_PASSES = 2

ROLE_LABELS = ("fringe", "bench", "starter", "star")

# Nominal PIT tail mass per side. The fringe/bench miss is asymmetric — realized seasons
# collapse below the predictive far more often than they exceed it — so the two tails are
# reported separately and a symmetric `sd_ratio` alone would hide it.
PIT_TAIL = 0.05

# The z streams, matched to the two scalar sweeps in `minutes_unification` so a graded row
# and a scalar row are the same draw at the same sigma rather than two samples of it.
Z_SEED_VAL = SEED + 7
Z_SEED_TRAIN = SEED + 11


# ── Scoring one arm, by role ──────────────────────────────────────────────────

def sd_ratio(totals: np.ndarray, y: np.ndarray) -> float:
    """Realized error over predictive spread — the number the injection exists to move to 1.

    RMSE rather than MAE in the numerator because the denominator is an sd, and a ratio of
    two different functionals of spread is not a calibration statistic. At `sigma = 0` this
    reads ~4.3 pooled, which is the "4.58x too narrow" README quotes arrived at from the
    other side (the marginal head's predictive sd over the composition's).
    """
    rmse = float(np.sqrt(((totals.mean(axis=0) - np.asarray(y, float)) ** 2).mean()))
    spread = float(totals.std(axis=0).mean())
    return rmse / spread if spread > 0 else float("inf")


def score_block(totals: np.ndarray, y: np.ndarray, seed: int = SEED) -> dict:
    """The season-unit metric set for one arm on one set of units."""
    y = np.asarray(y, dtype=float)
    pit = pit_from_samples(totals, y, seed)
    return {
        "n": int(len(y)),
        "n_draws": int(totals.shape[0]),
        "crps_minutes": float(crps_from_samples(totals, y).mean()),
        "mae_minutes": float(np.abs(totals.mean(axis=0) - y).mean()),
        "bias_minutes": float((totals.mean(axis=0) - y).mean()),
        "predictive_sd": float(totals.std(axis=0).mean()),
        "sd_ratio": sd_ratio(totals, y),
        "pit_ks": ks_uniform(pit),
        "pit_tail_lo": float((pit < PIT_TAIL).mean()),
        "pit_tail_hi": float((pit > 1 - PIT_TAIL).mean()),
    }


def role_rows(totals: np.ndarray, y: np.ndarray, bins: np.ndarray, sigma,
              arm: str, unit: str, split: str, seed: int = SEED) -> list[dict]:
    """One row per role bucket plus a pooled row, all from a single draw.

    `sigma_scalar` carries **the bucket's own** sigma on a bucket row and the shared value
    on the pooled one, so a graded arm's table can be read a row at a time without parsing
    the `sigma` cell back apart.
    """
    y = np.asarray(y, dtype=float)
    per_bin = broadcast_sigma(sigma)
    label = format_sigma(sigma)
    rows = [{"arm": arm, "unit": unit, "split": split, "role_bin": 0, "role": "pooled",
             "sigma": label, "sigma_scalar": scalar_or_nan(sigma),
             **score_block(totals, y, seed)}]
    for b in range(N_ROLE_BINS):
        sel = bins == b
        if not sel.any():
            continue
        rows.append({"arm": arm, "unit": unit, "split": split, "role_bin": b + 1,
                     "role": ROLE_LABELS[b], "sigma": label,
                     "sigma_scalar": float(per_bin[b]),
                     **score_block(totals[:, sel], y[sel], seed)})
    return rows


def broadcast_sigma(sigma) -> np.ndarray:
    """A sigma as one value per role bin, whether it arrived shared or graded."""
    arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    return np.full(N_ROLE_BINS, float(arr[0])) if arr.size == 1 else arr


def format_sigma(sigma) -> str:
    """A sigma vector as one CSV cell — `0.375` shared, `0.6|0.45|0.375|0.3` graded.

    Differs from `format_role_sigma` only in the shared case, and deliberately: a model card
    column answers "is this graded" and leaves shared blank, while this column has to carry
    the value a row was actually drawn at. The join is shared so there is one format.
    """
    arr = np.atleast_1d(np.asarray(sigma, dtype=float)).ravel()
    return f"{float(arr[0]):.4g}" if arr.size == 1 else format_role_sigma(arr)


def scalar_or_nan(sigma) -> float:
    """The shared value, or NaN when the sigma is graded and there is no single one."""
    arr = np.atleast_1d(np.asarray(sigma, dtype=float))
    return float(arr[0]) if arr.size == 1 else float("nan")


# ── The two searches ──────────────────────────────────────────────────────────

def shared_profile(draw, bins: np.ndarray, y: np.ndarray, split: str,
                   sigmas=GRADED_SIGMAS, seed: int = SEED
                   ) -> tuple[np.ndarray, list[dict]]:
    """Per-bucket CRPS along the **shared** sigma grid, and each bucket's argmin.

    The coordinate search's starting point, and it is nearly free: the grid is the one
    already being drawn, and the only new thing is scoring each bucket's units separately
    instead of pooling them. It is also the honest first answer to the falsifier — if the
    per-bucket curves all bottom out at the same rung, grading is not going to help and no
    amount of coordinate descent will change that.

    Read as a *marginal* profile, not as the optimum: every bucket moves together here, so
    what it locates is where bucket `b` would like the shared value to be, with the coupling
    still varying underneath it.
    """
    rows, curves = [], {b: {} for b in range(N_ROLE_BINS)}
    for sigma in sigmas:
        totals = draw(np.full(N_ROLE_BINS, sigma))
        crps = crps_from_samples(totals, np.asarray(y, float))
        rows.extend(role_rows(totals, y, bins, float(sigma), "shared_grid",
                              "role_profile", split, seed))
        for b in range(N_ROLE_BINS):
            sel = bins == b
            if sel.any():
                curves[b][float(sigma)] = float(crps[sel].mean())
    start = np.array([min(curves[b], key=curves[b].get) if curves[b] else np.nan
                      for b in range(N_ROLE_BINS)], dtype=float)
    return start, rows


def coordinate_search(draw, bins: np.ndarray, y: np.ndarray, start: np.ndarray,
                      split: str, sigmas=GRADED_SIGMAS, passes: int = COORD_PASSES,
                      seed: int = SEED) -> tuple[np.ndarray, list[dict], bool]:
    """Cycle the buckets, minimizing **each bucket's own** CRPS with the others held.

    Each bucket is scored on its own units rather than on the pooled total, because that is
    the quantity the ship rule is stated in and because a pooled objective would let the two
    largest buckets choose for the two smallest. The coupling still enters — the other
    buckets' sigmas are in every draw — which is exactly what the second pass tests.

    Returns the vector, the iteration rows, and whether the second pass moved anything.
    """
    current = np.asarray(start, dtype=float).copy()
    y = np.asarray(y, dtype=float)
    cache: dict[tuple, np.ndarray] = {}
    rows: list[dict] = []
    stable = True
    n_draws = 0

    def crps_at(vec: np.ndarray) -> np.ndarray:
        nonlocal n_draws
        key = tuple(np.round(vec, 6))
        if key not in cache:
            totals = draw(np.asarray(key, dtype=float))
            n_draws = int(totals.shape[0])
            cache[key] = crps_from_samples(totals, y)
        return cache[key]

    for p in range(1, passes + 1):
        for b in range(N_ROLE_BINS):
            own = bins == b
            if not own.any():
                continue
            scores = {}
            for sigma in sigmas:
                vec = current.copy()
                vec[b] = float(sigma)
                crps = crps_at(vec)
                scores[float(sigma)] = float(crps[own].mean())
                rows.append({"arm": "coordinate", "unit": "coordinate_search",
                             "split": split, "role_bin": b + 1, "role": ROLE_LABELS[b],
                             "sigma": format_sigma(vec), "sigma_scalar": float(sigma),
                             "pass": p, "n": int(own.sum()), "n_draws": n_draws,
                             "crps_minutes": scores[float(sigma)],
                             "crps_pooled": float(crps.mean())})
            best = min(scores, key=scores.get)
            if p > 1 and abs(best - current[b]) > 1e-9:
                stable = False
            current[b] = best
    return current, rows, stable


def ship_rule(train_vec: np.ndarray, val_vec: np.ndarray, fallback: float,
              step: float = GRID_STEP) -> tuple[np.ndarray, list[str]]:
    """Train's optimum where the two halves agree to a grid step; the scalar where not.

    Written as code for the reason `minutes_unification.verdict` is: a per-bucket rule that
    is applied after seeing which buckets happened to agree is not a rule. `fallback` is the
    shipped shared sigma, so a bucket the two halves disagree about is left exactly as it is
    today rather than given a compromise nothing selected.
    """
    shipped, notes = [], []
    for b in range(N_ROLE_BINS):
        agrees = abs(float(train_vec[b]) - float(val_vec[b])) <= step + 1e-9
        shipped.append(float(train_vec[b]) if agrees else float(fallback))
        notes.append("train" if agrees else "fallback_disagree")
    return np.asarray(shipped, dtype=float), notes


# ── Entry point ───────────────────────────────────────────────────────────────

def split_inputs(model, artifact, raw: pd.DataFrame, y_column: str = "y") -> dict:
    """Everything a search needs for one split: design frame, eta, codes, bins, realized."""
    frame = artifact.recipe.transform(raw)
    eta_base, rho = model._eta_base(frame)
    codes = unit_codes(raw)
    n_units = int(codes.max()) + 1
    bins = role_bins(frame, codes, n_units)
    realized = realized_totals(raw, y_column)["realized"].to_numpy(float)
    return {"frame": frame, "raw": raw, "eta_base": eta_base, "rho": rho, "codes": codes,
            "bins": bins, "realized": realized, "n_units": n_units}


def drawer(inputs: dict, z_seed: int, seed: int = SEED):
    """A `sigma -> (draws x units)` closure over one split's fixed inputs."""
    def draw(sigma) -> np.ndarray:
        return injected_totals(inputs["frame"], inputs["raw"], inputs["eta_base"],
                               inputs["rho"], inputs["codes"], sigma, z_seed=z_seed,
                               bins=inputs["bins"], seed=seed)
    return draw


def team_sum_sd(inputs: dict, sigma, z_seed: int, seed: int = SEED) -> float:
    """Predictive sd of a team's season minutes across draws — 0 by construction.

    The constraint the whole head exists to enforce, asserted rather than assumed at the
    graded vector. A per-unit shock re-allocates minutes *within* a team-game, so grading it
    by role cannot change the team total — but "cannot" is what a wiring bug looks like from
    the inside, and this costs one draw.
    """
    games = injected_games(inputs["frame"], inputs["eta_base"], inputs["rho"],
                           inputs["codes"], sigma, z_seed, inputs["bins"], seed)
    codes = inputs["raw"].groupby(["season", "team_id"], sort=True).ngroup().to_numpy()
    order = np.argsort(codes, kind="stable")
    starts = np.searchsorted(codes[order], np.arange(codes.max() + 1))
    totals = np.add.reduceat(games[:, order], starts, axis=1)
    return float(totals.std(axis=0).mean())


def run(cfg: dict) -> dict[str, Path]:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sim = cfg.get("sim", {})
    keep = int(cfg_sim.get("posterior_draws", POSTERIOR_DRAWS))
    window = str(cfg_sim.get("fit_window", FIT_WINDOW))
    shipped = shipped_sigma(cfg)

    print("Role-graded injection sigma — what does each bucket want?")
    print(f"  reading persisted posteriors at the `{window}` window; nothing is refitted.")
    artifacts = {name: load(name, posteriors_dir(cfg, window))
                 for name in ("minutes", "composition")}
    require_window(artifacts, window)
    if artifacts["composition"].draws.get("sigma_u_draws") is not None:
        raise SystemExit(
            "the composition artifact carries a FITTED `sigma_u`, which supersedes the "
            "injected constant this module calibrates — `rehydrate_composition` already "
            "prefers it, and grading a constant nothing consumes would be measuring "
            "nothing. See `docs/composition-quadrature-plan.md`.")

    composition = rehydrate_composition(artifacts["composition"], keep)
    (minutes_train, minutes_val, composition_val,
     composition_train) = validation_frames(cfg)

    seasons = sorted(composition_train["season"].unique())[-TRAIN_SIGMA_SEASONS:]
    train_raw = (composition_train[composition_train["season"].isin(seasons)]
                 .reset_index(drop=True))

    val = split_inputs(composition, artifacts["composition"], composition_val)
    train = split_inputs(composition, artifacts["composition"], train_raw)
    draw_val = drawer(val, Z_SEED_VAL)
    draw_train = drawer(train, Z_SEED_TRAIN)

    print(f"\n  The test split is LOCKED, and so is selection: the shipped vector is the "
          f"TRAIN\n  optimum ({', '.join(seasons)}, {train['n_units']:,} player-seasons). "
          f"Validation ({', '.join(sorted(composition_val['season'].unique()))}, "
          f"{val['n_units']:,})\n  confirms and never chooses.")
    for name, inputs in (("train", train), ("validation", val)):
        counts = np.bincount(inputs["bins"], minlength=N_ROLE_BINS)
        print(f"  {name:<11} units by role: " +
              ", ".join(f"{ROLE_LABELS[b]} {counts[b]:,}" for b in range(N_ROLE_BINS)))

    rows: list[dict] = []

    # ── The marginal profile: each bucket's CRPS along the SHARED grid ─────────
    print(f"\nStep 1 — the shared-sigma grid, scored by bucket "
          f"({len(GRADED_SIGMAS)} draws per split).")
    train_start, train_rows = shared_profile(draw_train, train["bins"],
                                             train["realized"], "train")
    val_start, val_rows = shared_profile(draw_val, val["bins"], val["realized"],
                                         "validation")
    rows.extend(train_rows)
    rows.extend(val_rows)
    print(_profile_table(train_rows, "train"))
    print(_profile_table(val_rows, "validation"))
    print(f"  marginal per-bucket optima — train "
          f"{format_sigma(train_start)}, validation {format_sigma(val_start)}")

    # ── The coordinate search ─────────────────────────────────────────────────
    print(f"\nStep 2 — coordinate-wise, {COORD_PASSES} passes, each bucket scored on its "
          f"own units.")
    train_vec, train_iters, train_stable = coordinate_search(
        draw_train, train["bins"], train["realized"], train_start, "train")
    val_vec, val_iters, val_stable = coordinate_search(
        draw_val, val["bins"], val["realized"], val_start, "validation")
    rows.extend(train_iters)
    rows.extend(val_iters)
    print(f"  train      optimum {format_sigma(train_vec)}  "
          f"({'stable' if train_stable else 'MOVED on the second pass'})")
    print(f"  validation optimum {format_sigma(val_vec)}  "
          f"({'stable' if val_stable else 'MOVED on the second pass'})")

    graded, notes = ship_rule(train_vec, val_vec, shipped)
    print(f"\n  Ship rule — train's value where the two halves agree to a grid step "
          f"({GRID_STEP}),\n  the shared {shipped:.3f} where they do not:")
    for b in range(N_ROLE_BINS):
        print(f"    {b + 1} {ROLE_LABELS[b]:<8} train {train_vec[b]:.3f}  validation "
              f"{val_vec[b]:.3f}  ->  {graded[b]:.3f}  ({notes[b]})")
    spread = float(graded.max() / graded.min()) if graded.min() > 0 else float("inf")
    flat = bool(np.allclose(graded, graded[0]))
    print(f"  GRADED VECTOR {format_sigma(graded)} — spread {spread:.2f}x"
          + ("  ⚠️ FLAT: the falsifier fired, see the plan." if flat else ""))

    # What CONFIG says, against what this search just selected. The search does not read the
    # config key — it re-derives the vector every run — so the two can drift exactly the way
    # `SHIPPED_PS_SIGMA` drifted off `player_season_sigma` between 2026-08-14 and 2026-08-16,
    # silently and without anything downstream failing. Printed rather than raised, because a
    # deliberate override is legitimate and a run that refuses to finish is not how you find
    # out about one.
    configured = shipped_sigma_by_role(cfg)
    if configured is None:
        print(f"  ⚠️ `sim.minutes.player_season_sigma_by_role` is ABSENT, so the simulator "
              f"still draws the\n  shared {shipped:.3f}. The vector above is what this "
              f"search selects; nothing consumes it until the key exists.")
    elif not np.allclose(configured, graded):
        print(f"  ⚠️ CONFIG AND THIS SEARCH DISAGREE. Config ships "
              f"[{', '.join(f'{s:.3f}' for s in configured)}] and the search selects "
              f"[{', '.join(f'{s:.3f}' for s in graded)}].\n  One of them is stale — the "
              f"readout below scores the SEARCH's vector, not the one being drawn.")

    # ── Confirmation on validation, against the incumbent ─────────────────────
    print(f"\nStep 3 — the graded vector against the shipped scalar, on validation.")
    # `uninjected` is the control the whole comparison is read against, and it is a row here
    # rather than a remembered number: the per-role sd_ratio at sigma = 0 is what says the
    # raw deficit is nearly role-FLAT, which is the finding that makes a graded sigma
    # interesting rather than obvious. Quoting it from a scratch table while the two arms
    # beside it come from an artifact is how a doc goes stale in one place out of three.
    for label, sigma in (("uninjected", 0.0), ("shipped_scalar", shipped),
                         ("graded", graded)):
        for split, inputs, draw in (("train", train, draw_train),
                                    ("validation", val, draw_val)):
            totals = draw(sigma)
            rows.extend(role_rows(totals, inputs["realized"], inputs["bins"], sigma,
                                  label, "role_readout", split))

    readout = pd.DataFrame([r for r in rows if r["unit"] == "role_readout"
                            and r["split"] == "validation"])
    print(_readout_table(readout))

    # ── The gate: no regression against the marginal head on the common rows ──
    minutes = rehydrate_minutes(artifacts["minutes"], keep)
    mins_frame = artifacts["minutes"].recipe.transform(minutes_val)
    mins_totals = minutes.predict_samples(mins_frame, SEED)
    mins_units = minutes_val[UNIT_KEYS].copy()
    mins_units["realized"] = minutes_val["successes"].to_numpy(float)

    comp_units = (composition_val.groupby(UNIT_KEYS, sort=True).size()
                  .reset_index(name="games"))
    comp_pos = {k: i for i, k in enumerate(map(tuple, comp_units[UNIT_KEYS].to_numpy()))}
    keys = [k for k in map(tuple, mins_units[UNIT_KEYS].to_numpy()) if k in comp_pos]
    idx_c = np.array([comp_pos[k] for k in keys])
    idx_m = np.array([i for i, k in
                      enumerate(map(tuple, mins_units[UNIT_KEYS].to_numpy()))
                      if k in comp_pos])
    y_mins = mins_units["realized"].to_numpy(float)[idx_m]
    y_comp = val["realized"][idx_c]
    crps_mins = crps_from_samples(mins_totals[:, idx_m], y_mins)

    print(f"\n  Gate — pooled season CRPS against the marginal head on the "
          f"{len(keys):,} rows both cover:")
    for label, sigma in (("shipped_scalar", shipped), ("graded", graded)):
        totals = draw_val(sigma)[:, idx_c]
        crps = crps_from_samples(totals, y_comp)
        delta = paired_bootstrap(crps, crps_mins)
        rows.append({"arm": label, "unit": "head_to_head", "split": "validation",
                     "role_bin": 0, "role": "pooled", "sigma": format_sigma(sigma),
                     "sigma_scalar": scalar_or_nan(sigma), "n": len(keys),
                     "n_draws": keep, "crps_minutes": float(crps.mean()),
                     **delta, "verdict": verdict(delta)})
        print(f"    {label:<15} CRPS {float(crps.mean()):8.4f} against "
              f"{float(crps_mins.mean()):.4f}  "
              f"{delta['crps_delta']:+.4f} [{delta['ci_lo']:+.4f}, "
              f"{delta['ci_hi']:+.4f}] — {verdict(delta).upper()}")

    # ── The recorded trade: teammate coupling at the graded values ────────────
    # `docs/draw-time-calibration-plan.md` step 3 names this as a re-read rather than a new
    # measurement. It is the dynamic the composition exists for — a team's season minutes
    # are a fixed pot, so teammates' totals are negatively correlated — and the injection
    # dilutes it, because a per-unit shock is the one thing in the draw that is NOT taken
    # out of a teammate. Grading sigma moves that dilution per bucket, so it cannot be
    # assumed to carry over from the shared reading.
    coupling_units = (composition_val.groupby(UNIT_KEYS, sort=True).size()
                      .reset_index(name="games"))
    print(f"\n  Teammate coupling — the zero-sum dynamic the injection dilutes:")
    for label, sigma in (("shipped_scalar", shipped), ("graded", graded)):
        couple = teammate_coupling(draw_val(sigma), coupling_units, composition_val, label)
        couple.update({"split": "validation", "role_bin": 0, "role": "pooled",
                       "sigma": format_sigma(sigma), "sigma_scalar": scalar_or_nan(sigma)})
        rows.append(couple)
        print(f"    {label:<15} r {couple['r_teammates']:+.4f} against the "
              f"{couple['r_implied_by_fixed_sum']:+.4f} a fixed sum over "
              f"{couple['roster_size']:.2f} players forces")

    # ── The constraint, asserted ──────────────────────────────────────────────
    team_sd = team_sum_sd(val, graded, Z_SEED_VAL)
    rows.append({"arm": "graded", "unit": "team_sum", "split": "validation",
                 "role_bin": 0, "role": "pooled", "sigma": format_sigma(graded),
                 "sigma_scalar": float("nan"), "n": val["n_units"], "n_draws": keep,
                 "team_season_sd": team_sd})
    if team_sd > 1e-6:
        raise AssertionError(f"a team's season minutes moved across draws (sd {team_sd:g}); "
                             f"the graded injection must re-allocate WITHIN a team-game")
    print(f"\n  Team-season total sd across draws at the graded vector: {team_sd:.3g} — "
          f"the constraint holds.")

    table = pd.DataFrame(rows)
    table["fit_window"] = window
    table["shipped_scalar_sigma"] = shipped
    table["graded_vector"] = format_sigma(graded)
    table["coordinate_stable"] = bool(train_stable and val_stable)
    dest = out_dir / "minutes_role_sigma.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} role-sigma rows → {dest}")
    return {"metrics": dest}


def _profile_table(rows: list[dict], split: str) -> str:
    frame = pd.DataFrame([r for r in rows if r["split"] == split])
    wide = frame.pivot_table(index="sigma_scalar", columns="role",
                             values="crps_minutes", aggfunc="first")
    order = [c for c in ("pooled",) + ROLE_LABELS if c in wide.columns]
    return (f"\n  {split} — season CRPS by role along the shared grid:\n"
            + wide[order].round(3).to_string())


def _readout_table(frame: pd.DataFrame) -> str:
    shown = ["arm", "role", "n", "crps_minutes", "predictive_sd", "sd_ratio", "pit_ks",
             "pit_tail_lo", "pit_tail_hi"]
    return ("\n  validation, by role (sd_ratio -> 1, tails -> 0.05):\n"
            + frame.sort_values(["role_bin", "arm"])[shown].round(4)
            .to_string(index=False))


if __name__ == "__main__":
    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg)
