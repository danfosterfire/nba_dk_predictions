"""Session 4b, gate 1: does a preseason reading improve the composition's PRIOR SHARE?

`make composition-preseason` → `outputs/predictions/composition_preseason.csv`. numpy only,
minutes, **no CmdStan and no fit of the head** — which is the whole design of this round.

## Why the prior share and not a feature column

Every other preseason block in this project is `docs/preseason-plan.md`'s house pattern: extra
columns on `beta`, difference-coded, coefficient zero recovering the incumbent. On the
composition head that pattern reaches **one of the three routes** `w_share` takes into the
model. A player's prior-season minutes share — or his draft bucket's expanding mean if he has
none — enters as:

1. a **feature**. `OWN = logit_share_lag1` is `logit(w_share)` and is in every variant's
   feature list, so a coefficient does modulate it and the house pattern *can* add a
   preseason column beside it;
2. the **offset**. `sequential_columns` turns `w_share` into `logit_prior`, the carry-forward
   `beta` only corrects — **no coefficient can move it**; and
3. the **allocation order**. `order_frame` sorts each team-season by `w_share` descending and
   the multinomial's decomposition into sequential binomials is taken in that order — again
   **unreachable by any coefficient**.

So routes 2 and 3 are the part of this head the round's house pattern cannot get at, and they
are exactly what P3 flagged when it opened this session: *"a plausible composition-specific
win worth checking there: preseason minutes share updating the ordering and prior-share
feature for players who changed teams."*

## Why this is cheap, and why cheap is the right first gate

`FloorComposition` — the head's own no-fit floor — has a mean function that is **the offset
alone**: `predict_samples` sets `eta = 0`, so route 1 is switched off entirely and only routes
2 and 3 remain. That is not a limitation of the screen, it is what makes it the right one —
the floor isolates precisely the two channels no coefficient can reach, with no sampler
involved, and it is the arm every fitted variant is scored against in `stan_composition`'s own
ladder.

That makes this the composition's version of P1: a screen that can reject an arm before any
of the head's 9.92 h (full window) or ~1 h (pilot) is spent. It is not a substitute for the
fit — `beta` can correct an offset the floor cannot — but an offset that is *worse* before
fitting is not a promising place to spend sampler time, and the house discipline is to say so
first.

## The arms

One axis, and the incumbent is on it. `w' = ω·pre + (1 − ω)·w_share` with
`ω = m / (m + k)` over preseason minutes, so:

- `k → ∞` is the incumbent **exactly** — the nesting discipline `n_rho = 1`, `U_n = 0` and
  `θ = 0` already carry elsewhere, and it is asserted rather than assumed here;
- `k = 0` is the raw preseason share for anyone who has one;
- a player with no preseason row has `m = 0`, so `ω = 0` and he keeps his incumbent share by
  the blend's own arithmetic rather than by a special case.

`k` is selected on an **inner carve of the fitting half** — `minutes_preseason`'s construction
and `rookie_priors`', for the same reason — and never on the split it is scored against.

## The unit, and the population

Both units the head is ever read at, because `make minutes-unification` is the standing
demonstration that this head's verdict belongs to a unit: **per player-game** (what
`stan_composition` selects on) and **per player-season** (where the same posterior loses).

Quoted on the season-start-roster population per P1 decision 5, with the pooled reading
beside it. The preseason share is `mpg_pre / 48` — a share of game length, the same statistic
`w_share` is — and **not** the panel's `min_share_pre`, which is a share of the team's
preseason minutes and differs by a factor of ~5. `rookie_priors` records that error.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.models.held_out import selection_split
from src.models.stan_composition import (GROUP_KEYS, PILOT_FIRST_SEASON, SHARE_CLIP,
                                         FloorComposition, composition_frame,
                                         ragged_arrays, score_samples)

#: Regulation length, putting `mpg_pre` on `w_share`'s scale (minutes over game length).
#: `rookie_priors.REGULATION_LENGTH` is the same constant for the same reason; it is repeated
#: rather than imported so that neither module's meaning depends on the other's survival.
REGULATION_LENGTH = 48.0

#: The blend grid, in preseason minutes. The last rung is the incumbent to numerical
#: precision and the first is the raw preseason share, so an interior optimum is a real one.
SHRINK_GRID = (0.0, 10.0, 20.0, 40.0, 80.0, 160.0, 320.0, 1e9)

#: The incumbent, named rather than inferred from the grid's last rung.
INCUMBENT_K = 1e9

#: Target seasons in the inner carve that selects `k`, off the END of the fitting half.
INNER_SCORE_SEASONS = 2

#: Draws per row from the floor's predictive. Lower than `stan_composition`'s 200 because
#: this round scores eight frames rather than one and the quantity being resolved is a
#: difference between two mean functions, not a tail.
PREDICTIVE_SAMPLES = 120

SEED = 20260814


def preseason_share(panel: pd.DataFrame) -> pd.DataFrame:
    """`(season, player_id) -> (pre_share, min_pre)` on `w_share`'s own scale."""
    out = panel[["season", "player_id", "mpg_pre", "min_pre"]].copy()
    out["pre_share"] = out["mpg_pre"] / REGULATION_LENGTH
    return out.drop(columns=["mpg_pre"])


#: The three routes a better prior share can take into this head, as `(offset, order)` flags.
#: `w_share` reaches the model **twice** — as `logit_prior` and as the allocation order — so a
#: single blended arm cannot say which route pays. `both` is the arm; the other two are the
#: attribution, and they are the instrument P2 had to re-run a harness to add.
ROUTES: dict[str, tuple[bool, bool]] = {
    "both": (True, True),
    "offset_only": (True, False),
    "order_only": (False, True),
}


def blend_hook(pre: pd.DataFrame, k: float, route: str = "both"):
    """A `composition_frame(share_hook=...)` callable blending the preseason into `w_share`.

    Returned as a closure because `composition_frame` owns the frame and this owns the rule.
    A player with no preseason row falls out with `min_pre = 0`, hence `ω = 0`, hence his
    incumbent share — which is why the join is `how="left"` with a fill rather than an inner
    join that would silently drop a third of the league.

    `route` decides which of the two channels the blended share is written to. `order_share`
    is always emitted so that the ordering is explicit in every arm rather than defaulting
    silently — `order_frame` falls back to `w_share` only for callers that never set it.
    """
    if route not in ROUTES:
        raise ValueError(f"unknown route {route!r}; expected one of {sorted(ROUTES)}")
    on_offset, on_order = ROUTES[route]

    def hook(lagged: pd.DataFrame) -> pd.DataFrame:
        out = lagged.merge(pre, on=["season", "player_id"], how="left")
        if len(out) != len(lagged):
            raise AssertionError("the preseason join duplicated a (season, player_id)")
        observed = out["pre_share"].to_numpy(float)
        seen = np.isfinite(observed)
        minutes = np.where(seen, out["min_pre"].fillna(0.0).to_numpy(float), 0.0)
        weight = minutes / (minutes + k) if k > 0 else seen.astype(float)
        weight = np.where(seen, weight, 0.0)
        incumbent = out["w_share"].to_numpy(float)
        blended = weight * np.nan_to_num(observed) + (1.0 - weight) * incumbent
        out["w_share"] = blended if on_offset else incumbent
        out["order_share"] = blended if on_order else incumbent
        return out.drop(columns=["pre_share", "min_pre"])
    return hook


def blended_frame(cfg: dict, pre: pd.DataFrame, k: float,
                  route: str = "both") -> pd.DataFrame:
    """The ordered composition frame at blend constant `k`, over **every** season.

    Split out from `frame_at` so a caller fitting the same blend at two windows builds it
    once. `composition_preseason_fit` does exactly that — its covered-window control and its
    full-window one are the same unblended frame cut in two places — and rebuilding it per
    arm re-runs `order_frame` over 700k rows for a result that is identical by construction.
    """
    hook = None if k >= INCUMBENT_K else blend_hook(pre, k, route)
    return composition_frame(cfg, share_hook=hook)


def frame_at(cfg: dict, pre: pd.DataFrame, k: float, first_season: str,
             route: str = "both") -> pd.DataFrame:
    """The ordered composition frame at blend constant `k`, cut to the fitting window."""
    return cut_window(blended_frame(cfg, pre, k, route), first_season)


def cut_window(frame: pd.DataFrame, first_season: str) -> pd.DataFrame:
    """The fitting window, as a season-label suffix of an already-built frame.

    The frame is built over every season regardless of this — the lag columns and the
    expanding rookie prior both reach backwards, so trimming earlier would drop the
    window's own first cohort rather than window it. `stan_composition.run` takes the same
    two steps in the same order.
    """
    return frame[frame["season"] >= first_season].reset_index(drop=True)


def season_start_roster(cfg: dict) -> set[tuple[str, int]]:
    """`{(season, player_id)}` for the draft pool — P1 decision 5's population.

    One definition, shared with `composition_preseason_fit`: every preseason figure in this
    round is quoted on the season-start rosters, and two modules deriving that set from the
    same three inputs by hand is how the two rounds would come to disagree about which rows
    they are quoting.
    """
    from src.eda.preseason_value import attach_season_start_roster

    features_dir = Path(cfg["data"]["features_dir"])
    window_games = int(cfg.get("features", {}).get("team_context", {})
                       .get("roster_window_games", 10))
    rows = attach_season_start_roster(
        pd.read_parquet(features_dir / "preseason.parquet",
                        columns=["season", "player_id"]),
        list(cfg["data"]["seasons"]), cfg["data"]["raw_dir"], window_games)
    return {(s, p) for s, p, on in zip(rows["season"], rows["player_id"],
                                       rows["on_season_start_roster"]) if on > 0}


def season_totals(samples: np.ndarray, frame: pd.DataFrame
                  ) -> tuple[np.ndarray, pd.DataFrame]:
    """Player-season totals per draw, and the unit keys they belong to.

    `minutes_unification.season_totals`' arithmetic, reimplemented on the floor's draws
    rather than imported, because that module rehydrates a fitted head and this one has none
    to rehydrate — importing it would pull the posterior loader into a target that needs no
    posteriors at all.
    """
    # Factorized rather than `np.unique(..., axis=0)`: `season` is a string column, so the
    # two-column array is dtype object and the axis form refuses it. `factorize` also keeps
    # first-appearance order, which makes the unit frame's row order reproducible.
    keys = pd.MultiIndex.from_arrays([frame["player_id"], frame["season"]])
    index, units = pd.factorize(keys, sort=True)
    totals = np.zeros((samples.shape[0], len(units)))
    np.add.at(totals.T, index, samples.T)
    return totals, pd.DataFrame({"player_id": units.get_level_values(0),
                                 "season": units.get_level_values(1)})


def crps_from_samples(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sample CRPS per row — `stan_utils`' estimator, imported at call time."""
    from src.models.stan_utils import crps_from_samples as _crps

    return _crps(samples, y)


def score_arm(train: pd.DataFrame, val: pd.DataFrame, k: float, roster: set,
              route: str = "both", seed: int = SEED) -> list[dict]:
    """One `k`, scored at both units and on both populations.

    The floor's dispersion is fitted on `train` — the fitting half of the same frame, so each
    arm's floor is fitted under its own ordering rather than under the incumbent's. Anything
    else would score a blended offset through a dispersion estimated for a different one.
    """
    floor = FloorComposition(PREDICTIVE_SAMPLES).fit(train)
    samples = floor.predict_samples(val, seed)
    totals, units = season_totals(samples, val)

    realized = (val.groupby(["player_id", "season"], as_index=False)["y"].sum()
                .rename(columns={"y": "realized"}))
    units = units.merge(realized, on=["player_id", "season"], how="left")
    on_roster = np.array([(s, p) in roster for p, s
                          in zip(units["player_id"], units["season"])])
    game_roster = np.array([(s, p) in roster for p, s
                            in zip(val["player_id"], val["season"])])

    # **The team metric exists only on the pooled frame, and that is a fact about the unit
    # rather than a limitation of the code.** `score_samples` reads the team-sum error off
    # contiguous team-game blocks; a draftable subset of a team-game is not a team-game, so
    # asking for the team constraint on it is asking a question with no referent. The
    # marginal columns are computed directly for the restricted population instead.
    pooled = score_samples(samples, val, f"k={k:g}", seed)
    rows = [{"analysis": "floor_arm", "k": k, "route": route, "population": "pooled",
             "unit": "player_game", "n": len(val),
             "crps": pooled["crps_minutes"], "r2": pooled["r2_minutes"],
             "mae": pooled["mae_minutes"], "rho": floor.rho,
             "team_sum_abs_error": pooled["team_sum_abs_error"]}]

    y_game = val["y"].to_numpy(float)
    game_crps = crps_from_samples(samples, y_game)
    game_pred = samples.mean(axis=0)
    for population, game_mask, unit_mask in (
            ("draftable", game_roster, on_roster),
            ("pooled", np.ones(len(val), bool), np.ones(len(units), bool))):
        if population == "draftable":
            y_m, pred_m = y_game[game_mask], game_pred[game_mask]
            ss = float(np.sum((y_m - y_m.mean()) ** 2))
            rows.append({
                "analysis": "floor_arm", "k": k, "route": route,
                "population": population,
                "unit": "player_game", "n": int(game_mask.sum()),
                "crps": float(game_crps[game_mask].mean()),
                "r2": 1.0 - float(np.sum((y_m - pred_m) ** 2)) / ss if ss > 0 else np.nan,
                "mae": float(np.abs(y_m - pred_m).mean()), "rho": floor.rho})
        y_season = units.loc[unit_mask, "realized"].to_numpy(float)
        drawn = totals[:, unit_mask]
        pred = drawn.mean(axis=0)
        ss_tot = float(np.sum((y_season - y_season.mean()) ** 2))
        rows.append({
            "analysis": "floor_arm", "k": k, "route": route, "population": population,
            "unit": "player_season", "n": int(unit_mask.sum()),
            "crps": float(crps_from_samples(drawn, y_season).mean()),
            "r2": 1.0 - float(np.sum((y_season - pred) ** 2)) / ss_tot
            if ss_tot > 0 else np.nan,
            "mae": float(np.abs(y_season - pred).mean()),
            "bias": float((pred - y_season).mean()),
            "predictive_sd": float(drawn.std(axis=0).mean()),
            "rho": floor.rho})
    return rows


def crps_series(samples: np.ndarray, val: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """`(player-game CRPS, player-season CRPS)` for one arm, **indexed by row identity**.

    Returned rather than folded into `score_arm` because a paired interval needs the rows
    themselves and a summary row cannot carry 50,000 of them.

    ⚠️ **Indexed, not positional, and that is load-bearing here in a way it is not in the
    other rounds.** Every arm re-runs `order_frame`, and a blended prior share reorders the
    team-game blocks — so two arms hold the *same* player-games in *different* row orders.
    Subtracting them positionally would pair each player-game against a different one and
    return a plausible interval around a meaningless difference. `paired` aligns on the
    index and asserts the two arms cover the same rows.

    Takes the draws rather than a `(train, val)` pair so that `composition_preseason_fit`
    can hand it a **fitted** head's predictive on the same footing as the floor's. The two
    rounds must measure the same functional of whatever draws they are given, or "the
    increment shrank under the posterior" compares two estimators rather than two heads.
    """
    totals, units = season_totals(samples, val)
    realized = (val.groupby(["player_id", "season"], as_index=False)["y"].sum()
                .rename(columns={"y": "realized"}))
    units = units.merge(realized, on=["player_id", "season"], how="left")
    game = pd.Series(
        crps_from_samples(samples, val["y"].to_numpy(float)),
        index=pd.MultiIndex.from_arrays(
            [val["player_id"], val["season"], val["game_id"]],
            names=["player_id", "season", "game_id"]))
    season = pd.Series(
        crps_from_samples(totals, units["realized"].to_numpy(float)),
        index=pd.MultiIndex.from_arrays([units["player_id"], units["season"]],
                                        names=["player_id", "season"]))
    return game.sort_index(), season.sort_index()


def per_row_crps(train: pd.DataFrame, val: pd.DataFrame, seed: int = SEED
                 ) -> tuple[pd.Series, pd.Series]:
    """`crps_series` for the FLOOR fitted on `train` — this round's only estimator."""
    floor = FloorComposition(PREDICTIVE_SAMPLES).fit(train)
    return crps_series(floor.predict_samples(val, seed), val)


def paired(arm: pd.Series, reference: pd.Series, mask: pd.Index | None = None,
           reps: int = 2000, seed: int = 42) -> tuple[float, float, float]:
    """`(mean difference, lo, hi)` on arm − reference CRPS, aligned by row identity."""
    if not arm.index.equals(reference.index):
        raise AssertionError("the two arms do not cover the same rows — a paired interval "
                             "over them would be meaningless")
    diff = (arm - reference)
    if mask is not None:
        diff = diff[mask]
    diff = diff.to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(reps, len(diff)))
    boot = diff[idx].mean(axis=1)
    return (float(diff.mean()), float(np.percentile(boot, 2.5)),
            float(np.percentile(boot, 97.5)))


def run(cfg: dict) -> dict[str, Path]:
    from src.eda.preseason_value import covered_seasons

    features_dir = Path(cfg["data"]["features_dir"])
    eda_dir = Path(cfg["evaluation"].get("eda_dir", "outputs/eda"))
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    comp_cfg = cfg.get("stan", {}).get("composition", {})
    # **Not the head's own `first_season`, deliberately.** The shipped composition fits from
    # 1996-97 and the preseason panel begins at 2004-05, so inheriting that window would put
    # a structurally absent preseason on a third of the rows — P3's coverage lesson, which
    # cost that round a quarter of its measured increment before it was caught. Session 4b is
    # specified at the PILOT window anyway (`docs/preseason-plan.md` P3 decision 3), and the
    # pilot starts well inside coverage, so the two constraints agree here.
    first_season = str(comp_cfg.get("preseason_first_season", PILOT_FIRST_SEASON))
    test_seasons = int(cfg.get("features", {}).get("availability", {})
                       .get("test_seasons", 2))

    print("Composition preseason — does a preseason reading improve the PRIOR SHARE?")
    print(f"  the head's own no-fit floor, so no CmdStan and no fit of the head.")

    covered = set(covered_seasons(pd.read_csv(eda_dir / "preseason_coverage.csv")))
    if first_season < min(covered):
        raise ValueError(f"the composition window starts at {first_season}, before the "
                         f"preseason panel's first covered season {min(covered)} — every "
                         f"arm would carry a structural zero on the early rows")
    pre = preseason_share(pd.read_parquet(features_dir / "preseason.parquet"))
    roster = season_start_roster(cfg)

    rows: list[dict] = []
    crps_by_arm: dict[tuple[float, str], tuple[pd.Series, pd.Series]] = {}
    inner: list[dict] = []
    for k in SHRINK_GRID:
        frame = frame_at(cfg, pre, k, first_season)
        train, val = selection_split(frame, test_seasons)
        if k == SHRINK_GRID[0]:
            print(f"  window {first_season} on: {len(train):,} fit / {len(val):,} select "
                  f"player-games over "
                  f"{val.groupby(GROUP_KEYS, sort=False).ngroups:,} validation team-games; "
                  f"{len(roster):,} season-start-roster keys")
        # The inner carve — the LAST two fitting seasons scored against everything before
        # them. It never touches validation, which is what lets `k` be chosen at all.
        fit_seasons = sorted(set(train["season"]))
        scored = set(fit_seasons[-INNER_SCORE_SEASONS:])
        inner_fit = train[~train["season"].isin(scored)]
        inner_score = train[train["season"].isin(scored)]
        if not inner_fit.empty and not inner_score.empty:
            game_crps, _ = per_row_crps(inner_fit, inner_score)
            inner.append({"analysis": "inner_grid", "k": k,
                          "inner_crps": float(game_crps.mean()),
                          "n": int(len(inner_score)),
                          "scored_seasons": ", ".join(sorted(scored))})
        rows += score_arm(train, val, k, roster, route="both")
        crps_by_arm[(k, "both")] = per_row_crps(train, val)
        print(f"    k = {k:>10,.0f}  fitted and scored")

    grid = pd.DataFrame(inner)
    chosen = float(grid.loc[grid["inner_crps"].idxmin(), "k"])
    print(f"\n  The inner grid, on the FITTING half ({grid['scored_seasons'].iloc[0]}):")
    pd.set_option("display.width", 220)
    print(grid[["k", "inner_crps", "n"]].round(5).to_string(index=False))
    print(f"  selected k = {chosen:,.0f}"
          + ("  — the INCUMBENT, i.e. the inner carve prefers no preseason at all"
             if chosen >= INCUMBENT_K else ""))

    # ── The attribution, at the SELECTED k only ───────────────────────────────
    # `w_share` reaches the head twice and the blended arm moves both. Two more frames say
    # which route pays; running them at every `k` would be seven times the cost for a
    # question that is only asked of the arm that would ship.
    if chosen < INCUMBENT_K:
        print(f"\n  Attribution at k = {chosen:,.0f} — which of the two routes pays:")
        for route in ("offset_only", "order_only"):
            frame = frame_at(cfg, pre, chosen, first_season, route)
            train, val = selection_split(frame, test_seasons)
            rows += score_arm(train, val, chosen, roster, route=route)
            crps_by_arm[(chosen, route)] = per_row_crps(train, val)
            print(f"    {route} scored")

    reference = crps_by_arm[(INCUMBENT_K, "both")]
    game_roster = pd.Index([(s, p) in roster for p, s, _ in reference[0].index])
    season_roster = pd.Index([(s, p) in roster for p, s in reference[1].index])
    margins = []
    for (k, route), (game, season) in crps_by_arm.items():
        for unit, arm, ref, draft_mask in (
                ("player_game", game, reference[0], game_roster),
                ("player_season", season, reference[1], season_roster)):
            for population, mask in (("pooled", None), ("draftable", draft_mask)):
                point, lo, hi = paired(arm, ref, mask)
                margins.append({
                    "analysis": "floor_margin", "k": k, "route": route, "unit": unit,
                    "population": population,
                    "n": int(len(arm) if mask is None else mask.to_numpy().sum()),
                    "crps_vs_incumbent": point, "crps_vs_incumbent_lo": lo,
                    "crps_vs_incumbent_hi": hi,
                    "selected": bool(np.isclose(k, chosen) and route == "both")})
    table = pd.concat([pd.DataFrame(rows), pd.DataFrame(margins), grid],
                      ignore_index=True)
    dest = out_dir / "composition_preseason.csv"
    table.to_csv(dest, index=False)

    arms = pd.DataFrame(rows)
    for population in ("draftable", "pooled"):
        for unit in ("player_game", "player_season"):
            shown = arms[(arms["population"] == population) & (arms["unit"] == unit)
                         & (arms["route"] == "both")]
            print(f"\n  {unit} · {population}:")
            print(shown[["k", "n", "crps", "r2", "mae", "rho"]]
                  .round(5).to_string(index=False))
    margin_frame = pd.DataFrame(margins)
    for population in ("draftable", "pooled"):
        print(f"\n  Paired against the incumbent (k = {INCUMBENT_K:,.0f}) · {population}:")
        print(margin_frame[(margin_frame["population"] == population)
                           & (margin_frame["route"] == "both")]
              [["k", "unit", "n", "crps_vs_incumbent", "crps_vs_incumbent_lo",
                "crps_vs_incumbent_hi", "selected"]].round(5).to_string(index=False))
    attribution = margin_frame[margin_frame["route"] != "both"]
    if not attribution.empty:
        print(f"\n  Attribution at k = {chosen:,.0f} — the blended arm moves BOTH routes, "
              f"and these move one each:")
        print(attribution[["route", "unit", "population", "crps_vs_incumbent",
                           "crps_vs_incumbent_lo", "crps_vs_incumbent_hi"]]
              .round(5).to_string(index=False))
    print(f"\nWrote {len(table):,} rows → {dest}")
    return {"composition_preseason": dest}


if __name__ == "__main__":
    from src.models.composition_preseason import run as _run

    _run(yaml.safe_load(open("configs/default.yaml")))
