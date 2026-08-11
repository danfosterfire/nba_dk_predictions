"""The season simulator — one tensor, and Gate A.

`docs/simulations-plan.md` ("The output contract that collapses the compute problem") fixes
what this module owes everything downstream:

```
sim_tensor[player, scoring_period, sim]   -> dk_pts         float32
games_played[player, scoring_period, sim] -> games played   uint8
```

Twenty scoring periods — Round 1's seventeen weeks plus three double weeks — so at ~450
draftable players and 2,000 sims the pair is ~90 MB, small enough to hold for a whole
strategy sweep and to load into a draft room in under a second. **Per-game draws happen
inside this module and are summed into periods immediately**, because the bonus is a
simultaneous threshold on five components and `E[bonus] != bonus(E[x])`. Nothing outside
this module ever materializes a `player x game x sim` array.

The bonus is a **staircase, not a single step**: +1.5 for a double-double and a further +3
for a triple-double, which stack to 4.5, capped at one of each per player per game. So the
threshold is crossed twice and the convexity is sharper than a double-double alone —
`preprocess.compute_dk_pts` carries the whole schedule (`{2: 1.5, 3: 4.5, 4: 4.5, 5: 4.5}`)
and this module never re-implements it.

## This is assembly. Every piece is fitted and measured elsewhere

| step | source |
|---|---|
| how long the game is | `stan_game_length.sample_game_length`, **once per game, shared by both teams** |
| which games he plays | the availability head's count, laid out by `games_played.allocate_spells` |
| minutes, team-constrained | `stan_composition.simulate_minutes`, verbatim |
| minutes, season-level spread | the per-(player, season) effect at `sim.minutes.player_season_sigma` |
| the eleven component heads | here, at the per-game unit — see "Why the draw is written here" |
| cross-component dependence | `residual_correlation.csv`, the **minutes-conditioned** matrix |
| the bonus and the score | `preprocess.compute_dk_pts` on the drawn integer box score |

## The four rules, and where each one lives

1. **Draw, never plug in.** Game length, games played, minutes, every component and
   therefore the bonus are all sampled. The bonus is not `expected_bonus` evaluated at a
   mean — it falls out of `compute_dk_pts` on a drawn integer box score, which is the
   strongest available form of the rule.
2. **One shared `min` draw per player-game feeds all eleven heads.** `_sim_one` draws
   minutes once and hands the same array to every count head as exposure and, through the
   attempt counts, to every conversion head as trials.
3. **Sequential structure goes on minutes and nowhere else.** Both field-goal conversion
   heads are measured nulls for a hot hand (`serial_correlation.csv`, ten-game block
   inflation 1.035x and 1.007x), so their per-game draw is binomial around a season-level
   rate.
4. **The posterior draw is the OUTER loop.** Sim `s` uses posterior draw `s % n_draws` for
   *every* player and every head. Players share `beta`, so one draw moves the whole board
   together; resampling per player would destroy exactly the cross-player correlation this
   layer exists to capture. It is also why `stan_games_played.HybridProcess.sequences` is
   **not** called: it draws its count from a pmf already marginalized over the posterior,
   which is the right thing for a marginal metric and the wrong thing here.

## Why the component draw is written here rather than reused

`season_terms._draw_components` already materializes the chain
`fga -> fg3a|fga -> fg2a = fga - fg3a -> makes`, and `DRAW_ORDER` below is that order made
explicit and test-pinned. What it cannot do is draw at the *player-game* unit: it draws one
negative binomial per player-**season**, which is the unit its heads were fitted at. The
simulator needs the same posterior expressed one level down, and the identity that gets it
there is the negative binomial's own Poisson-Gamma representation:

    y_season ~ NB(M*rate, phi)   ==   Poisson(M*rate*G),  G ~ Gamma(phi, 1/phi)

So a **season-level** frailty is drawn once per (player, sim) — that is the fitted head's
season-total spread, exactly — and per-game counts are Poisson around `rate*minutes_g*G`,
times a **per-game** frailty carrying the measured player-game overdispersion of 0.025
(`targets.BONUS_GAME_OVERDISPERSION`). The two do not meaningfully double-count: summed
over a season the season term contributes `(1/phi)*(M*rate)^2` and the game term
`0.025*sum(mu_g^2)`, which at a typical head is ~1.6% of it.

The conversion heads factorize the same way and more simply: draw `p` once per
(player, sim) from `Beta(a, b)` at the head's fitted `(mu, rho)`, then
`Binomial(attempts_g, p)` per game. Summed over the season that is **exactly**
`BetaBinomial(sum attempts, mu, rho)` — the fitted head, with no approximation — and it is
also what rule 3 asks for.

## The copula, and the one cell it does not carry

The per-game frailties are correlated across the seven counts by the **count block** of
`residual_correlation.csv`'s minutes-conditioned matrix, imposed as a Gaussian copula on
lognormal frailties (mean 1, variance 0.025). Lognormal rather than the Gamma
`expected_bonus` uses, because only the lognormal takes a Gaussian copula for free —
correlating the underlying normals *is* correlating the frailties, at one `exp` and one
matrix product. That is where the measured per-game dependence lives: the largest cell in
the whole matrix is `fga`-`reb` at +0.133, and the seven counts average +0.023 against
+0.007 over all eleven.

**The conversion rows of that matrix are not imposed, and it is a deliberate, measured
trade rather than an oversight.** Three of the four conversion heads have no per-game
structure to correlate — block inflations 1.035, 1.007 and 1.103, the measured nulls rule 3
rests on — so a season-level `p` is the right factorization and there is no per-game
residual left to couple. The exception is `fg3a | fga`, whose block inflation is **1.575**:
the three-point *mix* really does move game to game, and its largest cell is -0.083 against
`fga`. That coupling is not carried here. `gate_a` measures the simulated correlation
matrix against the target and reports the gap rather than asserting it away.

## Gate A

The simulator must reproduce the **marginals it was handed**. Every input head is already
calibrated, so a miss is a wiring fault rather than a modelling one. Four comparisons, each
with its own artifact and bar, reported individually and never as "close":

- season-total dk_pts against `season_total_metrics.csv` (`beta_binomial`);
- the games-played pmf against `stan_games_played_gp_pmf.csv`, and CRPS in games against
  the availability head's own **10.0057**;
- per-game bonus rate against `bonus_calibration.csv`'s realized player-game figure;
- season-total minutes predictive sd against `minutes_unification.csv`'s **302.75**.

Three diagnostics ride alongside and are reported rather than gated: the game-level minutes
dispersion against `stan_minutes_dispersion.csv`'s 4.65x (a *diagnostic* by decision — see
"The third prerequisite"), the ten-game block inflation against `serial_correlation.csv`'s
2.43x, and the simulated cross-component correlation against the copula's own target.

**The season total is scored over the games inside the tournament window**, not the whole
schedule: DK's Round 4 closes before the NBA season does, so the tensor carries ~20 of ~24
weeks and comparing its total against a full-season realized figure would report a
shortfall that is a calendar fact rather than a model error. Games played is scored over
the **whole** schedule, because that is the availability head's own denominator.

## The split

Selection reads validation and nothing else. `allowed_seasons` derives the legal set
through `held_out.selection_split`, and a request for a test season goes through
`held_out.assert_unlocked` rather than being silently honoured. The posterior artifacts are
read at the `train` window and `posteriors.require_window` refuses anything wider, because
a head fitted on `train_val` has read 2022-23 and 2023-24 *through the coefficients* and no
frame-level guard can see that.

Usage:
    python -m src.sim.season
    python -m src.sim.season --season 2022-23 --n-sims 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.data.preprocess import compute_dk_pts
from src.eda.residual_correlation import MINUTES_CONDITIONED
from src.eda.residual_correlation import to_matrix as residual_matrix
from src.features.targets import BONUS_GAME_OVERDISPERSION, bonus_part
from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, DERIVED_COUNTS,
                                        build_design as component_build_design)
from src.models.games_played import allocate_spells
from src.models.held_out import TEST_SEASONS, assert_unlocked, selection_split
from src.models.minutes_unification import rehydrate_composition, shipped_sigma
from src.models.posteriors import load_all, posteriors_dir, require_window
from src.models.stan_availability import availability_design
from src.models.stan_composition import OFFSET_CLIP, composition_frame, simulate_minutes
from src.models.stan_game_length import (forward_cells, posterior_inputs,
                                         sample_game_length)
from src.models.stan_minutes import beta_shapes
from src.models.stan_utils import crps_from_samples

# ── The output contract ───────────────────────────────────────────────────────

# Round 1's 17 weekly periods, then one slot per double week. `scoring_periods.parquet`
# owns the week grid and the round map; this is only how the two collapse onto the tensor's
# second axis.
ROUND_1_WEEKS = 17
N_SCORING_PERIODS = 20

# `sim.n_sims` / `sim.seed` if config is silent.
N_SIMS = 2000
SEED = 0

# The window whose heads may score 2022-23 / 2023-24. Not a preference — `train_val` fits
# on exactly those seasons. `minutes_unification` carries the same constant for the same
# reason.
FIT_WINDOW = "train"

# A team-game allocates `5 x game_length` minutes with a per-player cap of `game_length`,
# so it needs at least five available players or the allocation has no solution.
MIN_AVAILABLE = 5

# Realized minutes below which a player-season's predictive spread says nothing. The
# project's standing qualification threshold, reused rather than re-chosen.
MIN_QUALIFIED_MINUTES = 200

# Block length for the serial-dependence diagnostic. `serial_correlation.csv` measures the
# ten-game block variance inflation, so the check has to use the same block.
SERIAL_BLOCK = 10

# The scoring columns `compute_dk_pts` reads. `min` and the attempt counts matter solely
# through the exposure and trials they supply to these.
SCORING_COLUMNS = ("pts", "fg3m", "reb", "ast", "stl", "blk", "tov")


# ── The chain ─────────────────────────────────────────────────────────────────

def artifact_name(head: str) -> str:
    """`fg3a|fga` -> `fg3a_given_fga`. A filename cannot carry the pipe the heads use."""
    return head.replace("|", "_given_")


def draw_order() -> tuple[str, ...]:
    """The materialization order of the chain, derived from the head lists.

    Derived rather than written down, so a head-list change is caught here instead of
    producing a plausible season from the wrong conditional. The whole content of the
    shot-attempt basis is in the insertion step: a conversion head's own draw becomes a
    later head's trials, so `fg2a = fga - fg3a` cannot be materialized until the share head
    has drawn. `season_terms._draw_components` records the same order at the season unit;
    `tests/test_sim_season.py` fails loudly if either is reordered.
    """
    order: list[str] = list(COUNT_HEADS)
    have = set(COUNT_HEADS)
    for made, attempted in CONVERSION_HEADS:
        for name, (total, part) in DERIVED_COUNTS.items():
            if name == attempted and name not in have and {total, part} <= have:
                order.append(name)
                have.add(name)
        if attempted not in have:
            raise ValueError(
                f"{made}|{attempted} needs {attempted} as trials and nothing earlier in "
                f"the chain produces it — the head lists and DERIVED_COUNTS disagree")
        order.append(f"{made}|{attempted}")
        have.add(made)
    return tuple(order)


DRAW_ORDER = draw_order()


# ── Frames ────────────────────────────────────────────────────────────────────

def allowed_seasons(design: pd.DataFrame) -> list[str]:
    """Target seasons selection may read: train plus validation, never test.

    Routed through `held_out.selection_split` rather than re-derived, so this layer sits
    behind the same choke point every model head goes through.
    """
    train, val = selection_split(design, TEST_SEASONS)
    return sorted(set(train["season"]) | set(val["season"]))


def validation_seasons(design: pd.DataFrame) -> list[str]:
    """The two seasons the realized backtest scores — `make simulate-season`'s default."""
    return sorted(selection_split(design, TEST_SEASONS)[1]["season"].unique())


def assert_season_allowed(season: str, design: pd.DataFrame) -> None:
    """Refuse a test season unless something has explicitly unlocked the split."""
    if season not in allowed_seasons(design):
        assert_unlocked(f"simulating {season}")


def scoring_slots(features_dir: Path, season: str) -> pd.DataFrame:
    """`game_id -> slot` in `[0, 20)` for games inside the tournament, `-1` outside it.

    Round 1's seventeen weeks keep a slot each; each of the three double weeks collapses to
    one. `tournament_round == 0` is the tail of the NBA season DK's Round 4 closes before —
    two unscored weeks in 2025-26 — and those games still happen, so they keep a row and
    lose only their slot.
    """
    periods = pd.read_parquet(features_dir / "scoring_periods.parquet")
    periods = periods[periods["season"] == season]
    if periods.empty:
        raise ValueError(f"no scoring periods for {season}; run `make scoring-periods`")

    inside = periods["tournament_round"] > 0
    first_week = int(periods.loc[periods["tournament_round"] == 1, "period_index"].min())
    slot = np.where(periods["tournament_round"] == 1,
                    periods["period_index"] - first_week,
                    ROUND_1_WEEKS + periods["tournament_round"] - 2)
    out = periods[["game_id", "tournament_round"]].copy()
    out["slot"] = np.where(inside, slot, -1).astype(int)
    if out["slot"].max() >= N_SCORING_PERIODS:
        raise ValueError(f"{season} maps to slot {out['slot'].max()}, outside "
                         f"[0, {N_SCORING_PERIODS})")
    return out


def roster_grid(features_dir: Path, season: str, slots: pd.DataFrame) -> pd.DataFrame:
    """One row per (team, team-game, rostered player) — the frame a season is drawn on.

    `availability_panel.parquet` is the source rather than the game logs, for the reason
    `stan_composition.played_frame` gives: the logs are `min_games`-filtered and a team sum
    over a filtered frame is missing whole players' minutes. The panel carries every
    rostered player for every one of his team's games whether he played or not, which is
    exactly the grid a forward season needs.

    **Every game is kept, including the ones outside DK's window.** Availability is drawn
    against the head's own denominator — the full schedule — and only the *scoring* step
    drops games with no slot.

    **A traded player is attributed wholly to his last team**, which is
    `features.availability.season_availability`'s own convention and not a simplification
    invented here. It has to be, because the availability head's `team_games` is that
    team's schedule length: taking the panel as it stands gives a traded player rows on
    both teams and a denominator of **92.6** games against the head's 82.0, so every
    simulated season would have run the head's rate against ~13% too many opportunities.
    Reproducing the convention puts the two within one game on 432 of 433 players.
    Mid-season trades themselves are still not modelled — a known limit inherited from the
    prediction layer.
    """
    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "game_date",
                 "team_game_index", "played"])
    grid = panel[panel["season"] == season].copy()
    if grid.empty:
        raise ValueError(f"the availability panel carries no rows for {season}")
    last = (grid[grid["played"] == 1].sort_values(["game_date", "game_id"])
            .groupby(["season", "player_id"], as_index=False)
            .agg(team_id=("team_id", "last")))
    grid = grid.merge(last, on=["season", "player_id", "team_id"], how="inner")
    lookup = dict(zip(slots["game_id"], slots["slot"]))
    grid["slot"] = grid["game_id"].map(lookup).fillna(-1).astype(int)
    return grid.drop(columns=["played", "game_date"]).reset_index(drop=True)


def no_design_availability(features_dir: Path, season: str, design: pd.DataFrame,
                           allowed: list[str]) -> float:
    """Availability rate for rostered players the availability head has no row for.

    The head is a lag-1 design, so a player with no prior season is not in its frame at all
    — 106 of 539 rostered players in 2022-23. They still consume roster spots and, because
    the minutes allocation is zero-sum, whatever availability they are given comes straight
    out of their teammates' minutes. Scoring them at the head's **intercept** was the first
    version of this and it is badly wrong in a measurable direction: it puts them at 58.4
    simulated games against a realized 30.1, and moves ~29,500 minutes a season away from
    the players the tensor scores.

    So they get the empirical rate of *players like them*, on the same expanding-window,
    point-in-time construction `stan_composition.rookie_share_priors` already uses for their
    minutes share: the pooled `gp / team_games` of no-design player-seasons in the seasons
    strictly **before** the target, intersected with the seasons selection may read. Nothing
    is fitted and nothing from the target season enters.
    """
    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "game_date", "played"])
    earlier = [s for s in allowed if s < season]
    panel = panel[panel["season"].isin(earlier)]
    if panel.empty:
        raise ValueError(f"no seasons before {season} to estimate a no-design "
                         f"availability rate from")
    last = (panel[panel["played"] == 1].sort_values(["game_date", "game_id"])
            .groupby(["season", "player_id"], as_index=False)
            .agg(team_id=("team_id", "last")))
    primary = panel.merge(last, on=["season", "player_id", "team_id"], how="inner")
    cells = (primary.groupby(["season", "player_id"], as_index=False)
             .agg(gp=("played", "sum"), team_games=("played", "size")))
    covered = set(map(tuple, design[["season", "player_id"]].to_numpy()))
    keys = list(map(tuple, cells[["season", "player_id"]].to_numpy()))
    rows = cells[[k not in covered for k in keys]]
    if rows.empty:
        raise ValueError("no uncovered player-seasons before "
                         f"{season}; the availability design cannot be that complete")
    return float(rows["gp"].sum() / rows["team_games"].sum())


def composition_players(frame: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per player in `season`, carrying the composition head's design columns.

    `frame` is `stan_composition.composition_frame`'s output, built over every season
    because its lags and its expanding rookie prior need the history. Every column it
    carries is constant within a player-season except the three the simulator supplies per
    game — `n_overtimes`, its interaction with the own share, and `offset_clipped` — so one
    row per player is the whole design.
    """
    rows = frame[frame["season"] == season]
    if rows.empty:
        raise ValueError(f"the composition frame carries no rows for {season}")
    players = rows.drop_duplicates(subset=["player_id"]).reset_index(drop=True)
    players["n_overtimes"] = 0.0
    players["offset_clipped"] = 0.0
    return players


# ── Per-player posterior blocks ───────────────────────────────────────────────

def component_rates(artifacts: dict, frame: pd.DataFrame) -> dict:
    """`(draws x players)` per-minute rates, conversion probabilities and dispersions.

    Every feature in every component head is a lag-1 or bio column, so the linear predictor
    is **constant within a player-season** — which is what makes a per-game simulation
    affordable at all: the design is evaluated once per player and a game's mean is that
    rate times the minutes drawn for the game.
    """
    out: dict = {"count": {}, "conversion": {}, "phi": {}, "rho": {}}
    for component in COUNT_HEADS:
        art = artifacts[artifact_name(component)]
        # `mu_draws` on a negative-binomial head is `exp(eta)` before exposure — a rate per
        # minute, which is what a per-game draw multiplies by that game's minutes.
        out["count"][component] = art.mu_draws(frame)
        out["phi"][component] = np.asarray(art.draws["phi_draws"], dtype=float)
    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        art = artifacts[artifact_name(head)]
        out["conversion"][head] = art.mu_draws(frame)
        out["rho"][head] = np.asarray(art.draws["rho_draws"], dtype=float)
    return out


def composition_eta(artifact, model, players: pd.DataFrame) -> dict:
    """The composition's linear predictor, split into a base and two per-game slopes.

    `eta` is affine in the design and the design is affine in `n_overtimes` and
    `offset_clipped`, so evaluating the recipe at three corners recovers the whole surface
    exactly:

        eta(k, c) = eta(0,0) + k*[eta(1,0) - eta(0,0)] + c*[eta(0,1) - eta(0,0)]

    Taking the slopes by difference rather than by reaching into the fitted scaler is what
    keeps this correct when the recipe changes shape: `ot_x_own` is a *product* step, so the
    overtime slope is per-player, and a hand-written derivative would have to know that.
    Affinity is asserted at a fourth corner rather than assumed.
    """
    def eta_at(n_ot: float, clipped: float) -> np.ndarray:
        frame = players.copy()
        frame["n_overtimes"] = float(n_ot)
        frame["offset_clipped"] = float(clipped)
        X = artifact.recipe.matrix(frame)
        return X @ model.beta_draws.T + model.alpha_draws[None, :]

    base = eta_at(0.0, 0.0)
    per_ot = eta_at(1.0, 0.0) - base
    per_clip = eta_at(0.0, 1.0) - base
    error = float(np.max(np.abs(eta_at(2.0, 0.0) - base - 2.0 * per_ot)))
    if error > 1e-8:
        raise AssertionError(
            f"the composition's linear predictor is not affine in `n_overtimes` (max "
            f"error {error:.2e}); the overtime slope cannot be taken by difference")
    return {"base": base, "per_overtime": per_ot, "per_clip": per_clip,
            "affine_error": error}


def spell_shape(artifact, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(mu, kappa)` per posterior draw for the absence-spell length.

    Pooled over this season's players, because `games_played.allocate_spells` — the shipped
    hybrid's layout step, reused verbatim — takes a scalar pair. The shipped hybrid plugs in
    a single closed-form `fit_beta_geometric` estimate; this integrates over the posterior
    instead, one `(mu, kappa)` per draw, which is the same upgrade `make posteriors` bought
    everywhere else.
    """
    lo, hi = artifact.extras.get("mu_clip", (1e-6, 1 - 1e-6))
    mu = np.clip(artifact.mu_draws(frame), float(lo), float(hi)).mean(axis=1)
    return mu, np.asarray(artifact.draws["kappa_draws"], dtype=float)


def _mean_expected(mu: np.ndarray, floor: float = 0.5) -> float:
    """Mean expected per-game count over the rows a residual is measured on.

    `serial_correlation.count_residuals` drops rows whose expected count is at or below
    `MIN_EXPECTED`, because a Pearson residual on an expectation of 0.2 is dominated by
    discreteness. The pooled correlation is therefore taken over the *surviving* rows, so
    the mean that inverts it has to be too — otherwise `blk` would come in at a league mean
    of 0.5 when every row it was measured on is above that.
    """
    kept = mu[mu > floor]
    return float(kept.mean()) if kept.size else float(mu.mean())


def nearest_correlation(R: np.ndarray) -> np.ndarray:
    """The nearest positive semi-definite correlation matrix, by eigenvalue clipping.

    One projection step, not an iterative Higham solve: the input here is a moment-matched
    matrix whose off-diagonals are already inside [-1, 1] and whose indefiniteness is small,
    so the extra accuracy would not change a draw.
    """
    values, vectors = np.linalg.eigh(R)
    out = vectors @ np.diag(np.clip(values, 1e-8, None)) @ vectors.T
    d = np.sqrt(np.diag(out))
    return out / np.outer(d, d)


def count_copula(long: pd.DataFrame, window: str, mean_counts: np.ndarray,
                 overdispersion: float = BONUS_GAME_OVERDISPERSION
                 ) -> dict:
    """The per-game count frailties' correlation, moment-matched to the residual matrix.

    The frailty is lognormal with mean 1 and variance `overdispersion`, so a count's
    per-game marginal is `Poisson(mu*g)` with `Var = mu + overdispersion*mu^2` — the
    measured player-game overdispersion, calibrated by `make component-targets` to zero bias
    on the realized per-game bonus rate.

    **The frailty correlation is not the residual correlation, and conflating them was the
    first version of this.** `residual_correlation.csv` measures the correlation of *Pearson
    residuals*, and under this model

        corr(resid_a, resid_b) = R_ab * v * sqrt(mu_a*mu_b)
                                 / sqrt((1 + v*mu_a) * (1 + v*mu_b))

    so at `v = 0.025` and typical per-game means the residual correlation is roughly a
    **tenth** of the frailty correlation that produces it. Handing the copula the residual
    matrix directly therefore imposed about a tenth of the measured dependence — every cell
    present, every shape right, only the numbers wrong. This inverts the relation at the
    population mean per-game count, which is where the pooled measurement is taken, and then
    projects the result back to a valid correlation matrix.

    That projection is load-bearing rather than cosmetic: the inversion asks for
    off-diagonals above 1 on the pairs the measured correlation is largest for, which is
    itself a finding — **the measured residual coupling is at the ceiling a frailty of this
    variance can produce**, so the two artifacts are close to two views of one per-game "big
    night" factor. `achieved` reports what survives the projection, and Gate A measures the
    simulated matrix against the target rather than assuming this arithmetic held.
    """
    heads = list(COUNT_HEADS)
    target = residual_matrix(long, MINUTES_CONDITIONED, window).loc[heads, heads] \
        .to_numpy(dtype=float)
    mu = np.asarray(mean_counts, dtype=float)
    v = float(overdispersion)
    scale = v * np.sqrt(np.outer(mu, mu)) / np.sqrt(np.outer(1 + v * mu, 1 + v * mu))
    implied = np.clip(np.divide(target, scale, out=np.zeros_like(target),
                                where=scale > 0), -0.999, 0.999)
    np.fill_diagonal(implied, 1.0)
    R = nearest_correlation(implied)
    off = ~np.eye(len(heads), dtype=bool)
    return {"chol": np.linalg.cholesky(R), "target": target, "frailty": R,
            "implied": implied, "mean_counts": mu,
            "sigma": float(np.sqrt(np.log1p(v))),
            "achieved": R * scale,
            "saturated": int((np.abs(implied[off]) >= 0.999).sum() // 2)}


# ── One simulated season ──────────────────────────────────────────────────────

def feasibility_repair(available: np.ndarray, block_id: np.ndarray, n_blocks: int,
                       needed: int = MIN_AVAILABLE) -> tuple[np.ndarray, int]:
    """Guarantee every team-game can allocate `5 x game_length` minutes under the cap.

    The pot is `N = 5*game_length` and the per-player cap is `U = game_length`, so a
    team-game needs at least five available players or the allocation has no solution and
    `simulate_minutes` raises — the right behaviour, and the wrong outcome for a simulator.
    Real rotations never go below eight, so this fires only on the tail of the availability
    draw; the highest-ranked absentees are promoted (rows are already in rotation order) and
    the count is carried into the artifact rather than swallowed.
    """
    counts = np.bincount(block_id[available], minlength=n_blocks)
    short = np.flatnonzero(counts < needed)
    if not len(short):
        return available, 0
    repaired = available.copy()
    for block in short:
        rows = np.flatnonzero((block_id == block) & ~available)
        repaired[rows[:needed - counts[block]]] = True
    return repaired, int(len(short))


def block_bounds(block_id: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Start index and length of each contiguous run in a sorted block-id array."""
    starts = np.flatnonzero(np.r_[True, block_id[1:] != block_id[:-1]])
    return starts, np.diff(np.r_[starts, len(block_id)])


def suffix_sums(values: np.ndarray, starts: np.ndarray,
                lens: np.ndarray) -> np.ndarray:
    """Sum from each row to the end of its block — the stick-breaking tail mass.

    `stan_composition.sequential_columns` computes the same quantity with a groupby; this
    is the numpy form, because the simulator rebuilds it per simulated season.
    """
    total = np.repeat(np.add.reduceat(values, starts), lens)
    prefix = np.cumsum(values) - values
    return total - (prefix - np.repeat(prefix[starts], lens))


def offset_clipped(ratio: np.ndarray, w: np.ndarray, starts: np.ndarray, lens: np.ndarray,
                   n_total: np.ndarray, caps: np.ndarray) -> np.ndarray:
    """Whether the proportional carry-forward saturates, at the *expected* allocation.

    `offset_clipped` is a design column the fitted head reads off the realized allocation
    (`stan_composition.sequential_columns`), and a forward season has none — so it is
    evaluated at the deterministic proportional allocation, which is exactly what the column
    means: the carry-forward demands more of this player than the cap allows. It marks ~1%
    of rows and carries a correction coefficient, so the substitution is second order. It is
    named here rather than buried because it is the one design column the simulator supplies
    from an expectation rather than from a draw.
    """
    share = w / np.repeat(np.add.reduceat(w, starts), lens)
    expected = np.minimum(share * n_total, caps)
    before = np.cumsum(expected) - expected
    before = before - np.repeat(before[starts], lens)
    remaining = np.maximum(n_total - before, 0.0)
    trials = np.minimum(caps, remaining)
    raw = ratio * remaining / np.maximum(trials, 1.0)
    return (raw != np.clip(raw, *OFFSET_CLIP)).astype(float)


def draw_components(rng: np.random.Generator, minutes: np.ndarray, unit: np.ndarray,
                    rates: dict, draw: int, season_gamma: dict, season_p: dict,
                    frailty: np.ndarray) -> tuple[dict, dict, list[str]]:
    """Per-game counts and makes for one simulated season, in `DRAW_ORDER`.

    `minutes` is the **one shared draw** rule 2 names: the exposure for all seven counts
    and, through the attempt counts they produce, the trials for all four conversions.
    `trace` records what was materialized and in what order, so the chain is checkable from
    outside rather than asserted in a comment.
    """
    counts: dict = {}
    made: dict = {}
    trace: list[str] = []

    for i, component in enumerate(COUNT_HEADS):
        mu = rates["count"][component][draw][unit] * minutes
        counts[component] = rng.poisson(
            np.clip(mu * season_gamma[component] * frailty[:, i], 0.0, None)
        ).astype(np.float64)
        trace.append(component)

    def trials_for(name: str) -> np.ndarray:
        if name in counts:
            return counts[name]
        if name in DERIVED_COUNTS:
            total, part = DERIVED_COUNTS[name]
            counts[name] = np.maximum(trials_for(total) - trials_for(part), 0.0)
            trace.append(name)
            return counts[name]
        raise KeyError(
            f"{name} is needed as trials but is neither a fitted count head nor a member "
            f"of DERIVED_COUNTS — the head lists and the draw order disagree")

    for m, attempted in CONVERSION_HEADS:
        n = np.rint(trials_for(attempted)).astype(np.int64)
        made[m] = rng.binomial(n, season_p[f"{m}|{attempted}"]).astype(np.float64)
        counts[m] = made[m]
        trace.append(f"{m}|{attempted}")
    return counts, made, trace


def _sim_one(s: int, ctx: dict) -> dict:
    """One simulated season: game lengths, availability, minutes, components, dk_pts.

    The posterior draw is `s % n_draws` and is used by every head for every player, which is
    rule 4 — that shared `beta` sweep is the cross-player correlation the layer exists for.
    """
    draw = s % ctx["n_draws"]
    rng = np.random.default_rng([ctx["seed"], s])

    # ── game length: one draw per GAME, shared by both teams ─────────────────
    lengths = sample_game_length(rng, ctx["n_games"], ctx["length_draws"], draw=draw)
    row_length = lengths[ctx["row_game"]].astype(float)
    row_overtimes = (row_length - 48.0) / 5.0

    # ── availability: one beta-binomial rate per player, a binomial per stint ─
    a, b = beta_shapes(ctx["avail_mu"][draw],
                       np.full(ctx["n_players"], ctx["avail_rho"][draw]))
    p_available = rng.beta(a, b)
    gp = rng.binomial(ctx["cell_games"], p_available[ctx["cell_player"]])
    played = allocate_spells(gp, ctx["cell_games"], float(ctx["dur_mu"][draw]),
                             float(ctx["dur_kappa"][draw]),
                             seed=int(rng.integers(1 << 31)))
    available = played[ctx["row_cell"], ctx["row_index_in_cell"]] == 1
    available, short = feasibility_repair(available, ctx["row_block"], ctx["n_blocks"])

    # ── minutes: the composition head's own sequential allocation ────────────
    rows = np.flatnonzero(available)
    starts, lens = block_bounds(ctx["row_block"][rows])
    position = np.arange(len(rows)) - np.repeat(starts, lens)
    w = ctx["row_w"][rows]
    ratio = w / np.maximum(suffix_sums(w, starts, lens), 1e-12)
    n_total = np.rint(5.0 * row_length[rows])
    caps = np.rint(row_length[rows])
    clipped = offset_clipped(ratio, w, starts, lens, n_total, caps)

    player = ctx["row_player"][rows]
    # One `z` per (player, season) per posterior draw, shared across that player's games —
    # `PlayerSeasonTerm.shift`'s predictive, which is what supplies the season-level spread
    # the composition alone cannot produce (`make minutes-unification`).
    z = rng.standard_normal(ctx["n_players"])
    eta = (ctx["eta_base"][player, draw]
           + row_overtimes[rows] * ctx["eta_per_overtime"][player, draw]
           + clipped * ctx["eta_per_clip"][player, draw]
           + ctx["ps_sigma"][draw] * z[player])

    frame = pd.DataFrame({
        "position": position, "k_players": np.repeat(lens, lens),
        "N": n_total.astype(np.int64), "U": caps.astype(np.int64),
        "stick_ratio": ratio, "rho_bin": ctx["row_rho_bin"][rows]})
    minutes = simulate_minutes(frame, eta[:, None], ctx["comp_rho"][draw][None, :],
                               seed=int(rng.integers(1 << 31)))[0]

    # ── the eleven component heads, on that one minutes draw ─────────────────
    idx = np.flatnonzero(ctx["row_scorable"][rows] & (minutes > 0))
    unit = ctx["component_row"][player[idx]]
    exposure = minutes[idx]

    season_gamma = {c: rng.gamma(ctx["phi"][c][draw], 1.0 / ctx["phi"][c][draw],
                                 size=ctx["n_units"])[unit] for c in COUNT_HEADS}
    season_p = {}
    for m, att in CONVERSION_HEADS:
        head = f"{m}|{att}"
        pa, pb = beta_shapes(ctx["rates"]["conversion"][head][draw],
                            np.full(ctx["n_units"], ctx["rho"][head][draw]))
        season_p[head] = rng.beta(pa, pb)[unit]

    sigma = ctx["frailty_sigma"]
    normals = rng.standard_normal((len(idx), len(COUNT_HEADS))) @ ctx["chol"].T
    frailty = np.exp(sigma * normals - 0.5 * sigma ** 2)

    counts, made, trace = draw_components(rng, exposure, unit, ctx["rates"], draw,
                                          season_gamma, season_p, frailty)
    if trace != list(DRAW_ORDER):
        raise AssertionError(
            f"the component chain materialized as {trace}, not {list(DRAW_ORDER)}")

    box = pd.DataFrame({
        "pts": 2 * made["fg2m"] + 3 * made["fg3m"] + made["ftm"],
        "fg3m": made["fg3m"], "reb": counts["reb"], "ast": counts["ast"],
        "stl": counts["stl"], "blk": counts["blk"], "tov": counts["tov"]})
    dk = compute_dk_pts(box).to_numpy(dtype=float)

    scored = ctx["row_slot"][rows][idx] >= 0
    flat = ctx["row_flat"][rows][idx][scored]
    return {
        "points": np.bincount(flat, weights=dk[scored],
                              minlength=ctx["n_flat"]).astype(np.float32),
        "games": np.bincount(flat, minlength=ctx["n_flat"]).astype(np.uint8),
        "season_minutes": np.bincount(unit, weights=exposure, minlength=ctx["n_units"]),
        "season_gp": np.bincount(ctx["cell_player"], weights=gp,
                                 minlength=ctx["n_players"]),
        "bonus_total": float(bonus_part(box).sum()),
        "n_played": int(len(idx)),
        "short_blocks": short,
        "trace": trace,
        "probe": ({"unit": unit, "minutes": exposure, "length": row_length[rows][idx],
                   "counts": {c: counts[c] for c in COUNT_HEADS},
                   "game": ctx["row_game"][rows][idx]} if s == 0 else None),
    }


# ── Context ───────────────────────────────────────────────────────────────────

def build_context(cfg: dict, season: str, window: str, n_sims: int, seed: int,
                  composition: pd.DataFrame | None = None) -> dict:
    """Everything the per-sim loop reads, built once. No draws happen here."""
    features_dir = Path(cfg["data"]["features_dir"])

    artifacts = load_all(posteriors_dir(cfg, window))
    require_window(artifacts, window)

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = component_build_design(targets, cfg["data"]["seasons"],
                                    cfg["data"]["raw_dir"])
    assert_season_allowed(season, design)

    slots = scoring_slots(features_dir, season)
    grid = roster_grid(features_dir, season, slots)
    frame = composition_frame(cfg) if composition is None else composition
    players = composition_players(frame, season)

    # A scorable unit needs a component-head design row AND a place in the allocation, so
    # the two frames are intersected once here rather than in three places downstream.
    units = design[(design["season"] == season)
                   & design["player_id"].isin(set(players["player_id"]))]
    units = units.reset_index(drop=True)

    # The grid is cut to players the composition can weight, because the allocation is
    # zero-sum: a player left out of a team-game hands his minutes to his teammates.
    grid = grid[grid["player_id"].isin(set(players["player_id"]))]

    # Rotation order, fixed within a team-season and computable preseason — veterans by
    # prior share descending, then no-prior players by draft slot. `order_frame`'s key,
    # applied once so each game's available subset inherits it.
    order = players.sort_values(["no_prior", "w_share", "draft_number", "player_id"],
                                ascending=[True, False, True, True])
    rank = {pid: i for i, pid in enumerate(order["player_id"])}
    grid["rank"] = grid["player_id"].map(rank)
    grid = grid.sort_values(["team_id", "team_game_index", "rank"]).reset_index(drop=True)

    player_ids = players["player_id"].to_numpy()
    player_pos = {pid: i for i, pid in enumerate(player_ids)}
    unit_ids = units["player_id"].to_numpy()
    unit_pos = {pid: i for i, pid in enumerate(unit_ids)}

    row_player = grid["player_id"].map(player_pos).to_numpy(np.int64)
    component_row = np.array([unit_pos.get(pid, -1) for pid in player_ids], dtype=np.int64)
    row_slot = grid["slot"].to_numpy(np.int64)
    row_scorable = component_row[row_player] >= 0

    games = np.sort(grid["game_id"].unique())
    row_game = grid["game_id"].map({g: i for i, g in enumerate(games)}).to_numpy(np.int64)
    row_block = pd.factorize(pd.MultiIndex.from_frame(
        grid[["team_id", "team_game_index"]]), sort=False)[0].astype(np.int64)
    row_cell, cells = pd.factorize(pd.MultiIndex.from_frame(
        grid[["player_id", "team_id"]]), sort=False)
    row_cell = row_cell.astype(np.int64)
    cell_frame = (grid.assign(_cell=row_cell).groupby("_cell", sort=True)
                  .agg(player_id=("player_id", "first"), games=("game_id", "size")))
    row_index_in_cell = grid.assign(_cell=row_cell).groupby("_cell").cumcount()

    # The tensor's flat (player, slot) address for every roster row, so a simulated season
    # collapses into periods with one `bincount` and no player x game array survives.
    row_flat = np.where(row_scorable & (row_slot >= 0),
                        component_row[row_player] * N_SCORING_PERIODS
                        + np.maximum(row_slot, 0), 0)

    # ── the heads ────────────────────────────────────────────────────────────
    full_avail = availability_design(cfg)
    avail = full_avail[full_avail["season"] == season].drop_duplicates(
        subset=["player_id"]).set_index("player_id").reindex(player_ids).reset_index()
    present = avail["team_games"].notna().to_numpy()
    avail_art = artifacts["availability"]
    # A rostered player with no availability design row gets the empirical rate of players
    # like him, not the head's intercept — see `no_design_availability`. It is a constant
    # across posterior draws, which is the honest shape of a plugged-in empirical prior:
    # no posterior on it, so the predictive does not integrate over its uncertainty.
    no_design = no_design_availability(features_dir, season, full_avail,
                                       allowed_seasons(design))
    mu = np.full((avail_art.n_draws, len(player_ids)), no_design)
    if present.any():
        mu[:, present] = avail_art.mu_draws(avail[present])

    comp_art = artifacts["composition"]
    comp_model = rehydrate_composition(comp_art, comp_art.n_draws,
                                       injected_sigma=shipped_sigma(cfg))
    eta = composition_eta(comp_art, comp_model, players)
    ps_sigma = (np.asarray(comp_model.ps.sigma_draws, dtype=float)
                if comp_model.ps.enabled else np.zeros(comp_art.n_draws))
    row_rho_bin = comp_art.recipe.transform(players)["rho_bin"].to_numpy(np.int64)[row_player]

    dur_mu, dur_kappa = spell_shape(artifacts["gp_duration"], avail[present])
    rates = component_rates(artifacts, units)

    # The population mean per-game count the residual correlation was pooled at, built from
    # PRIOR-season minutes so nothing about the target season enters. Restricted to rows
    # clearing `serial_correlation`'s own `MIN_EXPECTED`, because that filter is what makes
    # the measured correlation a statement about players the component is meaningful for.
    mpg = units["mpg_lag1"].to_numpy(float)
    mean_counts = np.array([
        _mean_expected(rates["count"][c].mean(axis=0) * mpg) for c in COUNT_HEADS])
    long = pd.read_csv(Path(cfg["eda"]["output_dir"]) / "residual_correlation.csv")
    copula = count_copula(long, window, mean_counts)

    return {
        "season": season, "window": window, "n_sims": n_sims, "seed": seed,
        "n_draws": min(a.n_draws for a in artifacts.values()),
        "grid": grid, "players": players, "units": units,
        "player_ids": player_ids, "unit_ids": unit_ids,
        "n_players": len(player_ids), "n_units": len(units),
        "n_flat": len(units) * N_SCORING_PERIODS,
        "n_games": len(games), "games": games,
        "row_player": row_player, "row_game": row_game, "row_block": row_block,
        "row_cell": row_cell, "row_index_in_cell": row_index_in_cell.to_numpy(np.int64),
        "row_flat": row_flat, "row_scorable": row_scorable, "row_slot": row_slot,
        "row_w": players["w_share"].to_numpy(float)[row_player],
        "row_rho_bin": row_rho_bin, "n_blocks": int(row_block.max()) + 1,
        "n_cells": len(cells),
        "cell_player": cell_frame["player_id"].map(player_pos).to_numpy(np.int64),
        "cell_games": cell_frame["games"].to_numpy(np.int64),
        "component_row": component_row,
        "avail_mu": mu, "avail_rho": np.asarray(avail_art.draws["rho_draws"], float),
        "dur_mu": dur_mu, "dur_kappa": dur_kappa,
        "eta_base": eta["base"], "eta_per_overtime": eta["per_overtime"],
        "eta_per_clip": eta["per_clip"], "affine_error": eta["affine_error"],
        "ps_sigma": ps_sigma, "sigma_source": comp_model.sigma_source,
        "comp_rho": comp_model.rho_draws,
        "rates": rates, "phi": rates["phi"], "rho": rates["rho"],
        "length_draws": posterior_inputs(artifacts["game_length_ot"],
                                         artifacts["game_length_depth"],
                                         forward_cells(season, 1)),
        "chol": copula["chol"], "copula_target": copula["target"],
        "copula": copula, "frailty_sigma": copula["sigma"],
        "no_design_availability": no_design,
        "composition_variant": comp_art.recipe.variant,
    }


def simulate(ctx: dict) -> dict:
    """The sim loop. Returns the two tensors plus everything Gate A reads."""
    n_sims, n_units = ctx["n_sims"], ctx["n_units"]
    points = np.zeros((n_sims, ctx["n_flat"]), dtype=np.float32)
    games = np.zeros((n_sims, ctx["n_flat"]), dtype=np.uint8)
    season_minutes = np.zeros((n_sims, n_units))
    season_gp = np.zeros((n_sims, ctx["n_players"]))
    bonus_total, played_rows, short = 0.0, 0, 0
    probe = None

    step = max(n_sims // 10, 1)
    for s in range(n_sims):
        out = _sim_one(s, ctx)
        points[s] = out["points"]
        games[s] = out["games"]
        season_minutes[s] = out["season_minutes"]
        season_gp[s] = out["season_gp"]
        bonus_total += out["bonus_total"]
        played_rows += out["n_played"]
        short += out["short_blocks"]
        if out["probe"] is not None:
            probe = out["probe"]
        if (s + 1) % step == 0:
            print(f"    simulated {s + 1:,} / {n_sims:,} seasons")

    shape = (n_sims, n_units, N_SCORING_PERIODS)
    return {
        "dk_pts": np.ascontiguousarray(points.reshape(shape).transpose(1, 2, 0)),
        "games_played": np.ascontiguousarray(games.reshape(shape).transpose(1, 2, 0)),
        "season_minutes": season_minutes, "season_gp": season_gp,
        "bonus_per_game": bonus_total / max(played_rows, 1),
        "played_rows": played_rows, "short_blocks": short, "probe": probe,
    }


# ── Gate A ────────────────────────────────────────────────────────────────────

def marginal_metrics(samples: np.ndarray, y: np.ndarray) -> dict:
    """The marginal metric set every head in this project reports, on `(sims x rows)`.

    Public because Gate A is not the only place it is read: `src/sim/weekly.py` scores the
    same tensor at the scoring-period unit and has to do it with the *same* arithmetic, or
    a reader comparing a weekly MAE against the season-total row beside it is comparing two
    definitions rather than two units.
    """
    pred = samples.mean(axis=0)
    return {"n": int(len(y)),
            "mae": float(np.abs(pred - y).mean()),
            "rmse": float(np.sqrt(((pred - y) ** 2).mean())),
            "r2": float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "bias": float((pred - y).mean()),
            "crps": float(crps_from_samples(samples, y).mean())}


def realized_frame(cfg: dict, ctx: dict) -> pd.DataFrame:
    """Realized season dk_pts, minutes and games played, on the simulator's own units.

    `dk_total` is summed over the games that carry a **slot**, because DK's Round 4 closes
    before the NBA season does and the tensor only covers the tournament window. `gp` and
    `minutes` are over the whole **regular-season** schedule, which is the availability
    head's own denominator — the playoff rows in `component_targets.parquet` belong to no
    contest and are dropped on both sides.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(
        features_dir / "component_targets.parquet",
        columns=["player_id", "season", "season_type", "game_id", "dk_pts", "min",
                 "played"])
    played = targets[(targets["season"] == ctx["season"])
                     & (targets["season_type"] == "regular")
                     & (targets["played"] == 1)]
    slots = scoring_slots(features_dir, ctx["season"])
    inside = set(slots.loc[slots["slot"] >= 0, "game_id"])

    scored = played[played["game_id"].isin(inside)]
    out = pd.DataFrame({"player_id": ctx["unit_ids"]})
    out = out.merge(scored.groupby("player_id", as_index=False)
                    .agg(dk_total=("dk_pts", "sum")), on="player_id", how="left")
    out = out.merge(played.groupby("player_id", as_index=False)
                    .agg(minutes=("min", "sum"), gp=("played", "sum")),
                    on="player_id", how="left")
    return out.fillna({"dk_total": 0.0, "minutes": 0.0, "gp": 0.0})


def realized_played_rows(cfg: dict, ctx: dict) -> pd.DataFrame:
    """Realized played player-games for the simulator's own scorable units.

    The population matters and used to be assumed away. `bonus_calibration.csv` pools every
    player-game in the project's frame; the tensor scores the ~386 units that clear the
    component heads' `>= 200 prior minutes` filter, whose realized bonus rate is materially
    higher because they are the players who play. Comparing the simulator against the
    pooled figure would charge it for a population difference.
    """
    targets = pd.read_parquet(Path(cfg["data"]["features_dir"])
                              / "component_targets.parquet")
    rows = targets[(targets["season"] == ctx["season"])
                   & (targets["season_type"] == "regular")
                   & (targets["played"] == 1) & (targets["min"] > 0)
                   & targets["player_id"].isin(set(ctx["unit_ids"]))].copy()
    rows["pts"] = (2 * rows["fg2m"] + 3 * rows["fg3m"] + rows["ftm"])
    return rows


def bonus_on_realized_minutes(ctx: dict, rows: pd.DataFrame, draws: int = 12,
                              seed: int = 0) -> float:
    """The component chain's bonus rate when handed **realized** minutes.

    This is the row that turns Gate A from a pass/fail into a diagnosis. The simulator's
    bonus rate is a function of two things — the components given minutes, and the minutes
    themselves — and only one of them is written in this module. Running the identical
    `draw_components` call on the realized minutes of the realized played games isolates
    the first: if it lands on the realized bonus rate, any remaining miss belongs to the
    minutes draw and not to the assembly.
    """
    pos = {p: i for i, p in enumerate(ctx["unit_ids"])}
    unit = rows["player_id"].map(pos).to_numpy(np.int64)
    minutes = rows["min"].to_numpy(float)
    rng = np.random.default_rng(seed)
    total, n = 0.0, 0
    for d in range(min(draws, ctx["n_draws"])):
        season_gamma = {c: rng.gamma(ctx["phi"][c][d], 1.0 / ctx["phi"][c][d],
                                     size=ctx["n_units"])[unit] for c in COUNT_HEADS}
        season_p = {}
        for m, att in CONVERSION_HEADS:
            head = f"{m}|{att}"
            pa, pb = beta_shapes(ctx["rates"]["conversion"][head][d],
                                 np.full(ctx["n_units"], ctx["rho"][head][d]))
            season_p[head] = rng.beta(pa, pb)[unit]
        sigma = ctx["frailty_sigma"]
        normals = rng.standard_normal((len(unit), len(COUNT_HEADS))) @ ctx["chol"].T
        counts, made, _ = draw_components(
            rng, minutes, unit, ctx["rates"], d, season_gamma, season_p,
            np.exp(sigma * normals - 0.5 * sigma ** 2))
        box = pd.DataFrame({"pts": 2 * made["fg2m"] + 3 * made["fg3m"] + made["ftm"],
                            "reb": counts["reb"], "ast": counts["ast"],
                            "stl": counts["stl"], "blk": counts["blk"]})
        total += float(bonus_part(box).sum())
        n += len(box)
    return total / max(n, 1)


def _bar(frame: pd.DataFrame, **filters) -> pd.DataFrame:
    for column, value in filters.items():
        frame = frame[frame[column] == value]
    return frame


def gate_a(cfg: dict, ctx: dict, sim: dict) -> pd.DataFrame:
    """The marginals the simulator was handed, each against the artifact that set it.

    Every input head is already calibrated, so a miss here is a **wiring fault** rather
    than a modelling one — which is why each comparison is reported as its own row with the
    artifact and the bar beside it, and never collapsed into a verdict.
    """
    from src.eda.serial_correlation import (block_inflation, count_residuals,
                                            minutes_residuals, shuffled_null)
    from src.models.stan_minutes import game_level_dispersion

    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    eda_dir = Path(cfg["eda"]["output_dir"])
    window = ctx["window"]
    realized = realized_frame(cfg, ctx)
    rows: list[dict] = []

    # ── 1. season-total dk_pts ───────────────────────────────────────────────
    totals = sim["dk_pts"].sum(axis=1).T                 # (sims x units)
    keep = realized["gp"].to_numpy() > 0
    m = marginal_metrics(totals[:, keep].astype(float), realized["dk_total"].to_numpy()[keep])
    incumbent = _bar(pd.read_csv(out_dir / "season_total_metrics.csv"),
                     treatment="beta_binomial", group="all")
    bar = incumbent.set_index("metric")["value"]
    rows.append({"check": "season_total_dk", "unit": "player-season", "gate": "A",
                 "artifact": "season_total_metrics.csv", "bar_source": "beta_binomial/all",
                 **m,
                 "bar_mae": float(bar["mae_dk_total"]), "bar_r2": float(bar["r2_dk_total"]),
                 "bar_bias": float(bar["bias_dk_total"]),
                 "bar_crps": float(bar["crps_dk_total"]),
                 "bar_n": int(incumbent["n"].iloc[0])})

    # ── 2. games played ──────────────────────────────────────────────────────
    # `season_gp` is indexed by rostered player and everything else by scorable unit, so
    # the two are aligned explicitly rather than by a coincidence of ordering.
    has_unit = ctx["component_row"] >= 0
    order = np.empty(ctx["n_units"], dtype=np.int64)
    order[ctx["component_row"][has_unit]] = np.flatnonzero(has_unit)
    gp_unit = sim["season_gp"][:, order]
    gp_metrics = marginal_metrics(gp_unit[:, keep], realized["gp"].to_numpy()[keep])
    # The pmf is compared **pooled over players**, not player by player. A per-player
    # empirical pmf from `n_sims` draws over 83 support points is mostly Monte Carlo error:
    # its total variation against any smooth pmf falls with `n_sims` whatever the model
    # does, so it measures the budget rather than the wiring. Pooling first is stable and
    # is still the statistic the board cares about — how many games the population plays.
    pmf = pd.read_csv(out_dir / "stan_games_played_gp_pmf.csv")
    pmf = pmf[pmf["season"] == ctx["season"]]
    tvd, shared, head_mean = float("nan"), 0, float("nan")
    if len(pmf):
        wide = pmf.pivot_table(index="player_id", columns="gp", values="p", fill_value=0.0)
        ids = [p for p in wide.index if p in set(ctx["unit_ids"])]
        shared = len(ids)
        if shared:
            pos = {p: i for i, p in enumerate(ctx["unit_ids"])}
            cols = wide.columns.to_numpy()
            counts = np.bincount(
                gp_unit[:, [pos[p] for p in ids]].astype(int).ravel(),
                minlength=int(cols.max()) + 1).astype(float)
            sim_pmf = counts[cols] / max(counts.sum(), 1.0)
            head_pmf = wide.loc[ids].to_numpy().mean(axis=0)
            head_pmf = head_pmf / head_pmf.sum()
            tvd = float(0.5 * np.abs(sim_pmf - head_pmf).sum())
            head_mean = float((head_pmf * cols).sum())
    rows.append({"check": "games_played", "unit": "player-season", "gate": "A",
                 "artifact": "stan_games_played_gp_pmf.csv, "
                             "stan_games_played_metrics.csv",
                 "bar_source": "floor (the availability head)", **gp_metrics,
                 "bar_crps": float(_bar(pd.read_csv(out_dir /
                                                    "stan_games_played_metrics.csv"),
                                        variant="floor")["val_crps"].iloc[0]),
                 "pmf_total_variation": tvd, "pmf_players": shared,
                 "pmf_mean_gp": head_mean,
                 "simulated_mean_gp": float(gp_unit.mean()),
                 "pmf_arm": str(pmf["arm"].iloc[0]) if len(pmf) else ""})

    # ── 3. per-game bonus rate ───────────────────────────────────────────────
    bonus_bar = _bar(pd.read_csv(eda_dir / "bonus_calibration.csv"),
                     fit_window=window, analysis="calibration", unit="player_game",
                     bucket="all", overdispersion=BONUS_GAME_OVERDISPERSION)
    played = realized_played_rows(cfg, ctx)
    realized_rate = float(bonus_part(played).mean())
    rows.append({"check": "bonus_per_game", "unit": "player-game", "gate": "A",
                 "artifact": "bonus_calibration.csv",
                 "bar_source": "realized, on the simulator's own scorable units",
                 "n": int(sim["played_rows"]),
                 "value": float(sim["bonus_per_game"]),
                 "bar_value": realized_rate,
                 "pooled_artifact_bar": float(bonus_bar["realized_mean_bonus"].iloc[0]),
                 "bias": float(sim["bonus_per_game"] - realized_rate)})
    # The isolating row: the same `draw_components` call on REALIZED minutes. It splits the
    # bonus miss into the half this module writes and the half the minutes head hands it.
    rows.append({"check": "bonus_given_realized_minutes", "unit": "player-game",
                 "gate": "diagnostic", "artifact": "component_targets.parquet",
                 "bar_source": "realized, same rows", "n": int(len(played)),
                 "value": bonus_on_realized_minutes(ctx, played),
                 "bar_value": realized_rate})

    # ── 4. season-total minutes spread ───────────────────────────────────────
    # The bar is the marginal head's predictive sd, which conditions on **realized** games
    # played: `stan_minutes` takes the summed length of the games he actually played as its
    # trials. The simulator draws availability too, so its unconditional spread must be
    # WIDER and the bar is a floor rather than a target. `conditional_sd` is the
    # like-for-like number — the residual sd of season minutes after regressing out that
    # sim's own games played, per player — and that is the one to read against 302.75.
    qualified = realized["minutes"].to_numpy() >= MIN_QUALIFIED_MINUTES
    minutes = sim["season_minutes"]
    minutes_sd = float(minutes[:, qualified].std(axis=0).mean())
    x = gp_unit - gp_unit.mean(axis=0)
    y = minutes - minutes.mean(axis=0)
    slope = np.divide((x * y).sum(axis=0), (x * x).sum(axis=0),
                      out=np.zeros(minutes.shape[1]), where=(x * x).sum(axis=0) > 0)
    conditional = float((y - slope * x)[:, qualified].std(axis=0, ddof=1).mean())
    unification = _bar(pd.read_csv(out_dir / "minutes_unification.csv"),
                       unit="season_total")
    rows.append({"check": "season_minutes_spread", "unit": "player-season", "gate": "A",
                 "artifact": "minutes_unification.csv",
                 "bar_source": "minutes_head / composition_sum",
                 "n": int(qualified.sum()), "value": minutes_sd,
                 "conditional_sd": conditional,
                 "bar_value": float(_bar(unification, arm="minutes_head")
                                    ["predictive_sd"].iloc[0]),
                 "composition_only": float(_bar(unification, arm="composition_sum")
                                           ["predictive_sd"].iloc[0]),
                 **marginal_metrics(minutes[:, keep], realized["minutes"].to_numpy()[keep])})

    # ── diagnostics, reported and not gated ──────────────────────────────────
    probe = sim["probe"]
    frame = pd.DataFrame({"ps": probe["unit"], "min": probe["minutes"],
                          "player_id": ctx["unit_ids"][probe["unit"]],
                          "season": ctx["season"], "season_type": "regular",
                          "played": 1, "game_id": probe["game"],
                          "game_length": probe["length"]})
    for component in COUNT_HEADS:
        frame[component] = probe["counts"][component]

    lengths = (frame[["season", "season_type", "game_id", "game_length"]]
               .drop_duplicates())
    # `game_level_dispersion` joins the lengths on itself, so the probe hands it the
    # minutes side without one. `window="full"` because the probe is a single simulated
    # season and there is nothing to hold out of it.
    dispersion = game_level_dispersion(frame.drop(columns=["game_length"]), lengths,
                                       window="full")
    disp_bar = _bar(pd.read_csv(out_dir / "stan_minutes_dispersion.csv"),
                    metric="game_level_rho", fit_window=window)
    rows.append({"check": "minutes_game_dispersion", "unit": "player-game",
                 "gate": "diagnostic", "artifact": "stan_minutes_dispersion.csv",
                 "bar_source": f"{window} (a DIAGNOSTIC, not an input)",
                 "n": dispersion["n_player_games"],
                 "value": dispersion["implied_overdispersion"],
                 "bar_value": float(disp_bar["implied_overdispersion"].iloc[0])})

    resid, valid = minutes_residuals(frame)
    ps = frame["ps"].to_numpy()
    observed = block_inflation(resid, ps, valid, SERIAL_BLOCK)
    null, _ = shuffled_null(resid, ps, valid,
                            lambda r, p, v: block_inflation(r, p, v, SERIAL_BLOCK))
    serial = _bar(pd.read_csv(eda_dir / "serial_correlation.csv"),
                  fit_window=window, component="min")
    rows.append({"check": "minutes_block_inflation", "unit": "player-game",
                 "gate": "diagnostic", "artifact": "serial_correlation.csv",
                 "bar_source": f"{window}/min", "n": int(len(frame)),
                 "value": float(observed / null) if null else float("nan"),
                 "bar_value": float(serial["block_inflation"].iloc[0])})

    resids = {c: count_residuals(frame, c) for c in COUNT_HEADS}
    sim_R = np.eye(len(COUNT_HEADS))
    for i, a in enumerate(COUNT_HEADS):
        for j, b in enumerate(COUNT_HEADS):
            if i >= j:
                continue
            ra, va = resids[a]
            rb, vb = resids[b]
            both = va & vb
            r = float(np.corrcoef(ra[both], rb[both])[0, 1]) if both.sum() > 2 else np.nan
            sim_R[i, j] = sim_R[j, i] = r
    off = ~np.eye(len(COUNT_HEADS), dtype=bool)
    rows.append({"check": "cross_component_correlation", "unit": "player-game",
                 "gate": "diagnostic", "artifact": "residual_correlation.csv",
                 "bar_source": f"{window}/minutes_conditioned, counts only",
                 "n": int(len(frame)),
                 "value": float(np.nanmean(sim_R[off])),
                 "bar_value": float(ctx["copula_target"][off].mean()),
                 "max_abs_cell_error": float(np.nanmax(np.abs(
                     sim_R[off] - ctx["copula_target"][off])))})

    table = pd.DataFrame(rows)
    table.insert(0, "season", ctx["season"])
    table.insert(1, "fit_window", window)
    table["n_sims"] = ctx["n_sims"]
    return table


# ── Entry point ───────────────────────────────────────────────────────────────

def save_tensor(sim: dict, ctx: dict, dest: Path) -> Path:
    """The artifact: two aligned tensors, their index, and the provenance to read them by.

    `np.savez` rather than `savez_compressed` on purpose — the draft room's budget is a
    sub-second load, and 90 MB of float32 reads faster than it decompresses.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    pool = ctx["units"]
    np.savez(
        dest,
        dk_pts=sim["dk_pts"].astype(np.float32),
        games_played=sim["games_played"].astype(np.uint8),
        player_id=ctx["unit_ids"].astype(np.int64),
        season_minutes=sim["season_minutes"].astype(np.float32),
        season=np.array(ctx["season"]),
        fit_window=np.array(ctx["window"]),
        n_sims=np.array(ctx["n_sims"]),
        n_posterior_draws=np.array(ctx["n_draws"]),
        seed=np.array(ctx["seed"]),
        player_season_sigma=np.array(float(ctx["ps_sigma"].mean())),
        sigma_source=np.array(ctx["sigma_source"]),
        composition_variant=np.array(ctx["composition_variant"]),
        scoring_periods=np.arange(N_SCORING_PERIODS),
        tournament_round=np.r_[np.ones(ROUND_1_WEEKS, dtype=int), [2, 3, 4]],
        prior_minutes=pool["total_minutes_lag1"].to_numpy(np.float32),
    )
    return dest


def run(cfg: dict, seasons: list[str] | None = None, n_sims: int | None = None,
        window: str | None = None, seed: int | None = None) -> dict[str, Path]:
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sim = cfg.get("sim", {})
    window = window or str(cfg_sim.get("fit_window", FIT_WINDOW))
    n_sims = int(n_sims or cfg_sim.get("n_sims", N_SIMS))
    seed = int(SEED if seed is None else seed)

    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = component_build_design(targets, cfg["data"]["seasons"],
                                    cfg["data"]["raw_dir"])
    seasons = seasons or validation_seasons(design)

    print(f"Season simulator — the player x scoring-period x sim tensor")
    print(f"  posteriors at the `{window}` window; nothing is refitted, and nothing "
          f"here needs CmdStan.")
    print(f"  The test split is LOCKED — `allowed_seasons` goes through "
          f"`held_out.selection_split`.")
    print(f"  {N_SCORING_PERIODS} scoring periods "
          f"({ROUND_1_WEEKS} Round-1 weeks + 3 double weeks), {n_sims:,} sims, "
          f"seed {seed}")

    composition = composition_frame(cfg)
    paths: dict[str, Path] = {}
    gates = []
    for season in seasons:
        print(f"\n── {season} ──")
        ctx = build_context(cfg, season, window, n_sims, seed, composition=composition)
        print(f"  {len(ctx['grid']):,} rostered player-games over "
              f"{ctx['n_blocks']:,} team-games and {ctx['n_games']:,} games; "
              f"{ctx['n_players']:,} rostered players, {ctx['n_units']:,} of them "
              f"scorable")
        print(f"  minutes: composition @ {ctx['composition_variant']} with a "
              f"per-(player, season) effect, sigma "
              f"{float(ctx['ps_sigma'].mean()):.3f} ({ctx['sigma_source']})")
        print(f"  availability: the head where it has a row, {ctx['no_design_availability']:.4f} "
              f"where it does not (an expanding-window empirical rate)")
        print(f"  copula: {len(COUNT_HEADS)} count frailties, lognormal sigma "
              f"{ctx['frailty_sigma']:.4f} at overdispersion "
              f"{BONUS_GAME_OVERDISPERSION}; {ctx['copula']['saturated']} of "
              f"{len(COUNT_HEADS) * (len(COUNT_HEADS) - 1) // 2} pairs saturate the "
              f"frailty's reach")
        print(f"  draw order: {' -> '.join(DRAW_ORDER)}")

        sim = simulate(ctx)
        if sim["short_blocks"]:
            print(f"   /!\\  {sim['short_blocks']:,} team-games of "
                  f"{ctx['n_blocks'] * n_sims:,} needed a feasibility repair "
                  f"(fewer than {MIN_AVAILABLE} available players)")

        dest = features_dir / f"sim_tensor_{season}.npz"
        save_tensor(sim, ctx, dest)
        mb = (sim["dk_pts"].nbytes + sim["games_played"].nbytes) / 1e6
        print(f"Saved {sim['dk_pts'].size:,} dk_pts cells "
              f"({ctx['n_units']:,} x {N_SCORING_PERIODS} x {n_sims:,}, {mb:.0f} MB) "
              f"→ {dest}")
        paths[season] = dest

        table = gate_a(cfg, ctx, sim)
        gates.append(table)
        _report_gate(table)

    dest = out_dir / "sim_season_gate_a.csv"
    gate = merge_gate(pd.concat(gates, ignore_index=True), dest)
    gate.to_csv(dest, index=False)
    print(f"\nSaved {len(gate):,} Gate A rows over "
          f"{gate['season'].nunique()} seasons → {dest}")
    paths["gate_a"] = dest
    return paths


def merge_gate(fresh: pd.DataFrame, dest: Path) -> pd.DataFrame:
    """This run's Gate A rows over whatever the file already held, merged **by season**.

    `--season` is a real flag, so a run that names one season used to write a one-season
    file and silently drop the record for every other season. That is the mistake
    `make posteriors`' manifest already avoids by merging on `head`, and for the same
    reason: the expensive artifact is per unit and a partial run is the normal workflow.
    Re-running a season replaces its own rows rather than appending a second copy, so the
    file cannot grow two readings of one season and leave a consumer to pick.
    """
    if not dest.exists():
        return fresh
    kept = pd.read_csv(dest)
    kept = kept[~kept["season"].isin(set(fresh["season"]))]
    return pd.concat([kept, fresh], ignore_index=True).sort_values(
        ["season", "check"], kind="stable").reset_index(drop=True)


def _report_gate(table: pd.DataFrame) -> None:
    """Print every comparison explicitly. Never 'close'."""
    print("\nGate A — does the simulator reproduce the marginals it was handed?")
    for _, row in table.iterrows():
        mark = "  " if row["gate"] == "A" else "  · "
        if row["check"] == "season_total_dk":
            print(f"{mark}season-total dk_pts   n={row['n']:>5,}  "
                  f"MAE {row['mae']:8.2f} (bar {row['bar_mae']:.2f} on "
                  f"{int(row['bar_n'])} rows)  R2 {row['r2']:6.4f} "
                  f"(bar {row['bar_r2']:.4f})  bias {row['bias']:+8.2f} "
                  f"(bar {row['bar_bias']:+.2f})  CRPS {row['crps']:7.2f} "
                  f"(bar {row['bar_crps']:.2f})")
        elif row["check"] == "games_played":
            print(f"{mark}games played          n={row['n']:>5,}  "
                  f"MAE {row['mae']:8.3f}  bias {row['bias']:+7.3f}  "
                  f"CRPS {row['crps']:7.4f} (the head's own floor "
                  f"{row['bar_crps']:.4f});  pooled pmf total variation "
                  f"{row['pmf_total_variation']:.4f} against the "
                  f"`{row['pmf_arm']}` arm on {int(row['pmf_players'])} players "
                  f"(mean GP {row['simulated_mean_gp']:.2f} against "
                  f"{row['pmf_mean_gp']:.2f})")
        else:
            value, bar = float(row["value"]), float(row["bar_value"])
            extra = ""
            if row["check"] == "season_minutes_spread":
                extra = (f"  |  conditional on games played "
                         f"{float(row['conditional_sd']):.2f} — the like-for-like "
                         f"number (the un-injected composition reads "
                         f"{float(row['composition_only']):.2f})")
            if row["check"] == "cross_component_correlation":
                extra = f"  max |cell error| {float(row['max_abs_cell_error']):.4f}"
            if row["check"] == "bonus_per_game":
                extra = (f"  (the pooled artifact figure over every player-game is "
                         f"{float(row['pooled_artifact_bar']):.4f})")
            print(f"{mark}{row['check']:<28} {value:10.4f}  against "
                  f"{bar:10.4f}  ({value - bar:+.4f}){extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to the validation "
                             "seasons.")
    parser.add_argument("--n-sims", type=int, default=None)
    parser.add_argument("--window", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, n_sims=args.n_sims, window=args.window,
        seed=args.seed)
