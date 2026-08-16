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
from src.models.availability_no_prior import (KEY_LADDERS, PRESEASON_KEY_ARMS,
                                              SHIPPED_LEVEL_ARM, appearance_gap, level_keys,
                                              level_rates, level_tables, primary_team_cells)
from src.models.component_rates import CONVERSION_HEADS, COUNT_HEADS, DERIVED_COUNTS
from src.models.games_played import (EdgeResampler, allocate_spells, edge_blocks,
                                     layout_tenure)
from src.models.held_out import TEST_SEASONS, assert_unlocked, selection_split
from src.models.minutes_unification import rehydrate_composition, shipped_sigma
from src.models.posteriors import load_all, posteriors_dir, require_window
from src.models.stan_availability import (FIRST_SEASON, head_design,
                                          restrict_window, role_bins)
from src.models.stan_components import head_design as component_head_design
from src.models.stan_composition import (OFFSET_CLIP, draft_numbers, head_frame,
                                         simulate_minutes)
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


def no_design_level_arm(cfg: dict) -> str:
    """The configured key ladder for the no-design availability level.

    P4's preseason arms are refused here rather than allowed to fail deep inside
    `level_tables`: this module's own key builder does not join the preseason panel, so
    naming one would be a config that cannot work. They are measured by
    `make availability-no-prior` and `docs/preseason-plan.md` P4 records that none of them
    ships, so the refusal costs nothing today and is a clear message rather than a `KeyError`
    if that ever changes.
    """
    name = str(cfg.get("sim", {}).get("availability", {})
               .get("no_design_level", SHIPPED_LEVEL_ARM))
    usable = [a for a in KEY_LADDERS if a not in PRESEASON_KEY_ARMS]
    if name in PRESEASON_KEY_ARMS:
        raise ValueError(f"sim.availability.no_design_level {name!r} needs a preseason key "
                         f"this module does not build — see `docs/preseason-plan.md` P4, "
                         f"where it is measured and not shipped")
    if name not in usable:
        raise ValueError(f"unknown sim.availability.no_design_level {name!r}; expected one "
                         f"of {sorted(usable)}")
    return name


def no_design_availability(features_dir: Path, season: str, design: pd.DataFrame,
                           allowed: list[str], seasons: list[str],
                           player_ids: np.ndarray, arm: str) -> pd.Series:
    """Availability rate for rostered players the availability head has no row for.

    Returned for **every** id in `player_ids`, indexed by player, because the caller's own
    `present` mask is what decides which of them are used — a player with a design row gets
    his rate from the head and this one is overwritten. Returning the full vector rather than
    the subset keeps that alignment positional and impossible to get subtly wrong.

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

    **"Players like them" is a graded key rather than the whole population** — `arm` names
    the ladder and `pooled` recovers the single scalar this used to return, exactly. The
    grading is what `make availability-no-prior` §8b selected: an undrafted call-up and a
    top-5 pick realize 0.2500 and 0.8316, and the draft bucket says which — but only for a
    *first* appearance, since a returning veteran's bucket is a decade old and his realized
    rate collapses toward 0.30 whatever it says.

    **Two windows, deliberately different.** The *rate* may only be pooled from seasons
    selection may read, because it is an estimate taken from outcomes. The *key* — has this
    player appeared before, and where was he drafted — is read from the full panel strictly
    before the target, because it is a roster fact knowable at draft time, and restricting it
    to `allowed` would silently relabel a returning veteran as a rookie in production, where
    the seasons in between are held-out ones.
    """
    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "game_date", "played",
                 "min"])
    before = panel[panel["season"] < season]
    pooled = before[before["season"].isin([s for s in allowed if s < season])]
    if pooled.empty:
        raise ValueError(f"no seasons before {season} to estimate a no-design "
                         f"availability rate from")
    cells = primary_team_cells(pooled)
    covered = set(map(tuple, design[["season", "player_id"]].to_numpy()))
    history = cells[[(s, p) not in covered for s, p
                     in zip(cells["season"], cells["player_id"])]].copy()
    if history.empty:
        raise ValueError("no uncovered player-seasons before "
                         f"{season}; the availability design cannot be that complete")

    buckets = draft_numbers(features_dir)
    appearances = before.loc[before["played"] == 1, ["season", "player_id"]]
    history["gap"] = appearance_gap(history, appearances, seasons)
    history = level_keys(history.merge(buckets, on=["player_id", "season"], how="left"))

    rows = pd.DataFrame({"player_id": player_ids, "season": season})
    rows["gap"] = appearance_gap(rows, appearances, seasons)
    rows = level_keys(rows.merge(buckets, on=["player_id", "season"], how="left"))
    rate, _ = level_rates(rows, level_tables(history, season, allowed, arm))
    return pd.Series(rate, index=player_ids, name="no_design_availability")


#: The layout arms `sim.availability.layout` may name, as `(tenure factor, overflow policy)`.
#: `clustered` is the previously shipped corner and is recoverable exactly; `tenure_merge`
#: ships. `make availability-exchangeability` is the ladder and
#: `docs/availability-window-plan.md` §13 the readout.
LAYOUT_ARMS = {"clustered": (False, "collapse"), "merge": (False, "merge"),
               "tenure": (True, "collapse"), "tenure_merge": (True, "merge")}


def layout_arm(cfg: dict) -> tuple[str, bool, str]:
    """`(name, tenure, overflow)` for the configured availability layout."""
    name = str(cfg.get("sim", {}).get("availability", {}).get("layout", "tenure_merge"))
    if name not in LAYOUT_ARMS:
        raise ValueError(f"unknown sim.availability.layout {name!r}; expected one of "
                         f"{sorted(LAYOUT_ARMS)}")
    tenure, overflow = LAYOUT_ARMS[name]
    return name, tenure, overflow


def tenure_edges(features_dir: Path, season: str, design: pd.DataFrame,
                 allowed: list[str], first_season: str) -> EdgeResampler:
    """Edge-block fractions for the layout, pooled from seasons strictly before the target.

    The same point-in-time construction `no_design_availability` and
    `stan_composition.rookie_share_priors` use, and for the same reason: this is an
    empirical rate handed to the simulator, so it may see only seasons already played and
    only seasons selection may read. Restricted to the availability head's own fitting
    window as well, because that is the estimator `make availability-exchangeability`
    measured and a shipped rule transfers from a ladder only if it is the ladder's rule.

    Multi-team player-seasons drop out inside `edge_blocks`: a traded player's tenure with
    one team ends without his season ending, so his trailing block is a roster fact rather
    than an absence, and pooling it would teach the layout that stars vanish in February.
    """
    earlier = [s for s in allowed if s < season]
    rows = design[design["season"].isin(earlier)]
    rows = restrict_window(rows, first_season)
    if rows.empty:
        raise ValueError(f"no seasons before {season} in the availability head's "
                         f"{first_season}+ window to pool edge blocks from")
    rows = rows.assign(role_bin=role_bins(rows))
    panel = pd.read_parquet(
        features_dir / "availability_panel.parquet",
        columns=["season", "player_id", "team_id", "game_id", "team_game_index",
                 "played", "status"])
    return EdgeResampler(edge_blocks(panel[panel["season"].isin(earlier)], rows))


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


def availability_rates(rng: np.random.Generator, mu: np.ndarray, rho_by_bin: np.ndarray,
                       bins: np.ndarray, pi: np.ndarray | float = 0.0,
                       mu_low: float = 0.0, rho_low: float = 0.0) -> np.ndarray:
    """One beta-binomial availability rate per player, at that player's **own** `rho`.

    Two vectors and a gather rather than a scalar broadcast, because the shipped head grades
    dispersion by prior-MPG role bucket: `rho_by_bin` is this posterior draw's `(n_rho,)`
    row and `bins` is the 0-based bucket per player from `availability_rho_bin`. Under a
    shared dispersion `n_rho` is 1 and every player gathers the same entry, which reproduces
    the scalar form exactly.

    **The low-availability mixture enters by drawing the COMPONENT first**, per player, and
    taking the rate from whichever one won — `StanAvailability.predict_samples`' rule, for
    its reason: averaging the two rates would produce a season between healthy and disrupted,
    which is precisely the season the arm exists to say does not happen. `pi = 0` is the
    single-component head exactly, so both arms take one path.

    Named rather than inlined because it is the one place the simulator touches this head's
    likelihood, and it was inlined against a scalar for the whole window round without
    anything raising until `rho` became a vector.
    """
    a, b = beta_shapes(mu, rho_by_bin[bins])
    p = rng.beta(a, b)
    pi = np.asarray(pi, dtype=float)
    if not np.any(pi):
        return p
    a_low, b_low = beta_shapes(np.full_like(p, mu_low), np.full_like(p, rho_low))
    return np.where(rng.random(p.shape) < pi, rng.beta(a_low, b_low), p)


def availability_rho_bin(artifact, frame: pd.DataFrame) -> np.ndarray:
    """0-based dispersion column per row, from the availability head's **own** recipe.

    The head grades `rho` by prior-MPG role bucket, so `rho_draws` is `(draws x n_rho)` and
    a consumer has to say which column applies to which player. The artifact already
    carries that as a `cut` recipe step — the same door `build_context` opens on the
    composition head — so the assignment is reconstructed rather than re-derived, and a
    consumer never has to import `stan_availability.role_bins` or know the edges.

    Three things this has to get right, each of which fails silently rather than loudly:

    - **`rho_bin` is 1-based.** `recipe.transform` and `role_bins` both emit `[1..n_rho]`,
      matching the Stan source's own gather. Indexing `rho_draws` needs `bin - 1`, and an
      off-by-one hands a player the *wrong bucket's* dispersion at a legal index.
    - **A shared-`rho` artifact has no such step.** `rho_draws` is then `(draws,)` with an
      empty recipe, and every row belongs to the single column — so a missing step is the
      shared-dispersion arm rather than a broken recipe, and returns all zeros.
    - **A player with no design row still needs a bucket.** His `mu` is
      `no_design_availability`'s empirical rate; his prior MPG is NaN, which the `cut` step
      sends to the **lowest** bucket. That is the head's own rule (`role_bins`) and the
      conservative direction, since the fringe bucket carries the widest dispersion.

    Returned as an index rather than as `rho` itself because the dispersion is a per-draw
    quantity: the caller gathers `rho_draws[draw][bin]` inside the sim loop, where the
    posterior draw is the outer loop (rule 4).
    """
    n_rho = int(artifact.extras.get("n_rho", np.shape(artifact.draws["rho_draws"])[-1]
                                    if np.ndim(artifact.draws["rho_draws"]) > 1 else 1))
    column = artifact.extras.get("rho_bin_column", "rho_bin")
    transformed = artifact.recipe.transform(frame)
    if column not in transformed.columns:
        if n_rho != 1:
            raise KeyError(
                f"the availability artifact carries {n_rho} dispersion columns but its "
                f"recipe produces no {column!r}, so there is no way to say which column "
                f"applies to which player. Re-run `make posteriors`.")
        return np.zeros(len(frame), dtype=np.int64)

    bins = transformed[column].to_numpy(np.int64) - 1
    if bins.min(initial=0) < 0 or bins.max(initial=0) >= n_rho:
        raise ValueError(
            f"{column!r} produced buckets outside [1, {n_rho}] — the recipe's cut and the "
            f"persisted `rho_draws` disagree about the head's dispersion axis")
    return bins


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
    p_available = availability_rates(rng, ctx["avail_mu"][draw], ctx["avail_rho"][draw],
                                     ctx["avail_rho_bin"], ctx["avail_pi"][draw],
                                     float(ctx["avail_mu_low"][draw]),
                                     float(ctx["avail_rho_low"][draw]))
    gp = rng.binomial(ctx["cell_games"], p_available[ctx["cell_player"]])
    # WHERE those games fall is a separate draw from how many, because `gp` is invariant to
    # the arrangement and the head therefore cannot carry it — `make availability-
    # exchangeability`. The edge blocks go at the ends first when the shipped `tenure_merge`
    # arm is configured, and `allocate_spells` gets only the interior remainder.
    mu_dur, kappa_dur = float(ctx["dur_mu"][draw]), float(ctx["dur_kappa"][draw])
    layout_seed = int(rng.integers(1 << 31))
    if ctx["edges"] is None:
        played = allocate_spells(gp, ctx["cell_games"], mu_dur, kappa_dur,
                                 seed=layout_seed, overflow=ctx["layout_overflow"])
    else:
        pre, post = ctx["edges"].draw(gp, ctx["cell_games"], ctx["cell_role"], rng)
        played = layout_tenure(gp, ctx["cell_games"], pre, post, mu_dur, kappa_dur,
                               seed=layout_seed, overflow=ctx["layout_overflow"])
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
        # The same minutes over every ROSTERED player rather than every scorable unit. The
        # two differ by exactly the population this simulator allocates minutes to but never
        # scores — the players the availability head has no row for — so it is the only
        # place their share of a team's fixed pot can be read at all.
        "season_minutes_player": np.bincount(player, weights=minutes,
                                             minlength=ctx["n_players"]),
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

    # `component_head_design`, not `component_rates.build_design` — since 2026-08-15 ten of
    # the eleven rate heads carry preseason feature columns, and the persisted RECIPE demands
    # them. The plain builder produces a frame the recipe cannot evaluate, which is what
    # `PosteriorRecipe._block` raises on rather than silently predicting from a short design.
    design = component_head_design(cfg)
    assert_season_allowed(season, design)

    slots = scoring_slots(features_dir, season)
    grid = roster_grid(features_dir, season, slots)
    frame = head_frame(cfg) if composition is None else composition
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
    # `head_design`, not `availability_design`: the availability head ships a preseason
    # block (docs/preseason-plan.md P2) and the persisted recipe names those columns, so a
    # frame built from the shared builder would satisfy every other head here and be five
    # columns short for this one. The flag lives with the head; `preseason: false` in config
    # gives back the pre-2026-08-13 frame exactly.
    full_avail = head_design(cfg)
    avail = full_avail[full_avail["season"] == season].drop_duplicates(
        subset=["player_id"]).set_index("player_id").reindex(player_ids).reset_index()
    present = avail["team_games"].notna().to_numpy()
    avail_art = artifacts["availability"]
    # A rostered player with no availability design row gets the empirical rate of players
    # like him, not the head's intercept — see `no_design_availability`. It is a constant
    # across posterior draws, which is the honest shape of a plugged-in empirical prior:
    # no posterior on it, so the predictive does not integrate over its uncertainty. It is
    # a per-player constant rather than a league-wide one, because the population it covers
    # spans 3.33x in realized level (`make availability-no-prior`, §8b).
    level_arm = no_design_level_arm(cfg)
    no_design = no_design_availability(features_dir, season, full_avail,
                                       allowed_seasons(design), list(cfg["data"]["seasons"]),
                                       player_ids, level_arm)
    mu = np.tile(no_design.to_numpy(dtype=float), (avail_art.n_draws, 1))
    if present.any():
        mu[:, present] = avail_art.mu_draws(avail[present])
    # `(draws x n_rho)` in both arms, so the gather below has one shape to handle. The
    # shared-dispersion artifact persists `(draws,)`; `rehydrate` does the same promotion.
    avail_rho = np.asarray(avail_art.draws["rho_draws"], dtype=float)
    if avail_rho.ndim == 1:
        avail_rho = avail_rho[:, None]
    avail_rho_bin = availability_rho_bin(avail_art, avail)
    # The low-availability mixture's weight, from the artifact's OWN second design block —
    # `(draws x players)`, and all zeros when the persisted head carries no mixture, which
    # is the single-component draw exactly. A player with no design row keeps `pi = 0`
    # deliberately: his `mu` is `no_design_availability`'s empirical rate over players like
    # him, which already contains their disrupted seasons, so a mixture on top of it would
    # discount the same absences twice.
    avail_pi = np.zeros_like(mu)
    if present.any():
        avail_pi[:, present] = avail_art.pi_draws(avail[present])
    avail_mu_low = np.asarray(avail_art.draws.get("mu_low_draws",
                                                  np.zeros(avail_art.n_draws)), dtype=float)
    avail_rho_low = np.asarray(avail_art.draws.get("rho_low_draws",
                                                   np.zeros(avail_art.n_draws)), dtype=float)

    comp_art = artifacts["composition"]
    comp_model = rehydrate_composition(comp_art, comp_art.n_draws,
                                       injected_sigma=shipped_sigma(cfg))
    eta = composition_eta(comp_art, comp_model, players)
    ps_sigma = (np.asarray(comp_model.ps.sigma_draws, dtype=float)
                if comp_model.ps.enabled else np.zeros(comp_art.n_draws))
    row_rho_bin = comp_art.recipe.transform(players)["rho_bin"].to_numpy(np.int64)[row_player]

    dur_mu, dur_kappa = spell_shape(artifacts["gp_duration"], avail[present])
    layout_name, layout_tenure_on, layout_overflow = layout_arm(cfg)
    edges = (tenure_edges(features_dir, season, full_avail, allowed_seasons(design),
                          cfg.get("stan", {}).get("availability", {})
                          .get("first_season", FIRST_SEASON))
             if layout_tenure_on else None)
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
        "avail_mu": mu, "avail_rho": avail_rho, "avail_rho_bin": avail_rho_bin,
        "avail_pi": avail_pi, "avail_mu_low": avail_mu_low,
        "avail_rho_low": avail_rho_low,
        "avail_mixture": bool(avail_art.extras.get("mixture", False)),
        # One label per dispersion column, always — a legacy artifact carries none, and a
        # progress line that silently printed three of four buckets would be worse than one
        # that printed indices.
        "avail_rho_labels": (list(avail_art.extras["rho_labels"])
                             if len(avail_art.extras.get("rho_labels", ()))
                             == avail_rho.shape[1]
                             else [f"bucket {j + 1}" for j in range(avail_rho.shape[1])]),
        "dur_mu": dur_mu, "dur_kappa": dur_kappa,
        # The layout's role key comes from `role_bins` directly rather than from the
        # dispersion artifact's bucket, even though the two agree on every player of every
        # simulated season today. They agree only while the head is role-graded: a
        # shared-`rho` artifact makes `availability_rho_bin` return all zeros, which is
        # correct for a dispersion gather and would silently key every player in the league
        # to the fringe bucket here. `role_bins` is the function `EdgeResampler` pooled on,
        # so it is the one that cannot drift from it — and it sends a player with no prior
        # minutes to the lowest bucket, whose edge blocks are longest, which is the right
        # direction for a call-up.
        "edges": edges, "layout": layout_name, "layout_overflow": layout_overflow,
        "cell_role": role_bins(avail)[cell_frame["player_id"].map(player_pos).to_numpy(
            np.int64)],
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
        "no_design_availability": no_design[~present],
        "no_design_level": level_arm,
        "composition_variant": comp_art.recipe.variant,
    }


def simulate(ctx: dict) -> dict:
    """The sim loop. Returns the two tensors plus everything Gate A reads."""
    n_sims, n_units = ctx["n_sims"], ctx["n_units"]
    points = np.zeros((n_sims, ctx["n_flat"]), dtype=np.float32)
    games = np.zeros((n_sims, ctx["n_flat"]), dtype=np.uint8)
    season_minutes = np.zeros((n_sims, n_units))
    season_gp = np.zeros((n_sims, ctx["n_players"]))
    # Accumulated as a mean rather than kept per sim: the team-level allocation check below
    # is a share of a pot that is fixed within every draw, so nothing is lost by averaging
    # first, and a (sims x players) array is one the layer otherwise never materializes.
    player_minutes = np.zeros(ctx["n_players"])
    bonus_total, played_rows, short = 0.0, 0, 0
    probe = None

    step = max(n_sims // 10, 1)
    for s in range(n_sims):
        out = _sim_one(s, ctx)
        points[s] = out["points"]
        games[s] = out["games"]
        season_minutes[s] = out["season_minutes"]
        season_gp[s] = out["season_gp"]
        player_minutes += out["season_minutes_player"]
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
        "player_minutes": player_minutes / n_sims,
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


def no_design_team_minutes(cfg: dict, ctx: dict, sim: dict) -> dict:
    """Team-by-team, how much of the season's minutes the no-design players absorbed.

    The check the level grading has to pass and a player-level metric cannot see. A team's
    season minutes are a **fixed pot** — `5 x game_length` per team-game, allocated exactly
    by the composition — so every minute handed to a rostered player the availability head
    has no row for is a minute taken from a teammate the tensor *does* score. A rate that is
    too high for a call-up and too low for a top-5 pick can be right on average across the
    league and wrong on all thirty teams, and the errors do not cancel within a roster.

    Realized minutes are joined on `(player_id, game_id)` against the simulator's own grid
    rather than summed per player over the season, so a traded player contributes exactly
    the games the grid gave him and both sides share a denominator.
    """
    targets = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "component_targets.parquet",
                              columns=["player_id", "season", "season_type", "game_id",
                                       "min", "played"])
    rows = targets[(targets["season"] == ctx["season"])
                   & (targets["season_type"] == "regular") & (targets["played"] == 1)]
    grid = ctx["grid"][["player_id", "team_id", "game_id"]]
    realized = (grid.merge(rows[["player_id", "game_id", "min"]],
                           on=["player_id", "game_id"], how="left").fillna({"min": 0.0})
                .groupby("player_id", as_index=False).agg(minutes=("min", "sum")))

    frame = pd.DataFrame({"player_id": ctx["player_ids"],
                          "simulated": sim["player_minutes"]})
    frame = frame.merge(realized, on="player_id", how="left").fillna({"minutes": 0.0})
    frame = frame.merge(grid.drop_duplicates("player_id")[["player_id", "team_id"]],
                        on="player_id", how="left")
    frame["no_design"] = np.isin(ctx["player_ids"], np.asarray(
        ctx["no_design_availability"].index, dtype=ctx["player_ids"].dtype))

    by_team = frame.groupby("team_id").apply(
        lambda g: pd.Series({
            "sim_share": g.loc[g["no_design"], "simulated"].sum()
            / max(g["simulated"].sum(), 1e-9),
            "obs_share": g.loc[g["no_design"], "minutes"].sum()
            / max(g["minutes"].sum(), 1e-9)}), include_groups=False)
    error = (by_team["sim_share"] - by_team["obs_share"]).to_numpy()
    return {"n": int(len(by_team)),
            "value": float(frame.loc[frame["no_design"], "simulated"].sum()
                           / max(frame["simulated"].sum(), 1e-9)),
            "bar_value": float(frame.loc[frame["no_design"], "minutes"].sum()
                               / max(frame["minutes"].sum(), 1e-9)),
            "mae": float(np.abs(error).mean()), "bias": float(error.mean()),
            "rmse": float(np.sqrt((error ** 2).mean()))}


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

    # ── 5. where the no-design players' minutes came from ────────────────────
    rows.append({"check": "no_design_team_minutes_share", "unit": "team-season",
                 "gate": "diagnostic", "artifact": "component_targets.parquet",
                 "bar_source": "realized, on the simulator's own grid rows",
                 "no_design_level": ctx["no_design_level"],
                 **no_design_team_minutes(cfg, ctx, sim)})

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
        # The availability layout belongs beside the composition variant and the injected
        # sigma for the same reason those two are here: it changes the tensor materially
        # while leaving every season marginal identical, so a consumer holding two tensors
        # drawn under different layouts has no other way to tell them apart.
        availability_layout=np.array(ctx["layout"]),
        # And so does the no-design level key, for the same reason one level up: it moves
        # minutes between rostered players without changing any head.
        no_design_level=np.array(ctx["no_design_level"]),
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

    design = component_head_design(cfg)
    seasons = seasons or validation_seasons(design)

    print(f"Season simulator — the player x scoring-period x sim tensor")
    print(f"  posteriors at the `{window}` window; nothing is refitted, and nothing "
          f"here needs CmdStan.")
    print(f"  The test split is LOCKED — `allowed_seasons` goes through "
          f"`held_out.selection_split`.")
    print(f"  {N_SCORING_PERIODS} scoring periods "
          f"({ROUND_1_WEEKS} Round-1 weeks + 3 double weeks), {n_sims:,} sims, "
          f"seed {seed}")

    composition = head_frame(cfg)
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
        nd = ctx["no_design_availability"]
        print(f"  availability: the head where it has a row; where it does not, an "
              f"expanding-window empirical rate\n"
              f"    graded `{ctx['no_design_level']}` over {len(nd):,} players — "
              + (f"{nd.min():.4f} to {nd.max():.4f}, mean {nd.mean():.4f}"
                 if len(nd) else "none on this roster"))
        counts = np.bincount(ctx["avail_rho_bin"], minlength=ctx["avail_rho"].shape[1])
        print("    dispersion: " + ", ".join(
            f"{label} rho {ctx['avail_rho'][:, j].mean():.4f} ({n:,} players)"
            for j, (label, n) in enumerate(zip(ctx["avail_rho_labels"], counts))))
        if ctx["avail_mixture"]:
            pi = ctx["avail_pi"].mean(axis=0)
            print(f"    mixture: pi {pi.mean():.4f} mean, "
                  f"{np.percentile(pi, 10):.4f}-{np.percentile(pi, 90):.4f} "
                  f"p10-p90, low component {ctx['avail_mu_low'].mean():.4f} "
                  f"at rho {ctx['avail_rho_low'].mean():.4f}")
        print(f"    layout: {ctx['layout']} — "
              + (f"tenure edge blocks pooled from {len(ctx['edges'].pairs):,} prior "
                 f"player-seasons, " if ctx["edges"] is not None
                 else "no tenure factor, ")
              + f"overflow {ctx['layout_overflow']}")
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
            if row["check"] == "no_design_team_minutes_share":
                extra = (f"  |  per-team share error MAE {float(row['mae']):.4f}, "
                         f"bias {float(row['bias']):+.4f} over {int(row['n'])} teams "
                         f"at `{row['no_design_level']}`")
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
