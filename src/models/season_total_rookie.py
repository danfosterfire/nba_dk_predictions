"""The season-total readout for the structurally-missing population —
`docs/rookie-rates-plan.md` §5e, which is `docs/potential-to-dos.md` §16's settling gate.

`make season-total-rookie`. One artifact, `outputs/predictions/season_total_rookie.csv`.

## The question, and why it needs its own frame

Sessions 3 and 4 measured the rookie rate heads at the **head's** unit: a season total of
rebounds, of assists, of made threes. §4's per-head gate reads a CRPS in rebounds. That is
the right unit for deciding which arm a head ships, and it is not the unit anything is
drafted in. §16's own gate is one level down — **the head family against the floor family
in season-total dk_pts** — and eleven heads each moving by a fraction can compound or
cancel, which only composing them shows.

`season_total.build_frame` cannot carry this population, and it drops it **twice over on
lag columns**: the availability design is a lag-1 design with no row for a player who has
no prior season, and `RATE_FEATURES` then requires `dk_per_game_lag1`,
`minutes_per_game_lag1` and `gp_share_lag1`. So this module is that builder's
rookie-admitting twin. It shares the scorer — `season_total.evaluate`, whose group tuple
§5e widened — and nothing else.

## The three populations, and that they are not pooled

| group | rows | rate side | games-played side |
|---|---|---|---|
| `veteran` | the shipped design, rung 0 | the persisted `train` posteriors | the availability head |
| `lag_recovered` | the rungs §7c **admitted** | the same posteriors, imputed lag | `no_design_availability` |
| `rookie` | true rookies (§7d) | §7e's ship split | `no_design_availability` |

Two of these are §5e's required groups and the third is the bar. `lag_ladder` carries rung
0 beside its rungs for the same reason: a season-total MAE in dk_pts means nothing without
the shipped population's own number next to it. **Pooling `rookie` with `lag_recovered`
would report an average of two regimes**, which is the mistake §3 constraint 2' exists to
undo — they are different populations, served by different heads, through different designs.

The availability split is not a modelling choice here, it is a measured fact: the
availability design covers **100%** of the `veteran` and thin-prior rows and **0%** of the
rookie and returnee rows. So "the shipped graded `no_design_availability` level" is the
games treatment for exactly the population §5e names it for, and the veteran bar keeps the
head — which is what "shipped" means for each of them.

## Three rate arms, each the family's own version of the same thing

- **`unserved`** — the status quo for a row that is not in the design: it is not in the
  tensor, so every draw scores it at zero and its predicted season total is 0. This is
  §7c's gate-1 baseline, one level down.
- **`floor`** — the family's own no-fit floor. For `rookie` that is §7d's eleven
  dispersion-wrapped preseason blends; for `veteran` and `lag_recovered` it is
  `component_rates.carry_forward` / `carry_forward_conversion`, which is the floor those
  eleven heads have always been read against.
- **`head`** — what ships. For `rookie` that is §7e's split, read from
  `rookie_rate_metrics.csv` rather than pinned here: `reb` fitted at the spline rung, ten
  floors as plug-ins. For the veteran family it is the persisted `train`-window posteriors,
  which score the ladder's recovered rows with coefficients fitted before the ladder
  existed — §3 constraint 4, and the same discipline `lag_ladder` runs its gate under.

Each arm is crossed with the games-played side twice: once at the shipped level and once at
`oracle_gp`. That is `season_total`'s own oracle design and it is here for its own reason —
without it a headline MAE cannot say whether the error is the rate family or the
availability plug-in, and the rate-side number is the one comparable to
`stan_component_metrics.csv`.

## The composition, and where the bonus comes from

`season_terms.compose_season_dk`'s chain, one family over: the seven counts are drawn from
their heads, the share head draws `fg3a` on the **drawn** `fga`, `fg2a` is the difference,
and the three make-heads draw on drawn trials. Drawing makes on *realized* attempts would
leak the target and understate the spread, since attempt uncertainty is most of the
uncertainty in points.

The bonus is a per-game threshold and a season total genuinely cannot carry it — but the
project already has the device for exactly this unit. `expected_bonus` at
`BONUS_OVERDISPERSION` is calibrated on player-season **mean per-game counts**, which is
what an arm's drawn season totals divided by its games played are, and its recorded
aggregate bias there is +0.0009 dk_pts per game. So each arm's per-row bonus is its own
expectation at its own mean counts, added to its drawn linear total, and the target stays
the realized `dk_total` a board is scored in. It is **constant across draws**, deliberately
— the same honest shape as the availability plug-in, and the alternative is putting Monte
Carlo noise inside the CRPS the gate reads. The bonus is 0.4-0.7% of the realized season
total on these populations, and the artifact carries that share per group so the claim is
checkable rather than asserted.

## What is fitted here, and what is not

**One head.** `reb` is the single arm §7e ships fitted, and no posterior for it exists until
Session 6 (§5f) persists the `rookie-components` group — so it is refitted here, on the
rookie fitting half, at the variant the artifact names. Everything else is read: the veteran
posteriors off disk, the floors' constants out of `rookie_rate_floors.csv`, the admitted
rungs out of `lag_ladder.csv`. Nothing reads the test split; `held_out.selection_split` is
the only route to a season.

Usage:
    python -m src.models.season_total_rookie
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.eda.preseason_value import attach_season_start_roster
from src.features.availability import build_panel, season_availability
from src.features.targets import BONUS_CATEGORIES, BONUS_OVERDISPERSION, expected_bonus
from src.models import rookie_rates as rr
from src.models import stan_rookie as sr
from src.models.availability import fit_dispersion
from src.models.component_rates import (CONVERSION_HEADS, COUNT_HEADS, DERIVED_COUNTS,
                                        LADDER_RUNGS, TEST_SEASONS, build_design,
                                        carry_forward, carry_forward_conversion,
                                        fit_nb_dispersion, lag_ladder)
from src.models.held_out import selection_split
from src.models.minutes_unification import paired_bootstrap, verdict
from src.models.posteriors import load_all, posteriors_dir
from src.models.season_terms import DK_LINEAR_WEIGHTS
from src.models.season_total import (POPULATION_COLUMN, POPULATION_GROUPS,
                                     evaluate)
from src.models.availability import lag_ladder as availability_lag_ladder
from src.models.stan_availability import head_design as availability_head_design
from src.models.stan_components import (PRESEASON, SPLINE_KNOTS, _beta_shapes,
                                        head_design as component_head_design)
from src.models.stan_utils import crps_from_samples, thin
from src.sim.season import (allowed_seasons, artifact_name, no_design_availability,
                            no_design_level_arm)

SEED = 42

#: Posterior draws every arm is collapsed to. `season_terms.COMPOSITION_DRAWS`, and for its
#: reason: the composition is eleven heads deep, so the draw axis is the memory cost and 400
#: is already far below the Monte-Carlo noise of a season-total CRPS.
DRAWS = 400

#: The bootstrap replicate count every paired reading in this program uses
#: (`lag_ladder.BOOTSTRAP_REPS`, `stan_rookie.BOOTSTRAP_REPS`).
BOOTSTRAP_REPS = 2000

#: The window whose posteriors score the veteran family. `train` ends at 2021-22, so the
#: validation seasons are outside it — `lag_ladder.SCORING_WINDOW`, unchanged, because the
#: claim under test is the same one: these coefficients were fitted before the ladder.
SCORING_WINDOW = "train"

#: The three labels this module writes into `season_total.POPULATION_COLUMN`. `veteran` is
#: rung 0's name in `component_rates`' own `lag_rung` vocabulary, so a board reconciles with
#: §7c's census without a mapping table; the other two are §5e's required groups.
VETERAN_RUNG = "veteran"
LADDER_GROUP = "lag_recovered"
ROOKIE_GROUP = "rookie"

#: The order the two design families are stacked in. `run` builds the metadata frame in it
#: and `rate_arms` composes its draws in it, and the two are asserted against each other.
FAMILY_ORDER = ("veteran", "rookie")

#: The three populations, in the order every table reports them — `season_total`'s own
#: tuple, imported rather than restated, because `evaluate` is what builds the masks from it
#: and a second copy here could name a group the scorer never scores.
GROUPS = POPULATION_GROUPS
if set(GROUPS) != {VETERAN_RUNG, LADDER_GROUP, ROOKIE_GROUP}:
    raise AssertionError(
        f"season_total.POPULATION_GROUPS is {GROUPS}, which is not the set of labels this "
        f"module writes into its population column — every mask on the difference would be "
        f"empty and the table would lose a group silently")

#: The rate arms, in the order the report prints them. `oracle_gp_*` are the same two arms
#: with the games-played side replaced by the realized count.
RATE_ARMS = ("unserved", "floor", "head")
ORDER = RATE_ARMS + tuple(f"oracle_gp_{a}" for a in RATE_ARMS[1:])

#: The row restrictions the table is reported at. `draftable` is P1 decision 5's
#: season-start-roster population and the one §4 reads its per-head verdict on; `all` is
#: beside it because the gap between the two says whether a figure is carried by rows no
#: board ever prices.
RESTRICTIONS = ("all", "draftable")
DECISION_RESTRICTION = "draftable"


# ── Which rungs the ladder admitted ───────────────────────────────────────────

def admitted_rungs_from(table: pd.DataFrame) -> tuple[str, ...]:
    """The rungs a `lag_ladder.csv` table's own **verdict** rows admitted.

    Only `gate == "verdict"`, because the same rung has a passing row under `crps` that is
    half a conjunction rather than a verdict, and only names in `LADDER_RUNGS`, because rung
    0 is the bar and is admitted by not being a rung at all.
    """
    hit = table[(table["gate"] == "verdict") & (table["metric"] == "admitted")
                & (table["value"] == 1.0)]
    return tuple(str(g) for g in hit["group"] if g in LADDER_RUNGS)


def admitted_rungs(path: Path | str) -> tuple[str, ...]:
    """The rungs §7c's gate admitted, read from `lag_ladder.csv`.

    **Not from `stan.components.lag_ladder`**, and the difference matters this session.
    That key is deliberately still `[]`: turning the ladder on is Session 6's first act and
    is two edits, because flipping the key alone would widen eleven heads' *fitting*
    population. The verdict is nonetheless taken — `returnee_lag2` — and it is the verdict
    §5e asks this table to report a group for, so the artifact that recorded it is the
    right source.
    """
    return admitted_rungs_from(pd.read_csv(path))


# ── The rookie-admitting frame ────────────────────────────────────────────────

def realized_totals(targets: pd.DataFrame) -> pd.DataFrame:
    """Per (player, season): games played, the DK total, and its two parts.

    `dk_linear` and `dk_bonus` are per-game columns `features/targets.py` already writes and
    which sum to `dk_pts` exactly, so the split this table reports as context is read rather
    than re-derived. Keyed on games **played**, which is `season_total.season_rates`' own
    choice and for its reason: a zero-minute row scores zero and says nothing about a rate.
    """
    played = targets[targets["played"] == 1]
    out = (played.groupby(["player_id", "season"], as_index=False)
           .agg(gp_played=("dk_pts", "size"), dk_total=("dk_pts", "sum"),
                dk_linear_total=("dk_linear", "sum"),
                dk_bonus_total=("dk_bonus", "sum")))
    out["dk_per_game"] = out["dk_total"] / out["gp_played"]
    return out


def veteran_frames(cfg: dict, rungs: tuple[str, ...]
                   ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(train, validation)` on the veteran design widened by the ADMITTED rungs only.

    `lag_ladder` builds every rung because it is the measurement the config key is written
    from. This table reports the rungs that *ship*, so it builds those — and rung 0 comes
    along as the bar, since `build_design` labels every row with `lag_rung` whatever the
    ladder is set to.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    design = build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"],
                          ladder=lag_ladder(cfg, rungs=rungs) if rungs else None)
    preseason = bool(cfg.get("stan", {}).get("components", {})
                     .get("preseason", PRESEASON))
    return selection_split(component_head_design(cfg, preseason, design=design),
                           TEST_SEASONS)


def label_population(frame: pd.DataFrame, rungs: tuple[str, ...]) -> pd.Series:
    """`veteran` / `lag_recovered` off the design's own `lag_rung` column."""
    rung = frame["lag_rung"].to_numpy()
    return pd.Series(np.where(np.isin(rung, list(rungs)), LADDER_GROUP, VETERAN_RUNG),
                     index=frame.index)


def attach_context(frame: pd.DataFrame, cfg: dict, totals: pd.DataFrame,
                   team_games: pd.DataFrame) -> pd.DataFrame:
    """Realized season totals, the schedule denominator, and the roster restriction.

    Every column `season_total.evaluate` reads and every column a group mask reads, on a
    frame that is otherwise one family's design. Merged rather than recomputed so the two
    families are scored against literally the same realized numbers.
    """
    out = frame.merge(totals, on=["player_id", "season"], how="left")
    out = out.merge(team_games, on=["player_id", "season"], how="left")
    if len(out) != len(frame):
        raise AssertionError("a realized-total or team-games merge duplicated a row")
    missing = out[["dk_total", "team_games"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(
            f"{int(missing.sum())} design rows have no realized season total or no team "
            f"games; the component design and the availability panel disagree about which "
            f"player-seasons exist")
    if "on_season_start_roster" not in out.columns:
        window = int(cfg.get("features", {}).get("team_context", {})
                     .get("roster_window_games", 10))
        out = attach_season_start_roster(out, list(cfg["data"]["seasons"]),
                                         cfg["data"]["raw_dir"], window)
    return out


def build_frame(cfg: dict) -> dict:
    """The rookie-admitting frame, as its three parts plus the scoring metadata.

    Returned as parts rather than one concatenated design, because the three carry
    *different columns* — a lag block the rookie family does not have, a preseason level
    block the veteran family does not have — and a concatenation would be a frame of NaNs
    whose zero means two different things. The parts are scored where they live and only
    the metadata is stacked, in this order, which is what makes the row alignment between
    `frame` and every arm's draws a fact rather than a hope.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    targets = pd.read_parquet(features_dir / "component_targets.parquet")
    totals = realized_totals(targets)

    panel = build_panel(cfg["data"]["seasons"], cfg["data"]["raw_dir"])
    availability = season_availability(panel, "full")
    team_games = availability[["player_id", "season", "team_games"]]

    rungs = admitted_rungs(out_dir / "lag_ladder.csv")
    vet_train, vet_val = veteran_frames(cfg, rungs)
    rk_train, rk_val, covered = sr.fitting_frames(cfg)

    vet_val = attach_context(vet_val, cfg, totals, team_games)
    vet_val[POPULATION_COLUMN] = label_population(vet_val, rungs).to_numpy()
    rk_val = attach_context(rk_val, cfg, totals, team_games)
    rk_val[POPULATION_COLUMN] = ROOKIE_GROUP

    return {"rungs": rungs, "covered": covered,
            "veteran": {"train": vet_train, "val": vet_val.reset_index(drop=True)},
            "rookie": {"train": rk_train, "val": rk_val.reset_index(drop=True)}}


# ── The games-played side ─────────────────────────────────────────────────────

def games_played(cfg: dict, val: pd.DataFrame, artifact,
                 design: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """`(predicted games played, a `no_design` flag)` under the shipped treatment.

    Two sources, and which one a row gets is a fact about the availability design rather
    than a choice: a player it has a row for is scored by the head, and one it does not is
    scored by `no_design_availability`'s graded empirical level — the arrangement
    `sim/season.py::build_context` already ships, reused rather than restated so the games
    a rookie is given here are the games he is given in the tensor.

    The head's contribution is the **posterior mean** rather than its pmf, because the
    no-design half is a plug-in constant with no posterior at all, and a table whose two
    halves carried predictives of different kinds would be reporting the difference between
    them as if it were a difference between populations.
    """
    seasons = list(cfg["data"]["seasons"])
    features_dir = Path(cfg["data"]["features_dir"])
    arm = no_design_level_arm(cfg)
    allowed = allowed_seasons(design)

    rate = np.zeros(len(val), dtype=float)
    plugged = np.zeros(len(val), dtype=bool)
    covered = design.drop_duplicates(subset=["season", "player_id"])
    for season, block in val.groupby("season", sort=True):
        rows = block.index.to_numpy()
        ids = block["player_id"].to_numpy()
        seat = covered[covered["season"] == season].set_index("player_id")
        present = np.isin(ids, seat.index.to_numpy())
        if present.any():
            head_rows = seat.reindex(ids[present]).reset_index()
            rate[rows[present]] = artifact.mu_draws(head_rows).mean(axis=0)
        if (~present).any():
            level = no_design_availability(features_dir, str(season), design, allowed,
                                           seasons, ids[~present], arm)
            rate[rows[~present]] = level.to_numpy(dtype=float)
            plugged[rows[~present]] = True
    return np.clip(rate, 0.0, 1.0) * val["team_games"].to_numpy(dtype=float), plugged


# ── The rate side: one composition, four arms that feed it ────────────────────

def _nb_draws(mu: np.ndarray, phi: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """`(draws x rows)` negative-binomial season totals at `mu`, dispersion `phi`."""
    mean = np.clip(mu, 1e-9, None)
    shape = np.broadcast_to(np.asarray(phi, dtype=float)[:, None], mean.shape)
    return rng.negative_binomial(shape, shape / (shape + mean)).astype(float)


def _tile(values: np.ndarray, draws: int) -> np.ndarray:
    """A per-row constant as `(draws x rows)` — the honest shape of a plug-in estimator."""
    return np.repeat(np.asarray(values, dtype=float)[None, :], draws, axis=0)


def compose_linear_dk(counts: dict[str, np.ndarray],
                      conversions: dict[str, tuple[np.ndarray, np.ndarray]],
                      seed: int = SEED) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(draws x rows season dk excluding the bonus, the drawn season box score)`.

    `season_terms.compose_season_dk`'s chain exactly — attempts drawn from the count heads,
    then makes drawn **conditional on the drawn attempts** — reimplemented here for one
    reason only: that function reads fitted `StanCount` / `StanConversion` objects off a
    `models` dict, and three of the four arms this module composes are not fitted models at
    all.

    The whole drawn box score comes back rather than only the total, because a season total
    is only half of dk_pts: the bonus is a threshold on five per-game counts and needs them
    named. It is also what makes the chain checkable — `fg2a = fga - fg3a` and
    `made <= attempted` are identities that hold on every draw, and a chain assembled in the
    wrong order breaks them while still returning a plausible number.
    """
    rng = np.random.default_rng(seed)
    resolved = dict(counts)
    made: dict[str, np.ndarray] = {}

    def trials_for(name: str) -> np.ndarray:
        if name in resolved:
            return resolved[name]
        if name in made:
            resolved[name] = made[name]
            return resolved[name]
        if name in DERIVED_COUNTS:
            total, part = DERIVED_COUNTS[name]
            resolved[name] = np.maximum(trials_for(total) - trials_for(part), 0.0)
            return resolved[name]
        raise KeyError(f"{name} is needed as trials but is neither a count head nor a "
                       f"conversion head drawn earlier in the chain nor derived")

    for m, attempted in CONVERSION_HEADS:
        p, rho = conversions[f"{m}|{attempted}"]
        a, b = _beta_shapes(p, np.asarray(rho, dtype=float)[:, None])
        made[m] = rng.binomial(np.rint(trials_for(attempted)).astype(np.int64),
                               rng.beta(a, b)).astype(float)

    pts = 2.0 * made["fg2m"] + 3.0 * made["fg3m"] + made["ftm"]
    total = pts + 0.5 * made["fg3m"]
    for col, weight in DK_LINEAR_WEIGHTS.items():
        total = total + weight * counts[col]
    return total, {"pts": pts, **resolved, **made}


def season_bonus(box: dict[str, np.ndarray], gp_played: np.ndarray,
                 seed: int = SEED) -> np.ndarray:
    """Per-row expected season bonus from an arm's own mean per-game counts.

    `BONUS_OVERDISPERSION` is calibrated at exactly this unit — a player-season's mean
    per-game counts, where the frailty also stands in for the minutes variation a season
    mean hides — with a recorded aggregate bias of +0.0009 dk_pts per game
    (`outputs/eda/bonus_calibration.csv`). Called once per arm on the arm's posterior mean
    rather than once per draw: the bonus is 0.4-0.7% of these populations' season totals,
    and Monte-Carlo noise inside the CRPS the gate reads would cost more than the spread it
    would add.
    """
    gp = np.clip(np.asarray(gp_played, dtype=float), 1.0, None)
    per_game = np.column_stack([box[c].mean(axis=0) / gp for c in BONUS_CATEGORIES])
    return expected_bonus(np.clip(per_game, 0.0, None), BONUS_OVERDISPERSION,
                          seed=seed) * gp


def veteran_head_arm(artifacts: dict, frame: pd.DataFrame, draws: int = DRAWS,
                     seed: int = SEED) -> tuple[dict, dict]:
    """The persisted `train`-window posteriors, thinned to `draws`. Nothing is fitted.

    The recovered rows are scored by coefficients that were fitted before the ladder
    existed, which is the whole of §3 constraint 4 and the same arrangement `lag_ladder`'s
    gate runs under — only there the unit was one head's season total and here it is the
    eleven of them composed.
    """
    rng = np.random.default_rng(seed)
    exposure = frame["total_minutes"].to_numpy(dtype=float)[None, :]
    counts = {}
    for component in COUNT_HEADS:
        art = artifacts[artifact_name(component)]
        phi = np.asarray(art.draws["phi_draws"], dtype=float)
        idx = thin(len(phi), draws)
        counts[component] = _nb_draws(art.mu_draws(frame)[idx] * exposure, phi[idx], rng)
    conversions = {}
    for made, attempted in CONVERSION_HEADS:
        label = f"{made}|{attempted}"
        art = artifacts[artifact_name(label)]
        rho = np.asarray(art.draws["rho_draws"], dtype=float).reshape(-1)
        idx = thin(len(rho), draws)
        conversions[label] = (art.mu_draws(frame)[idx], rho[idx])
    return counts, conversions


def veteran_floor_arm(train: pd.DataFrame, frame: pd.DataFrame, draws: int = DRAWS,
                      seed: int = SEED) -> tuple[dict, dict]:
    """`carry_forward` and `carry_forward_conversion`, in each head's own likelihood.

    `stan_components.count_floor` / `conversion_floor` wrapped for composition rather than
    for scoring: the same `mu`, `phi`, `p` and `rho`, but a percentage on **every** row
    rather than only the rows with realized attempts, because the chain draws makes on
    drawn trials.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    for component in COUNT_HEADS:
        mu = np.clip(carry_forward(frame, component), 1e-6, None)
        phi = fit_nb_dispersion(train[component].to_numpy(dtype=float),
                                np.clip(carry_forward(train, component), 1e-6, None))
        counts[component] = _nb_draws(_tile(mu, draws), np.full(draws, phi), rng)
    conversions = {}
    for made, attempted in CONVERSION_HEADS:
        live = train[attempted].to_numpy(dtype=float) > 0
        rho = fit_dispersion(
            np.rint(train[made].to_numpy(dtype=float)[live]).astype(int),
            np.rint(train[attempted].to_numpy(dtype=float)[live]).astype(int),
            carry_forward_conversion(train, train, made, attempted)[live])
        p = carry_forward_conversion(train, frame, made, attempted)
        conversions[f"{made}|{attempted}"] = (_tile(p, draws), np.full(draws, rho))
    return counts, conversions


def rookie_floor_arm(train: pd.DataFrame, frame: pd.DataFrame, priors: dict,
                     constants: rr.RookieConstants, draws: int = DRAWS,
                     seed: int = SEED) -> tuple[dict, dict]:
    """§7d's eleven no-fit floors, in the shape the composition consumes.

    Every piece comes out of `rookie_rates` rather than being restated: the counts' `mu`
    and `phi` off `count_floor_predictive`, the conversions' percentage off
    `conversion_floor_p` and their `rho` off `conversion_floor_predictive`. A second
    implementation of the floor is a place where the benchmark §4's gate was read against
    and the benchmark this table composes could silently disagree.

    `train` is the whole fitting half and each head takes `stan_rookie.fitting_rows` of it
    with its **own** prior table, which is that module's arrangement and for its reason:
    `bucket_priors` emits a season only when the head's target has history, so a head is
    entitled to a different answer even though in practice every one of them loses exactly
    the first covered season.
    """
    rng = np.random.default_rng(seed)
    counts = {}
    for component in COUNT_HEADS:
        fit = sr.fitting_rows(train, priors[component])
        _, mu, _, phi = rr.count_floor_predictive(fit, frame, priors[component],
                                                  component, constants.volume[component],
                                                  rr.FLOOR_ARM, seed)
        counts[component] = _nb_draws(_tile(mu, draws), np.full(draws, phi), rng)
    conversions = {}
    for made, attempted in CONVERSION_HEADS:
        k_attempts, league = constants.conversion[made]
        volume_k = constants.volume[made]
        fit = sr.fitting_rows(train, priors[made])
        *_, rho = rr.conversion_floor_predictive(fit, frame, priors[made], made,
                                                 attempted, volume_k, k_attempts, league,
                                                 rr.FLOOR_ARM, seed)
        p = rr.conversion_floor_p(frame, priors[made], made, volume_k, k_attempts, league,
                                  rr.FLOOR_ARM)
        conversions[f"{made}|{attempted}"] = (_tile(p, draws), np.full(draws, rho))
    return counts, conversions


def ship_arms_from(table: pd.DataFrame) -> dict[str, str]:
    """`head -> the variant it ships`, off a `rookie_rate_metrics.csv` table.

    Read on `stan_rookie.DECISION_POPULATION` and nowhere else: `ships` is written onto
    every row of a head's block, but §4 reads its verdict on the draftable population and a
    table filtered any other way would be a different gate's answer.
    """
    block = table[table["population"] == sr.DECISION_POPULATION]
    return {str(head): str(part["ships"].iloc[0])
            for head, part in block.groupby("head")}


def ship_arms(path: Path | str) -> dict[str, str]:
    """`head -> the variant it ships`, from `rookie_rate_metrics.csv`.

    Read rather than pinned, because §4's gate is what decides it and a constant copied
    into a module is a constant that can stop matching the run that measured it. Today that
    is `reb` at `slot_interaction_spline` and ten `no_fit_floor`s.
    """
    return ship_arms_from(pd.read_csv(path))


def rookie_head_arm(train: pd.DataFrame, frame: pd.DataFrame, priors: dict,
                    constants: rr.RookieConstants, ships: dict[str, str], cfg: dict,
                    draws: int = DRAWS, seed: int = SEED) -> tuple[dict, dict, list[dict]]:
    """§7e's ship split — the fitted arms where the gate passed, the floor everywhere else.

    Starts from `rookie_floor_arm` and **overwrites** the heads that ship fitted, which is
    what "the gate decides which arm, never whether" means as code: every unit carries all
    eleven quantities either way, so the composition is never short a head.

    `reb` is the one head that ships today and no posterior for it exists yet — Session 6
    (§5f) is what persists the `rookie-components` group — so it is refitted here on the
    rookie fitting half at the variant the artifact names. At `dense_e` that is a few
    seconds; see `stan_rookie.METRIC` for why the mass matrix is not a preference.
    """
    counts, conversions = rookie_floor_arm(train, frame, priors, constants, draws, seed)
    cfg_stan = cfg.get("stan", {})
    chains = int(cfg_stan.get("chains", 4))
    iters = sr._iters(cfg_stan)
    n_knots = int(cfg_stan.get("components", {}).get("spline_knots", SPLINE_KNOTS))
    rng = np.random.default_rng(seed)

    diagnostics = []
    for label, component, attempted in rr.head_list():
        variant = ships.get(label, sr.FLOOR_VARIANT)
        if variant == sr.FLOOR_VARIANT:
            continue
        fit = sr.fitting_rows(train, priors[component])
        v_train, v_test, features = sr.variants(fit, frame, component, n_knots)[variant]
        model = _fit_head(v_train, features, component, attempted,
                          f"{label}/{variant}/season-total", chains, iters, seed)
        diagnostics.append(model.diagnostics)
        if attempted is None:
            mu, phi = model.mu_draws(v_test, draws)
            counts[component] = _nb_draws(mu, phi, rng)
        else:
            conversions[label] = model.p_draws(v_test, draws)
    return counts, conversions, diagnostics


def _fit_head(train: pd.DataFrame, features: list[str], component: str,
              attempted: str | None, label: str, chains: int, iters: dict, seed: int):
    """One fitted rookie head, count or conversion — `stan_rookie.fitted_predictive`'s fit.

    The model object rather than its metrics, because the composition needs `mu`/`p` draws
    to chain through, not a scored predictive on realized trials.
    """
    from src.models.stan_components import StanConversion, StanCount

    if attempted is None:
        return StanCount(features, component, name=label, chains=chains, seed=seed,
                         metric=sr.METRIC, **iters).fit(train)
    return StanConversion(features, component, attempted, name=label, chains=chains,
                          seed=seed, metric=sr.METRIC, **iters).fit(train)


# ── The arms, assembled per family ────────────────────────────────────────────

def head_priors(history: pd.DataFrame, covered: list[str]) -> dict[str, pd.DataFrame]:
    """Each rookie head's expanding draft-bucket prior table, keyed by component.

    `stan_rookie.head_priors` per head, built over the whole train+validation history for
    its reason: the window reads seasons strictly before the target, so a validation
    season's prior is knowable in September and handing it the restricted fitting frame
    would eat one more season on every pass.
    """
    return {component: sr.head_priors(history, component, attempted, covered)
            for _, component, attempted in rr.head_list()}


def rate_arms(cfg: dict, parts: dict, artifacts: dict, draws: int = DRAWS,
              seed: int = SEED) -> tuple[dict[str, np.ndarray], pd.DataFrame, list[dict]]:
    """`({arm: (draws x rows) season dk_pts}, the row keys, fit diagnostics)`.

    Composed per family and stacked in `FAMILY_ORDER`, which is the order `run` stacks the
    metadata in. The keys come back so that agreement is **asserted** rather than trusted:
    a silent misalignment here would pair one player's rebounds with another's games played
    and still produce an entirely plausible season total.
    """
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    constants = rr.rookie_constants(out_dir / "rookie_rate_floors.csv")
    ships = ship_arms(out_dir / "rookie_rate_metrics.csv")

    vet_train, vet_val = parts["veteran"]["train"], parts["veteran"]["val"]
    rk_train, rk_val = parts["rookie"]["train"], parts["rookie"]["val"]
    history = pd.concat([rk_train, rk_val], ignore_index=True)
    priors = head_priors(history, list(parts["covered"]))

    blocks: dict[str, list[np.ndarray]] = {"floor": [], "head": []}
    print("  composing the veteran family "
          f"({len(vet_val):,} rows) — floor, then the persisted {SCORING_WINDOW} heads")
    for name, arm in (("floor", veteran_floor_arm(vet_train, vet_val, draws, seed)),
                      ("head", veteran_head_arm(artifacts, vet_val, draws, seed))):
        blocks[name].append(_arm_total(arm, vet_val, seed))

    print(f"  composing the rookie family ({len(rk_val):,} rows) — §7d's floors, then "
          f"§7e's ship split")
    floor = rookie_floor_arm(rk_train, rk_val, priors, constants, draws, seed)
    counts, conversions, diagnostics = rookie_head_arm(rk_train, rk_val, priors, constants,
                                                       ships, cfg, draws, seed)
    blocks["floor"].append(_arm_total(floor, rk_val, seed))
    blocks["head"].append(_arm_total((counts, conversions), rk_val, seed))

    totals = {name: np.concatenate(parts_, axis=1) for name, parts_ in blocks.items()}
    totals["unserved"] = np.zeros_like(totals["floor"])
    keys = pd.concat([parts[f]["val"][["season", "player_id"]] for f in FAMILY_ORDER],
                     ignore_index=True)
    return totals, keys, diagnostics


def _arm_total(arm: tuple[dict, dict], frame: pd.DataFrame,
               seed: int = SEED) -> np.ndarray:
    """One arm's `(draws x rows)` season dk_pts — the linear chain plus its own bonus."""
    linear, box = compose_linear_dk(arm[0], arm[1], seed)
    bonus = season_bonus(box, frame["gp_played"].to_numpy(dtype=float), seed)
    return linear + bonus[None, :]


# ── The table ─────────────────────────────────────────────────────────────────

def treatments(totals: dict[str, np.ndarray], frame: pd.DataFrame,
               predicted_gp: np.ndarray) -> dict[str, dict]:
    """Every rate arm at the shipped games treatment, and at `oracle_gp` beside it.

    `season_total.evaluate` multiplies a treatment's games by its rate, so an arm's draws
    of the season total are divided by the games they were composed at — the realized ones,
    since the exposure every head consumed is realized minutes — and multiplied back by the
    games the treatment predicts. That is `season_total`'s own `gp x rate` decomposition
    with the rate carrying the distribution instead of the games.
    """
    gp_played = np.clip(frame["gp_played"].to_numpy(dtype=float), 1.0, None)
    out: dict[str, dict] = {}
    for arm, block in totals.items():
        rate = block / gp_played[None, :]
        for prefix, gp in (("", predicted_gp), ("oracle_gp_", gp_played)):
            if prefix and arm == "unserved":
                continue
            out[f"{prefix}{arm}"] = {
                "gp": gp, "pmf": None, "rate": rate.mean(axis=0),
                "samples": rate * gp[None, :]}
    return {name: out[name] for name in ORDER if name in out}


def context_rows(frame: pd.DataFrame, restriction: str, predicted_gp: np.ndarray,
                 plugged: np.ndarray, totals: dict[str, np.ndarray]) -> list[dict]:
    """What the table's own numbers rest on: rows, the bonus share, and the games source."""
    rows = []
    labels = frame[POPULATION_COLUMN].to_numpy()
    for group in ("all",) + GROUPS:
        mask = np.ones(len(frame), bool) if group == "all" else labels == group
        if not mask.any():
            continue
        block = frame[mask]
        total = float(block["dk_total"].sum())
        rows += [
            _context(restriction, group, "n_rows", int(mask.sum())),
            _context(restriction, group, "bonus_share",
                     float(block["dk_bonus_total"].sum() / total) if total else np.nan),
            _context(restriction, group, "mean_dk_total", float(block["dk_total"].mean())),
            _context(restriction, group, "mean_gp_played",
                     float(block["gp_played"].mean())),
            _context(restriction, group, "mean_predicted_gp",
                     float(predicted_gp[mask].mean())),
            _context(restriction, group, "no_design_share", float(plugged[mask].mean())),
            _context(restriction, group, "mean_team_games",
                     float(block["team_games"].mean())),
        ]
        for arm, draws in totals.items():
            rows.append(_context(restriction, group, f"mean_composed_{arm}",
                                 float(draws[:, mask].mean())))
    return rows


def _context(restriction: str, group: str, metric: str, value: float) -> dict:
    """A context row describes the frame rather than an arm, so it carries no `treatment`.

    Left out of the dict rather than written as `""`, which a CSV round trip turns into NaN
    anyway — a reader filtering on the empty string would find nothing and conclude the rows
    were never written.
    """
    return {"measurement": "context", "restriction": restriction, "group": group,
            "metric": metric, "n": 0, "value": float(value)}


def gate_rows(frame: pd.DataFrame, restriction: str, totals: dict[str, np.ndarray],
              predicted_gp: np.ndarray, seed: int = SEED) -> list[dict]:
    """§16's settling gate: the head family against the floor family, paired, per group.

    The **paired** delta rather than the difference of two means, and a bootstrap interval
    on it, which is the device §4 states for every other gate in this program
    (`lag_ladder`, `stan_rookie`) — the two arms score the same rows, so pairing removes
    the between-player variance that would otherwise swamp a family-level effect on 108
    rows. Reported at both games treatments, because a rate-family verdict that only
    survives an oracle on games is a different claim from one that survives the plug-in.
    """
    y = frame["dk_total"].to_numpy(dtype=float)
    gp_played = np.clip(frame["gp_played"].to_numpy(dtype=float), 1.0, None)
    labels = frame[POPULATION_COLUMN].to_numpy()
    rows = []
    for prefix, gp in (("", predicted_gp), ("oracle_gp_", gp_played)):
        scale = (gp / gp_played)[None, :]
        crps = {arm: crps_from_samples(totals[arm] * scale, y) for arm in ("floor", "head")}
        err = {arm: np.abs((totals[arm] * scale).mean(axis=0) - y)
               for arm in ("floor", "head")}
        for group in ("all",) + GROUPS:
            mask = np.ones(len(frame), bool) if group == "all" else labels == group
            if mask.sum() < 2:
                continue
            for metric, per_row in (("crps_dk_total", crps), ("mae_dk_total", err)):
                delta = paired_bootstrap(per_row["head"][mask], per_row["floor"][mask],
                                         n_boot=BOOTSTRAP_REPS, seed=seed)
                rows += [
                    _gate(restriction, f"{prefix}head_vs_floor", group,
                          f"{metric}_delta", delta["crps_delta"], int(mask.sum())),
                    _gate(restriction, f"{prefix}head_vs_floor", group, f"{metric}_lo",
                          delta["ci_lo"], int(mask.sum())),
                    _gate(restriction, f"{prefix}head_vs_floor", group, f"{metric}_hi",
                          delta["ci_hi"], int(mask.sum())),
                    _gate(restriction, f"{prefix}head_vs_floor", group,
                          f"{metric}_wins", float(verdict(delta) == "wins"),
                          int(mask.sum())),
                ]
    return rows


def _gate(restriction: str, treatment: str, group: str, metric: str, value: float,
          n: int) -> dict:
    return {"measurement": "gate", "restriction": restriction, "treatment": treatment,
            "group": group, "metric": metric, "n": int(n), "value": float(value)}


def table_rows(frame: pd.DataFrame, restriction: str, totals: dict[str, np.ndarray],
               predicted_gp: np.ndarray, plugged: np.ndarray, max_games: int,
               seed: int = SEED) -> tuple[list[dict], pd.DataFrame]:
    """One restriction's whole block: the metric table, the gate, and the context."""
    specs = treatments(totals, frame, predicted_gp)
    # No shared rate: every arm here IS a rate family, which is the whole comparison.
    metrics, predictions = evaluate(frame, None, specs, max_games)
    rows = [{"measurement": "metric", "restriction": restriction, **row}
            for row in metrics]
    rows += gate_rows(frame, restriction, totals, predicted_gp, seed)
    rows += context_rows(frame, restriction, predicted_gp, plugged, totals)
    predictions.insert(0, "restriction", restriction)
    # `evaluate` stacks one block of `len(frame)` rows per treatment, in `specs`' order, so
    # the population label tiles rather than repeats.
    predictions[POPULATION_COLUMN] = np.tile(frame[POPULATION_COLUMN].to_numpy(),
                                             len(specs))
    return rows, predictions


# ── Entry point ───────────────────────────────────────────────────────────────

def run(cfg: dict) -> Path:
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("stan", {}).get("seed", SEED))
    draws = int(cfg.get("evaluation", {}).get("composition_draws", DRAWS))

    print("Season-total dk_pts for the structurally-missing population — "
          "docs/rookie-rates-plan.md §5e")
    print("  The test split is LOCKED — every season here goes through "
          "`held_out.selection_split`;\n  this is §16's settling gate and it is read on "
          "VALIDATION.")

    # §16's ladder, if it is on, changes ONE thing in this readout — which rows the
    # availability head has a design row for — and that is precisely the open item §7f
    # left. It writes its own artifact rather than overwriting the shipped one, on
    # `make rookie-floor`'s `_rookiefloor` precedent: the two are the same code on the
    # same worlds under two availability treatments, and a reader has to be able to put
    # them side by side.
    suffix = LADDER_SUFFIX if availability_lag_ladder(cfg) is not None else ""

    parts = build_frame(cfg)
    frame = pd.concat([parts[f]["val"] for f in FAMILY_ORDER], ignore_index=True)[
        ["season", "player_id", POPULATION_COLUMN, "dk_total", "dk_linear_total",
         "dk_bonus_total", "dk_per_game", "gp_played", "team_games", "total_minutes",
         "on_season_start_roster"]]
    counts = frame[POPULATION_COLUMN].value_counts()
    print(f"  {len(frame):,} validation rows "
          f"({', '.join(sorted(frame['season'].unique()))}): "
          + ", ".join(f"{counts.get(g, 0):,} {g}" for g in GROUPS))
    print(f"  ladder rungs admitted by §7c and reported as `lag_recovered`: "
          f"{list(parts['rungs']) or 'none'}")

    # The AVAILABILITY head's design, which is what `no_design_availability` reads its
    # covered set and its allowed seasons from — not the component design this table scores.
    availability = availability_head_design(cfg, design=None)
    artifacts = load_all(posteriors_dir(cfg, SCORING_WINDOW),
                         heads=["availability"]
                         + [artifact_name(h) for h, _, _ in rr.head_list()])
    predicted_gp, plugged = games_played(cfg, frame, artifacts["availability"],
                                         availability)
    print(f"  games played: {int((~plugged).sum()):,} rows from the availability head, "
          f"{int(plugged.sum()):,} from\n  `no_design_availability` at the "
          f"`{no_design_level_arm(cfg)}` level — the split is the design's, not a choice.")

    totals, keys, diagnostics = rate_arms(cfg, parts, artifacts, draws, seed)
    if not keys.reset_index(drop=True).equals(
            frame[["season", "player_id"]].reset_index(drop=True)):
        raise AssertionError(
            "the composed arms are not row-aligned with the scoring frame; one family was "
            "stacked in a different order and the table would pair one player's rebounds "
            "with another's games played")

    max_games = int(frame["team_games"].max())
    rows, predictions = [], []
    for restriction in RESTRICTIONS:
        mask = (np.ones(len(frame), bool) if restriction == "all"
                else frame["on_season_start_roster"].to_numpy(dtype=float) > 0)
        block, preds = table_rows(
            frame[mask].reset_index(drop=True), restriction,
            {a: t[:, mask] for a, t in totals.items()}, predicted_gp[mask], plugged[mask],
            max_games, seed)
        rows += block
        predictions.append(preds)

    table = pd.DataFrame(rows)
    _report(table, frame)
    unconverged = [d for d in diagnostics if not d.get("converged", True)]
    if unconverged:
        print(f"\n/!\\  {len(unconverged)} of {len(diagnostics)} fits did not converge: "
              + ", ".join(sorted(d["label"] for d in unconverged)))

    dest = out_dir / f"season_total_rookie{suffix}.csv"
    table.to_csv(dest, index=False)
    print(f"\nSaved {len(table):,} season-total rookie rows → {dest}")
    pred_dest = out_dir / f"season_total_rookie{suffix}_predictions.csv"
    pd.concat(predictions, ignore_index=True).to_csv(pred_dest, index=False)
    print(f"Saved {sum(len(p) for p in predictions):,} prediction rows → {pred_dest}")
    return dest


#: The artifact suffix the §16 availability-ladder arm writes under. The shipped run and
#: the ladder run differ in the availability treatment of the `lag_recovered` group and in
#: nothing else, so they are two readings and not two versions — `make rookie-floor`'s
#: `_rookiefloor` precedent, for its reason.
LADDER_SUFFIX = "_lagladder"


#: Short labels for the report only — the artifact carries `ORDER` verbatim.
SHORT = {"unserved": "unserved (today)", "floor": "floor family",
         "head": "head family", "oracle_gp_floor": "floor · oracle GP",
         "oracle_gp_head": "head · oracle GP"}


def _report(table: pd.DataFrame, frame: pd.DataFrame) -> None:
    def cell(restriction: str, treatment: str, group: str, metric: str) -> float:
        hit = table[(table["restriction"] == restriction)
                    & (table["treatment"] == treatment) & (table["group"] == group)
                    & (table["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    def context(restriction: str, group: str, metric: str) -> float:
        hit = table[(table["measurement"] == "context")
                    & (table["restriction"] == restriction) & (table["group"] == group)
                    & (table["metric"] == metric)]
        return float(hit["value"].iloc[0]) if len(hit) else float("nan")

    for restriction in RESTRICTIONS:
        print(f"\nSeason-total dk_pts on VALIDATION · {restriction} "
              f"(MAE then CRPS, lower is better)")
        print(f"  {'arm':<19}" + "".join(f"{g:>23}" for g in GROUPS))
        counts = "".join(
            f"{int(context(restriction, g, 'n_rows')):>23,}" for g in GROUPS)
        print(f"  {'n':<19}{counts}")
        for arm in ORDER:
            cells = ""
            for group in GROUPS:
                mae = cell(restriction, arm, group, "mae_dk_total")
                crps = cell(restriction, arm, group, "crps_dk_total")
                cells += f"{mae:>13.1f}{crps:>10.1f}" if np.isfinite(mae) else " " * 23
            print(f"  {SHORT[arm]:<19}{cells}")

    print(f"\n§16's settling gate — the head family against the floor family, paired, "
          f"on {DECISION_RESTRICTION} rows")
    print(f"  {'group':<16}{'games treatment':<18}{'CRPS delta [95%]':>34}"
          f"{'MAE delta':>12}")
    for group in GROUPS:
        for prefix, label in (("", "shipped"), ("oracle_gp_", "oracle GP")):
            arm = f"{prefix}head_vs_floor"
            delta = cell(DECISION_RESTRICTION, arm, group, "crps_dk_total_delta")
            if not np.isfinite(delta):
                continue
            lo = cell(DECISION_RESTRICTION, arm, group, "crps_dk_total_lo")
            hi = cell(DECISION_RESTRICTION, arm, group, "crps_dk_total_hi")
            mark = "*" if hi < 0 else " "
            print(f"  {group:<16}{label:<18}{delta:>+15.4f} [{lo:+.4f}, {hi:+.4f}]{mark}"
                  f"{cell(DECISION_RESTRICTION, arm, group, 'mae_dk_total_delta'):>+12.4f}")
    print("  `*` marks a paired-bootstrap interval entirely below zero. The gate §4 states "
          "is the\n  `rookie` row: the head family against the floor family at the "
          "season-total dk_pts unit.")

    share = frame["dk_bonus_total"].sum() / frame["dk_total"].sum()
    print(f"\n  Context: the double-double bonus is {share:.2%} of the realized season "
          f"total on these\n  rows and every arm carries its own expectation of it "
          f"(`expected_bonus` at the calibrated\n  player-season overdispersion), so the "
          f"target is dk_pts and not its linear part.")


if __name__ == "__main__":
    import argparse

    from src.models.component_rates import LADDER_RUNGS
    from src.models.season_total_rookie import run as _run

    # The §16 arm is a CLI flag rather than a config edit, so both readings are
    # reproducible from a make target and neither can be produced by accident. It
    # overrides `stan.availability.lag_ladder` for this process only and writes
    # `LADDER_SUFFIX`'d artifacts, so the shipped table is never overwritten by it.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lag-ladder", nargs="*", default=None, choices=LADDER_RUNGS,
                        help="availability-ladder rungs to admit for this run "
                             "(docs/availability-window-plan.md §16); omit for the "
                             "shipped plug-in treatment")
    args = parser.parse_args()
    _cfg = yaml.safe_load(open("configs/default.yaml"))
    if args.lag_ladder is not None:
        _cfg.setdefault("stan", {}).setdefault("availability", {})["lag_ladder"] = \
            list(args.lag_ladder)
    _run(_cfg)
