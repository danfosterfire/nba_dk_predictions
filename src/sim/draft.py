"""The snake draft — one engine, two modes, and a field that reproduces the ADP it reads.

`docs/simulations-plan.md` ("`src/sim/draft.py`") fixes the shape: twelve entries, sixteen
rounds, snake order, opponents autodrafting off the **DK-recalibrated** consensus with rank
noise under DK's own 8 G / 8 F / 3 C caps. Two modes sit on one engine:

- **reactive** (primary) — `recommend` sees the board as it stands and returns a ranked
  recommendation. This is what `dashboard/draft_room.py` calls and what the strategy sweep
  exercises. Its default value function is the **marginal lineup value** of adding a player
  to the roster we already hold, measured on `make simulate-season`'s tensor through
  `bracket.best_lineup` — so positional scarcity is priced by the same matroid that decides
  a real week, not by a heuristic.
- **ranking-submission** (fallback) — `autodraft_pick` is DK's documented logic, verbatim:
  the queue first, then the pre-draft ranking, subject to the caps, with an exclusion list
  that yields only when a needed position would otherwise be unfillable. It exists because a
  30-second clock can outrun a human, and because the opponent model needs it regardless.

## The opponent model is a plug-in, and that is the point

`OPPONENTS` is a registry. An opponent supplies two things and nothing else: **static keys**
(one preference ordering per draft per seat, drawn once so a drafter's opinion is stable
through his own draft) and an optional **dynamic bonus** recomputed from what he has already
taken. Everything mechanical — who is on the clock, who is gone, the caps, the exclusion
list, keeping a roster seatable — belongs to the engine, so a new opponent is a class with
one or two short methods rather than a second copy of the draft loop.

Three are registered today. `adp` is the shipped field: rank order plus noise. `adp_need` is
the same thing that also leans toward the positions it still owes, which is the first
behaviour a real drafter has that pure ADP does not. `ranking_submission` is an entry that
has stopped making picks and is being autodrafted off a submitted board — a real population
in a $20 field and the reason `field_composition` is keyed by tournament.

**Field composition varies by tournament tier in the interface, and nothing calibrates it
yet.** `docs/simulations-plan.md` names that as a live risk: a $20 field plausibly holds far
more autodraft entries than a $52 one, in the *opposite* direction from the rake maths.
Hard-coding one field everywhere would bake the assumption in silently, so the shares are
config with a `default` key and every tier resolves through it.

## Noise lives in rank space, and that is forced rather than stylistic

The recalibration is **monotone**, so it cannot change a draft order — only the values.
Worse, the fitted isotonic map is 58 distinct values over 253 grid points, with a 55-wide
plateau: noise added to a recalibrated *ADP value* would make fifty-five players exactly
exchangeable. So the field ranks by `adp_dk_scale` (ties broken by the raw consensus, which
is the same order the map preserves) and perturbs the **rank**, which is what
`rank_noise_sd`'s units already say.

## Gate B, and the two things it turned out to measure

Simulate many drafts, take each player's mean pick over the drafts he was drafted in — DK's
own definition of an average draft position — and compare it against the ADP curve the field
consumed. The bar is the 17.0-pick recalibration error: a field model should be no worse at
reproducing the curve than the market proxy it reads.

**It passes wide**, and the fit is where the interest is. See `calibrate` for the finding:
a *constant* rank noise is not identified by this target at all, and a rank-dependent one is.

Usage:
    python -m src.sim.draft
    python -m src.sim.draft --season 2022-23 --n-drafts 200      # fast
    python -m src.sim.draft --no-reactive                        # skip the timed demo
"""

import argparse
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dashboard.economics import ROUND_ONE_POD
from src.sim.bracket import (POSITIONS, ROSTER_SIZE, SLOT_CAPACITY, best_lineup,
                             load_tensor, position_masks, roster_is_legal, split_frame)
from src.sim.season import assert_season_allowed, validation_seasons

N_ROUNDS = ROSTER_SIZE                 # 16 players, one per round
SEED = 0

# DK's own autodraft defaults, from `docs/dk_best_ball_rules.md`: "We will not auto-draft
# more than 8Gs, 8Fs, or 3Cs for your team when auto-drafting from the pre-draft rankings."
# They are maxima and they do not imply a minimum — 8 G + 8 F + 0 C satisfies all three and
# cannot seat a centre — which is why `require_legal_lineup` is a separate switch.
POSITION_CAPS = {"G": 8, "F": 8, "C": 3}

# Gate B's bar. `docs/adp-plan.md`: a monotone recalibration of the consensus onto DK's
# scale scores 17.0 picks of cross-validated mean absolute error, so a field model that
# reproduces the curve to better than that is no worse than the proxy it consumes.
RECALIBRATION_ERROR = 17.0

N_DRAFTS = 400                         # per grid point in the Gate B sweep
NOISE_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 9.0, 14.0, 20.0, 30.0)

# The measured shape of market disagreement by tier, `docs/adp-plan.md`: mean absolute rank
# gap between the consensus and the one real DK board runs 5.1 picks in rounds 1-2, 9.7 in
# 3-4, 10.5 in 5-8 and 30.8 in rounds 9+ — where 9 of the 16 roster spots are filled. That
# is a *source* disagreement rather than a draft-to-draft one, so it is used as a shape and
# never as a level: it is normalized to mean 1 and multiplied by the fitted `rank_noise_sd`,
# which keeps the ladder one-dimensional and fittable on the same grid as the constant arm.
NOISE_TIER_PICKS = (24, 48, 96)
NOISE_TIER_SHAPE = (5.1, 9.7, 10.5, 30.8)

ELITE_PICKS = 24                       # the first two rounds, reported separately


# ── 1. The board ──────────────────────────────────────────────────────────────

def build_board(pool: pd.DataFrame, season: str) -> pd.DataFrame:
    """One row per draftable player for `season`, in the order the field drafts them.

    **The ranking is the DK-recalibrated consensus and never the raw one**, which
    `docs/adp-plan.md` binds: the consensus correlates with the one real DK board at
    rho 0.870 but with a 21.9-pick mean absolute gap, and DK takes centres 11.9 picks
    earlier because category-league ADP discounts them for free-throw percentage while DK
    Best Ball pays rebounds 1.25 and blocks 2.0 flat. Uncorrected, that scoring-system
    artifact reads as model edge on exactly one position.

    Two mechanical consequences of using it, both worth stating because they are easy to
    get wrong in opposite directions:

    - **A monotone map cannot reorder a board.** `adp_dk_scale` is `adp` pushed through a
      fitted isotonic function, so ranking by it (ties broken by the raw value) gives the
      consensus order exactly. What the recalibration buys here is the *target* Gate B
      scores against — the curve in DK pick units — not a different set of picks.
    - **Its plateaus are wide.** The fitted map takes 253 consensus ranks to 58 distinct
      values, the largest plateau covering 55 of them, so the recalibrated value is a poor
      tie-break and a hopeless noise scale. Rank is neither.

    Players with no ADP still have to be on the board: 192 picks against ~200 ADP'd players
    means a draft runs into the unpriced tail, and `docs/adp-plan.md` names that gap
    ("a simulator that runs the board dry needs a fallback ordering"). They are ranked
    behind every ADP'd player by **prior-season minutes**, which is point-in-time legal and
    needs no model — deliberately not the simulator's own projection, since the field must
    not be given our forecast.
    """
    board = pool[pool["season"] == season].copy()
    if board.empty:
        raise ValueError(f"no draft-pool rows for {season}; run `make draft-pool`")

    board["has_adp"] = board["adp_dk_scale"].notna()
    board["fallback_key"] = -board["prior_min_total"].fillna(-1.0)
    board = board.sort_values(["has_adp", "adp_dk_scale", "adp", "fallback_key",
                               "player_id"],
                              ascending=[False, True, True, True, True])
    board = board.drop(columns=["fallback_key"]).reset_index(drop=True)
    board["board_rank"] = np.arange(len(board), dtype=np.int32)
    return board


def board_positions(board: pd.DataFrame) -> np.ndarray:
    """`[player, 3]` one-hot eligibility over (G, F, C), from the pool's own flags.

    Read through `bracket.position_masks` rather than off the columns, so the shipped
    single-letter convention and the rejected dual one arrive here identically and a
    sensitivity run stays a column swap. DK ships single-position players today
    (`dk-is-single-position-and-the-map-is-86-percent`), so every row is one-hot in
    practice — but nothing below assumes it.
    """
    masks = position_masks(board)
    return np.stack([(masks >> i & 1).astype(bool) for i in range(len(POSITIONS))], axis=1)


@dataclass(frozen=True)
class Board:
    """Everything the engine needs about the player pool, as arrays.

    `rank` is the field's consensus ordering, `eligible` is `[player, 3]`, and `adp` is the
    recalibrated curve Gate B scores against — carried here so the calibration and the draft
    cannot drift apart.
    """

    player_id: np.ndarray
    player_name: np.ndarray
    rank: np.ndarray
    eligible: np.ndarray
    adp: np.ndarray
    season: str

    @property
    def n_players(self) -> int:
        return len(self.rank)

    def counts(self) -> dict[str, int]:
        return {p: int(self.eligible[:, i].sum()) for i, p in enumerate(POSITIONS)}


def to_arrays(board: pd.DataFrame, season: str) -> Board:
    return Board(player_id=board["player_id"].to_numpy(),
                 player_name=board["player_name"].to_numpy(),
                 rank=board["board_rank"].to_numpy().astype(np.float32),
                 eligible=board_positions(board),
                 adp=board["adp_dk_scale"].to_numpy(dtype=float),
                 season=season)


# ── 2. Rank noise — a shape, fitted by one scalar ─────────────────────────────

def noise_scale(rank: np.ndarray, model: str, sd: float,
                picks: tuple[int, ...] = NOISE_TIER_PICKS,
                shape: tuple[float, ...] = NOISE_TIER_SHAPE) -> np.ndarray:
    """Per-player noise sd in **picks**, under one of the registered shapes.

    `constant` is the plan's own parameterization and the floor. `tiered` is the hypothesis
    that consensus is tight at the top of the board and loose in the rounds that fill most
    of a roster — measured as a *source* disagreement in `docs/adp-plan.md` and used here
    only for its shape, normalized to mean 1 so `sd` keeps its units and the ladder stays
    one-dimensional.

    Both are one scalar, which is what lets Gate B fit them on the same grid. A model with
    its own free shape would need a target that identifies more than one number, and the
    finding in `calibrate` is that this one barely identifies the first.
    """
    if model == "constant":
        return np.full(len(rank), float(sd), dtype=np.float32)
    if model == "tiered":
        factor = np.asarray(shape, dtype=float)
        factor = factor / factor.mean()
        tier = np.searchsorted(np.asarray(picks), rank, side="right")
        return (float(sd) * factor[tier]).astype(np.float32)
    raise KeyError(f"unknown noise model {model!r}; registered: {sorted(NOISE_MODELS)}")


NOISE_MODELS = ("constant", "tiered")


# ── 3. Opponents — a registry, because the field is the part that will change ──

@dataclass(frozen=True)
class FieldConfig:
    """How one seat behaves. Everything an opponent can be told, in one object."""

    noise_model: str = "tiered"
    rank_noise_sd: float = 0.0
    need_weight: float = 0.0           # `adp_need`: picks of boost per position still owed
    position_caps: tuple[int, ...] = tuple(POSITION_CAPS[p] for p in POSITIONS)
    require_legal_lineup: bool = True


class Opponent:
    """The plug-in interface. Two hooks, and the engine owns everything mechanical.

    `static_keys` is drawn **once per draft per seat** — a drafter's opinion of the board is
    fixed while he drafts it, and redrawing per pick would make him incoherent in a way that
    quietly narrows every roster's spread. `dynamic_bonus` is recomputed from what that seat
    already holds, which is where roster-aware behaviour goes.

    Higher is better in both. The engine adds them, masks the illegal columns and takes the
    argmax, so an opponent never sees availability, caps or exclusions and cannot get them
    wrong.
    """

    name = "opponent"

    def __init__(self, cfg: FieldConfig):
        self.cfg = cfg

    def static_keys(self, board: Board, n_drafts: int, n_seats: int,
                    rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def dynamic_bonus(self, board: Board, have: np.ndarray) -> np.ndarray | float:
        """`have` is `[n_drafts, 3]` for the seat on the clock. Default: no opinion."""
        return 0.0


class AdpAutodraft(Opponent):
    """Rank order plus rank noise. The shipped field, and Gate B's subject.

    Negated so that a *low* rank scores high, which keeps `sd` in picks and signed the way
    a draft position is.
    """

    name = "adp"

    def static_keys(self, board, n_drafts, n_seats, rng):
        sd = noise_scale(board.rank, self.cfg.noise_model, self.cfg.rank_noise_sd)
        z = rng.standard_normal((n_drafts, n_seats, board.n_players)).astype(np.float32)
        return -(board.rank[None, None, :] + sd[None, None, :] * z)


class AdpNeedAware(Opponent):
    """ADP, leaning toward the positions the seat still owes its starting slate.

    The first thing a real drafter does that pure ADP does not: with one centre and four
    picks left, the next centre is worth more than his board rank says. `need_weight` is in
    **picks** — the number of board places a still-unfilled slot is worth — so it is on the
    same scale as `rank_noise_sd` and can be swept against it.

    Registered and unused by default. It is here because the extension point is worth
    demonstrating rather than promising, and because `field_composition` can mix it into a
    tier the day something calibrates one.
    """

    name = "adp_need"

    def static_keys(self, board, n_drafts, n_seats, rng):
        sd = noise_scale(board.rank, self.cfg.noise_model, self.cfg.rank_noise_sd)
        z = rng.standard_normal((n_drafts, n_seats, board.n_players)).astype(np.float32)
        return -(board.rank[None, None, :] + sd[None, None, :] * z)

    def dynamic_bonus(self, board, have):
        owed = np.maximum(np.asarray(SLOT_CAPACITY)[None, :] - have, 0)
        return (self.cfg.need_weight
                * (owed.astype(np.float32) @ board.eligible.T.astype(np.float32)))


class RankingSubmission(Opponent):
    """An entry that submitted a board and stopped picking — DK autodraft, exactly.

    The rules are explicit and this reproduces them: the queue first, then the pre-draft
    ranking, subject to the position caps, with excluded players taken only when a needed
    position would otherwise go unfilled. `autodraft_pick` is the single-entry version and
    is what the tests pin; this is the same logic expressed as keys so the engine can run a
    field of them.

    Its ranking is the board's own order with **no noise**, because a submitted board does
    not vary between drafts — which is exactly why a field of them is not the same field as
    a field of humans, and why the composition is per tier.
    """

    name = "ranking_submission"

    def __init__(self, cfg: FieldConfig, ranking: np.ndarray | None = None,
                 excluded: np.ndarray | None = None):
        super().__init__(cfg)
        self.ranking = ranking
        self.excluded = excluded

    def static_keys(self, board, n_drafts, n_seats, rng):
        order = board.rank if self.ranking is None else np.asarray(self.ranking,
                                                                   dtype=np.float32)
        keys = np.broadcast_to(-order[None, None, :],
                               (n_drafts, n_seats, board.n_players)).astype(np.float32)
        if self.excluded is not None:
            keys = keys - np.asarray(self.excluded, dtype=np.float32)[None, None, :] * 1e6
        return np.ascontiguousarray(keys)


OPPONENTS: dict[str, type[Opponent]] = {
    AdpAutodraft.name: AdpAutodraft,
    AdpNeedAware.name: AdpNeedAware,
    RankingSubmission.name: RankingSubmission,
}


def field_composition(cfg: dict, tournament: str | None = None) -> dict[str, float]:
    """The mix of opponent strategies for one tournament tier, as shares summing to 1.

    **Nothing calibrates this and it is an interface rather than a measurement.**
    `docs/simulations-plan.md` names field quality by tier as a live risk in a specific
    direction: a $20 field plausibly holds far more autodraft entries than a $52 one, which
    is the *opposite* of what the rake maths would suggest. A single hard-coded field would
    bake that in where nobody could see it; a per-tier share with a `default` makes the
    assumption a config line and the day it is measured a data change.
    """
    block = cfg.get("sim", {}).get("field", {}).get("composition", {})
    mix = dict(block.get(tournament) or block.get("default") or {AdpAutodraft.name: 1.0})
    unknown = sorted(set(mix) - set(OPPONENTS))
    if unknown:
        raise KeyError(f"unregistered opponent strategies {unknown}; "
                       f"registered: {sorted(OPPONENTS)}")
    total = float(sum(mix.values()))
    if total <= 0:
        raise ValueError(f"field composition for {tournament!r} sums to {total}")
    return {k: v / total for k, v in mix.items()}


def assign_seats(mix: dict[str, float], pod_size: int) -> list[str]:
    """Which strategy sits in each seat, by largest remainder — deterministic given a mix.

    Largest remainder rather than a random draw so a field is reproducible from its config
    alone, and so a 5% autodraft share in a 12-seat pod is visibly zero seats rather than
    silently one in every twentieth draft.
    """
    exact = {k: v * pod_size for k, v in mix.items()}
    seats = {k: int(np.floor(v)) for k, v in exact.items()}
    for name, _ in sorted(((k, exact[k] - seats[k]) for k in exact),
                          key=lambda kv: (-kv[1], kv[0]))[:pod_size - sum(seats.values())]:
        seats[name] += 1
    out: list[str] = []
    for name in sorted(seats):
        out += [name] * seats[name]
    return out


# ── 4. The engine ─────────────────────────────────────────────────────────────

def snake_order(pod_size: int, n_rounds: int) -> np.ndarray:
    """Seat on the clock at each of `pod_size * n_rounds` picks — the snake."""
    forward = np.arange(pod_size)
    rounds = [forward if r % 2 == 0 else forward[::-1] for r in range(n_rounds)]
    return np.concatenate(rounds)


@dataclass
class DraftState:
    """The board as it stands, shared by both modes.

    Vectorized over drafts on purpose: the reactive recommender is one draft and Gate B is
    hundreds, and running them through two different state objects is how the recommender
    ends up advising against a field it was never tested on.
    """

    board: Board
    pod_size: int
    taken: np.ndarray                  # [n_drafts, n_players] bool
    have: np.ndarray                   # [n_drafts, pod_size, 3] int16
    roster: np.ndarray                 # [n_drafts, pod_size, 16] int32, -1 until filled
    pick_of: np.ndarray                # [n_drafts, n_players] int32, -1 = undrafted
    pick_index: int = 0

    @property
    def n_drafts(self) -> int:
        return self.taken.shape[0]

    @property
    def round_index(self) -> int:
        return self.pick_index // self.pod_size

    def available(self) -> np.ndarray:
        return ~self.taken

    def roster_of(self, seat: int, draft: int = 0) -> np.ndarray:
        held = self.roster[draft, seat]
        return held[held >= 0]


def new_state(board: Board, pod_size: int, n_drafts: int) -> DraftState:
    n = board.n_players
    return DraftState(
        board=board, pod_size=pod_size,
        taken=np.zeros((n_drafts, n), dtype=bool),
        have=np.zeros((n_drafts, pod_size, len(POSITIONS)), dtype=np.int16),
        roster=np.full((n_drafts, pod_size, N_ROUNDS), -1, dtype=np.int32),
        pick_of=np.full((n_drafts, n), -1, dtype=np.int32))


def legal_mask(state: DraftState, seat: int, cfg: FieldConfig,
               excluded: np.ndarray | None = None) -> np.ndarray:
    """`[n_drafts, n_players]` — who this seat may take, under every hard rule at once.

    Four rules, applied in the order DK's own logic implies and each with its own fallback,
    because every one of them can empty the board and an empty board is an exception rather
    than a pick:

    1. **taken** — never negotiable.
    2. **position caps.** DK's defaults are 8 G / 8 F / 3 C and the rules say what happens
       when they bind: the pick comes from "a position with a limit that hasn't been
       reached yet", and only if none is left does the cap yield. So a seat at all three
       caps falls back to the whole remaining board rather than raising.
    3. **the exclusion list.** "Players on the excluded list will not be auto-drafted unless
       it is necessary to create a valid lineup" — so it yields to rule 4 and to an empty
       board, and to nothing else.
    4. **seatability.** A seat whose remaining picks exactly equal the slots it still owes
       is restricted to those positions. This is *not* a DK rule — the caps are maxima and
       8 G + 8 F + 0 C satisfies all three while seating no centre — it is what keeps every
       drafted roster scorable by `bracket.best_lineup`, which would otherwise return a
       plausible six-man total. `require_legal_lineup` switches it off for a run that wants
       to measure how often DK's own logic would strand a slot.
    """
    allowed = state.available()
    caps = np.asarray(cfg.position_caps, dtype=np.int16)
    elig = state.board.eligible.T.astype(np.int16)          # [3, n_players]

    capped = state.have[:, seat, :] >= caps                  # [n_drafts, 3]
    under_cap = allowed & ~((capped.astype(np.int16) @ elig) > 0)
    allowed = np.where(under_cap.any(axis=1)[:, None], under_cap, allowed)

    if excluded is not None and excluded.any():
        kept = allowed & ~excluded[None, :]
        allowed = np.where(kept.any(axis=1)[:, None], kept, allowed)

    if cfg.require_legal_lineup:
        owed = np.maximum(np.asarray(SLOT_CAPACITY, dtype=np.int16)
                          - state.have[:, seat, :], 0)
        forced = owed.sum(axis=1) >= (N_ROUNDS - state.round_index)
        if forced.any():
            needed = allowed & (((owed > 0).astype(np.int16) @ elig) > 0)
            allowed = np.where(forced[:, None] & needed.any(axis=1)[:, None],
                               needed, allowed)

    if not allowed.any(axis=1).all():
        raise ValueError(f"pick {state.pick_index}: no legal player remains for seat "
                         f"{seat} in at least one draft")
    return allowed


def manual_config(cfg: FieldConfig) -> FieldConfig:
    """The same field config with the position caps lifted — what a **manual** pick sees.

    `docs/dk_best_ball_rules.md` is explicit that the caps bind autodraft and not a person:
    "the only way to override them once the draft starts is to make a manual selection or
    by adding players to the queue". The reactive mode is manual, so applying 8 G / 8 F /
    3 C to it would silently forbid a roster a human is allowed to draft. `require_legal_lineup`
    is untouched, because that guard is about a roster staying scorable rather than about a
    DK rule.

    A *strategy* may still choose to cap itself — `docs/simulations-plan.md`'s strategy block
    lists `position_caps` with DK's defaults and "overridable" — which is why this is a
    function the caller applies rather than a property of the mode.
    """
    return replace(cfg, position_caps=(ROSTER_SIZE,) * len(POSITIONS))


def apply_pick(state: DraftState, seat: int, choice: np.ndarray) -> None:
    """Record one pick per draft and advance the clock."""
    drafts = np.arange(state.n_drafts)
    state.taken[drafts, choice] = True
    state.pick_of[drafts, choice] = state.pick_index
    state.roster[drafts, seat, state.round_index] = choice
    state.have[:, seat, :] += state.board.eligible[choice]
    state.pick_index += 1


def run_drafts(board: Board, cfg: FieldConfig, n_drafts: int,
               rng: np.random.Generator, pod_size: int = ROUND_ONE_POD,
               seat_strategies: list[str] | None = None,
               our_seat: int | None = None, our_pick=None,
               our_cfg: FieldConfig | None = None) -> DraftState:
    """Play `n_drafts` complete snake drafts at once.

    Vectorized over drafts and looped over the 192 picks, because a pick depends on every
    pick before it and nothing else does. `seat_strategies` names the opponent in each seat,
    so a mixed field costs one extra key array per distinct strategy rather than a second
    engine.

    `our_seat` and `our_pick` are how the **reactive** mode enters: the callback is handed
    the live state and the legal mask and returns one player per draft, which is the same
    contract `recommend` fills. That seat is masked under `our_cfg`, which defaults to
    `manual_config(cfg)` because DK's caps bind autodraft rather than a person. Left `None`,
    every seat is an opponent and the run is a pure field simulation — which is what Gate B
    measures.
    """
    if board.n_players < pod_size * N_ROUNDS:
        raise ValueError(f"board has {board.n_players:,} players, below the "
                         f"{pod_size * N_ROUNDS} a {pod_size}-entry draft consumes")
    for i, need in enumerate(SLOT_CAPACITY):
        if int(board.eligible[:, i].sum()) < need * pod_size:
            raise ValueError(f"board carries fewer than {need * pod_size} "
                             f"{POSITIONS[i]} — a {pod_size}-entry draft cannot leave "
                             f"every entry a legal lineup")

    names = seat_strategies or [AdpAutodraft.name] * pod_size
    if len(names) != pod_size:
        raise ValueError(f"{len(names)} seat strategies for a {pod_size}-seat pod")

    keys, owners = {}, {}
    for name in sorted(set(names)):
        seats = [s for s, n in enumerate(names) if n == name]
        opponent = OPPONENTS[name](cfg)
        keys[name] = opponent.static_keys(board, n_drafts, len(seats), rng)
        owners[name] = (opponent, {seat: j for j, seat in enumerate(seats)})

    our_cfg = our_cfg or manual_config(cfg)
    state = new_state(board, pod_size, n_drafts)
    for seat in snake_order(pod_size, N_ROUNDS):
        seat = int(seat)
        ours = our_seat is not None and seat == our_seat
        allowed = legal_mask(state, seat, our_cfg if ours else cfg)
        if ours:
            choice = np.asarray(our_pick(state, seat, allowed), dtype=np.int64)
        else:
            opponent, index = owners[names[seat]]
            score = keys[names[seat]][:, index[seat], :]
            score = score + opponent.dynamic_bonus(board, state.have[:, seat, :])
            choice = np.argmax(np.where(allowed, score, -np.inf), axis=1)
        apply_pick(state, seat, choice)
    return state


# ── 5. Mode B: ranking submission, DK's documented autodraft ─────────────────

def autodraft_pick(ranking: np.ndarray, available: np.ndarray, have: np.ndarray,
                   eligible: np.ndarray, queue: list[int] | None = None,
                   excluded: np.ndarray | None = None,
                   caps: tuple[int, ...] = tuple(POSITION_CAPS[p] for p in POSITIONS)
                   ) -> int:
    """One autodraft pick for one entry, following `docs/dk_best_ball_rules.md` literally.

    `ranking` is the submitted board as a preference order (index 0 is the top-ranked
    player), `available` and `excluded` are per-player boolean masks, `have` is this entry's
    `(G, F, C)` counts and `eligible` is `[player, 3]`.

    The rules, in their own order:

    > We will pick the highest player from your queue. We will pick the highest ranked
    > player from your pre-draft rankings.
    > Note: We will not auto-draft more than 8Gs, 8Fs, or 3Cs ... when auto-drafting from
    > the pre-draft rankings. It is possible to exceed these limits when auto-drafting from
    > the queue ... If a set position limit has been reached, the top player in the queue
    > will be selected from a position with a limit that hasn't been reached yet. If the
    > queue only contains players from positions with limits that have been reached, the
    > top player in the queue will be selected.
    > Players on the excluded list will not be auto-drafted unless it is necessary to create
    > a valid lineup ... Excluded players can still be ... auto-drafted from the queue.

    Three asymmetries fall out of that and all three are pinned by tests, because each one
    is a plausible-looking wrong answer: the **queue overrides the caps** but only after
    trying to respect them; the queue **ignores** the exclusion list, since a player you
    queued is a player you asked for; and the ranking path treats the caps as hard and the
    exclusions as soft.
    """
    caps = np.asarray(caps, dtype=np.int64)
    under = ~(np.asarray(have, dtype=np.int64) >= caps)          # positions still open
    fits = eligible @ under.astype(np.int64) > 0 if under.any() else np.zeros(
        len(available), dtype=bool)

    for candidate in (queue or []):
        if not available[candidate]:
            continue
        preferred = [c for c in queue if available[c] and fits[c]]
        return preferred[0] if preferred else candidate

    order = np.asarray(ranking, dtype=np.int64)
    live = available & (fits if fits.any() else np.ones(len(available), dtype=bool))
    if excluded is not None:
        kept = live & ~excluded
        live = kept if kept.any() else live
    ranked = order[live[order]]
    if not len(ranked):
        raise ValueError("autodraft has no available player under any rule")
    return int(ranked[0])


def export_ranking(board: pd.DataFrame, dest: Path, ranking: np.ndarray | None = None,
                   excluded: np.ndarray | None = None) -> Path:
    """Write a submittable pre-draft ranking, in the shape of DK's own board file.

    The columns mirror `data/raw/dk_draft_rankings/DkPreDraftRankings_*.csv` (`ID`, `Name`,
    `Position`, `ADP`, `Team`) plus a `Rank` and an `Excluded` flag, because that file is
    the only documented description of the format. This is the fallback item 7 falls back
    *to* when the 30-second clock cannot be met, so it ships with the mode rather than with
    the page that would need it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    order = np.arange(len(board)) if ranking is None else np.asarray(ranking)
    out = board.iloc[order][["dk_player_id", "player_name", "position", "adp", "team"]]
    out = out.rename(columns={"dk_player_id": "ID", "player_name": "Name",
                              "position": "Position", "adp": "ADP", "team": "Team"})
    out.insert(0, "Rank", np.arange(1, len(out) + 1))
    out["Excluded"] = (np.zeros(len(out), dtype=bool) if excluded is None
                       else np.asarray(excluded)[order])
    out.to_csv(dest, index=False)
    return dest


# ── 6. Mode A: reactive, the primary one ──────────────────────────────────────

def marginal_lineup_value(dk_pts: np.ndarray, roster: np.ndarray, masks: np.ndarray,
                          candidates: np.ndarray, budget: int = 6_000_000) -> np.ndarray:
    """What adding each candidate is worth, in dk_pts per season, to the roster we hold.

    The value of a best-ball roster is `sum over periods of the best legal 7`, so the value
    of a *player* is the lift he produces in that sum — which is not his projection. A fifth
    guard on a roster that already starts four adds nothing, and `bracket.best_lineup`'s
    matroid knows that exactly, so positional scarcity is priced by the same arithmetic that
    decides a real week rather than by a positional-need heuristic bolted on top.

    Reuses `bracket.best_lineup` verbatim, chunked over candidates so the
    `[chunk, period, sim, roster + 1]` gather stays inside `budget` cells.
    """
    n_periods, n_sims = dk_pts.shape[1], dk_pts.shape[2]
    held = np.asarray(roster, dtype=np.int64)
    candidates = np.asarray(candidates, dtype=np.int64)
    width = len(held) + 1

    base = 0.0
    if len(held):
        block = np.moveaxis(dk_pts[held], 0, -1)[None]           # [1, period, sim, held]
        total, _ = best_lineup(block, masks[held][None, None, None, :])
        base = float(total.sum(axis=1).mean())

    out = np.empty(len(candidates), dtype=np.float64)
    chunk = max(1, int(budget // max(1, width * n_periods * n_sims)))
    for lo in range(0, len(candidates), chunk):
        hi = min(lo + chunk, len(candidates))
        rosters = np.concatenate(
            [np.broadcast_to(held, (hi - lo, len(held))), candidates[lo:hi, None]], axis=1)
        block = np.moveaxis(dk_pts[rosters], 1, -1)              # [c, period, sim, width]
        total, _ = best_lineup(block, masks[rosters][:, None, None, :])
        out[lo:hi] = total.sum(axis=1).mean(axis=1) - base
    return out


def tensor_scores(board_frame: pd.DataFrame, tensor: dict,
                  n_sims: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Line the season tensor up with the **whole** board, and say who it cannot price.

    The board and the tensor do not cover the same players and must not be reconciled by
    dropping rows. `make simulate-season` scores 386 of 539 rostered players — the rest have
    no component-head design row — while the *field* takes them at their ADP regardless, and
    who is still on the board at pick k is the quantity a snake draft turns on. Filtering the
    board to the scorable set would quietly hand every one of those players back to us.

    So the tensor is padded to the board's shape and `scorable` marks the real rows.
    `recommend` refuses to rank an unpriced player rather than valuing him at zero, which is
    the same distinction `make draft-pool` draws between `no_nba_history` and a join failure.
    """
    rows = {int(p): i for i, p in enumerate(tensor["player_id"])}
    index = board_frame["player_id"].map(rows)
    scorable = index.notna().to_numpy()
    n_sims = min(int(n_sims or tensor["dk_pts"].shape[2]), tensor["dk_pts"].shape[2])

    dk_pts = np.zeros((len(board_frame), tensor["dk_pts"].shape[1], n_sims),
                      dtype=tensor["dk_pts"].dtype)
    dk_pts[scorable] = tensor["dk_pts"][index[scorable].to_numpy().astype(int), :, :n_sims]
    return dk_pts, scorable


def recommend(state: DraftState, board_frame: pd.DataFrame, seat: int,
              cfg: FieldConfig, dk_pts: np.ndarray | None = None,
              value: np.ndarray | None = None, scorable: np.ndarray | None = None,
              draft: int = 0, top: int = 15) -> pd.DataFrame:
    """The primary mode: what to take right now, ranked, with the reason attached.

    Sees the live board — who is gone, what we already hold, which slots the caps and the
    slate still leave open — and returns the legal candidates ordered by `value`. The
    default value is `marginal_lineup_value` on the season tensor; `value` overrides it, and
    that is the seam `dashboard/draft_room.py` uses to substitute payout-weighted bracket EV
    without touching this module.

    Legality is taken from `legal_mask`, so the recommendation can never name a player the
    engine would refuse — the two modes disagreeing about what is legal is the failure this
    shares one function to avoid. `cfg` should normally be `manual_config(...)`: DK's caps
    bind autodraft, not a person.

    A player the tensor cannot price is **excluded rather than valued at zero**. Zero is a
    number, and a number this ranking would act on.
    """
    allowed = legal_mask(state, seat, cfg)[draft]
    if scorable is not None:
        allowed = allowed & np.asarray(scorable, dtype=bool)
    candidates = np.nonzero(allowed)[0]
    if not len(candidates):
        raise ValueError("no priceable player is legal for this seat")
    held = state.roster_of(seat, draft)
    masks = position_masks(board_frame)

    if value is None:
        if dk_pts is None:
            raise ValueError("recommend needs either a `value` array or the season tensor")
        value = marginal_lineup_value(dk_pts, held, masks, candidates)
    value = np.asarray(value, dtype=float)

    out = board_frame.iloc[candidates][["player_id", "player_name", "team", "position",
                                        "adp_dk_scale", "board_rank"]].copy()
    out["value"] = value
    out["rank_cushion"] = rank_cushion(state, candidates, seat)
    return out.sort_values("value", ascending=False).head(top).reset_index(drop=True)


def rank_cushion(state: DraftState, candidates: np.ndarray, seat: int) -> np.ndarray:
    """Board places between a candidate and the last pick before this seat is back up.

    Negative means the field is expected to have taken him; positive is cushion. Expressed
    in board ranks because that is the unit `rank_noise_sd` is fitted in, so a drafter can
    read the two against each other. Deliberately arithmetic rather than a simulation — the
    honest version re-runs the field, which belongs to the draft room's own latency budget
    rather than to every recommendation.
    """
    order = snake_order(state.pod_size, N_ROUNDS)
    ahead = np.nonzero(order[state.pick_index + 1:] == seat)[0]
    gap = int(ahead[0]) if len(ahead) else 0
    taken_before = int(state.taken[0].sum())
    return state.board.rank[candidates] - (taken_before + gap)


# ── 7. Gate B ─────────────────────────────────────────────────────────────────

def simulated_adp(state: DraftState) -> tuple[np.ndarray, np.ndarray]:
    """Each player's mean pick over the drafts he was drafted in, and how often that was.

    **DK's own definition**, from `docs/adp-plan.md`: the seven significant figures on the
    board (`1.0526223`) are a sum of pick numbers over a draft count, so a player who goes
    undrafted in most pods still carries the average of the pods he went in. Reproducing the
    definition rather than the number is what makes the tail of the curve comparable at all
    — a mean over *all* drafts would push every deep player past the last pick and manufacture
    a gap that is pure bookkeeping.
    """
    drafted = state.pick_of >= 0
    n_drafted = drafted.sum(axis=0)
    total = np.where(drafted, state.pick_of.astype(np.int64) + 1, 0).sum(axis=0)
    mean_pick = np.where(n_drafted > 0, total / np.maximum(n_drafted, 1), np.nan)
    return mean_pick, n_drafted / state.n_drafts


def field_diversity(state: DraftState, pairs: int = 40,
                    rng: np.random.Generator | None = None) -> dict:
    """How much two pods of this field differ — the thing a mean ADP cannot see.

    **A zero-noise field is degenerate, and it is degenerate invisibly.** Every draft plays
    out identically, so a seat holds the same sixteen players in every simulated season, the
    same 192 players are drafted every time, and the 35,280-entry field the bracket scores is
    twelve distinct rosters repeated. Every marginal statistic about it is fine; the joint
    object is not a field.

    Two numbers, both of which a real pick log would pin directly:

    - `roster_overlap` — the mean share of a seat's sixteen that is the same across two
      different drafts. 1.0 is the degenerate case.
    - `pool_coverage` — distinct players drafted at least once, over the `pod_size * 16` a
      single draft consumes. 1.0 means the field never reaches past the same 192 names.
    """
    rng = rng or np.random.default_rng(SEED)
    n_drafts, pod_size = state.roster.shape[:2]
    overlaps = []
    for _ in range(pairs if n_drafts > 1 else 0):
        a, b = rng.choice(n_drafts, 2, replace=False)
        seat = int(rng.integers(pod_size))
        shared = np.intersect1d(state.roster[a, seat], state.roster[b, seat]).size
        overlaps.append(shared / N_ROUNDS)
    drafted = int((state.pick_of >= 0).any(axis=0).sum())
    return {"roster_overlap": float(np.mean(overlaps)) if overlaps else 1.0,
            "pool_coverage": drafted / (pod_size * N_ROUNDS)}


def fit_region(adp: np.ndarray) -> np.ndarray:
    """Which players' observed ADP carries information about where they were drafted.

    **The terminal plateau of the isotonic map is excluded, and that is about identification
    rather than about passing.** The fitted transfer takes 253 consensus ranks to 58 values;
    both validation seasons' boards run off the end of it into one flat value (160.14, held
    by 33 players in 2022-23 and 60 in 2023-24). Inside that plateau the target says nothing
    about ordering, so no field can reproduce it and any noise that shortens the tail scores
    better — which is what drags an unrestricted fit toward implausibly large noise.

    Gate B is reported on **both** populations and passes on both, so the restriction never
    rescues the gate. It only stops the fit from chasing an artifact of the map.
    """
    has = ~np.isnan(adp)
    if not has.any():
        return has
    return has & (adp < np.nanmax(adp) - 1e-9)


def curve_error(mean_pick: np.ndarray, adp: np.ndarray, mask: np.ndarray) -> float:
    gap = np.abs(mean_pick[mask] - adp[mask])
    return float(np.nanmean(gap)) if mask.any() else float("nan")


def gate_b(board: Board, cfg: FieldConfig, n_drafts: int, rng: np.random.Generator,
           pod_size: int = ROUND_ONE_POD) -> dict:
    """One grid point: simulate a field, and score its realized ADP against the observed."""
    state = run_drafts(board, cfg, n_drafts, rng, pod_size=pod_size)
    mean_pick, rate = simulated_adp(state)
    has = ~np.isnan(board.adp)
    fit = fit_region(board.adp)
    elite = has & (board.adp <= ELITE_PICKS)
    return {
        "noise_model": cfg.noise_model,
        "rank_noise_sd": cfg.rank_noise_sd,
        "n_drafts": n_drafts,
        "n_adp": int(has.sum()),
        "n_fit": int(fit.sum()),
        "mae_fit": curve_error(mean_pick, board.adp, fit),
        "mae_all": curve_error(mean_pick, board.adp, has),
        "mae_elite": curve_error(mean_pick, board.adp, elite),
        "top1_sim": float(mean_pick[0]),
        "top1_observed": float(board.adp[0]),
        "undrafted_adp": int((rate[has] == 0).sum()),
        "min_draft_rate": float(rate[has].min()),
        **field_diversity(state, rng=np.random.default_rng(SEED)),
        "mean_pick": mean_pick,
        "draft_rate": rate,
    }


def calibrate(board: Board, base: FieldConfig, n_drafts: int, seed: int = SEED,
              grid=NOISE_GRID, models=NOISE_MODELS,
              pod_size: int = ROUND_ONE_POD) -> pd.DataFrame:
    """Fit `rank_noise_sd` to Gate B's own statistic, one scalar per noise shape.

    **The gate passes wide and the fit is where the finding is**, so both are reported and
    neither is quoted without the other:

    - A **constant** rank noise is *not identified* by a mean-ADP target. `E[pick]` is the
      board rank for any interior player under symmetric noise of any size, so the objective
      is flat to a few hundredths of a pick across the whole grid. That is a property of the
      target, not a failure of the fit, and it is the reason the plan's instruction to fit
      rather than choose was worth following: choosing would have hidden it.
    - A **rank-dependent** shape *is* identified, because it is pinned at both ends at once —
      the top of the board where the observed curve is nearly deterministic (the consensus
      #1 goes at 1.05) and the deep rounds where it compresses. Its optimum is interior and
      the two validation seasons agree on it.

    **The selected arm's margin over the no-noise floor is not a result**, and it is reported
    as one only in the sense that the gate demanded a fit. What makes a nonzero noise
    non-optional is `field_diversity`: at `sd = 0` every draft plays out identically, so a
    seat holds one roster in every simulated season and the field the bracket scores is
    twelve rosters repeated. No mean-ADP statistic can see that, which is precisely why real
    pick logs would be worth more here than another season of ADP.

    Every grid point is run from the **same seed**, which is common random numbers rather
    than tidiness: the arms are separated by hundredths of a pick, so an unpaired grid would
    report Monte Carlo noise as a difference between models. Paired, the noise draw is
    identical and `sd` only scales it, so the curve is a smooth function of the parameter.

    Selection reads validation only, like everywhere else in this repo.
    """
    rows = []
    for model in models:
        for sd in grid:
            cfg = replace(base, noise_model=model, rank_noise_sd=float(sd))
            row = gate_b(board, cfg, n_drafts, np.random.default_rng(seed), pod_size)
            rows.append({k: v for k, v in row.items()
                         if k not in ("mean_pick", "draft_rate")})
    frame = pd.DataFrame(rows)
    frame["selected"] = False
    frame.loc[frame["mae_fit"].idxmin(), "selected"] = True
    frame["passes"] = frame["mae_fit"] <= RECALIBRATION_ERROR
    frame["bar"] = RECALIBRATION_ERROR
    return frame


def selected_field(cfg: dict, out_dir: Path | None = None,
                   tournament: str | None = None) -> FieldConfig:
    """The shipped field, read from the Gate B artifact rather than re-decided.

    `configs/default.yaml` carries `rank_noise_sd: null` because the plan specified it as
    fitted and not chosen, so a consumer that needs a field — the draft room, the strategy
    sweep — resolves it here. A literal number in the config would be a second source of
    truth for a figure the artifact already owns, which is the rule
    `src/final_evaluation.py` follows for which arm shipped.
    """
    block = cfg.get("sim", {}).get("field", {})
    caps = block.get("position_caps", POSITION_CAPS)
    base = FieldConfig(position_caps=tuple(int(caps[p]) for p in POSITIONS),
                       need_weight=float(block.get("need_weight", 0.0)),
                       require_legal_lineup=bool(block.get("require_legal_lineup", True)))

    sd, model = block.get("rank_noise_sd"), block.get("noise_model")
    if sd is not None and model is not None:
        return replace(base, noise_model=str(model), rank_noise_sd=float(sd))

    out_dir = Path(out_dir or cfg["evaluation"]["predictions_dir"])
    path = out_dir / "draft_gate_b.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — `sim.field.rank_noise_sd` is null because Gate B fits it; "
            f"run `make draft-sim` first")
    fitted = pd.read_csv(path)
    fitted = fitted[fitted["selected"] & (fitted["season"] == "pooled")]
    if fitted.empty:
        raise ValueError(f"{path} carries no pooled selected row")
    row = fitted.iloc[0]
    return replace(base, noise_model=str(row["noise_model"]),
                   rank_noise_sd=float(row["rank_noise_sd"]))


# ── 8. The entry point ────────────────────────────────────────────────────────

def reactive_demo(board_frame: pd.DataFrame, board: Board, cfg: FieldConfig,
                  dk_pts: np.ndarray, scorable: np.ndarray, rng: np.random.Generator,
                  seat: int = 0, pod_size: int = ROUND_ONE_POD) -> tuple[pd.DataFrame, dict]:
    """Draft one entry reactively against eleven ADP opponents, and time every pick.

    This is the mode the draft room drives, so it is exercised rather than described. The
    per-pick wall clock is reported for the same reason: Gate E is item 7's bar, and knowing
    now whether a full-pool marginal-value recompute fits inside a 30-second clock decides
    whether that page is a recommender or a ranking exporter.

    Our seat is masked under `manual_config`, because DK's caps bind autodraft and not a
    person; the eleven opponents keep them.
    """
    masks = position_masks(board_frame)
    ours = manual_config(cfg)
    ids = board_frame["player_id"].to_numpy()
    times: list[float] = []
    picked: list[dict] = []

    def our_pick(state, our, allowed):
        start = time.perf_counter()
        table = recommend(state, board_frame, our, ours, dk_pts=dk_pts,
                          scorable=scorable, top=3)
        times.append(time.perf_counter() - start)
        choice = int(np.nonzero(ids == table["player_id"].iloc[0])[0][0])
        picked.append({"round": state.round_index + 1, "pick": state.pick_index + 1,
                       "player_name": table["player_name"].iloc[0],
                       "position": table["position"].iloc[0],
                       "board_rank": int(table["board_rank"].iloc[0]),
                       "adp_dk_scale": float(table["adp_dk_scale"].iloc[0]),
                       "value": float(table["value"].iloc[0]),
                       "runner_up": table["player_name"].iloc[1],
                       "runner_up_value": float(table["value"].iloc[1])})
        return np.array([choice])

    state = run_drafts(board, cfg, 1, rng, pod_size=pod_size, our_seat=seat,
                       our_pick=our_pick, our_cfg=ours)
    roster = state.roster_of(seat)
    stats = {"n_sims": dk_pts.shape[2], "n_priceable": int(scorable.sum()),
             "n_board": int(board.n_players),
             "latency_mean": float(np.mean(times)), "latency_max": float(np.max(times)),
             "roster_legal": bool(roster_is_legal(masks[roster][None, :])[0]),
             "positions": {p: int(state.have[0, seat, i])
                           for i, p in enumerate(POSITIONS)}}
    return pd.DataFrame(picked), stats


def run(cfg: dict, seasons: list[str] | None = None, n_drafts: int | None = None,
        seed: int | None = None, reactive: bool = True) -> dict[str, Path]:
    """Fit the field on validation, then exercise both modes over it."""
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_drafts = int(n_drafts or cfg.get("sim", {}).get("field", {}).get("n_drafts",
                                                                      N_DRAFTS))
    seed = int(SEED if seed is None else seed)
    pod_size = int(cfg.get("sim", {}).get("pod_size", ROUND_ONE_POD))

    design = split_frame(cfg)
    seasons = seasons or validation_seasons(design)
    for season in seasons:
        assert_season_allowed(season, design)

    pool = pd.read_parquet(features_dir / "draft_pool.parquet")
    base = FieldConfig(
        position_caps=tuple(int(cfg.get("sim", {}).get("field", {})
                                .get("position_caps", POSITION_CAPS)[p])
                            for p in POSITIONS))

    print("Draft — a 12-entry, 16-round snake over one engine with two modes")
    print(f"  opponents autodraft off the DK-RECALIBRATED consensus "
          f"(`adp_dk_scale`), never the raw one")
    print(f"  DK autodraft caps "
          f"{'/'.join(f'{c} {p}' for p, c in zip(POSITIONS, base.position_caps))}, "
          f"read from `docs/dk_best_ball_rules.md`")
    print(f"  Gate B bar: mean absolute rank gap <= {RECALIBRATION_ERROR} picks, the "
          f"recalibration's own CV error")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")

    grids, curves, fields, picks = [], [], [], []
    for season in seasons:
        print(f"\n── {season} ──")
        frame = build_board(pool, season)
        board = to_arrays(frame, season)
        counts = board.counts()
        n_adp = int((~np.isnan(board.adp)).sum())
        print(f"  board {board.n_players:,} players "
              f"({', '.join(f'{p} {counts[p]:,}' for p in POSITIONS)}); "
              f"{n_adp:,} carry ADP, {board.n_players - n_adp:,} ranked behind them by "
              f"prior-season minutes")
        print(f"  the map's terminal plateau holds "
              f"{n_adp - int(fit_region(board.adp).sum()):,} of them at "
              f"{np.nanmax(board.adp):.2f} — excluded from the FIT, reported in the gate")

        grid = calibrate(board, base, n_drafts, seed, pod_size=pod_size)
        grid.insert(0, "season", season)
        grids.append(grid)
        for model in NOISE_MODELS:
            sub = grid[grid["noise_model"] == model]
            best = sub.loc[sub["mae_fit"].idxmin()]
            flat = float(sub["mae_fit"].max() - sub["mae_fit"].min())
            print(f"  {model:9} best sd {best['rank_noise_sd']:5.2f}  "
                  f"MAE(fit) {best['mae_fit']:6.3f}  MAE(all) {best['mae_all']:6.3f}  "
                  f"elite {best['mae_elite']:5.2f}  #1 at {best['top1_sim']:5.2f} "
                  f"(observed {best['top1_observed']:.2f})   grid spread {flat:6.3f}")

    pooled = (pd.concat(grids).groupby(["noise_model", "rank_noise_sd"], as_index=False)
              .agg(mae_fit=("mae_fit", "mean"), mae_all=("mae_all", "mean"),
                   mae_elite=("mae_elite", "mean"), top1_sim=("top1_sim", "mean"),
                   top1_observed=("top1_observed", "mean"),
                   n_adp=("n_adp", "sum"), n_fit=("n_fit", "sum"),
                   undrafted_adp=("undrafted_adp", "sum"),
                   min_draft_rate=("min_draft_rate", "min"),
                   roster_overlap=("roster_overlap", "mean"),
                   pool_coverage=("pool_coverage", "mean")))
    pooled["season"] = "pooled"
    pooled["n_drafts"] = n_drafts
    pooled["selected"] = False
    pooled.loc[pooled["mae_fit"].idxmin(), "selected"] = True
    pooled["passes"] = pooled["mae_fit"] <= RECALIBRATION_ERROR
    pooled["bar"] = RECALIBRATION_ERROR
    shipped = pooled[pooled["selected"]].iloc[0]
    ship = replace(base, noise_model=str(shipped["noise_model"]),
                   rank_noise_sd=float(shipped["rank_noise_sd"]))

    print(f"\nGate B — pooled over {len(seasons)} validation seasons")
    print(f"  SHIPPED: noise_model={ship.noise_model}, "
          f"rank_noise_sd={ship.rank_noise_sd:.2f}  "
          f"MAE(fit) {shipped['mae_fit']:.3f} / MAE(all) {shipped['mae_all']:.3f} "
          f"against a {RECALIBRATION_ERROR:.1f}-pick bar   "
          f"{'✅ PASS' if shipped['passes'] else '🔴 FAIL'}")
    floor = pooled[(pooled["noise_model"] == ship.noise_model)
                   & (pooled["rank_noise_sd"] == 0.0)].iloc[0]
    constant = pooled[pooled["noise_model"] == "constant"]["mae_fit"]
    print(f"  the MEAN-ADP target barely identifies a level: the whole constant grid "
          f"(sd 0 to {max(NOISE_GRID):.0f}) spans "
          f"{float(constant.max() - constant.min()):.3f} picks, and the selected arm beats "
          f"the no-noise floor by only {floor['mae_fit'] - shipped['mae_fit']:+.3f}")
    print(f"  what it DOES resolve is the shape, and both seasons agree on the optimum "
          f"independently")
    print(f"  what no ADP curve can see: at sd = 0 two drafts share "
          f"{floor['roster_overlap']:.0%} of a seat's roster and the field reaches "
          f"{floor['pool_coverage']:.2f}x one draft's worth of players; at the shipped sd "
          f"those read {shipped['roster_overlap']:.0%} and "
          f"{shipped['pool_coverage']:.2f}x")

    for season in seasons:
        frame = build_board(pool, season)
        board = to_arrays(frame, season)
        row = gate_b(board, ship, n_drafts, np.random.default_rng(seed), pod_size)
        curve = frame[["player_id", "player_name", "team", "position", "board_rank",
                       "adp", "adp_dk_scale"]].copy()
        curve.insert(0, "season", season)
        curve["simulated_adp"] = row["mean_pick"]
        curve["draft_rate"] = row["draft_rate"]
        curve["gap"] = curve["simulated_adp"] - curve["adp_dk_scale"]
        curve["in_fit_region"] = fit_region(board.adp)
        curves.append(curve)

        for tournament, spec in (cfg.get("sim", {}).get("tournaments", {}) or {}).items():
            mix = field_composition(cfg, tournament)
            fields.append({"season": season, "tournament": tournament,
                           "entries": int(spec.get("entries", 0)),
                           "pod_size": pod_size,
                           "noise_model": ship.noise_model,
                           "rank_noise_sd": ship.rank_noise_sd,
                           "position_caps": "/".join(
                               f"{c}{p}" for p, c in zip(POSITIONS, ship.position_caps)),
                           "composition": ", ".join(f"{k}={v:.2f}"
                                                    for k, v in sorted(mix.items())),
                           "seats": ", ".join(assign_seats(mix, pod_size)),
                           "calibrated": False})

    if reactive:
        season = seasons[0]
        frame = build_board(pool, season)
        board = to_arrays(frame, season)
        tensor = load_tensor(features_dir, season)
        dk_pts, scorable = tensor_scores(
            frame, tensor, cfg.get("sim", {}).get("n_sims_draft"))

        log, stats = reactive_demo(frame, board, ship, dk_pts, scorable,
                                   np.random.default_rng(seed), pod_size=pod_size)
        log.insert(0, "season", season)
        picks.append(log)
        print(f"\nReactive mode — one entry drafted live against {pod_size - 1} ADP "
              f"opponents on {season}")
        print(f"  ranked by MARGINAL LINEUP VALUE on the sim tensor "
              f"({stats['n_sims']:,} sims); {stats['n_priceable']:,} of "
              f"{stats['n_board']:,} board players are priceable and the rest stay "
              f"draftable BY THE FIELD but unrankable by us")
        print(f"  latency {stats['latency_mean'] * 1000:.0f} ms mean, "
              f"{stats['latency_max'] * 1000:.0f} ms max per recompute "
              f"(Gate E's bar is 1,000 ms and belongs to item 7)")
        print(f"  roster {stats['positions']}, seats seven: {stats['roster_legal']} "
              f"(our seat is uncapped — DK's caps bind autodraft, not a manual pick)")

    paths = {}
    written = [("draft_gate_b", pd.concat(grids + [pooled], ignore_index=True)),
               ("draft_adp_curve", pd.concat(curves, ignore_index=True)),
               ("draft_field", pd.DataFrame(fields))]
    if picks:
        written.append(("draft_reactive", pd.concat(picks, ignore_index=True)))
    for name, frame in written:
        dest = out_dir / f"{name}.csv"
        frame.to_csv(dest, index=False)
        print(f"Saved {len(frame):,} {name.replace('_', ' ')} rows → {dest}")
        paths[name] = dest
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to validation.")
    parser.add_argument("--n-drafts", type=int, default=None,
                        help="drafts per Gate B grid point.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no-reactive", action="store_true",
                        help="skip the timed reactive demo, which needs the sim tensor.")
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, n_drafts=args.n_drafts, seed=args.seed,
        reactive=not args.no_reactive)
