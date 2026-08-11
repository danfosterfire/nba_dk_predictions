"""The live draft room's engine — marginal **bracket EV** under a 30-second clock.

`dashboard/draft_room.py` is the page; everything that computes anything is here, so the
recommender is testable without Streamlit and measurable without a browser.
`docs/simulations-plan.md` ("The live draft room") fixes the contract: load the precomputed
sim tensor and draft pool, take one click per pick, and return a ranked recommendation by
the marginal payout-weighted EV of adding each remaining player to the roster we hold.
**Gate E is the bar: under 1.0 s per recompute on the full remaining pool at
`n_sims = 500`.**

Item 6 left the seam this fills. `draft.recommend` ranks by `marginal_lineup_value` —
dk_pts of weekly-lineup lift — and its `value` argument exists so that decision 5's
objective can be substituted without touching the draft engine. That substitution is this
module.

## Three things stand between a tensor and a payout, and each is a decision

**1. A pick cannot be priced against a payout curve without the rest of the roster.**
Mid-draft we hold three players; the field holds sixteen. Scored as it stands, our entry
sits so far below the field that `P(top 2 of 12)` is zero for every candidate and the
ranking has no resolution at all — a payout is a step function of *place*, and place is a
property of a finished roster. So every candidate is evaluated inside a **completed**
roster: `plan_completion` fills our remaining picks with the best available at each one,
assuming the field consumes the board in rank order. The completion is a plug-in, it is
the same for every candidate, and it is stated here rather than buried because it is the
one modelling assumption in the module that is not read off an artifact.

**2. The field is a population, not a projection.** `q` — the probability a pod-mate
outscores us — is read from a real drafted field: `src/sim/draft.py`'s twelve-seat snake
over the DK-recalibrated board at the `rank_noise_sd` Gate B fitted, scored on the same
tensor and the same sims as our own entry, so a good week for the league is a good week
for everyone. Rounds 2-4 face the *survivor* population, which is that same field
reweighted by its own advance probability — `bracket.py`'s argument that scoring rounds
independently against a fresh field overstates continuation value, arrived at
analytically instead of by dealing.

**The identity that checks all of it is the symmetric-field null.** An entry drawn from
the field must reach round `r` with probability `prod(n_advance / pod_size)` and collect
`total_prizes / total_entries`. `null_check` measures both, and the reach probabilities
come back **exact** — which is what forced the midpoint plotting position: the naive
survival function is a right-endpoint Riemann sum of `∫(1-q)^(P-1) dq` and biases every
round's advance rate upward by about `1 / (2 n_eff)`.

**3. Best-7-by-slot is one exchange, not a re-solve.** The plan's second latency lever,
and it is exact rather than an approximation. `bracket.best_lineup` is matroid greedy, and
for a matroid the max-weight basis of `S + c` is either the old basis `B` or `B + c - x`
for a single `x` — so with `B` computed once per (period, sim), a candidate's lift is

    lift = max(0, score(c) - threshold[mask(c)])

where `threshold[m]` is the cheapest seated player a candidate of eligibility `m` can
displace. That collapses the per-candidate work from a 16-step greedy over a
`[candidate, period, sim, 17]` gather to one subtraction over `[candidate, period, sim]`.
`tests/test_draft_room.py` pins it against `best_lineup` itself, on single-position and
dual-eligible rosters both, because a wrong threshold is a plausible-looking number.

## What the ranking is, and what it is not

The objective is **payout-weighted EV over the full bracket**, in dollars per entry, for
the tournament the room is set to. Two companions ship beside it because they resolve at
different rates: `p_advance` — `P(top 2 of 12)`, which is what `docs/simulations-plan.md`
says the strategy sweep selects on and which is *exact* under this model — and
`lineup_value`, the dk_pts lift, which is the objective item 6 shipped.

**`600k_shootaround`'s EV is a level with a known bias, and the room says so.** Two thirds
of its EV sits in a 49-seat final table reached by 0.139% of entries and topped by a
10,000x prize, so a finite field cannot resolve it: `null_check` reads the EV about 17%
light at the default field size and converges as the field grows. That is the same
observation `docs/simulations-plan.md` already logs — "600k_shootaround's ROI is not
estimable at any affordable simulation budget" — arriving here as a caveat on a level
rather than on a ranking, since the bias is common to every candidate.

Usage:
    python -m src.sim.draft_room                       # build the field, run Gate E
    python -m src.sim.draft_room --season 2022-23
    python -m src.sim.draft_room --rebuild-field       # ignore the cached field artifact
"""

import argparse
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dashboard import economics
from dashboard.economics import ROUND_ONE_POD
from src.sim import draft
from src.sim.bracket import (LINEUP_LIMITS, LINEUP_SIZE, POSITIONS, ROSTER_SIZE,
                             SLOT_CAPACITY, best_lineup, load_tensor, position_masks,
                             roster_is_legal, score_rosters, split_frame, structure,
                             symmetric_null)
from src.sim.bracket import tournaments as bracket_tournaments
from src.sim.season import assert_season_allowed, validation_seasons

SEED = 0

# Gate E, from `docs/simulations-plan.md`: "in-draft recompute under 1.0 s at
# n_sims = 500 on the full remaining pool". A 30-second fast-draft clock has to hold a
# recompute, a human reading the table and a click.
GATE_E_BUDGET = 1.0

# Twelve-seat pods of the fitted field, scored once and cached as an artifact. 100 pods
# reproduce `20k_spin_move`'s symmetric null to under a tenth of a percent; the deep
# rounds of `600k_shootaround` want more than any affordable field can give — see
# `null_check`.
N_FIELD_DRAFTS = 100

# `q` is a weighted fraction over ~1,200 entries, so its own resolution is ~8e-4. The
# clip is three orders of magnitude finer than that and exists only to keep
# `(1 - q) ** 48` inside float64: at 1e-6 it is 1e-288, and at 1e-12 it is not.
Q_FLOOR = 1e-6

# What the table may be ordered by, and the column each one reads. `bracket_ev` is
# decision 5's objective and the default; `p_advance` is what
# `docs/simulations-plan.md` says the strategy sweep selects on, and it is the one figure
# here that `null_check` reproduces exactly; `lineup_value` is item 6's objective, kept so
# the two can be read against each other on the same board.
RANK_OBJECTIVES = {"bracket_ev": "ev",
                   "p_advance": "p_advance",
                   "lineup_value": "lineup_value"}


# ── 1. The exchange threshold — best-7-by-slot as one comparison ─────────────

def lineup_state(scores: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A fixed roster's lineup total, and what a newcomer has to beat to get in.

    Returns `(total, threshold)` where `total` is `best_lineup`'s own answer for this
    roster and `threshold[..., m]` is the score a player of eligibility `m` must exceed to
    enter the lineup, per leading cell. The identity behind it:

        best_lineup(S + c) = best_lineup(S) + max(0, score(c) - threshold[mask(c)])

    which holds because the seatable subsets of a fixed slate are the independent sets of a
    transversal matroid (`bracket.best_lineup`'s own argument), and adding one element to a
    matroid's ground set moves its max-weight basis by at most one exchange.

    **Which player gets displaced is decided by Hall's condition, not by score alone.** A
    newcomer eligible only for `m` pushes every constraint `A ⊇ m` up by one, so the man he
    replaces has to relieve *every* constraint that is already tight — i.e. his own
    eligibility must sit inside the intersection of those tight subsets. Taking the cheapest
    such man is the exchange. Ignoring that and displacing the lineup's lowest scorer
    outright is the plausible wrong answer: it lets a fifth guard evict a centre, which is
    not a lineup.

    The roster must be able to seat seven. `evaluate` guarantees it by construction — the
    completion reserves the slots the roster still owes — and this raises rather than
    returning a threshold computed against a six-man lineup, which would be silently low.
    """
    scores = np.asarray(scores)
    masks = np.broadcast_to(np.asarray(masks, dtype=np.int16), scores.shape)
    total, started = best_lineup(scores, masks)
    lead = scores.shape[:-1]

    if started.sum(axis=-1).min() < LINEUP_SIZE:
        raise ValueError("the base roster cannot seat seven, so no exchange threshold is "
                         "defined for it — complete the roster before pricing candidates")

    # `inside[a]` is "this player's eligibility fits within subset a", which is both the
    # occupancy test for Hall's condition and the exchange test below.
    tight = np.empty(lead + (8,), dtype=bool)
    for a in range(8):
        seated = (started & ((masks & ~np.int16(a)) == 0)).sum(axis=-1)
        tight[..., a] = seated >= LINEUP_LIMITS[a]

    threshold = np.empty(lead + (8,), dtype=scores.dtype)
    for m in range(8):
        keep = np.full(lead, 0b111, dtype=np.int16)
        for a in range(8):
            if m & ~a:                                   # a does not contain m
                continue
            keep = np.where(tight[..., a], keep & np.int16(a), keep)
        exchangeable = started & ((masks & ~keep[..., None]) == 0)
        threshold[..., m] = np.where(
            exchangeable.any(axis=-1),
            np.min(np.where(exchangeable, scores, np.inf), axis=-1),
            np.inf)                                      # nobody to displace: no entry
    return total, threshold


def sum_periods(values: np.ndarray, round_of_period: np.ndarray,
                axis: int) -> np.ndarray:
    """Collapse a scoring-period axis into the four tournament rounds."""
    rounds = np.unique(round_of_period)
    return np.stack([values.take(np.nonzero(round_of_period == r)[0], axis=axis).sum(axis)
                     for r in rounds], axis=axis)


# ── 2. The field, and the survivor population it becomes ─────────────────────

def field_artifact(features_dir: Path, season: str) -> Path:
    return Path(features_dir) / f"draft_room_field_{season}.npz"


def build_field(frame: pd.DataFrame, board: draft.Board, dk_pts: np.ndarray,
                masks: np.ndarray, round_of_period: np.ndarray,
                field_cfg: draft.FieldConfig, n_drafts: int = N_FIELD_DRAFTS,
                seed: int = SEED, pod_size: int = ROUND_ONE_POD,
                seat_strategies: list[str] | None = None) -> np.ndarray:
    """`[entry, round, sim]` round totals for a field of `n_drafts` real twelve-seat pods.

    The rosters come from `src/sim/draft.py` at the field Gate B fitted, so the population
    our entry is priced against is the same one `make draft-sim` calibrated against the
    observed ADP curve. Drafting in pods rather than sampling sixteen players per entry is
    not a detail: `bracket.placeholder_field` records that independent draws give every
    entry the same top ten and a field that is both far too strong and nearly identical.
    """
    state = draft.run_drafts(board, field_cfg, n_drafts, np.random.default_rng(seed),
                             pod_size=pod_size, seat_strategies=seat_strategies)
    rosters = state.roster.reshape(-1, ROSTER_SIZE)
    scored = score_rosters(dk_pts, rosters, masks, round_of_period)
    return np.ascontiguousarray(scored["round_total"], dtype=np.float32)


def _composition_key(seat_strategies: list[str] | None) -> str:
    """One string naming the field's seat mix, for the cache key.

    Order-free and pod-size-free — shares rather than counts — so `None`, the engine's
    all-ADP default, keys identically to an explicit twelve-seat all-ADP list and to the
    caches written before the field carried a composition at all.
    """
    seats = seat_strategies or [draft.AdpAutodraft.name]
    return ",".join(f"{name}:{seats.count(name) / len(seats):.4f}"
                    for name in sorted(set(seats)))


def save_field(dest: Path, field_round: np.ndarray, season: str, fit_window: str,
               field_cfg: draft.FieldConfig, n_drafts: int,
               seat_strategies: list[str] | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest, round_total=field_round, season=season,
                        fit_window=fit_window, n_drafts=n_drafts,
                        noise_model=field_cfg.noise_model,
                        rank_noise_sd=field_cfg.rank_noise_sd,
                        need_weight=field_cfg.need_weight,
                        composition=_composition_key(seat_strategies))
    return dest


def load_field(path: Path, field_cfg: draft.FieldConfig, n_sims: int,
               seat_strategies: list[str] | None = None) -> np.ndarray | None:
    """The cached field, or `None` when it does not describe the field being asked for.

    A field cached under a different noise model is not this field, and reusing it would
    price our entry against a population nothing calibrated. Cheaper to redraft than to
    explain. Caches written before the field carried a composition lack the last two keys
    and are read as the legacy pure-ADP field, which is exactly what they hold.
    """
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        held_need = float(z["need_weight"]) if "need_weight" in z else 0.0
        held_mix = (str(z["composition"]) if "composition" in z
                    else _composition_key(None))
        if (str(z["noise_model"]) != field_cfg.noise_model
                or float(z["rank_noise_sd"]) != field_cfg.rank_noise_sd
                or held_need != field_cfg.need_weight
                or held_mix != _composition_key(seat_strategies)
                or z["round_total"].shape[2] < n_sims):
            return None
        return np.ascontiguousarray(z["round_total"][:, :, :n_sims])


def _row_sorted(totals: np.ndarray,
                weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`[entry, sim]` totals to per-sim ascending order plus cumulative weight."""
    order = np.argsort(totals, axis=0, kind="stable")
    srt = np.ascontiguousarray(np.take_along_axis(totals, order, axis=0).T, dtype=np.float64)
    wsrt = np.take_along_axis(weights, order, axis=0).T
    cum = np.concatenate([np.zeros((srt.shape[0], 1)), np.cumsum(wsrt, axis=1)], axis=1)
    return srt, np.ascontiguousarray(cum)


def survival(srt: np.ndarray, cum: np.ndarray, values: np.ndarray) -> np.ndarray:
    """`P(a field entry outscores this)`, weighted, with ties split down the middle.

    `srt` is `[sim, entry]` ascending and `cum` is `[sim, entry + 1]` cumulative weight;
    `values` is `[k, sim]` and the answer is `[k, sim]`.

    **The half-tie is the midpoint plotting position and it is load-bearing, not
    cosmetic.** Counting only strictly-greater entries makes the survival function a
    right-endpoint Riemann sum, and every downstream advance rate is an integral against
    it: `E[(1-q)^(P-1)]` over the field comes out `1/P + 1/(2 n_eff)` instead of `1/P`. At
    the sample sizes the deep rounds leave — `600k_shootaround`'s round-3 population is a
    few dozen effective entries — that is a 44% overstatement of the advance rate, and
    `null_check` reads it directly. With the half-tie the reach probabilities are exact.

    One `searchsorted` over the whole thing rather than one per sim: each sim's block is
    shifted by a constant wider than the data, so the flattened array stays sorted and the
    per-sim searches cannot cross into each other.
    """
    n_sims, n_entry = srt.shape
    step = float(srt.max() - srt.min()) + 1.0
    offset = np.arange(n_sims, dtype=np.float64) * step * 2.0
    flat = (srt + offset[:, None]).ravel()
    shifted = np.asarray(values, dtype=np.float64) + offset[None, :]

    base = (np.arange(n_sims) * n_entry)[None, :]
    lo = np.clip(np.searchsorted(flat, shifted.ravel(), side="left").reshape(shifted.shape)
                 - base, 0, n_entry)
    hi = np.clip(np.searchsorted(flat, shifted.ravel(), side="right").reshape(shifted.shape)
                 - base, 0, n_entry)

    cum_t = np.ascontiguousarray(cum.T)                        # [entry + 1, sim]
    total = cum[:, -1][None, :]
    below = np.take_along_axis(cum_t, lo, axis=0)
    upto = np.take_along_axis(cum_t, hi, axis=0)
    q = ((total - upto) + 0.5 * (upto - below)) / np.maximum(total, 1e-12)
    return np.clip(q, Q_FLOOR, 1.0 - Q_FLOOR)


def round_outcome(q: np.ndarray, pod_size: int, payout: np.ndarray,
                  n_advance: int) -> tuple[np.ndarray, np.ndarray]:
    """Expected cash and `P(advance)` for one round, given `q` per (candidate, sim).

    Pod-mates are `pod_size - 1` draws from the round's population, so the number who
    outscore us is `Binomial(pod_size - 1, q)` and the place is that count plus one. The
    draw is without replacement in the contest and binomial here, which is the
    approximation `bracket.simulate_bracket` deliberately does *not* make — it deals real
    pods. It is safe in this direction and for this purpose: eleven pod-mates come out of
    a field of 35,280, and the quantity being computed is a ranking of candidates rather
    than the tournament's own accounting.

    Computed by the binomial recurrence rather than by `exp`, because the final round is a
    49-seat table and the exponentials there cost more than everything else in a recompute
    put together.
    """
    n = pod_size - 1
    ratio = q / (1.0 - q)
    term = np.exp(n * np.log1p(-q))                            # k = 0
    cash = np.zeros_like(q)
    advance = np.zeros_like(q)
    for k in range(pod_size):
        if payout[k]:
            cash += term * payout[k]
        if k < n_advance:
            advance += term
        if k + 1 < pod_size:
            term = term * ratio * (n - k) / (k + 1)
    return cash, advance


@dataclass(frozen=True)
class FieldReference:
    """One tournament's field, as the four populations its four rounds are played against.

    `sorted_total[i]` and `cum_weight[i]` describe round `i`'s population: the drafted
    field, weighted by the probability each entry is still alive. Rounds 2-4 therefore
    face survivors rather than a fresh field, which is the systematic overstatement
    `bracket.py` names — reached here by reweighting instead of by dealing, because a
    recompute cannot deal 35,280 entries inside a 30-second clock.
    """

    tournament: str
    rounds: list[dict]
    sorted_total: list[np.ndarray]
    cum_weight: list[np.ndarray]
    entry_fee: float
    effective_entries: np.ndarray          # per round — how much field is left to resolve


def field_reference(field_round: np.ndarray, tournament: str,
                    root: Path = economics.ROOT, rounds: list[dict] | None = None,
                    entry_fee: float | None = None) -> FieldReference:
    """Reweight the drafted field into each round's surviving population.

    `rounds` and `entry_fee` default to the captured tournament's own, through
    `bracket.structure` and `dashboard.economics` — nothing about a contest is written
    down here either. They are overridable so a test can put a four-seat pod and a $10
    prize through the same arithmetic and check it against a number worked out by hand.
    """
    rounds = rounds or structure(tournament, root)
    n_entry, _, n_sims = field_round.shape
    totals = np.asarray(field_round, dtype=np.float64)

    weights = np.ones((n_entry, n_sims))
    sorted_total, cum_weight, effective = [], [], []
    for i, rnd in enumerate(rounds):
        srt, cum = _row_sorted(totals[:, i, :], weights)
        sorted_total.append(srt)
        cum_weight.append(cum)
        # Kish effective sample size: what the round's population is really resolved by.
        effective.append(float((weights.sum(axis=0) ** 2
                                / np.maximum((weights ** 2).sum(axis=0), 1e-12)).mean()))
        if i + 1 < len(rounds):
            q = survival(srt, cum, totals[:, i, :])
            _, advance = round_outcome(q, rnd["pod_size"], rnd["payout"], rnd["n_advance"])
            weights = weights * advance

    if entry_fee is None:
        meta = economics.load_metadata(root).set_index("type")
        entry_fee = float(meta.loc[tournament, "entry_fee_per_team"])
    return FieldReference(tournament=tournament, rounds=rounds, sorted_total=sorted_total,
                          cum_weight=cum_weight, entry_fee=float(entry_fee),
                          effective_entries=np.asarray(effective))


def bracket_ev(round_totals: np.ndarray, ref: FieldReference,
               per_sim: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Expected payout per entry, and `P(top 2 of 12)`, for each candidate roster.

    `round_totals` is `[candidate, round, sim]`. Money is accumulated round by round
    against the population that round is actually played against, and the reach
    probability compounds our own advance rather than the field's — an entry collects a
    round's cash only in the sims where it got there.

    `per_sim` returns both quantities as `[candidate, sim]` instead of averaging over
    sims. The room never needs that — it is ranking candidates on one board state — but
    `src/sim/strategy.py` does, twice: a portfolio's `P(at least one entry advances)` is
    `1 - prod(1 - p)` **inside** a sim and cannot be recovered from per-entry means, and a
    bootstrap over simulated seasons has to resample the sim axis. Averaging is the
    default so no existing caller changes behaviour.
    """
    n_cand, _, n_sims = round_totals.shape
    ev = np.zeros((n_cand, n_sims))
    reach = np.ones((n_cand, n_sims))
    p_advance = np.zeros((n_cand, n_sims))
    for i, rnd in enumerate(ref.rounds):
        q = survival(ref.sorted_total[i], ref.cum_weight[i], round_totals[:, i, :])
        cash, advance = round_outcome(q, rnd["pod_size"], rnd["payout"], rnd["n_advance"])
        ev += reach * cash
        if i == 0:
            p_advance = advance
        reach = reach * advance
    if per_sim:
        return ev, p_advance
    return ev.mean(axis=1), p_advance.mean(axis=1)


def null_check(field_round: np.ndarray, ref: FieldReference) -> dict:
    """Does an entry drawn from the field earn what an exchangeable entry must?

    `bracket.symmetric_null` derives both figures exactly from the captured CSVs:
    `P(advance round 1) = n_advance / pod_size`, and `E[payout] = total_prizes /
    total_entries` because the pool is fully paid out. Running the field's own entries
    through the same arithmetic our candidates go through checks the survivor reweighting,
    the plotting position, the binomial place distribution and the payout tables at once —
    and it is what caught the Riemann bias in the survival function.

    **The advance rate is the row to read.** It comes back exact. The EV does not, for
    `600k_shootaround`, and that is a resolution limit rather than a fault: two thirds of
    its EV is a 49-seat final table reached by 0.139% of entries, so a field of a thousand
    leaves a handful of effective entries to resolve a 10,000x prize. The shortfall shrinks
    monotonically as the field grows and it is common to every candidate, so it moves the
    level and not the ranking.
    """
    ev, p_advance = bracket_ev(np.asarray(field_round, dtype=np.float64), ref)
    exact = symmetric_null(ref.tournament)
    return {"tournament": ref.tournament,
            "n_field_entries": int(field_round.shape[0]),
            "n_sims": int(field_round.shape[2]),
            "entry_fee": ref.entry_fee,
            "ev_simulated": float(ev.mean()),
            "ev_analytic": exact["expected_payout"],
            "ev_error": float(ev.mean()) - exact["expected_payout"],
            "ev_relative_error": float(ev.mean()) / exact["expected_payout"] - 1.0,
            "p_advance_simulated": float(p_advance.mean()),
            "p_advance_analytic": exact["p_advance_round_1"],
            "p_advance_error": float(p_advance.mean()) - exact["p_advance_round_1"],
            "effective_entries": ", ".join(f"{v:,.0f}" for v in ref.effective_entries)}


# ── 3. The room ───────────────────────────────────────────────────────────────

@dataclass
class Room:
    """Everything a recompute needs, loaded once and reused for the whole draft."""

    season: str
    frame: pd.DataFrame
    board: draft.Board
    dk_pts: np.ndarray                     # [player, period, sim], board-aligned
    scorable: np.ndarray                   # the tensor prices him at all
    masks: np.ndarray                      # DK eligibility bitmask per board row
    round_of_period: np.ndarray
    projection: np.ndarray                 # mean season-total dk_pts
    pod_size: int
    field_cfg: draft.FieldConfig
    field_round: np.ndarray
    refs: dict[str, FieldReference]
    fit_window: str
    seats: list[str] | None = None     # opponent per seat; None = all `adp`

    @property
    def n_sims(self) -> int:
        return self.dk_pts.shape[2]

    def new_state(self) -> draft.DraftState:
        return draft.new_state(self.board, self.pod_size, 1)


def load_room(cfg: dict, season: str, n_sims: int | None = None,
              tournaments: list[str] | None = None, n_field_drafts: int | None = None,
              seed: int = SEED, rebuild_field: bool = False) -> Room:
    """Open the artifacts, draft the reference field once, and cache it.

    Nothing here fits anything: the tensor is `make simulate-season`'s, the board is
    `make draft-pool`'s, and the field's noise is the row Gate B selected. The one
    computed artifact is the field's round totals, which cost ~12 s to score and are
    written next to the tensor so the second launch of a draft room is instant.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    cfg_sim = cfg.get("sim", {})
    pod_size = int(cfg_sim.get("pod_size", ROUND_ONE_POD))
    n_sims = int(n_sims or cfg_sim.get("n_sims_draft", 500))
    n_field_drafts = int(n_field_drafts or cfg_sim.get("draft_room", {})
                         .get("field_drafts", N_FIELD_DRAFTS))
    # **Every captured tournament, not only the two being entered.** All five run a
    # 12-entry Round-1 pod, so the draft is identical and only the bracket above it
    # differs — which means a reference is a reweighting of a field that has already been
    # drafted and scored, and costs milliseconds. It is the same argument `make bracket`
    # makes for simulating all five: four structures the money is not going into are four
    # more chances for a structural bug to show up, and a drafter who enters a sixth
    # contest should not have to edit config to price it.
    tournaments = tournaments or bracket_tournaments()

    assert_season_allowed(season, split_frame(cfg))

    tensor = load_tensor(features_dir, season)
    pool = pd.read_parquet(features_dir / "draft_pool.parquet")
    frame = draft.build_board(pool, season)
    board = draft.to_arrays(frame, season)
    dk_pts, scorable = draft.tensor_scores(frame, tensor, n_sims)
    masks = position_masks(frame)
    field_cfg = draft.selected_field(cfg)
    seats = draft.assign_seats(draft.field_composition(cfg), pod_size)

    path = field_artifact(features_dir, season)
    field_round = None if rebuild_field else load_field(path, field_cfg, n_sims, seats)
    if field_round is None:
        field_round = build_field(frame, board, dk_pts, masks,
                                  tensor["tournament_round"], field_cfg,
                                  n_field_drafts, seed, pod_size, seats)
        save_field(path, field_round, season, tensor["fit_window"], field_cfg,
                   n_field_drafts, seats)
    field_round = field_round[:, :, :n_sims]

    return Room(season=season, frame=frame, board=board, dk_pts=dk_pts,
                scorable=scorable, masks=masks,
                round_of_period=tensor["tournament_round"],
                projection=dk_pts.sum(axis=1).mean(axis=1),
                pod_size=pod_size, field_cfg=field_cfg, field_round=field_round,
                refs={t: field_reference(field_round, t) for t in tournaments},
                fit_window=tensor["fit_window"], seats=seats)


# ── 4. Completing the roster, because a payout needs a finished one ──────────

def plan_completion(room: Room, state: draft.DraftState, seat: int, n_picks: int,
                    n_spare: int = 0, draft_index: int = 0) -> np.ndarray:
    """Who we expect to take at our remaining picks — the plug-in rest of the roster.

    Two rules, and both are the cheapest defensible thing rather than the best thing.
    **Availability**: at overall pick `p` the board has lost about `p` players, so a
    future pick chooses among those ranked `p` or later — which is the field consuming the
    board in the order Gate B fitted it to. **The slate is filled first**: while the roster
    still owes one of its 2 G / 2 F / 1 C starting slots, the pick comes from a position
    that owes one. Within both, the best available by season projection.

    Filling the slate first is `bracket.top_roster`'s convention, reused rather than
    reinvented, and the reason it matters here is not legality — it is that **the
    completion is the baseline every candidate is measured against.** Deferring the centre
    to the last forced pick leaves the base roster with a replacement-level centre, and
    then every centre on the board prices as if the alternative were that scrub. It is not:
    the real alternative to taking a centre now is taking a decent one three rounds later.
    Measured on 2022-23's opening pick, deferring put five centres in the top seven and
    dropped Dončić to eighth.

    It is deliberately not the marginal-value ranking this module computes for the live
    pick: the completion is recomputed on every pick anyway, and a fixed point over sixteen
    future picks would cost the latency budget it is inside.

    `n_spare` extra names ride behind. They stand in for a candidate the completion has
    already claimed — a player cannot be his own marginal value — and they must not shift
    the base roster's slot reservation, which is why they are counted separately rather
    than by asking for one more pick.
    """
    order = draft.snake_order(state.pod_size, draft.N_ROUNDS)
    future = np.nonzero(order[state.pick_index + 1:] == seat)[0] + state.pick_index + 1
    if not len(future):
        future = np.array([state.pick_index], dtype=np.int64)
    available = state.available()[draft_index] & room.scorable
    have = state.have[draft_index, seat, :].astype(np.int64).copy()
    eligible = room.board.eligible

    picked: list[int] = []
    for j in range(n_picks + n_spare):
        allowed = available & (room.board.rank >= future[min(j, len(future) - 1)])
        if not allowed.any():
            allowed = available
        owed = np.maximum(np.asarray(SLOT_CAPACITY, dtype=np.int64) - have, 0)
        if j < n_picks and owed.sum():
            needed = allowed & (eligible[:, owed > 0].any(axis=1))
            if needed.any():
                allowed = needed
        if not allowed.any():
            break
        choice = int(np.argmax(np.where(allowed, room.projection, -np.inf)))
        picked.append(choice)
        available[choice] = False
        have += eligible[choice]
    return np.asarray(picked, dtype=np.int64)


def exact_round_totals(room: Room, rosters: np.ndarray) -> np.ndarray:
    """`[roster, round, sim]` through `bracket.best_lineup` itself, no exchange shortcut.

    The fallback for the handful of candidates the completion already names — their
    thresholds describe a base roster they are inside, so the exchange identity has nothing
    to say about them — and the reference the fast path is tested against.
    """
    rosters = np.asarray(rosters, dtype=np.int64)
    block = np.moveaxis(room.dk_pts[rosters], 1, -1)           # [c, period, sim, roster]
    total, _ = best_lineup(block, room.masks[rosters][:, None, None, :])
    return sum_periods(total, room.round_of_period, axis=1)


def candidate_round_totals(room: Room, threshold: np.ndarray, base_round: np.ndarray,
                           candidates: np.ndarray,
                           budget: int = 8_000_000) -> tuple[np.ndarray, np.ndarray]:
    """Every candidate's completed-roster round totals, by one exchange per (period, sim).

    Returns `(round_totals[c, round, sim], lineup_value[c])` — the second is the marginal
    dk_pts of weekly-lineup lift, which is `draft.marginal_lineup_value`'s objective
    computed as a by-product here rather than as a second pass.
    """
    rop = room.round_of_period
    rounds = np.unique(rop)
    n_sims = room.n_sims
    out = np.empty((len(candidates), len(rounds), n_sims), dtype=np.float64)
    value = np.empty(len(candidates), dtype=np.float64)

    chunk = max(1, int(budget // max(1, len(rop) * n_sims)))
    for lo in range(0, len(candidates), chunk):
        hi = min(lo + chunk, len(candidates))
        sub = candidates[lo:hi]
        bar = np.moveaxis(threshold[:, :, room.masks[sub]], 2, 0)   # [c, period, sim]
        lift = np.maximum(room.dk_pts[sub] - bar, 0.0)
        out[lo:hi] = sum_periods(lift, rop, axis=1)
        value[lo:hi] = lift.sum(axis=1).mean(axis=1)
    out += base_round[None, :, :]
    return out, value


# ── 5. The recommendation ─────────────────────────────────────────────────────

def evaluate(room: Room, state: draft.DraftState, seat: int, tournament: str,
             objective: str = "bracket_ev", top: int = 12,
             draft_index: int = 0) -> tuple[pd.DataFrame, dict]:
    """Rank every legal, priceable player by what taking him is worth right now.

    Legality comes from `draft.legal_mask` under `draft.manual_config`, so the room can
    never name a player the engine would refuse and never applies DK's autodraft caps to a
    manual pick — the rules are explicit that those bind autodraft and not a person.

    A player the tensor cannot price is **excluded rather than valued at zero**, exactly as
    `draft.recommend` excludes him: 153 of 539 rostered players have no component-head
    design row, they stay draftable by the field, and a zero is a number this ranking would
    act on.
    """
    if objective not in RANK_OBJECTIVES:
        raise KeyError(f"unknown objective {objective!r}; "
                       f"registered: {sorted(RANK_OBJECTIVES)}")
    column = RANK_OBJECTIVES[objective]
    ref = room.refs.get(tournament)
    if ref is None:
        raise KeyError(f"no field reference for {tournament!r}; "
                       f"loaded: {sorted(room.refs)}")

    allowed = draft.legal_mask(state, seat, draft.manual_config(room.field_cfg))[draft_index]
    allowed = allowed & room.scorable
    candidates = np.nonzero(allowed)[0]
    if not len(candidates):
        raise ValueError("no priceable player is legal for this seat")

    held = state.roster_of(seat, draft_index)
    n_future = ROSTER_SIZE - len(held) - 1
    completion = plan_completion(room, state, seat, n_future, 1, draft_index)
    if len(completion) < n_future:
        raise ValueError(f"only {len(completion)} players remain for {n_future} future "
                         f"picks — the board has run dry")
    base = np.concatenate([held, completion[:n_future]]).astype(np.int64)

    block = np.moveaxis(room.dk_pts[base], 0, -1)               # [period, sim, base]
    base_total, threshold = lineup_state(block, room.masks[base])
    base_round = sum_periods(base_total, room.round_of_period, axis=0).astype(np.float64)
    base_ev, base_advance = bracket_ev(base_round[None], ref)

    inside = np.isin(candidates, base)
    round_totals = np.empty((len(candidates), base_round.shape[0], room.n_sims))
    lineup_value = np.empty(len(candidates))
    fast = np.nonzero(~inside)[0]
    if len(fast):
        round_totals[fast], lineup_value[fast] = candidate_round_totals(
            room, threshold, base_round, candidates[fast])
    slow = np.nonzero(inside)[0]
    if len(slow):
        # Already inside the base roster, so the exchange says nothing about him: rebuild
        # the sixteen with the next planned pick standing in and score it outright.
        #
        # **They all land on the same roster, and that is the answer rather than a bug.**
        # If the completion says we get this player at a later pick anyway, then taking
        # him now yields exactly the sixteen we were going to have, plus the spare the
        # freed pick buys — the same sixteen whichever of them we take. So they tie, and
        # `rank_cushion` is what separates them: among players worth the same, take the
        # one the field is likeliest to take first. That is the table's second sort key.
        spare = int(completion[n_future]) if len(completion) > n_future else int(base[-1])
        rosters = np.stack([
            np.concatenate([[c], [p for p in base if p != c], [spare]])[:ROSTER_SIZE]
            for c in candidates[slow]])
        round_totals[slow] = exact_round_totals(room, rosters)
        lineup_value[slow] = (round_totals[slow].sum(axis=1)
                              - base_round.sum(axis=0)[None, :]).mean(axis=1)

    ev, p_advance = bracket_ev(round_totals, ref)

    out = room.frame.iloc[candidates][["player_id", "player_name", "team", "position",
                                       "adp", "adp_dk_scale", "board_rank"]].copy()
    out["projection"] = room.projection[candidates]
    out["lineup_value"] = lineup_value
    out["ev"] = ev
    out["d_ev"] = ev - float(base_ev[0])
    out["p_advance"] = p_advance
    out["d_p_advance"] = p_advance - float(base_advance[0])
    out["rank_cushion"] = draft.rank_cushion(state, candidates, seat)
    out["in_completion"] = inside

    out = out.sort_values([column, "rank_cushion"], ascending=[False, True]
                          ).reset_index(drop=True)
    # What each alternative costs against the top of the table, in the ranked unit and in
    # dollars — the number a drafter needs when the clock is at eight seconds.
    out["cost_vs_best"] = out[column] - out[column].iloc[0]
    out["ev_vs_best"] = out["ev"] - out["ev"].iloc[0]

    context = {
        "tournament": tournament,
        "objective": objective,
        # The **whole** ranking, not the slice the caller asked for. `pick_log` needs the
        # value of the player actually taken, and a drafter who overrides the
        # recommendation is precisely the case a top-N slice would drop — which is the
        # case the log exists to record.
        "ranking": out[["player_id", "player_name", column, "cost_vs_best"]]
        .rename(columns={column: "value"}).reset_index(drop=True),
        "n_candidates": int(len(candidates)),
        "n_priceable": int(room.scorable.sum()),
        "n_board": int(room.board.n_players),
        "held": held,
        "completion": completion[:n_future],
        "base_ev": float(base_ev[0]),
        "base_p_advance": float(base_advance[0]),
        "entry_fee": ref.entry_fee,
        "lead": float(out[column].iloc[0] - out[column].iloc[1]) if len(out) > 1 else 0.0,
        "positions": {p: int(state.have[draft_index, seat, i])
                      for i, p in enumerate(POSITIONS)},
    }
    return out.head(top), context


def roster_table(room: Room, state: draft.DraftState, seat: int,
                 draft_index: int = 0) -> pd.DataFrame:
    """The sixteen we hold so far, in the order they were taken."""
    held = state.roster_of(seat, draft_index)
    out = room.frame.iloc[held][["player_id", "player_name", "team", "position",
                                 "adp_dk_scale", "board_rank"]].copy()
    out.insert(0, "round", np.arange(1, len(held) + 1))
    out["projection"] = room.projection[held]
    return out.reset_index(drop=True)


def roster_health(room: Room, state: draft.DraftState, seat: int,
                  draft_index: int = 0) -> dict:
    """Position counts, DK's caps, the slate still owed, and whether it can still be filled.

    **The caps and the slate are different things and the page shows both**, because DK's
    8 G / 8 F / 3 C are *maxima* that bind autodraft only, while 2 G / 2 F / 1 C is a
    minimum that binds the lineup: 8 G + 8 F + 0 C satisfies every cap and seats no centre.
    `at_risk` is the one a drafter has to act on — the picks left no longer cover the slots
    still owed — and it fires a round before it becomes forced, which is when there is
    still a choice about how to fix it.
    """
    held = state.roster_of(seat, draft_index)
    counts = [int(state.have[draft_index, seat, i]) for i in range(len(POSITIONS))]
    owed = [max(SLOT_CAPACITY[i] - counts[i], 0) for i in range(len(POSITIONS))]
    picks_left = ROSTER_SIZE - len(held)
    return {
        "picks_made": int(len(held)),
        "picks_left": int(picks_left),
        "counts": dict(zip(POSITIONS, counts)),
        "caps": {p: draft.POSITION_CAPS[p] for p in POSITIONS},
        "owed": dict(zip(POSITIONS, owed)),
        "at_capacity": [p for i, p in enumerate(POSITIONS)
                        if counts[i] >= draft.POSITION_CAPS[p]],
        "at_risk": bool(sum(owed) >= picks_left > 0),
        "seats_seven": bool(len(held) == ROSTER_SIZE
                            and roster_is_legal(room.masks[held][None, :])[0]),
    }


def seat_on_clock(state: draft.DraftState) -> int:
    """Whose pick it is — the snake, so one click can mean "whoever is up took him"."""
    order = draft.snake_order(state.pod_size, draft.N_ROUNDS)
    return int(order[state.pick_index])


def mark_pick(state: draft.DraftState, player: int) -> int:
    """Record one pick for the seat on the clock. This is the room's only mutation."""
    seat = seat_on_clock(state)
    draft.apply_pick(state, seat, np.array([int(player)]))
    return seat


def replay(room: Room, picks: list[int]) -> draft.DraftState:
    """Rebuild the state from the pick log — how the page undoes a misclick."""
    state = room.new_state()
    for player in picks:
        mark_pick(state, player)
    return state


# ── 6. The pick log — the one thing in a live draft that cannot be rebuilt ────

LOG_COLUMNS = ("season", "tournament", "objective", "our_seat", "pick", "round", "seat",
               "ours", "board_index", "player_id", "player_name", "team", "position",
               "board_rank", "adp", "adp_dk_scale", "projection", "recommended",
               "recommended_value", "taken_value", "cost_vs_best", "followed",
               "taken_at", "recompute_ms")


def pick_log(room: Room, picks: list[int], seat: int, tournament: str = "",
             objective: str = "", annotations: list[dict] | None = None) -> pd.DataFrame:
    """The whole snake, one row per pick, with what the room advised beside what was taken.

    **A draft is the one artifact in this project that cannot be regenerated.** Every other
    figure here is a `make` target away; a pod is played once and the order the board came
    off in is gone the moment the tab closes. So the log is written for later analysis
    rather than for display: `board_index` replays the state through `replay`, the board
    columns make it joinable to `draft_pool.parquet` without a name match, and the
    recommendation columns are what turn it into evidence.

    Those last four are the point. `recommended` is what the room put at the top of the
    table at that moment and `taken_value` is what the player actually taken was worth on
    the same ranking, so on **our own** rows `cost_vs_best` is what overriding the model
    cost by the model's own reckoning, in the ranked unit and signed so that zero means the
    recommendation was followed. Twenty real drafts of that is the only honest record of
    whether a human under a 30-second clock helps or hurts, and it is the shape of the
    pick-log capture `docs/simulations-plan.md` already names as this layer's missing
    calibration.

    **Every ranking here is from our seat's point of view, including on an opponent's
    row**, where the same columns answer a different question: what the field took, priced
    against what our board wanted. That is worth keeping — it is the disagreement the whole
    strategy rests on — but it is not us deviating from advice, so `followed` is null on
    those rows rather than `False`. Two further things the columns will not say for
    themselves: values are **levels for a completed roster**, so they move with the roster
    and are not comparable across picks, only within one; and picks made with no
    recommendation on screen carry nulls rather than zeros, for the same reason `evaluate`
    refuses to price an unscorable player.
    """
    order = draft.snake_order(room.pod_size, draft.N_ROUNDS)
    annotations = annotations or []
    rows = []
    for i, player in enumerate(picks):
        row = room.frame.iloc[int(player)]
        note = annotations[i] if i < len(annotations) else {}
        taken_value, best_value = note.get("taken_value"), note.get("recommended_value")
        ours = bool(order[i] == seat)
        rows.append({
            "season": room.season, "tournament": tournament, "objective": objective,
            "our_seat": int(seat) + 1,
            "pick": i + 1, "round": i // room.pod_size + 1,
            "seat": int(order[i]) + 1, "ours": ours,
            "board_index": int(player),
            "player_id": row["player_id"], "player_name": row["player_name"],
            "team": row["team"], "position": row["position"],
            "board_rank": int(row["board_rank"]), "adp": row["adp"],
            "adp_dk_scale": row["adp_dk_scale"],
            "projection": float(room.projection[int(player)]),
            "recommended": note.get("recommended"),
            "recommended_value": best_value,
            "taken_value": taken_value,
            "cost_vs_best": (None if taken_value is None or best_value is None
                             else taken_value - best_value),
            "followed": note.get("followed") if ours else None,
            "taken_at": note.get("taken_at"),
            "recompute_ms": note.get("recompute_ms"),
        })
    return pd.DataFrame(rows, columns=list(LOG_COLUMNS))


def log_path(out_dir: str | Path, season: str, session: str) -> Path:
    return Path(out_dir) / f"draft_log_{season}_{session}.csv"


def save_pick_log(frame: pd.DataFrame, dest: Path) -> Path:
    """Write the log, creating the directory. Called after **every** pick, on purpose.

    A live draft is exactly the setting where a crash, a closed tab or an expired session
    costs something unrecoverable, and 192 rows of CSV is microseconds. Autosaving is
    cheaper than remembering to press a button with eight seconds on the clock.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dest, index=False)
    return dest


# ── 7. Injury notes — the one thing on this page the model cannot see ────────

# The two capture programs, newest word first. `espn` is the source that is alive in the
# offseason, which is when a best-ball draft happens; `nba_report` is sharper but is
# published about an hour before a game, so between June and October it carries no player
# rows at all. Both are shown with their capture date rather than blended, because they
# answer slightly different questions and a merged "status" would hide which one said it.
INJURY_SOURCES = ("espn", "nba_report")

# NBA reports carry placeholder rows for teams that have not filed yet. They are not a
# player's status and must not render as one.
_NOT_A_PLAYER = {"", "nan", "not yet submitted"}

# ESPN fills its own blanks with this rather than leaving them null, so it arrives as a
# body part called "Not Specified" unless it is taken back out.
_UNSPECIFIED = {"Not Specified": "", "not specified": "", "Not specified": ""}

# Worst first, so a scan down the board hits the season-enders before the sore ankles.
INJURY_SEVERITY = {"out": 0, "doubtful": 1, "day-to-day": 2, "gtd": 2, "questionable": 3,
                   "probable": 4, "available": 5}

# The badge has to fit on a click target beside a name, a position and a team.
INJURY_BADGES = {"day-to-day": "DTD", "questionable": "QUES", "doubtful": "DOUBT",
                 "probable": "PROB", "available": "AVAIL", "out": "OUT"}

INJURY_COLUMNS = ("source", "as_of", "age_days", "name_key", "player", "team", "status",
                  "headline", "detail")


def _severity(status: str) -> int:
    return INJURY_SEVERITY.get(str(status or "").strip().lower(), 6)


def _espn_notes(raw_dir: Path, today: date) -> pd.DataFrame:
    """The latest ESPN snapshot, one row per player.

    Read through the log's own `snapshot_date` rather than by globbing the directory, and
    only the newest one — `docs/availability-plan.md`'s point-in-time rule is about
    *training* rows, and this is a live display, but the same discipline applies for a
    different reason: a draft room showing a stale snapshot beside a fresh one would be
    silently telling a drafter that a player recovered.
    """
    from src.data.injuries import load_log

    log = load_log(raw_dir / "injuries")
    if log.empty:
        return pd.DataFrame(columns=list(INJURY_COLUMNS))
    latest = log[log["snapshot_date"] == log["snapshot_date"].max()].copy()

    # `injury_type` is the body part ("Achilles"), `injury_location` the coarse region
    # ("Leg"). Printing both gives "Right Leg Achilles"; the region adds nothing a reader
    # does not already know from the part, so it is dropped.
    parts = latest[["injury_side", "injury_type"]].fillna("").replace(_UNSPECIFIED, "")
    where = (parts["injury_side"] + " " + parts["injury_type"]
             ).str.split().str.join(" ").str.strip()
    detail = latest["injury_detail"].fillna("").replace(_UNSPECIFIED).str.strip()
    where = np.where(detail != "", where + ", " + detail, where)
    back = latest["return_date_forecast"].fillna("").astype(str).str.strip()

    out = pd.DataFrame({
        "source": "espn",
        "as_of": latest["snapshot_date"].astype(str),
        "name_key": latest["player"].map(_name_key),
        "player": latest["player"],
        "team": latest["team"],
        "status": latest["status"].fillna("").astype(str).str.strip(),
        "headline": [w.strip(" ,") + (f" · back ~{b}" if b and b != "nan" else "")
                     for w, b in zip(where, back)],
        "detail": latest["short_comment"].fillna("").astype(str).str.strip(),
    })
    return _finalize_notes(out, today)


def _nba_notes(raw_dir: Path, today: date) -> pd.DataFrame:
    """The latest NBA injury report that names anybody.

    **The report is game-day and dies with the season**, which is why the latest *capture*
    is not the latest *report*: the archive keeps fetching through the summer and gets
    placeholder rows or nothing. So the newest report date carrying a real player is what
    ships, and `age_days` is what says whether that is any use — during a pre-season draft
    it is months old and the page says so rather than presenting it as current.
    """
    from src.data.injury_reports import load_log

    log = load_log(raw_dir / "injury_reports")
    if log.empty:
        return pd.DataFrame(columns=list(INJURY_COLUMNS))
    named = log[~log["player_name"].fillna("").astype(str).str.strip().str.lower()
                .isin(_NOT_A_PLAYER)]
    named = named[~named["status"].fillna("").astype(str).str.strip().str.lower()
                  .isin(_NOT_A_PLAYER)]
    if named.empty:
        return pd.DataFrame(columns=list(INJURY_COLUMNS))

    latest = named[named["report_date"] == named["report_date"].max()].copy()
    # A player can appear once per scheduled game in one report; the last game date is his
    # most recent word.
    latest = latest.sort_values("game_date").drop_duplicates("player_name", keep="last")

    reason = latest["reason"].fillna("").astype(str).str.strip()
    out = pd.DataFrame({
        "source": "nba_report",
        "as_of": latest["report_date"].astype(str),
        "name_key": latest["player_name"].map(_name_key),
        "player": latest["player_name"],
        "team": latest["team"],
        "status": latest["status"].fillna("").astype(str).str.strip(),
        "headline": reason,
        "detail": latest["reason_detail"].fillna("").astype(str).str.strip(),
    })
    return _finalize_notes(out, today)


def _finalize_notes(frame: pd.DataFrame, today: date) -> pd.DataFrame:
    age = pd.to_datetime(frame["as_of"], errors="coerce", utc=True)
    frame["age_days"] = (pd.Timestamp(today, tz="UTC") - age).dt.days
    frame["as_of"] = age.dt.date.astype(str)
    return frame[list(INJURY_COLUMNS)].reset_index(drop=True)


def load_injury_notes(raw_dir: str | Path, today: date | None = None,
                      sources: tuple[str, ...] = INJURY_SOURCES) -> pd.DataFrame:
    """Both capture programs' latest word on who is hurt, as rows ready to render.

    **This is display-only and it is the only thing on the page the model has not seen.**
    Nothing in the pipeline consumes either feed today: `src/data/injuries.py` writes the
    ESPN log and no module reads it, and `src/data/injury_reports.py`'s only consumer is
    `src/eda/report_calibration.py`, which measures `P(play | designation)` as a study
    rather than as a feature. So the availability head knows how much a player missed
    *last* season and cannot know he had surgery in June — see `docs/availability-plan.md`,
    where the preseason snapshot is named as the only genuinely new input and is blocked
    until the daily capture spans an offseason boundary.

    That is exactly why it belongs beside the recommendation and **not inside it**. Folding
    a current-status feed into a ranking would be the leak `point-in-time-discipline`
    forbids — the feed describes today, and on a backtest board today's status is the
    resolved outcome. So it is attached to rows for a human to read, `evaluate` never sees
    it, and a test pins that the columns stay out of both the ranking and the pick log.
    """
    raw_dir = Path(raw_dir)
    today = today or date.today()
    frames = [f for f in ({"espn": _espn_notes, "nba_report": _nba_notes}[s](raw_dir, today)
                          for s in sources) if len(f)]
    if not frames:
        return pd.DataFrame(columns=list(INJURY_COLUMNS))
    notes = pd.concat(frames, ignore_index=True)
    notes["severity"] = notes["status"].map(_severity)
    return notes.sort_values(["severity", "player"]).reset_index(drop=True)


def _name_key(name) -> str:
    """`report_calibration.normalize_name`, imported where it is used.

    Reused rather than rewritten: it already flips the PDFs' `"Claxton, Nic"` into first-last
    order and strips accents and generational suffixes, which is the exact disagreement
    between these two feeds and the board.
    """
    from src.eda.report_calibration import normalize_name

    return normalize_name(name)


def match_injuries(room: Room, notes: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Attach notes to board rows by name, and **refuse** to guess when a name is ambiguous.

    Neither feed carries a player id — the PDFs are text and the ESPN payload is names — so
    this is one of the name joins `docs/model-development-notes.md` allows, and it gets the
    second guard that rule demands. The guard here is **uniqueness on both sides**: a key
    that names two board players, or two rows within one source, is reported as `ambiguous`
    and attached to nobody. Team would be the obvious third field and is deliberately not
    used, because the two feeds print `"Brooklyn Nets"` where the board prints `"BKN"` and
    inventing a thirty-row lookup to disambiguate a case that does not currently occur is
    how a join acquires a silent failure mode.

    Attaching a wrong note is worse here than attaching none: a drafter who passes on a
    healthy star because the room labelled him Out has lost the pick, and no downstream
    number would ever show it.

    Returns the matched rows with `board_index` attached, and a coverage dict. **Unmatched
    is expected and large** — both feeds list two-way players, summer-league invitees and
    players nobody can draft — so it is reported as a count rather than as an alarm.
    """
    empty = pd.DataFrame(columns=list(INJURY_COLUMNS) + ["board_index", "player_name"])
    coverage = {"sources": 0, "matched": 0, "unmatched": 0, "ambiguous": 0,
                "players_flagged": 0}
    if notes is None or notes.empty:
        return empty, coverage

    board = pd.DataFrame({"board_index": np.arange(len(room.frame)),
                          "player_name": room.frame["player_name"].to_numpy()})
    board["name_key"] = board["player_name"].map(_name_key)
    board_dupes = set(board.loc[board["name_key"].duplicated(keep=False), "name_key"])

    note_dupes = set(notes.loc[notes.duplicated(["source", "name_key"], keep=False),
                               "name_key"])
    ambiguous = board_dupes | note_dupes
    coverage["sources"] = int(len(notes))
    coverage["ambiguous"] = int(notes["name_key"].isin(ambiguous).sum())

    usable = notes[~notes["name_key"].isin(ambiguous)]
    matched = usable.merge(board[~board["name_key"].isin(ambiguous)], on="name_key",
                           how="inner")
    coverage["matched"] = int(len(matched))
    coverage["unmatched"] = coverage["sources"] - coverage["matched"] - coverage["ambiguous"]
    coverage["players_flagged"] = int(matched["board_index"].nunique())
    return matched.sort_values(["severity", "player_name"]).reset_index(drop=True), coverage


def injury_badges(matched: pd.DataFrame) -> dict[int, str]:
    """`board_index -> "OUT · ESPN"`, the compact form that fits on a click target.

    The worst status wins when the two feeds disagree, because a draft room that softens a
    disagreement is making a call it has no basis for.
    """
    if matched is None or matched.empty:
        return {}
    best = matched.sort_values("severity").drop_duplicates("board_index")
    return {int(r.board_index): INJURY_BADGES.get(str(r.status).strip().lower(),
                                                  str(r.status).upper()[:5])
            for r in best.itertuples()}


def injury_relevance(room: Room, notes: pd.DataFrame) -> dict:
    """Do these notes describe the season on the board, or a different one?

    **The feeds describe today and the board may be a practice season**, which is the one
    way this feature can mislead rather than merely be stale. On a 2023-24 board, an August
    2026 snapshot is not "out of date" — it is about other people's other season, and a
    drafter who reads it as roster news would be wrong in a way no number here would catch.

    The test is deliberately crude: a season label spans two calendar years, and a capture
    outside both is about another season. That needs no rule about which month a season
    belongs to — the rule `adp-freeze-rule` records getting wrong in the other direction,
    where a `month >= 10` cutoff misassigned six 2020 snapshots because 2020-21 tipped off
    in December.
    """
    from src.data.fetch import _season_start_year

    start = _season_start_year(room.season)
    years = sorted({int(str(d)[:4]) for d in notes["as_of"]}) if len(notes) else []
    off = [y for y in years if y not in (start, start + 1)]
    return {"season": room.season, "capture_years": years, "off_season_years": off,
            "describes_this_season": not off}


def injury_sources(notes: pd.DataFrame) -> pd.DataFrame:
    """One row per feed — what it said, when, and how old that is."""
    if notes is None or notes.empty:
        return pd.DataFrame(columns=["source", "as_of", "age_days", "rows"])
    return (notes.groupby("source", as_index=False)
            .agg(as_of=("as_of", "max"), age_days=("age_days", "min"),
                 rows=("name_key", "size")))


# ── 8. Gate E ─────────────────────────────────────────────────────────────────

def gate_e(room: Room, tournament: str, seat: int = 0, seed: int = SEED,
           objective: str = "bracket_ev") -> tuple[pd.DataFrame, dict]:
    """Draft one entry through a live pod, timing every recompute against the 1.0 s bar.

    The whole point is that this is the page's own code path — `evaluate` on the full
    remaining pool, on a real board, against eleven ADP opponents — so the number it
    reports is the number a drafter waits for. Wall clock, so it measures the machine as
    much as the model.
    """
    times: list[float] = []
    rows: list[dict] = []
    ids = room.frame["player_id"].to_numpy()

    def our_pick(state, our, allowed):
        start = time.perf_counter()
        # The whole board rather than a top slice: `head` is the only thing `top` changes,
        # and the objective comparison below has to see every candidate or it measures
        # agreement inside the EV ranking's own shortlist.
        table, context = evaluate(room, state, our, tournament, objective=objective,
                                  top=room.board.n_players)
        times.append(time.perf_counter() - start)
        choice = int(np.nonzero(ids == table["player_id"].iloc[0])[0][0])
        by_lineup = table.sort_values("lineup_value", ascending=False)
        rows.append({
            "round": state.round_index + 1, "pick": state.pick_index + 1,
            "n_candidates": context["n_candidates"],
            "player_name": table["player_name"].iloc[0],
            "position": table["position"].iloc[0],
            "board_rank": int(table["board_rank"].iloc[0]),
            "adp_dk_scale": float(table["adp_dk_scale"].iloc[0]),
            "ev": float(table["ev"].iloc[0]),
            "d_ev": float(table["d_ev"].iloc[0]),
            "p_advance": float(table["p_advance"].iloc[0]),
            "lead_over_next": context["lead"],
            "runner_up": table["player_name"].iloc[1] if len(table) > 1 else "",
            "lineup_value_pick": by_lineup["player_name"].iloc[0],
            "agrees_with_lineup_value": bool(
                by_lineup["player_name"].iloc[0] == table["player_name"].iloc[0]),
            "seconds": times[-1],
        })
        return np.array([choice])

    state = draft.run_drafts(room.board, room.field_cfg, 1, np.random.default_rng(seed),
                             pod_size=room.pod_size, our_seat=seat, our_pick=our_pick,
                             our_cfg=draft.manual_config(room.field_cfg))
    held = state.roster_of(seat)
    log = pd.DataFrame(rows)
    stats = {
        "season": room.season, "tournament": tournament, "objective": objective,
        "n_sims": room.n_sims, "n_board": int(room.board.n_players),
        "n_priceable": int(room.scorable.sum()),
        "n_field_entries": int(room.field_round.shape[0]),
        "budget_s": GATE_E_BUDGET,
        "mean_s": float(np.mean(times)), "p95_s": float(np.quantile(times, 0.95)),
        "max_s": float(np.max(times)), "first_s": float(times[0]),
        "passes": bool(np.max(times) < GATE_E_BUDGET),
        "agreement_with_lineup_value": float(log["agrees_with_lineup_value"].mean()),
        "positions": "/".join(f"{state.have[0, seat, i]}{p}"
                              for i, p in enumerate(POSITIONS)),
        "roster_projection": float(room.projection[held].sum()),
    }
    return log, stats


def advance_by_adp(room: Room, state: draft.DraftState, n_picks: int) -> None:
    """Walk the board forward by board rank — a stand-in draft for the diagnostics.

    Not a field model and not used by the page: the two measurements below need several
    *different* mid-draft states to evaluate at, and the cheapest honest way to reach one
    is to let the board go in the order the field's own consensus puts it.
    """
    for _ in range(n_picks):
        seat = seat_on_clock(state)
        allowed = draft.legal_mask(state, seat, room.field_cfg)[0]
        mark_pick(state, int(np.argmax(np.where(allowed, -room.board.rank, -np.inf))))


def stability(room: Room, alt_field: np.ndarray, tournament: str, n_states: int = 4,
              stride: int | None = None, seed: int = SEED) -> pd.DataFrame:
    """Would a *second* drafted field of the same size recommend the same player?

    The room's `q` is read from one simulated field, so every figure it prints inherits
    that field's sampling noise. This redraws the field from a different seed, rebuilds the
    reference, and re-ranks the same board states under both — which is the only way to
    tell a real preference from a resampling artifact.

    **It is worth running because the answer differs by objective**, and that is the
    module's least obvious measured result. `p_advance` is a statement about one 12-entry
    cut and reproduces almost exactly. `bracket_ev` puts two thirds of its weight on a
    49-seat final table that 0.139% of entries reach, so the survivor population resolving
    it is a few dozen effective entries however large the field is drafted — and its top
    pick can flip between two fields that agree on everything else.
    """
    alt = field_reference(alt_field, tournament)
    stride = stride or room.pod_size
    state = room.new_state()
    rows = []
    for step in range(n_states):
        for objective in RANK_OBJECTIVES:
            column = RANK_OBJECTIVES[objective]
            a, _ = evaluate(room, state, 0, tournament, objective=objective, top=30)
            held = replace_reference(room, alt)
            try:
                b, _ = evaluate(room, state, 0, tournament, objective=objective, top=30)
            finally:
                room.refs[tournament] = held
            merged = a[["player_id", column]].merge(b[["player_id", column]],
                                                    on="player_id", suffixes=("_a", "_b"))
            rows.append({
                "tournament": tournament, "objective": objective,
                "pick": state.pick_index + 1,
                "top1_agrees": bool(a["player_id"].iloc[0] == b["player_id"].iloc[0]),
                "top3_overlap": len(set(a["player_id"][:3]) & set(b["player_id"][:3])) / 3,
                "top5_overlap": len(set(a["player_id"][:5]) & set(b["player_id"][:5])) / 5,
                "rank_correlation": float(
                    pd.Series(merged[f"{column}_a"]).corr(
                        pd.Series(merged[f"{column}_b"]), method="spearman")),
            })
        if step + 1 < n_states:
            advance_by_adp(room, state, stride)
    return pd.DataFrame(rows)


def replace_reference(room: Room, ref: FieldReference) -> FieldReference:
    """Swap one tournament's field reference in place, returning the one displaced."""
    previous = room.refs[ref.tournament]
    room.refs[ref.tournament] = ref
    return previous


# ── 9. The entry point ────────────────────────────────────────────────────────

def run(cfg: dict, seasons: list[str] | None = None, n_sims: int | None = None,
        seed: int | None = None, rebuild_field: bool = False,
        n_field_drafts: int | None = None) -> dict[str, Path]:
    """Build each season's reference field, check the null, and measure Gate E."""
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(SEED if seed is None else seed)
    # The null is checked on every captured structure, for `make bracket`'s reason; Gate E
    # is measured on the two being entered, since it is a latency figure and the bracket
    # above it barely moves it.
    entered = list(cfg.get("sim", {}).get("tournaments", {}) or {})

    seasons = seasons or validation_seasons(split_frame(cfg))

    print("Draft room — the live recommender, ranked by marginal BRACKET EV")
    print(f"  the objective is decision 5's: payout-weighted EV over all four rounds, "
          f"against a field drafted by `src/sim/draft.py`")
    print(f"  best-7-by-slot is ONE matroid exchange per candidate, not a re-solve — "
          f"exact, and pinned against `bracket.best_lineup`")
    print(f"  Gate E bar: every recompute under {GATE_E_BUDGET:.1f} s on the full "
          f"remaining pool")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")

    gates, nulls, picks, stable = [], [], [], []
    for season in seasons:
        print(f"\n── {season} ──")
        start = time.perf_counter()
        room = load_room(cfg, season, n_sims=n_sims,
                         n_field_drafts=n_field_drafts, seed=seed,
                         rebuild_field=rebuild_field)
        load_s = time.perf_counter() - start
        print(f"  board {room.board.n_players:,} players, {int(room.scorable.sum()):,} "
              f"priceable; {room.n_sims:,} sims at the `{room.fit_window}` window; "
              f"field {room.field_round.shape[0]:,} entries — loaded in {load_s:.1f}s")

        for tournament in sorted(room.refs):
            row = null_check(room.field_round, room.refs[tournament]) | {"season": season}
            nulls.append(row)
            print(f"  null {tournament:<18} P(advance R1) "
                  f"{row['p_advance_simulated']:.6f} against "
                  f"{row['p_advance_analytic']:.6f} "
                  f"({row['p_advance_error']:+.2e})   E[payout] "
                  f"${row['ev_simulated']:.4f} against ${row['ev_analytic']:.4f} "
                  f"({row['ev_relative_error']:+.1%});  effective entries by round "
                  f"{row['effective_entries']}")

        for tournament in entered:
            log, stats = gate_e(room, tournament, seed=seed)
            log.insert(0, "tournament", tournament)
            log.insert(0, "season", season)
            picks.append(log)
            stats["load_s"] = load_s
            gates.append(stats)
            print(f"  Gate E {tournament:<16} {stats['mean_s'] * 1000:6.0f} ms mean  "
                  f"{stats['p95_s'] * 1000:6.0f} ms p95  {stats['max_s'] * 1000:6.0f} ms "
                  f"max against a {GATE_E_BUDGET * 1000:,.0f} ms bar   "
                  f"{'✅ PASS' if stats['passes'] else '🔴 FAIL'}")
            print(f"  {'':<23} roster {stats['positions']}, and the bracket-EV pick "
                  f"agrees with the marginal-lineup-value pick on "
                  f"{stats['agreement_with_lineup_value']:.0%} of the 16 rounds")

        alt = build_field(room.frame, room.board, room.dk_pts, room.masks,
                          room.round_of_period, room.field_cfg,
                          room.field_round.shape[0] // room.pod_size, seed + 1,
                          room.pod_size)[:, :, :room.n_sims]
        for tournament in entered:
            rows = stability(room, alt, tournament, seed=seed)
            rows.insert(0, "season", season)
            stable.append(rows)
            summary = rows.groupby("objective")[["top1_agrees", "top5_overlap",
                                                 "rank_correlation"]].mean()
            for objective, row in summary.iterrows():
                print(f"  field redraw {tournament:<18} {objective:<13} same top pick on "
                      f"{row['top1_agrees']:.0%} of states, top-5 overlap "
                      f"{row['top5_overlap']:.0%}, rank correlation "
                      f"{row['rank_correlation']:.3f}")

    paths = {}
    for name, frame in (("draft_room_gate_e", pd.DataFrame(gates)),
                        ("draft_room_null", pd.DataFrame(nulls)),
                        ("draft_room_stability", pd.concat(stable, ignore_index=True)),
                        ("draft_room_picks", pd.concat(picks, ignore_index=True))):
        dest = out_dir / f"{name}.csv"
        frame.to_csv(dest, index=False)
        print(f"Saved {len(frame):,} {name.replace('_', ' ')} rows → {dest}")
        paths[name] = dest
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--season", action="append", default=None,
                        help="target season; repeatable. Defaults to validation.")
    parser.add_argument("--n-sims", type=int, default=None,
                        help="in-draft sims. Gate E's bar is quoted at 500.")
    parser.add_argument("--field-drafts", type=int, default=None,
                        help="twelve-seat pods in the reference field.")
    parser.add_argument("--rebuild-field", action="store_true",
                        help="redraft the reference field, ignoring the cached artifact.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, n_sims=args.n_sims, seed=args.seed,
        rebuild_field=args.rebuild_field, n_field_drafts=args.field_drafts)
