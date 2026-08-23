"""The bracket — weekly lineups, the four-round chain, ties, wildcards and payouts.

`docs/simulations-plan.md` ("`src/sim/bracket.py`", "Contest mechanics") fixes what this
module owes: it turns `src/sim/season.py`'s `player x scoring_period x sim` tensor into an
entry's money. Everything structural — round count, pod size, advance count, cash table —
is read from `dashboard.economics`, which derives it from the two captured DraftKings CSVs.
**Nothing about a tournament is written down here**, so pointing this at the live 2026-27
contests is a data change rather than a code change.

Three things had to be exactly right, and each one is wrong in a way that is silent.

## 1. The weekly lineup is an assignment problem, not a greedy fill

Seven starters out of sixteen: 2 G, 2 F, 1 C, 2 UTIL. A dual-eligible player assigned to
the first slot he fits can block a better player out of the lineup — put a G/F into a
guard slot and the second-best forward drops to the bench even though a UTIL seat was
free. That understates every roster a little and some rosters a lot, and it never raises.

The fix is not a solver. **The value of a lineup depends only on *which* seven players are
picked, never on where they sit**, so the problem is "the highest-scoring 7-subset that can
be seated at all" — and the seatable subsets of a fixed slate of slots are exactly the
independent sets of a *transversal matroid*. Greedy is optimal on a matroid, so sorting the
sixteen by score and taking each player whose addition keeps the set seatable gives the
true maximum, at sixteen vectorized steps and no solver dependency.

Seatability is Hall's condition, and with three position types it is eight inequalities.
For every subset `A` of {G, F, C}, the players eligible **only** for positions in `A` must
not outnumber the seats those positions carry plus the two UTIL seats:

    #{p in S : eligible(p) subset of A}  <=  capacity(A) + 2

`A = {G,F,C}` is the roster-size constraint (5 + 2 = 7) and `A = {}` refuses a player with
no position at all, so the eight cover everything and no separate bookkeeping is needed.

## 2. The tie-break cascades, and the 2-vs-3 boundary is where the money is

Round 1 cuts 12 entries to 2 over 17 weeks, so the boundary is crossed constantly and the
rules are explicit about it: highest single week, then second-highest, down through the
round's weeks; then the highest-scoring individual player, cascading the same way. Both
cascades are implemented (`cascade_order`).

**One judgement call, made here and stated rather than buried.** "The overall highest
scoring player in that round" is read as *the player's contribution to the entry's round
total* — his dk_pts in the weeks he actually started. That makes the player cascade a
partition of the round total in exactly the way the weekly cascade is, which is the reading
that keeps the two levels coherent. Counting bench weeks is the alternative and would
change nothing that has ever decided a pod here.

Exact float ties surviving both cascades mean two entries scored identically every week and
player-for-player, which in practice means *the same roster*. Those break at random, which
is what a coin flip is.

## 3. Rounds 2-4 face the survivor population, and it is dealt rather than modelled

An entry in Round 2 is playing eleven entries that already finished top-2 of twelve. Score
the rounds independently against a fresh ADP field and continuation value comes out
systematically too high.

**So the whole contest is played out, at its real size.** `total_entries` from the captured
metadata is the field — 35,280 for `600k_shootaround` down to 216 for `88k_alley_oop` — and
each round shuffles the survivors, deals them into real pods, ranks each pod by the cascade
above, and carries the top `n_advance` forward. An entry's place is its place. There is no
distribution over pod-mates to get right, and our own entries are simply *in* the field at
known rows, which is what DK does with them.

**This replaced a parametric model, and the model was a workaround for a mistake made
earlier in the same file.** The first version sized the field by a knob (3,000) and cut it
by pods; `600k_shootaround` advances 1 in 720 across three cuts, so Round 4 was decided
against **four** surviving entries standing in for a 49-entry final table, and the null
below read ROI **+0.72** against an exact -0.1497. Rather than fix the field size, that
version carried the survivor population as a per-entry weight and drew place parametrically
from it — which cost two further bugs (a binomial pod-mate count where a pod is dealt
without replacement, worth +0.08 of ROI on `20k_spin_move`; and an off-by-one on whether an
entry joins the field or occupies one of its slots, which left last place nearly
unreachable). Sizing the field correctly dissolves the original problem: 35,280 -> 5,880 ->
490 -> 49 and 432 -> 72 -> 24 -> 8 are all real populations. The parametric layer is gone.

Two consequences are worth stating because they turn estimates into identities. **Every
round's survivor count is the published field size exactly**, and **the payouts sum to the
prize pool exactly** — both were Monte Carlo estimates under the model and are now
arithmetic. The tie-break also moved to where the rules put it: within the contest being
decided, rather than over a global ordering of the whole field.

## The check that ties it together: the symmetric-field null

A field of exchangeable entries has an exactly known answer, and it exercises the pod
sizes, the advance counts, the wildcard fill and every cash band at once:

    P(advance round 1) = n_advance / pod_size          = 1/6 in every one of the five
    E[payout] / fee - 1 = -rake                        because the prize pool is fully paid

The second identity holds only if the chain of field sizes and the cash tables reconcile to
the published prize pool, so a wrong pod size, a dropped band or a mis-scored round breaks
it. `symmetric_null` derives it analytically for all five tournaments and `make bracket`
reproduces it by simulation.

**It earned its keep twice on the day it was written.** It caught a tie-break that handed
every tie to our own entries — they carried a real per-player split and the field a column
of zeros, which lexicographically is not a *missing* level but a *winning* one — worth +68%
on P(reach round 4) and the difference between a -11% ROI and a +71% one. And it caught a
**transcription error in the captured prize CSV**: `15k_and_one` appeared to pay 24 of its
42 finalists for a total of $13,200 against a stated $15,000 pool, while the other four
reconciled to the cent. The source was corrected on 2026-08-09 (42 paid places, $9,771) and
all five now reconcile exactly.

That matters beyond one tournament, because **pod sizes for rounds 2+ are inferred** rather
than published: `economics.advance_table` takes each round's pod to be the largest place it
pays or advances, since the prize CSV records payouts and not contest sizes. The inference
has three independent corroborations, all at 5 of 5 — the field-size chain stays integral,
each final round's field equals its paid places, and the payouts reconcile to the stated
pool — and 12 is the only round-1 pod size for which any of them hold. Before the
correction a brute force over every pod-size assignment consistent with the CSV found
**none** that reconciled `15k_and_one`, which is what identified the rows rather than the
pods as the problem.

**All five tournaments are simulated**, not only the reference pair. They cost one
scoring pass between them — the field is drafted once per season at the largest tournament's
size and each contest takes the prefix it needs, which is valid because entries are
exchangeable and every field size is a multiple of the 12-entry round-1 pod. Four structures
the money is not going into are four more chances for a structural bug to show up.

## What is a placeholder here

The field's *rosters* are built by `placeholder_field`, Gumbel-noised twelve-entry snake
drafts over the board, because `src/sim/draft.py` (build item 6) does not exist yet. What
item 6 replaces is the *ranking* it reads — DK's recalibrated ADP consensus rather than the
simulator's own projection — plus DK's autodraft caps and the `rank_noise_sd` Gate B fits.
The draft mechanism is the real one and stays. Everything above it — lineups, ties,
advancement, payouts — is final.

Usage:
    python -m src.sim.bracket
    python -m src.sim.bracket --season 2022-23 --n-sims 60      # fast; n_sims is the knob
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from dashboard import economics
from dashboard.economics import ROUND_ONE_POD
from src.models.component_rates import build_design as component_build_design
from src.sim.season import (RUNG_ZERO, VETERAN_FAMILY, assert_season_allowed,
                            validation_seasons)

# ── The slate, and the eight inequalities that decide a lineup ────────────────

# Bit 0 = G, bit 1 = F, bit 2 = C. A player's eligibility is a subset of these.
POSITIONS = ("G", "F", "C")
SLOT_CAPACITY = (2, 2, 1)          # 2 G, 2 F, 1 C — docs/dk_best_ball_rules.md
UTIL_SLOTS = 2                     # open to G, F or C
LINEUP_SIZE = sum(SLOT_CAPACITY) + UTIL_SLOTS
ROSTER_SIZE = 16
BENCH_SIZE = ROSTER_SIZE - LINEUP_SIZE

N_SIMS_BRACKET = 250               # simulated seasons; the field array scales with it
N_NULL = 250                       # exchangeable entries scored to reproduce the null
FIELD_TAU = 20.0                   # Gumbel ranking-draft temperature, in board ranks
SEED = 0

# `roi == -rake` is an identity, not a fit, so the bar is floating point rather than a
# judgement. Four of the five captured tournaments clear it at 1e-12; `15k_and_one`'s bands
# pay $13,200 of a stated $15,000 and it is reported rather than silently absorbed.
NULL_TOLERANCE = 1e-9


def lineup_limits() -> np.ndarray:
    """`limit[A]` — how many players eligible only within `A` a lineup can seat.

    Hall's condition for seating a set of players into 2 G / 2 F / 1 C / 2 UTIL, indexed by
    the subset `A` of {G, F, C} as a bitmask. `limit[0b111] = 7` is the roster-size
    constraint and `limit[0] = 0` refuses a player with no position, since UTIL is
    documented as "G/F/C" rather than as an unrestricted seat.
    """
    limits = np.zeros(8, dtype=np.int16)
    for subset in range(1, 8):
        seats = sum(SLOT_CAPACITY[i] for i in range(len(POSITIONS)) if subset >> i & 1)
        limits[subset] = seats + UTIL_SLOTS
    return limits


def _contribution() -> np.ndarray:
    """`contrib[m, A] = 1` when a player with eligibility `m` counts against limit `A`."""
    m = np.arange(8)[:, None]
    a = np.arange(8)[None, :]
    return ((m & ~a) == 0).astype(np.int16)


LINEUP_LIMITS = lineup_limits()
_CONTRIB = _contribution()


def position_masks(frame: pd.DataFrame) -> np.ndarray:
    """`pos_g` / `pos_f` / `pos_c` from the draft pool, packed into one bitmask per row.

    `make draft-pool` ships a single letter per player (DK is single-position — see
    `dk-is-single-position-and-the-map-is-86-percent`) and carries the rejected dual
    convention alongside as `dual_g` / `dual_f` / `dual_c`. Both shapes arrive here as the
    same three booleans, so a sensitivity run is a column swap and never a code change.
    """
    columns = [f"pos_{p.lower()}" for p in POSITIONS]
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise KeyError(f"draft pool is missing eligibility columns {missing}")
    bits = np.zeros(len(frame), dtype=np.int16)
    for i, column in enumerate(columns):
        bits |= frame[column].to_numpy(bool).astype(np.int16) << i
    return bits


# ── 1. Lineup selection ───────────────────────────────────────────────────────

def best_lineup(scores: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The highest-scoring legal 7 of 16, vectorized over every leading axis.

    `scores` is `[..., roster]` and `masks` broadcasts to it. Returns the lineup total and
    a boolean `[..., roster]` marking who started.

    Greedy in descending score, accepting a player whenever the resulting set still
    satisfies all eight of `LINEUP_LIMITS`. That is the matroid greedy on a transversal
    matroid, so it is the exact maximum and not a heuristic — which is the whole point,
    because a first-fit assignment of a dual-eligible player is wrong silently.

    Negative dk_pts are real (a scoreless game with turnovers), and they change nothing: a
    lineup must seat seven whatever they score, and matroid greedy returns the maximum
    weight *basis* regardless of sign.
    """
    scores = np.asarray(scores)
    masks = np.broadcast_to(np.asarray(masks, dtype=np.intp), scores.shape)
    lead, n_roster = scores.shape[:-1], scores.shape[-1]

    order = np.argsort(-scores, axis=-1, kind="stable")
    seated = np.zeros(lead + (8,), dtype=np.int16)
    started = np.zeros(scores.shape, dtype=bool)
    total = np.zeros(lead, dtype=scores.dtype if scores.dtype.kind == "f" else float)

    for j in range(n_roster):
        pick = order[..., j:j + 1]
        mask = np.take_along_axis(masks, pick, axis=-1)
        value = np.take_along_axis(scores, pick, axis=-1)
        add = _CONTRIB[mask[..., 0]]
        ok = ((seated + add) <= LINEUP_LIMITS).all(axis=-1)
        seated += add * ok[..., None]
        total += np.where(ok, value[..., 0], 0)
        np.put_along_axis(started, pick, ok[..., None], axis=-1)
        if seated[..., 7].min() == LINEUP_SIZE:
            break
    return total, started


def roster_is_legal(masks: np.ndarray) -> np.ndarray:
    """Whether a roster `[..., roster]` of eligibility masks can seat seven at all.

    A 16-man roster with no center cannot fill the C slot; DK enforces this at draft time
    and the field builder has to as well, or `best_lineup` quietly returns a six-man total.
    """
    _, started = best_lineup(np.zeros(masks.shape), masks)
    return started.sum(axis=-1) == LINEUP_SIZE


def score_rosters(dk_pts: np.ndarray, rosters: np.ndarray, masks: np.ndarray,
                  round_of_period: np.ndarray, track_players: bool = False,
                  budget: int = 8_000_000) -> dict:
    """Every entry's per-period lineup total, and optionally its per-player round split.

    `dk_pts` is the season simulator's `[player, period, sim]` tensor, `rosters` is
    `[entry, 16]` of indices into its player axis, and `masks` is one eligibility bitmask
    per player. Chunked over entries so the `[chunk, 16, period, sim]` gather stays inside
    `budget` cells whatever the field size.

    `player_round` is the tie-break's third level and costs `entries x rounds x 16 x sims`,
    so it is opt-in: the field does not carry it, since a tie that survives seventeen weekly
    comparisons means an identical roster and breaks at random either way.
    """
    n_entries, n_roster = rosters.shape
    _, n_periods, n_sims = dk_pts.shape
    rounds = np.unique(round_of_period)

    period = np.empty((n_entries, n_periods, n_sims), dtype=np.float32)
    player_round = (np.zeros((n_entries, len(rounds), n_roster, n_sims), dtype=np.float32)
                    if track_players else None)

    chunk = max(1, int(budget // max(1, n_roster * n_periods * n_sims)))
    for lo in range(0, n_entries, chunk):
        hi = min(lo + chunk, n_entries)
        block = np.moveaxis(dk_pts[rosters[lo:hi]], 1, -1)      # [c, period, sim, roster]
        total, started = best_lineup(block, masks[rosters[lo:hi]][:, None, None, :])
        period[lo:hi] = total.astype(np.float32)
        if track_players:
            contribution = np.where(started, block, np.float32(0.0))
            for i, rnd in enumerate(rounds):
                weeks = contribution[:, round_of_period == rnd]  # [c, week, sim, roster]
                player_round[lo:hi, i] = np.moveaxis(weeks.sum(axis=1), 1, 2)

    return {"period": period,
            "round_total": round_totals(period, round_of_period),
            "player_round": player_round,
            "round_of_period": np.asarray(round_of_period),
            "rounds": rounds}


def round_weeks(scored: dict, rnd: int) -> np.ndarray:
    """`[entry, week, sim]` — one round's weekly lineup totals, the cascade's second level."""
    return scored["period"][:, scored["round_of_period"] == rnd]


def round_totals(period: np.ndarray, round_of_period: np.ndarray) -> np.ndarray:
    """`[entry, period, sim]` summed into `[entry, round, sim]`."""
    rounds = np.unique(round_of_period)
    return np.stack([period[:, round_of_period == r].sum(axis=1) for r in rounds], axis=1)


# ── 2. The cascading tie-break ────────────────────────────────────────────────

def cascade_order(totals: np.ndarray, weekly: np.ndarray, players: np.ndarray,
                  jitter: np.ndarray) -> np.ndarray:
    """Rank one pod, best first, by the rules' full cascade.

    Round total, then the round's weekly scores sorted descending and compared
    lexicographically, then the same over each entry's per-player contributions, then a
    random draw for entries that are identical all the way down.
    """
    keys = []
    for i in range(len(totals)):
        keys.append((-float(totals[i]),
                     tuple(-np.sort(np.asarray(weekly[i], dtype=float))[::-1]),
                     tuple(-np.sort(np.asarray(players[i], dtype=float))[::-1]),
                     float(jitter[i])))
    return np.array(sorted(range(len(totals)), key=keys.__getitem__), dtype=np.intp)


def rank_within_pod(totals: np.ndarray, tiebreak=None,
                    rng: np.random.Generator | None = None) -> np.ndarray:
    """Placement along the last axis — 0 is best — with the cascade where it is needed.

    The fast path is one `argsort`, because two float sums over a seventeen-week round are
    not equal by accident. Rows that *do* tie are re-ranked by `cascade_order`, and
    `tiebreak(row)` supplies that row's `(weekly[n, W], players[n, P])` — a callback rather
    than an argument so the pod-shaped weekly and per-player arrays are only ever
    materialized for the handful of rows that need them.
    """
    shape = totals.shape
    n = shape[-1]
    flat = np.ascontiguousarray(totals).reshape(-1, n)
    order = np.argsort(-flat, axis=-1, kind="stable")
    place = np.empty_like(order)
    np.put_along_axis(place, order, np.broadcast_to(np.arange(n), flat.shape).copy(),
                      axis=-1)

    if n > 1:
        ranked = np.take_along_axis(flat, order, axis=-1)
        tied = np.nonzero((ranked[:, :-1] == ranked[:, 1:]).any(axis=-1))[0]
        if len(tied):
            rng = rng or np.random.default_rng(SEED)
            for row in tied:
                weekly, players = ((np.zeros((n, 1)), np.zeros((n, 1))) if tiebreak is None
                                   else tiebreak(int(row)))
                for lo, hi in _tied_runs(ranked[row]):
                    members = order[row, lo:hi]
                    resolved = cascade_order(flat[row, members], weekly[members],
                                             players[members], rng.random(hi - lo))
                    place[row, members[resolved]] = np.arange(lo, hi)
    return place.reshape(shape)


def _tied_runs(ranked: np.ndarray) -> list[tuple[int, int]]:
    """The `[lo, hi)` spans of equal totals in one already-sorted row.

    Only the tied span is re-ordered, never the whole row. That matters more than it looks:
    dk_pts sums are `float32`, so in a field of a few thousand entries the birthday
    collisions alone put a tie in most sims, and re-sorting every row through a Python
    cascade would cost more than the rest of the module put together.
    """
    edges = np.nonzero(ranked[:-1] != ranked[1:])[0] + 1
    bounds = np.concatenate([[0], edges, [len(ranked)]])
    return [(int(a), int(b)) for a, b in zip(bounds[:-1], bounds[1:]) if b - a > 1]


# ── 3. Structure, read from the captured CSVs and never written down ──────────

def tournaments(root: Path = economics.ROOT) -> list[str]:
    return list(economics.load_metadata(root)["type"])


def round_payouts(tournament: str, rnd: int, pod_size: int,
                  root: Path = economics.ROOT) -> np.ndarray:
    """Cash by finishing place in one round's pod — index 0 is first place.

    `economics.payout_curve` expands the final round's banded rows; rounds 2 and 3 pay too
    (`600k_shootaround` pays 11 of 12 at $30) and are expanded the same way from the same
    frame. Advancing places carry no cash of their own, which is why an advancer's row is
    zero here and its money arrives from the round it stops at.
    """
    prizes = economics.load_prizes(root)
    bands = prizes[(prizes["tournament"] == tournament) & (prizes["round"] == rnd)
                   & (prizes["prize_type"] == "cash")]
    payout = np.zeros(pod_size, dtype=float)
    for band in bands.itertuples():
        lo, hi = int(band.high_place), min(int(band.low_place), pod_size)
        payout[lo - 1:hi] = float(band.cash_amount)
    return payout


def structure(tournament: str, root: Path = economics.ROOT) -> list[dict]:
    """One record per round: pod size, how many advance, and the cash by place.

    Every number comes from `economics.advance_table`, which derives them from the two
    captured CSVs. The one thing added here is the payout vector, and the one thing
    re-derived is the field-size chain — `derived_field` is computed forward from the entry
    count and compared against the table's own `field_entries` by `verify_chain`.
    """
    table = economics.advance_table(root)
    sub = table[table["tournament"] == tournament].sort_values("round")
    if sub.empty:
        raise KeyError(f"no tournament named {tournament!r} in {economics.METADATA}")

    rounds, field = [], float(sub["field_entries"].iloc[0])
    for row in sub.itertuples():
        pod, advance = int(row.pod_size), int(row.n_advance)
        payout = round_payouts(tournament, int(row.round), pod, root)
        rounds.append({
            "tournament": tournament,
            "round": int(row.round),
            "pod_size": pod,
            "n_advance": advance,
            "field_entries": float(row.field_entries),
            "derived_field": field,
            "payout": payout,
            "cash_places": int(row.cash_places),
            "prize_sum": float(payout.sum()),
            "zero_consolation": bool(row.zero_consolation),
        })
        field = field / pod * advance if advance else 0.0
    return rounds


def verify_chain(root: Path = economics.ROOT) -> pd.DataFrame:
    """Does chaining the entry count forward through the pods reproduce every field size?

    The arithmetic `economics.advance_table` does internally, redone from the outside and
    reported per round, so a pod size or advance count that does not compose shows up here
    rather than as a wrong payout three modules downstream.
    """
    rows = []
    for tournament in tournaments(root):
        for rnd in structure(tournament, root):
            rows.append({k: v for k, v in rnd.items() if k != "payout"}
                        | {"field_error": rnd["derived_field"] - rnd["field_entries"],
                           "integral": abs(rnd["derived_field"]
                                           - round(rnd["derived_field"])) < 1e-9})
    return pd.DataFrame(rows)


def symmetric_null(tournament: str, root: Path = economics.ROOT) -> dict:
    """What an exchangeable entry is worth — known exactly, and therefore a check.

    In a field where every entry is drawn from the same process each one reaches round `r`
    with probability `prod(n_advance / pod_size)` and, having reached it, collects the
    round's average cash per pod seat. Summed, that must be `total_prizes / total_entries`,
    because the prize pool is fully paid out — so `roi == -rake` exactly. A wrong pod size,
    a dropped cash band or a mis-chained field size all break the identity.
    """
    meta = economics.load_metadata(root).rename(columns={"type": "tournament"})
    row = meta[meta["tournament"] == tournament].iloc[0]
    fee, entries = float(row.entry_fee_per_team), float(row.total_entries)

    rounds = structure(tournament, root)
    reach, expected, by_round = 1.0, 0.0, []
    for rnd in rounds:
        cash = rnd["prize_sum"] / rnd["pod_size"] * reach
        expected += cash
        by_round.append({"round": rnd["round"], "p_reach": reach, "ev": cash})
        reach *= rnd["n_advance"] / rnd["pod_size"]

    share = economics.rake(entries, fee, float(row.total_prizes))
    return {"tournament": tournament,
            "entry_fee": fee,
            "rake": share,
            "break_even_hurdle": economics.break_even_hurdle(share),
            "p_advance_round_1": rounds[0]["n_advance"] / rounds[0]["pod_size"],
            "expected_payout": expected,
            "prize_pool_per_entry": float(row.total_prizes) / entries,
            "roi": expected / fee - 1.0,
            "by_round": by_round}


# ── 4. The bracket ────────────────────────────────────────────────────────────

def deal_pods(pool: np.ndarray, pod_size: int,
              rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Shuffle the surviving field within each sim and deal it into pods.

    Returns the pods and whatever did not fill one. Against all five captured structures
    the remainder is always empty — every round's field divides exactly by its pod size,
    which is what `verify_chain` asserts — but a partial pod is what wildcards exist for.
    """
    n_sims, n_pool = pool.shape
    order = np.argsort(rng.random((n_sims, n_pool)), axis=1)
    shuffled = np.take_along_axis(pool, order, axis=1)
    n_pods = n_pool // pod_size
    podded = shuffled[:, :n_pods * pod_size].reshape(n_sims, n_pods, pod_size)
    return podded, shuffled[:, n_pods * pod_size:]


def _pod_tiebreak(members: np.ndarray, weekly: np.ndarray,
                  players: np.ndarray | None, n_cols: int):
    """`rank_within_pod`'s callback for pods laid out as `[sim, pod, seat]`.

    `members[sim, pod, seat]` indexes the first axis of `weekly` (`[entry, week, sim]`) and
    `players` (`[entry, roster, sim]`), and only the rows that actually tie are built.
    """
    def tiebreak(row: int):
        sim, group = divmod(row, n_cols)
        seats = members[sim, group]
        week = weekly[seats, :, sim]
        split = (np.zeros((len(seats), 1)) if players is None
                 else players[seats, :, sim])
        return week, split
    return tiebreak


def advance_from_pods(podded: np.ndarray, place: np.ndarray, spare: np.ndarray,
                      n_advance: int, target: int, totals: np.ndarray,
                      tiebreak_for) -> np.ndarray:
    """Who reaches the next round: every pod's top `n_advance`, then wildcards.

    Wildcards are DK's own rule — "if there are not enough advancing entries to fill future
    round contests, there will be wild card entries ... the entries that scored the highest
    fantasy points among the non-automatically advancing entries". They fire only when the
    pods cannot deliver the round's target field, which against the captured structures
    never happens; the path is tested directly rather than left to trust.
    """
    n_sims = podded.shape[0]
    seat = np.argsort(place, axis=2)
    advanced = np.take_along_axis(podded, seat[:, :, :n_advance], axis=2)
    advanced = advanced.reshape(n_sims, -1)
    if advanced.shape[1] >= target:
        return advanced[:, :target]

    beaten = np.take_along_axis(podded, seat[:, :, n_advance:], axis=2).reshape(n_sims, -1)
    rest = np.concatenate([beaten, spare], axis=1)
    order = rank_within_pod(totals[rest, np.arange(n_sims)[:, None]],
                            tiebreak_for(rest[:, None, :], 1))
    picked = np.take_along_axis(rest, np.argsort(order, axis=1), axis=1)
    return np.concatenate([advanced, picked[:, :target - advanced.shape[1]]], axis=1)


def simulate_bracket(scored: dict, tournament: str, rng: np.random.Generator,
                     members: np.ndarray | None = None,
                     root: Path = economics.ROOT) -> dict:
    """Play one tournament out, pod by pod, over a field of its real size.

    **Progression is dealt and ranked, not modelled.** `scored` carries every entry in the
    contest — `total_entries` of them — so each round shuffles the survivors, deals them
    into real pods, ranks each pod by the rules' own cascade, and carries the top
    `n_advance` forward. An entry's place is its place; there is no distribution over
    pod-mates to get right, no survivor-population approximation, and the tie-break is
    applied **within the contest that is actually being decided**, which is what the rules
    describe.

    The version this replaced carried the survivor population as a per-entry weight and
    drew an entry's place from it parametrically. That existed only because the field had
    been sized by a knob rather than by the contest: at 3,000 entries `600k_shootaround`'s
    1-in-720 advance rate leaves four survivors against a 49-entry final table. Once the
    field is the real 35,280 the whole problem disappears — 5,880, 490 and 49 are all real
    populations — and with it the two bugs the parametric model had already cost (a
    binomial pod-mate count where the draw is without replacement, and an off-by-one on
    whether an entry joins the field or occupies one of its slots).

    Our own entries are simply *in* the field, at known row indices. That is what DK does:
    they are entries like any other. It also makes two checks free — every round's survivor
    count is the published field size exactly, and the payouts sum to the prize pool
    exactly — where before both were Monte Carlo estimates.

    `members` says which rows of `scored` fill the contest's seats, defaulting to the first
    `field_entries` of them. **It is what keeps a strong entry from contaminating the
    null**: an entry that wins a lot takes its winnings *from the other entries*, since the
    pool is fixed, so measuring an exchangeable entry's value in a field that also contains
    a deliberately strong one measures the wrong thing. The effect is not subtle in a small
    contest — `88k_alley_oop` has 216 entries and a $20,000 top prize on an $88,000 pool, so
    one entry that always reaches the final table moves everyone else's ROI by more than
    twenty points. So the null is run on a pure field and the benchmark is run separately,
    substituted into one seat.
    """
    rounds = structure(tournament, root)
    n_entries, _, n_sims = scored["round_total"].shape
    field = int(rounds[0]["field_entries"])
    if n_entries < field:
        raise ValueError(f"{tournament} needs {field:,} entries and only {n_entries:,} "
                         f"were scored")

    seats = np.arange(field) if members is None else np.asarray(members)
    if len(seats) != field:
        raise ValueError(f"{tournament} seats {field:,} entries and `members` names "
                         f"{len(seats):,}")
    pool = np.tile(seats, (n_sims, 1))
    sims = np.arange(n_sims)
    reached = np.zeros((n_entries, len(rounds), n_sims), dtype=bool)
    place = np.zeros((n_entries, len(rounds), n_sims), dtype=np.int32)
    cash = np.zeros((n_entries, len(rounds), n_sims), dtype=np.float64)
    survivors, paid = [], []

    for i, rnd in enumerate(rounds):
        pod, n_advance = rnd["pod_size"], rnd["n_advance"]
        survivors.append(int(pool.shape[1]))
        np.put_along_axis(reached[:, i, :].T, pool, True, axis=1)

        totals = scored["round_total"][:, i, :]
        weekly = round_weeks(scored, rnd["round"])
        players = (None if scored["player_round"] is None
                   else scored["player_round"][:, i])

        def tiebreak_for(members, n_cols):
            return _pod_tiebreak(members, weekly, players, n_cols)

        podded, spare = deal_pods(pool, pod, rng)
        seats = rank_within_pod(totals[podded, sims[:, None, None]],
                                tiebreak_for(podded, podded.shape[1]), rng)
        place[podded, i, sims[:, None, None]] = seats + 1
        cash[podded, i, sims[:, None, None]] = rnd["payout"][seats]
        paid.append(float(cash[:, i, :].sum(axis=0).mean()))

        if i + 1 < len(rounds):
            target = int(round(rounds[i + 1]["field_entries"]))
            pool = advance_from_pods(podded, seats, spare, n_advance, target, totals,
                                     tiebreak_for)

    return {"tournament": tournament, "rounds": rounds, "reached": reached, "cash": cash,
            "place": place, "survivors": survivors, "paid_per_round": paid,
            "field": field, "members": seats}


# ── 5. The placeholder field, which `src/sim/draft.py` replaces ──────────────

def placeholder_field(board: pd.DataFrame, masks: np.ndarray, n_entries: int,
                      rng: np.random.Generator, tau: float = FIELD_TAU,
                      pod_size: int = ROUND_ONE_POD) -> np.ndarray:
    """`n_entries x 16` rosters from Gumbel-noised snake drafts of `pod_size` entries each.

    A stand-in for a field, not a field model — build item 6 (`src/sim/draft.py`) owns the
    real one, and what it replaces is the *ranking* this reads (DK's recalibrated ADP
    consensus rather than the simulator's own projection) plus DK's autodraft caps and the
    `rank_noise_sd` Gate B fits. The draft mechanism here is the real one and stays.

    **Scarcity is the part that cannot be skipped, and the first version skipped it.**
    Drawing each entry's sixteen independently off one board gave every entry the same top
    ten — mean board rank of a pick **9.4**, against the ~96 a twelve-man draft implies —
    so the field was both far too strong and largely identical, and identical rosters tie
    at every level of the cascade. Drafting twelve entries against a shared board consumes
    192 players and makes rosters disjoint within a pod, which is what a real pod looks
    like.

    Legality is enforced during the draft rather than repaired afterwards: an entry whose
    remaining picks exactly equal the positions it still owes is restricted to those
    positions. That is roughly what DK's own 8 G / 8 F / 3 C autodraft caps do from the
    other direction, and it guarantees every roster can seat seven.
    """
    n_pool = len(board)
    if n_pool < ROSTER_SIZE:
        raise ValueError(f"board has {n_pool} players, below a {ROSTER_SIZE}-man roster")
    if n_pool < pod_size * ROSTER_SIZE:
        raise ValueError(f"board has {n_pool} players, below the "
                         f"{pod_size * ROSTER_SIZE} a {pod_size}-entry draft consumes")
    for i, need in enumerate(SLOT_CAPACITY):
        if int((masks >> i & 1).sum()) < need * pod_size:
            raise ValueError(f"board carries fewer than {need * pod_size} "
                             f"{POSITIONS[i]} — a {pod_size}-entry draft cannot leave "
                             f"every entry a legal lineup")

    n_pods = -(-n_entries // pod_size)
    rank = np.argsort(np.argsort(-board["projection"].to_numpy()))
    keys = -rank[None, None, :] / max(tau, 1e-9) + rng.gumbel(
        size=(n_pods, pod_size, n_pool))

    eligible = np.stack([(masks >> i & 1).astype(bool)
                         for i in range(len(POSITIONS))], axis=1)   # [player, position]
    taken = np.zeros((n_pods, n_pool), dtype=bool)
    have = np.zeros((n_pods, pod_size, len(POSITIONS)), dtype=np.int16)
    rosters = np.zeros((n_pods, pod_size, ROSTER_SIZE), dtype=np.int64)
    pods = np.arange(n_pods)

    for pick in range(ROSTER_SIZE):
        order = range(pod_size) if pick % 2 == 0 else reversed(range(pod_size))
        for seat in order:
            owed = np.maximum(np.array(SLOT_CAPACITY) - have[:, seat, :], 0)
            forced = owed.sum(axis=1) >= (ROSTER_SIZE - pick)
            allowed = ~taken
            if forced.any():
                needed = (owed[:, None, :] > 0) & eligible[None, :, :]
                allowed = np.where(forced[:, None], allowed & needed.any(axis=2), allowed)
            choice = np.argmax(np.where(allowed, keys[:, seat, :], -np.inf), axis=1)
            rosters[:, seat, pick] = choice
            taken[pods, choice] = True
            have[:, seat, :] += eligible[choice]

    rosters = rosters.reshape(-1, ROSTER_SIZE)[:n_entries]
    if (~roster_is_legal(masks[rosters])).any():
        raise ValueError("the drafted field contains a roster that cannot seat seven")
    return rosters


def bootstrap_roi(payout: np.ndarray, entry_fee: float, rng: np.random.Generator,
                  draws: int = 2000) -> tuple[float, float]:
    """A 95% interval on the null ROI, resampled over **entries**.

    The resample unit is the entry, and that is the correction to a first version that
    resampled sims. Two entries differ by their *rosters*, which is fixed across sims — so a
    sim-resample cannot see roster variability at all, and it reported intervals far too
    narrow to cover a sample whose rows were not drawn at random. The interval is not
    decoration here: `600k_shootaround` reaches round 4 on
    0.139% of entries and pays 10,000x at the top of it, so at any affordable number of sims
    the expected count of top prizes is order one and the ROI point estimate swings by tens
    of percent. `select-on-p-advance-report-roi` is the same observation from the other end
    — P(reach round r) resolves orders of magnitude faster, and it is what the null check
    should actually be read on.

    The interval collapses to a point when the sample is the **whole** field, and that is
    correct rather than broken: the pool is fully paid out, so the mean over every entry is
    `total_prizes / total_entries` in every sim by construction. `null_is_whole_field` says
    when that has happened — it does for `88k_alley_oop`, whose 216 entries are fewer than
    the default sample size.
    """
    per_entry = payout.mean(axis=1)
    idx = rng.integers(0, len(per_entry), size=(draws, len(per_entry)))
    means = per_entry[idx].mean(axis=1) / entry_fee - 1.0
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def top_roster(masks: np.ndarray, key: np.ndarray) -> np.ndarray:
    """The best legal sixteen by `key`, ignoring scarcity — the benchmark entry.

    Not a strategy and not drafted against anyone: it takes the board's best players
    outright, subject only to seating the slate's minimums. Its job is to show that the
    bracket discriminates at all, since an entry that could never be drafted should clear
    a field that had to take turns.
    """
    order = [int(p) for p in np.argsort(-key)]
    chosen: list[int] = []
    for i, need in enumerate(SLOT_CAPACITY):
        chosen += [p for p in order if (masks[p] >> i & 1) and p not in chosen][:need]
    chosen += [p for p in order if p not in chosen][:ROSTER_SIZE - len(chosen)]
    return np.array(chosen, dtype=np.int64)


# ── 6. The entry point ────────────────────────────────────────────────────────

def load_tensor(features_dir: Path, season: str, label: str = "") -> dict:
    """The season's tensor. `label` names a VARIANT written beside the shipped one.

    `docs/rookie-rates-plan.md` §5h: the rookie-inclusive replay needs a tensor drawn over
    a wider unit population without overwriting the one every audited downstream artifact
    was built on, so the label rides on the filename exactly as `--field`'s suffix rides
    on the sweep's. A labelled tensor is a different measurement, never a re-decision.
    """
    path = features_dir / f"sim_tensor_{season}{label}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `make simulate-season` first, it writes the "
            f"player x scoring_period x sim tensor this module scores"
            + (f" (labelled runs: `--tensor-label {label}`)" if label else ""))
    with np.load(path, allow_pickle=False) as z:
        out = {"dk_pts": z["dk_pts"], "player_id": z["player_id"],
               "tournament_round": z["tournament_round"],
               "season": str(z["season"]), "fit_window": str(z["fit_window"]),
               "n_sims": int(z["n_sims"])}
        # Written since `docs/rookie-rates-plan.md` §5f put two rate families in the
        # tensor; a tensor drawn before that carries one family and says so here rather
        # than leaving the consumer to guess from the population size.
        n = len(out["player_id"])
        out["unit_family"] = (z["unit_family"].astype(str) if "unit_family" in z
                              else np.full(n, VETERAN_FAMILY))
        out["lag_rung"] = (z["lag_rung"].astype(str) if "lag_rung" in z
                           else np.full(n, RUNG_ZERO))
    return out


def build_board(features_dir: Path, tensor: dict) -> pd.DataFrame:
    """The draftable board: the tensor's players, joined to their DK position eligibility.

    A player the tensor scores but `make draft-pool` does not carry cannot be drafted, and
    one the pool carries but the tensor does not cannot be scored; both are dropped and
    counted rather than silently filled.
    """
    pool = pd.read_parquet(features_dir / "draft_pool.parquet")
    pool = pool[pool["season"] == tensor["season"]].drop_duplicates("player_id")
    index = pd.DataFrame({"player_id": tensor["player_id"],
                          "tensor_row": np.arange(len(tensor["player_id"]))})
    board = index.merge(pool, on="player_id", how="inner")
    board["projection"] = tensor["dk_pts"][board["tensor_row"].to_numpy()].sum(
        axis=1).mean(axis=1)
    return board.sort_values("projection", ascending=False).reset_index(drop=True)


def split_frame(cfg: dict) -> pd.DataFrame:
    """The frame the split is derived from — the component design, not the draft pool.

    **`draft_pool.parquet` is the wrong frame and using it reads a test season.** The pool
    carries the live 2026-27 production board, so its last two labels are 2025-26 and
    2026-27 and `selection_split` hands back 2023-24 and **2024-25** as "validation" — one
    season further forward than the project's, and 2024-25 is held out. The guard did not
    object, because the guard believed the frame. So the split comes from the same design
    `make simulate-season` builds its tensors against, which is also the only frame that
    can be right here: the bracket scores those tensors.
    """
    targets = pd.read_parquet(Path(cfg["data"]["features_dir"]) / "component_targets.parquet")
    return component_build_design(targets, cfg["data"]["seasons"], cfg["data"]["raw_dir"])


def run(cfg: dict, seasons: list[str] | None = None,
        n_sims: int | None = None, seed: int | None = None) -> dict[str, Path]:
    """Play all five captured tournaments out on each validation season.

    **One field is drafted and scored per season**, sized to the largest tournament, and
    each contest takes the prefix it needs — 35,280 / 17,640 / 14,688 / 432 / 216 entries,
    every one of them a multiple of the 12-entry round-1 pod. Entries are exchangeable and
    drafted in pods, so a prefix is a valid field, and it costs one scoring pass instead of
    five. Row 0 is a benchmark entry that took the board's best sixteen outright; every
    other row is an ordinary drafted entry, which is what makes the per-entry null a
    genuine sample rather than a special case.
    """
    features_dir = Path(cfg["data"]["features_dir"])
    out_dir = Path(cfg["evaluation"]["predictions_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_sim = cfg.get("sim", {})
    cfg_bracket = cfg_sim.get("bracket", {})
    entered = cfg_sim.get("tournaments", {})
    n_null = int(cfg_bracket.get("n_null", N_NULL))
    seed = int(SEED if seed is None else seed)

    design = split_frame(cfg)
    seasons = seasons or validation_seasons(design)
    for season in seasons:
        assert_season_allowed(season, design)

    meta = economics.load_metadata().set_index("type")
    order = list(meta.sort_values("total_entries", ascending=False).index)
    n_field = int(meta["total_entries"].max())

    print("Bracket — weekly lineups, four rounds, ties, wildcards and payouts")
    print(f"  structure read from `dashboard.economics`; nothing per-tournament is "
          f"hardcoded here")
    print(f"  lineup: {LINEUP_SIZE} of {ROSTER_SIZE} — "
          f"{'/'.join(f'{c} {p}' for c, p in zip(SLOT_CAPACITY, POSITIONS))} + "
          f"{UTIL_SLOTS} UTIL, seated by matroid greedy (exact, not first-fit)")
    print(f"  progression is DEALT and RANKED, not modelled — real pods, real cuts")
    print(f"  The test split is LOCKED — seasons go through `held_out.selection_split`.")

    chain = verify_chain()
    bad = chain[(chain["field_error"].abs() > 1e-6) | (~chain["integral"])]
    print(f"\nField-size chain: {len(chain):,} rounds over "
          f"{chain['tournament'].nunique()} tournaments, {len(bad)} disagreeing with "
          f"`economics.advance_table`")
    if len(bad):
        raise ValueError(f"the advance chain does not reproduce the published field "
                         f"sizes:\n{bad}")

    nulls = pd.DataFrame([{k: v for k, v in symmetric_null(t).items() if k != "by_round"}
                          for t in order])
    nulls["roi_error"] = nulls["roi"] + nulls["rake"]
    print("\nThe symmetric-field null — an exchangeable entry's exact value:")
    for row in nulls.itertuples():
        flag = " " if abs(row.roi_error) <= NULL_TOLERANCE else "🔴"
        print(f"{flag} {row.tournament:<20} E[payout] ${row.expected_payout:>9.4f} on a "
              f"${row.entry_fee:>6.2f} entry   ROI {row.roi:+.6f}  against "
              f"-rake {-row.rake:+.6f}  ({row.roi_error:+.2e})")
    broken = nulls[nulls["roi_error"].abs() > NULL_TOLERANCE]
    if len(broken):
        raise ValueError(f"the payout tables for {sorted(broken['tournament'])} do not "
                         f"reconcile to their published prize pool")

    frames, entry_rows, null_rows = [], [], []
    for season in seasons:
        print(f"\n── {season} ──")
        tensor = load_tensor(features_dir, season)
        board = build_board(features_dir, tensor)
        sims = min(int(n_sims or cfg_bracket.get("n_sims", N_SIMS_BRACKET)),
                   tensor["n_sims"])
        dk_pts = tensor["dk_pts"][board["tensor_row"].to_numpy()][:, :, :sims]
        masks = position_masks(board)
        round_of_period = tensor["tournament_round"]
        rng = np.random.default_rng(seed)

        print(f"  board {len(board):,} draftable players of {len(tensor['player_id']):,} "
              f"scored ({', '.join(f'{p} {int((masks >> i & 1).sum()):,}' for i, p in enumerate(POSITIONS))})"
              f"; {sims:,} of {tensor['n_sims']:,} sims at the "
              f"`{tensor['fit_window']}` window")

        rosters = placeholder_field(board, masks, n_field, rng)
        benchmark = n_field                      # one extra row, outside every field
        rosters = np.concatenate(
            [rosters, top_roster(masks, board["projection"].to_numpy())[None, :]], axis=0)
        scored = score_rosters(dk_pts, rosters, masks, round_of_period)
        print(f"  field {n_field:,} entries drafted and scored once, reused by every "
              f"contest (`src/sim/draft.py` replaces the ranking it drafts off)")
        print(f"  plus 1 benchmark entry, held OUT of the field and substituted into one "
              f"seat for its own run — a strong entry's winnings come out of everyone "
              f"else's")

        for tournament in order:
            result = simulate_bracket(scored, tournament, rng)
            spot = np.concatenate([[benchmark],
                                   np.arange(1, int(result["field"]))])
            bench = simulate_bracket(scored, tournament, rng, members=spot)
            fee = float(meta.loc[tournament, "entry_fee_per_team"])
            analytic = symmetric_null(tournament)
            payout = result["cash"].sum(axis=1)
            bench_payout = bench["cash"].sum(axis=1)[benchmark]
            drawn = min(n_null, result["field"])
            rows = (np.arange(result["field"]) if drawn == result["field"]
                    else np.random.default_rng(seed).choice(result["field"], drawn,
                                                            replace=False))
            sample = payout[rows]
            null_roi = float(sample.mean()) / fee - 1.0
            lo, hi = bootstrap_roi(sample, fee, np.random.default_rng(seed))
            paid = float(payout.sum(axis=0).mean())
            pool_error = paid - float(meta.loc[tournament, "total_prizes"])

            for i, rnd in enumerate(result["rounds"]):
                frames.append({
                    "season": season, "tournament": tournament, "round": rnd["round"],
                    "pod_size": rnd["pod_size"], "n_advance": rnd["n_advance"],
                    "field_entries": rnd["field_entries"],
                    "survivors": result["survivors"][i],
                    "survivors_match": result["survivors"][i] == int(rnd["field_entries"]),
                    "cash_places": rnd["cash_places"], "prize_sum": rnd["prize_sum"],
                    "paid_simulated": result["paid_per_round"][i],
                    "paid_expected": rnd["prize_sum"] * rnd["field_entries"]
                    / rnd["pod_size"],
                    "p_reach_analytic": analytic["by_round"][i]["p_reach"],
                    "p_reach_benchmark": float(bench["reached"][benchmark, i].mean()),
                })
            null_rows.append({
                "season": season, "tournament": tournament, "entries": result["field"],
                "n_null_entries": drawn, "n_sims": sims, "entry_fee": fee,
                "null_is_whole_field": drawn == result["field"],
                "rake": analytic["rake"], "analytic_ev": analytic["expected_payout"],
                "analytic_roi": analytic["roi"], "simulated_ev": float(sample.mean()),
                "simulated_roi": null_roi, "simulated_roi_lo": lo, "simulated_roi_hi": hi,
                "roi_covers_analytic": bool(lo - NULL_TOLERANCE <= analytic["roi"]
                                            <= hi + NULL_TOLERANCE),
                "paid_simulated": paid,
                "paid_expected": float(meta.loc[tournament, "total_prizes"]),
                "pool_error": pool_error,
                "survivors_match_all": bool(all(
                    result["survivors"][i] == int(rnd["field_entries"])
                    for i, rnd in enumerate(result["rounds"]))),
            })
            entry_rows.append({
                "season": season, "tournament": tournament, "entry": "best_available",
                "entered": tournament in entered,
                "n_entries": int(entered.get(tournament, {}).get("entries", 0)),
                "entry_fee": fee,
                "p_advance": float(bench["reached"][benchmark, 1].mean()),
                "ev": float(bench_payout.mean()),
                "roi": float(bench_payout.mean()) / fee - 1.0,
                "null_roi": null_roi,
                "break_even_hurdle": analytic["break_even_hurdle"],
            })
            mark = "*" if tournament in entered else " "
            print(f" {mark}{tournament:<19} survivors "
                  f"{' -> '.join(f'{v:,}' for v in result['survivors'])}"
                  f"  paid ${paid:,.2f} of ${meta.loc[tournament, 'total_prizes']:,.0f} "
                  f"(err {pool_error:+.2e})")
            print(f"  {'':<19} null ROI {null_roi:+.4f} [{lo:+.4f}, {hi:+.4f}] against "
                  f"{analytic['roi']:+.4f}"
                  f"{'  ✅' if null_rows[-1]['roi_covers_analytic'] else '  /!\\ not covered'}"
                  f";  best-available P(advance) {entry_rows[-1]['p_advance']:.4f}")

    paths = {}
    for name, frame in (("bracket_structure", pd.DataFrame(frames)),
                        ("bracket_null", pd.DataFrame(null_rows)),
                        ("bracket_entries", pd.DataFrame(entry_rows))):
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
                        help="simulated seasons. The field's per-period array is "
                             "entries x 20 x n_sims float32, so this is the memory knob.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open("configs/default.yaml"))
    run(cfg, seasons=args.season, n_sims=args.n_sims, seed=args.seed)
