"""Tests for the bracket — lineups, tie-breaks, the advance chain and payouts.

None of these needs a sim tensor, a posterior artifact or a sampler. What is worth pinning
in `src/sim/bracket.py` is the arithmetic whose failures are **silent**: a lineup that seats
a dual-eligible player greedily still returns a plausible total, a tie-break that quietly
favours one population still produces a plausible winner, and a payout table with a dropped
band still pays somebody. Every one of those is arithmetic over synthetic arrays, so it runs
in milliseconds.

The structural tests do read the two captured DraftKings CSVs, because the whole design
claim is that nothing about a tournament is written down in the module — checking that
against a synthetic table would check the opposite of what matters.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

import numpy as np
import pandas as pd
import pytest

from dashboard import economics
from src.models import held_out
from src.models.held_out import HeldOutLocked, unlocked
from src.sim import bracket as B

G, F, C = 1, 2, 4       # eligibility bitmasks, matching B.POSITIONS order


# ── Synthetic builders ────────────────────────────────────────────────────────

def _roster(masks: list[int]) -> np.ndarray:
    """A 16-man roster of eligibility masks, padded with guards if short."""
    padded = list(masks) + [G] * (B.ROSTER_SIZE - len(masks))
    return np.array(padded[:B.ROSTER_SIZE], dtype=np.intp)


def _scores(values: list[float]) -> np.ndarray:
    """Sixteen player scores, padded with a bench nobody would start."""
    padded = list(values) + [-99.0] * (B.ROSTER_SIZE - len(values))
    return np.array(padded[:B.ROSTER_SIZE], dtype=float)


def _entry(period: np.ndarray) -> dict:
    """A `score_rosters`-shaped result from `[entry, period, sim]` scores alone."""
    round_of_period = np.array([1, 2, 3, 4])
    return {"period": period, "round_total": period.copy(), "player_round": None,
            "round_of_period": round_of_period, "rounds": round_of_period}


# ── 1. Lineup selection ───────────────────────────────────────────────────────

def test_the_slate_is_the_one_in_the_rules():
    """2 G, 2 F, 1 C, 2 UTIL out of 16 — seven starters and nine bench."""
    assert B.LINEUP_SIZE == 7
    assert B.ROSTER_SIZE == 16
    assert B.BENCH_SIZE == 9
    assert dict(zip(B.POSITIONS, B.SLOT_CAPACITY)) == {"G": 2, "F": 2, "C": 1}
    assert B.UTIL_SLOTS == 2


def test_lineup_limits_are_halls_condition_for_the_slate():
    """Each limit is the seats that subset commands, plus the two open UTIL seats."""
    limits = B.lineup_limits()
    assert limits[0b000] == 0                    # no position at all cannot be seated
    assert limits[0b001] == 2 + 2                # G-only:  2 G  + 2 UTIL
    assert limits[0b010] == 2 + 2                # F-only:  2 F  + 2 UTIL
    assert limits[0b100] == 1 + 2                # C-only:  1 C  + 2 UTIL
    assert limits[0b011] == 4 + 2                # G or F
    assert limits[0b111] == 5 + 2                # everyone — the roster-size constraint
    assert limits[0b111] == B.LINEUP_SIZE


def test_best_lineup_seats_exactly_seven_and_takes_the_top_of_each_position():
    """Five guards, five forwards, six centers, scores descending by index."""
    masks = _roster([G] * 5 + [F] * 5 + [C] * 6)
    scores = np.arange(16, 0, -1).astype(float)
    total, started = B.best_lineup(scores, masks)

    assert started.sum() == 7
    # 4 guards (2 G slots + 2 UTIL), 2 forwards, 1 center — the best of each group
    assert sorted(np.nonzero(started)[0]) == [0, 1, 2, 3, 5, 6, 10]
    assert total == pytest.approx(16 + 15 + 14 + 13 + 11 + 10 + 6)


def test_the_util_slots_absorb_the_overflow_of_the_deepest_position():
    """Twelve guards, two forwards and two centers: the UTIL seats both go to guards."""
    masks = _roster([G] * 12 + [F, F, C, C])
    scores = _scores([30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15])
    total, started = B.best_lineup(scores, masks)

    assert started.sum() == 7
    assert started[:12].sum() == 4                     # 2 G slots + both UTIL seats
    assert started[12] and started[13]                 # both forwards
    assert started[14] and not started[15]             # the better of the two centers
    assert total == pytest.approx(30 + 29 + 28 + 27 + 18 + 17 + 16)


def test_a_roster_with_only_one_forward_cannot_seat_seven():
    """Not a fallback case — the second F slot has nobody to fill it, and UTIL cannot help.

    Worth pinning because the failure is quiet: `best_lineup` returns a six-man total that
    still looks like a score. `roster_is_legal` is what callers check, and the field builder
    is what has to stop it happening.
    """
    masks = _roster([G] * 14 + [F, C])
    total, started = B.best_lineup(_scores([30, 29, 28, 27, 26, 25, 24, 23]), masks)
    assert started.sum() == 6
    assert not B.roster_is_legal(masks[None, :])[0]


def test_a_dual_eligible_player_is_placed_to_maximize_the_total_not_greedily():
    """The case a first-fit assigner gets wrong, and it is not a tie.

    Player 0 is a G/F dual and the top scorer. A first-fit assigner walking the slate in
    slot order seats him at guard, which fills the guard seats with him and the next
    guard — and then the 2 UTIL seats take two more guards and the *fifth* guard is locked
    out entirely, so the lineup has to reach past him to a 23-point center. Seating the
    dual at forward instead lets all four remaining guards in.
    """
    masks = _roster([G | F, G, G, G, G, F, F, C])
    scores = _scores([30, 29, 28, 27, 26, 25, 24, 23])
    total, started = B.best_lineup(scores, masks)

    assert started.sum() == 7
    assert total == pytest.approx(30 + 29 + 28 + 27 + 26 + 25 + 23)   # 188, not 186
    assert started[4]                    # the fifth guard makes the lineup
    assert not started[6]                # at the expense of the second pure forward


def test_best_lineup_beats_a_first_fit_assigner_on_that_roster():
    """The same claim stated as a comparison, so the margin is visible rather than asserted."""
    masks = _roster([G | F, G, G, G, G, F, F, C])
    scores = _scores([30, 29, 28, 27, 26, 25, 24, 23])
    optimal, _ = B.best_lineup(scores, masks)
    assert _first_fit(scores, masks) == pytest.approx(186.0)
    assert optimal == pytest.approx(188.0)
    assert optimal > _first_fit(scores, masks)


def _first_fit(scores: np.ndarray, masks: np.ndarray) -> float:
    """The wrong algorithm, written out once so the test compares against something real.

    Walk players best-first and seat each in the first slot he fits, position slots before
    UTIL. This is what anyone writes if they do not notice the problem is an assignment.
    """
    seats = list(B.SLOT_CAPACITY) + [B.UTIL_SLOTS]
    total = 0.0
    for p in np.argsort(-scores):
        for i in range(len(B.POSITIONS)):
            if seats[i] and (masks[p] >> i & 1):
                seats[i] -= 1
                total += scores[p]
                break
        else:
            if seats[-1] and masks[p]:
                seats[-1] -= 1
                total += scores[p]
    return total


def test_a_roster_with_no_center_cannot_seat_seven():
    """DK enforces this at draft time and the field builder has to as well."""
    assert not B.roster_is_legal(_roster([G] * 8 + [F] * 8)[None, :])[0]
    assert B.roster_is_legal(_roster([G] * 8 + [F] * 7 + [C])[None, :])[0]


def test_score_rosters_sums_periods_into_rounds_and_splits_by_player():
    """The three shapes everything downstream indexes by."""
    n_players, n_periods, n_sims = 20, 4, 3
    dk_pts = np.zeros((n_players, n_periods, n_sims), dtype=np.float32)
    dk_pts[:, :, :] = np.arange(n_players, dtype=np.float32)[:, None, None]
    rosters = np.arange(B.ROSTER_SIZE)[None, :]
    masks = np.array([G] * 7 + [F] * 7 + [C] * 6, dtype=np.intp)[:n_players]
    round_of_period = np.array([1, 1, 2, 3])

    out = B.score_rosters(dk_pts, rosters, masks, round_of_period, track_players=True)
    assert out["period"].shape == (1, n_periods, n_sims)
    assert out["round_total"].shape == (1, 3, n_sims)
    assert out["player_round"].shape == (1, 3, B.ROSTER_SIZE, n_sims)
    # round 1 spans two periods, so its total is twice a single period's
    assert out["round_total"][0, 0, 0] == pytest.approx(2 * out["period"][0, 0, 0])
    # and the per-player split is a partition of the round total
    assert out["player_round"][0, 0, :, 0].sum() == pytest.approx(out["round_total"][0, 0, 0])


def test_score_rosters_chunking_does_not_change_a_single_score():
    """The field runs to tens of thousands of entries, so the gather is chunked.

    A chunk-boundary bug is silent — it produces a real score for the wrong slice of
    entries — so the same rosters are scored at a budget that forces many chunks and at one
    that forces a single chunk, and the two must agree bit for bit.
    """
    rng = np.random.default_rng(0)
    n_players, n_periods, n_sims = 40, 6, 5
    dk_pts = rng.normal(20, 8, size=(n_players, n_periods, n_sims)).astype(np.float32)
    masks = np.array([G] * 16 + [F] * 16 + [C] * 8, dtype=np.intp)
    rosters = np.stack([rng.permutation(n_players)[:B.ROSTER_SIZE] for _ in range(9)])
    rosters = rosters[B.roster_is_legal(masks[rosters])]
    round_of_period = np.array([1, 1, 1, 2, 3, 4])

    one = B.score_rosters(dk_pts, rosters, masks, round_of_period, budget=10**9)
    many = B.score_rosters(dk_pts, rosters, masks, round_of_period, budget=1)
    assert np.array_equal(one["period"], many["period"])
    assert np.array_equal(one["round_total"], many["round_total"])


# ── 2. The cascading tie-break ────────────────────────────────────────────────

def test_equal_totals_break_on_the_highest_single_week():
    """The rules' first tie-break, and the one that decides almost every real pod."""
    totals = np.array([100.0, 100.0])
    weekly = np.array([[40.0, 30.0, 30.0],       # best week 40
                       [60.0, 20.0, 20.0]])      # best week 60 — wins
    players = np.zeros((2, 1))
    order = B.cascade_order(totals, weekly, players, np.zeros(2))
    assert list(order) == [1, 0]


def test_the_weekly_cascade_walks_down_through_the_round():
    """Identical best weeks fall through to the second-best, then the third."""
    totals = np.array([100.0, 100.0, 100.0])
    weekly = np.array([[50.0, 30.0, 20.0],
                       [50.0, 40.0, 10.0],       # same best, better second — wins
                       [50.0, 30.0, 20.0]])
    order = B.cascade_order(totals, weekly, np.zeros((3, 1)), np.zeros(3))
    assert order[0] == 1

    # and when the whole weekly vector ties, the edge has to come from the player level
    weekly[:] = [50.0, 30.0, 20.0]
    players = np.array([[60.0, 40.0], [95.0, 5.0], [50.0, 50.0]])
    order = B.cascade_order(totals, weekly, players, np.zeros(3))
    assert list(order) == [1, 0, 2]               # highest single player: 95, 60, 50


def test_the_player_cascade_walks_down_the_same_way():
    """Identical top scorers fall through to the second-highest player."""
    totals = np.array([100.0, 100.0])
    weekly = np.tile(np.array([50.0, 30.0, 20.0]), (2, 1))
    players = np.array([[60.0, 25.0, 15.0],
                        [60.0, 30.0, 10.0]])      # same top, better second — wins
    order = B.cascade_order(totals, weekly, players, np.zeros(2))
    assert list(order) == [1, 0]


def test_the_cascade_is_order_within_a_row_not_across_rows():
    """Weeks are compared sorted descending, so the calendar order does not matter."""
    totals = np.array([100.0, 100.0])
    weekly = np.array([[30.0, 60.0, 10.0],        # same multiset as the other, shuffled
                       [60.0, 10.0, 30.0]])
    players = np.array([[70.0], [70.0]])
    jitter = np.array([0.9, 0.1])                 # identical all the way down -> random
    order = B.cascade_order(totals, weekly, players, jitter)
    assert list(order) == [1, 0]                  # decided by the jitter alone


def test_rank_within_pod_fast_paths_when_nothing_ties():
    totals = np.array([[10.0, 30.0, 20.0]])
    place = B.rank_within_pod(totals)
    assert list(place[0]) == [2, 0, 1]


def test_rank_within_pod_resolves_only_the_tied_span():
    """A tie between two entries must not disturb the entries around them."""
    totals = np.array([[50.0, 20.0, 20.0, 10.0]])

    def tiebreak(row):
        weekly = np.array([[50.0], [5.0], [20.0], [10.0]])   # entry 2 has the better week
        return weekly, np.zeros((4, 1))

    place = B.rank_within_pod(totals, tiebreak, np.random.default_rng(0))
    assert place[0, 0] == 0                     # untouched
    assert place[0, 3] == 3                     # untouched
    assert place[0, 2] == 1 and place[0, 1] == 2


def test_tied_runs_finds_the_spans():
    assert B._tied_runs(np.array([5.0, 5.0, 4.0, 3.0, 3.0, 3.0])) == [(0, 2), (3, 6)]
    assert B._tied_runs(np.array([5.0, 4.0, 3.0])) == []


# ── 3. Structure, read from the captured CSVs ─────────────────────────────────

def test_the_advance_chain_reproduces_every_published_field_size():
    """Chaining the entry count forward through the pods, for all five tournaments."""
    chain = B.verify_chain()
    assert len(chain) == 20                      # five tournaments x four rounds
    assert set(chain["tournament"]) == set(economics.load_metadata()["type"])
    assert chain["integral"].all()
    assert chain["field_error"].abs().max() < 1e-9
    for tournament in B.tournaments():
        sub = chain[chain["tournament"] == tournament]
        assert list(sub["round"]) == [1, 2, 3, 4]
        assert (sub["pod_size"] > 0).all()


def test_round_one_is_two_of_twelve_and_pays_nothing_in_every_tournament():
    """The zero-consolation round, which is why P(any return) = P(top 2 of 12)."""
    for tournament in B.tournaments():
        first = B.structure(tournament)[0]
        assert first["pod_size"] == economics.ROUND_ONE_POD == 12
        assert first["n_advance"] == 2
        assert first["cash_places"] == 0
        assert first["zero_consolation"]
        assert first["payout"].sum() == 0.0


def test_the_payout_vector_expands_every_band_including_the_deep_ones():
    """`600k_shootaround` pays all 49 round-4 places, not only the headline few."""
    payout = B.round_payouts("600k_shootaround", 4, 49)
    assert len(payout) == 49
    assert (payout > 0).all()
    assert payout[0] == 200_000 and payout[1] == 60_000
    assert payout[20] == 1_000 and payout[48] == 1_000     # the 21-49 band
    assert payout.sum() == 394_200
    assert np.all(np.diff(payout) <= 0)                    # never increases with place


def test_advancing_places_carry_no_cash_of_their_own():
    """An advancer's money comes from the round it stops at, not from advancing."""
    for tournament in B.tournaments():
        for rnd in B.structure(tournament):
            if rnd["n_advance"]:
                assert (rnd["payout"][:rnd["n_advance"]] == 0).all()


def test_the_symmetric_field_null_returns_minus_rake_for_all_five():
    """The identity that checks pod sizes, advance counts and cash tables at once.

    An exchangeable entry reaches round `r` with the chained advance rate and collects the
    round's average cash per seat; summed, that has to be `total_prizes / total_entries`,
    because the pool is paid out in full. It is the tight check on the captured data —
    `check_round_one_pod` deliberately tolerates one tournament differing, and that
    tolerance is what hid a transcription slip in `15k_and_one` until this ran.
    """
    for tournament in B.tournaments():
        null = B.symmetric_null(tournament)
        assert null["expected_payout"] == pytest.approx(null["prize_pool_per_entry"],
                                                        abs=1e-9)
        assert null["roi"] == pytest.approx(-null["rake"], abs=1e-9)
        assert null["p_advance_round_1"] == pytest.approx(1 / 6)


def test_surviving_round_one_guarantees_a_cash_in_both_target_tournaments():
    """Why `P(any return) = P(top 2 of 12)` exactly — the plan's reframing, as a test."""
    for tournament in ("600k_shootaround", "20k_spin_move"):
        second = B.structure(tournament)[1]
        paid_or_advancing = second["cash_places"] + second["n_advance"]
        assert paid_or_advancing == second["pod_size"]
        assert second["payout"][second["n_advance"]:].min() > 0


# ── 4. Advancement, wildcards and the survivor population ────────────────────

def test_deal_pods_partitions_the_field_exactly():
    """Every survivor plays once per round, in exactly one pod."""
    rng = np.random.default_rng(0)
    pool = np.tile(np.arange(48), (5, 1))
    podded, spare = B.deal_pods(pool, 12, rng)
    assert podded.shape == (5, 4, 12)
    assert spare.shape == (5, 0)
    for sim in range(5):
        assert sorted(podded[sim].ravel().tolist()) == list(range(48))
    # a field that does not divide leaves the remainder for the wildcard path
    podded, spare = B.deal_pods(np.tile(np.arange(50), (5, 1)), 12, rng)
    assert podded.shape == (5, 4, 12) and spare.shape == (5, 2)


def test_advance_from_pods_takes_each_pods_top_n():
    """No model — the entries that finished top `n_advance` of a real pod, and no others."""
    podded = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])          # 1 sim, 2 pods of 4
    place = np.array([[[3, 0, 1, 2], [0, 2, 1, 3]]])           # 0 is best
    totals = np.zeros((8, 1))
    out = B.advance_from_pods(podded, place, np.zeros((1, 0), int), 2, 4, totals,
                              lambda m, n: None)
    assert sorted(out[0].tolist()) == [1, 2, 4, 6]


def test_wildcards_fill_a_shortfall_from_the_highest_scoring_non_advancers():
    """DK's rule for a round whose contests did not all fill.

    Forced by asking for a bigger next round than the pods deliver: two pods of four
    advancing one give two, and a target of four takes the best two non-advancers.
    Dormant against the captured structures, whose field-size chains divide exactly.
    """
    podded = np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    place = np.array([[[0, 1, 2, 3], [0, 1, 2, 3]]])
    totals = np.array([[80.], [70.], [60.], [50.], [75.], [65.], [55.], [45.]])
    out = B.advance_from_pods(podded, place, np.zeros((1, 0), int), 1, 4, totals,
                              lambda m, n: None)
    assert sorted(out[0][:2].tolist()) == [0, 4]        # the two pod winners
    assert sorted(out[0][2:].tolist()) == [1, 5]        # then the two best losers, 70 & 65


def test_the_survivor_count_is_the_published_field_size_at_every_round():
    """A structural identity once the field is real, where it used to be an estimate."""
    rng = np.random.default_rng(0)
    for tournament in ("20k_spin_move", "88k_alley_oop"):
        rounds = B.structure(tournament)
        field = int(rounds[0]["field_entries"])
        period = rng.normal(1200, 200, size=(field, 4, 12)).astype(np.float32)
        result = B.simulate_bracket(_entry(period), tournament, rng)
        assert result["survivors"] == [int(r["field_entries"]) for r in rounds]
        # and every entry plays exactly once per round it reaches
        assert result["reached"][:, 0].all()
        assert result["reached"][:, 1].sum(axis=0).tolist() == [
            int(rounds[1]["field_entries"])] * 12


def test_the_payouts_sum_to_the_prize_pool_exactly():
    """The accounting identity the whole contest layer has to satisfy.

    With the real field simulated this is exact rather than a Monte Carlo estimate: every
    prize is awarded to exactly one entry, so summing what the simulator paid must return
    the published pool. A wrong pod size, a dropped band or a double-paid place breaks it.
    """
    rng = np.random.default_rng(0)
    meta = economics.load_metadata().set_index("type")
    for tournament in ("20k_spin_move", "88k_alley_oop"):
        field = int(B.structure(tournament)[0]["field_entries"])
        period = rng.normal(1200, 200, size=(field, 4, 8)).astype(np.float32)
        result = B.simulate_bracket(_entry(period), tournament, rng)
        paid = result["cash"].sum(axis=(0, 1))
        assert np.allclose(paid, float(meta.loc[tournament, "total_prizes"]))


def test_the_round_two_field_is_the_selected_survivors_not_a_fresh_field():
    """Continuation value is priced against entries that already cleared a cut.

    The sharpest one-line form: the mean round-1 score among round-2 survivors has to sit
    well above the field mean. A fresh-field model would leave it at zero.
    """
    rng = np.random.default_rng(0)
    field = int(B.structure("20k_spin_move")[0]["field_entries"])
    period = rng.normal(0, 1, size=(field, 4, 40)).astype(np.float32)
    result = B.simulate_bracket(_entry(period), "20k_spin_move", rng)

    survived = result["reached"][:, 1]
    selected = (period[:, 0, :] * survived).sum(axis=0) / survived.sum(axis=0)
    assert selected.mean() > 1.0                   # ~1.27 sd, the mean of the top sixth
    assert survived.sum(axis=0).mean() == pytest.approx(field / 6)


def test_an_exchangeable_entry_advances_at_the_published_rate_and_earns_minus_rake():
    """The symmetric-field null, end to end, on entries drawn from one distribution.

    Every entry is exchangeable, so each round's reach probability is the chained advance
    rate and the whole bracket is worth exactly `-rake`. This is the test that would have
    caught the tie-break handing our own entries every tie — it was worth +68% on P(reach
    round 4) and turned a -11% ROI into +71%.
    """
    rng = np.random.default_rng(0)
    null = B.symmetric_null("20k_spin_move")
    field = int(B.structure("20k_spin_move")[0]["field_entries"])
    period = rng.normal(1200, 200, size=(field, 4, 400)).astype(np.float32)
    result = B.simulate_bracket(_entry(period), "20k_spin_move", rng)

    for i, expected in enumerate(b["p_reach"] for b in null["by_round"]):
        assert result["reached"][:, i].mean() == pytest.approx(expected, rel=1e-9)
    assert (result["cash"].sum(axis=1).mean()
            == pytest.approx(null["expected_payout"], rel=1e-9))


def test_a_field_too_small_for_the_contest_raises():
    """Better than quietly playing a 216-entry tournament as if it were the 35,280 one."""
    rng = np.random.default_rng(0)
    period = np.zeros((100, 4, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="216"):
        B.simulate_bracket(_entry(period), "88k_alley_oop", rng)


def test_members_seats_named_rows_so_a_benchmark_can_be_substituted():
    """The holdout mechanism: one seat filled by an entry that is not in the field."""
    rng = np.random.default_rng(0)
    field = int(B.structure("88k_alley_oop")[0]["field_entries"])
    period = rng.normal(1200, 200, size=(field + 1, 4, 20)).astype(np.float32)
    period[field] += 2000.0                       # the benchmark, far stronger

    pure = B.simulate_bracket(_entry(period), "88k_alley_oop", rng)
    assert not pure["reached"][field].any()       # row `field` never plays

    seats = np.concatenate([[field], np.arange(1, field)])
    swapped = B.simulate_bracket(_entry(period), "88k_alley_oop", rng, members=seats)
    assert swapped["reached"][field, 0].all()     # now it does
    assert not swapped["reached"][0].any()        # and the row it displaced does not
    assert swapped["reached"][:, 0].sum(axis=0).tolist() == [field] * 20


def test_a_stronger_entry_advances_more_often_than_the_field():
    """The bracket has to discriminate at all, which no identity checks."""
    rng = np.random.default_rng(0)
    field = int(B.structure("20k_spin_move")[0]["field_entries"])
    period = rng.normal(1200, 200, size=(field, 4, 200)).astype(np.float32)
    period[0] += 400.0                                    # two sd better every round
    result = B.simulate_bracket(_entry(period), "20k_spin_move", rng)
    assert result["reached"][0, 1].mean() > 0.5
    assert result["reached"][1:, 1].mean() == pytest.approx(1 / 6, rel=0.01)


def test_the_per_player_tiebreak_is_all_or_nothing():
    """A population with the split must not beat one without it on that level alone.

    This is the bug the null caught, in the shape it took when the field and our entries
    were scored separately. The pods now come from one scored frame, so either every entry
    carries the split or none does — the asymmetry is unrepresentable rather than merely
    fixed, and this pins that `_pod_tiebreak` passes zeros for all when it is absent.
    """
    weekly = np.arange(24, dtype=float).reshape(4, 3, 2)
    members = np.array([[[0, 1, 2, 3]]])
    week, split = B._pod_tiebreak(members, weekly, None, 1)(0)
    assert week.shape == (4, 3)
    assert split.shape == (4, 1) and not split.any()

    players = np.arange(32, dtype=float).reshape(4, 4, 2)
    week, split = B._pod_tiebreak(members, weekly, players, 1)(0)
    assert split.shape == (4, 4) and split.any()


# ── 5. The field builder ──────────────────────────────────────────────────────

def _board(n_g: int = 110, n_f: int = 110, n_c: int = 60) -> tuple[pd.DataFrame, np.ndarray]:
    frame = pd.DataFrame({
        "pos_g": [True] * n_g + [False] * (n_f + n_c),
        "pos_f": [False] * n_g + [True] * n_f + [False] * n_c,
        "pos_c": [False] * (n_g + n_f) + [True] * n_c,
    })
    frame["projection"] = np.arange(len(frame), 0, -1, dtype=float)
    return frame, B.position_masks(frame)


def test_position_masks_pack_the_three_eligibility_columns():
    frame, masks = _board(2, 2, 1)
    assert list(masks) == [G, G, F, F, C]
    frame["pos_f"] = [False, False, True, True, True]        # a dual C/F
    assert B.position_masks(frame)[4] == F | C


def test_the_placeholder_field_drafts_disjoint_rosters_within_a_pod():
    """Scarcity is the part that cannot be skipped — twelve entries consume 192 players."""
    board, masks = _board()
    rosters = B.placeholder_field(board, masks, 24, np.random.default_rng(0))
    assert rosters.shape == (24, B.ROSTER_SIZE)
    for pod in (rosters[:12], rosters[12:]):
        assert len(set(pod.ravel().tolist())) == 12 * B.ROSTER_SIZE


def test_every_drafted_roster_can_seat_seven():
    """Enforced during the draft rather than repaired afterwards."""
    board, masks = _board()
    rosters = B.placeholder_field(board, masks, 60, np.random.default_rng(1))
    assert B.roster_is_legal(masks[rosters]).all()
    for i, need in enumerate(B.SLOT_CAPACITY):
        assert ((masks[rosters] >> i & 1).sum(axis=1) >= need).all()


def test_a_board_too_thin_for_a_pod_raises_rather_than_dealing_a_dead_roster():
    board, masks = _board(n_g=120, n_f=120, n_c=8)   # 8 centers cannot cover 12 entries
    with pytest.raises(ValueError, match="C"):
        B.placeholder_field(board, masks, 12, np.random.default_rng(0))


def test_top_roster_takes_the_best_sixteen_that_can_still_seat_seven():
    board, masks = _board()
    roster = B.top_roster(masks, board["projection"].to_numpy())
    assert len(roster) == B.ROSTER_SIZE
    assert B.roster_is_legal(masks[roster][None, :])[0]
    assert 220 in roster                              # the best center, taken for the slot


# ── 6. The split guard ────────────────────────────────────────────────────────

def test_the_split_guard_refuses_a_test_season(monkeypatch):
    """Pinned the way `component_rates`' guard is, because prose already failed here once.

    The frame matters as much as the guard: `draft_pool.parquet` carries the 2026-27
    production board, so deriving the split from it shifts the window a season forward and
    hands back 2024-25 as "validation". The guard cannot see that — it believes the frame —
    so the seasons come from the component design instead.
    """
    monkeypatch.setattr(held_out, "_unlocked", False, raising=False)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]
    design = pd.DataFrame({"season": seasons, "player_id": range(len(seasons))})

    assert B.validation_seasons(design) == ["2022-23", "2023-24"]
    with pytest.raises(HeldOutLocked):
        B.assert_season_allowed("2025-26", design)
    with unlocked("a test"):
        B.assert_season_allowed("2025-26", design)
    B.assert_season_allowed("2023-24", design)         # validation is always legal


# ── The tensor's preseason stamp and the staleness guard at the read path ────

def _tensor_npz(tmp_path, season="2026-27", **extra):
    tmp_path.mkdir(parents=True, exist_ok=True)
    keys = dict(dk_pts=np.zeros((2, 3, 4), dtype=np.float32),
                player_id=np.array([7, 9]), tournament_round=np.array([1, 1, 2]),
                season=np.array(season), fit_window=np.array("full"),
                n_sims=np.array(4))
    keys.update(extra)
    np.savez(tmp_path / f"sim_tensor_{season}.npz", **keys)
    return tmp_path


def test_load_tensor_surfaces_the_preseason_stamp_and_none_for_legacy(tmp_path):
    """A consumer holding two tensors has no other way to tell the August board apart,
    and a tensor from before the stamp existed must read back as unstamped, not as 0%."""
    _tensor_npz(tmp_path, preseason_coverage=np.array(0.25),
                preseason_log_rows=np.array(0))
    out = B.load_tensor(tmp_path, "2026-27")
    assert out["preseason_coverage"] == pytest.approx(0.25)
    assert out["preseason_log_rows"] == 0

    legacy = B.load_tensor(_tensor_npz(tmp_path / "l", season="2022-23"), "2022-23")
    assert legacy["preseason_coverage"] is None
    assert legacy["preseason_log_rows"] is None


def test_load_tensor_arms_the_staleness_guard_only_when_given_the_raw_dir(tmp_path):
    """Both states at the read path: silent while no preseason log exists, a hard refusal
    the moment one lands beside a tensor stamped as built without it (C6)."""
    features = tmp_path / "features"
    features.mkdir()
    raw = tmp_path / "raw"
    (raw / "nbastats").mkdir(parents=True)
    _tensor_npz(features, preseason_coverage=np.array(0.0),
                preseason_log_rows=np.array(0))

    # August: the log does not exist, and the armed read is silent.
    assert B.load_tensor(features, "2026-27", raw_dir=str(raw))["n_sims"] == 4

    # October: the log lands; the armed read refuses, the unarmed one still loads —
    # the guard belongs to the consumer that names the raw dir (the draft room).
    (raw / "nbastats" / "game_logs_pre_season_2026_27.csv").write_text(
        "GAME_ID,PLAYER_ID,MIN\n1,7,12.0\n2,9,20.0\n")
    with pytest.raises(RuntimeError, match="STALE"):
        B.load_tensor(features, "2026-27", raw_dir=str(raw))
    assert B.load_tensor(features, "2026-27")["n_sims"] == 4
