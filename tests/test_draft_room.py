"""Tests for the draft room — the exchange, the field, and the money.

What is worth pinning in `src/sim/draft_room.py` is, again, the arithmetic whose failures
are **silent**. A wrong exchange threshold still returns a ranked table. A survival
function without its half-tie still returns probabilities, and every advance rate is
quietly a few percent high. A roster completion that never seats a centre still produces
sixteen names, and then prices every centre on the board against a replacement-level
alternative. None of those raises.

Three of these are the load-bearing ones. **The exchange identity** is checked against
`bracket.best_lineup` itself rather than against a remembered number, on single-position
and dual-eligible rosters both, because the whole latency argument rests on it being exact
rather than close. **The symmetric-field null** is checked on a hand-built two-round
tournament whose answer is arithmetic — an exchangeable entry collects the pool divided by
the field — which is the same identity `bracket.symmetric_null` uses one layer up. And
**the plotting position** is checked directly, since it is one term that moves no shape
and every level.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.sim import draft as D
from src.sim import draft_room as R
from src.sim.bracket import ROSTER_SIZE, best_lineup


# ── Synthetic builders ────────────────────────────────────────────────────────

def _pool(positions: str, season: str = "2022-23") -> pd.DataFrame:
    """A draft pool from a string of position letters, ADP ascending."""
    letters = list(positions)
    n = len(letters)
    return pd.DataFrame({
        "season": [season] * n,
        "player_id": np.arange(1, n + 1),
        "player_name": [f"P{i:03d}" for i in range(n)],
        "team": ["BOS"] * n,
        "position": letters,
        "pos_g": [c == "G" for c in letters],
        "pos_f": [c == "F" for c in letters],
        "pos_c": [c == "C" for c in letters],
        "adp": np.arange(1.0, n + 1.0),
        "adp_dk_scale": np.arange(1.0, n + 1.0),
        "prior_min_total": np.linspace(2000.0, 100.0, n),
        "dk_player_id": np.arange(1, n + 1),
    })


def _balanced_pool(n_g: int = 240, n_f: int = 220, n_c: int = 80) -> pd.DataFrame:
    letters: list[str] = []
    while len(letters) < n_g + n_f + n_c:
        for letter, cap in (("G", n_g), ("F", n_f), ("C", n_c)):
            if letters.count(letter) < cap:
                letters.append(letter)
    return _pool("".join(letters))


def _room(pool: pd.DataFrame | None = None, n_periods: int = 4, n_sims: int = 8,
          seed: int = 0, season: str = "2022-23") -> R.Room:
    """A room over a synthetic board — no tensor file, no posterior, no field draft."""
    pool = _balanced_pool() if pool is None else pool
    frame = D.build_board(pool, season)
    board = D.to_arrays(frame, season)
    rng = np.random.default_rng(seed)
    # Descending mean by board rank, so "best available" means something.
    level = np.linspace(40.0, 5.0, len(frame))[:, None, None]
    dk_pts = (level + rng.normal(0.0, 6.0, (len(frame), n_periods, n_sims))
              ).astype(np.float32)
    round_of_period = np.array([1] * (n_periods - 1) + [2])
    return R.Room(season=season, frame=frame, board=board, dk_pts=dk_pts,
                  scorable=np.ones(len(frame), dtype=bool),
                  masks=np.asarray([1 if c == "G" else 2 if c == "F" else 4
                                    for c in frame["position"]], dtype=np.int16),
                  round_of_period=round_of_period,
                  projection=dk_pts.sum(axis=1).mean(axis=1),
                  pod_size=12, field_cfg=D.FieldConfig(rank_noise_sd=0.0),
                  field_round=np.zeros((1, 2, n_sims), dtype=np.float32),
                  refs={}, fit_window="train")


def _rounds(*specs) -> list[dict]:
    """A tournament structure by hand: `(pod_size, n_advance, payout list)` per round."""
    return [{"round": i + 1, "pod_size": pod, "n_advance": advance,
             "payout": np.asarray(payout, dtype=float)}
            for i, (pod, advance, payout) in enumerate(specs)]


# ── 1. The exchange — exact, or the whole latency argument is wrong ───────────

def test_the_exchange_reproduces_best_lineup_on_single_position_rosters():
    """`base + max(0, score - threshold[mask])` IS `best_lineup(base + candidate)`.

    Checked against `bracket.best_lineup` itself over random rosters and random
    newcomers, because the claim is an identity rather than an approximation: if it were
    only close, the draft room would be ranking candidates on a number no other part of
    this project computes.
    """
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(7, 17))
        scores = rng.normal(30.0, 15.0, size=n)
        masks = rng.choice([1, 2, 4], size=n).astype(np.int16)
        masks[:2], masks[2:4], masks[4] = 1, 2, 4          # can seat seven
        base, threshold = R.lineup_state(scores, masks)
        for _ in range(5):
            score, mask = rng.normal(30.0, 25.0), int(rng.choice([1, 2, 4]))
            want, _ = best_lineup(np.append(scores, score),
                                  np.append(masks, mask).astype(np.int16))
            got = float(base) + max(0.0, score - threshold[mask])
            assert got == pytest.approx(float(want), abs=1e-9)


def test_the_exchange_reproduces_best_lineup_with_dual_eligibility():
    """The rejected dual convention has to work too, or a sensitivity run is a rewrite.

    `make draft-pool` ships one letter per player and carries `dual_g` / `dual_f` /
    `dual_c` alongside, so swapping conventions is a column swap. A threshold that only
    handles one-hot masks would make that swap silently wrong: with duals the man a
    newcomer displaces is no longer "the cheapest starter of his own position", and
    Hall's condition is the only thing that knows it.
    """
    rng = np.random.default_rng(1)
    for _ in range(200):
        n = int(rng.integers(8, 17))
        scores = rng.normal(30.0, 15.0, size=n)
        masks = rng.choice([1, 2, 4, 3, 5, 6, 7], size=n).astype(np.int16)
        masks[:2], masks[2:4], masks[4] = 1, 2, 4
        base, threshold = R.lineup_state(scores, masks)
        for _ in range(5):
            score = float(rng.normal(30.0, 25.0))
            mask = int(rng.choice([1, 2, 4, 3, 5, 6, 7]))
            want, _ = best_lineup(np.append(scores, score),
                                  np.append(masks, mask).astype(np.int16))
            got = float(base) + max(0.0, score - threshold[mask])
            assert got == pytest.approx(float(want), abs=1e-9)


def test_the_exchange_is_vectorized_over_periods_and_sims():
    """One threshold per (period, sim) cell, and each cell is its own lineup."""
    rng = np.random.default_rng(2)
    scores = rng.normal(30.0, 12.0, size=(5, 7, 9))
    masks = np.broadcast_to(np.array([1, 1, 2, 2, 4, 1, 2, 4, 1], dtype=np.int16),
                            scores.shape)
    base, threshold = R.lineup_state(scores, masks)
    assert base.shape == (5, 7)
    assert threshold.shape == (5, 7, 8)
    for period in (0, 3):
        for sim in (2, 6):
            cell, cell_threshold = R.lineup_state(scores[period, sim],
                                                  masks[period, sim])
            assert float(cell) == pytest.approx(float(base[period, sim]))
            for mask in (1, 2, 4):
                assert (float(cell_threshold[mask])
                        == pytest.approx(float(threshold[period, sim, mask])))


def test_a_roster_that_cannot_seat_seven_raises_instead_of_pricing_against_six():
    """The failure this guards is a *low* threshold, which reads as a bargain."""
    scores = np.arange(10.0, 0.0, -1.0)
    masks = np.full(10, 1, dtype=np.int16)                 # ten guards, no forward, no C
    with pytest.raises(ValueError, match="seat seven"):
        R.lineup_state(scores, masks)


# ── 2. The survival function, and the term that moves every level ────────────

def test_survival_uses_the_midpoint_plotting_position():
    """The k-th smallest of n equally weighted entries sits at `(n - k - 0.5) / n`.

    Counting only strictly-greater entries would put it at `(n - k - 1) / n`, which is
    the right-endpoint Riemann sum whose bias `null_check` reads directly.
    """
    totals = np.arange(10.0)[:, None]
    weights = np.ones((10, 1))
    srt, cum = R._row_sorted(totals, weights)
    q = R.survival(srt, cum, totals)
    assert q[:, 0] == pytest.approx((10 - np.arange(10) - 0.5) / 10)


def test_survival_splits_a_tie_down_the_middle():
    """A tie is a coin flip in the rules' cascade, so it is half a beat here."""
    totals = np.array([[5.0], [5.0], [5.0], [5.0]])
    srt, cum = R._row_sorted(totals, np.ones((4, 1)))
    assert float(R.survival(srt, cum, np.array([[5.0]]))[0, 0]) == pytest.approx(0.5)


def test_survival_reads_each_sim_against_its_own_field():
    """A good week for the league is a good week for everyone, so `q` is within-sim.

    Two sims with disjoint ranges: an entry that is the best in its own sim must read the
    same `q` in both, however far apart the two sims' levels are. Pooling across sims
    would make the high-scoring sim's field beat the low-scoring sim's entry every time.
    """
    totals = np.stack([np.arange(5.0), np.arange(5.0) + 1000.0], axis=1)
    srt, cum = R._row_sorted(totals, np.ones((5, 2)))
    q = R.survival(srt, cum, totals)
    assert q[:, 0] == pytest.approx(q[:, 1])


# ── 3. The place distribution ─────────────────────────────────────────────────

def test_the_place_distribution_is_a_proper_binomial():
    """Every seat is somewhere, so the pod's place probabilities sum to one."""
    q = np.linspace(0.01, 0.99, 21)[:, None]
    everywhere, advance = R.round_outcome(q, 12, np.ones(12), 2)
    assert everywhere.ravel() == pytest.approx(np.ones(21), abs=1e-9)
    top_two = (1 - q) ** 11 + 11 * q * (1 - q) ** 10
    assert advance.ravel() == pytest.approx(top_two.ravel(), abs=1e-9)


def test_the_final_round_stays_inside_float64():
    """A 49-seat table is 48 binomial terms, and the recurrence starts at `(1-q)**48`."""
    q = np.array([[R.Q_FLOOR, 0.5, 1.0 - R.Q_FLOOR]])
    total, _ = R.round_outcome(q, 49, np.ones(49), 0)
    assert np.isfinite(total).all()
    assert total.ravel() == pytest.approx(np.ones(3), abs=1e-6)


# ── 4. The money — an exchangeable entry earns the pool over the field ────────

def test_an_exchangeable_entry_collects_the_pool_divided_by_the_field():
    """The symmetric-field null, one layer down from `bracket.symmetric_null`.

    Two rounds, four-seat pods, two advancing, and the whole prize paid to the winner of
    round two. Every entry is drawn from one distribution, so each reaches round two with
    probability 1/2 and, having got there, collects $100 / 4 — so the entry is worth
    $12.50, which is `bracket.symmetric_null`'s `prize_sum / pod_size * reach` summed over
    the rounds and is arithmetic rather than a fit. It fails if the survivor reweighting
    is wrong, if the plotting position is wrong, if the place distribution is wrong, or if
    the payout is indexed off by one.
    """
    rng = np.random.default_rng(3)
    field = rng.normal(100.0, 10.0, size=(2000, 2, 40)).astype(np.float32)
    rounds = _rounds((4, 2, [0.0, 0.0, 0.0, 0.0]), (4, 0, [100.0, 0.0, 0.0, 0.0]))
    ref = R.field_reference(field, "synthetic", rounds=rounds, entry_fee=10.0)
    ev, advance = R.bracket_ev(field.astype(np.float64), ref)
    assert float(advance.mean()) == pytest.approx(0.5, abs=2e-3)
    assert float(ev.mean()) == pytest.approx(0.5 * 100.0 / 4, rel=2e-2)


def test_rounds_after_the_first_face_survivors_and_not_a_fresh_field():
    """The round-2 population is a reweighted field, and the reweighting has to bite.

    Scoring rounds independently against a fresh field is `bracket.py`'s named
    overstatement of continuation value, so the check is that round 2's population is a
    strict, large reduction of round 1's. `effective_entries` is Kish's — `(Σw)² / Σw²` —
    which is bounded below by the expected survivor count `n / pod_size` and well below
    the field, and is the honest measure of how much field is left to resolve a round
    with.
    """
    rng = np.random.default_rng(4)
    field = rng.normal(0.0, 1.0, size=(3000, 2, 20)).astype(np.float32)
    rounds = _rounds((6, 1, [0.0] * 6), (6, 0, [60.0] + [0.0] * 5))
    ref = R.field_reference(field, "synthetic", rounds=rounds, entry_fee=1.0)
    assert ref.effective_entries[0] == pytest.approx(3000.0, rel=1e-6)
    assert 3000 / 6 <= ref.effective_entries[1] <= 3000 * 0.4


# ── 5. The completion, which is the baseline every candidate is measured against ──

def test_the_completion_fills_the_starting_slate_before_anything_else():
    """2 G / 2 F / 1 C first, then best available — `bracket.top_roster`'s convention.

    Deferring the slate to the last forced pick leaves the base roster with a
    replacement-level centre and prices every centre on the board against it.
    """
    room = _room()
    state = room.new_state()
    completion = R.plan_completion(room, state, 0, 15)
    positions = [room.frame["position"].iloc[p] for p in completion]
    assert positions.count("G") >= 2 and positions.count("F") >= 2
    assert "C" in positions[:5]


def test_the_completion_never_names_a_player_already_gone():
    room = _room()
    state = room.new_state()
    for _ in range(30):
        seat = R.seat_on_clock(state)
        allowed = D.legal_mask(state, seat, room.field_cfg)[0]
        R.mark_pick(state, int(np.argmax(np.where(allowed, -room.board.rank, -np.inf))))
    completion = R.plan_completion(room, state, 0, 12)
    assert not state.taken[0][completion].any()
    assert len(set(completion.tolist())) == len(completion)


def test_the_completed_base_can_always_seat_seven():
    """The precondition `lineup_state` refuses to guess at, held across a whole draft."""
    room = _room()
    state = room.new_state()
    while state.pick_index < room.pod_size * D.N_ROUNDS:
        seat = R.seat_on_clock(state)
        if seat == 0 and len(state.roster_of(0)) < ROSTER_SIZE:
            held = state.roster_of(0)
            n_future = ROSTER_SIZE - len(held) - 1
            base = np.concatenate([held, R.plan_completion(room, state, 0, n_future)])
            assert len(base) == 15
            block = np.moveaxis(room.dk_pts[base.astype(int)], 0, -1)
            R.lineup_state(block, room.masks[base.astype(int)])       # raises if it cannot
        allowed = D.legal_mask(state, seat, room.field_cfg)[0]
        R.mark_pick(state, int(np.argmax(np.where(allowed, -room.board.rank, -np.inf))))


# ── 6. The state — one click, and the snake says the rest ────────────────────

def test_one_click_advances_the_snake_and_never_asks_whose_pick_it_was():
    room = _room()
    state = room.new_state()
    seats = []
    for player in range(24):
        seats.append(R.seat_on_clock(state))
        R.mark_pick(state, player)
    assert seats[:12] == list(range(12))
    assert seats[12:] == list(range(11, -1, -1))
    assert list(state.roster_of(0)) == [0, 23]


def test_replay_rebuilds_the_state_from_the_pick_log():
    """How the page undoes a misclick — the log is the state, the state is derived."""
    room = _room()
    picks = [5, 9, 2, 11, 30, 31]
    full = R.replay(room, picks)
    short = R.replay(room, picks[:-1])
    assert full.pick_index == len(picks)
    assert short.pick_index == len(picks) - 1
    assert full.taken[0].sum() == len(picks)
    assert not short.taken[0][picks[-1]]


def test_roster_health_separates_dks_caps_from_the_starting_slate():
    """8 G / 8 F / 3 C are maxima that bind autodraft; 2 G / 2 F / 1 C is a minimum.

    A roster can sit inside every cap and still seat nobody at centre, which is the whole
    reason both are on screen.
    """
    room = _room()
    state = room.new_state()
    guards = [i for i in range(len(room.frame))
              if room.frame["position"].iloc[i] == "G"][:3]
    for i, player in enumerate(guards):
        D.apply_pick(state, 0, np.array([player]))
        state.pick_index += room.pod_size - 1
    health = R.roster_health(room, state, 0)
    assert health["counts"]["G"] == 3 and health["owed"]["G"] == 0
    assert health["owed"]["F"] == 2 and health["owed"]["C"] == 1
    assert health["at_capacity"] == []
    assert health["at_risk"] is False
    assert health["seats_seven"] is False


def test_at_risk_fires_while_there_is_still_a_choice_about_how_to_fix_it():
    """Owed slots equal to picks left is the last round where the pick is not forced."""
    room = _room()
    state = room.new_state()
    forwards = [i for i in range(len(room.frame))
                if room.frame["position"].iloc[i] == "F"]
    for player in forwards[:ROSTER_SIZE - 3]:
        D.apply_pick(state, 0, np.array([player]))
        state.pick_index += room.pod_size - 1
    health = R.roster_health(room, state, 0)
    assert health["picks_left"] == 3
    assert sum(health["owed"].values()) == 3            # 2 G and 1 C still owed
    assert health["at_risk"] is True


# ── 7. The recommendation ─────────────────────────────────────────────────────

def test_a_player_the_tensor_cannot_price_is_excluded_rather_than_valued_at_zero():
    """Zero is a number, and a number this ranking would act on.

    `make simulate-season` scores 386 of 539 rostered players; the rest have no
    component-head design row and stay draftable *by the field*. Ranking them at zero
    would put them last, which is a claim; excluding them says nothing.
    """
    room = _room()
    room.scorable[::3] = False
    room.refs["synthetic"] = R.field_reference(
        np.zeros((50, 2, room.n_sims), dtype=np.float32), "synthetic",
        rounds=_rounds((12, 2, [0.0] * 12), (12, 0, [50.0] + [0.0] * 11)), entry_fee=5.0)
    table, context = R.evaluate(room, room.new_state(), 0, "synthetic", top=400)
    assert context["n_candidates"] == int(room.scorable.sum())
    assert room.scorable[table["player_id"].to_numpy() - 1].all()


def test_the_recommendation_never_names_a_player_who_is_gone():
    room = _room()
    room.refs["synthetic"] = R.field_reference(
        np.zeros((50, 2, room.n_sims), dtype=np.float32), "synthetic",
        rounds=_rounds((12, 2, [0.0] * 12), (12, 0, [50.0] + [0.0] * 11)), entry_fee=5.0)
    state = room.new_state()
    for player in range(40):
        R.mark_pick(state, player)
    table, _ = R.evaluate(room, state, 0, "synthetic", top=50)
    taken = np.nonzero(state.taken[0])[0]
    assert not set(table["player_id"] - 1) & set(taken.tolist())


def test_the_ranking_in_the_context_is_the_whole_board_not_the_displayed_slice():
    """`pick_log` has to price the player actually taken, including an override.

    A drafter who ignores the recommendation is the case the log exists to record, and he
    is exactly the case a top-N slice would drop.
    """
    room = _room()
    room.refs["synthetic"] = R.field_reference(
        np.zeros((50, 2, room.n_sims), dtype=np.float32), "synthetic",
        rounds=_rounds((12, 2, [0.0] * 12), (12, 0, [50.0] + [0.0] * 11)), entry_fee=5.0)
    table, context = R.evaluate(room, room.new_state(), 0, "synthetic", top=5)
    assert len(table) == 5
    assert len(context["ranking"]) == context["n_candidates"]
    assert list(context["ranking"].columns) == ["player_id", "player_name", "value",
                                                "cost_vs_best"]


# ── 8. The pick log — a draft is the one artifact that cannot be regenerated ──

def test_the_pick_log_replays_the_draft_it_recorded():
    """`board_index` is the round trip: the log rebuilds the state it was written from."""
    room = _room()
    picks = [4, 9, 1, 30, 12, 7]
    log = R.pick_log(room, picks, seat=0, tournament="synthetic")
    assert list(log["board_index"]) == picks
    assert list(log["pick"]) == [1, 2, 3, 4, 5, 6]
    assert list(log["seat"]) == [1, 2, 3, 4, 5, 6]
    assert list(log["ours"]) == [True, False, False, False, False, False]
    rebuilt = R.replay(room, list(log["board_index"]))
    assert rebuilt.pick_index == len(picks)
    assert list(rebuilt.roster_of(0)) == [4]


def test_the_pick_log_records_what_the_room_advised_beside_what_was_taken():
    """`cost_vs_best` is what overriding the recommendation cost, in the ranked unit."""
    room = _room()
    annotations = [{"recommended": "P000", "recommended_value": 10.0,
                    "taken_value": 10.0, "followed": True, "recompute_ms": 120.0},
                   {"recommended": "P001", "recommended_value": 9.0,
                    "taken_value": 4.0, "followed": False, "recompute_ms": 118.0}]
    log = R.pick_log(room, [0, 1], seat=0, annotations=annotations)
    assert log["cost_vs_best"].tolist() == [0.0, -5.0]
    assert log["recommended"].tolist() == ["P000", "P001"]


def test_followed_is_null_on_an_opponents_row_rather_than_false():
    """The same columns answer a different question when the pick was not ours.

    On an opponent's row the ranking is still ours, so it says what the field took priced
    against what our board wanted — which is worth keeping. It is not us deviating from
    advice, and a `False` there would read as exactly that.
    """
    room = _room()
    annotations = [{"recommended": "P000", "recommended_value": 10.0,
                    "taken_value": 10.0, "followed": True},
                   {"recommended": "P001", "recommended_value": 9.0,
                    "taken_value": 4.0, "followed": False}]
    log = R.pick_log(room, [0, 1], seat=0, annotations=annotations)
    assert log["followed"].tolist()[0] is True
    assert log["followed"].isna().tolist() == [False, True]
    assert log["taken_value"].notna().all()          # the value is kept either way


def test_a_pick_made_with_no_recommendation_on_screen_is_null_and_not_zero():
    room = _room()
    log = R.pick_log(room, [0, 1, 2], seat=0)
    assert log["recommended"].isna().all()
    assert log["cost_vs_best"].isna().all()
    assert log["player_name"].notna().all()          # the pick itself is still recorded


def test_the_log_keeps_its_column_order_so_a_saved_file_stays_readable(tmp_path):
    room = _room()
    log = R.pick_log(room, [3, 8], seat=0, tournament="synthetic",
                     objective="bracket_ev")
    assert tuple(log.columns) == R.LOG_COLUMNS
    dest = R.save_pick_log(log, R.log_path(tmp_path / "draft_logs", "2022-23", "s1"))
    assert dest.exists()
    assert tuple(pd.read_csv(dest).columns) == R.LOG_COLUMNS


# ── 9. Injury notes — display-only, and the one name join on this page ───────

def _injury_dirs(tmp_path, espn_rows: list[dict] | None = None,
                 nba_rows: list[dict] | None = None):
    """A raw directory with one snapshot from each feed, in each feed's own schema."""
    espn_dir = tmp_path / "injuries"
    espn_dir.mkdir(parents=True, exist_ok=True)
    espn = pd.DataFrame(espn_rows or [], columns=[
        "snapshot_date", "scraped_at", "team", "player", "position", "status",
        "injury_date", "injury_type", "injury_location", "injury_detail", "injury_side",
        "return_date_forecast", "fantasy_status", "short_comment", "long_comment"])
    espn.to_csv(espn_dir / "injuries_log.csv", index=False)

    nba_dir = tmp_path / "injury_reports"
    nba_dir.mkdir(parents=True, exist_ok=True)
    nba = pd.DataFrame(nba_rows or [], columns=[
        "report_date", "report_time", "published_at", "game_date", "game_time", "matchup",
        "team", "player_name", "status", "reason", "reason_category", "body_part",
        "reason_detail", "source_file"])
    nba.to_csv(nba_dir / "injury_reports_log.csv", index=False)
    return tmp_path


def test_only_the_newest_snapshot_of_each_feed_is_read(tmp_path):
    """A stale snapshot beside a fresh one silently tells a drafter a player recovered."""
    raw = _injury_dirs(tmp_path, espn_rows=[
        {"snapshot_date": "2026-07-01", "player": "Nic Claxton", "team": "BKN",
         "status": "Out", "injury_type": "Knee"},
        {"snapshot_date": "2026-08-03", "player": "Nic Claxton", "team": "BKN",
         "status": "Day-To-Day", "injury_type": "Knee"}])
    notes = R.load_injury_notes(raw, today=date(2026, 8, 9), sources=("espn",))
    assert len(notes) == 1
    assert notes["status"].iloc[0] == "Day-To-Day"
    assert notes["as_of"].iloc[0] == "2026-08-03"
    assert int(notes["age_days"].iloc[0]) == 6


def test_a_placeholder_report_row_is_not_a_players_status(tmp_path):
    """NBA reports carry "NOT YET SUBMITTED" for teams that have not filed.

    Rendering one as a status would put a fabricated designation on a real player's row,
    and the row it lands on is whichever name the blank sorts next to.
    """
    raw = _injury_dirs(tmp_path, nba_rows=[
        {"report_date": "2026-01-02", "game_date": "01/02/2026", "team": "Brooklyn Nets",
         "player_name": "Claxton, Nic", "status": "Out", "reason": "Injury - Knee"},
        {"report_date": "2026-07-19", "game_date": "07/19/2026", "team": "Brooklyn Nets",
         "player_name": "", "status": "NOT YET SUBMITTED", "reason": ""}])
    notes = R.load_injury_notes(raw, today=date(2026, 8, 9), sources=("nba_report",))
    assert len(notes) == 1
    assert notes["player"].iloc[0] == "Claxton, Nic"
    assert notes["as_of"].iloc[0] == "2026-01-02"        # the last report naming anybody


def test_the_pdfs_last_comma_first_form_matches_the_boards_first_last(tmp_path):
    """`"Claxton, Nic"` and `Nic Claxton` are the same player, and only the key knows it."""
    room = _room(_pool("GGFFC"))
    room.frame.loc[:, "player_name"] = ["Nic Claxton", "P001", "P002", "P003", "P004"]
    raw = _injury_dirs(tmp_path, nba_rows=[
        {"report_date": "2026-01-02", "game_date": "01/02/2026", "team": "Brooklyn Nets",
         "player_name": "Claxton, Nic", "status": "Out", "reason": "Injury - Knee"}])
    notes = R.load_injury_notes(raw, today=date(2026, 1, 3), sources=("nba_report",))
    matched, coverage = R.match_injuries(room, notes)
    assert coverage["matched"] == 1
    assert matched["board_index"].iloc[0] == 0
    assert R.injury_badges(matched)[0] == "OUT"


def test_an_ambiguous_name_is_attached_to_nobody(tmp_path):
    """Neither feed carries a player id, so uniqueness on both sides is the second guard.

    A wrong note is worse than no note: a drafter who passes on a healthy star because the
    room labelled him Out has lost the pick, and no downstream number would ever show it.
    """
    room = _room(_pool("GGFFC"))
    # The same normalized key on two board rows — `Gary Payton` and `Gary Payton II`
    # collapse to it once the generational suffix is stripped.
    room.frame.loc[:, "player_name"] = ["Gary Payton", "Gary Payton II", "P002", "P003",
                                        "P004"]
    raw = _injury_dirs(tmp_path, espn_rows=[
        {"snapshot_date": "2026-08-03", "player": "Gary Payton", "team": "GSW",
         "status": "Out", "injury_type": "Knee"}])
    notes = R.load_injury_notes(raw, today=date(2026, 8, 9), sources=("espn",))
    matched, coverage = R.match_injuries(room, notes)
    assert coverage["ambiguous"] == 1
    assert coverage["matched"] == 0
    assert matched.empty
    assert R.injury_badges(matched) == {}


def test_the_worst_status_wins_when_the_two_feeds_disagree(tmp_path):
    """A room that softens a disagreement is making a call it has no basis for."""
    room = _room(_pool("GGFFC"))
    room.frame.loc[:, "player_name"] = ["Nic Claxton", "P001", "P002", "P003", "P004"]
    raw = _injury_dirs(
        tmp_path,
        espn_rows=[{"snapshot_date": "2026-01-02", "player": "Nic Claxton", "team": "BKN",
                    "status": "Day-To-Day", "injury_type": "Knee"}],
        nba_rows=[{"report_date": "2026-01-02", "game_date": "01/02/2026",
                   "team": "Brooklyn Nets", "player_name": "Claxton, Nic",
                   "status": "Out", "reason": "Injury - Knee"}])
    notes = R.load_injury_notes(raw, today=date(2026, 1, 3))
    matched, coverage = R.match_injuries(room, notes)
    assert coverage["matched"] == 2 and coverage["players_flagged"] == 1
    assert R.injury_badges(matched)[0] == "OUT"
    assert set(matched["source"]) == {"espn", "nba_report"}


def test_a_feed_that_describes_another_season_is_reported_as_one(tmp_path):
    """The feeds describe *today*, and the board may be a practice season.

    That is the one way this can mislead rather than merely be stale: on a 2023-24 board an
    August 2026 snapshot is not out of date, it is about other people's other season.
    """
    room = _room()
    raw = _injury_dirs(tmp_path, espn_rows=[
        {"snapshot_date": "2026-08-03", "player": "P001", "team": "BOS",
         "status": "Out", "injury_type": "Knee"}])
    notes = R.load_injury_notes(raw, today=date(2026, 8, 9), sources=("espn",))
    assert R.injury_relevance(room, notes)["describes_this_season"] is False
    assert R.injury_relevance(room, notes)["off_season_years"] == [2026]

    room.season = "2026-27"
    assert R.injury_relevance(room, notes)["describes_this_season"] is True


def test_no_injury_capture_on_disk_is_an_empty_frame_and_not_a_crash(tmp_path):
    notes = R.load_injury_notes(tmp_path, today=date(2026, 8, 9))
    assert notes.empty
    matched, coverage = R.match_injuries(_room(), notes)
    assert matched.empty and coverage["matched"] == 0
    assert R.injury_sources(notes).empty


def test_a_current_status_feed_never_reaches_the_ranking_or_the_pick_log():
    """The discipline guard, pinned in the only place it can be.

    `point-in-time-discipline`: the ESPN feed and the NBA report describe **today**, so on
    a backtest board today's status *is* the resolved outcome. Folding either into a value
    would be leakage that no split guard can see, because the guards sit on frames rather
    than on displayed text. So the notes are attached for a human to read and the value
    path never sees them — this pins that a later edit cannot quietly fold them in.
    """
    room = _room()
    room.refs["synthetic"] = R.field_reference(
        np.zeros((50, 2, room.n_sims), dtype=np.float32), "synthetic",
        rounds=_rounds((12, 2, [0.0] * 12), (12, 0, [50.0] + [0.0] * 11)), entry_fee=5.0)
    table, context = R.evaluate(room, room.new_state(), 0, "synthetic", top=5)
    leaked = set(R.INJURY_COLUMNS) - {"player", "team", "status"}
    assert not leaked & set(table.columns)
    assert not leaked & set(context["ranking"].columns)
    assert not leaked & set(R.LOG_COLUMNS)


def test_an_unregistered_objective_raises_rather_than_silently_ranking_by_ev():
    room = _room()
    with pytest.raises(KeyError, match="unknown objective"):
        R.evaluate(room, room.new_state(), 0, "synthetic", objective="vibes")


def test_a_candidate_the_completion_already_claims_ties_and_breaks_on_cushion():
    """The exchange says nothing about a player who is already in the base roster.

    `evaluate` rebuilds his sixteen with the spare standing in and scores it through
    `bracket.best_lineup`, and every one of them lands on the *same* sixteen — if the
    completion says we get him at a later pick anyway, taking him now buys the roster we
    already had plus the spare, whoever he is. So they tie by construction, and without
    the fallback they would instead price at zero lift and sink below every scrub on the
    board. The tie is broken by `rank_cushion`: among players worth the same, take the
    one the field is likeliest to take first.
    """
    room = _room()
    room.refs["synthetic"] = R.field_reference(
        np.zeros((50, 2, room.n_sims), dtype=np.float32), "synthetic",
        rounds=_rounds((12, 2, [0.0] * 12), (12, 0, [50.0] + [0.0] * 11)), entry_fee=5.0)
    table, context = R.evaluate(room, room.new_state(), 0, "synthetic",
                                objective="lineup_value", top=400)
    inside = table[table["in_completion"]]
    assert len(inside) == len(context["completion"])
    assert (inside["lineup_value"] > 0).all()
    assert inside["lineup_value"].nunique() == 1
    assert list(inside["rank_cushion"]) == sorted(inside["rank_cushion"])


def test_the_field_cache_is_keyed_by_composition_and_need_weight(tmp_path):
    """A cached pure-ADP field must not be served to a need-aware room, or vice versa."""
    path = tmp_path / "field.npz"
    field = np.zeros((4, 2, 3), dtype=np.float32)
    cfg = D.FieldConfig(noise_model="tiered", rank_noise_sd=2.0)
    R.save_field(path, field, "2022-23", "train", cfg, 1, ["adp"] * 12)
    assert R.load_field(path, cfg, 3, ["adp"] * 12) is not None
    # The key is shares rather than counts, so the engine's all-ADP default matches an
    # explicit twelve-seat all-ADP list — and the caches written before the field carried
    # a composition at all.
    assert R.load_field(path, cfg, 3, None) is not None
    assert R.load_field(path, cfg, 3, ["adp_need"] * 12) is None
    assert R.load_field(path, replace(cfg, need_weight=8.0), 3, ["adp"] * 12) is None


# ── The tensor fingerprint on the field cache, and the production admission ──

def test_the_field_cache_is_keyed_by_the_tensor_it_was_scored_on(tmp_path):
    """A rebuild at the same filename was invisible to the key — the two validation
    caches on disk were older than the tensors they were read against, and draft night
    priced our entry against a field drafted off a board that no longer existed."""
    path = tmp_path / "field.npz"
    field = np.zeros((4, 2, 3), dtype=np.float32)
    cfg = D.FieldConfig(noise_model="tiered", rank_noise_sd=2.0)
    R.save_field(path, field, "2022-23", "train", cfg, 1, ["adp"] * 12,
                 tensor_fingerprint="aaaa")
    assert R.load_field(path, cfg, 3, ["adp"] * 12,
                        tensor_fingerprint="aaaa") is not None
    assert R.load_field(path, cfg, 3, ["adp"] * 12,
                        tensor_fingerprint="bbbb") is None

    # A legacy cache carries no fingerprint, and a caller who names the tensor —
    # `load_room` always does — must never be served one.
    R.save_field(path, field, "2022-23", "train", cfg, 1, ["adp"] * 12)
    assert R.load_field(path, cfg, 3, ["adp"] * 12,
                        tensor_fingerprint="aaaa") is None
    assert R.load_field(path, cfg, 3, ["adp"] * 12) is not None


def test_the_tensor_fingerprint_tracks_content_not_name_or_mtime(tmp_path):
    a = tmp_path / "sim_tensor_x.npz"
    a.write_bytes(b"board one")
    b = tmp_path / "sim_tensor_y.npz"
    b.write_bytes(b"board one")
    assert R.tensor_fingerprint(a) == R.tensor_fingerprint(b)
    b.write_bytes(b"board two")
    assert R.tensor_fingerprint(a) != R.tensor_fingerprint(b)


def test_a_production_tensor_admits_its_own_season_on_its_window(monkeypatch):
    """`fit_window == "full"` is evidence only the production unlock can have written,
    and the split frame is not even built to check it — the season it names has no rows
    there. Every other tensor still faces the split guard, with the wrong seasons
    refused."""
    from src.models import held_out
    from src.models.held_out import HeldOutLocked

    def _boom():
        raise AssertionError("the split frame must not be built for a production tensor")

    R.season_admissible({"fit_window": "full", "season": "2026-27"}, _boom)

    monkeypatch.setattr(held_out, "_unlocked", False, raising=False)
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(15, 25)]
    design = pd.DataFrame({"season": np.repeat(seasons, 2),
                           "player_id": np.tile([1, 2], len(seasons))})
    with pytest.raises(HeldOutLocked):
        R.season_admissible({"fit_window": "train_val", "season": seasons[-1]},
                            lambda: design)
    R.season_admissible({"fit_window": "train", "season": seasons[-3]}, lambda: design)


def test_the_room_lists_season_boards_only_never_labelled_variants(tmp_path):
    """`sim_tensor_2022-23_rookieinclusive.npz` is a measurement, not a board — globbed
    naively its label parses as a 'season' the room then fails to open."""
    from dashboard.draft_room import seasons_with_a_tensor

    for name in ("sim_tensor_2022-23.npz", "sim_tensor_2022-23_rookieinclusive.npz",
                 "sim_tensor_2026-27.npz", "sim_tensor_notes.txt"):
        (tmp_path / name).write_bytes(b"")
    assert seasons_with_a_tensor(tmp_path) == ["2022-23", "2026-27"]
