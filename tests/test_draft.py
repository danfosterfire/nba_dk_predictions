"""Tests for the draft — the engine, DK's autodraft logic, and both modes.

What is worth pinning in `src/sim/draft.py` is the arithmetic whose failures are **silent**.
A draft that quietly ignores a position cap still produces sixteen players. An autodraft that
takes an excluded player still produces a pick. A drafter whose board is redrawn every round
still produces a roster, and a slightly narrower one. Noise applied to the recalibrated ADP
*value* instead of the rank still produces a plausible field, with fifty-five players
exchangeable. None of those raises, and none of them shows up in a mean.

Every test here is arithmetic over synthetic builders — no sim tensor, no posterior artifact
and no sampler — so the file runs in seconds. The two that read `draft_pool.parquet` are the
ones whose whole claim is about the real board.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

import numpy as np
import pandas as pd
import pytest

from src.sim import draft as D
from src.sim.bracket import POSITIONS, ROSTER_SIZE, roster_is_legal


# ── Synthetic builders ────────────────────────────────────────────────────────

def _pool(positions: str, adp: list[float] | None = None,
          season: str = "2022-23") -> pd.DataFrame:
    """A draft pool from a string of position letters, ADP ascending unless given."""
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
        "adp": np.arange(1.0, n + 1.0) if adp is None else np.asarray(adp, dtype=float),
        "adp_dk_scale": (np.arange(1.0, n + 1.0) if adp is None
                         else np.asarray(adp, dtype=float)),
        "prior_min_total": np.linspace(2000.0, 100.0, n),
        "dk_player_id": np.arange(1, n + 1),
    })


def _balanced_pool(n_g: int = 240, n_f: int = 220, n_c: int = 80) -> pd.DataFrame:
    """A pool big enough for a real twelve-entry draft, positions interleaved by rank."""
    letters = []
    while len(letters) < n_g + n_f + n_c:
        for letter, remaining in (("G", n_g), ("F", n_f), ("C", n_c)):
            taken = letters.count(letter)
            if taken < remaining:
                letters.append(letter)
    return _pool("".join(letters))


def _board(pool: pd.DataFrame, season: str = "2022-23"):
    frame = D.build_board(pool, season)
    return frame, D.to_arrays(frame, season)


def _cfg(**kwargs) -> D.FieldConfig:
    return D.FieldConfig(noise_model="constant", rank_noise_sd=0.0, **kwargs)


# ── 1. The board and its ordering ─────────────────────────────────────────────

def test_the_board_ranks_adp_players_ahead_of_everyone_else():
    """A player with no ADP is behind every player who has one, however good he was."""
    pool = _pool("GGFFC", adp=[3.0, np.nan, 1.0, np.nan, 2.0])
    frame, board = _board(pool)
    assert list(frame["adp_dk_scale"].fillna(-1)) == [1.0, 2.0, 3.0, -1.0, -1.0]
    assert list(board.rank) == [0, 1, 2, 3, 4]


def test_unranked_players_are_ordered_by_prior_minutes_and_never_by_a_projection():
    """The fallback ordering is point-in-time and model-free — the field gets no forecast."""
    pool = _pool("GGG", adp=[np.nan] * 3)
    pool["prior_min_total"] = [100.0, 2500.0, 900.0]
    frame, _ = _board(pool)
    assert list(frame["player_name"]) == ["P001", "P002", "P000"]


def test_a_monotone_recalibration_cannot_reorder_the_board():
    """`adp_dk_scale` is `adp` through an isotonic map, so it induces the same order.

    That is the reason noise goes on the rank rather than on the recalibrated value: the
    map's plateaus make many players share a value, and noise on a shared value makes them
    exchangeable rather than merely close.
    """
    pool = _pool("GFCGF", adp=[1.0, 4.0, 9.0, 16.0, 25.0])
    pool["adp_dk_scale"] = [2.0, 5.0, 5.0, 5.0, 40.0]      # a three-wide plateau
    frame, _ = _board(pool)
    assert list(frame["adp"]) == [1.0, 4.0, 9.0, 16.0, 25.0]


# ── 2. Rank noise ─────────────────────────────────────────────────────────────

def test_the_tiered_shape_is_normalized_so_sd_keeps_its_units():
    """A tier multiplier averaging one keeps `rank_noise_sd` comparable across models."""
    rank = np.arange(200)
    tiered = D.noise_scale(rank, "tiered", 4.0)
    assert tiered[0] < 4.0 < tiered[-1]              # tight at the top, loose in the tail
    assert np.isclose(D.noise_scale(np.array([0, 30, 60, 120]), "tiered", 1.0).mean(),
                      1.0)


def test_zero_noise_makes_every_draft_identical_which_is_the_degenerate_case():
    """The no-fit floor, and the reason it cannot ship however well it scores.

    At `sd = 0` the first round is the board's top twelve and every draft plays out the
    same way, so a seat holds one roster in every simulated season. No mean-ADP statistic
    sees that — `field_diversity` is what does.
    """
    frame, board = _board(_balanced_pool())
    state = D.run_drafts(board, _cfg(), 3, np.random.default_rng(0))
    mean_pick, rate = D.simulated_adp(state)
    drafted = ~np.isnan(mean_pick)

    assert np.array_equal(mean_pick[:12], np.arange(1.0, 13.0))
    assert rate[drafted].min() == 1.0                    # drafted in every draft, or not
    assert (state.roster[0] == state.roster[1]).all()
    diversity = D.field_diversity(state)
    assert diversity["roster_overlap"] == 1.0
    assert diversity["pool_coverage"] == 1.0

    noisy = D.run_drafts(board, D.FieldConfig(noise_model="tiered", rank_noise_sd=4.0),
                         6, np.random.default_rng(0))
    spread = D.field_diversity(noisy)
    assert spread["roster_overlap"] < 1.0
    assert spread["pool_coverage"] > 1.0


def test_an_unknown_noise_model_raises_rather_than_falling_back():
    with pytest.raises(KeyError):
        D.noise_scale(np.arange(5), "gumbel", 1.0)


# ── 3. The engine's hard rules ────────────────────────────────────────────────

def test_the_snake_reverses_every_round_and_gives_everyone_sixteen_picks():
    order = D.snake_order(12, D.N_ROUNDS)
    assert len(order) == 12 * 16
    assert list(order[:12]) == list(range(12))
    assert list(order[12:24]) == list(reversed(range(12)))
    counts = np.bincount(order)
    assert counts.min() == counts.max() == 16


def test_position_caps_bind_the_field_and_are_dks_own_defaults():
    """8 G / 8 F / 3 C, from `docs/dk_best_ball_rules.md` — not a heuristic."""
    assert D.POSITION_CAPS == {"G": 8, "F": 8, "C": 3}
    frame, board = _board(_balanced_pool())
    state = D.run_drafts(board, _cfg(), 2, np.random.default_rng(0))
    caps = np.asarray([D.POSITION_CAPS[p] for p in POSITIONS])
    assert (state.have <= caps).all()


def test_a_capped_seat_falls_back_to_the_whole_board_rather_than_raising():
    """DK's rule when every limit is reached: take the top player anyway.

    Caps of 1/1/1 cannot fill sixteen rounds, so the fallback fires on pick four onward. A
    version that treated the caps as absolute would raise here, which is a crash where DK
    documents a pick.
    """
    frame, board = _board(_balanced_pool())
    state = D.run_drafts(board, _cfg(position_caps=(1, 1, 1),
                                     require_legal_lineup=False),
                         1, np.random.default_rng(0))
    assert (state.roster >= 0).all()
    assert int(state.have.sum()) == 12 * ROSTER_SIZE


def test_a_roster_stays_seatable_because_the_caps_do_not_guarantee_it():
    """8 G + 8 F + 0 C satisfies every DK cap and seats no centre.

    `require_legal_lineup` is the guard, and it is not a DK rule — without it
    `bracket.best_lineup` returns a plausible six-man total for such a roster and nothing
    anywhere raises.
    """
    frame, board = _board(_balanced_pool())
    masks = D.position_masks(frame)

    guarded = D.run_drafts(board, _cfg(require_legal_lineup=True), 4,
                           np.random.default_rng(1))
    assert roster_is_legal(masks[guarded.roster.reshape(-1, ROSTER_SIZE)]).all()

    # Every centre ranked behind the last pick, which DK's caps say nothing about.
    starved = _pool("G" * 190 + "F" * 190 + "C" * 20)
    frame, board = _board(starved)
    starved_masks = D.position_masks(frame)
    unguarded = D.run_drafts(board, _cfg(require_legal_lineup=False), 1,
                             np.random.default_rng(1))
    rosters = starved_masks[unguarded.roster.reshape(-1, ROSTER_SIZE)]
    assert not roster_is_legal(rosters).any()

    guarded = D.run_drafts(board, _cfg(require_legal_lineup=True), 1,
                           np.random.default_rng(1))
    assert roster_is_legal(starved_masks[guarded.roster.reshape(-1, ROSTER_SIZE)]).all()


def test_the_engines_exclusion_list_yields_only_when_the_board_would_be_empty():
    """Same rule as `autodraft_pick`'s, applied to the vectorized path.

    "Players on the excluded list will not be auto-drafted unless it is necessary to create
    a valid lineup" — so an exclusion that empties the board is ignored rather than raising.
    """
    frame, board = _board(_balanced_pool())
    state = D.new_state(board, 12, 1)
    excluded = np.zeros(board.n_players, dtype=bool)
    excluded[:20] = True
    assert not D.legal_mask(state, 0, _cfg(), excluded=excluded)[0][:20].any()

    excluded[:] = True
    assert D.legal_mask(state, 0, _cfg(), excluded=excluded)[0].all()


def test_nobody_is_drafted_twice_and_every_seat_ends_with_sixteen():
    frame, board = _board(_balanced_pool())
    state = D.run_drafts(board, D.FieldConfig(noise_model="tiered", rank_noise_sd=4.0),
                         5, np.random.default_rng(2))
    for draft in range(state.n_drafts):
        picked = state.roster[draft].reshape(-1)
        assert len(np.unique(picked)) == len(picked) == 12 * ROSTER_SIZE
    assert (state.pick_of >= 0).sum(axis=1).tolist() == [12 * ROSTER_SIZE] * 5


def test_a_drafters_opinion_is_drawn_once_and_held_for_the_whole_draft():
    """Redrawing the noise per pick would make a drafter incoherent — and narrower.

    Pinned through the interface rather than the outcome: `static_keys` is called once per
    strategy per run, so a future opponent cannot reintroduce per-pick redraws by accident.
    """
    frame, board = _board(_balanced_pool())
    calls = []

    class Counting(D.AdpAutodraft):
        name = "counting"

        def static_keys(self, board, n_drafts, n_seats, rng):
            calls.append((n_drafts, n_seats))
            return super().static_keys(board, n_drafts, n_seats, rng)

    D.OPPONENTS["counting"] = Counting
    try:
        D.run_drafts(board, _cfg(), 2, np.random.default_rng(0),
                     seat_strategies=["counting"] * 12)
    finally:
        del D.OPPONENTS["counting"]
    assert calls == [(2, 12)]


# ── 4. The opponent registry and field composition ────────────────────────────

def test_the_need_aware_opponent_reaches_for_a_slot_it_still_owes():
    """The roster-aware hook is exercised, not just declared.

    A seat holding two guards and no centre values the next centre above its board rank; a
    seat with the same board and no need weight does not.
    """
    frame, board = _board(_pool("GGGGGGCC"))
    state = D.new_state(board, pod_size=1, n_drafts=1)
    state.have[0, 0] = np.array([2, 2, 0])            # two G, two F, no C
    plain = D.AdpNeedAware(_cfg(need_weight=0.0)).dynamic_bonus(board, state.have[:, 0, :])
    keen = D.AdpNeedAware(_cfg(need_weight=50.0)).dynamic_bonus(board, state.have[:, 0, :])
    assert float(np.max(plain)) == 0.0
    centres = board.eligible[:, POSITIONS.index("C")]
    assert keen[0][centres].min() == 50.0
    assert keen[0][~centres].max() == 0.0


def test_field_composition_falls_back_to_default_and_refuses_an_unknown_strategy():
    cfg = {"sim": {"field": {"composition": {"default": {"adp": 1.0},
                                             "20k_spin_move": {"adp": 3.0,
                                                               "ranking_submission": 1.0}}}}}
    assert D.field_composition(cfg, "600k_shootaround") == {"adp": 1.0}
    assert D.field_composition(cfg, "20k_spin_move") == {"adp": 0.75,
                                                         "ranking_submission": 0.25}
    with pytest.raises(KeyError):
        D.field_composition({"sim": {"field": {"composition": {"default": {"psychic": 1}}}}})


def test_seats_are_allotted_deterministically_and_the_shares_are_the_closest_integers():
    """A field has to be reproducible from its config alone, not resampled per draft."""
    assert D.assign_seats({"adp": 1.0}, 12) == ["adp"] * 12
    mixed = D.assign_seats({"adp": 0.75, "ranking_submission": 0.25}, 12)
    assert mixed.count("adp") == 9 and mixed.count("ranking_submission") == 3
    assert D.assign_seats({"adp": 0.75, "ranking_submission": 0.25}, 12) == mixed
    lopsided = D.assign_seats({"adp": 0.99, "ranking_submission": 0.01}, 12)
    assert lopsided.count("ranking_submission") == 0     # 0.12 seats rounds down
    assert len(mixed) == len(lopsided) == 12


def test_a_mixed_field_runs_and_only_the_submitted_boards_repeat():
    """A submitted board does not vary between drafts; a human's opinion does.

    That difference is the whole reason `field_composition` is keyed by tournament — a tier
    with more autodraft entries is a *less* diverse field, not merely a weaker one.
    """
    frame, board = _board(_balanced_pool())
    seats = ["adp"] * 8 + ["ranking_submission"] * 4
    state = D.run_drafts(board, D.FieldConfig(noise_model="tiered", rank_noise_sd=6.0),
                         3, np.random.default_rng(0), seat_strategies=seats)
    assert (state.have.sum(axis=2) == ROSTER_SIZE).all()

    # Its *preference* is identical across drafts and seats; its roster still is not,
    # because what reaches it depends on the eleven picks in front of it.
    cfg = D.FieldConfig(noise_model="tiered", rank_noise_sd=6.0)
    submitted = D.RankingSubmission(cfg).static_keys(board, 3, 4,
                                                     np.random.default_rng(0))
    human = D.AdpAutodraft(cfg).static_keys(board, 3, 4, np.random.default_rng(0))
    assert (submitted[0] == submitted[1]).all()
    assert (human[0] != human[1]).any()


# ── 5. DK's documented autodraft ──────────────────────────────────────────────

def _autodraft_case(n: int = 8):
    eligible = np.zeros((n, 3), dtype=bool)
    eligible[np.arange(n), np.arange(n) % 3] = True         # G, F, C, G, F, C, ...
    return np.arange(n), np.ones(n, dtype=bool), eligible


def test_autodraft_takes_the_top_of_the_ranking_when_the_queue_is_empty():
    ranking, available, eligible = _autodraft_case()
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible) == 0
    available[0] = False
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible) == 1


def test_the_queue_comes_first_and_beats_a_better_ranked_player():
    ranking, available, eligible = _autodraft_case()
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible, queue=[5]) == 5


def test_the_queue_prefers_an_uncapped_position_but_will_break_the_cap_if_it_must():
    """Both halves of DK's sentence, and they point opposite ways.

    "If a set position limit has been reached, the top player in the queue will be selected
    from a position with a limit that hasn't been reached yet. If the queue only contains
    players from positions with limits that have been reached, the top player in the queue
    will be selected."
    """
    ranking, available, eligible = _autodraft_case()
    caps = (0, 8, 3)                                        # guards are full
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible,
                            queue=[0, 4], caps=caps) == 4   # 0 is a G, 4 is an F
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible,
                            queue=[0, 3], caps=caps) == 0   # both guards: cap yields


def test_the_ranking_path_treats_the_caps_as_hard():
    """"We will not auto-draft more than 8Gs, 8Fs, or 3Cs ... from the pre-draft rankings.\""""
    ranking, available, eligible = _autodraft_case()
    assert D.autodraft_pick(ranking, available, (8, 0, 0), eligible,
                            caps=(8, 8, 3)) == 1            # skips the capped guard
    assert D.autodraft_pick(ranking, available, (8, 8, 3), eligible,
                            caps=(8, 8, 3)) == 0            # everything full: take the top


def test_an_excluded_player_is_skipped_but_taken_when_nothing_else_fills_the_slot():
    """"...will not be auto-drafted unless it is necessary to create a valid lineup.\""""
    ranking, available, eligible = _autodraft_case()
    excluded = np.zeros(len(ranking), dtype=bool)
    excluded[0] = True
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible,
                            excluded=excluded) == 1
    excluded[:] = True
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible,
                            excluded=excluded) == 0


def test_the_queue_ignores_the_exclusion_list_because_you_asked_for_him():
    """"Excluded players can still be manually drafted or auto-drafted from the queue.\""""
    ranking, available, eligible = _autodraft_case()
    excluded = np.ones(len(ranking), dtype=bool)
    assert D.autodraft_pick(ranking, available, (0, 0, 0), eligible, queue=[6],
                            excluded=excluded) == 6


def test_the_exported_ranking_carries_dks_own_columns():
    frame, _ = _board(_pool("GFC"))
    dest = D.export_ranking(frame, D.Path("outputs/predictions/_test_ranking.csv"))
    out = pd.read_csv(dest)
    dest.unlink()
    assert list(out.columns[:6]) == ["Rank", "ID", "Name", "Position", "ADP", "Team"]
    assert list(out["Rank"]) == [1, 2, 3]


# ── 6. The reactive mode ──────────────────────────────────────────────────────

def _tensor(n_players: int, n_periods: int = 4, n_sims: int = 3,
            level: np.ndarray | None = None) -> np.ndarray:
    level = np.linspace(30.0, 1.0, n_players) if level is None else level
    return np.repeat(level[:, None, None], n_periods, axis=1).repeat(
        n_sims, axis=2).astype(np.float32)


def test_marginal_value_is_a_lineup_lift_and_not_a_projection():
    """A sixth guard adds nothing once the slate is full of guards, and the matroid knows.

    This is the whole argument for pricing candidates by `bracket.best_lineup` rather than
    by their own projection: with 2 G + 2 UTIL filled by guards, the next guard cannot start.
    """
    frame, board = _board(_pool("G" * 8 + "C" + "F" * 3))
    masks = D.position_masks(frame)
    dk_pts = _tensor(len(frame))

    # Guards can fill 2 G plus the 2 UTIL seats and no more, so the fifth is worth nothing
    # even though his projection is well above the twelfth-ranked player's.
    assert D.marginal_lineup_value(dk_pts, np.arange(3), masks, np.array([3]))[0] > 0
    assert D.marginal_lineup_value(dk_pts, np.arange(4), masks,
                                   np.array([4]))[0] == pytest.approx(0.0)

    # And on the same roster the lower-projected centre is worth more than that guard.
    value = D.marginal_lineup_value(dk_pts, np.arange(4), masks, np.array([4, 8]))
    assert value[1] > value[0] == pytest.approx(0.0)
    assert dk_pts[8, 0, 0] < dk_pts[4, 0, 0]


def test_recommend_never_names_a_player_the_engine_would_refuse():
    """Both modes share `legal_mask`, so a recommendation is always a legal pick."""
    frame, board = _board(_balanced_pool())
    state = D.new_state(board, 12, 1)
    state.taken[0, :50] = True
    dk_pts = _tensor(len(frame))
    table = D.recommend(state, frame, 0, _cfg(), dk_pts=dk_pts, top=5)
    assert (table["board_rank"] >= 50).all()


def test_an_unpriceable_player_is_excluded_from_the_ranking_rather_than_valued_at_zero():
    """Zero is a number, and a number this ranking would act on.

    The board deliberately carries players the tensor cannot score — the field takes them at
    their ADP and who is left at pick k is what a snake draft turns on — so they have to be
    refused explicitly.
    """
    frame, board = _board(_balanced_pool())
    state = D.new_state(board, 12, 1)
    dk_pts = _tensor(len(frame))
    scorable = np.ones(len(frame), dtype=bool)
    scorable[0] = False
    table = D.recommend(state, frame, 0, _cfg(), dk_pts=dk_pts, scorable=scorable, top=3)
    assert 0 not in set(table["board_rank"])


def test_our_seat_is_not_bound_by_dks_autodraft_caps():
    """"the only way to override them ... is to make a manual selection" — the rules doc.

    Applying 8 G / 8 F / 3 C to a manual pick would forbid a roster a human may draft, which
    is a silent restriction on the one seat we control.
    """
    assert D.manual_config(_cfg()).position_caps == (ROSTER_SIZE,) * 3
    frame, board = _board(_balanced_pool())
    state = D.new_state(board, 12, 1)
    state.have[0, 0] = np.array([8, 0, 0])              # already at the guard cap
    capped = D.legal_mask(state, 0, _cfg())[0]
    manual = D.legal_mask(state, 0, D.manual_config(_cfg()))[0]
    guards = board.eligible[:, POSITIONS.index("G")]
    assert not capped[guards].any()
    assert manual[guards].all()


# ── 7. Gate B ─────────────────────────────────────────────────────────────────

def test_simulated_adp_is_a_mean_over_the_drafts_he_went_in():
    """DK's own definition. A mean over *all* drafts would manufacture a tail gap."""
    board = D.Board(player_id=np.array([1, 2]), player_name=np.array(["a", "b"]),
                    rank=np.array([0.0, 1.0]), eligible=np.ones((2, 3), bool),
                    adp=np.array([1.0, 2.0]), season="2022-23")
    state = D.new_state(board, pod_size=1, n_drafts=4)
    state.pick_of[:] = np.array([[0, 9], [0, -1], [0, -1], [0, -1]])
    mean_pick, rate = D.simulated_adp(state)
    assert mean_pick[0] == 1.0 and rate[0] == 1.0
    assert mean_pick[1] == 10.0 and rate[1] == 0.25


def test_the_fit_region_drops_the_maps_terminal_plateau_and_nothing_else():
    adp = np.array([1.0, 5.0, 20.0, 160.0, 160.0, np.nan])
    assert list(D.fit_region(adp)) == [True, True, True, False, False, False]


def test_gate_b_reproduces_the_curve_on_the_real_validation_board():
    """The one test that reads the real board, because that is the whole claim.

    A cheap grid — the point is that the field's realized ADP tracks the curve it consumed,
    not the fitted value, which `make draft-sim` owns.
    """
    pool = pd.read_parquet("data/features/draft_pool.parquet")
    frame, board = _board(pool[pool["season"] == "2022-23"])
    row = D.gate_b(board, D.FieldConfig(noise_model="tiered", rank_noise_sd=3.0),
                   24, np.random.default_rng(0))
    assert row["mae_fit"] < D.RECALIBRATION_ERROR
    assert row["mae_all"] < D.RECALIBRATION_ERROR
    assert row["n_fit"] < row["n_adp"]                  # the plateau really is there


def test_the_shipped_field_is_read_from_the_artifact_rather_than_the_config():
    """`rank_noise_sd: null` means "Gate B fitted it", so a missing artifact must raise."""
    cfg = {"sim": {"field": {"rank_noise_sd": None, "noise_model": None}},
           "evaluation": {"predictions_dir": "outputs/predictions"}}
    with pytest.raises(FileNotFoundError):
        D.selected_field(cfg, out_dir=D.Path("outputs/predictions/_absent"))
    pinned = {"sim": {"field": {"rank_noise_sd": 2.5, "noise_model": "constant"}},
              "evaluation": {"predictions_dir": "outputs/predictions"}}
    assert D.selected_field(pinned).rank_noise_sd == 2.5
