"""Tests for the strategy sweep — the injection, the board, the portfolio, the split.

What is worth pinning in `src/sim/strategy.py` is, once again, the arithmetic whose failure
is **silent and plausible**. Four things here can be wrong and still produce a complete,
well-ordered sweep table:

- **the injection scored against the wrong mean.** `magnitude_check` on an injected tensor
  can take the model's belief from the tensor it is scoring, which has already absorbed the
  injected bias — the gate then reports a smaller error than the model actually makes, and
  reports it against the right bar with the right sign.
- **the Gate C bar read off the wrong rows.** The 0.81-0.95 band is the *count* heads'
  carry-forward floor. Filter the same file to the *selected* rows instead and it becomes
  [0.13, 0.96], which no world could fail.
- **the board keeping players the tensor scores at zero.** They cost the ADP field 1.26
  roster spots an entry and cost a model-ranked strategy nothing, which reads as edge.
- **the split.** Prose already failed here once, so the sweep's guard is pinned the way
  `component_rates`' is: by AST, so a future edit that carves its own split fails the suite
  rather than the review.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import held_out
from src.models.held_out import HeldOutLocked, unlocked
from src.sim import draft as D
from src.sim import draft_room as R
from src.sim import strategy as S
from src.sim.bracket import ROSTER_SIZE

MODULE = "src/sim/strategy.py"


# ── Synthetic builders ────────────────────────────────────────────────────────

def _pool(n_g: int = 200, n_f: int = 190, n_c: int = 70,
          season: str = "2022-23") -> pd.DataFrame:
    """A draft pool deep enough for a twelve-seat, sixteen-round snake."""
    letters: list[str] = []
    while len(letters) < n_g + n_f + n_c:
        for letter, cap in (("G", n_g), ("F", n_f), ("C", n_c)):
            if letters.count(letter) < cap:
                letters.append(letter)
    n = len(letters)
    teams = [f"T{i % 24:02d}" for i in range(n)]
    return pd.DataFrame({
        "season": [season] * n, "player_id": np.arange(1, n + 1),
        "player_name": [f"P{i:03d}" for i in range(n)], "team": teams,
        "position": letters,
        "pos_g": [c == "G" for c in letters], "pos_f": [c == "F" for c in letters],
        "pos_c": [c == "C" for c in letters],
        "adp": np.arange(1.0, n + 1.0), "adp_dk_scale": np.arange(1.0, n + 1.0),
        "prior_min_total": np.linspace(2000.0, 100.0, n),
        "dk_player_id": np.arange(1, n + 1)})


def _room(pool: pd.DataFrame | None = None, n_periods: int = 4, n_sims: int = 24,
          seed: int = 0, scorable: np.ndarray | None = None,
          season: str = "2022-23") -> R.Room:
    """A room over a synthetic board — no tensor file, no posterior, no field draft.

    The model's board order is deliberately **not** the ADP order: `level` descends with a
    shuffled index, so `model_mean` and `adp` are two genuinely different rankings and a
    blend has something to blend.
    """
    pool = _pool() if pool is None else pool
    frame = D.build_board(pool, season)
    board = D.to_arrays(frame, season)
    rng = np.random.default_rng(seed)
    shuffle = rng.permutation(len(frame))
    level = np.empty(len(frame))
    level[shuffle] = np.linspace(40.0, 5.0, len(frame))
    dk_pts = (level[:, None, None]
              + rng.normal(0.0, 6.0, (len(frame), n_periods, n_sims))).astype(np.float32)
    return R.Room(season=season, frame=frame, board=board, dk_pts=dk_pts,
                  scorable=(np.ones(len(frame), dtype=bool) if scorable is None
                            else scorable),
                  masks=np.asarray([1 if c == "G" else 2 if c == "F" else 4
                                    for c in frame["position"]], dtype=np.int16),
                  round_of_period=np.array([1] * (n_periods - 1) + [2]),
                  projection=dk_pts.sum(axis=1).mean(axis=1),
                  pod_size=12, field_cfg=D.FieldConfig(rank_noise_sd=4.0),
                  field_round=np.zeros((1, 2, n_sims), dtype=np.float32),
                  refs={}, fit_window="train")


def _design(seasons: list[str]) -> pd.DataFrame:
    return pd.DataFrame({"season": seasons, "player_id": range(len(seasons))})


def _names(path: str) -> set[str]:
    tree = ast.parse(Path(path).read_text())
    out = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            out.add(node.id)
        elif isinstance(node, ast.Attribute):
            out.add(node.attr)
        elif isinstance(node, ast.alias):
            out.add(node.asname or node.name.split(".")[-1])
    return out


# ── 1. The split guard ────────────────────────────────────────────────────────

def test_the_sweep_refuses_a_test_season(monkeypatch):
    """Pinned the way `component_rates`' guard is, because prose already failed here once.

    Two halves, and the second is the one that rots: the seasons must come through
    `selection_split`, and the module must not name `split_seasons`, which is what hands
    back the guarded frame. `component_rates` kept a private copy of exactly that function
    for as long as it went unguarded.
    """
    monkeypatch.setattr(held_out, "_unlocked", False, raising=False)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]
    design = _design(seasons)

    assert S.validation_seasons(design) == ["2022-23", "2023-24"]
    with pytest.raises(HeldOutLocked):
        S.assert_season_allowed("2025-26", design)
    with pytest.raises(HeldOutLocked):
        S.assert_season_allowed("2024-25", design)
    with unlocked("a test"):
        S.assert_season_allowed("2025-26", design)
    S.assert_season_allowed("2023-24", design)          # validation is always legal


def test_the_sweep_reaches_the_split_only_through_selection_split():
    names = _names(MODULE)
    assert "split_seasons" not in names, (
        f"{MODULE} names split_seasons, which hands back the guarded held-out frame")
    source = Path(MODULE).read_text()
    assert "assert_season_allowed" in source and "validation_seasons" in source


def test_run_raises_before_it_loads_anything(monkeypatch, tmp_path):
    """The guard fires on the *season list*, ahead of every artifact read.

    A guard that only fires after the tensor is loaded is a guard that has already read the
    season it was supposed to refuse — the frames are innocent, but the run is not.
    """
    monkeypatch.setattr(held_out, "_unlocked", False, raising=False)
    seasons = [f"{y}-{str(y + 1)[-2:]}" for y in range(2015, 2026)]
    monkeypatch.setattr(S, "split_frame", lambda cfg: _design(seasons))

    def _boom(*a, **k):
        raise AssertionError("load_room was reached for a held-out season")

    monkeypatch.setattr(R, "load_room", _boom)
    cfg = {"data": {"features_dir": str(tmp_path)},
           "evaluation": {"predictions_dir": str(tmp_path)},
           "sim": {"tournaments": {}}}
    with pytest.raises(HeldOutLocked):
        S.run(cfg, seasons=["2024-25"])


# ── 2. Gate C — the injection, and the two ways of reading it wrong ───────────

def _world(n: int = 300, n_sims: int = 40, seed: int = 0):
    """A model view, an ADP the model disagrees with, and the derived signal."""
    rng = np.random.default_rng(seed)
    level = np.linspace(3000.0, 200.0, n)
    dk = (level[:, None, None] / 4.0
          + rng.normal(0.0, 90.0, (n, 4, n_sims))).astype(np.float32)
    model_total = dk.sum(axis=1).mean(axis=1).astype(np.float64)
    adp = np.argsort(np.argsort(-model_total + rng.normal(0.0, 200.0, n))) + 1.0
    d, has = S.market_signal(model_total, adp)
    return dk, model_total, adp, d, has


def test_market_signal_is_positive_for_the_player_the_market_likes_more():
    """The market publishes an ordering; differencing it against a projection needs a scale.

    The scale used is the model's own, so the signal is *disagreement* and nothing else.
    Hand-built so the sign is arithmetic rather than remembered: four players the model
    ranks 0 > 1 > 2 > 3, an ADP that takes player 3 first, and the only positive gap must be
    his.
    """
    model_total = np.array([100.0, 90.0, 80.0, 70.0])
    adp = np.array([2.0, 3.0, 4.0, 1.0])           # the market takes player 3 first
    d, has = S.market_signal(model_total, adp)
    assert has.all()
    assert d[3] > 0 and (d[:3] < 0).all()
    assert abs(float(d.mean())) < 1e-9 and abs(float(d.std()) - 1.0) < 1e-9


def test_market_signal_is_silent_where_the_market_is():
    """A player with no ADP gets a zero signal rather than an imputed one — the market said
    nothing about him, and standardizing over him would invent an opinion."""
    model_total = np.array([100.0, 90.0, 80.0, 70.0, 60.0])
    adp = np.array([2.0, np.nan, 4.0, 1.0, np.nan])
    d, has = S.market_signal(model_total, adp)
    assert list(has) == [True, False, True, True, False]
    assert d[1] == 0.0 and d[4] == 0.0
    assert abs(float(d[has].mean())) < 1e-9 and abs(float(d[has].std()) - 1.0) < 1e-9


def test_the_injection_hits_both_of_its_targets():
    """The scale lands on the MAE bar and the mixing weight lands on the skill gap.

    Both are solved rather than plugged in, so both are checkable to a tolerance rather than
    remembered. The gap is the whole point of the injection: without it the market is a
    strictly noisier copy of the model however large the error is.
    """
    dk, model_total, adp, d, has = _world()
    sims = S.truth_sim_index(dk.shape[2], 12)
    base_gap = S.simulated_skill_gap(dk, model_total, adp, has, sims)[0]
    assert base_gap < 0.0            # the uninjected world has the model ahead

    truth, record = S.calibrate_injection(dk, d, has, adp, model_total,
                                          gap_target=0.05, mae_target=400.0,
                                          truth_sims=sims)
    assert abs(record["achieved_mae"] - 400.0) < 5.0
    assert abs(record["sim_skill_gap"] - 0.05) < 0.02
    assert record["rho_source"] == "skill_gap"
    assert 0.0 < record["rho"] < 0.95


def test_a_zero_market_weight_leaves_the_world_market_blind():
    """`rho = 0` is the uninjected structure at the injected magnitude — the control.

    It is what an injection that reproduced only the *magnitude* of the miss would produce,
    and the reason that is not enough: the error still correlates with nothing the market
    can see.
    """
    dk, model_total, adp, d, has = _world()
    sims = S.truth_sim_index(dk.shape[2], 12)
    truth, _ = S.inject_error(dk, d, has, 0.0, 400.0, sims)
    err = truth.sum(axis=1).astype(np.float64) - model_total[:, None]
    rho = np.mean([np.corrcoef(d[has], err[has, s])[0, 1] for s in sims])
    assert abs(rho) < 0.05

    hot, _ = S.inject_error(dk, d, has, 0.6, 400.0, sims)
    err = hot.sum(axis=1).astype(np.float64) - model_total[:, None]
    assert np.mean([np.corrcoef(d[has], err[has, s])[0, 1] for s in sims]) > 0.3


def test_the_gate_scores_the_models_belief_not_the_injected_worlds_own_mean():
    """The silent one. An injected world's own mean has already absorbed the market bias.

    Recomputing the model's prediction from the tensor being scored makes the gate report a
    smaller error than the model actually makes — with every column present, every shape
    right and only the number wrong, which is the failure mode this whole gate exists to
    catch one level up.
    """
    dk, model_total, adp, d, has = _world()
    sims = S.truth_sim_index(dk.shape[2], 12)
    truth, _ = S.inject_error(dk, d, has, 0.7, 400.0, sims)
    gp = np.full(dk.shape, 20, dtype=np.uint8)
    predictive = dk.sum(axis=1).astype(np.float64)

    honest = S.magnitude_check(truth, gp, sims, model_total=model_total,
                               predictive=predictive)
    self_scored = S.magnitude_check(truth, gp, sims)
    assert honest["season_total_mae"] > self_scored["season_total_mae"] + 5.0


def test_the_component_band_is_the_count_heads_carry_forward_floor(tmp_path):
    """The bar cannot be allowed to drift onto whichever rows are convenient.

    Read off the *selected* rows the same file gives [0.13, 0.96], a band no simulated world
    could fail; the 0.81-0.95 the plan and README quote is the count heads' no-fit floor.
    """
    pd.DataFrame({
        "head": ["fga", "blk", "fg3m|fg3a", "ftm|fta"],
        "kind": ["count", "count", "conversion", "conversion"],
        "variant": ["carry_forward", "carry_forward", "carry_forward", "carry_forward"],
        "val_r2": [0.9514, 0.8103, 0.1277, -0.0144],
        "selected": [True, True, True, True],
    }).to_csv(tmp_path / "stan_component_metrics.csv", index=False)
    lo, hi = S.component_r2_band(tmp_path)
    assert (round(lo, 4), round(hi, 4)) == (0.8103, 0.9514)


# ── 3. The board both sides draft from ────────────────────────────────────────

def test_the_priceable_board_drops_exactly_the_unscorable_rows(monkeypatch):
    """A player the tensor scores at zero is a handicap on whoever drafts him.

    The restriction is symmetric and the renumbering is the part that would fail silently:
    `board_rank` is a row index everywhere downstream, so a filtered frame that keeps its
    old ranks indexes the wrong player.
    """
    room = _room()
    scorable = np.ones(len(room.frame), dtype=bool)
    scorable[::7] = False
    room = _room(scorable=scorable)
    monkeypatch.setattr(R, "build_field",
                        lambda *a, **k: np.zeros((12, 2, room.dk_pts.shape[2]),
                                                 dtype=np.float32))
    monkeypatch.setattr(R, "field_reference", lambda *a, **k: None)
    out, dropped = S.priceable_room({}, room, seed=0, n_field_drafts=1)

    assert dropped["n_dropped"] == int((~scorable).sum())
    assert len(out.frame) == int(scorable.sum())
    assert out.scorable.all()
    assert list(out.frame["board_rank"]) == list(range(len(out.frame)))
    assert list(out.frame["player_id"]) == list(room.frame["player_id"][scorable])
    assert np.allclose(out.dk_pts, room.dk_pts[scorable])


# ── 4. The strategy object ────────────────────────────────────────────────────

def test_alpha_is_a_rank_blend_with_the_two_ends_exact():
    """`alpha = 0` is the model's board and `alpha = 1` is the market's, exactly."""
    model = np.array([3.0, 0.0, 1.0, 2.0])
    market = np.array([0.0, 1.0, 2.0, 3.0])
    pure_model = S.Strategy("m", ranking="blend", alpha=0.0)
    pure_market = S.Strategy("a", ranking="adp")
    assert np.allclose(S.strategy_keys(pure_model, model, market, 0), model)
    assert np.allclose(S.strategy_keys(pure_market, model, market, 0), market)
    half = S.Strategy("h", ranking="blend", alpha=0.5)
    assert np.allclose(S.strategy_keys(half, model, market, 0), 0.5 * (model + market))


def test_the_per_round_schedule_switches_where_the_disagreement_does():
    """Rounds 1-2, 3-8 and 9-16 — the tiers `docs/adp-plan.md` measured, not sixteen knobs."""
    s = S.Strategy("late", ranking="blend", alpha=0.4, alpha_rounds=(0.1, 0.4, 0.8))
    assert [s.alpha_at(r) for r in (0, 1)] == [0.1, 0.1]
    assert [s.alpha_at(r) for r in (2, 7)] == [0.4, 0.4]
    assert [s.alpha_at(r) for r in (8, 15)] == [0.8, 0.8]
    flat = S.Strategy("flat", ranking="blend", alpha=0.4)
    assert {flat.alpha_at(r) for r in range(16)} == {0.4}


def test_an_unregistered_ranking_or_objective_raises():
    with pytest.raises(KeyError):
        S.Strategy("x", ranking="vibes")
    with pytest.raises(KeyError):
        S.Strategy("x", objective="hunch")
    with pytest.raises(ValueError):
        S.Strategy("x", ranking="blend", alpha=1.4)


def test_entry_seats_spread_across_the_pod():
    """DK randomizes the draft slot, so a portfolio measured from seat 0 is measured on a
    board that never runs out of first-round talent."""
    for n in (1, 4, 10, 12):
        seats = S.entry_seats(n, 12)
        assert len(seats) == n
        assert seats.min() >= 0 and seats.max() < 12
        assert len(set(seats.tolist())) == n
    assert S.entry_seats(4, 12).tolist() == [0, 4, 7, 11]


# ── 5. Drafting a portfolio ───────────────────────────────────────────────────

def test_every_drafted_roster_is_sixteen_legal_players():
    room = _room()
    rosters, meta = S.draft_portfolio(room, S.Strategy("m"), "t", 4,
                                      np.random.default_rng(0))
    assert rosters.shape == (4, ROSTER_SIZE)
    for roster in rosters:
        assert len(set(roster.tolist())) == ROSTER_SIZE
    assert len(meta["seats"]) == 4


def test_the_exposure_cap_binds_across_entries():
    """The one axis that cannot show up in a per-entry mean at all.

    Ten entries holding the same sixteen players have the same `P(advance)` as one; what
    they do not have is the same `P(at least one advances)`. So the cap has to be enforced
    *across* the portfolio, which is why the entries are drafted in sequence rather than
    vectorized.
    """
    room = _room()
    n = 6
    for cap in (1.0, 0.5):
        rosters, meta = S.draft_portfolio(room, S.Strategy("c", exposure_cap=cap), "t", n,
                                          np.random.default_rng(0))
        counts = pd.Series(rosters.ravel()).value_counts()
        assert counts.max() <= int(np.ceil(cap * n))
        assert meta["max_exposure"] <= cap + 1e-9
    loose, _ = S.draft_portfolio(room, S.Strategy("c"), "t", n, np.random.default_rng(0))
    tight, _ = S.draft_portfolio(room, S.Strategy("c", exposure_cap=0.34), "t", n,
                                 np.random.default_rng(0))
    assert len(set(tight.ravel().tolist())) > len(set(loose.ravel().tolist()))


def test_stacking_buys_more_teammates():
    """Same-team pairs are the one correlation a drafter can buy on purpose."""
    room = _room()

    def pairs(rosters):
        teams = room.frame["team"].to_numpy()
        return float(np.mean([len(t) - len(set(t))
                              for t in (teams[r].tolist() for r in rosters)]))

    off, _ = S.draft_portfolio(room, S.Strategy("s"), "t", 4, np.random.default_rng(0))
    on, _ = S.draft_portfolio(room, S.Strategy("s", stacking=25.0), "t", 4,
                              np.random.default_rng(0))
    assert pairs(on) > pairs(off)


# ── 6. Scoring a portfolio ────────────────────────────────────────────────────

def test_p_any_advance_is_the_portfolio_identity():
    """`1 - prod(1 - p)` inside a sim, and it is not recoverable from per-entry means.

    Checked on a reference whose answer is arithmetic: identical entries in a two-round
    tournament give every entry the same `p` in every sim, so the portfolio's number is
    `1 - (1 - p)^N` exactly. Averaging the per-entry probabilities first gives `p`, which is
    the mistake this identity exists to prevent.
    """
    n_sims = 32
    rng = np.random.default_rng(0)
    field = rng.normal(100.0, 10.0, (240, 2, n_sims))
    rounds = [{"round": 1, "pod_size": 12, "n_advance": 2,
               "payout": np.zeros(12)},
              {"round": 2, "pod_size": 12, "n_advance": 0,
               "payout": np.r_[np.full(4, 30.0), np.zeros(8)]}]
    ref = R.field_reference(field.astype(np.float32), "t", rounds=rounds, entry_fee=10.0)

    ours = np.repeat(field[:1], 5, axis=0)             # five identical entries
    out = S.portfolio_outcome(ours, ref, np.random.default_rng(1), draws=200)
    _, p = R.bracket_ev(ours, ref, per_sim=True)
    expected = float((1.0 - np.prod(1.0 - p, axis=0)).mean())
    assert abs(out["p_any_advance"] - expected) < 1e-12
    assert out["p_any_advance"] > out["p_advance"]
    assert out["p_advance_lo"] <= out["p_advance"] <= out["p_advance_hi"]


def test_a_stronger_entry_advances_more_often():
    n_sims = 32
    rng = np.random.default_rng(0)
    field = rng.normal(100.0, 10.0, (240, 2, n_sims))
    rounds = [{"round": 1, "pod_size": 12, "n_advance": 2, "payout": np.zeros(12)},
              {"round": 2, "pod_size": 12, "n_advance": 0,
               "payout": np.r_[np.full(4, 30.0), np.zeros(8)]}]
    ref = R.field_reference(field.astype(np.float32), "t", rounds=rounds, entry_fee=10.0)
    weak = S.portfolio_outcome(field[:3] - 5.0, ref, np.random.default_rng(1), draws=200)
    strong = S.portfolio_outcome(field[:3] + 5.0, ref, np.random.default_rng(1), draws=200)
    assert strong["p_advance"] > weak["p_advance"]
    assert strong["roi"] > weak["roi"]


def test_roster_divergence_reads_identical_portfolios_as_identical():
    """The two ends, plus the control that makes the middle readable.

    `cross_overlap` is a mean over *every* pair, so a portfolio of two different rosters
    compared against itself reads 0.5 and not 1.0 — which is exactly why Gate D reports the
    within-tier overlap beside the cross-tier one instead of asking whether the cross figure
    is "high".
    """
    same = np.tile(np.arange(16), (2, 1))
    assert S.roster_divergence(same, same)["cross_overlap"] == 1.0
    a = np.arange(32).reshape(2, 16)
    assert S.roster_divergence(a, a)["cross_overlap"] == 0.5
    assert S.roster_divergence(a, a)["within_a"] == 0.0
    b = np.arange(100, 132).reshape(2, 16)
    assert S.roster_divergence(a, b)["cross_overlap"] == 0.0
    assert S.roster_divergence(a, b)["jaccard_pools"] == 0.0


# ── 7. Selection and the shipped artifact ─────────────────────────────────────

def test_selection_reads_lift_and_breaks_ties_toward_the_lower_alpha():
    """A strategy has to *earn* the market weight it carries."""
    table = pd.DataFrame({
        "tournament": ["t"] * 3, "strategy": ["a", "b", "c"],
        "alpha": [0.8, 0.2, 0.5], "objective": ["ranking"] * 3,
        "lift_vs_null": [0.05, 0.05, 0.04], "p_any_advance": [0.5, 0.5, 0.9]})
    assert S.select(table, "t") == "b"


def test_spearman_handles_the_recalibrations_plateaus():
    """`adp_dk_scale` is 58 values over 253 ranks, so average ranks are not optional."""
    x = np.array([1.0, 1.0, 1.0, 2.0, 3.0])
    y = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    assert S.spearman(x, y) == pytest.approx(
        float(np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]))
    assert S.spearman(y, y) == pytest.approx(1.0)


def test_the_truth_sims_are_thinned_rather_than_a_prefix():
    """Sim `s` uses posterior draw `s % n_draws`, so a prefix is one contiguous stretch of
    the posterior masquerading as a sample of worlds."""
    idx = S.truth_sim_index(2000, 24)
    assert idx.min() == 0 and idx.max() == 1999
    assert len(idx) == 24
    assert np.all(np.diff(idx) > 1)


# ── 9. The execution axis and the need-aware field ────────────────────────────

def test_autodraft_refuses_any_axis_a_static_ranking_cannot_express():
    """Silently dropping the axis would measure a different strategy under the old name."""
    for kwargs in ({"objective": "lineup_value"}, {"alpha_rounds": (0.1, 0.3, 0.6)},
                   {"exposure_cap": 0.5}, {"stacking": 4.0}):
        with pytest.raises(ValueError):
            S.Strategy("x", ranking="blend", alpha=0.3, autodraft=True, **kwargs)
    S.Strategy("ok", ranking="blend", alpha=0.3, autodraft=True)


def test_an_autodraft_arm_is_bound_by_dks_caps_and_the_manual_twin_is_not():
    """The whole difference between the two execution modes, on one roster.

    DK's 8 G / 8 F / 3 C bind a submitted pre-draft ranking and not a manual pick, so a
    model board that loves centres holds more than three of them only when someone clicks.
    """
    room = _room()
    is_c = room.frame["position"].to_numpy() == "C"
    level = np.where(is_c, 100.0, 50.0) - 0.01 * np.arange(len(is_c))
    room.dk_pts = np.ascontiguousarray(
        np.broadcast_to(level[:, None, None], room.dk_pts.shape).astype(np.float32))

    manual, _ = S.draft_portfolio(room, S.Strategy("m"), "t", 1,
                                  np.random.default_rng(0))
    auto, _ = S.draft_portfolio(room, S.Strategy("a", autodraft=True), "t", 1,
                                np.random.default_rng(0))
    assert int(is_c[manual[0]].sum()) > 3
    assert int(is_c[auto[0]].sum()) <= 3
    assert len(set(auto[0].tolist())) == ROSTER_SIZE


def test_a_need_zero_field_nests_the_shipped_adp_field_exactly():
    """`adp_need` at `need_weight = 0` is the pure-ADP field, bitwise.

    Same rng, same draws, a bonus of exactly zero — so the portfolio drafted against it is
    identical. That nesting is what keeps the two Gate B calibrations comparable, and a
    large need weight has to break it or the axis measures nothing. Our arm drafts by ADP
    so it contends with the field for exactly the players the need bonus reorders — a
    model-ranked arm's targets are scattered enough to dodge the reordering entirely.
    """
    room = _room()
    ours = S.Strategy("m", ranking="adp")
    base, _ = S.draft_portfolio(room, ours, "t", 2, np.random.default_rng(0))

    room.seats = ["adp_need"] * room.pod_size
    room.field_cfg = D.FieldConfig(rank_noise_sd=4.0, need_weight=0.0)
    nested, _ = S.draft_portfolio(room, ours, "t", 2, np.random.default_rng(0))
    assert np.array_equal(nested, base)

    room.field_cfg = D.FieldConfig(rank_noise_sd=4.0, need_weight=400.0)
    keen, _ = S.draft_portfolio(room, ours, "t", 2, np.random.default_rng(0))
    assert not np.array_equal(keen, base)
