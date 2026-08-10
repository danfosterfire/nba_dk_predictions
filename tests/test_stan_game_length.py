"""Tests for the game-length head — the overtime draw a forward simulation needs.

None of these needs a sampler. The head's two fits are `betabinomial_glm.stan` and
`betageometric_duration.stan` with different data, and both are already exercised by
`tests/test_stan_heads.py` and `tests/test_games_played.py`; what is new here is the frame
construction, the floor, the scoring decomposition and — above all — **the draw**.

The draw is where the load-bearing failures live, and they are all silent. A game length
drawn per team-game rather than per game gives a correct marginal and a wrong joint. A
frailty applied per game rather than per season gives correctly-sized seasons with no era
wander. A grid step of 1 instead of 5 gives plausible-looking minutes. Every one of those
leaves the fitted probabilities in this module exactly right, so they are pinned here
rather than left to the run's printout.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.models import held_out, stan_game_length as G


# ── Synthetic builders ────────────────────────────────────────────────────────

def _games(per_season=200, seasons=("2019-20", "2020-21", "2021-22", "2022-23",
                                    "2023-24", "2024-25"), ot_rate=0.06, seed=0):
    """One row per game, with a depth drawn from a geometric so the tail is real."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for i in range(per_season):
            depth = 0
            if rng.random() < ot_rate:
                depth = int(rng.geometric(0.86))
            rows.append({"season": season, "game_id": hash((season, i)) % 10 ** 8,
                         "n_overtimes": depth,
                         G.SEASON_COL: float(season[:4]),
                         "ot": int(depth >= 1),
                         G.MATCHUP_COL: float(rng.uniform(0, 20))})
    return pd.DataFrame(rows)


def _inputs(p_ot=0.06, rho=1e-4, mu_depth=0.86, kappa=38.0, n_draws=8, n_rows=1):
    """The `draw_inputs` contract, filled by hand so a draw can be checked against it."""
    return {"p_ot": np.full((n_draws, n_rows), float(p_ot)),
            "rho": np.full(n_draws, float(rho)),
            "mu_depth": np.full(n_draws, float(mu_depth)),
            "kappa_depth": np.full(n_draws, float(kappa))}


# ── Collapsing to the fitting frames ──────────────────────────────────────────

def test_season_cells_preserve_every_game_and_every_overtime():
    frame = _games()
    cells = G.overtime_cells(frame)
    assert len(cells) == frame["season"].nunique()
    assert int(cells["n"].sum()) == len(frame)
    assert int(cells["y"].sum()) == int((frame["n_overtimes"] >= 1).sum())
    # The collapse is exact for the mean, which is the whole reason 35,546 games can be
    # fitted as ~30 rows.
    assert np.isclose(cells["y"].sum() / cells["n"].sum(), frame["ot"].mean())


def test_matchup_cells_keep_trials_above_one_so_rho_stays_identified():
    frame = _games()
    edges = G.matchup_edges(frame, n_bins=5)
    cells = G.overtime_cells(frame, (G.SEASON_COL, G.MATCHUP_COL), edges)
    assert int(cells["n"].sum()) == len(frame)
    # A beta-binomial at n = 1 is a Bernoulli and its dispersion is unidentified; binning
    # exists precisely to keep that from happening.
    assert (cells["n"] > 1).all()
    assert len(cells) > frame["season"].nunique()


def test_matchup_cells_refuse_to_bin_without_train_fitted_edges():
    with pytest.raises(ValueError):
        G.overtime_cells(_games(), (G.SEASON_COL, G.MATCHUP_COL), edges=None)


def test_matchup_edges_come_from_the_frame_they_are_handed():
    frame = _games()
    train = frame[frame["season"] < "2022-23"]
    edges = G.matchup_edges(train, n_bins=4)
    assert len(edges) == 3
    assert edges.min() >= train[G.MATCHUP_COL].min()
    assert edges.max() <= train[G.MATCHUP_COL].max()


def test_depth_rows_start_at_one_and_carry_no_censoring():
    frame = _games()
    rows = G.depth_rows(frame)
    assert (rows["t"] >= 1).all()
    assert rows["w"].sum() == float((frame["n_overtimes"] >= 1).sum())
    # A game that finished, finished — so `betageometric_duration.stan`'s `H_open = 0`
    # branch disables the in-progress offset exactly, and nothing is right-censored.
    assert (rows["censored"] == 0).all() and (rows["truncated"] == 0).all()


# ── The no-fit floor ──────────────────────────────────────────────────────────

def test_floor_reproduces_the_retired_point_mle():
    """The exact case `tests/test_stan_composition.py` pinned for `fit_ot_tail`.

    The floor is that function moved, not rewritten, so the numbers it produced have to
    survive the move — otherwise the gate would be measured against a different incumbent.
    """
    frame = pd.DataFrame({"season": ["2018-19"] * 100,
                          "n_overtimes": [0] * 94 + [1] * 5 + [2] * 1})
    floor = G.FloorGameLength().fit(frame)
    assert abs(floor.p_any_ot - 0.06) < 1e-12
    assert abs(floor.p_more_ot - 1 / 6) < 1e-12
    assert floor.n_games == 100 and floor.n_ot_games == 6


def test_floor_depth_pmf_is_a_distribution_over_the_support():
    floor = G.FloorGameLength().fit(_games())
    pmf = np.exp(floor.log_depth_pmf(np.arange(1, 200)))
    assert np.isclose(pmf.sum(), 1.0, atol=1e-9)
    assert np.isclose(pmf[0], 1 - floor.p_more_ot)


# ── The scoring decomposition ─────────────────────────────────────────────────

def test_log_likelihood_factorizes_into_onset_and_depth():
    k = np.array([0, 1, 2])
    p = np.array([0.06, 0.06, 0.06])
    log_depth = np.log([0.5, 0.5, 0.25])
    onset, depth = G.log_lik(p, log_depth, k)
    assert np.isclose(onset[0], np.log(0.94))
    assert np.isclose(onset[1], np.log(0.06))
    # A regulation game contributes no depth term, because there is no depth to score.
    assert depth[0] == 0.0
    assert np.isclose(depth[2], np.log(0.25))
    assert np.isclose((onset + depth).sum(),
                      np.log(0.94) + np.log(0.06 * 0.5) + np.log(0.06 * 0.25))


def test_class_counts_are_a_partition_of_the_slate():
    frame = _games()
    floor = G.FloorGameLength().fit(frame)
    pmf = np.exp(floor.log_depth_pmf(np.arange(1, 4)))
    table = G.ppc(floor.p_ot(frame), {1: pmf[0], 2: pmf[1], 3: pmf[2]},
                  frame["n_overtimes"].to_numpy(int), "floor")
    assert list(table["class"]) == list(G.PPC_CLASSES)
    assert int(table["observed"].sum()) == len(frame)
    assert np.isclose(table["predicted"].sum(), len(frame))


def test_bootstrap_interval_brackets_a_real_margin_and_straddles_a_null():
    base = np.full(500, -0.2)
    assert G.bootstrap_delta(base + 0.5, base, seed=1)[0] > 0
    lo, hi = G.bootstrap_delta(base + np.random.default_rng(3).normal(0, 1, 500),
                               base, seed=1)
    assert lo < 0 < hi


# ── The draw — one per game, shared by both teams ─────────────────────────────

def test_draw_lands_on_the_five_minute_grid_and_never_below_regulation():
    drawn = G.sample_game_length(np.random.default_rng(0), 4000,
                                 _inputs(p_ot=0.3), draw=0)
    assert drawn.min() >= G.REGULATION_MINUTES
    residual = (drawn - G.REGULATION_MINUTES) % G.OVERTIME_MINUTES
    assert np.allclose(residual, 0.0)


def test_draw_hits_the_requested_overtime_rate():
    drawn = G.sample_game_length(np.random.default_rng(0), 20000,
                                 _inputs(p_ot=0.06, rho=1e-6), draw=0)
    assert abs((drawn > G.REGULATION_MINUTES).mean() - 0.06) < 0.01


def test_draw_reproduces_the_depth_distribution_it_was_given():
    drawn = G.sample_game_length(np.random.default_rng(0), 60000,
                                 _inputs(p_ot=1.0, rho=1e-6, mu_depth=0.86,
                                         kappa=1e6), draw=0)
    depth = np.rint((drawn - G.REGULATION_MINUTES) / G.OVERTIME_MINUTES).astype(int)
    # kappa -> infinity IS the plain geometric, so P(T = 1) must come back as mu.
    assert abs((depth == 1).mean() - 0.86) < 0.01
    assert abs((depth == 2).mean() - 0.86 * 0.14) < 0.01


def test_one_length_per_game_not_per_team():
    """The rule the whole module exists to protect.

    The function returns exactly `n_games` values and the caller shares each across both
    teams. A per-team-game draw would give the right marginal and destroy the correlation a
    same-team stack is drafted for — so the shape is pinned rather than assumed.
    """
    n_games = 137
    drawn = G.sample_game_length(np.random.default_rng(0), n_games, _inputs(), draw=0)
    assert drawn.shape == (n_games,)


def test_the_season_frailty_is_shared_across_the_slate():
    """`rho` must move whole slates together, not individual games.

    With a large `rho` the season-to-season spread in the OT count has to blow past
    binomial; with `rho` at its floor it has to sit on it. Both directions are checked,
    because a frailty drawn per *game* would reproduce the marginal and show binomial
    spread — the failure that looks right in every other statistic here.
    """
    def spread(rho):
        rng = np.random.default_rng(0)
        inputs = _inputs(p_ot=0.06, rho=rho, n_draws=300)
        counts = [(G.sample_game_length(rng, 1230, inputs, draw=s)
                   > G.REGULATION_MINUTES).sum() for s in range(300)]
        return float(np.std(counts))

    binomial = np.sqrt(1230 * 0.06 * 0.94)
    assert spread(1e-6) < binomial * 1.35
    assert spread(0.02) > binomial * 2.0


def test_the_draw_is_reproducible_and_the_posterior_draw_selects():
    inputs = _inputs(p_ot=0.06, n_draws=8)
    inputs["p_ot"][3] = 0.9
    first = G.sample_game_length(np.random.default_rng(7), 500, inputs, draw=0)
    again = G.sample_game_length(np.random.default_rng(7), 500, inputs, draw=0)
    assert np.array_equal(first, again)
    high = G.sample_game_length(np.random.default_rng(7), 500, inputs, draw=3)
    assert (high > G.REGULATION_MINUTES).mean() > 0.7
    # Out-of-range indices wrap rather than raising: a simulator running more seasons than
    # it kept posterior draws is a legitimate configuration.
    assert len(G.sample_game_length(np.random.default_rng(7), 10, inputs, draw=11)) == 10


def test_per_game_probabilities_are_honoured_when_the_arm_carries_a_covariate():
    inputs = _inputs(p_ot=0.0, n_draws=4, n_rows=200)
    inputs["p_ot"][:, :100] = 1.0
    inputs["p_ot"][:, 100:] = 1e-9
    drawn = G.sample_game_length(np.random.default_rng(0), 200, inputs, draw=0)
    assert (drawn[:100] > G.REGULATION_MINUTES).all()
    assert (drawn[100:] == G.REGULATION_MINUTES).all()


# ── The forward frame ─────────────────────────────────────────────────────────

def test_forward_cells_extrapolate_the_season_index_with_no_outcomes():
    cells = G.forward_cells("2026-27", n_games=1230, matchup_gap=7.5)
    assert len(cells) == 1230
    assert (cells[G.SEASON_COL] == 2026.0).all()
    assert (cells[G.MATCHUP_COL] == 7.5).all()
    # A forward frame must carry no realized outcome at all — the production board is built
    # in September, before any of these games exists.
    assert "n_overtimes" not in cells.columns and "ot" not in cells.columns


# ── The split ─────────────────────────────────────────────────────────────────

def test_selection_never_sees_the_last_two_seasons():
    """The head's frame is games, not player-seasons, and the guard has to survive that.

    `selection_split` sorts the season labels and drops the last two twice over, so the
    unit it is handed does not matter — but this head is the first to hand it a *game*
    frame, and a guard that quietly stopped applying is exactly the failure
    `src/models/held_out.py` was written after.
    """
    frame = _games()
    train, val = held_out.selection_split(frame, test_seasons=2)
    assert set(val["season"]) == {"2021-22", "2022-23"}
    assert set(train["season"]) == {"2019-20", "2020-21"}
    assert "2023-24" not in set(train["season"]) | set(val["season"])
    assert "2024-25" not in set(train["season"]) | set(val["season"])


# ── The two producers of one contract ─────────────────────────────────────────

def test_the_artifact_path_and_the_live_head_path_agree():
    """`posterior_inputs` (from pickles) must equal `draw_inputs` (from live heads).

    Two producers of one dict is how a simulator ends up drawing from something subtly
    different from what was fitted. The heads here carry injected draws rather than sampled
    ones — the fit is not what this checks — and the artifact is built through the real
    `posteriors._finish`, so its recipe, scaler and round-trip are the shipped ones.
    """
    from src.models import posteriors as P
    from src.models.games_played import MU_MAX, MU_MIN
    from src.models.stan_games_played import BetaBinomialHead, BetaGeometricHead

    rng = np.random.default_rng(0)
    frame = _games()
    cells = G.overtime_cells(frame)
    depth_frame = G.depth_rows(frame)
    features = [G.SEASON_COL]
    n = 16

    onset = BetaBinomialHead.__new__(BetaBinomialHead)
    onset.features, onset.rho = features, 1e-4
    onset.scaler = StandardScaler().fit(cells[features].to_numpy(float))
    onset.alpha_draws = rng.normal(-2.7, 0.05, n)
    onset.beta_draws = rng.normal(0, 0.05, (n, 1))
    onset.rho_draws = np.full(n, 1e-4)

    depth = BetaGeometricHead.__new__(BetaGeometricHead)
    depth.features, depth.scaler = [], None
    depth.alpha_draws = rng.normal(1.8, 0.05, n)
    depth.beta_draws = np.zeros((n, 0))
    depth.kappa_draws = np.full(n, 38.0)

    ot_art = P._finish(
        head="game_length_ot", head_label="overtime onset", family="betabinomial",
        response="plug_in_mu", variant="season_trend", features=features, model=onset,
        fit_frame=cells, probe_transformed=cells, probe_raw=cells, steps=[],
        builder="test", extras={"successes": "y", "trials": "n",
                                "dispersion": "rho_draws"},
        window="train", cfg_stan={}, draws_kept=n, seconds=0.0)
    depth_art = P._finish(
        head="game_length_depth", head_label="overtime depth", family="betageometric",
        response="mean_mu", variant="season_trend", features=[], model=depth,
        fit_frame=depth_frame, probe_transformed=depth_frame, probe_raw=depth_frame,
        steps=[], builder="test", extras={"length": "t", "weight": "w",
                                          "dispersion": "kappa_draws",
                                          "mu_clip": (MU_MIN, MU_MAX)},
        window="train", cfg_stan={}, draws_kept=n, seconds=0.0)

    live = G.draw_inputs(onset, depth, cells)
    stored = G.posterior_inputs(ot_art, depth_art, cells)
    for key in ("p_ot", "rho", "mu_depth", "kappa_depth"):
        assert np.allclose(live[key], stored[key], atol=1e-12), key
    # And the draw itself has to be indifferent to which one it was handed.
    assert np.array_equal(
        G.sample_game_length(np.random.default_rng(1), len(cells), live, draw=0),
        G.sample_game_length(np.random.default_rng(1), len(cells), stored, draw=0))


def test_a_cell_frame_handed_to_the_draw_raises_instead_of_truncating():
    """The one wiring error that produces a plausible season out of the wrong frame.

    `p_ot` built from the collapsed *cells* carries ~30 probabilities; a 1,230-game slate
    needs 1,230 or one. Silently taking the first 1,230 of 30 is a broadcast error today
    and would be a quietly wrong simulation the moment the shapes happened to line up.
    """
    inputs = _inputs(n_rows=30)
    with pytest.raises(ValueError, match="one per game"):
        G.sample_game_length(np.random.default_rng(0), 1230, inputs, draw=0)
