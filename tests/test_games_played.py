import numpy as np
import pandas as pd
import pytest

from src.features.availability import (collapse_transitions, spell_classes,
                                       tenure_frame)
from src.models.games_played import (KAPPA_MAX, beta_geometric_logpmf,
                                     beta_geometric_logsf, beta_shapes,
                                     closed_form_calibration, duration_rows,
                                     fit_beta_geometric, gp_pmf, simulate_gp,
                                     variance_inflation)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _panel(sequences: dict[tuple, str], season: str = "2021-22") -> pd.DataFrame:
    """A panel from `{(player_id, team_id): "PPMMP"}` strings — P played, M missed.

    Names stay distinct after any normalization for the reason `report_calibration`'s
    fixture had to: a builder whose keys collide silently exercises a merge that attaches
    one row's outcome to many and still passes on the row count.
    """
    rows = []
    for (player_id, team_id), pattern in sequences.items():
        played = [1 if c == "P" else 0 for c in pattern]
        first = played.index(1) if 1 in played else None
        last = len(played) - 1 - played[::-1].index(1) if 1 in played else None
        for i, p in enumerate(played):
            rows.append({
                "season": season, "season_start_year": 2021,
                "player_id": player_id, "team_id": team_id,
                "game_id": 100 + i, "game_date": pd.Timestamp("2021-10-19")
                + pd.Timedelta(days=2 * i),
                "team_game_index": i, "min": 20.0 if p else np.nan, "played": p,
                "in_appearance_window": int(first is not None
                                            and first <= i <= last),
                "status_covered": 1, "status": "played" if p else "inactive",
                "missed_reason": "",
            })
    return pd.DataFrame(rows)


def _game_level_loglik(pattern: str, h: float, r: float) -> float:
    """The explicit product over transitions — no collapse, no sufficient statistics."""
    played = [c == "P" for c in pattern]
    total = 0.0
    for prev, now in zip(played[:-1], played[1:]):
        if prev:
            total += np.log(h) if not now else np.log1p(-h)
        else:
            total += np.log(r) if now else np.log1p(-r)
    return total


def _collapsed_loglik(counts: pd.Series, h: float, r: float) -> float:
    a = counts["onsets"]
    b = counts["at_risk_played"] - a
    c = counts["recoveries"]
    d = counts["at_risk_missed"] - c
    return (a * np.log(h) + b * np.log1p(-h)
            + c * np.log(r) + d * np.log1p(-r))


# ── 1. The test that makes the whole design legitimate ────────────────────────

def test_collapse_reproduces_the_game_level_likelihood():
    """Four counts are sufficient: the collapsed likelihood IS the game-level one.

    Not "close to" — equal, because the features are constant within a player-season so
    the per-transition probabilities are too. Everything in this head rests on it.
    """
    patterns = {(1, 10): "PPMMPPPMPPPPMPP", (2, 10): "PMPMPMPPPPMMMPP",
                (3, 20): "MMPPPPPPMPPPPPM"}
    counts = collapse_transitions(_panel(patterns), "full").set_index("player_id")

    for h, r in [(0.1, 0.3), (0.42, 0.07), (0.005, 0.9)]:
        for player_id, pattern in [(1, patterns[(1, 10)]), (2, patterns[(2, 10)]),
                                   (3, patterns[(3, 20)])]:
            explicit = _game_level_loglik(pattern, h, r)
            collapsed = _collapsed_loglik(counts.loc[player_id], h, r)
            assert abs(explicit - collapsed) < 1e-12


# ── 2. What the collapse quietly loses ────────────────────────────────────────

def test_collapse_is_conditional_on_the_initial_state():
    """The first game contributes no transition, so `s0` needs its own head.

    Two sequences that differ only in their opening state give the identical four counts
    — which is exactly why the shipped frame is a tenure decomposition: a tenure opens on
    an appearance by construction, so the initial state is known rather than modelled.
    """
    counts = collapse_transitions(
        _panel({(1, 10): "PMMPP", (2, 10): "MMMPP"}), "full").set_index("player_id")
    quartet = ["onsets", "at_risk_played", "recoveries", "at_risk_missed"]
    # The opening state differs; the transition counts do not.
    assert counts.loc[1, "initial_state"] != counts.loc[2, "initial_state"]
    assert not counts.loc[1, quartet].equals(counts.loc[2, quartet])

    # The sharper version: append the same suffix to a shared body and only the head
    # differs, so the likelihood cannot distinguish them.
    same = collapse_transitions(
        _panel({(1, 10): "PMPP", (2, 10): "MMPP"}), "full").set_index("player_id")
    for h, r in [(0.2, 0.4), (0.6, 0.1)]:
        gap = (_collapsed_loglik(same.loc[1], h, r)
               - _collapsed_loglik(same.loc[2], h, r))
        explicit = (_game_level_loglik("PMPP", h, r)
                    - _game_level_loglik("MMPP", h, r))
        assert abs(gap - explicit) < 1e-12


# ── 3-5. The beta-geometric ───────────────────────────────────────────────────

def test_beta_geometric_pmf_sums_to_one_and_survival_is_its_own_tail():
    """`sum_{t<=K} P(T=t) + P(T>K) == 1` exactly, at every K.

    Stated with the survival term rather than as a bare sum to a large K, because at
    `a < 1` the tail is polynomial rather than geometric — `P(T >= t) ~ t^-a` — and 40,000
    terms capture only 73% of the mass. That is a property of the distribution, not a
    numerical problem, and it is the same fact as `mean_defined`: with proper censoring the
    full-window fit lands at a = 0.72 and has no finite mean at all.
    """
    for mu, kappa in [(0.48, 3.69), (0.10, 1.2), (0.75, 40.0), (0.4242, 1.6988)]:
        assert abs(np.exp(beta_geometric_logpmf(1, mu, kappa)) - mu) < 1e-12
        for K in (1, 5, 26, 500):
            t = np.arange(1, K + 1)
            head = np.exp(beta_geometric_logpmf(t, mu, kappa)).sum()
            tail = np.exp(beta_geometric_logsf(K + 1, mu, kappa))
            assert abs(head + tail - 1.0) < 1e-9
            # And the survival is its own tail sum wherever that sum converges usefully.
            assert abs(np.exp(beta_geometric_logsf(K, mu, kappa))
                       - (1.0 - np.exp(beta_geometric_logpmf(
                           np.arange(1, K), mu, kappa)).sum())) < 1e-9

    # E[T] = (a+b-1)/(a-1), and only for a > 1. The full-window censored fit lands below
    # that, which is why `mean_defined` exists rather than a quoted number.
    mu, kappa = 0.48, 3.69
    a, b = beta_shapes(mu, kappa)
    t = np.arange(1, 4_000_001)
    numeric = float((t * np.exp(beta_geometric_logpmf(t, mu, kappa))).sum())
    assert abs(numeric - (a + b - 1) / (a - 1)) < 1e-2


def test_right_censoring_recovers_a_known_shape_where_ignoring_it_does_not():
    """Assert the DIRECTION of the naive fit's bias, which is what makes it dangerous.

    Dropping censored spells and treating them as complete both converge, both return
    finite parameters, and both understate the tail. Nothing about either failure is loud.
    """
    rng = np.random.default_rng(0)
    mu_true, kappa_true = 0.35, 2.5
    a, b = beta_shapes(mu_true, kappa_true)
    q = np.clip(rng.beta(a, b, size=40_000), 1e-4, 1.0)
    t = rng.geometric(q)

    horizon = 20
    censored = (t > horizon).astype(float)
    observed = np.minimum(t, horizon)

    proper = fit_beta_geometric(observed, censored=censored)
    complete = fit_beta_geometric(observed)
    dropped = fit_beta_geometric(observed[censored == 0])

    truth = float(np.exp(beta_geometric_logsf(26, mu_true, kappa_true)))
    p = {name: float(np.exp(beta_geometric_logsf(26, f["mu"], f["kappa"])))
         for name, f in [("proper", proper), ("complete", complete),
                         ("dropped", dropped)]}
    assert abs(p["proper"] / truth - 1.0) < 0.15
    assert p["complete"] < truth
    assert p["dropped"] < p["complete"]


def test_left_truncated_residual_is_the_size_biased_beta_geometric():
    """A spell in progress has frailty Beta(a-1, b) — the identity `open_shift` is measured
    against.

    **The selection has to be length-biased, and that is the whole content of the
    identity.** A fixed time point — game 1 of the season — falls inside a spell with
    probability proportional to that spell's length, so the frailty of the spell you catch
    in progress is size-biased by `E[T|q] = 1/q`, which turns Beta(a, b) into Beta(a-1, b).
    Sampling spells *uniformly* instead reproduces the base distribution and the test fails
    by 0.29 in the first cell, which is how this was caught. It needs a > 1, since
    length-biasing is normalizable only then.
    """
    rng = np.random.default_rng(1)
    mu, kappa = 0.45, 6.0
    a, b = beta_shapes(mu, kappa)
    assert a > 1.0

    n = 400_000
    q = np.clip(rng.beta(a, b, size=n), 1e-4, 1.0)
    length = rng.geometric(q)
    caught = rng.choice(n, size=n, p=length / length.sum())
    # A uniformly chosen game inside the spell you caught, then the duration remaining.
    picked = length[caught]
    residual = picked - (rng.random(n) * picked).astype(int)

    grid = np.arange(1, 60)
    empirical = np.array([(residual == k).mean() for k in grid])
    mu_biased = (a - 1.0) / (a - 1.0 + b)
    predicted = np.exp(beta_geometric_logpmf(grid, mu_biased, kappa - 1.0))
    assert np.abs(empirical - predicted).max() < 0.01


# ── 6. The arithmetic that would move every figure in the censoring table ─────

def test_spell_classes_partition_the_missed_games():
    panel = _panel({(1, 10): "MMPPMPPMMM", (2, 10): "PPMMPP",
                    (3, 20): "MPMPMPMPMP"})
    spells = spell_classes(panel, "full")

    for (season, player_id, team_id), grp in spells.groupby(
            ["season", "player_id", "team_id"]):
        cell = panel[(panel["player_id"] == player_id)
                     & (panel["team_id"] == team_id)]
        assert grp["spell_games"].sum() == int((cell["played"] == 0).sum())
        # Every spell lands in exactly one class.
        assert set(grp["spell_class"]) <= {"interior", "left_truncated",
                                           "right_censored"}
        assert (grp["truncated"] + grp["censored"] <= 1).all()

    by_class = spells.groupby("spell_class")["spell_games"].sum()
    assert by_class.sum() == int((panel["played"] == 0).sum())

    # On the appearance window there are no edge spells at all — the window opens and
    # closes on a game he played, which is what removes censoring from the within-tenure
    # duration head rather than requiring it to be modelled.
    inside = spell_classes(panel, "appearance")
    assert (inside["spell_class"] == "interior").all()


# ── 7-8. The simulator ────────────────────────────────────────────────────────

def _flat_params(n_rows: int, hazard: float, mu_dur: float, kappa_dur: float,
                 pre=None, post=None) -> dict:
    onset_a, onset_b = beta_shapes(np.full(n_rows, hazard), np.full(n_rows, KAPPA_MAX))
    dur_a, dur_b = beta_shapes(np.full(n_rows, mu_dur), np.full(n_rows, kappa_dur))
    params = {"onset_a": onset_a, "onset_b": onset_b, "dur_a": dur_a, "dur_b": dur_b}
    if pre is not None:
        params["pre"], params["post"] = pre, post
    return params


def test_departure_is_absorbing_in_the_simulator():
    """Post-tenure games are missed, always — a departure is a hitting time, not a low
    recovery rate. That distinction is the reason the frame is a tenure decomposition."""
    n, T = 200, 82
    pre = np.zeros(n, dtype=np.int64)
    post = np.full(n, 30, dtype=np.int64)
    sims = simulate_gp(np.full(n, T), _flat_params(n, 0.1, 0.45, 3.7, pre, post),
                       n_sims=20, seed=3)
    assert sims.max() <= T - post[0]
    # And the last tenure game is always played, so GP is at least 1.
    assert sims.min() >= 1

    # A player who never leaves can reach the full schedule.
    open_ended = simulate_gp(np.full(n, T),
                             _flat_params(n, 0.02, 0.9, KAPPA_MAX,
                                          pre, np.zeros(n, dtype=np.int64)),
                             n_sims=20, seed=4)
    assert open_ended.max() == T


def test_gp_pmf_sums_to_one_and_is_supported_on_0_to_team_games():
    n, T = 50, 60
    pre = np.zeros(n, dtype=np.int64)
    post = np.zeros(n, dtype=np.int64)
    sims = simulate_gp(np.full(n, T), _flat_params(n, 0.15, 0.45, 3.7, pre, post),
                       n_sims=200, seed=5)
    pmf = gp_pmf(sims, max_games=82)
    assert np.allclose(pmf.sum(axis=1), 1.0)
    assert (pmf >= 0).all()
    # Mass above this row's own schedule is only the Monte Carlo pseudo-count, which is
    # what the regularization is worth: <= 0.5 spread over 83 cells out of 10,000 draws.
    assert pmf[:, T + 1:].sum(axis=1).max() < 1e-3


# ── 9. The multiplicity collapse ──────────────────────────────────────────────

def test_collapsed_multiplicity_weights_match_the_uncollapsed_likelihood():
    rng = np.random.default_rng(7)
    spells = pd.DataFrame({
        "spell_games": rng.integers(1, 15, size=3000),
        "censored": rng.integers(0, 2, size=3000),
        "truncated": 0,
    })
    collapsed = duration_rows(spells)
    assert collapsed["w"].sum() == len(spells)
    assert len(collapsed) < len(spells) / 10

    from src.models.games_played import beta_geometric_loglik
    for mu, kappa in [(0.4, 3.0), (0.2, 9.0)]:
        full = beta_geometric_loglik(spells["spell_games"].to_numpy(), mu, kappa,
                                     spells["censored"].to_numpy()).sum()
        weighted = float(np.dot(
            collapsed["w"].to_numpy(),
            beta_geometric_loglik(collapsed["t"].to_numpy(), mu, kappa,
                                  collapsed["censored"].to_numpy())))
        assert abs(full - weighted) < 1e-8


# ── 10. The identity the dispersion argument rests on ─────────────────────────

def test_closed_form_calibration_inverts_the_variance_identity():
    """`inflation = C + rho*(n - C)` round-trips, and it is ADDITIVE.

    The multiplicative misreading — "22.7 / 3.96, so the frailty must supply 5.7x" — is
    the specific error this guards, so the test asserts the two answers differ.
    """
    n, clustering = 82.0, 9.581
    for inflation in (22.70, 15.0, 30.0):
        cal = closed_form_calibration(np.array([0.8]), np.array([inflation]),
                                      clustering, n)
        rho = float(cal["rho_frailty"][0])
        assert abs(variance_inflation(clustering, rho, n) - inflation) < 1e-9

    # C = 1 is "no clustering", where the identity must return the plain beta-binomial.
    for rho in (0.05, 0.2757):
        assert abs(variance_inflation(1.0, rho, n) - (1 + (n - 1) * rho)) < 1e-12

    # The recorded multiplicative reading gives a different — and much larger — answer.
    additive = (22.70 - clustering) / (n - clustering)
    multiplicative = (22.70 / clustering - 1.0) / (n - 1.0)
    assert additive > multiplicative

    # Stacking the incumbent's rho on the measured clustering overshoots, which is the
    # plan's "there is no dispersion hole to fill, there is a surplus to avoid".
    assert variance_inflation(clustering, 0.2757, n) > 22.70


def test_calibrated_simulator_reproduces_the_marginal_it_is_given():
    """Option (b) end to end: the algebra above, actually run.

    The mean is exact by construction — a two-state chain with `h = (1-mu)(1-rho_M)` and
    `r = mu(1-rho_M)` has stationary play rate exactly `mu`. The variance lands ~2% low at
    C > 1 because `(1+rho)/(1-rho)` is the *asymptotic* inflation and a season is 82 games,
    not infinitely many; at C = 1 there is no clustering term and it is exact.
    """
    from src.models.games_played import simulate_calibrated

    n_games, clustering = 82.0, 9.5806
    for target, c, tol in [(23.33, clustering, 0.05), (15.0, clustering, 0.06),
                           (23.33, 1.0, 0.02)]:
        # Held at one mu: the identity is about the CONDITIONAL distribution, so pooling a
        # heterogeneous population would fold between-player mean spread into the check.
        sims = simulate_calibrated(np.full(4000, 82), np.full(4000, 0.80), target, c,
                                   60, seed=2)
        share = (sims / 82).reshape(-1)
        p = share.mean()
        inflation = share.var() / (p * (1 - p) / n_games)
        assert abs(p - 0.80) < 0.01
        assert abs(inflation / target - 1.0) < tol


# ── 11. The chain itself ──────────────────────────────────────────────────────

def test_two_state_chain_reproduces_a_known_transition_matrix():
    """A geometric duration IS the two-state chain, so the simulator must recover both
    hazards from its own output."""
    n, T = 400, 82
    h, r = 0.12, 0.30
    pre = np.zeros(n, dtype=np.int64)
    post = np.zeros(n, dtype=np.int64)
    # kappa -> infinity collapses the beta-geometric to the plain geometric at mu = r.
    sims = simulate_gp(np.full(n, T), _flat_params(n, h, r, KAPPA_MAX, pre, post),
                       n_sims=200, seed=11)
    # Stationary play rate of the chain, allowing for the two forced endpoints.
    stationary = r / (h + r)
    assert abs(sims.mean() / T - stationary) < 0.02


# ── 12. The Stan port ─────────────────────────────────────────────────────────

@pytest.mark.slow
def test_stan_beta_geometric_recovers_known_parameters():
    cmdstanpy = pytest.importorskip("cmdstanpy")
    try:
        cmdstanpy.cmdstan_path()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CmdStan toolchain")

    from src.models.stan_games_played import BetaGeometricHead

    rng = np.random.default_rng(13)
    mu, kappa = 0.42, 4.0
    a, b = beta_shapes(mu, kappa)
    q = np.clip(rng.beta(a, b, size=30_000), 1e-4, 1.0)
    t = rng.geometric(q)
    rows = duration_rows(pd.DataFrame({"spell_games": t, "censored": 0,
                                       "truncated": 0}))

    head = BetaGeometricHead([], "test/duration", chains=2, warmup=300,
                             samples=300, seed=13).fit(rows)
    assert abs(head.mu - mu) < 0.02
    assert abs(head.kappa - kappa) / kappa < 0.20
    assert head.diagnostics["divergences"] == 0


# ── 13. The exclusion, and that it is only an exclusion from FITTING ──────────

def test_multi_team_player_seasons_are_excluded_from_the_fit_but_not_the_evaluation():
    panel = pd.concat([
        _panel({(1, 10): "PPPPMMPPPP"}),
        _panel({(2, 10): "PPPPP", (2, 20): "PPPPP"}),      # traded mid-season
    ], ignore_index=True)

    from src.models.games_played import multi_team_seasons, process_frame
    multi = multi_team_seasons(panel)
    assert ("2021-22", 2) in multi and ("2021-22", 1) not in multi

    process = process_frame(panel)
    assert process[process["player_id"] == 2]["multi_team"].all()
    assert not process[process["player_id"] == 1]["multi_team"].any()

    # The exclusion is from the fit only: the traded player keeps a design row, and the
    # covariates he is predicted from are player-season level, so nothing blocks it.
    fittable = process[~process["multi_team"]]
    assert set(fittable["player_id"]) == {1}
    assert set(process["player_id"]) == {1, 2}


# ── 14. The bug that does not announce itself ─────────────────────────────────

def test_frailty_is_drawn_per_player():
    """`rng.beta(a, b)` without `size=` returns a SCALAR, and that silently gave a whole
    simulated population one shared hazard during planning.

    It does not announce itself — the marginal mean is unchanged and only the *spread*
    collapses — so this asserts the between-player variance of the simulated outcome, which
    is the thing that breaks.
    """
    n, T = 600, 82
    # A wide frailty: mean hazard 0.1 with real spread, so a shared draw is detectable.
    onset_a = np.full(n, 1.0)
    onset_b = np.full(n, 9.0)
    dur_a, dur_b = beta_shapes(np.full(n, 0.45), np.full(n, 3.7))
    params = {"onset_a": onset_a, "onset_b": onset_b, "dur_a": dur_a, "dur_b": dur_b,
              "pre": np.zeros(n, dtype=np.int64), "post": np.zeros(n, dtype=np.int64)}
    sims = simulate_gp(np.full(n, T), params, n_sims=40, seed=17)

    # Every row has identical parameters, so all spread is the frailty and the chain.
    assert sims.shape == (n, 40)
    across = sims.reshape(-1)
    binomial_sd = np.sqrt(T * across.mean() / T * (1 - across.mean() / T))
    assert across.std() > 2.0 * binomial_sd

    # And the frailty is redrawn per simulated season rather than per row: the spread
    # WITHIN a row is of the same order as the spread across rows.
    within = sims.std(axis=1).mean()
    assert within > 0.5 * across.std()

    # The failure mode itself: one shared scalar hazard for everyone.
    rng = np.random.default_rng(0)
    scalar = rng.beta(1.0, 9.0)
    shared = {**params,
              "onset_a": np.full(n, scalar * KAPPA_MAX),
              "onset_b": np.full(n, (1 - scalar) * KAPPA_MAX)}
    collapsed = simulate_gp(np.full(n, T), shared, n_sims=40, seed=17)
    assert collapsed.std() < across.std()


# ── The tenure identity everything else is built on ───────────────────────────

def test_tenure_factors_reconstruct_games_played_exactly():
    panel = _panel({(1, 10): "MMPPMPPPMM", (2, 10): "PPPPPPPPPP",
                    (3, 20): "MMMMMMMMMP"})
    tenure = tenure_frame(panel)
    assert (tenure["pre_tenure"] + tenure["tenure_games"]
            + tenure["post_tenure"] == tenure["team_games"]).all()
    for _, row in tenure.iterrows():
        cell = panel[(panel["player_id"] == row["player_id"])
                     & (panel["team_id"] == row["team_id"])]
        inside = cell[(cell["team_game_index"] >= row["entry_index"])
                      & (cell["team_game_index"] <= row["exit_index"])]
        assert row["gp"] == int(inside["played"].sum())
        assert row["gp"] == int(cell["played"].sum())


# ── The decoupled hybrid: exact marginal, fitted spell shape ──────────────────

def test_allocate_spells_preserves_the_games_played_count_exactly():
    """The whole point of decoupling: the marginal is not approximated, it is inherited.

    Whatever count goes in comes back out on every row, so every marginal metric — CRPS,
    PIT, MAE, the tail — is exactly the metric of the distribution the count was drawn
    from. Only the *placement* of the absences is this function's business.
    """
    from src.models.games_played import allocate_spells, spell_lengths_from

    rng = np.random.default_rng(3)
    team_games = rng.integers(60, 83, size=400)
    gp = np.array([rng.integers(0, t + 1) for t in team_games])
    played = allocate_spells(gp, team_games, mu=0.4777, kappa=3.7586, seed=5)

    assert (played.sum(axis=1) == gp).all()
    # Nothing is placed past a row's own schedule.
    for i, t in enumerate(team_games):
        assert played[i, t:].sum() == 0

    # And the absences really are spells rather than scattered singles: with a mean spell
    # length above 3, far fewer runs than missed games.
    lens = spell_lengths_from(played, team_games)
    assert lens.sum() == int((team_games - gp).sum())
    assert len(lens) < 0.6 * lens.sum()


def test_allocated_spells_follow_the_fitted_shape_not_a_geometric():
    """A geometric understates a three-week absence by ~43% and a season-ending one by 99%.
    The allocator has to reproduce the fitted beta-geometric's shape, not that.

    **The GP marginal has to be the realistic overdispersed one, and that is a finding
    rather than test hygiene.** A spell cannot be longer than the player's missed total, so
    the spell distribution is *not* independent of the count it is conditioned on: drawn
    from a plain Binomial(82, 0.78) — no overdispersion, everyone missing ~18 games — the
    allocator produces P(spell >= 26) = 0.0000, because no simulated player misses enough
    games for a month-long absence to be possible. Real games played is ~23x overdispersed,
    and it is that left tail which makes long absences representable at all.
    """
    from src.models.games_played import allocate_spells, spell_lengths_from

    rng = np.random.default_rng(4)
    team_games = np.full(4000, 82)
    # Beta-binomial at the incumbent's fitted dispersion, not a binomial.
    rho = 0.2806
    scale = (1 - rho) / rho
    gp = rng.binomial(82, rng.beta(0.78 * scale, 0.22 * scale, size=4000))
    lens = spell_lengths_from(
        allocate_spells(gp, team_games, mu=0.4777, kappa=3.7586, seed=6), team_games)

    # Observed on 68,530 interior spells: 0.4829 single-game, 0.0635 ten-plus, 0.0106 26+.
    assert abs((lens == 1).mean() - 0.4829) < 0.06
    assert (lens >= 10).mean() > 0.04          # a geometric at this mean gives ~0.037
    # The geometric's defining failure is the extreme tail, where it reads 0.0001 against
    # an observed 0.0106. The allocator must not share it.
    assert (lens >= 26).mean() > 0.005
