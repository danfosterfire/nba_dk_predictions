"""Tests for the season simulator — the tensor, the chain order, and the split guard.

None of these needs a posterior artifact or a sampler. The parts of `src/sim/season.py`
worth pinning are the ones whose failures are **silent**: a reordered draw chain still
produces a plausible box score, a stick-breaking allocation that sums to the wrong total
still produces plausible minutes, and a simulator that quietly reads a test season still
produces a plausible board. Every one of those is arithmetic over synthetic arrays, so it
runs in milliseconds.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py`.
"""

import numpy as np
import pandas as pd
import pytest

from src.models import held_out
from src.models import posteriors as P
from src.models.component_rates import CONVERSION_HEADS, COUNT_HEADS, DERIVED_COUNTS
from src.models.held_out import HeldOutLocked, unlocked
from src.sim import season as S


# ── Synthetic builders ────────────────────────────────────────────────────────

def _rates(n_units: int = 3, n_draws: int = 2, seed: int = 0) -> dict:
    """Per-minute rates and conversion probabilities in `component_rates`' shape."""
    rng = np.random.default_rng(seed)
    out = {"count": {}, "conversion": {}, "phi": {}, "rho": {}}
    for i, component in enumerate(COUNT_HEADS):
        out["count"][component] = np.full((n_draws, n_units), 0.02 * (i + 1))
        out["phi"][component] = np.full(n_draws, 40.0)
    for made, attempted in CONVERSION_HEADS:
        head = f"{made}|{attempted}"
        out["conversion"][head] = np.full((n_draws, n_units), 0.4)
        out["rho"][head] = np.full(n_draws, 0.01)
    return out


def _blocks(sizes: list[int]) -> np.ndarray:
    """A sorted block-id array with the given run lengths."""
    return np.repeat(np.arange(len(sizes)), sizes)


def _season_frame(seasons: list[str], n: int = 4) -> pd.DataFrame:
    return pd.DataFrame({"season": np.repeat(seasons, n),
                         "player_id": np.tile(np.arange(n), len(seasons))})


# ── The draw order — the one thing that must fail loudly if reordered ─────────

def test_draw_order_is_the_shot_attempt_chain():
    """`fga -> fg3a|fga -> fg2a -> makes`, with the derived count between them.

    The order is the whole content of the shot-attempt basis: `fg3a` is a *share* of `fga`,
    so `fg2a = fga - fg3a` cannot be materialized before the share head has drawn, and
    `fg2m | fg2a` cannot be drawn before that. Every other basis this project has used had
    every `attempted` column already fitted, so this edge did not exist.
    """
    order = S.draw_order()
    assert order.index("fga") < order.index("fg3a|fga")
    assert order.index("fg3a|fga") < order.index("fg2a")
    assert order.index("fg2a") < order.index("fg2m|fg2a")
    assert order.index("fga") < order.index("fg2a")
    assert set(order) == set(COUNT_HEADS) | set(DERIVED_COUNTS) | {
        f"{m}|{a}" for m, a in CONVERSION_HEADS}
    assert S.DRAW_ORDER == order


def test_draw_order_raises_when_a_head_needs_trials_nothing_produces(monkeypatch):
    """A head list that asks for trials the chain never makes is a build failure.

    This is the reordering guard with teeth: swapping the conversion heads so `fg2m | fg2a`
    comes before the share head that produces `fg2a` must raise rather than silently
    drawing makes on an attempt count that does not exist yet.
    """
    monkeypatch.setattr(S, "CONVERSION_HEADS",
                        [("fg2m", "fg2a"), ("fg3a", "fga")], raising=True)
    with pytest.raises(ValueError, match="nothing earlier in the chain produces it"):
        S.draw_order()


def test_draw_components_materializes_in_the_declared_order():
    """The trace the simulator asserts against is produced by the draw itself."""
    rng = np.random.default_rng(0)
    unit = np.array([0, 1, 2, 0])
    minutes = np.array([30.0, 20.0, 10.0, 25.0])
    rates = _rates()
    season_gamma = {c: np.ones(len(unit)) for c in COUNT_HEADS}
    season_p = {f"{m}|{a}": np.full(len(unit), 0.4) for m, a in CONVERSION_HEADS}
    frailty = np.ones((len(unit), len(COUNT_HEADS)))

    counts, made, trace = S.draw_components(rng, minutes, unit, rates, 0, season_gamma,
                                            season_p, frailty)
    assert trace == list(S.DRAW_ORDER)
    # The derived count is exactly the difference, never negative.
    assert np.array_equal(counts["fg2a"],
                          np.maximum(counts["fga"] - made["fg3a"], 0.0))
    assert (counts["fg2a"] >= 0).all()
    # Makes never exceed the attempts they were drawn on.
    assert (made["fg3a"] <= counts["fga"]).all()
    assert (made["fg2m"] <= counts["fg2a"]).all()
    assert (made["fg3m"] <= made["fg3a"]).all()
    assert (made["ftm"] <= counts["fta"]).all()


def test_minutes_are_the_shared_exposure_for_every_count_head():
    """Doubling the one minutes draw doubles every count head's mean, together.

    Rule 2 in one assertion: there is a single exposure and all seven counts read it. A
    head wired to its own minutes would not move with the others.
    """
    rates = _rates(n_units=1)
    unit = np.zeros(4000, dtype=int)
    season_gamma = {c: np.ones(len(unit)) for c in COUNT_HEADS}
    season_p = {f"{m}|{a}": np.full(len(unit), 0.4) for m, a in CONVERSION_HEADS}
    frailty = np.ones((len(unit), len(COUNT_HEADS)))

    low, _, _ = S.draw_components(np.random.default_rng(0), np.full(len(unit), 12.0),
                                  unit, rates, 0, season_gamma, season_p, frailty)
    high, _, _ = S.draw_components(np.random.default_rng(0), np.full(len(unit), 24.0),
                                   unit, rates, 0, season_gamma, season_p, frailty)
    for component in COUNT_HEADS:
        ratio = high[component].mean() / max(low[component].mean(), 1e-9)
        assert 1.8 < ratio < 2.2, component


# ── The stick-breaking scaffolding the composition head is handed ─────────────

def test_block_bounds_and_suffix_sums_match_a_groupby():
    """The numpy forms reproduce `sequential_columns`' groupby, which is what they replace."""
    block = _blocks([3, 1, 4])
    values = np.array([1.0, 2.0, 3.0, 5.0, 1.0, 1.0, 2.0, 4.0])
    starts, lens = S.block_bounds(block)
    assert np.array_equal(starts, [0, 3, 4])
    assert np.array_equal(lens, [3, 1, 4])

    frame = pd.DataFrame({"b": block, "w": values})
    total = frame.groupby("b")["w"].transform("sum").to_numpy()
    before = frame.groupby("b")["w"].cumsum().to_numpy() - values
    assert np.allclose(S.suffix_sums(values, starts, lens), total - before)


def test_offset_clipped_marks_a_player_the_carry_forward_cannot_pay():
    """A block whose weights demand more of one player than the cap allows saturates.

    The last player in a block is always marked, in the simulator and in
    `sequential_columns` alike: his stick ratio is 1 and he takes exactly what is left, so
    `raw = R/min(U, R) = 1`. That is a property of the stick-breaking decomposition rather
    than a quirk here, which is why the assertions are about the rotation *ahead* of him.
    """
    k = 10
    starts, lens = np.array([0]), np.array([k])
    caps, n_total = np.full(k, 48.0), np.full(k, 240.0)

    balanced = np.full(k, 1.0 / k)
    ratio = balanced / S.suffix_sums(balanced, starts, lens)
    even = S.offset_clipped(ratio, balanced, starts, lens, n_total, caps)
    assert even[:-1].sum() == 0 and even[-1] == 1.0

    lopsided = np.r_[0.9, np.full(k - 1, 0.1 / (k - 1))]
    ratio = lopsided / S.suffix_sums(lopsided, starts, lens)
    skewed = S.offset_clipped(ratio, lopsided, starts, lens, n_total, caps)
    assert skewed[0] == 1.0
    assert skewed.sum() > even.sum()


def test_feasibility_repair_promotes_only_what_the_allocation_needs():
    """Five available players per team-game, because `N = 5*U` has no solution below that."""
    block = _blocks([8, 8])
    available = np.zeros(16, dtype=bool)
    available[:3] = True                 # first block short by two
    available[8:] = True                 # second block complete
    repaired, short = S.feasibility_repair(available, block, 2)
    assert short == 1
    assert repaired[:8].sum() == S.MIN_AVAILABLE
    assert repaired[8:].sum() == 8
    # It promotes the highest-ranked absentees, and rows arrive in rotation order.
    assert list(np.flatnonzero(repaired[:8])) == [0, 1, 2, 3, 4]


def test_feasibility_repair_is_a_no_op_when_every_block_is_deep_enough():
    block = _blocks([6, 6])
    available = np.ones(12, dtype=bool)
    repaired, short = S.feasibility_repair(available, block, 2)
    assert short == 0
    assert repaired is available or np.array_equal(repaired, available)


# ── The availability draw's dispersion axis ───────────────────────────────────

def _availability_artifact(n_rho: int = 4, edges=(0.0, 12.0, 24.0, 30.0, 60.0)):
    """An availability posterior in the shape `make posteriors` writes.

    `n_rho = 1` is the shared-dispersion arm, which persists a `(draws,)` `rho_draws` and an
    empty recipe — the shape the `train_val` artifact on disk still carries.
    """
    graded = n_rho > 1
    rho = (np.tile(np.array([0.32, 0.27, 0.25, 0.21])[:n_rho], (5, 1))
           if graded else np.full(5, 0.28))
    steps = ({"kind": "cut", "column": "minutes_per_game_lag1",
              "edges": list(edges), "name": "rho_bin"},) if graded else ()
    return P.PosteriorArtifact(
        head="availability", head_label="availability", family="betabinomial",
        response="mean_mu",
        recipe=P.DesignRecipe("base", [], None, steps=steps, builder="test"),
        draws={"alpha_draws": np.zeros(5), "beta_draws": np.zeros((5, 0)),
               "rho_draws": rho},
        extras={"n_rho": n_rho, "rho_bin_column": "rho_bin",
                "rho_bin_source": "minutes_per_game_lag1", "role_rho": graded},
        provenance={"fit_window": "train"})


def test_availability_rho_bin_is_zero_based_against_a_one_based_recipe():
    """`rho_bin` is 1-based and `rho_draws` is 0-based, and the gap is silent if missed.

    An off-by-one here hands every player his *neighbour's* dispersion at a perfectly legal
    index — no exception, no shape error, just the wrong model. So the buckets are pinned
    against the edges rather than against each other: 5 mpg is the fringe bucket, which is
    column 0, and 45 mpg is the star bucket, which is the last column.
    """
    art = _availability_artifact()
    frame = pd.DataFrame({"minutes_per_game_lag1": [5.0, 18.0, 27.0, 45.0]})
    bins = S.availability_rho_bin(art, frame)
    assert list(bins) == [0, 1, 2, 3]
    # The head's own 1-based assignment, one subtraction away — the invariant, not the values.
    one_based = art.recipe.transform(frame)["rho_bin"].to_numpy(int)
    assert list(one_based) == [1, 2, 3, 4]
    np.testing.assert_array_equal(bins, one_based - 1)
    # And the gathered dispersion is monotone in role, which is the fitted direction.
    gathered = np.asarray(art.draws["rho_draws"])[0][bins]
    assert (np.diff(gathered) < 0).all()


def test_availability_rho_bin_gives_a_player_with_no_design_row_the_widest_bucket():
    """A rostered player the head has no row for still needs a bucket, and it is bucket 1.

    `build_context` reindexes the design onto the roster, so a no-design player arrives with
    a NaN prior MPG and an empirical mean. `role_bins` sends him to the **lowest** bucket —
    the widest dispersion, the conservative direction — and the persisted `cut` step has to
    reproduce that rather than raise or produce a NaN index.
    """
    art = _availability_artifact()
    frame = pd.DataFrame({"minutes_per_game_lag1": [np.nan, 90.0, 33.0]})
    bins = S.availability_rho_bin(art, frame)
    assert bins[0] == 0 and bins[1] == 0        # NaN and above the top edge both fall in
    assert bins[2] == 3
    assert np.asarray(art.draws["rho_draws"])[0][bins[0]] == max(
        np.asarray(art.draws["rho_draws"])[0])


def test_availability_rho_bin_reads_a_shared_dispersion_artifact_as_one_column():
    """The `train_val` artifact predates the graded head: `(draws,)` and no recipe step.

    Handling only the graded shape would trade one broken window for the other, so the
    absence of the step is read as the shared arm rather than as a broken recipe.
    """
    art = _availability_artifact(n_rho=1)
    frame = pd.DataFrame({"minutes_per_game_lag1": [5.0, 45.0, np.nan]})
    assert list(S.availability_rho_bin(art, frame)) == [0, 0, 0]


def test_availability_rho_bin_refuses_a_graded_artifact_whose_recipe_lost_the_cut():
    """Four dispersion columns and no way to address them is unrecoverable, not a default.

    Defaulting to column 0 would silently apply the fringe bucket's dispersion to every star
    in the league, which is the failure this whole path exists to stop.
    """
    art = _availability_artifact()
    art.recipe.steps = ()
    with pytest.raises(KeyError, match="which column applies"):
        S.availability_rho_bin(art, pd.DataFrame({"minutes_per_game_lag1": [5.0]}))


def test_availability_rates_gather_each_player_his_own_bucket_dispersion():
    """The regression test. The scalar broadcast this replaced *raises* on this input.

    `np.full(n_players, rho_draws[draw])` with a `(4,)` row is
    `ValueError: could not broadcast input array from shape (4,) into shape (n,)`, which is
    what `make simulate-season` did against the shipped `train` posterior. Beyond not
    raising, the draw has to be *graded*: the fringe bucket's rates must be more dispersed
    than the star bucket's at the same mean, or the vector is being gathered wrongly.
    """
    n = 40_000
    rho_by_bin = np.array([0.32, 0.27, 0.25, 0.21])
    bins = np.repeat(np.arange(4), n)
    mu = np.full(4 * n, 0.75)

    rates = S.availability_rates(np.random.default_rng(0), mu, rho_by_bin, bins)
    assert rates.shape == (4 * n,)
    spread = np.array([rates[bins == j].std() for j in range(4)])
    # Monotone in the fitted dispersion, and matching the beta's own sd = sqrt(mu(1-mu)rho).
    assert (np.diff(spread) < 0).all()
    np.testing.assert_allclose(spread, np.sqrt(0.75 * 0.25 * rho_by_bin), rtol=0.02)


def test_availability_rates_reproduce_the_scalar_form_under_a_shared_dispersion():
    """`n_rho = 1` must be the old behaviour exactly, not merely close to it.

    That is the rollback path: a shared-dispersion artifact has to give bit-identical draws
    to the scalar broadcast it replaced, or the fix has changed a shipped window's numbers
    while claiming to repair the other one.
    """
    mu = np.linspace(0.2, 0.95, 500)
    bins = np.zeros(500, dtype=np.int64)
    graded = S.availability_rates(np.random.default_rng(7), mu, np.array([0.28]), bins)
    a, b = S.beta_shapes(mu, np.full(500, 0.28))
    scalar = np.random.default_rng(7).beta(a, b)
    np.testing.assert_array_equal(graded, scalar)


def test_pi_zero_is_the_single_component_draw_bit_for_bit():
    """The mixture's rollback path in the simulator, and it has to cost no rng draws.

    `pi = 0` must not merely give the same *distribution* — it must give the same numbers,
    or the shipped single-component window's tensors would move the moment the mixture
    became expressible. Which means the low component's `rng.random` / `rng.beta` calls have
    to be skipped rather than drawn and discarded.
    """
    mu = np.linspace(0.2, 0.95, 400)
    bins = np.zeros(400, dtype=np.int64)
    rho = np.array([0.28])
    plain = S.availability_rates(np.random.default_rng(11), mu, rho, bins)
    nested = S.availability_rates(np.random.default_rng(11), mu, rho, bins,
                                  pi=np.zeros(400), mu_low=0.10, rho_low=0.05)
    np.testing.assert_array_equal(plain, nested)


def test_the_mixture_draws_the_component_first_rather_than_blending_the_rates():
    """`pi = 1` must land on the low component, not somewhere between the two.

    Averaging the two rates would produce a season between healthy and disrupted, which is
    precisely the season the arm exists to say does not happen — and it would look right in
    every mean-based check. So the test is on the whole distribution: at `pi = 1` the draws
    have the low component's mean AND its spread, and at an intermediate `pi` the sample is
    bimodal rather than shifted.
    """
    n = 40_000
    mu = np.full(n, 0.90)
    bins = np.zeros(n, dtype=np.int64)
    rho = np.array([0.05])

    low = S.availability_rates(np.random.default_rng(3), mu, rho, bins,
                               pi=np.ones(n), mu_low=0.10, rho_low=0.05)
    assert abs(low.mean() - 0.10) < 0.01
    np.testing.assert_allclose(low.std(), np.sqrt(0.10 * 0.90 * 0.05), rtol=0.05)

    mixed = S.availability_rates(np.random.default_rng(4), mu, rho, bins,
                                 pi=np.full(n, 0.25), mu_low=0.10, rho_low=0.05)
    # A quarter of the mass sits at the low component and three quarters at the main one,
    # with the midpoint nearly empty — the signature a blended rate cannot produce.
    assert abs(((mixed < 0.5).mean()) - 0.25) < 0.02
    assert ((mixed > 0.4) & (mixed < 0.6)).mean() < 0.02


# ── The copula ────────────────────────────────────────────────────────────────

def test_count_copula_inflates_the_residual_matrix_rather_than_using_it_raw():
    """The frailty correlation is not the residual correlation, and must be larger.

    Under a lognormal frailty of variance `v`, a residual correlation of `r` needs a frailty
    correlation of roughly `r / (v * mu)` — an order of magnitude larger at these means.
    Handing the copula the residual matrix directly imposes about a tenth of the measured
    dependence, with every cell present and only the numbers wrong.
    """
    heads = list(COUNT_HEADS)
    # `to_matrix` is strict about names in both directions, so the synthetic frame has to
    # carry every one of the eleven heads even though only the count block is consumed.
    every = heads + [f"{m}|{a}" for m, a in CONVERSION_HEADS]
    rows = [{"component_a": a, "component_b": b,
             "r": 1.0 if a == b else (0.05 if a in heads and b in heads else 0.0),
             "basis": S.MINUTES_CONDITIONED, "fit_window": "train"}
            for a in every for b in every]
    long = pd.DataFrame(rows)
    copula = S.count_copula(long, "train", np.full(len(heads), 4.0), overdispersion=0.025)

    off = ~np.eye(len(heads), dtype=bool)
    assert (copula["frailty"][off] > copula["target"][off]).all()
    # And the achieved residual correlation lands back on the target it was inverted from.
    assert np.allclose(copula["achieved"][off], 0.05, atol=5e-3)
    assert np.linalg.eigvalsh(copula["frailty"]).min() > 0


def test_nearest_correlation_repairs_an_indefinite_matrix():
    R = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    assert np.linalg.eigvalsh(R).min() < 0
    fixed = S.nearest_correlation(R)
    assert np.linalg.eigvalsh(fixed).min() >= -1e-12
    assert np.allclose(np.diag(fixed), 1.0)


def test_mean_expected_reads_the_rows_a_residual_is_measured_on():
    """`count_residuals` drops rows below `MIN_EXPECTED`, so the inverting mean must too."""
    mu = np.array([0.1, 0.2, 4.0, 6.0])
    assert S._mean_expected(mu) == pytest.approx(5.0)


# ── The split guard ───────────────────────────────────────────────────────────

def test_allowed_seasons_stops_at_validation():
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(15, 25)]
    design = _season_frame(seasons)
    allowed = S.allowed_seasons(design)
    assert seasons[-1] not in allowed and seasons[-2] not in allowed
    assert seasons[-3] in allowed and seasons[-4] in allowed
    assert S.validation_seasons(design) == sorted(seasons[-4:-2])


def test_simulating_a_test_season_raises_unless_unlocked(monkeypatch):
    """The recorded discipline failure was a gate specified on test figures. This is the
    guard that would have caught it, pinned the way `component_rates`' is.

    `conftest` unlocks the split suite-wide so synthetic four-season fixtures can be read
    from both sides, so this test re-locks first — otherwise it would pass vacuously,
    which is the one failure mode a guard test cannot afford.
    """
    monkeypatch.setattr(held_out, "_unlocked", False, raising=False)
    seasons = [f"20{y:02d}-{y + 1:02d}" for y in range(15, 25)]
    design = _season_frame(seasons)
    with pytest.raises(HeldOutLocked):
        S.assert_season_allowed(seasons[-1], design)
    with unlocked("a test"):
        S.assert_season_allowed(seasons[-1], design)
    S.assert_season_allowed(seasons[-3], design)      # validation is always legal


# ── The output contract ───────────────────────────────────────────────────────

def test_scoring_slots_partition_the_tournament_into_twenty_periods(tmp_path):
    """Round 1's 17 weeks keep a slot each; the three double weeks collapse to one each.

    Games outside DK's window keep a row and lose their slot, because they still happen —
    availability is drawn against the full schedule and only the scoring step drops them.
    """
    rows = []
    game = 0
    for week in range(1, 26):
        tournament = (1 if week <= 17 else
                      2 if week <= 19 else
                      3 if week <= 21 else
                      4 if week <= 23 else 0)
        for _ in range(3):
            rows.append({"season": "2022-23", "game_id": game,
                         "period_index": week, "tournament_round": tournament})
            game += 1
    frame = pd.DataFrame(rows)
    frame.to_parquet(tmp_path / "scoring_periods.parquet")

    slots = S.scoring_slots(tmp_path, "2022-23")
    inside = slots[slots["slot"] >= 0]
    assert sorted(inside["slot"].unique()) == list(range(S.N_SCORING_PERIODS))
    assert (slots.loc[slots["tournament_round"] == 0, "slot"] == -1).all()
    # Each double week is one slot, and Round 1's seventeen are seventeen.
    per_round = inside.groupby("tournament_round")["slot"].nunique()
    assert per_round.loc[1] == S.ROUND_1_WEEKS
    assert per_round.loc[2] == per_round.loc[3] == per_round.loc[4] == 1


def test_artifact_name_round_trips_the_head_naming():
    assert S.artifact_name("fg3a|fga") == "fg3a_given_fga"
    assert S.artifact_name("reb") == "reb"
