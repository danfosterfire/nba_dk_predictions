"""Tests for session 6b — the preseason increment on the five surviving rate heads.

Nothing here needs a sampler: the ladder is a point MLE by construction, the same argument
`tests/test_minutes_preseason.py` makes about the module it borrows its shape from.

Eight things are pinned, each of which would move every figure in
`components_preseason.csv` without raising:

- **the block nests the incumbent exactly**, on BOTH likelihoods. Every arm is "the shipped
  variant plus columns", and the whole comparison is a nested increment only if a zero block
  reproduces the head that ships. This head family has two link functions, so the property
  has to hold twice — the house nesting discipline (`pi = 0`, `K = 1`, `U_n = 0`,
  `n_rho = 1`);
- **difference coding means what it says, on each head's own link** — `log1p` on a per-36
  count rate, `logit` on a conversion percentage. A count head's delta measured on a logit
  (or vice versa) would still correlate with the target and would still fit, which is why
  the two scales are asserted rather than eyeballed;
- **the age split partitions the missing rows**, or P1's decision-3 census is being applied
  to a different population than the one it was measured on;
- **the reliability weight's left edge is the un-shrunk arm exactly**, so the grid's control
  is a control rather than a nearby model;
- **the centred delta removes a within-season level and nothing else**, and uses each season
  alone so it stays point-in-time;
- **the gate needs both halves.** A validation-only pass must not pass — this project has
  twice shipped a block that won validation and shrank 4-6x rolling, and the multiplicity is
  six heads here rather than one;
- **the armed heads are P1's short list and nothing else.** `blk`, `fta`, `fg2m|fg2a` and
  `fg3m|fg3a` are recorded nulls; a head added to `COUNT_ARMS` without a P1 reading would be
  re-opening a decision rather than executing one;
- **the shipped variant is read, not assumed.** A ladder holding a variant the head does not
  ship has no incumbent in it, so `shipped_variants` raises rather than defaulting.
"""

import numpy as np
import pandas as pd
import pytest

from src.eda.preseason_value import AGE_LABELS, attach_rate_block, log_rate, logit
from src.models import components_preseason as CP
from src.models.component_rates import (BIO_COLS, CONTEXT_COLS, CONVERSION_HEADS,
                                        COUNT_HEADS, fit_conversion_head, fit_count_head)


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n_per_season: int = 120, seasons=("2018-19", "2019-20", "2020-21"),
            seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the rate heads, `variants` and the block builder read.

    Built to the shape `component_rates.build_design` produces rather than by calling it,
    because that function needs 30 seasons of raw game logs on disk; every column any test
    touches is filled explicitly and the rest is noise with the right dtype.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        minutes = rng.uniform(400, 2600, n_per_season)
        frame = pd.DataFrame({
            "season": season,
            "player_id": np.arange(n_per_season),
            "total_minutes": minutes,
            "gp": rng.integers(30, 82, n_per_season).astype(float),
            "age": rng.uniform(21, 36, n_per_season),
        })
        frame["age_sq"] = frame["age"] ** 2
        frame["career_year"] = rng.integers(0, 15, n_per_season).astype(float)
        frame["mpg_lag1"] = rng.uniform(6, 34, n_per_season)
        frame["total_minutes_lag1"] = rng.uniform(400, 2600, n_per_season)
        frame["gp_lag1"] = rng.integers(30, 82, n_per_season).astype(float)
        for column in COUNT_HEADS:
            rate = rng.uniform(1.0, 9.0, n_per_season)
            frame[f"{column}_p36_lag1"] = rate
            frame[column] = np.maximum(rng.poisson(rate * minutes / 36.0), 1).astype(float)

        # The shot-attempt basis, in the order the generative chain reaches it: `fg3a` is a
        # MAKE of `fga` and simultaneously the trials for `fg3m`, and `fg2a` is derived. So
        # the two of them need a per-36 lag as well as a percentage lag — `rate_columns()`
        # asks for one on every trials column, not only on the count heads.
        share = rng.uniform(0.15, 0.6, n_per_season)
        frame["fg3a_pct_lag1"] = share
        frame["fg3a"] = np.maximum(rng.binomial(frame["fga"].to_numpy(int), share), 1.0)
        frame["fg3a_p36_lag1"] = frame["fga_p36_lag1"] * share
        frame["fg2a"] = np.maximum(frame["fga"] - frame["fg3a"], 1.0)
        frame["fg2a_p36_lag1"] = frame["fga_p36_lag1"] * (1.0 - share)
        for made, attempted in [("fg2m", "fg2a"), ("fg3m", "fg3a"), ("ftm", "fta")]:
            pct = rng.uniform(0.25, 0.9, n_per_season)
            frame[f"{made}_pct_lag1"] = pct
            frame[made] = rng.binomial(frame[attempted].to_numpy(int), pct).astype(float)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _panel(design: pd.DataFrame, missing_every: int | None = 7,
           seed: int = 1) -> pd.DataFrame:
    """A preseason panel covering all but every `missing_every`-th row of `design`.

    `None` covers everyone, which is what the difference-coding tests need — they are about
    what a *present* row encodes, and a hole in the frame would be answering P1's
    missingness census instead.
    """
    rng = np.random.default_rng(seed)
    keep = (design if missing_every is None
            else design.iloc[[i for i in range(len(design)) if i % missing_every]])
    n = len(keep)
    out = pd.DataFrame({
        "season": keep["season"].to_numpy(),
        "player_id": keep["player_id"].to_numpy(),
        "min_pre": rng.uniform(20, 160, n),
        "gp_share_pre": rng.uniform(0.3, 1.0, n),
        "min_share_pre": rng.uniform(0.02, 0.15, n),
        "min_share_pre_late": rng.uniform(0.02, 0.15, n),
        "min_rank_pre": rng.integers(1, 15, n).astype(float),
        "missed_tail": rng.integers(0, 2, n).astype(float),
        "played_final_game": rng.integers(0, 2, n).astype(float),
        "team_pre_games": np.full(n, 5.0),
        "pre_fg3a_share": rng.uniform(0.1, 0.6, n),
    })
    out["mpg_pre"] = out["min_pre"] / 5.0
    for column in ("fga_pre", "fg3a_pre", "fta_pre"):
        out[column] = rng.uniform(4, 40, n)
    out["fgm_pre"] = out["fga_pre"] * rng.uniform(0.3, 0.6, n)
    out["fg3m_pre"] = out["fg3a_pre"] * rng.uniform(0.2, 0.5, n)
    out["ftm_pre"] = out["fta_pre"] * rng.uniform(0.5, 0.95, n)
    for column in COUNT_HEADS:
        out[f"pre_per36_{column}"] = rng.uniform(0.5, 12, n)
    return out


def _attached(design: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The module's real attach step, over P1's real delta builder."""
    return CP.attach_preseason(design, panel)


# ── Nesting: the property the whole gate rests on ─────────────────────────────

def test_a_zero_block_reproduces_the_incumbent_count_head_exactly():
    """The nested-increment claim on the negative-binomial side.

    Every arm is the shipped feature list plus preseason columns, and the interval against
    the incumbent only measures the preseason if a block of zeros IS the incumbent.
    """
    design = _design()
    train = design[design["season"] != "2020-21"]
    val = design[design["season"] == "2020-21"]
    head = CP.heads()[0]
    tr, va, features = head.variants(train, val, 5)["log_own"]
    zeros = [f"zero_{i}" for i in range(3)]
    for frame in (tr, va):
        for column in zeros:
            frame[column] = 0.0

    mu, phi = fit_count_head(tr, va, features, head.component)
    mu_wide, phi_wide = fit_count_head(tr, va, features + zeros, head.component)
    assert np.allclose(mu, mu_wide, rtol=1e-6)
    assert np.isclose(phi, phi_wide, rtol=1e-6)


def test_a_zero_block_reproduces_the_incumbent_conversion_head_exactly():
    """And on the beta-binomial side, which is a different solver and a different link."""
    design = _design()
    train = design[design["season"] != "2020-21"]
    val = design[design["season"] == "2020-21"]
    head = [h for h in CP.heads() if h.kind == "conversion"][0]
    tr, va, features = head.variants(train, val, 5)["logit_own"]
    zeros = [f"zero_{i}" for i in range(3)]
    for frame in (tr, va):
        for column in zeros:
            frame[column] = 0.0

    _, p, rho = fit_conversion_head(tr, va, features, head.component, head.attempted)
    _, p_wide, rho_wide = fit_conversion_head(tr, va, features + zeros, head.component,
                                              head.attempted)
    assert np.allclose(p, p_wide, rtol=1e-6)
    assert np.isclose(rho, rho_wide, rtol=1e-6)


# ── Difference coding, on two link scales ─────────────────────────────────────

def test_a_count_head_delta_is_zero_when_the_preseason_rate_agrees():
    """A `log1p` per-36 delta: zero means "the preseason agrees with the prior season"."""
    design = _design(n_per_season=40, seasons=("2019-20",))
    panel = _panel(design, missing_every=None)
    for column in COUNT_HEADS:
        panel[f"pre_per36_{column}"] = design[f"{column}_p36_lag1"].to_numpy(dtype=float)
    out = _attached(design, panel)
    for head in CP.heads():
        if head.kind == "count":
            assert np.abs(out[head.delta].to_numpy(dtype=float)).max() < 1e-9


def test_the_conversion_head_delta_is_a_logit_of_a_percentage():
    """`ftm|fta` is a beta-binomial on made-out-of-attempted, so its twin is a PERCENTAGE.

    A per-36 free-throw *rate* would also correlate with the target and would also fit, and
    the coefficient would then live on a scale the link does not use — which is the category
    error `preseason_value.attach_rate_block` splits the two families to avoid.
    """
    design = _design(n_per_season=40, seasons=("2019-20",))
    panel = _panel(design, missing_every=None)
    out = _attached(design, panel)
    head = [h for h in CP.heads() if h.kind == "conversion"][0]

    share = panel["ftm_pre"].to_numpy(dtype=float) / panel["fta_pre"].to_numpy(dtype=float)
    expected = logit(share) - logit(design["ftm_pct_lag1"].to_numpy(dtype=float))
    assert np.allclose(out[head.delta].to_numpy(dtype=float), expected)
    # And it is NOT the log-rate form the count heads take.
    assert not np.allclose(out[head.delta].to_numpy(dtype=float),
                           log_rate(share) - log_rate(design["ftm_pct_lag1"]))


def test_a_row_with_no_preseason_carries_a_zero_delta_and_one_age_indicator():
    design = _design(n_per_season=60, seasons=("2019-20",))
    out = _attached(design, _panel(design, missing_every=5))
    missing = out["has_preseason"].to_numpy(dtype=float) <= 0
    assert missing.any()
    for head in CP.heads():
        assert np.allclose(out.loc[missing, head.delta].to_numpy(dtype=float), 0.0)
        assert np.allclose(
            out.loc[missing, CP.centered_column(head.delta)].to_numpy(dtype=float), 0.0)

    indicators = out[CP.MISSING_AGE_COLS].to_numpy(dtype=float)
    assert np.allclose(indicators.sum(axis=1), missing.astype(float))


def test_the_age_indicators_sum_to_the_plain_indicator_they_replace():
    """P1 decision 3 splits `has_preseason`; a split that is not a partition is a new column."""
    design = _design(n_per_season=80, seasons=("2018-19", "2019-20"))
    out = _attached(design, _panel(design, missing_every=4))
    split = out[CP.MISSING_AGE_COLS].to_numpy(dtype=float).sum(axis=1)
    plain = 1.0 - out["has_preseason"].to_numpy(dtype=float)
    assert np.allclose(split, plain)
    assert len(CP.MISSING_AGE_COLS) == len(AGE_LABELS)


# ── The reliability shrink ────────────────────────────────────────────────────

def test_the_shrinkage_weight_at_zero_is_the_unshrunk_delta_exactly():
    """The grid's left edge is a control, not a nearby model."""
    design = _design(n_per_season=50, seasons=("2019-20",))
    out = _attached(design, _panel(design, missing_every=6))
    for head in CP.heads():
        shrunk = CP.with_shrunk_delta(out, head.delta, 0.0)
        assert np.allclose(shrunk[CP.shrunk_column(head.delta)].to_numpy(dtype=float),
                           out[head.delta].to_numpy(dtype=float))


def test_a_row_with_no_preseason_is_shrunk_to_zero_rather_than_to_a_fill():
    design = _design(n_per_season=60, seasons=("2019-20",))
    attached = _attached(design, _panel(design, missing_every=5))
    head = CP.heads()[0]
    out = CP.with_shrunk_delta(attached, head.delta, 40.0)
    missing = out["has_preseason"].to_numpy(dtype=float) <= 0
    assert np.allclose(out.loc[missing, CP.shrunk_column(head.delta)].to_numpy(float), 0.0)


# ── The centred delta ─────────────────────────────────────────────────────────

def test_the_centred_delta_removes_a_within_season_level_and_leaves_the_spread():
    """Centring is a shift, not a rescale — otherwise the arm changes two things at once."""
    design = _design(n_per_season=100, seasons=("2018-19", "2019-20"))
    out = _attached(design, _panel(design, missing_every=6))
    head = CP.heads()[0]
    here = out[out["has_preseason"].to_numpy(dtype=float) > 0]
    for _, block in here.groupby("season"):
        assert abs(block[CP.centered_column(head.delta)].mean()) < 1e-9
        assert np.isclose(block[CP.centered_column(head.delta)].std(ddof=0),
                          block[head.delta].std(ddof=0))


def test_centring_uses_each_season_alone_so_it_stays_point_in_time():
    """The centring constant is a season's own preseason mean — on disk before its opener."""
    design = _design(n_per_season=60, seasons=("2018-19", "2019-20"))
    panel = _panel(design, missing_every=None)
    baseline = _attached(design, panel)
    head = CP.heads()[0]

    shifted = panel.copy()
    later = shifted["season"] == "2019-20"
    shifted.loc[later, f"pre_per36_{head.component}"] *= 1.5
    moved = _attached(design, shifted)

    earlier = (baseline["season"] == "2018-19").to_numpy()
    column = CP.centered_column(head.delta)
    assert np.allclose(baseline.loc[earlier, column].to_numpy(dtype=float),
                       moved.loc[earlier, column].to_numpy(dtype=float))


# ── The predictive ────────────────────────────────────────────────────────────

def test_the_conversion_predictive_never_draws_more_makes_than_attempts():
    """A beta-binomial draw is bounded by its trials, and CRPS is read in made free throws.

    An unbounded draw would still produce a finite CRPS and a plausible-looking table, which
    is why the support is asserted rather than assumed.
    """
    n = np.array([1, 3, 12, 40, 200])
    samples = CP.conversion_samples(np.full(len(n), 0.75), 0.02, n, 200, seed=0)
    assert samples.shape == (200, len(n))
    assert (samples <= n[None, :]).all() and (samples >= 0).all()


def test_the_count_predictive_is_centred_on_its_own_mean():
    mu = np.array([5.0, 50.0, 500.0])
    samples = CP.count_samples(mu, 400.0, 4000, seed=0)
    assert samples.shape == (4000, 3)
    assert np.allclose(samples.mean(axis=0), mu, rtol=0.1)


# ── The gate ──────────────────────────────────────────────────────────────────

def _gate_frames(val_hi: float, roll_hi: float, origins_won: int,
                 head: str = "ast") -> tuple:
    validation = pd.DataFrame([
        {"head": head, "arm": CP.PRIMARY_ARM, "population": CP.DECISION_POPULATION,
         "crps_vs_incumbent": -0.5, "crps_vs_incumbent_lo": -0.9,
         "crps_vs_incumbent_hi": val_hi, "crps_vs_primary": 0.0,
         "crps_vs_primary_hi": 0.0}])
    rolling = pd.DataFrame([
        {"head": head, "arm": CP.PRIMARY_ARM, "population": CP.DECISION_POPULATION,
         "crps_vs_incumbent": -0.6, "crps_vs_incumbent_lo": -0.9,
         "crps_vs_incumbent_hi": roll_hi, "origins_won": origins_won, "n_origins": 13,
         "crps_vs_primary": 0.0, "crps_vs_primary_hi": 0.0,
         "verdict_vs_primary": "primary", "origins_won_vs_primary": 0}])
    return validation, rolling


def test_the_gate_needs_both_halves():
    """Validation alone ships nothing — the §12e / §14f pattern, written as a conjunction.

    P2's round is the standing precedent that a bar re-read after seeing which side an arm
    landed on is not a bar, and this session reads SIX heads rather than one, so the rolling
    half is carrying more multiplicity than it did there.
    """
    assert CP.gate("ast", *_gate_frames(-0.2, -0.1, 12))["passes"]
    assert not CP.gate("ast", *_gate_frames(-0.2, +0.05, 12))["passes"]
    assert not CP.gate("ast", *_gate_frames(+0.05, -0.1, 12))["passes"]
    assert not CP.gate("ast", *_gate_frames(-0.2, -0.1, 6))["passes"]


def test_the_gate_is_what_earns_a_stan_port():
    """Recorded as a column rather than a printed sentence, so a later session reads a
    decision instead of re-deciding it."""
    assert CP.gate("ast", *_gate_frames(-0.2, -0.1, 12))["earns_stan_port"] is True
    assert CP.gate("ast", *_gate_frames(-0.2, +0.05, 12))["earns_stan_port"] is False


# ── Scope ─────────────────────────────────────────────────────────────────────

def test_the_armed_heads_are_p1s_short_list_and_nothing_else():
    """P1 decision 2, as code.

    `blk` and `fta` are recorded count nulls (both actively hurt by the block) and
    `fg2m|fg2a` / `fg3m|fg3a` are the conversion ones — `fg3m|fg3a`'s apparent +0.0149 was
    entirely `has_preseason` rather than preseason three-point percentage. Arming one of them
    here would be re-opening a P1 decision rather than executing it.
    """
    armed = {h.name for h in CP.heads()}
    assert armed == {"ast", "fga", "stl", "tov", "reb", "ftm|fta"}
    assert not (armed & set(CP.P1_NULL_HEADS))
    assert set(CP.P1_NULL_HEADS) == {"blk", "fta", "fg2m|fg2a", "fg3m|fg3a"}


def test_the_arms_are_all_nested_supersets_of_the_reference():
    """Every arm is "the incumbent plus columns", which is what makes the ladder an ablation."""
    for head in CP.heads():
        arms = CP.preseason_arms(head.delta)
        assert arms[CP.REFERENCE_ARM] == []
        assert CP.PRIMARY_ARM in arms
        for name, columns in arms.items():
            assert len(set(columns)) == len(columns), f"{head.name}/{name} repeats a column"
            if name != CP.REFERENCE_ARM:
                assert columns, f"{head.name}/{name} is the reference under another name"


def test_every_arm_carries_the_age_split_rather_than_a_plain_indicator():
    """P1 decision 3 applies "wherever it enters" — no arm may reintroduce the whole one."""
    for head in CP.heads():
        for name, columns in CP.preseason_arms(head.delta).items():
            if name == CP.REFERENCE_ARM:
                continue
            assert "has_preseason" not in columns, f"{head.name}/{name} unsplit indicator"
            assert all(column in columns for column in CP.MISSING_AGE_COLS), name


def test_the_shipped_variant_is_read_from_the_artifact_rather_than_assumed(tmp_path):
    """A ladder on a variant the head does not ship has no incumbent in it.

    The six heads do not agree on a variant (`log_own_spline` on three, `log_own` on two,
    `logit_own_spline` on the conversion), so this cannot be a module constant — and a
    silently-defaulted one is exactly how a `crps_vs_incumbent` column ends up measuring two
    changes at once.
    """
    with pytest.raises(FileNotFoundError):
        CP.shipped_variants(tmp_path)

    pd.DataFrame([
        {"head": "ast", "variant": "log_own_spline", "selected": True},
        {"head": "ast", "variant": "log_own", "selected": False},
    ]).to_csv(tmp_path / "stan_component_metrics.csv", index=False)
    assert CP.shipped_variants(tmp_path, ["ast"]) == {"ast": "log_own_spline"}
    with pytest.raises(ValueError):
        CP.shipped_variants(tmp_path, ["ast", "fga"])
