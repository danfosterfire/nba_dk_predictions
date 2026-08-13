"""Tests for P2 — the preseason increment on the availability head that ships.

Nothing here needs a sampler: the ladder is a point MLE by construction, the same argument
`tests/test_availability_window.py` makes about the module this one borrows its arms from.

Seven things are pinned, each of which would move every figure in
`availability_preseason.csv` without raising:

- **the block nests the shipped head exactly**, on `beta` *and* on `pi`. Every arm is
  "`mixture` plus columns", and an interval against it measures the preseason only if a zero
  block is the head that ships. On `pi` this is the stronger claim of the two: `pi = theta *
  sigmoid(gamma' z)` switches off through `theta` alone, so widening `z` adds parameters the
  nesting point does not depend on;
- **the block stays out of `FEATURE_COLS` and `PI_COLS`**, so the seven modules that import
  `build_design` cannot pick up a column that is structurally zero before 2004-05;
- **the age split partitions the missing rows**, or P1's decision-3 census is being applied
  to a different population than the one it was measured on;
- **a missing preseason is a zero everywhere else in the block**, so the fill cannot leak a
  level into a column the indicator is supposed to separate;
- **centring removes a within-season level and nothing else** — it is the arm that carried
  P3's finding, and if it also rescaled the cross-player part the result would be about two
  changes at once;
- **the gate needs both halves.** A validation-only pass must not pass: this head has twice
  admitted a block on a validation CRPS reading that shrank 4-6x on the rolling harness;
- **the gate reads the draftable population.** P1's first reading on this head was 6x too
  large because it pooled mid-season signings, and a gate that read the pooled row would
  reproduce exactly that error.
"""

import numpy as np
import pandas as pd

from src.eda.preseason_value import AGE_LABELS, MISSING_AGE_COLS, season_centered
from src.models import availability_preseason as AP
from src.models.availability import FEATURE_COLS
from src.models.availability_window import PI_COLS, MixtureFrailty


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n_per_season: int = 120, seasons=("2018-19", "2019-20", "2020-21"),
            seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the head, the block builder and the role cut read."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        frame = pd.DataFrame({
            "season": season,
            "player_id": np.arange(n_per_season),
            "team_games": np.full(n_per_season, 82),
            "gp": rng.integers(1, 83, n_per_season),
        })
        for column in FEATURE_COLS:
            frame[column] = rng.normal(0.0, 1.0, n_per_season)
        frame["age"] = rng.uniform(21, 36, n_per_season)
        frame["age_sq"] = frame["age"] ** 2
        frame["minutes_per_game_lag1"] = rng.uniform(5, 34, n_per_season)
        frame["min_share_lag1"] = rng.uniform(0.02, 0.15, n_per_season)
        frame["min_rank_lag1"] = rng.integers(1, 15, n_per_season).astype(float)
        frame["gp_share_lag1"] = rng.uniform(0.2, 1.0, n_per_season)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _panel(design: pd.DataFrame, missing_every: int | None = 7, level: float = 1.0,
           seed: int = 1) -> pd.DataFrame:
    """A preseason panel covering all but every `missing_every`-th row of `design`.

    `level` scales the preseason minutes of the *last* season only, which is how the seasons
    actually differ — 2 games a team in the 2011-12 lockout against 8 in an ordinary year —
    and is what the centring test needs to have something to remove.
    """
    rng = np.random.default_rng(seed)
    keep = (design if missing_every is None
            else design.iloc[[i for i in range(len(design)) if i % missing_every]])
    n = len(keep)
    scale = np.where(keep["season"].to_numpy() == sorted(design["season"].unique())[-1],
                     level, 1.0)
    out = pd.DataFrame({
        "season": keep["season"].to_numpy(),
        "player_id": keep["player_id"].to_numpy(),
        "min_pre": rng.uniform(20, 160, n) * scale,
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
    for column in ("fga_pre", "fgm_pre", "fg3a_pre", "fg3m_pre", "fta_pre", "ftm_pre"):
        out[column] = rng.uniform(1, 10, n)
    for column in ("fga", "fta", "reb", "ast", "stl", "blk", "tov"):
        out[f"pre_per36_{column}"] = rng.uniform(0.5, 20, n)
    return out


def _attached(design: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The real block builders, over P1's real attach step.

    `attach_prior_shares` is the one thing stubbed — it contributes exactly the two lag
    columns `_design` fills in, and rebuilding one from synthetic game logs would be testing
    `src/eda/` rather than this module.
    """
    from src.eda.preseason_value import (attach_availability_block,
                                         attach_missing_age_indicators)

    out = attach_missing_age_indicators(attach_availability_block(design.copy(), panel))
    out[AP.VOLUME_CENTERED] = season_centered(out, AP.VOLUME)
    return out


# ── Nesting: the property the whole gate rests on ─────────────────────────────

def test_a_zero_block_reproduces_the_shipped_head_on_beta_and_on_pi():
    """The nested-increment claim, as code, at both places a column can go.

    `assert_nests` is what the ladder runs on every arm, and it evaluates the arm's own
    log-likelihood at its nesting parameter values against the incumbent's on the same rows.
    Here it is asserted for a `beta` arm, a `pi` arm and an arm that widens both — because
    `theta = 0` has to keep nesting at any width of `pi`, which is the property that makes
    widening the disruption weight legal at all.
    """
    from src.models.availability_window import assert_nests

    design = _attached(_design(n_per_season=60, seasons=("2018-19", "2019-20")),
                       _panel(_design(n_per_season=60,
                                      seasons=("2018-19", "2019-20")), missing_every=5))
    for name in ("mixture", "mixture__volume", "mixture__pi", "mixture__volume_pi"):
        features, pi_features = AP.arm_columns(name)
        model = MixtureFrailty(l2=1.0, features=features,
                               pi_features=pi_features).fit(design)
        assert assert_nests(model, design) == 0.0
        # And the nesting point IS the incumbent's own fitted log-likelihood, not merely
        # some reproducible number: `_start` fits the shipped head first and every arm
        # begins there.
        assert np.isclose(model.nesting_loglik, model.incumbent_loglik, atol=1e-8)


def test_every_arm_is_the_shipped_lists_plus_columns():
    """Nesting as a statement about the column lists, which is cheaper to check than a fit.

    The reference arm must be the shipped head *exactly* — same features, same `pi` — and
    every other arm must be a superset that preserves the shipped order, or the comparison
    is between two models rather than along one coefficient path.
    """
    assert AP.arm_columns(AP.REFERENCE_ARM) == (list(FEATURE_COLS), list(PI_COLS))
    for name in AP.ARMS:
        features, pi_features = AP.arm_columns(name)
        assert features[:len(FEATURE_COLS)] == list(FEATURE_COLS)
        assert pi_features[:len(PI_COLS)] == list(PI_COLS)
        assert len(set(features)) == len(features)
        assert len(set(pi_features)) == len(pi_features)


def test_the_block_stays_out_of_the_shared_design():
    """`build_design` is imported by seven modules and must not learn a preseason column.

    The `attach_absence_mix` precedent, and the reason it exists: a column that is
    structurally zero before 2004-05 inside `FEATURE_COLS` would enter `stan_minutes`,
    `stan_composition`, `stan_games_played`, `model_cards`, `sim/season`, `season_terms` and
    `final_evaluation` without any of them asking for it.
    """
    for column in AP.BLOCK_COLS:
        assert column not in FEATURE_COLS
        assert column not in PI_COLS


def test_block_cols_covers_every_column_any_arm_adds():
    """`run` asserts `BLOCK_COLS` is finite before fitting, so a column missing from that
    list is a NaN that reaches an optimizer instead of a build failure."""
    added = {c for beta, pi in AP.ARMS.values() for c in list(beta) + list(pi)}
    assert added <= set(AP.BLOCK_COLS)


# ── The encodings ─────────────────────────────────────────────────────────────

def test_a_row_with_no_preseason_is_a_zero_plus_exactly_one_age_indicator():
    design = _design()
    out = _attached(design, _panel(design, missing_every=5))
    missing = out["has_preseason"].to_numpy(dtype=float) <= 0
    assert missing.any()
    for column in [AP.VOLUME, AP.VOLUME_CENTERED] + AP.PARTICIPATION + AP.P1_DELTAS:
        assert np.allclose(out.loc[missing, column].to_numpy(dtype=float), 0.0)
    indicators = out.loc[missing, MISSING_AGE_COLS].to_numpy(dtype=float)
    assert np.allclose(indicators.sum(axis=1), 1.0)
    assert np.allclose(out.loc[~missing, MISSING_AGE_COLS].to_numpy(dtype=float), 0.0)


def test_the_age_indicators_sum_to_the_plain_indicator_they_replace():
    """P1 decision 3: the split has to span what `has_preseason` spanned, or the arms are
    not comparable with the census the decision came from."""
    design = _design()
    out = _attached(design, _panel(design, missing_every=4))
    total = out[MISSING_AGE_COLS].to_numpy(dtype=float).sum(axis=1)
    assert np.allclose(total, 1.0 - out["has_preseason"].to_numpy(dtype=float))
    assert len(MISSING_AGE_COLS) == len(AGE_LABELS)


def test_centring_removes_a_within_season_level_and_nothing_else():
    """The arm that carried P3's finding, re-used on a level rather than a delta.

    A season whose preseason is a third the length of the others has a systematically lower
    `pre_log_min`, and this head has no season term to absorb it. Centring has to take that
    difference out **without** touching the cross-player spread, or the arm is two changes.
    """
    design = _design()
    out = _attached(design, _panel(design, missing_every=None, level=0.3))
    present = out["has_preseason"].to_numpy(dtype=float) > 0
    raw = out.loc[present].groupby("season")[AP.VOLUME].mean()
    centred = out.loc[present].groupby("season")[AP.VOLUME_CENTERED].mean()

    assert raw.max() - raw.min() > 0.5              # the level is really there
    assert np.allclose(centred.to_numpy(dtype=float), 0.0, atol=1e-10)
    # Within a season the two columns differ by a constant, so every spread is preserved.
    for season, part in out.loc[present].groupby("season"):
        assert np.isclose(part[AP.VOLUME].std(ddof=1),
                          part[AP.VOLUME_CENTERED].std(ddof=1), atol=1e-10)


def test_centring_never_pools_across_seasons():
    """Point-in-time: a season's centring constant is its OWN preseason mean, which is on
    disk before its opener. Pooling would make a 2018-19 row depend on 2020-21."""
    design = _design()
    panel = _panel(design, missing_every=None)
    full = _attached(design, panel)
    first = sorted(design["season"].unique())[0]
    part = _attached(design[design["season"] == first].reset_index(drop=True),
                     panel[panel["season"] == first].reset_index(drop=True))
    assert np.allclose(full.loc[full["season"] == first, AP.VOLUME_CENTERED]
                       .to_numpy(dtype=float),
                       part[AP.VOLUME_CENTERED].to_numpy(dtype=float))


# ── The gate ──────────────────────────────────────────────────────────────────

def _gate_frames(val_pass: bool, roll_pass: bool, population: str = "draftable",
                 origins_won: int = 8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Minimal ladder and rolling tables carrying only what `gate` reads."""
    def row(passes: bool, **extra) -> dict:
        return {"arm": AP.PRIMARY_ARM, "population": population,
                "crps_vs_mixture": -0.05, "crps_vs_mixture_lo": -0.10,
                "crps_vs_mixture_hi": -0.01 if passes else 0.02,
                "boundary_vs_mixture": -0.001, "boundary_vs_mixture_lo": -0.002,
                "boundary_vs_mixture_hi": 0.001,
                "wins_crps_holds_boundary": passes, **extra}
    validation = pd.DataFrame([row(val_pass)])
    rolling = pd.DataFrame([row(roll_pass, origins_won=origins_won, n_origins=10)])
    return validation, rolling


def test_the_gate_needs_both_halves():
    """Validation alone ships nothing, and neither does the rolling harness alone.

    This is not caution for its own sake: `docs/availability-window-plan.md` §12e and §14f
    are two blocks that won a validation CRPS interval on these exact rows and shrank 5.8x
    and 4.2x on the fitting half, with their intervals reopened across zero.
    """
    for val_pass, roll_pass, expected in [(True, True, True), (True, False, False),
                                          (False, True, False), (False, False, False)]:
        validation, rolling = _gate_frames(val_pass, roll_pass)
        result = AP.gate(validation, rolling)
        assert result["passes"] is expected
        assert result["earns_stan_port"] is expected


def test_a_minority_of_rolling_origins_fails_the_gate():
    """A pooled interval and a win count answer different questions, and the origins are the
    independent replicates — 4 of 7 is the coin flip §14f recorded."""
    validation, rolling = _gate_frames(True, True, origins_won=4)
    assert AP.gate(validation, rolling)["passes"] is False


def test_the_gate_reads_the_draftable_population():
    """P1 decision 5. A gate that read the pooled row would repeat P1's own first reading,
    where 6x of a +0.1171 R² was mid-season signings with no preseason row for a contract
    reason."""
    passing, _ = _gate_frames(True, True, population="all")
    failing, rolling = _gate_frames(False, True, population="draftable")
    validation = pd.concat([passing, failing], ignore_index=True)
    result = AP.gate(validation, rolling, population="draftable")
    assert result["population"] == "draftable"
    assert result["val_pass"] is False
    assert result["passes"] is False
