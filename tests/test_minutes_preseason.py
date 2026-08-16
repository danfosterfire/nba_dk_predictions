"""Tests for P3 — the preseason increment on the marginal minutes head.

Nothing here needs a sampler: the ladder is a point MLE by construction, the same argument
`tests/test_minutes_window.py` makes about the module this one borrows its head from.

Six things are pinned, each of which would move every figure in `minutes_preseason.csv`
without raising:

- **the block nests the incumbent exactly.** Every arm is "the shipped variant plus columns",
  and the whole comparison is a nested increment only if a zero block reproduces the head
  that ships. This is the house discipline (`pi = 0`, `K = 1`, `U_n = 0`, `n_rho = 1`) and
  it is the one property the gate's interval is meaningless without;
- **difference coding means what it says** — a preseason that agrees with the prior season
  is a zero, and a row with no preseason is a zero plus exactly one age indicator. A fill
  that leaked a level into the delta would be read as preseason signal;
- **the age split partitions the missing rows.** The four indicators sum to
  `1 - has_preseason`
  with no double-counted row, or P1's decision-3 census is being applied to a different
  population than the one it was measured on;
- **the reliability weight's left edge is the un-shrunk arm exactly**, so the grid's control
  is a control rather than a nearby model;
- **the centred delta removes a within-season level and nothing else** — it is the arm that
  beat the primary, and if it were also rescaling the cross-player part the finding would be
  about two changes at once;
- **the gate needs both halves.** A validation-only pass must not pass, because this project
  has twice shipped a block that won validation and shrank 4-6x on the rolling harness.
"""

import numpy as np
import pandas as pd

from src.eda.preseason_value import AGE_LABELS
from src.models import minutes_preseason as MP
from src.models.availability import FEATURE_COLS
from src.models.minutes_window import PointMinutes
from src.models.stan_minutes import OWN, variants


# ── Synthetic builders ────────────────────────────────────────────────────────

def _design(n_per_season: int = 150, seasons=("2018-19", "2019-20", "2020-21"),
            seed: int = 0) -> pd.DataFrame:
    """A frame carrying every column the point head, `variants` and the block builder read.

    `FEATURE_COLS` is filled in whole because `stan_minutes.variants` builds its expansions
    on top of it; the columns no test exercises are noise rather than dropped, so the design
    matrix has the shape the real one does.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        share = np.clip(rng.beta(5, 5, n_per_season), 0.05, 0.95)
        trials = np.full(n_per_season, 2000)
        frame = pd.DataFrame({
            "season": season,
            "player_id": np.arange(n_per_season),
            "trials": trials,
            "successes": rng.binomial(trials, share),
            OWN: np.log(share / (1 - share)),
            "minutes_share_lag1": share,
            "length_played": trials.astype(float),
        })
        for column in FEATURE_COLS:
            frame[column] = rng.normal(0.0, 1.0, n_per_season)
        frame["age"] = rng.uniform(21, 36, n_per_season)
        frame["minutes_per_game_lag1"] = rng.uniform(5, 34, n_per_season)
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def _panel(design: pd.DataFrame, missing_every: int | None = 7,
           seed: int = 1) -> pd.DataFrame:
    """A preseason panel covering all but every `missing_every`-th row of `design`.

    `None` covers everyone, which is what the difference-coding tests need — they are about
    what a *present* row encodes, and a hole in the frame would be answering measurement (c)
    instead. Only the columns `preseason_value.PANEL_COLS` reads, since the block builder
    merges through P1's own attach steps rather than reimplementing them.
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
    for column in ("fga_pre", "fgm_pre", "fg3a_pre", "fg3m_pre", "fta_pre", "ftm_pre"):
        out[column] = rng.uniform(1, 10, n)
    for column in ("fga", "fta", "reb", "ast", "stl", "blk", "tov"):
        out[f"pre_per36_{column}"] = rng.uniform(0.5, 20, n)
    return out


def _attached(design: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """The real `add_own_columns`, over P1's real attach step.

    Only the availability panel is stubbed — `attach_prior_shares` contributes exactly the
    two lag columns filled in below, and rebuilding one from synthetic game logs would test
    `src/eda/` rather than this module. Everything the tests assert on goes through the
    functions `run` calls, not through a copy of them.
    """
    from src.eda.preseason_value import attach_availability_block

    stub = design.copy()
    stub["min_share_lag1"] = 0.08
    stub["min_rank_lag1"] = 5.0
    return MP.add_own_columns(attach_availability_block(stub, panel))


# ── Nesting: the property the whole gate rests on ─────────────────────────────

def test_a_zero_block_reproduces_the_incumbent_exactly():
    """The nested-increment claim, as code.

    Every arm is the shipped feature list plus preseason columns, and the interval against
    the incumbent only measures the preseason if a block of zeros is the incumbent. A
    coefficient path through zero is what `docs/preseason-plan.md` calls the house nesting
    discipline; here the block is *held* at zero and the two fits must agree to the bit.
    """
    design = _design()
    train, val = design[design["season"] != "2020-21"], design[design["season"] == "2020-21"]
    tr, va, features = variants(train, val)["logit_own_spline"]
    zeros = [f"zero_{i}" for i in range(3)]
    for frame in (tr, va):
        for column in zeros:
            frame[column] = 0.0

    incumbent = PointMinutes(features).fit(tr)
    widened = PointMinutes(features + zeros).fit(tr)
    assert np.allclose(incumbent.mu(va), widened.mu(va), atol=1e-8)
    assert np.isclose(incumbent.rho, widened.rho, atol=1e-10)


# ── Difference coding ─────────────────────────────────────────────────────────

def test_a_preseason_that_agrees_with_the_prior_season_is_a_zero_delta():
    """Zero means "no new information" — the encoding's entire claim."""
    design = _design(n_per_season=40, seasons=("2019-20",))
    panel = _panel(design, missing_every=None)          # everyone has a row
    # Set each player's preseason MPG to exactly his prior-season share of a game.
    prior_share = 1.0 / (1.0 + np.exp(-design[OWN].to_numpy(dtype=float)))
    panel = panel.copy()
    panel["mpg_pre"] = prior_share * MP.PRESEASON_GAME_LENGTH
    out = _attached(design, panel)
    assert np.abs(out[MP.OWN_DELTA].to_numpy(dtype=float)).max() < 1e-9


def test_a_row_with_no_preseason_carries_a_zero_delta_and_one_age_indicator():
    design = _design(n_per_season=60, seasons=("2019-20",))
    out = _attached(design, _panel(design, missing_every=5))
    missing = out["has_preseason"].to_numpy(dtype=float) <= 0
    assert missing.any()
    assert np.allclose(out.loc[missing, MP.OWN_DELTA].to_numpy(dtype=float), 0.0)
    assert np.allclose(out.loc[missing, MP.CENTERED_DELTA].to_numpy(dtype=float), 0.0)

    indicators = out[MP.MISSING_AGE_COLS].to_numpy(dtype=float)
    # Exactly one cell per missing row, and none at all on a present row — so the four
    # columns partition the hole rather than overlapping it.
    assert np.allclose(indicators.sum(axis=1), missing.astype(float))


def test_the_age_indicators_sum_to_the_plain_indicator_they_replace():
    """P1 decision 3 splits `has_preseason`; a split that is not a partition is a new column.

    The census measured the outcome gap *within* age cells on the draftable frame, so the
    split has to reproduce the quantity it refines — otherwise the coefficients are read
    against a table that describes different rows.
    """
    design = _design(n_per_season=80, seasons=("2018-19", "2019-20"))
    out = _attached(design, _panel(design, missing_every=4))
    split = out[MP.MISSING_AGE_COLS].to_numpy(dtype=float).sum(axis=1)
    plain = 1.0 - out["has_preseason"].to_numpy(dtype=float)
    assert np.allclose(split, plain)


# ── The reliability shrink ────────────────────────────────────────────────────

def test_the_shrinkage_weight_at_zero_is_the_unshrunk_delta_exactly():
    """The grid's left edge is a control, not a nearby model.

    `k = 0` has to give back `own_delta` bit for bit, or "no shrinkage" is itself an arm and
    the selected `k` is chosen against a moving baseline.
    """
    design = _design(n_per_season=50, seasons=("2019-20",))
    out = _attached(design, _panel(design, missing_every=6))
    assert np.allclose(MP.reliability_weight(out, 0.0), 1.0)
    shrunk = MP.with_shrunk_delta(out, 0.0)
    assert np.allclose(shrunk[MP.SHRUNK_DELTA].to_numpy(dtype=float),
                       out[MP.OWN_DELTA].to_numpy(dtype=float))


def test_the_shrinkage_weight_is_monotone_in_preseason_minutes():
    frame = pd.DataFrame({"min_pre": [0.0, 20.0, 60.0, 140.0, 400.0]})
    weight = MP.reliability_weight(frame, 40.0)
    assert np.all(np.diff(weight) > 0)
    assert weight[0] == 0.0                 # no preseason minutes, no belief in the delta
    assert weight[-1] < 1.0


def test_a_row_with_no_preseason_is_shrunk_to_zero_rather_than_to_a_fill():
    design = _design(n_per_season=60, seasons=("2019-20",))
    out = MP.with_shrunk_delta(_attached(design, _panel(design, missing_every=5)), 40.0)
    missing = out["has_preseason"].to_numpy(dtype=float) <= 0
    assert np.allclose(out.loc[missing, MP.SHRUNK_DELTA].to_numpy(dtype=float), 0.0)


# ── The centred delta, the arm that beat the primary ──────────────────────────

def test_the_centred_delta_removes_a_within_season_level_and_leaves_the_spread():
    """Preseason minutes are compressed, so the delta's mean is far from zero.

    The centred arm is the one that beat the declared primary and repaired its bias, and it
    is only interpretable as "the level was a nuisance" if centring changes the level and
    nothing else. Missing rows keep their zero, so the nesting argument still holds.
    """
    design = _design(n_per_season=100, seasons=("2018-19", "2019-20"))
    out = _attached(design, _panel(design, missing_every=6))
    present = out["has_preseason"].to_numpy(dtype=float) > 0
    here = out[present]
    for _, block in here.groupby("season"):
        assert abs(block[MP.CENTERED_DELTA].mean()) < 1e-9
        # Spread untouched: centring is a shift, not a rescale.
        assert np.isclose(block[MP.CENTERED_DELTA].std(ddof=0),
                          block[MP.OWN_DELTA].std(ddof=0))


def test_centring_uses_each_season_alone_so_it_stays_point_in_time():
    """The centring constant is a season's own preseason mean — on disk before its opener.

    If it pooled across seasons, a validation season's column would depend on seasons the
    head has not reached yet, which is the calendar trap `adp-plan.md`'s freeze rule and P0's
    opener filter both exist to avoid.
    """
    design = _design(n_per_season=60, seasons=("2018-19", "2019-20"))
    panel = _panel(design, missing_every=None)
    baseline = _attached(design, panel)

    shifted = panel.copy()
    later = shifted["season"] == "2019-20"
    shifted.loc[later, "mpg_pre"] = shifted.loc[later, "mpg_pre"] * 1.5
    moved = _attached(design, shifted)

    earlier = (baseline["season"] == "2018-19").to_numpy()
    assert np.allclose(baseline.loc[earlier, MP.CENTERED_DELTA].to_numpy(dtype=float),
                       moved.loc[earlier, MP.CENTERED_DELTA].to_numpy(dtype=float))


# ── The gate ──────────────────────────────────────────────────────────────────

def _gate_frames(val_hi: float, roll_hi: float, origins_won: int) -> tuple:
    validation = pd.DataFrame([
        {"arm": MP.PRIMARY_ARM, "population": MP.DECISION_POPULATION,
         "crps_vs_incumbent": -5.0, "crps_vs_incumbent_lo": -9.0,
         "crps_vs_incumbent_hi": val_hi, "crps_vs_primary": 0.0,
         "crps_vs_primary_hi": 0.0}])
    rolling = pd.DataFrame([
        {"arm": MP.PRIMARY_ARM, "crps_vs_incumbent": -6.0, "crps_vs_incumbent_lo": -9.0,
         "crps_vs_incumbent_hi": roll_hi, "origins_won": origins_won, "n_origins": 13,
         "crps_vs_primary": 0.0, "crps_vs_primary_hi": 0.0,
         "verdict_vs_primary": "primary", "origins_won_vs_primary": 0}])
    return validation, rolling


def test_the_gate_needs_both_halves():
    """Validation alone ships nothing — the §12e / §14f pattern, written as a conjunction.

    Twice on this project a block won a validation reading and shrank 4-6x on the rolling
    harness, and this head's own fitting-window axis did the same (-2.687 validation,
    -0.079 rolling). So the bar is an AND, and a gate that could pass on one half would be
    the same mistake with a different block in it.
    """
    assert MP.gate(*_gate_frames(-1.5, -1.0, 12))["passes"]
    # Validation clear, rolling interval spanning zero.
    assert not MP.gate(*_gate_frames(-1.5, +0.5, 12))["passes"]
    # Rolling clear, validation interval spanning zero.
    assert not MP.gate(*_gate_frames(+0.5, -1.0, 12))["passes"]
    # Both intervals clear, but the win is carried by a minority of origins.
    assert not MP.gate(*_gate_frames(-1.5, -1.0, 6))["passes"]


def test_the_gate_gates_the_composition_and_nothing_else_does():
    """`docs/preseason-plan.md` P3: the composition is priced only if the marginal arm wins.

    Recorded as a column rather than as a printed sentence, so the go/no-go survives in the
    artifact and a later session reads a decision instead of re-deciding it.
    """
    passing = MP.gate(*_gate_frames(-1.5, -1.0, 12))
    failing = MP.gate(*_gate_frames(-1.5, +0.5, 12))
    assert passing["price_composition"] is True
    assert failing["price_composition"] is False


# ── Scope ─────────────────────────────────────────────────────────────────────

def test_the_arms_are_all_nested_supersets_of_the_reference():
    """Every arm is "the incumbent plus columns", which is what makes the ladder an ablation.

    An arm that *removed* a shipped feature would be a different head, and its interval
    against the incumbent would carry two changes at once.
    """
    arms = MP.preseason_arms()
    assert arms[MP.REFERENCE_ARM] == []
    assert MP.PRIMARY_ARM in arms
    for name, columns in arms.items():
        assert len(set(columns)) == len(columns), f"{name} repeats a column"
        if name != MP.REFERENCE_ARM:
            assert columns, f"{name} is the reference under another name"


def test_every_arm_carries_the_age_split_rather_than_a_plain_indicator():
    """P1 decision 3 applies "wherever it enters" — no arm may reintroduce the whole one."""
    for name, columns in MP.preseason_arms().items():
        if name == MP.REFERENCE_ARM:
            continue
        assert "has_preseason" not in columns, f"{name} carries the unsplit indicator"
        assert all(column in columns for column in MP.MISSING_AGE_COLS), name
