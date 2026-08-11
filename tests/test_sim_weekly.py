"""Tests for the weekly-scores emitter — the reduction, the gates, and the split guard.

None of these needs a tensor, a posterior artifact or a sampler. What is worth pinning in
`src/sim/weekly.py` is what fails **silently**: a period map that quietly loses a game, a
facet that quietly pools a two-week total into a distribution of one-week ones, a spread
statistic that reports the wrong one of three, and a split vocabulary that quietly widens.
Every one of those renders as a perfectly good-looking page.

Plain `assert` with synthetic builders, no fixtures or classes, mirroring
`tests/test_preprocess.py` and `tests/test_sim_season.py`. The last group reads the
**shipped** artifacts and skips rather than fails on a fresh checkout, the way
`tests/test_model_cards.py` does, since `make weekly-scores` needs `make simulate-season`
first.

`--draws` is deliberately absent from every synthetic case here: the budget is *measured*
by `check_budget` at build time, and a test that asserted a specific number of simulated
seasons would be pinning a decision rather than a mechanism.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.models.model_cards import SPLITS as CARD_SPLITS
from src.sim import weekly as W

ROOT = Path(__file__).resolve().parent.parent


# ── Synthetic builders ────────────────────────────────────────────────────────

def _period_frame(rounds: tuple[int, ...] = (1,) * 17 + (2, 3, 4)) -> pd.DataFrame:
    """One row per slot, in the shape `period_frame` returns."""
    weeks = [1 if r == 1 else 2 for r in rounds]
    return pd.DataFrame({
        "season": "2021-22", "slot": np.arange(len(rounds)),
        "tournament_round": list(rounds), "weeks": weeks,
        "games": [50] * len(rounds), "start": "2021-10-18", "end": "2021-10-24",
        "period_type": [W.WEEK if w == 1 else W.DOUBLE_WEEK for w in weeks]})


def _tensor(n_units: int = 4, n_sims: int = 8, value: float = 10.0,
            season: str = "2021-22") -> dict:
    return {"season": season,
            "dk_pts": np.full((n_units, W.N_SCORING_PERIODS, n_sims), value,
                              dtype=np.float32),
            "player_id": np.arange(100, 100 + n_units, dtype=np.int64),
            "tournament_round": np.r_[np.ones(W.ROUND_1_WEEKS, dtype=int), [2, 3, 4]],
            "fit_window": W.FIT_WINDOW, "n_sims": n_sims,
            "n_posterior_draws": 4, "seed": 0}


# ── The unit — three of the twenty periods are not a week ────────────────────

def test_the_two_period_types_are_the_two_lengths_a_slot_can_be():
    """Seventeen one-week slots and three double weeks, derived rather than declared.

    The facet is the whole reason the page's panels are not pooled: a double week carries
    twice the games, so its `dk_pts` distribution has a right tail that is a calendar fact.
    Nothing in the emitter hard-codes 17 and 3 — `period_frame` counts distinct
    `period_index` values inside each slot — so a season grid that moved would change this
    rather than be quietly mis-labelled.
    """
    frame = _period_frame()
    counts = frame["period_type"].value_counts()
    assert counts[W.WEEK] == W.ROUND_1_WEEKS
    assert counts[W.DOUBLE_WEEK] == W.N_SCORING_PERIODS - W.ROUND_1_WEEKS
    assert set(W.PERIOD_TYPES) == set(frame["period_type"])


def test_a_slot_spanning_a_third_number_of_weeks_raises_rather_than_facets():
    """A three-week slot is a period grid that has moved, not a new facet.

    `PERIOD_WEEKS` is a closed vocabulary for the same reason `SPLITS` is: the alternative
    is a page silently growing a third column that looks like a feature.
    """
    frame = _period_frame()
    assert set(frame["weeks"]) <= set(W.PERIOD_WEEKS.values())
    assert 3 not in W.PERIOD_WEEKS.values()


def test_a_season_missing_a_whole_round_is_refused_rather_than_scored_as_zeros():
    """2020-21 is the real case: it has no Round 4 at all, so slot 19 has no games.

    Pooled into the double-week facet it would contribute a column of structural zeros to
    both the observed and the simulated side — a season that "under-performs" for a reason
    that is the calendar. The check is why `TRAIN_SEASONS` is 2018-19 and 2021-22 rather
    than the last two.
    """
    short = _period_frame()[:-1]                          # no slot 19
    with pytest.raises(AssertionError, match="carry no scheduled game"):
        W.assert_covers_the_tensor(short, _tensor()["tournament_round"], "2020-21")


def test_a_tensor_whose_round_map_disagrees_with_the_grid_is_refused():
    """The tensor predating the current period grid is invisible in every number.

    Both files would be well-formed and every panel would render; the rows would simply be
    describing different weeks on the two sides of the comparison.
    """
    wrong = np.r_[np.ones(16, dtype=int), [2, 2, 3, 4]]
    with pytest.raises(AssertionError, match="disagrees with"):
        W.assert_covers_the_tensor(_period_frame(), wrong, "2021-22")


def test_a_tensor_declaring_a_different_period_count_is_refused():
    with pytest.raises(AssertionError, match="scoring periods against"):
        W.assert_covers_the_tensor(_period_frame(), np.ones(19, dtype=int), "2021-22")


def test_the_shipped_training_pair_excludes_the_two_broken_seasons():
    """A measured scope decision, pinned so it cannot drift back to "the last two"."""
    assert W.TRAIN_SEASONS == ("2018-19", "2021-22")
    assert "2020-21" not in W.TRAIN_SEASONS and "2019-20" not in W.TRAIN_SEASONS


# ── The split — locked, and by the same door every head goes through ─────────

def test_the_split_vocabulary_is_the_model_cards_one():
    """One closed pair across both emitters, so a page cannot render what another refuses."""
    assert W.SPLITS == CARD_SPLITS == ("train", "validation")


def test_a_third_split_is_refused_on_the_way_to_disk():
    frame = pd.DataFrame({"split": ["train", "validation", "test"], "value": [1, 2, 3]})
    with pytest.raises(AssertionError, match="outside"):
        W._check_splits(frame, "weekly_score_index.csv")


def test_a_frame_with_no_split_column_passes_through_unchanged():
    frame = pd.DataFrame({"slot": [0, 1]})
    assert W._check_splits(frame, "x.csv") is frame


def test_the_window_guard_refuses_a_tensor_that_read_the_seasons_it_scores():
    """A `train_val` tensor has seen 2022-23 and 2023-24 *through its coefficients*.

    No frame-level guard can see that, which is why the check sits where the artifact is
    consumed rather than where it was produced — `posteriors.require_window`'s argument one
    layer up.
    """
    tensor = _tensor()
    W.assert_window(tensor)                                # the shipped window passes
    tensor["fit_window"] = "train_val"
    with pytest.raises(AssertionError, match="train_val"):
        W.assert_window(tensor)


# ── The reduction ─────────────────────────────────────────────────────────────

def test_a_facet_takes_only_its_own_slots():
    """The one-week facet must not pick up a double week, which is twice the unit."""
    tensors = {"2021-22": _tensor(value=7.0)}
    periods = {"2021-22": _period_frame()}
    observed = {"2021-22": np.full((4, W.N_SCORING_PERIODS), 3.0)}
    keep = {"2021-22": np.ones(4, dtype=bool)}

    week = W.panel_rows(tensors, observed, periods, keep, W.WEEK, ["2021-22"], 8)
    double = W.panel_rows(tensors, observed, periods, keep, W.DOUBLE_WEEK, ["2021-22"], 8)
    assert len(week["observed"]) == 4 * W.ROUND_1_WEEKS
    assert len(double["observed"]) == 4 * 3
    assert set(week["index"]["slot"]) == set(range(W.ROUND_1_WEEKS))
    assert set(double["index"]["slot"]) == {17, 18, 19}


def test_the_draw_block_is_sims_by_rows_so_one_column_slice_is_one_season():
    """`(sims x rows)` is what the ECDF ribbon and `crps_from_samples` both require.

    A transposed block would give an ECDF *per player* rather than per replicate season,
    which still draws a ribbon and answers a different question.
    """
    tensors = {"2021-22": _tensor(n_units=4, n_sims=8)}
    panel = W.panel_rows(tensors, {"2021-22": np.zeros((4, W.N_SCORING_PERIODS))},
                         {"2021-22": _period_frame()},
                         {"2021-22": np.ones(4, dtype=bool)}, W.WEEK, ["2021-22"], 8)
    assert panel["draws"].shape == (8, 4 * W.ROUND_1_WEEKS)
    assert len(panel["observed"]) == panel["draws"].shape[1]


def test_a_player_with_no_realized_game_is_dropped_from_the_panel():
    """The same population `season.gate_a` scores: his twenty zeros are a roster fact."""
    keep = {"2021-22": np.array([True, False, True, True])}
    panel = W.panel_rows({"2021-22": _tensor()},
                         {"2021-22": np.zeros((4, W.N_SCORING_PERIODS))},
                         {"2021-22": _period_frame()}, keep, W.WEEK, ["2021-22"], 8)
    assert set(panel["index"]["player_id"]) == {100, 102, 103}


def test_seasons_stack_into_one_facet_rather_than_being_averaged():
    """A split is its seasons pooled at the row level, so a bigger season carries more."""
    tensors = {"2018-19": _tensor(n_units=2, season="2018-19"),
               "2021-22": _tensor(n_units=5)}
    periods = {s: _period_frame() for s in tensors}
    observed = {s: np.zeros((t["dk_pts"].shape[0], W.N_SCORING_PERIODS))
                for s, t in tensors.items()}
    keep = {s: np.ones(t["dk_pts"].shape[0], dtype=bool) for s, t in tensors.items()}
    panel = W.panel_rows(tensors, observed, periods, keep, W.WEEK,
                         ["2018-19", "2021-22"], 8)
    assert len(panel["observed"]) == (2 + 5) * W.ROUND_1_WEEKS
    assert set(panel["index"]["season"]) == {"2018-19", "2021-22"}


def test_an_empty_facet_returns_an_empty_panel_rather_than_raising():
    keep = {"2021-22": np.zeros(4, dtype=bool)}
    panel = W.panel_rows({"2021-22": _tensor()},
                         {"2021-22": np.zeros((4, W.N_SCORING_PERIODS))},
                         {"2021-22": _period_frame()}, keep, W.WEEK, ["2021-22"], 8)
    assert len(panel["observed"]) == 0 and panel["index"].empty


# ── The seed ──────────────────────────────────────────────────────────────────

def test_the_randomization_seed_is_stable_and_one_stream_per_facet():
    """Deterministic across rebuilds, or a reader cannot tell an RNG from a re-simulation.

    `zlib.crc32` rather than `hash`, which is salted per interpreter — the same device
    `model_cards._seed` uses.
    """
    seeds = {(p, s): W.facet_seed(p, s) for p in W.PERIOD_TYPES for s in W.SPLITS}
    assert len(set(seeds.values())) == len(seeds)
    assert W.facet_seed(W.WEEK, "train") == seeds[(W.WEEK, "train")]


# ── The budget gate ───────────────────────────────────────────────────────────

def _summary(**overrides) -> dict:
    base = {"ecdf_band_mc": 0.004, "ecdf_band_gated": True,
            "ks": 0.15, "ks_mc": 0.001, "ks_gated": True}
    return {**base, **overrides}


def test_a_wide_ks_distance_ships_and_only_the_budget_is_gated():
    """**The rule this whole reading exists under.** A distance is never a pass/fail.

    At n ≈ 10⁴ a strict uniformity test rejects every model in this project, so a bar on the
    distance would report failure everywhere. The only bars are on the simulated-season
    budget behind it.
    """
    W.check_budget(W.WEEK, "train", _summary(ks=0.9), draws=500)


def test_an_unstable_ribbon_fails_the_build_rather_than_being_drawn():
    with pytest.raises(AssertionError, match="ECDF ribbon moves"):
        W.check_budget(W.WEEK, "train", _summary(ecdf_band_mc=0.5), draws=500)


def test_an_unstable_ks_fails_the_build_rather_than_being_tiled():
    with pytest.raises(AssertionError, match="KS distance moves"):
        W.check_budget(W.WEEK, "train", _summary(ks_mc=0.5), draws=500)


def test_an_ungated_facet_is_reported_rather_than_failed():
    """Under `MIN_GATED_ROWS` a half-sample gap measures the frame, not the budget."""
    W.check_budget(W.DOUBLE_WEEK, "validation",
                   _summary(ecdf_band_mc=0.5, ecdf_band_gated=False,
                            ks_mc=0.5, ks_gated=False), draws=500)


def test_the_gated_row_count_is_the_model_cards_one():
    """One threshold across both emitters rather than a second copy that can drift."""
    from src.models.model_cards import BAND_MIN_ROWS

    assert W.MIN_GATED_ROWS == BAND_MIN_ROWS


# ── The shipped artifacts ─────────────────────────────────────────────────────

def _cfg() -> dict:
    return yaml.safe_load((ROOT / "configs" / "default.yaml").read_text())


def _shipped(name: str) -> pd.DataFrame:
    path = ROOT / _cfg()["evaluation"]["predictions_dir"] / name
    if not path.exists():
        pytest.skip(f"{name} not built — run `make weekly-scores`")
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def test_every_shipped_artifact_carries_only_the_two_splits():
    for name in W.ARTIFACTS:
        frame = _shipped(name)
        if "split" in frame.columns:
            assert set(frame["split"]) <= set(W.SPLITS), name


def test_no_shipped_row_comes_from_a_test_season():
    """The seasons themselves, not just the labels — the split guard's actual promise."""
    index = _shipped("weekly_score_index.csv")
    period = _shipped("weekly_score_period.csv")
    named = {s for row in index["seasons"] for s in str(row).split(",")}
    assert named == set(period["season"].astype(str))
    assert not {"2024-25", "2025-26"} & named


def test_every_facet_ships_all_four_panels_and_a_matching_row_count():
    """A facet present in one artifact and absent from another is a half-drawn page."""
    index = _shipped("weekly_score_index.csv")
    facets = set(map(tuple, index[["period_type", "split"]].to_numpy()))
    for name in ("weekly_score_ecdf.csv", "weekly_score_calibration.csv",
                 "weekly_score_quantile.csv", "weekly_score_sample.parquet"):
        frame = _shipped(name)
        assert set(map(tuple, frame[["period_type", "split"]].to_numpy())) == facets, name


def test_the_calibration_cells_count_every_row_of_their_own_facet():
    """Tails are clipped **into** the end bins rather than dropped, so nothing is lost."""
    calibration = _shipped("weekly_score_calibration.csv")
    for (period_type, split), cells in calibration.groupby(["period_type", "split"]):
        assert int(cells["count"].sum()) == int(cells["n"].iloc[0]), (period_type, split)


def test_the_quantile_panels_stay_inside_the_unit_square():
    """Both axes are a PIT and a rank transform, so anything outside [0, 1] is a bug."""
    quantile = _shipped("weekly_score_quantile.csv")
    for column in ("x", "y"):
        values = quantile[column].dropna()
        assert values.between(-1e-9, 1 + 1e-9).all(), column


def test_the_index_ks_is_the_one_the_panels_carry():
    """Two readings of one number, so a tile and its figure cannot disagree."""
    index = _shipped("weekly_score_index.csv")
    quantile = _shipped("weekly_score_quantile.csv")
    for _, row in index.iterrows():
        part = quantile[(quantile["period_type"] == row["period_type"])
                        & (quantile["split"] == row["split"])]
        assert abs(float(part["ks"].iloc[0]) - float(row["ks"])) < 1e-6


def test_every_shipped_facet_clears_the_budget_bars():
    from src.models.model_cards import ECDF_BAND_TOL, KS_MC_TOL

    for _, row in _shipped("weekly_score_index.csv").iterrows():
        if row["ecdf_band_gated"]:
            assert float(row["ecdf_band_mc"]) <= ECDF_BAND_TOL, row["period_type"]
        if row["ks_gated"]:
            assert float(row["ks_mc"]) <= KS_MC_TOL, row["period_type"]


def test_the_per_period_readout_covers_every_slot_of_every_season():
    period = _shipped("weekly_score_period.csv")
    for season, rows in period.groupby("season"):
        assert set(rows["slot"]) == set(range(W.N_SCORING_PERIODS)), season
        assert set(rows["period_type"]) <= set(W.PERIOD_TYPES)


def test_the_period_readout_and_the_index_agree_on_the_facet_row_counts():
    """The pooled panel is exactly its own periods stacked — the reduction, checked twice."""
    index = _shipped("weekly_score_index.csv")
    period = _shipped("weekly_score_period.csv")
    for _, row in index.iterrows():
        part = period[(period["split"] == row["split"])
                      & (period["period_type"] == row["period_type"])]
        assert int(part["n"].sum()) == int(row["n"]), (row["period_type"], row["split"])


def test_every_shipped_facet_is_at_the_train_fit_window():
    assert set(_shipped("weekly_score_index.csv")["fit_window"]) == {W.FIT_WINDOW}
