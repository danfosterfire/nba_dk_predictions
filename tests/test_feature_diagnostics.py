import numpy as np
import pandas as pd

from src.eda.feature_diagnostics import (
    AGGREGATE_FEATURES,
    SEQUENCE_FEATURES,
    _group_autocorr,
    _group_slope,
    above_null,
    cardinality,
    cell_importance,
    cell_means_r2,
    collinearity,
    condition_number,
    correlation_clusters,
    correlation_matrix,
    importance_table,
    join_keys,
    permute,
    sequence_ablation,
    sequence_features,
    vif_from_correlation,
)
from src.features.opponent import _cell_share


# ── Synthetic builders ────────────────────────────────────────────────────────

def _planted_cells(effect: float = 3.0, n_opponents: int = 30, n_archetypes: int = 9,
                   n_seasons: int = 10, per_cell: int = 8, noise: float = 1.0,
                   seed: int = 1) -> dict:
    """A panel with a real opponent effect and no archetype effect.

    `y` depends on opponent and season only, so permuting opponent must destroy
    signal while permuting archetype must not.
    """
    rng = np.random.default_rng(seed)
    opp, arch, season, y = [], [], [], []
    opp_effect = rng.normal(scale=effect, size=n_opponents)
    for o in range(n_opponents):
        for a in range(n_archetypes):
            for s in range(n_seasons):
                for _ in range(per_cell):
                    opp.append(f"o{o}")
                    arch.append(f"a{a}")
                    season.append(f"s{s}")
                    y.append(opp_effect[o] + rng.normal(scale=noise))
    return {"y": np.array(y),
            "keys": {"opponent": np.array(opp), "archetype": np.array(arch),
                     "season": np.array(season)}}


def _matrix(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A season-matrix-shaped frame carrying one exact duplicate pair."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "player_id": np.arange(n),
        "season": np.repeat(["2021-22", "2022-23"], n // 2),
        "gp": 70.0, "min": 30.0, "min_total": 2100.0, "age": 27.0,
        "bas_pts": rng.normal(20, 5, n),
        "bas_reb": rng.normal(6, 2, n),
        "adv_usg_pct": rng.normal(0.22, 0.05, n),
    })
    df["def_dreb"] = df["bas_reb"]                              # exact duplicate
    df["bas_fga"] = df["bas_pts"] * 0.5 + rng.normal(0, 0.6, n)  # ~0.97, not exact
    return df


def _seq_games(n_players: int = 200, n_games: int = 40, seed: int = 2) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seasons = ["2020-21", "2021-22", "2022-23"]
    rows = []
    for p in range(n_players):
        skill = rng.normal(25, 8)
        for s in seasons:
            for g in range(n_games):
                rows.append({
                    "player_id": p, "season": s,
                    "game_date": pd.Timestamp("2020-10-01") + pd.Timedelta(days=g),
                    "game_id": p * 1000 + g,
                    "dk_pts": skill + rng.normal(0, 6), "min": 28.0 + rng.normal(0, 3),
                })
    return pd.DataFrame(rows)


# ── cell_means_r2 agrees with the one-off it generalizes ──────────────────────

def test_cell_means_r2_matches_the_opponent_one_off():
    rng = np.random.default_rng(0)
    y = rng.normal(size=2000)
    g = rng.integers(0, 40, size=2000).astype(str)
    assert abs(cell_means_r2(y, g) - _cell_share(y, g)) < 1e-12


def test_cell_means_r2_is_one_when_cells_explain_everything():
    y = np.array([1.0, 1.0, 5.0, 5.0])
    assert abs(cell_means_r2(y, np.array(["a", "a", "b", "b"])) - 1.0) < 1e-12


def test_cell_means_r2_honours_weights():
    y = np.array([0.0, 10.0, 0.0, 10.0])
    labels = np.array(["a", "a", "b", "b"])
    w = np.array([1.0, 0.0, 0.0, 1.0])       # one row live per cell → cells fit exactly
    assert abs(cell_means_r2(y, labels, w) - 1.0) < 1e-12


def test_join_keys_combines_columns_without_collision():
    a, b = np.array(["x", "xy"]), np.array(["yz", "z"])
    assert len(np.unique(join_keys({"a": a, "b": b}))) == 2


# ── Permutation ───────────────────────────────────────────────────────────────

def test_permute_preserves_the_global_marginal():
    v = np.array(list("aaabbc"))
    out = permute(v, None, np.random.default_rng(0))
    assert sorted(out) == sorted(v)


def test_permute_within_preserves_each_stratum_marginal():
    v = np.array(["a", "a", "b", "c", "c", "d"])
    strata = np.array([1, 1, 1, 2, 2, 2])
    out = permute(v, strata, np.random.default_rng(0))
    assert sorted(out[strata == 1]) == ["a", "a", "b"]
    assert sorted(out[strata == 2]) == ["c", "c", "d"]


def test_permuting_a_variable_within_its_own_levels_is_the_identity():
    v = np.array(["s1", "s1", "s2", "s2"])
    assert list(permute(v, v, np.random.default_rng(0))) == list(v)


# ── above_null ────────────────────────────────────────────────────────────────

def test_above_null_is_zero_for_a_statistic_that_ignores_the_permuted_input():
    values = {"y": np.arange(10.0), "junk": np.arange(10.0)}
    out = above_null(lambda v: float(v["y"].sum()), values, "junk", n_shuffles=3)
    assert abs(out["above_null"]) < 1e-12
    assert out["permuted"] == "junk"


def test_above_null_reports_the_null_spread():
    rng = np.random.default_rng(0)
    values = {"y": rng.normal(size=500), "g": rng.integers(0, 50, 500).astype(str)}
    out = above_null(lambda v: cell_means_r2(v["y"], v["g"]), values, "g", n_shuffles=6)
    assert out["null_sd"] > 0
    # a random grouping over 50 cells explains something by chance, and the null says
    # how much — this is exactly the failure mode the helper exists to prevent
    assert out["null_mean"] > 0.05
    assert abs(out["above_null"]) < 0.05


# ── cell_importance / importance_table ────────────────────────────────────────

def test_cell_importance_recovers_a_planted_effect_above_its_null():
    p = _planted_cells()
    out = cell_importance(p["y"], p["keys"], "opponent",
                          within=p["keys"]["season"], n_shuffles=5)
    assert out["above_null"] > 0.10
    assert out["n_cells"] == 30 * 9 * 10
    assert out["keys"] == "opponent+archetype+season"


def test_cell_importance_finds_nothing_when_the_permuted_key_carries_nothing():
    p = _planted_cells()
    out = cell_importance(p["y"], p["keys"], "archetype",
                          within=p["keys"]["season"], n_shuffles=5)
    assert abs(out["above_null"]) < 0.02


def test_the_answer_depends_on_which_marginal_is_permuted():
    """The finding the helper is built around, planted so it cannot be missed."""
    p = _planted_cells()
    table = importance_table(p["y"], p["keys"], within_key="season",
                             n_shuffles=5).set_index("permuted")
    assert table.loc["opponent", "statistic"] == table.loc["archetype", "statistic"]
    assert table.loc["opponent", "above_null"] > 5 * table.loc["archetype", "above_null"]


def test_importance_table_skips_the_stratifying_key():
    p = _planted_cells()
    table = importance_table(p["y"], p["keys"], within_key="season", n_shuffles=2)
    assert set(table["permuted"]) == {"opponent", "archetype"}


def test_cell_importance_rejects_a_key_it_was_not_given():
    p = _planted_cells()
    try:
        cell_importance(p["y"], p["keys"], "not_a_key")
        raise AssertionError("expected KeyError for an unknown marginal")
    except KeyError as exc:
        assert "not_a_key" in str(exc)


# ── Collinearity ──────────────────────────────────────────────────────────────

def test_vif_is_infinite_for_an_exactly_duplicated_column():
    """A pseudo-inverse would report ~0.25 here and hide the worst case in the block."""
    table, R, summary = collinearity(_matrix())
    by_name = table.set_index("name")
    assert np.isinf(by_name.loc["def_dreb", "vif"])
    assert np.isinf(by_name.loc["bas_reb", "vif"])
    assert by_name.loc["def_dreb", "max_abs_corr"] > 0.999
    assert by_name.loc["def_dreb", "max_corr_partner"] == "bas_reb"
    # a merely-correlated column gets a large finite VIF, not infinity
    assert 5 < by_name.loc["bas_fga", "vif"] < 1e6
    assert by_name.loc["adv_usg_pct", "vif"] < 2


def test_correlation_clusters_group_duplicates_and_leave_singletons_alone():
    table, _, summary = collinearity(_matrix(), threshold=0.95)
    by_name = table.set_index("name")
    assert by_name.loc["def_dreb", "cluster_id"] == by_name.loc["bas_reb", "cluster_id"]
    assert by_name.loc["bas_pts", "cluster_id"] == by_name.loc["bas_fga", "cluster_id"]
    assert by_name.loc["adv_usg_pct", "cluster_size"] == 1
    assert summary["largest_cluster"] == 2


def test_correlation_clusters_split_when_the_threshold_rises():
    _, R, _ = collinearity(_matrix())
    loose = correlation_clusters(R, threshold=0.95)
    strict = correlation_clusters(R, threshold=0.999)
    # at 0.95 the ~0.97 pair clusters too; at 0.999 only the exact duplicate survives
    assert strict.nunique() > loose.nunique()


def test_condition_number_is_near_one_for_an_uncorrelated_block():
    X = np.random.default_rng(0).normal(size=(4000, 4))
    assert condition_number(correlation_matrix(X, list("abcd"))) < 1.3


def test_an_exact_duplicate_makes_the_block_singular_and_says_by_how_much():
    _, R, summary = collinearity(_matrix())
    assert np.isinf(summary["condition_number"])
    assert summary["matrix_rank"] == summary["n_features"] - 1   # one duplicate pair
    assert summary["n_vif_above_flag"] >= 2


def test_vif_from_correlation_does_not_raise_on_a_singular_block():
    R = pd.DataFrame([[1.0, 1.0], [1.0, 1.0]], index=["a", "b"], columns=["a", "b"])
    assert list(vif_from_correlation(R)) == [np.inf, np.inf]


# ── Cardinality ───────────────────────────────────────────────────────────────

def test_effective_category_count_falls_below_the_nominal_one_when_skewed():
    values = pd.Series(["big"] * 990 + ["rare1"] * 5 + ["rare2"] * 5)
    seasons = pd.Series(["2023-24"] * len(values))
    out = cardinality(values, seasons, min_n=50)
    assert out["n_categories"] == 3
    assert out["n_effective"] < 1.1          # behaves as if there is one category
    assert out["n_below_threshold"] == 2
    assert "rare1" in out["cold_start"]


def test_cardinality_reports_the_shortest_span_a_cold_start_entity_covers():
    values = pd.Series(["LAL"] * 30 + ["SEA"] * 3)
    seasons = pd.Series([f"s{i % 30}" for i in range(30)] + ["s0", "s1", "s2"])
    out = cardinality(values, seasons, min_n=10)
    assert out["min_seasons_spanned"] == 3
    assert out["min_n"] == 3


# ── Sequence features ─────────────────────────────────────────────────────────

def test_group_slope_recovers_a_planted_slope():
    df = pd.DataFrame({"player_id": [1] * 5 + [2] * 5, "season": "s",
                       "x": list(range(5)) * 2,
                       "y": [3.0 * i for i in range(5)] + [-2.0 * i for i in range(5)]})
    slopes = _group_slope(df, ["player_id", "season"], "x", "y")
    assert abs(slopes[(1, "s")] - 3.0) < 1e-9
    assert abs(slopes[(2, "s")] - (-2.0)) < 1e-9


def test_group_autocorr_is_negative_for_an_alternating_series():
    df = pd.DataFrame({"player_id": 1, "season": "s", "y": [0.0, 1.0] * 12})
    assert _group_autocorr(df, ["player_id", "season"], "y")[(1, "s")] < -0.9


def test_aggregate_features_are_invariant_to_shuffling_the_game_order():
    """The property that makes the shuffled arm a fair null rather than a weaker model."""
    games = _seq_games(n_players=30, n_games=25)
    real = sequence_features(games).set_index(["player_id", "season"])
    fake = sequence_features(games, shuffle_seed=11).set_index(["player_id", "season"])
    for col in AGGREGATE_FEATURES:
        assert np.allclose(real[col], fake.loc[real.index, col])


def test_order_dependent_features_do_move_when_the_order_is_shuffled():
    games = _seq_games(n_players=30, n_games=25)
    real = sequence_features(games).set_index(["player_id", "season"])
    fake = sequence_features(games, shuffle_seed=11).set_index(["player_id", "season"])
    moved = [c for c in SEQUENCE_FEATURES
             if not np.allclose(real[c], fake.loc[real.index, c], equal_nan=True)]
    assert set(moved) == set(SEQUENCE_FEATURES)


def test_sequence_ablation_finds_no_order_information_in_iid_games():
    """Games drawn i.i.d. within a season carry no order, so the arms must tie."""
    games = _seq_games(n_players=250, n_games=30)
    out = sequence_ablation(games, ["2020-21", "2021-22", "2022-23"], test_seasons=1)
    assert out["n_test"] > 100
    assert abs(out["delta_above_null"]) < 0.03
    assert out["r2_aggregate"] > 0.4          # the season mean is genuinely predictive
