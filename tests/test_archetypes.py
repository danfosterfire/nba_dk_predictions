import numpy as np
import pandas as pd

from src.eda.archetypes import (
    choose_k,
    cluster_profiles,
    name_cluster,
    name_clusters,
    team_composition,
)


# ── Choosing k ───────────────────────────────────────────────────────────────

def _sweep(sils: list[float], bics: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "k": range(4, 4 + len(sils)),
        "kmeans_silhouette": sils,
        "gmm_bic": bics,
    })


def test_choose_k_uses_bic_minimum_not_silhouette():
    """Silhouette declines monotonically on continuum data and would pin k to k_min."""
    sweep = _sweep([0.18, 0.15, 0.14, 0.13, 0.12], [503_740, 503_302, 502_908, 502_890, 503_341])
    k, why = choose_k(sweep)
    assert k == 7                                    # the BIC minimum, not k=4
    assert "BIC" in why and "silhouette declines" in why


def test_choose_k_honors_an_explicit_config_value():
    sweep = _sweep([0.18, 0.15, 0.14], [503_740, 503_302, 502_908])
    assert choose_k(sweep, configured=9) == (9, "configured")


def test_choose_k_omits_the_silhouette_note_when_it_is_informative():
    sweep = _sweep([0.10, 0.30, 0.12], [503_740, 503_302, 502_908])
    _, why = choose_k(sweep)
    assert "silhouette declines" not in why


# ── Naming ───────────────────────────────────────────────────────────────────

def test_name_cluster_uses_sign_and_deduplicates_concepts():
    # three rebounding columns share one label; it must appear once
    profile = pd.Series({
        "adv_reb_pct": 1.8, "usg_pct_reb": 1.7, "bas_reb": 1.6,
        "adv_usg_pct": -1.2, "bas_fg3a": 0.9,
    })
    name = name_cluster(profile, n_terms=3)
    assert name == "high rebounding, low usage, high 3-point volume"


def test_name_cluster_ignores_weak_deviations():
    profile = pd.Series({"adv_reb_pct": 1.5, "adv_usg_pct": 0.05})
    assert name_cluster(profile, n_terms=3, threshold=0.35) == "high rebounding"


def test_name_cluster_falls_back_when_nothing_stands_out():
    profile = pd.Series({"adv_reb_pct": 0.01, "adv_usg_pct": -0.02})
    assert name_cluster(profile) == "league-average profile"


def test_name_cluster_skips_unlabelled_features():
    profile = pd.Series({"xyz_unmapped_stat": 3.0, "adv_reb_pct": 1.0})
    assert name_cluster(profile) == "high rebounding"


def test_name_clusters_applies_overrides(monkeypatch):
    import src.eda.archetypes as arch
    monkeypatch.setattr(arch, "ARCHETYPE_NAME_OVERRIDES", {("A", 1): "stretch big"})
    profiles = pd.DataFrame(
        {"adv_reb_pct": [1.5, 1.5], "adv_usg_pct": [-1.0, -1.0]}, index=[0, 1])
    names = arch.name_clusters(profiles, "A")
    assert names[1] == "stretch big"
    assert names[0] != "stretch big"
    # an override for another tier must not leak
    assert arch.name_clusters(profiles, "B")[1] != "stretch big"


def test_cluster_profiles_averages_within_each_cluster():
    z = pd.DataFrame({"a": [1.0, 3.0, 10.0, 20.0], "b": [0.0, 0.0, 1.0, 1.0]})
    prof = cluster_profiles(z, np.array([0, 0, 1, 1]))
    assert prof.loc[0, "a"] == 2.0
    assert prof.loc[1, "a"] == 15.0
    assert prof.index.name == "archetype"


# ── Team composition ─────────────────────────────────────────────────────────

def _labeled() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": [1, 2, 3, 4],
        "player_name": ["A", "B", "C", "D"],
        "team_abbreviation": ["LAL", "LAL", "LAL", "BOS"],
        "season": ["2023-24"] * 4,
        "archetype": [0, 0, 1, 2],
        "min_total": [2000.0, 1000.0, 1000.0, 1500.0],
        "age": [30.0, 20.0, 25.0, 27.0],
        "dk_pts_per_game": [40.0, 20.0, 30.0, 35.0],
        "adv_usg_pct": [0.30, 0.15, 0.15, 0.25],
        "adv_pace": [100.0, 100.0, 94.0, 98.0],
        "bio_player_height_inches": [78.0, 78.0, 84.0, 80.0],
    })


def test_team_composition_shares_are_minutes_weighted_and_sum_to_one():
    comp = team_composition(_labeled(), k=3)
    lal = comp[comp["team_abbreviation"] == "LAL"].iloc[0]
    # archetype 0 holds 3000 of LAL's 4000 minutes
    assert abs(lal["share_arch0"] - 0.75) < 1e-9
    assert abs(lal["share_arch1"] - 0.25) < 1e-9
    assert abs(lal["share_arch2"] - 0.0) < 1e-9
    share_cols = [c for c in comp.columns if c.startswith("share_arch")]
    assert np.allclose(comp[share_cols].sum(axis=1), 1.0)


def test_team_composition_emits_a_column_per_archetype_even_when_unused():
    comp = team_composition(_labeled(), k=5)
    assert [c for c in comp.columns if c.startswith("share_arch")] == [
        f"share_arch{i}" for i in range(5)]
    bos = comp[comp["team_abbreviation"] == "BOS"].iloc[0]
    assert bos["share_arch2"] == 1.0 and bos["share_arch0"] == 0.0


def test_team_composition_roster_aggregates_are_minutes_weighted():
    comp = team_composition(_labeled(), k=3)
    lal = comp[comp["team_abbreviation"] == "LAL"].iloc[0]
    # (30*2000 + 20*1000 + 25*1000) / 4000
    assert abs(lal["mean_age"] - 26.25) < 1e-9
    assert abs(lal["mean_height_inches"] - (78 * 2000 + 78 * 1000 + 84 * 1000) / 4000) < 1e-9
    assert abs(lal["pace"] - (100 * 2000 + 100 * 1000 + 94 * 1000) / 4000) < 1e-9
    assert lal["roster_size"] == 3
    assert lal["minutes_covered"] == 4000.0


def test_usage_hhi_is_bounded_and_higher_when_concentrated():
    comp = team_composition(_labeled(), k=3)
    lal = comp[comp["team_abbreviation"] == "LAL"].iloc[0]
    bos = comp[comp["team_abbreviation"] == "BOS"].iloc[0]
    assert 1 / 3 <= lal["usage_hhi"] <= 1.0
    # BOS is a single player, so usage is maximally concentrated
    assert abs(bos["usage_hhi"] - 1.0) < 1e-9
    assert lal["usage_hhi"] < bos["usage_hhi"]


def test_team_composition_tolerates_missing_roster_stats():
    labeled = _labeled().drop(columns=["adv_pace", "bio_player_height_inches"])
    comp = team_composition(labeled, k=3)
    assert comp["pace"].isna().all()
    assert comp["mean_height_inches"].isna().all()
    assert not comp["mean_age"].isna().any()
