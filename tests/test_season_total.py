import numpy as np
import pandas as pd

from src.models.season_total import (
    RATE_FEATURES,
    RateModel,
    crps_from_atoms,
    evaluate,
    season_rates,
)


def _brute_crps(atoms: np.ndarray, weights: np.ndarray, y: float) -> float:
    """The definition, written literally: E|X - y| - 0.5 E|X - X'|."""
    w = weights / weights.sum()
    term1 = float(np.sum(w * np.abs(atoms - y)))
    term2 = float(sum(w[j] * w[k] * abs(atoms[j] - atoms[k])
                      for j in range(len(atoms)) for k in range(len(atoms))))
    return term1 - 0.5 * term2


# ── The CRPS reduction ────────────────────────────────────────────────────────

def test_crps_matches_the_literal_double_sum():
    """The O(K) reduction is the whole reason this is fast; pin it against the definition."""
    rng = np.random.default_rng(0)
    for _ in range(5):
        atoms = np.sort(rng.uniform(0, 100, 12))
        weights = rng.dirichlet(np.ones(12))
        y = float(rng.uniform(0, 100))
        got = crps_from_atoms(atoms[None, :], weights[None, :], np.array([y]))[0]
        assert abs(got - _brute_crps(atoms, weights, y)) < 1e-9


def test_crps_of_a_point_mass_is_absolute_error():
    """A degenerate distribution must score exactly MAE, or the units are wrong."""
    atoms = np.array([[0.0, 5.0, 10.0]])
    weights = np.array([[0.0, 1.0, 0.0]])
    assert abs(crps_from_atoms(atoms, weights, np.array([8.0]))[0] - 3.0) < 1e-12


def test_crps_rewards_a_sharper_distribution_centred_on_the_truth():
    atoms = np.array([[0.0, 10.0, 20.0]])
    sharp = crps_from_atoms(atoms, np.array([[0.05, 0.90, 0.05]]), np.array([10.0]))[0]
    vague = crps_from_atoms(atoms, np.array([[0.33, 0.34, 0.33]]), np.array([10.0]))[0]
    assert sharp < vague


def test_crps_normalizes_unnormalized_weights():
    atoms = np.array([[0.0, 10.0]])
    a = crps_from_atoms(atoms, np.array([[1.0, 1.0]]), np.array([4.0]))[0]
    b = crps_from_atoms(atoms, np.array([[0.5, 0.5]]), np.array([4.0]))[0]
    assert abs(a - b) < 1e-12


# ── The rate frame ────────────────────────────────────────────────────────────

def test_season_rates_ignores_games_he_did_not_play():
    """The rate composed with GP is "dk_pts on a night he plays"; a DNP is not a 0 rate."""
    targets = pd.DataFrame({
        "player_id": [1, 1, 1], "season": ["2021-22"] * 3,
        "dk_pts": [30.0, 20.0, 0.0], "min": [30.0, 20.0, 0.0], "played": [1, 1, 0]})
    out = season_rates(targets)
    assert out["gp_played"].iloc[0] == 2
    assert out["dk_total"].iloc[0] == 50.0
    assert out["dk_per_game"].iloc[0] == 25.0


# ── The oracle decomposition ──────────────────────────────────────────────────

def _test_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "season": ["2024-25"] * 3, "player_id": [1, 2, 3],
        "team_games": [82, 82, 82], "gp": [60, 80, 20], "gp_played": [60, 80, 20],
        "dk_total": [1800.0, 2400.0, 400.0], "dk_per_game": [30.0, 30.0, 20.0],
        "minutes_per_game_lag1": [30.0, 32.0, 12.0],
        "gp_share_lag1": [0.8, 0.9, 0.3]})


def test_both_oracles_together_reproduce_the_total_exactly():
    """`oracle_gp` must pair with the rate's own denominator or "perfect" leaves a residual."""
    frame = _test_frame()
    treatments = {"oracle_gp": {"gp": frame["gp_played"].to_numpy(dtype=float),
                                "pmf": None}}
    rows, preds = evaluate(frame, frame["dk_per_game"].to_numpy(), treatments, 82)
    assert np.allclose(preds["predicted_dk_total"], frame["dk_total"])
    mae = [r for r in rows if r["metric"] == "mae_dk_total" and r["group"] == "all"][0]
    assert mae["value"] < 1e-9


def test_full_season_treatment_is_biased_high():
    """Assuming everyone plays every game is the naive case, and it must show as bias."""
    frame = _test_frame()
    treatments = {"full_season": {"gp": frame["team_games"].to_numpy(dtype=float),
                                  "pmf": None}}
    rows, _ = evaluate(frame, frame["dk_per_game"].to_numpy(), treatments, 82)
    bias = [r for r in rows if r["metric"] == "bias_dk_total"][0]
    assert bias["value"] > 0


def test_rotation_group_is_reported_separately():
    frame = _test_frame()
    treatments = {"full_season": {"gp": frame["team_games"].to_numpy(dtype=float),
                                  "pmf": None}}
    rows, _ = evaluate(frame, frame["dk_per_game"].to_numpy(), treatments, 82)
    groups = {r["group"] for r in rows}
    assert "rotation" in groups
    rot = [r for r in rows if r["group"] == "rotation"][0]
    assert rot["n"] == 2  # player 3 is below both thresholds


# ── The fixed rate model ──────────────────────────────────────────────────────

def test_rate_model_never_predicts_a_negative_rate():
    train = pd.DataFrame({f: np.linspace(1, 10, 40) for f in RATE_FEATURES})
    train["dk_per_game"] = np.linspace(0, 40, 40)
    model = RateModel().fit(train)
    extreme = pd.DataFrame({f: np.full(3, -50.0) for f in RATE_FEATURES})
    assert (model.predict(extreme) >= 0).all()
