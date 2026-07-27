import numpy as np
import torch

from src.features.targets import BONUS_CATEGORIES, COMPONENTS, DK_WEIGHTS, expected_bonus
from src.models.multihead import (
    MultiHeadLoss,
    MultiHeadPlayerModel,
    expected_bonus_analytic,
    poisson_tail_ge,
)

B, T, F, S = 8, 10, 15, 4


def _model(static: int = 0) -> MultiHeadPlayerModel:
    torch.manual_seed(0)
    return MultiHeadPlayerModel(seq_input_size=F, static_input_size=static,
                                hidden_size=16, num_layers=1)


def _batch(static: int = 0):
    torch.manual_seed(1)
    seq = torch.randn(B, T, F)
    return seq, (torch.randn(B, static) if static else None)


def _target(played=None, minutes=None):
    played = torch.ones(B) if played is None else played
    minutes = torch.full((B,), 30.0) if minutes is None else minutes
    tgt = {"played": played, "minutes": minutes}
    for c in COMPONENTS:
        tgt[c] = torch.full((B,), 5.0)
    return tgt


# ── Poisson tail ─────────────────────────────────────────────────────────────

def test_poisson_tail_matches_scipy_style_reference():
    """P(X >= 10) computed by explicit pmf sum, checked against a direct calculation."""
    import math
    for lam in [1.0, 5.0, 9.5, 20.0]:
        ref = 1.0 - sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(10))
        got = float(poisson_tail_ge(torch.tensor([lam]))[0])
        assert abs(got - ref) < 1e-6, (lam, got, ref)


def test_poisson_tail_is_monotone_and_bounded():
    lam = torch.tensor([0.1, 1.0, 5.0, 10.0, 30.0, 100.0])
    p = poisson_tail_ge(lam)
    assert torch.all(p[1:] >= p[:-1])
    assert float(p.min()) >= 0.0 and float(p.max()) <= 1.0


# ── Analytic bonus ───────────────────────────────────────────────────────────

def test_analytic_bonus_agrees_with_independent_monte_carlo():
    rng = np.random.default_rng(0)
    lam = np.abs(rng.normal(8.0, 4.0, size=(500, 5)))
    ana = expected_bonus_analytic(
        {c: torch.tensor(lam[:, i]) for i, c in enumerate(BONUS_CATEGORIES)}).numpy()
    mc = expected_bonus(lam, overdispersion=1e-9, n_samples=8000, seed=3)
    assert np.corrcoef(ana, mc)[0, 1] > 0.999
    assert abs(ana.mean() - mc.mean()) < 0.01


def test_analytic_bonus_is_bounded_by_the_payout():
    tiny = expected_bonus_analytic({c: torch.tensor([0.01]) for c in BONUS_CATEGORIES})
    huge = expected_bonus_analytic({c: torch.tensor([200.0]) for c in BONUS_CATEGORIES})
    assert float(tiny[0]) < 1e-6
    assert 4.4 < float(huge[0]) <= 4.5


def test_analytic_bonus_probabilities_form_a_distribution():
    """Only two categories can plausibly clear 10, so the payout sits near 1.5."""
    counts = dict(zip(BONUS_CATEGORIES, [torch.tensor([40.0]), torch.tensor([40.0]),
                                         torch.tensor([0.01]), torch.tensor([0.01]),
                                         torch.tensor([0.01])]))
    assert abs(float(expected_bonus_analytic(counts)[0]) - 1.5) < 1e-3


def test_gradient_flows_through_the_bonus():
    lam = torch.tensor([8.0], requires_grad=True)
    expected_bonus_analytic({c: lam for c in BONUS_CATEGORIES}).sum().backward()
    assert lam.grad is not None and float(lam.grad) > 0.0


# ── Model plumbing ───────────────────────────────────────────────────────────

def test_forward_returns_expected_keys_and_shapes():
    out = _model()(*_batch())
    assert set(out) == {"play_logit", "log_minutes", "log_rates", "rates"}
    assert out["play_logit"].shape == (B,)
    assert out["log_minutes"].shape == (B,)
    assert out["log_rates"].shape == (B, len(COMPONENTS))
    assert torch.allclose(out["rates"], out["log_rates"].exp())


def test_static_features_are_used_when_declared():
    model = _model(static=S)
    seq, static = _batch(S)
    a = model(seq, static)["log_rates"]
    b = model(seq, static * 0 + 5.0)["log_rates"]
    assert not torch.allclose(a, b), "static path had no effect on the output"


def test_missing_static_features_raise_rather_than_pass_silently():
    model = _model(static=S)
    try:
        model(_batch()[0], None)
        raise AssertionError("expected ValueError when static features are omitted")
    except ValueError as exc:
        assert "static" in str(exc)


def test_expected_counts_apply_the_minutes_exposure():
    model = _model()
    out = model(*_batch())
    counts = model.expected_counts(out)
    exposure = out["log_minutes"].exp() / 36.0
    for i, c in enumerate(COMPONENTS):
        assert torch.allclose(counts[c], out["rates"][:, i] * exposure)


def test_expected_dk_pts_equals_weighted_counts_plus_bonus():
    model = _model()
    out = model(*_batch())
    counts = model.expected_counts(out)
    linear = sum(counts[c] * DK_WEIGHTS[c] for c in COMPONENTS)
    bonus = expected_bonus_analytic(counts)
    got = model.expected_dk_pts(out, scale_by_availability=False)
    assert torch.allclose(got, linear + bonus, atol=1e-5)


def test_expected_dk_pts_can_exclude_bonus_and_availability():
    model = _model()
    out = model(*_batch())
    with_bonus = model.expected_dk_pts(out, include_bonus=True, scale_by_availability=False)
    without = model.expected_dk_pts(out, include_bonus=False, scale_by_availability=False)
    assert torch.all(with_bonus >= without)

    scaled = model.expected_dk_pts(out, scale_by_availability=True)
    assert torch.allclose(scaled, with_bonus * torch.sigmoid(out["play_logit"]), atol=1e-5)


# ── Loss ─────────────────────────────────────────────────────────────────────

def test_loss_returns_all_three_parts():
    model = _model()
    total, parts = MultiHeadLoss()(model(*_batch()), _target())
    assert set(parts) == {"play", "minutes", "components"}
    assert np.isfinite(total.item())


def test_dnp_games_do_not_affect_the_rate_and_minutes_losses():
    """Rate heads must be masked on DNPs — a DNP carries no rate information."""
    model = _model()
    out = model(*_batch())
    played = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])

    a = _target(played=played, minutes=torch.tensor([30.0] * 4 + [0.0] * 4))
    b = {k: v.clone() for k, v in a.items()}
    for c in COMPONENTS:                       # corrupt the DNP rows only
        b[c][4:] = 999.0

    _, pa = MultiHeadLoss()(out, a)
    _, pb = MultiHeadLoss()(out, b)
    assert abs(pa["components"] - pb["components"]) < 1e-9
    assert abs(pa["minutes"] - pb["minutes"]) < 1e-9


def test_loss_is_finite_when_nobody_played():
    model = _model()
    out = model(*_batch())
    tgt = _target(played=torch.zeros(B), minutes=torch.zeros(B))
    total, parts = MultiHeadLoss()(out, tgt)
    assert np.isfinite(total.item())
    assert parts["components"] == 0.0 and parts["minutes"] == 0.0
    total.backward()          # must not raise on the empty-mask path


def test_component_loss_prefers_the_correct_rate():
    """Poisson NLL with exposure must be minimized at the true rate."""
    loss_fn = MultiHeadLoss()
    minutes = torch.full((B,), 36.0)
    true_rate = 6.0
    tgt = {"played": torch.ones(B), "minutes": minutes}
    for c in COMPONENTS:
        tgt[c] = torch.full((B,), true_rate)

    def component_loss(rate):
        out = {"play_logit": torch.zeros(B), "log_minutes": minutes.log(),
               "log_rates": torch.full((B, len(COMPONENTS)), float(np.log(rate)))}
        return loss_fn(out, tgt)[1]["components"]

    assert component_loss(true_rate) < component_loss(true_rate * 2)
    assert component_loss(true_rate) < component_loss(true_rate / 2)


def test_exposure_offset_makes_the_rate_minutes_invariant():
    """The same per-36 rate must be optimal whether the player played 18 or 36 minutes."""
    loss_fn = MultiHeadLoss()
    rate = 6.0

    def loss_at(minutes_val, rate_guess):
        minutes = torch.full((B,), minutes_val)
        tgt = {"played": torch.ones(B), "minutes": minutes}
        for c in COMPONENTS:                       # counts scale with exposure
            tgt[c] = torch.full((B,), rate * minutes_val / 36.0)
        out = {"play_logit": torch.zeros(B), "log_minutes": minutes.log(),
               "log_rates": torch.full((B, len(COMPONENTS)), float(np.log(rate_guess)))}
        return loss_fn(out, tgt)[1]["components"]

    for minutes_val in (18.0, 36.0):
        assert loss_at(minutes_val, rate) < loss_at(minutes_val, rate * 1.5)
        assert loss_at(minutes_val, rate) < loss_at(minutes_val, rate / 1.5)


def test_loss_weights_rescale_the_groups():
    model = _model()
    out, tgt = model(*_batch()), _target()
    base, _ = MultiHeadLoss()(out, tgt)
    upweighted, _ = MultiHeadLoss(weights={"play": 10.0})(out, tgt)
    assert not np.isclose(base.item(), upweighted.item())


def test_training_reduces_the_loss():
    model = _model(static=S)
    seq, static = _batch(S)
    tgt = _target(minutes=torch.full((B,), 28.0))
    loss_fn, opt = MultiHeadLoss(), torch.optim.Adam(model.parameters(), lr=1e-2)

    first = None
    for _ in range(60):
        opt.zero_grad()
        loss, _ = loss_fn(model(seq, static), tgt)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    assert loss.item() < first
