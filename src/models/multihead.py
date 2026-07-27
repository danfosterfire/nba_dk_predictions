"""Multi-head model: availability × per-36 component rates, recombined into dk_pts.

    dk_pts = [ P(play) · minutes ] × Σ wᵢ·rateᵢ  +  E[bonus]

Structure, and why it is this shape:

- A **sequence trunk** over the player's prior-season game logs (as the existing
  single-head models use), plus a **static path** for season-level features that are
  not sequences — team context, archetype membership, opponent profile. Under the
  project's prediction-time constraint those static features are constant within a
  player-season, so feeding them through the sequence trunk would be wasteful.
- A **minutes head** separate from the rate heads, because measured team-context
  effects push rates and minutes in opposite directions and cancel in the product.
  Minutes alone explains 18.6% of within-player-season residual variance.
- **Poisson rate heads with a minutes exposure offset**, the canonical form for counts
  observed over varying exposure. Points are overdispersed for a true Poisson, but
  Poisson NLL remains a consistent quasi-likelihood for the mean, which is what the
  linear dk_pts sum needs.
- An **analytic differentiable bonus** (see `expected_bonus_analytic`) rather than the
  Monte-Carlo version in src/features/targets.py, so it can sit inside a training loss.
"""

import math

import torch
import torch.nn as nn

from src.features.targets import BONUS_CATEGORIES, COMPONENTS, DK_WEIGHTS, EXPOSURE_MINUTES

BONUS_TWO = 1.5      # payout for a double-double
BONUS_THREE = 4.5    # payout for a triple-double or better


# ── Differentiable expected bonus ─────────────────────────────────────────────

def poisson_tail_ge(lam: torch.Tensor, threshold: int = 10) -> torch.Tensor:
    """P(X ≥ threshold) for X ~ Poisson(lam), by explicit pmf sum.

    Summing the `threshold` low-order pmf terms keeps this expressible in plain ops
    with reliable gradients, rather than depending on incomplete-gamma autograd.
    """
    lam = lam.clamp_min(1e-8)
    log_lam = lam.log()
    cdf = torch.zeros_like(lam)           # accumulates P(X ≤ threshold-1)
    log_fact = 0.0
    for k in range(threshold):
        if k > 0:
            log_fact += math.log(k)
        cdf = cdf + torch.exp(-lam + k * log_lam - log_fact)
    return (1.0 - cdf).clamp(0.0, 1.0)


def expected_bonus_analytic(counts: dict[str, torch.Tensor]) -> torch.Tensor:
    """E[bonus] treating the five bonus categories as independent Poissons.

    The distribution of "how many categories cleared 10" is Poisson-binomial; it is
    built here by the standard convolution recurrence over the five per-category
    P(≥10) probabilities. Independence understates the real positive correlation
    between components, so this is a known *under*-estimate of multi-category games —
    use the frailty Monte-Carlo in src/features/targets.py to calibrate the gap. It is
    used here because it is differentiable and cheap enough for a training loss.
    """
    probs = [poisson_tail_ge(counts[c]) for c in BONUS_CATEGORIES]

    # p_count[j] = P(exactly j categories cleared the threshold)
    p_count = [torch.ones_like(probs[0])] + [torch.zeros_like(probs[0])] * len(probs)
    for p in probs:
        for j in range(len(probs), 0, -1):
            p_count[j] = p_count[j] * (1.0 - p) + p_count[j - 1] * p
        p_count[0] = p_count[0] * (1.0 - p)

    return BONUS_TWO * p_count[2] + BONUS_THREE * sum(p_count[3:])


# ── Model ─────────────────────────────────────────────────────────────────────

class MultiHeadPlayerModel(nn.Module):
    """Sequence trunk + static path → availability, minutes and per-36 rate heads.

    Args:
        seq_input_size:    features per time step in the prior-season sequence.
        static_input_size: season-level features (team context, archetype, opponent).
        hidden_size:       LSTM hidden dimension.
        num_layers:        stacked LSTM layers.
        dropout:           applied between LSTM layers and in the heads.
        components:        component names, in dk_pts weight order.

    `forward` returns raw parameters (logits / log-rates); `expected_dk_pts` turns them
    into a points estimate. Keeping them separate lets the loss work in the natural
    parameterization of each head while inference reads a single number.
    """

    def __init__(self, seq_input_size: int, static_input_size: int = 0,
                 hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2,
                 components: list[str] | None = None):
        super().__init__()
        self.components = list(components or COMPONENTS)
        self.static_input_size = static_input_size

        self.lstm = nn.LSTM(
            input_size=seq_input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0.0,
        )
        fused = hidden_size + static_input_size
        self.trunk = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fused, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.play_head = nn.Linear(hidden_size, 1)        # logit P(play)
        self.minutes_head = nn.Linear(hidden_size, 1)     # log minutes | played
        self.rate_head = nn.Linear(hidden_size, len(self.components))  # log per-36 rates

    def forward(self, seq: torch.Tensor, static: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        _, (h_n, _) = self.lstm(seq)
        h = h_n[-1]
        if self.static_input_size:
            if static is None:
                raise ValueError("model was built with static features; `static` is required")
            h = torch.cat([h, static], dim=-1)
        h = self.trunk(h)

        log_rates = self.rate_head(h).clamp(-10.0, 5.0)
        return {
            "play_logit": self.play_head(h).squeeze(-1),
            "log_minutes": self.minutes_head(h).squeeze(-1).clamp(-5.0, 4.0),
            "log_rates": log_rates,
            "rates": log_rates.exp(),
        }

    # ── inference ────────────────────────────────────────────────────────────
    def expected_counts(self, out: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Expected component counts = per-36 rate × minutes exposure."""
        exposure = (out["log_minutes"].exp() / EXPOSURE_MINUTES).unsqueeze(-1)
        counts = out["rates"] * exposure
        return {c: counts[:, i] for i, c in enumerate(self.components)}

    def expected_dk_pts(self, out: dict[str, torch.Tensor], include_bonus: bool = True,
                        scale_by_availability: bool = True) -> torch.Tensor:
        counts = self.expected_counts(out)
        total = sum(counts[c] * DK_WEIGHTS[c] for c in self.components)
        if include_bonus:
            total = total + expected_bonus_analytic(counts)
        if scale_by_availability:
            total = total * torch.sigmoid(out["play_logit"])
        return total


# ── Loss ──────────────────────────────────────────────────────────────────────

class MultiHeadLoss(nn.Module):
    """Availability + minutes + Poisson rate losses, masked to games actually played.

    Rate and minutes heads are supervised only on played games: a DNP carries no rate
    information, and training a rate head toward zero there would corrupt it. The
    availability head is supervised on every scheduled game — that is the whole point
    of having it.

    `weights` rescales the three groups; component losses are averaged across
    components before weighting so the group does not dominate by having seven terms.
    """

    def __init__(self, components: list[str] | None = None,
                 weights: dict[str, float] | None = None):
        super().__init__()
        self.components = list(components or COMPONENTS)
        self.weights = {"play": 1.0, "minutes": 1.0, "components": 1.0} | (weights or {})
        self.play_loss = nn.BCEWithLogitsLoss()
        self.minutes_loss = nn.HuberLoss()
        self.count_loss = nn.PoissonNLLLoss(log_input=True, full=False, reduction="none")

    def forward(self, out: dict[str, torch.Tensor],
                target: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        played = target["played"].float()
        mask = played > 0
        parts: dict[str, torch.Tensor] = {
            "play": self.play_loss(out["play_logit"], played),
        }

        if mask.any():
            # Minutes in log space: errors are proportional, so a 5-minute miss on a
            # 30-minute night is not treated like one on a 6-minute night.
            parts["minutes"] = self.minutes_loss(
                out["log_minutes"][mask], target["minutes"][mask].clamp_min(1e-3).log())

            # Poisson NLL with a log-exposure offset — the canonical rate model.
            log_exposure = (target["minutes"][mask] / EXPOSURE_MINUTES).clamp_min(1e-3).log()
            log_mean = out["log_rates"][mask] + log_exposure.unsqueeze(-1)
            counts = torch.stack([target[c][mask] for c in self.components], dim=-1)
            parts["components"] = self.count_loss(log_mean, counts).mean()
        else:
            zero = out["log_minutes"].sum() * 0.0
            parts["minutes"], parts["components"] = zero, zero

        total = sum(self.weights[k] * v for k, v in parts.items())
        return total, {k: float(v.detach()) for k, v in parts.items()}
