#!/usr/bin/env python3
"""Minimal DDPM utilities for trajectory seed diffusion.

C3b (anti-over-claim): ``training_loss`` supports two *switchable* auxiliary
loss components on top of the base noise-prediction MSE, so the Method/ablation
may legitimately claim an "alignment loss" and a "collision guidance" term:

* ``L_level``   — differentiable alignment-deviation penalty. From the model's
  eps-prediction we reconstruct x0_hat, denormalise it to joint space, run the
  robot FK (autograd-differentiable cuRobo kinematics), measure the axis
  alignment angle, and penalise ``relu(angle_deg - tolerance_deg) ** 2`` for the
  samples whose condition marks ``level_active == 1``.
* ``L_collision`` — collision-cost guidance (depends on A1 collision replay).
  Penalises positive world-collision cost along the reconstructed trajectory.

Both are OFF by default: with ``aux=None`` (or all weights zero) ``training_loss``
is bit-for-bit the original pure-MSE objective, so legacy checkpoints and the
C5 "ablate-to-off" variants stay faithful. C5 flips one flag per variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn.functional as F


@dataclass
class AuxLossConfig:
    """Configuration + injected callbacks for the switchable C3b loss terms.

    The diffusion module stays free of any planner/curobo import: ``train.py``
    injects the differentiable FK/collision callbacks and the denormaliser. When
    ``enable_l_level``/``enable_l_collision`` are both False this whole object is
    inert and ``training_loss`` collapses to the base MSE.
    """

    enable_l_level: bool = False
    enable_l_collision: bool = False
    level_weight: float = 0.0
    collision_weight: float = 0.0
    tolerance_deg: float = 3.0
    # index of ``level_active`` inside the condition vector (17-dim C1b layout);
    # None => treat every sample as level-constrained.
    level_active_index: int | None = 15
    # denormalise a normalised trajectory tensor [B,T,DOF] -> joint radians.
    denormalize: Callable[[torch.Tensor], torch.Tensor] | None = None
    # FK alignment angle: (positions [N,DOF]) -> angle_deg [N] (differentiable).
    alignment_angle_fn: Callable[[torch.Tensor], torch.Tensor] | None = None
    # collision cost: (positions [N,DOF]) -> per-config world cost [N] (>=0).
    collision_cost_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    @property
    def active(self) -> bool:
        return bool(
            (self.enable_l_level and self.level_weight > 0.0)
            or (self.enable_l_collision and self.collision_weight > 0.0)
        )


class GaussianDiffusion1D:
    def __init__(self, steps: int = 64, beta_start: float = 1e-4, beta_end: float = 2e-2) -> None:
        self.steps = int(steps)
        betas = torch.linspace(beta_start, beta_end, self.steps)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

    def to(self, device: torch.device) -> "GaussianDiffusion1D":
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return alpha_bar.sqrt() * x0 + (1.0 - alpha_bar).sqrt() * noise

    def _predict_x0(
        self, xt: torch.Tensor, timesteps: torch.Tensor, predicted_noise: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct x0_hat from xt and the model's eps-prediction."""
        alpha_bar = self.alpha_bars[timesteps].view(-1, 1, 1)
        return (xt - (1.0 - alpha_bar).sqrt() * predicted_noise) / alpha_bar.sqrt().clamp_min(1e-8)

    def training_loss(
        self,
        model,
        x0: torch.Tensor,
        condition: torch.Tensor,
        aux: "AuxLossConfig | None" = None,
        *,
        return_components: bool = False,
    ):
        timesteps = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        predicted = model(xt, timesteps, condition)
        base = F.mse_loss(predicted, noise)

        components: dict[str, float] = {"base_mse": float(base.detach().item())}
        total = base
        if aux is not None and aux.active:
            x0_hat = self._predict_x0(xt, timesteps, predicted)
            # SNR (alpha_bar) weight per sample: x0_hat = (xt - sqrt(1-ab)*eps)/sqrt(ab)
            # so d(x0_hat)/d(eps) ~ 1/sqrt(ab) blows up as ab->0 (high-noise t). Left
            # unweighted, the aux gradient dominates clip_grad_norm and destroys the
            # base denoising signal (observed: base_mse stuck ~3e5, never learns).
            # Weighting the aux term by alpha_bar_t both bounds the effective gradient
            # (sqrt(ab)*1/sqrt(ab)=O(1)) and only penalises alignment where x0_hat is
            # actually signal (low noise), not where it is essentially noise.
            snr_weight = self.alpha_bars[timesteps]  # [B], in (0,1]
            aux_loss, aux_parts = self._aux_losses(x0_hat, condition, aux, snr_weight)
            total = total + aux_loss
            components.update(aux_parts)
        components["total"] = float(total.detach().item())
        if return_components:
            return total, components
        return total

    def _aux_losses(
        self,
        x0_hat: torch.Tensor,
        condition: torch.Tensor,
        aux: "AuxLossConfig",
        snr_weight: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the switchable L_level / L_collision terms on reconstructed x0.

        ``x0_hat`` is in *normalised* trajectory space; we denormalise to joint
        radians before FK/collision. ``snr_weight`` is the per-sample alpha_bar_t
        (in (0,1]); when provided the aux terms are weighted by it so high-noise
        timesteps (where x0_hat is essentially noise and the 1/sqrt(alpha_bar)
        reconstruction gradient explodes) contribute little. Returns
        (weighted_sum, per-term floats)."""
        device = x0_hat.device
        parts: dict[str, float] = {}
        aux_total = torch.zeros((), device=device)

        traj = x0_hat
        if aux.denormalize is not None:
            traj = aux.denormalize(x0_hat)
        batch, horizon, dof = traj.shape
        flat = traj.reshape(batch * horizon, dof)
        if snr_weight is None:
            snr_weight = torch.ones(batch, device=device)

        if aux.enable_l_level and aux.level_weight > 0.0 and aux.alignment_angle_fn is not None:
            angle_deg = aux.alignment_angle_fn(flat).reshape(batch, horizon)
            violation = torch.relu(angle_deg - float(aux.tolerance_deg))
            per_sample = (violation ** 2).mean(dim=1)  # [B]
            if aux.level_active_index is not None and condition.shape[-1] > aux.level_active_index:
                mask = (condition[:, aux.level_active_index] > 0.5).to(per_sample.dtype)
            else:
                mask = torch.ones_like(per_sample)
            # SNR-weight the per-sample penalty (see training_loss): bounds the
            # aux gradient and focuses it on low-noise reconstructions.
            weight = mask * snr_weight
            denom = weight.sum().clamp_min(1.0)
            l_level = (per_sample * weight).sum() / denom
            aux_total = aux_total + float(aux.level_weight) * l_level
            parts["l_level"] = float(l_level.detach().item())

        if (
            aux.enable_l_collision
            and aux.collision_weight > 0.0
            and aux.collision_cost_fn is not None
        ):
            cost = aux.collision_cost_fn(flat).reshape(batch, horizon)
            per_sample_c = torch.relu(cost).mean(dim=1)  # [B]
            cdenom = snr_weight.sum().clamp_min(1.0)
            l_collision = (per_sample_c * snr_weight).sum() / cdenom
            aux_total = aux_total + float(aux.collision_weight) * l_collision
            parts["l_collision"] = float(l_collision.detach().item())

        return aux_total, parts

    @torch.no_grad()
    def sample(self, model, shape: tuple[int, int, int], condition: torch.Tensor) -> torch.Tensor:
        device = condition.device
        x = torch.randn(shape, device=device)
        for step in reversed(range(self.steps)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            pred_noise = model(x, t, condition)
            mean = (x - beta / (1.0 - alpha_bar).sqrt() * pred_noise) / alpha.sqrt()
            if step > 0:
                x = mean + beta.sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x
