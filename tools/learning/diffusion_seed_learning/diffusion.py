#!/usr/bin/env python3
"""Minimal DDPM utilities for trajectory seed diffusion."""

from __future__ import annotations

import torch
import torch.nn.functional as F


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

    def training_loss(self, model, x0: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        timesteps = torch.randint(0, self.steps, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, timesteps, noise)
        predicted = model(xt, timesteps, condition)
        return F.mse_loss(predicted, noise)

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
