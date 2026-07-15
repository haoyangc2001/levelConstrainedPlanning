#!/usr/bin/env python3
"""Small temporal 1D U-Net style denoiser."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        scale = math.log(10000) / max(half - 1, 1)
        freqs = torch.exp(torch.arange(half, device=timesteps.device) * -scale)
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = torch.nn.functional.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


class TemporalUNet1D(nn.Module):
    """A compact temporal Conv1D denoiser for [B, T, DOF] trajectories."""

    def __init__(self, dof: int, condition_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.dof = int(dof)
        self.condition_dim = int(condition_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        self.condition_embedding = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.in_proj = nn.Conv1d(dof, hidden_dim, kernel_size=3, padding=1)
        self.down = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.mid = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )
        self.out = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, dof, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)
        emb = self.time_embedding(timesteps) + self.condition_embedding(condition)
        emb = emb.unsqueeze(-1)
        h0 = self.in_proj(h)
        h1 = self.down(h0 + emb)
        h2 = self.mid(h1 + emb)
        return self.out(h2 + h1).transpose(1, 2)
