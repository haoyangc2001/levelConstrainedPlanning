#!/usr/bin/env python3
"""Dataset utilities for offline diffusion seed learning."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


DEFAULT_VALIDATED_SAMPLES = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/"
    "datasets/sr5_phase2_20260713_lifecycle_baseline/samples_validated.jsonl"
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def resample_trajectory(points: list[list[float]], horizon: int) -> torch.Tensor:
    source = torch.tensor(points, dtype=torch.float32)
    if source.ndim != 2:
        raise ValueError(f"trajectory must be [T, DOF], got shape={list(source.shape)}")
    if source.shape[0] == horizon:
        return source
    src_x = torch.linspace(0.0, 1.0, source.shape[0])
    dst_x = torch.linspace(0.0, 1.0, horizon)
    columns = []
    for joint_index in range(source.shape[1]):
        y = source[:, joint_index]
        indices = torch.searchsorted(src_x, dst_x, right=False).clamp(0, source.shape[0] - 1)
        left = (indices - 1).clamp(0, source.shape[0] - 1)
        right = indices
        x0 = src_x[left]
        x1 = src_x[right]
        y0 = y[left]
        y1 = y[right]
        denom = (x1 - x0).clamp_min(1e-8)
        alpha = (dst_x - x0) / denom
        columns.append(y0 + alpha * (y1 - y0))
    return torch.stack(columns, dim=-1)


def build_condition(sample: dict[str, Any]) -> torch.Tensor:
    start = sample.get("start_state", {}).get("service_start_joint") or [0.0] * 6
    task = sample.get("task", {})
    target_pose = task.get("target_pose") or [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    tolerance = task.get("alignment", {}).get("tolerance_deg") or 3.0
    obstacle_count = sample.get("obstacle_world", {}).get("total_box_count") or 0
    return torch.tensor(
        [float(v) for v in start]
        + [float(v) for v in target_pose]
        + [float(tolerance), float(obstacle_count)],
        dtype=torch.float32,
    )


@dataclass
class Normalization:
    mean: torch.Tensor
    std: torch.Tensor

    def normalize(self, value: torch.Tensor) -> torch.Tensor:
        return (value - self.mean) / self.std.clamp_min(1e-6)

    def denormalize(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.std.clamp_min(1e-6) + self.mean

    def to_json(self) -> dict[str, Any]:
        return {
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "Normalization":
        return cls(
            mean=torch.tensor(data["mean"], dtype=torch.float32),
            std=torch.tensor(data["std"], dtype=torch.float32),
        )


class TrajectorySeedDataset(Dataset):
    def __init__(
        self,
        samples_path: Path = DEFAULT_VALIDATED_SAMPLES,
        horizon: int = 32,
        positive_only: bool = True,
    ) -> None:
        self.samples_path = Path(samples_path)
        self.horizon = int(horizon)
        raw_samples = load_jsonl(self.samples_path)
        self.samples = [
            sample for sample in raw_samples
            if sample.get("sample_type") == "candidate"
            and (not positive_only or bool(sample.get("labels", {}).get("positive_for_diffusion")))
            and sample.get("candidate")
            and sample.get("candidate", {}).get("trajectory", {}).get("points")
        ]
        if not self.samples:
            raise ValueError(f"No usable trajectory samples found in {self.samples_path}")
        trajectories = [
            resample_trajectory(sample["candidate"]["trajectory"]["points"], self.horizon)
            for sample in self.samples
        ]
        conditions = [build_condition(sample) for sample in self.samples]
        self.trajectories = torch.stack(trajectories, dim=0)
        self.conditions = torch.stack(conditions, dim=0)
        self.normalization = Normalization(
            mean=self.trajectories.mean(dim=(0, 1)),
            std=self.trajectories.std(dim=(0, 1)).clamp_min(1e-3),
        )

    @property
    def dof(self) -> int:
        return int(self.trajectories.shape[-1])

    @property
    def condition_dim(self) -> int:
        return int(self.conditions.shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        trajectory = self.trajectories[index]
        return {
            "trajectory": self.normalization.normalize(trajectory),
            "condition": self.conditions[index],
            "raw_trajectory": trajectory,
        }
