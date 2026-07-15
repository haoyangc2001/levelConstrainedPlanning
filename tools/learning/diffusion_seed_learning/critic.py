#!/usr/bin/env python3
"""Success critic utilities for diffusion seed candidate filtering."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import Dataset

from dataset import DEFAULT_VALIDATED_SAMPLES, build_condition, resample_trajectory


CRITIC_OUTPUT_KEYS = [
    "success_logit",
    "alignment_risk",
    "collision_risk",
    "joint_jump_risk",
    "expected_opt_time_ms",
]


def load_candidate_samples(samples_path: Path = DEFAULT_VALIDATED_SAMPLES) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(samples_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample.get("sample_type") != "candidate":
                continue
            points = sample.get("candidate", {}).get("trajectory", {}).get("points")
            if points:
                rows.append(sample)
    if not rows:
        raise ValueError(f"No candidate samples found in {samples_path}")
    return rows


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        value = float(value)
        if not math.isfinite(value):
            return float(default)
        return value
    except Exception:
        return float(default)


def _source_type_one_hot(source_type: str | None) -> list[float]:
    labels = ["planner", "rule", "diffusion", "fallback", "unknown"]
    normalized = str(source_type or "unknown")
    if normalized not in labels:
        normalized = "unknown"
    return [1.0 if item == normalized else 0.0 for item in labels]


def build_critic_feature(sample: dict[str, Any], horizon: int) -> torch.Tensor:
    candidate = sample["candidate"]
    points = candidate["trajectory"]["points"]
    trajectory = resample_trajectory(points, horizon)
    metrics = candidate.get("metrics") or {}
    labels = sample.get("labels") or {}
    task = sample.get("task") or {}
    source_type = candidate.get("source_type") or sample.get("source_lineage", {}).get("source_type")
    steps = trajectory[1:] - trajectory[:-1]
    step_l2 = torch.linalg.norm(steps, dim=-1) if int(trajectory.shape[0]) > 1 else torch.zeros(1)
    summary = torch.tensor(
        [
            float(trajectory.shape[0]),
            float(trajectory.shape[1]),
            float(torch.linalg.norm(trajectory[-1] - trajectory[0]).item()),
            float(torch.sum(step_l2).item()),
            float(torch.max(step_l2).item()) if step_l2.numel() else 0.0,
            float(torch.max(torch.abs(trajectory)).item()),
            _safe_float(metrics.get("max_alignment_deviation_deg")),
            _safe_float(metrics.get("mean_alignment_deviation_deg")),
            _safe_float(metrics.get("position_error_m")),
            _safe_float(metrics.get("orientation_error_deg")),
            _safe_float(metrics.get("joint_step_max_l2")),
            _safe_float(metrics.get("joint_step_jump_cost")),
            _safe_float(metrics.get("twist_smoothness_cost")),
            1.0 if labels.get("validator_valid") else 0.0,
            _safe_float(task.get("alignment", {}).get("tolerance_deg"), 3.0),
        ]
        + _source_type_one_hot(source_type),
        dtype=torch.float32,
    )
    return torch.cat([build_condition(sample), trajectory.reshape(-1), summary], dim=0)


def build_critic_labels(sample: dict[str, Any]) -> dict[str, float]:
    labels = sample.get("labels") or {}
    candidate = sample.get("candidate") or {}
    metrics = candidate.get("metrics") or {}
    trajectory = candidate.get("trajectory") or {}
    points = trajectory.get("points") or []
    tolerance = _safe_float(sample.get("task", {}).get("alignment", {}).get("tolerance_deg"), 3.0)
    max_alignment = _safe_float(metrics.get("max_alignment_deviation_deg"))
    joint_step_max = _safe_float(metrics.get("joint_step_max_l2"))
    joint_jump_cost = _safe_float(metrics.get("joint_step_jump_cost"))
    success = 1.0 if labels.get("positive_for_critic") else 0.0
    # Phase 7 smoke dataset has no per-candidate optimize-time trace. Keep the
    # regression head wired with a transparent proxy until phase 8 benchmark
    # produces real timing labels.
    expected_opt_time_ms = 50.0 + 1.5 * float(len(points)) + 500.0 * joint_jump_cost
    if not success:
        expected_opt_time_ms += 25.0
    return {
        "success": success,
        "alignment_risk": min(max_alignment / max(tolerance, 1e-6), 4.0),
        "collision_risk": 0.0,
        "joint_jump_risk": min(joint_step_max / 1.5, 4.0),
        "expected_opt_time_ms": expected_opt_time_ms,
    }


@dataclass
class CriticNormalization:
    feature_mean: torch.Tensor
    feature_std: torch.Tensor
    regression_mean: torch.Tensor
    regression_std: torch.Tensor

    def normalize_feature(self, feature: torch.Tensor) -> torch.Tensor:
        return (feature - self.feature_mean) / self.feature_std.clamp_min(1e-6)

    def normalize_regression(self, target: torch.Tensor) -> torch.Tensor:
        return (target - self.regression_mean) / self.regression_std.clamp_min(1e-6)

    def denormalize_regression(self, value: torch.Tensor) -> torch.Tensor:
        return value * self.regression_std.clamp_min(1e-6) + self.regression_mean

    def to_json(self) -> dict[str, Any]:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_std": self.feature_std.tolist(),
            "regression_mean": self.regression_mean.tolist(),
            "regression_std": self.regression_std.tolist(),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "CriticNormalization":
        return cls(
            feature_mean=torch.tensor(data["feature_mean"], dtype=torch.float32),
            feature_std=torch.tensor(data["feature_std"], dtype=torch.float32),
            regression_mean=torch.tensor(data["regression_mean"], dtype=torch.float32),
            regression_std=torch.tensor(data["regression_std"], dtype=torch.float32),
        )


class SuccessCriticDataset(Dataset):
    def __init__(
        self,
        samples_path: Path = DEFAULT_VALIDATED_SAMPLES,
        horizon: int = 32,
        normalization: CriticNormalization | None = None,
    ) -> None:
        self.samples_path = Path(samples_path)
        self.horizon = int(horizon)
        self.samples = load_candidate_samples(self.samples_path)
        self.features = torch.stack(
            [build_critic_feature(sample, self.horizon) for sample in self.samples],
            dim=0,
        )
        labels = [build_critic_labels(sample) for sample in self.samples]
        self.success = torch.tensor([item["success"] for item in labels], dtype=torch.float32)
        self.regression = torch.tensor(
            [
                [
                    item["alignment_risk"],
                    item["collision_risk"],
                    item["joint_jump_risk"],
                    item["expected_opt_time_ms"],
                ]
                for item in labels
            ],
            dtype=torch.float32,
        )
        self.normalization = normalization or CriticNormalization(
            feature_mean=self.features.mean(dim=0),
            feature_std=self.features.std(dim=0).clamp_min(1e-3),
            regression_mean=self.regression.mean(dim=0),
            regression_std=self.regression.std(dim=0).clamp_min(1e-3),
        )

    @property
    def input_dim(self) -> int:
        return int(self.features.shape[-1])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "feature": self.normalization.normalize_feature(self.features[index]),
            "success": self.success[index],
            "regression": self.normalization.normalize_regression(self.regression[index]),
            "raw_regression": self.regression[index],
        }


class SuccessCriticMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )
        self.success_head = nn.Linear(hidden_dim, 1)
        self.regression_head = nn.Linear(hidden_dim, 4)

    def forward(self, feature: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.net(feature)
        return {
            "success_logit": self.success_head(hidden).squeeze(-1),
            "regression": self.regression_head(hidden),
        }


def score_candidates(
    model: SuccessCriticMLP,
    dataset: SuccessCriticDataset,
    device: torch.device,
) -> list[dict[str, Any]]:
    model.eval()
    features = dataset.normalization.normalize_feature(dataset.features).to(device)
    with torch.no_grad():
        output = model(features)
        p_success = torch.sigmoid(output["success_logit"]).detach().cpu()
        regression = dataset.normalization.denormalize_regression(
            output["regression"].detach().cpu()
        )
    rows = []
    for index, sample in enumerate(dataset.samples):
        alignment_risk, collision_risk, joint_jump_risk, expected_opt_time_ms = regression[index].tolist()
        quality_score = (
            float(p_success[index].item())
            - 0.20 * float(alignment_risk)
            - 0.10 * float(collision_risk)
            - 0.15 * float(joint_jump_risk)
            - 0.0005 * float(max(expected_opt_time_ms, 0.0))
        )
        rows.append(
            {
                "index": int(index),
                "sample_id": sample.get("sample_id"),
                "request_key": sample.get("source", {}).get("move_file")
                or f"request_{sample.get('source', {}).get('plan_request_index')}",
                "source_label": sample.get("candidate", {}).get("source_label"),
                "positive_for_critic": bool(sample.get("labels", {}).get("positive_for_critic")),
                "p_success": float(p_success[index].item()),
                "alignment_risk": float(alignment_risk),
                "collision_risk": float(collision_risk),
                "joint_jump_risk": float(joint_jump_risk),
                "expected_opt_time_ms": float(expected_opt_time_ms),
                "quality_score": quality_score,
            }
        )
    return rows


def trajectory_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    horizon = max(int(a.shape[0]), int(b.shape[0]))
    if int(a.shape[0]) < horizon:
        a = torch.cat([a, a[-1:].repeat(horizon - int(a.shape[0]), 1)], dim=0)
    if int(b.shape[0]) < horizon:
        b = torch.cat([b, b[-1:].repeat(horizon - int(b.shape[0]), 1)], dim=0)
    return float(torch.mean(torch.linalg.norm(a - b, dim=-1)).item())


def select_diverse_topk(
    candidate_indices: list[int],
    score_rows: list[dict[str, Any]],
    dataset: SuccessCriticDataset,
    k: int,
    diversity_weight: float = 0.15,
) -> list[int]:
    score_by_index = {int(row["index"]): float(row["quality_score"]) for row in score_rows}
    remaining = list(candidate_indices)
    selected: list[int] = []
    trajectories = [
        resample_trajectory(sample["candidate"]["trajectory"]["points"], dataset.horizon)
        for sample in dataset.samples
    ]
    while remaining and len(selected) < int(k):
        best_index = None
        best_score = None
        for idx in remaining:
            base_score = score_by_index.get(int(idx), float("-inf"))
            if selected:
                min_dist = min(trajectory_distance(trajectories[idx], trajectories[j]) for j in selected)
            else:
                min_dist = 0.0
            score = base_score + float(diversity_weight) * min_dist
            if best_score is None or score > best_score:
                best_score = score
                best_index = int(idx)
        if best_index is None:
            break
        selected.append(best_index)
        remaining.remove(best_index)
    return selected
