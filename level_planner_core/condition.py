"""Shared condition vector builder for runtime and learning tools."""

from __future__ import annotations

from typing import Any

import torch


CONDITION_DIM = 15


def build_condition_values(payload: dict[str, Any]) -> list[float]:
    """Build the shared SR5 condition vector.

    Layout:
    - 6 start joint values
    - 7 target pose values [x, y, z, qw, qx, qy, qz]
    - alignment tolerance in degrees
    - obstacle box count
    """
    start = _extract_start_joint(payload)
    target_pose = _extract_target_pose(payload)
    alignment = _extract_alignment(payload)
    obstacle_count = _extract_obstacle_count(payload)
    values = [float(v) for v in start[:6]]
    values += [float(v) for v in target_pose[:7]]
    values += [float(alignment.get("tolerance_deg", 3.0)), float(obstacle_count)]
    if len(values) != CONDITION_DIM:
        raise ValueError(f"condition vector dimension mismatch: got {len(values)}, expected {CONDITION_DIM}")
    return values


def build_condition_tensor(payload: dict[str, Any]) -> torch.Tensor:
    return torch.tensor(build_condition_values(payload), dtype=torch.float32)


def _extract_start_joint(payload: dict[str, Any]) -> list[float]:
    start = (
        payload.get("start_joint")
        or payload.get("start_state", {}).get("service_start_joint")
        or payload.get("normalized_request", {}).get("start_joint")
        or [0.0] * 6
    )
    values = [float(v) for v in start]
    if len(values) < 6:
        values.extend([0.0] * (6 - len(values)))
    return values


def _extract_target_pose(payload: dict[str, Any]) -> list[float]:
    if payload.get("target_pose") and isinstance(payload["target_pose"], list):
        target_pose = payload["target_pose"]
    else:
        task = payload.get("task") or {}
        target_pose = task.get("target_pose")
    if not target_pose:
        normalized = payload.get("normalized_request") or {}
        target_pose = normalized.get("target_pose")
    if not target_pose:
        target_pose = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    values = [float(v) for v in target_pose]
    if len(values) < 7:
        values.extend([0.0] * (7 - len(values)))
        values[3] = 1.0
    return values


def _extract_alignment(payload: dict[str, Any]) -> dict[str, Any]:
    return (
        payload.get("alignment")
        or payload.get("task", {}).get("alignment")
        or payload.get("normalized_request", {}).get("alignment")
        or {}
    )


def _extract_obstacle_count(payload: dict[str, Any]) -> float:
    world = (
        payload.get("obstacle_world")
        or payload.get("world_summary")
        or payload.get("metrics", {}).get("world")
        or {}
    )
    return float(world.get("total_box_count") or world.get("box_count") or 0.0)
