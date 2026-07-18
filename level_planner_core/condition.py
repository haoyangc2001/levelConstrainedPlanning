"""Shared condition vector builder for runtime and learning tools."""

from __future__ import annotations

from typing import Any

import torch


CONDITION_DIM = 15
# C1b: two extra constraint-class axes (level_active, goal_orientation_active)
# appended only when the caller/checkpoint asks for the extended layout. Legacy
# 15-dim checkpoints stay runnable; C5 retrains with the 17-dim layout.
CONSTRAINT_CLASS_DIM = 2
CONDITION_DIM_WITH_CLASS = CONDITION_DIM + CONSTRAINT_CLASS_DIM


def build_condition_values(
    payload: dict[str, Any],
    *,
    condition_dim: int = CONDITION_DIM,
) -> list[float]:
    """Build the shared SR5 condition vector.

    Layout (base, 15-dim):
    - 6 start joint values
    - 7 target pose values [x, y, z, qw, qx, qy, qz]
    - alignment tolerance in degrees
    - obstacle box count

    C1b extended layout (``condition_dim == CONDITION_DIM_WITH_CLASS``, 17-dim)
    appends the two constraint-class axes:
    - level_active (1.0 if the level/alignment constraint is enforced)
    - goal_orientation_active (1.0 if goal orientation is enforced)

    ``condition_dim`` is normally driven by the loaded checkpoint's
    ``model_config['condition_dim']`` so runtime and training stay in lockstep.
    """
    start = _extract_start_joint(payload)
    target_pose = _extract_target_pose(payload)
    alignment = _extract_alignment(payload)
    obstacle_count = _extract_obstacle_count(payload)
    values = [float(v) for v in start[:6]]
    values += [float(v) for v in target_pose[:7]]
    values += [float(alignment.get("tolerance_deg", 3.0)), float(obstacle_count)]
    if int(condition_dim) == CONDITION_DIM_WITH_CLASS:
        level_active, goal_orientation_active = _extract_class_axes(payload)
        values += [float(level_active), float(goal_orientation_active)]
    if len(values) != int(condition_dim):
        raise ValueError(
            f"condition vector dimension mismatch: got {len(values)}, expected {int(condition_dim)}"
        )
    return values


def build_condition_tensor(
    payload: dict[str, Any],
    *,
    condition_dim: int = CONDITION_DIM,
) -> torch.Tensor:
    return torch.tensor(
        build_condition_values(payload, condition_dim=condition_dim), dtype=torch.float32
    )


def _extract_class_axes(payload: dict[str, Any]) -> tuple[float, float]:
    """Return (level_active, goal_orientation_active) from a request payload.

    Prefers the canonical ``constraint_axes`` block (written by the sampler /
    planner normalization); falls back to decoding ``constraint_class``; then to
    ``alignment.strict_level`` for legacy payloads. Defaults to (1, 1) — the
    fully-constrained LPO class — when nothing is present.
    """
    axes = payload.get("constraint_axes")
    if isinstance(axes, dict) and axes:
        return (
            1.0 if axes.get("level_active", True) else 0.0,
            1.0 if axes.get("goal_orientation_active", True) else 0.0,
        )
    class_id = payload.get("constraint_class")
    if class_id:
        try:
            from level_planner_core.constraint_class import get_spec

            spec = get_spec(class_id)
            return (
                1.0 if spec.level_active else 0.0,
                1.0 if spec.goal_orientation_active else 0.0,
            )
        except Exception:
            pass
    alignment = _extract_alignment(payload)
    level_active = 1.0 if alignment.get("strict_level", True) else 0.0
    return (level_active, 1.0)


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
