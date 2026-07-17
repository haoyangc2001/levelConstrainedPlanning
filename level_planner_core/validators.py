"""Unified hard-validation helpers for candidate trajectories."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import torch
import yaml


DEFAULT_THRESHOLDS = {
    "goal_position_tolerance_m": 0.02,
    "goal_orientation_tolerance_rad": 0.20,
    "max_start_gap_l2": 0.25,
    "max_joint_step_l2": 2.0,
    "max_joint_step_abs": 1.5,
    "max_acceleration_proxy_l2": 3.0,
    # A1.4: collision safety margin (meters). A trajectory is collision-valid iff the
    # minimum signed distance to world obstacles along the path is >= this margin.
    # cuRobo references 0-2.5cm activation bands; 0.005m is a conservative default.
    "collision_safety_margin_m": 0.005,
}


def load_joint_limits_from_robot_config(
    robot_config_path: Path,
    joint_names: list[str],
) -> list[dict[str, Any]]:
    """Load ordered joint limits from the robot config's URDF."""
    config_path = Path(robot_config_path)
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    kinematics = (payload.get("robot_cfg") or {}).get("kinematics") or {}
    urdf_raw = kinematics.get("urdf_path")
    if not urdf_raw:
        return [{"joint_name": name, "lower": None, "upper": None, "velocity": None} for name in joint_names]
    urdf_path = Path(str(urdf_raw))
    if not urdf_path.is_absolute():
        urdf_path = (config_path.parent / urdf_path).resolve()
    if not urdf_path.exists():
        return [{"joint_name": name, "lower": None, "upper": None, "velocity": None} for name in joint_names]

    root = ET.parse(urdf_path).getroot()
    by_name: dict[str, dict[str, Any]] = {}
    for joint in root.findall("joint"):
        name = joint.attrib.get("name")
        limit = joint.find("limit")
        if not name or limit is None:
            continue
        by_name[name] = {
            "joint_name": name,
            "lower": _optional_float(limit.attrib.get("lower")),
            "upper": _optional_float(limit.attrib.get("upper")),
            "velocity": _optional_float(limit.attrib.get("velocity")),
            "effort": _optional_float(limit.attrib.get("effort")),
        }
    return [
        by_name.get(name, {"joint_name": name, "lower": None, "upper": None, "velocity": None})
        for name in joint_names
    ]


def evaluate_hard_constraints(
    *,
    trajectory_points: list[list[float]],
    start_joint: list[float],
    joint_limits: list[dict[str, Any]],
    metrics: dict[str, Any],
    alignment_tolerance_deg: float,
    optimizer_success: bool,
    world_summary: dict[str, Any] | None = None,
    thresholds: dict[str, float] | None = None,
    collision_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate all currently available hard validation checks.

    ``collision_result`` is injected by the caller (the planner, which holds the
    CuRobo world) so this module stays free of any curobo import. It is the dict
    returned by ``planner._evaluate_collision(...)`` and carries the along-path
    minimum signed distance plus a raw collision cost. When ``None`` the check
    degrades to ``unchecked`` (kept valid) exactly as before A1.
    """
    active = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alignment = _evaluate_alignment(metrics, alignment_tolerance_deg)
    goal = _evaluate_goal(metrics, active)
    continuity = _evaluate_continuity(trajectory_points, start_joint, active)
    joint_limit = evaluate_joint_limits(trajectory_points, joint_limits)
    velocity_acceleration = evaluate_velocity_acceleration_proxy(trajectory_points, active)
    collision_safety = _evaluate_collision(collision_result, active, world_summary)
    optimizer = {
        "success": bool(optimizer_success),
        "status": "success" if optimizer_success else "failed",
        "valid": bool(optimizer_success),
        "failure_reason": None if optimizer_success else "repair_failed",
    }
    checks = {
        "optimizer": optimizer,
        "alignment": alignment,
        "goal": goal,
        "continuity": continuity,
        "joint_limit": joint_limit,
        "collision_safety": collision_safety,
        "velocity_acceleration": velocity_acceleration,
    }
    failure_reasons = [
        item.get("failure_reason")
        for item in checks.values()
        if not item.get("valid") and item.get("failure_reason")
    ]
    valid = not failure_reasons
    return {
        "status": "valid" if valid else "failed",
        "valid": valid,
        "failure_reason": None if valid else failure_reasons[0],
        "failure_reasons": failure_reasons,
        "checks": checks,
    }


def _evaluate_collision(
    collision_result: dict[str, Any] | None,
    thresholds: dict[str, float],
    world_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    """Interpret a planner-computed collision result against the safety margin.

    Pure interpretation only — the along-path distance/cost is measured by the
    planner (A1.2) and handed in via ``collision_result`` so validators never
    imports curobo (A1.3).

    Expected ``collision_result`` keys:
      - ``min_distance_m``: minimum signed distance to obstacles over the path
        (large positive when the world is empty; may be ``None`` if unavailable).
      - ``collision_cost``: raw cuRobo world collision cost (0 == safe/outside
        activation band, positive == within-margin/penetrating, meters).
      - ``num_points`` / ``num_spheres``: coverage metadata (optional).
    """
    margin = float(thresholds.get("collision_safety_margin_m", 0.005))
    world = dict(world_summary or {})

    if collision_result is None:
        return {
            "checked": False,
            "valid": True,
            "status": "unchecked",
            "min_distance_m": None,
            "collision_cost_proxy": None,
            "safety_margin_m": margin,
            "reason_if_unchecked": "collision_result not provided by planner.",
            "world_summary": world,
        }

    # Degenerate obstacle-free world: nothing to hit -> valid with large distance.
    total_obstacles = int(world.get("total_count", 0) or 0)
    if total_obstacles == 0:
        return {
            "checked": True,
            "valid": True,
            "status": "no_obstacles",
            "min_distance_m": float("inf"),
            "collision_cost_proxy": 0.0,
            "safety_margin_m": margin,
            "world_summary": world,
        }

    cost = collision_result.get("collision_cost")
    min_distance = collision_result.get("min_distance_m")
    cost_f = float(cost) if cost is not None else None
    dist_f = float(min_distance) if min_distance is not None else None

    # Primary gate: cuRobo world collision cost measured at activation = margin.
    # cost == 0 means every sphere is at least `margin` away from all obstacles.
    if cost_f is not None:
        in_collision = cost_f > 0.0
    elif dist_f is not None:
        in_collision = dist_f < margin
    else:
        # Result object present but empty -> treat as unchecked rather than pass.
        return {
            "checked": False,
            "valid": True,
            "status": "unchecked",
            "min_distance_m": None,
            "collision_cost_proxy": None,
            "safety_margin_m": margin,
            "reason_if_unchecked": "collision_result carried no cost or distance.",
            "world_summary": world,
        }

    valid = not in_collision
    return {
        "checked": True,
        "valid": valid,
        "status": "safe" if valid else "collision",
        "failure_reason": None if valid else "collision_detected",
        "min_distance_m": dist_f,
        "collision_cost_proxy": cost_f,
        "safety_margin_m": margin,
        "num_points": collision_result.get("num_points"),
        "num_spheres": collision_result.get("num_spheres"),
        "world_summary": world,
    }


def evaluate_joint_limits(
    trajectory_points: list[list[float]],
    joint_limits: list[dict[str, Any]],
    *,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    if not trajectory_points:
        return {
            "valid": False,
            "status": "missing_trajectory",
            "failure_reason": "required_validator_metric_missing",
            "min_margin_rad": None,
            "violating_joint_names": [],
            "max_abs_position_rad": None,
        }
    if not joint_limits:
        return {
            "valid": False,
            "status": "missing_joint_limits",
            "failure_reason": "required_validator_metric_missing",
            "min_margin_rad": None,
            "violating_joint_names": [],
            "max_abs_position_rad": None,
        }
    points = torch.tensor(trajectory_points, dtype=torch.float32)
    violations: list[dict[str, Any]] = []
    margins: list[float] = []
    for joint_index, limit in enumerate(joint_limits):
        if joint_index >= int(points.shape[1]):
            continue
        lower = limit.get("lower")
        upper = limit.get("upper")
        values = points[:, joint_index]
        joint_name = str(limit.get("joint_name") or f"joint_{joint_index}")
        if lower is not None:
            lower_margin = values - float(lower)
            margins.append(float(torch.min(lower_margin).item()))
            if bool(torch.any(lower_margin < -epsilon).item()):
                violations.append({"joint_name": joint_name, "side": "lower", "limit": float(lower)})
        if upper is not None:
            upper_margin = float(upper) - values
            margins.append(float(torch.min(upper_margin).item()))
            if bool(torch.any(upper_margin < -epsilon).item()):
                violations.append({"joint_name": joint_name, "side": "upper", "limit": float(upper)})
    valid = len(violations) == 0
    return {
        "valid": valid,
        "status": "valid" if valid else "failed",
        "failure_reason": None if valid else "failed_joint_limit",
        "min_margin_rad": round(min(margins), 8) if margins else None,
        "violating_joint_names": sorted({item["joint_name"] for item in violations}),
        "violations": violations,
        "max_abs_position_rad": round(float(torch.max(torch.abs(points)).item()), 8),
    }


def evaluate_velocity_acceleration_proxy(
    trajectory_points: list[list[float]],
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    active = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    if len(trajectory_points) < 2:
        return {
            "valid": True,
            "status": "not_enough_points",
            "max_velocity_proxy_l2": 0.0,
            "max_acceleration_proxy_l2": 0.0,
            "failure_reason": None,
        }
    points = torch.tensor(trajectory_points, dtype=torch.float32)
    velocity = points[1:] - points[:-1]
    velocity_l2 = torch.linalg.norm(velocity, dim=-1)
    if int(velocity.shape[0]) >= 2:
        acceleration = velocity[1:] - velocity[:-1]
        acceleration_l2 = torch.linalg.norm(acceleration, dim=-1)
        max_acceleration = float(torch.max(acceleration_l2).item())
    else:
        max_acceleration = 0.0
    max_velocity = float(torch.max(velocity_l2).item())
    valid = max_velocity <= float(active["max_joint_step_l2"]) and max_acceleration <= float(
        active["max_acceleration_proxy_l2"]
    )
    failure_reason = None
    if max_velocity > float(active["max_joint_step_l2"]):
        failure_reason = "failed_velocity_proxy"
    elif max_acceleration > float(active["max_acceleration_proxy_l2"]):
        failure_reason = "failed_acceleration_proxy"
    return {
        "valid": bool(valid),
        "status": "valid" if valid else "failed",
        "failure_reason": failure_reason,
        "max_velocity_proxy_l2": round(max_velocity, 8),
        "max_acceleration_proxy_l2": round(max_acceleration, 8),
        "velocity_proxy_threshold_l2": float(active["max_joint_step_l2"]),
        "acceleration_proxy_threshold_l2": float(active["max_acceleration_proxy_l2"]),
        "dt_sec": None,
    }


def _evaluate_alignment(metrics: dict[str, Any], tolerance_deg: float) -> dict[str, Any]:
    max_dev = _optional_float(metrics.get("max_alignment_deviation_deg"))
    mean_dev = _optional_float(metrics.get("mean_alignment_deviation_deg"))
    valid = max_dev is not None and max_dev <= float(tolerance_deg)
    return {
        "valid": bool(valid),
        "status": "valid" if valid else "failed",
        "failure_reason": None if valid else "failed_alignment_constraint",
        "max_alignment_deviation_deg": max_dev,
        "mean_alignment_deviation_deg": mean_dev,
        "alignment_tolerance_deg": float(tolerance_deg),
    }


def _evaluate_goal(metrics: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    position_error = _optional_float(metrics.get("position_error_m"))
    orientation_error = _optional_float(metrics.get("orientation_error_rad"))
    if orientation_error is None and metrics.get("orientation_error_deg") is not None:
        orientation_error = math.radians(float(metrics["orientation_error_deg"]))
    valid = (
        position_error is not None
        and orientation_error is not None
        and position_error <= float(thresholds["goal_position_tolerance_m"])
        and orientation_error <= float(thresholds["goal_orientation_tolerance_rad"])
    )
    failure_reason = None
    if not valid:
        failure_reason = "failed_goal" if position_error is not None or orientation_error is not None else "required_validator_metric_missing"
    return {
        "valid": bool(valid),
        "status": "valid" if valid else "failed",
        "failure_reason": failure_reason,
        "terminal_position_error_m": position_error,
        "terminal_orientation_error_rad": orientation_error,
        "position_tolerance_m": float(thresholds["goal_position_tolerance_m"]),
        "orientation_tolerance_rad": float(thresholds["goal_orientation_tolerance_rad"]),
    }


def _evaluate_continuity(
    trajectory_points: list[list[float]],
    start_joint: list[float],
    thresholds: dict[str, float],
) -> dict[str, Any]:
    if not trajectory_points:
        return {
            "valid": False,
            "status": "missing_trajectory",
            "failure_reason": "required_validator_metric_missing",
        }
    points = torch.tensor(trajectory_points, dtype=torch.float32)
    start = torch.tensor(start_joint, dtype=torch.float32)
    start_gap = float(torch.linalg.norm(points[0] - start).item()) if len(start_joint) == int(points.shape[1]) else math.inf
    if int(points.shape[0]) > 1:
        step = points[1:] - points[:-1]
        step_l2 = torch.linalg.norm(step, dim=-1)
        max_step_l2 = float(torch.max(step_l2).item())
        max_step_abs = float(torch.max(torch.abs(step)).item())
        mean_step_l2 = float(torch.mean(step_l2).item())
    else:
        max_step_l2 = 0.0
        max_step_abs = 0.0
        mean_step_l2 = 0.0
    valid = (
        start_gap <= float(thresholds["max_start_gap_l2"])
        and max_step_l2 <= float(thresholds["max_joint_step_l2"])
        and max_step_abs <= float(thresholds["max_joint_step_abs"])
    )
    failure_reason = None
    if start_gap > float(thresholds["max_start_gap_l2"]):
        failure_reason = "failed_start_joint_gap"
    elif max_step_l2 > float(thresholds["max_joint_step_l2"]):
        failure_reason = "failed_joint_step_l2"
    elif max_step_abs > float(thresholds["max_joint_step_abs"]):
        failure_reason = "failed_joint_step_abs"
    return {
        "valid": bool(valid),
        "status": "valid" if valid else "failed",
        "failure_reason": failure_reason,
        "start_joint_gap_l2": round(start_gap, 8) if math.isfinite(start_gap) else None,
        "joint_step_max_l2": round(max_step_l2, 8),
        "joint_step_max_abs": round(max_step_abs, 8),
        "joint_step_mean_l2": round(mean_step_l2, 8),
        "thresholds": {
            "max_start_gap_l2": float(thresholds["max_start_gap_l2"]),
            "max_joint_step_l2": float(thresholds["max_joint_step_l2"]),
            "max_joint_step_abs": float(thresholds["max_joint_step_abs"]),
        },
    }


def _optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except Exception:
        return None
