"""B3c classic constrained trajectory-optimization baseline.

This is a dependency-free, torch implementation of the minimum CHOMP-style
opponent specified by the paper plan.  It optimizes a joint trajectory with
fixed start/goal endpoints using smoothness, shared differentiable world
collision cost, and an all-waypoint level-axis constraint penalty.  The result
is still scored by the common hard validator; the optimizer never receives a
private success definition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from level_planner_core import constraints

from ..methods import MethodSpec, register_method
from . import ompl_bridge


METHOD_NAME = "baseline/chomp_constraint"
SOURCE_TYPE = "external_constrained_opt_chomp"


@dataclass(frozen=True)
class ChompConfig:
    iterations_per_budget: int = 80
    max_iterations: int = 400
    learning_rate: float = 0.03
    smoothness_weight: float = 1.0
    level_weight: float = 30.0
    collision_weight: float = 80.0


def _iteration_budget(request: dict[str, Any], config: ChompConfig) -> int:
    metadata = dict(request.get("metadata") or {})
    seed_policy = dict(request.get("seed_policy") or {})
    budget = int(metadata.get("compute_budget_solve_calls") or seed_policy.get("k_generate") or 1)
    return min(int(config.max_iterations), max(1, budget) * int(config.iterations_per_budget))


def _joint_bounds(planner: Any, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    lower = []
    upper = []
    for limit in planner._joint_limits:
        lo = limit.get("lower")
        hi = limit.get("upper")
        lower.append(float(lo) if lo is not None else -2.0 * torch.pi)
        upper.append(float(hi) if hi is not None else 2.0 * torch.pi)
    return (
        torch.tensor(lower, device=device, dtype=torch.float32),
        torch.tensor(upper, device=device, dtype=torch.float32),
    )


def _initial_trajectory(start: list[float], goal: list[float], horizon: int, device: str) -> torch.Tensor:
    start_t = torch.tensor(start, device=device, dtype=torch.float32)
    goal_t = torch.tensor(goal, device=device, dtype=torch.float32)
    alpha = torch.linspace(0.0, 1.0, int(horizon), device=device).unsqueeze(-1)
    return start_t.unsqueeze(0) + alpha * (goal_t - start_t).unsqueeze(0)


def _collision_loss(planner: Any, trajectory: torch.Tensor) -> torch.Tensor:
    if planner._collision_checker is None:
        return trajectory.sum() * 0.0
    world_cost, _self_cost = (
        planner._collision_checker.get_scene_self_collision_distance_from_joint_trajectory(
            trajectory.unsqueeze(0)
        )
    )
    # CuRobo's custom CUDA backward supports a scalar reduction of its per-sphere
    # activation-band costs, but the installed V2 build mis-handles the gradient
    # shape produced by an elementwise square over [B,T,S].  Costs are already
    # non-negative penetration/margin residuals, so their normalized linear sum
    # is the appropriate CHOMP obstacle potential and follows the supported path.
    world_loss = world_cost.sum() / max(int(world_cost.numel()), 1)
    # A1's shared acceptance metric is world collision.  The installed CuRobo
    # build returns self cost as [B,T,1] but its backward incorrectly assumes the
    # world-sphere dimension, so self collision remains the optimizer's native
    # concern and is not differentiated in this external baseline.
    return world_loss


def optimize_chomp_trajectory(
    planner: Any,
    normalized_request: dict[str, Any],
    goal_joint: list[float],
    *,
    iterations: int,
    timeout_sec: float,
    config: ChompConfig | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    """Optimize one fixed-endpoint CHOMP-style trajectory."""
    cfg = config or ChompConfig()
    device = str(planner.device)
    horizon = ompl_bridge.action_horizon(planner)
    initial = _initial_trajectory(
        list(normalized_request["start_joint"]), goal_joint, horizon, device
    )
    interior = torch.nn.Parameter(initial[1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.learning_rate))
    lower, upper = _joint_bounds(planner, device)
    alignment = normalized_request["alignment"]
    local_axis = torch.tensor(alignment["local_axis"], device=device, dtype=torch.float32)
    target_axis = torch.tensor(alignment["target_world_axis"], device=device, dtype=torch.float32)
    tolerance = float(alignment["tolerance_deg"])
    started = time.time()
    history: list[float] = []

    for _ in range(int(iterations)):
        if time.time() - started >= float(timeout_sec):
            break
        optimizer.zero_grad(set_to_none=True)
        trajectory = torch.cat([initial[:1], interior, initial[-1:]], dim=0)
        velocity = trajectory[1:] - trajectory[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        smoothness = velocity.square().mean() + 2.0 * acceleration.square().mean()
        angles = constraints.compute_axis_alignment_angle_batched(
            trajectory,
            local_axis,
            target_axis,
            planner._constraint_eval_kinematics_fn,
        )
        level_violation = torch.relu(angles - tolerance)
        level_loss = (level_violation / max(tolerance, 1.0)).square().mean()
        collision_loss = _collision_loss(planner, trajectory)
        loss = (
            float(cfg.smoothness_weight) * smoothness
            + float(cfg.level_weight) * level_loss
            + float(cfg.collision_weight) * collision_loss
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([interior], max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            interior.clamp_(lower, upper)
        history.append(float(loss.detach().item()))

    with torch.no_grad():
        trajectory = torch.cat([initial[:1], interior, initial[-1:]], dim=0)
        final_angles = constraints.compute_axis_alignment_angle_batched(
            trajectory,
            local_axis,
            target_axis,
            planner._constraint_eval_kinematics_fn,
        )
    return trajectory.detach().cpu().tolist(), {
        "iterations_requested": int(iterations),
        "iterations_completed": len(history),
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "final_max_alignment_deviation_deg": float(final_angles.max().item()),
        "timed_out": bool(len(history) < int(iterations)),
    }


def run_chomp_constraint(
    request: dict[str, Any],
    config: Any,
    *,
    planner: Any,
    out_dir: Any = None,
    **_context: Any,
) -> dict[str, Any]:
    if planner is None:
        raise ValueError(f"{METHOD_NAME} requires a live planner instance; got None")
    normalized = planner._normalize_request(request)
    goals = ompl_bridge.solve_goal_joint_configs(planner, normalized, return_seeds=8)
    started = time.time()
    if not goals:
        return ompl_bridge.build_result_dict(
            planner,
            normalized_request=normalized,
            method_name=METHOD_NAME,
            source_type=SOURCE_TYPE,
            waypoints=None,
            solve_time_sec=time.time() - started,
            solver_status="no_goal_ik",
            failure_reason="ik_no_solution",
            planner_extra={"planner_family": "chomp", "constrained": True},
            out_dir=out_dir,
        )

    start = torch.tensor(normalized["start_joint"], dtype=torch.float32)
    goal = min(goals, key=lambda q: float(torch.linalg.norm(torch.tensor(q) - start).item()))
    chomp_config = ChompConfig()
    iterations = _iteration_budget(request, chomp_config)
    timeout_sec = float((request.get("seed_policy") or {}).get("timeout_sec") or 10.0)
    points, trace = optimize_chomp_trajectory(
        planner,
        normalized,
        goal,
        iterations=iterations,
        timeout_sec=timeout_sec,
        config=chomp_config,
    )
    return ompl_bridge.build_result_dict(
        planner,
        normalized_request=normalized,
        method_name=METHOD_NAME,
        source_type=SOURCE_TYPE,
        waypoints=points,
        solve_time_sec=time.time() - started,
        solver_status="success",
        failure_reason=None,
        planner_extra={
            "planner_family": "chomp",
            "constrained": True,
            "constraint_mode": "all_waypoint_penalty",
            **trace,
        },
        out_dir=out_dir,
    )


register_method(
    MethodSpec(
        name=METHOD_NAME,
        mode="native",
        external=True,
        runner=run_chomp_constraint,
        description=(
            "B3c dependency-free CHOMP-style constrained optimizer with shared "
            "differentiable collision cost and all-waypoint level penalty."
        ),
    )
)
