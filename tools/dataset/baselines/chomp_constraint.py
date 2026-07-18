"""B3c classic constrained trajectory-optimization baseline.

This is a dependency-free, torch implementation of the minimum CHOMP-style
opponent specified by the paper plan.  It optimizes a joint trajectory with
fixed start/goal endpoints using smoothness, shared differentiable world
collision cost, and an all-waypoint level-axis constraint penalty.  The result
is still scored by the common hard validator; the optimizer never receives a
private success definition.
"""

from __future__ import annotations

import math
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
    iterations_per_budget: int = 120
    max_iterations: int = 600
    # lr 0.03 diverges with the un-normalized level hinge; 0.008 is the largest
    # step that stayed stable across the smoke requests (see chomp trace sweep).
    learning_rate: float = 0.008
    smoothness_weight: float = 1.0
    level_weight: float = 15.0
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


def _quat_slerp(q0: list[float], q1: list[float], t: float) -> list[float]:
    """Spherical linear interpolation of two wxyz quaternions."""
    a = torch.tensor(q0, dtype=torch.float64)
    b = torch.tensor(q1, dtype=torch.float64)
    a = a / a.norm()
    b = b / b.norm()
    dot = float((a * b).sum().item())
    if dot < 0.0:  # take the shorter arc
        b = -b
        dot = -dot
    if dot > 0.9995:  # nearly identical -> nlerp to avoid div-by-zero
        out = a + t * (b - a)
        out = out / out.norm()
        return out.tolist()
    theta0 = torch.acos(torch.tensor(dot, dtype=torch.float64))
    sin0 = torch.sin(theta0)
    s0 = torch.sin((1.0 - t) * theta0) / sin0
    s1 = torch.sin(t * theta0) / sin0
    out = s0 * a + s1 * b
    return (out / out.norm()).tolist()


def _task_space_initial_trajectory(
    planner: Any,
    start_joint: list[float],
    goal_joint: list[float],
    goal_pose: list[float],
    horizon: int,
    device: str,
) -> torch.Tensor | None:
    """Initialize by IK-tracking a smooth tool-pose path from start to goal.

    A joint-space straight line between two alignment-satisfying but joint-space
    distant endpoints swings the tool through anti-alignment (~180 deg), where
    the level residual ``1 - cos`` has a vanishing gradient -- CHOMP then stalls
    in that basin (empirically stuck at ~135 deg even at 2000 iters).  Tracking
    the tool pose instead (position lerp + quaternion slerp) keeps the tool near
    the level manifold the whole way, because both endpoints are aligned and the
    slerp geodesic stays close to alignment.  IK is warm-started from the prior
    waypoint so the joint path stays continuous.  Returns ``None`` if any IK step
    fails, so the caller can fall back to the joint-space line.
    """
    horizon = int(horizon)
    start_pose = planner._fk_pose_for_joint(list(start_joint))  # [x,y,z, qw,qx,qy,qz]
    p0 = start_pose[:3]
    p1 = list(goal_pose[:3])
    q0 = start_pose[3:7]
    q1 = list(goal_pose[3:7])
    trajectory: list[list[float]] = [list(start_joint)]
    prev = torch.tensor(start_joint, dtype=torch.float32)
    for index in range(1, horizon):
        t = index / (horizon - 1)
        if index == horizon - 1:
            trajectory.append(list(goal_joint))  # pin the exact goal config
            continue
        position = [float(p0[i] + t * (p1[i] - p0[i])) for i in range(3)]
        quaternion = _quat_slerp(q0, q1, t)
        solutions = planner._ik_solve_pose_candidates(
            position=position,
            quaternion=quaternion,
            prev_solution=prev,
            return_seeds=1,
        )
        if not solutions:
            return None
        vector = [float(v) for v in solutions[0].reshape(-1).tolist()]
        if len(vector) != len(start_joint):
            return None
        trajectory.append(vector)
        prev = torch.tensor(vector, dtype=torch.float32)
    return torch.tensor(trajectory, device=device, dtype=torch.float32)


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
    start_joint = list(normalized_request["start_joint"])
    init_kind = "joint_space_line"
    initial = _initial_trajectory(start_joint, goal_joint, horizon, device)
    interior = torch.nn.Parameter(initial[1:-1].clone())
    optimizer = torch.optim.Adam([interior], lr=float(cfg.learning_rate))
    lower, upper = _joint_bounds(planner, device)
    alignment = normalized_request["alignment"]
    local_axis = alignment["local_axis"]
    target_axis = alignment["target_world_axis"]
    tolerance = float(alignment["tolerance_deg"])
    # Level penalty is computed on the *cosine* residual r = 1 - cos(angle), not
    # the acos angle: acos has a singular gradient near 0 and 180 deg, so
    # descending the angle directly overshoots a marginally-violating waypoint
    # clean past 90 deg into the anti-aligned basin (empirically 11 deg init ->
    # 165 deg after 80 iters).  The cosine residual is smooth and monotonic in
    # the angle over [0, 180], so its gradient always points toward alignment.
    tolerance_cos = 1.0 - math.cos(math.radians(max(0.0, tolerance)))
    started = time.time()
    history: list[float] = []
    total_iters = max(1, int(iterations))

    for step_index in range(total_iters):
        if time.time() - started >= float(timeout_sec):
            break
        # Anneal the hard-constraint weights upward over the schedule (survey
        # §4b: "anneal w_level up so the final trajectory satisfies c(q) <= tol").
        # A low early weight lets smoothness/collision shape the gross path; the
        # ramp then drives the level (and collision) penalty hard enough to pull
        # every waypoint onto the manifold by convergence.  Linear 1x -> 6x.
        progress = step_index / max(1, total_iters - 1)
        anneal = 1.0 + 5.0 * progress
        optimizer.zero_grad(set_to_none=True)
        trajectory = torch.cat([initial[:1], interior, initial[-1:]], dim=0)
        velocity = trajectory[1:] - trajectory[:-1]
        acceleration = velocity[1:] - velocity[:-1]
        smoothness = velocity.square().mean() + 2.0 * acceleration.square().mean()
        cos_residual = ompl_bridge._alignment_cos_residual(
            planner, trajectory, local_axis, target_axis
        )
        # Squared hinge on the raw cosine-residual violation.  NOT normalized by
        # tolerance_cos: for a tight tolerance (e.g. 8 deg -> tol_cos ~= 0.0097)
        # dividing by tol_cos^2 scales the level gradient ~1e4x, which swamps
        # smoothness/collision and tears the path apart (11 deg init -> 165 deg).
        # The raw residual keeps all three terms on a comparable scale.
        level_violation = torch.relu(cos_residual - tolerance_cos)
        level_loss = level_violation.square().mean()
        collision_loss = _collision_loss(planner, trajectory)
        loss = (
            float(cfg.smoothness_weight) * smoothness
            + anneal * float(cfg.level_weight) * level_loss
            + anneal * float(cfg.collision_weight) * collision_loss
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
            torch.tensor(local_axis, device=device, dtype=torch.float32),
            torch.tensor(target_axis, device=device, dtype=torch.float32),
            planner._constraint_eval_kinematics_fn,
        )
    return trajectory.detach().cpu().tolist(), {
        "iterations_requested": int(iterations),
        "iterations_completed": len(history),
        "initial_loss": history[0] if history else None,
        "final_loss": history[-1] if history else None,
        "final_max_alignment_deviation_deg": float(final_angles.max().item()),
        "timed_out": bool(len(history) < int(iterations)),
        "init_kind": init_kind,
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

    # The goal endpoint is pinned, so it must itself lie on the level manifold --
    # otherwise it drags the optimized tail off-constraint no matter how the
    # interior converges.  Score each IK goal by its own alignment angle and pick
    # the best-aligned one; break near-alignment ties by proximity to the start
    # (shorter, smoother trajectory).  Picking purely by proximity can select an
    # anti-parallel IK branch (~165 deg) that makes the request look infeasible.
    start = torch.tensor(normalized["start_joint"], dtype=torch.float32)
    align = normalized["alignment"]
    device = str(planner.device)
    _local = torch.tensor(align["local_axis"], device=device, dtype=torch.float32)
    _target = torch.tensor(align["target_world_axis"], device=device, dtype=torch.float32)
    _tol = float(align["tolerance_deg"])

    def _goal_angle(q: list[float]) -> float:
        return float(ompl_bridge.alignment_angle_deg(planner, q, _local, _target))

    def _goal_key(q: list[float]) -> tuple[float, float]:
        angle = _goal_angle(q)
        # Aligned goals (angle <= tol) sort first as a group, then by proximity;
        # otherwise sort by how far out of tolerance they are.
        aligned = 0.0 if angle <= _tol else 1.0
        dist = float(torch.linalg.norm(torch.tensor(q) - start).item())
        return (aligned * 1.0e6 + max(0.0, angle - _tol), dist)

    goal = min(goals, key=_goal_key)
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
