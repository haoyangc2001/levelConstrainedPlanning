"""Shared OMPL <-> level-planner bridge for the B3b/B5 external baselines.

Both OMPL baselines (B3b unconstrained RRT-Connect, B5 projection-based
constrained planning) need the *same* four bridge points into the live
level-constrained planner, and both must emit a planner ``result_dict`` whose
``candidate_records`` are scored by the *identical* A1 collision + hard
validator as every internal method (B6). Centralising that here keeps the two
baseline modules to just their planner-construction differences.

Bridge points (all borrowed from the live ``planner``; see the B1 bridge map):

* **Goal joint config** -- OMPL plans in joint space, but a request carries a
  7-DOF ``target_pose``. We reuse the planner's CuRobo IK
  (``planner._ik_solve_pose_candidates``) to turn the pose goal into one or more
  goal joint configurations.
* **Joint bounds** -- ``planner._joint_limits`` (URDF lower/upper per joint) set
  the ``RealVectorBounds`` of the ``RealVectorStateSpace``.
* **Collision validity** -- a ``StateValidityChecker`` callable that runs each
  candidate config through ``planner._evaluate_collision`` (the same per-request
  world + safety margin the internal methods use).
* **Level constraint c(q)** -- the raw axis-alignment residual in radians,
  computed via ``constraints.compute_axis_alignment_angle_batched`` on
  ``planner._constraint_eval_kinematics_fn``. Only B5 uses this (as an
  ``ompl.base.Constraint``); B3b ignores it (unconstrained floor).

The OMPL solve produces a joint waypoint path; ``build_result_dict`` resamples
it to the planner's action horizon, then hands it to
``planner._build_candidate_record_from_summary`` -- the exact code path internal
candidates take -- so the returned dict is indistinguishable in schema and is
scored by the shared validator.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Callable

import torch

from level_planner_core import constraints


# OMPL is an optional external dependency (B3b/B5 only). Import guarded so the
# baselines package still imports on hosts without the wheel; the runners raise
# a clear error at call time instead.
try:  # pragma: no cover - availability depends on the host env
    from ompl import base as ob
    from ompl import geometric as og

    OMPL_AVAILABLE = True
    OMPL_IMPORT_ERROR: str | None = None
except Exception as exc:  # pragma: no cover
    ob = None  # type: ignore[assignment]
    og = None  # type: ignore[assignment]
    OMPL_AVAILABLE = False
    OMPL_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def require_ompl() -> None:
    """Raise a descriptive error if the OMPL bindings are unavailable."""
    if not OMPL_AVAILABLE:
        raise RuntimeError(
            "OMPL Python bindings are not importable "
            f"({OMPL_IMPORT_ERROR}); install the 'ompl' wheel to run the "
            "B3b/B5 baselines."
        )


# ---------------------------------------------------------------------------
# Bridge point 1: joint bounds from the planner's URDF-derived joint limits.
# ---------------------------------------------------------------------------
def make_state_space(planner: Any) -> tuple[Any, int]:
    """Build a ``RealVectorStateSpace`` bounded by the planner's joint limits.

    Returns ``(space, dof)``. Joints whose URDF gives no explicit limit fall
    back to a wide +-2*pi range so the sampler still has a finite box.
    """
    require_ompl()
    dof = len(planner._joint_names)
    space = ob.RealVectorStateSpace(dof)
    bounds = ob.RealVectorBounds(dof)
    for index, limit in enumerate(planner._joint_limits):
        lower = limit.get("lower")
        upper = limit.get("upper")
        lower_f = float(lower) if lower is not None else -2.0 * math.pi
        upper_f = float(upper) if upper is not None else 2.0 * math.pi
        if upper_f <= lower_f:  # degenerate / missing -> widen
            lower_f, upper_f = -2.0 * math.pi, 2.0 * math.pi
        bounds.setLow(index, lower_f)
        bounds.setHigh(index, upper_f)
    space.setBounds(bounds)
    return space, dof


def make_state(space_info: Any, values: list[float], dof: int) -> Any:
    """Allocate an OMPL state on ``space_info`` and fill it from ``values``.

    This binding does **not** expose the ``ob.State(space)`` /
    ``ob.ScopedState`` constructors (they raise "no constructor defined!"). The
    supported path is ``SpaceInformation.allocState()`` returning an indexable
    ``RealVectorStateType``. Works for both plain and constrained space info
    (the constrained state is likewise index-assignable over the ambient DOF).
    """
    state = space_info.allocState()
    for i in range(dof):
        state[i] = float(values[i])
    return state


# ---------------------------------------------------------------------------
# Bridge point 2: goal joint config via the planner's CuRobo IK.
# ---------------------------------------------------------------------------
def solve_goal_joint_configs(
    planner: Any,
    normalized_request: dict[str, Any],
    *,
    return_seeds: int = 4,
) -> list[list[float]]:
    """Turn the request's 7-DOF ``target_pose`` into goal joint configs via IK.

    Seeds the IK from ``start_joint`` (matching the internal planner). Returns a
    list of DOF-length joint vectors (possibly empty when IK finds nothing
    feasible, in which case the caller reports ``failed_planner``).
    """
    target_pose = list(normalized_request["target_pose"])
    start_joint = list(normalized_request["start_joint"])
    prev = torch.tensor(start_joint, dtype=torch.float32)
    solutions = planner._ik_solve_pose_candidates(
        position=target_pose[:3],
        quaternion=target_pose[3:7],
        prev_solution=prev,
        return_seeds=int(return_seeds),
    )
    configs: list[list[float]] = []
    for solution in solutions:
        vector = solution.reshape(-1).tolist()
        if len(vector) == len(start_joint):
            configs.append([float(v) for v in vector])
    return configs


# ---------------------------------------------------------------------------
# Bridge point 3: collision validity checker sharing the planner's world.
# ---------------------------------------------------------------------------
def make_validity_checker(planner: Any, dof: int) -> Callable[[Any], bool]:
    """Return an OMPL ``State -> bool`` validity callable backed by A1 collision.

    A config is valid iff it is within joint bounds *and* the planner's
    per-request collision query reports it clear of the safety margin. When the
    world is obstacle-free ``_evaluate_collision`` returns ``None`` (nothing to
    hit) -> valid.
    """
    margin = float(planner._validator_thresholds().get("collision_safety_margin_m", 0.005))

    def is_valid(state: Any) -> bool:
        config = [float(state[i]) for i in range(dof)]
        collision = planner._evaluate_collision([config])
        if collision is None:
            return True
        cost = collision.get("collision_cost")
        if cost is not None:
            return float(cost) <= 0.0
        distance = collision.get("min_distance_m")
        if distance is not None:
            return float(distance) >= margin
        return True

    return is_valid


# ---------------------------------------------------------------------------
# Bridge point 4: the level constraint c(q) as an OMPL Constraint (B5 only).
# ---------------------------------------------------------------------------
def _alignment_axes(normalized_request: dict[str, Any], planner: Any) -> tuple[list[float], list[float], float]:
    alignment = dict(normalized_request.get("alignment") or {})
    local_axis = list(alignment.get("local_axis") or planner.config.local_axis)
    target_axis = list(alignment.get("target_world_axis") or planner.config.target_world_axis)
    tolerance_deg = float(alignment.get("tolerance_deg", planner.config.level_tolerance_deg))
    return local_axis, target_axis, tolerance_deg


def alignment_angle_deg(planner: Any, config: list[float], local_axis: list[float], target_axis: list[float]) -> float:
    """Axis-alignment angle (degrees) of a single joint config via shared FK."""
    positions = torch.tensor([config], device=planner.device, dtype=torch.float32)
    local = torch.tensor(local_axis, device=planner.device, dtype=torch.float32)
    target = torch.tensor(target_axis, device=planner.device, dtype=torch.float32)
    angle = constraints.compute_axis_alignment_angle_batched(
        positions, local, target, planner._constraint_eval_kinematics_fn
    )
    return float(angle.reshape(-1)[0].item())


def make_level_constraint(planner: Any, normalized_request: dict[str, Any]) -> Any:
    """Build an ``ompl.base.Constraint`` for ``c(q) = align_angle(q) (radians)``.

    The projection space drives the residual to zero (perfect alignment), which
    is stricter than ``<= tolerance`` and therefore a faithful *hard-constraint*
    adversary: every projected state sits on the level manifold. ``function``
    returns the alignment angle in **radians** (OMPL projects toward 0);
    ``jacobian`` is left to OMPL's finite differencing.
    """
    require_ompl()
    dof = len(planner._joint_names)
    local_axis, target_axis, _ = _alignment_axes(normalized_request, planner)

    class LevelConstraint(ob.Constraint):
        def __init__(self) -> None:
            super().__init__(dof, 1)

        def function(self, x, out) -> None:  # noqa: N802 (OMPL API name)
            config = [float(x[i]) for i in range(dof)]
            out[0] = math.radians(alignment_angle_deg(planner, config, local_axis, target_axis))

    return LevelConstraint()


# ---------------------------------------------------------------------------
# OMPL path -> planner result_dict (shared A1 collision + hard validator).
# ---------------------------------------------------------------------------
def _resample_path(waypoints: list[list[float]], horizon: int) -> list[list[float]]:
    """Linearly resample a joint waypoint list to exactly ``horizon`` points."""
    if not waypoints:
        return []
    if len(waypoints) == horizon:
        return [list(map(float, wp)) for wp in waypoints]
    tensor = torch.tensor(waypoints, dtype=torch.float32)
    if tensor.shape[0] == 1:
        tensor = tensor.repeat(horizon, 1)
    else:
        import torch.nn.functional as F

        tensor = (
            F.interpolate(
                tensor.transpose(0, 1).unsqueeze(0),
                size=int(horizon),
                mode="linear",
                align_corners=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            .contiguous()
        )
    return tensor.tolist()


def action_horizon(planner: Any) -> int:
    """The planner's trajectory horizon (points per plan)."""
    try:
        return int(planner._planner.trajopt_solver.action_horizon)
    except Exception:
        return 32


def uniform_retiming_dt(
    points: list[list[float]],
    joint_limits: list[dict[str, Any]],
    *,
    speed_scale: float = 0.5,
    minimum_dt_sec: float = 0.025,
) -> float | None:
    """Return a conservative uniform timestep for an external geometric path.

    OMPL returns geometry without timestamps.  A2 requires dimensioned motion
    quality for every paper baseline, so we choose the smallest uniform ``dt``
    that respects every URDF joint velocity limit after applying the planner's
    configured speed scale.  This is deliberately simple and deterministic; a
    full acceleration-aware TOPP-RA pass can replace it later without changing
    the result schema.
    """
    if len(points) < 2:
        return None
    trajectory = torch.tensor(points, dtype=torch.float32)
    max_step = torch.max(torch.abs(trajectory[1:] - trajectory[:-1]), dim=0).values
    scale = max(float(speed_scale), 1.0e-6)
    required = float(minimum_dt_sec)
    for index, step in enumerate(max_step.tolist()):
        limit = joint_limits[index] if index < len(joint_limits) else {}
        velocity = limit.get("velocity")
        if velocity is None or float(velocity) <= 0.0:
            continue
        required = max(required, float(step) / (float(velocity) * scale))
    return float(required)


def build_result_dict(
    planner: Any,
    *,
    normalized_request: dict[str, Any],
    method_name: str,
    source_type: str,
    waypoints: list[list[float]] | None,
    solve_time_sec: float,
    solver_status: str,
    failure_reason: str | None,
    planner_extra: dict[str, Any] | None = None,
    out_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Turn an OMPL joint path into a fully scored planner ``result_dict``.

    Reuses ``planner._build_candidate_record_from_summary`` so the OMPL
    trajectory is scored by the *identical* A1 collision + hard validator as
    every internal method (B6). ``waypoints`` is the raw OMPL joint path (list
    of DOF-length configs); ``None``/empty means the OMPL solve found no path,
    which yields a ``failed_planner`` result with no successful candidate.
    """
    from level_planner_core.result_schema import (
        PlannerResult,
        STATUS_FAILED_PLANNER,
        STATUS_FAILED_VALIDATION,
        STATUS_SUCCESS,
    )

    request_id = str(normalized_request.get("request_id") or "request")
    start_joint = list(normalized_request["start_joint"])
    target_pose = list(normalized_request["target_pose"])
    local_axis, target_axis, tolerance_deg = _alignment_axes(normalized_request, planner)

    solved = bool(waypoints)
    horizon = action_horizon(planner)
    points = _resample_path(waypoints or [], horizon) if solved else []
    trajectory_tensor = (
        torch.tensor(points, device=planner.device, dtype=torch.float32) if points else None
    )
    interpolation_dt = uniform_retiming_dt(
        points,
        list(getattr(planner, "_joint_limits", []) or []),
        speed_scale=float(getattr(planner.config, "speed_scale", 0.5)),
    )

    # Compute the metrics the hard validator reads for alignment + goal. The
    # collision / continuity / joint-limit / velocity checks are computed by the
    # validator directly from the trajectory points.
    metrics: dict[str, Any] = {}
    goal_metrics: dict[str, Any] = {}
    if trajectory_tensor is not None:
        positions_t = torch.tensor(points, device=planner.device, dtype=torch.float32)
        local = torch.tensor(local_axis, device=planner.device, dtype=torch.float32)
        target = torch.tensor(target_axis, device=planner.device, dtype=torch.float32)
        angles = constraints.compute_axis_alignment_angle_batched(
            positions_t, local, target, planner._constraint_eval_kinematics_fn
        ).reshape(-1)
        max_dev = float(torch.max(angles).item())
        mean_dev = float(torch.mean(angles).item())
        goal_summary = planner._summarize_terminal_goal(trajectory_tensor, target_pose)
        goal_metrics = {
            "position_error_m": goal_summary.get("terminal_position_error_m"),
            "orientation_error_rad": goal_summary.get("terminal_orientation_error_rad"),
        }
        metrics = {
            "selected": True,
            "max_alignment_deviation_deg": round(max_dev, 8),
            "mean_alignment_deviation_deg": round(mean_dev, 8),
            **goal_metrics,
        }

    status = STATUS_SUCCESS if solved and solver_status == "success" else STATUS_FAILED_PLANNER
    candidate_status = "success" if solved else "failed"
    summary = {
        "candidate_id": f"{method_name}_00",
        "source_type": source_type,
        "source_label": method_name,
        "provider": method_name,
        "provider_mode": "external",
        "status": candidate_status,
        "selected": solved,
        "metrics": metrics,
        "failure_reason": None if solved else (failure_reason or "ompl_no_path"),
        "optimizer_result": {
            "status": solver_status,
            "success": bool(solved),
            "solve_time_sec": round(float(solve_time_sec), 6),
            "solver_status": solver_status,
            "interpolation_dt_sec": interpolation_dt,
            "retiming": "uniform_urdf_velocity_limit",
        },
        "solve_time_sec": round(float(solve_time_sec), 6),
        "result_status": solver_status,
    }

    record = planner._build_candidate_record_from_summary(
        summary,
        request_id=request_id,
        run_id=request_id,
        trajectory=trajectory_tensor,
        final_status=status,
        final_failure_reason=failure_reason,
        alignment_tolerance_deg=float(tolerance_deg),
        start_joint=start_joint,
    )
    validator = record.get("validator_metrics") or {}
    checks = validator.get("checks") or {}
    if status == STATUS_SUCCESS and not bool(validator.get("valid")):
        status = STATUS_FAILED_VALIDATION
        failure_reason = str(validator.get("failure_reason") or "external_path_failed_hard_validation")

    selected_id = record.get("candidate_id") if status == STATUS_SUCCESS else None
    summary["selected"] = status == STATUS_SUCCESS
    (summary.get("metrics") or {})["selected"] = status == STATUS_SUCCESS
    lifecycle = record.get("lifecycle") or {}
    lifecycle["selected"] = status == STATUS_SUCCESS
    labels = record.get("labels") or {}
    labels["planner_status"] = status
    labels["selected"] = status == STATUS_SUCCESS
    labels["positive_for_critic"] = bool(status == STATUS_SUCCESS and validator.get("valid"))
    labels["negative_for_critic"] = not labels["positive_for_critic"]
    result = PlannerResult(
        request_id=request_id,
        status=status,
        failure_reason=None if status == STATUS_SUCCESS else failure_reason,
        selected_trajectory=points if status == STATUS_SUCCESS else None,
        metrics={
            "solve_time_sec": round(float(solve_time_sec), 6),
            "selected_candidate_id": selected_id,
            "success_source": source_type if status == STATUS_SUCCESS else None,
            "alignment": {
                "tolerance_deg": float(tolerance_deg),
                "max_alignment_deviation_deg": metrics.get("max_alignment_deviation_deg"),
                "selected_max_alignment_deviation_deg": metrics.get("max_alignment_deviation_deg"),
                "mean_alignment_deviation_deg": metrics.get("mean_alignment_deviation_deg"),
            },
            "goal": {
                "terminal_position_error_m": goal_metrics.get("position_error_m"),
                "terminal_orientation_error_rad": goal_metrics.get("orientation_error_rad"),
            },
            "collision_safety": dict(checks.get("collision_safety") or {}),
            "velocity_acceleration": dict(checks.get("velocity_acceleration") or {}),
            "joint_limit": dict(checks.get("joint_limit") or {}),
            "hard_validator": validator,
            "world": dict(getattr(planner, "_world_summary", {}) or {}),
            "baseline": {"method": method_name, **(planner_extra or {})},
        },
        candidates=[summary],
        candidate_records=[record],
        planner_run_record={
            "method": method_name,
            "source_type": source_type,
            "solver_status": solver_status,
            "fallback_trace": [],
            **(planner_extra or {}),
        },
    )
    if out_dir is not None:
        planner._write_artifacts(result, out_dir)
    return result.to_dict()

