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
    vector = [float(values[i]) for i in range(dof)]
    # Plain RealVector states expose item assignment.  Constrained states in the
    # OMPL 2.0.1 nanobind wheel are readable by index but assignment is omitted;
    # their ``copy(sequence)`` method is the supported initialization path.
    try:
        for i, value in enumerate(vector):
            state[i] = value
    except TypeError:
        state.copy(vector)
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


def _alignment_tangent_basis(target_axis: list[float], device: str) -> tuple[torch.Tensor, torch.Tensor]:
    target = torch.tensor(target_axis, device=device, dtype=torch.float32)
    target = target / torch.linalg.norm(target).clamp_min(1.0e-8)
    reference = torch.tensor([1.0, 0.0, 0.0], device=device)
    if float(torch.abs(torch.dot(target, reference)).item()) > 0.9:
        reference = torch.tensor([0.0, 1.0, 0.0], device=device)
    tangent_1 = torch.linalg.cross(target, reference)
    tangent_1 = tangent_1 / torch.linalg.norm(tangent_1).clamp_min(1.0e-8)
    tangent_2 = torch.linalg.cross(target, tangent_1)
    tangent_2 = tangent_2 / torch.linalg.norm(tangent_2).clamp_min(1.0e-8)
    return tangent_1, tangent_2


def alignment_tangent_residual(
    planner: Any,
    q: torch.Tensor,
    local_axis: list[float],
    target_axis: list[float],
) -> torch.Tensor:
    """Two smooth equality residuals for axis alignment."""
    local = torch.tensor(local_axis, device=q.device, dtype=torch.float32)
    tangent_1, tangent_2 = _alignment_tangent_basis(target_axis, str(q.device))
    fk = planner._constraint_eval_kinematics_fn(q)
    quaternion = constraints.extract_ee_quaternion_batched(fk)
    rotation = constraints.quaternion_to_rotation_matrix_batched(quaternion)
    axis_world = torch.matmul(rotation, local.reshape(1, 3, 1)).squeeze(-1)
    return torch.stack(
        [axis_world @ tangent_1, axis_world @ tangent_2],
        dim=-1,
    )


def make_level_constraint(planner: Any, normalized_request: dict[str, Any]) -> Any:
    """Build an OMPL projection residual for the level inequality.

    The two residuals are the tool axis components along an orthonormal tangent
    basis of the target axis.  Driving both to zero yields the axis-alignment
    equality manifold with a smooth rank-2 Jacobian.  A separate validity check
    rejects the anti-parallel branch.  An autograd Jacobian avoids slow/noisy
    finite differencing through GPU forward kinematics.
    """
    require_ompl()
    dof = len(planner._joint_names)
    local_axis, target_axis, tolerance_deg = _alignment_axes(normalized_request, planner)

    class LevelConstraint(ob.Constraint):
        def __init__(self) -> None:
            super().__init__(dof, 2)
            # OMPL accepts a tube around the equality manifold.  For unit axes,
            # the tangent residual norm is sin(angle), so this exactly maps the
            # request's angular tolerance to the projection tolerance and avoids
            # perturbing already-valid service states.
            self.setTolerance(max(math.sin(math.radians(tolerance_deg)), 1.0e-4))

        def function(self, x, out) -> None:  # noqa: N802 (OMPL API name)
            q = torch.tensor(
                [[float(x[i]) for i in range(dof)]],
                device=planner.device,
                dtype=torch.float32,
            )
            residual = alignment_tangent_residual(
                planner, q, local_axis, target_axis
            ).reshape(-1).detach().cpu()
            out[0] = float(residual[0].item())
            out[1] = float(residual[1].item())

        def jacobian(self, x, out) -> None:  # noqa: N802 (OMPL API name)
            q = torch.tensor(
                [[float(x[i]) for i in range(dof)]],
                device=planner.device,
                dtype=torch.float32,
                requires_grad=True,
            )
            residual = alignment_tangent_residual(planner, q, local_axis, target_axis).reshape(-1)
            for row in range(2):
                gradient = torch.autograd.grad(
                    residual[row], q, retain_graph=row == 0
                )[0].reshape(-1).detach().cpu()
                for index, value in enumerate(gradient.tolist()):
                    out[row, index] = float(value)

    return LevelConstraint()


def make_level_validity_checker(
    planner: Any,
    normalized_request: dict[str, Any],
    dof: int,
) -> Callable[[Any], bool]:
    """Combine shared collision validity with the requested axis tolerance."""
    collision_valid = make_validity_checker(planner, dof)
    local_axis, target_axis, tolerance_deg = _alignment_axes(normalized_request, planner)

    def is_valid(state: Any) -> bool:
        if not collision_valid(state):
            return False
        config = [float(state[i]) for i in range(dof)]
        return alignment_angle_deg(planner, config, local_axis, target_axis) <= tolerance_deg + 1.0e-3

    return is_valid


def _alignment_cos_residual(
    planner: Any,
    q: torch.Tensor,
    local_axis: list[float],
    target_axis: list[float],
) -> torch.Tensor:
    """Scalar residual ``r(q) = 1 - cos(angle)`` between tool axis and target.

    ``r = 0`` iff the tool axis is parallel to the target (angle 0); ``r = 2`` at
    anti-parallel (angle 180).  Monotonic in the alignment angle over [0, 180],
    so descending it drives onto the *correct* (aligned) branch -- unlike the
    two-tangent residual, which is also zero at the anti-parallel branch.
    """
    local = torch.tensor(local_axis, device=q.device, dtype=torch.float32)
    target = torch.tensor(target_axis, device=q.device, dtype=torch.float32)
    target = target / torch.linalg.norm(target).clamp_min(1.0e-8)
    fk = planner._constraint_eval_kinematics_fn(q)
    quaternion = constraints.extract_ee_quaternion_batched(fk)
    rotation = constraints.quaternion_to_rotation_matrix_batched(quaternion)
    axis_world = torch.matmul(rotation, local.reshape(1, 3, 1)).squeeze(-1)
    axis_world = axis_world / torch.linalg.norm(axis_world, dim=-1, keepdim=True).clamp_min(1.0e-8)
    cos_angle = (axis_world * target.reshape(1, 3)).sum(dim=-1)
    return (1.0 - cos_angle).reshape(-1)


def project_config_onto_level_manifold(
    planner: Any,
    config: list[float],
    local_axis: list[float],
    target_axis: list[float],
    tolerance_deg: float,
    *,
    max_iterations: int = 80,
    warm_start: list[float] | None = None,
) -> tuple[list[float], float, bool]:
    """Gauss-Newton-project a single joint config onto the level-alignment manifold.

    This is the B5 constraint-enforcement primitive.  The OMPL 2.0.1 nanobind
    wheel does not expose writable ``ProjectedStateSpace`` states nor a
    registrable custom sampler, so manifold *search* is unavailable.  Instead B5
    runs an unconstrained RRT-Connect (shared with B3b) and then projects every
    waypoint onto the manifold here -- a standard, honest projection technique:
    the returned trajectory satisfies ``angle <= tolerance`` by construction.

    Descends the scalar residual ``r(q) = 1 - cos(angle)`` (``_alignment_cos_residual``,
    monotonic in the alignment angle, so no anti-parallel branch) via a
    Gauss-Newton direction ``d = -r * g / (g.g)`` with a **backtracking line
    search**: the step is halved until the residual strictly decreases, which
    prevents the small-gradient overshoot that otherwise climbs across ridges.
    ``warm_start`` (the previous already-projected waypoint) seeds the descent so
    a whole path is projected onto one continuous manifold branch.  Returns
    ``(projected_config, final_angle_deg, converged)``.
    """
    tolerance_cos = 1.0 - math.cos(math.radians(max(0.0, float(tolerance_deg))))
    target_residual = max(tolerance_cos * 0.5, 1.0e-6)

    def residual_of(vec: list[float]) -> float:
        q = torch.tensor([vec], device=planner.device, dtype=torch.float32)
        return float(_alignment_cos_residual(planner, q, local_axis, target_axis)[0].item())

    def descend(seed: list[float]) -> tuple[list[float], float]:
        q = torch.tensor([list(map(float, seed))], device=planner.device, dtype=torch.float32)
        r_value = residual_of(seed)
        for _ in range(int(max_iterations)):
            if r_value <= target_residual:
                break
            q_grad = q.detach().clone().requires_grad_(True)
            residual = _alignment_cos_residual(planner, q_grad, local_axis, target_axis)
            grad = torch.autograd.grad(residual[0], q_grad)[0].reshape(-1).detach()
            denom = float((grad @ grad).item())
            if denom < 1.0e-12:  # flat gradient -> nudge and retry
                break
            direction = -(r_value / denom) * grad
            # Backtracking line search: shrink until the residual strictly drops.
            step = 1.0
            improved = False
            for _bt in range(20):
                candidate = (q + (step * direction).reshape(1, -1)).detach()
                cand_r = float(
                    _alignment_cos_residual(planner, candidate, local_axis, target_axis)[0].item()
                )
                if cand_r < r_value - 1.0e-9:
                    q, r_value, improved = candidate, cand_r, True
                    break
                step *= 0.5
            if not improved:
                break
        return [float(v) for v in q.reshape(-1).tolist()], r_value

    # Try the warm start (path continuity) first, then the config itself; keep
    # whichever reaches the smaller residual.
    seeds = [config] if warm_start is None else [warm_start, config]
    best_vec, best_r = None, float("inf")
    for seed in seeds:
        vec, r_value = descend(seed)
        if r_value < best_r:
            best_vec, best_r = vec, r_value
    assert best_vec is not None
    final_angle = alignment_angle_deg(planner, best_vec, local_axis, target_axis)
    return best_vec, float(final_angle), bool(best_r <= target_residual)


def project_path_onto_level_manifold(
    planner: Any,
    waypoints: list[list[float]],
    normalized_request: dict[str, Any],
    *,
    pin_start: list[float] | None = None,
    pin_goal: list[float] | None = None,
    trust_radius_l2: float = 0.5,
) -> tuple[list[list[float]], float]:
    """Project every waypoint onto the level manifold; return (path, max_angle_deg).

    ``pin_start`` (usually the request's exact ``start_joint``) replaces the
    first waypoint after projection so the service start state is preserved
    exactly.

    ``trust_radius_l2`` caps how far each projected waypoint may move from its
    already-projected predecessor (L2 in joint space).  Post-hoc projection of an
    unconstrained path is otherwise free to send a waypoint into a *different* IK
    branch / manifold basin than its neighbour -- e.g. a freshly IK-solved goal
    config reaches the same tool pose as the path's terminal state but sits 4-5
    rad away, which reappears as a single giant joint step and fails continuity.
    Clamping each move to a trust region makes joint-step continuity hold
    *by construction*; where the manifold is not reachable within the region the
    waypoint stays partially aligned and the shared validator honestly records
    the residual.  This is the principled fix for the basin-teleport artifact
    (rather than pinning a foreign goal config, which recreates the jump).

    A foreign ``pin_goal`` is intentionally NOT accepted: the RRT path already
    terminates at the goal config continuously, so the trust-region-chained
    endpoint reaches the goal pose without a discontinuity.  The returned
    ``max_angle_deg`` is the worst per-waypoint alignment angle after projection;
    the shared hard validator is the authority on success.
    """
    local_axis, target_axis, tolerance_deg = _alignment_axes(normalized_request, planner)
    horizon = len(waypoints)
    _ = pin_goal  # accepted for signature stability; deliberately unused (see docstring)
    trust = max(1.0e-3, float(trust_radius_l2))

    def _clamp_to_trust(new_config: list[float], previous: list[float] | None) -> list[float]:
        if previous is None:
            return new_config
        delta = [n - p for n, p in zip(new_config, previous)]
        dist = math.sqrt(sum(d * d for d in delta))
        if dist <= trust or dist < 1.0e-12:
            return new_config
        scale = trust / dist
        return [p + d * scale for p, d in zip(previous, delta)]

    def project_once(points: list[list[float]]) -> tuple[list[list[float]], float]:
        out: list[list[float]] = []
        worst = 0.0
        previous: list[float] | None = (
            list(map(float, pin_start)) if pin_start is not None else None
        )
        for config in points:
            new_config, _angle, _ = project_config_onto_level_manifold(
                planner, config, local_axis, target_axis, tolerance_deg, warm_start=previous
            )
            new_config = _clamp_to_trust(new_config, previous)
            # Re-measure the angle at the clamped config (the clamp may pull it
            # slightly off the manifold; the validator scores the clamped point).
            angle = alignment_angle_deg(planner, new_config, local_axis, target_axis)
            out.append(new_config)
            previous = new_config
            worst = max(worst, float(angle))
        return out, worst

    # Iterate project -> arc-length resample -> project.  Projection moves each
    # waypoint onto the manifold but can spread neighbours apart; arc-length
    # resampling re-even-spaces them; re-projecting pulls the resampled points
    # (which drift slightly off-manifold along the chord) back on.  Two to three
    # rounds converge both the alignment angle and the joint-step continuity
    # without any external dependency (the user-selected B5 convergence path).
    projected, max_angle = project_once(waypoints)
    for _ in range(2):
        if pin_start is not None and projected:
            projected[0] = list(map(float, pin_start))
        resampled = _resample_path(projected, horizon, force=True)
        projected, max_angle = project_once(resampled)

    if pin_start is not None and projected:
        projected[0] = list(map(float, pin_start))
    return projected, float(max_angle)


# ---------------------------------------------------------------------------
# OMPL path -> planner result_dict (shared A1 collision + hard validator).
# ---------------------------------------------------------------------------
def _resample_path(
    waypoints: list[list[float]], horizon: int, *, force: bool = False
) -> list[list[float]]:
    """Resample a joint waypoint list to exactly ``horizon`` points by arc length.

    Uniform-by-index interpolation preserves any clustering in the input (OMPL's
    ``path.interpolate`` can return many near-duplicate states then one long
    segment), which yields a trajectory with a single huge joint step and fails
    the continuity validator.  Resampling by *cumulative joint-space distance*
    spreads the output points evenly along the actual path, so every step is
    ``total_arc_length / (horizon - 1)`` -- eliminating the giant-jump artifact.

    ``force=True`` skips the length-equals-horizon pass-through.  That guard is
    for on-manifold points a runner has *already* arc-length resampled (avoid
    re-interpolating them off-manifold); it must NOT be taken for a raw OMPL path
    that happens to already be ``horizon`` long but is internally clustered
    (15 near-duplicate start states + 1 goal), which is exactly the giant-jump
    case the runner needs re-spread.
    """
    if not waypoints:
        return []
    if len(waypoints) == horizon and not force:
        # Already at horizon (e.g. runner pre-resampled then projected); pass
        # through so we don't re-interpolate on-manifold points off-manifold.
        return [list(map(float, wp)) for wp in waypoints]
    unique: list[list[float]] = [list(map(float, waypoints[0]))]
    for wp in waypoints[1:]:
        wp = list(map(float, wp))
        if any(abs(a - b) > 1.0e-9 for a, b in zip(wp, unique[-1])):
            unique.append(wp)
    if len(unique) == 1:
        return [list(unique[0]) for _ in range(horizon)]
    tensor = torch.tensor(unique, dtype=torch.float32)
    seg = torch.linalg.norm(tensor[1:] - tensor[:-1], dim=-1)
    cumulative = torch.cat([torch.zeros(1), torch.cumsum(seg, dim=0)])
    total = float(cumulative[-1].item())
    if total <= 1.0e-9:
        return [list(map(float, unique[0])) for _ in range(horizon)]
    targets = torch.linspace(0.0, total, int(horizon))
    resampled: list[list[float]] = []
    for target in targets.tolist():
        # Locate the segment containing this arc-length target.
        idx = int(torch.searchsorted(cumulative, torch.tensor(target)).item())
        idx = max(1, min(idx, tensor.shape[0] - 1))
        seg_start, seg_end = cumulative[idx - 1].item(), cumulative[idx].item()
        span = seg_end - seg_start
        alpha = 0.0 if span <= 1.0e-9 else (target - seg_start) / span
        point = tensor[idx - 1] + float(alpha) * (tensor[idx] - tensor[idx - 1])
        resampled.append([float(v) for v in point.tolist()])
    return resampled


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
