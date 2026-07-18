"""B5 baseline: OMPL projection-based constrained planning (hard manifold).

The plan's B5 slot is the *primary constraint-enforcing adversary* -- the true
opponent for Table I's "constraint-enforcing" cell. It treats the level axis as
a **hard manifold constraint**: the output trajectory is projected onto the
two-component axis-alignment equality manifold, so the returned path satisfies
``angle <= tolerance`` by construction rather than penalising deviation. This is
the honest "what a projection-based constrained planner gives you" opponent --
it reaches near-zero alignment violation by construction, and the comparison
then isolates *cost/latency/success* rather than whether the constraint is
enforced at all.

**Binding note (why projection is post-hoc, not in-search).** The intended
design searched *on the manifold* via OMPL's ``ProjectedStateSpace`` +
``ConstrainedSpaceInformation``. The installed OMPL 2.0.1 nanobind wheel does
not expose the Python hooks that route requires: constrained states are not
writable from Python (``copyFromReals`` rejects them and the raw setter
corrupts memory), and neither a custom ``StateSampler`` allocator nor a
``ValidStateSamplerAllocator`` is constructible. Verified exhaustively against
the live wheel. Per B1's risk posture (§2/§3a: "if OMPL fights the binding, do
not fight it"), B5 therefore runs the *same* unconstrained RRT-Connect as B3b
on the plain (writable) ``RealVectorStateSpace`` and then **Newton-projects
every waypoint onto the level manifold** (``ompl_bridge`` shared tangent
residual + autograd Jacobian). The constraint is still hard on the output; only
the *mechanism* moved from in-search to post-processing. The re-projected path
is then re-collision-checked and scored by the identical shared hard validator.

**Fairness / provenance.** Same shared bridge as B3b: goal via CuRobo IK,
collision via the per-request world (A1), scoring via the shared hard validator
(B6). B5 adds the manifold projection of the output path.

**Why B5 is the safety net for the paper.** Per B1's risk posture, at least one
constraint-enforcing adversary must ship. B5 (external OMPL + projection) is the
primary; B3c (a dependency-free CHOMP-with-constraint reimplementation) is the
floor if this proves brittle on the SR5 URDF / per-request world.

**Empirically confirmed characterization (smoke, 2026-07-18).** Post-hoc
projection reaches near-zero alignment violation (1.9 deg) and excellent
continuity (joint step 0.03) *by construction*, but has two inherent, honestly
reported failure modes that B3c (joint optimization) does not: (1) **goal miss**
-- the CuRobo IK goal config often lies in a different manifold basin (IK branch)
than the start-warm-started projected chain, so the trust-region projection
cannot reach it without an IK-branch switch, leaving a terminal pose gap; and
(2) **collision** -- projection is blind to obstacles, so an interior waypoint
projected onto the manifold can penetrate the per-request world. These are
properties of projection-after-unconstrained-search (forced by the binding
block), not bugs; the shared hard validator records them as honest failures.
B3c resolves both by jointly descending smoothness + collision + level in one
optimization, which is why it is the guaranteed floor.
"""

from __future__ import annotations

import time
from typing import Any

from ..methods import MethodSpec, register_method
from . import ompl_bridge


METHOD_NAME = "baseline/ompl_projection"
SOURCE_TYPE = "external_ompl_projection"

_BASE_SOLVE_TIME_SEC = 2.0  # projection is costlier per node than plain RRTC
_MAX_SOLVE_TIME_SEC = 20.0
_PER_GOAL_MIN_SOLVE_SEC = 1.5  # guaranteed slice per IK goal so a 4-goal loop
#                                does not starve the last goals of solve time
_OMPL_RNG_SEED = 1  # deterministic RRT-Connect for reproducible paper numbers
#                     (OMPL rejects seed 0)


def _solve_time_budget(request: dict[str, Any]) -> float:
    metadata = dict(request.get("metadata") or {})
    seed_policy = dict(request.get("seed_policy") or {})
    budget = int(metadata.get("compute_budget_solve_calls") or seed_policy.get("k_generate") or 1)
    timeout = float(seed_policy.get("timeout_sec") or _MAX_SOLVE_TIME_SEC)
    return max(_BASE_SOLVE_TIME_SEC, min(_MAX_SOLVE_TIME_SEC, _BASE_SOLVE_TIME_SEC * max(1, budget), timeout))


def run_ompl_projection(
    request: dict[str, Any],
    config: Any,
    *,
    planner: Any,
    out_dir: Any = None,
    **_context: Any,
) -> dict[str, Any]:
    """B0 runner: RRT-Connect + post-hoc manifold projection (see module docstring)."""
    ompl_bridge.require_ompl()
    if planner is None:
        raise ValueError(f"{METHOD_NAME} requires a live planner instance; got None")

    from ompl import base as ob
    from ompl import geometric as og
    from ompl import util as ou

    # Deterministic RRT-Connect: fixed RNG seed so the baseline produces the same
    # path across repeats (paper reproducibility). Must be set before any planner
    # allocation.
    try:
        ou.RNG.setSeed(_OMPL_RNG_SEED)
    except Exception:
        pass

    normalized = planner._normalize_request(request)
    space, dof = ompl_bridge.make_state_space(planner)
    si = ob.SpaceInformation(space)
    # Search uses only the shared collision validity (plain, writable space);
    # the level constraint is enforced by projecting the output path.
    si.setStateValidityChecker(ompl_bridge.make_validity_checker(planner, dof))
    si.setup()

    start_joint = list(normalized["start_joint"])
    goal_configs = ompl_bridge.solve_goal_joint_configs(planner, normalized)

    solve_time = _solve_time_budget(request)
    started = time.time()
    waypoints: list[list[float]] | None = None
    projected_max_angle: float | None = None
    solver_status = "no_goal_ik" if not goal_configs else "failed"
    failure_reason: str | None = None if goal_configs else "ik_no_solution"

    planner_extra = {
        "planner_family": "rrt_connect",
        "constrained": True,
        "projection": "post_hoc_manifold",
        "projection_note": "ProjectedStateSpace unavailable in this OMPL binding; waypoints Newton-projected onto the level manifold.",
        "solve_time_budget_sec": solve_time,
    }

    if goal_configs:
        start_state = ompl_bridge.make_state(si, start_joint, dof)
        if not si.isValid(start_state):
            return ompl_bridge.build_result_dict(
                planner,
                normalized_request=normalized,
                method_name=METHOD_NAME,
                source_type=SOURCE_TYPE,
                waypoints=None,
                solve_time_sec=time.time() - started,
                solver_status="start_in_collision",
                failure_reason="start_state_invalid",
                planner_extra=planner_extra,
                out_dir=out_dir,
            )
        for goal_config in goal_configs:
            if time.time() - started >= solve_time:
                solver_status = "planner_timeout"
                failure_reason = "ompl_no_path_within_budget"
                break
            goal_state = ompl_bridge.make_state(si, goal_config, dof)
            if not si.isValid(goal_state):
                solver_status, failure_reason = "goal_in_collision", "goal_state_invalid"
                continue
            setup = og.SimpleSetup(si)
            setup.setStartAndGoalStates(start_state, goal_state)
            setup.setPlanner(og.RRTConnect(si))
            # Guarantee each goal a minimum solve slice so a multi-goal IK loop
            # does not starve later goals; still respect the overall budget.
            remaining = max(_PER_GOAL_MIN_SOLVE_SEC, solve_time - (time.time() - started))
            solved = setup.solve(remaining)
            if bool(solved):
                path = setup.getSolutionPath()
                try:
                    path.interpolate(ompl_bridge.action_horizon(planner))
                except Exception:
                    pass
                states = path.getStates()
                raw_waypoints = [[float(s[i]) for i in range(dof)] for s in states]
                # Densify to the fixed horizon by arc length FIRST (so the giant
                # start->goal jump becomes many small even steps), THEN project
                # each evenly-spaced waypoint onto the alignment manifold.  A
                # separate warm-started projection per point keeps continuity, so
                # the output satisfies both the level constraint and the
                # joint-step continuity validator.
                dense = ompl_bridge._resample_path(
                    raw_waypoints, ompl_bridge.action_horizon(planner), force=True
                )
                # Pin only the exact start.  The RRT path already terminates at
                # goal_config continuously, and the trust-region projection keeps
                # every waypoint within a bounded joint-step of its predecessor,
                # so the endpoint reaches the goal pose without a discontinuity.
                # Pinning a foreign IK goal config would recreate the giant jump.
                waypoints, projected_max_angle = ompl_bridge.project_path_onto_level_manifold(
                    planner, dense, normalized, pin_start=start_joint
                )
                solver_status, failure_reason = "success", None
                break
            solver_status = "planner_timeout"
            failure_reason = "ompl_no_path_within_budget"

    if projected_max_angle is not None:
        planner_extra["projected_max_alignment_deg"] = round(float(projected_max_angle), 6)

    return ompl_bridge.build_result_dict(
        planner,
        normalized_request=normalized,
        method_name=METHOD_NAME,
        source_type=SOURCE_TYPE,
        waypoints=waypoints,
        solve_time_sec=time.time() - started,
        solver_status=solver_status,
        failure_reason=failure_reason,
        planner_extra=planner_extra,
        out_dir=out_dir,
    )


register_method(
    MethodSpec(
        name=METHOD_NAME,
        mode="native",
        use_critic=False,
        fallback_to_rule=False,
        fallback_to_native=False,
        external=True,
        runner=run_ompl_projection,
        description=(
            "B5 hard-constraint adversary: OMPL RRT-Connect + post-hoc manifold "
            "projection. The level axis is enforced as a hard manifold constraint "
            "by Newton-projecting every output waypoint onto align_angle=0 (the "
            "ProjectedStateSpace search path is unavailable in this OMPL binding). "
            "Goal via shared IK, collision via the shared world, scored by the "
            "shared hard validator. Primary constraint-enforcing opponent for Table I."
        ),
    )
)
