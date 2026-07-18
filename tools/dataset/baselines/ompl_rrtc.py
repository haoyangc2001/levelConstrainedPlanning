"""B3b baseline: OMPL RRT-Connect (unconstrained joint-space planner).

The plan's B3b slot is the *unconstrained sampling* floor: a mature, widely
cited sampling planner (RRT-Connect) that solves the collision-free joint-space
motion problem but has **no** notion of the level axis constraint. It is the
honest "what a standard motion planner gives you" opponent -- it will usually
reach the goal and avoid collisions, and its alignment-violation rate is then a
direct measure of how badly an unconstrained planner ignores the level task.

**Fairness / provenance.** The goal is obtained by the *same* CuRobo IK the
internal planner uses (``planner._ik_solve_pose_candidates``); collision
checking uses the *same* per-request world via ``planner._evaluate_collision``
(A1); and the output path is scored by the *same* hard validator as every other
method (B6), via the shared ``ompl_bridge``. The only thing B3b lacks is the
level constraint -- which is exactly the point of the baseline.

Budget: ``metadata.compute_budget_solve_calls`` (the uniform K seam) maps to the
OMPL planning-time budget so B3b runs under the same compute budget as the
internal methods; ``seed_policy.timeout_sec`` is the uniform wall-clock guard.
"""

from __future__ import annotations

import time
from typing import Any

from ..methods import MethodSpec, register_method
from . import ompl_bridge


METHOD_NAME = "baseline/ompl_rrtc"
SOURCE_TYPE = "external_ompl_rrtc"

# Wall-clock seconds per OMPL solve, scaled by the compute budget K so a larger
# K buys proportionally more planning time (mirrors "more solve attempts").
_BASE_SOLVE_TIME_SEC = 0.5
_MAX_SOLVE_TIME_SEC = 10.0


def _solve_time_budget(request: dict[str, Any]) -> float:
    metadata = dict(request.get("metadata") or {})
    seed_policy = dict(request.get("seed_policy") or {})
    budget = int(metadata.get("compute_budget_solve_calls") or seed_policy.get("k_generate") or 1)
    timeout = float(seed_policy.get("timeout_sec") or _MAX_SOLVE_TIME_SEC)
    return max(_BASE_SOLVE_TIME_SEC, min(_MAX_SOLVE_TIME_SEC, _BASE_SOLVE_TIME_SEC * max(1, budget), timeout))


def run_ompl_rrtc(
    request: dict[str, Any],
    config: Any,
    *,
    planner: Any,
    out_dir: Any = None,
    **_context: Any,
) -> dict[str, Any]:
    """B0 runner: unconstrained RRT-Connect in joint space via OMPL."""
    ompl_bridge.require_ompl()
    if planner is None:
        raise ValueError(f"{METHOD_NAME} requires a live planner instance; got None")

    from ompl import base as ob
    from ompl import geometric as og

    normalized = planner._normalize_request(request)
    space, dof = ompl_bridge.make_state_space(planner)
    si = ob.SpaceInformation(space)
    si.setStateValidityChecker(ompl_bridge.make_validity_checker(planner, dof))
    si.setup()

    start_joint = list(normalized["start_joint"])
    goal_configs = ompl_bridge.solve_goal_joint_configs(planner, normalized)

    solve_time = _solve_time_budget(request)
    started = time.time()
    waypoints: list[list[float]] | None = None
    solver_status = "no_goal_ik" if not goal_configs else "failed"
    failure_reason: str | None = None if goal_configs else "ik_no_solution"

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
                planner_extra={"planner_family": "rrt_connect", "constrained": False, "solve_time_budget_sec": solve_time},
                out_dir=out_dir,
            )
        # Try each IK goal; take the first that yields a valid path.
        for goal_config in goal_configs:
            goal_state = ompl_bridge.make_state(si, goal_config, dof)
            if not si.isValid(goal_state):
                solver_status, failure_reason = "goal_in_collision", "goal_state_invalid"
                continue
            setup = og.SimpleSetup(si)
            setup.setStartAndGoalStates(start_state, goal_state)
            setup.setPlanner(og.RRTConnect(si))
            remaining = max(0.05, solve_time - (time.time() - started))
            solved = setup.solve(remaining)
            if bool(solved):
                path = setup.getSolutionPath()
                try:
                    path.interpolate(ompl_bridge.action_horizon(planner))
                except Exception:
                    pass
                states = path.getStates()
                waypoints = [[float(s[i]) for i in range(dof)] for s in states]
                solver_status, failure_reason = "success", None
                break
            solver_status = "planner_timeout"
            failure_reason = "ompl_no_path_within_budget"

    return ompl_bridge.build_result_dict(
        planner,
        normalized_request=normalized,
        method_name=METHOD_NAME,
        source_type=SOURCE_TYPE,
        waypoints=waypoints,
        solve_time_sec=time.time() - started,
        solver_status=solver_status,
        failure_reason=failure_reason,
        planner_extra={"planner_family": "rrt_connect", "constrained": False, "solve_time_budget_sec": solve_time},
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
        runner=run_ompl_rrtc,
        description=(
            "B3b unconstrained floor: OMPL RRT-Connect in joint space. Goal via "
            "shared CuRobo IK, collision via the shared per-request world, scored "
            "by the shared hard validator. No level constraint -- measures how far "
            "an unconstrained sampling planner drifts from the level axis."
        ),
    )
)
