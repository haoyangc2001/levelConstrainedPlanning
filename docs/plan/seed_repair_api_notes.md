# CuRobo Seed Repair API Notes

This note freezes the Phase 0 conclusion for feeding external trajectory seeds into CuRobo repair.

## Current Standalone State

`level_planner_core/planner.py` currently calls `MotionPlanner.plan_pose(...)` only. That API generates its own IK/graph/trajopt seeds and does not expose an external `seed_traj` argument:

```text
MotionPlanner.plan_pose(goal_tool_poses, current_state, use_implicit_goal=True, max_attempts=5, enable_graph_attempt=1)
```

So Phase 4 cannot inject rule or diffusion trajectories through `plan_pose` directly.

## Confirmed CuRobo Route

The installed CuRobo V2 package exposes `TrajOptSolver.solve_pose(...)` with explicit seed inputs:

```text
TrajOptSolver.solve_pose(
  goal_tool_poses,
  current_state,
  seed_config=None,
  seed_traj=None,
  return_seeds=1,
  num_seeds=None,
  dt=None,
  use_implicit_goal=False,
  finetune_attempts=1,
  goal_state=None,
  initial_iters=None,
  time_optimal_iters=None,
  finetune_iters=None,
  finetune_dt_scale=0.55
)
```

The seed manager contract expects `seed_traj` as:

```text
[batch, n, action_horizon, dof]
```

For the first SR5 single-request path this becomes:

```text
[1, K, action_horizon, 6]
```

The source project already uses the same path:

```text
self._planner.trajopt_solver.solve_pose(
  goal,
  current_state,
  seed_traj=prepared_seed_traj,
  use_implicit_goal=True,
  return_seeds=1,
  finetune_attempts=1
)
```

Relevant source snapshot references:

- `migration/source_snapshot/curobo_v2_planner/main.py:1616`: `_prepare_seed_traj_for_trajopt`
- `migration/source_snapshot/curobo_v2_planner/main.py:2492`: `_solve_alignment_seed_trajopt`
- installed CuRobo source: `curobo/_src/solver/solver_trajopt.py:685`
- installed CuRobo seed manager: `curobo/_src/solver/manager_seed.py:142`

## Required Adapter Shape

Phase 4 should introduce a small standalone repair adapter, not spread direct internal CuRobo calls through the planner:

```text
SeedCandidate [T, DOF]
  -> resample to planner.trajopt_solver.action_horizon
  -> pack as [1, 1, H, DOF] or [1, K, H, DOF]
  -> trajopt_solver.solve_pose(..., seed_traj=packed)
  -> split returned trajectories
  -> CandidateRecord optimizer_result + validator_metrics
```

The adapter must preserve:

- `candidate_id`
- `source_type`
- `source_label`
- `provider`
- model/checkpoint or rule family lineage
- raw seed trajectory summary
- repaired trajectory summary
- optimizer status and solve time
- failure stage and failure reason

## Phase 0 Live Spike

Two headless spikes were run in the `CuroboV2` conda environment with `CUDA_VISIBLE_DEVICES=2`, `use_cuda_graph=False` and `warmup_iterations=0`.

The constant-start manual seed proved API admission and result extraction:

```text
request_id: request_level_001
seed_shape: [1, 1, 16, 6]
success_any: false
interpolated_position_shape: [1, 1, 21, 6]
report: runs/phase0_seed_repair_spike/report.json
```

The native-plan-as-external-seed spike proved the successful repair path:

```text
request_id: request_level_001
native_success: true
external_seed_shape: [1, 1, 16, 6]
repair_success: true
repaired_interpolated_position_shape: [1, 1, 121, 6]
report: runs/phase0_seed_repair_spike/native_seed_report.json
```

The `runs/` directory is git-ignored; the reports are local execution evidence and should not be committed.

## G0 Feasibility Gate Result

Status: `passed`.

What is already confirmed:

- Installed CuRobo V2 imports successfully in `/pub/data/caohy/miniconda/envs/CuroboV2`.
- `MotionPlanner.plan_pose` does not accept external trajectory seeds.
- `TrajOptSolver.solve_pose` does accept `seed_traj`.
- Source-project V2 code successfully targets the same internal solver route.
- Seed tensor shape and padding behavior are confirmed from installed CuRobo code.
- A minimal manual seed reaches `TrajOptSolverResult` and returns an interpolated trajectory even when the seed itself does not solve.
- A successful external seed repair path is confirmed by feeding a native planned trajectory back through `seed_traj`.

What remains for Phase 4 implementation:

- Package the spike logic into a tested `RepairAdapter`.
- Verify per-candidate lineage survives batching when `K > 1`.
- Verify `js_solution.position` extraction in addition to `get_interpolated_plan().position`.

## Fallback Route If Direct Repair Breaks

If a CuRobo update breaks direct `trajopt_solver.solve_pose(seed_traj=...)`, use this fallback sequence:

1. Keep rule/diffusion seed generation and precheck unchanged.
2. Use external seed terminal states as `seed_config` or goal-state hints when possible.
3. Fall back to `MotionPlanner.plan_pose` native candidates for final output.
4. Continue recording failed external repair attempts as negative critic samples.
5. Keep learned seed mode in `shadow` until a repair path is restored.

This fallback preserves the data loop but weakens the online learned-first planner until repair injection is restored.
