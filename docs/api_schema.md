# API Schema

## Request

Required top-level fields:

```json
{
  "schema_version": "1.0",
  "request_id": "request_level_001",
  "robot_profile": "sr5",
  "start_joint": [0, 0, 0, 0, 0, 0],
  "target_pose": {
    "position": [0.0, 0.0, 0.0],
    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]
  },
  "alignment": {
    "local_axis": [0.0, 1.0, 0.0],
    "target_world_axis": [0.0, 0.0, -1.0],
    "tolerance_deg": 3.0,
    "strict_level": true
  },
  "seed_policy": {
    "mode": "rule",
    "k_generate": 0,
    "k_accept": 0,
    "fallback_to_rule_seed": true
  }
}
```

Field notes:

- `start_joint`: six SR5 joint positions in radians.
- `target_pose.position`: world-frame target position in meters.
- `target_pose.quaternion_wxyz`: quaternion order is `[w, x, y, z]`.
- `alignment.local_axis`: tool-frame axis, default `tool1 y+`.
- `alignment.target_world_axis`: world-frame target axis, default `z-`.
- `seed_policy.mode`: `off`, `rule`, `diffusion`, or `mixed`. Diffusion generates seed candidates only; final trajectory still comes from CuRobo planning and hard validation.

## Result

Required result fields:

```json
{
  "schema_version": "1.0",
  "request_id": "request_level_001",
  "status": "success",
  "failure_reason": null,
  "selected_trajectory": [[0, 0, 0, 0, 0, 0]],
  "metrics": {},
  "seed_provider_reports": [],
  "candidates": [],
  "artifacts": {}
}
```

Status values:

- `success`
- `failed_precheck`
- `failed_goal`
- `failed_alignment_constraint`
- `failed_planner`
- `failed_internal_error`

Important metrics:

- `metrics.alignment`: tolerance, selected max deviation, candidate deviations.
- `metrics.goal`: terminal position/orientation error.
- `metrics.continuity`: start gap, joint jump, twist smoothness.
- `metrics.joint_limit`: reserved for hard validator details.

Examples:

- `examples/requests/request_level_001.json`
- `examples/requests/request_level_alignment_hard.json`
- `examples/requests/request_level_planner_fail.json`
