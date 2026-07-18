# B3b OMPL RRT-Connect Smoke Report

Date: 2026-07-18

## Scope

The B3b baseline uses OMPL 2.0.1 `RRTConnect` in the SR5 joint space. Joint
bounds come from the SR5 URDF, pose goals are converted with the shared CuRobo
IK path, and state validity uses the same A1 world-collision query as internal
methods. The OMPL wheel reports `BSD-3-Clause` in its installed metadata.

## Validation

- Environment: `CuroboV2`, NVIDIA A100-PCIE-40GB.
- Request: `sample_00000_easy_mixed_none`, one solve-call budget.
- OMPL planning: solution found in `0.383 s`.
- Collision replay: checked and safe, minimum distance `0.005 m`.
- Retiming: uniform URDF velocity-limit timestep, dimensioned metrics enabled.
- Maximum velocity: `1.570796 rad/s`.
- Maximum path alignment deviation: `167.080246 deg` at a `15 deg` tolerance.
- Final status: `failed_hard_validation` with
  `failed_alignment_constraint`.

This is the expected behavior for an unconstrained sampling baseline: it finds
a collision-free goal-reaching joint path but does not preserve the level-axis
constraint. The shared hard validator rejects it, while the benchmark retains
the true violation and motion-quality metrics.

## Automated Checks

`24` project tests pass with third-party pytest plugin autoload disabled.
