# B3c CHOMP-Style Constraint Optimization Smoke Report

Date: 2026-07-18

## Implementation

`baseline/chomp_constraint` is the dependency-free classic constrained
trajectory-optimization floor selected by B1. It fixes the SR5 start and goal
joint configurations and optimizes the interior waypoints with:

- velocity and acceleration smoothness;
- an all-waypoint level-axis violation penalty;
- the shared differentiable A1 world-collision activation cost;
- URDF joint-limit projection after every update.

The compute budget controls optimizer iterations. Final success is assigned
only by the shared hard validator.

## A100 Smoke Results

Easy request, `K=2`, `15 deg` tolerance:

- final status: `success`;
- runtime: `3.79 s`;
- maximum alignment deviation: `13.438 deg`;
- collision: checked and safe;
- maximum velocity: `1.570796 rad/s`;
- motion time after external-path retiming: `1.974 s`.

Hard request, `K=4`, `3 deg` tolerance:

- final status: `failed_hard_validation`;
- alignment and collision checks both failed.

The hard failure is retained as benchmark evidence rather than converted into
a private baseline success. This verifies the common acceptance contract.

## Known CuRobo V2 Constraint

The installed CUDA self-collision backward returns a shape mismatch for its
`[B,T,1]` output. The optimizer therefore differentiates the world-collision
cost, matching the A1 acceptance metric; final hard validation remains the
authoritative gate.

## Automated Checks

`27` project tests pass with third-party pytest plugin autoload disabled.
