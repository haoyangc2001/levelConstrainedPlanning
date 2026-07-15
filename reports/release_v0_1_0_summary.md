# Release v0.1.0 Level Planner Core

## Scope

This release freezes the first lightweight SR5 level constrained planning project.

Included:

- Pure Python CuRobo-backed planning core.
- CLI entrypoint: `python -m level_planner.cli plan`.
- Optional ROS adapter: `level_planner_ros`.
- SR5 robot/world assets only.
- Phase10 diffusion seed artifact pointers and smoke tooling.
- Headless validation matrix.

Excluded:

- `motion` state machines.
- `external_comm` HTTP task entry.
- RViz panels and visual helper stack.
- camera, vision, pointcloud, gripper, tactile, calibration, scanner, HMI modules.
- large datasets, lifecycle logs, benchmark bundles, and checkpoint files.

## Baseline

- source repo: `/home/caohy/repositories/tashan_Manipulation`
- source repo commit: `b1db865a7f679cce578f66013e53f063f22dac30`
- standalone repo: `/home/caohy/repositories/levelConstrainedPlanning`
- artifact pointer schema: `standalone_level_planning.artifacts.v1`
- source pointer sha256: `c11fb62a2b2b5b39537b23469da0593abc197d2d4136238e3a27b24598709134`

## Artifacts

- dataset: `sr5_phase10_20260715_baseline_jitter_train`
- diffusion: `sr5_phase10_mature_diffusion_20260715`
- critic: `sr5_phase10_success_critic_20260715`
- public data root: `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning`

## Validation

Latest committed validation:

```text
reports/headless_validation_matrix.json
```

Summary:

- overall: `success`
- CLI single: `success`
- batch statuses: `success`, `failed_alignment_constraint`, `failed_planner`
- diffusion seed smoke valid ratio: `1.0`
- ROS adapter check: passed

Run again:

```bash
scripts/smoke_test.sh
```

## Extension Rules

- Core must not import ROS, `motion`, `external_comm`, state machine YAML, or product adapters.
- New adapters must depend on `level_planner_core`; core must not depend on adapters.
- Large data, checkpoints, generated samples, and benchmark bundles stay under `/pub/data/caohy` or an external artifact store.
- Real robot usage must first pass plan-only, dry-run, and a safety checklist.
- Diffusion output remains a seed source only; CuRobo repair and hard validation remain mandatory.

