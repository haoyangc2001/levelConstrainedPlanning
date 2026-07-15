# Documentation Index

This directory keeps project documentation grouped by purpose instead of mixing all Markdown files in one flat folder.

The project mainline is the closed-loop evolution system:

```text
data generation -> model learning -> optimization validation -> failure fallback -> data update
```

Use `guides/project_mainline.md` as the first document for understanding the intended direction.

## Guides

- `guides/project_mainline.md`: top-level closed-loop project direction and engineering contract.
- `guides/environment.md`: machine, conda, ROS, CuRobo, and headless validation environment.
- `guides/runtime.md`: planner runtime behavior and CLI/Python entrypoints.
- `guides/dataset_training.md`: dataset export, artifact pointers, and diffusion/critic training workflow.
- `guides/ros_adapter.md`: optional ROS adapter usage and boundaries.

## Reference

- `reference/api_schema.md`: request/result schema used by the pure planner core and CLI.

## Design

- `design/机械臂带末端位姿约束的轨迹优化.md`: original level/end-effector pose constrained trajectory optimization design.
- `design/末端约束扩散学习模型设计.md`: original diffusion seed learning model design.

## Reports

- `reports/source_phase10_training_report.md`: source-project phase 10 SR5 mature offline model report and artifact summary.
