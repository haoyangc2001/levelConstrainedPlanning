# Documentation Index

This directory keeps project documentation grouped by purpose instead of mixing all Markdown files in one flat folder.

The project mainline is the closed-loop learning-optimization system:

```text
data generation -> model learning -> optimization validation -> failure fallback -> data update
```

Use `guides/project_mainline.md` as the first document for understanding the intended direction. The other documents should be read as parts of that loop, not as unrelated utilities.

## Guides

- `guides/project_mainline.md`: top-level closed-loop project direction and engineering contract.
- `guides/environment.md`: machine, conda, ROS, CuRobo, and headless environment for offline generation and online validation.
- `guides/runtime.md`: online-loop behavior, CLI/Python entrypoints, learned-seed validation, and rule fallback.
- `guides/dataset_training.md`: offline-loop dataset export, artifact pointers, diffusion/critic training, and data update workflow.
- `guides/ros_adapter.md`: optional online task ingress around the same planner core.

## Reference

- `reference/api_schema.md`: request/result schema used by the pure planner core, CLI, dataset records, and feedback loop.

## Design

- `design/机械臂带末端位姿约束的轨迹优化.md`: original level/end-effector pose constrained trajectory optimization design.
- `design/末端约束扩散学习模型设计.md`: original diffusion seed learning model design.

## Reports

- `reports/source_phase10_training_report.md`: source-project phase 10 SR5 mature offline model report and artifact summary.
