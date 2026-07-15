# SR5 Level Constrained Planning

Lightweight closed-loop constrained planning and seed-learning project extracted from:

```text
/home/caohy/repositories/tashan_Manipulation
```

The first version focuses on SR5 level/end-effector pose constrained trajectory planning as the executable core of a learning-optimization loop:

```text
online/offline task JSON/YAML
-> rule or learned trajectory seed providers
-> CuRobo V2 repair and hard validation
-> success/failure artifacts and metrics
-> dataset update and next model training cycle
```

## Project Mainline

The project mainline is the closed-loop evolution system:

```text
data generation -> model learning -> optimization validation -> failure fallback -> data update
```

In Chinese design terms: `数据生成 - 模型学习 - 优化验收 - 失败回退 - 数据更新`.

The planner is the executable core of that loop. Rule-based seed families and CuRobo validation generate reliable training/evaluation data; diffusion and critic models learn better seed distributions from that data; every learned candidate still returns to CuRobo repair, hard validation, fallback handling, and dataset update.

The intended system has two coupled paths:

- Offline path: sample task scenes, build rule seeds, run CuRobo optimization, collect successful trajectories and failed seeds, then train diffusion/critic models.
- Online path: accept a live task, try learned trajectory seeds first, run CuRobo repair and constraint validation, fall back to rule seed search on failure, then feed all outcomes back to the dataset.

## Boundary

This repository is not the full robot product stack. The core package must not depend on:

- `motion`
- `external_comm`
- state machine YAML
- RViz
- camera/vision/pointcloud/gripper/tactile modules

ROS support, if enabled, is a thin optional adapter around the core planner.

## Directory Layout

```text
level_planner_core/      pure Python CuRobo planning core
level_planner/           user-facing CLI package
level_planner_ros/       optional ROS adapter
configs/                 SR5 robot/world/planner config
examples/requests/       success, alignment-hard, planner-fail fixtures
tools/                   asset checks, dataset export, diffusion learning tools
artifacts/               pointer JSON only; no large data/checkpoints
docs/                   design notes, user guides, API reference, reports
reports/                 committed smoke/validation summaries
scripts/                 headless validation entrypoints
```

## Documentation

The Markdown documents under `docs/` are split by role:

- `docs/guides/`: environment setup, runtime usage, dataset/training workflow, and optional ROS adapter usage.
- `docs/reference/`: stable request/result schema and interface reference.
- `docs/design/`: original design notes for constrained trajectory optimization and diffusion seed learning.
- `docs/reports/`: phase or source-project reports that record completed experiments and model status.

Start from `docs/guides/project_mainline.md` when checking whether a new tool or model change follows the intended project direction.

## Main Entrypoint

CLI:

```bash
python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_001.json \
  --out runs/request_level_001
```

Python API shape:

```python
from level_planner_core import LevelConstrainedPlanner

planner = LevelConstrainedPlanner.from_config("configs/sr5_level.yaml")
result = planner.plan(request)
```

This entrypoint is the online-loop validation kernel. Its outputs should be treated as data records as well as planner results: selected trajectory, candidate reports, failure reason, hard-validation metrics, and artifact paths are all useful for the next offline dataset update.

## Smoke Validation

Use the project environment first:

```bash
source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
```

Run the full headless matrix:

```bash
python scripts/headless_smoke.py
```

The committed result is:

```text
reports/headless_validation_matrix.json
```

It covers static checks, one CLI success, five batch request variants, phase10 diffusion seed smoke, and the optional ROS adapter check.

## Artifacts

Large datasets and checkpoints are not stored in this repository. Current model/data pointers live in:

```text
artifacts/current_artifacts.json
```

The actual files remain under:

```text
/pub/data/caohy/levelConstrainedPlanning
```

Legacy phase10 artifacts from `tashan_Manipulation` are still recorded in `artifacts/current_artifacts.json` as rollback baselines. The active standalone dataset and Phase 8 checkpoints are under `/pub/data/caohy/levelConstrainedPlanning`.

## Closed-Loop Smoke

Run the first closed-loop baseline smoke without RViz:

```bash
source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
PYTHONPATH=$PWD python scripts/closed_loop_smoke.py \
  --config configs/sr5_level.yaml \
  --device cuda:0
```

The smoke covers `rule_only`, `diffusion_shadow`, `diffusion_candidate`, and `mixed_fallback`, then exports and validates candidate-level dataset rows.
