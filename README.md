# SR5 Level Constrained Planning

Lightweight planning-first project extracted from:

```text
/home/caohy/repositories/tashan_Manipulation
```

The first version focuses on SR5 level/end-effector pose constrained trajectory planning:

```text
request JSON/YAML
-> LevelConstrainedPlanner.plan(request)
-> planner/rule/diffusion seed providers
-> CuRobo V2 repair and hard validation
-> result JSON + trajectory artifacts
```

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
reports/                 committed smoke/validation summaries
scripts/                 headless validation entrypoints
```

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
/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning
```
