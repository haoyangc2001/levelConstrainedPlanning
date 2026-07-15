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

## Main Entrypoint

Planned CLI shape:

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

## Artifacts

Large datasets and checkpoints are not stored in this repository. Current model/data pointers live in:

```text
artifacts/current_artifacts.json
```

The actual files remain under:

```text
/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning
```

