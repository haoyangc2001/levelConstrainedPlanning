# Environment

Current target environment:

```text
OS: Ubuntu 22.04
Python: /pub/data/caohy/miniconda/envs/CuroboV2
ROS: Humble, optional adapter only
CuRobo: V2 package with curobo.motion_planner, MotionPlannerCfg, GoalToolPose, ToolPoseCriteria
GPU: NVIDIA A100 / CUDA driver capability 12.6
```

Default validation is headless. RViz is not required for this project.

## Activate

```bash
cd /home/caohy/repositories/levelConstrainedPlanning
source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
```

The activation script currently sets `CUDA_VISIBLE_DEVICES=2` by default on this machine.

## Checks

```bash
python - <<'PY'
import torch
import rclpy
import curobo
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
print("rclpy", rclpy.__file__)
print("curobo", curobo.__file__)
PY

python tools/check_assets.py --config configs/sr5_level.yaml
python tools/check_artifacts.py --strict
```

No display server is needed. Do not use RViz as an acceptance condition for this lightweight project.
