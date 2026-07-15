# Optional ROS Adapter

The ROS adapter is intentionally thin:

```text
std_srvs/Trigger plan_default
-> load request JSON/YAML
-> LevelConstrainedPlanner.plan(request)
-> write result files
-> return status and result_json path
```

It does not start motion, HTTP, RViz, vision, scanner, gripper, or state machine nodes.

Smoke check without planner initialization:

```bash
source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
python -m level_planner_ros.planner_node --check
```

Launch shape:

```bash
source /home/caohy/repositories/tashan_Manipulation/scripts/activate_curobo_v2_conda_env.sh
colcon build --symlink-install --packages-select level_planner_ros
source install/setup.bash
ros2 launch level_planner_ros level_planner.launch.py \
  config:=configs/sr5_level.yaml \
  request:=examples/requests/request_level_001.json \
  out_dir:=runs/ros_adapter
```
