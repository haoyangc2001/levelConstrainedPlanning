# D6 Real-Robot Scope Decision

- task: `D6`
- decision: `prior_validation_only`
- new hardware experiment: `not_executed`

## Current Repository Evidence

`level_planner_ros/planner_node.py` is a planning ingress only. It exposes one `std_srvs/Trigger` service, calls `LevelConstrainedPlanner.plan`, writes result files, and returns the result path. An AST audit found `create_service=1`, `create_publisher=0`, `create_client=0`, `publish=0`, and `send_goal=0`.

`docs/guides/ros_adapter.md` states that motion, RViz, vision, scanner, gripper, and state-machine nodes are outside the adapter. `docs/guides/runtime.md` also frames runtime requests as validation jobs, not direct execution commands.

## Old Project Evidence

The old-project safety audit can be run against:

- `/home/caohy/repositories/tashan_Manipulation/resource/config/Level_Test_V2_caohy/start.launch.yaml`
- `/home/caohy/repositories/tashan_Manipulation/src/curobo_v2_planner/curobo_v2_planner/main.py`

The default safety guards pass, but `/home/caohy/repositories/tashan_Manipulation/readCaohy/plans/diffusionSeedLearning/single_trajectory_smoke_report.json` has status `not_executed_template_and_safety_audit_only`, and `single_trajectory_smoke.executed` is `false`.

## Paper Implication

E4 should be written as prior validation or engineering background only. It must not be presented as a first-party real-robot result from this repository.

Disallowed claims:

- This paper executed a new `20/20` or `18/20` real-robot trial.
- This repository contains a complete SR5 trajectory execution bridge.
- Learned diffusion candidates were executed on the real robot.

Before any new hardware claim, implement an explicit low-speed trajectory execution bridge or action client, add a human-confirmed safety checklist, preview one manually selected trajectory, and record controller feedback, final pose tolerance, full-path level tolerance, collision status, and joint tracking error.
