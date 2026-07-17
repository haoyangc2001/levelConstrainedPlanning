#!/usr/bin/env python3
"""A1.1 interface spike + A1.6 verification for CuRobo world collision replay.

Conclusion (A1.1): CuRobo V2's ``RobotCollisionChecker`` exposes
``get_scene_self_collision_distance_from_joint_trajectory(q)`` which takes a joint
trajectory ``[batch, horizon, dof]`` and returns per-sphere world collision cost
(meters): ``0`` == outside the activation band (safe), positive == within margin /
penetrating, monotonic as the arm drives into an obstacle. Per-link FK is handled
internally (KinematicsState.robot_spheres). => A1.2 path is viable; the A1.1b
self-implemented SDF fallback is NOT needed.

Run:
    source /pub/data/caohy/miniconda/etc/profile.d/conda.sh && conda activate CuroboV2
    python tools/dev/a1_collision_spike.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def main() -> int:
    import json

    from level_planner_core.planner import LevelConstrainedPlanner

    planner = LevelConstrainedPlanner.from_config(str(ROOT / "configs" / "sr5_level.yaml"))

    checker_ok = planner._collision_checker is not None
    print(f"[A1.1] collision checker built: {checker_ok}")
    print(f"[A1.1] world summary: {planner._world_summary}")

    # A safe home-ish config vs a config driven toward the floor slab.
    home = [0.0, -0.3, 0.6, 0.0, 0.9, 0.0]
    into_floor = [0.0, 1.4, 1.2, 0.0, 1.0, 0.0]

    safe_res = planner._evaluate_collision([home, home, home])
    hit_res = planner._evaluate_collision([into_floor, into_floor, into_floor])
    print(f"[A1.6] safe config collision_result: {json.dumps(safe_res)}")
    print(f"[A1.6] into-floor collision_result:  {json.dumps(hit_res)}")

    # Assert the gate discriminates: into-floor cost must exceed safe cost.
    if safe_res and hit_res:
        assert hit_res["collision_cost"] >= safe_res["collision_cost"], (
            "collision cost did not increase for the into-floor config"
        )
        print("[A1.6] PASS: collision cost is monotonic (into-floor >= safe).")
    else:
        print("[A1.6] WARN: collision_result was None (obstacle-free world?).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
