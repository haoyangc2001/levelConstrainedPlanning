#!/usr/bin/env python3
"""Audit real-robot safety defaults for diffusion candidate mode."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LAUNCH = REPO_ROOT / "resource/config/Level_Test_V2_caohy/start.launch.yaml"
DEFAULT_MAIN = REPO_ROOT / "src/curobo_v2_planner/curobo_v2_planner/main.py"
DEFAULT_OUT = REPO_ROOT / "readCaohy/plans/diffusionSeedLearning/single_trajectory_smoke_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", type=Path, default=DEFAULT_LAUNCH)
    parser.add_argument("--planner-main", type=Path, default=DEFAULT_MAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    launch_data = yaml.safe_load(args.launch.read_text(encoding="utf-8")) or {}
    launch = launch_data.get("launch", {})
    launch_args = launch.get("launch_arguments", {})
    nodes = launch.get("nodes", {})
    planner_cfg = nodes.get("curobo_v2_planner", {})
    planner_source = args.planner_main.read_text(encoding="utf-8")

    checks = [
        {
            "id": "default_diffusion_seed_mode_off",
            "passed": launch_args.get("diffusion_seed_mode", {}).get("default") == "off",
            "evidence": launch_args.get("diffusion_seed_mode", {}).get("default"),
        },
        {
            "id": "default_inference_node_disabled",
            "passed": launch_args.get("enable_diffusion_seed_inference", {}).get("default") == "false",
            "evidence": launch_args.get("enable_diffusion_seed_inference", {}).get("default"),
        },
        {
            "id": "default_real_robot_disabled",
            "passed": launch_args.get("use_real_robot", {}).get("default") == "false",
            "evidence": launch_args.get("use_real_robot", {}).get("default"),
        },
        {
            "id": "real_robot_candidate_requires_explicit_flag",
            "passed": launch_args.get("diffusion_seed_allow_real_robot_candidate", {}).get("default") == "false",
            "evidence": launch_args.get("diffusion_seed_allow_real_robot_candidate", {}).get("default"),
        },
        {
            "id": "planner_receives_use_real_robot",
            "passed": planner_cfg.get("use_real_robot", {}).get("launch_arg") == "use_real_robot",
            "evidence": planner_cfg.get("use_real_robot"),
        },
        {
            "id": "planner_receives_real_robot_candidate_flag",
            "passed": planner_cfg.get("diffusion_seed_allow_real_robot_candidate", {}).get("launch_arg") == "diffusion_seed_allow_real_robot_candidate",
            "evidence": planner_cfg.get("diffusion_seed_allow_real_robot_candidate"),
        },
        {
            "id": "planner_blocks_candidate_on_real_robot",
            "passed": "real_robot_candidate_blocked" in planner_source
            and "allow_real_robot_candidate" in planner_source,
            "evidence": "real_robot_candidate_blocked guard in curobo_v2_planner main.py",
        },
        {
            "id": "fallback_to_rule_seed_default_true",
            "passed": launch_args.get("diffusion_seed_fallback_to_rule_seed", {}).get("default") == "true",
            "evidence": launch_args.get("diffusion_seed_fallback_to_rule_seed", {}).get("default"),
        },
    ]
    report = {
        "schema_version": "diffusion_seed_learning.phase9.single_trajectory_smoke_report.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "not_executed_template_and_safety_audit_only",
        "reason": "Headless environment and task scope only require pre-real-robot safety closure; no real robot trajectory was executed.",
        "audit": {
            "launch": str(args.launch),
            "planner_main": str(args.planner_main),
            "passed": all(item["passed"] for item in checks),
            "checks": checks,
        },
        "required_before_real_smoke": [
            "Run candidate mode only with use_real_robot=false until fresh benchmark report passes.",
            "Generate plan-only lifecycle for exactly one manually selected trajectory.",
            "Verify selected_source_label, alignment, goal, joint jump, joint limit and collision status in lifecycle.",
            "Preview the exact trajectory in simulation/preview publisher; RViz is optional on headless server but a visual check is required before hardware.",
            "Confirm robot_stop / emergency_stop path is available.",
            "Execute at low speed only after explicit human confirmation; do not run the 20-point batch automatically."
        ],
        "single_trajectory_smoke": {
            "executed": False,
            "allowed_batch_size": 1,
            "auto_batch_execution_allowed": False,
            "diffusion_candidate_default": "blocked_on_real_robot",
            "fallback_required": True,
            "final_joint_feedback_error": None
        }
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
