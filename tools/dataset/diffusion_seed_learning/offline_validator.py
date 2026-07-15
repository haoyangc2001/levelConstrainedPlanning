#!/usr/bin/env python3
"""Offline hard validator for diffusion seed lifecycle datasets."""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src" / "curobo_v2_planner"))

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import JointState as CuJointState  # noqa: E402
from curobo_v2_planner import constraint_utils  # noqa: E402
from curobo_v2_planner.rokae_asset_utils import resolve_robot_config  # noqa: E402


VALIDATOR_SCHEMA_VERSION = "offline_hard_validator.v1"
DEFAULT_MANIFEST = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/"
    "datasets/sr5_phase2_20260713_lifecycle_baseline/manifest.json"
)
DEFAULT_REPORT_OUT = Path("readCaohy/plans/diffusionSeedLearning/validator_report.json")
DEFAULT_RULES_OUT = Path(
    "readCaohy/plans/diffusionSeedLearning/positive_sample_filter_rules.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--rules-out", type=Path, default=DEFAULT_RULES_OUT)
    parser.add_argument("--validated-samples-out", type=Path, default=None)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=Path("resource/config/Level_Test_V2_caohy/robot/xms5_r800_w4g3b4c_v2.yml"),
    )
    parser.add_argument("--alignment-tolerance-deg", type=float, default=None)
    parser.add_argument("--goal-position-tolerance-m", type=float, default=0.02)
    parser.add_argument("--goal-orientation-tolerance-deg", type=float, default=5.0)
    parser.add_argument("--joint-limit-margin-rad", type=float, default=1e-6)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_samples(samples_path: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with samples_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def normalize_quaternion(q: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in q))
    if norm < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in q]


def quaternion_angle_deg(q1: list[float], q2: list[float]) -> float:
    a = normalize_quaternion(q1)
    b = normalize_quaternion(q2)
    dot = abs(sum(x * y for x, y in zip(a, b)))
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(2.0 * math.acos(dot))


class Sr5OfflineValidator:
    """CuRobo-backed FK validator for the frozen SR5 profile."""

    def __init__(self, robot_config: Path, device: str) -> None:
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        robot_cfg = resolve_robot_config(robot_config, auto_generate_spheres=False)
        cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=None,
            collision_cache={"obb": 8},
            use_cuda_graph=False,
        )
        self.planner = MotionPlanner(cfg)
        self.joint_names = list(self.planner.joint_names)
        self.tool_frame = list(self.planner.tool_frames)[0]
        joint_limits = self.planner.kinematics.get_joint_limits()
        self.joint_lower = joint_limits.position[0].detach().to(self.device).reshape(-1)
        self.joint_upper = joint_limits.position[1].detach().to(self.device).reshape(-1)

    def kinematics_fn(self, positions: torch.Tensor) -> SimpleNamespace:
        state = CuJointState.from_position(positions, joint_names=self.joint_names)
        kin_state = self.planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self.tool_frame)
        return SimpleNamespace(
            ee_position=tool_pose.position,
            ee_quaternion=tool_pose.quaternion,
        )

    def terminal_pose(self, positions: torch.Tensor) -> tuple[list[float], list[float]]:
        kin = self.kinematics_fn(positions[-1:].to(self.device))
        pos = kin.ee_position.detach().cpu().reshape(-1).tolist()
        quat = kin.ee_quaternion.detach().cpu().reshape(-1).tolist()
        return [float(v) for v in pos], [float(v) for v in quat]

    def validate_candidate(
        self,
        sample: dict[str, Any],
        args: argparse.Namespace,
    ) -> dict[str, Any]:
        candidate = sample.get("candidate") or {}
        trajectory = candidate.get("trajectory") or {}
        points = trajectory.get("points") or []
        if not points:
            return {
                "status": "skipped_no_trajectory",
                "valid": False,
                "failure_reason": "sample has no candidate trajectory",
            }
        positions = torch.tensor(points, dtype=torch.float32, device=self.device).unsqueeze(0)
        target_pose = sample.get("task", {}).get("target_pose") or []
        start_joint = sample.get("start_state", {}).get("service_start_joint") or points[0]
        tolerance = args.alignment_tolerance_deg
        if tolerance is None:
            tolerance = sample.get("task", {}).get("alignment", {}).get("tolerance_deg") or 3.0

        level_eval = constraint_utils.evaluate_axis_alignment_batched(
            positions,
            self.kinematics_fn,
            float(tolerance),
        )
        continuity = constraint_utils.compute_candidate_continuity_metrics(
            positions,
            start_joint,
            target_pose[3:7],
            self.kinematics_fn,
        )
        lower = self.joint_lower - float(args.joint_limit_margin_rad)
        upper = self.joint_upper + float(args.joint_limit_margin_rad)
        limit_low = positions[0] < lower
        limit_high = positions[0] > upper
        limit_invalid = bool(torch.any(limit_low | limit_high).item())
        violations = []
        if limit_invalid:
            bad_points, bad_joints = torch.where(limit_low | limit_high)
            for point_index, joint_index in zip(bad_points.tolist(), bad_joints.tolist()):
                value = float(positions[0, point_index, joint_index].item())
                violations.append({
                    "point_index": int(point_index),
                    "joint_index": int(joint_index),
                    "joint_name": self.joint_names[int(joint_index)],
                    "value_rad": value,
                    "lower_rad": float(self.joint_lower[int(joint_index)].item()),
                    "upper_rad": float(self.joint_upper[int(joint_index)].item()),
                })

        terminal_position, terminal_quaternion = self.terminal_pose(positions[0])
        target_position = [float(v) for v in target_pose[:3]]
        target_quaternion = [float(v) for v in target_pose[3:7]]
        position_error = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(terminal_position, target_position))
        )
        orientation_error = quaternion_angle_deg(terminal_quaternion, target_quaternion)
        goal_valid = (
            position_error <= float(args.goal_position_tolerance_m)
            and orientation_error <= float(args.goal_orientation_tolerance_deg)
        )

        alignment_profile = level_eval["alignment_angle_map"][0].detach().cpu().tolist()
        max_alignment = float(level_eval["max_alignment_deviation"][0].item())
        mean_alignment = float(level_eval["mean_alignment_deviation"][0].item())
        alignment_valid = bool(level_eval["alignment_valid"][0].item())
        joint_limits_valid = not limit_invalid
        collision = {
            "status": "unchecked",
            "reason": "phase3 initial validator keeps collision replay interface but does not run cuRobo world collision yet",
        }
        valid = bool(alignment_valid and joint_limits_valid and goal_valid)
        source_type = sample.get("source_lineage", {}).get("source_type")
        selected = bool(sample.get("labels", {}).get("selected"))
        planner_status = sample.get("labels", {}).get("planner_status")
        positive_for_diffusion = bool(
            valid
            and selected
            and planner_status == "success"
            and source_type != "rule_raw"
        )

        return {
            "status": "validated",
            "valid": valid,
            "alignment": {
                "valid": alignment_valid,
                "max_deviation_deg": round(max_alignment, 6),
                "mean_deviation_deg": round(mean_alignment, 6),
                "profile_deg": [round(float(v), 6) for v in alignment_profile],
                "tolerance_deg": float(tolerance),
            },
            "joint_limits": {
                "valid": joint_limits_valid,
                "violations": violations,
            },
            "continuity": {
                "start_gap_l2": round(float(continuity["start_joint_gap_l2"][0].item()), 6),
                "joint_step_max_l2": round(float(continuity["joint_step_max_l2"][0].item()), 6),
                "joint_step_mean_l2": round(float(continuity["joint_step_mean_l2"][0].item()), 6),
                "joint_step_max_abs": round(float(continuity["joint_step_max_abs"][0].item()), 6),
                "twist_smoothness_cost": round(float(continuity["twist_smoothness_cost"][0].item()), 6),
            },
            "goal": {
                "valid": bool(goal_valid),
                "position_error_m": round(float(position_error), 6),
                "orientation_error_deg": round(float(orientation_error), 6),
                "position_tolerance_m": float(args.goal_position_tolerance_m),
                "orientation_tolerance_deg": float(args.goal_orientation_tolerance_deg),
                "terminal_position": [round(float(v), 6) for v in terminal_position],
                "terminal_quaternion": [round(float(v), 6) for v in terminal_quaternion],
            },
            "collision": collision,
            "label": {
                "positive_for_diffusion": positive_for_diffusion,
                "positive_for_critic": bool(valid and selected and planner_status == "success"),
                "failure_reason": None if valid else "offline_validator_failed",
            },
        }


def validate_samples(
    samples: list[dict[str, Any]],
    validator: Sr5OfflineValidator,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    validated = []
    for sample in samples:
        item = dict(sample)
        if sample.get("sample_type") == "candidate":
            result = validator.validate_candidate(sample, args)
            item["offline_validator"] = result
            item.setdefault("labels", {})
            item["labels"]["validator_valid"] = bool(result.get("valid"))
            item["labels"]["positive_for_diffusion"] = bool(
                result.get("label", {}).get("positive_for_diffusion")
            )
            item["labels"]["positive_for_critic"] = bool(
                result.get("label", {}).get("positive_for_critic")
            )
            item["labels"]["validator_failure_reason"] = result.get("label", {}).get("failure_reason")
        else:
            item["offline_validator"] = {
                "status": "skipped_request_failure",
                "valid": False,
                "failure_reason": "request_failure sample has no trajectory",
                "collision": {"status": "unchecked"},
            }
            item.setdefault("labels", {})
            item["labels"]["validator_valid"] = False
            item["labels"]["positive_for_diffusion"] = False
            item["labels"]["positive_for_critic"] = False
            item["labels"]["validator_failure_reason"] = "request_failure"
        validated.append(item)
    return validated


def write_jsonl(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_report(
    manifest: dict[str, Any],
    validated_samples: list[dict[str, Any]],
    validated_samples_path: Path,
) -> dict[str, Any]:
    status_counts = collections.Counter(
        sample.get("offline_validator", {}).get("status") for sample in validated_samples
    )
    valid_counts = collections.Counter(
        str(bool(sample.get("offline_validator", {}).get("valid"))) for sample in validated_samples
    )
    source_counts = collections.Counter(
        sample.get("source_lineage", {}).get("source_label") or "none"
        for sample in validated_samples
    )
    positive_diffusion = sum(
        1 for sample in validated_samples
        if bool(sample.get("labels", {}).get("positive_for_diffusion"))
    )
    positive_critic = sum(
        1 for sample in validated_samples
        if bool(sample.get("labels", {}).get("positive_for_critic"))
    )
    candidate_alignment_valid = sum(
        1 for sample in validated_samples
        if sample.get("sample_type") == "candidate"
        and bool(sample.get("offline_validator", {}).get("alignment", {}).get("valid"))
    )
    candidate_goal_valid = sum(
        1 for sample in validated_samples
        if sample.get("sample_type") == "candidate"
        and bool(sample.get("offline_validator", {}).get("goal", {}).get("valid"))
    )
    metric_pairs = (
        ("max_alignment_deviation_deg", ("alignment", "max_deviation_deg")),
        ("mean_alignment_deviation_deg", ("alignment", "mean_deviation_deg")),
        ("start_joint_gap_l2", ("continuity", "start_gap_l2")),
        ("joint_step_max_l2", ("continuity", "joint_step_max_l2")),
        ("joint_step_max_abs", ("continuity", "joint_step_max_abs")),
        ("twist_smoothness_cost", ("continuity", "twist_smoothness_cost")),
        ("position_error_m", ("goal", "position_error_m")),
        ("orientation_error_deg", ("goal", "orientation_error_deg")),
    )
    consistency: dict[str, Any] = {}
    for legacy_key, validator_path in metric_pairs:
        deltas = []
        for sample in validated_samples:
            if sample.get("sample_type") != "candidate":
                continue
            recorded_metrics = sample.get("candidate", {}).get("metrics", {})
            recorded = recorded_metrics.get(legacy_key)
            if recorded is None:
                continue
            current: Any = sample.get("offline_validator", {})
            for key in validator_path:
                if not isinstance(current, dict):
                    current = None
                    break
                current = current.get(key)
            if current is None:
                continue
            deltas.append(abs(float(current) - float(recorded)))
        consistency[legacy_key] = {
            "compared_count": len(deltas),
            "max_abs_delta": round(max(deltas), 9) if deltas else None,
            "mean_abs_delta": round(sum(deltas) / len(deltas), 9) if deltas else None,
        }
    return {
        "schema_version": "validator_report.v1",
        "validator_schema_version": VALIDATOR_SCHEMA_VERSION,
        "dataset_name": manifest.get("dataset_name"),
        "manifest_path": manifest.get("manifest_path"),
        "samples_path": manifest.get("samples_path"),
        "validated_samples_path": str(validated_samples_path),
        "sample_count": len(validated_samples),
        "status_counts": dict(sorted(status_counts.items())),
        "valid_counts": dict(sorted(valid_counts.items())),
        "source_label_counts": dict(sorted(source_counts.items())),
        "candidate_alignment_valid_count": candidate_alignment_valid,
        "candidate_goal_valid_count": candidate_goal_valid,
        "metric_consistency_with_lifecycle": consistency,
        "positive_for_diffusion": positive_diffusion,
        "positive_for_critic": positive_critic,
        "collision_policy": "unchecked_interface_reserved",
    }


def build_rules(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": "positive_sample_filter_rules.v1",
        "last_updated": "2026-07-13",
        "positive_for_diffusion": [
            "sample_type == candidate",
            "planner_status == success",
            "selected == true",
            "source_type != rule_raw",
            "offline_validator.alignment.valid == true",
            "offline_validator.joint_limits.valid == true",
            "offline_validator.goal.valid == true",
            "offline_validator.collision.status in ['unchecked', 'valid']",
        ],
        "positive_for_critic": [
            "sample_type == candidate",
            "planner_status == success",
            "selected == true",
            "offline_validator.valid == true",
        ],
        "thresholds": {
            "alignment_tolerance_deg": args.alignment_tolerance_deg,
            "goal_position_tolerance_m": args.goal_position_tolerance_m,
            "goal_orientation_tolerance_deg": args.goal_orientation_tolerance_deg,
            "joint_limit_margin_rad": args.joint_limit_margin_rad,
        },
        "collision_policy": {
            "phase3": "unchecked",
            "future_required_for_candidate_mode": True,
        },
    }


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    samples_path = Path(manifest["samples_path"])
    samples = load_samples(samples_path)
    validated_samples_path = args.validated_samples_out
    if validated_samples_path is None:
        validated_samples_path = samples_path.with_name("samples_validated.jsonl")

    validator = Sr5OfflineValidator(args.robot_config, args.device)
    validated_samples = validate_samples(samples, validator, args)
    write_jsonl(validated_samples_path, validated_samples)

    report = build_report(manifest, validated_samples, validated_samples_path)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rules = build_rules(args)
    args.rules_out.parent.mkdir(parents=True, exist_ok=True)
    args.rules_out.write_text(
        json.dumps(rules, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
