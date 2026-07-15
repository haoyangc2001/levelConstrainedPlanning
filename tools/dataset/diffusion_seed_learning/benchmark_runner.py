#!/usr/bin/env python3
"""Offline benchmark runner for diffusion seed learning reports.

This phase-8 runner creates a stable benchmark manifest and computes comparable
metrics from validated lifecycle samples. Full ROS/candidate execution can reuse
the same manifest paths and report schema after larger task runs are available.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN_DIR = REPO_ROOT / "readCaohy/plans/diffusionSeedLearning"
DEFAULT_PUBLIC_ROOT = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/benchmarks/"
    "sr5_phase8_benchmark_v1"
)
DEFAULT_VALIDATED_SAMPLES = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/"
    "datasets/sr5_phase2_20260713_lifecycle_baseline/samples_validated.jsonl"
)
DEFAULT_MANIFEST_OUT = DEFAULT_PLAN_DIR / "benchmark_manifest.json"
DEFAULT_REPORT_TEMPLATE = DEFAULT_PLAN_DIR / "benchmark_report_template.md"
DEFAULT_SUMMARY_OUT = DEFAULT_PUBLIC_ROOT / "benchmark_smoke_summary.json"
DEFAULT_SUMMARY_MD = DEFAULT_PLAN_DIR / "benchmark_smoke_report.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--validated-samples", type=Path, default=DEFAULT_VALIDATED_SAMPLES)
    parser.add_argument("--manifest-out", type=Path, default=DEFAULT_MANIFEST_OUT)
    parser.add_argument("--template-out", type=Path, default=DEFAULT_REPORT_TEMPLATE)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--summary-md", type=Path, default=DEFAULT_SUMMARY_MD)
    parser.add_argument("--random-seed", type=int, default=20260713)
    parser.add_argument("--random-task-count", type=int, default=100)
    parser.add_argument("--primitive-task-count", type=int, default=100)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        value = float(value)
        return value if math.isfinite(value) else default
    except Exception:
        return default


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    values = sorted(float(v) for v in values)
    if len(values) == 1:
        return values[0]
    position = (len(values) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return values[lower]
    ratio = position - lower
    return values[lower] * (1.0 - ratio) + values[upper] * ratio


def build_pose_task(task_id: str, rng: random.Random, obstacle_mode: str) -> dict[str, Any]:
    x = rng.uniform(0.22, 0.48)
    y = rng.uniform(-0.42, 0.28)
    z = rng.uniform(0.36, 0.68)
    yaw = rng.uniform(-math.pi, math.pi)
    qw = math.cos(yaw / 2.0) * math.sqrt(0.5)
    qx = -math.sqrt(0.5)
    qy = 0.0
    qz = math.sin(yaw / 2.0) * math.sqrt(0.5)
    return {
        "task_id": task_id,
        "robot_profile": "sr5",
        "target_pose": [
            round(x, 6),
            round(y, 6),
            round(z, 6),
            round(qw, 6),
            round(qx, 6),
            round(qy, 6),
            round(qz, 6),
        ],
        "alignment": {
            "tool_axis": "y+",
            "world_axis": "z-",
            "tolerance_deg": 15.0
        },
        "obstacle_mode": obstacle_mode,
        "speed_scale": 0.5
    }


def write_task_sets(public_root: Path, random_seed: int, random_count: int, primitive_count: int) -> dict[str, str]:
    public_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(int(random_seed))
    random_tasks = [
        build_pose_task(f"sr5_random_no_obstacle_{index:03d}", rng, "none")
        for index in range(int(random_count))
    ]
    primitive_tasks = [
        build_pose_task(f"sr5_primitive_obstacle_{index:03d}", rng, "abs_rel_autosave_cuboids")
        for index in range(int(primitive_count))
    ]
    random_path = public_root / "sr5_random_no_obstacle_100_tasks.json"
    primitive_path = public_root / "sr5_primitive_obstacle_100_tasks.json"
    random_path.write_text(json.dumps({
        "schema_version": "diffusion_seed_benchmark_tasks.v1",
        "task_set": "sr5_random_no_obstacle_100",
        "random_seed": int(random_seed),
        "task_count": len(random_tasks),
        "tasks": random_tasks,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    primitive_path.write_text(json.dumps({
        "schema_version": "diffusion_seed_benchmark_tasks.v1",
        "task_set": "sr5_primitive_obstacle_100",
        "random_seed": int(random_seed),
        "task_count": len(primitive_tasks),
        "tasks": primitive_tasks,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {
        "random_no_obstacle_100": str(random_path),
        "primitive_obstacle_100": str(primitive_path),
    }


def write_manifest(args: argparse.Namespace, task_paths: dict[str, str], available_request_count: int) -> dict[str, Any]:
    manifest = {
        "schema_version": "diffusion_seed_benchmark_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark_name": "sr5_phase8_benchmark_v1",
        "public_root": str(args.public_root),
        "validated_samples": str(args.validated_samples),
        "robot_profile": "sr5",
        "constraint": "tool1 y+ -> world z-",
        "task_sets": [
            {
                "id": "level_test_v2_sr5_20_regression",
                "kind": "lifecycle_or_ros_replay",
                "project": "Level_Test_V2_caohy",
                "planned_task_count": 20,
                "available_validated_request_count": int(available_request_count),
                "source": "resource/config/Level_Test_V2_caohy/files.main_flow=main_sr5_20.yaml",
            },
            {
                "id": "sr5_random_no_obstacle_100",
                "kind": "deterministic_pose_task_list",
                "planned_task_count": 100,
                "task_file": task_paths["random_no_obstacle_100"],
                "random_seed": int(args.random_seed),
            },
            {
                "id": "sr5_primitive_obstacle_100",
                "kind": "deterministic_pose_task_list",
                "planned_task_count": 100,
                "task_file": task_paths["primitive_obstacle_100"],
                "random_seed": int(args.random_seed),
                "obstacle_source": "resource/config/Level_Test_V2_caohy/obstacles/abs.autosave.json + rel.autosave.json",
            },
        ],
        "metrics": [
            "success_at_k",
            "fixed_budget_success",
            "selected_success",
            "failure_distribution",
            "source_label_distribution",
            "latency_p50_p95_or_proxy",
            "alignment_max_mean",
            "joint_jump",
            "collision_status",
            "joint_limit_status",
        ],
        "data_policy": "task lists and full benchmark outputs live under /pub/data/caohy; repo stores manifest and summaries only",
    }
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def summarize_validated_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = [sample for sample in samples if sample.get("sample_type") == "candidate"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in candidates:
        key = sample.get("source", {}).get("move_file") or str(sample.get("source", {}).get("plan_request_index"))
        groups[str(key)].append(sample)
    selected_success = 0
    success_at_k = 0
    failure_counter = Counter()
    source_counter = Counter()
    selected_source_counter = Counter()
    alignment_max_values = []
    alignment_mean_values = []
    jump_values = []
    latency_proxy_values = []
    joint_limit_valid_count = 0
    goal_valid_count = 0
    for group in groups.values():
        has_positive = any(bool(sample.get("labels", {}).get("positive_for_critic")) for sample in group)
        success_at_k += int(has_positive)
        for sample in group:
            labels = sample.get("labels") or {}
            candidate = sample.get("candidate") or {}
            metrics = candidate.get("metrics") or {}
            source_label = candidate.get("source_label") or "unknown"
            source_counter[str(source_label)] += 1
            if labels.get("selected"):
                selected_source_counter[str(source_label)] += 1
                selected_success += int(bool(labels.get("positive_for_critic")))
            if labels.get("validator_failure_reason"):
                failure_counter[str(labels.get("validator_failure_reason"))] += 1
            elif not labels.get("validator_valid"):
                failure_counter["validator_failed"] += 1
            max_alignment = _safe_float(metrics.get("max_alignment_deviation_deg"))
            mean_alignment = _safe_float(metrics.get("mean_alignment_deviation_deg"))
            joint_step = _safe_float(metrics.get("joint_step_max_l2"))
            jump_cost = _safe_float(metrics.get("joint_step_jump_cost"), 0.0) or 0.0
            point_count = len(candidate.get("trajectory", {}).get("points") or [])
            if max_alignment is not None:
                alignment_max_values.append(max_alignment)
            if mean_alignment is not None:
                alignment_mean_values.append(mean_alignment)
            if joint_step is not None:
                jump_values.append(joint_step)
            latency_proxy_values.append(50.0 + 1.5 * point_count + 500.0 * jump_cost)
            if labels.get("validator_valid"):
                joint_limit_valid_count += 1
            if metrics.get("goal_pose_valid"):
                goal_valid_count += 1
    request_count = len(groups)
    candidate_count = len(candidates)
    return {
        "request_count": int(request_count),
        "candidate_count": int(candidate_count),
        "success_at_k": _round(success_at_k / max(request_count, 1)),
        "selected_success": _round(selected_success / max(request_count, 1)),
        "failure_distribution": dict(failure_counter),
        "source_label_distribution": dict(source_counter),
        "selected_source_label_distribution": dict(selected_source_counter),
        "latency_proxy_ms": {
            "p50": _round(_percentile(latency_proxy_values, 0.50)),
            "p95": _round(_percentile(latency_proxy_values, 0.95)),
            "note": "proxy until phase8 ROS benchmark records model/critic/curobo/validator wall time",
        },
        "alignment": {
            "max_deviation_p50_deg": _round(_percentile(alignment_max_values, 0.50)),
            "max_deviation_p95_deg": _round(_percentile(alignment_max_values, 0.95)),
            "mean_deviation_p50_deg": _round(_percentile(alignment_mean_values, 0.50)),
        },
        "joint_jump": {
            "joint_step_max_l2_p50": _round(_percentile(jump_values, 0.50)),
            "joint_step_max_l2_p95": _round(_percentile(jump_values, 0.95)),
        },
        "joint_limit_valid_ratio": _round(joint_limit_valid_count / max(candidate_count, 1)),
        "goal_valid_ratio": _round(goal_valid_count / max(candidate_count, 1)),
        "collision_status": "unchecked_interface_reserved",
    }


def write_template(path: Path) -> None:
    path.write_text(
        "\n".join([
            "<!-- [caohy] diffusionSeedLearning phase 8 benchmark report template -->",
            "# Diffusion Seed Benchmark Report",
            "",
            "## Run Metadata",
            "",
            "- benchmark_manifest: `<path>`",
            "- planner_git_commit: `<commit>`",
            "- model_checkpoint: `<path>`",
            "- critic_checkpoint: `<path or none>`",
            "- mode: `rule_baseline | shadow | candidate | candidate_critic`",
            "- fixed_total_budget_ms: `<number>`",
            "",
            "## Required Metrics",
            "",
            "| Metric | Value | Notes |",
            "|---|---:|---|",
            "| success@K |  | any candidate passes hard validator |",
            "| selected_success |  | selected final trajectory passes hard validator |",
            "| fixed_budget_success |  | includes model + critic + CuRobo + validator time |",
            "| latency_p50_ms |  | wall time, not proxy |",
            "| latency_p95_ms |  | wall time, not proxy |",
            "| alignment_max_p95_deg |  | hard validator recompute |",
            "| joint_step_max_l2_p95 |  | hard validator recompute |",
            "| collision_valid_ratio |  | world collision replay |",
            "| joint_limit_valid_ratio |  | hard validator recompute |",
            "",
            "## Distributions",
            "",
            "- failure_distribution: `{}`",
            "- source_label_distribution: `{}`",
            "- selected_source_label_distribution: `{}`",
            "",
        ]),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    samples = read_jsonl(args.validated_samples)
    request_keys = {
        str(sample.get("source", {}).get("move_file") or sample.get("source", {}).get("plan_request_index"))
        for sample in samples
        if sample.get("sample_type") == "candidate"
    }
    task_paths = write_task_sets(
        args.public_root,
        args.random_seed,
        args.random_task_count,
        args.primitive_task_count,
    )
    manifest = write_manifest(args, task_paths, available_request_count=len(request_keys))
    write_template(args.template_out)
    summary = {
        "schema_version": "diffusion_seed_benchmark_smoke_summary.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest_out),
        "public_root": str(args.public_root),
        "task_files": task_paths,
        "validated_samples": str(args.validated_samples),
        "task_set_count": len(manifest["task_sets"]),
        "validated_sample_summary": summarize_validated_samples(samples),
        "limitations": [
            "Current summary is offline lifecycle/validator replay, not a fresh ROS benchmark run.",
            "Latency fields are proxies until runner is connected to live candidate/curobo timing.",
            "Collision replay remains unchecked because phase 3 validator reserved but did not execute world collision.",
        ],
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    metrics = summary["validated_sample_summary"]
    args.summary_md.write_text(
        "\n".join([
            "<!-- [caohy] diffusionSeedLearning phase 8 benchmark smoke report -->",
            "# Benchmark Smoke Report",
            "",
            f"- manifest: `{args.manifest_out}`",
            f"- public_root: `{args.public_root}`",
            f"- validated_samples: `{args.validated_samples}`",
            f"- request_count: `{metrics['request_count']}`",
            f"- candidate_count: `{metrics['candidate_count']}`",
            f"- success@K: `{metrics['success_at_k']}`",
            f"- selected_success: `{metrics['selected_success']}`",
            f"- latency_proxy_p50_ms: `{metrics['latency_proxy_ms']['p50']}`",
            f"- latency_proxy_p95_ms: `{metrics['latency_proxy_ms']['p95']}`",
            f"- collision_status: `{metrics['collision_status']}`",
            "",
            "该报告是 phase 8 benchmark 框架 smoke 输出，不是 fresh ROS 20/100 点实测；真实耗时和碰撞回放需要后续在同一 manifest 下补充。",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
