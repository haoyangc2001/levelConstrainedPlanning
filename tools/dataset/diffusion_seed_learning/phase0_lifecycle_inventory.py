#!/usr/bin/env python3
"""Phase 0 lifecycle inventory for diffusion seed learning.

The script audits existing level_plan_lifecycle JSON files and emits a small
repo-side report. It does not copy trajectories into a dataset and does not
write large artifacts.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_LIFECYCLE_ROOT = Path("readCaohy/logs/trajectory_planning/level_plan_lifecycle")
DEFAULT_PROJECT_ROOT = Path("resource/config/Level_Test_V2_caohy")
DEFAULT_OUTPUT = Path(
    "readCaohy/plans/diffusionSeedLearning/phase0_data_inventory.json"
)

TOP_LEVEL_REQUIRED = (
    "schema_version",
    "request",
    "start_state",
    "candidates",
    "selection",
    "result",
)

CANDIDATE_REQUIRED = (
    "candidate_id",
    "candidate_index",
    "source_label",
    "selected",
    "entered_pool",
    "trajectory_points",
    "trajectory_summary",
    "alignment_valid",
    "max_alignment_deviation_deg",
    "mean_alignment_deviation_deg",
    "start_joint_gap_l2",
    "joint_step_max_l2",
    "joint_step_max_abs",
    "twist_smoothness_cost",
    "goal_pose_valid",
)

EXPECTED_SOURCE_LABELS = {
    "planner": {
        "source_type": "planner",
        "zh": "CuRobo planner 直出候选",
    },
    "planner_legacy": {
        "source_type": "planner",
        "zh": "旧 planner 分支候选",
    },
    "alignment_seed_trajopt_1..N": {
        "source_type": "rule",
        "zh": "规则 alignment seed 经 CuRobo trajopt 修复后的候选",
    },
    "alignment_seed_family_1..N": {
        "source_type": "rule_raw",
        "zh": "原始规则 seed；默认不应直接入最终候选池",
    },
    "alignment_seed_split_*": {
        "source_type": "rule",
        "zh": "分段修复候选",
    },
    "alignment_seed_sequence_*": {
        "source_type": "rule",
        "zh": "sequence 分支候选",
    },
    "diffusion_seed_*": {
        "source_type": "diffusion",
        "zh": "后续 diffusion provider 生成的 seed 修复候选",
    },
    "critic_selected_*": {
        "source_type": "critic",
        "zh": "后续 success critic 预筛后的候选",
    },
    "fallback_*": {
        "source_type": "fallback",
        "zh": "模型失败或超时后的回退候选",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=None,
        help="Lifecycle run_dir to inspect. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=3,
        help="Use newest N run_* directories when --run-dir is omitted.",
    )
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def select_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path for path in args.run_dir if path.is_dir()]
    candidates = sorted(
        [path for path in args.lifecycle_root.glob("run_*") if path.is_dir()],
        key=lambda item: item.name,
    )
    return candidates[-max(1, int(args.max_runs)) :]


def contains_solve_time(data: Any) -> bool:
    wanted = ("solve_time", "solve_time_sec", "solve_time_s", "duration_sec", "elapsed_sec")
    stack = [data]
    while stack:
        item = stack.pop()
        if isinstance(item, dict):
            for key, value in item.items():
                key_lower = str(key).lower()
                if any(name in key_lower for name in wanted):
                    return True
                stack.append(value)
        elif isinstance(item, list):
            stack.extend(item)
    return False


def summarize_move(path: Path) -> dict[str, Any]:
    data = read_json(path)
    result = data.get("result") if isinstance(data.get("result"), dict) else {}
    selection = data.get("selection") if isinstance(data.get("selection"), dict) else {}
    candidates = data.get("candidates") if isinstance(data.get("candidates"), list) else []
    missing_top_level = [
        key for key in TOP_LEVEL_REQUIRED
        if key not in data or (key == "selection" and not isinstance(data.get(key), dict))
    ]

    candidate_missing = []
    source_labels = []
    trajectory_point_counts = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            candidate_missing.append({"candidate_id": None, "missing": list(CANDIDATE_REQUIRED)})
            continue
        source_labels.append(str(candidate.get("source_label")))
        points = candidate.get("trajectory_points")
        trajectory_point_counts.append(len(points) if isinstance(points, list) else 0)
        missing = [key for key in CANDIDATE_REQUIRED if key not in candidate]
        if missing:
            candidate_missing.append({
                "candidate_id": candidate.get("candidate_id"),
                "source_label": candidate.get("source_label"),
                "missing": missing,
            })

    selected_source_label = selection.get("selected_source_label")
    return {
        "file": str(path),
        "move_file": path.name,
        "plan_request_index": data.get("plan_request_index"),
        "schema_version": data.get("schema_version"),
        "status": result.get("status") or data.get("status"),
        "failure_reason": result.get("failure_reason") or selection.get("failure_reason"),
        "top_level_missing": missing_top_level,
        "request_present": isinstance(data.get("request"), dict),
        "start_state_present": isinstance(data.get("start_state"), dict),
        "selection_present": isinstance(data.get("selection"), dict) and bool(data.get("selection")),
        "result_failure_reason_present": (
            isinstance(data.get("result"), dict) and "failure_reason" in data["result"]
        ),
        "solve_time_present": contains_solve_time(data),
        "candidate_count": len(candidates),
        "candidate_source_labels": source_labels,
        "candidate_missing_required_fields": candidate_missing,
        "candidate_trajectory_point_counts": trajectory_point_counts,
        "candidate_trajectories_complete": (
            len(candidates) > 0
            and not candidate_missing
            and all(count > 0 for count in trajectory_point_counts)
        ),
        "selected_candidate_id": selection.get("selected_candidate_id"),
        "selected_source_label": selected_source_label,
        "selected_source_label_present": selected_source_label is not None,
    }


def summarize_run(run_dir: Path) -> dict[str, Any]:
    move_files = sorted(run_dir.glob("*.json"))
    moves = [summarize_move(path) for path in move_files]
    status_counts = collections.Counter(str(move["status"]) for move in moves)
    source_counts: collections.Counter[str] = collections.Counter()
    selected_source_counts: collections.Counter[str] = collections.Counter()
    missing_field_counts: collections.Counter[str] = collections.Counter()
    for move in moves:
        source_counts.update(move["candidate_source_labels"])
        selected = move.get("selected_source_label")
        if selected:
            selected_source_counts[str(selected)] += 1
        for key in move.get("top_level_missing", []):
            missing_field_counts[f"top_level.{key}"] += 1
        if not move.get("solve_time_present"):
            missing_field_counts["solve_time"] += 1
        for candidate_gap in move.get("candidate_missing_required_fields", []):
            for key in candidate_gap.get("missing", []):
                missing_field_counts[f"candidate.{key}"] += 1

    return {
        "run_dir": str(run_dir),
        "move_count": len(moves),
        "status_counts": dict(sorted(status_counts.items())),
        "success_count": status_counts.get("success", 0),
        "failed_count": len(moves) - status_counts.get("success", 0),
        "moves_with_candidates": sum(1 for item in moves if item["candidate_count"] > 0),
        "moves_without_candidates": sum(1 for item in moves if item["candidate_count"] == 0),
        "candidate_count_total": sum(item["candidate_count"] for item in moves),
        "source_label_counts": dict(sorted(source_counts.items())),
        "selected_source_label_counts": dict(sorted(selected_source_counts.items())),
        "missing_field_counts": dict(sorted(missing_field_counts.items())),
        "moves": moves,
    }


def summarize_obstacles(project_root: Path) -> dict[str, Any]:
    obstacle_dir = project_root / "obstacles"
    output: dict[str, Any] = {}
    total_boxes = 0
    for name in ("abs.autosave.json", "rel.autosave.json"):
        path = obstacle_dir / name
        item: dict[str, Any] = {
            "path": str(path),
            "exists": path.is_file(),
            "sha256": sha256_file(path),
            "box_count": 0,
            "frame_id": None,
        }
        if path.is_file():
            data = read_json(path)
            if isinstance(data, dict):
                boxes = data.get("boxes")
                item["box_count"] = len(boxes) if isinstance(boxes, list) else 0
                item["frame_id"] = data.get("frame_id")
                item["has_base_pose"] = "base_pose" in data
        total_boxes += int(item["box_count"])
        output[name] = item
    output["primitive_world_summary"] = {
        "representation": "abs/rel autosave JSON cuboids",
        "mesh_or_pointcloud_encoding": False,
        "total_box_count": total_boxes,
    }
    return output


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    run_dirs = select_run_dirs(args)
    runs = [summarize_run(path) for path in run_dirs]
    all_status = collections.Counter()
    all_sources = collections.Counter()
    all_selected = collections.Counter()
    for run in runs:
        all_status.update(run["status_counts"])
        all_sources.update(run["source_label_counts"])
        all_selected.update(run["selected_source_label_counts"])

    return {
        "schema_version": "phase0_inventory.v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "task": "diffusionSeedLearning phase 0 lifecycle inventory",
        "run_selection": {
            "lifecycle_root": str(args.lifecycle_root),
            "selected_run_dirs": [str(path) for path in run_dirs],
            "max_runs": args.max_runs,
        },
        "frozen_profile": {
            "robot_profile": "sr5",
            "robot_model": "XMS5-R800-W4G3B4C",
            "dof": 6,
            "joint_names": [
                "XMS5-R800-W4G3B4C_joint_1",
                "XMS5-R800-W4G3B4C_joint_2",
                "XMS5-R800-W4G3B4C_joint_3",
                "XMS5-R800-W4G3B4C_joint_4",
                "XMS5-R800-W4G3B4C_joint_5",
                "XMS5-R800-W4G3B4C_joint_6",
            ],
            "tool_frame": "tool1",
            "alignment_semantics": "tool1 y+ -> world z-",
            "first_version_profile_scope": "SR5 only; CR7 remains historical regression reference",
        },
        "obstacle_scope": summarize_obstacles(args.project_root),
        "source_label_whitelist": EXPECTED_SOURCE_LABELS,
        "aggregate": {
            "run_count": len(runs),
            "move_count": sum(run["move_count"] for run in runs),
            "candidate_count_total": sum(run["candidate_count_total"] for run in runs),
            "status_counts": dict(sorted(all_status.items())),
            "source_label_counts": dict(sorted(all_sources.items())),
            "selected_source_label_counts": dict(sorted(all_selected.items())),
            "known_schema_gaps": [
                "solve_time is not recorded as a unified top-level lifecycle field",
                "failed moves can have empty selection and empty candidates when no solver output survives",
            ],
        },
        "runs": runs,
    }


def main() -> None:
    args = parse_args()
    report = build_report(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")
    aggregate = report["aggregate"]
    print(
        "runs={run_count} moves={move_count} candidates={candidate_count_total} statuses={status_counts}".format(
            **aggregate
        )
    )


if __name__ == "__main__":
    main()
