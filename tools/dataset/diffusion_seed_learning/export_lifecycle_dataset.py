#!/usr/bin/env python3
"""Export level_plan_lifecycle logs into a versioned diffusion seed dataset."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from level_planner_core.seed_provider import infer_optimized, infer_source_type  # noqa: E402


SCHEMA_VERSION = "diffusion_lifecycle_dataset.v1"
SCRIPT_VERSION = "phase2_exporter.v1"
DEFAULT_LIFECYCLE_ROOT = Path("readCaohy/logs/trajectory_planning/level_plan_lifecycle")
DEFAULT_PUBLIC_ROOT = Path("/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/datasets")
DEFAULT_POINTER_OUT = Path("runs/diffusion_seed_learning/dataset_manifest_pointer.json")
DEFAULT_SUMMARY_OUT = Path("runs/diffusion_seed_learning/dataset_summary.md")
DEFAULT_SCHEMA_PATH = Path("artifacts/source_dataset_schema.json")
DEFAULT_PROJECT_ROOT = Path("configs")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--run-dir", type=Path, action="append", default=None)
    parser.add_argument("--max-runs", type=int, default=3)
    parser.add_argument("--dataset-name", type=str, default=None)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--project-root", type=Path, default=DEFAULT_PROJECT_ROOT)
    parser.add_argument("--pointer-out", type=Path, default=DEFAULT_POINTER_OUT)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_SUMMARY_OUT)
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except Exception:
        return None


def select_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        return [path for path in args.run_dir if path.is_dir()]
    candidates = sorted(
        [path for path in args.lifecycle_root.glob("run_*") if path.is_dir()],
        key=lambda item: item.name,
    )
    return candidates[-max(1, int(args.max_runs)) :]


def split_for_run(run_name: str, ordered_run_names: list[str]) -> str:
    if not ordered_run_names:
        return "train"
    index = ordered_run_names.index(run_name)
    if len(ordered_run_names) == 1:
        return "train"
    ratio = index / max(1, len(ordered_run_names) - 1)
    if ratio < 0.8:
        return "train"
    if ratio < 0.9:
        return "val"
    return "test"


def obstacle_summary(project_root: Path) -> dict[str, Any]:
    obstacle_dir = project_root / "obstacles"
    summary: dict[str, Any] = {
        "type": "abs_rel_autosave_json",
        "mesh_or_pointcloud_encoding": False,
        "files": {},
        "total_box_count": 0,
    }
    for name in ("abs.autosave.json", "rel.autosave.json"):
        path = obstacle_dir / name
        item: dict[str, Any] = {
            "path": str(path),
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
        summary["files"][name] = item
        summary["total_box_count"] += int(item["box_count"])
    return summary


def compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    source_label = str(candidate.get("source_label") or "unknown")
    trajectory_points = candidate.get("trajectory_points")
    if not isinstance(trajectory_points, list):
        trajectory_points = []
    source_type = candidate.get("source_type") or infer_source_type(source_label)
    optimized = candidate.get("optimized")
    if optimized is None:
        optimized = infer_optimized(source_label)
    metrics = candidate.get("metrics")
    if not isinstance(metrics, dict):
        metric_keys = (
            "alignment_valid",
            "max_alignment_deviation_deg",
            "mean_alignment_deviation_deg",
            "goal_pose_valid",
            "position_error_m",
            "orientation_error_deg",
            "start_joint_gap_l2",
            "joint_step_jump_cost",
            "joint_step_max_l2",
            "joint_step_max_abs",
            "twist_smoothness_cost",
        )
        metrics = {key: candidate.get(key) for key in metric_keys if key in candidate}
    return {
        "candidate_id": candidate.get("candidate_id"),
        "candidate_index": candidate.get("candidate_index"),
        "source_label": source_label,
        "source_type": source_type,
        "optimized": bool(optimized),
        "selected": bool(candidate.get("selected")),
        "entered_pool": bool(candidate.get("entered_pool", True)),
        "trajectory": {
            "format": "joint_position_rad",
            "shape": [
                len(trajectory_points),
                len(trajectory_points[0]) if trajectory_points else 0,
            ],
            "points": trajectory_points,
            "summary": candidate.get("trajectory_summary"),
        },
        "metadata": candidate.get("metadata") if isinstance(candidate.get("metadata"), dict) else {},
        "precheck": candidate.get("precheck") if isinstance(candidate.get("precheck"), dict) else {},
        "metrics": metrics,
        "terminal_goal_pose_summary": candidate.get("terminal_goal_pose_summary"),
    }


def candidate_labels(move: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    result = move.get("result") if isinstance(move.get("result"), dict) else {}
    status = result.get("status") or move.get("status")
    failure_reason = result.get("failure_reason")
    metrics = candidate.get("metrics") if isinstance(candidate.get("metrics"), dict) else {}
    alignment_valid = metrics.get("alignment_valid", candidate.get("alignment_valid"))
    goal_pose_valid = metrics.get("goal_pose_valid", candidate.get("goal_pose_valid"))
    selected = bool(candidate.get("selected"))
    positive = (
        status == "success"
        and selected
        and alignment_valid is True
        and goal_pose_valid is True
        and candidate.get("source_type") != "rule_raw"
    )
    return {
        "planner_status": status,
        "failure_reason": failure_reason,
        "selected": selected,
        "positive_for_diffusion": bool(positive),
        "positive_for_critic": status == "success" and selected,
        "negative_for_critic": status != "success" or not selected,
    }


def build_candidate_sample(
    *,
    dataset_name: str,
    run_dir: Path,
    move_file: Path,
    move: dict[str, Any],
    candidate: dict[str, Any],
    split: str,
    obstacle: dict[str, Any],
) -> dict[str, Any]:
    compact = compact_candidate(candidate)
    sample_id = f"{run_dir.name}:{move_file.stem}:{compact['candidate_id']}"
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "sample_type": "candidate",
        "split": split,
        "source": {
            "run_dir": str(run_dir),
            "move_file": str(move_file),
            "plan_request_index": move.get("plan_request_index"),
            "lifecycle_schema_version": move.get("schema_version"),
            "script_version": SCRIPT_VERSION,
        },
        "task": {
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
            "alignment": {
                "tool_axis": "y+",
                "world_axis": "z-",
                "tolerance_deg": move.get("request", {}).get("level_tolerance_deg"),
            },
            "target_pose": move.get("request", {}).get("target_pose"),
            "strict_level": move.get("request", {}).get("strict_level"),
        },
        "request": move.get("request"),
        "start_state": move.get("start_state"),
        "candidate": compact,
        "optimized_result": {
            "result": move.get("result"),
            "selection": move.get("selection"),
        },
        "labels": candidate_labels(move, compact),
        "source_lineage": {
            "source_label": compact["source_label"],
            "source_type": compact["source_type"],
            "selected_source_label": (
                move.get("selection", {}).get("selected_source_label")
                if isinstance(move.get("selection"), dict) else None
            ),
        },
        "obstacle_world": obstacle,
    }


def build_request_failure_sample(
    *,
    dataset_name: str,
    run_dir: Path,
    move_file: Path,
    move: dict[str, Any],
    split: str,
    obstacle: dict[str, Any],
) -> dict[str, Any]:
    result = move.get("result") if isinstance(move.get("result"), dict) else {}
    sample_id = f"{run_dir.name}:{move_file.stem}:request_failure"
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "sample_id": sample_id,
        "sample_type": "request_failure",
        "split": split,
        "source": {
            "run_dir": str(run_dir),
            "move_file": str(move_file),
            "plan_request_index": move.get("plan_request_index"),
            "lifecycle_schema_version": move.get("schema_version"),
            "script_version": SCRIPT_VERSION,
        },
        "task": {
            "robot_profile": "sr5",
            "robot_model": "XMS5-R800-W4G3B4C",
            "dof": 6,
            "tool_frame": "tool1",
            "alignment": {
                "tool_axis": "y+",
                "world_axis": "z-",
                "tolerance_deg": move.get("request", {}).get("level_tolerance_deg"),
            },
            "target_pose": move.get("request", {}).get("target_pose"),
            "strict_level": move.get("request", {}).get("strict_level"),
        },
        "request": move.get("request"),
        "start_state": move.get("start_state"),
        "candidate": None,
        "optimized_result": {
            "result": result,
            "selection": move.get("selection"),
        },
        "labels": {
            "planner_status": result.get("status") or move.get("status"),
            "failure_reason": result.get("failure_reason"),
            "selected": False,
            "positive_for_diffusion": False,
            "positive_for_critic": False,
            "negative_for_critic": True,
        },
        "source_lineage": {
            "source_label": None,
            "source_type": None,
            "selected_source_label": None,
        },
        "obstacle_world": obstacle,
    }


def export_samples(args: argparse.Namespace, dataset_name: str, dataset_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    run_dirs = select_run_dirs(args)
    non_empty_run_names = [
        run_dir.name for run_dir in run_dirs if any(run_dir.glob("*.json"))
    ]
    obstacle = obstacle_summary(args.project_root)
    samples: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        move_files = sorted(run_dir.glob("*.json"))
        if not move_files:
            continue
        split = split_for_run(run_dir.name, non_empty_run_names)
        for move_file in move_files:
            move = read_json(move_file)
            candidates = move.get("candidates")
            if isinstance(candidates, list) and candidates:
                for candidate in candidates:
                    if not isinstance(candidate, dict):
                        continue
                    samples.append(build_candidate_sample(
                        dataset_name=dataset_name,
                        run_dir=run_dir,
                        move_file=move_file,
                        move=move,
                        candidate=candidate,
                        split=split,
                        obstacle=obstacle,
                    ))
            else:
                samples.append(build_request_failure_sample(
                    dataset_name=dataset_name,
                    run_dir=run_dir,
                    move_file=move_file,
                    move=move,
                    split=split,
                    obstacle=obstacle,
                ))
    dataset_dir.mkdir(parents=True, exist_ok=True)
    samples_path = dataset_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False, separators=(",", ":")) + "\n")
    return samples, run_dirs


def build_manifest(
    *,
    args: argparse.Namespace,
    dataset_name: str,
    dataset_dir: Path,
    samples: list[dict[str, Any]],
    run_dirs: list[Path],
) -> dict[str, Any]:
    split_counts = collections.Counter(sample["split"] for sample in samples)
    sample_type_counts = collections.Counter(sample["sample_type"] for sample in samples)
    status_counts = collections.Counter(sample["labels"]["planner_status"] for sample in samples)
    source_label_counts = collections.Counter(
        sample["source_lineage"]["source_label"] or "none"
        for sample in samples
    )
    selected_source_counts = collections.Counter(
        sample["source_lineage"]["selected_source_label"] or "none"
        for sample in samples
    )
    samples_path = dataset_dir / "samples.jsonl"
    manifest_path = dataset_dir / "manifest.json"
    manifest = {
        "schema_version": "diffusion_dataset_manifest.v1",
        "dataset_schema_version": SCHEMA_VERSION,
        "script_version": SCRIPT_VERSION,
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "public_dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "samples_path": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "planner_status_counts": dict(sorted(status_counts.items())),
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "selected_source_label_counts": dict(sorted(selected_source_counts.items())),
        "git_commit": git_commit(),
        "schema_file": str(args.schema_path),
        "export_command_hint": (
            "python3 readCaohy/test/diffusion_seed_learning/export_lifecycle_dataset.py "
            f"--dataset-name {dataset_name}"
        ),
        "source_lifecycle_runs": [str(path) for path in run_dirs],
        "obstacle_world": obstacle_summary(args.project_root),
        "split_policy": "All samples from the same run_dir are assigned to one split.",
        "repo_side_policy": "Only schema, pointer and summary are stored in git; samples stay under /pub/data/caohy.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def write_pointer(path: Path, manifest: dict[str, Any]) -> None:
    pointer = {
        "schema_version": "dataset_manifest_pointer.v1",
        "last_updated": "2026-07-13",
        "current_dataset_name": manifest["dataset_name"],
        "manifest_path": manifest["manifest_path"],
        "samples_path": manifest["samples_path"],
        "sample_count": manifest["sample_count"],
        "dataset_schema_version": manifest["dataset_schema_version"],
        "git_commit": manifest["git_commit"],
        "notes": "Large dataset artifacts are stored under /pub/data/caohy and are not committed to the repository.",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_summary(path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "<!-- [caohy] diffusionSeedLearning 阶段 2 数据集导出摘要 -->",
        "# Dataset Summary",
        "",
        f"- dataset: `{manifest['dataset_name']}`",
        f"- schema: `{manifest['dataset_schema_version']}`",
        f"- manifest: `{manifest['manifest_path']}`",
        f"- samples: `{manifest['samples_path']}`",
        f"- sample_count: `{manifest['sample_count']}`",
        f"- git_commit: `{manifest['git_commit']}`",
        "",
        "## Splits",
        "",
    ]
    for key, value in manifest["split_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Sample Types", ""])
    for key, value in manifest["sample_type_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Planner Status", ""])
    for key, value in manifest["planner_status_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend(["", "## Source Labels", ""])
    for key, value in manifest["source_label_counts"].items():
        lines.append(f"- `{key}`: {value}")
    lines.extend([
        "",
        "## Notes",
        "",
        "- 大体量 `samples.jsonl` 和 `manifest.json` 位于 `/pub/data/caohy`，仓库只保留 pointer 和摘要。",
        "- 当前只有一个非空 lifecycle run，因此所有样本都进入 `train` split；后续新增 run 后 exporter 会按 run_dir 分配 split。",
        "- `request_failure` 样本保留无候选失败 move，供 critic 和失败分布分析使用，不作为 diffusion 正样本。",
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or f"sr5_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    dataset_dir = args.public_root / dataset_name
    samples, run_dirs = export_samples(args, dataset_name, dataset_dir)
    manifest = build_manifest(
        args=args,
        dataset_name=dataset_name,
        dataset_dir=dataset_dir,
        samples=samples,
        run_dirs=run_dirs,
    )
    write_pointer(args.pointer_out, manifest)
    write_summary(args.summary_out, manifest)
    print(f"dataset_dir={dataset_dir}")
    print(f"manifest={manifest['manifest_path']}")
    print(f"samples={manifest['samples_path']}")
    print(f"sample_count={manifest['sample_count']}")


if __name__ == "__main__":
    main()
