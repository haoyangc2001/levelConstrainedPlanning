#!/usr/bin/env python3
"""Export standalone core run artifacts into candidate-level training JSONL."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "standalone_candidate_dataset.v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _iter_result_files(inputs: list[Path]) -> list[Path]:
    result_files: list[Path] = []
    for item in inputs:
        if item.is_file():
            result_files.append(item)
        elif item.is_dir():
            result_files.extend(sorted(item.rglob("result.json")))
    return sorted(set(result_files))


def _artifact_path(result: dict[str, Any], result_path: Path, key: str, default_name: str) -> Path:
    raw = (result.get("artifacts") or {}).get(key)
    if raw:
        return Path(raw)
    return result_path.parent / default_name


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    trajectory = dict(candidate.get("trajectory") or {})
    points = trajectory.get("points") or []
    compact = {
        "candidate_id": candidate.get("candidate_id"),
        "source_type": (candidate.get("source_lineage") or {}).get("source_type"),
        "source_label": (candidate.get("source_lineage") or {}).get("source_label"),
        "selected": bool((candidate.get("lifecycle") or {}).get("selected")),
        "trajectory": {
            "format": trajectory.get("format", "joint_position_rad"),
            "shape": trajectory.get("shape") or [
                len(points),
                len(points[0]) if points else 0,
            ],
            "points": points,
            "artifact_path": trajectory.get("artifact_path"),
            "sha256": trajectory.get("sha256"),
        },
        "metrics": candidate.get("metrics") or (candidate.get("validator_metrics") or {}).get("metrics") or {},
        "validator_metrics": candidate.get("validator_metrics") or {},
        "optimizer_result": candidate.get("optimizer_result") or {},
        "failure_stage": candidate.get("failure_stage"),
        "failure_reason": candidate.get("failure_reason"),
    }
    return compact


def _sample_from_candidate(
    *,
    candidate: dict[str, Any],
    result: dict[str, Any],
    run_record: dict[str, Any],
    result_path: Path,
    candidates_jsonl: Path | None,
    dataset_name: str,
) -> dict[str, Any]:
    normalized = run_record.get("normalized_request") or {}
    alignment = normalized.get("alignment") or {}
    labels = dict(candidate.get("labels") or {})
    source_lineage = dict(candidate.get("source_lineage") or {})
    task = {
        "robot_profile": normalized.get("robot_profile") or run_record.get("robot_profile") or "sr5",
        "tool_frame": (run_record.get("environment") or {}).get("tool_frames", ["tool1"])[0],
        "target_pose": normalized.get("target_pose"),
        "alignment": {
            "local_axis": alignment.get("local_axis"),
            "target_world_axis": alignment.get("target_world_axis"),
            "tolerance_deg": alignment.get("tolerance_deg")
            or (result.get("metrics") or {}).get("alignment", {}).get("tolerance_deg"),
            "strict_level": alignment.get("strict_level"),
        },
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "sample_id": f"{result_path.parent.name}:{candidate.get('candidate_id')}",
        "sample_type": "candidate",
        "source": {
            "result_json": str(result_path),
            "planner_run_json": str(_artifact_path(result, result_path, "planner_run_json", "planner_run.json")),
            "candidates_jsonl": str(candidates_jsonl) if candidates_jsonl else None,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "task": task,
        "start_state": {
            "service_start_joint": normalized.get("start_joint") or [],
            "joint_names": (run_record.get("environment") or {}).get("joint_names") or [],
        },
        "obstacle_world": run_record.get("world_summary") or (result.get("metrics") or {}).get("world") or {},
        "candidate": _compact_candidate(candidate),
        "source_lineage": source_lineage,
        "labels": {
            "planner_status": labels.get("planner_status") or result.get("status"),
            "candidate_status": labels.get("candidate_status"),
            "failure_reason": labels.get("failure_reason") or candidate.get("failure_reason"),
            "selected": bool(labels.get("selected")),
            "validator_valid": bool(labels.get("validator_valid")),
            "positive_for_diffusion": bool(labels.get("positive_for_diffusion")),
            "positive_for_critic": bool(labels.get("positive_for_critic")),
            "negative_for_critic": bool(labels.get("negative_for_critic")),
            "fallback_recovered": bool(labels.get("fallback_recovered")),
        },
    }


def _legacy_candidate_from_result(result: dict[str, Any]) -> dict[str, Any] | None:
    selected = result.get("selected_trajectory") or []
    if not selected:
        return None
    status = result.get("status")
    metrics = result.get("metrics") or {}
    candidate_id = metrics.get("selected_candidate_id") or "legacy_selected"
    validator_valid = status == "success"
    return {
        "candidate_id": candidate_id,
        "request_id": result.get("request_id"),
        "source_lineage": {
            "source_type": "planner_native",
            "source_label": "legacy_selected",
            "provider": "planner_native",
            "provider_mode": "legacy_result",
        },
        "trajectory": {
            "format": "joint_position_rad",
            "shape": [
                len(selected),
                len(selected[0]) if selected else 0,
            ],
            "points": selected,
        },
        "lifecycle": {
            "generated": True,
            "entered_pool": True,
            "repair_attempted": True,
            "repair_success": validator_valid,
            "hard_validation_attempted": True,
            "hard_validation_passed": validator_valid,
            "selected": validator_valid,
            "fallback_recovered": False,
        },
        "metrics": metrics,
        "validator_metrics": {
            "valid": validator_valid,
            "status": "valid" if validator_valid else "failed",
            "metrics": metrics,
        },
        "labels": {
            "planner_status": status,
            "candidate_status": status,
            "failure_reason": result.get("failure_reason"),
            "selected": validator_valid,
            "validator_valid": validator_valid,
            "positive_for_diffusion": validator_valid,
            "positive_for_critic": validator_valid,
            "negative_for_critic": not validator_valid,
            "fallback_recovered": False,
        },
        "failure_stage": None if validator_valid else "legacy_result",
        "failure_reason": result.get("failure_reason"),
    }


def _samples_from_result(path: Path, dataset_name: str) -> list[dict[str, Any]]:
    result = _read_json(path)
    run_path = _artifact_path(result, path, "planner_run_json", "planner_run.json")
    run_record = _read_json(run_path) if run_path.exists() else result.get("planner_run_record") or {}
    candidates_jsonl = _artifact_path(result, path, "candidates_jsonl", "candidates.jsonl")
    candidates = _read_jsonl(candidates_jsonl)
    if not candidates:
        candidates = list(result.get("candidate_records") or [])
    if not candidates:
        legacy = _legacy_candidate_from_result(result)
        candidates = [legacy] if legacy else []
    return [
        _sample_from_candidate(
            candidate=candidate,
            result=result,
            run_record=run_record,
            result_path=path,
            candidates_jsonl=candidates_jsonl if candidates_jsonl.exists() else None,
            dataset_name=dataset_name,
        )
        for candidate in candidates
        if candidate
    ]


def _write_manifest(
    *,
    samples: list[dict[str, Any]],
    out: Path,
    manifest_out: Path,
    dataset_name: str,
    result_files: list[Path],
) -> dict[str, Any]:
    source_counter = Counter(
        str(sample.get("source_lineage", {}).get("source_type") or "unknown")
        for sample in samples
    )
    status_counter = Counter(
        str(sample.get("labels", {}).get("candidate_status") or "unknown")
        for sample in samples
    )
    failure_counter = Counter(
        str(
            sample.get("labels", {}).get("failure_reason")
            or sample.get("candidate", {}).get("failure_reason")
            or "none"
        )
        for sample in samples
    )
    request_ids = {
        str(sample.get("source", {}).get("planner_run_json") or sample.get("source", {}).get("result_json"))
        for sample in samples
    }
    difficulty_counter = Counter(
        str(
            (
                (
                    _read_json(Path(sample.get("source", {}).get("planner_run_json")))
                    if sample.get("source", {}).get("planner_run_json")
                    and Path(sample.get("source", {}).get("planner_run_json")).exists()
                    else {}
                ).get("request", {})
                .get("metadata", {})
                .get("sampling", {})
                .get("difficulty_bucket")
            )
            or "unknown"
        )
        for sample in samples
    )
    manifest = {
        "schema_version": "standalone_candidate_dataset_manifest.v1",
        "dataset_name": dataset_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "samples_path": str(out),
        "samples_sha256": _sha256(out) if out.exists() else None,
        "generation_command": sys.argv,
        "result_file_count": len(result_files),
        "request_count": len(request_ids),
        "sample_count": len(samples),
        "candidate_count": len(samples),
        "candidate_with_points_count": sum(
            1 for sample in samples if sample.get("candidate", {}).get("trajectory", {}).get("points")
        ),
        "positive_for_diffusion": sum(
            1 for sample in samples if sample.get("labels", {}).get("positive_for_diffusion")
        ),
        "positive_for_critic": sum(
            1 for sample in samples if sample.get("labels", {}).get("positive_for_critic")
        ),
        "negative_for_critic": sum(
            1 for sample in samples if sample.get("labels", {}).get("negative_for_critic")
        ),
        "source_type_counts": dict(sorted(source_counter.items())),
        "candidate_status_counts": dict(sorted(status_counter.items())),
        "failure_reason_counts": dict(sorted(failure_counter.items())),
        "difficulty_bucket_counts": dict(sorted(difficulty_counter.items())),
        "input_result_files": [str(path) for path in result_files],
    }
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/diffusion_seed_learning/candidate_samples.jsonl"))
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--dataset-name", default="standalone_candidate_dataset")
    args = parser.parse_args(argv)

    result_files = _iter_result_files(args.input)
    samples: list[dict[str, Any]] = []
    for path in result_files:
        samples.extend(_samples_from_result(path, args.dataset_name))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")

    manifest_out = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest = _write_manifest(
        samples=samples,
        out=args.out,
        manifest_out=manifest_out,
        dataset_name=args.dataset_name,
        result_files=result_files,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
