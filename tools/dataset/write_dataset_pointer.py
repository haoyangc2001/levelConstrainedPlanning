#!/usr/bin/env python3
"""Write small repository pointers for public standalone datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _sha256(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_info(path: Path | None, *, hash_file: bool = True) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": None, "sha256": None}
    exists = path.exists()
    return {
        "path": str(path),
        "exists": bool(exists),
        "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        "sha256": _sha256(path) if hash_file and exists and path.is_file() else None,
    }


def _read_json(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def build_pointer(
    *,
    dataset_name: str,
    samples: Path,
    manifest: Path,
    validator_report: Path | None = None,
    batch_summary: Path | None = None,
    sampling_manifest: Path | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    manifest_payload = _read_json(manifest)
    validation = _read_json(validator_report)
    batch = _read_json(batch_summary)
    sampling = _read_json(sampling_manifest)
    return {
        "schema_version": "standalone_closed_loop_dataset_pointer.v1",
        "status": "active",
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "dataset": {
            "name": dataset_name,
            "sample_count": manifest_payload.get("sample_count"),
            "candidate_count": manifest_payload.get("candidate_count")
            or manifest_payload.get("sample_count"),
            "request_count": manifest_payload.get("request_count")
            or batch.get("request_count")
            or sampling.get("request_count"),
            "positive_for_diffusion": manifest_payload.get("positive_for_diffusion"),
            "positive_for_critic": manifest_payload.get("positive_for_critic"),
            "negative_for_critic": manifest_payload.get("negative_for_critic"),
            "source_type_counts": manifest_payload.get("source_type_counts") or {},
            "candidate_status_counts": manifest_payload.get("candidate_status_counts") or {},
            "difficulty_bucket_counts": manifest_payload.get("difficulty_bucket_counts")
            or batch.get("difficulty_bucket_counts")
            or sampling.get("difficulty_bucket_counts")
            or {},
            "obstacle_case_counts": batch.get("obstacle_case_counts")
            or sampling.get("obstacle_case_counts")
            or {},
            "training_dataset": _file_info(samples),
            "dataset_manifest": _file_info(manifest),
            "validator_report": _file_info(validator_report),
            "batch_summary": _file_info(batch_summary),
            "sampling_manifest": _file_info(sampling_manifest),
        },
        "notes": notes or [],
        "large_file_policy": "Do not commit dataset JSONL, lifecycle run directories, or checkpoint .pt files.",
    }


def update_current_artifacts(current_path: Path, pointer: dict[str, Any], pointer_path: Path) -> dict[str, Any]:
    current = _read_json(current_path)
    previous_dataset = current.get("dataset")
    if previous_dataset and "legacy_phase10_dataset" not in current:
        current["legacy_phase10_dataset"] = previous_dataset
    current["schema_version"] = current.get("schema_version", "standalone_level_planning.artifacts.v1")
    current["status"] = "active_standalone_closed_loop_dataset"
    current["updated_at"] = _utc_now()
    current["dataset"] = dict(pointer.get("dataset") or {})
    current["closed_loop_dataset_pointer"] = {
        "path": str(pointer_path),
        "git_commit": pointer.get("git_commit"),
        "dataset_name": (pointer.get("dataset") or {}).get("name"),
    }
    current["large_file_policy"] = pointer.get("large_file_policy")
    current_path.parent.mkdir(parents=True, exist_ok=True)
    current_path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return current


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--validator-report", type=Path)
    parser.add_argument("--batch-summary", type=Path)
    parser.add_argument("--sampling-manifest", type=Path)
    parser.add_argument("--out", type=Path, default=Path("artifacts/closed_loop_dataset_pointer.json"))
    parser.add_argument("--update-current-artifacts", type=Path)
    parser.add_argument("--note", action="append", default=[])
    args = parser.parse_args(argv)

    pointer = build_pointer(
        dataset_name=args.dataset_name,
        samples=args.samples,
        manifest=args.manifest,
        validator_report=args.validator_report,
        batch_summary=args.batch_summary,
        sampling_manifest=args.sampling_manifest,
        notes=list(args.note or []),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pointer, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.update_current_artifacts:
        update_current_artifacts(args.update_current_artifacts, pointer, args.out)
    print(json.dumps(pointer, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
