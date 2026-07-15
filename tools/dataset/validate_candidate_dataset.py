#!/usr/bin/env python3
"""Validate candidate-level dataset JSONL rows for the closed-loop planner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "schema_version",
    "dataset_name",
    "sample_id",
    "sample_type",
    "source",
    "task",
    "start_state",
    "candidate",
    "source_lineage",
    "labels",
}

REQUIRED_LABELS = {
    "planner_status",
    "selected",
    "validator_valid",
    "positive_for_diffusion",
    "positive_for_critic",
    "negative_for_critic",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            row["_line_number"] = line_number
            rows.append(row)
    return rows


def _trajectory_has_points(candidate: dict[str, Any]) -> bool:
    return bool((candidate.get("trajectory") or {}).get("points"))


def _validate_row(row: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    line = row.get("_line_number", "?")
    missing = sorted(REQUIRED_TOP_LEVEL - set(row))
    if missing:
        errors.append(f"line {line}: missing top-level fields: {missing}")
        return errors
    if row.get("sample_type") != "candidate":
        errors.append(f"line {line}: sample_type must be candidate")
    candidate = row.get("candidate")
    if not isinstance(candidate, dict):
        errors.append(f"line {line}: candidate must be an object")
        return errors
    labels = row.get("labels")
    if not isinstance(labels, dict):
        errors.append(f"line {line}: labels must be an object")
        return errors
    missing_labels = sorted(REQUIRED_LABELS - set(labels))
    if missing_labels:
        errors.append(f"line {line}: missing labels: {missing_labels}")
    if not candidate.get("candidate_id"):
        errors.append(f"line {line}: candidate.candidate_id is required")
    source_type = row.get("source_lineage", {}).get("source_type") or candidate.get("source_type")
    if not source_type:
        errors.append(f"line {line}: source_lineage.source_type is required")
    trajectory = candidate.get("trajectory") or {}
    points = trajectory.get("points") or []
    shape = trajectory.get("shape") or []
    if points:
        if not all(isinstance(point, list) for point in points):
            errors.append(f"line {line}: candidate.trajectory.points must be list[list[number]]")
        elif len(shape) == 2 and (int(shape[0]) != len(points) or int(shape[1]) != len(points[0])):
            errors.append(f"line {line}: candidate.trajectory.shape does not match points")
    elif labels.get("positive_for_diffusion") or labels.get("positive_for_critic"):
        errors.append(f"line {line}: positive candidate must include trajectory points")
    if labels.get("positive_for_diffusion") and not labels.get("validator_valid"):
        errors.append(f"line {line}: diffusion positive must be validator_valid")
    if labels.get("positive_for_critic") and not labels.get("validator_valid"):
        errors.append(f"line {line}: critic positive must be validator_valid")
    if labels.get("positive_for_critic") and labels.get("negative_for_critic"):
        errors.append(f"line {line}: candidate cannot be both positive and negative for critic")
    if not labels.get("validator_valid") and not (
        labels.get("failure_reason") or candidate.get("failure_reason")
    ):
        errors.append(f"line {line}: invalid candidate should keep failure_reason")
    return errors


def validate_candidate_dataset(
    samples: Path,
    *,
    require_positive: bool = False,
    require_negative: bool = False,
) -> dict[str, Any]:
    rows = _load_jsonl(samples)
    errors: list[str] = []
    for row in rows:
        errors.extend(_validate_row(row))
    source_counter = Counter(
        str(row.get("source_lineage", {}).get("source_type") or "unknown")
        for row in rows
    )
    status_counter = Counter(
        str(row.get("labels", {}).get("candidate_status") or "unknown")
        for row in rows
    )
    positive_for_diffusion = sum(
        1 for row in rows if row.get("labels", {}).get("positive_for_diffusion")
    )
    positive_for_critic = sum(
        1 for row in rows if row.get("labels", {}).get("positive_for_critic")
    )
    negative_for_critic = sum(
        1 for row in rows if row.get("labels", {}).get("negative_for_critic")
    )
    if require_positive and positive_for_diffusion < 1 and positive_for_critic < 1:
        errors.append("dataset must contain at least one positive candidate")
    if require_negative and negative_for_critic < 1:
        errors.append("dataset must contain at least one negative candidate")
    report = {
        "schema_version": "candidate_dataset_validation_report.v1",
        "samples": str(samples),
        "valid": not errors,
        "sample_count": len(rows),
        "candidate_with_points_count": sum(1 for row in rows if _trajectory_has_points(row.get("candidate") or {})),
        "positive_for_diffusion": positive_for_diffusion,
        "positive_for_critic": positive_for_critic,
        "negative_for_critic": negative_for_critic,
        "source_type_counts": dict(sorted(source_counter.items())),
        "candidate_status_counts": dict(sorted(status_counter.items())),
        "errors": errors,
    }
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-positive", action="store_true")
    parser.add_argument("--require-negative", action="store_true")
    args = parser.parse_args(argv)

    report = validate_candidate_dataset(
        args.samples,
        require_positive=bool(args.require_positive),
        require_negative=bool(args.require_negative),
    )
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
