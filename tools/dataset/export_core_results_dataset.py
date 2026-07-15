#!/usr/bin/env python3
"""Export standalone core result artifacts into a small diffusion dataset JSONL."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "standalone_core_result_dataset.v1"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_result_files(inputs: list[Path]) -> list[Path]:
    result_files: list[Path] = []
    for item in inputs:
        if item.is_file():
            result_files.append(item)
        elif item.is_dir():
            result_files.extend(sorted(item.rglob("result.json")))
    return sorted(set(result_files))


def _sample_from_result(path: Path, dataset_name: str) -> dict[str, Any]:
    result = _read_json(path)
    selected = result.get("selected_trajectory") or []
    status = result.get("status")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "sample_id": f"{path.parent.name}:{result.get('request_id')}",
        "sample_type": "core_result",
        "source": {
            "result_json": str(path),
            "exported_at": datetime.now(timezone.utc).isoformat(),
        },
        "task": {
            "robot_profile": "sr5",
            "tool_frame": "tool1",
            "alignment": {
                "tool_axis": "y+",
                "world_axis": "z-",
                "tolerance_deg": result.get("metrics", {}).get("alignment", {}).get("tolerance_deg"),
            },
        },
        "candidate": {
            "candidate_id": result.get("metrics", {}).get("selected_candidate_id"),
            "source_type": "planner",
            "selected": status == "success",
            "trajectory": {
                "format": "joint_position_rad",
                "shape": [
                    len(selected),
                    len(selected[0]) if selected else 0,
                ],
                "points": selected,
            },
            "metrics": result.get("metrics", {}),
        },
        "labels": {
            "planner_status": status,
            "failure_reason": result.get("failure_reason"),
            "selected": status == "success",
            "positive_for_diffusion": status == "success",
            "positive_for_critic": status == "success",
            "negative_for_critic": status != "success",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/diffusion_seed_learning/core_result_samples.jsonl"))
    parser.add_argument("--dataset-name", default="standalone_core_results")
    args = parser.parse_args(argv)

    result_files = _iter_result_files(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for path in result_files:
            handle.write(json.dumps(_sample_from_result(path, args.dataset_name), ensure_ascii=False) + "\n")
    print(json.dumps({
        "schema_version": "standalone_core_result_export_report.v1",
        "dataset_name": args.dataset_name,
        "result_file_count": len(result_files),
        "out": str(args.out),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
