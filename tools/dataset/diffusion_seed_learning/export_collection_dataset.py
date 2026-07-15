#!/usr/bin/env python3
"""Export lifecycle datasets from one or more collection summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PUBLIC_ROOT = Path("/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/datasets")
DEFAULT_REPORT_ROOT = Path("/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection-summary", type=Path, action="append", default=[])
    parser.add_argument("--collection-dir", type=Path, action="append", default=[])
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--public-root", type=Path, default=DEFAULT_PUBLIC_ROOT)
    parser.add_argument("--pointer-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_summary_paths(args: argparse.Namespace) -> list[Path]:
    paths = list(args.collection_summary)
    for collection_dir in args.collection_dir:
        paths.append(collection_dir / "collection_summary.json")
    unique = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if not resolved.is_file():
            raise FileNotFoundError(f"collection summary not found: {resolved}")
        unique.append(resolved)
    if not unique:
        raise ValueError("at least one --collection-summary or --collection-dir is required")
    return unique


def extract_run_dirs(summary_paths: list[Path]) -> list[Path]:
    run_dirs: list[Path] = []
    seen = set()
    for summary_path in summary_paths:
        summary = read_json(summary_path)
        for run in summary.get("runs", []):
            for changed in run.get("changed_lifecycle_runs", []):
                run_dir = changed.get("run_dir")
                if not run_dir:
                    continue
                path = Path(run_dir).resolve()
                if path in seen or not path.is_dir() or not any(path.glob("*.json")):
                    continue
                seen.add(path)
                run_dirs.append(path)
    if not run_dirs:
        raise ValueError("no non-empty lifecycle run directories found in collection summaries")
    return run_dirs


def main() -> None:
    args = parse_args()
    summary_paths = resolve_summary_paths(args)
    run_dirs = extract_run_dirs(summary_paths)
    dataset_name = args.dataset_name or f"sr5_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    pointer_out = args.pointer_out or (DEFAULT_REPORT_ROOT / f"{dataset_name}_pointer.json")
    summary_out = args.summary_out or (DEFAULT_REPORT_ROOT / f"{dataset_name}_summary.md")
    command = [
        sys.executable,
        "readCaohy/test/diffusion_seed_learning/export_lifecycle_dataset.py",
        "--dataset-name",
        dataset_name,
        "--public-root",
        str(args.public_root),
        "--pointer-out",
        str(pointer_out),
        "--summary-out",
        str(summary_out),
    ]
    for run_dir in run_dirs:
        command.extend(["--run-dir", str(run_dir)])
    print(json.dumps({
        "dataset_name": dataset_name,
        "collection_summaries": [str(path) for path in summary_paths],
        "run_dir_count": len(run_dirs),
        "pointer_out": str(pointer_out),
        "summary_out": str(summary_out),
        "command": command,
    }, ensure_ascii=False, indent=2))
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
