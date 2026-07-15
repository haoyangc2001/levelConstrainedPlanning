#!/usr/bin/env python3
"""Reload and summarize an exported diffusion lifecycle dataset."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/"
            "datasets/sr5_phase2_20260713_lifecycle_baseline/manifest.json"
        ),
    )
    parser.add_argument("--json-out", type=Path, default=None)
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


def summarize(manifest: dict[str, Any], samples: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = collections.Counter(sample.get("split") for sample in samples)
    sample_type_counts = collections.Counter(sample.get("sample_type") for sample in samples)
    status_counts = collections.Counter(
        sample.get("labels", {}).get("planner_status") for sample in samples
    )
    failure_counts = collections.Counter(
        sample.get("labels", {}).get("failure_reason") or "none"
        for sample in samples
    )
    source_label_counts = collections.Counter(
        sample.get("source_lineage", {}).get("source_label") or "none"
        for sample in samples
    )
    source_type_counts = collections.Counter(
        sample.get("source_lineage", {}).get("source_type") or "none"
        for sample in samples
    )
    positive_for_diffusion = sum(
        1 for sample in samples
        if bool(sample.get("labels", {}).get("positive_for_diffusion"))
    )
    positive_for_critic = sum(
        1 for sample in samples
        if bool(sample.get("labels", {}).get("positive_for_critic"))
    )
    return {
        "schema_version": "dataset_reload_summary.v1",
        "dataset_name": manifest.get("dataset_name"),
        "manifest_path": manifest.get("manifest_path"),
        "samples_path": manifest.get("samples_path"),
        "sample_count": len(samples),
        "split_counts": dict(sorted(split_counts.items())),
        "sample_type_counts": dict(sorted(sample_type_counts.items())),
        "planner_status_counts": dict(sorted(status_counts.items())),
        "failure_reason_counts": dict(sorted(failure_counts.items())),
        "source_label_counts": dict(sorted(source_label_counts.items())),
        "source_type_counts": dict(sorted(source_type_counts.items())),
        "positive_for_diffusion": positive_for_diffusion,
        "positive_for_critic": positive_for_critic,
    }


def main() -> None:
    args = parse_args()
    manifest = read_json(args.manifest)
    samples_path = Path(manifest["samples_path"])
    samples = load_samples(samples_path)
    summary = summarize(manifest, samples)
    text = json.dumps(summary, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
