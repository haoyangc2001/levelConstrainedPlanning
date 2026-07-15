#!/usr/bin/env python3
"""Check external dataset/checkpoint/report pointers without copying large files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _path_items(artifacts: dict[str, Any]) -> list[tuple[str, str | None, bool]]:
    return [
        ("training_dataset", artifacts.get("dataset", {}).get("training_dataset", {}).get("path"), True),
        ("dataset_manifest", artifacts.get("dataset", {}).get("dataset_manifest", {}).get("path"), True),
        ("dataset_summary", artifacts.get("dataset", {}).get("dataset_summary", {}).get("path"), False),
        ("validator_report", artifacts.get("dataset", {}).get("validator_report", {}).get("path"), False),
        ("diffusion_best_checkpoint", artifacts.get("diffusion", {}).get("best_checkpoint"), True),
        ("diffusion_metadata", artifacts.get("diffusion", {}).get("metadata"), False),
        ("critic_best_checkpoint", artifacts.get("critic", {}).get("best_checkpoint"), True),
        ("critic_metadata", artifacts.get("critic", {}).get("metadata"), False),
        ("generated_samples", artifacts.get("generated_samples", {}).get("path"), False),
        ("offline_generation_report", artifacts.get("offline_generation_report", {}).get("path"), False),
    ]


def build_report(pointer: Path, strict: bool = False) -> dict[str, Any]:
    artifacts = json.loads(pointer.read_text(encoding="utf-8"))
    checks = []
    for name, value, required in _path_items(artifacts):
        path = Path(value) if value else None
        exists = bool(path and path.exists())
        checks.append({
            "name": name,
            "path": str(path) if path else None,
            "exists": exists,
            "required": bool(required),
            "size_bytes": path.stat().st_size if exists and path.is_file() else None,
        })
    ok = all(item["exists"] for item in checks if item["required"]) if strict else True
    missing = [item for item in checks if item["required"] and not item["exists"]]
    return {
        "schema_version": "standalone_level_planning.artifact_check.v1",
        "ok": ok,
        "strict": bool(strict),
        "pointer": str(pointer),
        "public_data_root": "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning",
        "missing_required": missing,
        "checks": checks,
        "sync_hint": (
            "If required files are missing, sync the phase10 dataset/checkpoints/reports "
            "to /pub/data/caohy/tashan_Manipulation/diffusionSeedLearning and rerun this check."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, default=Path("artifacts/current_artifacts.json"))
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    report = build_report(args.pointer, strict=args.strict)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

