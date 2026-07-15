#!/usr/bin/env python3
"""Release checklist for the closed-loop baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.check_artifacts import build_report as build_artifact_report


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_JSON_OUT = Path("reports/release_v0_2_0_closed_loop_checklist.json")
DEFAULT_MD_OUT = Path("reports/release_v0_2_0_closed_loop_summary.md")


def _git_output(args: list[str]) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _tracked_files() -> list[Path]:
    raw = subprocess.check_output(["git", "ls-files", "-z"], cwd=REPO_ROOT)
    return [REPO_ROOT / item.decode() for item in raw.split(b"\0") if item]


def _large_or_forbidden_files(max_bytes: int) -> list[dict[str, Any]]:
    forbidden_suffixes = {".pt", ".pth"}
    findings: list[dict[str, Any]] = []
    for path in _tracked_files():
        if not path.exists() or not path.is_file():
            continue
        rel = str(path.relative_to(REPO_ROOT))
        size = path.stat().st_size
        suffix = path.suffix.lower()
        if size > max_bytes or suffix in forbidden_suffixes:
            findings.append(
                {
                    "path": rel,
                    "size_bytes": size,
                    "reason": "forbidden_checkpoint_suffix" if suffix in forbidden_suffixes else "large_tracked_file",
                }
            )
    return findings


def _contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def build_checklist(args: argparse.Namespace) -> dict[str, Any]:
    artifact_report = build_artifact_report(args.pointer, strict=True)
    large_findings = _large_or_forbidden_files(int(args.max_tracked_bytes))
    project_mainline = REPO_ROOT / "docs/guides/project_mainline.md"
    entry_checks = {
        "python_api_exports_core": _contains(REPO_ROOT / "level_planner_core/__init__.py", "LevelConstrainedPlanner"),
        "cli_uses_core": _contains(REPO_ROOT / "level_planner/cli.py", "LevelConstrainedPlanner"),
        "ros_adapter_uses_core": _contains(REPO_ROOT / "level_planner_ros/planner_node.py", "LevelConstrainedPlanner"),
    }
    docs_checks = {
        "project_mainline_mentions_closed_loop": _contains(project_mainline, "data generation -> model learning -> optimization validation -> failure fallback -> data update"),
        "project_mainline_mentions_phase8_baseline": _contains(project_mainline, "Phase 8"),
    }
    pointer = json.loads(args.pointer.read_text(encoding="utf-8"))
    benchmark = pointer.get("closed_loop_benchmark") or {}
    checklist = {
        "schema_version": "closed_loop_release_checklist.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pointer": str(args.pointer),
        "artifact_check": artifact_report,
        "large_or_forbidden_tracked_files": large_findings,
        "docs_checks": docs_checks,
        "entry_checks": entry_checks,
        "benchmark_summary": benchmark.get("summary"),
        "dataset": pointer.get("dataset", {}),
        "diffusion": pointer.get("diffusion", {}),
        "critic": pointer.get("critic", {}),
    }
    checklist["passed"] = (
        bool(artifact_report.get("ok"))
        and not large_findings
        and all(docs_checks.values())
        and all(entry_checks.values())
        and bool((benchmark.get("summary") or {}).get("exists"))
    )
    return checklist


def write_summary(checklist: dict[str, Any], out: Path, tag: str) -> None:
    dataset = checklist.get("dataset") or {}
    diffusion = checklist.get("diffusion") or {}
    critic = checklist.get("critic") or {}
    benchmark = checklist.get("benchmark_summary") or {}
    lines = [
        f"# Release {tag}",
        "",
        "## Baseline",
        "",
        "- System mainline: `data generation -> model learning -> optimization validation -> failure fallback -> data update`",
        f"- Dataset: `{dataset.get('name')}`",
        f"- Dataset samples: `{dataset.get('sample_count')}`",
        f"- Diffusion checkpoint: `{diffusion.get('best_checkpoint')}`",
        f"- Critic checkpoint: `{critic.get('best_checkpoint')}`",
        f"- Benchmark summary: `{benchmark.get('path')}`",
        "",
        "## Checklist",
        "",
        f"- Artifact pointers valid: `{checklist.get('artifact_check', {}).get('ok')}`",
        f"- No large/checkpoint files tracked: `{not checklist.get('large_or_forbidden_tracked_files')}`",
        f"- Docs aligned: `{all((checklist.get('docs_checks') or {}).values())}`",
        f"- CLI/API/ROS use core: `{all((checklist.get('entry_checks') or {}).values())}`",
        "",
        "## Known Limits",
        "",
        "- Phase 8 model is a closed-loop smoke baseline trained from a very small standalone dataset.",
        "- Diffusion-only and diffusion+critic did not outperform rule-only in the first CuRobo benchmark.",
        "- Collision replay is still recorded as transparent `unchecked`; hard validator has joint, alignment, continuity, and goal checks.",
        "- Larger datasets should be generated under `/pub/data/caohy/levelConstrainedPlanning` before claiming model quality improvement.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pointer", type=Path, default=Path("artifacts/current_artifacts.json"))
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--tag", default="v0.2.0-closed-loop-baseline")
    parser.add_argument("--max-tracked-bytes", type=int, default=25 * 1024 * 1024)
    args = parser.parse_args(argv)
    checklist = build_checklist(args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(checklist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_summary(checklist, args.md_out, args.tag)
    print(json.dumps(checklist, ensure_ascii=False, indent=2))
    return 0 if checklist["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
