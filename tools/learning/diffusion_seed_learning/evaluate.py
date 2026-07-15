#!/usr/bin/env python3
"""Evaluate generated diffusion seeds against simple offline baselines."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch


DEFAULT_GENERATED = Path("runs/diffusion_seed_learning/generated_samples_smoke.json")
DEFAULT_JSON_OUT = Path("runs/diffusion_seed_learning/offline_generation_report.json")
DEFAULT_MD_OUT = Path("runs/diffusion_seed_learning/diffusion_vs_rule_seed_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated", type=Path, default=DEFAULT_GENERATED)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--joint-step-max-l2", type=float, default=1.5)
    parser.add_argument("--joint-abs-limit", type=float, default=2 * math.pi)
    parser.add_argument("--random-seed", type=int, default=20260715)
    return parser.parse_args()


def precheck(traj: torch.Tensor, q_start: torch.Tensor, args: argparse.Namespace) -> dict:
    finite = bool(torch.isfinite(traj).all().item())
    start_gap = float(torch.linalg.norm(traj[0] - q_start).item())
    steps = torch.linalg.norm(traj[1:] - traj[:-1], dim=-1)
    max_step = float(steps.max().item()) if steps.numel() else 0.0
    max_abs = float(torch.abs(traj).max().item())
    valid = (
        finite
        and start_gap <= 1e-5
        and max_step <= args.joint_step_max_l2
        and max_abs <= args.joint_abs_limit
    )
    return {
        "valid": bool(valid),
        "finite": finite,
        "start_gap_l2": start_gap,
        "joint_step_max_l2": max_step,
        "joint_abs_max": max_abs,
    }


def summarize_checks(checks: list[dict]) -> dict:
    if not checks:
        return {"count": 0, "valid_count": 0, "valid_ratio": 0.0}
    return {
        "count": len(checks),
        "valid_count": sum(1 for item in checks if item["valid"]),
        "valid_ratio": sum(1 for item in checks if item["valid"]) / len(checks),
        "max_joint_step_l2": max(item["joint_step_max_l2"] for item in checks),
        "max_start_gap_l2": max(item["start_gap_l2"] for item in checks),
    }


def main() -> None:
    args = parse_args()
    generator = torch.Generator().manual_seed(int(args.random_seed))
    generated = json.loads(args.generated.read_text(encoding="utf-8"))
    diffusion_checks = []
    rule_replay_checks = []
    random_checks = []
    for result in generated["results"]:
        q_start = torch.tensor(result["q_start"], dtype=torch.float32)
        reference = torch.tensor(result["reference_positive"], dtype=torch.float32)
        rule_replay_checks.append(precheck(reference, q_start, args))
        for traj_points in result["generated"]:
            traj = torch.tensor(traj_points, dtype=torch.float32)
            diffusion_checks.append(precheck(traj, q_start, args))
            random_traj = torch.randn(
                traj.shape,
                dtype=traj.dtype,
                generator=generator,
            ) * 0.5
            random_traj[0] = q_start
            random_checks.append(precheck(random_traj, q_start, args))
    report = {
        "schema_version": "offline_generation_report.v1",
        "generated": str(args.generated),
        "precheck_definition": {
            "joint_step_max_l2": args.joint_step_max_l2,
            "joint_abs_limit": args.joint_abs_limit,
            "start_gap_l2": 1e-5,
            "note": "Standalone diffusion seed precheck; CuRobo repair and hard FK/collision validation remain mandatory.",
        },
        "diffusion_seed": summarize_checks(diffusion_checks),
        "rule_positive_replay": summarize_checks(rule_replay_checks),
        "random_seed": summarize_checks(random_checks),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(
        "\n".join([
            "<!-- standaloneLevelPlanning diffusion seed precheck report -->",
            "# Diffusion Seed Precheck Report",
            "",
            f"- generated: `{args.generated}`",
            f"- diffusion valid ratio: `{report['diffusion_seed']['valid_ratio']:.3f}`",
            f"- rule positive replay valid ratio: `{report['rule_positive_replay']['valid_ratio']:.3f}`",
            f"- random valid ratio: `{report['random_seed']['valid_ratio']:.3f}`",
            "",
            "该报告只证明训练/采样/预检工具链可运行，不替代 CuRobo repair 与 hard validation；Phase 8 的真实验收以 closed-loop CuRobo benchmark 为准。",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
