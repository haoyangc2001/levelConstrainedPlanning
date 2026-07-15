#!/usr/bin/env python3
"""Sample trajectory seeds from the smoke diffusion baseline."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from dataset import DEFAULT_VALIDATED_SAMPLES, Normalization, TrajectorySeedDataset
from diffusion import GaussianDiffusion1D
from model_unet1d import TemporalUNet1D


DEFAULT_CHECKPOINT = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/"
    "sr5_phase4_smoke_baseline/best.pt"
)
DEFAULT_OUT = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/"
    "sr5_phase4_generated_samples.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples", type=Path, default=DEFAULT_VALIDATED_SAMPLES)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tasks", type=int, default=4)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--max-step-l2", type=float, default=1.0)
    parser.add_argument("--joint-abs-limit", type=float, default=2 * math.pi)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def recover_continuity(
    trajectories: torch.Tensor,
    q_start: torch.Tensor,
    max_step_l2: float,
    joint_abs_limit: float,
) -> torch.Tensor:
    recovered = trajectories.clone()
    recovered[:, 0, :] = q_start
    for batch_index in range(recovered.shape[0]):
        for point_index in range(1, recovered.shape[1]):
            prev = recovered[batch_index, point_index - 1]
            current = recovered[batch_index, point_index]
            delta = current - prev
            norm = torch.linalg.norm(delta)
            if float(norm.item()) > float(max_step_l2):
                delta = delta / norm.clamp_min(1e-8) * float(max_step_l2)
            recovered[batch_index, point_index] = prev + delta
    return torch.clamp(recovered, -float(joint_abs_limit), float(joint_abs_limit))


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    dataset = TrajectorySeedDataset(args.samples, horizon=int(checkpoint["horizon"]), positive_only=True)
    model = TemporalUNet1D(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = GaussianDiffusion1D(**checkpoint["diffusion_config"]).to(device)
    normalization = Normalization.from_json(checkpoint["normalization"])
    results = []
    with torch.no_grad():
        for task_index in range(min(args.tasks, len(dataset))):
            item = dataset[task_index]
            condition = item["condition"].to(device).unsqueeze(0).repeat(args.k, 1)
            normalized = diffusion.sample(
                model,
                (args.k, int(checkpoint["horizon"]), int(checkpoint["model_config"]["dof"])),
                condition,
            ).cpu()
            trajectories = normalization.denormalize(normalized)
            q_start = item["raw_trajectory"][0]
            trajectories = recover_continuity(
                trajectories,
                q_start,
                max_step_l2=args.max_step_l2,
                joint_abs_limit=args.joint_abs_limit,
            )
            results.append({
                "task_index": task_index,
                "condition": item["condition"].tolist(),
                "q_start": q_start.tolist(),
                "postprocess": {
                    "start_inpainting": True,
                    "max_step_l2": args.max_step_l2,
                    "joint_abs_limit": args.joint_abs_limit,
                    "continuity_recovery": True,
                },
                "generated": trajectories.tolist(),
                "reference_positive": item["raw_trajectory"].tolist(),
            })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "schema_version": "diffusion_seed_generated_samples.v1",
        "checkpoint": str(args.checkpoint),
        "generated_task_count": len(results),
        "k": args.k,
        "results": results,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
