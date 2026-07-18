#!/usr/bin/env python3
"""Train a smoke diffusion baseline for trajectory seeds."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import DEFAULT_VALIDATED_SAMPLES, TrajectorySeedDataset
from diffusion import AuxLossConfig, GaussianDiffusion1D
from model_unet1d import TemporalUNet1D
from artifact_paths import public_root
from level_planner_core.condition import CONDITION_DIM_WITH_CLASS


DEFAULT_CHECKPOINT_DIR = public_root() / "checkpoints/standalone_sr5_diffusion"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_VALIDATED_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--diffusion-steps", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    # C3b switchable auxiliary loss components (OFF by default => pure MSE, the
    # main model enables L_level; C5 ablates each one back off for a variant).
    parser.add_argument("--enable-l-level", action="store_true",
                        help="Add differentiable alignment-deviation penalty (L_level).")
    parser.add_argument("--enable-l-collision", action="store_true",
                        help="Add collision-cost guidance penalty (L_collision; needs A1).")
    parser.add_argument("--level-weight", type=float, default=0.1,
                        help="Weight of L_level when enabled (deg^2 penalty scale).")
    parser.add_argument("--collision-weight", type=float, default=0.1,
                        help="Weight of L_collision when enabled.")
    parser.add_argument("--planner-config", type=Path, default=Path("configs/sr5_level.yaml"),
                        help="Planner config used to build FK/collision callbacks for aux losses.")
    return parser.parse_args()


def build_aux(args, dataset, device) -> AuxLossConfig | None:
    """Construct the C3b AuxLossConfig, or None when no aux term is enabled.

    Only when a term is enabled do we import curobo / instantiate the planner,
    so pure-MSE training keeps zero curobo dependency."""
    if not (args.enable_l_level or args.enable_l_collision):
        return None
    from level_planner_core.planner import LevelConstrainedPlanner
    from aux_guidance import make_alignment_angle_fn, make_collision_cost_fn

    planner = LevelConstrainedPlanner.from_config(str(args.planner_config))
    # move normalization to the training device for the denormaliser
    mean = dataset.normalization.mean.to(device)
    std = dataset.normalization.std.clamp_min(1e-6).to(device)

    def denormalize(x: torch.Tensor) -> torch.Tensor:
        return x * std + mean

    tolerance_deg = float(getattr(planner.config, "level_tolerance_deg", 3.0))
    level_index = 15 if dataset.condition_dim == CONDITION_DIM_WITH_CLASS else None
    aux = AuxLossConfig(
        enable_l_level=bool(args.enable_l_level),
        enable_l_collision=bool(args.enable_l_collision),
        level_weight=float(args.level_weight) if args.enable_l_level else 0.0,
        collision_weight=float(args.collision_weight) if args.enable_l_collision else 0.0,
        tolerance_deg=tolerance_deg,
        level_active_index=level_index,
        denormalize=denormalize,
        alignment_angle_fn=make_alignment_angle_fn(planner) if args.enable_l_level else None,
        collision_cost_fn=make_collision_cost_fn(planner) if args.enable_l_collision else None,
    )
    return aux


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = TrajectorySeedDataset(args.samples, horizon=args.horizon, positive_only=True)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)
    model_config = {
        "dof": dataset.dof,
        "condition_dim": dataset.condition_dim,
        "hidden_dim": args.hidden_dim,
    }
    diffusion_config = {"steps": args.diffusion_steps}
    model = TemporalUNet1D(**model_config).to(device)
    diffusion = GaussianDiffusion1D(**diffusion_config).to(device)
    aux = build_aux(args, dataset, device)
    aux_config_record = {
        "enable_l_level": bool(args.enable_l_level),
        "enable_l_collision": bool(args.enable_l_collision),
        "level_weight": float(args.level_weight) if args.enable_l_level else 0.0,
        "collision_weight": float(args.collision_weight) if args.enable_l_collision else 0.0,
        "tolerance_deg": float(aux.tolerance_deg) if aux else None,
        "active": bool(aux.active) if aux else False,
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best_loss = float("inf")
    history = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        losses = []
        comp_accum: dict[str, float] = {}
        model.train()
        for batch in loader:
            x0 = batch["trajectory"].to(device)
            condition = batch["condition"].to(device)
            loss, components = diffusion.training_loss(
                model, x0, condition, aux, return_components=True
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            for key, value in components.items():
                comp_accum[key] = comp_accum.get(key, 0.0) + value
        epoch_loss = sum(losses) / max(len(losses), 1)
        n_batches = max(len(losses), 1)
        comp_mean = {key: value / n_batches for key, value in comp_accum.items()}
        history.append({"epoch": epoch, "loss": epoch_loss, "components": comp_mean})
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "diffusion_config": diffusion_config,
            "normalization": dataset.normalization.to_json(),
            "horizon": args.horizon,
            "samples_path": str(args.samples),
            "samples_sha256": sha256_file(args.samples),
            "aux_loss_config": aux_config_record,
            "schema_version": "diffusion_seed_checkpoint.v1",
        }
        torch.save(checkpoint, args.out_dir / "last.pt")
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(checkpoint, args.out_dir / "best.pt")
    metadata = {
        "schema_version": "diffusion_seed_checkpoint_metadata.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.out_dir.name,
        "samples_path": str(args.samples),
        "samples_sha256": sha256_file(args.samples),
        "sample_count": len(dataset),
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": str(device),
        },
        "best_checkpoint": str(args.out_dir / "best.pt"),
        "last_checkpoint": str(args.out_dir / "last.pt"),
        "horizon": args.horizon,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "hidden_dim": args.hidden_dim,
        "diffusion_steps": args.diffusion_steps,
        "best_loss": best_loss,
        "history": history,
        "aux_loss_config": aux_config_record,
        "profile": "sr5",
        "constraint": "tool1 y+ -> world z-",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
