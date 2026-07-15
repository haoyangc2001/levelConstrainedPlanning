#!/usr/bin/env python3
"""Train the phase-7 success critic smoke baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from critic import SuccessCriticDataset, SuccessCriticMLP
from dataset import DEFAULT_VALIDATED_SAMPLES
from artifact_paths import public_root


DEFAULT_CRITIC_DIR = public_root() / "checkpoints/standalone_sr5_success_critic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=Path, default=DEFAULT_VALIDATED_SAMPLES)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CRITIC_DIR)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = SuccessCriticDataset(args.samples, horizon=args.horizon)
    loader = DataLoader(dataset, batch_size=min(args.batch_size, len(dataset)), shuffle=True)
    model = SuccessCriticMLP(dataset.input_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    positives = int(torch.sum(dataset.success).item())
    negatives = int(len(dataset) - positives)
    pos_weight = torch.tensor(
        [max(float(negatives) / max(float(positives), 1.0), 1.0)],
        device=device,
    )
    best_loss = float("inf")
    history = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        success_losses = []
        regression_losses = []
        for batch in loader:
            feature = batch["feature"].to(device)
            success = batch["success"].to(device)
            regression = batch["regression"].to(device)
            output = model(feature)
            success_loss = F.binary_cross_entropy_with_logits(
                output["success_logit"],
                success,
                pos_weight=pos_weight,
            )
            regression_loss = F.smooth_l1_loss(output["regression"], regression)
            loss = success_loss + 0.35 * regression_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.item()))
            success_losses.append(float(success_loss.item()))
            regression_losses.append(float(regression_loss.item()))
        epoch_record = {
            "epoch": int(epoch),
            "loss": sum(losses) / max(len(losses), 1),
            "success_loss": sum(success_losses) / max(len(success_losses), 1),
            "regression_loss": sum(regression_losses) / max(len(regression_losses), 1),
        }
        history.append(epoch_record)
        checkpoint = {
            "schema_version": "success_critic_checkpoint.v1",
            "model_state_dict": model.state_dict(),
            "model_config": {
                "input_dim": dataset.input_dim,
                "hidden_dim": args.hidden_dim,
            },
            "normalization": dataset.normalization.to_json(),
            "samples_path": str(args.samples),
            "horizon": args.horizon,
            "label_schema": {
                "success": "labels.positive_for_critic",
                "alignment_risk": "candidate.metrics.max_alignment_deviation_deg / task.alignment.tolerance_deg",
                "collision_risk": "constant_zero_until_collision_replay_labels_exist",
                "joint_jump_risk": "candidate.metrics.joint_step_max_l2 / 1.5",
                "expected_opt_time_ms": "proxy: 50 + 1.5 * point_count + 500 * joint_step_jump_cost + 25 if negative",
            },
        }
        torch.save(checkpoint, args.out_dir / "last.pt")
        if epoch_record["loss"] < best_loss:
            best_loss = epoch_record["loss"]
            torch.save(checkpoint, args.out_dir / "best.pt")

    source_counts = Counter(
        sample.get("candidate", {}).get("source_type") for sample in dataset.samples
    )
    metadata = {
        "schema_version": "success_critic_metadata.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_name": args.out_dir.name,
        "samples_path": str(args.samples),
        "sample_count": len(dataset),
        "positive_count": positives,
        "negative_count": negatives,
        "source_type_counts": dict(source_counts),
        "horizon": args.horizon,
        "epochs": args.epochs,
        "best_loss": best_loss,
        "history": history,
        "label_schema": checkpoint["label_schema"],
        "profile": "sr5",
        "constraint": "tool1 y+ -> world z-",
        "note": "Smoke critic baseline. Collision and optimize-time labels are placeholders/proxies until phase 8 benchmark records real values.",
    }
    (args.out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
