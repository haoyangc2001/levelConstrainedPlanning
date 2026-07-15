#!/usr/bin/env python3
"""Evaluate success critic candidate filtering strategies."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from critic import (
    CriticNormalization,
    SuccessCriticDataset,
    SuccessCriticMLP,
    score_candidates,
    select_diverse_topk,
)
from dataset import DEFAULT_VALIDATED_SAMPLES
from train_critic import DEFAULT_CRITIC_DIR


DEFAULT_JSON_OUT = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/"
    "critic_ablation_report.json"
)
DEFAULT_MD_OUT = Path("readCaohy/plans/diffusionSeedLearning/critic_ablation_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CRITIC_DIR / "best.pt")
    parser.add_argument("--samples", type=Path, default=DEFAULT_VALIDATED_SAMPLES)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--k-select", type=int, default=8)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    parser.add_argument("--device", type=str, default="cuda:0")
    return parser.parse_args()


def _strategy_summary(name: str, groups: dict[str, list[int]], selected: dict[str, list[int]], rows: list[dict[str, Any]]) -> dict:
    row_by_index = {int(row["index"]): row for row in rows}
    request_count = len(groups)
    successes = 0
    selected_count = 0
    selected_positive = 0
    fixed_budget_ms_values = []
    source_counts = defaultdict(int)
    for key, indices in groups.items():
        chosen = selected.get(key, [])
        selected_count += len(chosen)
        positives = [idx for idx in chosen if row_by_index[idx]["positive_for_critic"]]
        if positives:
            successes += 1
        selected_positive += len(positives)
        budget = sum(max(float(row_by_index[idx]["expected_opt_time_ms"]), 0.0) for idx in chosen)
        fixed_budget_ms_values.append(budget)
        for idx in chosen:
            source_counts[str(row_by_index[idx]["source_label"])] += 1
    return {
        "strategy": name,
        "request_count": int(request_count),
        "selected_count": int(selected_count),
        "selected_positive_count": int(selected_positive),
        "success_at_k": float(successes / max(request_count, 1)),
        "mean_selected_per_request": float(selected_count / max(request_count, 1)),
        "mean_fixed_budget_ms_proxy": float(sum(fixed_budget_ms_values) / max(len(fixed_budget_ms_values), 1)),
        "source_label_distribution": dict(source_counts),
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    normalization = CriticNormalization.from_json(checkpoint["normalization"])
    dataset = SuccessCriticDataset(args.samples, horizon=args.horizon, normalization=normalization)
    model = SuccessCriticMLP(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    rows = score_candidates(model, dataset, device)

    groups: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        groups[str(row["request_key"])].append(int(row["index"]))

    k = int(args.k_select)
    no_critic = {
        key: indices[:k]
        for key, indices in groups.items()
    }
    critic_topk = {
        key: sorted(indices, key=lambda idx: rows[idx]["quality_score"], reverse=True)[:k]
        for key, indices in groups.items()
    }
    critic_diverse = {
        key: select_diverse_topk(
            indices,
            rows,
            dataset,
            k=k,
            diversity_weight=float(args.diversity_weight),
        )
        for key, indices in groups.items()
    }

    report = {
        "schema_version": "success_critic_ablation_report.v1",
        "checkpoint": str(args.checkpoint),
        "samples": str(args.samples),
        "k_select": k,
        "diversity_weight": float(args.diversity_weight),
        "label_note": "expected_opt_time_ms is a proxy in phase 7; replace with benchmark timing in phase 8.",
        "strategies": [
            _strategy_summary("no_critic_order", groups, no_critic, rows),
            _strategy_summary("critic_topk", groups, critic_topk, rows),
            _strategy_summary("critic_diversity", groups, critic_diverse, rows),
        ],
        "scored_candidates": rows,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "<!-- [caohy] diffusionSeedLearning phase 7 critic ablation report -->",
        "# Success Critic Ablation Report",
        "",
        f"- checkpoint: `{args.checkpoint}`",
        f"- samples: `{args.samples}`",
        f"- k_select: `{k}`",
        f"- diversity_weight: `{float(args.diversity_weight):.3f}`",
        "",
        "| Strategy | success@K | selected positives | mean budget proxy ms |",
        "|---|---:|---:|---:|",
    ]
    for item in report["strategies"]:
        lines.append(
            f"| {item['strategy']} | {item['success_at_k']:.3f} | "
            f"{item['selected_positive_count']} | {item['mean_fixed_budget_ms_proxy']:.2f} |"
        )
    lines += [
        "",
        "当前数据集中没有真实 collision label 和 optimize-time label；collision_risk 为结构占位，expected_opt_time_ms 为 proxy。phase 8 benchmark 产出真实 fixed-budget timing 后应替换该报告口径。",
        "",
    ]
    args.md_out.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "scored_candidates"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
