#!/usr/bin/env python3
"""D5b negative-result diagnostics for the C4 fallback narrative.

Consumes the closed-loop benchmark v2 report plus the optional C4 significance
verdict and emits a compact JSON + Markdown report.  The goal is not to create a
new success claim; it records why the learned branches failed the D1 go/no-go
gate and what narrative remains defensible.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


KEY_FAILURES = (
    "repair_failed",
    "alignment_failed",
    "collision_failed",
    "joint_limit_failed",
    "velocity_acceleration_failed",
    "collision_unchecked",
)


def _load(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _mean(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _std(values: list[float]) -> float | None:
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _method_groups(report: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for summary in report.get("summaries") or []:
        method = str(summary.get("method") or summary.get("strategy") or "unknown")
        groups[method].append(summary)
    return dict(groups)


def _sum_counters(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update({str(k): int(v) for k, v in (row.get(key) or {}).items()})
    return dict(sorted(counter.items()))


def _failure_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = _sum_counters(rows, "validator_failure_counts")
    total_events = sum(totals.values())
    selected = {
        name: {
            "count": int(totals.get(name, 0)),
            "share_of_validator_events": (
                None if total_events <= 0 else round(float(totals.get(name, 0)) / float(total_events), 6)
            ),
        }
        for name in KEY_FAILURES
    }
    return {
        "total_validator_events": int(total_events),
        "key_failures": selected,
        "all_validator_failure_counts": totals,
        "status_counts": _sum_counters(rows, "status_counts"),
        "failure_reason_counts": _sum_counters(rows, "failure_reason_counts"),
    }


def _latency_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in ("p50", "p75", "p95", "p98", "mean"):
        values = [
            float((row.get("latency_ms") or {}).get(field))
            for row in rows
            if isinstance((row.get("latency_ms") or {}).get(field), (int, float))
        ]
        out[field] = {
            "mean_ms": _round(_mean(values), 3),
            "std_ms": _round(_std(values), 3),
        }
    return out


def _constraint_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for field in (
        "selected_alignment_deviation_deg",
        "candidate_alignment_deviation_deg",
        "collision_min_distance_m",
        "max_jerk_rad_s3",
        "motion_time_sec",
    ):
        means = []
        counts = []
        for row in rows:
            metric = ((row.get("constraint_error") or {}).get(field)) or {}
            if isinstance(metric.get("mean"), (int, float)):
                means.append(float(metric["mean"]))
            if isinstance(metric.get("count"), int):
                counts.append(int(metric["count"]))
        out[field] = {
            "mean_of_cell_means": _round(_mean(means), 6),
            "total_count": int(sum(counts)),
        }
    return out


def _success_block(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed = [
        float(row.get("fixed_budget_success_rate"))
        for row in rows
        if isinstance(row.get("fixed_budget_success_rate"), (int, float))
    ]
    success_at_k = [
        float(row.get("success_at_k_rate"))
        for row in rows
        if isinstance(row.get("success_at_k_rate"), (int, float))
    ]
    return {
        "fixed_budget_success_rate_mean": _round(_mean(fixed), 6),
        "fixed_budget_success_rate_std": _round(_std(fixed), 6),
        "success_at_k_rate_mean": _round(_mean(success_at_k), 6),
        "success_at_k_rate_std": _round(_std(success_at_k), 6),
        "success_source_counts": _sum_counters(rows, "success_source_counts"),
    }


def analyze(report: dict[str, Any], significance: dict[str, Any] | None = None) -> dict[str, Any]:
    groups = _method_groups(report)
    method_blocks: dict[str, Any] = {}
    for method, rows in sorted(groups.items()):
        method_blocks[method] = {
            "n_cells": len(rows),
            "success": _success_block(rows),
            "latency_ms": _latency_block(rows),
            "failures": _failure_block(rows),
            "constraints": _constraint_block(rows),
        }

    verdicts = {
        str(item.get("method")): item
        for item in (significance or {}).get("verdicts", [])
        if isinstance(item, dict)
    }
    learned_methods = [m for m in method_blocks if m != "rule_only"]
    rule_rate = (method_blocks.get("rule_only") or {}).get("success", {}).get(
        "fixed_budget_success_rate_mean"
    )
    gap_to_rule = {}
    for method in learned_methods:
        rate = method_blocks[method]["success"]["fixed_budget_success_rate_mean"]
        gap_to_rule[method] = None if rate is None or rule_rate is None else _round(rate - rule_rate, 6)

    mixed_sources = (method_blocks.get("mixed_fallback") or {}).get("success", {}).get("success_source_counts", {})
    rule_sources = (method_blocks.get("rule_only") or {}).get("success", {}).get("success_source_counts", {})
    diagnosis = [
        "Pure learned branches produced no hard-valid selected trajectory in C4.",
        "Mixed fallback recovered only through fallback, not through learned-only success.",
        "The pre-registered C4 decision rule did not show any learned method superior to rule_only.",
        "The defensible paper narrative is therefore a verified integration/fallback architecture plus negative diagnostics, not a self-improving performance claim.",
    ]
    if mixed_sources:
        diagnosis.append(f"mixed_fallback success sources: {mixed_sources}")
    if rule_sources:
        diagnosis.append(f"rule_only success sources: {rule_sources}")

    return {
        "schema_version": "d5b_negative_diagnostics.v1",
        "source_report": report.get("out_dir") or "runs/c4_test_eval/result.json",
        "eval_split": report.get("eval_split"),
        "request_count": report.get("request_count"),
        "k_values": report.get("k_values"),
        "budget_values": report.get("budget_values"),
        "budget_semantics": report.get("budget_semantics"),
        "methods": method_blocks,
        "gap_to_rule_fixed_budget_success_rate": gap_to_rule,
        "c4_verdicts": verdicts,
        "diagnosis": diagnosis,
        "paper_implication": {
            "claim_status": "self_improving_not_supported",
            "recommended_narrative": "fallback_narrative_with_failure_taxonomy",
            "skip_full_recompute": True,
        },
    }


def _write_markdown(result: dict[str, Any], path: Path) -> None:
    lines = [
        "# D5b Negative Diagnostics",
        "",
        f"- eval split: `{result.get('eval_split')}`",
        f"- request count: `{result.get('request_count')}`",
        f"- budget semantics: `{result.get('budget_semantics')}`",
        "",
        "## Success Summary",
        "",
        "| Method | Success mean | Success std | Gap to rule | P50 ms | P98 ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    gaps = result.get("gap_to_rule_fixed_budget_success_rate") or {}
    for method, block in result.get("methods", {}).items():
        success = block["success"]
        lat = block["latency_ms"]
        lines.append(
            "| {method} | {rate} | {std} | {gap} | {p50} | {p98} |".format(
                method=method,
                rate=success.get("fixed_budget_success_rate_mean"),
                std=success.get("fixed_budget_success_rate_std"),
                gap=gaps.get(method, 0.0 if method == "rule_only" else None),
                p50=lat.get("p50", {}).get("mean_ms"),
                p98=lat.get("p98", {}).get("mean_ms"),
            )
        )
    lines += [
        "",
        "## Key Failure Events",
        "",
        "| Method | Repair | Alignment | Collision | Joint Limit | Vel/Accel |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method, block in result.get("methods", {}).items():
        failures = block["failures"]["key_failures"]
        lines.append(
            "| {method} | {repair} | {align} | {coll} | {joint} | {va} |".format(
                method=method,
                repair=failures["repair_failed"]["count"],
                align=failures["alignment_failed"]["count"],
                coll=failures["collision_failed"]["count"],
                joint=failures["joint_limit_failed"]["count"],
                va=failures["velocity_acceleration_failed"]["count"],
            )
        )
    lines += ["", "## Diagnosis", ""]
    lines += [f"- {item}" for item in result.get("diagnosis", [])]
    lines += ["", "## Paper Implication", ""]
    for key, value in (result.get("paper_implication") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D5b negative-result diagnostics.")
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--significance", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    args = parser.parse_args(argv)

    result = analyze(_load(args.benchmark), _load(args.significance) if args.significance else {})
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(result, args.md_out)
    print(json.dumps({
        "json_out": str(args.json_out),
        "md_out": str(args.md_out),
        "claim_status": result["paper_implication"]["claim_status"],
        "skip_full_recompute": result["paper_implication"]["skip_full_recompute"],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
