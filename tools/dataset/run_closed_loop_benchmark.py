#!/usr/bin/env python3
"""Run a fixed-budget closed-loop benchmark through the CuRobo planner core."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import time
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.dataset.run_lifecycle_batch import load_requests, request_output_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "closed_loop_curobo_benchmark.v1"


STRATEGY_ORDER = [
    "rule_only",
    "diffusion_only",
    "diffusion_critic",
    "mixed_fallback",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    ratio = position - lower
    return ordered[lower] * (1.0 - ratio) + ordered[upper] * ratio


def _round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(float(value), digits)


def strategy_request(request: dict[str, Any], strategy: str, total_budget_ms: float) -> dict[str, Any]:
    if strategy not in STRATEGY_ORDER:
        raise ValueError(f"unsupported strategy: {strategy}")
    updated = deepcopy(request)
    seed_policy = dict(updated.get("seed_policy") or {})
    metadata = dict(updated.get("metadata") or {})
    metadata["total_budget_ms"] = float(total_budget_ms)
    metadata["benchmark_strategy"] = strategy
    if strategy == "rule_only":
        seed_policy.update(
            {
                "mode": "rule",
                "k_generate": 2,
                "k_accept": 2,
                "fallback_to_rule_seed": True,
                "fallback_to_planner_native": False,
                "timeout_sec": 0.5,
            }
        )
    elif strategy == "diffusion_only":
        seed_policy.update(
            {
                "mode": "diffusion",
                "k_generate": 4,
                "k_accept": 2,
                "fallback_to_rule_seed": False,
                "fallback_to_planner_native": False,
                "timeout_sec": 2.0,
            }
        )
    elif strategy == "diffusion_critic":
        seed_policy.update(
            {
                "mode": "diffusion",
                "k_generate": 6,
                "k_accept": 2,
                "fallback_to_rule_seed": False,
                "fallback_to_planner_native": False,
                "timeout_sec": 2.0,
            }
        )
    elif strategy == "mixed_fallback":
        seed_policy.update(
            {
                "mode": "mixed",
                "k_generate": 6,
                "k_accept": 2,
                "fallback_to_rule_seed": True,
                "fallback_to_planner_native": True,
                "timeout_sec": 2.0,
            }
        )
    updated["seed_policy"] = seed_policy
    updated["metadata"] = metadata
    updated["request_id"] = f"{request.get('request_id', 'request')}_{strategy}"
    return updated


def _validator_failure_counts(result: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for candidate in result.get("candidate_records") or []:
        validator = candidate.get("validator_metrics") or {}
        checks = validator.get("checks") or {}
        if not validator.get("valid"):
            reason = (
                validator.get("failure_reason")
                or candidate.get("failure_reason")
                or (candidate.get("labels") or {}).get("failure_reason")
                or "unknown"
            )
            counts[str(reason)] += 1
        for name in ("alignment", "joint_limit", "velocity_acceleration"):
            check = checks.get(name) or {}
            if check and not check.get("valid", False):
                counts[f"{name}_failed"] += 1
        collision = checks.get("collision_safety") or {}
        status = collision.get("status")
        if status == "unchecked":
            counts["collision_unchecked"] += 1
        elif status == "no_obstacles":
            counts["collision_no_obstacles"] += 1
        elif collision.get("valid", False):
            # A1.5: real (non-degenerate) collision check that passed.
            counts["collision_checked_ok"] += 1
        elif collision:
            counts["collision_failed"] += 1
    return counts


def _candidate_source_counts(result: dict[str, Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for candidate in result.get("candidate_records") or []:
        counter[str((candidate.get("source_lineage") or {}).get("source_type") or "unknown")] += 1
    return counter


def _summarize_strategy(strategy: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    success_sources = Counter(str(record.get("success_source") or "none") for record in records)
    failures = Counter(str(record.get("failure_reason") or "none") for record in records)
    validator_failures: Counter[str] = Counter()
    candidate_sources: Counter[str] = Counter()
    latency_ms = []
    final_success = 0
    fallback_recovered = 0
    alignment_failure = 0
    joint_limit_failure = 0
    collision_failure = 0
    for record in records:
        result = record.get("result") or {}
        status = str(result.get("status") or record.get("status") or "")
        final_success += int(status == "success")
        trace = (result.get("planner_run_record") or {}).get("fallback_trace") or []
        if status == "success" and any(item.get("stage") == "rule_fallback" for item in trace):
            fallback_recovered += 1
        reason = str(result.get("failure_reason") or "")
        if "alignment" in reason:
            alignment_failure += 1
        validator_failures.update(_validator_failure_counts(result))
        candidate_sources.update(_candidate_source_counts(result))
        latency_ms.append(float(record.get("elapsed_sec") or 0.0) * 1000.0)
    for key, value in validator_failures.items():
        if "joint_limit" in key:
            joint_limit_failure += int(value)
        if "collision_failed" in key:
            collision_failure += int(value)
    request_count = len(records)
    return {
        "strategy": strategy,
        "request_count": int(request_count),
        "final_success_count": int(final_success),
        "final_success_rate": _round(final_success / max(request_count, 1)),
        "fixed_budget_success_rate": _round(final_success / max(request_count, 1)),
        "fallback_recovery_rate": _round(fallback_recovered / max(request_count, 1)),
        "status_counts": dict(sorted(statuses.items())),
        "success_source_counts": dict(sorted(success_sources.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "validator_failure_counts": dict(sorted(validator_failures.items())),
        "candidate_source_type_counts": dict(sorted(candidate_sources.items())),
        "alignment_failure_count": int(alignment_failure),
        "joint_limit_failure_count": int(joint_limit_failure),
        "collision_failure_count": int(collision_failure),
        "latency_ms": {
            "p50": _round(_percentile(latency_ms, 0.50)),
            "p95": _round(_percentile(latency_ms, 0.95)),
            "mean": _round(statistics.mean(latency_ms) if latency_ms else None),
        },
    }


def _build_config(args: argparse.Namespace, strategy: str):
    from level_planner_core.planner import LevelPlannerConfig

    config = LevelPlannerConfig.from_file(args.config)
    if args.device:
        config.device = str(args.device)
    if args.use_cuda_graph is not None:
        config.use_cuda_graph = bool(args.use_cuda_graph)
    if args.warmup_iterations is not None:
        config.warmup_iterations = int(args.warmup_iterations)
    if args.num_candidates is not None:
        config.num_candidates = int(args.num_candidates)
    config.learned_seed_use_critic = strategy in {"diffusion_critic", "mixed_fallback"}
    return config


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from level_planner_core import LevelConstrainedPlanner

    requests = load_requests(args.requests)
    selected = requests[int(args.offset):]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    strategies = [item.strip() for item in args.strategies.split(",") if item.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    strategy_records: dict[str, list[dict[str, Any]]] = {}
    for strategy in strategies:
        config = _build_config(args, strategy)
        planner = LevelConstrainedPlanner(config)
        records: list[dict[str, Any]] = []
        strategy_dir = args.out_dir / strategy
        strategy_dir.mkdir(parents=True, exist_ok=True)
        for local_index, request in enumerate(selected):
            index = int(args.offset) + local_index
            strategy_req = strategy_request(request, strategy, float(args.total_budget_ms))
            out_dir = request_output_dir(strategy_dir, index, strategy_req)
            result_path = out_dir / "result.json"
            started = time.time()
            if args.resume and result_path.exists():
                result = json.loads(result_path.read_text(encoding="utf-8"))
                skipped = True
            else:
                out_dir.mkdir(parents=True, exist_ok=True)
                (out_dir / "request.json").write_text(
                    json.dumps(strategy_req, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                result = planner.plan(strategy_req, out_dir=out_dir)
                skipped = False
            elapsed_sec = time.time() - started
            record = {
                "strategy": strategy,
                "index": int(index),
                "request_id": result.get("request_id"),
                "out_dir": str(out_dir),
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason"),
                "success_source": (result.get("metrics") or {}).get("success_source"),
                "selected_candidate_id": (result.get("metrics") or {}).get("selected_candidate_id"),
                "elapsed_sec": round(float(elapsed_sec), 6),
                "skipped": bool(skipped),
                "result": result,
            }
            records.append(record)
            if args.progress:
                print(
                    json.dumps(
                        {
                            "strategy": strategy,
                            "index": index,
                            "status": record["status"],
                            "success_source": record["success_source"],
                            "out_dir": str(out_dir),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        strategy_records[strategy] = records

    summaries = [_summarize_strategy(strategy, strategy_records[strategy]) for strategy in strategies]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "requests": str(args.requests),
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "strategy_order": strategies,
        "request_count": len(selected),
        "total_budget_ms": float(args.total_budget_ms),
        "device": str(args.device),
        "use_cuda_graph": bool(args.use_cuda_graph) if args.use_cuda_graph is not None else None,
        "warmup_iterations": args.warmup_iterations,
        "num_candidates": args.num_candidates,
        "summaries": summaries,
        "records": {
            strategy: [
                {key: value for key, value in record.items() if key != "result"}
                for record in records
            ]
            for strategy, records in strategy_records.items()
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        _write_markdown_report(report, args.md_out)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return report


def _write_markdown_report(report: dict[str, Any], out: Path) -> None:
    lines = [
        "# Closed-Loop CuRobo Benchmark",
        "",
        f"- requests: `{report['requests']}`",
        f"- total_budget_ms: `{report['total_budget_ms']}`",
        f"- request_count: `{report['request_count']}`",
        "",
        "| Strategy | Final Success | Fallback Recovery | P50 ms | P95 ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        lines.append(
            f"| {item['strategy']} | {item['final_success_rate']:.3f} | "
            f"{item['fallback_recovery_rate']:.3f} | "
            f"{(item['latency_ms']['p50'] or 0.0):.2f} | "
            f"{(item['latency_ms']['p95'] or 0.0):.2f} |"
        )
    lines += [
        "",
        "This benchmark runs the actual standalone CuRobo planner core. Collision replay remains transparent `unchecked` until the hard validator gains world collision distance labels.",
        "",
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sr5_level.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path)
    parser.add_argument("--strategies", default="rule_only,diffusion_only,diffusion_critic,mixed_fallback")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--total-budget-ms", type=float, default=2500.0)
    parser.add_argument("--device")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--use-cuda-graph", dest="use_cuda_graph", action="store_true", default=None)
    parser.add_argument("--no-use-cuda-graph", dest="use_cuda_graph", action="store_false")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
