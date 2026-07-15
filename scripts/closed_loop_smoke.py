#!/usr/bin/env python3
"""Headless closed-loop smoke for the standalone learning-optimization loop."""

from __future__ import annotations

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from level_planner_core import LevelConstrainedPlanner, LevelPlannerConfig
from tools.dataset.export_core_results_dataset import _iter_result_files, _samples_from_result, _write_manifest
from tools.dataset.validate_candidate_dataset import validate_candidate_dataset


DEFAULT_REQUEST = Path("examples/requests/request_level_alignment_hard.json")
DEFAULT_OUT_DIR = Path("runs/phase9_closed_loop_smoke")


STRATEGIES: dict[str, dict[str, Any]] = {
    "rule_only": {
        "mode": "rule",
        "k_generate": 2,
        "k_accept": 2,
        "fallback_to_rule_seed": True,
        "fallback_to_planner_native": False,
        "timeout_sec": 0.5,
    },
    "diffusion_shadow": {
        "mode": "shadow",
        "k_generate": 2,
        "k_accept": 1,
        "fallback_to_rule_seed": True,
        "fallback_to_planner_native": True,
        "timeout_sec": 2.0,
    },
    "diffusion_candidate": {
        "mode": "diffusion",
        "k_generate": 2,
        "k_accept": 1,
        "fallback_to_rule_seed": False,
        "fallback_to_planner_native": False,
        "timeout_sec": 2.0,
    },
    "mixed_fallback": {
        "mode": "mixed",
        "k_generate": 2,
        "k_accept": 1,
        "fallback_to_rule_seed": True,
        "fallback_to_planner_native": True,
        "timeout_sec": 2.0,
    },
}


def _load_request(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _strategy_request(base: dict[str, Any], strategy: str) -> dict[str, Any]:
    request = deepcopy(base)
    request["request_id"] = f"{request.get('request_id', 'request')}_{strategy}"
    request["seed_policy"] = dict(STRATEGIES[strategy])
    metadata = dict(request.get("metadata") or {})
    metadata["closed_loop_smoke_strategy"] = strategy
    metadata["total_budget_ms"] = 2500.0
    metadata["num_candidates"] = 2
    request["metadata"] = metadata
    return request


def _provider_reports(result: dict[str, Any], provider: str) -> list[dict[str, Any]]:
    return [
        report
        for report in result.get("seed_provider_reports") or []
        if (report.get("provider") or report.get("provider_name")) == provider
    ]


def _has_diffusion_sample(result: dict[str, Any]) -> bool:
    reports = _provider_reports(result, "diffusion_seed")
    return any(int(report.get("generated_count") or report.get("metadata", {}).get("available_generated_count") or 0) > 0 for report in reports)


def _has_critic_score(result: dict[str, Any]) -> bool:
    reports = _provider_reports(result, "diffusion_seed")
    for report in reports:
        metadata = report.get("metadata") or {}
        if metadata.get("critic_status") == "scored":
            return True
        for candidate in report.get("candidates") or []:
            if (candidate.get("metadata") or {}).get("critic_score"):
                return True
    return False


def _has_fallback_trace(result: dict[str, Any]) -> bool:
    trace = (result.get("planner_run_record") or {}).get("fallback_trace") or []
    return any(item.get("stage") == "rule_fallback" for item in trace)


def _write_dataset_from_results(result_root: Path, dataset_name: str, out: Path) -> dict[str, Any]:
    result_files = _iter_result_files([result_root])
    samples: list[dict[str, Any]] = []
    for path in result_files:
        samples.extend(_samples_from_result(path, dataset_name))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    return _write_manifest(
        samples=samples,
        out=out,
        manifest_out=out.with_suffix(".manifest.json"),
        dataset_name=dataset_name,
        result_files=result_files,
    )


def run_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = LevelPlannerConfig.from_file(args.config)
    if args.device:
        config.device = args.device
    config.use_cuda_graph = bool(args.use_cuda_graph)
    config.warmup_iterations = int(args.warmup_iterations)
    config.num_candidates = int(args.num_candidates)
    planner = LevelConstrainedPlanner(config)
    base_request = _load_request(args.request)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    result_by_strategy: dict[str, dict[str, Any]] = {}
    for strategy in STRATEGIES:
        request = _strategy_request(base_request, strategy)
        out_dir = args.out_dir / strategy
        started = time.time()
        result = planner.plan(request, out_dir=out_dir)
        elapsed_sec = time.time() - started
        result_by_strategy[strategy] = result
        records.append(
            {
                "strategy": strategy,
                "request_id": result.get("request_id"),
                "status": result.get("status"),
                "failure_reason": result.get("failure_reason"),
                "success_source": (result.get("metrics") or {}).get("success_source"),
                "selected_candidate_id": (result.get("metrics") or {}).get("selected_candidate_id"),
                "elapsed_sec": round(elapsed_sec, 6),
                "result_json": result.get("artifacts", {}).get("result_json"),
            }
        )

    dataset_path = args.out_dir / "closed_loop_smoke_candidates.jsonl"
    dataset_manifest = _write_dataset_from_results(
        args.out_dir,
        dataset_name="phase9_closed_loop_smoke",
        out=dataset_path,
    )
    validation = validate_candidate_dataset(dataset_path, require_negative=True)
    validation_path = args.out_dir / "closed_loop_smoke_validation.json"
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    checks = {
        "dataset_exported": dataset_manifest.get("sample_count", 0) > 0,
        "dataset_schema_valid": bool(validation.get("valid")),
        "model_sample": _has_diffusion_sample(result_by_strategy["diffusion_shadow"])
        and _has_diffusion_sample(result_by_strategy["diffusion_candidate"]),
        "critic_score": _has_critic_score(result_by_strategy["diffusion_candidate"])
        or _has_critic_score(result_by_strategy["mixed_fallback"]),
        "fallback_trace": _has_fallback_trace(result_by_strategy["mixed_fallback"]),
        "no_rviz_required": True,
    }
    summary = {
        "schema_version": "closed_loop_smoke.v1",
        "config": str(args.config),
        "request": str(args.request),
        "out_dir": str(args.out_dir),
        "records": records,
        "dataset": dataset_manifest,
        "validation": validation,
        "checks": checks,
        "passed": all(checks.values()),
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/sr5_level.yaml"))
    parser.add_argument("--request", type=Path, default=DEFAULT_REQUEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--summary-out", type=Path, default=DEFAULT_OUT_DIR / "summary.json")
    parser.add_argument("--device")
    parser.add_argument("--num-candidates", type=int, default=2)
    parser.add_argument("--warmup-iterations", type=int, default=0)
    parser.add_argument("--use-cuda-graph", action="store_true", default=False)
    args = parser.parse_args(argv)
    summary = run_smoke(args)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
