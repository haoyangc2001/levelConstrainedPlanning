#!/usr/bin/env python3
"""Run a batch of standalone planning requests and keep lifecycle artifacts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "planning_lifecycle_batch_summary.v1"


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


def load_requests(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: request must be a JSON object")
            rows.append(row)
    return rows


def _safe_id(value: Any, fallback: str) -> str:
    raw = str(value or fallback)
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
    return cleaned.strip("_") or fallback


def request_output_dir(out_dir: Path, index: int, request: dict[str, Any]) -> Path:
    request_id = _safe_id(request.get("request_id"), f"request_{index:05d}")
    return out_dir / f"{index:05d}_{request_id}"


def _read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate_counters(result: dict[str, Any]) -> tuple[Counter[str], Counter[str]]:
    source_counter: Counter[str] = Counter()
    status_counter: Counter[str] = Counter()
    candidates = result.get("candidate_records") or []
    for candidate in candidates:
        source_counter[str((candidate.get("source_lineage") or {}).get("source_type") or "unknown")] += 1
        status_counter[str((candidate.get("labels") or {}).get("candidate_status") or "unknown")] += 1
    return source_counter, status_counter


def _summarize_result(
    *,
    index: int,
    request: dict[str, Any],
    out_dir: Path,
    result: dict[str, Any],
    attempts: int,
    skipped: bool,
    elapsed_sec: float,
) -> dict[str, Any]:
    sampling = (request.get("metadata") or {}).get("sampling") or {}
    return {
        "index": int(index),
        "request_id": result.get("request_id") or request.get("request_id"),
        "out_dir": str(out_dir),
        "status": result.get("status"),
        "failure_reason": result.get("failure_reason"),
        "success_source": (result.get("metrics") or {}).get("success_source"),
        "selected_candidate_id": (result.get("metrics") or {}).get("selected_candidate_id"),
        "difficulty_bucket": sampling.get("difficulty_bucket"),
        "obstacle_case": sampling.get("obstacle_case"),
        "seed_policy_mode": (request.get("seed_policy") or {}).get("mode"),
        "attempts": int(attempts),
        "skipped": bool(skipped),
        "elapsed_sec": round(float(elapsed_sec), 6),
        "candidate_count": len(result.get("candidate_records") or []),
    }


def _build_config(args: argparse.Namespace):
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
    return config


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    from level_planner_core import LevelConstrainedPlanner

    requests = load_requests(args.requests)
    selected = requests[int(args.offset):]
    if args.limit is not None:
        selected = selected[: int(args.limit)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = _build_config(args)
    planner = LevelConstrainedPlanner(config)

    started = time.time()
    per_request: list[dict[str, Any]] = []
    result_records: list[dict[str, Any]] = []
    executed_count = 0
    skipped_count = 0
    for local_index, request in enumerate(selected):
        index = int(args.offset) + local_index
        req_dir = request_output_dir(args.out_dir, index, request)
        result_path = req_dir / "result.json"
        request_path = req_dir / "request.json"
        req_dir.mkdir(parents=True, exist_ok=True)
        request_path.write_text(json.dumps(request, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        request_started = time.time()
        skipped = bool(args.resume and result_path.exists())
        attempts = 0
        if skipped:
            result = _read_result(result_path)
            skipped_count += 1
        else:
            result = {}
            for attempt in range(int(args.retries) + 1):
                attempts = attempt + 1
                result = planner.plan(request, out_dir=req_dir)
                if result.get("status") != "failed_internal_error":
                    break
            executed_count += 1
        elapsed_sec = time.time() - request_started
        record = _summarize_result(
            index=index,
            request=request,
            out_dir=req_dir,
            result=result,
            attempts=attempts,
            skipped=skipped,
            elapsed_sec=elapsed_sec,
        )
        per_request.append(record)
        result_records.append(result)
        if args.progress:
            print(
                json.dumps(
                    {
                        "index": index,
                        "request_id": record["request_id"],
                        "status": record["status"],
                        "success_source": record["success_source"],
                        "out_dir": str(req_dir),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    status_counter = Counter(str(item.get("status") or "unknown") for item in per_request)
    success_source_counter = Counter(str(item.get("success_source") or "none") for item in per_request)
    failure_counter = Counter(str(item.get("failure_reason") or "none") for item in per_request)
    difficulty_counter = Counter(str(item.get("difficulty_bucket") or "unknown") for item in per_request)
    obstacle_counter = Counter(str(item.get("obstacle_case") or "unknown") for item in per_request)
    source_counter: Counter[str] = Counter()
    candidate_status_counter: Counter[str] = Counter()
    for result in result_records:
        src, status = _candidate_counters(result)
        source_counter.update(src)
        candidate_status_counter.update(status)

    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "request_jsonl": str(args.requests),
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "device": str(config.device),
        "use_cuda_graph": bool(config.use_cuda_graph),
        "warmup_iterations": int(config.warmup_iterations),
        "num_candidates": int(config.num_candidates),
        "request_count": len(selected),
        "executed_count": int(executed_count),
        "skipped_count": int(skipped_count),
        "success_count": int(status_counter.get("success", 0)),
        "failure_count": int(len(selected) - status_counter.get("success", 0)),
        "elapsed_sec": round(time.time() - started, 6),
        "status_counts": dict(sorted(status_counter.items())),
        "success_source_counts": dict(sorted(success_source_counter.items())),
        "failure_reason_counts": dict(sorted(failure_counter.items())),
        "difficulty_bucket_counts": dict(sorted(difficulty_counter.items())),
        "obstacle_case_counts": dict(sorted(obstacle_counter.items())),
        "candidate_source_type_counts": dict(sorted(source_counter.items())),
        "candidate_status_counts": dict(sorted(candidate_status_counter.items())),
        "per_request": per_request,
        "command": getattr(args, "_command", []),
    }
    summary_out = args.summary_out or (args.out_dir / "batch_summary.json")
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/sr5_level.yaml"))
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retries", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--warmup-iterations", type=int)
    parser.add_argument("--use-cuda-graph", dest="use_cuda_graph", action="store_true", default=None)
    parser.add_argument("--no-use-cuda-graph", dest="use_cuda_graph", action="store_false")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    args._command = sys.argv if argv is None else ["python", "tools/dataset/run_lifecycle_batch.py", *argv]
    run_batch(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
