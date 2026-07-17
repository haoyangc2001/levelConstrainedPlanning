#!/usr/bin/env python3
"""Run a fixed-budget closed-loop benchmark through the CuRobo planner core.

Budget semantics (A3.0): CuRobo ``solve_pose`` cannot be interrupted mid-solve,
so the budget axis is a *compute budget* -- the number of allowed seed/solve
attempts (``k_generate``) -- with a *uniform* wall-clock ``timeout_sec`` guard
across all methods (A3.0c). K and budget are swept via ``--k-values`` /
``--budget-values`` (A3.1/A3.2). Success@K is computed by post-processing the
planner's returned candidate records (A3.3), constraint errors are aggregated
into distributions (A3.4), latency reports p50/p75/p95/p98 (A3.5), the schema is
v2 with K/budget axes (A3.6), the method mapping is centralised in
``methods.py`` (A3.7), and each setting supports >=1 statistical repeat seeds
with per-problem success bits for Wilson CI / McNemar (A3.8).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.dataset import methods as method_dispatch
from tools.dataset.run_lifecycle_batch import load_requests, request_output_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "closed_loop_curobo_benchmark.v2"


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


def _hardware_spec(device: str | None) -> dict[str, Any]:
    """A3.8: record hardware so latency numbers are interpretable."""
    spec: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor() or None,
        "python": platform.python_version(),
        "device": str(device) if device else None,
    }
    try:  # torch/GPU is optional at import time
        import torch  # type: ignore

        spec["torch"] = torch.__version__
        if torch.cuda.is_available():
            idx = 0
            if device and str(device).startswith("cuda:"):
                try:
                    idx = int(str(device).split(":", 1)[1])
                except ValueError:
                    idx = 0
            spec["gpu_name"] = torch.cuda.get_device_name(idx)
            spec["cuda"] = torch.version.cuda
    except Exception:
        pass
    return spec


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    """A3.8: Wilson score CI for a binomial success rate."""
    if total <= 0:
        return {"low": None, "high": None, "z": z}
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return {"low": round(centre - margin, 6), "high": round(centre + margin, 6), "z": z}


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


def method_request(
    request: dict[str, Any],
    method: str,
    *,
    k_generate: int,
    k_accept: int,
    timeout_sec: float,
    compute_budget: int,
) -> dict[str, Any]:
    """A3.0c/A3.7: build a per-request seed policy via the uniform dispatch.

    All methods share the same ``timeout_sec`` guard; the compute budget is the
    number of allowed seed/solve attempts (``k_generate``).
    """
    return method_dispatch.build_seed_policy(
        request,
        method,
        k_generate=k_generate,
        k_accept=k_accept,
        timeout_sec=timeout_sec,
        compute_budget=compute_budget,
    )


def _candidate_alignment_deviations(result: dict[str, Any]) -> list[float]:
    """A3.4: pull real max-alignment-deviation (deg) from every candidate."""
    values: list[float] = []
    for candidate in result.get("candidate_records") or []:
        metrics = candidate.get("validator_metrics") or {}
        checks = metrics.get("checks") or {}
        align = checks.get("alignment") or {}
        dev = align.get("max_alignment_deviation_deg")
        if dev is None:
            dev = (candidate.get("metrics") or {}).get("max_alignment_deviation_deg")
        if isinstance(dev, (int, float)) and math.isfinite(float(dev)):
            values.append(float(dev))
    return values


def _selected_alignment_deviation(result: dict[str, Any]) -> float | None:
    """A3.4: alignment deviation of the *selected* (returned) trajectory."""
    metrics = result.get("metrics") or {}
    # Selected trajectory's alignment check lives under metrics.alignment.
    align = metrics.get("alignment") or {}
    dev = align.get("max_alignment_deviation_deg")
    if isinstance(dev, (int, float)) and math.isfinite(float(dev)):
        return float(dev)
    # Fallback: locate the selected candidate record by id.
    sel_id = metrics.get("selected_candidate_id")
    if sel_id:
        for candidate in result.get("candidate_records") or []:
            cid = (candidate.get("source_lineage") or {}).get("candidate_id") or candidate.get("candidate_id")
            if cid == sel_id:
                checks = (candidate.get("validator_metrics") or {}).get("checks") or {}
                cdev = (checks.get("alignment") or {}).get("max_alignment_deviation_deg")
                if isinstance(cdev, (int, float)) and math.isfinite(float(cdev)):
                    return float(cdev)
    return None


def _min_collision_distances(result: dict[str, Any]) -> list[float]:
    """A3.4: aggregate A1 min collision distance (m) when checked."""
    values: list[float] = []
    for candidate in result.get("candidate_records") or []:
        checks = (candidate.get("validator_metrics") or {}).get("checks") or {}
        collision = checks.get("collision_safety") or {}
        dist = collision.get("min_distance_m")
        if isinstance(dist, (int, float)) and math.isfinite(float(dist)):
            values.append(float(dist))
    return values


def _motion_quality(result: dict[str, Any]) -> dict[str, list[float]]:
    """A3.4: aggregate A2 dimensioned jerk / motion_time from candidates."""
    jerk: list[float] = []
    motion_time: list[float] = []
    for candidate in result.get("candidate_records") or []:
        checks = (candidate.get("validator_metrics") or {}).get("checks") or {}
        va = checks.get("velocity_acceleration") or {}
        if not va.get("dimensioned"):
            continue
        jmax = va.get("max_abs_jerk")
        mt = va.get("motion_time_sec")
        if isinstance(jmax, (int, float)) and math.isfinite(float(jmax)):
            jerk.append(float(jmax))
        if isinstance(mt, (int, float)) and math.isfinite(float(mt)):
            motion_time.append(float(mt))
    return {"max_abs_jerk": jerk, "motion_time_sec": motion_time}


def _distribution(values: list[float]) -> dict[str, float | int | None]:
    """A3.4: mean/p50/p95/max + count for a value list."""
    if not values:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(values),
        "mean": _round(statistics.mean(values)),
        "p50": _round(_percentile(values, 0.50)),
        "p95": _round(_percentile(values, 0.95)),
        "max": _round(max(values)),
    }


def _success_at_k(result: dict[str, Any], tolerance_deg: float | None = None) -> bool:
    """A3.3: at least one candidate passed the hard validator.

    Computed by post-processing candidate records (no re-run). A candidate
    counts as a Success@K hit when its hard validator reports ``valid`` (and, if
    a tolerance is supplied, its alignment deviation is within it).
    """
    for candidate in result.get("candidate_records") or []:
        metrics = candidate.get("validator_metrics") or {}
        if not metrics.get("valid"):
            continue
        if tolerance_deg is not None:
            checks = metrics.get("checks") or {}
            dev = (checks.get("alignment") or {}).get("max_alignment_deviation_deg")
            if isinstance(dev, (int, float)) and float(dev) > float(tolerance_deg):
                continue
        return True
    return False


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


def _summarize_strategy(
    method: str,
    records: list[dict[str, Any]],
    *,
    k_generate: int | None = None,
    compute_budget: int | None = None,
    tolerance_deg: float | None = None,
) -> dict[str, Any]:
    statuses = Counter(str(record.get("status") or "unknown") for record in records)
    success_sources = Counter(str(record.get("success_source") or "none") for record in records)
    failures = Counter(str(record.get("failure_reason") or "none") for record in records)
    validator_failures: Counter[str] = Counter()
    candidate_sources: Counter[str] = Counter()
    latency_ms: list[float] = []
    final_success = 0
    fallback_recovered = 0
    alignment_failure = 0
    joint_limit_failure = 0
    collision_failure = 0
    # A3.3/A3.8: per-problem success bits (index-aligned) for Success@K,
    # Wilson CI and paired McNemar tests downstream.
    success_at_k_bits: list[dict[str, Any]] = []
    final_success_bits: list[dict[str, Any]] = []
    success_at_k_count = 0
    # A3.4: constraint-error / motion-quality value pools.
    selected_align_devs: list[float] = []
    candidate_align_devs: list[float] = []
    collision_min_dists: list[float] = []
    jerk_values: list[float] = []
    motion_time_values: list[float] = []
    for record in records:
        result = record.get("result") or {}
        status = str(result.get("status") or record.get("status") or "")
        is_final = int(status == "success")
        final_success += is_final
        trace = (result.get("planner_run_record") or {}).get("fallback_trace") or []
        if status == "success" and any(item.get("stage") == "rule_fallback" for item in trace):
            fallback_recovered += 1
        reason = str(result.get("failure_reason") or "")
        if "alignment" in reason:
            alignment_failure += 1
        validator_failures.update(_validator_failure_counts(result))
        candidate_sources.update(_candidate_source_counts(result))
        latency_ms.append(float(record.get("elapsed_sec") or 0.0) * 1000.0)

        hit = _success_at_k(result, tolerance_deg=tolerance_deg)
        success_at_k_count += int(hit)
        req_id = record.get("request_id") or record.get("index")
        success_at_k_bits.append({"request_id": req_id, "success": bool(hit)})
        final_success_bits.append({"request_id": req_id, "success": bool(is_final)})

        sel_dev = _selected_alignment_deviation(result)
        if sel_dev is not None:
            selected_align_devs.append(sel_dev)
        candidate_align_devs.extend(_candidate_alignment_deviations(result))
        collision_min_dists.extend(_min_collision_distances(result))
        mq = _motion_quality(result)
        jerk_values.extend(mq["max_abs_jerk"])
        motion_time_values.extend(mq["motion_time_sec"])
    for key, value in validator_failures.items():
        if "joint_limit" in key:
            joint_limit_failure += int(value)
        if "collision_failed" in key:
            collision_failure += int(value)
    request_count = len(records)
    # A3.4: alignment violation rate against the tolerance (selected trajectories).
    violation_rate = None
    if tolerance_deg is not None and selected_align_devs:
        violations = sum(1 for d in selected_align_devs if d > float(tolerance_deg))
        violation_rate = _round(violations / len(selected_align_devs))
    return {
        "method": method,
        "strategy": method,  # backward-compat alias (schema v1 readers)
        "k_generate": int(k_generate) if k_generate is not None else None,
        "compute_budget_solve_calls": int(compute_budget) if compute_budget is not None else None,
        "request_count": int(request_count),
        "final_success_count": int(final_success),
        "final_success_rate": _round(final_success / max(request_count, 1)),
        # A3.3: Success@K -- at least one candidate passed the hard validator.
        "success_at_k_count": int(success_at_k_count),
        "success_at_k_rate": _round(success_at_k_count / max(request_count, 1)),
        # A3.0d: success conditioned on the compute budget.
        "fixed_budget_success_rate": _round(final_success / max(request_count, 1)),
        # A3.8: Wilson score CIs.
        "final_success_wilson_ci": _wilson_interval(final_success, request_count),
        "success_at_k_wilson_ci": _wilson_interval(success_at_k_count, request_count),
        "fallback_recovery_rate": _round(fallback_recovered / max(request_count, 1)),
        "status_counts": dict(sorted(statuses.items())),
        "success_source_counts": dict(sorted(success_sources.items())),
        "failure_reason_counts": dict(sorted(failures.items())),
        "validator_failure_counts": dict(sorted(validator_failures.items())),
        "candidate_source_type_counts": dict(sorted(candidate_sources.items())),
        "alignment_failure_count": int(alignment_failure),
        "joint_limit_failure_count": int(joint_limit_failure),
        "collision_failure_count": int(collision_failure),
        # A3.4: constraint-error and motion-quality distributions.
        "constraint_error": {
            "selected_alignment_deviation_deg": _distribution(selected_align_devs),
            "candidate_alignment_deviation_deg": _distribution(candidate_align_devs),
            "alignment_tolerance_deg": tolerance_deg,
            "alignment_violation_rate": violation_rate,
            "collision_min_distance_m": _distribution(collision_min_dists),
            "max_abs_jerk": _distribution(jerk_values),
            "motion_time_sec": _distribution(motion_time_values),
        },
        # A3.8: per-problem success bits (index-aligned across methods/repeats).
        "per_problem_success_bits": {
            "final": final_success_bits,
            "success_at_k": success_at_k_bits,
        },
        # A3.5: p50/p75/p95/p98 latency.
        "latency_ms": {
            "p50": _round(_percentile(latency_ms, 0.50)),
            "p75": _round(_percentile(latency_ms, 0.75)),
            "p95": _round(_percentile(latency_ms, 0.95)),
            "p98": _round(_percentile(latency_ms, 0.98)),
            "mean": _round(statistics.mean(latency_ms) if latency_ms else None),
        },
    }


def _build_config(args: argparse.Namespace, method: str):
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
    method_dispatch.apply_config(config, method)
    return config


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from level_planner_core import LevelConstrainedPlanner

    requests = load_requests(args.requests)
    selected = requests[int(args.offset):]
    if args.limit is not None:
        selected = selected[: int(args.limit)]
    methods = [item.strip() for item in args.strategies.split(",") if item.strip()]
    for method in methods:
        method_dispatch.get_method(method)  # validate up front
    k_values = _parse_int_list(args.k_values)
    budget_values = _parse_int_list(args.budget_values) if args.budget_values else list(k_values)
    seeds = _parse_int_list(args.seeds) if args.seeds else [0]
    tolerance_deg = float(args.tolerance_deg) if args.tolerance_deg is not None else None
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # A3.1/A3.2/A3.8: sweep method x K x budget x repeat-seed. Each (method, k,
    # budget, seed) is one benchmark *cell*. Cells sharing a method reuse one
    # planner instance to amortise CuRobo warmup.
    cells: list[dict[str, Any]] = []
    for method in methods:
        config = _build_config(args, method)
        planner = LevelConstrainedPlanner(config)
        for k_generate in k_values:
            for budget in budget_values:
                for seed in seeds:
                    cell_label = f"{method}_k{k_generate}_b{budget}_s{seed}"
                    cell_dir = args.out_dir / cell_label
                    cell_dir.mkdir(parents=True, exist_ok=True)
                    records: list[dict[str, Any]] = []
                    for local_index, request in enumerate(selected):
                        index = int(args.offset) + local_index
                        req = method_request(
                            request,
                            method,
                            k_generate=k_generate,
                            k_accept=min(int(args.k_accept), k_generate),
                            timeout_sec=float(args.timeout_sec),
                            compute_budget=budget,
                        )
                        # A3.8: repeat seed varies the request seed_policy seed so
                        # stochastic diffusion sampling differs across repeats.
                        req.setdefault("seed_policy", {})["repeat_seed"] = int(seed)
                        req["metadata"]["repeat_seed"] = int(seed)
                        out_dir = request_output_dir(cell_dir, index, req)
                        result_path = out_dir / "result.json"
                        started = time.time()
                        if args.resume and result_path.exists():
                            result = json.loads(result_path.read_text(encoding="utf-8"))
                            skipped = True
                        else:
                            out_dir.mkdir(parents=True, exist_ok=True)
                            (out_dir / "request.json").write_text(
                                json.dumps(req, ensure_ascii=False, indent=2) + "\n",
                                encoding="utf-8",
                            )
                            result = planner.plan(req, out_dir=out_dir)
                            skipped = False
                        elapsed_sec = time.time() - started
                        record = {
                            "method": method,
                            "strategy": method,
                            "k_generate": int(k_generate),
                            "compute_budget_solve_calls": int(budget),
                            "repeat_seed": int(seed),
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
                                        "cell": cell_label,
                                        "index": index,
                                        "status": record["status"],
                                        "success_source": record["success_source"],
                                        "out_dir": str(out_dir),
                                    },
                                    ensure_ascii=False,
                                ),
                                flush=True,
                            )
                    cells.append(
                        {
                            "label": cell_label,
                            "method": method,
                            "k_generate": int(k_generate),
                            "compute_budget_solve_calls": int(budget),
                            "repeat_seed": int(seed),
                            "records": records,
                        }
                    )

    summaries = [
        _summarize_strategy(
            cell["method"],
            cell["records"],
            k_generate=cell["k_generate"],
            compute_budget=cell["compute_budget_solve_calls"],
            tolerance_deg=tolerance_deg,
        )
        | {"label": cell["label"], "repeat_seed": cell["repeat_seed"]}
        for cell in cells
    ]
    report = {
        "schema_version": SCHEMA_VERSION,
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "requests": str(args.requests),
        "config": str(args.config),
        "out_dir": str(args.out_dir),
        "method_order": methods,
        "strategy_order": methods,  # backward-compat alias
        "request_count": len(selected),
        # A3.0: budget semantics recorded explicitly.
        "budget_semantics": "compute_budget_solve_calls",
        "uniform_timeout_sec": float(args.timeout_sec),
        "k_values": k_values,
        "budget_values": budget_values,
        "repeat_seeds": seeds,
        "alignment_tolerance_deg": tolerance_deg,
        "device": str(args.device),
        "hardware": _hardware_spec(args.device),  # A3.8
        "use_cuda_graph": bool(args.use_cuda_graph) if args.use_cuda_graph is not None else None,
        "warmup_iterations": args.warmup_iterations,
        "num_candidates": args.num_candidates,
        "summaries": summaries,
        "records": {
            cell["label"]: [
                {key: value for key, value in record.items() if key != "result"}
                for record in cell["records"]
            ]
            for cell in cells
        },
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.md_out:
        _write_markdown_report(report, args.md_out)
    print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
    return report


def _parse_int_list(spec: str | None) -> list[int]:
    if spec is None:
        return []
    return [int(item.strip()) for item in str(spec).split(",") if item.strip()]


def _write_markdown_report(report: dict[str, Any], out: Path) -> None:
    lines = [
        "# Closed-Loop CuRobo Benchmark",
        "",
        f"- requests: `{report['requests']}`",
        f"- budget semantics: `{report.get('budget_semantics')}` "
        f"(uniform timeout `{report.get('uniform_timeout_sec')}` s)",
        f"- K values: `{report.get('k_values')}` / budget values: `{report.get('budget_values')}`",
        f"- repeat seeds: `{report.get('repeat_seeds')}`",
        f"- request_count: `{report['request_count']}`",
        f"- hardware: `{(report.get('hardware') or {}).get('gpu_name') or (report.get('hardware') or {}).get('platform')}`",
        "",
        "| Cell | Method | K | Budget | Success@K | Final Success | Fallback | P50 | P75 | P95 | P98 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["summaries"]:
        lat = item["latency_ms"]
        lines.append(
            f"| {item.get('label', item['method'])} | {item['method']} | "
            f"{item.get('k_generate')} | {item.get('compute_budget_solve_calls')} | "
            f"{(item.get('success_at_k_rate') or 0.0):.3f} | "
            f"{item['final_success_rate']:.3f} | "
            f"{item['fallback_recovery_rate']:.3f} | "
            f"{(lat['p50'] or 0.0):.1f} | {(lat['p75'] or 0.0):.1f} | "
            f"{(lat['p95'] or 0.0):.1f} | {(lat['p98'] or 0.0):.1f} |"
        )
    lines += [
        "",
        "Budget is a *compute budget* (allowed seed/solve attempts): CuRobo `solve_pose` "
        "cannot be interrupted mid-solve, so all methods share one uniform wall-clock "
        "`timeout_sec` guard and success is conditioned on K/budget (A3.0).",
        "Collision replay validity depends on A1 world distance labels; where `unchecked`, "
        "collision-blind successes are flagged in `validator_failure_counts`.",
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
    parser.add_argument(
        "--strategies",
        default="rule_only,diffusion_only,diffusion_critic,mixed_fallback",
        help="Comma-separated method names (see tools/dataset/methods.py).",
    )
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    # A3.1/A3.2: sweep axes.
    parser.add_argument(
        "--k-values",
        default="6",
        help="Comma-separated k_generate values to sweep (A3.1).",
    )
    parser.add_argument(
        "--budget-values",
        default=None,
        help="Comma-separated compute-budget (solve-call) values (A3.2); "
        "defaults to --k-values.",
    )
    parser.add_argument("--k-accept", type=int, default=2, help="Accepted seeds per request.")
    # A3.0c: uniform timeout guard across all methods.
    parser.add_argument(
        "--timeout-sec",
        type=float,
        default=2.0,
        help="Uniform per-request seed-generation timeout guard (A3.0c).",
    )
    parser.add_argument(
        "--tolerance-deg",
        type=float,
        default=None,
        help="Alignment tolerance for Success@K / violation-rate (A3.3/A3.4).",
    )
    # A3.8: statistical repeats.
    parser.add_argument(
        "--seeds",
        default=None,
        help="Comma-separated repeat seeds (>=1) for statistical repeats (A3.8); default 0.",
    )
    # Deprecated: budget is now a compute budget, not wall-clock ms (A3.0).
    parser.add_argument(
        "--total-budget-ms",
        type=float,
        default=2500.0,
        help="DEPRECATED (A3.0): budget is now --budget-values (solve calls); ignored.",
    )
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
