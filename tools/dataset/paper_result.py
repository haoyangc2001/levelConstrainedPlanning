#!/usr/bin/env python3
"""Unified paper-level result contract (``paper_result.v1``) -- task A4.

One canonical record shape that *every* experiment (E1-E4) and *every* method
(ours + Phase B baselines) must emit, so paper tables/figures are generated from
a single schema instead of per-experiment ad-hoc JSON.  This is the contract
Phase B baselines must honour (A4 is deliberately specified early, before the
baselines exist), which is why the schema is defined here rather than inline in
any one runner.

Layout
------
* ``PaperResult`` -- a frozen dataclass mirroring ``paper_result.v1`` field for
  field (A4.1).  ``to_dict`` round-trips to JSON; ``SCHEMA_VERSION`` stamps it.
* ``from_benchmark_report`` -- convert a ``closed_loop_curobo_benchmark.v2``
  report (one record per benchmark *cell*) into ``PaperResult`` rows, computing
  Wilson CIs and passing through per-problem success bits (A4.2).
* ``write_paper_results`` / ``load_paper_results`` -- JSONL persistence.

The chart/table stubs live in ``paper_charts.py`` (A4.3) and consume these rows.

Field provenance (A4.1, finalised after A1/A2 landed):
* ``collision`` <- A1 collision-distance replay (validators ``collision_safety``).
* ``motion_quality`` <- A2 dimensioned kinematics (jerk/accel/vel/motion_time).
* ``constraint_error`` / ``latency`` / ``per_problem_success`` <- A3 harness v2.
* ``constraint_class`` / ``obstacle_*`` / ``reconfiguration_count`` <- request
  metadata; these are Phase C feature builds, so the converter records ``None``
  / ``"unknown"`` until the paper-scale sampler populates them (kept in the
  schema now so the contract is stable for Phase B).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "paper_result.v1"

# Recognised constraint classes (A4.1). LP=level, LPO=level+obstacle,
# PP=pose, PPO=pose+obstacle. "unknown" until the Phase C sampler tags requests.
CONSTRAINT_CLASSES = ("LP", "LPO", "PP", "PPO", "unknown")
# Budget semantics vocabulary shared with the A3 harness (A3.0/A4.1).
BUDGET_SEMANTICS = ("additive", "fixed", "compute")


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    """Wilson score CI for a binomial rate (A4.2). Mirrors the A3 harness helper
    so paper_result rows carry a CI even when converting older reports that lack
    one."""
    if total <= 0:
        return {"wilson_lo": None, "wilson_hi": None, "z": z}
    p = successes / total
    denom = 1.0 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    margin = (z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))) / denom
    return {"wilson_lo": round(centre - margin, 6), "wilson_hi": round(centre + margin, 6), "z": z}


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


@dataclass
class PaperResult:
    """One row of ``paper_result.v1`` (A4.1).

    A row is one (experiment, method, constraint_class, obstacle setting, K,
    budget, seed) cell aggregated over ``n_problems``.  Distribution-valued
    fields (``constraint_error``, ``collision``, ``latency``,
    ``motion_quality``, ``diversity``) hold the same ``{count,mean,p50,p95,max}``
    (or percentile) sub-dicts the A3 harness emits so no information is lost in
    the conversion.
    """

    # --- identity / conditioning axes ---------------------------------------
    method: str
    experiment: str = "unknown"           # E1..E4 tag (set by the caller)
    constraint_class: str = "unknown"     # LP / LPO / PP / PPO / unknown
    obstacle_density: float | None = None
    obstacle_topology: str = "unknown"    # A4.1 review add
    K: int | None = None
    budget: int | None = None
    budget_semantics: str = "compute"     # additive | fixed | compute
    seed: int | None = None
    n_problems: int = 0
    n_seeds: int = 1                       # A4.1 review add

    # --- primary outcomes ----------------------------------------------------
    success_rate: float | None = None
    success_rate_ci: dict[str, Any] = field(default_factory=dict)  # {wilson_lo,wilson_hi,z}
    success_at_k: float | None = None
    success_at_k_ci: dict[str, Any] = field(default_factory=dict)
    #逐问题成功位, index-aligned -> paired / McNemar tests (A4.1 review add).
    per_problem_success: list[dict[str, Any]] = field(default_factory=list)

    # --- hardware (cross-hardware fair disclosure, A4.1 review add) ----------
    hardware: dict[str, Any] = field(default_factory=dict)  # {gpu,cpu,ram,...}

    # --- constraint / safety / quality distributions -------------------------
    constraint_error: dict[str, Any] = field(default_factory=dict)  # + violation_rate
    collision: dict[str, Any] = field(default_factory=dict)          # {min_dist, collision_rate}
    latency: dict[str, Any] = field(default_factory=dict)            # {p50,p75,p95,p98,mean}
    motion_quality: dict[str, Any] = field(default_factory=dict)     # {max_jerk,max_accel,mean_vel,motion_time}
    diversity: dict[str, Any] = field(default_factory=dict)          # {waypoint_variance, ik_branch_count}
    reconfiguration_count: dict[str, Any] = field(default_factory=dict)

    # --- provenance ----------------------------------------------------------
    source_label: str | None = None       # benchmark cell label
    source_report: str | None = None      # path to the source benchmark JSON
    git_commit: str | None = None
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# A4.2: closed_loop_curobo_benchmark.v2 -> paper_result.v1 converter.
# ---------------------------------------------------------------------------
def _collision_block(constraint_error: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    """Assemble the ``collision`` block from the A1 min-distance distribution.

    The min-distance distribution and ``collision_failure_count`` are both
    *candidate-level* (one entry per generated candidate whose collision was
    checked), so ``collision_rate`` is defined over that same candidate
    population -- failures / checked-candidates -- which keeps it bounded in
    [0, 1].  Dividing by ``n_problems`` would be a category error (a problem
    fans out into many candidates, so the ratio could exceed 1).  A true
    per-problem selected-trajectory collision rate needs the Phase C per-request
    obstacle worlds and is left for that phase.
    """
    dist = (constraint_error or {}).get("collision_min_distance_m") or {}
    checked = int(dist.get("count") or 0)
    collisions = int(summary.get("collision_failure_count") or 0)
    return {
        "min_distance_m": dist,
        "collision_failure_count": collisions,
        "checked_candidate_count": checked,
        "collision_rate_candidate": round(collisions / checked, 6) if checked else None,
        "collision_rate_note": "candidate-level (failures / checked candidates); per-problem rate pending Phase C obstacle worlds",
    }


def _motion_quality_block(constraint_error: dict[str, Any]) -> dict[str, Any]:
    """Map the A3.4 dimensioned distributions onto the paper_result names."""
    ce = constraint_error or {}
    return {
        "max_jerk": ce.get("max_jerk_rad_s3") or {},
        "max_accel": ce.get("max_acceleration_rad_s2") or {},
        "mean_vel": ce.get("max_velocity_rad_s") or {},
        "motion_time": ce.get("motion_time_sec") or {},
    }


def _ci(summary_ci: dict[str, Any] | None, successes: int, total: int) -> dict[str, Any]:
    """Prefer the harness-provided CI; recompute (Wilson) if absent (A4.2)."""
    if summary_ci and summary_ci.get("low") is not None:
        return {"wilson_lo": summary_ci.get("low"), "wilson_hi": summary_ci.get("high"), "z": summary_ci.get("z", 1.96)}
    return wilson_interval(successes, total)


def from_benchmark_report(
    report: dict[str, Any],
    *,
    experiment: str = "unknown",
    source_report: str | None = None,
) -> list[PaperResult]:
    """Convert one v2 benchmark report into ``paper_result.v1`` rows (A4.2).

    Each summary (benchmark cell = method x K x budget x seed) becomes one row.
    Request-level metadata that only the Phase C sampler populates
    (constraint_class, obstacle_*, reconfiguration_count, diversity) is left at
    its schema default here; the contract stays stable so Phase B/C can fill it.
    """
    schema = str(report.get("schema_version") or "")
    if not schema.startswith("closed_loop_curobo_benchmark"):
        raise ValueError(f"unexpected benchmark schema_version: {schema!r}")

    hardware = report.get("hardware") or {}
    git_commit = report.get("git_commit")
    seeds = report.get("repeat_seeds") or [0]
    rows: list[PaperResult] = []
    for summary in report.get("summaries") or []:
        n = int(summary.get("request_count") or 0)
        final_c = int(summary.get("final_success_count") or 0)
        atk_c = int(summary.get("success_at_k_count") or 0)
        ce = summary.get("constraint_error") or {}
        bits = summary.get("per_problem_success_bits") or {}
        rows.append(
            PaperResult(
                method=str(summary.get("method") or summary.get("strategy") or "unknown"),
                experiment=experiment,
                K=summary.get("k_generate"),
                budget=summary.get("compute_budget_solve_calls"),
                budget_semantics="compute",
                seed=summary.get("repeat_seed"),
                n_problems=n,
                n_seeds=len(seeds),
                success_rate=summary.get("final_success_rate"),
                success_rate_ci=_ci(summary.get("final_success_wilson_ci"), final_c, n),
                success_at_k=summary.get("success_at_k_rate"),
                success_at_k_ci=_ci(summary.get("success_at_k_wilson_ci"), atk_c, n),
                per_problem_success=bits.get("final") or [],
                hardware=hardware,
                constraint_error=ce,
                collision=_collision_block(ce, summary),
                latency=summary.get("latency_ms") or {},
                motion_quality=_motion_quality_block(ce),
                reconfiguration_count={"count": 0, "note": "not_populated_until_phase_c"},
                source_label=summary.get("label"),
                source_report=source_report,
                git_commit=git_commit,
            )
        )
    return rows


def write_paper_results(rows: list[PaperResult], path: Path) -> Path:
    """Persist rows as JSONL (one paper_result.v1 record per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_paper_results(path: Path) -> list[dict[str, Any]]:
    """Load paper_result.v1 rows from a JSONL file."""
    path = Path(path)
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Convert a v2 benchmark report to paper_result.v1 JSONL (A4.2).")
    parser.add_argument("report", type=Path, help="closed_loop_curobo_benchmark.v2 JSON report")
    parser.add_argument("--out", type=Path, required=True, help="output paper_result.v1 JSONL path")
    parser.add_argument("--experiment", default="unknown", help="experiment tag (E1..E4)")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = from_benchmark_report(report, experiment=args.experiment, source_report=str(args.report))
    write_paper_results(rows, args.out)
    print(json.dumps({"schema_version": SCHEMA_VERSION, "rows": len(rows), "out": str(args.out)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
