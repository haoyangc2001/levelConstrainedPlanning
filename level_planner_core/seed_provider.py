"""Seed candidate abstractions for constrained trajectory planning.

[caohy] diffusionSeedLearning phase 1: keep the runtime behavior unchanged
while making candidate provenance explicit in lifecycle JSON.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Protocol


PLANNER_SOURCE_LABELS = {
    "planner",
    "planner_legacy",
}

DEFAULT_DIFFUSION_GENERATED_SAMPLES_PATH = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/"
    "sr5_phase4_generated_samples.json"
)


def infer_source_type(source_label: str | None) -> str:
    """Map a fine-grained lifecycle source_label to a stable source_type."""
    label = str(source_label or "unknown")
    if label in PLANNER_SOURCE_LABELS:
        return "planner"
    if label.startswith("diffusion_seed_"):
        return "diffusion"
    if label.startswith("critic_selected_"):
        return "critic"
    if label.startswith("fallback_"):
        return "fallback"
    if label.startswith("alignment_seed_family_"):
        return "rule_raw"
    if (
        label.startswith("alignment_seed_trajopt_")
        or label.startswith("alignment_seed_split_")
        or label.startswith("alignment_seed_sequence_")
        or label in {
            "alignment_seed_trajopt_split",
            "alignment_seed_trajopt_smoothed",
            "alignment_seed_trajopt_bridged",
        }
    ):
        return "rule"
    if label.startswith("alignment_seed_"):
        return "rule"
    return "unknown"


def infer_optimized(source_label: str | None, *, default: bool = True) -> bool:
    """Return whether a candidate should be treated as optimized in phase-1 schema."""
    source_type = infer_source_type(source_label)
    if source_type == "rule_raw":
        return False
    if source_type in {"planner", "rule", "diffusion", "critic", "fallback"}:
        return True
    return bool(default)


@dataclass
class SeedCandidate:
    """Runtime-neutral candidate schema shared by planner/rule/diffusion sources."""

    candidate_id: str
    source_label: str
    trajectory_points: list[list[float]]
    source_type: str | None = None
    optimized: bool | None = None
    selected: bool = False
    entered_pool: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    precheck: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)

    def to_lifecycle_dict(self) -> dict[str, Any]:
        source_type = self.source_type or infer_source_type(self.source_label)
        optimized = infer_optimized(self.source_label) if self.optimized is None else bool(self.optimized)
        return {
            "candidate_id": self.candidate_id,
            "source_label": self.source_label,
            "source_type": source_type,
            "optimized": optimized,
            "selected": bool(self.selected),
            "entered_pool": bool(self.entered_pool),
            "trajectory": {
                "format": "joint_position_rad",
                "shape": [
                    len(self.trajectory_points),
                    len(self.trajectory_points[0]) if self.trajectory_points else 0,
                ],
                "points": self.trajectory_points,
            },
            "metadata": dict(self.metadata),
            "precheck": dict(self.precheck),
            "metrics": dict(self.metrics),
        }


@dataclass
class SeedProviderResult:
    """Result returned by a SeedProvider without implying pool insertion."""

    provider_name: str
    mode: str
    status: str
    candidates: list[SeedCandidate] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def to_lifecycle_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "mode": self.mode,
            "status": self.status,
            "generated_count": len(self.candidates),
            "candidates": [candidate.to_lifecycle_dict() for candidate in self.candidates],
            "metadata": dict(self.metadata),
            "error": self.error,
        }


class SeedProvider(Protocol):
    """Protocol for future planner/rule/diffusion seed providers."""

    provider_name: str

    def generate(self, request_context: dict[str, Any]) -> SeedProviderResult:
        """Generate seed candidates for a planner request."""


class NullDiffusionSeedProvider:
    """Disabled diffusion provider used for phase-1 off/shadow/candidate wiring."""

    provider_name = "diffusion_seed"

    def __init__(self, mode: str = "off") -> None:
        self.mode = normalize_diffusion_mode(mode)

    def generate(self, request_context: dict[str, Any]) -> SeedProviderResult:
        status = "disabled" if self.mode == "off" else "empty_provider"
        return SeedProviderResult(
            provider_name=self.provider_name,
            mode=self.mode,
            status=status,
            candidates=[],
            metadata={
                "phase": "diffusionSeedLearning.phase1",
                "request_context_keys": sorted(str(key) for key in request_context.keys()),
                "runtime_effect": "no_candidates_generated_and_no_planner_behavior_change",
            },
        )


@dataclass
class DiffusionSeedProviderConfig:
    """Runtime config for file-backed diffusion seed smoke/candidate mode."""

    mode: str = "off"
    generated_samples_path: str = str(DEFAULT_DIFFUSION_GENERATED_SAMPLES_PATH)
    k_generate: int = 4
    k_accept: int = 2
    model_timeout_sec: float = 0.2
    max_start_gap_l2: float = 0.05
    max_step_l2: float = 1.0
    joint_abs_limit: float = 6.2832
    fallback_to_rule_seed: bool = True


class FileDiffusionSeedProvider:
    """File-backed diffusion seed provider for phase-6 candidate simulation.

    The first runtime version intentionally consumes the phase-4 generated
    sample artifact instead of requiring an online model service. This keeps the
    candidate-mode integration focused on pool admission, CuRobo repair,
    lifecycle accounting and fallback behavior.
    """

    provider_name = "diffusion_seed"

    def __init__(self, config: DiffusionSeedProviderConfig | None = None) -> None:
        self.config = config or DiffusionSeedProviderConfig()
        self.config.mode = normalize_diffusion_mode(self.config.mode)

    def generate(self, request_context: dict[str, Any]) -> SeedProviderResult:
        mode = normalize_diffusion_mode(self.config.mode)
        if mode == "off":
            return NullDiffusionSeedProvider(mode).generate(request_context)

        path = Path(self.config.generated_samples_path)
        metadata = {
            "phase": "diffusionSeedLearning.phase6",
            "source_kind": "phase4_generated_samples_file",
            "generated_samples_path": str(path),
            "k_generate": int(self.config.k_generate),
            "k_accept": int(self.config.k_accept),
            "model_timeout_sec": float(self.config.model_timeout_sec),
            "fallback_to_rule_seed": bool(self.config.fallback_to_rule_seed),
            "precheck_thresholds": {
                "max_start_gap_l2": float(self.config.max_start_gap_l2),
                "max_step_l2": float(self.config.max_step_l2),
                "joint_abs_limit": float(self.config.joint_abs_limit),
            },
        }
        if not path.exists():
            return SeedProviderResult(
                provider_name=self.provider_name,
                mode=mode,
                status="sample_file_missing",
                candidates=[],
                metadata=metadata,
                error=f"generated_samples_path_not_found:{path}",
            )

        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            return SeedProviderResult(
                provider_name=self.provider_name,
                mode=mode,
                status="sample_file_read_failed",
                candidates=[],
                metadata=metadata,
                error=str(exc),
            )

        request_start = request_context.get("start_joint") or []
        dof = int(request_context.get("dof") or len(request_start) or 0)
        flattened = []
        for result in payload.get("results", []):
            if not isinstance(result, dict):
                continue
            q_start = result.get("q_start") or []
            start_gap = _l2_distance(q_start, request_start)
            for sample_index, trajectory in enumerate(result.get("generated", []) or []):
                flattened.append(
                    {
                        "task_index": int(result.get("task_index", len(flattened))),
                        "sample_index": int(sample_index),
                        "q_start": q_start,
                        "start_gap_l2": start_gap,
                        "trajectory": trajectory,
                    }
                )
        flattened.sort(key=lambda item: (
            float("inf") if item["start_gap_l2"] is None else float(item["start_gap_l2"]),
            int(item["task_index"]),
            int(item["sample_index"]),
        ))

        candidates = []
        requested = max(0, int(self.config.k_generate))
        for local_index, item in enumerate(flattened[:requested]):
            precheck = _precheck_trajectory_points(
                item["trajectory"],
                request_start,
                dof=dof,
                max_start_gap_l2=float(self.config.max_start_gap_l2),
                max_step_l2=float(self.config.max_step_l2),
                joint_abs_limit=float(self.config.joint_abs_limit),
            )
            candidates.append(
                SeedCandidate(
                    candidate_id=f"diffusion_seed_{local_index:02d}",
                    source_label=f"diffusion_seed_{local_index:02d}",
                    source_type="diffusion",
                    optimized=False,
                    trajectory_points=item["trajectory"],
                    entered_pool=False,
                    metadata={
                        "provider": self.provider_name,
                        "model_version": str(payload.get("checkpoint") or "<unknown>"),
                        "sample_file_schema_version": payload.get("schema_version"),
                        "sample_task_index": int(item["task_index"]),
                        "sample_index": int(item["sample_index"]),
                        "retrieval_start_gap_l2": item["start_gap_l2"],
                        "source_q_start": item["q_start"],
                        "source_path": str(path),
                    },
                    precheck=precheck,
                )
            )

        status = "generated" if candidates else "no_generated_candidates"
        metadata.update(
            {
                "sample_file_schema_version": payload.get("schema_version"),
                "checkpoint": payload.get("checkpoint"),
                "available_generated_count": int(len(flattened)),
                "precheck_valid_count": int(sum(1 for c in candidates if c.precheck.get("valid"))),
            }
        )
        return SeedProviderResult(
            provider_name=self.provider_name,
            mode=mode,
            status=status,
            candidates=candidates,
            metadata=metadata,
        )


def normalize_diffusion_mode(value: str | None) -> str:
    mode = str(value or "off").strip().lower()
    if mode not in {"off", "shadow", "candidate", "fallback"}:
        return "off"
    return mode


def build_lifecycle_candidate_record(
    *,
    base_record: dict[str, Any],
    source_label: str,
    trajectory_points: list[list[float]],
    metrics_keys: Iterable[str] | None = None,
    metadata: dict[str, Any] | None = None,
    precheck: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a lifecycle candidate record with phase-1 unified fields added."""
    metrics_key_set = set(metrics_keys or ())
    metrics = {
        key: value
        for key, value in base_record.items()
        if key in metrics_key_set
    }
    candidate = SeedCandidate(
        candidate_id=str(base_record.get("candidate_id")),
        source_label=str(source_label),
        trajectory_points=trajectory_points,
        selected=bool(base_record.get("selected")),
        entered_pool=bool(base_record.get("entered_pool", True)),
        metadata=metadata or {},
        precheck=precheck or {},
        metrics=metrics,
    )
    unified = candidate.to_lifecycle_dict()
    return {
        **base_record,
        "source_type": unified["source_type"],
        "optimized": unified["optimized"],
        "trajectory": unified["trajectory"],
        "metadata": unified["metadata"],
        "precheck": unified["precheck"],
        "metrics": unified["metrics"],
    }


def _l2_distance(a: list[Any], b: list[Any]) -> float | None:
    if len(a) != len(b) or not a:
        return None
    try:
        return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))
    except Exception:
        return None


def _precheck_trajectory_points(
    trajectory: list[list[Any]],
    start_joint: list[Any],
    *,
    dof: int,
    max_start_gap_l2: float,
    max_step_l2: float,
    joint_abs_limit: float,
) -> dict[str, Any]:
    shape_valid = (
        isinstance(trajectory, list)
        and len(trajectory) > 0
        and all(isinstance(point, list) for point in trajectory)
        and all(len(point) == int(dof) for point in trajectory)
    )
    if not shape_valid:
        return {
            "valid": False,
            "shape_valid": False,
            "failure_reason": "invalid_shape_or_dof",
            "point_count": len(trajectory) if isinstance(trajectory, list) else 0,
            "dof": int(dof),
        }

    finite = True
    max_abs = 0.0
    max_step = 0.0
    for point_index, point in enumerate(trajectory):
        try:
            values = [float(value) for value in point]
        except Exception:
            finite = False
            break
        if not all(math.isfinite(value) for value in values):
            finite = False
            break
        max_abs = max(max_abs, max(abs(value) for value in values))
        if point_index > 0:
            prev = [float(value) for value in trajectory[point_index - 1]]
            step = math.sqrt(sum((value - prev_value) ** 2 for value, prev_value in zip(values, prev)))
            max_step = max(max_step, step)

    start_gap = _l2_distance(trajectory[0], start_joint)
    eps = 1e-6
    valid = (
        finite
        and start_gap is not None
        and float(start_gap) <= float(max_start_gap_l2) + eps
        and float(max_step) <= float(max_step_l2) + eps
        and float(max_abs) <= float(joint_abs_limit) + eps
    )
    failure_reasons = []
    if not finite:
        failure_reasons.append("non_finite")
    if start_gap is None or float(start_gap) > float(max_start_gap_l2) + eps:
        failure_reasons.append("start_gap_exceeds_threshold")
    if float(max_step) > float(max_step_l2) + eps:
        failure_reasons.append("joint_step_exceeds_threshold")
    if float(max_abs) > float(joint_abs_limit) + eps:
        failure_reasons.append("joint_abs_limit_exceeds_threshold")
    return {
        "valid": bool(valid),
        "shape_valid": True,
        "finite": bool(finite),
        "start_gap_l2": None if start_gap is None else round(float(start_gap), 6),
        "joint_step_max_l2": round(float(max_step), 6),
        "joint_abs_max": round(float(max_abs), 6),
        "thresholds": {
            "max_start_gap_l2": float(max_start_gap_l2),
            "max_step_l2": float(max_step_l2),
            "joint_abs_limit": float(joint_abs_limit),
        },
        "failure_reason": ";".join(failure_reasons) if failure_reasons else None,
    }
