"""Structured result helpers for the standalone level planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STATUS_SUCCESS = "success"
STATUS_FAILED_PRECHECK = "failed_precheck"
STATUS_FAILED_GOAL = "failed_goal"
STATUS_FAILED_ALIGNMENT = "failed_alignment_constraint"
STATUS_FAILED_PLANNER = "failed_planner"
STATUS_FAILED_INTERNAL = "failed_internal_error"


@dataclass
class PlannerArtifacts:
    result_json: str | None = None
    selected_trajectory_json: str | None = None
    candidate_summary_json: str | None = None
    lifecycle_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerResult:
    request_id: str
    status: str
    failure_reason: str | None = None
    selected_trajectory: list[list[float]] | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    seed_provider_reports: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[dict[str, Any]] = field(default_factory=list)
    artifacts: PlannerArtifacts = field(default_factory=PlannerArtifacts)
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "selected_trajectory": self.selected_trajectory,
            "metrics": self.metrics,
            "seed_provider_reports": self.seed_provider_reports,
            "candidates": self.candidates,
            "artifacts": self.artifacts.to_dict(),
        }

