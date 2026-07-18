"""Structured result helpers for the standalone level planner."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


STATUS_SUCCESS = "success"
STATUS_FAILED_PRECHECK = "failed_precheck"
STATUS_FAILED_GOAL = "failed_goal"
STATUS_FAILED_ALIGNMENT = "failed_alignment_constraint"
STATUS_FAILED_VALIDATION = "failed_hard_validation"
STATUS_FAILED_PLANNER = "failed_planner"
STATUS_FAILED_INTERNAL = "failed_internal_error"


@dataclass
class PlannerArtifacts:
    result_json: str | None = None
    selected_trajectory_json: str | None = None
    candidate_summary_json: str | None = None
    candidates_jsonl: str | None = None
    lifecycle_json: str | None = None
    planner_run_json: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateRecord:
    candidate_id: str
    run_id: str
    request_id: str
    source_lineage: dict[str, Any]
    trajectory: dict[str, Any]
    lifecycle: dict[str, Any]
    precheck: dict[str, Any] = field(default_factory=dict)
    optimizer_result: dict[str, Any] = field(default_factory=dict)
    validator_metrics: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    failure_stage: str | None = None
    failure_reason: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "candidate_record.v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PlannerRunRecord:
    run_id: str
    request_id: str
    created_at: str
    robot_profile: str
    request: dict[str, Any]
    normalized_request: dict[str, Any]
    world_summary: dict[str, Any]
    seed_policy: dict[str, Any]
    seed_provider_reports: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    fallback_trace: list[dict[str, Any]]
    result_status: str
    failure_reason: str | None = None
    selected_candidate_id: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, Any] = field(default_factory=dict)
    schema_version: str = "planner_run_record.v1"

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
    candidate_records: list[dict[str, Any]] = field(default_factory=list)
    planner_run_record: dict[str, Any] = field(default_factory=dict)
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
            "candidate_records": self.candidate_records,
            "planner_run_record": self.planner_run_record,
            "artifacts": self.artifacts.to_dict(),
        }
