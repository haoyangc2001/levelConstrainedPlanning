from __future__ import annotations

import json
from pathlib import Path

from tools.dataset.sample_tasks import (
    DEFAULT_BASE_REQUEST,
    build_manifest,
    generate_task_requests,
    write_requests_jsonl,
)
from tools.dataset.run_lifecycle_batch import load_requests, request_output_dir
from tools.dataset.write_dataset_pointer import build_pointer


def test_sample_tasks_are_deterministic_and_stratified(tmp_path: Path) -> None:
    tasks_a = generate_task_requests(
        base_request_paths=[DEFAULT_BASE_REQUEST],
        count=6,
        seed=1234,
        difficulty="mixed",
        obstacle_case="mixed",
        modes=["mixed", "rule", "shadow"],
    )
    tasks_b = generate_task_requests(
        base_request_paths=[DEFAULT_BASE_REQUEST],
        count=6,
        seed=1234,
        difficulty="mixed",
        obstacle_case="mixed",
        modes=["mixed", "rule", "shadow"],
    )
    assert tasks_a == tasks_b
    assert [task["metadata"]["sampling"]["difficulty_bucket"] for task in tasks_a] == [
        "easy",
        "medium",
        "hard",
        "easy",
        "medium",
        "hard",
    ]
    assert [task["seed_policy"]["mode"] for task in tasks_a] == [
        "mixed",
        "rule",
        "shadow",
        "mixed",
        "rule",
        "shadow",
    ]

    out = tmp_path / "requests.jsonl"
    write_requests_jsonl(tasks_a, out)
    assert len(load_requests(out)) == 6
    manifest = build_manifest(
        tasks=tasks_a,
        out=out,
        base_request_paths=[DEFAULT_BASE_REQUEST],
        seed=1234,
        modes=["mixed", "rule", "shadow"],
    )
    assert manifest["request_count"] == 6
    assert manifest["difficulty_bucket_counts"] == {"easy": 2, "hard": 2, "medium": 2}


def test_batch_request_output_dir_is_stable(tmp_path: Path) -> None:
    request = {"request_id": "bad/id with spaces"}
    out = request_output_dir(tmp_path, 7, request)
    assert out.name == "00007_bad_id_with_spaces"


def test_dataset_pointer_collects_manifest_counts(tmp_path: Path) -> None:
    samples = tmp_path / "samples_validated.jsonl"
    samples.write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "sample_count": 3,
                "candidate_count": 3,
                "request_count": 2,
                "positive_for_diffusion": 1,
                "positive_for_critic": 1,
                "negative_for_critic": 2,
                "source_type_counts": {"planner_native": 1, "rule_seed": 2},
                "candidate_status_counts": {"success": 1, "failed_planner": 2},
            }
        ),
        encoding="utf-8",
    )
    pointer = build_pointer(
        dataset_name="unit_dataset",
        samples=samples,
        manifest=manifest,
        validator_report=None,
        batch_summary=None,
        sampling_manifest=None,
    )
    assert pointer["dataset"]["request_count"] == 2
    assert pointer["dataset"]["training_dataset"]["exists"]
    assert pointer["dataset"]["training_dataset"]["sha256"]
