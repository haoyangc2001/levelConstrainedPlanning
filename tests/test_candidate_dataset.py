from __future__ import annotations

import sys
from pathlib import Path

from tools.dataset.validate_candidate_dataset import validate_candidate_dataset


FIXTURE = Path(__file__).parent / "fixtures" / "candidate_dataset_smoke.jsonl"


def test_candidate_dataset_validator_accepts_smoke_fixture() -> None:
    report = validate_candidate_dataset(
        FIXTURE,
        require_positive=True,
        require_negative=True,
    )
    assert report["valid"], report["errors"]
    assert report["sample_count"] == 4
    assert report["positive_for_diffusion"] == 1
    assert report["negative_for_critic"] == 3


def test_learning_datasets_read_candidate_fixture() -> None:
    learning_dir = Path(__file__).resolve().parents[1] / "tools" / "learning" / "diffusion_seed_learning"
    sys.path.insert(0, str(learning_dir))
    try:
        from critic import SuccessCriticDataset
        from dataset import TrajectorySeedDataset

        diffusion_dataset = TrajectorySeedDataset(FIXTURE, horizon=4, positive_only=True)
        critic_dataset = SuccessCriticDataset(FIXTURE, horizon=4)
    finally:
        sys.path.remove(str(learning_dir))

    assert len(diffusion_dataset) == 1
    assert diffusion_dataset.dof == 6
    assert len(critic_dataset) == 3
