from __future__ import annotations

from level_planner_core.condition import CONDITION_DIM, build_condition_values
from level_planner_core.learned_seed import (
    CheckpointDiffusionSeedProvider,
    CheckpointDiffusionSeedProviderConfig,
)


def test_condition_builder_uses_shared_runtime_schema() -> None:
    values = build_condition_values(
        {
            "start_joint": [1, 2, 3, 4, 5, 6],
            "target_pose": [0.1, 0.2, 0.3, 1, 0, 0, 0],
            "alignment": {"tolerance_deg": 7.5},
            "world_summary": {"total_box_count": 2},
        }
    )
    assert len(values) == CONDITION_DIM
    assert values[:6] == [1, 2, 3, 4, 5, 6]
    assert values[-2:] == [7.5, 2.0]


def test_checkpoint_provider_missing_checkpoint_is_non_blocking() -> None:
    provider = CheckpointDiffusionSeedProvider(
        CheckpointDiffusionSeedProviderConfig(
            mode="candidate",
            diffusion_checkpoint_path="/tmp/does-not-exist-diffusion.pt",
            critic_checkpoint_path="/tmp/does-not-exist-critic.pt",
            k_generate=2,
            k_accept=1,
            device="cpu",
        )
    )
    result = provider.generate(
        {
            "start_joint": [0.0] * 6,
            "target_pose": [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "alignment": {"tolerance_deg": 3.0},
        }
    )
    assert result.status == "checkpoint_missing"
    assert result.candidates == []
    assert result.error and "diffusion_checkpoint_not_found" in result.error
