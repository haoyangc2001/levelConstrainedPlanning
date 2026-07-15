from __future__ import annotations

import torch

from level_planner_core.rule_seed import RuleLevelSeedProvider, RuleSeedProviderConfig


def test_rule_seed_provider_generates_raw_seed_with_lineage() -> None:
    def fake_fk_pose(joint_position: list[float]) -> list[float]:
        return [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]

    def fake_ik_solve(
        position: list[float],
        quaternion: list[float],
        prev_solution: torch.Tensor,
        return_seeds: int,
    ) -> list[torch.Tensor]:
        step = torch.tensor([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
        return [prev_solution.detach().cpu() + step]

    provider = RuleLevelSeedProvider(
        RuleSeedProviderConfig(k_generate=1, k_accept=1, num_waypoints=4),
        fk_pose_fn=fake_fk_pose,
        ik_solve_fn=fake_ik_solve,
    )
    result = provider.generate(
        {
            "start_joint": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "target_pose": [0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "alignment": {"tolerance_deg": 180.0},
        }
    )

    assert result.status == "generated"
    assert len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.source_type == "rule_raw"
    assert candidate.optimized is False
    assert candidate.entered_pool is False
    assert candidate.metadata["seed_family_name"] == "baseline_default"
    assert len(candidate.trajectory_points) == 4
