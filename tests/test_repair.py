from __future__ import annotations

from types import SimpleNamespace

from level_planner_core.repair import SeedRepairAdapter


def test_prepare_seed_traj_resamples_and_packs_shape() -> None:
    fake_planner = SimpleNamespace(
        trajopt_solver=SimpleNamespace(action_horizon=5),
    )
    adapter = SeedRepairAdapter(
        motion_planner=fake_planner,
        device="cpu",
        joint_names=["j1", "j2"],
    )
    prepared = adapter.prepare_seed_traj(
        [
            [0.0, 0.0],
            [1.0, 1.0],
        ]
    )
    assert list(prepared.shape) == [1, 1, 5, 2]
    assert prepared[0, 0, 0, 0].item() == 0.0
    assert prepared[0, 0, -1, 0].item() == 1.0
