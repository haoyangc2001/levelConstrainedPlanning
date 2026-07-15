import torch

from level_planner_core import constraints


def test_compute_alignment_deviation_identity_quaternion():
    deviation = constraints.compute_alignment_deviation_from_quaternion([1.0, 0.0, 0.0, 0.0])
    assert round(deviation, 6) == 90.0


def test_select_level_first_candidate_prefers_valid_candidate():
    positions = torch.zeros((2, 3, 6), dtype=torch.float32)
    level_eval = {
        "max_alignment_deviation": torch.tensor([4.0, 1.0]),
        "alignment_valid": torch.tensor([False, True]),
    }
    continuity = {
        "start_joint_gap_l2": torch.tensor([0.0, 0.0]),
        "joint_step_jump_cost": torch.tensor([0.0, 0.0]),
        "twist_smoothness_cost": torch.tensor([0.0, 0.0]),
        "joint_step_max_abs": torch.tensor([0.0, 0.0]),
        "joint_step_max_l2": torch.tensor([0.0, 0.0]),
        "joint_step_mean_l2": torch.tensor([0.0, 0.0]),
    }
    result = constraints.select_level_first_candidate(
        positions,
        level_eval,
        continuity,
        level_tolerance_deg=3.0,
        strict_level=True,
    )
    assert result["planning_status"] == "success"
    assert result["selected_index"] == 1
