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


def test_select_level_first_candidate_goal_only_ignores_alignment():
    """B4 one-way-seed baseline: ``ignore_alignment_for_selection`` must pick the
    smoothest goal-reaching candidate regardless of its level violation, report
    success, and still record every candidate's *true* alignment deviation so the
    summariser can measure the real level-violation rate."""
    positions = torch.zeros((3, 5, 6), dtype=torch.float32)
    for b, scale in enumerate((0.01, 0.05, 0.2)):
        for t in range(5):
            positions[b, t, :] = scale * t
    level_eval = {
        # cand0 smoothest but violating (40 deg); cand1 level-valid; cand2 worst.
        "max_alignment_deviation": torch.tensor([40.0, 2.0, 60.0]),
        "alignment_valid": torch.tensor([False, True, False]),
    }
    continuity = {
        "start_joint_gap_l2": torch.tensor([0.0, 0.0, 0.0]),
        "joint_step_jump_cost": torch.tensor([0.01, 0.05, 0.2]),
        "twist_smoothness_cost": torch.tensor([0.01, 0.05, 0.2]),
        "joint_step_max_abs": torch.tensor([0.01, 0.05, 0.2]),
        "joint_step_max_l2": torch.tensor([0.02, 0.1, 0.5]),
        "joint_step_mean_l2": torch.tensor([0.02, 0.1, 0.5]),
    }

    level_first = constraints.select_level_first_candidate(
        positions, level_eval, continuity, level_tolerance_deg=3.0, strict_level=True
    )
    # Default level-first: picks the level-valid candidate.
    assert level_first["selected_index"] == 1
    assert level_first["planning_status"] == "success"

    goal_only = constraints.select_level_first_candidate(
        positions,
        level_eval,
        continuity,
        level_tolerance_deg=3.0,
        strict_level=True,
        ignore_alignment_for_selection=True,
    )
    # B4: picks the smoothest candidate ignoring its 40 deg violation, still success,
    # but the TRUE deviation is preserved on both the selected and per-candidate records.
    assert goal_only["selected_index"] == 0
    assert goal_only["planning_status"] == "success"
    assert goal_only["selection_mode"] == "goal_only_alignment_ignored"
    assert abs(goal_only["selected_max_alignment_deviation"] - 40.0) < 1e-6
    assert goal_only["candidate_max_alignment_deviation"] == [40.0, 2.0, 60.0]

