from __future__ import annotations

from pathlib import Path

from level_planner_core import validators


def test_load_joint_limits_from_urdf_ordered_by_joint_names() -> None:
    limits = validators.load_joint_limits_from_robot_config(
        Path("configs/robot/xms5_r800_w4g3b4c_v2.yml"),
        [
            "XMS5-R800-W4G3B4C_joint_1",
            "XMS5-R800-W4G3B4C_joint_2",
        ],
    )
    assert limits[0]["lower"] == -6.2832
    assert limits[0]["upper"] == 6.2832
    assert limits[1]["lower"] == -2.7925
    assert limits[1]["upper"] == 2.618


def test_joint_limit_validator_reports_violation() -> None:
    limits = [
        {"joint_name": "j1", "lower": -1.0, "upper": 1.0},
        {"joint_name": "j2", "lower": -2.0, "upper": 2.0},
    ]
    report = validators.evaluate_joint_limits(
        [[0.0, 0.0], [1.5, 0.0]],
        limits,
    )
    assert not report["valid"]
    assert report["failure_reason"] == "failed_joint_limit"
    assert report["violating_joint_names"] == ["j1"]


def test_hard_constraints_report_combines_required_checks() -> None:
    limits = [
        {"joint_name": f"j{i}", "lower": -2.0, "upper": 2.0}
        for i in range(6)
    ]
    report = validators.evaluate_hard_constraints(
        trajectory_points=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        start_joint=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        joint_limits=limits,
        metrics={
            "max_alignment_deviation_deg": 1.0,
            "mean_alignment_deviation_deg": 0.5,
            "position_error_m": 0.001,
            "orientation_error_rad": 0.01,
        },
        alignment_tolerance_deg=3.0,
        optimizer_success=True,
        world_summary={"total_box_count": 0},
    )
    assert report["valid"]
    assert report["checks"]["joint_limit"]["valid"]
    assert report["checks"]["collision_safety"]["status"] == "unchecked"
    assert report["checks"]["velocity_acceleration"]["valid"]
