from __future__ import annotations

from tools.dataset.baselines import ompl_bridge
from tools.dataset.baselines.ompl_rrtc import _solve_time_budget
from tools.dataset.methods import get_method


def test_uniform_retiming_respects_joint_velocity_limits() -> None:
    points = [[0.0, 0.0], [1.0, 0.5], [2.0, 1.0]]
    limits = [
        {"velocity": 2.0},
        {"velocity": 1.0},
    ]
    dt = ompl_bridge.uniform_retiming_dt(points, limits, speed_scale=0.5)
    assert dt is not None
    assert abs(dt - 1.0) < 1.0e-8


def test_resample_path_preserves_endpoints() -> None:
    result = ompl_bridge._resample_path([[0.0, 0.0], [1.0, 2.0]], 5)
    assert len(result) == 5
    assert result[0] == [0.0, 0.0]
    assert result[-1] == [1.0, 2.0]


def test_rrtc_budget_uses_compute_axis_and_timeout_guard() -> None:
    request = {
        "metadata": {"compute_budget_solve_calls": 8},
        "seed_policy": {"timeout_sec": 3.0},
    }
    assert _solve_time_budget(request) == 3.0


def test_ompl_rrtc_registered_as_external_method() -> None:
    spec = get_method("baseline/ompl_rrtc")
    assert spec.external
    assert spec.runner is not None

