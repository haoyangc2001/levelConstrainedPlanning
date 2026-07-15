"""CuRobo repair adapter for external trajectory seeds."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class SeedRepairResult:
    success: bool
    status: str
    trajectory: torch.Tensor | None
    optimizer_result: dict[str, Any]
    failure_reason: str | None = None


class SeedRepairAdapter:
    """Small adapter around MotionPlanner.trajopt_solver.solve_pose(seed_traj=...)."""

    def __init__(
        self,
        *,
        motion_planner: Any,
        device: str,
        joint_names: list[str],
    ) -> None:
        self.motion_planner = motion_planner
        self.device = str(device)
        self.joint_names = list(joint_names)

    @property
    def action_horizon(self) -> int:
        return int(self.motion_planner.trajopt_solver.action_horizon)

    def repair_pose_seed(
        self,
        *,
        seed_points: list[list[float]],
        goal_tool_pose: Any,
        current_state: Any,
        return_seeds: int = 1,
    ) -> SeedRepairResult:
        started = time.time()
        try:
            prepared = self.prepare_seed_traj(seed_points)
            result = self.motion_planner.trajopt_solver.solve_pose(
                goal_tool_pose,
                current_state,
                seed_traj=prepared,
                use_implicit_goal=True,
                return_seeds=int(return_seeds),
                finetune_attempts=1,
            )
            success = _result_success(result)
            trajectory = _extract_first_interpolated_trajectory(result) if success else None
            status = str(getattr(result, "status", "success" if success else "trajopt_failed"))
            optimizer_result = {
                "status": status,
                "success": bool(success),
                "result_status": status,
                "solve_time_sec": round(time.time() - started, 6),
                "result_total_time": float(getattr(result, "total_time", 0.0) or 0.0),
                "seed_traj_shape": list(prepared.shape),
                "trajectory_shape": list(trajectory.shape) if trajectory is not None else None,
                "action_horizon": self.action_horizon,
            }
            return SeedRepairResult(
                success=bool(success),
                status=status,
                trajectory=trajectory,
                optimizer_result=optimizer_result,
                failure_reason=None if success else status,
            )
        except Exception as exc:
            return SeedRepairResult(
                success=False,
                status="repair_exception",
                trajectory=None,
                optimizer_result={
                    "status": "repair_exception",
                    "success": False,
                    "solve_time_sec": round(time.time() - started, 6),
                    "exception_type": type(exc).__name__,
                    "failure_reason": str(exc),
                },
                failure_reason=f"{type(exc).__name__}: {exc}",
            )

    def prepare_seed_traj(self, seed_points: list[list[float]]) -> torch.Tensor:
        seed = torch.tensor(seed_points, device=self.device, dtype=torch.float32)
        if seed.ndim != 2:
            raise ValueError(f"seed trajectory must be [T, DOF], got shape={list(seed.shape)}")
        if int(seed.shape[-1]) != len(self.joint_names):
            raise ValueError(
                f"seed trajectory DOF mismatch: got {int(seed.shape[-1])}, expected {len(self.joint_names)}"
            )
        if int(seed.shape[0]) != self.action_horizon:
            seed = _resample_trajectory_linear(seed, self.action_horizon)
        return seed.view(1, 1, self.action_horizon, len(self.joint_names)).contiguous()


def _result_success(result: Any) -> bool:
    if result is None:
        return False
    success = getattr(result, "success", False)
    if hasattr(success, "any"):
        return bool(success.any().item())
    return bool(success)


def _extract_first_interpolated_trajectory(result: Any) -> torch.Tensor:
    plan = result.get_interpolated_plan()
    position = getattr(plan, "position", None)
    if position is None:
        raise ValueError("repair result has no interpolated position")
    if hasattr(position, "detach"):
        position = position.detach().cpu()
    while position.ndim > 2:
        if position.shape[0] == 1:
            position = position.squeeze(0)
        else:
            position = position.reshape(-1, position.shape[-2], position.shape[-1])[0]
    if position.ndim == 1:
        position = position.unsqueeze(0)
    return position.to(dtype=torch.float32)


def _resample_trajectory_linear(trajectory: torch.Tensor, target_horizon: int) -> torch.Tensor:
    if int(trajectory.shape[0]) == int(target_horizon):
        return trajectory.detach().clone()
    if int(trajectory.shape[0]) <= 1:
        return trajectory.detach().clone().repeat(int(target_horizon), 1)
    import torch.nn.functional as F

    return (
        F.interpolate(
            trajectory.transpose(0, 1).unsqueeze(0),
            size=int(target_horizon),
            mode="linear",
            align_corners=True,
        )
        .squeeze(0)
        .transpose(0, 1)
        .contiguous()
    )
