#!/usr/bin/env python3
"""Reachable-goal provider for the C1a/C2 sampler.

The C1a task requires *independent* start/goal sampling with a reachability /
IK pre-check so that scaled-up data generation (C2) does not burn GPU time on
structurally infeasible (start, goal) pairs.  The naive "jitter a pose in a
wide workspace box" approach produced ~1-3% positive yield because most goal
poses were simply unreachable by the SR5 arm.

This module closes that gap the robust way: instead of guessing a Cartesian
pose and hoping an IK solution exists, it samples a random *feasible joint
configuration* (uniform within URDF joint limits) and returns its forward-
kinematics image as the goal pose.  Every goal produced this way is reachable
by construction -- the FK config is itself a valid IK solution -- so the only
thing the downstream planner must still discover is a smooth, collision-free,
level-satisfying *trajectory* to get there.

FK is evaluated on the GPU via cuRobo's lightweight ``Kinematics`` model
(no collision-sphere generation, no full MotionPlanner), so loading is cheap
and a whole request batch's goals can be produced in one call.

Usage is intentionally optional: ``sample_tasks.py`` only imports and
constructs a :class:`ReachableGoalSampler` when ``--reachable-goals`` is passed,
keeping the default sampler a pure-CPU, dependency-free path.
"""

from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any


def _tool_frames_from_cfg(robot_cfg: dict[str, Any]) -> list[str] | None:
    """Best-effort tool frame from a resolved robot_cfg dict (else None)."""
    kinematics = (robot_cfg.get("robot_cfg") or {}).get("kinematics") or {}
    tool = kinematics.get("ee_link") or kinematics.get("tool_frame")
    return [str(tool)] if tool else None


class ReachableGoalSampler:
    """Sample reachable goal poses as FK images of feasible joint configs.

    Loads a standalone cuRobo ``Kinematics`` model once. ``sample_goal_pose``
    draws a uniform-in-limits joint config and returns its tool pose as a
    ``[x, y, z, qw, qx, qy, qz]`` list; for level-active classes the caller can
    override the returned orientation with a level-aligned one (the level
    constraint only cares about a single tool axis, not the full orientation).
    """

    def __init__(
        self,
        *,
        robot_config_path: Path,
        device: str = "cuda:0",
        margin_frac: float = 0.05,
    ) -> None:
        import torch  # local import: only when reachable goals are requested
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import JointState

        from level_planner_core.robot_assets import resolve_robot_config

        self._torch = torch
        self._JointState = JointState
        # Reuse the planner's own config resolution (absolute URDF/asset paths,
        # inline collision spheres skipped -- FK needs none) so this FK model
        # matches the planner's kinematic chain exactly.
        robot_cfg = resolve_robot_config(
            robot_config_path=Path(robot_config_path),
            auto_generate_spheres=False,
        )
        tool_frames = _tool_frames_from_cfg(robot_cfg)
        cfg = KinematicsCfg.from_data_dict(robot_cfg, tool_frames=tool_frames)
        self._robot = Kinematics(cfg)
        self._device = str(device)
        self._tool_frame = self._robot.tool_frames[0]
        self._joint_names = list(self._robot.joint_names)
        self._dof = int(self._robot.get_dof())
        self._lower, self._upper = self._joint_limit_bounds(margin_frac)

    # -- limits -----------------------------------------------------------
    def _joint_limit_bounds(self, margin_frac: float) -> tuple[list[float], list[float]]:
        """Return per-joint (lower, upper) sampling bounds, shrunk by a margin.

        cuRobo carries joint limits on the kinematics model; we pull those and
        pull the range in slightly (``margin_frac`` of the span on each side)
        so sampled configs sit away from the exact limit where IK/trajopt tend
        to struggle.
        """
        lower = [-math.pi] * self._dof
        upper = [math.pi] * self._dof
        limits = getattr(self._robot, "joint_limits", None) or getattr(
            self._robot, "get_joint_limits", None
        )
        try:
            jl = limits() if callable(limits) else limits
            pos = getattr(jl, "position", None)
            if pos is not None:
                low_t, up_t = pos[0].detach().cpu().tolist(), pos[1].detach().cpu().tolist()
                lower = [float(v) for v in low_t][: self._dof]
                upper = [float(v) for v in up_t][: self._dof]
        except Exception:
            pass  # fall back to [-pi, pi]; sampling still works, just wider
        out_low, out_up = [], []
        for lo, hi in zip(lower, upper):
            span = max(hi - lo, 1e-6)
            m = span * float(margin_frac)
            out_low.append(lo + m)
            out_up.append(hi - m)
        return out_low, out_up

    # -- sampling ---------------------------------------------------------
    def sample_goal_joint(self, rng: random.Random) -> list[float]:
        """Uniformly sample a feasible goal joint config within (shrunk) limits."""
        return [rng.uniform(lo, hi) for lo, hi in zip(self._lower, self._upper)]

    def fk_pose(self, joint_config: list[float]) -> list[float]:
        """Forward-kinematics tool pose for one joint config.

        Returns ``[x, y, z, qw, qx, qy, qz]`` (quaternion in wxyz order to
        match the request schema and the planner's expectations).
        """
        torch = self._torch
        q = torch.tensor([joint_config], device=self._device, dtype=torch.float32)
        state = self._robot.compute_kinematics(
            self._JointState.from_position(q, joint_names=self._joint_names)
        )
        pose = state.tool_poses.get_link_pose(self._tool_frame)
        pos = pose.position.reshape(-1, 3)[0].detach().cpu().tolist()
        quat = pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()
        return [float(v) for v in pos] + [float(v) for v in quat]

    def _fk_quat_batch(self, q):  # -> torch.Tensor [N, 4] (wxyz)
        """Batched FK returning tool quaternions [N, 4] (wxyz) for configs q [N, DOF]."""
        state = self._robot.compute_kinematics(
            self._JointState.from_position(q, joint_names=self._joint_names)
        )
        pose = state.tool_poses.get_link_pose(self._tool_frame)
        return pose.quaternion.reshape(-1, 4)

    def _level_deviation_deg_batch(self, quat, local_axis, target_world_axis):
        """Angle (deg) between the tool's local_axis (rotated to world) and target, per row."""
        torch = self._torch
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        lx, ly, lz = local_axis
        # Rotate local_axis by quaternion (vectorized q * v * q^-1).
        tx = 2.0 * (y * lz - z * ly)
        ty = 2.0 * (z * lx - x * lz)
        tz = 2.0 * (x * ly - y * lx)
        ax = lx + w * tx + (y * tz - z * ty)
        ay = ly + w * ty + (z * tx - x * tz)
        az = lz + w * tz + (x * ty - y * tx)
        norm = torch.sqrt(ax * ax + ay * ay + az * az).clamp_min(1e-9)
        ax, ay, az = ax / norm, ay / norm, az / norm
        tgt = target_world_axis
        dot = (ax * tgt[0] + ay * tgt[1] + az * tgt[2]).clamp(-1.0, 1.0)
        return torch.rad2deg(torch.arccos(dot))

    def sample_level_joint(
        self,
        rng: random.Random,
        *,
        local_axis: list[float],
        target_world_axis: list[float],
        tol_deg: float,
        batch: int = 4096,
        max_batches: int = 32,
    ) -> tuple[list[float], float] | tuple[None, None]:
        """Rejection-sample a feasible joint config whose FK tool axis is level.

        For a level-active class, *both* endpoints must sit on the level manifold
        (tool ``local_axis`` aligned with ``target_world_axis`` within
        ``tol_deg``); otherwise no constraint-satisfying trajectory can connect
        them. Random configs rarely land on this codim-2 manifold, so we draw
        large GPU batches, keep the rows already within tolerance, and return the
        closest-to-level accepted config. Returns ``(joint, deviation_deg)`` or
        ``(None, None)`` if no in-tolerance config was found within the budget.
        """
        torch = self._torch
        lo = torch.tensor(self._lower, device=self._device, dtype=torch.float32)
        hi = torch.tensor(self._upper, device=self._device, dtype=torch.float32)
        span = (hi - lo)
        best_joint: list[float] | None = None
        best_dev = float("inf")
        for _ in range(int(max_batches)):
            # Deterministic per-call seeding via rng so runs remain reproducible.
            g = torch.Generator(device=self._device)
            g.manual_seed(rng.randint(0, 2**63 - 1))
            u = torch.rand(int(batch), self._dof, device=self._device, generator=g)
            q = lo + u * span
            quat = self._fk_quat_batch(q)
            dev = self._level_deviation_deg_batch(quat, local_axis, target_world_axis)
            mask = dev <= float(tol_deg)
            if bool(mask.any()):
                idx = int(torch.argmin(torch.where(mask, dev, dev + 1e6)).item())
                cand_dev = float(dev[idx].item())
                if cand_dev < best_dev:
                    best_dev = cand_dev
                    best_joint = [round(float(v), 6) for v in q[idx].detach().cpu().tolist()]
                return best_joint, best_dev
            # Track global best even if none within tol yet (for diagnostics).
            idx = int(torch.argmin(dev).item())
            if float(dev[idx].item()) < best_dev:
                best_dev = float(dev[idx].item())
                best_joint = [round(float(v), 6) for v in q[idx].detach().cpu().tolist()]
        return None, None

    def sample_reachable_level_pose(
        self,
        rng: random.Random,
        *,
        local_axis: list[float],
        target_world_axis: list[float],
        tol_deg: float,
    ) -> tuple[list[float], list[float]] | tuple[None, None]:
        """Sample a level, reachable goal joint and its (exactly level-projected) pose."""
        joint, _dev = self.sample_level_joint(
            rng,
            local_axis=local_axis,
            target_world_axis=target_world_axis,
            tol_deg=tol_deg,
        )
        if joint is None:
            return None, None
        pose = self.fk_pose(joint)
        return pose, joint

    def sample_level_neighbor(
        self,
        rng: random.Random,
        anchor_joint: list[float],
        *,
        local_axis: list[float],
        target_world_axis: list[float],
        tol_deg: float,
        radius_rad: float,
        min_radius_rad: float = 0.0,
        batch: int = 4096,
        max_batches: int = 32,
    ) -> tuple[list[float], float] | tuple[None, None]:
        """Sample a level joint config within a joint-L2 ball of ``anchor_joint``.

        Two *independently* drawn level configs generally sit in disconnected
        charts of the level manifold (different elbow/base IK branches), so the
        joint-space geodesic between them leaves the manifold and no level path
        connects them (empirically ~0% yield). Restricting the goal to a bounded
        neighborhood of a level start keeps both endpoints on the *same* chart, so
        a level-preserving trajectory provably exists, while ``min_radius_rad``
        keeps the motion non-trivial (unlike base-jitter's ~0.08 rad micro-moves).

        Draws perturbations uniformly in the joint ball, clips to limits, keeps
        rows whose FK tool axis is level within ``tol_deg`` and whose L2 offset is
        in ``[min_radius_rad, radius_rad]``, and returns the accepted config
        farthest from the anchor (a real motion). ``(None, None)`` if the budget
        is exhausted.
        """
        torch = self._torch
        lo = torch.tensor(self._lower, device=self._device, dtype=torch.float32)
        hi = torch.tensor(self._upper, device=self._device, dtype=torch.float32)
        anchor = torch.tensor(anchor_joint, device=self._device, dtype=torch.float32)
        for _ in range(int(max_batches)):
            g = torch.Generator(device=self._device)
            g.manual_seed(rng.randint(0, 2**63 - 1))
            # Uniform in a ball: direction ~ normal, radius ~ U^(1/dof) scaled.
            direction = torch.randn(int(batch), self._dof, device=self._device, generator=g)
            direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-9)
            rad = torch.rand(int(batch), 1, device=self._device, generator=g) ** (1.0 / self._dof)
            offset = direction * rad * float(radius_rad)
            q = torch.clamp(anchor.unsqueeze(0) + offset, lo, hi)
            l2 = (q - anchor.unsqueeze(0)).norm(dim=1)
            quat = self._fk_quat_batch(q)
            dev = self._level_deviation_deg_batch(quat, local_axis, target_world_axis)
            mask = (dev <= float(tol_deg)) & (l2 >= float(min_radius_rad))
            if bool(mask.any()):
                # Prefer the farthest accepted config (largest real motion).
                score = torch.where(mask, l2, torch.full_like(l2, -1.0))
                idx = int(torch.argmax(score).item())
                return (
                    [round(float(v), 6) for v in q[idx].detach().cpu().tolist()],
                    float(dev[idx].item()),
                )
        return None, None

    def sample_reachable_pose(self, rng: random.Random) -> tuple[list[float], list[float]]:
        """Sample a feasible goal joint config and its reachable tool pose.

        Returns ``(pose7, goal_joint)`` so the caller can record the witness
        joint config (for traceability / optional start!=goal enforcement).
        """
        goal_joint = self.sample_goal_joint(rng)
        return self.fk_pose(goal_joint), goal_joint

    @property
    def dof(self) -> int:
        return self._dof

    @property
    def joint_names(self) -> list[str]:
        return list(self._joint_names)
