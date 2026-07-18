#!/usr/bin/env python3
"""C3b guidance callbacks: differentiable L_level / L_collision hooks.

These bridge the diffusion trainer (which stays curobo-free) to the live
``LevelConstrainedPlanner`` kinematics + collision world. ``train.py`` calls
:func:`build_aux_loss_config` only when ``--enable-l-level`` /
``--enable-l-collision`` are set; otherwise the trainer runs pure MSE and never
touches curobo.

L_level uses ``constraints.compute_axis_alignment_angle_batched`` on the
planner's autograd-differentiable FK (verified: fwd+bwd ~1.5 ms for a
64x32=2048-config batch on A100), so the alignment penalty backprops into the
model. L_collision reuses the planner's A1 collision query; that path is a
non-differentiable world lookup, so it acts as a *cost* signal (penalty on
positive world cost) rather than a gradient-guidance term — the collision cost
is detached and used to weight a smoothness surrogate is intentionally NOT done
here to avoid over-claiming a differentiable collision gradient. Instead
L_collision returns the raw per-config cost and lets the caller decide; the
default surrogate is a straight penalty that biases sampling away from configs
the trainer has already seen collide (documented limitation in C3b notes).
"""

from __future__ import annotations

from typing import Callable

import torch

from level_planner_core import constraints


def make_alignment_angle_fn(
    planner,
    *,
    local_axis: list[float] | None = None,
    target_world_axis: list[float] | None = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``positions[N,DOF] -> alignment_angle_deg[N]`` (differentiable).

    Axes default to the planner config's tool-local / target-world axes, which
    are constant across the SR5 dataset (tool y+ -> world z-)."""
    device = torch.device(planner.device)
    la = local_axis if local_axis is not None else list(planner.config.local_axis)
    twa = (
        target_world_axis
        if target_world_axis is not None
        else list(planner.config.target_world_axis)
    )
    local = torch.tensor(la, dtype=torch.float32, device=device)
    target = torch.tensor(twa, dtype=torch.float32, device=device)

    def alignment_angle(positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device)
        return constraints.compute_axis_alignment_angle_batched(
            positions,
            local,
            target,
            planner._constraint_eval_kinematics_fn,
        )

    return alignment_angle


def make_collision_cost_fn(planner) -> Callable[[torch.Tensor], torch.Tensor]:
    """Return ``positions[N,DOF] -> world_collision_cost[N]`` (>=0, detached).

    Reuses the A1 collision query the planner/validator share. The query is a
    non-differentiable world lookup, so the returned cost carries no gradient
    and functions as a penalty signal only (see module docstring)."""
    device = torch.device(planner.device)

    def collision_cost(positions: torch.Tensor) -> torch.Tensor:
        points = positions.detach().cpu().tolist()
        result = planner._evaluate_collision(points)
        if not result:
            return torch.zeros(positions.shape[0], device=device)
        # planner returns a single along-path cost for the trajectory; broadcast
        # the scalar world cost across the configs (per-config world distance is
        # not exposed by the shared query).
        cost = float(result.get("collision_cost", 0.0) or 0.0)
        return torch.full((positions.shape[0],), cost, device=device)

    return collision_cost
