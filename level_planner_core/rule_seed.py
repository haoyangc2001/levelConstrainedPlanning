"""Rule-based level seed generation for the standalone planner."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from . import constraints
from .seed_provider import SeedCandidate, SeedProviderResult


FkPoseFn = Callable[[list[float]], list[float]]
IkSolveFn = Callable[[list[float], list[float], torch.Tensor, int], list[torch.Tensor]]


def default_rule_seed_family_configs() -> list[dict[str, Any]]:
    return [
        {
            "source_label": "alignment_seed_family_1",
            "seed_family_name": "baseline_default",
            "twist_schedule_mode": "uniform_shortest",
            "goal_anchor_rank": 1,
            "selection_mode": "default_score",
        },
        {
            "source_label": "alignment_seed_family_2",
            "seed_family_name": "goal_anchor_rank_2",
            "twist_schedule_mode": "uniform_shortest",
            "goal_anchor_rank": 2,
            "selection_mode": "default_score",
        },
        {
            "source_label": "alignment_seed_family_3",
            "seed_family_name": "twist_delayed_to_goal",
            "twist_schedule_mode": "delayed_to_goal",
            "goal_anchor_rank": 1,
            "selection_mode": "default_score",
        },
        {
            "source_label": "alignment_seed_family_4",
            "seed_family_name": "branch_selection_in_limit_best",
            "twist_schedule_mode": "uniform_shortest",
            "goal_anchor_rank": 1,
            "selection_mode": "in_limit_best",
        },
    ]


@dataclass
class RuleSeedProviderConfig:
    mode: str = "candidate"
    k_generate: int = 4
    k_accept: int = 4
    timeout_sec: float = 1.0
    num_waypoints: int = 30
    ik_return_seeds: int = 32
    joint_abs_limit: float = 6.2832
    include_smooth_bridge_variants: bool = True
    smoothing_passes: int = 2
    bridge_radius: int = 2
    family_configs: list[dict[str, Any]] = field(default_factory=default_rule_seed_family_configs)


class RuleLevelSeedProvider:
    provider_name = "rule_seed"

    def __init__(
        self,
        config: RuleSeedProviderConfig | None = None,
        *,
        fk_pose_fn: FkPoseFn,
        ik_solve_fn: IkSolveFn,
    ) -> None:
        self.config = config or RuleSeedProviderConfig()
        self._fk_pose_fn = fk_pose_fn
        self._ik_solve_fn = ik_solve_fn

    def generate(self, request_context: dict[str, Any]) -> SeedProviderResult:
        started = time.time()
        metadata = {
            "phase": "closed_loop.phase3_rule_seed",
            "runtime_effect": "raw_rule_seed_generation_only_no_repair_pool_insertion",
            "num_waypoints": int(self.config.num_waypoints),
            "ik_return_seeds": int(self.config.ik_return_seeds),
            "configured_family_count": len(self.config.family_configs),
            "k_generate": int(self.config.k_generate),
            "k_accept": int(self.config.k_accept),
        }
        if int(self.config.k_generate) <= 0:
            return SeedProviderResult(
                provider_name=self.provider_name,
                mode=self.config.mode,
                status="available_not_generated",
                candidates=[],
                metadata=metadata,
            )

        start_joint = [float(v) for v in request_context.get("start_joint", [])]
        target_pose = [float(v) for v in request_context.get("target_pose", [])]
        alignment = request_context.get("alignment") or {}
        tolerance = float(alignment.get("tolerance_deg", 3.0))
        family_configs = self.config.family_configs[: max(1, int(self.config.k_generate))]
        candidates: list[SeedCandidate] = []
        attempts: list[dict[str, Any]] = []
        for index, family_config in enumerate(family_configs):
            if time.time() - started > float(self.config.timeout_sec):
                attempts.append(
                    {
                        "family_index": index,
                        "source_label": family_config.get("source_label"),
                        "success": False,
                        "failure_reason": "rule_seed_timeout",
                    }
                )
                break
            result = self._generate_family_seed(
                start_joint=start_joint,
                target_pose=target_pose,
                alignment_tolerance_deg=tolerance,
                family_config=family_config,
            )
            attempts.append({key: value for key, value in result.items() if key != "trajectory_points"})
            if result.get("success") and result.get("trajectory_points"):
                local_index = len(candidates)
                precheck = _precheck_rule_seed(
                    result["trajectory_points"],
                    start_joint=start_joint,
                    dof=len(start_joint),
                    joint_abs_limit=float(self.config.joint_abs_limit),
                )
                candidates.append(
                    self._make_candidate(
                        local_index=local_index,
                        family_config=family_config,
                        trajectory_points=result["trajectory_points"],
                        precheck=precheck,
                        result=result,
                        variant_name="raw",
                    )
                )
                if len(candidates) >= max(1, int(self.config.k_accept or self.config.k_generate)):
                    break
                if self.config.include_smooth_bridge_variants:
                    for variant_name, variant_points in self._make_variants(result["trajectory_points"]):
                        if len(candidates) >= max(1, int(self.config.k_accept or self.config.k_generate)):
                            break
                        variant_precheck = _precheck_rule_seed(
                            variant_points,
                            start_joint=start_joint,
                            dof=len(start_joint),
                            joint_abs_limit=float(self.config.joint_abs_limit),
                        )
                        candidates.append(
                            self._make_candidate(
                                local_index=len(candidates),
                                family_config=family_config,
                                trajectory_points=variant_points,
                                precheck=variant_precheck,
                                result=result,
                                variant_name=variant_name,
                            )
                        )
        metadata["attempts"] = attempts
        metadata["elapsed_sec"] = round(time.time() - started, 6)
        metadata["precheck_valid_count"] = sum(1 for item in candidates if item.precheck.get("valid"))
        return SeedProviderResult(
            provider_name=self.provider_name,
            mode=self.config.mode,
            status="generated" if candidates else "no_rule_seed_generated",
            candidates=candidates,
            metadata=metadata,
        )

    def _make_candidate(
        self,
        *,
        local_index: int,
        family_config: dict[str, Any],
        trajectory_points: list[list[float]],
        precheck: dict[str, Any],
        result: dict[str, Any],
        variant_name: str,
    ) -> SeedCandidate:
        base_label = str(family_config.get("source_label") or f"alignment_seed_family_{local_index + 1}")
        source_label = base_label if variant_name == "raw" else f"{base_label}_{variant_name}"
        return SeedCandidate(
            candidate_id=f"rule_seed_{local_index:02d}",
            source_label=source_label,
            source_type="rule_raw",
            optimized=False,
            selected=False,
            entered_pool=False,
            trajectory_points=trajectory_points,
            metadata={
                "provider": self.provider_name,
                "seed_family_name": family_config.get("seed_family_name"),
                "seed_family_config": dict(family_config),
                "variant_name": variant_name,
                "goal_anchor_rank_used": result.get("goal_anchor_rank_used"),
                "ik_fail_count": result.get("ik_fail_count"),
                "max_step_jump_l2": _max_step_l2(trajectory_points),
                "raw_seed_max_step_jump_l2": result.get("max_step_jump_l2"),
                "split_status": "deferred_to_phase4_cspace_repair",
                "runtime_effect": "raw_seed_waiting_for_phase4_repair",
            },
            precheck=precheck,
            metrics={
                "raw_seed_max_step_jump_l2": _max_step_l2(trajectory_points),
                "raw_seed_ik_fail_count": result.get("ik_fail_count"),
            },
        )

    def _make_variants(self, trajectory_points: list[list[float]]) -> list[tuple[str, list[list[float]]]]:
        variants: list[tuple[str, list[list[float]]]] = []
        variants.append(("smoothed", _smooth_points(trajectory_points, passes=int(self.config.smoothing_passes))))
        jump_index = _max_step_jump_index(trajectory_points)
        variants.append(
            (
                "bridged",
                _bridge_points(
                    trajectory_points,
                    jump_index=jump_index,
                    bridge_radius=int(self.config.bridge_radius),
                ),
            )
        )
        return variants

    def _generate_family_seed(
        self,
        *,
        start_joint: list[float],
        target_pose: list[float],
        alignment_tolerance_deg: float,
        family_config: dict[str, Any],
    ) -> dict[str, Any]:
        source_label = str(family_config.get("source_label") or "alignment_seed_family")
        try:
            start_fk = self._fk_pose_fn(start_joint)
            start_quat = start_fk[3:7]
            target_quat = target_pose[3:7]
            endpoint_check = constraints.check_alignment_endpoints(
                start_quat,
                target_quat,
                alignment_tolerance_deg,
            )
            if not endpoint_check["valid"]:
                return {
                    "source_label": source_label,
                    "success": False,
                    "failure_reason": endpoint_check["failure_reason"],
                    "endpoint_check": endpoint_check,
                }

            goal_anchor = self._resolve_goal_anchor(start_joint, target_pose, family_config)
            pos_start = torch.tensor(start_fk[:3], dtype=torch.float32)
            pos_goal = torch.tensor(target_pose[:3], dtype=torch.float32)
            quat_goal = constraints.normalize_quaternion(target_quat)
            start_twist_deg = constraints.extract_twist_deg_relative_to_goal(start_quat, quat_goal)

            trajectory: list[torch.Tensor] = []
            waypoint_debug: list[dict[str, Any]] = []
            prev_solution = torch.tensor(start_joint, dtype=torch.float32)
            prev_step_delta = None
            max_step_jump_l2 = 0.0
            max_step_jump_index = None
            ik_fail_count = 0

            for waypoint_index in range(int(self.config.num_waypoints)):
                t = waypoint_index / max(int(self.config.num_waypoints) - 1, 1)
                if waypoint_index == 0:
                    trajectory.append(prev_solution.clone())
                    waypoint_debug.append({"index": 0, "source": "start_joint", "ik_success": True})
                    continue

                pos_i = (pos_start * (1.0 - t) + pos_goal * t).tolist()
                twist_t = _twist_progress(t, str(family_config.get("twist_schedule_mode") or "uniform_shortest"))
                twist_i = constraints.interpolate_angle_shortest_deg(start_twist_deg, 0.0, twist_t)
                quat_i = constraints.compose_goal_relative_twist_quaternion(quat_goal, twist_i)
                solutions = self._ik_solve_fn(
                    [float(v) for v in pos_i],
                    [float(v) for v in quat_i],
                    prev_solution,
                    int(self.config.ik_return_seeds),
                )
                if not solutions:
                    ik_fail_count += 1
                    waypoint_debug.append(
                        {
                            "index": waypoint_index,
                            "source": "ik_failed_keep_previous",
                            "ik_success": False,
                        }
                    )
                    solution = prev_solution.clone()
                else:
                    selection = _select_branch_consistent_solution(
                        prev_solution=prev_solution,
                        feasible_solutions=solutions,
                        prev_step_delta=prev_step_delta,
                        goal_anchor=goal_anchor,
                        waypoint_t=t,
                        selection_mode=str(family_config.get("selection_mode") or "default_score"),
                        joint_abs_limit=float(self.config.joint_abs_limit),
                    )
                    solution = selection["solution"]
                    waypoint_debug.append(
                        {
                            "index": waypoint_index,
                            "source": "ik_solution",
                            "ik_success": True,
                            "selected_rank": selection.get("selected_rank"),
                            "score": selection.get("score"),
                        }
                    )

                step_delta = solution - prev_solution
                step_l2 = float(torch.linalg.norm(step_delta).item())
                if step_l2 > max_step_jump_l2:
                    max_step_jump_l2 = step_l2
                    max_step_jump_index = waypoint_index
                prev_step_delta = step_delta
                prev_solution = solution
                trajectory.append(solution.clone())

            points = [
                [round(float(v), 8) for v in point.detach().cpu().tolist()]
                for point in trajectory
            ]
            return {
                "source_label": source_label,
                "success": True,
                "trajectory_points": points,
                "ik_fail_count": int(ik_fail_count),
                "max_step_jump_l2": round(max_step_jump_l2, 8),
                "max_step_jump_index": max_step_jump_index,
                "goal_anchor_rank_used": goal_anchor.get("rank_used") if isinstance(goal_anchor, dict) else None,
                "waypoint_debug_sample": waypoint_debug[:3] + waypoint_debug[-3:],
            }
        except Exception as exc:
            return {
                "source_label": source_label,
                "success": False,
                "failure_reason": f"{type(exc).__name__}: {exc}",
            }

    def _resolve_goal_anchor(
        self,
        start_joint: list[float],
        target_pose: list[float],
        family_config: dict[str, Any],
    ) -> dict[str, Any] | None:
        start_tensor = torch.tensor(start_joint, dtype=torch.float32)
        solutions = self._ik_solve_fn(
            [float(v) for v in target_pose[:3]],
            constraints.normalize_quaternion(target_pose[3:7]),
            start_tensor,
            int(self.config.ik_return_seeds),
        )
        if not solutions:
            return None
        wrapped = torch.stack([_wrap_solution_to_prev(start_tensor, item) for item in solutions], dim=0)
        deltas = torch.linalg.norm(wrapped - start_tensor.unsqueeze(0), dim=-1)
        sorted_indices = torch.argsort(deltas)
        rank = max(1, int(family_config.get("goal_anchor_rank") or 1))
        rank_index = min(rank - 1, int(sorted_indices.shape[0]) - 1)
        selected_index = int(sorted_indices[rank_index].item())
        return {
            "solution": wrapped[selected_index].detach().clone(),
            "rank_used": rank_index + 1,
            "delta_l2": float(deltas[selected_index].item()),
        }


def _twist_progress(t: float, mode: str) -> float:
    if mode == "delayed_to_goal":
        return float(t * t) if t <= 0.5 else float(1.0 - 2.0 * (1.0 - t) * (1.0 - t))
    return float(t)


def _select_branch_consistent_solution(
    *,
    prev_solution: torch.Tensor,
    feasible_solutions: list[torch.Tensor],
    prev_step_delta: torch.Tensor | None,
    goal_anchor: dict[str, Any] | None,
    waypoint_t: float,
    selection_mode: str,
    joint_abs_limit: float,
) -> dict[str, Any]:
    wrapped = torch.stack([_wrap_solution_to_prev(prev_solution, item) for item in feasible_solutions], dim=0)
    delta = wrapped - prev_solution.unsqueeze(0)
    delta_l2 = torch.linalg.norm(delta, dim=-1)
    trend_cost = torch.zeros_like(delta_l2)
    if prev_step_delta is not None:
        trend_cost = torch.linalg.norm(delta - prev_step_delta.unsqueeze(0), dim=-1)
    goal_cost = torch.zeros_like(delta_l2)
    if goal_anchor and goal_anchor.get("solution") is not None:
        expected = prev_solution * (1.0 - float(waypoint_t)) + goal_anchor["solution"] * float(waypoint_t)
        goal_cost = torch.linalg.norm(wrapped - expected.unsqueeze(0), dim=-1)
    limit_cost = torch.zeros_like(delta_l2)
    if selection_mode == "in_limit_best":
        margin = torch.clamp(float(joint_abs_limit) - torch.abs(wrapped), min=0.0)
        limit_cost = torch.sum(1.0 / torch.clamp(margin, min=1e-3), dim=-1) * 0.01
    score = delta_l2 + 0.25 * trend_cost + 0.15 * goal_cost + limit_cost
    selected = int(torch.argmin(score).item())
    return {
        "solution": wrapped[selected].detach().clone(),
        "selected_rank": selected,
        "score": round(float(score[selected].item()), 8),
    }


def _wrap_solution_to_prev(prev_solution: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
    wrapped = solution.detach().clone().to(dtype=torch.float32, device=prev_solution.device)
    for joint_index in (0, 5):
        if joint_index >= int(wrapped.shape[0]):
            continue
        prev = float(prev_solution[joint_index].item())
        raw = float(wrapped[joint_index].item())
        wrapped[joint_index] = prev + math.atan2(math.sin(raw - prev), math.cos(raw - prev))
    return wrapped


def _precheck_rule_seed(
    trajectory: list[list[float]],
    *,
    start_joint: list[float],
    dof: int,
    joint_abs_limit: float,
) -> dict[str, Any]:
    shape_valid = (
        isinstance(trajectory, list)
        and len(trajectory) > 0
        and all(isinstance(point, list) and len(point) == int(dof) for point in trajectory)
    )
    if not shape_valid:
        return {
            "valid": False,
            "failure_reason": "invalid_shape_or_dof",
            "shape_valid": False,
        }
    points = torch.tensor(trajectory, dtype=torch.float32)
    start = torch.tensor(start_joint, dtype=torch.float32)
    step_l2 = torch.linalg.norm(points[1:] - points[:-1], dim=-1) if int(points.shape[0]) > 1 else torch.zeros(1)
    max_abs = float(torch.max(torch.abs(points)).item())
    start_gap = float(torch.linalg.norm(points[0] - start).item())
    valid = max_abs <= float(joint_abs_limit) + 1e-6 and start_gap <= 1e-5
    failure_reason = None
    if max_abs > float(joint_abs_limit) + 1e-6:
        failure_reason = "joint_abs_limit_exceeded"
    elif start_gap > 1e-5:
        failure_reason = "start_gap_exceeds_threshold"
    return {
        "valid": bool(valid),
        "failure_reason": failure_reason,
        "shape_valid": True,
        "start_gap_l2": round(start_gap, 8),
        "max_step_l2": round(float(torch.max(step_l2).item()), 8) if step_l2.numel() else 0.0,
        "max_abs_position": round(max_abs, 8),
        "joint_abs_limit": float(joint_abs_limit),
    }


def _smooth_points(points: list[list[float]], *, passes: int) -> list[list[float]]:
    tensor = torch.tensor(points, dtype=torch.float32)
    if int(tensor.shape[0]) < 3:
        return points
    smoothed = tensor.clone()
    for _ in range(max(1, int(passes))):
        smoothed[1:-1] = 0.25 * smoothed[:-2] + 0.5 * smoothed[1:-1] + 0.25 * smoothed[2:]
        smoothed[0] = tensor[0]
        smoothed[-1] = tensor[-1]
    return [[round(float(v), 8) for v in row.tolist()] for row in smoothed]


def _bridge_points(
    points: list[list[float]],
    *,
    jump_index: int | None,
    bridge_radius: int,
) -> list[list[float]]:
    tensor = torch.tensor(points, dtype=torch.float32)
    if jump_index is None or int(tensor.shape[0]) < 4:
        return points
    bridged = tensor.clone()
    start_anchor = max(0, int(jump_index) - max(int(bridge_radius), 1) - 1)
    end_anchor = min(int(tensor.shape[0]) - 1, int(jump_index) + max(int(bridge_radius), 1))
    if end_anchor - start_anchor < 2:
        return points
    start = bridged[start_anchor].clone()
    end = bridged[end_anchor].clone()
    span = end_anchor - start_anchor
    for offset in range(1, span):
        alpha = float(offset) / float(span)
        bridged[start_anchor + offset] = (1.0 - alpha) * start + alpha * end
    bridged[0] = tensor[0]
    bridged[-1] = tensor[-1]
    return [[round(float(v), 8) for v in row.tolist()] for row in bridged]


def _max_step_jump_index(points: list[list[float]]) -> int | None:
    if len(points) < 2:
        return None
    tensor = torch.tensor(points, dtype=torch.float32)
    step_l2 = torch.linalg.norm(tensor[1:] - tensor[:-1], dim=-1)
    return int(torch.argmax(step_l2).item()) + 1


def _max_step_l2(points: list[list[float]]) -> float:
    if len(points) < 2:
        return 0.0
    tensor = torch.tensor(points, dtype=torch.float32)
    step_l2 = torch.linalg.norm(tensor[1:] - tensor[:-1], dim=-1)
    return round(float(torch.max(step_l2).item()), 8)
