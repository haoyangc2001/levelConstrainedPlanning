"""Pure Python SR5 level constrained planner core."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

from . import constraints, validators
from .result_schema import (
    CandidateRecord,
    PlannerArtifacts,
    PlannerResult,
    PlannerRunRecord,
    STATUS_FAILED_ALIGNMENT,
    STATUS_FAILED_INTERNAL,
    STATUS_FAILED_PLANNER,
    STATUS_FAILED_PRECHECK,
    STATUS_SUCCESS,
)
from .learned_seed import CheckpointDiffusionSeedProvider, CheckpointDiffusionSeedProviderConfig
from .repair import SeedRepairAdapter
from .robot_assets import resolve_robot_config
from .rule_seed import RuleLevelSeedProvider, RuleSeedProviderConfig
from .seed_provider import (
    NullDiffusionSeedProvider,
)
from .world import build_world


LOGGER = logging.getLogger(__name__)


@dataclass
class LevelPlannerConfig:
    robot_profile: str = "sr5"
    robot_config: str = "configs/robot/xms5_r800_w4g3b4c_v2.yml"
    obstacle_json: str = "configs/obstacles/abs.autosave.json"
    obstacle_rel_json: str | None = "configs/obstacles/rel.autosave.json"
    device: str = "cuda:0"
    collision_cache_obb: int = 16
    use_cuda_graph: bool = True
    warmup_iterations: int = 3
    max_attempts: int = 5
    num_candidates: int = 1
    strict_level: bool = True
    level_tolerance_deg: float = 3.0
    local_axis: list[float] = field(default_factory=lambda: [0.0, 1.0, 0.0])
    target_world_axis: list[float] = field(default_factory=lambda: [0.0, 0.0, -1.0])
    speed_scale: float = 0.5
    goal_position_tolerance_m: float = 0.02
    goal_orientation_tolerance_rad: float = 0.20
    max_start_gap_l2: float = 0.25
    max_joint_step_l2: float = 2.0
    max_joint_step_abs: float = 1.5
    max_acceleration_proxy_l2: float = 3.0
    collision_safety_margin_m: float = 0.005
    rule_seed_num_waypoints: int = 30
    rule_seed_ik_return_seeds: int = 32
    rule_seed_include_smooth_bridge_variants: bool = True
    rule_seed_smoothing_passes: int = 2
    rule_seed_bridge_radius: int = 2
    diffusion_generated_samples_path: str = (
        "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/"
        "sr5_phase10_mature_diffusion_20260715_generated_samples.json"
    )
    diffusion_checkpoint_path: str = (
        "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/"
        "sr5_phase10_mature_diffusion_20260715/best.pt"
    )
    critic_checkpoint_path: str = (
        "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/"
        "sr5_phase10_success_critic_20260715/best.pt"
    )
    learned_seed_use_critic: bool = True
    artifact_pointer: str | None = None
    load_model_paths_from_artifacts: bool = False

    @classmethod
    def from_file(cls, path: str | Path) -> "LevelPlannerConfig":
        config_path = Path(path)
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config must be a mapping: {config_path}")
        base_dir = config_path.parent
        planner_cfg = payload.get("planner", payload)

        def _resolve_path(value: str | None) -> str | None:
            if not value:
                return value
            candidate = Path(str(value))
            if candidate.is_absolute():
                return str(candidate)
            return str((base_dir / candidate).resolve())

        cfg = cls()
        for key in (
            "robot_profile",
            "device",
            "collision_cache_obb",
            "use_cuda_graph",
            "warmup_iterations",
            "max_attempts",
            "num_candidates",
            "strict_level",
            "level_tolerance_deg",
            "local_axis",
            "target_world_axis",
            "speed_scale",
            "goal_position_tolerance_m",
            "goal_orientation_tolerance_rad",
            "max_start_gap_l2",
            "max_joint_step_l2",
            "max_joint_step_abs",
            "max_acceleration_proxy_l2",
            "rule_seed_num_waypoints",
            "rule_seed_ik_return_seeds",
            "rule_seed_include_smooth_bridge_variants",
            "rule_seed_smoothing_passes",
            "rule_seed_bridge_radius",
            "diffusion_generated_samples_path",
            "diffusion_checkpoint_path",
            "critic_checkpoint_path",
            "learned_seed_use_critic",
            "artifact_pointer",
            "load_model_paths_from_artifacts",
        ):
            if key in planner_cfg:
                setattr(cfg, key, planner_cfg[key])
        cfg.robot_config = _resolve_path(planner_cfg.get("robot_config", cfg.robot_config)) or cfg.robot_config
        cfg.obstacle_json = _resolve_path(planner_cfg.get("obstacle_json", cfg.obstacle_json)) or cfg.obstacle_json
        cfg.obstacle_rel_json = _resolve_path(
            planner_cfg.get("obstacle_rel_json", cfg.obstacle_rel_json)
        )
        cfg.artifact_pointer = _resolve_path(planner_cfg.get("artifact_pointer", cfg.artifact_pointer))
        if bool(cfg.load_model_paths_from_artifacts) and cfg.artifact_pointer:
            cfg._apply_artifact_pointer(Path(cfg.artifact_pointer))
        return cfg

    def _apply_artifact_pointer(self, pointer_path: Path) -> None:
        if not pointer_path.exists():
            LOGGER.warning("artifact pointer not found, keeping configured model paths: %s", pointer_path)
            return
        try:
            artifacts = json.loads(pointer_path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("failed to read artifact pointer %s: %s", pointer_path, exc)
            return
        diffusion = artifacts.get("diffusion") or {}
        critic = artifacts.get("critic") or {}
        diffusion_checkpoint = diffusion.get("best_checkpoint") or diffusion.get("best_checkpoint_file", {}).get("path")
        critic_checkpoint = critic.get("best_checkpoint") or critic.get("best_checkpoint_file", {}).get("path")
        generated_samples = artifacts.get("generated_samples", {}).get("path")
        if diffusion_checkpoint:
            self.diffusion_checkpoint_path = str(diffusion_checkpoint)
        if critic_checkpoint:
            self.critic_checkpoint_path = str(critic_checkpoint)
        if generated_samples:
            self.diffusion_generated_samples_path = str(generated_samples)


class LevelConstrainedPlanner:
    """CuRobo-backed plan-only SR5 level constrained planner."""

    def __init__(self, config: LevelPlannerConfig):
        self.config = config
        self.device = str(config.device)
        self._planner = None
        self._joint_names: list[str] = []
        self._tool_frames: list[str] = []
        self._robot_cfg: dict[str, Any] | None = None
        self._world_summary: dict[str, Any] = {}
        self._joint_limits: list[dict[str, Any]] = []
        self._collision_checker: Any | None = None
        self._init_curobo()

    @classmethod
    def from_config(cls, config_path: str | Path) -> "LevelConstrainedPlanner":
        return cls(LevelPlannerConfig.from_file(config_path))

    def _init_curobo(self) -> None:
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        from curobo.scene import Scene as SceneCfg

        robot_cfg = resolve_robot_config(
            robot_config_path=Path(self.config.robot_config),
            auto_generate_spheres=False,
        )
        self._robot_cfg = robot_cfg
        cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=None,
            collision_cache={"obb": int(self.config.collision_cache_obb)},
            use_cuda_graph=bool(self.config.use_cuda_graph),
        )
        self._planner = MotionPlanner(cfg)
        self._planner.warmup(
            enable_graph=bool(self.config.use_cuda_graph),
            num_warmup_iterations=int(self.config.warmup_iterations),
        )
        self._joint_names = list(self._planner.joint_names)
        self._tool_frames = list(self._planner.tool_frames)
        self._joint_limits = validators.load_joint_limits_from_robot_config(
            Path(self.config.robot_config),
            self._joint_names,
        )

        world_result = build_world(
            abs_json_path=Path(self.config.obstacle_json) if self.config.obstacle_json else None,
            rel_json_path=Path(self.config.obstacle_rel_json) if self.config.obstacle_rel_json else None,
        )
        self._world_summary = dict(world_result.get("world_summary") or {})
        self._planner.update_world(SceneCfg.create(world_result["world_dict"]))

        # A1.2: build a world collision checker sharing the planner's already-loaded
        # scene. Activation distance == safety margin so a returned world-cost of 0
        # means every robot sphere is at least `margin` from all obstacles, and any
        # positive cost (meters) means within-margin / penetrating.
        self._collision_checker = self._build_collision_checker()

        LOGGER.info(
            "CuRobo initialized: joints=%s tool_frames=%s world=%s",
            self._joint_names,
            self._tool_frames,
            self._world_summary,
        )

    def _build_collision_checker(self) -> Any | None:
        """A1.2: construct a RobotCollisionChecker sharing the planner's scene.

        Returns ``None`` if the world is obstacle-free (nothing to check) or if the
        checker cannot be built; callers degrade the collision check to
        ``no_obstacles`` / ``unchecked`` accordingly.
        """
        if int(self._world_summary.get("total_count", 0) or 0) == 0:
            return None
        try:
            from curobo.collision_checking import (
                RobotCollisionChecker,
                RobotCollisionCheckerCfg,
            )

            cfg = RobotCollisionCheckerCfg.load_from_config(
                robot_config=self._robot_cfg,
                scene_collision_checker=self._planner.scene_collision_checker,
                collision_activation_distance=float(self.config.collision_safety_margin_m),
            )
            return RobotCollisionChecker(cfg)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("collision checker unavailable, collision check degraded: %s", exc)
            return None

    def _evaluate_collision(self, trajectory_points: list[list[float]]) -> dict[str, Any] | None:
        """A1.2: measure the along-path world collision cost for a trajectory.

        Returns a dict consumed by ``validators._evaluate_collision`` (kept curobo
        free per A1.3), or ``None`` when no trajectory / no checker is available so
        the validator degrades to ``unchecked``.
        """
        if not trajectory_points:
            return None
        if self._collision_checker is None:
            # obstacle-free world (or checker failed to build); validator handles
            # the no_obstacles degenerate case from world_summary.
            return None
        try:
            import torch

            q = torch.as_tensor(
                trajectory_points, device=self.device, dtype=torch.float32
            )
            if q.ndim == 2:
                q = q.unsqueeze(0)  # [1, horizon, dof]
            world_dist, _self_dist = (
                self._collision_checker.get_scene_self_collision_distance_from_joint_trajectory(q)
            )
            # world_dist is the cuRobo world collision cost per sphere (meters):
            # 0 == outside activation band (safe), positive == within margin.
            cost = float(world_dist.max().item())
            margin = float(self.config.collision_safety_margin_m)
            # Recover a signed-distance proxy: cost>0 penetrates the margin band by
            # `cost`, so min signed distance ~= margin - cost. When cost==0 the true
            # distance is unknown-but->=margin, reported as the margin itself.
            min_distance_m = margin - cost if cost > 0.0 else margin
            num_points = int(q.shape[1])
            num_spheres = int(world_dist.shape[-1]) if world_dist.ndim >= 1 else None
            return {
                "collision_cost": cost,
                "min_distance_m": min_distance_m,
                "num_points": num_points,
                "num_spheres": num_spheres,
            }
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("collision evaluation failed, treating as unchecked: %s", exc)
            return None

    def plan(self, request: dict[str, Any], out_dir: str | Path | None = None) -> dict[str, Any]:
        t0 = time.time()
        request_id = str(request.get("request_id") or f"request_{int(t0)}")
        normalized: dict[str, Any] | None = None
        try:
            normalized = self._normalize_request(request)
            seed_reports = self._run_seed_providers(normalized)
            result = self._plan_with_control_flow(
                request_id=request_id,
                request=normalized,
                seed_reports=seed_reports,
                started_at=t0,
            )
        except ValueError as exc:
            result = PlannerResult(
                request_id=request_id,
                status=STATUS_FAILED_PRECHECK,
                failure_reason=str(exc),
                metrics=self._base_metrics(t0),
            )
        except Exception as exc:
            result = PlannerResult(
                request_id=request_id,
                status=STATUS_FAILED_INTERNAL,
                failure_reason=f"{type(exc).__name__}: {exc}",
                metrics=self._base_metrics(t0),
            )

        self._finalize_closed_loop_records(
            result=result,
            original_request=request if isinstance(request, dict) else {},
            normalized_request=normalized,
            started_at=t0,
        )
        if out_dir is not None:
            self._write_artifacts(result, out_dir)
        return result.to_dict()

    def _plan_with_control_flow(
        self,
        *,
        request_id: str,
        request: dict[str, Any],
        seed_reports: list[dict[str, Any]],
        started_at: float,
    ) -> PlannerResult:
        external_candidates, external_summaries = self._collect_external_seed_candidates(
            request,
            seed_reports,
        )
        learned_candidates, learned_summaries = self._filter_candidate_group(
            external_candidates,
            external_summaries,
            {"diffusion_seed"},
        )
        rule_candidates, rule_summaries = self._filter_candidate_group(
            external_candidates,
            external_summaries,
            {"rule_seed"},
        )
        all_summaries: list[dict[str, Any]] = list(external_summaries)
        trace: list[dict[str, Any]] = []
        mode = str(request.get("seed_policy", {}).get("mode") or "rule")
        fallback_to_rule = bool(request.get("seed_policy", {}).get("fallback_to_rule_seed", True))
        fallback_to_planner_native = bool(
            request.get("seed_policy", {}).get("fallback_to_planner_native", True)
        )
        total_budget_ms = float(request.get("metadata", {}).get("total_budget_ms") or 0.0)
        learned_modes = {"diffusion", "mixed", "candidate"}
        last_result: PlannerResult | None = None

        if mode in learned_modes:
            learned_result = self._select_branch_or_fail(
                branch_name="learned",
                request_id=request_id,
                request=request,
                candidates=learned_candidates,
                candidate_summaries=learned_summaries,
                seed_reports=seed_reports,
                started_at=started_at,
            )
            trace.append(self._branch_trace_record("learned", learned_result, started_at, total_budget_ms))
            last_result = learned_result
            if learned_result.status == STATUS_SUCCESS:
                learned_result.candidates = all_summaries
                learned_result.metrics["success_source"] = "learned"
                learned_result.metrics["control_flow_trace"] = trace
                return learned_result

            if fallback_to_rule:
                rule_result = self._select_branch_or_fail(
                    branch_name="rule_fallback",
                    request_id=request_id,
                    request=request,
                    candidates=rule_candidates,
                    candidate_summaries=rule_summaries,
                    seed_reports=seed_reports,
                    started_at=started_at,
                )
                trace.append(self._branch_trace_record("rule_fallback", rule_result, started_at, total_budget_ms))
                last_result = rule_result
                if rule_result.status == STATUS_SUCCESS:
                    rule_result.candidates = all_summaries
                    rule_result.metrics["success_source"] = "rule_fallback"
                    rule_result.metrics["control_flow_trace"] = trace
                    return rule_result

        elif mode == "rule" and rule_candidates:
            rule_result = self._select_branch_or_fail(
                branch_name="rule",
                request_id=request_id,
                request=request,
                candidates=rule_candidates,
                candidate_summaries=rule_summaries,
                seed_reports=seed_reports,
                started_at=started_at,
            )
            trace.append(self._branch_trace_record("rule", rule_result, started_at, total_budget_ms))
            last_result = rule_result
            if rule_result.status == STATUS_SUCCESS:
                rule_result.candidates = all_summaries
                rule_result.metrics["success_source"] = "rule"
                rule_result.metrics["control_flow_trace"] = trace
                return rule_result

        if not fallback_to_planner_native:
            return self._finalize_without_native_fallback(
                request_id=request_id,
                seed_reports=seed_reports,
                all_summaries=all_summaries,
                trace=trace,
                last_result=last_result,
                started_at=started_at,
                mode=mode,
            )

        native_candidates, native_summaries = self._collect_planner_candidates(request)
        all_summaries.extend(native_summaries)
        native_result = self._select_branch_or_fail(
            branch_name="planner_native",
            request_id=request_id,
            request=request,
            candidates=native_candidates,
            candidate_summaries=native_summaries,
            seed_reports=seed_reports,
            started_at=started_at,
        )
        trace.append(self._branch_trace_record("planner_native", native_result, started_at, total_budget_ms))
        native_result.candidates = all_summaries
        native_result.metrics["success_source"] = "planner_native" if native_result.status == STATUS_SUCCESS else None
        native_result.metrics["control_flow_trace"] = trace
        if native_result.status != STATUS_SUCCESS and mode in learned_modes:
            native_result.failure_reason = native_result.failure_reason or "failed_all_fallbacks"
            native_result.status = "failed_all_fallbacks"
        return native_result

    def _finalize_without_native_fallback(
        self,
        *,
        request_id: str,
        seed_reports: list[dict[str, Any]],
        all_summaries: list[dict[str, Any]],
        trace: list[dict[str, Any]],
        last_result: PlannerResult | None,
        started_at: float,
        mode: str,
    ) -> PlannerResult:
        if last_result is None:
            last_result = PlannerResult(
                request_id=request_id,
                status=STATUS_FAILED_PLANNER,
                failure_reason="planner_native_fallback_disabled_and_no_branch_candidates",
                metrics=self._base_metrics(started_at),
                seed_provider_reports=seed_reports,
                candidates=all_summaries,
            )
        last_result.candidates = all_summaries
        last_result.metrics["success_source"] = None
        last_result.metrics["control_flow_trace"] = trace
        last_result.metrics["planner_native_fallback_disabled"] = True
        if last_result.status != STATUS_SUCCESS and mode in {"diffusion", "mixed", "candidate"}:
            last_result.status = "failed_all_fallbacks"
            last_result.failure_reason = (
                last_result.failure_reason
                or "learned_and_rule_branches_failed_with_planner_native_disabled"
            )
        return last_result

    def _select_branch_or_fail(
        self,
        *,
        branch_name: str,
        request_id: str,
        request: dict[str, Any],
        candidates: list[torch.Tensor],
        candidate_summaries: list[dict[str, Any]],
        seed_reports: list[dict[str, Any]],
        started_at: float,
    ) -> PlannerResult:
        if not candidates:
            return PlannerResult(
                request_id=request_id,
                status=STATUS_FAILED_PLANNER,
                failure_reason=f"{branch_name}_returned_no_successful_candidate",
                metrics=self._base_metrics(started_at),
                seed_provider_reports=seed_reports,
                candidates=candidate_summaries,
            )
        result = self._select_candidate(
            request_id=request_id,
            request=request,
            candidates=candidates,
            candidate_summaries=candidate_summaries,
            seed_reports=seed_reports,
            started_at=started_at,
        )
        result.metrics["branch_name"] = branch_name
        return result

    @staticmethod
    def _filter_candidate_group(
        candidates: list[torch.Tensor],
        summaries: list[dict[str, Any]],
        source_types: set[str],
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        selected_candidates: list[torch.Tensor] = []
        selected_summaries: list[dict[str, Any]] = []
        success_index = 0
        for summary in summaries:
            trajectory = None
            if summary.get("status") == "success" and success_index < len(candidates):
                trajectory = candidates[success_index]
                success_index += 1
            if summary.get("source_type") in source_types:
                selected_summaries.append(summary)
                if trajectory is not None:
                    selected_candidates.append(trajectory)
        return selected_candidates, selected_summaries

    @staticmethod
    def _branch_trace_record(
        branch_name: str,
        result: PlannerResult,
        started_at: float,
        total_budget_ms: float,
    ) -> dict[str, Any]:
        elapsed_ms = round((time.time() - started_at) * 1000.0, 3)
        return {
            "stage": branch_name,
            "status": result.status,
            "failure_reason": result.failure_reason,
            "candidate_count": len(result.candidates),
            "selected_candidate_id": result.metrics.get("selected_candidate_id"),
            "elapsed_ms": elapsed_ms,
            "total_budget_ms": total_budget_ms or None,
            "timeout": bool(total_budget_ms and elapsed_ms > total_budget_ms),
        }

    def _normalize_request(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("request must be a mapping")
        robot_profile = str(request.get("robot_profile", self.config.robot_profile))
        if robot_profile != "sr5":
            raise ValueError(f"unsupported robot_profile={robot_profile!r}; first version only supports sr5")

        start_joint = [float(v) for v in request.get("start_joint", [])]
        if len(start_joint) != len(self._joint_names):
            raise ValueError(
                f"start_joint length mismatch: got {len(start_joint)}, expected {len(self._joint_names)}"
            )

        pose = request.get("target_pose") or {}
        if isinstance(pose, list):
            if len(pose) != 7:
                raise ValueError("target_pose list must be [x,y,z,qw,qx,qy,qz]")
            target_pose = [float(v) for v in pose]
        else:
            position = [float(v) for v in pose.get("position", [])]
            quat = [float(v) for v in pose.get("quaternion_wxyz", [])]
            if len(position) != 3 or len(quat) != 4:
                raise ValueError("target_pose requires position[3] and quaternion_wxyz[4]")
            target_pose = position + quat

        alignment = dict(request.get("alignment") or {})
        seed_policy = dict(request.get("seed_policy") or {})
        return {
            "schema_version": str(request.get("schema_version", "1.0")),
            "request_id": str(request.get("request_id") or "request"),
            "robot_profile": robot_profile,
            "start_joint": start_joint,
            "target_pose": target_pose,
            "alignment": {
                "local_axis": [float(v) for v in alignment.get("local_axis", self.config.local_axis)],
                "target_world_axis": [
                    float(v) for v in alignment.get("target_world_axis", self.config.target_world_axis)
                ],
                "tolerance_deg": float(alignment.get("tolerance_deg", self.config.level_tolerance_deg)),
                "strict_level": bool(alignment.get("strict_level", self.config.strict_level)),
            },
            "seed_policy": {
                "mode": str(seed_policy.get("mode", "rule")).strip().lower(),
                "k_generate": int(seed_policy.get("k_generate", 0)),
                "k_accept": int(seed_policy.get("k_accept", 0)),
                "fallback_to_rule_seed": bool(seed_policy.get("fallback_to_rule_seed", True)),
                "fallback_to_planner_native": bool(seed_policy.get("fallback_to_planner_native", True)),
                "timeout_sec": float(seed_policy.get("timeout_sec", 0.2)),
                "diffusion_artifact_pointer": seed_policy.get("diffusion_artifact_pointer"),
            },
            "metadata": dict(request.get("metadata") or {}),
        }

    def _run_seed_providers(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        policy = request["seed_policy"]
        mode = policy["mode"]
        reports: list[dict[str, Any]] = []

        if policy.get("fallback_to_rule_seed"):
            rule_provider = RuleLevelSeedProvider(
                RuleSeedProviderConfig(
                    mode="candidate" if int(policy.get("k_generate") or 0) > 0 else "available",
                    k_generate=int(policy.get("k_generate") or 0),
                    k_accept=int(policy.get("k_accept") or policy.get("k_generate") or 0),
                    timeout_sec=float(policy.get("timeout_sec") or 1.0),
                    num_waypoints=int(self.config.rule_seed_num_waypoints),
                    ik_return_seeds=int(self.config.rule_seed_ik_return_seeds),
                    include_smooth_bridge_variants=bool(
                        self.config.rule_seed_include_smooth_bridge_variants
                    ),
                    smoothing_passes=int(self.config.rule_seed_smoothing_passes),
                    bridge_radius=int(self.config.rule_seed_bridge_radius),
                ),
                fk_pose_fn=self._fk_pose_for_joint,
                ik_solve_fn=self._ik_solve_pose_candidates,
            )
            rule_result = rule_provider.generate(
                {
                    "start_joint": request["start_joint"],
                    "target_pose": request["target_pose"],
                    "alignment": request["alignment"],
                    "dof": len(self._joint_names),
                    "joint_names": list(self._joint_names),
                    "tool_frames": list(self._tool_frames),
                }
            )
            rule_report = rule_result.to_lifecycle_dict()
            rule_report["provider"] = rule_report.get("provider_name", "rule_seed")
            rule_report["accepted_count"] = int(
                sum(1 for c in rule_report.get("candidates", []) if c.get("precheck", {}).get("valid"))
            )
            rule_report["runtime_effect"] = "raw_rule_seed_report_only_until_phase4_repair_pool"
        else:
            rule_report = {
                "provider": "rule_seed",
                "provider_name": "rule_seed",
                "mode": "disabled",
                "status": "disabled",
                "generated_count": 0,
                "accepted_count": 0,
                "runtime_effect": "disabled_by_seed_policy",
            }
        reports.append(rule_report)

        diffusion_mode = "off"
        if mode in {"diffusion", "candidate"}:
            diffusion_mode = "candidate"
        elif mode == "mixed":
            diffusion_mode = "candidate"
        elif mode == "shadow":
            diffusion_mode = "shadow"
        provider = (
            CheckpointDiffusionSeedProvider(
                CheckpointDiffusionSeedProviderConfig(
                    mode=diffusion_mode,
                    diffusion_checkpoint_path=self.config.diffusion_checkpoint_path,
                    critic_checkpoint_path=self.config.critic_checkpoint_path,
                    k_generate=int(policy.get("k_generate") or 0),
                    k_accept=int(policy.get("k_accept") or 0),
                    timeout_sec=float(policy.get("timeout_sec") or 2.0),
                    device=self.device,
                    use_critic=bool(self.config.learned_seed_use_critic),
                )
            )
            if diffusion_mode != "off"
            else NullDiffusionSeedProvider("off")
        )
        provider_result = provider.generate(
            {
                "start_joint": request["start_joint"],
                "target_pose": request["target_pose"],
                "dof": len(self._joint_names),
                "joint_names": list(self._joint_names),
                "tool_frames": list(self._tool_frames),
                "alignment": request["alignment"],
                "world_summary": dict(self._world_summary),
            }
        )
        provider_report = provider_result.to_lifecycle_dict()
        provider_report["provider"] = provider_report.get("provider_name", "diffusion_seed")
        provider_report["accepted_count"] = int(
            sum(1 for c in provider_report.get("candidates", []) if c.get("precheck", {}).get("valid"))
        )
        provider_report["runtime_effect"] = (
            "candidate_mode_external_seed_repair_pool"
            if diffusion_mode == "candidate"
            else ("shadow_report_only" if diffusion_mode == "shadow" else "disabled")
        )
        reports.append(provider_report)
        return reports

    def _make_goal_and_current_state(self, request: dict[str, Any]):
        from curobo.types import GoalToolPose, JointState as CuJointState

        target_pose = request["target_pose"]
        pos = target_pose[:3]
        quat = constraints.normalize_quaternion(target_pose[3:7])
        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=torch.tensor([[[[[pos[0], pos[1], pos[2]]]]]], device=self.device, dtype=torch.float32),
            quaternion=torch.tensor([[[[[quat[0], quat[1], quat[2], quat[3]]]]]], device=self.device, dtype=torch.float32),
        )
        current_state = CuJointState.from_position(
            torch.tensor([request["start_joint"]], device=self.device, dtype=torch.float32),
            joint_names=self._joint_names,
        )
        return goal, current_state

    def _collect_external_seed_candidates(
        self,
        request: dict[str, Any],
        seed_reports: list[dict[str, Any]],
    ) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        goal, current_state = self._make_goal_and_current_state(request)
        adapter = SeedRepairAdapter(
            motion_planner=self._planner,
            device=self.device,
            joint_names=self._joint_names,
        )
        candidates: list[torch.Tensor] = []
        summaries: list[dict[str, Any]] = []
        for report in seed_reports:
            provider_name = str(report.get("provider") or report.get("provider_name") or "")
            provider_mode = str(report.get("mode") or "")
            if provider_name not in {"rule_seed", "diffusion_seed"}:
                continue
            if provider_mode == "shadow":
                continue
            for raw_candidate in report.get("candidates", []) or []:
                precheck = raw_candidate.get("precheck") or {}
                raw_id = str(raw_candidate.get("candidate_id") or "external_seed")
                source_type = self._repaired_source_type(raw_candidate.get("source_type"), provider_name)
                summary: dict[str, Any] = {
                    "candidate_id": f"{raw_id}_repaired",
                    "source_type": source_type,
                    "source_label": str(raw_candidate.get("source_label") or raw_id),
                    "provider": provider_name,
                    "provider_mode": provider_mode,
                    "status": "pending",
                    "entered_pool": False,
                    "parent_candidate_id": raw_id,
                    "metadata": {
                        "parent_candidate_id": raw_id,
                        "raw_source_type": raw_candidate.get("source_type"),
                        "raw_source_label": raw_candidate.get("source_label"),
                        "provider": provider_name,
                        "provider_mode": provider_mode,
                        "raw_metadata": raw_candidate.get("metadata") or {},
                    },
                    "precheck": dict(precheck),
                    "metrics": {},
                }
                seed_points = (raw_candidate.get("trajectory") or {}).get("points") or []
                if not precheck.get("valid"):
                    summary["status"] = "failed_precheck"
                    summary["failure_reason"] = precheck.get("failure_reason") or "external_seed_precheck_failed"
                    summaries.append(summary)
                    continue
                if not seed_points:
                    summary["status"] = "failed_precheck"
                    summary["failure_reason"] = "external_seed_missing_trajectory_points"
                    summaries.append(summary)
                    continue
                summary["entered_pool"] = True
                repair = adapter.repair_pose_seed(
                    seed_points=seed_points,
                    goal_tool_pose=goal,
                    current_state=current_state,
                    return_seeds=1,
                )
                summary["optimizer_result"] = dict(repair.optimizer_result)
                summary["solve_time_sec"] = repair.optimizer_result.get("solve_time_sec")
                summary["result_status"] = repair.status
                if repair.success and repair.trajectory is not None:
                    candidates.append(repair.trajectory)
                    summary["status"] = "success"
                    summary["trajectory_shape"] = list(repair.trajectory.shape)
                else:
                    summary["status"] = "failed_planner"
                    summary["failure_reason"] = repair.failure_reason or repair.status
                summaries.append(summary)
        return candidates, summaries

    def _fk_pose_for_joint(self, joint_position: list[float]) -> list[float]:
        from curobo.types import JointState as CuJointState

        state = CuJointState.from_position(
            torch.tensor([joint_position], device=self.device, dtype=torch.float32),
            joint_names=self._joint_names,
        )
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        pos = tool_pose.position.reshape(-1, 3)[0].detach().cpu().tolist()
        quat = tool_pose.quaternion.reshape(-1, 4)[0].detach().cpu().tolist()
        return [float(v) for v in pos + quat]

    def _ik_solve_pose_candidates(
        self,
        position: list[float],
        quaternion: list[float],
        prev_solution: torch.Tensor,
        return_seeds: int,
    ) -> list[torch.Tensor]:
        from curobo.types import GoalToolPose, JointState as CuJointState

        quat = constraints.normalize_quaternion(quaternion)
        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=torch.tensor(
                [[[[[position[0], position[1], position[2]]]]]],
                device=self.device,
                dtype=torch.float32,
            ),
            quaternion=torch.tensor(
                [[[[[quat[0], quat[1], quat[2], quat[3]]]]]],
                device=self.device,
                dtype=torch.float32,
            ),
        )
        prev = prev_solution.to(device=self.device, dtype=torch.float32).reshape(1, -1)
        current_state = CuJointState.from_position(prev, joint_names=self._joint_names)
        seed_config = prev.reshape(1, 1, -1)
        if hasattr(self._planner.ik_solver, "reset_seed"):
            self._planner.ik_solver.reset_seed()
        result = self._planner.ik_solver.solve_pose(
            goal,
            current_state=current_state,
            seed_config=seed_config,
            return_seeds=int(return_seeds),
        )
        feasible = getattr(result, "feasible", None)
        if feasible is None:
            feasible = getattr(result, "success", None)
        if feasible is None or not bool(feasible.any().item()):
            return []
        batch_idx, seed_idx = feasible.nonzero(as_tuple=True)
        solutions = result.solution[batch_idx, seed_idx]
        return [solutions[index].detach().cpu().to(dtype=torch.float32) for index in range(int(solutions.shape[0]))]

    def _collect_planner_candidates(self, request: dict[str, Any]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
        goal, current_state = self._make_goal_and_current_state(request)

        count = max(1, int(self.config.num_candidates))
        count = max(count, int(request.get("metadata", {}).get("num_candidates", 0) or 0))
        candidates: list[torch.Tensor] = []
        summaries: list[dict[str, Any]] = []
        for idx in range(count):
            started = time.time()
            summary: dict[str, Any] = {
                "candidate_id": f"planner_{idx:02d}",
                "source_type": "planner",
                "source_label": "planner",
                "status": "pending",
                "metrics": {},
            }
            try:
                result = self._planner.plan_pose(
                    goal,
                    current_state,
                    max_attempts=int(self.config.max_attempts),
                    enable_graph_attempt=1 if bool(self.config.use_cuda_graph) else 0,
                )
                summary["solve_time_sec"] = round(time.time() - started, 6)
                summary["result_status"] = self._result_status(result)
                if self._result_success(result):
                    trajectory = self._flatten_position_tensor(result.get_interpolated_plan().position)
                    candidates.append(trajectory)
                    summary["status"] = "success"
                    summary["trajectory_shape"] = list(trajectory.shape)
                    summary["trajectory_path"] = ""
                else:
                    summary["status"] = "failed_planner"
                    summary["failure_reason"] = self._result_status(result)
            except Exception as exc:
                summary["status"] = "failed_internal_error"
                summary["failure_reason"] = f"{type(exc).__name__}: {exc}"
            summaries.append(summary)
        return candidates, summaries

    @staticmethod
    def _repaired_source_type(source_type: Any, provider_name: str) -> str:
        raw = str(source_type or "")
        if provider_name == "rule_seed" or raw.startswith("rule"):
            return "rule_seed"
        if provider_name == "diffusion_seed" or raw.startswith("diffusion"):
            return "diffusion_seed"
        return raw or "external_seed"

    def _select_candidate(
        self,
        *,
        request_id: str,
        request: dict[str, Any],
        candidates: list[torch.Tensor],
        candidate_summaries: list[dict[str, Any]],
        seed_reports: list[dict[str, Any]],
        started_at: float,
    ) -> PlannerResult:
        target_horizon = max(int(traj.shape[0]) for traj in candidates)
        positions = torch.stack(
            [
                self._resample_trajectory_linear(
                    traj.to(device=self.device, dtype=torch.float32),
                    target_horizon,
                )
                for traj in candidates
            ],
            dim=0,
        )
        alignment = request["alignment"]
        local_axis = torch.tensor(alignment["local_axis"], device=self.device, dtype=torch.float32)
        target_axis = torch.tensor(alignment["target_world_axis"], device=self.device, dtype=torch.float32)
        level_eval = constraints.evaluate_axis_alignment_batched(
            positions,
            self._constraint_eval_kinematics_fn,
            alignment_tolerance_deg=float(alignment["tolerance_deg"]),
            local_axis=local_axis,
            target_world_axis=target_axis,
        )
        continuity = constraints.compute_candidate_continuity_metrics(
            positions,
            request["start_joint"],
            request["target_pose"][3:7],
            self._constraint_eval_kinematics_fn,
        )
        selection = constraints.select_level_first_candidate(
            positions,
            level_eval,
            continuity,
            level_tolerance_deg=float(alignment["tolerance_deg"]),
            strict_level=bool(alignment["strict_level"]),
        )
        selected_index = int(selection.get("selected_index", 0))
        selected = positions[selected_index].detach().cpu()
        trajectory = self._trajectory_tensor_to_list(selected)
        successful_summaries = [
            item for item in candidate_summaries if item.get("status") == "success"
        ]
        for idx, item in enumerate(successful_summaries):
            goal_metrics = self._summarize_terminal_goal(positions[idx], request["target_pose"])
            item["metrics"] = {
                "max_alignment_deviation_deg": selection["candidate_max_alignment_deviation"][idx],
                "mean_alignment_deviation_deg": round(float(level_eval["mean_alignment_deviation"][idx].item()), 4),
                "start_joint_gap_l2": selection["candidate_start_joint_gap_l2"][idx],
                "joint_step_jump_cost": selection["candidate_joint_step_jump_cost"][idx],
                "joint_step_max_l2": selection["candidate_joint_step_max_l2"][idx],
                "joint_step_max_abs": selection["candidate_joint_step_max_abs"][idx],
                "twist_smoothness_cost": selection["candidate_twist_smoothness_cost"][idx],
                "position_error_m": goal_metrics["terminal_position_error_m"],
                "orientation_error_rad": goal_metrics["terminal_orientation_error_rad"],
                "orientation_error_deg": round(
                    float(goal_metrics["terminal_orientation_error_rad"]) * 180.0 / 3.141592653589793,
                    6,
                ),
                "selected": idx == selected_index,
            }
        status = STATUS_SUCCESS if selection["planning_status"] == "success" else STATUS_FAILED_ALIGNMENT
        selected_candidate_id = None
        if 0 <= selected_index < len(successful_summaries):
            selected_candidate_id = str(successful_summaries[selected_index].get("candidate_id"))
        candidate_records: list[dict[str, Any]] = []
        success_index = 0
        for item in candidate_summaries:
            trajectory_tensor = None
            if item.get("status") == "success" and success_index < int(positions.shape[0]):
                trajectory_tensor = positions[success_index]
                success_index += 1
            candidate_records.append(
                self._build_candidate_record_from_summary(
                    item,
                    request_id=request_id,
                    run_id=request_id,
                    trajectory=trajectory_tensor,
                    final_status=status,
                    final_failure_reason=selection.get("failure_reason"),
                    alignment_tolerance_deg=float(alignment["tolerance_deg"]),
                    start_joint=request["start_joint"],
                )
            )
        selected_validation = {}
        for record in candidate_records:
            if record.get("candidate_id") == selected_candidate_id:
                selected_validation = dict(record.get("validator_metrics") or {})
                break
        selected_checks = selected_validation.get("checks") or {}
        return PlannerResult(
            request_id=request_id,
            status=status,
            failure_reason=selection.get("failure_reason"),
            selected_trajectory=trajectory if status == STATUS_SUCCESS else None,
            metrics={
                **self._base_metrics(started_at),
                "selected_candidate_id": selected_candidate_id,
                "alignment": {
                    "tolerance_deg": float(alignment["tolerance_deg"]),
                    "selected_max_alignment_deviation_deg": selection.get(
                        "selected_max_alignment_deviation"
                    ),
                    "alignment_valid_count": selection.get("alignment_valid_count"),
                    "candidate_max_alignment_deviation": selection.get(
                        "candidate_max_alignment_deviation"
                    ),
                },
                "goal": self._summarize_terminal_goal(positions[selected_index], request["target_pose"]),
                "continuity": {
                    "selected_start_joint_gap_l2": selection.get("selected_start_joint_gap_l2"),
                    "selected_joint_step_jump_cost": selection.get("selected_joint_step_jump_cost"),
                    "selected_joint_step_max_l2": selection.get("selected_joint_step_max_l2"),
                    "selected_twist_smoothness_cost": selection.get("selected_twist_smoothness_cost"),
                },
                "joint_limit": {
                    **dict(selected_checks.get("joint_limit") or {}),
                },
                "collision_safety": dict(selected_checks.get("collision_safety") or {}),
                "velocity_acceleration": dict(selected_checks.get("velocity_acceleration") or {}),
                "hard_validator": selected_validation,
                "world": dict(self._world_summary),
            },
            seed_provider_reports=seed_reports,
            candidates=candidate_summaries,
            candidate_records=candidate_records,
        )

    def _constraint_eval_kinematics_fn(self, positions: torch.Tensor):
        from curobo.types import JointState as CuJointState

        state = CuJointState.from_position(positions, joint_names=self._joint_names)
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        return SimpleNamespace(ee_quaternion=tool_pose.quaternion)

    def _summarize_terminal_goal(self, trajectory: torch.Tensor, target_pose: list[float]) -> dict[str, Any]:
        from curobo.types import JointState as CuJointState

        terminal = trajectory[-1:].to(device=self.device, dtype=torch.float32)
        state = CuJointState.from_position(terminal, joint_names=self._joint_names)
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        fk_pos = tool_pose.position.reshape(-1, 3)[0]
        fk_quat = tool_pose.quaternion.reshape(-1, 4)[0]
        goal_pos = torch.tensor(target_pose[:3], device=self.device, dtype=torch.float32)
        goal_quat = torch.tensor(
            constraints.normalize_quaternion(target_pose[3:7]),
            device=self.device,
            dtype=torch.float32,
        )
        pos_err = float(torch.linalg.norm(fk_pos - goal_pos).item())
        quat_dot = torch.abs(torch.sum(fk_quat * goal_quat)).clamp(max=1.0)
        ori_err_rad = float((2.0 * torch.acos(quat_dot)).item())
        return {
            "terminal_position_error_m": round(pos_err, 8),
            "terminal_orientation_error_rad": round(ori_err_rad, 8),
        }

    def _base_metrics(self, started_at: float) -> dict[str, Any]:
        return {
            "solve_time_sec": round(time.time() - started_at, 6),
            "alignment": {},
            "goal": {},
            "continuity": {},
            "joint_limit": {},
        }

    def _finalize_closed_loop_records(
        self,
        *,
        result: PlannerResult,
        original_request: dict[str, Any],
        normalized_request: dict[str, Any] | None,
        started_at: float,
    ) -> None:
        run_id = result.request_id
        alignment_tolerance = float(
            (normalized_request or {}).get("alignment", {}).get(
                "tolerance_deg", self.config.level_tolerance_deg
            )
        )
        start_joint = list((normalized_request or {}).get("start_joint") or [])
        if not result.candidate_records:
            result.candidate_records = [
                self._build_candidate_record_from_summary(
                    item,
                    request_id=result.request_id,
                    run_id=run_id,
                    trajectory=None,
                    final_status=result.status,
                    final_failure_reason=result.failure_reason,
                    alignment_tolerance_deg=alignment_tolerance,
                    start_joint=start_joint,
                )
                for item in result.candidates
            ]
        existing_candidate_record_ids = {str(item.get("candidate_id")) for item in result.candidate_records}
        for item in result.candidates:
            candidate_id = str(item.get("candidate_id") or "")
            if candidate_id and candidate_id not in existing_candidate_record_ids:
                result.candidate_records.append(
                    self._build_candidate_record_from_summary(
                        item,
                        request_id=result.request_id,
                        run_id=run_id,
                        trajectory=None,
                        final_status=result.status,
                        final_failure_reason=result.failure_reason,
                        alignment_tolerance_deg=alignment_tolerance,
                        start_joint=start_joint,
                    )
                )
                existing_candidate_record_ids.add(candidate_id)
        self._append_provider_candidate_records(
            result=result,
            run_id=run_id,
            start_joint=start_joint,
            alignment_tolerance_deg=alignment_tolerance,
        )

        selected_candidate_id = result.metrics.get("selected_candidate_id")
        if not selected_candidate_id:
            for candidate in result.candidate_records:
                if candidate.get("lifecycle", {}).get("selected"):
                    selected_candidate_id = candidate.get("candidate_id")
                    result.metrics["selected_candidate_id"] = selected_candidate_id
                    break

        fallback_trace = self._build_fallback_trace(result)
        result.planner_run_record = PlannerRunRecord(
            run_id=run_id,
            request_id=result.request_id,
            created_at=datetime.fromtimestamp(started_at, timezone.utc).isoformat(),
            robot_profile=str((normalized_request or original_request).get("robot_profile", self.config.robot_profile)),
            request=dict(original_request),
            normalized_request=dict(normalized_request or {}),
            world_summary=dict(self._world_summary),
            seed_policy=dict((normalized_request or {}).get("seed_policy") or {}),
            seed_provider_reports=list(result.seed_provider_reports),
            candidates=list(result.candidate_records),
            fallback_trace=fallback_trace,
            result_status=result.status,
            failure_reason=result.failure_reason,
            selected_candidate_id=selected_candidate_id,
            metrics=dict(result.metrics),
            timings={"total_solve_time_sec": result.metrics.get("solve_time_sec")},
            artifacts=result.artifacts.to_dict(),
            environment={
                "device": self.device,
                "joint_names": list(self._joint_names),
                "tool_frames": list(self._tool_frames),
            },
        ).to_dict()

    def _build_fallback_trace(self, result: PlannerResult) -> list[dict[str, Any]]:
        trace: list[dict[str, Any]] = []
        for index, report in enumerate(result.seed_provider_reports):
            trace.append(
                {
                    "stage_index": index,
                    "stage": "seed_provider",
                    "provider": report.get("provider") or report.get("provider_name"),
                    "mode": report.get("mode"),
                    "status": report.get("status"),
                    "generated_count": report.get("generated_count", 0),
                    "accepted_count": report.get("accepted_count", 0),
                    "runtime_effect": report.get("runtime_effect"),
                }
            )
        selected_candidate_id = result.metrics.get("selected_candidate_id")
        for record in result.metrics.get("control_flow_trace") or []:
            trace.append(
                {
                    "stage_index": len(trace),
                    "stage": "control_flow",
                    **dict(record),
                }
            )
        trace.append(
            {
                "stage_index": len(trace),
                "stage": "selection",
                "status": result.status,
                "selected_candidate_id": selected_candidate_id,
                "failure_reason": result.failure_reason,
            }
        )
        return trace

    def _build_candidate_record_from_summary(
        self,
        summary: dict[str, Any],
        *,
        request_id: str,
        run_id: str,
        trajectory: torch.Tensor | None,
        final_status: str,
        final_failure_reason: str | None,
        alignment_tolerance_deg: float,
        start_joint: list[float],
    ) -> dict[str, Any]:
        candidate_id = str(summary.get("candidate_id") or f"candidate_{len(summary)}")
        source_type = self._normalize_source_type(summary.get("source_type") or summary.get("source_label"))
        source_label = str(summary.get("source_label") or source_type)
        status = str(summary.get("status") or "unknown")
        metadata = dict(summary.get("metadata") or {})
        metrics = dict(summary.get("metrics") or {})
        selected = bool(metrics.get("selected") or summary.get("selected"))
        optimizer_success = status == "success"
        max_alignment = metrics.get("max_alignment_deviation_deg")
        failure_stage = None
        failure_reason = summary.get("failure_reason") or None
        if not optimizer_success:
            failure_stage = "repair"
            failure_reason = failure_reason or status
        trajectory_points = self._trajectory_tensor_to_list(trajectory) if trajectory is not None else []
        trajectory_shape = [
            len(trajectory_points),
            len(trajectory_points[0]) if trajectory_points else 0,
        ]
        validator_metrics = validators.evaluate_hard_constraints(
            trajectory_points=trajectory_points,
            start_joint=start_joint,
            joint_limits=self._joint_limits,
            metrics=metrics,
            alignment_tolerance_deg=float(alignment_tolerance_deg),
            optimizer_success=optimizer_success,
            world_summary=self._world_summary,
            thresholds=self._validator_thresholds(),
            collision_result=self._evaluate_collision(trajectory_points),
        )
        validator_valid = bool(validator_metrics.get("valid"))
        if optimizer_success and not validator_valid:
            failure_stage = "validation"
            failure_reason = str(
                validator_metrics.get("failure_reason")
                or final_failure_reason
                or "hard_validator_failed"
            )
        positive_for_critic = bool(validator_valid and selected and final_status == STATUS_SUCCESS)
        labels = {
            "planner_status": final_status,
            "candidate_status": status,
            "failure_reason": failure_reason,
            "selected": selected,
            "validator_valid": validator_valid,
            "positive_for_diffusion": bool(validator_valid),
            "positive_for_critic": positive_for_critic,
            "negative_for_critic": not positive_for_critic,
            "fallback_recovered": False,
        }
        return CandidateRecord(
            candidate_id=candidate_id,
            run_id=run_id,
            request_id=request_id,
            source_lineage={
                "source_type": source_type,
                "source_label": source_label,
                "provider": summary.get("provider") or ("planner_native" if source_type == "planner_native" else source_type),
                "provider_mode": summary.get("provider_mode") or "native",
                "parent_candidate_id": summary.get("parent_candidate_id") or metadata.get("parent_candidate_id"),
                "raw_source_type": metadata.get("raw_source_type"),
                "raw_source_label": metadata.get("raw_source_label"),
                "rule_family_name": (metadata.get("raw_metadata") or {}).get("seed_family_name"),
                "rule_family_config": (metadata.get("raw_metadata") or {}).get("seed_family_config"),
            },
            trajectory={
                "format": "joint_position_rad",
                "shape": trajectory_shape,
                "points": trajectory_points,
            },
            lifecycle={
                "generated": True,
                "precheck_passed": status != STATUS_FAILED_PRECHECK,
                "entered_pool": bool(summary.get("entered_pool", True)),
                "repair_attempted": True,
                "repair_success": optimizer_success,
                "hard_validation_attempted": optimizer_success,
                "hard_validation_passed": validator_valid,
                "selected": selected,
                "fallback_recovered": False,
            },
            precheck=dict(summary.get("precheck") or {}),
            optimizer_result={
                **dict(summary.get("optimizer_result") or {}),
                "status": status,
                "success": optimizer_success,
                "result_status": summary.get("result_status"),
                "solve_time_sec": summary.get("solve_time_sec"),
                "trajectory_shape": summary.get("trajectory_shape") or trajectory_shape,
                "failure_reason": summary.get("failure_reason"),
            },
            validator_metrics=validator_metrics,
            labels=labels,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
            metrics=metrics,
        ).to_dict()

    def _append_provider_candidate_records(
        self,
        *,
        result: PlannerResult,
        run_id: str,
        start_joint: list[float],
        alignment_tolerance_deg: float,
    ) -> None:
        existing_ids = {str(item.get("candidate_id")) for item in result.candidate_records}
        for report in result.seed_provider_reports:
            provider_name = str(report.get("provider") or report.get("provider_name") or "seed_provider")
            provider_mode = str(report.get("mode") or "unknown")
            for candidate in report.get("candidates", []) or []:
                candidate_id = str(candidate.get("candidate_id") or "")
                if not candidate_id or candidate_id in existing_ids:
                    continue
                trajectory = (candidate.get("trajectory") or {})
                points = trajectory.get("points") or []
                source_type = self._normalize_source_type(candidate.get("source_type"))
                optimizer_success = False
                metrics = dict(candidate.get("metrics") or {})
                validator_metrics = validators.evaluate_hard_constraints(
                    trajectory_points=points,
                    start_joint=start_joint,
                    joint_limits=self._joint_limits,
                    metrics=metrics,
                    alignment_tolerance_deg=float(alignment_tolerance_deg),
                    optimizer_success=optimizer_success,
                    world_summary=self._world_summary,
                    thresholds=self._validator_thresholds(),
                    collision_result=self._evaluate_collision(points),
                )
                failure_reason = (
                    candidate.get("precheck", {}).get("failure_reason")
                    or "raw_seed_parent_only"
                )
                record = CandidateRecord(
                    candidate_id=candidate_id,
                    run_id=run_id,
                    request_id=result.request_id,
                    source_lineage={
                        "source_type": source_type,
                        "source_label": candidate.get("source_label"),
                        "provider": provider_name,
                        "provider_mode": provider_mode,
                        "rule_family_name": (candidate.get("metadata") or {}).get("seed_family_name"),
                        "rule_family_config": (candidate.get("metadata") or {}).get("seed_family_config"),
                    },
                    trajectory={
                        "format": trajectory.get("format", "joint_position_rad"),
                        "shape": trajectory.get("shape") or [
                            len(points),
                            len(points[0]) if points else 0,
                        ],
                        "points": points,
                    },
                    lifecycle={
                        "generated": True,
                        "precheck_passed": bool((candidate.get("precheck") or {}).get("valid")),
                        "entered_pool": bool(candidate.get("entered_pool", False)),
                        "repair_attempted": False,
                        "repair_success": False,
                        "hard_validation_attempted": bool(points),
                        "hard_validation_passed": False,
                        "selected": False,
                        "fallback_recovered": False,
                    },
                    precheck=dict(candidate.get("precheck") or {}),
                    optimizer_result={
                        "status": "raw_seed_parent_only",
                        "success": False,
                        "failure_reason": "raw_seed_parent_only",
                    },
                    validator_metrics=validator_metrics,
                    labels={
                        "planner_status": result.status,
                        "candidate_status": "raw_seed_parent_only",
                        "failure_reason": failure_reason,
                        "selected": False,
                        "validator_valid": False,
                        "positive_for_diffusion": False,
                        "positive_for_critic": False,
                        "negative_for_critic": True,
                        "fallback_recovered": False,
                    },
                    failure_stage="repair",
                    failure_reason=failure_reason,
                    metrics=metrics,
                ).to_dict()
                result.candidate_records.append(record)
                existing_ids.add(candidate_id)

    def _validator_thresholds(self) -> dict[str, float]:
        return {
            "goal_position_tolerance_m": float(self.config.goal_position_tolerance_m),
            "goal_orientation_tolerance_rad": float(self.config.goal_orientation_tolerance_rad),
            "max_start_gap_l2": float(self.config.max_start_gap_l2),
            "max_joint_step_l2": float(self.config.max_joint_step_l2),
            "max_joint_step_abs": float(self.config.max_joint_step_abs),
            "max_acceleration_proxy_l2": float(self.config.max_acceleration_proxy_l2),
            "collision_safety_margin_m": float(self.config.collision_safety_margin_m),
        }

    @staticmethod
    def _normalize_source_type(source_type: Any) -> str:
        raw = str(source_type or "unknown")
        if raw == "planner":
            return "planner_native"
        if raw in {"rule", "rule_seed"}:
            return "rule_seed"
        if raw == "rule_raw":
            return "rule_raw_seed"
        if raw in {"diffusion", "diffusion_seed"}:
            return "diffusion_seed"
        if raw in {"critic", "critic_selected"}:
            return "critic_selected"
        if raw in {"fallback", "fallback_rule_seed"}:
            return "fallback_rule_seed"
        return raw

    @staticmethod
    def _flatten_position_tensor(position_tensor) -> torch.Tensor:
        if position_tensor is None:
            raise ValueError("result position tensor is missing")
        if hasattr(position_tensor, "detach"):
            position_tensor = position_tensor.detach().cpu()
        while position_tensor.ndim > 2:
            if position_tensor.shape[0] == 1:
                position_tensor = position_tensor.squeeze(0)
            else:
                position_tensor = position_tensor.reshape(-1, position_tensor.shape[-1])
        if position_tensor.ndim == 1:
            position_tensor = position_tensor.unsqueeze(0)
        return position_tensor.to(dtype=torch.float32)

    @staticmethod
    def _resample_trajectory_linear(trajectory: torch.Tensor, target_horizon: int) -> torch.Tensor:
        if int(target_horizon) <= 0:
            raise ValueError(f"target_horizon must be positive, got {target_horizon}")
        if trajectory.ndim != 2:
            raise ValueError(f"trajectory must be [T, DOF], got shape={list(trajectory.shape)}")
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

    @staticmethod
    def _trajectory_tensor_to_list(trajectory: torch.Tensor) -> list[list[float]]:
        return [
            [round(float(v), 8) for v in row]
            for row in trajectory.detach().cpu().reshape(-1, trajectory.shape[-1]).tolist()
        ]

    @staticmethod
    def _result_success(result) -> bool:
        if result is None:
            return False
        success = getattr(result, "success", False)
        if hasattr(success, "any"):
            return bool(success.any())
        return bool(success)

    @staticmethod
    def _result_status(result) -> str:
        if result is None:
            return "planner_returned_none"
        return str(getattr(result, "status", "unknown"))

    def _write_artifacts(self, result: PlannerResult, out_dir: str | Path) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        result_path = out_path / "result.json"
        selected_path = out_path / "selected_trajectory.json"
        candidates_path = out_path / "candidate_summary.json"
        candidates_jsonl_path = out_path / "candidates.jsonl"
        lifecycle_path = out_path / "lifecycle.json"
        planner_run_path = out_path / "planner_run.json"
        result.artifacts = PlannerArtifacts(
            result_json=str(result_path),
            selected_trajectory_json=str(selected_path),
            candidate_summary_json=str(candidates_path),
            candidates_jsonl=str(candidates_jsonl_path),
            lifecycle_json=str(lifecycle_path),
            planner_run_json=str(planner_run_path),
        )
        if result.planner_run_record:
            result.planner_run_record["artifacts"] = result.artifacts.to_dict()
        selected_path.write_text(
            json.dumps(
                {
                    "request_id": result.request_id,
                    "status": result.status,
                    "trajectory": result.selected_trajectory,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        candidates_path.write_text(
            json.dumps(result.candidates, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        with candidates_jsonl_path.open("w", encoding="utf-8") as handle:
            for candidate in result.candidate_records:
                handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        lifecycle_path.write_text(
            json.dumps(
                {
                    "request_id": result.request_id,
                    "status": result.status,
                    "planner_run_record": result.planner_run_record,
                    "seed_provider_reports": result.seed_provider_reports,
                    "candidates": result.candidates,
                    "candidate_records_path": str(candidates_jsonl_path),
                    "fallback_trace": result.planner_run_record.get("fallback_trace", []),
                    "metrics": result.metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        planner_run_path.write_text(
            json.dumps(result.planner_run_record, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
