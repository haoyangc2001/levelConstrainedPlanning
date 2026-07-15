"""Pure Python SR5 level constrained planner core."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import yaml

from . import constraints
from .result_schema import (
    PlannerArtifacts,
    PlannerResult,
    STATUS_FAILED_ALIGNMENT,
    STATUS_FAILED_INTERNAL,
    STATUS_FAILED_PLANNER,
    STATUS_FAILED_PRECHECK,
    STATUS_SUCCESS,
)
from .robot_assets import resolve_robot_config
from .seed_provider import (
    DiffusionSeedProviderConfig,
    FileDiffusionSeedProvider,
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
    diffusion_generated_samples_path: str = (
        "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/"
        "sr5_phase10_mature_diffusion_20260715_generated_samples.json"
    )
    diffusion_checkpoint_path: str = (
        "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/"
        "sr5_phase10_mature_diffusion_20260715/best.pt"
    )

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
            "diffusion_generated_samples_path",
            "diffusion_checkpoint_path",
        ):
            if key in planner_cfg:
                setattr(cfg, key, planner_cfg[key])
        cfg.robot_config = _resolve_path(planner_cfg.get("robot_config", cfg.robot_config)) or cfg.robot_config
        cfg.obstacle_json = _resolve_path(planner_cfg.get("obstacle_json", cfg.obstacle_json)) or cfg.obstacle_json
        cfg.obstacle_rel_json = _resolve_path(
            planner_cfg.get("obstacle_rel_json", cfg.obstacle_rel_json)
        )
        return cfg


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

        world_result = build_world(
            abs_json_path=Path(self.config.obstacle_json) if self.config.obstacle_json else None,
            rel_json_path=Path(self.config.obstacle_rel_json) if self.config.obstacle_rel_json else None,
        )
        self._world_summary = dict(world_result.get("world_summary") or {})
        self._planner.update_world(SceneCfg.create(world_result["world_dict"]))
        LOGGER.info(
            "CuRobo initialized: joints=%s tool_frames=%s world=%s",
            self._joint_names,
            self._tool_frames,
            self._world_summary,
        )

    def plan(self, request: dict[str, Any], out_dir: str | Path | None = None) -> dict[str, Any]:
        t0 = time.time()
        request_id = str(request.get("request_id") or f"request_{int(t0)}")
        try:
            normalized = self._normalize_request(request)
            seed_reports = self._run_seed_providers(normalized)
            candidates, planner_attempts = self._collect_planner_candidates(normalized)
            if not candidates:
                result = PlannerResult(
                    request_id=request_id,
                    status=STATUS_FAILED_PLANNER,
                    failure_reason="curobo_plan_pose_returned_no_successful_candidate",
                    metrics=self._base_metrics(t0),
                    seed_provider_reports=seed_reports,
                    candidates=planner_attempts,
                )
            else:
                result = self._select_candidate(
                    request_id=request_id,
                    request=normalized,
                    candidates=candidates,
                    candidate_summaries=planner_attempts,
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

        if out_dir is not None:
            self._write_artifacts(result, out_dir)
        return result.to_dict()

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
                "timeout_sec": float(seed_policy.get("timeout_sec", 0.2)),
                "diffusion_artifact_pointer": seed_policy.get("diffusion_artifact_pointer"),
            },
            "metadata": dict(request.get("metadata") or {}),
        }

    def _run_seed_providers(self, request: dict[str, Any]) -> list[dict[str, Any]]:
        policy = request["seed_policy"]
        mode = policy["mode"]
        reports: list[dict[str, Any]] = []

        rule_report = {
            "provider": "rule_seed",
            "provider_name": "rule_seed",
            "mode": "fallback_available" if policy.get("fallback_to_rule_seed") else "disabled",
            "status": "not_generated_in_phase3_minimal_core",
            "generated_count": 0,
            "accepted_count": 0,
            "runtime_effect": "planner_native_seed_path_is_used_first",
        }
        reports.append(rule_report)

        diffusion_mode = "off"
        if mode == "diffusion":
            diffusion_mode = "shadow"
        elif mode == "mixed":
            diffusion_mode = "shadow"
        provider = (
            FileDiffusionSeedProvider(
                DiffusionSeedProviderConfig(
                    mode=diffusion_mode,
                    generated_samples_path=self.config.diffusion_generated_samples_path,
                    k_generate=int(policy.get("k_generate") or 0),
                    k_accept=int(policy.get("k_accept") or 0),
                    model_timeout_sec=float(policy.get("timeout_sec") or 0.2),
                    fallback_to_rule_seed=bool(policy.get("fallback_to_rule_seed", True)),
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
            }
        )
        provider_report = provider_result.to_lifecycle_dict()
        provider_report["provider"] = provider_report.get("provider_name", "diffusion_seed")
        provider_report["accepted_count"] = int(
            sum(1 for c in provider_report.get("candidates", []) if c.get("precheck", {}).get("valid"))
        )
        provider_report["runtime_effect"] = (
            "shadow_report_only_no_pool_insertion_in_phase3"
            if diffusion_mode != "off"
            else "disabled"
        )
        reports.append(provider_report)
        return reports

    def _collect_planner_candidates(self, request: dict[str, Any]) -> tuple[list[torch.Tensor], list[dict[str, Any]]]:
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
            item["metrics"] = {
                "max_alignment_deviation_deg": selection["candidate_max_alignment_deviation"][idx],
                "start_joint_gap_l2": selection["candidate_start_joint_gap_l2"][idx],
                "joint_step_jump_cost": selection["candidate_joint_step_jump_cost"][idx],
                "joint_step_max_l2": selection["candidate_joint_step_max_l2"][idx],
                "twist_smoothness_cost": selection["candidate_twist_smoothness_cost"][idx],
                "selected": idx == selected_index,
            }
        status = STATUS_SUCCESS if selection["planning_status"] == "success" else STATUS_FAILED_ALIGNMENT
        return PlannerResult(
            request_id=request_id,
            status=status,
            failure_reason=selection.get("failure_reason"),
            selected_trajectory=trajectory if status == STATUS_SUCCESS else None,
            metrics={
                **self._base_metrics(started_at),
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
                    "status": "not_evaluated_in_phase3_minimal_core"
                },
                "world": dict(self._world_summary),
            },
            seed_provider_reports=seed_reports,
            candidates=candidate_summaries,
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
        lifecycle_path = out_path / "lifecycle.json"
        result.artifacts = PlannerArtifacts(
            result_json=str(result_path),
            selected_trajectory_json=str(selected_path),
            candidate_summary_json=str(candidates_path),
            lifecycle_json=str(lifecycle_path),
        )
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
        lifecycle_path.write_text(
            json.dumps(
                {
                    "request_id": result.request_id,
                    "status": result.status,
                    "seed_provider_reports": result.seed_provider_reports,
                    "candidates": result.candidates,
                    "metrics": result.metrics,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
