#!/usr/bin/env python3
# [caohy] Phase 5-6：实现 MotionPlanner 初始化、world、pose/joint planning、FK、结果转换、ROS service。
"""CuRobo V2 motion planner ROS 2 node."""

import json
import math
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
import torch.nn.functional as F

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose

# [caohy] Phase 6：使用 wrapper 脚本激活 conda env，无需手动设置 sys.path。

from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.scene import Scene as SceneCfg
from curobo.types import JointState as CuJointState, GoalToolPose, ToolPoseCriteria
import torch

from curobo_v2_planner.rokae_asset_utils import resolve_robot_config
from curobo_v2_planner.rokae_world_utils import build_world
from curobo_v2_planner import constraint_utils
from curobo_v2_planner.seed_provider import (
    DiffusionSeedProviderConfig,
    FileDiffusionSeedProvider,
    NullDiffusionSeedProvider,
    build_lifecycle_candidate_record,
    normalize_diffusion_mode,
)


class CuroboV2PlannerNode(Node):
    """CuRobo V2 轨迹规划节点。"""

    def __init__(self):
        super().__init__('curobo_v2_planner')

        # 声明参数
        self.declare_parameter('robot_config', '')
        self.declare_parameter('obstacle_json_path', '')
        self.declare_parameter('obstacle_rel_json_path', '')
        self.declare_parameter('speed_scale', 0.5)
        self.declare_parameter('collision_cache_obb', 2)
        self.declare_parameter('use_cuda_graph', True)
        self.declare_parameter('sim_joint_state_topic', 'sim_joint_states')
        self.declare_parameter('use_real_robot', False)
        self.declare_parameter('diffusion_seed_mode', 'off')
        self.declare_parameter(
            'diffusion_seed_generated_samples_path',
            '/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_phase4_generated_samples.json',
        )
        self.declare_parameter(
            'diffusion_seed_checkpoint_path',
            '/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/sr5_phase4_smoke_baseline/best.pt',
        )
        self.declare_parameter('diffusion_seed_k_generate', 4)
        self.declare_parameter('diffusion_seed_k_accept', 2)
        self.declare_parameter('diffusion_seed_model_timeout_sec', 0.2)
        self.declare_parameter('diffusion_seed_max_start_gap_l2', 0.05)
        self.declare_parameter('diffusion_seed_max_step_l2', 1.0)
        self.declare_parameter('diffusion_seed_joint_abs_limit', 6.2832)
        self.declare_parameter('diffusion_seed_fallback_to_rule_seed', True)
        self.declare_parameter('diffusion_seed_allow_real_robot_candidate', False)

        # 读取参数
        robot_config_path = self.get_parameter('robot_config').value
        obstacle_json = self.get_parameter('obstacle_json_path').value
        obstacle_rel_json = self.get_parameter('obstacle_rel_json_path').value
        self._speed_scale = float(self.get_parameter('speed_scale').value)
        # [caohy] Phase 6.5/6.6：记录是否启用 CUDA graph，后续 warmup 和 plan attempt 都按该开关执行。
        self._use_cuda_graph = bool(self.get_parameter('use_cuda_graph').value)
        collision_cache_obb = int(self.get_parameter('collision_cache_obb').value)
        sim_joint_topic = self.get_parameter('sim_joint_state_topic').value
        self._use_real_robot = self._coerce_bool_param(
            self.get_parameter('use_real_robot').value,
            default=False,
        )
        # [caohy] RViz 卡顿收敛：默认把超大的 level_check_info（约束详情）
        # 和 [S5-B] 探针日志降到轻量模式，只有排障时再通过环境变量开回完整输出。
        self._verbose_level_info_in_response = os.environ.get(
            'CUROBO_RVIZ_VERBOSE_LEVEL_INFO', ''
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        self._verbose_probe_logs = os.environ.get(
            'CUROBO_RVIZ_VERBOSE_PROBE_LOGS', ''
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        self._task36_marker_only_logs = os.environ.get(
            'CUROBO_TASK36_MARKER_ONLY_LOGS', ''
        ).strip().lower() in ('1', 'true', 'yes', 'on')
        # [caohy] Task 41：为“20 条轨迹种子生命周期落盘”建立本次 planner 运行级目录，
        # 后续每条 move 都把完整 seed/candidate/selector 证据写进这个目录，避免只剩零散日志。
        self._level_plan_lifecycle_root = self._init_level_plan_lifecycle_root()
        if self._task36_marker_only_logs:
            # [caohy] Task 36：仅本次 RViz 人工观察用，压住普通 INFO/WARN 日志，
            # 让右下角日志面板只突出 Move 02 / Move 04 的大字标记。
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.ERROR)

        self.get_logger().info('curobo_v2_planner initializing...')
        self.get_logger().info(f'  robot_config: {robot_config_path}')
        self.get_logger().info(f'  speed_scale: {self._speed_scale}')

        # 加载机器人配置
        robot_cfg = resolve_robot_config(
            robot_config_path=Path(robot_config_path),
            auto_generate_spheres=False,
        )
        # [caohy] Task 10：缓存当前 planner 实际使用的 robot config，
        # 后续 sequence（序列目标）旁支构建 MotionRetargeterCfg 时必须复用同一份运行态资产。
        self._robot_cfg = robot_cfg
        self._collision_cache = {'obb': collision_cache_obb}
        self._world_dict = None
        self._world_scene = None
        self._world_summary = {
            'abs_count': 0,
            'rel_count': 0,
            'total_count': 0,
        }
        # [caohy] RViz 障碍物面板会写 autosave JSON；规划前按 mtime 热刷新，
        # 避免 planner 只使用启动瞬间加载到的旧 collision world。
        self._obstacle_json_path = Path(obstacle_json) if obstacle_json else None
        self._obstacle_rel_json_path = Path(obstacle_rel_json) if obstacle_rel_json else None
        self._obstacle_json_mtime_ns = None
        self._obstacle_rel_json_mtime_ns = None

        # 初始化 MotionPlanner
        self.get_logger().info('Creating MotionPlannerCfg...')
        cfg = MotionPlannerCfg.create(
            robot=robot_cfg,
            scene_model=None,
            collision_cache=self._collision_cache,
            use_cuda_graph=self._use_cuda_graph,
        )

        self.get_logger().info('Initializing MotionPlanner...')
        self._planner = MotionPlanner(cfg)

        self.get_logger().info('Warming up MotionPlanner...')
        self._planner.warmup(
            enable_graph=self._use_cuda_graph,
            num_warmup_iterations=3,
        )

        self._joint_names = list(self._planner.joint_names)
        self._tool_frames = list(self._planner.tool_frames)
        # [caohy] Task 29：直接缓存当前 CuRobo 真正生效的关节位置限位（已包含 position_limit_clip），
        # 便于后续在 seed / prepared seed 上直接回答“是哪根关节先超限”。
        self._joint_position_limit_lower = None
        self._joint_position_limit_upper = None
        self._joint_dynamic_limit_ranges = {}
        try:
            joint_limits = self._planner.kinematics.get_joint_limits()
            lower_tensor = joint_limits.position[0].detach().cpu().reshape(-1)
            upper_tensor = joint_limits.position[1].detach().cpu().reshape(-1)
            self._joint_position_limit_lower = [float(v) for v in lower_tensor.tolist()]
            self._joint_position_limit_upper = [float(v) for v in upper_tensor.tolist()]
            for limit_name in ('velocity', 'acceleration', 'jerk', 'effort', 'torque'):
                limit_tensor = getattr(joint_limits, limit_name, None)
                if limit_tensor is None:
                    continue
                try:
                    lower = limit_tensor[0].detach().cpu().reshape(-1)
                    upper = limit_tensor[1].detach().cpu().reshape(-1)
                    self._joint_dynamic_limit_ranges[limit_name] = {
                        'lower': [float(v) for v in lower.tolist()],
                        'upper': [float(v) for v in upper.tolist()],
                    }
                except Exception:
                    continue
        except Exception as exc:
            self.get_logger().warn(f'Failed to cache joint position limits: {exc}')

        self.get_logger().info(f'  joint_names: {self._joint_names}')
        self.get_logger().info(f'  tool_frames: {self._tool_frames}')

        # [caohy] Phase 9.8：初始化 IK solver（用于 alignment seed 生成）
        self.get_logger().info('Initializing IK solver...')
        from curobo._src.solver.solver_ik import IKSolver
        from curobo._src.solver.solver_ik_cfg import IKSolverCfg
        ik_cfg = IKSolverCfg.create(
            robot=robot_cfg,
            num_seeds=32,
            use_cuda_graph=self._use_cuda_graph,
        )
        self._ik_solver = IKSolver(ik_cfg)
        self.get_logger().info('IK solver ready.')

        # 当前关节状态
        self._current_joint_position = None
        self._level_plan_request_index = 0
        self._active_tool_pose_criteria = None
        self._active_terminal_pose_axes_weight_factor = None
        # [caohy] SR5 100 条回归中单条规划可能超过 motion 的短超时；
        # 串行化 CuRobo 入口，避免多个 ROS service 回调同时重入同一个 GPU planner 实例。
        self._planning_lock = threading.Lock()
        # [caohy] Task 8 Phase 3：retract fallback 跟随当前 robot_config，
        # 避免 SR5 profile 下仍回退到 CR7 默认起点。
        self._retract_config = self._extract_default_joint_position(robot_cfg)

        # 订阅 sim_joint_states
        self._joint_sub = self.create_subscription(
            JointState,
            sim_joint_topic,
            self._joint_state_callback,
            10,
        )

        # 初始化 world
        self._init_world(obstacle_json, obstacle_rel_json)

        # 创建 service servers
        self._cb_group = ReentrantCallbackGroup()

        # [caohy] Phase 5：point_to_point_trajectory_planning service
        self._point_to_point_srv = self.create_service(
            self._load_service_type('PointToPointTrajectoryPlanning'),
            'point_to_point_trajectory_planning',
            self._handle_point_to_point_request,
            callback_group=self._cb_group,
        )

        # [caohy] Phase 6：joint_target_trajectory_planning service
        self._joint_target_srv = self.create_service(
            self._load_service_type('JointTargetTrajectoryPlanning'),
            'joint_target_trajectory_planning',
            self._handle_joint_target_request,
            callback_group=self._cb_group,
        )

        # [caohy] Phase 6：plan_joint_and_pose service (FK query)
        self._fk_srv = self.create_service(
            self._load_service_type('TargetPose'),
            'plan_joint_and_pose',
            self._handle_fk_request,
            callback_group=self._cb_group,
        )

        self.get_logger().info('curobo_v2_planner ready.')

    def _run_planning_callback_serialized(self, label: str, handler, request, response):
        """串行执行共享 CuRobo planner 的服务回调。"""
        wait_start = time.time()
        if self._planning_lock.locked():
            self.get_logger().warn(
                f'{label} waiting for active planning request to finish before starting.'
            )
        with self._planning_lock:
            waited = time.time() - wait_start
            if waited > 0.001:
                self.get_logger().info(f'{label} acquired planner lock after {waited:.3f}s')
            return handler(request, response)

    def _load_service_type(self, service_name: str):
        """动态加载 service 类型。"""
        import importlib
        module = importlib.import_module('interfaces.srv')
        return getattr(module, service_name)

    def _extract_default_joint_position(self, robot_cfg: dict) -> list[float]:
        """从当前运行态 robot_config 提取 default_joint_position。"""
        default_retract = [0.0, 0.11, 0.82, 0.0, 0.94, 0.0]
        try:
            cspace_cfg = robot_cfg['robot_cfg']['kinematics']['cspace']
            default_joint_position = cspace_cfg.get('default_joint_position')
            if isinstance(default_joint_position, list) and default_joint_position:
                resolved = [float(value) for value in default_joint_position]
                self.get_logger().info(
                    f'  retract_fallback(default_joint_position): {resolved}'
                )
                return resolved
        except Exception as exc:
            self.get_logger().warn(
                f'Failed to read default_joint_position from robot_config, '
                f'fallback to legacy retract config: {exc}'
            )
        return default_retract

    def _joint_state_callback(self, msg: JointState):
        """接收 sim_joint_states 更新当前关节位置。"""
        if msg.position and len(msg.position) >= len(self._joint_names):
            self._current_joint_position = list(msg.position[:len(self._joint_names)])

    def _get_current_joint_position(self) -> list:
        """获取当前关节位置。"""
        if self._current_joint_position is not None:
            return self._current_joint_position
        return self._retract_config

    def _get_current_joint_debug_info(self) -> dict:
        """返回当前规划入口使用的起始关节及其来源。"""
        if self._current_joint_position is not None:
            return {
                'current_joint_source': 'sim_joint_states',
                'joint_position': list(self._current_joint_position),
            }
        return {
            'current_joint_source': 'retract_fallback',
            'joint_position': list(self._retract_config),
        }

    def _log_probe_info(self, message: str) -> None:
        """[caohy] 默认把大体积探针日志降到 DEBUG，避免 RViz 日志面板被刷死。"""
        if self._verbose_probe_logs:
            self.get_logger().info(message)
        else:
            self.get_logger().debug(message)

    def _compact_level_check_info(self, level_info: dict) -> dict:
        """[caohy] 默认只返回约束结果摘要，避免把整包 waypoint / branch 明细灌进 RViz。"""
        if self._verbose_level_info_in_response or not isinstance(level_info, dict):
            return level_info

        summary_keys = [
            'planning_status',
            'failure_reason',
            'candidate_count',
            'alignment_valid_count',
            'alignment_tolerance_deg',
            'strict_level',
            'selected_index',
            'selected_max_alignment_deviation',
            'selected_start_joint_gap_l2',
            'selected_joint_step_jump_cost',
            'selected_joint_step_max_abs',
            'selected_joint_step_max_l2',
            'selected_twist_smoothness_cost',
            # [caohy] Task 30：显式保留最终选中来源标签，避免后续再通过 selected_index 反推。
            'selected_source_label',
            'alignment_seed_trajopt_preference_applied',
            'alignment_seed_trajopt_preference_reason',
            'alignment_seed_trajopt_preference_from_index',
            'alignment_seed_trajopt_preference_to_index',
            'candidate_source_labels',
            'seed_candidate_added',
            'seed_candidate_index',
            'seed_trajopt_candidate_added',
            'seed_trajopt_candidate_index',
            'seed_trajopt_smoothed_candidate_added',
            'seed_trajopt_smoothed_candidate_index',
            'seed_trajopt_bridged_candidate_added',
            'seed_trajopt_bridged_candidate_index',
            'start_state_debug',
            'lifecycle_artifact_path',
            'lifecycle_summary',
        ]
        compact = {key: level_info[key] for key in summary_keys if key in level_info}

        for key in (
            'candidate_alignment_valid',
            'candidate_max_alignment_deviation',
            'candidate_joint_step_max_l2',
            # [caohy] Task 35：保留候选逐点对齐剖面，定位 alignment_seed_trajopt
            # 在原 YAML 状态机链路中具体哪个轨迹点超过 3 度容差。
            'candidate_alignment_profiles',
        ):
            if key in level_info:
                compact[key] = level_info[key]

        seed_debug = level_info.get('seed_debug_info')
        if isinstance(seed_debug, dict):
            compact['seed_debug_info'] = {
                key: seed_debug[key]
                for key in (
                    'ik_fail_count',
                    'num_waypoints',
                    'plan_request_index',
                    'start_twist_deg',
                    'goal_twist_deg',
                    'seed_input_start_joint',
                    'seed_first_joint',
                    'seed_last_joint',
                    'seed_last_joint_limit_summary',
                    'seed_trajectory_limit_summary',
                    'max_step_jump_l2',
                    'max_step_jump_index',
                    'max_step_jump_source',
                    'max_step_jump_delta',
                )
                if key in seed_debug
            }

        omitted_keys = sorted(key for key in level_info.keys() if key not in compact)
        if omitted_keys:
            compact['omitted_detail_keys'] = omitted_keys

        return compact

    def _resolve_repo_root(self) -> Path:
        """[caohy] Task 41：定位仓库根目录，统一给生命周期日志和 JSON 落盘使用。"""
        search_dir = Path(__file__).resolve().parent
        for candidate in (search_dir,) + tuple(search_dir.parents):
            if (candidate / 'readCaohy').is_dir() and (candidate / 'launch').is_dir():
                return candidate
        return Path('/home/tanshan/tashan_robot')

    def _init_level_plan_lifecycle_root(self) -> Path:
        """[caohy] Task 41：创建本次 planner 进程对应的生命周期日志目录。"""
        repo_root = self._resolve_repo_root()
        session_label = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_root = (
            repo_root
            / 'readCaohy'
            / 'logs'
            / 'trajectory_planning'
            / 'level_plan_lifecycle'
            / f'run_{session_label}_pid{os.getpid()}'
        )
        log_root.mkdir(parents=True, exist_ok=True)
        return log_root

    def _round_nested_debug_value(self, value, float_digits: int = 6):
        """[caohy] Task 41：把 tensor / numpy-like / Path 统一转成可落 JSON 的 Python 值。"""
        if value is None or isinstance(value, (bool, int, str)):
            return value
        if isinstance(value, float):
            if math.isfinite(value):
                return round(float(value), int(float_digits))
            return str(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, SimpleNamespace):
            return self._round_nested_debug_value(vars(value), float_digits=float_digits)
        if isinstance(value, dict):
            return {
                str(key): self._round_nested_debug_value(val, float_digits=float_digits)
                for key, val in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [
                self._round_nested_debug_value(item, float_digits=float_digits)
                for item in value
            ]
        if hasattr(value, 'detach'):
            try:
                return self._round_nested_debug_value(
                    value.detach().cpu().tolist(),
                    float_digits=float_digits,
                )
            except Exception:
                return str(value)
        if hasattr(value, 'tolist') and not isinstance(value, str):
            try:
                return self._round_nested_debug_value(value.tolist(), float_digits=float_digits)
            except Exception:
                return str(value)
        return str(value)

    def _trajectory_tensor_to_list(self, trajectory) -> list:
        """[caohy] Task 41：把候选/种子轨迹统一规整成 [T, DOF] 的 Python list。"""
        if trajectory is None:
            return []
        data = trajectory
        if hasattr(data, 'detach'):
            data = data.detach().cpu()
        while hasattr(data, 'ndim') and data.ndim > 2:
            if data.shape[0] == 1:
                data = data.squeeze(0)
            else:
                data = data.reshape(-1, data.shape[-1])
        if hasattr(data, 'ndim') and data.ndim == 1:
            data = data.unsqueeze(0)
        return self._round_nested_debug_value(data, float_digits=6) or []

    def _summarize_trajectory_points(self, trajectory_points: list) -> dict:
        """[caohy] Task 41：给 lifecycle 中的每条轨迹补基础摘要，便于快速浏览。"""
        if not trajectory_points:
            return {
                'point_count': 0,
                'dof': 0,
                'first_point': None,
                'last_point': None,
            }
        return {
            'point_count': int(len(trajectory_points)),
            'dof': int(len(trajectory_points[0])) if trajectory_points[0] else 0,
            'first_point': trajectory_points[0],
            'last_point': trajectory_points[-1],
        }

    def _build_result_debug_payload(self, result) -> dict:
        """[caohy] Task 41：把 cuRobo result 的关键成功/失败摘要转成可落盘结构。"""
        if result is None:
            return {'result_is_none': True}
        payload = {
            'result_is_none': False,
            'status': self._result_status(result),
            'success': self._tensor_to_debug_value(getattr(result, 'success', None)),
            'valid_query': self._tensor_to_debug_value(getattr(result, 'valid_query', None)),
            'position_tolerance': self._tensor_to_debug_value(
                getattr(result, 'position_tolerance', None),
            ),
            'orientation_tolerance': self._tensor_to_debug_value(
                getattr(result, 'orientation_tolerance', None),
            ),
        }
        payload.update(self._extract_solution_debug_summary(result))
        payload.update(self._extract_js_solution_debug_summary(result))
        payload.update(self._extract_interpolated_plan_debug_summary(result))
        payload.update(self._extract_retained_result_decision_summary(result))
        return self._round_nested_debug_value(payload, float_digits=6)

    def _build_lifecycle_summary(self, lifecycle_data: dict) -> dict:
        """[caohy] Task 41：生成轻量摘要，透传进 level_check_info 供 trajectory.json 快速反查。"""
        candidates = lifecycle_data.get('candidates', []) if isinstance(lifecycle_data, dict) else []
        selected = lifecycle_data.get('selection', {}) if isinstance(lifecycle_data, dict) else {}
        return {
            'planner_attempt_count': len(lifecycle_data.get('planner_attempts', [])),
            'planner_legacy_attempt_count': len(lifecycle_data.get('planner_legacy_attempts', [])),
            'candidate_count': len(candidates),
            'selected_candidate_id': selected.get('selected_candidate_id'),
            'selected_source_label': selected.get('selected_source_label'),
        }

    def _persist_level_plan_lifecycle(self, lifecycle_data: dict) -> str:
        """[caohy] Task 41：把单条严格约束 move 的完整生命周期写成独立 JSON 文件。"""
        move_index = lifecycle_data.get('plan_request_index')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        filename = (
            f'move_{int(move_index):02d}_{timestamp}.json'
            if move_index is not None
            else f'move_unknown_{timestamp}.json'
        )
        file_path = self._level_plan_lifecycle_root / filename
        payload = self._round_nested_debug_value(lifecycle_data, float_digits=6)
        with open(file_path, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return str(file_path)

    def _attach_lifecycle_artifact(self, result_payload: dict, lifecycle_data: dict) -> dict:
        """[caohy] Task 41：在返回结果里补 lifecycle 文件路径和轻量摘要。"""
        lifecycle_path = self._persist_level_plan_lifecycle(lifecycle_data)
        level_info = result_payload.setdefault('level_check_info', {})
        level_info['lifecycle_artifact_path'] = lifecycle_path
        level_info['lifecycle_summary'] = self._build_lifecycle_summary(lifecycle_data)
        return result_payload

    def _summarize_joint_limit_violation(self, joint_values) -> dict:
        """[caohy] Task 29：把单个关节向量相对当前位置限位的超界情况压成易读摘要。"""
        summary = {
            'has_violation': False,
            'exceeded_joint_names': [],
            'exceeded_joint_indices': [],
            'max_violation': 0.0,
            'max_violation_joint_name': None,
            'max_abs_joint_value': None,
        }
        if (
            joint_values is None
            or self._joint_position_limit_lower is None
            or self._joint_position_limit_upper is None
        ):
            return summary

        if hasattr(joint_values, 'detach'):
            values = joint_values.detach().cpu().reshape(-1).tolist()
        else:
            values = list(joint_values)
        if len(values) != len(self._joint_names):
            summary['value_count'] = len(values)
            return summary

        max_violation = 0.0
        max_violation_joint_name = None
        exceeded_joint_names = []
        exceeded_joint_indices = []
        for idx, value in enumerate(values):
            lower = float(self._joint_position_limit_lower[idx])
            upper = float(self._joint_position_limit_upper[idx])
            violation = max(lower - float(value), float(value) - upper, 0.0)
            if violation > 0.0:
                exceeded_joint_names.append(self._joint_names[idx])
                exceeded_joint_indices.append(idx)
                if violation > max_violation:
                    max_violation = violation
                    max_violation_joint_name = self._joint_names[idx]

        summary['has_violation'] = bool(exceeded_joint_names)
        summary['exceeded_joint_names'] = exceeded_joint_names
        summary['exceeded_joint_indices'] = exceeded_joint_indices
        summary['max_violation'] = round(float(max_violation), 6)
        summary['max_violation_joint_name'] = max_violation_joint_name
        summary['max_abs_joint_value'] = round(
            float(max(abs(float(v)) for v in values)),
            6,
        ) if values else None
        return summary

    def _summarize_trajectory_joint_limit_violation(self, trajectory: torch.Tensor) -> dict:
        """[caohy] Task 29：回答整条 seed 轨迹从哪个 waypoint 开始越界、最严重越界在哪里。"""
        summary = {
            'has_violation': False,
            'first_exceed_waypoint_index': None,
            'first_exceed_joint_names': [],
            'worst_waypoint_index': None,
            'worst_joint_name': None,
            'max_violation': 0.0,
        }
        if trajectory is None or trajectory.ndim != 2:
            return summary

        first_index = None
        first_joint_names = []
        worst_index = None
        worst_joint_name = None
        worst_violation = 0.0
        for waypoint_index in range(int(trajectory.shape[0])):
            joint_summary = self._summarize_joint_limit_violation(trajectory[waypoint_index])
            if joint_summary['has_violation'] and first_index is None:
                first_index = waypoint_index
                first_joint_names = list(joint_summary['exceeded_joint_names'])
            if float(joint_summary['max_violation']) > worst_violation:
                worst_violation = float(joint_summary['max_violation'])
                worst_index = waypoint_index
                worst_joint_name = joint_summary['max_violation_joint_name']

        summary['has_violation'] = first_index is not None
        summary['first_exceed_waypoint_index'] = first_index
        summary['first_exceed_joint_names'] = first_joint_names
        summary['worst_waypoint_index'] = worst_index
        summary['worst_joint_name'] = worst_joint_name
        summary['max_violation'] = round(float(worst_violation), 6)
        return summary

    def _init_world(self, obstacle_json: str, obstacle_rel_json: str):
        """从障碍物 JSON 文件初始化 world。"""
        self._obstacle_json_path = Path(obstacle_json) if obstacle_json else None
        self._obstacle_rel_json_path = Path(obstacle_rel_json) if obstacle_rel_json else None
        self._reload_world_from_obstacle_files(force=True, reason='startup')

    def _get_obstacle_file_mtime_ns(self, path: Path | None):
        """[caohy] 返回障碍物 autosave 文件 mtime；文件不存在时返回 None。"""
        if path is None:
            return None
        try:
            if not path.exists():
                return None
            return int(path.stat().st_mtime_ns)
        except OSError as exc:
            self.get_logger().warn(f'Failed to stat obstacle file {path}: {exc}')
            return None

    def _reload_world_from_obstacle_files(self, force: bool = False, reason: str = '') -> bool:
        """[caohy] 从 abs/rel autosave JSON 重建并更新 CuRobo collision world。"""
        abs_mtime_ns = self._get_obstacle_file_mtime_ns(self._obstacle_json_path)
        rel_mtime_ns = self._get_obstacle_file_mtime_ns(self._obstacle_rel_json_path)
        if (
            not force
            and abs_mtime_ns == self._obstacle_json_mtime_ns
            and rel_mtime_ns == self._obstacle_rel_json_mtime_ns
        ):
            return False

        abs_path = (
            self._obstacle_json_path
            if self._obstacle_json_path is not None and self._obstacle_json_path.exists()
            else None
        )
        rel_path = (
            self._obstacle_rel_json_path
            if self._obstacle_rel_json_path is not None and self._obstacle_rel_json_path.exists()
            else None
        )

        world_result = build_world(abs_json_path=abs_path, rel_json_path=rel_path)
        world_dict = world_result['world_dict']
        summary = world_result['world_summary']
        world_scene = SceneCfg.create(world_dict)

        # [caohy] Task 10：缓存当前 planner 实际加载的 world dict / summary，
        # sequence（序列目标）旁支构建 MotionRetargeterCfg 时必须沿用同一份场景配置。
        self._world_dict = world_dict
        self._world_scene = world_scene if summary['total_count'] > 0 else None
        self._world_summary = summary
        self._obstacle_json_mtime_ns = abs_mtime_ns
        self._obstacle_rel_json_mtime_ns = rel_mtime_ns

        # [caohy] 当前 cuRobo V2 的 update_world(...) 需要 SceneCfg（场景配置对象），
        # 不能直接传 world dict；空 world 也走 update_world，用于删除障碍物后的清场。
        self._planner.update_world(world_scene)
        self.get_logger().info(
            f'World loaded: {summary["total_count"]} obstacles '
            f'(abs={summary["abs_count"]}, rel={summary["rel_count"]}, reason={reason or "manual"})'
        )
        self.get_logger().info('World updated in MotionPlanner.')
        return True

    def _refresh_world_if_obstacle_files_changed(self, reason: str) -> bool:
        """[caohy] 规划请求前热刷新 RViz 保存的障碍物 JSON。"""
        try:
            return self._reload_world_from_obstacle_files(force=False, reason=reason)
        except Exception as exc:
            self.get_logger().error(f'Failed to refresh obstacle world before {reason}: {exc}')
            raise

    def _handle_point_to_point_request(self, request, response):
        return self._run_planning_callback_serialized(
            'point_to_point_trajectory_planning',
            self._handle_point_to_point_request_unlocked,
            request,
            response,
        )

    def _handle_point_to_point_request_unlocked(self, request, response):
        """处理 point_to_point_trajectory_planning service 请求。"""
        old_criteria = None
        try:
            self._refresh_world_if_obstacle_files_changed('point_to_point_trajectory_planning')
            target_pose = request.target_pose
            pos = [target_pose.position.x, target_pose.position.y, target_pose.position.z]
            quat = [target_pose.orientation.w, target_pose.orientation.x,
                    target_pose.orientation.y, target_pose.orientation.z]

            # [caohy] Phase 8：提取 hold_vec_weight，支持 Int32[] 和普通 list
            hold_vec_weight = [float(w.data if hasattr(w, 'data') else w) for w in request.hold_vec_weight]
            speed_scale = float(request.speed_scale) if request.speed_scale > 0 else self._speed_scale

            # [caohy] Phase 10：检查是否使用约束规划模式
            use_level = bool(getattr(request, 'use_level_first_selection', False))

            self.get_logger().info(
                f'Plan request: pos={[round(v, 3) for v in pos]}, '
                f'quat={[round(v, 3) for v in quat]}, '
                f'hold_vec_weight={hold_vec_weight}, speed_scale={speed_scale}, '
                f'use_level_first_selection={use_level}'
            )

            # [caohy] Phase 10：如果 use_level_first_selection=True，调用完整约束规划
            if use_level:
                level_tolerance_deg = float(getattr(request, 'level_tolerance_deg', 3.0))
                strict_level = bool(getattr(request, 'strict_level', True))
                num_candidates = int(getattr(request, 'num_candidates', 4))
                enable_alignment_seed = bool(getattr(request, 'enable_alignment_seed', True))

                current_joint_debug = self._get_current_joint_debug_info()
                start_joint = list(current_joint_debug['joint_position'])
                full_target = pos + quat

                level_result = self.plan_single_level_constrained(
                    start_joint=start_joint,
                    target_pose=full_target,
                    hold_vec_weight=hold_vec_weight,
                    level_tolerance_deg=level_tolerance_deg,
                    strict_level=strict_level,
                    num_candidates=num_candidates,
                    enable_alignment_seed=enable_alignment_seed,
                    speed_scale=speed_scale,
                    start_state_debug=current_joint_debug,
                )

                level_info = level_result.get('level_check_info', {})
                response_level_info = self._compact_level_check_info(level_info)
                solve_time = level_result.get('solve_time')
                solve_time_str = f'{float(solve_time):.3f}s' if solve_time is not None else 'n/a'
                response.success = level_result['status'] not in (
                    'failed_alignment_precheck', 'failed_alignment_constraint', 'curobo_failed',
                )
                response.message = (
                    f'{level_result["status"]}, '
                    f'solve_time={solve_time_str}, '
                    f'steps={len(level_result.get("trajectory_points", []))}'
                    f'||level_check_info:{json.dumps(response_level_info, ensure_ascii=False)}'
                )
                level_info_file = os.environ.get('CUROBO_TASK36_LEVEL_INFO_FILE', '').strip()
                if level_info_file:
                    # [caohy] Task 36：marker-only 模式会压住 ROS INFO 日志；本次验证把
                    # level_check_info 另写到文件，既不刷 RViz，又能分析最终选中来源。
                    try:
                        with open(level_info_file, 'a', encoding='utf-8') as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        'message': response.message,
                                        'level_check_info': response_level_info,
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + '\n'
                            )
                    except Exception as exc:
                        self.get_logger().error(
                            f'Failed to write CUROBO_TASK36_LEVEL_INFO_FILE: {exc}'
                        )
                # [caohy] Task 31 Step 1：写入未经 compact 的完整 level_info，
                # 保留 waypoint_debug（含 raw / wrapped joint 位置）用于种子形成层分析。
                level_info_file_full = os.environ.get(
                    'CUROBO_TASK36_LEVEL_INFO_FILE_FULL', '',
                ).strip()
                if level_info_file_full:
                    try:
                        with open(level_info_file_full, 'a', encoding='utf-8') as handle:
                            handle.write(
                                json.dumps(
                                    {
                                        'message': response.message,
                                        'level_check_info': level_info,
                                    },
                                    ensure_ascii=False,
                                    default=str,
                                )
                                + '\n'
                            )
                    except Exception as exc:
                        self.get_logger().error(
                            f'Failed to write CUROBO_TASK36_LEVEL_INFO_FILE_FULL: {exc}'
                        )

                # 构造 JointTrajectory
                traj_points = level_result.get('trajectory_points', [])
                interp_dt = level_result.get('interpolation_dt', 0.016)
                if traj_points:
                    joint_traj = self._build_joint_trajectory(traj_points, interp_dt)
                    response.joint_trajectory = joint_traj
                else:
                    response.joint_trajectory = JointTrajectory()

                self.get_logger().info(f'Level plan: {response.message}')
                return response

            # [caohy] Phase 8：普通点到点模式，应用 hold_vec_weight 约束
            old_criteria = self._apply_hold_vec_weight(hold_vec_weight)

            start_joint = self._get_current_joint_position()

            goal = GoalToolPose(
                tool_frames=self._tool_frames,
                position=torch.tensor(
                    [[[[[pos[0], pos[1], pos[2]]]]]],
                    device='cuda:0', dtype=torch.float32,
                ),
                quaternion=torch.tensor(
                    [[[[[quat[0], quat[1], quat[2], quat[3]]]]]],
                    device='cuda:0', dtype=torch.float32,
                ),
            )

            current_state = CuJointState.from_position(
                torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )

            t0 = time.time()
            result = self._planner.plan_pose(
                goal,
                current_state,
                max_attempts=5,
                enable_graph_attempt=1 if self._use_cuda_graph else 0,
            )
            solve_time = time.time() - t0

            if self._result_success(result):
                self._log_plan_result_summary('point_to_point.plan_pose', result)
                js_result = result.get_interpolated_plan()
                trajectory = self._to_joint_trajectory(js_result, speed_scale)

                response.success = True
                response.message = f'success, solve_time={solve_time:.3f}s, steps={len(trajectory.points)}'
                response.joint_trajectory = trajectory

                self.get_logger().info(f'Plan success: {len(trajectory.points)} steps, {solve_time:.3f}s')
            else:
                response.success = False
                response.message = f'planning failed: {self._result_status(result)}'
                response.joint_trajectory = JointTrajectory()
                self.get_logger().warn(f'Plan failed: {self._result_status(result)}')

        except Exception as e:
            response.success = False
            response.message = f'error: {str(e)}'
            response.joint_trajectory = JointTrajectory()
            self.get_logger().error(f'Plan error: {e}')
        finally:
            # [caohy] Phase 8：恢复旧 criteria
            self._restore_criteria(old_criteria)

        return response

    def _handle_joint_target_request(self, request, response):
        return self._run_planning_callback_serialized(
            'joint_target_trajectory_planning',
            self._handle_joint_target_request_unlocked,
            request,
            response,
        )

    def _handle_joint_target_request_unlocked(self, request, response):
        """处理 joint_target_trajectory_planning service 请求。"""
        try:
            self._refresh_world_if_obstacle_files_changed('joint_target_trajectory_planning')
            target_joint = list(request.target_joint_position.position)
            speed_scale = float(request.speed_scale) if request.speed_scale > 0 else self._speed_scale

            self.get_logger().info(
                f'Joint target request: target={[round(v, 3) for v in target_joint]}, '
                f'speed_scale={speed_scale}'
            )

            start_joint = self._get_current_joint_position()

            start_state = CuJointState.from_position(
                torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )
            goal_state = CuJointState.from_position(
                torch.tensor([target_joint], device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )

            t0 = time.time()
            result = self._planner.plan_cspace(
                goal_state,
                start_state,
                max_attempts=5,
                enable_graph_attempt=1 if self._use_cuda_graph else 0,
            )
            solve_time = time.time() - t0

            if self._result_success(result):
                js_result = result.get_interpolated_plan()
                trajectory = self._to_joint_trajectory(js_result, speed_scale)

                response.success = True
                response.message = f'success, solve_time={solve_time:.3f}s, steps={len(trajectory.points)}'
                response.joint_trajectory = trajectory

                self.get_logger().info(f'Joint plan success: {len(trajectory.points)} steps, {solve_time:.3f}s')
            else:
                response.success = False
                response.message = f'planning failed: {self._result_status(result)}'
                response.joint_trajectory = JointTrajectory()
                self.get_logger().warn(f'Joint plan failed: {self._result_status(result)}')

        except Exception as e:
            response.success = False
            response.message = f'error: {str(e)}'
            response.joint_trajectory = JointTrajectory()
            self.get_logger().error(f'Joint plan error: {e}')

        return response

    def _handle_fk_request(self, request, response):
        return self._run_planning_callback_serialized(
            'plan_joint_and_pose',
            self._handle_fk_request_unlocked,
            request,
            response,
        )

    def _handle_fk_request_unlocked(self, request, response):
        """处理 plan_joint_and_pose service 请求（FK 查询）。

        TargetPose.srv:
          Request: speed, joint_state, pose
          Response: joint_state, pose, success
        """
        try:
            joint_position = list(request.joint_state.position)

            self.get_logger().info(f'FK request: joint={[round(v, 3) for v in joint_position]}')

            state = CuJointState.from_position(
                torch.tensor([joint_position], device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )

            kin_state = self._planner.compute_kinematics(state)
            tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])

            pos = tool_pose.position.reshape(-1, 3).tolist()[0]
            quat = tool_pose.quaternion.reshape(-1, 4).tolist()[0]

            response.joint_state = request.joint_state
            response.pose = Pose()
            response.pose.position.x = float(pos[0])
            response.pose.position.y = float(pos[1])
            response.pose.position.z = float(pos[2])
            response.pose.orientation.w = float(quat[0])
            response.pose.orientation.x = float(quat[1])
            response.pose.orientation.y = float(quat[2])
            response.pose.orientation.z = float(quat[3])
            response.success = True

            self.get_logger().info(f'FK result: pos={[round(v, 3) for v in pos]}, quat={[round(v, 3) for v in quat]}')

        except Exception as e:
            response.success = False
            self.get_logger().error(f'FK error: {e}')

        return response

    def _to_joint_trajectory(self, js_result, speed_scale: float) -> JointTrajectory:
        """将 CuRobo V2 JointState 结果转换为 ROS 2 JointTrajectory。"""
        traj = JointTrajectory()
        traj.joint_names = list(js_result.joint_names) if js_result.joint_names else self._joint_names

        positions = js_result.position
        raw_shape = list(positions.shape) if hasattr(positions, 'shape') else None
        if hasattr(positions, 'detach'):
            positions = positions.detach().cpu()
            if positions.ndim == 3 and positions.shape[0] == 1:
                positions = positions[0]
            elif positions.ndim > 2:
                positions = positions.reshape(-1, positions.shape[-1])
            positions = positions.tolist()
        elif hasattr(positions, 'shape') and len(positions.shape) > 1:
            positions = positions[0] if positions.shape[0] == 1 else positions

        # [caohy] S5-B：打印 CuRobo 原始轨迹张量与转换后前后几帧摘要，
        # 用于定位 planner -> JointTrajectory 链路中是否在这里就被压成重复点。
        if positions:
            first_point = [round(float(v), 6) for v in positions[0]]
            last_point = [round(float(v), 6) for v in positions[-1]]
            sample_mid = [round(float(v), 6) for v in positions[len(positions) // 2]]
            self._log_probe_info(
                '[S5-B] _to_joint_trajectory summary: '
                f'raw_shape={raw_shape}, converted_len={len(positions)}, '
                f'first={first_point}, mid={sample_mid}, last={last_point}'
            )
        else:
            self.get_logger().warn(
                f'[S5-B] _to_joint_trajectory received empty positions, raw_shape={raw_shape}'
            )

        base_dt = 0.008
        interpolation_dt = base_dt / max(speed_scale, 0.01)

        for i, point_positions in enumerate(positions):
            pt = JointTrajectoryPoint()
            pt.positions = [float(p) for p in point_positions]

            total_ns = int(round(i * interpolation_dt * 1e9))
            pt.time_from_start = Duration(
                sec=total_ns // 1_000_000_000,
                nanosec=total_ns % 1_000_000_000,
            )
            traj.points.append(pt)

        return traj

    # [caohy] Phase 8：hold_vec_weight 映射逻辑。
    # V1 语义：hold_vec_weight = [rx, ry, rz, x, y, z]，1=约束，0=自由
    # V2 语义：ToolPoseCriteria.terminal_pose_axes_weight_factor = [x, y, z, rx, ry, rz]，权重越大约束越强
    # 映射规则：
    #   1. 重排：[rx, ry, rz, x, y, z] → [x, y, z, rx, ry, rz]
    #   2. 不反转：1.0（V1 约束强）→ 高权重（V2 约束强）
    #   3. 任何出现"输入越大约束越弱"的现象都视为严重错误

    def _apply_hold_vec_weight(self, hold_vec_weight):
        """应用 hold_vec_weight 到 planner，返回旧 criteria 用于恢复。

        Args:
            hold_vec_weight: V1 语义 [rx, ry, rz, x, y, z]，1=约束，0=自由。

        Returns:
            dict: {tool_frame: old_criteria} 用于恢复，或 None。
        """
        if hold_vec_weight is None or len(hold_vec_weight) < 6:
            return None

        # 检查是否全为 0（无约束）
        if all(w == 0 for w in hold_vec_weight):
            return None

        # V1: [rx, ry, rz, x, y, z]
        v1_rx, v1_ry, v1_rz = hold_vec_weight[0], hold_vec_weight[1], hold_vec_weight[2]
        v1_x, v1_y, v1_z = hold_vec_weight[3], hold_vec_weight[4], hold_vec_weight[5]

        # [caohy] Task 21 最小验证：严格约束链路下目标位置本身仍必须被跟踪，
        # 不能因为 hold_vec_weight 的平移位为 0 就把 xyz 目标跟踪权重清零。
        # 当前先临时对齐 rokae_motion_gen 的做法，固定保留 terminal xyz=1.0，
        # 只观察这一步是否会消除 planner.plan_pose() 的重复点退化候选。
        v2_x, v2_y, v2_z = 1.0, 1.0, 1.0
        v2_rx, v2_ry, v2_rz = float(v1_rx), float(v1_ry), float(v1_rz)

        terminal_weight = [v2_x, v2_y, v2_z, v2_rx, v2_ry, v2_rz]
        # [caohy] Task 36：Move 02 / Move 04 的 shadow sweep 证明 0.1 会放松到超过
        # 3 度水平容差；默认提高到 0.5，让 cuRobo 在中间轨迹也更重视姿态保持，
        # 同时仍不对中间帧位置轴施加额外约束。
        non_terminal_weight = [0.0, 0.0, 0.0, v2_rx * 0.5, v2_ry * 0.5, v2_rz * 0.5]

        criteria = ToolPoseCriteria(
            terminal_pose_axes_weight_factor=terminal_weight,
            non_terminal_pose_axes_weight_factor=non_terminal_weight,
        )

        tool_frame = self._tool_frames[0]
        old_criteria = {tool_frame: criteria}
        # [caohy] Task 36：记录当前严格约束链路正在使用的 ToolPoseCriteria，
        # 后续 shadow sweep 临时改非终点姿态权重后需要恢复到这份正式配置。
        self._active_tool_pose_criteria = {tool_frame: criteria}
        self._active_terminal_pose_axes_weight_factor = list(terminal_weight)
        self._planner.update_tool_pose_criteria({tool_frame: criteria})

        self.get_logger().info(
            f'hold_vec_weight applied: V1={hold_vec_weight} -> V2 terminal={terminal_weight}'
        )

        return old_criteria

    def _restore_criteria(self, old_criteria):
        """恢复旧的 tool_pose_criteria。"""
        if old_criteria is None:
            return
        default_criteria = ToolPoseCriteria()
        for frame in self._tool_frames:
            self._planner.update_tool_pose_criteria({frame: default_criteria})
        self._active_tool_pose_criteria = None
        self._active_terminal_pose_axes_weight_factor = None

    # [caohy] Phase 9.7：多候选求解。V2 plan_pose 无 num_trajopt_seeds，通过多次调用收集候选。
    def _collect_candidates(
        self,
        start_joint,
        target_pose,
        num_candidates=4,
        max_attempts=5,
        source_label='planner',
        generation_mode='top1_x4',
    ):
        """多次调用 plan_pose 收集候选轨迹，并返回每次尝试的生命周期摘要。

        Args:
            start_joint: 起始关节角。
            target_pose: 目标位姿 [x,y,z,qw,qx,qy,qz]。
            num_candidates: 候选数量。
            max_attempts: 每次规划的最大尝试次数。

        Returns:
            tuple[list, list]: (候选轨迹列表, 尝试详情列表)。
        """
        pos = target_pose[:3]
        quat = target_pose[3:7]

        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=torch.tensor([[[[[pos[0], pos[1], pos[2]]]]]], device='cuda:0', dtype=torch.float32),
            quaternion=torch.tensor([[[[[quat[0], quat[1], quat[2], quat[3]]]]]], device='cuda:0', dtype=torch.float32),
        )

        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )

        candidates = []
        attempt_records = []
        for i in range(num_candidates):
            attempt_record = {
                'attempt_index': int(i),
                'candidate_rank': int(i + 1),
                'source_label': str(source_label),
                'generation_mode': str(generation_mode),
                'max_attempts': int(max_attempts),
                'enable_graph_attempt': int(max(1, 3 - i)),
                'accepted_to_pool': False,
                'candidate_pool_accepted': False,
                'final_selected': False,
            }
            try:
                result = self._planner.plan_pose(
                    goal, current_state,
                    max_attempts=max_attempts,
                    enable_graph_attempt=max(1, 3 - i),
                )
                attempt_record['result_summary'] = self._build_result_debug_payload(result)
                if self._result_success(result):
                    self._log_plan_result_summary(f'_collect_candidates.plan_pose[{i}]', result)
                    js_result = result.get_interpolated_plan()
                    pos_tensor = js_result.position
                    raw_shape = list(pos_tensor.shape) if hasattr(pos_tensor, 'shape') else None
                    if hasattr(pos_tensor, 'detach'):
                        pos_tensor = pos_tensor.detach().cpu()
                    raw_first = None
                    raw_last = None
                    if hasattr(pos_tensor, 'reshape') and getattr(pos_tensor, 'numel', lambda: 0)() > 0:
                        raw_flat = pos_tensor.reshape(-1, pos_tensor.shape[-1])
                        raw_first = [round(float(v), 6) for v in raw_flat[0].tolist()]
                        raw_last = [round(float(v), 6) for v in raw_flat[-1].tolist()]
                    # 展平所有前导 batch 维度，保留 [T, DOF]
                    while pos_tensor.ndim > 2:
                        if pos_tensor.shape[0] == 1:
                            pos_tensor = pos_tensor.squeeze(0)
                        else:
                            pos_tensor = pos_tensor.reshape(-1, pos_tensor.shape[-1])
                    flat_first = [round(float(v), 6) for v in pos_tensor[0].tolist()]
                    flat_last = [round(float(v), 6) for v in pos_tensor[-1].tolist()]
                    candidates.append(pos_tensor)
                    attempt_record['accepted_to_pool'] = True
                    attempt_record['candidate_pool_accepted'] = True
                    self._populate_planner_attempt_trajectory_fields(
                        attempt_record,
                        pos_tensor,
                        target_pose,
                    )
                    self._log_probe_info(
                        f'[S5-B] Candidate {i}: raw_shape={raw_shape}, '
                        f'flat_shape={list(pos_tensor.shape)}, raw_first={raw_first}, raw_last={raw_last}, '
                        f'flat_first={flat_first}, flat_last={flat_last}'
                    )
                else:
                    attempt_record['failure_reason'] = self._result_status(result)
                    self.get_logger().info(f'Candidate {i}: failed ({self._result_status(result)})')
            except Exception as e:
                attempt_record['failure_reason'] = str(e)
                attempt_record['exception_type'] = type(e).__name__
                self.get_logger().warn(f'Candidate {i}: error ({e})')
            attempt_records.append(self._round_nested_debug_value(attempt_record, float_digits=6))

        return candidates, attempt_records

    def _collect_planner_topk_main_candidates(
        self,
        start_joint,
        target_pose,
        alignment_tolerance_deg: float,
        num_candidates: int = 4,
    ):
        """[caohy] Task 11：正式主链路改走 1 constant + N-1 auto goal IK seed 的 topk 求解。"""
        total_outputs = max(1, min(4, int(num_candidates)))
        goal = self._make_goal_tool_pose(target_pose)
        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )

        candidates = []
        attempt_records = []
        try:
            explicit_seed_inputs = self._build_planner_topk_explicit_seed_inputs(
                start_joint,
                target_pose,
                requested_seed_count=int(total_outputs),
            )
            effective_num_seeds = int(total_outputs)
            solve_kwargs = {}
            if explicit_seed_inputs.get('success'):
                solve_kwargs['seed_config'] = explicit_seed_inputs.get('seed_config')
                solve_kwargs['seed_traj'] = explicit_seed_inputs.get('seed_traj')
                effective_num_seeds = int(
                    explicit_seed_inputs.get('selected_goal_seed_count') or total_outputs
                )
            result = self._planner.trajopt_solver.solve_pose(
                goal,
                current_state,
                use_implicit_goal=True,
                return_seeds=int(total_outputs),
                num_seeds=int(effective_num_seeds),
                finetune_attempts=1,
                **solve_kwargs,
            )
            result_summary = self._build_result_debug_payload(result)
            seed_input_mode = (
                'mixed_constant_plus_auto_goal_ik_seeded'
                if explicit_seed_inputs.get('success')
                else 'constant_seed_only_fallback'
            )
            if self._result_success(result):
                self._log_plan_result_summary(
                    '_collect_planner_topk_main_candidates.solve_pose',
                    result,
                )
                js_solution_candidates = self._extract_result_js_solution_candidates(result)
                interpolated_candidates = self._extract_result_interpolated_candidates(result)
                output_candidates = (
                    js_solution_candidates
                    if len(js_solution_candidates) > 0
                    else interpolated_candidates
                )
                candidate_extract_mode = (
                    'js_solution'
                    if len(js_solution_candidates) > 0
                    else ('interpolated_plan' if len(interpolated_candidates) > 0 else 'none')
                )
                for attempt_index in range(total_outputs):
                    attempt_record = {
                        'attempt_index': int(attempt_index),
                        'candidate_rank': int(attempt_index + 1),
                        'source_label': 'planner',
                        'generation_mode': 'top4_mixed_main',
                        'seed_input_mode': seed_input_mode,
                        'accepted_to_pool': False,
                        'candidate_pool_accepted': False,
                        'final_selected': False,
                        'success': False,
                        'status': 'missing_output_candidate',
                        'failure_reason': None,
                        'result_summary': result_summary,
                    }
                    if attempt_index < len(output_candidates):
                        candidate_traj = output_candidates[attempt_index]
                        attempt_record['success'] = True
                        attempt_record['status'] = 'success'
                        attempt_record['candidate_extract_mode'] = candidate_extract_mode
                        attempt_record['accepted_to_pool'] = True
                        attempt_record['candidate_pool_accepted'] = True
                        self._populate_planner_attempt_trajectory_fields(
                            attempt_record,
                            candidate_traj,
                            target_pose,
                            alignment_tolerance_deg=alignment_tolerance_deg,
                            start_joint=start_joint,
                        )
                        candidates.append(candidate_traj)
                    else:
                        attempt_record['failure_reason'] = self._result_status(result)
                    attempt_records.append(
                        self._round_nested_debug_value(attempt_record, float_digits=6)
                    )
            else:
                failure_reason = self._result_status(result)
                for attempt_index in range(total_outputs):
                    attempt_records.append(
                        self._round_nested_debug_value(
                            {
                                'attempt_index': int(attempt_index),
                                'candidate_rank': int(attempt_index + 1),
                                'source_label': 'planner',
                                'generation_mode': 'top4_mixed_main',
                                'seed_input_mode': seed_input_mode,
                                'accepted_to_pool': False,
                                'candidate_pool_accepted': False,
                                'final_selected': False,
                                'success': False,
                                'status': 'solver_failed',
                                'failure_reason': failure_reason,
                                'result_summary': result_summary,
                            },
                            float_digits=6,
                        )
                    )
        except Exception as exc:
            for attempt_index in range(total_outputs):
                attempt_records.append(
                    {
                        'attempt_index': int(attempt_index),
                        'candidate_rank': int(attempt_index + 1),
                        'source_label': 'planner',
                        'generation_mode': 'top4_mixed_main',
                        'accepted_to_pool': False,
                        'candidate_pool_accepted': False,
                        'final_selected': False,
                        'success': False,
                        'status': 'solver_exception',
                        'failure_reason': str(exc),
                        'exception_type': type(exc).__name__,
                    }
                )

        return candidates, attempt_records

    def _populate_planner_attempt_trajectory_fields(
        self,
        attempt_record: dict,
        trajectory: torch.Tensor,
        target_pose,
        alignment_tolerance_deg: Optional[float] = None,
        start_joint=None,
    ) -> None:
        """[caohy] Task 11：给 planner 候选补统一轨迹摘要和终点目标误差字段。"""
        if trajectory is None:
            return
        traj_cpu = trajectory.detach().cpu() if hasattr(trajectory, 'detach') else trajectory
        trajectory_points = self._trajectory_tensor_to_list(traj_cpu)
        attempt_record['trajectory_points'] = trajectory_points
        attempt_record['output_trajectory'] = trajectory_points
        attempt_record['trajectory_summary'] = self._summarize_trajectory_points(trajectory_points)
        attempt_record['output_summary'] = attempt_record['trajectory_summary']
        try:
            terminal_tensor = traj_cpu[-1:].to(device='cuda:0', dtype=torch.float32)
            terminal_pose_summary = self._summarize_pose_against_goal(
                terminal_tensor,
                torch.tensor(target_pose[:3], device='cuda:0', dtype=torch.float32),
                target_pose[3:7],
            )
            attempt_record['terminal_goal_pose_summary'] = (
                terminal_pose_summary[0] if terminal_pose_summary else None
            )
        except Exception as exc:
            attempt_record['terminal_goal_pose_summary_error'] = str(exc)
        if alignment_tolerance_deg is not None:
            try:
                metrics = self._summarize_single_candidate_selection_metrics(
                    traj_cpu,
                    start_joint,
                    target_pose,
                    alignment_tolerance_deg,
                )
                attempt_record['selection_metrics'] = metrics
                attempt_record['stage_alignment_probe'] = {
                    'probe_label': str(attempt_record.get('source_label') or 'planner'),
                    'candidate_metrics': metrics,
                }
            except Exception as exc:
                attempt_record['selection_metrics_error'] = str(exc)

    def _summarize_single_candidate_selection_metrics(
        self,
        trajectory: torch.Tensor,
        start_joint,
        target_pose,
        alignment_tolerance_deg: float,
    ) -> dict:
        """[caohy] Task 11：按 selector（筛选器）口径补单条候选的对齐/连续性指标。"""
        if start_joint is None:
            raise ValueError('start_joint_missing')
        traj = trajectory.detach().clone().to(device='cuda:0', dtype=torch.float32)
        while traj.ndim > 2:
            if traj.shape[0] == 1:
                traj = traj.squeeze(0)
            else:
                traj = traj.reshape(-1, traj.shape[-1])
        if traj.ndim == 1:
            traj = traj.unsqueeze(0)
        positions = traj.unsqueeze(0)
        y_tool = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:0')
        z_neg = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda:0')
        level_eval = constraint_utils.evaluate_axis_alignment_batched(
            positions,
            self._constraint_eval_kinematics_fn,
            alignment_tolerance_deg,
            y_tool,
            z_neg,
        )
        continuity = constraint_utils.compute_candidate_continuity_metrics(
            positions,
            start_joint,
            target_pose[3:7],
            self._constraint_eval_kinematics_fn,
        )
        return self._round_nested_debug_value(
            {
                'alignment_valid': bool(level_eval['alignment_valid'][0].item()),
                'max_alignment_deviation_deg': float(
                    level_eval['max_alignment_deviation'][0].item()
                ),
                'mean_alignment_deviation_deg': float(
                    level_eval['mean_alignment_deviation'][0].item()
                ),
                'start_joint_gap_l2': float(continuity['start_joint_gap_l2'][0].item()),
                'joint_step_jump_cost': float(continuity['joint_step_jump_cost'][0].item()),
                'joint_step_max_l2': float(continuity['joint_step_max_l2'][0].item()),
                'joint_step_max_abs': float(continuity['joint_step_max_abs'][0].item()),
                'twist_smoothness_cost': float(continuity['twist_smoothness_cost'][0].item()),
                'alignment_profile': self._build_alignment_profile_debug(
                    traj,
                    alignment_tolerance_deg,
                ),
            },
            float_digits=6,
        )

    def _collect_planner_topk_shadow_candidates(
        self,
        start_joint,
        target_pose,
        alignment_tolerance_deg: float,
        return_seeds: int = 4,
        num_seeds: Optional[int] = None,
    ) -> dict:
        """[caohy] Task 11：一次 trajopt_solver.solve_pose(return_seeds=K) 的只读 topk 实验。"""
        total_outputs = max(1, int(return_seeds))
        generation_mode = f'top{int(total_outputs)}_mixed_shadow'
        bundle_record = {
            'branch_mode': self._get_planner_topk_experiment_mode(),
            'source_label': 'planner_topk_shadow',
            'generation_mode': generation_mode,
            'success': False,
            'status': 'pending',
            'failure_reason': None,
            'return_seeds': int(total_outputs),
            'num_seeds': int(num_seeds) if num_seeds is not None else int(total_outputs),
            'candidate_pool_accepted': False,
            'final_selected': False,
            'seed_input_mode': 'constant_seed_only_fallback',
            'attempts': [],
        }
        goal = self._make_goal_tool_pose(target_pose)
        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        try:
            explicit_seed_inputs = self._build_planner_topk_explicit_seed_inputs(
                start_joint,
                target_pose,
                requested_seed_count=int(total_outputs),
            )
            bundle_record['explicit_seed_prepare'] = self._round_nested_debug_value(
                {
                    k: v
                    for k, v in explicit_seed_inputs.items()
                    if k not in ('seed_config', 'seed_traj')
                },
                float_digits=6,
            )
            effective_num_seeds = (
                int(num_seeds) if num_seeds is not None else int(total_outputs)
            )
            solve_kwargs = {}
            if explicit_seed_inputs.get('success'):
                solve_kwargs['seed_config'] = explicit_seed_inputs.get('seed_config')
                solve_kwargs['seed_traj'] = explicit_seed_inputs.get('seed_traj')
                effective_num_seeds = int(
                    explicit_seed_inputs.get('selected_goal_seed_count') or total_outputs
                )
                bundle_record['seed_input_mode'] = 'mixed_constant_plus_auto_goal_ik_seeded'
            result = self._planner.trajopt_solver.solve_pose(
                goal,
                current_state,
                use_implicit_goal=True,
                return_seeds=int(total_outputs),
                num_seeds=int(effective_num_seeds),
                finetune_attempts=1,
                **solve_kwargs,
            )
            bundle_record['result_summary'] = self._build_result_debug_payload(result)
            js_solution_candidates = self._extract_result_js_solution_candidates(result)
            interpolated_candidates = self._extract_result_interpolated_candidates(result)
            output_candidates = (
                js_solution_candidates if len(js_solution_candidates) > 0 else interpolated_candidates
            )
            bundle_record['candidate_extract_mode'] = (
                'js_solution'
                if len(js_solution_candidates) > 0 else (
                    'interpolated_plan' if len(interpolated_candidates) > 0 else 'none'
                )
            )
            bundle_record['js_solution_candidate_count'] = int(len(js_solution_candidates))
            bundle_record['interpolated_candidate_count'] = int(len(interpolated_candidates))
            bundle_record['output_candidate_count'] = int(len(output_candidates))
            bundle_record['success'] = bool(self._result_success(result))
            bundle_record['status'] = (
                'solver_success' if self._result_success(result) else 'solver_failed'
            )
            bundle_record['failure_reason'] = (
                None if self._result_success(result) else self._result_status(result)
            )
            for attempt_index in range(total_outputs):
                source_label = f'planner_top{int(total_outputs)}_seed_{attempt_index + 1}'
                attempt_record = {
                    'attempt_index': int(attempt_index),
                    'candidate_rank': int(attempt_index + 1),
                    'source_label': source_label,
                    'generation_mode': generation_mode,
                    'seed_input_mode': bundle_record.get('seed_input_mode'),
                    'candidate_pool_accepted': False,
                    'final_selected': False,
                    'success': False,
                    'status': 'missing_output_candidate',
                    'failure_reason': None,
                    'result_summary': bundle_record.get('result_summary'),
                }
                if attempt_index < len(output_candidates):
                    candidate_traj = output_candidates[attempt_index]
                    attempt_record['success'] = True
                    attempt_record['status'] = 'success'
                    attempt_record['candidate_extract_mode'] = bundle_record['candidate_extract_mode']
                    self._populate_planner_attempt_trajectory_fields(
                        attempt_record,
                        candidate_traj,
                        target_pose,
                        alignment_tolerance_deg=alignment_tolerance_deg,
                        start_joint=start_joint,
                    )
                else:
                    attempt_record['failure_reason'] = (
                        None if self._result_success(result) else self._result_status(result)
                    )
                bundle_record['attempts'].append(
                    self._round_nested_debug_value(attempt_record, float_digits=6)
                )
        except Exception as exc:
            bundle_record['success'] = False
            bundle_record['status'] = 'solver_exception'
            bundle_record['failure_reason'] = str(exc)
            bundle_record['exception_type'] = type(exc).__name__
            for attempt_index in range(total_outputs):
                bundle_record['attempts'].append(
                    {
                        'attempt_index': int(attempt_index),
                        'candidate_rank': int(attempt_index + 1),
                        'source_label': f'planner_top{int(total_outputs)}_seed_{attempt_index + 1}',
                        'generation_mode': generation_mode,
                        'candidate_pool_accepted': False,
                        'final_selected': False,
                        'success': False,
                        'status': 'solver_exception',
                        'failure_reason': str(exc),
                        'exception_type': type(exc).__name__,
                    }
                )
        return self._round_nested_debug_value(bundle_record, float_digits=6)

    def _make_goal_tool_pose(self, target_pose):
        """构造单目标 GoalToolPose。"""
        pos = target_pose[:3]
        quat = target_pose[3:7]
        return GoalToolPose(
            tool_frames=self._tool_frames,
            position=torch.tensor([[[[[pos[0], pos[1], pos[2]]]]]], device='cuda:0', dtype=torch.float32),
            quaternion=torch.tensor([[[[[quat[0], quat[1], quat[2], quat[3]]]]]], device='cuda:0', dtype=torch.float32),
        )

    def _prepare_seed_traj_for_trajopt(self, seed_traj: torch.Tensor) -> torch.Tensor:
        """将 alignment seed 轨迹适配为 trajopt_solver.solve_pose() 可消费的 seed_traj 形状。

        Args:
            seed_traj: [T, DOF]，通常来自 _generate_alignment_seed()。

        Returns:
            torch.Tensor: [1, 1, action_horizon, DOF]，可直接传给 trajopt_solver.solve_pose()。
        """
        if seed_traj is None:
            raise ValueError('seed_traj is None')

        if hasattr(seed_traj, 'detach'):
            seed_traj = seed_traj.detach()
        if seed_traj.ndim != 2:
            raise ValueError(f'seed_traj must be 2D [T, DOF], got shape={list(seed_traj.shape)}')

        seed_traj = seed_traj.to(device='cuda:0', dtype=torch.float32)
        target_horizon = int(self._planner.trajopt_solver.action_horizon)
        if seed_traj.shape[0] == target_horizon:
            return seed_traj.unsqueeze(0).unsqueeze(0)

        seed_traj_resampled = self._resample_seed_traj_linear(seed_traj, target_horizon)
        return seed_traj_resampled.unsqueeze(0).unsqueeze(0)

    def _compute_tool_pose_for_joint_trajectory(
        self,
        joint_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """[caohy] Task 10：对 [T, DOF] 关节轨迹批量做 FK，提取 tool0 位姿序列。"""
        if joint_trajectory is None:
            raise ValueError('joint_trajectory is None')
        if joint_trajectory.ndim != 2:
            raise ValueError(
                f'joint_trajectory must be 2D [T, DOF], got shape={list(joint_trajectory.shape)}'
            )
        state = CuJointState.from_position(
            joint_trajectory.detach().clone().to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        fk_position = tool_pose.position.reshape(-1, 3)
        fk_quaternion = tool_pose.quaternion.reshape(-1, 4)
        return fk_position, fk_quaternion

    def _make_horizon_goal_tool_pose(
        self,
        target_pose,
        prepared_seed_flat: torch.Tensor,
    ):
        """[caohy] Task 10：基于 prepared seed 姿态序列构造 horizon>1 GoalToolPose。"""
        if prepared_seed_flat is None:
            raise ValueError('prepared_seed_flat is None')
        if prepared_seed_flat.ndim != 2:
            raise ValueError(
                f'prepared_seed_flat must be 2D [T, DOF], got shape={list(prepared_seed_flat.shape)}'
            )
        goal_position = torch.tensor(
            list(target_pose[:3]),
            device='cuda:0',
            dtype=torch.float32,
        ).view(1, 3)
        fk_position, fk_quaternion = self._compute_tool_pose_for_joint_trajectory(prepared_seed_flat)
        horizon = int(prepared_seed_flat.shape[0])
        goal_position_seq = goal_position.expand(horizon, -1).contiguous()
        goal_quaternion_seq = fk_quaternion.detach().clone().to(device='cuda:0', dtype=torch.float32)
        # [caohy] Task 10：当前这版 cuRobo V2 的 ToolPose cost 消费链路仍在
        # cost_tool_pose.py 里对 goal_tool_poses.position / quaternion 做 squeeze(1)，
        # 实际稳定支持的是 horizon=1。这里先把“时序姿态序列”编码到 goalset 维：
        # [B=1, H=1, L=1, G=T, 3/4]，保证 shadow 分支能先跑通真实 solve_pose。
        # 这条路线当前语义是“序列姿态候选集合”，不是严格逐时间步一一跟踪。
        goal = GoalToolPose(
            tool_frames=self._tool_frames,
            position=goal_position_seq.view(1, 1, 1, horizon, 3),
            quaternion=goal_quaternion_seq.view(1, 1, 1, horizon, 4),
        )
        summary = {
            'goal_horizon': 1,
            'requested_sequence_length': horizon,
            'goalset_size': horizon,
            'tool_frame': str(self._tool_frames[0]) if self._tool_frames else None,
            'goal_position_mode': 'repeat_terminal_position',
            'goal_quaternion_mode': 'follow_prepared_seed_fk_sequence',
            'goal_encoding_mode': 'goalset_fallback_due_pose_cost_horizon1_only',
            'prepared_seed_fk_summary': {
                'point_count': int(fk_position.shape[0]),
                'first_position': self._round_nested_debug_value(
                    fk_position[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_position': self._round_nested_debug_value(
                    fk_position[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'first_quaternion': self._round_nested_debug_value(
                    goal_quaternion_seq[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_quaternion': self._round_nested_debug_value(
                    goal_quaternion_seq[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
            },
            'goal_position_summary': {
                'first_position': self._round_nested_debug_value(
                    goal_position_seq[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_position': self._round_nested_debug_value(
                    goal_position_seq[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
            },
        }
        return goal, self._round_nested_debug_value(summary, float_digits=6)

    def _build_alignment_seed_sequence_goal(
        self,
        target_pose,
        raw_seed_traj: torch.Tensor,
    ):
        """[caohy] Task 10：基于 alignment_seed.raw_trajectory 构造 sequence 目标摘要。"""
        if raw_seed_traj is None:
            raise ValueError('raw_seed_traj is None')
        if raw_seed_traj.ndim != 2:
            raise ValueError(
                f'raw_seed_traj must be 2D [T, DOF], got shape={list(raw_seed_traj.shape)}'
            )

        fk_position, fk_quaternion = self._compute_tool_pose_for_joint_trajectory(raw_seed_traj)
        sequence_length = int(raw_seed_traj.shape[0])
        goal_position = torch.tensor(
            list(target_pose[:3]),
            device='cuda:0',
            dtype=torch.float32,
        ).view(1, 3)
        goal_position_seq = goal_position.expand(sequence_length, -1).contiguous()
        goal_quaternion_seq = fk_quaternion.detach().clone().to(device='cuda:0', dtype=torch.float32)
        sequence_goal = {
            'tool_frames': list(self._tool_frames),
            'position_sequence': goal_position_seq,
            'quaternion_sequence': goal_quaternion_seq,
            'sequence_length': sequence_length,
        }
        summary = {
            'source_label': 'alignment_seed_sequence',
            'tool_frame': str(self._tool_frames[0]) if self._tool_frames else None,
            'sequence_length': sequence_length,
            'goal_position_mode': 'repeat_terminal_position',
            'goal_quaternion_mode': 'follow_raw_seed_fk_sequence',
            'raw_seed_summary': self._summarize_trajectory_points(
                self._trajectory_tensor_to_list(raw_seed_traj),
            ),
            'fk_pose_summary': {
                'point_count': int(fk_position.shape[0]),
                'first_position': self._round_nested_debug_value(
                    fk_position[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_position': self._round_nested_debug_value(
                    fk_position[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'first_quaternion': self._round_nested_debug_value(
                    goal_quaternion_seq[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_quaternion': self._round_nested_debug_value(
                    goal_quaternion_seq[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
            },
            'goal_position_summary': {
                'first_position': self._round_nested_debug_value(
                    goal_position_seq[0].detach().cpu().tolist(),
                    float_digits=6,
                ),
                'last_position': self._round_nested_debug_value(
                    goal_position_seq[-1].detach().cpu().tolist(),
                    float_digits=6,
                ),
            },
        }
        return sequence_goal, self._round_nested_debug_value(summary, float_digits=6)

    def _get_active_sequence_tool_pose_criteria(self) -> dict:
        """[caohy] Task 10：取当前严格约束链路正在生效的 ToolPoseCriteria。"""
        if self._active_tool_pose_criteria:
            return self._active_tool_pose_criteria
        return {self._tool_frames[0]: ToolPoseCriteria()}

    def _build_sequence_retargeter(
        self,
        use_mpc: bool = False,
        optimization_dt: float = 0.05,
        num_seeds_global: int = 64,
        global_ik_num_iters: Optional[int] = None,
        local_ik_num_iters: Optional[int] = None,
        velocity_regularization_weight: Optional[float] = None,
        acceleration_regularization_weight: Optional[float] = None,
        attempt_profile_name: Optional[str] = None,
    ):
        """[caohy] Task 10：按当前 planner 运行态严格构建 MotionRetargeter。"""
        tool_pose_criteria = self._get_active_sequence_tool_pose_criteria()
        summary = {
            'source_label': 'alignment_seed_sequence',
            'build_success': False,
            'failure_reason': None,
            'use_mpc': bool(use_mpc),
            'optimization_dt': round(float(optimization_dt), 6),
            'attempt_profile_name': attempt_profile_name,
            'inherits_runtime_from_planner': True,
            'robot_inherited': self._robot_cfg is not None,
            'scene_world_inherited': self._world_scene is not None,
            'scene_world_summary': self._world_summary,
            'self_collision_check': True,
            'collision_cache': self._collision_cache,
            'tool_frames': list(self._tool_frames),
            'num_seeds_global': int(num_seeds_global),
            'global_ik_num_iters': global_ik_num_iters,
            'local_ik_num_iters': local_ik_num_iters,
            'velocity_regularization_weight': velocity_regularization_weight,
            'acceleration_regularization_weight': acceleration_regularization_weight,
            'tool_pose_criteria': self._round_nested_debug_value(
                tool_pose_criteria,
                float_digits=6,
            ),
        }
        try:
            from curobo.motion_retargeter import MotionRetargeter, MotionRetargeterCfg
        except Exception:
            from curobo._src.motion.motion_retargeter import MotionRetargeter
            from curobo._src.motion.motion_retargeter_cfg import MotionRetargeterCfg

        retargeter = None
        try:
            retargeter_cfg = MotionRetargeterCfg.create(
                robot=self._robot_cfg,
                tool_pose_criteria=tool_pose_criteria,
                num_envs=1,
                use_mpc=bool(use_mpc),
                self_collision_check=True,
                scene_model=self._world_scene,
                optimization_dt=float(optimization_dt),
                num_seeds_global=int(num_seeds_global),
                load_collision_spheres=True,
                global_ik_num_iters=global_ik_num_iters,
                local_ik_num_iters=local_ik_num_iters,
                velocity_regularization_weight=velocity_regularization_weight,
                acceleration_regularization_weight=acceleration_regularization_weight,
            )
            retargeter = MotionRetargeter(retargeter_cfg)
            summary['build_success'] = True
            summary['retargeter_tool_frames'] = list(getattr(retargeter, 'tool_frames', []))
            summary['retargeter_joint_names'] = list(getattr(retargeter, 'joint_names', []))
            summary['retargeter_action_dim'] = int(getattr(retargeter, 'action_dim', 0) or 0)
            summary['retargeter_use_mpc'] = bool(
                getattr(getattr(retargeter, 'config', None), 'use_mpc', False)
            )
        except Exception as exc:
            summary['build_success'] = False
            summary['failure_reason'] = str(exc)
            summary['exception_type'] = type(exc).__name__
        return retargeter, self._round_nested_debug_value(summary, float_digits=6)

    def _get_alignment_seed_sequence_attempt_profiles(self) -> list[dict]:
        """[caohy] Task 10：定义 alignment_seed_sequence_1..5 的固定 attempt profile。"""
        return [
            {
                'attempt_profile_name': 'baseline_dt50_seed64',
                'optimization_dt': 0.05,
                'num_seeds_global': 64,
                'global_ik_num_iters': 200,
                'local_ik_num_iters': 120,
                'velocity_regularization_weight': None,
                'acceleration_regularization_weight': None,
            },
            {
                'attempt_profile_name': 'velocity_bias_dt50_seed64',
                'optimization_dt': 0.05,
                'num_seeds_global': 64,
                'global_ik_num_iters': 220,
                'local_ik_num_iters': 140,
                'velocity_regularization_weight': 0.003,
                'acceleration_regularization_weight': None,
            },
            {
                'attempt_profile_name': 'accel_bias_dt50_seed64',
                'optimization_dt': 0.05,
                'num_seeds_global': 64,
                'global_ik_num_iters': 220,
                'local_ik_num_iters': 140,
                'velocity_regularization_weight': 0.003,
                'acceleration_regularization_weight': 0.02,
            },
            {
                'attempt_profile_name': 'finer_dt40_seed96',
                'optimization_dt': 0.04,
                'num_seeds_global': 96,
                'global_ik_num_iters': 260,
                'local_ik_num_iters': 160,
                'velocity_regularization_weight': 0.004,
                'acceleration_regularization_weight': 0.02,
            },
            {
                'attempt_profile_name': 'coarser_dt60_seed128',
                'optimization_dt': 0.06,
                'num_seeds_global': 128,
                'global_ik_num_iters': 300,
                'local_ik_num_iters': 180,
                'velocity_regularization_weight': 0.002,
                'acceleration_regularization_weight': 0.01,
            },
        ]

    def _make_sequence_goal_tool_pose(self, sequence_goal: dict):
        """[caohy] Task 10：把内部 sequence_goal 摘要转成 SequenceGoalToolPose。"""
        try:
            from curobo._src.types.sequence_tool_pose import SequenceGoalToolPose
        except Exception:
            from curobo.types import SequenceGoalToolPose

        position_sequence = sequence_goal['position_sequence']
        quaternion_sequence = sequence_goal['quaternion_sequence']
        return SequenceGoalToolPose(
            tool_frames=list(sequence_goal['tool_frames']),
            position=position_sequence.view(position_sequence.shape[0], 1, len(sequence_goal['tool_frames']), 1, 3),
            quaternion=quaternion_sequence.view(quaternion_sequence.shape[0], 1, len(sequence_goal['tool_frames']), 1, 4),
        )

    def _make_sequence_frame_goal_tool_pose(self, sequence_goal: dict, frame_index: int):
        """[caohy] Task 10：把 sequence 目标中的单帧转成 GoalToolPose，供可控逐帧 IK 使用。"""
        tool_frames = list(sequence_goal['tool_frames'])
        position_sequence = sequence_goal['position_sequence']
        quaternion_sequence = sequence_goal['quaternion_sequence']
        frame_index = int(frame_index)
        if frame_index < 0 or frame_index >= int(position_sequence.shape[0]):
            raise IndexError(
                f'frame_index out of range: {frame_index}, sequence_length={int(position_sequence.shape[0])}'
            )
        frame_position = position_sequence[frame_index].detach().clone().to(
            device='cuda:0', dtype=torch.float32,
        )
        frame_quaternion = quaternion_sequence[frame_index].detach().clone().to(
            device='cuda:0', dtype=torch.float32,
        )
        return GoalToolPose(
            tool_frames=tool_frames,
            position=frame_position.view(1, 1, len(tool_frames), 1, 3),
            quaternion=frame_quaternion.view(1, 1, len(tool_frames), 1, 4),
        )

    def _extract_wrapped_sequence_ik_solution(
        self,
        ik_result,
        prev_solution: torch.Tensor,
        action_dim: int,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """[caohy] Task 10：从单帧 IK 结果里提取包角后的关节解和速度。"""
        if ik_result is None:
            raise ValueError('sequence_ik_result_is_none')
        feasible = getattr(ik_result, 'feasible', None)
        if feasible is None or not feasible.any():
            raise ValueError('sequence_ik_infeasible')

        batch_idx, seed_idx = feasible.nonzero(as_tuple=True)
        feasible_count = int(batch_idx.numel())
        raw_solution = ik_result.solution[batch_idx[0], seed_idx[0]].reshape(-1)
        wrapped_solution = self._wrap_seed_solution_to_prev(
            prev_solution.reshape(-1),
            raw_solution,
        ).detach().clone()

        velocity = None
        js_solution = getattr(ik_result, 'js_solution', None)
        if js_solution is not None and getattr(js_solution, 'velocity', None) is not None:
            velocity_tensor = js_solution.velocity.reshape(-1, action_dim)
            if velocity_tensor.shape[0] > 0:
                velocity = velocity_tensor[0].detach().clone()
        return wrapped_solution, velocity, feasible_count

    def _solve_alignment_seed_sequence_framewise(
        self,
        retargeter,
        sequence_goal: dict,
        start_joint,
        initial_seed_joint=None,
    ) -> dict:
        """[caohy] Task 10：首帧显式贴起点，后续帧逐帧 local IK 的可控 sequence 求解。"""
        if retargeter is None:
            raise ValueError('sequence_retargeter_missing')
        sequence_length = int(sequence_goal.get('sequence_length', 0))
        if sequence_length <= 0:
            raise ValueError(f'invalid_sequence_length={sequence_length}')

        start_joint_tensor = torch.as_tensor(
            start_joint, device='cuda:0', dtype=torch.float32,
        ).reshape(-1)
        if initial_seed_joint is None:
            initial_seed_tensor = start_joint_tensor.detach().clone()
            initial_seed_mode = 'start_joint'
        else:
            initial_seed_tensor = torch.as_tensor(
                initial_seed_joint, device='cuda:0', dtype=torch.float32,
            ).reshape(-1)
            initial_seed_mode = 'raw_seed_frame0'

        prev_solution = start_joint_tensor.detach().clone()
        prev_velocity = None
        solved_frames = []
        frame_debug = []
        failed_frame_index = None
        failure_reason = None

        for frame_index in range(sequence_length):
            goal_pose = self._make_sequence_frame_goal_tool_pose(sequence_goal, frame_index)
            goal_position = goal_pose.position.reshape(-1, 3)[0]
            goal_quaternion = goal_pose.quaternion.reshape(-1, 4)[0]
            current_state = CuJointState.from_position(
                prev_solution.view(1, -1), joint_names=self._joint_names,
            )
            seed_config = prev_solution.view(1, 1, -1)
            solver_kind = 'local_ik_warm_start'
            solver = getattr(retargeter, '_local_ik_solver', None)

            if frame_index == 0:
                solver_kind = 'global_ik_with_explicit_start'
                solver = getattr(retargeter, '_global_ik_solver', None)
                current_state = CuJointState.from_position(
                    start_joint_tensor.view(1, -1), joint_names=self._joint_names,
                )
                seed_config = initial_seed_tensor.view(1, 1, -1)
            elif prev_velocity is not None:
                current_state.velocity = prev_velocity.view(1, -1).detach().clone()

            if solver is None:
                failed_frame_index = int(frame_index)
                failure_reason = f'{solver_kind}_missing'
                frame_debug.append({
                    'frame_index': int(frame_index),
                    'solver_kind': solver_kind,
                    'success': False,
                    'failure_reason': failure_reason,
                })
                break

            try:
                if hasattr(solver, 'reset_seed'):
                    solver.reset_seed()
                ik_result = solver.solve_pose(
                    goal_tool_poses=goal_pose,
                    current_state=current_state,
                    seed_config=seed_config,
                    return_seeds=1,
                )
                solved_solution, solved_velocity, feasible_count = (
                    self._extract_wrapped_sequence_ik_solution(
                        ik_result,
                        prev_solution if frame_index > 0 else start_joint_tensor,
                        int(getattr(retargeter, '_action_dim', len(self._joint_names))),
                    )
                )
                start_gap_l2 = float(
                    torch.linalg.norm(solved_solution - start_joint_tensor).item()
                )
                prev_gap_l2 = float(
                    torch.linalg.norm(
                        solved_solution - (prev_solution if frame_index > 0 else start_joint_tensor)
                    ).item()
                )
                frame_debug.append(self._round_nested_debug_value({
                    'frame_index': int(frame_index),
                    'solver_kind': solver_kind,
                    'success': True,
                    'feasible_count': int(feasible_count),
                    'goal_position': goal_position.detach().cpu().tolist(),
                    'goal_quaternion': goal_quaternion.detach().cpu().tolist(),
                    'input_current_state': current_state.position.reshape(-1).detach().cpu().tolist(),
                    'input_seed_config': seed_config.reshape(-1).detach().cpu().tolist(),
                    'output_joint': solved_solution.detach().cpu().tolist(),
                    'start_joint_gap_l2': start_gap_l2,
                    'frame_to_prev_gap_l2': prev_gap_l2,
                }, float_digits=6))
                solved_frames.append(solved_solution.detach().clone())
                prev_solution = solved_solution.detach().clone()
                prev_velocity = solved_velocity.detach().clone() if solved_velocity is not None else None
            except Exception as exc:
                failed_frame_index = int(frame_index)
                failure_reason = str(exc)
                frame_debug.append(self._round_nested_debug_value({
                    'frame_index': int(frame_index),
                    'solver_kind': solver_kind,
                    'success': False,
                    'goal_position': goal_position.detach().cpu().tolist(),
                    'goal_quaternion': goal_quaternion.detach().cpu().tolist(),
                    'input_current_state': current_state.position.reshape(-1).detach().cpu().tolist(),
                    'input_seed_config': seed_config.reshape(-1).detach().cpu().tolist(),
                    'failure_reason': str(exc),
                    'exception_type': type(exc).__name__,
                }, float_digits=6))
                break

        if len(solved_frames) != sequence_length:
            return {
                'success': False,
                'failure_reason': failure_reason or 'sequence_framewise_failed',
                'failed_frame_index': failed_frame_index,
                'frame_debug': frame_debug,
                'summary': self._round_nested_debug_value({
                    'solver_strategy': 'explicit_start_then_local_ik',
                    'initial_seed_mode': initial_seed_mode,
                    'sequence_length': int(sequence_length),
                    'solved_frame_count': int(len(solved_frames)),
                    'failed_frame_index': failed_frame_index,
                    'failure_reason': failure_reason,
                }, float_digits=6),
            }

        output_trajectory = torch.stack(solved_frames, dim=0)
        frame_to_prev_gaps = []
        for frame_index in range(1, len(solved_frames)):
            frame_to_prev_gaps.append(
                float(torch.linalg.norm(solved_frames[frame_index] - solved_frames[frame_index - 1]).item())
            )
        return {
            'success': True,
            'trajectory': output_trajectory,
            'frame_debug': frame_debug,
            'summary': self._round_nested_debug_value({
                'solver_strategy': 'explicit_start_then_local_ik',
                'initial_seed_mode': initial_seed_mode,
                'sequence_length': int(sequence_length),
                'solved_frame_count': int(len(solved_frames)),
                'first_frame_start_joint_gap_l2': float(
                    torch.linalg.norm(solved_frames[0] - start_joint_tensor).item()
                ),
                'max_frame_to_prev_gap_l2': max(frame_to_prev_gaps) if frame_to_prev_gaps else 0.0,
                'mean_frame_to_prev_gap_l2': (
                    sum(frame_to_prev_gaps) / len(frame_to_prev_gaps) if frame_to_prev_gaps else 0.0
                ),
            }, float_digits=6),
        }

    def _extract_sequence_result_trajectory(self, result) -> torch.Tensor:
        """[caohy] Task 10：把 RetargetResult 统一规整成 [T, DOF] 轨迹。"""
        if result is None or getattr(result, 'joint_state', None) is None:
            raise ValueError('retarget_result_joint_state_missing')
        pos_tensor = getattr(result.joint_state, 'position', None)
        if pos_tensor is None:
            raise ValueError('retarget_result_position_missing')
        if hasattr(pos_tensor, 'detach'):
            pos_tensor = pos_tensor.detach().cpu()
        while pos_tensor.ndim > 2:
            if pos_tensor.shape[0] == 1:
                pos_tensor = pos_tensor.squeeze(0)
            else:
                pos_tensor = pos_tensor.reshape(-1, pos_tensor.shape[-1])
        if pos_tensor.ndim == 1:
            pos_tensor = pos_tensor.unsqueeze(0)
        return pos_tensor

    def _summarize_sequence_pose_error(
        self,
        output_trajectory: torch.Tensor,
        sequence_goal: dict,
    ) -> dict:
        """[caohy] Task 10：汇总 sequence 输出相对每帧目标的位姿误差。"""
        if output_trajectory.ndim != 2:
            raise ValueError(
                f'output_trajectory must be 2D [T, DOF], got shape={list(output_trajectory.shape)}'
            )
        joint_positions = output_trajectory.detach().clone().to(device='cuda:0', dtype=torch.float32)
        state = CuJointState.from_position(joint_positions, joint_names=self._joint_names)
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        fk_position = tool_pose.position.reshape(-1, 3)
        fk_quaternion = tool_pose.quaternion.reshape(-1, 4)

        goal_position = sequence_goal['position_sequence'].detach().clone().to(
            device='cuda:0', dtype=torch.float32
        )
        goal_quaternion = sequence_goal['quaternion_sequence'].detach().clone().to(
            device='cuda:0', dtype=torch.float32
        )
        frame_count = min(
            int(fk_position.shape[0]),
            int(goal_position.shape[0]),
            int(goal_quaternion.shape[0]),
        )
        fk_position = fk_position[:frame_count]
        fk_quaternion = fk_quaternion[:frame_count]
        goal_position = goal_position[:frame_count]
        goal_quaternion = goal_quaternion[:frame_count]

        goal_quaternion_conj = goal_quaternion.clone()
        goal_quaternion_conj[:, 1:] *= -1.0
        q_rel = constraint_utils.quaternion_multiply_batched(goal_quaternion_conj, fk_quaternion)
        rel_sign = torch.where(q_rel[:, :1] < 0.0, -1.0, 1.0)
        q_rel = q_rel * rel_sign
        position_error = torch.linalg.norm(fk_position - goal_position, dim=-1)
        orientation_error_deg = torch.rad2deg(
            2.0 * torch.acos(torch.clamp(q_rel[:, 0], min=-1.0, max=1.0))
        )
        worst_index = int(torch.argmax(position_error + torch.deg2rad(orientation_error_deg)).item())
        return self._round_nested_debug_value({
            'frame_count': frame_count,
            'position_error_m_max': float(torch.max(position_error).item()),
            'position_error_m_mean': float(torch.mean(position_error).item()),
            'orientation_error_deg_max': float(torch.max(orientation_error_deg).item()),
            'orientation_error_deg_mean': float(torch.mean(orientation_error_deg).item()),
            'worst_frame_index': worst_index,
            'worst_frame': {
                'fk_position': fk_position[worst_index].detach().cpu().tolist(),
                'goal_position': goal_position[worst_index].detach().cpu().tolist(),
                'position_error_m': float(position_error[worst_index].item()),
                'fk_quaternion': fk_quaternion[worst_index].detach().cpu().tolist(),
                'goal_quaternion': goal_quaternion[worst_index].detach().cpu().tolist(),
                'orientation_error_deg': float(orientation_error_deg[worst_index].item()),
            },
        }, float_digits=6)

    def _optimize_alignment_seed_sequence_candidates(
        self,
        start_joint,
        target_pose,
        raw_seed_traj,
        alignment_tolerance_deg=3.0,
        probe_label='alignment_seed_sequence',
    ):
        """[caohy] Task 10：按 5 条固定 profile 组织 sequence retarget attempt。"""
        profiles = self._get_alignment_seed_sequence_attempt_profiles()
        bundle_record = {
            'probe_label': str(probe_label),
            'source_label': str(probe_label),
            'success': False,
            'status': 'seed_missing',
            'failure_reason': None,
            'attempt_count': int(len(profiles)),
            'input_seed_trajectory': self._trajectory_tensor_to_list(raw_seed_traj),
            'attempts': [],
        }
        bundle_record['input_seed_summary'] = self._summarize_trajectory_points(
            bundle_record['input_seed_trajectory'],
        )
        if raw_seed_traj is None:
            for attempt_index in range(1, len(profiles) + 1):
                bundle_record['attempts'].append({
                    'probe_label': f'{probe_label}_{attempt_index}',
                    'source_label': f'{probe_label}_{attempt_index}',
                    'success': False,
                    'status': 'seed_missing',
                    'failure_reason': 'raw_seed_missing',
                })
            return bundle_record

        sequence_goal, sequence_goal_summary = self._build_alignment_seed_sequence_goal(
            target_pose,
            raw_seed_traj,
        )
        bundle_record['sequence_goal_summary'] = sequence_goal_summary
        bundle_record['sequence_length'] = int(sequence_goal.get('sequence_length', 0))

        success_count = 0
        for attempt_index, profile in enumerate(profiles, start=1):
            source_label = f'{probe_label}_{attempt_index}'
            attempt_record = {
                'probe_label': source_label,
                'source_label': source_label,
                'candidate_rank': int(attempt_index),
                'attempt_profile_name': profile.get('attempt_profile_name'),
                'sequence_length': int(sequence_goal.get('sequence_length', 0)),
                'candidate_pool_accepted': False,
                'final_selected': False,
                'retarget_success': False,
                'pose_error_summary': {'status': 'not_available'},
                'self_collision_summary': {'status': 'not_evaluated'},
                'scene_collision_summary': {'status': 'not_evaluated'},
                'success': False,
                'status': 'pending',
                'failure_reason': None,
                'sequence_goal_summary': sequence_goal_summary,
                'attempt_profile': self._round_nested_debug_value(profile, float_digits=6),
            }
            try:
                retargeter, retargeter_build_summary = self._build_sequence_retargeter(
                    use_mpc=False,
                    optimization_dt=profile.get('optimization_dt', 0.05),
                    num_seeds_global=profile.get('num_seeds_global', 64),
                    global_ik_num_iters=profile.get('global_ik_num_iters'),
                    local_ik_num_iters=profile.get('local_ik_num_iters'),
                    velocity_regularization_weight=profile.get('velocity_regularization_weight'),
                    acceleration_regularization_weight=profile.get('acceleration_regularization_weight'),
                    attempt_profile_name=profile.get('attempt_profile_name'),
                )
                attempt_record['retargeter_build_summary'] = retargeter_build_summary
                if not retargeter_build_summary.get('build_success') or retargeter is None:
                    attempt_record['status'] = 'retargeter_build_failed'
                    attempt_record['failure_reason'] = retargeter_build_summary.get('failure_reason')
                else:
                    framewise_result = self._solve_alignment_seed_sequence_framewise(
                        retargeter,
                        sequence_goal,
                        start_joint,
                        initial_seed_joint=raw_seed_traj[0],
                    )
                    attempt_record['solver_strategy'] = 'explicit_start_then_local_ik'
                    attempt_record['framewise_solver_summary'] = framewise_result.get('summary')
                    attempt_record['frame_debug'] = framewise_result.get('frame_debug')
                    if not framewise_result.get('success'):
                        attempt_record['status'] = 'framewise_sequence_failed'
                        attempt_record['failure_reason'] = framewise_result.get('failure_reason')
                        attempt_record['failed_frame_index'] = framewise_result.get('failed_frame_index')
                        bundle_record['attempts'].append(attempt_record)
                        continue
                    output_trajectory = framewise_result['trajectory']
                    pose_error_summary = self._summarize_sequence_pose_error(
                        output_trajectory,
                        sequence_goal,
                    )
                    stage_probe = {
                        'probe_label': source_label,
                        'alignment_tolerance_deg': float(alignment_tolerance_deg),
                        'raw_seed': self._build_alignment_profile_debug(
                            raw_seed_traj,
                            alignment_tolerance_deg,
                        ),
                        'sequence_output': self._build_alignment_profile_debug(
                            output_trajectory,
                            alignment_tolerance_deg,
                        ),
                    }
                    attempt_record['success'] = True
                    attempt_record['status'] = 'success'
                    attempt_record['retarget_success'] = True
                    attempt_record['pose_error_summary'] = pose_error_summary
                    attempt_record['self_collision_summary'] = {
                        'status': 'deferred_to_candidate_pool_evaluation',
                    }
                    attempt_record['scene_collision_summary'] = {
                        'status': 'deferred_to_candidate_pool_evaluation',
                    }
                    attempt_record['output_trajectory'] = self._trajectory_tensor_to_list(
                        output_trajectory,
                    )
                    attempt_record['output_summary'] = self._summarize_trajectory_points(
                        attempt_record['output_trajectory'],
                    )
                    attempt_record['stage_alignment_probe'] = stage_probe
                    attempt_record['trajectory'] = output_trajectory
                    success_count += 1
            except Exception as exc:
                attempt_record['success'] = False
                attempt_record['retarget_success'] = False
                attempt_record['status'] = 'solver_exception'
                attempt_record['failure_reason'] = str(exc)
                attempt_record['exception_type'] = type(exc).__name__
            bundle_record['attempts'].append(attempt_record)

        bundle_record['success'] = success_count > 0
        bundle_record['status'] = 'partial_success' if 0 < success_count < len(profiles) else (
            'success' if success_count == len(profiles) else 'all_failed'
        )
        bundle_record['failure_reason'] = None if success_count > 0 else 'no_sequence_attempt_succeeded'
        bundle_record['success_count'] = int(success_count)
        return bundle_record

    def _resample_seed_traj_linear(self, seed_traj: torch.Tensor, target_horizon: int) -> torch.Tensor:
        """当前默认重采样：线性插值到 trajopt 时域。"""
        if int(target_horizon) <= 0:
            raise ValueError(f'target_horizon must be positive, got {target_horizon}')
        if seed_traj.shape[0] == int(target_horizon):
            return seed_traj.detach().clone().to(device='cuda:0', dtype=torch.float32)

        # [caohy] Task 22 Phase 2：当前 alignment seed 的 waypoint 数量与 trajopt 时域可能不一致，
        # 第一版实验先只做纯轨迹重采样，不引入其它目标函数或筛选逻辑改动，避免实验变量混杂。
        seed_traj_resampled = F.interpolate(
            seed_traj.transpose(0, 1).unsqueeze(0),
            size=int(target_horizon),
            mode='linear',
            align_corners=True,
        ).squeeze(0).transpose(0, 1).contiguous()
        return seed_traj_resampled.to(device='cuda:0', dtype=torch.float32)

    def _resample_seed_traj_even_index(self, seed_traj: torch.Tensor, target_horizon: int) -> torch.Tensor:
        """只读 A/B：直接保留原轨迹中的等间距 waypoint，避免线性插值跨大步。"""
        target_horizon = int(target_horizon)
        if target_horizon <= 0:
            raise ValueError(f'target_horizon must be positive, got {target_horizon}')
        seed_traj = seed_traj.detach().clone().to(device='cuda:0', dtype=torch.float32)
        total_steps = int(seed_traj.shape[0])
        if total_steps == target_horizon:
            return seed_traj
        if total_steps < target_horizon:
            return self._resample_seed_traj_linear(seed_traj, target_horizon)

        index_float = torch.linspace(0, total_steps - 1, steps=target_horizon, device=seed_traj.device)
        index = torch.round(index_float).to(dtype=torch.long)
        index[0] = 0
        index[-1] = total_steps - 1
        for i in range(1, len(index)):
            min_allowed = index[i - 1] + 1
            remaining = len(index) - i - 1
            max_allowed = total_steps - 1 - remaining
            index[i] = torch.clamp(index[i], min=min_allowed, max=max_allowed)
        return seed_traj[index]

    def _resample_seed_traj_jump_preserving(
        self,
        seed_traj: torch.Tensor,
        target_horizon: int,
        jump_index: Optional[int],
    ) -> torch.Tensor:
        """只读 A/B：下采样时强制保留最大跳变前后锚点，观察 jump 是否被当前重采样放大。"""
        target_horizon = int(target_horizon)
        if target_horizon <= 0:
            raise ValueError(f'target_horizon must be positive, got {target_horizon}')
        seed_traj = seed_traj.detach().clone().to(device='cuda:0', dtype=torch.float32)
        total_steps = int(seed_traj.shape[0])
        if total_steps == target_horizon:
            return seed_traj
        if total_steps < target_horizon or jump_index is None:
            return self._resample_seed_traj_even_index(seed_traj, target_horizon)

        jump_index = int(jump_index)
        essential = {0, total_steps - 1}
        if 0 < jump_index < total_steps:
            essential.add(jump_index - 1)
            essential.add(jump_index)
        essential_sorted = sorted(essential)
        if len(essential_sorted) >= target_horizon:
            essential_sorted = essential_sorted[:target_horizon - 1] + [total_steps - 1]
            essential_sorted = sorted(set(essential_sorted))

        candidate_float = torch.linspace(0, total_steps - 1, steps=target_horizon, device=seed_traj.device)
        candidate_index = [int(round(float(v))) for v in candidate_float.detach().cpu().tolist()]
        selected = set(essential_sorted)
        for idx in candidate_index:
            selected.add(max(0, min(total_steps - 1, idx)))
            if len(selected) >= target_horizon:
                break
        if len(selected) < target_horizon:
            for idx in range(total_steps):
                selected.add(idx)
                if len(selected) >= target_horizon:
                    break
        final_index = sorted(selected)[:target_horizon]
        if final_index[-1] != total_steps - 1:
            final_index[-1] = total_steps - 1
        for i in range(1, len(final_index)):
            if final_index[i] <= final_index[i - 1]:
                final_index[i] = min(total_steps - 1, final_index[i - 1] + 1)
        return seed_traj[torch.tensor(final_index, device=seed_traj.device, dtype=torch.long)]

    def _build_seed_prepare_probe_summaries(
        self,
        seed_traj: torch.Tensor,
        prepared_seed_traj: torch.Tensor,
        jump_index: Optional[int],
    ) -> dict:
        """只读 A/B：统一汇总 raw / current / alternative prepared seed 的 jump 指标。"""
        target_horizon = int(self._planner.trajopt_solver.action_horizon)
        even_index_seed = self._resample_seed_traj_even_index(seed_traj, target_horizon)
        jump_preserving_seed = self._resample_seed_traj_jump_preserving(
            seed_traj,
            target_horizon,
            jump_index,
        )
        return {
            'action_horizon': target_horizon,
            'raw_seed_summary': self._summarize_seed_step_metrics(seed_traj),
            'prepared_seed_summary': self._summarize_seed_step_metrics(prepared_seed_traj[0, 0]),
            'even_index_prepared_summary': self._summarize_seed_step_metrics(even_index_seed),
            'jump_preserving_prepared_summary': self._summarize_seed_step_metrics(jump_preserving_seed),
        }

    def _solve_alignment_seed_trajopt(self, goal, current_state, prepared_seed_traj):
        """统一封装一次 trajopt solve，便于主路径与 shadow A/B 复用。"""
        return self._planner.trajopt_solver.solve_pose(
            goal,
            current_state,
            seed_traj=prepared_seed_traj,
            use_implicit_goal=True,
            return_seeds=1,
            finetune_attempts=1,
        )

    def _split_position_tensor_candidates(self, position_tensor) -> list[torch.Tensor]:
        """[caohy] Task 10：把 solve_pose(return_seeds>1) 的位置张量拆成候选列表。"""
        if position_tensor is None:
            return []
        if hasattr(position_tensor, 'detach'):
            position_tensor = position_tensor.detach().cpu()
        if position_tensor.ndim == 1:
            return [position_tensor.unsqueeze(0)]
        if position_tensor.ndim == 2:
            return [position_tensor]
        if position_tensor.ndim == 3:
            if position_tensor.shape[0] == 1:
                return [position_tensor.squeeze(0)]
            return [position_tensor[idx] for idx in range(int(position_tensor.shape[0]))]
        if position_tensor.ndim == 4:
            reshaped = position_tensor.reshape(-1, position_tensor.shape[-2], position_tensor.shape[-1])
            return [reshaped[idx] for idx in range(int(reshaped.shape[0]))]
        reshaped = position_tensor.reshape(-1, position_tensor.shape[-2], position_tensor.shape[-1])
        return [reshaped[idx] for idx in range(int(reshaped.shape[0]))]

    def _extract_result_interpolated_candidates(self, result) -> list[torch.Tensor]:
        """[caohy] Task 10：提取 trajopt result 中所有插值输出候选。"""
        if result is None:
            return []
        try:
            js_result = result.get_interpolated_plan()
            pos_tensor = getattr(js_result, 'position', None)
            return self._split_position_tensor_candidates(pos_tensor)
        except Exception:
            return []

    def _extract_result_js_solution_candidates(self, result) -> list[torch.Tensor]:
        """[caohy] Task 11：直接从 result.js_solution.position 拆出多 seed 关节轨迹候选。"""
        if result is None:
            return []
        try:
            js_solution = getattr(result, 'js_solution', None)
            pos_tensor = getattr(js_solution, 'position', None)
            return self._split_position_tensor_candidates(pos_tensor)
        except Exception:
            return []

    def _build_planner_topk_explicit_seed_inputs(
        self,
        start_joint,
        target_pose,
        requested_seed_count: int,
    ) -> dict:
        """[caohy] Task 11：生成 1 条 constant seed + N 条 auto goal IK seed 的显式输入。"""
        requested_seed_count = max(1, int(requested_seed_count))
        target_horizon = int(self._planner.trajopt_solver.action_horizon)
        auto_goal_seed_count = max(0, requested_seed_count - 1)
        record = {
            'requested_seed_count': int(requested_seed_count),
            'constant_seed_count': 1,
            'auto_goal_seed_count': int(auto_goal_seed_count),
            'target_horizon': int(target_horizon),
            'source': 'mixed_constant_plus_goal_ik_auto',
            'success': False,
            'failure_reason': None,
            'ik_return_seeds': int(max(32, auto_goal_seed_count * 8)),
            'unique_goal_seed_count': 0,
            'selected_goal_seed_count': 1,
            'selected_goal_seed_summaries': [],
        }
        start_tensor = torch.tensor(start_joint, device='cuda:0', dtype=torch.float32)
        constant_seed_traj = (
            start_tensor.view(1, 1, -1).repeat(1, target_horizon, 1).squeeze(0).contiguous()
        )
        selected_goal_seeds = [start_tensor.detach().clone()]
        selected_seed_trajs = [constant_seed_traj]
        record['selected_goal_seed_summaries'].append({
            'seed_kind': 'constant_seed',
            'solution_key': [round(float(v), 4) for v in start_tensor.detach().cpu().tolist()],
            'delta_l2_to_start': 0.0,
            'position_error_m': None,
            'orientation_error_rad': None,
        })
        if auto_goal_seed_count <= 0:
            seed_config_tensor = torch.stack(selected_goal_seeds, dim=0).unsqueeze(0)
            seed_traj_tensor = torch.stack(selected_seed_trajs, dim=0).unsqueeze(0)
            record['success'] = True
            record['seed_config_summary'] = {
                'shape': list(seed_config_tensor.shape),
                'first': [round(float(v), 6) for v in seed_config_tensor[0, 0].detach().cpu().tolist()],
                'last': [round(float(v), 6) for v in seed_config_tensor[0, -1].detach().cpu().tolist()],
            }
            record['seed_traj_summary'] = {
                'shape': list(seed_traj_tensor.shape),
                'first_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, 0]),
                'last_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, -1]),
            }
            record['seed_config'] = seed_config_tensor.to(device='cuda:0', dtype=torch.float32)
            record['seed_traj'] = seed_traj_tensor.to(device='cuda:0', dtype=torch.float32)
            return record
        current_state = CuJointState.from_position(
            start_tensor.view(1, -1),
            joint_names=self._joint_names,
        )
        goal = self._make_goal_tool_pose(target_pose)
        try:
            self._ik_solver.reset_seed()
            ik_result = self._ik_solver.solve_pose(
                goal,
                current_state=current_state,
                seed_config=start_tensor.view(1, 1, -1),
                return_seeds=int(record['ik_return_seeds']),
            )
            record['ik_result_summary'] = self._build_result_debug_payload(ik_result)
            feasible = getattr(ik_result, 'feasible', None)
            solution = getattr(ik_result, 'solution', None)
            if feasible is None or solution is None:
                record['failure_reason'] = 'goal_ik_missing_feasible_or_solution'
                return record
            feasible_batch_idx, feasible_seed_idx = feasible.nonzero(as_tuple=True)
            if feasible_batch_idx.numel() == 0:
                record['failure_reason'] = 'goal_ik_no_feasible_seed'
                return record
            position_error = getattr(ik_result, 'position_error', None)
            rotation_error = getattr(ik_result, 'rotation_error', None)
            unique_goal_seeds = []
            unique_seed_summaries = []
            seen_keys = {
                tuple(round(float(v), 4) for v in start_tensor.detach().cpu().tolist())
            }
            for item_idx in range(int(feasible_batch_idx.numel())):
                batch_idx = int(feasible_batch_idx[item_idx].item())
                seed_idx = int(feasible_seed_idx[item_idx].item())
                wrapped_solution = self._wrap_seed_solution_to_prev(
                    start_tensor,
                    solution[batch_idx, seed_idx],
                ).detach().clone()
                solution_key = tuple(
                    round(float(v), 4) for v in wrapped_solution.detach().cpu().tolist()
                )
                if solution_key in seen_keys:
                    continue
                seen_keys.add(solution_key)
                delta_l2 = float(
                    torch.linalg.norm(wrapped_solution - start_tensor, dim=-1).item()
                )
                pos_err = None
                ori_err = None
                if position_error is not None:
                    pos_err = float(position_error[batch_idx, seed_idx].item())
                if rotation_error is not None:
                    ori_err = float(rotation_error[batch_idx, seed_idx].item())
                unique_goal_seeds.append(wrapped_solution)
                unique_seed_summaries.append({
                    'seed_kind': 'auto_goal_ik_seed',
                    'solution_key': list(solution_key),
                    'delta_l2_to_start': delta_l2,
                    'position_error_m': pos_err,
                    'orientation_error_rad': ori_err,
                })
                if len(unique_goal_seeds) >= auto_goal_seed_count:
                    break
            record['unique_goal_seed_count'] = int(len(unique_goal_seeds))
            if len(unique_goal_seeds) < auto_goal_seed_count:
                record['failure_reason'] = (
                    f'goal_ik_unique_seed_count_insufficient:{len(unique_goal_seeds)}'
                )
                record['selected_goal_seed_summaries'].extend(unique_seed_summaries)
                return record
            selected_goal_seeds.extend(unique_goal_seeds[:auto_goal_seed_count])
            record['selected_goal_seed_summaries'].extend(
                unique_seed_summaries[:auto_goal_seed_count]
            )
            for goal_seed in unique_goal_seeds[:auto_goal_seed_count]:
                coarse_seed = torch.stack([start_tensor, goal_seed], dim=0)
                selected_seed_trajs.append(
                    self._resample_seed_traj_linear(coarse_seed, target_horizon)
                )
            seed_config_tensor = torch.stack(selected_goal_seeds, dim=0).unsqueeze(0)
            seed_traj_tensor = torch.stack(selected_seed_trajs, dim=0).unsqueeze(0)
            record['success'] = True
            record['selected_goal_seed_count'] = int(len(selected_goal_seeds))
            record['seed_config_summary'] = {
                'shape': list(seed_config_tensor.shape),
                'first': [round(float(v), 6) for v in seed_config_tensor[0, 0].detach().cpu().tolist()],
                'last': [round(float(v), 6) for v in seed_config_tensor[0, -1].detach().cpu().tolist()],
            }
            record['seed_traj_summary'] = {
                'shape': list(seed_traj_tensor.shape),
                'first_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, 0]),
                'last_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, -1]),
            }
            record['seed_config'] = seed_config_tensor.to(device='cuda:0', dtype=torch.float32)
            record['seed_traj'] = seed_traj_tensor.to(device='cuda:0', dtype=torch.float32)
            return record
        except Exception as exc:
            record['failure_reason'] = str(exc)
            record['exception_type'] = type(exc).__name__
            return record

    def _solve_alignment_seed_horizon_trajopt(
        self,
        goal,
        current_state,
        prepared_seed_traj,
        return_seeds: int = 5,
    ):
        """[caohy] Task 10：统一封装 horizon>1 目标的 trajopt solve。"""
        return self._planner.trajopt_solver.solve_pose(
            goal,
            current_state,
            seed_traj=prepared_seed_traj,
            use_implicit_goal=True,
            return_seeds=int(return_seeds),
            finetune_attempts=1,
        )

    def _solve_alignment_seed_trajopt_with_goal_state(
        self,
        goal,
        current_state,
        prepared_seed_traj,
        goal_state,
    ):
        """[caohy] Task 29：只读 shadow，对 solve_pose 额外显式传 prepared_seed_last 作为 goal_state。"""
        return self._planner.trajopt_solver.solve_pose(
            goal,
            current_state,
            seed_traj=prepared_seed_traj,
            use_implicit_goal=True,
            goal_state=goal_state,
            return_seeds=1,
            finetune_attempts=1,
        )

    def _get_alignment_sequence_branch_mode(self) -> str:
        """[caohy] Task 10：统一解析 sequence（序列目标旁支）运行模式。"""
        # [caohy] Task 10：先兼容旧环境变量名，避免容器脚本和历史运行命令立即失效；
        # 但从这里开始，代码语义正式以 sequence branch 为准。
        # 当前任务已明确先搁置该旁支，因此默认值统一收口到 off，
        # 避免未显式配置环境变量时继续产生额外 sequence 计算。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_SEQUENCE_BRANCH_MODE',
            os.environ.get('CUROBO_ALIGNMENT_HORIZON_BRANCH_MODE', 'off'),
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'shadow', 'candidate'):
            return raw_value
        self.get_logger().warn(
            'Invalid CUROBO_ALIGNMENT_SEQUENCE_BRANCH_MODE='
            f'{raw_value}, fallback to off'
        )
        return 'off'

    def _get_alignment_horizon_branch_mode(self) -> str:
        """[caohy] Task 10：兼容旧调用，实际统一走 sequence branch mode。"""
        return self._get_alignment_sequence_branch_mode()

    def _get_alignment_trajopt_family_mode(self) -> str:
        """[caohy] Task 13：统一解析 alignment trajopt family（多家族对齐种子优化旁支）模式。"""
        # [caohy] Task 15：旧 4*1 family 已下线为可选旧旁支。
        # 默认直接 off，不再默认生成 alignment_seed_trajopt_1..4，
        # 避免主链路继续消耗这套旧分支的计算开销。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_TRAJOPT_FAMILY_MODE', 'off'
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'shadow', 'candidate'):
            return raw_value
        self.get_logger().warn(
            f'Invalid CUROBO_ALIGNMENT_TRAJOPT_FAMILY_MODE={raw_value}, fallback to off'
        )
        return 'off'

    def _get_alignment_trajopt_family_topk_shadow_mode(self) -> str:
        """[caohy] Task 14：统一解析 alignment trajopt family topk shadow（1*4 影子实验）模式。"""
        # [caohy] Task 14：这条分支当前只是只读实验，默认必须收口到 off，
        # 避免在正式主链路里默认增加一次额外 solve_pose(...) 计算。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_TRAJOPT_FAMILY_TOPK_SHADOW_MODE', 'off'
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'shadow', 'candidate'):
            return raw_value
        self.get_logger().warn(
            'Invalid CUROBO_ALIGNMENT_TRAJOPT_FAMILY_TOPK_SHADOW_MODE='
            f'{raw_value}, fallback to off'
        )
        return 'off'

    def _get_alignment_trajopt_family_topk_mode(self) -> str:
        """[caohy] Task 15：统一解析 alignment trajopt family topk（1*4 正式替换）模式。"""
        # [caohy] Task 15：1*4 已正式接管 family 主链路，默认 candidate。
        # Task 14 的 topk_shadow 仍保持独立开关，避免“正式入池”和“只读诊断”混淆。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_TRAJOPT_FAMILY_TOPK_MODE', 'candidate'
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'candidate'
        if raw_value in ('off', 'shadow', 'candidate'):
            return raw_value
        self.get_logger().warn(
            'Invalid CUROBO_ALIGNMENT_TRAJOPT_FAMILY_TOPK_MODE='
            f'{raw_value}, fallback to candidate'
        )
        return 'candidate'

    def _get_alignment_trajopt_family_formal_variant(self) -> str:
        """[caohy] Task 15：返回当前正式 family（多家族对齐优化）候选方案标签。"""
        topk_mode = self._get_alignment_trajopt_family_topk_mode()
        family_mode = self._get_alignment_trajopt_family_mode()
        if topk_mode == 'candidate':
            return 'family_1x4_formal'
        if family_mode == 'candidate':
            return 'family_4x1_formal'
        if topk_mode == 'shadow':
            return 'family_1x4_shadow_only'
        if family_mode == 'shadow':
            return 'family_4x1_shadow_only'
        return 'family_formal_disabled'

    def _get_alignment_trajopt_legacy_mode(self) -> str:
        """[caohy] Task 13：统一解析老 raw alignment trajopt（旧旁支）模式。"""
        # [caohy] Task 13：family（多家族对齐种子优化）已接管正式主链路，
        # 老 raw（单条原始对齐种子优化）默认收口到 off，仅在对照/回退时显式开启。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_TRAJOPT_LEGACY_MODE', 'off'
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'shadow', 'candidate'):
            return raw_value
        self.get_logger().warn(
            f'Invalid CUROBO_ALIGNMENT_TRAJOPT_LEGACY_MODE={raw_value}, fallback to off'
        )
        return 'off'

    def _get_alignment_raw_seed_family_pool_mode(self) -> str:
        """[caohy] 解析 raw seed family（原始对齐种子族）是否允许进入最终候选池。"""
        # [caohy] 原始 alignment_seed_family_* 由插值 + 逐点 IK 生成，未经过 CuRobo
        # trajopt 的碰撞优化；默认只作为 alignment_seed_trajopt_* 的输入种子。
        raw_value = os.environ.get(
            'CUROBO_ALIGNMENT_RAW_SEED_FAMILY_POOL_MODE', 'off'
        ).strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'candidate'):
            return raw_value
        self.get_logger().warn(
            'Invalid CUROBO_ALIGNMENT_RAW_SEED_FAMILY_POOL_MODE='
            f'{raw_value}, fallback to off'
        )
        return 'off'

    def _get_diffusion_seed_mode(self) -> str:
        """[caohy] diffusionSeedLearning phase 1：解析 diffusion seed provider 运行模式。"""
        param_value = 'off'
        try:
            param_value = str(self.get_parameter('diffusion_seed_mode').value)
        except Exception:
            param_value = 'off'
        raw_value = os.environ.get('CUROBO_DIFFUSION_SEED_MODE', param_value)
        mode = normalize_diffusion_mode(raw_value)
        if mode != str(raw_value or 'off').strip().lower():
            self.get_logger().warn(
                'Invalid CUROBO_DIFFUSION_SEED_MODE='
                f'{raw_value}, fallback to off'
            )
        return mode

    @staticmethod
    def _coerce_bool_param(value, default: bool = False) -> bool:
        """把 launch/ROS 参数中的 bool/string 稳定规整为 bool。"""
        if isinstance(value, bool):
            return bool(value)
        if value is None:
            return bool(default)
        text = str(value).strip().lower()
        if text in ('1', 'true', 'yes', 'on'):
            return True
        if text in ('0', 'false', 'no', 'off', ''):
            return False
        return bool(default)

    def _get_diffusion_seed_runtime_config(self) -> dict:
        """[caohy] diffusionSeedLearning phase 6：读取 candidate/shadow 运行配置。"""

        def _param(name: str, default):
            try:
                value = self.get_parameter(name).value
            except Exception:
                return default
            return default if value is None else value

        def _int_param(name: str, default: int, minimum: int = 0) -> int:
            try:
                value = int(_param(name, default))
            except Exception:
                value = int(default)
            return max(int(minimum), int(value))

        def _float_param(name: str, default: float, minimum: float = 0.0) -> float:
            try:
                value = float(_param(name, default))
            except Exception:
                value = float(default)
            return max(float(minimum), float(value))

        mode = self._get_diffusion_seed_mode()
        return {
            'mode': mode,
            'generated_samples_path': str(_param('diffusion_seed_generated_samples_path', '')),
            'checkpoint_path': str(_param('diffusion_seed_checkpoint_path', '')),
            'k_generate': _int_param('diffusion_seed_k_generate', 4, minimum=0),
            'k_accept': _int_param('diffusion_seed_k_accept', 2, minimum=0),
            'model_timeout_sec': _float_param('diffusion_seed_model_timeout_sec', 0.2, minimum=0.0),
            'max_start_gap_l2': _float_param('diffusion_seed_max_start_gap_l2', 0.05, minimum=0.0),
            'max_step_l2': _float_param('diffusion_seed_max_step_l2', 1.0, minimum=0.0),
            'joint_abs_limit': _float_param('diffusion_seed_joint_abs_limit', 6.2832, minimum=0.0),
            'fallback_to_rule_seed': self._coerce_bool_param(
                _param('diffusion_seed_fallback_to_rule_seed', True),
                default=True,
            ),
            'allow_real_robot_candidate': self._coerce_bool_param(
                _param('diffusion_seed_allow_real_robot_candidate', False),
                default=False,
            ),
            'use_real_robot': bool(getattr(self, '_use_real_robot', False)),
        }

    def _build_initial_diffusion_seed_report(self, config: dict) -> dict:
        mode = normalize_diffusion_mode(config.get('mode'))
        return {
            'provider_name': 'diffusion_seed',
            'mode': mode,
            'status': 'disabled' if mode == 'off' else 'pending',
            'generated_count': 0,
            'accepted_to_pool_count': 0,
            'optimized_success_count': 0,
            'runtime_config': {
                key: val
                for key, val in dict(config).items()
                if key not in ('generated_samples_path', 'checkpoint_path')
            },
            'generated_samples_path': config.get('generated_samples_path'),
            'checkpoint_path': config.get('checkpoint_path'),
            'candidates': [],
            'optimization_attempts': [],
            'runtime_effect': (
                'no_candidates_generated'
                if mode == 'off'
                else 'pending_provider_generation'
            ),
        }

    def _precheck_diffusion_seed_tensor(
        self,
        seed_traj: torch.Tensor,
        start_joint,
        config: dict,
    ) -> dict:
        """运行态二次预检：补真实 joint limit，不只依赖文件 provider 的通用检查。"""
        traj = self._to_cpu_tensor(seed_traj)
        if traj is None:
            return {'valid': False, 'failure_reason': 'trajectory_tensor_missing'}
        while traj.ndim > 2:
            if traj.shape[0] == 1:
                traj = traj.squeeze(0)
            else:
                traj = traj.reshape(-1, traj.shape[-1])
        if traj.ndim == 1:
            traj = traj.unsqueeze(0)
        dof = len(self._joint_names)
        if traj.ndim != 2 or int(traj.shape[-1]) != int(dof) or int(traj.shape[0]) <= 0:
            return {
                'valid': False,
                'shape_valid': False,
                'trajectory_shape': list(traj.shape),
                'expected_dof': int(dof),
                'failure_reason': 'invalid_shape_or_dof',
            }
        finite = bool(torch.isfinite(traj).all().item())
        start_tensor = torch.tensor(start_joint, dtype=torch.float32)
        start_gap = float(torch.linalg.norm(traj[0] - start_tensor).item())
        if int(traj.shape[0]) > 1:
            step_l2 = torch.linalg.norm(traj[1:] - traj[:-1], dim=-1)
            joint_step_max_l2 = float(torch.max(step_l2).item())
        else:
            joint_step_max_l2 = 0.0
        joint_abs_max = float(torch.max(torch.abs(traj)).item())
        limit_summary = self._summarize_trajectory_joint_limit_violation(traj)
        thresholds = {
            'max_start_gap_l2': float(config.get('max_start_gap_l2', 0.05)),
            'max_step_l2': float(config.get('max_step_l2', 1.0)),
            'joint_abs_limit': float(config.get('joint_abs_limit', 6.2832)),
        }
        failure_reasons = []
        if not finite:
            failure_reasons.append('non_finite')
        eps = 1e-6
        if start_gap > thresholds['max_start_gap_l2'] + eps:
            failure_reasons.append('start_gap_exceeds_threshold')
        if joint_step_max_l2 > thresholds['max_step_l2'] + eps:
            failure_reasons.append('joint_step_exceeds_threshold')
        if joint_abs_max > thresholds['joint_abs_limit'] + eps:
            failure_reasons.append('joint_abs_limit_exceeds_threshold')
        if bool(limit_summary.get('has_violation')):
            failure_reasons.append('robot_joint_limit_violation')
        return {
            'valid': not failure_reasons,
            'shape_valid': True,
            'finite': finite,
            'trajectory_shape': list(traj.shape),
            'start_gap_l2': round(float(start_gap), 6),
            'joint_step_max_l2': round(float(joint_step_max_l2), 6),
            'joint_abs_max': round(float(joint_abs_max), 6),
            'robot_joint_limit_summary': limit_summary,
            'thresholds': thresholds,
            'failure_reason': ';'.join(failure_reasons) if failure_reasons else None,
        }

    def _run_diffusion_seed_provider_for_request(
        self,
        *,
        start_joint,
        target_pose,
        level_tolerance_deg: float,
        strict_level: bool,
        plan_request_index: int,
        config: dict,
        all_candidates: list,
        candidate_source_labels: list[str],
    ) -> tuple[dict, float]:
        """[caohy] diffusionSeedLearning phase 6：candidate/shadow 入池主流程。

        diffusion seed 只作为初始轨迹种子；candidate/fallback 模式下必须先经过
        `_optimize_alignment_seed()` 的 CuRobo trajopt 修复，成功后才进入最终候选池。
        """
        import time

        report = self._build_initial_diffusion_seed_report(config)
        mode = normalize_diffusion_mode(config.get('mode'))
        if mode == 'off':
            return report, 0.0

        if (
            mode == 'candidate'
            and bool(config.get('use_real_robot'))
            and not bool(config.get('allow_real_robot_candidate'))
        ):
            report.update({
                'status': 'real_robot_candidate_blocked',
                'runtime_effect': 'candidate_mode_blocked_on_real_robot',
                'safety_guard': {
                    'use_real_robot': bool(config.get('use_real_robot')),
                    'allow_real_robot_candidate': bool(config.get('allow_real_robot_candidate')),
                },
            })
            return report, 0.0

        provider = FileDiffusionSeedProvider(
            DiffusionSeedProviderConfig(
                mode=mode,
                generated_samples_path=str(config.get('generated_samples_path') or ''),
                k_generate=int(config.get('k_generate') or 0),
                k_accept=int(config.get('k_accept') or 0),
                model_timeout_sec=float(config.get('model_timeout_sec') or 0.0),
                max_start_gap_l2=float(config.get('max_start_gap_l2') or 0.0),
                max_step_l2=float(config.get('max_step_l2') or 0.0),
                joint_abs_limit=float(config.get('joint_abs_limit') or 0.0),
                fallback_to_rule_seed=bool(config.get('fallback_to_rule_seed', True)),
            )
        )
        request_context = {
            'plan_request_index': int(plan_request_index),
            'start_joint': [float(v) for v in list(start_joint)],
            'target_pose': [float(v) for v in list(target_pose)],
            'dof': int(len(self._joint_names)),
            'joint_names': list(self._joint_names),
            'tool_frames': list(self._tool_frames),
            'level_tolerance_deg': float(level_tolerance_deg),
            'strict_level': bool(strict_level),
        }

        t_provider = time.time()
        provider_result = provider.generate(request_context)
        provider_time_sec = time.time() - t_provider
        report = provider_result.to_lifecycle_dict()
        report['runtime_config'] = {
            key: val
            for key, val in dict(config).items()
            if key not in ('generated_samples_path', 'checkpoint_path')
        }
        report['generated_samples_path'] = config.get('generated_samples_path')
        report['checkpoint_path'] = config.get('checkpoint_path')
        report['provider_time_sec'] = round(float(provider_time_sec), 6)
        report['candidate_mode_allowed'] = bool(
            mode in ('candidate', 'fallback')
            and (
                not bool(config.get('use_real_robot'))
                or bool(config.get('allow_real_robot_candidate'))
            )
        )
        report['optimization_attempts'] = []
        report['accepted_to_pool_labels'] = []
        report['rejected_labels'] = []
        report['runtime_effect'] = (
            'shadow_only_no_pool_insertion'
            if mode == 'shadow'
            else 'prechecked_curobo_repaired_candidates_may_enter_pool'
        )

        total_extra_time = provider_time_sec
        accepted_count = 0
        k_accept = int(config.get('k_accept') or 0)
        candidate_reports_by_label = {
            str(item.get('source_label')): item
            for item in report.get('candidates', [])
            if isinstance(item, dict)
        }
        for candidate in provider_result.candidates:
            source_label = str(candidate.source_label)
            raw_traj = torch.tensor(
                candidate.trajectory_points,
                dtype=torch.float32,
                device='cpu',
            )
            runtime_precheck = self._precheck_diffusion_seed_tensor(
                raw_traj,
                start_joint,
                config,
            )
            candidate.precheck = {
                **dict(candidate.precheck or {}),
                'runtime': runtime_precheck,
                'valid': bool(candidate.precheck.get('valid')) and bool(runtime_precheck.get('valid')),
            }
            candidate_report = candidate_reports_by_label.get(source_label)
            if isinstance(candidate_report, dict):
                candidate_report['precheck'] = dict(candidate.precheck)
            if not bool(candidate.precheck.get('valid')):
                report['rejected_labels'].append(source_label)
                if isinstance(candidate_report, dict):
                    candidate_report['entered_pool'] = False
                    candidate_report['metadata']['pool_rejection_reason'] = (
                        candidate.precheck.get('failure_reason')
                        or runtime_precheck.get('failure_reason')
                        or 'precheck_failed'
                    )
                continue
            if mode == 'shadow':
                report['rejected_labels'].append(source_label)
                if isinstance(candidate_report, dict):
                    candidate_report['entered_pool'] = False
                    candidate_report['metadata']['pool_rejection_reason'] = 'shadow_mode'
                continue
            if accepted_count >= k_accept:
                report['rejected_labels'].append(source_label)
                if isinstance(candidate_report, dict):
                    candidate_report['entered_pool'] = False
                    candidate_report['metadata']['pool_rejection_reason'] = 'k_accept_limit_reached'
                continue

            t_opt = time.time()
            try:
                optimized_attempt = self._optimize_alignment_seed(
                    start_joint,
                    target_pose,
                    raw_traj,
                    alignment_tolerance_deg=level_tolerance_deg,
                    probe_label=source_label,
                )
            except Exception as exc:
                optimized_attempt = {
                    'probe_label': source_label,
                    'source_label': source_label,
                    'success': False,
                    'status': 'optimizer_exception',
                    'failure_reason': str(exc),
                    'exception_type': type(exc).__name__,
                }
            opt_time = time.time() - t_opt
            total_extra_time += opt_time
            optimized_attempt['source_label'] = source_label
            optimized_attempt['source_type'] = 'diffusion'
            optimized_attempt['candidate_pool_accepted'] = False
            optimized_attempt['final_selected'] = False
            optimized_attempt['diffusion_seed_metadata'] = dict(candidate.metadata)
            optimized_attempt['diffusion_seed_precheck'] = dict(candidate.precheck)
            optimized_attempt['optimize_time_sec'] = round(float(opt_time), 6)

            optimized_candidate = optimized_attempt.get('trajectory')
            if optimized_attempt.get('success') and optimized_candidate is not None:
                all_candidates.append(optimized_candidate)
                candidate_source_labels.append(source_label)
                pool_index = int(len(all_candidates) - 1)
                optimized_attempt['candidate_pool_accepted'] = True
                optimized_attempt['pool_candidate_index'] = pool_index
                candidate.entered_pool = True
                accepted_count += 1
                report['accepted_to_pool_labels'].append(source_label)
                if isinstance(candidate_report, dict):
                    candidate_report['entered_pool'] = True
                    candidate_report['metadata']['pool_candidate_index'] = pool_index
                    candidate_report['metadata']['optimizer_status'] = optimized_attempt.get('status')
            else:
                report['rejected_labels'].append(source_label)
                if isinstance(candidate_report, dict):
                    candidate_report['entered_pool'] = False
                    candidate_report['metadata']['pool_rejection_reason'] = (
                        optimized_attempt.get('failure_reason')
                        or optimized_attempt.get('status')
                        or 'optimizer_failed'
                    )
            report['optimization_attempts'].append(
                self._round_nested_debug_value(
                    {
                        key: value
                        for key, value in optimized_attempt.items()
                        if key != 'trajectory'
                    },
                    float_digits=6,
                )
            )

        report['generated_count'] = int(len(provider_result.candidates))
        report['accepted_to_pool_count'] = int(accepted_count)
        report['optimized_success_count'] = int(
            sum(1 for item in report['optimization_attempts'] if item.get('success'))
        )
        report['precheck_valid_count'] = int(
            sum(
                1 for item in report.get('candidates', [])
                if isinstance(item, dict) and bool(item.get('precheck', {}).get('valid'))
            )
        )
        if provider_result.error and bool(config.get('fallback_to_rule_seed', True)):
            report['fallback_triggered'] = True
            report['fallback_reason'] = provider_result.error
        else:
            report['fallback_triggered'] = bool(
                mode in ('candidate', 'fallback')
                and accepted_count == 0
                and bool(config.get('fallback_to_rule_seed', True))
            )
            report['fallback_reason'] = (
                'no_diffusion_candidate_entered_pool'
                if report['fallback_triggered'] else None
            )
        return self._round_nested_debug_value(report, float_digits=6), total_extra_time

    def _get_allow_failed_solver_output_candidates(self) -> bool:
        """[caohy] 是否允许 solver success=false 的输出轨迹作为诊断候选入池。"""
        # [caohy] 默认必须尊重 CuRobo 的 per-seed success mask；只有人工诊断时显式打开，
        # 才保留旧行为，并给候选打 unsafe_candidate_from_failed_solver 标记。
        raw_value = os.environ.get(
            'CUROBO_ALLOW_FAILED_SOLVER_OUTPUT_CANDIDATES', '0'
        ).strip().lower()
        if raw_value in ('1', 'true', 'yes', 'on'):
            return True
        if raw_value in ('', '0', 'false', 'no', 'off', 'default'):
            return False
        self.get_logger().warn(
            'Invalid CUROBO_ALLOW_FAILED_SOLVER_OUTPUT_CANDIDATES='
            f'{raw_value}, fallback to 0'
        )
        return False

    def _get_alignment_trajopt_family_configs(self) -> list[dict]:
        """[caohy] Task 13：第一版固定 4 条 alignment trajopt family 配置。"""
        return [
            {
                'source_label': 'alignment_seed_trajopt_1',
                'raw_pool_source_label': 'alignment_seed_family_1',
                'seed_family_name': 'baseline_default',
                'family_primary_variable': 'baseline_control',
                'seed_family_config': {
                    'twist_schedule_mode': 'uniform_shortest',
                    'goal_anchor_rank': 1,
                    'selection_mode': 'default_score',
                },
                'twist_schedule_mode': 'uniform_shortest',
                'goal_anchor_rank': 1,
                'selection_mode': None,
            },
            {
                'source_label': 'alignment_seed_trajopt_2',
                'raw_pool_source_label': 'alignment_seed_family_2',
                'seed_family_name': 'goal_anchor_rank_2',
                'family_primary_variable': 'goal_anchor_rank',
                'seed_family_config': {
                    'twist_schedule_mode': 'uniform_shortest',
                    'goal_anchor_rank': 2,
                    'selection_mode': 'default_score',
                },
                'twist_schedule_mode': 'uniform_shortest',
                'goal_anchor_rank': 2,
                'selection_mode': None,
            },
            {
                'source_label': 'alignment_seed_trajopt_3',
                'raw_pool_source_label': 'alignment_seed_family_3',
                'seed_family_name': 'twist_delayed_to_goal',
                'family_primary_variable': 'twist_schedule_mode',
                'seed_family_config': {
                    'twist_schedule_mode': 'delayed_to_goal',
                    'goal_anchor_rank': 1,
                    'selection_mode': 'default_score',
                },
                'twist_schedule_mode': 'delayed_to_goal',
                'goal_anchor_rank': 1,
                'selection_mode': None,
            },
            {
                'source_label': 'alignment_seed_trajopt_4',
                'raw_pool_source_label': 'alignment_seed_family_4',
                'seed_family_name': 'branch_selection_in_limit_best',
                'family_primary_variable': 'selection_mode',
                'seed_family_config': {
                    'twist_schedule_mode': 'uniform_shortest',
                    'goal_anchor_rank': 1,
                    'selection_mode': 'in_limit_best',
                },
                'twist_schedule_mode': 'uniform_shortest',
                'goal_anchor_rank': 1,
                'selection_mode': 'in_limit_best',
            },
        ]

    def _get_planner_topk_experiment_mode(self) -> str:
        """[caohy] Task 11：统一解析 planner topk（规划器前K候选）只读实验模式。"""
        raw_value = os.environ.get('CUROBO_PLANNER_TOPK_EXPERIMENT_MODE', 'off').strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'shadow'):
            return raw_value
        self.get_logger().warn(
            f'Invalid CUROBO_PLANNER_TOPK_EXPERIMENT_MODE={raw_value}, fallback to off'
        )
        return 'off'

    def _get_planner_legacy_branch_mode(self) -> str:
        """[caohy] Task 11：旧 1*4 planner 候选降级为可开关旁支，默认关闭。"""
        raw_value = os.environ.get('CUROBO_PLANNER_LEGACY_BRANCH_MODE', 'off').strip().lower()
        if raw_value in ('', 'default'):
            return 'off'
        if raw_value in ('off', 'candidate'):
            return raw_value
        self.get_logger().warn(
            f'Invalid CUROBO_PLANNER_LEGACY_BRANCH_MODE={raw_value}, fallback to off'
        )
        return 'off'

    def _get_planner_topk_shadow_k(self) -> int:
        """[caohy] Task 11：解析 planner topk shadow（影子模式）返回条数，默认 4。"""
        raw_value = os.environ.get('CUROBO_PLANNER_TOPK_SHADOW_K', '4').strip()
        try:
            shadow_k = int(raw_value)
        except ValueError:
            self.get_logger().warn(
                f'Invalid CUROBO_PLANNER_TOPK_SHADOW_K={raw_value}, fallback to 4'
            )
            return 4
        if shadow_k <= 0:
            self.get_logger().warn(
                f'Invalid CUROBO_PLANNER_TOPK_SHADOW_K={raw_value}, fallback to 4'
            )
            return 4
        return shadow_k

    def _build_alignment_sequence_seed_prepare_record(self, raw_seed_traj):
        """[caohy] Task 10：为 sequence 分支先落 raw seed 到标准时域的准备摘要。"""
        mode = self._get_alignment_sequence_branch_mode()
        record = {
            'source_label': 'alignment_seed_sequence',
            'branch_mode': mode,
            'branch_enabled': mode != 'off',
            'success': False,
            'status': 'disabled',
            'failure_reason': None,
            'action_horizon': int(self._planner.trajopt_solver.action_horizon),
            'raw_seed_trajectory': self._trajectory_tensor_to_list(raw_seed_traj),
        }
        record['raw_seed_summary'] = self._summarize_trajectory_points(
            record['raw_seed_trajectory'],
        )
        if raw_seed_traj is None:
            record['status'] = 'seed_missing'
            record['failure_reason'] = 'raw_seed_missing'
            return self._round_nested_debug_value(record, float_digits=6)
        if mode == 'off':
            return self._round_nested_debug_value(record, float_digits=6)

        prepared_seed_traj = self._prepare_seed_traj_for_trajopt(raw_seed_traj)
        prepared_seed_flat = prepared_seed_traj[0, 0]
        record['prepared_seed_trajectory'] = self._trajectory_tensor_to_list(prepared_seed_flat)
        record['prepared_seed_summary'] = self._summarize_trajectory_points(
            record['prepared_seed_trajectory'],
        )
        record['raw_seed_step_metrics'] = self._summarize_seed_step_metrics(raw_seed_traj)
        record['prepared_seed_step_metrics'] = self._summarize_seed_step_metrics(
            prepared_seed_flat,
        )
        record['seed_prepare_probe'] = self._build_seed_prepare_probe_summaries(
            raw_seed_traj,
            prepared_seed_traj,
            record['raw_seed_step_metrics'].get('max_step_jump_index'),
        )
        record['prepared_seed_traj_shape'] = list(prepared_seed_traj.shape)
        record['success'] = True
        record['status'] = 'prepared'
        record['failure_reason'] = None
        return self._round_nested_debug_value(record, float_digits=6)

    def _build_alignment_horizon_seed_prepare_record(self, raw_seed_traj):
        """[caohy] Task 10：兼容旧记录函数名，实际返回 sequence 分支摘要。"""
        return self._build_alignment_sequence_seed_prepare_record(raw_seed_traj)

    def _set_sequence_branch_record(self, lifecycle_data: dict, branch_record: dict) -> None:
        """[caohy] Task 10：统一写入 sequence_branch，并保留 horizon_branch 兼容别名。"""
        rounded_record = self._round_nested_debug_value(branch_record, float_digits=6)
        lifecycle_data['sequence_branch'] = rounded_record
        lifecycle_data['horizon_branch'] = rounded_record

    def _mark_sequence_branch_attempt_selected(
        self,
        lifecycle_data: dict,
        selected_source_label: Optional[str],
    ) -> None:
        """[caohy] Task 10：回填 sequence_branch attempt 的 final_selected 字段。"""
        sequence_branch = lifecycle_data.get('sequence_branch')
        if not isinstance(sequence_branch, dict):
            return
        attempts = sequence_branch.get('attempts')
        if not isinstance(attempts, list):
            return
        selected_label = str(selected_source_label) if selected_source_label is not None else None
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt['final_selected'] = bool(
                selected_label is not None
                and str(attempt.get('source_label')) == selected_label
            )
        lifecycle_data['horizon_branch'] = sequence_branch

    def _set_alignment_trajopt_family_branch_record(
        self,
        lifecycle_data: dict,
        branch_record: dict,
    ) -> None:
        """[caohy] Task 13：统一写入 trajopt_family_branch。"""
        lifecycle_data['trajopt_family_branch'] = self._round_nested_debug_value(
            branch_record,
            float_digits=6,
        )

    def _set_alignment_trajopt_family_topk_shadow_branch_record(
        self,
        lifecycle_data: dict,
        branch_record: dict,
    ) -> None:
        """[caohy] Task 14：统一写入 trajopt_family_topk_shadow_branch。"""
        lifecycle_data['trajopt_family_topk_shadow_branch'] = self._round_nested_debug_value(
            branch_record,
            float_digits=6,
        )

    def _set_alignment_trajopt_family_topk_branch_record(
        self,
        lifecycle_data: dict,
        branch_record: dict,
    ) -> None:
        """[caohy] Task 15：统一写入 trajopt_family_topk_branch（1*4 正式替换分支）。"""
        lifecycle_data['trajopt_family_topk_branch'] = self._round_nested_debug_value(
            branch_record,
            float_digits=6,
        )

    def _mark_alignment_trajopt_family_attempt_selected(
        self,
        lifecycle_data: dict,
        selected_source_label: Optional[str],
    ) -> None:
        """[caohy] Task 13：回填 family attempt 的 final_selected / family_selected_label。"""
        family_branch = lifecycle_data.get('trajopt_family_branch')
        if not isinstance(family_branch, dict):
            return
        attempts = family_branch.get('attempts')
        if not isinstance(attempts, list):
            return
        selected_label = str(selected_source_label) if selected_source_label is not None else None
        family_branch['family_selected_label'] = selected_label
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            pool_label = str(
                attempt.get('pool_source_label')
                or attempt.get('raw_pool_source_label')
                or attempt.get('source_label')
            )
            attempt['final_selected'] = bool(
                selected_label is not None
                and pool_label == selected_label
            )

    def _mark_alignment_trajopt_family_topk_attempt_selected(
        self,
        lifecycle_data: dict,
        selected_source_label: Optional[str],
    ) -> None:
        """[caohy] Task 15：回填 1*4 正式替换 attempt 的 final_selected 字段。"""
        topk_branch = lifecycle_data.get('trajopt_family_topk_branch')
        if not isinstance(topk_branch, dict):
            return
        attempts = topk_branch.get('attempts')
        if not isinstance(attempts, list):
            return
        selected_label = str(selected_source_label) if selected_source_label is not None else None
        topk_branch['family_selected_label'] = selected_label
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt['final_selected'] = bool(
                selected_label is not None
                and str(attempt.get('source_label')) == selected_label
            )

    def _sync_alignment_trajopt_family_attempt_pool_status(
        self,
        lifecycle_data: dict,
        lifecycle_candidates: list[dict],
    ) -> None:
        """[caohy] Task 13：按候选池结果回填 family attempt 的入池与选择指标。"""
        family_branch = lifecycle_data.get('trajopt_family_branch')
        if not isinstance(family_branch, dict):
            return
        attempts = family_branch.get('attempts')
        if not isinstance(attempts, list):
            return
        family_source_labels = {
            str(
                attempt.get('pool_source_label')
                or attempt.get('raw_pool_source_label')
                or attempt.get('source_label')
            )
            for attempt in attempts
            if isinstance(attempt, dict) and (
                attempt.get('pool_source_label') is not None
                or attempt.get('raw_pool_source_label') is not None
                or attempt.get('source_label') is not None
            )
        }
        family_candidate_metrics_by_pool_index = {
            int(item['candidate_index']): item
            for item in lifecycle_candidates
            if str(item.get('source_label')) in family_source_labels
        }
        in_pool_labels = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            pool_index = attempt.get('pool_candidate_index')
            if pool_index is None:
                attempt['candidate_pool_accepted'] = False
                attempt['final_selected'] = False
                continue
            candidate_metrics = family_candidate_metrics_by_pool_index.get(int(pool_index))
            if candidate_metrics is None:
                attempt['candidate_pool_accepted'] = False
                attempt['final_selected'] = False
                continue
            attempt['candidate_pool_accepted'] = True
            attempt['final_selected'] = bool(candidate_metrics.get('selected'))
            attempt['selection_metrics'] = {
                'alignment_valid': bool(candidate_metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': candidate_metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': candidate_metrics.get('mean_alignment_deviation_deg'),
                'start_joint_gap_l2': candidate_metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': candidate_metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': candidate_metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': candidate_metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': candidate_metrics.get('twist_smoothness_cost'),
            }
            attempt['selected_candidate_id'] = candidate_metrics.get('candidate_id')
            in_pool_labels.append(
                str(
                    attempt.get('pool_source_label')
                    or attempt.get('raw_pool_source_label')
                    or attempt.get('source_label')
                )
            )
        family_branch['family_in_pool_labels'] = in_pool_labels
        family_branch['family_selected_label'] = next(
            (
                str(
                    attempt.get('pool_source_label')
                    or attempt.get('raw_pool_source_label')
                    or attempt.get('source_label')
                )
                for attempt in attempts
                if isinstance(attempt, dict) and bool(attempt.get('final_selected'))
            ),
            None,
        )

    def _sync_alignment_trajopt_family_topk_attempt_pool_status(
        self,
        lifecycle_data: dict,
        lifecycle_candidates: list[dict],
    ) -> None:
        """[caohy] Task 15：按候选池结果回填 1*4 正式替换 attempt 的入池与选择指标。"""
        topk_branch = lifecycle_data.get('trajopt_family_topk_branch')
        if not isinstance(topk_branch, dict):
            return
        attempts = topk_branch.get('attempts')
        if not isinstance(attempts, list):
            return
        topk_source_labels = {
            str(attempt.get('source_label'))
            for attempt in attempts
            if isinstance(attempt, dict) and attempt.get('source_label') is not None
        }
        topk_candidate_metrics_by_pool_index = {
            int(item['candidate_index']): item
            for item in lifecycle_candidates
            if str(item.get('source_label')) in topk_source_labels
        }
        in_pool_labels = []
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            pool_index = attempt.get('pool_candidate_index')
            if pool_index is None:
                attempt['candidate_pool_accepted'] = False
                attempt['final_selected'] = False
                continue
            candidate_metrics = topk_candidate_metrics_by_pool_index.get(int(pool_index))
            if candidate_metrics is None:
                attempt['candidate_pool_accepted'] = False
                attempt['final_selected'] = False
                continue
            attempt['candidate_pool_accepted'] = True
            attempt['final_selected'] = bool(candidate_metrics.get('selected'))
            attempt['selection_metrics'] = {
                'alignment_valid': bool(candidate_metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': candidate_metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': candidate_metrics.get('mean_alignment_deviation_deg'),
                'start_joint_gap_l2': candidate_metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': candidate_metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': candidate_metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': candidate_metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': candidate_metrics.get('twist_smoothness_cost'),
            }
            attempt['selected_candidate_id'] = candidate_metrics.get('candidate_id')
            in_pool_labels.append(str(attempt.get('source_label')))
        topk_branch['family_in_pool_labels'] = in_pool_labels
        topk_branch['family_selected_label'] = next(
            (
                str(attempt.get('source_label'))
                for attempt in attempts
                if isinstance(attempt, dict) and bool(attempt.get('final_selected'))
            ),
            None,
        )

    def _sync_diffusion_seed_report_with_selection(
        self,
        lifecycle_data: dict,
        lifecycle_candidates: list[dict],
    ) -> None:
        """[caohy] diffusionSeedLearning phase 6：回填 diffusion seed 入池/选择统计。"""
        reports = lifecycle_data.get('seed_provider_reports')
        if not isinstance(reports, dict):
            return
        report = reports.get('diffusion_seed')
        if not isinstance(report, dict):
            return
        diffusion_candidates = [
            item for item in lifecycle_candidates
            if isinstance(item, dict) and str(item.get('source_label', '')).startswith('diffusion_seed_')
        ]
        metrics_by_label = {
            str(item.get('source_label')): item
            for item in diffusion_candidates
        }
        selected_label = None
        for item in diffusion_candidates:
            if bool(item.get('selected')):
                selected_label = str(item.get('source_label'))
                break
        report['entered_pool_count'] = int(len(diffusion_candidates))
        report['entered_pool_labels'] = [str(item.get('source_label')) for item in diffusion_candidates]
        report['selected_source_label'] = selected_label
        report['selected'] = selected_label is not None

        for candidate_record in report.get('candidates', []):
            if not isinstance(candidate_record, dict):
                continue
            source_label = str(candidate_record.get('source_label'))
            metrics = metrics_by_label.get(source_label)
            if metrics is None:
                candidate_record['entered_pool'] = bool(candidate_record.get('entered_pool', False))
                candidate_record['selected'] = False
                continue
            candidate_record['entered_pool'] = True
            candidate_record['selected'] = bool(metrics.get('selected'))
            candidate_record['metrics'] = {
                'alignment_valid': bool(metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': metrics.get('mean_alignment_deviation_deg'),
                'goal_pose_valid': bool(metrics.get('goal_pose_valid')),
                'position_error_m': metrics.get('position_error_m'),
                'orientation_error_deg': metrics.get('orientation_error_deg'),
                'start_joint_gap_l2': metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': metrics.get('twist_smoothness_cost'),
            }
            candidate_record.setdefault('metadata', {})['selected_candidate_id'] = (
                metrics.get('candidate_id')
            )

        for attempt in report.get('optimization_attempts', []):
            if not isinstance(attempt, dict):
                continue
            source_label = str(attempt.get('source_label') or attempt.get('probe_label'))
            metrics = metrics_by_label.get(source_label)
            if metrics is None:
                attempt['final_selected'] = False
                continue
            attempt['candidate_pool_accepted'] = True
            attempt['final_selected'] = bool(metrics.get('selected'))
            attempt['selected_candidate_id'] = metrics.get('candidate_id')
            attempt['selection_metrics'] = {
                'alignment_valid': bool(metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': metrics.get('mean_alignment_deviation_deg'),
                'goal_pose_valid': bool(metrics.get('goal_pose_valid')),
                'position_error_m': metrics.get('position_error_m'),
                'orientation_error_deg': metrics.get('orientation_error_deg'),
                'start_joint_gap_l2': metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': metrics.get('twist_smoothness_cost'),
            }

    def _rank_shadow_candidate_score(self, metrics: dict | None) -> tuple:
        """[caohy] Task 14：按 selector 主口径给 shadow 候选生成一个可比较的排序元组。"""
        metrics = metrics if isinstance(metrics, dict) else {}
        alignment_valid = bool(metrics.get('alignment_valid'))

        def value_or_inf(key: str) -> float:
            value = metrics.get(key)
            try:
                return float(value)
            except (TypeError, ValueError):
                return float('inf')

        return (
            0 if alignment_valid else 1,
            value_or_inf('max_alignment_deviation_deg'),
            value_or_inf('start_joint_gap_l2'),
            value_or_inf('joint_step_max_l2'),
            value_or_inf('joint_step_jump_cost'),
            value_or_inf('twist_smoothness_cost'),
        )

    def _build_alignment_trajopt_family_topk_shadow_collapse_summary(
        self,
        attempts: list[dict],
    ) -> dict:
        """[caohy] Task 14：按整条轨迹差异统计 topk shadow 是否塌缩。"""
        success_attempts = [
            attempt for attempt in attempts
            if isinstance(attempt, dict)
            and bool(attempt.get('success'))
            and attempt.get('trajectory_points') is not None
        ]
        summary = {
            'output_count': int(len(attempts)),
            'successful_output_count': int(len(success_attempts)),
            'unique_output_count': 0,
            'near_duplicate_pair_count': 0,
            'near_duplicate_pairs': [],
            'pairwise_mean_l2_avg': None,
            'pairwise_mean_l2_min': None,
            'pairwise_max_step_l2_max': None,
            'pairwise_terminal_l2_min': None,
            'collapsed': None,
        }
        if len(success_attempts) <= 0:
            summary['unique_output_count'] = 0
            summary['collapsed'] = False
            return summary
        if len(success_attempts) == 1:
            summary['unique_output_count'] = 1
            summary['collapsed'] = False
            return summary

        pairwise_mean_values = []
        pairwise_max_values = []
        pairwise_terminal_values = []
        unique_labels = []
        unique_trajs = []
        duplicate_pairs = []
        for attempt in success_attempts:
            traj = torch.tensor(
                attempt.get('trajectory_points'),
                dtype=torch.float32,
            )
            if traj.ndim == 1:
                traj = traj.unsqueeze(0)
            is_duplicate = False
            for other_label, other_traj in zip(unique_labels, unique_trajs):
                max_t = max(int(traj.shape[0]), int(other_traj.shape[0]))
                traj_pad = traj
                other_pad = other_traj
                if int(traj_pad.shape[0]) < max_t:
                    traj_pad = torch.cat(
                        [traj_pad, traj_pad[-1:].repeat(max_t - int(traj_pad.shape[0]), 1)],
                        dim=0,
                    )
                if int(other_pad.shape[0]) < max_t:
                    other_pad = torch.cat(
                        [other_pad, other_pad[-1:].repeat(max_t - int(other_pad.shape[0]), 1)],
                        dim=0,
                    )
                step_l2 = torch.linalg.norm(traj_pad - other_pad, dim=-1)
                mean_l2 = float(torch.mean(step_l2).item())
                max_l2 = float(torch.max(step_l2).item())
                terminal_l2 = float(torch.linalg.norm(traj_pad[-1] - other_pad[-1]).item())
                pairwise_mean_values.append(mean_l2)
                pairwise_max_values.append(max_l2)
                pairwise_terminal_values.append(terminal_l2)
                if mean_l2 <= 1e-3 and max_l2 <= 1e-3:
                    is_duplicate = True
                    duplicate_pairs.append(
                        {
                            'label_a': other_label,
                            'label_b': str(attempt.get('source_label')),
                            'pairwise_mean_l2': round(mean_l2, 6),
                            'pairwise_max_step_l2': round(max_l2, 6),
                            'pairwise_terminal_l2': round(terminal_l2, 6),
                        }
                    )
            if not is_duplicate:
                unique_labels.append(str(attempt.get('source_label')))
                unique_trajs.append(traj)

        summary['unique_output_count'] = int(len(unique_labels))
        summary['near_duplicate_pair_count'] = int(len(duplicate_pairs))
        summary['near_duplicate_pairs'] = duplicate_pairs
        if pairwise_mean_values:
            summary['pairwise_mean_l2_avg'] = round(
                float(sum(pairwise_mean_values) / len(pairwise_mean_values)), 6,
            )
            summary['pairwise_mean_l2_min'] = round(float(min(pairwise_mean_values)), 6)
        if pairwise_max_values:
            summary['pairwise_max_step_l2_max'] = round(float(max(pairwise_max_values)), 6)
        if pairwise_terminal_values:
            summary['pairwise_terminal_l2_min'] = round(float(min(pairwise_terminal_values)), 6)
        summary['collapsed'] = bool(
            len(unique_labels) <= 1
            or len(duplicate_pairs) >= max(1, len(success_attempts) - 1)
        )
        return summary

    def _simulate_shadow_candidate_winner(
        self,
        base_candidates: list[torch.Tensor],
        base_labels: list[str],
        shadow_candidate: torch.Tensor,
        shadow_label: str,
        start_joint,
        target_pose,
        alignment_tolerance_deg: float,
        strict_level: bool,
    ) -> dict:
        """[caohy] Task 14：把单条 shadow 候选临时加回 selector，做只读复算。"""
        sim_candidates = []
        for candidate in list(base_candidates) + [shadow_candidate]:
            candidate_cpu = candidate.detach().cpu() if hasattr(candidate, 'detach') else candidate
            while candidate_cpu.ndim > 2:
                if candidate_cpu.shape[0] == 1:
                    candidate_cpu = candidate_cpu.squeeze(0)
                else:
                    candidate_cpu = candidate_cpu.reshape(-1, candidate_cpu.shape[-1])
            if candidate_cpu.ndim == 1:
                candidate_cpu = candidate_cpu.unsqueeze(0)
            sim_candidates.append(candidate_cpu)

        max_t = max(int(candidate.shape[0]) for candidate in sim_candidates)
        padded = []
        for candidate in sim_candidates:
            if int(candidate.shape[0]) < max_t:
                candidate = torch.cat(
                    [candidate, candidate[-1:].repeat(max_t - int(candidate.shape[0]), 1)],
                    dim=0,
                )
            padded.append(candidate)
        positions_batch = torch.stack(padded, dim=0).to('cuda:0')

        y_tool = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:0')
        z_neg = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda:0')

        def kin_fn(pos):
            state = CuJointState.from_position(pos, joint_names=self._joint_names)
            kin_state = self._planner.compute_kinematics(state)
            tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
            return SimpleNamespace(ee_quaternion=tool_pose.quaternion)

        level_eval = constraint_utils.evaluate_axis_alignment_batched(
            positions_batch, kin_fn, alignment_tolerance_deg, y_tool, z_neg,
        )
        continuity = constraint_utils.compute_candidate_continuity_metrics(
            positions_batch, start_joint, target_pose[3:7], kin_fn,
        )
        labels = list(base_labels) + [str(shadow_label)]
        selection = constraint_utils.select_level_first_candidate(
            positions_batch, level_eval, continuity, alignment_tolerance_deg, strict_level,
        )
        self._prefer_smoother_alignment_trajopt_candidate(
            selection,
            labels,
            alignment_tolerance_deg,
        )
        selected_index = selection.get('selected_index')
        selected_label = None
        if selected_index is not None and 0 <= int(selected_index) < len(labels):
            selected_label = str(labels[int(selected_index)])
        shadow_index = len(labels) - 1
        return self._round_nested_debug_value(
            {
                'selected_source_label': selected_label,
                'shadow_selected': bool(selected_label == str(shadow_label)),
                'shadow_candidate_metrics': {
                    'alignment_valid': bool(level_eval['alignment_valid'][shadow_index].item()),
                    'max_alignment_deviation_deg': float(
                        level_eval['max_alignment_deviation'][shadow_index].item()
                    ),
                    'mean_alignment_deviation_deg': float(
                        level_eval['mean_alignment_deviation'][shadow_index].item()
                    ),
                    'start_joint_gap_l2': float(
                        continuity['start_joint_gap_l2'][shadow_index].item()
                    ),
                    'joint_step_jump_cost': float(
                        continuity['joint_step_jump_cost'][shadow_index].item()
                    ),
                    'joint_step_max_l2': float(
                        continuity['joint_step_max_l2'][shadow_index].item()
                    ),
                    'joint_step_max_abs': float(
                        continuity['joint_step_max_abs'][shadow_index].item()
                    ),
                    'twist_smoothness_cost': float(
                        continuity['twist_smoothness_cost'][shadow_index].item()
                    ),
                },
                'selection_status': selection.get('planning_status'),
                'alignment_valid_count': selection.get('alignment_valid_count'),
                'candidate_count': selection.get('candidate_count'),
            },
            float_digits=6,
        )

    def _sync_alignment_trajopt_family_topk_shadow_attempt_status(
        self,
        lifecycle_data: dict,
        base_candidates: list[torch.Tensor],
        candidate_source_labels: list[str],
        start_joint,
        target_pose,
        alignment_tolerance_deg: float,
        strict_level: bool,
        actual_selected_source_label: Optional[str] = None,
    ) -> None:
        """[caohy] Task 14：补全 topk shadow attempt 字段、塌缩摘要和理论胜出结论。"""
        branch_record = lifecycle_data.get('trajopt_family_topk_shadow_branch')
        if not isinstance(branch_record, dict):
            return
        attempts = branch_record.get('attempts')
        if not isinstance(attempts, list):
            return

        would_enter_pool_labels = []
        would_win_against_selected_labels = []
        best_shadow_label = None
        best_shadow_score = None
        actual_selected_label = actual_selected_source_label
        if actual_selected_label is None:
            selection_record = lifecycle_data.get('selection')
            if isinstance(selection_record, dict):
                actual_selected_label = selection_record.get('selected_source_label')

        base_labels = [str(label) for label in candidate_source_labels]
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            attempt['candidate_pool_accepted'] = False
            attempt['final_selected'] = False
            attempt['self_collision_summary'] = attempt.get('self_collision_summary') or {
                'status': 'deferred_to_candidate_pool_evaluation',
            }
            attempt['scene_collision_summary'] = attempt.get('scene_collision_summary') or {
                'status': 'deferred_to_candidate_pool_evaluation',
            }
            attempt['topk_shadow_result_summary'] = attempt.get('result_summary')
            attempt['topk_shadow_alignment_summary'] = (
                attempt.get('selection_metrics')
                if isinstance(attempt.get('selection_metrics'), dict)
                else {'status': 'not_available'}
            )
            attempt['topk_shadow_smoothness_summary'] = (
                attempt.get('shadow_trajectory_summary')
                if isinstance(attempt.get('shadow_trajectory_summary'), dict)
                else {'status': 'not_available'}
            )
            attempt['topk_shadow_goal_error_summary'] = (
                attempt.get('terminal_goal_pose_summary')
                if isinstance(attempt.get('terminal_goal_pose_summary'), dict)
                else {'status': 'not_available'}
            )
            if not bool(attempt.get('success')) or attempt.get('trajectory_points') is None:
                attempt['would_enter_pool'] = False
                attempt['would_win_against_selected'] = False
                continue
            would_enter_pool_labels.append(str(attempt.get('source_label')))
            attempt['would_enter_pool'] = True
            shadow_traj = torch.tensor(
                attempt.get('trajectory_points'),
                dtype=torch.float32,
            )
            if shadow_traj.ndim == 1:
                shadow_traj = shadow_traj.unsqueeze(0)
            simulation = self._simulate_shadow_candidate_winner(
                base_candidates,
                base_labels,
                shadow_traj,
                str(attempt.get('source_label')),
                start_joint,
                target_pose,
                alignment_tolerance_deg,
                strict_level,
            )
            attempt['theoretical_selector_simulation'] = simulation
            sim_metrics = simulation.get('shadow_candidate_metrics')
            if isinstance(sim_metrics, dict):
                attempt['theoretical_selection_metrics'] = sim_metrics
                score_tuple = self._rank_shadow_candidate_score(sim_metrics)
                attempt['best_shadow_score_summary'] = {
                    'rank_tuple': self._round_nested_debug_value(list(score_tuple), float_digits=6),
                }
                if best_shadow_score is None or score_tuple < best_shadow_score:
                    best_shadow_score = score_tuple
                    best_shadow_label = str(attempt.get('source_label'))
            if bool(simulation.get('shadow_selected')):
                attempt['would_win_against_selected'] = True
                would_win_against_selected_labels.append(str(attempt.get('source_label')))
            else:
                attempt['would_win_against_selected'] = False
            attempt['actual_selected_source_label'] = actual_selected_label

        branch_record['collapse_summary'] = self._build_alignment_trajopt_family_topk_shadow_collapse_summary(
            attempts,
        )
        branch_record['would_enter_pool_labels'] = would_enter_pool_labels
        branch_record['would_win_against_selected_labels'] = would_win_against_selected_labels
        branch_record['best_shadow_label'] = best_shadow_label
        branch_record['best_shadow_score_summary'] = (
            {'rank_tuple': self._round_nested_debug_value(list(best_shadow_score), float_digits=6)}
            if best_shadow_score is not None else None
        )
        branch_record['actual_selected_source_label'] = actual_selected_label

    def _alignment_shadow_weight_sweep_enabled(self) -> bool:
        """[caohy] Task 36：环境变量开关，避免日常运行默认多跑 10 组 shadow trajopt。"""
        return os.environ.get('CUROBO_ALIGNMENT_SHADOW_WEIGHT_SWEEP', '').strip().lower() in (
            '1',
            'true',
            'yes',
            'on',
        )

    def _get_alignment_shadow_weight_sweep_values(self) -> list[float]:
        """[caohy] Task 36：默认扫描 0.1 到 1.0 十组非终点姿态权重。"""
        raw_values = os.environ.get('CUROBO_ALIGNMENT_SHADOW_WEIGHT_VALUES', '').strip()
        if raw_values:
            values = []
            for item in raw_values.split(','):
                item = item.strip()
                if not item:
                    continue
                try:
                    values.append(float(item))
                except ValueError:
                    self.get_logger().warn(
                        f'Ignoring invalid CUROBO_ALIGNMENT_SHADOW_WEIGHT_VALUES item: {item}'
                    )
            if values:
                return values
        return [round(0.1 * index, 1) for index in range(1, 11)]

    def _apply_shadow_non_terminal_pose_weight(self, weight: float) -> None:
        """[caohy] Task 36：临时提高非终点姿态权重，只服务 shadow sweep，不入正式候选。"""
        terminal_weight = (
            list(self._active_terminal_pose_axes_weight_factor)
            if self._active_terminal_pose_axes_weight_factor is not None
            else [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        )
        non_terminal_weight = [0.0, 0.0, 0.0, float(weight), float(weight), float(weight)]
        criteria = ToolPoseCriteria(
            terminal_pose_axes_weight_factor=terminal_weight,
            non_terminal_pose_axes_weight_factor=non_terminal_weight,
        )
        self._planner.update_tool_pose_criteria({self._tool_frames[0]: criteria})

    def _restore_active_tool_pose_criteria(self) -> None:
        """[caohy] Task 36：shadow sweep 每组求解后恢复正式严格约束配置。"""
        if self._active_tool_pose_criteria:
            self._planner.update_tool_pose_criteria(self._active_tool_pose_criteria)
            return
        self._planner.update_tool_pose_criteria({self._tool_frames[0]: ToolPoseCriteria()})

    def _flatten_joint_position_tensor(self, position_tensor) -> torch.Tensor | None:
        """[caohy] Task 36：把 cuRobo 轨迹张量统一规整成 [T, DOF]，便于 shadow 指标统计。"""
        if position_tensor is None:
            return None
        if hasattr(position_tensor, 'detach'):
            position_tensor = position_tensor.detach().cpu()
        while position_tensor.ndim > 2:
            if position_tensor.shape[0] == 1:
                position_tensor = position_tensor.squeeze(0)
            else:
                position_tensor = position_tensor.reshape(-1, position_tensor.shape[-1])
        if position_tensor.ndim == 1:
            position_tensor = position_tensor.unsqueeze(0)
        return position_tensor

    def _summarize_shadow_trajectory_metrics(self, trajectory: torch.Tensor | None) -> dict:
        """[caohy] Task 36：汇总 shadow 输出轨迹的连续性与退化判定指标。"""
        if trajectory is None:
            return {
                'trajectory_present': False,
                'trajectory_shape': None,
                'joint_step_max_l2': None,
                'joint_path_cost': None,
                'first_last_gap_l2': None,
            }
        traj = self._flatten_joint_position_tensor(trajectory)
        if traj is None:
            return {'trajectory_present': False, 'trajectory_shape': None}
        payload = {
            'trajectory_present': True,
            'trajectory_shape': list(traj.shape),
            'first_joint': [round(float(v), 6) for v in traj[0].tolist()],
            'last_joint': [round(float(v), 6) for v in traj[-1].tolist()],
            'first_last_gap_l2': round(float(torch.linalg.norm(traj[-1] - traj[0]).item()), 6),
        }
        if int(traj.shape[0]) < 2:
            payload.update(
                {
                    'joint_step_max_l2': 0.0,
                    'joint_path_cost': 0.0,
                    'joint_step_max_index': None,
                }
            )
            return payload
        step_l2 = torch.linalg.norm(traj[1:] - traj[:-1], dim=-1)
        max_index = int(torch.argmax(step_l2).item())
        payload.update(
            {
                'joint_step_max_l2': round(float(step_l2[max_index].item()), 6),
                'joint_path_cost': round(float(torch.sum(step_l2).item()), 6),
                'joint_step_max_index': max_index + 1,
            }
        )
        return payload

    def _prefer_smoother_alignment_trajopt_candidate(
        self,
        selection: dict,
        candidate_source_labels: list[str],
        level_tolerance_deg: float,
    ) -> None:
        """[caohy] Task 36：阈值内优先让更平滑的 alignment_seed_trajopt 替代原始 seed。"""
        selected_index = selection.get('selected_index')
        if selected_index is None:
            return
        try:
            selected_index = int(selected_index)
        except (TypeError, ValueError):
            return
        if not (0 <= selected_index < len(candidate_source_labels)):
            return
        if str(candidate_source_labels[selected_index]) != 'alignment_seed':
            return

        alignment_valid = selection.get('candidate_alignment_valid') or []
        max_deviation = selection.get('candidate_max_alignment_deviation') or []
        start_gap = selection.get('candidate_start_joint_gap_l2') or []
        step_max_l2 = selection.get('candidate_joint_step_max_l2') or []
        jump_cost = selection.get('candidate_joint_step_jump_cost') or []
        step_max_abs = selection.get('candidate_joint_step_max_abs') or []
        twist_cost = selection.get('candidate_twist_smoothness_cost') or []

        def value_at(values, index, default=None):
            if index >= len(values):
                return default
            value = values[index]
            return default if value is None else value

        selected_step = value_at(step_max_l2, selected_index)
        if selected_step is None:
            return
        selected_step = float(selected_step)

        best_index = None
        best_score = None
        for index, label in enumerate(candidate_source_labels):
            if not str(label).startswith('alignment_seed_trajopt'):
                continue
            if value_at(alignment_valid, index, False) is not True:
                continue
            candidate_dev = float(value_at(max_deviation, index, float('inf')))
            if candidate_dev > float(level_tolerance_deg):
                continue
            candidate_start_gap = float(value_at(start_gap, index, float('inf')))
            if candidate_start_gap > 0.05:
                continue
            candidate_step = float(value_at(step_max_l2, index, float('inf')))
            if candidate_step >= selected_step:
                continue
            score = (
                candidate_step,
                float(value_at(jump_cost, index, float('inf'))),
                float(value_at(twist_cost, index, float('inf'))),
                candidate_dev,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_index = index

        if best_index is None:
            selection['alignment_seed_trajopt_preference_applied'] = False
            selection['alignment_seed_trajopt_preference_reason'] = 'no_smoother_valid_trajopt_candidate'
            return

        selection['selected_index'] = int(best_index)
        selection['selected_max_alignment_deviation'] = round(
            float(value_at(max_deviation, best_index, 0.0)),
            4,
        )
        selection['selected_start_joint_gap_l2'] = round(
            float(value_at(start_gap, best_index, 0.0)),
            6,
        )
        selection['selected_joint_step_jump_cost'] = round(
            float(value_at(jump_cost, best_index, 0.0)),
            6,
        )
        selection['selected_joint_step_max_abs'] = round(
            float(value_at(step_max_abs, best_index, 0.0)),
            6,
        )
        selection['selected_joint_step_max_l2'] = round(
            float(value_at(step_max_l2, best_index, 0.0)),
            6,
        )
        selection['selected_twist_smoothness_cost'] = round(
            float(value_at(twist_cost, best_index, 0.0)),
            6,
        )
        selection['alignment_seed_trajopt_preference_applied'] = True
        selection['alignment_seed_trajopt_preference_reason'] = (
            'valid_trajopt_candidate_has_smaller_joint_step_max_l2_than_alignment_seed'
        )
        selection['alignment_seed_trajopt_preference_from_index'] = int(selected_index)
        selection['alignment_seed_trajopt_preference_to_index'] = int(best_index)

    def _run_alignment_non_terminal_weight_sweep(
        self,
        goal,
        current_state,
        prepared_seed_traj: torch.Tensor,
        seed_traj: torch.Tensor,
        alignment_tolerance_deg: float,
        probe_label: str,
    ) -> None:
        """[caohy] Task 36：只读扫描非终点姿态权重，评估 cuRobo 软约束能否压住中间水平偏差。"""
        if not self._alignment_shadow_weight_sweep_enabled():
            return
        if probe_label != 'alignment_seed_trajopt':
            return

        weights = self._get_alignment_shadow_weight_sweep_values()
        prepared_seed_flat = prepared_seed_traj[0, 0]
        for weight in weights:
            non_terminal_weight = [0.0, 0.0, 0.0, float(weight), float(weight), float(weight)]
            try:
                self._apply_shadow_non_terminal_pose_weight(float(weight))
                shadow_result = self._solve_alignment_seed_trajopt(
                    goal,
                    current_state,
                    prepared_seed_traj,
                )
            except Exception as exc:
                self.get_logger().warn(
                    '[Task36] alignment_seed_trajopt_shadow_non_terminal_weight_sweep.error: '
                    f'{{"probe_label": {probe_label!r}, "weight": {float(weight)}, '
                    f'"non_terminal_pose_axes_weight_factor": {non_terminal_weight}, '
                    f'"error": {str(exc)!r}}}'
                )
                continue
            finally:
                self._restore_active_tool_pose_criteria()

            base_payload = {
                'probe_label': str(probe_label),
                'weight': round(float(weight), 3),
                'non_terminal_pose_axes_weight_factor': non_terminal_weight,
                'alignment_tolerance_deg': float(alignment_tolerance_deg),
            }
            if not self._result_success(shadow_result):
                self._log_failed_result_summary(
                    'alignment_seed_trajopt_shadow_non_terminal_weight_sweep.solve_pose.failed',
                    shadow_result,
                    extra_info=base_payload,
                )
                continue

            js_result = shadow_result.get_interpolated_plan()
            pos_tensor = self._flatten_joint_position_tensor(getattr(js_result, 'position', None))
            alignment_profile = self._build_alignment_profile_debug(
                pos_tensor,
                alignment_tolerance_deg,
            )
            max_deviation = alignment_profile.get('max_alignment_deviation')
            success_payload = {
                **base_payload,
                'status': self._result_status(shadow_result),
                'alignment_valid': (
                    bool(float(max_deviation) <= float(alignment_tolerance_deg))
                    if max_deviation is not None
                    else None
                ),
                'raw_seed': self._build_alignment_profile_debug(
                    seed_traj,
                    alignment_tolerance_deg,
                ),
                'prepared_seed': self._build_alignment_profile_debug(
                    prepared_seed_flat,
                    alignment_tolerance_deg,
                ),
                'shadow_trajopt_output': alignment_profile,
                **self._summarize_shadow_trajectory_metrics(pos_tensor),
                **self._extract_retained_result_decision_summary(shadow_result),
            }
            self.get_logger().info(
                '[Task36] alignment_seed_trajopt_shadow_non_terminal_weight_sweep.solve_pose.success: '
                f'{success_payload}'
            )

    def _extract_interpolated_plan_debug_summary(self, result) -> dict:
        """提取插值后轨迹摘要，便于 shadow success 直接比较是否真的走起来了。"""
        # [caohy] Task 29：失败诊断也要看插值后关节轨迹，避免只看 optimized_seeds（优化结点）
        # 误把内部 knot（结点轨迹）当成最终 joint trajectory（关节轨迹）。
        payload = {
            'interpolated_present': False,
            'interpolated_shape': None,
            'interpolated_first': None,
            'interpolated_last': None,
        }
        if result is None:
            return payload

        try:
            js_result = result.get_interpolated_plan()
            pos_tensor = getattr(js_result, 'position', None)
            if pos_tensor is None:
                return payload
            if hasattr(pos_tensor, 'detach'):
                pos_tensor = pos_tensor.detach().cpu()
            while pos_tensor.ndim > 2:
                if pos_tensor.shape[0] == 1:
                    pos_tensor = pos_tensor.squeeze(0)
                else:
                    pos_tensor = pos_tensor.reshape(-1, pos_tensor.shape[-1])
            if pos_tensor.ndim == 1:
                pos_tensor = pos_tensor.unsqueeze(0)
            payload['interpolated_present'] = True
            payload['interpolated_shape'] = list(pos_tensor.shape)
            payload['interpolated_first'] = [round(float(v), 6) for v in pos_tensor[0].tolist()]
            payload['interpolated_last'] = [round(float(v), 6) for v in pos_tensor[-1].tolist()]
        except Exception as exc:
            payload['interpolated_error'] = str(exc)
        return payload

    def _extract_joint_state_debug_summary(self, joint_state, prefix: str) -> dict:
        """提取 JointState.position 摘要，统一用于 js_solution / interpolated plan 诊断。"""
        payload = {
            f'{prefix}_present': False,
            f'{prefix}_shape': None,
            f'{prefix}_first': None,
            f'{prefix}_last': None,
            f'{prefix}_step_summary': None,
        }
        if joint_state is None:
            return payload

        pos_tensor = getattr(joint_state, 'position', None)
        if pos_tensor is None:
            return payload

        try:
            if hasattr(pos_tensor, 'detach'):
                pos_tensor = pos_tensor.detach().cpu()
            raw_shape = list(pos_tensor.shape)
            while pos_tensor.ndim > 2:
                if pos_tensor.shape[0] == 1:
                    pos_tensor = pos_tensor.squeeze(0)
                else:
                    pos_tensor = pos_tensor.reshape(-1, pos_tensor.shape[-1])
            if pos_tensor.ndim == 1:
                pos_tensor = pos_tensor.unsqueeze(0)
            payload[f'{prefix}_present'] = True
            payload[f'{prefix}_shape'] = raw_shape
            payload[f'{prefix}_first'] = [round(float(v), 6) for v in pos_tensor[0].tolist()]
            payload[f'{prefix}_last'] = [round(float(v), 6) for v in pos_tensor[-1].tolist()]
            payload[f'{prefix}_step_summary'] = self._summarize_seed_step_metrics(pos_tensor)
        except Exception as exc:
            payload[f'{prefix}_error'] = str(exc)
        return payload

    def _extract_js_solution_debug_summary(self, result) -> dict:
        """提取 trajopt result.js_solution（关节状态轨迹）摘要。"""
        # [caohy] Task 29：result.solution 是 optimized_seeds（优化结点），真正更接近执行轨迹的是
        # result.js_solution（关节状态轨迹），失败排查要把这层一并打出来。
        if result is None:
            return {
                'js_solution_present': False,
                'js_solution_shape': None,
                'js_solution_first': None,
                'js_solution_last': None,
                'js_solution_step_summary': None,
            }
        return self._extract_joint_state_debug_summary(getattr(result, 'js_solution', None), 'js_solution')

    def _run_shadow_jump_preserving_trajopt(
        self,
        goal,
        current_state,
        seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：主 trajopt 失败后追加一次不入池的 jump-preserving shadow solve。"""
        target_horizon = int(seed_prepare_probe.get('action_horizon') or self._planner.trajopt_solver.action_horizon)
        raw_seed_summary = seed_prepare_probe.get('raw_seed_summary') or {}
        jump_preserving_seed = self._resample_seed_traj_jump_preserving(
            seed_traj,
            target_horizon,
            raw_seed_summary.get('max_step_jump_index'),
        )
        current_prepared_seed = self._resample_seed_traj_linear(seed_traj, target_horizon)
        if torch.allclose(jump_preserving_seed, current_prepared_seed, atol=1e-6, rtol=1e-6):
            self._log_probe_info(
                '[S5-B] alignment_seed_trajopt_shadow_jump_preserving.solve_pose.skipped: '
                f'{{"reason": "identical_to_current_prepared", '
                f'"jump_preserving_prepared_summary": {seed_prepare_probe.get("jump_preserving_prepared_summary")}, '
                f'"prepared_seed_summary": {seed_prepare_probe.get("prepared_seed_summary")}}}'
            )
            return

        shadow_prepared_seed = jump_preserving_seed.unsqueeze(0).unsqueeze(0)
        shadow_result = self._solve_alignment_seed_trajopt(goal, current_state, shadow_prepared_seed)
        shadow_extra_info = {
            'start_joint': [round(float(v), 6) for v in start_joint],
            'target_pose': [round(float(v), 6) for v in target_pose],
            'shadow_seed_traj_shape': list(shadow_prepared_seed.shape),
            'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
            'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
            'jump_preserving_prepared_summary': seed_prepare_probe.get('jump_preserving_prepared_summary'),
        }
        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_jump_preserving.solve_pose.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            f'[S5-B] alignment_seed_trajopt_shadow_jump_preserving.solve_pose.success: {success_payload}'
        )

    def _run_shadow_cspace_to_prepared_goal(
        self,
        current_state,
        prepared_seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：用 prepared_seed_last 做显式 joint goal 的只读 cspace shadow。"""
        prepared_seed_flat = prepared_seed_traj[0, 0]
        prepared_goal_joint = prepared_seed_flat[-1]
        goal_state = CuJointState.from_position(
            prepared_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        shadow_result = self._planner.trajopt_solver.solve_cspace(
            goal_state=goal_state,
            current_state=current_state,
            seed_traj=prepared_seed_traj,
            return_seeds=1,
            finetune_attempts=1,
        )
        shadow_extra_info = {
            'start_joint': [round(float(v), 6) for v in start_joint],
            'target_pose': [round(float(v), 6) for v in target_pose],
            'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
            'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
            'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
            'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                prepared_goal_joint,
            ),
            'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                prepared_seed_flat,
            ),
            'cspace_goal_joint': [round(float(v), 6) for v in prepared_goal_joint.tolist()],
            'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
            'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
        }
        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_cspace_to_prepared_goal.solve_cspace.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_js_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            '[S5-B] alignment_seed_trajopt_shadow_cspace_to_prepared_goal.solve_cspace.success: '
            f'{success_payload}'
        )

    def _run_shadow_pose_with_prepared_goal_state(
        self,
        goal,
        current_state,
        prepared_seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：在 solve_pose 上显式补 prepared_seed_last 作为 goal_state，隔离 implicit goal 差异。"""
        prepared_seed_flat = prepared_seed_traj[0, 0]
        prepared_goal_joint = prepared_seed_flat[-1]
        goal_state = CuJointState.from_position(
            prepared_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        shadow_result = self._solve_alignment_seed_trajopt_with_goal_state(
            goal,
            current_state,
            prepared_seed_traj,
            goal_state,
        )
        shadow_extra_info = {
            'start_joint': [round(float(v), 6) for v in start_joint],
            'target_pose': [round(float(v), 6) for v in target_pose],
            'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
            'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
            'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
            'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                prepared_goal_joint,
            ),
            'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                prepared_seed_flat,
            ),
            'pose_goal_state_joint': [round(float(v), 6) for v in prepared_goal_joint.tolist()],
            'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
            'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
        }
        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_pose_with_prepared_goal_state.solve_pose.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_js_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            '[S5-B] alignment_seed_trajopt_shadow_pose_with_prepared_goal_state.solve_pose.success: '
            f'{success_payload}'
        )

    def _run_shadow_pose_with_prepared_goal_state_joint_tracking(
        self,
        goal,
        current_state,
        prepared_seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：只读 shadow，临时打开 joint_position_tracking 后再跑 pose trajopt。"""
        prepared_seed_flat = prepared_seed_traj[0, 0]
        prepared_goal_joint = prepared_seed_flat[-1]
        goal_state = CuJointState.from_position(
            prepared_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        shadow_extra_info = {
            'start_joint': [round(float(v), 6) for v in start_joint],
            'target_pose': [round(float(v), 6) for v in target_pose],
            'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
            'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
            'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
            'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                prepared_goal_joint,
            ),
            'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                prepared_seed_flat,
            ),
            'pose_goal_state_joint': [round(float(v), 6) for v in prepared_goal_joint.tolist()],
            'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
            'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
            # [caohy] Task 29：明确记录这次 shadow 与上一条 shadow 的唯一区别，
            # 避免后续解读时把“传 goal_state”和“打开关节目标跟踪”混为一谈。
            'joint_position_tracking_enabled': True,
        }
        try:
            self._planner.trajopt_solver.enable_joint_position_tracking()
            shadow_result = self._solve_alignment_seed_trajopt_with_goal_state(
                goal,
                current_state,
                prepared_seed_traj,
                goal_state,
            )
        finally:
            self._planner.trajopt_solver.disable_joint_position_tracking()

        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_pose_with_prepared_goal_state_joint_tracking.solve_pose.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_js_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            '[S5-B] alignment_seed_trajopt_shadow_pose_with_prepared_goal_state_joint_tracking.solve_pose.success: '
            f'{success_payload}'
        )

    def _run_shadow_pose_without_implicit_goal(
        self,
        goal,
        current_state,
        prepared_seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：只读 shadow，关闭 implicit goal，隔离隐式终点开关对静止解的影响。"""
        prepared_seed_flat = prepared_seed_traj[0, 0]
        prepared_goal_joint = prepared_seed_flat[-1]
        goal_state = CuJointState.from_position(
            prepared_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        shadow_extra_info = {
            'start_joint': [round(float(v), 6) for v in start_joint],
            'target_pose': [round(float(v), 6) for v in target_pose],
            'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
            'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
            'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
            'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                prepared_goal_joint,
            ),
            'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                prepared_seed_flat,
            ),
            'pose_goal_state_joint': [round(float(v), 6) for v in prepared_goal_joint.tolist()],
            'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
            'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
            'use_implicit_goal': False,
        }
        try:
            shadow_result = self._planner.trajopt_solver.solve_pose(
                goal,
                current_state,
                seed_traj=prepared_seed_traj,
                use_implicit_goal=False,
                goal_state=goal_state,
                return_seeds=1,
                finetune_attempts=1,
            )
        except Exception as exc:
            self._log_probe_info(
                '[S5-B] alignment_seed_trajopt_shadow_pose_without_implicit_goal.solve_pose.error: '
                f'{{"error": {str(exc)!r}, "extra_info": {shadow_extra_info}}}'
            )
            return

        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_pose_without_implicit_goal.solve_pose.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_js_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            '[S5-B] alignment_seed_trajopt_shadow_pose_without_implicit_goal.solve_pose.success: '
            f'{success_payload}'
        )

    def _run_shadow_pose_with_seed_config_goal(
        self,
        goal,
        current_state,
        prepared_seed_traj: torch.Tensor,
        start_joint,
        target_pose,
        seed_prepare_probe: dict,
    ) -> None:
        """[caohy] Task 29：只读 shadow，不传整条 seed_traj，只用 prepared_seed_last 做 seed_config。"""
        try:
            prepared_seed_flat = prepared_seed_traj[0, 0]
            prepared_goal_joint = prepared_seed_flat[-1]
            # [caohy] Task 29：上一轮 shadow 触发 CuRobo 内部 shape mismatch（4 != 1），
            # 这里按 trajopt 默认 num_seeds 复制同一个 seed_config，只验证维度问题，不改变主路径候选池。
            shadow_num_seeds = int(getattr(self._planner.trajopt_solver.config, 'num_seeds', 4) or 4)
            seed_config = (
                prepared_goal_joint.view(1, 1, -1)
                .repeat(1, shadow_num_seeds, 1)
                .to(device='cuda:0', dtype=torch.float32)
            )
            goal_state = CuJointState.from_position(
                prepared_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )
            shadow_extra_info = {
                'start_joint': [round(float(v), 6) for v in start_joint],
                'target_pose': [round(float(v), 6) for v in target_pose],
                'shadow_num_seeds': shadow_num_seeds,
                'seed_config_shape': list(seed_config.shape),
                'seed_config_goal_joint': [round(float(v), 6) for v in prepared_goal_joint.tolist()],
                'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
                'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
                'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
                'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                    prepared_goal_joint,
                ),
                'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                    prepared_seed_flat,
                ),
                'raw_seed_summary': seed_prepare_probe.get('raw_seed_summary'),
                'prepared_seed_summary': seed_prepare_probe.get('prepared_seed_summary'),
            }
            self._log_probe_info(
                '[S5-B] alignment_seed_trajopt_shadow_pose_with_seed_config_goal.enter: '
                f'{shadow_extra_info}'
            )
            shadow_result = self._planner.trajopt_solver.solve_pose(
                goal,
                current_state,
                seed_config=seed_config,
                use_implicit_goal=True,
                goal_state=goal_state,
                return_seeds=1,
                num_seeds=shadow_num_seeds,
                finetune_attempts=1,
            )
        except Exception as exc:
            self._log_probe_info(
                '[S5-B] alignment_seed_trajopt_shadow_pose_with_seed_config_goal.solve_pose.error: '
                f'{{"error": {str(exc)!r}, "extra_info": {shadow_extra_info}}}'
            )
            return

        if shadow_result is None:
            self.get_logger().warn(
                '[S5-B] alignment_seed_trajopt_shadow_pose_with_seed_config_goal.solve_pose.none: '
                f'{shadow_extra_info}'
            )
            return

        if not self._result_success(shadow_result):
            self._log_failed_result_summary(
                'alignment_seed_trajopt_shadow_pose_with_seed_config_goal.solve_pose.failed',
                shadow_result,
                extra_info=shadow_extra_info,
            )
            return

        success_payload = {
            'status': self._result_status(shadow_result),
            **shadow_extra_info,
            **self._extract_solution_debug_summary(shadow_result),
            **self._extract_js_solution_debug_summary(shadow_result),
            **self._extract_interpolated_plan_debug_summary(shadow_result),
        }
        self.get_logger().info(
            '[S5-B] alignment_seed_trajopt_shadow_pose_with_seed_config_goal.solve_pose.success: '
            f'{success_payload}'
        )

    def _optimize_alignment_seed(
        self,
        start_joint,
        target_pose,
        seed_traj,
        alignment_tolerance_deg=3.0,
        probe_label='alignment_seed_trajopt',
    ):
        """基于 alignment seed 轨迹调用底层 trajopt，返回结构化尝试结果。"""
        attempt_record = {
            'probe_label': str(probe_label),
            'success': False,
            'status': 'seed_missing',
            'input_seed_trajectory': self._trajectory_tensor_to_list(seed_traj),
        }
        attempt_record['input_seed_summary'] = self._summarize_trajectory_points(
            attempt_record['input_seed_trajectory'],
        )
        if seed_traj is None:
            return attempt_record

        goal = self._make_goal_tool_pose(target_pose)
        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        prepared_seed_traj = self._prepare_seed_traj_for_trajopt(seed_traj)
        prepared_seed_flat = prepared_seed_traj[0, 0]
        # [caohy] Task 29：当前主问题是“为什么 alignment_seed_trajopt 经常没能入池”，
        # 这里先把送入 trajopt 前的原始/重采样种子摘要和调用输入一起打出来，便于区分是 seed 本身过差、
        # 重采样后退化，还是 solver 返回了带 solution 但 success=false 的失败结果。
        seed_prepare_probe = self._build_seed_prepare_probe_summaries(
            seed_traj,
            prepared_seed_traj,
            self._summarize_seed_step_metrics(seed_traj).get('max_step_jump_index'),
        )
        attempt_record['prepared_seed_trajectory'] = self._trajectory_tensor_to_list(prepared_seed_flat)
        attempt_record['prepared_seed_summary'] = self._summarize_trajectory_points(
            attempt_record['prepared_seed_trajectory'],
        )
        attempt_record['seed_prepare_probe'] = seed_prepare_probe

        # [caohy] Task 22 Phase 3：第一版实验不替换现有 planner candidates，
        # 只额外生成一条基于 alignment_seed 的优化轨迹候选，观察其是否能兼顾对齐与连续性。
        result = self._solve_alignment_seed_trajopt(goal, current_state, prepared_seed_traj)
        attempt_record['result_summary'] = self._build_result_debug_payload(result)
        if not self._result_success(result):
            # [caohy] Task 22：move08/move09 样本里 trajopt 失败时仅返回 unknown，
            # 这里补更多原始结果摘要，便于区分是底层无解、结果为空还是内部提前退出。
            self._log_failed_result_summary(
                'alignment_seed_trajopt.solve_pose.failed',
                result,
                extra_info={
                    'start_joint': [round(float(v), 6) for v in start_joint],
                    'target_pose': [round(float(v), 6) for v in target_pose],
                    'raw_seed_traj_shape': list(seed_traj.shape),
                    'prepared_seed_traj_shape': list(prepared_seed_traj.shape),
                    'prepared_seed_first': [round(float(v), 6) for v in prepared_seed_flat[0].tolist()],
                    'prepared_seed_last': [round(float(v), 6) for v in prepared_seed_flat[-1].tolist()],
                    'prepared_seed_last_joint_limit_summary': self._summarize_joint_limit_violation(
                        prepared_seed_flat[-1],
                    ),
                    'prepared_seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                        prepared_seed_flat,
                    ),
                    **seed_prepare_probe,
                },
            )
            # [caohy] Task 29：主路径失败后追加一次 jump-preserving shadow solve，
            # 只做证据收集，不入候选池、不影响当前选择结果。
            self._run_shadow_jump_preserving_trajopt(
                goal,
                current_state,
                seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            # [caohy] Task 29：再加一条 cspace shadow，直接把 prepared_seed_last 当关节目标，
            # 用来区分“seed 终点本身不可行”还是“pose trajopt / implicit goal 机制有问题”。
            self._run_shadow_cspace_to_prepared_goal(
                current_state,
                prepared_seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            # [caohy] Task 29：再补一条 pose shadow，但显式传入 prepared_seed_last 作为 goal_state，
            # 用来判断当前静止失败更像“缺少显式 joint goal 引导”，还是更底层的 pose trajopt 目标/可行性问题。
            self._run_shadow_pose_with_prepared_goal_state(
                goal,
                current_state,
                prepared_seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            # [caohy] Task 29：再补一条只读 shadow，在显式 goal_state 的基础上临时打开
            # joint_position_tracking（关节目标跟踪），验证当前 pose trajopt 静止失败是否只是
            # 因为 goal_js（显式终点关节）这条代价线默认没有被启用。
            self._run_shadow_pose_with_prepared_goal_state_joint_tracking(
                goal,
                current_state,
                prepared_seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            # [caohy] Task 29：再补两条只读 shadow。第一条关闭 use_implicit_goal（隐式终点），
            # 第二条改用 seed_config（终点关节种子）而不是整条 seed_traj，区分静止解是否来自
            # 隐式终点开关或 seed_traj 被内部控制空间解释后的副作用。
            self._run_shadow_pose_without_implicit_goal(
                goal,
                current_state,
                prepared_seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            self._run_shadow_pose_with_seed_config_goal(
                goal,
                current_state,
                prepared_seed_traj,
                start_joint,
                target_pose,
                seed_prepare_probe,
            )
            self.get_logger().warn(
                f'Alignment seed trajopt failed: {self._result_status(result)}'
            )
            attempt_record['status'] = 'solver_failed'
            attempt_record['failure_reason'] = self._result_status(result)
            return self._round_nested_debug_value(attempt_record, float_digits=6)

        self._log_plan_result_summary('alignment_seed_trajopt.solve_pose', result)
        js_result = result.get_interpolated_plan()
        pos_tensor = js_result.position
        if hasattr(pos_tensor, 'detach'):
            pos_tensor = pos_tensor.detach().cpu()
        while pos_tensor.ndim > 2:
            if pos_tensor.shape[0] == 1:
                pos_tensor = pos_tensor.squeeze(0)
            else:
                pos_tensor = pos_tensor.reshape(-1, pos_tensor.shape[-1])
        if pos_tensor.ndim == 1:
            pos_tensor = pos_tensor.unsqueeze(0)

        flat_first = [round(float(v), 6) for v in pos_tensor[0].tolist()]
        flat_last = [round(float(v), 6) for v in pos_tensor[-1].tolist()]
        self._log_probe_info(
            f'[S5-B] Alignment seed trajopt candidate: flat_shape={list(pos_tensor.shape)}, '
            f'flat_first={flat_first}, flat_last={flat_last}'
        )
        # [caohy] Task 35：三阶段只读对照，判断中间对齐偏差是在 seed 准备阶段、
        # solve_pose 输出阶段，还是后续候选池评估阶段引入。
        stage_probe = {
            'probe_label': str(probe_label),
            'alignment_tolerance_deg': float(alignment_tolerance_deg),
            'raw_seed': self._build_alignment_profile_debug(
                seed_traj,
                alignment_tolerance_deg,
            ),
            'prepared_seed': self._build_alignment_profile_debug(
                prepared_seed_flat,
                alignment_tolerance_deg,
            ),
            'trajopt_output': self._build_alignment_profile_debug(
                pos_tensor,
                alignment_tolerance_deg,
            ),
        }
        self.get_logger().info(
            f'[Task35] alignment_seed_trajopt_stage_alignment_probe: {stage_probe}'
        )
        self._run_alignment_non_terminal_weight_sweep(
            goal,
            current_state,
            prepared_seed_traj,
            seed_traj,
            alignment_tolerance_deg,
            probe_label,
        )
        attempt_record['success'] = True
        attempt_record['status'] = 'success'
        attempt_record['output_trajectory'] = self._trajectory_tensor_to_list(pos_tensor)
        attempt_record['output_summary'] = self._summarize_trajectory_points(
            attempt_record['output_trajectory'],
        )
        attempt_record['stage_alignment_probe'] = stage_probe
        attempt_record['trajectory'] = pos_tensor
        return attempt_record

    def _optimize_alignment_seed_horizon(
        self,
        start_joint,
        target_pose,
        raw_seed_traj,
        alignment_tolerance_deg=3.0,
        probe_label='alignment_seed_horizon_trajopt',
        return_seeds: int = 5,
    ):
        """[caohy] Task 10：对 horizon>1 旁支输出结构化 attempt 记录。"""
        total_outputs = max(1, int(return_seeds))
        bundle_record = {
            'probe_label': str(probe_label),
            'source_label': str(probe_label),
            'success': False,
            'status': 'seed_missing',
            'failure_reason': None,
            'attempt_count': int(total_outputs),
            'input_seed_trajectory': self._trajectory_tensor_to_list(raw_seed_traj),
            'attempts': [],
        }
        bundle_record['input_seed_summary'] = self._summarize_trajectory_points(
            bundle_record['input_seed_trajectory'],
        )
        if raw_seed_traj is None:
            for attempt_index in range(1, total_outputs + 1):
                bundle_record['attempts'].append({
                    'probe_label': f'{probe_label}_{attempt_index}',
                    'source_label': f'{probe_label}_{attempt_index}',
                    'success': False,
                    'status': 'seed_missing',
                    'failure_reason': 'raw_seed_missing',
                })
            return self._round_nested_debug_value(bundle_record, float_digits=6)

        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        prepared_seed_traj = self._prepare_seed_traj_for_trajopt(raw_seed_traj)
        prepared_seed_flat = prepared_seed_traj[0, 0]
        seed_prepare_probe = self._build_seed_prepare_probe_summaries(
            raw_seed_traj,
            prepared_seed_traj,
            self._summarize_seed_step_metrics(raw_seed_traj).get('max_step_jump_index'),
        )
        bundle_record['prepared_seed_trajectory'] = self._trajectory_tensor_to_list(prepared_seed_flat)
        bundle_record['prepared_seed_summary'] = self._summarize_trajectory_points(
            bundle_record['prepared_seed_trajectory'],
        )
        bundle_record['seed_prepare_probe'] = seed_prepare_probe

        try:
            goal, goal_summary = self._make_horizon_goal_tool_pose(target_pose, prepared_seed_flat)
            bundle_record['prepared_goal_sequence_summary'] = goal_summary
            result = self._solve_alignment_seed_horizon_trajopt(
                goal,
                current_state,
                prepared_seed_traj,
                return_seeds=total_outputs,
            )
            bundle_record['result_summary'] = self._build_result_debug_payload(result)
            interpolated_candidates = self._extract_result_interpolated_candidates(result)
            bundle_record['output_candidate_count'] = int(len(interpolated_candidates))
            bundle_record['output_candidate_summaries'] = [
                self._summarize_trajectory_points(
                    self._trajectory_tensor_to_list(candidate),
                )
                for candidate in interpolated_candidates
            ]
            solver_success = self._result_success(result)
            bundle_record['success'] = bool(solver_success)
            bundle_record['status'] = 'solver_success' if solver_success else 'solver_failed'
            bundle_record['failure_reason'] = None if solver_success else self._result_status(result)

            for attempt_index in range(total_outputs):
                source_label = f'{probe_label}_{attempt_index + 1}'
                attempt_record = {
                    'probe_label': source_label,
                    'source_label': source_label,
                    'success': False,
                    'status': 'missing_output_candidate',
                    'failure_reason': None,
                    'input_seed_summary': bundle_record.get('input_seed_summary'),
                    'prepared_seed_summary': bundle_record.get('prepared_seed_summary'),
                    'prepared_goal_sequence_summary': bundle_record.get(
                        'prepared_goal_sequence_summary'
                    ),
                    'seed_prepare_probe': seed_prepare_probe,
                    'result_summary': bundle_record.get('result_summary'),
                    'candidate_rank': int(attempt_index + 1),
                }
                if attempt_index < len(interpolated_candidates):
                    candidate_traj = interpolated_candidates[attempt_index]
                    stage_probe = {
                        'probe_label': source_label,
                        'alignment_tolerance_deg': float(alignment_tolerance_deg),
                        'raw_seed': self._build_alignment_profile_debug(
                            raw_seed_traj,
                            alignment_tolerance_deg,
                        ),
                        'prepared_seed': self._build_alignment_profile_debug(
                            prepared_seed_flat,
                            alignment_tolerance_deg,
                        ),
                        'trajopt_output': self._build_alignment_profile_debug(
                            candidate_traj,
                            alignment_tolerance_deg,
                        ),
                    }
                    attempt_record['success'] = bool(solver_success)
                    attempt_record['status'] = 'success' if solver_success else 'solver_failed'
                    attempt_record['failure_reason'] = None if solver_success else self._result_status(
                        result
                    )
                    attempt_record['output_trajectory'] = self._trajectory_tensor_to_list(candidate_traj)
                    attempt_record['output_summary'] = self._summarize_trajectory_points(
                        attempt_record['output_trajectory'],
                    )
                    attempt_record['stage_alignment_probe'] = stage_probe
                    attempt_record['trajectory'] = candidate_traj
                else:
                    attempt_record['failure_reason'] = (
                        self._result_status(result)
                        if not solver_success
                        else 'candidate_not_returned'
                    )
                bundle_record['attempts'].append(attempt_record)
        except Exception as exc:
            bundle_record['success'] = False
            bundle_record['status'] = 'solver_exception'
            bundle_record['failure_reason'] = str(exc)
            bundle_record['exception_type'] = type(exc).__name__
            for attempt_index in range(total_outputs):
                bundle_record['attempts'].append({
                    'probe_label': f'{probe_label}_{attempt_index + 1}',
                    'source_label': f'{probe_label}_{attempt_index + 1}',
                    'success': False,
                    'status': 'solver_exception',
                    'failure_reason': str(exc),
                    'exception_type': type(exc).__name__,
                    'input_seed_summary': bundle_record.get('input_seed_summary'),
                    'prepared_seed_summary': bundle_record.get('prepared_seed_summary'),
                    'prepared_goal_sequence_summary': bundle_record.get(
                        'prepared_goal_sequence_summary'
                    ),
                    'seed_prepare_probe': bundle_record.get('seed_prepare_probe'),
                })

        return self._round_nested_debug_value(bundle_record, float_digits=6)

    def _generate_alignment_seed_family(
        self,
        start_joint,
        target_pose,
        family_config: dict,
        num_waypoints=30,
        alignment_tolerance_deg=3.0,
        plan_request_index: Optional[int] = None,
    ) -> dict:
        """[caohy] Task 13：按单条 family（家族）配置生成一条独立 alignment seed。"""
        family_config = dict(family_config or {})
        seed_result = self._generate_alignment_seed(
            start_joint,
            target_pose,
            num_waypoints=num_waypoints,
            alignment_tolerance_deg=alignment_tolerance_deg,
            plan_request_index=plan_request_index,
            seed_family_name=family_config.get('seed_family_name'),
            seed_family_config=family_config.get('seed_family_config'),
            twist_schedule_mode=family_config.get('twist_schedule_mode'),
            goal_anchor_rank=family_config.get('goal_anchor_rank'),
            selection_mode=family_config.get('selection_mode'),
        )
        seed_result['source_label'] = str(
            family_config.get('source_label')
            or seed_result.get('source_label')
            or 'alignment_seed_trajopt_family'
        )
        return seed_result

    def _generate_alignment_seed_families(
        self,
        start_joint,
        target_pose,
        num_waypoints=30,
        alignment_tolerance_deg=3.0,
        plan_request_index: Optional[int] = None,
        force_generate: bool = False,
    ) -> dict:
        """[caohy] Task 13：统一生成 alignment trajopt family（多家族对齐种子）骨架。"""
        mode = self._get_alignment_trajopt_family_mode()
        branch_enabled = bool(force_generate or mode != 'off')
        bundle = {
            'branch_mode': mode,
            'branch_enabled': branch_enabled,
            'generation_forced': bool(force_generate and mode == 'off'),
            'success': False,
            'status': 'disabled' if not branch_enabled else 'pending',
            'failure_reason': None,
            'family_success_count': 0,
            'family_seed_generation_success_count': 0,
            'family_in_pool_labels': [],
            'family_selected_label': None,
            'attempts': [],
        }
        if not branch_enabled:
            return bundle

        family_configs = self._get_alignment_trajopt_family_configs()
        bundle['configured_family_count'] = int(len(family_configs))
        for family_index, family_config in enumerate(family_configs):
            source_label = str(
                family_config.get('source_label') or f'alignment_seed_trajopt_{family_index + 1}'
            )
            attempt_record = {
                'attempt_index': int(family_index),
                'candidate_rank': int(family_index + 1),
                'source_label': source_label,
                'probe_label': source_label,
                'raw_pool_source_label': str(
                    family_config.get('raw_pool_source_label')
                    or f'alignment_seed_family_{family_index + 1}'
                ),
                'seed_family_name': family_config.get('seed_family_name'),
                'family_primary_variable': family_config.get('family_primary_variable'),
                'seed_family_config': family_config.get('seed_family_config'),
                'twist_schedule_mode': family_config.get('twist_schedule_mode'),
                'goal_anchor_rank': family_config.get('goal_anchor_rank'),
                'goal_anchor_joint': None,
                'selection_mode': family_config.get('selection_mode'),
                'seed_generation_success': False,
                'trajopt_success': False,
                'candidate_pool_accepted': False,
                'final_selected': False,
                'success': False,
                'status': 'seed_generation_pending',
                'failure_reason': None,
            }
            try:
                seed_result = self._generate_alignment_seed_family(
                    start_joint,
                    target_pose,
                    family_config=family_config,
                    num_waypoints=num_waypoints,
                    alignment_tolerance_deg=alignment_tolerance_deg,
                    plan_request_index=plan_request_index,
                )
                attempt_record['seed_generation_success'] = bool(seed_result.get('success'))
                attempt_record['status'] = (
                    'seed_generation_success'
                    if seed_result.get('success') else 'seed_generation_failed'
                )
                attempt_record['failure_reason'] = seed_result.get('failure_reason')
                attempt_record['goal_anchor_joint'] = seed_result.get('goal_anchor_joint')
                attempt_record['goal_anchor_rank_used'] = seed_result.get('goal_anchor_rank_used')
                attempt_record['seed_result_summary'] = self._round_nested_debug_value(
                    {
                        'ik_fail_count': seed_result.get('ik_fail_count'),
                        'num_waypoints': seed_result.get('num_waypoints'),
                        'start_twist_deg': seed_result.get('start_twist_deg'),
                        'goal_twist_deg': seed_result.get('goal_twist_deg'),
                        'raw_max_step_jump_l2': seed_result.get('raw_max_step_jump_l2'),
                        'raw_max_step_jump_index': seed_result.get('raw_max_step_jump_index'),
                    },
                    float_digits=6,
                )
                attempt_record['input_seed_trajectory'] = self._trajectory_tensor_to_list(
                    seed_result.get('trajectory')
                )
                attempt_record['input_seed_summary'] = self._summarize_trajectory_points(
                    attempt_record['input_seed_trajectory']
                )
                attempt_record['working_seed_trajectory'] = self._trajectory_tensor_to_list(
                    seed_result.get('trajectory')
                )
                attempt_record['working_seed_summary'] = self._summarize_trajectory_points(
                    attempt_record['working_seed_trajectory']
                )
                attempt_record['raw_seed_trajectory'] = self._trajectory_tensor_to_list(
                    seed_result.get('raw_trajectory')
                )
                attempt_record['raw_seed_summary'] = self._summarize_trajectory_points(
                    attempt_record['raw_seed_trajectory']
                )
                if seed_result.get('success'):
                    bundle['family_success_count'] += 1
                    bundle['family_seed_generation_success_count'] += 1
                    attempt_record['_seed_result'] = seed_result
            except Exception as exc:
                attempt_record['status'] = 'seed_generation_exception'
                attempt_record['failure_reason'] = str(exc)
                attempt_record['exception_type'] = type(exc).__name__
            bundle['attempts'].append(attempt_record)

        bundle['success'] = bool(bundle['family_success_count'] > 0)
        bundle['status'] = 'prepared' if bundle['success'] else 'seed_generation_failed'
        if not bundle['success']:
            bundle['failure_reason'] = 'no_family_seed_generated'
        return bundle

    def _optimize_single_alignment_seed_family(
        self,
        start_joint,
        target_pose,
        family_attempt: dict,
        alignment_tolerance_deg=3.0,
    ) -> dict:
        """[caohy] Task 13：对单条 family seed 独立执行一次 trajopt。"""
        attempt_record = dict(family_attempt or {})
        seed_result = attempt_record.pop('_seed_result', None)
        attempt_record['trajopt_success'] = False
        attempt_record['candidate_pool_accepted'] = False
        attempt_record['final_selected'] = False
        attempt_record['success'] = False

        if not bool(attempt_record.get('seed_generation_success')):
            attempt_record['status'] = str(
                attempt_record.get('status') or 'seed_generation_failed'
            )
            return attempt_record
        if seed_result is None or seed_result.get('trajectory') is None:
            attempt_record['status'] = 'seed_runtime_missing'
            attempt_record['failure_reason'] = 'seed_result_missing_for_trajopt'
            return attempt_record

        attempt_record['_seed_result'] = seed_result
        optimized_attempt = self._optimize_alignment_seed(
            start_joint,
            target_pose,
            seed_result.get('trajectory'),
            alignment_tolerance_deg=alignment_tolerance_deg,
            probe_label=str(attempt_record.get('source_label') or attempt_record.get('probe_label')),
        )
        attempt_record['trajopt_success'] = bool(optimized_attempt.get('success'))
        attempt_record['success'] = bool(optimized_attempt.get('success'))
        attempt_record['status'] = (
            'success' if optimized_attempt.get('success') else str(
                optimized_attempt.get('status') or 'trajopt_failed'
            )
        )
        attempt_record['failure_reason'] = optimized_attempt.get('failure_reason')
        attempt_record['result_summary'] = optimized_attempt.get('result_summary')
        attempt_record['prepared_seed_trajectory'] = optimized_attempt.get('prepared_seed_trajectory')
        attempt_record['prepared_seed_summary'] = optimized_attempt.get('prepared_seed_summary')
        attempt_record['seed_prepare_probe'] = optimized_attempt.get('seed_prepare_probe')
        attempt_record['stage_alignment_probe'] = optimized_attempt.get('stage_alignment_probe')
        attempt_record['output_trajectory'] = optimized_attempt.get('output_trajectory')
        attempt_record['output_summary'] = optimized_attempt.get('output_summary')
        if optimized_attempt.get('trajectory') is not None:
            attempt_record['trajectory'] = optimized_attempt.get('trajectory')
        return attempt_record

    def _optimize_alignment_seed_families(
        self,
        start_joint,
        target_pose,
        num_waypoints=30,
        alignment_tolerance_deg=3.0,
        plan_request_index: Optional[int] = None,
    ) -> dict:
        """[caohy] Task 13：先生成 4 条 family seed，再各自独立跑一次 trajopt。"""
        bundle = self._generate_alignment_seed_families(
            start_joint,
            target_pose,
            num_waypoints=num_waypoints,
            alignment_tolerance_deg=alignment_tolerance_deg,
            plan_request_index=plan_request_index,
        )
        if not bundle.get('branch_enabled'):
            return bundle

        optimized_attempts = []
        family_trajopt_success_count = 0
        for family_attempt in bundle.get('attempts', []):
            optimized_attempt = self._optimize_single_alignment_seed_family(
                start_joint,
                target_pose,
                family_attempt,
                alignment_tolerance_deg=alignment_tolerance_deg,
            )
            if optimized_attempt.get('trajopt_success'):
                family_trajopt_success_count += 1
            optimized_attempts.append(optimized_attempt)

        bundle['attempts'] = optimized_attempts
        bundle['family_trajopt_success_count'] = int(family_trajopt_success_count)
        bundle['family_success_count'] = int(family_trajopt_success_count)
        bundle['success'] = bool(family_trajopt_success_count > 0)
        bundle['status'] = 'trajopt_completed' if optimized_attempts else 'seed_generation_failed'
        if not bundle['success'] and bundle.get('family_seed_generation_success_count', 0) > 0:
            bundle['failure_reason'] = 'all_family_trajopt_failed'
        return bundle

    def _build_alignment_seed_family_topk_shadow_inputs(
        self,
        start_joint,
        target_pose,
        family_branch_bundle: dict,
        num_waypoints: int = 30,
        alignment_tolerance_deg: float = 3.0,
        plan_request_index: Optional[int] = None,
    ) -> dict:
        """[caohy] Task 14/15：为 1*4 topk 正式/影子分支准备 4 条输入 seed。

        优先复用旧 family 旁支已经生成好的 seed attempts；若旧 4x1 family 默认关闭，
        则直接按当前 4 组 family 配置现场生成 seed，避免 1*4 仍依赖旧 4x1 输入。
        """
        target_horizon = int(self._planner.trajopt_solver.action_horizon)
        record = {
            'success': False,
            'status': 'pending',
            'failure_reason': None,
            'target_horizon': int(target_horizon),
            'requested_seed_count': 4,
            'seed_count': 0,
            'input_family_labels': [],
            'input_seed_summaries': [],
            'input_seed_source_mode': 'reuse_family_attempts',
            'source_attempts': [],
        }
        bundle = dict(family_branch_bundle or {})
        attempts = bundle.get('attempts')
        if not isinstance(attempts, list) or len(attempts) == 0:
            record['input_seed_source_mode'] = 'direct_family_seed_generation'
            attempts = []
            for family_index, family_config in enumerate(self._get_alignment_trajopt_family_configs()):
                source_label = str(
                    family_config.get('source_label') or f'alignment_seed_trajopt_{family_index + 1}'
                )
                attempt_record = {
                    'attempt_index': int(family_index),
                    'candidate_rank': int(family_index + 1),
                    'source_label': source_label,
                    'probe_label': source_label,
                    'seed_family_name': family_config.get('seed_family_name'),
                    'family_primary_variable': family_config.get('family_primary_variable'),
                    'seed_family_config': family_config.get('seed_family_config'),
                    'twist_schedule_mode': family_config.get('twist_schedule_mode'),
                    'goal_anchor_rank': family_config.get('goal_anchor_rank'),
                    'selection_mode': family_config.get('selection_mode'),
                    'seed_generation_success': False,
                    'trajopt_success': False,
                    'candidate_pool_accepted': False,
                    'final_selected': False,
                    'success': False,
                    'status': 'seed_generation_pending',
                    'failure_reason': None,
                }
                try:
                    seed_result = self._generate_alignment_seed_family(
                        start_joint,
                        target_pose,
                        family_config=family_config,
                        num_waypoints=num_waypoints,
                        alignment_tolerance_deg=alignment_tolerance_deg,
                        plan_request_index=plan_request_index,
                    )
                    attempt_record['seed_generation_success'] = bool(seed_result.get('success'))
                    attempt_record['status'] = (
                        'seed_generation_success'
                        if seed_result.get('success') else 'seed_generation_failed'
                    )
                    attempt_record['failure_reason'] = seed_result.get('failure_reason')
                    attempt_record['goal_anchor_joint'] = seed_result.get('goal_anchor_joint')
                    attempt_record['goal_anchor_rank_used'] = seed_result.get('goal_anchor_rank_used')
                    attempt_record['seed_result_summary'] = self._round_nested_debug_value(
                        {
                            'ik_fail_count': seed_result.get('ik_fail_count'),
                            'num_waypoints': seed_result.get('num_waypoints'),
                            'start_twist_deg': seed_result.get('start_twist_deg'),
                            'goal_twist_deg': seed_result.get('goal_twist_deg'),
                            'raw_max_step_jump_l2': seed_result.get('raw_max_step_jump_l2'),
                            'raw_max_step_jump_index': seed_result.get('raw_max_step_jump_index'),
                        },
                        float_digits=6,
                    )
                    attempt_record['input_seed_trajectory'] = self._trajectory_tensor_to_list(
                        seed_result.get('trajectory')
                    )
                    attempt_record['input_seed_summary'] = self._summarize_trajectory_points(
                        attempt_record['input_seed_trajectory']
                    )
                    attempt_record['working_seed_trajectory'] = self._trajectory_tensor_to_list(
                        seed_result.get('trajectory')
                    )
                    attempt_record['working_seed_summary'] = self._summarize_trajectory_points(
                        attempt_record['working_seed_trajectory']
                    )
                    attempt_record['raw_seed_trajectory'] = self._trajectory_tensor_to_list(
                        seed_result.get('raw_trajectory')
                    )
                    attempt_record['raw_seed_summary'] = self._summarize_trajectory_points(
                        attempt_record['raw_seed_trajectory']
                    )
                    if seed_result.get('success'):
                        attempt_record['_seed_result'] = seed_result
                except Exception as exc:
                    attempt_record['status'] = 'seed_generation_exception'
                    attempt_record['failure_reason'] = str(exc)
                    attempt_record['exception_type'] = type(exc).__name__
                attempts.append(attempt_record)
        record['source_attempts'] = attempts

        prepared_seed_trajs = []
        prepared_seed_configs = []
        for attempt in attempts:
            if len(prepared_seed_trajs) >= int(record['requested_seed_count']):
                break
            if not isinstance(attempt, dict):
                continue
            if not bool(attempt.get('seed_generation_success')):
                continue
            seed_result = attempt.get('_seed_result')
            raw_seed_traj = None
            if isinstance(seed_result, dict):
                raw_seed_traj = seed_result.get('trajectory')
            if raw_seed_traj is None:
                continue
            raw_seed_traj = raw_seed_traj.detach().clone().to(device='cuda:0', dtype=torch.float32)
            prepared_seed = self._resample_seed_traj_linear(raw_seed_traj, target_horizon)
            prepared_seed_trajs.append(prepared_seed)
            prepared_seed_configs.append(prepared_seed[-1].detach().clone())
            record['input_family_labels'].append(str(attempt.get('source_label')))
            record['input_seed_summaries'].append(
                {
                    'source_label': str(attempt.get('source_label')),
                    'seed_family_name': attempt.get('seed_family_name'),
                    'seed_generation_success': bool(attempt.get('seed_generation_success')),
                    'trajopt_success_4x1': bool(attempt.get('trajopt_success')),
                    'input_seed_summary': attempt.get('input_seed_summary'),
                    'prepared_seed_summary': self._summarize_seed_step_metrics(prepared_seed),
                    'goal_anchor_rank': attempt.get('goal_anchor_rank'),
                    'selection_mode': attempt.get('selection_mode'),
                }
            )

        record['seed_count'] = int(len(prepared_seed_trajs))
        if len(prepared_seed_trajs) == 0:
            record['status'] = 'no_reusable_family_seed'
            record['failure_reason'] = 'no_reusable_family_seed'
            return record

        seed_traj_tensor = torch.stack(prepared_seed_trajs, dim=0).unsqueeze(0)
        seed_config_tensor = torch.stack(prepared_seed_configs, dim=0).unsqueeze(0)
        record['success'] = True
        record['status'] = 'prepared'
        record['seed_traj_summary'] = {
            'shape': list(seed_traj_tensor.shape),
            'first_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, 0]),
            'last_seed_summary': self._summarize_seed_step_metrics(seed_traj_tensor[0, -1]),
        }
        record['seed_config_summary'] = {
            'shape': list(seed_config_tensor.shape),
            'first': [round(float(v), 6) for v in seed_config_tensor[0, 0].detach().cpu().tolist()],
            'last': [round(float(v), 6) for v in seed_config_tensor[0, -1].detach().cpu().tolist()],
        }
        record['seed_traj'] = seed_traj_tensor.to(device='cuda:0', dtype=torch.float32)
        record['seed_config'] = seed_config_tensor.to(device='cuda:0', dtype=torch.float32)
        return record

    def _optimize_alignment_seed_families_topk_bundle(
        self,
        start_joint,
        target_pose,
        family_branch_bundle: dict,
        alignment_tolerance_deg: float = 3.0,
        branch_mode: str = 'off',
        bundle_source_label: str = 'alignment_seed_trajopt_topk',
        generation_mode: str = 'alignment_family_topk',
        attempt_label_prefix: str = 'alignment_seed_trajopt_topk',
        is_shadow_run: bool = False,
    ) -> dict:
        """[caohy] Task 15：统一构建 family seed 的 1*4 solve_pose(...) 输出 bundle。"""
        bundle_record = {
            'branch_mode': branch_mode,
            'branch_enabled': branch_mode != 'off',
            'source_label': bundle_source_label,
            'generation_mode': generation_mode,
            'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
            'success': False,
            'status': 'pending',
            'failure_reason': None,
            'candidate_pool_accepted': False,
            'final_selected': False,
            'family_in_pool_labels': [],
            'family_selected_label': None,
            'attempts': [],
        }
        if branch_mode == 'off':
            bundle_record['status'] = 'disabled'
            return bundle_record

        seed_input_record = self._build_alignment_seed_family_topk_shadow_inputs(
            start_joint,
            target_pose,
            family_branch_bundle,
            num_waypoints=30,
            alignment_tolerance_deg=alignment_tolerance_deg,
        )
        bundle_record['input_prepare'] = self._round_nested_debug_value(
            {
                key: val
                for key, val in seed_input_record.items()
                if key not in ('seed_traj', 'seed_config', 'source_attempts')
            },
            float_digits=6,
        )
        if not seed_input_record.get('success'):
            bundle_record['status'] = str(seed_input_record.get('status') or 'input_prepare_failed')
            bundle_record['failure_reason'] = seed_input_record.get('failure_reason')
            bundle_record['attempts'] = self._round_nested_debug_value(
                self._sanitize_alignment_seed_family_bundle(
                    {'attempts': seed_input_record.get('source_attempts', [])}
                ).get('attempts', []),
                float_digits=6,
            )
            return self._round_nested_debug_value(bundle_record, float_digits=6)

        goal = self._make_goal_tool_pose(target_pose)
        current_state = CuJointState.from_position(
            torch.tensor([start_joint], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        total_outputs = int(seed_input_record.get('seed_count') or 0)
        source_attempts = list(seed_input_record.get('source_attempts') or [])
        try:
            result = self._planner.trajopt_solver.solve_pose(
                goal,
                current_state,
                seed_config=seed_input_record.get('seed_config'),
                seed_traj=seed_input_record.get('seed_traj'),
                use_implicit_goal=True,
                return_seeds=int(total_outputs),
                num_seeds=int(total_outputs),
                finetune_attempts=1,
            )
            bundle_record['result_summary'] = self._build_result_debug_payload(result)
            js_solution_candidates = self._extract_result_js_solution_candidates(result)
            interpolated_candidates = self._extract_result_interpolated_candidates(result)
            output_candidates = (
                js_solution_candidates if len(js_solution_candidates) > 0 else interpolated_candidates
            )
            bundle_record['candidate_extract_mode'] = (
                'js_solution'
                if len(js_solution_candidates) > 0 else (
                    'interpolated_plan' if len(interpolated_candidates) > 0 else 'none'
                )
            )
            bundle_record['js_solution_candidate_count'] = int(len(js_solution_candidates))
            bundle_record['interpolated_candidate_count'] = int(len(interpolated_candidates))
            bundle_record['output_candidate_count'] = int(len(output_candidates))
            solver_success = bool(self._result_success(result))
            solver_status = self._result_status(result)
            solver_success_mask = self._extract_result_success_mask(result, total_outputs)
            allow_failed_solver_outputs = self._get_allow_failed_solver_output_candidates()
            output_exists_mask = [
                bool(attempt_index < len(output_candidates))
                for attempt_index in range(total_outputs)
            ]
            strict_successful_output_count = int(sum(
                1 for attempt_index in range(total_outputs)
                if output_exists_mask[attempt_index] and solver_success_mask[attempt_index]
            ))
            failed_solver_output_count = int(sum(
                1 for attempt_index in range(total_outputs)
                if output_exists_mask[attempt_index] and not solver_success_mask[attempt_index]
            ))
            accepted_output_count = int(sum(
                1 for attempt_index in range(total_outputs)
                if output_exists_mask[attempt_index]
                and (
                    solver_success_mask[attempt_index]
                    or allow_failed_solver_outputs
                )
            ))
            bundle_record['successful_output_count'] = accepted_output_count
            bundle_record['accepted_output_count'] = accepted_output_count
            bundle_record['strict_successful_output_count'] = strict_successful_output_count
            bundle_record['failed_solver_output_count'] = failed_solver_output_count
            bundle_record['solver_success_mask'] = list(solver_success_mask)
            bundle_record['allow_failed_solver_output_candidates'] = (
                bool(allow_failed_solver_outputs)
            )
            bundle_record['solver_success'] = solver_success
            bundle_record['solver_status'] = solver_status
            # [caohy] 失败 solver 的 js_solution 只允许诊断保存，默认不再视为可执行候选。
            # 旧行为可用 CUROBO_ALLOW_FAILED_SOLVER_OUTPUT_CANDIDATES=1 显式打开。
            if accepted_output_count <= 0:
                bundle_record['success'] = False
                bundle_record['status'] = (
                    'solver_failed_no_accepted_candidate'
                    if len(output_candidates) > 0 else 'solver_failed'
                )
                bundle_record['failure_reason'] = (
                    'all_output_candidates_reported_failed'
                    if len(output_candidates) > 0 else (solver_status or 'no_output_candidate')
                )
            elif (
                accepted_output_count < total_outputs
                or failed_solver_output_count > 0
                or not solver_success
            ):
                bundle_record['success'] = True
                bundle_record['status'] = 'partial_success'
                bundle_record['failure_reason'] = None
                partial_reasons = []
                if accepted_output_count < total_outputs:
                    partial_reasons.append('accepted_output_count_less_than_requested')
                if failed_solver_output_count > 0:
                    partial_reasons.append(
                        'failed_solver_outputs_allowed_by_debug_switch'
                        if allow_failed_solver_outputs
                        else 'failed_solver_outputs_rejected'
                    )
                if not solver_success:
                    partial_reasons.append(solver_status or 'solver_reported_failed')
                bundle_record['partial_reason'] = ';'.join(partial_reasons) or None
            else:
                bundle_record['success'] = True
                bundle_record['status'] = 'success'
                bundle_record['failure_reason'] = None
            for attempt_index in range(total_outputs):
                source_attempt = source_attempts[attempt_index] if attempt_index < len(source_attempts) else {}
                source_label = str(
                    source_attempt.get('source_label')
                    or f'{attempt_label_prefix}_{attempt_index + 1}'
                )
                attempt_record = {
                    'attempt_index': int(attempt_index),
                    'candidate_rank': int(attempt_index + 1),
                    'source_label': source_label,
                    'probe_label': source_label,
                    'seed_family_name': source_attempt.get('seed_family_name'),
                    'family_primary_variable': source_attempt.get('family_primary_variable'),
                    'seed_family_config': source_attempt.get('seed_family_config'),
                    'twist_schedule_mode': source_attempt.get('twist_schedule_mode'),
                    'goal_anchor_rank': source_attempt.get('goal_anchor_rank'),
                    'goal_anchor_joint': source_attempt.get('goal_anchor_joint'),
                    'goal_anchor_rank_used': source_attempt.get('goal_anchor_rank_used'),
                    'selection_mode': source_attempt.get('selection_mode'),
                    'seed_generation_success': bool(source_attempt.get('seed_generation_success')),
                    'seed_result_summary': source_attempt.get('seed_result_summary'),
                    'input_seed_summary': source_attempt.get('input_seed_summary'),
                    'working_seed_summary': source_attempt.get('working_seed_summary'),
                    'raw_seed_summary': source_attempt.get('raw_seed_summary'),
                    'generation_mode': generation_mode,
                    'topk_shadow_run': bool(is_shadow_run),
                    'topk_formal_run': bool(not is_shadow_run),
                    'topk_shadow_output_rank': int(attempt_index + 1),
                    'topk_shadow_seed_count': int(total_outputs),
                    'topk_shadow_input_family_labels': list(
                        seed_input_record.get('input_family_labels', [])
                    ),
                    'candidate_pool_accepted': False,
                    'final_selected': False,
                    'success': False,
                    'status': 'missing_output_candidate',
                    'failure_reason': None,
                    'self_collision_summary': {'status': 'deferred_to_candidate_pool_evaluation'},
                    'scene_collision_summary': {'status': 'deferred_to_candidate_pool_evaluation'},
                    'result_summary': bundle_record.get('result_summary'),
                }
                output_exists = bool(attempt_index < len(output_candidates))
                solver_seed_success = bool(
                    solver_success_mask[attempt_index]
                    if attempt_index < len(solver_success_mask) else False
                )
                unsafe_failed_solver_output = bool(
                    output_exists
                    and not solver_seed_success
                    and allow_failed_solver_outputs
                )
                attempt_record['output_exists'] = output_exists
                attempt_record['solver_seed_success'] = solver_seed_success
                attempt_record['allow_failed_solver_output_candidates'] = (
                    bool(allow_failed_solver_outputs)
                )
                attempt_record['unsafe_candidate_from_failed_solver'] = (
                    unsafe_failed_solver_output
                )
                if output_exists and (solver_seed_success or unsafe_failed_solver_output):
                    candidate_traj = output_candidates[attempt_index]
                    attempt_record['success'] = True
                    attempt_record['status'] = (
                        'success'
                        if solver_seed_success else 'unsafe_failed_solver_output_allowed'
                    )
                    attempt_record['failure_reason'] = (
                        None if solver_seed_success
                        else 'result_success_false_for_seed_allowed_by_debug_switch'
                    )
                    attempt_record['candidate_extract_mode'] = bundle_record.get(
                        'candidate_extract_mode'
                    )
                    attempt_record['trajectory'] = candidate_traj
                    self._populate_planner_attempt_trajectory_fields(
                        attempt_record,
                        candidate_traj,
                        target_pose,
                        alignment_tolerance_deg=alignment_tolerance_deg,
                        start_joint=start_joint,
                    )
                    attempt_record['shadow_trajectory_summary'] = (
                        self._summarize_shadow_trajectory_metrics(candidate_traj)
                    )
                elif output_exists:
                    candidate_traj = output_candidates[attempt_index]
                    attempt_record['status'] = 'solver_reported_failed'
                    attempt_record['failure_reason'] = 'result_success_false_for_seed'
                    attempt_record['candidate_extract_mode'] = bundle_record.get(
                        'candidate_extract_mode'
                    )
                    attempt_record['rejected_output_summary'] = (
                        self._summarize_trajectory_points(
                            self._trajectory_tensor_to_list(candidate_traj)
                        )
                    )
                    attempt_record['topk_shadow_alignment_summary'] = {
                        'status': 'solver_reported_failed'
                    }
                    attempt_record['topk_shadow_smoothness_summary'] = {
                        'status': 'solver_reported_failed'
                    }
                    attempt_record['topk_shadow_goal_error_summary'] = {
                        'status': 'solver_reported_failed'
                    }
                else:
                    attempt_record['failure_reason'] = (
                        None if self._result_success(result) else self._result_status(result)
                    )
                    attempt_record['topk_shadow_alignment_summary'] = {'status': 'missing_output_candidate'}
                    attempt_record['topk_shadow_smoothness_summary'] = {'status': 'missing_output_candidate'}
                    attempt_record['topk_shadow_goal_error_summary'] = {'status': 'missing_output_candidate'}
                bundle_record['attempts'].append(attempt_record)
        except Exception as exc:
            bundle_record['success'] = False
            bundle_record['status'] = 'solver_exception'
            bundle_record['failure_reason'] = str(exc)
            bundle_record['exception_type'] = type(exc).__name__
            for attempt_index in range(total_outputs):
                bundle_record['attempts'].append(
                    {
                        'attempt_index': int(attempt_index),
                        'candidate_rank': int(attempt_index + 1),
                        'source_label': f'{attempt_label_prefix}_{attempt_index + 1}',
                        'probe_label': f'{attempt_label_prefix}_{attempt_index + 1}',
                        'generation_mode': generation_mode,
                        'topk_shadow_run': bool(is_shadow_run),
                        'topk_formal_run': bool(not is_shadow_run),
                        'topk_shadow_output_rank': int(attempt_index + 1),
                        'topk_shadow_seed_count': int(total_outputs),
                        'topk_shadow_input_family_labels': list(
                            seed_input_record.get('input_family_labels', [])
                        ),
                        'candidate_pool_accepted': False,
                        'final_selected': False,
                        'success': False,
                        'status': 'solver_exception',
                        'failure_reason': str(exc),
                        'exception_type': type(exc).__name__,
                    }
                )
        return bundle_record

    def _optimize_alignment_seed_families_topk_shadow(
        self,
        start_joint,
        target_pose,
        family_branch_bundle: dict,
        alignment_tolerance_deg: float = 3.0,
    ) -> dict:
        """[caohy] Task 14：复用现有 4 条 family seed，一次 solve_pose(...) 跑 1*4 shadow。"""
        return self._optimize_alignment_seed_families_topk_bundle(
            start_joint,
            target_pose,
            family_branch_bundle,
            alignment_tolerance_deg=alignment_tolerance_deg,
            branch_mode=self._get_alignment_trajopt_family_topk_shadow_mode(),
            bundle_source_label='alignment_seed_trajopt_topk_shadow',
            generation_mode='alignment_family_topk_shadow',
            attempt_label_prefix='alignment_seed_trajopt_topk_shadow',
            is_shadow_run=True,
        )

    def _optimize_alignment_seed_families_topk(
        self,
        start_joint,
        target_pose,
        family_branch_bundle: dict,
        alignment_tolerance_deg: float = 3.0,
    ) -> dict:
        """[caohy] Task 15：复用 4 条 family seed，一次 solve_pose(...) 跑正式 1*4 替换分支。"""
        return self._optimize_alignment_seed_families_topk_bundle(
            start_joint,
            target_pose,
            family_branch_bundle,
            alignment_tolerance_deg=alignment_tolerance_deg,
            branch_mode=self._get_alignment_trajopt_family_topk_mode(),
            bundle_source_label='alignment_seed_trajopt_topk',
            generation_mode='alignment_family_topk_formal',
            attempt_label_prefix='alignment_seed_trajopt_topk',
            is_shadow_run=False,
        )

    def _sanitize_alignment_seed_family_bundle(self, bundle: dict) -> dict:
        """[caohy] Task 13：落盘前移除 family / topk 旁支里的运行时对象（tensor / seed_result）。"""
        bundle = dict(bundle or {})
        sanitized_attempts = []
        for attempt in bundle.get('attempts', []):
            sanitized_attempts.append(
                {
                    key: value
                    for key, value in dict(attempt).items()
                    if key not in ('trajectory', '_seed_result')
                }
            )
        bundle['attempts'] = sanitized_attempts
        return bundle

    # [caohy] Task 40：跨构型塌陷检测 — 判断优化器输出是否塌陷在起点附近。
    def _detect_optimizer_collapse(self, result, prepared_seed_traj) -> bool:
        """检测 trajopt result 是否塌陷：success=False 或轨迹总位移远小于期望。"""
        if not self._result_success(result):
            return True
        try:
            js_result = result.get_interpolated_plan()
            pos = js_result.position
            if pos is None:
                return True
            traj = pos.detach().cpu()
            while traj.ndim > 2:
                if traj.shape[0] == 1:
                    traj = traj.squeeze(0)
                else:
                    traj = traj.reshape(-1, traj.shape[-1])
            if traj.shape[0] < 2:
                return True
            total_movement = float(torch.sum(torch.linalg.norm(traj[1:] - traj[:-1], dim=-1)).item())
            seed_flat = prepared_seed_traj[0, 0].detach().cpu()
            expected_movement = float(torch.linalg.norm(seed_flat[-1] - seed_flat[0]).item())
            collapse_threshold = max(0.5, 0.1 * expected_movement)
            return total_movement < collapse_threshold
        except Exception:
            return True

    # [caohy] Task 40 分段优化：把 trajopt result 的插值轨迹展平成 [T, DOF]。
    def _flatten_trajopt_result_position(self, result):
        """从 trajopt result 提取插值后的关节轨迹，展平为 [T, DOF] 的 CPU tensor。失败返回 None。"""
        try:
            js_result = result.get_interpolated_plan()
            pos = js_result.position
            if pos is None:
                return None
            pos = pos.detach().cpu()
            while pos.ndim > 2:
                if pos.shape[0] == 1:
                    pos = pos.squeeze(0)
                else:
                    pos = pos.reshape(-1, pos.shape[-1])
            if pos.ndim == 1:
                pos = pos.unsqueeze(0)
            return pos
        except Exception:
            return None

    def _run_segment_cspace_shadow_probe(
        self,
        seg_start_joint: torch.Tensor,
        seg_goal_joint: torch.Tensor,
        prepared_seed: torch.Tensor,
        probe_label: str,
        *,
        joint_tracking: bool = False,
        replicate_seed_across_all_seeds: bool = False,
        finetune_attempts: int = 1,
    ) -> dict:
        """[caohy] Task 6：在 segment cspace 失败后追加只读 shadow probe。

        这里专门验证两类假设：
        1. solve_cspace（关节空间优化）是否因为自动补充多条非 seed 种子而偏离了可行走廊；
        2. 打开 joint_position_tracking（关节位置跟踪）后，是否能把主收敛过程重新拉回 seed 通道。
        """
        current_state = CuJointState.from_position(
            seg_start_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        goal_state = CuJointState.from_position(
            seg_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        probe_info = {
            'probe_label': str(probe_label),
            'joint_position_tracking_enabled': bool(joint_tracking),
            'replicate_seed_across_all_seeds': bool(replicate_seed_across_all_seeds),
            'finetune_attempts': int(finetune_attempts),
            'segment_start_joint': self._tensor_to_debug_value(seg_start_joint),
            'segment_goal_joint': self._tensor_to_debug_value(seg_goal_joint),
            'prepared_seed_summary': self._summarize_trajectory_points(
                self._trajectory_tensor_to_list(prepared_seed),
            ),
            'prepared_seed_step_metrics': self._summarize_seed_step_metrics(prepared_seed),
        }
        shadow_result = None
        try:
            shadow_seed = prepared_seed.to(device='cuda:0', dtype=torch.float32)
            shadow_num_seeds = int(getattr(self._planner.trajopt_solver.config, 'num_seeds', 4) or 4)
            if replicate_seed_across_all_seeds:
                # [caohy] Task 6：solve_cspace 会自动补 current->goal（起点到终点）的额外种子。
                # 这里把同一条 prepared seed 复制到全部 seed 槽位，隔离“多 seed 竞争带偏”因素，
                # 同时避免直接把 num_seeds 改成 1 触发 CuRobo 内部 shape mismatch。
                shadow_seed_traj = shadow_seed.unsqueeze(0).unsqueeze(0).repeat(1, shadow_num_seeds, 1, 1)
                shadow_num_seeds_arg = shadow_num_seeds
            else:
                shadow_seed_traj = shadow_seed.unsqueeze(0).unsqueeze(0)
                shadow_num_seeds_arg = None
            probe_info['shadow_seed_traj_shape'] = list(shadow_seed_traj.shape)
            probe_info['shadow_num_seeds'] = shadow_num_seeds_arg
            if joint_tracking:
                self._planner.trajopt_solver.enable_joint_position_tracking()
            shadow_result = self._planner.trajopt_solver.solve_cspace(
                goal_state=goal_state,
                current_state=current_state,
                seed_traj=shadow_seed_traj,
                return_seeds=1,
                num_seeds=shadow_num_seeds_arg,
                finetune_attempts=int(finetune_attempts),
            )
        except Exception as exc:
            probe_info['error'] = str(exc)
            probe_info['status'] = 'solve_cspace_exception'
            return self._round_nested_debug_value(probe_info, float_digits=6)
        finally:
            if joint_tracking:
                self._planner.trajopt_solver.disable_joint_position_tracking()

        probe_info['status'] = self._result_status(shadow_result)
        probe_info['success'] = self._result_success(shadow_result)
        if not self._result_success(shadow_result):
            probe_info['result_summary'] = self._build_failed_result_payload(shadow_result)
            js_solution_flat = self._flatten_joint_position_tensor(
                getattr(getattr(shadow_result, 'js_solution', None), 'position', None)
            )
            solver_output_flat = self._flatten_trajopt_result_position(shadow_result)
            if js_solution_flat is not None:
                probe_info['js_solution_vs_prepared_seed'] = self._summarize_trajectory_pair_deviation(
                    prepared_seed,
                    js_solution_flat,
                )
            if solver_output_flat is not None:
                probe_info['solver_output_vs_prepared_seed'] = (
                    self._summarize_trajectory_pair_deviation(
                        prepared_seed,
                        solver_output_flat,
                    )
                )
                if js_solution_flat is not None:
                    probe_info['solver_output_vs_js_solution'] = (
                        self._summarize_trajectory_pair_deviation(
                            js_solution_flat,
                            solver_output_flat,
                        )
                    )
            return self._round_nested_debug_value(probe_info, float_digits=6)

        probe_info.update(self._extract_solution_debug_summary(shadow_result))
        probe_info.update(self._extract_js_solution_debug_summary(shadow_result))
        probe_info.update(self._extract_interpolated_plan_debug_summary(shadow_result))
        js_solution_flat = self._flatten_joint_position_tensor(
            getattr(getattr(shadow_result, 'js_solution', None), 'position', None)
        )
        solver_output_flat = self._flatten_trajopt_result_position(shadow_result)
        if js_solution_flat is not None:
            probe_info['js_solution_vs_prepared_seed'] = self._summarize_trajectory_pair_deviation(
                prepared_seed,
                js_solution_flat,
            )
        if solver_output_flat is not None:
            probe_info['solver_output_vs_prepared_seed'] = (
                self._summarize_trajectory_pair_deviation(
                    prepared_seed,
                    solver_output_flat,
                )
            )
            if js_solution_flat is not None:
                probe_info['solver_output_vs_js_solution'] = (
                    self._summarize_trajectory_pair_deviation(
                        js_solution_flat,
                        solver_output_flat,
                    )
            )
        return self._round_nested_debug_value(probe_info, float_digits=6)

    def _constraint_summary_entry_has_positive(self, constraint_summary, constraint_name: str) -> bool:
        """[caohy] Task 6：快速判断某类约束在摘要里是否真的出现了正违规量。"""
        if not isinstance(constraint_summary, list):
            return False
        for item in constraint_summary:
            if not isinstance(item, dict):
                continue
            if str(item.get('name')) != str(constraint_name):
                continue
            max_value = item.get('max_value')
            try:
                return float(max_value) > 0.0
            except Exception:
                return False
        return False

    # [caohy] Task 40 分段优化：对单段（已知起点关节角 + 终点关节角）做 cspace trajopt。
    # [caohy] Task 6：把 segment（分段）级输入摘要和失败约束拆解一并落盘，
    # 区分“segment seed（分段种子）本身已坏”与“solve_cspace（关节空间求解）阶段推坏”。
    def _solve_segment_cspace(self, seg_start_joint, seg_goal_joint, seg_seed, probe_label='segment'):
        """用 solve_cspace 在两个已知关节配置之间优化一段平滑轨迹。
        seg_seed 为该段的原始 IK 子轨迹 [t, DOF]，用作初始种子。
        返回结构化结果：成功时包含 trajectory（轨迹），失败时包含 result_summary（结果摘要）。"""
        attempt_record = {
            'probe_label': str(probe_label),
            'success': False,
            'status': 'seed_missing',
            'segment_start_joint': self._tensor_to_debug_value(seg_start_joint),
            'segment_goal_joint': self._tensor_to_debug_value(seg_goal_joint),
            'raw_seed_trajectory': self._trajectory_tensor_to_list(seg_seed),
        }
        attempt_record['raw_seed_summary'] = self._summarize_trajectory_points(
            attempt_record['raw_seed_trajectory'],
        )
        attempt_record['raw_seed_step_metrics'] = self._summarize_seed_step_metrics(seg_seed)
        current_state = CuJointState.from_position(
            seg_start_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        goal_state = CuJointState.from_position(
            seg_goal_joint.unsqueeze(0).to(device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        prepared = self._prepare_seed_traj_for_trajopt(seg_seed)
        prepared_flat = prepared[0, 0].detach().cpu()
        attempt_record['prepared_seed_trajectory'] = self._trajectory_tensor_to_list(prepared_flat)
        attempt_record['prepared_seed_summary'] = self._summarize_trajectory_points(
            attempt_record['prepared_seed_trajectory'],
        )
        attempt_record['prepared_seed_step_metrics'] = self._summarize_seed_step_metrics(prepared_flat)
        result = self._planner.trajopt_solver.solve_cspace(
            goal_state=goal_state,
            current_state=current_state,
            seed_traj=prepared,
            return_seeds=1,
            finetune_attempts=1,
        )
        if not self._result_success(result):
            js_solution_flat = self._flatten_joint_position_tensor(
                getattr(getattr(result, 'js_solution', None), 'position', None)
            )
            solver_output_flat = self._flatten_trajopt_result_position(result)
            trajopt_solver = getattr(self._planner, 'trajopt_solver', None)
            metrics_rollout = getattr(trajopt_solver, 'metrics_rollout', None)
            interpolated_rollout = getattr(trajopt_solver, 'additional_metrics_rollouts', {}).get(
                'interpolated_rollout'
            ) if trajopt_solver is not None else None
            # [caohy] Task 6：失败时把 raw seed / prepared seed 直接送进同一组 rollout metrics，
            # 判断是种子自身已经自碰撞，还是 solve_cspace（关节空间优化）收敛过程才把它推坏。
            attempt_record['raw_seed_direct_check'] = self._analyze_trajectory_against_rollout(
                metrics_rollout,
                seg_seed,
                'raw_seed_direct',
                dt_seconds=1.0,
            )
            attempt_record['prepared_seed_direct_check'] = self._analyze_trajectory_against_rollout(
                interpolated_rollout or metrics_rollout,
                prepared_flat,
                'prepared_seed_direct',
                dt_seconds=1.0,
            )
            initial_result_summary = self._build_failed_result_payload(result)
            attempt_record['status'] = 'solver_failed'
            attempt_record['failure_reason'] = self._result_status(result)
            attempt_record['result_summary'] = initial_result_summary
            if js_solution_flat is not None:
                attempt_record['js_solution_vs_raw_seed'] = self._summarize_trajectory_pair_deviation(
                    seg_seed,
                    js_solution_flat,
                )
                attempt_record['js_solution_vs_prepared_seed'] = (
                    self._summarize_trajectory_pair_deviation(
                        prepared_flat,
                        js_solution_flat,
                    )
                )
            if solver_output_flat is not None:
                # [caohy] Task 6：补 solver（求解器）输出与 raw/prepared seed 的逐步偏移摘要，
                # 直接定位是从哪一步开始偏离安全 seed 通道并最终撞进 self_collision（自碰撞）。
                attempt_record['solver_output_vs_raw_seed'] = self._summarize_trajectory_pair_deviation(
                    seg_seed,
                    solver_output_flat,
                )
                attempt_record['solver_output_vs_prepared_seed'] = (
                    self._summarize_trajectory_pair_deviation(
                        prepared_flat,
                        solver_output_flat,
                    )
                )
                if js_solution_flat is not None:
                    attempt_record['solver_output_vs_js_solution'] = (
                        self._summarize_trajectory_pair_deviation(
                        js_solution_flat,
                        solver_output_flat,
                    )
                )
            # [caohy] Task 6：solve_cspace 即使传入 seed_traj（轨迹种子），也会自动补
            # current->goal（起点到终点）的额外种子一起竞争。这里补两条只读 shadow：
            # single_seed 用来验证“是不是多 seed 竞争把主路径带离了可行走廊”；
            # joint_tracking 用来验证“显式关节跟踪能否把收敛过程拉回 seed 通道”。
            attempt_record['shadow_segment_cspace_single_seed'] = (
                self._run_segment_cspace_shadow_probe(
                    seg_start_joint,
                    seg_goal_joint,
                    prepared_flat,
                    f'{probe_label}_shadow_single_seed',
                    replicate_seed_across_all_seeds=True,
                )
            )
            attempt_record['shadow_segment_cspace_single_seed_no_finetune'] = (
                self._run_segment_cspace_shadow_probe(
                    seg_start_joint,
                    seg_goal_joint,
                    prepared_flat,
                    f'{probe_label}_shadow_single_seed_no_finetune',
                    replicate_seed_across_all_seeds=True,
                    finetune_attempts=0,
                )
            )
            attempt_record['shadow_segment_cspace_joint_tracking'] = (
                self._run_segment_cspace_shadow_probe(
                    seg_start_joint,
                    seg_goal_joint,
                    prepared_flat,
                    f'{probe_label}_shadow_joint_tracking',
                    joint_tracking=True,
                )
            )
            return self._round_nested_debug_value(attempt_record, float_digits=6)
        flat_position = self._flatten_trajopt_result_position(result)
        if flat_position is None:
            attempt_record['status'] = 'missing_interpolated_plan'
            attempt_record['failure_reason'] = 'flatten_interpolated_plan_failed'
            return self._round_nested_debug_value(attempt_record, float_digits=6)
        attempt_record['success'] = True
        attempt_record['status'] = 'success'
        attempt_record['output_trajectory'] = self._trajectory_tensor_to_list(flat_position)
        attempt_record['output_summary'] = self._summarize_trajectory_points(
            attempt_record['output_trajectory'],
        )
        attempt_record['output_step_metrics'] = self._summarize_seed_step_metrics(flat_position)
        attempt_record['trajectory'] = flat_position
        return self._round_nested_debug_value(attempt_record, float_digits=6)

    # [caohy] Task 40 分段优化主入口：跨构型种子在跳变点劈成两段，各自 cspace 优化后拼接。
    def _optimize_alignment_seed_split(
        self,
        raw_trajectory,
        raw_jump_index,
        alignment_tolerance_deg=3.0,
        probe_label='alignment_seed_trajopt_split',
        attempt_stage='post_pose_trajopt',
    ):
        """对跨构型 alignment seed 做分段 trajopt，返回结构化尝试结果。
        raw_trajectory: 平滑前的原始 IK 种子 [T, DOF]，每点精确满足对齐。
        raw_jump_index: 构型切换的跳变点索引（该步前后构型不同）。
        每段在单一构型内用 solve_cspace 优化，拼接后校验整体对齐约束。
        成功返回 [T, DOF] CPU tensor，失败/对齐不达标返回 None。"""
        attempt_record = {
            'probe_label': str(probe_label),
            'success': False,
            'status': 'seed_missing',
            'raw_jump_index': int(raw_jump_index) if raw_jump_index is not None else None,
            'attempt_stage': str(attempt_stage),
            'input_seed_trajectory': self._trajectory_tensor_to_list(raw_trajectory),
        }
        attempt_record['input_seed_summary'] = self._summarize_trajectory_points(
            attempt_record['input_seed_trajectory'],
        )
        if raw_trajectory is None or raw_jump_index is None:
            return attempt_record
        traj = raw_trajectory.detach().cpu()
        num_wp = traj.shape[0]
        j = int(raw_jump_index)
        # 跳变点 j：traj[j-1] 是切换前最后一点（构型A末），traj[j] 是切换后第一点（构型B首）。
        # 段A = traj[0 : j]（构型A），段B = traj[j : num_wp]（构型B）。
        # [caohy] Task 40：跳变点极端靠近端点时，对应一段只剩 1~2 点无法 cspace 优化，
        # 此时只优化较长的那一段，短段保留原始 IK 点（仍精确满足对齐），拼接后校验。
        _MIN_SEG = 3  # cspace 优化所需最小段长
        if j < 1 or j > num_wp - 1:
            self.get_logger().warn(
                f'[Task40-split] jump_index={j} at boundary (num_wp={num_wp}), skip split'
            )
            attempt_record['status'] = 'invalid_jump_index'
            attempt_record['failure_reason'] = 'jump_index_at_boundary'
            return self._round_nested_debug_value(attempt_record, float_digits=6)

        seg_a_seed = traj[0:j].clone()
        seg_b_seed = traj[j:num_wp].clone()
        attempt_record['segment_a_seed_length'] = int(seg_a_seed.shape[0])
        attempt_record['segment_b_seed_length'] = int(seg_b_seed.shape[0])
        attempt_record['split_join_index'] = int(j)
        _any_optimized = False

        # 段A：start=traj[0]，goal=traj[j-1]（构型A内的终点）。太短则保留原始点。
        if seg_a_seed.shape[0] >= _MIN_SEG:
            seg_a_attempt = self._solve_segment_cspace(
                traj[0],
                traj[j - 1],
                seg_a_seed,
                probe_label=f'{probe_label}_segment_a',
            )
            attempt_record['segment_a_result'] = {
                key: val for key, val in seg_a_attempt.items()
                if key != 'trajectory'
            }
            seg_a_out = self._to_cpu_tensor(seg_a_attempt.get('trajectory'))
            if seg_a_out is None:
                self.get_logger().warn('[Task40-split] segment A cspace solve failed')
                attempt_record['status'] = 'segment_a_failed'
                attempt_record['failure_reason'] = 'segment_a_cspace_failed'
                return self._round_nested_debug_value(attempt_record, float_digits=6)
            _any_optimized = True
        else:
            seg_a_out = seg_a_seed
            attempt_record['segment_a_result'] = {
                'probe_label': f'{probe_label}_segment_a',
                'success': True,
                'status': 'kept_raw_due_short_segment',
                'raw_seed_summary': self._summarize_trajectory_points(
                    self._trajectory_tensor_to_list(seg_a_seed),
                ),
                'raw_seed_step_metrics': self._summarize_seed_step_metrics(seg_a_seed),
            }
            self.get_logger().info(
                f'[Task40-split] segment A too short ({seg_a_seed.shape[0]} pts), keep raw'
            )
        # 段B：start=traj[j]，goal=traj[num_wp-1]（构型B内的终点 = 全局目标）。太短则保留原始点。
        if seg_b_seed.shape[0] >= _MIN_SEG:
            seg_b_attempt = self._solve_segment_cspace(
                traj[j],
                traj[num_wp - 1],
                seg_b_seed,
                probe_label=f'{probe_label}_segment_b',
            )
            attempt_record['segment_b_result'] = {
                key: val for key, val in seg_b_attempt.items()
                if key != 'trajectory'
            }
            seg_b_out = self._to_cpu_tensor(seg_b_attempt.get('trajectory'))
            if seg_b_out is None:
                self.get_logger().warn('[Task40-split] segment B cspace solve failed')
                attempt_record['status'] = 'segment_b_failed'
                attempt_record['failure_reason'] = 'segment_b_cspace_failed'
                return self._round_nested_debug_value(attempt_record, float_digits=6)
            _any_optimized = True
        else:
            seg_b_out = seg_b_seed
            attempt_record['segment_b_result'] = {
                'probe_label': f'{probe_label}_segment_b',
                'success': True,
                'status': 'kept_raw_due_short_segment',
                'raw_seed_summary': self._summarize_trajectory_points(
                    self._trajectory_tensor_to_list(seg_b_seed),
                ),
                'raw_seed_step_metrics': self._summarize_seed_step_metrics(seg_b_seed),
            }
            self.get_logger().info(
                f'[Task40-split] segment B too short ({seg_b_seed.shape[0]} pts), keep raw'
            )

        if not _any_optimized:
            self.get_logger().warn('[Task40-split] both segments too short, no optimization benefit, skip')
            attempt_record['status'] = 'segments_too_short'
            attempt_record['failure_reason'] = 'both_segments_too_short'
            return self._round_nested_debug_value(attempt_record, float_digits=6)

        # 拼接：段A末点(构型A末) 与 段B首点(构型B首) 之间仍是构型切换的硬跳，
        # 这是物理必然——但现在跳变被压缩成单步，两侧各自连续平滑。
        merged = torch.cat([seg_a_out, seg_b_out], dim=0)

        # 校验整体对齐约束
        alignment = self._build_alignment_profile_debug(merged, alignment_tolerance_deg)
        max_dev = alignment.get('max_alignment_deviation', 999.0)
        # 计算拼接处的跳变
        join_idx = seg_a_out.shape[0]
        join_jump = float(torch.linalg.norm(merged[join_idx] - merged[join_idx - 1]).item())
        attempt_record['join_jump_l2'] = round(join_jump, 6)
        attempt_record['segment_a_output_length'] = int(seg_a_out.shape[0])
        attempt_record['segment_b_output_length'] = int(seg_b_out.shape[0])
        self.get_logger().info(
            f'[Task40-split] merged shape={list(merged.shape)}, '
            f'segA={seg_a_out.shape[0]}, segB={seg_b_out.shape[0]}, '
            f'join_jump_l2={join_jump:.4f}@wp{join_idx}, max_alignment_dev={max_dev:.4f} deg'
        )
        if max_dev > alignment_tolerance_deg:
            self.get_logger().warn(
                f'[Task40-split] rejected: max_alignment_dev={max_dev:.4f} > tol={alignment_tolerance_deg}'
            )
            attempt_record['status'] = 'alignment_rejected'
            attempt_record['failure_reason'] = 'max_alignment_deviation_exceeds_tolerance'
            attempt_record['alignment_summary'] = alignment
            return self._round_nested_debug_value(attempt_record, float_digits=6)
        attempt_record['success'] = True
        attempt_record['status'] = 'success'
        attempt_record['alignment_summary'] = alignment
        attempt_record['output_trajectory'] = self._trajectory_tensor_to_list(merged)
        attempt_record['output_summary'] = self._summarize_trajectory_points(
            attempt_record['output_trajectory'],
        )
        attempt_record['trajectory'] = merged
        return attempt_record

    def _summarize_seed_step_metrics(self, seed_traj: torch.Tensor) -> dict:
        """统计 seed 轨迹的最大单步跳变，便于比较平滑前后差异。"""
        if seed_traj is None or seed_traj.ndim != 2 or seed_traj.shape[0] < 2:
            return {
                'max_step_jump_l2': 0.0,
                'max_step_max_abs': 0.0,
                'max_step_jump_index': None,
            }

        step_delta = seed_traj[1:] - seed_traj[:-1]
        step_l2 = torch.linalg.norm(step_delta, dim=-1)
        best_idx = int(torch.argmax(step_l2).item())
        best_delta = step_delta[best_idx]
        return {
            'max_step_jump_l2': round(float(step_l2[best_idx].item()), 6),
            'max_step_max_abs': round(float(torch.max(torch.abs(best_delta)).item()), 6),
            'max_step_jump_index': best_idx + 1,
        }

    def _smooth_seed_traj_for_trajopt(self, seed_traj: torch.Tensor, passes: int = 2) -> torch.Tensor:
        """对大跳 seed 做最小时间域平滑，仅用于 trajopt 初始轨迹实验。"""
        if seed_traj is None:
            raise ValueError('seed_traj is None')
        if seed_traj.ndim != 2:
            raise ValueError(f'seed_traj must be 2D [T, DOF], got shape={list(seed_traj.shape)}')
        if seed_traj.shape[0] < 3:
            return seed_traj.detach().clone()

        smoothed = seed_traj.detach().clone().to(dtype=torch.float32)
        # [caohy] Task 22：只做最小时间域平滑实验，不改端点、不改候选筛选逻辑；
        # 目标是验证“大跳 seed 是否会直接导致 trajopt 候选缺失”。
        for _ in range(max(int(passes), 1)):
            prev_pts = smoothed[:-2]
            curr_pts = smoothed[1:-1]
            next_pts = smoothed[2:]
            smoothed[1:-1] = 0.25 * prev_pts + 0.5 * curr_pts + 0.25 * next_pts

        smoothed[0] = seed_traj[0]
        smoothed[-1] = seed_traj[-1]
        return smoothed

    def _bridge_seed_jump_for_trajopt(
        self,
        seed_traj: torch.Tensor,
        jump_index: Optional[int],
        bridge_radius: int = 2,
    ) -> torch.Tensor:
        """围绕最大跳变点做局部桥接插值，仅用于 trajopt 初始轨迹实验。"""
        if seed_traj is None:
            raise ValueError('seed_traj is None')
        if seed_traj.ndim != 2:
            raise ValueError(f'seed_traj must be 2D [T, DOF], got shape={list(seed_traj.shape)}')
        if jump_index is None or seed_traj.shape[0] < 4:
            return seed_traj.detach().clone()

        bridged = seed_traj.detach().clone().to(dtype=torch.float32)
        jump_index = int(jump_index)
        start_anchor = max(0, jump_index - max(int(bridge_radius), 1) - 1)
        end_anchor = min(seed_traj.shape[0] - 1, jump_index + max(int(bridge_radius), 1))
        if end_anchor - start_anchor < 2:
            return bridged

        start_pt = bridged[start_anchor].clone()
        end_pt = bridged[end_anchor].clone()
        span = end_anchor - start_anchor
        # [caohy] Task 22：如果最大跳变集中在局部 waypoint（路径点）附近，
        # 再补一条“局部桥接”实验 seed，验证 trajopt 是否主要受这一处 branch jump（分支跳变）影响。
        for offset in range(1, span):
            alpha = float(offset) / float(span)
            bridged[start_anchor + offset] = (1.0 - alpha) * start_pt + alpha * end_pt

        bridged[0] = seed_traj[0]
        bridged[-1] = seed_traj[-1]
        return bridged

    # [caohy] Phase 9.8：alignment seed 生成。位置线性插值 + twist 连续插值 + 逐点 IK。
    def _generate_alignment_seed(
        self,
        start_joint,
        target_pose,
        num_waypoints=30,
        alignment_tolerance_deg=3.0,
        plan_request_index: Optional[int] = None,
        seed_family_name: Optional[str] = None,
        seed_family_config: Optional[dict] = None,
        twist_schedule_mode: Optional[str] = None,
        goal_anchor_rank: Optional[int] = None,
        selection_mode: Optional[str] = None,
    ):
        """生成保持对齐的 seed 轨迹。

        Args:
            start_joint: 起始关节角。
            target_pose: 目标位姿 [x,y,z,qw,qx,qy,qz]。
            num_waypoints: waypoint 数量。
            alignment_tolerance_deg: 对齐容差（度）。

        Returns:
            dict: 包含 trajectory, success, ik_fail_count 等。
        """
        effective_seed_family_name = str(seed_family_name or 'baseline_default')
        effective_seed_family_config = dict(seed_family_config or {})
        effective_twist_schedule_mode = str(
            twist_schedule_mode or effective_seed_family_config.get('twist_schedule_mode') or 'uniform_shortest'
        ).strip().lower()
        try:
            effective_goal_anchor_rank = max(
                1,
                int(goal_anchor_rank if goal_anchor_rank is not None else effective_seed_family_config.get('goal_anchor_rank', 1)),
            )
        except Exception:
            effective_goal_anchor_rank = 1
        effective_selection_mode = selection_mode or effective_seed_family_config.get('selection_mode')

        # 端点对齐检查
        fk_start = self._fk_single(start_joint)
        start_quat = fk_start[3:7]
        target_quat = target_pose[3:7]

        endpoint_check = constraint_utils.check_alignment_endpoints(
            start_quat, target_quat, alignment_tolerance_deg,
        )
        if not endpoint_check['valid']:
            self.get_logger().warn(f'Seed skip: {endpoint_check["failure_reason"]}')
            return {
                'trajectory': None,
                'success': False,
                'ik_fail_count': 0,
                'failure_reason': endpoint_check['failure_reason'],
                'seed_family_name': effective_seed_family_name,
                'seed_family_config': effective_seed_family_config,
                'twist_schedule_mode': effective_twist_schedule_mode,
                'goal_anchor_rank': int(effective_goal_anchor_rank),
                'selection_mode': effective_selection_mode,
                'goal_anchor_joint': None,
            }

        # 起始位置通过 FK 获取
        pos_start = torch.tensor(fk_start[:3], dtype=torch.float32, device='cpu')
        pos_goal = torch.tensor(target_pose[:3], dtype=torch.float32, device='cpu')

        # 计算 twist
        quat_goal = constraint_utils.normalize_quaternion(list(target_pose[3:7]))
        start_twist_deg = constraint_utils.extract_twist_deg_relative_to_goal(start_quat, quat_goal)
        goal_twist_deg = 0.0

        # [caohy] Task 40：预求目标 IK，为后续逐 waypoint 选支提供 goal 锚定。
        # 先对 goal pose 求解 IK，得到与 start_joint 最近的目标关节构型；
        # 后续 waypoint 选支时按参数 t 渐增引导候选向此构型过渡，防止最后一步爆跳。
        goal_joint_anchor = None
        goal_joint_anchor_rank_used = None
        try:
            goal_pos_for_ik = torch.tensor(
                target_pose[:3], dtype=torch.float32, device='cuda:0',
            ).view(1, 1, 1, 1, 3)
            goal_quat_for_ik = torch.tensor(
                list(target_pose[3:7]), dtype=torch.float32, device='cuda:0',
            ).view(1, 1, 1, 1, 4)
            goal_pose_for_ik = GoalToolPose(
                tool_frames=self._tool_frames,
                position=goal_pos_for_ik,
                quaternion=goal_quat_for_ik,
            )
            start_tensor = torch.tensor(start_joint, dtype=torch.float32, device='cuda:0')
            goal_ik_current_state = CuJointState.from_position(
                start_tensor.view(1, -1), joint_names=self._joint_names,
            )
            goal_ik_seed_config = start_tensor.view(1, 1, -1)
            self._ik_solver.reset_seed()
            goal_ik_result = self._ik_solver.solve_pose(
                goal_pose_for_ik,
                current_state=goal_ik_current_state,
                seed_config=goal_ik_seed_config,
                return_seeds=32,
            )
            if goal_ik_result.feasible.any():
                gf_batch_idx, gf_seed_idx = goal_ik_result.feasible.nonzero(as_tuple=True)
                goal_feasible = goal_ik_result.solution[gf_batch_idx, gf_seed_idx]
                goal_wrapped = torch.stack([
                    self._wrap_seed_solution_to_prev(start_tensor, sol)
                    for sol in goal_feasible
                ], dim=0)
                goal_deltas_l2 = torch.linalg.norm(goal_wrapped - start_tensor.unsqueeze(0), dim=-1)
                goal_sorted_indices = torch.argsort(goal_deltas_l2)
                goal_rank_index = min(
                    max(int(effective_goal_anchor_rank) - 1, 0),
                    max(int(goal_sorted_indices.shape[0]) - 1, 0),
                )
                goal_best_idx = int(goal_sorted_indices[goal_rank_index].item())
                goal_joint_anchor_rank_used = int(goal_rank_index + 1)
                goal_joint_anchor = goal_wrapped[goal_best_idx].detach()
                self.get_logger().info(
                    f'Goal IK anchor resolved: feasible={int(goal_feasible.shape[0])}, '
                    f'anchor_rank={goal_joint_anchor_rank_used}, '
                    f'best_delta_l2={float(goal_deltas_l2[goal_best_idx]):.4f}, '
                    f'anchor={[round(float(v), 4) for v in goal_joint_anchor.cpu().tolist()]}'
                )
        except Exception as e:
            self.get_logger().warn(f'Goal IK pre-solve failed: {e}')

        # 逐点生成
        joint_trajectory = []
        ik_fail_count = 0
        prev_solution = torch.tensor(start_joint, dtype=torch.float32, device='cuda:0')
        prev_step_delta = None
        recent_step_l2_history = []
        seed_waypoint_debug = []
        max_step_jump_l2 = 0.0
        max_step_jump_index = None
        max_step_jump_source = None
        max_step_jump_delta = []
        # [caohy] Task 28：允许按 waypoint 索引抓取一次“进程内原始快照”，
        # 直接记录送进 IKSolver 的精确张量和返回候选签名，
        # 用来对比“独立探针重建输入”和“节点真实运行态输入”是否真的一致。
        debug_snapshot_waypoint = None
        debug_snapshot_waypoint_text = os.environ.get('CUROBO_DEBUG_SNAPSHOT_WAYPOINT')
        debug_snapshot_move_index = None
        debug_snapshot_move_index_text = os.environ.get('CUROBO_DEBUG_SNAPSHOT_MOVE_INDEX')
        if debug_snapshot_waypoint_text not in (None, ''):
            try:
                debug_snapshot_waypoint = int(debug_snapshot_waypoint_text)
            except ValueError:
                self.get_logger().warn(
                    f'Invalid CUROBO_DEBUG_SNAPSHOT_WAYPOINT={debug_snapshot_waypoint_text}, expected integer waypoint index.'
                )
        if debug_snapshot_move_index_text not in (None, ''):
            try:
                debug_snapshot_move_index = int(debug_snapshot_move_index_text)
            except ValueError:
                self.get_logger().warn(
                    f'Invalid CUROBO_DEBUG_SNAPSHOT_MOVE_INDEX={debug_snapshot_move_index_text}, expected integer move index.'
                )

        for i in range(num_waypoints):
            t = i / max(num_waypoints - 1, 1)
            if i == 0:
                joint_trajectory.append(prev_solution.clone())
                seed_waypoint_debug.append({
                    'index': 0,
                    't': round(float(t), 6),
                    'source': 'start_joint',
                    'ik_success': True,
                    'fallback_used': False,
                    'step_jump_l2': 0.0,
                    'step_max_abs': 0.0,
                    'step_delta': [0.0] * len(start_joint),
                })
                continue

            # 位置线性插值
            pos_i = (pos_start * (1 - t) + pos_goal * t).unsqueeze(0).to('cuda:0')

            # twist 连续插值
            twist_progress_t = float(t)
            if effective_twist_schedule_mode == 'delayed_to_goal':
                # [caohy] Task 13：delayed_to_goal（后段收扭转）先压慢前半段 twist 变化，
                # 让中前段更贴近起点扭转，到后半段再加快靠近 goal_twist。
                twist_progress_t = float(t * t) if t <= 0.5 else float(1.0 - 2.0 * (1.0 - t) * (1.0 - t))
            twist_i_deg = constraint_utils.interpolate_angle_shortest_deg(
                start_twist_deg,
                goal_twist_deg,
                twist_progress_t,
            )
            quat_i = constraint_utils.compose_goal_relative_twist_quaternion(quat_goal, twist_i_deg)

            # IKSolver.solve_pose() 需要 GoalToolPose 5D [B,H,L,G,3/4]，
            # 单点 seed 规划固定使用 batch=1, horizon=1, link=1, goalset=1。
            goal_pose = GoalToolPose(
                tool_frames=self._tool_frames,
                position=pos_i.view(1, 1, 1, 1, 3),
                quaternion=torch.tensor(
                    quat_i, device='cuda:0', dtype=torch.float32,
                ).view(1, 1, 1, 1, 4),
            )

            # IK 求解
            current_state = CuJointState.from_position(
                prev_solution.view(1, -1), joint_names=self._joint_names,
            )
            seed_config = prev_solution.view(1, 1, -1)
            fallback_used = False
            source = 'ik_solution'
            debug_snapshot = None
            snapshot_move_match = (
                debug_snapshot_move_index is None
                or plan_request_index is None
                or int(debug_snapshot_move_index) == int(plan_request_index)
            )
            if (
                debug_snapshot_waypoint is not None
                and snapshot_move_match
                and int(i) == int(debug_snapshot_waypoint)
            ):
                debug_snapshot = {
                    'snapshot_move_index': int(plan_request_index) if plan_request_index is not None else None,
                    'snapshot_waypoint_index': int(i),
                    'current_state_position': [
                        round(float(v), 9)
                        for v in current_state.position.reshape(-1).detach().cpu().tolist()
                    ],
                    'seed_config_position': [
                        round(float(v), 9)
                        for v in seed_config.reshape(-1).detach().cpu().tolist()
                    ],
                    'goal_pose_position': [
                        round(float(v), 9)
                        for v in goal_pose.position.reshape(-1).detach().cpu().tolist()
                    ],
                    'goal_pose_quaternion': [
                        round(float(v), 9)
                        for v in goal_pose.quaternion.reshape(-1).detach().cpu().tolist()
                    ],
                }
            try:
                # [caohy] Task 22 阶段1：为区分“求解器随机性”与“上下文/数值路径差异”，
                # 允许通过环境变量在每次单点 IK 前固定 torch 随机种子，便于做最小对照实验。
                forced_ik_seed = os.environ.get('CUROBO_DEBUG_FORCE_IK_SEED')
                if forced_ik_seed is not None:
                    seed_value = int(forced_ik_seed)
                    torch.manual_seed(seed_value)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(seed_value)
                # [caohy] Task 22 阶段1：单 waypoint 探针确认同一 IKSolver 实例重复调用会漂，
                # 而每次调用前 reset_seed() 可以稳定可行 IK 集；
                # 这里先做最小集成验证，不改其它选解逻辑，只在单点 IK 求解前重置内部 seed 状态。
                self._ik_solver.reset_seed()
                ik_result = self._ik_solver.solve_pose(
                    goal_pose,
                    current_state=current_state,
                    seed_config=seed_config,
                    return_seeds=32,
                )
                if ik_result.feasible.any():
                    feasible_batch_idx, feasible_seed_idx = ik_result.feasible.nonzero(as_tuple=True)
                    feasible_solutions = ik_result.solution[feasible_batch_idx, feasible_seed_idx]
                    candidate_pose_goal_summaries = self._summarize_pose_against_goal(
                        feasible_solutions,
                        pos_i.reshape(-1),
                        quat_i,
                        expected_twist_deg=twist_i_deg,
                    )
                    if debug_snapshot is not None:
                        raw_signature_rows = []
                        for raw_idx in range(int(feasible_solutions.shape[0])):
                            wrapped_solution = self._wrap_seed_solution_to_prev(
                                prev_solution, feasible_solutions[raw_idx],
                            )
                            raw_signature_rows.append({
                                'candidate_index': raw_idx,
                                'seed_index': int(feasible_seed_idx[raw_idx].item()),
                                'raw_joint_position': [
                                    round(float(v), 9)
                                    for v in feasible_solutions[raw_idx].detach().cpu().tolist()
                                ],
                                'wrapped_joint_position': [
                                    round(float(v), 9)
                                    for v in wrapped_solution.detach().cpu().tolist()
                                ],
                                'pose_goal_summary': candidate_pose_goal_summaries[raw_idx],
                            })
                        debug_snapshot['raw_feasible_candidate_count'] = int(feasible_solutions.shape[0])
                        debug_snapshot['raw_feasible_signature_rows'] = raw_signature_rows
                    forced_branch_candidate_index = None
                    forced_branch_move_waypoint_spec = os.environ.get(
                        'CUROBO_DEBUG_FORCE_BRANCH_MOVE_WAYPOINT'
                    )
                    if forced_branch_move_waypoint_spec:
                        # [caohy] Task 29：新增“按动作 + waypoint 精准强制选支”，
                        # 避免旧的全局 waypoint 开关污染前后其它动作。
                        try:
                            move_text, waypoint_text, candidate_text = (
                                forced_branch_move_waypoint_spec.split(':', 2)
                            )
                            if (
                                plan_request_index is not None
                                and int(move_text) == int(plan_request_index)
                                and int(waypoint_text) == int(i)
                            ):
                                forced_branch_candidate_index = int(candidate_text)
                        except ValueError:
                            self.get_logger().warn(
                                'Invalid CUROBO_DEBUG_FORCE_BRANCH_MOVE_WAYPOINT='
                                f'{forced_branch_move_waypoint_spec}, expected <move>:<waypoint>:<candidate>'
                            )
                    forced_branch_waypoint_spec = os.environ.get('CUROBO_DEBUG_FORCE_BRANCH_WAYPOINT')
                    if forced_branch_candidate_index is None and forced_branch_waypoint_spec:
                        # [caohy] Task 22 阶段1：为了验证“近似平分候选到底是不是选错支”，
                        # 允许在指定 waypoint 上强制选某个候选支，然后观察后续整段 seed 真实滚动结果。
                        try:
                            forced_waypoint_text, forced_candidate_text = forced_branch_waypoint_spec.split(':', 1)
                            if int(forced_waypoint_text) == int(i):
                                forced_branch_candidate_index = int(forced_candidate_text)
                        except ValueError:
                            self.get_logger().warn(
                                f'Invalid CUROBO_DEBUG_FORCE_BRANCH_WAYPOINT={forced_branch_waypoint_spec}, expected <waypoint>:<candidate>'
                            )
                    selection_override_mode = None
                    selection_override_mode_text = os.environ.get(
                        'CUROBO_DEBUG_BRANCH_SELECTION_MODE'
                    )
                    selection_override_move_index_text = os.environ.get(
                        'CUROBO_DEBUG_BRANCH_SELECTION_MOVE_INDEX'
                    )
                    if selection_override_mode_text not in (None, ''):
                        selection_override_mode = selection_override_mode_text.strip().lower()
                        if selection_override_move_index_text not in (None, ''):
                            try:
                                if (
                                    plan_request_index is None
                                    or int(selection_override_move_index_text) != int(plan_request_index)
                                ):
                                    selection_override_mode = None
                            except ValueError:
                                self.get_logger().warn(
                                    'Invalid CUROBO_DEBUG_BRANCH_SELECTION_MOVE_INDEX='
                                    f'{selection_override_move_index_text}, expected integer move index.'
                                )
                                selection_override_mode = None
                    if selection_override_mode is None and effective_selection_mode not in (None, ''):
                        selection_override_mode = str(effective_selection_mode).strip().lower()
                    # [caohy] Task 22：第一版 branch consistency（分支一致性）修法不改 seed 主框架，
                    # 只改”可行 IK 解里选哪一支”这一步，在最近解基础上叠加局部趋势连续性和大跳门控。
                    # [caohy] Task 40：新增 goal_joint_anchor + waypoint_t 参数，
                    # 让选支评分能渐进引导候选向目标构型过渡。
                    selection = self._select_branch_consistent_seed_solution(
                        prev_solution=prev_solution,
                        feasible_solutions=feasible_solutions,
                        prev_step_delta=prev_step_delta,
                        recent_step_l2_history=recent_step_l2_history,
                        forced_candidate_index=forced_branch_candidate_index,
                        selection_override_mode=selection_override_mode,
                        candidate_pose_goal_summaries=candidate_pose_goal_summaries,
                        goal_joint_anchor=goal_joint_anchor,
                        waypoint_t=t,
                    )
                    sol = selection['solution']
                    source = f'ik_solution_best_of_{int(feasible_solutions.shape[0])}'
                    step_delta = selection['step_delta']
                    step_jump_l2 = selection['step_jump_l2']
                    step_max_abs = selection['step_max_abs']
                    if step_jump_l2 > max_step_jump_l2:
                        max_step_jump_l2 = step_jump_l2
                        max_step_jump_index = i
                        max_step_jump_source = source
                        max_step_jump_delta = [round(float(v), 6) for v in step_delta.detach().cpu().tolist()]
                    joint_trajectory.append(sol)
                    seed_waypoint_debug.append({
                        'index': i,
                        't': round(float(t), 6),
                        'source': source,
                        'ik_success': True,
                        'fallback_used': False,
                        'step_jump_l2': round(step_jump_l2, 6),
                        'step_max_abs': round(step_max_abs, 6),
                        'step_delta': [round(float(v), 6) for v in step_delta.detach().cpu().tolist()],
                        'waypoint_goal_position': [round(float(v), 6) for v in pos_i.reshape(-1).detach().cpu().tolist()],
                        'waypoint_goal_quaternion': [round(float(v), 6) for v in quat_i],
                        'waypoint_goal_alignment_deviation_deg': round(
                            float(constraint_utils.compute_alignment_deviation_from_quaternion(quat_i)), 6
                        ),
                        'expected_twist_deg': round(float(twist_i_deg), 6),
                        'branch_guard_l2': round(float(selection['jump_guard_l2']), 6),
                        'branch_guard_applied': bool(selection['guard_applied']),
                        'branch_guard_kept_count': int(selection['guard_kept_count']),
                        'branch_selection_score': round(float(selection['selection_score']), 6),
                        'branch_trend_cost': round(float(selection['trend_cost']), 6),
                        'branch_direction_penalty': round(float(selection['direction_penalty']), 6),
                        # [caohy] Task 22：当大跳门控失效时，同时记下“最近解优先”与“当前 score 选中解”的差异，
                        # 用来确认退化究竟来自没有阈值内候选，还是来自阈值外回退时的评分策略本身。
                        'branch_nearest_step_jump_l2': round(float(selection['nearest_step_jump_l2']), 6),
                        'branch_nearest_step_max_abs': round(float(selection['nearest_step_max_abs']), 6),
                        'branch_nearest_index': int(selection['nearest_index']),
                        'branch_nearest_joint_limit_summary': selection['nearest_joint_limit_summary'],
                        'branch_nearest_pose_goal_summary': selection['nearest_pose_goal_summary'],
                        'branch_score_best_index': int(selection['score_best_index']),
                        'branch_score_best_joint_limit_summary': selection['score_best_joint_limit_summary'],
                        'branch_score_best_pose_goal_summary': selection['score_best_pose_goal_summary'],
                        'branch_guard_fallback_mode': selection['guard_fallback_mode'],
                        'branch_selected_joint_limit_summary': selection['selected_joint_limit_summary'],
                        'branch_selected_pose_goal_summary': selection['selected_pose_goal_summary'],
                        'branch_in_limit_candidate_count': int(selection['in_limit_candidate_count']),
                        'branch_in_limit_best_index': (
                            int(selection['in_limit_best_index'])
                            if selection['in_limit_best_index'] is not None else None
                        ),
                        'branch_in_limit_best_score': (
                            round(float(selection['in_limit_best_score']), 6)
                            if selection['in_limit_best_score'] is not None else None
                        ),
                        'branch_in_limit_best_joint_limit_summary': (
                            selection['in_limit_best_joint_limit_summary']
                        ),
                        'branch_in_limit_best_pose_goal_summary': (
                            selection['in_limit_best_pose_goal_summary']
                        ),
                        'branch_limit_violation_penalty_weight': round(
                            float(selection['limit_violation_penalty_weight']), 6
                        ),
                        'branch_penalized_best_index': int(selection['penalized_best_index']),
                        'branch_penalized_best_score': round(
                            float(selection['penalized_best_score']), 6
                        ),
                        'branch_penalized_best_joint_limit_summary': (
                            selection['penalized_best_joint_limit_summary']
                        ),
                        'branch_penalized_best_pose_goal_summary': (
                            selection['penalized_best_pose_goal_summary']
                        ),
                        'branch_guarded_in_limit_candidate_count': int(
                            selection['guarded_in_limit_candidate_count']
                        ),
                        'branch_guarded_in_limit_best_index': (
                            int(selection['guarded_in_limit_best_index'])
                            if selection['guarded_in_limit_best_index'] is not None else None
                        ),
                        'branch_guarded_in_limit_best_score': (
                            round(float(selection['guarded_in_limit_best_score']), 6)
                            if selection['guarded_in_limit_best_score'] is not None else None
                        ),
                        'branch_guarded_in_limit_best_joint_limit_summary': (
                            selection['guarded_in_limit_best_joint_limit_summary']
                        ),
                        'branch_guarded_in_limit_best_pose_goal_summary': (
                            selection['guarded_in_limit_best_pose_goal_summary']
                        ),
                        # [caohy] Task 22 阶段1：继续下钻“同一起点下 IK 可行集为什么还会漂”，
                        # 这里把最终选中的候选索引、关节值和整组候选指纹直接回写到诊断里，
                        # 用来区分“同一批解顺序变化”与“可行解集合本身变化”。
                        'selected_ik_index': int(selection['selected_index']),
                        'selected_score': round(float(selection['selection_score']), 6),
                        'selected_jump_from_prev_l2': round(float(selection['step_jump_l2']), 6),
                        'forced_candidate_index': (
                            int(selection['forced_candidate_index'])
                            if selection['forced_candidate_index'] is not None else None
                        ),
                        'forced_override_applied': bool(selection['forced_override_applied']),
                        'branch_selection_mode_requested': selection['selection_mode_requested'],
                        'branch_selection_mode_applied': bool(selection['selection_mode_applied']),
                        'branch_selection_mode_effective': selection['selection_mode_effective'],
                        'branch_selection_mode_fallback_reason': (
                            selection['selection_mode_fallback_reason']
                        ),
                        'near_tie_override_applied': bool(selection['near_tie_override_applied']),
                        'near_tie_override_from': (
                            int(selection['near_tie_override_from'])
                            if selection['near_tie_override_from'] is not None else None
                        ),
                        'near_tie_override_to': (
                            int(selection['near_tie_override_to'])
                            if selection['near_tie_override_to'] is not None else None
                        ),
                        'near_tie_score_gap': (
                            round(float(selection['near_tie_score_gap']), 6)
                            if selection['near_tie_score_gap'] is not None else None
                        ),
                        'near_tie_direction_penalty_gap': (
                            round(float(selection['near_tie_direction_penalty_gap']), 6)
                            if selection['near_tie_direction_penalty_gap'] is not None else None
                        ),
                        'near_tie_jump_gap': (
                            round(float(selection['near_tie_jump_gap']), 6)
                            if selection['near_tie_jump_gap'] is not None else None
                        ),
                        'limit_aware_override_applied': bool(selection['limit_aware_override_applied']),
                        'limit_aware_override_from': (
                            int(selection['limit_aware_override_from'])
                            if selection['limit_aware_override_from'] is not None else None
                        ),
                        'limit_aware_override_to': (
                            int(selection['limit_aware_override_to'])
                            if selection['limit_aware_override_to'] is not None else None
                        ),
                        'selected_joint_position': [
                            round(float(v), 6) for v in selection['solution'].detach().cpu().tolist()
                        ],
                        'branch_candidate_debug_rows': selection['candidate_debug_rows'],
                        **({'debug_snapshot': debug_snapshot} if debug_snapshot is not None else {}),
                    })
                    prev_solution = sol.detach()
                    prev_step_delta = step_delta.detach().clone()
                    recent_step_l2_history.append(float(step_jump_l2))
                    if len(recent_step_l2_history) > 6:
                        recent_step_l2_history = recent_step_l2_history[-6:]
                else:
                    ik_fail_count += 1
                    fallback_used = True
                    source = 'ik_infeasible_fallback_prev'
                    step_delta = prev_solution - prev_solution
                    joint_trajectory.append(prev_solution.clone())
                    seed_waypoint_debug.append({
                        'index': i,
                        't': round(float(t), 6),
                        'source': source,
                        'ik_success': False,
                        'fallback_used': fallback_used,
                        'step_jump_l2': 0.0,
                        'step_max_abs': 0.0,
                        'step_delta': [0.0] * prev_solution.numel(),
                        **({'debug_snapshot': debug_snapshot} if debug_snapshot is not None else {}),
                    })
                    prev_step_delta = None
            except Exception as e:
                self.get_logger().warn(f'Seed IK {i}: {e}')
                ik_fail_count += 1
                fallback_used = True
                source = 'ik_exception_fallback_prev'
                joint_trajectory.append(prev_solution.clone())
                seed_waypoint_debug.append({
                    'index': i,
                    't': round(float(t), 6),
                    'source': source,
                    'ik_success': False,
                    'fallback_used': fallback_used,
                    'step_jump_l2': 0.0,
                    'step_max_abs': 0.0,
                    'step_delta': [0.0] * prev_solution.numel(),
                    'exception': str(e),
                    **({'debug_snapshot': debug_snapshot} if debug_snapshot is not None else {}),
                })
                prev_step_delta = None

        trajectory = torch.stack(joint_trajectory, dim=0)
        success = ik_fail_count < num_waypoints

        # [caohy] Task 40：种子跳变后处理平滑。
        # 如果检测到单步大跳（>1.5 rad），在跳变点前后做关节空间线性插值，
        # 将突变摊开到一个窗口（前后各 smooth_half_window 步）。
        # 中间 waypoint 的关节角不再精确满足 pose，但作为 trajopt 种子足够；
        # 优化器本身会修正回来。
        # 注意：保留 raw_trajectory 用于 alignment_seed 候选（每个 waypoint 是精确 IK 解，保持对齐），
        # smoothed trajectory 仅给 trajopt 用作初始种子。
        raw_trajectory = trajectory.clone()
        # [caohy] Task 40 分段优化：保存平滑前的原始跳变点，供 split trajopt 在构型边界处分段。
        raw_max_step_jump_l2 = float(max_step_jump_l2)
        raw_max_step_jump_index = int(max_step_jump_index) if max_step_jump_index is not None else None
        _SMOOTH_THRESHOLD = 1.5
        if max_step_jump_l2 > _SMOOTH_THRESHOLD and max_step_jump_index is not None:
            smooth_half_window = min(5, max(2, num_waypoints // 6))
            jump_idx = int(max_step_jump_index)
            smooth_start = max(0, jump_idx - smooth_half_window)
            smooth_end = min(num_waypoints - 1, jump_idx + smooth_half_window)
            if smooth_end > smooth_start:
                anchor_left = trajectory[smooth_start].clone()
                anchor_right = trajectory[smooth_end].clone()
                for s_idx in range(smooth_start + 1, smooth_end):
                    alpha = float(s_idx - smooth_start) / float(smooth_end - smooth_start)
                    trajectory[s_idx] = (1.0 - alpha) * anchor_left + alpha * anchor_right
                # 重新计算平滑后的 max_step_jump
                smoothed_max_l2 = 0.0
                smoothed_max_idx = None
                for s_idx in range(1, trajectory.shape[0]):
                    step_l2 = float(torch.linalg.norm(trajectory[s_idx] - trajectory[s_idx - 1]).item())
                    if step_l2 > smoothed_max_l2:
                        smoothed_max_l2 = step_l2
                        smoothed_max_idx = s_idx
                self.get_logger().info(
                    f'Seed jump smoothed: original_jump={max_step_jump_l2:.4f}@wp{jump_idx}, '
                    f'window=[{smooth_start},{smooth_end}], '
                    f'smoothed_max={smoothed_max_l2:.4f}@wp{smoothed_max_idx}'
                )
                max_step_jump_l2 = smoothed_max_l2
                max_step_jump_index = smoothed_max_idx

        self.get_logger().info(
            f'Seed generated: waypoints={num_waypoints}, ik_fail={ik_fail_count}, '
            f'start_twist={start_twist_deg:.1f} deg, success={success}, '
            f'max_step_jump_l2={max_step_jump_l2:.6f}, '
            f'max_step_jump_index={max_step_jump_index}'
        )

        return {
            'trajectory': trajectory,
            'raw_trajectory': raw_trajectory,
            'raw_max_step_jump_l2': round(raw_max_step_jump_l2, 6),
            'raw_max_step_jump_index': raw_max_step_jump_index,
            'success': bool(success),
            'ik_fail_count': ik_fail_count,
            'num_waypoints': num_waypoints,
            'plan_request_index': int(plan_request_index) if plan_request_index is not None else None,
            'seed_family_name': effective_seed_family_name,
            'seed_family_config': effective_seed_family_config,
            'twist_schedule_mode': effective_twist_schedule_mode,
            'goal_anchor_rank': int(effective_goal_anchor_rank),
            'goal_anchor_rank_used': goal_joint_anchor_rank_used,
            'goal_anchor_joint': (
                [round(float(v), 6) for v in goal_joint_anchor.detach().cpu().tolist()]
                if goal_joint_anchor is not None else None
            ),
            'selection_mode': effective_selection_mode,
            'start_twist_deg': round(float(start_twist_deg), 6),
            'goal_twist_deg': round(float(goal_twist_deg), 6),
            'seed_input_start_joint': [round(float(v), 6) for v in list(start_joint)],
            'seed_first_joint': [round(float(v), 6) for v in trajectory[0].detach().cpu().tolist()],
            'seed_last_joint': [round(float(v), 6) for v in trajectory[-1].detach().cpu().tolist()],
            'seed_last_joint_limit_summary': self._summarize_joint_limit_violation(trajectory[-1]),
            'seed_trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(trajectory),
            'max_step_jump_l2': round(float(max_step_jump_l2), 6),
            'max_step_jump_index': int(max_step_jump_index) if max_step_jump_index is not None else None,
            'max_step_jump_source': max_step_jump_source,
            'max_step_jump_delta': max_step_jump_delta,
            'seed_alignment_discrete_summary': self._summarize_single_trajectory_alignment(
                trajectory,
                alignment_tolerance_deg=alignment_tolerance_deg,
                samples_per_segment=None,
                seed_waypoint_debug=seed_waypoint_debug,
            ),
            'seed_alignment_interpolated_summary': self._summarize_single_trajectory_alignment(
                trajectory,
                alignment_tolerance_deg=alignment_tolerance_deg,
                samples_per_segment=4,
            ),
            'waypoint_debug': seed_waypoint_debug,
        }

    @staticmethod
    def _wrap_to_nearest_angle(reference: float, target: float) -> float:
        return reference + math.atan2(
            math.sin(target - reference),
            math.cos(target - reference),
        )

    def _wrap_seed_solution_to_prev(self, prev_solution: torch.Tensor, solution: torch.Tensor) -> torch.Tensor:
        wrapped = solution.detach().clone()
        continuous_joint_indices = [0, 5]
        # [caohy] Task 31 Step 5：限位感知包角。
        # 原始逻辑只做最近角对齐，不检查结果是否超出 URDF 位置限位；
        # 当上一帧 joint_1 接近 +2pi 时，最近角对齐会把 raw 解推到 +2pi 以上。
        # 这里在包角后增加一步：如果 wrapped 值超出 [-limit, +limit]，
        # 就加减 2pi 拉回限位内（因为 2pi 周期内总有一个等价角在限位内）。
        # [caohy] Task 40：修复限位拉回大跳。
        # 当限位拉回产生的 delta 比 raw 值（未包角）的 delta 更大时，
        # 说明拉回反而制造了跳变；此时改用 raw 值（如果 raw 值本身在限位内）。
        _TWO_PI = 2.0 * math.pi
        _position_limit = 6.2832  # 与 URDF joint_1/joint_6 限位一致
        for joint_index in continuous_joint_indices:
            if joint_index < 0 or joint_index >= wrapped.shape[0]:
                continue
            prev_val = float(prev_solution[joint_index].item())
            raw_val = float(wrapped[joint_index].item())
            nearest_val = self._wrap_to_nearest_angle(prev_val, raw_val)
            if -_position_limit <= nearest_val <= _position_limit:
                wrapped[joint_index] = nearest_val
            else:
                # 限位拉回
                if nearest_val > _position_limit:
                    clamped_val = nearest_val - _TWO_PI
                else:
                    clamped_val = nearest_val + _TWO_PI
                # 比较拉回后的 delta 和 raw 值的 delta，取更小的那个
                delta_clamped = abs(clamped_val - prev_val)
                delta_raw = abs(raw_val - prev_val)
                if -_position_limit <= raw_val <= _position_limit and delta_raw < delta_clamped:
                    wrapped[joint_index] = raw_val
                else:
                    wrapped[joint_index] = clamped_val
        return wrapped

    def _constraint_eval_kinematics_fn(self, positions):
        """[caohy] Task 29：统一约束评估用 FK，避免 seed / candidate 诊断各自复制一套。"""
        state = CuJointState.from_position(positions, joint_names=self._joint_names)
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        return SimpleNamespace(ee_quaternion=tool_pose.quaternion)

    @staticmethod
    def _densify_joint_trajectory_linear(trajectory: torch.Tensor, samples_per_segment: int) -> torch.Tensor:
        """[caohy] Task 29：按关节线性插值把离散轨迹加密，便于判断点间是否把对齐拉坏。"""
        if trajectory.ndim != 2 or trajectory.shape[0] < 2 or int(samples_per_segment) <= 1:
            return trajectory.detach().clone()

        dense_points = [trajectory[0].detach().clone()]
        samples_per_segment = int(samples_per_segment)
        for seg_idx in range(int(trajectory.shape[0]) - 1):
            start_pt = trajectory[seg_idx]
            end_pt = trajectory[seg_idx + 1]
            for sub_idx in range(1, samples_per_segment + 1):
                alpha = float(sub_idx) / float(samples_per_segment)
                dense_points.append(((1.0 - alpha) * start_pt + alpha * end_pt).detach().clone())
        return torch.stack(dense_points, dim=0)

    @staticmethod
    def _summarize_alignment_angle_series(
        angle_series: torch.Tensor,
        alignment_tolerance_deg: float,
        samples_per_segment: Optional[int] = None,
    ) -> dict:
        """[caohy] Task 29：把对齐角序列压成“从哪开始坏、最坏在哪”的摘要。"""
        angle_flat = angle_series.reshape(-1).detach().cpu()
        if angle_flat.numel() == 0:
            return {
                'sample_count': 0,
                'max_alignment_deviation_deg': 0.0,
                'mean_alignment_deviation_deg': 0.0,
                'first_exceed_sample_index': None,
                'worst_sample_index': None,
            }

        first_exceed_sample_index = None
        exceed_mask = angle_flat > float(alignment_tolerance_deg)
        exceed_indices = torch.nonzero(exceed_mask, as_tuple=False).view(-1)
        if int(exceed_indices.numel()) > 0:
            first_exceed_sample_index = int(exceed_indices[0].item())
        worst_sample_index = int(torch.argmax(angle_flat).item())

        payload = {
            'sample_count': int(angle_flat.numel()),
            'max_alignment_deviation_deg': round(float(angle_flat[worst_sample_index].item()), 6),
            'mean_alignment_deviation_deg': round(float(torch.mean(angle_flat).item()), 6),
            'first_exceed_sample_index': first_exceed_sample_index,
            'worst_sample_index': worst_sample_index,
            'worst_alignment_deviation_deg': round(float(angle_flat[worst_sample_index].item()), 6),
        }

        if samples_per_segment is None:
            payload['first_exceed_waypoint_index'] = first_exceed_sample_index
            payload['worst_waypoint_index'] = worst_sample_index
            return payload

        samples_per_segment = int(samples_per_segment)

        def sample_to_segment(sample_index: Optional[int]):
            if sample_index is None or sample_index <= 0:
                return None, None
            segment_index = int((sample_index - 1) // samples_per_segment)
            alpha = float(sample_index - segment_index * samples_per_segment) / float(samples_per_segment)
            return segment_index, round(alpha, 6)

        first_seg_idx, first_seg_alpha = sample_to_segment(first_exceed_sample_index)
        worst_seg_idx, worst_seg_alpha = sample_to_segment(worst_sample_index)
        payload.update({
            'samples_per_segment': samples_per_segment,
            'first_exceed_segment_index': first_seg_idx,
            'first_exceed_segment_alpha': first_seg_alpha,
            'worst_segment_index': worst_seg_idx,
            'worst_segment_alpha': worst_seg_alpha,
        })
        return payload

    def _summarize_single_trajectory_alignment(
        self,
        trajectory: torch.Tensor,
        alignment_tolerance_deg: float,
        samples_per_segment: Optional[int] = None,
        seed_waypoint_debug: Optional[list] = None,
    ) -> dict:
        """[caohy] Task 29：给单条 seed 轨迹补“离散点 vs 点间插值”的对齐摘要。"""
        eval_traj = trajectory.detach().clone()
        if samples_per_segment is not None:
            eval_traj = self._densify_joint_trajectory_linear(eval_traj, samples_per_segment)

        positions = eval_traj.unsqueeze(0).to(device='cuda:0', dtype=torch.float32)
        y_tool = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:0')
        z_neg = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda:0')
        level_eval = constraint_utils.evaluate_axis_alignment_batched(
            positions,
            self._constraint_eval_kinematics_fn,
            alignment_tolerance_deg,
            y_tool,
            z_neg,
        )
        angle_series = level_eval['alignment_angle_map'][0]
        summary = self._summarize_alignment_angle_series(
            angle_series,
            alignment_tolerance_deg=alignment_tolerance_deg,
            samples_per_segment=samples_per_segment,
        )
        if seed_waypoint_debug is not None and samples_per_segment is None:
            angle_list = angle_series.detach().cpu().tolist()
            for idx, item in enumerate(seed_waypoint_debug):
                if idx >= len(angle_list):
                    break
                item['alignment_deviation_deg'] = round(float(angle_list[idx]), 6)
        return summary

    def _build_alignment_profile_debug(
        self,
        trajectory: torch.Tensor,
        alignment_tolerance_deg: float,
    ) -> dict:
        """[caohy] Task 35：生成单条轨迹逐点对齐剖面，隔离 seed / prepared / trajopt 哪一步破坏中间约束。"""
        if trajectory is None:
            return {'present': False}

        traj = trajectory.detach().clone()
        while traj.ndim > 2:
            if traj.shape[0] == 1:
                traj = traj.squeeze(0)
            else:
                traj = traj.reshape(-1, traj.shape[-1])
        if traj.ndim == 1:
            traj = traj.unsqueeze(0)

        positions = traj.to(device='cuda:0', dtype=torch.float32).unsqueeze(0)
        y_tool = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:0')
        z_neg = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda:0')
        level_eval = constraint_utils.evaluate_axis_alignment_batched(
            positions,
            self._constraint_eval_kinematics_fn,
            alignment_tolerance_deg,
            y_tool,
            z_neg,
        )
        angle_series = level_eval['alignment_angle_map'][0].detach().cpu()
        traj_cpu = traj.detach().cpu()
        exceed_indices = [
            int(i)
            for i, value in enumerate(angle_series.tolist())
            if float(value) > float(alignment_tolerance_deg)
        ]
        max_index = int(torch.argmax(angle_series).item()) if int(angle_series.numel()) > 0 else None
        if max_index is None:
            return {'present': True, 'sample_count': 0}
        window_start = max(0, max_index - 2)
        window_end = min(int(angle_series.shape[0]), max_index + 3)
        return {
            'present': True,
            'sample_count': int(angle_series.numel()),
            'max_alignment_deviation': round(float(angle_series[max_index].item()), 4),
            'max_alignment_point_index': max_index,
            'first_exceed_point_index': exceed_indices[0] if exceed_indices else None,
            'last_exceed_point_index': exceed_indices[-1] if exceed_indices else None,
            'exceed_count': len(exceed_indices),
            'alignment_profile_deg': [
                round(float(value), 4) for value in angle_series.tolist()
            ],
            'joint_position_at_max': [
                round(float(value), 6) for value in traj_cpu[max_index].tolist()
            ],
            'alignment_window': [
                {
                    'point_index': int(idx),
                    'alignment_deviation_deg': round(float(angle_series[idx].item()), 4),
                    'joint_position': [
                        round(float(value), 6) for value in traj_cpu[idx].tolist()
                    ],
                }
                for idx in range(window_start, window_end)
            ],
        }

    def _summarize_pose_against_goal(
        self,
        joint_positions: torch.Tensor,
        goal_position: torch.Tensor,
        goal_quaternion: list[float],
        expected_twist_deg: Optional[float] = None,
    ) -> list[dict]:
        """[caohy] Task 29：对单 waypoint 候选补 FK 实际位姿与当前目标位姿的直接误差。"""
        if joint_positions.ndim != 2:
            raise ValueError(f'joint_positions must be [N, DOF], got shape={list(joint_positions.shape)}')

        joint_positions = joint_positions.detach().clone().to(device='cuda:0', dtype=torch.float32)
        state = CuJointState.from_position(joint_positions, joint_names=self._joint_names)
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        fk_position = tool_pose.position.reshape(-1, 3)
        fk_quaternion = tool_pose.quaternion.reshape(-1, 4)

        goal_position = goal_position.detach().clone().reshape(1, 3).to(device='cuda:0', dtype=torch.float32)
        goal_quaternion_tensor = torch.tensor(
            [constraint_utils.normalize_quaternion(list(goal_quaternion))],
            device='cuda:0',
            dtype=torch.float32,
        ).expand(fk_quaternion.shape[0], -1)
        goal_quaternion_conj = goal_quaternion_tensor.clone()
        goal_quaternion_conj[:, 1:] *= -1.0
        q_rel = constraint_utils.quaternion_multiply_batched(goal_quaternion_conj, fk_quaternion)
        rel_sign = torch.where(q_rel[:, :1] < 0.0, -1.0, 1.0)
        q_rel = q_rel * rel_sign

        position_error = torch.linalg.norm(fk_position - goal_position, dim=-1)
        orientation_error_deg = torch.rad2deg(
            2.0 * torch.acos(torch.clamp(q_rel[:, 0], min=-1.0, max=1.0))
        )

        rotation_matrix = constraint_utils.quaternion_to_rotation_matrix_batched(fk_quaternion)
        y_tool = torch.tensor([0.0, 1.0, 0.0], device='cuda:0', dtype=torch.float32)
        z_neg = torch.tensor([0.0, 0.0, -1.0], device='cuda:0', dtype=torch.float32)
        axis_world = torch.matmul(rotation_matrix, y_tool.view(1, 3, 1)).squeeze(-1)
        alignment_dot = torch.sum(axis_world * z_neg.view(1, 3), dim=-1)
        alignment_deviation_deg = torch.rad2deg(
            torch.acos(torch.clamp(alignment_dot, min=-1.0, max=1.0))
        )

        twist_q = torch.stack([
            q_rel[:, 0],
            torch.zeros_like(q_rel[:, 0]),
            q_rel[:, 2],
            torch.zeros_like(q_rel[:, 0]),
        ], dim=-1)
        twist_q = twist_q / torch.linalg.norm(twist_q, dim=-1, keepdim=True).clamp_min(1e-12)
        twist_relative_goal_deg = torch.rad2deg(2.0 * torch.atan2(twist_q[:, 2], twist_q[:, 0]))
        twist_relative_goal_deg = constraint_utils.wrap_angle_deg_tensor(twist_relative_goal_deg)

        summaries = []
        for idx in range(int(joint_positions.shape[0])):
            summary = {
                'fk_position': [round(float(v), 6) for v in fk_position[idx].detach().cpu().tolist()],
                'fk_quaternion': [round(float(v), 6) for v in fk_quaternion[idx].detach().cpu().tolist()],
                'position_error_m': round(float(position_error[idx].item()), 6),
                'orientation_error_deg': round(float(orientation_error_deg[idx].item()), 6),
                'alignment_deviation_deg': round(float(alignment_deviation_deg[idx].item()), 6),
                'twist_relative_goal_deg': round(float(twist_relative_goal_deg[idx].item()), 6),
            }
            if expected_twist_deg is not None:
                twist_error_deg = constraint_utils.wrap_angle_deg(
                    float(twist_relative_goal_deg[idx].item()) - float(expected_twist_deg)
                )
                summary['expected_twist_deg'] = round(float(expected_twist_deg), 6)
                summary['twist_error_deg'] = round(float(twist_error_deg), 6)
            summaries.append(summary)
        return summaries

    def _get_selector_goal_gate_tolerances(self) -> tuple[float, float]:
        """[caohy] Task 15 bugfix：解析 selector 终点误差硬门槛。"""
        position_tolerance_m = 0.01
        orientation_tolerance_deg = 3.0
        position_text = os.environ.get(
            'CUROBO_LEVEL_SELECT_GOAL_POSITION_TOLERANCE_M', ''
        ).strip()
        orientation_text = os.environ.get(
            'CUROBO_LEVEL_SELECT_GOAL_ORIENTATION_TOLERANCE_DEG', ''
        ).strip()
        if position_text:
            try:
                position_tolerance_m = float(position_text)
            except ValueError:
                self.get_logger().warn(
                    'Invalid CUROBO_LEVEL_SELECT_GOAL_POSITION_TOLERANCE_M='
                    f'{position_text}, fallback to 0.01'
                )
        if orientation_text:
            try:
                orientation_tolerance_deg = float(orientation_text)
            except ValueError:
                self.get_logger().warn(
                    'Invalid CUROBO_LEVEL_SELECT_GOAL_ORIENTATION_TOLERANCE_DEG='
                    f'{orientation_text}, fallback to 3.0'
                )
        return float(position_tolerance_m), float(orientation_tolerance_deg)

    def _summarize_candidate_terminal_goal_pose_batch(
        self,
        positions_batch: torch.Tensor,
        target_pose,
    ) -> list[dict]:
        """[caohy] Task 15 bugfix：为所有候选统一计算终点 FK 误差摘要。"""
        if positions_batch.ndim != 3:
            raise ValueError(
                f'positions_batch must be [B, T, DOF], got shape={list(positions_batch.shape)}'
            )
        terminal_positions = positions_batch[:, -1, :]
        return self._summarize_pose_against_goal(
            terminal_positions,
            torch.tensor(target_pose[:3], device='cuda:0', dtype=torch.float32),
            target_pose[3:7],
        )

    def _build_selector_goal_gate_summary(
        self,
        candidate_goal_summaries: list[dict],
        position_tolerance_m: float,
        orientation_tolerance_deg: float,
    ) -> dict:
        """[caohy] Task 15 bugfix：按终点位姿误差给候选做硬门槛判定。"""
        candidate_position_error_m = []
        candidate_orientation_error_deg = []
        candidate_goal_pose_valid = []
        for summary in candidate_goal_summaries:
            position_error_m = None if not isinstance(summary, dict) else summary.get('position_error_m')
            orientation_error_deg = None if not isinstance(summary, dict) else summary.get('orientation_error_deg')
            candidate_position_error_m.append(
                None if position_error_m is None else round(float(position_error_m), 6)
            )
            candidate_orientation_error_deg.append(
                None if orientation_error_deg is None else round(float(orientation_error_deg), 6)
            )
            candidate_goal_pose_valid.append(bool(
                position_error_m is not None
                and orientation_error_deg is not None
                and float(position_error_m) <= float(position_tolerance_m)
                and float(orientation_error_deg) <= float(orientation_tolerance_deg)
            ))
        return {
            'goal_position_tolerance_m': round(float(position_tolerance_m), 6),
            'goal_orientation_tolerance_deg': round(float(orientation_tolerance_deg), 6),
            'candidate_position_error_m': candidate_position_error_m,
            'candidate_orientation_error_deg': candidate_orientation_error_deg,
            'candidate_goal_pose_valid': candidate_goal_pose_valid,
            'goal_pose_valid_count': int(sum(1 for item in candidate_goal_pose_valid if item)),
        }

    def _select_branch_consistent_seed_solution(
        self,
        prev_solution: torch.Tensor,
        feasible_solutions: torch.Tensor,
        prev_step_delta: Optional[torch.Tensor] = None,
        recent_step_l2_history: Optional[list[float]] = None,
        forced_candidate_index: Optional[int] = None,
        selection_override_mode: Optional[str] = None,
        candidate_pose_goal_summaries: Optional[list[dict]] = None,
        goal_joint_anchor: Optional[torch.Tensor] = None,
        waypoint_t: float = 0.0,
    ) -> dict:
        """在多个可行 IK 解中优先选局部趋势更连续的那一支。"""
        wrapped_solutions = torch.stack(
            [self._wrap_seed_solution_to_prev(prev_solution, sol) for sol in feasible_solutions],
            dim=0,
        )
        solution_delta = wrapped_solutions - prev_solution.unsqueeze(0)
        solution_delta_l2 = torch.linalg.norm(solution_delta, dim=-1)

        trend_cost = torch.zeros_like(solution_delta_l2)
        direction_penalty = torch.zeros_like(solution_delta_l2)
        prev_step_norm = 0.0
        if prev_step_delta is not None:
            prev_step_norm = float(torch.linalg.norm(prev_step_delta).item())
            trend_cost = torch.linalg.norm(solution_delta - prev_step_delta.unsqueeze(0), dim=-1)
            if prev_step_norm > 1e-6:
                prev_dir = prev_step_delta.unsqueeze(0) / prev_step_norm
                current_norm = torch.clamp(solution_delta_l2, min=1e-6).unsqueeze(-1)
                current_dir = solution_delta / current_norm
                cosine = torch.sum(current_dir * prev_dir, dim=-1)
                direction_penalty = torch.clamp(-cosine, min=0.0)

        # [caohy] Task 40：goal-proximity cost——按参数 t 渐增引导候选向目标构型过渡。
        # 当 goal_joint_anchor 可用时，计算每个候选与 expected_joint(t) 的距离作为额外代价；
        # expected_joint = lerp(start_of_seed, goal_joint_anchor, t)，
        # 但这里用 prev_solution 近似当前位置，直接算候选与 goal 的距离更稳定。
        goal_proximity_cost = torch.zeros_like(solution_delta_l2)
        if goal_joint_anchor is not None and waypoint_t > 0.0:
            goal_delta = wrapped_solutions - goal_joint_anchor.unsqueeze(0)
            goal_proximity_cost = torch.linalg.norm(goal_delta, dim=-1)

        history_baseline = 0.0
        if recent_step_l2_history:
            sorted_history = sorted(float(v) for v in recent_step_l2_history if v is not None)
            if sorted_history:
                history_baseline = float(sorted_history[len(sorted_history) // 2])

        jump_guard_l2 = max(
            0.35,
            prev_step_norm * 2.5 if prev_step_norm > 1e-6 else 0.0,
            history_baseline * 3.5 if history_baseline > 1e-6 else 0.0,
        )
        guarded_mask = solution_delta_l2 <= jump_guard_l2
        if bool(torch.any(guarded_mask).item()):
            candidate_indices = torch.nonzero(guarded_mask, as_tuple=False).view(-1)
            guard_applied = True
        else:
            candidate_indices = torch.arange(
                wrapped_solutions.shape[0], device=wrapped_solutions.device, dtype=torch.long,
            )
            guard_applied = False

        # [caohy] Task 40：评分公式加入 goal_proximity_cost，权重随 t 线性增长（最大 0.5）。
        # 实验验证发现：同一 waypoint 的所有可行 IK 候选通常在同一构型里，
        # goal_proximity 无法改变相对排名，反而会改变之前正常 move 的选支路径，
        # 导致后续 move 起点级联变化。因此当前只保留诊断记录，不参与评分。
        goal_weight = 0.0
        score = solution_delta_l2 + 0.35 * trend_cost + 0.25 * direction_penalty + goal_weight * goal_proximity_cost
        nearest_seed = int(torch.argmin(solution_delta_l2).item())
        candidate_scores = score.index_select(0, candidate_indices)
        best_local = int(torch.argmin(candidate_scores).item())
        score_best_seed = int(candidate_indices[best_local].item())
        best_seed = score_best_seed
        forced_override_applied = False
        near_tie_override_applied = False
        near_tie_override_from = None
        near_tie_override_to = None
        near_tie_score_gap = None
        near_tie_direction_penalty_gap = None
        near_tie_jump_gap = None
        if forced_candidate_index is not None and 0 <= int(forced_candidate_index) < int(wrapped_solutions.shape[0]):
            # [caohy] Task 22 阶段1：这里只用于实验分支滚动效果，不改变默认评分规则；
            # 当指定候选存在时，直接覆盖当前 waypoint 的选支，方便后续比较整段 seed 质量。
            best_seed = int(forced_candidate_index)
            forced_override_applied = True
        elif int(candidate_indices.numel()) >= 2:
            sorted_local = torch.argsort(candidate_scores)
            top1_seed = int(candidate_indices[int(sorted_local[0].item())].item())
            top2_seed = int(candidate_indices[int(sorted_local[1].item())].item())
            near_tie_score_gap = float(score[top2_seed].item() - score[top1_seed].item())
            near_tie_direction_penalty_gap = float(
                direction_penalty[top1_seed].item() - direction_penalty[top2_seed].item()
            )
            near_tie_jump_gap = float(solution_delta_l2[top2_seed].item() - solution_delta_l2[top1_seed].item())
            # [caohy] Task 22 阶段1：move09（第九条样本）的真实 A/B 已经证明，
            # guard 失效时可能出现“top1 只赢一点局部 jump，但严格对齐明显更差”的近分误选；
            # 因此这里把 near-tie override（近平分支覆盖）收成默认窄规则，只在 top2 极接近时才允许次优候选翻盘。
            if (
                not guard_applied
                and near_tie_score_gap is not None and near_tie_score_gap <= 0.03
                and near_tie_jump_gap is not None and near_tie_jump_gap <= 0.05
                and near_tie_direction_penalty_gap is not None and near_tie_direction_penalty_gap >= 0.01
            ):
                best_seed = top2_seed
                near_tie_override_applied = True
                near_tie_override_from = top1_seed
                near_tie_override_to = top2_seed
        # [caohy] Task 31 Step 5：限位内候选优先选支。
        # 在默认评分 + near-tie override 完成后，额外检查当前 best_seed 是否超限；
        # 如果超限且存在限位内候选，则切换到限位内最优候选。
        # 这是 wrapping 层修复后的第二道防线：即使包角修好了大部分问题，
        # 仍有部分 waypoint 可能因 raw IK 分布或 guard 门控而选中超限候选。
        limit_aware_override_applied = False
        limit_aware_override_from = None
        limit_aware_override_to = None
        # 先计算候选限位摘要（后面诊断代码也会用到）
        candidate_limit_summaries = [
            self._summarize_joint_limit_violation(wrapped_solutions[idx])
            for idx in range(int(wrapped_solutions.shape[0]))
        ]
        in_limit_candidate_indices = [
            idx for idx, summary in enumerate(candidate_limit_summaries)
            if not bool(summary.get('has_violation'))
        ]
        best_seed_has_violation = bool(
            candidate_limit_summaries[best_seed].get('has_violation')
        ) if best_seed < len(candidate_limit_summaries) else False
        if best_seed_has_violation and in_limit_candidate_indices:
            # 当前选中超限，但有限位内候选：切换到限位内最优
            in_limit_scores_for_override = [
                (idx, float(score[idx].item()))
                for idx in in_limit_candidate_indices
            ]
            in_limit_override_best, _ = min(
                in_limit_scores_for_override, key=lambda item: item[1],
            )
            limit_aware_override_from = best_seed
            limit_aware_override_to = int(in_limit_override_best)
            best_seed = int(in_limit_override_best)
            limit_aware_override_applied = True
        # [caohy] Task 29：继续下钻”最近解更平滑但包角后已超限”的可疑现象，
        # 这里额外记录每个候选在 wrapped（包角）后是否越过当前 CuRobo 实际生效的关节限位，
        # 同时给出”若只在不超限候选里按原评分挑，理论上会选谁”的对照信息。
        # [caohy] Task 31 Step 5：candidate_limit_summaries 和 in_limit_candidate_indices
        # 已在上方 limit-aware override 逻辑中计算，此处不再重复计算。
        in_limit_best_index = None
        in_limit_best_score = None
        if in_limit_candidate_indices:
            in_limit_scores = [
                (idx, float(score[idx].item()))
                for idx in in_limit_candidate_indices
            ]
            in_limit_best_index, in_limit_best_score = min(in_limit_scores, key=lambda item: item[1])
        # [caohy] Task 29：继续做“只读 penalty / gate（惩罚 / 门控）”诊断，
        # 不改默认选支，只额外回答“如果对包角后超限候选降分或禁用，会理论上选谁”。
        limit_violation_penalty_weight = 25.0
        limit_violation_penalty_weight_text = os.environ.get('CUROBO_DEBUG_LIMIT_VIOLATION_WEIGHT')
        if limit_violation_penalty_weight_text not in (None, ''):
            try:
                limit_violation_penalty_weight = float(limit_violation_penalty_weight_text)
            except ValueError:
                self.get_logger().warn(
                    'Invalid CUROBO_DEBUG_LIMIT_VIOLATION_WEIGHT='
                    f'{limit_violation_penalty_weight_text}, expected float.'
                )
        candidate_violation_values = torch.tensor(
            [
                float(summary.get('max_violation') or 0.0)
                for summary in candidate_limit_summaries
            ],
            device=score.device,
            dtype=score.dtype,
        )
        limit_penalty = candidate_violation_values * float(limit_violation_penalty_weight)
        penalized_score = score + limit_penalty
        penalized_candidate_scores = penalized_score.index_select(0, candidate_indices)
        penalized_best_local = int(torch.argmin(penalized_candidate_scores).item())
        penalized_best_index = int(candidate_indices[penalized_best_local].item())
        penalized_best_score = float(penalized_score[penalized_best_index].item())
        pose_goal_position_tolerance_m = 0.01
        pose_goal_orientation_tolerance_deg = 3.0
        pose_goal_position_tolerance_text = os.environ.get(
            'CUROBO_DEBUG_POSE_GOAL_POSITION_TOLERANCE_M'
        )
        pose_goal_orientation_tolerance_text = os.environ.get(
            'CUROBO_DEBUG_POSE_GOAL_ORIENTATION_TOLERANCE_DEG'
        )
        if pose_goal_position_tolerance_text not in (None, ''):
            try:
                pose_goal_position_tolerance_m = float(pose_goal_position_tolerance_text)
            except ValueError:
                self.get_logger().warn(
                    'Invalid CUROBO_DEBUG_POSE_GOAL_POSITION_TOLERANCE_M='
                    f'{pose_goal_position_tolerance_text}, expected float.'
                )
        if pose_goal_orientation_tolerance_text not in (None, ''):
            try:
                pose_goal_orientation_tolerance_deg = float(pose_goal_orientation_tolerance_text)
            except ValueError:
                self.get_logger().warn(
                    'Invalid CUROBO_DEBUG_POSE_GOAL_ORIENTATION_TOLERANCE_DEG='
                    f'{pose_goal_orientation_tolerance_text}, expected float.'
                )
        pose_goal_accurate_candidate_indices = []
        if candidate_pose_goal_summaries is not None:
            for idx, summary in enumerate(candidate_pose_goal_summaries):
                if summary is None:
                    continue
                position_error_m = summary.get('position_error_m')
                orientation_error_deg = summary.get('orientation_error_deg')
                if (
                    position_error_m is not None
                    and orientation_error_deg is not None
                    and float(position_error_m) <= float(pose_goal_position_tolerance_m)
                    and float(orientation_error_deg) <= float(pose_goal_orientation_tolerance_deg)
                ):
                    pose_goal_accurate_candidate_indices.append(idx)
        accurate_best_index = None
        accurate_best_score = None
        if pose_goal_accurate_candidate_indices:
            accurate_scores = [
                (idx, float(score[idx].item()))
                for idx in pose_goal_accurate_candidate_indices
            ]
            accurate_best_index, accurate_best_score = min(accurate_scores, key=lambda item: item[1])
        candidate_index_set = {int(idx.item()) for idx in candidate_indices}
        guarded_in_limit_candidate_indices = [
            idx for idx in in_limit_candidate_indices if idx in candidate_index_set
        ]
        guarded_pose_goal_accurate_candidate_indices = [
            idx for idx in pose_goal_accurate_candidate_indices if idx in candidate_index_set
        ]
        accurate_in_limit_candidate_indices = [
            idx for idx in pose_goal_accurate_candidate_indices if idx in in_limit_candidate_indices
        ]
        guarded_in_limit_best_index = None
        guarded_in_limit_best_score = None
        if guarded_in_limit_candidate_indices:
            guarded_in_limit_scores = [
                (idx, float(score[idx].item()))
                for idx in guarded_in_limit_candidate_indices
            ]
            guarded_in_limit_best_index, guarded_in_limit_best_score = min(
                guarded_in_limit_scores, key=lambda item: item[1]
            )
        guarded_pose_goal_accurate_best_index = None
        guarded_pose_goal_accurate_best_score = None
        if guarded_pose_goal_accurate_candidate_indices:
            guarded_pose_goal_accurate_scores = [
                (idx, float(score[idx].item()))
                for idx in guarded_pose_goal_accurate_candidate_indices
            ]
            guarded_pose_goal_accurate_best_index, guarded_pose_goal_accurate_best_score = min(
                guarded_pose_goal_accurate_scores, key=lambda item: item[1]
            )
        accurate_in_limit_best_index = None
        accurate_in_limit_best_score = None
        if accurate_in_limit_candidate_indices:
            accurate_in_limit_scores = [
                (idx, float(score[idx].item()))
                for idx in accurate_in_limit_candidate_indices
            ]
            accurate_in_limit_best_index, accurate_in_limit_best_score = min(
                accurate_in_limit_scores, key=lambda item: item[1]
            )
        selection_mode_requested = (
            str(selection_override_mode).strip().lower()
            if selection_override_mode not in (None, '')
            else None
        )
        selection_mode_applied = False
        selection_mode_effective = 'default_score'
        selection_mode_fallback_reason = None
        if not forced_override_applied and selection_mode_requested is not None:
            # [caohy] Task 29：新增“诊断专用真实改选”开关，
            # 只在显式环境变量打开时，把理论上的 penalized / in-limit 最优支真正滚动下去，
            # 用来验证只读诊断与整段 seed 实际演化是否一致；默认行为保持不变。
            if selection_mode_requested == 'penalized_best':
                best_seed = penalized_best_index
                selection_mode_applied = True
                selection_mode_effective = 'penalized_best'
            elif selection_mode_requested == 'guarded_in_limit_best':
                if guarded_in_limit_best_index is not None:
                    best_seed = int(guarded_in_limit_best_index)
                    selection_mode_applied = True
                    selection_mode_effective = 'guarded_in_limit_best'
                else:
                    selection_mode_effective = 'default_score'
                    selection_mode_fallback_reason = 'guarded_in_limit_empty'
            elif selection_mode_requested == 'in_limit_best':
                if in_limit_best_index is not None:
                    best_seed = int(in_limit_best_index)
                    selection_mode_applied = True
                    selection_mode_effective = 'in_limit_best'
                else:
                    selection_mode_effective = 'default_score'
                    selection_mode_fallback_reason = 'in_limit_empty'
            elif selection_mode_requested == 'accurate_best':
                if accurate_best_index is not None:
                    best_seed = int(accurate_best_index)
                    selection_mode_applied = True
                    selection_mode_effective = 'accurate_best'
                else:
                    selection_mode_effective = 'default_score'
                    selection_mode_fallback_reason = 'accurate_empty'
            elif selection_mode_requested == 'accurate_in_limit_best':
                if accurate_in_limit_best_index is not None:
                    best_seed = int(accurate_in_limit_best_index)
                    selection_mode_applied = True
                    selection_mode_effective = 'accurate_in_limit_best'
                else:
                    selection_mode_effective = 'default_score'
                    selection_mode_fallback_reason = 'accurate_in_limit_empty'
            elif selection_mode_requested == 'guarded_accurate_best':
                if guarded_pose_goal_accurate_best_index is not None:
                    best_seed = int(guarded_pose_goal_accurate_best_index)
                    selection_mode_applied = True
                    selection_mode_effective = 'guarded_accurate_best'
                else:
                    selection_mode_effective = 'default_score'
                    selection_mode_fallback_reason = 'guarded_accurate_empty'
            elif selection_mode_requested == 'score_best':
                selection_mode_applied = True
                selection_mode_effective = 'score_best'
            else:
                selection_mode_effective = 'default_score'
                selection_mode_fallback_reason = f'unknown_mode:{selection_mode_requested}'
        # [caohy] Task 22 阶段1：为定位单样本重复复打仍漂移的根因，
        # 记录当前 waypoint 上全部可行 IK 候选的原始/包角后关节值与评分信息，
        # 后续可直接判断是 solver 返回顺序在漂，还是可行解集合本身就在变。
        candidate_debug_rows = []
        for idx in range(int(wrapped_solutions.shape[0])):
            candidate_debug_rows.append({
                'candidate_index': idx,
                'guard_passed': bool(guarded_mask[idx].item()),
                'step_jump_l2': round(float(solution_delta_l2[idx].item()), 6),
                'step_max_abs': round(float(torch.max(torch.abs(solution_delta[idx])).item()), 6),
                'selection_score': round(float(score[idx].item()), 6),
                'trend_cost': round(float(trend_cost[idx].item()), 6),
                'direction_penalty': round(float(direction_penalty[idx].item()), 6),
                'goal_proximity_cost': round(float(goal_proximity_cost[idx].item()), 6),
                'limit_violation_penalty': round(float(limit_penalty[idx].item()), 6),
                'limit_penalized_score': round(float(penalized_score[idx].item()), 6),
                'raw_joint_position': [
                    round(float(v), 6) for v in feasible_solutions[idx].detach().cpu().tolist()
                ],
                'wrapped_joint_position': [
                    round(float(v), 6) for v in wrapped_solutions[idx].detach().cpu().tolist()
                ],
                'joint_limit_summary': candidate_limit_summaries[idx],
                'pose_goal_summary': (
                    candidate_pose_goal_summaries[idx]
                    if candidate_pose_goal_summaries is not None and idx < len(candidate_pose_goal_summaries)
                    else None
                ),
                'pose_goal_accurate': bool(idx in pose_goal_accurate_candidate_indices),
            })

        return {
            'solution': wrapped_solutions[best_seed],
            'step_delta': solution_delta[best_seed],
            'step_jump_l2': float(solution_delta_l2[best_seed].item()),
            'step_max_abs': float(torch.max(torch.abs(solution_delta[best_seed])).item()),
            'selection_score': float(score[best_seed].item()),
            'trend_cost': float(trend_cost[best_seed].item()),
            'direction_penalty': float(direction_penalty[best_seed].item()),
            'goal_proximity_cost': float(goal_proximity_cost[best_seed].item()),
            'jump_guard_l2': float(jump_guard_l2),
            'guard_applied': bool(guard_applied),
            'guard_kept_count': int(candidate_indices.numel()),
            'nearest_index': nearest_seed,
            'nearest_step_jump_l2': float(solution_delta_l2[nearest_seed].item()),
            'nearest_step_max_abs': float(torch.max(torch.abs(solution_delta[nearest_seed])).item()),
            'nearest_joint_limit_summary': candidate_limit_summaries[nearest_seed],
            'nearest_pose_goal_summary': (
                candidate_pose_goal_summaries[nearest_seed]
                if candidate_pose_goal_summaries is not None and nearest_seed < len(candidate_pose_goal_summaries)
                else None
            ),
            'selected_index': best_seed,
            'selected_joint_limit_summary': candidate_limit_summaries[best_seed],
            'selected_pose_goal_summary': (
                candidate_pose_goal_summaries[best_seed]
                if candidate_pose_goal_summaries is not None and best_seed < len(candidate_pose_goal_summaries)
                else None
            ),
            'score_best_index': score_best_seed,
            'score_best_joint_limit_summary': candidate_limit_summaries[score_best_seed],
            'score_best_pose_goal_summary': (
                candidate_pose_goal_summaries[score_best_seed]
                if candidate_pose_goal_summaries is not None and score_best_seed < len(candidate_pose_goal_summaries)
                else None
            ),
            'guard_fallback_mode': 'guarded_candidates_only' if guard_applied else 'all_candidates_score',
            'candidate_debug_rows': candidate_debug_rows,
            'forced_candidate_index': forced_candidate_index,
            'forced_override_applied': forced_override_applied,
            'selection_mode_requested': selection_mode_requested,
            'selection_mode_applied': selection_mode_applied,
            'selection_mode_effective': selection_mode_effective,
            'selection_mode_fallback_reason': selection_mode_fallback_reason,
            'near_tie_override_applied': near_tie_override_applied,
            'near_tie_override_from': near_tie_override_from,
            'near_tie_override_to': near_tie_override_to,
            'near_tie_score_gap': near_tie_score_gap,
            'near_tie_direction_penalty_gap': near_tie_direction_penalty_gap,
            'near_tie_jump_gap': near_tie_jump_gap,
            'limit_aware_override_applied': limit_aware_override_applied,
            'limit_aware_override_from': limit_aware_override_from,
            'limit_aware_override_to': limit_aware_override_to,
            'in_limit_candidate_count': len(in_limit_candidate_indices),
            'in_limit_best_index': in_limit_best_index,
            'in_limit_best_score': in_limit_best_score,
            'pose_goal_position_tolerance_m': float(pose_goal_position_tolerance_m),
            'pose_goal_orientation_tolerance_deg': float(pose_goal_orientation_tolerance_deg),
            'pose_goal_accurate_candidate_count': len(pose_goal_accurate_candidate_indices),
            'pose_goal_accurate_best_index': accurate_best_index,
            'pose_goal_accurate_best_score': accurate_best_score,
            'limit_violation_penalty_weight': float(limit_violation_penalty_weight),
            'penalized_best_index': penalized_best_index,
            'penalized_best_score': penalized_best_score,
            'penalized_best_joint_limit_summary': candidate_limit_summaries[penalized_best_index],
            'penalized_best_pose_goal_summary': (
                candidate_pose_goal_summaries[penalized_best_index]
                if candidate_pose_goal_summaries is not None and penalized_best_index < len(candidate_pose_goal_summaries)
                else None
            ),
            'guarded_in_limit_candidate_count': len(guarded_in_limit_candidate_indices),
            'guarded_in_limit_best_index': guarded_in_limit_best_index,
            'guarded_in_limit_best_score': guarded_in_limit_best_score,
            'guarded_in_limit_best_joint_limit_summary': (
                candidate_limit_summaries[guarded_in_limit_best_index]
                if guarded_in_limit_best_index is not None else None
            ),
            'guarded_in_limit_best_pose_goal_summary': (
                candidate_pose_goal_summaries[guarded_in_limit_best_index]
                if (
                    candidate_pose_goal_summaries is not None
                    and guarded_in_limit_best_index is not None
                    and guarded_in_limit_best_index < len(candidate_pose_goal_summaries)
                ) else None
            ),
            'in_limit_best_joint_limit_summary': (
                candidate_limit_summaries[in_limit_best_index]
                if in_limit_best_index is not None else None
            ),
            'in_limit_best_pose_goal_summary': (
                candidate_pose_goal_summaries[in_limit_best_index]
                if (
                    candidate_pose_goal_summaries is not None
                    and in_limit_best_index is not None
                    and in_limit_best_index < len(candidate_pose_goal_summaries)
                ) else None
            ),
            'accurate_in_limit_candidate_count': len(accurate_in_limit_candidate_indices),
            'accurate_in_limit_best_index': accurate_in_limit_best_index,
            'accurate_in_limit_best_score': accurate_in_limit_best_score,
            'guarded_pose_goal_accurate_candidate_count': len(guarded_pose_goal_accurate_candidate_indices),
            'guarded_pose_goal_accurate_best_index': guarded_pose_goal_accurate_best_index,
            'guarded_pose_goal_accurate_best_score': guarded_pose_goal_accurate_best_score,
        }

    def _fk_single(self, joint_position):
        """单次 FK 查询。"""
        state = CuJointState.from_position(
            torch.tensor([joint_position], device='cuda:0', dtype=torch.float32),
            joint_names=self._joint_names,
        )
        kin_state = self._planner.compute_kinematics(state)
        tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
        pos = tool_pose.position.reshape(-1, 3).tolist()[0]
        quat = tool_pose.quaternion.reshape(-1, 4).tolist()[0]
        return pos + quat

    # [caohy] Phase 10：从轨迹点列表和 interpolation_dt 构造 JointTrajectory 消息。
    def _build_joint_trajectory(self, trajectory_points, interpolation_dt):
        """从轨迹点列表构造 JointTrajectory 消息。

        Args:
            trajectory_points: list of list，每个元素是一帧关节角。
            interpolation_dt: 帧间时间间隔（秒）。

        Returns:
            JointTrajectory 消息。
        """
        traj = JointTrajectory()
        traj.joint_names = self._joint_names

        for i, point_positions in enumerate(trajectory_points):
            pt = JointTrajectoryPoint()
            pt.positions = [float(p) for p in point_positions]
            total_ns = int(round(i * interpolation_dt * 1e9))
            pt.time_from_start = Duration(
                sec=total_ns // 1_000_000_000,
                nanosec=total_ns % 1_000_000_000,
            )
            traj.points.append(pt)

        return traj

    # [caohy] Phase 9.10/9.11：V2 版 plan_single_level_constrained 入口。
    # 整合 9.1~9.8 所有模块：端点检查 → 多候选求解 → seed 生成 → 评估 → 筛选 → 返回。
    def plan_single_level_constrained(
        self,
        start_joint,
        target_pose,
        hold_vec_weight=None,
        level_tolerance_deg=3.0,
        strict_level=True,
        num_candidates=4,
        enable_alignment_seed=True,
        speed_scale=None,
        start_state_debug=None,
    ):
        """V2 版完整约束规划。

        Args:
            start_joint: 起始关节角。
            target_pose: 目标位姿 [x,y,z,qw,qx,qy,qz]。
            hold_vec_weight: V1 语义 [rx,ry,rz,x,y,z]。
            level_tolerance_deg: 对齐容差（度）。
            strict_level: 是否严格要求对齐。
            num_candidates: 候选数量。
            enable_alignment_seed: 是否启用 alignment seed。
            speed_scale: 速度缩放。

        Returns:
            dict: 包含 trajectory_points, interpolation_dt, solve_time, status, level_check_info。
        """
        import time

        self._level_plan_request_index += 1
        plan_request_index = int(self._level_plan_request_index)
        diffusion_seed_mode = self._get_diffusion_seed_mode()
        diffusion_seed_config = self._get_diffusion_seed_runtime_config()

        # [caohy] Task 22 阶段1：补充“起点状态四联信息”，排查顺序回归里 seed 分支为何漂移。
        planning_start_state_debug = {
            'plan_request_index': plan_request_index,
            'current_joint_source': (
                (start_state_debug or {}).get('current_joint_source') or 'unknown'
            ),
            'service_start_joint': [round(float(v), 6) for v in list(start_joint)],
        }
        lifecycle_data = {
            'schema_version': 1,
            'plan_request_index': int(plan_request_index),
            'request': {
                'target_pose': [round(float(v), 6) for v in list(target_pose)],
                'hold_vec_weight': self._round_nested_debug_value(hold_vec_weight, float_digits=6),
                'level_tolerance_deg': round(float(level_tolerance_deg), 6),
                'strict_level': bool(strict_level),
                'num_candidates': int(num_candidates),
                'enable_alignment_seed': bool(enable_alignment_seed),
                'speed_scale': None if speed_scale is None else round(float(speed_scale), 6),
                'planner_legacy_branch_mode': self._get_planner_legacy_branch_mode(),
                'planner_topk_experiment_mode': self._get_planner_topk_experiment_mode(),
                'alignment_sequence_branch_mode': self._get_alignment_sequence_branch_mode(),
                'alignment_horizon_branch_mode': self._get_alignment_horizon_branch_mode(),
                'alignment_trajopt_family_mode': self._get_alignment_trajopt_family_mode(),
                'alignment_trajopt_family_topk_mode': (
                    self._get_alignment_trajopt_family_topk_mode()
                ),
                'alignment_trajopt_family_topk_shadow_mode': (
                    self._get_alignment_trajopt_family_topk_shadow_mode()
                ),
                'alignment_trajopt_legacy_mode': self._get_alignment_trajopt_legacy_mode(),
                'alignment_raw_seed_family_pool_mode': (
                    self._get_alignment_raw_seed_family_pool_mode()
                ),
                'allow_failed_solver_output_candidates': (
                    self._get_allow_failed_solver_output_candidates()
                ),
                'alignment_trajopt_family_formal_variant': (
                    self._get_alignment_trajopt_family_formal_variant()
                ),
                'diffusion_seed_mode': diffusion_seed_mode,
                'diffusion_seed_runtime_config': {
                    key: val
                    for key, val in diffusion_seed_config.items()
                    if key not in ('generated_samples_path', 'checkpoint_path')
                },
            },
            'start_state': planning_start_state_debug,
            'planner_attempts': [],
            'planner_legacy_attempts': [],
            'planner_topk_shadow': {},
            'seed_provider_reports': {},
            'seed_generation': {},
            'trajopt_attempts': [],
            'trajopt_family_branch': {},
            'trajopt_family_topk_branch': {},
            'trajopt_family_topk_shadow_branch': {},
            'sequence_branch': {},
            'horizon_branch': {},
            'candidates': [],
            'selection': {},
            'result': {},
        }
        lifecycle_data['seed_provider_reports']['diffusion_seed'] = (
            self._build_initial_diffusion_seed_report(diffusion_seed_config)
        )

        self.get_logger().info(
            f'Level constrained planning: target={[round(v, 3) for v in target_pose[:3]]}, '
            f'plan_request_index={plan_request_index}, '
            f'tolerance={level_tolerance_deg} deg, strict={strict_level}, '
            f'candidates={num_candidates}, seed={enable_alignment_seed}, '
            f'start_joint={planning_start_state_debug["service_start_joint"]}, '
            f'start_joint_source={planning_start_state_debug["current_joint_source"]}'
        )
        # [caohy] Task 36：仅本次人工 RViz 观察用的醒目标记，默认关闭；
        # 打开后只在指定 move 上刷一段大日志，帮助肉眼定位 Move 02 / Move 04 执行时刻。
        # [caohy] Task 31：marker_only_logs 模式下跳过 !!!!!!!! 标记，
        # 因为紧凑摘要行已经提供了 move 编号信息，不需要重复标记。
        if (
            not self._task36_marker_only_logs
            and os.environ.get('CUROBO_TASK36_MOVE_MARKER', '').strip().lower() in (
                '1',
                'true',
                'yes',
                'on',
            )
        ):
            marker_targets_text = os.environ.get('CUROBO_TASK36_MOVE_MARKER_TARGETS', '2,4')
            marker_targets = set()
            for item in marker_targets_text.split(','):
                item = item.strip()
                if not item:
                    continue
                try:
                    marker_targets.add(int(item))
                except ValueError:
                    pass
            if plan_request_index in marker_targets:
                marker_text = f'MOVE{plan_request_index:02d}'
                self.get_logger().error(
                    '\n\n\n\n\n'
                    f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {marker_text} '
                    '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                    f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {marker_text} '
                    '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n'
                    f'!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! {marker_text} '
                    '!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!'
                    '\n\n\n\n\n'
                )

        # 1. 端点对齐检查
        fk_start = self._fk_single(start_joint)
        start_quat = fk_start[3:7]
        target_quat = target_pose[3:7]

        endpoint_check = constraint_utils.check_alignment_endpoints(
            start_quat, target_quat, level_tolerance_deg,
        )
        lifecycle_data['endpoint_precheck'] = endpoint_check
        if not endpoint_check['valid']:
            self.get_logger().warn(f'Endpoint precheck failed: {endpoint_check["failure_reason"]}')
            return self._attach_lifecycle_artifact({
                'trajectory_points': [],
                'interpolation_dt': None,
                'solve_time': None,
                'status': 'failed_alignment_precheck',
                'level_check_info': {
                    'planning_status': 'failed_alignment_precheck',
                    'failure_reason': endpoint_check['failure_reason'],
                    'endpoint_alignment_check': endpoint_check,
                    'start_state_debug': planning_start_state_debug,
                },
            }, {
                **lifecycle_data,
                'result': {
                    'status': 'failed_alignment_precheck',
                    'failure_reason': endpoint_check['failure_reason'],
                },
            })

        # 2. 应用 hold_vec_weight 约束
        old_criteria = self._apply_hold_vec_weight(hold_vec_weight)

        all_candidates = []
        candidate_source_labels = []
        total_solve_time = 0.0
        seed_candidate_added = False
        seed_candidate_index = None
        seed_debug_info = None
        seed_trajopt_candidate_added = False
        seed_trajopt_candidate_index = None
        seed_trajopt_smoothed_candidate_added = False
        seed_trajopt_smoothed_candidate_index = None
        seed_trajopt_bridged_candidate_added = False
        seed_trajopt_bridged_candidate_index = None
        seed_trajopt_enabled = True

        try:
            # 3. 多候选求解
            t0 = time.time()
            candidates, planner_attempt_records = self._collect_planner_topk_main_candidates(
                start_joint,
                target_pose,
                level_tolerance_deg,
                num_candidates=min(4, int(num_candidates)),
            )
            total_solve_time += time.time() - t0
            lifecycle_data['planner_attempts'] = planner_attempt_records

            planner_pool_index = 0
            for attempt_record in lifecycle_data['planner_attempts']:
                if attempt_record.get('accepted_to_pool'):
                    attempt_record['pool_candidate_index'] = int(planner_pool_index)
                    planner_pool_index += 1
            for c in candidates:
                all_candidates.append(c)
                candidate_source_labels.append('planner')
            if self._get_planner_legacy_branch_mode() == 'candidate':
                legacy_candidates, legacy_attempt_records = self._collect_candidates(
                    start_joint,
                    target_pose,
                    num_candidates=num_candidates,
                    max_attempts=5,
                    source_label='planner_legacy',
                    generation_mode='top1_x4_legacy',
                )
                lifecycle_data['planner_legacy_attempts'] = legacy_attempt_records
                for c in legacy_candidates:
                    all_candidates.append(c)
                    candidate_source_labels.append('planner_legacy')
            if self._get_planner_topk_experiment_mode() == 'shadow':
                planner_topk_shadow_k = max(1, int(self._get_planner_topk_shadow_k()))
                lifecycle_data['planner_topk_shadow'] = self._collect_planner_topk_shadow_candidates(
                    start_joint,
                    target_pose,
                    level_tolerance_deg,
                    return_seeds=planner_topk_shadow_k,
                    num_seeds=planner_topk_shadow_k,
                )
            else:
                lifecycle_data['planner_topk_shadow'] = {
                    'branch_mode': self._get_planner_topk_experiment_mode(),
                    'status': 'disabled',
                    'success': False,
                    'failure_reason': None,
                    'attempts': [],
                }

            diffusion_report, diffusion_extra_time = self._run_diffusion_seed_provider_for_request(
                start_joint=start_joint,
                target_pose=target_pose,
                level_tolerance_deg=level_tolerance_deg,
                strict_level=strict_level,
                plan_request_index=plan_request_index,
                config=diffusion_seed_config,
                all_candidates=all_candidates,
                candidate_source_labels=candidate_source_labels,
            )
            total_solve_time += float(diffusion_extra_time)
            lifecycle_data['seed_provider_reports']['diffusion_seed'] = diffusion_report

            # 4. alignment seed 生成
            if enable_alignment_seed:
                try:
                    seed_result = self._generate_alignment_seed(
                        start_joint, target_pose,
                        num_waypoints=30,
                        alignment_tolerance_deg=level_tolerance_deg,
                        plan_request_index=plan_request_index,
                    )
                    if seed_result['success'] and seed_result['trajectory'] is not None:
                        # [caohy] Task 40：alignment_seed（单条原始种子）不再直接正式入池，
                        # 改由 family_1..4（四条原始家族种子）统一替代单条 alignment_seed 参与候选池。
                        # [caohy] S5：把 alignment seed 内部逐 waypoint 调试信息缓存下来，
                        # 后续透传到 level_check_info，定位最大单步跳变来自哪个 waypoint / IK 分支 / fallback。
                        seed_debug_info = {
                            'ik_fail_count': int(seed_result.get('ik_fail_count', 0)),
                            'num_waypoints': int(seed_result.get('num_waypoints', 0)),
                            'plan_request_index': seed_result.get('plan_request_index'),
                            'start_twist_deg': seed_result.get('start_twist_deg'),
                            'goal_twist_deg': seed_result.get('goal_twist_deg'),
                            'seed_input_start_joint': seed_result.get('seed_input_start_joint'),
                            'seed_first_joint': seed_result.get('seed_first_joint'),
                            'seed_last_joint': seed_result.get('seed_last_joint'),
                            'seed_last_joint_limit_summary': seed_result.get(
                                'seed_last_joint_limit_summary',
                            ),
                            'seed_trajectory_limit_summary': seed_result.get(
                                'seed_trajectory_limit_summary',
                            ),
                            'seed_alignment_discrete_summary': seed_result.get(
                                'seed_alignment_discrete_summary',
                            ),
                            'seed_alignment_interpolated_summary': seed_result.get(
                                'seed_alignment_interpolated_summary',
                            ),
                            'max_step_jump_l2': seed_result.get('max_step_jump_l2'),
                            'max_step_jump_index': seed_result.get('max_step_jump_index'),
                            'max_step_jump_source': seed_result.get('max_step_jump_source'),
                            'max_step_jump_delta': seed_result.get('max_step_jump_delta'),
                            'waypoint_debug': seed_result.get('waypoint_debug', []),
                        }
                        lifecycle_data['seed_generation'] = {
                            'success': True,
                            'status': 'success',
                            'failure_reason': None,
                            'seed_result_summary': self._round_nested_debug_value({
                                'ik_fail_count': seed_result.get('ik_fail_count', 0),
                                'num_waypoints': seed_result.get('num_waypoints', 0),
                                'start_twist_deg': seed_result.get('start_twist_deg'),
                                'goal_twist_deg': seed_result.get('goal_twist_deg'),
                                'raw_max_step_jump_l2': seed_result.get('raw_max_step_jump_l2'),
                                'raw_max_step_jump_index': seed_result.get('raw_max_step_jump_index'),
                                'raw_max_step_jump_delta': seed_result.get('raw_max_step_jump_delta'),
                            }, float_digits=6),
                            'raw_seed_trajectory': self._trajectory_tensor_to_list(
                                seed_result.get('raw_trajectory'),
                            ),
                            'raw_seed_summary': self._summarize_trajectory_points(
                                self._trajectory_tensor_to_list(seed_result.get('raw_trajectory')),
                            ),
                            'working_seed_trajectory': self._trajectory_tensor_to_list(
                                seed_result.get('trajectory'),
                            ),
                            'working_seed_summary': self._summarize_trajectory_points(
                                self._trajectory_tensor_to_list(seed_result.get('trajectory')),
                            ),
                            'seed_debug_info': seed_debug_info,
                        }
                        self.get_logger().info(f'Alignment seed added: {seed_result["trajectory"].shape[0]} steps')
                        family_baseline_seed_result = None
                        if seed_trajopt_enabled:
                            try:
                                legacy_trajopt_mode = self._get_alignment_trajopt_legacy_mode()
                                family_branch_bundle = self._generate_alignment_seed_families(
                                    start_joint,
                                    target_pose,
                                    num_waypoints=30,
                                    alignment_tolerance_deg=level_tolerance_deg,
                                    plan_request_index=plan_request_index,
                                    force_generate=True,
                                )
                                family_branch_mode = self._get_alignment_trajopt_family_mode()
                                family_baseline_attempt = next(
                                    (
                                        attempt for attempt in family_branch_bundle.get('attempts', [])
                                        if str(attempt.get('source_label')) == 'alignment_seed_trajopt_1'
                                    ),
                                    None,
                                )
                                family_baseline_seed_result = (
                                    family_baseline_attempt.get('_seed_result')
                                    if isinstance(family_baseline_attempt, dict) else None
                                )
                                raw_seed_family_pool_mode = (
                                    self._get_alignment_raw_seed_family_pool_mode()
                                )
                                family_branch_bundle['raw_seed_family_pool_mode'] = (
                                    raw_seed_family_pool_mode
                                )
                                family_in_pool_labels = []
                                for family_attempt in family_branch_bundle.get('attempts', []):
                                    if not isinstance(family_attempt, dict):
                                        continue
                                    raw_seed_result = family_attempt.get('_seed_result')
                                    family_raw_traj = (
                                        raw_seed_result.get('raw_trajectory')
                                        if isinstance(raw_seed_result, dict) else None
                                    )
                                    if family_raw_traj is None and isinstance(raw_seed_result, dict):
                                        family_raw_traj = raw_seed_result.get('trajectory')
                                    if not bool(family_attempt.get('seed_generation_success')) or family_raw_traj is None:
                                        family_attempt['candidate_pool_accepted'] = False
                                        family_attempt.pop('pool_candidate_index', None)
                                        continue
                                    pool_source_label = str(
                                        family_attempt.get('raw_pool_source_label')
                                        or family_attempt.get('source_label')
                                    )
                                    family_attempt['pool_source_label'] = pool_source_label
                                    family_attempt['raw_seed_family_pool_mode'] = raw_seed_family_pool_mode
                                    if raw_seed_family_pool_mode == 'candidate':
                                        all_candidates.append(family_raw_traj)
                                        candidate_source_labels.append(pool_source_label)
                                        family_attempt['candidate_pool_accepted'] = True
                                        family_attempt['pool_candidate_index'] = int(
                                            len(all_candidates) - 1
                                        )
                                        family_in_pool_labels.append(pool_source_label)
                                    else:
                                        family_attempt['candidate_pool_accepted'] = False
                                        family_attempt.pop('pool_candidate_index', None)
                                        family_attempt['raw_candidate_pool_enabled'] = False
                                        family_attempt['pool_disabled_reason'] = (
                                            'raw_seed_family_pool_mode_off'
                                        )
                                family_branch_bundle['family_in_pool_labels'] = family_in_pool_labels
                                if family_in_pool_labels:
                                    self.get_logger().info(
                                        'Alignment seed family raw candidates added: '
                                        f'{", ".join(family_in_pool_labels)}'
                                    )
                                elif raw_seed_family_pool_mode == 'off':
                                    self.get_logger().info(
                                        'Alignment seed family raw candidates kept as seeds only '
                                        '(CUROBO_ALIGNMENT_RAW_SEED_FAMILY_POOL_MODE=off)'
                                    )
                                split_seed_result = family_baseline_seed_result or seed_result
                                raw_jump_l2_val = float(split_seed_result.get('raw_max_step_jump_l2') or 0.0)
                                raw_jump_idx = split_seed_result.get('raw_max_step_jump_index')
                                raw_traj_for_split = split_seed_result.get('raw_trajectory')
                                split_candidate = None
                                split_attempt = None
                                if (
                                    raw_jump_l2_val >= 1.0
                                    and raw_jump_idx is not None
                                    and raw_traj_for_split is not None
                                ):
                                    # [caohy] Task 6：当前已经确认 Move 01 / 02 / 10 这类跨构型大跳 case，
                                    # 真正能救回的是 split（分段优化），而不是把整条 raw seed 再塞回 pose trajopt。
                                    # 所以这里先前移 split，若成功就直接入池并跳过已知会塌的 raw trajopt。
                                    t_split = time.time()
                                    try:
                                        split_attempt = self._optimize_alignment_seed_split(
                                            raw_traj_for_split,
                                            raw_jump_idx,
                                            alignment_tolerance_deg=level_tolerance_deg,
                                            probe_label='alignment_seed_trajopt_split',
                                            attempt_stage='pre_pose_trajopt',
                                        )
                                    except Exception as e:
                                        self.get_logger().warn(f'[Task6-split-first] split optimize failed: {e}')
                                        split_attempt = {
                                            'probe_label': 'alignment_seed_trajopt_split',
                                            'success': False,
                                            'status': 'exception',
                                            'attempt_stage': 'pre_pose_trajopt',
                                            'failure_reason': str(e),
                                            'exception_type': type(e).__name__,
                                            'split_seed_source_label': (
                                                'alignment_seed_trajopt_1'
                                                if family_baseline_seed_result is not None else 'alignment_seed'
                                            ),
                                        }
                                    if split_attempt is not None:
                                        split_attempt['split_seed_source_label'] = (
                                            'alignment_seed_trajopt_1'
                                            if family_baseline_seed_result is not None else 'alignment_seed'
                                        )
                                    total_solve_time += time.time() - t_split
                                    lifecycle_data['trajopt_attempts'].append(
                                        self._round_nested_debug_value(
                                            {
                                                key: val for key, val in split_attempt.items()
                                                if key != 'trajectory'
                                            },
                                            float_digits=6,
                                        ),
                                    )
                                    split_candidate = split_attempt.get('trajectory')
                                    if split_attempt.get('success') and split_candidate is not None:
                                        all_candidates.append(split_candidate)
                                        candidate_source_labels.append('alignment_seed_trajopt_split')
                                        seed_trajopt_candidate_added = True
                                        seed_trajopt_candidate_index = len(all_candidates) - 1
                                        self.get_logger().info(
                                            'Alignment seed trajopt split-first candidate added: '
                                            f'{split_candidate.shape[0]} steps'
                                        )

                                optimized_seed_candidate = None
                                if legacy_trajopt_mode == 'off':
                                    lifecycle_data['trajopt_attempts'].append({
                                        'probe_label': 'alignment_seed_trajopt',
                                        'success': False,
                                        'status': 'legacy_branch_disabled',
                                        'failure_reason': 'alignment_trajopt_legacy_mode_off',
                                    })
                                elif split_candidate is not None:
                                    lifecycle_data['trajopt_attempts'].append({
                                        'probe_label': 'alignment_seed_trajopt',
                                        'success': False,
                                        'status': 'skipped_due_split_first_success',
                                        'failure_reason': 'raw_seed_large_jump_split_first_succeeded',
                                        'raw_seed_jump_l2': round(raw_jump_l2_val, 6),
                                        'raw_seed_jump_index': int(raw_jump_idx),
                                    })
                                else:
                                    t1 = time.time()
                                    legacy_seed_result = family_baseline_seed_result or seed_result
                                    optimized_seed_attempt = self._optimize_alignment_seed(
                                        start_joint,
                                        target_pose,
                                        legacy_seed_result['trajectory'],
                                        alignment_tolerance_deg=level_tolerance_deg,
                                        probe_label='alignment_seed_trajopt',
                                    )
                                    total_solve_time += time.time() - t1
                                    lifecycle_data['trajopt_attempts'].append(
                                        self._round_nested_debug_value(
                                            {
                                                key: val for key, val in optimized_seed_attempt.items()
                                                if key != 'trajectory'
                                            },
                                            float_digits=6,
                                        ),
                                    )
                                    optimized_seed_candidate = optimized_seed_attempt.get('trajectory')
                                    if (
                                        legacy_trajopt_mode == 'candidate'
                                        and optimized_seed_attempt.get('success')
                                        and optimized_seed_candidate is not None
                                    ):
                                        all_candidates.append(optimized_seed_candidate)
                                        candidate_source_labels.append('alignment_seed_trajopt')
                                        seed_trajopt_candidate_added = True
                                        seed_trajopt_candidate_index = len(all_candidates) - 1
                                        self.get_logger().info(
                                            f'Alignment seed trajopt candidate added: '
                                            f'{optimized_seed_candidate.shape[0]} steps'
                                        )

                                family_topk_mode = self._get_alignment_trajopt_family_topk_mode()
                                family_topk_branch_bundle = self._optimize_alignment_seed_families_topk(
                                    start_joint,
                                    target_pose,
                                    family_branch_bundle,
                                    alignment_tolerance_deg=level_tolerance_deg,
                                )
                                if family_topk_mode == 'candidate':
                                    topk_in_pool_labels = []
                                    for topk_attempt in family_topk_branch_bundle.get('attempts', []):
                                        topk_traj = topk_attempt.get('trajectory')
                                        if topk_attempt.get('success') and topk_traj is not None:
                                            all_candidates.append(topk_traj)
                                            topk_attempt['candidate_pool_accepted'] = True
                                            topk_attempt['pool_candidate_index'] = int(
                                                len(all_candidates) - 1
                                            )
                                            candidate_source_labels.append(
                                                str(topk_attempt.get('source_label'))
                                            )
                                            topk_in_pool_labels.append(
                                                str(topk_attempt.get('source_label'))
                                            )
                                        else:
                                            topk_attempt['candidate_pool_accepted'] = False
                                    family_topk_branch_bundle['family_in_pool_labels'] = (
                                        topk_in_pool_labels
                                    )
                                else:
                                    for topk_attempt in family_topk_branch_bundle.get('attempts', []):
                                        topk_attempt['candidate_pool_accepted'] = False
                                        topk_attempt.pop('pool_candidate_index', None)
                                    family_topk_branch_bundle['family_in_pool_labels'] = []
                                self._set_alignment_trajopt_family_topk_branch_record(
                                    lifecycle_data,
                                    self._sanitize_alignment_seed_family_bundle(
                                        family_topk_branch_bundle
                                    ),
                                )
                                self._set_alignment_trajopt_family_topk_shadow_branch_record(
                                    lifecycle_data,
                                    self._optimize_alignment_seed_families_topk_shadow(
                                        start_joint,
                                        target_pose,
                                        family_branch_bundle,
                                        alignment_tolerance_deg=level_tolerance_deg,
                                    ),
                                )
                                self._set_alignment_trajopt_family_branch_record(
                                    lifecycle_data,
                                    self._sanitize_alignment_seed_family_bundle(family_branch_bundle),
                                )
                                for family_attempt in lifecycle_data['trajopt_family_branch'].get('attempts', []):
                                    lifecycle_data['trajopt_attempts'].append(
                                        self._round_nested_debug_value(
                                            dict(family_attempt),
                                            float_digits=6,
                                        )
                                    )
                                for topk_attempt in lifecycle_data['trajopt_family_topk_branch'].get('attempts', []):
                                    lifecycle_data['trajopt_attempts'].append(
                                        self._round_nested_debug_value(
                                            dict(topk_attempt),
                                            float_digits=6,
                                        )
                                    )

                                if (
                                    legacy_trajopt_mode != 'off'
                                    and optimized_seed_candidate is None
                                    and split_candidate is None
                                    and float(seed_result.get('max_step_jump_l2') or 0.0) >= 1.0
                                ):
                                        smoothed_seed = self._smooth_seed_traj_for_trajopt(seed_result['trajectory'])
                                        raw_summary = self._summarize_seed_step_metrics(seed_result['trajectory'])
                                        smooth_summary = self._summarize_seed_step_metrics(smoothed_seed)
                                        self._log_probe_info(
                                            '[S5-B] Retrying trajopt with smoothed alignment seed: '
                                            f'raw_max_step_jump_l2={raw_summary["max_step_jump_l2"]}, '
                                            f'smoothed_max_step_jump_l2={smooth_summary["max_step_jump_l2"]}, '
                                            f'raw_jump_index={raw_summary["max_step_jump_index"]}, '
                                            f'smoothed_jump_index={smooth_summary["max_step_jump_index"]}'
                                        )
                                        t2 = time.time()
                                        optimized_smoothed_seed_attempt = self._optimize_alignment_seed(
                                            start_joint,
                                            target_pose,
                                            smoothed_seed,
                                            alignment_tolerance_deg=level_tolerance_deg,
                                            probe_label='alignment_seed_trajopt_smoothed',
                                        )
                                        total_solve_time += time.time() - t2
                                        lifecycle_data['trajopt_attempts'].append(
                                            self._round_nested_debug_value(
                                                {
                                                    key: val for key, val in optimized_smoothed_seed_attempt.items()
                                                    if key != 'trajectory'
                                                },
                                                float_digits=6,
                                            ),
                                        )
                                        optimized_smoothed_seed_candidate = (
                                            optimized_smoothed_seed_attempt.get('trajectory')
                                        )
                                        if (
                                            optimized_smoothed_seed_attempt.get('success')
                                            and optimized_smoothed_seed_candidate is not None
                                            and legacy_trajopt_mode == 'candidate'
                                        ):
                                            all_candidates.append(optimized_smoothed_seed_candidate)
                                            candidate_source_labels.append('alignment_seed_trajopt_smoothed')
                                            seed_trajopt_smoothed_candidate_added = True
                                            seed_trajopt_smoothed_candidate_index = len(all_candidates) - 1
                                            self.get_logger().info(
                                                'Alignment seed trajopt smoothed candidate added: '
                                                f'{optimized_smoothed_seed_candidate.shape[0]} steps'
                                            )
                                        elif raw_summary.get('max_step_jump_index') is not None:
                                            bridged_seed = self._bridge_seed_jump_for_trajopt(
                                                seed_result['trajectory'],
                                                raw_summary.get('max_step_jump_index'),
                                                bridge_radius=2,
                                            )
                                            bridged_summary = self._summarize_seed_step_metrics(bridged_seed)
                                            self._log_probe_info(
                                                '[S5-B] Retrying trajopt with bridged alignment seed: '
                                                f'raw_max_step_jump_l2={raw_summary["max_step_jump_l2"]}, '
                                                f'bridged_max_step_jump_l2={bridged_summary["max_step_jump_l2"]}, '
                                                f'raw_jump_index={raw_summary["max_step_jump_index"]}, '
                                                f'bridged_jump_index={bridged_summary["max_step_jump_index"]}'
                                            )
                                            t3 = time.time()
                                            optimized_bridged_seed_attempt = self._optimize_alignment_seed(
                                                start_joint,
                                                target_pose,
                                                bridged_seed,
                                                alignment_tolerance_deg=level_tolerance_deg,
                                                probe_label='alignment_seed_trajopt_bridged',
                                            )
                                            total_solve_time += time.time() - t3
                                            lifecycle_data['trajopt_attempts'].append(
                                                self._round_nested_debug_value(
                                                    {
                                                        key: val for key, val in optimized_bridged_seed_attempt.items()
                                                        if key != 'trajectory'
                                                    },
                                                    float_digits=6,
                                                ),
                                            )
                                            optimized_bridged_seed_candidate = (
                                                optimized_bridged_seed_attempt.get('trajectory')
                                            )
                                            if (
                                                optimized_bridged_seed_attempt.get('success')
                                                and optimized_bridged_seed_candidate is not None
                                                and legacy_trajopt_mode == 'candidate'
                                            ):
                                                all_candidates.append(optimized_bridged_seed_candidate)
                                                candidate_source_labels.append('alignment_seed_trajopt_bridged')
                                                seed_trajopt_bridged_candidate_added = True
                                                seed_trajopt_bridged_candidate_index = len(all_candidates) - 1
                                                self.get_logger().info(
                                                    'Alignment seed trajopt bridged candidate added: '
                                                    f'{optimized_bridged_seed_candidate.shape[0]} steps'
                                                )
                            except Exception as e:
                                self.get_logger().warn(f'Alignment seed trajopt failed: {e}')
                                lifecycle_data['trajopt_attempts'].append({
                                    'probe_label': 'alignment_seed_trajopt',
                                    'success': False,
                                    'status': 'exception',
                                    'failure_reason': str(e),
                                    'exception_type': type(e).__name__,
                                })
                                if not lifecycle_data.get('trajopt_family_topk_shadow_branch'):
                                    lifecycle_data['trajopt_family_topk_shadow_branch'] = {
                                        'branch_mode': self._get_alignment_trajopt_family_topk_shadow_mode(),
                                        'branch_enabled': (
                                            self._get_alignment_trajopt_family_topk_shadow_mode() != 'off'
                                        ),
                                        'success': False,
                                        'status': 'upstream_family_branch_exception',
                                        'failure_reason': str(e),
                                        'exception_type': type(e).__name__,
                                        'attempts': [],
                                    }
                                if not lifecycle_data.get('trajopt_family_topk_branch'):
                                    lifecycle_data['trajopt_family_topk_branch'] = {
                                        'branch_mode': self._get_alignment_trajopt_family_topk_mode(),
                                        'branch_enabled': (
                                            self._get_alignment_trajopt_family_topk_mode() != 'off'
                                        ),
                                        'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
                                        'success': False,
                                        'status': 'upstream_family_branch_exception',
                                        'failure_reason': str(e),
                                        'exception_type': type(e).__name__,
                                        'family_in_pool_labels': [],
                                        'family_selected_label': None,
                                        'attempts': [],
                                    }
                        else:
                            lifecycle_data['trajopt_family_branch'] = {
                                'branch_mode': self._get_alignment_trajopt_family_mode(),
                                'branch_enabled': False,
                                'success': False,
                                'status': 'alignment_seed_trajopt_disabled',
                                'failure_reason': 'seed_trajopt_disabled',
                                'family_success_count': 0,
                                'family_in_pool_labels': [],
                                'family_selected_label': None,
                                'attempts': [],
                            }
                            lifecycle_data['trajopt_family_topk_branch'] = {
                                'branch_mode': self._get_alignment_trajopt_family_topk_mode(),
                                'branch_enabled': False,
                                'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
                                'success': False,
                                'status': 'alignment_seed_trajopt_disabled',
                                'failure_reason': 'seed_trajopt_disabled',
                                'family_in_pool_labels': [],
                                'family_selected_label': None,
                                'attempts': [],
                            }
                            lifecycle_data['trajopt_family_topk_shadow_branch'] = {
                                'branch_mode': self._get_alignment_trajopt_family_topk_shadow_mode(),
                                'branch_enabled': False,
                                'success': False,
                                'status': 'alignment_seed_trajopt_disabled',
                                'failure_reason': 'seed_trajopt_disabled',
                                'attempts': [],
                            }

                        # [caohy] Task 10：sequence（序列目标）旁支正式挂在 alignment_seed_trajopt 之后，
                        # 当前先完成开关收口和 sequence goal（序列目标）构造/落盘，后续再接 solve_sequence。
                        sequence_source_seed_result = family_baseline_seed_result or seed_result
                        sequence_branch_record = self._build_alignment_sequence_seed_prepare_record(
                            sequence_source_seed_result.get('raw_trajectory'),
                        )
                        if (
                            sequence_branch_record.get('branch_enabled')
                            and sequence_branch_record.get('status') == 'prepared'
                        ):
                            try:
                                sequence_branch_mode = self._get_alignment_sequence_branch_mode()
                                sequence_bundle = self._optimize_alignment_seed_sequence_candidates(
                                    start_joint,
                                    target_pose,
                                    sequence_source_seed_result.get('raw_trajectory'),
                                    alignment_tolerance_deg=level_tolerance_deg,
                                    probe_label='alignment_seed_sequence',
                                )
                                if sequence_branch_mode == 'candidate':
                                    for attempt in sequence_bundle.get('attempts', []):
                                        attempt_traj = attempt.get('trajectory')
                                        if attempt.get('success') and attempt_traj is not None:
                                            all_candidates.append(attempt_traj)
                                            candidate_source_labels.append(
                                                str(attempt.get('source_label'))
                                            )
                                            attempt['candidate_pool_accepted'] = True
                                else:
                                    for attempt in sequence_bundle.get('attempts', []):
                                        attempt['candidate_pool_accepted'] = False
                                sequence_branch_record.update({
                                    key: val
                                    for key, val in sequence_bundle.items()
                                    if key not in ('attempts',)
                                })
                                sequence_branch_record['branch_mode'] = sequence_branch_mode
                                sequence_branch_record['attempts'] = self._round_nested_debug_value(
                                    [
                                        {
                                            key: val for key, val in attempt.items()
                                            if key != 'trajectory'
                                        }
                                        for attempt in sequence_bundle.get('attempts', [])
                                    ],
                                    float_digits=6,
                                )
                            except Exception as e:
                                sequence_branch_record['success'] = False
                                sequence_branch_record['status'] = 'sequence_branch_prepare_exception'
                                sequence_branch_record['failure_reason'] = str(e)
                                sequence_branch_record['exception_type'] = type(e).__name__
                        self._set_sequence_branch_record(
                            lifecycle_data,
                            sequence_branch_record,
                        )
                    else:
                        lifecycle_data['trajopt_family_branch'] = {
                            'branch_mode': self._get_alignment_trajopt_family_mode(),
                            'branch_enabled': self._get_alignment_trajopt_family_mode() != 'off',
                            'success': False,
                            'status': 'seed_generation_failed',
                            'failure_reason': 'alignment_seed_generation_failed',
                            'family_success_count': 0,
                            'family_in_pool_labels': [],
                            'family_selected_label': None,
                            'attempts': [],
                        }
                        lifecycle_data['trajopt_family_topk_branch'] = {
                            'branch_mode': self._get_alignment_trajopt_family_topk_mode(),
                            'branch_enabled': self._get_alignment_trajopt_family_topk_mode() != 'off',
                            'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
                            'success': False,
                            'status': 'seed_generation_failed',
                            'failure_reason': 'alignment_seed_generation_failed',
                            'family_in_pool_labels': [],
                            'family_selected_label': None,
                            'attempts': [],
                        }
                        lifecycle_data['trajopt_family_topk_shadow_branch'] = {
                            'branch_mode': self._get_alignment_trajopt_family_topk_shadow_mode(),
                            'branch_enabled': self._get_alignment_trajopt_family_topk_shadow_mode() != 'off',
                            'success': False,
                            'status': 'seed_generation_failed',
                            'failure_reason': 'alignment_seed_generation_failed',
                            'attempts': [],
                        }
                        lifecycle_data['seed_generation'] = {
                            'success': False,
                            'status': 'seed_generation_failed',
                            'failure_reason': seed_result.get('failure_reason'),
                            'ik_fail_count': int(seed_result.get('ik_fail_count', 0)),
                        }
                        self._set_sequence_branch_record(lifecycle_data, {
                            'source_label': 'alignment_seed_sequence',
                            'branch_mode': self._get_alignment_sequence_branch_mode(),
                            'branch_enabled': self._get_alignment_sequence_branch_mode() != 'off',
                            'success': False,
                            'status': 'seed_generation_failed',
                            'failure_reason': 'alignment_seed_generation_failed',
                        })
                except Exception as e:
                    self.get_logger().warn(f'Seed generation failed: {e}')
                    lifecycle_data['trajopt_family_branch'] = {
                        'branch_mode': self._get_alignment_trajopt_family_mode(),
                        'branch_enabled': self._get_alignment_trajopt_family_mode() != 'off',
                        'success': False,
                        'status': 'seed_generation_exception',
                        'failure_reason': str(e),
                        'exception_type': type(e).__name__,
                        'family_success_count': 0,
                        'family_in_pool_labels': [],
                        'family_selected_label': None,
                        'attempts': [],
                    }
                    lifecycle_data['trajopt_family_topk_branch'] = {
                        'branch_mode': self._get_alignment_trajopt_family_topk_mode(),
                        'branch_enabled': self._get_alignment_trajopt_family_topk_mode() != 'off',
                        'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
                        'success': False,
                        'status': 'seed_generation_exception',
                        'failure_reason': str(e),
                        'exception_type': type(e).__name__,
                        'family_in_pool_labels': [],
                        'family_selected_label': None,
                        'attempts': [],
                    }
                    lifecycle_data['trajopt_family_topk_shadow_branch'] = {
                        'branch_mode': self._get_alignment_trajopt_family_topk_shadow_mode(),
                        'branch_enabled': self._get_alignment_trajopt_family_topk_shadow_mode() != 'off',
                        'success': False,
                        'status': 'seed_generation_exception',
                        'failure_reason': str(e),
                        'exception_type': type(e).__name__,
                        'attempts': [],
                    }
                    lifecycle_data['seed_generation'] = {
                        'success': False,
                        'status': 'exception',
                        'failure_reason': str(e),
                        'exception_type': type(e).__name__,
                    }
                    self._set_sequence_branch_record(lifecycle_data, {
                        'source_label': 'alignment_seed_sequence',
                        'branch_mode': self._get_alignment_sequence_branch_mode(),
                        'branch_enabled': self._get_alignment_sequence_branch_mode() != 'off',
                        'success': False,
                        'status': 'seed_generation_exception',
                        'failure_reason': str(e),
                        'exception_type': type(e).__name__,
                    })
            else:
                lifecycle_data['trajopt_family_branch'] = {
                    'branch_mode': self._get_alignment_trajopt_family_mode(),
                    'branch_enabled': False,
                    'success': False,
                    'status': 'alignment_seed_disabled',
                    'failure_reason': 'enable_alignment_seed_false',
                    'family_success_count': 0,
                    'family_in_pool_labels': [],
                    'family_selected_label': None,
                    'attempts': [],
                }
                lifecycle_data['trajopt_family_topk_branch'] = {
                    'branch_mode': self._get_alignment_trajopt_family_topk_mode(),
                    'branch_enabled': False,
                    'formal_variant': self._get_alignment_trajopt_family_formal_variant(),
                    'success': False,
                    'status': 'alignment_seed_disabled',
                    'failure_reason': 'enable_alignment_seed_false',
                    'family_in_pool_labels': [],
                    'family_selected_label': None,
                    'attempts': [],
                }
                lifecycle_data['trajopt_family_topk_shadow_branch'] = {
                    'branch_mode': self._get_alignment_trajopt_family_topk_shadow_mode(),
                    'branch_enabled': False,
                    'success': False,
                    'status': 'alignment_seed_disabled',
                    'failure_reason': 'enable_alignment_seed_false',
                    'attempts': [],
                }
                self._set_sequence_branch_record(lifecycle_data, {
                    'source_label': 'alignment_seed_sequence',
                    'branch_mode': self._get_alignment_sequence_branch_mode(),
                    'branch_enabled': False,
                    'success': False,
                    'status': 'alignment_seed_disabled',
                    'failure_reason': 'enable_alignment_seed_false',
                })

        finally:
            # 5. 恢复约束
            self._restore_criteria(old_criteria)

        if not all_candidates:
            self.get_logger().warn('No successful candidates')
            return self._attach_lifecycle_artifact({
                'trajectory_points': [],
                'interpolation_dt': None,
                'solve_time': total_solve_time,
                'status': 'curobo_failed',
                'level_check_info': {
                    'planning_status': 'curobo_failed',
                    'failure_reason': 'No successful candidates',
                    'start_state_debug': planning_start_state_debug,
                },
            }, {
                **lifecycle_data,
                'result': {
                    'status': 'curobo_failed',
                    'failure_reason': 'No successful candidates',
                },
            })

        # 6. 单候选快速返回
        if len(all_candidates) == 1:
            traj = all_candidates[0]
            self.get_logger().info('Single candidate, skipping alignment selection')
            # [caohy] Task 30：单候选路径过去不会回写来源标签，导致“唯一候选到底是谁”只能靠日志推断。
            single_source_label = (
                str(candidate_source_labels[0])
                if candidate_source_labels else 'unknown'
            )
            single_goal_summary = self._summarize_pose_against_goal(
                traj[-1:].to(device='cuda:0', dtype=torch.float32),
                torch.tensor(target_pose[:3], device='cuda:0', dtype=torch.float32),
                target_pose[3:7],
            )
            goal_position_tolerance_m, goal_orientation_tolerance_deg = (
                self._get_selector_goal_gate_tolerances()
            )
            single_goal_gate_summary = self._build_selector_goal_gate_summary(
                single_goal_summary,
                goal_position_tolerance_m,
                goal_orientation_tolerance_deg,
            )
            single_goal_pose_valid = bool(
                single_goal_gate_summary['candidate_goal_pose_valid'][0]
            )
            self._mark_sequence_branch_attempt_selected(
                lifecycle_data,
                single_source_label,
            )
            self._mark_alignment_trajopt_family_attempt_selected(
                lifecycle_data,
                single_source_label,
            )
            self._mark_alignment_trajopt_family_topk_attempt_selected(
                lifecycle_data,
                single_source_label,
            )
            single_candidate_points = self._trajectory_tensor_to_list(traj)
            lifecycle_candidates = [build_lifecycle_candidate_record(
                base_record={
                'candidate_id': 'candidate_0',
                'candidate_index': 0,
                'source_label': single_source_label,
                'entered_pool': True,
                'selected': bool(single_goal_pose_valid),
                'alignment_valid': bool(single_goal_pose_valid),
                'max_alignment_deviation_deg': None,
                'mean_alignment_deviation_deg': None,
                'goal_pose_valid': bool(single_goal_pose_valid),
                'position_error_m': single_goal_gate_summary['candidate_position_error_m'][0],
                'orientation_error_deg': (
                    single_goal_gate_summary['candidate_orientation_error_deg'][0]
                ),
                'start_joint_gap_l2': None,
                'joint_step_jump_cost': None,
                'joint_step_max_l2': None,
                'joint_step_max_abs': None,
                'twist_smoothness_cost': None,
                'terminal_goal_pose_summary': (
                    single_goal_summary[0] if single_goal_summary else None
                ),
                'trajectory_points': single_candidate_points,
                'trajectory_summary': self._summarize_trajectory_points(
                    single_candidate_points,
                ),
                },
                source_label=single_source_label,
                trajectory_points=single_candidate_points,
                metrics_keys=(
                    'alignment_valid',
                    'max_alignment_deviation_deg',
                    'mean_alignment_deviation_deg',
                    'goal_pose_valid',
                    'position_error_m',
                    'orientation_error_deg',
                    'start_joint_gap_l2',
                    'joint_step_jump_cost',
                    'joint_step_max_l2',
                    'joint_step_max_abs',
                    'twist_smoothness_cost',
                ),
                metadata={
                    'phase1_adapter': 'single_candidate_path',
                    'behavior_changed': False,
                },
            )]
            self._sync_diffusion_seed_report_with_selection(
                lifecycle_data,
                lifecycle_candidates,
            )
            self._sync_alignment_trajopt_family_attempt_pool_status(
                lifecycle_data,
                lifecycle_candidates,
            )
            self._sync_alignment_trajopt_family_topk_attempt_pool_status(
                lifecycle_data,
                lifecycle_candidates,
            )
            self._sync_alignment_trajopt_family_topk_shadow_attempt_status(
                lifecycle_data,
                [traj],
                [single_source_label],
                start_joint,
                target_pose,
                level_tolerance_deg,
                strict_level,
                actual_selected_source_label=single_source_label,
            )
            if not single_goal_pose_valid:
                self._mark_sequence_branch_attempt_selected(
                    lifecycle_data,
                    None,
                )
                self._mark_alignment_trajopt_family_attempt_selected(
                    lifecycle_data,
                    None,
                )
                self._mark_alignment_trajopt_family_topk_attempt_selected(
                    lifecycle_data,
                    None,
                )
                failure_reason = (
                    'single_candidate_goal_pose_invalid: '
                    f'position_error_m={single_goal_gate_summary["candidate_position_error_m"][0]}, '
                    f'orientation_error_deg={single_goal_gate_summary["candidate_orientation_error_deg"][0]}, '
                    f'goal_position_tolerance_m={single_goal_gate_summary["goal_position_tolerance_m"]}, '
                    f'goal_orientation_tolerance_deg={single_goal_gate_summary["goal_orientation_tolerance_deg"]}'
                )
                return self._attach_lifecycle_artifact({
                    'trajectory_points': [],
                    'interpolation_dt': None,
                    'solve_time': total_solve_time,
                    'status': 'failed_goal_pose_constraint',
                    'level_check_info': {
                        'planning_status': 'failed_goal_pose_constraint',
                        'candidate_count': 1,
                        'alignment_valid_count': 0,
                        'candidate_source_labels': list(candidate_source_labels),
                        'selected_index': None,
                        'selected_source_label': None,
                        **single_goal_gate_summary,
                        'failure_reason': failure_reason,
                    },
                }, {
                    **lifecycle_data,
                    'candidates': lifecycle_candidates,
                    'selection': {
                        'planning_status': 'failed_goal_pose_constraint',
                        'candidate_count': 1,
                        'alignment_valid_count': 0,
                        'candidate_source_labels': list(candidate_source_labels),
                        'selected_index': None,
                        'selected_source_label': None,
                        **single_goal_gate_summary,
                        'failure_reason': failure_reason,
                    },
                    'result': {
                        'status': 'failed_goal_pose_constraint',
                        'failure_reason': failure_reason,
                    },
                })
            return self._attach_lifecycle_artifact({
                'trajectory_points': traj.tolist(),
                'interpolation_dt': 0.008 / max(speed_scale or 0.5, 0.01),
                'solve_time': total_solve_time,
                'status': 'success',
                'level_check_info': {
                    'planning_status': 'single_candidate',
                    'candidate_count': 1,
                    'alignment_valid_count': 1,
                    'candidate_source_labels': list(candidate_source_labels),
                    'selected_index': 0,
                    'selected_source_label': single_source_label,
                    **single_goal_gate_summary,
                    'seed_candidate_added': bool(seed_candidate_added),
                    'seed_candidate_index': int(seed_candidate_index) if seed_candidate_index is not None else None,
                    'seed_trajopt_candidate_added': bool(seed_trajopt_candidate_added),
                    'seed_trajopt_candidate_index': (
                        int(seed_trajopt_candidate_index) if seed_trajopt_candidate_index is not None else None
                    ),
                    'seed_trajopt_smoothed_candidate_added': bool(seed_trajopt_smoothed_candidate_added),
                    'seed_trajopt_smoothed_candidate_index': (
                        int(seed_trajopt_smoothed_candidate_index)
                        if seed_trajopt_smoothed_candidate_index is not None else None
                    ),
                    'seed_trajopt_bridged_candidate_added': bool(seed_trajopt_bridged_candidate_added),
                    'seed_trajopt_bridged_candidate_index': (
                        int(seed_trajopt_bridged_candidate_index)
                        if seed_trajopt_bridged_candidate_index is not None else None
                    ),
                    'start_state_debug': planning_start_state_debug,
                    'seed_debug_info': seed_debug_info,
                },
            }, {
                **lifecycle_data,
                'candidates': lifecycle_candidates,
                'selection': {
                    'selected_candidate_id': 'candidate_0',
                    'selected_index': 0,
                    'selected_source_label': single_source_label,
                    'planning_status': 'single_candidate',
                    **single_goal_gate_summary,
                },
                'result': {
                    'status': 'success',
                    'selected_candidate_id': 'candidate_0',
                },
            })

        # 7. 多候选评估和筛选
        # [caohy] Phase 14: 确保所有候选在同一设备（CPU），展平到 2D [T, DOF]
        for ci, c in enumerate(all_candidates):
            if hasattr(c, 'detach'):
                c = c.detach().cpu()
            while c.ndim > 2:
                if c.shape[0] == 1:
                    c = c.squeeze(0)
                else:
                    c = c.reshape(-1, c.shape[-1])
            if c.ndim == 1:
                c = c.unsqueeze(0)
            all_candidates[ci] = c

        max_T = max(c.shape[0] for c in all_candidates)
        DOF = all_candidates[0].shape[1]
        padded = []
        for c in all_candidates:
            if c.shape[0] < max_T:
                pad = c[-1:].expand(max_T - c.shape[0], -1)
                c = torch.cat([c, pad], dim=0)
            padded.append(c)
        positions_batch = torch.stack(padded, dim=0)

        self.get_logger().info(f'positions_batch shape: {list(positions_batch.shape)}')

        # [caohy] Phase 14: 移到 GPU 供 kinematics_fn 使用
        positions_batch = positions_batch.to('cuda:0')

        # alignment 评估
        y_tool = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device='cuda:0')
        z_neg = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device='cuda:0')

        def kin_fn(pos):
            state = CuJointState.from_position(pos, joint_names=self._joint_names)
            # [caohy] Phase 15：将 V2 KinematicsState 适配为约束评估侧稳定消费的简化 FK 结果，
            # 统一暴露 ee_quaternion，避免评估代码继续依赖旧版 FK 结构或 tool_frames 内部细节。
            kin_state = self._planner.compute_kinematics(state)
            tool_pose = kin_state.tool_poses.get_link_pose(self._tool_frames[0])
            return SimpleNamespace(ee_quaternion=tool_pose.quaternion)

        try:
            level_eval = constraint_utils.evaluate_axis_alignment_batched(
                positions_batch, kin_fn, level_tolerance_deg, y_tool, z_neg,
            )
        except Exception as e:
            self.get_logger().error(f'Alignment eval failed: {e}')
            import traceback
            traceback.print_exc()
            raise

        # 连续性指标
        continuity = constraint_utils.compute_candidate_continuity_metrics(
            positions_batch, start_joint, target_pose[3:7], kin_fn,
        )

        candidate_goal_summaries = self._summarize_candidate_terminal_goal_pose_batch(
            positions_batch,
            target_pose,
        )
        goal_position_tolerance_m, goal_orientation_tolerance_deg = (
            self._get_selector_goal_gate_tolerances()
        )
        goal_gate_summary = self._build_selector_goal_gate_summary(
            candidate_goal_summaries,
            goal_position_tolerance_m,
            goal_orientation_tolerance_deg,
        )
        goal_pose_valid_mask = torch.tensor(
            goal_gate_summary['candidate_goal_pose_valid'],
            device=positions_batch.device,
            dtype=torch.bool,
        )
        level_eval_for_selection = dict(level_eval)
        level_eval_for_selection['alignment_valid'] = (
            level_eval['alignment_valid'] & goal_pose_valid_mask
        )

        # 筛选
        selection = constraint_utils.select_level_first_candidate(
            positions_batch,
            level_eval_for_selection,
            continuity,
            level_tolerance_deg,
            strict_level,
        )
        selection['candidate_source_labels'] = list(candidate_source_labels)
        selection.update(goal_gate_summary)
        # [caohy] Task 36：在 alignment_seed_trajopt 已经通过 3 度阈值门控后，
        # 允许它凭更小的关节单步跳变替代原始 alignment_seed。底层通用 selector
        # 仍优先保护起点不跳，这里只对“原始对齐种子 vs 优化后对齐种子”做窄范围覆盖。
        self._prefer_smoother_alignment_trajopt_candidate(
            selection,
            candidate_source_labels,
            level_tolerance_deg,
        )
        # [caohy] Task 35：只读诊断字段，用于定位 alignment_seed_trajopt
        # 的最大对齐偏差出现在第几个轨迹点；不参与候选排序和规划行为。
        try:
            angle_map_cpu = level_eval['alignment_angle_map'].detach().cpu()
            positions_cpu = positions_batch.detach().cpu()
            alignment_profiles = []
            for ci, label in enumerate(candidate_source_labels):
                angles = angle_map_cpu[ci]
                max_index = int(torch.argmax(angles).item())
                window_start = max(0, max_index - 2)
                window_end = min(int(angles.shape[0]), max_index + 3)
                alignment_profiles.append({
                    'candidate_index': int(ci),
                    'source_label': str(label),
                    'max_alignment_point_index': max_index,
                    'max_alignment_deviation': round(float(angles[max_index].item()), 4),
                    'alignment_profile_deg': [
                        round(float(v), 4) for v in angles.tolist()
                    ],
                    'joint_position_at_max': [
                        round(float(v), 6) for v in positions_cpu[ci, max_index, :].tolist()
                    ],
                    'alignment_window': [
                        {
                            'point_index': int(pi),
                            'alignment_deviation_deg': round(float(angles[pi].item()), 4),
                            'joint_position': [
                                round(float(v), 6)
                                for v in positions_cpu[ci, pi, :].tolist()
                            ],
                        }
                        for pi in range(window_start, window_end)
                    ],
                })
            selection['candidate_alignment_profiles'] = alignment_profiles
        except Exception as exc:
            selection['candidate_alignment_profiles_error'] = str(exc)
        # [caohy] Task 30：把 selected_index 直接映射成 selected_source_label，
        # 让 trajectory.json 和后续统计脚本都能直接读到最终选中来源。
        selected_index = selection.get('selected_index')
        if (
            selected_index is not None
            and 0 <= int(selected_index) < len(candidate_source_labels)
        ):
            selection['selected_source_label'] = str(candidate_source_labels[int(selected_index)])
        else:
            selection['selected_source_label'] = None
        self._mark_sequence_branch_attempt_selected(
            lifecycle_data,
            selection.get('selected_source_label'),
        )
        self._mark_alignment_trajopt_family_attempt_selected(
            lifecycle_data,
            selection.get('selected_source_label'),
        )
        self._mark_alignment_trajopt_family_topk_attempt_selected(
            lifecycle_data,
            selection.get('selected_source_label'),
        )
        selection['seed_candidate_added'] = bool(seed_candidate_added)
        selection['seed_candidate_index'] = int(seed_candidate_index) if seed_candidate_index is not None else None
        selection['seed_trajopt_candidate_added'] = bool(seed_trajopt_candidate_added)
        selection['seed_trajopt_candidate_index'] = (
            int(seed_trajopt_candidate_index) if seed_trajopt_candidate_index is not None else None
        )
        selection['seed_trajopt_smoothed_candidate_added'] = bool(seed_trajopt_smoothed_candidate_added)
        selection['seed_trajopt_smoothed_candidate_index'] = (
            int(seed_trajopt_smoothed_candidate_index)
            if seed_trajopt_smoothed_candidate_index is not None else None
        )
        selection['seed_trajopt_bridged_candidate_added'] = bool(seed_trajopt_bridged_candidate_added)
        selection['seed_trajopt_bridged_candidate_index'] = (
            int(seed_trajopt_bridged_candidate_index)
            if seed_trajopt_bridged_candidate_index is not None else None
        )
        selection['start_state_debug'] = planning_start_state_debug
        selection['seed_debug_info'] = seed_debug_info
        lifecycle_candidates = []
        level_eval_cpu = {
            'alignment_angle_map': level_eval['alignment_angle_map'].detach().cpu(),
            'max_alignment_deviation': level_eval['max_alignment_deviation'].detach().cpu(),
            'mean_alignment_deviation': level_eval['mean_alignment_deviation'].detach().cpu(),
            'alignment_valid': level_eval['alignment_valid'].detach().cpu(),
        }
        continuity_cpu = {
            key: value.detach().cpu() if hasattr(value, 'detach') else value
            for key, value in continuity.items()
        }
        selected_candidate_id = None
        for ci, candidate in enumerate(all_candidates):
            candidate_points = self._trajectory_tensor_to_list(candidate)
            candidate_id = f'candidate_{ci}'
            if selection.get('selected_index') == ci:
                selected_candidate_id = candidate_id
            source_label = str(candidate_source_labels[ci])
            base_candidate_record = {
                'candidate_id': candidate_id,
                'candidate_index': int(ci),
                'source_label': source_label,
                'entered_pool': True,
                'selected': bool(selection.get('selected_index') == ci),
                'alignment_valid': bool(level_eval_for_selection['alignment_valid'][ci].item()),
                'max_alignment_deviation_deg': round(
                    float(level_eval_cpu['max_alignment_deviation'][ci].item()), 6,
                ),
                'mean_alignment_deviation_deg': round(
                    float(level_eval_cpu['mean_alignment_deviation'][ci].item()), 6,
                ),
                'start_joint_gap_l2': round(
                    float(continuity_cpu['start_joint_gap_l2'][ci].item()), 6,
                ),
                'joint_step_jump_cost': round(
                    float(continuity_cpu['joint_step_jump_cost'][ci].item()), 6,
                ),
                'joint_step_max_l2': round(
                    float(continuity_cpu['joint_step_max_l2'][ci].item()), 6,
                ),
                'joint_step_max_abs': round(
                    float(continuity_cpu['joint_step_max_abs'][ci].item()), 6,
                ),
                'twist_smoothness_cost': round(
                    float(continuity_cpu['twist_smoothness_cost'][ci].item()), 6,
                ),
                'twist_profile_deg': self._round_nested_debug_value(
                    continuity_cpu['twist_profile_deg'][ci], float_digits=6,
                ),
                'goal_pose_valid': bool(goal_gate_summary['candidate_goal_pose_valid'][ci]),
                'position_error_m': goal_gate_summary['candidate_position_error_m'][ci],
                'orientation_error_deg': goal_gate_summary['candidate_orientation_error_deg'][ci],
                'terminal_goal_pose_summary': candidate_goal_summaries[ci],
                'trajectory_points': candidate_points,
                'trajectory_summary': self._summarize_trajectory_points(candidate_points),
            }
            lifecycle_candidates.append(build_lifecycle_candidate_record(
                base_record=base_candidate_record,
                source_label=source_label,
                trajectory_points=candidate_points,
                metrics_keys=(
                    'alignment_valid',
                    'max_alignment_deviation_deg',
                    'mean_alignment_deviation_deg',
                    'goal_pose_valid',
                    'position_error_m',
                    'orientation_error_deg',
                    'start_joint_gap_l2',
                    'joint_step_jump_cost',
                    'joint_step_max_l2',
                    'joint_step_max_abs',
                    'twist_smoothness_cost',
                ),
                metadata={
                    'phase1_adapter': 'multi_candidate_selection_path',
                    'behavior_changed': False,
                },
            ))
        self._sync_diffusion_seed_report_with_selection(
            lifecycle_data,
            lifecycle_candidates,
        )
        planner_candidate_metrics_by_pool_index = {
            int(item['candidate_index']): item
            for item in lifecycle_candidates
            if str(item.get('source_label')) == 'planner'
        }
        planner_legacy_candidate_metrics_by_pool_index = {
            int(item['candidate_index']): item
            for item in lifecycle_candidates
            if str(item.get('source_label')) == 'planner_legacy'
        }
        for attempt_record in lifecycle_data.get('planner_attempts', []):
            pool_index = attempt_record.get('pool_candidate_index')
            if pool_index is None:
                attempt_record['candidate_pool_accepted'] = False
                attempt_record['final_selected'] = False
                continue
            candidate_metrics = planner_candidate_metrics_by_pool_index.get(int(pool_index))
            if candidate_metrics is None:
                continue
            attempt_record['candidate_pool_accepted'] = True
            attempt_record['final_selected'] = bool(candidate_metrics.get('selected'))
            attempt_record['selection_metrics'] = {
                'alignment_valid': bool(candidate_metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': candidate_metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': candidate_metrics.get('mean_alignment_deviation_deg'),
                'goal_pose_valid': bool(candidate_metrics.get('goal_pose_valid')),
                'position_error_m': candidate_metrics.get('position_error_m'),
                'orientation_error_deg': candidate_metrics.get('orientation_error_deg'),
                'start_joint_gap_l2': candidate_metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': candidate_metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': candidate_metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': candidate_metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': candidate_metrics.get('twist_smoothness_cost'),
            }
            attempt_record['selected_candidate_id'] = candidate_metrics.get('candidate_id')
        for attempt_record in lifecycle_data.get('planner_legacy_attempts', []):
            pool_index = attempt_record.get('pool_candidate_index')
            if pool_index is None:
                attempt_record['candidate_pool_accepted'] = False
                attempt_record['final_selected'] = False
                continue
            candidate_metrics = planner_legacy_candidate_metrics_by_pool_index.get(int(pool_index))
            if candidate_metrics is None:
                continue
            attempt_record['candidate_pool_accepted'] = True
            attempt_record['final_selected'] = bool(candidate_metrics.get('selected'))
            attempt_record['selection_metrics'] = {
                'alignment_valid': bool(candidate_metrics.get('alignment_valid')),
                'max_alignment_deviation_deg': candidate_metrics.get('max_alignment_deviation_deg'),
                'mean_alignment_deviation_deg': candidate_metrics.get('mean_alignment_deviation_deg'),
                'goal_pose_valid': bool(candidate_metrics.get('goal_pose_valid')),
                'position_error_m': candidate_metrics.get('position_error_m'),
                'orientation_error_deg': candidate_metrics.get('orientation_error_deg'),
                'start_joint_gap_l2': candidate_metrics.get('start_joint_gap_l2'),
                'joint_step_jump_cost': candidate_metrics.get('joint_step_jump_cost'),
                'joint_step_max_l2': candidate_metrics.get('joint_step_max_l2'),
                'joint_step_max_abs': candidate_metrics.get('joint_step_max_abs'),
                'twist_smoothness_cost': candidate_metrics.get('twist_smoothness_cost'),
            }
            attempt_record['selected_candidate_id'] = candidate_metrics.get('candidate_id')
        self._sync_alignment_trajopt_family_attempt_pool_status(
            lifecycle_data,
            lifecycle_candidates,
        )
        self._sync_alignment_trajopt_family_topk_attempt_pool_status(
            lifecycle_data,
            lifecycle_candidates,
        )
        self._sync_alignment_trajopt_family_topk_shadow_attempt_status(
            lifecycle_data,
            all_candidates,
            candidate_source_labels,
            start_joint,
            target_pose,
            level_tolerance_deg,
            strict_level,
            actual_selected_source_label=selection.get('selected_source_label'),
        )

        self.get_logger().info(
            f'Selection: valid={selection["alignment_valid_count"]}/{selection["candidate_count"]}, '
            f'status={selection["planning_status"]}'
        )
        # [caohy] Task 31：marker_only_logs 模式下的紧凑摘要，
        # 用 error 级别确保即使日志级别设为 ERROR 也能在 RViz 面板看到。
        if self._task36_marker_only_logs:
            move_label = f'MOVE{plan_request_index:02d}' if plan_request_index is not None else 'MOVE??'
            selected_src = selection.get('selected_source_label') or '?'
            self.get_logger().error(
                f'{move_label} | selected={selected_src}'
            )

        if selection['planning_status'] == 'failed_alignment_constraint':
            return self._attach_lifecycle_artifact({
                'trajectory_points': [],
                'interpolation_dt': None,
                'solve_time': total_solve_time,
                'status': 'failed_alignment_constraint',
                'level_check_info': selection,
            }, {
                **lifecycle_data,
                'candidates': lifecycle_candidates,
                'selection': {
                    'selected_candidate_id': selected_candidate_id,
                    **selection,
                },
                'result': {
                    'status': 'failed_alignment_constraint',
                    'selected_candidate_id': selected_candidate_id,
                    'failure_reason': selection.get('failure_reason'),
                },
            })

        # 8. 返回最优候选
        best_idx = selection['selected_index']
        best_traj = all_candidates[best_idx]

        return self._attach_lifecycle_artifact({
            'trajectory_points': best_traj.tolist(),
            'interpolation_dt': 0.008 / max(speed_scale or 0.5, 0.01),
            'solve_time': total_solve_time,
            'status': selection['planning_status'],
            'level_check_info': selection,
        }, {
            **lifecycle_data,
            'candidates': lifecycle_candidates,
            'selection': {
                'selected_candidate_id': selected_candidate_id,
                **selection,
            },
            'result': {
                'status': selection['planning_status'],
                'selected_candidate_id': selected_candidate_id,
                'failure_reason': selection.get('failure_reason'),
            },
        })

    # [caohy] Phase 6.5/6.6：兼容 cuRobo V2 TrajOptSolverResult 的 tensor/bool success 语义，避免直接按旧接口取值。
    def _result_success(self, result) -> bool:
        if result is None:
            return False
        success = getattr(result, 'success', False)
        if hasattr(success, 'any'):
            return bool(success.any())
        return bool(success)

    def _result_status(self, result) -> str:
        if result is None:
            return 'planner_returned_none'
        return str(getattr(result, 'status', 'unknown'))

    def _extract_result_success_mask(self, result, output_count: int) -> list[bool]:
        """[caohy] 将 CuRobo result.success 规整为逐候选 success mask（成功掩码）。"""
        count = max(0, int(output_count or 0))
        if count <= 0:
            return []

        raw_success = self._tensor_to_debug_value(
            getattr(result, 'success', None) if result is not None else None
        )
        flattened: list[bool] = []

        def _flatten(value):
            if isinstance(value, (list, tuple)):
                for item in value:
                    _flatten(item)
                return
            if value is None:
                return
            if isinstance(value, str):
                flattened.append(value.strip().lower() in ('1', 'true', 'yes', 'on'))
                return
            flattened.append(bool(value))

        _flatten(raw_success)
        if not flattened:
            overall_success = bool(self._result_success(result))
            return [overall_success for _ in range(count)]
        if len(flattened) == 1:
            return [bool(flattened[0]) for _ in range(count)]
        if len(flattened) < count:
            flattened.extend([False] * (count - len(flattened)))
        return [bool(value) for value in flattened[:count]]

    def _log_plan_result_summary(self, label: str, result) -> None:
        """S5-B：打印 CuRobo 原始规划结果摘要，定位重复点最早出现在哪一层。"""
        if result is None:
            self.get_logger().warn(f'[S5-B] {label}: result is None')
            return

        solution = getattr(result, 'solution', None)
        if solution is None:
            self.get_logger().warn(f'[S5-B] {label}: result.solution is None')
            return

        try:
            if hasattr(solution, 'detach'):
                solution = solution.detach().cpu()
            solution_shape = list(solution.shape)
            solution_flat = solution.reshape(-1, solution.shape[-1])
            solution_first = [round(float(v), 6) for v in solution_flat[0].tolist()]
            solution_last = [round(float(v), 6) for v in solution_flat[-1].tolist()]
            self._log_probe_info(
                f'[S5-B] {label}: solution_shape={solution_shape}, '
                f'solution_first={solution_first}, solution_last={solution_last}'
            )
        except Exception as exc:
            self.get_logger().warn(f'[S5-B] {label}: failed to summarize result.solution: {exc}')

    def _extract_solution_debug_summary(self, result) -> dict:
        """提取 result.solution 摘要，便于失败时继续判断是否已经生成了退化解。"""
        solution = getattr(result, 'solution', None)
        if solution is None:
            return {'solution_present': False}

        payload = {'solution_present': True}
        try:
            if hasattr(solution, 'detach'):
                solution = solution.detach().cpu()
            payload['solution_shape'] = list(solution.shape)
            solution_flat = solution.reshape(-1, solution.shape[-1])
            payload['solution_first'] = [round(float(v), 6) for v in solution_flat[0].tolist()]
            payload['solution_last'] = [round(float(v), 6) for v in solution_flat[-1].tolist()]
        except Exception as exc:
            payload['solution_summary_error'] = str(exc)
        return payload

    def _extract_retained_result_decision_summary(self, result) -> dict:
        """提取 topk 结果对象中仍然保留下来的成功判定相关字段。"""
        payload = {}
        for attr in (
            'feasible',
            'position_error',
            'rotation_error',
            'cspace_error',
            'seed_cost',
            'seed_rank',
            'total_cost_reshaped',
            'goalset_index',
        ):
            payload[attr] = self._tensor_to_debug_value(getattr(result, attr, None))
        debug_info = getattr(result, 'debug_info', None)
        if isinstance(debug_info, dict):
            payload['debug_info_keys'] = sorted(debug_info.keys())
        else:
            payload['debug_info_keys'] = None
        return payload

    @staticmethod
    def _tensor_to_debug_value(value):
        """将张量或标量安全转成便于日志打印的 Python 值。"""
        if value is None:
            return None
        if hasattr(value, 'detach'):
            try:
                return value.detach().cpu().tolist()
            except Exception:
                return str(value)
        return value

    def _normalize_joint_state_for_metrics(self, joint_state):
        """[caohy] Task 29：把失败结果里的 JointState 规整成 rollout metrics 可消费的 [B, H, DOF]。"""
        if joint_state is None or getattr(joint_state, 'position', None) is None:
            return None
        normalized = joint_state
        try:
            while normalized.position.ndim > 3 and normalized.position.shape[0] == 1:
                normalized = normalized.squeeze(0)
            if normalized.position.ndim == 2:
                normalized = normalized.unsqueeze(0)
            if normalized.position.ndim != 3:
                return None
        except Exception:
            return None
        return normalized

    def _summarize_constraint_collection(self, cost_collection) -> list:
        """[caohy] Task 29：按约束项汇总最大违规量，直接回答是哪一类 feasibility 没过。

        [caohy] Task 6：补充 first_positive_step（首次违规步）和 worst_step_index（最坏步），
        让 self_collision（自碰撞）不只停留在“有/没有”，还能落到“哪一段先坏”。
        """
        if cost_collection is None:
            return []
        names = list(getattr(cost_collection, 'names', []) or [])
        values = list(getattr(cost_collection, 'values', []) or [])
        summaries = []
        for name, value in zip(names, values):
            try:
                tensor = value.detach().cpu() if hasattr(value, 'detach') else value
                max_value = float(torch.max(tensor).item())
                min_value = float(torch.min(tensor).item())
                positive_count = int(torch.count_nonzero(tensor > 0.0).item())
                first_positive_step = None
                worst_step_index = None
                if tensor.ndim >= 2:
                    step_max = torch.amax(tensor, dim=0)
                    positive_steps = torch.nonzero(step_max > 0.0, as_tuple=False).reshape(-1)
                    if int(positive_steps.numel()) > 0:
                        first_positive_step = int(positive_steps[0].item())
                    worst_step_index = int(torch.argmax(step_max).item())
                    last_step = tensor[:, -1]
                    last_step_max = float(torch.max(last_step).item())
                else:
                    positive_steps = torch.nonzero(tensor > 0.0, as_tuple=False).reshape(-1)
                    if int(positive_steps.numel()) > 0:
                        first_positive_step = int(positive_steps[0].item())
                    worst_step_index = int(torch.argmax(tensor).item())
                    last_step_max = max_value
                summaries.append(
                    {
                        'name': name,
                        'max_value': round(max_value, 6),
                        'min_value': round(min_value, 6),
                        'last_step_max': round(last_step_max, 6),
                        'positive_count': positive_count,
                        'first_positive_step': first_positive_step,
                        'worst_step_index': worst_step_index,
                    }
                )
            except Exception as exc:
                summaries.append({'name': name, 'error': str(exc)})
        summaries.sort(
            key=lambda item: (
                float(item.get('max_value', float('-inf')))
                if isinstance(item.get('max_value'), (int, float))
                else float('-inf')
            ),
            reverse=True,
        )
        return summaries

    def _to_cpu_tensor(self, value):
        if value is None:
            return None
        try:
            tensor = value.detach().cpu() if hasattr(value, 'detach') else torch.as_tensor(value)
        except Exception:
            return None
        return tensor

    def _reshape_state_tensor(self, tensor) -> Optional[torch.Tensor]:
        tensor = self._to_cpu_tensor(tensor)
        if tensor is None:
            return None
        try:
            if tensor.ndim == 1:
                return tensor.view(1, 1, -1)
            if tensor.ndim == 2:
                return tensor.unsqueeze(0)
            if tensor.ndim == 3:
                return tensor
        except Exception:
            return None
        return None

    def _extract_joint_state_dt_tensor(
        self,
        joint_state,
        expected_batch: int,
        dtype: torch.dtype,
    ) -> Optional[torch.Tensor]:
        """[caohy] Task 35：把 JointState 里的 dt 规整成 [B,1,1]，给差分诊断复用。"""
        dt_tensor = self._to_cpu_tensor(getattr(joint_state, 'dt', None))
        if dt_tensor is None:
            return None
        try:
            dt_tensor = dt_tensor.to(dtype=dtype).reshape(-1)
            if dt_tensor.numel() == 0:
                return None
            if dt_tensor.numel() == 1 and expected_batch > 1:
                dt_tensor = dt_tensor.repeat(expected_batch)
            elif dt_tensor.numel() != expected_batch:
                return None
            dt_tensor = torch.clamp(dt_tensor, min=1e-6)
            return dt_tensor.view(expected_batch, 1, 1)
        except Exception:
            return None

    def _differentiate_trajectory_tensor(
        self,
        state_tensor: torch.Tensor,
        dt_tensor: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """[caohy] Task 35：按轨迹位置和 dt 做一阶差分，补只读动态项诊断。"""
        if state_tensor is None or dt_tensor is None:
            return None
        try:
            batch, horizon, dof = state_tensor.shape
            if horizon <= 0:
                return None
            diff_tensor = torch.zeros((batch, horizon, dof), dtype=state_tensor.dtype)
            if horizon == 1:
                return diff_tensor
            step_diff = (state_tensor[:, 1:, :] - state_tensor[:, :-1, :]) / dt_tensor
            diff_tensor[:, 1:, :] = step_diff
            diff_tensor[:, 0, :] = step_diff[:, 0, :]
            return diff_tensor
        except Exception:
            return None

    def _resolve_joint_dynamic_tensor(
        self,
        normalized_joint_state,
        augmented_state,
        state_name: str,
    ) -> tuple[Optional[torch.Tensor], str]:
        """[caohy] Task 35：优先读底层动态状态；拿不到时再用位置轨迹差分回推。"""
        direct_tensor = self._reshape_state_tensor(getattr(augmented_state, state_name, None))
        if direct_tensor is not None:
            return direct_tensor, f'augmented_state.{state_name}'

        joint_state_tensor = self._reshape_state_tensor(getattr(normalized_joint_state, state_name, None))
        if joint_state_tensor is not None:
            return joint_state_tensor, f'joint_state.{state_name}'

        position_tensor = self._reshape_state_tensor(getattr(normalized_joint_state, 'position', None))
        if position_tensor is None:
            return None, 'unavailable'
        dt_tensor = self._extract_joint_state_dt_tensor(
            normalized_joint_state,
            int(position_tensor.shape[0]),
            position_tensor.dtype,
        )
        if dt_tensor is None:
            return None, 'missing_dt'

        derived_velocity = self._differentiate_trajectory_tensor(position_tensor, dt_tensor)
        if derived_velocity is None:
            return None, 'differentiate_failed_velocity'
        if state_name == 'velocity':
            return derived_velocity, 'finite_difference.velocity'

        derived_acceleration = self._differentiate_trajectory_tensor(derived_velocity, dt_tensor)
        if derived_acceleration is None:
            return None, 'differentiate_failed_acceleration'
        if state_name == 'acceleration':
            return derived_acceleration, 'finite_difference.acceleration'

        derived_jerk = self._differentiate_trajectory_tensor(derived_acceleration, dt_tensor)
        if derived_jerk is None:
            return None, 'differentiate_failed_jerk'
        if state_name == 'jerk':
            return derived_jerk, 'finite_difference.jerk'
        return None, 'unsupported_state_name'

    def _summarize_position_limit_violations(self, position_tensor) -> dict:
        payload = {'available': False}
        position_tensor = self._reshape_state_tensor(position_tensor)
        if (
            position_tensor is None
            or self._joint_position_limit_lower is None
            or self._joint_position_limit_upper is None
        ):
            return payload
        try:
            lower = torch.tensor(self._joint_position_limit_lower, dtype=position_tensor.dtype)
            upper = torch.tensor(self._joint_position_limit_upper, dtype=position_tensor.dtype)
            below = torch.clamp(lower.view(1, 1, -1) - position_tensor, min=0.0)
            above = torch.clamp(position_tensor - upper.view(1, 1, -1), min=0.0)
            violation = torch.maximum(below, above)
            joint_violation = torch.amax(violation, dim=(0, 1))
            positive_joints = torch.nonzero(joint_violation > 0.0, as_tuple=False).reshape(-1)
            horizon_any = torch.any(violation > 0.0, dim=2)
            first_step = None
            if torch.any(horizon_any):
                first_step = int(torch.nonzero(horizon_any, as_tuple=False)[0][1].item())
            worst_joint_index = int(torch.argmax(joint_violation).item())
            payload.update(
                {
                    'available': True,
                    'max_violation': round(float(torch.max(violation).item()), 6),
                    'last_step_max_violation': round(
                        float(torch.max(violation[:, -1, :]).item()), 6
                    ),
                    'first_violation_step': first_step,
                    'violating_joint_indices': [int(idx.item()) for idx in positive_joints],
                    'violating_joint_names': [
                        self._joint_names[int(idx.item())]
                        for idx in positive_joints
                        if int(idx.item()) < len(self._joint_names)
                    ],
                    'worst_joint_index': worst_joint_index,
                    'worst_joint_name': (
                        self._joint_names[worst_joint_index]
                        if worst_joint_index < len(self._joint_names)
                        else None
                    ),
                    'worst_joint_max_violation': round(
                        float(joint_violation[worst_joint_index].item()), 6
                    ),
                    'max_abs_position': round(float(torch.max(torch.abs(position_tensor)).item()), 6),
                }
            )
        except Exception as exc:
            payload['error'] = str(exc)
        return payload

    def _summarize_symmetric_limit_violations(self, state_tensor, limit_name: str) -> dict:
        payload = {'available': False}
        state_tensor = self._reshape_state_tensor(state_tensor)
        limit_range = self._joint_dynamic_limit_ranges.get(limit_name)
        if state_tensor is None or not isinstance(limit_range, dict):
            return payload
        try:
            lower = torch.tensor(limit_range.get('lower') or [], dtype=state_tensor.dtype)
            upper = torch.tensor(limit_range.get('upper') or [], dtype=state_tensor.dtype)
            limit_abs = torch.maximum(torch.abs(lower), torch.abs(upper))
            if limit_abs.numel() != state_tensor.shape[-1]:
                payload['error'] = f'limit_size_mismatch:{limit_abs.numel()}!={state_tensor.shape[-1]}'
                return payload
            abs_state = torch.abs(state_tensor)
            violation = torch.clamp(abs_state - limit_abs.view(1, 1, -1), min=0.0)
            joint_violation = torch.amax(violation, dim=(0, 1))
            positive_joints = torch.nonzero(joint_violation > 0.0, as_tuple=False).reshape(-1)
            horizon_any = torch.any(violation > 0.0, dim=2)
            first_step = None
            if torch.any(horizon_any):
                first_step = int(torch.nonzero(horizon_any, as_tuple=False)[0][1].item())
            worst_joint_index = int(torch.argmax(joint_violation).item())
            payload.update(
                {
                    'available': True,
                    'max_violation': round(float(torch.max(violation).item()), 6),
                    'last_step_max_violation': round(
                        float(torch.max(violation[:, -1, :]).item()), 6
                    ),
                    'first_violation_step': first_step,
                    'violating_joint_indices': [int(idx.item()) for idx in positive_joints],
                    'violating_joint_names': [
                        self._joint_names[int(idx.item())]
                        for idx in positive_joints
                        if int(idx.item()) < len(self._joint_names)
                    ],
                    'worst_joint_index': worst_joint_index,
                    'worst_joint_name': (
                        self._joint_names[worst_joint_index]
                        if worst_joint_index < len(self._joint_names)
                        else None
                    ),
                    'worst_joint_max_violation': round(
                        float(joint_violation[worst_joint_index].item()), 6
                    ),
                    'max_abs_value': round(float(torch.max(abs_state).item()), 6),
                    'limit_max_abs': round(float(torch.max(limit_abs).item()), 6),
                }
            )
        except Exception as exc:
            payload['error'] = str(exc)
        return payload

    def _extract_rollout_goal_joint_target(
        self,
        rollout,
        expected_batch: int,
        dtype: torch.dtype,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], str]:
        """[caohy] Task 35：从 rollout 的 metrics goal 里取当前 cspace 真正看到的目标关节。"""
        goal = getattr(rollout, '_metrics_goal', None)
        if goal is None:
            return None, None, 'metrics_goal_missing'
        goal_js = getattr(goal, 'goal_js', None)
        goal_position = self._to_cpu_tensor(getattr(goal_js, 'position', None))
        if goal_position is None:
            return None, None, 'goal_js_missing'
        try:
            goal_position = goal_position.to(dtype=dtype)
            while goal_position.ndim > 2 and goal_position.shape[0] == 1:
                goal_position = goal_position.squeeze(0)
            if goal_position.ndim == 1:
                goal_position = goal_position.view(1, -1)
            elif goal_position.ndim != 2:
                return None, None, f'goal_js_shape_unsupported:{list(goal_position.shape)}'
        except Exception as exc:
            return None, None, f'goal_js_normalize_error:{exc}'

        idxs = self._to_cpu_tensor(getattr(goal, 'idxs_goal_js', None))
        if idxs is None:
            if goal_position.shape[0] == 1:
                idxs = torch.zeros((expected_batch,), dtype=torch.long)
            elif goal_position.shape[0] == expected_batch:
                idxs = torch.arange(expected_batch, dtype=torch.long)
            else:
                return None, None, 'idxs_goal_js_missing'
        else:
            try:
                idxs = idxs.to(dtype=torch.long).reshape(-1)
                if idxs.numel() == 1 and expected_batch > 1:
                    idxs = idxs.repeat(expected_batch)
                elif idxs.numel() != expected_batch:
                    return None, None, f'idxs_goal_js_mismatch:{idxs.numel()}!={expected_batch}'
            except Exception as exc:
                return None, None, f'idxs_goal_js_error:{exc}'
        return goal_position, idxs, 'metrics_goal.goal_js'

    def _summarize_joint_target_gap(
        self,
        position_tensor,
        target_joint_tensor: Optional[torch.Tensor],
        target_indices: Optional[torch.Tensor],
    ) -> dict:
        """[caohy] Task 35：显式回答 cspace 目标项离 goal_state（目标关节）还有多远。"""
        payload = {'available': False}
        position_tensor = self._reshape_state_tensor(position_tensor)
        if position_tensor is None or target_joint_tensor is None or target_indices is None:
            return payload
        try:
            target_joint_tensor = self._to_cpu_tensor(target_joint_tensor)
            target_indices = self._to_cpu_tensor(target_indices)
            if target_joint_tensor is None or target_indices is None:
                return payload
            target_joint_tensor = target_joint_tensor.to(dtype=position_tensor.dtype)
            target_indices = target_indices.to(dtype=torch.long).reshape(-1)
            if target_indices.numel() != position_tensor.shape[0]:
                payload['error'] = f'target_index_count_mismatch:{target_indices.numel()}!={position_tensor.shape[0]}'
                return payload
            indexed_target = target_joint_tensor.index_select(0, target_indices)
            if indexed_target.ndim != 2 or indexed_target.shape[-1] != position_tensor.shape[-1]:
                payload['error'] = (
                    f'target_shape_mismatch:{list(indexed_target.shape)}!={[position_tensor.shape[0], position_tensor.shape[-1]]}'
                )
                return payload
            target_horizon = indexed_target.unsqueeze(1).expand(-1, position_tensor.shape[1], -1)
            gap = position_tensor - target_horizon
            gap_l2 = torch.linalg.norm(gap, dim=-1)
            terminal_gap_l2 = gap_l2[:, -1]
            payload.update(
                {
                    'available': True,
                    'terminal_gap_l2_max': round(float(torch.max(terminal_gap_l2).item()), 6),
                    'terminal_gap_l2_mean': round(float(torch.mean(terminal_gap_l2).item()), 6),
                    'path_gap_l2_max': round(float(torch.max(gap_l2).item()), 6),
                    'path_gap_l2_mean': round(float(torch.mean(gap_l2).item()), 6),
                    'terminal_max_abs_gap': round(float(torch.max(torch.abs(gap[:, -1, :])).item()), 6),
                    'path_max_abs_gap': round(float(torch.max(torch.abs(gap)).item()), 6),
                    'goal_joint_at_last': [
                        round(float(v), 6) for v in indexed_target[0].tolist()
                    ] if indexed_target.shape[0] > 0 else None,
                }
            )
        except Exception as exc:
            payload['error'] = str(exc)
        return payload

    def _extract_cspace_runtime_info(self, rollout) -> dict:
        """[caohy] Task 35：记录 rollout 运行时 cspace 成本配置，避免只看外层猜。"""
        payload = {'available': False}
        try:
            manager_candidates = [
                ('rollout.metrics_constraint_manager', getattr(rollout, 'metrics_constraint_manager', None)),
                ('rollout.constraint_manager', getattr(rollout, 'constraint_manager', None)),
                ('rollout.metrics_cost_manager', getattr(rollout, 'metrics_cost_manager', None)),
                ('rollout.cost_manager', getattr(rollout, 'cost_manager', None)),
            ]
            cspace_cost = None
            manager_source = None
            for manager_name, manager in manager_candidates:
                if manager is None or not hasattr(manager, 'get_cost'):
                    continue
                cspace_cost = manager.get_cost('cspace')
                if cspace_cost is not None:
                    manager_source = manager_name
                    break
            if cspace_cost is None:
                payload['state_source'] = 'cspace_cost_missing'
                return payload
            config = getattr(cspace_cost, 'config', None)
            if config is None:
                payload['state_source'] = 'cspace_config_missing'
                return payload
            target_weight = self._to_cpu_tensor(getattr(config, 'cspace_target_weight', None))
            non_terminal_factor = self._to_cpu_tensor(
                getattr(config, 'cspace_non_terminal_weight_factor', None)
            )
            dof_weight = self._to_cpu_tensor(getattr(config, 'cspace_target_dof_weight', None))
            activation_distance = self._to_cpu_tensor(getattr(config, 'activation_distance', None))
            weight = self._to_cpu_tensor(getattr(config, 'weight', None))
            payload.update(
                {
                    'available': True,
                    'state_source': f'{manager_source}.cspace.config',
                    'compute_inverse_dynamics': bool(
                        getattr(getattr(rollout, 'transition_model', None), 'compute_inverse_dynamics', False)
                    ),
                    'cost_type': str(getattr(getattr(config, 'cost_type', None), 'name', None)),
                    'cspace_target_weight': self._tensor_to_debug_value(target_weight),
                    'cspace_non_terminal_weight_factor': self._tensor_to_debug_value(non_terminal_factor),
                    'cspace_target_dof_weight_max': (
                        round(float(torch.max(dof_weight).item()), 6)
                        if dof_weight is not None
                        else None
                    ),
                    'cspace_target_dof_weight_min': (
                        round(float(torch.min(dof_weight).item()), 6)
                        if dof_weight is not None
                        else None
                    ),
                    'activation_distance': self._tensor_to_debug_value(activation_distance),
                    'bound_weight': self._tensor_to_debug_value(weight),
                }
            )
        except Exception as exc:
            payload['error'] = str(exc)
        return payload

    def _build_joint_state_subconstraint_breakdown(self, rollout, normalized_joint_state, augmented_state) -> dict:
        """[caohy] Task 35：把 cspace 聚合项继续拆成位置/速度/加速度/jerk/力矩子项。"""
        velocity_tensor, velocity_source = self._resolve_joint_dynamic_tensor(
            normalized_joint_state,
            augmented_state,
            'velocity',
        )
        acceleration_tensor, acceleration_source = self._resolve_joint_dynamic_tensor(
            normalized_joint_state,
            augmented_state,
            'acceleration',
        )
        jerk_tensor, jerk_source = self._resolve_joint_dynamic_tensor(
            normalized_joint_state,
            augmented_state,
            'jerk',
        )
        target_joint_tensor, target_indices, target_source = self._extract_rollout_goal_joint_target(
            rollout,
            int(self._reshape_state_tensor(getattr(normalized_joint_state, 'position', None)).shape[0]),
            self._reshape_state_tensor(getattr(normalized_joint_state, 'position', None)).dtype,
        )
        breakdown = {
            'joint_position': self._summarize_position_limit_violations(
                getattr(normalized_joint_state, 'position', None)
            ),
            'joint_velocity': self._summarize_symmetric_limit_violations(
                velocity_tensor,
                'velocity',
            ),
            'joint_acceleration': self._summarize_symmetric_limit_violations(
                acceleration_tensor,
                'acceleration',
            ),
            'joint_jerk': self._summarize_symmetric_limit_violations(
                jerk_tensor,
                'jerk',
            ),
            'joint_target': self._summarize_joint_target_gap(
                getattr(normalized_joint_state, 'position', None),
                target_joint_tensor,
                target_indices,
            ),
        }
        breakdown['joint_velocity']['state_source'] = velocity_source
        breakdown['joint_acceleration']['state_source'] = acceleration_source
        breakdown['joint_jerk']['state_source'] = jerk_source
        breakdown['joint_target']['state_source'] = target_source

        torque_tensor = self._reshape_state_tensor(getattr(augmented_state, 'joint_torque', None))
        torque_limit_name = 'torque'
        torque_source = 'augmented_state.joint_torque' if torque_tensor is not None else 'unavailable'
        if torque_tensor is None:
            torque_tensor = self._reshape_state_tensor(getattr(normalized_joint_state, 'torque', None))
            if torque_tensor is not None:
                torque_source = 'joint_state.torque'
        if torque_tensor is None and 'effort' in self._joint_dynamic_limit_ranges:
            torque_tensor = self._reshape_state_tensor(getattr(augmented_state, 'effort', None))
            torque_limit_name = 'effort'
            if torque_tensor is not None:
                torque_source = 'augmented_state.effort'
        if torque_tensor is None and 'effort' in self._joint_dynamic_limit_ranges:
            torque_tensor = self._reshape_state_tensor(getattr(normalized_joint_state, 'effort', None))
            torque_limit_name = 'effort'
            if torque_tensor is not None:
                torque_source = 'joint_state.effort'
        breakdown['joint_torque'] = self._summarize_symmetric_limit_violations(
            torque_tensor,
            torque_limit_name,
        )
        breakdown['joint_torque']['state_source'] = torque_source
        breakdown['cspace_runtime_info'] = self._extract_cspace_runtime_info(rollout)
        return breakdown

    def _recompute_constraint_breakdown_from_joint_state(self, rollout, joint_state, prefix: str) -> dict:
        """[caohy] Task 29：对失败后的 joint trajectory 重新跑 metrics，保留 topk 前看不到的约束细项。"""
        payload = {
            f'{prefix}_metrics_recomputed': False,
            f'{prefix}_feasible_tensor_recomputed': None,
            f'{prefix}_constraint_summary_recomputed': [],
        }
        normalized_joint_state = self._normalize_joint_state_for_metrics(joint_state)
        if normalized_joint_state is None:
            payload[f'{prefix}_metrics_recomputed_error'] = 'joint_state_not_normalizable'
            return payload
        try:
            expected_batch = int(getattr(rollout, 'batch_size', normalized_joint_state.position.shape[0]))
            actual_batch = int(normalized_joint_state.position.shape[0])
            if actual_batch == 1 and expected_batch > 1:
                # [caohy] Task 29：topk 结果只剩 1 条轨迹，但 rollout 目标缓冲仍按原始多 seed batch
                # 保留；这里只做只读重复，目的是复用同一组目标缓冲拿到约束项拆解，不改变主求解行为。
                normalized_joint_state = normalized_joint_state.repeat([expected_batch, 1, 1])
            elif actual_batch != expected_batch:
                payload[f'{prefix}_metrics_recomputed_error'] = (
                    f'batch_mismatch_after_normalize:{actual_batch}!={expected_batch}'
                )
                return payload
            state_dt = getattr(normalized_joint_state, 'dt', None)
            if state_dt is not None:
                if state_dt.ndim == 0:
                    state_dt = state_dt.view(1)
                else:
                    state_dt = state_dt.reshape(-1)
                if state_dt.numel() == 1 and expected_batch > 1:
                    state_dt = state_dt.repeat(expected_batch)
                elif state_dt.numel() != expected_batch:
                    payload[f'{prefix}_metrics_recomputed_error'] = (
                        f'dt_mismatch_after_normalize:{state_dt.numel()}!={expected_batch}'
                    )
                    return payload
                normalized_joint_state.dt = state_dt
            augmented_state = rollout.transition_model.compute_augmented_state(normalized_joint_state)
            metrics = rollout.compute_metrics_from_state(augmented_state)
            payload[f'{prefix}_metrics_recomputed'] = True
            payload[f'{prefix}_subconstraint_breakdown_recomputed'] = (
                self._build_joint_state_subconstraint_breakdown(
                    rollout,
                    normalized_joint_state,
                    augmented_state,
                )
            )
            payload[f'{prefix}_feasible_tensor_recomputed'] = self._tensor_to_debug_value(
                metrics.costs_and_constraints.get_feasible(
                    include_all_hybrid=False,
                    sum_horizon=True,
                )
            )
            payload[f'{prefix}_constraint_summary_recomputed'] = self._summarize_constraint_collection(
                getattr(metrics.costs_and_constraints, 'constraints', None)
            )
        except Exception as exc:
            payload[f'{prefix}_metrics_recomputed_error'] = str(exc)
        return payload

    def _build_joint_state_from_trajectory(
        self,
        trajectory,
        dt_seconds: float = 1.0,
    ):
        """[caohy] Task 6：把 [T, DOF] 轨迹包装成 rollout metrics 可消费的 JointState。"""
        traj_tensor = self._to_cpu_tensor(trajectory)
        if traj_tensor is None:
            return None
        try:
            if traj_tensor.ndim == 1:
                traj_tensor = traj_tensor.view(1, -1)
            elif traj_tensor.ndim != 2:
                return None
            state = CuJointState.from_position(
                traj_tensor.to(device='cuda:0', dtype=torch.float32),
                joint_names=self._joint_names,
            )
            state.dt = torch.tensor(
                [max(float(dt_seconds), 1e-6)],
                device='cuda:0',
                dtype=torch.float32,
            )
            return state
        except Exception:
            return None

    def _analyze_trajectory_against_rollout(
        self,
        rollout,
        trajectory,
        prefix: str,
        dt_seconds: float = 1.0,
    ) -> dict:
        """[caohy] Task 6：对任意 seed / prepared seed 做只读约束重算，区分种子自身已坏还是求解过程推坏。"""
        payload = {
            'trajectory_summary': self._summarize_trajectory_points(
                self._trajectory_tensor_to_list(trajectory),
            ),
            'trajectory_step_metrics': self._summarize_seed_step_metrics(
                self._to_cpu_tensor(trajectory),
            ) if self._to_cpu_tensor(trajectory) is not None else None,
            'trajectory_limit_summary': self._summarize_trajectory_joint_limit_violation(
                self._to_cpu_tensor(trajectory),
            ) if self._to_cpu_tensor(trajectory) is not None else None,
            'state_dt_seconds': round(float(dt_seconds), 6),
        }
        if rollout is None:
            payload[f'{prefix}_metrics_recomputed_error'] = 'rollout_missing'
            return self._round_nested_debug_value(payload, float_digits=6)
        joint_state = self._build_joint_state_from_trajectory(trajectory, dt_seconds=dt_seconds)
        if joint_state is None:
            payload[f'{prefix}_metrics_recomputed_error'] = 'joint_state_build_failed'
            return self._round_nested_debug_value(payload, float_digits=6)
        payload.update(
            self._recompute_constraint_breakdown_from_joint_state(
                rollout,
                joint_state,
                prefix,
            )
        )
        return self._round_nested_debug_value(payload, float_digits=6)

    def _summarize_trajectory_pair_deviation(
        self,
        reference_trajectory,
        observed_trajectory,
        gap_thresholds: Optional[list[float]] = None,
    ) -> dict:
        """[caohy] Task 6：比较两条轨迹何时开始明显分叉，定位 solver（求解器）偏离 seed（种子）的首个关键步。"""
        summary = {
            'available': False,
            'reference_point_count': 0,
            'observed_point_count': 0,
        }
        ref_tensor = self._to_cpu_tensor(reference_trajectory)
        obs_tensor = self._to_cpu_tensor(observed_trajectory)
        if ref_tensor is None or obs_tensor is None:
            return summary
        try:
            if ref_tensor.ndim == 1:
                ref_tensor = ref_tensor.view(1, -1)
            if obs_tensor.ndim == 1:
                obs_tensor = obs_tensor.view(1, -1)
            if ref_tensor.ndim != 2 or obs_tensor.ndim != 2:
                return summary
            if ref_tensor.shape[-1] != obs_tensor.shape[-1]:
                summary['error'] = f'dof_mismatch:{ref_tensor.shape[-1]}!={obs_tensor.shape[-1]}'
                return summary
            if obs_tensor.shape[0] != ref_tensor.shape[0]:
                ref_tensor = self._resample_seed_traj_linear(
                    ref_tensor.to(device='cuda:0', dtype=torch.float32),
                    int(obs_tensor.shape[0]),
                ).detach().cpu()
            gap = obs_tensor - ref_tensor
            gap_l2 = torch.linalg.norm(gap, dim=-1)
            gap_max_abs = torch.amax(torch.abs(gap), dim=-1)
            worst_index = int(torch.argmax(gap_l2).item()) if int(gap_l2.numel()) > 0 else 0
            threshold_map = {}
            for threshold in (gap_thresholds or [0.1, 0.25, 0.5, 1.0]):
                positive = torch.nonzero(gap_l2 > float(threshold), as_tuple=False).reshape(-1)
                threshold_map[str(threshold)] = (
                    int(positive[0].item()) if int(positive.numel()) > 0 else None
                )
            window_start = max(0, worst_index - 2)
            window_end = min(int(gap_l2.shape[0]), worst_index + 3)
            summary.update(
                {
                    'available': True,
                    'reference_point_count': int(ref_tensor.shape[0]),
                    'observed_point_count': int(obs_tensor.shape[0]),
                    'path_gap_l2_max': round(float(torch.max(gap_l2).item()), 6),
                    'path_gap_l2_mean': round(float(torch.mean(gap_l2).item()), 6),
                    'path_gap_max_abs': round(float(torch.max(gap_max_abs).item()), 6),
                    'terminal_gap_l2': round(float(gap_l2[-1].item()), 6),
                    'terminal_gap_max_abs': round(float(gap_max_abs[-1].item()), 6),
                    'worst_step_index': worst_index,
                    'first_large_gap_steps': threshold_map,
                    'worst_window': [
                        {
                            'step_index': int(idx),
                            'gap_l2': round(float(gap_l2[idx].item()), 6),
                            'gap_max_abs': round(float(gap_max_abs[idx].item()), 6),
                            'reference_joint': [
                                round(float(v), 6) for v in ref_tensor[idx].tolist()
                            ],
                            'observed_joint': [
                                round(float(v), 6) for v in obs_tensor[idx].tolist()
                            ],
                        }
                        for idx in range(window_start, window_end)
                    ],
                }
            )
        except Exception as exc:
            summary['error'] = str(exc)
        return self._round_nested_debug_value(summary, float_digits=6)

    def _recompute_failure_constraint_breakdown(self, result) -> dict:
        """[caohy] Task 29：失败时补一层“按约束项拆解”的只读诊断，区分是关节约束还是碰撞等问题。"""
        payload = {}
        trajopt_solver = getattr(self._planner, 'trajopt_solver', None)
        if trajopt_solver is None:
            payload['recomputed_constraint_breakdown_error'] = 'trajopt_solver_missing'
            return payload

        payload.update(
            self._recompute_constraint_breakdown_from_joint_state(
                trajopt_solver.metrics_rollout,
                getattr(result, 'js_solution', None),
                'js_solution',
            )
        )
        interpolated_rollout = getattr(trajopt_solver, 'additional_metrics_rollouts', {}).get(
            'interpolated_rollout'
        )
        if interpolated_rollout is None:
            payload['interpolated_metrics_recomputed_error'] = 'interpolated_rollout_missing'
            return payload
        try:
            interpolated_joint_state = result.get_interpolated_plan()
        except Exception as exc:
            payload['interpolated_plan_recomputed_error'] = str(exc)
            return payload
        payload.update(
            self._recompute_constraint_breakdown_from_joint_state(
                interpolated_rollout,
                interpolated_joint_state,
                'interpolated_plan',
            )
        )
        return payload

    def _summarize_trajopt_failure_metrics(self, result) -> dict:
        """提取 trajopt 失败时最关键的可行性与收敛性指标，避免继续只看到 unknown。"""
        metrics = getattr(result, 'metrics', None)
        payload = {
            'has_metrics': metrics is not None,
            'metrics_type': type(metrics).__name__,
        }
        interpolated_metrics = getattr(result, 'interpolated_metrics', None)
        payload['has_interpolated_metrics'] = interpolated_metrics is not None
        payload['interpolated_metrics_type'] = type(interpolated_metrics).__name__
        if metrics is None:
            return payload

        try:
            feasible = metrics.costs_and_constraints.get_feasible(
                include_all_hybrid=False,
                sum_horizon=True,
            )
            payload['trajopt_feasible_tensor'] = self._tensor_to_debug_value(feasible)
        except Exception as exc:
            payload['trajopt_feasible_error'] = str(exc)

        convergence_names = []
        convergence_last_step = {}
        converged_flags = []
        try:
            convergence = getattr(metrics, 'convergence', None)
            names = list(getattr(convergence, 'names', []) or [])
            values = list(getattr(convergence, 'values', []) or [])
            for metric_name, metric_value in zip(names, values):
                convergence_names.append(metric_name)
                last_step_value = metric_value[:, -1:]
                convergence_last_step[metric_name] = self._tensor_to_debug_value(last_step_value)
                if 'position_tolerance' in metric_name:
                    converged_flags.append(last_step_value < float(getattr(result, 'position_tolerance', 0.0)))
                elif 'orientation_tolerance' in metric_name:
                    converged_flags.append(last_step_value < float(getattr(result, 'orientation_tolerance', 0.0)))

            payload['trajopt_convergence_metric_names'] = convergence_names
            payload['trajopt_convergence_last_step'] = convergence_last_step
            if converged_flags:
                converged = torch.all(torch.cat(converged_flags, dim=-1), dim=-1).squeeze(-1)
                payload['trajopt_converged_tensor'] = self._tensor_to_debug_value(converged)
        except Exception as exc:
            payload['trajopt_convergence_error'] = str(exc)

        try:
            if interpolated_metrics is not None:
                interpolated_feasible = interpolated_metrics.costs_and_constraints.get_feasible(
                    include_all_hybrid=False,
                    sum_horizon=True,
                )
                payload['trajopt_interpolated_feasible_tensor'] = self._tensor_to_debug_value(
                    interpolated_feasible
                )
        except Exception as exc:
            payload['trajopt_interpolated_feasible_error'] = str(exc)

        return payload

    def _build_failed_result_payload(self, result, extra_info: Optional[dict] = None) -> dict:
        """[caohy] Task 6：统一组装 trajopt / cspace 失败摘要，便于主路径与 segment 级对齐对比。"""
        payload = {
            'status': self._result_status(result),
            'success_type': type(getattr(result, 'success', None)).__name__,
            'valid_query_type': type(getattr(result, 'valid_query', None)).__name__,
            'has_solution': getattr(result, 'solution', None) is not None,
            'position_tolerance': getattr(result, 'position_tolerance', None),
            'orientation_tolerance': getattr(result, 'orientation_tolerance', None),
        }
        if extra_info:
            payload.update(extra_info)

        success = getattr(result, 'success', None)
        payload['success_tensor'] = self._tensor_to_debug_value(success)

        valid_query = getattr(result, 'valid_query', None)
        payload['valid_query_tensor'] = self._tensor_to_debug_value(valid_query)
        payload.update(self._extract_solution_debug_summary(result))
        payload.update(self._extract_js_solution_debug_summary(result))
        payload.update(self._extract_interpolated_plan_debug_summary(result))
        payload.update(self._extract_retained_result_decision_summary(result))
        payload.update(self._summarize_trajopt_failure_metrics(result))
        payload.update(self._recompute_failure_constraint_breakdown(result))
        return payload

    def _log_failed_result_summary(self, label: str, result, extra_info: Optional[dict] = None) -> None:
        """补充失败结果的原始字段摘要，避免 trajopt 失败时只剩 unknown。"""
        if result is None:
            self.get_logger().warn(f'[S5-B] {label}: result is None')
            return

        payload = self._build_failed_result_payload(result, extra_info=extra_info)
        self.get_logger().warn(f'[S5-B] {label}: {payload}')


def main(args=None):
    rclpy.init(args=args)
    node = CuroboV2PlannerNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
