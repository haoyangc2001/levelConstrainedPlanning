#!/usr/bin/env python3
# [caohy] Phase 9：V2 约束规划工具模块。
# 从 V1 curobo_motion_gen.py 迁移纯数学/逻辑函数，适配 V2 API。
"""Constraint planning utilities for CuRobo V2.

Contains math functions, alignment evaluation, candidate selection,
and seed generation logic migrated from V1.
"""

import math
from typing import Any, Dict, List, Optional

import torch


# ============================================================================
# Quaternion math (CPU, list-based)
# ============================================================================

def normalize_quaternion(q: List[float]) -> List[float]:
    """归一化四元数 [w, x, y, z]。"""
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [v / n for v in q]


def quaternion_multiply(q1: List[float], q2: List[float]) -> List[float]:
    """四元数乘法 q1 * q2，[w, x, y, z]。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return [
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ]


def quaternion_conjugate(q: List[float]) -> List[float]:
    """四元数共轭 [w, -x, -y, -z]。"""
    return [q[0], -q[1], -q[2], -q[3]]


def rotate_vector_by_quaternion(q: List[float], v: List[float]) -> List[float]:
    """用四元数 q 旋转向量 v。q = [w, x, y, z]。"""
    qv = [0.0, v[0], v[1], v[2]]
    q_conj = quaternion_conjugate(q)
    tmp = quaternion_multiply(q, qv)
    result = quaternion_multiply(tmp, q_conj)
    return [result[1], result[2], result[3]]


def vector_angle_deg(v1: List[float], v2: List[float]) -> float:
    """计算两个向量之间的角度（度）。"""
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    cos_angle = max(-1.0, min(1.0, dot / (n1 * n2)))
    return math.degrees(math.acos(cos_angle))


def local_y_twist_quaternion(twist_deg: float) -> List[float]:
    """构造绕局部 y 轴旋转 twist_deg 度的四元数。"""
    rad = math.radians(twist_deg)
    return [math.cos(rad / 2.0), 0.0, math.sin(rad / 2.0), 0.0]


def interpolate_angle_shortest_deg(a: float, b: float, t: float) -> float:
    """最短路径角度插值（度）。"""
    diff = (b - a + 180.0) % 360.0 - 180.0
    return a + diff * t


def wrap_angle_deg(angle: float) -> float:
    """将角度 wrapping 到 [-180, 180]。"""
    return (angle + 180.0) % 360.0 - 180.0


# ============================================================================
# Quaternion math (GPU, tensor-based)
# ============================================================================

def quaternion_to_rotation_matrix_batched(q: torch.Tensor) -> torch.Tensor:
    """四元数 batched 旋转矩阵。q [N,4] [qw,qx,qy,qz] -> R [N,3,3]。"""
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1.0 - 2.0 * (qy * qy + qz * qz)
    r01 = 2.0 * (qx * qy - qw * qz)
    r02 = 2.0 * (qx * qz + qw * qy)
    r10 = 2.0 * (qx * qy + qw * qz)
    r11 = 1.0 - 2.0 * (qx * qx + qz * qz)
    r12 = 2.0 * (qy * qz - qw * qx)
    r20 = 2.0 * (qx * qz - qw * qy)
    r21 = 2.0 * (qy * qz + qw * qx)
    r22 = 1.0 - 2.0 * (qx * qx + qy * qy)
    return torch.stack([
        torch.stack([r00, r01, r02], dim=-1),
        torch.stack([r10, r11, r12], dim=-1),
        torch.stack([r20, r21, r22], dim=-1),
    ], dim=-2)


def quaternion_multiply_batched(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数 batched 乘法。q1, q2 [N,4] [qw,qx,qy,qz]。"""
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def wrap_angle_deg_tensor(angle_deg: torch.Tensor) -> torch.Tensor:
    """将角度 wrapping 到 [-180, 180]（tensor 版）。"""
    return torch.remainder(angle_deg + 180.0, 360.0) - 180.0


def extract_ee_quaternion_batched(kinematics_out: Any) -> torch.Tensor:
    """兼容提取 V1/V2 风格 FK 结果中的末端四元数。

    支持两种返回形式：
    1. 旧式对象直接暴露 `ee_quaternion`
    2. CuRobo V2 `KinematicsState`，通过 `tool_poses.get_link_pose(...)` 访问
    """
    ee_quat = getattr(kinematics_out, 'ee_quaternion', None)
    if ee_quat is not None:
        return ee_quat

    tool_poses = getattr(kinematics_out, 'tool_poses', None)
    if tool_poses is None:
        raise AttributeError('FK result has neither ee_quaternion nor tool_poses')

    tool_frames = getattr(kinematics_out, 'tool_frames', None)
    if not tool_frames:
        raise AttributeError('FK result.tool_poses exists but tool_frames is missing or empty')

    tool_pose = tool_poses.get_link_pose(tool_frames[0])
    quat = getattr(tool_pose, 'quaternion', None)
    if quat is None:
        raise AttributeError('tool pose does not contain quaternion')
    return quat


# ============================================================================
# Alignment check (Phase 9.4)
# ============================================================================

def compute_alignment_deviation_from_quaternion(quaternion: List[float]) -> float:
    """计算 tool0 y+ 与 world z- 的对齐偏差角（度）。

    Args:
        quaternion: 末端姿态四元数 [w, x, y, z]。

    Returns:
        偏差角度（度），完美对齐时为 0°。
    """
    quat = normalize_quaternion(list(quaternion))
    axis_world = rotate_vector_by_quaternion(quat, [0.0, 1.0, 0.0])
    return vector_angle_deg(axis_world, [0.0, 0.0, -1.0])


def check_alignment_endpoints(
    start_fk_quat: List[float],
    target_quat: List[float],
    alignment_tolerance_deg: float,
) -> Dict[str, Any]:
    """检查起终点是否满足 tool0 y+ -> world z- 对齐约束。

    Args:
        start_fk_quat: 起点 FK 四元数 [w, x, y, z]。
        target_quat: 目标四元数 [w, x, y, z]。
        alignment_tolerance_deg: 对齐容差（度）。

    Returns:
        包含 valid, start_dev, target_dev, failure_reason 的字典。
    """
    start_dev = compute_alignment_deviation_from_quaternion(start_fk_quat)
    target_dev = compute_alignment_deviation_from_quaternion(target_quat)
    valid = (start_dev <= float(alignment_tolerance_deg)) and (target_dev <= float(alignment_tolerance_deg))
    return {
        'valid': bool(valid),
        'start_alignment_deviation_deg': round(float(start_dev), 6),
        'target_alignment_deviation_deg': round(float(target_dev), 6),
        'alignment_tolerance_deg': float(alignment_tolerance_deg),
        'failure_reason': None if valid else (
            f'start/target alignment precheck failed: '
            f'start_dev={start_dev:.4f} deg, target_dev={target_dev:.4f} deg, '
            f'tolerance={alignment_tolerance_deg:.4f} deg'
        ),
    }


# ============================================================================
# Twist extraction (Phase 9.4)
# ============================================================================

def extract_twist_deg_relative_to_goal(
    pose_quaternion: List[float],
    goal_quaternion: List[float],
) -> float:
    """提取当前姿态相对目标姿态绕 tool0 y 轴的 twist 角（度）。

    Args:
        pose_quaternion: 当前姿态四元数 [w, x, y, z]。
        goal_quaternion: 目标姿态四元数 [w, x, y, z]。

    Returns:
        twist 角度（度）。
    """
    goal_q = normalize_quaternion(list(goal_quaternion))
    pose_q = normalize_quaternion(list(pose_quaternion))
    q_rel = quaternion_multiply(quaternion_conjugate(goal_q), pose_q)
    if q_rel[0] < 0.0:
        q_rel = [-v for v in q_rel]

    twist_q = normalize_quaternion([q_rel[0], 0.0, q_rel[2], 0.0])
    angle_deg = 2.0 * math.degrees(math.atan2(twist_q[2], twist_q[0]))
    return wrap_angle_deg(angle_deg)


def compose_goal_relative_twist_quaternion(
    goal_quaternion: List[float],
    twist_deg: float,
) -> List[float]:
    """按"固定终点姿态 + 绕 tool0 y 的 twist"构造四元数。

    Args:
        goal_quaternion: 目标姿态四元数 [w, x, y, z]。
        twist_deg: twist 角度（度）。

    Returns:
        构造的四元数 [w, x, y, z]。
    """
    goal_q = normalize_quaternion(list(goal_quaternion))
    twist_q = local_y_twist_quaternion(twist_deg)
    return normalize_quaternion(quaternion_multiply(goal_q, twist_q))


# ============================================================================
# Alignment evaluation (Phase 9.5)
# ============================================================================

def compute_axis_alignment_angle_batched(
    positions: torch.Tensor,
    local_axis: torch.Tensor,
    target_world_axis: torch.Tensor,
    kinematics_fn,
) -> torch.Tensor:
    """GPU 批量计算 axis alignment 角度。

    Args:
        positions: 关节位置 [T, DOF] 或 [B*T, DOF]。
        local_axis: 工具坐标系局部轴 [3]。
        target_world_axis: 目标世界轴 [3]。
        kinematics_fn: FK 函数，输入 positions，返回含 ee_quaternion
            或可通过 tool_poses/tool_frames 提取末端四元数的结果。

    Returns:
        alignment_angle_deg [T]：对齐角度（度），完美对齐时为 0°。
    """
    out = kinematics_fn(positions)
    ee_quat = extract_ee_quaternion_batched(out)  # [T, 4] (qw, qx, qy, qz)
    R_ee = quaternion_to_rotation_matrix_batched(ee_quat)  # [T, 3, 3]
    axis_world = torch.matmul(R_ee, local_axis.unsqueeze(0).unsqueeze(-1)).squeeze(-1)  # [T, 3]
    dot_val = torch.sum(axis_world * target_world_axis.unsqueeze(0), dim=-1)  # [T]
    dot_clamped = torch.clamp(dot_val, -1.0, 1.0)
    alignment_error_rad = torch.acos(dot_clamped)
    return torch.rad2deg(alignment_error_rad)


def evaluate_axis_alignment_batched(
    positions: torch.Tensor,
    kinematics_fn,
    alignment_tolerance_deg: float = 3.0,
    local_axis: Optional[torch.Tensor] = None,
    target_world_axis: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """评估 batched 轨迹的 axis alignment。

    Args:
        positions: [B, T, DOF] 关节位置。
        kinematics_fn: FK 函数。
        alignment_tolerance_deg: 对齐容差（度）。
        local_axis: 工具坐标系局部轴 [3]，默认 y+ [0,1,0]。
        target_world_axis: 目标世界轴 [3]，默认 z- [0,0,-1]。

    Returns:
        包含 alignment_angle_map, max_alignment_deviation, alignment_valid 等的字典。
    """
    if local_axis is None:
        local_axis = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=positions.device)
    if target_world_axis is None:
        target_world_axis = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=positions.device)

    B, T, DOF = positions.shape
    positions_flat = positions.reshape(B * T, DOF)

    alignment_angle_flat = compute_axis_alignment_angle_batched(
        positions_flat, local_axis, target_world_axis, kinematics_fn,
    )
    alignment_angle_map = alignment_angle_flat.reshape(B, T)

    max_alignment_deviation = alignment_angle_map.amax(dim=-1)
    mean_alignment_deviation = alignment_angle_map.mean(dim=-1)
    alignment_valid = max_alignment_deviation <= float(alignment_tolerance_deg)

    return {
        'alignment_angle_map': alignment_angle_map,
        'max_alignment_deviation': max_alignment_deviation,
        'mean_alignment_deviation': mean_alignment_deviation,
        'alignment_valid': alignment_valid,
    }


def compute_twist_profile_batched(
    positions: torch.Tensor,
    goal_quaternion: List[float],
    kinematics_fn,
) -> torch.Tensor:
    """计算 batched 轨迹的 twist profile。

    Args:
        positions: [B, T, DOF] 关节位置。
        goal_quaternion: 固定终点四元数 [w, x, y, z]。
        kinematics_fn: FK 函数。

    Returns:
        twist_profile_deg [B, T]：twist 角度（度）。
    """
    B, T, DOF = positions.shape
    flat = positions.reshape(B * T, DOF)
    ee_quat = extract_ee_quaternion_batched(kinematics_fn(flat))  # [B*T, 4]

    goal_q = torch.tensor(
        [normalize_quaternion(list(goal_quaternion))],
        dtype=torch.float32,
        device=positions.device,
    ).expand(B * T, -1)

    goal_q_conj = goal_q.clone()
    goal_q_conj[:, 1:] *= -1.0
    q_rel = quaternion_multiply_batched(goal_q_conj, ee_quat)
    sign = torch.where(q_rel[:, :1] < 0.0, -1.0, 1.0)
    q_rel = q_rel * sign

    twist = torch.stack([
        q_rel[:, 0],
        torch.zeros_like(q_rel[:, 0]),
        q_rel[:, 2],
        torch.zeros_like(q_rel[:, 0]),
    ], dim=-1)
    twist_norm = torch.linalg.norm(twist, dim=-1, keepdim=True).clamp_min(1e-12)
    twist = twist / twist_norm
    angle_deg = torch.rad2deg(2.0 * torch.atan2(twist[:, 2], twist[:, 0]))
    angle_deg = wrap_angle_deg_tensor(angle_deg)
    return angle_deg.reshape(B, T)


def compute_candidate_continuity_metrics(
    positions: torch.Tensor,
    start_joint: List[float],
    goal_quaternion: List[float],
    kinematics_fn,
) -> Dict[str, torch.Tensor]:
    """计算候选轨迹的连续性指标。

    Args:
        positions: [B, T, DOF] 关节位置。
        start_joint: 起始关节角。
        goal_quaternion: 目标四元数 [w, x, y, z]。
        kinematics_fn: FK 函数。

    Returns:
        包含 start_joint_gap_l2、joint_step_jump_cost、twist_smoothness_cost 等的字典。
    """
    start_joint_tensor = torch.tensor(
        [list(start_joint)],
        dtype=positions.dtype,
        device=positions.device,
    ).expand(positions.shape[0], -1)

    first_gap = positions[:, 0, :] - start_joint_tensor
    start_joint_gap_l2 = torch.linalg.norm(first_gap, dim=-1)
    start_joint_gap_max_abs = torch.amax(torch.abs(first_gap), dim=-1)

    joint_step_delta = positions[:, 1:, :] - positions[:, :-1, :]
    joint_step_abs = torch.abs(joint_step_delta)
    joint_step_l2 = torch.linalg.norm(joint_step_delta, dim=-1)
    joint_step_max_abs = torch.amax(joint_step_abs, dim=(-1, -2))
    joint_step_mean_abs = torch.mean(joint_step_abs, dim=(-1, -2))
    joint_step_max_l2 = torch.amax(joint_step_l2, dim=-1)
    joint_step_mean_l2 = torch.mean(joint_step_l2, dim=-1)
    # 以最大单步跳变为主，均值为辅，显式惩罚中途关节突跳。
    joint_step_jump_cost = joint_step_max_l2 + 0.1 * joint_step_mean_l2 + 0.2 * joint_step_max_abs

    twist_profile_deg = compute_twist_profile_batched(positions, goal_quaternion, kinematics_fn)
    twist_delta_deg = wrap_angle_deg_tensor(twist_profile_deg[:, 1:] - twist_profile_deg[:, :-1])
    twist_delta_abs = torch.abs(twist_delta_deg)
    twist_smoothness_cost = torch.amax(twist_delta_abs, dim=-1) + 0.1 * torch.mean(twist_delta_abs, dim=-1)

    return {
        'twist_profile_deg': twist_profile_deg,
        'twist_delta_deg': twist_delta_deg,
        'start_joint_gap_l2': start_joint_gap_l2,
        'start_joint_gap_max_abs': start_joint_gap_max_abs,
        'joint_step_max_abs': joint_step_max_abs,
        'joint_step_mean_abs': joint_step_mean_abs,
        'joint_step_max_l2': joint_step_max_l2,
        'joint_step_mean_l2': joint_step_mean_l2,
        'joint_step_jump_cost': joint_step_jump_cost,
        'twist_smoothness_cost': twist_smoothness_cost,
        'twist_delta_max_abs_deg': torch.amax(twist_delta_abs, dim=-1),
        'twist_delta_mean_abs_deg': torch.mean(twist_delta_abs, dim=-1),
        'start_twist_deg': twist_profile_deg[:, 0],
        'end_twist_deg': twist_profile_deg[:, -1],
        'max_twist_deg': torch.amax(twist_profile_deg, dim=-1),
        'min_twist_deg': torch.amin(twist_profile_deg, dim=-1),
    }


# ============================================================================
# Candidate selection (Phase 9.6)
# ============================================================================

def select_level_first_candidate(
    positions: torch.Tensor,
    level_eval_result: Dict[str, Any],
    continuity_metrics: Dict[str, torch.Tensor],
    level_tolerance_deg: float = 3.0,
    strict_level: bool = True,
    ignore_alignment_for_selection: bool = False,
    candidate_eligibility: Optional[torch.Tensor] = None,
) -> Dict[str, Any]:
    """选择最优候选轨迹（alignment-first 策略）。

    排序规则：
    1. alignment validity（是否在容差内）
    2. start_joint_gap_l2（首点跳变）
    3. joint_step_jump_cost（中途关节突跳风险）
    4. twist_smoothness_cost（twist 平滑度）
    5. joint_path_cost（关节路径长度）

    Args:
        positions: [B, T, DOF] 候选轨迹。
        level_eval_result: evaluate_axis_alignment_batched 的结果。
        continuity_metrics: compute_candidate_continuity_metrics 的结果。
        level_tolerance_deg: 对齐容差（度）。
        strict_level: 是否严格要求对齐。
        ignore_alignment_for_selection: 若为 True，则完全忽略水平约束进行
            候选选择与 planning_status 判定（B4 单向学习种子 baseline：模拟
            level-agnostic 的 DiffusionSeeder，只按连续性/路径代价选点、以
            『到达目标』为成功定义）。候选级别的真实对齐偏差仍照常记录，供
            报告层量化真实水平违约率。默认 False（保持 level-first 语义）。

    Returns:
        包含 selected_index, planning_status 等的字典。
    """
    B = int(positions.shape[0])
    max_dev = level_eval_result['max_alignment_deviation']
    alignment_valid = level_eval_result['alignment_valid']

    start_joint_gap_l2 = continuity_metrics['start_joint_gap_l2']
    joint_step_jump_cost = continuity_metrics['joint_step_jump_cost']
    twist_smoothness_cost = continuity_metrics['twist_smoothness_cost']
    joint_step_max_abs = continuity_metrics['joint_step_max_abs']
    joint_step_max_l2 = continuity_metrics['joint_step_max_l2']
    # [caohy] Task 40：分段优化候选与原始种子在拼接处构型切换硬跳相同，
    # 导致 joint_step_max_l2 平局；引入 mean_l2 作为段内平滑度的次级区分。
    joint_step_mean_l2 = continuity_metrics['joint_step_mean_l2']

    # 关节路径成本
    diffs = positions[:, 1:, :] - positions[:, :-1, :]
    segment_cost = torch.linalg.norm(diffs, dim=-1).sum(dim=-1)
    start_to_end_cost = torch.linalg.norm(positions[:, -1, :] - positions[:, 0, :], dim=-1)
    joint_path_cost = segment_cost + 0.05 * start_to_end_cost

    alignment_valid_count = int(alignment_valid.sum().item())
    if candidate_eligibility is None:
        eligible = torch.ones(B, dtype=torch.bool, device=positions.device)
    else:
        eligible = candidate_eligibility.to(device=positions.device, dtype=torch.bool).reshape(-1)
        if int(eligible.numel()) != B:
            raise ValueError(
                f'candidate_eligibility length mismatch: got {int(eligible.numel())}, expected {B}'
            )
    hard_eligible_count = int(eligible.sum().item())

    result = {
        'candidate_alignment_valid': [bool(v) for v in alignment_valid.tolist()],
        'candidate_max_alignment_deviation': [round(float(v), 4) for v in max_dev.tolist()],
        'candidate_start_joint_gap_l2': [round(float(v), 6) for v in start_joint_gap_l2.tolist()],
        'candidate_joint_step_jump_cost': [round(float(v), 6) for v in joint_step_jump_cost.tolist()],
        'candidate_joint_step_max_abs': [round(float(v), 6) for v in joint_step_max_abs.tolist()],
        'candidate_joint_step_max_l2': [round(float(v), 6) for v in joint_step_max_l2.tolist()],
        'candidate_twist_smoothness_cost': [round(float(v), 6) for v in twist_smoothness_cost.tolist()],
        'candidate_joint_path_costs': [round(float(v), 6) for v in joint_path_cost.tolist()],
        'alignment_valid_count': alignment_valid_count,
        'hard_eligible_count': hard_eligible_count,
        'candidate_count': B,
        'alignment_tolerance_deg': float(level_tolerance_deg),
        'strict_level': bool(strict_level),
    }

    _MAX_L2_QUANT = 0.5
    _GAP_QUANT = 0.05

    def _continuity_sort_key(idx: int):
        return (
            round(float(start_joint_gap_l2[idx].item()) / _GAP_QUANT),
            round(float(joint_step_max_l2[idx].item()) / _MAX_L2_QUANT),
            float(joint_step_mean_l2[idx].item()),
            float(twist_smoothness_cost[idx].item()),
            float(joint_path_cost[idx].item()),
        )

    if ignore_alignment_for_selection:
        # B4 单向学习种子 baseline：level-agnostic 选择——无视对齐，仅按连续性/
        # 路径代价挑最平滑候选，以『到达目标』为成功定义。真实对齐偏差已在
        # candidate_max_alignment_deviation 中记录，报告层据此量化真实违约率。
        eligible_indices = torch.where(eligible)[0].tolist()
        if not eligible_indices:
            best_idx = min(range(B), key=_continuity_sort_key)
            result.update({
                'selected_index': int(best_idx),
                'selected_max_alignment_deviation': round(float(max_dev[best_idx].item()), 4),
                'selected_start_joint_gap_l2': round(float(start_joint_gap_l2[best_idx].item()), 6),
                'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_idx].item()), 6),
                'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_idx].item()), 6),
                'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_idx].item()), 6),
                'selected_twist_smoothness_cost': round(float(twist_smoothness_cost[best_idx].item()), 6),
                'planning_status': 'failed_hard_validation',
                'failure_reason': 'No goal-reaching candidate passed non-alignment hard validation.',
                'selection_mode': 'goal_only_alignment_ignored',
            })
            return result
        best_idx = min(eligible_indices, key=_continuity_sort_key)
        result.update({
            'selected_index': int(best_idx),
            'selected_max_alignment_deviation': round(float(max_dev[best_idx].item()), 4),
            'selected_start_joint_gap_l2': round(float(start_joint_gap_l2[best_idx].item()), 6),
            'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_idx].item()), 6),
            'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_idx].item()), 6),
            'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_idx].item()), 6),
            'selected_twist_smoothness_cost': round(float(twist_smoothness_cost[best_idx].item()), 6),
            'planning_status': 'success',
            'failure_reason': None,
            'selection_mode': 'goal_only_alignment_ignored',
        })
        return result

    hard_alignment_valid = alignment_valid & eligible
    hard_alignment_valid_count = int(hard_alignment_valid.sum().item())
    result['hard_alignment_valid_count'] = hard_alignment_valid_count
    if hard_alignment_valid_count > 0:
        valid_indices = torch.where(hard_alignment_valid)[0]
        # [caohy] Task 40：对 joint_step_max_l2 做容差量化（0.5 rad 一档）作为粗分主键，
        # 让拼接处构型切换硬跳相同的 split 候选与原始种子落到同一档（max_l2 物理必然相同）。
        # 同档内用 joint_step_mean_l2（全程平均步长，反映段内平滑度，不受拼接处单步大跳和点数影响）
        # 区分——段内经过 cspace 优化的 split 候选更平滑而胜出。
        # max_l2 差异大的候选（如 planner vs seed）仍落不同档，保持原有 max-jump 优先行为。
        # start_joint_gap_l2 也做量化（0.05 rad 一档）：split 经 cspace 优化后起点可能有微小偏移，
        # 不量化会让它在首键就与起点精确的原始种子分开，量化 max_l2 失去意义。
        best_valid_idx = min(valid_indices.tolist(), key=_continuity_sort_key)
        result.update({
            'selected_index': int(best_valid_idx),
            'selected_max_alignment_deviation': round(float(max_dev[best_valid_idx].item()), 4),
            'selected_start_joint_gap_l2': round(float(start_joint_gap_l2[best_valid_idx].item()), 6),
            'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_valid_idx].item()), 6),
            'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_valid_idx].item()), 6),
            'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_valid_idx].item()), 6),
            'selected_twist_smoothness_cost': round(float(twist_smoothness_cost[best_valid_idx].item()), 6),
            'planning_status': 'success',
            'failure_reason': None,
        })
    elif alignment_valid_count > 0 and hard_eligible_count == 0:
        best_effort_idx = int(torch.argmin(max_dev).item())
        result.update({
            'selected_index': best_effort_idx,
            'selected_max_alignment_deviation': round(float(max_dev[best_effort_idx].item()), 4),
            'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_effort_idx].item()), 6),
            'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_effort_idx].item()), 6),
            'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_effort_idx].item()), 6),
            'planning_status': 'failed_hard_validation',
            'failure_reason': 'Alignment-valid candidates exist, but none passed complete hard validation.',
        })
    elif not strict_level and hard_eligible_count > 0:
        eligible_indices = torch.where(eligible)[0]
        best_effort_idx = int(eligible_indices[torch.argmin(max_dev[eligible_indices])].item())
        result.update({
            'selected_index': best_effort_idx,
            'selected_max_alignment_deviation': round(float(max_dev[best_effort_idx].item()), 4),
            'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_effort_idx].item()), 6),
            'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_effort_idx].item()), 6),
            'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_effort_idx].item()), 6),
            'planning_status': 'best_effort_alignment_violation',
            'failure_reason': (
                f'No trajectory meets alignment_tolerance={level_tolerance_deg} deg. '
                f'Best effort has max_deviation={float(max_dev[best_effort_idx].item()):.4f} deg.'
            ),
        })
    else:
        best_effort_idx = int(torch.argmin(max_dev).item())
        result.update({
            'selected_index': best_effort_idx,
            'selected_max_alignment_deviation': round(float(max_dev[best_effort_idx].item()), 4),
            'selected_joint_step_jump_cost': round(float(joint_step_jump_cost[best_effort_idx].item()), 6),
            'selected_joint_step_max_abs': round(float(joint_step_max_abs[best_effort_idx].item()), 6),
            'selected_joint_step_max_l2': round(float(joint_step_max_l2[best_effort_idx].item()), 6),
            'planning_status': 'failed_alignment_constraint',
            'failure_reason': (
                f'No trajectory meets strict alignment_tolerance={level_tolerance_deg} deg. '
                f'Best candidate has max_deviation={float(max_dev[best_effort_idx].item()):.4f} deg.'
            ),
        })

    return result
