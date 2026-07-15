#!/usr/bin/env python3
# [caohy] Phase 5：适配 tashan_robot 项目，删除 curoboV2_demo 硬编码路径。
# 基于 curoboV2_demo/scripts/rokae_asset_utils.py 改写。
"""ROKAE 资产包路径与配置辅助函数。

支持两种碰撞球来源：
1. 从 YAML 文件加载（传统方式）
2. 使用 CuRobo V2 RobotBuilder 自动生成（推荐）
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: Path) -> Any:
    return yaml.safe_load(Path(path).read_text())


def generate_collision_spheres(
    urdf_path: Path,
    asset_path: Path,
    sphere_density: float = 1.0,
    num_collision_samples: int = 1000,
    compute_metrics: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """使用 CuRobo V2 RobotBuilder 自动生成碰撞球。

    Args:
        urdf_path: URDF 文件路径。
        asset_path: mesh 资产目录。
        sphere_density: 球密度倍数（默认 1.0，越大球越多）。
        num_collision_samples: 碰撞裁剪采样数。
        compute_metrics: 是否计算拟合质量指标。

    Returns:
        碰撞球字典，格式为 {link_name: [{"center": [x,y,z], "radius": r}, ...]}
    """
    from curobo.robot_builder import RobotBuilder

    print(f"Generating collision spheres using CuRobo V2 RobotBuilder...")
    print(f"  URDF: {urdf_path}")
    print(f"  Asset path: {asset_path}")

    builder = RobotBuilder(
        urdf_path=str(urdf_path),
        asset_path=str(asset_path),
        tool_frames=["tool0"],
    )

    print("\nFitting collision spheres...")
    builder.fit_collision_spheres(
        sphere_density=sphere_density,
        compute_metrics=compute_metrics,
    )
    print(f"Fitted {builder.num_spheres} spheres across {len(builder.collision_link_names)} links")

    print("\nComputing collision matrix...")
    builder.compute_collision_matrix(num_samples=num_collision_samples)
    print(f"Created collision ignore matrix with {len(builder.collision_matrix)} entries")

    if compute_metrics and builder.link_metrics:
        print(f"\n{'Link':<35s} {'n_sph':>5s} {'cover%':>7s} {'protr%':>7s}")
        print("-" * 60)
        for link_name, m in builder.link_metrics.items():
            print(
                f"{link_name:<35s} {m.num_spheres:5d} "
                f"{m.coverage * 100:6.1f}% {m.protrusion * 100:6.1f}%"
            )

    raw_spheres = builder.collision_spheres
    if raw_spheres:
        first_link = list(raw_spheres.keys())[0]
        if raw_spheres[first_link]:
            first_sphere = raw_spheres[first_link][0]
            center = first_sphere["center"]
            max_coord = max(abs(c) for c in center) if center else 0
            if max_coord > 10:
                print(f"  Detected millimeter-scale coordinates (max={max_coord:.1f}), scaling to meters...")
                for link_name in raw_spheres:
                    for sphere in raw_spheres[link_name]:
                        sphere["center"] = [c / 1000.0 for c in sphere["center"]]
                        sphere["radius"] = sphere["radius"] / 1000.0

    return raw_spheres


def resolve_robot_config(
    robot_config_path: Path,
    auto_generate_spheres: bool = True,
    sphere_density: float = 0.3,
) -> dict[str, Any]:
    """加载并归一化机器人配置。

    Args:
        robot_config_path: 机器人配置 YAML 的绝对路径。
        auto_generate_spheres: 如果为 True，使用 CuRobo V2 自动生成碰撞球。
        sphere_density: 自动生成时的球密度倍数。

    Returns:
        归一化后的机器人配置字典（路径已转绝对，碰撞球已内联）。
    """
    config_path = Path(robot_config_path)
    robot_cfg = copy.deepcopy(load_yaml(config_path))
    kinematics_cfg = robot_cfg["robot_cfg"]["kinematics"]

    for key in ("urdf_path", "asset_root_path"):
        value = kinematics_cfg.get(key)
        if not isinstance(value, str) or not value or "://" in value:
            continue
        path = Path(value)
        if not path.is_absolute():
            kinematics_cfg[key] = str((config_path.parent / path).resolve())

    if auto_generate_spheres:
        print("Using CuRobo V2 to generate collision spheres...")
        urdf_path = Path(kinematics_cfg["urdf_path"])
        asset_path = Path(kinematics_cfg.get("asset_root_path", config_path.parent / "curobo"))
        print(f"Collision spheres source: auto-generated from URDF {urdf_path}")

        kinematics_cfg["collision_spheres"] = generate_collision_spheres(
            urdf_path=urdf_path,
            asset_path=asset_path,
            sphere_density=sphere_density,
        )
    else:
        collision_spheres = kinematics_cfg.get("collision_spheres")
        if isinstance(collision_spheres, str):
            collision_spheres_path = Path(collision_spheres)
            if not collision_spheres_path.is_absolute():
                collision_spheres_path = (config_path.parent / collision_spheres_path).resolve()
            if collision_spheres_path.exists():
                # [caohy] Phase 15：兼容 V1 风格 sphere 文件的顶层包装结构。
                # 这类 YAML 顶层通常是 {"collision_spheres": {...}}，而 CuRobo V2 loader
                # 期望直接拿到按 link_name 分组的 dict，否则会在读取 Link_01 等键时报 KeyError。
                loaded_spheres = load_yaml(collision_spheres_path)
                if isinstance(loaded_spheres, dict) and "collision_spheres" in loaded_spheres:
                    loaded_spheres = loaded_spheres["collision_spheres"]
                kinematics_cfg["collision_spheres"] = loaded_spheres
            else:
                print(f"Spheres file not found: {collision_spheres_path}, auto-generating...")
                urdf_path = Path(kinematics_cfg["urdf_path"])
                asset_path = Path(kinematics_cfg.get("asset_root_path", config_path.parent / "curobo"))
                kinematics_cfg["collision_spheres"] = generate_collision_spheres(
                    urdf_path=urdf_path,
                    asset_path=asset_path,
                    sphere_density=sphere_density,
                )

    return robot_cfg
