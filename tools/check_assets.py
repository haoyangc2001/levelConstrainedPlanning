#!/usr/bin/env python3
"""Validate the standalone SR5 planning config and asset closure."""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve(base: Path, value: str | None) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else (base / p).resolve()


def _record(checks: list[dict[str, Any]], name: str, ok: bool, detail: str, required: bool = True) -> None:
    checks.append({
        "name": name,
        "ok": bool(ok),
        "required": bool(required),
        "detail": detail,
    })


def _check_exists(checks: list[dict[str, Any]], name: str, path: Path | None, required: bool = True) -> bool:
    ok = bool(path and path.exists())
    _record(checks, name, ok or not required, str(path) if path else "missing", required=required)
    return ok


def _check_urdf_meshes(checks: list[dict[str, Any]], urdf_path: Path) -> None:
    try:
        root = ET.fromstring(urdf_path.read_text(encoding="utf-8"))
    except Exception as exc:
        _record(checks, "urdf_parse", False, f"{urdf_path}: {exc}")
        return
    _record(checks, "urdf_parse", True, str(urdf_path))
    mesh_files = []
    missing = []
    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if not filename:
            continue
        if "://" in filename:
            missing.append(filename)
            continue
        mesh_path = (urdf_path.parent / filename).resolve()
        mesh_files.append(str(mesh_path))
        if not mesh_path.exists():
            missing.append(str(mesh_path))
    _record(
        checks,
        "urdf_mesh_paths",
        not missing,
        json.dumps({
            "mesh_reference_count": len(mesh_files),
            "missing": missing,
        }, ensure_ascii=False),
    )


def check_assets(config_path: Path, strict_public_artifacts: bool = False) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    repo_root = config_path.parent.parent
    config = _load_yaml(config_path)
    _record(checks, "planner_config_yaml", isinstance(config, dict), str(config_path))
    planner = config.get("planner", config)
    config_dir = config_path.parent

    robot_config = _resolve(config_dir, planner.get("robot_config"))
    obstacle_json = _resolve(config_dir, planner.get("obstacle_json"))
    obstacle_rel_json = _resolve(config_dir, planner.get("obstacle_rel_json"))
    _check_exists(checks, "robot_config_exists", robot_config)
    _check_exists(checks, "obstacle_json_exists", obstacle_json)
    _check_exists(checks, "obstacle_rel_json_exists", obstacle_rel_json, required=False)

    if robot_config and robot_config.exists():
        robot_payload = _load_yaml(robot_config)
        _record(checks, "robot_config_yaml", isinstance(robot_payload, dict), str(robot_config))
        kin = robot_payload.get("robot_cfg", {}).get("kinematics", {})
        urdf_path = _resolve(robot_config.parent, kin.get("urdf_path"))
        asset_root = _resolve(robot_config.parent, kin.get("asset_root_path"))
        spheres_path = _resolve(robot_config.parent, kin.get("collision_spheres"))
        _check_exists(checks, "robot_urdf_exists", urdf_path)
        _check_exists(checks, "robot_asset_root_exists", asset_root)
        _check_exists(checks, "collision_spheres_exists", spheres_path)
        if spheres_path and spheres_path.exists():
            spheres = _load_yaml(spheres_path)
            _record(checks, "collision_spheres_yaml", isinstance(spheres, dict), str(spheres_path))
        if urdf_path and urdf_path.exists():
            _check_urdf_meshes(checks, urdf_path)

    for name, path in (("obstacle_json_parse", obstacle_json), ("obstacle_rel_json_parse", obstacle_rel_json)):
        if path and path.exists():
            try:
                payload = _load_json(path)
                _record(checks, name, isinstance(payload, dict), str(path))
            except Exception as exc:
                _record(checks, name, False, f"{path}: {exc}")

    artifact_pointer = repo_root / "artifacts/current_artifacts.json"
    if _check_exists(checks, "artifact_pointer_exists", artifact_pointer):
        artifacts = _load_json(artifact_pointer)
        _record(checks, "artifact_pointer_json", isinstance(artifacts, dict), str(artifact_pointer))
        public_paths = [
            ("training_dataset", artifacts.get("dataset", {}).get("training_dataset", {}).get("path")),
            ("dataset_manifest", artifacts.get("dataset", {}).get("dataset_manifest", {}).get("path")),
            ("diffusion_best_checkpoint", artifacts.get("diffusion", {}).get("best_checkpoint")),
            ("critic_best_checkpoint", artifacts.get("critic", {}).get("best_checkpoint")),
            ("offline_generation_report", artifacts.get("offline_generation_report", {}).get("path")),
        ]
        for name, value in public_paths:
            path = Path(value) if value else None
            required = bool(strict_public_artifacts)
            exists = bool(path and path.exists())
            _record(
                checks,
                f"public_artifact_{name}",
                exists or not required,
                str(path) if path else "missing pointer",
                required=required,
            )

    manifest = repo_root / "configs/asset_manifest.json"
    if _check_exists(checks, "asset_manifest_exists", manifest):
        _record(checks, "asset_manifest_json", isinstance(_load_json(manifest), dict), str(manifest))

    ok = all(item["ok"] for item in checks if item["required"])
    return {
        "ok": ok,
        "config": str(config_path),
        "strict_public_artifacts": bool(strict_public_artifacts),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sr5_level.yaml")
    parser.add_argument("--strict-public-artifacts", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)

    report = check_assets(Path(args.config), strict_public_artifacts=args.strict_public_artifacts)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
