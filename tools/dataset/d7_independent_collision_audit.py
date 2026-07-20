#!/usr/bin/env python3
"""D7 independent world-collision audit for selected successful trajectories.

This is intentionally separate from ``level_planner_core.validators`` and from
cuRobo's collision checker.  It parses the SR5 URDF joint chain, applies the
repository collision-sphere YAML, reads the obstacle JSON files configured for
the planner run, and computes sphere-vs-oriented-box signed distances.  The
audit is a second implementation for world-obstacle clearance; it is not a
self-collision or FCL replacement.
"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml


Matrix = list[list[float]]
Vector = list[float]


def _identity() -> Matrix:
    return [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    return [[sum(a[i][k] * b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]


def _transform_point(t: Matrix, p: Vector) -> Vector:
    return [
        t[0][0] * p[0] + t[0][1] * p[1] + t[0][2] * p[2] + t[0][3],
        t[1][0] * p[0] + t[1][1] * p[1] + t[1][2] * p[2] + t[1][3],
        t[2][0] * p[0] + t[2][1] * p[1] + t[2][2] * p[2] + t[2][3],
    ]


def _translation(xyz: Vector) -> Matrix:
    t = _identity()
    t[0][3], t[1][3], t[2][3] = xyz
    return t


def _rot_x(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[1, 0, 0, 0], [0, c, -s, 0], [0, s, c, 0], [0, 0, 0, 1]]


def _rot_y(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1]]


def _rot_z(a: float) -> Matrix:
    c, s = math.cos(a), math.sin(a)
    return [[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]


def _rpy_matrix(rpy: Vector) -> Matrix:
    roll, pitch, yaw = rpy
    return _matmul(_matmul(_rot_z(yaw), _rot_y(pitch)), _rot_x(roll))


def _axis_angle(axis: Vector, angle: float) -> Matrix:
    norm = math.sqrt(sum(v * v for v in axis)) or 1.0
    x, y, z = [v / norm for v in axis]
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return [
        [x * x * C + c, x * y * C - z * s, x * z * C + y * s, 0.0],
        [y * x * C + z * s, y * y * C + c, y * z * C - x * s, 0.0],
        [z * x * C - y * s, z * y * C + x * s, z * z * C + c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _quat_mul(a: Vector, b: Vector) -> Vector:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def _quat_conj(q: Vector) -> Vector:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_normalize(q: Vector) -> Vector:
    norm = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / norm for v in q]


def _rotate_vector(q: Vector, v: Vector) -> Vector:
    qn = _quat_normalize(q)
    out = _quat_mul(_quat_mul(qn, [0.0, v[0], v[1], v[2]]), _quat_conj(qn))
    return [out[1], out[2], out[3]]


def _parse_vec(text: str | None, default: Vector) -> Vector:
    if not text:
        return list(default)
    return [float(v) for v in text.split()]


def _origin_matrix(joint: ET.Element) -> Matrix:
    origin = joint.find("origin")
    xyz = _parse_vec(origin.get("xyz") if origin is not None else None, [0.0, 0.0, 0.0])
    rpy = _parse_vec(origin.get("rpy") if origin is not None else None, [0.0, 0.0, 0.0])
    return _matmul(_translation(xyz), _rpy_matrix(rpy))


def _load_chain(urdf: Path, joint_names: list[str]) -> list[dict[str, Any]]:
    root = ET.parse(urdf).getroot()
    joints_by_child: dict[str, ET.Element] = {}
    for joint in root.findall("joint"):
        child = joint.find("child")
        if child is not None:
            joints_by_child[str(child.get("link"))] = joint

    chain: list[ET.Element] = []
    child = "tool1"
    while child in joints_by_child:
        joint = joints_by_child[child]
        chain.append(joint)
        parent = joint.find("parent")
        if parent is None:
            break
        child = str(parent.get("link"))
    chain.reverse()

    active_index = {name: i for i, name in enumerate(joint_names)}
    parsed: list[dict[str, Any]] = []
    for joint in chain:
        name = str(joint.get("name"))
        child_node = joint.find("child")
        axis_node = joint.find("axis")
        parsed.append(
            {
                "name": name,
                "type": str(joint.get("type", "fixed")),
                "child": str(child_node.get("link")) if child_node is not None else "",
                "origin": _origin_matrix(joint),
                "axis": _parse_vec(axis_node.get("xyz") if axis_node is not None else None, [0.0, 0.0, 1.0]),
                "active_index": active_index.get(name),
            }
        )
    return parsed


def _fk_link_transforms(chain: list[dict[str, Any]], q: list[float]) -> dict[str, Matrix]:
    tf = _identity()
    out: dict[str, Matrix] = {"world": tf}
    for joint in chain:
        tf = _matmul(tf, joint["origin"])
        if joint["type"] in {"revolute", "continuous"} and joint["active_index"] is not None:
            tf = _matmul(tf, _axis_angle(joint["axis"], float(q[int(joint["active_index"])])))
        out[joint["child"]] = tf
    return out


def _load_robot_config(config: Path) -> tuple[Path, Path, list[str]]:
    cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
    kin = ((cfg.get("robot_cfg") or {}).get("kinematics") or {})
    urdf = Path(kin["urdf_path"])
    spheres = Path(kin["collision_spheres"])
    if not urdf.is_absolute():
        urdf = (config.parent / urdf).resolve()
    if not spheres.is_absolute():
        spheres = (config.parent / spheres).resolve()
    joint_names = list(((kin.get("cspace") or {}).get("joint_names") or []))
    return urdf, spheres, joint_names


def _resolve_config_path(base_dir: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def _load_planner_config(planner_config: Path) -> dict[str, Any]:
    payload = yaml.safe_load(planner_config.read_text(encoding="utf-8")) or {}
    planner = payload.get("planner") or {}
    base_dir = planner_config.parent
    robot_config = Path(planner.get("robot_config", "robot/xms5_r800_w4g3b4c_v2.yml"))
    if not robot_config.is_absolute():
        robot_config = (base_dir / robot_config).resolve()
    return {
        "robot_config": robot_config,
        "obstacle_json": _resolve_config_path(base_dir, planner.get("obstacle_json")),
        "obstacle_rel_json": _resolve_config_path(base_dir, planner.get("obstacle_rel_json")),
        "per_request_world": bool(planner.get("per_request_world", False)),
    }


def _load_spheres(path: Path) -> dict[str, list[dict[str, Any]]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return payload.get("collision_spheres") or payload


def _load_abs_obstacles(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
    for item in data.get("boxes", []):
        boxes.append(
            {
                "name": f"obstacle_{item['id']}",
                "dims": [float(v) for v in item["size"]],
                "pose": [float(v) for v in item["position"]] + [float(v) for v in item["orientation"]],
                "source": str(path),
            }
        )
    return boxes


def _load_rel_obstacles(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    base_pose = data.get("base_pose")
    if not base_pose or len(base_pose) != 7:
        return []
    base_pos = [float(v) for v in base_pose[:3]]
    base_quat = [float(v) for v in base_pose[3:]]
    boxes = []
    for item in data.get("boxes", []):
        rel_pos = [float(v) for v in item["position"]]
        rel_quat = [float(v) for v in item["orientation"]]
        rotated = _rotate_vector(base_quat, rel_pos)
        abs_pos = [base_pos[i] + rotated[i] for i in range(3)]
        abs_quat = _quat_mul(base_quat, rel_quat)
        boxes.append(
            {
                "name": f"obstacle_rel_{item['id']}",
                "dims": [float(v) for v in item["size"]],
                "pose": abs_pos + abs_quat,
                "source": str(path),
            }
        )
    return boxes


def _boxes_from_sampled_request(request: dict[str, Any]) -> list[dict[str, Any]]:
    world = request.get("world") or {}
    boxes = []
    for index, box in enumerate(world.get("sampled_obstacles") or []):
        if not isinstance(box, dict) or box.get("type", "box") != "box":
            continue
        pos = [float(v) for v in box.get("position", [0.0, 0.0, 0.0])]
        quat = [float(v) for v in box.get("orientation", [1.0, 0.0, 0.0, 0.0])]
        boxes.append(
            {
                "name": str(box.get("name") or f"sample_box_{index}"),
                "dims": [float(v) for v in box["dims"]],
                "pose": pos + quat,
                "source": "request.sampled_obstacles",
            }
        )
    return boxes


def _configured_world_boxes(planner_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return _load_abs_obstacles(planner_cfg.get("obstacle_json")) + _load_rel_obstacles(
        planner_cfg.get("obstacle_rel_json")
    )


def _sphere_box_signed_distance(center: Vector, radius: float, box: dict[str, Any]) -> float:
    dims = [float(v) for v in box["dims"]]
    pose = [float(v) for v in box["pose"]]
    pos = pose[:3]
    quat = pose[3:]
    local = _rotate_vector(_quat_conj(quat), [center[i] - pos[i] for i in range(3)])
    half = [v / 2.0 for v in dims]
    sq_out = 0.0
    inside = True
    inside_margin = float("inf")
    for i in range(3):
        delta = abs(local[i]) - half[i]
        if delta > 0.0:
            inside = False
            sq_out += delta * delta
        else:
            inside_margin = min(inside_margin, -delta)
    if inside:
        return -inside_margin - radius
    return math.sqrt(sq_out) - radius


def _verify_dir(
    chain: list[dict[str, Any]],
    spheres: dict[str, list[dict[str, Any]]],
    configured_boxes: list[dict[str, Any]],
    per_request_world: bool,
    ignored_links: set[str],
    dir_path: Path,
) -> dict[str, Any] | None:
    sel_path = dir_path / "selected_trajectory.json"
    req_path = dir_path / "request.json"
    if not sel_path.exists() or not req_path.exists():
        return None
    selected = json.loads(sel_path.read_text(encoding="utf-8"))
    if selected.get("status") != "success":
        return None
    trajectory = selected.get("trajectory") or []
    if not trajectory:
        return None
    request = json.loads(req_path.read_text(encoding="utf-8"))
    boxes = _boxes_from_sampled_request(request) if per_request_world else configured_boxes
    if not boxes:
        return {
            "request_id": selected.get("request_id"),
            "dir": str(dir_path),
            "skipped": "no_world_boxes",
        }

    min_dist = float("inf")
    min_record: dict[str, Any] = {}
    for point_index, q in enumerate(trajectory):
        transforms = _fk_link_transforms(chain, [float(v) for v in q])
        for link, link_spheres in spheres.items():
            if link in ignored_links:
                continue
            tf = transforms.get(link)
            if tf is None:
                continue
            for sphere_index, sphere in enumerate(link_spheres):
                center = _transform_point(tf, [float(v) for v in sphere["center"]])
                radius = float(sphere["radius"])
                for box in boxes:
                    signed = _sphere_box_signed_distance(center, radius, box)
                    if signed < min_dist:
                        min_dist = signed
                        min_record = {
                            "point_index": point_index,
                            "link": link,
                            "sphere_index": sphere_index,
                            "box": box.get("name"),
                            "box_source": box.get("source"),
                            "signed_distance_m": round(signed, 9),
                        }

    return {
        "request_id": selected.get("request_id"),
        "dir": str(dir_path),
        "n_waypoints": len(trajectory),
        "n_boxes": len(boxes),
        "ignored_links": sorted(ignored_links),
        "world_source": "request.sampled_obstacles" if per_request_world else "planner_config_obstacle_json",
        "independent_min_sphere_box_distance_m": min_dist,
        "world_collision_free": bool(min_dist >= 0.0),
        "min_record": min_record,
    }


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(pct / 100.0 * len(ordered)) - 1))
    return ordered[idx]


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    dist = summary["independent_min_sphere_box_distance_m_distribution"]
    lines = [
        "# D7 Independent World-Collision Audit",
        "",
        f"- scanned selected trajectories: `{summary['n_selected_trajectory_files_scanned']}`",
        f"- world source: `{summary['world_source']}`",
        f"- ignored links: `{summary['ignored_links']}`",
        f"- checked successful trajectories with world boxes: `{summary['n_checked_success_with_boxes']}`",
        f"- skipped successes without world boxes: `{summary['n_skipped_success_no_boxes']}`",
        f"- independent collision findings: `{summary['n_independent_world_collision']}`",
        "",
        "## Distance Distribution",
        "",
        f"- min: `{dist['min']}` m",
        f"- median: `{dist['median']}` m",
        f"- p05: `{dist['p05']}` m",
        f"- p95: `{dist['p95']}` m",
        "",
        "## Scope",
        "",
        summary["scope_note"],
    ]
    if summary["collision_examples"]:
        lines += ["", "## First Collision Examples", ""]
        for item in summary["collision_examples"][:10]:
            lines.append(
                f"- `{item['request_id']}` min `{item['independent_min_sphere_box_distance_m']}` m at `{item['min_record']}`"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D7 independent SR5 sphere-vs-box collision audit.")
    parser.add_argument("eval_dir", type=Path)
    parser.add_argument("--planner-config", type=Path, default=Path("configs/sr5_level.yaml"))
    parser.add_argument("--robot-config", type=Path, help="override robot config; defaults to planner-config robot_config")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument("--max-dirs", type=int, default=0, help="0 = all selected trajectory files")
    parser.add_argument(
        "--ignore-link",
        action="append",
        default=["XMS5-R800-W4G3B4C_base"],
        help="collision-sphere link to ignore; default matches cuRobo dynamic sphere count (54 = total 64 minus base 10)",
    )
    args = parser.parse_args(argv)

    planner_cfg = _load_planner_config(args.planner_config)
    robot_config = args.robot_config or planner_cfg["robot_config"]
    urdf, sphere_path, joint_names = _load_robot_config(robot_config)
    chain = _load_chain(urdf, joint_names)
    spheres = _load_spheres(sphere_path)
    configured_boxes = _configured_world_boxes(planner_cfg)
    ignored_links = set(args.ignore_link or [])
    dirs = sorted(p.parent for p in args.eval_dir.glob("*/*/selected_trajectory.json"))
    if args.max_dirs > 0:
        dirs = dirs[: args.max_dirs]

    checked: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    collisions: list[dict[str, Any]] = []
    for directory in dirs:
        try:
            result = _verify_dir(
                chain,
                spheres,
                configured_boxes,
                bool(planner_cfg.get("per_request_world", False)),
                ignored_links,
                directory,
            )
        except Exception as exc:
            errors.append({"dir": str(directory), "error": f"{type(exc).__name__}: {exc}"})
            continue
        if result is None:
            continue
        if result.get("skipped"):
            skipped.append(result)
            continue
        result["independent_min_sphere_box_distance_m"] = round(
            float(result["independent_min_sphere_box_distance_m"]), 9
        )
        checked.append(result)
        if not result["world_collision_free"]:
            collisions.append(result)

    distances = [float(item["independent_min_sphere_box_distance_m"]) for item in checked]
    summary = {
        "schema_version": "d7_independent_collision_audit.v1",
        "eval_dir": str(args.eval_dir),
        "planner_config": str(args.planner_config),
        "robot_config": str(robot_config),
        "urdf": str(urdf),
        "collision_spheres": str(sphere_path),
        "world_source": "request.sampled_obstacles"
        if bool(planner_cfg.get("per_request_world", False))
        else "planner_config_obstacle_json",
        "configured_world_box_count": len(configured_boxes),
        "ignored_links": sorted(ignored_links),
        "independent_check": "URDF FK + repository collision spheres + sphere-vs-oriented-box signed distance",
        "scope_note": "World-obstacle audit only. This does not check self-collision and is not an FCL replacement; it is a second implementation independent of level_planner_core.validators and cuRobo collision APIs.",
        "n_selected_trajectory_files_scanned": len(dirs),
        "n_checked_success_with_boxes": len(checked),
        "n_skipped_success_no_boxes": len(skipped),
        "n_errors": len(errors),
        "n_independent_world_collision": len(collisions),
        "independent_min_sphere_box_distance_m_distribution": {
            "min": min(distances) if distances else None,
            "p05": _percentile(distances, 5),
            "median": _percentile(distances, 50),
            "p95": _percentile(distances, 95),
            "max": max(distances) if distances else None,
        },
        "collision_examples": collisions[:50],
        "errors": errors[:50],
        "checked": checked,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary, args.md_out)
    print(json.dumps({
        "checked": len(checked),
        "skipped": len(skipped),
        "errors": len(errors),
        "collisions": len(collisions),
        "json_out": str(args.json_out),
        "md_out": str(args.md_out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
