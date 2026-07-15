#!/usr/bin/env python3
"""Sample standalone SR5 level-constrained planning requests."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_REQUEST = REPO_ROOT / "examples/requests/request_level_alignment_hard.json"
DEFAULT_OUT = Path("runs/offline_sampling/requests.jsonl")
SCHEMA_VERSION = "planning_task_request_set.v1"


DIFFICULTY_PROFILES: dict[str, dict[str, float]] = {
    "easy": {
        "position_jitter_m": 0.008,
        "orientation_jitter_deg": 3.0,
        "joint_jitter_rad": 0.025,
        "tolerance_deg": 15.0,
        "num_candidates": 2,
    },
    "medium": {
        "position_jitter_m": 0.018,
        "orientation_jitter_deg": 6.0,
        "joint_jitter_rad": 0.045,
        "tolerance_deg": 8.0,
        "num_candidates": 3,
    },
    "hard": {
        "position_jitter_m": 0.032,
        "orientation_jitter_deg": 10.0,
        "joint_jitter_rad": 0.070,
        "tolerance_deg": 3.0,
        "num_candidates": 4,
    },
}


OBSTACLE_LAYOUTS: dict[str, list[dict[str, Any]]] = {
    "none": [],
    "single": [
        {
            "name": "sample_box_0",
            "type": "box",
            "position": [-0.08, -0.20, 0.32],
            "dims": [0.08, 0.08, 0.18],
        }
    ],
    "multi": [
        {
            "name": "sample_box_0",
            "type": "box",
            "position": [-0.08, -0.20, 0.32],
            "dims": [0.08, 0.08, 0.18],
        },
        {
            "name": "sample_box_1",
            "type": "box",
            "position": [-0.24, -0.12, 0.40],
            "dims": [0.06, 0.10, 0.14],
        },
    ],
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"base request must be a JSON object: {path}")
    return payload


def _target_pose_list(request: dict[str, Any]) -> list[float]:
    pose = request.get("target_pose") or {}
    if isinstance(pose, list):
        values = [float(v) for v in pose]
        if len(values) != 7:
            raise ValueError("target_pose list must have 7 values")
        return values
    position = [float(v) for v in pose.get("position", [])]
    quat = [float(v) for v in pose.get("quaternion_wxyz", [])]
    if len(position) != 3 or len(quat) != 4:
        raise ValueError("target_pose must contain position[3] and quaternion_wxyz[4]")
    return position + _normalize_quaternion(quat)


def _pose_dict(values: list[float], template: dict[str, Any]) -> dict[str, Any] | list[float]:
    if isinstance(template.get("target_pose"), list):
        return [round(float(v), 6) for v in values]
    pose = dict(template.get("target_pose") or {})
    pose["position"] = [round(float(v), 6) for v in values[:3]]
    pose["quaternion_wxyz"] = [round(float(v), 6) for v in _normalize_quaternion(values[3:7])]
    return pose


def _normalize_quaternion(quat: list[float]) -> list[float]:
    norm = math.sqrt(sum(float(v) * float(v) for v in quat))
    if norm <= 1e-12:
        return [1.0, 0.0, 0.0, 0.0]
    return [float(v) / norm for v in quat]


def _quat_multiply(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return _normalize_quaternion(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def _random_delta_quat(rng: random.Random, max_angle_deg: float) -> list[float]:
    axis = [rng.uniform(-1.0, 1.0) for _ in range(3)]
    norm = math.sqrt(sum(v * v for v in axis))
    if norm <= 1e-12:
        axis = [0.0, 0.0, 1.0]
    else:
        axis = [v / norm for v in axis]
    angle = math.radians(rng.uniform(-max_angle_deg, max_angle_deg))
    half = 0.5 * angle
    return _normalize_quaternion([math.cos(half), *(math.sin(half) * v for v in axis)])


def _cycle_choice(index: int, requested: str, choices: list[str]) -> str:
    if requested != "mixed":
        return requested
    return choices[index % len(choices)]


def _parse_modes(value: str) -> list[str]:
    modes = [item.strip().lower() for item in value.split(",") if item.strip()]
    allowed = {"rule", "mixed", "shadow", "diffusion", "candidate"}
    invalid = sorted(set(modes) - allowed)
    if invalid:
        raise ValueError(f"unsupported seed policy modes: {invalid}")
    return modes or ["mixed"]


def _sanitize_request_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return cleaned.strip("_") or "request"


def _sample_request(
    *,
    base: dict[str, Any],
    base_path: Path,
    rng: random.Random,
    index: int,
    seed: int,
    difficulty: str,
    obstacle_case: str,
    mode: str,
) -> dict[str, Any]:
    profile = DIFFICULTY_PROFILES[difficulty]
    request = json.loads(json.dumps(base))
    base_pose = _target_pose_list(base)
    position_jitter = float(profile["position_jitter_m"])
    pose = [
        base_pose[0] + rng.uniform(-position_jitter, position_jitter),
        base_pose[1] + rng.uniform(-position_jitter, position_jitter),
        base_pose[2] + rng.uniform(-position_jitter, position_jitter),
        *base_pose[3:7],
    ]
    pose[3:7] = _quat_multiply(
        _random_delta_quat(rng, float(profile["orientation_jitter_deg"])),
        _normalize_quaternion(pose[3:7]),
    )
    joint_jitter = float(profile["joint_jitter_rad"])
    start_joint = [
        round(float(value) + rng.uniform(-joint_jitter, joint_jitter), 6)
        for value in base.get("start_joint", [])
    ]
    request["schema_version"] = "1.0"
    request["request_id"] = _sanitize_request_id(f"sample_{index:05d}_{difficulty}_{mode}_{obstacle_case}")
    request["robot_profile"] = "sr5"
    request["start_joint"] = start_joint
    request["target_pose"] = _pose_dict(pose, base)

    alignment = dict(request.get("alignment") or {})
    alignment["tolerance_deg"] = float(profile["tolerance_deg"])
    alignment.setdefault("local_axis", [0.0, 1.0, 0.0])
    alignment.setdefault("target_world_axis", [0.0, 0.0, -1.0])
    alignment["strict_level"] = True
    request["alignment"] = alignment

    k_generate = 4 if mode in {"mixed", "shadow", "diffusion", "candidate"} else 2
    seed_policy = dict(request.get("seed_policy") or {})
    seed_policy.update(
        {
            "mode": mode,
            "k_generate": k_generate,
            "k_accept": min(2, k_generate),
            "fallback_to_rule_seed": True,
            "timeout_sec": 2.0 if mode in {"mixed", "diffusion", "candidate"} else 0.5,
        }
    )
    request["seed_policy"] = seed_policy

    request["world"] = {
        **dict(request.get("world") or {}),
        "sampled_obstacle_case": obstacle_case,
        "sampled_obstacles": OBSTACLE_LAYOUTS.get(obstacle_case, []),
        "note": "Current standalone planner loads obstacles from config; sampled layout is recorded for dataset stratification.",
    }
    metadata = dict(request.get("metadata") or {})
    metadata.update(
        {
            "sampling": {
                "schema_version": SCHEMA_VERSION,
                "sample_index": int(index),
                "global_seed": int(seed),
                "difficulty_bucket": difficulty,
                "obstacle_case": obstacle_case,
                "seed_policy_mode": mode,
                "base_request": str(base_path),
            },
            "num_candidates": int(profile["num_candidates"]),
        }
    )
    request["metadata"] = metadata
    return request


def generate_task_requests(
    *,
    base_request_paths: list[Path],
    count: int,
    seed: int,
    difficulty: str = "mixed",
    obstacle_case: str = "mixed",
    modes: list[str] | None = None,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if difficulty not in {"easy", "medium", "hard", "mixed"}:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if obstacle_case not in {"none", "single", "multi", "mixed"}:
        raise ValueError(f"unsupported obstacle_case: {obstacle_case}")
    rng = random.Random(seed)
    modes = modes or ["mixed", "rule", "shadow"]
    bases = [(path, _load_json(path)) for path in base_request_paths]
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        base_path, base = bases[index % len(bases)]
        bucket = _cycle_choice(index, difficulty, ["easy", "medium", "hard"])
        obstacle = _cycle_choice(index, obstacle_case, ["none", "single", "multi"])
        mode = modes[index % len(modes)]
        tasks.append(
            _sample_request(
                base=base,
                base_path=base_path,
                rng=rng,
                index=index,
                seed=seed,
                difficulty=bucket,
                obstacle_case=obstacle,
                mode=mode,
            )
        )
    return tasks


def write_requests_jsonl(tasks: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False) + "\n")


def _counter_from_tasks(tasks: list[dict[str, Any]], key: str) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for task in tasks:
        sampling = (task.get("metadata") or {}).get("sampling") or {}
        counter[str(sampling.get(key) or "unknown")] += 1
    return dict(sorted(counter.items()))


def build_manifest(
    *,
    tasks: list[dict[str, Any]],
    out: Path,
    base_request_paths: list[Path],
    seed: int,
    modes: list[str],
    command: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "planning_task_sampling_manifest.v1",
        "created_at": _utc_now(),
        "git_commit": _git_commit(),
        "request_jsonl": str(out),
        "request_count": len(tasks),
        "base_requests": [str(path) for path in base_request_paths],
        "seed": int(seed),
        "seed_policy_modes": modes,
        "difficulty_bucket_counts": _counter_from_tasks(tasks, "difficulty_bucket"),
        "obstacle_case_counts": _counter_from_tasks(tasks, "obstacle_case"),
        "mode_counts": _counter_from_tasks(tasks, "seed_policy_mode"),
        "command": command or [],
        "request_schema": {
            "required": [
                "request_id",
                "robot_profile",
                "start_joint",
                "target_pose",
                "alignment",
                "seed_policy",
                "metadata.sampling",
            ],
            "note": "Obstacle cases are stratification metadata until per-request world reload is introduced.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-request", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--obstacle-case", choices=["none", "single", "multi", "mixed"], default="mixed")
    parser.add_argument("--modes", default="mixed,rule,shadow")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path)
    args = parser.parse_args(argv)

    base_paths = args.base_request or [DEFAULT_BASE_REQUEST]
    modes = _parse_modes(args.modes)
    tasks = generate_task_requests(
        base_request_paths=base_paths,
        count=int(args.count),
        seed=int(args.seed),
        difficulty=str(args.difficulty),
        obstacle_case=str(args.obstacle_case),
        modes=modes,
    )
    write_requests_jsonl(tasks, args.out)
    manifest_path = args.manifest_out or args.out.with_suffix(".manifest.json")
    manifest = build_manifest(
        tasks=tasks,
        out=args.out,
        base_request_paths=base_paths,
        seed=int(args.seed),
        modes=modes,
        command=sys.argv if argv is None else ["python", "tools/dataset/sample_tasks.py", *argv],
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
