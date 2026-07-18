#!/usr/bin/env python3
"""Sample standalone SR5 level-constrained planning requests."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from level_planner_core.constraint_class import (  # noqa: E402
    CONSTRAINT_CLASS_ORDER,
    DEFAULT_CONSTRAINT_CLASS,
    get_spec as _constraint_class_spec,
    normalize_class_id as _normalize_constraint_class,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE_REQUEST = REPO_ROOT / "examples/requests/request_level_alignment_hard.json"
DEFAULT_OUT = Path("runs/offline_sampling/requests.jsonl")
SCHEMA_VERSION = "planning_task_request_set.v1"


# Level-neighbor goal ball radius, as a multiple of the difficulty profile's
# joint_jitter_rad. The goal joint is drawn within [min, max] joint-L2 of the
# level start so both endpoints share one manifold chart (connectable) while the
# motion stays non-trivial. Overridable via env for calibration sweeps (C2).
# Default max factor 3.0 is the C2-calibrated yield knee: it produced 9.0%
# diffusion-positive yield on medium LP (above the 8.5% base-jitter reference)
# with genuinely diverse, connectable problems (~0.13 rad motions) rather than
# base-jitter's trivial ~0.08 rad micro-moves.
_LEVEL_NEIGHBOR_MAX_FACTOR: float = float(os.environ.get("LEVEL_NEIGHBOR_MAX_FACTOR", "3.0"))
_LEVEL_NEIGHBOR_MIN_FACTOR: float = float(os.environ.get("LEVEL_NEIGHBOR_MIN_FACTOR", "1.0"))


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


def _quat_rotate_vec(quat: list[float], vec: list[float]) -> list[float]:
    """Rotate a 3-vector by a wxyz quaternion (v' = q v q*)."""
    w, x, y, z = quat
    vx, vy, vz = vec
    # t = 2 * cross(q_xyz, v)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return [
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    ]


def _level_align_quat(
    base_quat: list[float],
    local_axis: list[float],
    target_world_axis: list[float],
) -> list[float]:
    """Return a quaternion whose ``local_axis`` maps exactly onto ``target_world_axis``.

    Used for reachable-goal sampling of *level-active* classes: the FK image of
    a random joint config is reachable but its orientation is arbitrary, so its
    tool axis is almost never level.  We pre-multiply the FK orientation by the
    minimal rotation that carries the current world-frame tool axis onto the
    target axis, so the goal is level by construction while staying close to the
    reachable FK orientation (small rotation -> still near the reachable set).
    """
    q = _normalize_quaternion(base_quat)
    la = list(local_axis)
    n = math.sqrt(sum(v * v for v in la)) or 1.0
    la = [v / n for v in la]
    ta = list(target_world_axis)
    n = math.sqrt(sum(v * v for v in ta)) or 1.0
    ta = [v / n for v in ta]
    # current world-frame direction of the tool axis under the FK orientation
    cur = _quat_rotate_vec(q, la)
    n = math.sqrt(sum(v * v for v in cur)) or 1.0
    cur = [v / n for v in cur]
    dot = max(-1.0, min(1.0, sum(c * t for c, t in zip(cur, ta))))
    if dot > 1.0 - 1e-8:
        return q  # already aligned
    if dot < -1.0 + 1e-8:
        # antiparallel: rotate 180deg about any axis perpendicular to cur
        perp = [1.0, 0.0, 0.0]
        if abs(cur[0]) > 0.9:
            perp = [0.0, 1.0, 0.0]
        axis = [
            cur[1] * perp[2] - cur[2] * perp[1],
            cur[2] * perp[0] - cur[0] * perp[2],
            cur[0] * perp[1] - cur[1] * perp[0],
        ]
        n = math.sqrt(sum(v * v for v in axis)) or 1.0
        axis = [v / n for v in axis]
        delta = [0.0, *axis]  # cos(90deg)=0, sin(90deg)=1
    else:
        axis = [
            cur[1] * ta[2] - cur[2] * ta[1],
            cur[2] * ta[0] - cur[0] * ta[2],
            cur[0] * ta[1] - cur[1] * ta[0],
        ]
        n = math.sqrt(sum(v * v for v in axis)) or 1.0
        axis = [v / n for v in axis]
        angle = math.acos(dot)
        half = 0.5 * angle
        delta = [math.cos(half), *(math.sin(half) * v for v in axis)]
    return _quat_multiply(delta, q)


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


def assign_split(
    index: int,
    *,
    val_frac: float,
    test_frac: float,
    rng: random.Random,
) -> str:
    """Assign a problem-level train/val/test split label (C0c).

    The split is assigned per *base problem* (one sampler ``index`` == one base
    problem). Every candidate later derived from this request (the ``k_generate``
    seeds expanded by ``run_lifecycle_batch``) inherits the request's split via
    ``metadata.sampling.split``, so no problem leaks across splits and the C4/E2
    hold-out is instance-level disjoint from train.

    A deterministic per-index uniform draw (seeded RNG) is used rather than a
    modulo bucket so the split is not correlated with the difficulty/obstacle/mode
    cycles (which also key off ``index``).
    """
    draw = rng.random()
    if draw < test_frac:
        return "test"
    if draw < test_frac + val_frac:
        return "val"
    return "train"


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
    split: str = "train",
    independent: bool = False,
    constraint_class: str = DEFAULT_CONSTRAINT_CLASS,
    goal_sampler: Any | None = None,
) -> dict[str, Any]:
    profile = DIFFICULTY_PROFILES[difficulty]
    class_spec = _constraint_class_spec(constraint_class)
    request = json.loads(json.dumps(base))
    base_pose = _target_pose_list(base)
    position_jitter = float(profile["position_jitter_m"])
    goal_witness_joint: list[float] | None = None
    if independent and goal_sampler is not None:
        # C1a (reachable): draw the goal as the FK image of a random feasible
        # joint config -> reachable by construction (the config is itself a valid
        # IK solution), eliminating the ~98% infeasible-goal waste of the blind
        # workspace-box draw.
        local_axis = list((base.get("alignment") or {}).get("local_axis", [0.0, 1.0, 0.0]))
        target_axis = list(
            (base.get("alignment") or {}).get("target_world_axis", [0.0, 0.0, -1.0])
        )
        tol_deg = float(profile["tolerance_deg"])
        if class_spec.level_active:
            # For a level class BOTH endpoints must lie on the level manifold AND
            # be connectable by a level-preserving path. Two *independently* drawn
            # level configs sit in disconnected IK charts (huge joint steps ->
            # ~0% yield), while base-jitter's ~0.08 rad micro-moves are trivial.
            # So: draw a diverse independent level start, then a level GOAL within
            # a bounded joint neighborhood of it (same chart, real motion).
            start_joint = goal_sampler.sample_level_joint(
                rng, local_axis=local_axis, target_world_axis=target_axis, tol_deg=tol_deg
            )[0]
            goal_witness_joint = None
            reachable_pose = None
            if start_joint is not None:
                goal_witness_joint, _gdev = goal_sampler.sample_level_neighbor(
                    rng,
                    start_joint,
                    local_axis=local_axis,
                    target_world_axis=target_axis,
                    tol_deg=tol_deg,
                    radius_rad=float(profile["joint_jitter_rad"]) * _LEVEL_NEIGHBOR_MAX_FACTOR,
                    min_radius_rad=float(profile["joint_jitter_rad"]) * _LEVEL_NEIGHBOR_MIN_FACTOR,
                )
                if goal_witness_joint is not None:
                    reachable_pose = goal_sampler.fk_pose(goal_witness_joint)
            if start_joint is None or reachable_pose is None:
                # Rejection budget exhausted: fall back to a base-jitter start/goal
                # (both near the curated level base config) so the request is still
                # feasible rather than dropped.
                start_joint = [
                    round(float(v) + rng.uniform(-float(profile["joint_jitter_rad"]),
                                                 float(profile["joint_jitter_rad"])), 6)
                    for v in base.get("start_joint", [])
                ]
                pose = [
                    base_pose[0] + rng.uniform(-position_jitter, position_jitter),
                    base_pose[1] + rng.uniform(-position_jitter, position_jitter),
                    base_pose[2] + rng.uniform(-position_jitter, position_jitter),
                    *base_pose[3:7],
                ]
                pose[3:7] = _level_align_quat(pose[3:7], local_axis, target_axis)
            else:
                pose = list(reachable_pose)
                # Snap orientation exactly onto the manifold (FK is within tol_deg;
                # the projection removes the residual so the goal is exactly level).
                pose[3:7] = _level_align_quat(pose[3:7], local_axis, target_axis)
        else:
            # No level gate: a wide independent start is fine; goal is any
            # reachable FK pose.
            start_scale = float(profile["joint_jitter_rad"]) * 6.0
            start_joint = [
                round(float(value) + rng.uniform(-start_scale, start_scale), 6)
                for value in base.get("start_joint", [])
            ]
            reachable_pose, goal_witness_joint = goal_sampler.sample_reachable_pose(rng)
            pose = list(reachable_pose)
            pose[3:7] = _normalize_quaternion(pose[3:7])
    elif independent:
        # C1a: decorrelate start and goal. The start joint gets a *wide*
        # independent draw (a large fraction of the difficulty joint range,
        # scaled up ~6x over the correlated-jitter width) and the goal pose is
        # drawn from a wide independent workspace box around the base pose, so
        # the (start, goal) pair is not a single base config nudged twice.
        # Reachability/IK feasibility is enforced downstream (optional
        # --ik-precheck, else the lifecycle run + hard validators filter
        # infeasible pairs) rather than by a sample-time IK solve.
        start_scale = float(profile["joint_jitter_rad"]) * 6.0
        start_joint = [
            round(float(value) + rng.uniform(-start_scale, start_scale), 6)
            for value in base.get("start_joint", [])
        ]
        goal_pos_scale = float(profile["position_jitter_m"]) * 4.0
        pose = [
            base_pose[0] + rng.uniform(-goal_pos_scale, goal_pos_scale),
            base_pose[1] + rng.uniform(-goal_pos_scale, goal_pos_scale),
            base_pose[2] + rng.uniform(-goal_pos_scale, goal_pos_scale),
            *base_pose[3:7],
        ]
        pose[3:7] = _quat_multiply(
            _random_delta_quat(rng, float(profile["orientation_jitter_deg"]) * 2.0),
            _normalize_quaternion(pose[3:7]),
        )
    else:
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
    # C0c: namespace the request_id by split so the test hold-out is traceable
    # and cannot be silently mixed into the training request set downstream.
    request["request_id"] = _sanitize_request_id(
        f"{split}_sample_{index:05d}_{difficulty}_{mode}_{obstacle_case}"
    )
    request["robot_profile"] = "sr5"
    request["start_joint"] = start_joint
    request["target_pose"] = _pose_dict(pose, base)

    alignment = dict(request.get("alignment") or {})
    alignment["tolerance_deg"] = float(profile["tolerance_deg"])
    alignment.setdefault("local_axis", [0.0, 1.0, 0.0])
    alignment.setdefault("target_world_axis", [0.0, 0.0, -1.0])
    # C1b: the level axis (L vs P) drives strict_level. A ``P`` (no-level) class
    # keeps the alignment fields for reporting but marks the level gate inactive so
    # the planner/validator do not enforce it.
    alignment["strict_level"] = bool(class_spec.level_active)
    request["alignment"] = alignment
    # C1b: publish the canonical class + its two boolean axes on the request so the
    # planner reads one source of truth. The goal-orientation axis (PO vs P suffix)
    # is a position-only goal relaxation applied as a validation/selection gate.
    request["constraint_class"] = class_spec.class_id
    request["constraint_axes"] = {
        "level_active": bool(class_spec.level_active),
        "goal_orientation_active": bool(class_spec.goal_orientation_active),
    }

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
        "note": "Consumed as a real per-request world when the planner runs with per_request_world=True (C0b); otherwise recorded for dataset stratification.",
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
                "split": split,
                "sampling_mode": (
                    "reachable_fk"
                    if (independent and goal_sampler is not None)
                    else ("independent" if independent else "base_jitter")
                ),
                "goal_witness_joint": goal_witness_joint,
                "constraint_class": class_spec.class_id,
                "level_active": bool(class_spec.level_active),
                "goal_orientation_active": bool(class_spec.goal_orientation_active),
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
    constraint_class: str = "mixed",
    modes: list[str] | None = None,
    val_frac: float = 0.1,
    test_frac: float = 0.2,
    independent: bool = False,
    goal_sampler: Any | None = None,
) -> list[dict[str, Any]]:
    if count < 1:
        raise ValueError("count must be >= 1")
    if difficulty not in {"easy", "medium", "hard", "mixed"}:
        raise ValueError(f"unsupported difficulty: {difficulty}")
    if obstacle_case not in {"none", "single", "multi", "mixed"}:
        raise ValueError(f"unsupported obstacle_case: {obstacle_case}")
    if constraint_class != "mixed":
        # Validate + canonicalise a fixed class up-front (uppercases the id so it
        # matches CONSTRAINT_CLASS_ORDER in _cycle_choice); "mixed" cycles all four.
        constraint_class = _normalize_constraint_class(constraint_class)
    if not (0.0 <= val_frac < 1.0 and 0.0 <= test_frac < 1.0 and val_frac + test_frac < 1.0):
        raise ValueError("val_frac/test_frac must be in [0,1) with val+test < 1")
    rng = random.Random(seed)
    # C0c: split assignment uses its own RNG stream so it is statistically
    # independent of the pose-jitter draws and stays stable if jitter changes.
    split_rng = random.Random(seed ^ 0x5F3759DF)
    modes = modes or ["mixed", "rule", "shadow"]
    bases = [(path, _load_json(path)) for path in base_request_paths]
    tasks: list[dict[str, Any]] = []
    for index in range(count):
        base_path, base = bases[index % len(bases)]
        bucket = _cycle_choice(index, difficulty, ["easy", "medium", "hard"])
        obstacle = _cycle_choice(index, obstacle_case, ["none", "single", "multi"])
        klass = _cycle_choice(index, constraint_class, list(CONSTRAINT_CLASS_ORDER))
        mode = modes[index % len(modes)]
        split = assign_split(index, val_frac=val_frac, test_frac=test_frac, rng=split_rng)
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
                split=split,
                independent=independent,
                constraint_class=klass,
                goal_sampler=goal_sampler,
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
        "constraint_class_counts": _counter_from_tasks(tasks, "constraint_class"),
        "mode_counts": _counter_from_tasks(tasks, "seed_policy_mode"),
        "split_counts": _counter_from_tasks(tasks, "split"),
        "sampling_mode_counts": _counter_from_tasks(tasks, "sampling_mode"),
        "reachable_goal_witness_count": sum(
            1
            for t in tasks
            if ((t.get("metadata") or {}).get("sampling") or {}).get("goal_witness_joint")
        ),
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
            "note": "Obstacle cases become a real per-request world when the planner runs with per_request_world=True (C0b); otherwise they remain stratification metadata.",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-request", type=Path, action="append", default=[])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard", "mixed"], default="mixed")
    parser.add_argument("--obstacle-case", choices=["none", "single", "multi", "mixed"], default="mixed")
    parser.add_argument(
        "--constraint-class",
        choices=[*CONSTRAINT_CLASS_ORDER, *[c.lower() for c in CONSTRAINT_CLASS_ORDER], "mixed"],
        default="mixed",
        help="C1b: constraint class to sample. 'mixed' cycles LPO/LP/PPO/PP; "
             "a fixed id (e.g. LPO) samples that class only.",
    )
    parser.add_argument("--modes", default="mixed,rule,shadow")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--manifest-out", type=Path)
    parser.add_argument("--val-frac", type=float, default=0.1,
                        help="C0c: problem-level fraction assigned to the val split.")
    parser.add_argument("--test-frac", type=float, default=0.2,
                        help="C0c: problem-level fraction assigned to the held-out test split.")
    parser.add_argument("--independent-sampling", action="store_true",
                        help="C1a: draw start joint and goal pose independently (wide) "
                             "instead of jittering a single base config twice.")
    parser.add_argument("--reachable-goals", action="store_true",
                        help="C1a: with --independent-sampling, draw each goal as the FK "
                             "image of a random feasible joint config (reachable by "
                             "construction) instead of a blind workspace-box pose. "
                             "Requires a GPU + cuRobo; eliminates infeasible-goal waste.")
    parser.add_argument("--robot-config", type=Path,
                        default=REPO_ROOT / "configs/robot/xms5_r800_w4g3b4c_v2.yml",
                        help="Robot config YAML used to build the FK model for "
                             "--reachable-goals (default: SR5).")
    parser.add_argument("--goal-device", default="cuda:0",
                        help="CUDA device for the reachable-goal FK model.")
    args = parser.parse_args(argv)

    base_paths = args.base_request or [DEFAULT_BASE_REQUEST]
    modes = _parse_modes(args.modes)

    goal_sampler = None
    if args.reachable_goals:
        if not args.independent_sampling:
            parser.error("--reachable-goals requires --independent-sampling")
        from tools.dataset.reachable_goals import ReachableGoalSampler

        goal_sampler = ReachableGoalSampler(
            robot_config_path=Path(args.robot_config),
            device=str(args.goal_device),
        )

    tasks = generate_task_requests(
        base_request_paths=base_paths,
        count=int(args.count),
        seed=int(args.seed),
        difficulty=str(args.difficulty),
        obstacle_case=str(args.obstacle_case),
        constraint_class=str(args.constraint_class),
        modes=modes,
        val_frac=float(args.val_frac),
        test_frac=float(args.test_frac),
        independent=bool(args.independent_sampling),
        goal_sampler=goal_sampler,
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
