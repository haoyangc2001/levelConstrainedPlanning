#!/usr/bin/env python3
"""Collect repeated SR5 lifecycle runs for diffusion seed learning.

The script starts the headless Level_Test_V2_caohy launch, triggers STATE_TEST
through external_comm HTTP, and records which lifecycle files changed after each
trigger. It does not export or validate the dataset; use the phase-2 exporter
and phase-3 validator after collection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LIFECYCLE_ROOT = (
    REPO_ROOT / "readCaohy/logs/trajectory_planning/level_plan_lifecycle"
)
DATAGEN_SOURCE_MOTION = (
    REPO_ROOT
    / "resource/config/Level_Test_V2_caohy/state_machines/state_test/motion/level_test_100_sr5.yaml"
)
DATAGEN_GENERATED_MOTION = (
    REPO_ROOT
    / "resource/config/Level_Test_V2_caohy/state_machines/state_test/motion/generated/"
    "level_test_100_sr5_datagen.yaml"
)
DATAGEN_INDEPENDENT_GENERATED_MOTION = (
    REPO_ROOT
    / "resource/config/Level_Test_V2_caohy/state_machines/state_test/motion/generated/"
    "level_test_100_sr5_independent_datagen.yaml"
)
DEFAULT_REPORT_ROOT = Path(
    "/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports"
)
SR5_TARGET_BOUNDS = {
    "x": (-0.36, 0.36),
    "y": (-0.35, 0.35),
    "z": (0.43, 0.62),
}
DEFAULT_PAYLOAD = {
    "task_id": "SR5_LEVEL_100_DATA_GEN",
    "obj_type": "MGZ",
    "task_type": "STATE_TEST",
    "device_id": "WAREHOUSE_MAIN",
    "device_slot_id": "1",
    "agv_slot_id": "1",
    "refresh_reference": False,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--task-set",
        choices=("20", "100", "100_datagen", "100_independent_datagen"),
        default="100_independent_datagen",
    )
    parser.add_argument("--project", default="Level_Test_V2_caohy")
    parser.add_argument("--robot-profile", default="sr5")
    parser.add_argument("--diffusion-seed-mode", default="off")
    parser.add_argument("--http-host", default="127.0.0.1")
    parser.add_argument("--http-port", type=int, default=8082)
    parser.add_argument("--ready-timeout-sec", type=float, default=240.0)
    parser.add_argument("--run-timeout-sec", type=float, default=7200.0)
    parser.add_argument("--between-run-sleep-sec", type=float, default=2.0)
    parser.add_argument("--datagen-sim-delay", type=float, default=0.0)
    parser.add_argument("--datagen-sim-time-scale", type=float, default=0.2)
    parser.add_argument(
        "--datagen-position-jitter-m",
        type=float,
        default=0.0,
        help="Uniform x/y target pose jitter in meters for strict datagen moves.",
    )
    parser.add_argument(
        "--datagen-z-jitter-m",
        type=float,
        default=0.0,
        help="Uniform z target pose jitter in meters for strict datagen moves.",
    )
    parser.add_argument(
        "--datagen-jitter-seed",
        type=int,
        default=0,
        help="Deterministic seed for strict target pose jitter.",
    )
    parser.add_argument(
        "--datagen-max-strict-moves",
        type=int,
        default=None,
        help="Generate only the first N strict moves from level_test_100_sr5.yaml for pilot runs.",
    )
    parser.add_argument(
        "--datagen-start-strict-index",
        type=int,
        default=0,
        help="Zero-based strict-move offset used when generating a datagen subset.",
    )
    parser.add_argument(
        "--sweep-strict-moves",
        action="store_true",
        help="Generate one strict move per launch, advancing from datagen-start-strict-index.",
    )
    parser.add_argument("--lifecycle-root", type=Path, default=DEFAULT_LIFECYCLE_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--payload-json", type=Path, default=None)
    parser.add_argument("--keep-launch", action="store_true")
    parser.add_argument("--reuse-existing-launch", action="store_true")
    parser.add_argument(
        "--restart-launch-per-run",
        action="store_true",
        help="Restart ROS launch for every trigger so sim state and lifecycle run_dir are isolated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_payload(path: Path | None) -> dict[str, Any]:
    if path is None:
        return dict(DEFAULT_PAYLOAD)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"payload-json must contain a JSON object: {path}")
    return payload


def clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def jitter_strict_move(
    move: dict[str, Any],
    args: argparse.Namespace,
    *,
    strict_global_index: int,
) -> dict[str, Any]:
    """Apply bounded deterministic xyz jitter to a strict move copy."""
    item = copy.deepcopy(move)
    xy_jitter = max(0.0, float(args.datagen_position_jitter_m))
    z_jitter = max(0.0, float(args.datagen_z_jitter_m))
    if xy_jitter <= 0.0 and z_jitter <= 0.0:
        return item
    inputs = item.get("run_args", {}).get("input", {})
    pose = inputs.get("target_pose")
    if not isinstance(pose, list) or len(pose) < 7:
        return item
    rng = random.Random(int(args.datagen_jitter_seed) * 1_000_003 + int(strict_global_index))
    dx = rng.uniform(-xy_jitter, xy_jitter) if xy_jitter > 0.0 else 0.0
    dy = rng.uniform(-xy_jitter, xy_jitter) if xy_jitter > 0.0 else 0.0
    dz = rng.uniform(-z_jitter, z_jitter) if z_jitter > 0.0 else 0.0
    pose[0] = round(clamp(float(pose[0]) + dx, *SR5_TARGET_BOUNDS["x"]), 8)
    pose[1] = round(clamp(float(pose[1]) + dy, *SR5_TARGET_BOUNDS["y"]), 8)
    pose[2] = round(clamp(float(pose[2]) + dz, *SR5_TARGET_BOUNDS["z"]), 8)
    metadata = item.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["datagen_jitter"] = {
            "strict_global_index": int(strict_global_index),
            "seed": int(args.datagen_jitter_seed),
            "dx_m": round(dx, 8),
            "dy_m": round(dy, 8),
            "dz_m": round(dz, 8),
            "bounds": SR5_TARGET_BOUNDS,
        }
    return item


def ensure_datagen_motion(
    args: argparse.Namespace,
    *,
    strict_start_index: int | None = None,
    strict_move_count: int | None = None,
) -> str | None:
    if args.task_set not in {"100_datagen", "100_independent_datagen"}:
        return None
    doc = yaml.safe_load(DATAGEN_SOURCE_MOTION.read_text(encoding="utf-8")) or {}
    pipeline = doc.get("pipeline")
    if not isinstance(pipeline, list) or len(pipeline) < 2:
        raise ValueError(f"invalid source motion pipeline: {DATAGEN_SOURCE_MOTION}")

    bootstrap = pipeline[0]
    strict_moves = pipeline[1:]
    start_index = int(args.datagen_start_strict_index if strict_start_index is None else strict_start_index)
    if start_index < 0:
        raise ValueError("--datagen-start-strict-index must be >= 0")
    strict_moves = strict_moves[start_index:]
    if strict_move_count is not None:
        strict_moves = strict_moves[: max(0, int(strict_move_count))]
    elif args.datagen_max_strict_moves is not None:
        strict_moves = strict_moves[: max(0, int(args.datagen_max_strict_moves))]

    generated_pipeline: list[dict[str, Any]] = []
    if args.task_set == "100_independent_datagen":
        for offset, move in enumerate(strict_moves):
            generated_pipeline.append(copy.deepcopy(bootstrap))
            generated_pipeline.append(jitter_strict_move(
                move,
                args,
                strict_global_index=start_index + offset,
            ))
        target = DATAGEN_INDEPENDENT_GENERATED_MOTION
    else:
        generated_pipeline = [copy.deepcopy(bootstrap)]
        for offset, move in enumerate(strict_moves):
            generated_pipeline.append(jitter_strict_move(
                move,
                args,
                strict_global_index=start_index + offset,
            ))
        target = DATAGEN_GENERATED_MOTION

    for item in generated_pipeline:
        params = item.get("param")
        if isinstance(params, dict):
            if "sim_delay" in params:
                params["sim_delay"] = float(args.datagen_sim_delay)
            if "sim_time_scale" in params:
                params["sim_time_scale"] = float(args.datagen_sim_time_scale)

    generated_doc = {
        "pipeline": generated_pipeline,
        "metadata": {
            "schema_version": "sr5_datagen_motion.v1",
            "source_motion": str(DATAGEN_SOURCE_MOTION),
            "task_set": args.task_set,
            "strict_start_index": start_index,
            "strict_move_count": len(strict_moves),
            "bootstrap_each_move": args.task_set == "100_independent_datagen",
            "sim_delay": float(args.datagen_sim_delay),
            "sim_time_scale": float(args.datagen_sim_time_scale),
            "position_jitter_m": float(args.datagen_position_jitter_m),
            "z_jitter_m": float(args.datagen_z_jitter_m),
            "jitter_seed": int(args.datagen_jitter_seed),
            "jitter_bounds": SR5_TARGET_BOUNDS,
        },
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        yaml.safe_dump(generated_doc, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return str(target)


def lifecycle_snapshot(root: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    if not root.is_dir():
        return snapshot
    for run_dir in sorted(path for path in root.glob("run_*") if path.is_dir()):
        json_files = sorted(run_dir.glob("*.json"))
        latest_mtime = max((path.stat().st_mtime for path in json_files), default=0.0)
        snapshot[str(run_dir)] = {
            "json_count": len(json_files),
            "latest_mtime": latest_mtime,
            "latest_file": str(json_files[-1]) if json_files else None,
        }
    return snapshot


def snapshot_delta(
    before: dict[str, dict[str, Any]],
    after: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    changed = []
    for run_dir, after_item in sorted(after.items()):
        before_item = before.get(run_dir, {})
        before_count = int(before_item.get("json_count", 0))
        after_count = int(after_item.get("json_count", 0))
        before_mtime = float(before_item.get("latest_mtime", 0.0))
        after_mtime = float(after_item.get("latest_mtime", 0.0))
        if after_count != before_count or after_mtime > before_mtime:
            changed.append(
                {
                    "run_dir": run_dir,
                    "json_count_before": before_count,
                    "json_count_after": after_count,
                    "json_count_delta": after_count - before_count,
                    "latest_file": after_item.get("latest_file"),
                }
            )
    return changed


def wait_for_port(host: str, port: int, timeout_sec: float) -> None:
    deadline = time.monotonic() + float(timeout_sec)
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=2.0):
                return
        except OSError as exc:
            last_error = exc
            time.sleep(1.0)
    raise TimeoutError(f"HTTP port {host}:{port} not ready: {last_error}")


def post_json(url: str, payload: dict[str, Any], timeout_sec: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    started_at = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=float(timeout_sec)) as response:
            text = response.read().decode("utf-8", errors="replace")
            status_code = int(response.status)
    except urllib.error.HTTPError as exc:
        text = exc.read().decode("utf-8", errors="replace")
        status_code = int(exc.code)
    elapsed = time.monotonic() - started_at
    parsed: Any = None
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    return {
        "status_code": status_code,
        "elapsed_sec": round(elapsed, 3),
        "text": text,
        "json": parsed,
    }


def launch_command(args: argparse.Namespace) -> list[str]:
    return [
        "ros2",
        "launch",
        "launch/start.launch.py",
        f"project:={args.project}",
        "use_real_robot:=false",
        "enable_real_feedback:=false",
        "enable_http:=true",
        "enable_rviz:=false",
        "enable_visual_helpers:=false",
        "enable_vision_stack:=false",
        f"robot_profile:={args.robot_profile}",
        f"sr5_task_set:={args.task_set}",
        f"diffusion_seed_mode:={args.diffusion_seed_mode}",
        "diffusion_seed_allow_real_robot_candidate:=false",
    ]


def start_launch(args: argparse.Namespace, launch_log: Path) -> subprocess.Popen | None:
    if args.reuse_existing_launch:
        return None
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    log_handle = launch_log.open("w", encoding="utf-8")
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "2")
    return subprocess.Popen(
        launch_command(args),
        cwd=REPO_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )


def write_summary(path: Path, summary: dict[str, Any]) -> None:
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def terminate_launch(process: subprocess.Popen | None, keep_launch: bool) -> None:
    if process is None or keep_launch:
        return
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=20.0)
    except Exception:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
        except Exception:
            os.killpg(process.pid, signal.SIGKILL)


def main() -> None:
    args = parse_args()
    if args.sweep_strict_moves and args.task_set not in {"100_datagen", "100_independent_datagen"}:
        raise ValueError("--sweep-strict-moves requires a datagen task set")
    if args.sweep_strict_moves and not args.restart_launch_per_run:
        raise ValueError("--sweep-strict-moves requires --restart-launch-per-run")

    generated_motion = None if args.sweep_strict_moves else ensure_datagen_motion(args)
    payload = read_payload(args.payload_json)
    stamp = run_stamp()
    report_dir = args.report_root / f"sr5_lifecycle_collection_{stamp}"
    report_dir.mkdir(parents=True, exist_ok=True)
    launch_log = report_dir / "ros2_launch.log"
    summary_path = report_dir / "collection_summary.json"
    http_url = f"http://{args.http_host}:{int(args.http_port)}/api/cmd"
    command = launch_command(args)

    if args.reuse_existing_launch and args.restart_launch_per_run:
        raise ValueError("--reuse-existing-launch cannot be combined with --restart-launch-per-run")
    if args.keep_launch and args.restart_launch_per_run:
        raise ValueError("--keep-launch cannot be combined with --restart-launch-per-run")

    summary: dict[str, Any] = {
        "schema_version": "diffusion_seed_lifecycle_collection.v1",
        "created_at": utc_now(),
        "repo_root": str(REPO_ROOT),
        "task_set": args.task_set,
        "requested_runs": int(args.runs),
        "http_url": http_url,
        "payload": payload,
        "launch_command": command,
        "launch_log": str(launch_log),
        "lifecycle_root": str(args.lifecycle_root),
        "generated_motion": generated_motion,
        "datagen_sim_delay": args.datagen_sim_delay if generated_motion else None,
        "datagen_sim_time_scale": args.datagen_sim_time_scale if generated_motion else None,
        "datagen_position_jitter_m": float(args.datagen_position_jitter_m),
        "datagen_z_jitter_m": float(args.datagen_z_jitter_m),
        "datagen_jitter_seed": int(args.datagen_jitter_seed),
        "sweep_strict_moves": bool(args.sweep_strict_moves),
        "datagen_start_strict_index": int(args.datagen_start_strict_index),
        "datagen_max_strict_moves": args.datagen_max_strict_moves,
        "restart_launch_per_run": bool(args.restart_launch_per_run),
        "runs": [],
    }

    if args.dry_run:
        summary["dry_run"] = True
        write_summary(summary_path, summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    write_summary(summary_path, summary)
    process = None if args.restart_launch_per_run else start_launch(args, launch_log)
    try:
        for index in range(1, int(args.runs) + 1):
            per_run_process = None
            per_run_generated_motion = generated_motion
            per_run_strict_index = None
            if args.sweep_strict_moves:
                per_run_strict_index = int(args.datagen_start_strict_index) + index - 1
                per_run_generated_motion = ensure_datagen_motion(
                    args,
                    strict_start_index=per_run_strict_index,
                    strict_move_count=1,
                )
            if args.restart_launch_per_run:
                per_run_log = report_dir / f"ros2_launch_run_{index:03d}.log"
                summary["runs"].append({
                    "run_index": index,
                    "launch_log": str(per_run_log),
                    "status": "launching",
                    "started_at": utc_now(),
                    "generated_motion": per_run_generated_motion,
                    "strict_index": per_run_strict_index,
                })
                write_summary(summary_path, summary)
                per_run_process = start_launch(args, per_run_log)
                process = per_run_process
            wait_for_port(args.http_host, args.http_port, args.ready_timeout_sec)
            before = lifecycle_snapshot(args.lifecycle_root)
            payload_for_run = dict(payload)
            payload_for_run["task_id"] = f"{payload.get('task_id', 'SR5_LEVEL_DATA_GEN')}_{index:03d}"
            started_at = utc_now()
            response = post_json(http_url, payload_for_run, args.run_timeout_sec)
            after = lifecycle_snapshot(args.lifecycle_root)
            run_record = {
                "run_index": index,
                "started_at": started_at,
                "finished_at": utc_now(),
                "launch_log": str(report_dir / f"ros2_launch_run_{index:03d}.log")
                if args.restart_launch_per_run else str(launch_log),
                "generated_motion": per_run_generated_motion,
                "strict_index": per_run_strict_index,
                "payload": payload_for_run,
                "response": response,
                "changed_lifecycle_runs": snapshot_delta(before, after),
            }
            if args.restart_launch_per_run:
                summary["runs"][-1] = run_record
            else:
                summary["runs"].append(run_record)
            write_summary(summary_path, summary)
            print(json.dumps(run_record, ensure_ascii=False, indent=2))
            if args.restart_launch_per_run:
                terminate_launch(per_run_process, keep_launch=False)
                process = None
            if index < int(args.runs):
                time.sleep(max(0.0, float(args.between_run_sleep_sec)))
    finally:
        terminate_launch(process, args.keep_launch)

    summary["finished_at"] = utc_now()
    summary["completed_runs"] = len(summary["runs"])
    write_summary(summary_path, summary)
    print(f"summary={summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
