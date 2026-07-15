#!/usr/bin/env python3
"""Headless validation matrix for the standalone SR5 planner."""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _run(cmd: list[str], *, check: bool = True, timeout: int | None = None) -> dict[str, Any]:
    started = time.time()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    item = {
        "cmd": cmd,
        "returncode": proc.returncode,
        "elapsed_sec": round(time.time() - started, 6),
        "stdout_tail": proc.stdout.strip().splitlines()[-12:],
    }
    if check and proc.returncode != 0:
        raise RuntimeError(json.dumps(item, ensure_ascii=False, indent=2))
    return item


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_request(path: Path) -> dict[str, Any]:
    return _read_json(path)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _static_checks(report: dict[str, Any]) -> None:
    import level_planner_core  # noqa: F401

    report["static"] = {
        "compileall": _run([
            sys.executable,
            "-m",
            "compileall",
            "level_planner_core",
            "level_planner",
            "level_planner_cli",
            "level_planner_ros",
            "tools",
            "tests",
            "launch",
        ]),
        "artifact_json": _run([
            sys.executable,
            "-m",
            "json.tool",
            "artifacts/current_artifacts.json",
        ]),
        "asset_check": _run([
            sys.executable,
            "tools/check_assets.py",
            "--config",
            "configs/sr5_level.yaml",
        ]),
        "artifact_check": _run([
            sys.executable,
            "tools/check_artifacts.py",
            "--strict",
        ]),
        "config_yaml": bool(yaml.safe_load((REPO_ROOT / "configs/sr5_level.yaml").read_text(encoding="utf-8"))),
    }
    deny = subprocess.run(
        [
            "rg",
            "-n",
            r"\b(rclpy|motion|external_comm|state_machine|trajectory_msgs|geometry_msgs|sensor_msgs)\b",
            "level_planner_core",
        ],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    report["static"]["core_forbidden_import_scan"] = {
        "returncode": deny.returncode,
        "matches": deny.stdout.strip().splitlines(),
        "ok": deny.returncode == 1,
    }
    if deny.returncode not in (0, 1) or report["static"]["core_forbidden_import_scan"]["matches"]:
        raise RuntimeError("core forbidden import scan failed")


def _cli_single(report: dict[str, Any]) -> None:
    out_dir = REPO_ROOT / "runs/headless_matrix/cli_single"
    _run([
        sys.executable,
        "-m",
        "level_planner.cli",
        "plan",
        "--config",
        "configs/sr5_level.yaml",
        "--request",
        "examples/requests/request_level_001.json",
        "--out",
        str(out_dir),
    ])
    result = _read_json(out_dir / "result.json")
    if result.get("status") != "success" or not result.get("selected_trajectory"):
        raise RuntimeError(f"CLI single request failed: {result.get('status')}")
    report["cli_single"] = {
        "status": result.get("status"),
        "trajectory_points": len(result.get("selected_trajectory") or []),
        "alignment": result.get("metrics", {}).get("alignment", {}),
        "artifacts": result.get("artifacts", {}),
    }


def _batch_requests(report: dict[str, Any]) -> None:
    from level_planner_core import LevelConstrainedPlanner

    base_success = _load_request(REPO_ROOT / "examples/requests/request_level_001.json")
    alignment_hard = _load_request(REPO_ROOT / "examples/requests/request_level_alignment_hard.json")
    planner_fail = _load_request(REPO_ROOT / "examples/requests/request_level_planner_fail.json")

    diffusion_shadow = copy.deepcopy(base_success)
    diffusion_shadow["request_id"] = "request_level_001_diffusion_shadow"
    diffusion_shadow.setdefault("seed_policy", {})["mode"] = "diffusion"
    diffusion_shadow["seed_policy"]["k_generate"] = 2
    diffusion_shadow["seed_policy"]["k_accept"] = 1

    mixed_shadow = copy.deepcopy(base_success)
    mixed_shadow["request_id"] = "request_level_001_mixed_shadow"
    mixed_shadow.setdefault("seed_policy", {})["mode"] = "mixed"
    mixed_shadow["seed_policy"]["k_generate"] = 2
    mixed_shadow["seed_policy"]["k_accept"] = 1

    requests = [
        ("success", base_success),
        ("alignment_hard", alignment_hard),
        ("planner_fail", planner_fail),
        ("diffusion_shadow", diffusion_shadow),
        ("mixed_shadow", mixed_shadow),
    ]
    planner = LevelConstrainedPlanner.from_config("configs/sr5_level.yaml")
    rows = []
    for name, request in requests:
        out_dir = REPO_ROOT / "runs/headless_matrix/batch" / name
        result = planner.plan(request, out_dir=out_dir)
        rows.append({
            "name": name,
            "request_id": result.get("request_id"),
            "status": result.get("status"),
            "failure_reason": result.get("failure_reason"),
            "trajectory_points": len(result.get("selected_trajectory") or []),
            "result_json": result.get("artifacts", {}).get("result_json"),
            "seed_reports": [
                {
                    "provider": item.get("provider") or item.get("provider_name"),
                    "mode": item.get("mode"),
                    "status": item.get("status"),
                    "generated_count": item.get("generated_count"),
                    "runtime_effect": item.get("runtime_effect"),
                }
                for item in result.get("seed_provider_reports", [])
            ],
        })
    summary = {
        "request_count": len(rows),
        "status_counts": dict(Counter(row["status"] for row in rows)),
        "rows": rows,
    }
    _write_json(REPO_ROOT / "runs/headless_matrix/aggregate_summary.json", summary)
    report["batch"] = summary
    expected = {"success", "failed_alignment_constraint", "failed_planner"}
    if not expected.issubset(set(summary["status_counts"])):
        raise RuntimeError(f"batch did not cover expected statuses: {summary['status_counts']}")


def _diffusion_smoke(report: dict[str, Any]) -> None:
    generated = "runs/headless_matrix/diffusion/generated_samples_smoke.json"
    eval_json = "runs/headless_matrix/diffusion/offline_generation_report.json"
    eval_md = "runs/headless_matrix/diffusion/diffusion_vs_rule_seed_report.md"
    _run([
        sys.executable,
        "tools/learning/diffusion_seed_learning/sample.py",
        "--tasks",
        "1",
        "--k",
        "2",
        "--out",
        generated,
    ])
    _run([
        sys.executable,
        "tools/learning/diffusion_seed_learning/evaluate.py",
        "--generated",
        generated,
        "--json-out",
        eval_json,
        "--md-out",
        eval_md,
    ])
    eval_report = _read_json(REPO_ROOT / eval_json)
    if eval_report["diffusion_seed"]["valid_count"] < 1:
        raise RuntimeError("diffusion smoke produced no valid seeds")
    report["diffusion_smoke"] = eval_report


def _ros_check(report: dict[str, Any], skip_ros: bool) -> None:
    if skip_ros:
        report["ros_adapter"] = {"skipped": True}
        return
    build = _run(["colcon", "build", "--symlink-install", "--packages-select", "level_planner_ros"], timeout=120)
    run = subprocess.run(
        "source install/setup.bash && ros2 run level_planner_ros planner_node --check",
        cwd=REPO_ROOT,
        shell=True,
        executable="/bin/bash",
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=60,
    )
    report["ros_adapter"] = {
        "build": build,
        "run_check": {
            "returncode": run.returncode,
            "stdout_tail": run.stdout.strip().splitlines()[-12:],
        },
    }
    if run.returncode != 0:
        raise RuntimeError("ros2 run planner_node --check failed")


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Headless Validation Matrix",
        "",
        f"- overall: `{report['overall']}`",
        f"- cli single: `{report['cli_single']['status']}`",
        f"- batch status counts: `{report['batch']['status_counts']}`",
        f"- diffusion valid ratio: `{report['diffusion_smoke']['diffusion_seed']['valid_ratio']}`",
        f"- ros adapter skipped: `{report['ros_adapter'].get('skipped', False)}`",
        "",
        "All generated run artifacts are under `runs/` and are ignored by git.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json-out", type=Path, default=Path("reports/headless_validation_matrix.json"))
    parser.add_argument("--md-out", type=Path, default=Path("reports/headless_validation_matrix.md"))
    parser.add_argument("--skip-ros", action="store_true")
    args = parser.parse_args(argv)

    report: dict[str, Any] = {
        "schema_version": "standalone_level_planning.headless_validation_matrix.v1",
        "overall": "running",
    }
    try:
        _static_checks(report)
        _cli_single(report)
        _batch_requests(report)
        _diffusion_smoke(report)
        _ros_check(report, skip_ros=args.skip_ros)
        report["overall"] = "success"
    except Exception as exc:
        report["overall"] = "failed"
        report["failure_reason"] = f"{type(exc).__name__}: {exc}"
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        _write_json(args.json_out, report)
        raise
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    _write_json(args.json_out, report)
    _write_markdown(report, args.md_out)
    print(json.dumps({
        "overall": report["overall"],
        "json_out": str(args.json_out),
        "md_out": str(args.md_out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
