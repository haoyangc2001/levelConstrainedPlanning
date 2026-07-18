#!/usr/bin/env python3
"""C4 independent geometry verification (anti-self-confirmation).

The closed-loop benchmark accepts a trajectory when the planner's OWN validator
says the alignment constraint holds. That is circular: the same code that scores
candidates also gates success. This tool re-checks the SELECTED trajectory of
every claimed success with an INDEPENDENT alignment computation:

* joint waypoints are read from ``selected_trajectory.json`` (planner output),
* FK is run through cuRobo ``compute_kinematics`` to get the EE quaternion,
* the alignment angle is recomputed HERE with from-scratch quaternion->rotation
  ->axis math (numpy), NOT via ``level_planner_core.constraints`` — so a bug in
  the planner's alignment code cannot mask itself,
* the per-request axes + tolerance come from ``request.json`` (not hard-coded).

For every success dir we report whether the independent max alignment deviation
is within tolerance. A "success" whose selected trajectory violates the axis
constraint under this independent check is a FALSE SUCCESS and is listed
explicitly. Level-inactive requests (goal_orientation-only, level_active=False)
are skipped for the alignment gate (nothing to verify on that axis).

Sampling: verifying all ~thousands of success dirs is FK-cheap but IO-heavy;
``--max-dirs`` caps the audit to a random-free prefix (deterministic order).
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


def quat_to_axis_world(qw: float, qx: float, qy: float, qz: float, local_axis: np.ndarray) -> np.ndarray:
    """Rotate local_axis by quaternion (qw,qx,qy,qz) using an independent R build."""
    # normalize
    n = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz) or 1.0
    qw, qx, qy, qz = qw / n, qx / n, qy / n, qz / n
    # rotation matrix from quaternion (standard, written from scratch here)
    R = np.array([
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw),     2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw),     1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw),     2 * (qy * qz + qx * qw),     1 - 2 * (qx * qx + qy * qy)],
    ])
    return R @ local_axis


def alignment_angle_deg(axis_world: np.ndarray, target_world_axis: np.ndarray) -> float:
    a = axis_world / (np.linalg.norm(axis_world) or 1.0)
    t = target_world_axis / (np.linalg.norm(target_world_axis) or 1.0)
    dot = float(np.clip(np.dot(a, t), -1.0, 1.0))
    return math.degrees(math.acos(dot))


def verify_dir(planner, dir_path: Path) -> dict[str, Any] | None:
    """Independent alignment check for one success dir. None if not applicable."""
    sel_p = dir_path / "selected_trajectory.json"
    req_p = dir_path / "request.json"
    if not sel_p.exists() or not req_p.exists():
        return None
    sel = json.loads(sel_p.read_text(encoding="utf-8"))
    if sel.get("status") != "success":
        return None
    traj = sel.get("trajectory") or []
    if not traj:
        return None
    req = json.loads(req_p.read_text(encoding="utf-8"))
    axes = req.get("constraint_axes") or {}
    if not bool(axes.get("level_active", True)):
        return {"request_id": sel.get("request_id"), "skipped": "level_inactive"}
    align = req.get("alignment") or {}
    local_axis = np.array(align.get("local_axis", [0.0, 1.0, 0.0]), dtype=float)
    target_axis = np.array(align.get("target_world_axis", [0.0, 0.0, -1.0]), dtype=float)
    tol = float(align.get("tolerance_deg", 3.0))

    q = torch.tensor(traj, dtype=torch.float32, device=planner.device)
    kin = planner._constraint_eval_kinematics_fn(q)  # FK only; angle math is ours
    ee_quat = kin.ee_quaternion.detach().cpu().numpy()  # [T,4] (qw,qx,qy,qz)
    devs = [
        alignment_angle_deg(quat_to_axis_world(*ee_quat[i].tolist(), local_axis), target_axis)
        for i in range(ee_quat.shape[0])
    ]
    max_dev = max(devs) if devs else None
    return {
        "request_id": sel.get("request_id"),
        "constraint_class": req.get("constraint_class"),
        "tolerance_deg": tol,
        "independent_max_alignment_deg": max_dev,
        "within_tolerance": bool(max_dev is not None and max_dev <= tol + 1e-6),
        "n_waypoints": len(traj),
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="C4 independent alignment geometry verification.")
    parser.add_argument("eval_dir", type=Path, help="runs/c4_test_eval root (contains per-cell dirs)")
    parser.add_argument("--config", default="configs/sr5_level.yaml")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-dirs", type=int, default=0, help="0 = all success dirs")
    args = parser.parse_args()

    from level_planner_core.planner import LevelConstrainedPlanner
    planner = LevelConstrainedPlanner.from_config(args.config)

    # gather success dirs deterministically (sorted); each contains selected_trajectory.json
    dirs = sorted(p.parent for p in args.eval_dir.glob("*/*/selected_trajectory.json"))
    if args.max_dirs > 0:
        dirs = dirs[: args.max_dirs]

    checked, skipped, false_success = [], [], []
    for d in dirs:
        try:
            res = verify_dir(planner, d)
        except Exception as exc:  # a bad dir must not abort the audit
            checked.append({"dir": str(d), "error": str(exc)})
            continue
        if res is None:
            continue
        if res.get("skipped"):
            skipped.append(res)
            continue
        res["dir"] = str(d)
        checked.append(res)
        if not res["within_tolerance"]:
            false_success.append(res)

    devs_all = [c["independent_max_alignment_deg"] for c in checked
                if c.get("independent_max_alignment_deg") is not None]
    devs_sorted = sorted(devs_all)

    def _pct(p: float) -> float | None:
        if not devs_sorted:
            return None
        idx = min(len(devs_sorted) - 1, int(math.ceil(p / 100.0 * len(devs_sorted)) - 1))
        return devs_sorted[max(0, idx)]

    summary = {
        "schema_version": "c4_geometry_verify.v1",
        "eval_dir": str(args.eval_dir),
        "config": args.config,
        "n_success_dirs_scanned": len(dirs),
        "n_checked_level_active": len([c for c in checked if "independent_max_alignment_deg" in c]),
        "n_skipped_level_inactive": len(skipped),
        "n_false_success": len(false_success),
        "independent_check": "cuRobo FK + from-scratch quaternion->axis angle (NOT planner validator)",
        "independent_max_alignment_deg_distribution": {
            "min": (devs_sorted[0] if devs_sorted else None),
            "median": _pct(50),
            "p95": _pct(95),
            "max": (devs_sorted[-1] if devs_sorted else None),
        },
        "false_success": false_success,
        "checked": checked,
    }
    args.out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[C4-geom] scanned={len(dirs)} checked={summary['n_checked_level_active']} "
          f"skipped_inactive={len(skipped)} FALSE_SUCCESS={len(false_success)}")
    if false_success:
        for f in false_success[:10]:
            print(f"  FALSE: {f['request_id']} max_dev={f['independent_max_alignment_deg']:.3f} "
                  f"tol={f['tolerance_deg']}")
    print(f"[C4-geom] written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
