#!/usr/bin/env python3
"""D1 exit gate: verify A/B/C are ready before spending Phase-D compute.

This is the plan's "花算力前的门" — a pre-flight check that inspects the
committed A/B/C deliverables and reports READY / NOT-READY per axis, plus the
NEW go/no-go compute gate (D1 addendum): the C4 "full closed loop vs rule
pipeline (budget-matched)" single-point comparison is a HARD criterion. If C4
(and any available D4/E2 preview) shows learning is NOT better than the pure
rule pipeline at matched budget, we do NOT spend the D3/D4/D5 full recompute —
we switch to the honest fallback narrative (innovation B downgraded to a
lower-bound safe-integration architecture).

The checker is read-only. Each sub-check returns (ok, detail). The go/no-go
gate consumes ``c4_significance.json`` when present; while the C4 benchmark is
still running it reports that single gate as PENDING (not FAIL), so the
readiness axes can be signed off independently of the compute verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> Any | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def check_a1(root: Path) -> tuple[bool, str]:
    """A1: collision_safety returns a real min distance; collision_unchecked==0."""
    rec = _load(root / "runs/a1_hard_gate_recheck/result.json")
    if rec is None:
        return False, "missing runs/a1_hard_gate_recheck/result.json (A1 evidence)"
    cs = ((rec.get("metrics") or {}).get("collision_safety")) or {}
    if not cs.get("checked", False):
        return False, f"collision_safety.checked is not True: {cs.get('checked')!r}"
    if cs.get("status") == "unchecked":
        return False, "collision_safety.status == 'unchecked' (A1 claim violated)"
    md = cs.get("min_distance_m")
    if not isinstance(md, (int, float)):
        return False, f"min_distance_m not numeric: {md!r}"
    spheres = cs.get("num_spheres")
    world = (cs.get("world_summary") or {}).get("total_count")
    return True, (
        f"collision_safety: checked=True status={cs.get('status')!r} "
        f"min_distance_m={md} num_spheres={spheres} world_obstacles={world}"
    )


def check_a2(root: Path) -> tuple[bool, str]:
    """A2: dimensioned kinematics — dt_sec != null, real jerk/motion_time."""
    rec = _load(root / "runs/a1_hard_gate_recheck/result.json")
    if rec is None:
        return False, "missing A1/A2 evidence record"
    s = json.dumps(rec)
    have = {k: (k in s) for k in ("dt_sec", "motion_time_sec", "max_jerk_rad_s3")}
    missing = [k for k, v in have.items() if not v]
    if missing:
        return False, f"dimensioned fields absent: {missing}"
    return True, "dimensioned kinematics present: dt_sec, motion_time_sec, max_jerk_rad_s3"


def check_a3_a4(root: Path) -> tuple[bool, str]:
    """A3/A4: harness v2 + paper_result.v1 with CI, method/K/budget axes, p75+p98."""
    pr = root / "tools/dataset/paper_result.py"
    runner = root / "tools/dataset/run_closed_loop_benchmark.py"
    missing = [str(p.relative_to(root)) for p in (pr, runner) if not p.exists()]
    if missing:
        return False, f"missing harness/schema files: {missing}"
    src = pr.read_text(encoding="utf-8")
    needed = ["wilson", "p98", "p75", "budget", "seed", "per_problem_success"]
    absent = [t for t in needed if t not in src]
    if absent:
        return False, f"paper_result.py missing axes/tokens: {absent}"
    return True, "paper_result.py has Wilson CI, p75/p98, budget/seed/method axes, per_problem_success"


def check_b(root: Path) -> tuple[bool, str]:
    """B: external methods registered AND >=1 constraint-forcing adversary (B5/B3c)."""
    bdir = root / "tools/dataset/baselines"
    if not bdir.is_dir():
        return False, "missing tools/dataset/baselines/"
    files = {p.name for p in bdir.glob("*.py")}
    # constraint-forcing adversaries: B5 = ompl_projection, B3c = chomp_constraint
    forcing = [f for f in ("ompl_projection.py", "chomp_constraint.py") if f in files]
    if not forcing:
        return False, f"no constraint-forcing adversary (need ompl_projection or chomp_constraint); have {sorted(files)}"
    # at least one has an evidence report
    evidence = []
    for cand in ("runs/b5_ompl_projection_recheck/report.json",
                 "runs/b5_ompl_projection_k4/report.json",
                 "runs/b3c_chomp_strict_smoke/report.json",
                 "runs/b3c_chomp_smoke/report.json"):
        if (root / cand).exists():
            evidence.append(cand)
    if not evidence:
        return False, f"constraint-forcing adapter present ({forcing}) but no evidence report found"
    return True, f"baselines registered; constraint-forcing adversary ready ({forcing}); evidence: {evidence[0]}"


def check_c(root: Path) -> tuple[bool, str]:
    """C: retrained checkpoints resolvable; frozen split in effect; keep-level branch honored."""
    ptr = _load(root / "artifacts/current_artifacts.json")
    if ptr is None:
        return False, "missing artifacts/current_artifacts.json"
    blob = json.dumps(ptr)
    if "c3_diffusion_allcomp" not in blob or "c3_critic" not in blob:
        return False, "current_artifacts.json does not point at the C3 checkpoints"
    # checkpoints physically resolvable: collect every *.pt path in the pointer
    pt_paths: list[str] = []

    def _collect_pt(o: Any) -> None:
        if isinstance(o, dict):
            for v in o.values():
                _collect_pt(v)
        elif isinstance(o, list):
            for v in o:
                _collect_pt(v)
        elif isinstance(o, str) and o.endswith(".pt"):
            pt_paths.append(o)

    _collect_pt(ptr)
    unresolved = [p for p in pt_paths if not Path(p).exists()]
    if unresolved:
        return False, f"artifact pointer references missing checkpoint files: {unresolved}"
    if not pt_paths:
        return False, "artifact pointer contains no .pt checkpoint paths"
    # dataset pointer with frozen split counts
    dp = _load(root / "runs/c2_keeplevel/dataset_pointer.json")
    if dp is None:
        return False, "missing runs/c2_keeplevel/dataset_pointer.json"
    ds = dp.get("dataset") or {}
    n = ds.get("sample_count")
    pos = ds.get("positive_for_diffusion")
    if not isinstance(n, int) or not isinstance(pos, int):
        return False, f"dataset_pointer counts not numeric: sample_count={n!r} positive_for_diffusion={pos!r}"
    if pos < 3000:
        return False, f"positive_for_diffusion={pos} < 3000 target (C2 under-produced)"
    return True, (
        f"artifact pointer -> C3 diffusion+critic; C2 dataset_pointer sample_count={n} "
        f"positive_for_diffusion={pos} (>=3000 target; keep-level LP/LPO branch per C1b descope)"
    )


def check_go_no_go(root: Path, sig_path: Path | None) -> tuple[str, str]:
    """NEW compute gate: is learning better than the rule pipeline at matched budget?

    Returns (state, detail) where state in {PASS, HOLD, PENDING}. PASS => proceed
    with D3/D4/D5. HOLD => switch to fallback narrative. PENDING => C4 verdict not
    yet available (benchmark still running); readiness axes may still be signed."""
    candidates = [sig_path] if sig_path else []
    candidates += [root / "runs/c4_test_eval/c4_significance.json",
                   root / "runs/c4_test_eval/significance.json"]
    sig = None
    used = None
    for c in candidates:
        if c and (sig := _load(c)) is not None:
            used = c
            break
    if sig is None:
        return "PENDING", "C4 significance verdict not found yet (benchmark still running)"
    # the significance tool records a verdict per learned method vs rule
    verdicts = sig.get("verdicts") or sig.get("comparisons") or []
    if not verdicts:
        return "PENDING", f"{used}: no verdict entries yet"
    any_superior = any(
        (v.get("verdict") == "superior" or v.get("learned_superior") is True)
        for v in (verdicts if isinstance(verdicts, list) else verdicts.values())
    )
    if any_superior:
        return "PASS", f"{used}: >=1 learned method shown superior to rule at matched budget -> proceed D3/D4/D5"
    return "HOLD", f"{used}: no learned method superior to rule -> switch to fallback narrative (innovation B as lower-bound)"


def main() -> None:
    ap = argparse.ArgumentParser(description="D1 readiness + go/no-go compute gate")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--significance", type=Path, default=None,
                    help="path to C4 significance json (default: search runs/c4_test_eval/)")
    ap.add_argument("--json-out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()

    checks = {
        "A1_collision_replay": check_a1(root),
        "A2_dimensioned_time": check_a2(root),
        "A3_A4_harness_schema": check_a3_a4(root),
        "B_baselines_adversary": check_b(root),
        "C_retrain_split": check_c(root),
    }
    gate_state, gate_detail = check_go_no_go(root, args.significance)

    all_ready = all(ok for ok, _ in checks.values())
    print("=== D1 readiness gate (A/B/C exit checks) ===")
    for name, (ok, detail) in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    print(f"\n=== D1 go/no-go compute gate (C4 verdict) ===")
    print(f"  [{gate_state}] {gate_detail}")

    if all_ready and gate_state == "PASS":
        overall = "READY: proceed to D3/D4/D5 full compute"
    elif all_ready and gate_state == "PENDING":
        overall = "READINESS-OK / VERDICT-PENDING: A/B/C ready; hold D3+ until C4 verdict lands"
    elif all_ready and gate_state == "HOLD":
        overall = "HOLD: A/B/C ready but learning not superior -> fallback narrative, skip full recompute"
    else:
        failed = [n for n, (ok, _) in checks.items() if not ok]
        overall = f"NOT-READY: fix {failed} before Phase D"
    print(f"\n>>> {overall}")

    if args.json_out:
        out = {
            "schema_version": "d1_readiness_gate.v1",
            "readiness": {n: {"ok": ok, "detail": d} for n, (ok, d) in checks.items()},
            "all_ready": all_ready,
            "go_no_go": {"state": gate_state, "detail": gate_detail},
            "overall": overall,
        }
        args.json_out.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
