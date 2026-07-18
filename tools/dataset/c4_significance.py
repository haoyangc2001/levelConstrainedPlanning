#!/usr/bin/env python3
"""C4 significance analysis: paired McNemar + Wilson CI go/no-go verdict.

Consumes a ``closed_loop_curobo_benchmark`` report (the C4 test-split run) and
emits, for every learned method vs the ``rule_only`` reference:

* per-problem paired contingency (b = learned-win-only, c = rule-win-only),
* exact McNemar p-value (binomial on discordant pairs b,c — the exact test,
  valid for small discordant counts and needs no statsmodels),
* success-rate delta (learned - rule) with a Wilson CI on each rate,
* a PRE-REGISTERED go/no-go verdict.

PRE-REGISTERED DECISION RULE (fixed before inspecting the full test results, so
the positive/negative call is not post-hoc):

    A learned method is declared SUPERIOR-TO-RULE iff ALL hold:
      1. paired success-rate delta (learned - rule) > 0, AND
      2. exact McNemar p < ALPHA (default 0.05), AND
      3. the two Wilson success-rate CIs are disjoint (lower bound of the
         higher rate exceeds the upper bound of the lower rate).
    Otherwise the method is recorded as NOT-SHOWN-SUPERIOR (honest negative),
    never silently reworded — this directly answers the plan's anti-arbitrary
    C4 verdict requirement.

Results are aggregated across statistical-repeat seeds by taking, per problem,
the majority-success bit (>= ceil(n_seeds/2) successes) so each problem
contributes ONE paired outcome — this keeps the McNemar pairing at the
problem level (the unit the split freezes), not the seed level.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from scipy import stats

ALPHA_DEFAULT = 0.05
REFERENCE_METHOD = "rule_only"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None, "point": None, "z": z}
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    half = (z * math.sqrt(phat * (1 - phat) / total + z * z / (4 * total * total))) / denom
    return {"low": max(0.0, center - half), "high": min(1.0, center + half),
            "point": phat, "z": z}


def exact_mcnemar(b: int, c: int) -> float:
    """Exact (binomial) McNemar p-value on discordant pairs b, c.

    Two-sided: P(X <= min(b,c)) under Binom(b+c, 0.5), doubled and clamped."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # two-sided exact binomial test with p=0.5
    return float(min(1.0, 2.0 * stats.binom.cdf(k, n, 0.5)))


def _majority_bits(bits_by_seed: list[dict[str, bool]], n_seeds: int) -> dict[str, bool]:
    """Collapse per-seed success bits into one majority bit per request_id."""
    need = math.ceil(n_seeds / 2) if n_seeds > 0 else 1
    counts: dict[str, int] = defaultdict(int)
    for seed_bits in bits_by_seed:
        for rid, ok in seed_bits.items():
            counts[rid] += 1 if ok else 0
    return {rid: (cnt >= need) for rid, cnt in counts.items()}


def _collect(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """method -> {seeds: set, bits_by_seed: [ {rid: bool} ], n: int}."""
    summaries = report.get("summaries") or []
    by_method: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"seeds": set(), "bits_by_seed": [], "rate_num": 0, "rate_den": 0}
    )
    for s in summaries:
        method = str(s.get("method") or s.get("strategy") or "unknown")
        seed = s.get("repeat_seed")
        bits = ((s.get("per_problem_success_bits") or {}).get("final")) or []
        seed_map = {str(b.get("request_id")): bool(b.get("success")) for b in bits}
        m = by_method[method]
        m["seeds"].add(seed)
        m["bits_by_seed"].append(seed_map)
    return by_method


def analyze(report: dict[str, Any], alpha: float = ALPHA_DEFAULT) -> dict[str, Any]:
    by_method = _collect(report)
    if REFERENCE_METHOD not in by_method:
        raise SystemExit(f"reference method {REFERENCE_METHOD!r} absent from report")

    ref = by_method[REFERENCE_METHOD]
    ref_n_seeds = len(ref["seeds"])
    ref_bits = _majority_bits(ref["bits_by_seed"], ref_n_seeds)

    verdicts = []
    for method, m in by_method.items():
        if method == REFERENCE_METHOD:
            continue
        n_seeds = len(m["seeds"])
        learned_bits = _majority_bits(m["bits_by_seed"], n_seeds)
        # pair only on request_ids present for BOTH (strip the per-method suffix
        # so rule_only vs diffusion_only pair on the same underlying problem).
        def base_id(rid: str, method_name: str) -> str:
            suffix = f"_{method_name}"
            return rid[: -len(suffix)] if rid.endswith(suffix) else rid
        ref_by_base = {base_id(r, REFERENCE_METHOD): ok for r, ok in ref_bits.items()}
        learned_by_base = {base_id(r, method): ok for r, ok in learned_bits.items()}
        shared = sorted(set(ref_by_base) & set(learned_by_base))
        a = b = c = d = 0  # a: both win, b: learned-only, c: rule-only, d: both fail
        for rid in shared:
            lw, rw = learned_by_base[rid], ref_by_base[rid]
            if lw and rw: a += 1
            elif lw and not rw: b += 1
            elif (not lw) and rw: c += 1
            else: d += 1
        n = len(shared)
        learned_succ = a + b
        rule_succ = a + c
        p = exact_mcnemar(b, c)
        ci_learned = wilson_interval(learned_succ, n)
        ci_rule = wilson_interval(rule_succ, n)
        delta = (learned_succ - rule_succ) / n if n else 0.0
        cis_disjoint = bool(
            n and ci_learned["low"] is not None and ci_rule["high"] is not None
            and (
                (delta > 0 and ci_learned["low"] > ci_rule["high"])
                or (delta < 0 and ci_rule["low"] > ci_learned["high"])
            )
        )
        superior = bool(delta > 0 and p < alpha and cis_disjoint)
        verdicts.append({
            "method": method,
            "reference": REFERENCE_METHOD,
            "n_paired_problems": n,
            "n_seeds": n_seeds,
            "contingency": {"both_win": a, "learned_only": b, "rule_only": c, "both_fail": d},
            "learned_success_rate": learned_succ / n if n else None,
            "rule_success_rate": rule_succ / n if n else None,
            "success_rate_delta": delta,
            "learned_wilson_ci": ci_learned,
            "rule_wilson_ci": ci_rule,
            "mcnemar_exact_p": p,
            "wilson_cis_disjoint": cis_disjoint,
            "alpha": alpha,
            "verdict": "SUPERIOR_TO_RULE" if superior else "NOT_SHOWN_SUPERIOR",
        })

    verdicts.sort(key=lambda v: v["method"])
    return {
        "schema_version": "c4_significance.v1",
        "reference_method": REFERENCE_METHOD,
        "alpha": alpha,
        "eval_split": report.get("eval_split"),
        "n_problems_reference": len(ref_bits),
        "decision_rule": (
            "SUPERIOR_TO_RULE iff delta>0 AND exact-McNemar p<alpha AND "
            "Wilson success-rate CIs disjoint; else NOT_SHOWN_SUPERIOR "
            "(pre-registered, applied uniformly)."
        ),
        "verdicts": verdicts,
    }


def _main() -> int:
    parser = argparse.ArgumentParser(description="C4 paired McNemar + Wilson go/no-go analysis.")
    parser.add_argument("report", type=Path, help="closed_loop benchmark JSON report")
    parser.add_argument("--out", type=Path, required=True, help="output JSON verdict path")
    parser.add_argument("--alpha", type=float, default=ALPHA_DEFAULT)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    result = analyze(report, alpha=args.alpha)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # human-readable stderr summary
    print(f"[C4] eval_split={result['eval_split']} ref={REFERENCE_METHOD} alpha={args.alpha}")
    for v in result["verdicts"]:
        print(
            f"  {v['method']:18} n={v['n_paired_problems']:4} "
            f"learned={v['learned_success_rate']:.3f} rule={v['rule_success_rate']:.3f} "
            f"delta={v['success_rate_delta']:+.3f} McNemar_p={v['mcnemar_exact_p']:.4g} "
            f"-> {v['verdict']}"
        )
    print(f"[C4] verdict written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
