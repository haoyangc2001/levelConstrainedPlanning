#!/usr/bin/env python3
"""Paper table/figure generators (A4.3) over ``paper_result.v1`` rows.

These feed Phase E (paper writing).  They intentionally depend only on the
``paper_result.v1`` contract (``paper_result.py``), never on the raw benchmark
schema, so any method that emits a paper_result row (ours or a Phase B baseline)
plots identically.

Contents
--------
* ``success_at_k_curve`` -- Fig.4 data: Success@K vs K per method with Wilson CI
  error bands (A4.3).  Returns a plot-ready dict; if ``matplotlib`` is present
  and ``--png`` is given it also renders the figure, otherwise it stays a data
  stub (headless clusters have no display / may lack matplotlib).
* ``main_results_table`` -- the headline success/latency/constraint table
  (Markdown), one row per method/cell.

Kept deliberately small: Phase E refines styling; the contract is that the data
extraction is correct and CI-aware now.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from paper_result import load_paper_results


def success_at_k_curve(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Fig.4 data: per-method Success@K vs K with Wilson CI bands (A4.3).

    Groups rows by method, then orders by K.  Each point carries the rate plus
    the Wilson lo/hi so the figure can draw an error band.  Rows are averaged
    over repeat seeds at the same K (simple mean of rates; CI taken as the
    envelope of the per-seed CIs to stay conservative).
    """
    by_method: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        k = row.get("K")
        if k is None:
            continue
        by_method[str(row.get("method"))][int(k)].append(row)

    series: dict[str, Any] = {}
    for method, by_k in by_method.items():
        points = []
        for k in sorted(by_k):
            cells = by_k[k]
            rates = [c.get("success_at_k") for c in cells if c.get("success_at_k") is not None]
            los = [(c.get("success_at_k_ci") or {}).get("wilson_lo") for c in cells]
            his = [(c.get("success_at_k_ci") or {}).get("wilson_hi") for c in cells]
            los = [v for v in los if v is not None]
            his = [v for v in his if v is not None]
            points.append(
                {
                    "K": k,
                    "success_at_k": round(sum(rates) / len(rates), 6) if rates else None,
                    "wilson_lo": min(los) if los else None,
                    "wilson_hi": max(his) if his else None,
                    "n_cells": len(cells),
                }
            )
        series[method] = points
    return {"figure": "success_at_k_curve", "x": "K", "y": "success_at_k", "series": series}


def main_results_table(rows: list[dict[str, Any]]) -> str:
    """Headline Markdown results table (A4.3): success/latency/constraint/safety."""
    header = (
        "| Method | Exp | Class | K | Budget | n | Success | 95% CI | Success@K | "
        "P50 (ms) | P95 (ms) | Align p95 (deg) | Collision rate (cand) | Max jerk p95 |\n"
        "|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n"
    )
    lines = [header]
    for row in rows:
        ci = row.get("success_rate_ci") or {}
        ci_str = (
            f"[{ci.get('wilson_lo'):.3f}, {ci.get('wilson_hi'):.3f}]"
            if ci.get("wilson_lo") is not None and ci.get("wilson_hi") is not None
            else "-"
        )
        ce = row.get("constraint_error") or {}
        align_p95 = (ce.get("selected_alignment_deviation_deg") or {}).get("p95")
        jerk_p95 = (ce.get("max_jerk_rad_s3") or {}).get("p95")
        lat = row.get("latency") or {}
        coll = row.get("collision") or {}

        def _f(value: Any, spec: str = ".3f") -> str:
            return format(value, spec) if isinstance(value, (int, float)) else "-"

        lines.append(
            f"| {row.get('method')} | {row.get('experiment')} | {row.get('constraint_class')} | "
            f"{row.get('K')} | {row.get('budget')} | {row.get('n_problems')} | "
            f"{_f(row.get('success_rate'))} | {ci_str} | {_f(row.get('success_at_k'))} | "
            f"{_f(lat.get('p50'), '.1f')} | {_f(lat.get('p95'), '.1f')} | "
            f"{_f(align_p95, '.2f')} | {_f(coll.get('collision_rate_candidate'))} | {_f(jerk_p95, '.2f')} |"
        )
    return "\n".join(lines) + "\n"


def _render_curve_png(curve: dict[str, Any], png_path: Path) -> bool:
    """Render the Success@K curve to PNG if matplotlib is available (A4.3)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(5, 3.2))
    for method, points in curve["series"].items():
        xs = [p["K"] for p in points]
        ys = [p["success_at_k"] for p in points]
        ax.plot(xs, ys, marker="o", label=method)
        lo = [p["wilson_lo"] for p in points]
        hi = [p["wilson_hi"] for p in points]
        if all(v is not None for v in lo + hi):
            ax.fill_between(xs, lo, hi, alpha=0.15)
    ax.set_xlabel("Compute budget K (seed/solve attempts)")
    ax.set_ylabel("Success@K")
    ax.set_ylim(0.0, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return True


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Generate paper tables/figures from paper_result.v1 JSONL (A4.3).")
    parser.add_argument("results", type=Path, help="paper_result.v1 JSONL")
    parser.add_argument("--out-dir", type=Path, required=True, help="output directory for table/figure artifacts")
    parser.add_argument("--png", action="store_true", help="also render Fig.4 PNG if matplotlib is available")
    args = parser.parse_args()

    rows = load_paper_results(args.results)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    curve = success_at_k_curve(rows)
    (args.out_dir / "fig4_success_at_k.json").write_text(
        json.dumps(curve, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    table = main_results_table(rows)
    (args.out_dir / "table_main_results.md").write_text(table, encoding="utf-8")

    rendered = False
    if args.png:
        rendered = _render_curve_png(curve, args.out_dir / "fig4_success_at_k.png")

    print(
        json.dumps(
            {
                "rows": len(rows),
                "methods": sorted(curve["series"].keys()),
                "fig4_json": str(args.out_dir / "fig4_success_at_k.json"),
                "table_md": str(args.out_dir / "table_main_results.md"),
                "png_rendered": rendered,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())