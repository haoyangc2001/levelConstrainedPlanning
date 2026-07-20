# D7 Geometry And Paper-Result Summary

- source benchmark: `runs/c4_test_eval/result.json`
- paper-result rows: `reports/d7_c4_paper_results.jsonl`
- Fig.4 data: `reports/d7_paper_assets/fig4_success_at_k.json`
- main result table: `reports/d7_paper_assets/table_main_results.md`

## Independent Alignment Check

Artifact: `reports/d7_c4_geometry_verify.json`

- selected trajectory files scanned: `14028`
- successful level-active trajectories checked: `1578`
- false successes under per-request tolerance: `0`
- independent max alignment deviation distribution: min `0.1914 deg`, median `1.6009 deg`, p95 `3.2186 deg`, max `5.3348 deg`

The test split mixes request tolerances (`3`, `8`, and `15` degrees), so the p95 being above `3 deg` does not imply a false success. The pass/fail decision uses each request's own tolerance.

## Independent World-Collision Check

Artifact: `reports/d7_independent_collision_audit.json`

- successful trajectories checked with configured world boxes: `1578`
- independent world-collision findings: `0`
- min sphere-box clearance distribution: min `0.0061527 m`, p05 `0.014847119 m`, median `0.130074556 m`, p95 `0.205089127 m`
- ignored link: `XMS5-R800-W4G3B4C_base`

The ignored base link matches the C4 planner collision gate's dynamic sphere set: the repository sphere file has `64` spheres, while C4 validator records `num_spheres=54`; the difference is the fixed base's `10` spheres. The audit is a world-obstacle check only and does not claim self-collision/FCL replacement.

## Paper Implication

D7 supports using C4 as a traceable paper artifact for the fallback narrative:

- Geometry self-confirmation risk is reduced for alignment and world-obstacle collision on C4 selected successes.
- Fig.4/Table data are generated from `paper_result.v1`, but only at the surviving C4 budget point (`K=6`, compute budget `6`) because D1 HOLD descoped the full D3/D4/D5 recompute.
- E2/E3 should be written as negative/diagnostic evidence and an initial baseline, not as learned superiority.
