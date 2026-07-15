# Seed Repair Phase 4 Smoke Report

Date: 2026-07-15

Phase 4 connected external trajectory seeds to CuRobo repair through `SeedRepairAdapter`. Rule, file-backed diffusion, mixed and shadow modes were checked in headless mode.

## Smoke Matrix

| run | mode | external repaired candidates | raw parent candidates | selected | dataset |
| --- | --- | ---: | ---: | --- | --- |
| `phase4_seed_repair_smoke` | rule candidate | 1 rule | 1 rule raw | planner native | valid, 1 positive + 2 negative |
| `phase4_diffusion_file_repair_smoke` | diffusion candidate | 1 diffusion, 1 rule | 1 diffusion raw, 1 rule raw | planner native | valid, 1 positive + 4 negative |
| `phase4_shadow_smoke` | diffusion shadow | 0 | 1 diffusion raw | planner native | raw seed did not enter repair |
| `phase4_mixed_candidate_smoke` | mixed candidate | 1 rule, 1 diffusion | 1 rule raw, 1 diffusion raw | planner native | valid, 1 positive + 4 negative |

The rule and diffusion external seeds entered the repair pool and preserved parent lineage. In the current smoke requests, both external repaired candidates returned `trajopt_failed`, so they are recorded as negative critic samples. This is acceptable for Phase 4 because the phase goal is pool admission, repair execution, lineage, and failure recording. Improving external seed success rate is handled by Phase 5-8 model/critic/data-loop work.

## Artifact Semantics

- Raw provider candidates use source types such as `rule_raw_seed` and `diffusion_seed`, with `candidate_status=raw_seed_parent_only`.
- Repaired candidates use ids like `rule_seed_00_repaired` and `diffusion_seed_00_repaired`, and preserve `parent_candidate_id`.
- Shadow candidates remain raw-only and do not produce repaired candidate ids.
- Planner native candidates remain in the same candidate pool and can still be selected.

Large run artifacts remain under `runs/` and are not committed.
