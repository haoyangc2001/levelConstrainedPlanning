# Rule Seed Phase 3 Smoke Report

Date: 2026-07-15

Phase 3 moved rule-based horizontal seed generation into the standalone core as raw seed candidates. These candidates are recorded for the closed-loop dataset, but they intentionally do not enter CuRobo repair until Phase 4.

## Smoke Runs

| run | planner status | rule raw candidates | candidate shapes | dataset result |
| --- | --- | ---: | --- | --- |
| `phase3_rule_seed_variants_smoke` | `success` | 3 | `30x6` raw, smoothed, bridged | valid, 1 positive native + 3 negative rule raw |
| `phase3_rule_seed_planner_fail_smoke` | `failed_planner` | 3 | `30x6` raw, smoothed, bridged | valid, 4 negative samples |

The planner-fail smoke confirms the desired Phase 3 behavior: even when native CuRobo planning returns no successful candidate, the standalone rule provider can still generate rule raw seed trajectories for the future repair stage.

## Phase Boundary

- Implemented in Phase 3: baseline family, goal-anchor rank, twist-delayed/in-limit-best config support, raw/smoothed/bridged variants, lineage, lifecycle and dataset export.
- Deferred to Phase 4 by design: cspace split repair and admitting rule/diffusion seeds into the CuRobo repair pool.

Large run artifacts remain under `runs/` and are not committed.
