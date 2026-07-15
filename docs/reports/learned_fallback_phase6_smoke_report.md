# Learned-First Fallback Phase 6 Smoke Report

Date: 2026-07-15

Phase 6 introduced explicit online control flow:

```text
learned seed branch -> rule fallback branch -> planner native fallback
```

## Smoke Result

Run: `phase6_learned_fallback_smoke`

| branch | status | selected candidate | timeout |
| --- | --- | --- | --- |
| learned | `failed_planner` | none | false |
| rule_fallback | `failed_planner` | none | false |
| planner_native | `success` | `planner_00` | false |

Final result:

- status: `success`
- success_source: `planner_native`
- selected_candidate_id: `planner_00`
- exported candidate samples: 5
- positive_for_diffusion: 1
- negative_for_critic: 4

The lifecycle `fallback_trace` records seed provider reports, branch attempts, elapsed time, optional budget and final selection. Failed learned/rule candidates remain in the dataset as negative samples.

Large run artifacts remain under `runs/` and are not committed.
