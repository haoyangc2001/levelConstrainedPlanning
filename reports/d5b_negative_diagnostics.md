# D5b Negative Diagnostics

- eval split: `test`
- request count: `1169`
- budget semantics: `compute_budget_solve_calls`

## Success Summary

| Method | Success mean | Success std | Gap to rule | P50 ms | P98 ms |
|---|---:|---:|---:|---:|---:|
| diffusion_critic | 0.0 | 0.0 | -0.221557 | 201.485 | 244.709 |
| diffusion_only | 0.0 | 0.0 | -0.221557 | 186.269 | 232.507 |
| mixed_fallback | 0.2284 | 0.0 | 0.006843 | 1488.447 | 2648.898 |
| rule_only | 0.221557 | 0.0 | 0.0 | 4.998 | 13.472 |

## Key Failure Events

| Method | Repair | Alignment | Collision | Joint Limit | Vel/Accel |
|---|---:|---:|---:|---:|---:|
| diffusion_critic | 14024 | 14024 | 4415 | 7025 | 0 |
| diffusion_only | 14028 | 14028 | 4461 | 7034 | 0 |
| mixed_fallback | 43265 | 43374 | 10227 | 28114 | 6948 |
| rule_only | 17538 | 17556 | 5335 | 9698 | 6511 |

## Diagnosis

- Pure learned branches produced no hard-valid selected trajectory in C4.
- Mixed fallback recovered only through fallback, not through learned-only success.
- The pre-registered C4 decision rule did not show any learned method superior to rule_only.
- The defensible paper narrative is therefore a verified integration/fallback architecture plus negative diagnostics, not a self-improving performance claim.
- mixed_fallback success sources: {'none': 2706, 'planner_native': 24, 'rule_fallback': 777}
- rule_only success sources: {'none': 2730, 'rule': 777}

## Paper Implication

- `claim_status`: `self_improving_not_supported`
- `recommended_narrative`: `fallback_narrative_with_failure_taxonomy`
- `skip_full_recompute`: `True`
