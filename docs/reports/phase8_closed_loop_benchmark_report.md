# Closed-Loop CuRobo Benchmark

- requests: `runs/phase7_offline_pipeline_smoke/requests.jsonl`
- total_budget_ms: `2500.0`
- request_count: `3`

| Strategy | Final Success | Fallback Recovery | P50 ms | P95 ms |
|---|---:|---:|---:|---:|
| rule_only | 0.333 | 0.000 | 5004.15 | 10159.50 |
| diffusion_only | 0.000 | 0.000 | 1913.83 | 2014.79 |
| diffusion_critic | 0.000 | 0.000 | 1980.18 | 2104.81 |
| mixed_fallback | 0.333 | 0.333 | 8117.69 | 15038.97 |

This benchmark runs the actual standalone CuRobo planner core. Collision replay remains transparent `unchecked` until the hard validator gains world collision distance labels.
