# Phase 9 Closed-Loop Smoke Report

## Command

```bash
PYTHONPATH=$PWD python scripts/closed_loop_smoke.py \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_alignment_hard.json \
  --out-dir runs/phase9_closed_loop_smoke \
  --summary-out runs/phase9_closed_loop_smoke/summary.json \
  --device cuda:0 \
  --num-candidates 2 \
  --warmup-iterations 0
```

## Strategy Matrix

| Strategy | Status | Success Source | Notes |
|---|---|---|---|
| rule_only | `failed_planner` | `none` | Native fallback intentionally disabled for pure rule branch. |
| diffusion_shadow | `failed_alignment_constraint` | `none` | Diffusion generated shadow candidates and native planner produced candidates, but strict alignment failed. |
| diffusion_candidate | `failed_all_fallbacks` | `none` | Learned candidate entered repair/validation path and failed; failure was recorded as data. |
| mixed_fallback | `failed_all_fallbacks` | `none` | Learned, rule fallback, and native fallback traces were recorded. |

## Dataset Export

- Candidate samples exported: `17`
- Candidates with trajectory points: `11`
- Negative critic samples: `17`
- Source distribution: `diffusion_seed=5`, `planner_native=4`, `rule_raw_seed=4`, `rule_seed=4`
- Schema validation: passed

## Checks

- Dataset export: passed
- Dataset schema validation: passed
- Model sample path: passed
- Critic score path: passed
- Fallback trace path: passed
- RViz/display requirement: none

This smoke proves that learned seeds enter the same repair/validation/data-recording path as rule and native candidates. It does not claim the Phase 8 learned model improves success rate; that is covered by the Phase 8 benchmark report.
