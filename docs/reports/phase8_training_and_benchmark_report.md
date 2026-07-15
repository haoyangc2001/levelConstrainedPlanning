# Phase 8 Training And Benchmark Report

## Artifacts

- Dataset: `/pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/samples_validated.jsonl`
- Diffusion checkpoint: `/pub/data/caohy/levelConstrainedPlanning/checkpoints/sr5_closed_loop_phase8_diffusion_20260715/best.pt`
- Critic checkpoint: `/pub/data/caohy/levelConstrainedPlanning/checkpoints/sr5_closed_loop_phase8_success_critic_20260715/best.pt`
- Generated samples: `/pub/data/caohy/levelConstrainedPlanning/reports/sr5_closed_loop_phase8_diffusion_generated_samples.json`
- Offline precheck report: `/pub/data/caohy/levelConstrainedPlanning/reports/sr5_closed_loop_phase8_offline_generation_report.json`
- CuRobo benchmark summary: `/pub/data/caohy/levelConstrainedPlanning/reports/sr5_closed_loop_phase8_benchmark_summary.json`
- Runtime pointer: `artifacts/current_artifacts.json`

## Training

- Diffusion samples: `3` positives
- Diffusion best loss: `0.3456296325`
- Diffusion model config: `dof=6`, `condition_dim=15`, `hidden_dim=64`, `steps=32`, `horizon=32`
- Critic samples: `31` candidates with points
- Critic positives/negatives: `2 / 29`
- Critic best loss: `0.1167937568`

Both training metadata files record `samples_sha256`, model config, hyperparameters, best checkpoint, and last checkpoint. Phase10 model and critic pointers are retained in `legacy_phase10_diffusion` and `legacy_phase10_critic`.

## Offline Seed Precheck

- Diffusion generated seeds: `6 / 6` pass the lightweight continuity precheck
- Rule positive replay: `2 / 2` pass
- Random baseline: `0 / 6` pass

This precheck only verifies seed shape, start continuity, max joint step, and joint absolute bounds.

## CuRobo Benchmark

Benchmark command used the first 3 sampled requests with a fixed `2500 ms` budget marker and ran the actual standalone CuRobo planner core.

| Strategy | Final Success | Fallback Recovery | P50 ms | P95 ms |
|---|---:|---:|---:|---:|
| rule_only | 0.333 | 0.000 | 5004.15 | 10159.50 |
| diffusion_only | 0.000 | 0.000 | 1913.83 | 2014.79 |
| diffusion_critic | 0.000 | 0.000 | 1980.18 | 2104.81 |
| mixed_fallback | 0.333 | 0.333 | 8117.69 | 15038.97 |

The current standalone smoke dataset is enough to prove the closed-loop plumbing, but not enough for a strong learned seed model. The learned branches generated valid-shaped seeds, then failed in CuRobo repair; the mixed strategy still recovered one task through rule fallback. This is the expected first closed-loop baseline: data collection, learning, repair validation, fallback, and artifact update are all wired end to end.

## Runtime Resolution

`configs/sr5_level.yaml` now loads diffusion and critic checkpoint paths from `artifacts/current_artifacts.json` when `load_model_paths_from_artifacts=true`. Updating the artifact pointer is therefore sufficient to switch runtime models without editing planner code.
