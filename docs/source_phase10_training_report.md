<!-- [caohy] diffusionSeedLearning phase 10 SR5 mature offline model report -->
# Phase 10 SR5 Mature Offline Model

## Dataset

- dataset: `sr5_phase10_20260715_baseline_jitter_train`
- manifest: `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/datasets/sr5_phase10_20260715_baseline_jitter_train/manifest.json`
- validated samples: `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/datasets/sr5_phase10_20260715_baseline_jitter_train/samples_validated.jsonl`
- collection summaries:
  - `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_lifecycle_collection_20260714_232108/collection_summary.json`
  - `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_lifecycle_collection_20260714_232927/collection_summary.json`
  - `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_lifecycle_collection_20260715_003125/collection_summary.json`
  - `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_lifecycle_collection_20260715_014928/collection_summary.json`

## Counts

| item | count |
|---|---:|
| total samples | 993 |
| candidate samples | 833 |
| request failures | 160 |
| validator-valid candidates | 547 |
| diffusion positives | 140 |
| critic positives | 140 |
| critic negatives | 693 |

The dataset combines one baseline SR5 strict 100-target sweep and two target jitter sweeps. The jitter sweeps use `--datagen-position-jitter-m 0.015`, `--datagen-z-jitter-m 0.01`, and seeds `101` / `202`.

## Checkpoints

| model | best checkpoint | training size | best loss |
|---|---|---:|---:|
| diffusion | `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/sr5_phase10_mature_diffusion_20260715/best.pt` | 140 positives | 0.096022 |
| critic | `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/checkpoints/sr5_phase10_success_critic_20260715/best.pt` | 833 candidates | 0.185262 |

Diffusion training used `epochs=240`, `batch_size=64`, `hidden_dim=128`, `diffusion_steps=64`, and `horizon=32`.

Critic training used `epochs=160`, `batch_size=64`, `hidden_dim=128`, and `horizon=32`.

## Offline Generation Check

- generated samples: `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_phase10_mature_diffusion_20260715_generated_samples.json`
- evaluation json: `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning/reports/sr5_phase10_mature_diffusion_20260715_offline_generation_report.json`
- diffusion generated count: 512
- diffusion precheck valid ratio: 1.000
- diffusion max joint step L2: 0.644281
- rule positive replay valid ratio: 0.96875
- random seed valid ratio: 0.000

This is the first non-smoke SR5 diffusion seed checkpoint. It is suitable for offline sampling and simulation shadow/candidate experiments, but it is still not a final executable trajectory policy. CuRobo repair, fallback rule seeds, and hard validation remain required.

## Runtime Boundary

`resource/config/Level_Test_V2_caohy/start.launch.yaml` now points to the phase 10 checkpoint and generated samples for file-backed or inference experiments. Defaults still keep `diffusion_seed_mode=off`; real robot candidate mode remains blocked unless explicitly overridden.
