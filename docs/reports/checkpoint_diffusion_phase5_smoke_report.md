# Checkpoint Diffusion Phase 5 Smoke Report

Date: 2026-07-15

Phase 5 replaced file-backed diffusion replay with runtime checkpoint sampling and online critic selection.

## Smoke Results

| run | diffusion status | critic status | selected for repair | final status |
| --- | --- | --- | ---: | --- |
| `phase5_checkpoint_diffusion_smoke` | `generated` | `scored` | 1 diffusion seed | `success` via planner native |
| `phase5_missing_checkpoint_fallback_smoke` | `checkpoint_missing` | not run | 0 diffusion seeds | `success` via fallback/native path |

The checkpoint smoke uses the diffusion checkpoint from `configs/sr5_level.yaml`, builds the runtime condition vector from the current request, samples seeds online, applies q0 inpainting and continuity recovery, scores candidates with the Success Critic, and sends the selected learned seed into the Phase 4 repair pool.

In the current smoke request, the learned repaired candidate returned `trajopt_failed`; it is recorded as a negative critic sample with source lineage. This is expected until later data-loop phases improve model and critic quality.

## Safety Behavior

- Default request mode remains non-learned unless `seed_policy.mode` explicitly enables diffusion/mixed/candidate/shadow behavior.
- `shadow` mode generates and records learned raw seeds without repair-pool admission.
- Missing diffusion checkpoint returns `checkpoint_missing` and does not block rule/native fallback.
- Missing or failed critic falls back to precheck order and records the critic error in provider metadata.

Large run artifacts remain under `runs/` and are not committed.
