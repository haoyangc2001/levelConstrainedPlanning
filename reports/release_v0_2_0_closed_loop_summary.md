# Release v0.2.0-closed-loop-baseline

## Baseline

- System mainline: `data generation -> model learning -> optimization validation -> failure fallback -> data update`
- Dataset: `sr5_closed_loop_phase7_smoke_20260715`
- Dataset samples: `44`
- Diffusion checkpoint: `/pub/data/caohy/levelConstrainedPlanning/checkpoints/sr5_closed_loop_phase8_diffusion_20260715/best.pt`
- Critic checkpoint: `/pub/data/caohy/levelConstrainedPlanning/checkpoints/sr5_closed_loop_phase8_success_critic_20260715/best.pt`
- Benchmark summary: `/pub/data/caohy/levelConstrainedPlanning/reports/sr5_closed_loop_phase8_benchmark_summary.json`

## Checklist

- Artifact pointers valid: `True`
- No large/checkpoint files tracked: `True`
- Docs aligned: `True`
- CLI/API/ROS use core: `True`

## Known Limits

- Phase 8 model is a closed-loop smoke baseline trained from a very small standalone dataset.
- Diffusion-only and diffusion+critic did not outperform rule-only in the first CuRobo benchmark.
- Collision replay is still recorded as transparent `unchecked`; hard validator has joint, alignment, continuity, and goal checks.
- Larger datasets should be generated under `/pub/data/caohy/levelConstrainedPlanning` before claiming model quality improvement.
