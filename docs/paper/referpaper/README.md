# Related Work — downloaded references

PDFs downloaded from arXiv, relevant to the SR5 level-constrained planning /
closed-loop learning–optimization work. Filenames encode `topic_year_shortname`.
Grouped by the related-work sections in `../main.tex`.

## Downstream optimizer (the planner core is built on this)
- `curobo_2023_parallel_collisionfree_motion.pdf` — **cuRobo** (arXiv:2310.17274).
  GPU-batched IK + collision + minimum-jerk trajectory optimization. This is the
  optimizer our seeds are repaired/refined by; its soft-cost + seed-sensitivity
  limitations are exactly what our seed-construction front-end addresses.

## Sampling-based constrained / manifold planning
- `cpRRTC_2025_gpu_constrained_rrtconnect.pdf` — cpRRTC (arXiv:2505.06791).
  GPU-parallel RRT-Connect extended to constraint manifolds.
- `McVAMP_2026_vectorized_projection_manifold.pdf` — McVAMP (arXiv:2604.13323).
  SIMD-parallel projection for manifold-constrained whole-body planning.
- `VAMP_2023_motions_in_microseconds.pdf` — VAMP (arXiv:2309.14545). Vectorized
  sampling-based planning; the SIMD backbone the above build on.
- `pRRTC_2025_gpu_parallel_rrtconnect.pdf` — pRRTC (arXiv:2503.06757). GPU RRT-Connect.

## IK-based methods (multi-branch, continuity)
- `IKFlow_2021_diverse_ik_solutions.pdf` — IKFlow (arXiv:2111.08933). Normalizing
  flow over redundant-arm IK solution distribution; relevant to our multi-branch
  IK seed diversity.
- `IKLink_2024_ee_trajectory_tracking.pdf` — IKLink (arXiv:2402.16154).
  End-effector trajectory tracking with minimal configuration switches; directly
  related to our branch-consistent IK seed selection.

## Trajectory optimization
- `GPMP2_2017_continuous_time_gp_motion_planning.pdf` — GPMP2 (arXiv:1707.07383).
  Continuous-time GP trajectory optimization via probabilistic inference.
  (CHOMP / STOMP / TrajOpt are pre-arXiv or non-arXiv; cite from their venues.)

## Learning-based seeding & diffusion planning (the closed-loop contribution)
- `DiffusionSeeder_2024_seeding_motion_optimization.pdf` — DiffusionSeeder
  (arXiv:2410.16727). Diffusion generates seeds, cuRobo refines — the closest
  prior work to our diffusion-seed + cuRobo-repair loop.
- `PRESTO_2024_diffusion_key_configuration.pdf` — PRESTO (arXiv:2409.16012).
  Key-configuration environment representation + collision/smoothness losses in
  diffusion training; informs our constraint-aware seed learning.
- `MotionPlanningDiffusion_2023.pdf` — MPD (arXiv:2308.01557). Trajectory
  distribution prior + cost-guided diffusion planning.
- `Diffuser_2022_planning_with_diffusion.pdf` — Diffuser (arXiv:2205.09991).
  Foundational full-trajectory diffusion planning with test-time guidance.
- `DiffusionPolicy_2023_visuomotor_action_diffusion.pdf` — Diffusion Policy
  (arXiv:2303.04137). Temporal U-Net + conditioning practice we borrow for the
  1D U-Net seed model (not the end-to-end control objective).
- `ModelBasedDiffusion_2024_trajectory_optimization.pdf` — Model-Based Diffusion
  (arXiv:2407.01573). How differentiable dynamics/cost enter the denoising process.

## Generative-model acceleration (future-work / few-step sampling)
- `FlowMatching_2022_generative_modeling.pdf` — Flow Matching (arXiv:2210.02747).
- `ConsistencyModels_2023.pdf` — Consistency Models (arXiv:2303.01469).
  Both relevant to meeting the fixed-time-budget requirement via few-step sampling.

---

Notes:
- Not on arXiv (cite from original venues): CHOMP (ICRA 2009), STOMP (ICRA 2011),
  TrajOpt (IJRR 2014), TSR/CBiRRT (ICRA 2009), TRAC-IK (Humanoids 2015),
  RelaxedIK (RSS 2018), IKFast/OpenRAVE, OMPL constrained planning.
- CSVTO is resolved in `../references.bib` as "Constrained Stein Variational
  Trajectory Optimization" by Power and Berenson (arXiv:2308.12110). It is kept
  as the B3c constrained-optimization reference; no local PDF has been added yet.
- BibTeX keys should be added to `../references.bib` as these get cited.
