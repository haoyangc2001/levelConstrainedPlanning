# Dataset And Training

Dataset export, offline validation, diffusion training, critic training, sampling, and evaluation tools implement the offline half of the closed-loop system:

```text
task-scene sampling
-> rule seed construction + CuRobo optimization
-> trajectory dataset
-> learning model training
-> online validation feedback
```

Tools are migrated under:

```text
tools/
```

Large generated artifacts stay outside git under:

```text
/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning
```

This workflow serves the project mainline:

```text
data generation -> model learning -> optimization validation -> failure fallback -> data update
```

Dataset records should therefore preserve successes, failed seeds, fallback outcomes, validation metrics, and optimizer costs. Training only on polished final trajectories is not enough; the system needs evidence about which seeds are repairable, which ones fail, and why.

## Dataset Contract

Each generated or online-feedback record should keep enough information to support both diffusion training and critic training:

- task condition: start joints, target pose, obstacle layout, level constraint, robot profile;
- seed evidence: seed source, seed family, random seed, learned/rule mode, candidate index;
- optimization evidence: CuRobo status, optimization time, iteration/cost summary, selected or rejected trajectory;
- validation evidence: level error, collision distance/risk, joint-limit margin, continuity/jump metrics, goal error;
- outcome: success, learned-seed failure, rule-fallback success, unrecovered failure, and failure reason.

The diffusion model should learn high-success seed distributions. The critic should learn repairability, constraint risk, and expected optimization cost from both positive and negative candidates.

Check external artifact pointers:

```bash
python tools/check_artifacts.py --strict
```

Export samples from standalone core result artifacts:

```bash
python tools/dataset/export_core_results_dataset.py \
  --input runs/phase4_success \
  --input runs/phase4_alignment_hard \
  --input runs/phase4_planner_fail \
  --out runs/diffusion_seed_learning/core_result_samples.jsonl
```

Run a small phase10 diffusion seed sampling smoke:

```bash
python tools/learning/diffusion_seed_learning/sample.py \
  --tasks 1 \
  --k 2 \
  --out runs/diffusion_seed_learning/generated_samples_smoke.json

python tools/learning/diffusion_seed_learning/evaluate.py \
  --generated runs/diffusion_seed_learning/generated_samples_smoke.json \
  --json-out runs/diffusion_seed_learning/offline_generation_report.json \
  --md-out runs/diffusion_seed_learning/diffusion_vs_rule_seed_report.md
```

The old lifecycle exporters remain for compatibility with historical `tashan_Manipulation` logs. New data collection should prefer the standalone CLI/core result artifacts.

## Pointer Fields

`artifacts/current_artifacts.json` records:

- `dataset.training_dataset`: phase10 validated JSONL path.
- `dataset.dataset_manifest`: phase10 manifest path and hash.
- `diffusion.best_checkpoint`: current diffusion seed checkpoint.
- `critic.best_checkpoint`: current success critic checkpoint.
- `generated_samples`: offline generated seed report.
- `offline_generation_report`: seed-level evaluation report.

Only this pointer is committed. Large files remain under `/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning`.
