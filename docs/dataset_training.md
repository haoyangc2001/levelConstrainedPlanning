# Dataset And Training

Dataset export, offline validation, diffusion training, critic training, sampling, and evaluation tools are migrated under:

```text
tools/
```

Large generated artifacts stay outside git under:

```text
/pub/data/caohy/tashan_Manipulation/diffusionSeedLearning
```

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
