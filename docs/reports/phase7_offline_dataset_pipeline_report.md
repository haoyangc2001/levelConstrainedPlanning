# Phase 7 Offline Dataset Pipeline Report

## Scope

Phase 7 establishes the offline half of the closed-loop system:

```text
task sampling -> lifecycle batch planning -> candidate dataset export -> manifest/pointer registration
```

The smoke run is headless and does not require RViz.

## Dataset

- Dataset name: `sr5_closed_loop_phase7_smoke_20260715`
- Public dataset directory: `/pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715`
- Samples: `/pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/samples_validated.jsonl`
- Manifest: `/pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/manifest.json`
- Validator report: `/pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/validator_report.json`
- Repository pointer: `artifacts/closed_loop_dataset_pointer.json`

## Results

- Requests sampled: `6`
- Lifecycle runs executed: `6`
- Final successes: `2`
- Final failures: `4`
- Candidate samples: `44`
- Candidates with trajectory points: `31`
- Positive for diffusion: `3`
- Positive for critic: `2`
- Negative for critic: `42`
- Source distribution: `diffusion_seed=12`, `planner_native=16`, `rule_raw_seed=8`, `rule_seed=8`
- Candidate status distribution: `success=15`, `failed_planner=13`, `raw_seed_parent_only=16`

## Validation

```bash
python3 -m compileall level_planner_core level_planner tools tests
PYTHONPATH=$PWD PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q tests/test_phase7_dataset_pipeline.py tests/test_candidate_dataset.py
python3 tools/dataset/validate_candidate_dataset.py \
  --samples /pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/samples_validated.jsonl \
  --json-out /pub/data/caohy/levelConstrainedPlanning/datasets/sr5_closed_loop_phase7_smoke_20260715/validator_report.json \
  --require-positive \
  --require-negative
```

The validator passed with no schema errors. The dataset is intentionally small but exercises success, failure, learned/rule/planner-native candidate sources, and fallback trace recording.
