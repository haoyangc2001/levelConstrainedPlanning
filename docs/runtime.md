# Runtime

The primary runtime contract is plan-only:

```bash
python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_001.json \
  --out runs/request_level_001
```

ROS launch, HTTP task triggering, and motion state machines are outside the core runtime.

## Fixtures

```bash
python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_001.json \
  --out runs/request_level_001

python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_alignment_hard.json \
  --out runs/request_level_alignment_hard

python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_planner_fail.json \
  --out runs/request_level_planner_fail
```

Expected statuses:

```text
request_level_001              success
request_level_alignment_hard   failed_alignment_constraint
request_level_planner_fail     failed_planner
```

## Full Matrix

```bash
python scripts/headless_smoke.py
```

Generated artifacts are written under `runs/`. The committed summary is under `reports/`.
