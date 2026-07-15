# Runtime

The runtime is the online half of the closed-loop system:

```text
online task input
-> learned trajectory seed generation
-> CuRobo optimization and constraint validation
-> final trajectory or rule-seed fallback
-> data record for the offline loop
```

The primary contract is plan-only:

```bash
python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_001.json \
  --out runs/request_level_001
```

ROS launch, HTTP task triggering, robot execution, and motion state machines are outside the core runtime.

## Online Loop Behavior

Runtime requests should be interpreted as validation jobs, not direct execution commands.

- `seed_policy.mode=diffusion` or `mixed`: try learned seed candidates first.
- `fallback_to_rule_seed=true`: run the rule seed families when learned seeds fail validation.
- CuRobo repair and hard validation remain mandatory for every candidate.
- The result artifacts should record selected trajectory, rejected candidates, failure reason, and metrics so the run can feed the offline dataset.

If both learned seeds and fallback rule seeds fail, the correct result is a structured failure status plus metrics, not an unvalidated trajectory.

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

These artifacts are part of the feedback loop. Successful trajectories, failed learned seeds, fallback recoveries, and validator metrics should be exported into offline datasets before the next model training cycle.
