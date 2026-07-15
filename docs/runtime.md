# Runtime

The primary runtime contract is plan-only:

```bash
python -m level_planner.cli plan \
  --config configs/sr5_level.yaml \
  --request examples/requests/request_level_001.json \
  --out runs/request_level_001
```

ROS launch, HTTP task triggering, and motion state machines are outside the core runtime.

