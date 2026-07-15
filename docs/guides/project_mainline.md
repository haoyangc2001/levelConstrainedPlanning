# Project Mainline

The project is not just a single-shot planner wrapper. Its mainline is the closed-loop evolution system defined by the original design documents:

```text
data generation -> model learning -> optimization validation -> failure fallback -> data update
```

Chinese design phrase:

```text
数据生成 - 模型学习 - 优化验收 - 失败回退 - 数据更新
```

## Role Split

The planner core is the executable validation and data-production engine. It runs rule seeds, optional learned seeds, CuRobo repair, and hard validation so every candidate has explicit success/failure evidence.

The learning stack does not replace CuRobo. Diffusion models learn high-success seed distributions; the critic ranks quality and diversity; failed learned candidates must fall back to the rule seed families.

The dataset is not a side artifact. It is the memory of the system: successful trajectories, failed seeds, failure reasons, constraint metrics, optimization cost, and fallback outcomes should all feed the next training cycle.

## Loop Contract

1. Data generation: run constrained planning tasks through the standalone core and record every candidate, optimized result, validation metric, and failure reason.
2. Model learning: train diffusion seed and success critic models from validated records, with large data/checkpoints stored outside git.
3. Optimization validation: send learned seeds back through CuRobo repair and the hard validator before considering them usable.
4. Failure fallback: when learned seeds fail or score poorly, run the rule seed families and record the recovery path.
5. Data update: merge new successes, failures, and hard cases into versioned datasets for the next training/evaluation cycle.

## Engineering Rules

- Learned seeds are inputs to optimization, not final executable trajectories.
- Final success is measured after CuRobo repair and hard validation, under a fixed total time budget.
- Failure records are first-class data and should not be discarded.
- Large datasets, lifecycle logs, generated samples, and checkpoints stay under `/pub/data/caohy` or another artifact store, not in git.
- The pure planner core must remain independent of ROS adapters, RViz, state machines, and product-specific robot execution modules.

## Current Baseline

The current repository already contains the first loop baseline:

- rule seed and diffusion seed provider interfaces in the planner core;
- CLI/headless smoke paths for repeatable generation and validation;
- artifact pointers for the phase10 diffusion and critic checkpoints;
- dataset/export and evaluation tools under `tools/`;
- design documents under `docs/design/` that define the trajectory-optimization and diffusion-seed principles.
