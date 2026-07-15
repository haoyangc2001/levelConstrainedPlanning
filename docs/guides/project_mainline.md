# Project Mainline

The project is not a single-shot planner wrapper. Its final target is the closed-loop learning-optimization system defined by the original design documents and the system flow diagram:

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

## System Flow

The architecture has an offline learning path and an online planning path.

Offline path:

```text
task-scene sampling
-> rule seed construction + CuRobo optimization
-> trajectory dataset
-> learning model training
```

Online path:

```text
online task input
-> learned trajectory seed generation
-> CuRobo optimization and constraint validation
-> success decision
-> final trajectory or rule-seed fallback
```

Feedback path:

```text
online success/failure/fallback outcomes
-> trajectory dataset
-> next training cycle
```

The feedback path is mandatory. A successful online trajectory, a failed learned seed, and a rule-fallback recovery are all useful training/evaluation records.

## Loop Contract

1. Data generation: sample start states, target poses, obstacle layouts, and level constraints; run constrained planning tasks through the standalone core; record every candidate, optimized result, validation metric, and failure reason.
2. Model learning: train diffusion seed and success critic models from validated records, with large data/checkpoints stored outside git.
3. Optimization validation: send learned seeds back through CuRobo repair and the hard validator before considering them usable.
4. Failure fallback: when learned seeds fail or score poorly, run the rule seed families, expand the seed search space, and record the recovery path.
5. Data update: merge new successes, failed learned seeds, fallback recoveries, unrecovered failures, and hard cases into versioned datasets for the next training/evaluation cycle.

## Engineering Rules

- Learned seeds are inputs to optimization, not final executable trajectories.
- Final success is measured after CuRobo repair and hard validation, under a fixed total time budget.
- Failure records are first-class data and should not be discarded.
- The rule seed planner is both the reliability fallback and the offline data generator.
- Online outputs must preserve enough metrics to become offline training/evaluation samples.
- Large datasets, lifecycle logs, generated samples, and checkpoints stay under `/pub/data/caohy` or another artifact store, not in git.
- The pure planner core must remain independent of ROS adapters, RViz, state machines, and product-specific robot execution modules.

## Current Baseline

The current repository already contains the first loop baseline:

- rule seed and diffusion seed provider interfaces in the planner core;
- CLI/headless smoke paths for repeatable generation and validation;
- artifact pointers for the phase10 diffusion and critic checkpoints;
- dataset/export and evaluation tools under `tools/`;
- design documents under `docs/design/` that define the trajectory-optimization and diffusion-seed principles.

The next project work should close the loop more tightly: make online run artifacts directly exportable into the dataset schema, keep failure/fallback evidence complete, and evaluate learned seeds by final CuRobo-validated success rate rather than by model loss alone.
