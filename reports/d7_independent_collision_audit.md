# D7 Independent World-Collision Audit

- scanned selected trajectories: `14028`
- world source: `planner_config_obstacle_json`
- ignored links: `['XMS5-R800-W4G3B4C_base']`
- checked successful trajectories with world boxes: `1578`
- skipped successes without world boxes: `0`
- independent collision findings: `0`

## Distance Distribution

- min: `0.0061527` m
- median: `0.130074556` m
- p05: `0.014847119` m
- p95: `0.205089127` m

## Scope

World-obstacle audit only. This does not check self-collision and is not an FCL replacement; it is a second implementation independent of level_planner_core.validators and cuRobo collision APIs.
