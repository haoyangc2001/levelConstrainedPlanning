#!/usr/bin/env python3
"""Uniform benchmark *method* dispatch (A3.7 method-axis seam).

The closed-loop benchmark (``run_closed_loop_benchmark.py``) and the lifecycle
batch (``run_lifecycle_batch.py``) both need to turn a *method name* plus a
*compute budget* into a concrete planner configuration and per-request seed
policy.  Historically that mapping lived inline in the benchmark with hard-coded,
**unequal** ``timeout_sec`` per strategy (rule 0.5 s vs learned 2.0 s), which made
the fixed-budget comparison meaningless (A3.0).

This module centralises the mapping so that:

* every internal method shares the *same* budget semantics (A3.0c) -- the budget
  is expressed as ``k_generate`` (number of seeds / solve attempts allowed, i.e.
  a **compute budget**, since CuRobo ``solve_pose`` cannot be interrupted
  mid-solve) with a *uniform* wall-clock ``timeout_sec`` guard;
* Phase B external baselines slot in as new registry entries exposing the same
  ``build_seed_policy`` / ``apply_config`` / ``is_external`` surface (A3.7 seam);
* both runners import one dispatch instead of duplicating strategy tables.

Budget semantics note (A3.0b): because ``solve_pose`` has no wall-clock timeout
and cannot be pre-empted, the primary budget axis is the number of allowed seed
/ solve attempts (``k_generate``).  ``timeout_sec`` remains as a *uniform* guard
that truncates seed *generation* (rule_seed honours it at generation time), not
as a per-strategy tuning knob.  Fig.4's x-axis is therefore labelled as a
compute budget (K), with observed wall-clock latency reported alongside.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class MethodSpec:
    """Declarative description of a benchmark method.

    ``mode`` is the planner seed-policy mode. ``use_critic`` toggles
    ``config.learned_seed_use_critic``. ``fallback_to_rule`` /
    ``fallback_to_native`` control the control-flow fallbacks. ``external`` marks
    a Phase B baseline that does not use the internal planner control flow (its
    ``runner`` is invoked instead). ``report_goal_only`` (B4) removes the
    level-first selection gate at the *report layer*: the planner selects by
    continuity/path cost ignoring alignment and reports success on goal-reaching,
    while the true per-candidate alignment deviation is still recorded so the
    summariser measures the real level-violation rate of a level-agnostic seeder.
    """

    name: str
    mode: str
    use_critic: bool = False
    fallback_to_rule: bool = False
    fallback_to_native: bool = False
    external: bool = False
    runner: Callable[..., dict[str, Any]] | None = None
    report_goal_only: bool = False
    description: str = ""


# ---------------------------------------------------------------------------
# Internal method registry (Phase A). Phase B baselines register via
# ``register_method`` with ``external=True`` and a ``runner`` callable.
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, MethodSpec] = {}


def register_method(spec: MethodSpec) -> None:
    _REGISTRY[spec.name] = spec


def _register_defaults() -> None:
    register_method(
        MethodSpec(
            name="rule_only",
            mode="rule",
            fallback_to_rule=True,
            fallback_to_native=False,
            description="Rule seeds only, no learned seeds, no native fallback.",
        )
    )
    register_method(
        MethodSpec(
            name="diffusion_only",
            mode="diffusion",
            use_critic=False,
            fallback_to_rule=False,
            fallback_to_native=False,
            description="Diffusion seeds only, no critic re-ranking, no fallback.",
        )
    )
    register_method(
        MethodSpec(
            name="diffusion_critic",
            mode="diffusion",
            use_critic=True,
            fallback_to_rule=False,
            fallback_to_native=False,
            description="Diffusion seeds re-ranked by the success critic, no fallback.",
        )
    )
    register_method(
        MethodSpec(
            name="mixed_fallback",
            mode="mixed",
            use_critic=True,
            fallback_to_rule=True,
            fallback_to_native=True,
            description="Diffusion+critic with rule and native fallback (deployment mode).",
        )
    )
    # B3a: cuRobo native unconstrained lower bound. mode="native" is neither a
    # learned mode nor "rule", so _plan_with_control_flow falls straight through
    # to the planner_native branch (self._planner.plan_pose(goal, ...)), which
    # solves to the goal *pose* with NO level-axis constraint. The shared hard
    # validator then measures its (expected high) alignment violation. This is an
    # internal method (reuses the planner control flow), not an external runner.
    register_method(
        MethodSpec(
            name="baseline/curobo_native_unconstrained",
            mode="native",
            use_critic=False,
            fallback_to_rule=False,
            fallback_to_native=True,
            description=(
                "cuRobo native motion generation to the goal pose, no level-axis "
                "constraint (B3a unconstrained lower bound / strawman floor)."
            ),
        )
    )
    # B4: one-way learned seeds (DiffusionSeeder-style). Same diffusion seeding as
    # diffusion_only (critic off, no fallback), but with the level-first selection
    # gate stripped at the *report layer* (report_goal_only=True): the planner
    # selects the smoothest goal-reaching candidate ignoring alignment and reports
    # success on goal-reaching -- faithful to a level-agnostic learned seeder that
    # is repaired to the goal but never gated on the level constraint. The true
    # per-candidate alignment deviation is still recorded, so the summariser
    # quantifies the baseline's real level-violation rate. Shared with Phase D E3.
    register_method(
        MethodSpec(
            name="baseline/learned_seed_goal_only",
            mode="diffusion",
            use_critic=False,
            fallback_to_rule=False,
            fallback_to_native=False,
            report_goal_only=True,
            description=(
                "One-way learned (diffusion) seeds repaired to the goal with NO "
                "level-first gate; success = goal reached, true level-violation "
                "rate measured (B4 / E3 ablation arm)."
            ),
        )
    )


_register_defaults()


def known_methods() -> list[str]:
    return list(_REGISTRY.keys())


def get_method(name: str) -> MethodSpec:
    try:
        return _REGISTRY[name]
    except KeyError as exc:  # noqa: TRY003
        raise ValueError(
            f"unsupported method: {name!r}; known methods: {sorted(_REGISTRY)}"
        ) from exc


def is_external(name: str) -> bool:
    return get_method(name).external


def build_seed_policy(
    request: dict[str, Any],
    method: str,
    *,
    k_generate: int,
    k_accept: int,
    timeout_sec: float,
    compute_budget: int | None = None,
) -> dict[str, Any]:
    """Return a request copy with a uniform, budget-parameterised seed policy.

    A3.0c: ``timeout_sec`` is identical across methods. A3.1/A3.2: ``k_generate``
    (and the compute budget) is the swept quantity. A3.0b: the compute budget is
    recorded as the number of allowed seed/solve attempts.
    """

    spec = get_method(method)
    updated = deepcopy(request)
    seed_policy = dict(updated.get("seed_policy") or {})
    metadata = dict(updated.get("metadata") or {})

    k_generate = max(1, int(k_generate))
    k_accept = max(1, min(int(k_accept), k_generate))
    budget = int(compute_budget) if compute_budget is not None else k_generate

    seed_policy.update(
        {
            "mode": spec.mode,
            "k_generate": k_generate,
            "k_accept": k_accept,
            "fallback_to_rule_seed": bool(spec.fallback_to_rule),
            "fallback_to_planner_native": bool(spec.fallback_to_native),
            "timeout_sec": float(timeout_sec),
        }
    )
    metadata["benchmark_method"] = method
    metadata["compute_budget_solve_calls"] = budget
    metadata["budget_semantics"] = "compute_budget_solve_calls"
    metadata["uniform_timeout_sec"] = float(timeout_sec)
    # B4: strip the level-first gate at the report layer for one-way-seed baselines.
    if spec.report_goal_only:
        metadata["report_goal_only_no_level_gate"] = True
    # Legacy field kept for backward-compat readers; now interpreted as a guard,
    # not a per-strategy tuning knob.
    metadata["total_budget_ms"] = float(timeout_sec) * 1000.0

    updated["seed_policy"] = seed_policy
    updated["metadata"] = metadata
    updated["request_id"] = f"{request.get('request_id', 'request')}_{method}"
    return updated


def apply_config(config: Any, method: str) -> Any:
    """Mutate a ``LevelPlannerConfig`` for the given method (in place) and return it."""

    spec = get_method(method)
    config.learned_seed_use_critic = bool(spec.use_critic)
    return config


def run_method(
    method: str,
    request: dict[str, Any],
    *,
    planner: Any,
    config: Any,
    out_dir: Any = None,
    **context: Any,
) -> dict[str, Any]:
    """Dispatch one request through the method registry (B0 method-axis seam).

    This is the single choke point that turns a *method name* into a concrete
    ``result_dict``. It replaces the benchmark's former hard-coded
    ``planner.plan(...)`` call so external baselines (Phase B) slot in without
    the runner knowing anything about them:

    * **Internal methods** (``ours/*`` -- the historical rule/diffusion/critic/
      mixed strategies) run through the shared ``LevelConstrainedPlanner``
      control flow, exactly as before.
    * **External methods** (``spec.external is True``) must supply a ``runner``
      callable with the contract ``runner(request, config, *, planner, out_dir,
      **context) -> result_dict``. The ``planner`` is passed through so an
      external baseline can borrow the CuRobo collision world / repair adapter
      and emit ``candidate_records`` scored by the *same* hard validator (B6),
      keeping the comparison apples-to-apples. B0 only wires the seam; no
      external runner is registered yet.

    The returned dict is expected to follow the planner result contract
    (``request_id`` / ``status`` / ``metrics`` / ``candidate_records`` ...), so
    the benchmark's summariser and the A4 converter treat every method
    uniformly.
    """

    spec = get_method(method)
    if spec.external:
        if spec.runner is None:
            raise ValueError(
                f"external method {method!r} declares no runner callable; "
                "register_method(...) must supply one"
            )
        return spec.runner(
            request,
            config,
            planner=planner,
            out_dir=out_dir,
            **context,
        )
    if planner is None:
        raise ValueError(
            f"internal method {method!r} requires a planner instance; got None"
        )
    return planner.plan(request, out_dir=out_dir)
