"""B2 baseline: cuRobo soft-constraint (per-axis pose weight) adversary.

**What it is.** The plan's B2 slot is the *soft-cost* strawman: instead of
treating the level axis as a hard manifold constraint, ask cuRobo to keep the
tool orientation via a per-axis **non-terminal pose weight** in its trajectory
optimiser. cuRobo's ``ToolPoseCriteria`` exposes a
``non_terminal_pose_axes_weight_factor`` of the form ``[x, y, z, rx, ry, rz]``;
setting the rotational entries to ``lambda`` (``[0, 0, 0, l, l, l]``) adds a soft
rotational cost along the whole path, while the translational goal is untouched.

**Why it is a fair adversary and not a mirror of ours.** This is *not* our
post-hoc alignment gate. There is **no** level-aware seeding and **no** validator
gate steering candidate selection toward alignment: the soft cost is the *only*
thing pulling the path toward the level axis, exactly as a soft-cost planner
would deploy it. The output trajectory then goes through the *same* A1 collision
+ hard validator (``validators.evaluate_hard_constraints``) as every other
method (B6), so the comparison is apples-to-apples.

**The lambda-sweep (Fig. in E1).** ``lambda`` is read from
``request.metadata.soft_axis_weight`` and swept across benchmark cells (via the
benchmark's method/K/budget cell machinery, or an explicit weight list). The
resulting curve of *alignment violation vs lambda* is the paper's evidence that a
soft cost cannot **enforce** the constraint: small ``lambda`` ignores the axis,
large ``lambda`` fights the translational goal / collision term and either still
violates the tolerance or degrades success -- there is no ``lambda`` that drives
the hard alignment-violation rate to zero the way a manifold projection or our
level-aware pipeline does.

**Mechanism provenance.** Re-ported from the migration snapshot
``migration/source_snapshot/curobo_v2_planner/main.py``
(``_apply_shadow_non_terminal_pose_weight`` / ``update_tool_pose_criteria`` /
``_run_alignment_non_terminal_weight_sweep``), where it lived as a read-only
"shadow" probe. Here it is promoted to a first-class benchmark method behind the
B0 dispatch seam.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from ..methods import MethodSpec, register_method


METHOD_NAME = "baseline/curobo_soft_cost"

# Metadata key the benchmark sweeps to produce the lambda-curve. When absent we
# fall back to a single representative weight so a bare run still produces a
# meaningful soft-cost point rather than the (all-zero) native default.
WEIGHT_METADATA_KEY = "soft_axis_weight"
DEFAULT_SOFT_AXIS_WEIGHT = 1.0


def _resolve_soft_axis_weight(request: dict[str, Any]) -> float:
    """Read the swept lambda from ``request.metadata.soft_axis_weight``.

    The value is the rotational per-axis weight applied to the non-terminal pose
    criteria; ``0.0`` reduces B2 to the unconstrained native baseline (B3a), so a
    missing/invalid value defaults to a representative positive weight instead.
    """
    metadata = request.get("metadata") or {}
    raw = metadata.get(WEIGHT_METADATA_KEY)
    try:
        weight = float(raw)
    except (TypeError, ValueError):
        return float(DEFAULT_SOFT_AXIS_WEIGHT)
    if weight < 0.0:
        return 0.0
    return weight


def _build_criteria(weight: float, *, terminal_weight: Any):
    """Construct a ``ToolPoseCriteria`` with a soft non-terminal rotational cost.

    ``non_terminal_pose_axes_weight_factor = [0, 0, 0, w, w, w]`` mirrors the
    migration snapshot: no soft *position* path cost (translation is governed by
    the terminal goal), only a soft *rotational* path cost of magnitude ``w`` --
    which is exactly the soft analogue of the level-axis constraint.
    """
    # Imported lazily so the module imports cleanly on machines without a built
    # curobo (e.g. doc/CI hosts); only the live runner needs the real class.
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    non_terminal_weight = [0.0, 0.0, 0.0, float(weight), float(weight), float(weight)]
    kwargs: dict[str, Any] = {"non_terminal_pose_axes_weight_factor": non_terminal_weight}
    if terminal_weight is not None:
        kwargs["terminal_pose_axes_weight_factor"] = list(terminal_weight)
    return ToolPoseCriteria(**kwargs)


def _default_criteria():
    """The stock all-zero-non-terminal criteria to restore after the solve."""
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

    return ToolPoseCriteria()


def _native_routed_request(request: dict[str, Any], weight: float) -> dict[str, Any]:
    """Return a request copy that routes through the planner's native path.

    B2 must *not* reuse level-aware seeding or the rule/native fallback ladder --
    the soft cost is the only alignment mechanism under test. We therefore force
    ``mode="native"`` with both fallbacks off (identical routing to B3a), so the
    only difference between B2 and B3a is the applied ``ToolPoseCriteria`` weight.
    """
    routed = deepcopy(request)
    seed_policy = dict(routed.get("seed_policy") or {})
    seed_policy["mode"] = "native"
    seed_policy["fallback_to_rule_seed"] = False
    seed_policy["fallback_to_planner_native"] = True
    routed["seed_policy"] = seed_policy

    metadata = dict(routed.get("metadata") or {})
    metadata[WEIGHT_METADATA_KEY] = float(weight)
    metadata["baseline_method"] = METHOD_NAME
    metadata["soft_cost_axis_weight_factor"] = [0.0, 0.0, 0.0, float(weight), float(weight), float(weight)]
    routed["metadata"] = metadata
    return routed


def run_soft_cost(
    request: dict[str, Any],
    config: Any,
    *,
    planner: Any,
    out_dir: Any = None,
    **_context: Any,
) -> dict[str, Any]:
    """B0 runner: solve with a soft per-axis rotational path cost applied.

    Applies ``ToolPoseCriteria([0,0,0,l,l,l])`` to the live cuRobo solvers,
    routes the request through the planner's native path (so no level-aware
    seeding / gate is involved), then **always** restores the stock criteria so
    the shared planner instance is unchanged for the next method. The returned
    dict is a normal planner ``result_dict`` -- its ``candidate_records`` are
    scored by the same hard validator as every other method (B6).
    """
    if planner is None:
        raise ValueError(f"{METHOD_NAME} requires a live planner instance; got None")

    weight = _resolve_soft_axis_weight(request)
    tool_frame = planner._tool_frames[0] if getattr(planner, "_tool_frames", None) else None
    if tool_frame is None:
        raise ValueError(f"{METHOD_NAME}: planner exposes no tool frame to weight")

    curobo_planner = planner._planner  # the live curobo MotionPlanner

    applied = False
    try:
        criteria = _build_criteria(weight, terminal_weight=None)
        curobo_planner.update_tool_pose_criteria({tool_frame: criteria})
        applied = True
        result = planner.plan(_native_routed_request(request, weight), out_dir=out_dir)
    finally:
        # Restore stock (all-zero non-terminal) criteria unconditionally so the
        # soft cost never leaks into the next benchmark cell.
        if applied:
            try:
                curobo_planner.update_tool_pose_criteria({tool_frame: _default_criteria()})
            except Exception:
                # Best-effort restore; surface via metadata rather than masking
                # the real result/exception.
                pass

    # Annotate the result so the summariser / A4 converter can label the lambda
    # point without re-deriving it from the request.
    if isinstance(result, dict):
        metrics = dict(result.get("metrics") or {})
        metrics["soft_axis_weight"] = float(weight)
        result["metrics"] = metrics
    return result


register_method(
    MethodSpec(
        name=METHOD_NAME,
        mode="native",
        use_critic=False,
        fallback_to_rule=False,
        fallback_to_native=False,
        external=True,
        runner=run_soft_cost,
        description=(
            "B2 soft-constraint adversary: cuRobo non-terminal per-axis pose "
            "weight [0,0,0,l,l,l] (lambda from metadata.soft_axis_weight, swept "
            "for the alignment-violation-vs-lambda curve). No level-aware seeding "
            "or gate; scored by the shared hard validator."
        ),
    )
)
