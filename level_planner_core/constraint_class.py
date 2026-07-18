"""Canonical constraint-class taxonomy (C1b).

The paper studies four constraint classes on two independent binary axes:

* **level axis** — is the tool-axis *level* alignment constraint active?
  ``L`` = level enforced, ``P`` = plain (no level gate).
* **goal-orientation axis** — is the *goal orientation* (target quaternion)
  a hard requirement, or is only the goal *position* required?
  ``O`` suffix = full orientation constrained, no suffix = position-only.

That yields the four classes referenced across the pipeline (sampler, planner,
validators, ``paper_result``):

======  ===============  ============  ======================================
Class   level active?    goal orient?  meaning
======  ===============  ============  ======================================
LPO     yes              yes           level + full 6-DoF pose goal
LP      yes              no            level + position-only goal
PPO     no               yes           no level + full 6-DoF pose goal
PPO→PP  no               no            no level + position-only goal
======  ===============  ============  ======================================

The four canonical ids are ``LPO``, ``LP``, ``PPO``, ``PP``.

Both axes are enforced as **validation / selection gates**, uniformly with the
pre-existing ``report_goal_only_no_level_gate`` (B4) level relaxation. The solver
may still track full pose internally; a *position-only* problem simply does not
penalise a candidate for orientation error at goal-check / selection time. This
keeps the shared cuRobo solver untouched (no per-request ``update_tool_pose_criteria``
mutation) and every axis is one boolean flag on the normalized request.

This module is the single source of truth. Sampler (``sample_tasks``), planner
(``planner._normalize_request``), validators (``_evaluate_goal``), the condition
encoder (``condition.py``) and ``paper_result`` all import from here so a class id
never means two different things in two files.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConstraintClassSpec:
    """The two boolean axes a constraint-class id decodes to.

    ``level_active`` gates the axis-alignment (level) constraint at selection and
    validation. ``goal_orientation_active`` gates whether the target quaternion is
    a hard goal requirement (``True``) or only the goal position is required
    (``False``, position-only).
    """

    class_id: str
    level_active: bool
    goal_orientation_active: bool
    description: str = ""


#: Canonical order used for one-hot class encoding (condition.py) and reporting.
CONSTRAINT_CLASS_ORDER: tuple[str, ...] = ("LPO", "LP", "PPO", "PP")

#: Default class = the historical behaviour: level on + full pose goal.
DEFAULT_CONSTRAINT_CLASS = "LPO"

_REGISTRY: dict[str, ConstraintClassSpec] = {
    "LPO": ConstraintClassSpec("LPO", level_active=True, goal_orientation_active=True,
                               description="Level constraint + full 6-DoF pose goal."),
    "LP": ConstraintClassSpec("LP", level_active=True, goal_orientation_active=False,
                              description="Level constraint + position-only goal."),
    "PPO": ConstraintClassSpec("PPO", level_active=False, goal_orientation_active=True,
                               description="No level constraint + full 6-DoF pose goal."),
    "PP": ConstraintClassSpec("PP", level_active=False, goal_orientation_active=False,
                              description="No level constraint + position-only goal."),
}


def known_classes() -> list[str]:
    return list(CONSTRAINT_CLASS_ORDER)


def normalize_class_id(class_id: str | None) -> str:
    """Return a canonical class id, defaulting to ``LPO`` for empty/unknown input.

    Accepts case-insensitive ids and the ``unknown`` sentinel used by
    ``paper_result`` (mapped to the default so legacy rows stay runnable).
    """
    if not class_id:
        return DEFAULT_CONSTRAINT_CLASS
    key = str(class_id).strip().upper()
    if key in _REGISTRY:
        return key
    if key in {"UNKNOWN", ""}:
        return DEFAULT_CONSTRAINT_CLASS
    raise ValueError(
        f"unsupported constraint_class: {class_id!r}; known: {CONSTRAINT_CLASS_ORDER}"
    )


def get_spec(class_id: str | None) -> ConstraintClassSpec:
    return _REGISTRY[normalize_class_id(class_id)]


def one_hot(class_id: str | None) -> list[float]:
    """Fixed-order one-hot encoding of the class id (condition.py class encoding)."""
    canonical = normalize_class_id(class_id)
    return [1.0 if canonical == name else 0.0 for name in CONSTRAINT_CLASS_ORDER]


#: Width of the appended class encoding (see condition.py gated extension).
CONSTRAINT_CLASS_ONE_HOT_DIM = len(CONSTRAINT_CLASS_ORDER)
