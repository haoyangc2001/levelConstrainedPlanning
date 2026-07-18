"""External baseline runners (Phase B) for the uniform benchmark method seam.

Each module here registers one or more :class:`~tools.dataset.methods.MethodSpec`
entries with ``external=True`` and a ``runner`` callable following the B0
contract::

    runner(request, config, *, planner, out_dir, **context) -> result_dict

The ``planner`` is the live :class:`LevelConstrainedPlanner`, passed through so a
baseline can borrow its CuRobo collision world / repair adapter and emit
``candidate_records`` scored by the *same* A1 collision + hard validator
(``validators.evaluate_hard_constraints``) as the internal methods. No baseline
gets a private success definition (B6).

Importing this package registers every implemented baseline as a side effect, so
the benchmark and the lifecycle batch only need ``import tools.dataset.baselines``
to make the Phase-B methods available via ``methods.known_methods()``.
"""

from __future__ import annotations

# Import for registration side effects. Keep each import guarded so a single
# broken/optional baseline (e.g. one needing an unavailable external dep) does
# not prevent the others from registering.
from . import curobo_soft_cost as _curobo_soft_cost  # noqa: F401
from . import chomp_constraint as _chomp_constraint  # noqa: F401

# OMPL-backed baselines need the optional
# ``ompl`` wheel. Guard the import so hosts without it still load the other
# baselines; the runner raises a clear error at call time via
# ``ompl_bridge.require_ompl``.
_OMPL_IMPORT_ERROR: str | None = None
try:
    from . import ompl_rrtc as _ompl_rrtc  # noqa: F401
    from . import ompl_projection as _ompl_projection  # noqa: F401

    _OMPL_BASELINES = ["_ompl_rrtc", "_ompl_projection"]
except Exception as _exc:  # pragma: no cover - depends on host env
    _OMPL_IMPORT_ERROR = f"{type(_exc).__name__}: {_exc}"
    _OMPL_BASELINES = []

__all__ = ["_curobo_soft_cost", "_chomp_constraint", *_OMPL_BASELINES]
