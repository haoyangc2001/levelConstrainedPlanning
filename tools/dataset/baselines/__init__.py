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

__all__ = ["_curobo_soft_cost"]
