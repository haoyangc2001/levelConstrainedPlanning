# B1 — External Baseline Survey & Borrow/Reimplement Decision Matrix

*Task B1 deliverable. Scopes the constraint-enforcing adversaries (B5 projection,
B3c classic constraint-opt) so that at least one is guaranteed deliverable even
if all external borrowing fails.*

> **Verification status.** Live web/repo lookups were unavailable at authoring
> time (the search backend routed through an unreachable model and returned 503).
> The matrix below is built from established knowledge of these libraries and the
> reference PDFs in `docs/paper/referpaper/`. Every row's *repo URL* and *exact
> license* is marked **[verify]** and must be confirmed against the upstream
> repository before it is cited in the paper. The **decision** column (borrow vs
> reimplement) and the **minimum-viable paths** (§3, §4) do **not** depend on that
> verification — they are deliberately written so the constraint-enforcing
> adversary survives even if every "borrow" turns out infeasible.

---

## 0. What the baselines must bridge to

The level constraint is a single differentiable scalar per configuration `q`:

```
c(q) = angle_deg( R(q) · local_axis ,  target_world_axis ) − tolerance_deg   ≤ 0
```

- `R(q)` — tool-frame rotation from forward kinematics of joint config `q`
  (SR5 6-DOF; FK already available via the planner's kinematics_fn).
- `local_axis` — tool-frame axis, request field `alignment.local_axis`
  (default `[0,1,0]`), `planner.py:596-597`.
- `target_world_axis` — world-frame target, `alignment.target_world_axis`
  (default `[0,0,-1]`), `planner.py:598-600`.
- `tolerance_deg` — `alignment.tolerance_deg` (default 3.0), `planner.py:601`.
- Reference implementation of the angle: `constraints.evaluate_axis_alignment_batched`
  / `compute_axis_alignment_angle_batched` (`constraints.py:275-316`).

Every external baseline's output must be converted to a `CandidateRecord`
(`source_type=external_*`) and pass the **same** A1 collision + hard validator
(`validators.evaluate_hard_constraints`) as the internal methods, dispatched via
the B0 seam (`methods.py` `external=True` + `runner`). No baseline gets a private
success definition.

This scalar `c(q)` is the *only* thing a projection or constraint-cost baseline
needs from our problem — it is cheap to expose to any external library as a
Python callback (angle → gradient by autodiff or finite difference).

---

## 1. Decision matrix

Bridge-cost axes: **URDF** = can it load the SR5 URDF as-is; **World** = can it
consume our per-request collision world (A1); **Constraint** = can it accept
`c(q)` as a manifold/TSR/cost; **Budget** = can it run under the uniform
compute-budget/timeout seam.

| Method (PDF) | Class | Open source? | License | Bridge cost to SR5+world+`c(q)` | Decision |
|---|---|---|---|---|---|
| **OMPL** ConstrainedStateSpace / ProjectedStateSpace | projection (manifold) | **Yes** [verify] | BSD-3 [verify] | **Med.** URDF→FK via our kinematics; `c(q)` as `ompl.base.Constraint.function`; world via our collision callback as a `StateValidityChecker`. Python bindings exist. | **Borrow for B5** (primary) |
| **GPMP2** (`GPMP2_2017`, arXiv:1707.07383) | constraint-cost trajopt (GP + factor graph) | **Yes** [verify] | BSD [verify] (gtsam-based) | **Med-High.** Needs a custom unary factor wrapping `c(q)`; gtsam/gpmp2 Python bindings exist but SR5 URDF→gpmp2 robot model + our world as SDF factor is real work. | **Borrow candidate for B3c** |
| **CHOMP** (ICRA 2009, non-arXiv) | constraint-cost trajopt (covariant gradient) | Partial (MoveIt/standalone) [verify] | varies | **Low-Med as reimpl.** CHOMP is ~200 lines: smoothness + obstacle + our `c(q)` penalty, covariant gradient descent on the joint trajectory. No heavy dep. | **Reimplement for B3c** (fallback, see §4) |
| **cpRRTC** (`cpRRTC_2025`, arXiv:2505.06791) | GPU constrained RRT-Connect | Research code [verify] | unknown | **High.** GPU CUDA kernels; SR5 URDF + our world + `c(q)` into their kernel path is a port, not a wrapper. | **Cite only** (related work); not a run baseline |
| **McVAMP** (`McVAMP_2026`, arXiv:2604.13323) | SIMD projection manifold | Research code [verify] | unknown | **High.** SIMD/vectorized C++; robot & world are compiled-in. | **Cite only** |
| **VAMP** (`VAMP_2023`, arXiv:2309.14545) | SIMD sampling (unconstrained) | **Yes** [verify] | BSD/MIT [verify] | **High + off-target.** Fast but unconstrained; adding `c(q)` projection = reimplementing McVAMP. Robot is compile-time templated. | **Cite only** |
| **pRRTC** (`pRRTC_2025`, arXiv:2503.06757) | GPU RRT-Connect (unconstrained) | Research code [verify] | unknown | **High + off-target** (unconstrained; overlaps B3a). | **Cite only** |
| **cuRobo** (`curobo_2023`) soft-cost | soft-constraint trajopt | Yes (already in-repo) | — | **In-repo.** re-port axis-weight mechanism. | **B2** (separate task) |

### Reading of the matrix
- The **only low-friction constraint-enforcing borrow** is **OMPL projection**
  (B5 primary): mature Python bindings, and the constraint is exactly a
  `Constraint.function` returning `c(q)` (or the full alignment residual).
- Every GPU sampling paper (cpRRTC / McVAMP / VAMP / pRRTC) is **cite-only**:
  robot + world are compiled/templated in, so bridging SR5 + a *per-request*
  world is a port. They are related work, not runnable adversaries here.
- The **classic constraint-opt** slot (B3c) has two viable routes: borrow GPMP2
  (medium-high) *or* reimplement a minimal CHOMP-with-constraint (low-medium,
  no external dep). §4 specifies the reimplement route as the guaranteed floor.

---

## 2. Risk posture — why the adversary cannot vanish

The review constraint on B1 is explicit: **B5 or B3c must ship**, else Table I's
"constraint-enforcing" cell is empty and E1 has no true hard-constraint opponent
(only B2's self-admitted soft-cost strawman and B3a's unconstrained floor).

Two independent guarantees, so a single failure does not empty the cell:
1. **B5 primary** = OMPL projection borrow (external dep, medium bridge).
2. **B3c floor** = minimal CHOMP-with-constraint **reimplementation** (§4) — no
   external dependency, so it cannot be blocked by repo availability, license,
   or build issues. This is the true safety net.

If OMPL integrates cleanly, we ship **both** and Table I is strongest. If OMPL
fights the SR5 URDF or the per-request world, we still ship B3c. Neither path
requires the GPU sampling papers.

---

## 3. Minimum-viable path — B5 (projection-based constrained planning)

**Goal:** a planner that treats the level axis as a *hard manifold constraint*
(project every sampled/interpolated config back onto `c(q) ≤ 0`), producing a
joint trajectory that goes through our A1 collision + hard validator.

### 3a. Primary route — borrow OMPL constrained planning (preferred)
Effort estimate: **~2–3 days** if bindings install cleanly.

- **Deps:** `ompl` with Python bindings (`pip`/conda `ompl` or build). **[verify]**
  BSD-3.
- **Robot/FK:** wrap our existing kinematics_fn — OMPL only needs, per state,
  the joint vector `q` (RealVectorStateSpace of dim = DOF).
- **Constraint:** subclass `ompl.base.Constraint(dim=DOF, co_dim=1)`; implement
  `function(q, out)` → `out[0] = c(q)` (the alignment residual in **radians**,
  not clamped by tolerance — projection needs the raw residual) and optionally
  `jacobian(q, out)` (else OMPL finite-differences). Wrap the space in
  `ProjectedStateSpace` (or `AtlasStateSpace`) + `ConstrainedSpaceInformation`.
- **Collision (A1):** `StateValidityChecker` that runs the config through the
  same collision query the planner uses (per-request world). Reuse the A1
  collision path via a thin callback, so B5 and internal methods share one
  collision truth.
- **Plan → CandidateRecord:** run RRT-Connect/PRM on the constrained space,
  interpolate the geometric path to the action horizon, resample to our dt,
  convert to `source_type=external_projection_ompl`, hand to the shared
  validator (which re-measures alignment, collision, continuity, kinematics).
- **Budget:** OMPL solve takes a wall-clock `timeout` → maps directly onto the
  uniform `timeout_sec` guard; K = number of independent solve attempts.

**Kill criteria (fall back to 3b/§4 if any hit):** bindings won't build in
`CuroboV2` after 1 day; SR5 URDF FK cannot be wrapped as a state→angle callback;
per-request world cannot be pushed into a StateValidityChecker.

### 3b. Fallback route — reimplement a minimal tangent-space projection planner
Effort estimate: **~3–4 days**, no external dependency.

Pure-Python/torch RRT on joint space with a projection step:
1. Sample `q_rand` in joint limits; extend from nearest tree node toward it.
2. **Project** the new config onto the manifold by Newton steps on `c(q)`:
   `q ← q − c(q) · J⁺` where `J = ∂c/∂q` (autodiff through kinematics_fn),
   iterate until `|c(q)| ≤ tol` or max-iters (drop sample on failure).
3. Reject if the projected config or the edge to its parent fails the A1
   collision query.
4. On reaching the goal manifold cell, extract path → resample to dt →
   CandidateRecord (`source_type=external_projection_min`).

This reuses machinery we already have (kinematics_fn, autodiff, collision query,
resampler) and depends on nothing external, so it is *always* achievable. It is
the concrete "minimal self-implemented projection planner" the review requires.

---

## 4. Minimum-viable path — B3c (classic constraint-optimization) — **guaranteed floor**

**Goal:** a trajectory optimizer that enforces the level axis as a hard
constraint via TSR/manifold penalty, distinct from B2's soft axis-weights.

### 4a. Preferred — borrow GPMP2
Effort estimate: **~3–5 days** if gtsam/gpmp2 bindings install.

- **Deps:** `gtsam` + `gpmp2` Python. **[verify]** BSD/other.
- Build an `ArmModel` from the SR5 URDF; obstacle factor from our per-request
  world as a signed-distance field; add a **custom unary factor** on each support
  state whose error = `c(q)` (level residual). Optimize the factor graph; read
  out the MAP trajectory → resample → CandidateRecord
  (`source_type=external_constrained_opt_gpmp2`).
- **Kill criteria:** gpmp2/gtsam won't build; URDF→ArmModel or world→SDF factor
  is more than ~2 days.

### 4b. Guaranteed floor — reimplement minimal CHOMP-with-constraint (no deps)
Effort estimate: **~2–3 days**, pure torch, **no external dependency**.

This is the true safety net — nothing external can block it. Optimize a joint
trajectory `Q ∈ R[T×DOF]` initialized by straight-line (or a rule seed) by
gradient descent on:

```
J(Q) = w_smooth · Σ‖q_{t+1} − q_t‖²                      (smoothness)
     + w_obs    · Σ obstacle_cost(q_t)                    (A1 collision field)
     + w_level  · Σ relu(c(q_t))²                         (HARD level penalty, large w_level)
```

- `c(q_t)` and its gradient come from autodiff through the existing
  kinematics_fn / `evaluate_axis_alignment_batched`.
- `obstacle_cost` reuses the A1 collision query (differentiable distance where
  available, else finite-difference penalty).
- Covariant/precomputed-smoothness gradient step (the CHOMP metric `A⁻¹`) is a
  one-time banded matrix — ~200 lines total.
- Anneal `w_level` up so the final trajectory satisfies `c(q_t) ≤ tol`; on
  convergence resample to dt → CandidateRecord
  (`source_type=external_constrained_opt_chomp`).

Because 3b and 4b share the same three ingredients we already own — **(i)**
autodiff `c(q)` gradient, **(ii)** A1 collision query, **(iii)** dt resampler —
implementing one substantially de-risks the other. **At least one of §3/§4 is
therefore guaranteed regardless of any external repository's availability**,
satisfying the B-phase hard exit ("B5 or B3c, never both absent").

---

## 5. Recommendation to B2/B3c/B5

1. **B5:** attempt OMPL borrow (§3a) first; hard-stop after 1 day of build
   friction and switch to the minimal projection planner (§3b).
2. **B3c:** implement the **minimal CHOMP-with-constraint (§4b) first** as the
   guaranteed floor (no dep risk); only attempt the GPMP2 borrow (§4a) if time
   remains and a stronger classic optimizer is wanted for Table I.
3. **GPU sampling papers (cpRRTC/McVAMP/VAMP/pRRTC):** related-work citations
   only — do **not** budget engineering time to run them as adversaries.
4. Every baseline reports through the B0 `external` seam and the shared A1 +
   hard validator; no baseline gets a private success definition.
