# ANYgeometry: more general geometry, and speed (deep dive, 2026-10-13)

Status: study and proposed plan. Nothing in `src/` was changed. Measured on local
`main` 0a3a602 (a `git archive` snapshot), Python 3.14.2, NumPy 2.4.6, Windows,
on a machine other sessions were also using. Scripts are in `deep_dive_20261013/`.

## 1. Bottom line

1. **Speed.** Very little of the time goes to actual arithmetic. In five
   representative workloads, NumPy's compiled code accounts for 8-13% of self time.
   The rest is the Python interpreter: 5-21 million function calls per run at about
   0.4-0.6 µs each, and the many small NumPy calls on 3-vectors are the worst part.
   The time is also spread out: no single function takes more than 7%.
2. On one typical hot routine (`Cylinder.local_uv`):

   | Variant | ns/point | vs current |
   |---|---:|---:|
   | Current (NumPy on 3-vectors, per call) | 7,621 | 1x |
   | Same maths in plain Python floats | 527 | 14x |
   | C, called once per point from Python | 170 | 45x |
   | NumPy vectorised over a batch | 64 | 120x |
   | C over a batch (one call) | 26 | 290x |

   Results agree to 1e-12. A C rewrite only pays off when a whole loop moves into
   C. Replacing single functions one at a time buys roughly 3x more than plain
   Python floats, not 100x.
3. Only part of the time is geometry. Grouping self time by module, curve and
   surface evaluation plus arrangement and root-finding take **20-52%** of the
   intersection workloads. Model bookkeeping takes **30-55%** (validation, content
   checksums made by building JSON and hashing it, identity checks, serialization),
   and 76% of document round trips. So a native geometry core alone is capped at
   about 1.5-2x overall. The bookkeeping needs a redesign as well (cached and
   incremental digests and validation), not just a faster language.
4. **Generality.** Each new curved case has been added as its own exact curve
   family (`EllipticArc`, `CylinderIntersectionCurve`, `QuadricIntersectionCurve`,
   `BezierQuadricCurve`), each with its own charts, events, supports and schema
   record. The number of special cases grows with the square of the number of
   surface types. There is no B-spline/NURBS, sphere, torus, surface of
   revolution, sweep or offset surface, and no freeform-to-freeform intersection.
   Cubic-by-cubic non-parallel walls are refused. Ship hulls and other imported
   CAD are mostly NURBS, so this is the main gap for "all kinds of geometry".
5. **Recommendation:**
   - (a) Do two cheap pure-Python passes first: scalar 3-vectors in hot paths, and
     cached/incremental content digests. Both stay installable, need no decision on
     native code, and should give about 1.5-2.5x.
   - (b) In parallel, build a native C++17 core, `anygeometry._core`, using the
     CPython stable ABI. That gives one wheel per platform for Python 3.11+, the
     same model as ANYmesher's `_native`. Its interface takes whole batches, not
     single points.
   - (c) Make geometry general through **one common surface interface plus one
     general intersection-curve type**, with NURBS as the universal freeform type.
     Write new types natively from the start so nothing is written twice. The
     current exact families become fast special cases behind the same contract.

## 2. Measurements

Wall time without the profiler (best single run, noisy machine):

| Workload | What | Time | Faces after |
|---|---|---:|---:|
| pipes6 | 6 eight-segment cylinders through a plate, plan+apply | 2.49 s | 103 |
| rings8 | 1 cylinder, 8 ring plates, plan+apply | 1.80 s | 88 |
| grid16 | 8x8 crossing web plates on a base plate | 1.55 s | 225 |
| tilted8 | 8 mutually crossing tilted plates | 4.10 s | 914 |
| doc60 | 3,600-plate grid `to_dict` → `from_dict` → `validate_topology` | 1.47 s | 3,600 |

Self time by layer, under cProfile:

| Workload | ANYgeometry Python | NumPy Python wrappers | NumPy C | builtins | `fractions` | calls |
|---|---:|---:|---:|---:|---:|---:|
| pipes6 | 52% | 21% | 13% | 9% | 2% | 7.3 M |
| rings8 | 52% | 20% | 13% | 10% | 3% | 5.5 M |
| grid16 | 49% | 16% | 10% | 13% | 6% | 8.9 M |
| tilted8 | 47% | 20% | 11% | 12% | 5% | 21.2 M |
| doc60 | 51% | 24% | 8% | 13% | – | 8.7 M |

Self time by kind of work (NumPy and builtin time is charged to the calling
ANYgeometry module):

| Workload | Geometry kernels | Model/topology/bookkeeping | Unattributed library |
|---|---:|---:|---:|
| pipes6 | 51% | 31% | 18% |
| rings8 | 52% | 30% | 17% |
| grid16 | 37% | 41% | 21% |
| tilted8 | 21% | 55% | 24% |
| doc60 | 1% | 76% | 23% |

Typical hot spots: `Cylinder.local_uv`, `EllipticArc.evaluate/_parameters/bounds`,
`arrangement_geometry.LinePath.evaluate` and `_point_parameters` (which calls
`lstsq`), `np.linalg.norm` (up to 0.5 M calls), `definition_binding._definition_value`
(up to 0.27 M calls) and `json` encoding for checksums, `identity.validate_local_id`
(0.35 M calls), `model._validate_planar_straight_quad_geometry` (which uses
`np.cross`), and `fractions.Fraction` (up to 0.9 M calls).

Earlier measurements that are consistent with this: the 2026-10-01 perf-hunt
(`claude/perf-hunt-046`, unmerged) got 2x on cylinders through a plate with Python
changes alone. The 2026-10-07 sweep got 12-30% from scalar `LinePath` plus `_cross3`.
The 64-member crossing grid spent 25% in `arrange_material`, with 1.2 M `norm` calls.

## 3. Diagnosis

### Speed

- **Many tiny NumPy calls.** Each `np.linalg.norm`, `np.cross`, `lstsq` or
  `np.array(...)` copy on a 3-vector costs 1-5 µs of dispatch for nanoseconds of
  arithmetic. Validating at every entry point (`_vector3` copies the array and runs
  `isfinite` on every call) adds more.
- **Requiring bit-identical output makes speedups contorted.** `LinePath.evaluate`
  now carries a 20-line fast path whose only job is to repeat the vectorised route's
  floating-point operation order. This rule will block a native core unless it is
  replaced (decision D1).
- **Content checksums** are made by recursively expanding the definition, encoding
  it as JSON and taking SHA-256, and this is repeated per call. Immutable definitions
  could cache their digest once.
- **Exact arithmetic** uses Python `Fraction`: correct, but about 1 µs per operation.
  Adaptive floating-point expansion arithmetic (Shewchuk-style) with an exact
  fallback gives the same answers at close to plain-float speed.
- **The data layout is one Python object per entity,** so whole-model passes
  (validation, serialization, spatial checks) cannot be vectorised or handed to
  native code without first converting the data.

### Generality

- The **query** engine is already fairly general (sampling and contouring, Newton,
  certificates). The **topology-changing (imprint/apply)** route is the part that is
  written family by family, and that is where generality is lost.
- Every exact curve family repeats the same pieces: supports, branch chart, events,
  bounds, serialization, transforms and attachment remapping. Each new surface type
  needs pair handling against every existing type.
- Missing representations: multi-span and rational B-splines (NURBS) for curves and
  surfaces; sphere, torus and surfaces of revolution; sweeps; offset surfaces.
  `RuledSurface` and `CoonsSurface` interpolate sampled polylines, so they are
  approximations, not exact geometry.
- Known refusals (from the 2026-10-02 study) that a general method would remove:
  cubic×cubic non-parallel walls, tangential or overlapping branches on one support,
  and the engine-wide "second operation splitting a joint edge needs a parameter
  remap".

## 4. Target architecture

### 4.1 One contract for geometry

**Support interface** (every surface type exposes these capabilities; a missing one
means a typed refusal):

- parametric evaluation and derivatives up to order 2, in batches;
- point inversion and closest point, in batches;
- a guaranteed enclosure over a sub-domain: bounding box, and a normal cone where
  available;
- an implicit equation when the surface is algebraic (plane, quadrics, torus, and
  extrusions of polynomial curves);
- an exact persisted definition, kept separate from derived caches.

**One general intersection-curve type**, defined by (support A, support B, a
certified branch), with derived witnesses: a curve in each surface's own parameter
space plus a 3D B-spline approximation with a proven error bound. Evaluating the
witnesses does not define the curve, the same rule the current families follow.
The existing families stay, as exact fast special cases that produce the same
contract. Old schema records keep loading (patch-only rule).

**Algorithms by pair type:**

- *Implicit × parametric* (plane, quadric or torus against NURBS or Coons):
  substitute the surface into the implicit equation to get a plane curve in the
  freeform surface's parameter space. Its topology is found by certified 2D
  subdivision (Bernstein bounds, interval Newton/Krawczyk to isolate pieces,
  marching between proven-regular boxes). This builds on the Bernstein-basis lessons
  from `branch_algebra.py`.
- *Parametric × parametric* (NURBS × NURBS): subdivide with a loop-detection test
  (Sederberg normal cones) until each box holds at most one branch, then march with
  certified step control. Tangential contact and overlap fail closed with typed
  reasons, as they do today.

**NURBS first**: multi-span and rational curves, then surfaces. Bezier extraction
by knot insertion reuses the existing Bezier tools; sphere and torus come as exact
rational NURBS plus their implicit form.

### 4.2 Native core

- `anygeometry._core`: C++17 using the CPython C API (no pybind or nanobind, so no
  build dependencies beyond setuptools, as in ANYmesher), built against the
  **stable ABI** (`Py_LIMITED_API` 3.11). That means one wheel per OS and
  architecture covers 3.11, 3.12, 3.13, 3.14 and later versions. It also avoids the
  "no wheel yet for the new Python" problem recently seen with h5py on 3.15.
- **The interface takes batches**: arrays in, arrays out (curve/surface evaluate,
  invert, closest point, enclosure, root isolation, UV arrangement). Python handles
  orchestration, policy, typed results and identity, not inner loops.
- **No runtime dependencies.** Header-only, permissively licensed internals only
  (own expansion arithmetic; no GMP, which is LGPL; no OCCT).
- **Backend selection is explicit and reported**, mirroring
  `ANYMESHER_DISABLE_NATIVE`/`ANYMESHER_REQUIRE_NATIVE`. The pure-Python route stays
  as (1) the installation fallback when built from source without a compiler and
  (2) the differential-testing reference for the ported routines. Whether release
  wheels *require* the native core is decision D3.
- **Determinism.** Disable FMA contraction (`/fp:precise`, `-ffp-contract=off`).
  Results must not depend on the last bit of `sin`, `cos` or `atan2`: platform libm
  already differs today, because Python's `math` uses it too. Classification and
  topology must be certified identical; coordinates must be equal within a stated
  error bound (D1).
- **Builds:** cibuildwheel matrix for win-amd64, manylinux x86_64/aarch64,
  musllinux x86_64, macOS arm64 and x86_64, plus an sdist. ANYmesher's publish
  workflow is the template; OIDC Trusted Publishing stays as it is.
- **Later lever:** native batch routines that release the GIL make thread
  parallelism inside one model possible, including on free-threaded 3.14t. ANYmesher
  measured threads as *slower* than serial for that reason, because the work was
  bound to Python.

### Language options

| Option | Verdict |
|---|---|
| **C++17 + CPython stable ABI** | **Recommended.** Matches ANYmesher; MSVC 18 BuildTools is already installed; well-known numerics libraries; one wheel per platform. Risk: memory safety, handled with tests, sanitizers in CI, and fuzzing against the Python reference. |
| Rust + PyO3/maturin | Strong alternative: memory-safe, easy abi3 wheels. Adds a second native toolchain to the ecosystem, and no toolchain is installed here. Choose this if memory safety outweighs consistency. |
| Cython / mypyc | Gradual 2-4x on scalar Python. Keeps the per-object model, so it doesn't reach batch speed. Possible stopgap; not a destination. |
| Numba | No. Large dependency (llvmlite), JIT warm-up, lags new Python versions. |
| Pure-Python scalar rewrite | Do it anyway (step S1): costs nothing for installation and shrinks what has to be ported. |

## 5. Plan (slices, acceptance criteria, dependencies)

| Id | Slice | Output | Acceptance | Depends on |
|---|---|---|---|---|
| G0 | Benchmark corpus | Committed workloads (the five above, plus cone/Bezier-wall, 32/64-member grid, the barge preparation via ANYmesher) with digests and work counters | Repeatable digests; timings with spread recorded | – |
| G1 | Equivalence rule | Decision record (D1) | Owner approves | – |
| S1 | Scalar 3-vectors in hot paths (`local_uv`, `EllipticArc.*`, `_point_parameters` closed form instead of `lstsq`, `norm`/`cross`/`dot`; validate at API boundary only). Revisit the unmerged perf-hunt branch 36d534c | Python only | Plan/apply digests identical on G0; ≥1.5x on pipes6/rings8; full suite unchanged | G0 |
| S2 | Bookkeeping: digests cached on immutable definitions (same checksum format), fewer whole-model `to_dict`/`validate_topology` passes, cheaper identity checks | Python only | Same documents and checksums; doc60 ≥2x; bookkeeping share of tilted8 halved | G0 |
| N0 | Native scaffold: `_core` abi3 extension, environment switches, backend reporting, cibuildwheel CI, installed-wheel smoke test, sdist without a compiler falls back to Python | Build/CI | Wheels on all matrix platforms; installed import outside the tree; fallback reported | D2, D3, D4 |
| N1 | Batch evaluation/inversion/closest-point routines for current curve and surface types | Native + Python reference | Differential tests (random + adversarial) within the D1 bound; ≥20x on batches | N0, G1 |
| N2 | Robust predicates: adaptive expansion arithmetic with an exact fallback, replacing `Fraction` in hot paths | Native | Exact agreement with a `Fraction` oracle on a degenerate-case corpus | N0 |
| N3 | Native UV arrangement (core of `arrangement_geometry`/`material_arrangement`) behind a flat array interface | Native | Identical topology and certificates on G0; arrangement share ≥5x faster | N1, N2 |
| N4 | Root isolation and interval enclosures (Sturm/Bernstein) | Native | Same root counts as the exact rational oracle | N2 |
| A1 | Common surface interface + general intersection-curve contract; map existing families onto it without behaviour change | Design doc + adapters | No output change on G0; draft of a new schema record | G1 |
| A2 | B-spline/NURBS curves (multi-span, rational): evaluation, derivatives, knot insertion/Bezier extraction, transforms, serialization | Native + small Python reference | Independent check: de Boor against Bezier extraction, published textbook examples, round trip | A1, N0 |
| A3 | NURBS surfaces; sphere, torus, surface of revolution (exact rational + implicit) | Native | Same kinds of checks as A2; analytic area/normal checks | A2 |
| A4 | Implicit × parametric certified intersection, query first, then plan/apply | Native + Python | Independent oracles (grid minima + Newton, swapping the operand order, area conservation); typed refusals | A3, N4 |
| A5 | NURBS × NURBS certified intersection (loop detection + marching) | Native | Same oracles; tangential and overlap cases fail closed | A4 |
| A6 | Engine-wide parameter remap for joint edges split by a second operation | Python | Removes the existing refusal for all curve families | A1 |
| A7 | Hand-off for STEP B-spline mapping (owned by ANYfileIO / ANYfileio-occt) | Hand-off note | Consumer contract agreed | A3 |

Parallel tracks: S1/S2 (pure Python) ∥ N0 ∥ A1. New geometry types (A2+) are
written natively from the start.

Rough expectations (estimates, not measured beyond the microbenchmark):
S1+S2 about 1.5-2.5x; adding N1-N4 removes most of the 20-50% geometry share
(another 1.5-2x); N3 together with a struct-of-arrays topology store is needed for
5-10x on large connected models.

## 6. Decisions for the owner

- **D1 Equivalence rule** for speedups: bit-identical (as today) or certified-
  identical topology and classification with coordinates within a stated bound. A
  native core is not practical under the bit-identical rule.
- **D2 Language:** C++17 (recommended) or Rust.
- **D3 Native required or optional:** recommended to keep it optional, with a
  reported fallback, during the transition. Decide later whether release wheels
  require it.
- **D4 Platform matrix** for wheels (proposal in 4.2), plus the CI time budget.
- **D5 First geometry target and driving case:** NURBS curves only, or hull-type
  NURBS surfaces? Which real model is the acceptance case?
- **D6** Revive or drop the unmerged perf-hunt branch (36d534c) as input to S1.
- **D7 Roles and review:** who implements each track, and the Codex review setup.

## 7. Risks and open questions

- Certified freeform×freeform intersection (A5) is close to research territory;
  tangential and near-tangential contact will stay as typed refusals for some time.
- Two implementations (Python reference + native) can drift. Differential tests in
  CI on every platform are mandatory, and the reference covers only the ported
  routines.
- Floating-point results differ across platforms (libm, compilers); D1 must allow
  for that.
- Scope creep towards solid CAD (Booleans on solids, fillets) – out of scope unless
  decided.
- Patch-only versioning: new schema records must keep backward reading; the
  released 0.4.5 writes schema 5, and local `main` already has a schema-6 record.
- The machine was shared during measurement; timings are indicative, while call
  counts and shares are robust.
