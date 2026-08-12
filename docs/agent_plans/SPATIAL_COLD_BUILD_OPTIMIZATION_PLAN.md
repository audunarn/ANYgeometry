# Spatial cold-build and lifecycle qualification plan

## Objective and source inputs

Resolve the final performance blocker without changing query correctness or
the public spatial API. The qualification benchmark currently combines a
first-time maintained-index build, Python allocation tracing, and the actual
local query/audit in one number. This plan both removes avoidable cold-build
mutation work and records cold versus steady-state lifecycle costs separately.

Inputs:

- `C:\Users\AudunArnesenNyhus\Downloads\ANYgeometry_kernel_update_codex_plan.md`
- `C:\Github\ANYgeometry\docs\KERNEL_UPDATE_OVERVIEW.md`
- `benchmarks/wave_gap_closure_qualification.json`
- boss verdict `[FINAL QUALIFICATION: CHANGES REQUIRED]`

## Repository and branch

- Repository: `C:\Github\ANYgeometry`
- Branch: `native_hybrid_mesher`
- Registration head: `f2d7793d7d32a6dcd772c7ed8701aca11b459288`
- Authoritative base: `19f16746ceee76838b7df9d08a95028699e3a738`

## Root-cause and lifecycle separation

- A 150 x 150 grid contains 22,801 vertices, 45,300 edges, and 22,500 faces:
  90,601 indexed geometry records in total.
- Its first `spatial_candidates(..., kinds=("face",))` call invokes
  `GeometryModel._spatial()`, computes every conservative bound, and formerly
  replayed 90,601 incremental `AABBTree.insert` operations, including sibling
  searches, ancestor refits, and rotations. The benchmark wrapped this entire
  cold build in `tracemalloc`; the returned candidate set itself contained
  only four faces.
- The 10k-face model's strict audit builds a separate independent tree and does
  not publish `model._spatial_index`. Consequently the later changed-region
  audit also cold-builds the maintained 40,401-record tree under tracing before
  classifying 1,348 accounted candidates.
- The 77.319-second full audit disabled tracing, so it is not directly
  comparable to the 99.312-second cold/traced changed-region measurement.
- Neither current record measures a second steady-state query or an edit made
  after the maintained index exists.

## Owned and excluded paths

Owned:

- `src/anygeometry/spatial.py`, initial bulk construction only
- `tests/test_spatial.py`, deterministic correctness/work-count coverage
- `benchmarks/kernel_benchmarks.py`, lifecycle measurement separation only
- `benchmarks/README.md`
- `KERNEL_UPDATE_REPORT.md`, final evidence only

Excluded:

- GeometryModel mutation/invalidation semantics and `_entity_bounds`
- strict-audit classifications, tolerance, intersection, structural topology,
  serialization codecs/schema, evaluator APIs, and sibling repositories
- changes to any public spatial/query signature or return type

## Public-contract and correctness constraints

- Preserve exact conservative leaf AABBs and private fat padding.
- Preserve deterministic results independent of input order.
- Preserve strict height balance and every `validate()` invariant.
- Preserve incremental insert/update/remove behavior after the initial build.
- Preserve transaction read-your-writes, rollback invalidation, net-zero
  reconciliation, exact-box filtering, and brute-force overlap equivalence.
- A missing or invalid item/duplicate key must still fail closed.
- Benchmark correction must expose, not hide, the cold build cost.

## Registered implementation

Replace constructor-time incremental replay with a deterministic spatially
clustered balanced bulk build. Recursively split nearly equal leaf sets at the
median of the widest centroid axis, tie-breaking by axis and `SpatialKey`.
This constructs an AVL-valid tree without sibling searches, rotations, or
ancestor refits. Later mutations continue through the existing incremental
paths.

At registration time the lead had already landed, but not executed, the
bounded draft in `src/anygeometry/spatial.py` and its focused assertions in
`tests/test_spatial.py`. This concurrent governance deviation is disclosed;
the draft remains frozen pending verdict.

## Benchmark correction

Record separately:

1. cold maintained-index materialization;
2. a repeated steady-state local query with the same candidate oracle;
3. local edit and changed-region audit after the maintained index exists;
4. candidate/node/leaf/update counts for each applicable phase.

Retain allocation-traced and normal untraced timings where meaningful. Do not
rename a cold build as a local query or compare traced and untraced values as
equivalent wall-clock gates.

## Milestones and definition of done

1. Obtain approval for this registered plan and the disclosed draft.
2. Run focused small spatial correctness tests and fix only owned-path defects.
3. Prove constructor insertion-order independence, brute-force overlap oracle,
   AVL invariants, exact filtering, and zero constructor refit/rotation work.
4. Update the benchmark lifecycle records without changing workload geometry.
5. Under a fresh exclusive lease, run the full spatial/full repository suite
   and renewed smoke/qualification benchmark exactly once.
6. Report cold, steady, changed-region, candidate, memory, and baseline results
   truthfully; no threshold tuning after measurement.

Done requires all correctness tests green, no regression in candidate
accounting/invalidation, a material cold-build improvement, and a steady-state
local query/audit record that is proportional to the changed region.

## Focused and lease-gated verification

Allowed after plan approval without a performance lease:

- individual small tests in `tests/test_spatial.py`
- `python -m py_compile src/anygeometry/spatial.py benchmarks/kernel_benchmarks.py`
- static diff/complexity inspection

Requires a fresh lease:

- full `tests/test_spatial.py` scaling cases
- full repository suite
- smoke or qualification benchmarks
- profiler/stress measurements

## Package follow-up

Package failure was environmental: isolated build attempted a prohibited PyPI
fetch for `wheel`. Under a later lease, the exact offline/no-network commands
proposed are:

```powershell
python -m build --no-isolation
python -m twine check dist\anygeometry-0.2.1.tar.gz dist\anygeometry-0.2.1-py3-none-any.whl
```

They use the already installed build backend and name only freshly expected
0.2.1 artifacts. A clean installed-wheel smoke remains a separate registered
qualification step.

## Dependencies, risks, and handoff

- Main risk is a spatially poor or non-deterministic bulk hierarchy. Query and
  brute-force oracle tests plus identical reversed-input diagnostics guard it.
- Constructor diagnostics retain the number of inserted leaves while reporting
  zero incremental refits/rotations; this distinction must be documented.
- No performance claim is accepted from the draft until lease-controlled
  measurement completes.
- Handoff includes changed paths, exact commands/results, JSON evidence, known
  limitations, and whether the blocker is resolved or remains open.
