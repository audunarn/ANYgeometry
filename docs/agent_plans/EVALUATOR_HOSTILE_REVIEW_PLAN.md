# Evaluator hostile-review implementation plan

## Objective and source inputs

Close the remaining mesher-facing batch face-projection performance gap without
changing the public evaluator signatures or weakening qualified closest-point
semantics.  This bounded plan derives from:

- `C:\Users\AudunArnesenNyhus\Downloads\ANYgeometry_kernel_update_codex_plan.md`
- `C:\Github\ANYgeometry\docs\KERNEL_UPDATE_OVERVIEW.md`
- the hostile evaluator findings recorded in the active Codex task
- the accepted ANYmesher contract for deterministic vectorized evaluation

## Repository and branch

- Repository: `C:\Github\ANYgeometry`
- Working branch: `native_hybrid_mesher`
- Current branch head at registration: `f2d7793d7d32a6dcd772c7ed8701aca11b459288`
- Authoritative kernel base: `19f16746ceee76838b7df9d08a95028699e3a738`

## Owned and excluded paths

Owned paths:

- `src/anygeometry/model.py`, limited to batch projection/evaluator helpers
- `src/anygeometry/evaluation.py`, only if wrapper behavior requires alignment
- `tests/test_mesher_model_contract.py`
- `tests/test_surfaces_operations.py`, only for evaluator regressions

Explicitly excluded:

- serialization, identity, structural ownership, intersection/imprint, audit,
  overlap, packaging, benchmarks, and documentation outside this plan file
- `src/anygeometry/structural.py`
- `tests/test_structural_lifecycle.py`
- `tests/test_serialization_gap_closure.py`

These exclusions keep this work disjoint from `final_plan_audit`.

## Public-contract constraints

- Preserve `evaluate_face_many`, `face_derivatives_many`, `face_normal_many`,
  and `project_to_face_many` signatures and result shapes.
- Preserve strict model-local identity validation and immutable records.
- A supplied `initial_uv` may guide convergence but must never worsen the
  deterministic closest qualified result.
- Plane, Cylinder, and Cone projection must use batch-oriented built-in paths.
- Build face trim UV data at most once per batch call.
- RuledSurface, CoonsSurface, and custom SurfaceProtocol implementations retain
  deterministic qualified fallback behavior; unsupported cases fail closed.
- No schema, version, or mutation-policy changes.

## Milestones and definition of done

1. Inspect the current landed projection and derivative fixes and establish a
   focused regression baseline.
2. Implement vectorized/batched Plane, Cylinder, and Cone candidate generation,
   deterministic seed comparison, and one-time trim preparation.
3. Preserve shaped-empty behavior, large-coordinate robustness, and exact
   output ordering.
4. Add focused tests for scalar/batch equivalence, optional seed behavior,
   trim handling, empty inputs, and large coordinates.
5. Compile owned modules and run only the focused evaluator tests.
6. Report changed paths, exact commands/results, complexity implications,
   limitations, and integration notes to the lead agent.

Done means all focused tests pass, public signatures are unchanged, no per-point
trim reconstruction remains on built-in batch paths, and no excluded file was
edited.

## Verification and performance lease

Allowed without a lease:

- `python -m pytest -q tests/test_mesher_model_contract.py tests/test_surfaces_operations.py`
- `python -m py_compile` for owned modules
- static search and diff checks

Anticipated lease-gated work: full-suite execution, profiler runs, large batch
stress/scaling measurements, benchmarks, builds, or package qualification.  The
agent will not run those; the lead will request a fresh ecosystem performance
lease during final qualification.

## Dependencies, risks, and handoff

- Depends on the current Face support-versus-parameterization contract and
  TolerancePolicy behavior.
- Main risk is changing closest-point semantics at trims or cone/cylinder
  boundaries; tests must compare batched results with qualified scalar results.
- Shared `model.py` is a conflict hotspot.  The agent owns only the evaluator
  block and must stop if another active editor enters it.
- Handoff must identify remaining scalar fallback families explicitly and must
  not claim measured scaling without lease-controlled evidence.
