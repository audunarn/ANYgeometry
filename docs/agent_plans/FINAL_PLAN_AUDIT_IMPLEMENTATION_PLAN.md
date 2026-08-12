# Final-plan structural integrity implementation plan

## Objective and source inputs

Close the bounded structural-integrity holes discovered by the final plan audit:
validate every persisted Attachment source kind, enforce compatible attachment
semantics, and add missing hostile lifecycle/serialization and topology-ordering
coverage.  This plan derives from:

- `C:\Users\AudunArnesenNyhus\Downloads\ANYgeometry_kernel_update_codex_plan.md`
- `C:\Github\ANYgeometry\docs\KERNEL_UPDATE_OVERVIEW.md`
- the schema-4 persistence decision and the final read-only audit findings

## Repository and branch

- Repository: `C:\Github\ANYgeometry`
- Working branch: `native_hybrid_mesher`
- Current branch head at registration: `f2d7793d7d32a6dcd772c7ed8701aca11b459288`
- Authoritative kernel base: `19f16746ceee76838b7df9d08a95028699e3a738`

## Owned and excluded paths

Owned paths:

- `src/anygeometry/structural.py`
- `src/anygeometry/model.py`, limited exactly to the three membership guards in
  `_validate_structural_changed` that add an Attachment source/target vertex,
  edge, or face to the local known-ID set only when that entity is live
- `tests/test_structural_lifecycle.py`
- `tests/test_serialization_gap_closure.py`
- `tests/test_curved_split_contract.py`, a dedicated new file for isolated
  curved-split contract tests

Explicitly excluded:

- every other part of `src/anygeometry/model.py`, including evaluator/projection
  implementation, and all of `src/anygeometry/evaluation.py`
- `tests/test_mesher_model_contract.py` and `tests/test_surfaces_operations.py`
- intersection/imprint and policy implementation
- serialization schema/codec implementation unless a test exposes a loader bug
- benchmarks, package metadata, release report, and unrelated documentation

These exclusions are disjoint from `evaluator_hostile_review`.

## Structural invariants and schema-4 constraints

- Every Attachment source and target must resolve to a live entity of the
  declared kind during mutation validation and schema-4 load.
- Member-oriented AttachmentKind values require a member source; vertex-oriented
  values require a vertex source.  Generic intentionally-disconnected intent may
  use another currently supported source kind only when its referenced entity is
  live.
- Invalid public attachment creation must roll back atomically, preserve allocator
  high-water semantics, and not advance revision for net-zero state.
- Schema 4 remains the writer format; no field, enum, version, checksum, or
  migration behavior changes in this bounded task.
- Old v1-v3 migrations retain their accepted UNVERIFIED/non-certifying meaning.
- Radial face-use order must be deterministic; curved splits must preserve exact
  authoritative support or fail atomically when unsupported.

## Milestones and definition of done

1. Map all existing Attachment source-kind/kind combinations in code and tests.
2. Add fail-closed source existence and semantic compatibility validation.
   The local model validator must not mask a missing geometry source by adding
   its absent ID to the known-ID slice; this is the only approved model hunk.
3. Add public mutation rollback and canonical schema-4 hostile-load regressions.
4. Add direct deterministic `radial_face_uses` coverage.
5. If isolated from active editors, add cone support-preservation and unsupported
   curved split atomicity tests; report rather than patch another owner's module
   if a defect is found.
6. Run focused structural/serialization/surface tests only and report exact
   results, changed paths, compatibility impact, and unresolved risks.

Done means dangling or semantically incompatible attachments cannot commit or
load, valid existing workflows remain green, and no schema or public signature
has changed.

## Verification and performance lease

Allowed without a lease:

- focused individual tests in the three owned test files
- `python -m py_compile src/anygeometry/structural.py`
- static searches and diff checks

Anticipated lease-gated work: full-suite execution, stress/scaling, benchmarks,
builds, and package qualification.  The agent will not run them; the lead will
request a fresh lease for final qualification.

## Dependencies, risks, and handoff

- Depends on the accepted schema-4 Attachment representation and local structural
  validation path in GeometryModel.
- Compatibility risk: historical tests may have used member-specific kinds with
  generic sources.  The agent must report such a conflict before broadening or
  weakening the invariant.
- The agent must not edit evaluator code or intersection policy code.
- Handoff must list exact accepted source-kind/kind combinations and any typed
  unsupported combinations so downstream consumers do not infer semantics.
