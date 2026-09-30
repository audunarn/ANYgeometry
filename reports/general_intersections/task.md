# General intersections — living implementation record

Authorized 2026-09-30: implement the user-approved plan, including exact elliptic
and cylinder/cylinder curves, schema 5 with old-schema readers, general material
arrangements, atomic batch preparation, charts and accepted ANYfem meshes.
Publication is outside this implementation. Historical evidence stays frozen.

Base: ANYgeometry 7e797791727752aec21ddd98d08daf8f0916e280.
Unrelated ECOSYSTEM_GUIDE.md edits and untracked local instructions/reports are
preserved. Work is solo; no parallel source owners have been created.

## Prospective acceptance

Use exact replay (three 5x5 plates, cylinder radius 1, height 4, origin
(2.5,2.5,-2), twelve sectors, spacing .5/1, including undo/replacement).
Accepted automatic linear ANYfem meshes at sizes .5 and .25, shared joint nodes,
source non-mutation, topology and ownership, save/load and feature references.
Also cover general angle and cylinder/cylinder analytic cases, all components,
repeated cuts/holes/junctions, insertion permutations, atomic cancellation and
rollback. Keep numerical and mesh-quality thresholds. Scientific completeness
claims require analytic/high-precision independent checks and retained evidence;
ordinary local debugging is an integrated Level A experiment, not acceptance.

## Current decision and bounded experiment

Confirmed exact replay on local source: StructuralPreparationError at face:3 /
face:52, 'imprint component does not map to one active face descendant'.
Equivalent reduced replay has a generator from (3.5,2.5,-2) to (3.5,2.5,-1):
plate endpoint boundary distances .5 and 1.5, cylinder distances zero. The curve
is valid but interior to the plate. One prior plate/plate connection succeeded.

Competing issues: missing exact curve representations; interior-ended traces;
repeated fragmentation with curved boundaries; consumer chart restrictions.
First slice adds exact curve evaluation/serialization and focused independent
oracles. Then reproduce topology preparation before running complete meshing.
Do not infer acceptance from a successful isolated slice.

## Stages

1. Exact curves/schema — implemented locally; broader acceptance pending.
2. Arrangements/batch preparation — pending.
3. Mesher/FEM integration — pending.
4. Qualification/compatibility and installed package gates — pending.

## Stage 1 development evidence

New immutable curves, schema 5 codec with schema-4 preservation, root exports,
edge evaluation/tangents/length/bounds/projection/splitting, copy/reverse and
affine transforms are implemented. Sweep copies retain the analytic definition.
Support intersections isolate analytic finite-height, angular and branch events.
Rational Sturm isolation counts all distinct polynomial roots, including repeated
and narrowly separated roots. Display samples never exclude a root or branch.

Focused codec/automation/editing tests: 118 passed. Full existing kernel run:
1000 passed, one introduced missing-import failure; the import was repaired and
all 26 affected surface-operation tests pass in a 43-test targeted run including
the new support/root/curve tests. Full run retained in stage1-regressions.log.
This is local development evidence, not hosted matrix or meshing acceptance.

The next bounded experiment builds per-face arrangements from original operand
intersections. A useful result must retain exact curved edges, mark decomposition
seams separately, share high-valence junctions and preserve material. Existing
boundary-only imprint and sequential consumer restart loops remain unresolved.
