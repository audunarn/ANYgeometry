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
2. Arrangements/batch preparation — local implementation in progress.
3. Mesher/FEM integration — implemented locally; expanded admission pending.
4. Qualification/compatibility and installed package gates — in progress.

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

## Stage 2 development decision

The shared half-edge arrangement handles planar line/ellipse cuts, concave
material, closed loops and interior stubs; analytic predicates split all shared
junctions before cycles are built. Tangent ordering uses derivative/curvature
jets. Construction seams are tagged separately from physical traces. Original
native rectangular Cylinder charts use analytic angle/height interval clipping.

Focused stage-2 tests: 8 passed, including one canonical six-edge junction from
three crossing plates, positive-area-overlap refusal, cancellation and budget
refusal. Normal transactions retain allocator high-water marks on rollback;
batch application therefore stages on an identity-preserving detached clone,
then publishes a complete topology snapshot. The cancellation test now restores
the complete checksum and allocator state. Snapshot ChangeSet reporting also
includes newly committed geometry replacement lineage.

Exact replay creates 55 current faces (IDs retain gaps from cylinder undo).
The current generator honors spacing by generating thirteen angular sectors;
the requested twelve-sector parameter and the entire undo/replacement replay
are preserved. Planning all original operands succeeds with 39 affected face
arrangements. Application exposed duplicate numerical split witnesses on an
existing arc; canonical event de-duplication is now under test. All failed logs
are retained. Accepted ANYfem meshes, general branch arrangements, members,
idempotence and installed matrix evidence remain outstanding.

## Consumer integration experiment

The face replay now applies with 104 faces and 56 joint edges, topology valid.
Member axes participate in the same read-only plan and atomic application;
local tests verify parameter ranges, six-way shared vertices, interior contacts,
source non-mutation and fresh-plan idempotence. General Plane/Cylinder chart
bindings and support evaluation are implemented in development.

Prospective integration question: does complete batch preparation expose a
topology/trim mismatch, or a downstream mesher chart restriction? Replace the
ANYfem sequential loop, then run its existing intersection tests and the exact
automatic linear replay at .5 before expanding meshing fixtures. The first
consumer run had 1 pass and 10 failures (75 seconds), retained in
fem-integration-first.log. Most block in arrangement/preparation, including
misassigned negative hole cycles and duplicate affine edges; two old MPC tests
instead reach meshes with the newly requested shared-node connection. A generic
nonplanar Coons face and legacy observable diagnostics still need compatibility
work. The first exact .5 mesh run failed chart validation of a degenerate event;
no mesh acceptance is claimed. Refine the individual failed arrangement and
rerun affected cases, retaining numerical and quality thresholds.

Chart validation of the 104-face replay passes after merging only subintervals
whose conservative world bounds fit the existing length tolerance. The measured
material area is 100.13274122871833, matching the independent formula 75+8*pi.
Creating source Sheet owners before fragmentation fixes artificial multi-Sheet
seams. Proven rectangular cylinder/plane children now have their own parameter
extents. This removes the earlier aspect-ratio count of 363; the latest .5 mesh
still fails existing angle and element-growth gates (mesh-replay-050-authored-
frames.log). No tolerance or quality policy has been relaxed.

Next diagnostic is bounded to the first rejected native candidate: retain its
actual cells, source-face associations and quality report, then abort without
publication. Competing explanations are inadequate joint-boundary station
gradation and an unavoidable acute decomposition corner. The cell coordinates
and edge incidence will decide whether to change seeding or decomposition;
another unmodified full replay would not resolve that uncertainty.

## Accepted exact replay and boundary synchronization

The rectangular-cylinder proof now admits intermediate side vertices rather
than requiring every vertex to be a rectangle corner. Explicit schema-5
Sheet-joint records qualify intended radial incidence; metadata cannot authorize
nonmanifold meshing. The automatic controller also treats a typed quad-quality
refusal as a recoverable method failure without changing any quality threshold.

The exact source replay passes locally at both requested target sizes:

| Size | Nodes | Quads | Triangles | Selected method |
| --- | ---: | ---: | ---: | --- |
| 0.5 | 5165 | 5058 | 232 | auto after typed quad refusal |
| 0.25 | 1916 | 1646 | 386 | quad_first |

`accepted-replay-050.json` and `accepted-replay-025.json` retain admission,
all six complete shared-node/segment joints, analytic joint lengths, independent
material totals (25 per plate; 8*pi for the cylinder), source non-mutation,
project save/load and mesh-codec round trips. Actual meshes and source documents
are retained alongside them. The .25 result carries the existing warp advisory;
it is not suppressed or recategorized. These are local source-import results,
not installed-package or cross-platform acceptance. Core regressions at this
checkpoint: 1,025 passed in stage2-kernel-regressions.log.

The consumer compatibility run improved to ten passes and one real failure.
The remaining failure revealed construction seams splitting a shared boundary
on one face while its neighbor retained the whole edge. A final read-only
planning pass now propagates all such ports before rebuilding faces. The warped
Coons compatibility case then passes with complete shared nodes; general Coons
interior intersections still fail with a typed error. The scope remains Plane
and Cylinder material domains. New result-content bindings reject altered
analytic definitions and policy switches before application/evaluation.

Remaining work includes general periodic/curved arrangements and their analytic
coverage evidence, original public-pair compatibility, structural reference
remapping, extended meshing fixtures, feature/history verification, installed
consumer/matrix evidence and four-stage reviewable delivery. No release or
publication authority is consumed by this implementation.

## General joints and bounded admission diagnosis

The parallel-cylinder fixture now produces an admitted mesh (834 nodes, 816
quads) after boundary seeding propagates the existing adjacent-growth policy
into transverse mapped directions. Source geometry remains unchanged. Full
analytic joint coverage has now passed: two complete generator intervals with
30 shared stations, 28 segments, and independent cylindrical material areas.

Perpendicular cylinders pass kernel topology and analytic material conservation
(12*pi), including one-sided branch derivatives at cylinder seams. Their
first automatic and native triangular mesh candidates were unaccepted: bounded
S3 repair refused admission. The next experiment retained one actual native
triangle candidate before repair, with failed cell coordinates and authoritative
owner normals. Competing explanations are boundary station clustering, chart
triangulation and owner orientation; those measurements select the next change.
The retained UV-lattice candidate had 303 failed triangles with consistent
owner normals. Physical chart lengths removed those failures. The diagnostic
requested frontal options; the pure-triangle surface path used its existing
lattice filler, so the decisive change was the metric, not an independently
demonstrated improvement from frontal insertion. General analytic cylinder
components now receive physical charts for either point-placement selection;
explicit controls are retained. Completed chart validation is reused only
when complete document and result content bindings match. No repair budget,
quality threshold or scientific tolerance is changed.

ANYfem automatic meshing subsequently admitted perpendicular cylinders: 1,106
nodes, 2,266 triangles, zero quality warnings. The first structured candidate
was rejected; the native triangle fallback was selected. Independent checks
verify both full closed branches from x^2+y^2=x^2+z^2=1, all 86 shared stations
and 88 segments, two common high-valence nodes, and 6*pi material per cylinder.
The complete automatic run took 412.5 seconds; no performance improvement is
claimed. Retained evidence is in mesh-cylinders-perpendicular-general-chart-next.log,
accepted-cylinders-perpendicular-checks.json and verified-perpendicular-joints.log.

One-Sheet multi-face joints preserve Sheet identity and use its existing
explicit nonmanifold-edge policy, qualified against actual Sheet radial uses.
Focused batch tests pass (10); bound-chart tests pass (3), including altered
evidence and direct source edits. Next checks cover affected consumer regressions
and the skew finite-cylinder mesh, then reference lifecycle and remaining public
contracts. These results remain local source-import development evidence.

Public face-pair application now uses the batch contract for supported Plane and
Cylinder domains, including nested automation transactions. Exact attachment
clipping retains narrow descendant intervals; structural junction ranges are
validated against their attachment union. Latest focused checks: 75 nested
automation/attachment/overlap tests and 65 curve/arrangement/fault tests pass.
The prior full kernel checkpoint had 1,033 passes and three failures; all three
have focused fixes, pending the final full regression. Public member-pair
integration, further reference cases and complete platform acceptance remain
outstanding. These are development results, not release qualification.

## Replay lifecycle and current candidate limits

Kernel checkpoint: 1,043 tests passed in 191.85 seconds. The affected consumer
set passed 42 tests with one incorrectly added hard target-size assertion;
the owner rounds seeding demand. That assertion was replaced by independent
complete angular joint coverage, and all 11 ANYfem intersection tests pass.

The exact source lifecycle probe found an existing structural-generator replay
defect: equivalent newly inserted Members prevented removal of unpublished
duplicate core edges. Duplicate owners now require exact role correspondence
and equivalent axis/face-use and ownership definitions before removal. Their
groups and tags follow the same exact replacements. Original Members, Sheets,
output bindings and groups remain intact. Source undo/redo, equivalent feature
regeneration, all 55 downstream face references and topology pass locally;
40 focused feature regressions pass. Changed structural materializations and
reference-rich feature edits still require further verification.

The current .5 replay rerun is **unaccepted**: quad-first rejects the existing
13.59-degree cell; automatic and native fallback now reject element growth.
The earlier accepted checkpoint is retained under *-boundary-checkpoint artifacts.
The three actual method refusals are in replay-050-recovery-refusals.log. Next
mesh investigation must use these candidate quality records to fix seeding or
interior gradation, without relaxing growth or angle gates. The .25 replay has
not yet been rerun against this candidate.

The skew mesh run was stopped during exact kernel preparation after roughly
25 minutes without a completed candidate. A 30-second read-only profile then
cancelled cleanly with unchanged source. It identifies rational midpoint
evaluation in certified root isolation as the dominant cost (including millions
of gcd operations). Replace only that exact-zero test with homogeneous integer
Horner evaluation of the already qualified primitive polynomial; sign, roots,
isolation intervals and tolerances remain unchanged. Compare a bounded profile
and analytic-root regressions before another whole skew run. No skew mesh is
accepted, and no cross-platform or installed-wheel result is claimed.

## Replay admission and next cylinder slice

Both requested exact command replays now pass automatic linear admission and
source lifecycle checks: .5 uses 5,668 nodes and 11,376 triangles; .25 uses
2,021 nodes, 1,662 quads and 632 triangles. The underlying defect was the
normalized legacy cylinder chart still used on owner-qualified circular joints.
Such components now use physical material coordinates while retaining caller
meshing options and existing quality gates. Evidence: replay-050-joint-metric.log,
accepted-replay-025-lifecycle.log and their accepted-replay JSON records.
Earlier failures and the boundary checkpoint remain development history.

The integer midpoint change passes 17 exact curve/root checks. The complete skew
kernel plan/application now finishes in 84.75 seconds, with valid topology and
independently conserved total material 2*pi*3*1.8. Skew meshing remains unaccepted.
The next bounded application run asks whether the physical charts admit this
finite unequal-radius skew joint, or expose a discretization/quality refusal.
Retain any refusal and its stage; no quality threshold or root tolerance changes.

Public member-pair preparation now routes distinct point contacts through the
batch contract. Axis bends are not one-member structural joints. Fifty focused
workflow/batch checks pass, including two distinct crossings, idempotence and
candidate split-failure rollback. Multiple visits of the same member at one
world point remain typed unsupported under the current Junction representation;
inconsistent externally supplied query components also remain unqualified.
These results are local source-import development checks, not installed or
cross-platform acceptance. No release/publication action is authorized here.

Finite-height point contacts now bind analytic support events into the batch
plan and canonical topology, with exact Vertex-on-Face attachments. Fifty-two
workflow/batch checks pass, including a pair of tangent finite cylinders meeting
at one endpoint and endpoint-to-interior Member attachment evidence. Trim-only
isolated events and legacy public face-pair point routing remain outstanding.

The skew application reaches chart qualification but still spends minutes in
global Cauchy-bound root isolation. Next question: can two bounded rational
projective charts isolate the complete quadric event set without global large
parameter intervals? Use exact Sturm counts on [-1,1] for tangent-half-angle
and reciprocal charts, retaining seam roots, world residual filters and existing
angular error bounds. Verify known rational roots, endpoints, multiplicities,
projective seams and cancellation before another integrated run. This changes
root parameterization, not intersection truth or numerical admission thresholds.

## Current checkpoint after projective charts

The full kernel suite passes 1,053 tests (188.23 seconds). Exact rational
projective interval tests retain repeated roots, endpoints, very large parameters
and cancellation; 67 root/curve/intersection checks pass. Batched area evaluation
keeps Gauss orders 16..512 and the same convergence tolerance. A read-only
30-second skew-chart profile completes all 42 charts, material 33.92920065876977.
Public face-pair point routing and trim-isolated contacts now pass 60 focused
checks, including idempotence on existing high-valence joints.

Mesher seeding/recovery/structural checks pass 60 tests. Thirteen seeding checks
include physical growth, preservation of explicit pins and cancellation with
source non-mutation. The viewport adapter passes 12 checks on Python 3.13 with
its triangulation extra, including owner-decoded schema-5 exact curves. Python
3.14 viewport execution was unavailable (missing extra) and is retained as a
skip; do not count it as acceptance. Tangent whole-cylinder kernel preparation
passes, with one complete generator joint and total material 12*pi.

The slow source-based skew attempt was stopped in native generation. Its
120-second bounded diagnostic cancelled during triangulation after recording
11,996 generated interior points on a face; source was unchanged. Installed
candidate wheels now include the compiled mesher extension. The FEM wheel is
built from HEAD b9cd5945f26b8eceb8fcd717b0032753207b7dc9 with only the owned
structural_preparation.py overlay, excluding unrelated dirty frontend work.
Candidate geometry/mesher/FEM versions stay 0.4.4/0.5.1/0.4.1.

Installed runner tools/check_general_intersections.py verifies all owner import
origins, absence of editable installs, and candidate package bytes against wheel
contents before running copied fixtures outside source checkouts. Artifact hashes,
resolved dependencies, pip reports and logs are retained in the external report.
Public dependencies are pinned to solver 0.4.6, material 0.2.0 and fileio 0.3.2.
The first download was sandbox-network blocked; a network-enabled public wheel
download succeeded, followed by offline installation. Installed skew meshing is
still pending, in external Temp/anygeometry-general-consumer-20260930-offline.
No hosted matrix, installed replay or skew admission is claimed yet.

Installed Python 3.13/Windows candidate replay and tangent checks subsequently
pass in Temp/anygeometry-general-consumer-20260930-replay-oracle. The exact .5
replay retains 5,668 nodes/11,376 triangles, .25 retains 2,021 nodes/1,662 quads/
632 triangles; both pass source identity, complete joints, material, save/load,
undo/redo, equivalent regeneration and 55 face references. Tangent cylinders
admit 248 nodes/224 quads, with 8 shared stations and 7 segments covering the
full axial generator. Dependencies resolve to NumPy 2.5.3, SciPy 1.18.1,
Shapely 2.1.2 and the pinned owners. A verifier-only NumPy 2D-cross removal was
fixed by the explicit determinant, retaining the same analytic angle and
tolerance; the failed first verifier run remains retained.

Installed skew meshing with compiled triangulation confirmed available timed
out after the runner's 900-second bound. No accepted mesh or branch coverage
result exists. This excludes absent native acceleration as the sole explanation.
Next integrated decision: separate authoritative boundary stations from uniform
interior density in the existing frontal path, then test graded admission on
the same finite skew case. Keep all shared stations, geometry-owner curves,
source references and quality gates; do not simply raise the time budget.

## Skew frontal diagnostic and binding decision

Separating registered boundary stations from nominal interior density passes
station-preservation tests, but the explicit frontal skew diagnostic cancels
after 180 seconds during native-v2 queue maintenance. This is not default-policy
admission. Its 60-second trace also finds repeated full topology qualification
inside chart binding checks between faces.

Next bounded change: factor deterministic document encoding from public
serialization qualification. Public to_dict still validates topology, structural
state, feature persistence and requested certification. Chart evidence may reuse
its previous qualification only when the complete persisted content checksum
and complete chart definition checksum are unchanged; direct dictionary edits
must still invalidate it. Check equivalence, public invalid-topology refusal,
direct-edit rejection and cancellation before another installed diagnostic.
No acceptance tolerance or native insertion budget changes.

The content-bound candidate retains the 180-second skew cancellation, now in
metric anisotropy evaluation; the 60-second trace also enters scalar gradation.
Inspection finds the existing invocation-local deterministic metric cache is
activated only for explicitly 3D controls. The uniform metric has no declared
spatial dimension and repeatedly evaluates every unchanged point. Next bounded
change: permit the same exact-coordinate cache for all immutable SpatialMetricField
controls, preserving the uncached route for experimental providers. Require
byte-identical mesh, contract and native diagnostics with caching disabled for
uniform and spatial controls. This does not change the metric or its gradation.

The uniform-cache installed diagnostic produced an incomplete faulthandler
trace and stopped making progress at approximately 180 CPU seconds; the owned
process tree was interrupted. Do not claim a completed cancellation/rollback
from this run. The next diagnostic removes the scheduled traceback sampler and
records bounded profiling plus face/operation progress through existing
cancellation callbacks. Keep the same 180-second bound and artifact identities.
Feature radius edits crossing the generator's bay-count rounding boundary also
reproduce conflicting closure-role bindings with exact rollback. Test a radius
edit retaining the same generated bay count to distinguish role-count changes
from retirement of structural owners before choosing the repair.

A radius edit from 1 to 1.01 retains all 52 panel roles but fails because old
MemberEdgeUses still own the retiring edges. The .99 and 1.1 edits also change
the rounded generator bay count and have a distinct closure-role conflict.
Next end-to-end slice: transfer existing structural owners through proven
one-to-one output-role mappings, discard matching unpublished generated owner
duplicates, then retire core topology. Preserve Member/Sheet/Part/FaceUse IDs,
orientation and metadata. Ambiguous topology and attachments lacking a qualified
parameter remap must still fail atomically. Verify a changed-radius design and
downstream references, plus existing feature and serialization regressions.
Bay-count-changing regeneration remains outstanding until separately handled.

Uniform cached frontal profiling completes eight native faces before the
180-second diagnostic cancellation, with source non-mutation. The profile
records 4,726 scalar gradation calls (35.607 cumulative seconds) and 95.183
seconds in native face meshing; it confirms forward face progress rather than
an infinite loop. Full default-policy admission remains untested with the new
boundary/interior separation. Next bounded experiment extends that separation
to the existing legacy filler too: authoritative curved boundary stations stay
fixed while target_size controls interior lattice density. Existing repair and
quality gates choose admission; no thresholds change or automatic controls are
forced. The installed consumer keeps its existing 900-second command bound.

Checkpoint: 1,056 kernel tests pass after private content encoding and structural
owner transfer. Changed radius 1->1.01 preserves all existing Sheet/Member/
FaceUse/MemberEdgeUse identities and authored parent ranges, tags and serialized
geometry. A late injected owner-transfer failure leaves the live document exact.
The FEM reference test also passes: section/load/support values and region IDs
persist, save/load resolves historical convenience refs to live descendants,
and undo/redo retains authored geometry with monotone allocator high-water
marks (the established no-ID-reuse policy). Bay-count changes remain unresolved.

The bounded frontal profile identifies repeated gradation on face edge batches
below the generic 4,096-row native threshold. Next bounded change activates the
existing cancellable compiled scalar-gradation kernel from 64 edges upward,
retaining the exact reference sweep order, bounded convergence checks, original
exceptions and input non-mutation. Verify small-batch native/reference parity
and cancellation before a full installed frontal case. This changes execution
routing, not metrics, chart truth, quality gates or geometric tolerances.

Compiled Python 3.13 mesher checks pass 77 tests, including new 64/257/1,025-edge
reference parity, existing failure/cancellation propagation and byte-identical
cached/uncached cylindrical meshes. The default legacy filler experiment has
not yet completed within ten minutes. Next full installed frontal case uses
the verified small-batch compiled routing, periodic progress logging without
profiling/traceback sampling, and the consumer's existing 900-second command
bound. Explicit frontal controls remain separate from automatic-policy admission.

The default interior-density experiment completes but fails the existing
qualified-S3 bounded repair gate; it is not accepted. Explicit frontal face
generation reaches its completed mesh and then raises TypeError in element
growth: neighbour smoothing published moved node positions as tuples, while
the Mesh node contract and downstream arithmetic expect NumPy arrays. The
representation repair preserves the same coordinates as defensive float arrays.
Fifteen shared-split/neighbour/frontal checks pass, including a moved-node
quality-consumer regression. Rerun the installed frontal admission and independent
branch/material oracle; automatic route selection waits for this result.

Released comparison is recorded with installed geometry 0.4.3, mesher 0.5.0
and FEM 0.4.1. The original replay rejects face:3/face:4 with 'imprint component
does not map to one active face descendant', and source is unchanged. Released
mesher 0.5.0 has no MeshAutomationOptions API; this absent capability is explicit,
and the fixture uses the released application's automatic defaults. The first
comparison hit that missing API at import and remains separate failed evidence.

The tuple-coordinate repair lets explicit frontal generation complete, but an
automatic structured-quality rejection restarts through the native route. The
subsequent final stage exceeds the installed runner's unchanged 900-second
command bound. No skew acceptance, completed rollback or oracle result is
claimed for the timed-out process.

Next decision-oriented owner diagnostic uses explicit native frontal routing
with identical quality gates, captures an inspection-only pre-S3 mesh remapped
through public lineage to source face groups, and applies the independent
implicit/interval/graph/material checks without asserting solver admission.
Profile only the S3 owner bridge, terminating after 45 seconds at the next
owner projection call, to discriminate expensive trimmed projection from repair
work. It changes no production numerical values or acceptance thresholds.

Owner-stage diagnostic 2026-10-01: the pre-S3 skew mesh independently passes
all four implicit branches, interval/component coverage and material checks
(28,053 nodes; 56,517 triangles; 730 joint segments). It is inspection-only,
solver admission BLOCKED. The first 45-second cancellation hook does not reach
inside a long projection; that owned process was interrupted, so no completed
rollback is claimed. Save the raw mesh and prepared owner document once, then
bound inside analytic-curve boxes to distinguish projection from repair and
reuse those exact artifacts for subsequent owner-stage checks. Projection uses
conservative box branch-and-bound, which also needs an algebraic stationary
candidate path for flat-distance ellipse cases; retain complete global minima
and existing tolerance, rather than widening acceptance.

Isolated S3 inspection changed the next action: all 56,517 owner-normal
projections complete in about 14 seconds. The bounded stack instead locates
quadratic list membership in S3 directed-edge admission: for every shell it
searches the complete elements_of_sheet lists. Invert declared Sheet
associations once while retaining exactly the old sorted ownership tuples,
including missing/multiple owners. Verify declared/nondeclared junction
admission and growing fixtures, then reuse the captured mesh to test unchanged
S3 repair/admission. No geometry or quality threshold changes are warranted.

The indexed S3 topology gate passes 53 existing regressions, but profiling the
unchanged repair still spends its bound evaluating complete 56k-triangle shape
reports for each tiny candidate edit. The first target report has 569 target
shortfalls, six angles below the unchanged solver floor, and no topology
violations. Retain all bounded attempts and both quality policies. Reuse exact
scalar shape results only when element ID, connectivity, all coordinates,
normalized owner normal and complete policy match; recompute complete topology
on every candidate. Invalidating edits and cached/uncached repair parity pass;
all 55 S3 checks pass. Recheck the captured inspection mesh unprofiled with the
same 180-second diagnostic bound, preserving every failed artifact separately.

Next bounded decision: skew has six mandatory floor failures, but the existing
repair searches the preferred 30-degree target first, which can reject a flip
that repairs the required floor and consume the same budget elsewhere. Compare
with a two-stage mandatory-first search sharing all existing attempt, flip,
refinement and node budgets. Every candidate keeps the existing score and floor
non-regression tests; final floor admission remains mandatory and preferred
shortfalls remain explicit. A disconnected two-patch test discriminates budget
priority from intrinsically unrepairable geometry, followed by the saved mesh.

Stage 2 committed locally as 6557b2c: atomic material arrangements and trimmed
surface charts. Kernel 1062 tests pass; three new growing interior-cut fixtures
also pass, including 25 cutters with per-original-face material conservation.

The S3 index initially rebuilt the computed Mesh.shells dictionary per Sheet
association. Hoisting it once is necessary for actual linear indexing; tests
now assert one computed-shell read even for growing lists. All 56 S3 tests pass.
The corrected isolated stage completes in 40.7s and refuses six unchanged
triangles (no topology failures): curved-normal alignment blocks legacy planar
flips and fixed boundary edges refuse unauthoritative refinement. A logging
bug when profiling was disabled obscured the previous diagnostic result; that
failed record is retained and the repaired logger records the actual refusal.

Principal cause is now shared-boundary publication repair, not S3 tolerances.
Some bad cells have aspect ratio below the old >5 trigger, despite failing the
qualified S3 floor. Trigger owner-chart smoothing on the existing floor metrics,
keep all protected coordinates/coverage exact, and finish every completed
cylinder binding after cross-component station propagation. Sixteen focused
repair/integration checks pass. Reuse the exact saved mesh, repair only faces
25 and 28 through their owner charts, and then apply the unchanged final gate.

Owner-chart repair preserves every protected station and improves the six
failures to three, but the unchanged four smoothing iterations do not finish.
Before choosing more work or remeshing, the next diagnostic is eight bounded
passes on only these two saved faces, with per-pass floor counts and score,
fixed coordinate equality, and early stop on success or no selected progress.
This distinguishes convergence of existing constrained smoothing from a
station-density problem requiring fresh constrained discretization. The saved
input is reused; no numerical thresholds or production iteration limits change.

The eight-pass smoothing diagnostic plateaus with three failures; more
smoothing is not the next action. Exact corner tangents of faces 25/28 are
72--180 degrees, so this is not an unavoidable acute material wedge. Test
conforming longest-interior-edge subdivision and local flips in the owner's
cylinder chart, with each new midpoint qualified by the public exact material
predicate. Keep all existing boundary stations byte-exact. Diagnostic budget:
200 insertions per affected face, stop on qualified shape success. This
separates insufficient interior station density from a topology/curve problem;
it changes no production limits and never asserts application admission.

The saved owner-chart diagnostic required exactly one interior bisection on each
of faces 25 and 28. The resulting whole mesh passed qualified S3 admission and
independent implicit-curve, shared-node and material-area checks. This remains
inspection evidence, not automatic ANYfem acceptance. Implement the repair as a
staged face-local operation using owner material predicates, the existing native
insertion/work budgets and immutable boundary stations. Reject malformed flips,
budget exhaustion and cancellation before publication. Next bounded experiment:
one source ANYfem skew case with explicit frontal controls, followed by the
independent accepted-mesh verifier; the existing 900-second consumer bound
remains. A pass permits automatic-route integration; a failure identifies the
remaining application boundary without repeating curve diagnostics.

Production explicit-frontal preparation passed S3, but the enclosing structured
layout applies its separate unchanged regularity gate and retries native
meshing. This is not complete application acceptance. The automatic analytic
trim preference now selects the existing native triangle recipe first, retains
native insertion/work/cancellation budgets, and leaves strict/explicit methods
and historical circular patches unchanged. Contract/metric/seeding/S3 regression
batch: 147 passed. The current explicit-frontal application run is allowed to
finish its already-started native fallback (at most 1,200 seconds total); this
prospective change concerns a local disposable diagnostic only, not any frozen
scientific or execution authority. Updated default-route installed acceptance
remains the deciding experiment.
