# Making intersections more general: cone against cylinder, and beyond

## Large connected models, 2026-10-02

User approved 10/100/1,000 authored face/member operands and 100 connections on
one plate, supported families only. Acceptance includes full-scale automatic
linear meshes and smaller representative quadratic meshes through ANYmesher.
Base runtime d05cd3a / documentation f5abffd; earlier failed probes and releases
remain unchanged. Work is isolated on codex/large-connected-models.

First decision: distinguish pair-enumeration/bounds cost from arrangement/apply
cost using a connected panel strip with alternating beam and shell stiffeners.
Each tile contributes two authored operands, exact unit material and analytically
known boundary/interior joints. Start at 10, then 100; only proceed to 1,000 after
interpreting stage timings. Separate the dense 100-connection hub from distributed
growth. Keep small fixtures in ordinary tests and large cases in a bounded runner
with per-stage results, process peak memory, cancellation and retained failures.
No numerical tolerances, quality criteria or previous budgets are changed.

First results: strip 10/100/1,000 pass, including 999 analytic joint loci at the
largest size. Profiled 1,000 planning: 53.04 s before versus 25.01 s after per-call
face-bound caching; topology counts match. These are single development profiles,
not a controlled speed benchmark. Repeated bounds dominated the original scan;
after caching, arrangement is the largest component, so defer sweep-and-prune
until evidence warrants its budget/cancellation complexity. Existing batch and
performance-equivalence tests: 32 passed. New small contract tests: four passed.

The 100-connection hub passes 2,500 canonical host/shell/member crossing vertices
and produces 2,651 material faces. Mixed connected bays include floor plates,
cubic/parabolic walls and eight-panel cylinders/cones. Mixed 100 passes in about
44 seconds across measured build/plan/apply/chart/verification stages. Proceed
once to the new mixed-1,000 fixture with an 840-second cancellation deadline and
900-second owned-process ceiling; this is new development execution, not reuse
of any historical scientific budget. Export only a fully verified result. Final
mesh acceptance remains ANYmesher-owned and unconfirmed; its last task turns
returned empty completions, so no worker acceptance can be inferred.

Mixed-1,000 first run FAILED during planning: face 602 (cubic wall in bay 60,
world X about 600) failed material conservation. No deadline exhaustion or
accepted partial output. Preserve mixed1000-first.json/log. Next discriminating
probe: same isolated bay translated to X=600 versus origin, separating coordinate
conditioning from neighbor interactions. Fix only the demonstrated owner defect;
do not widen the conservation tolerance or rerun the large case before a focused
regression passes. Review also found the mixed fixture needed explicit required
wall/quadric incidence, not just connectivity through floor: that check was added.
Python optimization is now rejected so asserts cannot silently disable acceptance.
The external 900-second ceiling bounds the full large run, including serialization.

### Focused repairs and consumer development evidence

Translating extrusion area integration to its profile frame fixed bay 60, but the
next bounded mixed-1,000 run failed at bay 69 (face 692). Both failures are retained
in `reports/large-connected/mixed1000-{first,local-frame}.json` in the primary
checkout. The second fix integrates exactly constant-height polynomial boundaries
by their Green antiderivative. Exact rational control-height equality certifies
this shortcut; tolerances are unchanged. Centered Bezier subdivision preserves
constant coordinates and explicitly inherits its original outer endpoints.
Read review identified the mixed-magnitude endpoint issue; a regression now covers it.

The concave plate fixture exposed a separate real topology defect: a decomposition
port at y=5 split a face boundary but not its coincident member axis. Planning now
propagates member-labelled arrangement endpoints to every corresponding axis using
the existing exact parameter query and counted cancellation checks. The formerly
missing y=2..6 connection passes, as do point attachments at y=4.5 and y=5.
The fixture uses explicit circular trim arcs: the first `punch_hole` construction
failed projection on the concave host; that separate authoring limitation is retained
in `large-connected-concave-first.log`, not claimed fixed.

Checks actually run: 111 affected batch, attachment, material, branch and extrusion
regressions passed; eight focused cases passed after endpoint preservation; two
additional boundary-attachment cases passed. A too-strict new ownership oracle
initially rejected creation of sheets for previously unowned faces; corrected it
to require preservation of every authored sheet, descendant FaceUse and orientation.
The failed oracle log is retained. The new mixed-1,000 run has an 840-second
cancellation deadline and 900-second outer process ceiling. No repeated failed
run was launched without a focused repair and passing reproducer.

Read-only source integration probes used the existing d05 geometry wheel hash
`54367f21cd661413c43701ce629f90f09d525a87a82e8ccaf267dc68fc8112b8`
and ANYmesher source `f0cde877d6fd22048d2fa7af1e6af84c5b20d0fd`.
Strip 100: ready, 528 nodes / 300 Q4 / 50 beams, area 75, 74 shared edges.
Strip 1,000: ready, 5,253 nodes / 3,000 Q4 / 500 beams, area 750, 749 shared edges.
Every multi-owner prepared edge's node sequence belongs to its incident faces and
members; authored/prepared documents remained unchanged. The latter elapsed
202.69 seconds including preparation. These are source-integration development
results, not clean installed-consumer or new-candidate acceptance. Reports and
probe scripts live under the primary `reports/large-connected` directory.
ANYmesher has received the results and the request for mixed/quadratic acceptance;
its subsequent task turn again returned an empty completion. Owner acceptance
and the short-joint quadratic repair remain outstanding.

### Candidate 364a9f9 and corrected consumer oracles

Runtime commit `364a9f962657db7a65e097d81b6856326cfd56d7` passes the full
1,697-test kernel suite. Development geometry wheel SHA256:
`79b05a80b18f8f0f33c40830dde0a3b4972effc3116500e5c5e0bcbfadea4167`.
A clean disposable environment outside all checkouts has NumPy 2.5.3, Shapely
2.1.2 and pytest 9.1.1; all 13 copied focused tests pass there. Import origins and
pip's direct-url artifact hash confirm the candidate remained installed.
The first installed invocation suffered pytest's unrelated shared-temp cleanup
permission failure; its log is retained. The accepted invocation uses a dedicated
temporary directory and an outside-checkout working directory.

Mixed1000-chart-callback passed geometry, including serialization and repeat apply:
5,746 faces / 14,292 edges / 12,236 vertices; peak working set 430,022,656 bytes.
However, inventory review found the generator authored 800 extra longitudinal
Members not included in its 1,000 face operand list. This result is retained as
an explicitly scoped face-only preparation result, not acceptance of the requested
exact operand inventory. The corrected mixed builder removes default generated
Members through public APIs before insertion; strip/hub cover beam stiffeners.
It asserts total authored faces+members equals the requested count. The verifier
now rejects any unselected authored operand. Thirteen focused tests and the
bounded mixed10 driver pass. Mixed1000-exact-operands is the new bounded run.

The portable consumer driver initially omitted the explicit `beam_edges` selection
required by `prepare_structural_closure`. A stronger oracle comparing every
authored joint's actual node sequence found the omission. Earlier ready results
are only limited face/shared-edge evidence and are superseded for full structural
acceptance. The first handoff labelled the hub failure consumer-owned; a prompt
correction identified the probe-call defect. No consumer patch was requested for
that defect. The driver now supplies all authored Member edge uses and checks
complete intended joint sequences, endpoints and duplicate coincident nodes.

With that correction, installed strip1000 linear PASSES: 4,503 nodes, 3,000 Q4,
500 beam elements, all 999 intended joints and shared edges conforming. Preparation
13.13 s, meshing 4.12 s; this is one development run, not a speed comparison.
Installed strip100 linear and strip10 quadratic also pass. The latter has 117
nodes, 28 Q8, six beam elements and nine conforming intended joints.
ANYmesher input `f0cde877d6fd22048d2fa7af1e6af84c5b20d0fd` was archived without
modifying its checkout; its built wheel SHA256 is
`a0f9aa2876590788cc4ab8de51e110b7e875c3c6c883f6a91d5ee7bbf54b9eb2`.
All artifact/consumer reports are under the primary `reports/large-connected`.

Mixed10 linear remains UNACCEPTED: its 120-second recovery budget expired in
the overlap audit. Source review found a possible receipt-invalidating order:
ANYmesher applies owner preparation, then restores collinear face corners, which
changes revision/checksum. A subsequent attempt correctly cannot reuse that
receipt and mixed supports fall back to pairwise overlap classification. The
specific modified face is not proven by the retained log. The owner handoff asks
for certification of the final state, never manual receipt refresh or waived
auditing. The short-joint mixed quadratic case and full mixed linear meshes remain
unaccepted. No package publication, default switch or merge acceptance is claimed.

The exact-inventory mixed1000 run PASSES with 1,000 authored faces, zero Members,
5,746 retained faces, 14,292 edges and 12,236 vertices. Peak working set
411,590,656 bytes. Measured plan/apply/charts/verification: 120.90 / 66.26 /
228.09 / 4.98 seconds; serialization also completed within the outer ceiling.
The corrected installed hub100 linear probe PASSES all 100 intended joint node
sequences and 5,100 shared topology edges: 15,704 nodes, 15,351 Q4 and 2,550 beams.

A new bounded preparation-only diagnostic confirms the mixed receipt issue:
owner batch gives a current proof; consumer preparation does not. It changes
31 faces' corner indices (Plane, Cylinder and ExtrudedSurface), with 70 faces in
both preparations. Evidence: `mixed10-preparation-binding.json`. No meshing retry
was used for this diagnosis. Next decision: safe owner-certified corner-only
editing versus consumer ordering; never mint a receipt from an unchecked model.

### Corner-only owner certification

Implemented root-exported `set_prepared_face_corners`: requires a valid complete
receipt, explicit built-in Plane/Cylinder/Cone/ExtrudedSurface support and no
separate parameterization; stages all updates, compares the entire certified
document except allowed corner/revision/checksum fields, then commits and issues
only a preparation receipt. No stale proof or old application plan is refreshed.
Sixteen binding/corner tests pass, including all four support types, invalid
two-face rollback, nested-transaction refusal, parameterization refusal and old
plan rejection. Read review found no material-proof bypass; it noted change hooks
see no current receipt until postcommit publication, matching existing batch
application. This fail-closed observer behavior is documented and tested.

ANYmesher worker continues returning empty turns. Under the user's explicit
implementation authority, integration is isolated in
`C:/Github/ANYmesh/.worktrees/prepared-corner-binding`, base f0cde877. Only
preparation.py and a focused binding regression are owned there. Corner updates
are collected and sent to the owner operation when eligible; older owner builds
retain ordinary editing and auditing. Main and unrelated work are untouched.
Twenty-six mesher preparation tests pass, including the actual mixed fixture,
proof preservation through attempt cloning and re-auditing a later edited model.
A new bounded mixed10 meshing check is justified by this specific tested fix;
the previous timeout and all scientific limits remain retained.

### Coordinated corner-binding inputs and remaining blocker

Geometry `71b394a40a78f3a9d836bb18195e495bd58265be`; development wheel SHA256
`d023349d879a7dd8fe3118d7edcc378f861300533fc350bf69040f54f6cb1225`.
Mesher `7e2fded699d055b59ebe5b02d52e6e23255aa556`; Windows CPython 3.13 wheel SHA256
`0a956a0684594f40807d03ccac2c564bdbe2b012d1b032a8fd48ccdc108e1654`.
The fresh outside-checkout environment `anygeometry-large-71b394a` passes all
29 copied geometry/binding tests. Both import paths, both pip artifact hashes
and the loaded compiled mesher extension are retained in installed-71b394a-identity.json.
The earlier 1,697-test full run binds runtime 364a9f9; do not relabel it a full
run of the added corner API. New API/consumer-focused checks are separate.

The substantive binding fix moves the mixed10 failure past overlap auditing:
the Python backend now exhausts 120 seconds in native surface preprocessing.
Inspection of ANYfem confirms its automatic default uses `native_backend='auto'`;
the probe's previous forced Python configuration was not that default. Added an
explicit backend selector, default auto. A fresh installed probe with the compiled
extension available and auto selected also exhausts the unchanged 120-second
budget during native surface preprocessing. Preserve both reports/logs. Neither
is accepted meshing, and no larger mixed mesh or renewed short-joint quadratic
probe is justified before a further bounded owner repair.

The remaining work is mixed-surface discretization/performance, mixed quadratic
admission (including the previously blocked short joint), final platform/consumer
qualification and Git delivery through review. No full-stage completion, release,
main merge or changed numerical/quality budget is claimed. Existing full-scale
geometry results remain valid for unchanged preparation/arrangement functions;
the new corner API adds a separate finalization path. The current planar consumer
checks use the exact coordinated wheels and the corrected complete-joint oracle.

Final coordinated installed checks, auto backend: strip1000 linear PASSES all
999 intended joints (4,503 nodes / 3,000 Q4 / 500 beams; prepare 12.55 s, mesh
3.77 s); hub100 linear PASSES all 100 intended joints (15,704 nodes / 15,351 Q4 /
2,550 beams; prepare 46.34 s, mesh 51.26 s); strip10 quadratic PASSES all nine
intended joints (117 nodes / 28 Q8 / six beams). These are individual development
measurements, not comparative performance claims. Evidence SHA256s and remaining
work are indexed in `large_connected_capability.json` alongside this record.

Restart: geometry branch `codex/large-connected-models` in intersection-continuation;
mesher branch `codex/prepared-corner-binding` in its isolated worktree. Neither
main was changed or pushed. Geometry main remains 462ca593; mesher main f0cde877.
The next work must diagnose/fix mixed surface preprocessing, not re-run consumed
120-second probes unchanged. No owned checks remain running after this checkpoint.

## Continuation, 2026-10-02

PR12 development run 36989128888: Windows and Linux each passed 1,675 tests
and failed the existing bitwise scalar/batch inversion assertion for the tilted
cubic surface. Wheel build, Linux/macOS installed-wheel and installed-mesher jobs
passed. Preserve both job logs in the primary reports directory. Diagnostic
question: does BLAS choose different reduction arithmetic for one versus many
rows, or does the directrix Newton iteration depend on batch membership? Compare
extrusion coordinates and projected-directrix inversions separately in the clean
NumPy 2.5.3 environment; retain the exact-equality assertion and all tolerances.
Repair only the demonstrated arithmetic path, then run focused inversion tests
before a new hosted candidate. No blind CI retry.

The Haswell OpenBLAS kernel reproduces the failure locally: four extrusion
coordinates differ by at most 2.22e-16; directrix inversion of identical inputs
is bitwise equal. Explicit three-term arithmetic preserves the same formula
and reduction order for scalar, batch and higher-rank queries. Existing exact
assertions remain unchanged; new partition/reversal tests exercise the contract.
The Haswell focused surface, public-routing and intersection suites pass 84 tests
(`reports/pr12-inversion-Haswell-regressions.log`). Independent source review found
no shape or contract regression. The failed hosted logs and diagnostic logs are
retained; the old candidate wheel is not replaced or relabeled.

Follow-up runtime candidate: `d05cd3a37d884702e043b678808f50d03ed4a27e`.
Wheel SHA256: `54367f21cd661413c43701ce629f90f09d525a87a82e8ccaf267dc68fc8112b8`.
A new clean temporary environment passes 59 installed tests (six consumer cases
and 53 surface tests under Haswell BLAS). All 75 installed package files match
the wheel. New documents are in primary `reports/consumer-contract-d05cd3a`;
the immutable evidence snapshot is `reports/intersection-continuation-evidence-d05cd3a.json`.
Hosted run 36991408474 passes all eight jobs: Windows and Linux each pass 1,684
kernel tests; build, Linux/macOS installed-wheel and installed-mesher checks pass.
Full run log and status are retained and hashed in the evidence snapshot. No
release artifact was changed; full matrix/release qualification is not claimed.
ANYmesher's bounded seeding experiments did not resolve face 11: they move the
sub-15-degree cell or fail strict quadratic mapping. Its inspection-only fallback
is not acceptance. A front/topology correction proposal remains owner work, with
the original thresholds and consumed probe budgets retained. ANYfem application
validation remains pending the user's response to the ownership authorization
question. PR12 remains draft; the new contract is not fully consumer-accepted.

Human-authorised sequence: (1) repair existing planar-wall/cylinder tangency
failures, (2) resolve exact tangent junctions between wall branches on one pipe,
(3) explicitly remap attachments when a subsequent operation splits an older
joint edge. Coordinate consumer contracts with the ANYmesher and ANYfem workers;
geometry owns classification/topology, consumers own discretisation/application.

Restart source: `462ca593cec9bf36106b7baa72efe2587887625d`, confirmed on local and
GitHub main (the supplied "not pushed" note is stale). Work is isolated in
`.worktrees/intersection-continuation`, branch `codex/intersection-continuation`.
Preserve primary checkout edits and all historical evidence. No package release.

Stage-one question: do tangencies fail through out-of-domain curve parameters,
duplicate/degenerate traces, or face-loop validation of a legitimate contact?
Use minimal public-model replays and retained tracebacks to discriminate, then
fix the responsible owner operation. Required checks: analytic contact locus,
material conservation, shared source ownership, topology, query non-mutation,
idempotence and exact rollback. Keep numerical tolerances and quality gates.
Review the newly delivered Bezier arithmetic independently while investigating;
the reported 1,584-test run is prior evidence, not new qualification authority.

Coordination: ANYmesher received the consumer implementation brief. ANYfem
implementation delegation was rejected by automatic approval review citing
earlier ownership revocation; its read-only gap assessment/handoff was accepted.
Resolve any required ANYfem implementation authority with its owner/user after
the concrete gap assessment. No changes to ANYfem are authorised by this note.

Stage-one evidence: the stored-angle endpoint of a small elliptic arc normalized
to `1.0000000000065512`; preserving endpoint identity before division fixes the
typed-domain failure without widening tolerance. An exact tangent plate/pipe
returned no joint at seam rotations pi/8 and 0.2 and failed with a degenerate arc
at 0.7 on base 462ca593. The geometric tangent identity fixes all four tested seam
orientations. Focused tangent tests: 23 passed; the first combined tangent,
analytic support, branch model and public-routing run recorded 57 passed and four
test-authoring AttributeErrors, corrected in the subsequent 23-test run.
Logs: `reports/tangent-continuation-{baseline-seams,focused,focused-corrected}.log`
in the primary checkout. The specifically reported loop-self-intersection replay
has not yet been reproduced; do not claim that failure independently closed.

Independent Bezier review identified two constructor defects: unregularized
simple-fold endpoints and domain-roundoff bounds. The bounded repair retains the
existing 1e-12 domain tolerance by canonicalizing only accepted out-of-domain
endpoints; linear endpoint jets require an actual double root. Its three focused
test files passed 119 tests. Fold-offset behavior at scene scale remains a
separate review question, not an accepted regression or a waived criterion.

Consumer delta: ANYmesher main `847d6603288fb82a68f2b9ee9aec09eba48b551d`
adds schema-6 extrusion routing and ownership assertions. Its harder fixture
exposed a geometry boundary-projection upper-bound accuracy issue (direct
face inversion is accurate to 7.11e-16, trim-aware projection stops near 1e-9).
A bounded projection repair will retain global enclosure certification and the
consumer's 1e-10 shared-station threshold. The separate 7-degree cell-quality
failure remains mesher-owned. ANYfem requested four portable fixtures: fold-ended
cubic/oblique pipe, elliptic/parabolic pipe, tangent/secant/separated contact, and
subsequent splitting with vertex/member attachments. Deliver exact candidate
identity and expected topology/coordinates, not only a version number.

Stage-two decision: for supported parallel Bezier walls cutting one quadric,
eliminate their projected directrices with exact rational polynomials. Their
common generators contain every possible branch junction, including repeated
(tangent) roots. Isolate those roots and qualify candidates against both branch
charts. This addresses the subdivision ambiguity without increasing its budget.
Non-parallel cubic-wall models and overlapping supports retain their existing
typed refusal unless separately certified. Test known polynomial contacts,
nearby secant/separated cases, reversed charts and common affine images before
whole-model arrangement checks.

Stage-two implementation and review: both projected directrices are eliminated
with exact rational coefficients. An independent review found a missing second
visit on a self-crossing directrix and transformed residuals exceeding the
requested world tolerance. Both are now regression cases. Symmetric root sets
are paired using conservative enclosures of their isolated parameter intervals;
every accepted pair is checked in world coordinates. Ambiguity is a typed refusal.
`test_branch_wall_events.py` plus `test_branch_events.py`: 37 passed, including
whole-model material conservation, a shared high-valence tangent vertex, document
round-trip, insertion-order invariance and repeat apply. Repeat apply also exposed
the inversion cache in the plan checksum; explicit cache metadata now excludes
only that non-definition field while all geometric fields remain bound.

Stage-three implementation and review: `split_edge(..., remap_attachments=True)`
and batch preparation recover child parameters from exact world stations. The
first draft preserved target-edge relations but missed other references to a
member owning a regularized edge. Independent review reproduced a 3.22e-4
displacement; the repair snapshots and rebuilds source/member-target attachment
ranges and JunctionMemberUse ranges, including the no-target-attachment path.
Reversed uses and repeated splits are tested. Multiple splits in one batch also
require mapping planned contact stations to current member coordinates; linear
fraction arithmetic is insufficient. Failed explicit remaps restore identifiers
as well as topology and document contents. Source-edge interval attachments and
ambiguous inverse stations remain explicit refusals.

Current focused evidence: 13 edge/analytic attachment-remap tests passed; 23
portable-fixture/batch tests passed. Six authored/prepared fixture pairs and their
manifest are saved in primary `reports/consumer-contract-continuation/`; the
builder is tracked at `tools/general_intersections/consumer_contract_fixtures.py`.
The full local kernel run passed **1,676 tests, no failures/errors/skips**, in
407.77 seconds. A fresh Windows/Python 3.13 environment outside all checkouts
passed all six portable fixtures; all 75 installed package files matched the
candidate wheel byte-for-byte. The first install failed because the user process
could not read the sandbox-owned wheel; copying the identical wheel to the
temporary environment resolved access. Both failed attempts remain recorded.
Final documents were exported again using that installed wheel at primary
`reports/consumer-contract-9630844/`. This is development verification, not new
full-platform release qualification.

Delivery stages: `cd8b702` (tangency, branch validation, projection, plan binding),
`24f0150` (exact parallel-wall junctions),
`96308444054a7b002e2897cd3bbf46cf78021338` (attachment remapping and fixtures).
PR: https://github.com/audunarn/ANYgeometry/pull/12.
Candidate wheel SHA256:
`0dcc5c2585abf1e2e26b5844f4c440af7b0ed746fdcc155478424e1586f4846c`.
The development version is still 0.4.5; this wheel does not replace released
artifacts. Exact test identities and evidence hashes are recorded in
`intersection_continuation_evidence.md` beside this note (with the raw JSON
retained in the primary reports directory). Independent
OpenAI review findings and regressions were resolved. The supplementary Mistral
tangency review read the code but did not complete: its probe environment lacked
pytest and denied standalone Python before its turn limit. It is not counted as
a clean independent review. Earlier Bezier arithmetic review evidence remains
separate; no old release evidence was overwritten.

Consumer closure so far: ANYmesher `847d6603288fb82a68f2b9ee9aec09eba48b551d`
passed 26 focused tests against this exact wheel. Its former eight-sided child
projection failure is closed: 110 nodes, 19 Q8 and 4 T6 at h=0.75, with certified
positive mappings. The distinct 11-sided child's 7-degree corner remains
unaccepted under the existing 15-degree rule and is with the mesher owner.
Its bounded diagnosis found no authored acute corner (minimum source tangent
angle 119.22 degrees): a 0.049578 m side at h=0.75 caused excessive mapped seed
propagation. The automatic fallback returns a mesh but lacks strict high-order
certification, so remains unaccepted. A bounded mesher correction is authorised.
ANYfem received the read-only handoff; persistence, viewport and project-reference
acceptance there remain pending; explicit user authorization was requested for
that separate worker's validation/fixes after the earlier automatic approval
rejection. No consumer package release or GUI default
change is authorised by this continuation.

Study record, 2026-10-01. Base: `main` 9dbf3e4 (0.4.5) plus the unmerged performance branch
`claude/perf-hunt-046`. Branch: `claude/quadric-study`. Nothing here changes a released contract,
and nothing should merge before the testing-regime changes settle.

**Status (later the same day).** The recommendation below was approved and implemented on this branch:
see [cone supports and quadric intersection branches](../../docs/QUADRIC_INTERSECTIONS.md) and the
`quadric_algebra`, `quadric_curves`, `quadric_supports` and `quadric_events` modules. The private prototype
and its fuzz/oracle scripts that this record cites were superseded by the production modules and tests
(`tests/test_quadric_*.py`, `tests/test_cone_intersections_engine.py`) and removed; they remain in the history
at `011d161` and `95ea5c1`. The probes that only exercise the public workflow stay in
`tools/general_intersections/quadric_study/`.

**Question.** A cone meets a cylinder, say with its axis 10 degrees off perpendicular. How can the
intersection engine become general enough to handle that, and what else would the same change buy?

## Answer in brief

1. **Today the pair is refused.** `plan_intersections` stops at the first cone face with
   `face N has no qualified analytic or bilinear support`. The legacy certified engine
   (`query_intersection`) classifies the pair but is not a usable route (section 1).
2. **The structure generalizes cleanly.** Plane, Cylinder and Cone are all quadrics, and Cylinder and
   Cone are ruled: `S(t, s) = P(t) + s D(t)`. Along a ruling a second quadric gives
   `A(t) s^2 + B(t) s + C(t) = 0`. That is exactly the structure `CylinderIntersectionCurve` already
   uses, with a degree-8 tangent-half polynomial for the events. A prototype reproduces the production
   cylinder charts and extends to cones with verified completeness and accuracy.
3. **The chart side is small.** Letting an arrangement domain be a Cone face is about 20 changed lines
   in `material_arrangement.py`; the arrangement itself is chart-agnostic.
4. **The curve must stay analytic.** The same arrangement is 225 to 700 times slower when the exact
   curve is replaced by a fitted Bezier chain. A numeric or Bezier fallback is not a route to
   generality inside this engine.
5. **Recommendation.** Add a generalized quadric branch curve as a new family beside
   `CylinderIntersectionCurve` (leave the validated cylinder code alone), starting with
   cone x cylinder and cone x plane. Decisions that belong to the owner are in section 6.

## 1. What the kernel does today (measured)

Test model: a 12-facet cylinder (radius 2, axis z) and an 8-facet cone frustum (radius 0.5 to 1.0
over 5 units) whose axis is perpendicular, or 10 degrees off perpendicular, through the cylinder axis.

| Path | Plane/Cylinder pairs | Cone pairs |
| --- | --- | --- |
| `plan_intersections` (exact batch engine, used by the mesher) | exact; 12 x 12 facets, 10 deg off perpendicular: plan 2.4 s, apply 0.44 s | refused, see above |
| `query_trimmed_surface_charts` | Plane, Cylinder | `unsupported intersection surface` (the mesher already checks for this) |
| `query_intersection` (legacy certified subdivision) | n/a | 8 of 96 facet pairs cross at both 0 and 10 deg; 9.0 s for all 96 (0.094 s per pair) |
| `apply_imprint` (legacy) | n/a | fails on the first crossing pair at both angles: `imprint component does not map to one active face descendant` |

The legacy engine is also fragmentary here: 3 of the 8 crossing pairs at 10 degrees (4 at 0) return 4 to
14 components, mostly two-point pieces with interior ends. What it persists is quadratic Bezier pieces within the
1e-8 curve-fit residual. ANYmesh already meshes an isolated cone face with an exact developable chart
(`_conical_chart.py`) and states that cone intersections are outside the batch contract, so the gap is
on the geometry side.

Where generality stops in the code: `_pair_paths` is an `if/elif` on (Plane, Cylinder); curve families
(`LinePath`, `EllipticArc`, `CylinderIntersectionCurve`, `BezierPath`) are matched by `isinstance`
chains in `plane_roots`, `_point_parameters`, `_curve_junctions`, `MaterialDomain.curvature`,
`_coincident` and `freeze_edge`. The general engine's cylinder-specific footprint is roughly 60 references
in nine files; the 2,700 lines of cylinder atlas/patch qualification are historical and separate.

## 2. The structural idea

For a ruled first support and any implicit quadric `x.M.x + 2 l.x + c = 0`:

* `A = D.M.D`, `B = 2 (P.M.D + l.D)`, `C = P.M.P + 2 l.P + c` are trigonometric polynomials of degree
  at most 2, built exactly from the doubles (Cylinder: constant `D`; Cone: constant `P`, the apex).
* The branches are `s = (-B +- sqrt(B^2 - 4AC)) / (2A)`. Folds are roots of the degree-4 discriminant
  (odd multiplicity), double contacts are even multiplicity, poles are roots of `A`, and finite-patch
  boundaries are roots of `A s0^2 + B s0 + C` and of resultants against the other support's end and seam
  planes. All are Sturm-isolated with the existing `isolate_real_roots`.
* Curve events against any other quadric are the resultant of two such quadratics: again a degree-8
  tangent-half polynomial for Cylinder and Cone supports.
* Sphere, ellipsoid and similar quadrics are free **as the second support**. The ruled partner supplies
  the angle. A torus is quartic and is not covered.

Choice of the angle-supplying support matters for regularity. For the 10 degree case cone-first has one
smooth closed loop (four charts, cut only at seam events; no folds, no poles); cylinder-first folds twice
per loop (eight charts, four folds). The persisted curve should use whichever has fewer folds.

## 3. Prototype and evidence

`src/anygeometry/_quadric_branch.py` (private, not exported, no serialization) and
`tests/test_quadric_branch_prototype.py`, both removed after the production implementation replaced them
(history `011d161`); the claims below were reproduced there with the `qb_*.py` scripts of `95ea5c1`.

| Claim | Evidence |
| --- | --- |
| Reproduces the production cylinder charts | 5 configurations x 64 facet pairs (perpendicular, equal radius, skew, tangent, 10 deg): identical chart sets (angular interval and branch) in all 320 pairs; point difference at equal angles at most 2.6e-14. The singular equal-radius perpendicular case is included. |
| Complete | An independent grid oracle (contour of the implicit equation on a dense chart, no use of A, B, C or events) over 6 pairings x 150 random configurations: about 247,000 contour points, 1,700 charts, 831 folds, 88 poles, none uncovered away from fold angles. Pairings: cone/cylinder both orders, cone/cone, cone/plane, cylinder/plane, cylinder/cylinder. |
| Accurate | Every chart lies within 1e-13 (distance) of both surfaces in the 10 degree case; closed loop closes to 1e-13. Random pairings stay below 1e-10 (distance estimate; floating-point noise near a cone apex makes this estimate pessimistic there). |
| Events against another quadric | 471 charts against random planes, cylinders and cones: 240 exact roots, 240 sign changes found by dense sampling, none missed. About 64 ms per chart and quadric in unoptimized Python. |
| Sphere as second support | 60 random cone/cylinder x sphere pairs: 178 charts, 58 folds, no failures. |
| Fast enough | All 96 cone x cylinder facet pairs: 0.4 to 1.1 s (4 to 11 ms per pair), versus 9.0 s for the legacy engine, and exact. |
| Cone as a material chart | `MaterialDomain` accepts Cone faces with about 20 changed lines. Generator, ring and combined cuts on upright, 10 degree and generic-tilt cone facets give the exact product areas. |
| Curve class matters | Same arrangement, same cylinder x cylinder facets, same tolerances: analytic traces 0.09 to 0.22 s per face; degree-5 Bezier fits (1e-9) 25 to 120+ s per face, 225x to 700x slower (one case was cancelled at the 120 s budget). A Bezier-trace cone x cylinder arrangement at 10 degrees did succeed on all 8 cone facets (two cells each, area conserved) but took 28 to 44 s per facet. |

Fail-closed behavior in the prototype (typed `GeometryError`, never a guess): cone apex on the other
support within tolerance (the exact structure then depends on rounding of the apex position), contact
of multiplicity three or more, and rulings parallel to a plane for every angle.

## 4. Options

* **A. Numeric or Bezier fallback inside the batch engine.** Rejected by the data above: the legacy
  provider is fragmentary and persists 1e-8 fits, and Bezier curves make the exact arrangement two to
  three orders of magnitude slower.
* **B. Exact generalized quadric branch (recommended).** Covers Cone now and Sphere or any quadric as a
  second support later. Cost: a new curve class, a geometry schema bump (6), junction dispatch, chart and
  `_child_support` work, qualification, and consumer coordination.
* **C. Revolution and freeform surfaces (torus, profiles, Ruled/Coons/NURBS).** Needs certified numeric
  surface-surface tracing and an arrangement that can use numeric curves without exact resultants. A
  separate project; B does not block it.
* **D. Architecture.** Define a small curve protocol (evaluate, derivative, certified bounds, subcurve,
  roots on a quadric, parameters of a point) and a support registry, so the next family does not edit six
  `isinstance` chains. Worth doing with B, not before it.

## 5. What a full slice for the cone still needs (not done here)

Derivatives and second derivatives including one-sided limits at fold ends; certified interval bounds;
`subcurve` and affine `transformed`; `point_parameters`; junction dispatch for branch x {line, ellipse,
branch, Bezier}; `_pair_paths` for Cone; `_child_support` for Cone; exact cone area in
`query_trimmed_surface_charts` (the current `area_jacobian` for a Cone is only an upper bound); beam x
cone supports in `member_arrangements`; serialization and `EXACT_CURVES`; the rulings case (parallel
supports give generator lines); a qualification matrix (perpendicular, 10 degrees, skew, tangent,
coaxial, wide cone, apex cases, cone x plane, cone x cone); and mesher consumption of cone joints. For
scale: the analogous cylinder code is `exact_curves.py` (588 lines), `cylinder_curve_events.py` (144),
`quadric_curve_events.py` (67) and `analytic_supports.py` (220). Arrangement speed with an exact cone
curve is expected to match the cylinder analogue (same polynomial degrees) but was not measured.

## 6. Decisions for the owner

1. Schema 6 and a new curve class beside `CylinderIntersectionCurve`, or migrate the cylinder class later.
2. Scope of the first slice: Cone only, or also a bare-quadric second support (sphere) with its charts.
3. Policy for apex contact, tangency and higher-order contact: refuse with typed errors (as prototyped)
   or support.
4. Rule for choosing the angle-supplying support (fewest folds).
5. Who owns conical joint meshing in ANYmesh once geometry provides cone charts and joints.
6. How the new family is qualified under the testing-regime changes now in progress.

## 7. Other observations

* 16 or more mutually crossing plates through one common point fail with
  `arrangement failed material conservation` (12 and 14 pass; 20 plates in generic position plan and apply
  in 187 s and produce 16,139 faces). Not investigated; reproduction is `quadric_study/mutual2.py`.
* The prototype's distance-to-surface estimate is meaningless at a cone apex (the gradient vanishes);
  checks skip that point.

## Reproduction

```
python -m pytest tests/test_quadric_intersection_curve.py tests/test_quadric_supports.py tests/test_cone_intersections_engine.py
python tools/general_intersections/quadric_study/curve_repr_cost.py tools/general_intersections/quadric_study 5 1e-9
```
Each remaining script takes the directory containing its siblings as its first argument; the prototype's
`qb_*.py` scripts are in `95ea5c1`. Full suite on this
branch: 10 failures, all in the git-environment release-authority tests that also fail on `main`
(16 on the earlier run; the set varies with the environment), no new failures.
