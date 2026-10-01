# General intersection development contract

This is development on `codex/general-intersections`, not a new release or a
claim that every planned case is accepted. Existing release ledgers, artifacts
and specialized cylinder qualification records retain their original scope.
The [living implementation record](../reports/general_intersections/task.md)
contains the reproduction, decisions, failed experiments and current evidence.

## Geometry owner API

`EllipticArc` represents an affine elliptic arc, including an oblique
plane/cylinder cut. `CylinderIntersectionCurve` retains immutable cylinder
supports, a certified real branch chart and an affine copying transform.
Evaluation/display witnesses do not define the intersection. Analytic support
equations, exact rational root counts, branch transitions, seams and finite
material clipping determine its components.

The root-exported batch API is:

```python
from anygeometry import ConnectionIntent, plan_intersections, apply_intersections

operands = (
    *(model.handle("face", identifier) for identifier in model.faces),
    *(model.handle("member", identifier) for identifier in model.members),
)
plan = plan_intersections(model, operands, policy=ConnectionIntent.CONNECT)
application = apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
```

Planning is read-only and binds model identity, revision, complete persisted
content and immutable plan definitions. Applying is atomic and deterministic;
reapplying an accepted plan reuses its result. A stale, changed, wrong-model,
cancelled or exhausted plan raises `GeometryError` without an accepted partial
topology. `IntersectionBatchPolicy` supplies optional predicate/pair budgets and
a cancellation callback. Its defaults impose no operand/intersection count cap.

Single-pair Member CONNECT and IMPRINT plans use the same arrangement engine
for every qualified occurrence and material interval. CONTACT_ONLY and
KEEP_DISCONNECTED preserve topology and retain all qualified relations.
REUSE_EXISTING requires compatible relations at every parent interval; a
same-owner relation at another station is insufficient. Intent tags, unowned
face attachments and rollback are retained. Legacy relation-only plans remain
revision-bound; repeat their read-only planning after a successful edit.

The engine arranges original operands before fragmentation. Shared physical
joints and material decomposition seams retain separate provenance. Source
lineage, Sheet/Member identity, FaceUse orientation and labels remain owner
data. Positive-area overlaps require an explicit ownership operation and are
not silently removed or accepted by the default batch policy.

`query_trimmed_surface_charts(model, operands=None, *, expected_revision=None,
cancellation_check=None)` accepts Faces, FaceUses and Sheets; omission selects
all faces. Plane charts use their stored affine basis and Cylinder charts their
angular/axial basis. Exact boundary loops include all holes and seam sides.
`evaluate_trimmed_surface_chart` accepts finite arrays shaped `(..., 2)` and
returns shape-preserving world positions. `require_material=True` rejects void
samples. `validate_trimmed_surface_charts_binding` rejects changed source,
occurrences, definitions, tolerance or area evidence. Reused qualification is
bound to complete content, including direct persisted-data edits.

Geometry documents now write **schema 5** and read all previously supported
schemas 1–4. Readers supporting only schema 4 must reject new documents clearly.
Coordinate units, transforms and automation protocol 1 retain their established
semantics. CAD export and CRS reprojection are separate work.

## Current evidence and remaining acceptance

| Configuration or case | Current extension evidence |
| --- | --- |
| Windows, Python 3.14 kernel | 1,086 regressions pass, including growing original-operand arrangements (1, 8 and 25 cuts), repeated Member visits, all pair intents and analytic planar-area oracles |
| Windows, Python 3.13 compiled mesher | 147 metric/cache/gradation/frontal/seeding/automatic-route and S3 checks pass; five staged chart-refinement contract checks included |
| Windows, Python 3.13 FEM lifecycle | 12 intersection and reference-retention regressions pass |
| Exact installed ANYfem replay, targets 0.5 and 0.25 | Both accepted on Windows/Linux/macOS arm64; source, joints, material, save/load and equivalent regeneration checks pass |
| Installed tangent cylinders | Accepted on those three platforms; complete axial joint and material checks pass |
| Parallel/perpendicular cylinder pairs | Accepted on those three platforms; independent angular/axial coverage and high-valence checks pass |
| Skew finite unequal-radius cylinders | Default automatic meshes accepted on those three platforms; independent eliminated cylinder equations, 70-digit residuals, finite-height coverage, shared nodes and material checks pass |
| Changed radius with unchanged output roles | Kernel owner identity/parameter tests and FEM section/load/support, save/load and undo/redo checks pass |
| Expanded hosted matrix at `dedbb6130df37b9a182ebe15661323828d27afd5` | All 32 jobs pass: four-platform kernel matrix, minimum dependencies, wheel smoke and installed public/MCP/general consumers |
| Oblique closed ellipse and concave two-hole/three-wall/two-beam fixtures | Current local and clean installed default application meshes pass independent material/joint/beam coverage, source/reference non-mutation and save/load; hosted verification of the follow-up fixes is pending |
| One circular and two elliptic cuts on one plate | Local and clean installed automatic admission, complete closed shared-node loops, all material areas and source/save-load checks pass |
| Growing default application fixtures | 1, 8 and 25 interior-ended parallel stiffeners pass local and clean installed material and complete shared-node checks; no model-count exception is used |

Installed reports bind wheel hashes, resolved dependencies, package byte checks
and origins in clean environments outside source checkouts. Candidate package
versions remain geometry 0.4.4, mesher 0.5.1 and FEM 0.4.1; filenames/version
numbers alone do not identify these unpublished artifacts. Released comparison
uses geometry 0.4.3, mesher 0.5.0 and FEM 0.4.1: the original replay fails in
structural preparation with source unchanged. Mesher 0.5.0 lacks the later
automatic recovery API; its comparison uses released application defaults.

Historical matrix results in [compatibility](compatibility.md) apply only to
their recorded source/artifacts. Windows/Linux/macOS 15 arm64/macOS 15 x86_64
on Python 3.11–3.14 passed the extension-specific kernel matrix at the recorded
commit. Intel downstream
native-package coverage is not implied by kernel coverage. The Linux/Python
3.11 minimum-dependency and installed-wheel optional-Shapely gates passed there.
Changed follow-up sources require fresh evidence. Numerical and mesh-quality
thresholds remain unchanged.

The generalized installed-consumer CI builds native mesher commit
`e93073871937d91d2fab512451bfe972d6e53fe8` and FEM commit
`fac3c9a073c62485540a8629f8b2a3aeec64b730`, then installs those wheels and the
single candidate geometry wheel in clean environments outside all checkouts.
It requires compiled triangulation, checks every candidate package file against
the wheel, records import origins, resolved dependencies and artifact hashes,
and requires accepted default automatic meshes and independent joint/material
oracles. The Windows/Linux/Apple-silicon jobs do not imply Intel consumer
acceptance. Setup failures and unavailable artifacts are explicit failures.

The compact [installed evidence](../reports/general_intersections/installed-qualification-36795873823.json)
binds these passes to the recorded source commits and artifacts; the overall
[hosted run](https://github.com/audunarn/ANYgeometry/actions/runs/36795873823)
failed its five planar-area gates. The subsequent
[installed evidence](../reports/general_intersections/installed-qualification-36798593408.json)
records the six-case consumer passes and hashes for the
[fully passing run](https://github.com/audunarn/ANYgeometry/actions/runs/36798593408).
Neither record includes the later material-fixture fixes.

The expanded [ten-case run](https://github.com/audunarn/ANYgeometry/actions/runs/36809696642)
passed 31 of 32 jobs. Its [installed evidence](../reports/general_intersections/installed-qualification-36809696642.json)
records twelve accepted meshes each on Windows and Apple silicon, and four
accepted meshes on Linux before the skew-cylinder case exceeded its unchanged
900-second subprocess budget. Linux's remaining cases are unaccepted in that
run. Kernel, minimum-dependency, wheel, public-consumer and MCP jobs all passed.
Subsequent repair/compatibility changes require fresh installed evidence.

The next [hosted run](https://github.com/audunarn/ANYgeometry/actions/runs/36814130573)
passed 30 of 32 jobs. Its [installed evidence](../reports/general_intersections/installed-qualification-36814130573.json)
records all twelve Apple-silicon meshes; Windows and Linux each accepted four
before the skew fixture exceeded the unchanged 900-second budget. Retained
phase evidence identifies repeated historical overlap qualification on prepared
general domains as substantial work. The next candidate uses one read-only
geometry-owner batch for this audit, retaining positive-area rejection and
legacy unsupported-surface qualification. Thirty-one focused mesher checks pass;
full regression and installed/platform acceptance remain pending.

A conforming diagonal beam in a 2-by-1 plate imposes a 26.565-degree corner.
The strict mesh API retains its 30-degree gate and rejects this design with
source unchanged. The application's explicit automatic S3 policy admits it
under its existing 15-degree floor. Parameter-chart quality must be checked
after lifting to physical coordinates; favorable chart angles cannot override
the physical acceptance policy.

The current generator contract preserves downstream references for equivalent
and unambiguous output-role changes. Changing the generator bay count still
requires an explicit semantic role remap; it is an atomic refusal rather than
an inferred reassignment. Non-point face attachments require an exact source
axis or point/edge target for remapping. Other attachment definitions are
refused atomically. These cases have no new acceptance claim.

The final follow-up sources still need the complete installed/platform and
full mesher regression evidence. Typed failures and unavailable jobs are not
passes.

## Consumer handoff

- ANYmesher/ANYfem use the batch preparation contract on detached design copies,
  then remap accepted mesh associations through source lineage. Geometry owns
  intersection/topology truth; consumers own discretization and quality gates.
- ANY3dView evaluates schema-5 curves through geometry owner APIs. Exact-curve
  decoding/evaluation checks pass on Windows/Python 3.13; the Python 3.14 local
  viewport check skipped because its triangulation extra was unavailable.
- ANYfileio/native document readers should delegate schema-5 decoding to the
  matching geometry owner. Preserve semantic dependency failures when that
  owner is missing. Older readers must not reinterpret new curves as circles
  or polylines.
- ANYfileio-occt/CAD owners must preserve the elliptic basis and the two analytic
  cylinder supports/branch definitions when defining a future export contract.
  Any approximation needs an explicit separate error contract. This work does
  not activate CAD export.
- ANYgeometry-mcp discovery/schema resources and ANYopenSoft capability ledgers
  need new evidence before claiming generalized capability. Frozen release and
  specialized cylinder entries remain historical; this development report
  cannot replace their acceptance records.
