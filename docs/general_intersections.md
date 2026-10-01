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

`has_current_intersection_preparation(model)` checks a local owner receipt from
a complete successful batch. The receipt binds model identity, revision,
complete checksum, covered faces and immutable plan content; partial batches
and disabled face classification provide no exemption. It is not serialized.
`clone_prepared_geometry(model)` carries that proof only across an exact,
independently mutable mesh-attempt copy after verifying both checksums.
Ordinary clones, document loading and edits require fresh classification.
Consumers use these owner APIs to avoid arranging prepared descendants again.

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
schemas 1â€“4. Readers supporting only schema 4 must reject new documents clearly.
Coordinate units, transforms and automation protocol 1 retain their established
semantics. CAD export and CRS reprojection are separate work.

## Acceptance and compatibility

The [final hosted matrix](https://github.com/audunarn/ANYgeometry/actions/runs/36829763205)
passes **32 of 32 jobs**. Its [installed evidence](../reports/general_intersections/installed-qualification-36829763205.json)
binds geometry commit `9e61831b72f01c9beccfce67b8752117bb87e4b8`
(source unchanged from `f7b0a05e9a4a414c6de336431929f2e34969bdea`),
mesher `e21c0fc93662776762430e14450d54ac9192e2e8` and
FEM `83f3d6c81e405ca45f5b7edd7cb7c2afd32804c4` to exact wheel hashes,
resolved dependencies, package byte checks and installed origins.

| Configuration or case | Accepted evidence |
| --- | --- |
| Windows, Linux, macOS 15 Apple silicon and macOS 15 Intel; Python 3.11–3.14 | Full geometry kernel matrix passes; local Python 3.14 suite has 1,095 passes |
| Linux/Python 3.11 minimum dependencies | NumPy 1.26.0 and Shapely 2.0.0 gate passes |
| Installed universal geometry wheel; Python 3.13 on all four platforms | CLI, schema, typing, serialization, representative operations and optional-Shapely boundaries pass |
| Installed public and pinned SDK-2 MCP consumers | Established Windows/Linux/Apple-silicon consumer gates pass; all imports remain outside source checkouts |
| Windows/Python 3.13 full compiled mesher suite | 1,943 pass, 66 explicit skips, clean process completion |
| Windows/Python 3.13 full FEM suite | 1,044 pass, 130 explicit GUI/optional skips, clean process completion |
| Installed generalized FEM; Windows/Linux/Apple silicon, Python 3.13 | Sixteen accepted meshes per platform, forty-eight total; all expected solver admissions pass |
| Exact supplied command replay; target sizes 0.5 and 0.25 | Automatic linear policy; cylinder undo/replacement, requested segment/spacing parameters, all six owner-pair joints, material, source identity/content, downstream references, save/load, undo/redo and regeneration pass |
| Parallel, perpendicular, skew and tangent finite cylinders | Complete shared-node interfaces and analytic material checks; skew fixture checks all four branches with independent eliminated equations, 70-digit residuals and finite-height interval coverage |
| Oblique ellipse; one circular plus two elliptic cuts on one plate | Closed joint-loop coverage, exact material conservation and source/save-load pass |
| Concave two-hole plate, three walls and two beams | Complete material intervals, T-junctions, shared node sequences, source preservation and save/load pass |
| Growing fixtures with 1, 8 and 25 interior-ended cuts | Default automatic meshes pass complete shared-node and material checks without a model-count exception |
| Cylinder material with 1, 2 and 4 holes | General physical charts pass exact cylinder residuals, independent unrolled material areas and source/save-load checks |
| 260-sector cylinder beyond historical atlas scope | Frontal native linear mesh, target 0.05: exact area 2pi and all 260 shared panel interfaces pass; kernel-bound dispatch also covers 257 sectors and a boundary beyond historical patch occurrence limits |

The [current local evidence](../reports/general_intersections/final-user-policy-evidence.json)
records exact source/log hashes and sixteen installed Windows meshes in two
isolated environments with byte-identical wheels. The already passing capacity
check is reused rather than repeated. GUI skips and optional-tool skips are
explicit; they do not establish GUI execution coverage. Intel native consumer
coverage is not inferred from its kernel or wheel checks.

Candidate versions remain geometry 0.4.4, mesher 0.5.1 and FEM 0.4.1.
This is coordinated unpublished development: filenames and version numbers
alone cannot identify the required artifacts. The released comparison uses
geometry 0.4.3, mesher 0.5.0 and FEM 0.4.1; the original replay fails in
structural preparation with source unchanged. Mesher 0.5.0 lacks the later
recovery API, so that comparison uses its released application defaults.

Following the user's clarification, default direct calls use the existing
15-degree admission floor and keep recovery opt-in. A conforming diagonal
beam in a 2-by-1 plate imposes a 26.565-degree corner and now meshes directly.
An explicit caller-supplied 30-degree policy still rejects it atomically.
The historical preferred S3 repair target remains 30 degrees, without becoming
a default rejection threshold. Other numerical and quality limits remain
unchanged. Favorable parameter-chart angles cannot override physical quality.

Primary cylinder dispatch selects general material charts for holes and for
components beyond historical atlas/patch cardinality scope before invoking
historical qualification. Direct historical APIs retain their fixed evidence
scope; a failed historical certificate never becomes a general acceptance.

Equivalent and unambiguous feature output-role changes retain structural owners,
sections, loads, supports and authored parameters. Ambiguous bay-count changes
require an explicit semantic role remap and remain atomic refusals. Attachments
without an exact source axis, point or edge remapping definition are also refused
atomically. These refused inputs have no accepted-coverage claim. Positive-area
overlaps still require an explicit ownership operation.

## Retained earlier evidence

These records retain their original source/artifact scope and conclusions.
They do not qualify later changes. Complete logs and artifacts remain attached
to each hosted run; compact records retain failed and unaccepted configurations.

| Hosted record | Original outcome |
| --- | --- |
| [36795873823](../reports/general_intersections/installed-qualification-36795873823.json) | Earlier installed cylinder checks pass; overall matrix fails five planar-area gates |
| [36798593408](../reports/general_intersections/installed-qualification-36798593408.json) | All 32 jobs pass the earlier six-case consumer scope |
| [36809696642](../reports/general_intersections/installed-qualification-36809696642.json) | 31/32; Windows and Apple silicon twelve meshes each; Linux four before skew timeout |
| [36814130573](../reports/general_intersections/installed-qualification-36814130573.json) | 30/32; Apple silicon twelve meshes; Windows/Linux four each before skew timeout |
| [36818408300](../reports/general_intersections/installed-qualification-36818408300.json) | 29/32; all three native jobs accept four meshes each before skew timeout |
| [36823318546](../reports/general_intersections/installed-qualification-36823318546.json) | 29/32; seven meshes per platform before incorrect Sheet-target preflight lookup |
| [36825533614](../reports/general_intersections/installed-qualification-36825533614.json) | 31/32; Linux/Apple silicon fifteen meshes each; Windows four before skew timeout, with S3 preparation complete at 879 seconds |

The local [previous candidate](../reports/general_intersections/final-local-evidence.json)
passes fifteen meshes and its recorded full suites. The current candidate adds
capacity/default-policy coverage and passes the final complete matrix, preserving
the same 900-second per-command budget. Earlier failures remain failures in
their original evidence. Released 0.4.3/0.4.4 and specialized cylinder records
in [compatibility](compatibility.md) also retain their historical scope.

## Consumer handoff

The implementation is reviewable in four stages. Follow-up corrections stay
on the same development branches and retain the failed evidence above.

| Stage | Initial review checkpoint | Final owner binding |
| --- | --- | --- |
| Exact curves and schema 5 | [Geometry b2e20bd](https://github.com/audunarn/ANYgeometry/commit/b2e20bd) | Geometry source f7b0a05; schema-5 codecs and analytic curve regressions |
| Original-operand arrangements, batch preparation and material charts | [Geometry 6557b2c](https://github.com/audunarn/ANYgeometry/commit/6557b2c) | Geometry source f7b0a05; complete content binding, exact prepared copies and rollback |
| Mesher and FEM integration | [Mesher dda50e4](https://github.com/audunarn/ANYmesh/commit/dda50e4), [FEM fa7e666](https://github.com/audunarn/ANYfem/commit/fa7e666) | Mesher e21c0fc and FEM 83f3d6c; detached preparation, owner targets and physical material charts |
| Installed qualification and compatibility handoffs | [Geometry 11b989a](https://github.com/audunarn/ANYgeometry/commit/11b989a) | Sixteen-mesh installed gate per native platform; hashes and outcomes in the current evidence |

The [owner handoff table](general_intersections_owner_handoffs.md) records the
schema, exact-curve, coordinate and preparation contracts for other consumers.
These commits do not merge or publish a new package. Candidate versions alone
cannot identify this unpublished coordinated development; use source commits
and wheel hashes.

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
