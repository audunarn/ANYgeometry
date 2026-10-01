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
| Windows, Python 3.14 kernel | 1,074 regressions pass, including growing original-operand arrangements (1, 8 and 25 cuts), repeated Member visits and analytic planar-area oracles |
| Windows, Python 3.13 compiled mesher | 147 metric/cache/gradation/frontal/seeding/automatic-route and S3 checks pass; five staged chart-refinement contract checks included |
| Windows, Python 3.13 FEM lifecycle | 12 intersection and reference-retention regressions pass |
| Exact installed ANYfem replay, targets 0.5 and 0.25 | Both accepted on Windows/Linux/macOS arm64; source, joints, material, save/load and equivalent regeneration checks pass |
| Installed tangent cylinders | Accepted on those three platforms; complete axial joint and material checks pass |
| Parallel/perpendicular cylinder pairs | Accepted on those three platforms; independent angular/axial coverage and high-valence checks pass |
| Skew finite unequal-radius cylinders | Default automatic meshes accepted on those three platforms; independent eliminated cylinder equations, 70-digit residuals, finite-height coverage, shared nodes and material checks pass |
| Changed radius with unchanged output roles | Kernel owner identity/parameter tests and FEM section/load/support, save/load and undo/redo checks pass |
| First expanded hosted run | 27/32 jobs pass; Intel kernels and minimum dependencies fail one planar quadrature regression; analytic integration fix passes locally, updated hosted matrix pending |

Installed reports bind wheel hashes, resolved dependencies, package byte checks
and origins in clean environments outside source checkouts. Candidate package
versions remain geometry 0.4.4, mesher 0.5.1 and FEM 0.4.1; filenames/version
numbers alone do not identify these unpublished artifacts. Released comparison
uses geometry 0.4.3, mesher 0.5.0 and FEM 0.4.1: the original replay fails in
structural preparation with source unchanged. Mesher 0.5.0 lacks the later
automatic recovery API; its comparison uses released application defaults.

Historical matrix results in [compatibility](compatibility.md) apply only to
their recorded source/artifacts. Windows/Linux/macOS 15 arm64/macOS 15 x86_64
on Python 3.11–3.14 still require extension-specific evidence. Intel downstream
native-package coverage is not implied by kernel coverage. The Linux/Python
3.11 minimum-dependency and installed-wheel optional-Shapely gates remain
required. Numerical and mesh-quality thresholds remain unchanged.

The generalized installed-consumer CI builds native mesher commit
`dda50e4c813834af10bf52bf4451ce55bda83315` and FEM commit
`fa7e666f8cd236c17591ccde67f9b34f20f45f33`, then installs those wheels and the
single candidate geometry wheel in clean environments outside all checkouts.
It requires compiled triangulation, checks every candidate package file against
the wheel, records import origins, resolved dependencies and artifact hashes,
and requires accepted default automatic meshes and independent joint/material
oracles. The Windows/Linux/Apple-silicon jobs do not imply Intel consumer
acceptance. Setup failures and unavailable artifacts are explicit failures.

The compact [installed evidence](../reports/general_intersections/installed-qualification-36795873823.json)
binds these passes to the recorded source commits and artifacts; the overall
[hosted run](https://github.com/audunarn/ANYgeometry/actions/runs/36795873823)
failed its five planar-area gates. Updated source is not accepted merely because
those earlier installed artifacts passed.

Outstanding implementation/acceptance includes remaining pair-policy routing,
bay-count-changing feature regeneration,
remaining attachment parameter remaps and the complete growing/concave/holed/
seam/high-valence matrix. Typed failures and unavailable jobs are not passes.

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
