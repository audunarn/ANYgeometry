# ANYgeometry

ANYgeometry is the lightweight, deterministic structural-surface geometry
kernel shared by ANYmesher, ANYfem, and ANYstructure. It owns neutral geometry,
topology, persistent entity references, semantic groups, geometry evaluation,
edits, intersections, and geometry serialization. Meshing, finite-element
attributes, materials, loads, solver state, project data, and GUIs stay in
their owning packages.

The kernel is intentionally focused on engineering plates, beams, panels,
cylinders, cones, frames, and shell intersections. It is not a general solid
CAD kernel and does not depend on OpenCASCADE, Gmsh, a GUI toolkit, an FE
package, or a solver.

## Installation

ANYgeometry requires Python 3.11 or newer and NumPy. Until the first compatible
release is available from PyPI, install the sibling checkout directly:

```powershell
python -m pip install -e C:\Github\ANYgeometry
```

Planar clipping and strict planar qualification use the optional Shapely
backend. Install the `planar` extra for those workflows:

```powershell
python -m pip install -e "C:\Github\ANYgeometry[planar]"
```

## Quick start

Build topology directly:

```python
from anygeometry import GeometryModel

geometry = GeometryModel()
vertices = geometry.add_points(
    [(0, 0, 0), (4, 0, 0), (4, 3, 0), (0, 3, 0)]
)
face_id = geometry.add_plate(vertices)
face_ref = geometry.entity_ref("face", face_id)  # local compatibility reference
face_handle = geometry.handle("face", face_id)   # model-bound public identity
geometry.add_to_group("deck", [face_ref])
```

Or use a structural generator:

```python
from anygeometry.generators import stiffened_panel

geometry = stiffened_panel(
    4.0,
    3.0,
    longitudinal_spacing=1.0,
    transverse_spacing=2.0,
    semantic_group="deck",
)
deck_faces = geometry.group("deck")
stiffener_edges = geometry.group("longitudinal_stiffeners")
```

`EntityHandle(model_id, kind, id)` is the model-bound cross-package identity.
The legacy `EntityRef(kind, id)` remains a compact local compatibility value.
IDs are allocated monotonically and are never reused, including after rollback
or compatibility undo. Splitting or fragmenting an entity records descendants,
updates groups, tags, and structural uses, and lets clients resolve a stale
selection explicitly.

## Geometry model

- `Vertex`, `Edge`, and `Face` provide topology with persistent IDs.
- Public entity stores and records are read-only. Edits go through atomic model
  methods or a nested `GeometryModel.transaction()`.
- Model identity, revision, tolerance, units, and coordinate arrays are
  owner-controlled. Use `set_document_settings(...)` for revisioned coordinate
  or tolerance changes; returned arrays cannot be written in place.
- `Part`, `Sheet`, `FaceUse`, and `Coedge` persist plate ownership/incidence;
  `Member` and `MemberEdgeUse` persist a physical beam axis across edge splits.
- `Attachment` and `Junction` distinguish declared beam/plate and beam/beam
  relationships from mere geometric coincidence.
- `Straight`, `Arc`, and lightweight Bezier `Spline` curves are topology-owned.
- `Plane`, `Cylinder`, `Cone`, `RuledSurface`, and explicit or topology-backed
  Coons patches provide evaluation and local UV coordinates.
- Groups carry geometric meaning such as `shell`, `deck`, `bottom`,
  `boundaries`, `longitudinal_stiffeners`, and `ring_stiffeners`.
- Tags provide lightweight geometry annotations. Materials, sections,
  thicknesses, loads, supports, and mesh controls must reference geometry from
  outside this package.

General operations include projection, closest-point queries, transforms,
edge/face splitting, trimming, holes, fragmentation, and shell/shell
intersection imprinting. Analytical line/plane/cylinder intersections cover
common structural cases; a deterministic sampled fallback is available for
other supported parametric surface pairs. Planar crossings and axial
plane/cylinder cuts become real shared edges immediately. A transverse closed
ring through a complete conformal cylinder band is imprinted atomically as
exact shared arcs: the plane becomes an inner disk plus an outer annular face,
and every cylinder patch is split above and below the ring. Stable face
lineage, groups, tags, and exact plane/cylinder surfaces are preserved.
`FaceIntersection.edges` contains the complete ring while the compatible
`FaceIntersection.edge` accessor remains its deterministic first edge.

Topology-changing intersections require caller intent, for example
`intersect_faces(model, a, b, policy=MutationPolicy.IMPRINT)`. Query-only
calls use `fragment=False`; `KEEP_SEPARATE_PART` retains both inputs without
imprinting. `clip_line_to_face` returns every planar material interval and
subtracts holes instead of collapsing a concave or holed face to one span.

The qualified closed-ring topology path deliberately requires at least three
positive-sweep cylinder patches forming one complete conformal band, a cut
strictly inside the cylinder height, and a convex straight-edged plane face
without existing holes that fully contains the ring. Oblique cuts, partial or
nonconformal bands, and planes needing nested trim classification remain
non-mutating intersection-query workflows rather than being approximated with
unrelated edges.

Qualified predicates return typed `IntersectionResult` values that distinguish
crossing, touching, overlap, coincidence, disjoint, and unclassified cases.
The model-owned `TolerancePolicy` separates computational, merge, angular,
parameter, area, and surface-residual tolerances using local feature extent,
so translating a complete model does not change a local classification.

`geometry.strict_audit()` performs deterministic fail-closed full-model
qualification with a spatial broad phase. It checks duplicate/crossing/
overlapping edges, T-junctions, sheet manifoldness, structural member intent,
member-face relationships, face overlap, lineage, and unsupported candidates.
Any unclassified candidate blocks certification.

Large beam lattices should use `GeometryModel.add_members(...)`, which builds
all member chains under one part update and one structural validation. Public
`remove_member`, `remove_sheet`, `remove_part`, `remove_attachment`, and
`remove_junction` methods enforce dependency order and rollback atomically.

## Editable feature history and owner editing

`GeometryModel.features` stores an ordered, suppressible modelling history.
Feature inputs use `FeatureOutputRef` so downstream intent does not depend on
the materialized IDs allocated by a later regeneration. `EntityHandle` is the
model-bound identity for mesh and analysis packages; feature executors and
local compatibility APIs continue to use compact `EntityRef` values.

```python
from anygeometry import FeatureOutputRef, GeometryModel

geometry = GeometryModel()
first = geometry.features.append(
    "geometry.point", parameters={"position": [0.0, 0.0, 0.0]}
)
second = geometry.features.append(
    "geometry.point", parameters={"position": [2.0, 0.0, 0.0]}
)
geometry.features.append(
    "geometry.line",
    inputs={
        "start": [FeatureOutputRef(first.feature_id, "point", "vertex")],
        "end": [FeatureOutputRef(second.feature_id, "point", "vertex")],
    },
)
report = geometry.regenerate_features()
assert report.success
```

The registry is extensible by namespaced feature kind, allowing consumers to
add executors without a reverse dependency. Regeneration is atomic, retains
replacement lineage, and reserves IDs above the old materialization so a
stale `EntityRef` can never be silently reused for a different output.

Flat-face sketches use the built-in `geometry.sketch.extrude` feature. A
`SketchDefinition` stores named plane-local points, their ordered path,
distance/coincidence constraints, and the signed normal extrusion distance.
Points are not restricted to the support-face boundary. `on_edge` and
`on_vertex` constraints follow the oriented support boundary, and regeneration
returns stable `point/*`, `profile/edge/*`, and `extrusion/face/*` output keys.
The small constraint solver uses minimum-norm corrections and rejects
inconsistent dimensions without changing the live geometry.

High-level owner operations include `insert_model`, `copy_entities`, linear
and circular patterns, mirroring, edge/face reversal, deep `clone`, and typed
`measure` results. Insertion remaps all topology with fresh IDs, preserves
groups, tags, surfaces, holes, and metadata, and deliberately does not weld
coincident entities.

`AffineTransform` provides validated translation, axis-angle rotation, scale,
reflection, composition, inverse, and point-array evaluation. In-place
`translate_entities` / `rotate_entities` preserve model-bound identities;
`copy_translated` / `copy_rotated` allocate fresh identities. Arbitrary
transform lists use `pattern_entities`, while `rectangular_pattern` creates a
deterministic one-, two-, or three-axis Cartesian array. Rectangular counts are
copy steps beyond the unchanged original, so counts `(1, 1)` produce the other
three positions of a 2x2 array. Pattern implementations extract the selected
geometry and structural closure once rather than cloning unrelated model data.

## Serialization and CLI

Geometry schema 4 is deterministic and checksummed. It preserves model UUID
and revision, coordinates/CRS and tolerance policy, allocator high-water
marks, support surfaces and optional parameterizations, construction/control
ownership, curves/trims, structural ownership and qualified relationships,
groups, tags, geometry and structural lineage, extensions, and feature
history. Schemas 1–3 migrate conservatively and one-way; malformed current
documents fail closed:

```python
from anygeometry import read_geometry, write_geometry

write_geometry("panel.anygeometry.json", geometry)
restored = read_geometry("panel.anygeometry.json")
```

Certified output additionally requires a clean strict audit:

```python
write_geometry("panel.certified.anygeometry.json", geometry, certified=True)
```

`certified=True` is a validation gate, not a persisted certificate. The audit
report is an ephemeral result bound to the exact model UUID, revision, and
audit policy; schema 4 intentionally stores no reusable certification flag.
Consumers that require a qualified handoff must retain or rerun
`strict_audit()` for that exact revision.

JSON and gzip-compressed JSON are supported. Mesh and FEM/project
serialization remain outside ANYgeometry.

ANYgeometry 0.2.4 reads schemas 1–4 and writes schema 4. The Python API remains
within `ANYgeometry>=0.2,<0.3`, but a 0.2.0 reader intentionally rejects a
schema-4 document. Downstream packages should use the public codecs rather
than parse schema records. Legacy relationship evidence migrates as
`UNVERIFIED` and never implies exactness or certification.

Trusted importers and local script features that have already materialized
topology through `GeometryModel` operations can bind that exact last-good
materialization without mutating detached feature records:

```python
geometry.features.adopt_frozen(
    geometry,
    kind="vendor.script.feature",
    outputs={"face": EntityRef("face", face_id)},
)
```

`adopt_frozen` accepts only active IDs in the history's owning model; it never
retargets through replacement lineage or geometric proximity. ANYgeometry
computes the closure checksum and publishes the history update atomically.
Callers that already certified a closure may pass `expected_checksum=` to make
the adoption fail closed if the topology changed.

Feature edits mark the affected record dirty. For additive modelling features,
regeneration starts at the earliest dirty record on a clone of the live
materialization. The clean prefix keeps its exact entity IDs. A replayed output
keeps its prior binding only when its stable output-key set and complete
ID-independent topology closure are exactly equal; otherwise its replacement is
allocated above the previous ID high-water mark and lineage is extended. This
comparison uses no distance, tolerance, or nearest-geometry retargeting.

Ordinary output revalidates complete topological and structural integrity.
Certified output adds the full global geometric audit before writing, but the
ordinary and certified schema-4 payload shapes are identical. Feature history is
owner-observed and validated before writing; direct record tampering is never
accepted as a checksummed document.

The package module can create an example or inspect a saved geometry:

```powershell
python -m anygeometry --version
python -m anygeometry --write-example panel.anygeometry.json
python -m anygeometry panel.anygeometry.json --json
```

## Dependency direction

The intended dependency graph is one-way:

```text
ANYgeometry <- ANYmesher <- ANYfem
      ^             ^
      |             |
ANYstructure -------+
```

ANYgeometry never imports ANYmesher, ANYfem, ANYstructure, ANYsolver, a GUI,
or a finite-element package. See [MIGRATION.md](MIGRATION.md) when moving code
from the historical `anymesher.geometry` namespace.

## Development

```powershell
python -m pip install -e ".[dev]"
python -m pytest
python -m build
python -m twine check dist\*
```

The test suite qualifies persistent identity and history, topology, curves,
surfaces, generators, operations, serialization, intersections, CLI behavior,
and import boundaries.

The strict-kernel design, invariants, benchmark scope, and completed release
qualification are recorded in
[`docs/KERNEL_UPDATE_OVERVIEW.md`](docs/KERNEL_UPDATE_OVERVIEW.md) and
[`KERNEL_UPDATE_REPORT.md`](KERNEL_UPDATE_REPORT.md).
