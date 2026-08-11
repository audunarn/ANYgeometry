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

Planar/UV workflows may opt into Shapely without making it a core dependency:

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
face_ref = geometry.entity_ref("face", face_id)
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

`EntityRef(kind, id)` is the stable cross-package identity. IDs are allocated
monotonically and are not silently reused. Splitting or fragmenting an entity
records its descendants, updates geometry groups and tags, and lets clients
resolve a stale selection with `geometry.resolve_ref(old_ref)`.

## Geometry model

- `Vertex`, `Edge`, and `Face` provide topology with persistent IDs.
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

The qualified closed-ring topology path deliberately requires at least three
positive-sweep cylinder patches forming one complete conformal band, a cut
strictly inside the cylinder height, and a convex straight-edged plane face
without existing holes that fully contains the ring. Oblique cuts, partial or
nonconformal bands, and planes needing nested trim classification remain
non-mutating intersection-query workflows rather than being approximated with
unrelated edges.

## Editable feature history and owner editing

`GeometryModel.features` stores an ordered, suppressible modelling history.
Feature inputs use `FeatureOutputRef` so downstream intent does not depend on
the materialized IDs allocated by a later regeneration.  `EntityRef` remains
the topology identity passed to mesh and analysis packages.

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

High-level owner operations include `insert_model`, `copy_entities`, linear
and circular patterns, mirroring, edge/face reversal, deep `clone`, and typed
`measure` results. Insertion remaps all topology with fresh IDs, preserves
groups, tags, surfaces, holes, and metadata, and deliberately does not weld
coincident entities.

## Serialization and CLI

Geometry serialization is versioned and preserves IDs, ID counters, curves,
surfaces, holes, metadata, semantic groups, tags, and replacement history:

```python
from anygeometry import read_geometry, write_geometry

write_geometry("panel.anygeometry.json", geometry)
restored = read_geometry("panel.anygeometry.json")
```

JSON and gzip-compressed JSON are supported. Mesh and FEM/project
serialization remain outside ANYgeometry.

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
