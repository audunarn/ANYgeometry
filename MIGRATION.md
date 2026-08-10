# Migrating geometry ownership to ANYgeometry

ANYgeometry replaces the historical geometry implementation under
`anymesher.geometry`. The migration preserves the same owner classes and
value-based `EntityRef(kind, id)` identity; compatibility imports may remain
temporarily, but new code should import the owner package directly.

## Import mapping

| Historical import | Owner import |
| --- | --- |
| `anymesher.geometry.GeometryModel` | `anygeometry.GeometryModel` |
| `anymesher.geometry.EntityRef` | `anygeometry.EntityRef` |
| `anymesher.geometry.entities` | `anygeometry.entities` |
| `anymesher.geometry.curves` | `anygeometry.curves` |
| general `anymesher.geometry.operations` | `anygeometry.operations` |
| geometry chain sampling | `anygeometry.chains` |

Mapped-quad policies do not move. `check_mappable`, triangle-to-quad and
butterfly-hole decomposition, mapped partitioning, seeding, mesh generation,
quality checks, and geometry-to-mesh association remain in ANYmesher.

## Shared object contract

Consumers must pass the same `GeometryModel` and `EntityRef` values across
package boundaries. Do not copy geometry into an ANYfem-, ANYstructure-, or
mesher-specific representation.

Keep domain data external and keyed by geometry references:

```python
materials_by_face = {face_ref: material}
loads_by_face = {face_ref: pressure}
mesh_controls_by_edge = {edge_ref: seed_count}
```

When an edit splits or fragments geometry, use replacement history to remap
those attachments:

```python
geometry.begin_replacement_log()
geometry.split_edge(edge_ref.id, 0.5)
new_edge_refs = geometry.resolve_ref(edge_ref)
changes = geometry.replacement_log()
```

IDs are persistent identities, not list positions or reusable `pointN` and
`lineN` display names. Legacy project aliases should be resolved during load
and retained only for backward-compatible presentation or serialization.

## Generators and semantic groups

Automatic structural geometry now comes from `anygeometry.generators` and
returns a neutral model. Use groups such as `shell`, `plate`, `bulkhead`,
`boundaries`, `longitudinal_stiffeners`, `transverse_stiffeners`, `bottom`,
`top`, and `ring_stiffeners` to attach package-owned data.

ANYstructure keeps material, thickness, section, design, fatigue, load, and
calculation metadata. ANYfem keeps loads, supports, masses, imperfections,
sections, FE attributes, and results. ANYmesher keeps seeding, mapped-mesh
decomposition, refinement, quality, mesh associations, and optional Gmsh
integration.

## Serialization

Use `anygeometry.to_dict`, `from_dict`, `write_geometry`, and `read_geometry`
for the geometry payload. Project, FE, result, and mesh file formats should
embed or reference this payload but remain owned by their respective packages.
Legacy readers should translate old point/line/face documents once, then rely
on ANYgeometry for validation and future round trips.

## Dependency boundary

The allowed direction is:

```text
ANYgeometry -> NumPy
ANYmesher   -> ANYgeometry
ANYfem      -> ANYgeometry and optionally ANYmesher
ANYstructure-> ANYgeometry and optionally ANYmesher/ANYfem
```

ANYgeometry must never import ANYmesher, ANYfem, ANYstructure, ANYsolver, GUI,
FE, or mesh modules. Compatibility modules belong in the consuming package so
the dependency direction cannot reverse.
