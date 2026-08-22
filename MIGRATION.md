# Migrating geometry ownership to ANYgeometry

ANYgeometry replaces the historical geometry implementation under
`anymesher.geometry`. The migration preserves the same owner classes and the
value-based `EntityRef(kind, id)` local compatibility type; persistent
cross-package state should move to model-bound `EntityHandle` identity.
Compatibility imports may remain temporarily, but new code should import the
owner package directly.

## Updating from ANYgeometry 0.1 to 0.2

`EntityRef(kind, id)` remains available for local compatibility, but new
cross-package state should use `EntityHandle(model_id, kind, id)`. Handles can
distinguish identical local IDs from different documents and explicitly report
active, replaced, deleted, or unknown resolution.

Public entity, structural, group, and tag stores are now read-only. Replace
direct dictionary or record mutation with `GeometryModel` owner methods and
use `transaction()` to batch related edits. Model identity, revision,
tolerance, units, local origin, and coordinate transforms are owner-controlled;
change document settings through `set_document_settings(...)`.

Beam axes should be persisted as `Member` chains rather than inferred from an
edge group. Plates can be owned by `Part`/`Sheet`/`FaceUse` topology, while
`Attachment` and `Junction` records declare intended beam/plate and beam/beam
relationships. Groups remain useful presentation semantics, not physical
identity. Large imports should use `add_members(...)`.

Topology-changing intersections now require an explicit `MutationPolicy`.
Use query mode when no mutation is intended and `IMPRINT`, `WELD`,
`REUSE_EXISTING`, `KEEP_SEPARATE_PART`, or `REJECT` only after choosing the
desired ownership behavior. A clean `strict_audit()` is required for certified
output.

### 0.2.1 document boundary

ANYgeometry 0.2.1 reads geometry schemas 1, 2, 3, and 4 and writes only
canonical schema 4. Loading schemas 1–3 is a deterministic, one-way migration;
the next write is schema 4. Schema 4 adds CRS metadata, separate support and
parameterization fields, expanded tolerances, construction ownership, Member
orientation references, qualified Attachment/Junction evidence and context,
and structural replacement lineage.

| Reader | Reads | Writes |
| --- | --- | --- |
| ANYgeometry 0.2.0 | schemas 1–3 | schema 3 |
| ANYgeometry 0.2.1 | schemas 1–4 | schema 4 |
| ANYgeometry 0.2.2 | schemas 1–4 | schema 4 |

The Python package dependency remains compatible with `ANYgeometry>=0.2,<0.3`,
but the document format is forward-incompatible: a 0.2.0 reader intentionally
rejects a schema-4 document. Exchange persisted geometry with 0.2.1 or newer,
and use only `to_dict`, `from_dict`, `write_geometry`, and `read_geometry`
rather than parsing core records in downstream packages.

Version 0.2.2 does not change the document schema. Importers that previously
mutated detached `FeatureRecord` objects should instead call the atomic
`FeatureHistory.adopt_frozen(...)` owner API.

Legacy schema-3 Attachments did not carry qualification evidence. Migration
therefore sets their evidence to `UNVERIFIED`, with zero residual/tolerance
placeholders. Those zeros do not mean exact geometry and block certified
handoff until the relation is requalified through the intersection
query/plan/apply workflow. Code that edits a serialized dictionary must
recompute the complete document checksum through ANYgeometry.

Certification is not a document field. Passing `certified=True` to a public
writer runs a strict full-model audit as a write-time gate, but produces the
same schema-4 shape as an ordinary write. An `AuditReport` applies only to its
exact model UUID, revision, and policy and must be retained or recomputed by a
consumer that needs qualified handoff evidence.

## Import mapping

| Historical import | Owner import |
| --- | --- |
| `anymesher.geometry.GeometryModel` | `anygeometry.GeometryModel` |
| `anymesher.geometry.EntityRef` | `anygeometry.EntityRef` |
| no model-bound equivalent | `anygeometry.EntityHandle` |
| `anymesher.geometry.entities` | `anygeometry.entities` |
| `anymesher.geometry.curves` | `anygeometry.curves` |
| general `anymesher.geometry.operations` | `anygeometry.operations` |
| geometry chain sampling | `anygeometry.chains` |

Mapped-quad policies do not move. `check_mappable`, triangle-to-quad and
butterfly-hole decomposition, mapped partitioning, seeding, mesh generation,
quality checks, and geometry-to-mesh association remain in ANYmesher.

## Shared object contract

Consumers must pass the same `GeometryModel` and model-bound `EntityHandle`
values across package boundaries. `EntityRef` remains valid inside one known
model for compatibility and feature execution, but it cannot distinguish two
documents with the same local ID. Do not copy geometry into an ANYfem-,
ANYstructure-, or mesher-specific representation.

Keep domain data external and keyed by geometry references:

```python
materials_by_face = {geometry.handle("face", face_id): material}
loads_by_face = {geometry.handle("face", face_id): pressure}
mesh_controls_by_edge = {geometry.handle("edge", edge_id): seed_count}
```

When an edit splits or fragments geometry, use replacement history to remap
those attachments:

```python
geometry.begin_replacement_log()
geometry.split_edge(edge_ref.id, 0.5)
new_edge_refs = geometry.resolve_ref(edge_ref)
changes = geometry.replacement_log()

# Cross-package consumers should use the typed model-bound result instead.
resolution = geometry.resolve_handle(edge_handle)
new_edge_handles = resolution.require()
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
