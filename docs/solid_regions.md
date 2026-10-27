# Native closed solid regions

`GeometryModel.add_solid_prism(profile, vector, offset=0, name=None)` creates a
single capped, convex, planar polygonal normal prism and returns a persistent
region UUID string. `add_solid_box(lower, upper, name=None)` is the axis-aligned
convenience operation. These operations create ordinary existing vertices,
straight edges and analytically supported planar faces. They do not reinterpret
Parts, Sheets, shell meshes or material regions.

```python
from anygeometry import GeometryModel

model = GeometryModel()
region_id = model.add_solid_prism(
    [(0, 0, 0), (2, 0, 0), (2, 3, 0), (0, 3, 0)],
    (0, 0, -0.002), offset=0.001, name="thin plate",
)
snapshot = model.solid_region_snapshot(region_id)
model.set_solid_face_group(region_id, "loaded", snapshot.face_groups["end"])
```

Coordinates and offsets use metres. `profile` is ordered without a repeated
closing point; winding defines its oriented unit normal. `offset` translates
along that normal before applying the signed normal `vector`. Either winding
and either extrusion sign are supported. Thin solids use the same exact prism
representation; there is no implicit thickness, midsurface interpretation or
physical sizing rule. Original signed construction parameters are retained in
immutable provenance. A tiny dimension must also satisfy the model's existing
edge and plane representability/validation limits.

The profile must be strictly convex, simple, planar and finite, with at least
three distinct vertices and no redundant collinear corners. Zero, oblique,
nonfinite or unrepresentable extrusions are rejected. Holes, concavity, curves,
multiple components, touching/interior interfaces and general boundary
representations are outside this contract. Adjacent regions cannot share one
boundary face record. No automatic repair or tolerance relaxation occurs.

`model.solid_regions` is a read-only mapping of deeply immutable
`ClosedSolidRegion` records. A region has its own UUID namespace and a model UUID;
its ordered `face_refs` are owner-bound `EntityHandle` values and its parallel
`face_orientations` explicitly orient the existing faces. The default singleton
groups `start`, `end`, and `side_0`, ... are permanent face roles. Additional
named groups select complete owner-bound faces and can overlap; foreign handles,
empty selections, duplicate faces and replacement of the default roles fail.

`solid_region_snapshot(region_id)` requires committed state and returns the
version-1 `SolidRegionSnapshot` defined in `solid_volume_contract.md`: detached
immutable float64 coordinates, outward face loops, references, groups, positive
analytic volume, bounds, units, source model/revision/identity, provenance and a
content SHA-256. Coordinates have immutable byte backing. The signature includes
coordinates, source topology/edge identity, analytic supports, face ownership,
selections and provenance; revision is separately bound. Published snapshots
cannot be changed by subsequent source edits. `validate_solid_region_snapshot`
rejects foreign, stale or content-altered snapshots.

`regenerate_solid_prism(region_id, profile, vector, offset=0)` updates the complete
prism atomically, preserving its region, face, edge and vertex identities and all
face-group selections. Vertex order is the explicit role correspondence. A change
in vertex count or oriented topology reports conflict; it never guesses a
replacement selection. Existing source geometry edits are validated at commit;
a change that invalidates a retained prism rolls back. Generic transforms must
also maintain its planar normal-prism contract.

The builtin persistent `geometry.solid_prism` feature accepts `profile`, `vector`
and optional `offset` parameters. Standard `model.features.append(...)` and
`model.regenerate_features()` implement creation and same-topology regeneration.
Feature outputs expose `face/start`, `face/end`, `face/side_i`, `vertex/i` and
`edge/i` roles. Region identity survives incremental and full-history replay;
full replay stages the existing region topology before applying new parameters.
Suppression retires the region with its topology. Failed replay publishes no
partial geometry or moved selections.

Regions participate in the existing transaction journal, committed document
revision, topology/design snapshots, clone and strict audit. Clones retain region
UUIDs while rebinding every face/group handle to the clone owner (unless cloning
with `preserve_identity=True`). Region-bearing documents use top-level schema 8 and their `solid_regions`
field contains explicitly versioned records covered by the document checksum.
Documents without regions keep their previous schema versions and shape; old
schema documents remain readable. A region field under schema 5, 6 or 7 is
rejected rather than silently changing those older contracts. Strict decoding rejects malformed records,
foreign ownership, incompatible topology and invalid geometry even when the
outer document checksum is valid.

Ordinary imports use NumPy and the existing lightweight geometry owners; no
mesher, solver or provider dependency is added. Development evidence is recorded
in `geometry_implementation_checkpoint.md`. These focused checks establish the
implemented development contracts, not volume-mesh qualification or release
acceptance.
