# Solid volume handoff v1

Status: prospective Workflow 2 contract, 2026-10-27. Canonical owner: ANYgeometry.

Required envelope: a single closed convex polygonal normal prism, planar faces,
straight edges, SI metres. Boxes and thin signed-offset prisms are mandatory.
Holes, concavity, curved faces, multiple components and interior interfaces are
unsupported. A validation report is not a certificate for those classes.

## Geometry owner

`GeometryModel.add_solid_prism(profile, vector, *, offset=0, name=None)` creates
a capped region atomically. `profile` is an ordered finite (n,3) planar convex
polygon without a repeated closing point; `vector` is a nonzero normal extrusion.
`offset` translates the profile along its oriented unit normal before extrusion.
Reversed profile and signed vector remain explicit. Provide a box convenience API.

`GeometryModel.solid_regions` is a read-only collection of immutable
`ClosedSolidRegion` records with owner-bound persistent identity and oriented
existing face references. Region IDs have their own namespace; they do not
reinterpret structural Parts/Sheets. Existing document schemas remain readable;
new region records are versioned and participate in persisted checksums.

`GeometryModel.solid_region_snapshot(region_id)` returns an immutable snapshot:

- `schema_version=1`, `model_id` (UUID string), `revision` (integer), `region_id`
  (persistent string), `units='m'`, `geometry_class='convex_normal_prism'`;
- `coordinates`: owned immutable float64 (n,3) rows;
- `face_loops`: ordered tuples of coordinate-row indices, outward orientation;
- `face_refs`: one owner `EntityHandle` per face loop, same order;
- `face_groups`: immutable mapping of names to tuples of face handles;
- `volume`: positive analytic volume; `bounds`: lower/upper coordinates;
- `source_signature`: content SHA-256 including coordinates, topology, ownership,
  selections and provenance; `provenance`: immutable JSON-compatible metadata.

Default groups `start`, `end`, `side_0`, ... identify whole faces. Additional named
groups may overlap. Geometry owns an explicit group-selection API. A source edit
cannot mutate a published snapshot. Reuse against an owner requires matching
identity/revision/signature. Rebinding topology changes must be explicit.
Regeneration with unchanged topology preserves region identity and face roles;
incompatible topology reports conflict without moving selections to other faces.

## Mesher owner

`VolumeMesh` is a sibling of shell `Mesh`, never a shell codec reinterpretation.
It exposes the existing ANYsolid `SolidMesh` structural attributes: coordinates,
cells, node_ids, cell_ids, boundary_facets, boundary_ids, node_groups, cell_groups,
boundary_groups, units, geometry_order, provenance. Arrays are immutable and owned.
Connectivity indexes rows; group membership uses external IDs. All coordinates
belong to cells; unique exterior facets are complete and outward oriented.

Additional face ownership maps each boundary external facet ID to exactly one
source face handle. Source references, output facet IDs and group labels remain
distinct. Named groups may overlap without creating duplicate owner assignments.
Initial meshes contain one connected region, geometry_order=1, units='m'.

`generate_volume_mesh(snapshot, *, target_edge_length, boundary_tolerance,
timeout=120, cancel=None, backend='pytetwild')` has no implicit physical sizing or
boundary tolerance. It validates a snapshot before dispatch, invokes an owned
subprocess, and validates the complete output before publication. Typed errors
distinguish unavailable backend, unsupported geometry, invalid output, timeout
and cancellation. Ordinary imports do not load the backend or consumers.

Pinned initial provider: pytetwild 0.4.2, raw arrays, simplify=False,
coarsen=False, optimize=True, num_threads=1. epsilon=boundary_tolerance/diagonal;
record the exact effective options, version and input hash. No automatic repair,
tolerance relaxation, retry-until-pass or alternate provider.

Postchecks independently cover cell orientation/degeneracy/overlap, boundary
incidence, closure, connectivity, outward normals, complete face ownership,
projected face coverage, source bounds/volume and protected edges/corners.
Nearest-face selection alone is insufficient. Approximation and sampled checks
are identified honestly. Canonical row/ID generation gives recoverable mappings,
not cross-platform byte-identical remeshing guarantees.

Consumer check: unchanged `anysolid.SolidMesh.snapshot(volume_mesh)`. ANYsolid
is a test-only installed consumer. No solver changes or coupling promotion.

## Execution and acceptance

Development is recoverable Level A. Qualification uses prospective bounded case
manifests, exact source/environment identities, raw logs and non-author review.
Failed evidence stays immutable. Existing Workflow 1 failures and budgets are
separate and unchanged. No merging or package publication is authorized here.
