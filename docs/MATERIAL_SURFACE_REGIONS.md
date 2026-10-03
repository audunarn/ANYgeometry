# Exact material regions for discretization

`query_material_surface_regions(model, operands=None)` returns a read-only,
source-bound collection of exact material regions. Existing face charts,
document topology and schema selection (5 or 6) are unchanged.

This addresses an artificial meshing corner: a fold-ending physical curve can
be tangent to a construction seam. Meshing each fragment separately imposes a
zero-angle corner even though the original material has no corner there.
Cancelling a qualified construction seam removes that artificial boundary;
it does not remove the physical intersection curve or reduce quality gates.

The initial cancellation route supports straight, paired, oppositely traversed
decomposition seams between extruded charts with identical directrix/vector
definitions. Common support ranges cover the original ranges. Structural
occurrences, occurrence metadata, face metadata, tags and groups must agree.
Other support families retain their existing individual charts. Ownership
transitions, ambiguous incidence and unsupported boundary cycles fail closed.

Edges carrying member axes, orientation references, attachments, groups,
additional tags, meaningful Coedge records or declared physical joints cannot
be cancelled. A later physical cut through an old construction seam removes
the seam marker during atomic intersection application. Coincident fragments
in one Sheet suppress only proven shared artificial affine intervals;
independently authored coincident boundary edges still receive physical joints.

Each `MaterialSurfaceRegion` provides:

- `sources`: every original bound face chart, including its parameter frame
  and FaceUses. These remain the authoritative source mappings.
- `domain` and `boundaries`: exact outer/hole loops in the common support.
- `cancelled_seams`: persistent edge handles removed from region boundaries.
- `interior_constraints`: exact physical or protected edges inside the region.
- `retained_vertices` and `source_attachments`: persistent constraints and
  attachment definitions that consumers must preserve.
- `material_area`: qualified physical area, checked against all source charts.

`validate_material_surface_regions_binding` rejects another model, changed
source, stale revision and altered region evidence. The public evaluator
`evaluate_material_surface_region` accepts finite `(..., 2)` parameters in the
**region's common frame**, keyed by any source face handle. It returns support
positions, or exact derivatives with `derivatives=True`. Optional material
membership rejects void points. Source-face parameters must be mapped through
the retained source supports; they are not interchangeable with region UVs.

Consumers must retain shared node sequences on physical constraints, pin
retained stations, preserve authored associations and honor source attachments.
The current mesher integration admits unions only when every region source face
maps to the same face in the mesher's authored input. An already fragmented input
does not gain this coverage from historical ancestry alone. Assigning a
cross-source cell to a representative face is not valid when independently
authored preimages differ. Such cases need
an explicit provenance representation before acceptance.

`query_prepared_face_preimages(model, face_ids=None, expected_revision=None)`
provides that ancestry proof for a qualified preparation started from a known,
unfragmented authored snapshot. `PreparedFacePreimages` binds authored model ID,
revision, checksum and face IDs to the current model identity, checksum, complete
face coverage and immutable `source_to_current_faces` mapping. Authored IDs may
be retired; they are not live face handles. Validate external bindings with
`validate_prepared_face_preimages_binding` before use.

Only explicit owner application replacement deltas compose the mapping.
Idempotent application, `clone_prepared_geometry` and qualified corner edits
preserve its original anchor. Arbitrary edits, ordinary clones, loading a
fragmented document and unknown prior fragmentation cannot renew ancestry.
An application inside a caller's active transaction does not establish this
receipt; later reclassification cannot recover a lost authored anchor. The
query fails closed in these cases and during transaction/change hooks. The
classification receipt retains its existing meaning independently of ancestry.
No document schema or serialized lineage changes are introduced.

A consumer of an already prepared model must agree on the returned reference
namespace separately: ancestry used to admit one authored region does not by
itself authorize changing input-face IDs in mesh associations. Physical edge
stations and retained vertices must occur in active connectivity. The initial
mesher route refuses region source attachments until their consumer semantics
are qualified; quadratic region meshing is still outside this initial route.

The proposed ANYfem integration preserves its prepared neighborhood and declares
an explicit authored-face association basis bound to `PreparedFacePreimages`.
ANYfem would compose those roots through its pre-preparation source mapping once;
its existing descendant remapper cannot consume retired root IDs. This proposal
is not implemented or accepted. Child-local sections, loads and mesh controls
must retain their scope, or the consumer must decline the union.

Development evidence is recorded in
`reports/general_intersections/quadric_generalization_study.md`. Geometry region
qualification alone is not accepted meshing. The large mixed-model linear and
representative quadratic acceptance gates remain outstanding.

## Boundary refinement development

ANYmesher continuation commits `4a24f70a339be04d2399ea39c12f19460aa3ef35`
and `0b2351e031fcbfdebc5a8c9feb50fc941c54d09a` retain owner-bound boundary
station identity and stage detached Plane patches. Identity receipts do not
permit subdivision. The prototype consumes common owner UV coordinates,
preserves internal constraints and source nodes, and charges original attempt
allowances on failure. It neither allocates a shared node nor publishes a mesh.

The rational fixture removes shape violations but retains growth above the
existing limit. It demonstrates strict progress, not accepted meshing. Curved
extrusion coverage, source-error qualification, per-face cumulative receipts and
one atomic transaction across every incident representative remain prerequisites
for production routing. Historical collinear splitting and current accepted
meshes retain their contracts. No mixed-model acceptance or new release follows
from these prototypes.

### Whole-cell coverage query

`validate_material_surface_region_triangles(model, regions, face, triangles_uv)`
is a read-only coverage query for finite arrays shaped `(n, 3, 2)`. It returns
`None` only when every closed triangle lies in the selected, revision-bound
material region. It preserves input arrays and model state and accepts the
existing `cancellation_check` callback. Unsupported or unresolved proofs raise
`GeometryError`.

The initial implementation supports Plane charts and Bezier-directrix extrusion
charts with an exactly planar profile and a certified monotone projection.
Every boundary must have a coefficient-proved polynomial chart correspondence,
and each loop must close exactly. Side events, boundary membership and winding
use rational polynomial arithmetic and certified root intervals. Concave trims
and wholly enclosed holes are checked; isolated display samples are not a
coverage certificate.

Algebraic branch trims, periodic supports, tolerance-only support coincidence
and rounded fragment controls without an exact coefficient identity currently
refuse. In particular, this query does not yet qualify the failing mixed-model
Bezier wall. Historical owner membership queries retain their tolerances and
behavior. Coverage success grants no subdivision permission, curve-error,
shared-node, element-quality or mesh-publication qualification.

## Prepared polynomial edge ancestry

`query_prepared_edge_subcurve_preimages(model)` returns an immutable,
owner-bound `PreparedEdgeSubcurvePreimages` receipt after complete batch
preparation. Records cover tracked Straight and Spline edges. New joints,
unsupported curves and untracked replacements are explicitly listed in
`unavailable_edge_ids`; requesting one explicitly refuses. Validate a saved
receipt with `validate_prepared_edge_subcurve_preimages_binding` before use.

Each record binds an original polynomial edge definition, an exact rational
interval composed from the actual binary64 split parameters, and the current
edge definition. Its Bernstein error controls enclose the current approximation
relative to the original restriction over the entire interval. The squared
distance bound must fit the existing split tolerance. This does not establish
an exact algebraic intersection parameter or material containment.

Canonical vertex replacement preserves ancestry only when polynomial controls
remain exactly equal. A changed approximation loses this optional proof;
unexpected edits to a tracked edge reject detached preparation. Final callback
mutation rejects before topology adoption. Idempotent publication may normalize
only the revision; all other serialized fields must equal the sealed candidate.

Ordinary clone, save/load and edits do not transfer these transient receipts.
`clone_prepared_geometry` and the qualified corner-only update preserve current
proof. A later authorized preparation can establish new roots in an
unfragmented loaded document, but cannot recover historical split parameters
from replacement IDs or coordinates. No schema change is required.

These receipts grant no analytic boundary substitution, subdivision permission,
whole-cell coverage, element-quality or mesh-publication acceptance. In
particular, the rounded mixed-model wall remains outside the exact coverage
query's supported domain until its boundary-domain correspondence is qualified.

## Explicit original-domain development contract

`query_prepared_authored_face_definition` returns the prospectively captured
original face, all oriented outer/hole loops, support, boundary dependencies,
and scoped occurrence/group/tag data. Incoming attachments/member references
and unrelated neighbouring geometry are not fully snapshotted. Absence from
this record does not establish absence from the original or current model.
Legacy ID-only receipts refuse this lookup; schema and release version stay
unchanged.

For complete persisted owner-input visibility, use `query_prepared_model_scope`
and `validate_prepared_model_scope_binding`. This separate transient receipt
returns the **whole original model document**, captured before the first owner
application, and the **whole current model document**, bound to complete current
preparation. Both include all stored entities, occurrences, metadata, members,
attachments, neighbours, isolated vertices, groups, tags, features, extensions
and replacement history. `authored_document` and `current_document` decode fresh
detached dictionaries; public `from_dict` can read either. Original and current
parameters retain their respective meanings. Revalidate the receipt before and
after consumer work. Qualified prepared copying preserves the proof; ordinary
cloning, reloading, incomplete preparation and legacy per-face-only receipts
refuse. Same-revision edits and callback edits invalidate the binding.

Absence from a complete document proves absence only from that persisted owner
document, never from an external FEM project. Visibility is **not** an original
to current parameter remap, equivalent child-property proof, embedded-constraint
certificate, whole material/partition certificate or mesh-publication permit.
Consumers must separately qualify or refuse unsupported child-local properties,
incoming references, neighbour constraints and isolated-point mapping. No schema
or package-version change is required.

`query_prepared_authored_boundary_correspondence` proves that the exterior of
all authenticated descendants tiles each original polynomial root edge exactly
once in its original direction. Rational intervals, original controls/anchor
and exact endpoint closure are checked. Every original hole remains explicit.
Paired internal edges are returned without removing physical joints. This
boundary-chain proof does not establish an embedded material partition.

`validate_prepared_authored_face_triangles` certifies whole closed triangles in
the **original authored domain**, including all its holes. It does not certify
the literal current fragment union. The existing region coverage API retains
its separate semantics. Plane and supported Bezier extrusion charts use the
same exact polynomial/winding engine. A four-edge implicit Coons wall can use
the extrusion chart only when every original top coefficient is exactly the
bottom coefficient plus one exactly representable translation, both connectors
close exactly, the original corner layout matches, and profile-plane and
injectivity proofs succeed. Sampled recognition, alternate parameterizations,
incompatible charts and unsupported root families refuse.

`query_prepared_authored_boundary_stations` returns immutable rational pairs
for source parameters, original UV/XYZ, and the current polynomial XYZ.
`validate_prepared_authored_boundary_station_coordinates` checks supplied
document-unit coordinates against **both** original and current points within
the existing current edge tolerance. It never snaps or moves nodes. Consumers
must bind their own global node IDs, registry entries, retained constraints,
associations and unchanged coordinates; these owner assertions supply no node
identity, curve-chord-error or mesh-publication permission.

An explicit authored-root meshing route may consume these contracts while
preserving every current physical constraint/reference and associating cells
to the authored root. It must refuse unsupported child-local semantics or
unqualified adjacent occurrences. Shared station transactions, metric-chart
routing, curve approximation, final quality gates and installed-artifact
qualification remain consumer obligations. See the portable builder in
`examples/authored_boundary_handoff.py`; it rebuilds transient evidence and
does not produce an accepted mesh.
