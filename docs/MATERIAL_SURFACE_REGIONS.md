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
