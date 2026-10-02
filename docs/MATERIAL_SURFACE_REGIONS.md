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
The current mesher integration is restricted to regions whose source faces all
map to the same authored face. Assigning a cross-source cell to a representative
face is not valid when independently authored preimages differ. Such cases need
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

Development evidence is recorded in
`reports/general_intersections/quadric_generalization_study.md`. Geometry region
qualification alone is not accepted meshing. The large mixed-model linear and
representative quadratic acceptance gates remain outstanding.
