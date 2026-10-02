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

Development evidence is recorded in
`reports/general_intersections/quadric_generalization_study.md`. Geometry region
qualification alone is not accepted meshing. The large mixed-model linear and
representative quadratic acceptance gates remain outstanding.
