# Submodel volume cut contract (C0, proposed)

Status: proposed contract for owner review, 2026-10-09. Slice S2a (planar and cylindrical shells
without members, axis-aligned or rigidly placed boxes) is implemented in
`src/anygeometry/volume_cut.py` on branch `claude/submodel-s2a` (section 9); no existing
intersection code is changed.
Plan: `C:/Github/ANYfem/docs/handoff/submodelling-20261009/PLAN.md` (revision 3, gaps G1a-G1c).

ANYgeometry owns cutting a user-given volume out of a structural model: the imprint of its
boundary, the inside/outside classification, the partition of attached members, the bounded
extraction and the geometric part of the identity map. Names below are proposals.

## 1. Volume

`BoxVolume(lower, upper, transform: AffineTransform | None = None)`: a closed box in the
model frame, optionally placed by an affine transform. Containment uses the model's
`TolerancePolicy`. Cylinders and spheres are later candidates.

## 2. Plan and apply

- `plan_volume_cut(model, volume, *, policy) -> VolumeCutPlan` is read-only and bound to
  the model id and revision. It lists the faces and members the boundary crosses, the
  expected changes, and any typed refusal before anything mutates.
- `apply_volume_cut(model, plan, *, policy) -> VolumeCutResult` runs on a detached clone and
  never on the user's document. It applies the plan in one change and returns:
  - `working_model` and `change_set`;
  - a classification of every descendant face, edge and member span as `INSIDE`, `OUTSIDE`
    or `ON` the boundary;
  - `cut_edges`, the new edges created on the volume boundary (artificial cuts), kept
    separate from `original_boundary_edges`, the pre-existing free or loaded edges inside;
  - the lineage needed for the identity map (section 5).
- Route (informative): imprint the box planes through the existing batch intersection
  workflow with `ConnectionIntent.IMPRINT`, either by inserting temporary cutting faces into
  the clone or through a native plane operand (an open decision; S0 did not need it).
  Classify each descendant by a point strictly inside it, or a member span by its midpoint.

## 3. Members across the boundary (G1b)

`split_member` refuses members with attachments or junctions (`model.py:3998-4003`). The cut
needs a qualified remap: partition the member at the boundary parameter, remap its
attachments and junctions onto the inside and outside descendants, and record structural
lineage. A member whose relations cannot be remapped is a typed refusal, never a silent
detachment.

## 4. Bounded extraction (G1c)

`extract_model_closure` expands dependencies: members bring edge uses, attachments bring
both parents, and junctions bring members, sheets and attachments (`closure.py:377-413`). It
can therefore reintroduce outside structure. The cut needs
`extract_bounded_closure(working_model, inside_handles, volume)` with these rules:

- No face, edge or member span classified `OUTSIDE` enters the result.
- A relation (attachment, junction) whose parents are not all inside is remapped onto inside
  descendants when its own geometric support lies inside or on the volume. Otherwise it is
  listed as `straddling`, and the caller must refuse.
- After extraction, every retained face sample and member span is verified inside or on the
  volume.

## 5. Identity map (geometry part)

Per entity kind (vertex, edge, face, sheet, part, member, attachment, junction): source handle,
then clone handle, then descendants through the replacement history, then submodel handle
through the closure map. Each step is explicit. Anything that cannot be resolved goes into an
`unresolved` set and is never dropped silently. ANYfem composes this with its own attribute
bindings.

## 6. Refusals (typed)

- The volume contains no structure.
- A boundary plane coincides with, or is tangent to, a model face within tolerance
  (a zero-thickness slab).
- An unsupported surface family for the plane imprint, or a missing planar-overlay capability.
- A member with attachments or junctions when the qualified remap (section 3) is unavailable.
- A straddling relation without a remap.

## 7. Related findings from S0

Cut geometry that contains holes depends on two owner routes that fail on HEAD
(`s0/repro_hole_routes.py`): `punch_circular_hole` patches carry no qualified surface support
for structural preparation, and a neutral trimmed hole fails the mesher's qualified-S3 repair.
An O-grid on an explicit `Plane` surface works. Both failures are flagged separately.

## 8. Open for owner review

- Cutting faces inserted into the clone versus a native plane-imprint operand.
- Whether the classification should be returned per descendant or as a region query.
- The containment tolerance on the box boundary (`ON` versus `INSIDE`).

## 9. Implementation (slice S2a)

- API: `BoxVolume(lower, upper, transform=None)` (a 4x4 proper rigid transform),
  `plan_volume_cut(model, volume) -> VolumeCutPlan` (read-only, bound to model id, revision and
  checksum) and `apply_volume_cut(model, plan) -> VolumeCutResult`. Typed refusals are
  `VolumeCutError.code`: `VOLUME_INVALID`, `VOLUME_EMPTY`, `VOLUME_COINCIDENT`,
  `VOLUME_SURFACE_UNSUPPORTED`, `VOLUME_CURVE_UNSUPPORTED`, `VOLUME_MEMBERS_UNSUPPORTED` and
  `VOLUME_RELATIONS_UNSUPPORTED` (members, attachments and junctions near the box are slice
  S2b), `VOLUME_STALE_PLAN`, `VOLUME_IMPRINT_FAILED`, `VOLUME_SOURCE_INTERSECTION`,
  `VOLUME_FACE_DEGENERATE`, `VOLUME_LEAK`, `VOLUME_LINEAGE`.
- **Extents are conservative**, never sampled: exact per box axis for straight edges, circular
  arcs and elliptic arcs (the sinusoid extremes on the sweep), the control hull for Bezier
  splines, and the curve's own outward enclosure for the other exact curve families
  (transformed into the box frame through its corners); other curves refuse. Only the exact
  ones may decide box-side membership and containment: a loose bound that cannot prove either
  refuses as `VOLUME_CURVE_UNSUPPORTED` rather than as an intersection or a leak. A planar or cylindrical face's extent is its boundary's:
  a planar face lies in the convex hull of its boundary, and on a cylinder a linear function of
  position is linear in the axial coordinate along each generator, so its extremes over the
  face lie on the boundary.
- **Plan**: selects the faces whose extent reaches the box; refuses surfaces other than planes,
  cylinders and planar four-sided (Coons) patches with straight coplanar boundaries (ANYfem's
  default plate); refuses a planar face with material in a box plane, found by points verified
  strictly inside its material domain (`MaterialDomain`, holes respected; an overlap too small
  to be found there is still refused by the imprint itself); refuses members that reach the
  box unless the caller declares them droppable (`droppable_members`: members that carry no
  structure, such as those with no beam section; they are left out of the extract and listed in
  `dropped_members`, and still refuse if they have attachments or junctions), attachments that
  target a selected face or one of its edges, and junctions that involve a sheet using a
  selected face.
- **Apply**: clones the model, adds the six box faces with explicit `Plane` surfaces as
  temporary cutting faces (section 8's first option), and imprints them with
  `plan_intersections`/`apply_intersections` and `ConnectionIntent.IMPRINT` (member
  connections off). The batch also pairs source faces with one another; the slice requires
  them to be prepared already, and refuses (`VOLUME_SOURCE_INTERSECTION`) if the imprint traces
  any new edge on a source face away from the box. Only descendants of the plan's faces are
  classified (every other face's conservative extent misses the box). Each one is classified by a
  point **verified strictly inside its material** (candidates from a scanline of its boundary
  in the face's chart, checked with `MaterialDomain.contains_loop`; none verified refuses).
- **Bounded extraction**: `extract_model_closure` of the inside faces with geometry only (the
  structural closure follows coedges across the shared cut edges and keeps sheets complete, so
  it would pull the outside parts of split faces back in). Parts and sheets are rebuilt in the
  extract from their inside face uses (same name, topology policy and orientations); a sheet
  declaring non-manifold edges inside the box refuses. The returned `ModelClosure` is then
  completed: every rebuilt part, sheet, face use and coedge is mapped one to one to its
  counterpart in the working clone, so the maps are bijective over every extracted entity and
  `to_transport()` round-trips. Refused if the closure maps any outside or cutting face, or if
  any extracted face's extent leaves the box.
- **Cut edges**: an extracted edge is an artificial cut when material was removed beside it:
  it is new, or its source edge had more face uses in the source than it has in the extract.
  A pre-existing free edge lying on a box side is therefore an original boundary, not a cut.
  Every cut edge must lie on a box side (grouped `x0` to `z1`; an edge on a box edge is in
  both groups).
- **Lineage**: `lineage[(kind, extracted id)] -> source ids`, through the closure to the
  working clone and from there to the source: the replacement history for faces, edges and
  vertices; ids for sheets and parts; for face uses (not in the replacement history) the source
  use of the same sheet whose face is an ancestor; for coedges the source coedge of that use on
  an ancestor edge. Extracted entities without a
  source are `created` when they are the cut's own (on the box, the via vertex of an arc the
  imprint split or traced, or a point where the imprint split a source edge: shared by exactly
  two extracted edges descending from the same source edge) and `unresolved` otherwise;
  nothing is dropped.
- Evidence (`tests/test_volume_cut.py`, 27 cases): a plate through the box (area, lineage, cut
  sides, source checksum unchanged), a rotated box, a box past one plate edge, a plate edge
  lying on a box side (not a cut), a T-junction (one shared inside junction edge used by three
  faces, not a cut), an O-grid plate with a hole inside the box and crossing it, a cylinder
  patch across the seam of two cylinder faces, a cylinder cut by a box tilted 20 degrees about a
  transverse axis (elliptic traces), a disc whose x-extreme lies half a 129-sample step off the
  samples (a sampled extent would miss the box at x >= 0.99999; the cut finds it), sheets and
  parts with lineage and a complete, transportable closure, a box plane inside a hole of a face
  lying in that plane (not coincident), a distant non-planar Coons face (not classified),
  crossing unprepared faces, relations and members near the box, and the other refusals.
  Containment is checked independently by maximizing each box coordinate along every extracted
  edge (grid plus golden-section refinement) against the unenlarged box. Audit provenance is
  checked finding by finding: each extracted finding exists right after the imprint and lifts
  through the lineage to a source finding of the same code (cut-made entities standing in for
  removed material), except the finding below.
- **Finding for the intersection owner.** When the imprint splits an arc (a hole boundary) or
  traces a plane on a cylinder, the new arcs' via vertices are reported `VERTEX_FACE_INTERIOR`
  by the strict audit. The finding is present in the working model straight after
  `apply_intersections`, before extraction; the volume cut does not change it.
