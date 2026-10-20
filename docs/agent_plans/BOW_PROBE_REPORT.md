# Bow probe report (programme step 3, 2026-10-20)

Owner decision Q2: the bow is two patches meeting at the stem (architecture section 6 form),
chosen for symmetry investigations. Dimensions: the `floating_barge_large` midship (breadth
11 m, deck z = 2.5) with the bow from X = 15 to the stem near X = 20. Variant G0 (a knuckle at
X = 15) first; variant G1 (tangent to the flat bottom and side) measured.

Code: `src/anygeometry/bow_example.py` (`bow_two_patch(join)`, `build_bow_faces_below_deck`);
every finding below is pinned by `tests/test_bow_example.py`, including the refusals, so the
tests change when a design response lands.

## Geometry (P3a): met

Each side is one bidegree-(3, 3) polynomial patch: u from the stem (u = 0) to X = 15 (u = 1),
v from the keel over a hard chine (a C0 crease from a triple knot at v = 1/2) to z = 3 above the
deck. The barge's midship has a sharp bilge, which one smooth cubic cannot match; the chine
makes the aft boundary exactly the midship section (flat bottom z = 0 from the keel to the chine
at y = 5.5, flat side y = 5.5). The stem and the keel both lie in the centreline plane, so the
two patches share two boundary curves. The port patch is the exact mirror: its stem and keel
iso-curves are the same exact NURBS curves as the starboard ones, and its evaluation equals the
mirrored starboard evaluation bitwise (10^4 samples). The normal never vanishes (min |n| 0.56),
the orientation is consistent, and the two halves meet in a sharp V of about 24 degrees at the
stem.

## What works

| Operation | Result | Cost |
|---|---|---|
| Plane x NURBS sections of transversal planes: deck z = 2.5, waterline z = 1.2, frame x = 17, longitudinal y = 2 (across the chine), both variants | certified | milliseconds |
| The deck section of both patches | ends at one exact stem point (19.799..., 0, 2.5) on both | - |
| One bow face (keel, aft and stem iso-curves + the deck section) | validates; `strict_audit` clean | 1.2 s |
| The deck face bounded by the two patches' deck sections (meeting at the stem vertex) and the line at X = 15 | validates | 2.3 s (half deck 1.3 s) |
| The starboard half (bow face, half deck, half bulkhead, centreline face) | `closed_shell` clean (topologically closed) | - |
| The full bow with consistently oriented loops | `closed_shell` fails only on the duplicate stem and keel edges | - |

## Refusals, by class

**Class A: stem adjacency (two patches sharing a boundary curve).**
- `CurveOnSurface(..., also_on=(port,))` refuses at construction: "also_on requires an exact
  shared-boundary proof (not available yet)".
- A port face using the starboard stem edge refuses: the edge is not an exact chart curve on
  the face's support.
- With separate, coincident stem and keel edges per patch, each face validates, but
  `strict_audit` reports only unclassified cross-support pairs: the duplicate edges, each
  face's edges against the other face, and the two faces against each other (11 blockers).
  `closed_shell` sees the duplicates as edges used once.

**Class B: a patch boundary lying in a plane.**
- A plane face bounded by NURBS iso-curves that lie exactly in its plane (the collision
  bulkhead at X = 15, the centreline face y = 0) refuses: "only straight and plane x NURBS
  section edges may trim a plane face with section edges". `face_moments` refuses the same
  edges, so hydrostatics on the closed half model refuses.
- Sectioning a plane that contains a patch boundary refuses: bulkhead x = 15, flat side
  y = 5.5 and flat bottom z = 0 with "the section runs along a box edge"; the centreline y = 0
  with a singular box at the stem/keel corner.

**Class C: tangency (variant G1).** The G1 bow leaves the flat side and bottom tangentially at
X = 15 (S_u along -x exactly). Sectioning the side or bottom plane refuses with a singular box
at the aft corners. Classifying a midship plane face against the bow face along their shared
aft edge could not be probed: those plane faces are bounded by in-plane iso-curves (class B).

No unexpected refusal was found; probe-script errors (a duplicated vertex, face orientation)
were corrected and are not findings.

## Proposed design responses (input to step 4)

1. **Exact shared boundary (`also_on`) for identical iso-curves** (class A, size S-M). Two
   supports share a boundary edge when their boundary iso-curves are the same exact NURBS
   curve: equal degree, knots, controls and weights, compared exactly. For the mirrored bow
   that is a complete proof. One edge then serves both faces (each face reads its own chart
   iso-line), and `closed_shell` closes.
2. **Adjacent NURBS faces in the audit** (class A, size M-L): two NURBS faces that share only
   that exact edge are certified to touch only along it. That needs a joint band across the
   shared edge (the P4e argument for adjacent patches: a lower bound of the angle between the
   two patches along the stem, about 24 degrees here, from exact derivative bounds) and exact
   separation of the sub-patches away from the edge. This is a narrow part of A5, not general
   NURBS x NURBS.
3. **Plane faces trimmed by in-plane iso-curves** (class B, size M). Certificate: every 3D
   control point of the iso-curve lies exactly on the plane (here x = 15 and y = 0 exactly);
   with positive weights the curve then lies in the plane. Trim validation needs line x
   in-plane-curve predicates (the P4a engine on a planar curve) and `face_moments` the Green
   line integral along such edges. This unblocks the bulkhead, the centreline face, the half
   model's hydrostatics and the symmetry checks of P3d.
4. **Plane x NURBS pairs along an in-plane boundary** (classes B and C, size M for G0, L for
   G1): for the audit and for `plan/apply_intersections`, a plane face and a NURBS face that
   meet along a boundary lying in the plane are classified without P1 sectioning: transversal
   (G0) by the joint-band angle along the edge; tangent (G1) needs a second-order separation
   argument (the patch departs quadratically from the plane), which the P4e band cannot
   provide.

## P3c and P3d status

P3c (closed full bow) and P3d (symmetry checks) are blocked, not failed: the full bow needs
response 1 to close and responses 1-2 to audit; the half model closes topologically and needs
response 3 for validation and hydrostatics. P3d's checks (B on the centreline, GZ(-phi) =
-GZ(phi), half model against the full model) become possible after responses 1 and 3.

## Recommended order

Response 3 first (smallest, unblocks the half model, its hydrostatics and the bulkhead), then
response 1 (closes the full bow), then response 2 (audit across the stem), then response 4
(G0 first). Responses 1 and 3 together make P3c-P3d's symmetry checks possible.
