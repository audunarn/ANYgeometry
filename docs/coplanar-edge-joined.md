# Near-coplanar plane faces joined by a shared edge

Owner decisions 12 (2026-11-19, HbeamBoth blocker 2) and 20-22 (2026-11-20, zone contract). Code:
`batch_intersections._pair_paths_exact`, `_near_coplanar_edge_join`, `_plane_distance_bound`,
`_edge_joined_zones`. Tests: `tests/test_coplanar_edge_joined.py`.

## Where it applies

Only to a plane-plane pair that the support predicate `qualified_plane_plane` refuses with exactly
`ill_conditioned_plane_plane`: the sine of the plane angle is above the policy angular tolerance
(`TolerancePolicy.angular`, default 1e-10, below which the planes are already COINCIDENT) and at most
`sqrt(eps)` (~1.49e-8, below which the crossing line is not conditioned). That diagnostic is a float
classification; the rule then checks the band exactly on the stored planes (decision 23): the exact sine of
the stored normals must exceed the policy angular tolerance (else `near_coplanar_angle_below_band`, Codex z3 M2:
main's COINCIDENT route cannot take such a pair either, its polygon backend re-runs the float predicate and
refuses it) and be at most `2^-26` (else `near_coplanar_angle_outside_band`, Codex z1 M3). Every other plane-plane pair, and every other
support family, takes exactly the code path it took before: ordinary inputs are bit-identical.

## Admission (decision 12)

The pair is admitted to the zone contract only when all hold:

0. **Absolute-tolerance regime (option A).** The pair tolerance and both owning-face arrangement tolerances
   equal the absolute `TolerancePolicy.length`, and both face extents are at most 1000 m
   (`EDGE_JOINED_MAX_EXTENT`, kept by decision 22). Otherwise `near_coplanar_scale_above_absolute_tolerance`.
1. **Shared topological edge.** Some boundary path of each face has the same `source_edge` (model edge id).
   Geometric proximity, a shared vertex or a common sheet is not evidence.
2. **Certified coplanarity within the strict pair tolerance.** For each face D with plane P and the other
   face's plane Q: `max over D's boundary of |d_Q| + max over D's boundary of |d_P| <= t`, with
   `t = _owning_tolerance(pair tolerance, both faces' arrangement tolerances)`. Material points of D lie on P
   inside the projection of D's boundary; the distance to Q is affine on P, so its maximum is on the
   projected boundary, and `|d_Q(proj_P p)| <= |d_Q(p)| + |d_P(p)|`. Distances are to the exact plane of each
   `Plane` record (normal = exact rational `u x v` of its stored axes), in rationals rounded outward. Per
   curve: LinePath by its endpoints; EllipticArc by `|d(center)| + hypot(N.u, N.v)/|N|`; BezierPath and
   NurbsCurve (positive weights) by the control hull; any other family gives no bound.

## Zone contract (owner decisions 20-22)

Decision 23 (2026-11-20, owner option (a) "full margin", after Codex z3): every difference between the geometry
the rule decides on and the geometry used downstream is bounded and covered by the separation margin (class
audit below); the band is exact on both sides; loops are checked valid in projection; the zone point is exact.

Decision 20 replaces the decision 17 `k t` trace-accounting contract (and its decisions 15-19 machinery:
the float projection, the certified arrangement call and its `material_arrangement` hook, witnesses, trace
clipping, contact unions and trace subtraction). Decision 21: `c = 4` (`EDGE_JOINED_ZONE`). Decision 22: the
2^16 m coordinate fence and the 1000 m extent fence stay.

All decisions are exact, on the exact rational orthogonal projections onto P (the first face's stored exact
plane) of the STORED model vertex doubles behind each boundary path. Projection onto P is 1-Lipschitz, so a
projected distance `> t` is a world distance `> t`. No float is decided on.

**Inputs.** Every boundary path is a LinePath and, in order, its model loop's use of a straight model edge
(outer loop then holes, as `MaterialDomain.from_model` builds them); loops close by vertex identity and no
edge has equal end vertices; every domain path end lies within `t/8` of its stored vertex (exact; the
domain's reversed paths are re-evaluated curves, an ulp off, which at a tiny tolerance can exceed it, Codex z1
M2), and the largest such offset `d` is measured exactly and enters the margin; every stored coordinate is below
2^16 m in magnitude. Each face's loops, projected onto P, must be simple (consecutive segments meet only in their
common vertex, others are disjoint), pairwise disjoint, of non-zero area, with every hole strictly inside the
outer loop and outside the other holes (exact parity), else `near_coplanar_loops_invalid_in_projection` (Codex
z3 L1: model validation checks each face in its own chart, where a hole can be inside although it crosses the
outer loop after projection onto P). Otherwise `near_coplanar_overlap_unresolved` (or
`near_coplanar_coordinates_above_fence`).

1. **Shared edges.** S = the model edges both faces use. Each is used exactly once per face (else
   `near_coplanar_shared_edge_not_identical`); being the same model edge, it is the same stored segment in both
   faces. Z = the model vertices of S; `r = c t`.
2. **Contact only in S and its zones.** For every segment pair (a of A, b of B) except a shared edge with
   itself:
   - if a and b have no common model vertex in Z, their exact distance is `> m`;
   - else, for each common vertex V in Z, a rational point of a at world distance `>= r` from V (at parameter
     at most 1/2 from V, the midpoint when the segment is exactly `2r` long, else
     `near_coplanar_segment_shorter_than_zone`) is `> m` from b, and the same with a and b swapped.

   Here `m = t + sigma (t + d) + 2 d` (decision 23), `sigma = 2^-26` bounds the exact sine (the band gate) and
   `d` is the measured drift (an exact rational upper bound of its square root). `sigma (t + d)` covers the
   difference between the projected stored trims and the material (step (iv), Codex z2 M; the admission bound
   reads the domain curves, so a stored point is within `t + d` of its own plane), and `2 d` the difference
   between the stored and the arranged geometry of the two faces (Codex z3 M1).

   Otherwise `near_coplanar_contact_off_shared_edge`. The distance to b is convex along a and 0 at V, so a's
   contact set with b is an interval from V that ends before that point. The point is exact (an integer square
   root rounded up, `_zone_point`), so its distance from V is between `r` and `r (1 + 2^-60)` at any magnitude
   (Codex z1 L1, z3 L2: a float square root put it at 2r for a subnormal step): "within r" below means within
   that bound.
3. **Opposite sides.** On each shared edge the faces' material lies on opposite sides: material is left of a
   path's traversal in P's affine chart when its loop's exact signed area is positive and the loop is the outer
   loop, or negative and the loop is a hole (loops valid in projection, checked above; no dependence on which
   vertex starts a loop). Otherwise `near_coplanar_same_side_of_shared_edge`.
4. **No boundary inside the other face.** The midpoint of every non-shared segment lies outside the other
   face (exact even-odd parity over all its loops). Otherwise `positive-area overlap requires an explicit
   ownership policy`.

If all hold, the shared edges are emitted as traces, each as its model edge's own stored segment (same-sheet
decomposition seams that both faces carry are suppressed, as `_coplanar_traces` does), and
`_certify_edge_joined_traces` passes them by shared-edge identity.

**Proof.** Write A, B for the projected closed faces, int for interiors, all in P.
- (i) ∂A ∩ ∂B = S. A point x in both boundaries lies on segments a, b. If a, b have no common Z vertex, rule
  2 gives distance `> t > 0`. If they share V in Z, x lies in both contact sets, so x is within r of V on
  both; two distinct segments through V that meet again are collinear and overlap, which rule 2 refuses (the
  point r from V on one lies on the other). So x = V ∈ S. (Pairs with b = a in S contribute S itself.)
- (ii) Every non-shared segment a of A, without its end points, lies in ext B: by (i) its relative interior
  does not meet ∂B (a point of it in ∂B would be in S, but distinct segments of a valid face meet only at
  loop vertices), so it lies in one component of the complement of ∂B; its midpoint is outside B (rule 4; by
  rule 2 it is `> t` from ∂B, so parity is well defined). Symmetrically for B.
- (iii) Let C = int A ∩ int B, a bounded open set. Its boundary lies in ∂A ∪ ∂B; by (ii) only points of S
  and segment end points can be in ∂C. Near a relative-interior point of a shared edge s, no other boundary
  segment of either face comes close (each face is valid, and by (i) the other face's other segments miss s
  away from its ends), so there A and B are the two half-discs on either side of s (rule 3): C is empty
  there. So ∂C is a finite set of points. A bounded open set whose boundary is finite is empty (the plane
  minus a finite set is connected). So the interiors are disjoint and the boundaries meet exactly in S.
- (iv) World space (Codex z1 M1). Steps (i)-(iii) are exact for the projections onto P of the stored
  vertices. Each face's material, as admission 2 defines it, lies on its own plane inside the projection of its
  boundary onto that plane; for A that is exactly the projected A. B's material, projected onto P, differs from
  the projected B by at most `sigma t` (sigma <= 2^-26 the exact sine: a stored point within t of B's plane moves
  by at most t onto it, and only the out-of-P component of that move survives projection onto P, at most
  `sin` of the move). Every non-zone boundary point of B is outside A and `> t (1 + sigma)` from ∂A in
  projection, so this perturbation cannot bring it inside A, and the material boundaries stay `> t` apart there
  (in P, hence in 3D). The same holds for the arranged geometry (the domain paths), which is within `d` of the
  stored vertices on each face: separation `> m` leaves `> t` between the arranged boundaries. Hence any point common to both faces' material projects within
  `sigma t` (about 1.5e-17 m at t = 1 nm) of S or of a zone, and every such point is within t of the emitted
  shared edges or zones. Example (z1 M1): plates on `z = k x` and `z = -k x` (k = 1e-10) sharing a hinge stored
  0.25t above both planes: the supports cross on a line 0.25t from the emitted hinge, and the material overlap
  is a strip about 5e-20 m wide along it. The contract treats that as the shared edge (within t), not as an
  untraced intersection. Without the margin, a stored gap one ulp above t closed to 0.99999999988t on B's own
  plane (Codex z2 M).

Rules 2-4 are constant-size local predicates over segment pairs; there is no global accounting.

## Class audit: decided geometry versus geometry used downstream (decision 23)

The rule decides on the stored vertex doubles projected exactly onto P. Every place where other geometry is used:

| used downstream | differs from the decided geometry by | covered by |
|---|---|---|
| A's material (trims projected onto A's plane = P) | 0 | - |
| B's material (trims projected onto B's plane, seen in P) | `sigma (t + d)` | margin `m` |
| the arranged domain paths of either face (reversed paths re-evaluated) | `d` per face (measured, `<= t/8`) | margin `m` (`2 d`) |
| the admission bound (read on the domain curves) | `d` | in `sigma (t + d)` |
| the emitted traces (the shared model edges' stored segments) | 0 | - |
| the zone point (formerly float square root) | 0 (exact) | `_zone_point` |
| the band (formerly the float predicate's diagnostic) | 0 (exact sine, both sides) | the angle gates |
| loop validity and hole nesting (model validation, each face's own chart) | projection onto P | the projected-loop check |
| the coordinate fence (stored doubles; domain points within `d`) | scope only, not a soundness input | - |

## Typed refusals (all keep the `ill_conditioned_plane_plane` prefix)

| message suffix | meaning |
|---|---|
| `; near_coplanar_without_shared_edge` | no common topological edge (disjoint, overlapping, abutting with separate edges, or touching at a vertex only) |
| `; near_coplanar_outside_face_tolerance` | shared edge, but the certified coplanarity bound exceeds the strict tolerance |
| `; near_coplanar_boundary_not_bounded` | a boundary curve family without a proven distance bound |
| `; near_coplanar_scale_above_absolute_tolerance` | a participating tolerance exceeds the absolute `length`, or a face extent exceeds 1000 m |
| `; near_coplanar_coordinates_above_fence` | a stored trim coordinate is at or above 2^16 m in magnitude |
| `; near_coplanar_overlap_unresolved` | input outside the contract: a curved trim, a path that is not its model loop's straight edge use, a loop not closed by vertex identity, a domain path end more than t/8 off its stored vertex, a zero-area loop, a hole vertex on another loop |
| `; near_coplanar_shared_edge_not_identical` | a shared model edge used more than once by a face |
| `; near_coplanar_contact_off_shared_edge` | rule 2: contact within t away from the shared edges' zones (any overlap, near-tangent or point contact, a second geometric abutment, a sliver wedge at a zone) |
| `; near_coplanar_segment_shorter_than_zone` | rule 2: a segment at a shared vertex shorter than 2 c t = 8 nm (at t = 1 nm) |
| `; near_coplanar_same_side_of_shared_edge` | rule 3 |
| `; near_coplanar_angle_below_band` | the exact sine of the stored planes is at most the policy angular tolerance (the float predicate's diagnostic said ill-conditioned) |
| `; near_coplanar_loops_invalid_in_projection` | a face's loops are not simple, disjoint and nested once projected onto the first face's plane |
| `; near_coplanar_angle_outside_band` | the exact sine of the stored planes exceeds `2^-26` (the float predicate's diagnostic said ill-conditioned) |
| `; near_coplanar_trace_off_boundary` | an emitted trace is not certified on both boundaries (cannot happen with shared-edge identity; kept as the certificate) |

Rule 4 refuses with the existing `positive-area overlap requires an explicit ownership policy`.

**Fail closed by design (decision 20).** Refused, though the decision 17 contract could join them: every
overlap, however thin; every near-tangent or point contact away from the zones; a second, untagged geometric
abutment (two model edges at the same place: share the model edge instead, consistent with decision 15 M2);
wedges at a shared vertex sharper than about asin(1/c) = 14.5 degrees; a segment at a shared vertex shorter
than 2 c t; any vertex-only touch (consistent with decision 12b). HbeamBoth's sharpest wedge at a shared
vertex is 30 degrees (it joins for any c >= 2.5); face:12/face:16 joins with one trace (edge 62) in both face
orders.

Changed results: pairs that refused `ill_conditioned_plane_plane` on main now either plan as edge-joined or
refuse with that message plus one of the suffixes above. Nothing else changes.

## Changed tests (against the parked 078127d, which held the decision 17-19 contract)

New or changed tests (first commit) fail on the parked tip (45 of 103) and on main 11dddb1 (98 of 103: main refuses every
in-band pair).
- Joined there, refuse now (decision 20): `test_toothed_contact_shorter_than_k_t_refuses` (was `_joins`),
  `test_short_zigzag_contact_refuses` (was `_joins`), `test_contact_past_a_trace_end_shorter_than_k_t_refuses`
  (was `_joins`), `test_any_overlap_refuses` (replaces `test_overlap_just_above_k_t_refuses`,
  `test_overlap_within_the_tolerance_model_joins`, `test_overlap_just_below_k_t_is_joined_or_refused_typed`),
  `test_near_tangent_contact_off_the_shared_edge_refuses` (replaces the two `near_tangent_contact` tests),
  `test_duplicate_geometric_edge_beside_the_shared_edge_refuses` (was `test_trace_on_both_stored_trims_is_emitted`),
  `test_custom_relative_length_keeps_a_short_shared_hinge[5e-09-False]` (a 5 nm hinge < 2 c t, decision 21).
- The Codex d19 High (`test_seam_candidate_does_not_hide_contact_on_a_physical_edge`) was a strict xfail; it
  passes now.
- Same refusal, new typed code (rule 2, 3 or the fence decides first): the folded tests
  (`..._folded_back_over_each_other_refuse_typed`, `test_fold_inside_the_absolute_tolerance_regime_refuses_typed`,
  the chart-scale fold), `test_polygon_overlap_below_the_relative_area_tolerance_refuses`,
  `test_strip_overlap_dropped_by_the_arrangement_refuses` (straight cases), `test_trace_lifted_off_the_second_trim_refuses_typed`,
  `test_witness_closer_than_the_tolerance_to_an_oblique_edge_refuses_unresolved`,
  `test_witness_past_an_edge_end_is_not_a_boundary_point`, the toothed, zigzag and past-trace-end refusals,
  `test_projection_rounding_far_from_the_origin_refuses_at_the_fence` (1e7 m: the fence, no float projection).
- Refused there, joins now: `test_nearly_degenerate_stored_chart_is_decided_exactly` (was
  `test_projection_drift_is_measured_against_the_stored_plane`: the float chart's drift no longer exists).
- Removed with its function: `test_trace_is_emitted_from_exact_world_end_points` (`_emit_trace`), replaced by
  `test_emitted_trace_is_the_stored_shared_edge`.
- No monkeypatching: `test_exactly_coplanar_neighbours_do_not_use_the_band_rule` reads the rule's success
  counter instead.
- New: sliver wedges (5 and 14 degrees refuse, 15 and 30 degrees join), same side, a plate in a hole
  sharing the hole edge (joins) and over the material beside it (refuses), a same-sheet seam shared edge (not
  emitted), the far comb inside the fence (0 m and 2^15 m).
- Codex z1 regressions (each failed on 567653d): `test_stored_plane_angle_above_the_band_refuses_typed` (M3,
  joined), `test_domain_path_off_its_stored_vertex_refuses_typed` (M2, untyped arrangement refusal),
  `test_hinge_exactly_two_zones_long_joins` (L2, refused). Codex z2: `test_gap_just_above_t_closes_on_the_second_support`
  (joined on 282a2ee).
- Codex z3 / decision 23 regressions (each failed on b43187e): `test_reversed_path_drift_cannot_close_a_gap_below_t`
  (M1, joined), `test_exactly_coincident_stored_planes_never_take_the_band_rule` (M2, joined),
  `test_loops_invalid_after_projection_refuse_typed` (L1, a trace was returned; with a seam the planner joined),
  `test_zone_point_is_exact_at_any_magnitude` (L2, the float square root gave 2r).
- Unchanged: `test_in_band_neighbours_sharing_an_edge_are_edge_joined_like_exact_coplanar` (plan shape equal to
  the exactly coplanar plan), the fence boundary tests and all other admission tests.

## History

Decisions 15-19 (own overlap check, exact predicates, the `k t` proven bound with k = 7.4, the absolute
coordinate fence, untraced contact = all contact minus the traces) were implemented on
`claude/coplanar-edge-joined-20261119` and parked at 078127d after the final decision 19 review round: each
Codex round found another way contact could be hidden by trace accounting (toothed, zigzag, past a trace end,
seam candidate), one class. The zone contract removes the accounting.

## Open defects on main (pre-existing, not fixed here)

The exactly coplanar (COINCIDENT) route `_coplanar_traces` is unchanged and not certified for thin or small
overlaps (reproduce on origin/main with exactly coplanar input; the edge-joined route does not use it):
- **Silent overlap inside 1000 m (Codex a1).** Its polygon backend decides dimension by a relative area
  tolerance (1e-10 m^2 at 100 m), so a 1e-11 m^2 triangle overlap plans and applies; its curved backend probes
  32t from a boundary midpoint, so a 25 nm x 0.3 m strip plans. Reproducer:
  `python docs/coplanar-edge-joined-probes/silent_overlap_base.py src 0`.
- **Scale.** About 1 m^2 at 1e7 m: a small plate folded over a 1e7 m plate plans silently. Reproducer:
  `python docs/coplanar-edge-joined-probes/scale_fold_base.py src`.
- **Chart scale.** Chart-space area and length are compared with world tolerances, so a stored chart scaled by
  1e10 plans an overlapping exactly coplanar pair as empty.

## Not decided (owner decision D2, 2026-11-19)

- Coplanar faces touching only at a shared vertex (HbeamBoth face:12/face:18, the next refusal): no
  vertex-touch rule and no tolerance change now.
- A tolerance looser than the strict owning-face tolerance (for example the SAT source resolution 1e-6 m).
