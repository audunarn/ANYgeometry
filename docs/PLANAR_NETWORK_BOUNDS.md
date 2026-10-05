# Planar member/Sheet receipt conservative bounds index

Scope: the conservative performance repair of the large planar Member/Sheet
receipt in `src/anygeometry/prepared_planar_member_sheet_network.py`. The
public contract (`query_prepared_planar_member_sheet_network`,
`validate_prepared_planar_member_sheet_network_binding`), every tolerance,
schema, flag and refusal semantics are unchanged; only the internal
candidate enumeration of already certified material scans is pruned.

## Measured baseline (root profile, not renewed here)

`reports/planar-network-db5071c/strip1000-receipt-profile.json` (installed
wheel of db5071c, strip1000, 142.919 s instrumented receipt) attributes
72.343 s of cumulative time to `_material_membership` over 250 calls, with
376,500 `_point_in_face` calls: each carrier segment enumerated every face
for coplanarity, every coplanar face for boundary events, and every coplanar
face again for each interval midpoint and endpoint classification. The
transverse point-contact completeness scan repeated a full per-face plane
intersection for every carrier mapping.

## Design

A per-call `_face_bounds_index(charts)` is derived once per qualification
from the already certified literal Straight charts:

- **Exact face bounds.** For every owned face, an axis-aligned bound of its
  certified literal Straight `xyz_loops` vertices is cached as exact
  `Fraction` pairs, together with the exact support-plane normal
  `u x v`. No tolerance, rounding or float enters the index.
- **Exact plane grouping.** Faces are grouped under a canonical plane key
  `(n/lambda, (origin.n)/lambda)` where `lambda` is the first nonzero
  normal component, so segment coplanarity is decided once per distinct
  support plane instead of once per face. Plane containment is a property
  of the plane, so the grouped decision equals the per-face decision.
  Singular frames (impossible from `_frame`) stay per face and are never
  pruned.
- **Strict disjointness pruning.** A coplanar face is removed from the
  material candidate list only when its bound is *strictly* disjoint from
  the closed segment bound in at least one axis; equality and touching
  configurations always remain candidates. Point scans (transverse contact
  completeness, endpoint boundary fallback) skip a face only when the
  exact point lies strictly outside the closed face bound.

## Why the exhaustive result is unchanged

- A pruned face shares no point with the segment: its boundary edges lie in
  its bound, the segment in its own, and the bounds are exactly disjoint.
  It therefore contributes no `_boundary_events` station, no interval
  material face and no endpoint material face. The interval decomposition,
  endpoint incident faces and deterministic sorted outputs are identical.
- The material of a face is contained in the convex hull of its certified
  loop vertices, hence in its cached bound; a point outside the closed bound
  cannot be in the closed material, so point-scan pruning is exact.
- Typed refusals are preserved: the support-plane refusal is still decided
  over *all* faces (through the grouped planes), before any pruning; a
  carrier in one support plane whose every candidate is pruned still
  refuses with `leaves the original face material`, never with the
  support-plane refusal. Degenerate carriers are refused earlier by the
  unchanged carrier ancestry proof, so the per-face degenerate projected
  segment refusal is unreachable with or without pruning.
- Cancellation checks remain per retained candidate, per interval and per
  scanned face; whole-document orphan/ownership validation, callback,
  definition-binding and mutation checks are untouched.
- The index is a local object of one qualification call. There is no
  global, revision-keyed or cross-query cache.

## Deterministic pruning census

`_material_candidate_census(controls, charts)` returns exact face counts —
`certified_faces`, `coplanar_faces`, `material_candidates`,
`conservatively_pruned_faces` — and never times execution. A later
root-owned performance diagnostic can diff these counts against
instrumented runs; the census is the comparable deterministic helper.

## Verification

`tests/test_prepared_planar_member_sheet_network_bounds.py` (84 tests):

- An independently retained exhaustive oracle (the original unpruned
  algorithm, kept in the test file) is compared against the optimized
  membership on 22 deterministic battery segments over ten hand-built
  exact faces: touching squares, a material hole (crossed, grazed,
  bounded, beside), a concave notch, a far coplanar face, a parallel
  distinct plane, a rescaled plane frame, an arbitrarily oriented tilted
  triangle, and transverse endpoint/boundary configurations. Success
  payloads and typed refusal messages agree in every case.
- Grouped coplanarity equals the per-face scan for the whole battery.
  Every pruned face is directly proven to carry no boundary events and no
  endpoint material. Closed-bound containment is proven implied by
  material and boundary membership over an exact grid of every face.
- On real fixtures (`connected_strip(10)`, `connected_hub(4)`), the whole
  `_qualify` payload is identical under the independent oracle, under a
  never-pruning index, and under the optimized path; the transverse
  point-contact scan is identical under a never-pruning index; public
  query/validate receipts are unchanged and non-mutating.
- Deterministic census counts are pinned for the diverse charts and for
  every carrier of the real fixtures.

Focused verification log: `reports/planar-bounds-affected-final.log`
(162 passed: the existing planar receipt suite, the new bounds suite and
the boundary member network suite). Comparative measurement:
`reports/planar-bounds-comparative-measurement.log` — strip(400) receipt
6.48 s pruned vs 9.38 s never-pruned with identical payloads and census
19,900 of 20,000 coplanar face decisions pruned; strip(100) 1.13 s vs
1.31 s; hub(50) 10.51 s vs 10.55 s (no prunable candidates, no harm).

## Limitations

- The repair reduces expensive exact predicates and boundary-event scans;
  worst-case face enumeration remains `O(MF)` for `M` carriers and `F` faces.
  Each carrier still scans coplanar bounds and distinct support-plane groups. The
  remaining strip1000 receipt cost lives outside this owner slice
  (definition checksums, component dependency resolution, serialization,
  current-chart tolerance validation) and is left to the root-owned
  comparable performance diagnostic.
- The index is rebuilt per qualification call by design; a query plus its
  binding validation build it twice. This is deliberate: no global
  revision cache is permitted.
- The endpoint boundary fallback scan is defensive (an endpoint outside
  all coplanar materials always refuses earlier in the interval
  decomposition); it is pruned exactly but is not expected to run.
- Pruning helps when faces are spatially separated (strips); documents
  whose faces all overlap one region (hubs) legitimately prune nothing.
