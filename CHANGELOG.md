# Changelog

All notable user-visible changes to ANYgeometry are documented here.

## Unreleased

- Private `anygeometry._slab_triangulation.triangulate_rings`: exact-coverage slab (trapezoid)
  triangulation of trim rings with holes, pure NumPy, invalid rings refused as `TRIM_INVALID`
  (not yet used by any public display mesh).
- New `anygeometry.display_tessellation` (not re-exported from `anygeometry`): display edge
  samples shared by every face using an edge (`edge_display_samples`, `DisplayTolerance`,
  `EdgeSamples`, typed `DisplayTrimRefused`). Narrow contract: lines, circles and ellipses on
  plane, cylinder and cone faces only, each divided in closed form (tangent angle, support
  azimuth, the exact chord sag); any other curve or incident support is refused with a typed
  code. Declared numeric domain: coordinates up to 1e9, stored angles up to 1e6 rad, distance
  from a support axis at least 1e-9 x max(1, |coordinate|) (an end exactly on the axis
  excepted), ellipse aspect x max(1, |stored angle|) at most 1e4 (1e6 before round 6); outside it
  `DisplayDomainRefused` (`OUT_OF_DOMAIN`). Circles and ellipses are divided in their local
  angle, lines from their exact evaluated ends; a line has constant azimuth only when its ends
  are collinear with the axis point; the end chords' sag bound includes the vertex offsets;
  counts are plain ceilings. Round-2 stability audit: the azimuth ray inverse and the singular
  values are formed without cancellation; the aspect x stored-angle bound also applies to an
  arc's projection across each cylinder or cone axis, and one closest-approach rule covers
  cylinder and cone arcs; lines project their exact end difference (a ruling on a tilted axis no
  longer raises), an end within 8 eps of an axis is on it, a line on the axis is
  `OUT_OF_DOMAIN`; circles need orthogonal equal axes to 8 eps; the two-interval rule counts
  vertex offsets; every count is checked before any division is built; no raw arithmetic error
  escapes. Corpus counts unchanged, parameters moved by at most 2.2e-16. Round 3: lines take
  each end's radius from its own projection (no `A' + P(b - a)` reconstruction), check the
  segment's interior closest approach before any end shortcut, and admit an end on the axis
  only at a cone's apex (a cylinder or frustum end there is `OUT_OF_DOMAIN`); the radius bound of
  a line point uses the scale of the edge's definition points. One public
  `display_rounding_allowance` (`DisplayRoundingAllowance(angle, sag, axis_floor)`) defines
  "to rounding", counting the evaluator's stored-angle rounding and the projected inverse's
  aspect amplification; a request for which it exceeds 2 % of the angle (`OUT_OF_DOMAIN`) or of
  the sag allowance (`EDGE_UNSUPPORTED`) is refused; the summed division counts are checked
  against the 4096 cap before any division is built. All corpus results unchanged bit for bit.
  Round 4 (owner decision, the recommended option): a `LineRange` is supported only when its
  carrier IS the edge's segment (exact domain `[0, 1]`, carrier endpoints the evaluated ends
  within `model.tolerance.length`); a range drawn over part of its carrier is now
  `EDGE_UNSUPPORTED` (it was divided from its rounded ends, which can lie off the exact carrier
  beyond both bounds). The line terms of the rounding allowance add each evaluated end's
  absolute coordinate rounding (`eps x max |coordinate|`). The two-interval rule decides on the
  drawn chord (measured vertex offsets, never zeroed, plus the evaluator's rounding). An end on a
  support axis is an apex only on a cone with an exactly zero radius and within
  `model.tolerance.length` of the apex point (a frustum with a tiny radius, or a tip off the
  apex along the axis, is `OUT_OF_DOMAIN`). All 5744 corpus results unchanged bit for bit.
  Round 5 (owner decisions, the recommended options): a projected frame that is a similarity
  to 8 eps (a circle seen along its axis) has aspect exactly 1; a conic's closest approach is
  checked at its documented scale (definition points and stored-angle term, no longer the
  support origin) and, separately, against `64 eps |c - o|`; `DisplayRoundingAllowance` adds
  `axis_floors`, each incident support's own floor, which the dense oracle now uses; a
  `LineRange`'s carrier endpoints must also be the edge's topology vertices within
  `model.tolerance.length` (else `EDGE_UNSUPPORTED`); a nondegenerate ellipse beyond the aspect
  bound is `OUT_OF_DOMAIN` before the parallel-axes test. All 5744 corpus results unchanged bit
  for bit; 168 rounding allowances moved by at most 3.5e-16 relative.
  Round 6 (owner decision, the recommended option): the bound on aspect x max(1, |stored
  angle|), the conic's own and its projected frame's, is lowered from 1e6 to 1e4 (still
  `OUT_OF_DOMAIN` beyond; conic rounding amplified by the aspect had been measured beyond the
  published bound within a factor ~1e2 of the old one), and compared inclusively with a 4 eps
  relative slack; on a cylinder or cone a circle's stored angles are therefore bounded by 1e4.
  The in-plane frame's small diagonal is the residual's norm (it was `v'.e2`, a relative error
  of eps x aspect^2). `axis_floor` is documented as a summary only; callers measuring support
  turning use `axis_floors`. No corpus case is newly refused and no refusal code or rounding
  allowance changed. Near-duplicate cuts of different divisions are merged when the dropped cut
  lies, in its own division's angle, within that division's headroom of the published
  allowance (never below the former 64 eps rule), keeping the stricter division's cut: no
  sliver intervals (the `steep_cuts` ellipses had one, 1.04e-14). Against 2427aa3 every corpus
  count and refusal code is unchanged and 205 parameter arrays moved by at most 7.0e-15.
  Round 7: the union is "merge, then verify with exact forward maps; restore on failure":
  every merged interval is checked against every published bound (each division's angle span
  from its own forward map plus its rounding term; a circle's exact sag) and an interval that
  fails gets original cuts back (never a new one), at worst the no-merge union; the ends and
  the minimum-two-interval split are mandatory (a tangent interval turning 0.8000606 rad against
  0.8000418, a circle chord above its sag and a one-chord result are fixed). Against 9efd08e
  every corpus count, refusal code and allowance is unchanged; 2 parameter arrays (`tripod_skins`
  circles) moved by at most 2.2e-16; no sliver kept.
  Round 8 (owner decision, the recommended option): a new domain bound `MIN_DISPLAY_SWEEP =
  1e-8` rad on `|sweep|` of every `EllipticArc` and `Arc` (inclusive), checked before any
  division, `OUT_OF_DOMAIN` below it (zero sweep included; it was `EDGE_UNSUPPORTED`): with stored
  angles up to 1e4 their float spacing is at most 1.8e-12 rad, four orders below it. Both
  round-8 reproductions (1e-16 and 5e-11 rad) are now `OUT_OF_DOMAIN`. A forced two-interval
  split whose cut rounds onto an end uses the parametric midpoint, verified after the union; the
  azimuth branch keeps its sign when both end values snap to zero (decided from the unsnapped
  values and the midpoint). Tests pin the 2 % rounding-share cutoffs. Against 5f69ccb all 5744
  corpus results and allowances are unchanged bit for bit; no new refusal.
  Round 9 (owner decision, the recommended option): the azimuth branch of a conic on a
  cylinder or cone is decided from `g = 1 + z.w` only with margin. Every "snap `g` to zero at an
  end" rule is gone: `|g|` at both ends, every interior extreme and the midpoint must exceed a
  rigorous rounding bound (`g_bound`: input, inverse and evaluation terms, stated in the
  stability table), else the edge is `OUT_OF_DOMAIN` ("azimuth near-stationary: branch not
  decidable"). Both round-9 reproductions (a strictly monotone edge refused as a turnback and a
  real 1.8e-10 rad turnback drawn) are now `OUT_OF_DOMAIN`. Changed result: an arc whose azimuth
  is exactly stationary at an end (round 1 #6) and the round-8 internal both-ends-in-band case
  were drawn and are now `OUT_OF_DOMAIN`. Against 5bc1cd8 all 5744 corpus results and
  allowances are unchanged bit for bit; no new refusal.
  Round 10 (owner decision, the recommended option): the support azimuth (normal turning,
  branch decisions, `g` and its bound) is computed in the module's OWN orthonormal frame about
  the support's unit axis (`e1` from the axis crossed with its least-aligned basis vector,
  normalised; `e2 = axis x e1`), never by projecting onto the stored radial / circumferential
  directions; turning is independent of the frame's phase. A `Cylinder` or `Cone` whose stored
  frame is self-inconsistent (`|axis . radial|`, `|axis . circumferential|` or
  `||radial| - 1|` above 1e-9, absolute) is refused `OUT_OF_DOMAIN` ("support frame not
  orthonormal"), its own `evaluate()` disagreeing with its declared axis. (First version 8 eps:
  it refused 65 of 20,000 random ordinary cylinders, worst 117 eps; corrected to a sanity check,
  since a stored frame's rounding enters no bound; 0 of 20,000 random cylinders and cones
  refused, worst stored-frame error 1.8e-11.) Against c52a325 the corpus is unchanged bit for bit.
  Round 11: the stored axis need not be unit (a finite seed whose squared norm is subnormal
  stored it 0.9999999995 or 1.125 long). `_axis_frame` normalises the axis itself, and the 1e-9
  sanity check covers the lengths of axis, radial and circumferential directions and all three
  pairwise dot products. The 0.9999999995 support (cylinder and cone) is admitted and drawn
  within the bound (it was one interval 2e-11 rad over it); the 1.125 support is refused
  `OUT_OF_DOMAIN`. A cone's apex keeps the stored axis (it is the support's own point). Both round-10 reproductions (a radial seed
  nearly along the axis, stored 2.2e-5 off orthogonal: an arc with an interior turnback drawn,
  a strictly monotone arc refused) are now `OUT_OF_DOMAIN`; with the orthogonal seed `(1, -1, 0)`
  the turnback is refused and the monotone arc drawn. The projection error budgets carry the
  module frame's deviation (4 eps). Changed results against 3f066d2: 318 of 5744 corpus results
  move by at most 9.4e-16 in parameter (the frame's phase rounding; no count or code change, no
  new refusal); 2016 rounding allowances grow (at most 91 % relative, the line and projection
  error terms; the largest share of the angle and sag is unchanged).
  See `docs/DISPLAY_TESSELLATION.md` (stability table, rounding allowance). No existing module
  changes.

- Prepare version 0.4.7 for bounded B1 curved-hull hydrostatics point queries,
  including independently verified volume, body-origin first moments and raw
  waterplane moments. Results bind the model, displacement boundary, installed
  implementation, exact half-space and accuracy policy. Unsupported or
  unresolved cases refuse explicitly; event intervals, equilibrium, GZ and
  tanks remain outside this delivery. See `docs/CH3E_DELIVERY.md` for the pinned
  delivery and verification procedure; consumer admission is a separate gate.

- Add the certified-hydrostatics provider and independent verifier for closed
  planar shells (`anygeometry.certified_hydrostatics`): exact rational volume,
  first volume moments and waterplane quantities for a half-space, with
  one-sided limits at degenerate planes, binary64/decimal enclosures, canonical
  proofs and a separate verifier. This was the planar scope introduced in
  version 0.4.6; bounded curved support is added by the 0.4.7 work above.

- Add NURBS definitions (`anygeometry.nurbs`: `NurbsCurve` for 3D curves and 2D
  pcurves, `NurbsSurface`, `CurveOnSurface`) with validation (degree <= 7,
  clamped knots, weight ratio <= 1e6, numeric ranges), evaluation and strict
  records. Their records (`nurbs_curve`, `nurbs_pcurve`, `nurbs_surface`,
  `curve_on_surface`) are schema 7; readers accept schema 7 and ordinary
  documents are unchanged. No engine route uses NURBS yet.

- Add an optional compiled core (`anygeometry._core`, C++17, CPython stable
  ABI, one `cp311-abi3` wheel per platform) behind `anygeometry.backend`, with
  pure-Python references, `ANYGEOMETRY_DISABLE_NATIVE` /
  `ANYGEOMETRY_REQUIRE_NATIVE` and `anygeometry.backend_info()`. The native
  build is opt-in (`ANYGEOMETRY_BUILD_NATIVE=1`); default installs are
  unchanged. See `docs/NATIVE_CORE.md`.

- Add a read-only authored-root constraint/reference inventory bound to complete
  original/current geometry snapshots. It exposes trace incidence, neighbours
  and typed references without granting parameter remapping or mesh publication.

- Add public chart-definition fingerprints so consumers can retain the original
  chart/region content across callbacks. These read-only tokens supplement live
  owner validation; they do not certify geometry or permit meshing.

- Keep extrusion coordinates bitwise consistent between scalar and batch queries
  across BLAS kernels, preserving the existing inversion accuracy requirements.

- Resolve isolated tangent junctions of parallel Bezier walls on a shared
  quadric through exact projected-directrix elimination. Preserve seam-independent
  plane/pipe generator contacts and valid parameters at small elliptic-arc ends.
- Explicitly remap attachments, owning-member references and planned contacts
  when subsequent cuts split existing joints, including regularized child charts.
  Direct edge splitting opts in with `remap_attachments=True`; ambiguous mappings
  still fail atomically. Add portable schema-6 consumer fixtures.
- Validate Bezier branch fold endpoints and domain bounds; keep inversion caches
  out of analytic plan binding. Polish globally certified branch projections so
  exact boundary stations survive trim-aware face projection.
- Separate automatic development checks from explicitly dispatched full
  platform/installed-consumer qualification; retain every qualification job and
  budget. Add conservative local test selection and per-test CI timings. See
  [testing guidance](docs/testing.md).
- Faster general-intersection preparation, with identical plans, applied models
  and cancellation polls (runtime change; not yet requalified for release).
  Root isolation reuses Sturm counts across bisection nodes and, inside
  `plan_intersections`/`apply_intersections` only, shares identical isolations
  (cancellation polls are replayed). Topology validation filters segment pairs
  with a conservative vectorized test before the unchanged exact predicate.
  Applying a plan finds nearby shared vertices through a uniform grid and the
  boundary-event synchronization tests bounding boxes in one vectorized step.
  Measured on this machine: 6 cylinders through a plate 9.6 s to 4.6 s, an
  8 x 8 stiffened panel 2.4 s to 1.5 s.
  `tests/test_intersection_performance_equivalence.py` keeps the previous
  algorithms as oracles.
- Cone x Plane, Cone x Cylinder and Cone x Cone in `plan_intersections` /
  `apply_intersections` and `query_trimmed_surface_charts`, including a cone
  meeting a cylinder at any angle (for example 10 degrees off the
  perpendicular), which is not a conic. New root-exported immutable
  `QuadricIntersectionCurve` stores one exact branch of the ruling equation over
  an angular chart (evaluation, first and second derivatives, bounds,
  projection, splitting and affine copying, regular fold ends); circles,
  ellipses and coaxial rings stay `EllipticArc`. Cone material areas are exact
  and rectangular cone children of a cut become exact smaller `Cone` patches.
  Pointed cones, cone tips on planes and walls and cones with a common apex are
  supported; a face stored clockwise in its chart keeps its sense when split.
  `insert_model` now copies neutral faces that have no mapped corners.
  **Geometry documents are written at the lowest schema that can express them:
  schema 6 only when a document stores a `QuadricIntersectionCurve`**; every
  other document stays schema 5 (schemas 1-6 are read). See
  [cone and quadric intersections](docs/QUADRIC_INTERSECTIONS.md).
- Faster arrangement predicates without changing results beyond rounding: a
  plane meets an ellipse through the exact half-angle quadratic instead of
  Sturm isolation of an equivalent quartic, exact curve/quadric roots are
  remembered, and loop areas stop at their rounding-noise floor. Measured on
  this machine, the first plan of a 12 x 8 facet cone/cylinder pair 10 degrees
  off the perpendicular takes 0.6 s against 1.5 s for the equivalent cylinder
  pair (best of three, support plans emptied before each run).

- Extruded polycurve faces in `plan_intersections` / `apply_intersections` and
  `query_trimmed_surface_charts`: a face made by `GeometryModel.extrude` from a
  spline or an oblique circular arc is recognised as an exact translational
  surface. New root-exported `ExtrudedSurface`, `BezierDirectrix` and
  `EllipseDirectrix`. A plane meets it in an exact Bezier of the same degree
  (or an ellipse arc, or generators); extrusions of one direction meet in
  generators; an elliptic extrusion (an oblique circular tube) and a quadratic
  Bezier extrusion (a spline wall with one control point, a parabolic cylinder)
  meet cylinders, cones and elliptic extrusions through the quadric branch
  engine, including rulings parallel up to rounding, exact tangency and
  coincidence. A Bezier extrusion of degree three or more against a cylinder,
  a cone or an extrusion of another direction, and two quadratic extrusions of
  different directions, were refused with a typed error (see the next entry for the polynomial charts
  that take them). Children of a split keep their exact
  support, and so does a face whose boundary edge a contact splits. `query_intersection`
  classifies a pair that includes an extrusion with the exact engine first (one exact curve
  where the certified subdivision returned many sampled pieces) and falls back to the
  established query only where the exact engine refuses; a convex support that
  strictly contains a wall's edge keeps its established CONTAINED result and quadratic
  CONNECT partition. **A wall edge in a non-convex or trimmed support, formerly an explicit
  `UNCLASSIFIED` refusal, is now classified exactly** (the pinned fail-closed test of
  `tests/test_boundary_curve_connect.py` was replaced by checks of the exact curve, the
  planned imprint, topology and material conservation). Schema 6 is now also
  written for a document that stores an `ExtrudedSurface`. See
  [extruded surfaces](docs/EXTRUDED_SURFACES.md).
- Bezier walls of degree three and more in `plan_intersections` /
  `apply_intersections` / `query_trimmed_surface_charts` and the public query and
  imprint workflow: a spline wall made by `GeometryModel.extrude` meets cylinders,
  cones and elliptic or quadratic extrusions of any other direction, and two
  quadratic walls of different directions meet each other, through new exact curves.
  New root-exported immutable `BezierQuadricCurve` stores one exact branch of
  `A s^2 + B(t) s + C(t) = 0` (a constant `A`) over a chart of the directrix
  parameter, with the square and sine charts of `QuadricIntersectionCurve` at its
  folds (evaluation, first and second derivatives, bounds, projection, splitting and
  affine copying). The branch polynomials are evaluated in the Bernstein basis: in the
  power basis a degree-seven discriminant is wrong by 8e-10 and a degree-eight one by
  9e-9, the Bernstein form stays at 3e-14 (`docs/BEZIER_QUADRIC_INTERSECTIONS.md`
  gives the table), which also takes a quartic wall through whole models. Its junctions
  with lines, ellipses, Bezier paths, quadric branches and other Bezier branches are
  exact eliminations, or a certified box subdivision for two walls cutting one pipe
  (curves that never meet are proved disjoint, crossings are bracketed to the
  tolerance, a tangential or overlapping contact is refused). Roots and point inversions
  near a fold are polished in the chart parameter; before, a crossing within 1e-8 of a
  fold was dropped and a wall piece failed its material-conservation check (a pipe
  whose axis lies in the wall's base plane). Two non-parallel walls of degree three and
  more are still refused with a typed error. **Schema 6 is now also written for a
  document that stores a `BezierQuadricCurve`.** Measured on this machine, a cubic
  wall cut by one 8-facet pipe plans in 0.99 s and applies in 0.58 s (best of three,
  plans and memos emptied), against 0.63 s and 0.38 s for the quadratic wall of the
  angular engine. See
  [Bezier walls against quadrics](docs/BEZIER_QUADRIC_INTERSECTIONS.md).
- Faster exact root isolation without changing any isolated interval: Sturm
  rows, the square-free part, Yun classes and polynomial gcds use integer
  pseudo-remainder sequences instead of Fractions (every row is the same
  primitive integer polynomial; `tests/test_intersection_performance_equivalence.py`
  keeps the Fraction versions as oracles). Measured on this machine, the first
  plan of an oblique tube (4 arcs) through an 8-facet pipe takes 0.65 s against
  1.07 s, and of a cylinder pair 10 degrees off the perpendicular (12 x 8
  facets) 1.52 s against 1.71 s.
- Faster applying and planning for walls of straight chart edges: the
  conservative segment-pair filter of topology validation also rejects
  collinear stretches whose bounding boxes are apart, and the Bezier directrix
  inversion reuses its sample table and evaluates in the power basis. Measured
  on this machine, a spline wall cut by four plates applies in 0.31 s against
  0.82 s and plans in 0.36 s against 0.54 s.

## 0.4.5 — 2026-10-01

General intersections and coordinated meshing (accepted in the documented configurations):

- Root-exported immutable `EllipticArc` and `CylinderIntersectionCurve`
  definitions, with evaluation, derivatives, bounds, projection, splitting and
  affine copying. Analytic support intersections retain exact curves and isolate
  finite-height, angular seam and branch-transition events.
- Geometry schema 5 stores the new edge definitions and reads schemas 1–4.
  Automation remains protocol 1. Readers limited to schema 4 must update before
  reading documents written by this development head.
- `plan_intersections` / `apply_intersections` prepare a complete original
  operand set with revision and content binding, atomic application, shared
  topology, source lineage and explicit cancellation/resource budgets. Default
  batch preparation has no operand or intersection count cap.
- `query_trimmed_surface_charts` exposes bound Plane/Cylinder material domains,
  exact trim definitions, holes and batched owner evaluation. Changed source or
  evidence invalidates previously qualified chart bindings.
- Member pair CONNECT/IMPRINT plans prepare all occurrences and clipped
  material intervals through the batch engine. Relation-only policies retain
  all occurrences without changing topology; reuse requires every matching
  parent interval. Failed relation creation rolls back exactly.
- Changed generator output roles can retain existing structural owners and
  authored axis parameters when their topology correspondence is unambiguous.
  Ambiguous role changes remain atomic failures.
- Coordinated mesher/FEM integration passes sixteen installed meshes each on
  Windows, Linux and Apple silicon, including the exact replay at 0.5/0.25 and
  a 260-sector cylinder. The full 32-job platform matrix passes. Direct API
  recovery remains opt-in; default admission uses the qualified 15-degree
  floor, with caller-supplied stricter policies retained. Published as 0.4.5
  through the exact-artifact ledger/OIDC gate; consumer packages remain
  unpublished. See [the release evidence and consumer gates](docs/general_intersections_release_handoff.md).

The previously recorded projection and cylinder-atlas improvements below are
bit-for-bit identical to 0.4.4 within their recorded scope, including atlas
certificates, work counts, cancellation callbacks and budget refusals. Those
records do not qualify the new general-intersection capabilities above.

- Ruled and Coons surfaces evaluate without per-call NumPy scalar clipping.
  `face_local_uv_many` and `face_trim_loops_uv` invert explicit ruled and
  Coons patches for all points at once. The batch runs the same Gauss-Newton
  steps as `closest_uv`, row for row.
- Cylinders and cones compute their circumferential direction once.
- `face_trim_loops_uv` reuses loops computed at the same committed revision
  for the same `Face`. Nothing is cached inside a transaction, callers receive
  copies, and clone/deserialization start empty.
- Point-in-trim tests for small batches evaluate all edges at once.
- Topology-backed Coons evaluation reuses its four corner points at the same
  committed revision (same rule as trim loops), reuses boundary-chain length
  arrays keyed by the exact edge lengths, and blends per component in the
  former operation order. Curved face-face qualification on hole-punched
  plates, which calls this evaluation millions of times, is several times
  faster.
- The cylinder-atlas interval engine uses exact integer rounding for
  addition and multiplication. Repeated pure interval operations within one
  proof are memoized and replay their recorded work charges. Every query and
  binding validation still requalifies from scratch.
- `tests/test_projection_equivalence.py` keeps the 0.4.4 implementations as
  bit-exact oracles.

## 0.4.4 - 2026-09-28

- Add public, read-only model/world point and vector conversion helpers while
  preserving document units, metadata and automation protocol 1 behavior.
- Add explicit macOS Apple silicon/Intel and minimum-dependency CI coverage,
  installed-wheel smoke checks, and isolated consumer compatibility checks.
- Correct current contract documentation and add compatibility and owner
  handoff records. Newly configured CI coverage is not a historical qualification.
- Add qualified, batched surface-chart evaluation and a public proof contract
  for connected two-sector open-cylinder components, retaining the separate
  full-period cylinder atlas contract.
- Keep the schema 4 document format and automation protocol 1 while adding
  public affine coordinate-conversion helpers and cross-platform consumer
  compatibility coverage. The earlier maintenance evidence is retained in the
  [maintenance record](docs/MAINTENANCE_20260922.md).

## 0.4.3 - 2026-09-16

Overlap safety and strict bookkeeping. Published release evidence is linked
from [the compatibility record](docs/COMPATIBILITY.md).

- Require explicit `OverlapOwnershipPolicy.FIRST_SELECTED` for planar
  fragmentation and newly authored overlap features. Add immutable,
  revision-bound preview/apply APIs documenting owner and label consequences.
  Legacy saved overlap features retain their recorded first-selected semantics.
- Stop using sampled curved boundaries to return overlap/disjoint decisions.
  Incomplete curved qualification or unavailable certified region area raises
  `OverlapQualificationError`; no partial result is returned.
- Reject noncanonical Sheet/fragmentation IDs before allocation, freeze overlap
  result mappings, and avoid empty inherited tag entries.
- Limit fragmentation edge searches to incident/output topology and batch label
  propagation. Reuse projected polygons within overlap queries.
- Share dimensional planar segment checks between validation paths, including
  tiny plates and unequal-length near-parallel crossings. Preserve original
  Sheet/FaceUse orientation, labels, and both-source lineage during CONNECT.
- Keep cache invalidation kind-bound: deleting or rolling back a face/vertex
  no longer clears an unrelated edge cache with the same integer ID.
- Keep geometry schema 4, automation protocol 1, and MPL-2.0 unchanged.

See [0.4.3 compatibility notes](docs/OVERLAP_SAFETY_043.md) for migration examples
and the intentionally conservative curved/dependency boundaries.

- Add public feature-topology roles and exact `feature_entity_owners()`
  resolution so consumers can present composite generators as lightweight
  intent objects without duplicating feature-kind policy or changing topology.

## 0.4.2 - 2026-09-03

- Change the license for ANYgeometry source code distributed from this release
  onward from GPL-3.0-or-later to MPL-2.0. Earlier published versions retain
  the license terms under which they were released.
- License original project documentation under CC BY 4.0, preserve distinct
  terms for code snippets and third-party material, and add explicit copyright
  and third-party dependency notices.
- Add deterministic licensing validation to CI and release builds, including
  package-metadata, dependency-inventory, wheel, and source-archive gates.
- Keep geometry serialization schema 4 and automation protocol version 1
  unchanged.

## 0.4.1 - 2026-08-29

- Fix corner-to-corner planar imprinting by snapping qualified intersection
  endpoints to existing trim vertices before polygonization.
- Harden immutable release-ledger verification and the hosted distribution
  qualification workflow.

## 0.4.0 - 2026-08-24

- Add a shared NumPy-only qualified intersection engine for every unordered
  Straight/Arc/Spline pair and every curve/face and face/face combination of
  Plane, Cylinder, Cone, RuledSurface, and CoonsSurface. Public results carry
  bounded-work certificates, oriented parameter ranges, parameter-space
  paths/regions, residuals, enclosure widths, and typed certified traces.
- Route curved strict and changed-region audit candidates through the same
  public engine. Incomplete work, singularity, nonfinite arithmetic, and
  resource exhaustion remain blocking and fail closed; audit metrics expose
  boxes, subdivisions, trace segments, spatial work, and affected structural
  closure size.
- Add atomic multi-component curved imprinting and qualified full/contained
  coincident-region CONNECT while preserving Sheet/FaceUse/Coedge ownership,
  support and parameterization, semantic lineage, and structural relations.
  Reapplication is idempotent and rollback restores live state and derived
  caches without reusing provisional IDs.
- Add translation/scale, trim/hole/nonconvex, ownership-remap, locality,
  rollback, and complete built-in family matrices. Geometry document schema 4
  and automation protocol version 1 remain unchanged.

## 0.3.0 - 2026-08-24

- Add dependency-free automation protocol version 1 under
  `anygeometry.automation`. It provides strict immutable request/response
  records, JSON Schema/tool discovery, model-bound descriptions, bounded
  declarative selectors, explicit units/frames, measurements, and qualified
  intersection queries without parsing prompts or executing generated code.
- Add revision-bound `CommandBatch` planning and canonical SHA-256 `EditPlan`
  application. Symbolic point/edge/owned-plate construction, transforms,
  copies, mirrors, patterns, groups/tags, dependency-safe deletion, and
  explicit imprint policies commit through one outer kernel transaction.
  Planning and failed/stale/tampered application preserve revision, identity
  allocator high-water marks, and caches.
- Add the separate `ANYgeometry-mcp` reference package. It exposes exactly
  seven MCP tools against one startup-bound document, isolates session retry
  receipts, requires explicit mutation approval, and makes write-back both a
  startup capability and a per-call opt-in. The kernel has no MCP dependency.
- Keep persistent geometry document schema 4 unchanged. Automation protocol
  version 1 is a separate transport contract.

## 0.2.4 - 2026-08-22

- Add atomic exact `FeatureHistory.adopt_frozen(...)` for imported, scripted,
  unavailable-executor and committed mesh-partition materializations. Adoption
  validates active outputs and structural topology, persists a canonical
  closure checksum, and never follows lineage or geometric proximity.
- Preserve stable prefix outputs during additive feature regeneration and make
  dirty suffix replay fail closed on incompatible structural ownership.
- Add oriented Sheet ownership, deterministic structural intersection
  contracts, and the boundary-curve support required by ANYmesher 0.2.5.

## 0.2.2 - 2026-08-21

- Qualify exact face/face CONNECT when a complete topology-owned straight,
  circular-arc, or Bezier boundary of a nonplanar face lies strictly inside a
  convex hole-free planar support. Reuse the original Edge and persist shared
  FaceUse/Coedge ownership without sampled or coordinate-inferred topology.
- Keep partial, ambiguous, holed, nonconvex, trim-touching, and general
  nonplanar intersections typed and fail-closed.
- Make the release workflow artifact-building only; PyPI publication remains
  an explicit manual Twine operation.
- Added owner-aware `FeatureHistory.adopt_frozen(...)` for trusted importers
  and script features. It binds exact active output IDs, computes and persists
  the canonical materialization checksum, optionally verifies a caller's
  expected checksum, and rolls back without publication on any failure.
- Made additive feature regeneration start at the earliest dirty record on a
  working clone of the live materialization. Clean-prefix IDs remain active;
  dirty outputs are matched only by stable output key plus exact complete
  closure, and changed outputs allocate above the previous high-water mark.
  Appending an unrelated feature and editing or undoing a suffix therefore no
  longer invalidates raw IDs belonging to unchanged upstream outputs.
- Qualified the package-owner boundary for crossing Sheets, Member/Sheet
  relations, and Member/Member connections. Queries and plans remain strictly
  read-only; topology and structural ownership change only through an explicit
  `apply_imprint(...)` with declared connection intent. No proximity welding is
  inferred or restored.
- Kept geometry document schema 4 and feature-history schema 1 unchanged;
  0.2.2 is an API and contract-hardening release.

## 0.2.1 - 2026-08-12

- Added the mesher-facing `ModelClosure` extractor with model-bound
  bidirectional handle maps, complete structural parents, preserved document
  settings, and deterministic vectorized edge/face evaluation and projection.
- Added typed `query_intersection` / `plan_imprint` / `apply_imprint` APIs,
  multi-component and coplanar planar imprinting, explicit connection intent,
  qualified member/member and member/sheet relations, persisted face/face
  shell-sheet T-junctions under explicit `CONNECT`, and idempotent reuse.
- Made `REUSE_EXISTING` strictly non-creative, preflighted invalid pair/policy
  combinations, preserved endpoint and Sheet-specific relationship kinds, and
  made disjoint/same-parent/point-only face operations fail closed before edit.
- Canonicalized straight sketch-extrusion walls to exact Plane support and
  aligned arc walls to Cylinder support; genuine non-planar Coons walls remain
  typed unsupported.
- Added batched analytic closest projection for Plane, Cylinder, and Cone,
  one-time trim preparation, deterministic sweep-boundary behavior, and
  bounded-memory spline evaluation.
- Added construction/control vertex roles, optional face parameterization
  separate from authoritative support, Member orientation references,
  structural split lineage, richer Attachment/Junction evidence, and complete
  local reverse-incidence queries.
- Added changed-region non-certifying audit, richer audit evidence, indexed
  overlap selectors and nearest queries, local-scope batch/bounds paths, and
  pair-local tolerance scaling.
- Advanced geometry writing to strict checksummed schema 4. The 0.2.1 reader
  accepts schemas 1–4 and migrates 1–3 one-way; schema-3 attachment evidence
  becomes explicitly `UNVERIFIED` and cannot certify. Older 0.2.0 readers fail
  closed on schema-4 documents despite the compatible Python version range.

## 0.2.0 - 2026-08-12

- Added model-bound UUID entity handles, monotonic non-reused geometry and
  structural IDs, immutable public records/stores, and revisioned document
  settings.
- Made feature-history edits owner-observed and revisioned, published feature
  regeneration atomically with topology, prohibited snapshot mutation from
  change hooks, and preserved coedge identity across orientation-only edits.
- Replaced whole-model edit snapshots with nested delta transactions,
  deterministic change sets, incremental reverse incidence, and a maintained
  dynamic AABB index for changed-region work.
- Added persistent parts, sheets, face uses, coedges, physical members,
  member-edge uses, beam/plate attachments, and beam/beam junctions, including
  split/copy/reversal/lifecycle bookkeeping and batched member construction.
- Added a model-owned scale- and translation-stable tolerance policy, typed
  qualified line/plane/cylinder and segment predicates, multi-interval planar
  face clipping, and explicit mutation policies for welding and imprinting.
- Added deterministic fail-closed full-model auditing for duplicates,
  crossings, overlaps, T-junctions, manifoldness, ownership, member intent,
  face intersections, spatial-index consistency, and unsupported candidates.
- Added schema-v3 checksummed serialization for model identity, revision,
  coordinate settings, tolerance, allocator high-water marks, all structural
  ownership and relationships, semantics, extensions, and feature history.
  Schema 1 and 2 documents migrate conservatively; certified output requires
  a clean strict audit.
- Added a persistent, dependency-aware, suppressible feature history with
  extensible namespaced executors, stable feature-output references, atomic
  regeneration, and replacement-lineage preservation.
- Added atomic model insertion, deep cloning, topology/structural-closure
  copying, mirroring, linear and circular patterns, orientation reversal, and
  typed geometry measurements.
- Added qualification benchmarks, hostile regression tests, package typing
  verification, and a public kernel architecture/report set. ANYmesher and
  downstream analysis packages remain unchanged in this release.

## 0.1.0 - 2026-08-08

- Established ANYgeometry as the single lightweight geometry authority for the
  ANY ecosystem.
- Extracted persistent `GeometryModel`, `Vertex`, `Edge`, `Face`, `EntityRef`,
  curve, topology, evaluation, and general editing behavior from ANYmesher.
- Added semantic groups, tags, stable replacement history, and stale-reference
  resolution across geometry edits.
- Added explicit plane, cylinder, cone, ruled, and Coons surface support,
  projection and closest-point queries, transforms, trimming, holes, and
  fragmentation.
- Added analytical line/plane/cylinder intersections, planar shell
  intersection imprinting, and a deterministic numerical surface fallback.
- Added atomic transverse plane-cylinder closed-ring imprinting for qualified
  conformal shell bands. Exact shared arcs fragment the plane and every
  cylinder patch while preserving IDs, replacement history, groups, tags, and
  serializable surface definitions.
- Added neutral plate, stiffened-panel, cylinder, cone, shell, frame,
  bulkhead, girder, and stiffener generators with semantic geometry groups.
- Added versioned JSON and gzip geometry serialization that preserves topology,
  entity IDs, ID counters, surfaces, groups, tags, holes, metadata, and lineage.
- Hardened general edits so face splitting, trimming, hole punching,
  fragmentation, and qualified intersection imprinting either complete with
  valid topology or restore the exact pre-edit model. Splits retain intact
  holes deterministically and reject cuts that touch or cross a trim.
- Made topology snapshots repeatably restore mutable point/edge state, nested
  metadata, explicit surfaces, semantic groups, tags, ID counters, and
  replacement history; public deletion now records explicit empty lineage.
- Added strict topology, surface-boundary, trim-containment, overlap,
  self-intersection, degenerate-curve, replacement-graph, and serialized-ID
  validation. Triangular planar fragments remain evaluable after edits.
- Preserved cylinder and cone evaluation under reflected uniform transforms by
  correcting parameter handedness and sweep direction, and made public normal
  evaluation boundary-safe for ruled surfaces.
- Added a typed-package marker and a lightweight `python -m anygeometry` CLI.
- Kept NumPy as the only core runtime dependency; Shapely is an optional
  backend for planar clipping and qualification workflows.
