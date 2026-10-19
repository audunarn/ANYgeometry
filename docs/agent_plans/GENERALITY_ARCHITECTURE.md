# ANYgeometry generality architecture (ARCH slice) — revision 6, 2026-10-13

> **Status note (2026-10-19).** Implementation has departed from this revision in places;
> the current status is in `GENERALITY_PROGRAMME_PLAN.md` ("Current status"). In
> particular: plane x NURBS sections are evaluated exactly (records `plane_nurbs_exact_v2`,
> rational root texts), not by the interval-Newton `CertifiedBranch`/`kappa` design of
> sections 2.1, 2.3 and 4 (record `plane_nurbs_v1` is refused on load); the section 8a
> "not yet" list (A1b/A1c/A3b/A4a) is superseded; section 4.5 (junction pairs) must be
> redesigned on the exact design before B2. The bow form of section 6 (two patches sharing
> the stem) is not decided (Q2).

Status: design, no source changed. Rev 1 (dada559), rev 2 (09ac434), rev 3 (685af6e),
rev 4 (69a30d2) and rev 5 (693cf1a) received independent Codex reviews (GPT 6 Astra,
high). Rev 5's review found 2 blocking + 4 major findings; rev 6 is the **third and last
attempt on the narrowed design** under the retry discipline: each finding is either
resolved by a simple sound rule or narrowed to a refusal with an owner question (section
10, rev-5 -> rev-6 table); no speculative machinery is added. The A0 data model (3.1-3.2,
`nurbs_*` records) is **unchanged** by rev 6; only the `plane_nurbs_intersection` record
(A1c) changes. Rev 4 followed the owner decisions of 2026-10-13:

1. **Robustness first.** The current engine's decisions at tangencies, seams, endpoint and
   station membership and vertex merging depend on the last bits of floating-point chart
   inversion and evaluation (finding F1, confirmed by three S1 reviews and by the probe in
   R.0). These decisions are made robust *before* any rounding-changing speed-up or native
   kernel is admitted on those paths. Until then only bit-identical speed-ups merge.
   Section R is the new main part: the inventory, the certified replacements, the
   equivalence contract, the perturbation harness and slices R0-R6.
2. **Scope.** Plane x NURBS is designed rigorously (section 4): curves on the bow shell,
   iso-curves, the deck trimmed by a plane x NURBS curve, frames and stiffener planes
   cutting the shell, members on the curved shell. NURBS x NURBS, quadric x NURBS and the
   general `SupportIntersectionCurve` machinery are deferred to section 4L ("later") and are
   not designed in detail.
3. NURBS weight ratio `max/min <= 1e6`; new NURBS records go to **schema 7** (schema 6 stays
   as it is); bow loading is hydrostatic pressure on the bow with the existing clamp at
   X = 0; the wheel CI workflow is not run without the owner. Q1, Q3, Q6 and Q11 are closed.

Base: `claude/generality-programme` on local main 0a3a602 (schema 6 unreleased; 0.4.5 writes
schema 5). Inputs: the deep dive, plan rev 2 (D1-D7), `ECOSYSTEM_GUIDE.md` (correctness
first, fail closed with typed errors, no silent fallback, backward-compatible schemas,
patch-only versions), the code cited by file and line, the three S1 reviews
(`review_s1*_result.md`) and the N0/N1a scaffold.

Reading guide: 0 facts; **R the robust-predicate programme**; 1-3 contracts (capabilities,
curve contract narrowed to plane x NURBS, NURBS data model); 4 plane x NURBS design; 4L
deferred work; 5 native core; 6-7 driving case and consumers; 8/8a slices and the
safe-to-implement list; 9 risks and open questions; 10 changes from rev 5 (10a-10d
history).

Vocabulary. *Exact*: defined by finitely many stored doubles, in real arithmetic (the
convention of `QuadricSupport`, `planar_area`, `cylinder_angular_lifts`). *Enclosure*: an
outward-rounded interval object proven to contain the exact quantity. *Certified*: decided
from enclosures or exact arithmetic only, never from samples and never from a rounded
intermediate compared against a threshold. *Decision*: a discrete output (classification,
membership, snap, merge, count, ordering, refusal). *Refuse*: raise a typed `GeometryError`
subclass naming the quantity, its enclosure and the threshold; an empty result or a snap is
never a substitute. *Rounding-sensitive*: a decision whose value can change when one
floating intermediate that is not exact in the stored doubles changes by one unit in the
last place.

## 0. What the code does today (facts the design builds on)

- **Surfaces** (`surfaces.py`): `Plane`, `Cylinder`, `Cone`, `ExtrudedSurface`,
  `RuledSurface`, `CoonsSurface`; contract `SurfaceProtocol` = `evaluate(u, v)`,
  `local_uv(point)` (56-62). `Plane.local_uv` is `np.linalg.lstsq` (85-89, SVD-based,
  platform- and BLAS-dependent); `Cylinder.local_uv`/`Cone.local_uv` are `arctan2` plus the
  64-eps wrap snap `_angle_on_sweep` (146-157, 215-226, 712-723); `ExtrudedSurface`
  carries its own comment that BLAS reduces a single row differently from a batch, FMA
  included (303-306). Batch evaluation/derivatives are `isinstance` chains (514-586,
  589-710); `closest_uv` (726-753) is finite-difference Gauss-Newton with an `lstsq` step.
- **Edge curves** (`curves.py` 69-103): `Straight`, `Arc(via_vertex)`, `Spline`
  read geometry from vertices; the exact families (`EXACT_CURVES`, `exact_curves.py:604`) are
  self-contained definitions added by `add_curve` (4604-4614). Duck-typed exact-curve
  protocol: `evaluate`, `derivative`, `second_derivative`, outward `bounds`, `subcurve`,
  `transformed`, hashability, `roots_on`, `parameters_of`. Derived caches are excluded from
  content checksums by `field(metadata={"definition": False})`.
- **Pair dispatch**: `batch_intersections._pair_paths` (377-433) over `*_support` functions
  returning `SupportIntersection(curves, segments, points, coincident)`. Curve-in-support
  containment `member_arrangements._support_roots` (39-105): exact polynomial/trigonometric
  roots for plane/cylinder/cone/quadric families, a *sampled residual* for `ExtrudedSurface`
  (42-45).
- **Arrangement predicates** (`arrangement_geometry.py`): `plane_roots` (151-196),
  `point_parameters` (199-266), `curve_junctions` (377-392) ending in the refusal at 462 for
  unlisted pairs; `_merge_touching` (395-427) merges junctions of quadric branches within
  `_TOUCH = 1e-6`.
- **Material domains** (`material_arrangement.MaterialDomain` 62-436): chart points via
  `support.local_uv` (218-237) with a seam-side choice by `round()` (230-236); planar
  tangent/curvature via `lstsq` (242, 269); `horizontal_roots` via `pinv` (291);
  `contains_loop` ray parity (305-346); `area_loop` is chart area (exact rational/Decimal for
  planar loops of lines/Beziers/ellipses, `planar_area.py:70-118`; Gauss quadrature
  otherwise); `arrange_material` (581-831) merges vertices by first-fit distance (670-675),
  detects duplicate edges (685-726), orders half-edges by `atan2` (738-767), drops cells by
  `area <= tol^2` (796-798) and checks conservation (828-829).
- **Attachment remap** (`attachment_remapping.py`): `_target_bounds` (65-87), the sound
  outward-rounded box filter `_box_rejects` (147-158), `remap_face_attachments` (161-235).
- **Trimmed charts** (`trimmed_charts.py`): `_qualified_chart` (113-131) requires
  `_support_roots is None` for every boundary path; `evaluate_trimmed_surface_chart` (255-282).
- **Query route** (`curved_intersections.py`, `intersections.py`, `predicates.py`): planar
  faces become shapely polygons of `plane.local_uv(vertex)` and are classified by
  `polygon.covers`/`polygon.boundary.covers` (`intersections.py:125-171`); curved faces use a
  17-samples-per-edge UV polygon (`_support_trim_loops` 806-832) with float ray parity
  (`_point_in_polygon` 786-803); curve/surface roots from 65/257 samples with sign changes
  and Newton (`lstsq` at 297, 766, 1869; candidates 1178-1196); kinds from
  `|t.n| <= angular |t||n|` and `parameter <= parameter_tolerance` (1233-1238);
  `_snap_unit_parameter` (`predicates.py:706-711`).
- **Existing certified machinery (to be reused, not duplicated)**: exact rational chart
  projection `planar_area._fraction_vector/projected` (17-26, 79-83); exact ray comparison
  `cylinder_angular_lifts.exact_ray/compare_exact_rays` (131-200); the `Fraction`-interval
  prover `cylinder_charts._I/_Proof` with budgets, `sqrt`, `atan2` and `sincos` enclosures
  and outward `out()` (446-784); exact root isolation `analytic_roots.isolate_real_roots`
  (196, exact on its rational coefficients); `native_arc_station_incidence._harmonic_bound`
  (189) certified distances in `Fraction`; `polynomial_extrusion_support.
  prove_polynomial_extrusion_support` (176); outward curve boxes (`arrangement_geometry.py:
  66, 108`, `exact_curves.py:204-205`, `branch_curves.py:659, 681`).
- **Serialization**: `VERSION = 5`, `_QUADRIC_VERSION = 6`; the document is written at the
  lowest version that holds it (734); readers accept 1-6 (1806); unknown curve types are
  refused (1601). **Fingerprints**: `definition_checksum` hashes floats (content binding).
- **Consumers**: ANYmesher dispatches on surface classes (~25 places) with one generic route
  (`_analytic_metric_chart.py:21`); ANYfem authors planar faces, attaches members by
  `project_to_face` UVs, runs `plan/apply_intersections`.
- **N0 scaffold** (`claude/geom-n0-native`): `native/_core.cpp`, `native/include/*.hpp`,
  `backend.py` selection switches, `reference/` twins, abi3 wheel tag, opt-in build.

## R. Robust-predicate programme for the existing engine

### R.0 The problem, measured

Finding F1: a decision made by comparing a rounded intermediate against a threshold (or
against another rounded intermediate) changes when the intermediate changes by one ulp.
The probe `scratchpad/probe_rev4.py` (read-only, Python 3.14, this worktree) confirms the
three headline cases:

| Case | Observation |
|---|---|
| `Plane.local_uv` lstsq vs closed-form Cramer | for 2000 points stored as the rounded evaluation of the `u = 0` edge of a generic affine plane, the two routes disagree on `sign(u)` 794 times; `intersections.py:156-171` then flips `polygon.covers`/`boundary.covers` (TOUCH_POINT vs DISJOINT in `test_curved_intersection_matrix.py::test_plane_support_completes_fifteen_face_pair_families`) |
| `Cylinder.local_uv` NumPy vs scalar (review s1c case 2) | NumPy snaps `u` to `0.0`, the scalar route returns `1.42e-14`; `MaterialDomain.contains` flips False/True |
| `_angle_on_sweep` wrap snap | `angle - start = 63 eps` -> `0.0`; `65 eps` -> `1.44e-14` (a hard threshold on a transcendental intermediate) |

The S1 reviews document further concrete inputs: endpoint membership
(`arrangement_geometry.py:207`, a 1-ulp norm difference), vertex identity
(`material_arrangement.py` vertex lookup at exact tolerance, underflow, overflow, float32),
split-station dedupe, planar rates (`lstsq` at `material_arrangement.py:242/269`) and the
elliptic point parameter (`arrangement_geometry.py:229-242`) whose chosen station feeds the
native arc maps selected in `test_native_arc_station_incidence.py::
test_direct_ancestor_versus_double_tolerance` (the test picks the record whose
`current_bound == 0`; a different station changes which record that is).

Consequence for D1: "classification identical" cannot be shown for a rounding-changing
speed-up on these paths because the reference decision itself is not a function of the
stored data. The cure is to make every decision a function of the stored doubles (exact) or
of certified enclosures with a stated margin (certified), after which "identical" is
well-defined and testable.

### R.1 Inventory of rounding-sensitive decisions

Grouped by mechanism. *What* is decided, *why* it is rounding-sensitive, the *certified
replacement* (R.2 toolkit letters), and the slice. Line numbers are of this worktree
(9ff7ed0). Entries marked (sound) are already conservative and are listed so that the
pattern is explicit.

**Group A — chart inversion feeding decisions**

| Id | Location | Decision | Why sensitive | Replacement | Slice |
|---|---|---|---|---|---|
| A1 | `surfaces.py:85-89` `Plane.local_uv` (lstsq); consumers `intersections.py:129-171`, `curved_intersections.py:1101, 1113 -> _inside_face 835`, `predicates.py:945-948 -> _snap_unit_parameter 706-711`, `material_arrangement.py:222, 429`, `attachment_remapping.py:192-195`, `model.py:6734` | planar membership (inside/boundary/outside), parameter snaps to 0/1, chart ordering | SVD solve of a 3x2 system; the result differs from the exact rational solution by several ulps and its sign at an edge is arbitrary (R.0) | T1: exact rational Cramer on stored doubles (`planar_area.projected` already does this); decisions through T3 on the exact values; the float `(u, v)` returned for coordinates is the correctly rounded exact value | R1 |
| A2 | `surfaces.py:146-157, 215-226`, `quadric_supports.py:91-102, 105-132`, `analytic_supports.py:51-56`, `predicates.py:819-835` | point inside the angular/axial patch; `u` snapped to 0 (seam) | `arctan2` of rounded dot products, then `_angle_on_sweep` (712-723) snaps within 64 eps | T2/T3: azimuth membership is `compare_exact_rays` (stored-double radial coordinates are exact rationals) against the enclosure of the boundary rays `(cos, sin)(start)`, `(cos, sin)(start + sweep)` from `_Proof.sincos`; the ray of a stored angle `0.0` is exact `(1, 0)`; an undecided straddle refuses `SeamAmbiguous` unless topology decides (R.2 rule T6) | R4 |
| A3 | `material_arrangement.py:230-236` (and `207-216` extruded) seam-side selection by `round((middle - uv) / period)` | which side of a full-turn seam an endpoint lies on | `round()` of a quotient of rounded angles | T6: the side is taken from the curve's own exact chart (an `EllipticArc` on the cylinder has angle `start + t sweep` relative to its frame; the phase between the arc frame and the cylinder frame is an exact-ray comparison); curves without an exact chart on the support refuse | R4 |
| A4 | `surfaces.py:712-723` `_angle_on_sweep`; `quadric_supports.py:120-128` | wrap of an angle to the sweep's side | 64-eps threshold on a transcendental intermediate (probe case 3) | withdrawn from decisions (A2/A3 replace it); kept for coordinate output only | R4 |
| A5 | `batch_intersections.py:300-333` `_rectangular_cylinder` | which rectangle side a boundary path lies on (`<= tolerance` on `domain.uv`) and whether the sides are covered | compares inverted chart values with thresholds | T3 on exact rational quantities: side identity from the path's stored definition (`LinePath` direction x axis = 0 exactly? -> compare squares exactly; `EllipticArc` centre on axis, radius equality: exact rational squares); coverage intervals from exact chart endpoints | R4 |
| A6 | `surfaces.py:726-753` `closest_uv`, `curved_intersections.py:297, 766, 1869` (lstsq Newton), `model.py:6800-6820` | none directly (coordinates), but their outputs feed A1/B-group decisions in the query route | local Newton; convergence `<= 1e-12` | coordinates only: allowed to vary within the bound of R.3; every decision downstream must come from R-certified predicates, never from these outputs compared to thresholds | R1/R5 |
| A7 | `batch_intersections.py:387-391` plane x plane trace: `LinePath` built from `component.witnesses[0]`/`direction` of `qualified_plane_plane` (`predicates.py:1139-1147`: `cross / sine`, `np.linalg.solve`) | the **definition** of the generated arrangement curve | the witness and direction are rounded; every station later placed on that `LinePath` certifies the rounded line, not the two planes (rev-5 major 3) | T0 extended to **generated-curve definitions**: the arrangement path is `GeneratedLine(plane_a, plane_b)` whose anchor (the point of the line closest to the origin: exact Cramer solution of `[n_a; n_b; n_a x n_b] x = [d_a; d_b; 0]`) and direction `n_a x n_b` are exact rationals of the two plane records; stations on it have exact rational positions; its float `LinePath` is emitted at the output boundary only (`add_line` vertices are canonical roundings, R.3). Same rule for seams (D7) and for the plane x cylinder/cylinder x cylinder `segments` of `analytic_supports` (E6: endpoints are exact trigonometric stations) | R1 (planes), R4 |
| A8 | `batch_intersections.py:764-768` child-plane rebasing: `_detect_corners` + `norm(p0 + p2 - p1 - p3) <= tolerance` selects `Plane(p0, p1 - p0, p3 - p0)` | which affine chart a planar child face gets (changes UVs, digests, attachment parameters) | parallelogram test on rounded vertex sums | T1 exact squares on stored vertex positions (rational); the choice is then an exact function of the child's stored loop | R1 |
| A9 | `model.py:7561-7570` `_detect_corners`: `arccos` of evaluated tangents, ranking `sorted(key=(-deviation, i))` — a discrete choice **without a tolerance constant** | which four loop vertices become mapped-face corners (ties broken by index) | evaluated tangents and `arccos` are rounded; equal or near-equal deviations are ranked by rounding | ordering decisions go through T3 too: the ranking key is `compare_exact_rays`-based turning (the cosine is a rational of exact tangent rays for lines/Beziers, an enclosure for arcs) with ties and undecided comparisons refusing (`"mapped-face corners not separable"`); listed here as the pattern for *ordering decisions without thresholds*, which the audit must enumerate (R.4) | R1 |

**Group B — point/parameter membership on curves**

| Id | Location | Decision | Why sensitive | Replacement | Slice |
|---|---|---|---|---|---|
| B1 | `arrangement_geometry.py:220-228` `_point_parameters(LinePath)` | point on segment (`-tol <= t <= 1 + tol` and `norm(evaluate(t) - point) <= tol`) | both the parameter and the distance are rounded; review s1: a 1-ulp norm change flipped membership | T1: `t = ((p - s).d) / (d.d)` exact rational; distance squared at the exact `t` exact rational; T3 exact compare with `tol^2` (never undecided) | R2 |
| B2 | `arrangement_geometry.py:229-242` `_point_parameters(EllipticArc)`, `357-361` arc-domain acceptance after `ellipse_curve_junctions` | point on ellipse (`|c^2 + s^2 - 1|`, residual), turn range `±tol`, angle, **arc-parameter membership** of a candidate | lstsq for `(c, s)`, `atan2`, thresholds on rounded values; candidates are evaluated then re-inverted | T0 station input; T1 for `(c, s)` (exact Cramer in the ellipse frame), exact on-ellipse residual compare; turn/endpoint and arc-domain decisions by exact-ray comparison of `(c, s)` against the canonical enclosures of the start/end rays (T2/T3); the returned float parameter is a coordinate | R2 |
| B3 | `arrangement_geometry.py:199-214` `point_parameters` endpoint snap (`<= 128 eps`), `207` endpoint distance | endpoint identity | thresholds on rounded parameter and norm | T3: endpoint identity = exact distance-squared compare `dist^2(point, c(0)) <= tol^2` on exact end points (stored vertices for lines; exact rational evaluation for Beziers; `sincos` enclosure for arcs -> three-valued) | R2 |
| B4 | `arrangement_geometry.py:151-196` `plane_roots` (160, 162-165, 169, 188, 193-195) | coincidence with the plane, root in `[−tol, 1 + tol]`, root count | coefficients formed in float before exact isolation (`isolate_real_roots` is exact on *rounded* inputs); coincidence by rounded sums | T1: form the univariate coefficients in `Fraction` from stored doubles (as `member_arrangements.py:88-98` already does for the cylinder case); coincidence exact (all coefficients zero) or T3; roots from `isolate_real_roots` with exact isolating intervals; range tests exact | R2 |
| B5 | `arrangement_geometry.py:130-148` `_angle_parameter`, `material_arrangement.py:301-303` endpoint snap, `322-324` (`8 eps`) | endpoint identity of angular roots | 128/8-eps thresholds | as B3 (exact distance compare) ; `trigonometric_roots` outputs keep their isolating intervals so endpoint coincidence is decided on intervals, never on the float witness | R2 |
| B6 | `material_arrangement.py:564-578` `_parameters` (`<= 32 eps`, `norm(hi - lo) <= tol`) | merging of split parameters | thresholds on rounded parameters and outward boxes | T3 on exact quantities: parameters are compared exactly (they are stored floats); the sub-curve diameter test uses the exact box of the exact sub-curve where available, else the outward box (sound) with three-valued compare | R2 |
| B7 | `batch_intersections.py:842-849, 855-857` split-station dedupe and unique-descendant check | which stations split an edge; refusal when `len(stations) != 1` | rounded distances `<= tol` | B1/B2/B3 exact membership; the station set is then a function of the stored data | R2 |
| B8 | `arrangement_geometry.py:269-298, 301-362` line/ellipse junctions (`pinv` at 316, 326, 343; `<= tol` at 272, 349) | junction parameters and coincidence | `pinv` is SVD; coefficient sums rounded | T1: exact Cramer instead of `pinv`; coefficients in `Fraction`; B4 isolation | R2 |
| B9 | `member_arrangements.py:103-105` root acceptance `[-32 eps, 1 + 32 eps]` on `root.witness` | whether a root is on the curve | threshold on the float witness of an exact isolating interval | T3 on the exact isolating interval (`root.lower/upper`) against `[0, 1]`; an interval straddling 0 or 1 is refined by exact bisection until decided (canonical schedule) | R2 |
| B10 | `arrangement_geometry.py:259-265` `_point_parameters(BezierPath)`: final acceptance `norm(curve.evaluate(t) - point) <= tol` after `plane_roots` | whether a root of the coordinate plane is a point on the Bezier | the root witness is rounded, the curve is re-evaluated in float, the norm is rounded | T0: the root is kept as its exact isolating interval; the distance to the station point is an exact rational function of the parameter (Bezier) evaluated as an enclosure over the isolating interval, refined by the canonical schedule; decided by T3 | R2 |
| B11 | `batch_intersections.py:959`, `material_arrangement.py:681-682`, `member_arrangements.py:162`, `material_arrangement.py:607-613` | vertex/endpoint identity from `curve.evaluate(0.)`, `subcurve(a, b).evaluate(...)`, `point.position` | the evaluated tuple is a rounded intermediate; exact arithmetic on it certifies the rounding, not the geometry (rev-4 blocking 2) | T0 stations: `(definition, parameter)` with the parameter exact (stored double or isolating interval); positions are derived enclosures; see C1 | R1-R3 |

**Group C — vertex identity and duplicate edges**

| Id | Location | Decision | Why sensitive | Replacement | Slice |
|---|---|---|---|---|---|
| C1 | `material_arrangement.py:670-675` `vertex()` first-fit `norm <= tol` on `subcurve(a, b).evaluate(0./1.)` (681-682) | arrangement vertex identity | rounded norm of rounded evaluations (review s1: exact-tolerance, underflow, float32 cases; rev-4 blocking 2) | T0 + T3: `vertex(station)` where a station is `(path, parameter)` with `parameter` a stored double or a junction root's isolating interval; the squared distance between two stations is an exact rational (line/Bezier stations at rational parameters: exact de Casteljau) or a canonical enclosure (arc stations, interval parameters) compared with `tol^2`; deterministic first-fit order kept (part of R.3); the vertex *position* written to the model is the rounded evaluation (a coordinate) | R3 |
| C2 | `batch_intersections.py:907-952` `vertex(point, sources)` grid + `<= tolerance` + canonical scan (943-946), input `curve.evaluate(0.)` (959) | canonical vertex reuse during apply | rounded norm of rounded evaluations; grid cell rounding (`floor(value / size)` 897) can disagree with the norm by one cell at exact tolerance (review s1b/s1c) | as C1 (stations against stored vertex positions, which are exact stored data); the grid stays a *filter* only if proven conservative: cells computed outward from the station's enclosure, cell size `>= 2 tol`; otherwise dropped on this path | R3 |
| C3 | `material_arrangement.py:685-726` duplicate-edge detection (`abs(x) <= 1e-12` at 708, `pinv` 717, `<= tolerance` 720-725), `batch_intersections.py:570-600` `_coincident` (`pinv` 574) | whether two arrangement edges are the same curve | rounded junction parameters, SVD pseudo-inverse | T1: exact Cramer; coincidence of two `EllipticArc`s = exact rational equality of centre/frame up to tolerance (T3 on exact squares) plus exact midpoint membership (B2); lines: exact by endpoints (already); Beziers: exact control-difference hull (T1) | R3 |
| C4 | `batch_intersections.py:980-984` circular-arc test before `add_arc` | representation of an ellipse edge as `Arc` vs `add_curve` | `<= tolerance` on rounded norms | T1 exact squares; a representation choice but it changes the record type (topology digest) and must be exact | R3 |

**Group D — containment, ordering, area on material domains**

| Id | Location | Decision | Why sensitive | Replacement | Slice |
|---|---|---|---|---|---|
| D1 | `material_arrangement.py:305-346` `contains_loop` (`point_uv[0] <= uv[0]` 322, `before <= 0 < after` 342-345, boundary `norm(world_delta) <= tol` 320) | point in loop (winding) | all operands are inverted chart values (A1/A2) and midpoint samples | R.2 T4: ray parity in the chart with crossings isolated exactly: planar faces -> exact rational UVs (A1) and exact univariate roots (B4); cylinder faces -> exact axial coordinate (rational) and angular comparison (A2); a ray through a vertex or tangent to a path -> next ray from a fixed sequence, 8 rays then refuse | R1 (plane), R4 (cylinder) |
| D2 | `material_arrangement.py:738-767` half-edge ordering (`atan2` 751, `64 eps` 752/760, curvature 754) and `239-281` `tangent`/`curvature` (`lstsq` 242/269) | cyclic order of edges around a vertex, tangent ties | rounded angles grouped by 64 eps; planar tangents via lstsq | T1: `compare_exact_rays` on exact chart tangents (lines, Beziers: rational; arcs at endpoints: `sincos` enclosures -> T2 with refusal when undecided); exact ties broken by exact curvature where rational, else refuse ("tangent junction not separable") | R1 (plane), R4 |
| D3 | `material_arrangement.py:796-798` cell drop `abs(area) <= tol^2`; `828-829` conservation; `trimmed_charts.py:229` area binding | whether a cycle is a cell; whether the arrangement is complete | planar loops of lines/Beziers are rational but `planar_area.py:104` rounds the rational to float before the comparison; ellipse terms use a finite-precision Decimal series (105-118) and round again; other loops are Gauss quadrature with an *estimated* error (`_adaptive_gauss` 468-509) | T3 on a certified area object: `area_loop` returns `AreaEnclosure(lo, hi)` (rational `lo = hi` for line/Bezier loops; for ellipse terms a rational interval from the canonical `sincos` enclosures of the stored angles, replacing the Decimal series; for quadrature loops a rigorous bound from interval evaluation of the integrand's derivative per panel, R5); the float area is reported from the enclosure; decisions only when the enclosure is separated from the threshold, else refuse (rev-4 major 6a) | R1 (plane), R5 |
| D4 | `material_arrangement.py:169-184` `from_model` authored frame choice (`> tolerance.length`, `> tolerance.area`) | which affine frame the plane chart uses | thresholds on rounded norms | T1 exact squares (the chart choice changes UVs and hence digests) | R1 |
| D5 | `material_arrangement.py:154-156` `left_probe` offset `16 tol` (called with `2 tol` at 802/815, i.e. a `32 tol` displacement) | material side of a cell; hole ownership | a probe point, not a certificate; a `32 tol` displacement can cross a narrow strip or hole and the containment then answers for the wrong region (rev-5 major 4) | **certified probe-in-cycle**: the probe `q_h = m + h n` (station `m` at `t = 1/2` of the cycle's first edge, `n` its certified left normal) is accepted only when the certified containment of D1 shows `q_h` **inside the cycle's own loop** (ray parity on the cycle, not on the boundary) — then `q_h` is in the cell by definition and its material test against the outer loop and holes answers for the right region; `h` starts at `32 tol` and is halved while `q_h` is outside the cycle or undecided, at most 8 times, then refuse (`"cell probe not certified"`). No upper-bound argument on strips is needed: the probe is validated against the cell itself. Hole ownership (815-820): the hole's probe is validated inside the hole cycle the same way, and the owner is the containing positive cycle with the smallest area enclosure strictly separated from the others (D3), else refuse | R1 (plane), R4 |
| D6 | `material_arrangement.py:606-628` seam construction for interior-ended traces: `point_parameters(boundary, point)` (607-608, 612), `length == 0` (619), ray dedupe `norm(ray - old) <= 64 eps` (624), `parameter == 1 / == 0` (622), `len(rays) > 1` (626), endpoint dedupe `<= tolerance` (628) | whether a trace end needs a decomposition seam and how many seams | all operands are evaluated points/tangents of rounded roots (rev-4 major 5) | T0 stations for trace ends and for `points`; tangent rays as exact rational vectors (line/Bezier) or canonical enclosures (arc) compared by `compare_exact_rays`/T3; endpoint parameter identity exact on the station parameter; endpoint dedupe by certified station distance (C1) | R1 (plane), R4 (cylinder) |
| D7 | `material_arrangement.py:630-652` seam geometry: planar seams through the endpoint along the authored direction (643); cylinder seams at `support.local_uv(point)[0]` (649-650) | the seam line's position and hence every junction with it | planar: endpoint is an evaluated tuple (coordinate only, but the seam must pass *through the station* for the later junction to be exact); cylinder: the seam angle is an inverted rounded chart value | planar: the seam is defined as `LinePath` through the station's exact rational position (line/Bezier stations) or refuses for arc stations whose position is not rational (seam through an enclosure is not a line); cylinder: the seam ruling is the station's exact angular chart (T6) or refuses | R1, R4 |
| D8 | `material_arrangement.py:654-667` split collection `point_parameters(path.curve, point.position)` and pairwise `curve_junctions` | which parameters split each path | inherits B-group; `point.position` is an `ArrangementPoint` evaluated by `member_arrangements.py:162` | T0: `ArrangementPoint` carries its station (curve, parameter) in addition to the float position | R2 |

**Group E — support/coincidence certificates and query-route classification**

| Id | Location | Decision | Why sensitive | Replacement | Slice |
|---|---|---|---|---|---|
| E1 | `member_arrangements.py:42-45` sampled residual (17 samples) for `ExtrudedSurface` | curve lies on the extruded support | sampling is not a certificate at all | `prove_polynomial_extrusion_support` for Bezier extrusions (exists); ellipse extrusions: exact trigonometric coefficients as 65-76; otherwise refuse | R5 |
| E2 | `member_arrangements.py:65-76` sampled deviation + Lipschitz slack, `85-86, 99-100` coefficient sums `<= implicit_tolerance` | arc/line/Bezier on cylinder | sums of rounded coefficients; the sampled bound is sound in exact arithmetic but evaluated in floats | T1: coefficients in `Fraction` (99 already partly); the Lipschitz bound in exact arithmetic with `sincos` enclosures (T2) | R4 |
| E3 | `curved_intersections.py:856-986` `_curve_exactly_on_support` (`<= tolerance` 885-888, 895, 918, 929, 936-937, 945, 953-955, 970, 978, 986; angular `>` 891, 973) | the query route's "exact on support" shortcut | rounded dot products, norms, cross-product norms | T1 exact squares for every test (all are polynomial in stored doubles); angular tests as `|a x b|^2 <= angular^2 |a|^2 |b|^2` exact | R5 |
| E4 | `curved_intersections.py:806-839` sampled trim polygon + float ray parity; `1107, 1190` `_inside_face`; `intersections.py:134-171` shapely polygon of vertex UVs | curve/face material membership in queries | 17-sample polygons of inverted UVs; crossing arithmetic in float; a polygon of vertex UVs does not represent arc or Bezier edges (the barge's rounded openings: points between an arc and its chord change membership, rev-4 major 6b) | planar faces: **certified curved-loop containment** = `MaterialDomain.contains` of D1 over the exact chart images of the loop edges (the plane chart is affine, so a world line/Bezier maps to an exact rational line/Bezier in UV via `exact_uv`, and an `EllipticArc` to an exact ellipse arc with rational centre and axes); ray crossings from exact univariate roots (B4) and, for arcs, from canonical trigonometric enclosures with arc-domain membership (B2); shapely is withdrawn from decisions (witnesses only); cylinder faces: D1 on the native chart; other supports: unchanged, reported as "sampled" and excluded from the R-gate | R1 (plane), R4 |
| E5 | `curved_intersections.py:1178-1196` sampled sign-change root candidates, `1233-1238` CROSS/TANGENT/TOUCH_POINT kinds, `predicates.py:984, 1059, 1122, 1254` `sine <= angular` | curve/surface root set and kind | samples; thresholds on rounded cross products | plane faces x lines/arcs/Beziers: the exact predicates of `predicates.qualified_*` are kept but their comparisons move to T1 exact squares; sampled roots stay outside the R-gate (query route on curved non-cylinder supports) and are reported as such | R1/R5 |
| E6 | `analytic_supports.py:42-49 _ordered (46), 87, 112, 194, 200-232 _parallel (205, 228)` | plane x cylinder, cylinder x cylinder event merging, segment identity, coincidence | rounded distances and `|r1 - r2| <= tol` | T1 exact compare on stored doubles; event angles from `trigonometric_roots` carry isolating intervals | R4 |
| E7 | `member_arrangements.py:145, 175`, `material_arrangement.py:663` box pruning `hi1 < lo2 - tolerance` | pair skipped | `lo2 - tolerance` rounds; a 1-ulp inward rounding could prune a touching pair | (sound after fix) outward rounding as `attachment_remapping._box_rejects` (147-158) already does | R2 |
| E8 | `attachment_remapping.py:147-158` `_box_rejects` | prune | — | (sound): the reference pattern: conservative filter, exact fallback | — |
| E9 | `batch_intersections.py:337-374` `_coincident_cylinder_rectangles` (`angle_tolerance`, `zhi - zlo > tolerance`) | overlap vs line vs arc of two coincident cylinder rectangles | rounded `atan2` phase, thresholds | T2/T3 phase by exact rays + `sincos` enclosures; axial intervals exact | R4 |

**Group F — deferred families (not on the barge path)**

| Id | Location | Decision | Replacement | Slice |
|---|---|---|---|---|
| F1 | `arrangement_geometry.py:395-427` `_merge_touching` (`_TOUCH = 1e-6`) | merges junction pairs of quadric branches into one contact | a topology-inventing merge: withdrawn; tangential junctions refuse (as section 2.6 already states for new curves) | R6 |
| F2 | `quadric_supports.py:197, 235, 283-324, 377, 430-432, 558, 561` cone/quadric events, apex, coincidence | event merging, apex membership, coincidence | T1/T2/T3 on the `QuadricSupport` exact forms | R6 |
| F3 | `material_arrangement.py:198-205` `at_apex` (`<= 1e-9 r`) | cone apex chart singularity | exact axial/radial test | R6 |
| F4 | `batch_intersections.py:570-600` `_coincident` quadric/cylinder-branch cases | branch coincidence | existing exact `roots is None` plus B2 membership | R6 |

Inventory size: 45 entries (A 9, B 11, C 4, D 8, E 9, F 4): 1 already sound (E8), 4 deferred
(F1-F4), 1 off the barge path (E1, extruded supports), and 39 on the curved-bow barge's
path (planes, lines, arcs, Beziers, cylinders and the query route over them). **No
completeness claim is made for this table** (rev-5 major 3): it is a reviewed, growing
list; R.4 states what the mechanical audit can and cannot enumerate.

### R.2 Certified replacement toolkit (module `src/anygeometry/exact_predicates.py`)

- **T0 Provenance: stations, not evaluated tuples** (rev-4 blocking 2). Within one
  operation (`plan_intersections` + `apply_intersections`, a query, a chart qualification,
  an attachment remap) every point that enters a decision is a **station**
  `Station(definition, parameter)` where `definition` is a frozen curve definition built
  from stored data (`freeze_edge`, a support curve, a seam) and `parameter` is either a
  stored double (exact rational) or an **exact root** (`isolate_real_roots` isolating
  interval plus the exact polynomial it isolates; for trigonometric roots the
  `trigonometric_roots` isolating interval plus its exact coefficients). `subcurve(a, b)`
  keeps the parent definition and the exact affine parameter map, so a sub-curve end is the
  station `(parent, a)`; a junction is a pair of stations; an `ArrangementPoint` carries its
  station; a trace end is a station. The world position of a station is a *derived*
  quantity: an exact rational for lines and Beziers at rational parameters (exact de
  Casteljau), otherwise a canonical enclosure (T2/T3) over the parameter's isolating
  interval. Floats are produced only at the output boundary (`model.add_point`, witness
  tuples, UV coordinates). Generated curves (plane x plane lines, seams, support
  segments) are **generated-curve definitions** (A7) whose geometry is an exact expression
  of the operand records, never a rounded witness. **Boundary of the contract**: once an
  operation has written the model, the stored doubles are the definition for the next
  operation (the engine's existing convention); provenance is kept *within* an operation.
  To keep decisions identical across operations on every backend, everything an operation
  **persists** as a decision operand (vertex positions, curve definitions written by
  `add_point`/`add_line`/`add_curve`, attachment parameters) is the **canonical rounding**
  of the station's exact position (rational -> round-to-nearest, unique) or of the
  canonical procedure's level-`L_max` enclosure midpoint (deterministic, defined in exact
  arithmetic, backend-independent); fast routes may produce non-persisted coordinates only
  (display, witnesses, chart samples). See R.3.
- **T1 Exact rational layer.** Every quantity that is a polynomial in stored doubles and
  rational station parameters is evaluated in `Fraction` (or in a double-double/expansion
  filter whose a-priori error bound is checked first, section 5): dot/cross products,
  squared distances, Cramer solutions of 2x2/3x3 systems, Bernstein-to-power conversion,
  de Casteljau on rationals. Existing implementations are reused (`planar_area`,
  `cylinder_angular_lifts`, `member_arrangements.py:88-98`, `analytic_roots`). Thresholds
  are stored floats, hence rational; an exact comparison is never undecided.
- **T2 Canonical enclosures for non-rational quantities.** `cos`, `sin`, `atan2`, `sqrt`
  of stored doubles, and any quantity depending on an interval-valued station parameter,
  come from one **canonical procedure**: `cylinder_charts._Proof`-style `Fraction` interval
  arithmetic with a fixed, documented refinement schedule `L = 0..L_max` (level `L`: series
  precision grid `_GRID * 2**L`, isolating intervals bisected `L` extra times exactly, with
  the existing budget charges). The procedure is deterministic, depends only on the stored
  doubles and the level, and is implemented once in Python (`exact_predicates`, the
  reference). Where a decision can be rewritten to avoid the transcendental (compare
  squares instead of norms; compare rays instead of angles) it is rewritten.
- **T3 One decision policy for every arithmetic route** (rev-4 major 4).
  `certified_compare(q, threshold) -> BELOW | ABOVE | REFUSE` is *defined* as: if `q` is
  rational, compare exactly (never REFUSE; equality follows the predicate's stated rule);
  otherwise run the canonical procedure at levels `0, 1, ..., L_max` and return the first
  level's separated outcome; if level `L_max` is not separated, REFUSE with
  `DegenerateWithinBound(quantity, enclosure at L_max, threshold, entities)`. Every other
  route (NumPy, scalar Python, double-double, native) is a **filter**: it may return the
  outcome without running the canonical procedure only when its own valid enclosure is
  separated from the threshold by more than `W_max(q)`, the documented upper bound of the
  canonical level-`L_max` enclosure width for that predicate kind and input magnitude (so
  the canonical procedure is guaranteed to decide, and necessarily the same way, since every
  valid enclosure contains the exact `q`); in every other case the filter hands the
  predicate to the canonical procedure (in the native core: the undecided list of 5.3).
  Consequently all routes return identical outcomes *and identical refusals*. No snap, no
  "nearest side", no sample. The `Fraction` oracle therefore does not "adjudicate"
  transcendental predicates by exact evaluation; the canonical interval procedure with its
  fixed schedule is the adjudicator, and REFUSE is one of its legitimate outcomes (R.4).
- **T4 Exact chart predicates.** Point-in-loop by ray parity with crossings from exact
  univariate roots (planar line/Bezier edges: rational; arcs: canonical trigonometric
  roots with arc-domain membership), orientation tests `sign(cross(a - o, b - o))` exact,
  `compare_exact_rays` ordering; all through T3.
- **T5 No symbolic perturbation.** Simulation of simplicity is forbidden: it resolves
  degeneracies by inventing an order that is not in the data. Degenerate = REFUSE.
- **T6 Topology before inversion.** Where the stored topology or an exact chart already
  decides (an edge end is a vertex; a station's parameter is 0 or 1; an arc on a cylinder
  has an exact chart relative to its own frame), the decision is read from there and never
  recomputed by inverting a rounded world point. Chart inversion of world points is
  reserved for coordinates and for points that have no exact chart (refuse at seams).
- **T7 Operand-order symmetry.** Every binary predicate is defined on the canonical order of
  its operands (ordered by `definition_checksum`), so forward and reverse calls evaluate the
  same expression under the same policy and return the same decision or the same refusal.
- **T8 Decision trace.** Each certified decision records `(predicate_id, operand_digests,
  station_or_box, level_decided, outcome)` into a trace available through
  `runtime_diagnostics` (counters `certified_decisions`, `filter_decisions`,
  `canonical_decisions`, `undecided_refusals`). The trace is the object the equivalence
  contract compares.

### R.3 Equivalence contract (what a speed-up must show after an R-slice)

A path is **R-certified** when every decision in its trace comes from T0-T4/T6 under the
T3 policy (no raw float comparison remains and no evaluated tuple enters a decision). This
is shown **structurally** by the audit of R.4, not by the harness alone. On an R-certified
path a speed-up (pure Python or native) is admissible iff on the full suite, G0 and the
perturbation corpus:

1. the decision trace is identical (same predicates, same outcomes, same refusals, same
   order) — this includes vertex identity, split stations, kinds, seam sides, cell counts;
2. the float-free `topology_digest` (5.5) is identical and documents are byte-identical
   where the slice claims byte preservation;
3. coordinates (chart inversions, evaluations, witnesses, areas) are within
   `B = max(16 ulp of the exact rational value where one exists, 4 * tolerance.length)`
   of the exact or reference value — the reference is the exact value, not main's rounding;
4. the same inputs are refused (no new refusals, no removed refusals).

Before a path is R-certified only bit-identical changes merge on it (owner decision 1).
The deterministic order of first-fit merges (C1, C2) is part of the contract: inputs are
processed in sorted entity order, which both routes must keep.

**Across model writes** (rev-5 major 5). Item 3 alone would let two admissible backends
persist a vertex one ulp apart and disagree in a later operation at a degenerate
configuration. Therefore item 3 is split: (3a) **persisted decision operands are
canonical** (T0: canonical rounding of exact/canonical positions and definitions), hence
**bit-identical** across backends, and the documents written by an R-certified operation
are byte-identical; (3b) the bound `B` applies only to non-persisted coordinates. The
equivalence claim is thereby *per operation sequence*: with identical starting documents,
every operation of a sequence writes identical documents on every backend, so every later
decision has identical operands. Acceptance includes **multi-operation tests** (plan ->
apply -> plan again on the written document -> remap -> chart query, on G0 and the corpus,
across backends and under the harness) asserting byte-identical documents after each
step. Where an operation cannot produce a canonical persisted value (a coordinate with no
exact or canonical definition, e.g. a `closest_uv` result written as an attachment
parameter), it is an authored input, not a decision operand of the engine, and is recorded
as such in the trace.

### R.4 Evidence: what proves what (slice R0)

Three instruments, with their limits stated:

1. **Structural audit** (`tests/test_decision_sites.py`): an AST scan of the R-certified
   modules (`arrangement_geometry`, `material_arrangement`, `batch_intersections`,
   `member_arrangements`, `attachment_remapping`, `trimmed_charts`, `analytic_supports`,
   the planar parts of `intersections`/`predicates`/`curved_intersections`, and the
   construction routines they call in `model.py` such as `_detect_corners`,
   `_qualified_face_plane`, `freeze_edge` inputs) that enumerates three kinds of site:
   (a) comparisons involving a float expression and a tolerance/eps constant; (b)
   **ordering and selection without thresholds** (`sorted`/`min`/`max`/`argmin`/`argmax`
   with float keys, `round`, `floor`, `==` on floats, boolean branches on float signs);
   (c) **construction provenance**: every constructor of an arrangement/trace definition
   (`LinePath(`, `EllipticArc(`, `BezierPath(`, `Plane(`, `Cylinder(`, `Cone(`) whose
   arguments derive from an `evaluate(`/witness/`local_uv` result rather than from stored
   records or `exact_predicates` constructions. The test fails unless each site is an
   allowlisted `exact_predicates` call or is annotated `# coordinate-only` with a
   reviewer-checked justification. **What it proves**: on the scanned modules, every
   enumerable decision site of kinds (a)-(c) is certified or justified; absence of flips on
   those sites follows by construction. **What it cannot prove** (rev-5 major 3): that the
   module list is complete, that a decision hidden in a helper outside the list or in a
   library call (`np.linalg`, shapely) is caught, or that an annotation's justification is
   right; those remain reviewed claims per slice. The audit's module list is part of each
   R-slice's deliverable and is extended whenever a review finds a site outside it (as
   `model.py:_detect_corners` was).
2. **Perturbation harness** (`tools/perturbation_harness.py`): hooks at the capability
   boundary (A1a adapters) and at `freeze_edge` perturb every chart inversion,
   surface/curve evaluation and derivative by a deterministic `±1 ulp` pattern (seeded;
   modes `none`, `outputs`, `outputs+evaluations`) and switch the arithmetic route
   (`numpy`, `scalar`, `fma-emulated`). For each case of G0 plus the adversarial corpus
   (exact-tolerance vertices, underflow/overflow coordinates, float32 inputs, the cylinder
   seam case, the plane `u = 0` edge family, stations at `0.75 tol`/`1.5 tol`, the
   rev-4 paraboloid-cap plane section, a Bezier opening with points between arc and chord)
   it runs the engine operations under every perturbation and under operand reversal and
   asserts identical decision trace, identical `topology_digest`, coordinates within `B`,
   identical refusals. **What it proves**: it *falsifies* — a flip found is a defect with a
   reproducer. **What it cannot prove**: absence of flips (finite corpus, finite
   perturbation patterns); that is instrument 1's job. The baseline run on 9ff7ed0 reports
   the current flip count per inventory entry as evidence of F1, nothing more.
3. **Oracle cross-check**: every traced predicate is re-evaluated independently from its
   recorded stations: rational predicates by an independent `Fraction` implementation
   (exact agreement required); non-rational predicates by an independent interval
   implementation (`mpmath` at a precision exceeding `L_max`) which must *contain* the
   canonical enclosure's exact value and must agree with every non-refused outcome; a
   canonical REFUSE is checked by showing the exact value's distance to the threshold is
   below `W_max(q)` (so refusing was within policy). The oracle never overrides a refusal.

### R.5 Slices (ordered by barge impact and by what unblocks S1)

| Id | Scope (inventory ids) | Acceptance | Unblocks |
|---|---|---|---|
| R0 | `exact_predicates.py` (T0-T4, T7, T8: `Station`, canonical procedure with its level schedule and the `W_max` table, filters), decision trace in `runtime_diagnostics`, structural audit (initially reporting, not failing), perturbation harness, adversarial corpus, `topology_digest` in the G0 harness; no engine route changed | byte preservation on G0; audit report lists every decision site in the named modules (the completed inventory); harness baseline report lists flips for A1, A2, B1, C1, D6 at least; canonical procedure agrees with `_Proof` and `mpmath` on 1e5 inputs; filter/canonical identical-outcome test on 1e6 random predicates incl. constructed near-threshold cases | everything below |
| R1 | Station model in `arrange_material`/`_clip_intervals` (T0: stations for trace ends, sub-curve ends, `ArrangementPoint`s, seams D6/D7), generated-curve definitions and canonical persisted operands (A7, A8, A9), certified probe-in-cycle (D5), planar chart and planar membership: A1, A6 (plane consumers), D1-D7 (plane), E4/E5 (plane faces), `intersections.py:125-171` certified curved-loop decisions, `predicates._plane_parameters`; multi-operation byte-identity tests (R.3) | audit passes for the planar routes; zero flips under perturbation and reversal on G0 + corpus for every plane-only case; `test_plane_support_completes_fifteen_face_pair_families` stable under the harness; full suite: same failures as main (22/30 subset documented); G0 digests unchanged or every change explained as a corrected degenerate decision with its oracle proof (Q13) | S1 `Plane.local_uv` closed form, planar rates; all barge plate-plate work |
| R2 | Curve membership and stations: B1-B11, D8, E7 (stations through `point_parameters`, `plane_roots`, junctions, split stations) | audit passes; zero flips on the station corpus (`test_native_arc_station_incidence` selection stable); the review-s1 endpoint case decided exactly; split stations identical across routes; full suite unchanged | S1 `_norm3` paths, `point_parameters` scalarisation; stiffener/bulkhead junctions |
| R3 | Vertex identity and duplicates on stations: C1-C4 (`vertex(station)` in both engines, grid filter proven or removed) | audit passes; the review cases (`nextafter(1, 0)` at `tol = 1`, `1e-200`, `1e-170`, float32, mixed dimension, overflow order reversal) give the exact answer on all routes or the same typed refusal; digests unchanged | S1 grid lookup and canonical-scan skip |
| R4 | Cylinder/cone angular decisions: A2-A5, D1/D2 (cylinder), E2, E6, E9, E4 (cylinder faces) | the review-s1c seam case decided by T6 (identical on both routes) or refused on both; zero flips on the rings8/pipes6 cases; `_angle_on_sweep` no longer in any decision; full suite unchanged | S1 scalar `Cylinder.local_uv`; barge R250 openings |
| R5 | Support certificates and area bounds: E1, E3, E5 (remaining), D3 quadrature bound | sampled residual gone from `_support_roots`; area decisions three-valued with a rigorous bound; the review-s1c NaN-area case refuses on both routes | S1 area shortcut; M1 chart qualification |
| R6 | Deferred families: F1-F4 | as R4 for the quadric/cone corpora; `_merge_touching` removed, tangential junctions refuse | native N3 arrangement |

Each slice: byte preservation except where a decision was provably wrong before (then the
exact oracle proof and the changed digest are reported case by case for owner approval),
full suite, harness zero-flip report, review by Codex, completion checklist per AGENTS.md.
R0-R3 are the critical path for both the barge (B1/B2 run through planes, lines, arcs and
the new plane x NURBS curves) and for admitting S1's blocked changes.

## 1. Surface capability interface

### 1.1 Module and types

New module `src/anygeometry/surface_capabilities.py`: an adapter registry, not a base class.
Existing dataclasses are untouched (equality, hashing, repr, records, checksums unchanged).

```python
class SurfaceKind(StrEnum):
    PLANE = "plane"; CYLINDER = "cylinder"; CONE = "cone"; EXTRUDED = "extruded"
    RULED = "ruled"; COONS = "coons"; NURBS = "nurbs"

@dataclass(frozen=True, slots=True)
class Enclosure:            # outward-rounded box proven to contain the exact surface sub-patch
    lower: tuple[float, float, float]; upper: tuple[float, float, float]

@dataclass(frozen=True, slots=True)
class JacobianBounds:       # certified over a sub-domain (1.4)
    sigma_max: float        # upper bound of the largest singular value of [S_u S_v]
    sigma_min: float        # lower bound of the smallest singular value (0.0 = no bound)

@dataclass(frozen=True, slots=True)
class ImplicitForm:         # F(x) = 0 on the surface; degree 1 (plane) in the plane x NURBS scope
    degree: int; coefficients: tuple[float, ...]; gradient_inf: float

@dataclass(frozen=True, slots=True)
class SurfaceCapabilities:
    kind: SurfaceKind
    definition: object; support_key: object
    domain: tuple[tuple[float, float], tuple[float, float]]
    periodic_u: float | None
    evaluate: Callable[[np.ndarray], np.ndarray]                               # (n,2)->(n,3)
    derivatives: Callable[[np.ndarray, int], tuple[np.ndarray, ...]]
    invert_local: Callable[[np.ndarray, np.ndarray | None, float], InversionResult]   # local only
    exact_uv: Callable[[tuple[float, float, float]], tuple[Fraction, Fraction]] | None  # T1, planes only (R1)
    enclosure: Callable[[UVBox], Enclosure]
    jacobian_bounds: Callable[[UVBox], JacobianBounds] | None
    implicit: ImplicitForm | None
    transformed: Callable[[np.ndarray], object]
    is_exact: bool
```

Rules: (1) callables take and return C-contiguous float64 arrays validated once; bodies live
in `reference/` (section 5) and are the differential-test oracle; the perturbation hooks of
R.4 live here. (2) `invert_local` is never a closest-point guarantee and never feeds a
decision (R.1 A6). (3) `exact_uv` is the only inversion a decision may use; it exists for
`Plane` (R1) and is `None` elsewhere: cylinders decide by T2/T6, NURBS faces never invert
world points for decisions (4.6). (4) `enclosure`, `jacobian_bounds` are proof objects;
`None` means the route refuses. (5) `implicit` exists for degree 1 in this programme
(quadrics deferred, 4L). (6) Derived data is memoised in a `WeakKeyDictionary` keyed by the
frozen definition, never as a definition field. `normal_cone`, `closest_certified`,
`preimages_certified` and `subpatch` of rev 3 are deferred with NURBS x NURBS (4L).

### 1.2 Mapping of existing types (no behaviour change)

| Type | enclosure | jacobian_bounds | implicit | notes |
|---|---|---|---|---|
| `Plane` | sub-rectangle corners | exact singular values of `[u_vector v_vector]` | degree 1 | `exact_uv` = rational Cramer (T1) |
| `Cylinder` | ring stationary points, `EllipticArc.bounds` with `nextafter` | `max/min(r*sweep, h)` exact | none in scope | angular decisions by T2/T6 |
| `Cone` | as cylinder with both radii | interval; `sigma_min = 0` at the apex row | none in scope | apex deferred (R6) |
| `ExtrudedSurface` | `bounds()` (336-341) | hodograph hull x vector | none in scope | degree-n Bezier extrusion is a bidegree (n, 1) NURBS (`nurbs_of_extruded`) |
| `RuledSurface` | hull of boundary samples (convex combination: sound) | None | None | `is_exact=False` |
| `CoonsSurface` | interval evaluation of the blend (box arithmetic; the sample hull alone is unsound, `surfaces.py:465-480`) | None | None | `is_exact=False` |

The bodies of `_evaluate_surface_many`/`_surface_derivatives_many` move verbatim into
`reference/analytic_surfaces.py`; a test holds a frozen copy and asserts bit-identical
output on 1e5 random UVs per type.

### 1.3 Closest point: local only in this programme

`closest_certified`/`preimages_certified` (rev 3) are deferred (4L). `project_to_face`
remains a local Newton for authoring and display with `AttachmentEvidence` never EXACT on
NURBS faces; attachments that must be exact are authored on curves (4.6).

### 1.4 Chart/world tolerance conversion (forward only, infinity norm)

Chart boxes and the positional enclosure `delta` (4.3) are in the **infinity norm**. The
forward rule is `tolerance_chart = tolerance_world / (sqrt(2) * sigma_max)`: a chart
displacement of infinity-norm `delta` has Euclidean length at most `sqrt(2) delta` and moves
the point by at most `sqrt(2) sigma_max delta` (rev-3 blocking 3). Whenever a *world*
enclosure of a chart box is needed it is obtained by direct interval evaluation of `S` over
the box (never by `sigma_max * delta`). No reverse rule exists (rev-2 major 5).

## 2. Intersection-curve contract (narrowed to plane x NURBS)

### 2.1 Definition and certificate (module `src/anygeometry/support_curves.py`)

```python
@dataclass(frozen=True, slots=True)
class EdgeRoot:                       # a box-edge event: an algebraic point, never a rounded coordinate
    patch: int                        # Bezier sub-patch index (u-span, v-span)
    box: tuple[float, float, float, float]   # the dyadic quad-tree box whose edge carries the root
    edge: int                         # 0: u = u_lo, 1: u = u_hi, 2: v = v_lo, 3: v = v_hi
    isolating: tuple[float, float]    # dyadic interval on the edge containing exactly one root of the restriction p
    # p itself is not stored: it is the exact Bernstein restriction of G to that edge, recomputed from
    # the plane and surface records in rational arithmetic (identical on every implementation).

@dataclass(frozen=True, slots=True)
class CertifiedPiece:                 # one piece = one in-box arc of one regular box
    t_lo: float; t_hi: float          # curve parameter range (stored doubles, t_lo < t_hi)
    patch: int; box: tuple[float, float, float, float]
    slicing: int                      # 0: s = u, the piece gives v(s); 1: s = v, the piece gives u(s)
    direction: int                    # +1: t increases with s; -1: t decreases with s
    start: int; end: int              # indices into CertifiedBranch.roots: the two edge roots this arc joins
    s_ext: tuple[float, float]        # certification interval S_ext (polynomial extension) containing the arc's s-range
    x_ext: tuple[float, float]        # certification interval X_ext for the dependent coordinate
    kappa: float                      # contraction factor of the parameter-wide interval Newton on (X_ext, S_ext), < 1

@dataclass(frozen=True, slots=True)
class CertifiedBranch:
    algorithm: str                    # "plane_nurbs_v1" only in this programme
    tolerance: float
    roots: tuple[EdgeRoot, ...]       # every edge root of the path, in path order (roots[0], roots[-1] on the patch boundary)
    pieces: tuple[CertifiedPiece, ...]   # pieces[i] joins roots[i] to roots[i + 1]; t-contiguous, covering [0, 1]
    boxes_examined: int; subdivisions: int; complete: bool = True

@dataclass(frozen=True, slots=True)
class PlaneNurbsCurve:                # the SupportIntersectionCurve of this programme
    plane: Plane                      # the implicit side (exact stored doubles n, d derived by T1)
    surface: NurbsSurface             # the parametric side (support_key identity)
    branch: CertifiedBranch
    chart: tuple[float, float]        # (start, sweep) in s
    transform: tuple = IDENTITY
    witnesses: tuple = field(default=(), metadata={"definition": False})   # Approximant, 2.2
```

**Parameterisation with root-defined limits (rev-4 major 3, rev-5 blocking 2).** Piece
`i` runs from `roots[i]` to `roots[i + 1]`. Its native `s`-limits are **algebraic
numbers defined by those roots**: the `s`-coordinate of an edge root is dyadic when the
root lies on an edge perpendicular to `s` (then `s` is the box's `u_lo`/`u_hi` or
`v_lo`/`v_hi`) and otherwise it is the root of the exact edge restriction `p` in the stored
isolating interval (e.g. `G = u^2 + u + v^2 + v - 1/2`: the endpoints `(0, α)`, `(α, 0)`
with `α = (√3 - 1)/2` are the roots of `p(v) = v^2 + v - 1/2` on the edge `u = 0` and of
`p(u)` on `v = 0`, each stored as `EdgeRoot` with a dyadic isolating interval; no rounded
`α` exists in the record). The map `t -> s` on piece `i` is
`s(t) = s_i + (t - t_lo) / (t_hi - t_lo) * (s_{i+1} - s_i)` with `s_i`, `s_{i+1}` the exact
root values: a function defined in exact terms and *evaluated* as a canonical enclosure
(the root enclosure is refined by exact bisection of `p` on the isolating interval to the
level the consumer needs). Hence `s(t_lo) = s_i` and `s(t_hi) = s_{i+1}` **by definition**
and the mapped endpoints equal the certified edge roots; no equality proof between a
rounded endpoint and a root is needed, and validation checks only that `pieces[i].start/end`
index the right roots and that consecutive pieces share their root. The stored `t_lo/t_hi`
are arbitrary strictly increasing doubles (chosen as rounded cumulative fractions of the
enclosure midpoints' `s`-lengths; the map is defined by the record, not recomputed). A
change of slicing coordinate, of sub-patch (knot line) or of `direction` is an internal
piece boundary and never ends the curve; the curve ends at `roots[0]` and `roots[-1]`
(patch-boundary roots) or where an event of 4.2 step 4 splits it (`subcurve` in `t`).
`chart = (start, sweep)` of 2.1 is an affine map in `t` as for the exact families.

**What a piece certifies.** With slicing coordinate `s` and the dependent coordinate `x`,
the slice system is univariate: `g(x; s) = G(u, v)` with `G = n . S_h - d W` the Bernstein
polynomial of the sub-patch (4.1), which as a polynomial is defined on all of `R^2`. The
piece is certified when the parameter-wide interval Newton operator
`K(X, S) = m - Y g(m; S) + (1 - Y g_x(X; S)) (X - m)`, `m = mid(X)`, `Y = 1 / mid(g_x(X; S))`,
maps `X = X_ext` strictly into its interior for the whole `S = S_ext`; this proves for every
`s` in `S_ext` existence and uniqueness of `x(s)` in `X_ext` and continuity in `s`;
`kappa = |1 - Y g_x(X_ext; S_ext)|_sup < 1` is stored. **Certification boxes versus the real box.** Strict inclusion cannot hold on a box
clipped at its own boundary (`g = x - s`, `X = S = [0, 1]` gives `K = [0, 1]`), so `X_ext`
and `S_ext` are the polynomial extension beyond the quad-tree box (dyadic margins; the
NURBS is not evaluated there, only the polynomial piece is) such that the arc's `s`-range
and dependent range are interior. The extended box certifies **only** existence,
uniqueness and continuity of the graph `x(s)` on `S_ext`; it never decides which roots an
arc joins (rev-5 blocking 1): that is the **clipping contract** of 4.2 step 3, decided on
the real box between consecutive edge events. The positional
enclosure for a point `x` at parameter `s` is `delta(x, s) = |Y g(x; s)| / (1 - kappa)`
(Krawczyk a-posteriori bound, rev-2 finding 4) and is the only membership/position
certificate.

**Joins.** Pieces `i`, `i + 1` meet at `t_j = pieces[i].t_hi`. Same slicing coordinate and
same sub-patch: the join is certified iff `X_i ∩ X_{i+1}` is non-empty and
`K(X_i ∩ X_{i+1}, {s_j})` maps it strictly into itself (one solution in the intersection;
uniqueness in each piece forces equality: rev-2 finding 2). Different slicing coordinates
or different sub-patches (knot line): the **common system** is the univariate restriction
`p` of `G` to the shared box edge (an exact Bernstein polynomial of the stored doubles;
adjacent Bezier sub-patches restrict to the *same* exact polynomial on a shared knot line
by C0 continuity of the B-spline, and both interval versions enclose it). The join is
certified iff the two pieces' edge intervals `I_i`, `I_{i+1}` intersect, `p'` is certified
non-zero on `I_i ∪ I_{i+1}` (connected, so `p` is monotone there and has at most one root)
and each side certifies one root of `p` in its interval: then both roots are the same root
(rev-3 blocking 2 for this scope). Disjoint edge intervals or an undecided `p'` refuse.

**Evaluation constrained to the certificate.** `evaluate(t)`: map through `chart` and the
piece map to `s`; Newton on `g(.; s)` from the piece midpoint with every iterate inside
`X_ext`; return `(uv, delta)`; accepted only when `delta <= tolerance_chart` (1.4), else
refuse (impossible for a valid certificate; a guard against corrupted records). World
point `S(uv)`; world enclosure = interval evaluation of `S` over `uv ± delta` (1.4).
`derivative` by implicit differentiation; `bounds(lower, upper)` = union of the
reconstructed world enclosures of the pieces meeting `[lower, upper]`;
`subcurve`/`transformed` edit chart/transform only. Hash/equality by definition fields
(`witnesses` excluded, `roots` and `pieces` included).

### 2.2 Witnesses with continuous error bounds

`Approximant(curve: NurbsCurve degree 3, max_error)`: per box `max_error_i =
diam(hull(E_i, A_i))` with `E_i` the reconstructed world enclosure and `A_i` the control
hull of the approximant over `I_i` after knot insertion at `s_lo, s_hi`; outward-rounded
maximum; recomputed on load, a stored value below the recomputed one refuses. Pcurve
witnesses are unnecessary in this scope: the box chain *is* the chart preimage.

### 2.3 Schema record (schema 7) and validation on load

```json
{"type": "plane_nurbs_intersection",
 "plane": {<plane record>}, "surface": {<nurbs_surface record>},
 "branch": {"algorithm": "plane_nurbs_v1", "tolerance": 1e-9,
            "roots": [[patch, [u_lo,u_hi,v_lo,v_hi], edge, [iso_lo, iso_hi]], ...],
            "pieces": [[t_lo, t_hi, patch, [u_lo,u_hi,v_lo,v_hi], slicing, direction, start, end,
                        [s_ext_lo, s_ext_hi], [x_ext_lo, x_ext_hi], kappa], ...],
            "boxes_examined": 412, "subdivisions": 9, "complete": true},
 "chart": [0.0, 1.0], "transform": [[...4 rows...]],
 "witnesses": [{"kind": "approximant", "curve": {<nurbs_curve record>}, "max_error": 5.2e-8}]}
```

- Validation on load: for every `EdgeRoot`, recompute the exact restriction `p` and verify
  exactly one root in the isolating interval (exact sign change plus `p'` of one sign on
  it, or Sturm); for every piece, re-run the parameter-wide interval Newton test on
  `(X_ext, S_ext)`, recompute `kappa` (stored `>=` recomputed), verify that its `start`/`end`
  roots lie on edges of its box and that the arc between them passes the **clipping
  contract** of 4.2 step 3 (no other edge root of that box strictly between them along the
  independent coordinate; certified inside at the midpoint); check `t`-contiguity, strict
  monotonicity of `t_lo/t_hi`, coverage of `[0, 1]`, that consecutive pieces share their
  root, every join (2.1), that `roots[0]`/`roots[-1]` lie on the patch boundary and no
  other root does; recompute witness bounds (2.2). Any failure refuses with the root,
  piece or join index. Box and isolating-interval coordinates are dyadic and round-trip
  exactly as JSON doubles; `t_lo/t_hi` are arbitrary stored doubles. World enclosures are
  never stored.
- Record names: `nurbs_surface`, `nurbs_curve`, `nurbs_pcurve`, `curve_on_surface`,
  `plane_nurbs_intersection`. `_NURBS_VERSION = 7` (owner decision 3); schema 6 is
  unchanged; `required_version(record)` walks nested records and returns the maximum;
  writer `"version": max(VERSION, *(...))` replacing line 734. Ordinary documents stay
  schema 5 byte-identical.
- Older readers are tested for real (`tests/compat/` with the pinned 0.4.5 wheel,
  `tools/compat_reader_check.py`, CI job not run without the owner): refusal with
  `GeometryError`.

### 2.4 Affine propagation

`transformed(M)`: chart data unchanged; reconstructed world enclosures -> image of the 8
corners, outward-rounded; world error bounds multiply by an upper bound of `||L||_2`
(outward); the certificate is invariant (the slice system composes with the affine map).
Non-invertible `M` is refused by `coerce_affine_transform`.

### 2.5 Existing exact families

Unchanged classes, records and evaluators; `support_curve_view` of rev 3 is deferred (4L).

### 2.6 Arrangement over the new curve: certified predicates, no approximate trims

- **Exact trim rule.** On a NURBS face a boundary path is a `CurveOnSurface` (its pcurve is
  the definition) or a `BranchPath(PlaneNurbsCurve)` whose `uv(t)` is the certified preimage
  and whose chart `bounds` is the union of the pieces' chart boxes `[s_lo, s_hi] x x_ext`. No witness is ever a trim.
- **Predicate module** `src/anygeometry/branch_predicates.py` extends the pair table of
  `_curve_junctions`; the required pairs are those of 4.5. Missing pairs still raise at 462.
- **Ordering** around a junction by certified tangent directions (implicit
  differentiation in interval arithmetic over the junction's isolating box); overlapping
  tangent intervals refuse.
- **Overlap and coincidence.** Two branch paths are the same curve only under branch
  identity (bit-equal `plane`, `surface`, `branch`, `transform`, exact chart relation); any
  other pair is separated by the junction predicate or refuses (`"possible shared arc"`).
- **Containment** on the NURBS chart: ray parity with crossings isolated on the univariate
  restrictions `G(u, v0)` / `G(u0, v)` (exact Bernstein in the ray coordinate) for branch
  paths and on the pcurve for `CurveOnSurface`; a ray through a junction or tangent -> next
  ray of a fixed sequence; 8 rays then refuse.
- `_support_roots(curve, support)`: `None` by construction for a `PlaneNurbsCurve` on its own
  plane or own surface; against another plane: the 1D exhaustive isolation of 4.4; against a
  cylinder/cone/other NURBS: refuse in this programme.

### 2.7 Plan/apply, attachments, charts

- `_pair_paths` -> dispatch table keyed by `(kind_a, kind_b)`; existing entries 1:1; new
  entry `(PLANE, NURBS)` returning `SupportIntersection`. **Reverse dispatch (rev-5 major
  6)**: the table is consulted for `(kind_a, kind_b)` and, failing that, for
  `(kind_b, kind_a)` with the operands swapped; the swapped result is remapped
  explicitly: `SupportIntersection.curves` are support-symmetric objects
  (`PlaneNurbsCurve` names its `plane` and `surface`, not first/second) and need no
  remap; `segments`/`points` are world-space and need no remap; the clip is made
  order-independent by clipping the curve against **both** domains to exact station
  intervals and intersecting the intervals (T0) instead of `clip(second, clip(first, .))`;
  for `query_intersection` results, `first_*`/`second_*` parameter fields and
  `first_subparent`/`second_subparent` are swapped. A test asserts `(NURBS, PLANE)` and
  `(PLANE, NURBS)` give identical `topology_digest`, traces and refusals on the corpus (T7
  at the dispatch level). Every other NURBS pair refuses in both orders.
- **Attachments on NURBS faces** (4.6): `_target_bounds` NURBS branch = union of the pieces' chart boxes `[s_lo, s_hi] x x_ext`
  (branch) or pcurve control hull (`CurveOnSurface`); `_BOUNDED_CURVE_TYPES` gains both
  types; a member axis on a NURBS face must be one of them with the face's support.
- **Physical area.** `world_area_loop` NURBS path: `-oint F(u, v) du`, `F = int_0^v |S_u x
  S_v| dw`, adaptive Gauss-Kronrod returning `(value, bound)` and decided three-valued (R.1
  D3); `material_area` is diagnostic.
- `TrimmedSurfaceChart` keeps its shape; `evaluate_trimmed_surface_chart` evaluates in one
  batch and gains `derivatives=True`.

## 3. NURBS data model (module `src/anygeometry/nurbs.py`)

### 3.1 Curves

```python
@dataclass(frozen=True, slots=True)
class NurbsCurve:
    degree: int                                        # 1..MAX_DEGREE (7, Q4)
    knots: tuple[float, ...]                           # n + degree + 1, non-decreasing, clamped
    controls: tuple[tuple[float, float, float], ...]   # n >= degree + 1, finite
    weights: tuple[float, ...] | None                  # None = polynomial; else > 0, max/min <= 1e6 (owner)
    domain: tuple[float, float] | None = None
    _cache: dict = field(default_factory=dict, init=False, repr=False, compare=False, hash=False,
                         metadata={"definition": False})
```

Engine parameter `t in [0, 1]` maps affinely onto `domain`; `subcurve` edits `domain`
(reversal allowed); `transformed` maps controls, keeps weights. Validation refuses
unclamped/periodic knots (Q5), zero-length spans (`<= 64 eps * range`), non-positive
weights or ratio `> 1e6`, knot magnitudes above `2**100` or spans below `2**-100` of the
range (Q12: the N1a round-3 extreme-range cases become validation refusals), inconsistent
lengths. Evaluation: de Boor in homogeneous form (Piegl & Tiller A3.1/A4.1), derivatives to
order 2, Bezier extraction (A5.6) as the derived cache, `knot_insert` (A5.1) public.
`Nurbs2dCurve` is the same class with 2-tuples (pcurves). `Spline` is unchanged.

### 3.2 Surfaces and curves on surfaces

```python
@dataclass(frozen=True, slots=True)
class NurbsSurface:
    degree_u: int; degree_v: int
    knots_u: tuple[float, ...]; knots_v: tuple[float, ...]
    controls: tuple[tuple[tuple[float, float, float], ...], ...]     # rows along u
    weights: tuple[tuple[float, ...], ...] | None
    u_range: tuple[float, float] | None = None; v_range: tuple[float, float] | None = None
    _cache: dict = field(..., metadata={"definition": False})
    def support_key(self): return (degree_u, degree_v, knots_u, knots_v, controls, weights)

@dataclass(frozen=True, slots=True)
class CurveOnSurface:
    """Exact composition S(q(t)): the pcurve is the definition."""
    support: object; pcurve: Nurbs2dCurve
    also_on: tuple[tuple[object, Nurbs2dCurve], ...] = ()
    transform: tuple = IDENTITY
```

Capabilities of `NurbsSurface`: `evaluate`, `derivatives` (order 2), `invert_local`,
`enclosure` (control-net box of the Bezier sub-patch, outward), `jacobian_bounds` (interval
Gram matrix of the hodograph hulls), `implicit=None`, `exact_uv=None`, `is_exact=True`.
`iso_curve(surface, "u"|"v", value)` extracts the exact row/column NURBS as a
`CurveOnSurface` (its pcurve is the exact line `u = value`).

### 3.3 Faces: trim loops and the two-support rule

A NURBS face is `Face(loop, corners, holes, surface=NurbsSurface)` whose every loop edge is a
`CurveOnSurface` on that support (or with it in `also_on`) or a `PlaneNurbsCurve` having it
as the surface; `add_face_from_loop` refuses anything else. `also_on` requires a proof: both
pcurves are iso-curves and the boundary control rows, weights and knot vector are bit-equal
after the chart map (Q9); otherwise refuse. `face_trim_loops_uv` samples pcurves for display
only; `trim_loops_uv_certified(face, tol)` returns polylines with a world chord bound from
`JacobianBounds.sigma_max` for ANYmesher.

### 3.4 Identity: control points are owned arrays, not vertices

Interior control points are not on the surface; vertex semantics would be meaningless and
would let edits silently alter certified geometry; weights and knots have no vertex
analogue; frozen tuples give hashable definitions. Edge end vertices remain topology and
must coincide with curve ends (`add_curve` 4610-4613, extended: the coincidence test is T1
exact on the stored vertex and the exact rational evaluation of the curve end). Descendants
keep the parent surface object (same `support_key`, same chart), differing only by loops.

### 3.5 Serialization, transforms, tolerances, exact circles

Records (strict field sets) at schema 7: `nurbs_curve`, `nurbs_pcurve`, `nurbs_surface`,
`curve_on_surface`, `plane_nurbs_intersection`. Doubles through the canonical encoder
(exact round trip, NaN/inf rejected). `operations._transform_surface` gains `NurbsSurface`.
`surface_residual` (1e-8) stays the on-surface acceptance. `nurbs_circular_arc` (Piegl &
Tiller A7.1) is exact in the stored doubles but a different exact object from a `Cylinder`;
analytic types stay canonical; a `NurbsSurface` is never reclassified numerically.

## 4. Plane x NURBS: certified design (A4 scope)

Inputs: a `Plane` (stored `origin`, `u_vector`, `v_vector`; implicit form `F(x) = n . x - d`
with `n = u_vector x v_vector` and `d = n . origin` computed exactly by T1 and stored as the
rational pair's correctly rounded doubles plus their exact derivation from the plane record
— the oracle recomputes them from the plane, never from the rounded `n, d`), a
`NurbsSurface` of bidegree `(p, q)` with weights `W` (ratio `<= 1e6`), a world tolerance
`tol`. Output: `SupportIntersection(curves=(PlaneNurbsCurve, ...), segments=(), points=(),
coincident)` or a typed refusal.

### 4.1 Substitution

Per Bezier sub-patch (knot insertion, exact in rationals for the oracle; interval for the
engine): `G(u, v) = n . S_h(u, v) - d W(u, v)` with `S_h` the homogeneous control net, so `G`
is a polynomial of bidegree `(p, q)` in Bernstein form with coefficients
`g_ij = n . P_ij w_ij - d w_ij` (rational in stored doubles; interval enclosures of a few
ulps in the engine). The zero set of `G` on the sub-patch is exactly the plane section
(`W > 0`). The plane being degree 1, no degree growth occurs (plane x cubic -> bidegree
(3, 3)).

### 4.2 Certified topology (transversal only)

1. **Quad-tree** on each sub-patch with exact dyadic midpoints (de Casteljau in intervals).
   Box classes: *empty* (all `g_ij` intervals of one strict sign); *regular* (`G_v`'s
   coefficient intervals of one strict sign -> inside the box the zero set is the **graph**
   `v = f(u)` of a continuous function over the `u`-interval, *not* monotone in general
   (rev-4 blocking 1: `G = v - u^2` is a graph over `u`), or symmetrically `G_u` of one
   strict sign -> graph `u = f(v)`); else subdivide to `IntersectionQualificationPolicy.
   max_subdivision_depth` / `max_boxes_per_pair`. In a regular box every zero-set component
   is a closed sub-interval of the graph and its two ends lie on the box boundary (where `f`
   meets the box's dependent-coordinate range or `u` meets the box's `u`-range); no
   component is closed *inside one box*. Nothing is claimed about closure across boxes.
2. **Singular boxes at the depth limit refuse** (`PlaneNurbsNotCertified(uv box)`): isolated
   contact points, loops without a regular decomposition, branch crossings and tangencies
   are outside the scope (owner decision 2). The certified-empty case of rev 3 (verified
   critical point + definite Hessian + certified critical-value sign) is kept as the only
   classification of a singular box.
3. **Box-edge roots and chaining.** `G` restricted to any box edge is a univariate
   Bernstein polynomial (exact restriction); its roots on every edge of every regular box
   are isolated exhaustively (Bernstein sign variations on interval coefficients; a
   coefficient interval containing 0 at the finest level is an undecided predicate for the
   oracle, 5.3; a root of multiplicity `> 1` on an edge — the curve **tangent to a box edge
   or to the patch boundary** — is never isolated with a certified sign change and
   therefore refuses, which is how boundary tangencies are handled in the accounting). The
   set `E` of all box-edge roots (an edge shared by two boxes carries one root object, the
   common-system identification of 2.1; a root at a box corner is identified exactly:
   `G(corner)` is a Bernstein coefficient and its zero is decided exactly, so a corner root
   counts once) is the vertex set of a graph whose edges are the within-box arcs of regular
   boxes, determined by the **clipping contract** (rev-5 blocking 1; the certification box
   of 2.1 plays no part in pairing): in a regular box with graph `x = f(s)` over the
   independent coordinate `s`, project every edge root of that box to `s` (roots on the two
   edges perpendicular to `s` have the dyadic `s` of that edge; roots on the two edges
   parallel to `s` have their isolating interval in `s`); refine isolating intervals by
   exact bisection of the restrictions until all projections are pairwise disjoint (two
   roots that cannot be separated are a double event: refuse); sort them. Between two
   consecutive events `s_k < s_{k+1}` the graph cannot leave or enter the real box (leaving
   through the parallel edges would create a root of the edge restriction strictly between
   the events, which exhaustive isolation would have found; `f` is single-valued and
   continuous on `S_ext`), so the arc over `[s_k, s_{k+1}]` is either entirely inside or
   entirely outside the real box; which one is decided **once per gap** by the canonical
   enclosure of `f` at the midpoint `(s_k + s_{k+1})/2` (constrained Newton with `delta`,
   2.1) strictly inside or strictly outside the box's dependent range (T3; undecided ->
   refuse). The in-box arcs are exactly the "inside" gaps; they are disjoint and cover every
   point of the zero set in the real box (exhaustive: every point of the zero set lies over
   some gap or at an event). The rev-5 counterexample `G = v - (u - 1/2)^2 + 1/16` on
   `[0, 1]^2` has events `u = 0, 1/4, 3/4, 1`; the gaps `[0, 1/4]` and `[3/4, 1]` are inside
   (`f = 3/16`-ish above `v = 0`) and `[1/4, 3/4]` is outside (`f(1/2) = -1/16 < 0`), giving
   the correct pairings left-bottom and bottom-right; the extended graph's existence over
   `[1/4, 3/4]` is irrelevant. Every root in `E` has degree
   exactly 2 (one arc on each side) except roots on the **patch boundary**, which have
   degree 1; any other degree refuses. The graph is then a disjoint union of **paths**
   (degree-1 ends on the patch boundary) and **cycles**. **Closed chains are detected
   explicitly and refuse** (`PlaneNurbsClosedSection`): the rev-4 counterexample
   `S = (u, v, (u-1/2)^2 + (v-1/2)^2 - r^2)` cut by `z = 0` yields a cycle of regular boxes
   with no patch-boundary root and is refused, not dropped. **Accounting**: the number of
   patch-boundary roots equals twice the number of paths; every root in `E` belongs to
   exactly one path; a leftover or a cycle refuses. Each path becomes one `PlaneNurbsCurve`
   whose pieces are its arcs in order, with `slicing`/`direction` per arc and the joins of
   2.1 at every shared edge root; its `roots[0]` and `roots[-1]` are the two patch-boundary roots.
4. **Events** split curves (`subcurve` in `t`): junctions with the other operand's boundary
   edges (2.6/4.5). Slicing-coordinate and sub-patch changes are internal piece boundaries
   (2.1), not events. The chart `(start, sweep)` is in `t`.
5. **Coincidence**: `G` identically zero is decided exactly (every coefficient enclosure is
   `[0, 0]` after exact oracle confirmation) -> `coincident=True`; nearly zero -> refuse
   (tangential contact of the plane with the shell).

### 4.3 The certificate, positional enclosures and norms

As in 2.1: box certificate = parameter-wide 1D interval Newton with contraction `kappa`;
positional enclosure `delta = |Y g| / (1 - kappa)` in the infinity norm of the chart;
world enclosure by direct interval evaluation of `S` over `uv ± delta`; the forward rule of
1.4 with the `sqrt(2)` factor. No residual, no sample, no `sigma_max * delta` conversion is
ever used for a decision (rev-3 blocking 3).

### 4.4 Curve-level predicates on a `PlaneNurbsCurve`

- `roots_on(curve, plane2)` (a second plane: frame or stiffener plane crossing the deck-edge
  curve, or the deck plane crossing a frame curve): `h(s) = n2 . S(x(s), s) - d2` along the
  branch; for each certified box the interval `h(E_i)` with `E_i` the reconstructed
  enclosure; strictly one sign -> no root in `I_i`; else bisect (re-certify halves) until
  exclusion or **isolation**: `h'` (implicit differentiation, interval over the box)
  excludes 0 and `h` has opposite **certified** signs at the ends (interval `h` over the
  positional enclosure `x_end ± delta` excludes 0); then exactly one root, polished by Newton
  inside the box, returned with its own `delta`. Unresolved at depth -> refuse (tangency).
- `parameters_of(point, tol)` (rev-3 blocking 1, rev-4 major 7): the solution set is the
  **sublevel set** `{s : D(s) <= tol^2}` of `D(s) = |c(s) - P|^2`, and its connected
  components are the stations. They are isolated exactly, not merged heuristically:
  (i) *exclusion*: an interval with `dist_lower(P, E_I) > tol` (interval distance to the
  reconstructed enclosure, outward) contains no solution; (ii) *level roots*: the roots of
  `D(s) - tol^2` on the remaining intervals are isolated exhaustively by the 1D scheme of
  `roots_on` (`D'` interval excludes 0 on the isolating interval and `D - tol^2` has
  certified opposite signs at its ends, from interval `D` over `x_end ± delta`); an
  interval where `D - tol^2` can neither be excluded (interval `D` strictly above `tol^2`)
  nor certified strictly below `tol^2` nor isolated to a simple root at the budget refuses
  (`D` tangent to the level `tol^2`: the point sits exactly at tolerance distance);
  (iii) *components*: between consecutive level roots (and the chart ends) `D - tol^2` has
  a certified constant sign; the intervals with sign `<=` are the components, each reported
  **once**. Two visits of the tolerance ball separated by an excursion above `tol^2` are two
  components because the excursion contains two level roots; a shallow excursion that
  cannot be separated from the level refuses in (ii) (the rev-4 double-visit case is
  refused, never merged). (iv) *station of a component*: the unique certified root of `D'`
  with `D'' > 0` inside the component (the closest point), polished by Newton inside the
  piece and returned with `delta`; a component without a unique certified minimum (several
  minima, or `D'` undecided) refuses; a component of world length `> 2 tol` refuses (the
  curve runs along the point). The rev-3 counterexample (`c(s) = (s, s, 0)`, `P = (0, 1, 0)`,
  `tol = 5/8`) has `D > tol^2` everywhere and is excluded in (i)/(ii): no station.
  Existing behaviour at `branch_curves.py:575-577` (reject an out-of-tolerance polished
  point) is preserved by construction: nothing is reported without being inside a certified
  component.
- `point_parameters`/`plane_roots`/`curve_junctions` dispatch to these; `_support_roots`
  per 2.6.

### 4.5 Junctions needed by the barge (pair table of `branch_predicates.py`)

| Pair | System | Method |
|---|---|---|
| `BranchPath(P1 ∩ B)` x `BranchPath(P2 ∩ B)` on the shell `B` (frame plane x stiffener plane) | `G1(u, v) = G2(u, v) = 0` on the intersection of the two box chains | exhaustive 2D subdivision on intersected `uv` boxes with interval exclusion; each root isolated by 2D Krawczyk (existence + uniqueness per box); unresolved at budget -> refuse; equal planes -> branch identity (2.6) |
| `BranchPath` x `CurveOnSurface` iso-curve `u = u0` (shell patch boundary, stem iso-curve) | `G(u0, v) = 0` | exact univariate Bernstein isolation in `v` (T1/oracle), then the branch parameter from the box containing `(u0, v*)` |
| `BranchPath` x `LinePath` on the deck/bulkhead plane (deck-edge curve x deck line edges; frame curve x bulkhead lines) | second plane through the line (exact from stored endpoints and the face normal): `roots_on` 1D isolation of 4.4; the line parameter from the root point by T1 (B1) | 1D |
| `BranchPath` x `EllipticArc` on a plane face (frame curve x a rounded opening's arc) | ellipse implicit (exact rational quadratic form of the *supporting* ellipse) substituted along the branch: 1D isolation on the branch as 4.4 with `h = Q(S(x(s), s))`; **then arc-domain acceptance** for every candidate (rev-4 major 8, as `arrangement_geometry.py:357-361` does today): the candidate's frame coordinates `(c, s)` are an enclosure (exact Cramer of the world enclosure `S(uv ± delta)` in the ellipse frame) compared by `compare_exact_rays` against the canonical enclosures of the arc's start and end rays (B2, T3); outside -> dropped, inside -> junction with the arc parameter from the same enclosure, undecided -> refuse | 1D + arc membership |
| `BranchPath` x `BranchPath` on a plane face (two shells cutting the same bulkhead) | out of scope (one shell per side in the driving case; the stem is an `also_on` iso-curve); refuse | — |
| touching / tangential junctions | any system whose isolation fails at depth | refuse |

Ordering around junctions and containment per 2.6.

### 4.6 Attachment remapping and members on the curved shell

- A longitudinal continuing onto the bow shell is a member whose axis edge is a
  `PlaneNurbsCurve` (stiffener plane x shell), added with `add_curve`; its end vertices are
  exact (3.4). The member's `MEMBER_ON_FACE` attachment to the shell face is captured by
  `capture_face_attachments` as an axis snapshot (`attachment_remapping.py:34-48`); remap
  after imprinting a frame uses `_support_roots(axis, shell) is None` by construction,
  `_clip_intervals` with the junction predicates of 4.5 and `_target_bounds` = union of
  the pieces' chart boxes of the clipped sub-chain (2.7).
- Point attachments on NURBS faces carry their chart point `(u, v)` as authored (the
  `project_to_face` result at authoring time, local evidence); remapping decides descendant
  membership by the certified containment of 2.6 on that chart point and **never inverts
  the world point** (rev-3 major 4 is thereby outside the scope: no world -> chart preimage
  question is asked on NURBS faces). An attachment authored by world point only refuses on a
  NURBS face (`"attachment on a NURBS face needs a chart point"`).
- Plane faces (deck, bulkheads, frames) keep today's route, made exact by R1-R3.

### 4.7 Area and charts

`world_area_loop` NURBS path with `(value, bound)` (2.7, R.1 D3); `arrange_material` on the
NURBS chart uses `area_loop` = chart area with the same three-valued rule; conservation
check with the bound. `trim_loops_uv_certified` and batch chart evaluation for ANYmesher
(section 7).

### 4.8 Oracle keys and independent checking

Exact predicates of this route and their identity keys (rev-3 major 5): `("bernstein_sign",
plane_id, surface_id, subpatch_index, dyadic_box, (i, j))`; `("edge_root_count", ...,
dyadic_box, edge_side, interval)`; `("coincident", plane_id, surface_id, subpatch_index)`;
`("newton_box", ..., dyadic_box, s_interval)`. A replay decision is looked up by the full
key, never by position (5.3). `tools/verify_certificate.py` (independent `fractions`/
`mpmath` implementation) checks validity (every box passes the interval Newton, `kappa` and
join tests), coverage/exclusion (an independent exhaustive subdivision proves every domain
box is excluded, covered by a branch chain or reported as refused) and connectivity (every
boundary root found by the checker is consumed by exactly one branch).

### 4.9 Fail-closed summary (plane x NURBS)

Budget exhausted; singular box not certified-empty; box-edge root undecided or multiple
(tangency to a box edge or to the patch boundary); a box-edge root of degree other than 2
(or 1 on the patch boundary); a closed chain (cycle); patch-boundary root count not twice
the path count; a piece whose arc is not interior to its certification box; two edge events of one box not separable by exact bisection; a gap whose midpoint enclosure is not strictly inside or outside the real box; a cell probe not certified inside its cycle after 8 halvings; hole owner not separable by area enclosures; mapped-face corners not separable;
coincidence nearly zero; `parameters_of` level root not simple, component without a unique
certified minimum; junction not isolable or tangent; ordering not
separable; possible shared arc; degenerate patch edges (Q5); witness bound below the
recomputed bound, failed box/join re-check on load; `delta` above the chart tolerance; sign
not certified over `x ± delta`; `parameters_of` region too long or interval undecided at
budget; NURBS-face edge without an exact chart curve; member axis on a NURBS face that is not
a chart curve; attachment on a NURBS face without a chart point; unknown `(kind_a, kind_b)`
or curve pair; oracle replay rounds exhausted.

### 4.10 End-to-end slices to B1/B2

- **B1-pre** (A3b): bow patches with exact iso-curve boundaries; collision bulkhead, deck,
  frames and stiffener planes authored, not imprinted; `strict_audit` clean.
- **B1** (A4a + A4b'): deck and bottom planes trimmed by the plane x NURBS deck-edge curves;
  stiffener and frame curves added on the shell with `add_curve`; `strict_audit` clean.
- **B2** (A4c' + A4d'): `plan/apply_intersections` connects deck, collision bulkhead,
  frames and stiffeners to the shell; the end-to-end test of 8 (A4e) passes with
  constructed face/edge/attachment counts; `topology_digest` stable under the perturbation
  harness.

## 4L. Later (deferred, not designed in detail)

NURBS x NURBS (rev-3 §4.2: normal-cone loop exclusion, both-boundary starts, start
accounting across different systems — rev-3 blocking 2 for that scope stays open),
quadric x NURBS and degree-<=2 extrusion x NURBS (`ImplicitForm` degree 2, bidegree growth
(2p, 2q)), the general `SupportIntersectionCurve` with `reference_side`/pcurve witnesses and
`support_curve_view` of the exact families, `closest_certified`/`preimages_certified` and
the preimage-isolation requirement (rev-3 major 4), `normal_cone` capability, `subpatch`,
NURBS x NURBS coincidence (no exact test; refuse), interior loops/isolated contacts/branch
crossings for plane x NURBS (singular boxes), two shells cutting one bulkhead. Each comes
with its own design revision and Codex review before implementation.

## 5. Native core (aligned with the N0 scaffold)

### 5.1 Layout

```
native/_core.cpp                 # the only translation unit: module init, exported functions, build_info()
native/include/common.hpp        # buffer views (Py_buffer), NativeError, status codes      [exists in N0]
native/include/expansion.hpp     # two-sum/two-product, double-double; the T1 *filter* with a-priori error bounds
native/include/interval.hpp      # outward-rounded intervals; validated sin/cos/atan2/sqrt enclosures (5.4)
native/include/vec3.hpp, bernstein.hpp, bspline.hpp, analytic_surfaces.hpp, nurbs_surface.hpp,
               newton1d.hpp, krawczyk.hpp, intersect_plane_nurbs.hpp
src/anygeometry/backend.py       # import-time selection, backend_info()                       [exists in N0]
src/anygeometry/reference/       # same-signature pure-Python kernels
src/anygeometry/exact_predicates.py   # T1-T4, T7, T8 (Python, Fraction); the exact oracle of 5.3
```

Sources stay at the top level (`native/`), never under `src/anygeometry/`; `pyproject.toml`
excludes `*.pyd`/`*.so`; `MANIFEST.in` ships `setup.py`, `native/_core.cpp`,
`native/include/*.hpp`. Header-only, no third-party code.

### 5.2 Boundary and selection

`Py_LIMITED_API 0x030B0000`, `py_limited_api=True`; buffer protocol; no NumPy C API;
outputs preallocated in Python. Every exported function has a `reference/` twin bound once
by `backend.py` (no per-call switching, no monkeypatching). Selection as implemented in N0
(`ANYGEOMETRY_DISABLE_NATIVE`, absent module -> Python with reason, present-but-failing ->
`ImportError`, `ANYGEOMETRY_REQUIRE_NATIVE`); `backend_info()`; CLI `anygeometry --backend`;
diagnostics counters `native_calls`, `python_kernel_calls`, `exact_oracle_decisions`,
`certified_decisions`, `undecided_refusals`. Documents never record the backend. Batch
kernels release the GIL around the C++ loop from N1 (Q10).

**Admission rule (owner decision 1).** A native kernel is admitted on a path only after that
path is R-certified (R.3) and the kernel satisfies the equivalence contract; until then
native kernels serve coordinates that feed no decision (evaluation batches for display,
meshing and witnesses) and the certified interval/Bernstein primitives whose decisions are
replayed exactly (5.3).

### 5.3 Exact decisions: the oracle-replay contract

The native core makes **no exact decision**. All sign/zero decisions are made on interval
enclosures; an interval containing 0 at the finest level is *undecided*. A native routine
returns `(result_or_partial, undecided)` with keys referring to the original stored-double
operands (4.8 keys: predicate kind, operand ids, dyadic box or interval, coefficient
multi-index or edge side). `backend.py` evaluates each undecided predicate exactly in
Python (`exact_predicates`, `Fraction` on the original operands) and replays with a
`decisions` table keyed by predicate identity, never by position; a replay meeting a
predicate without a key adds it to the undecided list; rounds are bounded (8) and then the
routine refuses with the remaining keys. Subdivision is deterministic (5.4); budgets are
cumulative and reported. A predicate the exact oracle cannot decide (a genuine tangency)
is a refusal with the box.

### 5.4 Determinism and verified arithmetic

Flags as in N0 (`/fp:precise /fp:contract-`; `-ffp-contract=off -fno-fast-math`), no
`-march=native`; import-time contraction and rounding-mode probes refusing the native
backend on failure. Transcendentals: no libm ulp claim; `interval.hpp` implements its own
enclosures (double-double Cody-Waite reduction with a rigorous error term, Taylor with
alternating remainder in interval arithmetic; `sqrt` correctly rounded then outward) and is
differential-tested against `cylinder_charts._Proof` and `mpmath`. Deterministic
refinement (subdivision order, dyadic midpoints, ray sequence, budgets) depends only on
inputs and `interval.hpp`, so box trees, certificates, counts and refusals are identical on
every platform. D1 bound for coordinates: as R.3 item 3.

### 5.5 Two implementations kept honest; byte preservation versus equivalence

`reference/*.py` are the only Python bodies; `tests/native/test_differential_*.py` call both
routes on seeded random + adversarial inputs: integer/boolean/topological outputs and
decision traces compared exactly, coordinates within `B`; a test enumerates `_core`
symbols and fails on any without a differential test; a CI job with
`ANYGEOMETRY_REQUIRE_NATIVE=1` forbids skipping. *Byte preservation* (Python refactoring
slices) vs *equivalence under R.3* (native backend, admitted speed-ups) are never
conflated: the latter is shown by `topology_digest`, the decision trace and the
coordinate-deviation report. Sanitizer job on Linux; `tools/fuzz_native.py` on request;
`tools/verify_certificate.py` per 4.8.

### 5.6 Build and CI

`setup.py` as in N0 (`AbiThreeWheel.get_tag` -> `cp311-abi3`); native build opt-in
(`ANYGEOMETRY_BUILD_NATIVE=1`) until the owner decides the first release that ships it
(D3); `ANYGEOMETRY_REQUIRE_NATIVE=1` turns compile failure into a build error. Three
install tests (disabled; compiler-absent; required). `publish.yml` wheel job with pinned
`cibuildwheel`, `cp311-*`, Linux x86_64/aarch64 (QEMU), musllinux, macOS arm64/x86_64,
`tools/release_wheel_smoke.py --require-native`, abi3 cross-version install job on
3.12-3.14; free-threaded 3.14t uses the Python route (Q7). `ci.yml` runs pytest with and
without the native build; same failures required. **No workflow runs without the owner
(owner decision 3).**

## 6. Driving case: curved-bow barge

Midship 0-15 m as `floating_barge_large`, collision bulkhead at X = 15, bow 15-20 m as two
bidegree-(3, 3) `NurbsSurface` patches sharing the stem iso-curve exactly (`also_on`),
deck and bottom planes trimmed by plane x NURBS curves, longitudinals as members along
plane x NURBS curves on the shell, one or two transverse web frames (plane x NURBS).
Loading (owner decision 3): hydrostatic pressure on the bow (ANYfem wet-surface load) with
the existing clamp at X = 0; the end moment at X = 15 is dropped. Q2 (a real hull lines plan)
stays open; the single patch per side is the v1 form. Milestones B1-pre/B1/B2 per 4.10; B3
needs M1; B4 needs F1.

## 7. Consumer handoffs

**ANYmesher (M1):** replace surface-class `isinstance` chains by
`capabilities_of(surface).kind`; add `NURBS` to the `AnalyticMetricChart` route (gate at
`_analytic_metric_chart.py:21`), family `"parametric_trimmed"`; seed the CDT from
`trim_loops_uv_certified` (bound kept in provenance); batch
`evaluate_trimmed_surface_chart(derivatives=True)` and `project_to_face_many(initial_uv=...)`
(local projection is sufficient for smoothing); replace the class reasoning in
`preparation.py:401-430` by a new owner query `edge_dihedral_angles(model, edge_id, samples)`.

**ANYfem (F1):** `floating_barge_bow()` with `NurbsSurface`, `iso_curve`, `add_curve`
(plane x NURBS curves from `query_intersection` on the shell and the stiffener/frame planes),
`add_face_from_loop(surface=NurbsSurface)`; stiffeners attached `MEMBER_ON_FACE` with the
axis being the chart curve (4.6); hydrostatic load per owner decision 3.

**ANYfileIO / ANYfileio-occt (A7):** STEP B-spline entities -> `NurbsSurface`/`NurbsCurve`
(clamp by knot insertion; refuse periodic/unclamped per Q5, degree > `MAX_DEGREE`, weight
ratio > 1e6); analytic STEP entities -> analytic ANYgeometry types; STEP `PCURVE` ->
`CurveOnSurface`.

## 8. Slices, acceptance criteria, independent references

Evidence rules: samples, grid/Newton searches, role swaps and area conservation are
*consistency* checks only; **completeness** is shown by exact-count adversarial corpora and
by `tools/verify_certificate.py`; **robustness** is shown by the perturbation harness (zero
flips, reversal agreement, oracle agreement). Every slice reports commands, pass/fail
counts, G0 byte preservation or R.3 comparison, timings with spread, review status and what
was not run (AGENTS.md completion checklist).

| Id | Slice | Acceptance (must be shown) | Independent reference | Deps |
|---|---|---|---|---|
| R0 | `exact_predicates.py`, decision trace, perturbation harness, adversarial corpus, `topology_digest` | R.5 | `planar_area`, `cylinder_angular_lifts`, `Fraction` oracle | G0 |
| R1 | planar chart + membership (A1, A6-plane, D1-D5-plane, E4/E5-plane) | R.5; F1 test stable | exact rational oracle; shapely retained as a witness-only cross-check | R0, A1a hooks |
| R2 | curve membership and stations (B1-B9, E7) | R.5 | `Fraction` oracle; `isolate_real_roots` intervals | R0 |
| R3 | vertex identity and duplicates (C1-C4) | R.5; review cases | exact squared distances | R0 |
| R4 | cylinder/cone angular decisions (A2-A5, D1/D2-cyl, E2, E6, E9, E4-cyl) | R.5; review s1c seam case | `_Proof` enclosures, `compare_exact_rays` | R0 |
| R5 | support certificates and area bounds (E1, E3, E5 rest, D3) | R.5 | `prove_polynomial_extrusion_support`; rigorous quadrature bound vs `mpmath` | R2, R4 |
| R6 | deferred families (F1-F4) | R.5 | `QuadricSupport` exact forms | R4 |
| A0 | NURBS definitions only (3.1-3.5), records at schema 7, recursive `required_version`, validation incl. weight ratio and Q12 ranges | round trip exact; ordinary documents byte-identical; 0.4.5 reader refusal shown; validation refusals | Piegl & Tiller Ex 2.3, 3.1, 4.1, 5.1; de Boor vs Bezier extraction | — |
| A1a | capability registry + adapters for the six types; bodies to `reference/analytic_surfaces.py`; perturbation hooks | byte preservation on G0; bit-identical to the frozen chain on 1e5 UVs/type; Coons enclosure contains 1e5 samples incl. the centre-height case | frozen chain | G0 |
| A1b | `support_curves.py`: `CertifiedBox/Branch`, 1D parameter-wide interval Newton, joins (same and different slicing), `delta`, `PlaneNurbsCurve` evaluation `(uv, delta)`, reconstructed enclosures, witness bounds, affine propagation | a `PlaneNurbsCurve` rebuilt for a rational cylinder patch cut by a plane evaluates within `1e-12 * extent` of the `EllipticArc` and `delta` bounds the deviation at 1e5 samples; the rev-2 jump counterexample and the `eps u` residual case refuse; `verify_certificate.py` validity | `EllipticArc`; independent checker | A0, A1a, N1c |
| A1c | `plane_nurbs_intersection` record (2.3) + validation on load; writer version generalisation | tampered boxes/witnesses refused with index; older-reader refusal; byte preservation | hand-edited documents | A1b |
| A1d | `_pair_paths` dispatch table; `_support_roots` generic branch (refusing unknown); call sites of 1.5 switched; `topology_digest` in G0 | byte preservation on G0 | — | A1a |
| A1e | `MaterialDomain` over capabilities; `BranchPath`; `_target_bounds` NURBS branch; `world_area_loop` NURBS path with bound; batch trimmed-chart evaluation + derivatives | byte preservation; batch chart evaluation bit-identical to per-point; NURBS area of a rational cylinder patch vs `r*sweep*h` `<= 1e-10` relative | analytic area | A0, A1d, R5 (area rule) |
| N0 | scaffold: probes, counters, CLI flag, three install tests, `release_wheel_smoke.py`, `publish.yml` job (not run) | local build + import; `cp311-abi3` tag loads on 3.11/3.14; probes refuse a contracted build | — | — |
| N1a | `bspline.hpp` + `reference/bspline.py` (exists on `claude/geom-n1a-bspline`): add the A0 validation ranges so the round-3 extreme cases are refused at the boundary | differential within 1e-14 relative on valid inputs; `>= 20x` on 1e5 points | Piegl & Tiller | N0, A0 |
| N1b | `analytic_surfaces.hpp`: evaluate/derivatives for Plane/Cylinder/Cone/Extruded (coordinates only; no `local_uv` decision path until R1/R4) | differential within `B`; `>= 20x` | closed forms | N0, A1a |
| N1c | `interval.hpp` (now); `bernstein.hpp` 1D isolation + `newton1d.hpp`/`krawczyk.hpp` + the T3 filter with the `W_max` table (after acceptance of section 4) | enclosures contain `mpmath`/`_Proof` values at 1e6 arguments; root counts equal the `Fraction` Sturm oracle on `analytic_roots` + Wilkinson/clustered cases; filter/canonical identical outcomes and refusals on 1e6 predicates | `mpmath`; `_Proof` | N0, (R0 for the filter) |
| A2a | `NurbsCurve` engine integration: evaluation, `add_curve` acceptance (exact end test, 3.4), `freeze_edge`, outward `bounds`, `transformed`, `nurbs_circular_arc` | transform commutes (1e-13); arc radius error `<= 4 eps`; arc length vs `2 pi r` `<= 1e-12` | `EllipticArc` | A0, N1a |
| A2b | exhaustive curve predicates for NURBS edges against lines/arcs/Beziers (`plane_roots`, `point_parameters`, `curve_junctions`) via Bernstein isolation + 1D Newton; `parameters_of` per 4.4 | exact-count corpus incl. delta-families correct or refused; the rev-3 `parameters_of` counterexample excluded; `verify_certificate.py` agrees | `Fraction` resultant oracle | A2a, N1c, R2 |
| A3a | `NurbsSurface` capabilities (`evaluate`, `derivatives`, `invert_local`, `enclosure`, `jacobian_bounds`), `iso_curve`, `cylinder_to_nurbs` | enclosure/Jacobian bounds contain 1e5 sampled values; bilinear patch exact | rational cylinder patch | A2a, N1b |
| A3b | NURBS faces: 3.3 rules, `MaterialDomain` with `CurveOnSurface` boundaries, `arrange_material` on NURBS charts, `trim_loops_uv_certified`, `strict_audit`, `transform` | **B1-pre**; area via `world_area_loop` vs Gauss-Legendre on the rational cylinder patch `<= 1e-10`; applicable refusals of 4.9 exercised | rational cylinder patch | A3a, A1e |
| A3c | trimmed charts/evaluation API for NURBS faces | ANYmesher `AnalyticMetricChart` constructs on a NURBS face with the gate removed | — | A3b |
| A4a | plane x NURBS topology (4.1-4.3), `(PLANE, NURBS)` route, `roots_on`/`parameters_of` (4.4) | exact-count corpus (planes through constructed wavy patches, delta-families); `verify_certificate.py` validity + coverage/exclusion + connectivity; every refusal of 4.9 reachable here exercised; role swap and area conservation as consistency; harness zero-flip | independent checker; `EllipticArc` on the rational cylinder | A3b, N1c, A2b, A1b/A1c |
| A4b' | **B1**: deck/bottom trims and shell curves on the bow model | `strict_audit` clean; constructed counts | construction | A4a |
| A4c' | `branch_predicates.py` pairs of 4.5 (junctions, ordering, overlap, containment) | exact-count junction corpus; ordering refusals at constructed near-tangencies; `arrange_material` on two-cut NURBS faces gives constructed cell counts | independent checker | A4a |
| A4d' | attachment remap on NURBS faces (4.6), `material_area` | split-member corpus: UV ranges after `apply_intersections` equal the constructed values; world-point-only attachment refused | construction | A4c', A1e |
| A4e | **B2** end-to-end before M1/F1 | `strict_audit` clean; constructed face/edge/attachment counts; `topology_digest` stable under the harness and across backends; deviation report | construction | A4d', R1-R3 |
| A6 / A7 / M1 / F1 | as plan rev 2 (A5 moved to 4L) | | | A4e |
| S1 (re-admission) | the blocked S1 changes, path by path | R.3 on each R-certified path | harness | R1-R4 |

## 8a. What is safe to implement now

| Slice | Depends only on | Note |
|---|---|---|
| R0 | R.2, R.4 | pure additions, byte-preserving; the baseline report is evidence, not a behaviour change |
| A0 | 3.1-3.5, owner decision 3 (schema 7, weight ratio), Q4/Q5/Q12 | pure definitions; no engine route |
| A1a | 1.1-1.2 | adapter registry with the harness hooks; byte preservation |
| A1d (dispatch table, call-site switch, `topology_digest`) | 1.5 of rev 3, 5.5 | no new entries yet |
| N0, N1a (+ A0 validation ranges) | section 5 | generic primitives independent of the intersection design |
| N1c `interval.hpp` only | 5.4 | validated transcendental enclosures are design-independent; `bernstein.hpp`/`newton1d.hpp`/`krawczyk.hpp` wait: their contracts (edge restriction, root identification, `(X_ext, S_ext)` certification, the `W_max` filter policy of T3) are **provisional** until the owner accepts section 4 rev 5 (rev-4 note on A:959) |
| A2a | 3.1, N1a | NURBS edges, bounds, transforms, exact arcs |
| A3a | 1.1-1.4, 3.2 | capabilities without closest point |
| R1, R2, R3 | R.1-R.5 | sequential by owner approval of each digest change; R1 first |

Not yet: A1b/A1c (freeze 2.1/2.3 only after the owner accepts section 4), A1e `BranchPath`,
A2b (needs R2 semantics), A3b (needs A1e), A4a-e, R4-R6 (after R1-R3), any S1 re-admission,
any native kernel on a decision path. The rev-3 sentence "contracts now final" is withdrawn
(rev-3 coordinator note): contracts are final per section only when the R-gate and the
owner's acceptance say so.

## 9. Risks and open questions

Risks: refusals increase where today's engine snaps (exact-tolerance constructions, seam
straddles without topology): acceptable by owner decision 1, each such refusal is typed and
actionable; digest changes in R1-R4 where today's decision was rounding-arbitrary must be
approved case by case with the oracle proof; `Fraction` arithmetic is slow (T1 is a filter +
exact fallback in the native core; in Python the exact path runs only on the degenerate
band, plus the sampled per-call cost of the trace); two notions of "exact" (NURBS circle vs
`Cylinder`) produce near-coincident supports refused as tangential; plane x NURBS covers
transversal intersections only; certificate records enlarge documents; the validated
transcendental library is new code (N1c against `mpmath` and `_Proof`); the deterministic
first-fit vertex order is part of the contract and must not be changed by a speed-up.

Open questions for the owner: **Q2** bow form (single (3, 3) patch per side in v1).
**Q4** `MAX_DEGREE = 7`, interior multiplicity `<= degree`. **Q5** refuse periodic/unclamped
knots and degenerate patch edges in v1. **Q7** free-threaded CPython: Python route only.
**Q8** persist approximant witnesses or regenerate on load. **Q9** bit-equal `also_on` rule.
**Q10** GIL release from N1. **Q12** knot magnitude/span validation ranges (3.1). **Q13**
approval procedure for digest changes found by R1-R4 (per case, or per slice with a listed
table). **Q14** closed plane sections (cycles): refuse (rev 5) or accept later. **Q15**
margin rule for polynomial-extension certification boxes. **Q16** arc stations as planar
seam anchors (refused in rev 5). **Q17** thin-cell probe depth. **Q18** mapped-face corner
ties. Closed by owner decision 2026-10-13: Q1 (hydrostatic bow
load), Q3 (CI only by the owner), Q6 (schema 7), Q11 (transversal-only plane x NURBS).

## 10. Changes from rev 5 (rev-5 finding -> resolution in rev 6)

| Rev-5 finding | Resolution | Solved or narrowed |
|---|---|---|
| Blocking 1: uniqueness in the extended box does not certify within-box pairing | 4.2 step 3 **clipping contract**: edge events of a regular box are projected to the independent coordinate, separated by exact bisection, sorted; between consecutive events the arc is entirely inside or outside the real box (an exit would be an edge root between the events), decided once per gap by the canonical midpoint enclosure strictly inside/outside; in-box arcs = inside gaps, disjoint and exhaustive; the extended box certifies only existence/uniqueness/continuity and never pairing. The `G = v - (u - 1/2)^2 + 1/16` case is traced to the correct pairings. Load validation re-checks the contract. | solved (refuses on unseparable events or undecided gaps) |
| Blocking 2: record cannot represent exact (non-dyadic) piece endpoints | 2.1/2.3: `EdgeRoot` records (patch, box, edge, dyadic isolating interval; the restriction `p` is recomputed exactly) are the only endpoints; pieces reference roots by index; the `t -> s` map is defined on the exact root values and evaluated as canonical enclosures, so mapped endpoints equal the certified roots **by definition** (`α = (√3 - 1)/2` example traced); validation verifies one root per isolating interval exactly. Schema-7 `plane_nurbs_intersection` record updated (`roots`, `pieces` with `start/end`). The A0 `nurbs_*` records are unchanged. | solved |
| Major 3: inventory/provenance incomplete (generated curves, child-plane selection, corner selection) | A7 generated-curve definitions (exact plane x plane line from the two plane records; seams and support segments likewise); A8 exact parallelogram test; A9 corner ranking as an ordering decision through T3 with refusal on ties; all assigned to R1. R.4 audit extended to ordering/selection sites and construction provenance, with its limits stated; the inventory's completeness claim is **withdrawn** (R.1). | solved for the named sites; completeness claim withdrawn |
| Major 4: D5 probe relation to the cell | D5 **probe-in-cycle**: the probe is accepted only when certified inside the cycle's own loop (D1 ray parity on the cycle), so its material test answers for the right region; `h` halved up to 8 times, then refuse; hole ownership by the containing positive cycle with a strictly separated smallest area enclosure, else refuse. No strip upper bound needed. | solved by narrowing (refusal when the cell is thinner than the smallest probe) |
| Major 5: R.3 not preserved across model writes | R.3 split: **persisted decision operands are canonical** (T0: canonical rounding of exact/canonical positions and definitions), hence byte-identical documents per operation across backends; `B` applies to non-persisted coordinates only; multi-operation byte-identity tests added to R1 acceptance; values with no canonical definition (`closest_uv` attachment parameters) are authored inputs, recorded in the trace. | solved by a narrower, explicit claim |
| Major 6: reverse dispatch and result remapping | 2.7: table consulted in both orders with explicit remapping (support-symmetric curve objects, world-space segments/points, order-independent clipping by station-interval intersection, swapped `first_*`/`second_*` fields in query results); symmetric test on the corpus. | solved |

Owner questions raised by rev 6 (no machinery added for them): **Q17** whether a cell
thinner than `32 tol / 2^8` should be refused (rev 6) or the probe search deepened;
**Q18** whether `_detect_corners` ties (equal turning angles, e.g. a regular octagon) should
refuse (rev 6) or keep today's index tie-break as an *authored* convention.

## 10a. Changes from rev 4 (rev-4 finding -> resolution in rev 5)

| Rev-4 finding | Resolution |
|---|---|
| Blocking 1: topology argument excludes closed components wrongly | 4.2: "regular" means *graph*, not monotone graph; nothing is claimed across boxes. Chaining is a graph on **all box-edge roots** with certified within-box arcs; degree-1 roots are patch-boundary roots, every other root must have degree 2; the components are paths and cycles; **cycles are detected and refuse** (`PlaneNurbsClosedSection`, the paraboloid-cap case traced); the accounting counts patch-boundary roots = 2 x paths. Boundary tangencies: a multiple edge root is never certified (no sign change) and refuses; corner roots are identified exactly and counted once. |
| Blocking 2: exact arithmetic on rounded intermediates is not a cure | R.2 T0: provenance by **stations** `(definition, exact parameter or isolating interval)` through evaluation, splitting (`subcurve` keeps the parent map) and merging; positions are derived (rational or canonical enclosure); B11, C1/C2, D6-D8 reassigned; the contract boundary (stored doubles become the definition for the next operation) is stated. R.4 states that the harness only falsifies; R-certification is shown by the structural audit. |
| Major 3: branch parameter and schema-7 record inconsistent | 2.1/2.3: `CertifiedPiece` with `t_lo/t_hi`, `patch`, `slicing`, `direction`, native `s`-range and the certification box `(X_ext, S_ext)`; coordinate changes are internal piece boundaries, never new curves; events split by `subcurve` in `t`; endpoints are `EndpointCertificate`s (patch-edge roots); strict inclusion on clipped boxes replaced by polynomial-extension boxes with the chain part interior; validation on load checks all of it. |
| Major 4: T3 does not guarantee identical refusals | R.2 T2/T3: one canonical procedure (Fraction intervals, fixed level schedule `0..L_max`) *defines* the outcome incl. REFUSE; every other route is a filter that may decide only when separated by more than `W_max(q)` and otherwise defers to the canonical procedure; hence identical outcomes and identical refusals. The `Fraction` oracle adjudicates rational predicates only; non-rational predicates are adjudicated by the canonical procedure, cross-checked by an independent `mpmath` interval implementation (R.4 item 3). |
| Major 5: inventory incomplete on the barge path | D6 (seam-construction decisions at 606-628), D7 (seam geometry 630-652), D8 (split collection 654-667), B10 (Bezier membership acceptance 259-265), B11 (evaluated endpoints feeding vertex identity); 42 entries; completeness delegated to the mechanical audit of R.4 (R0 acceptance). |
| Major 6a: `planar_area` rounds | D3: `AreaEnclosure(lo, hi)`: rational for line/Bezier loops, rational interval for ellipse terms via canonical `sincos` enclosures (Decimal series withdrawn), rigorous quadrature bound otherwise; comparisons on the enclosure, float only for reporting. |
| Major 6b: exact polygon of vertex UVs misrepresents arcs/Beziers | E4: certified curved-loop containment on the exact chart images (affine chart: rational lines/Beziers, exact ellipse arcs) with exact univariate crossings and arc-domain membership; shapely withdrawn from decisions. |
| Major 7: `parameters_of` distinct-solution completeness | 4.4: components of the distance sublevel set isolated through the level roots of `D - tol^2` (certified simple roots or refusal at the level), one station per component at its unique certified minimum; the double-visit case yields two components or refuses, never one. |
| Major 8: ellipse junction row omits arc-domain acceptance | 4.5 row: arc membership of every candidate by exact-ray comparison of the enclosed frame coordinates against the canonical start/end rays; B2 extended with `arrangement_geometry.py:357-361`. |
| A:959 N1c contracts "fixed" | 8/8a: only `interval.hpp` is design-independent; the Bernstein/Newton/Krawczyk primitives and the filter policy are provisional until section 4 rev 5 is accepted. |

Not resolved by new machinery, recorded for the owner: **Q14** whether closed plane
sections of the shell (cycles) should be accepted as closed `PlaneNurbsCurve`s in a later
slice or remain refused; **Q15** the margin rule for polynomial-extension certification
boxes (fixed dyadic fraction of the box, or adaptive), which affects refusal rates at patch
boundaries; **Q16** whether arc stations whose position is not rational may anchor planar
seams (D7 refuses them in rev 5).

## 10b. Changes from rev 3 (rev-3 finding -> resolution in rev 4)

| Rev-3 finding | Resolution |
|---|---|
| Blocking 1: `parameters_of` can report a parameter with no point within tolerance | 4.4: intervals are *excluded* (certified lower bound `> tol`) or *accepted* (certified upper bound at a witness `<= tol`), nothing else is reported; distinct solutions = accepted runs separated by excluded intervals; long regions and undecided intervals refuse; the `(s, s, 0)` counterexample is traced to exclusion. |
| Blocking 2: start merging reuses the same-system join proof across different systems | For plane x NURBS (2.1 joins): a join between boxes with different slicing coordinates or across a knot line is certified in the **common univariate system** `p = G` on the shared edge (one exact polynomial, enclosed by both sides), with `p'` non-zero on the union of the edge intervals and one root certified per side -> same root. NURBS x NURBS start accounting across different boundary systems: **deferred with NURBS x NURBS** (4L). |
| Blocking 3: incompatible norms in the positional certificate | 1.4/4.3: chart boxes and `delta` are infinity-norm quantities; the forward rule carries the `sqrt(2)` factor; world enclosures come from direct interval evaluation of `S` over `uv ± delta`; `sigma_max * delta` is withdrawn. |
| Major 4: one surviving preimage region is not one preimage | Out of scope by construction: NURBS faces never invert world points for decisions (1.1 rule 3, 4.6: attachments carry chart points, members are chart curves; world-point-only attachments refuse). `preimages_certified` and preimage isolation **deferred with NURBS x NURBS** (4L). |
| Major 5: oracle key does not fully specify predicate identity | 4.8/5.3: keys carry the coefficient multi-index `(i, j)`, the edge side and interval for univariate restrictions, the sub-patch index and the `s` interval for Newton boxes; replay by full key only. |
| Coordinator: "contracts now final" in 8a too strong | 8a rewritten: sentence withdrawn; A1b/A1c/A2b/A3b wait for owner acceptance of section 4 and for R2; the R-gate (R.3) is the new admission criterion for every speed-up and native kernel. |
| (new, owner) F1 rounding-sensitive decisions | Section R: inventory (37 entries in rev 4, 42 in rev 5), toolkit T1-T8, equivalence contract, perturbation harness, slices R0-R6 ordered by barge impact and S1 unblocking. |
| (new, owner) scope narrowing | Section 2 narrowed to `PlaneNurbsCurve`; rev-3 §4.2 (NURBS x NURBS), quadric x NURBS, general `SupportIntersectionCurve`, `support_curve_view`, `normal_cone`, `closest_certified`, `subpatch` moved to 4L. |
| (new, owner) weight ratio, schema 7, bow load, CI | 3.1 validation; 2.3/3.5 `_NURBS_VERSION = 7`; section 6 loading; 5.6 no workflow runs without the owner. N1a round-3 extreme-range findings become A0 validation refusals (Q12). |

Claims narrowed in rev 4: no decision anywhere in the programme may be made by comparing a
rounded intermediate with a threshold (T3 replaces every such comparison on R-certified
paths); only plane x NURBS is certified; shells never answer world -> chart questions for
decisions; `_merge_touching` is withdrawn for every new curve and scheduled for removal in
R6.

## 10c. Changes from rev 2 (kept for traceability)

New 1 (boundary starts omit `∂B`) -> rev-3 4.2 both-boundary search and start accounting
(now 4L). New 2 (contiguity is not continuity; stored enclosures) -> join test with
intersection-box Krawczyk; world enclosures reconstructed, re-checked on load (kept, 2.1/2.3).
New 3 (overlap equivalence false) -> branch identity only; unresolved shared arcs refuse
(kept, 2.6). New 4 (residual is not a position certificate) -> positional enclosure `delta`
from the contraction factor (kept, 2.1/4.3). Major 5 (reverse chart-distance rule) ->
forward rule only (kept, 1.4). Partial 3 (`parameters_of`) -> rev-3 clusters, now replaced
by 4.4. Partial 13 / note on 11 -> `verify_certificate.py` three checks; replay keyed by
predicate identity (kept, 4.8/5.3/5.5). Coordinator -> section 8a.

## 10d. Changes from rev 1 (kept for traceability)

1. Loop test by strict positivity of `(n_A x n_B) . p` over hodograph hulls (now 4L).
2. Singular boxes: only the certified-empty case classified (kept, 4.2). 3. Exhaustive root
discovery on certified enclosures (kept, 4.4). 4. Certificate as a contiguous chain of
parameter-wide boxes, re-verified on load (kept, 2.1/2.3). 5. Pcurve witnesses never trims;
`BranchPath` with certified preimages; per-pair certified predicates (kept, 2.6). 6. Barge
integration: `_target_bounds`, `_BOUNDED_CURVE_TYPES`, chart-curve member axes, physical
area, A4e before M1/F1 (kept, 2.7/4.6). 7. `invert_local` vs certified closest point
(certified part now 4L); chart/world conversion via singular-value bounds (1.4).
8. Coons enclosure by interval blend; Bezier extrusions route parametrically (kept, 1.2).
9. Byte preservation vs equivalence separated; record names fixed; recursive
`required_version`; real older-reader test (kept, 2.3/5.5). 10. Own validated
transcendentals; contraction/rounding probes; deterministic refinement (kept, 5.4).
11. Oracle-replay contract (kept, 5.3). 12. abi3 wheels aligned with N0 (kept, 5.6).
13. Slices split, exact-count corpora, independent checker (kept, 8).
