# R0 structural decision-site audit

Slice R0, instrument 1 of the robust-predicate programme
(`docs/agent_plans/GENERALITY_ARCHITECTURE.md` on the generality branch, sections R.1, R.4,
R.5). The audit enumerates every rounding-sensitive decision site it can find in the
R-certified modules. It runs in **reporting mode** (R.5: "initially reporting, not
failing"): nothing in the engine was changed and no site is certified yet.

## How to run

```
# report (JSON + Markdown); exit code 0
python tools/decision_audit.py --json decision_sites.json --markdown decision_sites.md
# failing mode for a later R-slice: exit 1 while an uncertified site exists
python tools/decision_audit.py --fail

# tests: scanner self-checks, inventory anchors, report written to <basetemp>/decision_audit/
python -m pytest tests/test_decision_sites.py -p no:cacheprovider -o addopts="" --basetemp <dir>
# the same tests in failing mode
ANYGEOMETRY_DECISION_AUDIT_STRICT=1 python -m pytest tests/test_decision_sites.py ...
```

`check_certified(report, strict=True)` in the test module is the same switch as a function
argument. The scan uses the standard library `ast` only and takes about 3 s.

## What is scanned

Whole modules: `arrangement_geometry`, `material_arrangement`, `batch_intersections`,
`member_arrangements`, `attachment_remapping`, `trimmed_charts`, `analytic_supports`,
`intersections`, `predicates`, `curved_intersections`. R.4 names only the *planar parts* of
the last three; there is no mechanical boundary for "planar part", so the whole modules are
scanned and their curved and non-planar sites appear in the report (most of the unassigned
sites below are there).

Selected functions (the list is `SCOPE` in the tool):

- `model.py`: `_detect_corners` (A9), `face_local_uv`, `face_local_uv_many` (A1),
  `face_support_local_uv` (A6), `_arc_frame` and `_spline_points` (inputs of `freeze_edge`);
- `surfaces.py`: `Plane/Cylinder/Cone.local_uv`, `_angle_on_sweep`, `closest_uv` (A1, A2, A4, A6);
- `quadric_supports.py`: `_inside_patch(_many)`, `_merge`, `charts_support`,
  `_apex_contacts`, `_plane_through_apex`, `_coaxial_rings` (A2, A4, F2);
- `curves.py`: `arc_frame`, `_wrapped_angle` (section-10 open item, three-point arc frames);
- `tolerance.py`: `TolerancePolicy.effective_length/_coincidence/_healing/_area/_parameter`
  (section-10 open item, tolerances derived from extents).

## Site kinds

| kind | R.4 | subkinds |
|---|---|---|
| `tolerance` | (a) comparison of a float expression with a tolerance/eps-like quantity | `tolerance-compare`, `isclose-call` |
| `ordering` | (b) ordering and selection without thresholds | `select-by-float-key` (`sorted`, `.sort`, `argmin/argmax/argsort`, any `key=` into records), `min-max-value` (`min`/`max`/`np.clip` used as a value: clamps, bounding boxes), `round-floor` (`round`, `floor`, `ceil`, `int()` of a float, `%`, `//`), `float-equality` (`==`, `!=`, `in` on floats), `sign-branch` (comparison with 0, `np.sign`, truthiness of a float expression), `float-compare` (two float expressions, no tolerance) |
| `provenance` | (c) construction provenance | `derived-construction`: `LinePath`, `EllipticArc`, `BezierPath`, `Plane`, `Cylinder`, `Cone`, `Sphere`, `Torus`, `ExtrudedSurface`, the exact intersection curves, `ArrangementPoint`, `ArcFrame`, `add_point` whose arguments derive from `evaluate`/`derivative`/`local_uv`/`uv`/`closest_uv`/`tangent`/`left_probe`/`lstsq`/`solve`/`pinv`, `.witness(es)` or `.position` |
| `library` | added: R.4 says library decisions are not provable; their call sites are listed | `library-predicate`: shapely-style `covers`, `within`, `intersects`, `touches`, ... |

Heuristics (explicit in the tool's vocabularies):

- *Tolerance-like*: identifiers containing `tolerance` or a `tol`/`eps`/`epsilon`/`tiny`/
  `angular`/`touch`/`threshold`/`margin`/`slack`/`rtol`/`atol` word, float literals with
  `0 < |c| <= 1e-3`, module constants assigned such literals, and local names assigned a
  tolerance-shifted value (`lo - tolerance`, `np.nextafter(lo - margin, -inf)`). Call
  arguments are not searched (`f(x, tolerance=t) == y` is not a threshold). Comparisons
  whose only variables are tolerances (`tolerance <= 0`) are parameter checks, not sites.
- *Float-like*: float literals, `/` and `@`, calls of known float-returning functions
  (`evaluate`, `norm`, `atan2`, `local_uv`, `point_parameters`, ...), value-passing wrappers
  (`abs`, `np.asarray`, `min`, ...) of float arguments, known float attributes
  (`witness`, `center`, `start_angle`, `length`, ...), parameters with float names
  (`point`, `uv`, `t`, `angle`, ...), and names assigned from float-like values (a small
  fixed point over the assignments of each function; nested functions inherit).
- *Derived* (kind c): names assigned from an expression containing a derived source.
- *Certified*: a site whose lines (or the line above) carry `# coordinate-only: <reason>`
  (a reason is required), or whose statement calls a function listed in
  `CERTIFIED_EXACT_PREDICATES` imported from `exact_predicates`. Both are absent today, so
  every site is reported as uncertified.

## Inventory mapping

`INVENTORY` in the tool locates the R.1 entries by module, function (regular expression on
the qualified name; a nested function belongs to its enclosing function), an optional line
region inside the function delimited by two source anchors, and an optional source-text
pattern. This survives line drift: the architecture's line numbers are of 9ff7ed0 and the
code has moved since (material_arrangement by up to +54 lines after `_VertexLookup` was
added, batch_intersections by up to +15). A site may match several entries
(`inventory_ids`); `inventory_id` is the first, or `unassigned`.

The section-10 accepted open item "inventory sites" has no R.1 ids yet; the tool uses
provisional ids:

| id | site |
|---|---|
| X1 | three-point arc frames: `curves.arc_frame`, `_wrapped_angle`, `GeometryModel._arc_frame`, `arrangement_geometry.freeze_edge` |
| X2 | broad-phase pruning in batch_intersections: `_synchronize_boundary_events` (vectorized box test, lines 177-178), `_plan_intersections` (lines 507, 516, 530) |
| X3 | derived tolerance extents: `_plan_intersections` extents `norm(hi - lo)` (497) fed through `effective_length(max(...))` (sites at 505, 514) into `TolerancePolicy.effective_*` (tolerance.py 121-160) |
| X4 | positive-area overlap probe in `_coplanar_traces` (270-279) |

`model._detect_corners` (also in the open item) is A9.

## What it proves and what it cannot (R.4, instrument 1)

**What it proves** (once the test runs in failing mode and passes): on the scanned modules,
every enumerable decision site of kinds (a)-(c) is certified or justified; absence of flips
on those sites follows by construction.

**What it cannot prove** (rev-5 major 3): that the module list is complete, that a decision
hidden in a helper outside the list or in a library call (`np.linalg`, shapely) is caught,
or that an annotation's justification is right; those remain reviewed claims per slice.
The audit's module list is part of each R-slice's deliverable and is extended whenever a
review finds a site outside it (as `model.py:_detect_corners` was).

Limits of this implementation in addition (it is a heuristic scanner, not a type checker):

- **Missed sites** (false negatives). Floatness is inferred from names and calls; a float
  hidden behind an unknown helper, a record field or a container is not seen (for example
  `item[2] > 0` on the unpacked cycle records in `arrange_material`, a D3/D5 sign decision,
  is *not* reported). An evaluated point passed into a decision helper
  (`vertex(curve.evaluate(0.), sources)`, B11) is not itself a site; only the comparisons
  inside the helper are. Decisions inside `np.linalg` (`lstsq`, `solve`, `pinv`,
  `matrix_rank`) are not enumerated, and calls of the engine's own predicates
  (`domain.contains(...)`) are not sites (their bodies are scanned when in scope).
- **Spurious sites** (false positives). Name reuse makes integers look like floats
  (`orientation != direction` in `arrange_material`); `min`/`max` of floats used as bounding
  boxes or clamps are listed (`min-max-value`) although taking a min is exact; finiteness
  checks (`x - x == 0.`) are listed as float equality.
- **Mapping**: the inventory table matches by function and anchors. A rewrite that renames a
  function or an anchor line makes its sites `unassigned` (visible, never silent); the
  test asserts that every inventory id still has at least one site.
- The count is a baseline for triage, not a measure of risk: one inventory entry can have
  many sites, and one site can decide several things.

## Inventory entries not found as written

- `_qualified_face_plane` (R.4 lists it among the `model.py` construction routines) is in
  `intersections.py` (line 636) and is scanned with that module; `freeze_edge` is in
  `arrangement_geometry.py`.
- A1's anchor `surfaces.Plane.local_uv` contains no decision (an `lstsq` solve); its
  decisions are at the consumers, e.g. `intersections.clip_line_to_face`
  `polygon.covers`/`polygon.boundary.covers` (lines 160/169), found as `library` sites.
  A6 (`closest_uv`, Newton `lstsq`) and A7's `predicates.qualified_plane_plane`
  (`cross / sine`, `np.linalg.solve`) are coordinate computations; A7 is detected at the
  `LinePath` constructions in `batch_intersections._pair_paths`.
- B11 (`batch_intersections.py:959`, `material_arrangement.py:681-682` at 9ff7ed0) are
  evaluated tuples passed into `vertex()`; they are not enumerable sites (see limits). B11
  has one site, `ArrangementPoint(position, ...)` in `member_arrangements.plan_member_arrangements`.
- In main the material_arrangement vertex lookup (C1) is `_VertexLookup` (lines 569-615), a
  grid filter with `math.floor` cells in the style of C2; the scan finds its `floor` and its
  two norm comparisons.

## Baseline (main 99a4712)

Headline: 966 sites, all uncertified; 408 matched to an inventory entry, **558 unassigned**
(listed in full below). Of the unassigned sites, 471 are in the three query modules that R.4
scans only for their planar parts (intersections 263, curved_intersections 136, predicates
72). In the R.4 core modules 87 are unassigned; the groups that look like real inventory
candidates are:

- `batch_intersections._child_support` (17) and `_extruded_child_support` (15): the
  cylinder/cone/extruded analogue of A8 (child-support rebasing by `<= native_tolerance`
  row tests, `min(..., key=...)` corner choice, `Cylinder`/`Cone` built from evaluated rows);
- `batch_intersections._coplanar_traces` lines 262/264: `LinePath(component.witnesses[0],
  component.witnesses[-1])`, the coplanar analogue of A7 (generated curve from witnesses);
- `batch_intersections._pair_paths` 390 (seam extent) and 431 (point-contact dedupe
  `norm(point - old) <= tolerance`), `_apply_intersections_in_place` 835 (min tolerance per
  edge), 1115 (`residual > tolerance`), `.edge` 992 (`add_point` of an evaluated point);
- `arrangement_geometry._point_parameters` 250-254 (the `CylinderIntersectionCurve` branch:
  `atan2`, `ceil`/`floor` turn range, clamp, `norm <= tolerance`), `LinePath.subcurve` /
  `BezierPath.subcurve` (sub-curve ends evaluated or compared `== 0`/`== 1`; T0 stations);
- `member_arrangements.plan_member_arrangements` 184-185 (`a == 0.`, `a == 1.` on junction
  parameters), `_member_parameters` 116 (`sorted(values)`);
- `material_arrangement._integrate_loop` / `_adaptive_gauss` (quadrature convergence
  thresholds; part of D3's area bound in substance) and `_extruded_uv` 213 (`1e-12` full
  turn test);
- `attachment_remapping._target_bounds` clamps, `trimmed_charts` 224
  (`chart.world_tolerance != effective_length(length)`), `analytic_supports._events` 34,
  `_plane_branch_events` 130 (`denominator == 0`).

- sites: 966 (uncertified 966, certified 0)
- unassigned to an inventory entry: 558

| module | tolerance | ordering | provenance | library | total | unassigned |
|---|---:|---:|---:|---:|---:|---:|
| arrangement_geometry | 19 | 40 | 1 | 0 | 60 | 13 |
| material_arrangement | 36 | 35 | 2 | 0 | 73 | 10 |
| batch_intersections | 58 | 41 | 11 | 0 | 110 | 41 |
| member_arrangements | 10 | 14 | 1 | 0 | 25 | 8 |
| attachment_remapping | 2 | 9 | 0 | 0 | 11 | 7 |
| trimmed_charts | 2 | 4 | 1 | 0 | 7 | 6 |
| analytic_supports | 15 | 23 | 1 | 0 | 39 | 2 |
| intersections | 127 | 133 | 2 | 3 | 265 | 263 |
| predicates | 37 | 49 | 0 | 0 | 86 | 72 |
| curved_intersections | 99 | 106 | 0 | 0 | 205 | 136 |
| model | 0 | 6 | 0 | 0 | 6 | 0 |
| surfaces | 4 | 8 | 0 | 0 | 12 | 0 |
| quadric_supports | 40 | 15 | 0 | 0 | 55 | 0 |
| curves | 1 | 5 | 0 | 0 | 6 | 0 |
| tolerance | 0 | 6 | 0 | 0 | 6 | 0 |
| **total** | 450 | 494 | 19 | 3 | 966 | 558 |

Subkinds: library/library-predicate 3, ordering/float-compare 32, ordering/float-equality 48, ordering/min-max-value 280, ordering/round-floor 28, ordering/select-by-float-key 52, ordering/sign-branch 54, provenance/derived-construction 19, tolerance/isclose-call 3, tolerance/tolerance-compare 447

Sites per inventory id: A1 15, A2 26, A3 2, A4 9, A5 13, A6 23, A7 2, A8 2, A9 2, B1 5, B2 8, B3 3, B4 12, B5 15, B6 4, B7 3, B8 5, B9 3, B10 2, B11 1, C1 4, C2 6, C3 19, C4 3, D1 9, D2 8, D3 5, D4 2, D5 2, D6 5, D7 7, D8 2, E1 2, E2 12, E3 23, E4 6, E5 33, E6 33, E7 6, E8 2, E9 20, F1 3, F2 35, F3 1, F4 9, X1 6, X2 8, X3 8, X4 1

### Unassigned sites (558)

Candidates for the inventory (R.1). Per function:

| module | function | sites |
|---|---|---:|
| analytic_supports.py | _events | 1 |
| analytic_supports.py | _plane_branch_events | 1 |
| arrangement_geometry.py | _point_parameters | 6 |
| arrangement_geometry.py | BezierPath.subcurve | 3 |
| arrangement_geometry.py | LinePath.evaluate | 3 |
| arrangement_geometry.py | LinePath.subcurve | 1 |
| attachment_remapping.py | _target_bounds | 4 |
| attachment_remapping.py | remap_face_attachments | 2 |
| attachment_remapping.py | remap_face_attachments.descendant_domain | 1 |
| batch_intersections.py | _child_support | 17 |
| batch_intersections.py | _extruded_child_support | 15 |
| batch_intersections.py | _apply_intersections_in_place | 2 |
| batch_intersections.py | _coplanar_traces | 2 |
| batch_intersections.py | _domain_bounds | 2 |
| batch_intersections.py | _pair_paths | 2 |
| batch_intersections.py | _apply_intersections_in_place.edge | 1 |
| curved_intersections.py | _line_spline_intersection | 19 |
| curved_intersections.py | _supports_coincident | 19 |
| curved_intersections.py | qualified_curve_curve | 17 |
| curved_intersections.py | _segment_breaks | 13 |
| curved_intersections.py | qualified_face_face | 11 |
| curved_intersections.py | _coincident_face_region | 9 |
| curved_intersections.py | _material_intersection_loops | 8 |
| curved_intersections.py | _same_bezier_overlap | 7 |
| curved_intersections.py | _contour_crossing | 5 |
| curved_intersections.py | _refine_curve_surface_root | 5 |
| curved_intersections.py | _bezier_subcurve | 4 |
| curved_intersections.py | _point_on_polygon_boundary | 4 |
| curved_intersections.py | _canonical_polygon | 3 |
| curved_intersections.py | _angle_candidates | 2 |
| curved_intersections.py | _convex_polygon | 2 |
| curved_intersections.py | _curve_bounds | 2 |
| curved_intersections.py | _convex_intersection | 1 |
| curved_intersections.py | _convex_intersection.inside | 1 |
| curved_intersections.py | _line_intersection_2d | 1 |
| curved_intersections.py | _material_intersection_loops.key | 1 |
| curved_intersections.py | _result | 1 |
| curved_intersections.py | _trace_segments.key | 1 |
| intersections.py | _query_same_circle_arcs | 24 |
| intersections.py | _fragment_planar_face_by_segments | 19 |
| intersections.py | _transverse_cylinder_band | 19 |
| intersections.py | _query_member_material | 16 |
| intersections.py | _query_planar_faces | 12 |
| intersections.py | _trace_vertices_and_edges | 12 |
| intersections.py | _query_edge_face | 10 |
| intersections.py | plane_cylinder | 9 |
| intersections.py | _certified_curve_inside_convex_support | 8 |
| intersections.py | _face_imprint_application | 7 |
| intersections.py | intersect_surfaces | 7 |
| intersections.py | _canonical_member_point_components | 6 |
| intersections.py | _query_member_member | 6 |
| intersections.py | _same_cylinder_band | 6 |
| intersections.py | clip_line_to_face | 6 |
| intersections.py | _point_on_segment | 5 |
| intersections.py | line_cylinder | 5 |
| intersections.py | line_line | 5 |
| intersections.py | _canonicalize_coincident_boundary_edges | 4 |
| intersections.py | _component_boundary_edge | 4 |
| intersections.py | _member_sheet_application | 4 |
| intersections.py | _query_plane_cylinder_faces | 4 |
| intersections.py | _remap_face_attachments | 4 |
| intersections.py | numerical_surface_intersection | 4 |
| intersections.py | _angle_on_arc_sweep | 3 |
| intersections.py | _boundary_vertex | 3 |
| intersections.py | _canonical_member_point_components.equivalent | 3 |
| intersections.py | _range_from_component | 3 |
| intersections.py | _region_boundary_matches_loop | 3 |
| intersections.py | line_plane | 3 |
| intersections.py | plan_imprint | 3 |
| intersections.py | _face_length_scale | 2 |
| intersections.py | _fragment_planar_face_by_segments.snap_endpoint | 2 |
| intersections.py | _member_connection_application | 2 |
| intersections.py | _member_relation_matches | 2 |
| intersections.py | _parameter_is_endpoint | 2 |
| intersections.py | _point_segment_distance_value | 2 |
| intersections.py | _qualified_face_plane | 2 |
| intersections.py | _reuse_stale_face_plan | 2 |
| intersections.py | _shared_edges_for_components | 2 |
| intersections.py | intersect_faces | 2 |
| intersections.py | _angular_distance | 1 |
| intersections.py | _axial_plane_cylinder_segment | 1 |
| intersections.py | _canonical_member_point_components.witness_distance | 1 |
| intersections.py | _face_accepts_component_endpoints | 1 |
| intersections.py | _fragment_cylinder_transverse | 1 |
| intersections.py | _fragment_planar_face_by_segments.vertex | 1 |
| intersections.py | _fragment_plane_with_ring | 1 |
| intersections.py | _fragment_with_closed_chain | 1 |
| intersections.py | _imprint_components | 1 |
| intersections.py | _imprint_transverse_plane_cylinder | 1 |
| intersections.py | _member_sheet_component_kind | 1 |
| intersections.py | _overlay_components | 1 |
| intersections.py | _query_planar_faces.append_touch | 1 |
| intersections.py | _world_key | 1 |
| intersections.py | clip_line_to_face.collect | 1 |
| intersections.py | plane_plane | 1 |
| material_arrangement.py | _adaptive_gauss | 5 |
| material_arrangement.py | _integrate_loop | 2 |
| material_arrangement.py | MaterialDomain._extruded_uv | 1 |
| material_arrangement.py | _clip_intervals | 1 |
| material_arrangement.py | arrange_material | 1 |
| member_arrangements.py | plan_member_arrangements | 8 |
| predicates.py | qualified_segment_segment | 25 |
| predicates.py | qualified_line_cylinder | 17 |
| predicates.py | IntersectionCertificate.__post_init__ | 5 |
| predicates.py | _closest_endpoint_pair | 5 |
| predicates.py | IntersectionComponent.__post_init__ | 3 |
| predicates.py | qualified_plane_plane | 3 |
| predicates.py | _canonical_direction | 2 |
| predicates.py | _extent | 2 |
| predicates.py | qualified_line_line | 2 |
| predicates.py | qualified_line_plane | 2 |
| predicates.py | IntersectionResult.__post_init__ | 1 |
| predicates.py | _canonical_plane_basis | 1 |
| predicates.py | _component | 1 |
| predicates.py | _positive_integer | 1 |
| predicates.py | _stable_norm | 1 |
| predicates.py | _unit | 1 |
| trimmed_charts.py | validate_trimmed_surface_charts_binding | 3 |
| trimmed_charts.py | _qualified_chart | 2 |
| trimmed_charts.py | evaluate_trimmed_surface_chart | 1 |

All unassigned sites:

| file:line | function | kind/subkind | source |
|---|---|---|---|
| arrangement_geometry.py:53 | LinePath.evaluate | ordering/float-equality | `x0-x0 == 0.` |
| arrangement_geometry.py:53 | LinePath.evaluate | ordering/float-equality | `x1-x1 == 0.` |
| arrangement_geometry.py:53 | LinePath.evaluate | ordering/float-equality | `x2-x2 == 0.` |
| arrangement_geometry.py:62 | LinePath.subcurve | provenance/derived-construction | `LinePath(tuple(self.evaluate(lower)), tuple(self.evaluate(upper)))` |
| arrangement_geometry.py:101 | BezierPath.subcurve | ordering/float-compare | `upper < lower` |
| arrangement_geometry.py:103 | BezierPath.subcurve | ordering/float-equality | `upper == 1` |
| arrangement_geometry.py:104 | BezierPath.subcurve | ordering/float-equality | `lower == 0` |
| arrangement_geometry.py:250 | _point_parameters | ordering/select-by-float-key | `sorted((curve.start_angle, curve.start_angle+curve.sweep_angle))` |
| arrangement_geometry.py:252 | _point_parameters | ordering/round-floor | `math.ceil((lower-angle-tolerance)/math.tau)` |
| arrangement_geometry.py:252 | _point_parameters | ordering/round-floor | `math.floor((upper-angle+tolerance)/math.tau)` |
| arrangement_geometry.py:253 | _point_parameters | ordering/min-max-value | `min(1., max(0., _angle_parameter(curve, angle+turn*math.tau)))` |
| arrangement_geometry.py:253 | _point_parameters | ordering/min-max-value | `max(0., _angle_parameter(curve, angle+turn*math.tau))` |
| arrangement_geometry.py:254 | _point_parameters | tolerance/tolerance-compare | `np.linalg.norm(curve.evaluate(parameter)-point) <= tolerance` |
| material_arrangement.py:213 | MaterialDomain._extruded_uv | tolerance/tolerance-compare | `abs(abs(directrix.sweep_angle)-math.tau) <= 1e-12` |
| material_arrangement.py:458 | _integrate_loop | tolerance/tolerance-compare | `abs(value-previous) <= max(tolerance, 64*_EPS*magnitude)` |
| material_arrangement.py:458 | _integrate_loop | ordering/min-max-value | `max(tolerance, 64*_EPS*magnitude)` |
| material_arrangement.py:497 | _adaptive_gauss | tolerance/tolerance-compare | `error <= max(tolerance, 2.**-40*mass)` |
| material_arrangement.py:497 | _adaptive_gauss | ordering/min-max-value | `max(tolerance, 2.**-40*mass)` |
| material_arrangement.py:501 | _adaptive_gauss | tolerance/tolerance-compare | `b-a < 1e-13` |
| material_arrangement.py:509 | _adaptive_gauss | tolerance/tolerance-compare | `error > max(tolerance, 2.**-36*sum(item[6] for item in leaves))` |
| material_arrangement.py:509 | _adaptive_gauss | ordering/min-max-value | `max(tolerance, 2.**-36*sum(item[6] for item in leaves))` |
| material_arrangement.py:561 | _clip_intervals | ordering/float-compare | `a<=parameter<=b` |
| material_arrangement.py:839 | arrange_material | ordering/float-equality | `orientation != direction` |
| batch_intersections.py:226 | _domain_bounds | ordering/min-max-value | `np.min([box[0] for box in boxes], axis=0)` |
| batch_intersections.py:226 | _domain_bounds | ordering/min-max-value | `np.max([box[1] for box in boxes], axis=0)` |
| batch_intersections.py:262 | _coplanar_traces | provenance/derived-construction | `LinePath(component.witnesses[0], component.witnesses[-1])` |
| batch_intersections.py:264 | _coplanar_traces | provenance/derived-construction | `LinePath(component.witnesses[0], component.witnesses[-1])` |
| batch_intersections.py:390 | _pair_paths | ordering/min-max-value | `max(np.linalg.norm(lo-point)+np.linalg.norm(hi-point) for lo, hi in boxes)` |
| batch_intersections.py:431 | _pair_paths | tolerance/tolerance-compare | `np.linalg.norm(np.asarray(point)-old)<=tolerance` |
| batch_intersections.py:674 | _extruded_child_support | ordering/min-max-value | `np.min(rows,axis=0)` |
| batch_intersections.py:674 | _extruded_child_support | ordering/min-max-value | `np.max(rows,axis=0)` |
| batch_intersections.py:677 | _extruded_child_support | ordering/min-max-value | `min(float(np.max(np.linalg.norm(support.directrix.derivative(t),axis=1)))*abs(u1-u0), float(np.linalg.norm(support.vector))*abs(v1-v0))` |
| batch_intersections.py:677 | _extruded_child_support | ordering/min-max-value | `np.max(np.linalg.norm(support.directrix.derivative(t),axis=1))` |
| batch_intersections.py:680 | _extruded_child_support | tolerance/tolerance-compare | `abs(row[0]-lower[0])<=native_tolerance` |
| batch_intersections.py:680 | _extruded_child_support | tolerance/tolerance-compare | `abs(row[0]-upper[0])<=native_tolerance` |
| batch_intersections.py:681 | _extruded_child_support | tolerance/tolerance-compare | `abs(row[1]-lower[1])<=native_tolerance` |
| batch_intersections.py:681 | _extruded_child_support | tolerance/tolerance-compare | `abs(row[1]-upper[1])<=native_tolerance` |
| batch_intersections.py:686 | _extruded_child_support | tolerance/tolerance-compare | `abs(first[0]-last[0]) <= native_tolerance` |
| batch_intersections.py:686 | _extruded_child_support | tolerance/tolerance-compare | `abs(first[0]-middle[0]) <= native_tolerance` |
| batch_intersections.py:688 | _extruded_child_support | tolerance/tolerance-compare | `abs(first[1]-last[1]) <= native_tolerance` |
| batch_intersections.py:688 | _extruded_child_support | tolerance/tolerance-compare | `abs(first[1]-middle[1]) <= native_tolerance` |
| batch_intersections.py:689 | _extruded_child_support | tolerance/tolerance-compare | `upper-lower>native_tolerance` |
| batch_intersections.py:691 | _extruded_child_support | ordering/select-by-float-key | `min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance),key=lambda i:rows[i][0])` |
| batch_intersections.py:691 | _extruded_child_support | tolerance/tolerance-compare | `abs(row[1]-lower[1])<=native_tolerance` |
| batch_intersections.py:729 | _child_support | ordering/min-max-value | `np.min(rows,axis=0)` |
| batch_intersections.py:729 | _child_support | ordering/min-max-value | `np.max(rows,axis=0)` |
| batch_intersections.py:731 | _child_support | ordering/min-max-value | `min(widest*abs(support.sweep_angle),support.height)` |
| batch_intersections.py:732 | _child_support | tolerance/tolerance-compare | `abs(row[0]-lower[0])<=native_tolerance` |
| batch_intersections.py:732 | _child_support | tolerance/tolerance-compare | `abs(row[0]-upper[0])<=native_tolerance` |
| batch_intersections.py:733 | _child_support | tolerance/tolerance-compare | `abs(row[1]-lower[1])<=native_tolerance` |
| batch_intersections.py:733 | _child_support | tolerance/tolerance-compare | `abs(row[1]-upper[1])<=native_tolerance` |
| batch_intersections.py:738 | _child_support | tolerance/tolerance-compare | `abs(first[0]-last[0]) <= native_tolerance` |
| batch_intersections.py:740 | _child_support | tolerance/tolerance-compare | `abs(first[1]-last[1])<=native_tolerance` |
| batch_intersections.py:741 | _child_support | tolerance/tolerance-compare | `abs(float(np.asarray(path.curve.u_vector) @ support.axis))<=tolerance` |
| batch_intersections.py:742 | _child_support | tolerance/tolerance-compare | `abs(float(np.asarray(path.curve.v_vector) @ support.axis))<=tolerance` |
| batch_intersections.py:745 | _child_support | tolerance/tolerance-compare | `upper-lower>native_tolerance` |
| batch_intersections.py:746 | _child_support | ordering/select-by-float-key | `min(range(len(rows)),key=lambda i:(rows[i][1],rows[i][0]))` |
| batch_intersections.py:749 | _child_support | ordering/select-by-float-key | `min((i for i,row in enumerate(rows) if abs(row[1]-lower[1])<=native_tolerance), key=lambda i:rows[i][0])` |
| batch_intersections.py:749 | _child_support | tolerance/tolerance-compare | `abs(row[1]-lower[1])<=native_tolerance` |
| batch_intersections.py:758 | _child_support | provenance/derived-construction | `Cone(origin,support.axis,support.radial_direction,radius(lower[1]),radius(upper[1]), height,start,sweep)` |
| batch_intersections.py:761 | _child_support | provenance/derived-construction | `Cylinder(origin,support.axis,support.radial_direction,support.radius,height,start,sweep)` |
| batch_intersections.py:835 | _apply_intersections_in_place | ordering/min-max-value | `min((tolerances_by_face[face_id] for face_id in model.faces_using_edge(edge_id) if face_id in tolerances_by_face), default=next((axis.world_tolerance for axis in plan.axes if axis.edge_id == edge_id), model.tolerance.length))` |
| batch_intersections.py:992 | _apply_intersections_in_place.edge | provenance/derived-construction | `model.add_point(*point)` |
| batch_intersections.py:1115 | _apply_intersections_in_place | tolerance/tolerance-compare | `residual>tolerance` |
| member_arrangements.py:134 | plan_member_arrangements | ordering/min-max-value | `np.min([box[0] for box in boxes], axis=0)` |
| member_arrangements.py:135 | plan_member_arrangements | ordering/min-max-value | `np.max([box[1] for box in boxes], axis=0)` |
| member_arrangements.py:144 | plan_member_arrangements | ordering/min-max-value | `max(np.linalg.norm(hi1-lo1),np.linalg.norm(hi2-lo2))` |
| member_arrangements.py:174 | plan_member_arrangements | ordering/min-max-value | `max(tolerances[first_id],tolerances[second_id])` |
| member_arrangements.py:184 | plan_member_arrangements | ordering/float-equality | `a == 0.` |
| member_arrangements.py:184 | plan_member_arrangements | ordering/float-equality | `a == 1.` |
| member_arrangements.py:185 | plan_member_arrangements | ordering/float-equality | `b == 0.` |
| member_arrangements.py:185 | plan_member_arrangements | ordering/float-equality | `b == 1.` |
| attachment_remapping.py:77 | _target_bounds | ordering/min-max-value | `np.clip(lower[i],0.,1.)` |
| attachment_remapping.py:78 | _target_bounds | ordering/min-max-value | `np.clip(upper[i],0.,1.)` |
| attachment_remapping.py:86 | _target_bounds | ordering/min-max-value | `max(0.,axial-radius)` |
| attachment_remapping.py:86 | _target_bounds | ordering/min-max-value | `min(1.,axial+radius)` |
| attachment_remapping.py:170 | remap_face_attachments.descendant_domain | ordering/min-max-value | `max( (np.linalg.norm(hi-lo) for lo,hi in boxes),default=1.)` |
| attachment_remapping.py:193 | remap_face_attachments | ordering/min-max-value | `min(1.,max(0.,float(value)))` |
| attachment_remapping.py:193 | remap_face_attachments | ordering/min-max-value | `max(0.,float(value))` |
| trimmed_charts.py:117 | _qualified_chart | ordering/min-max-value | `np.max([box[1] for box in boxes],axis=0)` |
| trimmed_charts.py:118 | _qualified_chart | ordering/min-max-value | `np.min([box[0] for box in boxes],axis=0)` |
| trimmed_charts.py:222 | validate_trimmed_surface_charts_binding | ordering/min-max-value | `np.max([box[1] for box in boxes],axis=0)` |
| trimmed_charts.py:223 | validate_trimmed_surface_charts_binding | ordering/min-max-value | `np.min([box[0] for box in boxes],axis=0)` |
| trimmed_charts.py:224 | validate_trimmed_surface_charts_binding | tolerance/tolerance-compare | `chart.world_tolerance != model.tolerance.effective_length(length)` |
| trimmed_charts.py:277 | evaluate_trimmed_surface_chart | provenance/derived-construction | `LinePath(tuple(point),tuple(point))` |
| analytic_supports.py:34 | _events | ordering/min-max-value | `min(tolerance, 4*np.finfo(float).eps)` |
| analytic_supports.py:130 | _plane_branch_events | ordering/float-equality | `denominator == 0` |
| intersections.py:109 | clip_line_to_face | ordering/sign-branch | `direction_length <= 0.0` |
| intersections.py:191 | clip_line_to_face | ordering/min-max-value | `max(float(np.ptp(parameters)), 1.0)` |
| intersections.py:192 | clip_line_to_face | ordering/min-max-value | `np.min(parameters)` |
| intersections.py:193 | clip_line_to_face | ordering/min-max-value | `np.max(parameters)` |
| intersections.py:215 | clip_line_to_face.collect | ordering/float-compare | `line_parameters[1] < line_parameters[0]` |
| intersections.py:243 | clip_line_to_face | ordering/select-by-float-key | `line_components.sort( key=lambda item: item.first_parameter_range.lower # type: ignore[union-attr] )` |
| intersections.py:254 | clip_line_to_face | ordering/select-by-float-key | `point_components.sort(key=lambda item: item.first_parameter or ())` |
| intersections.py:284 | line_line | ordering/sign-branch | `float(np.linalg.norm(a)) <= 0.0` |
| intersections.py:284 | line_line | ordering/sign-branch | `float(np.linalg.norm(b)) <= 0.0` |
| intersections.py:286 | line_line | tolerance/tolerance-compare | `float(np.linalg.norm(np.cross(a, b))) <= tolerance * float(np.linalg.norm(a)) * float(np.linalg.norm(b))` |
| intersections.py:294 | line_line | ordering/min-max-value | `max( float(np.linalg.norm(q - p)), float(np.linalg.norm(first - p)), float(np.linalg.norm(second - q)), 1.0, )` |
| intersections.py:300 | line_line | tolerance/tolerance-compare | `float(np.linalg.norm(first-second)) <= tolerance*scale` |
| intersections.py:308 | line_plane | ordering/sign-branch | `float(np.linalg.norm(direction)) <= 0.0` |
| intersections.py:311 | line_plane | tolerance/tolerance-compare | `abs(denominator) <= tolerance * max(float(np.linalg.norm(direction)), 1.0)` |
| intersections.py:311 | line_plane | ordering/min-max-value | `max(float(np.linalg.norm(direction)), 1.0)` |
| intersections.py:322 | plane_plane | tolerance/tolerance-compare | `squared <= tolerance` |
| intersections.py:337 | line_cylinder | ordering/sign-branch | `float(np.linalg.norm(direction)) <= 0.0` |
| intersections.py:348 | line_cylinder | tolerance/tolerance-compare | `abs(a) <= tolerance` |
| intersections.py:351 | line_cylinder | tolerance/tolerance-compare | `discriminant < -tolerance` |
| intersections.py:353 | line_cylinder | tolerance/tolerance-compare | `abs(discriminant) <= tolerance` |
| intersections.py:359 | line_cylinder | tolerance/tolerance-compare | `-tolerance <= v <= 1.0+tolerance` |
| intersections.py:374 | plane_cylinder | tolerance/tolerance-compare | `abs(denominator) > tolerance` |
| intersections.py:376 | plane_cylinder | tolerance/tolerance-compare | `axial >= min(0.0, cylinder.height)-tolerance` |
| intersections.py:376 | plane_cylinder | ordering/min-max-value | `min(0.0, cylinder.height)` |
| intersections.py:376 | plane_cylinder | tolerance/tolerance-compare | `axial <= max(0.0, cylinder.height)+tolerance` |
| intersections.py:376 | plane_cylinder | ordering/min-max-value | `max(0.0, cylinder.height)` |
| intersections.py:392 | plane_cylinder | tolerance/tolerance-compare | `abs(float(signed[index])) <= tolerance` |
| intersections.py:394 | plane_cylinder | ordering/sign-branch | `signed[index]*signed[index+1] > 0.0` |
| intersections.py:402 | plane_cylinder | tolerance/tolerance-compare | `abs(value) <= tolerance` |
| intersections.py:404 | plane_cylinder | ordering/sign-branch | `lower_value * value <= 0.0` |
| intersections.py:427 | numerical_surface_intersection | ordering/min-max-value | `max(float(np.ptp(second_points, axis=0).max()), 1.0)` |
| intersections.py:432 | numerical_surface_intersection | tolerance/tolerance-compare | `float(distances.min()) <= tolerance*scale` |
| intersections.py:438 | numerical_surface_intersection | ordering/min-max-value | `min(float(np.linalg.norm(point-other)) for other in unique)` |
| intersections.py:438 | numerical_surface_intersection | tolerance/tolerance-compare | `min(float(np.linalg.norm(point-other)) for other in unique) > tolerance*scale` |
| intersections.py:471 | intersect_surfaces | ordering/min-max-value | `max(float((end-start) @ (end-start)), 1e-30)` |
| intersections.py:476 | intersect_surfaces | ordering/min-max-value | `min(parameters)` |
| intersections.py:476 | intersect_surfaces | ordering/min-max-value | `max(parameters)` |
| intersections.py:478 | intersect_surfaces | ordering/min-max-value | `max(item[0] for item in intersections)` |
| intersections.py:479 | intersect_surfaces | ordering/min-max-value | `min(item[1] for item in intersections)` |
| intersections.py:480 | intersect_surfaces | tolerance/tolerance-compare | `upper > lower+tolerance` |
| intersections.py:482 | intersect_surfaces | ordering/round-floor | `int(np.sqrt(samples))` |
| intersections.py:673 | _qualified_face_plane | ordering/float-compare | `float(np.max(residual)) > geometry.tolerance.effective_surface_residual( extent )` |
| intersections.py:673 | _qualified_face_plane | ordering/min-max-value | `np.max(residual)` |
| intersections.py:691 | _face_length_scale | ordering/min-max-value | `max(extent, surface.radius, abs(surface.height))` |
| intersections.py:692 | _face_length_scale | ordering/min-max-value | `max(extent, 1.0)` |
| intersections.py:707 | _boundary_vertex | tolerance/tolerance-compare | `float( np.linalg.norm( geometry.vertex_position(vertex_id) - point ) ) <= length_tolerance` |
| intersections.py:728 | _boundary_vertex | ordering/min-max-value | `min(max(best[2], parameter_tolerance), 1.0 - parameter_tolerance)` |
| intersections.py:728 | _boundary_vertex | ordering/min-max-value | `max(best[2], parameter_tolerance)` |
| intersections.py:923 | _remap_face_attachments | ordering/min-max-value | `min(u_values)` |
| intersections.py:923 | _remap_face_attachments | ordering/min-max-value | `max(u_values)` |
| intersections.py:924 | _remap_face_attachments | ordering/min-max-value | `min(v_values)` |
| intersections.py:924 | _remap_face_attachments | ordering/min-max-value | `max(v_values)` |
| intersections.py:1101 | _fragment_with_closed_chain | ordering/sign-branch | `_loop_signed_area(geometry, face_id, face.loop) * _loop_signed_area( geometry, face_id, ring ) < 0.0` |
| intersections.py:1145 | _point_segment_distance_value | tolerance/tolerance-compare | `length_squared <= np.finfo(float).tiny` |
| intersections.py:1147 | _point_segment_distance_value | ordering/min-max-value | `np.clip((point - start) @ direction / length_squared, 0.0, 1.0)` |
| intersections.py:1167 | _region_boundary_matches_loop | tolerance/tolerance-compare | `distance > tolerance` |
| intersections.py:1171 | _region_boundary_matches_loop | ordering/min-max-value | `min(float(np.linalg.norm(point - candidate)) for candidate in points)` |
| intersections.py:1171 | _region_boundary_matches_loop | tolerance/tolerance-compare | `min(float(np.linalg.norm(point - candidate)) for candidate in points) > tolerance` |
| intersections.py:1264 | _face_accepts_component_endpoints | tolerance/tolerance-compare | `best > tolerance` |
| intersections.py:1282 | _component_boundary_edge | tolerance/tolerance-compare | `float(np.linalg.norm(start - endpoints[0])) <= tolerance` |
| intersections.py:1283 | _component_boundary_edge | tolerance/tolerance-compare | `float(np.linalg.norm(end - endpoints[1])) <= tolerance` |
| intersections.py:1286 | _component_boundary_edge | tolerance/tolerance-compare | `float(np.linalg.norm(start - endpoints[1])) <= tolerance` |
| intersections.py:1287 | _component_boundary_edge | tolerance/tolerance-compare | `float(np.linalg.norm(end - endpoints[0])) <= tolerance` |
| intersections.py:1303 | _trace_vertices_and_edges | ordering/min-max-value | `max( geometry.tolerance.effective_curve_fit_residual( max(_face_length_scale(geometry, first_face), _face_length_scale(geometry, second_face)) ), geometry.tolerance.length, )` |
| intersections.py:1311 | _trace_vertices_and_edges | tolerance/tolerance-compare | `float(np.linalg.norm(point - compact[-1])) > tolerance` |
| intersections.py:1313 | _trace_vertices_and_edges | tolerance/tolerance-compare | `float(np.linalg.norm(compact[0] - compact[-1])) <= tolerance` |
| intersections.py:1325 | _trace_vertices_and_edges | ordering/sign-branch | `chord_length > 0.0` |
| intersections.py:1326 | _trace_vertices_and_edges | ordering/min-max-value | `max( float( np.linalg.norm( (point - compact[0]) - float((point - compact[0]) @ chord / (chord_length * chord_length)) * chord ) ) for point in compact )` |
| intersections.py:1335 | _trace_vertices_and_edges | tolerance/tolerance-compare | `line_error <= tolerance` |
| intersections.py:1355 | _trace_vertices_and_edges | ordering/min-max-value | `max( max( abs(float(np.linalg.norm(point - frame.center)) - frame.radius), abs(float((point - frame.center) @ frame.normal)), ) for point in compact )` |
| intersections.py:1356 | _trace_vertices_and_edges | ordering/min-max-value | `max( abs(float(np.linalg.norm(point - frame.center)) - frame.radius), abs(float((point - frame.center) @ frame.normal)), )` |
| intersections.py:1362 | _trace_vertices_and_edges | tolerance/tolerance-compare | `circle_error <= tolerance` |
| intersections.py:1383 | _trace_vertices_and_edges | ordering/min-max-value | `max( max( abs(float(np.linalg.norm(point - first_frame.center)) - first_frame.radius), abs(float((point - first_frame.center) @ first_frame.normal)), ) for point in compact )` |
| intersections.py:1384 | _trace_vertices_and_edges | ordering/min-max-value | `max( abs(float(np.linalg.norm(point - first_frame.center)) - first_frame.radius), abs(float((point - first_frame.center) @ first_frame.normal)), )` |
| intersections.py:1393 | _trace_vertices_and_edges | tolerance/tolerance-compare | `circle_error <= tolerance` |
| intersections.py:1494 | _imprint_components | tolerance/tolerance-compare | `float(np.linalg.norm(witnesses[0] - witnesses[-1])) <= tolerance` |
| intersections.py:1667 | _axial_plane_cylinder_segment | tolerance/tolerance-compare | `abs(float(plane.normal @ cylinder.axis)) > geometry.tolerance.angular` |
| intersections.py:1732 | _same_cylinder_band | ordering/min-max-value | `max(abs(reference.height), reference.radius, 1.0)` |
| intersections.py:1734 | _same_cylinder_band | tolerance/tolerance-compare | `float(np.linalg.norm(candidate.origin - reference.origin)) <= tolerance * scale` |
| intersections.py:1736 | _same_cylinder_band | tolerance/tolerance-compare | `float(np.linalg.norm(candidate.axis - reference.axis)) <= tolerance` |
| intersections.py:1737 | _same_cylinder_band | tolerance/tolerance-compare | `float( np.linalg.norm( candidate.radial_direction - reference.radial_direction ) ) <= tolerance` |
| intersections.py:1743 | _same_cylinder_band | tolerance/tolerance-compare | `abs(candidate.radius - reference.radius) <= tolerance * scale` |
| intersections.py:1744 | _same_cylinder_band | tolerance/tolerance-compare | `abs(candidate.height - reference.height) <= tolerance * scale` |
| intersections.py:1750 | _angular_distance | ordering/round-floor | `(first - second + np.pi) % period` |
| intersections.py:1785 | _transverse_cylinder_band | ordering/min-max-value | `max(reference.radius, abs(reference.height), 1.0)` |
| intersections.py:1792 | _transverse_cylinder_band | ordering/min-max-value | `max(abs(reference.height), reference.radius)` |
| intersections.py:1796 | _transverse_cylinder_band | tolerance/tolerance-compare | `abs(alignment - 1.0) > angular_tolerance` |
| intersections.py:1801 | _transverse_cylinder_band | ordering/sign-branch | `reference.height <= 0.0` |
| intersections.py:1811 | _transverse_cylinder_band | tolerance/tolerance-compare | `fraction < -parameter_tolerance` |
| intersections.py:1812 | _transverse_cylinder_band | tolerance/tolerance-compare | `fraction > 1.0 + parameter_tolerance` |
| intersections.py:1817 | _transverse_cylinder_band | tolerance/tolerance-compare | `abs(fraction) <= parameter_tolerance` |
| intersections.py:1819 | _transverse_cylinder_band | tolerance/tolerance-compare | `abs(fraction - 1.0) <= parameter_tolerance` |
| intersections.py:1821 | _transverse_cylinder_band | tolerance/tolerance-compare | `parameter_tolerance < fraction < 1.0 - parameter_tolerance` |
| intersections.py:1831 | _transverse_cylinder_band | ordering/min-max-value | `max(angular_tolerance, length_tolerance / scale)` |
| intersections.py:1839 | _transverse_cylinder_band | tolerance/tolerance-compare | `angular_tolerance < surface.sweep_angle < period - angular_tolerance` |
| intersections.py:1844 | _transverse_cylinder_band | ordering/round-floor | `surface.start_angle % period` |
| intersections.py:1845 | _transverse_cylinder_band | ordering/select-by-float-key | `candidates.sort(key=lambda item: (item[0], item[1]))` |
| intersections.py:1850 | _transverse_cylinder_band | tolerance/tolerance-compare | `abs(sum(item[2].sweep_angle for item in candidates) - period) > angular_tolerance` |
| intersections.py:1856 | _transverse_cylinder_band | tolerance/tolerance-compare | `_angular_distance(end_angle, following[0]) > angular_tolerance` |
| intersections.py:1862 | _transverse_cylinder_band | tolerance/tolerance-compare | `float(np.linalg.norm(current_end - following_start)) > length_tolerance` |
| intersections.py:1902 | _transverse_cylinder_band | tolerance/tolerance-compare | `abs(cross) > angular_tolerance` |
| intersections.py:1903 | _transverse_cylinder_band | ordering/sign-branch | `np.sign(cross)` |
| intersections.py:1916 | _transverse_cylinder_band | tolerance/tolerance-compare | `boundary_distance <= reference.radius + length_tolerance` |
| intersections.py:1996 | _fragment_cylinder_transverse | ordering/float-compare | `float(np.mean(axial_values)) > fraction` |
| intersections.py:2021 | _fragment_plane_with_ring | ordering/sign-branch | `_loop_signed_area(geometry, plane_face, outer_loop) * _loop_signed_area( geometry, plane_face, ring_loop ) < 0.0` |
| intersections.py:2116 | _imprint_transverse_plane_cylinder | provenance/derived-construction | `geometry.add_point(*surface.evaluate(0.5, fraction))` |
| intersections.py:2183 | intersect_faces | tolerance/tolerance-compare | `abs(abs(float(first_surface.normal @ second_surface.axis)) - 1.0) <= geometry.tolerance.angular` |
| intersections.py:2195 | intersect_faces | tolerance/tolerance-compare | `abs(abs(float(second_surface.normal @ first_surface.axis)) - 1.0) <= geometry.tolerance.angular` |
| intersections.py:2387 | _overlay_components | ordering/select-by-float-key | `components.sort(key=lambda item: (item.witnesses[0], len(item.witnesses)))` |
| intersections.py:2472 | _query_planar_faces | tolerance/tolerance-compare | `float(overlap.area) > area_tolerance` |
| intersections.py:2476 | _query_planar_faces | tolerance/tolerance-compare | `symmetric_area <= area_tolerance` |
| intersections.py:2478 | _query_planar_faces | tolerance/tolerance-compare | `first_remainder <= area_tolerance` |
| intersections.py:2478 | _query_planar_faces | tolerance/tolerance-compare | `second_remainder <= area_tolerance` |
| intersections.py:2493 | _query_planar_faces | tolerance/tolerance-compare | `length > tolerance` |
| intersections.py:2568 | _query_planar_faces.append_touch | tolerance/tolerance-compare | `float( np.linalg.norm( np.asarray(existing.witnesses[0], dtype=float) - witness ) ) <= tolerance` |
| intersections.py:2594 | _query_planar_faces | tolerance/tolerance-compare | `upper < lower - tolerance` |
| intersections.py:2596 | _query_planar_faces | tolerance/tolerance-compare | `upper - lower <= tolerance` |
| intersections.py:2617 | _query_planar_faces | tolerance/tolerance-compare | `interval.lower - tolerance <= parameter <= interval.upper + tolerance` |
| intersections.py:2619 | _query_planar_faces | tolerance/tolerance-compare | `abs(parameter - other) <= tolerance` |
| intersections.py:2623 | _query_planar_faces | tolerance/tolerance-compare | `interval.lower - tolerance <= parameter <= interval.upper + tolerance` |
| intersections.py:2628 | _query_planar_faces | ordering/select-by-float-key | `intersections.sort(key=lambda item: item.first_parameter_range.lower)` |
| intersections.py:2640 | _query_planar_faces | ordering/select-by-float-key | `touches.sort(key=lambda item: item.witnesses)` |
| intersections.py:2676 | _angle_on_arc_sweep | ordering/sign-branch | `sweep >= 0.0` |
| intersections.py:2677 | _angle_on_arc_sweep | tolerance/tolerance-compare | `-angular_tolerance <= candidate <= sweep + angular_tolerance` |
| intersections.py:2679 | _angle_on_arc_sweep | tolerance/tolerance-compare | `sweep - angular_tolerance <= candidate <= angular_tolerance` |
| intersections.py:2722 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `lengths <= tolerance` |
| intersections.py:2733 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `abs(area_twice) <= area_tolerance` |
| intersections.py:2735 | _certified_curve_inside_convex_support | ordering/sign-branch | `area_twice > 0.0` |
| intersections.py:2744 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `turns < -area_tolerance` |
| intersections.py:2754 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `abs(float((point - surface.origin) @ surface.normal)) > residual_tolerance` |
| intersections.py:2785 | _certified_curve_inside_convex_support | ordering/min-max-value | `min( constant + cosine * np.cos(angle) + sine * np.sin(angle) for angle in angles )` |
| intersections.py:2789 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `minimum <= tolerance` |
| intersections.py:2802 | _certified_curve_inside_convex_support | tolerance/tolerance-compare | `margins <= tolerance` |
| intersections.py:2986 | _query_plane_cylinder_faces | tolerance/tolerance-compare | `distance > tolerance` |
| intersections.py:2997 | _query_plane_cylinder_faces | ordering/min-max-value | `max( 4.0 * scale / max(len(points) - 1, 1), 16.0 * tolerance, )` |
| intersections.py:3002 | _query_plane_cylinder_faces | ordering/float-compare | `float(np.linalg.norm(point - current[-1])) > maximum_step` |
| intersections.py:3041 | _query_plane_cylinder_faces | ordering/select-by-float-key | `components.sort(key=lambda item: item.witnesses)` |
| intersections.py:3310 | _canonical_member_point_components.witness_distance | ordering/float-equality | `distance == 0.0` |
| intersections.py:3332 | _canonical_member_point_components.equivalent | tolerance/tolerance-compare | `abs( first_component.first_parameter[0] - second_component.first_parameter[0] ) <= first_parameter_tolerance` |
| intersections.py:3337 | _canonical_member_point_components.equivalent | tolerance/tolerance-compare | `abs( first_component.second_parameter[0] - second_component.second_parameter[0] ) <= second_parameter_tolerance` |
| intersections.py:3342 | _canonical_member_point_components.equivalent | tolerance/tolerance-compare | `witness_distance( first_component.witnesses[0], second_component.witnesses[0] ) <= tolerance` |
| intersections.py:3386 | _canonical_member_point_components | ordering/select-by-float-key | `group_indices.sort(key=lambda group: group[0])` |
| intersections.py:3430 | _canonical_member_point_components | ordering/float-equality | `item.witnesses != representative.witnesses` |
| intersections.py:3432 | _canonical_member_point_components | ordering/min-max-value | `max( displacement + max( item.max_residual, 0.0 if certificate is None else certificate.max_residual, ) for item, certificate, displacement in zip( group, certificates, canonicalization_displacements ) )` |
| intersections.py:3434 | _canonical_member_point_components | ordering/min-max-value | `max( item.max_residual, 0.0 if certificate is None else certificate.max_residual, )` |
| intersections.py:3442 | _canonical_member_point_components | ordering/min-max-value | `max( displacement + ( 0.0 if certificate is None else certificate.max_enclosure_width ) for certificate, displacement in zip( certificates, canonicalization_displacements ) )` |
| intersections.py:3466 | _canonical_member_point_components | ordering/min-max-value | `max( tolerance, maximum_residual, *( item.tolerance for item in certificates if item is not None ), )` |
| intersections.py:3508 | _query_same_circle_arcs | ordering/min-max-value | `max(first_frame.radius, second_frame.radius, 1.0)` |
| intersections.py:3511 | _query_same_circle_arcs | tolerance/tolerance-compare | `float(np.linalg.norm(first_frame.center - second_frame.center)) > tolerance` |
| intersections.py:3512 | _query_same_circle_arcs | tolerance/tolerance-compare | `abs(first_frame.radius - second_frame.radius) > tolerance` |
| intersections.py:3513 | _query_same_circle_arcs | tolerance/tolerance-compare | `abs(float(first_frame.normal @ second_frame.normal)) < 1.0 - geometry.tolerance.angular` |
| intersections.py:3535 | _query_same_circle_arcs | ordering/select-by-float-key | `sorted((0.0, first_frame.sweep))` |
| intersections.py:3538 | _query_same_circle_arcs | ordering/min-max-value | `max(first_lower, float(min(shifted[0], shifted[-1])))` |
| intersections.py:3538 | _query_same_circle_arcs | ordering/min-max-value | `min(shifted[0], shifted[-1])` |
| intersections.py:3539 | _query_same_circle_arcs | ordering/min-max-value | `min(first_upper, float(max(shifted[0], shifted[-1])))` |
| intersections.py:3539 | _query_same_circle_arcs | ordering/min-max-value | `max(shifted[0], shifted[-1])` |
| intersections.py:3541 | _query_same_circle_arcs | ordering/float-compare | `overlap > best[0]` |
| intersections.py:3544 | _query_same_circle_arcs | ordering/min-max-value | `max(first_frame.radius, tolerance)` |
| intersections.py:3545 | _query_same_circle_arcs | tolerance/tolerance-compare | `best[0] < -angular_tolerance` |
| intersections.py:3552 | _query_same_circle_arcs | tolerance/tolerance-compare | `upper - lower <= angular_tolerance` |
| intersections.py:3553 | _query_same_circle_arcs | ordering/min-max-value | `np.clip(0.5 * (lower + upper) / first_frame.sweep, 0.0, 1.0)` |
| intersections.py:3572 | _query_same_circle_arcs | ordering/min-max-value | `np.clip(value / first_frame.sweep, 0.0, 1.0)` |
| intersections.py:3583 | _query_same_circle_arcs | ordering/min-max-value | `min(first_parameters)` |
| intersections.py:3583 | _query_same_circle_arcs | tolerance/tolerance-compare | `min(first_parameters) <= geometry.tolerance.parameter` |
| intersections.py:3583 | _query_same_circle_arcs | ordering/min-max-value | `max(first_parameters)` |
| intersections.py:3583 | _query_same_circle_arcs | tolerance/tolerance-compare | `max(first_parameters) >= 1.0 - geometry.tolerance.parameter` |
| intersections.py:3584 | _query_same_circle_arcs | ordering/min-max-value | `min(second_parameters)` |
| intersections.py:3584 | _query_same_circle_arcs | tolerance/tolerance-compare | `min(second_parameters) <= geometry.tolerance.parameter` |
| intersections.py:3584 | _query_same_circle_arcs | ordering/min-max-value | `max(second_parameters)` |
| intersections.py:3584 | _query_same_circle_arcs | tolerance/tolerance-compare | `max(second_parameters) >= 1.0 - geometry.tolerance.parameter` |
| intersections.py:3604 | _query_same_circle_arcs | ordering/min-max-value | `max( geometry.closest_edge_point(second_edge_id, witness)[2] for witness in witnesses )` |
| intersections.py:3638 | _query_member_member | ordering/min-max-value | `max(tolerance, local.tolerance_used or 0.0)` |
| intersections.py:3671 | _query_member_member | ordering/float-equality | `kind in unresolved_kinds` |
| intersections.py:3691 | _query_member_member | ordering/min-max-value | `max( tolerance, maximum_residual, *(item.tolerance for item in local_certificates), )` |
| intersections.py:3722 | _query_member_member | ordering/select-by-float-key | `components.sort( key=lambda item: ( (item[1].first_parameter_range.lower,) if item[1].first_parameter_range is not None else item[1].first_parameter or (), (item[1].second_parameter_range.lower,) if item[1].second_parameter_range is not ...` |
| intersections.py:3741 | _query_member_member | ordering/float-equality | `item in kinds` |
| intersections.py:3743 | _query_member_member | ordering/float-equality | `kind in ( IntersectionKind.CROSS, IntersectionKind.TOUCH_POINT, IntersectionKind.TANGENT, )` |
| intersections.py:3795 | _query_member_material | ordering/min-max-value | `max(tolerance, local.tolerance_used or 0.0)` |
| intersections.py:3842 | _query_member_material | ordering/sign-branch | `length <= 0.0` |
| intersections.py:3852 | _query_member_material | ordering/min-max-value | `max(tolerance, clipped.tolerance_used or 0.0)` |
| intersections.py:3862 | _query_member_material | ordering/min-max-value | `max(0.0, component.first_parameter_range.lower)` |
| intersections.py:3863 | _query_member_material | ordering/min-max-value | `min(length, component.first_parameter_range.upper)` |
| intersections.py:3864 | _query_member_material | tolerance/tolerance-compare | `upper < lower - tolerance` |
| intersections.py:3866 | _query_member_material | ordering/min-max-value | `np.clip(lower / length, 0.0, 1.0)` |
| intersections.py:3867 | _query_member_material | ordering/min-max-value | `np.clip(upper / length, 0.0, 1.0)` |
| intersections.py:3914 | _query_member_material | tolerance/tolerance-compare | `physical < -tolerance` |
| intersections.py:3914 | _query_member_material | tolerance/tolerance-compare | `physical > length + tolerance` |
| intersections.py:3916 | _query_member_material | ordering/min-max-value | `np.clip(physical / length, 0.0, 1.0)` |
| intersections.py:3964 | _query_member_material | ordering/select-by-float-key | `components.sort( key=lambda item: ( (item.first_parameter_range.lower,) if item.first_parameter_range is not None else item.first_parameter or (), item.second_subparent.sort_key if item.second_subparent else (), ) )` |
| intersections.py:3976 | _query_member_material | ordering/select-by-float-key | `sorted( ( item.first_parameter_range.lower, item.first_parameter_range.upper, ) for item in interval_components )` |
| intersections.py:3987 | _query_member_material | tolerance/tolerance-compare | `lower > cursor + parameter_tolerance` |
| intersections.py:3990 | _query_member_material | ordering/min-max-value | `max(cursor, upper)` |
| intersections.py:3991 | _query_member_material | tolerance/tolerance-compare | `cursor >= 1.0 - parameter_tolerance` |
| intersections.py:4066 | _query_edge_face | tolerance/tolerance-compare | `length <= tolerance` |
| intersections.py:4097 | _query_edge_face | ordering/min-max-value | `max(0.0, component.first_parameter_range.lower)` |
| intersections.py:4098 | _query_edge_face | ordering/min-max-value | `min(length, component.first_parameter_range.upper)` |
| intersections.py:4099 | _query_edge_face | tolerance/tolerance-compare | `upper < lower - tolerance` |
| intersections.py:4133 | _query_edge_face | tolerance/tolerance-compare | `physical < -tolerance` |
| intersections.py:4133 | _query_edge_face | tolerance/tolerance-compare | `physical > length + tolerance` |
| intersections.py:4135 | _query_edge_face | ordering/min-max-value | `np.clip(physical / length, 0.0, 1.0)` |
| intersections.py:4170 | _query_edge_face | ordering/select-by-float-key | `components.sort( key=lambda item: ( (item.first_parameter_range.lower,) if item.first_parameter_range is not None else item.first_parameter or () ) )` |
| intersections.py:4182 | _query_edge_face | tolerance/tolerance-compare | `item.first_parameter_range.lower <= geometry.tolerance.parameter` |
| intersections.py:4183 | _query_edge_face | tolerance/tolerance-compare | `item.first_parameter_range.upper >= 1.0 - geometry.tolerance.parameter` |
| intersections.py:4637 | plan_imprint | tolerance/isclose-call | `np.allclose(original.first_parameter,current.first_parameter, rtol=0.,atol=geometry.tolerance.parameter)` |
| intersections.py:4639 | plan_imprint | tolerance/isclose-call | `np.allclose(original.second_parameter,current.second_parameter, rtol=0.,atol=geometry.tolerance.parameter)` |
| intersections.py:4641 | plan_imprint | tolerance/isclose-call | `np.allclose(original.witnesses,current.witnesses, rtol=0.,atol=result.tolerance_used)` |
| intersections.py:4848 | _parameter_is_endpoint | ordering/sign-branch | `value <= 0.0` |
| intersections.py:4848 | _parameter_is_endpoint | ordering/float-compare | `value >= 1.0` |
| intersections.py:4885 | _member_relation_matches | tolerance/tolerance-compare | `abs(use.member_range.start-expected.start)<=tolerance` |
| intersections.py:4886 | _member_relation_matches | tolerance/tolerance-compare | `abs(use.member_range.end-expected.end)<=tolerance` |
| intersections.py:4948 | _range_from_component | ordering/min-max-value | `np.clip(interval.lower, 0.0, 1.0)` |
| intersections.py:4949 | _range_from_component | ordering/min-max-value | `np.clip(interval.upper, 0.0, 1.0)` |
| intersections.py:4953 | _range_from_component | ordering/min-max-value | `np.clip(parameter[0], 0.0, 1.0)` |
| intersections.py:5084 | _member_connection_application | tolerance/tolerance-compare | `local <= geometry.tolerance.parameter` |
| intersections.py:5086 | _member_connection_application | tolerance/tolerance-compare | `local >= 1.0 - geometry.tolerance.parameter` |
| intersections.py:5238 | _member_sheet_component_kind | tolerance/tolerance-compare | `boundary_distance <= tolerance` |
| intersections.py:5485 | _member_sheet_application | ordering/min-max-value | `min(u_values)` |
| intersections.py:5485 | _member_sheet_application | ordering/min-max-value | `max(u_values)` |
| intersections.py:5486 | _member_sheet_application | ordering/min-max-value | `min(v_values)` |
| intersections.py:5486 | _member_sheet_application | ordering/min-max-value | `max(v_values)` |
| intersections.py:5621 | _world_key | ordering/round-floor | `round(float(component) / quantum)` |
| intersections.py:5633 | _point_on_segment | tolerance/tolerance-compare | `squared <= tolerance * tolerance` |
| intersections.py:5634 | _point_on_segment | tolerance/tolerance-compare | `float(np.linalg.norm(point - start)) <= tolerance` |
| intersections.py:5636 | _point_on_segment | ordering/min-max-value | `np.clip(parameter, 0.0, 1.0)` |
| intersections.py:5638 | _point_on_segment | tolerance/tolerance-compare | `-tolerance <= parameter <= 1.0 + tolerance` |
| intersections.py:5639 | _point_on_segment | tolerance/tolerance-compare | `float(np.linalg.norm(point - projected)) <= tolerance` |
| intersections.py:5682 | _shared_edges_for_components | ordering/min-max-value | `min( float(np.linalg.norm(sample - endpoint)) for sample in geometry.sample_edge( edge_id, np.asarray((0.0, 1.0)) ) )` |
| intersections.py:5682 | _shared_edges_for_components | tolerance/tolerance-compare | `min( float(np.linalg.norm(sample - endpoint)) for sample in geometry.sample_edge( edge_id, np.asarray((0.0, 1.0)) ) ) <= tolerance` |
| intersections.py:5822 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[0] - start)) <= tolerance` |
| intersections.py:5823 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[1] - end)) <= tolerance` |
| intersections.py:5826 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[1] - start)) <= tolerance` |
| intersections.py:5827 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[0] - end)) <= tolerance` |
| intersections.py:5871 | _fragment_planar_face_by_segments.snap_endpoint | ordering/min-max-value | `min( ( float(np.linalg.norm(raw - position)), candidate_id, position, ) for candidate_id, position in trim_vertices )` |
| intersections.py:5879 | _fragment_planar_face_by_segments.snap_endpoint | tolerance/tolerance-compare | `distance > tolerance` |
| intersections.py:5891 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `direction_length <= tolerance` |
| intersections.py:5894 | _fragment_planar_face_by_segments | ordering/select-by-float-key | `np.argmax(np.abs(direction))` |
| intersections.py:5895 | _fragment_planar_face_by_segments | ordering/sign-branch | `direction[canonical_index] < 0.0` |
| intersections.py:5900 | _fragment_planar_face_by_segments | ordering/float-compare | `float(start @ direction) > float(end @ direction)` |
| intersections.py:5903 | _fragment_planar_face_by_segments | ordering/select-by-float-key | `ordered_segments.sort( key=lambda item: ( float(item[0] @ direction), float(item[1] @ direction), ) )` |
| intersections.py:5924 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(Point(point_uv).distance(polygon.boundary)) <= tolerance` |
| intersections.py:5940 | _fragment_planar_face_by_segments | ordering/min-max-value | `max( math.hypot(maximum_x - minimum_x, maximum_y - minimum_y), 1.0, )` |
| intersections.py:5946 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `float(point.distance(polygon.boundary)) <= tolerance` |
| intersections.py:5952 | _fragment_planar_face_by_segments | tolerance/tolerance-compare | `away_length <= tolerance` |
| intersections.py:5996 | _fragment_planar_face_by_segments | ordering/select-by-float-key | `sorted( ( cell for cell in polygonize(linework) if polygon.covers(cell.representative_point()) and float(cell.area) > geometry.tolerance.effective_area( _face_length_scale(geometry, face_id) ) ), key=lambda cell: ( round(float(cell.centr...` |
| intersections.py:6000 | _fragment_planar_face_by_segments | library/library-predicate | `polygon.covers(cell.representative_point())` |
| intersections.py:6001 | _fragment_planar_face_by_segments | ordering/float-compare | `float(cell.area) > geometry.tolerance.effective_area( _face_length_scale(geometry, face_id) )` |
| intersections.py:6006 | _fragment_planar_face_by_segments | ordering/round-floor | `round(float(cell.centroid.x), 14)` |
| intersections.py:6007 | _fragment_planar_face_by_segments | ordering/round-floor | `round(float(cell.centroid.y), 14)` |
| intersections.py:6008 | _fragment_planar_face_by_segments | ordering/round-floor | `round(float(cell.area), 14)` |
| intersections.py:6027 | _fragment_planar_face_by_segments.vertex | provenance/derived-construction | `geometry.add_point(*world)` |
| intersections.py:6113 | _canonicalize_coincident_boundary_edges | tolerance/tolerance-compare | `float(np.linalg.norm(duplicate_points[0] - canonical_points[0])) <= tolerance` |
| intersections.py:6115 | _canonicalize_coincident_boundary_edges | tolerance/tolerance-compare | `float(np.linalg.norm(duplicate_points[1] - canonical_points[1])) <= tolerance` |
| intersections.py:6119 | _canonicalize_coincident_boundary_edges | tolerance/tolerance-compare | `float(np.linalg.norm(duplicate_points[0] - canonical_points[1])) <= tolerance` |
| intersections.py:6121 | _canonicalize_coincident_boundary_edges | tolerance/tolerance-compare | `float(np.linalg.norm(duplicate_points[1] - canonical_points[0])) <= tolerance` |
| intersections.py:6424 | _face_imprint_application | ordering/sign-branch | `float(normal @ source_normals[use.sheet_id]) > 0` |
| intersections.py:6467 | _face_imprint_application | tolerance/tolerance-compare | `abs(alignment - 1.0) <= geometry.tolerance.angular` |
| intersections.py:6475 | _face_imprint_application | tolerance/tolerance-compare | `abs(alignment - 1.0) <= geometry.tolerance.angular` |
| intersections.py:6564 | _face_imprint_application | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[0] - start)) <= tolerance` |
| intersections.py:6565 | _face_imprint_application | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[1] - end)) <= tolerance` |
| intersections.py:6567 | _face_imprint_application | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[1] - start)) <= tolerance` |
| intersections.py:6568 | _face_imprint_application | tolerance/tolerance-compare | `float(np.linalg.norm(endpoints[0] - end)) <= tolerance` |
| intersections.py:6860 | _reuse_stale_face_plan | ordering/min-max-value | `min( float(np.linalg.norm(sample - endpoint)) for sample in geometry.sample_edge( candidate, np.asarray((0.0, 1.0)) ) )` |
| intersections.py:6860 | _reuse_stale_face_plan | tolerance/tolerance-compare | `min( float(np.linalg.norm(sample - endpoint)) for sample in geometry.sample_edge( candidate, np.asarray((0.0, 1.0)) ) ) <= tolerance` |
| predicates.py:51 | _positive_integer | ordering/round-floor | `int(value)` |
| predicates.py:142 | IntersectionCertificate.__post_init__ | ordering/sign-branch | `value < 0.0` |
| predicates.py:153 | IntersectionCertificate.__post_init__ | ordering/sign-branch | `separation < 0.0` |
| predicates.py:161 | IntersectionCertificate.__post_init__ | ordering/round-floor | `int(value)` |
| predicates.py:163 | IntersectionCertificate.__post_init__ | ordering/round-floor | `int(value)` |
| predicates.py:167 | IntersectionCertificate.__post_init__ | tolerance/tolerance-compare | `self.max_residual > self.tolerance` |
| predicates.py:341 | IntersectionComponent.__post_init__ | ordering/sign-branch | `residual < 0.0` |
| predicates.py:395 | IntersectionComponent.__post_init__ | ordering/min-max-value | `max(residual, 0.0)` |
| predicates.py:399 | IntersectionComponent.__post_init__ | tolerance/tolerance-compare | `certificate.max_residual + np.finfo(float).eps < residual` |
| predicates.py:492 | IntersectionResult.__post_init__ | ordering/min-max-value | `max( 0.0 if tolerance is None else tolerance, summary_residual, *(item.certificate.tolerance for item in components if item.certificate is not None), )` |
| predicates.py:664 | _stable_norm | ordering/float-equality | `largest == 0.0` |
| predicates.py:672 | _unit | ordering/float-equality | `length == 0.0` |
| predicates.py:682 | _canonical_direction | ordering/select-by-float-key | `np.argmax(np.abs(made))` |
| predicates.py:683 | _canonical_direction | ordering/sign-branch | `made[index] < 0.0` |
| predicates.py:701 | _extent | ordering/sign-branch | `made < 0.0` |
| predicates.py:703 | _extent | ordering/min-max-value | `max(made, derived)` |
| predicates.py:738 | _component | ordering/min-max-value | `max(0.0, float(residual))` |
| predicates.py:744 | _canonical_plane_basis | ordering/select-by-float-key | `np.argmin(np.abs(normal))` |
| predicates.py:794 | qualified_line_cylinder | ordering/sign-branch | `radius <= 0.0` |
| predicates.py:796 | qualified_line_cylinder | ordering/float-equality | `height == 0.0` |
| predicates.py:801 | qualified_line_cylinder | ordering/min-max-value | `max(radius, abs(height), _stable_norm(point - origin))` |
| predicates.py:814 | qualified_line_cylinder | ordering/min-max-value | `max( np.finfo(float).eps * 64.0, tolerance_policy.angular * tolerance_policy.angular, )` |
| predicates.py:837 | qualified_line_cylinder | tolerance/tolerance-compare | `a <= coefficient_tolerance` |
| predicates.py:839 | qualified_line_cylinder | tolerance/tolerance-compare | `radial_residual > length_tolerance` |
| predicates.py:847 | qualified_line_cylinder | tolerance/tolerance-compare | `abs(axial_speed) <= tolerance_policy.angular` |
| predicates.py:850 | qualified_line_cylinder | ordering/select-by-float-key | `sorted(((-axial_at_point) / axial_speed, (height - axial_at_point) / axial_speed))` |
| predicates.py:866 | qualified_line_cylinder | ordering/min-max-value | `max( length_tolerance * max(abs(a), abs(b), abs(c), 1.0), 64.0 * np.finfo(float).eps * (b * b + abs(4.0 * a * c)), )` |
| predicates.py:867 | qualified_line_cylinder | ordering/min-max-value | `max(abs(a), abs(b), abs(c), 1.0)` |
| predicates.py:870 | qualified_line_cylinder | tolerance/tolerance-compare | `discriminant < -discriminant_tolerance` |
| predicates.py:872 | qualified_line_cylinder | tolerance/tolerance-compare | `abs(discriminant) <= discriminant_tolerance` |
| predicates.py:876 | qualified_line_cylinder | ordering/min-max-value | `max(discriminant, 0.0)` |
| predicates.py:878 | qualified_line_cylinder | ordering/sign-branch | `np.copysign(square_root, b)` |
| predicates.py:879 | qualified_line_cylinder | ordering/select-by-float-key | `sorted((q / a, c / q))` |
| predicates.py:879 | qualified_line_cylinder | ordering/float-equality | `q != 0.0` |
| predicates.py:891 | qualified_line_cylinder | tolerance/tolerance-compare | `residual > length_tolerance` |
| predicates.py:986 | qualified_line_line | tolerance/tolerance-compare | `distance > length_tolerance` |
| predicates.py:1014 | qualified_line_line | tolerance/tolerance-compare | `distance > length_tolerance` |
| predicates.py:1060 | qualified_line_plane | tolerance/tolerance-compare | `abs(signed_distance) > length_tolerance` |
| predicates.py:1078 | qualified_line_plane | tolerance/tolerance-compare | `residual > length_tolerance` |
| predicates.py:1125 | qualified_plane_plane | tolerance/tolerance-compare | `distance > length_tolerance` |
| predicates.py:1154 | qualified_plane_plane | ordering/min-max-value | `max(first_residual, second_residual)` |
| predicates.py:1155 | qualified_plane_plane | tolerance/tolerance-compare | `residual > length_tolerance` |
| predicates.py:1179 | _closest_endpoint_pair | ordering/min-max-value | `min( max(float((point - second_start) @ second_vector) / second_squared, 0.0), 1.0, )` |
| predicates.py:1180 | _closest_endpoint_pair | ordering/min-max-value | `max(float((point - second_start) @ second_vector) / second_squared, 0.0)` |
| predicates.py:1188 | _closest_endpoint_pair | ordering/min-max-value | `min( max(float((point - first_start) @ first_vector) / first_squared, 0.0), 1.0, )` |
| predicates.py:1189 | _closest_endpoint_pair | ordering/min-max-value | `max(float((point - first_start) @ first_vector) / first_squared, 0.0)` |
| predicates.py:1196 | _closest_endpoint_pair | ordering/select-by-float-key | `min( candidates, key=lambda item: (item[0], item[1], item[2]) )` |
| predicates.py:1242 | qualified_segment_segment | tolerance/tolerance-compare | `first_length <= length_tolerance` |
| predicates.py:1242 | qualified_segment_segment | tolerance/tolerance-compare | `second_length <= length_tolerance` |
| predicates.py:1259 | qualified_segment_segment | tolerance/tolerance-compare | `max(endpoint_distances) > length_tolerance` |
| predicates.py:1266 | qualified_segment_segment | ordering/min-max-value | `max(0.0, min(second_on_first))` |
| predicates.py:1266 | qualified_segment_segment | ordering/min-max-value | `min(second_on_first)` |
| predicates.py:1267 | qualified_segment_segment | ordering/min-max-value | `min(1.0, max(second_on_first))` |
| predicates.py:1267 | qualified_segment_segment | ordering/min-max-value | `max(second_on_first)` |
| predicates.py:1268 | qualified_segment_segment | ordering/min-max-value | `max(0.0, upper - lower)` |
| predicates.py:1269 | qualified_segment_segment | ordering/float-compare | `upper < lower` |
| predicates.py:1269 | qualified_segment_segment | tolerance/tolerance-compare | `overlap_length <= length_tolerance` |
| predicates.py:1277 | qualified_segment_segment | tolerance/tolerance-compare | `distance > length_tolerance` |
| predicates.py:1315 | qualified_segment_segment | tolerance/tolerance-compare | `max(residuals) > length_tolerance` |
| predicates.py:1328 | qualified_segment_segment | tolerance/tolerance-compare | `lower <= first_parameter_tolerance` |
| predicates.py:1328 | qualified_segment_segment | tolerance/tolerance-compare | `upper >= 1.0 - first_parameter_tolerance` |
| predicates.py:1330 | qualified_segment_segment | ordering/min-max-value | `min(second_parameters)` |
| predicates.py:1330 | qualified_segment_segment | tolerance/tolerance-compare | `min(second_parameters) <= second_parameter_tolerance` |
| predicates.py:1331 | qualified_segment_segment | ordering/min-max-value | `max(second_parameters)` |
| predicates.py:1331 | qualified_segment_segment | tolerance/tolerance-compare | `max(second_parameters) >= 1.0 - second_parameter_tolerance` |
| predicates.py:1355 | qualified_segment_segment | tolerance/tolerance-compare | `first_parameter < -first_parameter_tolerance` |
| predicates.py:1356 | qualified_segment_segment | tolerance/tolerance-compare | `first_parameter > 1.0 + first_parameter_tolerance` |
| predicates.py:1357 | qualified_segment_segment | tolerance/tolerance-compare | `second_parameter < -second_parameter_tolerance` |
| predicates.py:1358 | qualified_segment_segment | tolerance/tolerance-compare | `second_parameter > 1.0 + second_parameter_tolerance` |
| predicates.py:1372 | qualified_segment_segment | tolerance/tolerance-compare | `distance > length_tolerance` |
| predicates.py:1382 | qualified_segment_segment | ordering/float-equality | `first_parameter in (0.0, 1.0)` |
| predicates.py:1382 | qualified_segment_segment | ordering/float-equality | `second_parameter in (0.0, 1.0)` |
| curved_intersections.py:116 | _result | tolerance/tolerance-compare | `float( np.linalg.norm( np.asarray(component.witnesses[0]) - np.asarray(component.witnesses[-1]) ) ) <= tolerance` |
| curved_intersections.py:196 | _bezier_subcurve | ordering/sign-branch | `lower <= 0.0` |
| curved_intersections.py:196 | _bezier_subcurve | ordering/float-compare | `upper >= 1.0` |
| curved_intersections.py:199 | _bezier_subcurve | ordering/sign-branch | `lower <= 0.0` |
| curved_intersections.py:201 | _bezier_subcurve | ordering/sign-branch | `upper > 0.0` |
| curved_intersections.py:219 | _angle_candidates | ordering/round-floor | `ceil((lower - phase) / pi)` |
| curved_intersections.py:220 | _angle_candidates | ordering/round-floor | `floor((upper - phase) / pi)` |
| curved_intersections.py:239 | _curve_bounds | ordering/float-equality | `frame.sweep != 0.0` |
| curved_intersections.py:240 | _curve_bounds | ordering/min-max-value | `np.clip(angle / frame.sweep, lower, upper)` |
| curved_intersections.py:320 | _same_bezier_overlap | ordering/min-max-value | `np.max(np.linalg.norm(first_points - second_points, axis=1))` |
| curved_intersections.py:321 | _same_bezier_overlap | ordering/min-max-value | `np.max(np.linalg.norm(first_points - second_points[::-1], axis=1))` |
| curved_intersections.py:322 | _same_bezier_overlap | ordering/min-max-value | `min(direct, reverse)` |
| curved_intersections.py:322 | _same_bezier_overlap | tolerance/tolerance-compare | `min(direct, reverse) > tolerance` |
| curved_intersections.py:324 | _same_bezier_overlap | ordering/float-compare | `reverse < direct` |
| curved_intersections.py:329 | _same_bezier_overlap | ordering/min-max-value | `min(direct, reverse)` |
| curved_intersections.py:338 | _same_bezier_overlap | ordering/min-max-value | `min(direct, reverse)` |
| curved_intersections.py:377 | _line_spline_intersection | tolerance/tolerance-compare | `length_squared <= tolerance * tolerance` |
| curved_intersections.py:383 | _line_spline_intersection | ordering/select-by-float-key | `np.argmax(np.linalg.norm(cross_coefficients, axis=0))` |
| curved_intersections.py:386 | _line_spline_intersection | ordering/min-max-value | `max( float(np.max(np.linalg.norm(cross_coefficients, axis=1))), tolerance, )` |
| curved_intersections.py:387 | _line_spline_intersection | ordering/min-max-value | `np.max(np.linalg.norm(cross_coefficients, axis=1))` |
| curved_intersections.py:390 | _line_spline_intersection | tolerance/tolerance-compare | `abs(float(polynomial[-1])) <= np.finfo(float).eps * coefficient_scale` |
| curved_intersections.py:395 | _line_spline_intersection | ordering/min-max-value | `max(np.sqrt(parameter_tolerance), 32.0 * np.finfo(float).eps)` |
| curved_intersections.py:396 | _line_spline_intersection | ordering/select-by-float-key | `sorted( float(np.clip(root.real, 0.0, 1.0)) for root in roots if abs(float(root.imag)) <= imaginary_tolerance and -parameter_tolerance <= float(root.real) <= 1.0 + parameter_tolerance )` |
| curved_intersections.py:397 | _line_spline_intersection | ordering/min-max-value | `np.clip(root.real, 0.0, 1.0)` |
| curved_intersections.py:399 | _line_spline_intersection | tolerance/tolerance-compare | `abs(float(root.imag)) <= imaginary_tolerance` |
| curved_intersections.py:400 | _line_spline_intersection | tolerance/tolerance-compare | `-parameter_tolerance <= float(root.real) <= 1.0 + parameter_tolerance` |
| curved_intersections.py:409 | _line_spline_intersection | tolerance/tolerance-compare | `residual > tolerance` |
| curved_intersections.py:410 | _line_spline_intersection | tolerance/tolerance-compare | `line_parameter < -parameter_tolerance` |
| curved_intersections.py:411 | _line_spline_intersection | tolerance/tolerance-compare | `line_parameter > 1.0 + parameter_tolerance` |
| curved_intersections.py:420 | _line_spline_intersection | tolerance/tolerance-compare | `abs(spline_parameter - accepted[-1][1]) <= imaginary_tolerance` |
| curved_intersections.py:421 | _line_spline_intersection | ordering/float-compare | `residual < accepted[-1][3]` |
| curved_intersections.py:441 | _line_spline_intersection | tolerance/tolerance-compare | `float(np.linalg.norm(np.cross(direction, spline_tangent))) <= ( model.tolerance.angular * float(np.linalg.norm(direction)) * float(np.linalg.norm(spline_tangent)) )` |
| curved_intersections.py:446 | _line_spline_intersection | ordering/min-max-value | `min( line_parameter, spline_parameter, 1.0 - line_parameter, 1.0 - spline_parameter, )` |
| curved_intersections.py:446 | _line_spline_intersection | tolerance/tolerance-compare | `min( line_parameter, spline_parameter, 1.0 - line_parameter, 1.0 - spline_parameter, ) <= parameter_tolerance` |
| curved_intersections.py:489 | _line_spline_intersection | ordering/float-equality | `item in kinds` |
| curved_intersections.py:561 | qualified_curve_curve | ordering/min-max-value | `min(minimum_separation, separation)` |
| curved_intersections.py:562 | qualified_curve_curve | tolerance/tolerance-compare | `separation > tolerance` |
| curved_intersections.py:570 | qualified_curve_curve | tolerance/tolerance-compare | `first_span <= parameter_tolerance` |
| curved_intersections.py:571 | qualified_curve_curve | tolerance/tolerance-compare | `second_span <= parameter_tolerance` |
| curved_intersections.py:573 | qualified_curve_curve | tolerance/tolerance-compare | `enclosure <= 2.0 * tolerance` |
| curved_intersections.py:584 | qualified_curve_curve | tolerance/tolerance-compare | `residual <= tolerance` |
| curved_intersections.py:586 | qualified_curve_curve | tolerance/tolerance-compare | `enclosure > 8.0 * tolerance` |
| curved_intersections.py:637 | qualified_curve_curve | ordering/select-by-float-key | `roots.sort(key=lambda item: (item[0], item[1], tuple(item[2])))` |
| curved_intersections.py:641 | qualified_curve_curve | tolerance/tolerance-compare | `abs(root[0] - unique[-1][0]) <= parameter_tolerance` |
| curved_intersections.py:642 | qualified_curve_curve | tolerance/tolerance-compare | `abs(root[1] - unique[-1][1]) <= parameter_tolerance` |
| curved_intersections.py:670 | qualified_curve_curve | ordering/min-max-value | `max(0.0, separation)` |
| curved_intersections.py:681 | qualified_curve_curve | tolerance/tolerance-compare | `first_length <= tolerance` |
| curved_intersections.py:682 | qualified_curve_curve | tolerance/tolerance-compare | `second_length <= tolerance` |
| curved_intersections.py:683 | qualified_curve_curve | tolerance/tolerance-compare | `float(np.linalg.norm(np.cross(first_derivative, second_derivative))) <= model.tolerance.angular * first_length * second_length` |
| curved_intersections.py:686 | qualified_curve_curve | ordering/min-max-value | `min(first_t, second_t, 1.0 - first_t, 1.0 - second_t)` |
| curved_intersections.py:686 | qualified_curve_curve | tolerance/tolerance-compare | `min(first_t, second_t, 1.0 - first_t, 1.0 - second_t) <= parameter_tolerance` |
| curved_intersections.py:714 | qualified_curve_curve | ordering/float-equality | `candidate in kinds` |
| curved_intersections.py:1000 | _refine_curve_surface_root | ordering/float-compare | `candidate[:2] < best[:2]` |
| curved_intersections.py:1002 | _refine_curve_surface_root | tolerance/tolerance-compare | `residual <= model.tolerance.length` |
| curved_intersections.py:1002 | _refine_curve_surface_root | tolerance/tolerance-compare | `upper - lower <= model.tolerance.parameter` |
| curved_intersections.py:1004 | _refine_curve_surface_root | ordering/float-equality | `left_value == 0.0` |
| curved_intersections.py:1004 | _refine_curve_surface_root | ordering/sign-branch | `left_value * value <= 0.0` |
| curved_intersections.py:1287 | _canonical_polygon | tolerance/tolerance-compare | `float(np.linalg.norm(made[0] - made[-1])) <= tolerance` |
| curved_intersections.py:1291 | _canonical_polygon | ordering/sign-branch | `_polygon_area(made) < 0.0` |
| curved_intersections.py:1293 | _canonical_polygon | ordering/select-by-float-key | `min( range(len(made)), key=lambda index: (float(made[index, 0]), float(made[index, 1]), index), )` |
| curved_intersections.py:1306 | _convex_polygon | tolerance/tolerance-compare | `abs(cross) > tolerance` |
| curved_intersections.py:1307 | _convex_polygon | ordering/sign-branch | `cross > 0.0` |
| curved_intersections.py:1322 | _line_intersection_2d | tolerance/tolerance-compare | `abs(denominator) <= np.finfo(float).eps` |
| curved_intersections.py:1334 | _convex_intersection | ordering/sign-branch | `_polygon_area(clip) >= 0.0` |
| curved_intersections.py:1345 | _convex_intersection.inside | tolerance/tolerance-compare | `orientation * float(edge[0] * offset[1] - edge[1] * offset[0]) >= -tolerance` |
| curved_intersections.py:1389 | _point_on_polygon_boundary | tolerance/tolerance-compare | `length_squared <= np.finfo(float).tiny` |
| curved_intersections.py:1390 | _point_on_polygon_boundary | tolerance/tolerance-compare | `float(np.linalg.norm(point - start)) <= tolerance` |
| curved_intersections.py:1394 | _point_on_polygon_boundary | ordering/min-max-value | `np.clip((point - start) @ direction / length_squared, 0.0, 1.0)` |
| curved_intersections.py:1396 | _point_on_polygon_boundary | tolerance/tolerance-compare | `float(np.linalg.norm(point - (start + parameter * direction))) <= tolerance` |
| curved_intersections.py:1431 | _segment_breaks | ordering/min-max-value | `max( float(np.linalg.norm(first_direction)), float(np.linalg.norm(second_direction)), 1.0, )` |
| curved_intersections.py:1436 | _segment_breaks | tolerance/tolerance-compare | `abs(denominator) > tolerance * scale` |
| curved_intersections.py:1439 | _segment_breaks | tolerance/tolerance-compare | `-tolerance <= first_parameter <= 1.0 + tolerance` |
| curved_intersections.py:1439 | _segment_breaks | tolerance/tolerance-compare | `-tolerance <= second_parameter <= 1.0 + tolerance` |
| curved_intersections.py:1440 | _segment_breaks | ordering/min-max-value | `np.clip(first_parameter, 0.0, 1.0)` |
| curved_intersections.py:1441 | _segment_breaks | ordering/min-max-value | `np.clip(second_parameter, 0.0, 1.0)` |
| curved_intersections.py:1443 | _segment_breaks | tolerance/tolerance-compare | `abs(_cross_2d(offset, first_direction)) > tolerance * scale` |
| curved_intersections.py:1447 | _segment_breaks | tolerance/tolerance-compare | `first_length <= np.finfo(float).tiny` |
| curved_intersections.py:1447 | _segment_breaks | tolerance/tolerance-compare | `second_length <= np.finfo(float).tiny` |
| curved_intersections.py:1451 | _segment_breaks | tolerance/tolerance-compare | `-tolerance <= value <= 1.0 + tolerance` |
| curved_intersections.py:1452 | _segment_breaks | ordering/min-max-value | `np.clip(value, 0.0, 1.0)` |
| curved_intersections.py:1455 | _segment_breaks | tolerance/tolerance-compare | `-tolerance <= value <= 1.0 + tolerance` |
| curved_intersections.py:1456 | _segment_breaks | ordering/min-max-value | `np.clip(value, 0.0, 1.0)` |
| curved_intersections.py:1499 | _material_intersection_loops | ordering/min-max-value | `np.clip(value, 0.0, 1.0)` |
| curved_intersections.py:1502 | _material_intersection_loops | tolerance/tolerance-compare | `upper - lower <= tolerance` |
| curved_intersections.py:1512 | _material_intersection_loops | ordering/min-max-value | `max(tolerance, 64.0 * np.finfo(float).eps)` |
| curved_intersections.py:1515 | _material_intersection_loops.key | ordering/round-floor | `round(float(value) / quantum)` |
| curved_intersections.py:1534 | _material_intersection_loops | ordering/select-by-float-key | `values.sort(key=lambda item: item[0])` |
| curved_intersections.py:1561 | _material_intersection_loops | ordering/select-by-float-key | `max( candidates, key=lambda item: ( float( np.arctan2( _cross_2d(incoming, item[2] - item[1]), float(incoming @ (item[2] - item[1])), ) ), tuple(-value for value in item[0]), ), )` |
| curved_intersections.py:1577 | _material_intersection_loops | tolerance/tolerance-compare | `abs(_polygon_area(polygon)) > tolerance * tolerance` |
| curved_intersections.py:1580 | _material_intersection_loops | ordering/select-by-float-key | `sorted( loops, key=lambda loop: ( _polygon_area(loop) < 0.0, float(np.min(loop[:, 0])), float(np.min(loop[:, 1])), -abs(_polygon_area(loop)), ), )` |
| curved_intersections.py:1583 | _material_intersection_loops | ordering/sign-branch | `_polygon_area(loop) < 0.0` |
| curved_intersections.py:1625 | _coincident_face_region | ordering/sign-branch | `_polygon_area(loop) > 0.0` |
| curved_intersections.py:1626 | _coincident_face_region | ordering/sign-branch | `_polygon_area(loop) < 0.0` |
| curved_intersections.py:1648 | _coincident_face_region | ordering/min-max-value | `max(first_area, second_area, 1.0)` |
| curved_intersections.py:1651 | _coincident_face_region | tolerance/tolerance-compare | `abs(overlap_area - first_area) <= area_tolerance` |
| curved_intersections.py:1652 | _coincident_face_region | tolerance/tolerance-compare | `abs(overlap_area - second_area) <= area_tolerance` |
| curved_intersections.py:1654 | _coincident_face_region | tolerance/tolerance-compare | `abs(overlap_area - min(first_area, second_area)) <= area_tolerance` |
| curved_intersections.py:1659 | _coincident_face_region | ordering/select-by-float-key | `sorted(region_holes, key=lambda item: tuple(item[0]))` |
| curved_intersections.py:1677 | _coincident_face_region | ordering/min-max-value | `max( float( np.linalg.norm( model.face_support_point(second.id, float(uv[0]), float(uv[1])) - point ) ) for world_loop, second_loop in zip(world_loops, second_parameter_loops) for point, uv in zip(world_loop, second_loop) )` |
| curved_intersections.py:1687 | _coincident_face_region | tolerance/tolerance-compare | `residual > tolerance` |
| curved_intersections.py:1737 | _supports_coincident | tolerance/tolerance-compare | `1.0 - axis_alignment <= model.tolerance.angular` |
| curved_intersections.py:1738 | _supports_coincident | tolerance/tolerance-compare | `float(np.linalg.norm(perpendicular)) <= tolerance` |
| curved_intersections.py:1739 | _supports_coincident | tolerance/tolerance-compare | `abs(first_surface.radius - second_surface.radius) <= tolerance` |
| curved_intersections.py:1744 | _supports_coincident | tolerance/tolerance-compare | `abs(first_slope) <= np.finfo(float).tiny` |
| curved_intersections.py:1744 | _supports_coincident | tolerance/tolerance-compare | `abs(second_slope) <= np.finfo(float).tiny` |
| curved_intersections.py:1753 | _supports_coincident | tolerance/tolerance-compare | `abs(abs(float(first_surface.axis @ second_surface.axis)) - 1.0) <= model.tolerance.angular` |
| curved_intersections.py:1755 | _supports_coincident | tolerance/tolerance-compare | `float(np.linalg.norm(first_apex - second_apex)) <= tolerance` |
| curved_intersections.py:1756 | _supports_coincident | tolerance/tolerance-compare | `abs(abs(first_slope) - abs(second_slope)) <= model.tolerance.angular * max(abs(first_slope), abs(second_slope), 1.0)` |
| curved_intersections.py:1757 | _supports_coincident | ordering/min-max-value | `max(abs(first_slope), abs(second_slope), 1.0)` |
| curved_intersections.py:1770 | _supports_coincident | tolerance/tolerance-compare | `float(np.max(np.linalg.norm(first_boundary - first_surface.first_boundary, axis=1))) <= tolerance` |
| curved_intersections.py:1770 | _supports_coincident | ordering/min-max-value | `np.max(np.linalg.norm(first_boundary - first_surface.first_boundary, axis=1))` |
| curved_intersections.py:1771 | _supports_coincident | tolerance/tolerance-compare | `float(np.max(np.linalg.norm(second_boundary - first_surface.second_boundary, axis=1))) <= tolerance` |
| curved_intersections.py:1771 | _supports_coincident | ordering/min-max-value | `np.max(np.linalg.norm(second_boundary - first_surface.second_boundary, axis=1))` |
| curved_intersections.py:1798 | _supports_coincident | tolerance/tolerance-compare | `float(np.max(np.linalg.norm(left - right, axis=1))) <= tolerance` |
| curved_intersections.py:1798 | _supports_coincident | ordering/min-max-value | `np.max(np.linalg.norm(left - right, axis=1))` |
| curved_intersections.py:1810 | _supports_coincident | tolerance/tolerance-compare | `residual > tolerance` |
| curved_intersections.py:1814 | _supports_coincident | ordering/sign-branch | `source_length <= 0.0` |
| curved_intersections.py:1817 | _supports_coincident | tolerance/tolerance-compare | `alignment < 1.0 - model.tolerance.angular` |
| curved_intersections.py:1819 | _supports_coincident | tolerance/tolerance-compare | `float(np.linalg.norm(projected - point)) > tolerance` |
| curved_intersections.py:1840 | _contour_crossing | ordering/float-equality | `first_value == 0.0` |
| curved_intersections.py:1842 | _contour_crossing | ordering/float-equality | `second_value == 0.0` |
| curved_intersections.py:1844 | _contour_crossing | ordering/sign-branch | `first_value * second_value > 0.0` |
| curved_intersections.py:1847 | _contour_crossing | ordering/float-equality | `denominator == 0.0` |
| curved_intersections.py:1849 | _contour_crossing | ordering/min-max-value | `np.clip(first_value / denominator, 0.0, 1.0)` |
| curved_intersections.py:1891 | _trace_segments.key | ordering/round-floor | `round(float(value) / quantum)` |
| curved_intersections.py:1951 | qualified_face_face | tolerance/tolerance-compare | `separation > tolerance` |
| curved_intersections.py:2052 | qualified_face_face | ordering/round-floor | `int(np.sqrt(policy.max_trace_segments))` |
| curved_intersections.py:2064 | qualified_face_face | ordering/select-by-float-key | `np.argmin(np.linalg.norm(flat_seed_points - point, axis=1))` |
| curved_intersections.py:2111 | qualified_face_face | tolerance/tolerance-compare | `residual <= tolerance` |
| curved_intersections.py:2115 | qualified_face_face | tolerance/tolerance-compare | `float(np.linalg.norm(crossing[0] - item[0])) <= tolerance` |
| curved_intersections.py:2126 | qualified_face_face | ordering/select-by-float-key | `min(range(len(remaining)), key=lambda index: float(np.linalg.norm(start[0] - remaining[index][0])))` |
| curved_intersections.py:2147 | qualified_face_face | ordering/min-max-value | `max(0.0, minimum - tolerance)` |
| curved_intersections.py:2149 | qualified_face_face | ordering/min-max-value | `max(tolerance, extent * 1.0e-10, np.finfo(float).eps)` |
| curved_intersections.py:2159 | qualified_face_face | ordering/min-max-value | `max(float(item[3]) for item in path)` |
| curved_intersections.py:2165 | qualified_face_face | ordering/min-max-value | `max(float(np.linalg.norm(first_normal) * np.linalg.norm(second_normal)), np.finfo(float).tiny)` |
| curved_intersections.py:2166 | qualified_face_face | tolerance/tolerance-compare | `alignment >= 1.0 - model.tolerance.angular` |
