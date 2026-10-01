# Making intersections more general: cone against cylinder, and beyond

Study record, 2026-10-01. Base: `main` 9dbf3e4 (0.4.5) plus the unmerged performance branch
`claude/perf-hunt-046`. Branch: `claude/quadric-study`. Nothing here changes a released contract,
and nothing should merge before the testing-regime changes settle.

**Question.** A cone meets a cylinder, say with its axis 10 degrees off perpendicular. How can the
intersection engine become general enough to handle that, and what else would the same change buy?

## Answer in brief

1. **Today the pair is refused.** `plan_intersections` stops at the first cone face with
   `face N has no qualified analytic or bilinear support`. The legacy certified engine
   (`query_intersection`) classifies the pair but is not a usable route (section 1).
2. **The structure generalizes cleanly.** Plane, Cylinder and Cone are all quadrics, and Cylinder and
   Cone are ruled: `S(t, s) = P(t) + s D(t)`. Along a ruling a second quadric gives
   `A(t) s^2 + B(t) s + C(t) = 0`. That is exactly the structure `CylinderIntersectionCurve` already
   uses, with a degree-8 tangent-half polynomial for the events. A prototype reproduces the production
   cylinder charts and extends to cones with verified completeness and accuracy.
3. **The chart side is small.** Letting an arrangement domain be a Cone face is about 20 changed lines
   in `material_arrangement.py`; the arrangement itself is chart-agnostic.
4. **The curve must stay analytic.** The same arrangement is 225 to 700 times slower when the exact
   curve is replaced by a fitted Bezier chain. A numeric or Bezier fallback is not a route to
   generality inside this engine.
5. **Recommendation.** Add a generalized quadric branch curve as a new family beside
   `CylinderIntersectionCurve` (leave the validated cylinder code alone), starting with
   cone x cylinder and cone x plane. Decisions that belong to the owner are in section 6.

## 1. What the kernel does today (measured)

Test model: a 12-facet cylinder (radius 2, axis z) and an 8-facet cone frustum (radius 0.5 to 1.0
over 5 units) whose axis is perpendicular, or 10 degrees off perpendicular, through the cylinder axis.

| Path | Plane/Cylinder pairs | Cone pairs |
| --- | --- | --- |
| `plan_intersections` (exact batch engine, used by the mesher) | exact; 12 x 12 facets, 10 deg off perpendicular: plan 2.4 s, apply 0.44 s | refused, see above |
| `query_trimmed_surface_charts` | Plane, Cylinder | `unsupported intersection surface` (the mesher already checks for this) |
| `query_intersection` (legacy certified subdivision) | n/a | 8 of 96 facet pairs cross at both 0 and 10 deg; 9.0 s for all 96 (0.094 s per pair) |
| `apply_imprint` (legacy) | n/a | fails on the first crossing pair at both angles: `imprint component does not map to one active face descendant` |

The legacy engine is also fragmentary here: 3 of the 8 crossing pairs at 10 degrees (4 at 0) return 4 to
14 components, mostly two-point pieces with interior ends. What it persists is quadratic Bezier pieces within the
1e-8 curve-fit residual. ANYmesh already meshes an isolated cone face with an exact developable chart
(`_conical_chart.py`) and states that cone intersections are outside the batch contract, so the gap is
on the geometry side.

Where generality stops in the code: `_pair_paths` is an `if/elif` on (Plane, Cylinder); curve families
(`LinePath`, `EllipticArc`, `CylinderIntersectionCurve`, `BezierPath`) are matched by `isinstance`
chains in `plane_roots`, `_point_parameters`, `_curve_junctions`, `MaterialDomain.curvature`,
`_coincident` and `freeze_edge`. The general engine's cylinder-specific footprint is roughly 60 references
in nine files; the 2,700 lines of cylinder atlas/patch qualification are historical and separate.

## 2. The structural idea

For a ruled first support and any implicit quadric `x.M.x + 2 l.x + c = 0`:

* `A = D.M.D`, `B = 2 (P.M.D + l.D)`, `C = P.M.P + 2 l.P + c` are trigonometric polynomials of degree
  at most 2, built exactly from the doubles (Cylinder: constant `D`; Cone: constant `P`, the apex).
* The branches are `s = (-B +- sqrt(B^2 - 4AC)) / (2A)`. Folds are roots of the degree-4 discriminant
  (odd multiplicity), double contacts are even multiplicity, poles are roots of `A`, and finite-patch
  boundaries are roots of `A s0^2 + B s0 + C` and of resultants against the other support's end and seam
  planes. All are Sturm-isolated with the existing `isolate_real_roots`.
* Curve events against any other quadric are the resultant of two such quadratics: again a degree-8
  tangent-half polynomial for Cylinder and Cone supports.
* Sphere, ellipsoid and similar quadrics are free **as the second support**. The ruled partner supplies
  the angle. A torus is quartic and is not covered.

Choice of the angle-supplying support matters for regularity. For the 10 degree case cone-first has one
smooth closed loop (four charts, cut only at seam events; no folds, no poles); cylinder-first folds twice
per loop (eight charts, four folds). The persisted curve should use whichever has fewer folds.

## 3. Prototype and evidence

`src/anygeometry/_quadric_branch.py` (private, not exported, no serialization) and
`tests/test_quadric_branch_prototype.py`. Reproduction scripts are in
`tools/general_intersections/quadric_study/`.

| Claim | Evidence |
| --- | --- |
| Reproduces the production cylinder charts | 5 configurations x 64 facet pairs (perpendicular, equal radius, skew, tangent, 10 deg): identical chart sets (angular interval and branch) in all 320 pairs; point difference at equal angles at most 2.6e-14. The singular equal-radius perpendicular case is included. |
| Complete | An independent grid oracle (contour of the implicit equation on a dense chart, no use of A, B, C or events) over 6 pairings x 150 random configurations: about 247,000 contour points, 1,700 charts, 831 folds, 88 poles, none uncovered away from fold angles. Pairings: cone/cylinder both orders, cone/cone, cone/plane, cylinder/plane, cylinder/cylinder. |
| Accurate | Every chart lies within 1e-13 (distance) of both surfaces in the 10 degree case; closed loop closes to 1e-13. Random pairings stay below 1e-10 (distance estimate; floating-point noise near a cone apex makes this estimate pessimistic there). |
| Events against another quadric | 471 charts against random planes, cylinders and cones: 240 exact roots, 240 sign changes found by dense sampling, none missed. About 64 ms per chart and quadric in unoptimized Python. |
| Sphere as second support | 60 random cone/cylinder x sphere pairs: 178 charts, 58 folds, no failures. |
| Fast enough | All 96 cone x cylinder facet pairs: 0.4 to 1.1 s (4 to 11 ms per pair), versus 9.0 s for the legacy engine, and exact. |
| Cone as a material chart | `MaterialDomain` accepts Cone faces with about 20 changed lines. Generator, ring and combined cuts on upright, 10 degree and generic-tilt cone facets give the exact product areas. |
| Curve class matters | Same arrangement, same cylinder x cylinder facets, same tolerances: analytic traces 0.09 to 0.22 s per face; degree-5 Bezier fits (1e-9) 25 to 120+ s per face, 225x to 700x slower (one case was cancelled at the 120 s budget). A Bezier-trace cone x cylinder arrangement at 10 degrees did succeed on all 8 cone facets (two cells each, area conserved) but took 28 to 44 s per facet. |

Fail-closed behavior in the prototype (typed `GeometryError`, never a guess): cone apex on the other
support within tolerance (the exact structure then depends on rounding of the apex position), contact
of multiplicity three or more, and rulings parallel to a plane for every angle.

## 4. Options

* **A. Numeric or Bezier fallback inside the batch engine.** Rejected by the data above: the legacy
  provider is fragmentary and persists 1e-8 fits, and Bezier curves make the exact arrangement two to
  three orders of magnitude slower.
* **B. Exact generalized quadric branch (recommended).** Covers Cone now and Sphere or any quadric as a
  second support later. Cost: a new curve class, a geometry schema bump (6), junction dispatch, chart and
  `_child_support` work, qualification, and consumer coordination.
* **C. Revolution and freeform surfaces (torus, profiles, Ruled/Coons/NURBS).** Needs certified numeric
  surface-surface tracing and an arrangement that can use numeric curves without exact resultants. A
  separate project; B does not block it.
* **D. Architecture.** Define a small curve protocol (evaluate, derivative, certified bounds, subcurve,
  roots on a quadric, parameters of a point) and a support registry, so the next family does not edit six
  `isinstance` chains. Worth doing with B, not before it.

## 5. What a full slice for the cone still needs (not done here)

Derivatives and second derivatives including one-sided limits at fold ends; certified interval bounds;
`subcurve` and affine `transformed`; `point_parameters`; junction dispatch for branch x {line, ellipse,
branch, Bezier}; `_pair_paths` for Cone; `_child_support` for Cone; exact cone area in
`query_trimmed_surface_charts` (the current `area_jacobian` for a Cone is only an upper bound); beam x
cone supports in `member_arrangements`; serialization and `EXACT_CURVES`; the rulings case (parallel
supports give generator lines); a qualification matrix (perpendicular, 10 degrees, skew, tangent,
coaxial, wide cone, apex cases, cone x plane, cone x cone); and mesher consumption of cone joints. For
scale: the analogous cylinder code is `exact_curves.py` (588 lines), `cylinder_curve_events.py` (144),
`quadric_curve_events.py` (67) and `analytic_supports.py` (220). Arrangement speed with an exact cone
curve is expected to match the cylinder analogue (same polynomial degrees) but was not measured.

## 6. Decisions for the owner

1. Schema 6 and a new curve class beside `CylinderIntersectionCurve`, or migrate the cylinder class later.
2. Scope of the first slice: Cone only, or also a bare-quadric second support (sphere) with its charts.
3. Policy for apex contact, tangency and higher-order contact: refuse with typed errors (as prototyped)
   or support.
4. Rule for choosing the angle-supplying support (fewest folds).
5. Who owns conical joint meshing in ANYmesh once geometry provides cone charts and joints.
6. How the new family is qualified under the testing-regime changes now in progress.

## 7. Other observations

* 16 or more mutually crossing plates through one common point fail with
  `arrangement failed material conservation` (12 and 14 pass; 20 plates in generic position plan and apply
  in 187 s and produce 16,139 faces). Not investigated; reproduction is `quadric_study/mutual2.py`.
* The prototype's distance-to-surface estimate is meaningless at a cone apex (the gradient vanishes);
  checks skip that point.

## Reproduction

```
python -m pytest tests/test_quadric_branch_prototype.py tests/test_cone_material_arrangement.py
python tools/general_intersections/quadric_study/qb_fuzz2.py tools/general_intersections/quadric_study 202 150 cone:cyl
python tools/general_intersections/quadric_study/qb_vs_old.py tools/general_intersections/quadric_study
python tools/general_intersections/quadric_study/curve_repr_cost.py tools/general_intersections/quadric_study 5 1e-9
```
Each script takes the directory containing its siblings as its first argument. Full suite on this
branch: 10 failures, all in the git-environment release-authority tests that also fail on `main`
(16 on the earlier run; the set varies with the environment), no new failures.
