# Cone supports and quadric intersection branches

Development head (not yet requalified for release). A Cone face now meets Planes, Cylinders and Cones
through the same exact material-arrangement engine as Plane and Cylinder faces. The intersection of a
cone with a cylinder at an angle (for example 10 degrees off the perpendicular) is not a conic, so the
engine gets one new exact curve family instead of a sampled fit.

## Supported pairs

| Pair | Exact result |
| --- | --- |
| Cone x Plane | Perpendicular plane or ellipse: `EllipticArc`. Parabola and hyperbola branches: `QuadricIntersectionCurve`. Plane through the apex: the generators it contains as line segments (or the apex as a point contact). Tangent plane: its generator |
| Cone x Cylinder | Coaxial: circles (`EllipticArc`). Otherwise `QuadricIntersectionCurve` charts, folded or closed, at any angle and offset |
| Cone x Cone | Coaxial: rings. Same surface (any patch): coincident, boundary traces only. Otherwise `QuadricIntersectionCurve` charts |
| Beam or other member axis x Cone | Exact line/curve roots on the cone's implicit equation |

`query_intersection(face, face)` still answers cone faces through the established certified (sampled)
engine, exactly as it does for cylinder faces; the exact results above are the
`plan_intersections` / `apply_intersections` / `query_trimmed_surface_charts` workflow.

## `QuadricIntersectionCurve`

Root-exported, immutable. One real branch of the ruling equation over an angular chart. The first
support (a Cylinder or Cone) supplies the angle `t` about its axis and is ruled, `S(t, s) = P(t) + s D(t)`;
the second is any quadric (Plane, Cylinder, Cone or a general quadric). Substituting the ruling into the
second implicit equation gives `A(t) s^2 + B(t) s + C(t) = 0` with trigonometric coefficients of degree at
most two, so a branch is `s = (-B +- sqrt(B^2 - 4AC)) / (2A)` (a single rational branch against a plane).

* Fields: the two supports, `start_angle`, `sweep_angle`, `branch` (`-1` or `1`), `parameterization` and an
  affine `transform` (an affine image is kept exactly).
* Charts never contain a fold or a double contact of the discriminant in their interior. A chart end at a
  simple fold uses a square (`left_square`, `right_square`) or sine (`both_sine`) reparametrization so the
  curve, its first and its second derivative stay regular there.
* Operations: `evaluate`, `derivative`, `second_derivative`, `bounds` (interval enclosure), `subcurve`
  (including reversal), `transformed`, `roots_on(surface)` (exact, with `None` when the chart lies on the
  surface) and `parameters_of(point)`.
* Serialization: **schema 6 is written only for a document that stores a `QuadricIntersectionCurve`**; every
  other document stays schema 5, so schema-5 readers are unaffected. Readers accept schemas 1-6; a schema
  below 6 refuses the new record.

## How a support pair is solved

* All exact work depends on the support pair only and is shared by every facet (`quadric_algebra.Plan`,
  keyed by the two supports without their patches): the homogenized integer polynomials in `x = tan(t/2)`
  for `A`, `B`, `C`, the discriminant and the resultants with patch boundaries, their real roots by Sturm
  isolation and multiplicities by squarefree decomposition. A cone's apex is an exact rational point, which
  halves the degree of the resultants.
* Events that cut a chart: simple folds, double contacts (nodes), poles of the branch, the axial patch
  heights, and the boundary planes of the other patch. Each interval is kept when its midpoint lies in both
  native rectangles.
* Contact of third or higher order (a cusp of the discriminant) is refused with a typed error.
* Exact classification follows the stored doubles, as for plane/cylinder: a plane that is parallel to a
  generator in the stored data is a parabola, one off by an ulp is an ellipse or a hyperbola.

## Material charts

A cone's area element grows with its radius, so its material area is `-integral(F(v) du)` over the exact
boundary loops with `F` the antiderivative of the radius, never a native-area estimate times a constant.
Applying a cut turns each rectangular piece of a cone facet (rings and generators) into its own exact smaller
`Cone` patch, as it does for a Cylinder; general trims keep the parent support with native-chart loops.

Loop areas use Gauss rules of doubling size and stop at the rounding noise of their own sums; when a branch
point just outside a path makes them converge slowly, bisection toward it resolves the same integral.

## Tangential and degenerate contacts

A trace that touches a facet boundary tangentially splits into a pair of crossings as far apart as the
square root of the data's rounding, many times the tolerance. Junction pairs that are close along both curves
while the curves coincide within tolerance are merged into one contact (the curve end, else the mean). A
boundary arc rebuilt from its end points can carry a centre error far above an ulp; its containment in its
own support is bounded on the arc's angular range rather than over the full circle.

## Cost

The exact solve is per support pair, so a shell pair costs one plan plus a dictionary lookup per facet pair.
Exact curve/quadric roots are remembered, lines are solved by their exact discriminant, and a plane meets an
ellipse (every ring and plane test) through the half-angle quadratic instead of Sturm isolation of an
equivalent quartic. Measured on this machine (12 x 8 facets, 10 degrees off the perpendicular), planning the
cone/cylinder pair took 0.8 s against 1.9 s for the equivalent cylinder/cylinder pair, before applying.

## Evidence

`tests/test_quadric_intersection_curve.py` (curve contract), `tests/test_quadric_supports.py` (independent
contour oracle, conic classification, coaxial/coincident pairs, shared plans),
`tests/test_cone_intersections_engine.py` (whole shells through plan/apply: topology, exact area
conservation, joint-edge residuals, tangent and folding contacts, beams) and
`tests/test_serialization_quadric.py` (schema policy).
