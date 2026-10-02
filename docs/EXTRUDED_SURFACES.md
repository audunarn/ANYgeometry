# Extruded polycurve faces and elliptic cylinders

Development head (not yet requalified for release). A face made by `GeometryModel.extrude` from a spline
or an oblique circular arc is an exact translational surface, `c(t) + s d`: a planar profile `c` swept
along a vector `d`. The general intersection engine now treats it as one, so such a face meets plates,
cylinders, cones and other extrusions through the same exact material-arrangement workflow as every other
analytic face, and a face whose profile is an ellipse (an oblique circular tube is one) is a ruled quadric
that enters the quadric branch engine of [cone and quadric intersections](QUADRIC_INTERSECTIONS.md).

## `ExtrudedSurface`

Root-exported with `BezierDirectrix` and `EllipseDirectrix`. A directrix is a planar Bezier curve (any
degree, control points in one plane) or an ellipse arc, `center + u cos a + v sin a`. The chart of a patch
is `(u, v)` in `[0, 1]^2` with `t = t0 + u (t1 - t0)` along the directrix and `s = s0 + v (s1 - s0)` along
the vector, so every patch cut from one surface shares its directrix and vector (`support_key()`) and
differs only in `u_range` and `v_range`.

Because the profile is planar, the extrusion coordinate `s = n . (p - o) / (n . d)` is linear in position:
its level lines are plane sections, exactly like a cylinder's rings, and the directrix parameter of a point
is that of its projection `p - s d`. Chart rates, accelerations and area density follow in closed form.

A face is an extruded face when it **stores** an `ExtrudedSurface` or when `extruded_support` **recognises**
its topology: four edges, a Bezier or elliptic bottom, a translated top and two equal parallel straight
connectors. Recognition changes nothing the model stores; the `extrude()` result keeps its Coons patch.

## Supported pairs

| Pair | Exact result |
| --- | --- |
| Extrusion x Plane | Bezier directrix: a `BezierPath` of the same degree, an exact affine image of the directrix restricted to the patch. Ellipse: an `EllipticArc`. Plane parallel to `d`: the generators through the roots of `n . c(t) = h` |
| Extrusion x extrusion, same direction | Generators through the crossings of the two directrices carried into one plane |
| Elliptic extrusion x Cylinder, Cone or elliptic extrusion, any direction | `QuadricIntersectionCurve` charts; rulings that agree to rounding give the shared generators; exact tangency gives a point; the same surface gives coincidence |
| Quadratic Bezier extrusion (a spline with one control point) x Cylinder, Cone or elliptic extrusion, any direction | The same, with the parabolic cylinder as the second support |
| Quadratic x quadratic Bezier extrusion of another direction; Bezier extrusion of degree three or more x Cylinder, Cone or extrusion of another direction | Not supported: a typed `GeometryError`, never an approximation |

A quadratic Bezier is a parabola, so its extrusion is a quadric and needs nothing new. A cubic or higher
profile is not, and its pair with a quadric needs a polynomial-chart counterpart of `QuadricIntersectionCurve`
(the branch `s = (-B +- sqrt(B^2 - 4AC)) / 2A` has a constant `A` along such a directrix); that is the open step.

## The elliptic cylinder as a quadric

An elliptic extrusion with the unit ruling direction `a` has the exact implicit form `w . M . w = 1` with
`M = r1 r1^T + r2 r2^T` and `r1`, `r2` the first two rows of the inverse of `[u v a]`, `w = p - center`. It is
kept in the engine as the quadric kind `elliptic`, defined by its stored doubles (`origin`, `axis`,
`u_vector`, `v_vector`) and expanded exactly on demand, so tangency and parallelism survive whichever of the
two supports supplies the angle. A cylinder is divided by the exact norm of its stored axis for the same
reason: its null direction is the stored axis itself.

Rulings that agree to `64 eps` are made exactly parallel before the plan is built (two directions from one
vector normalized twice differ in the last bits and would otherwise cross far outside any patch), including
when the float cross product of the two is exactly zero but the doubles are not equal. The change in the
support is below `64 eps` of its length.

## The parabolic cylinder as a quadric

The extrusion of the quadratic Bezier with control points `P0, P1, P2` is `X(t) = P0 + 2 t e + t^2 f`
swept along `d`, with `e = P1 - P0` and `f = P2 - 2 P1 + P0`. A point `p` projected along `d` onto the plane
of the parabola has coordinates `(a, b)` in the basis `(e, f)`, both linear in `p`, and lies on the surface
exactly when `a^2 = 4 b`; the parameter is `t = a / 2`. The engine keeps this as the quadric kind
`parabolic`, defined by the three stored control points and `d` and expanded exactly on demand. A parabola
has no angular chart, so the parabolic cylinder is always the *second* support; the other surface (a Cylinder,
a Cone or an elliptic extrusion) supplies the angle.

The patch of a parabolic cylinder is tested in its own unclipped coordinates (`t = a / 2`), because the inverse
of a Bezier directrix clips to the patch and would hide every point of the parabola beyond its ends. One plane
parallel to `d` through the two end generators holds both seams: a line meets a conic in two points at most, so
that plane meets the whole quadric in exactly those generators (a tangent plane would be degenerate: a rounding
error of its normal separates it from the surface altogether).

## Faces in the public workflow

`query_intersection` classifies a pair that includes a stored or recovered extrusion with the exact material
engine: one exact curve where the certified subdivision returned many sampled pieces (a plate across a spline
wall used to give about ten). `plan_imprint` and `apply_imprint` then plan and apply it through the batch engine.

* **Kept as it was:** the complete-boundary-curve CONNECT. A convex support that strictly contains a wall's
  edge keeps its established `CONTAINED` result and its quadratic partition (the pinned public contract).
* **Exact where it used to refuse:** a wall edge lying in a non-convex support, a trimmed support or leaving
  the trim used to be an explicit `UNCLASSIFIED` refusal; the exact engine now classifies it (the part of the
  curve inside the support) and the imprint is planned and applied with exact material conservation.
* **Fallback:** a pair the exact engine cannot classify (for example two cubic walls of different
  directions) goes to the established certified query, and to its refusal where that refuses too.

`plan_intersections` / `apply_intersections` / `query_trimmed_surface_charts` use the exact engine directly.
A child of a split keeps its exact support with rebased ranges, so a horizontal cut of a spline wall leaves
two exact patches of one surface. A recovered extrusion whose boundary edge is only split by a contact (no
cut through the face) is no longer four edges, so apply stores its exact support on the face instead; its
attachments are remapped by position exactly as for the children of a split.

## Documents

An `ExtrudedSurface` is a stored support, and an elliptic quadric curve names its supports by their defining
vectors. **Schema 6 is written when a document stores an `ExtrudedSurface` or a `QuadricIntersectionCurve`**;
every other document stays schema 5, and a reader below 6 refuses the new records.

## Cost

The exact solve is per support pair, so the facets of a tube against the facets of a pipe cost one plan
plus a lookup per facet pair; a plane section is closed form. Measured on this machine (best of three
runs, support plans emptied before each cold run; plan and apply of the whole shell):

| Shell | Plan, cold | Apply |
| --- | --- | --- |
| Oblique tube of 4 arcs x pipe of 8 facets | 0.65 s | 0.33 s |
| Oblique tube of 8 arcs x pipe of 12 facets | 1.05 s | 0.66 s |
| Spline wall x 4 plates | 0.36 s | 0.31 s |

Three changes keep these low without altering any result: integer pseudo-remainder sequences for the
exact root isolation (the cold plan of the first shell fell from 1.07 s), a conservative segment-pair
filter in topology validation that also rejects collinear stretches of a sampled straight chart edge (the
spline wall applied in 0.82 s), and a Bezier inversion that keeps its sample table and works in the power
basis (the wall planned in 0.54 s).

## Evidence

`tests/test_extruded_surface.py` (the surface), `tests/test_extruded_support_recognition.py` (recognition),
`tests/test_extruded_intersections.py` (charts, plane sections, parallel extrusions and whole models:
topology, exact area conservation, joint-edge residuals, schema), `tests/test_extruded_public_routing.py`
(the public workflow), `tests/test_elliptic_quadric_supports.py` (elliptic supports against an independent
grid oracle, role-swap consistency, degenerate contacts, whole models and the document round trip) and
`tests/test_intersection_performance_equivalence.py` (the integer remainder sequences against their
Fraction oracles).
