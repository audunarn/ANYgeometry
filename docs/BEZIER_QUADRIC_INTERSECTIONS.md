# Bezier walls of degree three and more against quadrics

Development head (not yet requalified for release). A wall `c(t) + s d` whose planar profile is a Bezier
curve of degree three or more is a ruled surface that is **not** a quadric, so its pairs with cylinders,
cones and quadric extrusions cannot use the angular [`QuadricIntersectionCurve`](QUADRIC_INTERSECTIONS.md).
This page describes the polynomial-chart counterpart, `BezierQuadricCurve`, and what the rest of the engine
needed so that such a wall takes part in `plan_intersections`, `apply_intersections`,
`query_trimmed_surface_charts` and the public query and imprint workflow like every other analytic face (see
[extruded polycurve faces](EXTRUDED_SURFACES.md)).

## The equation

Substituting the ruled surface into a quadric `Q` gives

    Q(c(t) + s d) = A s^2 + B(t) s + C(t) = 0

with a **constant** `A = d . M . d`, `B` of degree `n` and `C` of degree `2n`, all exact rational polynomials
of the stored doubles (`BezierRuledSupport`, `PolyPlan`; one plan per support pair, shared by every patch).
A branch is `s = (-B +- sqrt(B^2 - 4 A C)) / 2A` over an interval of the directrix parameter `t`. The chart
parameter *is* `t`, so nothing wraps around, and nothing has a pole unless `A = 0` (a ruling parallel to an
asymptotic direction of the quadric), where the single branch is rational, `s = -C / B`.

## Events

Every event is a real root in `[0, 1]` of an exact polynomial, isolated with the integer Sturm machinery of
the existing engine:

| Event | Polynomial | Meaning |
| --- | --- | --- |
| Fold | simple root of `B^2 - 4 A C` | the ruling is tangent to the quadric: the two branches meet and a chart ends |
| Double contact | double root of the discriminant | two branches cross; charts split there |
| Cusp | root of higher multiplicity | refused with a typed error |
| Pole | root of `B` (linear branch only) | the chart runs to infinity and ends |
| Whole generator | `gcd(A, B, C)` | the ruling lies on the quadric: a segment, not a curve |
| Patch limit of the wall | `A s0^2 + B s0 + C` | the branch crosses the ruling station `s0` |
| Boundary of the other patch | resultant with each boundary plane | the branch crosses that plane |

A chart between two events is kept when its midpoint lies in both patches. A quadric that touches the wall at
one point gives that point exactly.

## `BezierQuadricCurve`

Root-exported next to `QuadricIntersectionCurve` and following its conventions: one real branch (`branch`
is `-1` or `+1`) over `[start, start + sweep]` of `t`, with `linear`, `left_square`, `right_square` and
`both_sine` parameterizations, so that a chart that ends at a fold stays regular in the chart parameter
there. It evaluates, differentiates twice (with one-sided jets at fold ends), encloses, slices, reverses,
takes exact affine images, finds its roots on planes and quadrics and inverts points.

* **The radical of a fold chart** is `sqrt(Delta(t) - Delta(anchor))`, the discriminant relative to the fold
  that ends the chart. It comes from an exact rational Taylor shift at the stored anchor, so it vanishes
  *exactly* at the fold (a rounded root would leave a residue of order `1e-14`, a square root of which is
  `1e-7`), keeps its relative accuracy all the way in, and gives both branches **one** point at the fold, bit
  for bit.
* **The second root** of the quadratic is computed as `C / q` only where that agrees with the direct form
  `(-B +- radical) / 2A` to a few ulp. Where `B`, `C` and the radical vanish together (a fold lying on the base
  curve, for example a pipe whose axis lies in the plane of the wall's base) the quotient is a ratio of two
  rounding residues and put one end of a chart eight units outside the wall; the direct form decides there.

## Accuracy: the Bernstein basis

`C` has degree `2n`. In the power basis a double evaluation of `C` and of the discriminant loses about one
digit per degree to cancellation; the Bernstein form of `[0, 1]` has only convex combinations of its
coefficients and stays at a few ulp. Largest absolute error against exact rational evaluation (5 random
planar walls per degree with control ordinates in multiples of 1/8, 151 parameters each, against a cylinder
of radius 0.7):

| Degree | `C`, power basis | `C`, Bernstein | discriminant, power basis | discriminant, Bernstein |
| --- | --- | --- | --- | --- |
| 3 | 2.2e-14 | 8.9e-16 | 2.2e-13 | 5.3e-15 |
| 4 | 3.7e-13 | 4.4e-16 | 4.2e-12 | 3.6e-15 |
| 5 | 4.7e-13 | 1.8e-15 | 5.0e-12 | 1.8e-14 |
| 6 | 1.1e-11 | 1.8e-15 | 7.6e-11 | 1.1e-14 |
| 7 | 1.3e-10 | 3.6e-15 | 7.8e-10 | 2.5e-14 |
| 8 | 7.7e-10 | 1.3e-15 | 9.1e-09 | 1.1e-14 |

All branch polynomials, their derivatives, the directrix and the fold-relative radical are kept as
`BernsteinForm` objects with coefficients rounded once from the exact rationals and evaluated by the
Bernstein-Horner scheme. A joint edge of a whole model lies on both of its supports to about `1e-15`, where
the power basis gave `2e-14` for a cubic wall and a seam-snapping failure for a quartic one.

## Enclosures

`bounds(a, b)` is an outward-rounded axis-aligned box, from the Bernstein coefficients of every polynomial
over the piece (two de Casteljau splits and a pad of `16 (m + 2) eps` times the largest coefficient) and
interval arithmetic for `s`. The radicand of a fold chart is one polynomial, so its range carries none of the
dependency a product of ranges would. The boxes are valid down to the tolerance and at the fold ends (tested
at every width from the whole chart to `1e-12`), tight to a small factor on the whole curve and first-order
tight on a slice (interval arithmetic cannot see the cancellation between the directrix and the ruling). They
feed projection, the clipping of charts and the proofs below.

## Junctions with the other curves

The arrangement asks for every shared point of two curves. A Bezier branch lies on a quadric and on a wall
that is not one, so the elimination used for quadric branches ("the other curve's roots on both supports")
has only its quadric half; that half is enough whenever the other curve is not contained in the quadric,
because every shared point is a root of the other curve on it and the point inversion qualifies it.

| Other curve | Method |
| --- | --- |
| Line, ellipse | exact roots of the branch on the plane(s) through it |
| Bezier path | exact roots of the path on the branch's quadric, then point inversion; a path drawn on that quadric: certified subdivision |
| Branch of two quadrics, cylinder/cylinder branch | exact roots of the Bezier branch on their supports, then point inversion on theirs |
| Bezier branch, other quadric | exact roots of one on the other's quadric (both orders agree) |
| Bezier branch, same wall, quadric and image | charts of one curve: they meet only at chart ends, which the arrangement names itself |
| Bezier branch, same quadric, another parallel wall | rational elimination of both projected directrices, including repeated roots; qualify every pair of branch parameters |
| Bezier branch, same quadric, other support configuration | certified subdivision |

**Certified subdivision** (`branch_events.subdivision_junctions`) cuts both parameter intervals at the midpoint
of the longer box until the boxes are smaller than an eighth of the tolerance. Two boxes farther apart than
the tolerance cannot contain a shared point, so curves that never meet are *proved* disjoint and a crossing
is bracketed to the tolerance and polished by Newton iteration. A pair that cannot be resolved within a fixed
budget (a tangential or overlapping contact) is refused with a typed error rather than guessed.

Parallel-wall elimination retains the isolated root intervals through each branch
chart and its conservative world bounds. Only separated enclosures exclude a
candidate pair. A returned pair must meet the requested world-space tolerance;
an unresolved candidate raises a typed error. This includes distinct visits to a
self-crossing projected directrix. Common projected components retain the
existing overlap/refusal path. No subdivision budget was increased.

## Roots and point inversion near a fold

Near a fold end `t -> tau` is a square root: a crossing a few ulp of `t` from the fold keeps only a few digits
of `tau` and moved the point by `1e-9`, so the root was dropped by the tolerance test and a face boundary
lost its winding crossing ("arrangement failed material conservation"). The curve is regular and accurately
evaluated in `tau`, so every root on a quadric and every point inversion is polished by a few Newton steps in
the chart parameter, and a point within the tolerance of a chart end is reported as that end.

## Documents

A stored branch is the record `bezier_quadric_intersection`: the wall by its control points and ruling
direction, the quadric by its defining data, the chart and the affine image. **Schema 6** is written when a
document stores one (as for a `QuadricIntersectionCurve` or an `ExtrudedSurface`); a reader below 6 refuses
the record, and an unknown or missing field refuses the whole document.

Simple-fold endpoints require square/sine charts. Tolerated endpoint roundoff
outside the directrix domain is canonicalized before evaluation and bounds.
Point-inversion caches do not participate in analytic plan content binding;
geometric definitions still do. Reusing an applied plan remains idempotent.

## Subsequent cuts and attachments

Batch preparation explicitly remaps attachments when splitting an earlier joint.
For a point at original parameter `t`, the owner evaluates `P = C(t)` and finds
the unique parameter of `P` on the retained child curve. It does not assume the
child parameter is `(t-a)/(b-a)`: fold-regularized charts can change that map.
Intervals are partitioned at the split and expressed in the exact child charts.
Vertex/member stations, sheet relations, attachment lineage, junction attachment
lists, and owner-held member source/target/junction ranges are retained.

Direct `model.split_edge(edge, t)` still refuses attached targets by default.
`model.split_edge(edge, t, remap_attachments=True)` selects the explicit atomic
remap and exact rollback, including identifier state. Ambiguous inversions or
unqualified source-edge attachment intervals fail closed. Consumers use the
updated owner attachment records; they must not infer replacement parameters.
Detached preparation keeps the authored model unchanged.

Trim-boundary projection polishes a Bezier branch candidate only after its global
distance certificate is established, and accepts the polish only when closer.
This retains exact shared mesh stations without changing projection tolerances
or the consumer's shared-station and cell-quality checks.

Portable installed-consumer builders and authored/prepared document export are
in `tools/general_intersections/consumer_contract_fixtures.py`. They cover an
oblique cubic wall/pipe with fold-ending branches, a parabolic wall/pipe, tangent
and nearby secant/separated plate/pipe cases, and a second cut with persistent
vertex/member attachments. The manifest records expected station coordinates,
joint identities and checksums. Source commit and wheel identity remain required;
the development version alone does not identify this contract.

## Limits

* Two non-parallel Bezier walls of degree three or more are refused (`unsupported`); a quadratic profile is a
  quadric and goes to the angular engine.
* Parallel-wall isolated tangencies use exact elimination. Overlapping branch
  components and other unresolved support configurations retain typed refusal.
* A fold of higher multiplicity (a cusp of the discriminant) is refused.
* Source-edge attachment intervals without a qualified source map, or a station
  with multiple admissible inverse parameters, are refused atomically.
* Plane/pipe generator tangencies preserve the cylinder support's geometric
  tangency identity before forming rounded trigonometric coefficients. Nearby
  secants and separated supports keep their classification and tolerances.

## Cost

The exact solve is per support pair, so the facets of a pipe against the facets of a wall cost one plan plus
a lookup per facet pair. Measured on this machine (best of three runs, support plans, enclosure and root
memos emptied before each run; plan and apply of the whole shell; pipes of radius 0.12 at staggered
heights across a wall with the ruling `(.25, 0, 1.5)`):

| Shell | Plan, cold | Apply |
| --- | --- | --- |
| Cubic spline wall x 1 pipe of 8 facets | 0.99 s | 0.58 s |
| Quadratic spline wall x 1 pipe of 8 facets (angular charts, for comparison) | 0.63 s | 0.38 s |
| Cubic spline wall x 4 pipes of 8 facets | 2.77 s | 1.49 s |
| Quadratic spline wall x 4 pipes of 8 facets (angular charts, for comparison) | 2.43 s | 1.31 s |

The polynomial charts plan in 1.6 times the time of the angular ones for one pipe and 1.1 times for four.
A profile of the first implementation put three costs first, and three changes address them without
altering a result: a scalar directrix inversion (`BezierDirectrix.invert_one`, the same Newton iteration
without numpy per step), point inversions remembered per curve (about half of the queries of one
arrangement repeat a vertex) and a plain-float path of the Bernstein forms for the one-element arrays the
arrangement evaluates tens of thousands of times.

## Evidence

`tests/test_branch_algebra.py` (the exact polynomials against exact quadric values at dyadic points, the
events against brute force, the Bernstein forms against rational evaluation from degree three to eight),
`tests/test_branch_curves.py` (the curve on both supports, derivatives against Richardson differences,
enclosures at every width and at fold ends, roots, inversion, slicing, affine images, folds on the base
curve and a plane a hair from a fold), `tests/test_branch_supports.py` (branches against an independent grid
oracle and, for a quadratic wall, against the established angular engine), `tests/test_branch_events.py`
(every junction family against an independent distance-grid oracle in both orders, certified subdivision) and
`tests/test_branch_models.py` (whole models: topology, exact area conservation, joint residuals, determinism,
the document round trip, an affine transform, the public query and imprint, iterative modelling and the
refusals).
