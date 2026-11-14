# Plane x NURBS exact pre-pass: soundness argument (proposal 10, narrow slice)

Module: `nurbs_surface_intersection/_plane_prepass.py`. Opt-in only: `qualification.plane_prepass is True`
lets `query_nurbs_surface_intersection` use it for the DISJOINT decision; `plane_nurbs_prepass(a, b)` returns
the full typed `PrepassResult` (DISJOINT / DECIDED with components / DECLINED with a typed reason).
Everything is exact `Fraction` arithmetic. DECLINED always means "use the unchanged atlas route".

## Class admitted
One operand is a full-range bilinear parallelogram (2x2 net, uniform weights, non-degenerate): a plane patch
`P`, `F(u,v) = n . S(u,v) - c` on the other operand `S` (uniform positive weights, full knot ranges, interior
knot multiplicity <= degree so `F` is continuous). Every control point of `S` must project (orthogonally)
into the closed patch `P`; the projection of the hull contains the projection of every point of `S`, so every
point of `S` on the plane lies in `P`: the plane's own boundary can never trim the section.
The section is then exactly `{F = 0}` on the domain of `S`.

## Cell certificate
Per Bezier cell (dyadic sub-box of a knot span, Bernstein grid by exact blossoming):

1. **EXCLUDED** if all Bernstein coefficients of `F` are strictly positive or all strictly negative
   (convex-hull property: `F != 0` on the cell). A zero coefficient never excludes.
2. **Monotone certificate**: `t = (F_v, -F_u)` is the tangent field of the level curve. If for one direction
   `d` all Bernstein coefficients of `d . t` (after exact degree elevation to a common degree) are strictly
   one-signed, then `d . t != 0` on the cell, so `grad F != 0` (regular, no tangency or singular point) and
   along every component of `{F = 0}` the coordinate `d . x` is strictly monotone (its derivative along the
   curve is `d . t`). Hence there is no closed loop inside the cell.
3. **Edge roots**: for each cell edge, `F` restricted to its line is a univariate polynomial, required nonzero,
   squarefree (Sturm chain, `gcd(p, p')` constant) and nonzero at both segment ends. Its simple roots in the open
   segment are counted exactly by Sturm variations.

## What is proven per cell, and what is declined
A component of `{F = 0}` inside a closed cell that satisfies 2 is a compact regular 1-manifold without loop; its
boundary points lie on the cell boundary, they are transversal crossings (all edge roots are simple, none at a
corner), and the curve cannot touch the boundary without a root (a tangential touch would be a double root).
So every component is an arc with exactly two distinct boundary roots, and the number of arcs is `roots / 2`.

* **0 roots**: no arc, so `{F = 0}` is empty in the cell (an arc needs two roots). Accepted as empty.
* **exactly 2 roots**: exactly one arc and the pairing is forced. Accepted.
* **4 or more roots** (or an odd count, which cannot occur for simple roots with no corner root): the pairing
  across monotone arcs is NOT claimed. The cell is split and re-tested; at the depth cap (10) the whole
  pre-pass declines (`DEPTH_CAP`). Nothing partial is ever returned.

## Stitching across cells
Roots are identified on (axis, absolute line, transverse span) by canonical dyadic isolating intervals from
one Sturm bisection per line polynomial (depth = deepest leaf + 1). A cell edge segment is a dyadic interval
of depth <= deepest leaf, so each root interval lies wholly inside or wholly outside it; a partial overlap, an
endpoint root, a multiple root or two different polynomials on one shared line DECLINE (`ROOT_IDENTITY`,
`CORNER_OR_DYADIC_ROOT`, `MULTIPLE_EDGE_ROOT`). Because the same line polynomial serves both neighbours (also
across T-junctions of different depth), two cells name the same root identically.
Every accepted cell contributes one arc between its two roots. The graph is accepted only if each interior
root has exactly two incident arcs and each root on the domain boundary exactly one (`ROOT_DEGREE`
otherwise). Then each component is a path (between two boundary roots) or a cycle, and these are exactly the
connected components of `{F = 0}`: leaves tile the domain, an excluded or empty cell contains no zero, and
every zero lies in some accepted arc.

DISJOINT = every leaf excluded or empty.

## Not claimed
No claim for tangent, multiple-root, corner-root, coincident or trimmed cases (all decline), for rational
(non-uniform weight) surfaces, for non-plane pairs, or about the atlas' own records: a decided result carries
exact root intervals and arcs in the NURBS operand's native chart, not witness or scalar-box records, so it is
not (yet) usable by the arc evaluators; the query hook therefore uses it for DISJOINT only.
