# Certified hydrostatics CH3: certified point queries for the curved B1 hull (design, revision 2, 2026-10-20)

Status: design for owner review; no library source changed. Revision 1 (af9aae5) was reviewed by Codex
Astra (4 blocking, 10 major, 1 minor); revision 2 answers every finding (section 12) and corrects the
formulas, not only the wording. Owner decision 2026-10-20: D-CH3-1..10 accepted as recommended (section
11); revision 2 adds one new decision, D-CH3-11, where a fix changes an approach the owner had not
decided. Extends `CERTIFIED_HYDROSTATICS_DESIGN.md` (revision 5, "the planar design") and its
acceptance manifest; it does not replace them. Consumer request:
`C:/Github/ANYstability/docs/ANYgeometry_CERTIFIED_HYDROSTATICS_HANDOFF.md` sections 1, 3 and 5 (Q08).
Working rules: correctness first and fail closed; a typed refusal beats a plausible number; probes find
limits. Facts are marked **measured** (read-only experiments on e26e31f, `PYTHONPATH=src`, Python 3.14.0,
numpy 2.3.4) or **assumed**/derived.

**Scope statement.** CH3 delivers certified half-space point queries (`V`, `J`, `A`, `Mu`, `Mv`, `Iuu`,
`Ivv`, `Iuv`, states, one-sided limits, topology, proof, verifier) for closed shells whose faces are
either the planar faces of the planar design (now also with conic edges, section 2.3) or faces of the
**ruled rational class** of section 2, trimmed by planes, which contains both B1 bows
(`build_b1_half_hull`, `build_b1_hull`). It serves the geometry portion of Q08 and nothing more: no
event/interval API (CH4), no openings or clearances, no general NURBS, no multi-span surfaces (listed
extensions in section 2.4), no change to planar results (planar preparations stay byte-identical,
convention version 1). The consumer's gates (adapter mapping of enclosures, installed verification,
Capytaine comparison, P0/P1) remain theirs.

## 1. Facts this design rests on

Measured on the source (`src/anygeometry/b1_example.py`, `nurbs.py`, `support_curves.py`):

- **B1 bow surface** (`b1_bow_surface`): `NurbsSurface` of degree `(2, 1)`, knots `(0,0,0,1,1,1)` x
  `(0,0,1,1)` (one clamped span each), `u_range = (0, 0.9)`, `v_range = (0, 1)`, controls bottom row
  `(0,6,0), (5.4,6,0), (5.4,0,0)` and deck row `(0,7.5,4), (6,7.5,4), (6,0,4)`, weights
  `((1,1),(w,w),(1,1))`, `w = 0.7071067811865476`. **The two weight rows are equal.**
  `Fraction(w)**2 - 1/2 = +6.8e-17`: the iso-`v` curves are conics, not exact ellipse quadrants.
- **`z = 4 v` exactly in knot space**: with the exact homogeneous net, the bottom row has `z~ = 0` and
  the deck row `z~ - 4 W = 0` as polynomials in the knot parameter `u` (`W` the common weight
  polynomial). (Revision 2: this is a statement in knot parameters; the chart relation through
  `v_range` is a separate affine map, section 2.2.)
- **Rational plane sections**: for a random rational plane, `alpha(u) = n . A~(u) - d W(u)` and
  `beta(u) = n . (B~(u) - A~(u))` are quadratics in `u`, and `S(u, -alpha/beta)` satisfies the plane
  equation with residual exactly `0` (and matches `NurbsSurface.evaluate` to `1e-15`).
- **Corners lie inside the span**: the four corners of the starboard bow face have knot parameters
  `u` in `[0.1749, 0.4446]` (roots of quadratics isolated to `2^-40`), inside `u_range = (0, 0.9)`,
  with `v = 1/8` (bottom) and `v = 3/4` (deck) exactly.
- **Trims and corners**: the half hull has 8 vertices, 12 edges (4 `PlaneNurbsCurve`, 8 `Straight`), 6
  faces; `build_b1_half_hull()` takes 8.4 s. Section-edge `domain` values are **path parameters** of the
  certified section (floats such as `(0.2074…, 0.4939…)`), not knot `u`; the stored corner vertices have
  exactly the frame/plane coordinates in `x` and `z` and a rounded irrational `y`; the curve's float
  evaluation at each domain end equals the stored vertex exactly. On the deck edge, the library's knot
  points at `t = 0, 1/2, 1` have `|v - 3/4| = 0` exactly and `u = 0.4129, 0.2939, 0.1749`; a replaced
  domain `(0.95, 0.99)` gives `u = 0.855, 0.873, 0.891` (still `v = 3/4`): **a `v`-deviation test alone
  does not detect a distant subcurve, the `u`-interval test does** (section 2.3, H2).
- **Algebraic corners are cheap**: a corner quadratic has 108-bit coefficients and isolates to `2^-60`
  in `0.4 ms` (Sturm); an inclined-plane x frame crossing is a quartic with 213-bit coefficients,
  isolated in `< 1 ms`.
- **Exact rational Taylor enclosures work**: for `int_{0.2}^{0.4} p(u) / W(u)^3 du` (`p` of degree 6)
  the complete bound of section 4.3 gives widths `8.9e-8`, `6.9e-12`, `8.0e-16`, `3.1e-24`, `1.6e-32`
  at orders 8, 12, 16, 24, 32, with largest bit lengths 701, 939, 1176, 1657, 2130 and times 3-52 ms;
  one piece over `[0, 0.9]` at order 16 is useless (`3e-2`), 8 pieces give `1e-19` in 0.1 s; a 40-point
  Gauss-Legendre reference agrees to `2e-15`.
- **Review counterexamples, checked exactly** (revision 2): (1) the planar region at `z = 1` between
  `(u, u^2, 1)` and its chord gives `1/18` with the apex `(0, 0, 1)` in the face plane (the chord's
  determinant is `0`); (2) `L = 10^6`, `Q = 1 + u`, `[0, 1]`, midpoint order 8: actual error `1.1332`,
  the complete bound `rho = 195.3` (the pointwise-only bound was `0.0002`); (3) for
  `p = (x^2 - 2)(x^2 - 3)`, `(1, 8/5)` and `(3/2, 2)` each hold one root, their intersection `(3/2, 8/5)`
  holds none, and the ordinals are 2 and 3; (4) the review's chart hexagon has shoelace area `57/200`,
  the Green chain gives `57/200`, the vertical line `u = 0.65` meets it in 4 points (two intervals), and
  clipping by `v <= 1/2` through cut points + vertical-ray parity + Green gives `107/500`, equal to the
  exact polygon clip.
- **H1 (`hydrostatics`) handles the full B1 hull** in 1-2.4 s per plane (settling estimate `~1e-16`):
  upright `z <= 1.5`: `V = 22.569943729716`, `A = 23.319938010999`; `z <= 2.9`: `V = 56.691072446746`;
  `z <= 3.0`: `V = 59.241180737020`; twice the half hull equals the full hull to 12 digits. Heel `0.3`
  through `(2.5, 0, 1.5)`: `V = 25.579680188803` for trims `0, +-0.05, 0.1` alike — geometry, not a
  defect (that plane leaves the hull through the bottom at `y ~ 3.2` and the deck at `y ~ -4.85`, inside
  the bows' `|y|` extent, so the trim wedges are prismatic; hand check, **assumed** beyond that).
- **Today's certified preparation refuses the B1 hull** with `UNSUPPORTED_SURFACE`.
- The repository has exact Sturm isolation (`analytic_roots.py`, `exact_levels.py`) and an exact
  interval class with `sqrt`, `atan`, `sincos`, `pi` but no `log` (`exact_predicates.py`), none metered
  by `WorkBudget`; `mpmath` is **not installed** in the Python used.

Derived (assumed until CH3b measures it): with partial fractions over a coprime factor basis every 1-D
integrand of section 4 reduces to terms `L(u) / Q(u)^k` with `Q` of degree `<= 2` for B1, so Taylor
bit growth stays in the measured regime.

## 2. Certified domain (D-CH3-1: option A, decided)

### 2.1 Options (as decided)

| Option | Content | Refuses | Cost and risk |
|---|---|---|---|
| A. **Ruled rational class** (decided) | planar faces (straight or conic edges), plus faces whose support is a rational tensor-product surface of degree `(p, 1)`, `p in {1, 2}`, one clamped span in each direction, two control rows with **equal weight rows**, positive weights, trimmed only by plane sections; vertices are exact triple intersections | general NURBS, interior knots, `p >= 3`, unequal weight rows, straight/iso-parametric edges on curved faces, curved edges between two curved faces, holes on curved faces | plane sections are rational graphs `v(u)` wherever `beta != 0` (section 2.2); integrals are 1-D; algebraic numbers are univariate |
| B. Rational NURBS trimmed by planes | any `(p, q)`, knots | free-form trims | sections are algebraic curves, not graphs; 2-D enclosures, bivariate isolation |
| C. General trims | any face | nothing | no exact predicate base |

### 2.2 Parametrisation and the structural fact

**Knot space is the model.** `NurbsSurface.evaluate(s, t)` maps chart coordinates `(s, t) in [0, 1]^2`
affinely onto `u_range x v_range` and evaluates the B-spline in knot parameters `(u, v)`. CH3 works
**only in knot parameters**: the exact homogeneous net defines, over the single clamped span
`[k_p, k_{p+1}] x [k_1, k_2]` (`[0, 1] x [0, 1]` for B1),

```
S~(u, v) = (1 - v') A~(u) + v' B~(u),  v' = (v - k_1) / (k_2 - k_1)  (B1: v' = v),
A~, B~ in Q[u]^4 of degree <= p (Bernstein over [k_p, k_{p+1}], converted exactly to power form),
W(u) = A~_w(u) = B~_w(u)   (equal weight rows; W > 0 on the span: positive Bernstein coefficients),
S(u, v) = P(u, v) / W(u),  P = (1 - v') A~_xyz + v' B~_xyz, bidegree (p, 1).
```

The face's parameter domain is the **range box** `[u_0, u_1] x [v_0, v_1] = u_range x v_range` (exact
rationals of the stored doubles), a sub-box of the span. Every admitted face must lie strictly inside
the range box (check C3: every corner has `u_0 < u* < u_1` and `v_0 < r(u*) < v_1`, and every trim
graph stays inside `[v_0, v_1]` on its edge interval, decided exactly), so the chart boundary never
bounds a face and no reparametrisation of the Bernstein net is needed. The chart maps are recorded in
the snapshot (both ranges) and used only to interpret float hints (section 2.3) and to state the
range-box check. (Revision 2: with `v_range = (1/4, 3/4)` B1's *chart* relation would be
`z = 1 + 2 t`; the knot-space relation `z = 4 v` and every formula below are unchanged.)

A plane `n . r = d` meets the support where

```
f(u, v) = n . P(u, v) - d W(u) = alpha(u) + v' beta(u) = 0,
alpha = n . A~_xyz - d W,  beta = n . (B~_xyz - A~_xyz)   (both in Q[u], degree <= p),
```

so wherever `beta(u) != 0` the section is the **rational graph** `v' = r(u) = -alpha(u) / beta(u)`
(the reviewer confirmed this reduction). Consequences used throughout:

- a trim edge is `{S(u, r_k(u)) : u in [u_a, u_b]}` with `r_k = N_k / D_k` **reduced** (exact gcd), and
  `D_k != 0` on `[u_a, u_b]`; its world point is `C_k(u) = E_k(u) / M_k(u)`,
  `E_k = A~_xyz D_k + (B~_xyz - A~_xyz) N_k` (degree `<= 2p`), `M_k = W D_k`, each coordinate reduced
  (for B1's `z`-planes `r = h/4` after reduction and `C_z` is the constant `h`);
- two graphs cross where `N_i D_j - N_j D_i = 0` (degree `<= 2p`, **4 for B1**);
- `alpha` depends on `d` affinely and `beta` not at all;
- `beta(u*) = 0` with `alpha(u*) != 0`: no section point has `u = u*`; `alpha(u*) = beta(u*) = 0`: the
  plane contains the ruling `u = u*` (section 5); `beta ≡ 0`: the plane is parallel to every ruling and
  the section is a union of rulings at the roots of `alpha` (section 5, E4); `alpha ≡ beta ≡ 0`: the
  face lies in the plane (section 5, E7).

Degree bookkeeping (revision 2, used in sections 4 and 7): `deg N_k, D_k <= p`; `deg E_k, M_k <= 2p`;
the edge height numerator `g_k = n . E_k - d M_k` has degree `<= 2p`; its critical-point polynomial
`g_k' M_k - g_k M_k'` has degree `<= 4p - 1` (7 for B1); cap integrands carry `M_w` to the fifth power
(denominator degree `<= 10p`), which partial fractions over the coprime basis reduce to terms with
`deg Q <= p` per factor.

### 2.3 Admission checks for the class (CH3a)

Applied per face of the selected set after the type checks; each decided exactly:

- **C1 class**: `NurbsSurface` with `degree_v == 1`, two control rows, `degree_u in {1, 2}`, clamped
  knots without interior knots in both directions, finite coordinates, `weights` `None` or with
  `weights[i][0] == weights[i][1]` exactly and all positive; finite `u_range`, `v_range` inside the knot
  domain. Else `UNSUPPORTED_SURFACE` (sub-kind recorded).
- **C2 trims**: every edge of a curved face is a `PlaneNurbsCurve` whose `surface` equals the face's
  support (`support_key`, `u_range`, `v_range` equal) and whose plane record is finite; its exact graph
  `r_k = N_k / D_k` comes from the plane's exact normal `m = u_vec x v_vec` and offset `m . origin`
  (stored doubles, exact). Planes that are also iso-`v` (B1's `z`-planes: `D_k` constant after
  reduction) are ordinary plane trims. `D_k` must be nonzero on the edge's `u`-interval (one sign of its
  Bernstein coefficients over the interval's outer bounds; else refine the algebraic ends to the policy
  depth, then `UNSUPPORTED_TRIM`). `Straight` or `CurveOnSurface` edges of curved faces refuse
  `UNSUPPORTED_TRIM` (rulings and iso-curves are out of scope).
- **C3 corners and hints**: every vertex of a curved face is the intersection of the two trim planes of
  its incident section edges with the support: the polynomial `N_i D_j - N_j D_i`, square-free-reduced,
  its real roots isolated in `(u_0, u_1)` and the candidates with `v_0 < r_i(u*) < v_1`. **Hint H1**
  (vertex): the candidate whose exact world enclosure (section 3.2, refined to width below
  `delta_hint`) is within `delta_hint` of the stored vertex position in the max-norm
  (`|x* - x_stored| <= delta_hint` per coordinate, exact rational comparisons against the enclosure
  bounds); exactly one candidate must pass, else `TRIM_HINT_MISMATCH`. The edge's exact `u`-interval
  is the interval between its two corners' roots. **Hint H2** (curve domain and path): the stored
  curve's own defined knot points at path parameters `t = 0, 1/2, 1` (`PlaneNurbsCurve._knot_points`,
  the library's exact-bisection definition, rounded) must each satisfy (a) `u_t` inside the exact
  edge's `u`-interval widened by `delta_chart` and (b) `|v_t - r_k(u_t)| <= delta_chart` (exact rational
  comparison at the rational `u_t`), with `delta_chart = 2^-40` in knot units; else `TRIM_HINT_MISMATCH`.
  Measured: B1's stored edges pass with deviation `0`, a distant subcurve fails (a). Hints are not part
  of the certified claim (section 3.1); H1/H2 only make the selection of exact candidates
  deterministic and reject a model whose stored derived data do not describe the exact solid.
- **C4 chart injectivity**: a rational direction `m` with `m . A~_xyz = c_0 W` and
  `m . (B~_xyz - A~_xyz) = c_1 W` as polynomials, `c_1 != 0` (one body coordinate is an affine function of
  `v'` alone; B1: `m = e_z`, `c_0 = 0`, `c_1 = 4`), and the control triple of every `v'`-row
  non-collinear: some component of `(P_1(v') - P_0(v')) x (P_2(v') - P_0(v'))` (quadratic in `v'`) has
  one sign on `[0, 1]` by its Bernstein coefficients (for `p = 1`: a segment of nonzero length). Then
  `(u, v) -> S` is injective on the range box. Else `UNSUPPORTED_SURFACE` (`not_graph_like`).
  (Revision 2: C4 does **not** imply anything about the face region's shape; see 3.3.)
- **Checks 1-6 of the planar design, extended** (revision 2 states the replacement predicates):
  (1) planar faces may now have **conic edges** (`PlaneNurbsCurve`s of a curved face sharing the edge,
  checked by C2 from that curved face's side; a planar face's conic edge not shared with an admitted
  curved face refuses `UNSUPPORTED_TRIM`); every vertex of a planar face lies exactly in its plane (for
  an algebraic corner: the plane's `m . C_k(u*) - d_k` is a single-root exact test; it is structurally
  zero for the planes that define the corner);
  (2) loops: a planar face's loop is a chain of straight segments and conic arcs, a curved face's loop a
  chain of graph pieces; simplicity is pairwise non-crossing of pieces plus a nonzero signed area, with
  the predicates of section 3.3;
  (3) holes: planar faces with conic edges and curved faces refuse `INVALID_HOLES` if they have holes
  (none in B1); planar faces with straight edges only keep the planar rules on `AlgebraicPoint`s;
  (4) shell combinatorics exactly as in the planar design (reused: every edge in two loops oppositely,
  one connected shell, manifold vertices);
  (5) embedding: *planar-planar*: the two planes' intersection line (exact) meets each face's loop in
  points (segments: planar predicates on `AlgebraicPoint`s; conic arcs: substitute the arc's rational
  parametrisation into the other plane: a univariate polynomial of degree `<= 2p` in `u`), the
  alternation along the line gives each face's intervals, and the two interval sets must meet exactly
  in the declared shared cells; coplanar pairs with conic edges refuse `EMBEDDING_UNDECIDED` unless
  their exact bounding boxes are disjoint; *curved-planar*: the planar face's plane section of the
  support is a graph `r_m`; its portions inside the curved face's region (section 3.3 parity) must be
  exactly the shared edge intervals (empty for a non-adjacent pair), and the planar face's loop must
  meet the support only on shared cells (its straight segments: the segment's line meets the support
  where a degree-`2p` polynomial in `u` vanishes with `v'` rational in `u`, single-root tests; its
  conic edges are shared or disjoint by the previous test); *curved-curved*: exact bounding boxes of the
  sub-patches over the range boxes (`knot_patch_hull`-style exact projected controls) must be disjoint,
  else `EMBEDDING_UNDECIDED` (B1's bows are separated by `|y| >= 4.75`);
  (6) orientation: the exact signed volume is an enclosure, certified positive (`lo > 0`) or refined
  until decided, else `UNDECIDED_PREDICATE`.

### 2.4 Listed extensions (not in CH3)

Interior knots, `p = 3`, cylinders via `cylinder_to_nurbs` (multi-span), rulings as straight edges,
holes on curved faces, unequal weight rows (logarithmic inner integrals).

## 3. The exact model (D-CH3-2, decided)

### 3.1 Rule

"Stored doubles are exact data" applies to **supports and plane records**: nets, weights, knots, ranges
and every trim plane's `origin`, `u_vector`, `v_vector`. **Derived stored data are hints**: section-curve
`domain` ends, `path` indices and corner vertex positions were produced numerically and only select
among exact candidates (C3, H1/H2). The exact solid is defined by the supports and the combinatorics;
quantities are certified for it, and the claim does not mention the hints beyond recording
`delta_hint`, `delta_chart` and the selected root identities. `delta_hint` is a policy value in world
units, default `2^-40 (1 + max |stored coordinate|)` (measured hint errors are at most about `2^-49`
of the magnitude: the library's `2^-50` certification of section points plus one rounding), compared
per coordinate in the max-norm; `delta_chart = 2^-40` in knot units.

### 3.2 Algebraic numbers and points (the predicate layer)

```
AlgebraicNumber: poly (square-free primitive integer polynomial), ordinal (index among the real roots of
                 poly in increasing order, from 0), interval (lo, hi) with exactly one root of poly inside,
                 established by a Sturm count of 1 (not by endpoint signs alone); refine() bisects exactly
AlgebraicPoint:  root (AlgebraicNumber), coords (three reduced rational functions in Q[u])
Enclosure:       (lo, hi) Fractions, invariant: the exact value is in [lo, hi]
```

**Identity** (revision 2): two roots are equal iff (same `poly`) their ordinals are equal; (different
`poly`, `q`) the intersection `I` of their isolating intervals satisfies `count_poly(I) = 1`,
`count_q(I) = 1` and `count_gcd(poly, q)(I) = 1` (then the unique root of `poly` in `I` is the root of
`poly` in its own interval, likewise for `q`, and both are the root of the gcd in `I`); otherwise they
are distinct, which is certified by refining until the intervals are disjoint (a bounded loop: distinct
roots separate). The review's `(x^2-2)(x^2-3)` example: same polynomial, ordinals 2 and 3, distinct;
treated as different polynomials, `I = (3/2, 8/5)` has count 0, distinct. Ordering uses the same tools.
Identity is **never** inferred from overlapping intervals.

Predicates on one root are exact: the sign of `g(u*)`, `g in Q[u]`, is the sign of `g`'s Bernstein
coefficients over the isolating interval once they agree, or `0` exactly iff `gcd(g, poly)` has a root in
the interval (count 1), decided first. Expressions in several roots are `Enclosure`s by rational
interval arithmetic, refined to `max_isolation_bits`; an undecided sign refuses `UNDECIDED_PREDICATE`
(the proof records the depth). Mirrored corners of the full hull have the same `poly` and ordinal
(the corner polynomial does not involve `y`), so across-line directions are exact in `x` and `z`.

### 3.3 Loops of graph pieces: region, parity, orientation (revision 2, replaces "v-monotone")

The chart region of a curved face is the **interior of its simple closed loop** of graph pieces; no
monotonicity in `v` is assumed or required (the review's hexagon `(0.1,0.1), (0.9,0.1), (0.8,0.9),
(0.6,0.8), (0.7,0.3), (0.2,0.4)` on `S = (u, u^2, v)` is a valid loop of plane trims `z = a x + b` with
two intervals at `u = 0.65`; B1's loops happen to be convex in `v`). Tools, all exact:

- **Crossing of two pieces**: roots of `N_i D_j - N_j D_i` in the open intersection of their
  `u`-intervals; consecutive pieces may meet only at their shared corner (the corner root, by identity).
- **Vertical-ray parity**: a point `(u_q, v_q)` with rational `u_q` (chosen between the isolating
  intervals of the breakpoints it must avoid) and rational or algebraic `v_q` is inside the loop iff the
  number of pieces whose open `u`-interval contains `u_q` and whose graph value `r_k(u_q)` exceeds
  `v_q` is odd (exact rational comparisons; a tie is impossible for a point chosen off the pieces).
- **Signed area and orientation**: `sigma_chart = sign(-oint v du) = sign(-sum_k dir_k int N_k / D_k du)`
  over the loop pieces (`dir_k = +1` for a piece traversed in `+u`), an `Enclosure` whose sign is
  certified (nonzero for a simple loop; refined, else `UNDECIDED_PREDICATE`); the loop runs
  anticlockwise about the outward normal, so `sigma_f = sigma_chart` is the sign with which
  `S_u x S_v` is outward.
- **Clipping by a graph half-region** `H_w = {alpha_w + v' beta_w <= 0}`: on a `u`-piece where
  `beta_w > 0` this is `v' <= r_w`, where `beta_w < 0` it is `v' >= r_w`, where `beta_w = 0` it is all or
  nothing by the sign of `alpha_w` (pieces are split at the roots of `beta_w`). Cut points are the roots
  of `N_w D_k - N_k D_w` on each loop piece (ordinal along the edge recorded); the loop pieces are split
  there and each sub-piece kept iff a test point on it is in `H_w` (exact at a rational `u`); the
  graph `r_w` is split at its cut points and each portion kept iff its test point is inside the loop
  (vertical-ray parity). Kept loop sub-pieces keep their direction; kept graph portions are directed so
  that the clipped region is on their left: `-u` where `H_w` is `v' <= r_w`, `+u` where it is
  `v' >= r_w`, times `sigma_chart`. The result is an oriented chain, possibly several cycles, used
  **only as a chain** (Green's theorem needs no cycle assembly). Checked exactly on the review hexagon
  against polygon clipping (`107/500`).

Planar faces with conic edges use the same pattern in their plane: pieces are straight segments and
conic arcs; segment-segment crossings are the planar predicates on `AlgebraicPoint`s; segment-arc
crossings are the roots of the segment line's equation along the arc (degree `<= 2p` in `u`, filtered
to the segment's parameter range by single-root tests); arc-arc crossings (B1's deck: the starboard
and port arcs) substitute one arc's parametrisation into the exact implicit conic of the other
(implicitisation of a rational quadratic is exact and standard), degree `<= 4`. The fan formulas of
section 4.1 need no parity.

### 3.4 Snapshot document (convention version 2)

The planar snapshot document gains, per curved face, `"support": {"kind": "ruled", "degree_u", "net":
[[[wx, wy, wz, w] pairs]], "span": [[k_p, k_{p+1}], [k_1, k_2]], "u_range": [u_0, u_1], "v_range":
[v_0, v_1], "sigma_chart"}`; per section edge `{"kind": "section", "plane": [origin, u, v],
"domain_hint": [hex, hex], "path_hint": int, "graph": [N, D] (reduced, integer-primitive)}`; per vertex
on a curved face `{"corner": {"poly": [...] (square-free primitive), "ordinal": int, "hint": [hex x3]}}`.
**Isolating intervals are not in the document** (revision 2): the root identity is `(poly, ordinal)`,
which every correct isolation algorithm reproduces; a verifier re-isolates and matches ordinals. The
digest covers hints because they select candidates. Planar-only preparations produce exactly the
convention-1 document.

## 4. Quantities and bounds (D-CH3-3, D-CH3-4, decided)

### 4.1 Decomposition

For the `below` convention, `K = S ∩ {n . r <= d}` is bounded by clipped planar faces, clipped curved
faces and the cap. All reduce to signed integrals along graph pieces:

- **Planar face** (revision 2: apex in the face plane). Fan from a point `q_f` **in the face's plane**
  (the stored plane origin, exact rational); the tetrahedra have their apex at the body origin, as in
  the planar design. A straight edge `a -> b` contributes `V += det(q_f, a, b) / 6` and
  `J += det(q_f, a, b) (q_f + a + b) / 24` on `AlgebraicPoint`s; a conic edge `C_k(u)`, `u` from `u_a`
  to `u_b` in the loop's direction, contributes

  ```
  V += int det(q_f, C_k, C_k') / 6 du,      J += int det(q_f, C_k, C_k') (q_f + 2 C_k) / 24 du,
  ```

  the limits of the triangle formulas for `(q_f, C, C + C' du)`. With `C = E / M`:
  `det(q_f, C, C') = det(q_f, E, E') / M^2` (the `E M'` term vanishes) and the `J` integrand has
  denominator `M^3`. Checked on the review's region (`1/18`). Clipping a conic edge against the
  half-space: its height numerator `g_k = n . E_k - d M_k` has roots (degree `<= 2p`) at which the edge
  enters or leaves `H`; the kept portions are the sub-intervals whose test point has `g_k M_k <= 0`,
  i.e. `n . C_k <= d`.
- **Curved face**. `V_f = sigma_f int_R P . (P_u x P_v) / (3 W^3) du dv` (`S . (S_u x S_v) =
  P . (P_u x P_v) / W^3` since `P . (P x P_v) = 0`; `P_v` means `dP/dv`, which carries the factor
  `1 / (k_2 - k_1)`), and `J_f,i = sigma_f int_R P_i P . (P_u x P_v) / (4 W^4) du dv` with `P_i` the
  homogeneous numerator of coordinate `i` (`S_i = P_i / W`, hence `W^4`). The integrands are polynomials
  in `v` of degree 2 and 3 over `W^3`, `W^4`; by Green's theorem each is `-oint_{dR} F(u, v) du` with
  `F(u, v) = int_0^v (integrand) dv'` polynomial in `v`; on a graph piece `v = N / D`, `F(u, N/D)` has
  denominator `W^3 D^3` (`V`) or `W^4 D^4` (`J`). `R` is the clipped region of section 3.3 and `dR` its
  oriented chain.
- **Cap**. The oriented cut chain is the union of straight cut segments on planar faces (planar design
  section 5, on `AlgebraicPoint`s; a cut point on a conic edge of a planar face is an `AlgebraicPoint`
  of the edge's height root) and **cut arcs** `C_w(u) = E_w / M_w` on curved faces (the kept graph
  portions of 3.3, each between two cut points). The fan from a plane point `q` (the waterplane origin
  projected exactly onto the plane: `q = o + ((d - n . o) / (n . n)) n`, rational): for a straight cut
  segment the planar triangle formulas; for a cut arc, with `(u_p, v_p) = (e1 . (C_w - o), e2 . (C_w - o))`,

  ```
  dA~ = ((C_w - q) x C_w') . n / 2 du        (the numerator; A = int dA~ / |n|),
  Mu~ = int (q_u + 2 u_p) / 3 dA~,   Iuu~ = int (q_u^2 + 2 q_u u_p + 3 u_p^2) / 6 dA~,
  Iuv~ = int (q_u q_v + q_u v_p + u_p q_v + 3 u_p v_p) / 6 dA~   (Mv~, Ivv~ likewise),
  ```

  the limits of the planar triangle formulas. Denominators are powers of `M_w` up to the fifth.

Every contribution is a sum over pieces of `int_{u_a}^{u_b} Num(u) / Den(u) du` with `Num, Den in Q[u]`,
`Den` a product of powers of `W`, `D_k`, `D_w` (each of one sign on the piece, certified by Bernstein
coefficients over the piece's outer interval), and `u_a`, `u_b` rational or algebraic. No 2-D quadrature
and no transcendental function is evaluated anywhere.

### 4.2 Breakpoints and pieces

Per face: corners; cut points (roots of `N_w D_k - N_k D_w` on each edge interval, with ordinals);
roots of `beta_w` in the face's `u`-extent; for conic edges of planar faces, the roots of `g_k`. Sorting
uses section 3.2. Between consecutive breakpoints every chain piece has a fixed graph and a fixed
denominator sign, and is one 1-D integral per quantity. Piece counts are `O(corners + cut points + 2)`
per face.

### 4.3 Enclosing a 1-D rational integral (the only approximate step; D-CH3-3 option (ii))

**Reduction.** Collect the denominators of a quantity's pieces; compute a **pairwise-coprime factor
basis** `{Q_1, ..., Q_m}` by gcd refinement (no irreducibility needed; for B1 the basis consists of
`W`, `u`, and linear/quadratic factors of the `D_k`); write each `Den` as a product of basis powers and
split `Num / Den` by partial fractions into a polynomial part (integrated exactly) and terms
`L(u) / Q^k` with `deg L < deg Q`. (Revision 2: the "common factor" refusal of revision 1 is removed; a
shared factor such as `W` in two denominators is simply a basis element. A trim plane equal to the
waterplane is E5/E7 of section 5, handled before integration.)

**Term enclosure** (revision 2: the complete statement). For `T = int_a^b L(u) g(u) du`, `g = Q^{-k}`,
on a rational piece `[a, b]`, `c = (a + b) / 2`, `h = (b - a) / 2`:

```
g^{(j)} = P_j / Q^{k+j},  P_0 = 1,  P_{j+1} = P_j' Q - (k + j) P_j Q'     (exact in Q[u]);
Taylor polynomial  t_K(u) = sum_{j<=K} g_j (u - c)^j,  g_j = P_j(c) / (Q(c)^{k+j} j!)   (exact);
main = int_a^b L(u) t_K(u) du   (exact rational: a polynomial integral);
pointwise remainder: |g(u) - t_K(u)| <= B_g |u - c|^{K+1},
     B_g = max_{[a,b]} |P_{K+1}| / (min_{[a,b]} |Q|)^{K+1+k} / (K+1)!,
     max |P_{K+1}| from the absolute Bernstein coefficients of P_{K+1} over [a, b],
     min |Q| = min of the absolute Bernstein coefficients of Q over [a, b], valid because they all have
     one sign (certified; a sign change on the piece refuses UNSUPPORTED_TRIM at admission or splits
     the piece at the roots of beta during the query; the sign itself is irrelevant to the bound);
integral remainder: |int_a^b L (g - t_K)| <= rho,
     rho = B_g * L_max * int_a^b |u - c|^{K+1} du = B_g * L_max * 2 h^{K+2} / (K + 2),
     L_max = max of the absolute Bernstein coefficients of L over [a, b];
T in [main - rho, main + rho].
```

Checked on the review's `L = 10^6`, `Q = 1 + u`, `[0, 1]`, `K = 8`: error `1.133`, `rho = 195.3`.

**Algebraic endpoints.** For an end `u_a` with isolating interval `[a_lo, a_hi]`, `T` over
`[u_a, u_b]` is enclosed by the term enclosure over the inner rational interval `[a_hi, b_lo]` plus the
endpoint enclosures `[-e_a, e_a]`, `e_a = (a_hi - a_lo) L_max(a) / min|Q|(a)^k` over `[a_lo, a_hi]`
(likewise at `b`); the inner interval must be non-empty (else refine the ends).

**Accumulation and allocation** (revision 2). A quantity's enclosure is
`exact_part + sum over pieces, terms and endpoints of [lo, hi]`, summed as intervals; its width is the
**sum** of all `2 rho` and `2 e`. The core target `tau` is allocated as follows: each quantity's pieces
receive `tau / n_pieces`; within a piece each term receives the piece's share divided by its term
count; when a term's piece is bisected, each child receives **half** of the parent term's allowance
(never the full allowance again); endpoint enclosures draw from the piece's share before the terms
and are reduced by refining the root (bisection halves `e`). Orders start at `K_0 = 8` and rise before
bisection up to `max_taylor_order`; bisection depth and the total piece count are bounded by
`max_pieces_per_integral`; refinement by `max_isolation_bits`. If a share cannot be met within these
limits the quantity refuses `ACCURACY_UNACHIEVABLE` with the achieved width.

### 4.4 Forms, bounds and the claim (D-CH3-4, revised in revision 2 after findings 9-10)

A quantity in a CH3 preparation is one of:

- **exact**: only when no algebraic or enclosed term entered it (`EMPTY`'s `V = 0`; planar-only
  sections through faces whose vertices are all rational; the degree-`(1,1)` box of section 8 for planes
  missing its ruled face); same forms and rules as the planar design.
- **enclosed**: the claim is a rational interval `[lo, hi]` with the stated meaning "the exact value of
  the exact solid for the exact query lies in `[lo, hi]`", `hi - lo <= a` for the requested accuracy
  `a`. Construction: the engine computes a rigorous `core = [c_lo, c_hi]` of width `w <= a / 3` and
  publishes `claim = [c_lo - w, c_hi + w]` (width `3 w <= a`). **Only the claim is a verified
  statement.** `core`, piece counts, orders and isolation depths are reported under `"execution"` as
  untrusted metadata, excluded from the verified object and never compared by the verifier beyond
  well-formedness (revision 2, finding 10). The widening exists so that an independent rigorous
  enclosure of width `<= w` fits inside the claim (section 6).
  Forms: `"enclosure": {"lo": PAIR, "hi": PAIR}` (the claim), `"binary64": {"lo", "hi"}` the outward
  rounding of the claim to doubles with `hi - lo` as its bound (adjacency is not promised; a request
  finer than that width refuses `ACCURACY_UNACHIEVABLE` for the binary64 form), `"decimal"` the
  decimal enclosure of the claim of the requested width; the `exact` key is absent (its presence on an
  enclosed quantity is `MALFORMED_PROOF`).
- **Waterplane quantities and `1/|n|`** (revision 2, finding 9). The integrals of 4.1 give
  **numerator** enclosures `[N_lo, N_hi]` for `A~, Mu~, ...` with the value `= numerator / |n|`,
  `|n|^2` rational. Certified bounds `s_lo <= |n| <= s_hi` come from integer square roots:
  `s_lo = isqrt(|n|^2 * 2^(2m)) / 2^m`, `s_hi = s_lo + 2^-m`, with `m` chosen so that
  `2^-m <= 2^-200 s_lo` (exact, cheap). The value enclosure is
  `[min(N_lo/s_lo, N_lo/s_hi), max(N_hi/s_lo, N_hi/s_hi)]`; its width is at most
  `(N_hi - N_lo) / s_lo + max(|N_lo|, |N_hi|) (s_hi - s_lo) / s_lo^2`. Allocation: the numerator core
  target is `tau_N = (a / 3) s_lo / 2` and the sqrt term is kept below `tau_N` by increasing `m`
  (bounded, logarithmic), so the value core has width `<= a / 3` and the claim `<= a`. The binary64
  outward rounding then adds at most 2 ulps; if that breaks `hi - lo <= a` the binary64 form refuses
  `ACCURACY_UNACHIEVABLE` as in the planar design (the rational and decimal forms meet `a`).
  Scale invariance: `(n, d)` and `(2 n, 2 d)` define the same plane; numerators double and `|n|`
  doubles; the two claims of every quantity must intersect and both meet `a` (a CH3c test).

Accuracy requests: `absolute_accuracy[name] = a` sets `tau = a / 3`; without a request the policy's
`default_enclosure_width = 10^-12` applies to the claim (`tau = 10^-12 / 3`). `accuracy_met` is `True`
for a returned enclosed quantity and is recomputed by the verifier from the claim's width.

## 5. Section topology, degeneracy and one-sided limits (D-CH3-5, decided; revised for finding 8)

### 5.1 Events of a curved face for fixed `n`, and zero polynomials

Zero-polynomial cases are decided **first**, exactly, per face and per edge, in this precedence:

| Code | Condition | Handling |
|---|---|---|
| E7 face in plane | `alpha_w ≡ 0` and `beta_w ≡ 0` as polynomials (possible only for a planar ruled patch) | the face lies in the plane: the planar rule (entirely above for `below`, entirely below for `above`; never double-counted); no cut points on its edges |
| E4 ruling(s) in plane | `beta_w ≡ 0` with a root of `alpha_w` in the face's `u`-extent, or `gcd(alpha_w, beta_w)` with a root there | refuse `CURVED_SECTION_DEGENERACY / ruling_in_plane` (E3 interior tangency is the special case where the plane is also tangent: `n . S_u = 0` on that ruling; recorded as sub-reason `interior_tangency`); `beta_w ≡ 0` without such a root: the face is entirely on one side (sign of `alpha_w`, constant) |
| E5 edge in plane | `g_k = n . E_k - d M_k ≡ 0` | the edge contributes no cut points; E2 is **not** evaluated for it (its derivative polynomial is identically zero by construction); the edge's two corners are E1 contacts; the adjacent faces' sides are decided by the perturbed plane (5.2) |
| E1 corner on plane | `g_k(u*) = 0` at a corner (single-root exact test) | supported (5.2) |
| E2 edge tangency | a root of `g_k' M_k - g_k M_k'` in the open edge interval at which `g_k = 0` (single-root test), evaluated only for edges with `g_k` not identically zero | refuse `CURVED_SECTION_DEGENERACY / edge_tangency` |
| E6 planar events | vertex heights of planar-only vertices (rational), E1 at algebraic corners, planar edges and faces in the plane | planar design |

Heights are compared with `d` exactly (rational `d`, single-root tests); the proof records which codes
occurred (`contacts` lists corner contacts, edges in the plane, faces in the plane).

**Completeness of the inventory** (revision 2). Fix `n`; let `d` vary in an open interval containing no
E1/E2/E4/E5/E6 height. On every edge the roots of `g_k(.; d)` are simple (no E2) and interior (no E1),
so by the implicit function theorem they move continuously and keep their number and order: cut points
and their ordinals are constant. On every curved face the graph `r_w(.; d) = (d W - n . A~_xyz) / beta_w`
is defined off the roots of `beta_w`, moves monotonically in `v'` at each `u` (`d r_w / dd = W / beta_w`),
and has no point on a root of `beta_w` (no E4), so its portions inside the region begin and end only at
cut points on the loop; their number and their endpoints' identities are therefore constant. The
cut-point graph (vertices: cut points by identity; edges: cut arcs and planar cut segments) is thus
constant on the interval, and so are loops and nesting. Hence every combinatorial change of the section
happens at an event height of the inventory: the inventory is complete.

### 5.2 The certified event gap (replaces symbolic perturbation)

1. **Gap certification.** For a degenerate plane (some event at `d`), isolate the heights of all events
   not at `d` as algebraic numbers in `d`-space (E1: `n . C_k(u*)`; E2: the heights at the critical
   roots; E4: the heights at the roots of `beta_w` where `alpha_w(.; d)` would vanish, i.e. `d` with
   `d W(u*) = n . A~(u*)`; E6: vertex heights) and refine until a rational `eps_0 > 0` has no event in
   `(d - eps_0, d) ∪ (d, d + eps_0)` (undecidable separations refuse `UNDECIDED_PREDICATE`).
2. **Limit combinatorics** at the generic rational planes `d ∓ eps_0 / 2`: cut points by
   `(edge id, ordinal)`, cut arcs, segments, loops, nesting (planar machinery for planar faces with
   rational vertices; faces with algebraic vertices and curved faces through section 3.3).
3. **Limit positions and quantities at `d`**: each limit cut point converges to a root of `g_k(.; d)`
   (transversal), a corner (E1) or an edge end (E5), identified by continuity of ordinals inside the
   gap; quantities are the section-4 integrals **at `d`** over the limit chain; an edge in the plane is
   counted on the side where the perturbed plane crosses it (the planar convention). `V` and `J` are
   continuous in `d`: one value at `d`, `below` convention.

Topology identities: a loop is the sorted tuple of `(edge id, ordinal)` cut-point identities
(convention 2); nesting uses signed area and ray parity on `Enclosure`s with the predicate layer.

## 6. The independent verifier (D-CH3-10, decided; revised for findings 3, 10, 11)

`verify_certified_hydrostatics` keeps steps (0)-(2) and (5) of the planar design and extends (3)-(4):

- **Class and model**: re-derives the exact net, graphs, corners and chart loops with its own code
  (Descartes/Bernstein subdivision where the provider uses Sturm), re-establishes the single-root
  invariant of every isolating interval it uses by its own count, matches corner identities by
  `(poly, ordinal)` (never by intervals, which the proof does not contain), re-runs C1-C4 and 1-6 and
  requires the same `sigma_chart`, breakpoints (by identity and order) and chains.
- **Combinatorics and exact parts**: recomputes contacts, event codes, occupancies, its own event gaps,
  cut-point identities, loops, nesting and every exact quantity; requires exact equality.
- **Enclosed quantities**: the verified statement is only `exact value in claim` and
  `width(claim) <= a`. The verifier computes its own rigorous enclosure `E_v` with a **different
  scheme** (one-sided Taylor at the piece's left end, `int_0^h g = sum_j g^{(j)}(u_a) h^{j+1} / (j+1)! + R`
  with its own remainder bound of the same Bernstein form, its own order schedule and bisection) and
  refines until `width(E_v) <= (hi - lo) / 3` or its budget stops. It accepts iff `E_v ⊆ [lo, hi]`;
  else `VERIFICATION_FAILED / enclosure`. Soundness: both enclosures contain the exact value; an honest
  claim is `core ± w` with `w >= width(E_v)`, so `E_v ⊆ claim`. A claim narrowed so far that
  `(hi - lo) / 3` is below what the verifier reaches within its budget ends in `BUDGET_EXHAUSTED`
  (phase `verify`), never in acceptance; a claim that excludes the true value can never be accepted:
  `E_v` contains the value, so it always has a point outside the claim and `E_v ⊆ claim` fails at every
  width. Execution metadata is checked for well-formedness only.
- **Trust model**: the claim holds for the exact solid of 3.1 given (a) the identities of section 4
  (divergence and Green theorems on simple loops of graph pieces, the clipping chain of 3.3, the fan
  with a planar apex) and the completeness argument of 5.1; (b) the correctness of the verifier's
  polynomial arithmetic, root isolation with Sturm/Descartes counts, and the Taylor/Bernstein bound
  (stated in its documentation and reviewed); (c) outward rounding. The verifier imports nothing from
  the provider modules.

## 7. Work, checkpoints and bulk operations (D-CH3-7, decided; revised for finding 12)

Primitive units (each reserved before it runs, each a checkpoint for budget, deadline and
cancellation): `poly_coefficient` (one coefficient multiply-add), `sign_count` (one sign of one
polynomial value in a Sturm or Descartes count; the value is accumulated by `poly_coefficient` units),
`root_refine` (one bisection decision after its evaluations), `interval_op`, `bernstein_coefficient`,
`piece_split`, `claim_element`, `serialize_element`. Composite steps are **bookkeeping only**: a Taylor
coefficient `g_j` is the metered polynomial evaluations and divisions that produce it; an event height
is its metered root isolation; nothing is charged twice and nothing unmetered runs.

Bulk operations (amendment 6b rows added; each charged before it starts by the stated bound, followed
by a checkpoint):

| Bulk operation | Bound |
|---|---|
| one polynomial product, pseudo-remainder step, shift, or Bernstein conversion | `(deg a + 1)(deg b + 1)` or `(deg + 1)^2` coefficient units; degrees are bounded by section 2.2's bookkeeping: inputs `<= 2p`, derivative polynomials `<= 4p - 1`, denominators `<= 10p`, `P_j` of order `j` `<= j deg Q + deg Q - j` |

Policy limits bound every loop: `max_isolation_bits`, `max_taylor_order`, `max_pieces_per_integral`,
`max_refinement_rounds`. CH3b's acceptance includes, for each bulk operation, a deadline test and a
cancellation test that stop at the checkpoint immediately after it with the charged units retained.

## 8. Q08: evidence and the consumer's part (revised for findings 13-15)

### 8.1 Run set and success matrix (fixed here, before any assessment run)

Models: `build_b1_half_hull()` (6 faces) and `build_b1_hull()` (6 faces) with default frames
`(1.5, 3.5)`, bottom `0.5`, deck `3.0`; units `m`; forms `("enclosure", "binary64", "decimal")`;
requested accuracies `V, J: 1e-9`, `A, Mu, Mv, Iuu, Ivv, Iuv: 1e-9` (quantity units). ANYstability's
convention: `n = R[2]`, `d = n . point`, basis `R[0]`, `R[1]`, origin `point`, `R = Ry(-trim) Rx(heel)`
with the float sines and cosines as given.

| Group | Planes | Expected |
|---|---|---|
| U upright | `z <= h` through `(2.5, 0, h)`, `h in {0.75, 1.5, 2.25, 2.9}` | PARTIAL, generic, one section, one outer loop, no hole; every quantity enclosed |
| B boundaries | `h = 0.5` (bottom), `h = 3.0` (deck) | degenerate: `h = 3`: occupancy FULL, `below` PRESENT (one loop, `A` enclosed = deck area), `above` ABSENT; contacts: 4 corners (E1, both bows... 8 in the full hull), 2 (4) edges in plane (E5), the deck face; `h = 0.5`: occupancy EMPTY, `below` ABSENT, `above` PRESENT; `V` exact `0` at `h = 0.5` |
| N near-event | `h = 3 - 2^-40`, `h = 1/2 + 2^-40` | generic PARTIAL; must answer (nearby distinct events) |
| I inclined | heel `in {-0.5, -0.3, -0.1, 0.1, 0.3, 0.5}` x trim `in {-0.02, 0, 0.02}` through `(2.5, 0, z)`, `z in {0.75, 1.5, 2.25, 2.9, 3.0}` (90 planes per model) | PARTIAL unless the plane misses the hull (then EMPTY/FULL, generic); planes through `(2.5, 0, 1.5)` at heel `+-0.3` with no cut arc (section 1) must still answer; any degeneracy refusal must be typed and recorded with its code (none is expected: the data are generic floats, but the run accepts a recorded refusal for a plane that hits an exact event, since a refusal is a correct answer) |
| S scale | every I plane repeated with `(2n, 2d)` | claims intersect, both meet `a` |
| D constructed degeneracies (half hull) | E2: `n = C_k'(1/4) x e_z` on the deck arc with `d = n . C_k(1/4)` (rational: the conic point and tangent at `u = 1/4` are rational); E4: `n = (B(1/4) - A(1/4)) x (1, 0, 0)`, `d = n . A(1/4)` (contains the ruling `u = 1/4`); E3: `n = S_u x S_v` at `(1/4, 1/2)`, `d = n . S`; E1: the plane through the corner `(1.5, y*, 0.5)` with `n = (1, 0, 2)`, `d = 5/2` (both limits); E1 x2: the full hull's mirrored deck corners with `n = (1, 0, 0)`, `d = 3/2` (identical-root recognition, both limits) | E2/E3/E4 refuse `CURVED_SECTION_DEGENERACY` with the sub-reasons; E1 cases answer with both limits |
| M multiple roots | `n = (1, 1, 0)`, `d` chosen in CH3c so that `x + y = d` crosses the deck arc twice (if the arc's `x + y` has an interior maximum between the frames, to be established exactly in CH3c; else the case is replaced by one on the bottom arc or dropped with a note) | two cut points with ordinals 0 and 1 on one edge |

Success: every plane answers or refuses with a typed category; every answer verifies with the CH3d
verifier against the run's artifact identity; recorded per plane: state, contacts, loop counts,
claims, widths, work counts, wall time; the resource limits of the manifest (10^8 units, 4096 bits,
600 s per model and group) apply.

### 8.2 References (uncertified; consistency checks, not proof)

- **H1** `hydrostatics()` at `order = 24, rtol = 1e-8`: recorded with its `estimate`; the run flags a
  plane when `|H1 - mid(claim)| > max(width(claim), 1e-8 * max(1, |H1|))` and the flag triggers an
  investigation of the run record; **H1 agreement is not an acceptance gate** (an uncertified estimate
  need not lie inside a tight correct enclosure).
- **Station closed forms** (upright only): at `z = h` the section is bounded by the frames, the
  centreline (half hull) and the conic arc `S(u, h/4)`; its area and `u, v` moments have closed forms
  through the rational parametrisation, evaluated with `mpmath` at 40 digits; they check `A`, `Mu`,
  `Mv`, `Iuu`, `Ivv`, `Iuv` of group U and nothing else.
- **High-precision 2-D quadrature** (all groups): `mpmath.quad` at 30 digits of the direct divergence
  theorem integrands `S . (S_u x S_v) / 3` and `x_i S . (S_u x S_v) / 4` over the clipped chart regions
  and the planar faces, with the regions from the exact graphs (an independent *numerical* route for
  `V`, `J` and inclined quantities; its own error estimate recorded); a reference outside a claim flags
  the plane.
- **Sandwich** (group U): for consecutive `h_1 < h_2` in `{0.75, 1.5, 2.25, 2.9}`,
  `claim(V(h_2)) - claim(V(h_1))` must intersect `(h_2 - h_1) [lo A(h_1), hi A(h_2)]` (the B1 bow flares:
  `A(z)` increases with `z`, a fixture-specific expectation stated here, not a theorem of the engine).
- **Half/full relation** (groups U, B, N and the trim-0 planes of I; symmetric planes `n_y = 0`,
  `o_y = 0`, basis `e1 = R[0]`, `e2 = R[1]` with `R[1]` parallel to `y`): with the mirror `y -> -y`,
  `V`, `Jx`, `Jz`, `A`, `Mu`, `Iuu`, `Ivv` are even and `Jy`, `Mv`, `Iuv` are odd, so the full hull's
  claims must intersect `2 x` the half hull's claims for the even quantities and must contain `0` for
  the odd ones; both sets meet `a`. Bitwise equality of cores is recorded as an observation.
- **Transformations on the curved domain** (half hull, groups U and I at heel `0.3`): translation by
  `(7, -3, 11)`; rotation by the exact `3/5, 4/5` matrix after scaling the model by 5 (controls and plane
  records scaled exactly, weights unchanged); reflection `y -> -y` (the mirrored net, loops reversed);
  moment origin `o` shifted by `(1, 2, 3)` in the plane (projected exactly); basis change
  `(e1, e2) -> (e2, e1)` (handedness flips: `Mu <-> Mv`, `Iuu <-> Ivv`, `Iuv` unchanged) and
  `(e1, e2) -> (2 e1, e2)` (`Mu x2`, `Iuu x4`, `Iuv x2`); normal rescaling (group S). Expected: `V`, `A`
  invariant; `J' = R J + V t` (scaled by `s^4`, `s^3`); the transformed claims must intersect the
  transformed originals and meet `a`.
- **Planar cross-check**: a box `[0,2] x [0,3] x [0,1]` whose top face is a degree-`(1,1)` patch
  (controls `(0,0,1),(2,0,1)` / `(0,3,1),(2,3,1)`, `weights = None`, ranges `(0,1)`), its four edges
  `PlaneNurbsCurve(plane_record, surface, 0)` records for the planes `x = 0`, `x = 2`, `y = 0`, `y = 3`
  (built directly, without `from_plane`'s numerical certification if that refuses a planar patch),
  corners the rational roots of linear polynomials: for planes not cutting the top (`z <= 1/2`) every
  term is a polynomial integral with rational ends and the engine returns **exact** forms equal to CH1's
  for the same box with a `Plane` top; for planes cutting the top (`x <= 1`) the cut arc is
  `C_w = E / beta` (a rational parametrisation of a straight line) and the claims must contain CH1's
  exact values.
- **Conic fixture** (replaces the quarter-cylinder prism, whose rulings C2 excludes): the "cylinder
  corner" `{x >= 1/5, y >= 1/5, x^2 + y^2 <= 1, 1/2 <= z <= 3/2}` with the curved face a `(2, 1)` patch
  of the unit-circle quadrant (weights `(1, w, 1)`, `z` along `v`, `v_range = (0, 2)` or controls at
  `z = 0, 2`) trimmed by the planes `z = 1/2`, `z = 3/2`, `x = 1/5`, `y = 1/5` (all admissible graph
  trims; the two vertical planes meet along the line `(1/5, 1/5, z)` inside the conic), closed by two
  planar sectors and two planar rectangles. Reference: `V = 1 x` the area of the exact-`w` conic region
  (closed form by the rational parametrisation, `mpmath` 40 digits); upright and inclined planes.
- **Review counterexamples as fixtures**: the `(u, u^2, v)` hexagon face (as a face of a closed solid
  built with planar trims `z = a x + b` and planar closures; its clipped area and volume by Green and
  by exact polygon clipping), the `(x^2-2)(x^2-3)` root identity, the fan region of finding 1 (as a
  unit test of the planar-face integrals), the `10^6 / (1 + u)` term bound.

### 8.3 Consumer's part

Mapping claims into its `IntegralSample` intervals, PSD checks, Capytaine panel comparison, P0/P1
measurements, equilibrium/GZ, admission.

## 9. Slices

| Slice | Output | Acceptance criterion | Author | Depends on |
|---|---|---|---|---|
| CH3a | class recognition C1-C4, exact net and graphs in knot space, hints H1/H2, loops of graph pieces (crossings, parity, `sigma_chart`), planar faces with conic edges, extended checks 1-6, snapshot document v2 (ranges, `(poly, ordinal)` identities), new refusal categories (`UNSUPPORTED_TRIM`, `TRIM_HINT_MISMATCH`, `CURVED_SECTION_DEGENERACY`, `UNDECIDED_PREDICATE`, `EMBEDDING_UNDECIDED`), policy fields | `pytest tests/test_certified_hydrostatics_ch3_admission.py`: both B1 builds admit; every new refusal from a minimal case (unequal weight rows, interior knot, a `Straight` edge on the bow, a vertex hint moved by `1e-6`, an edge domain replaced by `(0.95, 0.99)`, a non-graph-like patch, two bows translated to overlap, a face reaching the range box); the review hexagon admits as a face (two intervals at `u = 0.65` by the parity tool); every CH1 fixture's snapshot digest and proof bytes unchanged | Sonnet (Opus review) | CH3b predicate layer |
| CH3b | the exact kernel: polynomials over `Q` (gcd, square-free, Sturm and Descartes counts, Bernstein conversion and bounds, coprime basis, partial fractions), `AlgebraicNumber` identity and ordering, `AlgebraicPoint`, `Enclosure`, the 1-D term integrator of 4.3 with allocation, `1/|n|` bounds, metering and bulk charging | `pytest tests/test_certified_hydrostatics_ch3_kernel.py`: enclosures contain `mpmath` references (40 digits) for `int L / Q^k` over 50 random rational pieces and `k in 1..5` at targets `1e-6..1e-20` (every target met; max bit length recorded); the finding-2 term (`rho >= error`); the finding-3 identity cases; allocation sums (claim width `<=` target in every case, children halve); metered unit counts equal sums; deadline and cancellation stop at the checkpoint after each bulk operation with units retained; `UNSUPPORTED_TRIM`/`ACCURACY_UNACHIEVABLE`/`UNDECIDED_PREDICATE` at each limit | Opus | - |
| CH3c | the curved query engine: breakpoints, chains (3.3), clipping, cut arcs, planar faces with algebraic vertices and conic edges, events E1-E7 with the gap method, enclosed quantities, `1/|n|`, forms and claim, proof v2 | `pytest tests/test_certified_hydrostatics_ch3_query.py`: the planar cross-check box (exact equality to CH1 for `z <= 1/2`; containment for `x <= 1`), the hexagon solid (Green vs polygon clip), the cylinder corner (conic references inside claims at 5 planes), B1 half hull groups U, B, N, D, M and 6 planes of I, scale pairs, transformations, sandwich; references of 8.2 inside every claim or flagged with the record | Sonnet for the engine from this specification; Opus for 4.3, 5 and review | CH3a, CH3b |
| CH3d | verifier extension (section 6), written by a separate session from this document | `pytest tests/test_certified_hydrostatics_ch3_verify.py`: agrees on every CH3c fixture; refuses: claim shifted by its own width (excludes the value: `enclosure`), `lo`/`hi` swapped (`MALFORMED_PROOF`), claim wider than `a` with `accuracy_met` true (`accuracy flag`), `exact` key on an enclosed quantity (`MALFORMED_PROOF`), altered `(poly, ordinal)` or hint (`snapshot`), a half-hull result presented for the full hull (`model`), planar attacks of the CH1 manifest re-run on a CH3 preparation; a claim narrowed to width `2^-200` (expected: typed `BUDGET_EXHAUSTED` in phase `verify`, never acceptance); altered execution metadata leaves acceptance unchanged (documented as not a claim) | separate Sonnet session (Opus review) | CH3c |
| CH3e | Q08 run set of 8.1 on both models, references of 8.2, delivery addendum (wheel, hashes, commands, requests, responses, proofs, verifier runs, refusals, work, timings, flags) | all planes answer or refuse typed; every answer verifies with CH3d; the success matrix of 8.1 met; flags of 8.2 investigated and recorded; run record written before assessment | Opus | CH3c, CH3d |

Codex Sol reviews every slice; this design gets Codex Astra reviews until clean before CH3a starts.
Patch version bump at CH3e (`0.4.7`).

## 10. Open risks

- **Bit growth**: order 32 reached 2130 bits on one quadratic factor; terms with `k = 5` and
  partial-fraction coefficients of hundreds of bits may approach the 4096 cap on long pieces; mitigation
  is more pieces at lower order (measured: order 12 on two halves beats order 16 on one); measured in
  CH3b before the engine is written.
- **Multi-root predicates**: mixed-root signs refuse rather than resolve when a true zero is not
  structurally recognised (a plane through two corners of different roots). Q08 planes are generic
  floats; the acceptance set includes the recognisable symmetric case and one unrecognisable case
  expected to refuse `UNDECIDED_PREDICATE` (to be added to group D in CH3c if constructible with
  rational data).
- **Event isolation in `d`**: near-coincident distinct events force deep refinement (bounded, refusable);
  B1's deck/bottom corner heights are rational, so those gaps are cheap.
- **Runtime**: unmeasured end to end; CH3c records timings; the 600 s ceiling stands until measured.
- **Embedding of curved pairs** by box disjointness only; touching shells refuse.
- **Hint tolerances** are policy numbers (`delta_hint`, `delta_chart`); models with looser stored data
  refuse `TRIM_HINT_MISMATCH`.
- **Group M** depends on a property of the deck arc to be established in CH3c.
- `mpmath` as a test-only dependency (D-CH3-8).
- Whether a given curved-shell quantity is transcendental is not decided by the engine; it reports an
  enclosure whenever an enclosed term entered, and the exact form only when none did (a rational value
  reached through enclosed terms is still reported as an enclosure).

## 11. Owner decisions

Decided 2026-10-20, all as recommended in revision 1:

- **D-CH3-1** domain: option A, the ruled rational class (section 2).
- **D-CH3-2** exact model: supports exact; trims = exact plane ∩ surface; stored domains, paths and
  corners are hints checked by H1 (`delta_hint = 2^-40 (1 + max |coord|)`, max-norm, world units) and
  H2 (`delta_chart = 2^-40`, knot units) and recorded.
- **D-CH3-3** quantity method: exact rational Taylor models with Bernstein remainders over a coprime
  factor basis (4.3); closed forms only as test references.
- **D-CH3-4** forms: the claim protocol of 4.4 (claim = core ± width, `exact` absent, binary64 as
  outward rounding with its width as bound, default claim width `10^-12`); revision 2 clarifies that
  only the claim is verified and that `1/|n|` is bounded by integer square roots with its own allocation.
- **D-CH3-5** degenerate planes: E1, E5, E6, E7 supported by the certified event gap; E2, E3, E4 refuse
  `CURVED_SECTION_DEGENERACY`.
- **D-CH3-6** kernel re-implemented inside `certified_hydrostatics` with metering; existing modules are
  references only; the verifier has its own copies.
- **D-CH3-7** policy defaults: `max_bits` 4096, `max_isolation_bits` 4096, `max_taylor_order` 32,
  `max_pieces_per_integral` 1024, `max_refinement_rounds` 8, `default_enclosure_width` 10^-12,
  `delta_hint`, `delta_chart` as above; raised only on measurement.
- **D-CH3-8** `mpmath` as a test-only dependency for the references.
- **D-CH3-9** convention version 2 for curved preparations; planar stays 1 byte for byte.
- **D-CH3-10** verifier by a separate session from this document, different root-isolation algorithm
  and Taylor scheme.

New in revision 2 (finding 4; the approach of revision 1 was wrong, and the fix is not covered by the
decided items):

- **D-CH3-11 Face regions that are not `v`-monotone.** Options: (a) an added admission check
  "half-space equivalence" (the loop's interior must equal the intersection of its trims' half-regions,
  so one `v`-interval per `u`), refusing the review hexagon and narrowing the admitted class beyond what
  D-CH3-1 states; (b) **integrate general simple loops**: regions are loop interiors, clipping produces
  an oriented chain with any number of intervals per `u`, decided by vertical-ray parity, and Green's
  theorem needs no cycle assembly (section 3.3; checked exactly on the hexagon). *Recommend (b)*: no
  change to D-CH3-1's domain, no extra check, the planar design's "signed chain, no loop assembly"
  principle carried over; B1 is unaffected either way.

## 12. Revision 1 review (Codex Astra): findings and resolutions

| # | Severity | Finding | Resolution in revision 2 |
|---|---|---|---|
| 1 | B | planar fan apex at the body origin makes `det(q, C, C')` vanish | 4.1: apex `q_f` in the face plane (the stored plane origin), tetrahedra at the body origin; `V` and `J` formulas restated; checked on the review region (`1/18`) |
| 2 | B | remainder bound was pointwise only; sign of `Q`; accumulation | 4.3: complete integral bound `rho = B_g L_max 2 h^{K+2} / (K+2)` with `min |Q|` from one-signed Bernstein coefficients, endpoint enclosures, interval summation and halving allocation on bisection; checked on `10^6 / (1 + u)` (`rho = 195.3 >= 1.133`) |
| 3 | B | same polynomial + overlapping intervals is not identity; single-root invariant | 3.2: identity by `(poly, ordinal)` or by the three-count intersection test; invariant established by counts (Sturm/Descartes), verifier re-establishes it; checked on `(x^2-2)(x^2-3)` |
| 4 | B | loops need not be `v`-monotone | 3.3 rewritten: loop interiors, vertical-ray parity, oriented clipping chain, Green without cycle assembly; 2.3 C4 no longer claims region shape; checked on the review hexagon (`57/200`, `107/500`); new decision D-CH3-11 |
| 5 | M | chart vs knot parameters, `u_range`/`v_range` | 2.2: knot space is the model, range box must contain every face strictly, ranges in the snapshot (3.4), `z = 4 v` stated in knot space |
| 6 | M | curve domain/path hints unchecked; metric undefined | 2.3 H1 (max-norm world, `delta_hint`) and H2 (three knot points, `u`-interval and `v`-deviation, `delta_chart`); measured that a distant domain fails H2(a); 3.1 states hints are outside the claim |
| 7 | M | planar faces with conic edges lack predicates | 2.3 (1)-(5) and 3.3: vertex-in-plane tests, piece crossings (segment-arc, arc-arc by implicitisation), signed area, line-intersection alternation for face pairs, coplanar pairs refuse unless disjoint |
| 8 | M | zero polynomials, E5 vs E2, `beta ≡ 0`, common factors, completeness | 5.1: precedence table E7/E4/E5/E1/E2/E6, E3 ⊂ E4 (tangency implies the ruling is in the plane), E2 skipped for E5 edges, `beta ≡ 0` cases; 4.3: coprime factor basis replaces the common-factor refusal; completeness argument written |
| 9 | M | `1/|n|` irrational; allocation; `J` numerator | 4.4: integer-sqrt bounds `s_lo, s_hi`, value interval and width bound, allocation `tau_N`, binary64 after rounding; scale test `(n, d)` vs `(2n, 2d)` in 8.1 group S; 4.1: `J_f,i` integrand `P_i P . (P_u x P_v) / W^4` with `P_i` homogeneous |
| 10 | M | core not verifiable; metadata | 4.4 and 6: only the claim is verified; `core`, pieces, orders, depths are untrusted `execution` metadata; verifier target `(hi - lo) / 3`; CH3d attacks revised (shift, swap, width/flag, narrowed claim ends in typed non-acceptance, metadata irrelevant) |
| 11 | M | isolating intervals in the digest | 3.4: identity `(poly, ordinal)`, intervals excluded from the snapshot document |
| 12 | M | composite units; degree bounds; bulk tests | 7: composite steps are bookkeeping over primitive units; bulk row with degree bookkeeping from 2.2; deadline/cancellation tests per bulk operation in CH3b |
| 13 | M | Q08 acceptance undefined; H1 as gate; station formulas insufficient; missing cases | 8.1 success matrix (groups U, B, N, I, S, D, M with expected states and refusals, accuracies, drafts); 8.2 H1 as flagged diagnostic, 2-D high-precision quadrature for `V`, `J`, inclined; transformations, basis, normal rescaling, E3/E4, coincident edges (group B), nearby events (N), denominator signs and factor basis (CH3b), multiple roots (M), review counterexamples as fixtures |
| 14 | M | quarter-cylinder prism has ruling edges; ruled box unspecified | 8.2: cylinder-corner fixture with plane trims only; the `(1,1)` box with explicit controls, ranges, edge records and corners |
| 15 | m | CH3c/CH3d swap; `2^-49` vs `delta_hint`; parity of half/full identities; "transcendental" | fixed: slice names in 8-9; 3.1 separates the measured `2^-49` from the `2^-40` policy; 8.2 lists even/odd quantities with the basis; 10 states transcendence as undecided by the engine |
