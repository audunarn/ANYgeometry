# Certified hydrostatics CH3: certified point queries for the curved B1 hull (design, revision 4, 2026-10-20)

Status: design for owner review; no library source changed. Revisions 1-3 (af9aae5, da45088, 52cc918)
were reviewed by Codex Astra; revision 3 left no blocking finding; revision 4 is the owner-approved
final revision covering exactly the six remaining findings (5 major, 1 minor; section 12, round 3,
each with the exact check that was run). Anything still open after the focused review of revision 4
becomes a slice obligation tied to a test. Section 7 follows the owner's restated work contract
(planar design amendment 6d, as relayed; that amendment is on `claude/ch-fixes-6b`, not yet on this
branch). Owner decisions 2026-10-20: D-CH3-1..11 as recommended
(section 11). Extends `CERTIFIED_HYDROSTATICS_DESIGN.md` (revision 5, "the planar design") and its
acceptance manifest. Consumer request:
`C:/Github/ANYstability/docs/ANYgeometry_CERTIFIED_HYDROSTATICS_HANDOFF.md` sections 1, 3 and 5 (Q08).
Working rules: correctness first and fail closed; a typed refusal beats a plausible number; probes find
limits. Facts are marked **measured** (read-only experiments on e26e31f, `PYTHONPATH=src`, Python
3.14.0, numpy 2.3.4; exact checks in rational arithmetic) or **assumed**/derived.

**Scope statement.** CH3 delivers certified half-space point queries (`V`, `J`, `A`, `Mu`, `Mv`, `Iuu`,
`Ivv`, `Iuv`, states, one-sided limits, topology, proof, verifier) for closed shells whose faces are
either planar faces (straight edges, or rational-arc edges shared with an admitted curved face) or
faces of the **ruled rational class** of section 2, trimmed by planes, which contains both B1 bows
(`build_b1_half_hull`, `build_b1_hull`). It serves the geometry portion of Q08 and nothing more: no
event/interval API (CH4), no openings or clearances, no general NURBS, no multi-span surfaces (section
2.4), no change to planar results (planar preparations stay byte-identical, convention version 1).
The consumer's gates (adapter mapping of enclosures, installed verification, Capytaine comparison,
P0/P1) remain theirs.

## 1. Facts this design rests on

Measured on the source (`src/anygeometry/b1_example.py`, `nurbs.py`, `support_curves.py`):

- **B1 bow surface**: `NurbsSurface` of degree `(2, 1)`, knots `(0,0,0,1,1,1)` x `(0,0,1,1)`,
  `u_range = (0, 0.9)`, `v_range = (0, 1)`, controls bottom row `(0,6,0), (5.4,6,0), (5.4,0,0)` and
  deck row `(0,7.5,4), (6,7.5,4), (6,0,4)`, weights `((1,1),(w,w),(1,1))`, `w = 0.7071067811865476`,
  `Fraction(w)**2 - 1/2 = +6.8e-17`. **The two weight rows are equal.**
- **`z = 4 v` in knot space**: bottom row `z~ = 0`, deck row `z~ - 4 W = 0` as polynomials, `W` the
  common weight polynomial; for a surface with `v`-knots `(2,2,4,4)` the same formulas give
  `z = 2 (v - 2)`, and the knot-space point formula of section 2.2 agrees with the library's
  `exact_point` to `8e-17` (a float comparison of exact values); the bottom plane then has knot
  `v = 9/4`, inside `v_range = (2, 4)`.
- **Rational plane sections**: `alpha`, `beta` are quadratics in `u` for B1; `S(u, -alpha/beta)` has
  plane residual exactly `0`.
- **Frame sections are not conics**: on the frame section `x = 3/2`, the matrix of `[1, y, z, y^2, yz,
  z^2]` at six exact rational points `u = 0.175 .. 0.180` has exact rank 6; the section is the rational
  curve `E/M` with `deg E = deg M = 4`. Only the iso-`v` sections (`z`-planes) are conics.
- **Corners** of the starboard bow face: knot `u` in `[0.1749, 0.4446]` (quadratics, isolated to
  `2^-40`), `v = 1/8` (bottom) or `3/4` (deck), strictly inside the range box.
- **B1 planar-face predicates hold** (exact, section 2.3): on the aft face `x = 3/2` the starboard arc
  has `E_y` and `M` with positive Bernstein coefficients (`min 0.81`, `0.13`), so the two mirrored arcs
  are separated; the across lines `z = 1/2`, `z = 3` meet the arc at no interior root (only at its
  corner); on the deck face the arc meets the line `{x = 3/2, z = 3}` at no interior root; the deck
  across line lies in that line (the shared edge) and the other across line is disjoint; the deck
  plane's graph `v = 3/4` is on the bow's loop between the deck corners, and vertical-ray parity puts
  `(0.17, 3/4)`, `(0.43, 3/4)` and `(0.3, 3/4 + 10^-6)` outside and `(0.3, 3/4 - 10^-6)` inside.
- **Trims and corners**: the half hull has 8 vertices, 12 edges (4 `PlaneNurbsCurve`, 8 `Straight`), 6
  faces; section-edge `domain` values are path parameters; stored corners have exact `x`, `z`. On the
  deck edge the library's knot points at `t = 0, 1/2, 1` match the exact corners to `2.3e-13` and
  `4.1e-13` in `u` (the width of the `2^-40` isolating intervals) with `|v - 3/4| = 0`; the shortened
  domain `((3a+b)/4, (a+3b)/4)` keeps `v = 3/4` and its midpoint inside, but its ends are `0.0595` off
  in `u` (`0.61 m` in world), and a distant domain `(0.95, 0.99)` has `u = 0.855..0.891`.
- **Orientation convention, exact end-to-end check** (section 3.3/4.1): on `S = (u, u^2, v)` the diamond
  `(1/4,1/4), (1/2,1/8), (3/4,1/4), (1/2,3/8)` gives `V = +25/4608` traversed anticlockwise and
  `-25/4608` clockwise with `V_f = -(1/3) oint_chain F du` (no extra sign); the parabolic prism of
  section 8.2 gives `V = 53/1536` from its five faces (curved face `-53/4608` with its clockwise loop),
  `V(z <= 1/2) = 29/3072` from the clipped faces plus the cap with every cut piece reversed, and the
  cap area `49/768`, all equal to hand integrals.
- **Simplicity**: the loop `(1/8,1/8), (7/8,1/8), (5/8,5/8), (1/2,1/8), (1/4,5/8)` has area `3/16`, no
  open-interval crossing between non-adjacent pieces, and one endpoint-interior contact (vertex 3 on
  piece 0), found by the endpoint test of section 3.3.
- **Square-root allocation**: for `|n|^2 = 2^201`, numerator `[2^100, 2^100]`, `a = 2^-300`, the
  revision-2 rule gave a value interval `1.585e29 a` wide; the revision-3 rule (section 4.4) gives
  `0.125 a` with `m = 202`.
- **Continuous E4**: `S = (u, u^2, v)`, `n = (1, 0, 0)`: `beta ≡ 0` and `alpha = u - d`; `d = 1/2` has a
  root inside the face's `u`-extent `[1/8, 7/8]` (refusal); the forbidden heights are the interval
  `[1/8, 7/8]`, and `d = 15/16` is generic with a gap below `1/16`.
- **Fixtures of section 8.2**: for the parabolic prism, the conic slab and the diamond prism every trim
  has `beta` not identically zero and every corner lies strictly inside the range box (corners listed
  in 8.2).
- **Group M construction**: on B1's deck arc, `n = C'(1/4) x e_z`, `d_t = n . C(1/4)` gives
  `g(1/4) = g'(1/4) = 0` (edge tangency); `d = d_t + 2^-10` gives two roots on the deck edge
  (`u = 0.24658`, `0.25342`); `d = d_t - 2^-10` gives none.
- **Round-3 review counterexamples, checked exactly** (revision 4): (1) for the arcs
  `C_1 = (1/u, 1/(u+1))`, `C_2 = (1/s + 1/(s+1), 1/s - 1/(s+1))` the raw resultant has degree 7 and
  equals `u^2 (u+1)^2 (1 + 2u - 4u^2 - 4u^3)`; its only root in `(1/4, 2)` is `u* = 0.58504`, the gcd of
  `F(u*, s)`, `G(u*, s)` over `Q(u*)` has degree 1 with `s* = 0.85464` in `(1/4, 2)`, and
  `C_1(u*) = C_2(s*)`; (2) the starboard unit square with the genuinely mirrored basis sums to
  `Mv = Iuv = 0`, the full rectangle's values, and an asymmetric triangle with a tilted basis agrees
  with the relation; (3) the trim `z <= 12x - 12y - 2` on `S = (u, u^2, v)` has corners at `v = 1/4` but
  reaches `v = 1` at its interior extremum `u = 1/2`; (4) for the diamond prism `n = (0, 1, 1)`,
  `d = 1` gives the graph `v = 0`, outside the top, while `d = 11/8` gives `v = 3/8` with cut points at
  `u = 5/8` and `3/8`, and `V(y + z <= 11/8) = 167/1536`; (5) on the parabolic prism
  `n = (-3/2, 1, 1)`, `d = 3/16` has the deck-edge residual `(u - 3/4)^2` (E2 inside `[5/8, 7/8]`) while
  `z <= 1/2` and `y + z <= 3/4` have only simple roots; (6) the conic slab's midpoint has
  `x^2 + y^2 - 1 = 2.346e-17`.
- **Algebraic corners are cheap**: a corner quadratic isolates to `2^-60` in `0.4 ms`; a crossing
  quartic (213-bit coefficients) in `< 1 ms`.
- **Exact rational Taylor enclosures**: `int_{0.2}^{0.4} p / W^3` (degree-6 `p`): widths `8.9e-8`,
  `6.9e-12`, `8.0e-16`, `3.1e-24`, `1.6e-32` at orders 8-32 with the complete bound of 4.3 (prototype range; the policy default order is 16, see D-CH3-7), bit lengths
  701-2130, 3-52 ms; 8 pieces of `[0, 0.9]` at order 16: `1e-19` in 0.1 s. The bound exceeds the actual
  error on the review's `10^6 / (1 + u)` term (`195.3 >= 1.133`).
- **H1 on the full B1 hull**: upright `z <= 1.5`: `V = 22.569943729716`, `A = 23.319938010999`;
  `z <= 2.9`: `V = 56.691072446746`; `z <= 3.0`: `V = 59.241180737020`; `2 x` half = full to 12 digits;
  heel `0.3` through `(2.5, 0, 1.5)` is trim-invariant because that plane misses the curved faces (hand
  check, **assumed** beyond that).
- Today's certified preparation refuses the B1 hull with `UNSUPPORTED_SURFACE`; the repository has
  exact Sturm isolation and an exact interval class without `log`, unmetered; `mpmath` is not installed.

Derived (assumed until CH3b measures it): partial fractions over a coprime factor basis keep every
Taylor term's denominator at degree `<= p` per factor, so bit growth stays in the measured regime.

## 2. Certified domain (D-CH3-1: option A, decided)

### 2.1 The class

Planar faces (straight edges, or rational-arc edges shared with an admitted curved face), plus faces
whose support is a rational tensor-product surface of degree `(p, 1)`, `p in {1, 2}`, one clamped knot
span in each direction, two control rows with **equal weight rows** and positive weights, trimmed only
by plane sections, every vertex an exact triple intersection. Refused: general NURBS, interior knots,
`p >= 3`, unequal weight rows, straight or iso-parametric edges on curved faces, curved edges shared by
two curved faces, holes on curved faces or on planar faces with arc edges, free-form trims. (Options B
and C of revision 1 were not chosen.)

### 2.2 Knot-space parametrisation and the structural fact (revision 3: one coordinate throughout)

`NurbsSurface.evaluate(s, t)` maps chart coordinates affinely onto `u_range x v_range` and evaluates in
knot parameters `(u, v)`. **CH3 uses knot parameters `(u, v)` everywhere**: in graphs, corner tests,
hint comparisons, Green primitives and the snapshot. With the exact homogeneous net over the single
clamped span `[k_p, k_{p+1}] x [k_1, k_2]`,

```
S~(u, v) = ((k_2 - v) A~(u) + (v - k_1) B~(u)) / (k_2 - k_1),     A~, B~ in Q[u]^4, degree <= p,
W(u) = A~_w = B~_w  (equal rows; W > 0 on the span),
S(u, v) = P(u, v) / W(u),   P = ((k_2 - v) A~_xyz + (v - k_1) B~_xyz) / (k_2 - k_1),   bidegree (p, 1),
P_v = dP/dv = (B~_xyz - A~_xyz) / (k_2 - k_1).
```

The face's parameter domain is the range box `[u_0, u_1] x [v_0, v_1] = u_range x v_range` (exact
rationals), inside the span. Every admitted face lies strictly inside it: every corner **and every
interior point of every trim edge** (the whole-edge range test of C3; corners alone do not suffice,
revision 4), so the chart boundary never bounds a face. The chart maps are recorded (3.4) and used
only to interpret float hints.

A plane `n . r = d` meets the support where

```
f(u, v) = n . P - d W = alpha(u) + v beta(u),
alpha = (k_2 n . A~_xyz - k_1 n . B~_xyz) / (k_2 - k_1) - d W,     beta = n . (B~_xyz - A~_xyz) / (k_2 - k_1),
```

so wherever `beta(u) != 0` the section is the rational graph `v = r(u) = -alpha(u) / beta(u)` **in knot
`v`** (checked: B1 on the span `(2,2,4,4)` gives `r = 9/4` for `z = 1/2`). Consequences:

- a trim edge is `{S(u, r_k(u)) : u in [u_a, u_b]}` with `r_k = N_k / D_k` reduced and `D_k != 0` on
  `[u_a, u_b]`; its world curve is `C_k = E_k / M_k`,
  `E_k = (k_2 D_k - N_k) A~_xyz + (N_k - k_1 D_k) B~_xyz`, `M_k = (k_2 - k_1) W D_k`, coordinates reduced
  (B1 `z`-planes: `r = h / 4`, `C_z = h`);
- two graphs cross where `N_i D_j - N_j D_i = 0` (degree `<= 2p`);
- `alpha` depends on `d` affinely, `beta` not at all;
- `beta(u*) = 0`, `alpha(u*) != 0`: no section point has `u = u*`; `alpha(u*) = beta(u*) = 0`: the plane
  contains the ruling `u = u*`; `beta ≡ 0`: the plane is parallel to every ruling (section 5, zero-beta
  rule); `alpha ≡ beta ≡ 0`: the face lies in the plane (E7).

Degree bookkeeping: `deg N_k, D_k <= p`; `deg E_k, M_k <= 2p`; the edge height numerator
`g_k = n . E_k - d M_k` has degree `<= 2p`; its critical polynomial `g_k' M_k - g_k M_k'` degree
`<= 4p - 1`; cap integrands carry `M_w^5` (degree `<= 10p`), reduced by partial fractions to factors of
degree `<= p`. A planar face's arc edge is the rational curve `E_k / M_k` of degree `<= 2p` (**not** a
conic unless the trim is iso-`v`); every predicate on it works at this degree.

### 2.3 Admission checks for the class (CH3a)

- **C1 class**: `NurbsSurface` with `degree_v == 1`, two control rows, `degree_u in {1, 2}`, clamped
  knots without interior knots, finite coordinates, `weights` `None` or equal rows and positive; finite
  ranges inside the knot domain. Else `UNSUPPORTED_SURFACE` (sub-kind recorded).
- **C2 trims**: every edge of a curved face is a `PlaneNurbsCurve` on the face's support (`support_key`,
  ranges equal) with a finite plane record; its graph `r_k = N_k / D_k` from the exact normal
  `m = u_vec x v_vec` and offset `m . origin`; `D_k` nonzero on the edge's `u`-interval (one-signed
  Bernstein coefficients over the interval's outer bounds; refine the ends to the policy depth, then
  `UNSUPPORTED_TRIM`). **Whole-edge range test** (revision 4): `v_0 < r_k(u) < v_1` on the whole closed
  edge interval `[u_a, u_b]`: sufficient when the Bernstein coefficients of `N_k - v_0 D_k` and of
  `v_1 D_k - N_k` over the interval's outer bounds all have the sign of `D_k` there; otherwise decided
  exactly at the corners (C3) and at every interior extremum of `r_k`, the roots of `N_k' D_k - N_k D_k'`
  in the open interval, where `r_k(u_e)` is compared with `v_0`, `v_1` by single-root tests; a value
  `<= v_0` or `>= v_1` anywhere refuses `UNSUPPORTED_TRIM` (sub-kind `leaves_range_box`). Checked on
  the review's upper trim `z <= 12x - 12y - 2` of `S = (u, u^2, v)` (`r = 12u - 12u^2 - 2`): corners
  `r(1/4) = r(3/4) = 1/4` pass, the interior extremum at `u = 1/2` has `r = 1 = v_1` and refuses.
  Plane trims that are iso-`v` are ordinary trims. `Straight`/`CurveOnSurface`
  edges of curved faces refuse `UNSUPPORTED_TRIM`.
- **C3 corners and hints**: every vertex of a curved face is a root `u*` of the square-free
  `N_i D_j - N_j D_i` of its two incident trims with `u_0 < u* < u_1` and `v_0 < r_i(u*) < v_1`
  (exact). **H1** (vertex): exactly one candidate's world enclosure (refined below `delta_hint`) is
  within `delta_hint` of the stored position per coordinate; else `TRIM_HINT_MISMATCH`. The edge's
  exact `u`-interval is between its corners' roots. **H2** (domain, path, direction; revision 3): with
  the library's defined knot points `(u_t, v_t)` of the stored curve at path parameters `t = 0, 1/2, 1`
  (`PlaneNurbsCurve._knot_points`), (a) `(u_0, v_0)` is within `delta_chart` of the exact corner of the
  edge's **start vertex** (`edge.start`; corner refined below `delta_chart / 4`) and `(u_1, v_1)` of the
  end vertex's corner (this fixes correspondence and direction), (b) `u_{1/2}` lies inside the exact
  `u`-interval and `|v_{1/2} - r_k(u_{1/2})| <= delta_chart`; else `TRIM_HINT_MISMATCH`. Measured: B1's
  stored edges pass (`2e-13`, `4e-13` in `u`, `0` in `v`); a shortened domain fails (a) by `0.06`; a
  distant domain fails (a) and (b). **Guarantee narrowed explicitly**: H2 establishes that the stored
  record's ends and midpoint agree with the exact edge; nothing is claimed about its other points, and
  the certified solid uses the exact edge only. `delta_chart = 2^-40` knot units.
- **C4 chart injectivity**: a rational `m` with `m . A~_xyz = c_0 W`, `m . (B~_xyz - A~_xyz) = c_1 W`,
  `c_1 != 0`, and every `v`-row's control triple non-collinear (one-signed Bernstein coefficients of a
  component of `(P_1(v) - P_0(v)) x (P_2(v) - P_0(v))` over `[k_1, k_2]`; for `p = 1` a nonzero
  segment). Else `UNSUPPORTED_SURFACE` (`not_graph_like`). C4 says nothing about the face region's shape.
- **Checks 1-6 of the planar design, extended**:
  (1) planar faces may have arc edges shared with an admitted curved face (the arc is that face's
  `E_k / M_k`, degree `<= 2p`); an unshared arc edge refuses `UNSUPPORTED_TRIM`; every vertex lies in the
  face's plane (single-root exact test);
  (2) loops: a planar face's loop is a chain of segments and rational arcs, a curved face's loop a
  chain of graph pieces; simplicity is decided by the four tests of section 3.3 (open-interval
  crossings, endpoint-interior contacts, non-adjacent endpoint equality, coincident pieces) and a
  nonzero signed area;
  (3) holes: refused (`INVALID_HOLES`) on curved faces and on planar faces with arc edges; planar faces
  with straight edges keep the planar rules on `AlgebraicPoint`s;
  (4) shell combinatorics as in the planar design (reused);
  (5) embedding: *planar-planar*: the planes' intersection line meets each loop at points (segments:
  planar predicates on `AlgebraicPoint`s; arcs: the line's two plane equations substituted into
  `E_k / M_k`, polynomials of degree `<= 2p` in `u`, common roots by gcd), collinear edges are shared
  cells, alternation along the line gives each face's intervals, and the two interval sets must meet
  exactly in the declared shared cells (B1 deck/aft: checked, section 1); coplanar pairs with arc edges
  refuse `EMBEDDING_UNDECIDED` unless their exact boxes are disjoint; *curved-planar*: the planar
  face's plane section of the support is a graph `r_m` (or the zero-beta case, which makes the face
  meet the plane only along rulings: if any ruling of the face's region lies in the plane the pair
  refuses `EMBEDDING_UNDECIDED` unless it is the declared shared edge); its portions inside the curved
  face's region (3.3 parity) must be exactly the shared edge intervals (B1 deck: checked); the planar
  face's straight segments meet the support where a degree-`2p` polynomial vanishes with `v` rational
  in `u`, which must happen only on shared cells; *curved-curved*: exact bounding boxes of the
  sub-patches over the range boxes disjoint, else `EMBEDDING_UNDECIDED`;
  (6) orientation: the exact signed volume is an enclosure certified positive, else
  `UNDECIDED_PREDICATE`.

### 2.4 Listed extensions (not in CH3)

Interior knots, `p = 3`, cylinders via `cylinder_to_nurbs`, rulings as straight edges, holes on curved
faces, unequal weight rows.

## 3. The exact model (D-CH3-2, decided)

### 3.1 Rule

Stored supports and plane records are exact data; section-curve `domain`/`path` and corner positions are
hints that select exact candidates (H1/H2) and are outside the claim beyond the recorded `delta_hint`
(`2^-40 (1 + max |coord|)`, max-norm, world units), `delta_chart` (`2^-40`, knot units) and the
selected root identities. Measured hint errors are about `2^-49` of the magnitude.

### 3.2 Algebraic numbers and points

```
AlgebraicNumber: poly (square-free primitive integer polynomial), ordinal (index among the real roots of
                 poly, increasing, from 0), interval (lo, hi) holding exactly one root by a Sturm/Descartes
                 count of 1 (never by endpoint signs alone); refine() bisects exactly
AlgebraicPoint:  root, coords (three reduced rational functions in Q[u])
Enclosure:       (lo, hi) Fractions; the exact value is in [lo, hi]
```

Identity: same `poly` and same ordinal; or, for `poly != q`, the intersection `I` of the isolating
intervals has `count_poly(I) = count_q(I) = count_gcd(poly,q)(I) = 1`; otherwise distinct, certified by
refining until the intervals are disjoint. (`(x^2-2)(x^2-3)`: ordinals 2 and 3; `I = (3/2, 8/5)` has
count 0.) Single-root predicates are exact (Bernstein signs over the interval, or `gcd(g, poly)` with a
root in it); multi-root expressions are `Enclosure`s refined to `max_isolation_bits`, else
`UNDECIDED_PREDICATE`. Mirrored corners of the full hull share `(poly, ordinal)`.

### 3.3 Loops of pieces: simplicity, region, parity, orientation (revision 3)

A face's region is the **interior of its simple closed loop** of pieces (graph pieces `v = r_k(u)` on a
curved face; segments and rational arcs on a planar face). No monotonicity is assumed (D-CH3-11 (b)).

**Simplicity** (four exact tests; the review's loop `(1/8,1/8), (7/8,1/8), (5/8,5/8), (1/2,1/8),
(1/4,5/8)` fails test (ii) only):
(i) non-adjacent pieces have no crossing in the open intersection of their `u`-intervals (roots of
`N_i D_j - N_j D_i`; for planar faces the segment-segment, segment-arc and arc-arc predicates below),
and adjacent pieces have none except at their shared corner;
(ii) **endpoint-interior**: no corner lies on a non-incident piece: for corner `u*` (root of its own
polynomial) and piece `k`, `u*` in the piece's closed `u`-interval (algebraic comparison) and
`(N_k D_i - N_i D_k)(u*) = 0` (single-root exact test, `r_i` the corner's defining graph) means a
contact and refuses `INVALID_LOOP`;
(iii) **non-adjacent endpoint-endpoint**: two non-consecutive corners equal by 3.2 refuses `INVALID_LOOP`;
(iv) **coincident pieces**: `N_i D_j - N_j D_i ≡ 0` with `u`-intervals overlapping in an open set refuses
`INVALID_LOOP`.
Planar faces: segment-segment on `AlgebraicPoint`s (planar design); segment-arc: substitute `E_k / M_k`
into the segment's line (two plane equations in space, or one in-plane line equation), degree `<= 2p`,
roots filtered to the segment's parameter range by single-root tests; arc-arc: first exact box
separation (Bernstein bounds of `E / M` over the intervals; B1's mirrored arcs are separated, section
1), else the **raw resultant** (revision 4): with in-plane coordinates `C_1 = (A_1, B_1) / M_1` in `u`
and `C_2 = (A_2, B_2) / M_2` in `s`, the cleared equations `F = A_1(u) M_2(s) - A_2(s) M_1(u)` and
`G = B_1(u) M_2(s) - B_2(s) M_1(u)` have `deg_u, deg_s <= 2p`, so `R(u) = Res_s(F, G)` has degree
`<= deg_u F deg_s G + deg_u G deg_s F <= 8 p^2` (8 for `p = 1`, 32 for `p = 2`; the review's example has
degree 7 and factors as `M_1(u)^2 (1 + 2u - 4u^2 - 4u^3)`). `R` has **spurious factors**: powers of
`M_1(u)` (at a pole of `C_1`, `F` and `G` share the factor `M_2(s)`), which have no root in the arc's
open `u`-interval by C2, and roots `u*` at which the leading coefficients of `F` and `G` in `s` vanish
together (a common "root at `s = infinity`"), which can lie inside the interval. No factor is removed
by division; instead every root `u*` of `R` isolated in the open `u`-interval of `C_1` is **validated
in the number field `Q(u*)`**: with `q` the square-free factor of `R` having `u*` as a root (identity
by 3.2), compute `gcd_s(F(u*, s), G(u*, s))` by the Euclidean algorithm over `Q[u]/(q)` (coefficient
arithmetic modulo `q`, inverses by the extended Euclidean algorithm; every zero test of a coefficient
is a single-root exact test); `u*` is a genuine crossing iff the gcd has positive degree and a real
root `s*` in the open `s`-interval of `C_2` (Sturm counts over `Q(u*)`, each sign a single-root test;
for a degree-1 gcd, `s* = -g_0 / g_1` evaluated on `u*`'s isolating interval). The verifier performs
the same validation from its own isolation. Checked on the review's pair: `R` has the single root
`u* = 0.58504` in `(1/4, 2)`, the gcd over `Q(u*)` has degree 1 and `s* = 0.85464` in `(1/4, 2)`, and
`C_1(u*) = C_2(s*)` to `1e-12` in floats. (Revision 3: no conic assumption anywhere.)

**Vertical-ray parity**: `(u_q, v_q)` with rational `u_q` chosen off every breakpoint is inside iff the
number of pieces whose open `u`-interval contains `u_q` with `r_k(u_q) > v_q` is odd (B1 deck checked).

**Orientation (one convention end to end; revision 3)**: the chart loop is traversed in the model's
direction (edge forward flags, corner order along each edge); `sigma_chart = sign(-oint_loop v du)`
(an `Enclosure` with a certified sign) is `+1` for anticlockwise. The model's loops run anticlockwise
about the outward normal, so the chain **already carries the outward orientation** and no face sign is
applied anywhere: `V_f = -(1/3) oint_chain F du` (4.1) gives `+25/4608` for the anticlockwise diamond
and `-25/4608` for the clockwise one (checked). Admission check 6 (signed volume `> 0`) confirms the
global orientation.

**Clipping by a graph half-region** `H_w = {alpha_w + v beta_w <= 0}` (`v <= r_w` where `beta_w > 0`,
`v >= r_w` where `beta_w < 0`, all or nothing where `beta_w = 0`, pieces split at the roots of `beta_w`):
cut points are the roots of `N_w D_k - N_k D_w` on each loop piece (ordinal along the edge recorded);
loop pieces are split there and a sub-piece is kept iff its rational test point is in `H_w`; the graph
`r_w` is split at its cut points and a portion is kept iff its test point is inside the loop (parity).
Kept loop sub-pieces keep the model direction; a kept graph portion is directed
`-sign(beta_w) * sigma_chart` (`+1` = increasing `u`), which places the clipped region on the same side
as the loop does. The result is an oriented chain (any number of cycles), used only as a chain.
**Cap**: every cut piece (a planar cut segment of the planar design's clipped loop, or a kept graph
portion) appears on the cap **reversed**; the cap is then anticlockwise about `n`, the outward normal of
the cap of `K = S ∩ {n . r <= d}`. The verifier uses the same convention (its independence is in the
methods). Checked end to end on the parabolic prism (`53/1536`, `29/3072`, `49/768`).

### 3.4 Snapshot document (convention version 2)

Per curved face `"support": {"kind": "ruled", "degree_u", "net", "span": [[k_p, k_{p+1}], [k_1, k_2]],
"u_range", "v_range", "sigma_chart"}`; per section edge `{"kind": "section", "plane": [origin, u, v],
"domain_hint": [hex, hex], "path_hint": int, "graph": [N, D]}`; per vertex on a curved face
`{"corner": {"poly", "ordinal", "hint": [hex x3]}}`. Isolating intervals are not in the document.
Planar-only preparations produce exactly the convention-1 document.

## 4. Quantities and bounds (D-CH3-3, D-CH3-4, decided)

### 4.1 Decomposition (revision 3: one orientation, knot `v`)

For the `below` convention, `K = S ∩ {n . r <= d}` is bounded by clipped planar faces, clipped curved
faces and the cap:

- **Planar face**: fan from `q_f` in the face's plane (the stored plane origin), tetrahedra at the body
  origin, edges in the model's loop direction. Straight edge `a -> b`: `V += det(q_f, a, b) / 6`,
  `J += det(q_f, a, b) (q_f + a + b) / 24` on `AlgebraicPoint`s. Arc edge `C_k = E_k / M_k` from `u_a`
  to `u_b` in the loop's direction: `V += int det(q_f, C_k, C_k') / 6 du`,
  `J += int det(q_f, C_k, C_k') (q_f + 2 C_k) / 24 du`, with `det(q_f, C, C') = det(q_f, E, E') / M^2`
  (checked: `1/18`). Clipping: the kept portions of the loop (planar design's clipped loop, in the
  loop's direction) with the arc portions where `g_k M_k <= 0` and the planar cut segments.
- **Curved face**: `V_f = -(1/3) oint_chain F_V du`, `J_{f,i} = -(1/4) oint_chain F_{J,i} du`, where
  `F_V(u, v) = int_{k_1}^{v} P . (P_u x P_v) / W^3 dv'` and `F_{J,i}(u, v) = int_{k_1}^{v} P_i P . (P_u x P_v) / W^4 dv'`
  (polynomials in `v` of degree 3 and 4; `P_i` the homogeneous numerator of coordinate `i`;
  `S . (S_u x S_v) = P . (P_u x P_v) / W^3` since `P . (P x P_v) = 0`; the lower limit is immaterial for a
  closed chain). On a graph piece `v = N / D` the integrand `F(u, N/D)` has denominator `W^3 D^3`
  (`V`) or `W^4 D^4` (`J`). The chain is the clipped chain of 3.3. Checked on the diamond and the prism.
- **Cap**: the chain is the reverse of every cut piece. Fan from `q = o + ((d - n . o) / (n . n)) n`:
  planar cut segments by the planar triangle formulas; a cut arc `C_w = E_w / M_w` (now traversed in the
  cap's direction) by

  ```
  dA~ = ((C_w - q) x C_w') . n / 2 du      (A = int dA~ / |n|),
  Mu~ = int (q_u + 2 u_p) / 3 dA~,  Iuu~ = int (q_u^2 + 2 q_u u_p + 3 u_p^2) / 6 dA~,
  Iuv~ = int (q_u q_v + q_u v_p + u_p q_v + 3 u_p v_p) / 6 dA~  (Mv~, Ivv~ likewise),
  ```

  `(u_p, v_p) = (e1 . (C_w - o), e2 . (C_w - o))`. Checked: `A = 49/768` on the prism.

Every contribution is a sum of `int Num / Den du` over pieces with `Den` a product of powers of `W`,
`D_k`, `D_w` (one-signed on the piece) and rational or algebraic ends. No 2-D quadrature, no
transcendental function.

### 4.2 Breakpoints and pieces

Corners; cut points (with ordinals); roots of `beta_w` in the face's `u`-extent; roots of `g_k` on arc
edges of planar faces. Sorted by 3.2. Between consecutive breakpoints each chain piece is one 1-D
integral per quantity.

### 4.3 Enclosing a 1-D rational integral (D-CH3-3 (ii))

**Reduction**: a pairwise-coprime factor basis of all denominators by gcd refinement; partial fractions
into a polynomial part (exact) and terms `L / Q^k`, `deg L < deg Q`.

**Term enclosure** on a rational piece `[a, b]`, `c = (a + b) / 2`, `h = (b - a) / 2`, `g = Q^{-k}`:

```
g^{(j)} = P_j / Q^{k+j},  P_0 = 1,  P_{j+1} = P_j' Q - (k + j) P_j Q';
t_K(u) = sum_{j<=K} g_j (u - c)^j,  g_j = P_j(c) / (Q(c)^{k+j} j!);     main = int_a^b L t_K du  (exact);
B_g = max_{[a,b]} |P_{K+1}| / (min_{[a,b]} |Q|)^{K+1+k} / (K+1)!,  extrema from absolute Bernstein
      coefficients over [a, b] (|Q|'s minimum is valid because Q's coefficients are one-signed there);
rho = B_g * L_max * 2 h^{K+2} / (K + 2),   L_max from |Bernstein coefficients of L|;
T in [main - rho, main + rho].
```

Checked: `10^6 / (1 + u)`, `K = 8`: error `1.133`, `rho = 195.3`.

**Algebraic endpoints**: inner rational interval `[a_hi, b_lo]` by the term enclosure; end pieces by
`[-e, e]`, `e = (a_hi - a_lo) L_max / min|Q|^k` over `[a_lo, a_hi]` (refinable by bisection).

**Accumulation and allocation**: a quantity's enclosure is `exact_part + sum` of all term and endpoint
intervals; its width is the sum of all `2 rho` and `2 e`. Target `tau` per quantity is split equally
over pieces, then over terms; a bisected piece gives each child **half** of its parent's allowance;
endpoint enclosures draw from the piece's share first. Orders start at 8 and rise to
`max_taylor_order` before bisection (8, 16 by default; 32 only if opted in; 2026-10-21 owner decision (a): policy default 16; 32 opt-in; reason: bit cap, ~140 bits per end-denominator bit at order 32); `max_pieces_per_integral`, `max_isolation_bits` bound the rest;
failure refuses `ACCURACY_UNACHIEVABLE` with the achieved width.

### 4.4 Forms, bounds and the claim (D-CH3-4; revision 3: square-root allocation)

- **exact**: only when no algebraic or enclosed term entered the quantity (`EMPTY`'s `V = 0`; the
  fixtures of 8.2 with `W = 1`, `D = 1` and rational corners, where every integral is a polynomial with
  rational ends).
- **enclosed**: the claim is `[lo, hi]` with `hi - lo <= a`; the engine computes a rigorous
  `core` of width `w <= a / 3` and publishes `claim = [c_lo - w, c_hi + w]`. **Only the claim is
  verified**; `core`, piece counts, orders and depths are untrusted `execution` metadata. Forms:
  `"enclosure": {"lo", "hi"}`, `"binary64"` the outward rounding of the claim (its width is its bound;
  a finer request refuses `ACCURACY_UNACHIEVABLE` for that form), `"decimal"` of the claim; `exact`
  absent (presence is `MALFORMED_PROOF`).
- **Waterplane quantities and `1/|n|`**: the integrals give numerator enclosures `[N_lo, N_hi]`, value
  `= numerator / |n|`. With `s_lo = isqrt(|n|^2 2^{2m}) / 2^m`, `s_hi = s_lo + 2^-m`, the value interval
  `[min(N_lo/s_lo, N_lo/s_hi), max(N_hi/s_lo, N_hi/s_hi)]` has width at most
  `(N_hi - N_lo) / s_lo + max(|N_lo|, |N_hi|) (s_hi - s_lo) / s_lo^2`. **Allocation in output units**
  (revision 3): the numerator core target is `tau_N = (a / 6) s_lo` so the first term is `<= a / 6`,
  and `m` is increased until the second term `max(|N_lo|, |N_hi|) 2^-m / s_lo^2 <= a / 6` (a logarithmic
  loop, bounded by `max_bits`), so the value core has width `<= a / 3` and the claim `<= a`. Checked on
  `|n|^2 = 2^201`, `N = 2^100`, `a = 2^-300`: width `0.125 a` (`m = 202`) where the revision-2 rule gave
  `1.585e29 a`. Binary64 outward rounding adds at most 2 ulps; if that breaks `hi - lo <= a` the
  binary64 form refuses `ACCURACY_UNACHIEVABLE`. Scale invariance: `(n, d)` vs `(2n, 2d)` claims
  intersect and both meet `a` (group S).

Accuracy requests: `tau = a / 3`; default claim width `10^-12` without a request; `accuracy_met`
recomputed by the verifier from the claim.

## 5. Section topology, degeneracy and one-sided limits (D-CH3-5; revision 3: zero-beta rule)

### 5.1 Events and zero polynomials (precedence order)

| Code | Condition | Handling |
|---|---|---|
| E7 face in plane | `alpha_w ≡ 0` and `beta_w ≡ 0` | the face lies in the plane: planar rule (entirely above for `below`, entirely below for `above`; never double-counted); no cut points on its edges; its corners are E1 contacts |
| Z zero-beta face | `beta_w ≡ 0`, `alpha_w` not `≡ 0` | the height along rulings is `h(u) = n . S(u, .) = ((k_2 n . A~ - k_1 n . B~) / (k_2 - k_1)) / W` (constant along each ruling). **Forbidden heights** = the range `[h_min, h_max]` of `h` over the face's `u`-extent (ends and critical points of `h`, algebraic). `d` in `[h_min, h_max]`: the plane contains a ruling of the face: refuse `CURVED_SECTION_DEGENERACY / ruling_in_plane`. `d` outside: the face is entirely on one side (sign of `alpha_w`, constant), no cut arcs; in the gap rule the whole interval `[h_min, h_max]` is excluded (checked: `S = (u, u^2, v)`, `n = (1,0,0)`, range `[1/8, 7/8]`, `d = 15/16` generic with `eps_0 < 1/16`) |
| E4 ruling in plane | `beta_w` not `≡ 0`, `gcd(alpha_w, beta_w)` with a root in the face's `u`-extent | refuse `CURVED_SECTION_DEGENERACY / ruling_in_plane` (E3 interior tangency is the sub-case with `n . S_u = 0` there: sub-reason `interior_tangency`) |
| E5 edge in plane | `g_k ≡ 0` | no cut points on the edge; E2 not evaluated for it; its corners are E1 contacts; sides from the perturbed plane (5.2) |
| E1 corner on plane | `g_k(u*) = 0` at a corner | supported (5.2) |
| E2 edge tangency | a root of `g_k' M_k - g_k M_k'` in the open edge interval with `g_k = 0` there (edges with `g_k` not `≡ 0`) | refuse `CURVED_SECTION_DEGENERACY / edge_tangency` |
| E6 planar events | planar vertex heights, planar edges/faces in the plane | planar design |

**Completeness.** For fixed `n` and `d` in an open interval free of E1/E2/E4/E5/E6 heights and of every
zero-beta face's forbidden interval: roots of `g_k(.; d)` are simple and interior, so cut points keep
their number and order; on every face with `beta_w` not `≡ 0` the graph `r_w(.; d)` moves monotonically
in `v` at each `u` (`dr_w/dd = W / beta_w`), is defined off the roots of `beta_w` and has no point on
them (no E4), so its portions inside the region begin and end only at cut points; zero-beta faces stay
entirely on one side. The cut-point graph is constant, hence loops and nesting are constant: the
inventory is complete.

### 5.2 The certified event gap

1. Isolate every event height not at `d` (E1 `n . C_k(u*)`; E2 heights at the critical roots; E4
   heights `d` with `d W(u*) = (k_2 n . A~ - k_1 n . B~)(u*) / (k_2 - k_1)` at the roots `u*` of `beta_w`;
   zero-beta forbidden intervals `[h_min, h_max]`; E6 vertex heights) and refine until a rational
   `eps_0 > 0` has no event height and no forbidden interval point in `(d - eps_0, d) ∪ (d, d + eps_0)`
   (an event at `d` itself is the degeneracy; undecidable separations refuse `UNDECIDED_PREDICATE`).
2. Compute the combinatorics at `d ∓ eps_0 / 2`.
3. Limit positions at `d` by continuity of ordinals; quantities at `d` over the limit chain; edges in
   the plane counted on the side where the perturbed plane crosses them; `V`, `J` continuous, `below`.

Topology identities: sorted `(edge id, ordinal)` cut-point identities per loop.

## 6. The independent verifier (D-CH3-10)

Keeps steps (0)-(2) and (5) of the planar design; (3)-(4) extended: re-derives nets, graphs, corners
(`(poly, ordinal)`, own isolation with its own single-root counts), loops, chains and `sigma_chart` with
its own code; recomputes contacts, event codes, occupancies, gaps, cut-point identities, loops,
nesting and exact quantities (exact equality); for enclosed quantities computes its own rigorous
enclosure `E_v` by a different scheme (one-sided Taylor at the left end, own order schedule and
bisection), refines until `width(E_v) <= (hi - lo) / 3` or its budget stops, and accepts iff
`E_v ⊆ [lo, hi]` (else `VERIFICATION_FAILED / enclosure`); a claim excluding the true value can never
be accepted (`E_v` always contains the value); an insufficient verifier budget ends in
`BUDGET_EXHAUSTED` (phase `verify`). Execution metadata: well-formedness only. Same orientation
convention as section 3.3/4.1. Trust model as in revision 2 (identities of section 4 and the
completeness argument of 5.1; the verifier's exact polynomial arithmetic, counts and Taylor/Bernstein
bound; outward rounding); the verifier imports nothing from the provider.

## 7. Work and checkpoints (D-CH3-7; revision 3: aligned with the restated contract, amendment 6d)

The restated planar contract, applied to CH3: (1) every loop written in the package checkpoints at least
once per element or per 64 KiB, before that element's work; (2) every non-preemptible span is a native
operation over at most one value or collection whose size is bounded by a named limit times a constant;
(3) a span whose size is not bounded by a policy limit known before it runs is charged in proportion
before it runs and followed by a checkpoint.

Consequences for CH3 (replacing revision 2's bulk rows):

- **Coefficient loops are loops**: polynomial products, pseudo-remainders, shifts, Bernstein
  conversions, Sturm/Descartes evaluations, Taylor coefficient recurrences, interval operations over
  vectors — all are written as Python loops that reserve one unit (`poly_coefficient`, `sign_count`,
  `root_refine`, `interval_op`, `bernstein_coefficient`, `piece_split`, `claim_element`,
  `serialize_element`) **before each element's work**. No composite unit exists; `taylor_coefficient`,
  `sturm_step`, `event_height` of earlier revisions are removed.
- **Native spans**: the only non-preemptible spans are single native big-integer or `Fraction`
  operations (multiply, add, floor-divide, `gcd`, `isqrt`, `Fraction` normalisation) on values whose bit
  length is bounded by `max_bits` (checked on every stored value, as in the planar design) times a
  constant (a product of two capped values has at most `2 max_bits` bits before its own check).
- **Unbounded-before-running spans**: `isqrt` of `|n|^2 2^{2m}` in 4.4 (`m` chosen during the run) is
  bounded by `max_bits` through the value check before it runs (`|n|^2 2^{2m}` is stored and checked
  first); the proof's UTF-8 decode and the model document encoding keep the planar rules (charged in
  proportion, checkpoint after). Nothing else runs un-charged.
- **Tests** (CH3b): for each coefficient loop a cancellation injected at element `k` stops with exactly
  the units of the first `k` elements reserved; a deadline set during a loop stops at the next element;
  the bit cap refuses before any native operation on an over-cap value.

Policy limits bound every loop: `max_isolation_bits`, `max_taylor_order`, `max_pieces_per_integral`,
`max_refinement_rounds`.

## 8. Q08: evidence and the consumer's part (revision 3: fixtures, matrix, relations)

### 8.1 Run set and success matrix (fixed before any assessment run)

Models: `build_b1_half_hull()` and `build_b1_hull()` (defaults); units `m`; forms `("enclosure",
"binary64", "decimal")`; accuracies `1e-9` for every quantity. ANYstability's convention: `n = R[2]`,
`d = n . point`, basis `R[0]`, `R[1]`, origin `point`, `R = Ry(-trim) Rx(heel)` with float entries.

| Group | Planes | Expected |
|---|---|---|
| U upright | `z <= h` through `(2.5, 0, h)`, `h in {0.75, 1.5, 2.25, 2.9}` | PARTIAL, generic, one outer loop, no hole; all quantities enclosed |
| B boundaries | `h = 0.5`, `h = 3.0` | degenerate. `h = 3`: occupancy FULL; `below` PRESENT (one loop; `A` = deck area), `above` ABSENT; **contacts** half hull: 4 vertices (2 bow corners, 2 centreline points), 4 edges in plane (1 arc + 3 straight), 1 face (deck); full hull: 4 vertices (2 starboard + 2 port corners), 4 edges (2 arcs + 2 across lines), 1 face. `h = 1/2`: occupancy EMPTY, `below` ABSENT, `above` PRESENT; `V` exact `0`; contacts likewise on the bottom |
| N near-event | `h = 3 - 2^-40`, `h = 1/2 + 2^-40` | generic PARTIAL; must answer |
| I inclined | heel `in {-0.5, -0.3, -0.1, 0.1, 0.3, 0.5}` x trim `in {-0.02, 0, 0.02}` through `(2.5, 0, z)`, `z in {0.75, 1.5, 2.25, 2.9, 3.0}` (90 per model) | PARTIAL unless the plane misses the hull (EMPTY/FULL, generic); planes without a cut arc (heel `+-0.3` at `z = 1.5`) answer; a typed degeneracy refusal is a correct answer and is recorded (none expected) |
| T trim-only | heel `0`, trim `in {-0.02, 0.02}` through the five draft points (10 per model) | PARTIAL, generic; used for the half/full relation (symmetric planes) |
| S scale | every I and T plane with `(2n, 2d)` | claims intersect, both meet `a` |
| D constructed degeneracies (half hull) | E2: `n = C'(1/4) x e_z` on the deck arc, `d_t = n . C(1/4)` (rational; `g(1/4) = g'(1/4) = 0` checked); E4: `n = (B(1/4) - A(1/4)) x (1, 0, 0)`, `d = n . A(1/4)`; E3: `n = S_u x S_v` at `(1/4, 1/2)`, `d = n . S`; Z: a parabolic-prism plane `n = (1, 0, 0)`, `d = 1/2` (zero-beta, forbidden interval) and `d = 15/16` (generic, empty section); E2 on F-A: `n = (-3/2, 1, 1)`, `d = 3/16` (deck-edge residual `(u - 3/4)^2`); E1: the corner `(3/2, y*, 1/2)` with `n = (1, 0, 2)`, `d = 5/2`; E1 x2 (full hull): the mirrored deck corners with `n = (1, 0, 0)`, `d = 3/2` | E2/E3/E4/Z-inside refuse `CURVED_SECTION_DEGENERACY` with the sub-reasons; Z-outside answers; E1 cases answer with both limits |
| M multiple roots (mandatory) | `n = C'(1/4) x e_z`, `d = d_t + 2^-10` on the half hull (checked: two roots on the deck edge at `u = 0.24658`, `0.25342`; `d_t - 2^-10` gives none) | generic; two cut points with ordinals 0 and 1 on the deck edge |

Success: every plane answers or refuses with a typed category; every answer verifies with the CH3d
verifier against the run's artifact identity; recorded per plane: state, contacts, loop counts,
claims, widths, work counts, wall time; manifest resource limits apply.

### 8.2 References and fixtures (uncertified; consistency checks, not proof)

- **H1** `hydrostatics()` (`order = 24`, `rtol = 1e-8`): recorded with its `estimate`; flagged when
  `|H1 - mid(claim)| > max(width(claim), 1e-8 max(1, |H1|))`; a flag triggers investigation; **not an
  acceptance gate**.
- **Station closed forms** (group U, waterplane quantities only): the section at `z = h` bounded by the
  frames, the centreline (half hull) and the conic arc `S(u, h/4)` (an iso-`v` section is a conic),
  evaluated with `mpmath` at 40 digits through the rational parametrisation.
- **High-precision 2-D quadrature** (all groups): `mpmath.quad` at 30 digits of `S . (S_u x S_v) / 3`
  and `x_i S . (S_u x S_v) / 4` over the clipped chart regions and the planar faces (regions from the
  exact graphs; own error estimate recorded); a reference outside a claim flags the plane.
- **Sandwich** (group U): `claim(V(h_2)) - claim(V(h_1))` intersects `(h_2 - h_1) [lo A(h_1), hi A(h_2)]`
  (B1's `A(z)` increases with `z`, a fixture-specific expectation).
- **Half/full relation** (revision 4; convention stated explicitly). Let `R` be the reflection
  `(x, y, z) -> (x, -y, z)`. The full hull is `stb ∪ R(stb)`, so its clipped solid is
  `[stb ∩ H(n, d)] ∪ R[stb ∩ H(n~, d)]` with `n~ = R n`. The second term is computed on the starboard
  half hull with the **genuinely mirrored query** `(n~, d, o~ = R o, e1~ = R e1, e2~ = R e2)`; for a
  point `r = R r'` of the mirrored part, `e1 . (r - o) = (R e1) . (r' - R o) = e1~ . (r' - o~)` and
  likewise for `e2`, so both plane coordinates and the area element are preserved and **every
  waterplane quantity adds**: `A, Mu, Mv, Iuu, Ivv, Iuv` of the full hull `= Q(n, d, o, e1, e2)
  + Q(n~, d, o~, e1~, e2~)` on the half hull. `V` adds; `J_full = J(n, d) + R J(n~, d)`, i.e. `Jx`, `Jz`
  add and `Jy = Jy(n, d) - Jy(n~, d)` (the only sign flip; it is the world coordinate that reflects).
  The mirrored basis has opposite handedness (`(e1~ x e2~) . n~ = -(e1 x e2) . n`), which the query
  allows. The centre face contributes `0` to `V`, `J` (its plane contains the body origin, so every
  determinant with `q_f` vanishes) and its cut segment cancels between the two caps. Checked: the
  starboard unit square with `e1 = x`, `e2 = y` has `Mv = 1/2`, `Iuv = 1/4`; the mirrored term with
  `e2~ = R e2` gives `-1/2`, `-1/4`; the sums `0`, `0` equal the full rectangle's values; an asymmetric
  triangle with a tilted basis `e1 = (1, 1, 0)`, `e2 = (-1, 2, 0)` gives `(A, Mu, Mv, Iuv) =
  (1/2, 1/3, -1/3, -5/24)` both directly and by the relation. Criterion for every group: the full
  hull's claim intersects the interval sum of the two half-hull claims and both meet `a`. For
  **symmetric queries** (`n_y = 0`, `o_y = 0`, `e1_y = 0`, `e2` parallel to `y`: groups U, B, N, T)
  `n~ = n`, `o~ = o`, `e1~ = e1`, `e2~ = -e2`, so the relation reduces to doubling `V`, `Jx`, `Jz`, `A`,
  `Mu`, `Iuu`, `Ivv` and `0` for `Jy`, `Mv`, `Iuv`. Group I (nonzero heel) uses the general relation
  only. Bitwise equality of cores is an observation.
- **Transformations on the curved domain** (half hull, groups U and I at heel `0.3`): translation
  `(7, -3, 11)`; rotation by the exact `3/5, 4/5` matrix after scaling by 5; reflection `y -> -y`;
  moment origin shifted by `(1, 2, 3)` projected into the plane; basis `(e2, e1)` and `(2 e1, e2)`;
  normal rescaling (S). Expected: `V`, `A` invariant; `J' = R J + V t` (scaled `s^4`, `s^3`); claims
  intersect the transformed originals and meet `a`.
- **Fixtures (all checked to satisfy C1-C4: `beta` not identically zero on every trim, corners strictly
  inside the range box; section 1)**:
  - **F-A parabolic prism** (exact values): support `S = (u, u^2, v)` as a degree-`(2,1)` patch
    (rows `(0,0,0), (1/2,0,0), (1,1,0)` and the same at `z = 1`, weights `None`, ranges `(0, 1)`);
    trims `z = 1/4`, `z = 3/4` (`beta = 1`), `z = x + 1/8` and `z = x - 1/8` (`n = (-1, 0, 1)`,
    `beta = 1`); corners `(1/8, 1/4), (3/8, 1/4), (7/8, 3/4), (5/8, 3/4)`; closed by the planes `y = 0`,
    `z = 1/4`, `z = 3/4`, `z = x ± 1/8`. Every integral is polynomial with rational ends, so the engine
    returns **exact** forms: `V = 53/1536`, `V(z <= 1/2) = 29/3072`, `A(z = 1/2) = 49/768` (hand
    integrals, and the chain convention reproduces them). `beta = 1` for every plane with `n_z = 1`
    makes the waterplane a graph but does **not** make the plane generic (revision 4): each listed
    query is subjected to the event tests of 5.1, with these exact outcomes: `z <= 1/2`: no root on the
    bottom and deck edges, one simple root on each slanted edge (generic); `y + z <= 3/4`: likewise
    generic; `n = (-3/2, 1, 1)`, `d = 3/16` (the review's plane): one simple root on each slanted edge
    and on the deck edge `v = 3/4` the height numerator `(u - 3/4)^2` with a double root at `u = 3/4`
    inside `[5/8, 7/8]`: **expected refusal** `CURVED_SECTION_DEGENERACY / edge_tangency` (listed in
    group D); `n = (1, 0, 0)` is the zero-beta case (group D).
  - **F-B conic slab**: support the quarter unit circle `(1,0), (1,1), (0,1)` with weights `(1, w, 1)`
    and `z = 2 v` (rows at `z = 0`, `z = 2`); trims `z = 1/2`, `z = 3/2`, `3x + z = 2`, `3y + z = 2`
    (`beta = 2W` for all); corners `u = 0.1147, 0.3411, 0.6589, 0.8853` at `v = 3/4, 1/4, 1/4, 3/4`;
    solid `{x, y >= (2 - z)/3, (x, y) inside the conic, 1/2 <= z <= 3/2}` closed by the planes `z = 1/2`,
    `z = 3/2` (regions bounded by the arc and the lines `x = (2 - z)/3`, `y = (2 - z)/3`) and the two
    slanted planes (meeting along `x = y = (2 - z)/3`). **The curved boundary is the conic of the stored
    `w`, not the unit circle** (revision 4): the rational quadratic through `(1, 0)` and `(0, 1)` with
    tangents `x = 1`, `y = 1` and the double `w = 0.7071067811865476`; its midpoint `u = 1/2` is
    `x = y = (1 + 2w) / (2 (1 + w))` with `x^2 + y^2 - 1 = +2.346e-17` exactly rational and nonzero. The
    reference is the `mpmath` quadrature (40 digits) of this exact-`w` conic through its rational
    parametrisation (never the circle formulas); upright and inclined planes; the only fixture with
    `W != 1`.
  - **F-C diamond prism** (planar cross-check): top support `S = (u, v, 1)` as a degree-`(1,1)` patch
    (rows `(0,0,1), (1,0,1)` and `(0,1,1), (1,1,1)`, ranges `(0, 1)`), trims `y = x + 1/4`,
    `y = -x + 5/4`, `y = x - 1/4`, `y = -x + 3/4` (`beta = ±1`), corners `(1/4,1/2), (1/2,1/4),
    (3/4,1/2), (1/2,3/4)`; the prism over the diamond down to `z = 0` with four vertical planar sides
    and the planar diamond bottom. `z <= 1/2` misses the top; the crossing query is `n = (0, 1, 1)`,
    `d = 11/8` (revision 4: on the top `n . r = y + 1` lies in `[5/4, 7/4]`, so `d = 1` missed it; `11/8`
    cuts the top along `y = 3/8`, the graph `v = 3/8`, meeting the trims `y = x - 1/4` at `u = 5/8` and
    `y = -x + 3/4` at `u = 3/8`, both strictly inside their edge intervals and away from the corners;
    `beta = 1`; the cut arc is a straight line with rational parametrisation). In both cases every
    integral is polynomial with rational ends and the engine must return **exact** forms equal to CH1's
    for the same prism with a `Plane` top; the hand value of the crossing query is
    `V(y + z <= 11/8) = 167/1536` (exact piecewise integration of the diamond area below `y = 11/8 - z`).
  - **Review counterexamples as unit fixtures**: the `(u, u^2, v)` hexagon and the endpoint-contact
    loop (3.3), the fan region (`1/18`), the `10^6 / (1 + u)` term, the sqrt case of 4.4, the
    `(2,2,4,4)` span, the zero-beta case.

### 8.3 Consumer's part

Mapping claims into its `IntegralSample` intervals, PSD checks, Capytaine comparison, P0/P1,
equilibrium/GZ, admission.

## 9. Slices

| Slice | Output | Acceptance criterion | Author | Depends on |
|---|---|---|---|---|
| CH3a | C1-C4, knot-space nets and graphs, H1/H2, loops of pieces (four simplicity tests, parity, `sigma_chart`), planar faces with rational-arc edges, extended checks 1-6, snapshot v2, new refusal categories, policy fields | `pytest tests/test_certified_hydrostatics_ch3_admission.py`: both B1 builds and F-A/F-B/F-C admit; each new refusal from a minimal case (unequal weight rows, interior knot, `Straight` edge on the bow, vertex hint moved `1e-6`, shortened and distant edge domains, reversed edge direction, non-graph-like patch, overlapping bows, a face reaching the range box, the endpoint-contact loop, coincident pieces); the hexagon admits; every CH1 fixture's digest and proof bytes unchanged | Sonnet (Opus review) | CH3b predicate layer |
| CH3b | exact kernel: polynomials over `Q`, coprime basis, partial fractions, `AlgebraicNumber` identity/order, `Enclosure`, 1-D term integrator with allocation, `1/|n|` bounds, bivariate resultants and the `Q(u*)` gcd validation (3.3), whole-edge range test, per-element metering | `pytest tests/test_certified_hydrostatics_ch3_kernel.py`: enclosures contain `mpmath` references for `int L / Q^k` (50 random pieces, `k in 1..5`, targets `1e-6..1e-20`; max bits recorded); the finding-2 term; identity cases; allocation sums; sqrt case (`|n|^2 = 2^201`); the round-3 resultant pair (`u* = 0.58504`, `s* = 0.85464`) plus a spurious leading-coefficient root and a pole factor rejected; **obligation from review round 4:** the `Q[u]/(q)` gcd runs over the irreducible factor of `q` containing `u*` (or refines factors when an inverse fails), tested on the review pair started from the reducible square-free `q = u(u+1)(1+2u-4u^2-4u^3)`, engine and verifier both recovering `u* ~ 0.5850432433`, `s* ~ 0.8546376797`; the range-test trim `12u - 12u^2 - 2` refused; cancellation at element `k` and deadlines per loop; bit-cap refusals | Opus | - |
| CH3c | curved query engine: breakpoints, chains, clipping, cap reversal, events E1-E7 and Z with the gap, enclosed quantities, forms, proof v2 | `pytest tests/test_certified_hydrostatics_ch3_query.py`: F-A exact values, F-C exact equality with CH1, F-B references, hexagon solid, B1 groups U, B, N, D, M, T and 6 planes of I, scale pairs, transformations, sandwich, half/full general relation; references inside claims or flagged | Sonnet (engine), Opus (4.3, 5, review) | CH3a, CH3b |
| CH3d | verifier extension (section 6), separate session | `pytest tests/test_certified_hydrostatics_ch3_verify.py`: agrees on every CH3c fixture; refuses: claim shifted by its width (`enclosure`), `lo`/`hi` swapped (`MALFORMED_PROOF`), claim wider than `a` with `accuracy_met` true (`accuracy flag`), `exact` on an enclosed quantity, altered `(poly, ordinal)` or hint (`snapshot`), half-hull result for the full hull (`model`), CH1 attacks on a CH3 preparation; **an explicitly insufficient verifier budget** on an honest claim ends in `BUDGET_EXHAUSTED` (phase `verify`), never acceptance (revision 3); altered execution metadata leaves acceptance unchanged | separate Sonnet session (Opus review) | CH3c |
| CH3e | Q08 run set (8.1), references (8.2), delivery addendum | all planes answer or refuse typed; all answers verify; matrix met; flags investigated; run record before assessment | Opus | CH3c, CH3d |

Codex Sol reviews every slice; this design gets Codex Astra reviews until clean before CH3a starts.
Patch version bump at CH3e (`0.4.7`).

## 10. Open risks

- Bit growth at order 32 with `k = 5` terms (measured 2130 bits on one factor); mitigation: more pieces
  at lower order; measured in CH3b. **Resolved by 2026-10-21 owner decision (a): policy default 16; 32 opt-in; reason: bit cap, ~140 bits per end-denominator bit at order 32** (see CH3b measurements: algebraic ends exceed the cap at order 32 beyond ~27-bit ends).
- Multi-root predicates refuse rather than resolve unrecognised true zeros.
- Event isolation near coincident distinct events (bounded, refusable).
- Runtime unmeasured end to end.
- Curved-curved embedding by box disjointness only.
- Hint tolerances are policy numbers.
- Arc-arc raw resultants of degree `<= 8 p^2 = 32` with the `Q(u*)` validation (3.3) when boxes
  overlap (not needed for B1, whose mirrored arcs are box-separated); the number-field gcd is the most
  intricate kernel routine and gets its own CH3b tests (the review's pair, a pair with a spurious
  leading-coefficient root, a pair with a pole factor).
- `mpmath` as a test-only dependency (D-CH3-8).
- Whether a quantity is transcendental is not decided; the engine reports an enclosure whenever an
  enclosed term entered.

## 11. Owner decisions (decided 2026-10-20, all as recommended)

- **D-CH3-1** domain: option A (section 2).
- **D-CH3-2** exact model: supports exact; trims = exact plane ∩ surface; hints checked by H1/H2
  (`delta_hint = 2^-40 (1 + max |coord|)`, `delta_chart = 2^-40`), recorded, outside the claim.
- **D-CH3-3** quantity method: exact rational Taylor models with Bernstein remainders over a coprime
  factor basis; closed forms only as test references.
- **D-CH3-4** forms: claim protocol (claim = core ± width, only the claim verified, `exact` absent,
  binary64 as outward rounding, default `10^-12`); square-root allocation in output units (4.4).
- **D-CH3-5** degenerate planes: E1, E5, E6, E7 and zero-beta-outside supported by the certified event
  gap; E2, E3, E4 and zero-beta-inside refuse `CURVED_SECTION_DEGENERACY`.
- **D-CH3-6** kernel re-implemented inside `certified_hydrostatics` with metering; verifier has its own.
- **D-CH3-7** policy defaults: `max_bits` 4096, `max_isolation_bits` 4096, `max_taylor_order` 16,
  `max_pieces_per_integral` 1024, `max_refinement_rounds` 8, `default_enclosure_width` 10^-12.
  **Amended 2026-10-21 owner decision (a): policy default 16; 32 opt-in; reason: bit cap, ~140 bits per end-denominator bit at order 32.** (Earlier value: 32.)
- **D-CH3-8** `mpmath` as a test-only dependency.
- **D-CH3-9** convention version 2 for curved preparations; planar stays 1 byte for byte.
- **D-CH3-10** verifier by a separate session, different isolation algorithm and Taylor scheme.
- **D-CH3-11** face regions: (b) general simple loops (interiors of loops of pieces, parity, oriented
  chain, Green without cycle assembly); no monotonicity check; domain unchanged.

## 12. Review history

### Revision 1 review (Codex Astra; resolved in revision 2)

| # | Finding | Resolution |
|---|---|---|
| 1 B | planar fan apex at the body origin | apex in the face plane (4.1); `1/18` |
| 2 B | pointwise-only remainder; sign; accumulation | complete bound, `min |Q|`, endpoint terms, halving allocation (4.3); `195.3 >= 1.133` |
| 3 B | root identity by overlap | `(poly, ordinal)` / three-count test (3.2) |
| 4 B | loops not `v`-monotone | loop interiors, parity, chains (3.3); D-CH3-11 |
| 5 M | chart vs knot | knot space, range box (2.2) — completed in revision 3 |
| 6 M | hints unchecked | H1/H2 (2.3) — strengthened in revision 3 |
| 7 M | planar faces with curved edges | predicates (2.3, 3.3) — corrected to true degree in revision 3 |
| 8 M | zero polynomials, common factors, completeness | precedence table, coprime basis, completeness (5.1) — zero-beta rule added in revision 3 |
| 9 M | `1/|n|`; `J` numerator | sqrt bounds (4.4) — allocation corrected in revision 3; `P_i` (4.1) |
| 10 M | core not verifiable | claim-only verification (4.4, 6) |
| 11 M | intervals in digest | `(poly, ordinal)` (3.4) |
| 12 M | composite units | bookkeeping (7) — superseded by the restated contract in revision 3 |
| 13 M | Q08 acceptance | matrix and references (8) — corrected in revision 3 |
| 14 M | inadmissible fixtures | replaced — replaced again in revision 3 |
| 15 m | editorial | fixed |

### Revision 2 review (Codex Astra; resolved in revision 3, each with the exact check run)

| # | Severity | Finding | Resolution | Exact check run |
|---|---|---|---|---|
| 1 | B | plane sections are not conics; planar-face predicates at the wrong degree | 2.2, 2.3, 3.3: planar arc edges are `E_k / M_k` of degree `<= 2p` (no conic assumption); segment-arc by substitution (degree `<= 2p`), arc-arc by box separation then a resultant of degree `<= (2p)^2`; endpoint tests; B1's frames, deck and bottom proved simple, contained and correctly intersecting | rank 6 of `[1, y, z, y^2, yz, z^2]` at six exact frame points (`deg E = deg M = 4`); aft face: mirrored arcs separated (`E_y`, `M` Bernstein minima `0.81`, `0.13`), across lines `z = 1/2`, `z = 3` meet the arc only at its corner; deck arc meets `{x = 3/2, z = 3}` only at the corner, across line collinear (shared), other across line disjoint; bow-deck parity: `(0.17, 3/4)`, `(0.43, 3/4)`, `(0.3, 3/4 + 10^-6)` outside, `(0.3, 3/4 - 10^-6)` inside |
| 2 | B | orientation counted twice; cap direction | 3.3, 4.1: the chain carries the model orientation, no `sigma_f` factor; cut-graph direction `-sign(beta_w) sigma_chart`; cap = every cut piece reversed; verifier same convention | diamond on `S = (u, u^2, v)`: `+25/4608` anticlockwise, `-25/4608` clockwise; parabolic prism `V = 53/1536` (curved face `-53/4608` with its clockwise loop), `V(z <= 1/2) = 29/3072` with reversed cut pieces on the cap, `A = 49/768`, all equal to hand integrals |
| 3 | M | `v'` vs knot `v` | 2.2: knot `v` throughout (`S~ = ((k_2 - v) A~ + (v - k_1) B~) / (k_2 - k_1)`, `alpha`, `beta`, `E_k`, `M_k` restated); corners, hints and primitives in knot `v` | B1 with `v`-knots `(2,2,4,4)`: `z = 1/2` gives knot `v = 9/4` inside `v_range = (2, 4)`; knot-space point equals the library's `exact_point` to `8e-17` |
| 4 | M | simplicity misses endpoint contacts and coincident pieces | 3.3: four tests (open crossings, endpoint-interior, non-adjacent endpoint-endpoint, coincident pieces) | review loop: area `3/16`, 0 open-interval crossings, endpoint-interior contact (vertex 3, piece 0) detected |
| 5 | M | H2 accepts a shortened subcurve | 2.3 H2(a): endpoint correspondence to the start/end vertices' corners within `delta_chart` (fixes direction); guarantee narrowed explicitly | B1 deck edge: stored ends within `2.3e-13`/`4.1e-13` in `u`; shortened domain `((3a+b)/4, (a+3b)/4)` ends off by `0.0595` (`0.61 m`) though its midpoint passes (b) |
| 6 | M | sqrt allowance wrong scale | 4.4: numerator target `(a/6) s_lo`; sqrt term `<= a/6` in output units by raising `m` | `|n|^2 = 2^201`, `N = 2^100`, `a = 2^-300`: revision-2 rule `1.585e29 a`; revision-3 rule `0.125 a` (`m = 202`) |
| 7 | M | E4 can be continuous | 5.1 row Z (zero-beta face): forbidden height interval `[h_min, h_max]`, refusal inside, one-sided outside, excluded from the gap; E7 explicit | `S = (u, u^2, v)`, `n = (1, 0, 0)`: `beta = 0`, `alpha = u - d`; `d = 1/2` root inside `[1/8, 7/8]`; `d = 15/16` generic, gap `< 1/16` |
| 8 | M | whole products as spans | 7 rewritten to the restated contract: per-element checkpoints in every coefficient loop; native single operations on `max_bits`-capped values are the only spans; proportional charging for the rest; tests per loop | not a numerical item; contract text aligned (amendment 6d as relayed) |
| 9 | M | fixtures inadmissible | 8.2: F-A parabolic prism, F-B conic slab, F-C diamond prism, all with non-ruling plane trims | for each fixture and trim: `beta` not identically zero; corners strictly inside the range box (F-A `(1/8,1/4), (3/8,1/4), (7/8,3/4), (5/8,3/4)`; F-B `u = 0.1147, 0.3411, 0.6589, 0.8853`; F-C `(1/4,1/2), (1/2,1/4), (3/4,1/2), (1/2,3/4)`); F-A's exact values by the engine's formulas |
| 10 | M | half/full with heel; group B counts; group M optional | 8.2: general relation `full = half(n, d) ⊕ mirror(half(n~, d))` with the centre face contributing `0`; symmetric doubling only for `n_y = 0` (groups U, B, N, T); 8.1: contact counts corrected (4 vertices, 4 edges, 1 face for both hulls); group M mandatory with a rational construction | `n = C'(1/4) x e_z`, `d_t = n . C(1/4)`: `g(1/4) = g'(1/4) = 0`; `d_t + 2^-10`: two roots on the deck edge (`0.24658`, `0.25342`); `d_t - 2^-10`: none |
| 11 | m | narrowed-claim test cannot force `BUDGET_EXHAUSTED` | CH3d: an explicitly insufficient verifier budget on an honest claim; exclusion by a shifted claim | none needed |

### Revision 3 review (Codex Astra; resolved in revision 4, each with the exact check run)

| # | Severity | Finding | Resolution | Exact check run |
|---|---|---|---|---|
| 1 | M | arc-arc resultant degree bound `(2p)^2` wrong; spurious factors | 3.3: raw resultant bound `8 p^2`; spurious factors identified (powers of `M_1`, outside the interval by C2; common vanishing of leading coefficients, possibly inside); no division — every isolated root `u*` validated by the gcd of `F(u*, s)`, `G(u*, s)` over `Q(u*)` with a real root `s*` in the open `s`-interval; the verifier does the same; 10: own CH3b tests | the review's pair: `Res_s(F, G)` degree 7 `= u^2 (u+1)^2 (1 + 2u - 4u^2 - 4u^3)` (quotient 1, remainder 0); one root `u* = 0.58504` in `(1/4, 2)`; gcd over `Q(u*)` of degree 1; `s* = 0.85464` in `(1/4, 2)`; `C_1(u*) = C_2(s*)` to `1e-12` |
| 2 | M | half/full applied the reflection sign twice | 8.2: convention stated (`n~ = R n`, `o~ = R o`, `e1~ = R e1`, `e2~ = R e2`): all waterplane quantities add, only `Jy` flips; symmetric case `e2~ = -e2` recovers the doubling / zero rule | starboard unit square (`e1 = x`, `e2 = y`): `Mv = 1/2`, `Iuv = 1/4`; mirrored term `-1/2`, `-1/4`; sums `0`, `0` = full rectangle; asymmetric triangle with `e1 = (1,1,0)`, `e2 = (-1,2,0)`: direct `(1/2, 1/3, -1/3, -5/24)` = relation |
| 3 | M | interior corners do not ensure the range box | 2.2, 2.3 C2: whole-edge range test (Bernstein sufficient test, else corners and interior extrema at the roots of `N' D - N D'` by single-root tests); `UNSUPPORTED_TRIM / leaves_range_box` | `z <= 12x - 12y - 2` on `S = (u, u^2, v)`: `r = 12u - 12u^2 - 2`, corners `r(1/4) = r(3/4) = 1/4` pass, interior extremum at `u = 1/2` has `r = 1 = v_1`: refused |
| 4 | M | F-C crossing query missed the top | 8.2 F-C: `d = 11/8` (`v = 3/8`), hand value `V = 167/1536` | `d = 1` gives `v = 0` (outside `[1/4, 3/4]`); `d = 11/8` gives `v = 3/8`, cut points `u = 5/8` on `y = x - 1/4` and `u = 3/8` on `y = -x + 3/4` (open intervals `(1/2, 3/4)`, `(1/4, 1/2)`); `V = 167/1536` by exact piecewise integration |
| 5 | M | `beta = 1` is not genericity | 8.2 F-A: blanket claim removed; per-query event outcomes listed; the review's plane is an expected `edge_tangency` refusal in group D | `z <= 1/2` and `y + z <= 3/4`: 0 roots on bottom/deck, 1 simple root on each slanted edge; `n = (-3/2, 1, 1)`, `d = 3/16`: deck residual `(u - 3/4)^2`, E2 at `u = 3/4` in `[5/8, 7/8]` |
| 6 | m | F-B is not the unit circle | 8.2 F-B: the curved boundary is the exact-`w` conic; the solid and the reference defined by it | midpoint `x = y = (1 + 2w) / (2(1 + w)) = 0.70710678118654757`, `x^2 + y^2 - 1 = +2.346e-17` (exact rational) |

**Round 4 (focused, Codex Astra, 2026-10-20): accepted.** Findings 2-6 of round 3 resolved; the degree-bound
correction valid. One MAJOR remains as a CH3b slice obligation (section 9, CH3b row): `Q[u]/(q)` with a merely
square-free `q` need not be a field (the reducible `q` above makes `M1 = u(u+1)` non-invertible although
`M1(u*) != 0`). Per the owner's decision no further design round; the design is accepted for implementation.
