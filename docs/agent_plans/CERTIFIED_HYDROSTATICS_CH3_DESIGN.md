# Certified hydrostatics CH3: certified point queries for the curved B1 hull (design, revision 1, 2026-10-20)

Status: design for owner review; no library source changed. Extends
`CERTIFIED_HYDROSTATICS_DESIGN.md` (revision 5, "the planar design" below) and its acceptance manifest;
it does not replace them. Consumer request: `C:/Github/ANYstability/docs/ANYgeometry_CERTIFIED_HYDROSTATICS_HANDOFF.md`
sections 1, 3 and 5 (Q08: full B1 upright/inclined queries, half/full relation, independent slice
references). Working rules: correctness first and fail closed; a typed refusal beats a plausible number;
probes find limits. Facts below are marked **measured** (read-only experiments on e26e31f with
`PYTHONPATH=src`, Python 3.14.0, numpy 2.3.4) or **assumed**/derived.

**Scope statement.** CH3 delivers certified half-space point queries (`V`, `J`, `A`, `Mu`, `Mv`, `Iuu`,
`Ivv`, `Iuv`, states, one-sided limits, topology, proof, verifier) for closed shells whose faces are
either the planar faces of the planar design or faces of the **ruled rational class** of section 2,
trimmed by planes, which contains both B1 bows (`build_b1_half_hull`, `build_b1_hull`). It serves the
geometry portion of Q08 and nothing more: no event/interval API (CH4), no openings or clearances, no
general NURBS, no cylinders or multi-span surfaces (listed extensions in section 2.4), no change to
planar results (planar preparations stay bit-identical, convention version 1). The consumer's own gates
(adapter mapping of enclosures, installed verification, Capytaine comparison, P0/P1) remain theirs.

## 1. Facts this design rests on

Measured on the source (`src/anygeometry/b1_example.py`, `nurbs.py`, `support_curves.py`):

- **B1 bow surface** (`b1_bow_surface`): `NurbsSurface` of degree `(2, 1)`, knots `(0,0,0,1,1,1)` x
  `(0,0,1,1)` (one clamped span in each direction), `u_range = (0, 0.9)`, `v_range = (0, 1)`,
  controls bottom row `(0,6,0), (5.4,6,0), (5.4,0,0)` at `v = 0` and deck row
  `(0,7.5,4), (6,7.5,4), (6,0,4)` at `v = 1`, weights `((1,1),(w,w),(1,1))` with
  `w = 0.7071067811865476` (the double nearest `cos(pi/4)`). **The two weight rows are equal.**
  `Fraction(w)**2 - 1/2 = +6.8e-17`: the iso-`v` curves are conics (rational quadratics), not the
  exact ellipse quadrants a weight of `1/sqrt 2` would give.
- **`z = 4 v` exactly**: with the exact homogeneous net `(w x, w y, w z, w)` the bottom row has
  `z~ = 0` and the deck row has `z~ - 4 W = 0` as polynomials (`W` the common weight polynomial).
- **Rational plane sections (the structural fact of section 2.2)**: for a random rational plane
  `n . r = d`, `alpha(u) = n . A~(u) - d W(u)` and `beta(u) = n . (B~(u) - A~(u))` are quadratics in
  `u`, and the points `S(u, -alpha(u)/beta(u))` satisfy the plane equation with residual exactly `0`
  in rational arithmetic (and match `NurbsSurface.evaluate` to `1e-15`).
- **Trims and corners**: the half hull has 8 vertices, 12 edges (4 `PlaneNurbsCurve`, 8 `Straight`), 6
  faces; `build_b1_half_hull()` takes 8.4 s (numerical section certification). The section-edge
  `domain` values are floats such as `(0.20745090823802123, 0.4939160021836192)` and `(0.125, 0.75)`;
  the stored corner vertices have exactly the frame/plane coordinates in `x` and `z` (`1.5`, `3.5`,
  `0.5`, `3.0` as doubles) and an irrational-valued `y` rounded to a double; the curve's own float
  evaluation at each domain end equals the stored vertex exactly (gap `0.0`).
- **Algebraic corners are cheap**: the corner `x = 1.5, z = 1/2` on the bow is the root of a quadratic
  with 108-bit coefficients, isolated to width `2^-60` in `0.4 ms` (`analytic_roots.isolate_real_roots`,
  Sturm); the crossing of an inclined-plane section with the frame section is a quartic with 213-bit
  coefficients, isolated in `< 1 ms`.
- **Exact rational Taylor enclosures work**: for `int_{0.2}^{0.4} p(u) / W(u)^3 du` (`p` a degree-6
  product of net polynomials) the polynomial-only enclosure of section 4.3 gives widths `8.9e-8`,
  `6.9e-12`, `8.0e-16`, `3.1e-24`, `1.6e-32` at Taylor orders 8, 12, 16, 24, 32, with largest
  numerator/denominator bit lengths 701, 939, 1176, 1657, 2130 and times 3-52 ms; one piece over the
  whole span `[0, 0.9]` at order 16 is useless (`3e-2`) while 8 pieces give `1e-19` in 0.1 s. A 40-point
  Gauss-Legendre reference agrees to `2e-15`.
- **H1 (`hydrostatics`) handles the full B1 hull** in 1-2.4 s per plane with its own settling estimate
  `~1e-16`: upright `z <= 1.5`: `V = 22.569943729716`, `A = 23.319938010999`; `z <= 2.9`:
  `V = 56.691072446746`; `z <= 3.0` (deck): `V = 59.241180737020`; twice the half hull equals the full
  hull to the 12 printed digits. Heel `0.3` through `(2.5, 0, 1.5)`: `V = 25.579680188803` for trims
  `0, +-0.05, 0.1` alike; this is geometry, not a defect: that plane leaves the hull through the bottom
  at `y ~ 3.2` and the deck at `y ~ -4.85`, inside the bows' `y` extent (`|y| >= 4.76` at the bottom,
  `>= 5.71` at the deck), so the trim wedges lie in the prismatic planar part and cancel (hand check,
  **assumed** to the extent it was not recomputed exactly). It is also a useful CH3 fixture: a section
  through planar faces only (no cut arc on a curved face; its quantities are still enclosures because
  the cut segments end on straight edges with algebraic corners), over fully immersed curved faces.
- **Today's certified preparation refuses the B1 hull** with `UNSUPPORTED_SURFACE` (`face 1 has no
  Plane support (NurbsSurface)`), as the planar design states.
- The repository has exact Sturm root isolation (`analytic_roots.py`, `exact_levels.py`) and an exact
  interval class with `sqrt`, `atan`, `sincos`, `pi` but no `log` (`exact_predicates.py`), none of it
  metered by `WorkBudget`; `mpmath` is **not installed** in the Python used for the experiments.

Derived (assumed until CH3b measures it): with partial fractions every 1-D integrand of section 4
reduces to terms `(a u + b) / Q(u)^k` with `Q` one of the fixed quadratics `W` or `D_k`, so Taylor
bit growth stays in the measured regime (denominator degree 2) rather than the degree-40 denominators a
naive expansion of the second moments would produce.

## 2. Certified domain (decision 1)

### 2.1 Options

| Option | Content | Refuses | Cost and risk |
|---|---|---|---|
| A. **Ruled rational class** (recommended) | planar faces of the planar design, plus faces whose support is a rational tensor-product surface of degree `(p, 1)`, `p in {1, 2}`, one clamped knot span in each direction, two control rows with **equal weight rows** and positive weights, trimmed only by plane sections; vertices are exact triple intersections | general NURBS, interior knots, `p >= 3`, unequal weight rows, iso-parametric or free-form trims, curved edges between two curved faces | plane sections are rational graphs `v(u)` (section 2.2): all integrals reduce to 1-D rational integrals; algebraic numbers are univariate roots of degree `<= 2p` polynomials; everything exact-rational or polynomial-bounded |
| B. Rational NURBS faces of any degree trimmed by planes | as A with any `(p, q)`, knots | free-form trims | plane sections are algebraic curves `f(u, v) = 0` of bidegree `(p, q)`, not graphs: needs certified curve tracing (the P1 machinery, numerical), 2-D Taylor models, bivariate root isolation; integrals no longer 1-D; an order of magnitude more work with weaker guarantees |
| C. General trims (`PlaneNurbsCurve` chains with arbitrary domains, `CurveOnSurface`) | any model face | nothing | the exact model is defined by numerical root records; no exact predicate base; not certifiable in this programme |

**Recommendation: A**, the narrowest class that contains both B1 bows (every property is measured in
section 1: degree `(2,1)`, one span, equal weight rows, four plane trims). Refusals are typed
(`UNSUPPORTED_SURFACE` with sub-kind, `UNSUPPORTED_TRIM`). Option B is CH5 material if a consumer
model needs it; nothing in Q08 does.

### 2.2 The structural fact (verified for B1)

Write the exact homogeneous net of an admitted support as two rows of `p + 1` points
`(x~, y~, z~, w)`; with knot parameter `u` on the span `[u_0, u_1]` (the exact affine image of the
chart `[0, 1]` under `u_range`) and `v in [0, 1]`:

```
S~(u, v) = (1 - v) A~(u) + v B~(u),   A~, B~ in Q[u]^4 of degree <= p (Bernstein over the span),
W(u) = A~_w(u) = B~_w(u)              (equal weight rows; W > 0 on the span: positive Bernstein coefficients),
S(u, v) = P(u, v) / W(u),             P = (1 - v) A~_xyz + v B~_xyz, bidegree (p, 1).
```

A plane `n . r = d` (`n`, `d` exact rationals) meets the support where

```
f(u, v) = n . P(u, v) - d W(u) = alpha(u) + v beta(u) = 0,
alpha = n . A~_xyz - d W,  beta = n . (B~_xyz - A~_xyz)   (both in Q[u], degree <= p),
```

so away from the roots of `beta` the section is the **rational graph** `v = r(u) = -alpha(u) / beta(u)`.
Measured for B1 (section 1): degrees 2 and 2, residual exactly 0. Consequences used throughout:

- a trim edge is `{S(u, r_k(u)) : u in [u_a, u_b]}` with `r_k = N_k / D_k`, `D_k != 0` on `[u_a, u_b]`;
- the world point of the edge is `C_k(u) = E_k(u) / M_k(u)` with `E_k = A~_xyz D_k + (B~_xyz - A~_xyz) N_k`
  (degree `<= 2p`) and `M_k = W D_k`; `E_k / M_k` is reduced by exact polynomial gcd (for `z`-planes of
  B1 this makes `C_z` the constant `h` exactly: `z = 4 v`, `r = h / 4`);
- two graphs cross where `N_i D_j - N_j D_i = 0`, a polynomial of degree `<= 2p` (**4 for B1**);
- for the waterplane, `alpha` depends on `d` affinely and `beta` not at all: `d` enters every event
  polynomial of section 5 polynomially, which is what the event-gap argument needs.

If `beta` vanishes at `u*` with `alpha(u*) != 0` the section leaves the chart before `u*` (no point of
the section has `u = u*`); if both vanish the plane contains the ruling `u = u*` (section 5, refused).

### 2.3 Admission checks for the class (CH3a)

Applied per face of the selected set after the type checks; each decided exactly:

- **C1 class**: `NurbsSurface` with `degree_v == 1`, two control rows, `degree_u in {1, 2}`, clamped
  knots without interior knots in both directions, finite coordinates, `weights` either `None` or with
  `weights[i][0] == weights[i][1]` exactly and all positive; `u_range`/`v_range` finite with
  `u_range` inside the knot domain. Else `UNSUPPORTED_SURFACE` (sub-kind recorded).
- **C2 trims**: every edge of a curved face is a `PlaneNurbsCurve` whose `surface` is the face's
  support (`support_key` equal and `u_range`, `v_range` equal) and whose plane record is a finite
  `Plane`; its exact graph `r_k = N_k / D_k` is formed from the plane's exact normal `m = u_vec x v_vec`
  and offset `m . origin` (the stored doubles, exact); the edge's `u`-interval is derived in C3;
  `D_k` must be nonzero on that interval (one sign of its Bernstein coefficients over the interval,
  else refine the interval's algebraic ends and retry to the policy depth, then `UNSUPPORTED_TRIM`).
  `Straight` edges of curved faces are refused `UNSUPPORTED_TRIM` (rulings and iso-curves are out of
  scope); straight edges of planar faces may end at algebraic corners.
- **C3 corners (hints)**: every vertex of a curved face is the intersection of the two trim planes of its
  incident section edges with the support: `r_i(u) = r_j(u)`, the polynomial `N_i D_j - N_j D_i`
  square-free-reduced and its real roots isolated inside the span. The stored vertex position and the
  edge's stored `domain` ends are **hints**: the exact corner is the root whose world enclosure
  (section 3.2) is within `delta_hint` of the stored vertex position; the choice must be unique and the
  corner must lie in the chart (`0 <= r_i(u*) <= 1`, decided exactly). No candidate, two candidates, or
  a corner off the chart refuses `TRIM_HINT_MISMATCH`. The edge's `u`-interval is the interval between
  its two corners' roots (orientation from the loop, section 3.3); a trim graph must have no pole and no
  chart exit (`r_k` leaves `[0, 1]`) strictly inside that interval (exact: roots of `D_k`, `N_k` and
  `N_k - D_k` in the open interval).
- **C4 chart injectivity** (the curved face is an embedded patch): a rational direction `m` with
  `m . A~_xyz = c_0 W` and `m . (B~_xyz - A~_xyz) = c_1 W` as polynomials, `c_1 != 0` (one body
  coordinate is an affine function of `v` alone; B1: `m = e_z`, `c_0 = 0`, `c_1 = 4`), and the control
  triple of every `v`-row non-collinear: some component of `(P_1(v) - P_0(v)) x (P_2(v) - P_0(v))`
  (quadratic in `v`, `P_i(v) = (1 - v) P_i^A + v P_i^B`) has one sign on `[0, 1]` by its Bernstein
  coefficients (for `p = 1` the row is a segment of nonzero length). Then `(u, v) -> S` is injective
  on the chart (the `m`-coordinate fixes `v`; a nondegenerate rational quadratic with positive weights
  is an injective conic arc). Else `UNSUPPORTED_SURFACE` (sub-kind `not_graph_like`).
- **Checks 1-6 of the planar design, extended**: (1) planar faces as before; curved faces by C1-C4;
  (2) loops: the chart loop of a curved face is a simple closed curve made of graph pieces, checked
  as in section 3.3 (consecutive pieces meet only at their shared corner, non-consecutive pieces do not
  cross: crossing polynomials have no root in the open intervals); (3) holes on curved faces are refused
  `INVALID_HOLES` in CH3 (none in B1); (4) shell combinatorics exactly as in the planar design (reused);
  (5) embedding: planar-planar pairs as before but with algebraic vertices evaluated through the
  predicate layer of section 3.2; curved-planar pairs: the planar face's plane section of the support,
  restricted to the curved face's chart region, must be exactly the shared edge(s) (the graph pieces
  inside the region, computed as in section 4.1, equal the declared edge intervals; a non-adjacent
  pair must have an empty restriction) and the planar face's loop must meet the support only on the
  shared cells (its straight edges: the edge line's section of the support is a univariate polynomial
  root problem; its curved edges: shared by definition or disjoint by the previous test);
  curved-curved pairs: exact bounding boxes of the two sub-patches over their chart regions
  (`knot_patch_hull` style exact projected controls) must be disjoint, else `EMBEDDING_UNDECIDED`
  (fail closed; B1's two bows are separated by `|y| >= 4.75`); (6) orientation: the exact signed
  volume is an enclosure in CH3; it must be certified positive (`lo > 0`), refining until decided or
  `UNDECIDED_PREDICATE`.

### 2.4 Listed extensions (not in CH3)

Interior knots (per-span pieces, the same formulas), `p = 3`, cylinders via `cylinder_to_nurbs`
(multi-span), rulings as straight edges, holes on curved faces, unequal weight rows (the inner
`v`-integral then has a linear denominator and produces logarithms: closed-form or 2-D enclosures).

## 3. The exact model (decision 2)

### 3.1 Rule

The planar rule "stored doubles are exact data" applies to **supports and plane records**: the surface
net, weights, knots, ranges and every trim plane's `origin`, `u_vector`, `v_vector` are exact rationals
of the stored doubles. **Derived stored data are hints**: section-curve `domain` ends, `path` indices
and corner vertex positions were produced by numerical root finding (`section_roots`, `_parameter`)
and are used only to select among exact candidates (C3). The exact solid is therefore defined by the
supports and the combinatorics: edges are `plane_k ∩ support` restricted between exact corners, corners
are exact triple intersections, and the loops are the model's loops. Quantities are certified for this
exact solid. `delta_hint` is a preparation-policy value recorded in the proof (default
`2^-40 (1 + max |stored coordinate|)`; measured hint errors are `<= 2^-49` of the magnitude by the
library's own `2^-50` certification of section points plus one rounding, so the default has a `2^9`
margin and is far below any geometric feature of B1).

### 3.2 Algebraic numbers and points (the predicate layer)

```
AlgebraicNumber: (poly: square-free primitive integer polynomial, lo, hi: Fractions with poly(lo) poly(hi) < 0,
                  or lo == hi for a rational root; refine(): one bisection step, exact sign)
AlgebraicPoint:  (root: AlgebraicNumber, coords: three reduced rational functions E_i / M in Q[u])
Enclosure:       (lo, hi: Fractions)  with the invariant exact value in [lo, hi]
```

Predicates on one root are **exact**: the sign of `g(u*)` for `g in Q[u]` is the sign of `g` on the
isolating interval once its Bernstein coefficients over the interval have one sign; `g(u*) = 0` exactly
iff `gcd(g, poly)` has a root in the interval (Sturm count 1), decided before any refinement. Identical
roots are recognised (same `poly`, overlapping intervals), so mirrored corners of the full hull share
`u*` and the across-line directions `(0, -2 y*, 0)` are exact in `x` and `z`. Expressions in **several**
roots (a plane crossing a straight edge between corners of different roots) are evaluated as
`Enclosure`s with rational interval arithmetic over the isolating intervals and refined on demand to
the policy depth `max_isolation_bits`; a sign still undecided refuses `UNDECIDED_PREDICATE` (fail
closed; the proof records the depth reached). Every refinement step is a metered unit.

### 3.3 Chart loop of a curved face

The chart region of a face in class A is **`v`-monotone**: each trim plane is a half-space in `v` for
fixed `u` (`alpha_k + v beta_k <= 0` or `>= 0`, with `beta_k` of one sign on the edge's interval by C2),
so for every `u` in the face's `u`-extent the face is `{v : lo(u) <= v <= hi(u)}` with `lo` the upper
envelope of the lower graphs and `hi` the lower envelope of the upper graphs. The loop is the lower
envelope traversed in `+u` followed by the upper envelope in `-u` (anticlockwise in the chart) or the
reverse; which one the model's oriented loop is follows from the corner order of each edge (its start
corner's root compared with its end corner's root), and gives the sign `sigma_f = +1` if `S_u x S_v`
is the outward normal and `-1` otherwise (the loop runs anticlockwise about the outward normal). The
breakpoints of the envelopes are the corners (C3). The same decomposition, with the waterplane graph
added, is the clipped region of section 4.1.

### 3.4 Snapshot document (convention version 2)

The planar snapshot document gains, per curved face, `"support": {"kind": "ruled", "degree_u", "net":
[[[wx, wy, wz, w] pairs]], "span": [u_0, u_1], "sigma"}`, per section edge `{"kind": "section",
"plane": [origin, u, v], "domain_hint": [hex, hex], "graph": [N, D]}` and per vertex on a curved face
`{"corner": {"poly": [...], "interval": [lo, hi], "hint": [hex x3]}}`; the digest covers hints because
they select the exact candidates. Planar-only preparations produce exactly the convention-1 document
(ordinary behaviour unchanged, checked by CH3a's test that every CH1 fixture's digest is unchanged).

## 4. Quantities and bounds (decision 3)

### 4.1 Decomposition

For the `below` convention, `K = S ∩ {n . r <= d}` is bounded by (i) clipped planar faces, (ii) clipped
curved faces and (iii) the cap. All three reduce to **signed boundary integrals along graph pieces**:

- **Planar face**: the fan of the planar design over its boundary chain; a curved edge `C_k(u)` of a
  planar face contributes `V += sigma int det(q, C_k, C_k') / 6 du` and
  `J += sigma int det(q, C_k, C_k') (q + 2 C_k) / 24 du` (the triangle `(q, C, C + C' du)` with apex
  `q` the body origin); straight edges with algebraic ends contribute the planar formulas evaluated on
  `AlgebraicPoint`s. Clipping a curved edge against the half-space: its height `h_k(u) = n . C_k(u) - d`
  has roots (univariate, degree `<= 4p`) at which the edge enters or leaves `H`.
- **Curved face**: `V_f = sigma_f int_R P . (P_u x P_v) / (3 W^3) du dv` over the clipped chart region
  `R` (`S . (S_u x S_v) = P . (P_u x P_v) / W^3` because `P . (P x P_v) = 0`), and
  `J_f = sigma_f int_R x_i P . (P_u x P_v) / (4 W^4) du dv`. The integrands are polynomials in `v`
  (degree 2 and 3) over `W^3`, `W^4`, so by Green's theorem each is `- oint_{dR} F(u, v) du` with
  `F = int_0^v (integrand) dv'` polynomial in `v`; on a graph piece `v = N / D`, `F(u, N / D)` is a
  rational function with denominator `W^3 D^3` (`V`) or `W^4 D^4` (`J`). `R` is the `v`-monotone
  region of section 3.3 intersected with `alpha_w + v beta_w <= 0` (the waterplane graph
  `r_w = -alpha_w / beta_w`, a lower or upper graph according to the sign of `beta_w` on the piece;
  pieces are split at the real roots of `beta_w`, outside which the half-space contains the whole
  ruling or none of it, decided by the sign of `alpha_w`).
- **Cap**: the oriented cut chain is the union of straight cut segments on planar faces (planar design
  section 5, on `AlgebraicPoint`s) and **cut arcs** `C_w(u) = E_w / M_w` on curved faces, each a graph
  piece between two cut points on trim edges (roots of `N_w D_k - N_k D_w`). The fan from a plane point
  `q` over the chain: `dA = ((C_w - q) x C_w') . n / (2 |n|) du`, and with `(u_p, v_p)` the waterplane
  coordinates of `C_w` (exact: `e1 . (C_w - o)`, `e2 . (C_w - o)`) the differential triangle
  `(q, C, C + C' du)` gives `Mu = int (q_u + 2 u_p) / 3 dA`, `Iuu = int (q_u^2 + 2 q_u u_p + 3 u_p^2) / 6 dA`,
  `Iuv = int (q_u q_v + q_u v_p + u_p q_v + 3 u_p v_p) / 6 dA` (and `Mv`, `Ivv` likewise), the limits of
  the planar triangle formulas. Denominators are powers of `M_w = W D_w` up to the fifth.

Every contribution is `sum over pieces of int_{u_a}^{u_b} Num(u) / Den(u) du` with `Num, Den in Q[u]`,
`Den` a product of powers of `W` and the `D_k` (all of one sign on the piece, certified by Bernstein
coefficients over the piece's outer interval), and `u_a`, `u_b` rational or `AlgebraicNumber`s. **No
2-D quadrature and no transcendental function is evaluated anywhere.**

### 4.2 Breakpoints and pieces

The `u`-breakpoints of a face are: its corners; the cut points (roots of `N_w D_k - N_k D_w` on each
edge interval); the real roots of `beta_w` inside the face's `u`-extent (`alpha_w` of one sign there, else
section 5); chart exits of `r_w` (`alpha_w = 0`, `alpha_w + beta_w = 0`) are not breakpoints of `R`
because they lie outside the trims (the trims bound `v` away from the chart edge only where the face
reaches it; B1 faces are interior). Sorting algebraic breakpoints uses the predicate layer (distinct
roots of the same polynomial are ordered by their disjoint intervals; roots of different polynomials
by refinement, with equality detected by gcd). Between consecutive breakpoints the lower and upper
graphs of `R` are fixed; each (piece, graph) pair is one 1-D integral. Pieces count is
`O(corners + cut points + 2)` per face: for B1 at most about a dozen per bow.

### 4.3 Enclosing a 1-D rational integral (the only approximate step)

Options:

| Option | Bound | Assessment |
|---|---|---|
| (i) closed forms: rational part + `lambda log Q(u)` + `(mu / s) atan((u - u_0) / s)` with `s^2 in Q`, evaluated by rigorous interval `log`/`atan` at algebraic endpoints | tight at any precision; the symbolic antiderivative is an "exact representation" | needs a certified `log` (absent), `atan` with `Q(s)` coefficients, square roots of rationals, Hermite reduction; a second transcendental library to review; the verifier would share most of it |
| (ii) **exact rational Taylor models with Bernstein remainders** (recommended) | per term `(a u + b) / Q^k`: `g = Q^{-k}` has `g^{(j)} = P_j / Q^{k + j}` with `P_{j+1} = P_j' Q - (k + j) P_j Q'` exact; expand at the piece midpoint `c` to order `K`, integrate the product with the numerator exactly, bound the Lagrange remainder by `max |P_{K+1}| / (min Q)^{K + k + 1} / (K + 1)! h^{K+1}` with both extrema from Bernstein coefficients over the piece | measured (section 1): `1e-12` at order 12, `1e-16` at order 16 on a `0.2` piece, `~1e3` bits; only `+ - * /` on `Fraction`s, metered per coefficient; partial fractions keep `deg Q = 2` |
| (iii) rigorous Gauss or Simpson with derivative bounds | similar | the derivative bounds are the same Bernstein computations as (ii) with worse constants; no advantage |
| (iv) exact rational reduction | exact | impossible in general: the values are transcendental (`atan` of algebraic numbers) |

Rules for (ii): (a) **partial fractions** over `Q`: `Num / Den` is split into a polynomial part (integrated
exactly) and terms `(a u + b) / Q^j` for each factor `Q in {W, D_k}` and power `j` (extended Euclid; the
factors must be coprime, else the piece refuses `UNSUPPORTED_TRIM`, sub-kind `common_factor`, which also
catches a trim plane equal to the waterplane); (b) endpoints: for an algebraic end `u_a in [a_lo, a_hi]`
the integral over `[a_lo, a_hi]` is enclosed by `[-1, 1] (a_hi - a_lo) max |Num| / min |Den|` over that
interval (Bernstein), the inner rational interval by the Taylor model; (c) **adaptivity**: each piece
starts at order `K_0 = 8`, is bisected while its remainder exceeds its share of the target (share =
target width / number of terms, tracked exactly), and the corner intervals are refined (3.2) while the
endpoint contribution exceeds its share; orders and subdivision depth are bounded by the policy
(`max_taylor_order = 32`, `max_pieces_per_integral = 1024`, `max_isolation_bits = 4096`); (d) the result
of one query's quantity is `exact_part + sum of Enclosures` = an `Enclosure`, exact when there is no
enclosed term (the engine tracks this; a `FULL`/`EMPTY` occupancy over a shell with curved faces still
has a transcendental `V`, except `EMPTY`'s exact `0`).

### 4.4 Forms, bounds and the claim protocol (decision 4)

A quantity in a CH3 preparation is one of:

- **exact** (rational, or `Q / |n|`): only when no algebraic or enclosed term entered it; same forms and
  rules as the planar design;
- **enclosed**: the proof carries `"enclosure": {"lo": PAIR, "hi": PAIR, "core": {"lo": PAIR, "hi": PAIR},
  "target": PAIR, "pieces": N, "max_order": K, "isolation_bits": B}` where `core` is the computed
  enclosure and the **claim** is `[core.lo - w, core.hi + w]` with `w = core.hi - core.lo`
  (three core widths; the claim's width is at most the requested accuracy because the core targets a
  third of it). The widening is what lets an independent verifier establish the claim (section 6).
  `binary64` is the outward rounding of the claim to doubles with its width `hi - lo` as the bound
  (not adjacent doubles: adjacency is unprovable for a transcendental value in bounded work; a request
  finer than the binary64 width refuses `ACCURACY_UNACHIEVABLE` as before); `decimal` is a decimal
  enclosure of the claim of the requested width; the `exact` key is absent (a verifier that finds it
  present for an enclosed quantity refuses `MALFORMED_PROOF`).

Accuracy requests: `absolute_accuracy[name] = a` sets the core target `a / 3`; without a request the
policy default `default_enclosure_width = 10^-12` (model units to the quantity's power) applies to the
core. A target not reached within the policy's order/piece/isolation limits refuses
`ACCURACY_UNACHIEVABLE` with the achieved width in the refusal; the budget, deadline, cancellation and
bit cap refuse as before. `accuracy_met` is `True` for a returned enclosed quantity (its claim width is
at most the request) and the verifier recomputes the flag.

Bound semantics stay those of the planar design: a binary64 or decimal enclosure contains the exact
value; the exact value is that of the exact solid of section 3.1 for the exact query.

## 5. Section topology, degeneracy and one-sided limits (decision 4 of the brief)

### 5.1 Events of a curved face for fixed `n`

Every change of the section's combinatorics on a curved face as `d` varies happens at one of these
**event heights**, each an algebraic number:

| Event | Where | Polynomial |
|---|---|---|
| E1 corner on the plane | a corner `u*` (root of its corner polynomial) | sign of `n . E_k(u*) - d M_k(u*)`: exact single-root test |
| E2 edge tangency | critical points of `h_k(u) = n . C_k(u)` on the edge interval | roots of `(n . E_k)' M_k - (n . E_k) M_k'` in the open interval; `d` equals the height there iff `n . E_k - d M_k` vanishes at the root (single-root test) |
| E3 interior tangency | `beta_w(u*) = 0` and `n . S_u(u*, v*) = 0` with `(u*, v*)` inside `R`'s face region | `beta_w` roots; `v*` rational in `u*`; height `n . S(u*, v*)` compared with `d` (single-root test) |
| E4 ruling in the plane | `alpha_w(u*) = beta_w(u*) = 0` | `gcd(alpha_w, beta_w)` with a root in the face's `u`-extent |
| E5 edge in the plane | `n . E_k - d M_k` the zero polynomial | exact |
| E6 planar events | vertex heights of planar-only vertices (rational) and E1 | planar design |

E1, E5 and E6 are **supported degeneracies**; E2, E3 and E4 refuse `CURVED_SECTION_DEGENERACY` with
sub-reason `edge_tangency`, `interior_tangency`, `ruling_in_plane`. Rationale: at E2-E4 the one-sided
limits exist mathematically but the cut arcs are not graphs with simple endpoints on both sides; CH3
does not need them for Q08 (they are measure-zero coincidences for the consumer's rational planes),
and a typed refusal at the exact plane is the honest answer. A plane is **generic** when no event of
E1-E6 is at `d`; then the section is computed directly (one section, as in the planar design).

### 5.2 Replacing symbolic perturbation: the certified event gap

The planar route perturbs `d` symbolically because every predicate is a polynomial in `epsilon`. On a
curved face the perturbed cut points are algebraic functions of `epsilon` (Puiseux series at
tangencies), so the first-nonzero-coefficient rule does not transfer. CH3 uses instead:

1. **Gap certification.** For a degenerate plane, isolate every event height of E1-E6 (and, for the
   planar faces, the vertex heights) in `d`-space as algebraic numbers; refine until an open interval
   `(d - eps_0, d)` and `(d, d + eps_0)` with rational `eps_0 > 0` contains no event height (events equal
   to `d` are the degeneracies themselves; events whose intervals cannot be separated from `d` after
   `max_isolation_bits` refuse `UNDECIDED_PREDICATE`). Within each gap the section combinatorics is
   constant (no event, no change).
2. **Limit combinatorics.** Compute the section at the generic rational planes `d - eps_0 / 2` and
   `d + eps_0 / 2` (cut points identified by `(edge id, crossing ordinal along the edge)`, cut arcs by
   the pair of cut points they join, segments, loops, nesting). These are the `below` and `above`
   combinatorics.
3. **Limit positions and quantities.** Each limit cut point converges, as `epsilon -> 0`, to a root of
   the edge height at `d` (transversal crossing: the root itself) or to a corner (E1) or an edge end
   (E5); which one is determined by continuity inside the gap (the ordinal and the root order are
   constant). The limit section's quantities are then the section 4 integrals **at `d`** over the limit
   chain (cut arcs between limit cut points; an edge in the plane counted on the side where the
   perturbed plane crosses it, exactly the planar convention). `V` and `J` are continuous in `d`: one
   value, at `d`, with the `below` convention.
4. The generic planar-vertex machinery (symbolic perturbation) remains in use for planar faces whose
   vertices are rational; faces with algebraic vertices go through steps 1-3 as well, so one mechanism
   covers the whole shell of a curved preparation.

Topology identities: a loop is the sorted tuple of `(edge id, ordinal)` cut-point identities
(convention 2 wording "sorted (edge id, crossing ordinal) pairs of the crossed shell edges"); the cut
arc on a curved face and the straight segments on planar faces are both edges of the degree-2
cut-point graph, so loops and nesting (signed area and ray parity, now on `Enclosure`s with exact
fallback through the predicate layer) assemble as in the planar design. Contacts list vertices, edges
and faces in the plane, now including curved edges in the plane (E5) and corner contacts (E1).

## 6. The independent verifier (decision 5)

`verify_certified_hydrostatics` keeps steps (0)-(2) and (5) of the planar design (query binding, proof
re-encoding, snapshot and artifact identity, completeness) and extends (3)-(4):

- **Class and model**: re-derives the exact net, graphs, corners and chart loops from the model with
  its own code (Descartes/Bernstein subdivision for root isolation where the provider uses Sturm;
  the two are different algorithms with the same exact output semantics), re-runs C1-C4 and 1-6, and
  requires the same corner polynomials, the same ordered breakpoints (algebraic equality by gcd and
  disjointness) and the same `sigma_f`.
- **Combinatorics and exact parts**: recomputes contacts, genericity, occupancies, event gaps (its own
  `eps_0`, which need not equal the provider's: only the resulting combinatorics must), cut-point
  identities, loops and nesting, and every exact quantity; requires exact equality.
- **Enclosed quantities**: recomputes each quantity's enclosure with a **different scheme**: one-sided
  Taylor expansion at the piece's left end (`int_0^h g = sum g^{(j)}(u_a) h^{j+1} / (j + 1)! + R`) with
  its own fixed order schedule and bisection, refining until its enclosure `E_v` has width at most
  the proof's `core` width (or the budget stops it). It then requires `E_v ⊆ claim` — which holds
  whenever the claim is honest, since `E_v` contains the exact value and
  `width(E_v) <= width(core)` puts it inside `core ± width(core)` — and that `claim = core ± width(core)`
  exactly, `binary64`/`decimal` forms are correct outward roundings of the claim, and `accuracy_met`
  recomputes. `E_v` not inside the claim refuses `VERIFICATION_FAILED`, sub-reason `enclosure`; a
  verifier that cannot reach the core width within its budget refuses `BUDGET_EXHAUSTED` in phase
  `verify` (undecided, never accepted).
- **Trust model** (extends the planar one): the claim holds for the exact solid of section 3.1 given
  (a) the integral identities of section 4 (divergence theorem, Green's theorem on graph pieces, the
  envelope decomposition of `v`-monotone regions) and the event-gap argument of section 5 (constant
  combinatorics between consecutive events); (b) the correctness of the verifier's exact polynomial
  arithmetic, root isolation and Taylor/Bernstein bounds (the mathematical bound is stated in the
  verifier's documentation and is the one reviewed), and of Python's integers and `Fraction`s; (c)
  outward rounding of enclosures. Agreement of two Taylor-model implementations corroborates; the
  reviewed bound in the verifier justifies. The verifier imports nothing from the provider modules.

## 7. Work, checkpoints and bulk operations (decision 6)

New primitive units (each reserved before it runs; each a checkpoint): `poly_coefficient` (one
coefficient multiply-add in a polynomial product, pseudo-remainder, shift or Bernstein conversion),
`sturm_step` / `descartes_step` (one sign-variation count or one subdivision), `root_refine` (one
bisection of an isolating interval, including its exact sign), `interval_op` (one rational interval
operation), `taylor_coefficient` (one derivative polynomial `P_j` step), `bernstein_bound` (one
coefficient of a Bernstein bound), `piece_split`, `event_height`. Loops are bounded by named policy
limits: `max_isolation_bits`, `max_taylor_order`, `max_pieces_per_integral`, `max_refinement_rounds`
(outer refine-and-retry rounds of 4.3(c) and 5.2(1)); exceeding one refuses `ACCURACY_UNACHIEVABLE`
(precision limits) or `UNDECIDED_PREDICATE` (sign limits), the budget refuses `BUDGET_EXHAUSTED`.

Amendment 6b table, added rows (each charged before it starts, bounded by a named size, followed by a
checkpoint):

| Bulk operation | Bound |
|---|---|
| one polynomial product or pseudo-remainder step | `(deg a + 1)(deg b + 1)` coefficient units, degrees bounded by the class (`<= 2p` inputs) and the Taylor order |
| one Bernstein conversion of a polynomial over an interval | `(deg + 1)^2` units |

Nothing else is exempt; bit lengths are checked on every stored value as before (the measured 1-2 k
bits at orders 16-32 fit the 4096-bit default cap; CH3b measures the real denominators and raises the
default only on evidence, D-CH3-7).

## 8. Q08: what CH3 evidences and what remains the consumer's (decision 7)

- **Full B1 upright and inclined queries**: the CH3d run set (section 9) answers every plane with a
  claim or a typed refusal; each result verifies with the CH3c verifier; work and timings recorded.
- **Half/full relation**: for symmetric planes (`n_y = 0`, `o_y = 0`, `e1`, `e2` symmetric) the half
  hull's `V`, `Jx`, `Jz`, `A`, `Mu`, `Iuu`, `Ivv` claims doubled must intersect the full hull's claims
  and `Jy`, `Mv`, `Iuv` of the full hull must enclose `0` (criterion: intersection non-empty and both
  within the requested width; measured bitwise equality of the cores is reported as an observation,
  not required). The relation is evidence of consistency, not a proof; the proof is the verifier.
- **Independent slice references** (not certified, must be enclosed by the claims): (i) H1
  `hydrostatics()` values with its settling estimate (measured in section 1; a different method:
  float quadrature on the model); (ii) per-station closed forms: at upright `z = h` the B1 section is
  bounded by the frames and the conic arc `S(u, h/4)` whose area, first and second moments have closed
  forms (rational quadratic with weight `w`; the exact-`w` conic, not the ideal ellipse) evaluated in
  high precision by an independent tool (`mpmath` at 40 digits, a test-only dependency, D-CH3-8);
  (iii) the **sandwich check** replacing the planar derivative check: for `h_1 < h_2` with no event
  between, `V(h_2) - V(h_1)` lies in `[(h_2 - h_1) min A, (h_2 - h_1) max A]` over `[h_1, h_2]`; for B1
  upright planes `A(z)` is increasing (flared bow), so the claim intervals must satisfy
  `claim(V(h_2)) - claim(V(h_1))` intersects `(h_2 - h_1) [A_lo(h_1), A_hi(h_2)]`; (iv) the planar
  cross-check: a box with rational corners whose top is a degree-`(1,1)` ruled face (weights 1, so
  `W = 1`, corner roots rational): for planes that do not cut the ruled face every term is a polynomial
  integral with rational ends and the engine must return **exact** forms equal to CH1's for the same
  box with a `Plane` top; for planes cutting it (cut arc `C_w = E / beta`, a rational parametrisation of
  a straight line) the claims must contain CH1's exact values; (v) a quarter-cylinder prism of degree
  `(2,1)` with weights `(1, w, 1)`: the conic sector reference by (ii).
- **Consumer's job**: mapping claims into its `IntegralSample` intervals, PSD checks, Capytaine panel
  comparison, P0/P1 measurements, equilibrium/GZ, admission.

## 9. Slices

| Slice | Output | Acceptance criterion | Author | Depends on |
|---|---|---|---|---|
| CH3a | class recognition C1-C4, exact net and graphs, hint checks, chart loops, extended checks 1-6, snapshot document v2, new refusal categories (`UNSUPPORTED_TRIM`, `TRIM_HINT_MISMATCH`, `CURVED_SECTION_DEGENERACY`, `UNDECIDED_PREDICATE`, `EMBEDDING_UNDECIDED`), policy fields | `pytest tests/test_certified_hydrostatics_ch3_admission.py`: both B1 builds admit (half: 6 faces, 2 curved-planar shared-edge checks per bow edge; full: the bow-bow box test); every new refusal from a minimal case (unequal weight rows, interior knot, a `Straight` edge on the bow, a vertex hint moved by `1e-6`, a non-graph-like patch, two bows translated to overlap); every CH1 fixture's snapshot digest and proof bytes unchanged (`git diff` of the recorded proofs empty) | Sonnet (Opus review) | CH3b predicate layer (may be stubbed by exact Sturm for the first tests) |
| CH3b | the exact kernel: polynomials over `Q` (gcd, square-free, Sturm, Descartes, Bernstein conversion and bounds, partial fractions), `AlgebraicNumber`/`AlgebraicPoint`/`Enclosure`, the 1-D Taylor-model integrator of 4.3 with metering, measured bit lengths | `pytest tests/test_certified_hydrostatics_ch3_kernel.py`: enclosures contain `mpmath` references (40 digits) for `int (a u + b) / Q^k` over 50 random rational pieces at target widths `1e-6..1e-20` (every width met, bit length recorded, max recorded `< 4096` or the policy proposal stated); exact zero tests on constructed common roots; metered unit counts equal sums; refusals at each limit | Opus | - |
| CH3c | the curved query engine: breakpoints and pieces, clipped regions, cut arcs, planar faces with algebraic vertices, events E1-E6 and the gap method, enclosed quantities, forms and claim protocol, proof v2 | `pytest tests/test_certified_hydrostatics_ch3_query.py` with fixtures: the degree-`(1,1)` planar box (exact forms equal to CH1 for planes missing the ruled face, claims containing CH1's values otherwise, section 8 (iv)); the quarter-cylinder prism at 5 drafts (conic references inside claims); B1 half hull upright at `h = 0.5` (EMPTY/boundary), `1.5`, `2.9`, `3.0` (FULL/boundary, both limits), the heel `0.3` plane through `(2.5, 0, 1.5)` (planar-only section: no cut arc, enclosed quantities, H1 value inside), the ANYstability-convention grid `heel in {-0.5, -0.3, -0.1, 0.1, 0.3, 0.5} x trim in {-0.02, 0, 0.02}` through `(2.5, 0, 1.5)`, a plane through a bow corner (E1, both limits), a plane through two mirrored corners of the full hull (identical-root recognition), an edge-tangent plane (E2 refusal); the sandwich check at five upright draft pairs; H1 values inside every claim | Sonnet for the engine from this specification; Opus for sections 4.3 and 5.2 and review | CH3a, CH3b |
| CH3d | verifier extension (section 6), written by a separate session from this document | `pytest tests/test_certified_hydrostatics_ch3_verify.py`: agrees on every CH3c fixture; refuses each attack: core narrowed by `1 ulp` of its width, claim not equal to `core ± width`, `exact` key present on an enclosed quantity, altered `pieces`/`isolation_bits`, altered corner interval or hint, a result for the mirrored hull presented for the half hull, planar attacks of the CH1 manifest re-run on a CH3 preparation | separate Sonnet session (Opus review) | CH3c |
| CH3e | Q08 run set and delivery addendum: full and half B1, the planes of CH3c plus `heel in {+-0.1, +-0.3, +-0.5} x trim in {0, +-0.02}` through five draft points, `delta_hint`, policies, work counts, timings, references, refusals retained | all planes answer or refuse typed; every answer verifies with CH3d; references of section 8 inside every claim; half/full criterion met; run record written before assessment per the manifest rules | Opus | CH3c, CH3d |

Codex Sol reviews every slice; this design gets a Codex Astra review before CH3a starts. Patch version
bump at CH3e (`0.4.7`).

## 10. Open risks

- **Bit growth**: Taylor coefficients of `Q^{-k}` at order 32 reached 2130 bits on one factor; with
  `k` up to 5 (second moments) and partial-fraction coefficients of ~200-bit size the 4096 cap may be
  reached on long pieces; mitigation is more pieces at lower order (measured: order 12 on two halves
  beats order 16 on one piece), measured in CH3b before the engine is written.
- **Multi-root predicates**: the full hull's across-lines and cut segments on its planar faces mix
  roots; undecided signs refuse rather than resolve (a true zero that is not structurally recognised,
  e.g. a plane through two unrelated corners, refuses `UNDECIDED_PREDICATE`). Q08 planes are rational
  and generic except by construction; the acceptance set includes the recognisable symmetric case.
- **Event isolation in `d`**: the gap method isolates heights of all events; near-coincident but
  distinct events force deep refinement (bounded, refusable). The B1 deck/bottom planes have corner
  heights exactly rational (`z = 4 v`), so their gaps are cheap.
- **Runtime**: unmeasured end to end; the kernel pieces are milliseconds, root isolations sub-millisecond,
  but a query touches two curved faces x (corners + cut points) pieces x up to 10 quantities. CH3c
  records timings; the manifest's 600 s ceiling per plane set is kept until measured.
- **Embedding of curved pairs** is decided only by box disjointness; shells with curved faces that
  touch or nearly touch refuse. Fine for B1.
- **The hint tolerance** is a policy number; a model whose stored corners were computed more loosely
  than `2^-49` refuses `TRIM_HINT_MISMATCH` rather than being accepted (documented limit, D-CH3-2).
- `mpmath` as a test-only dependency (not installed today) or an in-repo high-precision reference
  using `decimal` (more code to review) — D-CH3-8.

## 11. Owner decisions

- **D-CH3-1 Certified domain.** Options: A ruled rational class (section 2.1), B rational NURBS trimmed
  by planes, C general trims. *Recommend A.*
- **D-CH3-2 Exact model.** Options: (a) supports exact, trims redefined as exact plane ∩ surface,
  stored domains and corners as hints checked within `delta_hint = 2^-40 (1 + max |coord|)` and
  recorded; (b) stored corners exact and the shell admitted only if exactly closed (B1 fails: its
  corners are rounded, so this means refusing B1); (c) hints without any check. *Recommend (a).*
- **D-CH3-3 Quantity method.** Options: (i) closed forms with rigorous `log`/`atan`, (ii) exact
  rational Taylor models with Bernstein remainders and partial fractions, (iii) rigorous quadrature.
  *Recommend (ii); (i) only as a high-precision reference in tests.*
- **D-CH3-4 Forms for enclosed quantities.** Options: (a) the claim protocol of 4.4 (`core`, claim
  `= core ± width`, binary64 as outward rounding with its width as bound, `exact` absent, default core
  target `10^-12`); (b) adjacent-double binary64 attempted up to a depth limit then refused; (c)
  enclosures only, no binary64 form. *Recommend (a).*
- **D-CH3-5 Degenerate planes on curved faces.** Options: (a) support E1, E5, E6 by the certified event
  gap, refuse E2-E4 `CURVED_SECTION_DEGENERACY`; (b) refuse every degenerate plane touching a curved
  face; (c) support tangencies too (Puiseux-type limits, CH4+ work). *Recommend (a).*
- **D-CH3-6 Kernel provenance.** Options: (a) re-implement polynomial algebra, root isolation and
  intervals inside `certified_hydrostatics` with metering, the verifier with its own copies; (b) reuse
  `analytic_roots`/`exact_levels`/`exact_predicates` (unmetered, memoising, shared with the provider,
  so the verifier would not be independent). *Recommend (a); the existing modules are references.*
- **D-CH3-7 Policy defaults.** `max_bits` default stays 4096 and the CH3 policy adds
  `max_isolation_bits = 4096`, `max_taylor_order = 32`, `max_pieces_per_integral = 1024`,
  `max_refinement_rounds = 8`, `default_enclosure_width = 10^-12`, `delta_hint` as in D-CH3-2; raised
  only on CH3b/CH3c measurements. *Recommend as stated.*
- **D-CH3-8 Reference tooling.** Options: (a) `mpmath` as a test-only dependency for the 40-digit
  references of section 8; (b) an in-repo `decimal`-based reference. *Recommend (a).*
- **D-CH3-9 Convention version.** Curved preparations write convention version `2` (new snapshot
  fields, `enclosure` form, `(edge, ordinal)` loop identities, curved contacts); planar preparations
  keep version `1` byte for byte. *Recommend as stated.*
- **D-CH3-10 Verifier authorship.** As D-CH4 of the planar design: CH3d written by a separate session
  from this document only, with a different root-isolation algorithm and a different Taylor scheme.
  *Recommend as stated.*
