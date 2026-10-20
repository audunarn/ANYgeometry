# Certified hydrostatics: design (CH0, revision 2, 2026-10-20)

Status: design for owner review; no source changed. Revision 2 answers the Codex Astra review of
revision 1 (4 blocking, 6 major, 2 minor; table in section 12). Request: ANYstability,
`C:/Github/ANYstability/docs/ANYgeometry_CERTIFIED_HYDROSTATICS_HANDOFF.md` (20 October),
requirements GEO-01..GEO-11 in `ANYstability/docs/ANYgeometry_REQUIREMENTS.md`. Owner decision
2026-10-20: certified point queries for planar shells first (CH1 + CH2), the curved B1 hull
later (CH3), events/intervals after (CH4). The bow programme (step 4) waits.

Risk and maturity (policy `ANY_ECOSYSTEM_RISK_PROPORTIONATE_ENGINEERING_V1`, 2026-09-24.2):
high consequence (certified engineering quantities a consumer relies on); CH1 is Level-A
development with early independent review; the CH1d/CH2 runs are a Level-B candidate with the
acceptance criteria of section 10 fixed before they run.

**Scope statement.** CH1/CH2 deliver a certified provider for a *planar subset* (closed
polyhedral shells such as every recipe3 solid). They do not by themselves open ANYstability's S2
admission: the full B1 fixture and Q08 (CH3), the event/interval API (CH4), opening/sill and
clearance support, and the consumer's own gates (reviewed adapter mapping, installed
verification, Capytaine panel comparison, P0/P1 measurements and their authority) remain open
and are listed in the CH2 delivery inventory.

## 1. Facts this design rests on

- The consumer's reference vessel (recipe3: `HULL`, `PUMPROOM`, `V1..V4`, `WB1..WB8`) has 830
  `Plane` faces, all straight edges, no holes, loops of at most 4 vertices, every solid a clean
  `closed_shell`, every face loop exactly coplanar in its float vertices and contained exactly
  in its stored `Plane` support (both checked with exact rational arithmetic, 2026-10-20).
- Inputs are the exact binary values of the stored floats (`Fraction(x)`,
  `x.as_integer_ratio()`), never decimal strings, rounded intermediates or normalised vectors.
- A rational polyhedron cut by a rational half-space is a rational polyhedron: its volume and
  first volume moments are rational. A rational planar polygon in a plane with rational normal
  `n` has area and polynomial moments `Q / |n|` with `Q` rational and `|n|^2` rational.
- A signed fan over a closed oriented boundary chain is an exact signed integration
  decomposition (non-convex, with holes, with cancelling bridges); it needs no loop assembly.

The current `hydrostatics()` (H1) stays a development API (quadrature estimates). The certified
API is new and does not reuse `Frame` (which normalises axes).

## 2. Admitted domain (CH1): exact validity of the oriented solid

A preparation takes a model and an explicit face set (the displacement boundary chosen by the
caller; excluding frames, compartment walls or lids is the caller's choice) and admits it only
if every check holds, each decided exactly in rational arithmetic:

1. **Faces**: every support is a `Plane`, every loop edge straight, every coordinate finite,
   every loop vertex exactly in the face's support plane (D-CH3: refuse otherwise).
2. **Loops**: every edge has non-zero length; every loop is simple (non-consecutive edges do not
   touch; consecutive edges meet only at their shared vertex; no zero-angle spike); every face
   has non-zero area.
3. **Holes**: every hole loop lies strictly inside its outer loop, hole loops are pairwise
   disjoint (not touching), and every hole winds opposite to its outer loop with respect to the
   face normal.
4. **Shell combinatorics**: every edge is used by exactly two face loops, in opposite directions;
   the faces form one connected shell; the link of every vertex (its incident faces ordered
   through shared edges) is a single cycle (manifold vertex).
5. **Embedding**: any two faces intersect exactly in their declared shared cells (shared edges,
   shared vertices) and nowhere else, for all pairs including adjacent ones (exact
   polygon-polygon intersection with a bounding-box prefilter).
6. **Orientation**: the exact signed volume is positive.

With 1-6 the face set is the oriented boundary of a polyhedral solid (a closed, embedded,
oriented 2-manifold, outward loops), which the identities of section 4 need. Anything else is
refused with a structured category (section 7). Curved faces refuse `UNSUPPORTED_SURFACE` until
CH3. Each recipe3 solid (the hull and each tank) is its own preparation.

## 3. Frame, request and semantics

- **Frame and units**: the model's coordinates are the body frame; no rotation is applied. The
  caller builds the half-space in body coordinates (ANYstability: `R[2] . r <= -origin_z`).
  Units are recorded in the preparation (metres for recipe3).
- **Half-space**: `n . r <= d`, `n` three finite floats not all zero, `d` finite, used as exact
  rationals, never normalised or substituted. With this unnormalised form `dV/dd = A / |n|`.
- **Waterplane basis**: origin `o`, axes `e1`, `e2` (finite floats, exact rationals),
  `u = e1 . (r - o)`, `v = e2 . (r - o)`, used exactly as given (D-CH2; no orthonormalisation).
  Required: `(e1 x e2) . n != 0` exactly (else `INVALID_QUERY`); handedness is its sign.
- **Quantities** (consumer order):

  | Group | Quantities | Exact form |
  |---|---|---|
  | Volume | `V = int dV` | rational |
  | First volume moments about the body origin | `J = int (x, y, z) dV` | rational |
  | Waterplane area | `A = int dA` | rational / `|n|` |
  | First waterplane moments | `(int u dA, int v dA)` | rational / `|n|` |
  | Second waterplane moments | `(int u^2 dA, int v^2 dA, int u v dA)` | rational / `|n|` |

  `dA` is the true area element of the plane. Centroids, BM and `V * CB` are not certified
  outputs; the consumer derives them from the bounded raw quantities.
- **Forms and bounds** (D-CH1): each quantity is returned (a) exactly (a rational, or a rational
  numerator with `|n|^2` as a rational); (b) as its binary64 enclosure, the adjacent doubles
  `[lo, hi]` containing the exact value (`lo == hi` when it is a double); (c) on request as a
  decimal enclosure of a requested absolute width (arbitrary-precision decimal strings from
  integer square-root bounds). Bound semantics: the binary64 midpoint `m = (lo + hi) / 2`
  rounded to a double satisfies `|exact - m| <= hi - lo`; a decimal enclosure `[a, b]` contains
  the exact value and `b - a` is at most the requested width. A requested binary64 accuracy finer
  than the double spacing at the value refuses `ACCURACY_UNACHIEVABLE` (the decimal form remains
  available); values outside the binary64 range refuse `RESULT_OUT_OF_RANGE` for that form.
- **Generic and degenerate planes**: a plane is *generic* when no shell vertex lies on it (exact
  test `n . p == d`), otherwise *degenerate*; the result lists the contacts (vertices, edges and
  faces in the plane).
  - `V` and `J` are continuous in `d`: one exact value for every plane.
  - The section is reported as **one-sided limits** `below` (`d - epsilon`) and `above`
    (`d + epsilon`), each with its state, waterplane quantities and topology (section 5). For a
    generic plane both coincide with the exact section and one section is returned. For a
    degenerate plane both are always returned and no "plain" section is claimed, even when the
    quantities agree (the review's saddle solid `z <= |x| - |y|`: connected below, two components
    above, equal quantities).
  - **States per side**: `EMPTY` (no part of the solid below the perturbed plane), `FULL` (all of
    it), `PARTIAL`. A box cut at its top face is `FULL` with `V` = box volume; its `below` section
    is the top rectangle, its `above` section is empty. Absent quantities are `None` with the state
    as the reason, never zero-filled.
  - The consumer's upright WB1/WB2/WB7/WB8 main-deck query (GEO-06) thereby has an exact,
    documented meaning instead of an exclusion.

## 4. Quantities: the integral argument and the computation

**Argument.** Let `S` be the admitted solid, `H = {n . r <= d}` and `K = S cap H`. For the
`below` convention (vertices on the plane counted above), the boundary of `K` is (i) the parts of
the faces of `S` in `H` and (ii) the cap `C = S cap {n . r = d}` with outward normal `n / |n|`;
`V`, `J` are the same for either convention (the parts differ by measure zero). By the
divergence theorem with the fields `r / 3` and `x_i r / 4`, `V` and `J` are sums of boundary
integrals. On each planar boundary piece, with apex at the body origin, these integrals are the
signed tetrahedron sums `V += det(a, b, c) / 6`, `J += det(a, b, c) (a + b + c) / 24` over the
triangles `(a, b, c)` of a fan of its oriented boundary chain, exact for any closed oriented
chain (non-convex pieces, holes, cancelling bridges).

**Computation.**
1. *Faces*: every face loop is clipped against the half-space with the `below` convention by
   Sutherland-Hodgman in exact rationals; the output is used **only** as a signed boundary chain
   for the tetrahedron sums, never as section loops.
2. *Cap*: its boundary chain is the set of directed **cut segments** of section 5 (the `below`
   limit), oriented with the cap on their left seen from `+n`; a fan from a point `q` on the
   plane over these segments gives the cap's tetrahedra. No loop assembly is needed.
3. *Waterplane quantities* (per limit): from that limit's cut segments, `int u^a v^b dA` by the
   exact triangle formulas over `(q, a, b)`, with signed area element
   `((b - a) x (c - a)) . n / (2 |n|)` (rational / `|n|`).

Arithmetic is exact (`Fraction` or integer-scaled). The number of arithmetic operations is
linear in the number of loop vertices for the query, but bit lengths grow with the summands and
their differing denominators; work is therefore counted in operations *and* each value is
checked against a bit-length cap (section 7). Runtime is measured in CH1d, not assumed.

## 5. Sections: symbolic cut points, segments, loops, topology

Simulation of simplicity on `d`: for `below` every vertex with `n . p == d` is treated as
strictly above, for `above` as strictly below. No vertex then lies on the perturbed plane, so:

- every **cut point** is the crossing of one shell edge whose endpoints are on opposite sides,
  identified symbolically by that edge, with exact limit coordinates (the exact crossing, or the
  on-plane vertex for an edge with one endpoint on the plane);
- every face crossed by the plane has an even number of crossed edges. Its cut points lie on the
  face's cut line; they are ordered along it by exact limit position, ties (several cut points at
  one on-plane vertex) broken by the first-order perturbation term (each crossing's position as
  `epsilon -> 0`: a rational slope in `epsilon`); consecutive points in that order alternate
  between entering and leaving the face (crossing parity along the line, holes included), and
  each inside interval is one **cut segment**, directed from the face orientation;
- a face lying in the plane is entirely above (`below`) or entirely below (`above`): it adds no
  segment and is never counted twice;
- every cut point (edge) belongs to exactly two faces, so the cut-point graph has degree 2 and
  splits into closed **loops** with no coordinate matching. Exact signed areas classify loops as
  outer (positive about `n`) or holes; nesting is decided exactly (point in polygon on limit
  coordinates, ties at shared limit points broken by the perturbation order). Components are
  outer loops with their holes. Section topology comes only from these loops; the quantities
  never depend on loop assembly.
- **Identity**: a loop's identity is the sorted tuple of the edges it crosses (with the
  preparation's identities); meaningful within one preparation.

For a degenerate plane the exact (unperturbed) section is not claimed; the contacts are listed,
and point or edge contacts appear as differences between the two limits.

## 6. Preparation lifecycle, proof and verification

**Snapshot semantics** (no live-model promise). `prepare_certified_hydrostatics(model, faces,
units, policy, budget)` reads the model once and returns an immutable
`PreparedCertifiedHydrostatics`: the exact oriented geometry snapshot (rational loops, edge and
face identities, shared cells), the domain verdict, units and frame convention, policy, model
identity (model id, revision, document checksum) and implementation identity. Queries answer for
the snapshot. `check_preparation(model, prepared)` and verification rebuild the snapshot from
the model they are given and compare its canonical digest exactly, so mutations that bypass
revision counters, clones sharing an identity and mutation during preparation are caught by
content, not by counters.

**Proof schema** (canonical JSON, sorted keys, rationals as integer pairs): `artifact`
(distribution name and version, SHA-256 of every file of the installed `anygeometry` package,
versions of its runtime dependencies, Python version); `model` (model id, revision, document
checksum, selected face ids, snapshot digest = SHA-256 of the canonical exact snapshot);
`convention` (convention version, units, body-frame statement, half-space form, area element,
moment order); `preparation_policy`; `query` (exact `n`, `d`, `o`, `e1`, `e2`, requested
accuracies and forms); `result` (the complete claim: degeneracy and contacts; `V`, `J`; per side
the state, every quantity in every returned form with its bound, the topology with identities;
achieved-accuracy flags; work counts).

**Verifier.** `verify_certified_hydrostatics(model, faces, query, result, expected, budget,
cancellation_check)` receives the *complete result object* and the consumer's **expected
identities** (`expected`: artifact hashes or wheel SHA-256, convention version, units, policy),
never trusting the proof's own declarations. It (1) recomputes the canonical proof from the
result object and requires it equal to `result.proof` and consistent with `expected`;
(2) rebuilds the snapshot from the supplied model and faces, re-runs the section 2 checks and
requires the digest to match; (3) recomputes every exact quantity, both limits and the topology
independently: volume and first moments by the divergence theorem with Mirtich's projected face
integrals on the clipped faces and the cap; cut points by intersecting every edge with the plane
and segments by exact interval classification along each face's cut line; area and moments by
Green's theorem in the coordinates of the two smallest components of `n`; (4) requires exact
equality and checks every enclosure (contains the exact value, meets its claimed width). It
returns a `VerifiedHydrostatics` object for the consumer to use exclusively; any discrepancy
raises `VERIFICATION_FAILED` with a sub-reason.

**Trust model** (trusted checker). The claim holds for the exact rational problem defined by the
stored floats and the request, given (a) the integral identities of section 4 for solids
admitted by section 2 (stated and argued in the documentation); (b) the correctness of the
verifier (small, reviewed, written separately from this specification, D-CH4) and of Python's
exact integer and `Fraction` arithmetic; (c) the outward rounding of enclosures. The main
implementation is untrusted. Agreement of the two implementations corroborates; (a) with the
trusted checker (b) is the justification. Nothing is claimed about the physical vessel, its
source drawings, tank communication or loading.

## 7. Refusals, work, cancellation

`CertifiedHydrostaticsRefusal(GeometryError)` carries `category` (public enum), `phase`
(`prepare`, `query`, `verify`, `check`), the exact request, model and preparation identity, the
work consumed so far, and the original cause (`__cause__`). Categories: `UNSUPPORTED_SURFACE`,
`NOT_EXACTLY_PLANAR`, `INVALID_LOOP`, `INVALID_HOLES`, `NOT_CLOSED`, `NON_MANIFOLD`,
`SELF_INTERSECTING`, `INVERTED_ORIENTATION`, `NONFINITE_INPUT`, `INVALID_QUERY`,
`ACCURACY_UNACHIEVABLE`, `RESULT_OUT_OF_RANGE`, `BIT_LENGTH_EXCEEDED`, `BUDGET_EXHAUSTED`,
`CANCELLED`, `DEADLINE_EXCEEDED`, `STALE_PREPARATION` (from `check_preparation`),
`MALFORMED_PROOF`, `VERIFICATION_FAILED` (sub-reasons: artifact, model, face set, snapshot,
convention, policy, query, value, enclosure, topology). Consumers never parse messages; a
refusal is final for its request (no retry with a changed problem).

**Work**: one caller-owned `WorkBudget` (shareable across preparation, queries, batches and
verification) limits arithmetic operations, the bit length of any value and optionally a
deadline. Work is reserved before each unit (a loop vertex clipped, an edge intersected, a
triangle accumulated, a face pair tested, a square-root refinement step) and checked there, so
the overshoot after a refusal is at most one unit (bounded by the bit-length cap).
`cancellation_check` and the deadline are polled at the same checkpoints; the non-preemptible
span is one unit. Work counts are returned on success and on refusal. A proof is size-checked
before parsing (`MALFORMED_PROOF`).

## 8. Events and intervals (CH4 preview; not delivered in CH1)

For fixed `n` the shell vertex heights `n . p` partition the `d` axis into open intervals of
fixed section combinatorics, on which `V(d)` is a polynomial of degree at most 3 (`A(d)` at most
2). Vertex heights are *candidate* events: a redundant vertex or a combinatorial change need not
change component/hole topology; actual topology events are classified by comparing the one-sided
limits. Attitude paths need a restricted path representation (for example rational
parametrisations) and certified root isolation, designed in CH4. Path, branch and stability
responsibilities stay with the consumer.

## 9. API sketch (names final in CH1)

```python
budget = WorkBudget(max_operations=..., max_bits=..., deadline=None)
prepared = prepare_certified_hydrostatics(model, faces, units="m", policy=policy,
                                          budget=budget, cancellation_check=None)
query = HalfSpaceQuery(normal=n, offset=d, waterplane_origin=o, waterplane_axes=(e1, e2),
                       absolute_accuracy={"A": 1e-12}, forms=("exact", "binary64", "decimal"))
result = query_certified_hydrostatics(prepared, query, budget=budget, cancellation_check=None)
verified = verify_certified_hydrostatics(model, faces, query, result, expected=identities,
                                         budget=budget, cancellation_check=None)
check_preparation(model, prepared)          # STALE_PREPARATION when the snapshot no longer matches
```

## 10. CH1 slices and acceptance

| Slice | Output | Acceptance | Author |
|---|---|---|---|
| CH1a | snapshot, admission checks 1-6, refusal enum, implementation identity, `WorkBudget` | recipe3 hull and every tank admit; each refusal category is raised by a minimal case, including the review's wrong-winding bored box (`INVALID_HOLES`), a touching-hole face, two cubes sharing only a vertex (`NON_MANIFOLD`), an adjacent-face overlap and a self-intersecting shell (`SELF_INTERSECTING`) | Sonnet from this specification; Opus review |
| CH1b | query: clipped-face sums, symbolic cut points, segments, loops, topology, both limits, forms and enclosures | closed forms (Q01): box, tetrahedron, prism, extruded U (two components), bored box (a hole), the saddle solid (below connected, above two components, equal quantities), a box at its top face (FULL; below = top rectangle; above empty; no double cap); `dV/dd = A/|n|` against finite differences of exact `V` on an interval; Q02: translations, exact rational rotations, moment-origin changes, reflection; binary64 width limits and `ACCURACY_UNACHIEVABLE`; planes built with ANYstability's own convention at finite heel/trim | Sonnet; Opus review |
| CH1c | the independent verifier (section 6) | agrees exactly with CH1b on every CH1b case and on recipe3 for a registered plane set; refuses each substitution, alteration, staleness, wrong expected identity and malformed proof | a separate Sonnet session given only this specification |
| CH1d | recipe3 runs: hull and WB tanks, upright and inclined, the WB1/2/7/8 deck plane | all admit and verify; the deck plane returns both limits; work and timings recorded | Opus |
| CH2 | delivery folder (the handoff's section 5 inventory for the planar subset): wheel + SHA-256, source commit, docs (API, domain, argument, trust model), commands, requests, responses, proofs, verifier runs, refusals, work accounting, and the open obligations of the scope statement | installed-wheel runs in an isolated environment outside the checkouts | Sonnet; Opus check |

Codex Sol reviews every slice; revision 2 of this design gets a Codex Astra review first. Patch
version bump at CH2.

## 11. Owner decisions (2026-10-20, all as recommended)

D-CH1 exact rationals with enclosures, one-sided limits by perturbing `d`; D-CH2 basis used as
given (with the `(e1 x e2) . n != 0` requirement added in revision 2); D-CH3 refuse
`NOT_EXACTLY_PLANAR`; D-CH4 verifier written from the specification by a separate session;
D-CH5 curved shells in CH3.

## 12. Revision 1 review: findings and responses

| # | Finding | Response |
|---|---|---|
| B1 | admission did not prove a valid oriented solid | section 2: exact loop, hole, manifold and full embedding checks; the bored-box counterexample is a CH1a test |
| B2 | clipped loops are not section loops (bridges) | sections 4-5: clipped faces only as signed chains; sections from symbolic cut points |
| B3 | perturbation lost incidence; equal quantities do not imply equal topology; double caps | section 5: cut points identified by edge, degree-2 graph without coordinate matching; both limits always for degenerate planes; faces in the plane never double-counted |
| B4 | the verifier checked the proof, not the result; agreement is not proof | section 6: verifies the complete result against expected identities and returns the verified object; trusted-checker model with the integral argument |
| M5 | arbitrarily narrow binary64 intervals impossible | section 3: exact, binary64 and decimal forms; defined bound semantics; `ACCURACY_UNACHIEVABLE`, `RESULT_OUT_OF_RANGE` |
| M6 | bindings underspecified | section 6: proof schema; whole-package hashes and dependency versions; expected identities from the consumer |
| M7 | staleness vs a snapshot API | section 6: snapshot semantics; `check_preparation` and verification compare content |
| M8 | work and cancellation incomplete | section 7: one shared budget, per-unit reservation, bit-length cap, deadline, counts on refusal, bounded proof parsing |
| M9 | basis validity, FULL/EMPTY with sections, `dV/dd` | section 3 |
| M10 | scoped delivery | scope statement; the CH2 inventory lists open obligations |
| m11 | complexity claim | section 4: operation counts, bit-length cap, measured runtime |
| m12 | vertex heights are candidate events | section 8 |
