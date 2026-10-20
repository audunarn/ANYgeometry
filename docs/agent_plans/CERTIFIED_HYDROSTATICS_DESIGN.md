# Certified hydrostatics: design (CH0, revision 5, 2026-10-20)

Status: design for owner review; no source changed. Revision 2 answered the Codex Astra review of
revision 1 (4 blocking, 6 major, 2 minor); revision 3 the review of revision 2 (2 blocking,
4 major, 1 minor); revision 4 the review of revision 3 (1 blocking, 3 major, 2 minor). All are
tabled in section 12; revision 5 fixes the two majors of the round-4 review. Request: ANYstability,
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
  the exact value and `b - a` is at most the requested width. A requested binary64 accuracy is
  judged against the returned form's own bound: a zero-width enclosure (`lo == hi`, the exact
  value is a double) meets every accuracy; otherwise the bound is `hi - lo`, and a request finer
  than it refuses `ACCURACY_UNACHIEVABLE` (the decimal form remains available). Values outside the
  binary64 range refuse `RESULT_OUT_OF_RANGE` for that form.
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
  - **States.** The result has an **occupancy** for the exact half-space, `EMPTY` (`V = 0`),
    `FULL` (`V` = the solid's volume) or `PARTIAL`, which governs `V` and `J` (always present,
    exact; `0` for `EMPTY` is an exact value, not an absence). Each side (`below`, `above`) has
    its own **occupancy** for its perturbed half-space and a **section** flag `PRESENT` or
    `ABSENT` (no crossed edge); waterplane quantities and topology are present exactly when the
    side's section is `PRESENT`, otherwise `None`, never zero-filled. Example, a box cut at its top
    face: occupancy `FULL`; `below`: occupancy `PARTIAL`, section `PRESENT` (the top rectangle);
    `above`: occupancy `FULL`, section `ABSENT`.
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

Arithmetic is exact (`Fraction` or integer-scaled). Clipping and accumulation take a number of
operations linear in the number of loop vertices; section construction adds the ordering of the
crossings on each face's cut line (`O(k log k)` comparisons for `k` crossings) and the nesting
tests (at most loops x loop edges); admission's embedding check is quadratic in faces in the
worst case (bounding boxes prefilter it). Bit lengths grow with the summands and their differing
denominators, so work is counted in primitive units *and* each value is checked against a
bit-length cap (section 7). Runtime is measured in CH1d, not assumed.

## 5. Sections: symbolic cut points, segments, loops, topology

Simulation of simplicity on `d` (Edelsbrunner and Muecke): the `below` side uses the plane
`n . r = d - epsilon`, the `above` side `n . r = d + epsilon`, for all sufficiently small
`epsilon > 0`. No shell vertex lies on a perturbed plane. The crossing of an edge `(p, q)` with
`n . p < d - epsilon < n . q` (or the reverse) is `p + t (q - p)` with
`t = (d - epsilon - n . p) / (n . (q - p))`, affine in `epsilon`; every cut point is therefore an
exact **affine point** `c0 + epsilon c1` (rational vectors). Every predicate below (comparison of
positions along a line, orientation of three points, signed area, point-in-polygon side tests)
is evaluated on these affine points as a polynomial in `epsilon` with rational coefficients,
and its sign is **the sign of its first nonzero coefficient**. Perturbing `d` alone does not
remove every degeneracy (two holes in one face can have a vertex of one collinear with an edge of
the other for every `epsilon`), so a zero polynomial is an *outcome* each predicate must handle,
not an error. Only the predicates proved nonzero for a valid admitted solid refuse
`SYMBOLIC_DEGENERACY` when they come out zero: the order of two distinct cut points on one cut
line, a loop's signed area, and a test point lying on another loop's edge (loops are disjoint).
Comparisons along a line are degree 1, orientations and signed areas degree 2. Quantities use the
limit points `c0` (the limits of the integrals as `epsilon -> 0`). Then:

- every **cut point** is the crossing of one shell edge whose endpoints are on opposite sides,
  identified symbolically by that edge, with exact limit coordinates (the exact crossing, or the
  on-plane vertex for an edge with one endpoint on the plane);
- every face crossed by the plane has an even number of crossed edges. Its cut points lie on the
  face's cut line; they are ordered along it by the symbolic comparison (degree 1); consecutive
  points in that order alternate
  between entering and leaving the face (crossing parity along the line, holes included), and
  each inside interval is one **cut segment**, directed from the face orientation;
- a face lying in the plane is entirely above (`below`) or entirely below (`above`): it adds no
  segment and is never counted twice;
- every cut point (edge) belongs to exactly two faces, so the cut-point graph has degree 2 and
  splits into closed **loops** with no coordinate matching. A loop's symbolic signed area about
  `n` (degree 2) classifies it as outer (positive) or hole (negative), also when its limit area
  is zero (the tetrahedron `(0,0,0), (1,0,0), (0,1,0), (0,0,1)` at `z = 1`, `below`: area
  `epsilon^2 / 2`, one outer loop; a pyramidal recess gives the collapsing hole). Nesting: the
  perturbed loops are disjoint simple polygons, so one vertex `p` of a loop decides whether it lies
  inside another loop `L`. The test is ray parity in the section's projection coordinates (drop
  the index `k` maximising `|n[k]|`, lowest index on ties; coordinates `(a, b)` are the other two
  in increasing index order): cast the ray from `p` in the `+a` direction; an edge `(q, r)` of `L`
  counts when exactly one endpoint has `b` coordinate `<= p.b` (half-open rule: `q.b <= p.b < r.b`
  or `r.b <= p.b < q.b`, symbolic comparisons) and the crossing lies strictly right of `p` (the
  sign of the degree-2 orientation of `(q, r, p)` against the edge's `b` direction). Collinear
  configurations outside an edge's half-open span (the two-hole example) are thereby resolved
  deterministically; a zero orientation for a counted edge means `p` is on `L` and refuses
  `SYMBOLIC_DEGENERACY`. Components are outer loops with their holes.
  Section topology comes only from these loops; the quantities never depend on loop assembly.
- **Identity**: a loop's identity is the sorted tuple of the edges it crosses (with the
  preparation's identities); meaningful within one preparation.

For a degenerate plane the exact (unperturbed) section is not claimed; the contacts are listed,
and point or edge contacts appear as differences between the two limits.

## 6. Preparation lifecycle, proof and verification

**Snapshot semantics** (no live-model promise). `prepare_certified_hydrostatics(model, faces,
units, policy, budget)` reads the model in one consistent read and returns an immutable
`PreparedCertifiedHydrostatics`: the exact oriented geometry snapshot (rational loops, edge and
face identities, shared cells), the domain verdict, units and frame convention, policy, model
identity (model id, revision, document checksum) and implementation identity. Queries answer for
the snapshot. **Consistent read**: the preparation refuses (`CONCURRENT_MUTATION`) if a model
transaction is open, and records the model's revision before and after collecting the snapshot,
refusing if they differ; it then validates only the collected snapshot. `GeometryModel` is not
thread-safe: the contract requires that no other thread mutates the model during preparation,
`check_preparation` or verification (a documented precondition, not detected beyond the revision
guard; public mutation always advances the revision). Later changes are caught by content:
`check_preparation(model, prepared, budget, cancellation_check)` and verification rebuild the
snapshot from the model they are given (under the same consistent-read guard) and compare its
canonical digest exactly, so clones sharing an identity or edits that leave counters unchanged
cannot pass as the prepared geometry.

**Proof schema** (canonical JSON, sorted keys, rationals as integer pairs): `artifact`
(distribution name and version, SHA-256 of every file of the installed `anygeometry` package,
versions of its runtime dependencies, Python version); `model` (model id, revision, document
checksum, selected face ids, snapshot digest). The **snapshot digest** is the SHA-256 of the
canonical exact snapshot: the selected face ids in order, and per face its loops as oriented
edge identities and exact vertex coordinates, and the vertex and edge identities with their
shared-cell incidences; it does not include model id, revision or document checksum, which are
separate fields;
`convention` (convention version, units, body-frame statement, half-space form, area element,
moment order); `preparation_policy`; `query` (exact `n`, `d`, `o`, `e1`, `e2`, requested
accuracies and forms); `result` (the complete claim: degeneracy and contacts; `V`, `J`; per side
the state, every quantity in every returned form with its bound, the topology with identities;
achieved-accuracy flags; work counts).

**Verifier.** `verify_certified_hydrostatics(model, faces, query, result, expected, budget,
cancellation_check)` receives the *complete result object* and the consumer's **expected
identities** (`expected`: artifact hashes or wheel SHA-256, convention version, units, policy),
never trusting the proof's own declarations. It (0) requires the query bound in the result to
equal the caller-supplied `query` exactly (`n`, `d`, `o`, `e1`, `e2`, requested forms and
accuracies): equal quantities do not identify a query (a fully submerged box gives the same
result at `z <= 2` and `z <= 3`); (1) recomputes the canonical proof from the result object and
requires it equal to `result.proof` and consistent with `expected`;
(2) rebuilds the snapshot from the supplied model and faces, re-runs the section 2 checks and
requires the digest to match, and recomputes the model id, revision and document checksum from the
supplied model, and the complete artifact descriptor from the running environment (distribution
name and version, every package file hash, runtime dependency versions, Python version), and
requires each to equal the proof's fields; the consumer's `expected` artifact identity is the per-file SHA-256 list of
the wheel it admitted (the wheel's own `RECORD`, from the delivery manifest), compared with the
installed files, not a hash of the wheel archive; (3) recomputes every exact quantity, the degeneracy and the
contact lists (vertices, edges and faces in the plane), the occupancies, both limits and the
topology
independently: volume and first moments by the divergence theorem with Mirtich's projected face
integrals on the clipped faces and the cap; cut points by intersecting every edge with the plane
and segments by exact interval classification along each face's cut line; area and moments by
Green's theorem in projection coordinates (drop the index `k` maximising `|n[k]|`, lowest index
on ties; remaining coordinates in increasing index order; the projected area is multiplied by
`|n| / |n[k]|` and its sign by `(-1)^k sign(n[k])` with zero-based `k`: dropping the middle index
with increasing-order coordinates reverses the orientation, e.g. the unit cube at `y <= 1/2`);
(4) requires exact
equality, and checks every returned form against the request and its own rules: a binary64
enclosure has `lo == hi` (a double equal to the exact value) or `lo`, `hi` adjacent doubles with
`lo < exact < hi`; a decimal enclosure contains the exact value and its width is at most the
requested width; every achieved-accuracy flag is recomputed (`True` exactly when the returned
form meets the requested absolute accuracy, a refusal otherwise) and must match; (5) checks
**completeness** against an exact inventory derived from the request and the recomputed states:
`V` and `J`; for each side its occupancy and section flag; for each side with a `PRESENT`
section every waterplane quantity in every requested form, its bound and its achieved-accuracy
flag, and its topology; nothing else (no missing and no extra entries; absence only where the
state rules allow it). It
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

## 6a. Normative formats (added 2026-10-20 after CH1c)

Both CH1b (provider) and CH1c (independent verifier) produce these byte for byte; a change is a
new convention version.

- **Canonical JSON**: UTF-8, sorted keys, separators `,` and `:` without whitespace, integers in
  full (no digit limit), rationals as `[numerator, denominator]` in lowest terms with a positive
  denominator, doubles as `float.hex()` strings, no JSON floats.
- **Snapshot document** (hashed with SHA-256 for the snapshot digest): `{"faces", "vertices",
  "edges", "vertex_incidence"}`. `faces`: in the selected order, each `{"id", "support" (see below), "loops": [[edge id, forward, start vertex id], ...] per loop, outer loop
  first}`, with `support` the list `[origin, u, v]` of exact points (corrected 2026-10-20: an
  earlier wording said a mapping; both implementations write the list). `vertices`:
  `[id, [x, y, z] exact]` sorted by id. `edges`: `[id, start, end,
  sorted [[face id, forward], ...]]` sorted by id. `vertex_incidence`: `[id, sorted face ids,
  sorted edge ids]` sorted by id.
- **Artifact descriptor**: distribution name and version, `__version__`, every file under the
  installed package directory except `__pycache__` directories, sorted by POSIX relative path,
  each with its SHA-256; runtime dependencies = the distribution's `Requires-Dist` entries without
  an `extra ==` marker, compared as a set; the Python version string.
- **`expected` argument of the verifier**: a mapping with `artifact_files` (the per-file SHA-256
  list of the admitted wheel), `convention_version`, `units` and `policy` (the verifier takes its
  parse limits from `expected["policy"]`, never from the proof).
- **Convention block** (convention version 1): exactly these values, compared exactly by the
  verifier (a different string is a different convention and refuses `convention`):
  `body_frame` = "model coordinates are the body frame; no rotation, no normalisation";
  `half_space` = "n.r <= d with exact rational n, d (unnormalised)";
  `area_element` = "true area element dA of the plane; waterplane quantities are numerator/|n|";
  `moment_order` = ["A", "Mu", "Mv", "Iuu", "Ivv", "Iuv"];
  `limits` = "below: n.r = d - eps, above: n.r = d + eps, eps -> 0+; V and J use the below convention";
  `loop_identity` = "sorted ids of the crossed shell edges"; plus `convention_version` and `units`.
- **Work counts** in a result: non-negative integers; `operations` equals the sum of the unit
  counts.
- **Refusal sub-reason**: `CertifiedHydrostaticsRefusal.sub_reason` (a public attribute; `None`
  except for `VERIFICATION_FAILED`), one of the section 7 sub-reasons.

## 7. Refusals, work, cancellation

`CertifiedHydrostaticsRefusal(GeometryError)` carries `category` (public enum), `phase`
(`prepare`, `query`, `verify`, `check`), the exact request, model and preparation identity, the
work consumed so far, and the original cause (`__cause__`). Categories: `UNSUPPORTED_SURFACE`,
`NOT_EXACTLY_PLANAR`, `INVALID_LOOP`, `INVALID_HOLES`, `NOT_CLOSED`, `NON_MANIFOLD`,
`SELF_INTERSECTING`, `INVERTED_ORIENTATION`, `NONFINITE_INPUT`, `INVALID_QUERY`,
`ACCURACY_UNACHIEVABLE`, `RESULT_OUT_OF_RANGE`, `BIT_LENGTH_EXCEEDED`, `BUDGET_EXHAUSTED`,
`CONCURRENT_MUTATION`, `SYMBOLIC_DEGENERACY`,
`CANCELLED`, `DEADLINE_EXCEEDED`, `STALE_PREPARATION` (from `check_preparation`),
`MALFORMED_PROOF`, `VERIFICATION_FAILED` (sub-reasons: artifact, model, face set, snapshot,
convention, policy, query, request, value, enclosure, accuracy flag, state, topology). Consumers never parse messages; a
refusal is final for its request (no retry with a changed problem).

**Work**: one caller-owned `WorkBudget` (shareable across preparation, queries, batches,
verification and `check_preparation`) limits primitive operations, the bit length of any value
and optionally a deadline. A **primitive unit** is one bounded predicate or accumulation: reading
one vertex or edge into the snapshot, one exact coplanarity or orientation test, one edge-pair
intersection test (face-pair tests iterate these, reserving each), one loop-vertex clip, one
edge-plane crossing, one comparison in a sort, one point-in-polygon edge test, one triangle
accumulation, one square-root refinement step, one scalar or array element serialised, parsed or
compared (topology arrays and identity lists count per element), one chunk of at most 64 KiB
hashed (snapshot, document and package files). Each unit
reserves its cost before it runs and every unit is a checkpoint for the budget, the deadline and
`cancellation_check`. These guarantees cover every public phase (preparation, query, verification,
`check_preparation`). Work counts are returned on success and on refusal.

*Span bound (amendment 6d, owner decision 2026-10-20; replaces the earlier claim that every
non-preemptible span is one unit).* In Python, built-in operations (string and tuple comparison,
hashing, set operations, slicing, joins, encodes, sorts' native passes) run as single native calls
over a whole value, so a literal one-unit span is not achievable. The guarantee is:
1. every loop written in this package checkpoints at least once per element, or per 64 KiB of data
   it processes, before that element's work;
2. every non-preemptible span is a native operation over at most one value or collection whose size
   is bounded by a named limit times a constant: `max_proof_bytes` for any string, key, decimal or
   encoded text; `max_integer_digits` (and the bit cap) for integers; the model's entity counts and
   the proof's element count (itself bounded by `max_proof_bytes`) for collections;
3. a span whose data size is not bounded by a policy limit known before it runs is charged in
   proportion before it runs (the rows below), and the bulk step is followed by a checkpoint.
Hence the overshoot after a refusal, and the longest non-preemptible span, are bounded by one such
operation on data within those limits; with the bit-length cap no exact value exceeds the cap.

*Charged bulk operations (amendments 6b/6c, owner decisions 2026-10-20).* These spans are charged in
proportion to their size before they start and are followed by a checkpoint:

| Bulk operation | Bound |
|---|---|
| canonical encoding of the whole model document for `document_checksum` (library serialisation and JSON encoding) | the model's own size (its vertex, edge and face counts); charged one unit per model entity before it runs |
| one UTF-8 decode of the proof bytes | `max_proof_bytes` (checked before decoding) |
| one linear copy, comparison, escape or encoding of a single string scalar, or the final join and UTF-8 encoding of the proof (amendment 6c, owner decision 2026-10-20) | `max_proof_bytes` (a string longer than the limit is refused before it is escaped); charged one unit per 64 KiB before it runs (`string_bulk` in the verifier; `serialize_bulk` in the provider, charged per 64 KiB block as the output grows) |

Every other token scan is bounded by the limit it enforces: an integer
token is scanned to at most `max_integer_digits + 1` digits before the refusal, never to its end.
After a refusal no further outside data are read: refusal context is built only from values already
ingested. The verifier's returned object exposes only values it has checked or recomputed; it never
forwards reads to the supplied result.

*Known limits of the CH1c verifier against 6b: fixed in commit b92a261 (branch claude/ch-fixes-6b).* The three defects found in the final CH1c review are closed:
(a) canonical re-encoding in the verifier now escapes long strings window by window (at most 4096 characters per checkpoint), streams the encoded document as byte windows and compares them incrementally against the proof (a checkpoint per window, no join, no whole-proof `==`); hashing consumes the same stream;
(b) a decimal bound whose denominator `10 ** len(frac)` or numerator is certain to exceed the bit cap is refused `BIT_LENGTH_EXCEEDED` before any power of ten is built (exact integer bounds on log2(10), no floats);
(c) provider side: `write_canonical` no longer builds `10 ** max_digits`; the digit limit is checked by bit length with an exact chunked digit count only in the narrow band at the limit (`_arith.digits_exceed`), so a policy with `max_integer_digits = 2**32` runs inside its budget.
Before parsing, a proof is checked against a byte-size limit and every integer token against a
digit limit (both in the policy); a violation refuses `MALFORMED_PROOF` without converting the
token.

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
| CH1b | query: clipped-face sums, symbolic cut points, segments, loops, topology, both limits, states, forms and enclosures | the fixtures and expected values of the acceptance manifest (below), including closed forms (Q01) for box, tetrahedron, prism, extruded U (two components), bored box (a hole), the saddle solid (below one loop, above two, equal quantities), the box at its top face (states as in section 3, no double cap), the tetrahedron at its apex (collapsing outer loop) and a pyramidal recess at its apex (collapsing hole), a nesting tie at a shared limit point; the exact derivative check; Q02 translations, exact rational rotations, moment-origin changes, reflection; binary64 width limits and `ACCURACY_UNACHIEVABLE`; planes built with ANYstability's convention at finite heel/trim | Sonnet; Opus review |
| CH1c | the independent verifier (section 6) | agrees exactly with CH1b on every manifest case; refuses each attack of the manifest: a result for another query with equal quantities (submerged box at `z <= 2` vs `z <= 3`), altered value, bound, state, flag or topology, deleted forms or flags (a requested decimal form
removed and the proof regenerated), altered contacts or identity fields (including a claimed
Python or dependency version differing from the running one), an enclosure wider than
requested (`[0, 2]` for `V = 1` at accuracy `1e-12`), non-adjacent binary64 endpoints, substituted model or face set, stale snapshot, wrong expected identity, malformed or oversized proof | a separate Sonnet session given only this specification |
| CH1d | recipe3 runs: hull and WB tanks, upright and inclined, the WB1/2/7/8 deck plane | all admit and verify; the deck plane returns both limits; work and timings recorded | Opus |
| CH2 | delivery folder (the handoff's section 5 inventory for the planar subset): wheel + SHA-256, source commit, docs (API, domain, argument, trust model), commands, requests, responses, proofs, verifier runs, refusals, work accounting, and the open obligations of the scope statement | installed-wheel runs in an isolated environment outside the checkouts | Sonnet; Opus check |

**Acceptance manifest** (`docs/agent_plans/CERTIFIED_HYDROSTATICS_ACCEPTANCE.md`, written in CH1a
and reviewed before any CH1d/CH2 assessment run; the criteria are fixed there, not after the runs):
- fixtures: exact definitions (vertices, faces) of every CH1b case, with expected exact values,
  per-side occupancies, section flags, loop counts and nesting;
- the **derivative check**: on an open interval between consecutive vertex heights `V(d)` is a
  polynomial of degree at most 3; it is recovered exactly from four exact evaluations, then its
  exact derivative at an interior `d` must equal `A / |n|`, which is rational (`A = Q / |n|`):
  exact equality, no finite-difference tolerance;
- the recipe3 plane set: for each solid, the upright planes at the drafts listed in the manifest,
  a heel/trim grid built with ANYstability's convention (`R = Ry(-trim) Rx(heel)`), and the
  WB1/WB2/WB7/WB8 deck plane, all fixed before the runs;
- comparison rules: exact equality for every exact form; enclosure rules of section 6;
- refusal coverage per phase: budget exhaustion, deadline, cancellation and bit-length refusal in
  preparation, query, verification and `check_preparation`, each with retained work counts;
- resource limits for the runs (operation budget, bit-length cap, wall-time ceiling per run).

Codex Sol reviews every slice; this design gets Codex Astra reviews until clean. Patch version
bump at CH2.

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

Revision 2 review (round 2):

| # | Finding | Response |
|---|---|---|
| B1 | symbolic orientation and nesting underspecified (collapsing loops have zero limit area) | section 5: affine cut points, every predicate a polynomial in `epsilon` decided by its first nonzero coefficient (degree 2 for orientation and area); `SYMBOLIC_DEGENERACY` for a zero polynomial |
| B2 | verifier did not check the caller's query, requested forms, accuracies, flags and form rules | section 6 steps (0) and (4) |
| M3 | top-face example contradicted the side states | section 3: exact-half-space occupancy separate from per-side occupancy and section flag |
| M4 | hashing a snapshot does not make preparation atomic | section 6: consistent-read guard (no open transaction, equal revision before and after), documented no-concurrent-mutation precondition |
| M5 | a face-pair test is not a bounded unit; phases without checkpoints | section 7: primitive units, every unit a checkpoint, all public phases including `check_preparation` |
| M6 | acceptance not concrete or prospective; derivative check not exact | section 10: acceptance manifest fixed before assessment runs; exact polynomial derivative check; verifier attacks; refusals per phase |
| m7 | linear-cost claim | section 4: linear only for clipping and accumulation; sort, nesting and embedding costs stated |

Revision 3 review (round 3):

| # | Finding | Response |
|---|---|---|
| B1 | zero-polynomial predicates are not always inconsistencies (collinear hole vertex) | section 5: zero is an outcome; only predicates proved nonzero refuse; ray-parity nesting with the half-open rule |
| M2 | verifier lacked a completeness check; contacts not recomputed | section 6 steps (3) and (5): exact inventory, contact lists recomputed; CH1c deletion attacks |
| M3 | identity fields beyond the snapshot digest unchecked; wheel hash relationship | section 6: snapshot digest content defined; model id, revision, checksum and file hashes recomputed; expected identity = the admitted wheel's per-file `RECORD` hashes |
| M4 | serialization, hashing and composite fields unbounded | section 7: per-element and 64 KiB-chunk units; proof byte and integer-digit limits before parsing |
| m5 | binary64 refusal contradicted exact doubles | section 3: judged on the returned bound; zero width always meets |
| m6 | projection axis ambiguous for negative normals | sections 5 and 6: drop the index maximising `|n[k]|`, lowest index on ties, sign from `n[k]` |

Revision 4 review (round 4): B1, M2, M4, m5 adequate. M3 (runtime fields of the artifact descriptor
unchecked) and m6 (projection sign when dropping the middle index) fixed in revision 5: the
verifier rebuilds the complete artifact descriptor; the orientation factor is `(-1)^k sign(n[k])`.
