# Certified hydrostatics: design (CH0, 2026-10-20)

Status: design for owner review; no source changed. Request: ANYstability,
`C:/Github/ANYstability/docs/ANYgeometry_CERTIFIED_HYDROSTATICS_HANDOFF.md` (20 October),
requirements GEO-01..GEO-11 in `ANYstability/docs/ANYgeometry_REQUIREMENTS.md`. Owner decision
2026-10-20: deliver certified point queries for planar shells first (CH1 + CH2), the curved B1
hull later (CH3), events/intervals after (CH4). The bow programme (step 4) waits.

## 1. Key facts this design rests on

- The consumer's reference vessel (recipe3: `HULL`, `PUMPROOM`, `V1..V4`, `WB1..WB8`) is planar:
  830 `Plane` faces, all straight edges, no holes, loops of at most 4 vertices, every solid a
  clean `closed_shell`, **every face loop exactly coplanar in its float vertices** and contained
  exactly in its stored `Plane` support (both checked with exact rational arithmetic,
  2026-10-20). Each solid is therefore an exact polyhedron.
- A rational polyhedron cut by a rational half-space is a rational polyhedron. Its volume and
  first volume moments are rational: they can be computed **exactly** (absolute bound 0).
- A planar polygon in a plane with rational normal `n` has an area `Q / |n|` with `Q` rational
  and `|n|^2` rational (generally irrational): waterplane area and moments are an exact
  rational divided by `|n|`. They are returned exactly in that form and as an outward-rounded
  interval of any requested width.
- Hence for planar shells the whole S2 quantity set is exactly computable, and a verifier can be
  an independent exact recomputation. No integration error, no tolerance, no plane nudging.

The current `hydrostatics()` (H1) returns quadrature estimates for curved and planar shells;
it stays as it is (a development API). The certified API is new and separate.

## 2. Supported domain (CH1) and refusals

A preparation accepts a model and an explicit face set (the displacement boundary selected by
the caller) and admits it only if all of these hold, each checked exactly:

1. every face's support is a `Plane` and every edge in its loops is straight;
2. every loop (outer and holes) is exactly coplanar in its stored float vertices, and lies in the
   face's support plane exactly or the support is replaced by the loop's exact plane (decision
   D-CH3);
3. the face set is a closed, consistently oriented shell (`closed_shell` clean) with positive
   exact volume (outward orientation; a negative volume means inverted orientation: refused);
4. the shell is not self-intersecting: no two non-adjacent faces intersect, decided exactly
   (planar polygon pairs, rational arithmetic);
5. every coordinate is finite.

Anything else is refused with a structured category (section 6). Curved faces are
`UNSUPPORTED_SURFACE` until CH3. Selecting which faces form the external buoyancy envelope
(and excluding frames, compartment walls, numerical lids) is the caller's explicit choice; the
preparation proves only that the selected faces form a valid closed solid. Each recipe3 solid
is its own preparation (the hull and each tank).

## 3. Frame, request and semantics

- **Frame**: the model's coordinates are the body frame; ANYgeometry applies no rotation. The
  consumer builds the half-space in body coordinates (`R[2] . r <= -origin_z` in its own
  convention) and passes it. Units are those of the model (metres for recipe3), stated in the
  preparation.
- **Half-space**: `n . r <= d` with `n` (3 finite floats, not all zero) and `d` (finite float)
  taken as exact rationals. No normalisation, no substitution: the queried plane is exactly the
  given one.
- **Waterplane basis**: the caller passes an origin `o` and two axes `e1`, `e2` (finite floats,
  exact rationals) with `u = e1 . (r - o)`, `v = e2 . (r - o)` for `r` in the section. They are
  used exactly as given: no orthonormalisation (the consumer's `R[0]`, `R[1]` are floats that are
  not exactly orthonormal; any normalisation would change the problem). Handedness: reported as
  the exact sign of `(e1 x e2) . n`.
- **Quantities** (consumer order):

  | Group | Quantities | Exact form |
  |---|---|---|
  | Volume | `V = int dV` | rational |
  | First volume moments about the body origin | `J = int (x, y, z) dV` | rational |
  | Waterplane area | `A = int dA` | rational / `|n|` |
  | First waterplane moments | `(int u dA, int v dA)` | rational / `|n|` |
  | Second waterplane moments | `(int u^2 dA, int v^2 dA, int u v dA)` | rational / `|n|` |

  `dA` is the true area element of the section plane (not a projection). Each quantity is
  returned as its exact form (rational numerator as an integer pair; `|n|^2` as a rational) and
  as a float interval `[lo, hi]` that contains it; the absolute bound is `hi - lo` (0 when the
  float is exact). Centroids, BM and `V * CB` are not returned as certified quantities (the
  consumer derives them from bounded raw quantities).
- **States**: `EMPTY` (no part of the solid below the plane: V = 0, no section), `FULL` (the whole
  solid below: V = solid volume, no section), `PARTIAL`. Absent quantities are `None` with the
  state saying why, never zero-filled.
- **Planes through features**: `V` and `J` are continuous in `d` and exact for every plane,
  including planes through vertices, along edges or containing whole faces. The waterplane
  section is not continuous where the plane contains a face of the shell (the consumer's upright
  WB1/WB2/WB7/WB8 main-deck query, GEO-06, is such a case if the deck lies in the plane). The
  result then carries `BOUNDARY_COINCIDENT` and both **one-sided** waterplane results,
  `below = lim (d -> d-)` and `above = lim (d -> d+)`, each exact, computed by symbolic
  perturbation of `d` (simulation of simplicity on the single parameter `d`: a vertex on the
  plane counts as strictly above for the `below` limit and strictly below for the `above`
  limit). When both limits agree the plain section is returned. This gives GEO-05/GEO-06 a
  defined, exact meaning instead of an exclusion.
- **Section topology** (GEO-10): the section (or each one-sided section) is returned as exact
  loops grouped into components with their holes (exact nesting), with counts, and with a stable
  identity per loop: the sorted tuple of the shell edge identities the loop crosses (and face
  identities for loops lying in a face). Identities are meaningful only within one preparation.

## 4. Algorithms

**Main computation (`query`).** For every face, clip each loop against the closed half-space
exactly (rational Sutherland-Hodgman, with the perturbation rule above when a one-sided result
is requested). Accumulate the signed tetrahedra of a fan of each clipped polygon with the body
origin: `V += det(a, b, c) / 6`, `J += det(a, b, c) (a + b + c) / 24`. The cut segments lie in
the plane; joined by exact endpoint equality (a cut point is the exact intersection of a shell
edge with the plane, so both faces sharing the edge produce the identical rational point) they
form the section loops, oriented by the outward normal `n`. The section (the cap) closes the
clipped solid: its tetrahedra are added to `V`, `J`. Waterplane quantities come from a fan
triangulation of the section loops: triangle area `= ((b - a) x (c - a)) . n / (2 |n|)`, and
polynomial moments by the exact vertex formulas for linear and quadratic integrands over a
triangle. Everything is Python `Fraction` arithmetic on exact inputs.

**Independent verifier (`verify`).** A separate module, written from this specification only
(not from the main implementation's code; see section 8), recomputes every quantity by a
different decomposition: volume and first moments by the divergence theorem with projected
face integrals (Mirtich's polyhedral mass properties, exact), the section by intersecting the
plane with each edge and ordering the cut points along each face's cut line (no clipping), its
area and moments by Green's theorem in the plane coordinates of the largest component of `n`.
It then requires exact equality with the claimed exact values and checks that every claimed
interval contains its value. Two different exact algorithms agreeing is the proof; the trust
assumptions are listed in section 7.

**Cost.** Linear in the face and edge count per query (recipe3 hull: 320 faces); rational
numbers stay small (float inputs, degree <= 4 products). Measured in CH1; an integer-scaled
variant replaces `Fraction` if needed (same exact results).

## 5. Public API (sketch; names final in CH1)

```python
policy = CertifiedHydrostaticsPolicy(max_work=..., interval_digits=...)        # bounded work
prepared = prepare_certified_hydrostatics(model, faces, units="m", policy=policy,
                                          cancellation_check=None)               # admit or refuse
query = HalfSpaceQuery(normal=(nx, ny, nz), offset=d,
                       waterplane_origin=(ox, oy, oz), waterplane_axes=(e1, e2),
                       absolute_accuracy={"V": 1e-9, ...} | None, one_sided=True)
result = query_certified_hydrostatics(prepared, query, cancellation_check=None)
report = verify_certified_hydrostatics(model, faces, query, result.proof)       # independent
```

- `prepare_*` returns an immutable `PreparedCertifiedHydrostatics`: model id, revision, the
  document checksum, the face ids, an exact snapshot of the face loops (rational vertices,
  orientations) and its digest, units, the domain verdict, the implementation identity and the
  policy.
- `query_*` returns `CertifiedHydrostaticsResult`: state, quantities (exact form + interval +
  absolute bound), one-sided results when applicable, section topology, accuracy achieved per
  quantity (`True` / refused, never relabelled), work counts, and `proof`: a canonical JSON
  document binding the implementation identity, the preparation digest, the exact query, the
  exact results and the intervals.
- `verify_*` takes the model (or its document), the face ids, the exact query and the proof,
  and returns a report or raises a refusal. It refuses a substituted model, face set, query,
  altered quantity or interval, a stale preparation (model revision or snapshot digest
  differs), a different implementation identity, and any unsupported claim.
- **Implementation identity**: the package version plus SHA-256 of the certified-hydrostatics
  modules as installed (read from the files at import), so a source change or a different wheel
  is a different identity even at the same version (the readiness review's point).
- **Invalidation**: a preparation is bound to `(model id, revision, document checksum, face
  set, snapshot digest, implementation identity, policy)`; a query against a changed model
  refuses `STALE_PREPARATION`; a proof from another identity refuses verification.

## 6. Refusals, accuracy, work and cancellation

`CertifiedHydrostaticsRefusal(GeometryError)` carries a `category` from a public enum, the
`phase` (`prepare`, `query`, `verify`), the exact request, model/preparation identity and the
original cause (`__cause__` preserved). Categories: `UNSUPPORTED_SURFACE`, `NOT_EXACTLY_PLANAR`,
`NOT_CLOSED`, `INVERTED_ORIENTATION`, `SELF_INTERSECTING`, `NONFINITE_INPUT`, `INVALID_QUERY`,
`STALE_PREPARATION`, `ACCURACY_UNACHIEVABLE`, `BUDGET_EXHAUSTED`, `CANCELLED`,
`VERIFICATION_FAILED` (with a sub-reason: model, face set, query, value, interval, identity).
Consumers never parse messages. No retries with a changed problem: a refusal is final for that
request.

Accuracy: `V` and `J` are exact (their float intervals are the two nearest floats or a single
float). The waterplane quantities' intervals can be narrowed to any requested width by integer
square-root bounds on `|n|` (`interval_digits`); a request below what the policy allows refuses
`ACCURACY_UNACHIEVABLE`. Work: counted in documented units (face-loop vertices clipped,
triangles accumulated, square-root refinement steps) and checked against `max_work` before each
face; `cancellation_check` is called between faces (each face's arithmetic is bounded and not
pre-emptible).

## 7. Trust assumptions (stated in the documentation)

The certificate holds for the exact rational problem defined by the stored float vertices and
the given float request, assuming: Python's `fractions.Fraction` and integer arithmetic are
exact; the two algorithms are correct implementations of their textbook formulas (two
independent implementations agreeing exactly makes a shared error unlikely, not impossible);
the domain checks of section 2 are correct (the verifier repeats them). It says nothing about
the physical vessel, the source drawings, tank communication or loading (consumer and domain
claims, GEO handoff section 4).

## 8. Events and intervals (preview of CH4)

For planar shells, the section topology changes only at planes through a vertex. For a sweep
in `d` at fixed `n` the events are exactly the vertex heights `n . p`; between consecutive
events the topology is constant and `V(d)` is an exact cubic polynomial (`A(d)` quadratic),
which supports exact interval statements (no sampling). For attitude paths the event condition
`n(theta) . p = d(theta)` per vertex gives exact or enclosed event locations for a parametric
path the consumer supplies. Delivered in CH4; the consumer keeps path, branch and stability
responsibilities.

## 9. CH1 slices and acceptance

| Slice | Output | Acceptance | Author |
|---|---|---|---|
| CH1a | preparation + domain checks + refusal enum + implementation identity | recipe3 hull and every tank prepare; each refusal category raised by a minimal case; the exact coplanarity and self-intersection checks fail on perturbed copies | Sonnet (spec here), Opus review |
| CH1b | `query`: exact clip, V, J, section loops and topology, waterplane quantities, one-sided results | closed forms: box, tetrahedron, triangular prism, L-shaped prism, a box with a through-hole (holes), any plane: V, J, A and waterplane moments exact; one-sided values at a coplanar face; empty/full; Q02 translations, rotations by exact rational matrices, moment-origin changes, reflection | Sonnet, Opus review |
| CH1c | independent verifier (separate module, from this spec only) | agrees exactly with CH1b on all CH1b cases and on recipe3 hull/tanks for a registered set of planes; refuses every substitution/alteration/staleness case | a separate Sonnet session given only this spec |
| CH1d | recipe3 runs: hull and WB tanks, upright and inclined, including the WB1/2/7/8 deck plane | all prepare and verify; the deck plane returns `BOUNDARY_COINCIDENT` with both one-sided results; timings recorded | Opus |
| CH2 | delivery folder: wheel + SHA-256, source commit, API/domain/proof docs, fixtures (B1 out of scope until CH3; recipe3 documents referenced, not copied), commands, requests/responses/proofs, verifier runs, refusals, work accounting | ANYstability's handoff section 5 inventory complete for planar shells; installed-wheel runs in an isolated environment | Sonnet, Opus check |

Every slice gets a Codex Sol review; the design gets a Codex Astra review first. Version: a
patch bump (owner rule) when CH2 builds the wheel.

## 10. Owner decisions (2026-10-20: all as recommended)

Decided: D-CH1 exact rationals + intervals with one-sided limits; D-CH2 basis used as given;
D-CH3 refuse `NOT_EXACTLY_PLANAR`; D-CH4 verifier from the specification by a separate session;
D-CH5 curved shells in CH3, refused until then.

- D-CH1: exact rational quantities with float intervals, one-sided waterplane limits by
  symbolic perturbation of `d` (recommended), rather than refusing planes through faces.
- D-CH2: the waterplane basis is used exactly as the caller gives it (not orthonormalised).
- D-CH3: a face whose loop is exactly coplanar but whose stored `Plane` support differs from the
  loop's exact plane: refuse `NOT_EXACTLY_PLANAR` (recommended; all 830 recipe3
  faces lie exactly in their supports) or use the loop's exact plane.
- D-CH4: the verifier written by a separate session from the specification only, as the
  independence measure; final independent admission remains ANYstability's.
- D-CH5: B1/curved shells follow in CH3 (certified enclosures by exact sub-patch subdivision);
  until then they refuse `UNSUPPORTED_SURFACE`.
