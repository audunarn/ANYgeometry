# Cylinder projected-boundary correspondence

`src/anygeometry/cylinder_boundary_correspondence.py` provides
`query_prepared_cylinder_boundary_correspondence` and
`validate_prepared_cylinder_boundary_correspondence_binding`: a
**boundary-only diagnostic certificate** comparing the ACTUAL authored and
current trim boundary definitions of one authored Cylinder face through one
explicitly defined projection.

This is not material conservation, interior partition coverage, reference
completeness or meshing permission. No such proof flags exist on the result,
and a certified source/current boundary identity does not imply interior
partition coverage. The existing material contract
(`docs/CYLINDER_ATLAS_CONTRACT.md`) is unchanged, as is the direct atlas
256 face-use cap.

## The projection

The certificate uses the **authored rational-frame projection**, defined ONLY
from the serialized `surface` components of the authenticated authored face
definition (origin, axis, radial_direction, radius, height, start_angle,
sweep_angle). Every stored float component is read as an exact binary
rational. The frame is never normalized, decoded, reconstructed or replaced
by the live descendant supports. The serialized Cylinder stores **no
circumferential coefficient**: the projection recomputes
`axis x radial_direction` exactly in rational arithmetic as part of its own
definition. That recomputed cross is therefore not a stored native raw-frame
coefficient, and this projection is **not** the native
`Cylinder.evaluate`/`local_uv`/atlas chart semantics — no equivalence with
them is claimed. For a world point `p` with `w = p - origin`:

- axial coordinate `a = w . axis`
- radial coordinates `x = w . radial_direction`,
  `y = w . (axis x radial_direction)` (exact rational cross product)

The projected curve of a boundary edge is `(theta, a)` with
`theta = atan2(y, x)`. The projection is **finite** only where the certified
exact rational minimum of `x^2 + y^2` over the parameter interval is strictly
positive; a zero minimum refuses the root (`radial zero`).

Branch and seam identity: the angle branch is the principal branch
`(-pi, pi]` with the seam on the negative radial ray. Per straight root the
certificate records the continuous lift through the exact `(cross, dot)`
pair of the endpoint radial vectors plus endpoint radial coordinates —
`atan2` is never evaluated. `constant_angle` is true exactly when
`cross == 0 and dot > 0` (the segment then stays on that ray). A straight
segment that crosses the seam ray is flagged `seam_crossed`; a ruling lying
exactly on the seam ray is also flagged (its principal value is `pi`).

**No nominal generator angle is substituted.** A Straight trim
`(1, eps*t, t)` genuinely projects to the nonconstant `(atan(eps*t), t)`; the
certificate proves nonconstancy exactly (nonzero lift cross means
`theta(1) != theta(0)`), which refutes any constant start-angle carrier
substitution. Refuting the *linear endpoint polygon* would require
transcendental bounds and is explicitly not claimed.

## Families

- **Straight**: the native affine parameter function through the two stored
  vertices, with exact rational controls. All comparisons are exact rational
  polynomial decisions; no witness sample is truth and no constant Jacobian
  is assumed.
- **Arc**: the native three-point arc through the stored start/via/end
  vertices. The native parameter function is transcendental and is **never
  evaluated**. An unsplit arc is decided by exact native definition identity
  (same edge ID, curve type, start/via/end vertex IDs and exactly equal
  rational vertex positions) PLUS an exact **whole-circle finiteness
  certificate**: the whole native arc lies on the circumcircle of its three
  authored points, which lies in their exact plane, and the projection is
  undefined exactly on the radial-null line
  `{origin + t*(radial x (axis x radial))}` through the authored origin.
  Exact plane/line/circle algebra (no witness samples) decides whether the
  whole circle avoids that line: the line misses the circle iff it is
  parallel to and disjoint from the plane, or meets the plane at a point
  that is not ON the circle, or lies in the plane at squared distance
  greater than the squared radius from the circumcenter. `exact` requires
  BOTH identity and this certificate (`finiteness_certified` is true, with
  the circumcenter, squared radius, case and exact separation recorded).
  When the whole circle cannot certify avoidance — the adjudicated
  counterexample circle through `(2,0,0),(1,1,0),(1,-1,0)` is tangent to the
  radial-null line at the radial origin, separation exactly 0 — the
  projected correspondence is **refused** while the native definition
  identity is retained as separate evidence
  (`native_identity_certified` is true). A partial arc might still avoid the
  line, but that is explicitly not claimed.
- **Spline and other curve types**: refused (`unsupported family`).

## Outcomes

Per authored boundary root (loop and holes) and aggregated per face:

- `exact` — oriented projected boundary equality with family-specific evidence.
  For a straight root,
  every authenticated descendant child's native function equals the exact
  restriction of the authored function to a directed source interval, and the
  intervals exactly partition `[0, 1]` (gaps, duplicates, overlaps and
  degenerate spans refuse). For an arc root, unsplit native definition
  identity AND the exact whole-circle finiteness certificate both hold when
  unsplit. Split arcs instead require exact common-circle definitions,
  complete directed geometric subarc tiling, global order inside the authored
  span and finiteness of each child. Split Arc source-parameter correspondence
  is explicitly unqualified; numeric split intervals are provenance only.
  Exactness is oriented: each straight descendant's directed source interval
  sign combined with its current EdgeUse direction must equal the original
  authored EdgeUse direction, and an unsplit arc must keep the authored use
  direction. Native parameter identity alone is not oriented boundary
  equality — the tiling check drops direction and cannot prove orientation —
  so a reversed traversal refuses (`descendant orientation reversed`).
- `certified_discrepancy` — the projected curves are certified different with
  exact rational bounds: the exact sup of the axial difference on `[0, 1]`,
  and/or a rational witness parameter with an exact rational LOWER bound on
  the angular difference (`|sin(delta theta)| = |cross| / (rho1*rho2)`, or
  `opposite_directions` certifying `|delta theta| = pi` at the minimizing
  station, reported with the conservative rational bound `3 < pi`). A
  whole-curve angular UPPER bound is explicitly not certified.
- `refused` — missing authentication or unsupported mathematics, with the
  reason recorded: `ancestry unavailable`, `ancestry incomplete/conflict`,
  `descendant not on current boundary`, `descendant orientation reversed`,
  `radial zero on authored/current
  straight`, `arc ancestry unavailable`, `arc finiteness uncertified:
  circle tangent to / meets the radial-null line` (native identity kept as
  separate evidence), `arc finiteness uncertified: collinear arc points` /
  `degenerate authored radial frame`, `unsupported family`, or face-level
  `incomplete descendant boundary scope` (a current boundary edge of the
  descendant scope that no authenticated root accounts for).

Paired interior incidences between descendants are recorded but not
compared; the certificate concerns the boundary only. No loop winding,
closure, partition, material, reference or mesh claim is made: per-curve
lift evidence is limited to Straight curves. Arc evidence is unsplit native
identity or split geometric-image tiling, never a numeric parameter map or
a whole-face winding certificate.

## Authentication

The producer consumes existing owners only — never caller-provided trusted
arrays: the immutable authored face definition
(`prepared_face_preimages.query_prepared_authored_face_definition`), the
complete descendant scope (`query_prepared_face_preimages`), and the
polynomial split ancestry (`edge_subcurve_preimages`, records and unified
aliases). Ancestor model/revision/checksum and exact controls must match the
authored definition. Busy models, stale revisions, changed checksums and
incomplete coverage are refused by those owners.

Detached entry snapshot: all live current-model inputs — the descendant face
loops and incidence and the active edge/Arc vertex definitions — are detached
into immutable snapshots BEFORE the first callback, and the derivation uses
only those snapshots plus the owner receipts; captured values never alias
live arrays or mutable receipt objects. A callback that temporarily mutates
and later restores the live model therefore either causes a typed owner
refusal or yields exactly the unpolluted entry result, never a contaminated
certificate. Cancellation is checked at every stage; after the last
callback-bearing stage the owner receipts, the revision and the authored
document identity are revalidated.

`validate_..._binding` pins the request target/revision and the caller
result's `definition_checksum` at entry, BEFORE any callback or
callback-bearing rederivation. After rederivation it requires the current
input checksum to still equal the pinned entry checksum (a result repaired
or mutated in place during callbacks refuses) and the owner's rederived
checksum to equal the same pinned entry checksum, so an initially forged
result cannot be repaired into acceptance even when its final fields are
legitimate.

## Arc split lineage: actionable integration requirement

`edge_subcurve_preimages` now carries `arc_records` and `arc_alias_records`,
recorded only by the actual detached preparation wrapper. Each immutable
record binds the original three-point definition, actual current child,
recorded local split parameter and composed numeric interval. These intervals
authenticate history, not an exact native parameter restriction.

Record-level `exact` proves common plane/circle membership only. The boundary
consumer separately proves complete geometric tiling: oriented selected
subarcs, shared endpoint positions, strict global ordering from the original
start, complete original span and no unused children. Local neighbour tests
alone can admit an extra revolution on a major arc and are insufficient.

Rounded refitted circles can instead be `enclosed` within the unchanged
recorded tolerance, using plane-aware point distances and conservative
whole-circle bounds. Such roots refuse as `arc split enclosed within
tolerance, not exact`; enclosure is neither exact equality nor proof of
nonzero discrepancy. Unknown or lost history still refuses. No Arc record
qualifies native parameter remapping, complete material or references.

## Test evidence

`tests/test_cylinder_boundary_correspondence.py` exercises the production
producer on generated rounded 12-panel definitions (no nominal snapping),
unsplit exact identity, the actual nonconstant projected Straight
counterexample, on-seam and seam-crossing lift identity, recorded rounded Arc
split refusal, unsupported families, reversed (reflected) uses, the
immutable raw frame, cancellation/mutating callbacks, temporary
mutation-restored callbacks (unpolluted entry result) and unrestored
mutations (typed refusal), entry-pinned validator forgery repair and
in-place input mutation refusals, directed-use reversal refusals (straight
and arc, with consistent reflected traversals as positive controls), stale
revisions and malformed bindings, plus the exact-rational comparison engine
(discrepancy bounds, tiling, helpers). See
`reports/cylinder-projected-correspondence-mistral-01/` for run logs and
JUnit output.

`tests/test_arc_split_preimages.py` adds independent 450/630-degree false
tiling counterexamples, valid major arcs and reversed storage, overlapping
square-root interval bounds, plane displacement, small coplanar refits and
actual repeated split recording. The bounded source evidence is in
`reports/cylinder-arc-ancestry-mistral-01/`: initial 89 passes/one failure,
then 11 affected passes after repair. Failed evidence is retained. This is
focused development verification, not hosted/installed qualification or
large-model mesh acceptance.
