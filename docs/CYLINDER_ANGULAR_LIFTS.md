# Cylinder Angular Lifts (experimental singleton producer)

`anygeometry.cylinder_angular_lifts` proves, with exact rational arithmetic
only, the authenticated angular lift of ONE whole straight boundary ruling of
ONE active Cylinder face. It is an experimental, singleton-scoped producer:
each query certifies a single face/edge occurrence, returns one immutable
`CylinderRulingLift` receipt, and establishes nothing about partitions,
materials, meshes, references, joints or source-current equivalence.

## What is proved per query

For one active Cylinder face and one of its actual boundary edge
occurrences, entirely on the RAW stored live surface coefficients (no
decode, no reconstruction, no normalization round trip):

1. **Frame rank and subturn.** The chart frame
   `[radius*radial, radius*circumferential, height*axis]` has a nonzero
   exact `Fraction` determinant, and `abs(sweep) < Fraction(math.tau)` is a
   strict subturn (the float-tau boundary is conservative below the true
   `2*pi`), so the chart is injective on the support window. No constant
   Jacobian, area or orthonormality claim is made or needed.
2. **Whole-ruling carrier membership.** A straight boundary edge
   (`Straight` truth via `freeze_edge`) solves exactly to CONSTANT radial
   coefficients `(qx, qy)` with `qx*qx + qy*qy == 1` and distinct `v`
   endpoints in `[0, 1]`. The solve is linear, so the edge's complete
   analytic line lies on the captured carrier; off-carrier rulings (for
   example a rounded 45-degree footpoint, or any footpoint authored against
   a non-cardinal stored radial whose normalized floats make the point
   off-carrier by one rounding), curved edges, singular or full-turn
   carriers, explicit parameterizations, inactive entities and wrong-owner
   occurrences refuse without projection or snapping.
3. **Exact angular support membership.** The exact angle of `(qx, qy)` is
   represented as a canonical primitive rational ray plus an integer winding
   (`atan2(y, x)` in `[0, 2*pi) + 2*pi*winding`; the angle itself is
   generally irrational and is never stored as a number). Membership in the
   support window `[start, start + sweep]` (either orientation) is
   certified for exactly one winding by counted outward-arithmetic interval
   comparison. Candidate windings are enclosed by dividing
   `[low - theta.hi, high - theta.lo]` by the positive `2*pi` enclosure with
   the existing `cylinder_charts._Proof.div`; ceil/floor of that outward
   enclosure is valid for negative offsets too. Every tested period is
   charged counted arithmetic, so the explicit `CylinderAtlasPolicy` budget
   alone controls the work; unresolved boundary equality, ambiguous period
   lifts, angles outside the window and exhausted budgets refuse typed and
   never produce a false outside claim. Negative sweeps and far periodic
   gauges certify through nonzero windings, never clamped.

## Entry-evidence discipline (callback safety)

Every proof input - the face/edge occurrence, the raw frame, the frozen
`LinePath` endpoints and the edge vertices - is captured DETACHED from the
live model BEFORE any cancellation callback runs, and the proof compiles
only that immutable entry evidence. Consequences:

- A callback that mutates the model and later restores it can never publish
  temporary geometry: the receipt always reflects the entry evidence, which
  equals the restored committed state.
- After EVERY cancellation callback invocation the guards recheck the
  revision and a raw frame PIN that includes the stored circumferential
  coefficient of the currently installed face surface, including replacement
  surfaces. `Cylinder` serialization excludes `surface._circumferential`
  even though `evaluate` uses it, so the ordinary document checksum alone
  cannot see a private late mutation of that coefficient; the pin refuses it
  typed (`CHANGED_MODEL`).
- After the FINAL callback the full document checksum is rechecked as well,
  so a callback that changed model content without bumping the revision
  still refuses.

## Binding validation

`validate_cylinder_straight_ruling_lift_binding` pins the entry receipt
digest and the query parameters BEFORE any callback runs, rederives the
whole proof live (including the FINAL-callback freshness check), requires
the original receipt to be unchanged, and accepts only a fresh receipt whose
digest equals the pinned entry digest - never a caller object a callback
could have repaired in flight. Stale, foreign, forged and tampered receipts
refuse typed; a refusal never returns a partial receipt.

## Cost profile and pending scope

- One scalar query hashes the full document twice (entry and post-FINAL
  freshness) and the validator rederives the whole proof once more. This is
  the accepted cost of this experimental producer and is NOT a large-model
  route; it does not scale to looping thousands of scalar queries and must
  not be used that way.
- Batch sharing (one document hash shared across many lifts) and whole-face
  semantics (partitions, materials, meshes, references) remain PENDING and
  are deliberately not attempted here. No batching architecture exists in
  this module.
- Angles are published only as outward enclosures; the exact angle is never
  materialized. Existing public material-partition refusals are unchanged.
