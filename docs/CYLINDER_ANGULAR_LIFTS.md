# Cylinder Angular Lifts (experimental singleton and batch producer)

`anygeometry.cylinder_angular_lifts` proves, with exact rational arithmetic
only, the authenticated angular lift of ONE whole straight boundary ruling of
ONE active Cylinder face. It is an experimental, singleton-scoped producer:
each query certifies a single face/edge occurrence, returns one immutable
`CylinderRulingLift` receipt, and establishes nothing about partitions,
materials, meshes, references, joints or source-current equivalence.

`anygeometry.cylinder_ruling_batch` extends the same proof to a complete
finite set of occurrences with ONE reusable batch query and binding
validator; see "Batch producer" below.

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
- Angles are published only as outward enclosures; the exact angle is never
  materialized. Existing public material-partition refusals are unchanged.
- Whole-face semantics (partitions, materials, meshes, references) remain
  PENDING and are deliberately not attempted here.

## Batch producer (`anygeometry.cylinder_ruling_batch`)

`query_cylinder_ruling_batch(model, selection, *, expected_revision=None,
cancellation_check=None, policy=None)` proves a complete finite set of actual
face/straight-ruling occurrences in ONE shared scope instead of N scalar
queries, and `validate_cylinder_ruling_batch_binding` authenticates the
result by complete live batch rederivation. The mathematical scope per
operand is exactly the scalar scope above; no whole-material,
source-current, reference, meshing or "1000" acceptance is claimed, and the
historical `CylinderAtlasPolicy` face-use/occurrence admission semantics are
untouched (the batch imposes no new face-count cap).

**Shared scope and cost.** All operand inputs are captured detached from the
live model BEFORE any cancellation callback (one raw frame capture per
distinct selected face, reused across that face's occurrences), and the
whole batch hashes the full document at most TWICE: once at entry, once at
final freshness. Binding validation adds one pre-check hash and then
rederives ONE complete batch - never per-ruling document hashes.

**One aggregate budget.** All items compile under a single
`cylinder_charts._Proof`: the recorded `CylinderAtlasPolicy` limits apply to
the counted arithmetic of every operand and are never reset per edge. The
proof's memoized enclosures (for example the shared `pi` enclosure and a
repeated ruling ray) replay their recorded charges on reuse, so cached exact
work is never laundered out of the accounting; per-item receipts carry the
CUMULATIVE aggregate counts at that item's completion, and the batch receipt
carries the final aggregate counts plus the aggregate policy limits.

**Determinism.** The selection is compiled in canonical (face ID, edge ID)
order regardless of caller order, so the same occurrence set always yields
the same receipt; a reversed selection yields the identical digest.
Malformed, empty or duplicate selections refuse typed before any capture.
Selections and each `(face_id, edge_id)` pair accept only plain finite tuples
or lists. Iterators, generators and container subclasses refuse before
consumption. This restricts input representation without a model-count ceiling.
Binding requires the complete canonical work-counter inventory for the batch
and each item; missing, extra and duplicate keys refuse as invalid results.

**Callback safety.** During callbacks the guards run only inexpensive
revision/busy checks - no per-surface rescan at every arithmetic callback
(which would be quadratic in the operand count). After the FINAL callback
the guards recheck the CURRENT raw frame pin of EVERY selected surface -
including the stored circumferential coefficient the document checksum does
not serialize, so a finally installed replacement object with equal
serialized fields and a different private circumferential refuses typed -
and then the full document checksum. Callbacks may temporarily mutate and
restore content; only the immutable entry evidence is published.

**Whole-batch refusals.** Unsupported operands (curved edges, off-carrier
rulings, wrong owners, inactive entities, non-Cylinder supports), aggregate
budget exhaustion (even where every standalone scalar would succeed),
cancellation, stale revision, busy models and callback-visible changes
refuse typed with NO partial accepted receipts and NO model mutation.

**Binding.** `validate_cylinder_ruling_batch_binding` pins the caller's
batch digest BEFORE any callback, rederives the COMPLETE batch live under
the recorded aggregate policy, and accepts only a fresh batch whose digest
equals the pinned digest while the caller's receipt stayed unchanged -
never an object a callback repaired in flight. Because scalar digests bind
`work_counts`, batch items (cumulative counts, extra batch cancellation
checks) can never masquerade as standalone scalar receipts, and a standalone
scalar receipt is not batch item evidence.
