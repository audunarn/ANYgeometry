# Performance architecture

Correctness and scalable cost are co-equal kernel requirements.

## Local mutation

Ordinary edits journal only changed records and their dependency closure.
Reverse incidence finds dependent edges, faces, sheets, members, attachments,
and junctions. Entity versions invalidate only relevant caches. The dynamic
AABB tree receives logarithmic insert/remove/update operations for changed
bounds. Exceptional rollback invalidates provisional derived state without a
full-model copy.

Deterministic batch APIs amortize transaction validation and Python dispatch
for points, straight edges, faces, members, bounds, and vectorized curve and
surface evaluation.

## Query broad phase

Spatial queries use conservative exact entity AABBs in a deterministic
height-balanced tree. Padded internal bounds cannot leak false returned hits;
exact leaf bounds are checked before a candidate is returned. Candidate keys
and pair results are sorted independently of tree traversal.

Strict audit, changed-region audit, overlap discovery, and closest-entity
queries use the maintained index. Explicit precomputed candidate sets are
accepted for downstream incremental workflows. No public default overlap path
starts with `n*(n-1)/2` pairs.

## Narrow phase and tolerances

Fixed-size analytical predicates are allocation-light. Curved numerical work
uses local feature scale, verified residuals, deterministic bounded iteration,
and vectorized NumPy evaluation. Unqualified work fails closed rather than
increasing sampling until a plausible answer appears.

## Serialization and features

Serialization traverses immutable records once, validates referential and
geometric invariants, canonicalizes deterministic payloads, and computes one
document checksum. Feature materialization checksums traverse only resolved
feature closure; they never serialize the complete model per feature.

## Diagnostics and acceptance

Benchmarks report wall time, peak memory, visited entities, broad-phase
candidates, narrow-phase tests, and spatial updates for plate grids, member
lattices, mixed panels, cylinders, local edits, audit, intersection query,
imprint, serialization, migration, feature checksums, and long lineage.

Normal tests assert algorithmic work counts rather than unstable wall-clock
limits. Sparse-model candidate growth must remain materially subquadratic;
doubling a sparse case must not approximately quadruple narrow-phase tests.
Any construction/query regression above ten percent or stored-topology memory
growth above twenty-five percent requires measured correctness justification
in the release report.
