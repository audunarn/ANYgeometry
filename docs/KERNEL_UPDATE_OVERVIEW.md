# ANYgeometry strict-kernel update overview

This document is the shared architecture and wave contract for the
`codex/strict-kernel-update` implementation.  It is deliberately concise
enough to stay useful during integration.  When an implementation decision
changes, the lead agent updates this document before dependent work proceeds.

## Authority and scope

- Repository: `audunarn/ANYgeometry`
- Authoritative base: local `kernel_update`
- Base commit: `19f16746ceee76838b7df9d08a95028699e3a738`
- Working branch: `codex/strict-kernel-update`
- `origin/kernel_update` did not exist when Wave 0 fetched all remotes.
- `main` is not substituted as the task base, even though it currently points
  to the same commit.
- ANYmesher, ANYfem, ANYstructure, ANYsolver, and sibling repositories are
  outside the write scope.

ANYgeometry owns neutral structural geometry, topology, persistent structural
member identity, ownership/incidence, qualified geometric relationships,
strict audit, changed-region information, and deterministic geometry
documents.  Meshing, FE beam/shell elements, sections, thicknesses, materials,
loads, supports, contacts, solver state, and results remain downstream.

## Wave 0 baseline

The base is ANYgeometry `0.1.0`, geometry schema 2, feature-history schema 1,
on Python 3.13.9 / Windows 11 with NumPy 2.4.3 and Shapely available.

The tracked baseline collected 132 tests: 131 passed and one failed because
`pyproject.toml` declares `src/anygeometry/py.typed`, but the marker is not in
the base commit.  The original user checkout contains that marker as an
untracked file; it is treated as a baseline finding, not silently imported as
branch state.  The CLI smoke workflow passed when run with `src` on
`PYTHONPATH`.  Source and wheel builds succeeded, and both artifacts passed
`twine check`.

The current architecture is a useful structural-surface modelling layer:

- `model.py` owns mutable vertex/edge/face dictionaries, construction,
  evaluation, lineage, groups, tags, snapshots, and many edits.
- `entities.py` has only `Vertex`, `Edge`, `Face`, `OrientedEdge`, and the
  model-unbound `EntityRef(kind, id)`.
- `operations.py`, `editing.py`, `intersections.py`, and `overlaps.py` mutate
  model internals directly and obtain atomicity through whole-model snapshots.
- `features.py` rebuilds and validates complete models and serializes the full
  model repeatedly while computing per-feature checksums.
- `serialization.py` writes schema 2 without model identity, revision,
  tolerance/coordinate metadata, structural ownership, or document checksum.
- Structural generators identify beams/stiffeners/girders only through edge
  groups.  A physical member has no persistent identity across subdivision.
- No maintained reverse-incidence or spatial index exists.
- General global duplicate, T-junction, crossing, overlap, and nonconformity
  audit does not exist.

## Release invariants

The integrated kernel enforces these invariants:

1. Every successful public mutation returns a valid committed state.
2. A failed public mutation has no externally observable topology, semantic,
   cache, index, identity, or revision effect.
3. Committed IDs are monotonic per kind and are never reused for a different
   object.  Gaps are valid.  Public undo never lowers a high-water mark.
4. Public cross-package handles are bound to a persistent model UUID.
   Wrong-model handles fail explicitly.  Legacy unbound `EntityRef` values are
   accepted only by explicit local compatibility paths and are normalized.
5. Public stores, coordinate arrays, loops, use lists, and metadata are
   read-only.  All internal mutation passes through the model transaction
   owner.
6. Query operations never mutate.  Mutation based on a query requires a
   verified result and an explicit overlap/imprint policy.
7. Coincident geometry is not silently welded.  Intentional coincidence is
   represented by separate-part ownership or an explicit relationship.
8. An unclassified broad-phase candidate makes a strict audit fail closed.
9. Translation of the complete model does not change a local geometric
   classification.  Tolerance scaling uses participating feature extent, not
   distance from the global origin.
10. Face subdivision preserves the authoritative support surface and alters
    trims/uses.  An approximate mapping cannot silently replace it.
11. Member identity survives axis-edge subdivision.  One geometric edge may
    have multiple explicit shell-loop and member-axis uses.
12. Local edit cost is governed primarily by the changed dependency closure.
    Full strict audit uses broad phase and never starts with naive all-pairs.
13. Serialization is deterministic, model-bound, checksummed, migration-aware,
    duplicate-record rejecting, and validated before certified output.

## Identity and record model

Internal algorithms use compact `(kind, integer_id)` keys.  The model UUID is
stored once on `GeometryModel` and serialized once in the document.

Public identity is:

```text
EntityHandle(model_id, kind, id)
```

The legacy `EntityRef(kind, id)` remains a local compatibility value.  New
cross-model APIs return or accept `EntityHandle`.  Equality of two handles is
model-aware.  Resolution produces a typed `Resolution` with `ACTIVE`,
`REPLACED`, `DELETED`, `UNKNOWN`, `WRONG_MODEL`, `SUPPRESSED`, or `BLOCKED`.

IDs have independent high-water marks for geometry and structural topology.
Deserialization verifies that each high-water value is greater than every
persisted identifier of that kind.  A document-exact internal copy may retain
identity; an ordinary public clone receives a new model UUID.

Geometry records become externally immutable.  NumPy arrays exposed by value
objects are marked read-only.  Mapping metadata is copied into immutable
views.  Internal updates replace records through journal-aware owner methods.
This prevents a stale spatial index or revision from being created through a
field assignment outside a transaction.

## Transaction and ChangeSet design

`GeometryModel.transaction()` provides nested, delta-journalled transactions.
Nested public operations join the outer transaction.  The journal captures an
original record only on first write and separately records staged additions,
removals, structural uses, ownership, lineage, groups/tags, adjacency, cache,
and spatial-index deltas.

Outer commit:

- performs changed-closure validation;
- applies deterministic index/cache invalidation;
- increments model revision exactly once;
- produces one deterministic `ChangeSet`;
- notifies optional downstream invalidation hooks without importing a
  downstream package.

Rollback restores only journalled records and semantic entries.  Provisional
IDs may become gaps; allocator high-water marks never move backwards.  Public
snapshot compatibility is retained only as an explicit expensive document
snapshot and cannot rewind ID identity.

`ChangeSet` records revision before/after; added, removed, and modified keys;
replacement lineage; ownership/member/attachment/group/tag changes; affected
AABBs; cache invalidations; and spatial-index update counts.

## Topology and ownership model

Geometry definitions remain lightweight (`Straight`, `Arc`, `Spline`, and
supported surfaces).  Connectivity and semantic ownership are explicit:

```text
Part
 +- Sheet -> FaceUse -> Face -> Coedge/EdgeUse -> Edge -> Vertex
 `- Member -> MemberEdgeUse --------------------^

Attachment  (member/face or member/edge incidence with parameters)
Junction    (typed multi-member and/or member-sheet connection)
```

- A `Part` owns zero or more sheets and members and distinguishes intentional
  assembly separation.
- A `Sheet` owns oriented `FaceUse` records.  Sheet validation checks face
  uniqueness, orientation, boundary/manifold rules, and permitted
  non-manifold intersection relations.
- A persistent `Coedge` belongs to one face loop and records edge orientation.
  An edge may have any number of radial uses.
- A `Member` is a structural member, not an FE beam element.  It owns an
  ordered axis composed of `MemberEdgeUse` records with orientation and parent
  parameter ranges.  Edge subdivision rewrites the member chain without
  changing member identity.
- An `Attachment` records qualified member-on-face, member-on-boundary,
  member-through-face, or endpoint attachment semantics and parameters on both
  parents.
- A `Junction` records endpoint, crossing, overlap, or multi-way connection
  intent.  Mere geometric coincidence is not automatically a connection.

Compatibility face loops remain available while persistent coedges become the
authoritative incidence.  Existing schema-1/2 documents migrate into a
default part/sheet only where ownership can be inferred without inventing
physical member intent.

## Tolerance and coordinate policy

The model owns and serializes a `TolerancePolicy` with distinct positive,
finite values for computational length, intentional merge/heal length,
angular, parameter, area, and surface-residual tolerances.  It also records
units, a local modelling origin, and an optional external/CRS transform.

Predicates derive an effective tolerance from the policy plus the extent,
length, radius, or bounding box of the participating features.  No predicate
uses `norm(global_point)` as scale.  Changing the document's external origin
must leave local classifications unchanged.

## Spatial, incidence, and cache strategy

The model maintains deterministic reverse incidence for vertex-to-edge,
edge-to-face/coedge, face-to-sheet, and edge-to-member-use queries.

A lightweight deterministic dynamic AABB tree supplies incremental insert,
remove, update, and overlap queries for vertices, edges, and faces.  Curved
bounds are conservative.  Query results are sorted by entity key, independent
of tree traversal.  Diagnostics expose candidates visited, narrow-phase tests,
and index updates.

Entity versions drive caches for edge length/frame/AABB, face trim/AABB,
member accumulated length/ranges, replacement resolution, and adjacency.
Rollback restores versions and invalidates any value touched by the journal.

## Typed intersection algebra

Qualified results distinguish at least:

```text
DISJOINT
TOUCH_POINT
TANGENT
CROSS
OVERLAP_CURVE
OVERLAP_REGION
COINCIDENT
UNCLASSIFIED
```

Every component includes parent parameter values/ranges, witnesses, and
`EXACT`, `VERIFIED_APPROXIMATE`, or `UNVERIFIED` quality.  Primitive
line/line, line/plane, plane/plane, and line/cylinder compatibility functions
remain, but new qualified functions do not collapse parallel, contained, and
coincident cases to `None`.

Face-line clipping returns all material intervals and subtracts holes; it does
not take only the minimum and maximum boundary hit.  Planar clipping uses the
qualified Shapely backend.  Numerical curves require maximum-residual checks
against both parents.  Unsupported or uncertain narrow phase yields
`UNCLASSIFIED` and blocks strict mutation/audit.

Mutation policy is explicit: `REJECT`, `REUSE_EXISTING`, `WELD`, `IMPRINT`, or
`KEEP_SEPARATE_PART`.  Queries never choose one implicitly.

## Strict audit architecture

`strict_audit()` is a deterministic complete-model qualification separate
from incremental commit validation.  It uses AABB broad phase and exact or
verified narrow phase for:

- vertex coincidence;
- vertex-in-edge T-junctions;
- edge crossing, duplicate, reversed duplicate, and collinear overlap;
- member-member crossing/overlap and declared junction consistency;
- member-face embedding, boundary coincidence, crossing, and attachment;
- face-face crossing, coplanar overlap/containment/coincidence;
- sheet orientation/manifoldness and nonconformal interfaces;
- slivers, unresolved lineage, unowned/doubly owned structural uses;
- unclassified candidates.

Intentional cross-part coincidence is reported but is not an accidental weld.
The report has stable ordering, codes, severity, handles, witnesses,
classification, candidate/narrow-phase counts, and a clean/certifiable flag.

Incremental commit validation operates only on changed entities, their reverse
dependency closure, and nearby spatial candidates.  Certified serialization
and future mesher handoff require a clean full strict audit.

## Surface and transform rules

Faces permanently separate support surface from trim loops and optional
parameterization.  Splitting a plane, cylinder, cone, or ruled face retains
that exact support whenever the children lie on it.  Coons may be a mapping,
not a silent replacement support.

Rigid transforms, reflection, and uniform scale are supported for circular
geometry.  Singular transforms are rejected.  Anisotropic scale and shear are
rejected when the affected closure contains circular arcs/cylinders/cones
until ellipse/NURBS support exists.  Every transform is atomic and validates
the actual affected closure.

## Serialization and migration

The next document schema stores:

- required schema/version;
- model UUID and revision;
- units, local origin, optional coordinate transform, tolerance policy;
- all allocator high-water marks;
- geometry plus structural topology/uses/relationships;
- deterministic groups, tags, provenance, and replacement lineage;
- feature history and semantic output keys;
- namespaced extensions;
- SHA-256 checksum over the canonical document payload.

Schema-1/2 loading is a one-way migration and records migration metadata.
Schema-current loading rejects missing required fields, duplicate identifiers,
duplicate tag/history records, invalid high-water marks, unexpected core
fields, checksum mismatch, and invalid topology.  Ordinary serialization
validates topology; certified serialization additionally requires a clean
strict audit.

Feature materialization checksums operate on the selected closure directly,
not by serializing the complete model per feature.  Mapped face corner values
are loop positions and must be translated to actual corner vertex IDs before
hashing.  Important feature outputs use semantic names; allocation-order names
remain compatibility aliases where required.

## Performance qualification

Correctness and performance have equal release priority.  Benchmarks cover:

- structured plate grids (10,000 faces and a practical larger case);
- a 10,000-member beam lattice;
- a mixed stiffened panel;
- cylinder with axial/ring members;
- duplicate-heavy and crossing-heavy models;
- a local edit in a large mixed model;
- long replacement lineage;
- construction, query, audit, serialization, memory, candidates, narrow-phase
  tests, and index updates.

Acceptance is based on measured scaling rather than one absolute machine
time.  A local point/member edit must not visit or copy the complete unrelated
model.  Audit candidate growth on structured sparse models must be near-linear
and materially below `n*(n-1)/2`.  Performance diagnostics are retained in the
final report.

## Module ownership and integration waves

High-conflict ownership is serialized through the lead agent:

- `model.py`, `entities.py`, `serialization.py`, `features.py`, and package
  exports have one integration owner at a time.
- New identity/transaction, topology, spatial/audit, and predicate modules may
  be implemented in parallel against the contracts above.
- Direct mutations in editing/operations/intersections/overlaps/generators are
  migrated only after journal-aware owner methods exist.

Waves:

0. Baseline, profiling, this overview, and representative benchmarks.
1. Model identity, immutable public access, delta transactions, revisions,
   ChangeSet, reverse incidence, and incremental validation.
2. Part/Sheet/Member/use/attachment/junction storage and subdivision rules.
3. Tolerance policy, typed predicates/intersections, multi-interval clipping,
   safe transforms, and support-surface preservation.
4. Incremental AABB tree, strict audit, schema migration/checksum, and feature
   checksum/history fixes.
5. Adversarial/property/metamorphic/rollback/mixed-model tests, performance
   qualification, documentation, packaging, independent review, and report.

## Test strategy

Each wave adds focused unit tests plus integration tests.  Required hostile
cases include wrong-model handles, ID non-reuse after failed edit/undo,
mutation attempts through public views, nested rollback, cache/index rollback,
large translated coordinates, concave and hole clipping, multiple intersection
components, coincident separate parts, overlapping member axes, member split
identity, member at a plate intersection, non-manifold sheet uses, anisotropic
arc transform rejection, checksum corruption, duplicate serialized records,
schema migration, deterministic ordering, false-clean audit attempts, and
candidate-count scaling.

## Explicitly deferred

- FE shell or beam element generation and numbering
- meshing, seeding, size fields, smoothing, or transitions
- materials, sections, thicknesses, releases, loads, supports, contact, solver
  data, or results
- general solid CAD, booleans, NURBS/ellipse implementation
- edits to any sibling repository
