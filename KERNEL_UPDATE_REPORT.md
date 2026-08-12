# ANYgeometry strict-kernel update report

Status: implementation and release qualification complete.

This report records the mixed plate-and-beam geometry-kernel update based on
`kernel_update`. It covers neutral geometry and structural bookkeeping only.
ANYmesher, ANYfem, ANYstructure, ANYsolver, and sibling repositories were not
modified.

## Branch and baseline

- Authoritative task branch: local `kernel_update`
- Resolved base SHA: `19f16746ceee76838b7df9d08a95028699e3a738`
- Remote resolution: `origin/kernel_update` did not exist after fetching all
  remotes; the local branch was used exactly as required. `main` was never
  substituted.
- Working branch: `codex/strict-kernel-update`
- Original checkout: not modified
- Baseline platform: Windows 11, Python 3.13.9, NumPy 2.4.3, Shapely available
- Baseline release: ANYgeometry 0.1.0, geometry schema 2, feature-history
  schema 1

The base collected 132 tests: 131 passed and one failed because the tracked
branch omitted the `py.typed` marker declared by its package metadata. The
baseline CLI, source distribution, wheel build, and `twine check` otherwise
passed.

## Delivered kernel

### Identity, ownership, and public state

- `EntityHandle(model_id, kind, id)` is the model-bound public identity.
  Wrong-model, active, replaced, deleted, and unknown resolution outcomes are
  explicit. The model-unbound `EntityRef(kind, id)` remains for local
  compatibility.
- Geometry and structural IDs use independent monotonic high-water marks.
  Rollback, undo-compatible snapshot restoration, and feature regeneration do
  not reuse retired IDs.
- Vertex, edge, face, surface, structural, group, tag, and coordinate records
  are read-only at the public boundary. Mutations pass through the model owner.
- Model UUID, revision, tolerance, units, local origin, and external coordinate
  transform are serialized document state. Setting changes are revisioned and
  published through a `ChangeSet`.

### Transactions and performance architecture

- Ordinary edits use nested delta-journalled transactions instead of
  full-model snapshots. First writes capture only the affected records.
- Outer commit validates the changed dependency closure, updates reverse
  incidence and the spatial index, increments revision once, and publishes one
  deterministic `ChangeSet`.
- Failure rolls back geometry, semantics, structural records, caches, and
  spatial state while retaining allocator high-water marks.
- Change hooks are read-only observers. Snapshot restoration and feature
  regeneration publish atomically, so observers cannot see mixed topology and
  feature-history state.
- Reverse vertex/edge/face incidence and edge/member and face/sheet incidence
  are maintained explicitly. A lazy height-balanced AABB tree supports
  deterministic changed-region and full-audit candidate generation.

### Persistent mixed plate-and-beam topology

- `Part`, `Sheet`, `FaceUse`, and persistent `Coedge` records represent plate
  ownership and oriented shell incidence.
- `Member` and ordered `MemberEdgeUse` chains represent physical beams. They
  are distinct from geometry edges and from finite-element beam elements.
- Member identity survives axis subdivision. Coedge identity survives
  orientation-only edge and face reversals.
- `Attachment` represents declared member-to-face or member-to-edge
  relationships. `Junction` represents endpoint, crossing, overlap, and
  multi-way member intent with parameter-specific uses.
- Split, copy, insert, mirror, pattern, reversal, and removal paths preserve or
  reject structural dependencies atomically. Dependency-aware public removal
  methods prevent dangling relationships.
- `GeometryModel.add_members(...)` batches large beam lattices into one part
  update and one structural validation.

### Geometry qualification and mutation policy

- `TolerancePolicy` separates computational length, merge/heal, angular,
  parameter, area, and surface-residual tolerances. Relative scaling uses only
  participating local feature extents, never global coordinate magnitude or an
  unrelated model extent.
- Typed qualified predicates distinguish disjoint, touch, tangent, cross,
  overlap, coincidence, and unclassified results, with parameters, witnesses,
  residuals, and quality.
- Qualified analytical coverage includes line/line, line/plane, plane/plane,
  line/cylinder, and bounded segment/segment cases.
- Planar `clip_line_to_face` returns every material interval and subtracts
  holes instead of collapsing a concave or holed face to one span.
- Topology-changing face intersections require an explicit `MutationPolicy`.
  Query-only calls remain non-mutating; unsupported weld/reuse combinations
  reject instead of approximating a result.
- Support surfaces are retained across qualified edits. Circular geometry
  rejects anisotropic or singular transforms that cannot be represented
  exactly by the current curve/surface types.

### Strict fail-closed audit

- `strict_audit()` builds an independent broad phase and classifies every
  vertex/vertex, vertex/edge, edge/edge, edge/face, and face/face candidate
  admitted by conservative bounds.
- Checks cover duplicate and reversed geometry, T-junctions, crossings,
  collinear/circular overlap, face overlap and containment, crossing faces,
  sheet manifoldness and orientation, replacement lineage, structural
  ownership, member/member relationships, member/face attachments, junction
  geometry, and maintained-index consistency.
- Unsupported or ill-conditioned narrow phases become blocker-level
  `UNCLASSIFIED_CANDIDATE` issues. They cannot produce a false clean result.
- Declared overlap junctions are proved at all piecewise-straight member-use
  breakpoints. Multiple circular-arc crossing components are qualified
  separately. Unrelated distant geometry cannot relax a local classification.
- Certified serialization requires a complete, verified, certifiable audit.

### Serialization and feature history

- Geometry schema 3 is canonical and SHA-256 checksummed. It persists model
  identity/revision, coordinates, tolerance, all allocator high-water marks,
  geometry, surfaces and trims, structural ownership and relationships,
  groups, tags, lineage, extensions, and feature history.
- The current-schema reader rejects missing or unexpected core fields,
  duplicate JSON keys and records, malformed enums, invalid counters,
  non-finite data, checksum mismatch, unresolved references, and invalid
  topology.
- Schema 1 and 2 load through conservative one-way migration. Face ownership
  is inferred without inventing beam identity or cross-face orientation, and
  migration provenance is recorded.
- JSON and deterministic gzip JSON round-trip model identity and content.
- Feature history is owner-observed: append, update, move, suppress, remove,
  restore, and regeneration are revisioned. Records returned to callers are
  detached copies, and persistence is validated before checksumming.
- Feature regeneration stages topology and history together, preserves
  replacement lineage, and publishes one feature-aware change event.

## Compatibility and downstream impact

- Package version: 0.2.0
- Geometry document version: 3
- Feature-history version: 1
- `EntityRef`, existing geometry construction, query, serialization, and CLI
  entry points remain available where practical.
- Consumers should migrate cross-document state to `EntityHandle`, replace
  direct store mutation with owner methods, and represent physical beams with
  `Member` records rather than edge groups.
- Schema migration is one-way on write. A schema-3 document is not expected to
  load in a schema-2-only consumer.
- No finite-element generation, mesh numbering, seeding, smoothing, mapped
  decomposition, materials, sections, thicknesses, loads, supports, solver
  state, or results were moved into this kernel.

## Verification

- Final full suite: **272 passed in 7.84 seconds**
- Focused audit, predicate, and spatial suite after the final local-tolerance
  hardening: 52 passed
- Import-boundary verification: ANYgeometry imports no sibling, mesher, FE,
  solver, or GUI package
- CLI qualification: version, example write, schema-3 read, and JSON inspection
  passed in the repository suite; the isolated wheel reported `0.2.0`.
- Source and wheel builds: passed with `python -m build`.
- `twine check`: passed for both artifacts.
- Clean-environment wheel smoke test: passed for import, `py.typed`, model
  construction, schema-3 round trip, model UUID preservation, topology
  validation, and the module CLI.
- Artifact SHA-256 hashes:
  - `anygeometry-0.2.0.tar.gz`:
    `c01e1e1d605fb8466f0e269a92f55777eedc665eb53b3d46f83dcd9377cc3b7a`
  - `anygeometry-0.2.0-py3-none-any.whl`:
    `96201dd09361d11c661bd154297cb994097b79c0fed6019449e212ed982ba391`

## Performance qualification

The benchmark runner uses public APIs and records elapsed time, peak Python
allocation where practical, entity counts, changed-entity/index-update counts,
and strict-audit broad/narrow-phase work. Machine times are evidence, not
hard-coded acceptance thresholds.

Final qualification profile and measurements (normal uninstrumented runtime is
shown where the runner records both normal and allocation-instrumented time):

| Workload | Delivered result | Qualification evidence |
|---|---:|---:|
| 100 x 100 plate grid | 10,201 vertices, 20,200 edges, 10,000 faces | 1.964 s construction; 30.2 MB traced peak |
| Clean grid strict audit | clean and certifiable | 259,000/259,000 candidates classified, 0 unclassified, 72.725 s |
| One large-grid point edit | 5 records changed, 8 index updates | 0.200 s, 128,544 B traced peak; baseline 12.066 s |
| Grid schema-3 serialization | 40,401 geometry records | 0.920 s normal, 5.723 s allocation-instrumented; baseline 0.937 s |
| 10,000-member lattice | 20,000 vertices, 10,000 edges, 10,000 persistent members | 0.712 s normal, 3.652 s allocation-instrumented |
| Mixed stiffened panel | 2,601 vertices, 5,100 edges, 2,500 faces, 98 members | 0.661 s normal; baseline 19.843 s |
| Curved cylinder | 10,496 vertices, 10,368 edges, 5,120 faces, 167 members | 6.678 s normal, 35.241 s allocation-instrumented |
| Hostile duplicate/crossing model | 8,000 vertices, 6,000 edges, 2,000 members | 30,000/30,000 candidates classified, 0 unclassified, 6,000 issues, 7.102 s audit |
| 1,000-step replacement lineage | 1,001 descendants | 0.00294 s resolution; baseline 0.184 s |

The local edit is about 60 times faster than the schema-2 baseline while
touching only its five-record dependency closure. The mixed structural panel
is about 30 times faster in normal runtime than the baseline despite now
creating persistent member ownership. Serialization normal runtime remains at
baseline scale while carrying the larger checksummed schema-3 document.

The clean grid's 259,000 broad-phase candidates are 99.97% fewer than the
approximately 816 million naive unordered pairs among its 40,401 geometry
records. Every admitted candidate was qualified; no uncertainty was hidden.

The retained baseline file is `benchmarks/wave0_qualification.json`; the final
file is `benchmarks/wave5_qualification.json`. Both are committed alongside the
public benchmark runner for reproducibility.

The performance release criterion is scaling behavior: local edits must remain
proportional to the dependency closure, sparse audit must begin with spatial
candidates rather than all pairs, and no ordinary edit may deep-copy the full
model. Construction and persistence costs are reported explicitly even when
the richer identity/structural model exceeds the schema-2 baseline.

## Known limitations and fail-closed boundaries

- This remains a structural-surface kernel, not a general solid CAD kernel.
- Certified curved coverage is intentionally narrower than query coverage.
  Unsupported spline/surface, cone, ruled, Coons, oblique, or curved-trim
  candidates block certification rather than being sampled as disjoint.
- General ellipses and NURBS are not represented. Anisotropic transforms of
  arcs, cylinders, and cones reject.
- The qualified transverse plane/cylinder closed-ring imprint requires a
  complete conformal cylinder band and a supported plane trim. More general
  shell booleans remain non-mutating query workflows.
- `EntityRef` is model-unbound and therefore unsafe as persistent
  cross-document identity; it remains only for local compatibility.
- Schema-1/2 migration cannot infer physical members, attachments, junctions,
  or historical ownership intent that was never serialized.
- Full strict audit is deliberately more expensive than incremental commit
  validation. Certified output pays that full-model qualification cost.
- ANYmesher requires a separate downstream update to consume the new member,
  attachment, junction, and `ChangeSet` contracts. That work is outside this
  run.

## Commit and artifact record

- Base: `19f16746ceee76838b7df9d08a95028699e3a738`
- Implementation commit:
  `dc1860feb418fb5ab43b38297ca7aa710a7dce34`
  (`feat: add strict mixed plate-member kernel`)
- Test commit:
  `9f8df6f181c6e8e40e5a365768dede0de01c1910`
  (`test: qualify strict kernel invariants`)
- Qualification and migration documentation commit:
  `59c48a59467aae4be603f96f5e882b3a1c337228`
  (`docs: record kernel qualification and migration`)
- This report is committed separately as the final branch handoff record; its
  enclosing commit is listed by `git log kernel_update..HEAD`.
- Release artifacts: `anygeometry-0.2.0.tar.gz` and
  `anygeometry-0.2.0-py3-none-any.whl`
