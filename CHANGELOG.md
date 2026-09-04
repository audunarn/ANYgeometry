# Changelog

All notable user-visible changes to ANYgeometry are documented here.

## Unreleased

- Add public feature-topology roles and exact `feature_entity_owners()`
  resolution so consumers can present composite generators as lightweight
  intent objects without duplicating feature-kind policy or changing topology.

## 0.4.2 - 2026-09-03

- Change the license for ANYgeometry source code distributed from this release
  onward from GPL-3.0-or-later to MPL-2.0. Earlier published versions retain
  the license terms under which they were released.
- License original project documentation under CC BY 4.0, preserve distinct
  terms for code snippets and third-party material, and add explicit copyright
  and third-party dependency notices.
- Add deterministic licensing validation to CI and release builds, including
  package-metadata, dependency-inventory, wheel, and source-archive gates.
- Keep geometry serialization schema 4 and automation protocol version 1
  unchanged.

## 0.4.1 - 2026-08-29

- Fix corner-to-corner planar imprinting by snapping qualified intersection
  endpoints to existing trim vertices before polygonization.
- Harden immutable release-ledger verification and the hosted distribution
  qualification workflow.

## 0.4.0 - 2026-08-24

- Add a shared NumPy-only qualified intersection engine for every unordered
  Straight/Arc/Spline pair and every curve/face and face/face combination of
  Plane, Cylinder, Cone, RuledSurface, and CoonsSurface. Public results carry
  bounded-work certificates, oriented parameter ranges, parameter-space
  paths/regions, residuals, enclosure widths, and typed certified traces.
- Route curved strict and changed-region audit candidates through the same
  public engine. Incomplete work, singularity, nonfinite arithmetic, and
  resource exhaustion remain blocking and fail closed; audit metrics expose
  boxes, subdivisions, trace segments, spatial work, and affected structural
  closure size.
- Add atomic multi-component curved imprinting and qualified full/contained
  coincident-region CONNECT while preserving Sheet/FaceUse/Coedge ownership,
  support and parameterization, semantic lineage, and structural relations.
  Reapplication is idempotent and rollback restores live state and derived
  caches without reusing provisional IDs.
- Add translation/scale, trim/hole/nonconvex, ownership-remap, locality,
  rollback, and complete built-in family matrices. Geometry document schema 4
  and automation protocol version 1 remain unchanged.

## 0.3.0 - 2026-08-24

- Add dependency-free automation protocol version 1 under
  `anygeometry.automation`. It provides strict immutable request/response
  records, JSON Schema/tool discovery, model-bound descriptions, bounded
  declarative selectors, explicit units/frames, measurements, and qualified
  intersection queries without parsing prompts or executing generated code.
- Add revision-bound `CommandBatch` planning and canonical SHA-256 `EditPlan`
  application. Symbolic point/edge/owned-plate construction, transforms,
  copies, mirrors, patterns, groups/tags, dependency-safe deletion, and
  explicit imprint policies commit through one outer kernel transaction.
  Planning and failed/stale/tampered application preserve revision, identity
  allocator high-water marks, and caches.
- Add the separate `ANYgeometry-mcp` reference package. It exposes exactly
  seven MCP tools against one startup-bound document, isolates session retry
  receipts, requires explicit mutation approval, and makes write-back both a
  startup capability and a per-call opt-in. The kernel has no MCP dependency.
- Keep persistent geometry document schema 4 unchanged. Automation protocol
  version 1 is a separate transport contract.

## 0.2.4 - 2026-08-22

- Add atomic exact `FeatureHistory.adopt_frozen(...)` for imported, scripted,
  unavailable-executor and committed mesh-partition materializations. Adoption
  validates active outputs and structural topology, persists a canonical
  closure checksum, and never follows lineage or geometric proximity.
- Preserve stable prefix outputs during additive feature regeneration and make
  dirty suffix replay fail closed on incompatible structural ownership.
- Add oriented Sheet ownership, deterministic structural intersection
  contracts, and the boundary-curve support required by ANYmesher 0.2.5.

## 0.2.2 - 2026-08-21

- Qualify exact face/face CONNECT when a complete topology-owned straight,
  circular-arc, or Bezier boundary of a nonplanar face lies strictly inside a
  convex hole-free planar support. Reuse the original Edge and persist shared
  FaceUse/Coedge ownership without sampled or coordinate-inferred topology.
- Keep partial, ambiguous, holed, nonconvex, trim-touching, and general
  nonplanar intersections typed and fail-closed.
- Make the release workflow artifact-building only; PyPI publication remains
  an explicit manual Twine operation.
- Added owner-aware `FeatureHistory.adopt_frozen(...)` for trusted importers
  and script features. It binds exact active output IDs, computes and persists
  the canonical materialization checksum, optionally verifies a caller's
  expected checksum, and rolls back without publication on any failure.
- Made additive feature regeneration start at the earliest dirty record on a
  working clone of the live materialization. Clean-prefix IDs remain active;
  dirty outputs are matched only by stable output key plus exact complete
  closure, and changed outputs allocate above the previous high-water mark.
  Appending an unrelated feature and editing or undoing a suffix therefore no
  longer invalidates raw IDs belonging to unchanged upstream outputs.
- Qualified the package-owner boundary for crossing Sheets, Member/Sheet
  relations, and Member/Member connections. Queries and plans remain strictly
  read-only; topology and structural ownership change only through an explicit
  `apply_imprint(...)` with declared connection intent. No proximity welding is
  inferred or restored.
- Kept geometry document schema 4 and feature-history schema 1 unchanged;
  0.2.2 is an API and contract-hardening release.

## 0.2.1 - 2026-08-12

- Added the mesher-facing `ModelClosure` extractor with model-bound
  bidirectional handle maps, complete structural parents, preserved document
  settings, and deterministic vectorized edge/face evaluation and projection.
- Added typed `query_intersection` / `plan_imprint` / `apply_imprint` APIs,
  multi-component and coplanar planar imprinting, explicit connection intent,
  qualified member/member and member/sheet relations, persisted face/face
  shell-sheet T-junctions under explicit `CONNECT`, and idempotent reuse.
- Made `REUSE_EXISTING` strictly non-creative, preflighted invalid pair/policy
  combinations, preserved endpoint and Sheet-specific relationship kinds, and
  made disjoint/same-parent/point-only face operations fail closed before edit.
- Canonicalized straight sketch-extrusion walls to exact Plane support and
  aligned arc walls to Cylinder support; genuine non-planar Coons walls remain
  typed unsupported.
- Added batched analytic closest projection for Plane, Cylinder, and Cone,
  one-time trim preparation, deterministic sweep-boundary behavior, and
  bounded-memory spline evaluation.
- Added construction/control vertex roles, optional face parameterization
  separate from authoritative support, Member orientation references,
  structural split lineage, richer Attachment/Junction evidence, and complete
  local reverse-incidence queries.
- Added changed-region non-certifying audit, richer audit evidence, indexed
  overlap selectors and nearest queries, local-scope batch/bounds paths, and
  pair-local tolerance scaling.
- Advanced geometry writing to strict checksummed schema 4. The 0.2.1 reader
  accepts schemas 1–4 and migrates 1–3 one-way; schema-3 attachment evidence
  becomes explicitly `UNVERIFIED` and cannot certify. Older 0.2.0 readers fail
  closed on schema-4 documents despite the compatible Python version range.

## 0.2.0 - 2026-08-12

- Added model-bound UUID entity handles, monotonic non-reused geometry and
  structural IDs, immutable public records/stores, and revisioned document
  settings.
- Made feature-history edits owner-observed and revisioned, published feature
  regeneration atomically with topology, prohibited snapshot mutation from
  change hooks, and preserved coedge identity across orientation-only edits.
- Replaced whole-model edit snapshots with nested delta transactions,
  deterministic change sets, incremental reverse incidence, and a maintained
  dynamic AABB index for changed-region work.
- Added persistent parts, sheets, face uses, coedges, physical members,
  member-edge uses, beam/plate attachments, and beam/beam junctions, including
  split/copy/reversal/lifecycle bookkeeping and batched member construction.
- Added a model-owned scale- and translation-stable tolerance policy, typed
  qualified line/plane/cylinder and segment predicates, multi-interval planar
  face clipping, and explicit mutation policies for welding and imprinting.
- Added deterministic fail-closed full-model auditing for duplicates,
  crossings, overlaps, T-junctions, manifoldness, ownership, member intent,
  face intersections, spatial-index consistency, and unsupported candidates.
- Added schema-v3 checksummed serialization for model identity, revision,
  coordinate settings, tolerance, allocator high-water marks, all structural
  ownership and relationships, semantics, extensions, and feature history.
  Schema 1 and 2 documents migrate conservatively; certified output requires
  a clean strict audit.
- Added a persistent, dependency-aware, suppressible feature history with
  extensible namespaced executors, stable feature-output references, atomic
  regeneration, and replacement-lineage preservation.
- Added atomic model insertion, deep cloning, topology/structural-closure
  copying, mirroring, linear and circular patterns, orientation reversal, and
  typed geometry measurements.
- Added qualification benchmarks, hostile regression tests, package typing
  verification, and a public kernel architecture/report set. ANYmesher and
  downstream analysis packages remain unchanged in this release.

## 0.1.0 - 2026-08-08

- Established ANYgeometry as the single lightweight geometry authority for the
  ANY ecosystem.
- Extracted persistent `GeometryModel`, `Vertex`, `Edge`, `Face`, `EntityRef`,
  curve, topology, evaluation, and general editing behavior from ANYmesher.
- Added semantic groups, tags, stable replacement history, and stale-reference
  resolution across geometry edits.
- Added explicit plane, cylinder, cone, ruled, and Coons surface support,
  projection and closest-point queries, transforms, trimming, holes, and
  fragmentation.
- Added analytical line/plane/cylinder intersections, planar shell
  intersection imprinting, and a deterministic numerical surface fallback.
- Added atomic transverse plane-cylinder closed-ring imprinting for qualified
  conformal shell bands. Exact shared arcs fragment the plane and every
  cylinder patch while preserving IDs, replacement history, groups, tags, and
  serializable surface definitions.
- Added neutral plate, stiffened-panel, cylinder, cone, shell, frame,
  bulkhead, girder, and stiffener generators with semantic geometry groups.
- Added versioned JSON and gzip geometry serialization that preserves topology,
  entity IDs, ID counters, surfaces, groups, tags, holes, metadata, and lineage.
- Hardened general edits so face splitting, trimming, hole punching,
  fragmentation, and qualified intersection imprinting either complete with
  valid topology or restore the exact pre-edit model. Splits retain intact
  holes deterministically and reject cuts that touch or cross a trim.
- Made topology snapshots repeatably restore mutable point/edge state, nested
  metadata, explicit surfaces, semantic groups, tags, ID counters, and
  replacement history; public deletion now records explicit empty lineage.
- Added strict topology, surface-boundary, trim-containment, overlap,
  self-intersection, degenerate-curve, replacement-graph, and serialized-ID
  validation. Triangular planar fragments remain evaluable after edits.
- Preserved cylinder and cone evaluation under reflected uniform transforms by
  correcting parameter handedness and sweep direction, and made public normal
  evaluation boundary-safe for ruled surfaces.
- Added a typed-package marker and a lightweight `python -m anygeometry` CLI.
- Kept NumPy as the only core runtime dependency; Shapely is an optional
  backend for planar clipping and qualification workflows.
