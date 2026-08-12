# Changelog

All notable user-visible changes to ANYgeometry are documented here.

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
