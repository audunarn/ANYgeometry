# Changelog

All notable user-visible changes to ANYgeometry are documented here.

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
- Kept NumPy as the only core runtime dependency; Shapely remains optional for
  future planar/UV topology workflows.
