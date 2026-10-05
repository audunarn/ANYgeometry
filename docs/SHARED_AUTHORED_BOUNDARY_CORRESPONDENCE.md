# Shared authored boundary correspondence

`PreparedEdgeSubcurvePreimages.alias_records` is an additive, transient tuple of
`EdgeSubcurvePreimage` rows. It records producer-authenticated occurrences of a
unified shared boundary: when the detached batch producer reuses one canonical
current edge for coincident duplicates, each duplicate's sealed ancestry and
every occurrence already captured for it re-seals onto the canonical edge's
exact current definition. `records` stays single-ancestor and unchanged in
meaning; a model without unification prepares with `alias_records == ()` and
behaves exactly as before. Aliases are never persisted, never seeded from a
document, and never created by ordinary edits.

## Exact API

- `alias_records: tuple[EdgeSubcurvePreimage, ...] = ()` on
  `PreparedEdgeSubcurvePreimages`, sorted by
  `(edge_id, ancestor.definition.edge_id, interval)` and deduplicated by
  `(edge_id, ancestor, interval)` at finalize.
- Each row carries the same fields as a primary record: the participating
  original boundary's `ancestor`, its oriented source `interval`, the canonical
  edge's `current_definition`, exact Bernstein `error_controls`,
  `coordinate_bounds`, `squared_distance_bound`, and the occurrence's own
  recorded `tolerance`.
- `query_prepared_edge_subcurve_preimages` returns primary `records` plus
  `alias_records`; `unavailable_edge_ids` still lists every edge without a
  primary record, so unqualified children stay fail-closed even when aliases
  exist elsewhere.
- `query_prepared_authored_boundary_correspondence` treats
  `(*records, *alias_records)` as candidates and selects exactly the rows whose
  ancestor matches the requested authored root's model ID, revision, source
  checksum and original polynomial; a different root's occurrence is never
  substituted.
- `query_prepared_authored_boundary_stations` likewise selects the unique
  exterior occurrence belonging to this correspondence's authored root and uses
  that row's own interval and orientation; ambiguity or absence refuses.

## Proof rule

Every alias is produced by `_reseal_occurrence`: orientation is fixed by
endpoint-ID topology alone, never by residual size, tolerance or samples. The
occurrence's sealed current definition and the canonical edge's current
definition must bind the same two distinct endpoint IDs; the same ordered pair
preserves the source interval, the swapped pair reverses it, and any mismatched
or degenerate pair leaves the occurrence unauthenticated. Only that one
selected interval is certified: the ancestor's exact rational controls are
restricted to it by exact Bernstein restriction, elevated, and differenced
against the canonical edge's exact current controls over the whole interval,
accepting only a residual within the occurrence's own recorded tolerance.
Nothing is inferred from proximity, geometry, samples or replacement metadata.
A failed seal leaves that occurrence unauthenticated; unification never creates
ancestry, intervals or tolerances.

## Bindings and staleness

The binding carries `model_id`, `revision`, `source_checksum` and full edge
`coverage`. Queries re-fingerprint the serialized model state and refuse on any
same-revision edit, revision change, coverage change or definition checksum
change. Re-preparation re-captures aliases from the current receipt
idempotently. Forged or hand-assembled alias bindings fail the receipt,
checksum and coverage checks. A later split composes carried occurrences
through the split with the same tolerance-minimum rule; out-of-tolerance
occurrences become unavailable. A canonicalizing vertex merge re-seals
incidence only through the unchanged-tolerance rule. When an optional split
enclosure proof is unavailable, the primary records and every captured
occurrence of the dead parent and its children are dropped, so proof queries
explicitly refuse those unqualified children.

## Scope

Supported: canonical reuse of one current edge for coincident authenticated
duplicates during the detached batch producer's planar shared-boundary
unification, with each participating authored root keeping its own sealed
occurrence for correspondence and station queries.

Unsupported: any alias without a sealed same-tolerance proof, stale or
re-seeded receipts after ordinary edits, cross-model or cross-revision
ancestry, and any document schema, material containment, analytic
substitution, split permission or meshing permission. Receipts confer
ancestry and a whole-interval approximation bound only.
