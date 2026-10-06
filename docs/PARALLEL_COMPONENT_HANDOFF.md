# Geometry component handoff

`plan_independent_components`, exported from `anygeometry`, is a read-only,
conservative partition of selected face, member and explicit edge roots:

```python
from anygeometry import (
    plan_independent_components, validate_component_partition_binding,
    extract_model_closure, ModelClosure,
)

partition = plan_independent_components(model, separation=0.1)
if partition.certified:
    validate_component_partition_binding(model, partition)
    messages = [extract_model_closure(model, component.handles).to_transport()
                for component in partition.components]
    # Send messages with JSON or multiprocessing spawn transport.
    working_closures = [ModelClosure.from_transport(message) for message in messages]
else:
    # Use the consumer's ordinary serial workflow, retaining refusal reasons.
    reasons = partition.refusals
```

`face_ids=None` and `member_ids=None` select all active owners; `edge_ids=()`
selects no additional standalone edges. Explicit IDs must be active local IDs.
The result binds the model UUID, revision, complete current document checksum,
sorted requested handles and separation. Each component exposes `handles`,
`face_ids`, `member_ids` and `edge_ids`. `merge_reasons` contain two roots, a
reason and, where applicable, the shared dependency or declared relationship.
`refusals` contain a reason and source handles. Invalid IDs, stale revisions,
invalid topology, cancellation and mutation during the query raise
`GeometryError`. A refused partition has **no executable components**; it does
not mean there is no selected work.

Separation is finite and nonnegative, in model document units. It expands each
conservative dependency box in every direction, in addition to the existing
model length tolerance; consequently a gap up to twice the expansion merges.
Box overlap certifies neither contact nor noncontact. False positive merging is
intentional. The planner never cuts a face or grants authored region admission.

The graph closes shared vertices, control dependencies and edges, complete
sheet membership, member orientation references, every attachment connection
intent, attachment context sheets, active lineage and declared junctions.
Standalone geometry referenced by a relation is a shared extraction dependency
even when it has no original face/member owner. Its complete geometry enters
consumer extents. Active Part lineage closes its complete children; an ordinary
context Part may remain partial in several independent closures. A selection
omitting a dependent/interacting face/member owner or part of a selected sheet
refuses instead of silently dropping connections or enlarging the selection.

Planar material uses exact trim-edge AABBs. Cylinder and Cone material uses a
full radial revolution over conservative boundary axial extents, including
legal support extrapolation. Supported edge curves use their analytic or
control-hull bounds. Missing/nonfinite bounds refuse. ExtrudedSurface,
RuledSurface, CoonsSurface and absent/unknown supports currently refuse because
their normalized support-patch boxes alone do not prove enclosure of arbitrary
legal trimmed extrapolation. This explicit fallback is part of the contract.
Member, member-use or attachment metadata declaring eccentricity/offsets
refuses every nonempty selection, including when the declaring owner is
unselected: this schema has no geometry-owned physical offset extent contract,
so disjoint root-axis bounds cannot exclude interaction with that unknown reach.
Consumers must not encode additional physical reach outside that contract and
infer certification from a root-axis box. Caller-known additional reach can be
included in `separation`, but it does not turn an explicit refusal into proof.

`validate_component_partition_binding` checks fresh content and independently
reconstructs the deterministic partition, rejecting stale source or changed
evidence. It must run before using a previously retained certificate. It is a
development API with focused regression evidence; it confers no release or
scientific qualification beyond the stated conservative partition contract.

## Transport and identity

The existing `to_dict`/`from_dict` geometry document is plain JSON/pickle-safe
data and preserves its model UUID, revision and every active local ID, including
sparse numbering. It is the worker transport contract; a live `GeometryModel`
and its mapping proxies are not the transport representation. The current
geometry schema/version is unchanged.

`extract_model_closure` creates a distinct working UUID and dense local IDs.
`ModelClosure.to_transport` / `ModelClosure.from_transport` (also exported as
`model_closure_to_dict` / `model_closure_from_dict`) transport that validated
geometry document, source UUID/revision/selection and a complete bidirectional
mapping. Mappings cover vertices, edges, faces, Parts, Sheets, FaceUses,
Coedges, members, member uses, attachments and junctions. Duplicate, incomplete,
wrong-kind or inconsistent mappings refuse. The transport envelope has its own
integrity checksum and uses the current geometry schema inside it. Reconstructed
maps are read-only. Decoding copies arrays, so a worker's raw array mutation
cannot alter the source model. A worker retains declared source provenance;
the absent original source document cannot be independently requalified there.

## Connected chart preparation

`query_trimmed_surface_charts` keeps its all-or-error result and deterministic
face order. It still qualifies the same material domains, tolerances and areas.
Read-only query topology qualification is reusable only for a byte-identical
complete content binding and identical validation-relevant reverse incidence
maps. A fresh immutable index snapshot detects raw derived-index corruption
without changing the persisted schema or checksum. Every call freshly encodes source content at entry
and exit; raw edits at an unchanged revision are detected. Public `to_dict`
continues to validate every serialization request. Existing chart validation
also binds chart definitions, occurrence lists, tolerance and area, including
direct changes to previously returned evidence.

`query_trimmed_surface_charts_by_face` accepts the same selection and returns
`TrimmedSurfaceChartQueryResults` with sorted per-face `results`. Each row has a
face handle and either a normal bound `TrimmedSurfaceCharts` in `charts`, or the
original qualification error text in `error`. Source/selection errors,
cancellation and raw mutation abort the whole query, including after partial
work. Each successful row uses the existing
`validate_trimmed_surface_charts_binding` API before consumption. This lets a
consumer replace repeated single-face queries with one complete document
binding while preserving face-specific unsupported outcomes.

Preparation additionally avoids repeated topology validation at post-qualified
plan/application/provenance freshness guards. Those guards still freshly encode
all current content and compare it against the prior qualified checksum.
Definition checksums fast-path exact primitive values without changing canonical
JSON, hashing, array mutation detection or output independence. No cross-call
entity checksum or revision-only source cache is introduced. Measurements and
installed consumer integration remain separately assessed; this document makes
no unmeasured performance claim.
