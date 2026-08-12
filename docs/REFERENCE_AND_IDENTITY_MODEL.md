# Reference and identity model

ANYgeometry separates compact local storage keys from references that may
cross package or document boundaries.

## Public and internal references

- Internal algorithms use `(kind, integer_id)` keys and `EntityRef` values.
- Public cross-package APIs use `EntityHandle(model_id, kind, id)`.
- `model_id` is a persistent, non-nil UUID stored once in the document.
- A numeric ID is unique only within its entity kind and owning model.
- IDs are monotonic high-water allocations. Retired committed IDs are never
  reassigned, and gaps are valid.

`GeometryModel.handle()` validates both kind and raw integer identity before
binding it to the model. `resolve_handle()` returns a typed `Resolution`; it
never silently resolves a handle from another model.

## Resolution states

`ACTIVE`, `REPLACED`, `DELETED`, `UNKNOWN`, `WRONG_MODEL`, `SUPPRESSED`, and
`BLOCKED` are mutually exclusive. Replacement resolution stays within one
entity kind and returns deterministic active descendants. A terminal state
contains no descendant handles.

Geometry and structural records are immutable through public views. Public
NumPy arrays are read-only. Metadata, loops, use chains, groups, and tags
cannot be mutated without a model-owned operation and transaction.

## Geometry and structural kinds

Geometry kinds are vertex, edge, and face. Structural topology adds part,
sheet, face-use, coedge, member, member-edge-use, attachment, and junction.
Curve-control and construction/reference usage is explicit and does not make a
point a structural connectivity vertex merely because a curve depends on it.

## Copies and extracted working models

An ordinary clone is an independent document and receives a new model UUID.
`extract_model_closure()` also creates an independent working model. It
returns model-bound source-to-working and working-to-source handle maps plus
the source UUID, revision, and canonical selected source handles. Downstream
packages must use those maps; matching integer IDs are never provenance.

Only complete structural parents are copied. A partial member or sheet is not
invented. Attachments and junctions are copied only when all participating
parents belong to the extracted closure.

## Serialization

Current documents persist model UUID, revision, every allocator high-water
mark, replacement lineage, structural ownership, and feature identity.
Loading validates kind, identity, uniqueness, ownership, and high-water
consistency before publishing the model. Legacy unbound references are bound
once to the UUID created or read during migration.
