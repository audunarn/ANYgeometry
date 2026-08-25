# ANYgeometry automation protocol 1

## Boundary

The automation package is a deterministic kernel adapter, not a natural
language subsystem. Providers translate user intent into strict protocol
records. ANYgeometry validates those records and delegates to existing owner
APIs. Prompt parsing, Python/code execution, arbitrary method dispatch,
filesystem selection, model inference, and provider SDKs remain outside the
kernel.

The geometry document schema remains 4. Automation protocol version 1 is an
independent request/response contract.

## Request lifecycle

Every model-bound request carries `protocol_version`, `request_id`,
`model_id`, and `expected_revision`. Wrong UUIDs and stale revisions fail with
stable typed errors. Query results return canonical model-bound handles and the
exact revision used.

Mutations are always two phase:

1. `plan_commands(model, batch)` validates units, selectors, cardinality,
   symbolic references, capabilities, and policies; freezes active handles;
   runs an exact preview on an unpublished schema clone; and produces a
   canonical SHA-256 `EditPlan`. The plan reports net entity-count changes,
   symbolic Part owners, affected bounds, required capabilities, and blocking
   diagnostics. Planning restores cache identity and cannot consume live IDs
   or alter revision/high-water marks.
2. `apply_plan(model, plan)` verifies UUID, revision, digest, blocking state, and duplicate
   application, then executes the complete batch inside one outer transaction.
   Failure rolls back; success returns one `ChangeSet`, actual output handles,
   replacement resolutions, and non-certifying changed-region audit evidence.

## Selection

Selectors are one bounded AST with maximum Boolean depth 8 and maximum 64
predicates. Results default to 100 per page and cannot exceed 1,000. Mutation
selectors require `[minimum, maximum]` cardinality. Nearest selectors require
both maximum distance and result limit. Canonical handle ordering resolves all
ties.

Supported leaves include kind/handle, group/tag, Part owner, incidence,
boundary/connectivity, curve/support type, namespaced metadata equality, AABB,
centroid axis, nearest, and length/area/radius ranges. `all`, `any`, and `not`
compose leaves. Geometry AABB and nearest candidates use the maintained model
index; topology predicates use reverse incidence.

Input aliases are accepted only at selector boundaries: `point` means
`vertex`, and `plate` means the structurally owned `sheet`. Responses always
use canonical kinds. A raw trimmed surface is `face`.

## Units and frames

Dimensional inputs are `Quantity` objects. Length units are `m`, `mm`, `cm`,
`in`, and `ft`; squared forms apply to area; angles are `rad` or `deg`.
Positions, directions, boxes, and transforms require `model_local` or `world`.
Unknown/nonfinite values and unsupported document unit metadata fail closed.

## Commands

Batches contain 1–256 uniquely named commands. A command may consume an
earlier output such as `p1.vertex`; forward references and unknown ports are
errors. Initial operations are point/straight-edge/raw-face/owned-plate
creation, translate/rotate/move/copy/mirror/pattern, group/tag,
dependency-safe delete, and qualified imprint with explicit policy.

`create_plate` creates a complete `Part` → `Sheet` → `FaceUse`/`Coedge`
ownership closure around the resulting Face and returns every created handle.
`create_face` is the explicit geometry-only alternative.

## Reference MCP adapter

The separate `ANYgeometry-mcp` package for this release depends on
`ANYgeometry>=0.4,<0.5` and
the official MCP Python SDK. It exposes exactly seven tools: discovery, model
summary, selection, entity description, query, planning, and application. One
geometry file is bound at startup; tool input cannot select another path.
Application requires explicit approval. Write-back is disabled unless enabled
at startup, must additionally be requested per application, and writes only
the canonical startup path. Per-session receipt ledgers make transport retries
idempotent; stale model revision blocks repeats outside that receipt context.
