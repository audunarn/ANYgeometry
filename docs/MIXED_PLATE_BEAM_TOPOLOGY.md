# Mixed plate-beam topology

ANYgeometry stores neutral structural geometry. A `Member` is a persistent
physical beam-axis identity; it is not a finite-element beam. A `Sheet` is an
oriented shell-topology owner; it is not a shell mesh.

## Ownership graph

```text
Part
 +- Sheet -> FaceUse -> Face -> Coedge -> Edge -> Vertex
 `- Member -> MemberEdgeUse --------------------^

Attachment: qualified member/point incidence with an edge, face, or sheet
Junction: explicit connected, disconnected, or contact relationship
```

A `Part` distinguishes physical ownership and intentionally separate assembly
components. Coincident geometry in separate parts is not silently welded.

## Sheets and face uses

A sheet owns ordered oriented face uses. Each face use owns persistent coedge
loops that mirror the face trim loops. Coedge identity survives orientation or
loop-order edits where the underlying use remains the same. Several qualified
face uses may share one edge, including declared non-manifold intersections.

## Members and geometric subdivision

A member owns an ordered oriented member-edge-use chain with continuous
parent parameter ranges. Splitting an axis edge rewrites that chain and keeps
the member ID. An explicit `split_member()` is different: it creates member
descendants and records member lineage. Reverse operations preserve geometry,
reverse direction and parameterization, and update qualified relations
atomically.

One edge may simultaneously carry shell-boundary, trim, member-axis, and
intersection uses. Semantic roles do not require duplicate geometry.

## Attachments and junctions

Qualified relationships record both parents, classifications, parameter or
UV ranges, connection intent, exact/verified status, residual, tolerance,
part/sheet context, provenance, and lineage where applicable.

Connection intent is explicit:

- `CONNECT`: create or reuse conforming topology and a persistent relation;
- `KEEP_DISCONNECTED`: preserve separate topology and record that intent;
- `CONTACT_ONLY`: record incidence without a welded topology connection;
- `REJECT`: reject the candidate;
- `REUSE_EXISTING`: require compatible existing topology;
- `IMPRINT`: create qualified shared subdivisions/uses.

Repeated connection or attachment operations are idempotent. An unexplained
same-part crossing or overlap blocks strict qualification.

`REUSE_EXISTING` never creates an edge, Attachment, or Junction. Endpoint and
Sheet relations retain their qualified distinctions:
`MEMBER_ENDPOINT_ON_MEMBER`, `MEMBER_ENDPOINT_ON_SHEET`,
`MEMBER_CROSS_SHEET`, and `MEMBER_ON_SHEET`. Direct Face queries remain
face-specific; a Sheet relation is never inferred merely because a face has a
nearby owner.

Four or more qualified Sheet uses may share an intersection edge. Persistent
Sheet IDs and Coedge incidence are retained, and `radial_face_uses` supplies a
deterministic geometric order independent of insertion order. Each Sheet's
`NonManifoldPolicy` remains authoritative; global sharing does not silently
weaken a fail-closed Sheet policy.

## Public adjacency

Maintained indices answer vertex-to-edge, edge-to-face, edge-to-sheet,
edge-to-member, vertex-to-member, vertex-to-sheet, face/sheet-to-attachment,
member-to-attachment, non-manifold radial-use, and structural-owner queries.
Results are immutable and sorted by stable identity.

## Mesher boundary

ANYmesher consumes sheets, members, ordered uses, attachments, and junctions.
It does not discover structural intersections, infer member identity from
arbitrary edges, or heal geometry.
