# Intersection and overlap policy

Geometry queries and topology mutation are deliberately separate.

## Typed results

`IntersectionResult` identifies both parents, classification, dimension,
deterministically ordered components, parameters/ranges on both parents,
witnesses, qualification quality, maximum residual, and the tolerance used.
Curved results additionally expose an `IntersectionCertificate` and typed
`CertifiedCurveTrace`, `ParameterLoop`, and `ParameterRegion` records. The
certificate reports residual and enclosure bounds plus boxes examined,
subdivisions, and trace segments.

Classifications include disjoint, touch, tangent, cross, overlap, contained,
coincident, unsupported, capability missing, and unclassified. Dimensions are
none, point, curve, and region. Unsupported or capability-missing is a typed
result, not an empty result or an approximate guess.
For a built-in curved pair, exhausted work or unresolved degeneracy is
`UNCLASSIFIED`, never a classified sampled approximation.

## Three-stage workflow

1. `query_intersection(...)` is read-only and returns a typed result.
2. `plan_imprint(...)` validates explicit policy, resolves reusable topology,
   and returns an immutable deterministic plan with expected changes and
   affected structural owners.
3. `apply_imprint(...)` revalidates the plan against model UUID/revision and
   participating geometry, applies it in one transaction, and returns the
   committed relation/result and `ChangeSet`.

A stale, unverified, unsupported, or capability-missing plan cannot mutate.
Repeating a qualified plan reuses compatible topology and does not duplicate
vertices, edges, relationships, or lineage.

`query_intersection`, `plan_imprint`, `strict_audit`, and
`audit_changed_region` accept an optional `IntersectionQualificationPolicy`.
Defaults cap work at 200,000 parameter boxes, subdivision depth 32, 4,096
components, 65,536 trace segments, and 16 Newton iterations. Geometry
tolerances remain model-owned and use only participating local geometry.

`DISJOINT`, a same-parent query, or a face/face point contact plans
`NO_TOPOLOGY`; none can fail late inside a mutation transaction. Point-only
face contact is a typed unsupported imprint because it creates no persistent
face subdivision.

## Policy

Callers must choose among reject, reuse-existing, weld, imprint,
keep-separate-part, connect, keep-disconnected, and contact-only semantics.
Being within computational tolerance never authorizes welding.

`REUSE_EXISTING` is strictly non-creative: a compatible shared edge,
Junction, or Attachment must already exist, otherwise application rejects
without advancing revision. Pair/policy combinations are checked during
planning, before any transaction begins.

Qualified Member endpoint contact persists `JunctionKind.ENDPOINT` and
`MEMBER_ENDPOINT_ON_MEMBER`. A Sheet operand distinguishes
`MEMBER_ENDPOINT_ON_SHEET`, `MEMBER_CROSS_SHEET`, and `MEMBER_ON_SHEET` from
the corresponding direct-Face evidence. Classifications that cannot be proved
remain typed unsupported rather than being collapsed to a generic crossing.

## Face clipping and planar overlay

Line/face clipping returns every deterministic material interval and subtracts
holes. Concave openings, multiple holes, tangencies, and loop reversal do not
collapse to one min/max span. Qualified simple cases may use a NumPy path;
general planar polygon overlay uses the optional planar backend. If that
capability is absent, general cases return `CAPABILITY_MISSING` and mutation
is blocked.

## Supported and fail-closed cases

Analytical predicates cover straight segment/segment, line/line, line/plane,
plane/plane, line/cylinder, qualified plane/cylinder, same-circle arc overlap,
straight edge/planar face, and planar face overlay. The shared bounded engine
covers all six unordered Straight/Arc/Spline families, all fifteen curve and
Plane/Cylinder/Cone/Ruled/Coons families, and all fifteen unordered built-in
face-support families in either operand order. Same-support trims return every
connected coincident material cell with parameter loops and holes.

Curved components are persisted only from complete certificates; adaptive
traces become fitted Bezier chains rather than uncertified polylines.
Multi-component imprints commit in one transaction. Full coincidence and
certified contained partial regions can share one Face between distinct Sheet
owners. Unresolved singular or degenerate configurations are `UNCLASSIFIED`;
optional capability absence remains `CAPABILITY_MISSING`. Both block mutation
and strict handoff.

## Indexed overlap queries

Public overlap queries accept selected face IDs, changed AABBs, or explicit
candidate pairs. When candidates are not supplied, the maintained spatial
index supplies them. Results and diagnostics expose broad-phase candidates and
narrow-phase classifications; the default path never begins with all face
pairs.
