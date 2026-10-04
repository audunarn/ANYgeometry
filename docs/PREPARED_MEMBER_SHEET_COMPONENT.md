# Bounded member-bearing prepared component

`query_prepared_member_sheet_joint_component(model, current_joint_edge_id)`
and `validate_prepared_member_sheet_joint_component_binding(model, receipt)`
provide an additive owner contract. The legacy Sheet-only query and validator
retain their existing member and attachment refusals.

This qualification covers a complete document with two authored Plane/Straight
Sheets, one source member with one full-range straight boundary edge use, and
one exact vertex-on-edge point attachment. Both member orientations are supported.
There must be no pre-existing Junctions or member orientation references.
Additional members, original attachments, unrelated Sheets, unsupported current
relations, nonlinear carriers and unavailable required ancestry explicitly refuse.
These are boundaries of this new qualification case, not general model-count
limits or acceptance of large-model meshing.

The distinct `PreparedMemberSheetJointComponent` includes the complete live
original/current model scope and the existing component incidence, Sheet,
FaceUse occurrence and declared joint accounting. `member_relation` parses a
fresh detached dictionary containing original/current Member, ordered edge uses,
point Attachment and source vertex records. The point squared-distance bound
uses exact rational arithmetic on stored floating-point coordinates; the allowed
coordinate tolerance retains the original attachment tolerance and owner policy.

Polynomial ancestry must tile the original carrier's whole [0,1] interval with
zero restriction error. For a child interval [a,b], station s maps to a+(b-a)s;
reversed member traversal uses the reversed child order and parent ranges
[1-b,1-a]. This formula is not a remap for regularized analytic branch curves.
Generated joint edges may lack polynomial ancestry; only the source-bearing
member/attachment carrier requires that proof.

The query and validator bind all persisted input content, original document and
curve identities, current orientation/ranges, retained point station/coordinates,
attachment payload/lineage and declared Sheet joints. Forged omissions, changed
qualification flags, wrong owners, stale evidence and same-revision callback edits
refuse. Ordinary load/clone operations cannot reconstruct transient preparation
authority. Querying does not mutate the model.

Receipt validation accepts only the exact immutable owner definitions and plain
values. It captures their content before model serialization and verifies it
after the final binding guard. Copy hooks, behavioral substitutes, cyclic
definitions and malformed nesting raise `GeometryError`; they cannot repair
invalid evidence during validation. Generated receipts receive the same closing
content check.

`bounded_relation_mapping_qualified` is scoped to these relations.
`semantic_mapping_qualified`, `beam_discretization_qualified`,
`external_reference_transfer_qualified` and `publication_qualified` remain false.
The receipt does not qualify material partitions, beam/coupling discretization,
external loads or project references, solver readiness or mesh publication.
ANYmesher must separately account for these relations in its candidate mesh;
ANYfem must independently bind its live Project manifest.
