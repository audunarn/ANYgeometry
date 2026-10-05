# Prepared boundary-member Sheet networks

This additive owner contract captures one whole connected document of supported
Plane/Straight Sheets, with finite collections of original boundary Members and
exact vertex-on-edge point Attachments. It introduces no model-count ceiling.
The earlier Sheet-only and two-Sheet/one-Member receipts retain their refusals.

`PreparedMemberSheetJointNetwork`,
`query_prepared_member_sheet_joint_network()` and
`validate_prepared_member_sheet_joint_network_binding()` are distinct from those
receipts. A successful query includes all selected Sheet occurrences, original
and current Member/use/point records, point source vertices and Parts, including
unchanged empty Parts. The document must be one connected declared Sheet-joint
component. Disconnected records are refused rather than excluded.

Each original Member uses one continuous finite chain of full-range Straight
boundary carriers. A Member may not repeat a carrier within its own chain: the
traversal semantics of such a repeated use are ambiguous and explicitly
refused. Different Members may share carriers, and multiple points may target
them. Each original MemberEdgeUse carries a global normalized arclength span
[p,q]; the spans of one Member tile [0,1] in chain order, and each use keeps
its own forward/reversed orientation. Exact polynomial ancestry must tile [0,1]
without gaps or overlap per carrier, with original model/revision/checksum/
control anchors. The owner's certified rounding enclosure is retained and must
satisfy its unchanged registered tolerance. The current uses of one Member form
one contiguous run per original use, in the original chain order, with no
duplicate, missing or excluded descendant rows. Their parent ranges tile the
original use span [p,q] and retain the original use orientation. The affine
station mapping to the original carrier is independently checked: a current
parent range [u,v] maps to native [(u-p)/(q-p), (v-p)/(q-p)] forward or
[1-(v-p)/(q-p), 1-(u-p)/(q-p)] reversed, and the maximum of the two exact
endpoint squared residuals bounds the whole linear restriction. The receipt
records that bound and its unchanged owner tolerance; no rounding error is
silently treated as zero. Every Member payload exposes the ordered
`source_use_mappings` (one entry per original use with its source use, carrier,
exact span, orientation, current uses, carriers and station certificates) and
the ordered `source_carriers`. The single-use `source_carrier` field is
retained for one-use Members; a multi-use Member sets it to null instead of
inventing one root. Point coordinates are checked by rational interpolation
and squared residuals against the minimum of the unchanged Attachment
tolerance and the owner's effective carrier-length tolerance. Point
Attachments remain edge-local stations on their explicit source carrier; no
global Member station is inferred for them. For an original station t on
interval [a,b], the ideal edge parameter is (t-a)/(b-a); reversed member
traversal is accounted separately. Persisted floating parameters remain
unchanged. Exact rational residuals check the persisted target point against
both the original station and the unchanged source vertex. Their bounds are
included in the receipt. At a rounded split endpoint, the ideal ratio can lie
just outside the retained child interval. It is diagnostic data; consumers use
the unchanged persisted current parameter in [0,1] and its certified
coordinate residuals. Display sampling does not establish this relation.

Ancestry is queried once per call and indexed by original carrier. Cancellation
is checked while processing Members and points, and final binding guards reprove
the unchanged model. Receipts and generated outputs use immutable exact owner
definitions, captured before serialization and checked after the last guard.
Behavioral substitutes, stale inputs, omissions and callback mutations refuse
with `GeometryError`. No geometry document schema change is introduced.
Point lineage follows the unique persisted edge replacement path, including
intermediate carriers from repeated cuts. Its iterative traversal has no
recursion-depth ceiling on valid cut chains. The earlier strict receipt retains
its zero-restriction-error requirement.

Pre-existing Junction remapping, interior Member semantics, nonlinear carriers,
unproved orientation references and unsupported surface pairs remain outside this
contract. `bounded_relation_mapping_qualified` applies only to the relations
above. Semantic mapping, beam discretization, external reference transfer and
publication remain unqualified. Consumers must independently prove their mesh
station chains, material partitions, live project references and quality gates.

Small regressions cover 3, 5 and 10 authored faces, several Members and points,
reversed traversal and shared carriers, plus independent two- and three-use
continuous boundary chains with unequal use lengths, reversed chains and unsplit
uses, mixed native orientations, repeated cuts through one chain carrier, points on multiple carriers
including endpoints, and refused malformed source/current span tiling,
ownership, descendant coverage and repeated-carrier semantics. These checks
establish the implemented geometry contract; they are not accepted
100/1,000-operand meshes, quadratic meshing, platform qualification or release
evidence.
