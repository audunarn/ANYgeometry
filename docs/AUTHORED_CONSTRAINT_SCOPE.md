# Authored-root constraint and reference inventory

The public development APIs `query_prepared_authored_constraint_scope` and
`validate_prepared_authored_constraint_scope_binding` bind a read-only inventory
to complete original/current geometry snapshots and the selected authored-root
boundary correspondences. They complement station mapping and material coverage;
they do not replace those contracts.

```python
from anygeometry import (
    query_prepared_authored_constraint_scope,
    validate_prepared_authored_constraint_scope_binding,
)

# IDs belong to the original document captured before batch preparation.
receipt = query_prepared_authored_constraint_scope(
    prepared_geometry, authored_face_ids,
    expected_revision=prepared_geometry.revision,
)
validate_prepared_authored_constraint_scope_binding(prepared_geometry, receipt)
```

`selected_root_ids`, `current_face_ids` and `outside_root_ids` identify the
selection and its literal shared-trim neighbours. Other reference-connected
owners remain visible in the complete typed records; this is not an automatic
reference-closure selector. `inventory` returns a fresh detached dictionary
decoded from the frozen `inventory_json`; `scope` retains the complete original
and current documents. `typed_inventory_complete` concerns this persisted owner
inventory only. Semantic mapping, parameter remapping, material, external
reference scope and publication qualification flags remain false.

The selected roots must be unique integer IDs with authenticated current
descendants. Every current trim trace remains visible, including interior seams,
physical interfaces, endpoints, adjacent face occurrences and neighbouring roots.
Outside shared-trim neighbours are reported explicitly; the query does not
select them or authorize a consumer to ignore them. A relationship through a
Member, Attachment or Junction requires separate reference-scope interpretation.

All original and current Member, MemberEdgeUse, Attachment and Junction records
remain explicit, including incoming and transitive relations. Their IDs and
parameters retain their respective document namespaces. A literal unchanged
record is distinguished from a changed, newly present or missing record, but
equality alone does not establish semantic equivalence or a parameter remap.
General member/point attachment remapping remains unqualified by this receipt.

Isolated vertex detection uses actual edge endpoints and curve control vertices.
`Face.corners` contains loop offsets and is never interpreted as vertex IDs.
Groups, tags, metadata, features and extensions remain opaque unqualified scope.
Absence from the complete persisted geometry documents proves nothing about
external FEM project loads, sections, supports or application references.

Revalidate after callbacks and before using the inventory. Stale, modified or
forged bindings refuse; the query does not mutate geometry or restore transient
preparation evidence after an ordinary document reload.

This is complete typed visibility, not a complete constraint-preservation proof.
The consumer must retain every relevant constraint and adjacent occurrence,
obtain supported station/remap evidence, refuse unsupported semantics and bind
external references. Whole-cell/partition coverage, shared node sequences,
source associations, resource accounting, quality gates and atomic publication
remain separate obligations. This API grants no meshing or publication permission
and introduces no schema, package version or release change. Large-model runtime
and general reference-remapping performance are unmeasured for this addition.

`examples/authored_constraint_scope_handoff.py` rebuilds a cubic wall and planar
cut, then queries the complete pair and a wall-only selection. The latter must
retain its outside neighbour in the inventory. It performs owner queries only;
it is not an accepted meshing harness.
