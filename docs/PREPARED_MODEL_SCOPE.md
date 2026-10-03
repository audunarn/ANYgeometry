# Complete persisted owner-input scope

Consumers cannot infer absent incoming references from a per-face dependency
snapshot. `query_prepared_model_scope(model)` supplies a transient owner-bound
`PreparedModelScope` containing complete original and current ANYgeometry
documents. The original is captured prospectively at the first qualified batch
application. The current document binds the complete prepared model, its UUID,
revision and content fingerprint. No schema or package-version change is made.

```python
from anygeometry import (
    from_dict, query_prepared_model_scope, validate_prepared_model_scope_binding,
)

scope = query_prepared_model_scope(prepared_model)
original_input = from_dict(scope.authored_document)
current_input = from_dict(scope.current_document)
# Inspect every persisted input; qualify or refuse unsupported interpretations.
validate_prepared_model_scope_binding(prepared_model, scope)
# Revalidate after consumer work and immediately before publishing its result.
```

Both documents cover all serialized owner inputs: vertices including isolated
ones, edges, faces and child-local properties, FaceUses/coedges, Sheets/Parts,
members and orientation references, attachments with source/target coordinates,
junctions, groups/tags, construction vertices, features, coordinate metadata,
extensions and geometry/structural replacement history. Neighbours and incoming
references are not filtered by root-face membership. Document properties decode
fresh detached dictionaries; the binding itself contains plain immutable JSON
strings. Original and current records keep their respective IDs and parameter
systems. Existing face ancestry remains available as `scope.face_preimages`.

Absence proves absence only from the respective **persisted owner document**.
External FEM project loads, boundary conditions or other application state are
outside this claim unless actually stored as owner document data. The receipt
does not certify equivalent child properties, an isolated-point surface map,
original/current attachment parameter remapping, complete embedded constraints,
material coverage, node identity or mesh-publication permission. Consumers must
qualify those separately or refuse. Decoded documents do not inherit local
preparation evidence. Public original-support/chart/correspondence APIs remain
the authorities for geometric interpretation.

Queries require committed, complete all-face preparation. Partial, stale,
legacy per-face-only and ordinary clone/load receipts refuse. Qualified
`clone_prepared_geometry` and qualified preparation updates preserve the
original snapshot. Validation rejects omitted/modified snapshot data, changed
ancestry, foreign/stale models, raw same-revision edits and callback edits.
Cancellation returns no accepted scope and queries do not modify the model.

`examples/prepared_model_scope_handoff.py` is a portable small fixture with
crossing plates, an unrelated plate, incoming member and vertex attachments,
and an isolated vertex. Its complete original document is compared against
the pre-application public serialization; its current document is compared
against the prepared serialization. It asserts no accepted mesh outcome.

Development checks cover those records, public decoding, immutable/detached
views, removed-record forgery, adversarial string equality, current-property
changes, legacy/partial scope, qualified copying, corner updates, repeat
application, cancellation and callback mutation. The extension-bearing fixture
also exposed and regressed detached cloning's previous extension-data loss.

The complete original snapshot adds storage and fingerprint hashing. Its cost
at 100/1,000 operands is unmeasured; this change makes no scaling claim and
does not renew consumed diagnostic budgets. Existing release artifacts and
ledgers remain unchanged. Large/mixed mesh acceptance remains consumer work.
