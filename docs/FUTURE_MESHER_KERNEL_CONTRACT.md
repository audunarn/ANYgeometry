# Future mesher-kernel contract

ANYmesher reads a qualified geometry revision and produces a discretization.
It does not mutate or heal the source model.

## Read-only handoff

The kernel exposes model UUID/revision, model-bound handles, parts, sheets,
members, immutable vertex coordinates, exact curves, ordered coedges and
member-edge uses, face trims and holes, authoritative support surfaces,
optional parameterizations, attachments, junctions, non-manifold radial uses,
groups, tags, an ephemeral strict `AuditReport`, and `ChangeSet` data. Audit
status is not serialized or cached by the schema; it is valid only for the
exact model UUID, revision, and audit policy recorded by the caller.

The coordinated native-mesher API additionally provides:

```python
extract_model_closure(
    geometry,
    handles,
    include_structural_closure=True,
    include_features=False,
)
evaluate_edge_many(edge_id, parameters)
edge_tangent_many(edge_id, parameters)
evaluate_face_many(face_id, uv)
face_derivatives_many(face_id, uv)
face_normal_many(face_id, uv)
project_to_face_many(face_id, xyz, initial_uv=None)
audit_changed_region(geometry, change_set, policy=None)
find_coplanar_overlaps(geometry, *, face_ids=None,
                       changed_aabbs=None, candidate_pairs=None)
```

The evaluation names above are `GeometryModel` methods. The same names are
also available as module functions in `anygeometry.evaluation`, with
`geometry` as their first argument. `extract_model_closure` returns a
`ModelClosure`; `GeometryModel.extract_model_closure(...)` is the equivalent
bound method.

Closure extraction returns a working model, bidirectional source/working
`EntityHandle` maps, source UUID and revision, and canonical source handles.
Only complete selected structural parents and complete qualifying
attachments/junctions are included.

Vectorized evaluation returns deterministic NumPy arrays. Face derivatives
return `(du, dv)` arrays; projection returns projected points, UV coordinates,
and distances. Plane, Cylinder, Cone, RuledSurface, and CoonsSurface are
supported without a mesher dependency.

Shell/sheet T-junctions are persisted through the ordinary qualified
face/face workflow:

```python
result = query_intersection(geometry, source_face, target_face)
plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
application = apply_imprint(
    geometry, plan, policy=ConnectionIntent.CONNECT
)
```

For a classified curve intersection, the plan operation is `FACE_IMPRINT`.
Application creates or reuses the shared B-rep edge and rewrites the
participating Sheets' `FaceUse`/`Coedge` records atomically. This shared
topology—not coordinate proximity—is the connection contract consumed by the
mesher. Region intersections and unsupported curved cases fail closed with a
typed result; they are never silently converted into a shell connection.

Changed-region audit is explicitly scoped `CHANGED_REGION` and never claims
full certification. A cached full audit is reusable only for the exact model
UUID/revision and audit policy.

`write_geometry(..., certified=True)` is only a write-time validation gate.
It does not add a certification field to schema 4, so mesher handoff must carry
or recompute the exact-revision full `AuditReport` rather than infer status from
the serialized document.

## Package and document versions

The supported Python dependency remains:

```text
ANYgeometry>=0.2,<0.3
```

ANYgeometry 0.2.1 reads geometry schemas 1–4 and writes canonical schema 4.
Schemas 1–3 migrate one-way. A 0.2.0 reader intentionally rejects schema 4;
therefore package-range compatibility does not imply forward document
compatibility. ANYmesher uses the public geometry codecs and must not parse or
rewrite core schema records itself.

Schema-3 Attachments migrate with `UNVERIFIED` evidence because those records
did not persist qualification residuals or tolerance. Zero placeholder values
do not mean exactness and cannot contribute to a certified handoff. The
relationship must be requalified through `query_intersection`,
`plan_imprint`, and `apply_imprint`.

## Responsibilities

ANYgeometry owns identity, tolerance, ownership, topology, intersection
classification, connection intent, healing/imprint policy, and provenance.
ANYmesher must not:

- perform geometry healing or silently merge nodes from coordinate proximity;
- independently discover structural intersections already represented by
  attachments and junctions;
- infer `Member` identity from arbitrary edge geometry;
- reinterpret separate-part coincidence as connectivity;
- use changed-region audit as full-model certification.

ANYmesher later discretizes members into beam elements and sheets into shell
elements. Mesh entity provenance must point back through model-bound source
handles and the closure maps, never through coincident coordinates or local ID
equality.
