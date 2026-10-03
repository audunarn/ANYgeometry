# Literal prepared-child coverage in original coordinates

`validate_prepared_authored_face_child_triangles(model, correspondence,
current_face, triangles_uv, *, cancellation_check=None)` is root-exported.
Success returns `None` and proves that each whole **closed** triangle is inside
both the original material and the named active prepared descendant. Input is
a finite real `(n, 3, 2)` array in the **original** UV system. An empty batch
proves no cells, and still checks identity and the applicable support/trim contract.

Original-domain containment alone cannot assign a prepared child's pressure,
boundary condition or reference scope. A triangle across a split may be inside
the original face while leaving each descendant. Use this API when the reference
manifest requires a current-child scope; retain original-root associations when
the independently validated manifest is actually root-scoped.

```python
from anygeometry import validate_prepared_authored_face_child_triangles

validate_prepared_authored_face_child_triangles(
    prepared_model, original_correspondence, selected_child, original_uv_triangles,
)
# Revalidate these exact cells and the independent project/owner/reference
# manifests immediately before publishing the mesh.
```

Supported literal mappings are exact coplanar Plane supports and qualified
same-carrier Bezier extrusions. Plane origin, affine basis, scale, shear and
orientation may differ. Extrusion directrix/vector must agree exactly, and the
triangles must remain within the literal child's ranges, including reversed
ranges. Actual oriented outer and hole trims are transformed coefficientwise
into the original chart. Qualified implicit four-edge Coons extrusions use the
existing exact coefficient cancellation proof. Explicit sampled Coons surfaces,
explicit child parameterizations, unsupported algebraic trims and unresolved
support changes refuse with `GeometryError`.

Stored split-spline controls can be rounded even at a nominal dyadic split.
Their literal world coefficients may no longer lift exactly into the original
chart. The current implementation refuses those cases; it does not snap controls,
fit a tolerant carrier, or substitute an ancestral curve. Fragmented cubic,
BezierQuadricCurve and QuadricIntersectionCurve child coverage is **not qualified**
by this addition. Exact unsplit cropped extrusion coverage is exercised.

The caller input is copied before callbacks. The current child/support and its
actual paths are detached and authenticated before any cancellation callback.
Binding checks reject foreign/stale/raw-mutated data and run again on successful
completion. A transient callback edit cannot replace the captured child proof.
Queries do not edit the model. Cancellation never returns a successful proof.

This contract supplies no partition equality or uniqueness against other children,
complete triangulation, retained station/node IDs, attachment remapping, quality,
high-order certification or mesh-publication permission. Consumers retain all
those gates and the existing execution/resource limits. No schema, package
version, release artifact or numerical tolerance changes are made.

`examples/authored_child_coverage_handoff.py` supplies a portable planar fixture:
a 4-by-4 original plate cut at x=3, independently stated left/right triangles,
and a crossing triangle that original-domain validation admits while both
child validators refuse. The unit tests also exercise holes, concavity, affine
frames, stale input, cancellation, literal support changes and mutate/restore
callbacks. This is source/installed contract evidence, not accepted meshing.
