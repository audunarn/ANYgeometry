# Standalone Cylinder patch contract — candidate

This implementation is under qualification. It is not consumer activation or a
release announcement. Independent numerical review and remaining hostile-input,
platform and package gates are required before adoption.

## Scope and authority

`query_cylinder_patch(model, face_uses, *, expected_revision, policy=None,
cancellation_check=None)` qualifies exactly one active, model-bound FaceUse.
It reads the actual Part/Sheet/Face/Coedge ownership. It never constructs a
hidden ring, virtual sectors or replacement trim topology.

The supported sufficient-proof family is a regular Cylinder with rectangular
outer material and at most one strictly interior rectangular hole. Horizontal
sides use actual coaxial minor Arcs, each at most a quarter turn; axial sides use
actual Straight edges. Multiple consecutive Arcs can make a wide side. Support
sweeps extend through 15*pi/8 but remain strictly below a full period with a
tolerance-qualified gap. Either sweep/height sign is allowed. Axial trimming
does not replace the original support. A separate nonidentical parameterization,
full-period patch, nonrectangular trim, notch, multiple holes, Spline or oblique
carrier is outside this sufficient-proof scope. Refusal is not a declaration
that the source geometry is invalid.

## Evidence and coordinates

Frozen records expose only bounded immutable tuples, canonical source handles,
model UUID/revision, diagnostics, policy and digest. Successful results include
all actual boundary occurrences, native endpoint enclosures, oriented traversal
ranges, source incidences, outer/hole bounds and seeds. Non-success contains no
consumable loops or chart mapping and has an incomplete certificate.

The source support is the chart frame. Local coordinates are its u/v parameters;
physical chart coordinates are `(radius*sweep_angle*u, height*v)`. This is not
a principal-angle chart. Jacobian scales, sweep sign, height sign, loop winding
and FaceUse orientation are separate quantities. No outward normal is inferred
from a sign alone.

Principal inverse angles follow (-pi,pi], with an exact negative radial ray
represented at +pi. Cut-straddling intervals retain both principal images, join
only overlapping period-shifted images, and enumerate 17 local period choices.
Crossing 2*pi is not the same event as crossing the principal +/-pi cut.

Proof uses outward arithmetic, whole affine/harmonic carrier residuals, exact
source-circle conditioning, monotone derivative cones and separated material
bands. Sampled polygons or sampled residuals never certify whole boundaries.

## Binding and native sampling

`validate_cylinder_patch_binding(model, result, face_uses, *, expected_revision,
cancellation_check=None)` checks the digest and freshly qualifies the source.
There is no reusable validation lease or revision-only cache.

`evaluate_cylinder_patch_occurrences(model, result, requests, *, face_uses,
expected_revision, cancellation_check=None)` binds once for the bounded batch.
Each `CylinderPatchOccurrenceRequest` names an actual selected Coedge and native
Edge parameter in [0,1], regardless of Coedge traversal direction. Native owner
evaluators retain authority. Every returned row includes world point, chart
coordinates, error enclosures and a conservative residual bound. Any failed row
rejects the entire call, not just that row.

At endpoints identity keys bind to the actual Vertex; interiors bind to Edge and
exact float.hex(parameter), together with model UUID/revision and patch digest.
They are source identity keys, not mesh IDs or coordinate-based weld decisions.

## Safety and limits

Policy defaults/caps: 32 occurrences, 96 vertices, 200000 interval operations,
65536 pair tests, 4096 evaluations and 256 external incidences. Callers can only
lower them. One FaceUse and two loops are fixed limits. Evidence is bounded by
200000 nodes, 3.2MB UTF-8 and fixed nesting/tuple limits. Arithmetic uses the
existing 80-bit outward grid with bounded rational/series work.

Queries do not mutate source stores, revision, IDs, incidence or caches. Busy
transactions reject. Callback exceptions propagate unchanged, and freshness is
checked before and after callbacks. Sampling shares qualification arithmetic
work rather than resetting its budget per row. Budget exhaustion fails closed.

This contract adds no persistence schema, protocol, version or license change.
It does not supersede the separate complete-periodic Cylinder atlas contract.

## Qualification status

Focused authoring/source/carrier/material/query/binding/sampling coverage includes
all mandatory quarter, wide, hole, sign, reversal, translated/rotated and axial
trim fixtures. Independent Decimal residual checks are supplemental test oracles,
not substitutes for the production interval proof or independent source review.
Full hostile evidence/accounting review and consumer artifact qualification are
still pending. No speedup or release-readiness claim is made here.
