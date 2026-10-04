# Exact material containment of XYZ cells

`validate_material_surface_region_triangles_xyz(model, regions, face, triangles_xyz,
cancellation_check=None)` is root-exported. Obtain a fresh bound region for one
source face with `query_material_surface_regions(model, operands=(face,))`, then
pass actual cell coordinates with shape `(n, 3, 3)`. Split Q4 cells into their
chosen constituent triangles before validation.

The supported carrier is an implicit `Plane`. The function interprets stored
coefficients and ordinary input numbers as literal binary64 rationals; plain
`Fraction` input coordinates retain their exact values. It derives rational UV,
proves exact affine lifting back to every supplied XYZ corner, and certifies
each entire closed triangle against all outer and hole trims. Concave material
and holes participate in the existing exact winding and event proof. Unsupported
trim mathematics, explicit face parameterizations, non-Plane carriers, regions
spanning multiple faces and noncoplanar XYZ refuse with `GeometryError`.

For example, a Plane with `U=(6,0,0)` maps physical `x=1` to exact `u=1/6`.
Floating projection cannot represent that correspondence exactly. Consumers
must pass actual XYZ to this API instead of treating rounded projected UV as
the same physical cell. Neither sample distances nor a global mesh diameter
establish support or material truth. This API adds no tolerance or schema change.

Coordinates use document units and the face's stored support. No unit conversion,
world-coordinate transform, metadata origin or CRS reprojection is performed.
Inputs and detached owner truth are captured before callbacks; current persisted
owner bindings are checked before and after the proof. Cancellation returns no
proof and propagates caller exceptions. The query never changes source geometry
or cell arrays. A consumer must still bind its unchanged cells and owner context
through subsequent work.

Success is material containment only. It does not establish full material
coverage, current-to-source association or application reference transfer,
shared nodes, mesh quality, physical normals, solver admission or publication.
Those independent owner and consumer gates remain required. The new regression
fixtures include scaled, skewed, translated, reflected and rotated supports,
an exact nonbinary coordinate, off-plane rejection, concave crossings,
unsampled holes, unsupported carriers and callback/input binding checks.
