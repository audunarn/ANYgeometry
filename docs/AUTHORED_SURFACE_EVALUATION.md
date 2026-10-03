# Original support evaluation and curve stations

These development APIs use current preparation evidence and the complete
original boundary correspondence described in `MATERIAL_SURFACE_REGIONS.md`.
They keep original-domain semantics separate from rounded prepared fragments.
They do not establish an embedded partition or accept a mesh.
Positions and station XYZ use stored model coordinates in document units;
coordinate-transform metadata does not trigger conversion or reprojection.

`evaluate_prepared_authored_face(model, correspondence, parameters,
derivatives=False)` accepts copied finite real `(..., 2)` original UV arrays.
It returns new `(..., 3)` positions; `derivatives=True` returns the two
derivative arrays in original chart coordinates. The option must be Boolean.
Plane and qualified polynomial extrusion supports are eligible, including the
exact coefficient recognition of an implicit Coons extrusion. Evaluation uses
the polynomial support extension: outer loops, holes and patch membership are
not certified by evaluation. Use the separate whole-cell coverage contract.
Metric construction from these derivatives remains discretization arithmetic.

`query_prepared_authored_curve_stations(model, correspondence, edge_id,
parameters)` maps a paired internal `BezierQuadricCurve` by its source edge and
source parameters. Its first Bezier carrier must equal the original controls
and extrusion vector exactly; affine transforms and other curve families
refuse. Parameters must be real, within `[0, 1]` and exactly representable as
binary64. For example, `Fraction.from_float(0.1)` is a binary64 source station;
`Fraction(1, 10)` is a different rational parameter and refuses. Exterior
polynomial edges retain their separate exact-rational station API.

The result contains immutable rational pairs for the owner's evaluated carrier
coordinates `(t, s)`, normalized original UV, exact original polynomial XYZ at
those coordinates, and the current curve's binary64 XYZ. For original ranges
`(u0, u1)` and `(v0, v1)`, UV is `(t-u0)/(u1-u0), (s-v0)/(v1-v0)`, including
cropped or reversed ranges. The owner handles branch and fold parameterization;
the consumer must not recover it by inverting XYZ. These are original
**carrier** coordinates of the current numerical evaluation, not an assertion
of an exact algebraic intersection point or material membership.

`validate_prepared_authored_curve_station_coordinates(model, stations, xyz)`
copies finite `(n, 3)` coordinates and rederives the immutable receipt. It
checks error against both the original carrier and current curve using the
existing current-edge tolerance. Queries and assertions leave model and
supplied coordinates unchanged and reject stale, forged or changed bindings,
including changes made by callbacks. This supplies no global node identity,
curve approximation or subdivision permission. Consumers must preserve their
registered node IDs and XYZ, all physical constraints and references, and
their existing conformity, quality and resource gates.

The BQC mapping does not qualify QuadricIntersectionCurve, other carriers,
unidentified vertices, transformed branches or an entire constraint graph.
Those cases require separate owner contracts or explicit refusal. Ordinary
save/load does not restore transient preparation evidence. Schema, package
version and released artifacts remain unchanged.
