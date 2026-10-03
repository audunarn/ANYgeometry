# Authenticated straight planar internal stations

`query_prepared_authored_internal_stations` returns an immutable
`AuthoredInternalStations` receipt for a paired internal **Straight** edge of an
authenticated original **Plane** face. The existing owner correspondence must
be current. Both stored endpoints must lie exactly on that Plane; tolerance
coincidence, nonplanar supports and curved edges refuse.

The receipt preserves current edge and endpoint IDs, exact stored endpoint
coordinates, the two child incidences and their traversal directions, structural
coedge/face-use/sheet IDs and orientations, and the literal decomposition-seam
tag as `decomposition_seam_tag`. Parameter zero means the stored start vertex and one means the stored end
vertex, irrespective of a child's traversal direction. Caller parameter order
and duplicates are retained. Parameters accept finite rational/real values in
`[0,1]`, including NumPy integer/floating values; booleans refuse. Rational
parameters need not be representable in binary64. All rational coordinates and
parameters are stored as immutable numerator/denominator pairs.

Original UV follows the exact authenticated Plane chart. Original XYZ is its
exact lift, while current XYZ is exact interpolation of the stored endpoints.
The coefficient identity is checked with the existing owner chart machinery.
Coordinates use document/model units; no metadata conversion is applied.

`validate_prepared_authored_internal_station_coordinates` rederives the receipt
and checks copied, finite binary64 `(n,3)` coordinates against **both** exact XYZ
representations using the unchanged current edge Euclidean length tolerance.
It never snaps, moves, welds or assigns global node IDs. Geometry/source inputs
are captured before cancellation callbacks and bindings are checked finally.
Caller exceptions propagate; cancellation supplies no partial proof.

Paired incidence is not proof of a physical joint. A decomposition-seam tag is
reported literally; its absence is **not** a physical-edge classification.
Structural occurrences are scoped to those two authenticated descendants; they
do not identify or certify incident faces from other authored roots. They supply
identity/orientation visibility, not permission to discard occurrence properties.
No material membership, whole-edge approximation,
cell partition, constraint completeness, subdivision, remapping, quality or
mesh-publication permission is supplied. Consumers retain all those gates and
must revalidate after their work. Ordinary reloads and unqualified clones cannot
recreate transient owner evidence.

`examples/authored_internal_stations_handoff.py` uses the portable 4-by-4 face
cut at x=3 and proves only three source stations on its paired internal line.
The APIs and receipt type are exported from `anygeometry`. The example also
authenticates complete vertex ancestry. Its generated joint endpoints have no
original vertex ancestor; this does not exclude their original boundary-edge
references (see `PREPARED_VERTEX_PREIMAGES.md`).
