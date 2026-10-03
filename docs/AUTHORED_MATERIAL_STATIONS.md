# Current material trace stations in original Plane UV

`query_prepared_authored_material_stations` authenticates a current literal
Straight trace that is exterior or paired interior in a prepared authored-root
correspondence. It returns frozen `AuthoredMaterialStations` with current
endpoint IDs, exact stored endpoints, child incidence, current source parameters,
original rational UV, current exact XYZ, the identical original Plane lift,
definition checksum and the existing current-edge Euclidean tolerance.

The owner proves an exact polynomial identity of the current LinePath on the
original Plane. There is no consumer XYZ inverse, coordinate snapping, tolerance
recognition or change of global node IDs. Parameter 0 follows the current stored
start vertex and 1 its end. Order and duplicate parameters are preserved;
rational parameters need not be representable in binary64. All coordinates and
parameters are stored as immutable numerator/denominator pairs.

This receipt deliberately differs from `AuthoredBoundaryStations`: it does not
replace ancestral parameters or their original-polynomial UV. Rounded ancestral
split intervals can give distinct UV at one current vertex. Current material
stations instead give the exact current trace in the original Plane chart. The
old API retains its original meaning and its independent rounding evidence.

Coordinate validation rederives the receipt and compares copied XYZ against
BOTH owner lifts using the unchanged edge tolerance. It neither moves points
nor certifies ancestral property remapping, material/cell containment, whole
partition coverage, physical classification, quality or mesh publication.
Original supports other than exact Plane and current curved/off-Plane traces
refuse. Fresh owner bindings, detached inputs and final definition checks guard
coercion, callbacks, wrong models, staleness and receipt tampering.

Coverage inputs now preserve explicitly supplied plain `fractions.Fraction`
entries exactly. Ordinary numeric inputs keep their prior binary64 meaning.
This affects cell UV only; stored support/trim coefficients and all tolerances
are unchanged. Exact rational partition coverage remains the existing restricted
Plane/simple straight outer-loop theorem, with no holes or mesh permission.

The portable two-Sheet fixture has a 4×4 plate and a cutter whose original frame
is `(3,-1,-1)+(0,6,0)u+(0,0,2)v`. Its oracle is
`u=(y+1)/6, v=(z+1)/2`. Current shared endpoint UV agrees exactly at 1/6 and 5/6,
while ancestral stations retain their documented differences. Rational cells
cover both roots and all eight current children; binary64-only cells need not
satisfy those exact equalities. No native mesher is run or accepted.

The receipt, station query and coordinate validator are root-exported from
`anygeometry`.
