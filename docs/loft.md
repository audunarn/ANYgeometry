# Ruled loft construction

GeometryModel.loft(first_edges, second_edges, *, reverse_first=False,
reverse_second=False, first_start=0, second_start=0, flip_normal=False,
cancellation_check=None) joins ordered connected profiles into uncapped shells.
Both profiles must be open or both closed. Closed start indices select an edge
before direction reversal. Source entities are retained unchanged.

geometry.loft persists input edges and settings. loft_profile_options proposes
endpoint/closed-seam alignment; applications persist the accepted flags rather
than silently rematching on regeneration. loft_profiles_from_sketch resolves
explicit authored wires to materialized output edges.

Canonical NURBS/Bezier profiles use relative edge-length and knot anchors, not
exact continuous arc-length reparameterization. Product denominators preserve
straight world interpolation with unequal rational weights. Rational construction
is stored as double controls; rounding is not claimed as exact source identity.
Affine boundaries remain Straight edges; curved boundaries are canonical NURBS.
Existing topology-backed Coons support evaluates the ruled map: straight connectors
cancel its bilinear corner blend. Adjacent faces and closed seams share actual
edge and vertex IDs. No optional parameterization or source receipt is fabricated.

Internal canonical tensors receive exact Bernstein-sign regularity proofs and
existing projected-boundary simplicity audits. Pairwise positive-weight hull
separation allows only authored shared connectors. LoftNotCertified indicates an
unresolved sufficient proof, not a proven intersection. Difficult mappings refuse.

Circular Arc chains support closed circles and matching open angular spans,
including unequal radii, offset centers and tilted planes when the global
admission bound succeeds. Actual sketch circles comprise four quarter Arcs;
two-semicircle inputs also work. Original endpoint/via definitions are copied
or restricted as Arc geometry, never silently approximated as NURBS. All
circular loft patches use topology Coons support, the exact corrected boundary
map covered by admission. No fitted analytic support or optional display map
replaces that authority. Noncircular and mixed
Arc/other-curve chains explicitly refuse.

Normal flips preserve stable vertex/edge feature output roles while changing
face uses. Cancellation rolls back entities, features and revision; the existing
allocator policy retains monotonic high-water marks after late cancellation.
Degrees above the canonical NURBS limit, mixed open/closed profiles, branching,
disconnected and repeated-edge inputs refuse. No caps, solids or guide rails.
Geometry creation is independent of meshing and solver admission/readiness.

Storage-error admission uses positive rational weights. If r is the maximum
relative normalized-weight perturbation, normalized coefficients differ in L1
by at most 2*r/(1-r), for r<1. Each coordinate boundary-error bound is the maximum
control displacement plus that coefficient bound times the local control span.
The exact rational sum of squared coordinate bounds must be below the existing
model curve-fit tolerance squared. Endpoint snapping is included. Global world
offset does not enlarge the tolerance; unresolved representation refuses.

CurveOnSurface input profiles explicitly refuse with an actionable GeometryError.
Their internal iso-world record includes support parameter maps and numerical
slack and is not a standalone NurbsCurve conversion. Both automatic correspondence
and construction retain this boundary; source geometry and transforms are not
silently reinterpreted. Select authored standalone profile curves instead.

## Circular admission bounds

A circular profile is represented by c + U cos(theta) + V sin(theta), together
with whole-interval position error e and angular derivative error d. All stored
frame coefficients are interpreted as exact rationals for these bounds. Rational
Taylor polynomials and Lagrange remainders enclose sine and cosine; no sampled
normal test or assumed libm error factor certifies the construction. Bounds
include the topology Coons linear endpoint correction. Closed profiles use one
common 2*pi span; accepted open profiles also share one span, and differences
from their actual sweeps remain in the position and derivative bounds.

The radial extension of a boundary error has each Cartesian derivative bounded
by e+d. The complete disk homotopy therefore has a bounded Jacobian perturbation.
A fixed invertible midpoint frame normalizes positions and derivatives. A positive
symmetric Jacobian lower bound on this convex domain proves strong monotonicity
and hence global injectivity. Failure of this sufficient condition is reported
as not certified, never as proof of an intersection.

For strongly differing radii, use r(v)=(1-v)r0+v*r1 and coordinates
(X,Y)=r(v)*(x,y)/rmean. Their domain is a convex frustum. Unit-radius frames blend
with lambda=v*r1/r(v). The reference derivative perturbation is bounded by
||M^-1 Delta||F * sqrt(1/4 + (rmax/rmean)^2), since |lambda-1/2|<=1/2
and |lambda'| times the domain radius is at most rmax/rmean. Source error adds
||M^-1||F * sqrt(2*A^2+B^2), where
A=rmean*max((e0+d0)/r0,(e1+d1)/r1) and
B=rmax*(e0/r0+e1/r1). A sum below one proves strong monotonicity.
This avoids an artificial dependence on global plane orientation or radius ratio.
Exactly parallel authored planes also admit a section determinant proof.

Correspondence station reconciliation uses the owner's parameter tolerance and
never collapses two distinct stations from the same source. Paired polynomial
boundaries avoid unnecessary rational degree inflation; ordinary rotated polygon
boundaries remain Straight. Near-coincident authored stations that cannot be
resolved safely refuse explicitly.

The closed disk uses mathematical 2*pi, enclosed by the rational Machin identity
2*pi = 32*atan(1/5)-8*atan(1/239), with alternating-series remainder bounds.
The discrepancy from the stored floating common span contributes to both position
and angular derivative error. For an open angular span S<2*pi, the error vector
is linearly bridged over the remaining gap to obtain a continuous periodic disk
extension: its derivative bound is max(d,2*e/(2*pi-S)). Open spans at or above a
full turn refuse. This is a proof extension only; output geometry remains open.

Hull separation tests exact stored rational controls. Adjacent hulls may meet
only on their shared ruling; candidate planes through that ruling include the
control-induced boundary rays and their pairwise sums. Other hull pairs use
strict separating-axis tests from control chords, within-hull cross-products and
cross-hull chord pairs. Candidate generation never establishes acceptance by
itself: all controls must satisfy the exact separating inequalities. This admits
ordinary tilted face pairs with unequal edge lengths without changing their
persisted relative-length correspondence.

The parallel-plane shortcut additionally requires exact containment of every
stored ArcFrame center and both radius-scaled basis coefficients in its authored
plane. Coplanar vertices alone are insufficient. This also establishes containment
of the Coons endpoint corrections. If any check fails, admission uses the general
spacing-aware homotopy bound.

Feature regeneration permits support transitions among canonical Coons, Cylinder
and Cone only for lofts with the same complete stable edge roles, four-edge mapped
loops, no holes and no optional parameterizations. Other feature kinds retain
their existing strict support-type compatibility rule.
