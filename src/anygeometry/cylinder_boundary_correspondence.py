"""Projected-boundary correspondence for authored Cylinder faces.

This is a boundary-only diagnostic certificate.  It compares the ACTUAL
authored and current trim boundary definitions of one authored Cylinder face
through ONE explicitly defined projection, with distinct ``exact``,
``certified_discrepancy`` and ``refused`` outcomes.  It is never material
conservation, interior partition coverage, reference completeness or meshing
permission; no such proof flags exist on the result.  A source/current
geometric identity certified here does not imply interior partition coverage.

Projection.  This is the **authored rational-frame projection**, defined
ONLY from the serialized ``surface`` components of the authenticated
authored face definition (origin, axis, radial_direction, radius, height,
start_angle, sweep_angle — read as exact binary rationals; never normalized,
decoded, reconstructed or replaced by the live descendant supports).  The
serialized Cylinder stores NO circumferential coefficient, so this
projection separately recomputes ``axis x radial_direction`` exactly in
rational arithmetic as part of its own definition.  The recomputed cross is
therefore NOT a stored native raw-frame coefficient and this projection is
NOT the native ``Cylinder.evaluate``/``local_uv``/atlas chart semantics; no
equivalence with them is claimed.  For a world point ``p`` with
``w = p - origin``::

    axial  a = w . axis
    radial x = w . radial_direction
    radial y = w . (axis x radial_direction)   (exact rational cross)

The projected curve is ``(theta, a)`` with ``theta = atan2(y, x)``.  It is
finite only where certified radial coordinates avoid zero, i.e. where the
exact rational ``x**2 + y**2`` is certified positive on the whole parameter
interval.  The angle branch is the principal branch ``(-pi, pi]`` with the
seam on the negative radial ray; per straight curve evidence records the
continuous lift through exact ``(cross, dot)`` pairs of endpoint radial
vectors, never by evaluating ``atan2``.  For arcs the continuous native
identity is anchored identically (same edge and start/via/end vertex ids
with exactly equal rational positions); no numeric winding, closure or
loop claim is made.  No witness sample is truth, no constant Jacobian is
assumed and no nominal generator angle is substituted: a Straight trim
``(1, eps*t, t)`` genuinely projects to the nonconstant ``(atan(eps*t), t)``.

Families.  Straight trims keep their native affine parameter function through
the two stored vertices; comparisons are exact rational polynomial decisions.
Arc trims keep their native three-point arc definition; the native parameter
function is transcendental and is never evaluated.  An unsplit arc is decided
by exact native definition identity PLUS an exact whole-circle finiteness
certificate: the whole native arc lies on the circumcircle of its three
authored points, which lies in their exact plane, and the projection is
undefined exactly on the radial-null line through the authored origin; exact
plane/line/circle decisions certify that the whole circle avoids that line
(``exact`` only then).  When the whole circle cannot certify avoidance the
projected correspondence is refused, with the native definition identity
retained as separate evidence.

Arc split ancestry is a bounded geometric-subarc-image contract, never a
source-parameter restriction.  The recorded numeric split parameter and
composed interval are authenticated provenance only
(``parameter_mapping_qualified`` is False on every record); no exact
ancestor source-parameter restriction is claimed from same-circle or
oriented-wedge evidence.  A split arc root is ``exact`` only when every
child's three actual points lie EXACTLY on the ancestor's exact circumcircle
(common plane/circle), each child's own whole-circle finiteness is certified,
the directed use agrees with the original authored EdgeUse direction, and an
independently proven COMPLETE GEOMETRIC TILING of the ancestor's directed
span holds: authenticated shared anchors (exact rational position chaining
from the ancestor's start to its end), strict directed ordering by exact
wedge tests (no gap, no overlap) and complete ancestor span.  Numeric
interval tiling is never a shortcut for geometric equality.  An ``enclosed``
child (nonzero certified bounds within the recorded split tolerance, plane
distance included) refuses the root as enclosed-within-tolerance-not-exact;
a refused child refuses the root with its recorded reason.  Enclosure alone
does not prove different curves and never grants mesh or material permission.

Orientation.  Exact boundary correspondence is oriented: a straight
descendant's directed source interval sign combined with its current EdgeUse
direction must equal the original authored EdgeUse direction, an unsplit
arc must keep the authored use direction, and a split arc child's stored
direction relative to the ancestor's, combined with its current EdgeUse
direction, must equal the original authored EdgeUse direction.  Native
parameter identity alone is not oriented boundary equality; a reversed
traversal refuses.
"""
from dataclasses import dataclass
from fractions import Fraction
import json
import math

from .definition_binding import definition_checksum
from .edge_subcurve_preimages import (
    ArcEdgeAncestor, ArcEdgeDefinition, ArcSubcurvePreimage,
    EdgeSubcurvePreimage, PolynomialEdgeAncestor, PolynomialEdgeDefinition,
    PreparedEdgeSubcurvePreimages, query_prepared_edge_subcurve_preimages,
    validate_prepared_edge_subcurve_preimages_binding,
    _arc_circle, _arc_positions)
from .curves import Arc
from .errors import GeometryError
from .prepared_face_preimages import (
    query_prepared_authored_face_definition, query_prepared_face_preimages,
    validate_prepared_face_preimages_binding)


def _check(callback):
    if callback is not None and callback('cylinder boundary correspondence'):
        raise GeometryError('cylinder boundary correspondence cancelled')


def _pack(value):
    value = Fraction(value)
    return value.numerator, value.denominator


def _unpack(value):
    return Fraction(*value)


def _pack_point(point):
    return tuple(_pack(float(x)) for x in point)


def _point(controls):
    return tuple(_unpack(x) for x in controls)


def _rational_position(position):
    return tuple(Fraction(float(x)) for x in position)


def _sqrt_ceiling(value):
    """Rational upper bound q with q*q >= value (value a non-negative rational)."""
    value = Fraction(value)
    if value < 0:
        raise GeometryError('cylinder boundary correspondence needs a non-negative bound')
    root = math.isqrt(value.numerator * value.denominator)
    q = Fraction(root, value.denominator)
    if q * q < value:
        q += Fraction(1, value.denominator)
    return q


def _quadratic_min(alpha, beta, gamma):
    """Exact minimum of alpha*s^2 + beta*s + gamma on [0, 1]."""
    values = [gamma, alpha + beta + gamma]
    if alpha > 0:
        vertex = -beta / (2 * alpha)
        if 0 <= vertex <= 1:
            values.append(gamma - beta * beta / (4 * alpha))
    return min(values)


@dataclass(frozen=True, slots=True)
class CylinderProjectionFrame:
    """Authored rational-frame projection inputs; components are packed rationals.

    ``rational_cross`` is the exact rational cross ``axis x radial_direction``
    recomputed by this projection from the serialized components.  The
    serialized Cylinder stores no circumferential coefficient, so this is not
    a stored native raw-frame coefficient and carries no native
    evaluate/local_uv/atlas semantics.
    """
    origin: tuple
    axis: tuple
    radial: tuple
    rational_cross: tuple
    radius: tuple
    height: tuple
    start_angle: tuple
    sweep_angle: tuple


def _frame(surface):
    if not isinstance(surface, dict) or surface.get('type') != 'cylinder':
        raise GeometryError('cylinder boundary correspondence needs an authored cylinder support')
    origin = _rational_position(surface['origin'])
    axis = _rational_position(surface['axis'])
    radial = _rational_position(surface['radial_direction'])
    cross = tuple(axis[(i + 1) % 3] * radial[(i + 2) % 3]
                  - axis[(i + 2) % 3] * radial[(i + 1) % 3] for i in range(3))
    return CylinderProjectionFrame(tuple(_pack(x) for x in origin), tuple(_pack(x) for x in axis),
        tuple(_pack(x) for x in radial), tuple(_pack(x) for x in cross),
        _pack(float(surface['radius'])), _pack(float(surface['height'])),
        _pack(float(surface['start_angle'])), _pack(float(surface['sweep_angle'])))


@dataclass(frozen=True, slots=True)
class _RadialAffine:
    """Affine frame coordinates of one native straight parameter function."""
    a0: Fraction
    da: Fraction
    x0: Fraction
    dx: Fraction
    y0: Fraction
    dy: Fraction


def _radial_affine(frame, controls):
    """Affine coordinates of p(s) = P0 + s*(P1 - P0) in the rational frame."""
    first, second = controls
    origin = tuple(_unpack(x) for x in frame.origin)
    axis = tuple(_unpack(x) for x in frame.axis)
    radial = tuple(_unpack(x) for x in frame.radial)
    rational_cross = tuple(_unpack(x) for x in frame.rational_cross)
    delta = tuple(b - a for a, b in zip(first, second))
    base = tuple(a - o for a, o in zip(first, origin))
    return _RadialAffine(
        sum((w * v for w, v in zip(base, axis)), Fraction(0)),
        sum((w * v for w, v in zip(delta, axis)), Fraction(0)),
        sum((w * v for w, v in zip(base, radial)), Fraction(0)),
        sum((w * v for w, v in zip(delta, radial)), Fraction(0)),
        sum((w * v for w, v in zip(base, rational_cross)), Fraction(0)),
        sum((w * v for w, v in zip(delta, rational_cross)), Fraction(0)))


def _radial_squared_min(affine):
    """Certified exact minimum of x(s)^2 + y(s)^2 on [0, 1]."""
    return _quadratic_min(affine.dx * affine.dx + affine.dy * affine.dy,
                          2 * (affine.x0 * affine.dx + affine.y0 * affine.dy),
                          affine.x0 * affine.x0 + affine.y0 * affine.y0)


def _value(affine, parameter):
    s = Fraction(parameter)
    return (affine.a0 + s * affine.da, affine.x0 + s * affine.dx, affine.y0 + s * affine.dy)


def _sub3(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _dot3(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross3(a, b):
    return (a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0])


@dataclass(frozen=True, slots=True)
class ArcFinitenessEvidence:
    """Exact whole-circle avoidance of the radial-null line; packed rationals.

    ``separation_squared`` is ``|q - center|**2 - radius**2`` for the
    line-meets-plane case (q the unique intersection of the radial-null line
    with the arc's plane; avoidance iff it is nonzero) and
    ``distance(center, line)**2 - radius**2`` for the line-in-plane case
    (avoidance iff positive); ``None`` when the line is parallel to and
    disjoint from the plane, where avoidance is structural.
    """
    center: tuple
    radius_squared: tuple
    case: str
    separation_squared: tuple | None


def _arc_finiteness(frame, positions):
    """Exact whole-circle finiteness certificate for one three-point arc.

    The whole native arc lies on the circumcircle of its three authored
    points, which lies in their exact plane.  The projection is undefined
    exactly on the radial-null line ``{origin + t*(radial x rational_cross)}``
    through the authored origin (the unique line on which both radial
    coordinates vanish).  The whole circle avoids that line iff the line
    misses the circle, decided by exact rational plane/line/circle algebra
    with no witness samples.  Whole-circle avoidance certifies finiteness of
    the projection on the entire native arc; when the whole circle cannot
    certify, the arc is refused (a partial arc might still avoid the line,
    but that is not claimed).

    Returns ``(certified, evidence, reason)``.
    """
    origin = tuple(_unpack(x) for x in frame.origin)
    radial = tuple(_unpack(x) for x in frame.radial)
    rational_cross = tuple(_unpack(x) for x in frame.rational_cross)
    if not any(radial) or not any(rational_cross):
        return False, None, 'arc finiteness uncertified: degenerate authored radial frame'
    first, via, last = positions
    u = _sub3(via, first)
    v = _sub3(last, first)
    uu, vv, uv = _dot3(u, u), _dot3(v, v), _dot3(u, v)
    determinant = uu * vv - uv * uv
    if determinant == 0:
        return False, None, 'arc finiteness uncertified: collinear arc points'
    alpha = vv * (uu - uv) / (2 * determinant)
    beta = uu * (vv - uv) / (2 * determinant)
    center = tuple(first[i] + alpha * u[i] + beta * v[i] for i in range(3))
    radius_squared = _dot3(_sub3(center, first), _sub3(center, first))
    evidence = ArcFinitenessEvidence(tuple(_pack(x) for x in center),
        _pack(radius_squared), 'line_meets_plane', None)
    normal = _cross3(u, v)
    direction = _cross3(radial, rational_cross)
    plane_dot = _dot3(direction, normal)
    offset = _dot3(_sub3(first, origin), normal)
    if plane_dot != 0:
        station = offset / plane_dot
        point = tuple(origin[i] + station * direction[i] for i in range(3))
        separation = _dot3(_sub3(point, center), _sub3(point, center)) - radius_squared
        evidence = ArcFinitenessEvidence(evidence.center, evidence.radius_squared,
            'line_meets_plane', _pack(separation))
        if separation != 0:
            return True, evidence, None
        return False, evidence, 'arc finiteness uncertified: circle tangent to radial-null line'
    if offset != 0:
        return True, ArcFinitenessEvidence(evidence.center, evidence.radius_squared,
            'line_parallel_to_plane', None), None
    offset_vector = _sub3(center, origin)
    distance_squared = _dot3(offset_vector, offset_vector) \
        - _dot3(offset_vector, direction) ** 2 / _dot3(direction, direction)
    separation = distance_squared - radius_squared
    evidence = ArcFinitenessEvidence(evidence.center, evidence.radius_squared,
        'line_in_plane', _pack(separation))
    if separation > 0:
        return True, evidence, None
    return False, evidence, 'arc finiteness uncertified: circle meets radial-null line'


@dataclass(frozen=True, slots=True)
class ProjectedStraightChild:
    """One authenticated current descendant of a straight root."""
    edge_id: int
    # Directed source interval in root orientation; second < first means the
    # child traverses the source interval backwards.
    interval: tuple
    use_forward: bool
    # Owner-certified exact whole-interval world residual (packed rational).
    squared_residual: tuple
    outcome: str
    # Exact sup |delta axial| on [0, 1] when the projected curves differ.
    axial_error_bound: tuple | None
    # Rational witness parameter and exact rational LOWER bound on the
    # angular difference; a whole-curve angular upper bound is explicitly
    # not certified.
    angular_witness: tuple | None
    angular_error_lower_bound: tuple | None
    opposite_directions: bool


@dataclass(frozen=True, slots=True)
class ProjectedArcChild:
    """Unsplit current arc edge identical to its authored definition."""
    edge_id: int
    use_forward: bool
    start_vertex: int
    via_vertex: int
    end_vertex: int


@dataclass(frozen=True, slots=True)
class ProjectedArcSplitChild:
    """One authenticated current arc descendant of a split arc root.

    ``interval`` is authenticated NUMERIC PROVENANCE only;
    ``parameter_mapping_qualified`` is always False and no field claims the
    child equals the ancestor restricted to that interval.  ``anchor_bounds``
    holds certified exact-rational (lower, upper) distance intervals from each
    defining point to the ancestor's exact circumcircle with the plane
    distance included; ``whole_circle_bound`` is the certified upper bound
    over the child's whole circumcircle.
    """
    edge_id: int
    use_forward: bool
    start_vertex: int
    via_vertex: int
    end_vertex: int
    classification: str
    direction: str
    anchor_bounds: tuple
    whole_circle_bound: tuple | None
    tolerance: tuple | None
    interval: tuple
    parameter_mapping_qualified: bool


def _directed_wedge(normal, center, first, second, probe):
    """True iff probe lies STRICTLY inside the directed arc from first to
    second (counterclockwise around normal), with span in (0, 2*pi).

    Exact rational cross/dot sign logic; the strictly-below-pi, antipodal and
    strictly-above-pi cases are decided separately, and the endpoints are
    never strictly inside.  Returns None when first and second coincide."""
    u = _sub3(first, center)
    v = _sub3(second, center)
    w = _sub3(probe, center)
    if u == v:
        return None
    cross_ab = _dot3(_cross3(u, v), normal)
    if cross_ab > 0:  # directed span strictly below pi
        return _dot3(_cross3(u, w), normal) > 0 and _dot3(_cross3(v, w), normal) < 0
    if cross_ab < 0:  # directed span strictly above pi
        inside_complement = (_dot3(_cross3(v, w), normal) > 0
                             and _dot3(_cross3(u, w), normal) < 0)
        return not inside_complement and w != u and w != v
    # Antipodal anchors: the directed span is exactly pi.
    return _dot3(_cross3(u, w), normal) > 0


def _arc_geometric_tiling(positions, records):
    """Complete geometric tiling of the ancestor's directed span, or a reason.

    Exact children only, and GEOMETRY only: the children's images are directed
    subarcs of the ancestor's circumcircle containing their via points,
    chained through authenticated shared anchors by exact rational position
    equality from the ancestor's start to its end.  Every child anchor lies
    on the ancestor's ORIGINAL directed span (closed only at the ancestor's
    own start/end anchors, strictly inside otherwise), and the chain advances
    in global monotonic order from the original start: each shared anchor
    lies strictly inside the directed wedge from that original start to the
    following end, so each child's selected via-containing arc equals the
    corresponding consecutive subinterval of that span — no gap, no overlap,
    no wrap-around past the end (a 450-degree chain on a 90-degree ancestor
    is rejected).  Numeric intervals are never consulted for geometric
    equality.
    """
    circle = _arc_circle(positions)
    if circle is None:
        return 'arc split tiling uncertified: degenerate ancestor'
    center, _, normal = circle
    first, via, last = positions
    if first == last:
        return 'arc split tiling uncertified: degenerate ancestor span'
    if _directed_wedge(normal, center, first, last, via):
        counterclockwise = True
    elif _directed_wedge(normal, center, last, first, via):
        counterclockwise = False
    else:
        return 'arc split tiling uncertified: degenerate ancestor'
    def wedge(a, b, probe):
        if counterclockwise:
            return _directed_wedge(normal, center, a, b, probe)
        return _directed_wedge(normal, center, b, a, probe)
    images = []
    for record in records:
        child_first, child_via, child_last = _arc_positions(record.current_definition)
        if wedge(child_first, child_last, child_via):
            anchors = (child_first, child_last)
        elif wedge(child_last, child_first, child_via):
            anchors = (child_last, child_first)
        else:
            return 'arc split tiling uncertified: child via outside a directed subarc'
        # Root global span containment: every child anchor lies on the
        # ancestor's ORIGINAL directed span — closed only at the ancestor's
        # own start/end anchors, strictly inside otherwise.  A chain whose
        # anchors leave the span (for example a 450-degree wrap on a
        # 90-degree ancestor) is rejected before any chaining.
        if anchors[0] != first and not wedge(first, last, anchors[0]):
            return 'arc split tiling conflict: child anchor outside ancestor span'
        if anchors[1] != last and not wedge(first, last, anchors[1]):
            return 'arc split tiling conflict: child anchor outside ancestor span'
        images.append((anchors, record))
    by_first = {}
    for anchors, record in images:
        by_first.setdefault(anchors[0], []).append((anchors, record))
    starts = by_first.get(first, [])
    if not starts:
        return 'arc split tiling incomplete'
    if len(starts) > 1:
        return 'arc split tiling conflict'
    chain = [starts[0]]
    used = {starts[0][1].edge_id}
    while True:
        anchors, record = chain[-1]
        if anchors[1] == last:
            break
        following = [item for item in by_first.get(anchors[1], [])
                     if item[1].edge_id not in used]
        if not following:
            return 'arc split tiling incomplete'
        if len(following) > 1:
            return 'arc split tiling conflict'
        following = following[0]
        # Strict GLOBAL advancement from the original start: local wedges
        # can wrap on a major arc and admit a full extra revolution even
        # when every anchor lies inside the original span.
        if not wedge(first, following[0][1], anchors[1]):
            return 'arc split tiling conflict'
        used.add(following[1].edge_id)
        chain.append(following)
    if len(chain) != len(images):
        return 'arc split tiling conflict'
    return None


@dataclass(frozen=True, slots=True)
class ProjectedBoundaryRoot:
    root_edge_id: int
    loop_index: int
    forward: bool
    family: str
    outcome: str
    reason: str | None
    # Straight-only exact evidence, packed rationals.
    constant_angle: bool | None
    lift_cross: tuple | None
    lift_dot: tuple | None
    seam_crossed: bool | None
    endpoint_radial: tuple | None
    radial_min_squared: tuple | None
    # Arcs: whole-circle finiteness is certified exactly (plane/line/circle
    # algebra); the transcendental native parameter function is not evaluated.
    finiteness_certified: bool | None
    # Arcs: unsplit native definition identity (same edge id, curve type,
    # start/via/end vertex ids and exact rational positions) certified
    # separately from the projected finiteness outcome.
    native_identity_certified: bool | None
    arc_finiteness: ArcFinitenessEvidence | None
    children: tuple


@dataclass(frozen=True, slots=True)
class CylinderBoundaryCorrespondence:
    face_preimages: object
    edge_preimages: object
    authored_definition: object
    descendants: tuple[int, ...]
    frame: CylinderProjectionFrame
    boundary_roots: tuple
    interior_incidence: tuple
    unaccounted_boundary_edges: tuple
    outcome: str
    refusals: tuple


def _root_record(root, loop_index, forward, family, outcome, reason, evidence, children):
    return ProjectedBoundaryRoot(root, loop_index, forward, family, outcome, reason,
        evidence.get('constant_angle'), evidence.get('lift_cross'), evidence.get('lift_dot'),
        evidence.get('seam_crossed'), evidence.get('endpoint_radial'),
        evidence.get('radial_min_squared'), evidence.get('finiteness_certified'),
        evidence.get('native_identity_certified'), evidence.get('arc_finiteness'), children)


def _straight_root_evidence(frame, controls):
    affine = _radial_affine(frame, controls)
    x0, y0 = affine.x0, affine.y0
    x1, y1 = x0 + affine.dx, y0 + affine.dy
    cross = x0 * y1 - y0 * x1
    dot = x0 * x1 + y0 * y1
    seam = False
    if affine.dy != 0:
        station = -y0 / affine.dy
        if 0 <= station <= 1 and (x0 + station * affine.dx) < 0:
            seam = True
    elif y0 == 0:
        # The segment lies on the radial axis; it touches the seam ray
        # wherever the axial radial coordinate is negative.
        seam = min(x0, x0 + affine.dx) < 0
    return {
        'constant_angle': cross == 0 and dot > 0,
        'lift_cross': _pack(cross),
        'lift_dot': _pack(dot),
        'seam_crossed': seam,
        'endpoint_radial': (_pack(x0), _pack(y0), _pack(x1), _pack(y1)),
        'radial_min_squared': _pack(_radial_squared_min(affine)),
        'finiteness_certified': True,
    }


def _compare_straight(frame, root_controls, interval, child_controls):
    """Exact projected comparison of one child against its source restriction.

    The decision is an exact rational polynomial identity test: the axial
    difference must vanish identically and the radial direction functions
    must coincide with a strictly positive dot product on [0, 1].
    """
    lower, upper = interval
    # Restriction of the native source function to the directed interval.
    first = tuple(a + lower * (b - a) for a, b in zip(root_controls[0], root_controls[1]))
    second = tuple(a + upper * (b - a) for a, b in zip(root_controls[0], root_controls[1]))
    restriction = (first, second)
    source = _radial_affine(frame, restriction)
    child = _radial_affine(frame, child_controls)
    axial_zero = (source.a0 == child.a0) and (source.da == child.da)
    f0 = source.x0 * child.y0 - source.y0 * child.x0
    f1 = source.x0 * child.dy + source.dx * child.y0 - source.y0 * child.dx - source.dy * child.x0
    f2 = source.dx * child.dy - source.dy * child.dx
    g0 = source.x0 * child.x0 + source.y0 * child.y0
    g1 = source.x0 * child.dx + source.dx * child.x0 + source.y0 * child.dy + source.dy * child.y0
    g2 = source.dx * child.dx + source.dy * child.dy
    if axial_zero and f0 == 0 and f1 == 0 and f2 == 0 and _quadratic_min(g2, g1, g0) > 0:
        return 'exact', {}
    evidence = {'axial_error_bound': None, 'angular_witness': None,
                'angular_error_lower_bound': None, 'opposite_directions': False}
    if not axial_zero:
        delta_a0 = source.a0 - child.a0
        delta_da = source.da - child.da
        evidence['axial_error_bound'] = _pack(max(abs(delta_a0), abs(delta_a0 + delta_da)))
    if f0 != 0 or f1 != 0 or f2 != 0:
        if f0 != 0:
            witness = Fraction(0)
        elif f0 + f1 + f2 != 0:
            witness = Fraction(1)
        else:
            witness = Fraction(1, 2)
        _, sx, sy = _value(source, witness)
        _, cx, cy = _value(child, witness)
        cross_value = f0 + witness * f1 + witness * witness * f2
        product = _sqrt_ceiling((sx * sx + sy * sy) * (cx * cx + cy * cy))
        evidence['angular_witness'] = _pack(witness)
        evidence['angular_error_lower_bound'] = _pack(abs(cross_value) / product)
        return 'certified_discrepancy', evidence
    minimum = _quadratic_min(g2, g1, g0)
    if minimum == 0:
        # Parallel radial directions with a zero dot somewhere would force a
        # radial zero; finiteness excludes it, so this is undefinable input.
        return 'refused', {'reason': 'radial zero in projected comparison'}
    if minimum < 0:
        # Parallel radial directions with a negative dot minimum: the
        # directions are opposite at the minimizing station and the angular
        # difference there is exactly pi; 3 < pi is a truthful rational bound.
        if g2 > 0:
            station = -g1 / (2 * g2)
            if not 0 <= station <= 1:
                station = Fraction(0) if g0 <= g0 + g1 + g2 else Fraction(1)
        else:
            station = Fraction(0) if g0 <= g0 + g1 + g2 else Fraction(1)
        evidence['angular_witness'] = _pack(station)
        evidence['opposite_directions'] = True
        evidence['angular_error_lower_bound'] = _pack(Fraction(3))
    return 'certified_discrepancy', evidence


def _tile_spans(children):
    """Exact partition check of directed source intervals over ``[0, 1]``.

    Direction is deliberately dropped here; this checks coverage only.
    Oriented agreement with the original authored EdgeUse direction is
    enforced separately per child and cannot be inferred from this tiling.
    """
    spans = []
    for edge, first, second in children:
        if first == second:
            return 'conflict'
        spans.append((min(first, second), max(first, second)))
    spans.sort()
    previous = Fraction(0)
    for first, second in spans:
        if first < previous:
            return 'conflict'
        if first > previous:
            return 'incomplete'
        previous = second
    if previous != 1:
        return 'incomplete'
    return None


def _detach_current_boundary(model, descendants):
    """Detach the live current descendant loops and edge/vertex definitions.

    Returns ``(loops, edges)``: ``loops`` maps each descendant face id to a
    tuple of per-use ``(edge_id, forward)`` pairs (outer loop first), and
    ``edges`` maps each loop edge id to a detached definition snapshot —
    ``('arc', start, via, end, positions)`` with the exact rational positions
    of the start/via/end vertices for an Arc, ``('nonarc',)`` for any other
    active edge, or ``None`` for an id with no active edge.  Every captured
    value is a fresh tuple of plain ints, bools or exact rationals; nothing
    aliases live arrays, entities or mutable receipt objects, so a callback
    that later temporarily mutates the live model can never contaminate a
    derivation driven by this snapshot.
    """
    loops = {}
    for face in descendants:
        current = model.faces[face]
        loops[face] = tuple(tuple((use.edge, use.forward) for use in loop)
                            for loop in (current.loop, *current.holes))
    edges = {}
    for face_loops in loops.values():
        for loop in face_loops:
            for edge, _ in loop:
                if edge in edges:
                    continue
                entity = model.edges[edge] if edge in model.edges else None
                if entity is None:
                    edges[edge] = None
                elif isinstance(entity.curve, Arc):
                    ids = (entity.start, entity.curve.via_vertex, entity.end)
                    edges[edge] = ('arc', entity.start, entity.curve.via_vertex,
                        entity.end, tuple(_rational_position(model.vertices[vertex].position)
                                          for vertex in ids))
                else:
                    edges[edge] = ('nonarc',)
    return loops, edges


def _detach_edge_preimage_binding(binding):
    """Independent immutable value snapshot of one owner edge-preimage binding.

    The owner receipt's records are data-only frozen dataclasses, but they
    are shared with the owner registry and remain mutable in principle
    through ``object.__setattr__``.  Every record — ancestor and current
    definitions, intervals, error fields, bounds and aliases — is rebuilt
    as fresh frozen dataclasses over fresh tuples, so the snapshot never
    aliases a registry record object.  The derivation and the returned
    proof read only this snapshot; a callback that temporarily mutates and
    later restores the live registry records can never contaminate the
    certificate, and the returned proof shares no record object with the
    owner registry.
    """
    def definition(value):
        return PolynomialEdgeDefinition(value.edge_id, value.start, value.end,
            tuple(tuple(point) for point in value.controls), value.checksum)

    def record(value):
        return EdgeSubcurvePreimage(value.edge_id,
            PolynomialEdgeAncestor(value.ancestor.model_id, value.ancestor.revision,
                value.ancestor.source_checksum, definition(value.ancestor.definition)),
            tuple(tuple(bound) for bound in value.interval),
            definition(value.current_definition),
            tuple(tuple(point) for point in value.error_controls),
            tuple(tuple(bound) for bound in value.coordinate_bounds),
            tuple(value.squared_distance_bound),
            tuple(value.tolerance) if value.tolerance is not None else None)

    def arc_definition(value):
        return ArcEdgeDefinition(value.edge_id, value.start, value.via, value.end,
            tuple(tuple(point) for point in value.positions), value.checksum)

    def arc_record(value):
        return ArcSubcurvePreimage(value.edge_id,
            ArcEdgeAncestor(value.ancestor.model_id, value.ancestor.revision,
                value.ancestor.source_checksum, arc_definition(value.ancestor.definition)),
            tuple(value.local_split_parameter) if value.local_split_parameter is not None else None,
            tuple(tuple(bound) for bound in value.interval),
            arc_definition(value.current_definition),
            value.classification, value.reason, value.direction,
            tuple(tuple(pair) for pair in value.anchor_bounds),
            tuple(value.whole_circle_bound) if value.whole_circle_bound is not None else None,
            tuple(value.tolerance) if value.tolerance is not None else None,
            value.parameter_mapping_qualified)

    return PreparedEdgeSubcurvePreimages(binding.model_id, binding.revision,
        binding.source_checksum, tuple(record(row) for row in binding.records),
        tuple(binding.unavailable_edge_ids), tuple(binding.coverage),
        tuple(record(row) for row in binding.alias_records),
        tuple(arc_record(row) for row in binding.arc_records),
        tuple(arc_record(row) for row in binding.arc_alias_records))


def query_prepared_cylinder_boundary_correspondence(model, authored_face_id, *,
        expected_revision=None, cancellation_check=None):
    """Compare authored and current projected trim boundaries of one face.

    Consumes the existing authored-face definition, the complete descendant
    scope and the polynomial split ancestry owners; caller-provided arrays
    are never trusted.  All live current-model inputs — the descendant face
    loops and incidence, the active edge/Arc vertex definitions and the
    complete owner edge-preimage binding with all its records and aliases —
    are detached into immutable snapshots BEFORE the first callback, and the
    derivation uses only those snapshots: a callback that temporarily
    mutates and later restores the live model or the owner registry
    therefore either causes a typed owner refusal or yields exactly the
    unpolluted entry result, never a contaminated certificate.  Owner
    receipts, the revision, the authored document identity and the
    supporting edge-preimage entry digest are revalidated after the last
    callback, and the returned proof shares no record object with the owner
    registry.  Per boundary root the outcome is ``exact`` (whole curve
    parameter correspondence proved with directed traversal agreement),
    ``certified_discrepancy`` (projected curves certified different with
    exact rational bounds) or ``refused`` (missing authentication or
    unsupported mathematics, reason recorded).  This is not a material,
    partition, reference, meshing, loop-winding or closure certificate.
    """
    original = query_prepared_authored_face_definition(model, authored_face_id,
        expected_revision=expected_revision)
    faces = query_prepared_face_preimages(model, expected_revision=expected_revision)
    descendants = tuple(dict(faces.face_descendants)[int(authored_face_id)])
    payload = json.loads(original.definition_json)
    frame = _frame(payload['face'].get('surface'))
    source_face = payload['face']
    original_loops = (source_face['loop'], *source_face['holes'])
    original_edges = {row['id']: row for row in payload['edges']}
    vertices = {row['id']: row for row in payload['vertices']}
    root_ids = [edge for loop in original_loops for edge, _ in loop]
    if len(set(root_ids)) != len(root_ids):
        raise GeometryError('cylinder boundary correspondence has repeated original edge occurrence')

    controls = {}
    arc_definitions = {}
    for edge in root_ids:
        curve = original_edges[edge]['curve']
        if curve['type'] == 'straight':
            ids = (original_edges[edge]['start'], original_edges[edge]['end'])
            controls[edge] = tuple(_rational_position(vertices[i]['position']) for i in ids)
        elif curve['type'] == 'arc':
            start = original_edges[edge]['start']
            end = original_edges[edge]['end']
            ids = (start, curve['via_vertex'], end)
            arc_definitions[edge] = (start, curve['via_vertex'], end,
                tuple(_rational_position(vertices[i]['position']) for i in ids))

    # Detach every live current input before the first callback: the derived
    # certificate can only ever reflect this immutable entry snapshot.
    current_loops, current_edges = _detach_current_boundary(model, descendants)
    edges = query_prepared_edge_subcurve_preimages(model, expected_revision=expected_revision)
    # The owner receipt's records are shared with the owner registry and
    # mutable in principle through object.__setattr__.  Detach the complete
    # binding into an independent immutable value snapshot before any
    # further callback and pin the live entry digest; the derivation and
    # the returned proof use only the detached copies.
    edge_preimages = _detach_edge_preimage_binding(edges)
    entry_edge_digest = definition_checksum(edges)
    _check(cancellation_check)

    # Authenticated polynomial ancestry per straight root of this face only.
    ancestry = {}
    for row in (*edge_preimages.records, *edge_preimages.alias_records):
        _check(cancellation_check)
        ancestor = row.ancestor
        root = ancestor.definition.edge_id
        if root not in controls:
            continue
        if (ancestor.model_id != original.model_id or ancestor.revision != original.revision
                or ancestor.source_checksum != original.source_checksum):
            continue
        definition = original_edges[root]
        ancestor_controls = tuple(_point(point) for point in ancestor.definition.controls)
        if (ancestor_controls != controls[root] or ancestor.definition.start != definition['start']
                or ancestor.definition.end != definition['end']):
            raise GeometryError('cylinder boundary correspondence original polynomial changed')
        ancestry.setdefault(root, []).append(row)

    # Authenticated arc ancestry per arc root of this face only.  The ancestor
    # definition must match the authored arc definition exactly (vertex ids
    # and exact rational positions); anything else is a changed original.
    arc_ancestry = {}
    for row in (*edge_preimages.arc_records, *edge_preimages.arc_alias_records):
        _check(cancellation_check)
        ancestor = row.ancestor
        root = ancestor.definition.edge_id
        if root not in arc_definitions:
            continue
        if (ancestor.model_id != original.model_id or ancestor.revision != original.revision
                or ancestor.source_checksum != original.source_checksum):
            continue
        start, via, end, positions = arc_definitions[root]
        if (ancestor.definition.start != start or ancestor.definition.via != via
                or ancestor.definition.end != end
                or _arc_positions(ancestor.definition) != positions):
            raise GeometryError('cylinder boundary correspondence original arc changed')
        arc_ancestry.setdefault(root, []).append(row)

    incidence = {}
    for face in descendants:
        _check(cancellation_check)
        for loop in current_loops[face]:
            for edge, use_forward in loop:
                incidence.setdefault(edge, []).append((face, use_forward))
    interior = []
    boundary_uses = {}
    for edge, uses in sorted(incidence.items()):
        _check(cancellation_check)
        if len(uses) == 2 and uses[0][1] != uses[1][1] and uses[0][0] != uses[1][0]:
            interior.append((edge, tuple(uses)))
            continue
        if len(uses) != 1:
            raise GeometryError('cylinder boundary correspondence has ambiguous internal incidence')
        boundary_uses[edge] = uses[0]

    roots, refusals, unaccounted, claimed = [], [], [], {}
    for loop_index, loop in enumerate(original_loops):
        for root, forward in loop:
            _check(cancellation_check)
            curve = original_edges[root]['curve']
            if curve['type'] == 'straight':
                record = _straight_root(frame, controls, ancestry, boundary_uses,
                                        root, loop_index, forward)
            elif curve['type'] == 'arc':
                record = _arc_root(frame, current_edges, arc_definitions, boundary_uses,
                                   root, loop_index, forward, arc_ancestry)
            else:
                record = _root_record(root, loop_index, forward, curve['type'], 'refused',
                    'unsupported family', {'finiteness_certified': None}, ())
            roots.append(record)
            if record.outcome == 'refused':
                refusals.append(f'edge {root}: {record.reason}')
            for child in record.children:
                claimed[child.edge_id] = root
    for edge in sorted(boundary_uses):
        if edge not in claimed:
            unaccounted.append(edge)
    if unaccounted:
        refusals.append('incomplete descendant boundary scope: '
                        + ', '.join(str(edge) for edge in unaccounted))
    if any(root.outcome == 'refused' for root in roots) or unaccounted:
        outcome = 'refused'
    elif any(child.outcome == 'certified_discrepancy'
             for root in roots if root.family == 'straight' for child in root.children):
        outcome = 'certified_discrepancy'
    else:
        outcome = 'exact'
    result = CylinderBoundaryCorrespondence(faces, edge_preimages, original, descendants, frame,
        tuple(roots), tuple(interior), tuple(unaccounted), outcome, tuple(refusals))
    _check(cancellation_check)
    validate_prepared_edge_subcurve_preimages_binding(model, edge_preimages,
        cancellation_check=cancellation_check)
    if definition_checksum(edges) != entry_edge_digest:
        raise GeometryError('cylinder boundary correspondence edge preimages changed')
    validate_prepared_face_preimages_binding(model, faces)
    final_original = query_prepared_authored_face_definition(model, authored_face_id,
        expected_revision=expected_revision)
    if (final_original.face_id != original.face_id
            or final_original.revision != original.revision
            or final_original.source_checksum != original.source_checksum
            or final_original.definition_json != original.definition_json):
        raise GeometryError('cylinder boundary correspondence authored definition changed')
    return result


def _straight_root(frame, controls, ancestry, boundary_uses, root, loop_index, forward):
    evidence = _straight_root_evidence(frame, controls[root])
    matches = ancestry.get(root, ())
    if not matches:
        return _root_record(root, loop_index, forward, 'straight', 'refused',
                            'ancestry unavailable', evidence, ())
    if _unpack(evidence['radial_min_squared']) == 0:
        return _root_record(root, loop_index, forward, 'straight', 'refused',
                            'radial zero on authored straight', evidence, ())
    children = []
    for record in matches:
        if record.edge_id not in boundary_uses:
            return _root_record(root, loop_index, forward, 'straight', 'refused',
                               'descendant not on current boundary', evidence, ())
        interval = tuple(_unpack(x) for x in record.interval)
        use_forward = boundary_uses[record.edge_id][1]
        # Oriented boundary equality: the directed source interval sign
        # combined with the current EdgeUse direction must equal the original
        # authored EdgeUse direction.  The tiling check drops direction and
        # cannot prove this; a reversal refuses.
        if (interval[1] > interval[0]) != (use_forward == forward):
            return _root_record(root, loop_index, forward, 'straight', 'refused',
                                'descendant orientation reversed', evidence, ())
        child_controls = tuple(_point(point) for point in record.current_definition.controls)
        if _radial_squared_min(_radial_affine(frame, child_controls)) == 0:
            return _root_record(root, loop_index, forward, 'straight', 'refused',
                                'radial zero on current straight', evidence, ())
        child_outcome, child_evidence = _compare_straight(frame, controls[root],
                                                          interval, child_controls)
        if child_outcome == 'refused':
            return _root_record(root, loop_index, forward, 'straight', 'refused',
                                child_evidence['reason'], evidence, ())
        children.append(ProjectedStraightChild(record.edge_id,
            (_pack(interval[0]), _pack(interval[1])), boundary_uses[record.edge_id][1],
            record.squared_distance_bound, child_outcome,
            child_evidence.get('axial_error_bound'), child_evidence.get('angular_witness'),
            child_evidence.get('angular_error_lower_bound'),
            child_evidence.get('opposite_directions', False)))
    tiling = _tile_spans((child.edge_id, _unpack(child.interval[0]), _unpack(child.interval[1]))
                         for child in children)
    if tiling:
        return _root_record(root, loop_index, forward, 'straight', 'refused',
                            f'ancestry {tiling}', evidence, ())
    outcome = ('certified_discrepancy'
               if any(child.outcome == 'certified_discrepancy' for child in children)
               else 'exact')
    return _root_record(root, loop_index, forward, 'straight', outcome, None, evidence,
                        tuple(children))


def _arc_root(frame, current_edges, arc_definitions, boundary_uses, root, loop_index,
              forward, arc_ancestry=None):
    """Arc root: unsplit identity or bounded geometric-subarc-image ancestry.

    Unsplit: exact native definition identity plus whole-circle finiteness.
    ``exact`` requires BOTH plus directed-use agreement with the original
    authored EdgeUse; identity without a finiteness certificate refuses the
    projected correspondence while retaining the native definition identity as
    separate evidence.

    Split: authenticated arc ancestry records only, never inference from
    samples.  Each child must lie on the current boundary, its stored
    direction relative to the ancestor's combined with the current EdgeUse
    direction must equal the original authored EdgeUse direction, and its own
    whole-circle finiteness must be certified.  An ``exact`` child
    (three actual points exactly on the ancestor's exact circumcircle)
    contributes to the complete GEOMETRIC tiling proof; an ``enclosed``
    child refuses the root as enclosed-within-tolerance-not-exact; a refused
    child refuses the root with its recorded reason.  Root ``exact`` requires
    the complete geometric tiling (authenticated shared anchors, strict
    directed ordering, complete ancestor span); numeric interval tiling is
    never a shortcut.  All current inputs come from the detached entry
    snapshot, never from the live model.
    """
    start, via, end, positions = arc_definitions[root]
    snapshot = current_edges.get(root)
    identity = root in boundary_uses and snapshot is not None and snapshot[0] == 'arc'
    if identity:
        identity = (snapshot[1] == start and snapshot[2] == via and snapshot[3] == end
                    and snapshot[4] == positions)
    if identity:
        if boundary_uses[root][1] != forward:
            return _root_record(root, loop_index, forward, 'arc', 'refused',
                'descendant orientation reversed', {'finiteness_certified': None,
                                                    'native_identity_certified': True}, ())
        child = ProjectedArcChild(root, boundary_uses[root][1], start, via, end)
        certified, evidence, reason = _arc_finiteness(frame, positions)
        if not certified:
            return _root_record(root, loop_index, forward, 'arc', 'refused', reason,
                {'finiteness_certified': False, 'native_identity_certified': True,
                 'arc_finiteness': evidence}, (child,))
        return _root_record(root, loop_index, forward, 'arc', 'exact', None,
            {'finiteness_certified': True, 'native_identity_certified': True,
             'arc_finiteness': evidence}, (child,))
    records = (arc_ancestry or {}).get(root, ())
    if not records:
        return _root_record(root, loop_index, forward, 'arc', 'refused',
            'arc ancestry unavailable', {'finiteness_certified': None,
                                         'native_identity_certified': False}, ())
    children = []
    exact_records = []
    for record in records:
        if record.edge_id not in boundary_uses:
            return _root_record(root, loop_index, forward, 'arc', 'refused',
                'descendant not on current boundary', {'finiteness_certified': None,
                    'native_identity_certified': False}, tuple(children))
        use_forward = boundary_uses[record.edge_id][1]
        # Oriented boundary equality: the child's stored direction relative to
        # the ancestor's, combined with the current EdgeUse direction, must
        # equal the original authored EdgeUse direction.
        if (use_forward == (record.direction == 'forward')) != forward:
            return _root_record(root, loop_index, forward, 'arc', 'refused',
                'descendant orientation reversed', {'finiteness_certified': None,
                    'native_identity_certified': False}, tuple(children))
        child = ProjectedArcSplitChild(record.edge_id, use_forward,
            record.current_definition.start, record.current_definition.via,
            record.current_definition.end, record.classification, record.direction,
            record.anchor_bounds, record.whole_circle_bound, record.tolerance,
            record.interval, record.parameter_mapping_qualified)
        children.append(child)
        if record.classification == 'refused':
            return _root_record(root, loop_index, forward, 'arc', 'refused',
                record.reason, {'finiteness_certified': None,
                    'native_identity_certified': False}, tuple(children))
        if record.classification == 'enclosed':
            # Enclosure is honest evidence, never exactness and never proof the
            # curves differ; the root refuses as enclosed-within-tolerance.
            return _root_record(root, loop_index, forward, 'arc', 'refused',
                'arc split enclosed within tolerance, not exact',
                {'finiteness_certified': None, 'native_identity_certified': False},
                tuple(children))
        certified, evidence, reason = _arc_finiteness(
            frame, _arc_positions(record.current_definition))
        if not certified:
            return _root_record(root, loop_index, forward, 'arc', 'refused', reason,
                {'finiteness_certified': False, 'native_identity_certified': False},
                tuple(children))
        exact_records.append(record)
    tiling = _arc_geometric_tiling(positions, exact_records)
    if tiling is not None:
        return _root_record(root, loop_index, forward, 'arc', 'refused', tiling,
            {'finiteness_certified': True, 'native_identity_certified': False},
            tuple(children))
    return _root_record(root, loop_index, forward, 'arc', 'exact', None,
        {'finiteness_certified': True, 'native_identity_certified': False},
        tuple(children))


def validate_prepared_cylinder_boundary_correspondence_binding(model, result, *,
        cancellation_check=None):
    """Validate immutable correspondence against the owner's current proof.

    The request target/revision and the caller result's definition checksum
    are pinned at entry, BEFORE any callback or callback-bearing rederivation.
    After rederivation the current input checksum must still equal the pinned
    entry checksum — a result repaired or mutated in place during callbacks is
    refused — and the owner's rederived checksum must equal the same pinned
    entry checksum, so an initially forged result cannot be repaired into
    acceptance even when its final fields are legitimate.
    """
    if not isinstance(result, CylinderBoundaryCorrespondence):
        raise GeometryError('cylinder boundary correspondence needs an owner result')
    entry_face_id = result.authored_definition.face_id
    entry_revision = result.face_preimages.revision
    entry_checksum = definition_checksum(result)
    expected = query_prepared_cylinder_boundary_correspondence(model, entry_face_id,
        expected_revision=entry_revision, cancellation_check=cancellation_check)
    if definition_checksum(result) != entry_checksum:
        raise GeometryError('cylinder boundary correspondence input changed during validation')
    if definition_checksum(expected) != entry_checksum:
        raise GeometryError('cylinder boundary correspondence definition binding changed')
