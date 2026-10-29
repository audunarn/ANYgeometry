"""Length enclosures of actual polynomial translation-section evaluators.

This is an exact image correspondence plus a POSITION error tube. It neither
changes PlaneNurbsCurve's midpoint/root law nor reuses a Green-area error.
"""
from fractions import Fraction as Q
import math

from .errors import GeometryError


def structural_section_hull(curve, lower, upper):
    """Return exact world controls and a component tube, or None.

    The registered subset has uniform weights, one linear translation axis,
    profile-u slicing and authentic fixed-u endpoint roots. Other cases keep
    the native implicit enclosure backend.
    """
    from .support_curves import PlaneNurbsCurve
    from .nurbs import NurbsSurface, _exact_basis, _exact_span, _float_up
    from .nurbs_runtime_binding import validate_section_runtime
    from .polynomial_green import bezier_spans
    from .plane_nurbs import charge_box
    if type(curve) is not PlaneNurbsCurve or type(curve.surface) is not NurbsSurface:
        return None
    charge_box()  # before native validation, section extraction or insertion
    # This registration includes the native unconditional initial bisection
    # stage. Read the EFFECTIVE attribute used by _exact_point; raw class or
    # instance edits must not silently inherit a nominal sixty-step theorem.
    initial_bisections = curve._INITIAL_ROOT_BISECTIONS
    if type(initial_bisections) is not int or initial_bisections != 60:
        return None
    validate_section_runtime(curve)
    surface = curve.surface
    if (surface.degree_v != 1 or len(surface.knots_v) != 4
            or surface.knots_v[:2] != (surface.knots_v[0],)*2
            or surface.knots_v[2:] != (surface.knots_v[2],)*2
            or any(len(row) != 2 for row in surface.controls)
            or surface.weights is not None and len({Q(w) for row in surface.weights for w in row}) != 1):
        return None
    profile = tuple(tuple(map(Q, row[0])) for row in surface.controls)
    direction = tuple(Q(b)-Q(a) for a, b in zip(*surface.controls[0]))
    if not any(direction) or any(tuple(Q(b)-Q(a) for a, b in zip(*row)) != direction
                                 for row in surface.controls):
        return None
    normal, offset, _norm2, _slab2, slab = curve._plane_slab()
    denominator = sum(n*d for n, d in zip(normal, direction))
    if not denominator:
        return None
    try:
        lower, upper = Q(lower), Q(upper)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as cause:
        raise GeometryError('structural section hull requires finite rational parameters') from cause
    if not 0 <= lower <= upper <= 1:
        raise GeometryError('structural section hull interval must lie in [0,1]')
    start, finish = map(Q, curve.domain)
    low, high = sorted((start+lower*(finish-start), start+upper*(finish-start)))
    arcs = curve._arcs()
    count = len(arcs)
    # Admission is for the actual whole retained definition, not an accidental
    # eligible tile of an otherwise unsupported endpoint/axis protocol.
    participating = []
    retained_low, retained_high = sorted((start, finish))
    for index, arc in enumerate(arcs):
        if Q(index+1, count) < retained_low or Q(index, count) > retained_high:
            continue
        if (arc.slicing != 0 or arc.first.axis != 0 or arc.second.axis != 0
                or arc.first.value != arc.s0 or arc.second.value != arc.s1):
            return None
        participating.append((index, arc))
    world_controls = []
    beta_error = Q(0)
    span_v = Q(surface.knots_v[2])-Q(surface.knots_v[1])
    def project(point):
        charge_box()  # an examined exact projected profile control
        beta = (offset-sum(n*x for n, x in zip(normal, point)))/denominator
        return tuple(x+d*beta for x, d in zip(point, direction))
    for index, arc in participating:
        a, b = max(low, Q(index, count)), min(high, Q(index+1, count))
        if a > b:
            continue
        charge_box()
        tau0, tau1 = a*count-index, b*count-index
        u0, u1 = (arc.s0+t*(arc.s1-arc.s0) for t in (tau0, tau1))
        box_low, box_high = map(Q, arc.box.u_range)
        if not box_low <= min(u0, u1) <= max(u0, u1) <= box_high:
            return None
        # S=P(u)+D*beta(v), so its exact plane image is this affine
        # projection of ORIGINAL profile controls; knot insertion is exact.
        if u0 == u1:
            knots = list(map(Q, surface.knots_u))
            span = _exact_span(surface.degree_u, knots, len(profile), u0)
            basis = _exact_basis(surface.degree_u, knots, span, u0)
            point = tuple(sum(w*profile[span-surface.degree_u+j][axis] for j, w in enumerate(basis))
                          for axis in range(3))
            world_controls.append(project(point))
        else:
            for controls in bezier_spans(surface.degree_u, surface.knots_u, profile, u0, u1):
                world_controls.extend(project(point) for point in controls)
        # Interior _exact_point first bisects the full transverse bracket
        # sixty times; only an EXACT root can terminate that loop early.
        # Subsequent _refine only narrows the bracket. Its defined midpoint
        # differs from the true root by at most HALF that bracket's width.
        # This independent length bound avoids magnifying the residual slab
        # by an ill-conditioned n.D. Endpoint/root branches do NOT use sixty.
        transverse_width = Q(arc.box.v_range[1])-Q(arc.box.v_range[0])
        interior_error = min(slab/abs(denominator),
                             transverse_width/(Q(2)**(initial_bisections+1)*span_v))
        # At each endpoint (including clamped root branches), true root AND
        # actual root-point remain in the authentic fixed-u isolating v range;
        # their displacement is at most its FULL width / knot span_v.
        # No zero authored/proxy endpoint correction is assumed here.
        beta_error = max(beta_error, interior_error,
                         (arc.first.hi-arc.first.lo)/span_v,
                         (arc.second.hi-arc.second.lo)/span_v)
    if not world_controls:
        raise GeometryError('structural section hull has no covered native arc')
    scales = tuple(max(abs(Q(p[i])) for row in surface.controls for p in row) for i in range(3))
    try:
        rounding = tuple(Q(math.ulp(_float_up(scale)))/2 if scale else Q(0) for scale in scales)
    except (ValueError, OverflowError) as cause:
        raise GeometryError('structural section rounding bound is not finite') from cause
    # Actual S is in the original positive-weight surface hull, so these
    # scales bound its one final coordinate rounding, even if projected
    # Bernstein controls themselves lie outside that original hull.
    margin = tuple(abs(d)*beta_error+r for d, r in zip(direction, rounding))
    return tuple(world_controls), tuple((-value, value) for value in margin)
