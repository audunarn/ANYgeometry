"""Exact native control hulls with the native evaluator's existing error tube.

Only concrete kernels whose bounds are built from these SAME hulls and an
additive coordinate error margin are registered. An arbitrary bounding box
does not establish a control-hull certificate.
"""
from fractions import Fraction as Q
import math

from .errors import GeometryError


def registered_world_hull(curve, lower, upper):
    """Return (rational controls, coordinate error tube), or None.

    Native hull extraction already widens the parameter interval for the
    actual floating parameter map. Native bounds additionally account for
    evaluation arithmetic with an explicit componentwise margin. Use that SAME
    mathematical margin exactly, without outward conversion to doubles.
    No certificate or source authority is cached or persisted here.
    """
    from .nurbs import NurbsCurve, NurbsSurface, CurveOnSurface
    from .line_ranges import LineRange
    from .plane_nurbs import charge_box
    from .nurbs_runtime_binding import validate_curve_runtime
    from .support_curves import PlaneNurbsCurve
    if type(curve) is PlaneNurbsCurve:
        from .structural_section_hulls import structural_section_hull
        return structural_section_hull(curve, lower, upper)
    if type(curve) not in (NurbsCurve, CurveOnSurface, LineRange):
        return None
    charge_box()  # charge before native knot insertion or enclosure work
    if type(curve) is NurbsCurve:
        if curve.dimension != 3:
            return None
        validate_curve_runtime(curve)
        controls = tuple(curve._active_controls(lower, upper))
        # SAME native bounds theorem: parameter-map error is already in the
        # hull; remaining coordinate evaluation error is bounded by this
        # explicit native margin. This is stronger than bbox containment.
        margin = tuple(16*(curve.degree+1)*Q(1, 2**52)*max(abs(Q(p[i])) for p in curve.controls)
                       for i in range(3))
    elif type(curve) is CurveOnSurface:
        if type(curve.support) is not NurbsSurface or type(curve.pcurve) is not NurbsCurve:
            return None
        validate_curve_runtime(curve.pcurve)
        if curve._iso_world(fresh=True) is None:
            return None
        charge_box()
        controls = tuple(curve._iso_hull(lower, upper, fresh=True))
        # Literal surface.exact_net() builds the homogeneous iso controls;
        # neither prepared surface caches nor _ISO_CACHE are proof authority.
        # The iso-chain parameter slack is ALREADY in the hull. The following
        # is exactly the existing _float_margin theorem, in rational units.
        margin = tuple(16*(curve.support.degree_u+curve.support.degree_v+2)*Q(1, 2**52)
                       *max(abs(Q(p[i])) for row in curve.support.controls for p in row)
                       for i in range(3))
    else:
        curve.validate_definition()
        controls = (curve.exact_point(lower), curve.exact_point(upper))
        # One final coordinate rounding. Endpoint bbox expansion alone is
        # insufficient when binary endpoints have nonbinary interior points.
        from .nurbs import _float_up
        scales = tuple(max(abs(p[i]) for p in controls) for i in range(3))
        try:
            margin = tuple(Q(math.ulp(_float_up(value)))/2 if value else Q(0) for value in scales)
        except (ValueError, OverflowError) as cause:
            raise GeometryError('registered world hull rounding tube is not finite') from cause
    if not controls:
        raise GeometryError('registered world hull has no controls')
    tube = tuple((-value, value) for value in margin)
    return controls, tube


def fixed_chord_hull_bound(curve, lower, upper, endpoints):
    """Convex-distance upper bound, keeping all controls as exact Fractions."""
    from itertools import product
    from .nurbs_domain import _exact_segment_distance
    from .plane_nurbs import charge_box
    hull = registered_world_hull(curve, lower, upper)
    if hull is None:
        return None
    controls, tube = hull
    # A convex function on a finite convex hull is bounded by its vertex
    # maximum. Distance to a FIXED closed segment is convex, including where
    # its closest point is either segment endpoint.
    corners = tuple(product(*tube))
    bound = Q(0)
    for point in controls:
        charge_box()  # one examined control-point error box
        bound = max(bound, *(_exact_segment_distance(tuple(x+d for x, d in zip(point, delta)), *endpoints)
                             for delta in corners))
    return bound
