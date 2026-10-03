"""Source-parameter correspondence to an original polynomial carrier chart.

This is not an algebraic intersection, material-membership or mesh certificate.
"""
from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral, Real

import numpy as np

from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_boundary_stations import _pack, _point, _assert_coordinate_errors
from .authored_domain_coverage import _original_domain
from .branch_algebra import BezierRuledSupport
from .branch_curves import BezierQuadricCurve
from .definition_binding import definition_checksum
from .errors import GeometryError
from .material_cell_coverage import _frame
from .surfaces import ExtrudedSurface


@dataclass(frozen=True, slots=True)
class AuthoredCurveStations:
    correspondence: object
    edge_id: int
    curve_checksum: str
    occurrences: tuple
    parameters: tuple
    carrier_parameters: tuple
    authored_uv: tuple
    authored_points: tuple
    current_curve_points: tuple
    tolerance: tuple


def _parameters(parameters):
    try:
        values = tuple(parameters)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in values):
            raise GeometryError('authored curve parameters must be real source stations')
        exact = tuple(F(*value.as_integer_ratio()) if isinstance(value, np.floating) else
                      F(int(value)) if isinstance(value, np.integer) else F(value)
                      for value in values)
        floats = tuple(float(value) for value in exact)
        if any(not 0 <= value <= 1 or F(rounded) != value
               for value, rounded in zip(exact, floats)):
            raise GeometryError('authored curve parameters must be binary64-representable source stations in [0,1]')
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('authored curve parameters must be finite source stations') from error
    return exact, np.array(floats, dtype=float)


def _carrier(curve, domain):
    if type(curve) is not BezierQuadricCurve or not curve._identity:
        raise GeometryError('authored curve stations require an untransformed BezierQuadricCurve')
    frame = _frame(domain.support)
    if type(domain.support) is not ExtrudedSurface or frame['kind'] != 'bezier':
        raise GeometryError('authored curve stations require an original polynomial extrusion')
    if curve.first != BezierRuledSupport.from_surface(domain.support):
        raise GeometryError('authored curve station carrier differs from the original definition')
    return frame


def query_prepared_authored_curve_stations(model, correspondence, edge_id, parameters, *,
        cancellation_check=None):
    """Map internal BQC source stations to the ORIGINAL Bezier carrier chart.

    The edge identity and parameter select the branch; no XYZ inverse is used.
    Only an identical first carrier and identity affine transform qualify. All
    parameters must be exactly binary64 representable. Returned rational UV
    uses the owner's evaluated binary64 branch coordinates, NOT an exact
    algebraic intersection point. Original XYZ is the exact carrier polynomial
    at those coordinates. Current evaluation error uses the existing edge
    tolerance. Patch/trim membership and whole-curve coverage remain separate.
    """
    if isinstance(edge_id, (bool, np.bool_)) or not isinstance(edge_id, Integral):
        raise GeometryError('authored curve stations require an internal edge ID')
    try:
        edge_id = int(edge_id)  # no caller-defined coercion after binding guards
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored curve stations require an internal edge ID') from error
    exact, tau = _parameters(parameters)  # detach before invoking callbacks

    def check():
        if cancellation_check is not None and cancellation_check('authored curve stations'):
            raise GeometryError('authored curve station query cancelled')

    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence,
        cancellation_check=cancellation_check)
    occurrences = dict(correspondence.interior_incidence).get(int(edge_id))
    if occurrences is None:
        raise GeometryError('authored curve stations require a paired internal edge')
    curve = model.edges[int(edge_id)].curve
    domain = _original_domain(correspondence.authored_definition, check)
    frame = _carrier(curve, domain)

    signature = definition_checksum(curve)
    t, _dt, anchor, delta, f, _fp, side = curve._chart(tau)
    s, _radical = curve._root(curve.plan().floats(), t, anchor, delta, f, side)
    points = np.asarray(curve.evaluate(tau), dtype=float).reshape((-1, 3))
    if not np.isfinite(t).all() or not np.isfinite(s).all() or not np.isfinite(points).all():
        raise GeometryError('authored curve station mapping produced nonfinite results')
    u0, u1 = frame['urange']; v0, v1 = frame['vrange']
    carrier, uv, original, represented = [], [], [], []
    for first, second, row in zip(t, s, points):
        check()  # exact rational station work remains cooperatively cancellable
        a, b = F(float(first)), F(float(second))
        carrier.append((_pack(a), _pack(b)))
        uv.append((_pack((a-u0)/(u1-u0)), _pack((b-v0)/(v1-v0))))
        original.append(tuple(_pack(x+b*d) for x, d in zip(_point(frame['controls'], a), frame['direction'])))
        represented.append(tuple(_pack(F(float(x))) for x in row))
    original, represented = tuple(original), tuple(represented)
    tolerance = F(model.tolerance.effective_length(model.edge_length(int(edge_id))))
    _assert_coordinate_errors(points, original, represented, tolerance, check)
    result = AuthoredCurveStations(correspondence, int(edge_id), signature, occurrences,
        tuple(map(_pack, exact)), tuple(carrier), tuple(uv), original, represented, _pack(tolerance))
    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence,
        cancellation_check=cancellation_check)
    if definition_checksum(model.edges[int(edge_id)].curve) != signature:
        raise GeometryError('authored curve station source definition changed')
    return result


def validate_prepared_authored_curve_station_coordinates(model, stations, coordinates, *,
        cancellation_check=None):
    """Assert unchanged supplied XYZ against both original carrier and source.

    Success supplies no global node identity, material/curve approximation,
    subdivision permission or mesh acceptance. Coordinates are never moved.
    """
    if not isinstance(stations, AuthoredCurveStations):
        raise GeometryError('authored curve coordinate assertion requires owner stations')
    try:
        if np.iscomplexobj(np.asarray(coordinates)):
            raise GeometryError('authored curve coordinates must be real')
        points = np.array(coordinates, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored curve coordinates must be finite (n,3) values') from error
    if points.shape != (len(stations.parameters), 3) or not np.isfinite(points).all():
        raise GeometryError('authored curve coordinates must be finite (n,3) values')
    signature = definition_checksum(stations)
    try:
        parameters = tuple(F(*value) for value in stations.parameters)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('authored curve station parameters are malformed') from error
    expected = query_prepared_authored_curve_stations(model, stations.correspondence, stations.edge_id,
        parameters, cancellation_check=cancellation_check)
    if definition_checksum(expected) != signature or definition_checksum(stations) != signature:
        raise GeometryError('authored curve station definition binding changed')

    def check():
        if cancellation_check is not None and cancellation_check('authored curve coordinate assertion'):
            raise GeometryError('authored curve coordinate assertion cancelled')

    _assert_coordinate_errors(points, stations.authored_points, stations.current_curve_points,
        F(*stations.tolerance), check)
    validate_prepared_authored_boundary_correspondence_binding(model, stations.correspondence,
        cancellation_check=cancellation_check)
    if definition_checksum(stations) != signature:
        raise GeometryError('authored curve station definition binding changed')
