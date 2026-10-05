"""Original polynomial boundary stations with explicit representation errors.

These queries do not move nodes or authenticate a consumer's global registry.
Original-domain UV is separate from the current rounded fragment chart.
"""
from dataclasses import dataclass
from fractions import Fraction as F
import json
from numbers import Integral

import numpy as np

from .arrangement_geometry import BezierPath, LinePath
from .authored_boundary_correspondence import (
    _polynomial_definition, validate_prepared_authored_boundary_correspondence_binding)
from .authored_domain_coverage import _original_domain
from .definition_binding import definition_checksum
from .errors import GeometryError
from .material_cell_coverage import _chart_polynomial, _frame, _value


@dataclass(frozen=True, slots=True)
class AuthoredBoundaryStations:
    correspondence: object
    edge_id: int
    parameters: tuple
    authored_parameters: tuple
    authored_uv: tuple
    authored_points: tuple
    current_polynomial_points: tuple
    tolerance: tuple


def _pack(x):
    return x.numerator, x.denominator


def _point(controls, t):
    rows = controls
    while len(rows) > 1:
        rows = tuple(tuple((1-t)*a+t*b for a, b in zip(first, second))
                     for first, second in zip(rows, rows[1:]))
    return rows[0]


def _check(callback):
    if callback is not None and callback('authored boundary stations'):
        raise GeometryError('authored boundary station query cancelled')


def _assert_coordinate_errors(points, original, current, tolerance, check):
    """Intersect the two Euclidean error balls; neither substitutes for the other."""
    for actual, authored, represented in zip(points, original, current):
        check()
        for reference in (authored, represented):
            squared = sum((F(float(value))-F(*target))**2 for value, target in zip(actual, reference))
            if squared > tolerance*tolerance:
                raise GeometryError('authored boundary coordinate error exceeds existing edge tolerance')


def query_prepared_authored_boundary_stations(model, correspondence, edge_id, parameters, *,
        cancellation_check=None):
    """Return exact original UV/XYZ and current polynomial XYZ at source stations.

    Parameters are exact rational values (binary64 inputs retain their exact
    value). The result is immutable; callers convert rational pairs explicitly.
    This is an original-domain correspondence, not permission to modify node
    coordinates or replace the current material-domain interpretation.
    """
    if isinstance(edge_id, (bool, np.bool_)) or not isinstance(edge_id, Integral):
        raise GeometryError('authored boundary stations need an exterior edge ID')
    try:
        values = tuple(F(value) for value in parameters)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('authored boundary stations need finite rational parameters') from error
    if any(value < 0 or value > 1 for value in values):
        raise GeometryError('authored boundary stations need parameters in [0, 1]')
    _check(cancellation_check)
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence,
        cancellation_check=cancellation_check)
    exterior = {edge for loop in correspondence.exterior_loops for _, _, edges in loop for edge in edges}
    if int(edge_id) not in exterior:
        raise GeometryError('authored boundary stations need a qualified exterior edge')
    # A unified shared boundary carries one sealed occurrence per participating
    # authored root; select this correspondence's own occurrence and interval,
    # never an arbitrary first primary record.
    authored = correspondence.authored_definition
    root_edges = {root for loop in correspondence.exterior_loops for root, _, _ in loop}
    matches = [row for row in (*correspondence.edge_preimages.records,
                               *correspondence.edge_preimages.alias_records)
               if row.edge_id == int(edge_id)
               and row.ancestor.model_id == authored.model_id
               and row.ancestor.revision == authored.revision
               and row.ancestor.source_checksum == authored.source_checksum
               and row.ancestor.definition.edge_id in root_edges]
    if len(matches) != 1:
        raise GeometryError('authored boundary stations need a unique exterior ancestry occurrence')
    record = matches[0]
    payload = json.loads(correspondence.authored_definition.definition_json)
    domain = _original_domain(correspondence.authored_definition, lambda: _check(cancellation_check))
    frame = _frame(domain.support)
    edges = {row['id']: row for row in payload['edges']}
    vertices = {row['id']: row for row in payload['vertices']}
    original = edges[record.ancestor.definition.edge_id]
    controls = _polynomial_definition(original, vertices)
    path = (LinePath(*controls) if original['curve']['type'] == 'straight' else BezierPath(controls))
    chart = _chart_polynomial(frame, path, lambda: _check(cancellation_check))
    first, second = (F(*value) for value in record.interval)
    authored = tuple(first+(second-first)*value for value in values)
    uv = tuple(tuple(_pack(_value(row, value)) for row in chart) for value in authored)
    original_xyz = tuple(tuple(_pack(x) for x in _point(controls, value)) for value in authored)
    current_controls = tuple(tuple(F(*x) for x in point) for point in record.current_definition.controls)
    current_xyz = tuple(tuple(_pack(x) for x in _point(current_controls, value)) for value in values)
    # The existing current-edge tolerance is a coordinate assertion bound only.
    # Retain the recorded whole-interval rounding enclosure independently.
    tolerance = F(model.tolerance.effective_length(model.edge_length(int(edge_id))))
    result = AuthoredBoundaryStations(correspondence, int(edge_id), tuple(map(_pack, values)),
        tuple(map(_pack, authored)), uv, original_xyz, current_xyz, _pack(tolerance))
    _check(cancellation_check)
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence,
        cancellation_check=cancellation_check)
    return result


def validate_prepared_authored_boundary_station_coordinates(model, stations, coordinates, *,
        cancellation_check=None):
    """Assert supplied binary64 XYZ against BOTH original and current polynomials.

    Success certifies only coordinate error at the supplied source parameters.
    A consumer must still bind global node IDs, registry entries and physical
    constraints; coordinates here provide no node identity or movement license.
    """
    if not isinstance(stations, AuthoredBoundaryStations):
        raise GeometryError('authored boundary coordinate assertion needs owner stations')
    try:
        if np.iscomplexobj(np.asarray(coordinates)):
            raise GeometryError('authored boundary coordinates must be real')
        points = np.array(coordinates, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored boundary coordinates must be finite (n,3) values') from error
    if points.shape != (len(stations.parameters), 3) or not np.isfinite(points).all():
        raise GeometryError('authored boundary coordinates must be finite (n,3) values')
    signature = definition_checksum(stations)
    expected = query_prepared_authored_boundary_stations(model, stations.correspondence, stations.edge_id,
        tuple(F(*x) for x in stations.parameters), cancellation_check=cancellation_check)
    if definition_checksum(expected) != signature or definition_checksum(stations) != signature:
        raise GeometryError('authored boundary station definition binding changed')
    tolerance = F(*stations.tolerance)
    _assert_coordinate_errors(points, stations.authored_points, stations.current_polynomial_points,
        tolerance, lambda: _check(cancellation_check))
    validate_prepared_authored_boundary_correspondence_binding(model, stations.correspondence,
        cancellation_check=cancellation_check)
    if definition_checksum(stations) != signature:
        raise GeometryError('authored boundary station definition binding changed')
