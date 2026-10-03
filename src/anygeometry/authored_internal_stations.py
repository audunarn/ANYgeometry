"""Exact source stations on authenticated straight planar internal edges.

Paired incidence and a literal seam tag are descriptions, not a physical-edge
classification, material certificate, node identity or publication permission.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral, Real

import numpy as np

from .arrangement_geometry import LinePath
from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_domain_coverage import _original_domain
from .curves import Straight
from .definition_binding import definition_checksum
from .entities import EntityRef
from .errors import GeometryError
from .material_cell_coverage import _chart_polynomial, _frame, _value
from .surfaces import Plane


@dataclass(frozen=True, slots=True)
class AuthoredInternalStations:
    correspondence: object
    edge_id: int
    endpoint_ids: tuple
    endpoint_points: tuple
    source_definition_checksum: str
    child_incidence: tuple
    # (coedge ID, face-use ID, sheet ID, face ID, coedge/face-use orientation).
    occurrences: tuple
    decomposition_seam_tag: bool
    parameters: tuple
    authored_uv: tuple
    authored_points: tuple
    current_points: tuple
    tolerance: tuple


def _pack(value):
    return value.numerator, value.denominator


def _signature(value):
    try:
        return definition_checksum(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored internal station definition binding is malformed') from error


def _parameters(parameters):
    try:
        raw = tuple(parameters)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) for value in raw):
            raise GeometryError('authored internal stations need real rational source parameters')
        values = tuple(F(*value.as_integer_ratio()) if isinstance(value, np.floating) else
                       F(int(value)) if isinstance(value, np.integer) else F(value) for value in raw)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('authored internal stations need finite rational source parameters') from error
    if any(not 0 <= value <= 1 for value in values):
        raise GeometryError('authored internal stations need source parameters in [0,1]')
    return values


def _source(model, correspondence, identifier):
    incidence = dict(correspondence.interior_incidence).get(identifier)
    if (incidence is None or len(incidence) != 2 or incidence[0][0] == incidence[1][0]
            or incidence[0][1] == incidence[1][1]):
        raise GeometryError('authored internal stations require a paired internal edge')
    edge = deepcopy(model.edges[identifier])
    if type(edge.curve) is not Straight:
        raise GeometryError('authored internal stations require a literal Straight edge')
    endpoints = tuple(deepcopy(model.vertices[key]) for key in (edge.start, edge.end))
    controls = tuple(tuple(F(float(value)) for value in vertex.position) for vertex in endpoints)
    if controls[0] == controls[1]:
        raise GeometryError('authored internal stations require a nondegenerate edge')
    child_incidence = tuple((int(face), bool(forward)) for face, forward in incidence)
    faces = {face for face, _ in child_incidence}
    occurrences = tuple(sorted((row.id, row.face_use_id, use.sheet_id, use.face_id,
                                row.orientation.value, use.orientation.value)
        for row in model.coedges.values() if row.edge_id == identifier
        for use in (model.face_uses[row.face_use_id],) if use.face_id in faces))
    seam = 'intersection_decomposition_seam' in model.tags_for(EntityRef('edge', identifier))
    # Same Straight-edge extent and effective length rule as edge_length, using
    # detached endpoints so this query does not populate model length caches.
    extent = float(np.linalg.norm(endpoints[1].position-endpoints[0].position))
    tolerance = F(model.tolerance.effective_length(extent))
    signature = definition_checksum((edge, endpoints, child_incidence, occurrences, seam, _pack(tolerance)))
    return edge, controls, child_incidence, occurrences, seam, tolerance, signature


def query_prepared_authored_internal_stations(model, correspondence, current_edge_id, parameters, *,
                                             cancellation_check=None):
    """Map current line parameters exactly into the ORIGINAL Plane chart.

    Parameters follow the stored edge start/end direction and retain caller
    order and duplicates. Rational stations need not be binary64-representable.
    UV and both XYZ representations are immutable numerator/denominator pairs.
    Only an authenticated paired internal Straight edge exactly on the original
    Plane qualifies. No material membership or physical-joint classification is
    inferred from paired incidence, occurrences, or presence/absence of a tag.
    """
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    correspondence_signature = definition_checksum(correspondence)
    if isinstance(current_edge_id, (bool, np.bool_)) or not isinstance(current_edge_id, Integral):
        raise GeometryError('authored internal stations need an integer internal edge ID')
    try:
        identifier = int(current_edge_id)
    except (TypeError, ValueError, OverflowError) as error:
        raise GeometryError('authored internal stations need a usable internal edge ID') from error
    values = _parameters(parameters)
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    if definition_checksum(correspondence) != correspondence_signature:
        raise GeometryError('authored internal station correspondence changed')
    definition = deepcopy(correspondence.authored_definition)
    edge, controls, incidence, occurrences, seam, tolerance, signature = _source(model, correspondence, identifier)
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)

    def check():
        if cancellation_check is not None and cancellation_check('authored internal stations'):
            raise GeometryError('authored internal station query cancelled')

    check()
    domain = _original_domain(definition, check)
    if type(domain.support) is not Plane:
        raise GeometryError('authored internal stations require the original exact Plane support')
    frame = _frame(domain.support)
    chart = _chart_polynomial(frame, LinePath(*controls), check)
    uv, original, current = [], [], []
    for value in values:
        check()
        first, second = (_value(row, value) for row in chart)
        uv.append((_pack(first), _pack(second)))
        original.append(tuple(_pack(o+first*u+second*v)
                              for o, u, v in zip(frame['origin'], frame['u'], frame['v'])))
        current.append(tuple(_pack((1-value)*a+value*b) for a, b in zip(*controls)))
    result = AuthoredInternalStations(correspondence, identifier, (edge.start, edge.end),
        tuple(tuple(_pack(x) for x in point) for point in controls), signature, incidence,
        occurrences, bool(seam), tuple(map(_pack, values)), tuple(uv), tuple(original),
        tuple(current), _pack(tolerance))
    check()
    validate_prepared_authored_boundary_correspondence_binding(model, correspondence)
    if (definition_checksum(correspondence) != correspondence_signature or
            _source(model, correspondence, identifier)[-1] != signature):
        raise GeometryError('authored internal station source definition changed')
    return result


def validate_prepared_authored_internal_station_coordinates(model, receipt, xyz, *, cancellation_check=None):
    """Check copied binary64 coordinates against BOTH exact XYZ representations.

    Uses the unchanged current edge Euclidean tolerance. This does not move
    coordinates, bind global node IDs, or authorize subdivision/publication.
    """
    if type(receipt) is not AuthoredInternalStations:
        raise GeometryError('authored internal coordinates require owner stations')
    signature = _signature(receipt)
    captured = deepcopy(receipt)
    try:
        if np.iscomplexobj(np.asarray(xyz)):
            raise GeometryError('authored internal coordinates must be real')
        points = np.array(xyz, dtype=float, copy=True)
        parameters = tuple(F(*value) for value in captured.parameters)
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as error:
        raise GeometryError('authored internal coordinates/parameters must be finite values') from error
    if points.shape != (len(parameters), 3) or not np.isfinite(points).all():
        raise GeometryError('authored internal coordinates must be finite (n,3) values')
    expected = query_prepared_authored_internal_stations(model, captured.correspondence, captured.edge_id,
        parameters, cancellation_check=cancellation_check)
    if _signature(expected) != signature or _signature(receipt) != signature:
        raise GeometryError('authored internal station definition binding changed')
    tolerance = F(*captured.tolerance)
    for actual, authored, represented in zip(points, captured.authored_points, captured.current_points):
        if cancellation_check is not None and cancellation_check('authored internal coordinate assertion'):
            raise GeometryError('authored internal coordinate assertion cancelled')
        for reference in (authored, represented):
            squared = sum((F(float(value))-F(*target))**2 for value, target in zip(actual, reference))
            if squared > tolerance*tolerance:
                raise GeometryError('authored internal coordinate error exceeds existing edge tolerance')
    validate_prepared_authored_boundary_correspondence_binding(model, captured.correspondence)
    if _signature(receipt) != signature:
        raise GeometryError('authored internal station definition binding changed')
