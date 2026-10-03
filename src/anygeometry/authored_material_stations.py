"""Current Straight trace stations in an authenticated original Plane chart.

No ancestral parameter, property remap, global node identity or mesh permission
is supplied. Exact rational UV comes from the owner Plane/trace identity.
"""
from copy import deepcopy
from dataclasses import dataclass
from fractions import Fraction as F
from numbers import Integral

import numpy as np

from .arrangement_geometry import LinePath
from .authored_boundary_correspondence import validate_prepared_authored_boundary_correspondence_binding
from .authored_domain_coverage import _original_domain
from .authored_internal_stations import _pack, _parameters, _signature
from .curves import Straight
from .definition_binding import definition_checksum
from .errors import GeometryError
from .material_cell_coverage import _chart_polynomial, _frame, _value
from .surfaces import Plane


@dataclass(frozen=True, slots=True)
class AuthoredMaterialStations:
    correspondence: object
    edge_id: int
    endpoint_ids: tuple
    endpoint_points: tuple
    source_definition_checksum: str
    trace_kind: str
    child_incidence: tuple
    parameters: tuple
    authored_uv: tuple
    authored_points: tuple
    current_points: tuple
    tolerance: tuple


def _source(model, correspondence, identifier):
    exterior = [edge for loop in correspondence.exterior_loops for _,_,edges in loop for edge in edges]
    interior = dict(correspondence.interior_incidence).get(identifier)
    incidence = tuple(sorted((face, use.forward) for face in correspondence.descendants
        for loop in (model.faces[face].loop,*model.faces[face].holes)
        for use in loop if use.edge == identifier))
    if identifier in exterior:
        if exterior.count(identifier)!=1 or interior is not None or len(incidence)!=1:
            raise GeometryError('authored material stations need unique exterior root incidence')
        kind='exterior'
    elif interior is not None:
        if (len(interior)!=2 or interior[0][0]==interior[1][0] or
                interior[0][1]==interior[1][1] or tuple(sorted(interior))!=incidence):
            raise GeometryError('authored material stations need paired interior root incidence')
        kind='paired_interior'
    else:
        raise GeometryError('authored material stations need an authenticated root trace')
    edge=deepcopy(model.edges[identifier])
    if type(edge.curve) is not Straight:
        raise GeometryError('authored material stations require a literal Straight trace')
    endpoints=tuple(deepcopy(model.vertices[key]) for key in (edge.start,edge.end))
    controls=tuple(tuple(F(float(value)) for value in row.position) for row in endpoints)
    if controls[0]==controls[1]:
        raise GeometryError('authored material stations require a nondegenerate trace')
    extent=float(np.linalg.norm(endpoints[1].position-endpoints[0].position))
    tolerance=F(model.tolerance.effective_length(extent))
    signature=definition_checksum((edge,endpoints,kind,incidence,_pack(tolerance)))
    return edge,controls,kind,incidence,tolerance,signature


def query_prepared_authored_material_stations(model,correspondence,current_edge_id,parameters,*,
                                            cancellation_check=None):
    """Lift ordered current [0,1] parameters into ORIGINAL exact Plane UV.

    0 is the stored CURRENT start vertex and 1 its end, irrespective of root
    boundary orientation. Exterior or paired interior incidence must already
    be authenticated. Fractions need not be binary64-representable. Results
    are packed numerator/denominator tuples and carry no ancestral parameters.
    """
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
    correspondence_signature=definition_checksum(correspondence)
    if isinstance(current_edge_id,(bool,np.bool_)) or not isinstance(current_edge_id,Integral):
        raise GeometryError('authored material stations need an integer current edge ID')
    try:
        identifier=int(current_edge_id)
    except (TypeError,ValueError,OverflowError) as error:
        raise GeometryError('authored material stations need a usable current edge ID') from error
    values=_parameters(parameters)
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
    if definition_checksum(correspondence)!=correspondence_signature:
        raise GeometryError('authored material station correspondence changed')
    definition=deepcopy(correspondence.authored_definition)
    edge,controls,kind,incidence,tolerance,signature=_source(model,correspondence,identifier)
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)

    def check():
        if cancellation_check is not None and cancellation_check('authored material stations'):
            raise GeometryError('authored material stations cancelled')

    check()
    domain=_original_domain(definition,check)
    if type(domain.support) is not Plane:
        raise GeometryError('authored material stations require the original exact Plane')
    frame=_frame(domain.support)
    chart=_chart_polynomial(frame,LinePath(*controls),check)
    uv,original,current=[],[],[]
    for parameter in values:
        check()
        first,second=(_value(row,parameter) for row in chart)
        uv.append((_pack(first),_pack(second)))
        original.append(tuple(_pack(o+first*u+second*v)
                              for o,u,v in zip(frame['origin'],frame['u'],frame['v'])))
        current.append(tuple(_pack((1-parameter)*a+parameter*b) for a,b in zip(*controls)))
    result=AuthoredMaterialStations(correspondence,identifier,(edge.start,edge.end),
        tuple(tuple(_pack(value) for value in row) for row in controls),signature,kind,incidence,
        tuple(map(_pack,values)),tuple(uv),tuple(original),tuple(current),_pack(tolerance))
    check()
    validate_prepared_authored_boundary_correspondence_binding(model,correspondence)
    if (definition_checksum(correspondence)!=correspondence_signature or
            _source(model,correspondence,identifier)[-1]!=signature):
        raise GeometryError('authored material station source definition changed')
    return result


def validate_prepared_authored_material_station_coordinates(model,receipt,xyz,*,cancellation_check=None):
    """Assert copied XYZ against BOTH owner lifts using the existing edge tolerance."""
    if type(receipt) is not AuthoredMaterialStations:
        raise GeometryError('authored material coordinates require owner stations')
    signature=_signature(receipt)
    captured=deepcopy(receipt)
    try:
        raw=np.array(xyz,copy=True)
        if np.iscomplexobj(raw):
            raise GeometryError('authored material coordinates must be real')
        points=np.array(raw,dtype=float,copy=True)
        parameters=tuple(F(*value) for value in captured.parameters)
    except (TypeError,ValueError,OverflowError,ZeroDivisionError) as error:
        raise GeometryError('authored material coordinates/parameters must be finite values') from error
    if points.shape!=(len(parameters),3) or not np.isfinite(points).all():
        raise GeometryError('authored material coordinates require finite (n,3) rows')
    expected=query_prepared_authored_material_stations(model,captured.correspondence,captured.edge_id,
        parameters,cancellation_check=cancellation_check)
    if _signature(expected)!=signature or _signature(receipt)!=signature:
        raise GeometryError('authored material station definition binding changed')
    tolerance=F(*captured.tolerance)
    for actual,original,current in zip(points,captured.authored_points,captured.current_points):
        if cancellation_check is not None and cancellation_check('authored material coordinate assertion'):
            raise GeometryError('authored material coordinate assertion cancelled')
        for reference in (original,current):
            squared=sum((F(float(value))-F(*target))**2 for value,target in zip(actual,reference))
            if squared>tolerance*tolerance:
                raise GeometryError('authored material coordinate error exceeds existing edge tolerance')
    validate_prepared_authored_boundary_correspondence_binding(model,captured.correspondence)
    if (_signature(receipt)!=signature or
            _source(model,captured.correspondence,captured.edge_id)[-1]!=captured.source_definition_checksum):
        raise GeometryError('authored material station definition binding changed')
