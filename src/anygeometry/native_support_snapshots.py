"""Detached actual native support coefficients, including nonserialized fields."""
import math

from .errors import GeometryError
from .curves import Straight, Spline
from .extrusions import BezierDirectrix
from .surfaces import Plane, Cylinder, CoonsSurface, ExtrudedSurface
from .polynomial_extrusion_support import COONS_POLYNOMIAL, BEZIER_EXTRUSION


def _vector(value):
    row = tuple(float(x) for x in value)
    if len(row) != 3 or any(not math.isfinite(x) for x in row):
        raise GeometryError('native support coefficients are invalid')
    return row


def _range(value):
    pair = tuple(float(x) for x in value)
    if len(pair) != 2 or any(not math.isfinite(x) for x in pair):
        raise GeometryError('native support range coefficients are invalid')
    return pair


def _coons_snapshot(model, face, support):
    if support.has_boundaries:
        # These arrays are piecewise-linear samples, NOT Bernstein controls.
        return 'coons_sampled', tuple(tuple(_vector(point) for point in getattr(support, key))
                                     for key in ('bottom', 'right', 'top', 'left'))
    sides = []
    for use in face.loop:
        edge = model.edges[use.edge]
        if type(edge.curve) not in (Straight, Spline):
            return 'unsupported_coons_topology', (tuple(face.corners),
                tuple((item.edge, item.forward) for item in face.loop))
        controls = () if type(edge.curve) is Straight else edge.curve.control_vertices
        points = tuple(_vector(model.vertices[i].position) for i in
                       (edge.start, *controls, edge.end))
        sides.append((edge.id, use.forward,
                      'straight' if type(edge.curve) is Straight else 'spline', points))
    return COONS_POLYNOMIAL, (tuple(face.corners), tuple(sides))


def capture_native_supports(model):
    """Actual stored maps, never reconstructed through document normalization."""
    rows = []
    for identifier, face in sorted(model.faces.items()):
        support = face.surface
        if face.parameterization is not None:
            rows.append((identifier, 'unsupported_parameterization', ()))
        elif type(support) is Plane:
            rows.append((identifier, 'plane', tuple(_vector(getattr(support, key))
                for key in ('origin', 'u_vector', 'v_vector'))))
        elif type(support) is Cylinder:
            values = tuple(float(getattr(support, key)) for key in
                ('radius', 'height', 'start_angle', 'sweep_angle'))
            if any(not math.isfinite(x) for x in values):
                raise GeometryError('native cylinder coefficients are invalid')
            rows.append((identifier, 'cylinder', tuple(_vector(getattr(support, key))
                for key in ('origin', 'axis', 'radial_direction', '_circumferential')) + values))
        elif type(support) is CoonsSurface:
            kind, data = _coons_snapshot(model, face, support)
            rows.append((identifier, kind, data))
        elif type(support) is ExtrudedSurface and type(support.directrix) is BezierDirectrix:
            # Keep both declared and actually evaluated coefficients. A mismatch
            # cannot be hidden by serializing only public controls/vector.
            directrix = support.directrix
            data = (tuple(_vector(p) for p in directrix.controls),
                    tuple(_vector(p) for p in directrix._array),
                    _vector(support.vector), _vector(support._vector),
                    _range(support.u_range), _range(support.v_range))
            rows.append((identifier, BEZIER_EXTRUSION, data))
        else:
            rows.append((identifier, 'unsupported', ()))
    return tuple(rows)
