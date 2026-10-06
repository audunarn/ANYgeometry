"""Detached actual native support coefficients, including nonserialized fields."""
import math

from .errors import GeometryError
from .surfaces import Plane, Cylinder


def _vector(value):
    row = tuple(float(x) for x in value)
    if len(row) != 3 or any(not math.isfinite(x) for x in row):
        raise GeometryError('native support coefficients are invalid')
    return row


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
        else:
            rows.append((identifier, 'unsupported', ()))
    return tuple(rows)
