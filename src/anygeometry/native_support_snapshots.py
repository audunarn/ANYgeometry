"""Detached actual native support coefficients, including nonserialized fields."""
import math

from .errors import GeometryError
from .curves import Straight, Spline, LineRange
from .extrusions import BezierDirectrix
from .surfaces import Plane, Cylinder, CoonsSurface, ExtrudedSurface
from .polynomial_extrusion_support import COONS_POLYNOMIAL, BEZIER_EXTRUSION
from .nurbs import NurbsSurface, NurbsCurve, nurbs_record

NURBS_TENSOR = 'nurbs_tensor'
NURBS_TRANSLATION = 'coons_nurbs_translation'


def _literal(value):
    """Detached finite native coefficients; no normalization through serde."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise GeometryError('native NURBS coefficients are invalid')
        return value
    if type(value) in (tuple, list):
        return tuple(_literal(item) for item in value)
    if type(value) is dict:
        return tuple(sorted((key, _literal(item)) for key, item in value.items()))
    raise GeometryError('native NURBS definition has unsupported mutable coefficients')


def _nurbs_tensor(surface):
    if type(surface) is not NurbsSurface:
        raise GeometryError('native NURBS support requires the literal owner type')
    return _literal((surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v,
                     surface.controls, surface.weights, surface.u_range, surface.v_range))


def _translation_snapshot(model, face):
    from .nurbs_sweep_chart import has_sweep_chart, chart_model, _tensor
    if has_sweep_chart(face):
        view = chart_model(model, face.id)
        profile, vector, tensor = view.profile, view.vector, view.support
    else:
        from ._translational_pairs import _sweep
        recognized = _sweep(model, face.id)
        if recognized is None:
            return None
        profile, vector, _ = recognized
        tensor = _tensor(profile, vector)
    rows = []
    for loop in (face.loop, *face.holes):
        boundary = []
        for use in loop:
            edge = model.edges[use.edge]
            if type(edge.curve) is Straight:
                curve = ('straight',)
            elif type(edge.curve) is LineRange:
                curve = ('line_range', edge.curve.a, edge.curve.b, edge.curve.exact_domain)
            elif type(edge.curve) is NurbsCurve:
                curve = ('profile', _literal(nurbs_record(edge.curve)))
            else:
                from .support_curves import PlaneNurbsCurve, plane_nurbs_record
                from .nurbs import CurveOnSurface
                if type(edge.curve) is PlaneNurbsCurve:
                    curve = ('section', _literal(plane_nurbs_record(edge.curve)))
                elif type(edge.curve) is CurveOnSurface:
                    curve = ('chart', _literal(nurbs_record(edge.curve)))
                else:
                    raise GeometryError('native exact sweep boundary family changed')
            boundary.append((edge.id, use.forward, _vector(model.vertex_position(edge.start)),
                             _vector(model.vertex_position(edge.end)), curve))
        rows.append(tuple(boundary))
    return (_literal(nurbs_record(profile)), tuple((v.numerator, v.denominator) for v in vector),
            _nurbs_tensor(tensor), tuple(rows))


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
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    validate_model_curve_runtimes(model)
    rows = []
    for identifier, face in sorted(model.faces.items()):
        support = face.surface
        translation = _translation_snapshot(model, face) if type(support) is CoonsSurface else None
        if translation is not None:
            rows.append((identifier, NURBS_TRANSLATION, translation))
        elif face.parameterization is not None:
            rows.append((identifier, 'unsupported_parameterization', ()))
        elif type(support) is NurbsSurface:
            rows.append((identifier, NURBS_TENSOR, _nurbs_tensor(support)))
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


def _material_support_read_pin(support):
    """Actual map coefficients without reconstructing an unchanged sweep recipe.

    The material-validation caller ALSO binds the complete current document
    and incidence. This is only the complementary nonserialized/native audit.
    Unknown kernels cannot acquire reusable material qualification.
    """
    if type(support) is NurbsSurface:
        return NURBS_TENSOR, _nurbs_tensor(support)
    if type(support) is Plane:
        return 'plane', tuple(_vector(getattr(support, key)) for key in ('origin', 'u_vector', 'v_vector'))
    if type(support) is Cylinder:
        import numpy as np
        if not np.array_equal(support._circumferential, np.cross(support.axis, support.radial_direction)):
            raise GeometryError('material validation native cylinder basis differs from its definition')
        return 'cylinder', (tuple(_vector(getattr(support, key)) for key in
            ('origin', 'axis', 'radial_direction', '_circumferential')),
            _literal((support.radius, support.height, support.start_angle, support.sweep_angle)))
    if type(support) is CoonsSurface:
        boundaries = tuple(getattr(support, key) for key in ('bottom', 'right', 'top', 'left'))
        if all(value is not None for value in boundaries):
            return 'coons_sampled', tuple(tuple(_vector(point) for point in getattr(support, key))
                for key in ('bottom', 'right', 'top', 'left'))
        return None  # empty/partial markers need FACE-scoped proof; no generic topology-cache audit
    if type(support) is ExtrudedSurface and type(support.directrix) is BezierDirectrix:
        import numpy as np
        if (not np.array_equal(support.directrix._array, np.asarray(support.directrix.controls))
                or not np.array_equal(support._vector, np.asarray(support.vector))):
            raise GeometryError('material validation native extrusion differs from its definition')
        # This directrix also owns mutable inversion/normal/origin caches.
        # They are outside this narrow read-pin service: keep FULL existing
        # qualification rather than grant longer-lived reuse from a partial
        # native fingerprint. Existing capture_native_supports is unchanged.
        return None
    return None


def capture_material_validation_read_pin(model):
    """Fresh native audit, with no sweep recognition/chart reconstruction.

    Public coefficients, recipes and complete incidence are independently
    fingerprinted by _qualified_model_state on EVERY cache lookup. Thus an
    unchanged geometric sweep recipe needs no repeated proof of its tensor:
    its stored parameterization is read literally, while actual curve kernels
    and mutable point/section caches are validated against public definitions.
    Existing capture_native_supports semantics remain unchanged.
    """
    from .nurbs_runtime_binding import validate_model_curve_runtimes
    validate_model_curve_runtimes(model)
    rows = []
    for identifier, face in sorted(model.faces.items()):
        if (type(face.surface) is CoonsSurface
                and all(getattr(face.surface, key) is None for key in ('bottom', 'right', 'top', 'left'))):
            from .nurbs_sweep_chart import has_sweep_chart
            if not has_sweep_chart(face) or type(face.parameterization) is not NurbsSurface:
                return None
            # Full initial qualification established this geometric recipe.
            # Its complete current encoding and literal tensor pin preserve
            # it without evaluating a generic topology-backed Coons map.
            support = ('coons_exact_sweep', ())
        else:
            support = _material_support_read_pin(face.surface)
        parameterization = None if face.parameterization is None else _material_support_read_pin(face.parameterization)
        if support is None or face.parameterization is not None and parameterization is None:
            return None
        rows.append((identifier, support, parameterization))
    return tuple(rows)
