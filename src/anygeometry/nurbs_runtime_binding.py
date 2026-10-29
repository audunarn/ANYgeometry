"""Validate evaluated NURBS kernels/caches against frozen public definitions.

Legitimate cache fills are not geometry changes. Divergent evaluated values
are rejected rather than incorporated into a geometry checksum.
"""
import numpy as np
from dataclasses import fields, replace
from fractions import Fraction
from .errors import GeometryError
from .nurbs import NurbsCurve, CurveOnSurface, backend

_SECTION_CANONICAL = {}


def _section_literal(value):
    from .support_curves import _Arc
    from .plane_nurbs import Root, Box
    kind = type(value)
    if kind in (int, float, str, bool) or value is None:
        return kind.__name__, value
    if kind is Fraction:
        return 'Q', value.numerator, value.denominator
    if kind in (tuple, list):
        return kind.__name__, tuple(_section_literal(v) for v in value)
    if kind in (_Arc, Root, Box):
        return kind.__name__, tuple((f.name, _section_literal(getattr(value, f.name))) for f in fields(value))
    if kind is np.ndarray:
        if value.dtype != np.dtype(float) or not np.all(np.isfinite(value)):
            raise GeometryError('plane NURBS runtime coefficient array is invalid')
        return 'array', value.shape, value.dtype.str, value.tobytes()
    raise GeometryError('plane NURBS runtime cache contains an unsupported field')


def _section_runtime(curve):
    from .support_curves import _surface_net, _absolute_box, _scaled_float_coefficients
    arcs = curve._arcs()
    return _section_literal((arcs, curve._plane_slab(), curve._world_bound(), _surface_net(curve.surface),
        tuple((_absolute_box(arc.box), _scaled_float_coefficients(arc.box)) for arc in arcs)))


def validate_section_runtime(curve):
    """Compare evaluated caches to a freshly derived, detached public oracle.

    Isolated rederivation retains the current qualification Work context.
    Canonical signatures contain no live Box/Root/array objects, and are
    retained only under the complete validated public source definition.
    """
    from .support_curves import PlaneNurbsCurve, _RUNTIME_CACHE_BANK, _plane_record
    from .nurbs import NurbsSurface
    from .surfaces import Plane
    if type(curve) is not PlaneNurbsCurve:
        raise GeometryError('plane NURBS runtime requires its native curve definition')
    # Reconstruction validates public key shape/ranges before hashing or using
    # it to launch any fresh numerical extraction.
    if (type(curve.path) is not int or curve.path < 0 or type(curve.surface) is not NurbsSurface
            or type(curve.plane_record) is not tuple or len(curve.plane_record) != 3
            or any(type(row) is not tuple or len(row) != 3 for row in curve.plane_record)):
        raise GeometryError('plane NURBS evaluated public definition is invalid')
    declared = PlaneNurbsCurve(_plane_record(Plane(*curve.plane_record)), replace(curve.surface), curve.path, curve.domain)
    key = (declared.plane_record, declared.surface, declared.path)
    canonical = _SECTION_CANONICAL.get(key)
    if canonical is None:
        token = _RUNTIME_CACHE_BANK.set({})
        try:
            canonical = _section_runtime(declared)
        finally:
            _RUNTIME_CACHE_BANK.reset(token)
        if len(_SECTION_CANONICAL) >= 64:
            _SECTION_CANONICAL.clear()
        _SECTION_CANONICAL[key] = canonical
    try:
        actual = _section_runtime(curve)
    except (ValueError, TypeError, AttributeError, IndexError, ZeroDivisionError) as cause:
        raise GeometryError('plane NURBS evaluated runtime cache is invalid') from cause
    if actual != canonical:
        raise GeometryError('plane NURBS evaluated runtime cache differs from its public definition')


def validate_curve_runtime(curve):
    return _validate_curve_runtime(curve)


def _validate_curve_runtime(curve, bank=None):
    from ._canonical_curve_runtime import (expected_prepared, expected_row,
                                           kernel_context, begin_validation, _compute_prepared)
    try:
        curve._require_prepared_binding()
    except (ValueError, TypeError, AttributeError) as error:
        raise GeometryError('NURBS evaluated definition binding is invalid') from error
    try:
        initial_context = begin_validation()
        fresh = _compute_prepared(curve)
    except (ValueError, TypeError, OverflowError) as error:
        raise GeometryError('NURBS evaluated definition is invalid') from error
    actual = curve._prepared
    if (type(actual) is not tuple or len(actual) != len(fresh) or type(actual[0]) is not int or actual[0] != fresh[0]
            or any(type(a) is not np.ndarray or a.dtype != b.dtype or a.shape != b.shape
                   or not np.array_equal(a, b)
                   for a, b in zip(actual[1:], fresh[1:]))):
        raise GeometryError('NURBS evaluated kernel differs from its public definition')
    canonical = expected_prepared(curve, bank, prepared=fresh)
    endpoints, points = curve._endpoint_values, curve._point_values
    if type(endpoints) is not tuple or len(endpoints) != 2 or type(points) is not dict or len(points) > 128:
        raise GeometryError('NURBS evaluated point cache is invalid')
    entries = [(float(k), v) for k, v in zip(curve.domain, endpoints) if v is not None]
    for key, value in tuple(points.items()):
        if type(key) is not bytes or len(key) != 8:
            raise GeometryError('NURBS evaluated point cache has an invalid station')
        knot = float(np.frombuffer(key, dtype=np.float64)[0])
        if not np.isfinite(knot) or not min(curve.domain) <= knot <= max(curve.domain):
            raise GeometryError('NURBS evaluated point cache station exceeds its definition')
        entries.append((knot, value))
    for knot, value in entries:
        try:
            expected = (backend._evaluate_prepared_bspline_curve(fresh, np.array([knot]), 0)
                        if canonical is None else expected_row(canonical[0], canonical[1], np.float64(knot).tobytes()))
            if canonical is not None:
                # Expected bytes are immutable independent truth, but actual
                # admission retains the original numeric equality contract:
                # signed zeros compare equal and NaNs never compare equal.
                dtype, shape, data = expected
                expected = np.frombuffer(data, dtype=dtype).reshape(shape)
        except (ValueError, TypeError, ArithmeticError) as error:
            raise GeometryError('NURBS evaluated point cache cannot be qualified') from error
        if (type(value) is not np.ndarray or value.shape != expected.shape
                or value.dtype != expected.dtype or not np.array_equal(value, expected)):
            raise GeometryError('NURBS evaluated point cache differs from its public definition')
    if kernel_context() != initial_context:
        raise GeometryError('NURBS backend changed during runtime validation')


def validate_model_curve_runtimes(model):
    from .support_curves import PlaneNurbsCurve
    from .line_ranges import LineRange
    from ._canonical_curve_runtime import model_bank, kernel_context
    curves = [edge.curve if type(edge.curve) is NurbsCurve else edge.curve.pcurve
              for edge in model.edges.values() if type(edge.curve) in (NurbsCurve, CurveOnSurface)]
    bank = model_bank(model, curves)
    sections = set()
    for edge in model.edges.values():
        if type(edge.curve) is LineRange:
            edge.curve.validate_definition()
        elif type(edge.curve) is NurbsCurve:
            _validate_curve_runtime(edge.curve, bank)
        elif type(edge.curve) is CurveOnSurface:
            _validate_curve_runtime(edge.curve.pcurve, bank)
        elif type(edge.curve) is PlaneNurbsCurve:
            key = (edge.curve.plane_record, edge.curve.surface, edge.curve.path)
            if key not in sections:
                validate_section_runtime(edge.curve)
                sections.add(key)
    if bank is not None and kernel_context() != bank.context:
        raise GeometryError('NURBS backend changed during model runtime validation')
