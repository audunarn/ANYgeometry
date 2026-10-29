"""Branch-chart clipping reuses its angle- and quadric-independent terms, with bit-identical results.

Tripod finite-domain clipping spent most of its time re-forming the same support
terms of each cylinder/cylinder branch: three to ten ``np.cross`` products per
point evaluation and the exact rational branch chart for every plane root. These
tests fail on the old code (the work is repeated) and pin that reuse changes no
value: every evaluation, root and eliminated polynomial is identical to a fresh
computation and to a frozen copy of the original elimination formula.
"""
import math

import numpy as np
import pytest

import anygeometry.exact_curves as exact_curves
import anygeometry.quadric_curve_events as quadric_curve_events
from anygeometry.analytic_supports import cylinder_cylinder_support
from anygeometry.arrangement_geometry import plane_roots
from anygeometry.cylinder_curve_events import _add, _multiply, _quadratic, _scale
from anygeometry.exact_curves import CylinderIntersectionCurve
from anygeometry.surfaces import Cylinder


def _clear():
    # Tolerant of the original modules, so the old code fails on the repeated work itself.
    getattr(exact_curves, "_BRANCH_CONSTANTS", {}).clear()
    getattr(quadric_curve_events, "_BRANCH_POLYNOMIALS", {}).clear()


def _branches():
    """Leg/brace-like crossings, full and sector skins, including regular-endpoint charts."""
    leg = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6.)
    sector = Cylinder((0., 0., -3.), (0., 0., 1.), (1., 0., 0.), 2., 6., .3, 1.1)
    curves = []
    for axis, radius in (((0., 1., 0.), 1.), ((1., 2., 1.), .75), ((0., 1., 0.), 2.), ((.3, 1., .2), .5)):
        unit = np.asarray(axis)/np.linalg.norm(axis)
        brace = Cylinder(-3*unit, unit, (1., 0., 0.), radius, 6.)
        for first in (leg, sector):
            curves.extend(c for c in cylinder_cylinder_support(first, brace).curves
                          if isinstance(c, CylinderIntersectionCurve))
    assert {c.parameterization for c in curves} >= {"linear"}
    extra = []
    for curve in curves[:6]:
        extra.extend((curve.subcurve(0., .37), curve.subcurve(.21, 1.), curve.subcurve(.8, .1)))
        extra.append(curve.transformed(((0., -1., 0., .5), (1., 0., 0., -.25), (0., 0., 1., 2.), (0., 0., 0., 1.))))
    return curves+extra


PARAMETERS = np.asarray((0., 1e-9, .125, .25, .5, .75, .999999, 1.))


class _CountingNumpy:
    def __init__(self):
        self.count = 0

    def __getattr__(self, name):
        attribute = getattr(np, name)
        if name == "cross":
            def counted(*args, **kwargs):
                self.count += 1
                return attribute(*args, **kwargs)
            return counted
        return attribute


def test_branch_evaluation_forms_its_support_cross_products_once(monkeypatch):
    curve = _branches()[0]
    _clear()
    counter = _CountingNumpy()
    monkeypatch.setattr(exact_curves, "np", counter)
    for t in np.linspace(0., 1., 50):
        curve.evaluate(float(t))
    # Old code: two to ten products per evaluation (>= 100 here). Reused: one set per support pair.
    assert counter.count <= 3


def test_plane_roots_form_the_rational_branch_chart_once(monkeypatch):
    curve = _branches()[0]
    _clear()
    calls = []
    monkeypatch.setattr(quadric_curve_events, "_quadratic",
                        lambda *args: calls.append(1) or _quadratic(*args))
    for normal, offset in (((0., 0., 1.), .1), ((1., 0., 0.), .3), ((0., 1., 0.), -.2),
                           ((1., 1., 0.), .05), ((0., 1., 1.), .4), ((1., -1., 2.), -.6)):
        plane_roots(curve, np.asarray(normal)/np.linalg.norm(normal), offset, tolerance=1e-10)
        plane_roots(curve.subcurve(.1, .9), np.asarray(normal)/np.linalg.norm(normal), offset, tolerance=1e-10)
    # Old code: once per call (12). Subcurves share the immutable supports.
    assert len(calls) == 1


def _bits(value):
    array = np.asarray(value, dtype=float)
    return array.tobytes(), np.signbit(array).tobytes()


def _snapshot(curve, quadrics):
    record = [_bits(curve.evaluate(PARAMETERS)), _bits(curve.derivative(PARAMETERS[2:-2]))]
    for t in PARAMETERS:
        record.append(_bits(curve.evaluate(float(t))))
    for matrix, linear, constant in quadrics:
        record.append(quadric_curve_events.quadric_roots(curve, matrix, linear, constant, tolerance=1e-10))
    return record


def _quadrics():
    rng = np.random.default_rng(20261029)
    quadrics = [(np.zeros((3, 3)), .5*np.asarray(n, float), -o)
                for n, o in (((0., 0., 1.), .1), ((1., 0., 0.), .3), ((0., .6, .8), -.2), ((-0., 1., 0.), 0.))]
    for _ in range(4):
        m = rng.normal(size=(3, 3)); m = m+m.T
        m[0, 2] = m[2, 0] = 0.
        quadrics.append((m, rng.normal(size=3), float(rng.normal())))
    return quadrics


def test_reused_terms_are_bit_identical_to_fresh_computation():
    quadrics = _quadrics()
    for curve in _branches():
        _clear()
        fresh = []
        # A fresh computation for every query: clear the reuse tables before each one.
        fresh.append(_bits(curve.evaluate(PARAMETERS))); _clear()
        fresh.append(_bits(curve.derivative(PARAMETERS[2:-2]))); _clear()
        for t in PARAMETERS:
            fresh.append(_bits(curve.evaluate(float(t)))); _clear()
        for matrix, linear, constant in quadrics:
            fresh.append(quadric_curve_events.quadric_roots(curve, matrix, linear, constant, tolerance=1e-10))
            _clear()
        _snapshot(curve, quadrics)                      # warm
        assert _snapshot(curve, quadrics) == fresh


def _reference_polynomial(curve, matrix, linear, constant):
    """Frozen copy of the original (pre-reuse) elimination, every zero-weight term included."""
    from fractions import Fraction
    matrix = np.asarray(matrix); linear = np.asarray(linear)
    transform = np.asarray(curve.transform)
    first = curve.first
    origin = transform[:3, :3] @ first.origin+transform[:3, 3]
    cosine = transform[:3, :3] @ (first.radius*np.asarray(first.radial_direction))
    sine = transform[:3, :3] @ (first.radius*np.cross(first.axis, first.radial_direction))
    direction = transform[:3, :3] @ first.axis
    numerator = [tuple(Fraction(float(value)) for value in (o+c, 2*s, o-c)) for o, c, s in zip(origin, cosine, sine)]
    radius = (1, 0, 1)
    d = Fraction(float(direction @ matrix @ direction))
    e = _scale(radius, Fraction(float(2*linear @ direction)))
    h = _scale(_multiply(radius, radius), Fraction(float(constant)))
    for i in range(3):
        e = _add(e, _scale(numerator[i], Fraction(float(2*(matrix @ direction)[i]))))
        h = _add(h, _scale(_multiply(numerator[i], radius), Fraction(float(2*linear[i]))))
        for j in range(3):
            h = _add(h, _scale(_multiply(numerator[i], numerator[j]), Fraction(float(matrix[i, j]))))
    a, b, c = _quadratic(curve, curve.second, curve.transform)
    m = _add(_scale(h, a), _scale(c, -d))
    l = _add(_scale(e, a), _scale(b, -d))
    bh_ec = _add(_multiply(b, h), _scale(_multiply(e, c), -1))
    return _add(_multiply(m, m), _scale(_multiply(l, bh_ec), -1))


def test_eliminated_polynomial_equals_the_original_formula(monkeypatch):
    seen = []
    original = quadric_curve_events.isolate_real_roots

    def recording(coefficients, **kwargs):
        seen.append(tuple(coefficients))
        return original(coefficients, **kwargs)

    monkeypatch.setattr(quadric_curve_events, "isolate_real_roots", recording)
    for curve in _branches()[:10]:
        for matrix, linear, constant in _quadrics():
            seen.clear()
            quadric_curve_events.quadric_roots(curve, matrix, linear, constant, tolerance=1e-10)
            reference = _reference_polynomial(curve, matrix, linear, constant)
            if reference == (0,):
                assert not seen
                continue
            assert seen[0] == reference and seen[1] == tuple(reversed(reference))
            assert [type(v) for v in seen[0]] == [type(v) for v in reference]


def test_reuse_tables_are_bounded_and_hold_their_keys_alive():
    curve = _branches()[0]
    _clear()
    curve.evaluate(.5)
    plane_roots(curve, np.asarray((0., 0., 1.)), .1, tolerance=1e-10)
    (first, second, _), = exact_curves._BRANCH_CONSTANTS.values()
    assert first is curve.first and second is curve.second
    (first, second, _), = quadric_curve_events._BRANCH_POLYNOMIALS.values()
    assert first is curve.first and second is curve.second
    constants = exact_curves._branch_constants(curve)
    with pytest.raises(ValueError):
        constants.cross[0] = math.nan                  # shared arrays are read-only
    assert exact_curves._BRANCH_CONSTANTS_LIMIT == quadric_curve_events._BRANCH_POLYNOMIALS_LIMIT == 4096
