"""Surface capability registry (generality slice A1a, architecture section 1).

References: the existing batch evaluation itself (bit-identical delegation),
exact rational geometry for the plane's proof objects (enclosure, Jacobian
bounds, implicit form, exact chart), and the existing transform helper.
"""
from __future__ import annotations

from fractions import Fraction
import math
import random

import numpy as np
import pytest

from anygeometry import GeometryError, Cone, CoonsSurface, Cylinder, ExtrudedSurface, Plane, RuledSurface
from anygeometry.exact_predicates import exact_vector
from anygeometry.extrusions import BezierDirectrix, EllipseDirectrix
from anygeometry.planar_exact import ExactPlane
from anygeometry.surface_capabilities import SurfaceKind, capabilities
from anygeometry.surfaces import _evaluate_surface_many, _surface_derivatives_many


def same_bits(first, second):
    first, second = np.ascontiguousarray(first), np.ascontiguousarray(second)
    return first.shape == second.shape and np.array_equal(first.view(np.uint64), second.view(np.uint64))


def surfaces():
    square = np.linspace(0., 1., 5)
    return {
        SurfaceKind.PLANE: Plane((1., -2., .5), (1.3, .2, -.1), (-.2, .9, .4)),
        SurfaceKind.CYLINDER: Cylinder((0., 0., -2.), (0., 0., 1.), (1., 0., 0.), 1., 4.),
        SurfaceKind.CONE: Cone((-2., .3, .8), (1., 0., 0.), (0., 1., 0.), .15, 1.1, 8.),
        SurfaceKind.EXTRUDED: ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 2., 0.), (2., 0., 0.))), (0., 0., 2.)),
        SurfaceKind.RULED: RuledSurface(np.array(((2., 0., 0.), (2., -1., .3))), np.array(((2., 0., 3.), (2.4, -1., 3.)))),
        SurfaceKind.COONS: CoonsSurface(
            bottom=np.column_stack((square, 0 * square, .1 * np.sin(3 * square))),
            right=np.column_stack((1 + 0 * square, square, .1 * np.sin(3.) + .2 * square)),
            top=np.column_stack((square, 1 + 0 * square, .2 + .1 * np.sin(3 * square))),
            left=np.column_stack((0 * square, square, .2 * square))),
    }


@pytest.mark.parametrize("kind", list(SurfaceKind)[:6])
def test_evaluation_and_derivatives_are_bit_identical_to_the_engine(kind):
    surface = surfaces()[kind]
    capability = capabilities(surface)
    assert capability.kind is kind and capability.definition is surface
    uv = np.random.default_rng(3).uniform(0., 1., (10000, 2))
    assert same_bits(capability.evaluate(uv), _evaluate_surface_many(surface, uv))
    for got, expected in zip(capability.derivatives(uv), _surface_derivatives_many(surface, uv)):
        assert same_bits(got, expected)


@pytest.mark.parametrize("kind", list(SurfaceKind)[:6])
def test_local_inversion_and_transform_delegate(kind):
    from anygeometry.operations import _transform_surface
    surface = surfaces()[kind]
    capability = capabilities(surface)
    points = capability.evaluate(np.array([[.3, .4], [.7, .2]]))
    result = capability.invert_local(points)
    assert np.array_equal(result.uv, np.array([surface.local_uv(p) for p in points]))
    matrix = np.eye(4)
    matrix[:3, 3] = (1., 2., 3.)
    moved = capability.transformed(matrix)
    expected = _transform_surface(surface, matrix)
    assert np.array_equal(_evaluate_surface_many(moved, np.array([[.25, .5]])),
                          _evaluate_surface_many(expected, np.array([[.25, .5]])))


@pytest.mark.parametrize("kind", list(SurfaceKind)[1:6])
def test_proof_objects_are_absent_until_a_route_provides_them(kind):
    """Architecture 1.1 rule 4: None means a route needing the object refuses."""
    capability = capabilities(surfaces()[kind])
    assert capability.enclosure is None and capability.jacobian_bounds is None
    assert capability.exact_uv is None and capability.implicit is None


def test_plane_enclosure_contains_the_exact_patch():
    plane = surfaces()[SurfaceKind.PLANE]
    capability = capabilities(plane)
    chart = ExactPlane.from_plane(plane)
    rng = random.Random(5)
    for _ in range(200):
        u0, u1 = sorted((rng.uniform(-3, 3), rng.uniform(-3, 3)))
        v0, v1 = sorted((rng.uniform(-3, 3), rng.uniform(-3, 3)))
        box = capability.enclosure(((u0, u1), (v0, v1)))
        for _ in range(20):
            u, v = Fraction(rng.uniform(u0, u1)), Fraction(rng.uniform(v0, v1))
            point = chart.world(u, v)
            assert all(Fraction(lo) <= x <= Fraction(hi) for lo, x, hi in zip(box.lower, point, box.upper))
        for u in (u0, u1):                              # tight: the corners attain the extremes
            for v in (v0, v1):
                point = chart.world(Fraction(u), Fraction(v))
                assert all(abs(float(x) - lo) <= 2 * math.ulp(abs(float(x)) + 1e-300) or float(x) > lo
                           for lo, x in zip(box.lower, point))


def gram_brackets(plane, bounds):
    """Exact: sigma_max^2 I - G is PSD and G - sigma_min^2 I is PSD (G the Gram matrix of [u v])."""
    u, v = exact_vector(plane.u_vector), exact_vector(plane.v_vector)
    uu, uv, vv = (sum(a * b for a, b in zip(x, y)) for x, y in ((u, u), (u, v), (v, v)))
    high, low = Fraction(bounds.sigma_max) ** 2, Fraction(bounds.sigma_min) ** 2
    upper_ok = high >= uu and high >= vv and (high - uu) * (high - vv) >= uv * uv
    lower_ok = uu >= low and vv >= low and (uu - low) * (vv - low) >= uv * uv
    return upper_ok and lower_ok


def test_plane_jacobian_bounds_bracket_the_singular_values():
    plane = surfaces()[SurfaceKind.PLANE]
    bounds = capabilities(plane).jacobian_bounds(((0., 1.), (0., 1.)))
    assert gram_brackets(plane, bounds)
    singular = np.linalg.svd(np.column_stack((plane.u_vector, plane.v_vector)), compute_uv=False)
    assert bounds.sigma_max <= singular[0] * (1 + 1e-14) and bounds.sigma_min >= singular[1] * (1 - 1e-14)
    rng = random.Random(3)
    for _ in range(300):
        scale = 10. ** rng.uniform(-150, 150)
        u = [rng.uniform(-1, 1) * scale for _ in range(3)]
        v = [rng.uniform(-1, 1) * scale * 10. ** rng.uniform(-5, 5) for _ in range(3)]
        try:
            candidate = Plane((0., 0., 0.), u, v)
        except Exception:
            continue
        assert gram_brackets(candidate, capabilities(candidate).jacobian_bounds(((0., 1.), (0., 1.))))


def test_plane_capabilities_at_extreme_magnitudes():
    """Review A1a-2: no overflow for representable planes; outward rounding saturates to +-inf."""
    wide = Plane((0., 0., 0.), (1e155, 0., 0.), (0., 1., 0.))
    bounds = capabilities(wide).jacobian_bounds(((0., 1.), (0., 1.)))
    assert gram_brackets(wide, bounds) and bounds.sigma_max >= 1e155
    big = Plane((0., 0., 0.), (1e155, 0., 0.), (0., 1e155, 0.))
    assert capabilities(big).implicit.gradient_inf == math.inf               # 1e310 is above the float range
    far = Plane((1e308, 0., 0.), (1., 0., 0.), (0., 1., 0.))
    box = capabilities(far).enclosure(((1e308, 1e308), (0., 0.)))
    assert box.upper[0] == math.inf and box.lower[0] == np.finfo(float).max


def test_plane_enclosure_over_infinite_boxes():
    """Review A1a-3: the advertised domain and half-infinite boxes are enclosed (0 * inf = 0)."""
    plane = Plane((1., 2., 3.), (1., 0., 0.), (0., 1., 0.))
    capability = capabilities(plane)
    whole = capability.enclosure(capability.domain)
    assert whole.lower == (-math.inf, -math.inf, 3.) and whole.upper == (math.inf, math.inf, 3.)
    half = capability.enclosure(((0., math.inf), (1., 0.)))
    assert half.lower == (1., 2., 3.) and half.upper == (math.inf, 3., 3.)
    tilted = capabilities(Plane((0., 0., 0.), (1., -1., 0.), (0., 0., 1.))).enclosure(((-math.inf, 0.), (0., 1.)))
    assert tilted.lower == (-math.inf, 0., 0.) and tilted.upper == (0., math.inf, 1.)
    for bad in (((math.inf, math.inf), (0., 1.)), ((0., 1.), (-math.inf, -math.inf)), ((math.nan, 1.), (0., 1.))):
        with pytest.raises(GeometryError):
            capability.enclosure(bad)


def test_plane_exact_chart_and_implicit_form_are_exact():
    plane = surfaces()[SurfaceKind.PLANE]
    capability = capabilities(plane)
    chart = ExactPlane.from_plane(plane)
    a, b, c, d = capability.implicit.coefficients
    rng = random.Random(7)
    for _ in range(100):
        u, v = Fraction(rng.uniform(-5, 5)), Fraction(rng.uniform(-5, 5))
        point = chart.world(u, v)
        assert a * point[0] + b * point[1] + c * point[2] + d == 0
        assert capability.exact_uv(point) == (u, v)
    off = tuple(float(x) for x in np.asarray(plane.origin) + 1e-3 * np.cross(plane.u_vector, plane.v_vector))
    assert capability.exact_uv(off) == chart.uv(off)
    assert capability.implicit.degree == 1 and capability.implicit.gradient_inf >= max(abs(float(a)), abs(float(b)), abs(float(c)))


def test_derived_data_is_memoised_without_keeping_definitions_alive():
    """Review A1a-1/4: one derived record per definition, dropped on collection, consistent across threads."""
    import gc
    import threading
    import weakref
    from anygeometry import surface_capabilities as module
    plane = surfaces()[SurfaceKind.PLANE]
    first, second = capabilities(plane), capabilities(plane)
    assert first.implicit is second.implicit and first.jacobian_bounds(None) is second.jacobian_bounds(None)
    reference, key = weakref.ref(plane), id(plane)
    del plane, first, second
    gc.collect()
    assert reference() is None and key not in module._DERIVED
    plane = Plane((0., 1., 2.), (1., 2., 0.), (0., 1., 3.))
    results, barrier = [], threading.Barrier(8)

    def worker():
        barrier.wait()
        results.append(capabilities(plane).implicit)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert all(result is results[0] for result in results)
    with pytest.raises(TypeError):
        capabilities(object())
