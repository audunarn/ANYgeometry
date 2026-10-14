"""NURBS surface capabilities (generality slice A3a, architecture 1.1, 3.1, 3.5).

References: exact rational evaluation (Fraction de Boor on the stored doubles),
the analytic cylinder and the exact bilinear formula, finite differences for
derivatives, and 1e5 float samples for the proof objects.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError
from anygeometry.nurbs import NurbsCurve, NurbsSurface, cylinder_to_nurbs
from anygeometry.surface_capabilities import SurfaceKind, capabilities


def rational_patch():
    rng = np.random.default_rng(11)
    count_u, count_v = 7, 5
    controls = np.zeros((count_u, count_v, 3))
    controls[..., 0] = np.linspace(0., 6., count_u)[:, None]
    controls[..., 1] = np.linspace(0., 3., count_v)[None, :]
    controls[..., 2] = rng.uniform(-1., 1., (count_u, count_v))
    controls += rng.uniform(-.2, .2, controls.shape)
    weights = rng.uniform(.5, 4., (count_u, count_v))
    return NurbsSurface(
        3, 2, (0., 0., 0., 0., .3, .3, .7, 1., 1., 1., 1.), (-1., -1., -1., .5, .5, 2., 2., 2.),
        tuple(map(tuple, map(lambda row: tuple(map(tuple, row)), controls))),
        tuple(map(tuple, weights)), u_range=(.9, .1), v_range=(-.5, 1.7))


def bilinear_patch():
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        (((0., 0., 0.), (0., 2., 1.)), ((3., 0., 1.), (2.5, 2.5, -1.))))


def cylinder_patch():
    return cylinder_to_nurbs(Cylinder((1., -2., .5), (0., 0., 1.), (1., 0., 0.), 2., 3., .4, 4.))


SURFACES = {"rational": rational_patch, "bilinear": bilinear_patch, "cylinder": cylinder_patch}


def random_boxes(rng, count):
    boxes = []
    for k in range(count):
        width = 10. ** rng.uniform(-6, 0)
        u0, v0 = rng.uniform(0., 1. - width, 2)
        boxes.append(((u0, u0 + width), (v0 + width, v0)) if k % 2 else ((u0 + width, u0), (v0, v0 + width)))
    boxes.append(((0., 1.), (0., 1.)))
    boxes.append(((.3, .3), (.6, .6)))                                   # a point
    return boxes


def samples_in(rng, box, count):
    (u0, u1), (v0, v1) = box
    return np.column_stack((rng.uniform(min(u0, u1), max(u0, u1), count),
                            rng.uniform(min(v0, v1), max(v0, v1), count)))


@pytest.mark.parametrize("name", list(SURFACES))
def test_enclosures_contain_1e5_samples_and_the_exact_patch(name):
    surface = SURFACES[name]()
    capability = capabilities(surface)
    assert capability.kind is SurfaceKind.NURBS
    rng = np.random.default_rng(5)
    boxes = random_boxes(rng, 48)
    total = 0
    for box in boxes:
        enclosure = capability.enclosure(box)
        uv = samples_in(rng, box, 2000)
        points = capability.evaluate(uv)
        total += len(points)
        assert np.all(points >= np.asarray(enclosure.lower)) and np.all(points <= np.asarray(enclosure.upper))
        for u, v in uv[:3]:
            point = surface.exact_point(Fraction(u), Fraction(v))
            assert all(Fraction(lo) <= x <= Fraction(hi) for lo, x, hi in zip(enclosure.lower, point, enclosure.upper))
    assert total >= 100000


@pytest.mark.parametrize("name", list(SURFACES))
def test_jacobian_bounds_hold_on_1e5_samples_and_tighten(name):
    surface = SURFACES[name]()
    capability = capabilities(surface)
    rng = np.random.default_rng(6)
    for box in random_boxes(rng, 48):
        bound = capability.jacobian_bounds(box)
        assert bound.sigma_min == 0.0
        s_u, s_v = capability.derivatives(samples_in(rng, box, 2000))
        sigma = np.linalg.svd(np.stack((s_u, s_v), axis=2), compute_uv=False)[:, 0]
        assert np.all(sigma <= bound.sigma_max * (1 + 1e-12))
    # a small box: the bound converges to the true maximum (measured 1.07-1.28x; Frobenius and knot ratios)
    box = ((.41, .41 + 1e-4), (.52, .52 + 1e-4))
    s_u, s_v = capability.derivatives(samples_in(rng, box, 200))
    sigma = np.linalg.svd(np.stack((s_u, s_v), axis=2), compute_uv=False)[:, 0].max()
    assert capability.jacobian_bounds(box).sigma_max <= 1.5 * sigma


@pytest.mark.parametrize("name", list(SURFACES))
def test_derivatives_match_finite_differences_and_exact_points_match_evaluation(name):
    surface = SURFACES[name]()
    capability = capabilities(surface)
    rng = np.random.default_rng(7)
    uv = rng.uniform(.05, .95, (200, 2))
    h = 1e-6
    s_u, s_v = capability.derivatives(uv)
    fd_u = (capability.evaluate(uv + [h, 0.]) - capability.evaluate(uv - [h, 0.])) / (2 * h)
    fd_v = (capability.evaluate(uv + [0., h]) - capability.evaluate(uv - [0., h])) / (2 * h)
    scale = max(np.abs(s_u).max(), np.abs(s_v).max())
    assert np.abs(s_u - fd_u).max() <= 1e-6 * scale and np.abs(s_v - fd_v).max() <= 1e-6 * scale
    points = capability.evaluate(uv[:30])
    for (u, v), point in zip(uv[:30], points):
        exact = surface.exact_point(u, v)
        assert np.allclose([float(x) for x in exact], point, rtol=0, atol=1e-13 * (1 + np.abs(point).max()))
    with pytest.raises(ValueError):
        capability.derivatives(uv, order=2)


def test_bilinear_patch_is_exact():
    surface = bilinear_patch()
    corners = [[tuple(map(Fraction, point)) for point in row] for row in surface.controls]
    for u in (Fraction(0), Fraction(1, 3), Fraction(1, 2), Fraction(5, 7), Fraction(1)):
        for v in (Fraction(0), Fraction(1, 4), Fraction(2, 3), Fraction(1)):
            expected = tuple((1 - u) * (1 - v) * a + u * (1 - v) * b + (1 - u) * v * c + u * v * d
                             for a, b, c, d in zip(corners[0][0], corners[1][0], corners[0][1], corners[1][1]))
            assert surface.exact_point(u, v) == expected
    enclosure = capabilities(surface).enclosure(((0., 1.), (0., 1.)))
    flat = [point for row in surface.controls for point in row]
    corner_low = np.array([min(p[k] for p in flat) for k in range(3)])
    corner_high = np.array([max(p[k] for p in flat) for k in range(3)])
    margin = 16 * 4 * np.finfo(float).eps * 3. * 2                    # float-evaluation margin (and outward steps)
    assert np.all(np.asarray(enclosure.lower) <= corner_low) and np.all(np.asarray(enclosure.lower) >= corner_low - margin)
    assert np.all(np.asarray(enclosure.upper) >= corner_high) and np.all(np.asarray(enclosure.upper) <= corner_high + margin)


def test_iso_curves_follow_the_surface():
    for surface in (rational_patch(), cylinder_patch()):
        t = np.linspace(0., 1., 101)
        for value in (0., .37, 1.):
            along_v = surface.iso_curve("u", value)
            along_u = surface.iso_curve("v", value)
            assert isinstance(along_v, NurbsCurve) and along_v.rational == (surface.weights is not None)
            expected_v = surface.evaluate_many(np.column_stack((np.full_like(t, value), t)))
            expected_u = surface.evaluate_many(np.column_stack((t, np.full_like(t, value))))
            scale = 1 + np.abs(expected_v).max()
            assert np.abs(along_v.evaluate(t) - expected_v).max() <= 1e-13 * scale
            assert np.abs(along_u.evaluate(t) - expected_u).max() <= 1e-13 * scale
    with pytest.raises(GeometryError):
        rational_patch().iso_curve("w", .5)
    with pytest.raises(GeometryError):
        rational_patch().iso_curve("u", 1.5)


@pytest.mark.parametrize("sweep", (math.pi / 3, 4., 2 * math.pi, -1.5))
def test_cylinder_to_nurbs_is_the_cylinder(sweep):
    cylinder = Cylinder((1., -2., .5), (0., 1., 1.), (1., 0., 0.), 2., 3., .4, sweep)
    surface = cylinder_to_nurbs(cylinder)
    uv = np.random.default_rng(8).uniform(0., 1., (20000, 2))
    points = surface.evaluate_many(uv)
    axis = np.asarray(cylinder.axis)
    relative = points - np.asarray(cylinder.origin)
    height = relative @ axis
    radial = np.linalg.norm(relative - height[:, None] * axis, axis=1)
    assert np.abs(radial - 2.).max() <= 1e-14 * 8
    assert np.allclose(height, 3. * uv[:, 1], rtol=0, atol=1e-14 * 8)
    from anygeometry.surface_capabilities import capabilities as caps
    cylinder_points = caps(cylinder).evaluate(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]))
    nurbs_points = surface.evaluate_many(np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.]]))
    assert np.allclose(cylinder_points, nurbs_points, rtol=0, atol=1e-14 * 8)
    enclosure = capabilities(surface).enclosure(((0., 1.), (0., 1.)))
    on_cylinder = caps(cylinder).evaluate(uv)                         # the same point set, other chart
    assert np.all(on_cylinder >= np.asarray(enclosure.lower) - 1e-14 * 8)
    assert np.all(on_cylinder <= np.asarray(enclosure.upper) + 1e-14 * 8)


def test_local_inversion_recovers_parameters():
    surface = rational_patch()
    capability = capabilities(surface)
    uv = np.random.default_rng(9).uniform(.02, .98, (300, 2))
    result = capability.invert_local(capability.evaluate(uv))
    assert np.median(result.residual) <= 1e-12
    good = result.residual <= 1e-12
    assert np.abs(result.uv[good] - uv[good]).max() <= 1e-9
    seeded = capability.invert_local(capability.evaluate(uv[:10]), uv[:10] + 1e-3)
    assert seeded.residual.max() <= 1e-12
    assert capability.invert_local(np.zeros((0, 3))).uv.shape == (0, 2)


def test_transform_and_refusals():
    from anygeometry.operations import _transform_surface
    surface = rational_patch()
    matrix = np.eye(4)
    c, s = math.cos(.3), math.sin(.3)
    matrix[:3, :3] = ((c, 0., s), (0., 1., 0.), (-s, 0., c))
    matrix[:3, 3] = (1., 2., -3.)
    moved = capabilities(surface).transformed(matrix)
    assert isinstance(moved, NurbsSurface) and moved == _transform_surface(surface, matrix)
    uv = np.random.default_rng(10).uniform(0., 1., (100, 2))
    expected = surface.evaluate_many(uv) @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.abs(moved.evaluate_many(uv) - expected).max() <= 1e-13 * 10
    capability = capabilities(surface)
    assert capability.exact_uv is None and capability.implicit is None and capability.periodic_u is None
    with pytest.raises(GeometryError):
        capability.enclosure(((-.1, .5), (0., 1.)))
    with pytest.raises(GeometryError):
        capability.jacobian_bounds(((0., 1.), (0., 1.5)))


def far_patch():
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        tuple(tuple((float(i), float(j), 1e100) for j in range(2)) for i in range(2)),
                        ((1., 2.), (3., 4.)))


def test_derivatives_are_translation_invariant():
    """Review A3a blocking: a translated patch has the same (here exactly zero z) derivatives."""
    surface = far_patch()
    uv = np.random.default_rng(12).uniform(0., 1., (500, 2))
    s_u, s_v = surface.derivatives_many(uv)
    assert np.all(s_u[:, 2] == 0.) and np.all(s_v[:, 2] == 0.)
    near = NurbsSurface(1, 1, surface.knots_u, surface.knots_v,
                        tuple(tuple((p[0], p[1], 0.) for p in row) for row in surface.controls), surface.weights)
    n_u, n_v = near.derivatives_many(uv)
    np.testing.assert_allclose(s_u[:, :2], n_u[:, :2], rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(s_v[:, :2], n_v[:, :2], rtol=1e-14, atol=1e-14)
    bound = capabilities(surface).jacobian_bounds(((0., 1.), (0., 1.))).sigma_max
    sigma = np.linalg.svd(np.stack((s_u, s_v), axis=2), compute_uv=False)[:, 0]
    assert np.all(sigma <= bound)


def test_enclosures_contain_float_evaluations_of_far_patches():
    surface = far_patch()
    capability = capabilities(surface)
    rng = np.random.default_rng(13)
    for box in random_boxes(rng, 30):
        enclosure = capability.enclosure(box)
        points = capability.evaluate(samples_in(rng, box, 500))
        assert np.all(points >= np.asarray(enclosure.lower)) and np.all(points <= np.asarray(enclosure.upper))


def test_iso_curves_below_the_curve_coordinate_range_are_refused_clearly():
    tiny = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        tuple(tuple(((2 * i - 1) * 2. ** -500, float(j), 0.) for j in range(2)) for i in range(2)))
    with pytest.raises(GeometryError, match="coordinate range"):
        tiny.iso_curve("u", .75)
    assert tiny.iso_curve("u", 1.).evaluate(.5)[0] == 2. ** -500


def test_box_results_are_shared_between_enclosure_and_jacobian():
    import time
    capability = capabilities(rational_patch())
    box = ((.2, .3), (.4, .5))
    start = time.perf_counter()
    capability.enclosure(box)
    first = time.perf_counter() - start
    start = time.perf_counter()
    capability.jacobian_bounds(box)
    assert time.perf_counter() - start <= max(first / 5, 1e-3)


def test_box_memo_is_order_free_validated_and_thread_safe():
    """Review A3a round 2: reversed boxes share work; validation does not depend on cache history; no races."""
    import threading
    capability = capabilities(bilinear_patch())
    with pytest.raises(GeometryError):
        capability.enclosure(((Fraction(0), Fraction(1, 2)), (Fraction(0), Fraction(1, 2))))
    capability.enclosure(((0., .5), (0., .5)))
    with pytest.raises(GeometryError):
        capability.enclosure(((Fraction(0), Fraction(1, 2)), (Fraction(0), Fraction(1, 2))))
    assert capability.enclosure(((.5, 0.), (.5, 0.))) == capability.enclosure(((0., .5), (0., .5)))
    errors, barrier = [], threading.Barrier(4)

    def worker(offset):
        barrier.wait()
        try:
            for i in range(600):
                value = ((i * 4 + offset) % 2048) / 2048
                capability.enclosure(((value, value), (.25, .25)))
        except Exception as error:                                  # pragma: no cover - the failure being tested
            errors.append(error)

    threads = [threading.Thread(target=worker, args=(k,)) for k in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
