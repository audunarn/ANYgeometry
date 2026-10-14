"""Material domains on NURBS charts (generality slice A1e-lite).

References: analytic cylinder areas (r |sweep| h), exact shoelace areas, the
exact area of an affine (parallelogram) patch, circles in the chart, and
finite differences for curve-on-surface derivatives.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, Plane
from anygeometry.arrangement_geometry import LinePath
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface, cylinder_to_nurbs, nurbs_circular_arc
from anygeometry import nurbs_domain


def wavy_patch():
    rng = np.random.default_rng(21)
    controls = np.zeros((5, 4, 3))
    controls[..., 0] = np.linspace(0., 4., 5)[:, None]
    controls[..., 1] = np.linspace(0., 3., 4)[None, :]
    controls[..., 2] = rng.uniform(-.5, .5, (5, 4))
    weights = rng.uniform(.6, 2., (5, 4))
    return NurbsSurface(3, 2, (0., 0., 0., 0., .5, 1., 1., 1., 1.), (0., 0., 0., .4, 1., 1., 1.),
                        tuple(tuple(map(tuple, row)) for row in controls), tuple(map(tuple, weights)))


def segment(surface, a, b):
    return ArrangementPath(CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b))))


def polygon(surface, points):
    return tuple(segment(surface, a, b) for a, b in zip(points, points[1:] + points[:1]))


def chart_circle(surface, center, radius):
    arc = nurbs_circular_arc(center, (1., 0.), (0., 1.), radius, 0., 2 * math.pi)
    return (ArrangementPath(CurveOnSurface(surface, arc)),)


def shoelace(points):
    return .5 * sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(points, points[1:] + points[:1]))


# ------------------------------------------------------------------ L1


def test_curve_on_surface_derivatives_match_finite_differences():
    surface = wavy_patch()
    curve = CurveOnSurface(surface, NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .2), (.8, .3), (.5, .9)),
                                               (1., .6, 1.3)))
    t, h = np.linspace(.05, .95, 41), 1e-6
    first = curve.derivative(t)
    assert np.abs(first - (curve.evaluate(t + h) - curve.evaluate(t - h)) / (2 * h)).max() <= 1e-7 * np.abs(first).max()
    second = curve.second_derivative(t)
    assert np.abs(second - (curve.derivative(t + h) - curve.derivative(t - h)) / (2 * h)).max() \
        <= 1e-6 * np.abs(second).max()
    scalar = curve.derivative(.3)
    assert scalar.shape == (3,) and np.array_equal(scalar, curve.derivative(np.array([.3]))[0])


def test_surface_second_derivatives_match_finite_differences():
    surface = wavy_patch()
    uv = np.random.default_rng(22).uniform(.05, .95, (60, 2))
    h = 1e-5
    s_uu, s_uv, s_vv = surface.second_derivatives_many(uv)

    def first(offset):
        return surface.derivatives_many(uv + offset)

    fd_uu = (first([h, 0.])[0] - first([-h, 0.])[0]) / (2 * h)
    fd_uv = (first([0., h])[0] - first([0., -h])[0]) / (2 * h)
    fd_vv = (first([0., h])[1] - first([0., -h])[1]) / (2 * h)
    for exact, approximate in ((s_uu, fd_uu), (s_uv, fd_uv), (s_vv, fd_vv)):
        assert np.abs(exact - approximate).max() <= 1e-7 * np.abs(exact).max()


def test_batch_chart_evaluation_is_bit_identical_to_per_point():
    surface = wavy_patch()
    uv = np.random.default_rng(23).uniform(0., 1., (200, 2))
    batch = surface.evaluate_many(uv)
    single = np.array([surface.evaluate(u, v) for u, v in uv])
    assert np.array_equal(batch.view(np.uint64), single.view(np.uint64))
    s_u, s_v = surface.derivatives_many(uv)
    for k in range(0, 200, 17):
        one_u, one_v = surface.derivatives_many(uv[k:k + 1])
        assert np.array_equal(one_u[0], s_u[k]) and np.array_equal(one_v[0], s_v[k])


# ------------------------------------------------------------------ L2


def test_chart_areas_of_pcurve_polygons_and_circles():
    surface = wavy_patch()
    points = [(.1, .1), (.9, .2), (.7, .8), (.3, .6)]
    domain = MaterialDomain(0, surface, (polygon(surface, points),))
    assert abs(domain.area_loop(domain.boundaries[0], 1e-14) - shoelace(points)) <= 1e-14
    assert abs(domain.area_loop(tuple(reversed(polygon(surface, points[::-1]))), 1e-14) + shoelace(points)) <= 1e-14
    circle = chart_circle(surface, (.5, .5), .3)
    assert abs(domain.area_loop(circle, 1e-14) - math.pi * .09) <= 1e-13


def test_containment_in_the_chart_with_a_hole():
    surface = wavy_patch()
    outer = polygon(surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    hole = chart_circle(surface, (.5, .5), .2)
    domain = MaterialDomain(0, surface, (outer, hole))
    cases = {(.1, .1): True, (.5, .5): False, (.5, .75): True, (.5, .65): False, (1.2, .5): False,
             (.5, 0.): True, (.7, .5): True}
    for point, expected in cases.items():
        uv = np.asarray(point)
        inside = (domain.contains_loop(outer, uv, 1e-9)
                  and not domain.contains_loop(hole, uv, 1e-9, boundary=False))
        assert inside == expected, point


def test_chart_tangent_curvature_and_level_roots():
    surface = wavy_patch()
    circle = chart_circle(surface, (.5, .5), .25)[0].curve
    domain = MaterialDomain(0, surface, ((ArrangementPath(circle),),))
    for t in (.1, .37, .8):
        assert abs(domain.curvature(circle, t, True) - 4.) <= 1e-9
        tangent = domain.tangent(circle, t)
        radial = domain.uv(circle, t) - (.5, .5)
        assert abs(float(tangent @ radial)) <= 1e-12 * np.linalg.norm(tangent)
    roots = domain.horizontal_roots(circle, .6, 1e-9)
    assert len(roots) == 2
    for t in roots:
        assert abs(domain.uv(circle, t)[1] - .6) <= 1e-12
    assert domain.horizontal_roots(circle, .9, 1e-9) == ()
    flat = segment(surface, (.2, .4), (.8, .4)).curve
    assert domain.horizontal_roots(flat, .4, 1e-9) is None


def test_nurbs_domains_refuse_world_curves_and_other_supports():
    surface = wavy_patch()
    domain = MaterialDomain(0, surface, (polygon(surface, [(0., 0.), (1., 0.), (1., 1.)]),))
    with pytest.raises(GeometryError):
        domain.uv(LinePath((0., 0., 0.), (1., 0., 0.)), .5)
    other = CurveOnSurface(cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 1.)),
                           NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 1.))))
    with pytest.raises(GeometryError):
        domain.uv(other, .5)
    moved = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 1.))),
                           transform=((1., 0., 0., 1.), (0., 1., 0., 0.), (0., 0., 1., 0.), (0., 0., 0., 1.)))
    with pytest.raises(GeometryError):
        domain.tangent(moved, .5)


def test_planar_domains_are_unchanged():
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    corners = [(0., 0., 0.), (2., 0., 0.), (2., 1., 0.), (0., 1., 0.)]
    loop = tuple(ArrangementPath(LinePath(a, b)) for a, b in zip(corners, corners[1:] + corners[:1]))
    domain = MaterialDomain(0, plane, (loop,))
    assert domain.area_loop(loop, 1e-12) == 2. and domain.world_area_loop(loop, 1e-12) == 2.


# ------------------------------------------------------------------ L3


@pytest.mark.parametrize("sweep", (2 * math.pi, 1.3, -2., .01))
def test_world_area_of_rational_cylinder_patches(sweep):
    radius, height = 2.5, 4.
    surface = cylinder_to_nurbs(Cylinder((1., 2., 3.), (0., 0., 1.), (1., 0., 0.), radius, height, .3, sweep))
    square = polygon(surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    domain = MaterialDomain(0, surface, (square,))
    value, estimate = nurbs_domain.world_area_estimate(domain, square, 1e-12)
    expected = radius * abs(sweep) * height
    assert abs(value - expected) <= 1e-10 * expected and estimate <= 1e-9 * expected
    assert abs(domain.world_area_loop(square, 1e-12) - expected) <= 1e-10 * expected
    assert abs(domain.original_world_area(1e-12) - expected) <= 1e-10 * expected


def test_world_area_of_a_cylinder_sub_rectangle_uses_the_arc_angles():
    radius, height = 1.5, 2.
    cylinder = Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), radius, height, 0., 4.5)
    surface = cylinder_to_nurbs(cylinder)
    u0, u1, v0, v1 = .13, .71, .2, .9
    loop = polygon(surface, [(u0, v0), (u1, v0), (u1, v1), (u0, v1)])
    domain = MaterialDomain(0, surface, (loop,))
    points = surface.evaluate_many(np.array([[u0, 0.], [u1, 0.]]))
    angles = np.unwrap(np.arctan2(points[:, 1], points[:, 0]))
    expected = radius * abs(angles[1] - angles[0]) * height * (v1 - v0)
    assert abs(domain.world_area_loop(loop, 1e-12) - expected) <= 1e-10 * expected


def test_world_area_of_an_affine_patch_is_the_scaled_chart_area():
    a, b, origin = np.array((2., .5, 1.)), np.array((-.3, 1.5, .4)), np.array((1., 1., 1.))
    controls = tuple(tuple(tuple(map(float, origin + i * a + j * b)) for j in range(2)) for i in range(2))
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.), controls)
    points = [(.1, .1), (.9, .2), (.7, .8), (.3, .6)]
    loop = polygon(surface, points)
    domain = MaterialDomain(0, surface, (loop,))
    expected = float(np.linalg.norm(np.cross(a, b))) * shoelace(points)
    assert abs(domain.world_area_loop(loop, 1e-13) - expected) <= 1e-12 * expected
    circle = chart_circle(surface, (.5, .5), .3)
    expected_circle = float(np.linalg.norm(np.cross(a, b))) * math.pi * .09
    assert abs(domain.world_area_loop(circle, 1e-13) - expected_circle) <= 1e-12 * expected_circle


# ------------------------------------------------------------------ review A1e round 1


def test_level_roots_terminate_on_tangencies_and_runs_along_the_level():
    tangent = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .75), (.5, .25), (.9, .75)))
    roots = nurbs_domain.level_roots(tangent, 1, .5, 1e-10)
    assert len(roots) == 1 and abs(roots[0] - .5) <= 1e-6
    run = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .5), (.5, .5), (.9, .9)))
    roots = nurbs_domain.level_roots(run, 1, .5, 1e-10)
    assert roots[0] == 0. and abs(roots[-1] - .5) <= 1e-9 and len(roots) == 2
    crossing = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .75), (.5, 0.), (.9, .75)))
    roots = nurbs_domain.level_roots(crossing, 1, .5, 1e-10)
    assert len(roots) == 2 and all(abs(crossing.evaluate(t)[1] - .5) <= 1e-12 for t in roots)


def test_containment_with_a_ray_tangent_to_a_hole():
    surface = wavy_patch()
    outer = polygon(surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    hole = chart_circle(surface, (.5, .5), .2)
    domain = MaterialDomain(0, surface, (outer, hole))
    # rays at v = .7 and .3 touch the hole's circle; (.5, .7) is on it (material boundary)
    for point, expected in (((.1, .7), True), ((.1, .3), True), ((.5, .7), True), ((.5, .69), False),
                            ((.9, .7), True), ((.5, .31), False)):
        uv = np.asarray(point)
        inside = (domain.contains_loop(outer, uv, 1e-9)
                  and not domain.contains_loop(hole, uv, 1e-9, boundary=False))
        assert inside == expected, point


def test_world_area_splits_at_surface_knot_crossings():
    heights = (0., .5, 1.)
    narrow_v = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., .999999, 1., 1.),
                            tuple(tuple((float(i), heights[j], 0.) for j in range(3)) for i in range(2)))
    square = polygon(narrow_v, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    assert abs(MaterialDomain(0, narrow_v, (square,)).world_area_loop(square, 1e-12) - 1.) <= 1e-12
    narrow_u = NurbsSurface(1, 1, (0., 0., .999999, 1., 1.), (0., 0., 1., 1.),
                            tuple(tuple((heights[i], float(j), 0.) for j in range(2)) for i in range(3)))
    triangle = polygon(narrow_u, [(0., 0.), (1., 1.), (0., 1.)])
    # x(u) is piecewise linear with a knot at .999999: the image triangle's exact area is .2500005
    value = MaterialDomain(0, narrow_u, (triangle,)).world_area_loop(triangle, 1e-12)
    assert abs(value - .2500005) <= 1e-9


def test_trimmed_curve_end_derivatives_are_inward():
    curve = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .1), (.4, .2), (.9, .9)), domain=(0., .5))
    np.testing.assert_allclose(curve.derivative(1.), (.3, .1), atol=1e-15)
    reversed_curve = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .1), (.4, .2), (.9, .9)), domain=(.5, 0.))
    np.testing.assert_allclose(reversed_curve.derivative(0.), (-.3, -.1), atol=1e-15)
    affine = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    on_surface = CurveOnSurface(affine, curve)
    np.testing.assert_allclose(on_surface.derivative(1.), (.3, .1, 0.), atol=1e-15)
    np.testing.assert_allclose(on_surface.second_derivative(1.), (0., 0., 0.), atol=1e-15)


# ------------------------------------------------------------------ review A1e round 2


def test_close_genuine_crossings_are_kept_and_containment_holds():
    dip = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .75 - 1e-14), (.5, .25 - 1e-14), (.9, .75 - 1e-14)))
    roots = nurbs_domain.level_roots(dip, 1, .5, 1e-17)
    assert len(roots) == 2 and roots[1] - roots[0] < 1e-6
    lifted = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .75 + 1e-14), (.5, .25 + 1e-14), (.9, .75 + 1e-14)))
    assert nurbs_domain.level_roots(lifted, 1, .5, 1e-17) == ()
    stretched = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                             (((0., 0., 0.), (0., 1e8, 0.)), ((1., 0., 0.), (1., 1e8, 0.))))
    chord = NurbsCurve(1, (0., 0., 1., 1.), ((.9, .75 - 1e-14), (.1, .75 - 1e-14)))
    loop = (ArrangementPath(CurveOnSurface(stretched, dip)), ArrangementPath(CurveOnSurface(stretched, chord)))
    assert MaterialDomain(0, stretched, (loop,)).contains_loop(loop, np.array([.5, .5]), 1e-9)


def test_points_on_a_run_along_the_ray_are_boundary():
    affine = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                          (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    run = NurbsCurve(1, (0., 0., .5, 1., 1.), ((.1, .5), (.5, .5), (.9, .9)))
    loop = (ArrangementPath(CurveOnSurface(affine, run)), segment(affine, (.9, .9), (.1, .9)),
            segment(affine, (.1, .9), (.1, .5)))
    domain = MaterialDomain(0, affine, (loop,))
    assert domain.contains_loop(loop, np.array([.3, .5]), 1e-9, boundary=False) is False
    assert domain.contains_loop(loop, np.array([.3, .5]), 1e-9) is True
    assert domain.contains_loop(loop, np.array([.3, .6]), 1e-9) and not domain.contains_loop(loop, np.array([.3, .45]), 1e-9)


def test_inward_end_derivatives_at_extreme_weights_and_tiny_coordinates():
    tiny_weights = NurbsCurve(2, (0., 0., 0., .5, 1., 1., 1.), ((0., 0.), (1e-100, 0.), (2e-100, 0.), (3e-100, 0.)),
                              (1e-300,) * 4, (0., .5))
    values = tiny_weights.derivatives(1., 2)
    np.testing.assert_allclose(values[1], (1e-100, 0.), rtol=1e-12)
    np.testing.assert_allclose(values[2], (-1e-100, 0.), rtol=1e-12)
    a = 2. ** -499
    small = NurbsCurve(2, (0., 0., 0., .5, 1., 1., 1.), ((0., a), (1., a), (2., -a * (1 - 2. ** -52)), (3., a)),
                       domain=(0., .5))
    assert np.all(np.isfinite(small.derivatives(1., 2)))


# ------------------------------------------------------------------ review A1e round 3 (exact level sets)


def unit_support():
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))


def test_four_close_crossings_keep_containment():
    surface = unit_support()
    four = NurbsCurve(1, (0., 0., .4999999, .50000005, .5000003, .50000055, .5000007, 1., 1.),
                      ((.1, .3), (.2, .3), (.3, .6), (.4, .1), (.5, .6), (.6, .3), (.9, .3)))
    loop = (ArrangementPath(CurveOnSurface(surface, four)),) + polygon(surface, [(.9, .3), (.9, .05), (.1, .05),
                                                                                (.1, .3)])[:3]
    domain = MaterialDomain(0, surface, (loop,))
    assert domain.contains_loop(loop, np.array([.3, .5]), 1e-9) and domain.contains_loop(loop, np.array([.5, .5]), 1e-9)


def test_points_on_a_flat_start_run_are_boundary():
    surface = unit_support()
    run = NurbsCurve(1, (0., 0., .1, 1., 1.), ((.1, .9), (.5, .9), (.9, .95)))
    loop = (ArrangementPath(CurveOnSurface(surface, run)),) + polygon(surface, [(.9, .95), (.9, .98), (.1, .98),
                                                                               (.1, .9)])[:3]
    domain = MaterialDomain(0, surface, (loop,))
    assert domain.contains_loop(loop, np.array([.2, .9]), 1e-9) is True
    assert domain.contains_loop(loop, np.array([.2, .9]), 1e-9, boundary=False) is False
    assert domain.contains_loop(loop, np.array([.2, .93]), 1e-9) and not domain.contains_loop(loop, np.array([.2, .85]), 1e-9)


def test_a_high_order_touch_is_not_a_run():
    stretched = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                             (((0., 0., 0.), (0., 1e8, 0.)), ((1., 0., 0.), (1., 1e8, 0.))))
    quartic = NurbsCurve(4, (0.,) * 5 + (1.,) * 5,
                         ((.1, .5625), (.3, .4375), (.5, .5625), (.7, .4375), (.9, .5625)))
    loop = (ArrangementPath(CurveOnSurface(stretched, quartic)),
            ArrangementPath(CurveOnSurface(stretched, NurbsCurve(1, (0., 0., 1., 1.), ((.9, .5625), (.1, .5625))))))
    domain = MaterialDomain(0, stretched, (loop,))
    assert domain.contains_loop(loop, np.array([.50008, .5]), 1e-9) is False


@pytest.mark.parametrize("domain_range, end", (((0., .5), 1.), ((.5, 0.), 0.)))
def test_constant_curve_end_derivatives_are_exactly_zero(domain_range, end):
    curve = NurbsCurve(2, (0., 0., 0., .5, 1., 1., 1.), ((1e16, 1.),) * 4, (1e-300, 3e-301, 7e-301, 9e-301),
                       domain_range)
    values = curve.derivatives(end, 2)
    assert np.all(values[1:] == 0.)
