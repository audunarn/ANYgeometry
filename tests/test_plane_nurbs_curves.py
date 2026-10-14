"""Plane x NURBS section curves as model edges (generality slice B1-P2, architecture 2.1, 4.3).

References: the analytic ellipse of a plane section of a cylinder (points on
both exact surfaces; arc length by high-order quadrature of the ellipse
parameterisation), exact rational enclosures, dense sampling for bounds and
projection.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, Plane
from anygeometry.nurbs import cylinder_to_nurbs
from anygeometry.support_curves import PlaneNurbsCurve

RADIUS, HEIGHT = 2., 3.
OBLIQUE = Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2))          # z = 1.5 + .3 x + .2 y


def cylinder(sweep=2 * math.pi):
    return cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), RADIUS, HEIGHT, 0., sweep))


def edge_model(curve):
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    return model, model.add_curve(start, end, curve)


def ellipse_length(theta0, theta1):
    nodes, weights = np.polynomial.legendre.leggauss(200)
    theta = .5 * (theta1 - theta0) * (nodes + 1) + theta0
    speed = np.sqrt(RADIUS ** 2 + (RADIUS * (-.3 * np.sin(theta) + .2 * np.cos(theta))) ** 2)
    return .5 * (theta1 - theta0) * float(weights @ speed)


def test_points_lie_on_both_exact_surfaces():
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(), 0)
    points = curve.evaluate(np.linspace(0., 1., 100001))
    normal = np.cross(OBLIQUE.u_vector, OBLIQUE.v_vector)
    extent = 2 * RADIUS + HEIGHT
    assert np.abs((points - OBLIQUE.origin) @ normal).max() <= 1e-12 * extent
    assert np.abs(np.hypot(points[:, 0], points[:, 1]) - RADIUS).max() <= 1e-12 * extent


def test_exact_enclosures_contain_the_evaluated_points():
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(), 0)
    for t in np.linspace(0., 1., 301):
        slicing, s, x_lo, x_hi = curve.chart_enclosure(t)
        u, v = curve._knot_points(np.array([t]))
        knot = (u[0], v[0])
        assert float(s) == knot[slicing]
        x = Fraction(knot[1 - slicing])
        width = x_hi - x_lo
        assert x_lo - 4 * Fraction(2.2e-16) <= x <= x_hi + 4 * Fraction(2.2e-16) and width <= Fraction(1, 2 ** 50)


def test_length_matches_the_analytic_ellipse():
    from anygeometry.exact_curves import analytic_curve_length
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(), 0)                  # closed: no single edge
    expected = ellipse_length(0., 2 * math.pi)
    assert abs(analytic_curve_length(curve, 1e-12) - expected) <= 1e-10 * expected
    half = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(math.pi), 0)
    model, edge = edge_model(half)
    expected = ellipse_length(0., math.pi)
    assert abs(model.edge_length(edge) - expected) <= 1e-10 * expected


def test_bounds_distance_bounds_and_projection():
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(math.pi), 0)
    model, edge = edge_model(curve)
    rng = np.random.default_rng(51)
    target = np.array((3., 2., 1.))
    for _ in range(20):
        a, b = np.sort(rng.uniform(0., 1., 2))
        low, high = curve.bounds(a, b)
        samples = curve.evaluate(np.linspace(a, b, 401))
        assert np.all(samples >= low) and np.all(samples <= high)
        assert curve.distance_lower_bound(a, b, target) <= np.linalg.norm(samples - target, axis=1).min()
    dense = curve.evaluate(np.linspace(0., 1., 40001))
    for point in ((3., 2., 1.), (0., 0., 1.5), (-1., 3., 2.5)):
        _closest, _parameter, distance = model.closest_edge_point(edge, point)
        reference = np.linalg.norm(dense - np.asarray(point), axis=1).min()
        assert distance <= reference + 1e-12 and distance >= reference - 1e-6


def test_split_end_test_and_refusals():
    from anygeometry import IntersectionKind, query_intersection
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(math.pi), 0)
    model, edge = edge_model(curve)
    total = model.edge_length(edge)
    vertex, (first, second) = model.split_edge(edge, .4)
    assert abs(model.edge_length(first) + model.edge_length(second) - total) <= 1e-12 * total
    np.testing.assert_allclose(model.vertex_position(vertex), curve.evaluate(.4), atol=1e-15 * 8)
    other = GeometryModel()
    start = other.add_point(*curve.evaluate(0.))
    far = other.add_point(*(curve.evaluate(1.) + (1e-3, 0., 0.)))
    with pytest.raises(GeometryError, match="ends do not match"):
        other.add_curve(start, far, curve)
    with pytest.raises(GeometryError, match="NURBS profile"):
        model.revolve([first], (0., 0., 0.), (0., 0., 1.), 1.)
    corners = [model.add_point(x, y, -1.) for x, y in ((-5., -5.), (5., -5.), (5., 5.), (-5., 5.))]
    face = model.add_face([model.add_line(a, b) for a, b in zip(corners, corners[1:] + corners[:1])])
    result = query_intersection(model, model.handle("edge", first), model.handle("face", face))
    assert not result.classified or result.kind is IntersectionKind.DISJOINT


def test_a_frame_section_of_the_bow():
    import sys
    sys.path.insert(0, "tests")
    from test_nurbs_bow_audit import bow_surface
    bow = bow_surface()
    frame = Plane((4., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    curve = PlaneNurbsCurve.from_plane(frame, bow, 0)
    points = curve.evaluate(np.linspace(0., 1., 2001))
    assert np.abs(points[:, 0] - 4.).max() <= 1e-12
    model, edge = edge_model(curve)
    assert model.edge_length(edge) > 0
