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
    points = curve.evaluate(np.linspace(0., 1., 10001))   # exact evaluation (owner option a): ~0.5 ms per point
    normal = np.cross(OBLIQUE.u_vector, OBLIQUE.v_vector)
    extent = 2 * RADIUS + HEIGHT
    assert np.abs((points - OBLIQUE.origin) @ normal).max() <= 1e-12 * extent
    assert np.abs(np.hypot(points[:, 0], points[:, 1]) - RADIUS).max() <= 1e-12 * extent


def test_exact_enclosures_contain_the_evaluated_points():
    curve = PlaneNurbsCurve.from_plane(OBLIQUE, cylinder(), 0)
    for t in np.linspace(0., 1., 301):
        (u_lo, u_hi), (v_lo, v_hi) = curve.chart_enclosure(t)
        u, v = curve._knot_points(np.array([t]))
        slack = Fraction(2.0 ** -44) * 4                     # the evaluation bracket (fraction of a unit-sized box)
        assert u_lo - slack <= Fraction(u[0]) <= u_hi + slack and v_lo - slack <= Fraction(v[0]) <= v_hi + slack
        assert u_hi - u_lo <= Fraction(1, 2 ** 50) and v_hi - v_lo <= Fraction(1, 2 ** 50)


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
        samples = curve.evaluate(np.linspace(a, b, 101))
        assert np.all(samples >= low) and np.all(samples <= high)
        assert curve.distance_lower_bound(a, b, target) <= np.linalg.norm(samples - target, axis=1).min()
    dense = curve.evaluate(np.linspace(0., 1., 4001))
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



# ------------------------------------------------------------------ review B1-P2 round 1


def flat(z_rows, v_knots=(0., 0., 1., 1.), v_values=(0., 1.)):
    from anygeometry.nurbs import NurbsSurface
    return NurbsSurface(2, len(v_values) - 1, (0., 0., 0., 1., 1., 1.), v_knots,
                        tuple(tuple((i / 2, v, z(v)) for v in v_values) for i, z in enumerate(z_rows)))


def test_evaluation_stays_inside_bounds_on_an_ill_conditioned_section():
    from anygeometry.nurbs import NurbsSurface
    a, e = .3, 2. ** -40
    heights = (a * a, a * a - a, (1 - a) * (1 - a))
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                           tuple(((i / 2, 0., z - e / 2), (i / 2, 1., z + e / 2)) for i, z in enumerate(heights)))
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
    for t in (.1, .37, .8):
        point = curve.evaluate(t)
        low, high = curve.bounds(t - 1e-8, t + 1e-8)
        assert np.all(point >= low) and np.all(point <= high)
        assert curve.distance_lower_bound(t - 1e-8, t + 1e-8, point) <= 1e-15


def test_split_at_a_point_where_rounding_leaves_the_gap():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., .375, 1., 1.),
                           tuple(tuple((i / 2, v, z + v - .5) for v in (0., .375, 1.)) for i, z in enumerate((0., .5, 2.))))
    plane = Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
    low, high = curve.subcurve(0., .5).end_box(1)
    point = curve.evaluate(.5)
    assert np.all(point >= low - 1e-12) and np.all(point <= high + 1e-12)
    model, edge = edge_model(curve)
    model.split_edge(edge, .5)


# ------------------------------------------------------------------ review B1-P2 round 2


def test_tiny_uniform_weights_do_not_break_the_sign_filter():
    from anygeometry.nurbs import NurbsSurface
    w = 2. ** -1070
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((float(i), float(j), i + 2 * j - .7) for j in range(2)) for i in range(2)),
                           ((w, w), (w, w)))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)), surface, 0)
    point = curve.evaluate(.37)
    low, high = curve.bounds(.37 - 1e-8, .37 + 1e-8)
    assert np.all(point >= low) and np.all(point <= high)
    assert abs(point[1] - .2205) <= 1e-12
    model, edge = edge_model(curve)
    model.split_edge(edge, .37)


def test_the_slicing_coordinate_is_the_long_one():
    from anygeometry.nurbs import NurbsSurface
    e = 2. ** -70
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((i - .3, float(j), e * j) for j in range(2)) for i in range(2)))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., 0.), (1., 0., -1.), (0., 1., 0.)), surface, 0)
    assert abs(curve.evaluate(.37)[1] - .37) <= 1e-12
    model, edge = edge_model(curve)
    assert abs(model.edge_length(edge) - 1.) <= 1e-12


# ------------------------------------------------------------------ review B1-P2 round 3 (exact evaluation)


def z_plane():
    return Plane((0., 0., 0.), (1., 0., 0.), (0., 1., 0.))


def test_wide_knot_ranges_evaluate_exactly():
    from anygeometry.nurbs import NurbsSurface
    e = 2. ** -40
    surface = NurbsSurface(2, 1, (-1., -1., -1., 2., 2., 2.), (0., 0., 1., 1.),
                           tuple(((i / 2, 0., z - e / 2), (i / 2, 1., z + e / 2)) for i, z in enumerate((1., -1., 0.))))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    point = curve.evaluate(.37)
    low, high = curve.bounds(.37 - 1e-8, .37 + 1e-8)
    assert np.all(point >= low) and np.all(point <= high)
    model, edge = edge_model(curve)
    model.split_edge(edge, .37)


def test_a_nearly_vertical_arc_does_not_collapse_onto_its_ends():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((i / 2, float(j), z) for j in range(2)) for i, z in enumerate((1., -2., 4.))))
    plane = Plane((0., 0., .5 * 2. ** -110), (1., 0., 0.), (0., 1., -2. ** -110))
    curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
    point = curve.evaluate(.37)
    assert .3 < point[1] < .5                                  # not an end (y = 0)
    (u_lo, u_hi), (v_lo, v_hi) = curve.chart_enclosure(.37)
    assert v_lo <= Fraction(point[1]) <= v_hi or abs(float(v_lo) - point[1]) <= 1e-15


def test_large_weight_ratios_keep_split_vertices_inside_end_boxes():
    from anygeometry.nurbs import NurbsSurface
    weights, offset, e = (1., 1e6), 1.3e-8, 2. ** -20
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((i / 2, float(j), (z + e * (j - offset)) / weights[j]) for j in range(2))
                                 for i, z in enumerate((1., -1., 1.))),
                           tuple(weights for _ in range(3)))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    point = curve.evaluate(.5)
    low, high = curve.subcurve(0., .5).end_box(1)
    assert np.all(point >= low) and np.all(point <= high)
    model, edge = edge_model(curve)
    model.split_edge(edge, .5)


def test_tiny_uniform_weights_give_the_unit_weight_length():
    from anygeometry.nurbs import NurbsSurface
    controls = tuple(tuple((float(i), float(j), i + .3 * j - .2) for j in range(2)) for i in range(2))
    lengths = []
    for w in (1., 2. ** -1070):
        surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.), controls, ((w, w), (w, w)))
        model, edge = edge_model(PlaneNurbsCurve.from_plane(z_plane(), surface, 0))
        lengths.append(model.edge_length(edge))
    assert abs(lengths[0] - lengths[1]) <= 1e-12


# ------------------------------------------------------------------ review B1-P2 exact round 1


def test_distance_bounds_hold_for_evaluated_points_far_from_the_origin():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((1e10, float(i), j - .3) for j in range(2)) for i in range(2)))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    assert curve.distance_lower_bound(0., 1., curve.evaluate(.05)) == 0.


def test_bounds_hold_with_many_arcs():
    from anygeometry.nurbs import NurbsSurface
    knots_u = np.r_[np.linspace(0., .1, 513), 1.]
    surface = NurbsSurface(1, 1, (0., *knots_u, 1.), (0., 0., 1., 1.),
                           tuple(((float(u), 0., -.5), (float(u), 1., .5)) for u in knots_u))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    for t in (.9984307992202729, .5, .0123, .99999):
        point = curve.evaluate(t)
        low, high = curve.bounds(t, t)
        assert np.all(point >= low) and np.all(point <= high)


def test_near_vertical_lengths_refuse_rather_than_vanish():
    from anygeometry.exact_curves import analytic_curve_length
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((i / 2, float(j), z) for j in range(2)) for i, z in enumerate((1., -2., 4.))))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., .5 * 2. ** -110), (1., 0., 0.), (0., 1., -2. ** -110)),
                                       surface, 0)
    try:
        length = analytic_curve_length(curve, 1e-9)
    except GeometryError:
        return
    assert length > .9                                          # never the silent ~1e-17 of float partials


def test_shifted_knot_domains_evaluate_exactly():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (1e16, 1e16, 1e16 + 256, 1e16 + 256), (0., 0., 1., 1.),
                           tuple(tuple((float(i), float(j), i + 2 * j - .7) for j in range(2)) for i in range(2)))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    point = curve.evaluate(.37)
    assert abs(point[2]) <= 1e-15
    model, edge = edge_model(curve)
    assert model.closest_edge_point(edge, point)[2] <= 1e-15


# ------------------------------------------------------------------ review B1-P2 exact round 2


def test_steep_surfaces_keep_points_in_the_distance_bound():
    from anygeometry.exact_curves import project_analytic_curve
    from anygeometry.nurbs import NurbsSurface
    controls = tuple(tuple((float(i), h, h) for h in (1e20 * (3 * j - 1 + .1 * i) for j in range(2))) for i in range(2))
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.), controls)
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    target = curve.evaluate(.83)
    assert abs(target[2]) <= 2. ** -50 * 3e20                 # the defined points lie in the plane slab
    assert curve.distance_lower_bound(0., 1., target) == 0.
    assert project_analytic_curve(curve, target, 1e-9)[1] <= 1e-9 * 3e20


def test_arc_selection_covers_non_monotone_parameter_maps():
    from anygeometry.nurbs import NurbsSurface
    knots_u = (0., *np.linspace(0., 1., 514), 1.)
    controls = tuple(tuple((float(i - 201), float(j), j - .3) for j in range(2)) for i in range(514))
    surface = NurbsSurface(1, 1, knots_u, (0., 0., 1., 1.), controls)
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0).subcurve(0.3918128654970759, 0.3918128654970761)
    lower, upper = .3472, .34740000000000004
    for t in np.linspace(lower, upper, 41):
        point = curve.evaluate(t)
        low, high = curve.bounds(lower, upper)
        assert np.all(point >= low) and np.all(point <= high)
        assert curve.distance_lower_bound(lower, upper, point) == 0.


def test_shifted_knot_domains_differentiate_exactly():
    from anygeometry.exact_curves import analytic_curve_length
    from anygeometry.nurbs import NurbsSurface
    controls = tuple(tuple((i / 2, float(i == 2), j - .3) for j in range(2)) for i in range(3))
    shifted = NurbsSurface(2, 1, (1e16,) * 3 + (1e16 + 256,) * 3, (0., 0., 1., 1.), controls)
    plain = NurbsSurface(2, 1, (0.,) * 3 + (1.,) * 3, (0., 0., 1., 1.), controls)
    curves = [PlaneNurbsCurve.from_plane(z_plane(), surface, 0) for surface in (shifted, plain)]
    derivatives = [curve.derivative(.37) for curve in curves]
    direction = derivatives[1] / derivatives[1][0]
    assert np.allclose(direction, (1., .74, 0.), atol=1e-12)
    assert np.allclose(derivatives[0], derivatives[1], rtol=1e-12, atol=1e-15)
    lengths = [analytic_curve_length(curve, 1e-9) for curve in curves]
    assert abs(lengths[0] - lengths[1]) <= 1e-8


# ------------------------------------------------------------------ review B1-P2 exact round 3


def test_derivatives_at_a_trimmed_knot_end_stay_on_the_arc_side():
    from anygeometry.nurbs import NurbsSurface
    x, z = (0., .5, 1.), (-.2, -.4, -.3)
    controls = tuple(tuple((x[i], float(j), z[i] + j) for j in range(2)) for i in range(3))
    surface = NurbsSurface(1, 1, (0., 0., .5, 1., 1.), (0., 0., 1., 1.), controls, u_range=(0., .5))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    assert np.allclose(curve.evaluate(1.), (.5, .4, 0.), atol=1e-15)
    end, inside = curve.derivative(1.), curve.derivative(1. - 1e-9)
    assert abs(end[2]) <= 1e-15                                 # tangent stays in the section plane
    assert np.allclose(end, inside, rtol=1e-6)


def test_huge_plane_vectors_evaluate():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((float(i), float(j), j - .3) for j in range(2)) for i in range(2)))
    plane = Plane((0., 0., 0.), (1e100, 0., 0.), (0., 1e100, 0.))
    curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
    assert abs(curve.evaluate(.5)[2]) <= 1e-15


def test_distance_bounds_to_targets_near_the_float_limit():
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           tuple(tuple((float(i), float(j), j - .3) for j in range(2)) for i in range(2)))
    curve = PlaneNurbsCurve.from_plane(z_plane(), surface, 0)
    bound = curve.distance_lower_bound(0., 1., (1.7e308, 1.7e308, 1.7e308))
    assert 1e300 <= bound <= 1.7e308 * math.sqrt(3.)
