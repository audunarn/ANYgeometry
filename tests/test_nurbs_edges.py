"""NURBS curves as model edges (generality slice A2a, architecture 3.1, 3.4, 3.5).

References: analytic circles (radius, length, tangents), exact rational curve
ends (Fraction de Boor), the schema-7 record rules, and transform commutation.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.nurbs import NurbsCurve, nurbs_circular_arc


def quarter_arc(radius=2.):
    return nurbs_circular_arc((1., 2., 3.), (1., 0., 0.), (0., 1., 0.), radius, 0., math.pi / 2)


def edge_model(curve):
    model = GeometryModel()
    start, end = (model.add_point(*map(float, curve.exact_point(t))) for t in (0, 1))
    return model, model.add_curve(start, end, curve)


@pytest.mark.parametrize("sweep", (math.pi / 3, math.pi / 2, 1.3 * math.pi, -2 * math.pi + 1e-9, 2 * math.pi))
def test_circular_arcs_are_circles(sweep):
    radius, center = 2.5, np.array((1., -2., .5))
    x_axis, y_axis = np.array((0., 1., 0.)), np.array((0., 0., 1.))
    arc = nurbs_circular_arc(center, x_axis, y_axis, radius, .3, sweep)
    t = np.linspace(0., 1., 2001)
    points = arc.evaluate(t)
    assert np.max(np.abs(np.linalg.norm(points - center, axis=1) - radius)) <= 4 * np.finfo(float).eps * radius * 4
    length = GeometryModel().tolerance  # noqa: F841 (policy object not needed)
    derivatives = arc.derivative(t)
    from anygeometry.exact_curves import analytic_curve_length
    assert abs(analytic_curve_length(arc, 1e-13) - radius * abs(sweep)) <= 1e-12 * radius * abs(sweep)
    start_angle = .3
    assert np.allclose(points[0], center + radius * (math.cos(start_angle) * x_axis + math.sin(start_angle) * y_axis),
                       atol=1e-15 * radius * 8)
    assert np.all(np.abs(np.einsum("ij,ij->i", derivatives, points - center)) <= 1e-12 * radius ** 2 * 10)


def test_exact_points_match_the_float_evaluation():
    arc = quarter_arc()
    for t in (0, Fraction(1, 3), Fraction(1, 2), 1):
        exact = arc.exact_point(t)
        assert np.allclose([float(v) for v in exact], arc.evaluate(float(t)), rtol=0, atol=1e-15 * 8)
    assert arc.exact_point(0) == tuple(Fraction(v) for v in arc.controls[0])
    assert arc.exact_point(1) == tuple(Fraction(v) for v in arc.controls[-1])


def test_a_nurbs_edge_works_through_the_model_api():
    arc = quarter_arc()
    model, edge = edge_model(arc)
    assert abs(model.edge_length(edge) - math.pi) <= 1e-12 * math.pi
    np.testing.assert_allclose(model.sample_edge(edge, [0., .5, 1.]), arc.evaluate([0., .5, 1.]), atol=0)
    for factor in (3., .25):                                          # outside and inside the circle, on the ray
        target = np.array((1. + 2. * math.cos(.4) * factor, 2. + 2. * math.sin(.4) * factor, 3.))
        point, _parameter, distance = model.closest_edge_point(edge, target)
        np.testing.assert_allclose(point, (1. + 2. * math.cos(.4), 2. + 2. * math.sin(.4), 3.), atol=1e-8)
        assert abs(distance - 2. * abs(factor - 1.)) <= 1e-8
    lower, upper = model.entity_bounds_many([("edge", edge)])[0][:3], model.entity_bounds_many([("edge", edge)])[0][3:]
    samples = arc.evaluate(np.linspace(0., 1., 501))
    assert np.all(samples >= np.asarray(lower)) and np.all(samples <= np.asarray(upper))


def test_splitting_a_nurbs_edge_keeps_the_curve():
    arc = quarter_arc()
    model, edge = edge_model(arc)
    vertex, (first, second) = model.split_edge(edge, .5)
    for edge_id in (first, second):
        assert isinstance(model.edges[edge_id].curve, NurbsCurve)
    np.testing.assert_allclose(model.vertex_position(vertex), arc.evaluate(.5), atol=0)
    assert abs(model.edge_length(first) + model.edge_length(second) - math.pi) <= 1e-12 * math.pi


def test_transform_commutes_with_evaluation():
    arc = quarter_arc()
    matrix = np.eye(4)
    c, s = math.cos(.7), math.sin(.7)
    matrix[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    matrix[:3, 3] = (3., -1., 2.)
    t = np.linspace(0., 1., 101)
    moved = arc.transformed(matrix).evaluate(t)
    expected = arc.evaluate(t) @ matrix[:3, :3].T + matrix[:3, 3]
    assert np.max(np.abs(moved - expected)) <= 1e-13 * 8


def test_nurbs_edges_round_trip_at_schema_7_and_ordinary_documents_are_unchanged():
    ordinary = GeometryModel()
    a, b = ordinary.add_point(0., 0., 0.), ordinary.add_point(1., 0., 0.)
    ordinary.add_line(a, b)
    assert to_dict(ordinary)["version"] < 7
    model, edge = edge_model(quarter_arc())
    document = to_dict(model)
    assert document["version"] == 7
    restored = from_dict(document)
    assert restored.edges[edge].curve == model.edges[edge].curve
    assert to_dict(restored) == document
    from anygeometry.serialization import _checksum
    old = dict(document, version=6)
    old["checksum"] = _checksum(old)                                 # a valid document that only lies about its schema
    with pytest.raises(GeometryError, match="7"):
        from_dict(old)


def test_curve_ends_must_match_the_vertices_exactly_within_tolerance():
    arc = quarter_arc()
    model = GeometryModel()
    tol = model.tolerance.effective_curve_fit_residual(2 * math.sqrt(2))
    start = model.add_point(*map(float, arc.exact_point(0)))
    near = model.add_point(*(float(v) for v in arc.exact_point(1)))
    model.add_curve(start, near, arc)
    far = model.add_point(float(arc.exact_point(1)[0]) + 3 * tol, *(float(v) for v in arc.exact_point(1)[1:]))
    with pytest.raises(GeometryError, match="NURBS curve ends"):
        model.add_curve(start, far, arc)
    planar = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0.), (1., 0.)))
    with pytest.raises(GeometryError, match="3D NURBS"):
        model.add_curve(start, near, planar)


def test_arc_axes_are_orthonormalised_and_degenerate_axes_refused():
    arc = nurbs_circular_arc((0., 0., 0.), (2., 0., 0.), (1., 1., 0.), 1., 0., math.pi / 2)
    points = arc.evaluate(np.linspace(0., 1., 201))
    assert np.abs(np.linalg.norm(points, axis=1) - 1.).max() <= 1e-15 * 8
    np.testing.assert_allclose(points[[0, -1]], ((1., 0., 0.), (0., 1., 0.)), atol=1e-15 * 8)
    for x_axis, y_axis in (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (0., 0., 0.)), ((1., 0., 0.), (-3., 0., 0.))):
        with pytest.raises(GeometryError):
            nurbs_circular_arc((0., 0., 0.), x_axis, y_axis, 1., 0., 1.)
    with pytest.raises(GeometryError):
        nurbs_circular_arc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., 0., 2 * math.pi + 1e-12)


@pytest.mark.parametrize("start", (1e16, -3e9, 123456.789))
def test_large_start_angles_give_the_libm_circle(start):
    arc = nurbs_circular_arc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 1., start, math.pi / 2)
    points = arc.evaluate(np.linspace(0., 1., 401))
    assert np.abs(np.linalg.norm(points, axis=1) - 1.).max() <= 1e-15 * 8
    np.testing.assert_allclose(points[0], (math.cos(start), math.sin(start), 0.), atol=1e-15 * 8)
    from anygeometry.exact_curves import analytic_curve_length
    assert abs(analytic_curve_length(arc, 1e-13) - math.pi / 2) <= 1e-12


def test_exact_points_outside_the_domain_are_refused():
    line = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 0.), (1., 0., 0.)), (1., 2.))
    for parameter in (-1, 2, Fraction(-1, 10**30)):
        with pytest.raises(GeometryError):
            line.exact_point(parameter)


def test_bounds_shrink_with_the_interval():
    arc = quarter_arc()
    for lower, upper in ((.499999, .500001), (0., 1e-9), (.3, .3), (1., 1.)):
        low, high = arc.bounds(lower, upper)
        samples = arc.evaluate(np.linspace(lower, upper, 11))
        assert np.all(samples >= low) and np.all(samples <= high)
        assert np.max(high - low) <= 4. * (upper - lower) + 1e-12


def test_revolving_a_nurbs_profile_is_refused_atomically():
    model, edge = edge_model(nurbs_circular_arc((2., 0., 0.), (1., 0., 0.), (0., 0., 1.), 1., 0., math.pi / 2))
    before = to_dict(model)
    with pytest.raises(GeometryError, match="NURBS profile"):
        model.revolve([edge], (0., 0., 0.), (0., 0., 1.), math.pi / 2)
    assert to_dict(model) == before


def test_nurbs_curve_face_queries_are_not_certified_from_samples():
    """Review A2a blocking: a spike between samples must not be classified DISJOINT."""
    from anygeometry import IntersectionKind, query_intersection
    model = GeometryModel()
    corners = [model.add_point(x, y, 0.) for x, y in ((-1., -1.), (2., -1.), (2., 1.), (-1., 1.))]
    face = model.add_face([model.add_line(a, b) for a, b in zip(corners, corners[1:] + corners[:1])])
    spike = NurbsCurve(1, (0., 0., .5001, .5002, .5003, 1., 1.),
                       ((0., 0., 1.), (.5001, 0., 1.), (.5002, 0., -1.), (.5003, 0., 1.), (1., 0., 1.)))
    start, end = model.add_point(0., 0., 1.), model.add_point(1., 0., 1.)
    edge = model.add_curve(start, end, spike)
    result = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    assert result.kind is not IntersectionKind.DISJOINT
    assert not result.classified
    far = model.add_curve(model.add_point(0., 0., 5.), model.add_point(1., 0., 5.),
                          NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 5.), (1., 0., 5.))))
    separated = query_intersection(model, model.handle("edge", far), model.handle("face", face))
    assert separated.kind is IntersectionKind.DISJOINT                 # the sound bounds test still decides


def test_bounds_are_exact_for_rational_insertion_and_float_parameter_maps():
    """Review A2a round 2 blocking: bounds contain the exact curve and the float-evaluated points."""
    heavy = NurbsCurve(1, (0., 0., 3., 3.), ((0., 0., 0.), (1e5, 0., 0.)), (1e6, 1.), (2.9999997, 3.))
    low, high = heavy.bounds()
    start, end = heavy.exact_point(0), heavy.exact_point(1)
    assert Fraction(low[0]) <= min(start[0], end[0]) and max(start[0], end[0]) <= Fraction(high[0])
    narrow = NurbsCurve(1, (1., 1., 1. + 2 ** -44, 1. + 2 ** -44), ((0., 0., 0.), (1., 0., 0.)))
    low, high = narrow.bounds(.1, .1)
    assert Fraction(low[0]) <= Fraction(1, 10) <= Fraction(high[0])
    point = narrow.evaluate(.1)
    assert low[0] <= point[0] <= high[0]
    rng = np.random.default_rng(4)
    for curve in (heavy, narrow, quarter_arc()):
        for _ in range(40):
            a, b = np.sort(rng.uniform(0., 1., 2))
            low, high = curve.bounds(a, b)
            for value in (Fraction(a), Fraction(b), (Fraction(a) + Fraction(b)) / 2):
                exact = curve.exact_point(value)
                assert all(Fraction(lo) <= x <= Fraction(hi) for lo, x, hi in zip(low, exact, high))


def test_a_touching_heavy_rational_curve_is_not_disjoint():
    from anygeometry import IntersectionKind, query_intersection
    curve = NurbsCurve(1, (0., 0., 3., 3.), ((0., 0., 0.), (1e5, 0., 0.)), (1e6, 1.), (2.9999997, 3.))
    x0 = float(curve.exact_point(0)[0])
    model = GeometryModel()
    corners = [model.add_point(x0, y, z) for y, z in ((-1., -1.), (1., -1.), (1., 1.), (-1., 1.))]
    face = model.add_face([model.add_line(a, b) for a, b in zip(corners, corners[1:] + corners[:1])])
    edge = model.add_curve(model.add_point(*map(float, curve.exact_point(0))),
                           model.add_point(*map(float, curve.exact_point(1))), curve)
    result = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    assert result.kind is not IntersectionKind.DISJOINT


def test_distance_lower_bounds_are_sound_and_finite():
    line = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 0.), (1., 0., 0.)), (1., 1.))
    far = line.distance_lower_bound(0., 1., (1e200, 0., 0.))
    assert math.isfinite(far) and Fraction(far) <= Fraction(1e200) - 1
    zero = NurbsCurve(1, (0., 0., 1., 1.), ((0., 0., 0.), (0., 0., 0.)))
    assert Fraction(zero.distance_lower_bound(0., 1., (1., 1., 0.))) ** 2 <= 2
    arc = quarter_arc()
    target = np.array((1. + 6. * math.cos(.4), 2. + 6. * math.sin(.4), 3.))
    for a, b in ((0., 1.), (.2, .3), (.26, .27)):
        bound = arc.distance_lower_bound(a, b, target)
        samples = arc.evaluate(np.linspace(a, b, 201))
        assert bound <= np.linalg.norm(samples - target, axis=1).min()


def test_nearly_parallel_axes_still_give_a_circle():
    for x_axis, y_axis in (((1., 1., 0.), (1., 1. + 1e-10, 0.)), ((1., 1.), (1., 1. + 1e-10))):
        arc = nurbs_circular_arc((0.,) * len(x_axis), x_axis, y_axis, 1., 0., math.pi / 2)
        assert np.abs(np.linalg.norm(arc.evaluate(np.linspace(0., 1., 1001)), axis=1) - 1.).max() <= 1e-15 * 8


def test_nearly_parallel_axes_keep_the_input_plane():
    """Review A2a round 3: the arc lies in the plane of the given axes."""
    for epsilon in (1e-10, 1e-11):
        x_axis, y_axis = np.array((1., 2., 3.)), np.array((1. + epsilon, 2., 3.))
        arc = nurbs_circular_arc((0., 0., 0.), x_axis, y_axis, 1., 0., math.pi / 2)
        normal = np.array((0., 3., -2.)) / math.sqrt(13.)            # the exact plane of the stored axes
        points = arc.evaluate(np.linspace(0., 1., 201))
        assert np.abs(points @ normal).max() <= 1e-14
        assert np.abs(np.linalg.norm(points, axis=1) - 1.).max() <= 1e-15 * 8



@pytest.mark.parametrize("scale", [1e-108, 1e-150, 1e-200, 1e150, 1e200])
def test_arc_axes_scaled_far_from_one_give_the_circle_in_the_input_plane(scale):
    """Confirmation review of A2a: the orthogonalisation rounds the whole exact quotient once;
    a numerator rounded first underflowed (radius 1.023 at 1e-108) or overflowed (1e150)."""
    x_axis, y_axis = np.array((1., 2., 3.)) * scale, np.array((1., 4., 2.)) * scale
    arc = nurbs_circular_arc((0., 0., 0.), x_axis, y_axis, 1., 0., math.pi / 2)
    normal = np.cross((1., 2., 3.), (1., 4., 2.))
    normal = normal / np.linalg.norm(normal)
    points = arc.evaluate(np.linspace(0., 1., 201))
    assert np.abs(np.linalg.norm(points, axis=1) - 1.).max() <= 1e-15 * 8
    assert np.abs(points @ normal).max() <= 1e-14
    assert np.allclose(points[0], np.array((1., 2., 3.)) / math.sqrt(14.), atol=1e-15)


def test_bounds_contain_the_float_evaluation():
    curve = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((0., 1e6, 0.), (.5, 1e6, 0.), (1., 1e6, 0.)))
    t = np.linspace(0., 1., 4001)
    low, high = curve.bounds()
    points = curve.evaluate(t)
    assert np.all(points >= low) and np.all(points <= high)
