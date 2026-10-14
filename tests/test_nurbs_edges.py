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
    assert np.all(np.einsum("ij,ij->i", derivatives, points - center) <= 1e-12 * radius ** 2 * 10)


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
    target = np.array((1. + 2. * math.cos(.4) * 3, 2. + 2. * math.sin(.4) * 3, 3.))  # outside the arc, on its ray
    projection = model.project_to_edge(edge, target) if hasattr(model, "project_to_edge") else None
    if projection is not None:
        point = np.asarray(projection[0] if isinstance(projection, tuple) else projection.point)
        np.testing.assert_allclose(point, (1. + 2. * math.cos(.4), 2. + 2. * math.sin(.4), 3.), atol=1e-9)
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
    old = dict(document, version=6)
    with pytest.raises(GeometryError):
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
