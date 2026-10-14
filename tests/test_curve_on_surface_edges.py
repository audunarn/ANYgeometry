"""Curves on NURBS supports as model edges (generality slice A3b, B-S1).

References: the analytic cylinder (arc lengths r * dtheta, heights), exact
rational points, dense sampling for bounds and projection, schema-7 round trips.
"""
from __future__ import annotations

from fractions import Fraction
import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface, cylinder_to_nurbs

RADIUS, HEIGHT = 2., 3.


def support():
    return cylinder_to_nurbs(Cylinder((1., -1., .5), (0., 0., 1.), (1., 0., 0.), RADIUS, HEIGHT, .2, 2.5))


def on(surface, a, b, weights=None):
    return CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b), weights))


def edge_model(curve):
    model = GeometryModel()
    start, end = (model.add_point(*map(float, curve.exact_point(t))) for t in (0, 1))
    return model, model.add_curve(start, end, curve)


def angle(surface, u):
    x, y, _z = surface.evaluate(u, 0.) - np.array((1., -1., .5))
    return math.atan2(y, x)


ISO_CASES = ((((.1, .4), (.8, .4)), "u"), (((.3, 0.), (.3, 1.)), "v"), (((.9, 1.), (.2, 1.)), "u"))


@pytest.mark.parametrize("controls, running", ISO_CASES)
def test_iso_edges_have_exact_lengths_bounds_and_projection(controls, running):
    surface = support()
    curve = on(surface, *controls)
    assert curve._iso_world() is not None
    model, edge = edge_model(curve)
    if running == "u":
        expected = RADIUS * abs(angle(surface, controls[1][0]) - angle(surface, controls[0][0]))
    else:
        expected = HEIGHT * abs(controls[1][1] - controls[0][1])
    assert abs(model.edge_length(edge) - expected) <= 1e-12 * expected
    rng = np.random.default_rng(31)
    for _ in range(20):
        a, b = np.sort(rng.uniform(0., 1., 2))
        low, high = curve.bounds(a, b)
        samples = curve.evaluate(np.linspace(a, b, 301))
        assert np.all(samples >= low) and np.all(samples <= high)
        exact = curve.exact_point(Fraction(a))
        assert all(Fraction(lo) <= x <= Fraction(hi) for lo, x, hi in zip(low, exact, high))
    target = np.array((4., 2., 1.))
    point, parameter, distance = model.closest_edge_point(edge, target)
    dense = np.linalg.norm(curve.evaluate(np.linspace(0., 1., 20001)) - target, axis=1)
    assert distance <= dense.min() + 1e-12 and distance >= dense.min() - 1e-6


def test_general_pcurve_edges_use_the_surface_enclosure():
    surface = support()
    curve = CurveOnSurface(surface, NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((.1, .1), (.8, .3), (.6, .9)),
                                               (1., .7, 1.)))
    assert curve._iso_world() is None
    model, edge = edge_model(curve)
    samples = curve.evaluate(np.linspace(0., 1., 4001))
    low, high = curve.bounds()
    assert np.all(samples >= low) and np.all(samples <= high)
    polyline = float(np.sum(np.linalg.norm(np.diff(samples, axis=0), axis=1)))
    assert abs(model.edge_length(edge) - polyline) <= 1e-6 * polyline


def test_exact_end_test_split_and_reverse():
    surface = support()
    curve = on(surface, (.1, .4), (.8, .4))
    model = GeometryModel()
    start = model.add_point(*map(float, curve.exact_point(0)))
    far = model.add_point(*(np.array(list(map(float, curve.exact_point(1)))) + (1e-3, 0., 0.)))
    with pytest.raises(GeometryError, match="ends do not match"):
        model.add_curve(start, far, curve)
    model, edge = edge_model(curve)
    total = model.edge_length(edge)
    vertex, (first, second) = model.split_edge(edge, .37)
    for part in (first, second):
        assert isinstance(model.edges[part].curve, CurveOnSurface) and model.edges[part].curve.support == surface
    assert abs(model.edge_length(first) + model.edge_length(second) - total) <= 1e-12 * total
    np.testing.assert_allclose(model.vertex_position(vertex), curve.evaluate(.37), atol=0)


def test_transform_folds_into_the_support():
    surface = support()
    curve = on(surface, (.3, 0.), (.3, 1.))
    matrix = np.eye(4)
    c, s = math.cos(.4), math.sin(.4)
    matrix[:3, :3] = ((c, -s, 0.), (s, c, 0.), (0., 0., 1.))
    matrix[:3, 3] = (2., 0., -1.)
    moved = curve.transformed(matrix)
    assert moved.transform == curve.transform and moved.support == surface.transformed(matrix)
    t = np.linspace(0., 1., 51)
    np.testing.assert_allclose(moved.evaluate(t), curve.evaluate(t) @ matrix[:3, :3].T + matrix[:3, 3], atol=1e-13)


def test_schema_7_round_trip_and_refusals():
    from anygeometry import IntersectionKind, query_intersection
    surface = support()
    curve = on(surface, (.1, .4), (.8, .4))
    model, edge = edge_model(curve)
    document = to_dict(model)
    assert document["version"] == 7
    restored = from_dict(document)
    assert restored.edges[edge].curve == curve and to_dict(restored) == document
    with pytest.raises(GeometryError, match="NURBS profile"):
        model.revolve([edge], (0., 0., 0.), (0., 0., 1.), math.pi / 2)
    corners = [model.add_point(x, y, 1.) for x, y in ((-5., -5.), (5., -5.), (5., 5.), (-5., 5.))]
    face = model.add_face([model.add_line(a, b) for a, b in zip(corners, corners[1:] + corners[:1])])
    result = query_intersection(model, model.handle("edge", edge), model.handle("face", face))
    assert not result.classified or result.kind is IntersectionKind.DISJOINT


# ------------------------------------------------------------------ review A3b B-S1 round 1


def unit_plane():
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))


def test_iso_bounds_hold_with_a_narrow_surface_knot_span():
    d = 2. ** -44
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., .3 - d, .3 + d, 1., 1.),
                           tuple(tuple((float(u), y, 0.) for y in (0., 0., 1., 1.)) for u in (0, 1)))
    curve = on(surface, (0., .3), (1., .3))
    low, high = curve.bounds()
    samples = curve.evaluate(np.linspace(0., 1., 4001))
    assert np.all(samples >= low) and np.all(samples <= high)
    model, edge = edge_model(curve)
    assert model.closest_edge_point(edge, curve.evaluate(.179))[2] == 0.


def test_iso_bounds_hold_with_a_narrow_pcurve_knot_span():
    curve = CurveOnSurface(unit_plane(), NurbsCurve(1, (1., 1., 1. + 2. ** -40, 1. + 2. ** -40), ((0., .5), (1., .5))))
    for t in np.linspace(0., 1., 401):
        low, high = curve.bounds(t, t)
        point = curve.evaluate(t)
        assert np.all(point >= low) and np.all(point <= high)


def test_transform_folding_keeps_huge_transforms_and_extrusion_is_refused():
    huge = CurveOnSurface(unit_plane(), NurbsCurve(1, (0., 0., 1., 1.), ((0., .5), (1., .5))),
                          transform=((1e200, 0., 0., 0.), (0., 1e200, 0., 0.), (0., 0., 1e200, 0.), (0., 0., 0., 1.)))
    moved = huge.transformed(np.eye(4))
    np.testing.assert_allclose(moved.evaluate(np.linspace(0., 1., 5)), huge.evaluate(np.linspace(0., 1., 5)), rtol=1e-15)
    model, edge = edge_model(on(support(), (.1, .4), (.8, .4)))
    with pytest.raises(GeometryError, match="NURBS support"):
        model.extrude([edge], (0., 0., 1.))
