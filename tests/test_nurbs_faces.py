"""NURBS faces in the model (generality slice A3b, B-S2).

References: the analytic cylinder area r |sweep| h, an independent tensor
Gauss-Legendre integral of |S_u x S_v| over a chart triangle (Duffy map), exact
chart chaining, transform commutation and schema-7 round trips.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, from_dict, to_dict, transform
from anygeometry.material_arrangement import MaterialDomain
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface, cylinder_to_nurbs


def wavy_patch():
    rng = np.random.default_rng(41)
    controls = np.zeros((5, 4, 3))
    controls[..., 0] = np.linspace(0., 4., 5)[:, None]
    controls[..., 1] = np.linspace(0., 3., 4)[None, :]
    controls[..., 2] = rng.uniform(-.5, .5, (5, 4))
    weights = rng.uniform(.6, 2., (5, 4))
    return NurbsSurface(3, 2, (0., 0., 0., 0., .5, 1., 1., 1., 1.), (0., 0., 0., .4, 1., 1., 1.),
                        tuple(tuple(map(tuple, row)) for row in controls), tuple(map(tuple, weights)))


def nurbs_face(surface, chart_points, model=None):
    model = model or GeometryModel()
    curves = [CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b)))
              for a, b in zip(chart_points, chart_points[1:] + chart_points[:1])]
    vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in curves]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % len(curves)], curve) for i, curve in enumerate(curves)]
    return model, model.add_face(edges, surface=surface), edges


def triangle_area_reference(surface, a, b, c, order=60):
    """Gauss-Legendre over the chart triangle via the Duffy map (independent of the Green route)."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s, w = .5 * (nodes + 1), .5 * weights
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    total = 0.
    for si, wi in zip(s, w):
        for ti, wj in zip(s, w):
            point = a + si * (b - a) + si * ti * (c - b)
            jacobian_chart = abs((b - a)[0] * (c - b)[1] - (b - a)[1] * (c - b)[0]) * si
            s_u, s_v = surface.derivatives_many(point[None, :])
            total += wi * wj * jacobian_chart * float(np.linalg.norm(np.cross(s_u[0], s_v[0])))
    return total


@pytest.mark.parametrize("sweep", (math.pi, 2.4, -1.7))
def test_cylinder_face_area(sweep):
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., .5, sweep))
    model, face, _edges = nurbs_face(surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    expected = 2. * abs(sweep) * 3.
    area = MaterialDomain.from_model(model, face).original_world_area(1e-12)
    assert abs(area - expected) <= 1e-10 * expected
    assert model.validate_topology() == ()


def clip(polygon, axis, value, keep_below):
    """Sutherland-Hodgman clip of a convex chart polygon by the line ``q[axis] = value``."""
    result = []
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        inside_a = (a[axis] <= value) if keep_below else (a[axis] >= value)
        inside_b = (b[axis] <= value) if keep_below else (b[axis] >= value)
        if inside_a:
            result.append(a)
        if inside_a != inside_b:
            r = (value - a[axis]) / (b[axis] - a[axis])
            result.append(tuple(a[k] + r * (b[k] - a[k]) for k in range(2)))
    return result


def piecewise_reference(surface, triangle, u_breaks, v_breaks):
    """Gauss-Legendre on the triangle split along the surface's knot lines (each piece smooth)."""
    pieces = [list(triangle)]
    for axis, breaks in ((0, u_breaks), (1, v_breaks)):
        for value in breaks:
            pieces = [part for piece in pieces for part in (clip(piece, axis, value, True), clip(piece, axis, value, False))
                      if len(part) >= 3]
    return sum(triangle_area_reference(surface, piece[0], piece[k], piece[k + 1], order=30)
               for piece in pieces for k in range(1, len(piece) - 1))


@pytest.mark.parametrize("triangle", ([(.05, .05), (.45, .1), (.2, .35)],          # inside one knot span
                                      [(.1, .1), (.9, .3), (.4, .8)]))            # across the knot lines
def test_trimmed_face_area_against_gauss_legendre(triangle):
    surface = wavy_patch()
    model, face, _edges = nurbs_face(surface, triangle)
    area = MaterialDomain.from_model(model, face).original_world_area(1e-12)
    reference = piecewise_reference(surface, triangle, (.5,), (.4,))
    assert abs(area - reference) <= 1e-10 * reference


def test_face_bounds_contain_the_surface():
    surface = wavy_patch()
    model, face, _edges = nurbs_face(surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    bounds = np.asarray(model.conservative_face_bounds(face))
    samples = surface.evaluate_many(np.random.default_rng(42).uniform(0., 1., (20000, 2)))
    assert np.all(samples >= bounds[:3]) and np.all(samples <= bounds[3:])


def test_nurbs_face_rules_refuse_foreign_edges_and_open_chart_loops():
    surface = wavy_patch()
    model = GeometryModel()
    curves = [CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b)))
              for a, b in (((0., 0.), (1., 0.)), ((1., 0.), (1., 1.)))]
    a, b, c = (model.add_point(*map(float, point)) for point in
               (curves[0].exact_point(0), curves[0].exact_point(1), curves[1].exact_point(1)))
    first, second = model.add_curve(a, b, curves[0]), model.add_curve(b, c, curves[1])
    straight = model.add_line(c, a)
    with pytest.raises(GeometryError, match="exact chart curve"):
        model.add_face([first, second, straight], surface=surface)
    # the same face, re-supported by another NURBS surface: its edges are not curves on that support
    plane_like = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                              (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))
    model, face, edges = nurbs_face(plane_like, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    with pytest.raises(GeometryError, match="exact chart curve"):
        model.set_face_surface(face, surface)
    assert model.validate_topology() == ()


def test_transform_and_round_trip():
    surface = wavy_patch()
    model, face, edges = nurbs_face(surface, [(.1, .1), (.9, .3), (.4, .8)])
    area = MaterialDomain.from_model(model, face).original_world_area(1e-12)
    document = to_dict(model)
    assert document["version"] == 7
    assert to_dict(from_dict(document)) == document
    matrix = np.eye(4)
    c, s = math.cos(.6), math.sin(.6)
    matrix[:3, :3] = ((1., 0., 0.), (0., c, -s), (0., s, c))
    matrix[:3, 3] = (5., -2., 1.)
    transform(model, matrix)
    moved = model.faces[face].surface
    assert moved == surface.transformed(matrix)
    assert all(model.edges[edge].curve.support == moved for edge in edges)
    assert model.validate_topology() == ()
    moved_area = MaterialDomain.from_model(model, face).original_world_area(1e-12)
    assert abs(moved_area - area) <= 1e-10 * area
