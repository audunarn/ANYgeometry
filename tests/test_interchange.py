"""Exact owner materialization, refusal and source-preservation regressions."""
from dataclasses import replace
from fractions import Fraction as Q

import numpy as np
import pytest

from anygeometry import GeometryModel, GeometryError
from anygeometry.entities import OrientedEdge
from anygeometry.interchange import face_support_for_interchange, InterchangeSupportError
from anygeometry.nurbs import NurbsCurve, NurbsSurface
from anygeometry.serialization import to_dict
from anygeometry.surfaces import CoonsSurface, Plane


def polynomial_face(curved=True, reverse=False):
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    controls = ((1, -.5, .5), (2.5, 1, .75), (1, 2.5, 1), (-.5, 1, .25))
    loop = []
    for i in range(4):
        start, end = vertices[i], vertices[(i+1) % 4]
        if reverse:
            start, end = end, start
        if curved:
            middle = model.add_point(*controls[i])
            edge = model.add_spline(start, (middle,), end)
        else:
            edge = model.add_line(start, end)
        loop.append(OrientedEdge(edge, not reverse))
    face = model.add_face_from_loop(tuple(loop), (0, 1, 2, 3), surface=CoonsSurface())
    return model, face


def bezier(points, t):
    points = [tuple(Q(float(x)) for x in p) for p in points]
    while len(points) > 1:
        points = [tuple((1-t)*a+t*b for a, b in zip(p, q)) for p, q in zip(points, points[1:])]
    return points[0]


@pytest.mark.parametrize("curved", (False, True))
@pytest.mark.parametrize("reverse", (False, True))
def test_polynomial_coons_is_exact_and_does_not_mutate(curved, reverse):
    model, face_id = polynomial_face(curved, reverse)
    before = to_dict(model)
    support = face_support_for_interchange(model, face_id)
    assert isinstance(support, NurbsSurface)
    face = model.faces[face_id]
    sides = []
    for chain in face.sides():
        use = chain[0]
        edge = model.edges[use.edge]
        controls = getattr(edge.curve, "control_vertices", ())
        points = [model.vertex_position(i) for i in (edge.start, *controls, edge.end)]
        sides.append(points if use.forward else points[::-1])
    bottom, right, top, left = sides
    top, left = top[::-1], left[::-1]
    for u, v in ((Q(0), Q(0)), (Q(1, 3), Q(2, 5)), (Q(1), Q(1)), (Q(3, 4), Q(1, 4))):
        b, r, t, l = bezier(bottom, u), bezier(right, v), bezier(top, u), bezier(left, v)
        c00, c10, c01, c11 = map(lambda p: tuple(Q(float(x)) for x in p),
                                 (bottom[0], bottom[-1], top[0], top[-1]))
        expected = tuple((1-v)*b[k]+v*t[k]+(1-u)*l[k]+u*r[k]
                         - ((1-u)*(1-v)*c00[k]+u*(1-v)*c10[k]
                            +(1-u)*v*c01[k]+u*v*c11[k]) for k in range(3))
        assert support.exact_point(u, v) == expected
        np.testing.assert_allclose(support.evaluate(float(u), float(v)),
                                   model.face_support_point(face_id, float(u), float(v)), atol=2e-15)
    assert to_dict(model) == before


@pytest.mark.parametrize("support", (
    Plane((0, 0, 0), (1, 0, 0), (0, 1, 0)),
    NurbsSurface(1, 1, (0, 0, 1, 1), (0, 0, 1, 1),
                 (((0, 0, 0), (0, 2, 0)), ((2, 0, 0), (2, 2, 0)))),
    CoonsSurface(((0, 0, 0), (2, 0, 0)), ((2, 0, 0), (2, 2, 0)),
                 ((0, 2, 0), (2, 2, 0)), ((0, 0, 0), (0, 2, 0))),
))
def test_explicit_support_is_returned_unchanged(support):
    model = GeometryModel()
    vertices = model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    if isinstance(support, NurbsSurface):
        from anygeometry.nurbs import CurveOnSurface
        uv = ((0, 0), (1, 0), (1, 1), (0, 1))
        edges = [model.add_curve(vertices[i], vertices[(i+1) % 4], CurveOnSurface(
            support, NurbsCurve(1, (0, 0, 1, 1), (uv[i], uv[(i+1) % 4])))) for i in range(4)]
    else:
        edges = [model.add_line(vertices[i], vertices[(i+1) % 4]) for i in range(4)]
    face_id = model.add_face(edges, surface=support)
    assert face_support_for_interchange(model, face_id) is support


def rational_sweep():
    model = GeometryModel()
    profile = NurbsCurve(2, (0, 0, 0, 1, 1, 1), ((0, 0, 0), (0, 1, 1), (0, 2, 0)), (1, .5, 1))
    opposite = replace(profile, controls=tuple((x+4, y, z) for x, y, z in profile.controls))
    a, b, c, d = model.add_points(((0, 0, 0), (0, 2, 0), (4, 2, 0), (4, 0, 0)))
    edges = (model.add_curve(a, b, profile), model.add_line(b, c),
             model.add_curve(d, c, opposite), model.add_line(d, a))
    face_id = model.add_face_from_loop(tuple(OrientedEdge(e, forward) for e, forward in
        zip(edges, (True, True, False, True))), (0, 1, 2, 3), surface=CoonsSurface())
    return model, face_id, profile


@pytest.mark.parametrize("declared", (False, True))
def test_existing_rational_sweep_materialization_is_reused(declared):
    model, face_id, profile = rational_sweep()
    if declared:
        from anygeometry.nurbs_sweep_chart import install_sweep_chart
        install_sweep_chart(model, face_id)
    before = to_dict(model)
    support = face_support_for_interchange(model, face_id)
    assert isinstance(support, NurbsSurface)
    assert support.exact_point(Q(1, 3), Q(1, 4)) == tuple(
        x+d for x, d in zip(profile.exact_point(Q(1, 3)), (Q(1), Q(0), Q(0))))
    assert to_dict(model) == before


def test_nonpolynomial_coons_refuses_without_mutation():
    model = GeometryModel()
    a, b, c, d, via = model.add_points(((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0), (1, -1, 0)))
    edges = (model.add_arc(a, via, b), model.add_line(b, c), model.add_line(c, d), model.add_line(d, a))
    face_id = model.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges), (0, 1, 2, 3), surface=CoonsSurface())
    before = to_dict(model)
    with pytest.raises(InterchangeSupportError, match="not admitted polynomial"):
        face_support_for_interchange(model, face_id)
    assert to_dict(model) == before


def test_translated_bezier_profile_is_exact_without_new_model_state():
    model = GeometryModel()
    a, b, c, d, lower, upper = model.add_points(
        ((0, 0, 0), (0, 2, 0), (4, 2, 0), (4, 0, 0), (0, 1, 1), (4, 1, 1)))
    edges = (model.add_spline(a, (lower,), b), model.add_line(b, c),
             model.add_spline(c, (upper,), d), model.add_line(d, a))
    face_id = model.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges),
                                       (0, 1, 2, 3), surface=CoonsSurface())
    before = to_dict(model)
    support = face_support_for_interchange(model, face_id)
    u, v = Q(1, 3), Q(2, 5)
    assert support.exact_point(u, v) == (4*v, 2*u, 2*u*(1-u))
    assert to_dict(model) == before


def test_multiple_edges_per_coons_side_refuse_without_mutation():
    model = GeometryModel()
    a, middle, b, c, d = model.add_points(((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0)))
    edges = (model.add_line(a, middle), model.add_line(middle, b), model.add_line(b, c),
             model.add_line(c, d), model.add_line(d, a))
    face_id = model.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges),
                                       (0, 2, 3, 4), surface=CoonsSurface())
    before = to_dict(model)
    with pytest.raises(InterchangeSupportError, match="four single-edge sides"):
        face_support_for_interchange(model, face_id)
    assert to_dict(model) == before


def test_unrepresentable_degree_elevation_refuses_exact_claim():
    model = GeometryModel()
    a, b, c, d, bottom, top1, top2 = model.add_points(
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0),
         (1, -.5, 1), (.5, 2, 0), (1.5, 2, 0)))
    edges = (model.add_spline(a, (bottom,), b), model.add_line(b, c),
             model.add_spline(c, (top2, top1), d), model.add_line(d, a))
    face_id = model.add_face_from_loop(tuple(OrientedEdge(e, True) for e in edges),
                                       (0, 1, 2, 3), surface=CoonsSurface())
    before = to_dict(model)
    with pytest.raises(InterchangeSupportError, match="not representable"):
        face_support_for_interchange(model, face_id)
    assert to_dict(model) == before


@pytest.mark.parametrize("face_id", (False, -1, 999, "1"))
def test_invalid_face_is_typed_refusal(face_id):
    model, _ = polynomial_face(False)
    with pytest.raises(GeometryError):
        face_support_for_interchange(model, face_id)
