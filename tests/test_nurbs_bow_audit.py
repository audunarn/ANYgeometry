"""B1-pre: one curved bow face audited clean (generality slice A3b, B-S4, owner option a).

A rational NURBS half-bow (plan shape a rational quadratic from the side to the
stem, flared towards the deck) bounded by its four exact iso-curves. The audit
classifies every pair: the face's own edges and vertices by shared topology,
edge pairs by the exact certificate of ``nurbs_audit`` (they meet only at shared
ends). Area vs an independent Gauss-Legendre integral.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import GeometryModel
from anygeometry.material_arrangement import MaterialDomain
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.nurbs_audit import certify_only_shared_ends

BREADTH, LENGTH, DEPTH = 7.5, 6., 4.


def bow_surface(flare=1.):
    w = math.cos(math.pi / 4)
    bottom = [(0., .8 * BREADTH, 0.), (.9 * LENGTH, .8 * BREADTH, 0.), (.9 * LENGTH, 0., 0.)]
    deck = [(0., BREADTH * flare, DEPTH), (LENGTH, BREADTH * flare, DEPTH), (LENGTH, 0., DEPTH)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)))


def bow_model(surface):
    model = GeometryModel()
    corners = [(0., 0.), (1., 0.), (1., 1.), (0., 1.)]
    curves = [CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b)))
              for a, b in zip(corners, corners[1:] + corners[:1])]
    vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in curves]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], curve) for i, curve in enumerate(curves)]
    return model, model.add_face(edges, surface=surface), curves


def area_reference(surface, order=40):
    nodes, weights = np.polynomial.legendre.leggauss(order)
    s, w = .5 * (nodes + 1), .5 * weights
    uv = np.array([(a, b) for a in s for b in s])
    s_u, s_v = surface.derivatives_many(uv)
    return float(np.kron(w, w) @ np.linalg.norm(np.cross(s_u, s_v), axis=1))


def test_the_bow_face_audits_clean():
    model, face, _curves = bow_model(bow_surface())
    report = model.strict_audit()
    assert report.clean, [(issue.code, issue.details) for issue in report.issues]


def test_the_bow_face_area_matches_gauss_legendre():
    surface = bow_surface()
    model, face, _curves = bow_model(surface)
    area = MaterialDomain.from_model(model, face).original_world_area(1e-12)
    assert abs(area - area_reference(surface)) <= 1e-10 * area


def test_edge_certificate_proves_shared_ends_only_and_refuses_crossings():
    surface = bow_surface()
    _model, _face, curves = bow_model(surface)
    for i in range(4):
        for j in range(i + 1, 4):
            assert certify_only_shared_ends(curves[i], curves[j]) is True
    # two iso-lines that cross in the chart (u = .5 and v = .5) cross in the world: not certified
    crossing_u = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((.5, 0.), (.5, 1.))))
    crossing_v = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((0., .5), (1., .5))))
    assert certify_only_shared_ends(crossing_u, crossing_v) is None


def test_an_unauditable_pair_still_blocks():
    surface = bow_surface()
    model, face, curves = bow_model(surface)
    # a free iso-curve across the face: its crossing with the face is not classified (plane/NURBS pairs: A4)
    a = model.add_point(*map(float, surface.exact_point(.5, 0.)))
    b = model.add_point(*map(float, surface.exact_point(.5, 1.)))
    model.add_curve(a, b, CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((.5, 0.), (.5, 1.)))))
    assert not model.strict_audit().clean


# ------------------------------------------------------------------ review A3b round 1


def unit_plane():
    return NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                        (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 0.), (1., 1., 0.))))


def test_shared_ends_must_be_the_same_vertices():
    """A near-duplicate vertex C (exact end coincides with A's) must not pass as shared topology."""
    surface = unit_plane()
    d = 2. ** -27
    first = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((.25, .25), (.75, .25))))
    second = CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((.25, .25 + d), (.25, .25))))
    model = GeometryModel()
    a = model.add_point(.25, .25, 0.)
    b = model.add_point(.75, .25, 0.)
    c = model.add_point(.25, .25 + d / 2, 0.)
    first_edge, second_edge = model.add_curve(a, b, first), model.add_curve(a, c, second)
    from anygeometry.nurbs_audit import shared_ends_consistent
    assert not shared_ends_consistent(model.edges[first_edge], model.edges[second_edge])
    _bow, _face, curves = bow_model(bow_surface())
    bow = _bow
    edges = sorted(bow.edges)
    assert shared_ends_consistent(bow.edges[edges[0]], bow.edges[edges[1]])


def test_exact_trim_validation_refuses_a_narrow_excursion():
    from anygeometry import GeometryError
    surface = unit_plane()
    excursion = NurbsCurve(1, (0., 0., 513 / 1024, 514 / 1024, 515 / 1024, 516 / 1024, 1., 1.),
                           ((1 / 8, 1 / 8), (3 / 8, 1 / 8), (3 / 8, 15 / 16), (5 / 8, 15 / 16), (5 / 8, 1 / 8),
                            (7 / 8, 1 / 8)))
    rest = [((7 / 8, 1 / 8), (7 / 8, 7 / 8)), ((7 / 8, 7 / 8), (1 / 8, 7 / 8)), ((1 / 8, 7 / 8), (1 / 8, 1 / 8))]
    curves = [CurveOnSurface(surface, excursion)] + [
        CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), pair)) for pair in rest]
    model = GeometryModel()
    vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in curves]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 4], curve) for i, curve in enumerate(curves)]
    with pytest.raises(GeometryError, match="not certified simple|self-intersects"):
        model.add_face(edges, surface=surface)


def test_exact_trim_validation_accepts_a_hole_close_to_a_curved_trim():
    surface = unit_plane()
    arch = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((0., 0.), (.5, 1.), (1., 0.)))      # v = 2u(1 - u)
    model = GeometryModel()
    outer = [CurveOnSurface(surface, arch.subcurve(1., 0.)) if False else CurveOnSurface(surface, arch),
             CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), ((1., 0.), (0., 0.))))]
    u0, u1, v0, v1 = 33 / 64 - 1 / 8192, 33 / 64 + 1 / 8192, 4090 / 8192, 4091 / 8192
    hole_points = [(u0, v0), (u0, v1), (u1, v1), (u1, v0)]
    hole = [CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (p, q)))
            for p, q in zip(hole_points, hole_points[1:] + hole_points[:1])]
    # the arch and its chord as two edges plus a third point: faces need three edges
    halves = [CurveOnSurface(surface, arch.subcurve(0., .5)), CurveOnSurface(surface, arch.subcurve(.5, 1.)), outer[1]]
    vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in halves]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % 3], curve) for i, curve in enumerate(halves)]
    face = model.add_face(edges, surface=surface)
    hole_vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in hole]
    hole_edges = tuple(model.add_curve(hole_vertices[i], hole_vertices[(i + 1) % 4], curve)
                       for i, curve in enumerate(hole))
    errors = model._validate_nurbs_trims(face, surface, (model.faces[face].loop,
                                                         tuple(model._order_loop(hole_edges))))
    assert errors == []


def test_extrude_refuses_unknown_edges_with_a_geometry_error():
    from anygeometry import GeometryError
    with pytest.raises(GeometryError):
        GeometryModel().extrude([999], (0., 0., 1.))


# ------------------------------------------------------------------ review A3b round 2


def add_edges(model, curves, close=True):
    vertices = [model.add_point(*map(float, curve.exact_point(0))) for curve in curves]
    if not close:
        vertices.append(model.add_point(*map(float, curves[-1].exact_point(1))))
    return [model.add_curve(vertices[i], vertices[(i + 1) % len(vertices)], curve) for i, curve in enumerate(curves)]


def line(surface, a, b, weights=None, domain=None):
    return CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (a, b), weights, domain))


def test_uncertified_curve_pairs_block_without_a_general_fallback():
    surface = unit_plane()
    knots = (0., 0.) + tuple(.5 + k * 2. ** -40 for k in range(1, 5)) + (1., 1.)
    zigzag = CurveOnSurface(surface, NurbsCurve(1, knots, ((1 / 8, 1 / 8), (1 / 2, 1 / 8), (1 / 2, 7 / 8),
                                                           (5 / 8, 7 / 8), (5 / 8, 1 / 8), (7 / 8, 1 / 8))))
    # crosses the excursion at x = 1/2; each end vertex lies outside the other edge's box, so only
    # the edge pair needs a classification (projection on general pcurves is slow: box bounds)
    diagonal = line(surface, (1 / 8, 1 / 8), (3 / 4, 15 / 16))
    model = GeometryModel()
    start = model.add_point(1 / 8, 1 / 8, 0.)
    model.add_curve(start, model.add_point(7 / 8, 1 / 8, 0.), zigzag)
    model.add_curve(start, model.add_point(3 / 4, 15 / 16, 0.), diagonal)
    report = model.strict_audit()
    assert not report.clean
    assert any("curve_on_nurbs_pair_not_certified" in str(dict(issue.details).get("reason")) for issue in report.issues)


def test_a_hole_outside_a_scaled_arch_is_refused_exactly():
    scale = 2. ** 60
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., scale, 0.)), ((scale, 0., 0.), (scale, scale, 0.))))
    arch = NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((0., 0.), (.5, 1.), (1., 0.)), (1., 2., 1.))
    outer = [CurveOnSurface(surface, arch.subcurve(0., .5)), CurveOnSurface(surface, arch.subcurve(.5, 1.)),
             line(surface, (1., 0.), (0., 0.))]
    model = GeometryModel()
    face = model.add_face(add_edges(model, outer), surface=surface)
    tiny = 2. ** -55
    q_to_b = line(surface, (.5, tiny), (.5, 1.), (1., 2.), (.5, 1.))
    b_to_c = line(surface, (.5, 1.), (.75, 1.))
    c_to_q = line(surface, (0., tiny), (.75, 1.), (1., 2.), (1., .5))
    hole_edges = add_edges(model, [q_to_b, b_to_c, c_to_q])
    errors = model._validate_nurbs_trims(face, surface, (model.faces[face].loop, tuple(model._order_loop(hole_edges))))
    assert any("not strictly inside" in error or "not certified" in error for error in errors)


def test_a_self_crossing_trim_edge_is_refused():
    from anygeometry import GeometryError
    surface = unit_plane()
    knots = (0., 0., .125, .25, .375, .5, .625, .75, 1., 1.)
    loop_edge = CurveOnSurface(surface, NurbsCurve(1, knots, ((.125, .125), (.5, .125), (.5, .375), (.25, .375),
                                                              (.25, .25), (.625, .25), (.625, .125), (.875, .125))))
    curves = [loop_edge, line(surface, (.875, .125), (.875, .875)), line(surface, (.875, .875), (.125, .875)),
              line(surface, (.125, .875), (.125, .125))]
    model = GeometryModel()
    with pytest.raises(GeometryError, match="not certified simple"):
        model.add_face(add_edges(model, curves), surface=surface)


def test_backward_edges_and_small_faces_are_accepted():
    surface = unit_plane()
    model = GeometryModel()
    corners = [(.25, .25), (.75, .25), (.75, .75), (.25, .75)]
    a, b, c, d = (model.add_point(x, y, 0.) for x, y in corners)
    bottom = model.add_curve(a, b, line(surface, corners[0], corners[1]))
    right = model.add_curve(c, b, line(surface, corners[2], corners[1]))           # stored backward
    top = model.add_curve(c, d, line(surface, corners[2], corners[3]))
    left = model.add_curve(a, d, line(surface, corners[0], corners[3]))            # stored backward
    model.add_face([bottom, right, top, left], surface=surface)
    side = 2. ** -24
    small = [(.25, .25), (.25 + side, .25), (.25 + side, .25 + side), (.25, .25 + side)]
    model2 = GeometryModel()
    model2.add_face(add_edges(model2, [line(surface, p, q) for p, q in zip(small, small[1:] + small[:1])]),
                    surface=surface)


def test_polyline_bounds_include_rounded_chord_ends():
    from anygeometry.nurbs_domain import certified_chart_polyline
    scale = 2. ** 60
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., scale, 0.)), ((scale, 0., 0.), (scale, scale, 0.))))
    pcurve = NurbsCurve(1, (0., 0., 1., 1.), ((0., .5), (1., .5)), (1., 2.), (.5, 1.))
    polyline, bound = certified_chart_polyline(surface, pcurve, 1e3)
    from fractions import Fraction
    overhang = abs(Fraction(float(polyline[0][0])) - pcurve.exact_point(0)[0]) * Fraction(scale)
    assert bound >= float(overhang) > 0


# ------------------------------------------------------------------ review A3b round 3


def test_an_edge_revisiting_a_point_is_not_simple():
    from anygeometry import GeometryError
    from anygeometry.nurbs_audit import certify_simple
    surface = unit_plane()
    revisit = NurbsCurve(1, (0., 0., .25, .3125, .375, .4375, .5, 1., 1.),
                         ((.125, .125), (.5, .25), (.75, .5), (.5, .75), (.25, .5), (.5, .25), (.5, .125)))
    assert certify_simple(revisit) is None
    trimmed_line = NurbsCurve(1, (0., 0., 1., 1.), ((.125, .125), (.875, .875)), domain=(.1, .9))
    assert certify_simple(trimmed_line) is True
    curves = [CurveOnSurface(surface, revisit), line(surface, (.5, .125), (.5, 0.)), line(surface, (.5, 0.), (.125, 0.)),
              line(surface, (.125, 0.), (.125, .125))]
    model = GeometryModel()
    with pytest.raises(GeometryError, match="not certified simple"):
        model.add_face(add_edges(model, curves), surface=surface)


def test_zero_area_is_relative_to_the_chart():
    scale = 2. ** 60
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., scale, 0.)), ((scale, 0., 0.), (scale, scale, 0.))))
    side = 2. ** -40
    square = [(.25, .25), (.25 + side, .25), (.25 + side, .25 + side), (.25, .25 + side)]
    model = GeometryModel()
    model.add_face(add_edges(model, [line(surface, p, q) for p, q in zip(square, square[1:] + square[:1])]),
                   surface=surface)
