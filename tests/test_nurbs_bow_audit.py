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
