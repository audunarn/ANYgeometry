"""Strict audit of plane and NURBS faces sharing plane x NURBS section edges (generality slice B1-P4d).

The B1 bow below the deck plane (z = 3 cuts it along v = 3/4) and the deck face share the
deck-edge section curve: every pair must be certified, no generic fallback. A plane that cuts the
bow without shared section edges, a line piercing the bow and a half-shared deck edge are reported.
"""
from __future__ import annotations

import math

import numpy as np

from anygeometry import GeometryModel, Plane
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.support_curves import PlaneNurbsCurve

DECK = Plane((0., 0., 3.), (1., 0., 0.), (0., 1., 0.))


def bow_surface():
    w = math.cos(math.pi / 4)
    bottom = [(0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)]
    deck = [(0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)))


class Builder:
    def __init__(self, surface):
        self.surface, self.model, self.vertices = surface, GeometryModel(), {}

    def vertex(self, chart):
        if chart not in self.vertices:
            self.vertices[chart] = self.model.add_point(*map(float, self.surface.exact_point(*chart)))
        return self.vertices[chart]

    def segment(self, start, end):
        curve = CurveOnSurface(self.surface, NurbsCurve(1, (0., 0., 1., 1.), (start, end)))
        return self.model.add_curve(self.vertex(start), self.vertex(end), curve)

    def section(self, curve, start, end):
        return self.model.add_curve(self.vertex(start), self.vertex(end), curve)


def deck_edge(surface):
    curve = PlaneNurbsCurve.from_plane(DECK, surface, 0)
    return curve if round(float(curve.chart(0.)[0])) == 1 else curve.subcurve(1., 0.)


def bow_and_deck():
    surface = bow_surface()
    build = Builder(surface)
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(deck_edge(surface), (1., .75), (0., .75)), build.segment((0., .75), (0., 0.))]
    model = build.model
    bow = model.add_face(edges, surface=surface)
    a, b = build.vertex((1., .75)), build.vertex((0., .75))
    centre = model.add_point(0., 0., 3.)
    deck = model.add_face([edges[2], model.add_line(b, centre), model.add_line(centre, a)], surface=DECK)
    return model, bow, deck


def reasons(report):
    return sorted({value for issue in report.issues for key, value in issue.details if key == "reason"})


def test_the_bow_and_the_deck_audit_clean():
    model, _bow, _deck = bow_and_deck()
    report = model.strict_audit()
    assert report.clean, [(issue.code, issue.details) for issue in report.issues]


def test_a_plane_face_cutting_the_bow_without_shared_edges_is_reported():
    model, _bow, _deck = bow_and_deck()
    level = Plane((0., 0., 2.), (1., 0., 0.), (0., 1., 0.))
    corners = [model.add_point(x, y, 2.) for x, y in ((1., 1.), (7., 1.), (7., 9.), (1., 9.))]
    model.add_face([model.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)], surface=level)
    report = model.strict_audit()
    assert not report.clean and '"plane_nurbs_faces_not_certified"' in reasons(report)


def test_a_line_piercing_the_bow_is_reported():
    model, _bow, _deck = bow_and_deck()
    a, b = model.add_point(3., -1., 1.5), model.add_point(3., 10., 1.5)
    model.add_line(a, b)
    report = model.strict_audit()
    assert not report.clean


def test_a_half_shared_deck_edge_is_reported():
    surface = bow_surface()
    build = Builder(surface)
    model = build.model
    full = deck_edge(surface)
    first, second = full.subcurve(0., .5), full.subcurve(.5, 1.)
    middle = tuple(float(value) for value in first.chart(1.))
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(first, (1., .75), middle), build.section(second, middle, (0., .75)),
             build.segment((0., .75), (0., 0.))]
    model.add_face(edges, surface=surface)
    a, m = build.vertex((1., .75)), build.vertex(middle)
    centre = model.add_point(model.vertices[m].position[0], 0., 3.)
    model.add_face([edges[2], model.add_line(m, centre), model.add_line(centre, a)], surface=DECK)
    report = model.strict_audit()
    assert not report.clean and '"plane_nurbs_faces_not_certified"' in reasons(report)


# ------------------------------------------------------------------ review B1-P4d round 1


def bernstein(power):
    """Bernstein coefficients (degree len - 1) of a polynomial in u with power coefficients (low first)."""
    n = len(power) - 1
    return [sum(math.comb(j, i) / math.comb(n, i) * power[i] for i in range(j + 1)) for j in range(n + 1)]


def chart_face(model, surface, corners):
    from anygeometry.nurbs import CurveOnSurface, NurbsCurve
    vertices = [model.add_point(*map(float, surface.exact_point(*c))) for c in corners]
    edges = [model.add_curve(vertices[i], vertices[(i + 1) % len(corners)],
                             CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (corners[i], corners[(i + 1) % len(corners)]))))
             for i in range(len(corners))]
    return model.add_face(edges, surface=surface)


def test_two_pieces_of_a_self_crossing_section_are_not_cleared():
    # S(u, v) = (9 t^2/8 - 1/32, 33 t^3/32 - 9 t/128, v), t = u - 1/2: the section v = 1/2 crosses itself
    t = np.polynomial.polynomial
    shift = [-.5, 1.]
    x = t.polyadd(t.polymul(t.polypow(shift, 2), [9 / 8]), [-1 / 32])
    y = t.polysub(t.polymul(t.polypow(shift, 3), [33 / 32]), t.polymul(shift, [9 / 128]))
    xs, ys = bernstein(list(x) + [0.] * (4 - len(x))), bernstein(list(y))
    surface = NurbsSurface(3, 1, (0.,) * 4 + (1.,) * 4, (0., 0., 1., 1.),
                           tuple(((xi, yi, 0.), (xi, yi, 1.)) for xi, yi in zip(xs, ys)))
    curve = PlaneNurbsCurve.from_plane(Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)), surface, 0)
    model = GeometryModel()
    for lo, hi in ((7 / 32, 9 / 32), (23 / 32, 25 / 32)):
        piece = curve.subcurve(lo, hi)
        model.add_curve(model.add_point(*piece.evaluate(0.)), model.add_point(*piece.evaluate(1.)), piece)
    assert not model.strict_audit().clean


def test_a_plane_face_within_tolerance_of_a_bowl_is_reported():
    from anygeometry import TolerancePolicy
    b = bernstein([1., -4., 4.])                                      # (2u - 1)^2
    xs = bernstein([0., 1., 0.])
    surface = NurbsSurface(2, 2, (0., 0., 0., 1., 1., 1.), (0., 0., 0., 1., 1., 1.),
                           tuple(tuple((xs[i], xs[j], b[i] + b[j]) for j in range(3)) for i in range(3)))
    model = GeometryModel(tolerance=TolerancePolicy(length=1e-4))
    chart_face(model, surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    level = Plane((0., 0., -5e-5), (1., 0., 0.), (0., 1., 0.))
    corners = [model.add_point(x, y, -5e-5) for x, y in ((-1., -1.), (2., -1.), (2., 2.), (-1., 2.))]
    model.add_face([model.add_line(corners[i], corners[(i + 1) % 4]) for i in range(4)], surface=level)
    assert not model.strict_audit().clean


def test_vertices_use_the_surface_residual_tolerance():
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 1.), (1., 1., 1.))))
    model = GeometryModel()
    chart_face(model, surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    model.add_point(.5, .5, .5 + 5e-9)                                  # 3.5e-9 from the face: below 1e-8
    assert not model.strict_audit().clean


def test_section_work_follows_the_qualification_budget_and_is_reported():
    from anygeometry.predicates import IntersectionQualificationPolicy
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., 1., 0.)), ((1., 0., 1.), (1., 1., 1.))))
    model = GeometryModel()
    chart_face(model, surface, [(0., 0.), (1., 0.), (1., 1.), (0., 1.)])
    model.add_point(.5, .5, .6)                                         # needs a few boxes to clear
    tight = model.strict_audit(qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1,
                                                                             max_subdivision_depth=1))
    assert not tight.clean
    loose = model.strict_audit()
    assert loose.clean and loose.metrics.boxes_examined > 0
