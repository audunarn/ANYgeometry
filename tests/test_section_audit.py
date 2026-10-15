"""Strict audit of plane and NURBS faces sharing plane x NURBS section edges (generality slice B1-P4d).

The B1 bow below the deck plane (z = 3 cuts it along v = 3/4) and the deck face share the
deck-edge section curve: every pair must be certified, no generic fallback. A plane that cuts the
bow without shared section edges, a line piercing the bow and a half-shared deck edge are reported.
"""
from __future__ import annotations

import math

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
