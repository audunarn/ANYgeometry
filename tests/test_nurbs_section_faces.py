"""NURBS faces trimmed by chart segments and plane x NURBS section curves (generality slice B1-P4c).

The B1 bow (a rational half-bow of degree (2, 1), as B1-pre) has z = 4 v, so the deck plane z = 3
cuts it along the iso-line v = 3/4: the trimmed face below the deck is the chart rectangle
[0, 1] x [0, 3/4] (chart area 3/4 exactly).
"""
from __future__ import annotations

import math

import numpy as np

import pytest

from anygeometry import GeometryError, GeometryModel, Plane
from anygeometry.nurbs import CurveOnSurface, NurbsCurve, NurbsSurface
from anygeometry.nurbs_section_faces import chart_loop_area, validate_nurbs_section_face
from anygeometry.support_curves import PlaneNurbsCurve

DECK = Plane((0., 0., 3.), (1., 0., 0.), (0., 1., 0.))


def bow_surface():
    w = math.cos(math.pi / 4)
    bottom = [(0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)]
    deck = [(0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)))


def segment(surface, start, end):
    return CurveOnSurface(surface, NurbsCurve(1, (0., 0., 1., 1.), (start, end)))


class Builder:
    """Chart-point vertices (exact surface points) and edges on one support."""

    def __init__(self, surface):
        self.surface, self.model, self.vertices = surface, GeometryModel(), {}

    def vertex(self, chart):
        if chart not in self.vertices:
            self.vertices[chart] = self.model.add_point(*map(float, self.surface.exact_point(*chart)))
        return self.vertices[chart]

    def segment(self, start, end):
        return self.model.add_curve(self.vertex(start), self.vertex(end), segment(self.surface, start, end))

    def section(self, curve, start, end):
        return self.model.add_curve(self.vertex(start), self.vertex(end), curve)


def deck_edge(surface):
    curve = PlaneNurbsCurve.from_plane(DECK, surface, 0)
    u_start = round(float(curve.chart(0.)[0]))                     # the path runs u = 0 -> 1 or 1 -> 0
    return curve if u_start == 1 else curve.subcurve(1., 0.)        # oriented (1, 3/4) -> (0, 3/4)


def test_the_bow_below_the_deck_validates_with_chart_area_three_quarters():
    surface = bow_surface()
    build = Builder(surface)
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(deck_edge(surface), (1., .75), (0., .75)), build.segment((0., .75), (0., 0.))]
    face = build.model.add_face(edges, surface=surface)
    assert validate_nurbs_section_face(build.model, face) == []
    area, bound = chart_loop_area(build.model, build.model.faces[face].loop, surface)
    assert abs(abs(area) - .75) <= 1e-12 and bound < .05


def test_a_segment_crossing_the_deck_edge_is_refused():
    surface = bow_surface()
    build = Builder(surface)
    edges = [build.section(deck_edge(surface), (1., .75), (0., .75)), build.segment((0., .75), (.5, .9)),
             build.segment((.5, .9), (.5, .6)), build.segment((.5, .6), (1., .75))]
    with pytest.raises(GeometryError, match="self-intersects|not certified"):
        build.model.add_face(edges, surface=surface)


def test_a_section_of_another_surface_is_refused():
    surface = bow_surface()
    other = NurbsSurface(surface.degree_u, surface.degree_v, surface.knots_u, surface.knots_v,
                         tuple(tuple((x, y, z * 1.01) for x, y, z in row) for row in surface.controls),
                         surface.weights)
    build = Builder(surface)
    foreign = PlaneNurbsCurve.from_plane(DECK, other, 0)
    a, b = build.model.add_point(*foreign.evaluate(0.)), build.model.add_point(*foreign.evaluate(1.))
    c = build.model.add_point(3., 3., 3.)
    edge = build.model.add_curve(a, b, foreign)
    with pytest.raises(GeometryError, match="another surface"):
        build.model.add_face([edge, build.model.add_line(b, c), build.model.add_line(c, a)], surface=surface)


def bow_below_deck():
    surface = bow_surface()
    build = Builder(surface)
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(deck_edge(surface), (1., .75), (0., .75)), build.segment((0., .75), (0., 0.))]
    return surface, build, build.model.add_face(edges, surface=surface)


def test_curved_pcurves_next_to_sections_are_refused():
    surface = bow_surface()
    build = Builder(surface)
    curved = CurveOnSurface(surface, NurbsCurve(2, (0., 0., 0., 1., 1., 1.), ((0., .75), (-.0, .3), (0., 0.))))
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(deck_edge(surface), (1., .75), (0., .75)),
             build.model.add_curve(build.vertex((0., .75)), build.vertex((0., 0.)), curved)]
    with pytest.raises(GeometryError, match="straight chart segments"):
        build.model.add_face(edges, surface=surface)


def with_hole(surface, build, face, corners):
    from dataclasses import replace
    from anygeometry.entities import OrientedEdge
    edges = [build.segment(corners[i], corners[(i + 1) % len(corners)]) for i in range(len(corners))]
    with build.model.transaction():
        build.model._put_entity("face", replace(build.model.faces[face],
                                                holes=(tuple(OrientedEdge(e, True) for e in edges),)))


def test_a_hole_inside_validates():
    surface, build, face = bow_below_deck()
    with_hole(surface, build, face, [(.4, .3), (.6, .3), (.6, .5), (.4, .5)])
    assert len(build.model.faces[face].holes) == 1


def test_a_hole_crossing_the_deck_edge_is_refused():
    surface, build, face = bow_below_deck()
    with pytest.raises(GeometryError, match="intersect|not certified"):
        with_hole(surface, build, face, [(.4, .6), (.6, .6), (.6, .9), (.4, .9)])


def test_a_hole_above_the_deck_is_refused():
    surface, build, face = bow_below_deck()
    with pytest.raises(GeometryError, match="not strictly inside|containment"):
        with_hole(surface, build, face, [(.4, .8), (.6, .8), (.6, .9), (.4, .9)])


def test_the_trimmed_bow_round_trips():
    from anygeometry import from_dict, to_dict
    _surface, build, face = bow_below_deck()
    document = to_dict(build.model)
    restored = from_dict(document)
    assert to_dict(restored) == document and restored.faces[face].loop == build.model.faces[face].loop


# ------------------------------------------------------------------ review B1-P4c round 1


def test_a_hole_sharing_the_outer_boundary_is_refused():
    from dataclasses import replace
    from anygeometry.entities import OrientedEdge
    surface = bow_surface()
    build = Builder(surface)
    outer = [build.segment((0., 0.), (.4, 0.)), build.segment((.4, 0.), (.6, 0.)), build.segment((.6, 0.), (1., 0.)),
             build.segment((1., 0.), (1., .75)), build.section(deck_edge(surface), (1., .75), (0., .75)),
             build.segment((0., .75), (0., 0.))]
    face = build.model.add_face(outer, surface=surface)
    model = build.model
    corners = [(.4, .2), (.4, 0.), (.6, 0.), (.6, .2)]
    points = [model.add_point(*map(float, surface.exact_point(*c))) for c in corners]       # distinct vertex IDs
    hole = [model.add_curve(points[i], points[(i + 1) % 4], segment(surface, corners[i], corners[(i + 1) % 4]))
            for i in range(4)]
    with pytest.raises(GeometryError, match="intersect"):
        with model.transaction():
            model._put_entity("face", replace(model.faces[face], holes=(tuple(OrientedEdge(e, True) for e in hole),)))


def test_pcurve_joints_stay_exact_next_to_sections():
    surface = bow_surface()
    build = Builder(surface)
    model = build.model
    corner = build.vertex((1., 0.))
    right = model.add_curve(corner, build.vertex((1., .75)), segment(surface, (1., 1e-12), (1., .75)))
    edges = [build.segment((0., 0.), (1., 0.)), right,
             build.section(deck_edge(surface), (1., .75), (0., .75)), build.segment((0., .75), (0., 0.))]
    with pytest.raises(GeometryError, match="not continuous"):
        model.add_face(edges, surface=surface)


def test_a_collinear_segment_continuing_a_section_validates():
    surface = bow_surface()
    build = Builder(surface)
    half = deck_edge(surface).subcurve(0., .5)                       # (1, 3/4) -> (1/2, 3/4)
    middle = tuple(float(value) for value in half.chart(1.))
    edges = [build.segment((0., 0.), (1., 0.)), build.segment((1., 0.), (1., .75)),
             build.section(half, (1., .75), middle), build.segment(middle, (0., .75)),
             build.segment((0., .75), (0., 0.))]
    face = build.model.add_face(edges, surface=surface)
    assert validate_nurbs_section_face(build.model, face) == []


def test_certified_trim_polylines_include_declared_section_edges():
    _surface, build, face = bow_below_deck()
    loops, bound = build.model.trim_loops_uv_certified(face, 1e-6)
    assert len(loops) == 1 and bound <= 1e-6
    assert np.max(loops[0][:, 1]) <= .75 + 1e-12


def test_display_trim_loops_of_nurbs_faces_are_chart_samples():
    _surface, build, face = bow_below_deck()
    (loop,) = build.model.face_trim_loops_uv(face)
    assert loop.shape[1] == 2 and abs(loop[:, 1].max() - .75) <= 1e-12 and abs(loop[:, 1].min()) <= 1e-12


# ------------------------------------------------------------------ B1-P5a: corners between two section curves


def b1_bow():
    w = math.cos(math.pi / 4)
    bottom = [(0., 6., 0.), (5.4, 6., 0.), (5.4, 0., 0.)]
    deck = [(0., 7.5, 4.), (6., 7.5, 4.), (6., 0., 4.)]
    return NurbsSurface(2, 1, (0., 0., 0., 1., 1., 1.), (0., 0., 1., 1.),
                        tuple((b, d) for b, d in zip(bottom, deck)), ((1., 1.), (w, w), (1., 1.)), u_range=(0., .9))


def section_parameter(curve, normal, offset):
    from anygeometry.branch_predicates import section_roots
    (root,) = section_roots(curve, normal, offset, max_depth=90, refine=80)
    return .5 * (root.t_lo + root.t_hi)


def test_a_bow_piece_bounded_by_four_sections_validates():
    surface = b1_bow()
    planes = {"deck": (Plane((0., 0., 3.), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), 3),
              "bottom": (Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.)), (0, 0, 1), .5),
              "a": (Plane((1.5, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), 1.5),
              "b": (Plane((3.5, 0., 0.), (0., 1., 0.), (0., 0., 1.)), (1, 0, 0), 3.5)}
    curves = {key: PlaneNurbsCurve.from_plane(plane, surface, 0) for key, (plane, _n, _d) in planes.items()}
    t = {(c, p): section_parameter(curves[c], planes[p][1], planes[p][2])
         for c, p in (("deck", "a"), ("deck", "b"), ("bottom", "a"), ("bottom", "b"),
                      ("a", "deck"), ("a", "bottom"), ("b", "deck"), ("b", "bottom"))}
    model = GeometryModel()
    corner = {frozenset(key): model.add_point(*curves[key[0]].evaluate(t[key]))
              for key in (("deck", "a"), ("deck", "b"), ("bottom", "a"), ("bottom", "b"))}

    def edge(curve, start, end):
        return model.add_curve(corner[frozenset((curve, start))], corner[frozenset((curve, end))],
                               curves[curve].subcurve(t[(curve, start)], t[(curve, end)]))
    face = model.add_face([edge("bottom", "a", "b"), edge("b", "bottom", "deck"), edge("deck", "b", "a"),
                           edge("a", "deck", "bottom")], surface=surface)
    assert validate_nurbs_section_face(model, face) == []


def test_a_section_crossing_the_deck_edge_again_from_a_shared_corner_is_not_certified():
    from anygeometry.nurbs_section_faces import _Chart, _pair, _chart_tolerance
    surface = b1_bow()
    deck_plane = Plane((0., 0., 3.), (1., 0., 0.), (0., 1., 0.))
    deck = PlaneNurbsCurve.from_plane(deck_plane, surface, 0)
    t_a = section_parameter(deck, (1, 0, 0), 1.5)
    t_b = section_parameter(deck, (1, 0, 0), 3.5)
    c1, c2 = deck.evaluate(t_a), deck.evaluate(.5 * (t_a + t_b))
    below = np.asarray(surface.evaluate(.4, .3), dtype=float)
    slanted = Plane(tuple(c1), tuple(c2 - c1), tuple(below - c1))
    other = PlaneNurbsCurve.from_plane(slanted, surface, 0)
    from anygeometry.branch_predicates import section_roots
    crossings = sorted(.5 * (r.t_lo + r.t_hi) for r in section_roots(other, (0, 0, 1), 3, max_depth=90, refine=80))
    assert len(crossings) == 2                                  # the slanted section meets the deck at c1 and c2
    near_c1 = min(crossings, key=lambda t: np.linalg.norm(other.evaluate(t) - c1))
    beyond = 1.0 if near_c1 == crossings[0] else 0.0            # run from c1 through c2 to the path's end
    model = GeometryModel()
    start = model.add_point(*c1)
    first = model.add_curve(start, model.add_point(*deck.evaluate(t_b)), deck.subcurve(t_a, t_b))
    piece = other.subcurve(near_c1, beyond)
    second = model.add_curve(start, model.add_point(*piece.evaluate(1.)), piece)
    a, b = model.edges[first], model.edges[second]
    verdict = _pair(model, a, b, _Chart(surface), [], {a.start, a.end} & {b.start, b.end},
                    _chart_tolerance(surface, 1e-9))
    assert verdict is not True
