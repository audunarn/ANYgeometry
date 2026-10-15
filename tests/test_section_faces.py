"""Plane faces trimmed by lines and plane x NURBS section curves (generality slice B1-P4b).

Reference: the section of a half cylinder (radius 2) by an oblique plane is a half ellipse
whose plan view is the half disc of radius 2; its area in the plane is the plan area times
``|n| / |n_z|`` for the plane normal ``n``.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import Cylinder, GeometryError, GeometryModel, Plane
from anygeometry.nurbs import cylinder_to_nurbs
from anygeometry.section_faces import loop_area
from anygeometry.support_curves import PlaneNurbsCurve

OBLIQUE = Plane((0., 0., 1.5), (1., 0., .3), (0., 1., .2))          # z = 1.5 + .3 x + .2 y
SCALE = math.sqrt(1 + .3 ** 2 + .2 ** 2)


def z(x, y):
    return 1.5 + .3 * x + .2 * y


def half_section(plane=OBLIQUE):
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    return PlaneNurbsCurve.from_plane(plane, surface, 0)                 # (2, 0) -> (0, 2) -> (-2, 0)


def deck(curve=None, *, middle=(0., 0.), surface=OBLIQUE, holes=()):
    """A face bounded by the section curve and the diameter (split at ``middle``)."""
    curve = curve or half_section()
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    mid = model.add_point(middle[0], middle[1], z(*middle))
    edges = [model.add_curve(start, end, curve), model.add_line(end, mid), model.add_line(mid, start)]
    face = model.add_face(edges, surface=surface)
    return model, face


def test_a_half_ellipse_deck_validates_and_has_the_analytic_area():
    model, face = deck()
    from anygeometry.section_faces import validate_plane_section_face
    assert validate_plane_section_face(model, face) == []          # the certified route, not sampling
    assert not model.validate_topology()
    area, bound = loop_area(model, model.faces[face].loop, OBLIQUE)
    expected = 2 * math.pi * SCALE                                      # half disc of radius 2, tilted
    assert abs(abs(area) - expected) <= 1e-9 * expected
    assert bound < .1 * expected


def test_a_line_crossing_the_curve_is_refused():
    # the closing lines pass outside the curve: (2,0) -> (0, 3) -> (-2, 0) crosses the half ellipse
    with pytest.raises(GeometryError, match="self-intersects|not certified"):
        deck(middle=(0., 3.))


def test_a_section_curve_of_another_plane_is_refused():
    other = Plane((0., 0., 1.6), (1., 0., .3), (0., 1., .2))
    curve = half_section(other)
    with pytest.raises(GeometryError, match="another plane"):
        deck(curve, surface=OBLIQUE)


def with_hole(model, face, corners):
    """Give the face a straight hole loop (no public straight-hole API; the transaction validates)."""
    from dataclasses import replace
    from anygeometry.entities import OrientedEdge
    points = [model.add_point(x, y, z(x, y)) for x, y in corners]
    edges = [model.add_line(points[i], points[(i + 1) % len(points)]) for i in range(len(points))]
    with model.transaction():
        model._put_entity("face", replace(model.faces[face], holes=(tuple(OrientedEdge(e, True) for e in edges),)))


def test_a_square_hole_inside_the_deck():
    model, face = deck()
    with_hole(model, face, [(-.5, .5), (.5, .5), (.5, 1.), (-.5, 1.)])
    assert len(model.faces[face].holes) == 1


def test_a_hole_crossing_the_curve_is_refused():
    model, face = deck()
    with pytest.raises(GeometryError, match="intersect|not certified"):
        with_hole(model, face, [(-.5, 1.5), (.5, 1.5), (.5, 2.5), (-.5, 2.5)])        # straddles (0, 2)


def test_a_hole_outside_the_deck_is_refused():
    model, face = deck()
    with pytest.raises(GeometryError, match="not strictly inside|containment"):
        with_hole(model, face, [(-.5, 2.5), (.5, 2.5), (.5, 3.), (-.5, 3.)])


def test_the_area_bound_tightens_with_pieces():
    model, face = deck()
    loop = model.faces[face].loop
    coarse, fine = loop_area(model, loop, OBLIQUE)[1], loop_area(model, loop, OBLIQUE, 512)[1]
    assert fine < coarse / 8


# ------------------------------------------------------------------ review B1-P4b round 1

LEVEL = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))


def test_a_section_that_crosses_itself_is_refused():
    # the extruded cubic with plan controls (-1,0) (2,2) (-2,2) (1,0) loops through itself at (0, 6/7)
    from anygeometry.nurbs import NurbsSurface
    plan = ((-1., 0.), (2., 2.), (-2., 2.), (1., 0.))
    surface = NurbsSurface(3, 1, (0.,) * 4 + (1.,) * 4, (0., 0., 1., 1.),
                           tuple(((x, y, 0.), (x, y, 1.)) for x, y in plan))
    curve = PlaneNurbsCurve.from_plane(LEVEL, surface, 0)
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    middle = model.add_point(0., 0., .5)
    edges = [model.add_curve(start, end, curve), model.add_line(end, middle), model.add_line(middle, start)]
    with pytest.raises(GeometryError, match="not certified simple"):
        model.add_face(edges, surface=LEVEL)


def level_half():
    surface = cylinder_to_nurbs(Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 2., 3., 0., math.pi))
    return PlaneNurbsCurve.from_plane(LEVEL, surface, 0)


def test_a_retraced_excursion_is_refused():
    from anygeometry.entities import OrientedEdge
    curve = level_half()
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    top = model.add_point(*curve.subcurve(0., .5).evaluate(1.))
    inner = model.add_point(0., 1., .5)
    first = model.add_curve(start, top, curve.subcurve(0., .5))
    out, back = model.add_line(top, inner), model.add_line(inner, top)
    second = model.add_curve(top, end, curve.subcurve(.5, 1.))
    diameter = model.add_line(end, start)
    loop = [OrientedEdge(e, True) for e in (first, out, back, second, diameter)]
    with pytest.raises(GeometryError, match="more than once|self-intersects"):
        model.add_face_from_loop(loop, surface=LEVEL)


def test_areas_use_local_coordinates():
    far = Plane((1e8, 1e8, .5), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(far, level_half().surface, 0)
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    mid = model.add_point(0., 0., .5)
    edges = [model.add_curve(start, end, curve), model.add_line(end, mid), model.add_line(mid, start)]
    face = model.add_face(edges, surface=far)
    area, bound = loop_area(model, model.faces[face].loop, far)
    assert abs(abs(area) - 2 * math.pi) <= 1e-9 and bound < .1 * 2 * math.pi


def test_small_holes_and_small_loops_are_judged_at_their_own_scale():
    curve = level_half()
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    mid = model.add_point(0., 0., .5)
    edges = [model.add_curve(start, end, curve), model.add_line(end, mid), model.add_line(mid, start)]
    face = model.add_face(edges, surface=LEVEL)
    from dataclasses import replace
    from anygeometry.entities import OrientedEdge
    side = 3e-7
    corners = [(0., 1.), (side, 1.), (side, 1. + side), (0., 1. + side)]
    points = [model.add_point(x, y, .5) for x, y in corners]
    hole = [model.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    with model.transaction():
        model._put_entity("face", replace(model.faces[face], holes=(tuple(OrientedEdge(e, True) for e in hole),)))
    assert len(model.faces[face].holes) == 1


# ------------------------------------------------------------------ review B1-P4b round 2


def extruded(x_power, y_power):
    """The extrusion over z in [0, 1] of the cubic plan curve with power-basis coefficients (low first)."""
    from anygeometry.nurbs import NurbsSurface
    to_bernstein = np.array([[math.comb(j, i) / math.comb(3, i) if j >= i else 0. for j in range(4)]
                             for i in range(4)])                     # c_j = sum_{i<=j} C(j,i)/C(3,i) a_i
    xs = np.pad(np.asarray(x_power, dtype=float), (0, 4 - len(x_power))) @ to_bernstein
    ys = np.pad(np.asarray(y_power, dtype=float), (0, 4 - len(y_power))) @ to_bernstein
    return NurbsSurface(3, 1, (0.,) * 4 + (1.,) * 4, (0., 0., 1., 1.),
                        tuple(((float(x), float(y), 0.), (float(x), float(y), 1.)) for x, y in zip(xs, ys)))


def poly(*roots_and_scale):
    *roots, scale = roots_and_scale
    return list(np.polynomial.polynomial.polyfromroots(roots) * scale)


def test_a_self_crossing_inside_one_piece_is_refused():
    # x = (t-.1)(t-.3)(t-.2), y = (t-.1)(t-.3): crosses itself at t = .1, .3 (inside the first half)
    curve = PlaneNurbsCurve.from_plane(LEVEL, extruded(poly(.1, .3, .2, 1.), poly(.1, .3, 1.)), 0)
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    corner = model.add_point(-1., 2., .5)
    edges = [model.add_curve(start, end, curve), model.add_line(end, corner), model.add_line(corner, start)]
    with pytest.raises(GeometryError, match="not certified simple"):
        model.add_face(edges, surface=LEVEL)


def test_an_excursion_before_the_joint_is_refused():
    # C+-(t) = (+-[t(t-.25) - 1e-11], t(t-.25)(t-.24) - 1e-10 t): they cross near (-.0113, .00055)
    def section(sign):
        x = [sign * value for value in (-1e-11, -.25, 1.)]
        y = [0., .25 * .24 - 1e-10, -.49, 1.]
        return PlaneNurbsCurve.from_plane(LEVEL, extruded(x, y), 0)
    plus, minus = section(1.), section(-1.)
    model = GeometryModel()
    origin = model.add_point(0., 0., .5)
    a, b = model.add_point(*plus.evaluate(1.)), model.add_point(*minus.evaluate(1.))
    edges = [model.add_curve(origin, a, plus), model.add_line(a, b), model.add_curve(b, origin, minus.subcurve(1., 0.))]
    with pytest.raises(GeometryError, match="not certified simple"):
        model.add_face(edges, surface=LEVEL)


def test_acute_corners_validate():
    # straight world sections (t, +-2e-7 t): a sliver of width 4e-7 at x = 1
    def section(sign):
        return PlaneNurbsCurve.from_plane(LEVEL, extruded([0., 1.], [0., sign * 2e-7]), 0)
    upper, lower = section(1.), section(-1.)
    model = GeometryModel()
    origin = model.add_point(0., 0., .5)
    a, b = model.add_point(*upper.evaluate(1.)), model.add_point(*lower.evaluate(1.))
    model.add_face([model.add_curve(origin, a, upper), model.add_line(a, b),
                    model.add_curve(b, origin, lower.subcurve(1., 0.))], surface=LEVEL)


def test_vertices_within_the_surface_residual_validate():
    curve = level_half()
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    mid = model.add_point(0., 0., .5 + 5e-9)
    model.add_face([model.add_curve(start, end, curve), model.add_line(end, mid), model.add_line(mid, start)],
                   surface=LEVEL)


def test_far_plane_origins_validate():
    far = Plane((1e17, 1e17, .5), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(far, level_half().surface, 0)
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    mid = model.add_point(0., 0., .5)
    model.add_face([model.add_curve(start, end, curve), model.add_line(end, mid), model.add_line(mid, start)],
                   surface=far)


# ------------------------------------------------------------------ review B1-P4b round 3


def test_faces_far_from_the_origin_validate():
    # S(u, v) = (1e6 + u, 0, v): the section at z = .5 is the segment (1e6 + t, 0, .5)
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((1e6, 0., 0.), (1e6, 0., 1.)), ((1e6 + 1, 0., 0.), (1e6 + 1, 0., 1.))))
    plane = Plane((0., 0., .5), (1., 0., 0.), (0., 1., 0.))
    curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
    model = GeometryModel()
    start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
    corner = model.add_point(1e6, 1., .5)
    model.add_face([model.add_curve(start, end, curve), model.add_line(end, corner), model.add_line(corner, start)],
                   surface=plane)


def test_far_origins_of_oblique_planes_validate():
    # S(u, v) = (u, -2u, v) cut by x + y + z = 0, the plane given with an origin far along it
    from anygeometry.nurbs import NurbsSurface
    surface = NurbsSurface(1, 1, (0., 0., 1., 1.), (0., 0., 1., 1.),
                           (((0., 0., 0.), (0., 0., 1.)), ((1., -2., 0.), (1., -2., 1.))))
    for origin in ((0., 0., 0.), (1e17, -1e17, 0.)):
        plane = Plane(origin, (1., -1., 0.), (1., 0., -1.))
        curve = PlaneNurbsCurve.from_plane(plane, surface, 0)
        model = GeometryModel()
        start, end = model.add_point(*curve.evaluate(0.)), model.add_point(*curve.evaluate(1.))
        corner = model.add_point(0., -.5, .5)
        model.add_face([model.add_curve(start, end, curve), model.add_line(end, corner),
                        model.add_line(corner, start)], surface=plane)
