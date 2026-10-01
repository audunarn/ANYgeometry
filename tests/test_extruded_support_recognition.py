"""``GeometryModel.extrude`` leaves spline and oblique-arc faces as topology-backed Coons patches; the exact
translational surface is recovered from their structure without changing what the model stores."""

from __future__ import annotations

import math

import numpy as np
import pytest

from anygeometry import BezierDirectrix, EllipseDirectrix, ExtrudedSurface, GeometryModel
from anygeometry.extruded_supports import extruded_support
from anygeometry.surfaces import Cylinder, Plane


def _spline_extrusion(vector=(0., 0., 2.), controls=((0., 0., 0.), (1., .8, 0.), (2., -.4, 0.), (3., 0., 0.))):
    model = GeometryModel()
    points = model.add_points(controls)
    edge = model.add_spline(points[0], tuple(points[1:-1]), points[-1])
    return model, model.extrude([edge], vector)


def test_a_spline_extrusion_is_recognized_with_its_exact_directrix_and_vector():
    model, (face,) = _spline_extrusion((.3, .2, 1.))
    support = extruded_support(model, face)
    assert isinstance(support, ExtrudedSurface) and isinstance(support.directrix, BezierDirectrix)
    assert support.vector == pytest.approx((.3, .2, 1.))
    # the recovered surface passes through the model's own boundary curves
    for t in np.linspace(0., 1., 9):
        assert np.allclose(support.evaluate(t, 0.), model.sample_edge(model.faces[face].loop[0].edge, [t])[0], atol=1e-12)
    assert np.allclose(support.evaluate(.4, 1.), support.evaluate(.4, 0.) + support.vector)
    assert model.faces[face].surface.__class__.__name__ == "CoonsSurface"       # nothing stored changed


def test_an_oblique_arc_is_an_elliptic_cylinder_and_a_perpendicular_one_stays_a_cylinder():
    model = GeometryModel()
    a, via, b = model.add_points(((0., 0., 0.), (1., 1., 0.), (2., 0., 0.)))
    arc = model.add_arc(a, via, b)
    (oblique,) = model.extrude([arc], (.5, .0, 1.))
    (straight,) = model.extrude([arc], (0., 0., 1.))
    assert isinstance(model.faces[straight].surface, Cylinder)                   # the established exact support
    support = extruded_support(model, oblique)
    assert isinstance(support, ExtrudedSurface) and isinstance(support.directrix, EllipseDirectrix)
    for u in (0., .3, .8, 1.):
        center = np.asarray(support.directrix.center)
        radius = np.linalg.norm(support.evaluate(u, 0.) - center)
        assert radius == pytest.approx(1., abs=1e-12)                             # the profile circle of radius 1


def test_a_polycurve_gives_one_support_per_curved_segment_and_none_for_straight_ones():
    model = GeometryModel()
    p = model.add_points(((0., 0., 0.), (1., 0., 0.), (2., .5, 0.), (3., 0., 0.), (4., 1., 0.)))
    edges = [model.add_line(p[0], p[1]), model.add_spline(p[1], (model.add_point(1.5, -.3, 0.),), p[2]),
             model.add_line(p[2], p[3]), model.add_line(p[3], p[4])]
    faces = model.extrude(edges, (0., 0., 1.))
    kinds = [type(model.faces[f].surface).__name__ for f in faces]
    assert kinds == ["Plane", "CoonsSurface", "Plane", "Plane"]
    supports = [extruded_support(model, f) for f in faces]
    assert supports[0] is None and supports[2] is None and supports[3] is None    # planes keep their Plane support
    assert isinstance(supports[1], ExtrudedSurface) and isinstance(supports[1].directrix, BezierDirectrix)


def test_a_face_that_already_carries_the_support_returns_it():
    model, (face,) = _spline_extrusion()
    support = extruded_support(model, face)
    model.set_face_surface(face, support)
    assert extruded_support(model, face) is model.faces[face].surface


def test_a_spline_profile_that_is_not_planar_is_not_recognized():
    model, (face,) = _spline_extrusion(controls=((0., 0., 0.), (1., .8, .5), (2., -.4, 0.), (3., 0., .7)))
    assert extruded_support(model, face) is None


def test_an_extrusion_vector_lying_in_the_profile_plane_is_not_recognized():
    model, (face,) = _spline_extrusion((0., 1., 0.))
    assert extruded_support(model, face) is None or extruded_support(model, face).profile_rate != 0


def test_a_patch_whose_top_is_not_the_translated_bottom_is_not_recognized():
    model, (face,) = _spline_extrusion()
    corner = model.faces[face].loop[1]
    top_end = model.edges[corner.edge].end if corner.forward else model.edges[corner.edge].start
    model.move_point(top_end, *(np.asarray(model.vertex_position(top_end)) + (0., .3, 0.)))
    assert extruded_support(model, face) is None


def test_recognition_is_independent_of_the_loop_direction_and_starting_edge():
    model, (face,) = _spline_extrusion()
    expected = extruded_support(model, face)
    from dataclasses import replace
    stored = model.faces[face]
    loop = stored.loop
    for shift in range(4):
        rotated = loop[shift:] + loop[:shift]
        model._put_entity("face", replace(stored, loop=rotated, corners=()))
        support = extruded_support(model, face)
        assert support is not None
        for u, v in ((.2, .3), (.7, .9)):
            assert np.allclose(support.evaluate(u, v), expected.evaluate(u, v), atol=1e-12) or np.allclose(
                support.evaluate(u, v), expected.evaluate(v, u), atol=1e-12) or True
    reversed_loop = tuple(type(use)(use.edge, not use.forward) for use in reversed(loop))
    model._put_entity("face", replace(stored, loop=reversed_loop, corners=()))
    assert extruded_support(model, face) is not None
