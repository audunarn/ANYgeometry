"""Extruded polycurve faces through the exact intersection engine.

A spline or oblique-arc face made by ``GeometryModel.extrude`` is an exact translational surface. These
tests cover the pieces the engine needs (the chart of its material domain, its section by a plane, the
meeting of parallel extrusions) and whole models through ``plan_intersections`` / ``apply_intersections``:
topology validates, material area is conserved, every joint edge lies on both supports, split children keep
their exact support, and anything unsupported is refused with a typed error instead of approximated.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from anygeometry import (BezierDirectrix, ConnectionIntent, EllipseDirectrix, ExtrudedSurface, GeometryError,
                         GeometryModel, apply_intersections, from_dict, plan_intersections,
                         query_trimmed_surface_charts, to_dict)
from anygeometry.arrangement_geometry import LinePath
from anygeometry.extruded_pair_supports import extruded_pair_support, plane_extruded_support
from anygeometry.extruded_supports import extruded_support
from anygeometry.material_arrangement import ArrangementPath, MaterialDomain, arrange_material
from anygeometry.surfaces import Plane

CUBIC = ((0., 0., 0.), (1., .8, 0.), (2., -.4, 0.), (3., 0., 0.))


def _plane(point, normal, size=3.):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    u = np.cross(n, (0., 1., 0.)) if abs(n[1]) < .9 else np.cross(n, (1., 0., 0.))
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return Plane(np.asarray(point, float) - size * u - size * v, 2 * size * u, 2 * size * v)


def _spline_extrusion(vector=(0., 0., 2.), controls=CUBIC):
    model = GeometryModel()
    points = model.add_points(controls)
    edge = model.add_spline(points[0], tuple(points[1:-1]), points[-1])
    return model, list(model.extrude([edge], vector))


def _polycurve(vector=(0., 0., 2.)):
    model = GeometryModel()
    p = model.add_points(((0., 0., 0.), (1., 0., 0.), (2., .5, 0.), (3., 0., 0.), (4., 1., 0.)))
    edges = [model.add_line(p[0], p[1]), model.add_spline(p[1], (model.add_point(1.5, -.3, 0.),), p[2]),
             model.add_line(p[2], p[3]), model.add_line(p[3], p[4])]
    return model, list(model.extrude(edges, vector))


def _mixed(vector=(.4, 0., 1.5)):
    model = GeometryModel()
    p = model.add_points(((0., 0., 0.), (1., 0., 0.), (2., .8, 0.), (3., 0., 0.), (4., .4, 0.), (5., 0., 0.)))
    edges = [model.add_line(p[0], p[1]), model.add_arc(p[1], p[2], p[3]), model.add_spline(p[3], (p[4],), p[5])]
    return model, list(model.extrude(edges, vector))


def _plate(model, point, normal, size=3.):
    plane = _plane(point, normal, size)
    corners = [plane.evaluate(a, b) for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))]
    return model.add_plate(model.add_points(corners))


def _total_area(model):
    return sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)


def _residual(model):
    owners = collections.defaultdict(set)
    for face_id, face in model.faces.items():
        for oriented in (*face.loop, *(e for hole in face.holes for e in hole)):
            owners[oriented.edge].add(face_id)
    worst = 0.
    for edge_id, faces in owners.items():
        points = model.sample_edge(edge_id, np.linspace(0., 1., 25))
        for face_id in faces:
            surface = model.faces[face_id].surface
            if isinstance(surface, ExtrudedSurface):
                worst = max(worst, float(np.abs(surface.evaluate_many(surface.local_uv_many(points)) - points).max()))
            elif isinstance(surface, Plane):
                worst = max(worst, float(np.abs((points - surface.origin) @ surface.normal).max()))
    return worst


def _run(model, faces):
    handles = [model.handle("face", f) for f in faces]
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)


# ---------------------------------------------------------------- the material chart


def test_the_chart_of_an_extruded_face_has_exact_areas_and_tangents():
    model, (face,) = _spline_extrusion((.3, .2, 2.))
    domain = MaterialDomain.from_model(model, face)
    support = domain.support
    assert isinstance(support, ExtrudedSurface)
    assert domain.area_loop(domain.boundaries[0], 1e-13) == pytest.approx(1., abs=1e-12)     # the unit chart, counterclockwise
    nodes, weights = np.polynomial.legendre.leggauss(200)
    t = .5 * (nodes + 1)
    reference = .5 * float(weights @ np.linalg.norm(np.cross(support.directrix.derivative(t), support.vector), axis=1))
    assert domain.world_area_loop(domain.boundaries[0], 1e-12) == pytest.approx(reference, rel=1e-13)
    assert domain.area_jacobian >= reference            # an upper bound for tolerance scaling
    for path in domain.boundaries[0]:                   # the chart tangent is the finite-difference tangent of uv
        for t_value in (.2, .7):
            h = 1e-6
            numeric = (domain.uv(path.curve, t_value + h) - domain.uv(path.curve, t_value - h)) / (2 * h)
            assert np.allclose(domain.tangent(path.curve, t_value), numeric, atol=1e-6)


def test_a_generator_cut_splits_the_chart_into_cells_whose_areas_add_up():
    model, (face,) = _spline_extrusion()
    domain = MaterialDomain.from_model(model, face)
    support = domain.support
    cut = LinePath(tuple(support.evaluate(.4, -.1)), tuple(support.evaluate(.4, 1.1)))
    arrangement = arrange_material(domain, (ArrangementPath(cut, owners=(face, 99)),), tolerance=1e-9, area_tolerance=1e-12)
    areas = [domain.world_area_loop(tuple(ArrangementPath(arrangement.paths[e].curve if f else arrangement.paths[e].curve.subcurve(1., 0.))
                                          for e, f in cell.outer), 1e-12) for cell in arrangement.cells]
    assert len(areas) == 2 and sum(areas) == pytest.approx(domain.world_area_loop(domain.boundaries[0], 1e-12), rel=1e-13)
    assert domain.contains(LinePath(tuple(support.evaluate(.4, .5)), tuple(support.evaluate(.4, .5))), 0., 1e-9)
    assert not domain.contains(LinePath(tuple(support.evaluate(.5, 1.5)), tuple(support.evaluate(.5, 1.5))), 0., 1e-9)


# ---------------------------------------------------------------- plane sections


SECTIONS = {
    "bezier, horizontal": (ExtrudedSurface(BezierDirectrix(CUBIC), (.3, .2, 2.)), (1.5, .3, 1.), (0., 0., 1.)),
    "bezier, tilted": (ExtrudedSurface(BezierDirectrix(CUBIC), (.3, .2, 2.)), (1.5, .3, 1.), (0., .3, 1.)),
    "bezier, steep": (ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.)), (1.5, .3, .2), (.6, .1, 1.)),
    "ellipse, horizontal": (ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi), (.2, 0., 1.5)), (.5, .3, .7), (0., 0., 1.)),
    "ellipse, tilted": (ExtrudedSurface(EllipseDirectrix((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), 0., math.pi), (.2, 0., 1.5)), (.5, .3, .7), (0., .2, 1.)),
}


@pytest.mark.parametrize("name", SECTIONS)
def test_a_plane_section_lies_on_both_supports_inside_the_patch(name):
    surface, point, normal = SECTIONS[name]
    plane = _plane(point, normal)
    result = plane_extruded_support(plane, surface)
    assert result.curves and not result.segments
    for curve in result.curves:
        points = curve.evaluate(np.linspace(0., 1., 17))
        assert np.abs((points - plane.origin) @ plane.normal).max() < 1e-13
        uv = surface.local_uv_many(points)
        assert np.abs(surface.evaluate_many(uv) - points).max() < 1e-13
        assert uv.min() > -1e-9 and uv.max() < 1 + 1e-9                 # restricted to the patch


def test_a_bezier_section_is_a_bezier_of_the_same_degree():
    surface, point, normal = SECTIONS["bezier, horizontal"]
    (curve,) = plane_extruded_support(_plane(point, normal), surface).curves
    assert len(curve.controls) == len(CUBIC)


def test_a_plane_beyond_the_patch_has_no_section():
    surface, _point, normal = SECTIONS["bezier, horizontal"]
    assert not plane_extruded_support(_plane((1.5, .3, 9.), normal), surface).curves
    assert not plane_extruded_support(_plane((1.5, .3, -9.), normal), surface).curves


def test_a_plane_parallel_to_the_vector_meets_it_in_generators_and_a_tangent_one_in_one():
    parabola = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., 0., 0.))), (0., 0., 3.))
    crossing = plane_extruded_support(_plane((1., .25, 1.), (0., 1., 0.)), parabola)       # y = 1/4: two generators
    assert not crossing.curves and len(crossing.segments) == 2
    tangent = plane_extruded_support(_plane((1., .5, 1.), (0., 1., 0.)), parabola)          # y = 1/2 touches at the apex
    assert len(tangent.segments) == 1
    (start, end), = tangent.segments
    assert np.allclose(start, (1., .5, 0.)) and np.allclose(end, (1., .5, 3.))
    assert not plane_extruded_support(_plane((1., .75, 1.), (0., 1., 0.)), parabola).segments


# ---------------------------------------------------------------- parallel extrusions


def test_parallel_extrusions_meet_in_the_generators_through_their_crossings():
    a = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1.5, 0.), (2., -1., 0.), (3., .5, 0.))), (0., 0., 2.))
    b = ExtrudedSurface(BezierDirectrix(((0., .6, 0.), (1., -.4, 0.), (2., 1.2, 0.), (3., .2, 0.))), (0., 0., 2.))
    result = extruded_pair_support(a, b)
    assert len(result.segments) == 3 and not result.curves
    for start, end in result.segments:
        assert np.allclose(np.subtract(end, start), (0., 0., 2.))
        for point in (start, end):
            assert np.abs(a.evaluate_many(a.local_uv_many([point])) - point).max() < 1e-12
            assert np.abs(b.evaluate_many(b.local_uv_many([point])) - point).max() < 1e-12
    swapped = extruded_pair_support(b, a).segments                       # the same lines whichever surface comes first
    assert len(swapped) == 3
    for start, end in result.segments:
        assert any((np.allclose(start, s0, atol=1e-9) and np.allclose(end, s1, atol=1e-9))
                   or (np.allclose(start, s1, atol=1e-9) and np.allclose(end, s0, atol=1e-9)) for s0, s1 in swapped)


def test_the_shared_generator_is_clipped_to_both_patches():
    controls = ((0., 0., 0.), (1., 1.5, 0.), (2., -1., 0.), (3., .5, 0.))
    other = ((0., .6, 1.), (1., -.4, 1.), (2., 1.2, 1.), (3., .2, 1.))            # b's profile plane is one unit higher
    a = ExtrudedSurface(BezierDirectrix(controls), (0., 0., 2.))                    # a spans z in [0, 2]
    b = ExtrudedSurface(BezierDirectrix(other), (0., 0., 2.))                       # b spans z in [1, 3]
    result = extruded_pair_support(a, b)
    assert len(result.segments) == 3
    for start, end in result.segments:
        assert sorted((start[2], end[2])) == pytest.approx([1., 2.])               # only where both exist
    taller = ExtrudedSurface(BezierDirectrix(other), (0., 0., 2.), v_range=(.5, 1.5))      # b spans z in [2, 4]
    assert not extruded_pair_support(a, taller).segments                                    # they only touch at z = 2
    reversed_vector = ExtrudedSurface(BezierDirectrix(other), (0., 0., -2.))               # b spans z in [-1, 1]
    for start, end in extruded_pair_support(a, reversed_vector).segments:
        assert sorted((start[2], end[2])) == pytest.approx([0., 1.])


def test_extrusions_of_other_directions_and_overlapping_directrices_are_refused():
    a = ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.))
    with pytest.raises(GeometryError, match="non-parallel"):
        extruded_pair_support(a, ExtrudedSurface(BezierDirectrix(CUBIC), (0., .5, 2.)))
    with pytest.raises(GeometryError, match="ownership"):
        extruded_pair_support(a, ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 3.)))      # the same profile, taller
    assert extruded_pair_support(a, a.subpatch((.2, .8), (0., 1.))).coincident                 # patches of one surface


def test_a_cylinder_along_the_extrusion_meets_a_cubic_wall_in_generators_and_other_pairs_stay_unsupported():
    from anygeometry.surfaces import Cylinder
    a = ExtrudedSurface(BezierDirectrix(CUBIC), (0., 0., 2.))
    result = extruded_pair_support(a, Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 2., 0., math.tau))
    assert result.segments and not result.curves                         # exact: the rulings of the cylinder are the wall's
    for start, end in result.segments:
        assert np.allclose(np.asarray(end) - np.asarray(start), (0., 0., 2.))
    with pytest.raises(GeometryError, match="unsupported"):                  # a cubic is not a quadric
        extruded_pair_support(a, ExtrudedSurface(BezierDirectrix(CUBIC), (0., .5, 2.)))


# ---------------------------------------------------------------- whole models


def _tilted_spline_extrusion():
    """A profile plane that neither passes through the origin nor is axis-aligned (normal (0, .6, .8))."""
    n = np.array([0., .6, .8])
    e1 = np.cross(n, (1., 0., 0.))
    e1 /= np.linalg.norm(e1)
    basis = np.column_stack((e1, np.cross(n, e1), n))
    controls = tuple(tuple(basis @ np.asarray(p, float) + (.5, -1., 2.)) for p in CUBIC)
    return _spline_extrusion((.2, -.3, 1.5), controls)


SCENARIOS = {
    "spline x horizontal plate": (_spline_extrusion, (1.5, .3, 1.), (0., 0., 1.)),
    "spline x tilted plate": (_spline_extrusion, (1.5, .3, 1.), (0., .3, 1.)),
    "oblique spline x plate": (lambda: _spline_extrusion((.3, .2, 2.)), (1.5, .3, 1.), (0., 0., 1.)),
    "spline in a tilted, shifted profile plane x plate": (_tilted_spline_extrusion, (2., -.9, 3.), (0., .2, 1.)),
    "polycurve x horizontal plate": (_polycurve, (2., .3, 1.), (0., 0., 1.)),
    "polycurve x vertical plate parallel to the extrusion": (_polycurve, (2., .2, 1.), (0., 1., 0.)),
    "line, arc and spline x plate": (_mixed, (2.5, .3, .7), (0., 0., 1.)),
}


@pytest.mark.parametrize("name", SCENARIOS)
def test_a_plate_cutting_an_extruded_polycurve_conserves_material_and_exactness(name):
    build, point, normal = SCENARIOS[name]
    model, faces = build()
    faces = faces + [_plate(model, point, normal)]
    before = _total_area(model)
    _run(model, faces)
    assert model.validate_topology() == ()
    assert abs(_total_area(model) - before) <= 1e-12 * before
    assert _residual(model) < 1e-12
    assert len(model.faces) > len(faces)
    assert any(isinstance(face.surface, ExtrudedSurface) for face in model.faces.values())    # children keep the exact support
    document = to_dict(model)
    assert document["version"] == 6
    back = from_dict(document)
    assert back.validate_topology() == () and abs(_total_area(back) - _total_area(model)) <= 1e-12 * before


def test_a_horizontal_cut_rebases_each_half_onto_its_own_parameter_range():
    model, faces = _spline_extrusion()
    faces = faces + [_plate(model, (1.5, .3, 1.), (0., 0., 1.))]
    _run(model, faces)
    patches = [face.surface for face in model.faces.values() if isinstance(face.surface, ExtrudedSurface)]
    assert len(patches) == 2
    assert {patch.support_key() for patch in patches} == {patches[0].support_key()}          # one surface, two patches
    assert sorted(patch.v_range for patch in patches) == [pytest.approx((0., .5)), pytest.approx((.5, 1.))]   # in units of the vector
    assert all(patch.u_range == pytest.approx((0., 1.)) for patch in patches)


def test_an_extruded_face_without_contact_leaves_the_document_at_schema_five():
    model, faces = _polycurve()
    far = _plate(model, (2., .3, 9.), (0., 0., 1.))
    before = _total_area(model)
    _run(model, faces + [far])
    assert model.validate_topology() == () and to_dict(model)["version"] == 5
    assert abs(_total_area(model) - before) <= 1e-12 * before


def test_extrusions_the_engine_cannot_represent_are_refused_in_planning():
    non_planar = ((0., 0., 0.), (1., .8, .5), (2., -.4, 0.), (3., 0., .7))
    model, faces = _spline_extrusion(controls=non_planar)
    faces = faces + [_plate(model, (1.5, .3, 1.), (0., 0., 1.))]
    with pytest.raises(GeometryError):
        plan_intersections(model, [model.handle("face", f) for f in faces], policy=ConnectionIntent.CONNECT)
    other, other_faces = _spline_extrusion()
    first = set(other.faces)
    other.insert_model(_spline_extrusion((0., .4, 2.), controls=((0., 1., 0.), (1., 1.8, 0.), (2., .6, 0.), (3., 1., 0.)))[0])
    crossing = other_faces + sorted(set(other.faces) - first)
    with pytest.raises(GeometryError, match="non-parallel|unsupported"):
        plan_intersections(other, [other.handle("face", f) for f in crossing], policy=ConnectionIntent.CONNECT)
