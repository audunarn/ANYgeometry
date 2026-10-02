"""Quadratic Bezier extrusions (spline walls) as exact quadrics in the quadric intersection engine.

A quadratic Bezier is a parabola, so the extrusion of a spline with one control point is a parabolic cylinder:
a quadric whose implicit form is exact in its control points. It meets Cylinders, Cones and elliptic tubes
through the quadric branch engine as their second support (its chart is polynomial, so it never supplies the
angle). These tests cover the exact quadric, the curves against an independent grid oracle, degenerate
contacts, the refusals that remain and whole models through plan/apply and the document round trip.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from anygeometry import (ConnectionIntent, GeometryError, GeometryModel, apply_intersections, from_dict,
                         plan_intersections, query_trimmed_surface_charts, to_dict)
from anygeometry.extruded_pair_supports import extruded_pair_support
from anygeometry.extrusions import BezierDirectrix, EllipseDirectrix
from anygeometry.generators import cylinder
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.quadric_supports import _boundary_planes, quadric_pair_support
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

ARCH = ((0., 0., 0.), (1., 2., 0.), (2., 0., 0.))                 # the parabola y = 1 - (x - 1)^2, apex (1, 1)


def wall(controls=ARCH, vector=(0., 0., 2.), u_range=(0., 1.), v_range=(0., 1.)):
    return ExtrudedSurface(BezierDirectrix(controls), vector, u_range, v_range)


def pipe_x(radius=.5, z=1., y=.3, height=6., **angles):
    return Cylinder((-2., y, z), (1., 0., 0.), (0., 1., 0.), radius, height, **angles)


# ------------------------------------------------------------------------------------------------ the quadric


def test_the_parabolic_quadric_is_exactly_the_wall_and_its_value_is_the_parabola_defect():
    surface = wall(controls=((.2, -.1, .3), (1.4, 1.7, .9), (2.3, .2, 1.1)), vector=(.3, -.2, 1.6))
    quadric = QuadricSupport.from_surface(surface)
    assert quadric.kind == "parabolic"
    rng = np.random.default_rng(3)
    uv = rng.random((200, 2))
    assert np.abs(quadric.value(surface.evaluate_many(uv))).max() < 1e-13               # the wall lies on its quadric
    # in the profile plane, a point y' = a e + b f has value a^2 - 4 b whatever its height along the ruling
    p0, p1, p2 = (np.asarray(c) for c in surface.directrix.controls)
    e, f = p1 - p0, p2 - 2 * p1 + p0
    a, b, height = rng.uniform(-1, 3, 60), rng.uniform(-1, 3, 60), rng.uniform(-2, 2, 60)
    points = p0 + a[:, None] * e + b[:, None] * f + height[:, None] * np.asarray(surface.vector)
    assert np.allclose(quadric.value(points), a * a - 4 * b, atol=1e-12)
    for point in points[:5]:                                                 # the gradient norm against central differences
        numeric = np.array([(quadric.value(point + 1e-6 * unit) - quadric.value(point - 1e-6 * unit)) / 2e-6
                            for unit in np.eye(3)])
        assert np.linalg.norm(numeric) == pytest.approx(float(quadric.gradient_norm(point)), rel=1e-6)


def test_the_parabolic_quadric_needs_a_real_parabola_and_a_ruling_out_of_its_plane():
    with pytest.raises(GeometryError):
        QuadricSupport("parabolic", (0., 0., 0.), (0., 0., 1.), u_vector=(1., 0., 0.), v_vector=(2., 0., 0.))   # a line
    with pytest.raises(GeometryError):
        QuadricSupport("parabolic", (0., 0., 0.), (1., 0., 0.), u_vector=(1., 2., 0.), v_vector=(2., 0., 0.))   # in plane


# ---------------------------------------------------------------------------------------------- the curves


def chart_polyline(curve, surface, samples=600):
    return surface.local_uv_many(np.asarray(curve.evaluate(np.linspace(0., 1., samples))))


def slice_crossings(polylines, level):
    found = []
    for uv in polylines:
        d = uv[:, 1] - level
        for i in np.nonzero(d[:-1] * d[1:] < 0)[0]:
            w = d[i] / (d[i] - d[i + 1])
            found.append(uv[i, 0] + w * (uv[i + 1, 0] - uv[i, 0]))
    return sorted(found)


def grid_crossings(first, second, level, columns=3000):
    u = np.linspace(0., 1., columns + 1)
    values = QuadricSupport.from_surface(second).value(first.evaluate_many(np.column_stack((u, np.full_like(u, level)))))
    return [u[i] + values[i] / (values[i] - values[i + 1]) * (u[i + 1] - u[i])
            for i in np.nonzero(values[:-1] * values[1:] < 0)[0]]


def inside_patch(second, point):
    if isinstance(second, ExtrudedSurface):
        u, v = second.local_uv(point)
        return -1e-9 <= u <= 1 + 1e-9 and -1e-9 <= v <= 1 + 1e-9
    z = float((np.asarray(point) - second.origin) @ second.axis)
    return min(0., second.height) - 1e-9 <= z <= max(0., second.height) + 1e-9


def tube(center=(1., 0., 1.), a=.8, b=.6, vector=(0., 1., .3), u=(1., 0., 0.), v=(0., 0., 1.)):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return ExtrudedSurface(EllipseDirectrix(tuple(center), tuple(a * u), tuple(b * v)), vector)


SECONDS = {
    "a pipe across the wall": pipe_x(),
    "a pipe 10 degrees off perpendicular": Cylinder((-2., .3, 1.), (math.cos(math.radians(10)), 0., math.sin(math.radians(10))),
                                                    (0., 1., 0.), .5, 6.),
    "a wide pipe cutting the arch twice": pipe_x(radius=.9, y=.2),
    "a cone": Cone((-2., .3, 1.), (1., 0., 0.), (0., 1., 0.), .2, 1.0, 6.),
    "a skew pipe along y": Cylinder((1.2, -2., 1.), (0., 1., 0.), (1., 0., 0.), .4, 6.),
    "an elliptic tube": tube(),
}


@pytest.mark.parametrize("name", SECONDS)
def test_a_wall_meets_a_ruled_quadric_on_both_supports_with_the_topology_of_a_grid_oracle(name):
    first, second = wall(), SECONDS[name]
    result = extruded_pair_support(first, second)
    assert (result.curves or result.segments) and not result.coincident
    quadric = QuadricSupport.from_surface(second)
    for curve in result.curves:
        points = np.asarray(curve.evaluate(np.linspace(0., 1., 101)))
        assert np.abs(first.evaluate_many(first.local_uv_many(points)) - points).max() < 1e-12
        assert (np.abs(quadric.value(points)) / quadric.gradient_norm(points)).max() < 1e-12
    polylines = [chart_polyline(curve, first) for curve in result.curves]
    for level in np.linspace(.0137, .9871, 13):
        exact = slice_crossings(polylines, level)
        grid = [u for u in grid_crossings(first, second, level)
                if inside_patch(second, first.evaluate_many(np.array([[u, level]]))[0])]
        assert len(exact) == len(grid), (name, float(level))
        assert np.allclose(exact, grid, atol=2e-3), (name, float(level))


def test_the_patch_cuts_of_the_wall_clip_the_curves_exactly():
    pipe = pipe_x(radius=.9, y=.2)
    whole = extruded_pair_support(wall(), pipe)
    for cut in (wall(u_range=(.1, .8)), wall(v_range=(.15, .7)), wall(u_range=(.3, .95), v_range=(.2, .9))):
        result = extruded_pair_support(cut, pipe)
        points = np.vstack([c.evaluate(np.linspace(0., 1., 80)) for c in result.curves])
        uv = cut.local_uv_many(points)
        assert uv.min() > -1e-9 and uv.max() < 1 + 1e-9                            # every point lies in the patch
        whole_length = sum(float(np.linalg.norm(np.diff(c.evaluate(np.linspace(0., 1., 600)), axis=0), axis=1).sum())
                           for c in whole.curves)
        cut_length = sum(float(np.linalg.norm(np.diff(c.evaluate(np.linspace(0., 1., 600)), axis=0), axis=1).sum())
                         for c in result.curves)
        assert 0. < cut_length < whole_length


def test_one_plane_holds_both_seams_of_a_patch_and_meets_the_wall_nowhere_else():
    cut = wall(u_range=(.25, .75))
    planes = _boundary_planes(cut)
    assert len(planes) == 3                                                             # two end planes, one seam plane
    seam = planes[2]
    normal = np.asarray(seam.axis) / np.linalg.norm(seam.axis)
    assert abs(float(np.asarray(cut.vector) @ normal)) < 1e-12                          # parallel to the ruling
    for t in cut.u_range:
        assert abs(float((cut.directrix.point(t) - seam.origin) @ normal)) < 1e-12       # through both end generators
    # a conic meets a line twice at most: along the whole parabola the plane vanishes at the two seams only
    t = np.linspace(-1., 2., 4001)
    values = (cut.directrix.point(t) - seam.origin) @ normal
    assert np.count_nonzero(values[:-1] * values[1:] < 0) == 2
    assert wall(u_range=(0., 1.)).directrix is not None and len(_boundary_planes(wall())) == 3


# ------------------------------------------------------------------------------ degenerate configurations


def test_a_pipe_touching_the_arch_at_its_apex_gives_that_point_exactly():
    touching = Cylinder((-2., 1.5, 1.), (1., 0., 0.), (0., 1., 0.), .5, 6.)            # lowest line y = 1 at z = 1
    for a, b in ((wall(), touching), (touching, wall())):
        result = quadric_pair_support(a, b)
        assert not result.curves and len(result.points) == 1
        assert np.allclose(result.points[0], (1., 1., 1.), atol=1e-12)


def test_rulings_parallel_to_the_extrusion_meet_the_arch_in_generators():
    pipe = Cylinder((1., .4, -1.), (0., 0., 1.), (1., 0., 0.), .7, 5.)                  # meets the arch at x = 1 +- .688
    result = extruded_pair_support(wall(), pipe)
    assert not result.curves and len(result.segments) == 2
    for start, end in result.segments:
        assert np.allclose(np.asarray(start)[:2], np.asarray(end)[:2], atol=1e-12)           # along the ruling
        assert sorted((start[2], end[2])) == pytest.approx([0., 2.], abs=1e-9)             # the wall's two ends
    oblique = (.3, 0., 2.)                                                              # a pipe along the wall's own vector
    skew_wall = wall(vector=oblique)
    axis = tuple(np.asarray(oblique) / np.linalg.norm(oblique))
    result = extruded_pair_support(skew_wall, Cylinder((.2, .5, -1.), axis, (0., 1., 0.), .6, 6.))
    assert not result.curves and result.segments                                        # parallel up to rounding


def test_the_same_surface_is_coincident_and_other_pairs_keep_their_established_behavior():
    assert extruded_pair_support(wall(), wall(v_range=(.5, 1.5))).coincident
    cubic = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., -1., 0.), (3., 0., 0.))), (0., 0., 2.))
    assert extruded_pair_support(cubic, pipe_x()).curves                                  # a cubic is the polynomial engine's wall
    other = wall(vector=(0., .5, 2.))                                                      # two parabolic cylinders: neither has an
    both = extruded_pair_support(wall(), other)                                            # angle, so the polynomial engine takes one
    assert both.curves and not both.segments
    length = lambda result: sum(float(np.linalg.norm(np.diff(c.evaluate(np.linspace(0., 1., 4001)), axis=0), axis=1).sum())
                                for c in result.curves)
    assert length(both) == pytest.approx(length(extruded_pair_support(other, wall())), rel=1e-6)        # from either wall
    with pytest.raises(GeometryError, match="unsupported"):                                # two cubic walls are still refused
        extruded_pair_support(cubic, ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., -1., 0.), (3., 0., 0.))),
                                                     (0., .5, 2.)))
    parallel = extruded_pair_support(wall(), wall(controls=((0., .5, 0.), (1., 2.5, 0.), (2., .5, 0.))))
    assert not parallel.curves                                                         # parallel extrusions: generators


# ------------------------------------------------------------------------------------------- whole models


def spline_wall_with_a_pipe(radius=.5, y=.3, pipe_segments=8):
    model = GeometryModel()
    a, b, c = model.add_points(ARCH)
    faces = list(model.extrude([model.add_spline(a, (b,), c)], (0., 0., 2.)))
    before = set(model.faces)
    model.insert_model(cylinder(radius, 6., origin=(-2., y, 1.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=pipe_segments))
    return model, faces + sorted(set(model.faces) - before)


def material_area(model):
    return sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)


def joint_residual(model):
    owners = collections.defaultdict(set)
    for face_id, face in model.faces.items():
        for use in (*face.loop, *(e for hole in face.holes for e in hole)):
            owners[use.edge].add(face_id)
    worst = 0.
    for edge_id, faces in owners.items():
        points = model.sample_edge(edge_id, np.linspace(0., 1., 25))
        for face_id in faces:
            surface = model.faces[face_id].surface
            if isinstance(surface, ExtrudedSurface) and surface.directrix.__class__ is BezierDirectrix:
                worst = max(worst, float(np.abs(surface.evaluate_many(surface.local_uv_many(points)) - points).max()))
            elif isinstance(surface, (Cylinder, Cone, ExtrudedSurface)):
                support = QuadricSupport.from_surface(surface)
                worst = max(worst, float((np.abs(support.value(points)) / support.gradient_norm(points)).max()))
    return worst


def run_plan(model, faces):
    plan = plan_intersections(model, [model.handle("face", f) for f in faces], policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)


def test_a_pipe_through_a_spline_wall_conserves_material_and_round_trips_through_the_document():
    model, faces = spline_wall_with_a_pipe()
    before = material_area(model)
    run_plan(model, faces)
    assert model.validate_topology() == ()
    assert abs(material_area(model) - before) <= 1e-12 * before
    assert joint_residual(model) < 1e-12
    kinds = collections.Counter(type(edge.curve).__name__ for edge in model.edges.values())
    assert kinds["QuadricIntersectionCurve"] >= 4
    document = to_dict(model)
    assert document["version"] == 6
    records = [edge["curve"] for edge in document["edges"] if edge["curve"]["type"] == "quadric_intersection"]
    assert any(r["second"]["kind"] == "parabolic" for r in records)
    for record in records:
        assert ({"u_vector", "v_vector"} <= set(record["second"])) == (record["second"]["kind"] in ("elliptic", "parabolic"))
    back = from_dict(document)
    assert back.validate_topology() == ()
    for edge_id, edge in model.edges.items():
        if type(edge.curve).__name__ == "QuadricIntersectionCurve":
            assert np.array_equal(back.sample_edge(edge_id, np.linspace(0., 1., 9)),
                                  model.sample_edge(edge_id, np.linspace(0., 1., 9)))


@pytest.mark.parametrize("pipe", [{"radius": .5, "y": .5}, {"radius": .9, "y": .2}])
def test_pipes_in_other_positions_keep_the_topology_valid_and_the_material_conserved(pipe):
    model, faces = spline_wall_with_a_pipe(**pipe)
    before = material_area(model)
    run_plan(model, faces)
    assert model.validate_topology() == () and joint_residual(model) < 1e-12
    assert abs(material_area(model) - before) <= 1e-12 * before
