"""Extruded ellipses (oblique circular or elliptic tubes) in the exact quadric intersection engine.

An extrusion of an ellipse along a vector that leaves its plane is an elliptic cylinder: a ruled quadric. These
tests cover its exact support (the rulings, the implicit form, the stored-double exactness), its intersection
with Cylinders, Cones and other tubes (residuals on both supports, an independent grid oracle, the same set
from either role), the degenerate cases (rulings parallel up to rounding, exact tangency, coincidence) and
whole models through plan/apply and the document round trip.
"""

from __future__ import annotations

import collections
import math
import warnings

import numpy as np
import pytest

from anygeometry import (ConnectionIntent, GeometryError, GeometryModel, apply_intersections, from_dict,
                         plan_intersections, query_trimmed_surface_charts, to_dict)
from anygeometry.extruded_pair_supports import extruded_pair_support
from anygeometry.extrusions import BezierDirectrix, EllipseDirectrix
from anygeometry.generators import cylinder
from anygeometry.quadric_algebra import EllipticRuledSupport, QuadricSupport, ruled_support
from anygeometry.quadric_supports import charts_support
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

TWO_PI = 2 * math.pi


def tube(center=(0., 0., 0.), a=1., b=1., vector=(.6, 0., 2.), start=0., sweep=TWO_PI, v_range=(0., 1.),
         u=(1., 0., 0.), v=(0., 1., 0.), u_range=(0., 1.)):
    u, v = np.asarray(u, float), np.asarray(v, float)
    return ExtrudedSurface(EllipseDirectrix(tuple(center), tuple(a * u), tuple(b * v), start, sweep), vector,
                           u_range, v_range)


def unit(vector):
    return tuple(np.asarray(vector, float) / np.linalg.norm(vector))


def cylinder_x(radius=.5, z=1., y=0., height=6., **angles):
    return Cylinder((-3., y, z), (1., 0., 0.), (0., 1., 0.), radius, height, **angles)


# ------------------------------------------------------------------------------------------------ the support


def test_the_support_describes_the_extruded_ellipse_exactly_in_its_stored_doubles():
    surface = tube(a=1.3, b=.7, vector=(.4, -.3, 1.7), center=(.2, -.1, .5))
    support = ruled_support(surface)
    quadric = QuadricSupport.from_surface(surface)
    assert isinstance(support, EllipticRuledSupport) and quadric.kind == "elliptic"
    rng = np.random.default_rng(3)
    uv = np.column_stack((rng.random(200), rng.random(200)))
    points = surface.evaluate_many(uv)
    assert np.abs(quadric.value(points)).max() < 1e-13                                   # the surface lies on its quadric
    assert np.abs(quadric.value(points + 0.05 * np.asarray(support.axis)) ).max() < 1e-13     # and so does every ruling
    scaled = tube(a=1.5 * 1.3, b=1.5 * .7, vector=(.4, -.3, 1.7), center=(.2, -.1, .5)).evaluate_many(uv)
    assert np.allclose(quadric.value(scaled), 1.5 ** 2 - 1, atol=1e-12)               # the value is |(x, y)|^2 - 1 in the ellipse frame
    # the ruling parameter is the height along the unit axis and the angle is the ellipse parameter
    t = uv[:, 0] * TWO_PI
    assert np.allclose(support.axial(points), uv[:, 1] * np.linalg.norm(surface.vector), atol=1e-13)
    wrapped = (support.angle_of(points) - t + math.pi) % TWO_PI - math.pi
    assert np.abs(wrapped).max() < 1e-12
    assert support.angle_of(points[0]) == pytest.approx(support.angle_of(points)[0])         # a single point too


def test_the_unit_axis_survives_reconstruction_bit_for_bit_so_the_document_round_trips_exactly():
    support = ruled_support(tube(vector=(-2.2, 2.1, 2.4)))          # normalizing this unit vector again changes its bits
    again = EllipticRuledSupport(support.origin, support.u_vector, support.v_vector, support.axis)
    assert again == support and again.axis == support.axis


def test_the_ruling_must_leave_the_plane_of_the_ellipse():
    with pytest.raises(GeometryError):
        EllipticRuledSupport((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), (1., 1., 0.))
    with pytest.raises(GeometryError):
        EllipticRuledSupport((0., 0., 0.), (1., 0., 0.), (2., 0., 0.), (0., 0., 1.))


def test_the_elliptic_quadric_is_exactly_a_cylinder_for_a_circle_and_a_perpendicular_ruling():
    elliptic = QuadricSupport.from_surface(tube(vector=(0., 0., 2.)))
    cylinder_form = QuadricSupport("cylinder", (0., 0., 0.), (0., 0., 1.), 1.)
    exact = elliptic.exact_relative((0., 0., 0.))
    assert exact == cylinder_form.exact_relative((0., 0., 0.))


# --------------------------------------------------------------------------------------- the curves


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


FULL_SECOND = {
    "a cylinder through the tube": cylinder_x(),
    "a cylinder 10 degrees off perpendicular":
        Cylinder((-3., 0., 1.), (math.cos(math.radians(10)), 0., math.sin(math.radians(10))), (0., 1., 0.), .5, 6.),
    "a skew cylinder": Cylinder((.3, -3., 1.), (0., 1., 0.), (1., 0., 0.), .4, 6.),
    "a cone": Cone((-3., 0., 1.), (1., 0., 0.), (0., 1., 0.), .2, 1.2, 6.),
    "another oblique tube": tube(center=(0., -2., 1.), a=.5, b=.5, vector=(0., 2., .5), v=(0., 0., 1.)),
    "an elliptic tube": tube(center=(.2, 0., .8), a=1.2, b=.6, vector=(-.5, .1, 1.5)),
}


@pytest.mark.parametrize("name", FULL_SECOND)
def test_a_tube_meets_a_ruled_quadric_on_both_supports_with_the_topology_of_a_grid_oracle(name):
    first, second = tube(), FULL_SECOND[name]
    result = extruded_pair_support(first, second)
    assert result.curves and not result.coincident
    quadric = QuadricSupport.from_surface(second)
    for curve in result.curves:
        points = np.asarray(curve.evaluate(np.linspace(0., 1., 101)))
        assert np.abs(first.evaluate_many(first.local_uv_many(points)) - points).max() < 1e-12
        assert (np.abs(quadric.value(points)) / quadric.gradient_norm(points)).max() < 1e-12
    polylines = [chart_polyline(curve, first) for curve in result.curves]
    for level in np.linspace(.0137, .9871, 13):               # slices of the first chart; never a split height
        exact = slice_crossings(polylines, level)
        grid = [u for u in grid_crossings(first, second, level)
                if inside_patch(second, first.evaluate_many(np.array([[u, level]]))[0])]
        assert len(exact) == len(grid), (name, float(level))
        assert np.allclose(exact, grid, atol=2e-3), (name, float(level))


def curve_set(result, samples=600):
    points = [np.asarray(curve.evaluate(np.linspace(0., 1., samples))) for curve in result.curves]
    steps = [np.linalg.norm(np.diff(p, axis=0), axis=1) for p in points]
    length = sum(float(s.sum()) for s in steps)
    return (np.vstack(points) if points else np.empty((0, 3))), length, max((float(s.max()) for s in steps), default=0.)


def one_sided(p, q):
    return float(np.sqrt(((p[:, None, :] - q[None, :, :]) ** 2).sum(-1)).min(axis=1).max())


CUT = tube(start=.4, sweep=2.2, v_range=(.2, .9))
SWAPPED = {
    "full tube, full cylinder": (tube(), cylinder_x()),
    "half tube": (tube(sweep=math.pi), cylinder_x()),
    "tube patch cut in both ranges": (CUT, cylinder_x()),
    "an angular sub-range of the tube": (tube(u_range=(.05, .45), v_range=(.1, .95)), cylinder_x()),
    "tube against a partial cylinder": (tube(), cylinder_x(start_angle=.3, sweep_angle=3.0)),
    "two cut patches": (tube(start=-1.5, sweep=2.4, v_range=(.2, .9)),
                        tube(center=(0., -2., 1.), a=.5, b=.5, vector=(0., 2., .5), v=(0., 0., 1.), start=-.5,
                             sweep=4.0, v_range=(.3, .9))),
    "cut tube against a cone": (CUT, FULL_SECOND["a cone"]),
    "cone with its apex almost on the tube": (tube(), Cone((-3., 0., 1.), (1., 0., 0.), (0., 1., 0.), .02, 1., 6.)),
}


@pytest.mark.parametrize("name", SWAPPED)
def test_either_role_gives_the_same_curve_set(name):
    """The patch cuts act as the first support's chart or as the second support's boundary planes."""
    a, b = SWAPPED[name]
    forward, backward = charts_support(a, b), charts_support(b, a)
    p, length, spacing_p = curve_set(forward)
    q, other, spacing_q = curve_set(backward)
    assert len(p) and len(q)
    assert length == pytest.approx(other, rel=1e-5)                             # polylines of 600 points
    # a point of one curve is within half a sample step of the other's samples when the curves coincide
    assert one_sided(p, q) <= .5 * spacing_q + 1e-9 and one_sided(q, p) <= .5 * spacing_p + 1e-9


def test_the_vectorized_patch_test_agrees_with_the_scalar_one():
    from anygeometry.quadric_supports import _inside_patch, _inside_patch_many
    rng = np.random.default_rng(5)
    for surface in (CUT, tube(u_range=(.15, .7), v_range=(.1, .95)), tube(sweep=math.pi)):
        uv = np.column_stack((rng.uniform(-.3, 1.3, 400), rng.uniform(-.3, 1.3, 400)))      # beyond every edge of the patch
        points = surface.evaluate_many(uv)
        many = _inside_patch_many(surface, points, 1e-8)
        assert many.tolist() == [_inside_patch(surface, p, 1e-8) for p in points]
        assert 20 < many.sum() < 380                                                       # both outcomes are exercised


def test_a_pair_with_nothing_in_common_is_empty():
    assert not extruded_pair_support(tube(), cylinder_x(radius=1.5)).curves        # the tube lies inside the cylinder
    short = cylinder_x(height=1.1).__class__((-.4, 0., 1.), (1., 0., 0.), (0., 1., 0.), .5, 1.1)
    assert not extruded_pair_support(tube(), short).curves                         # the contact lies beyond its end


# -------------------------------------------------------------------------------- degenerate configurations


def test_rulings_parallel_up_to_rounding_meet_in_generators():
    axis = unit((.6, 0., 2.))                         # the same direction as the tube's vector, normalized separately
    for x in (.2, 1.1):
        with warnings.catch_warnings():
            warnings.simplefilter("error")                # a pole of the branch chart is no reason for a floating warning
            result = extruded_pair_support(tube(), Cylinder((x, 0., -1.), axis, (0., 1., 0.), .5, 8.))
        assert not result.curves and len(result.segments) == 2
        for start, end in result.segments:
            direction = np.asarray(end) - np.asarray(start)
            assert np.linalg.norm(np.cross(direction, axis)) < 1e-12 * np.linalg.norm(direction)
            assert tube().evaluate_many(tube().local_uv_many(np.array([start, end])))[0] == pytest.approx(start, abs=1e-12)
            assert 0. <= float(start[2]) < 1e-9 and float(end[2]) == pytest.approx(2., abs=1e-9)   # the tube's two ends
    assert not extruded_pair_support(tube(), Cylinder((4., 0., -1.), axis, (0., 1., 0.), .5, 8.)).segments
    # a direction whose normalized copy has a float cross product of exactly zero is still not exactly parallel
    steep = (.3, 0., 2.)
    result = extruded_pair_support(tube(vector=steep), Cylinder((1.1, 0., -1.), unit(steep), (0., 1., 0.), .5, 8.))
    assert not result.curves and len(result.segments) == 2


def test_shared_generators_are_cut_by_the_patch_of_either_support():
    axis = unit((.6, 0., 2.))
    pipe = Cylinder((.2, 0., -1.), axis, (0., 1., 0.), .5, 8.)             # runs from z = -1 to z = 6.66 along the rulings
    cut = tube(v_range=(.2, .9))                                           # the tube between z = .4 and z = 1.8
    for a, b in ((cut, pipe), (pipe, cut)):
        result = charts_support(a, b)
        assert len(result.segments) == 2
        for start, end in result.segments:
            assert sorted((start[2], end[2])) == pytest.approx([.4, 1.8], abs=1e-9)
    short = Cylinder((.2, 0., -1.), axis, (0., 1., 0.), .5, 1.5)           # its end cap is perpendicular to the rulings
    for a, b in ((tube(), short), (short, tube())):
        result = charts_support(a, b)
        assert len(result.segments) == 2
        for start, end in result.segments:
            low, high = sorted((np.asarray(start), np.asarray(end)), key=lambda p: p[2])
            assert low[2] == pytest.approx(0., abs=1e-9)                                     # the tube's lower end
            assert (high - short.origin) @ short.axis == pytest.approx(1.5, abs=1e-9)        # the pipe's end cap


def test_a_cylinder_touching_the_tube_at_one_point_gives_that_point_in_either_role():
    first, second = tube(), cylinder_x(y=1.5)                                    # tangent at (0.3, 1, 1)
    for a, b in ((first, second), (second, first)):
        result = charts_support(a, b)
        assert not result.curves and len(result.points) == 1
        assert np.allclose(result.points[0], (.3, 1., 1.), atol=1e-12)


def test_a_cylinder_equal_to_the_tube_or_another_patch_of_it_is_coincident():
    assert extruded_pair_support(tube(vector=(0., 0., 2.)),
                                 Cylinder((0., 0., 0.), (0., 0., 1.), (1., 0., 0.), 1., 2.)).coincident
    assert extruded_pair_support(tube(), tube(v_range=(.5, 1.5))).coincident


def test_the_other_pairs_of_extruded_surfaces_keep_their_established_behavior():
    # parallel extrusions still meet in generators through the crossings of their directrices
    parallel = extruded_pair_support(tube(), tube(center=(.8, 0., 0.), a=.6, b=.6))
    assert parallel.segments and not parallel.curves
    # a Bezier of degree three or more is not a quadric: only planes and parallel extrusions are exact for it
    bezier = ExtrudedSurface(BezierDirectrix(((0., 0., 0.), (1., 1., 0.), (2., -1., 0.), (3., 0., 0.))), (.3, 0., 2.))
    with pytest.raises(GeometryError, match="unsupported"):
        extruded_pair_support(bezier, cylinder_x())
    with pytest.raises(GeometryError, match="unsupported"):
        extruded_pair_support(bezier, tube())


# ------------------------------------------------------------------------------------------- whole models


def arcs_around(model, center, radius, vector, segments=4, u=(1., 0., 0.), v=(0., 1., 0.)):
    u, v, c = np.asarray(u, float), np.asarray(v, float), np.asarray(center, float)
    at = lambda k: tuple(c + radius * (math.cos(TWO_PI * k / segments) * u + math.sin(TWO_PI * k / segments) * v))
    ids = model.add_points([at(k) for k in range(segments)])
    via = model.add_points([at(k + .5) for k in range(segments)])
    arcs = [model.add_arc(ids[k], via[k], ids[(k + 1) % segments]) for k in range(segments)]
    return list(model.extrude(arcs, vector))


def tube_with_a_pipe(radius=.5, y=0.):
    model = GeometryModel()
    faces = arcs_around(model, (0., 0., 0.), 1., (.6, 0., 2.))
    before = set(model.faces)
    model.insert_model(cylinder(radius, 6., origin=(-3., y, 1.), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=8))
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
            if isinstance(surface, (Cylinder, Cone, ExtrudedSurface)) and not isinstance(
                    getattr(surface, "directrix", None), BezierDirectrix):
                support = QuadricSupport.from_surface(surface)
                worst = max(worst, float((np.abs(support.value(points)) / support.gradient_norm(points)).max()))
    return worst


def run_plan(model, faces):
    plan = plan_intersections(model, [model.handle("face", f) for f in faces], policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)


def test_a_pipe_through_an_oblique_tube_conserves_material_and_round_trips_through_the_document():
    model, faces = tube_with_a_pipe()
    before = material_area(model)
    run_plan(model, faces)
    assert model.validate_topology() == ()
    assert abs(material_area(model) - before) <= 1e-12 * before
    assert joint_residual(model) < 1e-12
    kinds = collections.Counter(type(edge.curve).__name__ for edge in model.edges.values())
    assert kinds["QuadricIntersectionCurve"] >= 8
    document = to_dict(model)
    assert document["version"] == 6
    records = [edge["curve"] for edge in document["edges"] if edge["curve"]["type"] == "quadric_intersection"]
    assert any(r["first"]["kind"] == "elliptic" or r["second"]["kind"] == "elliptic" for r in records)
    for record in records:                                    # the elliptic quadric is written by its defining vectors
        assert ({"u_vector", "v_vector"} <= set(record["second"])) == (record["second"]["kind"] == "elliptic")
    back = from_dict(document)
    assert back.validate_topology() == ()
    for edge_id, edge in model.edges.items():
        if type(edge.curve).__name__ == "QuadricIntersectionCurve":
            assert np.array_equal(back.sample_edge(edge_id, np.linspace(0., 1., 9)),
                                  model.sample_edge(edge_id, np.linspace(0., 1., 9)))


@pytest.mark.parametrize("pipe", [{"radius": .5, "y": .3}, {"radius": .2, "y": .5}])
def test_pipes_in_other_positions_keep_the_topology_valid_and_the_material_conserved(pipe):
    model, faces = tube_with_a_pipe(**pipe)                   # an offset pipe, and a narrow one meeting the tube twice
    before = material_area(model)
    run_plan(model, faces)
    assert model.validate_topology() == () and joint_residual(model) < 1e-12
    assert abs(material_area(model) - before) <= 1e-12 * before
