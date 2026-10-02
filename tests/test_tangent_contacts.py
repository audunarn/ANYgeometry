"""Public topology regression for an exact generator contact with a plate."""
import math

import numpy as np
import pytest

from anygeometry import (
    ConnectionIntent, GeometryModel, apply_intersections, plan_intersections,
    query_trimmed_surface_charts, to_dict, Cylinder, Plane,
)
from anygeometry.analytic_supports import plane_cylinder_support
from anygeometry.generators import cylinder
from anygeometry.exact_curves import EllipticArc
from anygeometry.arrangement_geometry import LinePath, line_curve_junctions, plane_roots


def tangent_plate_pipe(angle=0.0):
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0., -2., -1.), (0., 2., -1.),
                                              (0., 2., 3.), (0., -2., 3.))))
    model.add_sheet((plate,))
    model.insert_model(cylinder(1., 2., origin=(1., 0., 0.),
                                radial_direction=(math.cos(angle), math.sin(angle), 0.),
                                circumferential_segments=8))
    return model, [model.handle("face", face) for face in model.faces]


@pytest.mark.parametrize("angle", [0.0, math.pi / 8, .2, .7])
def test_exact_tangent_plate_and_pipe_prepare_atomically(angle):
    model, operands = tangent_plate_pipe(angle)
    original = to_dict(model)
    plan = plan_intersections(model, operands, policy=ConnectionIntent.CONNECT)
    assert to_dict(model) == original
    application = apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert application.joint_edges, "tangent generator must connect plate and cylinder"
    assert model.validate_topology() == ()
    # Material domains must still include the entire plate and cylinder shell.
    charts = query_trimmed_surface_charts(model)
    assert sum(chart.material_area for chart in charts.charts) == pytest.approx(16. + 4. * math.pi)
    for edge in application.joint_edges:
        assert len(model.faces_using_edge(edge.id)) >= 2
        np.testing.assert_allclose(model.sample_edge(edge.id, np.array([0., 1.]))[:, :2], 0., atol=1e-12)
    after = to_dict(model)
    assert apply_intersections(model, plan, policy=ConnectionIntent.CONNECT).reused
    assert to_dict(model) == after


@pytest.mark.parametrize("angle", [0., math.pi / 8, .2, .7])
@pytest.mark.parametrize("offset,count", [(-1e-6, 2), (0., 1), (1e-6, 0)])
def test_generator_contacts_preserve_nearby_secant_and_separated_classification(angle, offset, count):
    pipe = Cylinder((1. + offset, 0., 0.), (0., 0., 1.),
                    (math.cos(angle), math.sin(angle), 0.), 1., 2.)
    plane = Plane((0., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    result = plane_cylinder_support(plane, pipe)
    assert len(result.segments) == count
    for start, end in result.segments:
        assert (start[0], end[0]) == pytest.approx((0., 0.), abs=1e-14)
        assert (start[2], end[2]) == (0., 2.)
        assert start[1] ** 2 == pytest.approx(1. - (1. + offset) ** 2, abs=1e-14)


@pytest.mark.parametrize("start,sweep,count", [(0., math.pi/2, 0), (math.pi/2, math.pi, 1),
                                               (3*math.pi/2, -math.pi, 1)])
def test_tangent_generator_respects_native_patch_and_negative_height(start, sweep, count):
    pipe = Cylinder((1., 0., 2.), (0., 0., 1.), (1., 0., 0.), 1., -2., start, sweep)
    plane = Plane((0., 0., 0.), (0., 1., 0.), (0., 0., 1.))
    result = plane_cylinder_support(plane, pipe)
    assert len(result.segments) == count
    if count:
        assert result.segments == (((0., 0., 2.), (0., 0., 0.)),)


@pytest.mark.parametrize("start,sweep", [(1., 1e-5), (1., -1e-5), (3., 1e-7), (4., -1e-7)])
def test_small_arc_endpoint_is_a_valid_line_junction(start, sweep):
    arc = EllipticArc((0., 0., 0.), (1., 0., 0.), (0., 1., 0.), start, sweep)
    y = math.sin(start + sweep)
    roots = plane_roots(arc, (0., 1., 0.), y)
    assert roots and all(0. <= parameter <= 1. for parameter in roots)
    line = LinePath((-2., y, 0.), (2., y, 0.))
    junctions = line_curve_junctions(line, arc)
    assert junctions
    for first, second in junctions:
        assert line.evaluate(first) == pytest.approx(arc.evaluate(second), abs=1e-12)
