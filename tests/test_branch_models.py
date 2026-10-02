"""Bezier walls of degree three and more cut by pipes, cones and tubes, through plan/apply and the document.

A cubic or quartic spline wall made by ``GeometryModel.extrude`` is an exact ruled surface that is not a quadric. Its
intersection with a quadric is a polynomial-chart curve (``BezierQuadricCurve``). These tests run whole models
through ``plan_intersections`` / ``apply_intersections`` and check what the exact engine promises: the topology
validates, the material area is conserved, every edge lies on both of its supports to rounding, the plan is
deterministic, the document round-trips every curve bit for bit, and an affine transform keeps all of it. The cases
the engine cannot represent are refused with a typed error instead of approximated.
"""

from __future__ import annotations

import collections
import math

import numpy as np
import pytest

from anygeometry import (ConnectionIntent, GeometryError, GeometryModel, ImprintOperation, IntersectionKind,
                         apply_imprint, apply_intersections, from_dict, plan_imprint, plan_intersections,
                         query_intersection, query_trimmed_surface_charts, to_dict)
from anygeometry.branch_curves import BezierQuadricCurve
from anygeometry.generators import cone, cylinder
from anygeometry.material_pair import EXACT_PAIR_ALGORITHMS
from anygeometry.operations import transform
from anygeometry.quadric_algebra import QuadricSupport
from anygeometry.surfaces import Cone, Cylinder, ExtrudedSurface

CUBIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 1., 0.))
QUARTIC = ((0., 0., 0.), (1., 2., 0.), (2., -1., 0.), (3., 2., 0.), (4., 0., 0.))
APEX = ((0., 0., 0.), (1., 2., 0.), (2., 2., 0.), (3., 0., 0.))        # y(t) = 6 t (1 - t): an arch with its top at 1.5
TWO_PI = 2 * math.pi


# ----------------------------------------------------------------------------------------------- the models


def spline_wall(controls=CUBIC, vector=(.25, 0., 1.5), shift=(0., 0., 0.)):
    model = GeometryModel()
    points = model.add_points([tuple(np.asarray(p) + shift) for p in controls])
    edge = model.add_spline(points[0], tuple(points[1:-1]), points[-1])
    return model, list(model.extrude([edge], vector))


def add_pipe(model, faces, radius=.7, y=.4, z=.6, axis=(1., 0., 0.), origin=None, radial=(0., 1., 0.), length=8.):
    before = set(model.faces)
    origin = (-2., y, z) if origin is None else origin
    model.insert_model(cylinder(radius, length, origin=origin, axis=axis, radial_direction=radial,
                                circumferential_segments=8))
    return faces + sorted(set(model.faces) - before)


def add_cone(model, faces):
    before = set(model.faces)
    model.insert_model(cone(.1, .9, 8., origin=(-2., .5, .8), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                            circumferential_segments=8))
    return faces + sorted(set(model.faces) - before)


def add_apex_cone(model, faces):
    before = set(model.faces)
    model.insert_model(cone(0., 1., 3., origin=(1.5, 1.2, 1.), axis=(0., 1., 0.), radial_direction=(1., 0., 0.),
                            circumferential_segments=8))
    return faces + sorted(set(model.faces) - before)


def add_tube(model, faces, center=(1.5, -1., .7), a=1., b=.5, vector=(0., 2.5, .3), segments=4):
    before = set(model.faces)
    c = np.asarray(center, float)
    u, v = np.array([1., 0., 0.]), np.array([0., 0., 1.])
    at = lambda k: tuple(c + a * math.cos(TWO_PI * k / segments) * u + b * math.sin(TWO_PI * k / segments) * v)
    ids = model.add_points([at(k) for k in range(segments)])
    via = model.add_points([at(k + .5) for k in range(segments)])
    arcs = [model.add_arc(ids[k], via[k], ids[(k + 1) % segments]) for k in range(segments)]
    model.extrude(arcs, vector)
    return faces + sorted(set(model.faces) - before)


def two_walls(gap=2.):
    model, faces = spline_wall()
    before = set(model.faces)
    model.insert_model(spline_wall(shift=(0., gap, 0.))[0])
    return model, faces + sorted(set(model.faces) - before)


def crossing_walls(first, second, first_vector=(0., 0., 2.), second_vector=(.3, 0., 1.6)):
    """Two walls of different directions: neither has an angle to offer, so the pair is the polynomial engine's alone."""
    model = GeometryModel()
    faces = []
    for controls, vector in ((first, first_vector), (second, second_vector)):
        points = model.add_points(controls)
        faces += list(model.extrude([model.add_spline(points[0], tuple(points[1:-1]), points[-1])], vector))
    return model, faces


ARCH = ((0., 0., 0.), (1.5, 2., 0.), (3., 0., 0.))
LEANING = ((1., -1., 0.), (1.6, .8, 0.), (1.2, 2.5, 0.))


def mixed_wall():
    """A line, an arc and a spline swept together: planes, an elliptic tube and a cubic wall in one model."""
    model = GeometryModel()
    p = model.add_points(((0., 0., 0.), (1., 0., 0.), (2., .8, 0.), (3., 0., 0.), (4., .4, 0.), (5., 0., 0.)))
    edges = [model.add_line(p[0], p[1]), model.add_arc(p[1], p[2], p[3]),
             model.add_spline(p[3], (model.add_point(3.5, 1., 0.), model.add_point(4.2, -.6, 0.)), p[5])]
    return model, list(model.extrude(edges, (.4, 0., 1.5)))


SCENARIOS = {
    "a pipe across the cubic wall": (lambda: (lambda mf: (mf[0], add_pipe(*mf)))(spline_wall()), 8),
    "a narrow pipe meeting the wall at two folds": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.3, y=.5, z=.7)))(spline_wall()), 16),
    "a wide pipe": (lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.9, y=.2, z=.9)))(spline_wall()), 4),
    "a pipe leaving through the top of the wall": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.5, y=.4, z=1.3)))(spline_wall()), 6),
    "a tilted pipe": (lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.4, axis=(math.cos(.3), 0., math.sin(.3)),
                                                              origin=(-2., .5, .3))))(spline_wall()), 8),
    "a cone": (lambda: (lambda mf: (mf[0], add_cone(*mf)))(spline_wall()), 10),
    "an elliptic tube": (lambda: (lambda mf: (mf[0], add_tube(*mf)))(spline_wall()), 4),
    "a pipe through a quartic wall": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.5, y=.6, z=.7)))(spline_wall(QUARTIC, (.2, 0., 1.5))), 8),
    "a pipe through two parallel cubic walls": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.3, axis=(0., 1., 0.), origin=(1.5, -1.5, .7),
                                                radial=(1., 0., 0.), length=6.)))(two_walls()), 16),
    "a pipe through a mixed wall of a line, an arc and a spline": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.33, origin=(-1., .45, .8))))(mixed_wall()), 4),
    "a pipe lying in the wall's base plane (every fold on the boundary)": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.3, y=0., z=0.)))(spline_wall()), 3),
    "a pipe grazing the apex of an arch": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=1., y=2.4, z=1.)))(spline_wall(APEX, (0., 0., 2.))), 3),
    "a pipe touching the apex of an arch at one point": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=1., y=2.5, z=1.)))(spline_wall(APEX, (0., 0., 2.))), 0),
    "a cone whose apex is off the wall": (
        lambda: (lambda mf: (mf[0], add_apex_cone(*mf)))(spline_wall(APEX, (0., 0., 2.))), 8),
    "two quadratic walls of different directions": (lambda: crossing_walls(ARCH, LEANING), 1),
    "a cubic and a quadratic wall of different directions": (lambda: crossing_walls(CUBIC, LEANING), 1),
    "a pipe along the extrusion (whole generators)": (
        lambda: (lambda mf: (mf[0], add_pipe(*mf, radius=.35, axis=(.25, 0., 1.5), origin=(1.3, .5, -.5),
                                                length=3.)))(spline_wall()), 0),
}


# ----------------------------------------------------------------------------------------------- the checks


def material_area(model):
    return sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)


def joint_residual(model):
    """Largest distance of a sampled edge from the supports of the faces that use it."""
    owners = collections.defaultdict(set)
    for face_id, face in model.faces.items():
        for use in (*face.loop, *(e for hole in face.holes for e in hole)):
            owners[use.edge].add(face_id)
    worst = 0.
    for edge_id, faces in owners.items():
        points = model.sample_edge(edge_id, np.linspace(0., 1., 25))
        for face_id in faces:
            surface = model.faces[face_id].surface
            if isinstance(surface, ExtrudedSurface) and hasattr(surface.directrix, "controls"):
                worst = max(worst, float(np.abs(surface.evaluate_many(surface.local_uv_many(points)) - points).max()))
            elif isinstance(surface, (Cylinder, Cone, ExtrudedSurface)):
                support = QuadricSupport.from_surface(surface)
                worst = max(worst, float((np.abs(support.value(points)) / np.maximum(support.gradient_norm(points), 1e-300)).max()))
    return worst


def run(model, faces):
    handles = [model.handle("face", f) for f in faces]
    plan = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    return plan


def branch_edges(model):
    return [edge_id for edge_id, edge in model.edges.items() if isinstance(edge.curve, BezierQuadricCurve)]


@pytest.mark.parametrize("name", SCENARIOS)
def test_whole_models_keep_their_topology_material_and_supports_and_round_trip_exactly(name):
    build, expected = SCENARIOS[name]
    model, faces = build()
    before = material_area(model)
    run(model, faces)
    assert model.validate_topology() == ()
    assert abs(material_area(model) - before) <= 1e-12 * before
    assert joint_residual(model) < 1e-12
    assert len(branch_edges(model)) >= expected
    document = to_dict(model)
    assert document["version"] == 6 or not expected
    back = from_dict(document)
    assert back.validate_topology() == ()
    grid = np.linspace(0., 1., 9)
    for edge_id in branch_edges(model):
        assert np.array_equal(back.sample_edge(edge_id, grid), model.sample_edge(edge_id, grid))
        assert back.edges[edge_id].curve == model.edges[edge_id].curve


def test_the_plan_is_deterministic():
    model, faces = SCENARIOS["a pipe across the cubic wall"][0]()
    handles = [model.handle("face", f) for f in faces]
    first = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    second = plan_intersections(model, handles, policy=ConnectionIntent.CONNECT)
    assert first.content_checksum == second.content_checksum            # planning reads the model and changes nothing


def test_branch_curves_follow_an_affine_transform_of_the_model():
    model, faces = SCENARIOS["a pipe across the cubic wall"][0]()
    run(model, faces)
    area = material_area(model)
    c, s = math.cos(.7), math.sin(.7)
    matrix = np.eye(4)
    matrix[:3, :3] = 1.7 * np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]]) @ np.array(
        [[1., 0., 0.], [0., math.cos(.3), -math.sin(.3)], [0., math.sin(.3), math.cos(.3)]])
    matrix[:3, 3] = (3., -2., 1.)
    transform(model, matrix)
    assert model.validate_topology() == ()
    assert material_area(model) == pytest.approx(area * 1.7 ** 2, rel=1e-12)
    assert joint_residual(model) < 1e-12
    assert from_dict(to_dict(model)).validate_topology() == ()


# ----------------------------------------------------------------------------------------------- queries


def test_the_public_query_classifies_a_wall_and_a_pipe_by_the_exact_engine_and_imprints_it():
    model, faces = SCENARIOS["a pipe across the cubic wall"][0]()
    wall = faces[0]
    kinds = collections.Counter()
    for face in faces[1:]:
        result = query_intersection(model, model.handle("face", face), model.handle("face", wall))
        assert result.classified
        for component in result.components:
            assert {component.certificate.algorithm} <= set(EXACT_PAIR_ALGORITHMS)
            kinds[type(component.analytic_curve).__name__] += 1
        if result.kind is IntersectionKind.DISJOINT:
            assert not result.components
    assert kinds["BezierQuadricCurve"] >= 6
    result = query_intersection(model, model.handle("face", faces[1]), model.handle("face", wall))
    plan = plan_imprint(model, result, policy="connect")
    assert plan.operation is ImprintOperation.FACE_IMPRINT and plan.batch_plan is not None
    before = material_area(model)
    apply_imprint(model, plan, policy="connect")
    assert model.validate_topology() == ()
    assert abs(material_area(model) - before) <= 1e-12 * before


# ----------------------------------------------------------------------------------------------- refusals


def test_two_cubic_walls_of_different_directions_are_refused_in_planning():
    model, faces = spline_wall()
    first = set(model.faces)
    model.insert_model(spline_wall(controls=((0., 1., 0.), (1., 1.8, 0.), (2., .6, 0.), (3., 1., 0.)), vector=(0., .4, 2.))[0])
    crossing = faces + sorted(set(model.faces) - first)
    with pytest.raises(GeometryError, match="non-parallel|unsupported"):
        plan_intersections(model, [model.handle("face", f) for f in crossing], policy=ConnectionIntent.CONNECT)


# ----------------------------------------------------------------------------------------------- iterative modelling


def test_a_second_pipe_through_an_already_cut_wall_works_on_the_stored_child_supports():
    """After the first operation the wall is several faces that store exact patches of one surface (rebased
    ranges); a second operation meets those patches, not the original wall."""
    model, faces = spline_wall()
    faces = add_pipe(model, faces, radius=.5, y=.4, z=.5)
    run(model, faces)
    children = [f for f, face in model.faces.items() if isinstance(face.surface, ExtrudedSurface)]
    assert len(children) >= 2 and all(model.faces[f].surface.support_key() == model.faces[children[0]].surface.support_key()
                                      for f in children)
    before = set(model.faces)
    model.insert_model(cylinder(.3, 8., origin=(-2., .5, 1.3), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=8))
    new = sorted(set(model.faces) - before)
    area, older = material_area(model), len(branch_edges(model))
    run(model, children + new)
    assert model.validate_topology() == ()
    assert abs(material_area(model) - area) <= 1e-12 * area
    assert joint_residual(model) < 1e-12
    assert len(branch_edges(model)) > older


def test_a_second_pipe_that_splits_an_earlier_joint_is_refused_by_the_established_attachment_policy():
    """The same refusal as for every curve family: a joint edge that carries attachments is not split silently."""
    model, faces = spline_wall()
    faces = add_pipe(model, faces, radius=.5, y=.4, z=.5)
    run(model, faces)
    children = [f for f, face in model.faces.items() if isinstance(face.surface, ExtrudedSurface)]
    before = set(model.faces)
    model.insert_model(cylinder(.25, 8., origin=(-2., .6, 1.1), axis=(1., 0., 0.), radial_direction=(0., 1., 0.),
                                circumferential_segments=8))
    new = sorted(set(model.faces) - before)
    with pytest.raises(GeometryError, match="attachments .* explicit parameter remap"):
        run(model, children + new)


# ----------------------------------------------------------------------------------------------- editing operations


def test_models_with_branch_edges_can_be_copied_and_inserted():
    model, faces = SCENARIOS["a pipe across the cubic wall"][0]()
    run(model, faces)
    area = material_area(model)
    target = GeometryModel()
    target.insert_model(model)
    assert target.validate_topology() == ()
    assert len(branch_edges(target)) == len(branch_edges(model)) > 0
    assert abs(material_area(target) - area) <= 1e-12 * area


def test_a_branch_edge_without_attachments_can_be_reversed():
    from anygeometry.branch_supports import bezier_quadric_support
    from anygeometry.editing import reverse_edge
    from anygeometry.extrusions import BezierDirectrix
    wall = ExtrudedSurface(BezierDirectrix(CUBIC), (.25, 0., 1.5))
    curve = list(bezier_quadric_support(wall, Cylinder((-2., .5, .7), (1., 0., 0.), (0., 1., 0.), .3, 8.)).curves)[0]
    model = GeometryModel()
    start, end = model.add_points(curve.evaluate(np.asarray((0., 1.))))
    edge = model.add_curve(start, end, curve)
    grid = np.linspace(0., 1., 9)
    before = model.sample_edge(edge, grid)
    reverse_edge(model, edge)
    assert np.allclose(model.sample_edge(edge, grid), before[::-1], atol=1e-13)             # the same chart run backwards
    assert isinstance(model.edges[edge].curve, BezierQuadricCurve) and model.validate_topology() == ()
