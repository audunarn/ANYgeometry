"""Extruded walls through the public query and imprint workflow.

A wall made by ``GeometryModel.extrude`` is recognised as an exact extruded support, but the public workflow
keeps its established behaviour wherever that behaviour classified the pair: the complete-boundary-curve CONNECT
partition and every classification of the certified subdivision engine stay as they were. The exact engine
classifies what the established query leaves unclassified, except an explicit boundary-curve refusal, which
stands. These tests pin that routing; the geometry itself is covered by ``test_extruded_intersections``.
"""

from __future__ import annotations

import numpy as np

from anygeometry import (GeometryModel, ImprintOperation, IntersectionKind, apply_imprint, plan_imprint,
                         query_intersection, query_trimmed_surface_charts)
from anygeometry.arrangement_geometry import BezierPath
from anygeometry.extruded_supports import extruded_support
from anygeometry.material_pair import EXACT_PAIR_ALGORITHMS

CUBIC = ((0., 0., 0.), (1., .8, 0.), (2., -.4, 0.), (3., 0., 0.))
NONCONVEX = ((-1., -2., 1.), (4., -2., 1.), (1.5, .3, 1.), (4., 3., 1.), (-1., 3., 1.))
CONVEX = ((-1., -2., 1.), (4., -2., 1.), (4., 3., 1.), (-1., 3., 1.))


def _wall(model, controls=CUBIC, vector=(0., 0., 2.)):
    points = model.add_points(controls)
    return model.extrude([model.add_spline(points[0], tuple(points[1:-1]), points[-1])], vector)[0]


def _pair(corners):
    model = GeometryModel()
    wall = _wall(model)
    plate = model.add_plate(model.add_points(corners))
    return model, plate, wall


def _algorithms(result):
    return {component.certificate.algorithm for component in result.components}


def _area(model):
    return sum(chart.material_area for chart in query_trimmed_surface_charts(model).charts)


def test_a_wall_is_recovered_as_an_exact_extrusion_but_stays_a_coons_patch_in_the_model():
    model, _plate, wall = _pair(CONVEX)
    assert type(model.faces[wall].surface).__name__ == "CoonsSurface"
    assert type(extruded_support(model, wall)).__name__ == "ExtrudedSurface"


def test_a_classification_of_the_established_query_keeps_its_imprint_planning():
    model, plate, wall = _pair(CONVEX)
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    assert result.kind is IntersectionKind.CROSS and result.classified
    assert not _algorithms(result) & set(EXACT_PAIR_ALGORITHMS)           # classified before the exact engine
    plan = plan_imprint(model, result, policy="connect")
    assert plan.operation is ImprintOperation.FACE_IMPRINT and plan.batch_plan is None


def test_the_complete_boundary_curve_connect_keeps_its_established_planning():
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0))))
    wall = _wall(model, ((.5, .5, 0.), (1.5, 1.5, 0.), (2.5, .5, 0.)), (0., 0., 1.))
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    assert result.kind is IntersectionKind.CONTAINED
    assert plan_imprint(model, result, policy="connect").batch_plan is None


def test_an_explicit_boundary_curve_refusal_stands_for_a_curve_lying_in_the_support():
    for corners in (((0, 0, 0), (3, 0, 0), (1.5, .75, 0), (3, 2, 0), (0, 2, 0)),      # non-convex support
                    ((1, 0, 0), (3, 0, 0), (3, 2, 0), (1, 2, 0))):                    # the curve leaves the trim
        model = GeometryModel()
        plate = model.add_plate(model.add_points(corners))
        wall = _wall(model, ((.5, .5, 0.), (1.5, 1.5, 0.), (2.5, .5, 0.)), (0., 0., 1.))
        result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
        assert result.kind is IntersectionKind.UNCLASSIFIED and not result.classified
        assert plan_imprint(model, result, policy="connect").operation is ImprintOperation.NO_TOPOLOGY


def test_the_refusal_does_not_hold_back_a_support_the_wall_merely_crosses():
    model, plate, wall = _pair(NONCONVEX)                  # the non-convex refusal is raised for any such support
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    assert result.kind is IntersectionKind.CROSS and result.classified
    assert _algorithms(result) <= set(EXACT_PAIR_ALGORITHMS)
    (component,) = result.components
    curve = component.analytic_curve
    assert isinstance(curve, BezierPath)
    points = curve.evaluate(np.linspace(0., 1., 33))
    support = extruded_support(model, wall)
    assert np.abs(points[:, 2] - 1.).max() < 1e-12                                       # on the plate's plane
    assert np.abs(support.evaluate_many(support.local_uv_many(points)) - points).max() < 1e-12     # on the wall
    assert -1. < points[:, 0].min() and points[:, 0].max() < 3.0000001


def test_the_exact_classification_is_planned_and_applied_with_material_conserved():
    model, plate, wall = _pair(NONCONVEX)
    model.add_sheet((plate,))
    model.add_sheet((wall,))
    before = _area(model)
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    plan = plan_imprint(model, result, policy="connect")
    assert plan.operation is ImprintOperation.FACE_IMPRINT and plan.batch_plan is not None
    apply_imprint(model, plan, policy="connect")
    assert model.validate_topology() == ()
    assert len(model.faces) > 2
    assert abs(_area(model) - before) <= 1e-12 * before
