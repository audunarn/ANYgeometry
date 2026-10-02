"""Extruded walls through the public query and imprint workflow.

A wall made by ``GeometryModel.extrude`` is recognised as an exact extruded support. The public query
classifies a pair that includes one with the exact material engine (one exact curve where the certified
subdivision returns many sampled pieces) and falls back to the established query only where the exact engine
refuses. The one exception is the complete-boundary-curve CONNECT: a convex support that strictly contains a
wall's edge keeps its established result and quadratic partition. These tests pin that routing; the geometry
itself is covered by ``test_extruded_intersections``.
"""

from __future__ import annotations

import numpy as np

from anygeometry import (GeometryModel, ImprintOperation, IntersectionKind, apply_imprint, plan_imprint,
                         query_intersection, query_trimmed_surface_charts)
from anygeometry.arrangement_geometry import BezierPath
from anygeometry.extruded_supports import extruded_support
from anygeometry.surfaces import ExtrudedSurface
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


def test_a_plate_across_a_recovered_wall_is_classified_by_the_exact_engine():
    model, plate, wall = _pair(CONVEX)
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    assert result.kind is IntersectionKind.CROSS and result.classified
    assert _algorithms(result) <= set(EXACT_PAIR_ALGORITHMS)               # not the certified subdivision's pieces
    (component,) = result.components                                         # one exact curve, not many sampled pieces
    assert isinstance(component.analytic_curve, BezierPath)
    plan = plan_imprint(model, result, policy="connect")
    assert plan.operation is ImprintOperation.FACE_IMPRINT and plan.batch_plan is not None


def test_the_complete_boundary_curve_connect_keeps_its_established_planning():
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0))))
    wall = _wall(model, ((.5, .5, 0.), (1.5, 1.5, 0.), (2.5, .5, 0.)), (0., 0., 1.))
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    assert result.kind is IntersectionKind.CONTAINED
    assert plan_imprint(model, result, policy="connect").batch_plan is None


def test_a_curve_lying_in_a_non_convex_or_trimmed_support_is_no_longer_refused():
    for corners in (((0, 0, 0), (3, 0, 0), (1.5, .75, 0), (3, 2, 0), (0, 2, 0)),      # non-convex support
                    ((1, 0, 0), (3, 0, 0), (3, 2, 0), (1, 2, 0))):                    # the curve leaves the trim
        model = GeometryModel()
        plate = model.add_plate(model.add_points(corners))
        wall = _wall(model, ((.5, .5, 0.), (1.5, 1.5, 0.), (2.5, .5, 0.)), (0., 0., 1.))
        result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
        assert result.kind is IntersectionKind.CROSS and result.classified
        assert _algorithms(result) <= set(EXACT_PAIR_ALGORITHMS)
        assert plan_imprint(model, result, policy="connect").operation is ImprintOperation.FACE_IMPRINT


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


def test_a_wall_whose_boundary_edge_is_split_keeps_its_exact_support_its_attachments_and_can_be_cut_again(monkeypatch):
    from anygeometry import ConnectionIntent, apply_intersections, plan_intersections, to_dict
    from anygeometry.structural import AttachmentKind, AttachmentTargetKind, ParameterRange
    model = GeometryModel()
    plate = model.add_plate(model.add_points(((0., 0., 0.), (3., 0., 0.), (1.5, .75, 0.), (3., 2., 0.), (0., 2., 0.))))
    wall = _wall(model, ((.5, .5, 0.), (1.5, 1.5, 0.), (2.5, .5, 0.)), (0., 0., 1.))
    model.add_sheet((plate,))
    model.add_sheet((wall,))
    vertex = model.add_point(1.5, 1.0, .5)                                           # the middle of the wall
    u, v = model.face_local_uv(wall, model.vertex_position(vertex))
    attachment = model.add_attachment(None, AttachmentKind.VERTEX_ON_FACE, AttachmentTargetKind.FACE, wall,
                                      ParameterRange(0., 0.), (ParameterRange(u, u), ParameterRange(v, v)),
                                      source_kind="vertex", source_id=vertex)
    import anygeometry.batch_intersections as batch
    named = []
    store = batch._store_extruded_support

    def spy(model, face_id, support, check):                  # the point the attachment names just before the swap
        if face_id == wall and not named:
            parameters = [item.start for item in model.attachments[attachment].target_parameters]
            named.append(np.asarray(model.face_point(face_id, *parameters)))
        return store(model, face_id, support, check)

    monkeypatch.setattr(batch, "_store_extruded_support", spy)
    result = query_intersection(model, model.handle("face", plate), model.handle("face", wall))
    apply_imprint(model, plan_imprint(model, result, policy="connect"), policy="connect")

    face = model.faces[wall]                                  # the same face: its bottom edge is now two edges
    assert len(face.loop) > 4 and isinstance(face.surface, ExtrudedSurface)
    assert extruded_support(model, wall) is face.surface
    moved = model.attachments[attachment]
    assert moved.target_id == wall
    parameters = [item.start for item in moved.target_parameters]
    # the same point in the new chart; the topology chart of the split chain is itself off the surface by about
    # 1e-4 there, and the remap moves the point onto the exact support
    assert named and np.allclose(model.face_point(wall, *parameters), named[0], atol=1e-3)
    assert all(0. <= value <= 1. for value in parameters)
    assert to_dict(model)["version"] == 6

    before = _area(model)                                     # a second operation on the edited wall
    upper = model.add_plate(model.add_points(((-1., -1., .8), (4., -1., .8), (4., 3., .8), (-1., 3., .8))))
    plan = plan_intersections(model, [model.handle("face", wall), model.handle("face", upper)],
                              policy=ConnectionIntent.CONNECT)
    apply_intersections(model, plan, policy=ConnectionIntent.CONNECT)
    assert model.validate_topology() == ()
    assert abs(_area(model) - before - 20.) <= 1e-9                                  # only the new 5 x 4 plate is added
