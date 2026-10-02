"""Exact nonplanar boundary-curve CONNECT qualification."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    GeometryModel,
    ImprintOperation,
    IntersectionDimension,
    IntersectionKind,
    apply_imprint,
    plan_imprint,
    query_intersection,
    query_trimmed_surface_charts,
)
from anygeometry.arrangement_geometry import BezierPath
from anygeometry.serialization import to_dict


def _plate(geometry: GeometryModel, points) -> int:
    return geometry.add_plate([geometry.add_point(*point) for point in points])


def _spline_wall(geometry: GeometryModel) -> tuple[int, int]:
    start, control, end = geometry.add_points(
        ((0.5, 0.5, 0.0), (1.5, 1.5, 0.0), (2.5, 0.5, 0.0))
    )
    spline = geometry.add_spline(start, (control,), end)
    wall = geometry.extrude((spline,), (0.0, 0.0, 1.0))[0]
    return spline, wall


def test_complete_spline_boundary_connect_reuses_edge_and_sheet_topology() -> None:
    geometry = GeometryModel()
    support = _plate(
        geometry,
        ((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)),
    )
    spline, wall = _spline_wall(geometry)
    support_sheet = geometry.add_sheet((support,))
    wall_sheet = geometry.add_sheet((wall,))
    before = to_dict(geometry)
    revision = geometry.revision

    queried = query_intersection(
        geometry, geometry.handle("face", support), geometry.handle("face", wall)
    )
    plan = plan_imprint(geometry, queried, policy="connect")

    assert queried.kind is IntersectionKind.CONTAINED
    assert queried.dimension is IntersectionDimension.CURVE
    assert queried.components[0].second_subparent == geometry.handle("edge", spline)
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    assert geometry.revision == revision
    assert to_dict(geometry) == before

    application = apply_imprint(geometry, plan, policy="connect")

    assert application.face_intersection is not None
    assert application.face_intersection.edge.id == spline
    assert application.face_intersection.edges == (application.face_intersection.edge,)
    face_use_ids = geometry.face_uses_using_edge(spline)
    assert {
        geometry.face_uses[face_use_id].sheet_id for face_use_id in face_use_ids
    } == {support_sheet, wall_sheet}
    assert {
        geometry.face_uses[geometry.coedges[coedge_id].face_use_id].sheet_id
        for coedge_id in geometry.coedges_using_edge(spline)
    } == {support_sheet, wall_sheet}
    assert geometry.validate_topology() == ()

    support_child = next(
        reference.id
        for reference in application.face_intersection.first_faces
        if spline in {
            item.edge
            for loop in (geometry.faces[reference.id].loop,)
            + geometry.faces[reference.id].holes
            for item in loop
        }
    )
    repeated_revision = geometry.revision
    repeated = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("face", wall),
            geometry.handle("face", support_child),
            policy="connect",
        ),
        policy="connect",
    )
    assert repeated.reused
    assert repeated.face_intersection is not None
    assert repeated.face_intersection.edge.id == spline
    assert geometry.revision == repeated_revision


def test_corner_to_corner_diagonal_extrusion_splits_support_and_reuses_edge() -> None:
    geometry = GeometryModel()
    corners = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    support = geometry.add_plate(corners)
    diagonal = geometry.add_line(corners[0], corners[2])
    wall = geometry.extrude((diagonal,), (0.0, 0.0, 1.0))[0]
    support_sheet = geometry.add_sheet((support,))
    wall_sheet = geometry.add_sheet((wall,))

    result = query_intersection(
        geometry,
        geometry.handle("face", support),
        geometry.handle("face", wall),
    )
    plan = plan_imprint(geometry, result, policy="connect")
    application = apply_imprint(geometry, plan, policy="connect")

    assert result.kind is IntersectionKind.CROSS
    assert application.face_intersection is not None
    assert application.face_intersection.edge.id == diagonal
    assert application.face_intersection.edges == (
        application.face_intersection.edge,
    )
    assert len(application.face_intersection.first_faces) == 2
    assert tuple(
        reference.id for reference in application.face_intersection.second_faces
    ) == (wall,)
    assert {
        geometry.face_uses[face_use_id].sheet_id
        for face_use_id in geometry.face_uses_using_edge(diagonal)
    } == {support_sheet, wall_sheet}
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()


def test_separate_adjacent_wall_extrusions_share_coincident_boundary() -> None:
    geometry = GeometryModel()
    corners = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    support = geometry.add_plate(corners)
    support_edges = tuple(item.edge for item in geometry.faces[support].loop)
    first_wall = geometry.extrude((support_edges[0],), (0.0, 0.0, 1.0))[0]
    second_wall = geometry.extrude((support_edges[1],), (0.0, 0.0, 1.0))[0]
    first_sheet = geometry.add_sheet((first_wall,))
    second_sheet = geometry.add_sheet((second_wall,))

    application = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("face", first_wall),
            geometry.handle("face", second_wall),
            policy="connect",
        ),
        policy="connect",
    )

    assert application.face_intersection is not None
    shared_edge = application.face_intersection.edge.id
    assert shared_edge in {item.edge for item in geometry.faces[first_wall].loop}
    assert shared_edge in {item.edge for item in geometry.faces[second_wall].loop}
    assert {
        geometry.face_uses[face_use_id].sheet_id
        for face_use_id in geometry.face_uses_using_edge(shared_edge)
    } == {first_sheet, second_sheet}
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()


def _material_area(geometry: GeometryModel) -> float:
    return sum(chart.material_area for chart in query_trimmed_surface_charts(geometry).charts)


def test_a_boundary_curve_in_a_non_convex_or_trimmed_support_is_classified_by_the_exact_engine() -> None:
    """Formerly pinned as UNCLASSIFIED (the CONNECT qualification covers only a convex support that strictly
    contains the curve). The exact material engine classifies these supports; the qualified case above keeps
    its established result and partition."""
    cases = (
        # non-convex: the spline leaves the plate through the upper edge of the notch
        (((0, 0, 0), (3, 0, 0), (1.5, 0.75, 0), (3, 2, 0), (0, 2, 0)), False, (0.5, 0.5)),
        # the support's corners are the spline's end points (the curve touches the trim)
        (((0.5, 0.5, 0), (2.5, 0.5, 0), (2.5, 1.5, 0), (0.5, 1.5, 0)), True, (0.5, 0.5)),
        # the spline leaves the trim at x = 1
        (((1, 0, 0), (3, 0, 0), (3, 2, 0), (1, 2, 0)), False, (1.0, 0.875)),
    )
    for points, whole, start in cases:
        geometry = GeometryModel()
        support = _plate(geometry, points)
        _spline, wall = _spline_wall(geometry)
        geometry.add_sheet((support,))
        geometry.add_sheet((wall,))
        before = to_dict(geometry)
        revision = geometry.revision
        area = _material_area(geometry)

        result = query_intersection(
            geometry, geometry.handle("face", support), geometry.handle("face", wall)
        )
        plan = plan_imprint(geometry, result, policy="connect")

        assert result.classified and result.kind is IntersectionKind.CROSS
        assert result.dimension is IntersectionDimension.CURVE and len(result.components) == 1
        curve = result.components[0].analytic_curve
        assert isinstance(curve, BezierPath)
        samples = curve.evaluate(np.linspace(0.0, 1.0, 33))
        u = (samples[:, 0] - 0.5) / 2.0                       # the spline is (0.5 + 2u, 0.5 + 2u - 2u^2, 0)
        assert np.abs(samples[:, 1] - (0.5 + 2.0 * u - 2.0 * u * u)).max() < 1e-12
        assert np.abs(samples[:, 2]).max() == 0.0
        assert samples[0, :2] == pytest.approx(start, abs=1e-12)
        end = samples[-1]
        if whole:
            assert end[:2] == pytest.approx((2.5, 0.5), abs=1e-12)
        elif start == (1.0, 0.875):
            assert end[:2] == pytest.approx((2.5, 0.5), abs=1e-12)
        else:                                                  # on the notch's upper edge, between the plate's corners
            assert 1.5 < end[0] < 2.5
            assert (end[1] - 0.75) * 1.5 == pytest.approx((end[0] - 1.5) * 1.25, abs=1e-12)
        assert plan.operation is ImprintOperation.FACE_IMPRINT and plan.batch_plan is not None
        assert geometry.revision == revision
        assert to_dict(geometry) == before

        apply_imprint(geometry, plan, policy="connect")

        assert geometry.validate_topology() == ()
        assert geometry._validate_structural() == ()
        assert abs(_material_area(geometry) - area) <= 1e-12 * area
        assert len(geometry.faces) > 2
