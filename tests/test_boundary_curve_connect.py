"""Exact nonplanar boundary-curve CONNECT qualification."""

from __future__ import annotations

from anygeometry import (
    GeometryModel,
    ImprintOperation,
    IntersectionDimension,
    IntersectionKind,
    apply_imprint,
    plan_imprint,
    query_intersection,
)
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


def test_boundary_curve_connect_rejects_nonconvex_or_trim_touching_support() -> None:
    cases = (
        ((0, 0, 0), (3, 0, 0), (1.5, 0.75, 0), (3, 2, 0), (0, 2, 0)),
        ((0.5, 0.5, 0), (2.5, 0.5, 0), (2.5, 1.5, 0), (0.5, 1.5, 0)),
    )
    for points in cases:
        geometry = GeometryModel()
        support = _plate(geometry, points)
        _spline, wall = _spline_wall(geometry)
        before = to_dict(geometry)
        revision = geometry.revision

        result = query_intersection(
            geometry,
            geometry.handle("face", support),
            geometry.handle("face", wall),
        )
        plan = plan_imprint(geometry, result, policy="connect")

        assert result.kind is IntersectionKind.UNSUPPORTED
        assert not result.classified
        assert plan.operation is ImprintOperation.NO_TOPOLOGY
        assert geometry.revision == revision
        assert to_dict(geometry) == before
