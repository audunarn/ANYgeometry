from __future__ import annotations

import numpy as np
import pytest
from itertools import combinations_with_replacement

from anygeometry import (
    AttachmentKind,
    AttachmentTargetKind,
    Cone,
    ConnectionIntent,
    CoonsSurface,
    Cylinder,
    GeometryModel,
    GeometryError,
    ImprintOperation,
    IntersectionDimension,
    IntersectionKind,
    IntersectionQualificationPolicy,
    JunctionKind,
    JunctionMemberUse,
    TolerancePolicy,
    ParameterRange,
    OrientedEdge,
    Plane,
    RuledSurface,
    apply_imprint,
    plan_imprint,
    query_intersection,
    strict_audit,
)


def _surface_face(geometry: GeometryModel, surface: object) -> int:
    if isinstance(surface, (Cylinder, Cone)):
        lower = tuple(
            geometry.add_point(*surface.evaluate(u, 0.0))
            for u in (0.0, 0.5, 1.0)
        )
        upper = tuple(
            geometry.add_point(*surface.evaluate(u, 1.0))
            for u in (0.0, 0.5, 1.0)
        )
        bottom = geometry.add_arc(lower[0], lower[1], lower[2])
        right = geometry.add_line(lower[2], upper[2])
        top = geometry.add_arc(upper[0], upper[1], upper[2])
        left = geometry.add_line(lower[0], upper[0])
        return geometry.add_face_from_loop(
            (
                OrientedEdge(bottom, True),
                OrientedEdge(right, True),
                OrientedEdge(top, False),
                OrientedEdge(left, False),
            ),
            (0, 1, 2, 3),
            surface=surface,
        )
    corners = tuple(
        geometry.add_point(*surface.evaluate(u, v))
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    )
    return geometry.add_face(
        geometry.add_polyline(corners, close=True),
        (0, 1, 2, 3),
        surface=surface,
    )


def _supports() -> tuple[object, ...]:
    return (
        Cylinder(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            3.0,
            sweep_angle=1.4,
        ),
        Cone(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            1.0,
            3.0,
            sweep_angle=1.4,
        ),
        RuledSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.25))),
            np.asarray(((0.0, 2.0, 0.5), (2.0, 2.0, 0.75))),
        ),
        CoonsSurface(
            np.asarray(((0.0, 0.0, 0.0), (2.0, 0.0, 0.25))),
            np.asarray(((2.0, 0.0, 0.25), (2.0, 2.0, 0.75))),
            np.asarray(((0.0, 2.0, 0.5), (2.0, 2.0, 0.75))),
            np.asarray(((0.0, 0.0, 0.0), (0.0, 2.0, 0.5))),
        ),
    )


def _plane_support() -> Plane:
    return Plane(
        np.asarray((0.0, 0.0, 0.0)),
        np.asarray((2.0, 0.0, 0.0)),
        np.asarray((0.0, 2.0, 0.0)),
    )


@pytest.mark.parametrize("surface", _supports(), ids=("cylinder", "cone", "ruled", "coons"))
@pytest.mark.parametrize("curve_kind", ("line", "arc", "spline"))
def test_curve_surface_matrix_is_deterministic_and_certified(
    surface: object, curve_kind: str
) -> None:
    geometry = GeometryModel()
    face = _surface_face(geometry, surface)
    point = geometry.face_support_point(face, 0.5, 0.5)
    normal = geometry.face_normal_many(face, ((0.5, 0.5),))[0]
    tangent = geometry.face_derivatives_many(face, ((0.5, 0.5),))[0][0]
    tangent = tangent / np.linalg.norm(tangent)
    if curve_kind == "line":
        start, end = geometry.add_points((point - normal, point + normal))
        edge = geometry.add_line(start, end)
    elif curve_kind == "arc":
        start, via, end = geometry.add_points(
            (point + normal + tangent, point, point - normal + tangent)
        )
        edge = geometry.add_arc(start, via, end)
    else:
        start, control, end = geometry.add_points(
            (point - normal, point, point + normal)
        )
        edge = geometry.add_spline(start, (control,), end)

    first = query_intersection(
        geometry, geometry.handle("edge", edge), geometry.handle("face", face)
    )
    second = query_intersection(
        geometry, geometry.handle("edge", edge), geometry.handle("face", face)
    )
    reverse = query_intersection(
        geometry, geometry.handle("face", face), geometry.handle("edge", edge)
    )

    assert first == second
    assert reverse.kind is first.kind
    assert len(reverse.components) == len(first.components)
    assert reverse.certificate is not None and reverse.certificate.complete
    assert first.classified
    assert first.kind in {
        IntersectionKind.CROSS,
        IntersectionKind.TANGENT,
        IntersectionKind.TOUCH_POINT,
    }
    assert first.dimension is IntersectionDimension.POINT
    assert first.components
    assert first.certificate is not None and first.certificate.complete
    assert first.certificate.max_residual <= first.certificate.tolerance


@pytest.mark.parametrize("curve_kind", ("line", "arc", "spline"))
def test_curve_plane_completes_fifteen_family_matrix(curve_kind: str) -> None:
    geometry = GeometryModel()
    surface = _plane_support()
    face = _surface_face(geometry, surface)
    point = surface.evaluate(0.5, 0.5)
    normal = surface.normal
    tangent = surface.u_vector / np.linalg.norm(surface.u_vector)
    if curve_kind == "line":
        start, end = geometry.add_points((point - normal, point + normal))
        edge = geometry.add_line(start, end)
    elif curve_kind == "arc":
        start, via, end = geometry.add_points(
            (point + normal + tangent, point, point - normal + tangent)
        )
        edge = geometry.add_arc(start, via, end)
    else:
        start, control, end = geometry.add_points(
            (point - normal, point, point + normal)
        )
        edge = geometry.add_spline(start, (control,), end)
    forward = query_intersection(
        geometry, geometry.handle("edge", edge), geometry.handle("face", face)
    )
    reverse = query_intersection(
        geometry, geometry.handle("face", face), geometry.handle("edge", edge)
    )
    assert forward.classified and reverse.classified
    assert forward.kind is reverse.kind
    assert forward.certificate is not None and forward.certificate.complete
    assert reverse.certificate is not None and reverse.certificate.complete


def test_curve_curve_work_budget_fails_closed_without_mutation() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        (
            (-1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (-1.0, 0.5, 0.0),
            (0.0, -0.5, 0.0),
            (1.0, 0.5, 0.0),
        )
    )
    first = geometry.add_spline(vertices[0], (vertices[1],), vertices[2])
    second = geometry.add_spline(vertices[3], (vertices[4],), vertices[5])
    revision = geometry.revision
    result = query_intersection(
        geometry,
        geometry.handle("edge", first),
        geometry.handle("edge", second),
        qualification=IntersectionQualificationPolicy(max_boxes_per_pair=1),
    )
    assert result.kind is IntersectionKind.UNCLASSIFIED
    assert not result.classified
    assert geometry.revision == revision


def test_line_arc_spline_pair_matrix_is_symmetric_and_deterministic() -> None:
    geometry = GeometryModel()
    line_vertices = geometry.add_points(((-2.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    line = geometry.add_line(*line_vertices)
    arc_vertices = geometry.add_points(
        ((-1.0, -1.0, 0.0), (0.0, 0.5, 0.0), (1.0, -1.0, 0.0))
    )
    arc = geometry.add_arc(*arc_vertices)
    spline_vertices = geometry.add_points(
        ((-1.0, 1.0, 0.0), (0.0, -2.0, 0.0), (1.0, 1.0, 0.0))
    )
    spline = geometry.add_spline(
        spline_vertices[0], (spline_vertices[1],), spline_vertices[2]
    )
    edges = (line, arc, spline)
    for first in edges:
        for second in edges:
            if first == second:
                continue
            forward = query_intersection(
                geometry,
                geometry.handle("edge", first),
                geometry.handle("edge", second),
            )
            repeated = query_intersection(
                geometry,
                geometry.handle("edge", first),
                geometry.handle("edge", second),
            )
            reverse = query_intersection(
                geometry,
                geometry.handle("edge", second),
                geometry.handle("edge", first),
            )
            assert forward == repeated
            assert forward.classified and reverse.classified
            assert forward.kind is reverse.kind
            assert len(forward.components) == len(reverse.components)
            assert forward.certificate is not None
            assert forward.certificate.boxes_examined <= 200_000


@pytest.mark.parametrize(
    "first_index,second_index",
    tuple(combinations_with_replacement(range(4), 2)),
)
def test_curved_face_pair_matrix_returns_complete_typed_result(
    first_index: int, second_index: int
) -> None:
    geometry = GeometryModel()
    supports = _supports()
    first = _surface_face(geometry, supports[first_index])
    second = _surface_face(geometry, supports[second_index])
    result = query_intersection(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
    )
    repeated = query_intersection(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
    )
    reverse = query_intersection(
        geometry,
        geometry.handle("face", second),
        geometry.handle("face", first),
    )
    assert result == repeated
    assert reverse.kind is result.kind
    assert len(reverse.components) == len(result.components)
    assert reverse.certificate is not None and reverse.certificate.complete
    assert result.classified
    assert result.certificate is not None and result.certificate.complete
    assert result.certificate.boxes_examined <= 200_000


@pytest.mark.parametrize("second_surface", (_plane_support(), *_supports()))
def test_plane_support_completes_fifteen_face_pair_families(
    second_surface: object,
) -> None:
    geometry = GeometryModel()
    first = _surface_face(geometry, _plane_support())
    second = _surface_face(geometry, second_surface)
    forward = query_intersection(
        geometry, geometry.handle("face", first), geometry.handle("face", second)
    )
    reverse = query_intersection(
        geometry, geometry.handle("face", second), geometry.handle("face", first)
    )
    assert forward.classified and reverse.classified
    assert forward.kind is reverse.kind
    assert len(forward.components) == len(reverse.components)
    assert forward.certificate is not None and forward.certificate.complete
    assert reverse.certificate is not None and reverse.certificate.complete


def test_full_curved_coincidence_connects_one_face_to_both_sheets() -> None:
    geometry = GeometryModel()
    first_surface, second_surface = _supports()[0], _supports()[0]
    first = _surface_face(geometry, first_surface)
    second = _surface_face(geometry, second_surface)
    first_sheet = geometry.add_sheet((first,))
    second_sheet = geometry.add_sheet((second,))

    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=ConnectionIntent.CONNECT,
    )
    assert plan.result.kind is IntersectionKind.COINCIDENT
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    shared = [item for item in applied.relations if item.kind == "face"]
    assert len(shared) == 1
    owners = {
        geometry.face_uses[use_id].sheet_id
        for use_id in geometry._face_structural_uses[shared[0].id]
    }
    assert owners == {first_sheet, second_sheet}
    assert geometry.validate_topology() == ()


def test_contained_curved_region_connect_preserves_both_sheet_owners() -> None:
    geometry = GeometryModel()
    surface = _supports()[0]
    assert isinstance(surface, Cylinder)
    outer = _surface_face(geometry, surface)
    corners = tuple(
        geometry.add_point(*surface.evaluate(u, v))
        for u, v in (
            (0.25, 0.25),
            (0.75, 0.25),
            (0.75, 0.75),
            (0.25, 0.75),
        )
    )
    bottom_mid = geometry.add_point(*surface.evaluate(0.5, 0.25))
    top_mid = geometry.add_point(*surface.evaluate(0.5, 0.75))
    bottom = geometry.add_arc(corners[0], bottom_mid, corners[1])
    right = geometry.add_line(corners[1], corners[2])
    top = geometry.add_arc(corners[3], top_mid, corners[2])
    left = geometry.add_line(corners[0], corners[3])
    inner = geometry.add_face_from_loop(
        (
            OrientedEdge(bottom, True),
            OrientedEdge(right, True),
            OrientedEdge(top, False),
            OrientedEdge(left, False),
        ),
        (0, 1, 2, 3),
        surface=surface,
    )
    outer_sheet = geometry.add_sheet((outer,))
    inner_sheet = geometry.add_sheet((inner,))

    plan = plan_imprint(
        geometry,
        geometry.handle("face", outer),
        geometry.handle("face", inner),
        policy=ConnectionIntent.CONNECT,
    )
    assert plan.result.kind is IntersectionKind.CONTAINED
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    shared_faces = [item for item in applied.relations if item.kind == "face"]
    assert shared_faces
    assert any(
        {
            geometry.face_uses[use_id].sheet_id
            for use_id in geometry._face_structural_uses[face.id]
        }
        == {outer_sheet, inner_sheet}
        for face in shared_faces
    )
    assert geometry.validate_topology() == ()


def test_curved_region_with_hole_connects_without_mixing_trim_frames() -> None:
    from anygeometry.operations import trim_face

    geometry = GeometryModel()
    surface = _supports()[0]
    assert isinstance(surface, Cylinder)
    holed = _surface_face(geometry, surface)
    lower = tuple(
        geometry.add_point(*surface.evaluate(u, 0.4))
        for u in (0.4, 0.5, 0.6)
    )
    upper = tuple(
        geometry.add_point(*surface.evaluate(u, 0.6))
        for u in (0.4, 0.5, 0.6)
    )
    hole = (
        OrientedEdge(geometry.add_arc(lower[0], lower[1], lower[2]), True),
        OrientedEdge(geometry.add_line(lower[2], upper[2]), True),
        OrientedEdge(geometry.add_arc(upper[0], upper[1], upper[2]), False),
        OrientedEdge(geometry.add_line(lower[0], upper[0]), False),
    )
    trim_face(geometry, holed, (hole,))
    full = _surface_face(geometry, surface)
    holed_sheet = geometry.add_sheet((holed,))
    full_sheet = geometry.add_sheet((full,))

    query = query_intersection(
        geometry,
        geometry.handle("face", holed),
        geometry.handle("face", full),
    )
    assert query.kind is IntersectionKind.CONTAINED
    assert len(query.components) == 1
    assert query.components[0].first_region is not None
    assert len(query.components[0].first_region.holes) == 1
    plan = plan_imprint(geometry, query, policy=ConnectionIntent.CONNECT)
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    shared = [item for item in applied.relations if item.kind == "face"]
    assert any(
        geometry.faces[item.id].holes
        and {
            geometry.face_uses[use_id].sheet_id
            for use_id in geometry._face_structural_uses[item.id]
        }
        == {holed_sheet, full_sheet}
        for item in shared
    )
    assert geometry.validate_topology() == ()


def test_nonconvex_coincident_region_is_complete_and_connectable() -> None:
    geometry = GeometryModel()
    surface = _supports()[2]
    assert isinstance(surface, RuledSurface)
    outer = _surface_face(geometry, surface)
    uv = (
        (0.15, 0.15),
        (0.85, 0.15),
        (0.85, 0.45),
        (0.45, 0.45),
        (0.45, 0.85),
        (0.15, 0.85),
    )
    vertices = tuple(geometry.add_point(*surface.evaluate(u, v)) for u, v in uv)
    inner = geometry.add_face(
        geometry.add_polyline(vertices, close=True),
        surface=surface,
    )
    outer_sheet = geometry.add_sheet((outer,))
    inner_sheet = geometry.add_sheet((inner,))

    query = query_intersection(
        geometry,
        geometry.handle("face", outer),
        geometry.handle("face", inner),
    )
    assert query.kind is IntersectionKind.CONTAINED
    assert len(query.components) == 1
    assert query.components[0].first_region is not None
    assert len(query.components[0].first_region.outer.points) >= 6
    plan = plan_imprint(geometry, query, policy=ConnectionIntent.CONNECT)
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    assert any(
        handle.kind == "face"
        and {
            geometry.face_uses[use_id].sheet_id
            for use_id in geometry._face_structural_uses[handle.id]
        }
        == {outer_sheet, inner_sheet}
        for handle in applied.relations
    )
    assert geometry.validate_topology() == ()


def test_curve_surface_classification_is_translation_invariant() -> None:
    kinds = []
    parameters = []
    for offset in (np.zeros(3), np.asarray((1.0e6, -2.0e6, 3.0e6))):
        geometry = GeometryModel()
        surface = Cylinder(
            offset,
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0,
            3.0,
            sweep_angle=1.4,
        )
        face = _surface_face(geometry, surface)
        point = surface.evaluate(0.5, 0.5)
        normal = geometry.face_normal_many(face, ((0.5, 0.5),))[0]
        start, end = geometry.add_points((point - normal, point + normal))
        edge = geometry.add_line(start, end)
        result = query_intersection(
            geometry,
            geometry.handle("edge", edge),
            geometry.handle("face", face),
        )
        kinds.append(result.kind)
        parameters.append(result.components[0].first_parameter)
    assert kinds[0] is kinds[1]
    assert parameters[0] == pytest.approx(parameters[1], abs=1.0e-7)


def test_curve_surface_certificate_and_trace_scale_dimensionally() -> None:
    results = []
    for factor in (1.0e-6, 1.0, 1.0e6):
        geometry = GeometryModel(tolerance=TolerancePolicy().scaled(factor))
        surface = Cylinder(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray((1.0, 0.0, 0.0)),
            2.0 * factor,
            3.0 * factor,
            sweep_angle=1.4,
        )
        face = _surface_face(geometry, surface)
        point = surface.evaluate(0.5, 0.5)
        normal = geometry.face_normal_many(face, ((0.5, 0.5),))[0]
        start, end = geometry.add_points(
            (point - factor * normal, point + factor * normal)
        )
        edge = geometry.add_line(start, end)
        results.append(
            query_intersection(
                geometry,
                geometry.handle("edge", edge),
                geometry.handle("face", face),
            )
        )
    assert {item.kind for item in results} == {results[0].kind}
    parameters = [item.components[0].first_parameter for item in results]
    assert parameters[0] == pytest.approx(parameters[1], abs=1.0e-7)
    assert parameters[1] == pytest.approx(parameters[2], abs=1.0e-7)
    for factor, result in zip((1.0e-6, 1.0, 1.0e6), results):
        assert result.certificate is not None and result.certificate.complete
        assert result.certificate.max_residual <= result.certificate.tolerance
        assert result.tolerance_used == pytest.approx(
            results[1].tolerance_used * factor, rel=1.0e-8
        )


def test_curve_components_publish_typed_certified_traces() -> None:
    geometry = GeometryModel()
    face = _surface_face(geometry, _supports()[0])
    boundary = geometry.faces[face].loop[0].edge
    result = query_intersection(
        geometry,
        geometry.handle("edge", boundary),
        geometry.handle("face", face),
    )
    assert result.dimension is IntersectionDimension.CURVE
    trace = result.components[0].curve_traces[0]
    assert trace.certificate.complete
    assert trace.points == result.components[0].witnesses
    assert len(trace.first_parameter_path) == len(trace.points)
    assert len(trace.second_parameter_path) == len(trace.points)


def test_curved_face_query_clips_points_against_hole_material() -> None:
    from anygeometry.operations import trim_face

    geometry = GeometryModel()
    cylinder = _supports()[0]
    assert isinstance(cylinder, Cylinder)
    face = _surface_face(geometry, cylinder)
    lower = tuple(
        geometry.add_point(*cylinder.evaluate(u, 0.4))
        for u in (0.4, 0.5, 0.6)
    )
    upper = tuple(
        geometry.add_point(*cylinder.evaluate(u, 0.6))
        for u in (0.4, 0.5, 0.6)
    )
    hole = (
        OrientedEdge(geometry.add_arc(lower[0], lower[1], lower[2]), True),
        OrientedEdge(geometry.add_line(lower[2], upper[2]), True),
        OrientedEdge(geometry.add_arc(upper[0], upper[1], upper[2]), False),
        OrientedEdge(geometry.add_line(lower[0], upper[0]), False),
    )
    trim_face(geometry, face, (hole,))

    results = []
    for u in (0.5, 0.25):
        point = cylinder.evaluate(u, 0.5)
        normal = geometry.face_normal_many(face, ((u, 0.5),))[0]
        start, end = geometry.add_points((point - normal, point + normal))
        edge = geometry.add_line(start, end)
        results.append(
            query_intersection(
                geometry,
                geometry.handle("edge", edge),
                geometry.handle("face", face),
            )
        )
    assert results[0].kind is IntersectionKind.DISJOINT
    assert results[1].kind in {
        IntersectionKind.CROSS,
        IntersectionKind.TOUCH_POINT,
        IntersectionKind.TANGENT,
    }


def test_two_component_plane_cylinder_imprint_is_atomic_and_idempotent() -> None:
    geometry = GeometryModel()
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        sweep_angle=1.5 * np.pi,
    )
    cylinder_face = _surface_face(geometry, cylinder)
    plane = Plane(
        np.asarray((-3.0, 0.0, 0.0)),
        np.asarray((6.0, 0.0, 0.0)),
        np.asarray((0.0, 0.0, 3.0)),
    )
    plane_face = _surface_face(geometry, plane)
    first_sheet = geometry.add_sheet((cylinder_face,))
    second_sheet = geometry.add_sheet((plane_face,))

    query = query_intersection(
        geometry,
        geometry.handle("face", plane_face),
        geometry.handle("face", cylinder_face),
    )
    assert query.dimension is IntersectionDimension.CURVE
    assert len(query.components) == 2
    revision = geometry.revision
    geometry.spatial_candidates((-4.0, -4.0, -1.0), (4.0, 4.0, 4.0))
    spatial = geometry._spatial_index
    spatial_items = spatial.items
    arc_cache = dict(geometry._arc_cache)
    length_cache = dict(geometry._edge_length_cache)
    high_water = (dict(geometry._next_id), dict(geometry._next_structural_id))
    plan = plan_imprint(
        geometry, query, policy=ConnectionIntent.CONNECT
    )
    assert geometry.revision == revision
    assert geometry._spatial_index is spatial
    assert geometry._spatial_index.items == spatial_items
    assert geometry._arc_cache == arc_cache
    assert geometry._edge_length_cache == length_cache
    assert (geometry._next_id, geometry._next_structural_id) == high_water
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    assert geometry.revision == revision + 1
    assert applied.face_intersection is not None
    assert len(applied.face_intersection.edges) == 2
    for edge in applied.face_intersection.edges:
        sheets = {
            geometry.face_uses[use_id].sheet_id
            for use_id in geometry.face_uses_using_edge(edge.id)
        }
        assert sheets == {first_sheet, second_sheet}
    assert geometry.validate_topology() == ()
    repeated = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )
    assert repeated.reused and repeated.change_set.is_empty
    assert geometry.revision == revision + 1


def test_multi_component_curved_imprint_rolls_back_on_late_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import anygeometry.intersections as workflow

    geometry = GeometryModel()
    cylinder = Cylinder(
        np.zeros(3),
        np.asarray((0.0, 0.0, 1.0)),
        np.asarray((1.0, 0.0, 0.0)),
        2.0,
        3.0,
        sweep_angle=1.5 * np.pi,
    )
    cylinder_face = _surface_face(geometry, cylinder)
    plane_face = _surface_face(
        geometry,
        Plane(
            np.asarray((-3.0, 0.0, 0.0)),
            np.asarray((6.0, 0.0, 0.0)),
            np.asarray((0.0, 0.0, 3.0)),
        ),
    )
    geometry.add_sheet((cylinder_face,))
    geometry.add_sheet((plane_face,))
    plan = plan_imprint(
        geometry,
        geometry.handle("face", plane_face),
        geometry.handle("face", cylinder_face),
        policy=ConnectionIntent.CONNECT,
    )
    before = (
        geometry.revision,
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
        tuple(sorted(geometry.face_uses)),
        tuple(sorted(geometry.coedges)),
    )
    before_geometry_high_water = dict(geometry._next_id)
    before_structural_high_water = dict(geometry._next_structural_id)
    original = workflow._fragment_with_edge_chain
    calls = 0

    def fail_late(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise GeometryError("injected late curved-fragment failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(workflow, "_fragment_with_edge_chain", fail_late)
    with pytest.raises(GeometryError, match="injected late"):
        apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)
    after = (
        geometry.revision,
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(sorted(geometry.faces)),
        tuple(sorted(geometry.face_uses)),
        tuple(sorted(geometry.coedges)),
    )
    assert after == before
    # Rollback restores live state and revision, while allocator high-water
    # marks deliberately remain monotonic so provisional IDs can never be
    # rebound to different persistent entities.
    assert all(
        geometry._next_id[kind] >= value
        for kind, value in before_geometry_high_water.items()
    )
    assert all(
        geometry._next_structural_id[kind] >= value
        for kind, value in before_structural_high_water.items()
    )
    next_vertex = geometry._next_id["vertex"]
    made = geometry.add_point(20.0, 20.0, 20.0)
    assert made == next_vertex
    assert geometry.validate_topology() == ()


def test_imprint_splits_face_attachment_and_rewrites_junction_atomically() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
        )
    )
    part = geometry.add_part()
    sheet = geometry.add_sheet((face,), part_id=part)
    member_vertices = geometry.add_points(((0.0, 0.5, 0.0), (2.0, 0.5, 0.0)))
    member_edge = geometry.add_line(*member_vertices)
    member = geometry.add_member((member_edge,), part_id=part)
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.25)),
        part_id=part,
        sheet_id=sheet,
    )
    junction = geometry.add_junction(
        JunctionKind.OVERLAP,
        (JunctionMemberUse(member, ParameterRange(0.0, 1.0)),),
        sheet_ids=(sheet,),
        attachment_ids=(attachment,),
    )
    cutter = geometry.add_plate(
        geometry.add_points(
            ((1.0, 0.0, -1.0), (1.0, 2.0, -1.0), (1.0, 2.0, 1.0), (1.0, 0.0, 1.0))
        )
    )

    plan = plan_imprint(
        geometry,
        geometry.handle("face", face),
        geometry.handle("face", cutter),
        policy=ConnectionIntent.CONNECT,
    )
    apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    member_attachments = tuple(
        item
        for item in geometry.attachments.values()
        if item.member_id == member and item.target_kind is AttachmentTargetKind.FACE
    )
    assert len(member_attachments) == 2
    assert attachment in {item.id for item in member_attachments}
    assert set(geometry.junctions[junction].attachment_ids) == {
        item.id for item in member_attachments
    }
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()
