"""Qualified query/plan/apply intersection contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from anygeometry.errors import GeometryError
from anygeometry.intersections import (
    ImprintOperation,
    _canonical_member_point_components,
    apply_imprint,
    plan_imprint,
    query_intersection,
)
from anygeometry.model import GeometryModel
from anygeometry.policies import ConnectionIntent, MutationPolicy
from anygeometry.predicates import (
    IntersectionCertificate,
    IntersectionComponent,
    IntersectionDimension,
    IntersectionKind,
    IntersectionQuality,
    IntersectionResult,
)
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
)


def _crossing_faces() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    horizontal = geometry.add_plate(
        geometry.add_points(
            ((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0))
        )
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            ((-1, 0, -1), (1, 0, -1), (1, 0, 1), (-1, 0, 1))
        )
    )
    return geometry, horizontal, vertical


@pytest.mark.parametrize("invalid", (True, 1.0, "1"))
def test_query_operands_reject_noncanonical_local_ids(invalid: object) -> None:
    geometry, first, second = _crossing_faces()

    with pytest.raises(GeometryError, match="positive integer face IDs"):
        query_intersection(geometry, invalid, second)  # type: ignore[arg-type]
    with pytest.raises(GeometryError, match="positive integer"):
        query_intersection(
            geometry,
            ("face", invalid),  # type: ignore[arg-type]
            geometry.handle("face", first),
        )


def test_planning_preflights_invalid_pair_policy_combinations() -> None:
    geometry, first, second = _crossing_faces()
    edge = geometry.faces[first].loop[0].edge

    revision = geometry.revision
    with pytest.raises(GeometryError, match="does not support operand pair"):
        plan_imprint(
            geometry,
            geometry.handle("edge", edge),
            geometry.handle("face", second),
            policy=MutationPolicy.IMPRINT,
        )
    assert geometry.revision == revision
    with pytest.raises(GeometryError, match="not valid.*face/face"):
        plan_imprint(
            geometry,
            geometry.handle("face", first),
            geometry.handle("face", second),
            policy=MutationPolicy.WELD,
        )
    assert geometry.revision == revision
    with pytest.raises(GeometryError, match="not valid.*member/member"):
        vertices = geometry.add_points(((5, 0, 0), (6, 0, 0), (5.5, -1, 0), (5.5, 1, 0)))
        first_member = geometry.add_member((geometry.add_line(vertices[0], vertices[1]),))
        second_member = geometry.add_member((geometry.add_line(vertices[2], vertices[3]),))
        plan_imprint(
            geometry,
            geometry.handle("member", first_member),
            geometry.handle("member", second_member),
            policy=MutationPolicy.KEEP_SEPARATE_PART,
        )

    with pytest.raises(GeometryError, match="invalid intersection policy"):
        plan_imprint(
            geometry,
            geometry.handle("face", first),
            geometry.handle("face", second),
            policy="invent_topology",
        )


def test_disjoint_and_same_identity_plan_as_no_topology() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((5, 0, 0), (6, 0, 0), (6, 1, 0), (5, 1, 0)))
    )

    disjoint = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=MutationPolicy.IMPRINT,
    )
    same = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", first),
        policy=MutationPolicy.REUSE_EXISTING,
    )

    assert disjoint.result.kind is IntersectionKind.DISJOINT
    assert disjoint.operation is ImprintOperation.NO_TOPOLOGY
    assert disjoint.expected_changes == ()
    assert same.operation is ImprintOperation.NO_TOPOLOGY
    assert same.expected_changes == ()


def test_point_only_face_imprint_is_typed_unsupported() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, 1, 0), (2, 1, 0), (2, 1, 1), (1, 1, 1)))
    )
    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=MutationPolicy.IMPRINT,
    )

    assert plan.result.kind is IntersectionKind.UNSUPPORTED
    assert plan.result.dimension is IntersectionDimension.NONE
    assert "point-only contact" in plan.result.diagnostics[-1]
    assert plan.operation is ImprintOperation.NO_TOPOLOGY


def _crossing_members() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    first_points = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_points = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first = geometry.add_member((geometry.add_line(*first_points),))
    second = geometry.add_member((geometry.add_line(*second_points),))
    return geometry, first, second


def test_reuse_existing_absent_member_relation_never_mutates() -> None:
    geometry, first, second = _crossing_members()
    revision = geometry.revision
    counts = (
        len(geometry.vertices),
        len(geometry.edges),
        len(geometry.attachments),
        len(geometry.junctions),
    )
    plan = plan_imprint(
        geometry,
        geometry.handle("member", first),
        geometry.handle("member", second),
        policy=MutationPolicy.REUSE_EXISTING,
    )

    assert plan.operation is ImprintOperation.NO_TOPOLOGY
    assert plan.expected_changes == ()
    with pytest.raises(GeometryError, match="requires compatible existing"):
        apply_imprint(
            geometry, plan, policy=MutationPolicy.REUSE_EXISTING
        )
    assert geometry.revision == revision
    assert counts == (
        len(geometry.vertices),
        len(geometry.edges),
        len(geometry.attachments),
        len(geometry.junctions),
    )


def test_reuse_existing_member_relation_returns_existing_idempotently() -> None:
    geometry, first, second = _crossing_members()
    created = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("member", first),
            geometry.handle("member", second),
            policy=ConnectionIntent.KEEP_DISCONNECTED,
        ),
        policy=ConnectionIntent.KEEP_DISCONNECTED,
    )
    junction = created.relations[0]
    revision = geometry.revision
    plan = plan_imprint(
        geometry,
        geometry.handle("member", first),
        geometry.handle("member", second),
        policy=MutationPolicy.REUSE_EXISTING,
    )
    reused = apply_imprint(
        geometry, plan, policy=MutationPolicy.REUSE_EXISTING
    )

    assert plan.operation is ImprintOperation.MEMBER_CONNECTION
    assert reused.reused
    assert reused.relations == (junction,)
    assert reused.change_set.is_empty
    assert geometry.revision == revision


def test_member_endpoint_contact_has_endpoint_junction_kind() -> None:
    geometry = GeometryModel()
    first_points = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    second_points = geometry.add_points(((1, 0, 0), (1, 1, 0)))
    first = geometry.add_member((geometry.add_line(*first_points),))
    second = geometry.add_member((geometry.add_line(*second_points),))

    application = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("member", first),
            geometry.handle("member", second),
            policy=ConnectionIntent.CONNECT,
        ),
        policy=ConnectionIntent.CONNECT,
    )

    attachment = next(
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    )
    junction = next(
        geometry.junctions[item.id]
        for item in application.relations
        if item.kind == "junction"
    )
    assert attachment.kind is AttachmentKind.MEMBER_ENDPOINT_ON_MEMBER
    assert attachment.target_kind is AttachmentTargetKind.MEMBER
    assert junction.kind is JunctionKind.ENDPOINT
    assert junction.attachment_ids == (attachment.id,)


@pytest.mark.parametrize(
    ("member_points", "expected_attachment", "expected_junction"),
    (
        (((0, 0, -1), (0, 0, 0)), AttachmentKind.MEMBER_ENDPOINT_ON_SHEET, JunctionKind.ENDPOINT),
        (((0, 0, -1), (0, 0, 1)), AttachmentKind.MEMBER_CROSS_SHEET, JunctionKind.CROSSING),
    ),
)
def test_member_sheet_point_relations_are_precisely_classified(
    member_points: tuple[tuple[int, int, int], tuple[int, int, int]],
    expected_attachment: AttachmentKind,
    expected_junction: JunctionKind,
) -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)))
    )
    sheet = geometry.add_sheet((face,))
    member = geometry.add_member(
        (geometry.add_line(*geometry.add_points(member_points)),)
    )

    application = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("member", member),
            geometry.handle("sheet", sheet),
            policy=ConnectionIntent.CONNECT,
        ),
        policy=ConnectionIntent.CONNECT,
    )
    attachment = next(
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    )
    junction = next(
        geometry.junctions[item.id]
        for item in application.relations
        if item.kind == "junction"
    )

    assert attachment.kind is expected_attachment
    assert attachment.target_kind is AttachmentTargetKind.SHEET
    assert attachment.target_id == sheet
    assert junction.kind is expected_junction


def test_member_on_sheet_interval_uses_sheet_attachment_kind() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)))
    )
    sheet = geometry.add_sheet((face,))
    points = geometry.add_points(((-2, 0, 0), (2, 0, 0)))
    member = geometry.add_member((geometry.add_line(*points),))

    application = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("member", member),
            geometry.handle("sheet", sheet),
            policy=ConnectionIntent.CONTACT_ONLY,
        ),
        policy=ConnectionIntent.CONTACT_ONLY,
    )
    attachment = next(
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    )

    assert attachment.kind is AttachmentKind.MEMBER_ON_SHEET
    assert attachment.target_kind is AttachmentTargetKind.SHEET
    assert attachment.target_id == sheet


def test_direct_face_point_relation_retains_face_specific_kind() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)))
    )
    points = geometry.add_points(((0, 0, -1), (0, 0, 1)))
    member = geometry.add_member((geometry.add_line(*points),))

    application = apply_imprint(
        geometry,
        plan_imprint(
            geometry,
            geometry.handle("member", member),
            geometry.handle("face", face),
            policy=ConnectionIntent.CONTACT_ONLY,
        ),
        policy=ConnectionIntent.CONTACT_ONLY,
    )
    attachment = next(
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    )

    assert attachment.kind is AttachmentKind.MEMBER_THROUGH_FACE
    assert attachment.target_kind is AttachmentTargetKind.FACE
    assert attachment.target_id == face


def test_sequential_four_sheet_connect_preserves_ids_and_radial_incidence() -> None:
    geometry = GeometryModel()
    supporting_face = geometry.add_plate(
        geometry.add_points(((-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0)))
    )
    supporting_sheet = geometry.add_sheet((supporting_face,))
    sheet_ids = [supporting_sheet]
    shared_edge = None

    for extent in (1.0, 2.0, 3.0):
        terminating_face = geometry.add_plate(
            geometry.add_points(
                ((-1, 0, -extent), (1, 0, -extent), (1, 0, extent), (-1, 0, extent))
            )
        )
        sheet_ids.append(geometry.add_sheet((terminating_face,)))
        if shared_edge is not None:
            supporting_face = next(
                geometry.face_uses[use_id].face_id
                for use_id in geometry.face_uses_using_edge(shared_edge)
                if geometry.face_uses[use_id].sheet_id == supporting_sheet
            )
        application = apply_imprint(
            geometry,
            plan_imprint(
                geometry,
                geometry.handle("face", terminating_face),
                geometry.handle("face", supporting_face),
                policy=ConnectionIntent.CONNECT,
            ),
            policy=ConnectionIntent.CONNECT,
        )
        assert application.face_intersection is not None
        if shared_edge is None:
            shared_edge = application.face_intersection.edge.id
        assert application.face_intersection.edge.id == shared_edge

    assert set(sheet_ids) == set(geometry.sheets)
    assert shared_edge is not None
    radial = geometry.radial_face_uses(shared_edge)
    assert radial == geometry.radial_face_uses(shared_edge)
    assert len(radial) == 8
    assert set(radial) == set(geometry.face_uses_using_edge(shared_edge))
    assert geometry.nonmanifold_face_uses(shared_edge) == tuple(
        sorted(geometry.face_uses_using_edge(shared_edge))
    )
    # Separate one-face Sheets each have one radial use, so their default
    # fail-closed non-manifold policy is not silently weakened by the global
    # eight-use edge.
    assert geometry._validate_structural() == ()


def test_query_plan_apply_is_non_mutating_then_atomic_and_immutable() -> None:
    geometry, first, second = _crossing_faces()
    revision = geometry.revision
    result = query_intersection(
        geometry, geometry.handle("face", first), geometry.handle("face", second)
    )

    assert result.kind is IntersectionKind.CROSS
    assert result.dimension is IntersectionDimension.CURVE
    assert result.first_parent == geometry.handle("face", first)
    assert result.second_parent == geometry.handle("face", second)
    assert result.tolerance_used is not None
    assert geometry.revision == revision

    plan = plan_imprint(geometry, result, policy=MutationPolicy.IMPRINT)
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    assert [item.action for item in plan.expected_changes] == [
        "create_or_reuse",
        "fragment",
        "fragment",
    ]
    assert geometry.revision == revision
    with pytest.raises(FrozenInstanceError):
        plan.revision = 99  # type: ignore[misc]

    application = apply_imprint(
        geometry, plan, policy=MutationPolicy.IMPRINT
    )
    assert application.face_intersection is not None
    assert application.change_set.revision_before == revision
    assert application.change_set.revision_after == revision + 1
    assert len(geometry.faces) == 4


def test_face_connect_persists_shell_sheet_t_junction_topology() -> None:
    geometry = GeometryModel()
    supporting_face = geometry.add_plate(
        geometry.add_points(
            ((-2, -2, 0), (2, -2, 0), (2, 2, 0), (-2, 2, 0))
        )
    )
    terminating_face = geometry.add_plate(
        geometry.add_points(
            ((-1, 0, 0), (1, 0, 0), (1, 0, 1), (-1, 0, 1))
        )
    )
    supporting_sheet = geometry.add_sheet((supporting_face,))
    terminating_sheet = geometry.add_sheet((terminating_face,))

    result = query_intersection(
        geometry,
        geometry.handle("face", terminating_face),
        geometry.handle("face", supporting_face),
    )
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)

    assert result.kind is IntersectionKind.CROSS
    assert result.dimension is IntersectionDimension.CURVE
    assert plan.operation is ImprintOperation.FACE_IMPRINT

    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )
    assert application.face_intersection is not None
    shared_edge = application.face_intersection.edge.id
    face_uses = {
        geometry.face_uses[use_id]
        for use_id in geometry.face_uses_using_edge(shared_edge)
    }
    assert {use.sheet_id for use in face_uses} == {
        supporting_sheet,
        terminating_sheet,
    }
    assert geometry.validate_topology() == ()
    assert geometry._validate_structural() == ()

    revision = geometry.revision
    face_by_sheet = {
        geometry.face_uses[use_id].sheet_id:
        geometry.face_uses[use_id].face_id
        for use_id in geometry.face_uses_using_edge(shared_edge)
    }
    repeated_plan = plan_imprint(
        geometry,
        geometry.handle("face", face_by_sheet[terminating_sheet]),
        geometry.handle("face", face_by_sheet[supporting_sheet]),
        policy=ConnectionIntent.CONNECT,
    )
    repeated = apply_imprint(
        geometry, repeated_plan, policy=ConnectionIntent.CONNECT
    )
    assert repeated.reused
    assert repeated.change_set.is_empty
    assert geometry.revision == revision


def test_face_connect_region_creates_shared_brep_ownership() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, -1, 0), (4, -1, 0), (4, 1, 0), (1, 1, 0)))
    )
    geometry.add_sheet((first,))
    geometry.add_sheet((second,))
    first_sheet = geometry.face_uses[next(iter(geometry._face_structural_uses[first]))].sheet_id
    second_sheet = geometry.face_uses[next(iter(geometry._face_structural_uses[second]))].sheet_id

    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=ConnectionIntent.CONNECT,
    )

    assert plan.result.kind is IntersectionKind.OVERLAP_REGION
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    applied = apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)
    overlap_faces = [
        handle.id for handle in applied.relations if handle.kind == "face"
    ]
    assert overlap_faces
    for face_id in overlap_faces:
        owners = {
            geometry.face_uses[use_id].sheet_id
            for use_id in geometry._face_structural_uses[face_id]
        }
        assert owners == {first_sheet, second_sheet}
    assert geometry.validate_topology() == ()


def test_plan_staleness_policy_mismatch_and_failed_apply_do_not_mutate() -> None:
    geometry, first, second = _crossing_faces()
    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=MutationPolicy.IMPRINT,
    )
    counts = (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    with pytest.raises(GeometryError, match="does not match"):
        apply_imprint(geometry, plan, policy=MutationPolicy.REJECT)
    assert counts == (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    geometry.add_point(20, 20, 20)
    stale_counts = (len(geometry.vertices), len(geometry.edges), len(geometry.faces))
    with pytest.raises(GeometryError, match="stale"):
        apply_imprint(geometry, plan, policy=MutationPolicy.IMPRINT)
    assert stale_counts == (len(geometry.vertices), len(geometry.edges), len(geometry.faces))


def test_edge_face_query_respects_hole_and_reports_all_material_intervals() -> None:
    from anygeometry.entities import OrientedEdge
    from anygeometry.operations import trim_face

    geometry = GeometryModel()
    outer = geometry.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)))
    face = geometry.add_plate(outer)
    hole = geometry.add_points(((1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)))
    hole_edges = tuple(
        geometry.add_line(hole[index], hole[(index + 1) % 4])
        for index in range(4)
    )
    trim_face(
        geometry,
        face,
        (tuple(OrientedEdge(edge, True) for edge in hole_edges),),
    )
    endpoints = geometry.add_points(((-1, 2, 0), (5, 2, 0)))
    edge = geometry.add_line(*endpoints)

    result = query_intersection(
        geometry, geometry.handle("edge", edge), geometry.handle("face", face)
    )

    assert result.kind is IntersectionKind.OVERLAP_CURVE
    assert result.dimension is IntersectionDimension.CURVE
    assert len(result.components) == 2
    assert [
        component.first_parameter_range.start for component in result.components
    ] == pytest.approx((1 / 6, 4 / 6))
    assert [
        component.first_parameter_range.end for component in result.components
    ] == pytest.approx((2 / 6, 5 / 6))


def test_planar_backend_absence_is_typed_capability_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins

    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    )
    points = geometry.add_points(((-1, 0.5, 0), (2, 0.5, 0)))
    edge = geometry.add_line(*points)
    original_import = builtins.__import__

    def unavailable(name, *args, **kwargs):
        if name.startswith("shapely"):
            raise ImportError("injected missing planar backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable)
    result = query_intersection(
        geometry, geometry.handle("edge", edge), geometry.handle("face", face)
    )

    assert result.kind is IntersectionKind.CAPABILITY_MISSING
    assert result.dimension is IntersectionDimension.NONE
    assert not result.classified
    assert result.diagnostics == ("planar_backend_unavailable",)


def test_member_query_preserves_parent_parameters_across_edge_chain() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-2, 0, 0), (0, 0, 0), (2, 0, 0)))
    first_edges = (
        geometry.add_line(first_vertices[0], first_vertices[1]),
        geometry.add_line(first_vertices[1], first_vertices[2]),
    )
    first_member = geometry.add_member(first_edges)
    second_vertices = geometry.add_points(((1, -1, 0), (1, 1, 0)))
    second_member = geometry.add_member(
        (geometry.add_line(*second_vertices),)
    )

    result = query_intersection(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
    )

    assert result.kind is IntersectionKind.CROSS
    assert result.components[0].first_parameter == pytest.approx((0.75,))
    assert result.components[0].second_parameter == pytest.approx((0.5,))
    assert result.components[0].first_subparent == geometry.handle(
        "edge", first_edges[1]
    )


def test_same_circle_arc_overlap_is_analytically_typed() -> None:
    geometry = GeometryModel()
    first_points = geometry.add_points(((1, 0, 0), (1, 1, 0), (0, 1, 0)))
    first_arc = geometry.add_arc(*first_points)
    second_points = geometry.add_points(
        (
            tuple(geometry.sample_edge(first_arc, np.asarray((0.25,)))[0]),
            tuple(geometry.sample_edge(first_arc, np.asarray((0.625,)))[0]),
            tuple(geometry.sample_edge(first_arc, np.asarray((1.0,)))[0]),
        )
    )
    second_arc = geometry.add_arc(*second_points)

    result = query_intersection(
        geometry,
        geometry.handle("edge", first_arc),
        geometry.handle("edge", second_arc),
    )

    assert result.kind is IntersectionKind.CONTAINED
    assert result.dimension is IntersectionDimension.CURVE
    assert result.max_residual <= result.tolerance_used


def test_keep_disconnected_member_relation_is_persistent_and_idempotent() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_member = geometry.add_member((geometry.add_line(*first_vertices),))
    second_member = geometry.add_member((geometry.add_line(*second_vertices),))

    plan = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.KEEP_DISCONNECTED,
    )
    first = apply_imprint(
        geometry, plan, policy=ConnectionIntent.KEEP_DISCONNECTED
    )
    assert len(first.relations) == 1
    assert geometry.junctions[first.relations[0].id].connection_intent is (
        ConnectionIntent.KEEP_DISCONNECTED
    )
    assert len(geometry.edges) == 2

    repeat = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.KEEP_DISCONNECTED,
    )
    revision = geometry.revision
    second = apply_imprint(
        geometry, repeat, policy=ConnectionIntent.KEEP_DISCONNECTED
    )
    assert second.reused
    assert second.relations == first.relations
    assert geometry.revision == revision


def test_connect_member_crossing_splits_axes_preserves_members_and_is_idempotent() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_member = geometry.add_member((geometry.add_line(*first_vertices),))
    second_member = geometry.add_member((geometry.add_line(*second_vertices),))

    plan = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.CONNECT,
    )
    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    assert set(geometry.members) == {first_member, second_member}
    assert all(len(geometry.members[item].edge_use_ids) == 2 for item in (first_member, second_member))
    assert len(geometry.vertices) == 5
    assert len(application.relations) == 1
    assert geometry.junctions[application.relations[0].id].connection_intent is ConnectionIntent.CONNECT

    repeat = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.CONNECT,
    )
    revision = geometry.revision
    repeated = apply_imprint(
        geometry, repeat, policy=ConnectionIntent.CONNECT
    )
    assert repeated.reused
    assert geometry.revision == revision


def test_fragmented_members_share_one_connectable_internal_vertex_component() -> None:
    geometry = GeometryModel()
    left, centre, right, lower, upper = geometry.add_points(
        ((-1, 0, 0), (0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0))
    )
    first_member = geometry.add_member(
        (geometry.add_line(left, centre), geometry.add_line(centre, right))
    )
    second_member = geometry.add_member(
        (geometry.add_line(lower, centre), geometry.add_line(centre, upper))
    )
    geometry_counts = (len(geometry.vertices), len(geometry.edges))

    result = query_intersection(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
    )

    assert result.kind is IntersectionKind.TOUCH_POINT
    assert result.dimension is IntersectionDimension.POINT
    assert len(result.components) == 1
    assert result.components[0].witnesses == ((0.0, 0.0, 0.0),)
    assert result.components[0].first_parameter == pytest.approx((0.5,))
    assert result.components[0].second_parameter == pytest.approx((0.5,))

    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
    assert plan.operation is ImprintOperation.MEMBER_CONNECTION
    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    assert geometry_counts == (len(geometry.vertices), len(geometry.edges))
    assert len(application.relations) == 1
    junction = geometry.junctions[application.relations[0].id]
    assert junction.kind is JunctionKind.CROSSING
    assert geometry.validate_topology() == ()


def test_fragmented_member_point_is_symmetric_after_member_reversal() -> None:
    geometry = GeometryModel()
    left, centre, right, lower, upper = geometry.add_points(
        ((-2, 0, 0), (0, 0, 0), (1, 0, 0), (0, -1, 0), (0, 3, 0))
    )
    first_member = geometry.add_member(
        (geometry.add_line(left, centre), geometry.add_line(centre, right))
    )
    second_member = geometry.add_member(
        (geometry.add_line(lower, centre), geometry.add_line(centre, upper))
    )
    first_handle = geometry.handle("member", first_member)
    second_handle = geometry.handle("member", second_member)

    forward = query_intersection(geometry, first_handle, second_handle)
    swapped = query_intersection(geometry, second_handle, first_handle)

    assert len(forward.components) == len(swapped.components) == 1
    assert forward.components[0].first_parameter == pytest.approx((2 / 3,))
    assert forward.components[0].second_parameter == pytest.approx((1 / 4,))
    assert swapped.components[0].first_parameter == pytest.approx(
        forward.components[0].second_parameter
    )
    assert swapped.components[0].second_parameter == pytest.approx(
        forward.components[0].first_parameter
    )
    assert swapped.components[0].witnesses == forward.components[0].witnesses
    for result in (forward, swapped):
        assert plan_imprint(
            geometry, result, policy=ConnectionIntent.CONNECT
        ).operation is ImprintOperation.MEMBER_CONNECTION

    geometry.reverse_member(first_member)
    geometry.reverse_member(second_member)
    reversed_forward = query_intersection(geometry, first_handle, second_handle)
    reversed_swapped = query_intersection(geometry, second_handle, first_handle)

    assert len(reversed_forward.components) == len(reversed_swapped.components) == 1
    assert reversed_forward.components[0].first_parameter == pytest.approx((1 / 3,))
    assert reversed_forward.components[0].second_parameter == pytest.approx((3 / 4,))
    assert reversed_swapped.components[0].first_parameter == pytest.approx(
        reversed_forward.components[0].second_parameter
    )
    assert reversed_swapped.components[0].second_parameter == pytest.approx(
        reversed_forward.components[0].first_parameter
    )
    assert (
        reversed_swapped.components[0].witnesses
        == reversed_forward.components[0].witnesses
    )
    for result in (reversed_forward, reversed_swapped):
        assert plan_imprint(
            geometry, result, policy=ConnectionIntent.CONNECT
        ).operation is ImprintOperation.MEMBER_CONNECTION


def test_fragmented_member_query_fails_closed_on_unresolved_edge_pair() -> None:
    geometry = GeometryModel()
    lower, centre, upper, near_parallel_end, left, right = geometry.add_points(
        (
            (0, -1, 0),
            (0, 0, 0),
            (0, 1, 0),
            (1, 1.000000001, 0),
            (-1, 0, 0),
            (1, 0, 0),
        )
    )
    first_edges = (
        geometry.add_line(lower, centre),
        geometry.add_line(centre, upper),
        geometry.add_line(upper, near_parallel_end),
    )
    second_edges = (
        geometry.add_line(left, centre),
        geometry.add_line(centre, right),
    )
    first_member = geometry.add_member(first_edges)
    second_member = geometry.add_member(second_edges)
    first_handle = geometry.handle("member", first_member)
    second_handle = geometry.handle("member", second_member)
    local_kinds = tuple(
        query_intersection(
            geometry,
            geometry.handle("edge", first_edge),
            geometry.handle("edge", second_edge),
        ).kind
        for first_edge in first_edges
        for second_edge in second_edges
    )

    assert local_kinds.count(IntersectionKind.TOUCH_POINT) == 4
    assert local_kinds.count(IntersectionKind.UNCLASSIFIED) == 2

    for first, second in (
        (first_handle, second_handle),
        (second_handle, first_handle),
    ):
        result = query_intersection(geometry, first, second)

        assert result.kind is IntersectionKind.UNCLASSIFIED
        assert result.dimension is IntersectionDimension.NONE
        assert not result.components
        assert not result.classified
        assert result.certificate is not None
        assert not result.certificate.complete
        assert result.certificate.algorithm == "member_axis_edge_pair_incomplete"
        assert "member_axis_edge_pair_incomplete" in result.diagnostics
        assert "ill_conditioned_segment_segment" in result.diagnostics
        assert any(" unclassified" in item for item in result.diagnostics)

        plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
        assert plan.operation is ImprintOperation.NO_TOPOLOGY
        assert plan.operation is not ImprintOperation.MEMBER_CONNECTION


def test_member_component_tolerance_chain_remains_expanded_and_fail_closed() -> None:
    geometry = GeometryModel()
    first_member = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((-1, 0, 0), (1, 0, 0)))
            ),
        )
    )
    second_member = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((0, -1, 0), (0, 1, 0)))
            ),
        )
    )
    parameter_tolerance = geometry.tolerance.effective_parameter(2.0, 2.0)
    base = 0.25

    def component(first: float, second: float) -> IntersectionComponent:
        return IntersectionComponent(
            ((0.0, 0.0, 0.0),),
            IntersectionQuality.EXACT,
            first_parameter=(first,),
            second_parameter=(second,),
        )

    source = (
        component(base, base),
        component(
            base + 0.4 * parameter_tolerance,
            base + 1.6 * parameter_tolerance,
        ),
        component(
            base + 0.8 * parameter_tolerance,
            base + 0.8 * parameter_tolerance,
        ),
    )
    forward = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        source,
        geometry.tolerance.length,
    )
    swapped_source = tuple(
        sorted(
            (
                replace(
                    item,
                    first_parameter=item.second_parameter,
                    second_parameter=item.first_parameter,
                )
                for item in source
            ),
            key=lambda item: (item.first_parameter, item.second_parameter),
        )
    )
    swapped = _canonical_member_point_components(
        geometry,
        second_member,
        first_member,
        swapped_source,
        geometry.tolerance.length,
    )

    assert len(forward) == len(swapped) == 3
    result_pairs = (
        (
            geometry.handle("member", first_member),
            geometry.handle("member", second_member),
            forward,
        ),
        (
            geometry.handle("member", second_member),
            geometry.handle("member", first_member),
            swapped,
        ),
    )
    for first_parent, second_parent, components in result_pairs:
        result = IntersectionResult(
            IntersectionKind.CROSS,
            components,
            first_parent=first_parent,
            second_parent=second_parent,
            tolerance_used=geometry.tolerance.length,
        )
        plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
        assert plan.operation is ImprintOperation.NO_TOPOLOGY
        assert plan.result.kind is IntersectionKind.UNSUPPORTED


def test_single_member_point_component_preserves_certificate_provenance() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_edge = geometry.add_line(*first_vertices)
    second_edge = geometry.add_line(*second_vertices)
    first_member = geometry.add_member((first_edge,))
    second_member = geometry.add_member((second_edge,))
    tolerance = geometry.tolerance.length
    certificate = IntersectionCertificate(
        "source_member_point_certificate",
        tolerance,
        max_residual=0.5 * tolerance,
        max_enclosure_width=0.25 * tolerance,
        boxes_examined=3,
        subdivisions=2,
        trace_segments=1,
    )
    component = IntersectionComponent(
        ((0.0, 0.0, 0.0),),
        IntersectionQuality.VERIFIED_APPROXIMATE,
        first_parameter=(0.5,),
        second_parameter=(0.5,),
        max_residual=0.25 * tolerance,
        certificate=certificate,
        first_subparent=geometry.handle("edge", first_edge),
        second_subparent=geometry.handle("edge", second_edge),
    )

    canonical = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        (component,),
        tolerance,
    )

    assert canonical == (component,)
    assert canonical[0] is component
    assert canonical[0].certificate is certificate
    assert canonical[0].certificate.algorithm == "source_member_point_certificate"


def test_member_point_approximate_merge_bounds_displacement_and_is_symmetric() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_edge = geometry.add_line(*first_vertices)
    second_edge = geometry.add_line(*second_vertices)
    first_member = geometry.add_member((first_edge,))
    second_member = geometry.add_member((second_edge,))
    tolerance = geometry.tolerance.length
    displacement = 0.5 * tolerance

    def component(witness: tuple[float, float, float]) -> IntersectionComponent:
        return IntersectionComponent(
            (witness,),
            IntersectionQuality.EXACT,
            first_parameter=(0.25,),
            second_parameter=(0.75,),
            first_subparent=geometry.handle("edge", first_edge),
            second_subparent=geometry.handle("edge", second_edge),
        )

    source = (
        component((0.0, 0.0, 0.0)),
        component((displacement, 0.0, 0.0)),
    )
    forward = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        source,
        tolerance,
    )
    reversed_input = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        tuple(reversed(source)),
        tolerance,
    )
    swapped_source = tuple(
        replace(
            item,
            first_parameter=item.second_parameter,
            second_parameter=item.first_parameter,
            first_subparent=item.second_subparent,
            second_subparent=item.first_subparent,
        )
        for item in source
    )
    swapped = _canonical_member_point_components(
        geometry,
        second_member,
        first_member,
        swapped_source,
        tolerance,
    )

    assert forward == reversed_input
    assert len(forward) == len(swapped) == 1
    for made in (*forward, *swapped):
        assert made.witnesses == ((0.0, 0.0, 0.0),)
        assert made.quality is IntersectionQuality.VERIFIED_APPROXIMATE
        assert made.max_residual == pytest.approx(displacement)
        assert made.certificate is not None
        assert made.certificate.complete
        assert made.certificate.max_residual == pytest.approx(displacement)
        assert made.certificate.max_enclosure_width == pytest.approx(displacement)
    assert swapped[0].certificate == forward[0].certificate
    assert swapped[0].first_parameter == forward[0].second_parameter
    assert swapped[0].second_parameter == forward[0].first_parameter
    assert swapped[0].first_subparent == forward[0].second_subparent
    assert swapped[0].second_subparent == forward[0].first_subparent


def test_member_point_certificate_aggregation_is_conservative_and_ordered() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_edge = geometry.add_line(*first_vertices)
    second_edge = geometry.add_line(*second_vertices)
    first_member = geometry.add_member((first_edge,))
    second_member = geometry.add_member((second_edge,))
    tolerance = geometry.tolerance.length
    first_certificate = IntersectionCertificate(
        "first_source_certificate",
        tolerance,
        boxes_examined=2,
        subdivisions=3,
        trace_segments=5,
    )
    second_certificate = IntersectionCertificate(
        "second_source_certificate",
        tolerance,
        max_residual=0.8 * tolerance,
        max_enclosure_width=0.2 * tolerance,
        boxes_examined=7,
        subdivisions=11,
        trace_segments=13,
        complete=False,
    )

    def component(
        witness: tuple[float, float, float],
        quality: IntersectionQuality,
        residual: float,
        certificate: IntersectionCertificate,
    ) -> IntersectionComponent:
        return IntersectionComponent(
            (witness,),
            quality,
            first_parameter=(0.5,),
            second_parameter=(0.5,),
            max_residual=residual,
            certificate=certificate,
            first_subparent=geometry.handle("edge", first_edge),
            second_subparent=geometry.handle("edge", second_edge),
        )

    source = (
        component(
            (0.0, 0.0, 0.0),
            IntersectionQuality.EXACT,
            0.0,
            first_certificate,
        ),
        component(
            (0.1 * tolerance, 0.0, 0.0),
            IntersectionQuality.VERIFIED_APPROXIMATE,
            0.2 * tolerance,
            second_certificate,
        ),
    )
    forward = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        source,
        tolerance,
    )
    reversed_input = _canonical_member_point_components(
        geometry,
        first_member,
        second_member,
        tuple(reversed(source)),
        tolerance,
    )

    assert forward == reversed_input
    assert len(forward) == 1
    made = forward[0]
    assert made.quality is IntersectionQuality.VERIFIED_APPROXIMATE
    assert made.max_residual == pytest.approx(0.9 * tolerance)
    assert made.certificate is not None
    assert made.certificate.algorithm == "canonical_member_point_component"
    assert made.certificate.max_residual == pytest.approx(0.9 * tolerance)
    assert made.certificate.max_enclosure_width == pytest.approx(0.3 * tolerance)
    assert made.certificate.boxes_examined == 9
    assert made.certificate.subdivisions == 14
    assert made.certificate.trace_segments == 18
    assert not made.certificate.complete


def test_same_world_point_distinct_member_visits_remain_separate() -> None:
    geometry = GeometryModel()
    left, right, lower, centre, upper_right, upper_left, upper = (
        geometry.add_points(
            (
                (-2, 0, 0),
                (2, 0, 0),
                (0, -2, 0),
                (0, 0, 0),
                (1, 1, 0),
                (-1, 1, 0),
                (0, 2, 0),
            )
        )
    )
    first_member = geometry.add_member((geometry.add_line(left, right),))
    second_path = (lower, centre, upper_right, upper_left, centre, upper)
    second_member = geometry.add_member(
        tuple(
            geometry.add_line(first, second)
            for first, second in zip(second_path, second_path[1:])
        )
    )
    first_handle = geometry.handle("member", first_member)
    second_handle = geometry.handle("member", second_member)

    forward = query_intersection(geometry, first_handle, second_handle)
    swapped = query_intersection(geometry, second_handle, first_handle)

    assert forward.kind is swapped.kind is IntersectionKind.TOUCH_POINT
    assert len(forward.components) == len(swapped.components) == 2
    assert all(
        component.witnesses == ((0.0, 0.0, 0.0),)
        for component in (*forward.components, *swapped.components)
    )
    total_length = 6.0 + 2.0 * np.sqrt(2.0)
    visits = (2.0 / total_length, (4.0 + 2.0 * np.sqrt(2.0)) / total_length)
    assert [
        component.second_parameter[0] for component in forward.components
    ] == pytest.approx(visits)
    assert [
        component.first_parameter[0] for component in swapped.components
    ] == pytest.approx(visits)
    for result in (forward, swapped):
        plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
        assert plan.operation is ImprintOperation.NO_TOPOLOGY
        assert plan.result.kind is IntersectionKind.UNSUPPORTED


def test_distinct_member_crossings_remain_separate_and_fail_closed() -> None:
    geometry = GeometryModel()
    first_member = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((-2, 0, 0), (2, 0, 0)))
            ),
        )
    )
    zigzag = geometry.add_points(
        ((-1, -1, 0), (-0.5, 1, 0), (0.5, 1, 0), (1, -1, 0))
    )
    second_member = geometry.add_member(
        tuple(
            geometry.add_line(first, second)
            for first, second in zip(zigzag, zigzag[1:])
        )
    )

    result = query_intersection(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
    )

    assert result.kind is IntersectionKind.CROSS
    assert result.dimension is IntersectionDimension.POINT
    assert len(result.components) == 2
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)
    assert plan.operation is ImprintOperation.NO_TOPOLOGY
    assert plan.result.kind is IntersectionKind.UNSUPPORTED
    assert "exactly one qualified intersection component" in plan.result.diagnostics[-1]


def test_connected_member_overlap_is_typed_unsupported_during_planning() -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((0, 0, 0), (2, 0, 0)))
    second_vertices = geometry.add_points(((1, 0, 0), (3, 0, 0)))
    first_member = geometry.add_member((geometry.add_line(*first_vertices),))
    second_member = geometry.add_member((geometry.add_line(*second_vertices),))

    plan = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.CONNECT,
    )

    assert plan.result.kind is IntersectionKind.UNSUPPORTED
    assert plan.operation is ImprintOperation.NO_TOPOLOGY
    assert "overlap-topology" in plan.result.diagnostics[-1]
    revision = geometry.revision
    with pytest.raises(GeometryError, match="unqualified intersection plan"):
        apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)
    assert geometry.revision == revision


def test_face_imprint_with_member_owned_boundary_is_typed_unsupported() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, -1, 0), (4, -1, 0), (4, 1, 0), (1, 1, 0)))
    )
    boundary_edge = geometry.faces[first].loop[0].edge
    geometry.add_member((boundary_edge,))

    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=MutationPolicy.IMPRINT,
    )

    assert plan.result.kind is IntersectionKind.UNSUPPORTED
    assert plan.operation is ImprintOperation.NO_TOPOLOGY
    assert "member owners" in plan.result.diagnostics[-1]
    revision = geometry.revision
    with pytest.raises(GeometryError, match="unqualified intersection plan"):
        apply_imprint(geometry, plan, policy=MutationPolicy.IMPRINT)
    assert geometry.revision == revision


def test_connected_member_apply_rolls_back_when_second_split_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    first_vertices = geometry.add_points(((-1, 0, 0), (1, 0, 0)))
    second_vertices = geometry.add_points(((0, -1, 0), (0, 1, 0)))
    first_member = geometry.add_member((geometry.add_line(*first_vertices),))
    second_member = geometry.add_member((geometry.add_line(*second_vertices),))
    plan = plan_imprint(
        geometry,
        geometry.handle("member", first_member),
        geometry.handle("member", second_member),
        policy=ConnectionIntent.CONNECT,
    )
    before = (
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(geometry.members[first_member].edge_use_ids),
        tuple(geometry.members[second_member].edge_use_ids),
        geometry.revision,
    )
    original = geometry.split_edge
    calls = 0

    def interrupted(edge_id: int, t: float = 0.5):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected second split failure")
        return original(edge_id, t)

    monkeypatch.setattr(geometry, "split_edge", interrupted)
    with pytest.raises(RuntimeError, match="injected"):
        apply_imprint(geometry, plan, policy=ConnectionIntent.CONNECT)

    after = (
        tuple(sorted(geometry.vertices)),
        tuple(sorted(geometry.edges)),
        tuple(geometry.members[first_member].edge_use_ids),
        tuple(geometry.members[second_member].edge_use_ids),
        geometry.revision,
    )
    assert after == before


def test_member_sheet_hole_crossing_records_only_material_intervals() -> None:
    from anygeometry.entities import OrientedEdge
    from anygeometry.operations import trim_face

    geometry = GeometryModel()
    outer = geometry.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)))
    face = geometry.add_plate(outer)
    hole = geometry.add_points(((1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)))
    hole_edges = tuple(
        geometry.add_line(hole[index], hole[(index + 1) % 4])
        for index in range(4)
    )
    trim_face(
        geometry,
        face,
        (tuple(OrientedEdge(edge, True) for edge in hole_edges),),
    )
    sheet = geometry.add_sheet((face,))
    points = geometry.add_points(((-1, 2, 0), (5, 2, 0)))
    member = geometry.add_member((geometry.add_line(*points),))

    result = query_intersection(
        geometry, geometry.handle("member", member), geometry.handle("sheet", sheet)
    )
    assert result.kind is IntersectionKind.OVERLAP_CURVE
    assert len(result.components) == 2

    plan = plan_imprint(
        geometry, result, policy=ConnectionIntent.CONTACT_ONLY
    )
    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONTACT_ONLY
    )
    attachments = [
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    ]
    assert len(attachments) == 2
    assert all(item.connection_intent is ConnectionIntent.CONTACT_ONLY for item in attachments)
    assert [item.member_range.start for item in attachments] == pytest.approx((1 / 6, 4 / 6))
    assert [item.member_range.end for item in attachments] == pytest.approx((2 / 6, 5 / 6))


def test_coplanar_region_imprint_fragments_atomically_and_reapply_reuses() -> None:
    geometry = GeometryModel()
    first = geometry.add_plate(
        geometry.add_points(((0, 0, 0), (3, 0, 0), (3, 2, 0), (0, 2, 0)))
    )
    second = geometry.add_plate(
        geometry.add_points(((1, -1, 0), (4, -1, 0), (4, 1, 0), (1, 1, 0)))
    )
    plan = plan_imprint(
        geometry,
        geometry.handle("face", first),
        geometry.handle("face", second),
        policy=MutationPolicy.IMPRINT,
    )
    assert plan.result.kind is IntersectionKind.OVERLAP_REGION
    assert plan.result.dimension is IntersectionDimension.REGION

    application = apply_imprint(
        geometry, plan, policy=MutationPolicy.IMPRINT
    )
    assert application.face_intersection is None
    assert application.relations
    assert all(item.kind == "face" for item in application.relations)
    assert geometry.validate_topology() == ()

    revision = geometry.revision
    repeated = apply_imprint(
        geometry, plan, policy=MutationPolicy.IMPRINT
    )
    assert repeated.reused
    assert repeated.change_set.is_empty
    assert geometry.revision == revision


def test_multi_component_face_imprint_with_hole_creates_all_shared_edges() -> None:
    from anygeometry.entities import OrientedEdge
    from anygeometry.operations import trim_face

    geometry = GeometryModel()
    outer = geometry.add_points(((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0)))
    horizontal = geometry.add_plate(outer)
    hole = geometry.add_points(((1, 1, 0), (3, 1, 0), (3, 3, 0), (1, 3, 0)))
    hole_edges = tuple(
        geometry.add_line(hole[index], hole[(index + 1) % 4])
        for index in range(4)
    )
    trim_face(
        geometry,
        horizontal,
        (tuple(OrientedEdge(edge, True) for edge in hole_edges),),
    )
    vertical = geometry.add_plate(
        geometry.add_points(
            ((-1, 2, -1), (5, 2, -1), (5, 2, 1), (-1, 2, 1))
        )
    )
    plan = plan_imprint(
        geometry,
        geometry.handle("face", horizontal),
        geometry.handle("face", vertical),
        policy=MutationPolicy.IMPRINT,
    )
    assert plan.result.kind is IntersectionKind.CROSS
    assert len(plan.result.components) == 2

    application = apply_imprint(
        geometry, plan, policy=MutationPolicy.IMPRINT
    )
    result = application.face_intersection
    assert result is not None
    assert len(result.edges) == 2
    assert all(len(geometry.faces_using_edge(item.id)) >= 2 for item in result.edges)
    assert geometry.validate_topology() == ()

    revision = geometry.revision
    repeated = apply_imprint(
        geometry, plan, policy=MutationPolicy.IMPRINT
    )
    assert repeated.reused
    assert repeated.face_intersection is not None
    assert repeated.face_intersection.edges == result.edges
    assert geometry.revision == revision


def test_member_sheet_connect_persists_attachment_and_sheet_junction() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(((-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)))
    )
    sheet = geometry.add_sheet((face,))
    member_points = geometry.add_points(((0, 0, -1), (0, 0, 1)))
    member = geometry.add_member((geometry.add_line(*member_points),))

    plan = plan_imprint(
        geometry,
        geometry.handle("member", member),
        geometry.handle("sheet", sheet),
        policy=ConnectionIntent.CONNECT,
    )
    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    attachments = tuple(
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    )
    junctions = tuple(
        geometry.junctions[item.id]
        for item in application.relations
        if item.kind == "junction"
    )
    assert len(attachments) == len(junctions) == 1
    assert attachments[0].connection_intent is ConnectionIntent.CONNECT
    assert attachments[0].sheet_id == sheet
    assert junctions[0].connection_intent is ConnectionIntent.CONNECT
    assert junctions[0].sheet_ids == (sheet,)
