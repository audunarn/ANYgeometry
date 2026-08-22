"""Package-owner contracts for declared structural intersections.

These tests deliberately stop at the ANYgeometry boundary.  Coincidence and
crossing queries never imply welding: only an explicit, qualified apply step
may change topology or create a structural relationship for downstream meshers.
"""

from __future__ import annotations

from anygeometry import (
    AttachmentKind,
    ConnectionIntent,
    GeometryModel,
    ImprintOperation,
    IntersectionKind,
    JunctionKind,
    apply_imprint,
    plan_imprint,
    query_intersection,
)


def test_crossing_plate_sheets_are_connected_only_by_explicit_apply() -> None:
    geometry = GeometryModel()
    supporting_face = geometry.add_plate(
        geometry.add_points(
            ((-2.0, -2.0, 0.0), (2.0, -2.0, 0.0), (2.0, 2.0, 0.0), (-2.0, 2.0, 0.0))
        )
    )
    crossing_face = geometry.add_plate(
        geometry.add_points(
            ((-1.0, 0.0, -1.0), (1.0, 0.0, -1.0), (1.0, 0.0, 1.0), (-1.0, 0.0, 1.0))
        )
    )
    supporting_sheet = geometry.add_sheet((supporting_face,))
    crossing_sheet = geometry.add_sheet((crossing_face,))
    revision = geometry.revision
    counts = (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    result = query_intersection(
        geometry,
        geometry.handle("face", supporting_face),
        geometry.handle("face", crossing_face),
    )
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)

    assert result.kind is IntersectionKind.CROSS
    assert plan.operation is ImprintOperation.FACE_IMPRINT
    assert geometry.revision == revision
    assert counts == (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    assert application.face_intersection is not None
    shared_edge = application.face_intersection.edge.id
    uses = tuple(
        geometry.face_uses[item]
        for item in geometry.face_uses_using_edge(shared_edge)
    )
    assert {item.sheet_id for item in uses} == {
        supporting_sheet,
        crossing_sheet,
    }
    assert set(geometry.sheets) == {supporting_sheet, crossing_sheet}
    assert geometry.validate_topology() == ()


def test_beam_shell_crossing_requires_declared_member_sheet_relation() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((-1.0, -1.0, 0.0), (1.0, -1.0, 0.0), (1.0, 1.0, 0.0), (-1.0, 1.0, 0.0))
        )
    )
    sheet = geometry.add_sheet((face,))
    member = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((0.0, 0.0, -1.0), (0.0, 0.0, 1.0)))
            ),
        )
    )
    revision = geometry.revision

    result = query_intersection(
        geometry,
        geometry.handle("member", member),
        geometry.handle("sheet", sheet),
    )
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)

    assert result.kind is IntersectionKind.CROSS
    assert plan.operation is ImprintOperation.MEMBER_SHEET_RELATION
    assert geometry.revision == revision
    assert geometry.attachments == {}
    assert geometry.junctions == {}

    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    attachments = [
        geometry.attachments[item.id]
        for item in application.relations
        if item.kind == "attachment"
    ]
    junctions = [
        geometry.junctions[item.id]
        for item in application.relations
        if item.kind == "junction"
    ]
    assert len(attachments) == 1
    assert attachments[0].kind is AttachmentKind.MEMBER_CROSS_SHEET
    assert attachments[0].member_id == member
    assert attachments[0].target_id == sheet
    assert len(junctions) == 1
    assert junctions[0].kind is JunctionKind.CROSSING
    assert set(geometry.members) == {member}
    assert set(geometry.sheets) == {sheet}


def test_crossing_beams_preserve_members_and_split_only_on_explicit_connect() -> None:
    geometry = GeometryModel()
    first = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
            ),
        )
    )
    second = geometry.add_member(
        (
            geometry.add_line(
                *geometry.add_points(((0.0, -1.0, 0.0), (0.0, 1.0, 0.0)))
            ),
        )
    )
    revision = geometry.revision
    counts = (len(geometry.vertices), len(geometry.edges))

    result = query_intersection(
        geometry,
        geometry.handle("member", first),
        geometry.handle("member", second),
    )
    plan = plan_imprint(geometry, result, policy=ConnectionIntent.CONNECT)

    assert result.kind is IntersectionKind.CROSS
    assert plan.operation is ImprintOperation.MEMBER_CONNECTION
    assert geometry.revision == revision
    assert counts == (len(geometry.vertices), len(geometry.edges))
    assert geometry.junctions == {}

    application = apply_imprint(
        geometry, plan, policy=ConnectionIntent.CONNECT
    )

    assert set(geometry.members) == {first, second}
    assert len(geometry.vertices) == 5
    assert all(
        len(geometry.members[item].edge_use_ids) == 2
        for item in (first, second)
    )
    assert len(application.relations) == 1
    junction = geometry.junctions[application.relations[0].id]
    assert junction.kind is JunctionKind.CROSSING
    assert junction.connection_intent is ConnectionIntent.CONNECT
