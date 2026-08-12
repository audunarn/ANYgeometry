"""Hostile lifecycle and scaling checks for persistent structural topology."""

from __future__ import annotations

from dataclasses import replace

import pytest

from anygeometry import EntityRef, GeometryError, GeometryModel, OrientedEdge
from anygeometry.structural import (
    Attachment,
    AttachmentKind,
    AttachmentTargetKind,
    Coedge,
    ConnectionIntent,
    JunctionMemberUse,
    ParameterRange,
)


def _generalized_source_model() -> tuple[GeometryModel, dict[str, int]]:
    model = GeometryModel()
    corners = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = model.add_plate(corners)
    part = model.add_part(name="source-validation")
    sheet = model.add_sheet((face,), part_id=part)
    edges = tuple(item.edge for item in model.faces[face].loop)
    return model, {
        "edge": edges[0],
        "face": face,
        "sheet": sheet,
        "target_edge": edges[1],
    }


def _add_generalized_attachment(
    model: GeometryModel,
    identifiers: dict[str, int],
    source_kind: str,
    source_id: int,
) -> int:
    return model.add_attachment(
        None,
        AttachmentKind.INTENTIONALLY_DISCONNECTED,
        AttachmentTargetKind.EDGE,
        identifiers["target_edge"],
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
        source_kind=source_kind,
        source_id=source_id,
        connection_intent=ConnectionIntent.KEEP_DISCONNECTED,
    )


def _owned_plate_and_member() -> tuple[GeometryModel, int, int, int, int, int]:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0, 0, 0), (2, 0, 0), (2, 2, 0), (0, 2, 0))
    )
    face = geometry.add_plate(points)
    edge = geometry.faces[face].loop[0].edge
    part = geometry.add_part()
    sheet = geometry.add_sheet((face,), part_id=part)
    member = geometry.add_member((edge,), part_id=part)
    return geometry, part, sheet, member, face, edge


def test_add_members_batches_one_part_update_and_one_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    count = 256
    points = geometry.add_points((float(index), 0.0, 0.0) for index in range(count + 1))
    edges = [
        geometry.add_line(points[index], points[index + 1])
        for index in range(count)
    ]
    part = geometry.add_part()
    validations = 0
    original = geometry._validate_structural_changed  # noqa: SLF001

    def counted_validation(journal) -> tuple[str, ...]:
        nonlocal validations
        validations += 1
        return original(journal)

    monkeypatch.setattr(
        geometry, "_validate_structural_changed", counted_validation
    )
    members = geometry.add_members(((edge,) for edge in edges), part_id=part)

    assert len(members) == count
    assert geometry.parts[part].member_ids == tuple(members)
    assert validations == 1
    assert geometry.last_structural_validation_diagnostics.visited_count < count * 6
    assert all(len(geometry._edge_member_uses[edge]) == 1 for edge in edges)  # noqa: SLF001


@pytest.mark.parametrize("source_kind", ("edge", "face", "sheet"))
def test_public_add_attachment_rolls_back_a_missing_generalized_source(
    source_kind: str,
) -> None:
    model, identifiers = _generalized_source_model()
    revision_before = model.revision
    change_before = model.last_change_set
    attachments_before = dict(model.attachments)

    with pytest.raises(GeometryError, match=rf"missing source {source_kind}"):
        _add_generalized_attachment(model, identifiers, source_kind, 999_999)

    assert dict(model.attachments) == attachments_before
    assert model.revision == revision_before
    assert model.last_change_set == change_before
    assert model.validate_topology() == ()


@pytest.mark.parametrize(
    ("kind", "target_kind", "target_parameters", "member_id", "source_kind", "expected"),
    (
        (
            AttachmentKind.MEMBER_ON_FACE,
            AttachmentTargetKind.FACE,
            (ParameterRange.point(0.5), ParameterRange.point(0.5)),
            None,
            "edge",
            "requires a member source",
        ),
        (
            AttachmentKind.ENDPOINT,
            AttachmentTargetKind.EDGE,
            (ParameterRange.point(0.5),),
            None,
            "sheet",
            "requires a member source",
        ),
        (
            AttachmentKind.VERTEX_ON_FACE,
            AttachmentTargetKind.FACE,
            (ParameterRange.point(0.5), ParameterRange.point(0.5)),
            1,
            "member",
            "requires a vertex source",
        ),
    ),
)
def test_attachment_kind_rejects_incompatible_source_classification(
    kind: AttachmentKind,
    target_kind: AttachmentTargetKind,
    target_parameters: tuple[ParameterRange, ...],
    member_id: int | None,
    source_kind: str,
    expected: str,
) -> None:
    member_range = (
        ParameterRange.point(0.0)
        if kind is AttachmentKind.ENDPOINT
        else ParameterRange.point(0.5)
    )

    with pytest.raises(GeometryError, match=expected):
        Attachment(
            1,
            member_id,
            kind,
            target_kind,
            1,
            member_range,
            target_parameters,
            source_kind=source_kind,
            source_id=1,
        )


def test_structural_removals_are_dependency_ordered_and_fail_closed() -> None:
    geometry, part, sheet, member, _face, edge = _owned_plate_and_member()
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.ENDPOINT,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.0),),
    )
    junction = geometry.add_junction(
        "endpoint",
        (JunctionMemberUse(member, ParameterRange.point(0.0)),),
        sheet_ids=(sheet,),
        attachment_ids=(attachment,),
    )
    revision = geometry.revision

    with pytest.raises(GeometryError, match="junction"):
        geometry.remove_attachment(attachment)
    with pytest.raises(GeometryError, match="attachments.*junctions"):
        geometry.remove_member(member)
    with pytest.raises(GeometryError, match="junction"):
        geometry.remove_sheet(sheet)
    with pytest.raises(GeometryError, match="non-empty part"):
        geometry.remove_part(part)

    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001

    geometry.remove_junction(junction)
    geometry.remove_attachment(attachment)
    geometry.remove_member(member)
    geometry.remove_sheet(sheet)
    geometry.remove_part(part)

    assert not geometry.parts
    assert not geometry.sheets
    assert not geometry.face_uses
    assert not geometry.coedges
    assert not geometry.members
    assert not geometry.member_edge_uses
    assert not geometry.attachments
    assert not geometry.junctions
    assert geometry._edge_member_uses == {}  # noqa: SLF001
    assert geometry._face_structural_uses == {}  # noqa: SLF001
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_failed_add_members_rolls_back_the_whole_batch_and_incidence() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    edge = geometry.add_line(first, second)
    part = geometry.add_part()
    revision = geometry.revision

    with pytest.raises(GeometryError, match="no edge 999"):
        geometry.add_members(((edge,), (999,)), part_id=part)

    assert geometry.parts[part].member_ids == ()
    assert not geometry.members
    assert not geometry.member_edge_uses
    assert geometry._edge_member_uses == {}  # noqa: SLF001
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_member_axis_cannot_retrace_the_same_geometry_edge() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    edge = geometry.add_line(first, second)
    revision = geometry.revision

    with pytest.raises(GeometryError, match="uses edge .* more than once"):
        geometry.add_member((edge, OrientedEdge(edge, False)))

    assert not geometry.members
    assert not geometry.member_edge_uses
    assert geometry._edge_member_uses == {}  # noqa: SLF001
    assert geometry.revision == revision


def test_member_sheet_junction_attachment_must_cover_its_member_witness() -> None:
    geometry, _part, sheet, member, _face, edge = _owned_plate_and_member()
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.ENDPOINT,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.0),),
    )
    revision = geometry.revision

    with pytest.raises(GeometryError, match="does not cover.*member range"):
        geometry.add_junction(
            "endpoint",
            (JunctionMemberUse(member, ParameterRange.point(1.0)),),
            sheet_ids=(sheet,),
            attachment_ids=(attachment,),
        )

    assert not geometry.junctions
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_failed_structural_removal_restores_reverse_incidence() -> None:
    geometry, _part, sheet, member, face, edge = _owned_plate_and_member()
    member_index = {key: set(value) for key, value in geometry._edge_member_uses.items()}  # noqa: SLF001
    face_index = {key: set(value) for key, value in geometry._face_structural_uses.items()}  # noqa: SLF001
    revision = geometry.revision

    with pytest.raises(RuntimeError, match="rollback"):
        with geometry.transaction():
            geometry.remove_member(member)
            geometry.remove_sheet(sheet)
            raise RuntimeError("rollback")

    assert member in geometry.members
    assert sheet in geometry.sheets
    assert face in geometry.faces
    assert edge in geometry.edges
    assert geometry._edge_member_uses == member_index  # noqa: SLF001
    assert geometry._face_structural_uses == face_index  # noqa: SLF001
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_atomic_face_replacement_transfers_sheet_ownership() -> None:
    geometry, _part, sheet, _member, old_face, _edge = _owned_plate_and_member()
    original_use_id = geometry.sheets[sheet].face_use_ids[0]
    original = geometry.faces[old_face]
    new_face = geometry.add_face(
        tuple(item.edge for item in original.loop),
        corners=original.corners,
        surface=original.surface,
    )

    with geometry.transaction():
        geometry._delete_entity("face", old_face)  # noqa: SLF001
        geometry.record_replacements_atomic(
            ((EntityRef("face", old_face), (EntityRef("face", new_face),)),)
        )

    assert geometry.face_uses[original_use_id].face_id == new_face
    assert old_face not in geometry._face_structural_uses  # noqa: SLF001
    assert geometry._face_structural_uses[new_face] == {original_use_id}  # noqa: SLF001
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_face_use_cannot_claim_a_different_closed_trim_loop() -> None:
    geometry, _part, sheet, _member, _face, _edge = _owned_plate_and_member()
    other_points = geometry.add_points(
        ((10, 0, 0), (12, 0, 0), (12, 2, 0), (10, 2, 0))
    )
    other_face = geometry.add_plate(other_points)
    face_use = geometry.face_uses[geometry.sheets[sheet].face_use_ids[0]]
    original_coedges = face_use.coedge_ids
    revision = geometry.revision

    with pytest.raises(GeometryError, match="do not match.*trim loops"):
        with geometry.transaction():
            replacement_coedges = []
            for oriented in geometry.faces[other_face].loop:
                coedge_id = geometry._allocate_structural("coedge")  # noqa: SLF001
                geometry._put_structural(  # noqa: SLF001
                    "coedge",
                    Coedge(
                        coedge_id,
                        face_use.id,
                        oriented.edge,
                        1 if oriented.forward else -1,
                    ),
                )
                replacement_coedges.append(coedge_id)
            geometry._put_structural(  # noqa: SLF001
                "face_use",
                replace(face_use, loops=(tuple(replacement_coedges),)),
            )
            for coedge_id in original_coedges:
                geometry._delete_structural("coedge", coedge_id)  # noqa: SLF001

    assert geometry.face_uses[face_use.id] == face_use
    assert all(coedge_id in geometry.coedges for coedge_id in original_coedges)
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_face_replacement_rejects_ambiguous_attachment_remap_atomically() -> None:
    geometry, _part, _sheet, member, old_face, _edge = _owned_plate_and_member()
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        old_face,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0), ParameterRange.point(0.0)),
    )
    original = geometry.faces[old_face]
    new_face = geometry.add_face(
        tuple(item.edge for item in original.loop),
        corners=original.corners,
        surface=original.surface,
    )
    revision = geometry.revision

    with pytest.raises(GeometryError, match="explicit parameter remap"):
        with geometry.transaction():
            geometry._delete_entity("face", old_face)  # noqa: SLF001
            geometry.record_replacement(
                EntityRef("face", old_face),
                (EntityRef("face", new_face),),
            )

    assert old_face in geometry.faces
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_edge_split_rejects_ambiguous_attachment_remap_before_mutating() -> None:
    geometry, _part, _sheet, member, _face, edge = _owned_plate_and_member()
    geometry.add_attachment(
        member,
        AttachmentKind.ENDPOINT,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.0),),
    )
    revision = geometry.revision
    entity_counts = (len(geometry.vertices), len(geometry.edges), len(geometry.faces))

    with pytest.raises(GeometryError, match="explicit parameter remap"):
        geometry.split_edge(edge, 0.5)

    assert edge in geometry.edges
    assert entity_counts == (
        len(geometry.vertices),
        len(geometry.edges),
        len(geometry.faces),
    )
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001
