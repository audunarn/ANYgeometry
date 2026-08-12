"""Gap-closure coverage for identity-safe model insertion and copying."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import (
    AttachmentEvidence,
    AttachmentKind,
    AttachmentTargetKind,
    ConnectionIntent,
    EntityRef,
    GeometryError,
    GeometryModel,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
    Plane,
    copy_entities,
    insert_model,
    reverse_face,
)


def _plate(model: GeometryModel) -> tuple[tuple[int, ...], int]:
    vertices = tuple(
        model.add_points(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    return vertices, model.add_plate(vertices)


def _full_contract_model() -> tuple[GeometryModel, dict[str, int]]:
    model = GeometryModel()
    corners, face = _plate(model)
    model.set_face_parameterization(
        face,
        Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    part = model.add_part(name="panel", metadata={"owner": "contract"})
    sheet = model.add_sheet((face,), part_id=part, name="plating")
    axis_start, axis_end = model.add_points(
        ((0.25, 0.5, 0.0), (1.75, 0.5, 0.0))
    )
    axis = model.add_line(axis_start, axis_end)
    first_member = model.add_member(
        (axis,),
        part_id=part,
        name="primary",
        metadata={"section": "T"},
        orientation_reference=EntityRef("vertex", corners[0]),
    )
    second_member = model.add_member(
        (axis,),
        part_id=part,
        name="secondary",
        orientation_reference=EntityRef("face", face),
    )
    construction = model.add_point(3.0, 3.0, 0.0)
    model.mark_construction_vertices((construction,), part_id=part)

    member_sheet = model.add_attachment(
        first_member,
        AttachmentKind.MEMBER_ON_SHEET,
        AttachmentTargetKind.SHEET,
        sheet,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.1, 0.9), ParameterRange.point(0.5)),
        connection_intent=ConnectionIntent.CONTACT_ONLY,
        evidence=AttachmentEvidence.EXACT,
        max_residual=1.0e-8,
        tolerance_used=2.0e-8,
        part_id=part,
        sheet_id=sheet,
        provenance={"classifier": "qualified"},
        lineage=(
            ("member", first_member),
            ("face", face),
            ("part", part),
        ),
        metadata={"token": "member-sheet"},
    )
    model.add_attachment(
        first_member,
        AttachmentKind.COINCIDENT_MEMBER_AXES,
        AttachmentTargetKind.MEMBER,
        second_member,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
        evidence=AttachmentEvidence.VERIFIED_APPROXIMATE,
        tolerance_used=1.0e-8,
        metadata={"token": "member-member"},
    )
    model.add_attachment(
        None,
        AttachmentKind.VERTEX_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.0), ParameterRange.point(0.0)),
        source_kind="vertex",
        source_id=corners[0],
        evidence=AttachmentEvidence.EXACT,
        metadata={"token": "vertex-face"},
    )
    boundary = model.faces[face].loop[0].edge
    model.add_attachment(
        None,
        AttachmentKind.INTENTIONALLY_DISCONNECTED,
        AttachmentTargetKind.EDGE,
        boundary,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
        source_kind="face",
        source_id=face,
        connection_intent=ConnectionIntent.KEEP_DISCONNECTED,
        metadata={"token": "face-edge"},
    )
    model.add_attachment(
        None,
        AttachmentKind.INTENTIONALLY_DISCONNECTED,
        AttachmentTargetKind.VERTEX,
        corners[1],
        ParameterRange(0.0, 1.0),
        (),
        source_kind="sheet",
        source_id=sheet,
        connection_intent=ConnectionIntent.KEEP_DISCONNECTED,
        metadata={"token": "sheet-vertex"},
    )
    junction = model.add_junction(
        JunctionKind.OVERLAP,
        (JunctionMemberUse(first_member, ParameterRange(0.0, 1.0)),),
        sheet_ids=(sheet,),
        attachment_ids=(member_sheet,),
        connection_intent=ConnectionIntent.CONTACT_ONLY,
        provenance={"classifier": "qualified"},
        metadata={"token": "junction"},
    )
    assert model.validate_topology() == ()
    return model, {
        "face": face,
        "part": part,
        "sheet": sheet,
        "first_member": first_member,
        "second_member": second_member,
        "construction": construction,
        "corner": corners[0],
        "boundary": boundary,
        "member_sheet": member_sheet,
        "junction": junction,
    }


def _by_token(values, token: str):
    return next(value for value in values if value.metadata.get("token") == token)


def test_insert_remaps_and_preserves_complete_additive_contract_records() -> None:
    source, ids = _full_contract_model()
    destination = GeometryModel()
    _existing_corners, existing_face = _plate(destination)
    existing_part = destination.add_part(name="existing")
    destination.add_sheet((existing_face,), part_id=existing_part, name="existing")
    existing_edge = destination.faces[existing_face].loop[0].edge
    existing_member = destination.add_member(
        (existing_edge,), part_id=existing_part, name="existing"
    )
    destination.add_attachment(
        existing_member,
        AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE,
        existing_edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
    )

    result = insert_model(destination, source)

    mapped_face = result.mapped(EntityRef("face", ids["face"])).id
    mapped_corner = result.mapped(EntityRef("vertex", ids["corner"])).id
    mapped_construction = result.mapped(
        EntityRef("vertex", ids["construction"])
    ).id
    mapped_boundary = result.mapped(EntityRef("edge", ids["boundary"])).id
    imported_part = next(value for value in destination.parts.values() if value.name == "panel")
    imported_sheet = next(
        value for value in destination.sheets.values() if value.name == "plating"
    )
    primary = next(
        value for value in destination.members.values() if value.name == "primary"
    )
    secondary = next(
        value for value in destination.members.values() if value.name == "secondary"
    )

    assert imported_part.id != ids["part"]
    assert imported_sheet.id != ids["sheet"]
    assert primary.id != ids["first_member"]
    assert primary.orientation_reference == ("vertex", mapped_corner)
    assert secondary.orientation_reference == ("face", mapped_face)
    assert destination.construction_owner(mapped_construction) == imported_part.id
    assert destination.evaluate_face_many(mapped_face, ((0.25, 0.75),)) == pytest.approx(
        source.evaluate_face_many(ids["face"], ((0.25, 0.75),))
    )
    assert destination.faces[mapped_face].parameterization is not None

    member_sheet = _by_token(destination.attachments.values(), "member-sheet")
    assert member_sheet.member_id == primary.id
    assert member_sheet.source_key == ("member", primary.id)
    assert member_sheet.target_key == ("sheet", imported_sheet.id)
    assert member_sheet.part_id == imported_part.id
    assert member_sheet.sheet_id == imported_sheet.id
    assert member_sheet.connection_intent is ConnectionIntent.CONTACT_ONLY
    assert member_sheet.evidence is AttachmentEvidence.EXACT
    assert member_sheet.max_residual == pytest.approx(1.0e-8)
    assert member_sheet.tolerance_used == pytest.approx(2.0e-8)
    assert member_sheet.provenance["classifier"] == "qualified"
    assert member_sheet.lineage == (
        ("member", primary.id),
        ("face", mapped_face),
        ("part", imported_part.id),
    )

    member_member = _by_token(destination.attachments.values(), "member-member")
    assert member_member.source_key == ("member", primary.id)
    assert member_member.target_key == ("member", secondary.id)
    vertex_face = _by_token(destination.attachments.values(), "vertex-face")
    assert vertex_face.member_id is None
    assert vertex_face.source_key == ("vertex", mapped_corner)
    assert vertex_face.target_key == ("face", mapped_face)
    face_edge = _by_token(destination.attachments.values(), "face-edge")
    assert face_edge.source_key == ("face", mapped_face)
    assert face_edge.target_key == ("edge", mapped_boundary)
    sheet_vertex = _by_token(destination.attachments.values(), "sheet-vertex")
    assert sheet_vertex.source_key == ("sheet", imported_sheet.id)
    assert sheet_vertex.target_key == (
        "vertex",
        result.mapped(EntityRef("vertex", 2)).id,
    )

    junction = _by_token(destination.junctions.values(), "junction")
    assert junction.member_ids == (primary.id,)
    assert junction.sheet_ids == (imported_sheet.id,)
    assert junction.attachment_ids == (member_sheet.id,)
    assert junction.connection_intent is ConnectionIntent.CONTACT_ONLY
    assert junction.provenance["classifier"] == "qualified"
    assert destination.validate_topology() == ()


def test_partial_copy_keeps_only_relationships_with_every_parent_present() -> None:
    source, ids = _full_contract_model()
    original_attachment_ids = set(source.attachments)
    original_sheet_ids = set(source.sheets)
    original_member_ids = set(source.members)
    original_junction_ids = set(source.junctions)

    result = copy_entities(source, (EntityRef("face", ids["face"]),))

    copied_attachment_ids = set(source.attachments) - original_attachment_ids
    copied = {source.attachments[value].metadata.get("token") for value in copied_attachment_ids}
    assert copied == {"vertex-face", "face-edge", "sheet-vertex"}
    assert len(set(source.sheets) - original_sheet_ids) == 1
    assert set(source.members) == original_member_ids
    assert set(source.junctions) == original_junction_ids
    copied_face = result.mapped(EntityRef("face", ids["face"])).id
    assert _by_token(
        (source.attachments[value] for value in copied_attachment_ids),
        "vertex-face",
    ).target_id == copied_face
    assert source.validate_topology() == ()


def test_copy_preserves_owned_and_unowned_construction_vertices() -> None:
    geometry = GeometryModel()
    owner = geometry.add_part(name="construction owner")
    owned, unowned = geometry.add_points(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)))
    geometry.mark_construction_vertices((owned,), part_id=owner)
    geometry.mark_construction_vertices((unowned,))
    original_parts = set(geometry.parts)

    result = copy_entities(
        geometry,
        (EntityRef("vertex", owned), EntityRef("vertex", unowned)),
    )

    copied_owned = result.mapped(EntityRef("vertex", owned)).id
    copied_unowned = result.mapped(EntityRef("vertex", unowned)).id
    new_parts = set(geometry.parts) - original_parts
    assert len(new_parts) == 1
    assert geometry.construction_owner(copied_owned) == next(iter(new_parts))
    assert geometry.construction_owner(copied_unowned) is None
    assert geometry.validate_topology() == ()


def test_copy_rejects_float_identity_instead_of_aliasing_an_integer_id() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)

    with pytest.raises(GeometryError, match="positive integer"):
        copy_entities(geometry, (EntityRef("vertex", float(vertex)),))

    assert len(geometry.vertices) == 1


def test_reverse_face_reparameterizes_explicit_map_with_the_support() -> None:
    geometry = GeometryModel()
    _corners, face = _plate(geometry)
    geometry.set_face_parameterization(
        face,
        Plane(
            np.asarray((0.0, 0.0, 0.0)),
            np.asarray((2.0, 0.0, 0.0)),
            np.asarray((0.0, 1.0, 0.0)),
        ),
    )
    point = geometry.face_point(face, 0.2, 0.7)
    normal = geometry.face_normal(face, 0.2, 0.7)

    reverse_face(geometry, face)

    assert geometry.face_point(face, 0.7, 0.2) == pytest.approx(point)
    assert geometry.face_normal(face, 0.7, 0.2) == pytest.approx(-normal)
    assert geometry.validate_topology() == ()
