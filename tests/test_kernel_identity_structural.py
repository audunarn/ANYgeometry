"""Focused contracts for model-bound identity and structural topology values."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from uuid import UUID

import pytest

from anygeometry.errors import GeometryError, GeometryTopologyError
from anygeometry.identity import EntityHandle, Resolution, ResolutionStatus
from anygeometry.structural import (
    Attachment,
    AttachmentKind,
    AttachmentTargetKind,
    BoundaryPolicy,
    Coedge,
    FaceUse,
    FrozenMetadata,
    Junction,
    JunctionKind,
    JunctionMemberUse,
    Member,
    MemberEdgeUse,
    Orientation,
    ParameterRange,
    Part,
    Sheet,
    SheetTopologyPolicy,
    raise_for_structural_topology,
    replace_member_edge_use,
    structural_entity_keys,
    validate_structural_topology,
)


MODEL_A = UUID("12345678-1234-5678-1234-567812345678")
MODEL_B = UUID("87654321-4321-8765-4321-876543218765")


def test_entity_handle_is_model_bound_validated_and_deterministically_ordered() -> None:
    handle = EntityHandle(str(MODEL_A), "edge", 4)

    assert handle.model_id == MODEL_A
    assert handle.key == ("edge", 4)
    assert not hasattr(handle, "__dict__")
    assert handle.belongs_to(str(MODEL_A))
    assert handle != EntityHandle(MODEL_B, "edge", 4)
    assert sorted(
        (
            EntityHandle(MODEL_A, "part", 1),
            EntityHandle(MODEL_A, "edge", 9),
            EntityHandle(MODEL_A, "vertex", 8),
            EntityHandle(MODEL_A, "edge", 2),
        )
    ) == [
        EntityHandle(MODEL_A, "vertex", 8),
        EntityHandle(MODEL_A, "edge", 2),
        EntityHandle(MODEL_A, "edge", 9),
        EntityHandle(MODEL_A, "part", 1),
    ]
    with pytest.raises(FrozenInstanceError):
        handle.id = 8  # type: ignore[misc]
    with pytest.raises(GeometryError, match="unknown entity kind"):
        EntityHandle(MODEL_A, "solid", 1)
    with pytest.raises(GeometryError, match="positive integer"):
        EntityHandle(MODEL_A, "edge", True)
    with pytest.raises(GeometryError, match="nil UUID"):
        EntityHandle(UUID(int=0), "edge", 1)


def test_resolution_has_explicit_exhaustive_outcomes_and_strict_invariants() -> None:
    requested = EntityHandle(MODEL_A, "edge", 3)
    descendants = (
        EntityHandle(MODEL_A, "edge", 9),
        EntityHandle(MODEL_A, "edge", 7),
    )

    active = Resolution.active(requested)
    replaced = Resolution.replaced(requested, descendants)
    wrong_model = Resolution.terminal(
        requested, ResolutionStatus.WRONG_MODEL, model_id=MODEL_B
    )
    deleted = Resolution.terminal(requested, ResolutionStatus.DELETED)

    assert active.resolved == (requested,)
    assert active.require() == (requested,)
    assert replaced.resolved == tuple(sorted(descendants))
    assert replaced.is_resolved
    assert wrong_model.status is ResolutionStatus.WRONG_MODEL
    assert not deleted.is_resolved
    with pytest.raises(GeometryError, match="deleted"):
        deleted.require()
    with pytest.raises(GeometryError, match="another model"):
        Resolution(
            MODEL_A,
            requested,
            ResolutionStatus.REPLACED,
            (EntityHandle(MODEL_B, "edge", 7),),
        )
    with pytest.raises(GeometryError, match="cannot change entity kind"):
        Resolution.replaced(requested, (EntityHandle(MODEL_A, "face", 7),))
    with pytest.raises(GeometryError, match="WRONG_MODEL"):
        Resolution.terminal(requested, ResolutionStatus.WRONG_MODEL)


def test_metadata_and_structural_records_are_deeply_immutable() -> None:
    source = {"z": [1, {"enabled": True}], "a": {"label": "web"}}
    part = Part(1, member_ids=(3, 1), name="frame", metadata=source)
    source["z"][1]["enabled"] = False  # type: ignore[index]

    assert part.member_ids == (1, 3)
    assert not hasattr(part, "__dict__")
    assert tuple(part.metadata) == ("a", "z")
    assert isinstance(part.metadata["a"], FrozenMetadata)
    assert part.metadata.to_dict() == {
        "a": {"label": "web"},
        "z": [1, {"enabled": True}],
    }
    with pytest.raises(TypeError):
        part.metadata["new"] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        part.name = "changed"  # type: ignore[misc]
    with pytest.raises(GeometryError, match="must be finite"):
        Part(2, metadata={"bad": float("nan")})
    with pytest.raises(GeometryError, match="not JSON metadata"):
        Part(2, metadata={"bad": {1, 2}})


def _mixed_structural_model() -> dict[str, object]:
    parts = {1: Part(1, sheet_ids=(1,), member_ids=(1, 2))}
    sheets = {1: Sheet(1, 1, (1,))}
    face_uses = {1: FaceUse(1, 1, 10, ((1, 2, 3, 4),))}
    coedges = {
        1: Coedge(1, 1, 1),
        2: Coedge(2, 1, 2),
        3: Coedge(3, 1, 3),
        4: Coedge(4, 1, 4),
    }
    members = {
        1: Member(1, 1, (1, 2), name="longitudinal"),
        2: Member(2, 1, (3,), name="transverse"),
    }
    member_edge_uses = {
        1: MemberEdgeUse(1, 1, 5, ParameterRange(0.0, 0.5)),
        2: MemberEdgeUse(2, 1, 6, ParameterRange(0.5, 1.0)),
        3: MemberEdgeUse(3, 2, 7),
    }
    attachments = {
        1: Attachment(
            1,
            1,
            AttachmentKind.ENDPOINT,
            AttachmentTargetKind.EDGE,
            1,
            ParameterRange.point(0.0),
            (ParameterRange.point(0.5),),
        ),
        2: Attachment(
            2,
            1,
            AttachmentKind.MEMBER_THROUGH_FACE,
            AttachmentTargetKind.FACE,
            10,
            ParameterRange.point(0.5),
            (ParameterRange.point(0.25), ParameterRange.point(0.75)),
        ),
    }
    junctions = {
        1: Junction(
            1,
            JunctionKind.ENDPOINT,
            (JunctionMemberUse(1, ParameterRange.point(0.0)),),
            sheet_ids=(1,),
            attachment_ids=(1,),
        ),
        2: Junction(
            2,
            JunctionKind.CROSSING,
            (
                JunctionMemberUse(2, ParameterRange.point(0.5)),
                JunctionMemberUse(1, ParameterRange.point(0.5)),
            ),
        ),
    }
    edge_vertices = {
        1: (1, 2),
        2: (2, 3),
        3: (3, 4),
        4: (4, 1),
        5: (5, 6),
        6: (6, 7),
        7: (8, 9),
    }
    return {
        "parts": parts,
        "sheets": sheets,
        "face_uses": face_uses,
        "coedges": coedges,
        "members": members,
        "member_edge_uses": member_edge_uses,
        "attachments": attachments,
        "junctions": junctions,
        "edge_ids": tuple(edge_vertices),
        "face_ids": (10,),
        "edge_vertices": edge_vertices,
    }


def test_mixed_plate_member_topology_validates_and_keys_are_canonical() -> None:
    topology = _mixed_structural_model()

    assert validate_structural_topology(**topology) == ()  # type: ignore[arg-type]
    assert structural_entity_keys(
        **{
            name: topology[name]
            for name in (
                "parts",
                "sheets",
                "face_uses",
                "coedges",
                "members",
                "member_edge_uses",
                "attachments",
                "junctions",
            )
        }
    ) == (
        ("part", 1),
        ("sheet", 1),
        ("face_use", 1),
        ("coedge", 1),
        ("coedge", 2),
        ("coedge", 3),
        ("coedge", 4),
        ("member", 1),
        ("member", 2),
        ("member_edge_use", 1),
        ("member_edge_use", 2),
        ("member_edge_use", 3),
        ("attachment", 1),
        ("attachment", 2),
        ("junction", 1),
        ("junction", 2),
    )


def test_geometry_edge_can_have_shell_and_multiple_member_axis_uses() -> None:
    topology = _mixed_structural_model()
    topology["member_edge_uses"] = {
        1: MemberEdgeUse(1, 1, 1, ParameterRange(0.0, 0.5)),
        2: MemberEdgeUse(2, 1, 2, ParameterRange(0.5, 1.0)),
        3: MemberEdgeUse(3, 2, 1),
    }
    topology["attachments"] = {}
    topology["junctions"] = {}
    # This value-layer check intentionally does not infer a physical junction
    # from the shared geometry definition; strict geometric audit owns that
    # qualification.  Omitting endpoint data skips unrelated axis continuity.
    topology["edge_vertices"] = None

    assert validate_structural_topology(**topology) == ()  # type: ignore[arg-type]


def test_validation_reports_stale_ownership_ranges_and_incidence_stably() -> None:
    topology = _mixed_structural_model()
    topology["sheets"] = {
        1: Sheet(
            1,
            1,
            (1,),
            SheetTopologyPolicy(boundary=BoundaryPolicy.REQUIRE_CLOSED),
        )
    }
    topology["coedges"] = {
        **topology["coedges"],  # type: ignore[dict-item]
        2: Coedge(2, 1, 2, Orientation.REVERSED),
    }
    topology["member_edge_uses"] = {
        **topology["member_edge_uses"],  # type: ignore[dict-item]
        1: MemberEdgeUse(1, 1, 5, ParameterRange(0.1, 0.5)),
    }

    first = validate_structural_topology(**topology)  # type: ignore[arg-type]
    second = validate_structural_topology(**topology)  # type: ignore[arg-type]

    assert first == second == tuple(sorted(first))
    assert any("boundary edge" in error for error in first)
    assert any("loop 0 is not continuous" in error for error in first)
    assert any("parent ranges are not contiguous" in error for error in first)
    with pytest.raises(GeometryTopologyError, match="invalid structural topology"):
        raise_for_structural_topology(**topology)


def test_member_edge_subdivision_rewrites_only_axis_use_and_keeps_member_id() -> None:
    member = Member(8, 2, (11, 12, 13), metadata={"section_key": "T"})
    original = MemberEdgeUse(12, 8, 22, ParameterRange(0.2, 0.8))
    children = (
        MemberEdgeUse(21, 8, 31, ParameterRange(0.2, 0.45)),
        MemberEdgeUse(22, 8, 32, ParameterRange(0.45, 0.8)),
    )

    rewritten = replace_member_edge_use(member, original, children)

    assert rewritten.id == member.id
    assert rewritten.edge_use_ids == (11, 21, 22, 13)
    assert rewritten.metadata is member.metadata
    with pytest.raises(GeometryError, match="not contiguous"):
        replace_member_edge_use(
            member,
            original,
            (
                MemberEdgeUse(21, 8, 31, ParameterRange(0.2, 0.4)),
                MemberEdgeUse(22, 8, 32, ParameterRange(0.5, 0.8)),
            ),
        )


def test_parameter_attachment_and_junction_semantics_fail_closed() -> None:
    with pytest.raises(GeometryError, match="0 <= start"):
        ParameterRange(-0.1, 0.2)
    with pytest.raises(GeometryError, match="cannot be degenerate"):
        MemberEdgeUse(1, 1, 1, ParameterRange.point(0.4))
    with pytest.raises(GeometryError, match="requires a face target"):
        Attachment(
            1,
            1,
            AttachmentKind.MEMBER_ON_FACE,
            AttachmentTargetKind.EDGE,
            1,
            ParameterRange(0.0, 1.0),
            (ParameterRange(0.0, 1.0),),
        )
    with pytest.raises(GeometryError, match="must be point-valued"):
        Attachment(
            1,
            1,
            AttachmentKind.MEMBER_THROUGH_FACE,
            AttachmentTargetKind.FACE,
            1,
            ParameterRange(0.2, 0.8),
            (ParameterRange.point(0.5), ParameterRange.point(0.5)),
        )
    with pytest.raises(GeometryError, match="non-degenerate ranges"):
        Junction(
            1,
            JunctionKind.OVERLAP,
            (
                JunctionMemberUse(1, ParameterRange.point(0.5)),
                JunctionMemberUse(2, ParameterRange(0.2, 0.8)),
            ),
        )
