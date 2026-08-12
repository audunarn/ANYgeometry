"""Schema-4 and legacy schema-3 coverage for the gap-closure contract."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from anygeometry import AuditCode, GeometryError, GeometryModel, from_dict, to_dict
from anygeometry.structural import (
    AttachmentEvidence,
    AttachmentKind,
    AttachmentTargetKind,
    ConnectionIntent,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)
from anygeometry.surfaces import Plane


def _rehash(document: dict[str, object]) -> None:
    payload = {key: value for key, value in document.items() if key != "checksum"}
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    document["checksum"] = {
        "algorithm": "sha256",
        "value": hashlib.sha256(encoded).hexdigest(),
    }


def _gap_closure_model() -> GeometryModel:
    model = GeometryModel()
    corners = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face_id = model.add_plate(corners)
    model.set_face_parameterization(
        face_id,
        Plane((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    )
    part_id = model.add_part(name="assembly")
    sheet_id = model.add_sheet((face_id,), part_id=part_id)
    boundary_edge = model.faces[face_id].loop[0].edge
    member_id = model.add_member(
        (boundary_edge,),
        part_id=part_id,
        orientation_reference=("vertex", corners[0]),
    )
    attachment_id = model.add_attachment(
        member_id,
        AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE,
        boundary_edge,
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
        connection_intent=ConnectionIntent.CONNECT,
        evidence=AttachmentEvidence.EXACT,
        max_residual=0.0,
        tolerance_used=model.tolerance.effective_coincidence(2.0),
        part_id=part_id,
        sheet_id=sheet_id,
        provenance={"qualified_by": "test"},
        lineage=(("edge", boundary_edge),),
    )
    model.add_junction(
        JunctionKind.OVERLAP,
        (JunctionMemberUse(member_id, ParameterRange(0.0, 1.0)),),
        sheet_ids=(sheet_id,),
        attachment_ids=(attachment_id,),
        connection_intent=ConnectionIntent.CONNECT,
        provenance={"qualified_by": "test"},
    )

    chain_vertices = model.add_points(
        ((0.0, 2.0, 0.0), (1.0, 2.0, 0.0), (2.0, 2.0, 0.0))
    )
    chain_edges = (
        model.add_line(chain_vertices[0], chain_vertices[1]),
        model.add_line(chain_vertices[1], chain_vertices[2]),
    )
    split_source = model.add_member(chain_edges, part_id=part_id)
    model.split_member(split_source, 0.5)

    construction_vertex = model.add_point(3.0, 3.0, 0.0)
    model.mark_construction_vertices((construction_vertex,), part_id=part_id)
    model.set_document_settings(
        crs_metadata={
            "authority": "EPSG",
            "code": 25832,
            "axis_order": ["easting", "northing", "height"],
        }
    )
    return model


def test_gap_closure_state_round_trips_in_schema_4() -> None:
    model = _gap_closure_model()

    document = to_dict(model, include_features=False)
    restored = from_dict(deepcopy(document))

    assert to_dict(restored, include_features=False) == document
    assert restored.crs_metadata == model.crs_metadata
    assert restored.construction_vertices == model.construction_vertices
    assert (
        restored.structural_replacement_history()
        == model.structural_replacement_history()
    )
    assert restored.members == model.members
    assert restored.attachments == model.attachments
    assert restored.junctions == model.junctions
    restored_map = next(iter(restored.faces.values())).parameterization
    source_map = next(iter(model.faces.values())).parameterization
    assert isinstance(restored_map, Plane)
    assert isinstance(source_map, Plane)
    assert np.array_equal(restored_map.origin, source_map.origin)
    assert np.array_equal(restored_map.u_vector, source_map.u_vector)
    assert np.array_equal(restored_map.v_vector, source_map.v_vector)


def test_certified_write_is_an_exact_revision_validation_gate_not_a_schema_flag() -> None:
    model = GeometryModel()
    corners = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    model.add_plate(corners)

    ordinary = to_dict(model, include_features=False)
    certified = to_dict(model, include_features=False, certified=True)

    assert certified == ordinary
    assert "certified" not in certified
    assert "audit" not in certified


@pytest.mark.parametrize("source_kind", ("edge", "face", "sheet"))
def test_schema_4_rejects_a_missing_generalized_attachment_source(
    source_kind: str,
) -> None:
    model = GeometryModel()
    corners = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = model.add_plate(corners)
    part = model.add_part(name="source-validation")
    sheet = model.add_sheet((face,), part_id=part)
    edges = tuple(item.edge for item in model.faces[face].loop)
    identifiers = {"edge": edges[0], "face": face, "sheet": sheet}
    attachment_id = model.add_attachment(
        None,
        AttachmentKind.INTENTIONALLY_DISCONNECTED,
        AttachmentTargetKind.EDGE,
        edges[1],
        ParameterRange(0.0, 1.0),
        (ParameterRange(0.0, 1.0),),
        source_kind=source_kind,
        source_id=identifiers[source_kind],
        connection_intent=ConnectionIntent.KEEP_DISCONNECTED,
    )
    document = to_dict(model, include_features=False)
    structural = document["structural"]
    assert isinstance(structural, dict)
    records = structural["attachments"]
    assert isinstance(records, list)
    record = next(item for item in records if item["id"] == attachment_id)
    record["source_id"] = 999_999
    _rehash(document)

    with pytest.raises(GeometryError, match=rf"missing source {source_kind}"):
        from_dict(document)


def test_public_codec_does_not_mutate_caller_owned_current_or_legacy_mappings() -> None:
    current = to_dict(_gap_closure_model(), include_features=False)
    current_before = deepcopy(current)
    restored = from_dict(MappingProxyType(current))
    assert current == current_before
    assert to_dict(restored, include_features=False) == current_before

    legacy_model = GeometryModel()
    first, second = legacy_model.add_points(((0, 0, 0), (1, 0, 0)))
    legacy_model.add_line(first, second)
    encoded = to_dict(legacy_model, include_features=False)
    legacy = {
        key: deepcopy(encoded[key])
        for key in (
            "schema",
            "id_state",
            "vertices",
            "edges",
            "faces",
            "groups",
            "tags",
            "replacement_history",
        )
    }
    legacy["version"] = 2
    legacy["id_state"] = {
        kind: legacy["id_state"][kind]  # type: ignore[index]
        for kind in ("vertex", "edge", "face")
    }
    legacy_before = deepcopy(legacy)
    migrated = from_dict(MappingProxyType(legacy))
    assert legacy == legacy_before
    assert migrated.entity_keys() == {("vertex", 1), ("vertex", 2), ("edge", 1)}


def test_schema_1_migrates_once_to_canonical_schema_4_with_bound_identity() -> None:
    legacy_source = GeometryModel()
    corners = legacy_source.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face_id = legacy_source.add_plate(corners)
    encoded = to_dict(legacy_source, include_features=False)
    legacy = {
        key: deepcopy(encoded[key])
        for key in (
            "schema",
            "id_state",
            "vertices",
            "edges",
            "faces",
            "groups",
            "tags",
            "replacement_history",
        )
    }
    legacy["id_state"] = {
        kind: legacy["id_state"][kind]  # type: ignore[index]
        for kind in ("vertex", "edge", "face")
    }

    migrated = from_dict(MappingProxyType(legacy))
    canonical = to_dict(migrated, include_features=False)
    restored = from_dict(deepcopy(canonical))

    assert canonical["version"] == 4
    assert restored.model_id == migrated.model_id
    assert to_dict(restored, include_features=False) == canonical
    assert set(migrated.faces) == {face_id}
    assert len(migrated.parts) == 1
    assert len(migrated.sheets) == 1
    assert len(migrated.face_uses) == 1
    assert len(migrated.coedges) == 4
    assert not migrated.members
    assert not migrated.attachments
    assert canonical["extensions"]["anygeometry:migration"] == {
        "source_version": 1,
        "target_version": 4,
        "inferred": "face ownership only; no members inferred",
    }


def test_earlier_schema_3_optional_field_shape_still_loads() -> None:
    document = to_dict(_gap_closure_model(), include_features=False)
    document["version"] = 3
    document.pop("construction_vertices")
    document.pop("structural_replacement_history")
    document["coordinates"].pop("crs")  # type: ignore[union-attr]
    for name in ("coincidence", "healing", "curve_fit_residual", "aabb_padding"):
        document["tolerance"].pop(name)  # type: ignore[union-attr]
    for face in document["faces"]:  # type: ignore[union-attr]
        face.pop("parameterization")
    for member in document["structural"]["members"]:  # type: ignore[index]
        member.pop("orientation_reference")
    for attachment in document["structural"]["attachments"]:  # type: ignore[index]
        for name in (
            "connection_intent",
            "evidence",
            "max_residual",
            "tolerance_used",
            "part_id",
            "sheet_id",
            "provenance",
            "lineage",
            "source_kind",
            "source_id",
        ):
            attachment.pop(name)
    for junction in document["structural"]["junctions"]:  # type: ignore[index]
        junction.pop("connection_intent")
        junction.pop("provenance")
    _rehash(document)

    restored = from_dict(document)

    assert not restored.crs_metadata
    assert not restored.construction_vertices
    assert not restored.structural_replacement_history()
    assert all(face.parameterization is None for face in restored.faces.values())
    assert all(
        member.orientation_reference is None for member in restored.members.values()
    )
    assert all(
        attachment.evidence is AttachmentEvidence.UNVERIFIED
        and attachment.max_residual == 0.0
        and attachment.tolerance_used == 0.0
        for attachment in restored.attachments.values()
    )
    report = restored.audit()
    assert not report.certifiable
    assert any(
        issue.code is AuditCode.UNVERIFIED_CLASSIFICATION
        for issue in report.issues
    )
