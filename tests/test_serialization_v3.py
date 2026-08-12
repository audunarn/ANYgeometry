"""Schema-3 identity, structural topology, migration and corruption tests."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import numpy as np
import pytest

from anygeometry import (
    EntityRef,
    GeometryError,
    GeometryModel,
    OrientedEdge,
    from_dict,
    read_geometry,
    to_dict,
    write_geometry,
)
from anygeometry.structural import (
    AttachmentKind,
    AttachmentTargetKind,
    JunctionKind,
    JunctionMemberUse,
    ParameterRange,
)


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


def _as_schema_2(model: GeometryModel) -> dict[str, object]:
    current = to_dict(model, include_features=False)
    legacy = {
        key: deepcopy(value)
        for key, value in current.items()
        if key
        in {
            "schema",
            "id_state",
            "vertices",
            "edges",
            "faces",
            "groups",
            "tags",
            "replacement_history",
        }
    }
    legacy["version"] = 2
    legacy["id_state"] = {
        kind: legacy["id_state"][kind]  # type: ignore[index]
        for kind in ("vertex", "edge", "face")
    }
    return legacy


def _mixed_model() -> GeometryModel:
    model = GeometryModel()
    corners = model.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0))
    )
    face = model.add_plate(corners)
    model.tag(EntityRef("face", face), "structural")
    first = model.add_points(((0.0, 1.0, 0.0), (2.0, 1.0, 0.0)))
    second = model.add_points(((1.0, 0.0, 0.0), (1.0, 2.0, 0.0)))
    first_edge = model.add_line(*first)
    second_edge = model.add_line(*second)
    part = model.add_part(name="Assembly", metadata={"source": ["test", 3]})
    sheet = model.add_sheet((face,), part_id=part, name="Deck")
    first_member = model.add_member(
        (first_edge,), part_id=part, name="Longitudinal", metadata={"section": "T"}
    )
    second_member = model.add_member(
        (second_edge,), part_id=part, name="Transverse"
    )
    attachment = model.add_attachment(
        first_member,
        AttachmentKind.MEMBER_THROUGH_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange.point(0.5),
        (ParameterRange.point(0.5), ParameterRange.point(0.5)),
    )
    model.add_junction(
        JunctionKind.CROSSING,
        (
            JunctionMemberUse(first_member, ParameterRange.point(0.5)),
            JunctionMemberUse(second_member, ParameterRange.point(0.5)),
        ),
        sheet_ids=(sheet,),
        attachment_ids=(attachment,),
    )
    model.set_document_settings(
        units="mm",
        local_origin=(1000.0, -250.0, 4.0),
        coordinate_transform=(
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )
    return model


def test_schema_3_round_trip_preserves_identity_coordinates_and_structural_uses() -> None:
    model = _mixed_model()
    document = to_dict(model)

    restored = from_dict(deepcopy(document))

    assert document["version"] == 3
    assert to_dict(restored) == document
    assert restored.model_id == model.model_id
    assert restored.revision == model.revision
    assert restored.units == "mm"
    assert np.array_equal(restored.local_origin, model.local_origin)
    assert np.array_equal(restored.coordinate_transform, model.coordinate_transform)
    assert restored.tolerance == model.tolerance
    assert restored.parts == model.parts
    assert restored.sheets == model.sheets
    assert restored.face_uses == model.face_uses
    assert restored.coedges == model.coedges
    assert restored.members == model.members
    assert restored.member_edge_uses == model.member_edge_uses
    assert restored.attachments == model.attachments
    assert restored.junctions == model.junctions
    assert restored._edge_member_uses == model._edge_member_uses  # noqa: SLF001


def test_serialization_rejects_mutated_invalid_feature_history() -> None:
    model = GeometryModel()
    first = model.features.append("vendor.first")
    model.features.append("vendor.second")
    model.features._records[1].feature_id = first.feature_id  # noqa: SLF001

    with pytest.raises(GeometryError, match="duplicate feature ID"):
        to_dict(model)


def test_public_feature_records_are_detached_from_persisted_history() -> None:
    model = GeometryModel()
    feature = model.features.append(
        "vendor.first", parameters={"value": [1, 2, 3]}, suppressed=True
    )
    revision = model.revision

    feature.feature_id = 99
    feature.parameters["value"][0] = 999
    model.features.records.clear()

    persisted = model.features.get(1)
    assert persisted.feature_id == 1
    assert persisted.parameters == {"value": [1, 2, 3]}
    assert len(model.features.records) == 1
    assert model.revision == revision
    document = to_dict(model)
    assert document["features"]["records"][0]["id"] == 1
    assert document["features"]["records"][0]["parameters"] == {
        "value": [1, 2, 3]
    }
    assert to_dict(from_dict(deepcopy(document))) == document


def test_serialization_rejects_unmaterialized_unknown_feature_history() -> None:
    model = GeometryModel()
    model.features.append("vendor.future")

    with pytest.raises(GeometryError, match="verified last-good outputs"):
        to_dict(model)

    # Geometry-only exports intentionally exclude and therefore do not audit
    # the omitted construction history.
    assert "features" not in to_dict(model, include_features=False)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("feature_id", False),
        ("kind", ""),
        ("name", 17),
        ("kind_version", 0),
        ("suppressed", "yes"),
        ("parameters", {"not_finite": float("nan")}),
        ("dependencies", (999,)),
        ("inputs", {"source": (object(),)}),
        ("outputs", {"result": object()}),
        ("state", "mystery"),
        ("diagnostic", 17),
        ("materialization_checksum", "not-a-sha256"),
    ),
)
def test_serialization_rejects_hostile_feature_record_fields(
    field: str, value: object
) -> None:
    model = GeometryModel()
    model.features.append("vendor.future", suppressed=True)
    setattr(model.features._records[0], field, value)  # noqa: SLF001

    with pytest.raises(GeometryError):
        to_dict(model)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("_next_id", True),
        ("_next_id", 1),
        ("_baseline", {"hostile": object()}),
    ),
)
def test_serialization_rejects_hostile_feature_history_fields(
    field: str, value: object
) -> None:
    model = GeometryModel()
    model.features.append("vendor.future", suppressed=True)
    setattr(model.features, field, value)

    with pytest.raises(GeometryError):
        to_dict(model)


def test_serialization_rejects_known_active_feature_without_outputs() -> None:
    model = GeometryModel()
    model.features.append(
        "geometry.point", parameters={"position": [0.0, 0.0, 0.0]}
    )
    model.features._records[0].state = "ok"  # noqa: SLF001

    with pytest.raises(GeometryError, match="no materialized outputs"):
        to_dict(model)

    assert "features" not in to_dict(model, include_features=False)


def test_checksum_is_canonical_and_detects_payload_corruption() -> None:
    document = to_dict(_mixed_model())
    assert document["checksum"] == deepcopy(to_dict(from_dict(document))["checksum"])

    document["extensions"]["test:tampered"] = True
    with pytest.raises(GeometryError, match="checksum mismatch"):
        from_dict(document)


def test_certified_output_requires_a_clean_strict_audit() -> None:
    clean = GeometryModel()
    vertices = clean.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    clean.add_plate(vertices)
    certified = to_dict(clean, certified=True)
    assert certified["checksum"]["algorithm"] == "sha256"

    hostile = GeometryModel()
    points = hostile.add_points(
        ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0))
    )
    hostile.add_line(points[0], points[1])
    hostile.add_line(points[2], points[3])
    with pytest.raises(GeometryError, match="clean strict audit"):
        to_dict(hostile, certified=True)


@pytest.mark.parametrize(
    "mutation, message",
    (
        (lambda data: data["vertices"].append(deepcopy(data["vertices"][0])), "duplicate vertex"),
        (lambda data: data["tags"].extend(deepcopy(data["tags"])), "duplicate tag record"),
        (lambda data: data["id_state"].update(member=1), "counter would reuse"),
        (
            lambda data: data["structural"]["member_edge_uses"][0].update(orientation="sideways"),
            "orientation",
        ),
    ),
)
def test_schema_3_rejects_duplicate_records_invalid_allocator_and_enum(
    mutation: object, message: str
) -> None:
    document = to_dict(_mixed_model())
    mutation(document)  # type: ignore[operator]
    _rehash(document)
    with pytest.raises(GeometryError, match=message):
        from_dict(document)


def test_schema_3_rejects_missing_and_unexpected_core_fields() -> None:
    missing = to_dict(_mixed_model())
    del missing["tolerance"]
    with pytest.raises(GeometryError, match="missing required field.*tolerance"):
        from_dict(missing)

    unexpected = to_dict(_mixed_model())
    unexpected["vendor_data"] = {}
    with pytest.raises(GeometryError, match="unexpected field.*vendor_data"):
        from_dict(unexpected)


def test_file_reader_rejects_duplicate_json_object_keys(tmp_path) -> None:
    path = tmp_path / "duplicate.json"
    path.write_text('{"schema":"anygeometry","schema":"anygeometry","version":3}')

    with pytest.raises(GeometryError, match="duplicate JSON object key 'schema'"):
        read_geometry(path)


def test_gzip_output_is_byte_deterministic_across_paths(tmp_path) -> None:
    model = _mixed_model()
    first = tmp_path / "first.json.gz"
    second = tmp_path / "second.json.gz"

    write_geometry(first, model)
    write_geometry(second, model)

    assert first.read_bytes() == second.read_bytes()
    assert to_dict(read_geometry(first)) == to_dict(model)


def test_schema_2_migration_infers_only_face_ownership_and_records_provenance() -> None:
    legacy = _as_schema_2(_mixed_model())

    migrated = from_dict(legacy)

    assert len(migrated.parts) == 1
    assert len(migrated.sheets) == 1
    assert migrated.members == {}
    assert migrated.attachments == {}
    migrated_document = to_dict(migrated)
    assert migrated_document["extensions"]["anygeometry:migration"] == {
        "source_version": 2,
        "target_version": 3,
        "inferred": "face ownership only; no members inferred",
    }


def test_schema_2_migration_does_not_invent_cross_face_orientation() -> None:
    model = GeometryModel()
    vertices = model.add_points(
        (
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (2.0, 0.0, 0.0),
            (2.0, 1.0, 0.0),
        )
    )
    lower = model.add_line(vertices[0], vertices[1])
    shared = model.add_line(vertices[1], vertices[2])
    upper = model.add_line(vertices[2], vertices[3])
    left = model.add_line(vertices[3], vertices[0])
    upper_right = model.add_line(vertices[2], vertices[5])
    right = model.add_line(vertices[5], vertices[4])
    lower_right = model.add_line(vertices[4], vertices[1])
    model.add_face_from_loop(
        tuple(
            OrientedEdge(edge, True)
            for edge in (lower, shared, upper, left)
        )
    )
    # Schema 2 had no sheet orientation rule, so this independently valid
    # second face may traverse the shared edge in the same direction.
    model.add_face_from_loop(
        tuple(
            OrientedEdge(edge, True)
            for edge in (shared, upper_right, right, lower_right)
        )
    )

    migrated = from_dict(_as_schema_2(model))

    assert len(migrated.parts) == 1
    assert len(migrated.sheets) == 2
    assert len(migrated.face_uses) == 2
    assert migrated.validate_topology() == ()
    assert migrated._validate_structural() == ()  # noqa: SLF001
