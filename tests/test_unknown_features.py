"""Forward-compatible frozen feature materializations."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json

import pytest

from anygeometry import (
    EntityRef,
    FeatureOutputRef,
    FeatureRegistry,
    GeometryError,
    GeometryModel,
    from_dict,
    to_dict,
)


def _rehash(document: dict) -> None:
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


def _point_feature_document() -> dict:
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    record = geometry.features.append(
        "geometry.point", parameters={"position": [1.0, 2.0, 3.0]}
    )
    assert geometry.regenerate_features().success
    document = to_dict(geometry)
    document["features"]["records"][0]["kind"] = "vendor.future.point"
    _rehash(document)
    assert document["features"]["records"][0]["materialization_checksum"]
    return document


def test_unknown_executor_uses_verified_frozen_materialization() -> None:
    geometry = from_dict(_point_feature_document())
    record = geometry.features.records[0]
    anchor = FeatureOutputRef(record.feature_id, "point", "vertex")

    assert record.state == "frozen"
    assert len(geometry.features.resolve(anchor, geometry)) == 1
    before = to_dict(geometry, include_features=False)

    report = geometry.regenerate_features()

    assert not report.success
    assert "regeneration is disabled" in (report.diagnostic or "")
    assert to_dict(geometry, include_features=False) == before


def test_unknown_feature_serialization_is_stable_across_repeated_round_trips() -> None:
    geometry = from_dict(_point_feature_document())

    first = to_dict(geometry)
    second = to_dict(from_dict(deepcopy(first)))
    third = to_dict(from_dict(deepcopy(second)))

    assert first == second == third
    assert first["features"]["records"][0]["state"] == "frozen"


def test_custom_executor_output_serializes_as_portable_frozen_materialization() -> None:
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    geometry.features.append(
        "vendor.custom.point", parameters={"position": [4.0, 5.0, 6.0]}
    )
    registry = FeatureRegistry()

    def custom_point(model: GeometryModel, _feature: object, _inputs: object) -> dict:
        identifier = model.add_point(4.0, 5.0, 6.0)
        return {"point": EntityRef("vertex", identifier)}

    registry.register("vendor.custom.point", custom_point)
    assert geometry.regenerate_features(registry).success

    first = to_dict(geometry)
    second = to_dict(from_dict(deepcopy(first)))

    assert first == second
    record = first["features"]["records"][0]
    assert record["state"] == "frozen"
    assert "verified last-good materialization" in record["diagnostic"]


def test_tampered_unknown_materialization_fails_document_integrity() -> None:
    document = deepcopy(_point_feature_document())
    document["vertices"][0]["position"][0] = 99.0

    with pytest.raises(GeometryError, match="checksum mismatch"):
        from_dict(document)


def test_current_schema_unknown_feature_requires_a_materialization_checksum() -> None:
    document = _point_feature_document()
    document["features"]["records"][0].pop("materialization_checksum")

    _rehash(document)
    with pytest.raises(GeometryError, match="missing required field.*materialization_checksum"):
        from_dict(document)


def test_unknown_feature_with_a_missing_output_opens_unresolved_for_repair() -> None:
    document = _point_feature_document()
    document["features"]["records"][0]["outputs"]["point"] = ["vertex", 999]
    _rehash(document)

    geometry = from_dict(document)
    record = geometry.features.records[0]

    assert record.state == "invalid"
    assert "missing entity" in (record.diagnostic or "")
    assert geometry.vertices  # unaffected materialization remains inspectable
