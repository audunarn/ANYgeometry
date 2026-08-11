"""Forward-compatible frozen feature materializations."""

from __future__ import annotations

from copy import deepcopy

from anygeometry import FeatureOutputRef, GeometryModel, from_dict, to_dict


def _point_feature_document() -> dict:
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    record = geometry.features.append(
        "geometry.point", parameters={"position": [1.0, 2.0, 3.0]}
    )
    assert geometry.regenerate_features().success
    document = to_dict(geometry)
    document["features"]["records"][0]["kind"] = "vendor.future.point"
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


def test_tampered_unknown_materialization_is_visible_but_unresolvable() -> None:
    document = deepcopy(_point_feature_document())
    document["vertices"][0]["position"][0] = 99.0

    geometry = from_dict(document)
    record = geometry.features.records[0]
    anchor = FeatureOutputRef(record.feature_id, "point", "vertex")

    assert record.state == "invalid"
    assert "checksum does not match" in (record.diagnostic or "")
    assert geometry.features.resolve(anchor, geometry) == ()
    assert geometry.vertices  # retained for viewing/export and repair


def test_legacy_unknown_feature_acquires_a_checksum_on_load() -> None:
    document = _point_feature_document()
    document["features"]["records"][0].pop("materialization_checksum")

    geometry = from_dict(document)

    assert geometry.features.records[0].state == "frozen"
    assert geometry.features.records[0].materialization_checksum
    assert (
        to_dict(geometry)["features"]["records"][0]["materialization_checksum"]
        == geometry.features.records[0].materialization_checksum
    )


def test_unknown_feature_with_a_missing_output_opens_unresolved_for_repair() -> None:
    document = _point_feature_document()
    document["features"]["records"][0]["outputs"]["point"] = ["vertex", 999]

    geometry = from_dict(document)
    record = geometry.features.records[0]

    assert record.state == "invalid"
    assert "missing entity" in (record.diagnostic or "")
    assert geometry.vertices  # unaffected materialization remains inspectable
