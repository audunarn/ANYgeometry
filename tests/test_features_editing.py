"""Persistent feature history and high-level owner editing APIs."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from anygeometry import (
    EntityRef,
    FeatureOutputRef,
    GeometryError,
    GeometryModel,
    circular_pattern,
    copy_entities,
    from_dict,
    linear_pattern,
    measure,
    mirror_entities,
    reverse_edge,
    reverse_face,
    to_dict,
)
from anygeometry.generators import plate


def _feature_line() -> tuple[GeometryModel, object, object, object]:
    geometry = GeometryModel()
    history = geometry.features
    history.capture_baseline(geometry)
    first = history.append(
        "geometry.point", parameters={"position": [0.0, 0.0, 0.0]}
    )
    second = history.append(
        "geometry.point", parameters={"position": [1.0, 0.0, 0.0]}
    )
    line = history.append(
        "geometry.line",
        inputs={
            "start": [FeatureOutputRef(first.feature_id, "point", "vertex")],
            "end": [FeatureOutputRef(second.feature_id, "point", "vertex")],
        },
    )
    assert geometry.regenerate_features().success
    return geometry, first, second, line


def test_feature_regeneration_uses_fresh_ids_and_preserves_lineage() -> None:
    geometry, _first, second, _line = _feature_line()
    old_outputs = {
        key: reference
        for feature in geometry.features.records
        for key, reference in feature.outputs.items()
    }
    old_state = geometry.id_state()

    geometry.features.get(second.feature_id).parameters["position"] = [2.0, 0.0, 0.0]
    report = geometry.regenerate_features()

    assert report.success
    assert min(geometry.vertices) >= old_state["vertex"]
    assert min(geometry.edges) >= old_state["edge"]
    assert geometry.validate_topology() == ()
    for old in old_outputs.values():
        resolved = geometry.resolve_ref(old)
        assert resolved
        assert all(item.id >= old_state[item.kind] for item in resolved)


def test_failed_regeneration_leaves_materialized_geometry_untouched() -> None:
    geometry, _first, second, _line = _feature_line()
    before = to_dict(geometry, include_features=False)
    outputs = [dict(item.outputs) for item in geometry.features.records]
    geometry.features.get(second.feature_id).parameters["position"] = [
        float("nan"),
        2.0,
        0.0,
    ]

    report = geometry.regenerate_features()

    assert report.success is False
    assert "failed" in (report.diagnostic or "")
    assert to_dict(geometry, include_features=False) == before
    assert [item.outputs for item in geometry.features.records] == outputs


def test_suppression_blocks_dependents_and_output_anchor_recovers() -> None:
    geometry, first, _second, line = _feature_line()
    anchor = FeatureOutputRef(first.feature_id, "point", "vertex")
    geometry.features.set_suppressed(first.feature_id)

    report = geometry.regenerate_features()

    assert report.success
    assert geometry.features.get(first.feature_id).state == "suppressed"
    assert geometry.features.get(line.feature_id).state == "blocked"
    assert geometry.features.resolve(anchor, geometry) == ()

    geometry.features.set_suppressed(first.feature_id, False)
    assert geometry.regenerate_features().success
    assert len(geometry.features.resolve(anchor, geometry)) == 1
    assert geometry.features.get(line.feature_id).state == "ok"


def test_feature_history_round_trip_and_v1_migration() -> None:
    geometry, _first, _second, _line = _feature_line()
    document = to_dict(geometry)

    restored = from_dict(deepcopy(document))

    assert document["version"] == 2
    assert to_dict(restored) == document
    assert len(restored.features.records) == 3

    legacy = to_dict(plate(2.0, 1.0))
    legacy["version"] = 1
    legacy.pop("features")
    migrated = from_dict(legacy)
    assert migrated.features.records == []
    assert migrated.features.baseline is not None
    assert to_dict(migrated)["version"] == 2


def test_generator_feature_inserts_stable_local_output_keys() -> None:
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    feature = geometry.features.append(
        "generator.plate",
        parameters={"length": 3.0, "width": 2.0, "semantic_group": "deck"},
    )

    assert geometry.regenerate_features().success

    outputs = geometry.features.get(feature.feature_id).outputs
    assert {"vertex/1", "edge/1", "face/1"} <= set(outputs)
    assert len(geometry.group("deck")) == 1
    assert measure(geometry, geometry.group("deck")[0]).value == pytest.approx(6.0)


def test_downstream_split_owns_precise_lineage_on_generator_regeneration() -> None:
    geometry = GeometryModel()
    geometry.features.capture_baseline(geometry)
    generated = geometry.features.append(
        "generator.plate", parameters={"length": 3.0, "width": 2.0}
    )
    split = geometry.features.append(
        "geometry.split_face",
        parameters={"axis": 0, "fraction": 0.5},
        inputs={
            "face": [
                FeatureOutputRef(generated.feature_id, "face/1", "face")
            ]
        },
    )
    assert geometry.regenerate_features().success
    old_children = tuple(geometry.faces)

    geometry.features.update(
        generated.feature_id,
        parameters={"length": 4.0, "width": 2.0},
    )
    report = geometry.regenerate_features()

    assert report.success
    assert geometry.features.get(split.feature_id).state == "ok"
    assert len(geometry.faces) == 2
    assert all(len(geometry.resolve_ref(EntityRef("face", item))) == 1 for item in old_children)
    assert geometry.validate_topology() == ()


def test_design_snapshot_restores_feature_definitions_and_topology() -> None:
    geometry, _first, second, _line = _feature_line()
    snapshot = geometry.design_snapshot()
    geometry.features.get(second.feature_id).parameters["position"] = [4.0, 0.0, 0.0]
    assert geometry.regenerate_features().success

    geometry.restore_design(snapshot)

    assert geometry.features.get(second.feature_id).parameters["position"] == [1.0, 0.0, 0.0]
    assert sorted(geometry.vertices) == [1, 2]
    assert geometry.edge_length(1) == pytest.approx(1.0)


def test_insert_model_is_atomic_and_copies_owner_data() -> None:
    source = plate(2.0, 1.0, semantic_group="deck")
    face = source.group("deck")[0]
    source.faces[face.id].metadata["purpose"] = "qualification"
    source.tag(face, "primary")
    source_before = to_dict(source)
    destination = GeometryModel()
    destination.add_point(9.0, 9.0, 9.0)

    result = destination.insert_model(source, group_prefix="import")

    inserted_face = result.outputs["face/1"]
    assert inserted_face.id == 1
    assert destination.group("import/deck") == (inserted_face,)
    assert destination.tags_for(inserted_face) == ("primary",)
    assert destination.faces[inserted_face.id].metadata["purpose"] == "qualification"
    assert to_dict(source) == source_before
    assert destination.validate_topology() == ()

    invalid = source.clone()
    invalid.edges[1].start = 999
    before = to_dict(destination)
    with pytest.raises(GeometryError, match="invalid geometry"):
        destination.insert_model(invalid)
    assert to_dict(destination) == before


def test_clone_copy_mirror_and_patterns_are_independent_and_valid() -> None:
    geometry = plate(2.0, 1.0, semantic_group="deck")
    original = geometry.group("deck")[0]
    clone = geometry.clone()
    clone.move_point(1, -5.0, 0.0, 0.0)
    assert geometry.vertex_position(1) == pytest.approx((0.0, 0.0, 0.0))

    translate = np.eye(4)
    translate[0, 3] = 3.0
    copied = copy_entities(geometry, (original,), matrix=translate)
    copied_face = copied.entity_map[original]
    assert measure(geometry, copied_face).value == pytest.approx(2.0)

    mirrored = mirror_entities(
        geometry, (original,), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)
    )
    assert mirrored.entity_map[original].id in geometry.faces

    linear = linear_pattern(
        geometry, (original,), (0.0, 1.0, 0.0), 2.0, 2
    )
    circular = circular_pattern(
        geometry,
        (original,),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        np.pi / 2.0,
        2,
    )
    assert len(linear.instances) == len(circular.instances) == 2
    assert geometry.validate_topology() == ()


def test_reverse_and_measure_preserve_topology_and_flip_orientation() -> None:
    geometry = plate(2.0, 1.0)
    face = next(iter(geometry.faces))
    edge = geometry.faces[face].loop[0].edge
    samples = geometry.sample_edge(edge, np.linspace(0.0, 1.0, 9))
    normal = geometry.face_normal(face, 0.5, 0.5)

    reverse_edge(geometry, edge)
    assert geometry.sample_edge(
        edge, np.linspace(0.0, 1.0, 9)
    ) == pytest.approx(samples[::-1])
    assert geometry.validate_topology() == ()

    reverse_face(geometry, face)
    assert geometry.face_normal(face, 0.5, 0.5) == pytest.approx(-normal)
    assert measure(
        geometry, EntityRef("face", face), quantity="area"
    ).value == pytest.approx(2.0)
    assert measure(
        geometry, EntityRef("face", face), quantity="perimeter"
    ).value == pytest.approx(6.0)
    assert geometry.validate_topology() == ()
