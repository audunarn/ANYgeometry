"""Persistent feature history and high-level owner editing APIs."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np
import pytest

from anygeometry import (
    AttachmentKind,
    AttachmentTargetKind,
    EntityRef,
    FeatureOutputRef,
    GeometryError,
    GeometryModel,
    circular_pattern,
    copy_entities,
    from_dict,
    insert_model,
    linear_pattern,
    measure,
    mirror_entities,
    Orientation,
    ParameterRange,
    ResolutionStatus,
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


def test_public_feature_history_edits_publish_exactly_once() -> None:
    geometry = GeometryModel()
    events = []
    geometry.add_change_hook(events.append)

    def publishes_once(action: Callable[[], object]) -> object:
        before_revision = geometry.revision
        before_events = len(events)
        result = action()
        assert geometry.revision == before_revision + 1
        assert len(events) == before_events + 1
        assert events[-1].revision_before == before_revision
        assert events[-1].revision_after == geometry.revision
        assert events[-1].feature_history_changed
        return result

    publishes_once(lambda: geometry.features.capture_baseline(geometry))
    first = publishes_once(
        lambda: geometry.features.append(
            "vendor.first", parameters={"value": 1}, suppressed=True
        )
    )
    second = publishes_once(
        lambda: geometry.features.append("vendor.second", suppressed=True)
    )
    publishes_once(
        lambda: geometry.features.update(
            first.feature_id, name="First", parameters={"value": 2}
        )
    )
    publishes_once(lambda: geometry.features.set_suppressed(first.feature_id, False))
    publishes_once(lambda: geometry.features.move(second.feature_id, 0))
    publishes_once(lambda: geometry.features.remove(second.feature_id))

    # A semantic no-op remains a no-op at the document boundary.
    before_revision = geometry.revision
    before_events = len(events)
    geometry.features.update(
        first.feature_id, name="First", parameters={"value": 2}
    )
    assert geometry.revision == before_revision
    assert len(events) == before_events


def test_feature_regeneration_publishes_topology_and_outputs_atomically() -> None:
    geometry, _first, second, _line = _feature_line()
    geometry.features.update(
        second.feature_id, parameters={"position": [3.0, 0.0, 0.0]}
    )
    seen = []

    def observe(change: object) -> None:
        current = geometry.features.get(second.feature_id)
        output = current.outputs["point"]
        seen.append(
            (
                change,  # type: ignore[misc]
                current.parameters["position"],
                geometry.vertex_position(output.id),
            )
        )

    geometry.add_change_hook(observe)
    before_revision = geometry.revision

    report = geometry.regenerate_features()

    assert report.success
    assert geometry.revision == before_revision + 1
    assert len(seen) == 1
    change, parameters, position = seen[0]
    assert change.feature_history_changed
    assert parameters == [3.0, 0.0, 0.0]
    assert position == pytest.approx((3.0, 0.0, 0.0))


def test_materialization_checksum_hashes_closure_without_document_serialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)
    reference = EntityRef("face", face)
    geometry.add_to_group("deck", (reference,))
    geometry.tag(reference, "shell")
    feature = geometry.features.append("vendor.test")
    feature.outputs = {"face": reference}

    import anygeometry.serialization as serialization

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("materialization checksum serialized the complete model")

    monkeypatch.setattr(serialization, "to_dict", forbidden)

    assert geometry.features.materialization_checksum(feature, geometry) == (
        "606449159c76f59abf8ef175de27829a7eee2dc380ef6045c1a87ceeb9e58bf6"
    )


def test_materialization_checksum_resolves_face_corner_positions_to_vertices() -> None:
    geometry = GeometryModel()
    unrelated = geometry.add_points(
        ((-3.0, 0.0, 0.0), (-2.0, 0.0, 0.0), (-1.0, 0.0, 0.0))
    )
    corners = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(corners)
    feature = geometry.features.append("vendor.test")
    feature.outputs = {"face": EntityRef("face", face)}

    checksum = geometry.features.materialization_checksum(feature, geometry)
    geometry.move_point(unrelated[0], -30.0, 0.0, 0.0)

    assert geometry.features.materialization_checksum(feature, geometry) == checksum

    geometry.move_point(corners[0], -0.25, 0.0, 0.0)
    assert geometry.features.materialization_checksum(feature, geometry) != checksum


def test_feature_regeneration_uses_fresh_ids_and_preserves_lineage() -> None:
    geometry, _first, second, _line = _feature_line()
    old_outputs = {
        key: reference
        for feature in geometry.features.records
        for key, reference in feature.outputs.items()
    }
    old_state = geometry.id_state()

    geometry.features.update(
        second.feature_id, parameters={"position": [2.0, 0.0, 0.0]}
    )
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
    outputs = [dict(item.outputs) for item in geometry.features.records]
    geometry.features.update(
        second.feature_id, parameters={"position": [0.0, 2.0, 0.0, 7.0]}
    )
    # Editing the definition is itself a published document change.  A failed
    # replay must leave the materialized topology (and revision) at that
    # post-edit state.
    before = to_dict(geometry, include_features=False)

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

    assert document["version"] == 4
    assert to_dict(restored) == document
    assert len(restored.features.records) == 3

    current = to_dict(plate(2.0, 1.0))
    legacy = {
        key: deepcopy(value)
        for key, value in current.items()
        if key
        in {
            "schema",
            "vertices",
            "edges",
            "faces",
            "groups",
            "tags",
            "replacement_history",
        }
    }
    legacy["version"] = 1
    migrated = from_dict(legacy)
    assert migrated.features.records == []
    assert migrated.features.baseline is not None
    assert to_dict(migrated)["version"] == 4


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
    geometry.features.update(
        second.feature_id, parameters={"position": [4.0, 0.0, 0.0]}
    )
    assert geometry.regenerate_features().success

    geometry.restore_design(snapshot)

    assert geometry.features.get(second.feature_id).parameters["position"] == [1.0, 0.0, 0.0]
    assert sorted(geometry.vertices) == [1, 2]
    assert geometry.edge_length(1) == pytest.approx(1.0)


def test_design_snapshot_restore_stages_invalid_features_and_notifies_once() -> None:
    geometry, _first, second, _line = _feature_line()
    snapshot = geometry.design_snapshot()
    geometry.features.update(
        second.feature_id, parameters={"position": [7.0, 0.0, 0.0]}
    )
    geometry.add_point(8.0, 8.0, 8.0)
    events = []
    geometry.add_change_hook(events.append)
    revision = geometry.revision

    malformed = dict(snapshot)
    malformed["features"] = {
        "baseline": None,
        "records": [object()],
        "next_id": 2,
    }
    with pytest.raises(GeometryError, match="invalid feature snapshot"):
        geometry.restore_design(malformed)

    assert geometry.revision == revision
    assert len(geometry.vertices) == 3
    assert events == []

    geometry.restore_design(snapshot)
    assert geometry.revision == revision + 1
    assert len(events) == 1
    assert events[0].feature_history_changed


def test_insert_model_is_atomic_and_copies_owner_data() -> None:
    source = plate(2.0, 1.0, semantic_group="deck")
    face = source.group("deck")[0]
    source.set_face_metadata(face.id, {"purpose": "qualification"})
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
    # Manufacture an invalid source without exposing a public mutation path.
    object.__setattr__(invalid._edges[1], "start", 999)  # noqa: SLF001
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


def test_full_model_insert_preserves_sheet_and_member_identity_closures() -> None:
    source = GeometryModel()
    vertices = source.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = source.add_plate(vertices)
    axis = source.faces[face].loop[0].edge
    part = source.add_part(name="panel")
    source.add_sheet((face,), part_id=part, name="plating")
    source.add_member((axis,), part_id=part, name="edge member")

    destination = GeometryModel()
    result = insert_model(destination, source)

    assert result.mapped(EntityRef("face", face)).id in destination.faces
    assert len(destination.parts) == 1
    assert len(destination.sheets) == 1
    assert len(destination.face_uses) == 1
    assert len(destination.coedges) == 4
    assert len(destination.members) == 1
    assert len(destination.member_edge_uses) == 1
    assert destination._validate_structural() == ()  # noqa: SLF001


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


def test_reverse_edge_preserves_persistent_coedge_ids_and_handles() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    sheet = geometry.add_sheet((face,))
    face_use_id = geometry.sheets[sheet].face_use_ids[0]
    before = geometry.face_uses[face_use_id]
    edge = geometry.faces[face].loop[0].edge
    coedge_id = next(
        identifier
        for identifier in before.coedge_ids
        if geometry.coedges[identifier].edge_id == edge
    )
    original_ids = before.coedge_ids
    original_orientation = geometry.coedges[coedge_id].orientation
    handle = geometry.handle("coedge", coedge_id)
    next_coedge_id = geometry._next_structural_id["coedge"]  # noqa: SLF001

    reverse_edge(geometry, edge)

    assert geometry.face_uses[face_use_id].coedge_ids == original_ids
    assert geometry.coedges[coedge_id].orientation is not original_orientation
    assert geometry._next_structural_id["coedge"] == next_coedge_id  # noqa: SLF001
    assert ("coedge", coedge_id) in geometry.last_change_set.ownership_changes
    resolution = geometry.resolve_handle(handle)
    assert resolution.status is ResolutionStatus.ACTIVE
    assert resolution.resolved == (handle,)
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_reverse_face_preserves_and_reorders_persistent_coedge_ids() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            (
                (0.0, 0.0, 0.0),
                (2.0, 0.0, 0.0),
                (2.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        )
    )
    sheet = geometry.add_sheet((face,))
    face_use_id = geometry.sheets[sheet].face_use_ids[0]
    original_loop = geometry.face_uses[face_use_id].loops[0]
    original_orientations = {
        identifier: geometry.coedges[identifier].orientation
        for identifier in original_loop
    }
    handles = tuple(
        geometry.handle("coedge", identifier) for identifier in original_loop
    )
    next_coedge_id = geometry._next_structural_id["coedge"]  # noqa: SLF001

    reverse_face(geometry, face)

    assert geometry.face_uses[face_use_id].loops == (tuple(reversed(original_loop)),)
    assert set(geometry.coedges) == set(original_loop)
    assert geometry._next_structural_id["coedge"] == next_coedge_id  # noqa: SLF001
    assert {
        ("coedge", identifier) for identifier in original_loop
    }.issubset(geometry.last_change_set.ownership_changes)
    assert ("face_use", face_use_id) in geometry.last_change_set.ownership_changes
    for identifier, orientation in original_orientations.items():
        assert geometry.coedges[identifier].orientation is not orientation
    for handle in handles:
        resolution = geometry.resolve_handle(handle)
        assert resolution.status is ResolutionStatus.ACTIVE
        assert resolution.resolved == (handle,)
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_reverse_edge_preserves_member_axis_and_attachment_witness() -> None:
    geometry = GeometryModel()
    first, middle, last = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    reversed_edge = geometry.add_line(first, middle)
    following_edge = geometry.add_line(middle, last)
    member = geometry.add_member((reversed_edge, following_edge))
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.ENDPOINT,
        AttachmentTargetKind.EDGE,
        reversed_edge,
        ParameterRange.point(0.0),
        (ParameterRange.point(0.0),),
    )
    use_id = geometry.members[member].edge_use_ids[0]

    reverse_edge(geometry, reversed_edge)

    assert geometry.member_edge_uses[use_id].orientation is Orientation.REVERSED
    assert geometry.attachments[attachment].target_parameters == (
        ParameterRange.point(1.0),
    )
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_reverse_face_preserves_planar_attachment_witness() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    start, end = geometry.add_points(((0.4, 0.2, 0.0), (1.6, 0.2, 0.0)))
    member = geometry.add_member((geometry.add_line(start, end),))
    first_uv = geometry.face_local_uv(face, geometry.vertex_position(start))
    second_uv = geometry.face_local_uv(face, geometry.vertex_position(end))
    attachment = geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_FACE,
        AttachmentTargetKind.FACE,
        face,
        ParameterRange(0.0, 1.0),
        (
            ParameterRange(first_uv[0], second_uv[0]),
            ParameterRange(first_uv[1], second_uv[1]),
        ),
    )

    reverse_face(geometry, face)

    assert geometry.attachments[attachment].target_parameters == (
        ParameterRange(first_uv[1], second_uv[1]),
        ParameterRange(first_uv[0], second_uv[0]),
    )
    assert "attachment_inconsistent" not in geometry.strict_audit().issue_counts


def test_reverse_edge_rejects_unrepresentable_attachment_path_atomically() -> None:
    geometry = GeometryModel()
    start, end = geometry.add_points(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    edge = geometry.add_line(start, end)
    member = geometry.add_member((edge,))
    geometry.add_attachment(
        member,
        AttachmentKind.MEMBER_ON_BOUNDARY,
        AttachmentTargetKind.EDGE,
        edge,
        ParameterRange(0.2, 0.8),
        (ParameterRange(0.2, 0.8),),
    )
    before = geometry.edges[edge]
    use_before = geometry.member_edge_uses[geometry.members[member].edge_use_ids[0]]

    with pytest.raises(GeometryError, match="spans a positive interval"):
        reverse_edge(geometry, edge)

    assert geometry.edges[edge] == before
    assert geometry.member_edge_uses[use_before.id] == use_before
    assert geometry._validate_structural() == ()  # noqa: SLF001
