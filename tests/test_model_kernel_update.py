"""Integrated contracts for model identity, journals, and structural ownership."""

from __future__ import annotations

from uuid import UUID, uuid4

import numpy as np
import pytest

from anygeometry import EntityRef, GeometryError, GeometryModel
from anygeometry.identity import EntityHandle, ResolutionStatus


def test_handles_are_model_bound_and_clone_receives_new_identity() -> None:
    with pytest.raises(GeometryError, match="nil UUID"):
        GeometryModel(model_id=UUID(int=0))
    first = GeometryModel()
    second = GeometryModel()
    vertex = first.add_point(0.0, 0.0, 0.0)

    handle = first.handle("vertex", vertex)

    assert handle != EntityHandle(second.model_id, "vertex", vertex)
    assert first.resolve_handle(handle).status is ResolutionStatus.ACTIVE
    wrong = second.resolve_handle(handle)
    assert wrong.status is ResolutionStatus.WRONG_MODEL
    assert second.clone().model_id != second.model_id


def test_handle_validation_rejects_coercion_and_structural_deletion_is_terminal() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.handle("vertex", True)
    with pytest.raises(GeometryError, match="positive integer"):
        geometry.handle("vertex", 1.9)  # type: ignore[arg-type]
    assert geometry.handle("vertex", vertex).id == vertex

    part = geometry.add_part()
    handle = geometry.handle("part", part)
    geometry.remove_part(part)
    assert geometry.resolve_handle(handle).status is ResolutionStatus.DELETED
    never_allocated = EntityHandle(geometry.model_id, "part", part + 10)
    assert geometry.resolve_handle(never_allocated).status is ResolutionStatus.UNKNOWN


def test_failed_delta_transaction_rolls_back_without_reusing_ids() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    geometry.add_line(first, second)
    original_revision = geometry.revision

    with pytest.raises(GeometryError, match="zero geometric length"):
        geometry.move_point(second, 0.0, 0.0, 0.0)

    assert geometry.vertex_position(second) == pytest.approx((1.0, 0.0, 0.0))
    assert geometry.revision == original_revision
    made = geometry.add_point(2.0, 0.0, 0.0)
    assert made == 3
    assert geometry.last_change_set.added == (("vertex", 3),)


def test_local_point_edit_reports_only_dependency_closure() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (100, 0, 0), (101, 0, 0))
    )
    face = geometry.add_plate(points[:4])
    unrelated = geometry.add_line(points[4], points[5])

    geometry.move_point(points[2], 1.5, 1.0, 0.0)

    changed = set(geometry.last_change_set.changed)
    assert ("vertex", points[2]) in changed
    assert ("face", face) in changed
    assert ("edge", unrelated) not in changed
    assert len(changed) <= 4


def test_member_identity_survives_axis_edge_subdivision() -> None:
    geometry = GeometryModel()
    a, b, c = geometry.add_points(((0, 0, 0), (1, 0, 0), (2, 0, 0)))
    first = geometry.add_line(a, b)
    second = geometry.add_line(b, c)
    member = geometry.add_member((first, second), name="girder")
    before = geometry.members[member]

    _point, children = geometry.split_edge(first, 0.25)

    after = geometry.members[member]
    assert after.id == before.id == member
    assert len(after.edge_use_ids) == 3
    used_edges = {
        geometry.member_edge_uses[item].edge_id for item in after.edge_use_ids
    }
    assert set(children) <= used_edges
    assert second in used_edges
    assert first not in geometry.edges
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_sheet_coedges_follow_hole_and_edge_subdivision() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0, 0, 0), (4, 0, 0), (4, 4, 0), (0, 4, 0))
    )
    face = geometry.add_plate(points)
    sheet = geometry.add_sheet((face,))
    use = geometry.face_uses[geometry.sheets[sheet].face_use_ids[0]]
    edge = geometry.coedges[use.loops[0][0]].edge_id

    _point, children = geometry.split_edge(edge, 0.5)

    updated = geometry.face_uses[use.id]
    coedge_edges = {
        geometry.coedges[identifier].edge_id for identifier in updated.coedge_ids
    }
    assert edge not in coedge_edges
    assert set(children) <= coedge_edges
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_spatial_index_updates_incrementally_after_local_edit() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    edge = geometry.add_line(first, second)
    assert ("edge", edge) in geometry.spatial_candidates((-1, -1, -1), (2, 1, 1))

    geometry.move_point(second, -5.0, 5.0, 0.0)

    assert ("edge", edge) in geometry.last_change_set.spatial_updates
    assert ("edge", edge) not in geometry.spatial_candidates((0.9, -0.1, -0.1), (1.1, 0.1, 0.1))
    assert ("edge", edge) in geometry.spatial_candidates((-1, -1, -1), (2, 6, 1))


def test_caught_nested_failure_marks_outer_transaction_for_rollback() -> None:
    geometry = GeometryModel()

    with pytest.raises(GeometryError, match="nested edit failed"):
        with geometry.transaction():
            try:
                with geometry.transaction():
                    geometry.add_point(1.0, 2.0, 3.0)
                    raise RuntimeError("deliberate nested failure")
            except RuntimeError:
                pass

    assert geometry.vertices == {}
    assert geometry.revision == 0
    assert geometry.add_point(0.0, 0.0, 0.0) == 2


def test_compatibility_snapshot_restore_never_reuses_allocated_ids() -> None:
    geometry = GeometryModel()
    geometry.add_point(0.0, 0.0, 0.0)
    snapshot = geometry.topology_snapshot()
    discarded = geometry.add_point(1.0, 0.0, 0.0)

    geometry.restore_topology(snapshot)

    assert discarded not in geometry.vertices
    assert geometry.add_point(2.0, 0.0, 0.0) == discarded + 1


def test_rollback_discards_values_cached_from_provisional_geometry() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    edge = geometry.add_line(first, second)
    assert geometry.edge_length(edge) == pytest.approx(1.0)

    with pytest.raises(RuntimeError, match="rollback"):
        with geometry.transaction():
            geometry.move_point(second, 5.0, 0.0, 0.0)
            assert geometry.edge_length(edge) == pytest.approx(5.0)
            provisional = geometry.add_point(9.0, 9.0, 9.0)
            assert ("vertex", provisional) in geometry.spatial_candidates(
                (8.5, 8.5, 8.5), (9.5, 9.5, 9.5)
            )
            raise RuntimeError("rollback")

    assert geometry.edge_length(edge) == pytest.approx(1.0)
    assert geometry.spatial_candidates(
        (8.5, 8.5, 8.5), (9.5, 9.5, 9.5)
    ) == ()


def test_removal_rejects_live_structural_references_atomically() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    edge = geometry.add_line(first, second)
    geometry.add_member((edge,))
    revision = geometry.revision

    with pytest.raises(GeometryError, match="member uses"):
        geometry.remove_edge(edge)

    assert edge in geometry.edges
    assert geometry.revision == revision
    assert geometry._validate_structural() == ()  # noqa: SLF001


def test_semantic_views_and_change_hooks_cannot_bypass_owner_mutation() -> None:
    geometry = GeometryModel()
    vertex = geometry.add_point(0.0, 0.0, 0.0)
    reference = EntityRef("vertex", vertex)
    geometry.add_to_group("selected", (reference,))
    geometry.tag(reference, "primary")

    with pytest.raises(TypeError):
        geometry.groups["unsafe"] = frozenset()  # type: ignore[index]
    with pytest.raises(AttributeError):
        geometry.groups["selected"].add(reference)  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        geometry.tags[reference] = frozenset(("unsafe",))  # type: ignore[index]

    events = []

    def mutating_observer(change) -> None:
        events.append(change)
        geometry.add_point(9.0, 9.0, 9.0)

    geometry.add_change_hook(mutating_observer)
    revision = geometry.revision
    geometry.add_point(1.0, 0.0, 0.0)

    assert geometry.revision == revision + 1
    assert len(events) == 1
    assert len(geometry.vertices) == 2
    assert events[0].revision_before == revision


def test_document_identity_revision_and_coordinate_state_are_owner_controlled() -> None:
    geometry = GeometryModel()
    model_id = geometry.model_id
    with pytest.raises(AttributeError):
        geometry.model_id = uuid4()  # type: ignore[misc]
    with pytest.raises(AttributeError):
        geometry.revision = 0  # type: ignore[misc]
    with pytest.raises(ValueError):
        geometry.local_origin[:] = (1.0, 2.0, 3.0)

    events = []
    geometry.add_change_hook(events.append)
    geometry.set_document_settings(
        units="mm",
        local_origin=(100.0, 200.0, 300.0),
        coordinate_transform=(
            (1.0, 0.0, 0.0, 10.0),
            (0.0, 1.0, 0.0, 20.0),
            (0.0, 0.0, 1.0, 30.0),
            (0.0, 0.0, 0.0, 1.0),
        ),
    )

    assert geometry.model_id == model_id
    assert geometry.units == "mm"
    assert tuple(geometry.local_origin) == (100.0, 200.0, 300.0)
    assert geometry.coordinate_transform is not None
    with pytest.raises(ValueError):
        geometry.coordinate_transform[0, 0] = 5.0
    assert len(events) == 1
    assert events[0].document_settings_changed

    nearly_singular = np.eye(4)
    nearly_singular[2, 2] = np.finfo(float).eps / 2.0
    with pytest.raises(GeometryError, match="invertible"):
        geometry.set_document_settings(coordinate_transform=nearly_singular)


def test_spatial_queries_observe_provisional_and_dependent_face_bounds() -> None:
    geometry = GeometryModel()
    points = geometry.add_points(
        ((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0))
    )
    face = geometry.add_plate(points)
    geometry.spatial_candidates((-1, -1, -1), (2, 2, 1))

    with geometry.transaction():
        provisional = geometry.add_point(9.0, 9.0, 9.0)
        assert ("vertex", provisional) in geometry.spatial_candidates(
            (8.9, 8.9, 8.9), (9.1, 9.1, 9.1)
        )
        geometry.move_point(points[2], 5.0, 5.0, 0.0)
        assert ("face", face) in geometry.spatial_candidates(
            (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), kinds=("face",)
        )

    assert ("face", face) in geometry.spatial_candidates(
        (4.9, 4.9, -0.1), (5.1, 5.1, 0.1), kinds=("face",)
    )


def test_snapshot_restore_is_staged_atomic_and_notifies_revision_consumers() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0, 0, 0), (1, 0, 0)))
    geometry.add_line(first, second)
    snapshot = geometry.topology_snapshot()
    baseline_keys = geometry.entity_keys()
    baseline_revision = geometry.revision

    malformed = dict(snapshot)
    malformed["vertex_state"] = dict(snapshot["vertex_state"])  # type: ignore[arg-type]
    malformed["vertex_state"][first] = [1.0, 2.0]  # type: ignore[index]
    with pytest.raises(GeometryError, match="finite 3-vector"):
        geometry.restore_topology(malformed)

    assert geometry.entity_keys() == baseline_keys
    assert geometry.revision == baseline_revision

    discarded = geometry.add_point(2.0, 0.0, 0.0)
    events = []
    geometry.add_change_hook(events.append)
    revision = geometry.revision
    geometry.restore_topology(snapshot)

    assert geometry.revision == revision + 1
    assert len(events) == 1
    assert ("vertex", discarded) in events[0].removed


def test_change_hooks_cannot_restore_topology_or_design() -> None:
    geometry = GeometryModel()
    geometry.add_point(0.0, 0.0, 0.0)
    topology = geometry.topology_snapshot()
    design = geometry.design_snapshot()
    attempts: list[str] = []

    def restoring_observer(_change) -> None:
        for label, callback in (
            ("topology", lambda: geometry.restore_topology(topology)),
            ("design", lambda: geometry.restore_design(design)),
        ):
            with pytest.raises(GeometryError, match="change hook"):
                callback()
            attempts.append(label)

    geometry.add_change_hook(restoring_observer)
    made = geometry.add_point(1.0, 0.0, 0.0)

    assert made in geometry.vertices
    assert attempts == ["topology", "design"]
