"""Persistent identity, lineage, topology, and curve behavior."""

from __future__ import annotations

import numpy as np
import pytest

from anygeometry import Arc, EntityRef, GeometryError, GeometryModel, Spline
from anygeometry.generators import cone


def test_ids_are_monotonic_per_entity_kind_and_not_reused() -> None:
    geometry = GeometryModel()
    first, second, removed = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0))
    )
    edge = geometry.add_line(first, second)

    geometry.remove_vertex(removed)

    assert (first, second, removed, geometry.add_point(3.0, 0.0, 0.0)) == (1, 2, 3, 4)
    assert edge == 1
    assert geometry.entity_ref("vertex", first) == EntityRef("vertex", 1)
    with pytest.raises(GeometryError, match="no face 99"):
        geometry.entity_ref("face", 99)
    with pytest.raises(GeometryError, match="unknown entity kind"):
        geometry.entity_ref("solid", 1)


def test_split_edge_updates_groups_tags_faces_and_cascading_history(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, edges = rectangle
    original = EntityRef("edge", edges[0])
    geometry.add_to_group("loaded_boundary", (original,))
    geometry.tag(original, "pressure_edge", "selected")
    geometry.begin_replacement_log()

    _point, halves = geometry.split_edge(original.id, 0.25)
    first_halves = tuple(EntityRef("edge", edge) for edge in halves)

    assert original.id not in geometry.edges
    assert set(geometry.resolve_ref(original)) == set(first_halves)
    assert set(geometry.group("loaded_boundary")) == set(first_halves)
    assert all(
        geometry.tags_for(reference) == ("pressure_edge", "selected")
        for reference in first_halves
    )
    assert len(geometry.faces[face].loop) == 5
    assert geometry.replacement_log() == [(original, first_halves)]

    _point, grandchildren = geometry.split_edge(halves[0], 0.5)
    current = set(geometry.resolve_ref(original))
    assert current == {
        EntityRef("edge", grandchildren[0]),
        EntityRef("edge", grandchildren[1]),
        first_halves[1],
    }
    assert set(geometry.group("loaded_boundary")) == current
    assert geometry.validate_topology() == ()


def test_topology_snapshot_restores_identity_groups_tags_and_history(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, edges = rectangle
    face_ref = EntityRef("face", face)
    edge_ref = EntityRef("edge", edges[0])
    geometry.add_to_group("shell", (face_ref,))
    geometry.tag(edge_ref, "boundary")
    snapshot = geometry.topology_snapshot()
    id_state = geometry.id_state()

    geometry.split_edge(edge_ref.id, 0.5)
    geometry.restore_topology(snapshot)

    assert geometry.id_state() == id_state
    assert edge_ref.id in geometry.edges
    assert geometry.group("shell") == (face_ref,)
    assert geometry.tags_for(edge_ref) == ("boundary",)
    assert geometry.replacement_history() == {}
    assert geometry.validate_topology() == ()


def test_topology_snapshot_restores_mutated_vertex_and_edge_state() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    edge = geometry.add_line(first, second)
    snapshot = geometry.topology_snapshot()

    geometry.vertices[first].position[:] = (4.0, 5.0, 6.0)
    geometry.edges[edge].start, geometry.edges[edge].end = second, first
    geometry.restore_topology(snapshot)

    assert geometry.vertex_position(first) == pytest.approx((0.0, 0.0, 0.0))
    assert (geometry.edges[edge].start, geometry.edges[edge].end) == (first, second)
    assert geometry.validate_topology() == ()


def test_topology_snapshot_can_restore_nested_mutable_state_repeatedly() -> None:
    geometry = GeometryModel()
    face_id = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    face = geometry.faces[face_id]
    face.metadata["nested"] = {"values": [1, 2]}
    snapshot = geometry.topology_snapshot()

    for replacement in ((8, 9), (10, 11)):
        face.metadata["nested"]["values"][:] = replacement
        assert face.surface is not None
        face.surface.origin[:] = (7.0, 7.0, 7.0)  # type: ignore[union-attr]
        geometry.restore_topology(snapshot)
        face = geometry.faces[face_id]
        assert face.metadata == {"nested": {"values": [1, 2]}}
        assert face.surface is not None
        assert face.surface.origin == pytest.approx((0.0, 0.0, 0.0))  # type: ignore[union-attr]


def test_move_point_rebuilds_surface_and_rejects_degenerate_curves_atomically() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    face = geometry.add_plate(vertices)

    geometry.move_point(vertices[2], 2.0, 2.0, 0.0)

    assert geometry.face_point(face, 1.0, 1.0) == pytest.approx((2.0, 2.0, 0.0))
    assert geometry.validate_topology() == ()

    curved = GeometryModel()
    start, via, end = curved.add_points(
        ((0.0, 0.0, 0.0), (0.5, 0.5, 0.0), (1.0, 0.0, 0.0))
    )
    curved.add_arc(start, via, end)
    before = curved.vertex_position(via).copy()

    with pytest.raises(GeometryError, match="invalid arc geometry"):
        curved.move_point(via, 0.5, 0.0, 0.0)

    assert curved.vertex_position(via) == pytest.approx(before)
    assert curved.validate_topology() == ()

    triangle = GeometryModel()
    triangle_vertices = triangle.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (0.0, 2.0, 0.0))
    )
    triangle_face = triangle.add_face(
        triangle.add_polyline(triangle_vertices, close=True)
    )
    triangle.move_point(triangle_vertices[1], 3.0, 0.0, 0.0)
    assert np.all(np.isfinite(triangle.face_point(triangle_face, 0.2, 0.2)))
    assert triangle.validate_topology() == ()

    curved_face = cone(2.0, 0.0, 3.0, circumferential_segments=4)
    apex = curved_face.group("top")[0].id
    curved_before = curved_face.vertex_position(apex).copy()
    with pytest.raises(GeometryError, match="without a valid evaluable surface"):
        curved_face.move_point(apex, 0.1, 0.0, 3.0)
    assert curved_face.vertex_position(apex) == pytest.approx(curved_before)
    assert curved_face.validate_topology() == ()


def test_public_removal_retires_groups_tags_and_reference() -> None:
    geometry = GeometryModel()
    face = geometry.add_plate(
        geometry.add_points(
            ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
        )
    )
    reference = EntityRef("face", face)
    geometry.add_to_group("shell", (reference,))
    geometry.tag(reference, "selected")

    geometry.remove_face(face)

    assert geometry.resolve_ref(reference) == ()
    assert geometry.group("shell") == ()
    assert geometry.tags_for(reference) == ()
    assert geometry.replacement_history()[reference] == ()
    assert geometry.validate_topology() == ()


def test_replacement_api_rejects_invalid_kinds_survivors_and_cycles() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    with pytest.raises(GeometryError, match="surviving"):
        geometry.record_replacement(
            EntityRef("vertex", first), (EntityRef("vertex", second),)
        )
    geometry.remove_vertex(first, record=False)
    with pytest.raises(GeometryError, match="invalid entity kind"):
        geometry.record_replacement(
            EntityRef("vertex", first), (EntityRef("solid", 1),)  # type: ignore[arg-type]
        )

    geometry._replacement_history[EntityRef("vertex", first)] = (  # noqa: SLF001
        EntityRef("vertex", second),
    )
    geometry._replacement_history[EntityRef("vertex", second)] = (  # noqa: SLF001
        EntityRef("vertex", first),
    )
    errors = geometry.validate_topology()
    assert any("cycle" in error for error in errors)


def test_unordered_loop_is_oriented_and_open_chain_is_rejected() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0))
    )
    edges = geometry.add_polyline(vertices, close=True)
    face = geometry.add_face((edges[2], edges[0], edges[3], edges[1]))

    loop = geometry.faces[face].loop
    assert geometry.faces[face].corners == (0, 1, 2, 3)
    assert all(
        geometry.oriented_end_vertex(current)
        == geometry.oriented_start_vertex(following)
        for current, following in zip(loop, loop[1:] + loop[:1])
    )

    open_geometry = GeometryModel()
    points = open_geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0))
    )
    with pytest.raises(GeometryError, match="closed"):
        open_geometry.add_face(open_geometry.add_polyline(points))


def test_topology_validation_detects_self_intersection_and_zero_length() -> None:
    geometry = GeometryModel()
    vertices = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 2.0, 0.0), (0.0, 2.0, 0.0), (2.0, 0.0, 0.0))
    )
    geometry.add_face(geometry.add_polyline(vertices, close=True))

    errors = geometry.validate_topology()
    assert any("self-intersects" in error for error in errors)

    line_geometry = GeometryModel()
    first, second = line_geometry.add_points(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    )
    line_geometry.add_line(first, second)
    before = line_geometry.vertex_position(second).copy()
    with pytest.raises(GeometryError, match="zero geometric length"):
        line_geometry.move_point(second, 0.0, 0.0, 0.0)
    assert line_geometry.vertex_position(second) == pytest.approx(before)


def test_split_side_preserves_mapped_corners_and_lengths(
    rectangle: tuple[GeometryModel, int, tuple[int, ...], tuple[int, ...]],
) -> None:
    geometry, face, _vertices, edges = rectangle
    before = geometry.face_side_lengths(face)

    geometry.split_edge(edges[0], 0.3)

    assert [len(side) for side in geometry.faces[face].sides()] == [2, 1, 1, 1]
    assert geometry.face_side_lengths(face) == pytest.approx(before)


def test_arc_and_spline_split_preserve_curve_and_endpoints() -> None:
    geometry = GeometryModel()
    arc_vertices = geometry.add_points(
        ((2.0, 0.0, 0.0), (np.sqrt(2.0), np.sqrt(2.0), 0.0), (0.0, 2.0, 0.0))
    )
    arc = geometry.add_arc(*arc_vertices)
    arc_length = geometry.edge_length(arc)
    _point, arc_halves = geometry.split_edge(arc, 0.5)

    assert all(isinstance(geometry.edges[edge].curve, Arc) for edge in arc_halves)
    assert sum(geometry.edge_length(edge) for edge in arc_halves) == pytest.approx(arc_length)

    start, control_a, control_b, end = geometry.add_points(
        ((0.0, 0.0, 1.0), (1.0, 2.0, 1.0), (2.0, 2.0, 1.0), (3.0, 0.0, 1.0))
    )
    spline = geometry.add_spline(start, (control_a, control_b), end)
    expected_midpoint = geometry.sample_edge(spline, np.asarray([0.4]))[0]
    new_vertex, spline_halves = geometry.split_edge(spline, 0.4)

    assert geometry.vertex_position(new_vertex) == pytest.approx(expected_midpoint)
    assert all(isinstance(geometry.edges[edge].curve, Spline) for edge in spline_halves)
    joined = np.vstack(
        (
            geometry.sample_edge(spline_halves[0], np.linspace(0.0, 1.0, 9)),
            geometry.sample_edge(spline_halves[1], np.linspace(0.0, 1.0, 9))[1:],
        )
    )
    assert joined[0] == pytest.approx((0.0, 0.0, 1.0))
    assert joined[-1] == pytest.approx((3.0, 0.0, 1.0))


def test_extruded_arc_and_revolved_line_remain_exact() -> None:
    radius = 2.0
    geometry = GeometryModel()
    start, via, end = geometry.add_points(
        ((radius, 0.0, 0.0), (radius / np.sqrt(2.0), radius / np.sqrt(2.0), 0.0), (0.0, radius, 0.0))
    )
    arc = geometry.add_arc(start, via, end)
    face = geometry.extrude((arc,), (0.0, 0.0, 3.0))[0]
    top_edge = geometry.faces[face].loop[2].edge
    samples = geometry.sample_edge(top_edge, np.linspace(0.0, 1.0, 9))

    assert np.linalg.norm(samples[:, :2], axis=1) == pytest.approx(radius)
    assert samples[:, 2] == pytest.approx(np.full(9, 3.0))

    revolved = GeometryModel()
    bottom, top = revolved.add_points(((radius, 0.0, 0.0), (radius, 0.0, 3.0)))
    profile = revolved.add_line(bottom, top)
    faces = revolved.revolve((profile,), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0), 2.0 * np.pi)
    assert len(faces) == 4
    assert revolved.validate_topology() == ()
    assert len(revolved.vertices) == len(
        {tuple(np.round(vertex.position, 12)) for vertex in revolved.vertices.values()}
    )
