"""Positive-area plate overlap is fragmented without deleting structural area."""

from __future__ import annotations

import pytest

from anygeometry import (
    EntityRef,
    GeometryError,
    GeometryModel,
    find_coplanar_overlaps,
    fragment_coplanar_overlaps,
    to_dict,
)


def _overlapping() -> tuple[GeometryModel, int, int]:
    geometry = GeometryModel()
    first_points = geometry.add_points(
        ((0, 0, 0), (2, 0, 0), (2, 1, 0), (0, 1, 0))
    )
    second_points = geometry.add_points(
        ((1, 0, 0), (3, 0, 0), (3, 1, 0), (1, 1, 0))
    )
    first = geometry.add_plate(first_points)
    second = geometry.add_plate(second_points)
    geometry.set_face_metadata(first, {"section": "first"})
    geometry.set_face_metadata(second, {"section": "second"})
    return geometry, first, second


def test_overlap_audit_ignores_shared_boundary_but_reports_positive_area():
    geometry, first, second = _overlapping()
    overlap = find_coplanar_overlaps(geometry)[0]
    assert (overlap.first, overlap.second, overlap.area) == pytest.approx(
        (first, second, 1.0)
    )

    touching = GeometryModel()
    one = touching.add_plate(
        touching.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    )
    two = touching.add_plate(
        touching.add_points(((1, 0, 0), (2, 0, 0), (2, 1, 0), (1, 1, 0)))
    )
    assert {one, two} == set(touching.faces)
    assert find_coplanar_overlaps(touching) == ()


def test_fragmentation_makes_a_only_overlap_and_b_only_conformal_plates():
    geometry, first, second = _overlapping()
    original_edges = tuple(
        item.edge for face_id in (first, second) for item in geometry.faces[face_id].loop
    )

    result = fragment_coplanar_overlaps(geometry, (first, second))

    assert len(geometry.faces) == 3
    assert len(result.descendants[first]) == 2
    assert len(result.descendants[second]) == 1
    assert len(result.overlap_faces) == 1
    assert result.overlap_area == pytest.approx(1.0)
    assert find_coplanar_overlaps(geometry) == ()
    assert geometry.validate_topology() == ()
    assert geometry.faces[result.overlap_faces[0]].metadata["section"] == "first"
    assert geometry.resolve_ref(EntityRef("face", first)) == tuple(
        EntityRef("face", item) for item in result.descendants[first]
    )
    assert all(
        geometry.resolve_ref(EntityRef("edge", edge_id))
        for edge_id in original_edges
    )
    # The three unit rectangles cover the original union area of 3 m2.
    assert sum(
        abs(
            sum(
                geometry.vertex_position(geometry.oriented_start_vertex(item))[0]
                * geometry.vertex_position(geometry.oriented_end_vertex(item))[1]
                - geometry.vertex_position(geometry.oriented_end_vertex(item))[0]
                * geometry.vertex_position(geometry.oriented_start_vertex(item))[1]
                for item in face.loop
            )
        )
        / 2.0
        for face in geometry.faces.values()
    ) == pytest.approx(3.0)


def test_overlap_feature_regenerates_and_failed_operation_is_atomic():
    geometry, first, second = _overlapping()
    geometry.features.capture_baseline(geometry)
    record = geometry.features.append(
        "geometry.fragment.overlaps",
        inputs={"faces": (EntityRef("face", first), EntityRef("face", second))},
    )
    report = geometry.regenerate_features()
    assert report.success
    assert len(geometry.features.get(record.feature_id).outputs) == 3

    separate = GeometryModel()
    a = separate.add_plate(
        separate.add_points(((0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)))
    )
    b = separate.add_plate(
        separate.add_points(((2, 0, 0), (3, 0, 0), (3, 1, 0), (2, 1, 0)))
    )
    before = to_dict(separate)
    with pytest.raises(GeometryError, match="no positive-area overlap"):
        fragment_coplanar_overlaps(separate, (a, b))
    assert to_dict(separate) == before
