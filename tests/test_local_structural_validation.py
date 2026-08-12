"""Bounded structural commit validation and conservative model bounds."""

from __future__ import annotations

from dataclasses import replace

import pytest

from anygeometry import GeometryError, GeometryModel


def test_local_member_edit_accepts_an_unchanged_shared_edge_user() -> None:
    geometry = GeometryModel()
    first, second = geometry.add_points(((0.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
    edge = geometry.add_line(first, second)
    part = geometry.add_part()
    first_member, second_member = geometry.add_members(
        ((edge,), (edge,)), part_id=part
    )

    geometry.reverse_member(first_member)

    assert second_member in geometry.members
    assert geometry._validate_structural() == ()  # noqa: SLF001
    diagnostics = geometry.last_structural_validation_diagnostics
    assert ("member", first_member) in diagnostics.visited_keys
    assert ("member", second_member) not in diagnostics.visited_keys


def test_local_sheet_edit_accepts_shared_edge_coedges_from_another_sheet() -> None:
    geometry = GeometryModel()
    a, b, c, d = geometry.add_points(
        ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (1.0, 1.0, 0.0), (1.0, -1.0, 0.0))
    )
    shared = geometry.add_line(a, b)
    first_face = geometry.add_face(
        (shared, geometry.add_line(b, c), geometry.add_line(c, a))
    )
    second_face = geometry.add_face(
        (shared, geometry.add_line(b, d), geometry.add_line(d, a))
    )
    part = geometry.add_part()
    first_sheet = geometry.add_sheet((first_face,), part_id=part)
    second_sheet = geometry.add_sheet((second_face,), part_id=part)

    with geometry.transaction():
        geometry._put_structural(  # noqa: SLF001
            "sheet", replace(geometry.sheets[first_sheet], name="edited")
        )

    assert second_sheet in geometry.sheets
    assert geometry._validate_structural() == ()  # noqa: SLF001
    diagnostics = geometry.last_structural_validation_diagnostics
    assert first_sheet in diagnostics.expanded_sheets
    assert second_sheet not in diagnostics.expanded_sheets


def test_local_member_validation_is_constant_in_unrelated_member_count() -> None:
    def edited_work_set(count: int) -> int:
        geometry = GeometryModel()
        vertices = geometry.add_points(
            (float(index), float(index % 7), 0.0) for index in range(count * 2)
        )
        edges = geometry.add_lines(
            (vertices[index * 2], vertices[index * 2 + 1])
            for index in range(count)
        )
        part = geometry.add_part()
        members = geometry.add_members(((edge,) for edge in edges), part_id=part)
        geometry.reverse_member(members[0])
        diagnostics = geometry.last_structural_validation_diagnostics
        assert not diagnostics.full_model
        return diagnostics.visited_count

    assert edited_work_set(8) == edited_work_set(128)


def test_failed_local_structural_validation_rolls_back_and_diagnostics_publish() -> None:
    geometry = GeometryModel()
    a, b = geometry.add_points(((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))
    edge = geometry.add_line(a, b)
    member = geometry.add_member((edge,))
    use_id = geometry.members[member].edge_use_ids[0]
    before = geometry.last_structural_validation_diagnostics
    revision = geometry.revision

    with pytest.raises(GeometryError, match="owned by member"):
        with geometry.transaction():
            geometry._put_structural(  # noqa: SLF001
                "member_edge_use",
                replace(geometry.member_edge_uses[use_id], member_id=member + 999),
            )

    assert geometry.member_edge_uses[use_id].member_id == member
    assert geometry.revision == revision
    assert geometry.last_structural_validation_diagnostics == before


def test_coons_interior_is_in_maintained_face_bounds_and_candidates() -> None:
    geometry = GeometryModel()
    boundary = geometry.add_points(
        (
            (0.0, 0.0, 0.0), (0.5, 0.0, 1.0), (1.0, 0.0, 0.0),
            (1.0, 0.5, 1.0), (1.0, 1.0, 0.0), (0.5, 1.0, 1.0),
            (0.0, 1.0, 0.0), (0.0, 0.5, 1.0),
        )
    )
    face = geometry.add_face(
        geometry.add_polyline(boundary, close=True), corners=(0, 2, 4, 6)
    )
    assert geometry.face_point(face, 0.5, 0.5)[2] == pytest.approx(2.0)
    lower, upper = geometry.add_points(((0.5, 0.5, 1.5), (0.5, 0.5, 2.5)))
    crossing = geometry.add_line(lower, upper)

    face_bounds = geometry.entity_bounds_many((("face", face),))[0]
    assert face_bounds is not None and face_bounds[5] >= 2.0
    candidates = geometry.spatial_candidates(
        (0.49, 0.49, 1.49), (0.51, 0.51, 2.51), kinds=("face", "edge")
    )
    assert ("face", face) in candidates
    assert ("edge", crossing) in candidates
